"""tools/c8_u_imu.py (C8, IMU dynamic bound of U_IMU; design CLAUDE_THESIS
03_CODE/docs/2026-11-15_Thiet_ke_C8_U_IMU_v2.md, Liam 15/11 CODE_C13): synthetic ESP32 logs with known
answers (scale per direction, drift, late reports, missing reports, latency); the real logs of
CAR_LIFTED/2026-11-14_run01 and CAR_FLOOR/2026-11-10_run01 when they are on this machine."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import c8_u_imu as U  # noqa: E402

T0 = 0.05            # report period of the campaign firmware (20 Hz)
COLS = ['timestamp_us', 'log_seq', 'yaw_rad', 'rv_n', 'age_r_ms', 'gz', 'gyro_n', 'age_g_ms',
        'roll', 'pitch', 'run_active']


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_log(path, rate, t_end, s_plus=0.0, s_minus=0.0, drift=0.0, lat=0.004, late=(), miss=(),
             run=None, tilt=0.0, t_start=10.0):
    """ESP32 log at 100 Hz. rate(t) = true yaw rate [rad/s]; the sensor samples every T0 from
    t_start (sensor time = ESP32 time), the IMU yaw integrates the rate with scale (1 + s_plus) for
    positive steps and (1 + s_minus) for negative ones, plus drift [rad/s]; a report reaches the ESP32
    lat seconds after it is taken, plus extra for the indexes in late ({k: extra}); indexes in miss
    never arrive. run = (t_on, t_off) for run_active. Returns the true sample times and yaw."""
    late = dict(late)
    ts = t_start + T0 * np.arange(int(round((t_end - t_start) / T0)) + 1)
    fine = np.arange(t_start, t_end + 1e-9, 1e-3)
    r = np.array([rate(x) for x in fine])
    true = np.concatenate([[0.0], np.cumsum((r[1:] + r[:-1]) / 2 * 1e-3)])
    st = np.interp(ts, fine, true)
    d = np.diff(st)
    imu = np.concatenate([[0.0], np.cumsum(np.where(d > 0, d * (1 + s_plus), d * (1 + s_minus)))])
    imu = imu + drift * (ts - t_start)
    gzs = np.array([rate(x) for x in ts])
    recv = [(ts[k] + lat + late.get(k, 0.0), k) for k in range(len(ts)) if k not in miss]
    recv.sort()
    rows, seq, j, n = [], 1, 0, 0
    last = None
    for t in np.arange(t_start + 0.003, t_end, 0.01):
        while j < len(recv) and recv[j][0] <= t:
            last = recv[j]
            j += 1
            n += 1
        if last is None:
            continue
        rt, k = last
        on = '1' if run and run[0] <= t <= run[1] else '0'
        rows.append({'timestamp_us': str(int(round(t * 1e6))), 'log_seq': str(seq),
                     'yaw_rad': repr(float(wrap(imu[k]))), 'rv_n': str(n), 'age_r_ms': repr(float((t - rt) * 1e3)),
                     'gz': repr(float(gzs[k])), 'gyro_n': str(n), 'age_g_ms': repr(float((t - rt) * 1e3)),
                     'roll': repr(float(tilt * math.sin(t))), 'pitch': '0.0', 'run_active': on})
        seq += 1
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return ts, imu


def turns(n_turns, w, sign, t_still=3.0, t_start=10.0):
    """Rate profile: still, n_turns at |w| rad/s with sign, still. Returns (rate, t_end)."""
    t_rot = n_turns * 2 * math.pi / w
    a, b = t_start + t_still, t_start + t_still + t_rot
    return (lambda t: sign * w if a <= t < b else 0.0), b + t_still


# ── samples ────────────────────────────────────────────────────────────────────────────

def test_samples_on_rv_n_change_and_receive_time(tmp_path):
    p = str(tmp_path / 'a.csv')
    ts, _ = make_log(p, lambda t: 0.3, 14.0, lat=0.004)
    S = U.rv_samples(U.load_log(p), period_s=T0, tau_hat=0.0, j_lim=0.005)
    assert S['n_samples'] == len(ts) - 1 or S['n_samples'] == len(ts)       # the last report may arrive after the end
    # grid time = receive time minus the constant latency is absorbed by the fit (a + b n): within 1 us of ts + lat
    k = np.searchsorted(ts + 0.004, S['t'][0] - 1e-4)
    assert np.allclose(S['t'], (ts + 0.004)[k:k + len(S['t'])], atol=2e-6)
    assert S['n_late'] == 0


def test_late_report_is_dropped(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.3, 14.0, late={30: 0.020})
    S = U.rv_samples(U.load_log(p), period_s=T0, tau_hat=0.0, j_lim=0.005)
    assert S['n_late'] == 1
    assert np.all(np.abs(S['j'][S['keep']]) <= 0.005)


def test_scale_correction_and_unwrap(tmp_path):
    """+3 turns then -3 turns with different scale per direction: the corrected yaw follows the truth across
    the +-pi wrap; uncorrected it does not."""
    p = str(tmp_path / 'a.csv')
    t1 = 10 + 3 * 2 * math.pi / 1.5
    t2 = t1 + 3 * 2 * math.pi / 1.5
    rate = lambda t: 1.5 if t < t1 else (-1.5 if t < t2 else 0.0)      # noqa: E731
    make_log(p, rate, t2 + 1.0, s_plus=0.01, s_minus=-0.006)
    L = U.load_log(p)
    raw = U.rv_samples(L, period_s=T0, tau_hat=0.0)
    cor = U.rv_samples(L, period_s=T0, tau_hat=0.0, s_plus=0.01, s_minus=-0.006)
    turn = 3 * 2 * math.pi
    assert abs(cor['psi'][-1] - cor['psi'][0]) < 2e-3                   # back to the start heading
    assert abs((raw['psi'][-1] - raw['psi'][0]) - turn * 0.016) < 2e-3  # (1.01 - 0.994) of 3 turns left over
    assert raw['psi'].max() - raw['psi'][0] > 3 * math.pi                 # unwrapped across +-pi


def test_interp_one_missing_ok_two_missing_nan(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.5, 14.0, miss={40, 60, 61})
    S = U.rv_samples(U.load_log(p), period_s=T0, tau_hat=0.0)
    t1 = 10 + 40 * T0 + 0.004          # inside the one-report gap
    t2 = 10 + 60.5 * T0 + 0.004        # inside the two-report gap
    v = U.interp_yaw(S, np.array([t1, t2]), h_lim=0.105)
    assert np.isfinite(v[0]) and np.isnan(v[1])


def test_tau_hat_shifts_sample_times(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.2, 12.0)
    L = U.load_log(p)
    a = U.rv_samples(L, period_s=T0, tau_hat=0.0)
    b = U.rv_samples(L, period_s=T0, tau_hat=0.012)
    assert np.allclose(a['t'] - b['t'], 0.012)


def test_duplicates_dropped_with_segments(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.2, 12.0)
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    rows.insert(10, dict(rows[9]))
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with open(p + '.segments.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['row', 'log_seq', 'timestamp_us', 'segment', 'status'])
        for i, r in enumerate(rows, start=1):
            w.writerow([i, r['log_seq'], r['timestamp_us'], 1, 'duplicate' if i == 11 else 'ok'])
    L = U.load_log(p)
    assert L['n_dup'] == 1 and len(L['timestamp_us']) == len(rows) - 1


def test_forbidden_paths(tmp_path):
    for part in ('sealed', 'sealed_test', 'O1_FEAS'):
        d = tmp_path / part
        d.mkdir()
        p = str(d / 'a.csv')
        make_log(p, lambda t: 0.2, 11.0)
        with pytest.raises(U.InputError):
            U.load_log(p)


# ── scale (thang) ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('sign,chieu,s', [(1, 'ccw', 0.004), (-1, 'cw', -0.007)])
def test_thang_recovers_scale(tmp_path, sign, chieu, s):
    p = str(tmp_path / 'a.csv')
    rate, t_end = turns(5, 2.5, sign)
    make_log(p, rate, t_end, s_plus=s if sign > 0 else 0.0, s_minus=s if sign < 0 else 0.0)
    r = U.thang(p, n_vong=5, chieu=chieu, period_s=T0)
    assert abs(r['s'] - s) < 1e-4
    assert r['chieu'] == chieu


def test_thang_wrong_direction_refused(tmp_path):
    p = str(tmp_path / 'a.csv')
    rate, t_end = turns(5, 2.5, 1)
    make_log(p, rate, t_end)
    with pytest.raises(U.InputError, match='not the direction'):
        U.thang(p, n_vong=5, chieu='cw', period_s=T0)


def test_thang_needs_still_ends(tmp_path):
    p = str(tmp_path / 'a.csv')
    rate, t_end = turns(5, 2.5, 1, t_still=1.0)
    make_log(p, rate, t_end)
    with pytest.raises(U.InputError):
        U.thang(p, n_vong=5, chieu='ccw', period_s=T0)


def test_thang_gop():
    runs = [dict(s=0.004, chieu='ccw', cach='rc', n_vong=5), dict(s=0.006, chieu='ccw', cach='rc', n_vong=5),
            dict(s=0.001, chieu='ccw', cach='tay', n_vong=5), dict(s=-0.002, chieu='cw', cach='rc', n_vong=5),
            dict(s=-0.003, chieu='cw', cach='tay', n_vong=5)]
    g = U.thang_gop(runs, delta_gach_deg=2.0)
    base = math.radians(2.0) / (5 * 2 * math.pi)
    assert g['s_plus'] == pytest.approx(0.005)
    assert g['delta_plus'] == pytest.approx(base + 0.004)            # |0.001 - 0.005|
    assert g['s_minus'] == pytest.approx(-0.002)
    assert g['delta_minus'] == pytest.approx(base + 0.001)
    assert g['f4_mo'] is False
    g2 = U.thang_gop([r for r in runs if r['cach'] == 'tay'], delta_gach_deg=2.0)
    assert g2['f4_mo'] is True and g2['s_plus'] == pytest.approx(0.001)
    with pytest.raises(U.InputError):
        U.thang_gop([r for r in runs if r['chieu'] == 'ccw'], delta_gach_deg=2.0)


# ── drift, q (troi) ─────────────────────────────────────────────────────────────────────

def test_troi_drift_rate(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.0, 30.0, drift=math.radians(0.01))
    r = U.troi([p], period_s=T0, min_dt=2.0)
    assert r['beta_max'] == pytest.approx(math.radians(0.01), rel=0.05)


def test_troi_ignores_short_spans(tmp_path):
    """A 0.2 deg jump in the first second would give 0.2 deg/s over 1 s; with min_dt 2 s it is
    at most 0.1 deg/s, falling as the span grows."""
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: math.radians(0.2) if 10.2 <= t < 11.2 else 0.0, 30.0)
    r = U.troi([p], period_s=T0, min_dt=2.0)
    assert r['beta_max'] <= math.radians(0.2) / 2.0 + 1e-6


def test_troi_reads_imu_capture_raw(tmp_path):
    p = str(tmp_path / 'raw_static.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kind', 't_us', 'n', 'v0', 'v1', 'v2', 'qdrop'])
        for k in range(1200):
            t = 1_000_000 + k * 50_000
            w.writerow(['G', t, k + 1, 0.0, 0.0, 0.0, 0])
            w.writerow(['R', t + 600, k + 1, 0.1, -0.2, 0.004 + 0.01 * (k * 50e-3) / 60 + (0.007 if k % 7 == 0 else 0), 0])
    r = U.troi([p], period_s=T0, min_dt=2.0)
    assert r['q'] == pytest.approx(math.radians(0.007), abs=math.radians(0.0015))
    assert r['beta_max'] < math.radians(0.01)


# ── total (tong) ───────────────────────────────────────────────────────────────────────

def _c8_log(tmp_path, name='c8.csv'):
    p = str(tmp_path / name)
    # still 3 s, +300 deg at 2 rad/s, -300 deg at 2 rad/s, still 3 s; run_active over the motion
    a = 13.0
    t1 = a + math.radians(300) / 2.0
    t2 = t1 + math.radians(300) / 2.0
    make_log(p, lambda t: 2.0 if a <= t < t1 else (-2.0 if t1 <= t < t2 else 0.0), t2 + 3.0, run=(a, t2))
    return p, a, t2


def _globals():
    return dict(thang=dict(s_plus=0.0, s_minus=0.0, delta_plus=0.0015, delta_minus=0.002, f4_mo=False,
                           nguon=[{'file': 'thang.json', 'sha256': 'x'}]),
                troi=dict(beta_max=math.radians(0.016), q=math.radians(0.01), nguon=[{'file': 'troi.json', 'sha256': 'y'}]),
                tre=dict(tau_hat_s=0.012, dtau_fit_s=0.003, thu=True, nguon=[]))


def test_tong_terms(tmp_path):
    p, a, t2 = _c8_log(tmp_path)
    g = _globals()
    out = str(tmp_path / 'u.json')
    U.tong('O1-F08', [p], g['thang'], g['troi'], g['tre'], u_g=math.radians(1.0), le=0.25, out=out,
           period_s=T0, j_lim=0.005, h_lim=0.105, pre_s=2.0, tail_s=1.0)
    d = json.load(open(out, encoding='utf-8'))
    m = d['muc']['O1-F08']
    P, N = math.radians(300), math.radians(300)
    assert m['P_max'] == pytest.approx(P, rel=0.01) and m['N_max'] == pytest.approx(N, rel=0.01)
    assert m['phan']['thang'] == pytest.approx(0.0015 * P + 0.002 * N, rel=0.01)
    assert m['omega_max'] == pytest.approx(2.0)
    assert m['alpha_max'] == pytest.approx(4.0 / T0, rel=0.02)      # +2 -> -2 rad/s between two reports
    assert m['T_win'] == pytest.approx((t2 - a) + 2.0 + 1.0, abs=0.03)
    assert m['phan']['thoi_gian'] == pytest.approx((0.003 + 0.005) * 2.0)
    assert m['phan']['noi_suy'] == pytest.approx(m['alpha_max'] * 0.105 ** 2 / 8)
    s = sum(m['phan'].values())
    assert m['U_IMU_dong_rad'] == pytest.approx(math.radians(1.0) + 1.25 * s)
    assert d['thu'] is True          # tre given as a trial value: not lockable


def test_tong_adds_level_and_refuses_changes(tmp_path):
    p, _, _ = _c8_log(tmp_path)
    g = _globals()
    out = str(tmp_path / 'u.json')
    kw = dict(period_s=T0, j_lim=0.005, h_lim=0.105, pre_s=2.0, tail_s=1.0)
    U.tong('O1-F08', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, **kw)
    U.tong('O1-F05', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, **kw)
    assert set(json.load(open(out, encoding='utf-8'))['muc']) == {'O1-F08', 'O1-F05'}
    with pytest.raises(U.InputError):
        U.tong('O1-F05', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, **kw)
    with pytest.raises(U.InputError):
        U.tong('O1-F02', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.5, out=out, **kw)


def test_tong_needs_run_active(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.2, 14.0)
    g = _globals()
    with pytest.raises(U.InputError):
        U.tong('O1-F08', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               period_s=T0, j_lim=0.005, h_lim=0.105, pre_s=2.0, tail_s=1.0)


def test_main_exit_codes(tmp_path):
    p = str(tmp_path / 'a.csv')
    rate, t_end = turns(5, 2.5, 1)
    make_log(p, rate, t_end, s_plus=0.003)
    out = str(tmp_path / 't.json')
    assert U.main(['thang', '--log', p, '--n-vong', '5', '--chieu', 'ccw', '--cach', 'tay', '--out', out]) == 0
    assert json.load(open(out, encoding='utf-8'))['s'] == pytest.approx(0.003, abs=1e-4)
    assert U.main(['thang', '--log', p, '--n-vong', '5', '--chieu', 'cw', '--cach', 'tay',
                   '--out', str(tmp_path / 't2.json')]) == 2
    assert U.main(['thang', '--log', p, '--n-vong', '5', '--chieu', 'ccw', '--cach', 'tay', '--out', out]) == 2  # no overwrite


# ── camera - IMU (tre, kiem) ─────────────────────────────────────────────────────────────

FCOLS = ['idx', 't_s', 'valid', 'car_seen', 'psi', 'wz']


def true_yaw(rate, t_start, t_end):
    fine = np.arange(t_start, t_end + 1e-9, 1e-3)
    r = np.array([rate(x) for x in fine])
    return fine, np.concatenate([[0.0], np.cumsum((r[1:] + r[:-1]) / 2 * 1e-3)]), r


def write_frames(path, rate, t_start, t_end, offset, psi0=0.4, fps=60.0, err=None):
    """Camera frames: t_s = t_log - offset; psi = true yaw + psi0 (+ err(t_log)), wrapped; wz exact."""
    fine, tru, r = true_yaw(rate, t_start, t_end)
    tl = np.arange(t_start + 0.01, t_end - 0.01, 1 / fps)
    psi = np.interp(tl, fine, tru) + psi0 + (np.array([err(x) for x in tl]) if err else 0.0)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(FCOLS)
        for i, t in enumerate(tl):
            w.writerow([i, repr(float(t - offset)), 1, 1, repr(float(wrap(psi[i]))), repr(float(rate(t)))])


def _c8_pair(tmp_path, lat=0.004, s_plus=0.0, err=None, name='r'):
    a = 13.0
    t1 = a + math.radians(300) / 2.0
    t2 = t1 + math.radians(300) / 2.0
    rate = lambda t: 2.0 if a <= t < t1 else (-2.0 if t1 <= t < t2 else 0.0)      # noqa: E731
    lg, fr = str(tmp_path / f'{name}.csv'), str(tmp_path / f'{name}_frames.csv')
    make_log(lg, rate, t2 + 4.0, lat=lat, s_plus=s_plus, run=(a, t2))
    write_frames(fr, rate, 10.0, t2 + 4.0, offset=0.5, err=err)
    return fr, lg


def test_tre_finds_delay(tmp_path):
    fr, lg = _c8_pair(tmp_path, lat=0.004)
    r = U.tre_one(fr, lg, 0.5)
    # grid times are receive times (sample + 4 ms); the camera sees the true yaw: tau = 4 ms
    assert r['tau_s'] == pytest.approx(0.004, abs=0.001)
    fr2, lg2 = _c8_pair(tmp_path, lat=0.015, name='r2')
    g = U.tre([(fr, lg, 0.5), (fr2, lg2, 0.5)], dict(s_plus=0.0, s_minus=0.0))
    assert g['tau_hat_s'] == pytest.approx(0.0095, abs=0.001)
    assert g['dtau_fit_s'] == pytest.approx(0.0055 + 0.001, abs=0.001)


def test_tre_wrong_offset_refused(tmp_path):
    fr, lg = _c8_pair(tmp_path)
    with pytest.raises(U.InputError, match='edge'):
        U.tre_one(fr, lg, 0.6)


def _u_file(tmp_path, p, tau, delta=0.002, s_plus=0.0):
    out = str(tmp_path / 'u.json')
    th = dict(s_plus=s_plus, s_minus=0.0, delta_plus=delta, delta_minus=delta, f4_mo=False)
    tr = dict(beta_max=math.radians(0.016), q=math.radians(0.01))
    U.tong('O1-F08', [p], th, tr, dict(tau_hat_s=tau, dtau_fit_s=0.002), u_g=math.radians(0.5), le=0.25, out=out,
           pre_s=2.0, tail_s=1.0)
    return json.load(open(out, encoding='utf-8'))


def test_kiem_clean_run_ok(tmp_path):
    fr, lg = _c8_pair(tmp_path)
    u = _u_file(tmp_path, lg, tau=0.004)
    r = U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    assert r['ok'] and r['n_over'] == 0 and r['closure_ok'] is True
    assert r['n_compared'] > 0.95 * r['n_win']


def test_kiem_catches_camera_error(tmp_path):
    fr, lg = _c8_pair(tmp_path, err=lambda t: math.radians(20.0) if 15.0 <= t < 15.5 else 0.0)  # step profile: alpha 80 rad/s2, limit about 10 deg
    u = _u_file(tmp_path, lg, tau=0.004)
    r = U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    assert r['n_over'] > 0 and not r['ok']


def test_kiem_closure_catches_uncorrected_scale(tmp_path):
    """IMU 2 % fast counter-clockwise, the U_IMU file says scale exact with a small delta: the closure
    after +300 / -300 deg is 6 deg, beyond its bound."""
    fr, lg = _c8_pair(tmp_path, s_plus=0.02)
    u = _u_file(tmp_path, lg, tau=0.004, delta=0.001)
    r = U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    assert r['closure_ok'] is False and not r['ok']


def test_main_tre_kiem(tmp_path):
    fr, lg = _c8_pair(tmp_path)
    th = str(tmp_path / 'thang.json')
    json.dump(dict(s_plus=0.0, s_minus=0.0, delta_plus=0.002, delta_minus=0.002, f4_mo=False), open(th, 'w'))
    tr = str(tmp_path / 'troi.json')
    json.dump(dict(beta_max=0.0003, q=0.0002), open(tr, 'w'))
    te = str(tmp_path / 'tre.json')
    assert U.main(['tre', '--cap', f'{fr},{lg},0.5', '--thang', th, '--out', te]) == 0
    uf = str(tmp_path / 'u.json')
    assert U.main(['tong', '--muc', 'O1-F08', '--c8-log', lg, '--thang', th, '--troi', tr, '--tre', te,
                   '--u-g-deg', '0.5', '--out', uf]) == 0
    assert json.load(open(uf, encoding='utf-8'))['thu'] is False
    assert U.main(['kiem', '--frames', fr, '--log', lg, '--offset-s', '0.5', '--u-imu', uf, '--muc', 'O1-F08',
                   '--u-psi-deg', '0.5', '--out', str(tmp_path / 'k.json')]) == 0


# ── real logs (skipped when not on this machine) ───────────────────────────────────────────────

DATA = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'CLAUDE_THESIS', '05_DATA', 'results'))
LIFTED = os.path.join(DATA, 'CAR_LIFTED', '2026-11-14_run01', 'pi', 'pid_k1a.csv')
FLOOR = os.path.join(DATA, 'CAR_FLOOR', '2026-11-10_run01', 'pi', 'c6_f08.csv')


@pytest.mark.skipif(not os.path.isfile(LIFTED), reason='real lifted log not here')
def test_real_lifted_drift_small():
    r = U.troi([LIFTED], period_s=T0, min_dt=2.0)
    assert r['beta_max'] < math.radians(0.05)


@pytest.mark.skipif(not os.path.isfile(FLOOR), reason='real floor log not here')
def test_real_floor_samples():
    S = U.rv_samples(U.load_log(FLOOR), period_s=T0, tau_hat=0.0, j_lim=0.005)
    assert S['n_samples'] > 150
    assert S['n_late'] <= 0.05 * S['n_samples']
