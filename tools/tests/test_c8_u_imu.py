"""tools/c8_u_imu.py (C8, IMU dynamic bound of U_IMU; design CLAUDE_THESIS
03_CODE/docs/2026-11-15_Thiet_ke_C8_U_IMU_v2.md, Liam 15/11 CODE_C13): synthetic ESP32 logs with known
answers (scale per direction, drift, late reports, missing reports, latency, resets); tests named cu<N> are the
counter-examples of Codex Sol round 1 (Liam 15/11 "Sua ca 23 + vong 2"); the real logs of
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
COLS = ['timestamp_us', 'log_seq', 'yaw_rad', 'rv_n', 'age_r_ms', 'gz', 'gyro_n', 'age_g_ms', 'reset_n',
        'roll', 'pitch', 'run_active', 'r_valid', 'g_valid']


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_log(path, rate, t_end, s_plus=0.0, s_minus=0.0, drift=0.0, lat=0.004, late=(), miss=(),
             run=None, tilt=0.0, t_start=10.0, reset_at=None, no_gyro=False, gyro_miss=None, runs=None,
             reset_jump=0.0, valid_from=None):
    """ESP32 log at 100 Hz. rate(t) = true yaw rate [rad/s]; the sensor samples every T0 from t_start (sensor
    time = ESP32 time); the IMU yaw integrates the rate with scale (1 + s_plus) for positive steps and
    (1 + s_minus) for negative ones, plus drift [rad/s]; a report reaches the ESP32 lat seconds after it is
    taken, plus extra for the indexes in late ({k: extra}); indexes in miss never arrive (gyro reports too,
    unless gyro_miss gives another set). run = (t_on, t_off) or runs = [(...), ...] for run_active. reset_at:
    time of an IMU reset (reset_n + 1, the yaw restarts from 0). Returns the true sample times and IMU yaw."""
    late = dict(late)
    gyro_miss = set(miss) if gyro_miss is None else set(gyro_miss)
    runs = runs or ([run] if run else [])
    ts = t_start + T0 * np.arange(int(round((t_end - t_start) / T0)) + 1)
    fine = np.arange(t_start, t_end + 1e-9, 1e-3)
    r = np.array([rate(x) for x in fine])
    true = np.concatenate([[0.0], np.cumsum((r[1:] + r[:-1]) / 2 * 1e-3)])
    st = np.interp(ts, fine, true)
    d = np.diff(st)
    imu = np.concatenate([[0.0], np.cumsum(np.where(d > 0, d * (1 + s_plus), d * (1 + s_minus)))])
    imu = imu + drift * (ts - t_start)
    if reset_at is not None:
        k0 = int(np.searchsorted(ts, reset_at))
        imu[k0:] = imu[k0:] - imu[k0] + reset_jump
    gzs = np.array([rate(x) for x in ts])
    rv = sorted((ts[k] + lat + late.get(k, 0.0), k) for k in range(len(ts)) if k not in miss)
    gy = sorted((ts[k] + lat + late.get(k, 0.0), k) for k in range(len(ts)) if k not in gyro_miss)
    rows, seq = [], 1
    jr = jg = nr = ng = 0
    lr = lg = None
    for t in np.arange(t_start + 0.003, t_end, 0.01):
        while jr < len(rv) and rv[jr][0] <= t:
            lr = rv[jr]
            jr += 1
            nr += 1
        while jg < len(gy) and gy[jg][0] <= t:
            lg = gy[jg]
            jg += 1
            ng += 1
        if lr is None or lg is None:
            continue
        on = '1' if any(a <= t <= b for a, b in runs) else '0'
        rows.append({'timestamp_us': str(int(round(t * 1e6))), 'log_seq': str(seq),
                     'yaw_rad': repr(float(wrap(imu[lr[1]]))), 'rv_n': str(nr),
                     'age_r_ms': repr(float((t - lr[0]) * 1e3)),
                     'gz': repr(float(gzs[lg[1]])), 'gyro_n': '0' if no_gyro else str(ng),
                     'age_g_ms': '-1.0' if no_gyro else repr(float((t - lg[0]) * 1e3)),
                     'reset_n': '1' if reset_at is not None and t >= reset_at else '0',
                     'roll': repr(float(tilt * math.sin(t))), 'pitch': '0.0', 'run_active': on,
                     'r_valid': '0' if valid_from is not None and t < valid_from else '1',
                     'g_valid': '0' if valid_from is not None and t < valid_from else '1'})
        seq += 1
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return ts, imu


def turns(n_turns, w, sign, t_still=3.5, t_start=10.0):
    """Rate profile: still, n_turns at |w| rad/s with sign, still. Returns (rate, t_end)."""
    t_rot = n_turns * 2 * math.pi / w
    a, b = t_start + t_still, t_start + t_still + t_rot
    return (lambda t: sign * w if a <= t < b else 0.0), b + t_still


def _c8_log(tmp_path, name='c8.csv', **kw):
    """Still 3 s, +300 deg at 2 rad/s, -300 deg at 2 rad/s, still; run_active 16 s (>= T_cmd of F08, F05, K2)."""
    p = str(tmp_path / name)
    a = 13.0
    t1 = a + math.radians(300) / 2.0
    t2 = t1 + math.radians(300) / 2.0
    make_log(p, lambda t: 2.0 if a <= t < t1 else (-2.0 if t1 <= t < t2 else 0.0), a + 21.0, run=(a, a + 16.0), **kw)
    return p, a, a + 16.0


def _globals():
    return dict(thang=dict(s_plus=0.0, s_minus=0.0, delta_plus=0.0015, delta_minus=0.002, f4_mo=False,
                           nguon=[{'file': 'thang.json', 'sha256': 'x'}]),
                troi=dict(beta_max=math.radians(0.016), q=math.radians(0.01), nguon=[{'file': 'troi.json', 'sha256': 'y'}]),
                tre=dict(tau_hat_s=0.012, dtau_fit_s=0.003, thu=True, nguon=[]))


# ── samples ────────────────────────────────────────────────────────────────────────────

def test_samples_on_rv_n_change_and_receive_time(tmp_path):
    p = str(tmp_path / 'a.csv')
    ts, _ = make_log(p, lambda t: 0.3, 14.0, lat=0.004)
    S = U.rv_samples(U.load_log(p), period_s=T0, tau_hat=0.0, j_lim=0.005)
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
    p = str(tmp_path / 'a.csv')
    t1 = 10 + 3 * 2 * math.pi / 1.5
    t2 = t1 + 3 * 2 * math.pi / 1.5
    rate = lambda t: 1.5 if t < t1 else (-1.5 if t < t2 else 0.0)      # noqa: E731
    make_log(p, rate, t2 + 1.0, s_plus=0.01, s_minus=-0.006)
    L = U.load_log(p)
    raw = U.rv_samples(L, period_s=T0, tau_hat=0.0)
    cor = U.rv_samples(L, period_s=T0, tau_hat=0.0, s_plus=0.01, s_minus=-0.006)
    turn = 3 * 2 * math.pi
    assert abs(cor['psi'][-1] - cor['psi'][0]) < 2e-3
    assert abs((raw['psi'][-1] - raw['psi'][0]) - turn * 0.016) < 2e-3
    assert raw['psi'].max() - raw['psi'][0] > 3 * math.pi


def test_interp_one_missing_ok_two_missing_nan(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.5, 14.0, miss={40, 60, 61}, gyro_miss=set())
    S = U.rv_samples(U.load_log(p), period_s=T0, tau_hat=0.0)
    t1 = 10 + 40 * T0 + 0.004
    t2 = 10 + 60.5 * T0 + 0.004
    t3 = 10 + 70 * T0 + 0.004          # after the two-report gap: unwrapped by the gyro, usable again
    v = U.interp_yaw(S, np.array([t1, t2, t3]), h_lim=0.105)
    assert np.isfinite(v[0]) and np.isnan(v[1]) and np.isfinite(v[2])


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


# ── Codex round 1 counter-examples ─────────────────────────────────────────────────────────

def test_cu1_wrong_period_grid_refused(tmp_path):
    """Delays repeating [40, 20, 0] ms on a 50 ms grid look like an exact 30 ms grid."""
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.3, 14.0, lat=0.0, late={k: (0.040, 0.020, 0.0)[k % 3] for k in range(200)})
    with pytest.raises(U.InputError, match='CU-1'):
        U.rv_samples(U.load_log(p), period_s=T0)


def test_cu2_turn_totals_keep_late_reports(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 2.0 if 11.0 <= t < 11.5 else (-2.0 if 11.5 <= t < 12.0 else 0.0), 14.0,
             late={30: 0.020})
    S = U.rv_samples(U.load_log(p), period_s=T0)
    assert S['n_late'] == 1
    P, N = U.turn_totals(S, 10.5, 13.5)
    assert P == pytest.approx(1.0, abs=0.005) and N == pytest.approx(1.0, abs=0.005)      # kept reports only: 0.9


def test_cu3_long_gap_without_gyro_breaks_the_yaw(tmp_path):
    """One full turn inside a 0.7 s gap with no gyro: the yaw after the gap must not be used."""
    p = str(tmp_path / 'a.csv')
    w = 2 * math.pi / 0.6
    make_log(p, lambda t: w if 12.0 <= t < 12.6 else 0.0, 14.0, s_plus=0.01, miss=set(range(38, 52)))
    S = U.rv_samples(U.load_log(p), period_s=T0, s_plus=0.01)
    assert math.isfinite(S['t_break'])
    assert np.isnan(U.interp_yaw(S, np.array([13.5]))[0])


def test_cu3_long_gap_with_gyro_unwraps_right(tmp_path):
    p = str(tmp_path / 'a.csv')
    w = 2 * math.pi / 0.6
    make_log(p, lambda t: w if 12.0 <= t < 12.6 else 0.0, 14.0, s_plus=0.01, miss=set(range(38, 52)),
             gyro_miss=set())
    S = U.rv_samples(U.load_log(p), period_s=T0, s_plus=0.01)
    assert not math.isfinite(S['t_break'])
    v = U.interp_yaw(S, np.array([11.0, 13.5]))
    assert v[1] - v[0] == pytest.approx(2 * math.pi, abs=5e-3)


def test_cu4_reset_inside_window_refused(tmp_path):
    p, _, _ = _c8_log(tmp_path, reset_at=20.0)
    g = _globals()
    with pytest.raises(U.InputError, match='CU-4'):
        U.tong('O1-F08', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=1.0)


def test_cu5_alpha_uses_true_spacing():
    assert U.alpha_max(np.array([0.0, 0.01]), np.array([0.0, 1.0])) == pytest.approx(100.0)


def test_cu6_no_gyro_refused(tmp_path):
    p, _, _ = _c8_log(tmp_path, no_gyro=True)
    g = _globals()
    with pytest.raises(U.InputError, match='CU-6'):
        U.tong('O1-F08', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=1.0)


def test_cu7_negative_age_first_row():
    n = np.array([1, 2, 2, 3], dtype=float)
    age = np.array([0.0, -1.0, 9.0, 0.0])
    assert list(U.report_rows(n, age)) == [0, 2, 3]
    with pytest.raises(U.InputError):
        U.report_rows(np.array([1, 2, 3.0]), np.array([0.0, -1.0, 0.0]))


def test_cu8_cut_or_split_run_refused(tmp_path):
    g = _globals()
    p = str(tmp_path / 'cut.csv')
    make_log(p, lambda t: 2.0 if 13 <= t < 15 else 0.0, 22.0, run=(13.0, 18.0))      # 5 s < T_cmd of F08
    with pytest.raises(U.InputError, match='CU-8'):
        U.tong('O1-F08', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=1.0)
    q = str(tmp_path / 'split.csv')
    make_log(q, lambda t: 0.0, 40.0, runs=[(12.0, 24.0), (26.0, 38.0)])
    with pytest.raises(U.InputError, match='CU-8'):
        U.tong('O1-F08', [q], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=1.0)


def test_cu9_troi_per_run_active_part(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.001 if t >= 100.0 else 0.0, 104.0, runs=[(11.0, 14.0), (100.0, 103.5)])
    r = U.troi([p], period_s=T0, min_dt=2.0, doan='run_active')
    assert r['beta_max'] == pytest.approx(0.001, rel=0.05)


def test_cu10_troi_scale_corrected_and_tong_checks_it(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: -0.001, 30.0, s_minus=-0.01)
    raw = U.troi([p], period_s=T0, min_dt=2.0)
    cor = U.troi([p], period_s=T0, min_dt=2.0, s_minus=-0.01)
    assert raw['beta_max'] == pytest.approx(0.00099, rel=0.01)
    assert cor['beta_max'] == pytest.approx(0.001, rel=0.01)
    c8, _, _ = _c8_log(tmp_path)
    g = _globals()
    g['thang']['s_minus'] = -0.01
    with pytest.raises(U.InputError, match='CU-10'):
        U.tong('O1-F08', [c8], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=1.0)


def test_cu11_q_with_two_report_blocks(tmp_path):
    p = str(tmp_path / 'raw.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kind', 't_us', 'n', 'v0', 'v1', 'v2', 'qdrop'])
        for k in range(20):
            w.writerow(['R', 1_000_000 + k * 500_000, k + 1, 0, 0, 0.002 if k == 1 else 0.0, 0])
    r = U.troi([p], min_dt=2.0)
    assert r['q'] == pytest.approx(math.radians(0.001), rel=1e-6)


def test_cu12_nan_never_written(tmp_path):
    with pytest.raises(ValueError):
        U._write_new(str(tmp_path / 'x.json'), {'tau_hat_s': float('nan')})


def test_cu13_closure_near_pi_fails(tmp_path):
    fr, lg = _c8_pair(tmp_path, err=lambda t: (math.pi + (0.001 if int(t * 60) % 2 else -0.001)) if t > 29.5 else 0.0)
    u = _u_file(tmp_path, lg, tau=0.004)
    r = U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    assert r['closure_ok'] is False and abs(r['closure_rad']) > 3.0


def test_cu14_closure_window_fixed(tmp_path):
    fr, lg = _c8_pair(tmp_path, t_tail=60.0)
    u = _u_file(tmp_path, lg, tau=0.004)
    r = U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    assert r['closure_T_s'] <= 2.0 + 16.0 + 1.0 + U.END_STILL_S + 0.05


def test_cu15_incomplete_scale_set_refused():
    def run(chieu, cach, toc, s):
        return dict(s=s, chieu=chieu, cach=cach, toc_do=toc, n_vong=5)
    full = [run(c, a, t, 0.001) for c in ('ccw', 'cw') for a, t in sorted(U.SCALE_RUNS)]
    assert U.thang_gop(full)['f4_mo'] is False
    hand = [r for r in full if r['cach'] == 'tay']
    assert U.thang_gop(hand)['f4_mo'] is True
    one_rc = [r for r in full if not (r['cach'] == 'rc' and r['toc_do'] == 'nhanh')]
    with pytest.raises(U.InputError, match='CU-15'):
        U.thang_gop(one_rc)


def test_cu16_still_under_three_seconds_refused(tmp_path):
    p = str(tmp_path / 'a.csv')
    rate, t_end = turns(5, 2.5, 1, t_still=2.5)
    make_log(p, rate, t_end)
    with pytest.raises(U.InputError, match='still part'):
        U.thang(p, n_vong=5, chieu='ccw', period_s=T0)


def test_cu17_bad_numbers_and_json(tmp_path):
    c8, _, _ = _c8_log(tmp_path)
    g = _globals()
    with pytest.raises(U.InputError):
        U.tong('O1-F08', [c8], g['thang'], g['troi'], g['tre'], u_g=0.01, le=-1.0, out=str(tmp_path / 'u.json'))
    bad = dict(g['thang'], delta_plus=float('nan'))
    with pytest.raises(U.InputError):
        U.tong('O1-F08', [c8], bad, g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'))
    j = str(tmp_path / 'list.json')
    json.dump([], open(j, 'w'))
    tr = str(tmp_path / 'troi.json')
    json.dump(dict(beta_max=0.0003, q=0.0002), open(tr, 'w'))
    assert U.main(['tong', '--muc', 'O1-F08', '--c8-log', c8, '--thang', j, '--troi', tr, '--tre-thu', '0.01,0.002',
                   '--u-g-deg', '0.5', '--out', str(tmp_path / 'u2.json')]) == 2


def test_cu18_forbidden_output_refused(tmp_path):
    d = tmp_path / 'sealed'
    d.mkdir()
    p = str(tmp_path / 'a.csv')
    rate, t_end = turns(5, 2.5, 1)
    make_log(p, rate, t_end)
    assert U.main(['thang', '--log', p, '--n-vong', '5', '--chieu', 'ccw', '--cach', 'tay', '--toc-do', 'cham',
                   '--out', str(d / 't.json')]) == 2
    assert not os.listdir(d)


def test_cu19_existing_tmp_untouched(tmp_path):
    c8, _, _ = _c8_log(tmp_path)
    g = _globals()
    out = str(tmp_path / 'u.json')
    with open(out + '.tmp', 'w') as f:
        f.write('keep')
    U.tong('O1-F08', [c8], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, tail_s=1.0)
    assert open(out + '.tmp').read() == 'keep'


def test_cu20_e_thang_per_run(tmp_path):
    a = 13.0
    p1, p2 = str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv')
    for p, (x, y) in ((p1, (300, 200)), (p2, (200, 300))):
        t1 = a + math.radians(x) / 2.0
        t2 = t1 + math.radians(y) / 2.0
        make_log(p, lambda t, t1=t1, t2=t2: 2.0 if a <= t < t1 else (-2.0 if t1 <= t < t2 else 0.0), a + 21.0,
                 run=(a, a + 16.0))
    g = _globals()
    g['thang'].update(delta_plus=0.001, delta_minus=0.001)
    out = str(tmp_path / 'u.json')
    d = U.tong('O1-F08', [p1, p2], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, tail_s=1.0)
    assert d['muc']['O1-F08']['E_thang'] == pytest.approx(0.001 * math.radians(500), rel=0.01)


def test_cu21_eps_and_times(tmp_path):
    c8, _, _ = _c8_log(tmp_path, tilt=2.0)
    g = _globals()
    d = U.tong('O1-F08', [c8], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=1.0)
    m = d['muc']['O1-F08']
    assert m['eps_max_deg'] > 1.0 and m['luc_chay'] and 'n_late' in m['logs'][0]


# ── Codex round 2 (Liam 15/11 "Sua + vong 3") ────────────────────────────────────────────────

def test_cu2_1_turn_totals_on_sample_time():
    """Step at sample time 0.095 s inside [0, 0.10]; receive times end the window one report early."""
    S = dict(t_all=np.array([-.055, -.005, .045, .095, .145]), psi_all=np.array([0, 0, 0, .1, .1]),
             n_all=np.arange(5), t_recv=np.array([0, .05, .10, .15, .20]))
    P, N = U.turn_totals(S, 0.0, 0.10)
    assert P == pytest.approx(0.1) and N == 0.0


def test_cu2_2_no_yaw_from_the_report_before_an_unrecoverable_gap(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.3, 13.0, miss=set(range(20, 34)), late={34: 0.004})
    S = U.rv_samples(U.load_log(p), period_s=T0)
    t_after = S['t'][S['t'] > 10.0 + 33 * T0][0]           # first kept report after the gap
    t_before = S['t'][S['t'] < 10.0 + 20 * T0][-1]          # last kept report before it
    v = U.interp_yaw(S, np.array([t_before - 0.03, t_before, t_before + 0.01, t_after, t_after + 0.002]))
    assert np.all(np.isfinite(v[:2])) and np.all(np.isnan(v[2:]))       # usable up to the report before the gap


def test_cu2_3_valid_jitter_accepted(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.3, 15.0, lat=0.0, late={k: (0.0, 0.004, 0.008)[k % 3] for k in range(200)})
    S = U.rv_samples(U.load_log(p), period_s=T0, j_lim=0.005)
    assert S['n_late'] == 0 and S['period_fit_s'] == pytest.approx(T0, rel=1e-3)


def test_cu2_missing_report_at_reversal_bounded(tmp_path):
    """The report at the turn-around never arrives (gyro does): P, N must still cover the true 1 rad each."""
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 2.0 if 11.0 <= t < 11.5 else (-2.0 if 11.5 <= t < 12.0 else 0.0), 14.0,
             miss={30}, gyro_miss=set())
    S = U.rv_samples(U.load_log(p), period_s=T0)
    P, N = U.turn_totals(S, 10.5, 13.5)
    assert 1.0 <= P <= 1.15 and 1.0 <= N <= 1.15


def test_cu4_api_no_yaw_after_reset(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.3, 14.0, reset_at=12.0)
    S = U.rv_samples(U.load_log(p), period_s=T0)
    v = U.interp_yaw(S, np.array([11.0, 12.5, 13.5]))
    assert np.isfinite(v[0]) and np.isnan(v[1]) and np.isnan(v[2])


def test_cu8_log_ending_while_active_refused(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.0, 24.0, run=(12.0, 30.0))
    g = _globals()
    with pytest.raises(U.InputError, match='CU-8'):
        U.tong('O1-K1', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=0.5)
    c8, _, _ = _c8_log(tmp_path, name='c.csv')
    with pytest.raises(U.InputError):
        U.tong('O1-F08', [c8], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=0.0)


def test_cu16_still_29_s_refused(tmp_path):
    p = str(tmp_path / 'a.csv')
    rate, t_end = turns(5, 2.5, 1, t_still=2.9)
    make_log(p, rate, t_end)
    with pytest.raises(U.InputError, match='still part'):
        U.thang(p, n_vong=5, chieu='ccw', period_s=T0)


def test_cu17_kiem_checks_u_file(tmp_path):
    fr, lg = _c8_pair(tmp_path)
    u = _u_file(tmp_path, lg, tau=0.004)
    u['q'] = -1.0
    with pytest.raises(U.InputError):
        U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    u2 = dict(u, q=0.0002, version='v0')
    with pytest.raises(U.InputError):
        U.kiem(fr, lg, 0.5, u2, 'O1-F08', u_psi=math.radians(0.5))


def test_cu21_n_late_in_troi_and_kiem(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.0, 30.0)
    r = U.troi([p], period_s=T0, min_dt=2.0)
    assert r['files'][0]['n_late'] == 0
    fr, lg = _c8_pair(tmp_path)
    u = _u_file(tmp_path, lg, tau=0.004)
    assert U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))['n_late'] == 0


def test_cu12_tre_common_frame_set(tmp_path):
    """Two missing reports in the run: frames near the gap are compared at some delays only; the fit stays on one
    frame set and finds the delay."""
    a = 13.0
    t1 = a + math.radians(300) / 2.0
    t2 = t1 + math.radians(300) / 2.0
    rate = lambda t: 2.0 if a <= t < t1 else (-2.0 if t1 <= t < t2 else 0.0)      # noqa: E731
    lg, fr = str(tmp_path / 'g.csv'), str(tmp_path / 'g_frames.csv')
    make_log(lg, rate, a + 23.0, lat=0.004, run=(a, a + 16.0), miss={90, 91}, gyro_miss=set())
    write_frames(fr, rate, 10.0, a + 23.0, offset=0.5)
    r = U.tre_one(fr, lg, 0.5)
    assert r['tau_s'] == pytest.approx(0.004, abs=0.001) and math.isfinite(r['rms_rad'])


# ── Codex round 3 (Liam 15/11 "sua + soan goi vong 4") ────────────────────────────────────────────

def test_cu2_r3_gap_bound_on_both_time_bases():
    """tau 55 ms: the reversal happens in the sample-time gap [0, 0.1], before the receive-time gap."""
    tg = np.arange(-0.05, 0.2001, 0.005)
    gz = np.where((tg >= 0) & (tg < 0.025), 4.0, np.where((tg >= 0.025) & (tg < 0.05), -4.0, 0.0))
    S = dict(t_all=np.array([0.0, 0.1]), t_recv=np.array([0.055, 0.155]), n_all=np.array([0, 2]),
             psi_all=np.array([0.0, 0.0]), tg=tg, gz=gz, period_fit_s=0.05, h_lim=0.105)
    P, N = U.turn_totals(S, 0.0, 0.1)
    assert P >= 0.1 and N >= 0.1


def test_cu4_r3_no_interpolation_to_a_report_after_reset(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.0, 13.0, late={19: 0.015}, reset_at=11.0, reset_jump=1.0)
    S = U.rv_samples(U.load_log(p), period_s=T0)
    q = np.arange(10.5, 11.2, 0.005)
    v = U.interp_yaw(S, q)
    assert np.all(np.isnan(v) | (np.abs(v) < 1e-9))                 # never a value pulled toward the reset jump
    assert np.isfinite(v[0])


def test_cu4_r3_settle_flag(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.3, 14.0, valid_from=11.0)
    S = U.rv_samples(U.load_log(p), period_s=T0)
    v = U.interp_yaw(S, np.array([10.5, 12.0]))
    assert np.isnan(v[0]) and np.isfinite(v[1]) and S['n_invalid'] > 0
    c8, a, _ = _c8_log(tmp_path, name='c.csv', valid_from=12.5)        # flag still 0 inside [t_on - 2 s, ...]
    g = _globals()
    with pytest.raises(U.InputError, match='CU-4'):
        U.tong('O1-F08', [c8], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=str(tmp_path / 'u.json'),
               tail_s=1.0)


def test_cu3_1_long_gap_with_off_nominal_period():
    I = np.r_[np.arange(0, 100), np.arange(200, 300)]
    r = 10.0 + 0.0508 * I
    n, t_fit, b, j, inl = U._grid(r, 0.05, 0.005)
    # CU6: a gap above SPLIT_S carries no index, only 'reports missing' (step >= 2); every report on its grid
    assert inl.all() and n[100] - n[99] >= 2 and np.all(np.abs(t_fit - r) < 1e-6) and b == pytest.approx(0.0508)


def test_cu12_r3_common_set_smaller_than_some_set(tmp_path):
    a = 13.0
    t1 = a + math.radians(300) / 2.0
    t2 = t1 + math.radians(300) / 2.0
    rate = lambda t: 2.0 if a <= t < t1 else (-2.0 if t1 <= t < t2 else 0.0)      # noqa: E731
    lg, fr = str(tmp_path / 'g.csv'), str(tmp_path / 'g_frames.csv')
    make_log(lg, rate, a + 23.0, lat=0.004, run=(a, a + 16.0), miss={90, 91}, gyro_miss=set())
    write_frames(fr, rate, 10.0, a + 23.0, offset=0.5)
    r = U.tre_one(fr, lg, 0.5)
    assert r['n_frames'] < r['n_frames_some'] and r['n_used'] == [r['n_frames']]   # one set at every delay


# ── Codex round 4 ──────────────────────────────────────────────────────────────────────

def test_cu4_1_jitter_and_long_gap():
    I = np.r_[np.arange(0, 100), np.arange(200, 300)]
    r = 10.0 + 0.050 * I + 0.0008 * (I % 3)
    n, t_fit, b, j, inl = U._grid(r, 0.05, 0.005)
    assert inl.all() and n[100] - n[99] >= 2 and np.all(np.abs(t_fit - r) < 0.005)     # CU6: no index across 5 s


def test_cu4_2_gyro_gap_outside_the_window_ignored():
    S = dict(t_all=np.array([0.0, 0.1]), t_recv=np.array([0.0, 0.1]), n_all=np.array([0, 2]),
             psi_all=np.array([0.0, 0.0]), tg=np.array([-0.2, 0.0, 0.05, 0.1, 0.15]), gz=np.zeros(5),
             period_fit_s=0.05, h_lim=0.105)
    assert U.turn_totals(S, 0.0, 0.1) == (0.0, 0.0)


def test_cu5_1_short_runs_off_nominal_period():
    I = np.r_[np.arange(0, 10), np.arange(100, 110), np.arange(200, 210)]
    r = 10.0 + 0.0508 * I
    n, t_fit, b, j, inl = U._grid(r, 0.05, 0.005)
    assert inl.all() and np.all(np.diff(n)[[9, 19]] >= 2) and np.all(np.abs(t_fit - r) < 1e-6)     # CU6


# ── Codex round 6 (Liam 15/11 "Tach doan + vong 7") ───────────────────────────────────────────

@pytest.mark.parametrize('starts,L', [([0, 10000, 30000, 45000], 24), ([0, 17000, 36000, 65000], 46),
                                      ([0, 6005, 29019, 43408], 31)])
def test_cu6_long_gaps_split_into_segments(starts, L):
    I = np.concatenate([np.arange(s, s + L) for s in starts])
    r = 10.0 + 0.050805 * I
    n, t_fit, b, j, inl = U._grid(r, 0.05, 0.005)
    assert inl.all() and np.all(np.abs(t_fit - r) < 1e-6) and np.all(np.diff(n) >= 1)
    assert b == pytest.approx(0.050805, rel=1e-6)


def test_cu6_short_segment_kept_off_grid():
    I = np.r_[np.arange(0, 60), np.arange(100, 103), np.arange(200, 260)]
    r = 10.0 + 0.05 * I
    n, t_fit, b, j, inl = U._grid(r, 0.05, 0.005)
    assert inl.sum() == 120 and not inl[60:63].any() and np.all(np.isnan(j[60:63]))
    assert np.all(np.diff(n) >= 1) and n[60] - n[59] >= 2 and n[63] - n[62] >= 2


def test_cu6_split_gap_is_a_missing_report_for_turn_totals(tmp_path):
    """A 2 s IMU gap (gyro present): the grid is split there and the gap still counts as missing reports."""
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 1.0 if 12.0 <= t < 13.0 else 0.0, 16.0, miss=set(range(30, 70)), gyro_miss=set())
    S = U.rv_samples(U.load_log(p), period_s=T0)
    P, N = U.turn_totals(S, 11.0, 15.0)
    assert P >= 1.0 and np.all(np.diff(S['n_all']) >= 1)


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
    def run(chieu, cach, toc, s):
        return dict(s=s, chieu=chieu, cach=cach, toc_do=toc, n_vong=5)
    runs = [run('ccw', 'rc', 'cham', 0.004), run('ccw', 'rc', 'nhanh', 0.006), run('ccw', 'tay', 'cham', 0.001),
            run('ccw', 'tay', 'nhanh', 0.005), run('cw', 'rc', 'cham', -0.002), run('cw', 'rc', 'nhanh', -0.002),
            run('cw', 'tay', 'cham', -0.003), run('cw', 'tay', 'nhanh', -0.002)]
    g = U.thang_gop(runs, delta_gach_deg=2.0)
    base = math.radians(2.0) / (5 * 2 * math.pi)
    assert g['s_plus'] == pytest.approx(0.005)
    assert g['delta_plus'] == pytest.approx(base + 0.004)
    assert g['s_minus'] == pytest.approx(-0.002)
    assert g['delta_minus'] == pytest.approx(base + 0.001)
    assert g['f4_mo'] is False
    g2 = U.thang_gop([r for r in runs if r['cach'] == 'tay'], delta_gach_deg=2.0)
    assert g2['f4_mo'] is True and g2['s_plus'] == pytest.approx(0.003)
    with pytest.raises(U.InputError):
        U.thang_gop([r for r in runs if r['chieu'] == 'ccw'], delta_gach_deg=2.0)


# ── drift, q (troi) ─────────────────────────────────────────────────────────────────────

def test_troi_drift_rate(tmp_path):
    p = str(tmp_path / 'a.csv')
    make_log(p, lambda t: 0.0, 30.0, drift=math.radians(0.01))
    r = U.troi([p], period_s=T0, min_dt=2.0)
    assert r['beta_max'] == pytest.approx(math.radians(0.01), rel=0.05)


def test_troi_ignores_short_spans(tmp_path):
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

def test_tong_terms(tmp_path):
    p, a, t_off = _c8_log(tmp_path)
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
    assert m['alpha_max'] == pytest.approx(4.0 / T0, rel=0.02)
    assert m['T_win'] == pytest.approx((t_off - a) + 2.0 + 1.0, abs=0.03)
    assert m['phan']['thoi_gian'] == pytest.approx((0.003 + 0.005) * 2.0)
    assert m['phan']['noi_suy'] == pytest.approx(m['alpha_max'] * 0.105 ** 2 / 8)
    s = sum(m['phan'].values())
    assert m['U_IMU_dong_rad'] == pytest.approx(math.radians(1.0) + 1.25 * s)
    assert d['thu'] is True


def test_tong_adds_level_and_refuses_changes(tmp_path):
    p, _, _ = _c8_log(tmp_path)
    g = _globals()
    out = str(tmp_path / 'u.json')
    kw = dict(period_s=T0, j_lim=0.005, h_lim=0.105, pre_s=2.0, tail_s=1.0)
    U.tong('O1-F08', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, **kw)
    U.tong('O1-F05', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, **kw)
    assert set(json.load(open(out, encoding='utf-8'))['muc']) == {'O1-F08', 'O1-F05'}
    with pytest.raises(U.InputError, match='already there'):
        U.tong('O1-F05', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.25, out=out, **kw)
    with pytest.raises(U.InputError, match='shared parameters'):
        U.tong('O1-K2', [p], g['thang'], g['troi'], g['tre'], u_g=0.01, le=0.5, out=out, **kw)


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
    base = ['thang', '--log', p, '--n-vong', '5', '--cach', 'tay', '--toc-do', 'cham']
    assert U.main(base + ['--chieu', 'ccw', '--out', out]) == 0
    assert json.load(open(out, encoding='utf-8'))['s'] == pytest.approx(0.003, abs=1e-4)
    assert U.main(base + ['--chieu', 'cw', '--out', str(tmp_path / 't2.json')]) == 2
    assert U.main(base + ['--chieu', 'ccw', '--out', out]) == 2          # no overwrite


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


def _c8_pair(tmp_path, lat=0.004, s_plus=0.0, err=None, name='r', t_tail=7.0):
    """+300 / -300 deg in a 16 s run_active (>= T_cmd of F08), still t_tail s after it."""
    a = 13.0
    t1 = a + math.radians(300) / 2.0
    t2 = t1 + math.radians(300) / 2.0
    rate = lambda t: 2.0 if a <= t < t1 else (-2.0 if t1 <= t < t2 else 0.0)      # noqa: E731
    lg, fr = str(tmp_path / f'{name}.csv'), str(tmp_path / f'{name}_frames.csv')
    end = a + 16.0 + t_tail
    make_log(lg, rate, end, lat=lat, s_plus=s_plus, run=(a, a + 16.0))
    write_frames(fr, rate, 10.0, end, offset=0.5, err=err)
    return fr, lg


def _u_file(tmp_path, p, tau, delta=0.002, s_plus=0.0):
    out = str(tmp_path / 'u.json')
    th = dict(s_plus=s_plus, s_minus=0.0, delta_plus=delta, delta_minus=delta, f4_mo=False)
    tr = dict(beta_max=math.radians(0.016), q=math.radians(0.01), s_plus=s_plus, s_minus=0.0)
    U.tong('O1-F08', [p], th, tr, dict(tau_hat_s=tau, dtau_fit_s=0.002), u_g=math.radians(0.5), le=0.25, out=out,
           pre_s=2.0, tail_s=1.0)
    return json.load(open(out, encoding='utf-8'))


def test_tre_finds_delay(tmp_path):
    fr, lg = _c8_pair(tmp_path, lat=0.004)
    r = U.tre_one(fr, lg, 0.5)
    assert r['tau_s'] == pytest.approx(0.004, abs=0.001)
    fr2, lg2 = _c8_pair(tmp_path, lat=0.015, name='r2')
    g = U.tre([(fr, lg, 0.5), (fr2, lg2, 0.5)], dict(s_plus=0.0, s_minus=0.0))
    assert g['tau_hat_s'] == pytest.approx(0.0095, abs=0.001)
    assert g['dtau_fit_s'] == pytest.approx(0.0055 + 0.001, abs=0.001)


def test_tre_wrong_offset_refused(tmp_path):
    fr, lg = _c8_pair(tmp_path)
    with pytest.raises(U.InputError, match='edge'):
        U.tre_one(fr, lg, 0.6)


def test_kiem_clean_run_ok(tmp_path):
    fr, lg = _c8_pair(tmp_path)
    u = _u_file(tmp_path, lg, tau=0.004)
    r = U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    assert r['ok'] and r['n_over'] == 0 and r['closure_ok'] is True
    assert r['n_compared'] > 0.95 * r['n_win']


def test_kiem_catches_camera_error(tmp_path):
    fr, lg = _c8_pair(tmp_path, err=lambda t: math.radians(20.0) if 15.0 <= t < 15.5 else 0.0)  # limit about 10 deg
    u = _u_file(tmp_path, lg, tau=0.004)
    r = U.kiem(fr, lg, 0.5, u, 'O1-F08', u_psi=math.radians(0.5))
    assert r['n_over'] > 0 and not r['ok']


def test_kiem_closure_catches_uncorrected_scale(tmp_path):
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
