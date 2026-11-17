"""Tests of the C8b scripts (c8b_gio, c8b_luoi, c8b_3b, c8b_u), CODE_C8B 17/11/2026, design
CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_C8b_v4.md. Synthetic data with the answers planted."""
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'fwsim'))
import c8b_3b as TB  # noqa: E402
import c8b_gio as G  # noqa: E402
import c8b_luoi as LU  # noqa: E402
import c8b_u as CU  # noqa: E402

LOG_COLS = ['timestamp_us', 'log_seq', 'fl_count', 'fr_count', 'rl_count', 'rr_count', 'vx', 'vy', 'wz', 'sp_vx',
            'sp_vy', 'sp_wz', 'gz_fused', 'age_g_ms', 'gyro_n', 'yaw_rad', 'age_r_ms', 'rv_n', 'reset_n', 'g_valid',
            'r_valid', 'run_active']
FR_COLS = ['idx', 't_s', 'valid', 'x', 'y', 'psi', 'speed', 'wz', 'row_px', 'col_px']


# ── synthetic run ────────────────────────────────────────────────────────────────────────────

def _ramp(t, a, b):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def motion(t):
    s = _ramp(t, 3.0, 4.5) * (1 - _ramp(t, 22.0, 23.5))
    v = 0.5 * s * (1 + 0.3 * np.sin(0.7 * t))
    w = 0.8 * s * np.sin(0.45 * t + 0.4)
    return v, w


def make_run(d, D0=1.234, lam_g=0.020, lam_r=0.015, T=30.0, lam_enc=0.0, mutate=None, frames_mutate=None):
    """Log clock = true time; camera clock t_cam = t - D0. Gyro / RV received lam_g / lam_r after the sample, so the
    IMU channels give D0 + lam; the encoder channels D0 + lam_enc. Returns (frames path, log path)."""
    dt = 0.0005
    tt = np.arange(0, T, dt)
    v, w = motion(tt)
    psi = np.cumsum(w) * dt
    x = np.cumsum(v * np.cos(psi)) * dt
    y = np.cumsum(v * np.sin(psi)) * dt
    r, L, ppr = G.R_W, G.LXY, G.PPR
    vx, vy = v, 0 * v
    wheels = np.column_stack([(vx - vy - L * w) / r, (vx + vy + L * w) / r, (vx + vy - L * w) / r,
                              (vx - vy + L * w) / r])
    ang = np.cumsum(wheels, axis=0) * dt * ppr / (2 * math.pi)
    rows = []
    tl = np.arange(0.0, T - 0.1, 0.010) + 0.0003
    gk = np.arange(0.0, T, 0.05) + 0.011            # gyro sample times
    rk = np.arange(0.0, T, 0.05) + 0.027            # RV sample times
    for i, t in enumerate(tl):
        k = int(round((t - lam_enc) / dt))
        cnt = np.round(ang[max(k, 0)] + 50000).astype(int)
        jg = np.searchsorted(gk + lam_g, t, side='right') - 1
        jr = np.searchsorted(rk + lam_r, t, side='right') - 1
        vv, ww = np.interp(t, tt, v), np.interp(t, tt, w)
        run = 1 if 2.9 <= t <= 24.0 else 0
        cmd = vv if run else 0.0
        row = dict(timestamp_us=int(round(t * 1e6)), log_seq=1000 + i, fl_count=cnt[0], fr_count=cnt[1],
                   rl_count=cnt[2], rr_count=cnt[3], vx=f'{vv:.4f}', vy='0', wz=f'{ww:.4f}',
                   sp_vx=f'{cmd:.3f}', sp_vy='0', sp_wz='0',
                   gz_fused=f'{np.interp(gk[jg], tt, w):.6f}', age_g_ms=f'{(t - gk[jg] - lam_g) * 1e3:.2f}',
                   gyro_n=jg + 1, yaw_rad=f'{math.remainder(np.interp(rk[jr], tt, psi) + 0.5, 2 * math.pi):.6f}',
                   age_r_ms=f'{(t - rk[jr] - lam_r) * 1e3:.2f}', rv_n=jr + 1, reset_n=3, g_valid=1, r_valid=1,
                   run_active=run)
        rows.append(row)
    if mutate:
        mutate(rows)
    lp = os.path.join(d, 'run.csv')
    with open(lp, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=LOG_COLS)
        wr.writeheader()
        wr.writerows(rows)
    fr = []
    tc = np.arange(0.2, T - D0 - 0.2, 1 / 60)
    for i, c in enumerate(tc):
        t = c + D0
        fr.append(dict(idx=i, t_s=f'{c:.6f}', valid=1, x=f'{np.interp(t, tt, x):.6f}', y=f'{np.interp(t, tt, y):.6f}',
                       psi=f'{math.remainder(np.interp(t, tt, psi), 2 * math.pi):.6f}',
                       speed=f'{np.interp(t, tt, v):.6f}', wz=f'{np.interp(t, tt, w):.6f}',
                       row_px=f'{1000 + 300 * math.sin(0.1 * t):.3f}', col_px=f'{1900 + 100 * math.cos(0.1 * t):.3f}'))
    if frames_mutate:
        frames_mutate(fr)
    fp = os.path.join(d, 'frames.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=FR_COLS)
        wr.writeheader()
        wr.writerows(fr)
    return fp, lp


@pytest.fixture(scope='module')
def base_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp('run'))
    fp, lp = make_run(d)
    out, F, S = G.analyse(fp, lp, 'O1-F05')
    return out, fp, lp


# ── c8b_gio ──────────────────────────────────────────────────────────────────────────────────

def test_fk_inverse():
    vx, vy, wz = 0.4, -0.1, 0.7
    r, L = G.R_W, G.LXY
    wh = np.array([[(vx - vy - L * wz) / r, (vx + vy + L * wz) / r, (vx + vy - L * wz) / r, (vx - vy + L * wz) / r]])
    np.testing.assert_allclose(G.fk(wh * G.PPR / (2 * math.pi))[0], [vx, vy, wz], atol=1e-12)


def test_gapped_no_bridge():
    t, v = G._gapped([0, 1, 2, 3], [1.0, 1.0, 5.0, 5.0], np.array([False, True, False]))
    assert np.isnan(np.interp(1.5, t, v)) and np.isnan(np.interp(1.2, t, v))
    assert np.interp(0.5, t, v) == 1.0 and np.interp(2.5, t, v) == 5.0


def test_cam_grid_no_bridge_over_rejected_frame():
    t = np.arange(10) / 60
    v = np.arange(10, dtype=float)
    valid = np.ones(10, bool)
    valid[4] = False
    tg, vg = G.cam_grid(t, v, valid, 0.0, 9 / 60)
    between = (tg > 3 / 60) & (tg < 5 / 60)
    assert np.all(np.isnan(vg[between]))
    ok = (tg > 1 / 60) & (tg < 3 / 60)
    np.testing.assert_allclose(vg[ok], tg[ok] * 60, atol=1e-9)


def test_interval_planted(base_run):
    out = base_run[0]
    D0 = 1.234
    assert out['ok'], out['ly_do']
    assert abs(out['L_s'] - D0) < 0.0015
    assert abs(out['U_s'] - (D0 + 0.015)) < 0.0015          # min of gyro (20 ms) and RV (15 ms)
    assert out['U_kenh'] == 'rv_rx' and out['L_kenh'] in G.LOWER
    assert abs(out['kenh']['gyro_rx']['D_s'] - (D0 + 0.020)) < 0.0015
    nd = out['nodes_s']
    c, h = out['center_s'], out['half_s']
    assert abs(h - G.LE * 0.5 * (out['U_s'] - out['L_s'])) < 1e-12
    assert abs(nd[0] - (c - h)) < 1e-12 and abs(nd[-1] - (c + h)) < 1e-12
    assert max(np.diff(nd)) <= G.STEP_S + 1e-12
    assert out['moc_mau'] is True and out['ket_luan'] == 'ok'


def test_nodes_zero_width():
    assert G.nodes(1.0, 0.0) == [1.0]
    for h in (0.005, 0.0105, 0.00631, 0.0125, 0.003 * 7 / 3):
        nd = G.nodes(1.234, h)
        assert max(np.diff(nd)) <= 0.001 and nd[0] == pytest.approx(1.234 - h) and nd[-1] == pytest.approx(1.234 + h)
    n = G.nodes(1.0, 0.0105)
    assert len(n) in (22, 23) and max(np.diff(n)) <= 0.001     # 23 when float rounding puts a step above 1 ms


def test_drift_distance_and_spans(base_run):
    out = base_run[0]
    cw = out['cua_so_ghep_log_s']
    t0, td = out['t0_log_s'], out['t_deadline_s']
    assert out['troi_khoang_xa_s'] == pytest.approx(max(abs(t0 + td - cw[0]), abs(cw[1] - t0)))
    mt = out['man_trap']
    assert mt['frame_s'] == pytest.approx(1 / 60, rel=1e-3)
    assert 700 <= mt['row_px'][0] < mt['row_px'][1] <= 1300 and mt['col_px'] is not None


def test_imu_lead_gives_L_above_U(tmp_path):
    fp, lp = make_run(str(tmp_path), lam_g=-0.010, lam_r=-0.012)      # IMU ahead of the wheels: assumption broken
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert not out['ok'] and out['ket_luan'] == 'hong_he_do'               # O1 v11.34 Q2: per-run check -> lost
    assert any('> U~' in x for x in out['ly_do_hong']) and out['nodes_s'] == []


def test_pairs_invalid_lost(tmp_path):
    def m(rows):
        for r in rows[600:900:4]:
            r['fl_count'] = G.INT32_MIN
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ket_luan'] == 'hong_he_do' and any('count pairs' in x for x in out['ly_do_hong'])


def test_log_hole_lost(tmp_path):
    def m(rows):
        del rows[1000:1012]                  # 120 ms without lines (log_seq jump 12 > 4 also refused by log_window)
    fp, lp = make_run(str(tmp_path), mutate=m)
    with pytest.raises(G.GioError):
        G.analyse(fp, lp, 'O1-F05')


def test_log_hole_small_seq_jump_lost(tmp_path):
    def m(rows):
        for r in rows[1000:]:
            r['timestamp_us'] = int(r['timestamp_us']) + 110000         # 120 ms gap, log_seq continuous
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ket_luan'] == 'hong_he_do' and any('log hole' in x for x in out['ly_do_hong'])


@pytest.mark.parametrize('col,val,txt', [('reset_n', 4, 'reset_n'), ('g_valid', 0, 'g_valid'), ('r_valid', 0, 'r_valid'),
                                         ('age_g_ms', '-2.5', 'IMU age')])
def test_blocking_checks_not_concluded(tmp_path, col, val, txt):
    def m(rows):
        rows[1200][col] = val
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ket_luan'] == 'chua_ket_luan' and any(txt in x for x in out['ly_do_chua'])


def test_negative_age_down_to_2ms_ok(tmp_path):
    def m(rows):
        rows[1200]['age_g_ms'] = '-1.9'
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ok'] and out['kiem']['n_age_neg'] >= 1


def test_repeated_log_seq_refused(tmp_path):
    def m(rows):
        rows[500]['log_seq'] = rows[499]['log_seq']
    fp, lp = make_run(str(tmp_path), mutate=m)
    with pytest.raises(G.GioError, match='repeated log_seq'):
        G.analyse(fp, lp, 'O1-F05')


def test_c8_range_check(tmp_path, base_run):
    out = base_run[0]
    w = out['width_s']
    khoa = dict(dai={'O1-F05': dict(width_s=[w + 0.002, w + 0.004], gyro_minus_rv_s=[-1, 1])})
    out2, _, _ = G.analyse(base_run[1], base_run[2], 'O1-F05', khoa)
    assert out2['ket_luan'] == 'hong_he_do' and any('C8 range' in x for x in out2['ly_do_hong'])
    khoa['dai']['O1-F05']['width_s'] = [w - 0.001, w + 0.001]
    out3, _, _ = G.analyse(base_run[1], base_run[2], 'O1-F05', khoa)
    assert out3['ok']


def _startup_inputs(fp, lp):
    F = G.read_frames(fp)
    rows, _ = G.read_log(lp)
    tl = np.array([float(r['timestamp_us']) for r in rows]) * 1e-6
    tm, rate, br, _, _ = G.count_pairs(rows, tl)
    tr, yr, _ = G.imu_reports(rows, tl, 'rv_n', 'age_r_ms', 'yaw_rad')
    return F, tm, G.fk(rate), br, tr, yr


def test_startup_bounds_planted(base_run):
    out = base_run[0]
    st = out['khoi_dong']
    assert st['ket_qua'] == 'nhat_quan' and st['lo_s'] is not None
    # synthetic: the tag is the wheel centre, no slip: the lower estimate sits at D0 (within the camera allowance)
    assert 1.234 - 0.08 < st['lo_s'] <= 1.234 + 0.003


@pytest.mark.parametrize('lo,hi,want', [(1.015, 1.005, 'khong_nhat_quan'), (1.005, 1.015, 'nhat_quan'),
                                          (1.025, None, 'khong_nhat_quan'), (None, 0.995, 'khong_nhat_quan')])
def test_startup_intersection_counterexample(base_run, monkeypatch, lo, hi, want):
    # C8BS-4, C8BS-17: L = 1.002, U~ = 1.018 -> scoring interval exactly [1.000, 1.020]; drift 0
    D = dict(speed_cnt=1.002, wz_cnt=1.002, gyro_rx=1.018, rv_rx=1.018)

    def fake_match(t_cam, v_cam, valid, t_log, v_log, window_log, coarse):
        ch = fake_match.order.pop(0)
        return dict(D_s=D[ch], corr=0.99, window_cam_s=[3.0, 20.0], window_log_s=[4.0, 21.0], short=False,
                    D_start_s=D[ch], D_end_s=D[ch], corr_start=0.99, corr_end=0.99, rate=0.0, ok=True, why='')
    fake_match.order = list(G.CHANNELS)
    monkeypatch.setattr(G, 'match', fake_match)
    monkeypatch.setattr(G, 'startup', lambda *a, **k: dict(lo_s=lo, hi_s=hi, lo_why='', hi_why='', t_start_log_s=4.0,
                                                           e_p_m=0, e_psi_deg=0))
    out, _, _ = G.analyse(base_run[1], base_run[2], 'O1-F05')
    assert out['center_s'] - out['half_s'] == pytest.approx(1.000) and out['center_s'] + out['half_s'] == pytest.approx(1.020)
    assert out['khoi_dong']['ket_qua'] == want
    assert (out['ket_luan'] == 'hong_he_do') == (want == 'khong_nhat_quan')


def test_controller_from_events(tmp_path):
    lp = tmp_path / 'run.csv'
    lp.write_text('x\n')
    assert G.controller_of(str(lp)) == (None, None)
    (tmp_path / 'run.csv.events.txt').write_text('t #gain,id=0,ctrl=LADRC,kp_xy=1\nt #gain,id=0,ctrl=LADRC,kp_xy=1\n')
    assert G.controller_of(str(lp))[0] == 'LADRC'
    (tmp_path / 'run.csv.events.txt').write_text('t #gain,ctrl=LADRC\nt #gain,ctrl=PID\n')
    assert G.controller_of(str(lp))[0] is None


def test_startup_not_bridged(tmp_path):
    def fm(fr):                                           # a rejected frame just after the car leaves
        for r in fr:
            if 2.9 + 0.5 < float(r['t_s']) + 1.234 < 2.9 + 0.52:
                r['valid'] = 0
    fp, lp = make_run(str(tmp_path), frames_mutate=fm)
    F, tm, vel, br, tr, yr = _startup_inputs(fp, lp)
    st = G.startup(F, 2.9003, 1.234, tm, vel, br, tr, yr)
    assert st['hi_s'] is None and 'rejected or lost camera frame' in st['hi_why']
    br2 = br.copy()
    k = int(np.searchsorted(tm, 3.2))
    br2[k] = True                                         # a break in the count pairs inside the start-up span
    st2 = G.startup(F, 2.9003, 1.234, tm, vel, br2, tr, yr)
    assert st2['lo_s'] is None and 'break in the count pairs' in st2['lo_why']


def test_third_correlation_checked(monkeypatch):
    # C8BS-1: whole window 0.96, thirds 0.2 / 0.3: not ok
    seq = iter([(1.0, 0.96), (1.0, 0.2), (1.0, 0.3)])
    monkeypatch.setattr(G, '_fit', lambda *a, **k: next(seq))
    t = np.arange(0, 30, 1 / 60)
    r = G.match(t, np.ones_like(t), np.ones_like(t, bool), t, np.ones_like(t), (0.0, 29.0), 0.0)
    assert not r['ok'] and 'start third' in r['why'] and 'end third' in r['why']


def test_count_pair_time_gap_not_used(tmp_path):
    def m(rows):
        for r in rows[1000:]:
            r['timestamp_us'] = int(r['timestamp_us']) + 40000           # 50 ms between two lines, log_seq +1
    fp, lp = make_run(str(tmp_path), mutate=m)
    rows, _ = G.read_log(lp)
    tl = np.array([float(r['timestamp_us']) for r in rows]) * 1e-6
    tm, rate, br, pair_ok, ip = G.count_pairs(rows, tl)
    assert not pair_ok[999] and pair_ok[998] and pair_ok[1000]
    j = int(np.searchsorted(ip, 999))
    assert br[j - 1]                                      # the pairs on both sides are not joined
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ok']                                      # 50 ms < LOG_HOLE_S: the run itself is not lost


def test_cli_writes_once(tmp_path, base_run):
    out_p = str(tmp_path / 'gio.json')
    args = ['--frames', base_run[1], '--esp32', base_run[2], '--task', 'O1-F05', '--out', out_p]
    assert G.main(args) == 0
    d = json.load(open(out_p, encoding='utf-8'))
    assert d['format'] == 'c8b_gio_v1' and len(d['nguon']) == 2
    import c8_u_imu
    assert d['sha256'] == dict(esp32=c8_u_imu.sha256(base_run[2]), frames=c8_u_imu.sha256(base_run[1]))
    assert G.main(args) == 2                                   # never overwritten


def test_sealed_path_refused(tmp_path, base_run):
    p = tmp_path / 'sealed'
    p.mkdir()
    assert G.main(['--frames', base_run[1], '--esp32', base_run[2], '--task', 'O1-F05',
                   '--out', str(p / 'g.json')]) == 2


# ── c8b_luoi ─────────────────────────────────────────────────────────────────────────────────

def node(states, rows=None, nums=None, ok=True, u_t=0.03, kind='hong', cat=False, bien=(None, None)):
    if not ok:
        return dict(ok=False, kind=kind, why=kind)
    st = dict(zip(LU.CONDS, states)) if isinstance(states, (list, tuple)) else states
    nm = nums or {}
    for c in LU.MEASURED:                         # numbers that follow the rows and the state (C8BS-17)
        if c in nm:
            continue
        los = [r[c] for r in (rows or {}).values() if r['mien_chung']] or [0.04]
        lo = max(los)
        hi = (LU.THRESHOLDS[c] + 0.05 if st[c] != 'dat' else max(lo + 0.02, 0.06)) if lo <= LU.THRESHOLDS[c] \
            else lo + 0.02
        nm[c] = dict(q=lo + 0.01, lo=lo, hi=hi, n_undef=0, q_chung=lo + 0.01, lo_chung=lo, hi_chung=hi,
                     n_undef_chung=0)
    gate = {c: dict(state=st[c]) for c in LU.CONDS}
    gate['g1'] = dict(state=st['g1'], t_done=[1.0, 1.0])
    return dict(ok=True, kind='ok', outcome='x', states=st, nums=nm, rows=rows or {}, u_t_s=u_t, cat=cat, gate=gate,
                bien=bien)


def rows_with(viol_key=None, c='g3', lo=0.12, in_dom=True, extra=()):
    r = {k: dict(mien_chung=True, **{m: 0.05 for m in LU.MEASURED}) for k in (0.1, 0.2, 0.3)}
    for k in extra:
        r[k] = dict(mien_chung=True, **{m: 0.05 for m in LU.MEASURED})
    if viol_key is not None:
        r.setdefault(viol_key, dict(mien_chung=True, **{m: 0.05 for m in LU.MEASURED}))
        r[viol_key][c] = lo
        r[viol_key]['mien_chung'] = in_dom
    return r


def test_thresholds_match_o1_gate():
    import o1_gate
    assert LU.THRESHOLDS == o1_gate.THRESHOLDS


def test_all_dat():
    res = LU.combine([node(['dat'] * 5) for _ in range(4)])
    assert all(v == 'dat' for v in res['states'].values()) and res['outcome'] == 'dat'


def test_one_node_chua():
    n = [node(['dat'] * 5) for _ in range(4)]
    n[2] = node(['dat', 'dat', 'chua_ket_luan', 'dat', 'dat'])
    res = LU.combine(n)
    assert res['states']['g3'] == 'chua_ket_luan' and res['outcome'] == 'chua_ket_luan'


def test_khong_dat_same_frame_every_cell():
    st = ['dat', 'dat', 'khong_dat', 'dat', 'dat']
    n = [node(st, rows_with(0.2)) for _ in range(3)]
    res = LU.combine(n)
    assert res['states']['g3'] == 'khong_dat' and res['outcome'] == 'khong_dat'
    assert res['nums']['g3']['chung']['lo'] == pytest.approx(0.12)
    assert res['bieu_dien']['g3']


def test_khong_dat_needs_same_frame_both_nodes():
    st = ['dat', 'dat', 'khong_dat', 'dat', 'dat']
    # C8B4R2-2 round 4: frame 0.2 violates at node a but is out of the domain at node b, frame 0.3 the other way
    a = node(st, rows_with(0.2))
    a['rows'][0.3] = dict(mien_chung=False, **{m: 0.05 for m in LU.MEASURED})
    b = node(st, rows_with(0.3))
    b['rows'][0.2] = dict(mien_chung=False, **{m: 0.05 for m in LU.MEASURED})
    res = LU.combine([a, b])
    assert res['states']['g3'] == 'chua_ket_luan'
    # the same frames in the domain at both nodes: each one's bound (U_t with the whole step) holds over the cell
    a['rows'][0.3]['mien_chung'] = True
    b['rows'][0.2]['mien_chung'] = True
    assert LU.combine([a, b])['states']['g3'] == 'khong_dat'


def test_khong_dat_frame_out_of_domain_at_one_node():
    st = ['dat', 'dat', 'khong_dat', 'dat', 'dat']
    a = node(st, rows_with(0.2))
    b = node(st, rows_with(0.2, lo=0.05, in_dom=False))
    res = LU.combine([a, b])
    assert res['states']['g3'] == 'chua_ket_luan'


def test_khong_dat_lo_of_either_node_counts():
    st = ['dat', 'dat', 'khong_dat', 'dat', 'dat']
    a = node(st, rows_with(0.2, lo=0.12))
    b = node(['dat', 'dat', 'chua_ket_luan', 'dat', 'dat'], rows_with(0.2, lo=0.09))
    res = LU.combine([a, b])
    assert res['states']['g3'] == 'khong_dat'


def test_g1_rule():
    n = [node(['khong_dat', 'dat', 'dat', 'dat', 'dat']) for _ in range(3)]
    assert LU.combine(n)['states']['g1'] == 'khong_dat'
    n[1] = node(['chua_ket_luan', 'dat', 'dat', 'dat', 'dat'])
    assert LU.combine(n)['states']['g1'] == 'chua_ket_luan'
    n[1] = node(['dat', 'dat', 'dat', 'dat', 'dat'])
    assert LU.combine(n)['states']['g1'] == 'chua_ket_luan'


def test_error_node_chua_and_all_lost():
    n = [node(['dat'] * 5) for _ in range(3)]
    n[1] = node(None, ok=False)
    res = LU.combine(n)
    assert all(v == 'chua_ket_luan' for v in res['states'].values()) and res['outcome'] == 'chua_ket_luan'
    assert res['nums']['g2']['n_undef'] >= 1 and res['nums']['g2']['chung']['n_undef'] >= 1   # GATE v2 2 point 3
    assert res['bieu_dien']['g2'] and res['hong_nut'] == [False, True, False]
    res2 = LU.combine([node(None, ok=False)] * 3)
    assert res2['outcome'] == 'hong_he_do' and res2['nut_tam'] is None
    res3 = LU.combine([node(None, ok=False, kind='loi')] * 3)          # tool problems are not the measurement
    assert res3['outcome'] == 'chua_ket_luan' and res3['hong_nut'] == [False] * 3


def test_cat_every_node():
    st = ['chua_ket_luan', 'dat', 'khong_dat', 'dat', 'dat']
    n = [node(st, rows_with(0.2), cat=True) for _ in range(3)]
    res = LU.combine(n)
    assert res['outcome'] == 'khong_dat' and res['fail_cause'] == 'vi_pham_truoc_loi_he_do' and all(res['cat_nut'])
    n[1] = node(st, rows_with(0.2), cat=False)
    res = LU.combine(n)
    assert res['fail_cause'] == '' and res['cat_nut'] == [True, False, True]


def test_gate_block_and_luoi_v1():
    st = ['dat', 'dat', 'khong_dat', 'dat', 'dat']
    n = [node(st, rows_with(0.2), u_t=0.02 + 0.001 * i) for i in range(5)]
    n[2] = node(None, ok=False)                     # centre lost: nut_tam the nearest scored node
    res = LU.combine(n)
    gio = dict(nodes_s=[1.0, 1.001, 1.002, 1.003, 1.004])
    g = LU.gate_block(n, res, gio, 'a' * 64)
    lb = g['luoi']
    assert lb['format'] == 'luoi_v1' and lb['n_nut'] == 5 and lb['nut_tam'] == 1 and lb['d_dau_s'] == 1.0
    assert lb['buoc_max_s'] == pytest.approx(0.001) and lb['hong_nut'] == [False, False, True, False, False]
    assert lb['g1_nut'][2] == 'chua_ket_luan' and lb['g1_state'] == 'chua_ket_luan' == g['g1']['state']
    assert g['g1']['t_done'] == [1.0, 1.0] and g['u_t'] == pytest.approx(0.024)
    assert g['g3']['state'] == 'chua_ket_luan' and g['g3']['n_undef'] >= 1


def test_domain_end_moves_between_nodes():
    # C8BS-6: both nodes dat, but t_done is set by another frame at the next node: not concluded
    a = node(['dat'] * 5, rows_with(), bien=(10.0, 10.0))
    b = node(['dat'] * 5, rows_with(), bien=(10.0, 10.016667))
    res = LU.combine([a, b])
    assert all(res['states'][c] == 'chua_ket_luan' for c in LU.MEASURED) and res['bien_doi'] == [True]
    assert res['nums']['g2']['n_undef'] >= 1 and res['bieu_dien']['g2']
    assert LU.combine([a, node(['dat'] * 5, rows_with(), bien=(10.0, 10.0))])['states']['g2'] == 'dat'


def test_domain_moves_with_violation_gives_no_finite_lo():
    # C8BS-R2-1: frame 0.2 violates G3 in both domains but the domain moves: not concluded, written (not refused)
    a = node(['dat', 'dat', 'khong_dat', 'dat', 'dat'], rows_with(0.2), bien=(10.0, 10.0))
    b = node(['dat', 'dat', 'khong_dat', 'dat', 'dat'], rows_with(0.2), bien=(10.0, 10.016667))
    res = LU.combine([a, b])
    assert res['states']['g3'] == 'chua_ket_luan' and math.isnan(res['nums']['g3']['chung']['lo'])
    assert all(res['bieu_dien'][c] for c in LU.MEASURED)


def test_domain_key_needs_t0_and_D():
    with pytest.raises(LU.LuoiError, match='C8BS-6'):
        LU.domain_key(dict(t0_log_us=5e6), dict(g1=dict(t_done=[2.0, 3.0])))


def test_domain_key_camera_clock():
    sc = dict(t0_log_us=5e6, gio_c8b=dict(D=1.5), cat_t1=False)
    g = dict(g1=dict(t_done=[2.0, None]))
    assert LU.domain_key(sc, g) == (5.5, None)               # 2.0 + 5.0 - 1.5 on the camera clock


def test_cat_only_report_only_is_lost():
    # C8BS-7: the exception rests only on G5, which is report-only: the node is lost, not khong_dat
    st = ['chua_ket_luan', 'dat', 'dat', 'dat', 'khong_dat']
    n = [node(st, rows_with(0.2, c='g5', lo=8.0), cat=True) for _ in range(3)]
    assert LU.combine(n)['outcome'] == 'khong_dat'
    res = LU.combine(n, frozenset({'g5'}))
    assert res['outcome'] == 'hong_he_do' and res['hong_nut'] == [True] * 3 and res['cat_nut'] == [False] * 3


def test_gate_block_decide_follows_mask():
    # C8BS-8: G5 report-only: the combined block decides G1-G4 and its outcome_decide is dat
    st = ['dat', 'dat', 'dat', 'dat', 'khong_dat']
    n = [node(st, rows_with(0.2, c='g5', lo=8.0)) for _ in range(2)]
    res = LU.combine(n, frozenset({'g5'}))
    g = LU.gate_block(n, res, dict(nodes_s=[1.0, 1.001]), 'a' * 64, 'O1-F08', 'b' * 64)
    assert g['decide'] == ['g1', 'g2', 'g3', 'g4'] and g['outcome_decide'] == 'dat' and g['outcome'] == 'khong_dat'
    assert g['report_only'] == ['g5'] and g['gate_file_sha256'] == 'b' * 64 and g['level'] == 'O1-F08'


def test_node_state_must_follow_numbers(tmp_path):
    # C8BS-9: the recorded G3 state dat with lo 0.12 > 0.10 in its own numbers: the node is a tool error
    sp = tmp_path / 's.json'
    g = {'g1': dict(state='dat')}
    for c in LU.MEASURED:
        lo = 0.12 if c == 'g3' else 0.04
        g[c] = dict(state='dat', q=lo, lo=lo, hi=lo + 0.01, n_undef=0, chung=dict(q=lo, lo=lo, hi=lo + 0.01, n_undef=0))
    json.dump(dict(outcome='dat', gate_v3=g), open(sp, 'w'))
    rp = tmp_path / 'r.csv'
    rp.write_text('t_cam_s,mien_chung,mien_hop,lo_g2,lo_g3,lo_g4,lo_g5\n1,1,1,0,0,0,0\n')
    n = LU.node_from_files(str(sp), str(rp))
    assert not n['ok'] and n['kind'] == 'loi' and 'does not follow' in n['why']


def test_single_node_uses_its_state():
    res = LU.combine([node(['dat', 'chua_ket_luan', 'khong_dat', 'dat', 'dat'], rows_with(0.2))])
    assert res['states']['g2'] == 'chua_ket_luan' and res['states']['g3'] == 'khong_dat'


def test_report_only_not_in_outcome():
    st = ['dat', 'dat', 'dat', 'dat', 'khong_dat']
    n = [node(st, rows_with(0.2, c='g5', lo=8.0)) for _ in range(2)]
    assert LU.combine(n)['outcome'] == 'khong_dat'
    res = LU.combine(n, frozenset({'g5'}))
    assert res['states']['g5'] == 'khong_dat' and res['outcome'] == 'dat' and 'g5' not in res['decide']


def test_state_from_numbers_matches_rq3_log():
    import rq3_log
    rng = np.random.default_rng(3)
    vals = lambda: rng.choice([math.nan, math.inf, rng.uniform(0, 0.2)])
    for _ in range(2000):
        lo_c, lo_u, hi_u = vals(), vals(), vals()
        nu = [0, 1, None][int(rng.integers(0, 3))]
        assert LU._state_from_numbers(lo_c, lo_u, hi_u, nu, 0.1) == rq3_log.cond_state2(lo_c, lo_u, hi_u, nu, 0.1)


def _write_node(d, k, states, rows, u_t=0.031):
    g = {'g1': dict(state=states[0])}
    for i, (c, s) in enumerate(zip(LU.MEASURED, states[1:])):
        lo = max(r[2][i] for r in rows)                  # numbers that follow the rows and the state (C8BS-17)
        hi = lo + 0.02 if s != 'dat' else min(lo + 0.02, LU.THRESHOLDS[c])
        g[c] = dict(state=s, q=lo + 0.01, lo=lo, hi=hi, n_undef=0, chung=dict(q=lo + 0.01, lo=lo, hi=hi, n_undef=0))
    with open(os.path.join(d, f'nut_{k:03d}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(outcome='dat', gate_v3=g, U_t_s=u_t, t0_log_us=5e6, gio_c8b=dict(D=1.0 + 0.001 * k)), f)
    with open(os.path.join(d, f'nut_{k:03d}_rows.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t', 't_cam_s', 'mien_chung', 'mien_hop'] + [f'lo_{c}' for c in LU.MEASURED])
        for t_cam, mc, los in rows:
            w.writerow([0.0, t_cam, mc, 1] + los)


def test_node_files_and_gop_only(tmp_path):
    d = str(tmp_path / 'luoi')
    os.makedirs(d)
    gio = dict(format='c8b_gio_v1', task='O1-F05', ok=True, nodes_s=[1.0, 1.001])
    gp = str(tmp_path / 'gio.json')
    json.dump(gio, open(gp, 'w', encoding='utf-8'))
    import c8_u_imu
    for k in range(2):
        with open(os.path.join(d, f'nut_{k:03d}_nguon.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(scorer=dict(commit='abc1234', dirty=False), gio_sha256=c8_u_imu.sha256(gp),
                           D_s=gio['nodes_s'][k]), f)
        _write_node(d, k, ['dat', 'dat', 'khong_dat', 'dat', 'dat'],
                    [(10.000001, 1, [0.05, 0.12, 0.05, 1.0]), (10.016668, 1, [0.05, 0.05, 0.05, 1.0])])
    with open(os.path.join(d, 'nut_000.json'), encoding='utf-8') as f:
        assert json.load(f)['gate_v3']['g3']['lo'] == 0.12
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['states']['g3'] == 'khong_dat' and r['outcome'] == 'khong_dat' and r['u_t_nodes'] == [0.031, 0.031]
    assert r['scorer']['commit'] == 'abc1234' and r['scorer']['dirty'] is False      # from the node records
    os.remove(os.path.join(d, 'luoi.json'))
    os.remove(os.path.join(d, 'nut_001_nguon.json'))                                # C8BS-16: no record, refused
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 2


def test_node_rows_without_c8b_columns_is_error(tmp_path):
    sp = tmp_path / 's.json'
    g = {c: dict(state='dat', q=0.05, lo=0.04, hi=0.06, n_undef=0, chung=dict(q=0.05, lo=0.04, hi=0.06, n_undef=0))
         for c in LU.MEASURED}
    g['g1'] = dict(state='dat')
    json.dump(dict(outcome='dat', gate_v3=g, t0_log_us=5e6, gio_c8b=dict(D=1.0)), open(sp, 'w'))
    rp = tmp_path / 'r.csv'
    rp.write_text('t,lo_g2,lo_g3,lo_g4,lo_g5\n0,0,0,0,0\n')
    n = LU.node_from_files(str(sp), str(rp))
    assert not n['ok'] and 'interface' in n['why']


def test_not_bounded_clock_file(tmp_path):
    gp = str(tmp_path / 'gio.json')
    json.dump(dict(format='c8b_gio_v1', task='O1-F05', ok=False, ket_luan='chua_ket_luan', ly_do=['x'], nodes_s=[]),
              open(gp, 'w', encoding='utf-8'))
    d = str(tmp_path / 'o')
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['outcome'] == 'chua_ket_luan' and set(r['states'].values()) == {'chua_ket_luan'}


FAKE_SCORER = r'''
import sys, json, csv
a = sys.argv[1:]
g = lambda k: a[a.index(k) + 1]
k = int(g('--nut'))
lo3 = 0.13 if k < 3 else 0.05
gate = {'g1': {'state': 'dat', 't_done': [None, None]}}
for c in ('g2', 'g3', 'g4', 'g5'):
    lo = lo3 if c == 'g3' else 0.05
    thr = 7.5 if c == 'g5' else 0.10
    lo = 1.0 if c == 'g5' else lo
    s = 'khong_dat' if lo > thr else 'dat'
    hi = lo + 0.01
    gate[c] = dict(state=s, q=lo, lo=lo, hi=hi, n_undef=0, chung=dict(q=lo, lo=lo, hi=hi, n_undef=0))
json.dump(dict(outcome='x', gate_v3=gate, U_t_s=0.02 + 0.001 * k, t0_log_us=5e6, gio_c8b=dict(D=1.0 + 0.001 * k)),
          open(g('--out'), 'w'))
with open(g('--rows'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['t_cam_s', 'mien_chung', 'mien_hop', 'lo_g2', 'lo_g3', 'lo_g4', 'lo_g5'])
    w.writerow([5.0, 1 if k < 3 else 0, 1, 0.05, 0.13 if k < 3 else 0.05, 0.05, 1.0])   # out of the domain at node 3
print('#ket_qua,ma=0,outcome=khong_dat')
'''


def test_run_nodes_with_fake_scorer(tmp_path):
    sc = tmp_path / 'fake_score.py'
    sc.write_text(FAKE_SCORER)
    gp = str(tmp_path / 'gio.json')
    json.dump(dict(format='c8b_gio_v1', task='O1-F05', ok=True, nodes_s=[1.0, 1.001, 1.002, 1.003]),
              open(gp, 'w', encoding='utf-8'))
    d = str(tmp_path / 'luoi')
    rc = LU.main(['--gio', gp, '--out-dir', d, '--frames', 'f.csv', '--esp32', 'l.csv', '--hang-so', 'h.json',
                  '--scorer', str(sc)])
    assert rc == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert len(r['scorer']['nut']) == 4 and all('commit' in x for x in r['scorer']['nut'])
    assert r['n_nodes'] == 4 and r['states']['g3'] == 'chua_ket_luan'       # last cell: frame not in both domains
    assert r['nodes_ok'] == [True] * 4 and not any(r['node_errors'])
    assert r['u_t_nodes'] == pytest.approx([0.02, 0.021, 0.022, 0.023])


# ── c8b_3b ───────────────────────────────────────────────────────────────────────────────────

def make_wheel(path, R, alpha_r=7.7e-6, wob=0.002, noise=0.0001, seed=1, drift_centre=0.0, one_way=False,
               slow=False, with_col=True):
    rng = np.random.default_rng(seed)
    T = 1 / 60
    tt = np.arange(0, 70, T)
    om = np.where(tt < 34, 6.0 * np.exp(-tt / 12) + 0.3, np.where(tt < 35, 0.0, -(6.0 * np.exp(-(tt - 35) / 12) + 0.3)))
    if one_way:
        om = np.abs(om)
    if slow:
        om = np.clip(om, -1.0, 1.0)
    th = np.cumsum(om) * T
    out = []
    for ti, a, w in zip(tt, th, om):
        cx = 1.0 + drift_centre * math.sin(ti)
        row = 1100 + R * math.sin(a) / 0.0006
        col = 1900 + R * math.cos(a) / 0.0006
        a2 = a + w * alpha_r * (row - 1100) + wob * math.cos(a)
        xm = cx + R * math.cos(a2) + rng.normal(0, noise)
        ym = 0.5 + R * math.sin(a2) + rng.normal(0, noise)
        psi = a2 + 0.3 + rng.normal(0, 0.0003)
        out.append(dict(t_s=f'{ti:.6f}', valid=1, x=xm, y=ym, tag_x=xm, tag_y=ym,
                        psi=math.remainder(psi, 2 * math.pi), row_px=row, col_px=col if with_col else ''))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


@pytest.fixture(scope='module')
def wheel(tmp_path_factory):
    d = tmp_path_factory.mktemp('wheel')
    make_wheel(str(d / 'a.csv'), 0.0005, wob=0.0)
    make_wheel(str(d / 'b.csv'), 0.25)
    return str(d / 'a.csv'), str(d / 'b.csv')


def test_3b_alpha_recovered(wheel):
    r = TB.analyse(*wheel)
    sh = r['B']['shutter']
    assert sh['whole']['alpha_r_s_px'] == pytest.approx(7.7e-6, rel=0.08)
    assert abs(sh['whole']['alpha_c_s_px']) < 1e-6
    assert r['ga_ok'] and r['B']['R0_m'] == pytest.approx(0.25, abs=0.002)
    assert sh['use_alpha'], sh['why']
    assert sh['alpha_r_le_s_px'] == pytest.approx(1.25 * sh['alpha_r_s_px'])
    assert r['dung_duoc'] and r['d_psi_quay_deg'] is not None and r['d_p_quay_m'] is not None


def test_3b_no_shutter_gives_small_alpha(tmp_path):
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0)
    make_wheel(str(tmp_path / 'b.csv'), 0.25, alpha_r=0.0)
    r = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))
    assert abs(r['B']['shutter']['whole']['alpha_r_s_px']) < 0.8e-6


def test_3b_ga_fail(tmp_path):
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, drift_centre=0.004)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    r = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))
    assert not r['A']['ga_ok'] and not r['ga_ok'] and r['d_p_quay_m'] is None


def test_3b_no_col_one_frame(tmp_path):
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0)
    make_wheel(str(tmp_path / 'b.csv'), 0.25, with_col=False)
    sh = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['B']['shutter']
    assert not sh['use_alpha'] and 'col_px' in sh['why']


@pytest.mark.parametrize('kw,part', [(dict(one_way=True), 'phu'), (dict(slow=True), 'phu')])
def test_3b_coverage_needed(tmp_path, kw, part):
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, **kw)
    make_wheel(str(tmp_path / 'b.csv'), 0.25, **kw)
    r = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))
    assert not r['dung_duoc'] and r['d_psi_quay_deg'] is None and r['d_p_quay_m'] is None
    assert any(part in x for x in r['thieu'])


def test_3b_consistency_both_coefficients(wheel, monkeypatch):
    # C8BS-12: four check fits with the same summary number but different (alpha_r, alpha_c): not used
    vals = iter([(5.5e-6, 4.5e-6), (10e-6, 0.0), (1e-6, 9e-6), (5.5e-6, 4.5e-6), (5.5e-6, 4.5e-6)])
    real = TB.fit_shutter

    def fake(*a, **k):
        f = real(*a, **k)
        ar, ac = next(vals)
        f['groups'][0].update(alpha_r_s_px=ar, alpha_c_s_px=ac, has_col=True)
        return f
    monkeypatch.setattr(TB, 'fit_shutter', fake)
    sh = TB.analyse(*wheel)['B']['shutter']
    assert not sh['use_alpha'] and 'spread' in sh['why']


def test_3b_check_fit_without_col(wheel, monkeypatch):
    real = TB.fit_shutter
    calls = []

    def fake(*a, **k):
        f = real(*a, **k)
        calls.append(1)
        if len(calls) == 3:                       # one check fit without columns
            f['groups'][0]['has_col'] = False
        return f
    monkeypatch.setattr(TB, 'fit_shutter', fake)
    sh = TB.analyse(*wheel)['B']['shutter']
    assert not sh['use_alpha'] and 'col_px' in sh['why']


def test_3b_envelope_covers_every_check_fit(wheel, monkeypatch):
    # C8BS-12 round 2: alpha_r = 10 everywhere, alpha_c = -1, +1, -1, +1 us/px, whole (10, 0): the value used must
    # cover a check fit at any span (Dr 10, Dc 700: 0.8 ms)
    vals = iter([(10e-6, 0.0), (10e-6, -1e-6), (10e-6, 1e-6), (10e-6, -1e-6), (10e-6, 1e-6)])
    real = TB.fit_shutter

    def fake(*a, **k):
        f = real(*a, **k)
        ar, ac = next(vals)
        f['groups'][0].update(alpha_r_s_px=ar, alpha_c_s_px=ac, has_col=True)
        return f
    monkeypatch.setattr(TB, 'fit_shutter', fake)
    sh = TB.analyse(*wheel)['B']['shutter']
    assert sh['use_alpha']
    used = sh['alpha_r_le_s_px'] * 10 + sh['alpha_c_le_s_px'] * 700
    assert used >= 10e-6 * 10 + 1e-6 * 700


def test_3b_window_over_turnaround_not_used():
    T = 1 / 60
    tt = np.arange(0, 80, T)
    om = np.where(tt < 40, 6.0 * np.exp(-tt / 15) + 0.3, -(6.0 * np.exp(-(tt - 40) / 15) + 0.3))
    F = dict(t=tt, x=np.zeros_like(tt), y=np.zeros_like(tt), psi=np.cumsum(om) * T)
    assert TB.analyse_a(F, 2.614)['psi_osc_deg'] < 0.01


# ── c8b_u ────────────────────────────────────────────────────────────────────────────────────

CAM = {"version": "c8_cam_u_vao_v2",
       "P1": {"bo_nhom_max_m": 0.002, "tung_khung_max_m": 0.0018, "bo_nhom_rms_m": 0.0014},
       "moc_dich": {"nguong_m": 0.001, "u_so_m": 0.0004, "moc_loai": []},
       "P3": {"vi_tri_max_m": 0.0008, "n_lan": 3},
       "A2": {"goc_max_deg": 0.1, "n_lan": 3},
       "than_dich": {"max_m": 0.0001, "kiem_duoc": True},
       "P7": {"lech_max_m": 0.0006, "sigma_max_m": 0.00029, "dinh_dinh_max_m": 0.0011},
       "A3": {"max_deg": 0.15, "rms_deg": 0.074, "n_cap": 24},
       "A5": {"lech_max_deg": 0.12, "sigma_max_deg": 0.04},
       "day_doi": {"dn_m": 0.016, "dzc_m": 0.01, "kappa": 0.12 / 2.6},
       "do_cao": {"rho_max_m": 1.5, "H_m": 2.6, "h_m": 0.12},
       "bo_cham": {"tag_offset": None, "side_m": 0.0964}}
CAM_PATHS = ('P1.bo_nhom_max_m', 'P1.tung_khung_max_m', 'moc_dich.nguong_m', 'moc_dich.u_so_m', 'P3.vi_tri_max_m',
             'A2.goc_max_deg', 'P7.lech_max_m', 'A3.max_deg', 'A5.lech_max_deg', 'day_doi.dn_m', 'day_doi.dzc_m',
             'day_doi.kappa', 'do_cao.rho_max_m', 'do_cao.H_m', 'do_cao.h_m', 'bo_cham.side_m')
TK = {"version": "c8b_thuoc_kep_v1", "cao_4_goc_mm": [120.2, 120.0, 119.8, 120.1], "e_mm": 0.1,
      "goc_dan": {"a_mm": 10.3, "b_mm": 10.1, "L_mm": 96.4, "canh": "trai"}, "tam_than_deg": 0.05}
XO = {"version": "c8b_xoay_v1", "q_tag_m": [0.003, -0.004], "du_m": 0.0007}
TBD = dict(ga_ok=True, d_p_quay_m=0.0007, d_psi_quay_deg=0.25,
           B=dict(shutter=dict(use_alpha=True, alpha_r_s_px=7e-6, alpha_c_s_px=1e-7, alpha_r_le_s_px=8.75e-6,
                               alpha_c_le_s_px=1.25e-7, mot_khung_s=1 / 60)))


def _cp(d):
    return json.loads(json.dumps(d))


def KH(cam=None, tk=None, xo=None, tb=None, **kw):
    return CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, tb or TBD, CAM if cam is None else cam,
                   TK if tk is None else tk, XO if xo is None else xo, **kw)


def gio_d(task, rate, width, gmr, ok=True, half=0.007):
    return dict(format='c8b_gio_v1', task=task, ok=ok, rate_luot=rate, width_s=width, gyro_minus_rv_s=gmr,
                half_s=half)


def _expect(cam=CAM, tk=TK, xo=XO):
    H, h, rho, s = cam['do_cao']['H_m'], cam['do_cao']['h_m'], cam['do_cao']['rho_max_m'], cam['bo_cham']['side_m']
    c = np.array(tk['cao_4_goc_mm']) * 1e-3
    e = tk['e_mm'] * 1e-3
    dh = np.max(np.abs(c - h)) + e
    cao = (rho + s / math.sqrt(2)) * dh / (H - h - dh)
    dn, dzc = cam['day_doi']['dn_m'], cam['day_doi']['dzc_m']
    dd = h / H * dn + h * dzc * (rho + dn) / (H * (H - dzc - h))
    g = tk['goc_dan']
    a1 = math.degrees(math.atan(abs(g['a_mm'] - g['b_mm']) / g['L_mm']) + math.atan(2 * tk['e_mm'] / g['L_mm'])) \
        + tk['tam_than_deg']
    q = math.hypot(*xo['q_tag_m'])
    p2 = q + xo['du_m'] + q * math.radians(a1)
    a6 = math.degrees(rho * (np.sum(np.abs(c - c.mean())) + 8 * e) / (2 * math.sqrt(2) * s * (H - c.max() - e)))
    p = dict(P1_anh_san=0.002 + 0.001 + 2 * 0.0004, P2_lech_tam=p2, P3_nhac_tam=0.0008, P4_quay=0.0007, P5_day_doi_zc=dd,
             P6_do_cao=cao, P7_nhieu_khung=0.0006)
    a = dict(A1_goc_dan=a1, A2_nhac_tam=0.1, A3_huong_vung=0.15, A4_quay=0.25, A5_nhieu_khung=0.12, A6_nghieng_ma=a6)
    return p, a


def test_khoa_sums():
    k = CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.010, 0.005), 'b': gio_d('O1-F05', 4e-4, 0.012, 0.003),
                 'c': gio_d('O1-F08', 1e-4, 0.014, 0.004), 'd': gio_d('O1-F08', 9e-3, 0.5, 0.5, ok=False)},
                TBD, CAM, TK, XO)
    p, a = _expect()
    assert k['U_p_phan_m'] == pytest.approx(p) and k['U_psi_phan_deg'] == pytest.approx(a)
    assert k['U_p_m'] == pytest.approx(1.25 * sum(p.values()))
    assert k['U_psi_deg'] == pytest.approx(1.25 * sum(a.values()))
    # hand numbers: P1 = 2 + 1 + 2 x 0.4 = 3.8 mm; A1 = 2 atan(0.2 / 96.4) + 0.05 = 0.2877 deg; dh = 0.3 mm
    assert p['P1_anh_san'] == pytest.approx(0.0038) and a['A1_goc_dan'] == pytest.approx(0.28773, abs=1e-4)
    assert tuple(k['U_p_phan_m']) == CU.P_PARTS and tuple(k['U_psi_phan_deg']) == CU.A_PARTS
    assert k['troi_ppm'] == pytest.approx(1.25 * 4e-4 * 1e6)             # run d (not ok) left out
    assert k['dai']['O1-F05']['width_s'] == [0.010, 0.012] and k['dai']['O1-F05']['gyro_minus_rv_s'] == [0.003, 0.005]
    assert k['man_trap']['alpha_r_s_px'] == pytest.approx(8.75e-6) and k['man_trap_kieu'] == 'alpha'


def test_khoa_p1_takes_the_larger_of_clip_mean_and_per_frame():
    cam = _cp(CAM)
    cam['P1']['tung_khung_max_m'] = 0.0031
    assert KH(cam)['U_p_phan_m']['P1_anh_san'] == pytest.approx(0.0031 + 0.0018)


def test_khoa_refuses_bad_3b_and_missing_field():
    with pytest.raises(CU.UError, match='ga_ok'):
        KH(tb=dict(TBD, ga_ok=False))
    for path in CAM_PATHS:                                  # every number used is required, none defaults to 0
        bad = _cp(CAM)
        blk, fld = path.split('.')
        del bad[blk][fld]
        with pytest.raises(CU.UError, match=path.replace('.', r'\.')):
            KH(bad)
        bad[blk][fld] = -1e-6 if fld != 'kappa' else None
        with pytest.raises(CU.UError, match=path.replace('.', r'\.')):
            KH(bad)
    bad = _cp(CAM)
    del bad['bo_cham']['tag_offset']
    with pytest.raises(CU.UError, match='bo_cham.tag_offset'):
        KH(bad)
    bad['bo_cham']['tag_offset'] = [0.0, 0.0]
    with pytest.raises(CU.UError, match='bo_cham.tag_offset'):
        KH(bad)


def test_khoa_camera_input_names_only():
    # Liam 18/11 at CODE_C8B: names by page v10; a file with the names of tests 1, 2 (b740f2a) is refused
    old = {"phep1": {"loi_khung_max_m": 0.002}, "dat_ma_kiem_m": 0.0,
           "phep2": {"du_tam_m": 0.0015, "du_huong_deg": 0.2, "chenh_tham_so_m": 0.0005, "huong_chuan_deg": 0.3},
           "thao_dat": {"vi_tri_m": 0.0008, "goc_deg": 0.1}, "do_cao": CAM['do_cao'], "day_doi": CAM['day_doi']}
    with pytest.raises(CU.UError, match='version is not c8_cam_u_vao_v2'):
        KH(old)
    with pytest.raises(CU.UError, match='unknown top-level fields'):
        KH(dict(old, **{k: v for k, v in CAM.items() if k not in old}))
    with pytest.raises(CU.UError, match=r"unknown top-level fields \['A6'\]"):
        KH(dict(CAM, A6={'max_deg': 0.15}))
    assert CU.check_cam(_cp(CAM)) is None                   # the camera file alone (for its writers' tests)
    with pytest.raises(CU.UError, match='version is not'):
        CU.check_cam(old)
    info = _cp(CAM)                                        # extra keys inside a block are information, kept out of U
    info['A3'].update(L_m=[0.55, 0.78], cach='bo_cap')
    info['nguon'], info['ghi_chu'] = ['x.json'], 'thu'
    assert KH(info)['U_psi_deg'] == KH()['U_psi_deg']


def test_khoa_per_frame_noise_inside_margin():
    # O1 v11.39 J1: U_p, U_psi bound one frame: P7, A5 are parts of the sum and take the margin like every part
    z = _cp(CAM)
    z['P7']['lech_max_m'], z['A5']['lech_max_deg'] = 0.0, 0.0
    k0, k1 = KH(z), KH()
    assert k1['U_p_m'] - k0['U_p_m'] == pytest.approx(1.25 * 0.0006)
    assert k1['U_psi_deg'] - k0['U_psi_deg'] == pytest.approx(1.25 * 0.12)


def test_khoa_caliper_input():
    for k_, v in (('version', 'x'), ('cao_4_goc_mm', [120.0, 120.0, 120.0]), ('e_mm', -0.1),
                  ('tam_than_deg', None), ('goc_dan', None)):
        bad = _cp(TK)
        bad[k_] = v
        with pytest.raises(CU.UError):
            KH(tk=bad)
    bad = _cp(TK)
    bad['goc_dan']['L_mm'] = 0.0
    with pytest.raises(CU.UError, match='L_mm'):
        KH(tk=bad)
    bad = _cp(TK)
    bad['tam_than_deg'] = 0.0                              # 0 only with the limit written
    with pytest.raises(CU.UError, match='tam_than_gioi_han'):
        KH(tk=bad)
    bad['tam_than_gioi_han'] = 'sai che tao tam-than ngoai ban ve chua bao'
    assert KH(tk=bad)['U_psi_phan_deg']['A1_goc_dan'] == pytest.approx(_expect(tk=bad)[1]['A1_goc_dan'])
    bad = _cp(TK)
    bad['cao_4_goc_mm'] = [2700.0, 120.0, 120.0, 120.0]   # above the camera
    with pytest.raises(CU.UError, match='not below the camera'):
        KH(tk=bad)


def test_khoa_offset_used_by_the_scorer():
    # the scorer corrects by tag_offset: P2, A1 keep only what is left; a dpsi needs the signed caliper angle
    g = math.atan(0.2 / 96.4)
    q = XO['q_tag_m']
    cam = _cp(CAM)
    c, s = math.cos(g), math.sin(g)
    cam['bo_cham']['tag_offset'] = [c * q[0] - s * q[1], s * q[0] + c * q[1], g]
    with pytest.raises(CU.UError, match='goc_dan_deg'):
        KH(cam)
    tk = dict(_cp(TK), goc_dan_deg=-math.degrees(g))
    k_bad = KH(cam, tk=tk)                                  # opposite sign: the mismatch 2 g is counted, not hidden
    tk = dict(_cp(TK), goc_dan_deg=math.degrees(g))
    k = KH(cam, tk=tk)
    a1 = math.degrees(math.atan(2 * 0.1 / 96.4)) + 0.05
    assert k['U_psi_phan_deg']['A1_goc_dan'] == pytest.approx(a1)
    assert k_bad['U_psi_phan_deg']['A1_goc_dan'] == pytest.approx(a1 + 2 * math.degrees(g))
    assert k['U_p_phan_m']['P2_lech_tam'] == pytest.approx(0.0007 + 0.005 * math.radians(a1))
    bad = dict(_cp(TK), goc_dan_deg=0.5)                   # size not atan(|a - b| / L)
    with pytest.raises(CU.UError, match='goc_dan_deg'):
        KH(cam, tk=bad)
    cam['bo_cham']['tag_offset'] = [0.0, 0.0, 0.0]          # offset given as zero = none: the whole q counts
    assert KH(cam)['U_p_phan_m']['P2_lech_tam'] == pytest.approx(_expect()[0]['P2_lech_tam'])


def test_khoa_turning_input():
    for bad in (dict(XO, version='x'), dict(XO, q_tag_m=[0.003]), dict(XO, du_m=-1.0), dict(XO, q_tag_m=[0.003, None])):
        with pytest.raises(CU.UError):
            KH(xo=bad)


class _FloorCam0:
    def pix2floor(self, pts, z):
        return np.asarray(pts, float)


def test_tilt_parts_bound_car_pose():
    # A6 and P6 checked against c7_measure.car_pose with four corner heights (not a copy of the formula)
    import c7_measure as CM
    rng = np.random.default_rng(11)
    side, H = 0.0964, 2.632
    worst = 0.0
    for _ in range(400):
        h = rng.uniform(0.08, 0.16)
        c = h + rng.uniform(-0.0008, 0.0008, 4)
        rho = rng.uniform(0.1, 1.8)
        a = rng.uniform(0, 2 * math.pi)
        P = np.array([rho * math.cos(a), rho * math.sin(a)])
        psi = rng.uniform(-math.pi, math.pi)
        cs, sn = math.cos(psi), math.sin(psi)
        corners = P + CM.car_corners_body(side) @ np.array([[cs, sn], [-sn, cs]])
        p = corners * (H / (H - c))[:, None]                 # central projection from (0, 0, H) of each corner
        r = CM.car_pose(_FloorCam0(), p, side, h / H, (0.0, 0.0))
        err_psi = abs((r['tag_psi'] - psi + math.pi) % (2 * math.pi) - math.pi)
        err_p = math.hypot(r['tag_x'] - P[0], r['tag_y'] - P[1])
        cam = _cp(CAM)
        cam['do_cao'].update(H_m=H, h_m=h, rho_max_m=rho)
        cam['day_doi'] = dict(dn_m=0.0, dzc_m=0.0, kappa=h / H)
        tk = dict(_cp(TK), cao_4_goc_mm=list(c * 1e3), e_mm=0.0)
        _, pp, _, aa = CU.u_p_u_psi(cam, TBD, tk, XO)
        assert err_psi <= math.radians(aa['A6_nghieng_ma']) * (1 + 1e-6) + 1e-12
        assert err_p <= pp['P6_do_cao'] * (1 + 1e-6) + 1e-12
        worst = max(worst, err_psi / math.radians(aa['A6_nghieng_ma']))
    assert worst > 0.3                                       # the bound is not loose by orders of magnitude


def test_khoa_plumb_zc_required_and_bounded():
    # plumb point, zc reused from 16/11 (Liam 17/11 evening): both bounds required, zc bound below the camera height
    for k_ in ('dn_m', 'dzc_m', 'kappa'):
        bad = json.loads(json.dumps(CAM))
        del bad['day_doi'][k_]
        with pytest.raises(CU.UError, match='day_doi.' + k_):
            CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, bad, TK, XO)
    bad = json.loads(json.dumps(CAM))
    bad['day_doi']['dn_m'] = -0.001
    with pytest.raises(CU.UError, match='day_doi.dn_m'):
        CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, bad, TK, XO)
    bad['day_doi'] = dict(dn_m=0.0, dzc_m=2.5, kappa=0.12 / 2.6)    # H - dzc not above h
    with pytest.raises(CU.UError, match='not above the tag height'):
        CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, bad, TK, XO)
    bad['day_doi'] = dict(dn_m=0.0, dzc_m=0.01, kappa=0.12 / 2.632)   # C8BS-R7-3: H is not the zc of the scoring
    with pytest.raises(CU.UError, match='not do_cao.h_m / do_cao.H_m'):
        CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, bad, TK, XO)
    z = json.loads(json.dumps(CAM))
    z['day_doi'] = dict(dn_m=0.0, dzc_m=0.0, kappa=0.12 / 2.6)     # no plumb / zc error: the part is zero
    k0 = CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, z, TK, XO)
    k1 = CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, CAM, TK, XO)
    assert k0['U_p_phan_m']['P5_day_doi_zc'] == 0.0
    assert k1['U_p_m'] - k0['U_p_m'] == pytest.approx(1.25 * k1['U_p_phan_m']['P5_day_doi_zc'])
    assert k1['U_psi_deg'] == k0['U_psi_deg']                       # a shift and a scale about C keep every angle


class _FloorCam:
    """pix2floor of a camera whose 'pixels' are already floor points (the parallax step of car_pose alone)."""
    def pix2floor(self, pts, z):
        return np.asarray(pts, float)


def _plumb_case(H, h, dn, dzc, rho, Cs, Hs, P, psi, side=0.1):
    """car_pose tag-centre error with the session's C = (0, 0), zc = H, when the camera centre is truly over Cs at
    Hs and the tag centre is at P (height h); and the c8b_u part for the bounds dn, dzc, rho."""
    import c7_measure as CM
    c, s = math.cos(psi), math.sin(psi)
    body = CM.car_corners_body(side)
    Pc = np.asarray(P, float) + body @ np.array([[c, s], [-s, c]])
    Cs = np.asarray(Cs, float)
    p = Cs + (Pc - Cs) * Hs / (Hs - h)                                # central projection onto the floor
    r = CM.car_pose(_FloorCam(), p, side, h / H, (0.0, 0.0))
    err = math.hypot(r['tag_x'] - P[0], r['tag_y'] - P[1])
    cam = json.loads(json.dumps(CAM))
    cam['do_cao'].update(H_m=H, h_m=h, rho_max_m=rho)
    cam['day_doi'] = dict(dn_m=dn, dzc_m=dzc, kappa=h / H)
    _, parts, _, _ = CU.u_p_u_psi(cam, TBD, TK, XO)
    return err, parts['P5_day_doi_zc']


def test_plumb_zc_part_bounds_car_pose():
    # C8BS-R7-3: checked against the parallax step of c7_measure.car_pose, not a copy of the formula
    err, b = _plumb_case(2.6, 1.0, 0.0, 0.01, 1.5, (0, 0), 2.59, (1.5, 0.0), 0.3)   # Codex counter-example
    assert err == pytest.approx(0.003628, abs=2e-6) and err <= b * (1 + 1e-9)
    err, b = _plumb_case(2.632, 0.12, 0.0, 0.02, 1.5, (0, 0), 2.612, (1.5, 0.0), 0.0)
    assert err <= b * (1 + 1e-9) and err >= 0.99 * b                  # the worst sign of dzc reaches the bound
    rng = np.random.default_rng(7)
    for _ in range(300):
        H, h = 2.632, rng.uniform(0.05, 0.5)
        dn, dzc, rho = rng.uniform(0, 0.03), rng.uniform(0, 0.03), rng.uniform(0.1, 2.0)
        a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0, dn)
        Cs = (rr * math.cos(a), rr * math.sin(a))
        a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0, rho)
        err, b = _plumb_case(H, h, dn, dzc, rho, Cs, H + rng.uniform(-dzc, dzc), (rr * math.cos(a), rr * math.sin(a)),
                             rng.uniform(-math.pi, math.pi))
        assert err <= b * (1 + 1e-9) + 1e-12


def test_cong_refuses_v1_lock():
    # C8BS-R7-4: a lock written before the plumb / zc part (c8b_khoa_v1, cc22feb) cannot make the gate file
    k = CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, CAM, TK, XO)
    CU.check_khoa(json.loads(json.dumps(CU._clean(k))))
    for ver in ('c8b_khoa_v1', 'c8b_khoa_v2'):              # v2: parts named after tests 1, 2 (b740f2a)
        with pytest.raises(CU.UError, match='parts P1-P7, A1-A6'):
            CU.check_khoa(dict(k, version=ver))
    old = dict(k, U_p_phan_m={x: v for x, v in k['U_p_phan_m'].items() if x != 'P5_day_doi_zc'})
    with pytest.raises(CU.UError, match='parts P1-P7, A1-A6'):
        CU.check_khoa(old)
    old = dict(k, U_psi_phan_deg={x: v for x, v in k['U_psi_phan_deg'].items() if x != 'A5_nhieu_khung'})
    with pytest.raises(CU.UError, match='parts P1-P7, A1-A6'):
        CU.check_khoa(old)
    old = dict(k, U_p_phan_m=dict(k['U_p_phan_m'], phep1_khung_max=0.002))
    with pytest.raises(CU.UError, match='parts P1-P7, A1-A6'):
        CU.check_khoa(old)


def test_khoa_phep5_and_one_frame():
    tb = json.loads(json.dumps(TBD))
    tb['B']['shutter']['use_alpha'] = False
    k = CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, tb, CAM, TK, XO, phep5=[('d1', 0.003), ('d2', -0.02)])
    assert k['man_trap_kieu'] == 'mot_khung' and k['man_trap'] is None
    assert k['phep5']['ket_qua'] == 'xem_lai' and k['phep5']['vuot'] == ['d2']


def _plan():
    return {'1': dict(task='O1-F05', controller='PID'), '2': dict(task='O1-F05', controller='LADRC'),
            '3': dict(task='O1-F08', controller='PID'), '4': dict(task='O1-F08', controller='LADRC')}


META = dict(nhanh_tick='bo_tick', commit_hil='abc1234')
SCRIPT = dict(commit='abc1234ffff', dirty=False)
SCORER = dict(commit='abc1234ffff', dirty=False, nut=[dict(commit='abc1234ffff', dirty=False, file='tools/c7_score.py',
                                                            tracked=True)])


def _pair(rid, task, half, outcome, ut, nodes=None, gio_ok=True, ctrl='PID'):
    nodes = nodes or [1.0 + 0.001 * i for i in range(len(ut) or 1)]
    g = dict(gio_d(task, 1e-4, 0.01, 0.0, ok=gio_ok, half=half), nodes_s=nodes, script=SCRIPT,
             ket_luan='ok' if gio_ok else 'hong_he_do', controller=ctrl)
    sha = f'{rid:0<64}'[:64]
    l = dict(version='c8b_luoi_v1', gio_sha256=sha, task=task, nodes_s=nodes, report_only=[], outcome=outcome,
             u_t_nodes=ut, script=SCRIPT, scorer=SCORER, nhanh_tick=['bo'])
    return g, l, sha


def _luot(lost_first=False):
    out = []
    if lost_first:
        g, l, s = _pair('r3a', 'O1-F08', 0.009, 'hong_he_do', [], gio_ok=True)
        l['nhanh_tick'] = []
        out.append(('3', 'r3a', g, l, s))
    for slot, rid, task, half, outc, ut in (('1', 'r1', 'O1-F05', 0.006, 'dat', [0.020, 0.021]),
                                            ('2', 'r2', 'O1-F05', 0.008, 'dat', [0.020, 0.021]),
                                            ('3', 'r3', 'O1-F08', 0.009, 'dat', [0.020, 0.021]),
                                            ('4', 'r4', 'O1-F08', 0.007, 'khong_dat', [0.03])):
        g, l, s = _pair(rid, task, half, outc, ut, ctrl=_plan()[slot]['controller'])
        out.append((slot, rid, g, l, s))
    return out


K = dict(U_p_m=0.004, U_psi_deg=2.0)


def test_cong_rule_2b():
    import gate_c8
    import o1_gate
    d = CU.cong(K, _plan(), _luot(lost_first=True), ['O1-F05', 'O1-F08'], 'bo_tick', 'C8/x', meta=META)
    p8 = d['dieu_2b']['per_level']['O1-F08']
    assert p8['u_t_c8_s'] == pytest.approx(max(0.021 + 0.009, 0.03 + 0.007))
    v, wz = gate_c8.schedule_peaks('O1-F08')                    # peaks without the margin (CODE_GATE GC1-3)
    assert (p8['v_cmd_max'], p8['wz_cmd_max']) == (v, wz)
    assert p8['uhat']['g3'] == pytest.approx(0.004 + p8['u_t_c8_s'] * v)
    assert p8['uhat']['g5'] == pytest.approx(2.0 + math.degrees(p8['u_t_c8_s'] * wz))
    assert p8['report_only'] == [c for c in ('g2', 'g3', 'g5') if p8['uhat'][c] > 0.5 * o1_gate.THRESHOLDS[c]]
    assert [r['run_id'] for r in p8['runs']] == ['r3', 'r4']
    lu = d['camera']['c8b']['u_t_luot_c8']
    assert lu['r3']['h_s'] == pytest.approx(0.009) and lu['r3']['u_max_s'] == pytest.approx(0.021)
    assert d['format'] == 'gate_c8_v1' and d['u_t_branch'] == 'bo_tick'
    gate_c8.check(d)                                             # the real contract of the gate file


def test_cong_written_and_read_back_with_real_reader(tmp_path):
    k = dict(K, version=CU.VERSION)
    d = CU.cong(k, _plan(), _luot(), ['O1-F05', 'O1-F08'], 'bo_tick', 'C8/x', meta=META,
                written_by=dict(script='tools/c8b_u.py', repo='hil-framework', commit='abc1234', dirty=False))
    g = CU.write_gate(str(tmp_path / 'gate_c8.json'), CU._clean(d))
    assert g.levels == ('O1-F05', 'O1-F08') or list(g.levels) == ['O1-F05', 'O1-F08']
    assert tuple(g.report_only('O1-F08')) == tuple(d['dieu_2b']['per_level']['O1-F08']['report_only'])
    with pytest.raises(CU.UError, match='written once'):
        CU.write_gate(str(tmp_path / 'gate_c8.json'), CU._clean(d))
    (tmp_path / 'claimed.json').write_bytes(b'')                     # C8BS-R3-1: another writer holds the path
    with pytest.raises(CU.UError, match='written once'):
        CU.write_gate(str(tmp_path / 'claimed.json'), CU._clean(d))
    assert (tmp_path / 'claimed.json').read_bytes() == b''
    bad = json.loads(json.dumps(CU._clean(d)))
    bad['dieu_2b']['per_level']['O1-F08']['report_only'] = []    # contradicts uhat: the reader refuses, no file
    if d['dieu_2b']['per_level']['O1-F08']['report_only']:
        with pytest.raises(CU.UError, match='refused'):
            CU.write_gate(str(tmp_path / 'g2.json'), bad)
        assert not (tmp_path / 'g2.json').exists()


def test_cong_strict_half():
    import gate_c8
    assert gate_c8.uhat_of(0.05, 1.0, 0.0, 1.0, 1.0)['g2'] == 0.05
    lu = [(s, r, dict(g, half_s=0.0), dict(l, u_t_nodes=[0.001] * len(l['u_t_nodes'])), sh)
          for s, r, g, l, sh in _luot()]
    d = CU.cong(dict(U_p_m=0.05, U_psi_deg=1.0), _plan(), lu, ['O1-F05'], 'bo_tick', 'x', meta=META)
    assert 'g2' not in d['dieu_2b']['per_level']['O1-F05']['report_only']      # 0.05 = half of D: still decides


def test_cong_needs_both_controllers_and_rerun_rule():
    lu = [x for x in _luot() if x[0] != '2']
    with pytest.raises(CU.UError, match='no run'):
        CU.cong(K, _plan(), lu, ['O1-F05'], 'bo_tick', 'x', meta=META)
    g, l, s = _pair('r1b', 'O1-F05', 0.006, 'dat', [0.02, 0.02])
    with pytest.raises(CU.UError, match='rerun only replaces'):
        CU.cong(K, _plan(), _luot() + [('1', 'r1b', g, l, s)], ['O1-F05'], 'bo_tick', 'x', meta=META)
    g, l, s = _pair('r9', 'O1-F05', 0.006, 'dat', [0.02, 0.02])
    l = dict(l)
    with pytest.raises(CU.UError, match='not in the C8 plan'):
        CU.cong(K, _plan(), _luot() + [('9', 'r9', g, l, s)], ['O1-F05'], 'bo_tick', 'x', meta=META)
    lu = [(sl, r, g, dict(l, u_t_nodes=[math.inf, 0.02]) if r == 'r2' else l, sh) for sl, r, g, l, sh in _luot()]
    with pytest.raises(CU.UError, match='each controller'):
        CU.cong(K, _plan(), lu, ['O1-F05'], 'bo_tick', 'x', meta=META)


def test_cong_refuses_run_twice_and_foreign_grid():
    lu = _luot()
    dup = [lu[0], ('2', lu[0][1], dict(lu[0][2], controller='LADRC'), lu[0][3], lu[0][4])] + lu[2:]   # one run, 2 slots
    with pytest.raises(CU.UError, match='twice'):
        CU.cong(K, _plan(), dup, ['O1-F05'], 'bo_tick', 'x', meta=META)
    s0, r0, g0, l0, sh0 = lu[0]
    with pytest.raises(CU.UError, match='another clock file'):
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, gio_sha256='f' * 64), sh0)] + lu[1:], ['O1-F05'], 'bo_tick', 'x',
                meta=META)
    with pytest.raises(CU.UError, match='nodes differ'):
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, nodes_s=[9.0]), sh0)] + lu[1:], ['O1-F05'], 'bo_tick', 'x',
                meta=META)
    with pytest.raises(CU.UError, match='all five deciding'):
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, report_only=['g5']), sh0)] + lu[1:], ['O1-F05'], 'bo_tick',
                'x', meta=META)


def test_cong_branch_and_commit_from_plan():
    lu = _luot()
    with pytest.raises(CU.UError, match='the C8 plan says'):
        CU.cong(K, _plan(), lu, ['O1-F05'], 'giu_tick', 'x', meta=META)
    with pytest.raises(CU.UError, match='needs commit_hil'):
        CU.cong(K, _plan(), lu, ['O1-F05'], 'bo_tick', 'x', meta=dict(nhanh_tick='bo_tick'))
    s0, r0, g0, l0, sh0 = lu[0]
    with pytest.raises(CU.UError, match='not from commit_hil'):
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, script=dict(commit='abc1234', dirty=True)), sh0)] + lu[1:],
                ['O1-F05'], 'bo_tick', 'x', meta=META)
    with pytest.raises(CU.UError, match='tick branch'):
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, nhanh_tick=['giu']), sh0)] + lu[1:], ['O1-F05'], 'bo_tick', 'x',
                meta=META)
    with pytest.raises(CU.UError, match='tick branch'):                 # no evidence of the branch: refused
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, nhanh_tick=[]), sh0)] + lu[1:], ['O1-F05'], 'bo_tick', 'x',
                meta=META)
    with pytest.raises(CU.UError, match='scorer not from commit_hil'):
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, scorer=dict(commit='abc1234', dirty=True)), sh0)] + lu[1:],
                ['O1-F05'], 'bo_tick', 'x', meta=META)


def test_cong_scorer_must_be_canonical_and_lost_run_needs_none():
    lu = _luot()
    s0, r0, g0, l0, sh0 = lu[0]
    other = dict(SCORER, nut=[dict(SCORER['nut'][0], file='tools/scorer_thu.py', tracked=False)])
    with pytest.raises(CU.UError, match='C8BS-16'):
        CU.cong(K, _plan(), [(s0, r0, g0, dict(l0, scorer=other), sh0)] + lu[1:], ['O1-F05'], 'bo_tick', 'x', meta=META)
    # C8BS-R4-1: slot 1 lost at the clock check (no node, scorer None), then a valid rerun: accepted
    g, l, s = _pair('r1x', 'O1-F05', 0.006, 'chua_ket_luan', [], gio_ok=False)
    l = dict(l, scorer=None, nhanh_tick=[], outcome='hong_he_do')
    d = CU.cong(K, _plan(), [('1', 'r1x', g, l, s)] + lu, ['O1-F05'], 'bo_tick', 'x', meta=META)
    assert [r['run_id'] for r in d['dieu_2b']['per_level']['O1-F05']['runs']] == ['r1', 'r2']


def test_cong_scorer_checked_when_every_node_lost():
    # C8BS-R5-1: clock ok, nodes scored by another commit, every node lost: the scorer is still checked
    lu = _luot()
    g, l, s = _pair('r1y', 'O1-F05', 0.006, 'hong_he_do', [0.02, 0.02])
    bad = dict(SCORER, commit='bbbbbbb', nut=[dict(SCORER['nut'][0], commit='bbbbbbb')])
    l = dict(l, scorer=bad, nhanh_tick=[])
    with pytest.raises(CU.UError, match='scorer not from commit_hil'):
        CU.cong(K, _plan(), [('1', 'r1y', g, l, s)] + lu, ['O1-F05'], 'bo_tick', 'x', meta=META)
    l = dict(l, scorer=SCORER)                       # same run scored by the planned commit: lost, rerun accepted
    d = CU.cong(K, _plan(), [('1', 'r1y', g, l, s)] + lu, ['O1-F05'], 'bo_tick', 'x', meta=META)
    assert [r['run_id'] for r in d['dieu_2b']['per_level']['O1-F05']['runs']] == ['r1', 'r2']


def test_repo_state_marks_untracked_scorer(tmp_path):
    p = os.path.join(HERE, '..', 'c8b_tmp_untracked_scorer.py')
    open(p, 'w').write('# temporary, untracked' + chr(10))
    try:
        st = G.repo_state(p)
        assert st['tracked'] is False and st['dirty'] is True and st['file'].endswith('c8b_tmp_untracked_scorer.py')
    finally:
        os.remove(p)
    st = G.repo_state(os.path.join(HERE, '..', 'c7_score.py'))
    assert st['file'] == 'tools/c7_score.py' and st['tracked'] is True


def test_cong_controller_from_log():
    lu = _luot()
    s1, r1, g1, l1, h1 = lu[1]                       # slot 2 is LADRC; its log says PID
    with pytest.raises(CU.UError, match='C8BS-14'):
        CU.cong(K, _plan(), [lu[0], (s1, r1, dict(g1, controller='PID'), l1, h1)] + lu[2:], ['O1-F05'], 'bo_tick',
                'x', meta=META)
    with pytest.raises(CU.UError, match='C8BS-14'):
        CU.cong(K, _plan(), [lu[0], (s1, r1, dict(g1, controller=None), l1, h1)] + lu[2:], ['O1-F05'], 'bo_tick',
                'x', meta=META)


def test_read_plan_meta(tmp_path):
    p = tmp_path / 'ke_hoach.csv'
    p.write_text('# nhanh_tick=bo_tick\n# commit_hil=abc1234\nslot,task,controller\n1,O1-F05,PID\n2,O1-F05,LADRC\n',
                 encoding='utf-8')
    plan, meta = CU.read_plan(str(p))
    assert meta == META and plan['2'] == dict(task='O1-F05', controller='LADRC')


def test_write_gate_needs_reader(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'gate_c8', None)
    with pytest.raises(CU.UError, match='gate_c8'):
        CU.write_gate(str(tmp_path / 'g.json'), {})
    assert not (tmp_path / 'g.json').exists()
