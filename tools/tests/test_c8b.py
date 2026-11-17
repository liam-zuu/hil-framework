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
    n = G.nodes(1.0, 0.0105)
    assert len(n) == 22 and max(np.diff(n)) <= 0.001 + 1e-12


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
    assert not out['ok'] and out['ket_luan'] == 'chua_ket_luan'
    assert any('> U~' in x for x in out['ly_do']) and out['nodes_s'] == []


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
    assert out2['ket_luan'] == 'chua_ket_luan' and any('C8 range' in x for x in out2['ly_do_chua'])
    khoa['dai']['O1-F05']['width_s'] = [w - 0.001, w + 0.001]
    out3, _, _ = G.analyse(base_run[1], base_run[2], 'O1-F05', khoa)
    assert out3['ok']


def test_startup_consistent_and_inconsistent(base_run):
    out = base_run[0]
    st = out['khoi_dong']
    assert st['ket_qua'] in ('nhat_quan', 'khong_xac_dinh')
    if st['lo_s'] is not None:
        assert st['lo_s'] <= out['center_s'] + out['half_s'] + 1e-9


def test_startup_lower_estimate_flags_far_offset(tmp_path):
    # camera clock shifted 80 ms after the matching: the start-up lower estimate must disagree with a forced interval
    fp, lp = make_run(str(tmp_path))
    F = G.read_frames(fp)
    rows, _ = G.read_log(lp)
    tl = np.array([float(r['timestamp_us']) for r in rows]) * 1e-6
    tm, rate, br, _, _ = G.count_pairs(rows, tl)
    tr, yr, _ = G.imu_reports(rows, tl, 'rv_n', 'age_r_ms', 'yaw_rad')
    t0 = 2.9003
    st = G.startup(F, t0, 1.234, tm, G.fk(rate), tr, yr)
    assert st['lo_s'] is not None and st['lo_s'] <= 1.234 + 0.01
    assert st['lo_s'] > 1.234 - 0.2


def test_cli_writes_once(tmp_path, base_run):
    out_p = str(tmp_path / 'gio.json')
    args = ['--frames', base_run[1], '--esp32', base_run[2], '--task', 'O1-F05', '--out', out_p]
    assert G.main(args) == 0
    d = json.load(open(out_p, encoding='utf-8'))
    assert d['format'] == 'c8b_gio_v1' and len(d['nguon']) == 2
    assert G.main(args) == 2                                   # never overwritten


def test_sealed_path_refused(tmp_path, base_run):
    p = tmp_path / 'sealed'
    p.mkdir()
    assert G.main(['--frames', base_run[1], '--esp32', base_run[2], '--task', 'O1-F05',
                   '--out', str(p / 'g.json')]) == 2


# ── c8b_luoi ─────────────────────────────────────────────────────────────────────────────────

def node(states, rows=None, nums=None, ok=True, u_t=0.03):
    if not ok:
        return dict(ok=False, why='lost')
    st = dict(zip(LU.CONDS, states)) if isinstance(states, (list, tuple)) else states
    nm = nums or {c: dict(q=0.05, lo=0.04, hi=0.06, n_undef=0, q_chung=0.05, lo_chung=0.04, hi_chung=0.06,
                          n_undef_chung=0) for c in LU.MEASURED}
    return dict(ok=True, outcome='x', states=st, nums=nm, rows=rows or {}, u_t_s=u_t)


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
    assert not res['bieu_dien']['g2']
    res2 = LU.combine([node(None, ok=False)] * 3)
    assert res2['outcome'] == 'hong_he_do'


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
    for c, s in zip(LU.MEASURED, states[1:]):
        g[c] = dict(state=s, q=0.05, lo=0.04, hi=0.06, n_undef=0, chung=dict(q=0.05, lo=0.04, hi=0.06, n_undef=0))
    with open(os.path.join(d, f'nut_{k:03d}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(outcome='dat', gate_v3=g, U_t_s=u_t), f)
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
    for k in range(2):
        _write_node(d, k, ['dat', 'dat', 'khong_dat', 'dat', 'dat'],
                    [(10.000001, 1, [0.05, 0.12, 0.05, 1.0]), (10.016668, 1, [0.05, 0.05, 0.05, 1.0])])
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['states']['g3'] == 'khong_dat' and r['outcome'] == 'khong_dat' and r['u_t_nodes'] == [0.031, 0.031]


def test_node_rows_without_c8b_columns_is_error(tmp_path):
    sp = tmp_path / 's.json'
    json.dump(dict(outcome='dat', gate_v3={c: dict(state='dat') for c in LU.CONDS}), open(sp, 'w'))
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
st = 'khong_dat' if k < 3 else 'chua_ket_luan'
gate = {'g1': {'state': 'dat'}}
for c in ('g2', 'g3', 'g4', 'g5'):
    s = st if c == 'g3' else 'dat'
    gate[c] = dict(state=s, q=0.05, lo=0.04, hi=0.06, n_undef=0, chung=dict(q=0.05, lo=0.04, hi=0.06, n_undef=0))
json.dump(dict(outcome='khong_dat', gate_v3=gate, U_t_s=0.02 + 0.001 * k), open(g('--out'), 'w'))
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
    assert r['n_nodes'] == 4 and r['states']['g3'] == 'chua_ket_luan'       # last cell: frame not in both domains
    assert r['u_t_nodes'] == pytest.approx([0.02, 0.021, 0.022, 0.023])


# ── c8b_3b ───────────────────────────────────────────────────────────────────────────────────

def make_wheel(path, R, alpha_r=7.7e-6, wob=0.002, noise=0.0002, seed=1, drift_centre=0.0):
    rng = np.random.default_rng(seed)
    T = 1 / 60
    tt = np.arange(0, 70, T)
    om = np.where(tt < 34, 6.0 * np.exp(-tt / 12) + 0.3, np.where(tt < 35, 0.0, -(6.0 * np.exp(-(tt - 35) / 12) + 0.3)))
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
                        psi=math.remainder(psi, 2 * math.pi), row_px=row, col_px=col))
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
    if sh['use_alpha']:
        assert sh['alpha_r_le_s_px'] == pytest.approx(1.25 * sh['alpha_r_s_px'])


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


def test_3b_window_over_turnaround_not_used():
    T = 1 / 60
    tt = np.arange(0, 80, T)
    om = np.where(tt < 40, 6.0 * np.exp(-tt / 15) + 0.3, -(6.0 * np.exp(-(tt - 40) / 15) + 0.3))
    F = dict(t=tt, x=np.zeros_like(tt), y=np.zeros_like(tt), psi=np.cumsum(om) * T)
    assert TB.analyse_a(F, 2.614)['psi_osc_deg'] < 0.01


# ── c8b_u ────────────────────────────────────────────────────────────────────────────────────

CAM = {"phep1": {"loi_khung_max_m": 0.002}, "dat_ma_kiem_m": 0.001,
       "phep2": {"du_tam_m": 0.0015, "du_huong_deg": 0.2, "chenh_tham_so_m": 0.0005, "huong_chuan_deg": 0.3},
       "thao_dat": {"vi_tri_m": 0.0008, "goc_deg": 0.1}, "do_cao": {"rho_max_m": 1.5, "dh_m": 0.002, "H_m": 2.6,
                                                                    "h_m": 0.12}}
TBD = dict(ga_ok=True, d_p_quay_m=0.0007, d_psi_quay_deg=0.25,
           B=dict(shutter=dict(use_alpha=True, alpha_r_s_px=7e-6, alpha_c_s_px=1e-7, alpha_r_le_s_px=8.75e-6,
                               alpha_c_le_s_px=1.25e-7, mot_khung_s=1 / 60)))


def gio_d(task, rate, width, gmr, ok=True, half=0.007):
    return dict(format='c8b_gio_v1', task=task, ok=ok, rate_luot=rate, width_s=width, gyro_minus_rv_s=gmr,
                half_s=half)


def test_khoa_sums():
    k = CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.010, 0.005), 'b': gio_d('O1-F05', 4e-4, 0.012, 0.003),
                 'c': gio_d('O1-F08', 1e-4, 0.014, 0.004), 'd': gio_d('O1-F08', 9e-3, 0.5, 0.5, ok=False)}, TBD, CAM)
    cao = 1.5 * 0.002 / (2.6 - 0.12)
    assert k['U_p_m'] == pytest.approx(1.25 * (0.002 + 0.001 + 0.0015 + 0.0005 + 0.0008 + cao + 0.0007))
    assert k['U_psi_deg'] == pytest.approx(1.25 * (0.2 + 0.3 + 0.1 + 0.25))
    assert k['troi_rate'] == pytest.approx(1.25 * 4e-4)                  # run d (not ok) left out
    assert k['dai']['O1-F05']['width_s'] == [0.010, 0.012] and k['dai']['O1-F05']['gyro_minus_rv_s'] == [0.003, 0.005]
    assert k['alpha']['alpha_r_s_px'] == pytest.approx(8.75e-6) and k['man_trap'] == 'alpha'


def test_khoa_refuses_bad_3b_and_missing_field():
    with pytest.raises(CU.UError, match='ga_ok'):
        CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, dict(TBD, ga_ok=False), CAM)
    bad = json.loads(json.dumps(CAM))
    del bad['phep2']['du_tam_m']
    with pytest.raises(CU.UError, match='phep2.du_tam_m'):
        CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, bad)


def test_khoa_phep5_and_one_frame():
    tb = json.loads(json.dumps(TBD))
    tb['B']['shutter']['use_alpha'] = False
    k = CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, tb, CAM, phep5=[('d1', 0.003), ('d2', -0.02)])
    assert k['man_trap'] == 'mot_khung' and k['alpha'] is None
    assert k['phep5']['ket_qua'] == 'xem_lai' and k['phep5']['vuot'] == ['d2']


def _plan():
    return {'1': dict(task='O1-F05', controller='PID'), '2': dict(task='O1-F05', controller='LADRC'),
            '3': dict(task='O1-F08', controller='PID'), '4': dict(task='O1-F08', controller='LADRC')}


def _luot(lost_first=False):
    ok = dict(outcome='dat', u_t_nodes=[0.020, 0.021])
    out = []
    if lost_first:
        out.append(('3', 'r3a', gio_d('O1-F08', 1e-4, 0.01, 0.0), dict(outcome='hong_he_do', u_t_nodes=[])))
    out += [('1', 'r1', gio_d('O1-F05', 1e-4, 0.01, 0.0, half=0.006), ok),
            ('2', 'r2', gio_d('O1-F05', 1e-4, 0.01, 0.0, half=0.008), ok),
            ('3', 'r3', gio_d('O1-F08', 1e-4, 0.01, 0.0, half=0.009), ok),
            ('4', 'r4', gio_d('O1-F08', 1e-4, 0.01, 0.0, half=0.007), dict(outcome='khong_dat', u_t_nodes=[0.03]))]
    return out


def test_cong_rule_2b():
    import o1_gate
    from o1_feasibility import Schedule
    from o1_tasks import FIG8
    k = dict(U_p_m=0.004, U_psi_deg=2.0)
    d = CU.cong(k, _plan(), _luot(lost_first=True), ['O1-F05', 'O1-F08'], 'bo_tick', 'C8/x')
    p8 = d['dieu_2b']['per_level']['O1-F08']
    assert p8['u_t_c8_s'] == pytest.approx(max(0.021 + 0.009, 0.03 + 0.007))
    v, wz = o1_gate.fig8_cmd_max(Schedule(FIG8['O1-F08']))
    assert p8['uhat']['g3'] == pytest.approx(0.004 + p8['u_t_c8_s'] * v)
    assert p8['uhat']['g5'] == pytest.approx(2.0 + math.degrees(p8['u_t_c8_s'] * wz))
    assert p8['report_only'] == [c for c in ('g2', 'g3', 'g5') if p8['uhat'][c] > 0.5 * o1_gate.THRESHOLDS[c]]
    assert [r['run_id'] for r in p8['runs']] == ['r3', 'r4']
    assert d['format'] == 'gate_c8_v1' and d['u_t_branch'] == 'bo_tick'


def test_cong_strict_half():
    assert CU.uhat(0.05, 1.0, 0.0, 1.0, 1.0)['g2'] == 0.05
    k = dict(U_p_m=0.05, U_psi_deg=1.0)
    lu = [(s, r, g, dict(l, u_t_nodes=[0.0])) for s, r, g, l in _luot()]
    lu = [(s, r, dict(g, half_s=0.0), l) for s, r, g, l in lu]
    d = CU.cong(k, _plan(), lu, ['O1-F05'], 'bo_tick', 'x')
    assert 'g2' not in d['dieu_2b']['per_level']['O1-F05']['report_only']      # 0.05 = half of D: still decides


def test_cong_needs_both_controllers_and_rerun_rule():
    k = dict(U_p_m=0.004, U_psi_deg=2.0)
    lu = [x for x in _luot() if x[0] != '2']
    with pytest.raises(CU.UError, match='no run'):
        CU.cong(k, _plan(), lu, ['O1-F05'], 'bo_tick', 'x')
    lu = _luot() + [('1', 'r1b', gio_d('O1-F05', 1e-4, 0.01, 0.0), dict(outcome='dat', u_t_nodes=[0.02]))]
    with pytest.raises(CU.UError, match='rerun only replaces'):
        CU.cong(k, _plan(), lu, ['O1-F05'], 'bo_tick', 'x')
    lu = _luot() + [('9', 'r9', gio_d('O1-F05', 1e-4, 0.01, 0.0), dict(outcome='dat', u_t_nodes=[0.02]))]
    with pytest.raises(CU.UError, match='not in the C8 plan'):
        CU.cong(k, _plan(), lu, ['O1-F05'], 'bo_tick', 'x')
    lu = [(s, r, g, dict(l, u_t_nodes=[math.inf]) if r == 'r2' else l) for s, r, g, l in _luot()]
    with pytest.raises(CU.UError, match='each controller'):
        CU.cong(k, _plan(), lu, ['O1-F05'], 'bo_tick', 'x')


def test_write_gate_needs_reader(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'gate_c8', None)
    with pytest.raises(CU.UError, match='gate_c8'):
        CU.write_gate(str(tmp_path / 'g.json'), {})
    assert not (tmp_path / 'g.json').exists()
