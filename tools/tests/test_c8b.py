"""Tests of the C8b scripts (c8b_gio, c8b_luoi, c8b_3b, c8b_u), CODE_C8B 17/11/2026, design
CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_C8b_v4.md. Synthetic data with the answers planted."""
import csv
import hashlib
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


@pytest.mark.parametrize('col,val,txt', [('reset_n', 4, 'reset_n'), ('g_valid', 0, 'g_valid'), ('r_valid', 0, 'r_valid')])
def test_blocking_checks_not_concluded(tmp_path, col, val, txt):
    def m(rows):
        rows[1200][col] = val
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ket_luan'] == 'chua_ket_luan' and any(txt in x for x in out['ly_do_chua'])


@pytest.mark.parametrize('age', ['-1.9', '-2.5', '-7.0'])
def test_negative_age_has_no_limit(tmp_path, age):
    # J v6 section 5 (J4-14, item 12(f)): no negative age limit; an invalid sample is told by r_valid / rv_n
    def m(rows):
        rows[1200]['age_g_ms'] = age
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ok'] and out['kiem']['n_age_neg'] >= 1 and out['kiem']['age_min_ms'] == pytest.approx(float(age))


def test_f11_blocks_on_a_good_run(tmp_path):
    fp, lp = make_run(str(tmp_path))
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    k = out['kiem_gio_d1']
    assert k['trang_thai'] == 'dat' and k['L_le_U'] and k['ghep_ok'] and k['trong_dai_c8'] is None
    assert k['truoc_boot']['co_boot'] is False and k['truoc_boot']['trang_thai'] is None
    assert not out['cap_dem']['truot'] and not out['ho_log']['truot'] and out['dem_truot_kiem_gio'] is False


def test_f11_hole_is_ho_log_not_d1(tmp_path):
    def m(rows):
        for r in rows[1000:]:
            r['timestamp_us'] = int(r['timestamp_us']) + 110000
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['ket_luan'] == 'hong_he_do' and out['ho_log']['truot']
    assert out['ly_do_theo_loai']['ho_log'] and not any('log hole' in x for x in out['ly_do_theo_loai']['kiem_gio_d1'])


def _f11_case(st_fail, t_boot, cw_end, tt=()):
    kenh = {c: dict(D_s=0.01, ok=not st_fail) for c in G.CHANNELS}
    out = dict(L_s=0.0, U_s=0.02)
    kiem = dict(pairs_valid_share=1.0, log_hole_max_s=0.02, t_boot_log_s=t_boot)
    loai = dict(kiem_gio_d1=['x'] if st_fail else [], cap_dem=[], ho_log=[], khac=[])
    return G.f11(out, kenh, kiem, loai, [], None, [0.0, cw_end], tt)


@pytest.mark.parametrize('fail,t_boot,cw_end,dem,truoc', [
    (True, None, 10.0, True, None),               # failed, no boot: counts
    (True, 12.0, 10.0, True, 'truot'),            # failed on rows all before the boot (whole window before): counts
    (True, 8.0, 10.0, False, 'khong_kiem'),       # the window reaches past the boot: not independent, not counted
    (False, 12.0, 10.0, False, 'dat'),
])
def test_m2_boot_and_the_clock_check_count(fail, t_boot, cw_end, dem, truoc):
    # the boot comes from the scorer's thao_tac (one source, CODE 19/11); t_boot_log_s is its time in the log
    r = _f11_case(fail, t_boot, cw_end, ('boot',) if t_boot is not None else ())
    assert r['kiem_gio_d1']['trang_thai'] == ('truot' if fail else 'dat')
    assert r['kiem_gio_d1']['truoc_boot']['trang_thai'] == truoc and r['dem_truot_kiem_gio'] is dem


@pytest.mark.parametrize('tt', ['khoa_sieu_am', 'loi_nap_gain'])
def test_m2_any_operation_run_not_counted(tt):
    # CODE 19/11 (J v6 section 5 'do thao tac'): a sonar lock or gain load error with a failed clock check: not counted
    r = _f11_case(True, None, 10.0, (tt,))
    assert r['kiem_gio_d1']['trang_thai'] == 'truot' and r['dem_truot_kiem_gio'] is False
    assert r['kiem_gio_d1']['truoc_boot']['thao_tac'] == [tt]
    with pytest.raises(G.GioError):
        _f11_case(True, None, 10.0, ('bam_tay',))


def test_m2_reset_n_is_not_a_boot(tmp_path):
    # CODE 19/11 (CAM checked AGV logger.c 147, main.c 299): reset_n counts IMU chip resets, not firmware boots. Its
    # change is not an operator loss (the old code took it for a boot: an error of ours); it still leaves the clock
    # without a finite bound (IMU report counters restart)
    def m(rows):
        for r in rows[1200:]:
            r['reset_n'] = 4
    fp, lp = make_run(str(tmp_path), mutate=m)
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    tb = out['kiem_gio_d1']['truoc_boot']
    assert not tb['co_boot'] and tb['t_boot_log_s'] is None and tb['thao_tac'] == []
    assert any('reset_n' in x and 'IMU' in x for x in out['ly_do_theo_loai']['chua'])
    # a time in kiem without 'boot' among the operator kinds is not a boot either
    r = _f11_case(True, 12.0, 10.0)
    assert r['kiem_gio_d1']['truoc_boot']['co_boot'] is False and r['dem_truot_kiem_gio'] is True


def test_m2_boot_from_the_score_without_its_time_not_counted():
    # a boot told by the scorer, its time not in the log rows (one boot per file): not independent evidence (J5-9)
    r = _f11_case(True, None, 10.0, ('boot',))
    tb = r['kiem_gio_d1']['truoc_boot']
    assert tb['co_boot'] and not tb['du_cua_so'] and tb['trang_thai'] == 'khong_kiem' and 'ly_do' in tb
    assert r['dem_truot_kiem_gio'] is False


def _score(tmp_path, log_path, thao=None, sha=None, name='score.json'):
    import c8_u_imu
    d = dict(outcome='hong_he_do', fail_cause='thao_tac' if thao else 'he_do',
             esp32_sha256=sha or c8_u_imu.sha256(log_path))
    if thao is not None:
        d['thao_tac'] = thao
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding='utf-8')
    return str(p)


def test_cli_operator_kinds_from_the_score(tmp_path, base_run):
    base = ['--frames', base_run[1], '--esp32', base_run[2], '--task', 'O1-F05']
    sc = _score(tmp_path, base_run[2], ['boot', 'khoa_sieu_am'])
    out_p = str(tmp_path / 'g.json')
    assert G.main(base + ['--score', sc, '--out', out_p]) == 0
    d = json.load(open(out_p, encoding='utf-8'))
    tb = d['kiem_gio_d1']['truoc_boot']
    assert tb['thao_tac'] == ['boot', 'khoa_sieu_am'] and tb['co_boot'] and d['dem_truot_kiem_gio'] is False
    import c8_u_imu
    assert d['thao_tac_nguon']['sha256'] == c8_u_imu.sha256(sc)
    # the score of another run, or a kind outside the shared labels (CT3-6), is refused
    assert G.main(base + ['--score', _score(tmp_path, base_run[2], ['boot'], sha='0' * 64, name='s2.json'),
                          '--out', str(tmp_path / 'g2.json')]) == 2
    assert G.main(base + ['--score', _score(tmp_path, base_run[2], ['sieu_am'], name='s3.json'),
                          '--out', str(tmp_path / 'g3.json')]) == 2
    # no operator kind in the score: none
    assert G.main(base + ['--score', _score(tmp_path, base_run[2], None, name='s4.json'),
                          '--out', str(tmp_path / 'g4.json')]) == 0
    assert json.load(open(tmp_path / 'g4.json', encoding='utf-8'))['kiem_gio_d1']['truoc_boot']['thao_tac'] == []


def test_f11_missing_c8_range_is_not_checked(tmp_path):
    fp, lp = make_run(str(tmp_path))
    out, _, _ = G.analyse(fp, lp, 'O1-F05', khoa=dict(dai={}))
    assert out['kiem_gio_d1']['trong_dai_c8'] is None and out['kiem_gio_d1']['trang_thai'] == 'khong_kiem'


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
    assert d['luoi_so_bo_sha256'] is None and d['cua_so_v1_sha256'] is None


def test_cli_ties_the_preliminary_grid_and_window(tmp_path, base_run):
    # CAM CT2-7: the final clock file names the preliminary grid and the reference window built from it
    sb, cs = tmp_path / 'so_bo.json', tmp_path / 'cua_so.json'
    sb.write_text('{"format": "c8b_gio_v1"}', encoding='utf-8')
    cs.write_text('{"schema": "cua_so_v1"}', encoding='utf-8')
    base = ['--frames', base_run[1], '--esp32', base_run[2], '--task', 'O1-F05']
    out_p = str(tmp_path / 'gio.json')
    # the final file needs the scorer's score.json (the operator kinds, one source)
    assert G.main(base + ['--so-bo', str(sb), '--cua-so', str(cs), '--out', str(tmp_path / 'g0.json')]) == 2
    sc = _score(tmp_path, base_run[2], None)
    base = base + ['--score', sc]
    assert G.main(base + ['--so-bo', str(sb), '--cua-so', str(cs), '--out', out_p]) == 0
    d = json.load(open(out_p, encoding='utf-8'))
    import c8_u_imu
    assert d['luoi_so_bo_sha256'] == c8_u_imu.sha256(str(sb)) and d['cua_so_v1_sha256'] == c8_u_imu.sha256(str(cs))
    assert G.main(base + ['--cua-so', str(cs), '--out', str(tmp_path / 'g2.json')]) == 2


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


def test_khong_dat_needs_the_frame_above_at_both_nodes():
    """J v6 line 112 (Codex D2-16): the frame violates at BOTH nodes of the cell; lo 0.12 at one node and 0.09 at
    the other is not a violation of the cell; a lo not finite at one node neither."""
    st = ['dat', 'dat', 'khong_dat', 'dat', 'dat']
    a = node(st, rows_with(0.2, lo=0.12))
    b = node(['dat', 'dat', 'chua_ket_luan', 'dat', 'dat'], rows_with(0.2, lo=0.09))
    res = LU.combine([a, b])
    assert res['states']['g3'] == 'chua_ket_luan' and res['lo_cells']['g3'] == [pytest.approx(0.09)]
    assert res['bieu_dien']['g3']
    b2 = node(st, rows_with(0.2, lo=0.11))
    res = LU.combine([a, b2])
    assert res['states']['g3'] == 'khong_dat' and res['nums']['g3']['chung']['lo'] == pytest.approx(0.11)
    b3 = node(['dat', 'dat', 'chua_ket_luan', 'dat', 'dat'], rows_with(0.2, lo=-math.inf))
    assert LU.combine([a, b3])['states']['g3'] == 'chua_ket_luan'
    # three nodes: frame A above at nodes 0-1, frame B above at nodes 1-2 (J v6: the frame may differ between cells)
    n0 = node(st, rows_with(0.2, lo=0.12, extra=(0.4,)))
    n1 = node(st, rows_with(0.2, lo=0.12, extra=(0.4,)))
    n1['rows'][0.4]['g3'] = 0.13
    n2 = node(st, rows_with(0.4, lo=0.13))
    n2['rows'][0.2]['g3'] = 0.05
    assert LU.combine([n0, n1, n2])['states']['g3'] == 'khong_dat'


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


def _so_do_nodes(q_mid, n_nodes=3, mid_ok=True, other_ok=True):
    """G5 three-branch khong_dat at every node (lo 8.0); q of G5 per node: q_mid at the middle node, 9.0 elsewhere."""
    st = ['dat', 'dat', 'dat', 'dat', 'khong_dat']
    out = []
    for i in range(n_nodes):
        ok = mid_ok if i == n_nodes // 2 else other_ok
        x = node(st, rows_with(0.2, c='g5', lo=8.0), ok=ok, kind='loi')
        if ok:
            q = q_mid if i == n_nodes // 2 else 9.0
            x['nums']['g5'].update(q=q, q_chung=q)
        out.append(x)
    return out


def test_code_23_11_so_do_at_the_middle_node():
    # CODE 23/11 (2): a so_do condition decides by q of the middle node of the grid (U not added), (4): a node not
    # scored leaves q below the threshold chua_ket_luan; the three-branch grid state is kept
    ns = [1.0, 1.001, 1.002]
    res = LU.combine(_so_do_nodes(7.0), frozenset({'g5'}), ns)
    assert res['states']['g5'] == 'dat' and res['states_ba_nhanh'] == {'g5': 'khong_dat'} and res['outcome'] == 'dat'
    assert res['q_so_do']['g5'] == dict(nut=1, q=7.0, q_chung=7.0) and res['decide'] == list(LU.CONDS)
    assert LU.combine(_so_do_nodes(7.6), frozenset({'g5'}), ns)['states']['g5'] == 'khong_dat'
    r = LU.combine(_so_do_nodes(7.0, other_ok=False), frozenset({'g5'}), ns)
    assert r['states']['g5'] == 'chua_ket_luan' and r['outcome'] == 'chua_ket_luan'
    r = LU.combine(_so_do_nodes(7.0, mid_ok=False), frozenset({'g5'}), ns)          # the nearest scored node, lower
    assert r['q_so_do']['g5']['nut'] == 0 and r['states']['g5'] == 'khong_dat'      # index on a tie (q 9.0 there)
    assert LU.combine(_so_do_nodes(7.0), frozenset(), ns)['states']['g5'] == 'khong_dat'   # no so_do: three branches
    assert LU.nut_giua([1.0, 1.001, 1.002, 1.004], [True] * 4) == 2                 # by time: middle 1.002
    import gate_c8
    for nodes_s, ok in [([1.0, 1.001, 1.002, 1.004], [True] * 4), ([1.0, 1.001], [True, True]),
                        ([1.0, 1.001, 1.002], [True, False, True]), ([1.0], [False]), (None, [True, True, True])]:
        assert LU.nut_giua(nodes_s, ok) == gate_c8._nut_giua(nodes_s, ok)


def test_code_23_11_gate_block_so_do():
    # every condition decides; G5 by q of the middle node (gate_v3.4, q_so_do), its three-branch state kept
    n = _so_do_nodes(7.0)
    res = LU.combine(n, frozenset({'g5'}), [1.0, 1.001, 1.002])
    g = LU.gate_block(n, res, dict(nodes_s=[1.0, 1.001, 1.002]), 'a' * 64, 'O1-F08', 'b' * 64)
    assert g['decide'] == list(LU.CONDS) and g['outcome_decide'] == g['outcome'] == 'dat' and g['version'] == 'gate_v3.4'
    assert g['so_do'] == ['g5'] and g['gate_file_sha256'] == 'b' * 64 and g['level'] == 'O1-F08'
    assert g['g5']['state_ba_nhanh'] == 'khong_dat' and g['g5']['q_so_do'] == dict(nut=1, q=7.0, q_chung=7.0)
    with pytest.raises(LU.LuoiError, match='without its source'):
        LU.gate_block(n, res, dict(nodes_s=[1.0, 1.001, 1.002]), 'a' * 64, 'O1-F08', None)


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


def test_so_do_in_outcome():
    # CODE 23/11: a so_do condition decides (report only before: left out of the outcome)
    st = ['dat', 'dat', 'dat', 'dat', 'khong_dat']
    n = [node(st, rows_with(0.2, c='g5', lo=8.0)) for _ in range(2)]
    assert LU.combine(n)['outcome'] == 'khong_dat'
    res = LU.combine(n, frozenset({'g5'}), [1.0, 1.001])
    assert res['states']['g5'] == 'khong_dat' and res['outcome'] == 'khong_dat' and 'g5' in res['decide']


def test_state_from_numbers_matches_rq3_log():
    import rq3_log
    rng = np.random.default_rng(3)
    vals = lambda: rng.choice([math.nan, math.inf, rng.uniform(0, 0.2)])
    for _ in range(2000):
        lo_c, lo_u, hi_u = vals(), vals(), vals()
        nu = [0, 1, None][int(rng.integers(0, 3))]
        assert LU._state_from_numbers(lo_c, lo_u, hi_u, nu, 0.1) == rq3_log.cond_state2(lo_c, lo_u, hi_u, nu, 0.1)


def _write_node(d, k, states, rows, u_t=0.031, so_do=(), sha=None):
    import gate_c8
    g = {'g1': dict(state=states[0])}
    for i, (c, s) in enumerate(zip(LU.MEASURED, states[1:])):
        lo = max(r[2][i] for r in rows)                  # numbers that follow the rows and the state (C8BS-17)
        hi = lo + 0.02 if s != 'dat' else min(lo + 0.02, LU.THRESHOLDS[c])
        g[c] = dict(state=s, q=lo + 0.01, lo=lo, hi=hi, n_undef=0, chung=dict(q=lo + 0.01, lo=lo, hi=hi, n_undef=0))
    for c in so_do:                                      # a node scored with so_do (CODE 23/11): c7_score's block
        g[c].update(state_ba_nhanh=g[c]['state'], nhanh='so_do',
                    state=gate_c8.so_do_state(g[c]['q'], g[c]['q'], 0, LU.THRESHOLDS[c]))
    if so_do:
        g.update(version='gate_v3.4', so_do=list(so_do), gate_file_sha256=sha)
    with open(os.path.join(d, f'nut_{k:03d}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(outcome='dat', gate_v3=g, U_t_s=u_t, t0_log_us=5e6, gio_c8b=dict(D=1.0 + 0.001 * k),
                       so_do=list(so_do)), f)
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


def _luoi_dir(tmp_path, n=2):
    d = str(tmp_path / 'luoi')
    os.makedirs(d)
    gio = dict(format='c8b_gio_v1', task='O1-F05', ok=True, nodes_s=[1.0 + 0.001 * k for k in range(n)])
    gp = str(tmp_path / 'gio.json')
    json.dump(gio, open(gp, 'w', encoding='utf-8'))
    import c8_u_imu
    for k in range(n):
        with open(os.path.join(d, f'nut_{k:03d}_nguon.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(scorer=dict(commit='abc1234', dirty=False), gio_sha256=c8_u_imu.sha256(gp),
                           D_s=gio['nodes_s'][k]), f)
    return d, gp


def test_f11_luoi_no_node_scored(tmp_path):
    # F11: every node a tool error (no score file): no gate block, the run closed not concluded, the clock file SHA kept
    d, gp = _luoi_dir(tmp_path)
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['gate_v3'] is None and r['n_nut_cham'] == 0 and r['trang_thai_luot'] == 'chua_ket_luan_khong_nut'
    assert r['outcome'] == 'chua_ket_luan' and len(r['ly_do_nut']) == 2 and all(r['ly_do_nut'])
    import c8_u_imu
    assert r['gio_sha256'] == c8_u_imu.sha256(gp)


def test_f11_luoi_every_node_lost(tmp_path):
    d, gp = _luoi_dir(tmp_path)
    for k in range(2):
        with open(os.path.join(d, f'nut_{k:03d}.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(outcome='hong_he_do', fail_cause='he_do', reasons=['gap']), f)
        open(os.path.join(d, f'nut_{k:03d}_rows.csv'), 'w').close()
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['trang_thai_luot'] == 'hong_he_do' and r['outcome'] == 'hong_he_do' and r['gate_v3'] is None


def test_ct3_3_node_not_a_c8_scoring_is_not_taken(tmp_path):
    # Codex CAM CT3-3: a node whose score.json says cham_c8 false (clock not from c8b_gio_v1, ...) is not scored
    d, gp = _luoi_dir(tmp_path)
    for k in range(2):
        _write_node(d, k, ['dat'] * 5, [(10.000001, 1, [0.05, 0.05, 0.05, 1.0])])
    p0 = os.path.join(d, 'nut_000.json')
    sc = json.load(open(p0, encoding='utf-8'))
    sc.update(cham_c8=False, cham_c8_ly_do='gio khong tu c8b_gio_v1')
    json.dump(sc, open(p0, 'w', encoding='utf-8'))
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['nodes_ok'] == [False, True] and r['n_nut_cham'] == 1
    assert 'cham_c8 false' in r['node_errors'][0] and 'c8b_gio_v1' in r['node_errors'][0]
    # every node so: no node scored, the run closed not concluded
    p1 = os.path.join(d, 'nut_001.json')
    json.dump(sc, open(p1, 'w', encoding='utf-8'))
    os.remove(os.path.join(d, 'luoi.json'))
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['n_nut_cham'] == 0 and r['trang_thai_luot'] == 'chua_ket_luan_khong_nut' and r['gate_v3'] is None


def test_ct4_4_g5_of_f08_so_do_without_a_gate_mask(tmp_path):
    # Codex CAM CT4-4: G5 of the figure 8 at F08 is in the mask by the J source (J v6 section 6, L130), also when no
    # gate file (empty mask) is given; since 23/11 it decides by q (CODE, Liam): G5 10 deg makes the run khong_dat
    d = str(tmp_path / 'luoi')
    os.makedirs(d)
    gio = dict(format='c8b_gio_v1', task='O1-F08', ok=True, nodes_s=[1.0, 1.001])
    gp = str(tmp_path / 'gio.json')
    json.dump(gio, open(gp, 'w', encoding='utf-8'))
    import c8_u_imu
    for k in range(2):
        with open(os.path.join(d, f'nut_{k:03d}_nguon.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(scorer=dict(commit='abc1234', dirty=False), gio_sha256=c8_u_imu.sha256(gp),
                           D_s=gio['nodes_s'][k]), f)
        _write_node(d, k, ['dat', 'dat', 'dat', 'dat', 'khong_dat'], [(10.000001, 1, [0.05, 0.05, 0.05, 10.0])])
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) != 0      # nodes scored without the so_do of J
    for k in range(2):
        _write_node(d, k, ['dat', 'dat', 'dat', 'dat', 'khong_dat'], [(10.000001, 1, [0.05, 0.05, 0.05, 10.0])],
                    so_do=('g5',), sha=LU.J_V6_SHA256)
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['states']['g5'] == 'khong_dat' and r['so_do'] == ['g5'] and 'g5' in r['decide']
    assert r['outcome'] == 'khong_dat' and r['q_so_do']['g5']['nut'] == 0 and r['gate_v3']['version'] == 'gate_v3.4'
    assert r['gate_v3']['gate_file_sha256'] == LU.J_V6_SHA256
    assert LU.so_do_of(None, 'O1-F05') == (frozenset(), None)
    assert LU.so_do_of(None, 'O1-F08') == (frozenset({'g5'}), LU.J_V6_SHA256)
    import c7_score
    assert LU.J_V6_SHA256 == c7_score.J_V6_SHA256


def test_f19_luoi_chain_files(tmp_path):
    # F19 (CODE 19/11): the cell chain fields of every run's luoi.json: clip {ten, clip_sha256}; file_so_h, and with
    # the H1 scoring file_cham_h1 + file_gio_h1 {duong_dan, sha256}
    import c8_u_imu
    d, gp = _luoi_dir(tmp_path)
    for k in range(2):
        _write_node(d, k, ['dat'] * 5, [(10.000001, 1, [0.05, 0.05, 0.05, 1.0])])
    clip = tmp_path / 'IMG_9999.MOV'
    clip.write_bytes(b'video')
    so_h = tmp_path / 'drift.json'
    so_h.write_text('{}', encoding='utf-8')
    ch1, gh1 = tmp_path / 'cham_h1.json', tmp_path / 'gio_h1.json'
    ch1.write_text('{"a": 1}', encoding='utf-8')
    gh1.write_text('{"b": 2}', encoding='utf-8')
    base = ['--gio', gp, '--out-dir', d, '--gop-only']
    assert LU.main(base + ['--clip', str(clip), '--file-so-h', str(so_h), '--file-cham-h1', str(ch1),
                           '--file-gio-h1', str(gh1)]) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['clip'] == dict(ten='IMG_9999.MOV', clip_sha256=c8_u_imu.sha256(str(clip)))
    assert r['file_so_h'] == dict(duong_dan=str(so_h.resolve()), sha256=c8_u_imu.sha256(str(so_h)))
    assert r['file_gio_h1']['sha256'] == c8_u_imu.sha256(str(gh1)) and r['file_cham_h1']['sha256']
    os.remove(os.path.join(d, 'luoi.json'))
    assert LU.main(base + ['--file-cham-h1', str(ch1)]) == 2                    # H1 needs its own clock file
    assert LU.main(base + ['--clip', str(clip), '--clip-sha256', 'a' * 64]) == 2   # SHA not of the clip
    assert LU.main(base) == 0                                                   # not given: null (GATE refuses)
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['clip'] is None and r['file_so_h'] is None and r['file_gio_h1'] is None


def test_f19_luoi_lost_kinds_and_node_sources(tmp_path):
    # F19 (GATE 8705918): luoi.json names the score.json of every node (path, SHA) and gathers the lost-run kinds
    # (J v6 section 5 order: thao_tac, cua_so_tham_chieu, duong_lenh, he_do)
    d, gp = _luoi_dir(tmp_path, 3)
    nodes = [dict(outcome='hong_he_do', fail_cause='he_do', hong_cua_so_tc=True, outcome_reasons=['tc']),
             dict(outcome='hong_he_do', fail_cause='thao_tac', thao_tac=['boot'], outcome_reasons=['boot']),
             dict(outcome='hong_he_do', fail_cause='he_do', outcome_reasons=['gap'])]
    for k, sc in enumerate(nodes):
        with open(os.path.join(d, f'nut_{k:03d}.json'), 'w', encoding='utf-8') as f:
            json.dump(sc, f)
        open(os.path.join(d, f'nut_{k:03d}_rows.csv'), 'w').close()
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['loai_hong'] == ['thao_tac', 'cua_so_tham_chieu', 'he_do']
    assert r['loai_hong_nut'] == ['cua_so_tham_chieu', 'thao_tac', 'he_do']
    import c8_u_imu
    assert [x['nut'] for x in r['nguon_nut']] == [0, 1, 2]
    assert r['nguon_nut'][1]['sha256'] == c8_u_imu.sha256(os.path.join(d, 'nut_001.json'))
    assert os.path.isabs(r['nguon_nut'][0]['duong_dan'])
    # a scored run: no lost kind
    for k in range(3):
        _write_node(d, k, ['dat'] * 5, [(10.000001, 1, [0.05, 0.05, 0.05, 1.0])])
    os.remove(os.path.join(d, 'luoi.json'))
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['loai_hong'] == [] and r['loai_hong_nut'] == [None, None, None] and len(r['nguon_nut']) == 3
    assert LU.loai_hong_of(dict(fail_cause='duong_lenh')) == 'duong_lenh'


def test_f11_luoi_scored_and_not_bounded(tmp_path):
    d, gp = _luoi_dir(tmp_path)
    for k in range(2):
        _write_node(d, k, ['dat'] * 5, [(10.000001, 1, [0.05, 0.05, 0.05, 1.0])])
    assert LU.main(['--gio', gp, '--out-dir', d, '--gop-only']) == 0
    r = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    assert r['trang_thai_luot'] == 'cham' and r['n_nut_cham'] == 2 and r['gate_v3'] is not None
    gp2 = str(tmp_path / 'gio2.json')
    json.dump(dict(format='c8b_gio_v1', task='O1-F05', ok=False, ket_luan='chua_ket_luan', nodes_s=[]),
              open(gp2, 'w', encoding='utf-8'))
    d2 = str(tmp_path / 'l2')
    assert LU.main(['--gio', gp2, '--out-dir', d2]) == 0
    r2 = json.load(open(os.path.join(d2, 'luoi.json'), encoding='utf-8'))
    assert r2['trang_thai_luot'] == 'chua_ket_luan_khong_nut' and r2['n_nut_cham'] == 0 and r2['gio_sha256']


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
               slow=False, with_col=True, r_drift=0.0, centre_step=0.0, tag_err=None, gap=0.0, long=False):
    # tag_err(w): an error (ex, ey) in the tag frame added to the tag centre (rotates with the tag); gap: seconds added
    # to t_s after 70 s (two joined clips, as ghep_3b.py)
    rng = np.random.default_rng(seed)
    T = 1 / 60
    if long:                                      # v2: a long slow tail each way (the slowest part covers > 2 rad)
        tt = np.arange(0, 140, T)
        om = np.where(tt < 69, 6.0 * np.exp(-tt / 6) + 0.4,
                      np.where(tt < 70, 0.0, -(6.0 * np.exp(-(tt - 70) / 6) + 0.4)))
    else:
        tt = np.arange(0, 70, T)
        om = np.where(tt < 34, 6.0 * np.exp(-tt / 12) + 0.3,
                      np.where(tt < 35, 0.0, -(6.0 * np.exp(-(tt - 35) / 12) + 0.3)))
    t_mid = tt[-1] / 2
    if one_way:
        om = np.abs(om)
    if slow:
        om = np.clip(om, -1.0, 1.0)
    th = np.cumsum(om) * T
    out = []
    for ti, a, w in zip(tt, th, om):
        cx = 1.0 + drift_centre * math.sin(ti) + (centre_step if ti > t_mid else 0.0)
        Rt = R + r_drift * ti / tt[-1]
        row = 1100 + Rt * math.sin(a) / 0.0006
        col = 1900 + Rt * math.cos(a) / 0.0006
        a2 = a + w * alpha_r * (row - 1100) + wob * math.cos(a)
        xm = cx + Rt * math.cos(a2) + rng.normal(0, noise)
        ym = 0.5 + Rt * math.sin(a2) + rng.normal(0, noise)
        psi = a2 + 0.3 + rng.normal(0, 0.0003)
        if tag_err is not None:
            ex, ey = tag_err(w)
            xm += math.cos(psi) * ex - math.sin(psi) * ey
            ym += math.sin(psi) * ex + math.cos(psi) * ey
        ts = ti + (gap if ti > t_mid else 0.0)
        out.append(dict(t_s=f'{ts:.6f}', valid=1, x=xm, y=ym, tag_x=xm, tag_y=ym,
                        psi=math.remainder(psi, 2 * math.pi), row_px=row, col_px=col if with_col else ''))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


@pytest.fixture(scope='module')
def wheel(tmp_path_factory):
    d = tmp_path_factory.mktemp('wheel')
    make_wheel(str(d / 'a.csv'), 0.0005, wob=0.0, long=True)
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


def test_3b_v2_axis_offset_is_geometry_not_a_bad_setup(tmp_path):
    # the tag 1.5 mm off the axis: v1 (fixed point) fails at about 3 mm, v2 (c0 + R(psi) q) passes and finds q
    make_wheel(str(tmp_path / 'a.csv'), 0.002, wob=0.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    r = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))
    A = r['A']
    assert A['ga_dev_tho_m'] > TB.GA_TOL_M and A['ga_ok'], A['ga_why']      # v1 would fail
    assert A['q_abs_m'] == pytest.approx(0.002, abs=1e-4) and A['q_nhan_dang']
    assert A['ga_dev_m'] < 0.0006 and A['ga_dev_giu_ngoai_m'] < 0.0006
    assert A['p_dev_m'] < 0.0006 < TB.GA_TOL_M < A['p_dev_tho_m']
    assert r['version'] == 'c8b_3b_v2' and r['ga_ok']


def test_3b_v2_centre_moved_between_halves_fails_held_out(tmp_path):
    # the set-up moved 2.4 mm between the two spins: the joint fit splits it (residual about 1.2 mm < 2 mm), the
    # held-out check (fit one half, check the other) sees 2.4 mm
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, centre_step=0.0024, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    A = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['A']
    assert A['ga_dev_m'] < 0.002 and A['ga_dev_giu_ngoai_m'] > 0.002
    assert not A['ga_ok'] and any('held-out' in w for w in A['ga_why'])


def test_3b_v2_centre_wandering_still_fails(tmp_path):
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, drift_centre=0.004, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    A = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['A']
    assert not A['ga_ok'] and A['ga_why']


def test_3b_v2_q_not_identified_on_one_heading():
    t = np.arange(200) / 60.0
    psi = np.full(200, 0.4) + 1e-4 * np.sin(t)
    x, y = 1.0 + 0.001 * np.cos(psi), 0.5 + 0.001 * np.sin(psi)
    c0, q, spread, ok = TB.axis_fit(x, y, psi, np.arange(200))
    assert not ok and spread < TB.Q_SPREAD_MIN


def test_3b_v2_radius_drift_passes_gate_but_stays_in_radial_error(tmp_path):
    # radius drifting 2 mm over the clip: the in-fit model takes the drift, the held-out check (no drift term) still
    # passes at this size, the radial part keeps it (an unexplained drift is not modelled away)
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25, r_drift=0.002)
    B = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['B']
    assert B['ga_ok'], B['ga_why']
    assert B['theo_clip'][0]['mo_hinh_ban_kinh']['troi_m_s'] == pytest.approx(0.002 / 70, rel=0.25)
    assert B['rad_dev_m'] > 0.0008


def test_3b_v2_b_gate_independent_of_the_shutter_check(tmp_path):
    # a shutter reason (no col_px) must not enter the set-up check of B (first run on the real data: it did)
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25, with_col=False)
    B = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['B']
    assert B['shutter']['why'] and B['ga_ok'] and B['ga_why'] == []
    make_wheel(str(tmp_path / 'b2.csv'), 0.25, drift_centre=0.004)
    B2 = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b2.csv'))['B']
    assert not B2['ga_ok'] and isinstance(B2['ga_why'], list) and B2['ga_why']


def test_3b_v2_setup_moved_between_clips_is_per_clip_geometry(tmp_path):
    # two joined clips (gap 10 s), the set-up moved 1.6 mm between them (as A2 of session 1): each clip has its own
    # axis, the check passes, the move is reported (Codex G3B-1)
    make_wheel(str(tmp_path / 'a.csv'), 0.0015, wob=0.0, centre_step=0.0016, gap=10.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    A = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['A']
    assert len(A['theo_clip']) == 2 and A['ga_ok'], A['ga_why']
    assert A['doi_giua_clip'][0]['dc0_m'] == pytest.approx(0.0016, abs=2e-4)


def _spin_F(err_k=0.0, q=(0.0005, 0.0), T_s=1 / 60):
    """A dict F for analyse_a: each way 4 s fast then a linear slow-down 2.0 -> 0.3 rad/s over 40 s; tag centre =
    c0 + R(psi) (q + (err_k |Omega|, 0)) (a tag-frame error growing with the rate)."""
    seg = []
    for sgn in (1.0, -1.0):
        tt = np.arange(0, 164, T_s)
        w = np.where(tt < 4, 6.0, 2.0 - 1.7 * (tt - 4) / 160)
        seg.append(sgn * w)
        seg.append(np.zeros(int(1 / T_s)))
    om = np.concatenate(seg)
    t = np.arange(len(om)) * T_s
    psi = np.cumsum(om) * T_s
    ex = q[0] + err_k * np.abs(om)
    ey = q[1] + 0.0 * om
    x = 1.0 + np.cos(psi) * ex - np.sin(psi) * ey
    y = 0.5 + np.sin(psi) * ex + np.cos(psi) * ey
    return dict(t=t, x=x, y=y, psi=psi)


def test_3b_v2_speed_dependent_error_added_to_position():
    # a tag-frame error growing with the rate (1 mm per rad/s): the slow-part q differs between the speed halves
    # and that change is added to the position part (Codex G3B-3, J v6 section 6)
    A = TB.analyse_a(_spin_F(err_k=0.001), 2.614)
    dq = A['hap_thu']['dq_toc_do_max_m']
    assert dq > 5e-5
    A0 = TB.analyse_a(_spin_F(err_k=0.0), 2.614)
    assert A0['hap_thu']['dq_toc_do_max_m'] < 1e-6
    assert A['p_dev_m'] >= dq


def test_3b_v2_direction_dependent_error_added_to_position(tmp_path):
    # a tag-frame error of +-1 mm with the direction (blur): q of the two directions differ by about 2 mm, added
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, tag_err=lambda w: (0.001 * np.sign(w), 0.0), long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    A = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['A']
    assert A['hap_thu']['dq_chieu_m'] == pytest.approx(0.002, abs=3e-4)
    assert A['p_dev_m'] >= 0.0017


def test_3b_v2_one_direction_only_position_not_finite(tmp_path):
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, one_way=True, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    A = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))['A']
    assert A['p_dev_m'] is None and A['p4_why']


def test_3b_v2_cross_check_flag_is_only_the_fit(tmp_path):
    # Codex G3B-7: a tag 5 mm off the axis still fits the model: the flag says the model fits, nothing about the axis
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    make_wheel(str(tmp_path / 'c.csv'), 0.005, wob=0.0, seed=4, long=True)
    r = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'), fc=str(tmp_path / 'c.csv'))
    r0 = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'))
    assert r['A_cheo']['khop_mo_hinh_ok'] and r['A_cheo']['q_abs_m'] == pytest.approx(0.005, abs=3e-4)
    assert r['d_p_quay_m'] == r0['d_p_quay_m'] and r['d_psi_quay_deg'] == r0['d_psi_quay_deg']


def test_3b_v2_clip_from_all_rows_not_valid_gaps(tmp_path):
    # a long stretch of rejected frames inside one clip is not a joint between clips (B IMG_3602 of session 1)
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, long=True)
    with open(tmp_path / 'a.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if 20.0 < float(r['t_s']) < 40.0:
            r['valid'] = '0'
    with open(tmp_path / 'a2.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    F = TB.read(str(tmp_path / 'a2.csv'))
    assert len(TB.clips(F['t'], F['clip'])) == 1 and len(TB.clips(F['t'])) == 2


def test_3b_v2_cli_cross_check(tmp_path, capsys):
    make_wheel(str(tmp_path / 'a.csv'), 0.0015, wob=0.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    make_wheel(str(tmp_path / 'c.csv'), 0.0002, wob=0.0, seed=3, long=True)
    out = tmp_path / 'o.json'
    assert TB.main(['--frames-a', str(tmp_path / 'a.csv'), '--frames-b', str(tmp_path / 'b.csv'),
                    '--frames-a-cheo', str(tmp_path / 'c.csv'), '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    c = d['A_cheo']
    assert c['khop_mo_hinh_ok'] and c['q_abs_m'] < 0.0004 and 'ga_ok' not in c
    assert [n['vai'] for n in d['nguon']] == ['A', 'B', 'A_cheo']
    assert '#ket_qua,ma=0,ga_ok=1' in capsys.readouterr().out


def _segs(parts, T_s=1 / 60):
    """Rate profile from parts: (seconds at the top, top rate, end rate of a linear slow-down, seconds of the
    slow-down, gap after in s); returns t, Omega and the clip index of every frame (a gap > CLIP_GAP_S is a new clip)."""
    t_l, om_l, t0 = [], [], 0.0
    for top_s, w0, w1, down_s, gap in parts:
        tt = np.arange(0, top_s + down_s, T_s)
        w = np.where(tt < top_s, w0, w0 + (w1 - w0) * (tt - top_s) / down_s)
        t_l.append(t0 + tt)
        om_l.append(w)
        t0 = t0 + tt[-1] + T_s + gap
    return np.concatenate(t_l), np.concatenate(om_l)


def _psi_of(t, om):
    psi = np.zeros(len(t))
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        psi[i] = psi[i - 1] + (om[i] * dt if dt < 1.0 else 0.0)
    return psi


def _axis_F(parts, q=(0.0005, 0.0), step_at=None, step=0.0):
    t, om = _segs(parts)
    psi = _psi_of(t, om)
    cx = 1.0 + (step if step_at is None else np.where(t >= step_at, step, 0.0))
    x = cx + np.cos(psi) * q[0] - np.sin(psi) * q[1]
    y = 0.5 + np.sin(psi) * q[0] + np.cos(psi) * q[1]
    return dict(t=t, x=x, y=y, psi=psi)


def test_3b_v3_runs_split_a_clip_at_a_respin():
    t, om = _segs([(4, 3.0, 0.35, 40, 0.0), (4, 3.0, 1.4, 40, 0.0)])
    lab = TB.spin_runs(t, om)
    assert set(lab[lab >= 0]) == {0, 1}
    assert np.all(lab[t < 43] == 0) and np.all(lab[(t > 45) & (np.abs(om) > TB.MOVE_RAD_S)] == 1)


def test_3b_v3_respin_across_a_gap_is_a_new_run():
    # a slow start (0.5 rad/s), 5 s without frames (spun by hand), then a spin from 4 rad/s: two runs (B IMG_3602)
    t, om = _segs([(0, 0.5, 0.45, 2, 5.0), (4, 4.0, 0.35, 40, 0.0)])
    lab = TB.spin_runs(t, om)
    assert set(lab[lab >= 0]) == {0, 1} and np.all(lab[t < 2.1][np.abs(om[t < 2.1]) > TB.MOVE_RAD_S] == 0)
    # a gap inside a spin-down (frames lost at speed) is not a new run
    t2, om2 = _segs([(4, 4.0, 2.0, 10, 3.0), (0, 1.9, 0.35, 30, 0.0)])
    lab2 = TB.spin_runs(t2, om2)
    assert set(lab2[lab2 >= 0]) == {0}


def test_3b_v3_setup_moved_between_spins_of_one_clip():
    # Codex G3B-1 round 2: one clip, two spins both down to 0.35 rad/s, the set-up moved 3 mm at the re-spin: the
    # held-out check by spin sees it
    F = _axis_F([(4, 3.0, 0.35, 40, 0.0), (4, -3.0, -0.35, 40, 0.0)], step_at=45.0, step=0.003)
    A = TB.analyse_a(F, 2.614)
    assert not A['ga_ok'] and any('held-out by run' in w for w in A['ga_why']), A['ga_why']
    F0 = _axis_F([(4, 3.0, 0.35, 40, 0.0), (4, -3.0, -0.35, 40, 0.0)])
    A0 = TB.analyse_a(F0, 2.614)
    assert A0['ga_ok'], A0['ga_why']
    assert A0['theo_clip'][0]['n_luot_quay'] == 2 and A0['theo_clip'][0]['ga_giu_ngoai_luot_m'] < 1e-4


def test_3b_v3_spin_without_slow_frames_is_not_used():
    # the second spin only slows to 1.4 rad/s: no slow frame, its set-up not checked, its frames not used (a move at
    # the re-spin would otherwise pass, Codex G3B-1 round 2)
    F = _axis_F([(4, 3.0, 0.35, 40, 0.0), (4, 3.0, 1.4, 40, 0.0)], step_at=45.0, step=0.003)
    A = TB.analyse_a(F, 2.614)
    c = A['theo_clip'][0]
    assert c['luot_khong_kiem'] == [1] and c['n_khung_khong_kiem'] > 2000
    assert A['p_dev_m'] is None or A['p_dev_m'] < 0.001          # the moved spin is not in the position part


def test_3b_v3_direction_between_clips_only_one_way_clips():
    # Codex G3B2-1: clip 0 has a few slow frames the other way; clip 1 is one way: the direction part is not
    # estimated from the pair (it would read about 0), so the position part is not finite
    F = _axis_F([(4, 3.0, 0.35, 40, 0.0), (0, -0.4, -0.4, 0.2, 12.0), (4, 3.0, 0.35, 40, 0.0)])
    A = TB.analyse_a(F, 2.614)
    ch = [c['chieu'] for c in A['theo_clip']]
    assert ch[0] == [-1, 1] and ch[1] == [1], ch
    assert A['hap_thu']['dq_chieu_m'] is None and A['p_dev_m'] is None
    assert any('direction' in w for w in A['p4_why'])


def _wheel_F(parts, R_of_clip, T_s=1 / 60, r_drift=None, c_step=None):
    t, om = _segs(parts, T_s)
    th = _psi_of(t, om)
    clip = np.concatenate([[0], np.cumsum(np.diff(t) > TB.CLIP_GAP_S)])
    R = np.array([R_of_clip[c] for c in clip], float)
    if r_drift is not None:
        R = R + r_drift(t)
    cx = 1.0 + (0.0 if c_step is None else np.where(t >= c_step[0], c_step[1], 0.0))
    x, y = cx + R * np.cos(th), 0.5 + R * np.sin(th)
    return dict(t=t, x=x, y=y, psi=th + 0.3, row=1100 + R * np.sin(th) / 0.0006,
                col=1900 + R * np.cos(th) / 0.0006, clip=clip)


def test_3b_v3_b_coverage_from_each_clips_radius():
    # Codex G3B2-2: clips of 260 and 240 mm; the negative way reaches 4.1 x 0.24 = 0.984 m/s < 1.0 m/s; a joint radius
    # (about 0.25 m) would report 1.025 m/s and cover
    F = _wheel_F([(4, 4.5, 0.35, 40, 10.0), (4, -4.1, -0.35, 40, 0.0)], {0: 0.26, 1: 0.24})
    B = TB.analyse_b(F, 1.0, 2.614, 1 / 60)
    assert B['ga_ok'], B['ga_why']
    assert [round(c['R0_m'], 3) for c in B['theo_clip']] == [0.26, 0.24]
    assert B['theo_clip'][1]['v_t_max_m_s'] == pytest.approx(0.984, abs=0.005)
    assert not B['du_phu'] and not B['phu']['ok']


def test_3b_v3_full_turn():
    a = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    assert TB.full_turn(a) and not TB.full_turn(a[:150]) and not TB.full_turn(np.array([]))


def test_3b_v3_b_part_turn_is_not_a_circle():
    # Codex G3B-5 round 2: a clip turning less than one turn: no circle, set-up not checked (a free circle on a part of
    # a turn takes a drift of the radius away)
    F = _wheel_F([(0, 0.6, 0.35, 12, 0.0)], {0: 0.25}, r_drift=lambda t: 0.006 * t / 12)
    B = TB.analyse_b(F, 1.0, 2.614, 1 / 60)          # about 0.9 turn in all
    assert not B['ga_ok'] and any('full turn' in w for w in B['ga_why']) and B['theo_clip'][0]['slow_n'] == 0
    # the slowest 10 % turn about 0.4 turn: the slow part is grown to a full turn; a drift of 6 mm over the slow end
    # stays in the radial part or fails the set-up check
    parts = [(4, 4.5, 0.35, 40, 0.0)]
    F2 = _wheel_F(parts, {0: 0.25}, r_drift=lambda t: np.where(t > 30, 0.006 * (t - 30) / 14, 0))
    B2 = TB.analyse_b(F2, 1.0, 2.614, 1 / 60)
    n_spin = int(np.sum(np.abs(_segs(parts)[1]) > TB.MOVE_RAD_S))
    assert B2['theo_clip'][0]['slow_n'] > 1.5 * TB.SLOW_SHARE * n_spin
    assert not B2['ga_ok'] or B2['rad_dev_m'] > 0.002


def test_3b_v3_cli_cross_check_null_fields(tmp_path, capsys):
    # Codex G3B-7 round 2: the cross-check with two clips too short for a slow part gives null q and residual; the CLI
    # prints them as null (it raised TypeError)
    make_wheel(str(tmp_path / 'a.csv'), 0.0015, wob=0.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25)
    rows = []
    for c0 in (0.0, 20.0):
        for i in range(27):
            tt = c0 + i / 60
            a = 2.0 * tt
            rows.append(dict(t_s=f'{tt:.6f}', valid=1, x=1 + 0.001 * math.cos(a), y=0.5 + 0.001 * math.sin(a),
                             psi=math.remainder(a, 2 * math.pi), row_px=1100, col_px=1900))
    with open(tmp_path / 'c.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / 'o.json'
    rc = TB.main(['--frames-a', str(tmp_path / 'a.csv'), '--frames-b', str(tmp_path / 'b.csv'),
                  '--frames-a-cheo', str(tmp_path / 'c.csv'), '--out', str(out)])
    o = capsys.readouterr()
    assert rc == 0, o.err
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['A_cheo']['q_abs_m'] is None and '|q| null' in o.out


def test_3b_v4_one_spin_with_slow_frames_is_not_a_failure():
    # Codex G3B-1 round 3: only one spin has slow frames (the other is not used): the set-up still passes (control)
    A0 = TB.analyse_a(_axis_F([(4, 3.0, 0.35, 40, 0.0), (4, 3.0, 1.4, 40, 0.0)]), 2.614)
    c = A0['theo_clip'][0]
    assert A0['ga_ok'], A0['ga_why']
    assert c['luot_khong_kiem'] == [1] and c['ga_giu_ngoai_luot_m'] is None


def test_3b_v4_small_rise_across_a_stop_is_a_new_spin():
    # Codex G3B-1 round 3: 3 -> 1.4 rad/s, 3 s without frames, 1.8 -> 0.45 rad/s (a rise of 0.4 only), set-up moved 3 mm:
    # two spins; the first has no slow frame, so its frames are not used and the move cannot enter the position part
    F = _axis_F([(4, 3.0, 1.4, 40, 3.0), (0, 1.8, 0.45, 40, 0.0)], step_at=46.0, step=0.003)
    A = TB.analyse_a(F, 2.614)
    c = A['theo_clip'][0]
    assert c['n_luot_quay'] == 2 and c['luot_khong_kiem'] == [0], c
    t, om = _segs([(4, 3.0, 1.4, 40, 3.0), (0, 1.8, 0.45, 40, 0.0)])
    lab = TB.spin_runs(t, om)
    assert set(lab[lab >= 0]) == {0, 1}


def test_3b_v4_radius_drift_peak_to_peak():
    # Codex G3B-5 round 3: two clips each way, 4.5 rad/s for 4 s then down to 0.35 over 40 s, the radius growing 5.94 mm
    # over each clip about a fixed centre: the radial part is the peak-to-peak radius over the frames used, not the
    # deviation from the mean
    parts = [(4, 4.5, 0.35, 40, 10.0), (4, -4.5, -0.35, 40, 0.0)]
    t, _ = _segs(parts)
    t1 = t[np.flatnonzero(np.diff(t) > 5)[0] + 1]
    F = _wheel_F(parts, {0: 0.25, 1: 0.25}, r_drift=lambda tt: 0.00594 * np.where(tt < t1, tt / 44.0, (tt - t1) / 44.0))
    B = TB.analyse_b(F, 0.2915, 0.6536, 1 / 60)
    assert B['rad_dev_m'] >= B['rad_lech_tb_m'] and B['rad_dev_m'] > 0.00109
    assert B['rad_kieu'].startswith('dinh-dinh')


def test_3b_v4_l132_blocked_by_an_identified_centre_move():
    # Codex G3B3-1: two full spins of one clip, radius 250 mm, the centre moved 2.6 mm at the turnaround: a moved
    # set-up, not a drift of the radius; L132 does not open
    # Codex G3B4-1 (round 4): 5 rad/s for 4 s, then down to 0.35 over 85 s, each way; the slow frames of each spin are
    # under a turn (time halves and spins on the slow frames: 'khong_nhan_dang'), the frames used of each spin cover
    # many turns: the move is identified there and L132 closes. Strict: no branch through rad_dev >= 2.60 mm
    parts = [(4, 5.0, 0.35, 85, 0.0), (4, -5.0, -0.35, 85, 0.0)]
    t, om = _segs(parts)
    tc = t[np.flatnonzero(np.sign(om[1:]) != np.sign(om[:-1]))[0] + 1]
    F = _wheel_F(parts, {0: 0.25}, c_step=(tc, 0.0026))
    B = TB.analyse_b(F, 0.2915, 0.6536, 1 / 60)
    kl = B['theo_clip'][0]['kieu_lech']
    assert not B['ga_ok'] and B['theo_clip'][0]['ga_giu_ngoai_thoi_gian_m'] > TB.GA_TOL_M
    by = {k['phep']: k for k in kl}
    assert by['thoi_gian']['kieu'] == 'khong_nhan_dang' and by['luot 0']['kieu'] == 'khong_nhan_dang'
    assert by['luot 0 khung_dung']['kieu'] == 'tam'
    assert by['luot 0 khung_dung']['dich_tam_m'] == pytest.approx(0.0026, abs=5e-5)
    assert not TB.b_troi_only(B)
    # control: a radius step of 2.6 mm at the same turnaround (centre kept): every split that can tell says
    # 'ban_kinh', none 'tam'; L132 may open, the step stays in the radial part
    F3 = _wheel_F(parts, {0: 0.25}, r_drift=lambda tt: np.where(tt >= tc, 0.0026, 0.0))
    B3 = TB.analyse_b(F3, 0.2915, 0.6536, 1 / 60)
    assert not B3['ga_ok']
    k3 = [k['kieu'] for k in B3['theo_clip'][0]['kieu_lech']]
    assert 'ban_kinh' in k3 and 'tam' not in k3 and TB.b_troi_only(B3)
    assert B3['rad_dev_m'] >= 0.0026 - 2e-4
    # the identified case itself (each part a full turn, centres 2.6 mm apart)
    th = np.linspace(0, 4 * math.pi, 800)
    x = np.r_[1.0 + 0.25 * np.cos(th), 1.0026 + 0.25 * np.cos(th)]
    y = np.r_[0.5 + 0.25 * np.sin(th), 0.5 + 0.25 * np.sin(th)]
    k = TB.split_kind(x, y, np.arange(800), np.arange(800, 1600))
    assert k['kieu'] == 'tam' and k['dich_tam_m'] == pytest.approx(0.0026, abs=1e-6)
    Bk = dict(ga_why=['clip 0: held-out by run residual 2.60 mm > 2 mm'], theo_clip=[dict(kieu_lech=[dict(k)])])
    assert not TB.b_troi_only(Bk)
    Bk['theo_clip'][0]['kieu_lech'][0]['kieu'] = 'khong_nhan_dang'
    assert TB.b_troi_only(Bk)
    # a radius step of 2.6 mm at the same place (centre kept): a radius change, L132 may open
    F2 = _wheel_F([(4, 4.0, 0.35, 40, 0.0), (4, -4.0, -0.35, 40, 0.0)], {0: 0.25},
                  r_drift=lambda tt: np.where(tt >= 44.05, 0.0026, 0.0))
    B2 = TB.analyse_b(F2, 0.2915, 0.6536, 1 / 60)
    if not B2['ga_ok']:
        assert all(k['kieu'] != 'tam' for c in B2['theo_clip'] for k in c['kieu_lech']) and TB.b_troi_only(B2)
        assert B2['rad_dev_m'] >= 0.0026 - 2e-4


def test_3b_v6_centre_move_between_two_groups_of_spins():
    # Codex G3B5-1 (round 5): four spins alternating, 4 rad/s for 2 s then down to 0.35 over 27 / 25 / 101 / 143 s, 1 s
    # apart; radius 250 mm, the centre moved 2.6 mm between spin 1 and spin 2. The first two spins alone are under a
    # turn, and each spin against the rest mixes two centres: only a cut between the two groups sees the move. The
    # searched cut finds it, L132 closes
    parts = [(2, 4.0, 0.35, 27, 1.0), (2, -4.0, -0.35, 25, 1.0), (2, 4.0, 0.35, 101, 1.0), (2, -4.0, -0.35, 143, 0.0)]
    t, _ = _segs(parts)
    starts = [t[0]] + list(t[1:][np.diff(t) > 0.5])
    tc = float(starts[2]) - 0.5
    B = TB.analyse_b(_wheel_F(parts, {0: 0.25}, c_step=(tc, 0.0026)), 0.2915, 0.6536, 1 / 60)
    by = {k['phep']: k for k in B['theo_clip'][0]['kieu_lech']}
    assert not B['ga_ok'] and B['du_phu']
    d = by['diem_chia khung_dung']
    assert d['kieu'] == 'tam' and d['dich_tam_m'] == pytest.approx(0.0026, abs=5e-5)
    assert d['t_chia_s'] > tc            # the first frame used after the move (spin 2 is used only once it slows down)
    assert not TB.b_troi_only(B)


def test_3b_v6_linear_radius_drift_is_not_a_centre_move():
    # Codex G3B5-2 (round 5, a regression of round 4's fix): fixed centre, the radius growing 250 -> 262 mm over the
    # clip; two spins each way, 4.5 rad/s for 4 s then down to 0.35 over 62 / 75 s. A fixed-radius circle on each time
    # half moved its centre 1.1 mm and was taken for a centre move ('tam' blocked L132; c0c47ea published P4). With a
    # drifting radius in each part every split says 'ban_kinh'; L132 opens and the drift stays in rad_dev
    parts = [(4, 4.5, 0.35, 62, 0.0), (4, -4.5, -0.35, 75, 0.0)]
    t, _ = _segs(parts)
    F = _wheel_F(parts, {0: 0.25}, r_drift=lambda tt: 0.012 * (tt - t[0]) / (t[-1] - t[0]))
    B = TB.analyse_b(F, 0.2915, 0.6536, 1 / 60)
    kl = B['theo_clip'][0]['kieu_lech']
    assert not B['ga_ok'] and B['du_phu'] and kl
    assert all(k['kieu'] == 'ban_kinh' and k['dich_tam_m'] < 1e-4 for k in kl), kl
    assert TB.b_troi_only(B) and B['rad_dev_m'] > 0.009
    # the drifting circle itself: centre and both radii back
    th = np.linspace(0, 6 * math.pi, 900)
    tt = np.linspace(0, 30, 900)
    R = 0.25 + 0.0004 * tt
    x, y = 1.0 + R * np.cos(th), 0.5 + R * np.sin(th)
    cx, cy, R0, R1 = TB.circle_drift_fit(x, y, tt, np.arange(900))
    assert (cx, cy) == (pytest.approx(1.0, abs=1e-9), pytest.approx(0.5, abs=1e-9))
    assert R1 == pytest.approx(0.0004, rel=1e-6) and R0 == pytest.approx(0.25 + 0.0004 * 15, abs=1e-9)


def test_3b_v5_clip_without_frames_used_does_not_void_rad_dev():
    # Codex G3B4-2 (round 4): two B clips covering F02, then a third clip at 3 -> 2 rad/s on 250 mm (0.50 m/s and more,
    # above the band used 1.1 x 0.2915 m/s): it has a slow part but no frame used; rad_dev stays the larger of the two
    # first clips (it was null, so P4 was lost)
    two = [(4, 4.0, 0.35, 40, 10.0), (4, -4.0, -0.35, 40, 10.0)]
    F2 = _wheel_F(two, {0: 0.25, 1: 0.25, 2: 0.25}, r_drift=lambda tt: 0.0008 * np.sin(0.3 * tt))
    B2 = TB.analyse_b(F2, 0.2915, 0.6536, 1 / 60)
    F3 = _wheel_F(two + [(2, 3.0, 2.0, 10, 0.0)], {0: 0.25, 1: 0.25, 2: 0.25},
                  r_drift=lambda tt: 0.0008 * np.sin(0.3 * tt))
    B3 = TB.analyse_b(F3, 0.2915, 0.6536, 1 / 60)
    c3 = B3['theo_clip'][2]
    assert c3['slow_n'] > 0 and c3['ban_kinh_dinh_dinh_m'] is None
    assert B2['du_phu'] and B3['du_phu']
    assert B2['rad_dev_m'] is not None and B3['rad_dev_m'] == pytest.approx(B2['rad_dev_m'])
    assert B3['rad_dev_m'] == max(c['ban_kinh_dinh_dinh_m'] for c in B3['theo_clip'][:2])
    # no clip with a frame used: still null
    Fh = _wheel_F([(2, 3.0, 2.0, 10, 0.0)], {0: 0.25})
    assert TB.analyse_b(Fh, 0.2915, 0.6536, 1 / 60)['rad_dev_m'] is None


def test_khoa_takes_a_3b_file_usable_by_l132():
    # Codex round 4 note (G3B3-2): positive test of the L132 read path of c8b_u; the 3b numbers are used as given
    base = KH()
    l132 = dict(TBD, ga_ok=False, dung_duoc=True, b_dung_ke_ca_troi=True)
    k = KH(tb=l132)
    assert k['U_p_phan_m'] == base['U_p_phan_m'] and k['U_psi_phan_deg'] == base['U_psi_phan_deg']
    for bad in (dict(l132, b_dung_ke_ca_troi=False), dict(l132, dung_duoc='true'), dict(l132, b_dung_ke_ca_troi=1)):
        with pytest.raises(CU.UError, match='ga_ok'):
            KH(tb=bad)


def test_3b_v3_l132_b_used_with_its_drift(tmp_path):
    # L132: B failing the set-up check only by the drift of its radius is used when asked; the drift stays in rad_dev
    make_wheel(str(tmp_path / 'a.csv'), 0.0005, wob=0.0, long=True)
    make_wheel(str(tmp_path / 'b.csv'), 0.25, r_drift=0.006)
    r0 = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'), 'O1-F02')
    assert not r0['B']['ga_ok'] and TB.b_troi_only(r0['B']), r0['B']['ga_why']
    assert not r0['dung_duoc'] and not r0['b_dung_ke_ca_troi']
    r = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv'), 'O1-F02', b_ke_ca_troi=True)
    assert r['dung_duoc'] and r['b_dung_ke_ca_troi'] and r['b_ke_ca_troi_nguon'].startswith('L132')
    assert r['d_p_quay_m'] >= r['B']['rad_dev_m'] > 0.002
    # another reason (centre wandering: residual in the fit) still blocks
    make_wheel(str(tmp_path / 'b2.csv'), 0.25, drift_centre=0.006)
    r2 = TB.analyse(str(tmp_path / 'a.csv'), str(tmp_path / 'b2.csv'), 'O1-F02', b_ke_ca_troi=True)
    if not r2['B']['ga_ok']:
        assert TB.b_troi_only(r2['B']) or not r2['dung_duoc']
    assert not TB.b_troi_only(dict(ga_why=['clip 0: residual 2.50 mm > 2 mm']))
    assert not TB.b_troi_only(dict(ga_why=['clip 0: held-out by time: a half of the slowest part has too few frames']))


def test_3b_window_over_turnaround_not_used():
    T = 1 / 60
    tt = np.arange(0, 80, T)
    om = np.where(tt < 40, 6.0 * np.exp(-tt / 15) + 0.3, -(6.0 * np.exp(-(tt - 40) / 15) + 0.3))
    F = dict(t=tt, x=np.zeros_like(tt), y=np.zeros_like(tt), psi=np.cumsum(om) * T)
    assert TB.analyse_a(F, 2.614)['psi_osc_deg'] < 0.01


# ── c8b_u ────────────────────────────────────────────────────────────────────────────────────

def _fsha(run):
    """A stand-in SHA-256 of the final frames.csv of a run."""
    import hashlib
    return hashlib.sha256(('frames ' + run).encode()).hexdigest()


MOC_SHA = {k: ch * 64 for k, ch in zip(('lich', 'tap_moc', 'u_so', 'session'), 'abcd')}
CAM = {"version": "c8_cam_u_vao_v3",
       "P1": {"bo_nhom_max_m": 0.002, "tung_khung_max_m": 0.0018, "bo_nhom_rms_m": 0.0014},
       # proposal v3 (Liam 18/11 "Duyet ban 3"): U_moc before the margin, approved method, session passed, sources;
       # nguong_m, u_so_m only information
       "moc_dich": {"U_moc_xy_m": 0.0021, "U_moc_psi_deg": 0.07, "phuong_phap_duyet": "Liam dd/mm, phien X (thu)",
                    "tien_de_buoi": "dat",
                    "nguon": dict(frames={'a': _fsha('a')}, **MOC_SHA, bo_lay_mau='c7_moc_lay_mau_v1 (thu)',
                                  car_pose=dict(tag_offset=None, side_m=0.0964, h_m=0.12)),
                    "nguong_m": 0.001, "u_so_m": 0.0004, "moc_loai": []},
       "P3": {"vi_tri_max_m": 0.0008, "n_lan": 3},
       "A2": {"goc_max_deg": 0.1, "n_lan": 3},
       "than_dich": {"max_m": 0.0001, "kiem_duoc": True},
       "P7": {"lech_max_m": 0.0006, "sigma_max_m": 0.00029, "dinh_dinh_max_m": 0.0011},
       "A3": {"max_deg": 0.15, "quy_ve_ma_noc": True, "rms_deg": 0.074, "n_cap": 24},
       "A5": {"lech_max_deg": 0.12, "sigma_max_deg": 0.04},
       "day_doi": {"dn_m": 0.016, "dzc_m": 0.01, "kappa": 0.12 / 2.6},
       "do_cao": {"rho_max_m": 1.5, "H_m": 2.6, "h_m": 0.12},
       "bo_cham": {"tag_offset": None, "side_m": 0.0964}}
CAM_PATHS = ('P1.bo_nhom_max_m', 'P1.tung_khung_max_m', 'moc_dich.U_moc_xy_m', 'moc_dich.U_moc_psi_deg', 'P3.vi_tri_max_m',
             'A2.goc_max_deg', 'P7.lech_max_m', 'A3.max_deg', 'A5.lech_max_deg', 'day_doi.dn_m', 'day_doi.dzc_m',
             'day_doi.kappa', 'do_cao.rho_max_m', 'do_cao.H_m', 'do_cao.h_m', 'bo_cham.side_m')
TK = {"version": "c8b_thuoc_kep_v2", "cao_4_goc_mm": [120.2, 120.0, 119.8, 120.1], "e_mm": 0.1, "s_mm": 96.4,
      # a tag turned counter-clockwise by atan(0.2 / 96.4) on the plate: every edge shows the same angle
      "canh": {"truoc": [10.3, 10.1], "phai": [10.2, 10.0], "sau": [10.1, 10.3], "trai": [10.0, 10.2]},
      "tam_than_deg": 0.05, "tam_tam_mm": [0.4, -0.3], "tam_tam_dung_sai_mm": 0.5, "ban_ve_nguon": "CAD tam ma (thu)"}
XO = {"version": "c8b_xoay_v1", "q_tag_m": [0.0006, -0.0005], "du_m": 0.0007}   # near the geometry o_g = (0.4, -0.3) mm
TBD = dict(ga_ok=True, d_p_quay_m=0.0007, d_psi_quay_deg=0.25,
           B=dict(shutter=dict(use_alpha=True, alpha_r_s_px=7e-6, alpha_c_s_px=1e-7, alpha_r_le_s_px=8.75e-6,
                               alpha_c_le_s_px=1.25e-7, mot_khung_s=1 / 60)))


def _cp(d):
    return json.loads(json.dumps(d))


def sync(cam, runs=None, vang=()):
    """The camera file with moc_dich.nguon.car_pose set to its own set-up (as the exporter writes it), and frames of
    the given runs; luot_cham = the runs with frames plus vang (scored runs without U_moc frames, KCJ3-1, as
    c7_kiem_cheo 565c0b7 writes them); a file without the block is left as it is."""
    cam = _cp(cam)
    md = cam.get('moc_dich')
    if isinstance(md, dict) and isinstance(md.get('nguon'), dict):
        bc, dc = cam.get('bo_cham') or {}, cam.get('do_cao') or {}
        md['nguon']['car_pose'] = dict(tag_offset=bc.get('tag_offset'), side_m=bc.get('side_m'), h_m=dc.get('h_m'))
        if runs is not None:
            md['nguon']['frames'] = {r: _fsha(r) for r in runs}
        fr = md['nguon'].get('frames')
        if isinstance(fr, dict):
            md['luot_cham'] = dict(fr, **{v: _fsha(v) for v in vang})
            md['luot_vang_u_moc'] = sorted(vang)
    return cam


LICH_BODY = '9' * 64                          # the lich_sha256 inside lich.json (a schedule-body hash, not the file's)


def _dsha(run):
    """A stand-in SHA-256 of the detections of a run's clip."""
    return hashlib.sha256(('dets ' + run).encode()).hexdigest()


def _gsha(run):
    """A stand-in SHA-256 of the clock file of a run."""
    import hashlib
    return hashlib.sha256(('gio ' + run).encode()).hexdigest()


def track_doc(run, cam=CAM, **over):
    """track.json of the final scoring of a run (c7_measure, CODE_CAM feat/c8-33-moc e8af52b), consistent with cam."""
    off = (cam.get('bo_cham') or {}).get('tag_offset')
    d = dict(frames_sha256=_fsha(run), session_sha256=MOC_SHA['session'], u_so_sha256=MOC_SHA['u_so'],
             lich_sha256=LICH_BODY, cham_lai=2, thu=False, tag_offset=list(off) if off else [0.0, 0.0, 0.0],
             run='clip_' + run, dets_sha256=_dsha(run), lich_file_sha256=MOC_SHA['lich'])
    d.update(over)
    return d


def MT(runs=('a',), cam=CAM, **over):
    """The files khoa is given (the right ones, consistent with cam, unless overridden): SHAs, track.json of each run,
    lich.json and session.json contents."""
    bc, dc, dd = cam.get('bo_cham') or {}, cam.get('do_cao') or {}, cam.get('day_doi') or {}
    d = dict(frames={r: _fsha(r) for r in runs}, track={r: track_doc(r, cam) for r in runs}, **MOC_SHA,
             lich_doc=dict(lich_sha256=LICH_BODY, session_sha256=MOC_SHA['session'], u_so_sha256=MOC_SHA['u_so'],
                           tap_moc_file_sha256=MOC_SHA['tap_moc'],
                           clips=[dict(name='clip_' + r, kind='luot', dets_sha256=_dsha(r), n=100) for r in runs]),
             session_doc=dict(car_side_m=bc.get('side_m'), tag_h_m=dc.get('h_m'), kappa=dd.get('kappa')))
    d.update(over)
    return d


def GIOS(gios):
    """Clock files naming the frames of their run (c8b_gio writes sha256.frames)."""
    return {r: dict(g, sha256=g.get('sha256', dict(frames=_fsha(r), esp32='0' * 64))) for r, g in gios.items()}


def KHG(gios, cam, moc_tep=None, tb=None, tk=None, xo=None, **kw):
    """khoa with the clock-file SHAs and (unless given) the right files of the scoring; cam not synced."""
    runs = sorted((((cam.get('moc_dich') or {}).get('nguon') or {}).get('frames') or {'a': 0}))
    return CU.khoa(GIOS(gios), tb or TBD, cam, TK if tk is None else tk, XO if xo is None else xo,
                   moc_tep=MT(runs, cam) if moc_tep is None else moc_tep, gio_sha={r: _gsha(r) for r in gios}, **kw)


def UPP(cam, *a, **k):
    return CU.u_p_u_psi(sync(cam), *a, **k)


def KH(cam=None, tk=None, xo=None, tb=None, moc_tep=None, **kw):
    return KHG({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, sync(CAM if cam is None else cam), moc_tep, tb, tk, xo, **kw)


def gio_d(task, rate, width, gmr, ok=True, half=0.007):
    return dict(format='c8b_gio_v1', task=task, ok=ok, rate_luot=rate, width_s=width, gyro_minus_rv_s=gmr,
                half_s=half)


def _expect(cam=CAM, tk=TK, xo=XO):
    H, h, rho, s = cam['do_cao']['H_m'], cam['do_cao']['h_m'], cam['do_cao']['rho_max_m'], cam['bo_cham']['side_m']
    c = np.array(tk['cao_4_goc_mm']) * 1e-3
    e = tk['e_mm'] * 1e-3
    dn, dzc = cam['day_doi']['dn_m'], cam['day_doi']['dzc_m']
    Hl = H - dzc
    dh = np.max(np.abs(c - h)) + e
    cao = (rho + dn + s / math.sqrt(2)) * dh / (Hl - h - dh)
    dd = h / H * dn + h * dzc * (rho + dn) / (H * (H - dzc - h))
    sg = dict(truoc=1, phai=1, sau=-1, trai=-1)
    theta = sum(sg[k] * math.atan((a - b) / tk['s_mm']) for k, (a, b) in tk['canh'].items()) / 4
    dpsi = (cam['bo_cham']['tag_offset'] or [0, 0, 0])[2]
    xx = max(math.atan((a - b) / tk['s_mm']) ** 2 for (a, b) in tk['canh'].values()) + 0.01 ** 2
    a1 = math.degrees(abs(dpsi - theta) + math.atan(2 * tk['e_mm'] / tk['s_mm']) + xx) + tk['tam_than_deg']
    a6 = math.degrees((rho + dn) * (np.sum(np.abs(c - c.mean())) + 8 * e) / (2 * math.sqrt(2) * s * (Hl - c.max() - e)))
    a = dict(A1_goc_dan=a1, A2_nhac_tam=0.1, A3_huong_vung=0.15, A4_quay=0.25, A5_nhieu_khung=0.12, A6_nghieng_ma=a6,
             A7_moc_dich=cam['moc_dich']['U_moc_psi_deg'])
    dx, dy, dp = cam['bo_cham']['tag_offset'] or [0, 0, 0]
    m = {k: (a + b) / 2 for k, (a, b) in tk['canh'].items()}
    ogx = (tk['tam_tam_mm'][0] + (m['sau'] - m['truoc']) / 2) * 1e-3
    ogy = (tk['tam_tam_mm'][1] + (m['phai'] - m['trai']) / 2) * 1e-3
    rx, ry = (m['sau'] - m['truoc']) / 2 * 1e-3, (m['phai'] - m['trai']) / 2 * 1e-3
    u_g = (tk['tam_tam_dung_sai_mm'] + math.sqrt(2) * tk['e_mm']) * 1e-3 + math.hypot(rx, ry) * math.radians(tk['tam_than_deg'])
    p2 = math.hypot(dx - ogx, dy - ogy) + u_g + math.hypot(dx, dy) * math.radians(sum(a.values()))
    p = dict(P1_anh_san=0.002 + cam['moc_dich']['U_moc_xy_m'], P2_lech_tam=p2, P3_nhac_tam=0.0008, P4_quay=0.0007, P5_day_doi_zc=dd,
             P6_do_cao=cao, P7_nhieu_khung=0.0006)
    return p, a


def test_khoa_sums():
    k = KHG({'a': gio_d('O1-F05', 3e-4, 0.010, 0.005), 'b': gio_d('O1-F05', 4e-4, 0.012, 0.003),
                 'c': gio_d('O1-F08', 1e-4, 0.014, 0.004), 'd': gio_d('O1-F08', 9e-3, 0.5, 0.5, ok=False)},
                sync(CAM, 'abc', vang='d'))                             # run d (clock not ok) needs no frames
    p, a = _expect()
    assert k['U_p_phan_m'] == pytest.approx(p) and k['U_psi_phan_deg'] == pytest.approx(a)
    assert k['U_p_m'] == pytest.approx(1.25 * sum(p.values()))
    assert k['U_psi_deg'] == pytest.approx(1.25 * sum(a.values()))
    # hand numbers: P1 = 2 + U_moc 2.1 = 4.1 mm (not 2 + 1 + 2 x 0.4: nguong + 2 u_so no longer added, proposal v3); A1 = 2 atan(0.2 / 96.4) + atan(0.2 / 96.4)^2 + 0.01^2 + 0.05 = 0.2937 deg
    assert p['P1_anh_san'] == pytest.approx(0.0041) and a['A1_goc_dan'] == pytest.approx(0.29372, abs=2e-5)
    assert tuple(k['U_p_phan_m']) == CU.P_PARTS and tuple(k['U_psi_phan_deg']) == CU.A_PARTS
    assert k['troi_ppm'] == pytest.approx(1.25 * 4e-4 * 1e6)             # run d (not ok) left out
    assert k['dai']['O1-F05']['width_s'] == [0.010, 0.012] and k['dai']['O1-F05']['gyro_minus_rv_s'] == [0.003, 0.005]
    assert k['man_trap']['alpha_r_s_px'] == pytest.approx(8.75e-6) and k['man_trap_kieu'] == 'alpha'


def test_khoa_p1_takes_the_larger_of_clip_mean_and_per_frame():
    cam = _cp(CAM)
    cam['P1']['tung_khung_max_m'] = 0.0031
    assert KH(cam)['U_p_phan_m']['P1_anh_san'] == pytest.approx(0.0031 + 0.0021)


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
    with pytest.raises(CU.UError, match='version is not c8_cam_u_vao_v3'):
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
    for k_, v in (('version', 'c8b_thuoc_kep_v1'), ('cao_4_goc_mm', [120.0, 120.0, 120.0]), ('e_mm', -0.1),
                  ('tam_than_deg', None), ('canh', None), ('s_mm', 0.0), ('goc_dan', {'a_mm': 1.0})):
        bad = _cp(TK)
        bad[k_] = v
        with pytest.raises(CU.UError):
            KH(tk=bad)
    bad = _cp(TK)
    del bad['canh']['sau']                                 # C8BU-R1-6: one edge (or three) does not show a distorted print
    with pytest.raises(CU.UError, match='four edges'):
        KH(tk=bad)
    bad = _cp(TK)
    bad['canh']['trai'] = [10.0, 16.0]                     # 3.6 deg: outside the small-angle domain
    with pytest.raises(CU.UError, match='small-angle'):
        KH(tk=bad)
    bad = _cp(TK)
    bad['tam_than_deg'] = 0.0                              # 0 only with the limit written
    with pytest.raises(CU.UError, match='tam_than_gioi_han'):
        KH(tk=bad)
    bad['tam_than_gioi_han'] = 'sai che tao tam-than ngoai ban ve chua bao'
    assert KH(tk=bad)['U_psi_phan_deg']['A1_goc_dan'] == pytest.approx(_expect(tk=bad)[1]['A1_goc_dan'])
    bad = _cp(TK)
    bad['cao_4_goc_mm'] = [2700.0, 120.0, 120.0, 120.0]   # above the camera
    with pytest.raises(CU.UError):
        KH(tk=bad)


def test_caliper_sheared_print_is_seen():
    # C8BU-R1-6 counter-example: x' = x + 0.01 y. The front / back edges stay parallel to the plate, the left / right
    # edges turn by atan(0.01); one edge read 0, the Kabsch angle of the four corners is about 0.286 deg
    import c7_measure as CM
    s = 96.4
    sh = np.array([[1.0, 0.01], [0.0, 1.0]])
    body = CM.car_corners_body(s / 1000) @ sh.T                 # corners FL, FR, BR, BL of the sheared print
    kab, *_ = CM.rigid_fit(CM.car_corners_body(s / 1000), body)
    tk = _cp(TK)
    # black edge to plate edge, plate edges at x = +-0.06, y = +-0.06 m (the a corner first, as the guide says)
    m = 1000.0
    FL, FR, BR, BL = body
    tk['canh'] = dict(truoc=[(0.06 - FL[0]) * m, (0.06 - FR[0]) * m], phai=[(FR[1] + 0.06) * m, (BR[1] + 0.06) * m],
                      sau=[(BL[0] + 0.06) * m, (BR[0] + 0.06) * m], trai=[(0.06 - FL[1]) * m, (0.06 - BL[1]) * m])
    tk['e_mm'] = 0.0
    k = KH(tk=tk)
    assert k['thuoc_kep']['goc_kabsch_deg'] == pytest.approx(math.degrees(kab), rel=0.02)
    assert abs(k['thuoc_kep']['goc_canh_deg']['trai']) < 1e-9 and abs(k['thuoc_kep']['goc_canh_deg']['phai']) < 1e-9
    assert k['U_psi_phan_deg']['A1_goc_dan'] >= abs(math.degrees(kab)) + 0.05 - 1e-9


def test_khoa_offset_used_by_the_scorer():
    # the scorer corrects by tag_offset = (o_g, dpsi) with o_g the geometric offset and dpsi the caliper angle: P2 keeps
    # the tolerance and the heading term, A1 what is left of the angle
    g = math.atan(0.2 / 96.4)                                # the angle of TK
    cam = _cp(CAM)
    cam['bo_cham']['tag_offset'] = [0.0004, -0.0003, g]       # o_g of TK
    k = KH(cam)
    a1 = math.degrees(math.atan(2 * 0.1 / 96.4) + g * g + 1e-4) + 0.05
    assert k['U_psi_phan_deg']['A1_goc_dan'] == pytest.approx(a1)
    assert k['U_p_phan_m'] == pytest.approx(_expect(cam=cam)[0])
    sumA = sum(k['U_psi_phan_deg'].values())
    u_g = 0.0005 + math.sqrt(2) * 0.0001                          # r = 0 in TK: no plate-angle term (R4-1)
    assert k['U_p_phan_m']['P2_lech_tam'] == pytest.approx(u_g + 0.0005 * math.radians(sumA))
    assert k['thuoc_kep']['o_hinh_hoc_m'] == pytest.approx([0.0004, -0.0003])
    cam['bo_cham']['tag_offset'][2] = -g                       # opposite sign: the mismatch 2 g is counted
    assert KH(cam)['U_psi_phan_deg']['A1_goc_dan'] == pytest.approx(a1 + 2 * math.degrees(g))


def test_geometric_offset_from_the_edges():
    # tag centre on the plate: x = (back - front) / 2, y = (right - left) / 2 (no plate size needed)
    tk = _cp(TK)
    tk['canh'] = dict(truoc=[9.0, 9.0], sau=[11.0, 11.0], phai=[10.5, 10.5], trai=[9.5, 9.5])
    tk['tam_tam_mm'] = [0.0, 0.0]
    xo = dict(XO, q_tag_m=[0.001, 0.0005])
    assert KH(tk=tk, xo=xo)['thuoc_kep']['o_hinh_hoc_m'] == pytest.approx([0.001, 0.0005])
    for k_ in ('tam_tam_mm', 'tam_tam_dung_sai_mm', 'ban_ve_nguon'):
        bad = _cp(TK)
        del bad[k_]
        with pytest.raises(CU.UError, match=k_):
            KH(tk=bad)


def test_turning_cross_check_refuses_a_disagreement():
    # C8BU-R3-2: turning in place is a cross-check of the geometry; a slip about an off-centre point shows as an offset
    # away from the drawing and stops the lock ('xem lai') instead of entering U
    with pytest.raises(CU.UError, match='xem lai'):
        KH(xo=dict(XO, q_tag_m=[0.004, -0.0003]))
    KH(xo=dict(XO, q_tag_m=[0.0012, -0.0003]))                # within u_g + du + ...: passes


def _geo_case(tc, r_meas, tol, tt_deg, e_mm=0.1):
    """Caliper file whose geometry gives o_g = tc + r_meas (mm, body / plate frame), all edges parallel."""
    tk = _cp(TK)
    rx, ry = r_meas
    tk['canh'] = dict(truoc=[50.0 - rx, 50.0 - rx], sau=[50.0 + rx, 50.0 + rx],
                      trai=[50.0 - ry, 50.0 - ry], phai=[50.0 + ry, 50.0 + ry])
    tk.update(tam_tam_mm=list(tc), tam_tam_dung_sai_mm=tol, tam_than_deg=tt_deg, e_mm=e_mm)
    return tk


def test_offset_part_bounds_car_pose():
    # C8BU-R1-3, R3-2, R4-1, R4-2: P2 bounds the body-centre error of c7_measure.car_pose for a true offset built from
    # real geometry: the drawing centre within its tolerance, the plate turned by up to the plate-body angle, each
    # caliper coordinate off by up to e, and the heading error within sum A
    import c7_measure as CM
    rng = np.random.default_rng(5)
    side = 0.1445
    worst = 0.0
    cases = [((40.0, 0.0), (-40.0, 0.0), 0.1, 1.0)]                   # Codex R4-1: o_g = 0, plate turned 1 deg
    cases += [(tuple(rng.uniform(-30, 30, 2)), tuple(rng.uniform(-30, 30, 2)), rng.uniform(0.05, 1.0),
               rng.uniform(0.05, 1.0)) for _ in range(300)]
    for tc, r_meas, tol, tt in cases:
        tk = _geo_case(tc, r_meas, tol, tt)
        o_g = (np.array(tc) + np.array(r_meas)) * 1e-3
        dpsi_u = rng.uniform(-0.02, 0.02)
        o_u = o_g + rng.uniform(-0.002, 0.002, 2)
        cam = _cp(CAM)
        cam['bo_cham']['tag_offset'] = [float(o_u[0]), float(o_u[1]), dpsi_u]
        cam['bo_cham']['side_m'] = side
        xo = dict(XO, q_tag_m=list(o_g), du_m=0.0007)
        _, pp, _, aa = UPP(cam, TBD, tk, xo)
        sumA = math.radians(sum(aa.values()))
        for sgn in (1, -1):
            ang = rng.uniform(0, 2 * math.pi)
            tc_t = np.array(tc) * 1e-3 + rng.uniform(0, tol * 1e-3) * np.array([math.cos(ang), math.sin(ang)])
            r_t = np.array(r_meas) * 1e-3 + rng.choice([-1, 1], 2) * 0.1e-3          # each coordinate off by e
            d = sgn * math.radians(tt)
            o_t = tc_t + np.array([math.cos(d) * r_t[0] - math.sin(d) * r_t[1],
                                   math.sin(d) * r_t[0] + math.cos(d) * r_t[1]])
            err = rng.choice([-1, 1]) * rng.uniform(0, sumA)
            B = rng.uniform(-1.5, 1.5, 2)
            psi_b = rng.uniform(-math.pi, math.pi)
            cb, sb = math.cos(psi_b), math.sin(psi_b)
            T = B + np.array([cb * o_t[0] - sb * o_t[1], sb * o_t[0] + cb * o_t[1]])
            psi_tag_seen = psi_b + dpsi_u + err                # car_pose: body heading = tag heading - dpsi_u
            ct, st = math.cos(psi_tag_seen), math.sin(psi_tag_seen)
            corners = T + CM.car_corners_body(side) @ np.array([[ct, st], [-st, ct]])
            res = CM.car_pose(_FloorCam0(), corners, side, 0.0, (0.0, 0.0), tag_offset=(o_u[0], o_u[1], dpsi_u))
            e_b = math.hypot(res['x'] - B[0], res['y'] - B[1])
            assert e_b <= pp['P2_lech_tam'] * (1 + 1e-9) + 1e-12, (tc, r_meas, tol, tt)
            worst = max(worst, e_b / pp['P2_lech_tam'])
    assert worst > 0.3


def test_cross_check_strength_recorded():
    # C8BU-R4-3, R5-1: the lock gives the slip size surely refused, room + u_g + du (a smaller slip may pass when the
    # drawing tolerance and the noise cancel it, Codex R5-1: tolerance 1 mm, slip 2 mm, gap 1 mm passes)
    k = KH()
    kx = k['kiem_xoay']
    u_g = 0.0005 + math.sqrt(2) * 0.0001
    assert kx['room_m'] > kx['gap_m']
    assert kx['truot_bat_chac_tu_m'] == pytest.approx(kx['room_m'] + u_g + XO['du_m'])
    tk = dict(_cp(TK), tam_tam_mm=[0.0, 0.0], tam_tam_dung_sai_mm=1.0, e_mm=0.1)
    tk['canh'] = dict(truoc=[10.0, 10.0], sau=[10.0, 10.0], phai=[10.0, 10.0], trai=[10.0, 10.0])
    xo = dict(XO, q_tag_m=[0.001, 0.0], du_m=0.0004)            # Codex R5-1 case: true centre -1 mm, slip 2 mm
    kx = KH(tk=tk, xo=xo)['kiem_xoay']                          # passes (gap 1 mm < room) ...
    assert kx['gap_m'] < kx['room_m'] and kx['truot_bat_chac_tu_m'] > 0.002   # ... and the lock says it may


def test_khoa_turning_input():
    for bad in (dict(XO, version='x'), dict(XO, q_tag_m=[0.003]), dict(XO, du_m=-1.0),
                dict(XO, q_tag_m=[0.003, None]), dict(XO, la=1)):
        with pytest.raises(CU.UError):
            KH(xo=bad)
    with pytest.raises(CU.UError, match='ga_ok'):               # C8BU-R1-9: a string is not a boolean
        KH(tb=dict(TBD, ga_ok='false'))
    bad = _cp(CAM)
    bad['A3']['quy_ve_ma_noc'] = False                          # C8BU-R1-4: a pair angle is not the car-tag bound
    with pytest.raises(CU.UError, match='quy_ve_ma_noc'):
        KH(bad)


class _FloorCam0:
    def pix2floor(self, pts, z):
        return np.asarray(pts, float)


def _tilt_case(c, h, rho_pt, psi, H, side=0.0964):
    import c7_measure as CM
    a = 0.7
    P = np.array([rho_pt * math.cos(a), rho_pt * math.sin(a)])
    cs, sn = math.cos(psi), math.sin(psi)
    corners = P + CM.car_corners_body(side) @ np.array([[cs, sn], [-sn, cs]])
    p = corners * (H / (H - c))[:, None]                     # central projection from (0, 0, H) of each corner
    r = CM.car_pose(_FloorCam0(), p, side, h / H, (0.0, 0.0))
    err_psi = abs((r['tag_psi'] - psi + math.pi) % (2 * math.pi) - math.pi)
    err_p = math.hypot(r['tag_x'] - P[0], r['tag_y'] - P[1])
    return err_psi, err_p


def test_tilt_parts_bound_car_pose():
    # A6 and P6 checked against c7_measure.car_pose with four corner heights (not a copy of the formula)
    rng = np.random.default_rng(11)
    H = 2.632
    worst = 0.0
    for _ in range(400):
        h = rng.uniform(0.08, 0.16)
        c = h + rng.uniform(-0.00045, 0.00045, 4)            # inside the domain (sum |c - mean| <= 2 mm)
        rho = rng.uniform(0.1, 1.8)
        psi = rng.uniform(-math.pi, math.pi)
        err_psi, err_p = _tilt_case(c, h, rho, psi, H)
        cam = _cp(CAM)
        cam['do_cao'].update(H_m=H, h_m=h, rho_max_m=rho)
        cam['day_doi'] = dict(dn_m=0.0, dzc_m=0.0, kappa=h / H)
        tk = dict(_cp(TK), cao_4_goc_mm=list(c * 1e3), e_mm=0.0)
        _, pp, _, aa = UPP(cam, TBD, tk, XO)
        assert err_psi <= math.radians(aa['A6_nghieng_ma']) * (1 + 1e-6) + 1e-12
        assert err_p <= pp['P6_do_cao'] * (1 + 1e-6) + 1e-12
        worst = max(worst, err_psi / math.radians(aa['A6_nghieng_ma']))
    assert worst > 0.3                                       # the bound is not loose by orders of magnitude


def test_tilt_domain_edges():
    # C8BU-R1-8: the edges of the enforced domain (sum |c - mean| = 2 mm, rho = 3 m, every heading) still bound
    H, h = 2.632, 0.12
    for c in (h + np.array([0.0005, 0.0005, -0.0005, -0.0005]), h + np.array([0.0005, -0.0005, -0.0005, 0.0005]),
              h + np.array([0.001, -0.000333, -0.000333, -0.000334])):
        cam = _cp(CAM)
        cam['do_cao'].update(H_m=H, h_m=h, rho_max_m=3.0)
        cam['day_doi'] = dict(dn_m=0.0, dzc_m=0.0, kappa=h / H)
        _, pp, _, aa = UPP(cam, TBD, dict(_cp(TK), cao_4_goc_mm=list(c * 1e3), e_mm=0.0), XO)
        for psi in np.linspace(-math.pi, math.pi, 73):
            err_psi, err_p = _tilt_case(c, h, 3.0, psi, H)
            assert err_psi <= math.radians(aa['A6_nghieng_ma']) * (1 + 1e-6)
            assert err_p <= pp['P6_do_cao'] * (1 + 1e-6)
    # C8BU-R2-3 counter-example: rho 0 but dn 16 mm: the tag may sit 16 mm from the true point under the camera
    c = h + np.array([0.0005, -0.0005, -0.0005, 0.0005])
    cam = _cp(CAM)
    cam['do_cao'].update(H_m=H, h_m=h, rho_max_m=0.0)
    cam['day_doi'] = dict(dn_m=0.016, dzc_m=0.0, kappa=h / H)
    _, pp, _, aa = UPP(cam, TBD, dict(_cp(TK), cao_4_goc_mm=list(c * 1e3), e_mm=0.0), XO)
    assert aa['A6_nghieng_ma'] > 0
    for psi in np.linspace(-math.pi, math.pi, 73):
        err_psi, err_p = _tilt_case(c, h, 0.016, psi, H)
        assert err_psi <= math.radians(aa['A6_nghieng_ma']) * (1 + 1e-6)
    cam = _cp(CAM)                                             # camera too low for the A6 form: refused
    cam['do_cao'].update(H_m=0.131, h_m=0.12)
    cam['day_doi'] = dict(dn_m=0.0, dzc_m=0.0, kappa=0.12 / 0.131)
    with pytest.raises(CU.UError, match='at least'):
        KH(cam, tk=dict(_cp(TK), cao_4_goc_mm=[120.5, 119.5, 119.5, 120.5]))
    cam = _cp(CAM)
    cam['do_cao']['rho_max_m'] = 3.1
    with pytest.raises(CU.UError, match='domain'):
        KH(cam)
    with pytest.raises(CU.UError, match='domain'):
        KH(tk=dict(_cp(TK), cao_4_goc_mm=[121.2, 119.0, 120.0, 120.0]))


def test_khoa_plumb_zc_required_and_bounded():
    # plumb point, zc reused from 16/11 (Liam 17/11 evening): both bounds required, zc bound below the camera height
    for k_ in ('dn_m', 'dzc_m', 'kappa'):
        bad = json.loads(json.dumps(CAM))
        del bad['day_doi'][k_]
        with pytest.raises(CU.UError, match='day_doi.' + k_):
            KH(bad)
    bad = json.loads(json.dumps(CAM))
    bad['day_doi']['dn_m'] = -0.001
    with pytest.raises(CU.UError, match='day_doi.dn_m'):
        KH(bad)
    bad['day_doi'] = dict(dn_m=0.0, dzc_m=2.5, kappa=0.12 / 2.6)    # H - dzc not above h
    with pytest.raises(CU.UError, match='not above the tag height'):
        KH(bad)
    bad['day_doi'] = dict(dn_m=0.0, dzc_m=0.01, kappa=0.12 / 2.632)   # C8BS-R7-3: H is not the zc of the scoring
    with pytest.raises(CU.UError, match='not do_cao.h_m / do_cao.H_m'):
        KH(bad)
    z = json.loads(json.dumps(CAM))
    z['day_doi'] = dict(dn_m=0.0, dzc_m=0.0, kappa=0.12 / 2.6)     # no plumb / zc error: the part is zero
    k0 = KH(z)
    k1 = KH()
    assert k0['U_p_phan_m']['P5_day_doi_zc'] == 0.0
    # the zc bound also lowers the camera height used in P6, A6 (C8BU-R1-8), and A6 enters P2 only through |o_u| = 0
    dP = {x: k1['U_p_phan_m'][x] - k0['U_p_phan_m'][x] for x in CU.P_PARTS}
    dA = {x: k1['U_psi_phan_deg'][x] - k0['U_psi_phan_deg'][x] for x in CU.A_PARTS}
    assert {x for x, v in dP.items() if v != 0} == {'P5_day_doi_zc', 'P6_do_cao'} and dP['P6_do_cao'] > 0
    assert {x for x, v in dA.items() if v != 0} == {'A6_nghieng_ma'} and dA['A6_nghieng_ma'] > 0
    assert k1['U_p_m'] - k0['U_p_m'] == pytest.approx(1.25 * sum(dP.values()))


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
    _, parts, _, _ = UPP(cam, TBD, TK, XO)
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


def test_check_khoa_refuses_old_locks():
    # C8BS-R7-4: a lock written before the plumb / zc part (c8b_khoa_v1, cc22feb) is refused
    k = KH()
    CU.check_khoa(json.loads(json.dumps(CU._clean(k))))
    # v2: parts named after tests 1, 2 (b740f2a); v3: no U_moc, no A7 (f295020; proposal v3 section 4)
    for ver in ('c8b_khoa_v1', 'c8b_khoa_v2', 'c8b_khoa_v3'):
        with pytest.raises(CU.UError, match='parts P1-P7, A1-A7'):
            CU.check_khoa(dict(k, version=ver))
    old = dict(k, U_p_phan_m={x: v for x, v in k['U_p_phan_m'].items() if x != 'P5_day_doi_zc'})
    with pytest.raises(CU.UError, match='parts P1-P7, A1-A7'):
        CU.check_khoa(old)
    for drop in ('A5_nhieu_khung', 'A7_moc_dich'):          # a lock without A7 (a v3 lock renamed) is refused
        old = dict(k, U_psi_phan_deg={x: v for x, v in k['U_psi_phan_deg'].items() if x != drop})
        with pytest.raises(CU.UError, match='parts P1-P7, A1-A7'):
            CU.check_khoa(old)
    old = dict(k, U_p_phan_m=dict(k['U_p_phan_m'], phep1_khung_max=0.002))
    with pytest.raises(CU.UError, match='parts P1-P7, A1-A7'):
        CU.check_khoa(old)


# ── c8b_u: mark-shift bound (proposal v3, Liam 18/11 "Duyet ban 3 (Recommended)") ──────────────────────────────────

def test_moc_parts_used_and_kept_before_margin():
    # P1 = max(clip, per frame) + U_moc_xy_m; A7 = U_moc_psi_deg; nguong + 2 u_so is information only
    k = KH()
    assert k['U_p_phan_m']['P1_anh_san'] == pytest.approx(0.002 + 0.0021)
    assert k['U_psi_phan_deg']['A7_moc_dich'] == 0.07
    big = _cp(CAM)
    big['moc_dich'].update(nguong_m=0.5, u_so_m=0.5)
    assert KH(big)['U_p_m'] == k['U_p_m']
    cam = _cp(CAM)
    cam['moc_dich'].update(U_moc_xy_m=0.0048, U_moc_psi_deg=0.57)
    k2 = KH(cam)
    assert k2['U_p_phan_m']['P1_anh_san'] - k['U_p_phan_m']['P1_anh_san'] == pytest.approx(0.0027)
    assert k2['U_psi_deg'] - k['U_psi_deg'] == pytest.approx(1.25 * 0.5)
    # the lock keeps U_moc BEFORE the margin, the method, the session result and the sources
    assert k2['moc_dich']['U_moc_xy_m'] == 0.0048 and k2['moc_dich']['U_moc_psi_deg'] == 0.57
    assert k2['moc_dich']['tien_de_buoi'] == 'dat' and k2['moc_dich']['nguon']['lich'] == MOC_SHA['lich']
    CU.check_khoa(json.loads(json.dumps(CU._clean(k2))))


def test_moc_a7_enters_p2_through_the_offset():
    # proposal v3 section 3: A7 is in the A sum of P2 (|o_u| x sum A, the heading error turns the offset vector)
    cam = _cp(CAM)
    cam['bo_cham']['tag_offset'] = [0.03, -0.04, 0.0]                  # |o_u| = 50 mm
    cam2 = _cp(cam)
    cam2['moc_dich']['U_moc_psi_deg'] = 1.07
    d = KH(cam2)['U_p_phan_m']['P2_lech_tam'] - KH(cam)['U_p_phan_m']['P2_lech_tam']
    assert d == pytest.approx(0.05 * math.radians(1.0))


def test_moc_refuses_v2_camera_file():
    v2 = _cp(CAM)
    v2['version'] = 'c8_cam_u_vao_v2'
    with pytest.raises(CU.UError, match='version is not c8_cam_u_vao_v3'):
        CU.check_cam(v2)
    with pytest.raises(CU.UError, match='version is not c8_cam_u_vao_v3'):
        KH(v2)
    # a v2-shaped moc_dich (nguong, u_so only) under the v3 name is refused for the missing numbers
    old = _cp(CAM)
    old['moc_dich'] = {"nguong_m": 0.001, "u_so_m": 0.0004, "moc_loai": []}
    with pytest.raises(CU.UError, match='moc_dich.U_moc_xy_m'):
        CU.check_cam(old)


def test_moc_missing_fields_refused():
    for fld in ('phuong_phap_duyet', 'tien_de_buoi', 'nguon'):
        bad = _cp(CAM)
        del bad['moc_dich'][fld]
        with pytest.raises(CU.UError, match=fld):
            CU.check_cam(bad)
        with pytest.raises(CU.UError, match=fld):
            KH(bad)
    for fld in ('frames', 'lich', 'tap_moc', 'u_so', 'session', 'car_pose', 'bo_lay_mau'):
        bad = _cp(CAM)
        del bad['moc_dich']['nguon'][fld]
        with pytest.raises(CU.UError, match=fld):        # not through KH: its sync would put car_pose back
            KHG({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, bad, MT())
    for fld, val in (('lich', 'A' * 64), ('u_so', 'a' * 63), ('session', None), ('bo_lay_mau', ' ')):
        bad = _cp(CAM)
        bad['moc_dich']['nguon'][fld] = val
        with pytest.raises(CU.UError, match=fld):
            KH(bad)
    bad = _cp(CAM)
    bad['moc_dich']['nguon']['frames'] = {}
    with pytest.raises(CU.UError, match='frames'):
        KH(bad)


def test_moc_method_and_session_needed():
    # "Tu choi khoa" (Liam 18/11): no approved method, or a session that did not pass: the lock is refused; the camera
    # file alone is still well formed (its writers may export a session that did not pass)
    for val in ('', '   '):
        bad = _cp(CAM)
        bad['moc_dich']['phuong_phap_duyet'] = val
        CU.check_cam(bad)
        with pytest.raises(CU.UError, match='phuong_phap_duyet empty'):
            KH(bad)
    for val in ('khong_dat', 'chua', 'DAT', ''):
        bad = _cp(CAM)
        bad['moc_dich']['tien_de_buoi'] = val
        CU.check_cam(bad)
        with pytest.raises(CU.UError, match='tien_de_buoi'):
            KH(bad)
    bad = _cp(CAM)
    bad['moc_dich']['tien_de_buoi'] = True
    with pytest.raises(CU.UError, match='tien_de_buoi'):
        CU.check_cam(bad)


def test_moc_sha_of_the_files_given():
    # C8BM-R2-2: every SHA in moc_dich.nguon must be the SHA of the file khoa is given
    with pytest.raises(CU.UError, match='files of the final scoring'):
        CU.khoa(GIOS({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}), TBD, sync(CAM), TK, XO, gio_sha={'a': _gsha('a')})
    for kind in ('lich', 'tap_moc', 'u_so', 'session'):
        with pytest.raises(CU.UError, match=f'nguon.{kind} '):
            KH(moc_tep=MT(**{kind: 'e' * 64}))
        mt = MT()
        del mt[kind]
        with pytest.raises(CU.UError, match=f'file {kind} not given'):
            KH(moc_tep=mt)
    with pytest.raises(CU.UError, match=r'nguon.frames.a is not'):
        KH(moc_tep=MT(frames={'a': 'f' * 64}))
    with pytest.raises(CU.UError, match='frames files given'):
        KH(moc_tep=MT(('a', 'b')))
    # the runs of U_moc against the C8 clock files: a run outside them, or an ok run missing, refused
    gios = {'a': gio_d('O1-F05', 3e-4, 0.01, 0.0), 'b': gio_d('O1-F05', 3e-4, 0.01, 0.0),
            'c': gio_d('O1-F05', 3e-4, 0.01, 0.0, ok=False)}
    # (KCJ3-1: every scored run is in luot_cham; a run without U_moc frames must have a failed clock)
    with pytest.raises(CU.UError, match=r"run b has no U_moc frames .*ok is not false"):
        KHG(gios, sync(CAM, 'a', vang='bc'), MT('a'))
    with pytest.raises(CU.UError, match=r"clock file missing for \['z'\]"):
        KHG(gios, sync(CAM, 'abz', vang='c'), MT('abz'))
    KHG(gios, sync(CAM, 'ab', vang='c'), MT('ab'))            # c not ok: may lack frames
    KHG(gios, sync(CAM, 'abc'), MT('abc'))


def test_khoa_kcj3_1_every_scored_run_needs_its_clock_file():
    # Codex KCJ3-1 (KIEMCHEO J round 3): two scored runs, drift 3e-4 and 6e-4; a lock given the clock file of run a only
    # took 1.25 x 300 = 375 ppm instead of 750 ppm. Now refused: luot_cham names b, its clock file is missing
    gios = {'a': gio_d('O1-F05', 3e-4, 0.01, 0.0), 'b': gio_d('O1-F05', 6e-4, 0.01, 0.0)}
    cam = sync(CAM, 'ab')
    with pytest.raises(CU.UError, match=r"clock file missing for \['b'\]"):
        KHG({'a': gios['a']}, cam, MT('ab'))
    # the counterexample itself: b written as a run without U_moc and its clock file left out (the old check passed it
    # and locked 375 ppm)
    with pytest.raises(CU.UError, match=r"clock file missing for \['b'\]"):
        KHG({'a': gios['a']}, sync(CAM, 'a', vang='b'), MT('a'))
    # claiming b has no U_moc frames does not help while its clock is ok
    with pytest.raises(CU.UError, match=r"run b has no U_moc frames"):
        KHG(gios, sync(CAM, 'a', vang='b'), MT('a'))
    k = KHG(gios, cam, MT('ab'))
    assert k['troi_ppm'] == pytest.approx(750.0)
    # a camera file without the KCJ3-1 fields (older writer) is refused; luot_cham must agree with nguon.frames
    old = _cp(cam)
    del old['moc_dich']['luot_cham']
    with pytest.raises(CU.UError, match='luot_cham'):
        KHG(gios, old, MT('ab'))
    bad = _cp(cam)
    bad['moc_dich']['luot_cham']['b'] = 'e' * 64
    with pytest.raises(CU.UError, match=r'nguon.frames.b vs moc_dich.luot_cham.b'):
        KHG(gios, bad, MT('ab'))
    bad = _cp(cam)
    bad['moc_dich']['luot_vang_u_moc'] = ['b']
    with pytest.raises(CU.UError, match='luot_vang_u_moc'):
        KHG(gios, bad, MT('ab'))


def test_moc_car_pose_set_up_must_match():
    # U_moc computed with another tag_offset, tag side or tag height than the scorer's: refused (also check_cam)
    for path, val in (('tag_offset', [0.001, 0.0, 0.0]), ('side_m', 0.1445), ('h_m', 0.121)):
        bad = _cp(CAM)
        bad['moc_dich']['nguon']['car_pose'][path] = val
        with pytest.raises(CU.UError, match=f'car_pose.{path}'):
            CU.check_cam(bad)
        with pytest.raises(CU.UError, match=f'car_pose.{path}'):
            KHG({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, bad, MT())
    bad = _cp(CAM)
    bad['moc_dich']['nguon']['car_pose']['extra'] = 1
    with pytest.raises(CU.UError, match='car_pose needs exactly'):
        CU.check_cam(bad)


def test_check_khoa_moc_block():
    k = json.loads(json.dumps(CU._clean(KH())))
    CU.check_khoa(k)
    bad = dict(k)
    del bad['moc_dich']
    with pytest.raises(CU.UError, match='no block moc_dich'):
        CU.check_khoa(bad)
    for fld, val, txt in (('tien_de_buoi', 'khong_dat', 'session that passed'),
                          ('phuong_phap_duyet', '', 'approved method'),
                          ('U_moc_xy_m', -1.0, 'U_moc_xy_m'), ('U_moc_psi_deg', None, 'U_moc_psi_deg'),
                          ('U_moc_psi_deg', 0.08, 'A7_moc_dich'), ('U_moc_xy_m', 0.0042, 'below')):
        bad = json.loads(json.dumps(k))
        bad['moc_dich'][fld] = val
        with pytest.raises(CU.UError, match=txt):
            CU.check_khoa(bad)


def test_khoa_phep5_and_one_frame():
    tb = json.loads(json.dumps(TBD))
    tb['B']['shutter']['use_alpha'] = False
    k = KH(tb=tb, phep5=[('d1', 0.003), ('d2', -0.02)])
    assert k['man_trap_kieu'] == 'mot_khung' and k['man_trap'] is None
    assert k['phep5']['ket_qua'] == 'xem_lai' and k['phep5']['vuot'] == ['d2']


def _plan():
    return {'1': dict(task='O1-F05', controller='PID'), '2': dict(task='O1-F05', controller='LADRC'),
            '3': dict(task='O1-F08', controller='PID'), '4': dict(task='O1-F08', controller='LADRC')}


SCRIPT = dict(commit='abc1234ffff', dirty=False)
SCORER = dict(commit='abc1234ffff', dirty=False, nut=[dict(commit='abc1234ffff', dirty=False, file='tools/c7_score.py',
                                                            tracked=True)])


def _pair(rid, task, half, outcome, ut, nodes=None, gio_ok=True, ctrl='PID'):
    nodes = nodes or [1.0 + 0.001 * i for i in range(len(ut) or 1)]
    g = dict(gio_d(task, 1e-4, 0.01, 0.0, ok=gio_ok, half=half), nodes_s=nodes, script=SCRIPT,
             ket_luan='ok' if gio_ok else 'hong_he_do', controller=ctrl)
    sha = _gsha(rid)
    g['sha256'] = dict(frames=_fsha(rid), esp32='0' * 64)       # the clock file names its frames (c8b_gio)
    l = dict(version='c8b_luoi_v1', gio_sha256=sha, task=task, nodes_s=nodes, so_do=[], outcome=outcome,
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


LOCK_RUNS = ('r1', 'r2', 'r3', 'r4', 'r3a', 'r1b', 'r9', 'r1x', 'r1y')


def _lock(U_p, U_psi):
    """A lock that check_khoa accepts (C8BU-R1-7): all of U_p in P1, all of U_psi in A1."""
    return dict(version=CU.VERSION, le=CU.LE, U_p_m=U_p, U_psi_deg=U_psi,
                U_p_phan_m={x: (U_p / CU.LE if x == 'P1_anh_san' else 0.0) for x in CU.P_PARTS},
                U_psi_phan_deg={x: (U_psi / CU.LE if x == 'A1_goc_dan' else 0.0) for x in CU.A_PARTS},
                moc_dich=dict(U_moc_xy_m=0.0, U_moc_psi_deg=0.0, phuong_phap_duyet='thu', tien_de_buoi='dat',
                              nguon=dict(CAM['moc_dich']['nguon'], frames={r: _fsha(r) for r in LOCK_RUNS})),
                luot_c8={r: dict(gio_sha256=_gsha(r), frames_sha256=_fsha(r)) for r in LOCK_RUNS})


K = _lock(0.004, 2.0)


def test_check_khoa_totals_and_parts():
    # C8BU-R1-7: positive parts with totals set to 0, a wrong margin, a bad part: refused
    CU.check_khoa(K)
    for bad in (dict(K, U_p_m=0.0), dict(K, U_psi_deg=0.0), dict(K, le=1.0),
                dict(K, U_p_phan_m=dict(K['U_p_phan_m'], P7_nhieu_khung=-1e-4)),
                dict(K, U_psi_phan_deg=dict(K['U_psi_phan_deg'], A3_huong_vung='0.1'))):
        with pytest.raises(CU.UError):
            CU.check_khoa(bad)
    with pytest.raises(CU.UError, match='U_p_m'):
        CU.check_khoa(dict(K, U_p_m=0.0))


def _scoring(d, tag, cam, runs=('r1', 'r2', 'r3', 'r4'), body=None, clip_tag=None, reverse=False):
    """Files of one scoring in directory d/tag, chained as c7_measure writes them (CODE_CAM feat/c8-33-moc e8af52b):
    session.json, u_so.json, tap_moc.json, lich.json (session_sha256, u_so_sha256, tap_moc_file_sha256), frames.csv and
    track.json of each run (frames_sha256, session_sha256, u_so_sha256, lich_sha256, cham_lai 2, tag_offset), the clock
    file of each run naming its frames (sha256.frames). Returns (paths, camera file consistent with them)."""
    import c8_u_imu as cu
    p = d / tag
    p.mkdir()
    off = cam['bo_cham']['tag_offset'] or [0.0, 0.0, 0.0]
    (p / 'session.json').write_text(json.dumps(dict(car_side_m=cam['bo_cham']['side_m'], tag_h_m=cam['do_cao']['h_m'],
                                                    kappa=cam['day_doi']['kappa'], buoi=tag)), encoding='utf-8')
    (p / 'u_so.json').write_text(json.dumps({'buoi': tag}), encoding='utf-8')
    (p / 'tap_moc.json').write_text(json.dumps({'C1': [[1, 2, 3]], 'buoi': tag}), encoding='utf-8')
    sh = {k: cu.sha256(str(p / f'{k}.json')) for k in ('session', 'u_so', 'tap_moc')}
    body = body or hashlib.sha256(('lich ' + tag).encode()).hexdigest()
    ct = clip_tag or tag                                             # the clips (name, dets) this schedule was built on
    clips = [dict(name=f'clip_{r}', kind='luot', dets_sha256=_dsha(r + ct), n=100) for r in runs]
    clips = clips[::-1] if reverse else clips
    (p / 'lich.json').write_text(json.dumps(dict(lich_sha256=body, session_sha256=sh['session'], u_so_sha256=sh['u_so'],
                                                 tap_moc_file_sha256=sh['tap_moc'], hoi_tu=True, clips=clips)),
                                 encoding='utf-8')
    sh['lich'] = cu.sha256(str(p / 'lich.json'))
    out = dict(files={k: str(p / f'{k}.json') for k in ('lich', 'tap_moc', 'u_so', 'session')}, frames={}, track={},
               gio={}, luoi={}, slot={})
    fr_sha = {}
    for slot, rid, g, l, _ in _luot():
        if rid not in runs:
            continue
        fp = p / f'frames_{rid}.csv'
        fp.write_text(f't_s,x\n0,{rid}{tag}\n', encoding='utf-8')
        fr_sha[rid] = cu.sha256(str(fp))
        tp = p / f'track_{rid}.json'
        tp.write_text(json.dumps(dict(frames_sha256=fr_sha[rid], session_sha256=sh['session'], u_so_sha256=sh['u_so'],
                                      lich_sha256=body, cham_lai=2, thu=False, tag_offset=list(off),
                                      run=f'clip_{rid}', dets_sha256=_dsha(rid + (clip_tag or tag)),
                                      lich_file_sha256=sh['lich'])),
                      encoding='utf-8')
        gp = p / f'gio_{rid}.json'
        gp.write_text(json.dumps(dict(g, sha256=dict(frames=fr_sha[rid], esp32='0' * 64))), encoding='utf-8')
        lp = p / f'luoi_{rid}.json'
        lp.write_text(json.dumps(dict(l, gio_sha256=cu.sha256(str(gp)))), encoding='utf-8')
        out['frames'][rid], out['track'][rid], out['gio'][rid], out['luoi'][rid] = str(fp), str(tp), str(gp), str(lp)
        out['slot'][rid] = slot
    c = _cp(cam)
    c['moc_dich']['nguon'].update(frames=fr_sha, **{k: sh[k] for k in ('lich', 'tap_moc', 'u_so', 'session')})
    c['moc_dich'].update(luot_cham=dict(fr_sha), luot_vang_u_moc=[])          # KCJ3-1 (c7_kiem_cheo 565c0b7)
    (p / 'cam.json').write_text(json.dumps(c), encoding='utf-8')
    out['cam'] = str(p / 'cam.json')
    return out, c


def _khoa_args(d, S, gio=None, track=None, files=None):
    for name, obj in (('tb.json', TBD), ('tk.json', TK), ('xo.json', XO)):
        if not (d / name).exists():
            (d / name).write_text(json.dumps(obj), encoding='utf-8')
    gio, track, files = gio or S['gio'], track or S['track'], files or S['files']
    a = ['khoa']
    for r in sorted(S['frames']):
        a += ['--gio', f'{r}={gio[r]}', '--moc-frames', f'{r}={S["frames"][r]}', '--moc-track', f'{r}={track[r]}']
    a += ['--tb', str(d / 'tb.json'), '--cam-u', S['cam'], '--thuoc-kep', str(d / 'tk.json'), '--xoay',
          str(d / 'xo.json'), '--moc-lich', files['lich'], '--moc-tap-moc', files['tap_moc'], '--moc-u-so',
          files['u_so'], '--moc-session', files['session']]
    return a


def test_moc_cli_export_lock(tmp_path):
    # C8BM-R2-4: camera file (as exported) -> khoa (CLI, real files and SHA); the gate file is gate_c8.py ghep's (cong
    # removed 20/11)
    import c8_u_imu as cu
    d = tmp_path
    S, cam = _scoring(d, 'A', sync(CAM))
    kp = d / 'khoa.json'
    assert CU.main(_khoa_args(d, S) + ['--out', str(kp)]) == 0
    k = json.loads(kp.read_text(encoding='utf-8'))
    assert k['version'] == 'c8b_khoa_v4' and k['moc_dich']['U_moc_xy_m'] == 0.0021
    assert {n['file'] for n in k['nguon']} >= {'lich.json', 'u_so.json', 'frames_r1.csv', 'track_r1.json'}
    assert k['luot_c8']['r2'] == dict(gio_sha256=cu.sha256(S['gio']['r2']), frames_sha256=cu.sha256(S['frames']['r2']))
    assert k['moc_dich']['U_moc_psi_deg'] == 0.07
    assert k['moc_dich']['nguon']['bo_lay_mau'] == cam['moc_dich']['nguon']['bo_lay_mau']
    CU.check_khoa(k)
    with pytest.raises(CU.UError, match='parts P1-P7, A1-A7'):          # a v3 lock (before this proposal) is refused
        CU.check_khoa(dict(k, version='c8b_khoa_v3'))


def test_moc_cli_sources_of_another_scoring_refused(tmp_path):
    # C8BU4-R1-1, R1-2 (Codex): an older scoring A and the final scoring B of the SAME runs; the camera file and every
    # --moc-* file of A agree among themselves, but the clock files are B's: refused, no lock. Each link alone too.
    import c8_u_imu as cu
    d = tmp_path
    A, camA = _scoring(d, 'A', sync(CAM))
    B, camB = _scoring(d, 'B', sync(CAM))
    for name, args in (('clock files of B', _khoa_args(d, A, gio=B['gio'])),
                       ('track.json of B', _khoa_args(d, A, track=B['track'])),
                       ('lich of B', _khoa_args(d, A, files=dict(A['files'], lich=B['files']['lich']))),
                       ('session of B', _khoa_args(d, A, files=dict(A['files'], session=B['files']['session'])))):
        out = d / f'k_{name.replace(" ", "_")}.json'
        assert CU.main(args + ['--out', str(out)]) == 2, name
        assert not out.exists()
    # the set-up used: a scorer offset written in both places of the camera file but not the one of the tracks
    off = _cp(camA)
    off['bo_cham']['tag_offset'] = [0.03, -0.04, 0.0]
    off['moc_dich']['nguon']['car_pose']['tag_offset'] = [0.03, -0.04, 0.0]
    (d / 'A' / 'cam_off.json').write_text(json.dumps(off), encoding='utf-8')
    S = dict(A, cam=str(d / 'A' / 'cam_off.json'))
    assert CU.main(_khoa_args(d, S) + ['--out', str(d / 'k_off.json')]) == 2
    # tag side, tag height, kappa of the session differ from the camera file (in both of its places)
    for key, path in (('side_m', ('bo_cham', 'side_m')), ('h_m', ('do_cao', 'h_m'))):
        c = _cp(camA)
        v = c[path[0]][path[1]] + 0.001
        c[path[0]][path[1]] = v
        c['moc_dich']['nguon']['car_pose'][key] = v
        if key == 'h_m':
            c['day_doi']['kappa'] = v / c['do_cao']['H_m']
        (d / 'A' / f'cam_{key}.json').write_text(json.dumps(c), encoding='utf-8')
        S = dict(A, cam=str(d / 'A' / f'cam_{key}.json'))
        assert CU.main(_khoa_args(d, S) + ['--out', str(d / f'k_{key}.json')]) == 2, key
    # B alone is accepted
    assert CU.main(_khoa_args(d, B) + ['--out', str(d / 'kB.json')]) == 0
    assert cu.sha256(B['frames']['r1']) == json.loads((d / 'kB.json').read_text())['luot_c8']['r1']['frames_sha256']


def test_moc_cli_same_schedule_body_other_clips_refused(tmp_path):
    # C8BU4-R2-1 (Codex): the schedule hash covers the body, not the clips. Scoring A, and a schedule B with the SAME
    # body hash built on other clips (other detections); camera file naming B's files: refused (the tracks' clips are
    # not B's)
    import c8_u_imu as cu
    d = tmp_path
    A, _ = _scoring(d, 'A', sync(CAM))
    body_a = json.loads(open(A['files']['lich'], encoding='utf-8').read())['lich_sha256']
    B, _ = _scoring(d, 'B', sync(CAM), body=body_a, clip_tag='B')
    files = dict(A['files'], lich=B['files']['lich'], tap_moc=B['files']['tap_moc'], u_so=B['files']['u_so'],
                 session=B['files']['session'])
    c = json.loads(open(A['cam'], encoding='utf-8').read())       # camera file of A naming B's schedule files
    c['moc_dich']['nguon'].update(**{k: cu.sha256(files[k]) for k in ('lich', 'tap_moc', 'u_so', 'session')})
    (d / 'A' / 'cam_B.json').write_text(json.dumps(c), encoding='utf-8')
    # tracks of A named B's session / u_so: only the clips tell them apart
    for r in A['track']:
        t = json.loads(open(A['track'][r], encoding='utf-8').read())
        t.update(session_sha256=cu.sha256(files['session']), u_so_sha256=cu.sha256(files['u_so']))
        open(A['track'][r], 'w', encoding='utf-8').write(json.dumps(t))
    assert CU.main(_khoa_args(d, dict(A, cam=str(d / 'A' / 'cam_B.json')), files=files)
                   + ['--out', str(d / 'k.json')]) == 2
    assert not (d / 'k.json').exists()


def test_moc_cli_schedule_clips_reordered_refused(tmp_path):
    # C8BU4-R3-1 (Codex): a schedule with the same body, the same clips (names, detections) in another order: the clip
    # index of an event then names another clip. Tracks of A, schedule file of B: refused by lich_file_sha256
    import c8_u_imu as cu
    d = tmp_path
    A, _ = _scoring(d, 'A', sync(CAM))
    body_a = json.loads(open(A['files']['lich'], encoding='utf-8').read())['lich_sha256']
    B, _ = _scoring(d, 'B', sync(CAM), body=body_a, clip_tag='A', reverse=True)
    files = dict(A['files'], lich=B['files']['lich'], tap_moc=B['files']['tap_moc'], u_so=B['files']['u_so'],
                 session=B['files']['session'])
    c = json.loads(open(A['cam'], encoding='utf-8').read())
    c['moc_dich']['nguon'].update(**{k: cu.sha256(files[k]) for k in ('lich', 'tap_moc', 'u_so', 'session')})
    (d / 'A' / 'cam_B.json').write_text(json.dumps(c), encoding='utf-8')
    for r in A['track']:                       # every other link of the tracks made to agree with B
        t = json.loads(open(A['track'][r], encoding='utf-8').read())
        t.update(session_sha256=cu.sha256(files['session']), u_so_sha256=cu.sha256(files['u_so']))
        open(A['track'][r], 'w', encoding='utf-8').write(json.dumps(t))
    args = _khoa_args(d, dict(A, cam=str(d / 'A' / 'cam_B.json')), files=files)
    assert CU.main(args + ['--out', str(d / 'k.json')]) == 2
    assert not (d / 'k.json').exists()
    for r in A['track']:                       # with the tracks naming B's file it passes: only that field tells
        t = json.loads(open(A['track'][r], encoding='utf-8').read())
        t['lich_file_sha256'] = cu.sha256(files['lich'])
        open(A['track'][r], 'w', encoding='utf-8').write(json.dumps(t))
    assert CU.main(args + ['--out', str(d / 'k2.json')]) == 0


def test_moc_api_lich_file_and_clip_names():
    # C8BU4-R3-1: a track without lich_file_sha256 (CAM before the field) or with another: refused;
    # C8BU4-R3-2: a clip name not a string: refused with UError (not a TypeError)
    for val in (None, 'f' * 64):
        mt = MT()
        mt['track']['a']['lich_file_sha256'] = val
        with pytest.raises(CU.UError, match='lich_file_sha256'):
            KH(moc_tep=mt)
    for val in ([], {}, '', None):
        mt = MT()
        mt['lich_doc']['clips'][0]['name'] = val
        with pytest.raises(CU.UError, match='clip name'):
            KH(moc_tep=mt)


def test_moc_api_track_clip_in_schedule():
    # C8BU4-R2-1: each track's clip (run, dets_sha256) must be a clip of lich.json, a different clip per run
    for key, val, txt in (('run', 'clip_x', 'not a clip of lich.json'), ('run', None, 'not a clip of lich.json'),
                          ('dets_sha256', 'f' * 64, 'dets_sha256'), ('dets_sha256', None, 'dets_sha256')):
        mt = MT()
        mt['track']['a'][key] = val
        with pytest.raises(CU.UError, match=txt):
            KH(moc_tep=mt)
    mt = MT()
    del mt['lich_doc']['clips']
    with pytest.raises(CU.UError, match='no clips list'):
        KH(moc_tep=mt)
    mt = MT()
    mt['lich_doc']['clips'] = mt['lich_doc']['clips'] * 2
    with pytest.raises(CU.UError, match='twice'):
        KH(moc_tep=mt)
    gios = {'a': gio_d('O1-F05', 3e-4, 0.01, 0.0), 'b': gio_d('O1-F05', 3e-4, 0.01, 0.0)}
    mt = MT('ab', sync(CAM, 'ab'))
    mt['track']['b'].update(run='clip_a', dets_sha256=_dsha('a'))              # two runs, one clip
    with pytest.raises(CU.UError, match='same clip'):
        KHG(gios, sync(CAM, 'ab'), mt)
    KHG(gios, sync(CAM, 'ab'))


def test_lock_frames_consistent_inside(tmp_path):
    # C8BU4-R2-2 (Codex): a lock edited by hand to drop the U_moc frames of an ok run (both copies), or to name other
    # frames in both copies, is consistent inside (only the clock file shows it: the gate's check since cong was
    # removed 20/11); a malformed frames sha is refused
    d = tmp_path
    A, _ = _scoring(d, 'A', sync(CAM))
    kp = d / 'kA.json'
    assert CU.main(_khoa_args(d, A) + ['--out', str(kp)]) == 0
    k = json.loads(kp.read_text(encoding='utf-8'))
    for how in ('drop', 'other'):
        b = json.loads(json.dumps(k))
        if how == 'drop':
            del b['moc_dich']['nguon']['frames']['r1']
            b['luot_c8']['r1']['frames_sha256'] = None
        else:
            b['moc_dich']['nguon']['frames']['r1'] = b['luot_c8']['r1']['frames_sha256'] = 'f' * 64
        CU.check_khoa(b)                                   # consistent inside: only the clock file shows it
    b = json.loads(json.dumps(k))
    b['luot_c8']['r1']['frames_sha256'] = 'xyz'
    with pytest.raises(CU.UError, match='frames_sha256'):
        CU.check_khoa(b)


def test_check_khoa_car_pose_values():
    # C8BU4-R2-3: the lock's car_pose values are checked, not only their names
    k = json.loads(json.dumps(CU._clean(KH())))
    for key, val in (('tag_offset', 'sai'), ('tag_offset', [0.0, 0.0]), ('tag_offset', [0.0, None, 0.0]),
                     ('side_m', -1), ('side_m', 0), ('h_m', 'sai'), ('h_m', -0.1)):
        b = json.loads(json.dumps(k))
        b['moc_dich']['nguon']['car_pose'][key] = val
        with pytest.raises(CU.UError, match=f'car_pose.{key}'):
            CU.check_khoa(b)


def test_check_khoa_sources_shape_and_runs():
    # C8BU4-R1-4: the lock's sources checked like the camera file's, and tied to luot_c8
    k = json.loads(json.dumps(CU._clean(KH())))
    CU.check_khoa(k)
    for fld in ('frames', 'lich', 'tap_moc', 'u_so', 'session', 'car_pose', 'bo_lay_mau'):
        bad = json.loads(json.dumps(k))
        del bad['moc_dich']['nguon'][fld]
        with pytest.raises(CU.UError, match=fld):
            CU.check_khoa(bad)
    bad = json.loads(json.dumps(k))
    bad['moc_dich']['nguon'] = {}
    with pytest.raises(CU.UError, match='lacks'):
        CU.check_khoa(bad)
    bad = json.loads(json.dumps(k))
    bad['moc_dich']['nguon']['lich'] = 'x' * 64
    with pytest.raises(CU.UError, match='nguon.lich'):
        CU.check_khoa(bad)
    bad = json.loads(json.dumps(k))
    del bad['luot_c8']
    with pytest.raises(CU.UError, match='no luot_c8'):
        CU.check_khoa(bad)
    bad = json.loads(json.dumps(k))
    bad['moc_dich']['nguon']['frames']['zz'] = _fsha('zz')
    with pytest.raises(CU.UError, match='not in luot_c8'):
        CU.check_khoa(bad)
    bad = json.loads(json.dumps(k))
    bad['luot_c8']['a']['frames_sha256'] = 'f' * 64
    with pytest.raises(CU.UError, match='luot_c8.a.frames_sha256'):
        CU.check_khoa(bad)


def test_moc_api_chain_links():
    # each link of check_moc alone (C8BU4-R1-1, R1-2), through the API
    with pytest.raises(CU.UError, match='sha256.frames'):                 # the clock file names other frames
        CU.khoa({'a': dict(gio_d('O1-F05', 3e-4, 0.01, 0.0), sha256=dict(frames='f' * 64))}, TBD, sync(CAM), TK, XO,
                moc_tep=MT(), gio_sha={'a': _gsha('a')})
    with pytest.raises(CU.UError, match='names no frames'):               # an ok clock file naming no frames
        CU.khoa({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}, TBD, sync(CAM), TK, XO, moc_tep=MT(),
                gio_sha={'a': _gsha('a')})
    with pytest.raises(CU.UError, match='SHA-256 of every C8 clock file'):
        CU.khoa(GIOS({'a': gio_d('O1-F05', 3e-4, 0.01, 0.0)}), TBD, sync(CAM), TK, XO, moc_tep=MT())
    for key, val in (('frames_sha256', 'f' * 64), ('session_sha256', 'f' * 64), ('u_so_sha256', 'f' * 64),
                     ('lich_sha256', 'f' * 64), ('cham_lai', 1), ('thu', True), ('tag_offset', [0.001, 0, 0])):
        mt = MT()
        mt['track']['a'][key] = val
        with pytest.raises(CU.UError, match=key if key not in ('cham_lai', 'thu') else 'final scoring'):
            KH(moc_tep=mt)
    with pytest.raises(CU.UError, match='track.json runs'):
        KH(moc_tep=MT(track={}))
    for key in ('session_sha256', 'u_so_sha256', 'tap_moc_file_sha256', 'lich_sha256'):
        mt = MT()
        mt['lich_doc'][key] = 'f' * 64 if key != 'lich_sha256' else None
        with pytest.raises(CU.UError, match=key if key != 'lich_sha256' else 'no lich_sha256'):
            KH(moc_tep=mt)
    for key in ('car_side_m', 'tag_h_m', 'kappa'):
        mt = MT()
        mt['session_doc'][key] = mt['session_doc'][key] * 1.01
        with pytest.raises(CU.UError, match=f'session.json {key}'):
            KH(moc_tep=mt)
    KH()                                                                    # the consistent set is accepted


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



def test_code_23_11_scorer_cmd_passes_so_do():
    # CODE 23/11 (3): every node is scored with the so_do of the run and the SHA of its source (C8 gate file or J v6)
    import argparse
    a = argparse.Namespace(scorer=None, frames='f.csv', esp32='e.csv', hang_so='h.json', sender=None, scorer_arg=None,
                           so_do_run=frozenset({'g5'}), so_do_sha_run=LU.J_V6_SHA256)
    cmd = LU.scorer_cmd('gio.json', 0, a, 's.json', 'r.csv', 'O1-F08')
    i = cmd.index('--so-do')
    assert cmd[i + 1:i + 4] == ['g5', '--so-do-sha', LU.J_V6_SHA256]
    a.so_do_run = frozenset()
    assert '--so-do' not in LU.scorer_cmd('gio.json', 0, a, 's.json', 'r.csv', 'O1-F05')
