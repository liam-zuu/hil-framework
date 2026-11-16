"""Tests for tools/c7_score.py (O1 v11.28 camera rules) on SYNTHETIC runs built here from the command (NOT a
measurement): frames as c7_measure.read_frames returns them, an agv_logger-like CSV with the command-path and IMU
columns, a segments file when asked. The command path (o1_cmd_path.analyse) is stubbed except where a test says
so: its own tests cover it; the 16/11 trial logs ran through it unstubbed (design viec b)."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_score as cs  # noqa: E402
import rq3_log  # noqa: E402

FPS = 60.0
T0 = 20.0         # s, command start on the log clock
OFFSET = 0.25     # s, t_log = t_cam + OFFSET
YAW0 = 0.5        # rad, IMU yaw = floor yaw - YAW0


@pytest.fixture(autouse=True)
def cmd_ok(monkeypatch):
    """o1_cmd_path.analyse stubbed: no flag (tests that need a flag set CMD_FLAGS)."""
    state = dict(flags=[])

    def fake(rows, task, ev, has_age, sched, bao, ok):
        return dict(flags=list(state['flags']), notes=[], cond2={}, cond3={}, cond4={})
    monkeypatch.setattr(cs.ocp, 'analyse', fake)
    return state


def hs_file(tmp_path, **over):
    d = dict(version='c7_hang_so_v1', trang_thai='de_xuat', nguon='test', U_p_m=0.005, U_psi_deg=2.0,
             u_t_ghep_s=0.005, e_max=0.0002, t_trong_s=0.10, t_mat_s=0.68, bump_px=12.0, drift_px=1.5, pred_m=0.02,
             pred_half=6, pred_min=4, yaw_nghi_deg=3.0, v_d=0.01,
             imu=dict(period_s=0.05, tau_hat_s=0.0, s_plus=0.0, s_minus=0.0, j_lim_s=0.005, h_lim_s=0.105, pre_s=3.0,
                      wz_still=0.05, n_min=30,
                      U_IMU_deg={'O1-F02': 1.0, 'O1-F05': 1.0, 'O1-F08': 1.0, 'O1-K1': 1.0, 'O1-K2': 1.0}))
    imu_over = over.pop('imu', {})
    d.update(over)
    d['imu'].update(imu_over)
    p = tmp_path / 'hang_so.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    return cs.load_hang_so(str(p))


_LANES = {}


def lane_of(task, k_start=None):
    key = (task, None if k_start is None else tuple(k_start))
    if key not in _LANES:                               # Path8 has 400001 samples: build once per task
        _LANES[key] = cs.Fig8Task(task) if task.startswith('O1-F') else cs.KTask(task, k_start)
    return _LANES[key]


def true_pose(task, k_start, tk, lateral=0.0, lag=0.0, back=None):
    """Floor pose (x, y, psi) of the synthetic car at command time tk."""
    lane = lane_of(task, k_start)
    tw = tk - lag
    if back and back[0] <= tk <= back[1]:
        tw -= back[2] * math.sin(math.pi * (tk - back[0]) / (back[1] - back[0]))
    _, px, py, hc = lane.command(max(0.0, tw))
    x, y = px - lateral * math.sin(hc), py + lateral * math.cos(hc)
    ps = hc if task.startswith('O1-F') else 0.0
    if not task.startswith('O1-F'):
        x0, y0, h0 = k_start
        c, s = math.cos(h0), math.sin(h0)
        x, y = x0 + c * x - s * y, y0 + s * x + c * y
        ps += h0
    return x, y, ps


def make_frames(task, lateral=0.0, k_start=(0.0, 0.0, 0.0), pre=4.0, post=3.0, drop=(), shift=None, lag=0.0,
                jump=None, back=None, lat_fn=None):
    """Car on the command (lateral offset along the lane normal, delayed by lag), still at the end after T_cmd.
    drop: command-time spans with no valid frame; jump: {frame index: (dx, dy)} added to the measured centre;
    lat_fn(t): extra lateral offset."""
    lane = lane_of(task, k_start)
    t = np.arange(-pre, lane.Tcmd + post, 1.0 / FPS)
    P = np.array([true_pose(task, k_start, tk, lateral + (lat_fn(tk) if lat_fn else 0.0), lag, back) for tk in t])
    xs, ys, ps = P[:, 0].copy(), P[:, 1].copy(), P[:, 2]
    vx, vy = np.gradient(xs, t), np.gradient(ys, t)
    wz = np.gradient(np.unwrap(ps), t)
    for k, (dx, dy) in (jump or {}).items():
        xs[k] += dx
        ys[k] += dy
    rows = []
    for k, tk in enumerate(t):
        ok = not any(a <= tk < b for a, b in drop)
        rows.append(dict(idx=k, t_s=tk + T0 - OFFSET, valid=ok, why=[] if ok else ['khong_thay_xe'], n_marks=24,
                         rms_px=0.3, shift_px=(shift.get(k, 0.2) if shift else 0.2), cam_z=2.6, car_seen=ok,
                         x=xs[k], y=ys[k], psi=cs.wrap(ps[k]), tag_x=xs[k], tag_y=ys[k], res_m=0.0002,
                         res_sim_m=0.0002, scale=1.0, scale_ref=1.0, row_px=500.0, vx=vx[k], vy=vy[k], wz=wz[k],
                         speed=0.0 if tk > lane.Tcmd else math.hypot(vx[k], vy[k])))
    return rows


LOG_COLS = ('timestamp_us', 'log_seq', 'tick', 'motor_on', 'run_active', 'pos_reset_n', 'pos_x_ref', 'pos_y_ref',
            'traj_age_ms', 'sp_vx', 'sp_vy', 'sp_wz', 'vx', 'vy', 'wz', 'yaw_rad', 'rv_n', 'age_r_ms', 'gz', 'gyro_n',
            'age_g_ms', 'reset_n', 'r_valid', 'g_valid')


def make_log(path, task, k_start=(0.0, 0.0, 0.0), pre=6.0, post=4.0, lost_before_edge=False, dup=0,
             segments=True, yaw_err=None, no_imu=False, reset_at=None):
    """agv_logger-like CSV at path (100 Hz lines, IMU and gyro reports every 50 ms). yaw_err(t) [rad] added to the
    IMU yaw; dup: that many identical lines repeated (marked in <path>.segments.csv when segments)."""
    lane = lane_of(task, k_start)
    lines, k = [], 0
    for tk in np.arange(-pre, lane.Tcmd + post, 0.01):
        x, y, ps = true_pose(task, k_start, tk)
        x2, y2, ps2 = true_pose(task, k_start, tk + 0.005)
        x1, y1, ps1 = true_pose(task, k_start, tk - 0.005)
        v = math.hypot(x2 - x1, y2 - y1) / 0.01
        wzv = cs.wrap(ps2 - ps1) / 0.01
        if task.startswith('O1-F'):
            bvx, bvy = v, 0.0
        else:
            h = k_start[2]
            gx, gy = (x2 - x1) / 0.01, (y2 - y1) / 0.01
            bvx, bvy = math.cos(h) * gx + math.sin(h) * gy, -math.sin(h) * gx + math.cos(h) * gy
        run = tk >= -1e-9
        cmd = run and tk <= lane.Tcmd
        rep = k // 5
        age = (k % 5) * 10
        yaw = cs.wrap(ps - YAW0 + (yaw_err(tk) if yaw_err else 0.0))
        lines.append(dict(timestamp_us=str(int(round((tk + T0) * 1e6))), log_seq=str(k), tick=str(k), motor_on='1',
                          run_active='1' if run else '0', pos_reset_n='1', pos_x_ref='0', pos_y_ref='0',
                          traj_age_ms='5', sp_vx=f'{bvx if cmd else 0.0:.5f}', sp_vy=f'{bvy if cmd else 0.0:.5f}',
                          sp_wz='0', vx=f'{bvx:.5f}', vy=f'{bvy:.5f}', wz=f'{wzv:.5f}', yaw_rad=f'{yaw:.6f}',
                          rv_n='1' if no_imu else str(rep), age_r_ms=str(age), gz=f'{wzv:.5f}', gyro_n=str(rep),
                          age_g_ms=str(age), reset_n='4' if reset_at is not None and tk >= reset_at else '3',
                          r_valid='1', g_valid='1'))
        k += 1
    if lost_before_edge:
        i = next(i for i, r in enumerate(lines) if r['run_active'] == '1')
        del lines[i - 1]
    out, seg = [], []
    for i, r in enumerate(lines):
        out.append(r)
        seg.append('ok')
        if dup and i % 200 == 100 and dup > sum(s == 'duplicate' for s in seg):
            out.append(dict(r))
            seg.append('duplicate')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write('# synthetic log\n')
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        w.writeheader()
        w.writerows(out)
    if segments:
        with open(str(path) + '.segments.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['row', 'log_seq', 'timestamp_us', 'segment', 'status'])
            for i, (r, s) in enumerate(zip(out, seg), start=1):
                w.writerow([i, r['log_seq'], r['timestamp_us'], 0, s])
    return str(path)


def run(tmp_path, task='O1-F05', frames=None, log_kw=None, hs_kw=None, **kw):
    hs = hs_file(tmp_path, **(hs_kw or {}))
    log = make_log(tmp_path / 'esp32.csv', task, kw.get('k_start') or (0.0, 0.0, 0.0), **(log_kw or {}))
    fr = frames if frames is not None else make_frames(task, k_start=kw.get('k_start') or (0.0, 0.0, 0.0))
    kw.setdefault('offset_s', OFFSET)
    kw.setdefault('offset_reason', 'test')
    return cs.score(fr, log, task, hs, **kw)


def gate_ok(out):
    """The gate block passes the rq3_log checks (HW) and agrees with the outcome."""
    g = out['gate_v3']
    rq3_log.validate_gate(g, 'HW')
    rq3_log.check_outcome_vs_gate(out['outcome'], g, 'HW', out['fail_cause'])
    return g


# ---------------------------------------------------------------- basic outcomes
def test_on_lane_passes(tmp_path):
    out, rows = run(tmp_path)
    g = gate_ok(out)
    assert out['outcome'] == 'dat', (out['outcome_reasons'], {c: g[c]['state'] for c in ('g1', 'g2', 'g3', 'g4', 'g5')})
    assert g['g1']['t_done'][0] == g['g1']['t_done'][1] and g['g1']['t1'] is None and not g['g1']['cat_t1']
    assert abs(g['g1']['t_done'][0] - (out['T_cmd'] + 1.0)) < 0.05              # still for t_d after T_cmd
    assert out['e_perp_max'] < 1e-3 and out['yaw_check']['compared'] and out['yaw_check']['n_suspect'] == 0
    assert abs(out['yaw_check']['align_c_deg'] - math.degrees(YAW0)) < 0.1
    assert out['n_suspect_pos'] == 0 and out['n_g4_formula'] == out['frames_window']
    for k in ('e_theta_lane_max_deg', 'e_theta_cmd_max_deg', 'e_s_absmax', 'e_trk_max'):   # s05_final_table keys
        assert k in out


@pytest.mark.parametrize('lat,want', [(0.12, 'khong_dat'), (0.098, 'chua_ket_luan'), (0.04, 'dat')])
def test_lateral_offset_three_states(tmp_path, lat, want):
    out, _ = run(tmp_path, frames=make_frames('O1-F05', lateral=lat))
    g = gate_ok(out)
    assert abs(g['g2']['q'] - lat) < 2e-3
    assert out['outcome'] == want, (out['outcome_reasons'], {c: g[c]['state'] for c in ('g1', 'g2', 'g3', 'g4', 'g5')})


def test_bounds_use_U_p_and_U_t(tmp_path):
    out, rows = run(tmp_path, frames=make_frames('O1-F05', lateral=0.03))
    g = gate_ok(out)
    up = 0.005
    assert abs(g['g2']['hi'] - g['g2']['q'] - up) < 1e-9 and abs(g['g2']['q'] - g['g2']['lo'] - up) < 1e-9
    vmax, wzmax = out['gate_v3']['v_cmd_max'], out['gate_v3']['wz_cmd_max']
    r = rows[100]
    assert abs((r['hi_g3'] - r['e_trk']) - (up + vmax * out['U_t_s'])) < 1e-9
    assert abs((r['hi_g5'] - abs(r['e_theta_cmd_deg'])) - math.degrees(math.radians(2.0) + wzmax * out['U_t_s'])) < 1e-9
    assert out['U_t_parts']['drift_src'].startswith('e_max')                    # --offset-s given
    assert abs(out['U_t_s'] - (0.005 + 0.0002 * out['t_deadline'] + 0.01)) < 1e-6


# ---------------------------------------------------------------- lost runs and the exception
def test_gap_above_t_trong_spoils_the_run(tmp_path):
    out, _ = run(tmp_path, frames=make_frames('O1-F05', drop=[(4.0, 4.2)]))
    g = gate_ok(out)
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == 'he_do' and 'khoang trong' in out['outcome_reasons'][0]
    assert abs(g['g1']['t1'] - 4.0) < 0.02 and not g['g1']['cat_t1']


def test_short_gap_is_reported_not_spoiling(tmp_path):
    out, _ = run(tmp_path, frames=make_frames('O1-F05', drop=[(4.0, 4.06)]))
    assert out['outcome'] == 'dat' and out['n_gaps_over_frame'] == 1 and 0.06 < out['gaps_total_s'] < 0.09


def test_gap_from_t0_to_the_first_frame_counts(tmp_path):
    """RB-2: the gap from t = 0 (not from the last frame before 0) to the first valid frame."""
    out, _ = run(tmp_path, frames=make_frames('O1-F05', drop=[(-0.05, 0.12)]))
    assert out['outcome'] == 'hong_he_do' and out['gate_v3']['g1']['t1'] == 0.0
    out, _ = run(tmp_path, frames=make_frames('O1-F05', drop=[(-0.2, 0.06)]))
    assert out['outcome'] == 'dat'


def test_violation_before_t1_is_khong_dat(tmp_path):
    """R2V-4: a certain G2 violation before a long gap: khong_dat, cat_t1, numbers before t1 only, G1 chua."""
    fr = make_frames('O1-F05', drop=[(8.0, 8.3)], lat_fn=lambda t: 0.13 if 5.0 <= t < 6.0 else 0.0)
    out, _ = run(tmp_path, frames=fr)
    g = gate_ok(out)
    assert out['outcome'] == 'khong_dat' and out['fail_cause'] == rq3_log.VIOL_BEFORE and out['cat_t1']
    assert g['g1']['state'] == 'chua_ket_luan' and g['g1']['t_done'] == [None, None] and abs(g['g1']['t1'] - 8.0) < 0.02
    assert g['g2']['state'] == 'khong_dat' and abs(g['g2']['n_frames'] - 8.0 * FPS) <= 2      # frames before t1 only
    assert g['g2']['n_frames'] < out['frames_window'] - 300


def test_violation_after_t1_is_lost_run(tmp_path):
    fr = make_frames('O1-F05', drop=[(4.0, 4.3)], lat_fn=lambda t: 0.13 if 5.0 <= t < 6.0 else 0.0)
    out, _ = run(tmp_path, frames=fr)
    gate_ok(out)
    assert out['outcome'] == 'hong_he_do' and not out['cat_t1']


def test_camera_bumped_is_lost_even_with_a_violation_before(tmp_path):
    fr = make_frames('O1-F05', shift={700: 15.0}, drop=[(8.0, 8.3)], lat_fn=lambda t: 0.13 if 5.0 <= t < 6.0 else 0.0)
    out, _ = run(tmp_path, frames=fr)
    assert out['bumped'] and out['outcome'] == 'hong_he_do' and not out['cat_t1']


def test_long_gap_is_tracking_error(tmp_path):
    out, _ = run(tmp_path, 'O1-F02', frames=make_frames('O1-F02', drop=[(5.0, 5.8)]))
    assert out['n_miss_frames'] == 1 and out['n_track_err'] >= 1 and out['outcome'] == 'hong_he_do'


def test_command_path_lost(tmp_path, cmd_ok):
    cmd_ok['flags'] = ['cond1_pos_reset: pos_reset_n [1, 2] after the edge']
    out, _ = run(tmp_path)
    gate_ok(out)
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == 'duong_lenh' and out['cmd_path']['lost']


def test_edge_unknown_is_lost(tmp_path):
    out, _ = run(tmp_path, log_kw=dict(lost_before_edge=True))
    assert out['outcome'] == 'hong_he_do' and out['gate_v3'] is None and 'edge_unknown' in out['outcome_reasons'][0]


# ---------------------------------------------------------------- position-suspect frames (RS2-1, R2V-1)
def test_jump_is_kept_as_position_suspect(tmp_path):
    lane = lane_of('O1-F05')
    k = 4 * 60 + 5 * 60                                   # t = 5 s
    _, _, _, hc = lane.command(5.0)
    fr = make_frames('O1-F05', jump={k: (-0.03 * math.sin(hc), 0.03 * math.cos(hc))})
    out, rows = run(tmp_path, frames=fr)
    gate_ok(out)
    r = min(rows, key=lambda r: abs(r['t'] - 5.0))
    assert out['n_suspect_pos'] == 1 and r['suspect'] == 1 and r['scored'] == 1
    assert abs(r['U_p_k'] - (0.005 + r['pred_m'])) < 1e-12 and r['pred_m'] > 0.025
    assert abs(r['e_perp'] - 0.03) < 2e-3
    # U_p,k 35 mm: G2 still dat, but G4 by the formula U_p,k + r_c (U_psi + 9.2 U_p,k) = 0.115 m > D (O1 3.5)
    g = out['gate_v3']
    assert g['g2']['state'] == 'dat' and g['g4']['state'] == 'chua_ket_luan' and out['outcome'] == 'chua_ket_luan'
    assert abs((r['hi_g4'] - r['e_bao']) - (r['U_p_k'] + cs.R_C * (math.radians(2.0) + 9.2 * r['U_p_k']))) < 1e-9


def test_suspect_does_not_push_progress(tmp_path):
    """A frame 0.3 m ahead ALONG the lane: way 1 (suspects as frames) moves the progress, the main tracker not."""
    lane = lane_of('O1-F05')
    k = 4 * 60 + 5 * 60
    _, x1, y1, _ = lane.command(5.6)
    _, x0, y0, _ = lane.command(5.0)
    fr = make_frames('O1-F05', jump={k: (x1 - x0, y1 - y0)})
    out, rows = run(tmp_path, frames=fr)
    j = min(range(len(rows)), key=lambda i: abs(rows[i]['t'] - 5.0))
    assert rows[j]['suspect'] == 1
    assert rows[j]['s_car'] == rows[j - 1]['s_car'] and rows[j]['s_car_way1'] > rows[j - 1]['s_car_way1'] + 0.1


def test_suspect_chain_longer_than_t_mat_is_tracking_error(tmp_path):
    fr = make_frames('O1-F05', lat_fn=lambda t: 0.05 if 5.0 <= t < 5.8 else 0.0)
    out, rows = run(tmp_path, frames=fr, hs_kw=dict(pred_half=60))
    # R2V-1: the first suspect past t_mat, then the first non-suspect frame after the chain (dt above t_mat)
    assert out['n_suspect_pos'] > 40 and out['n_track_err'] == 2 and out['n_unscored'] >= 1
    assert out['outcome'] == 'hong_he_do' and abs(out['t_first_track_err'] - (5.0 + 0.68)) < 0.03
    late = [r for r in rows if r['suspect'] and r['t'] > out['t_first_track_err'] + 1e-9]
    assert late and all(r['scored'] == 0 for r in late)


def test_two_ways_of_using_suspects(tmp_path):
    """Blocks of 8 suspect frames every 0.4 s after T_cmd: way 1 completes, way 2 (suspects as gaps above t_trong)
    never holds t_d: G1 chua_ket_luan, the common domain ends at way 1's t_done, the union at the deadline."""
    lane = lane_of('O1-F05')
    t = np.arange(-4.0, lane.Tcmd + 3.0, 1.0 / FPS)
    jump = {k: (0.03, 0.0) for k, tk in enumerate(t)
            if tk > lane.Tcmd - 0.5 and ((tk - lane.Tcmd) % 0.4) < 8.5 / FPS}
    out, _ = run(tmp_path, frames=make_frames('O1-F05', jump=jump), hs_kw=dict(pred_half=30))
    g = gate_ok(out)
    td = g['g1']['t_done']
    assert td[0] is not None and td[1] is None and g['g1']['state'] == 'chua_ket_luan'
    assert out['t_end_common'] == td[0] and out['t_end_union'] == out['t_deadline']
    assert g['g2']['chung']['n_undef'] == 0 and g['g2']['n_frames'] > 0


# ---------------------------------------------------------------- yaw check (IMU)
def test_yaw_suspect_widens_U_psi(tmp_path):
    out, rows = run(tmp_path, log_kw=dict(yaw_err=lambda t: math.radians(6.0) if 3.0 <= t < 4.0 else 0.0))
    g = gate_ok(out)
    r = min(rows, key=lambda r: abs(r['t'] - 3.5))
    assert out['yaw_check']['n_suspect'] > 40 and abs(r['U_psi_k_deg'] - (6.0 + 1.0)) < 0.05
    assert g['g5']['state'] == 'chua_ket_luan' and out['outcome'] == 'chua_ket_luan'


def test_no_imu_yaw_makes_G4_G5_undefined(tmp_path):
    out, _ = run(tmp_path, log_kw=dict(no_imu=True))
    g = gate_ok(out)
    assert not out['yaw_check']['compared']
    assert g['g4']['n_undef'] == g['g4']['n_frames'] and g['g5']['n_undef'] == g['g5']['n_frames']
    assert g['g2']['n_undef'] == 0 and out['outcome'] == 'chua_ket_luan'


def test_imu_reset_in_the_run_undefines_later_frames(tmp_path):
    out, rows = run(tmp_path, log_kw=dict(reset_at=6.0))
    g = gate_ok(out)
    assert 0 < g['g5']['n_undef'] < g['g5']['n_frames']
    assert all(math.isinf(r['U_psi_k_deg']) for r in rows if r['t'] > 6.1)


def test_imu_gap_long_before_the_run_does_not_matter(tmp_path):
    """slice_log (16/11 trial r2): a report gap 4 s before the alignment part."""
    hs = hs_file(tmp_path)
    p = make_log(tmp_path / 'esp32.csv', 'O1-F05', pre=12.0)
    rows = list(csv.DictReader(l for l in open(p, encoding='utf-8') if not l.startswith('#')))
    for r in rows:                                      # reports frozen in [-10, -9) s: an unbridgeable gap
        t = int(r['timestamp_us']) * 1e-6 - T0
        if -10.0 <= t < -9.0:
            r['rv_n'], r['gyro_n'] = '-5', '-5'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        w.writeheader()
        w.writerows(rows)
    os.remove(p + '.segments.csv')
    out, _ = cs.score(make_frames('O1-F05'), p, 'O1-F05', hs, offset_s=OFFSET, offset_reason='test')
    assert out['yaw_check']['compared'] and out['outcome'] == 'dat'


# ---------------------------------------------------------------- log rules (L92)
def test_duplicates_marked_in_segments_are_dropped(tmp_path):
    out, _ = run(tmp_path, log_kw=dict(dup=3))
    assert out['n_duplicates'] == 3 and out['outcome'] == 'dat'


def test_duplicates_without_segments_are_refused(tmp_path):
    with pytest.raises(ValueError, match='L92'):
        run(tmp_path, log_kw=dict(dup=2, segments=False))


def test_no_duplicates_without_segments_is_fine(tmp_path):
    out, _ = run(tmp_path, log_kw=dict(segments=False))
    assert out['n_duplicates'] == 0 and out['outcome'] == 'dat'


# ---------------------------------------------------------------- K1, K2, clock, CLI, constants
def test_k1_on_a_rotated_start_pose(tmp_path):
    ks = (0.3, -0.4, math.radians(30.0))
    out, _ = run(tmp_path, 'O1-K1', frames=make_frames('O1-K1', k_start=ks), k_start=ks, allow_final_set=True)
    gate_ok(out)
    assert out['outcome'] == 'dat' and out['e_perp_max'] < 1e-3 and 'e_theta_lane_max_deg' not in out


def test_k2_needs_flag_and_start(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, 'O1-K2', frames=make_frames('O1-K2'))
    with pytest.raises(ValueError):
        run(tmp_path, 'O1-K2', frames=make_frames('O1-K2'), allow_final_set=True)


def test_offset_needs_a_reason(tmp_path):
    with pytest.raises(ValueError, match='offset-reason'):
        run(tmp_path, offset_reason='')


def test_clock_from_speed_matching(tmp_path):
    out, _ = run(tmp_path, offset_s=None, offset_reason='')
    assert out['offset_src'] == 'o1_time_sync' and abs(out['offset_s'] - OFFSET) < 0.01
    assert out['U_t_parts']['drift_src'].startswith('C7b') and out['outcome'] == 'dat'


def test_late_car_misses_completion(tmp_path):
    out, _ = run(tmp_path, frames=make_frames('O1-F05', lag=2.5, post=6.0), log_kw=dict(post=7.0))
    gate_ok(out)
    assert out['gate_v3']['g1']['state'] == 'khong_dat' and out['outcome'] == 'khong_dat'


@pytest.mark.parametrize('bad', [dict(version='x'), dict(trang_thai='khoa'), dict(t_trong_s=0.7),
                                 dict(t_mat_s=0.69), dict(pred_half=1), dict(imu=dict(n_min=2)), dict(nguon=' ')])
def test_constants_checked(tmp_path, bad):
    with pytest.raises(ValueError):
        hs_file(tmp_path, **bad)


def test_cli_writes_score_json(tmp_path):
    import c7_measure as cm
    cm.write_frames(tmp_path / 'frames.csv', make_frames('O1-F05'))
    log = make_log(tmp_path / 'esp32.csv', 'O1-F05')
    hs_file(tmp_path)
    args = ['--frames', str(tmp_path / 'frames.csv'), '--esp32', log, '--task', 'O1-F05', '--hang-so',
            str(tmp_path / 'hang_so.json'), '--offset-s', str(OFFSET), '--offset-reason', 'test',
            '--out', str(tmp_path / 'score.json'), '--rows', str(tmp_path / 'rows.csv')]
    assert cs.main(args) == 0
    s = json.loads((tmp_path / 'score.json').read_text(encoding='utf-8'))
    assert s['outcome'] == 'dat' and s['hang_so']['trang_thai'] == 'de_xuat' and len(s['frames_sha256']) == 64
    rq3_log.validate_gate(rq3_log.read_gate_json(str(tmp_path / 'score.json')), 'HW')
    assert cs.main(args) == 2                           # outputs are never overwritten


# ---------------------------------------------------------------- G4 helpers
@pytest.mark.parametrize('a,b', [(0.17, 0.145), (0.145, 0.17)])
def test_g_range_against_dense_sampling(a, b):
    rng = np.random.default_rng(1)
    for _ in range(300):
        lo = rng.uniform(-4, 4)
        hi = lo + rng.uniform(0, 3.5)
        th = np.linspace(lo, hi, 20001)
        g = a * np.abs(np.sin(th)) + b * (np.abs(np.cos(th)) - 1)
        mn, mx = cs.g_range(lo, hi, a, b)
        assert mn <= g.min() + 1e-12 and mx >= g.max() - 1e-12
        assert mn >= g.min() - 2e-4 and mx <= g.max() + 2e-4      # the grid misses a kink by < a x step / 2


def test_g4_bounds_near_a_centre_of_curvature():
    """Measured centre 2 mm from the centre of the tightest curve: the lane points of a long arc are almost equally
    far, so a true centre within U_p projects far along the arc with a very different heading. J_k must take the
    points within the smallest distance + 2 U_p (a set of one point does not bound it)."""
    lane = lane_of('O1-F05')
    p = lane.path
    kap = np.gradient(np.unwrap(p.head), p.s)
    i = int(np.argmax(np.abs(kap[1000:-1000]))) + 1000
    r = 1.0 / kap[i]
    cx, cy = p.x[i] - math.sin(p.head[i]) * r, p.y[i] + math.cos(p.head[i]) * r
    x, y, psi = cx + 0.002, cy - 0.001, p.head[i] + 0.4
    w_lo, w_hi = p.s[i] - 0.3, p.s[i] + 0.3
    upk, ups = 0.01, math.radians(2.0)
    lo, hi = cs.g4_bounds_jk(lane, x, y, psi, w_lo, w_hi, upk, ups)
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(400):
        rr, a = upk * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        _, e_t, h_t, _ = p.nearest(x + rr * math.cos(a), y + rr * math.sin(a), w_lo, w_hi)
        th = cs.wrap(psi + rng.uniform(-ups, ups) - h_t)
        vals.append(e_t + 0.17 * abs(math.sin(th)) + 0.145 * (abs(math.cos(th)) - 1))
    assert lo - 1e-6 <= min(vals) and max(vals) <= hi + 1e-6, (lo, min(vals), max(vals), hi)
    assert max(vals) - min(vals) > 0.05                 # the case is hard: headings spread along the arc


@pytest.mark.parametrize('task,ks', [('O1-F05', None), ('O1-K2', (0.0, 0.0, 0.0))])
def test_g4_bounds_contain_every_true_pose(task, ks):
    """O1 3.4 J_k: for random true poses within U_p, U_psi of the measured one, the true G4 (nearest lane point in
    the window, its heading) lies in [lo, hi]."""
    lane = lane_of(task, ks)
    rng = np.random.default_rng(7)
    upk, ups = 0.01, math.radians(3.0)
    for _ in range(40):
        s0 = rng.uniform(0.3, lane.S - 0.3)
        if isinstance(lane, cs.KTask):
            px, py, hd = lane.path.point_at(s0)
        else:
            i = int(np.searchsorted(lane.path.s, s0))
            px, py, hd = lane.path.x[i], lane.path.y[i], lane.path.head[i]
        x = px + rng.uniform(-0.06, 0.06)
        y = py + rng.uniform(-0.06, 0.06)
        psi = hd + rng.uniform(-0.3, 0.3)
        w_lo, w_hi = s0 - 0.25, s0 + 0.25
        lo, hi = cs.g4_bounds_jk(lane, x, y, psi, w_lo, w_hi, upk, ups)
        for _ in range(30):
            r, a = upk * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
            xt, yt = x + r * math.cos(a), y + r * math.sin(a)
            pt = psi + rng.uniform(-ups, ups)
            s_t, e_t, h_t, _ = lane.path.nearest(xt, yt, w_lo, w_hi)
            psi0, al, bl = lane.nominal(h_t)
            th = cs.wrap(pt - h_t - psi0)
            g_true = e_t + al * abs(math.sin(th)) + bl * (abs(math.cos(th)) - 1)
            assert lo - 2e-4 <= g_true <= hi + 2e-4, (lo, g_true, hi)
