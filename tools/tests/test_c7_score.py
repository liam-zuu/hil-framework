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
import c7_measure as cm  # noqa: E402
import c7_score as cs  # noqa: E402
import rq3_log  # noqa: E402

FPS = 60.0
T0 = 20.0         # s, command start on the log clock
OFFSET = 0.25     # s, t_log = t_cam + OFFSET
YAW0 = 0.5        # rad, IMU yaw = floor yaw - YAW0


@pytest.fixture(autouse=True)
def cmd_ok(monkeypatch, request):
    """o1_cmd_path.analyse stubbed: no flag (tests that need a flag set state['flags'], values state['cond']).
    Tests named test_real_cmd_* run the real o1_cmd_path (Codex 16/11: the stub covered every scorer test)."""
    state = dict(flags=[], cond={})
    if request.node.name.startswith('test_real_cmd'):
        return state

    def fake(rows, task, ev, has_age, sched, bao, ok):
        c = dict(cond2={}, cond3={}, cond4={})
        c.update(state['cond'])
        return dict(flags=list(state['flags']), notes=[], **c)
    monkeypatch.setattr(cs.ocp, 'analyse', fake)
    return state


def hs_file(tmp_path, **over):
    d = dict(version='c7_hang_so_v1', trang_thai='de_xuat', nguon='test', U_p_m=0.005, U_psi_deg=2.0,
             u_t_ghep_s=0.005, troi_ppm=200.0, t_trong_s=0.10, t_mat_s=0.68, bump_px=12.0, drift_px=1.5, pred_m=0.02,
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
                jump=None, back=None, lat_fn=None, measured=True):
    """Car on the command (lateral offset along the lane normal, delayed by lag), still at the end after T_cmd.
    drop: command-time spans with no valid frame; jump: {frame index: (dx, dy)} added to the measured centre;
    lat_fn(t): extra lateral offset. measured: the gates and speeds by c7_measure.apply_gates AFTER the jumps, as the
    real chain (Codex 16/11: speeds built before the jumps hid CS1-9); False: the true speeds (0 after T_cmd)."""
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
                         speed=0.0 if tk > lane.Tcmd else math.hypot(vx[k], vy[k]),
                         shift_mag_px=(shift.get(k, 0.2) if shift else 0.2)))
    if measured:
        cm.apply_gates(rows, 0.3)
    return rows


LOG_COLS = ('timestamp_us', 'log_seq', 'tick', 'motor_on', 'run_active', 'pos_reset_n', 'pos_x_ref', 'pos_y_ref',
            'traj_age_ms', 'sp_vx', 'sp_vy', 'sp_wz', 'vx', 'vy', 'wz', 'yaw_rad', 'rv_n', 'age_r_ms', 'gz', 'gyro_n',
            'age_g_ms', 'reset_n', 'r_valid', 'g_valid')


def make_log(path, task, k_start=(0.0, 0.0, 0.0), pre=6.0, post=4.0, lost_before_edge=False, dup=0,
             segments=True, yaw_err=None, no_imu=False, reset_at=None, cmd_real=False, imu_gaps=()):
    """agv_logger-like CSV at path (100 Hz lines, IMU and gyro reports every 50 ms). yaw_err(t) [rad] added to the
    IMU yaw; imu_gaps: command-time spans with no IMU nor gyro report (a gap the gyro cannot bridge);
    dup: that many identical lines repeated (marked in <path>.segments.csv when segments). cmd_real: the
    command path as the firmware logs it for an on-time sender (packets at 50 Hz from t = 0 through the zero tail,
    traj_age_ms from them, run_active while a packet is younger than 200 ms, the command pose = the ideal schedule
    in the firmware frame), plus <path>.events.txt with '#us' / '#us_end' and <path>_sender.txt with '#test_end'."""
    lane = lane_of(task, k_start)
    lines, k = [], 0
    last_rep, last_k = 0, 0
    ct = cs.ocp.Task(task, True) if cmd_real else None
    n_pk = int(math.ceil((ct.duration + cs.ocp.o1_tasks.O1_TAIL_S) * 50 - 1e-9)) if cmd_real else 0
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
        pxr, pyr, tage = '0', '0', '5'
        if cmd_real:
            j = min(int(math.floor((tk + 0.001) * 50 + 1e-9)), n_pk - 1)       # last packet arrived (k/50 - 1 ms)
            tage = '-1' if j < 0 else str(int((tk - (j / 50 - 0.001)) * 1000))
            run = j >= 0 and int(tage) < 200
            if run:
                ix, iy = ct.ideal(max(tk, 0.0))
                pxr, pyr = f'{ix:.4f}', f'{iy:.4f}'
        rep = k // 5
        age = (k % 5) * 10
        if any(a <= tk < b for a, b in imu_gaps):          # no new report: the counters keep their last value
            rep = last_rep
            age = min(age + 10 * (k - last_k), 9990)
        else:
            last_rep, last_k = rep, k
        yaw = cs.wrap(ps - YAW0 + (yaw_err(tk) if yaw_err else 0.0))
        lines.append(dict(timestamp_us=str(int(round((tk + T0) * 1e6))), log_seq=str(k), tick=str(k), motor_on='1',
                          run_active='1' if run else '0', pos_reset_n='1', pos_x_ref=pxr, pos_y_ref=pyr,
                          traj_age_ms=tage, sp_vx=f'{bvx if cmd else 0.0:.5f}', sp_vy=f'{bvy if cmd else 0.0:.5f}',
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
    if cmd_real:
        seqs = [r['log_seq'] for r in lines]
        with open(str(path) + '.events.txt', 'w', encoding='utf-8') as f:
            f.write(f'#us,ev=start,boot=00000001,on=1,swb=0,first_seq={seqs[0]},line_drop=0\n'
                    f'#us_end,boot=00000001,on=1,swb=0,changes=0,blk_ticks=0,cut_ticks=0,ticks={len(seqs)},'
                    f'last_data_seq={seqs[-1]},line_drop=0\n')
        with open(str(path) + '_sender.txt', 'w', encoding='utf-8') as f:
            f.write(f'#test_start,test={task},t_epoch=1.000,iso=x,duration_s={ct.duration:g},send_hz=50\n'
                    f'#test_end,test={task},t_epoch={1.0 + n_pk / 50:.3f},zeros={n_pk - ct.n_sched},'
                    f'tail_planned={n_pk - ct.n_sched},sent={ct.n_sched},planned={ct.n_sched},complete=1,'
                    f'sched_first_s=0.0,sched_last_s={(ct.n_sched - 1) / 50}\n')
    return str(path)


def run(tmp_path, task='O1-F05', frames=None, log_kw=None, hs_kw=None, **kw):
    hs = hs_file(tmp_path, **(hs_kw or {}))
    log = make_log(tmp_path / 'esp32.csv', task, kw.get('k_start') or (0.0, 0.0, 0.0), **(log_kw or {}))
    fr = frames if frames is not None else make_frames(task, k_start=kw.get('k_start') or (0.0, 0.0, 0.0))
    kw.setdefault('offset_s', OFFSET)
    kw.setdefault('offset_reason', 'test')
    if kw['offset_s'] is not None:              # the synthetic offset is exact at t = 0 (CS2-1 anchor)
        kw.setdefault('offset_anchor_s', 0.0)
        kw.setdefault('offset_err_s', 0.0)
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
    # frames whose projection sits on the window end cut at 0 or S (run start, the stop at the end) use J_k (CS1-3)
    assert out['n_suspect_pos'] == 0 and 0.8 * out['frames_window'] < out['n_g4_formula'] < out['frames_window']
    assert all(r['g4_formula'] == 0 for r in rows if r['at_end'])
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
    up, h = 0.005, lane_of('O1-F05').h
    assert abs(g['g2']['hi'] - g['g2']['q'] - up) < 1e-9 and abs(g['g2']['q'] - g['g2']['lo'] - up - h / 2) < 1e-9
    vmax, wzmax = out['gate_v3']['v_cmd_max'], out['gate_v3']['wz_cmd_max']
    r, ut = rows[100], out['U_t_s']
    vk, wk = cs.CmdRates(lane_of('O1-F05')).sup(r['t'] - ut, r['t'] + ut)      # O1 v11.33 2: per frame
    assert vk < vmax and wk < wzmax                     # rows[100] (t about 1.7 s) is on the ramp, below the peaks
    assert abs((r['hi_g3'] - r['e_trk']) - (up + vk * ut)) < 1e-9
    assert abs((r['hi_g5'] - abs(r['e_theta_cmd_deg'])) - math.degrees(math.radians(2.0) + wk * ut)) < 1e-9
    assert out['U_t_parts']['drift_src'].startswith('troi_ppm')                 # --offset-s given
    assert abs(out['U_t_s'] - (0.005 + 0.0002 * out['t_deadline'] + 0.01)) < 1e-6
    assert 'man_trap_s' not in out['U_t_parts'] and out['gio_c8b'] is None     # C8b v4 table only with --gio-c8b


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
    out, rows = run(tmp_path, frames=fr, hs_kw=dict(gia_dinh_nghi_vi_tri='dung'))   # C8 assumption held
    gate_ok(out)
    r = min(rows, key=lambda r: abs(r['t'] - 5.0))
    assert out['n_suspect_pos'] == 1 and r['suspect'] == 1 and r['scored'] == 1
    assert abs(r['U_p_k'] - (0.005 + r['pred_m'])) < 1e-12 and r['pred_m'] > 0.025
    assert abs(r['e_perp'] - 0.03) < 2e-3
    # U_p,k 35 mm: G2 still dat, but G4 by the formula U_p,k + r_c (U_psi + 9.2 U_p,k) = 0.115 m > D (O1 3.5)
    g = out['gate_v3']
    assert g['g2']['state'] == 'dat' and g['g4']['state'] == 'chua_ket_luan' and out['outcome'] == 'chua_ket_luan'
    h = lane.h                                         # grid slack of the figure 8 (CS1-2)
    assert abs((r['hi_g4'] - r['e_bao']) - (r['U_p_k'] + cs.R_C * (math.radians(2.0) + 9.2 * r['U_p_k']
                                                                   + cs.KAPPA_MAX * h / 2))) < 1e-9


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


def test_two_ways_of_using_suspects(tmp_path, monkeypatch):
    """Blocks of 8 suspect frames every 0.4 s after T_cmd (stub, the car still): way 1 completes, way 2 (suspects as
    gaps above t_trong) never holds t_d: G1 chua_ket_luan, the common domain ends at way 1's t_done, the union at the
    deadline. (With real 30 mm jumps the speeds of every valid frame, which way 1 uses since CS2-3, keep way 1 from
    holding t_d too, CS1-9; the stub keeps the poses on the car.)"""
    lane = lane_of('O1-F05')
    suspects_stub(monkeypatch, lambda tk: tk > lane.Tcmd - 0.5 and ((tk - lane.Tcmd) % 0.4) < 8.5 / FPS)
    out, _ = run(tmp_path, frames=make_frames('O1-F05'), hs_kw=dict(pred_half=30, gia_dinh_nghi_vi_tri='dung'))
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
    out, _ = cs.score(make_frames('O1-F05'), p, 'O1-F05', hs, offset_s=OFFSET, offset_reason='test', offset_anchor_s=0.0,
                      offset_err_s=0.0)
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
    assert out['U_t_parts']['drift_src'].startswith('max(C7b') and out['outcome'] == 'dat'


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
            '--offset-anchor-s', '0', '--offset-err-s', '0',
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
            if isinstance(lane, cs.KTask):                      # K: exact, heading of Polyline.head_at (CS1-1)
                s_t, e_t, h_t = lane.poly.solutions(xt, yt, w_lo, w_hi)[0]
            else:
                s_t, e_t, h_t, _ = lane.path.nearest(xt, yt, w_lo, w_hi)
            psi0, al, bl = lane.nominal(h_t)
            th = cs.wrap(pt - h_t - psi0)
            g_true = e_t + al * abs(math.sin(th)) + bl * (abs(math.cos(th)) - 1)
            assert lo - 1e-9 <= g_true <= hi + 1e-9, (lo, g_true, hi)      # Codex: 2e-4 hid an 8.9 um miss


# ---------------------------------------------------------------- Codex vong 1 (16/11, CS1-1..CS1-20), each failing on
# the code before the fixes (hil 00121a4); review 03_CODE/docs/review/2026-11-16_Tham_dinh_c7_score_o1_vong1.md
def frame_at(rows, t):
    return min(range(len(rows)), key=lambda i: abs(rows[i]['t'] - t))


def suspects_stub(monkeypatch, fn=None):
    """mark_suspects replaced: the frames where fn(t) is true are the suspects (|p - p^| 0.03 m), none by default.
    The real rule picks neighbours by index across a gap, so a frame next to a gap still gets a prediction."""
    def fake(t, x, y, thr, half, nmin):
        sus = np.array([bool(fn(v)) for v in t]) if fn else np.zeros(len(t), bool)
        return sus, np.where(sus, 0.03, np.nan), 1, True
    monkeypatch.setattr(cs, 'mark_suspects', fake)


def k_true_g4(lane, xt, yt, pt, w_lo, w_hi):
    """True G4 of a K pose by the exact projection of o1_gate_eval.Polyline (heading of head_at), not the scorer."""
    _, e, h = lane.poly.solutions(xt, yt, w_lo, w_hi)[0]
    psi0, al, bl = lane.nominal(h)
    return cs.o1_gate_eval.g4_value(e, pt, h, psi0, al, bl)


def test_cs1_1_k2_closing_corner_counterexample():
    """Codex CS1-1: K2, window [3.8, 4], the centre projects onto the closing corner s = S: the true G4 takes the
    outgoing side (side 0); the old bounds used the incoming side only (hi 0.09832 < true 0.10036)."""
    lane = lane_of('O1-K2', (0.0, 0.0, 0.0))
    x, y, psi = 0.05656854, -0.05656854, 0.127
    lo, hi = cs.g4_bounds_jk(lane, x, y, psi, 3.8, 4.0, 0.0003, math.radians(0.2))
    g_true = k_true_g4(lane, x, y, psi, 3.8, 4.0)
    assert abs(g_true - 0.10036423) < 1e-6 and lo <= g_true <= hi
    assert not cs.one_heading(lane, x, y, 3.8, 4.0, 0.0003)         # no U formula with a corner in J_k


@pytest.mark.parametrize('w', [(0.7, 1.3), (1.8, 2.2), (2.6, 3.4), (3.6, 4.0), (3.9, 4.0)])
def test_cs1_1_k2_bounds_near_every_corner(w):
    """True poses within U_p, U_psi around each corner of K2 (interior corners and the closing one, window cut at S):
    the exact true G4 (Polyline.head_at) lies in [lo, hi] (CS1-1)."""
    lane = lane_of('O1-K2', (0.0, 0.0, 0.0))
    rng = np.random.default_rng(11)
    s_c = [c for c in lane.poly.cum if w[0] < c <= w[1] + 1e-9][0]
    cx, cy, _ = lane.path.point_at(min(s_c, lane.S))
    upk, ups = 0.004, math.radians(1.0)
    for _ in range(60):
        x, y = cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)
        psi = rng.uniform(-0.4, 0.4)
        lo, hi = cs.g4_bounds_jk(lane, x, y, psi, w[0], w[1], upk, ups)
        for _ in range(40):
            r, a = upk * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
            g = k_true_g4(lane, x + r * math.cos(a), y + r * math.sin(a), psi + rng.uniform(-ups, ups), w[0], w[1])
            assert lo - 1e-9 <= g <= hi + 1e-9, (w, lo, g, hi)


def test_cs1_2_fig8_window_end_between_grid_samples():
    """Codex CS1-2: Path8 window [w_lo, w_lo + 0.1] with w_lo between two grid samples; the true G4 0.0999999 m was
    below the old lo 0.1000088 (a certain violation that is not). The window ends are now in J_k; G2 lo - h/2."""
    lane = lane_of('O1-F05')
    w_lo = 0.00222366062
    xm, ym = -0.07267391251, -0.07267363911
    xt, yt, pt = -0.06913837205, -0.06913811176, 0.78539630914
    lo, hi = cs.g4_bounds_jk(lane, xm, ym, pt, w_lo, w_lo + 0.1, 0.005, math.radians(2.0))
    ex, ey = np.interp(w_lo, lane.path.s, lane.path.x), np.interp(w_lo, lane.path.s, lane.path.y)
    e_true = math.hypot(ex - xt, ey - yt)                 # the nearest lane point of the true centre is the window end
    assert abs(e_true - 0.0999999) < 1e-6 and lo <= e_true <= hi
    _, e_grid, _ = lane.project(xm, ym, w_lo, w_lo + 0.1)
    assert e_grid <= math.hypot(xm - ex, ym - ey) + 1e-12      # the window end is a candidate of e_perp
    assert e_grid - 0.005 - lane.h / 2 <= e_true              # G2 lo of the scorer


def test_cs1_3_at_end_on_the_window_cut_at_zero():
    """SC-4: a projection on s = 0 (window cut at 0) is at a window end: no U formula for G4 (CS1-3)."""
    lane = lane_of('O1-F05')
    tr = cs.run_tracker(np.array([0.01]), np.array([0.0]), np.array([0.0]), lane, np.array([True]), 0.68, 0.0)
    assert tr[0]['s_perp'] == 0.0 and tr[0]['at_end']


def _sync_fake(lane, offset_used, rate, short=False):
    from o1_time_sync import SyncResult
    c0, c1 = -0.5, lane.Tcmd + 0.5                         # C7b window, command time
    cmid = 0.5 * (c0 + c1)
    third = (c1 - c0) / 3.0
    ds = OFFSET + rate * (c0 + third / 2 - cmid)
    de = OFFSET + rate * (c1 - third / 2 - cmid)
    if short:
        ds = de = float('nan')
    return SyncResult(offset_used, 0.99, ds, 0.99, de, 0.99, de - ds, OFFSET, (c0 + T0 - OFFSET, c1 + T0 - OFFSET),
                      500, short), cmid


@pytest.mark.parametrize('extra', [0.0, 0.002])
def test_cs1_4_drift_covers_the_whole_window_to_the_deadline(tmp_path, monkeypatch, extra):
    """Codex CS1-4: linear drift 1 ms/s; the old term (|end - start| / 2 x t_deadline / T_cmd) was below the offset
    error at the deadline; the offset used may also differ from the mean of the thirds (extra)."""
    lane = lane_of('O1-F05')
    rate = 1e-3
    sync, cmid = _sync_fake(lane, OFFSET + extra, rate)
    monkeypatch.setattr(cs, 'clock_match', lambda frames, rows, task: (sync, None))
    out, _ = run(tmp_path, offset_s=None, offset_reason='', hs_kw=dict(troi_ppm=0.0))
    dl = out['t_deadline']
    worst = max(abs(OFFSET + rate * (t - cmid) - (OFFSET + extra)) for t in np.linspace(0.0, dl, 2001))
    assert out['U_t_parts']['drift_s'] >= worst - 1e-12, (out['U_t_parts'], worst)


def test_cs1_4_troi_ppm_is_a_floor_of_the_measured_rate(tmp_path, monkeypatch):
    lane = lane_of('O1-F05')
    sync, cmid = _sync_fake(lane, OFFSET, 0.0)
    monkeypatch.setattr(cs, 'clock_match', lambda frames, rows, task: (sync, None))
    out, _ = run(tmp_path, offset_s=None, offset_reason='', hs_kw=dict(troi_ppm=100.0))
    far = max(cmid, out['t_deadline'] - cmid)
    assert abs(out['U_t_parts']['drift_s'] - 100e-6 * far) < 1e-12


def test_cs1_5_tick_bound_over_lost_lines_and_window_ends():
    """Codex CS1-5: seq 101 lost hides a 25 ms tick; the pair 100 -> 102 bounds it; pairs across the window ends
    count; a log that does not reach both ends gives no finite bound."""
    rows = [dict(_seq=s, _t=int(round(t * 1e6))) for s, t in ((98, 0.985), (99, 0.995), (100, 1.000), (102, 1.035),
                                                              (103, 1.045), (104, 1.055))]
    assert cs.tick_max_s(rows, 1.000e6, 1.040e6) == pytest.approx(0.035)
    rows2 = [dict(_seq=s, _t=int(round(t * 1e6))) for s, t in ((1, 0.95), (2, 1.01), (3, 1.02), (4, 1.09))]
    assert cs.tick_max_s(rows2, 1.0e6, 1.05e6) == pytest.approx(0.07)     # both window ends crossed
    assert math.isinf(cs.tick_max_s(rows, 1.000e6, 1.2e6)) and math.isinf(cs.tick_max_s(rows, 0.9e6, 1.04e6))


def test_cs1_6_k1_with_a_short_c7b_window_scores(tmp_path):
    """Codex CS1-6: K1 (5 s) gives a short C7b window (no thirds): the old U_t was NaN and the scorer raised."""
    ks = (0.3, -0.4, math.radians(30.0))
    out, _ = run(tmp_path, 'O1-K1', frames=make_frames('O1-K1', k_start=ks), k_start=ks, allow_final_set=True,
                 offset_s=None, offset_reason='')
    gate_ok(out)
    assert out['sync']['short'] and 'short' in out['U_t_parts']['drift_src'] and math.isfinite(out['U_t_s'])
    assert out['outcome'] == 'dat'


def test_cs1_6_cli_any_error_is_exit_2(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError('bug')
    monkeypatch.setattr(cs, 'score', boom)
    cm.write_frames(tmp_path / 'frames.csv', make_frames('O1-F05'))
    hs_file(tmp_path)
    args = ['--frames', str(tmp_path / 'frames.csv'), '--esp32', 'x.csv', '--task', 'O1-F05', '--hang-so',
            str(tmp_path / 'hang_so.json')]
    assert cs.main(args) == 2 and '#ket_qua,ma=2' in capsys.readouterr().out


def test_cs1_7_tracking_error_after_the_run_window_closed(tmp_path):
    """Codex CS1-7: done at T_cmd + 1 s, then 0.8 s without frames: the frame caught after it is a tracking error
    OUTSIDE the union domain and does not spoil the run."""
    lane = lane_of('O1-F05')
    out, _ = run(tmp_path, frames=make_frames('O1-F05', drop=[(lane.Tcmd + 1.1, lane.Tcmd + 1.9)]))
    gate_ok(out)
    assert out['outcome'] == 'dat' and out['t_first_track_err'] is None and out['t1'] is None


def test_cs1_8_tracking_error_on_the_frame_that_starts_a_long_gap(tmp_path, monkeypatch):
    """Codex CS1-8: the frame at t = 5 s is a tracking error (beyond the reopened window; not a suspect here, stub)
    AND the last frame before a 0.3 s gap; its own lo exceeds D. The error wins: no violation before t1, the run is
    lost to the measurement system (the old code took the frame as 'before t1': khong_dat)."""
    suspects_stub(monkeypatch)
    lane = lane_of('O1-F05')
    k = (4 + 5) * 60
    s5, x5, y5, _ = lane.command(5.0)
    p = lane.path
    xa, ya = np.interp(s5 + 1.6, p.s, p.x), np.interp(s5 + 1.6, p.s, p.y)
    fr = make_frames('O1-F05', jump={k: (xa - x5, ya - y5)}, drop=[(5.0 + 0.5 / FPS, 5.3)])
    out, rows = run(tmp_path, frames=fr)
    gate_ok(out)
    j = frame_at(rows, 5.0)
    assert rows[j]['suspect'] == 0 and rows[j]['track_err'] == 1 and rows[j]['lo_g2'] > 0.1
    assert out['outcome'] == 'hong_he_do' and not out['cat_t1']
    assert out['t1_kind'] == 'loi_theo_doi' and abs(out['t1'] - 5.0) < 1e-6


def test_cs1_9_way_2_speeds_without_the_suspect_frames(tmp_path):
    """Codex CS1-9: still at the end after T_cmd, one frame jumps 30 mm every 0.3 s. With the speeds of c7_measure
    (every valid frame) neither way held t_d (G1 khong_dat); way 2 (suspects as gaps) now completes: G1 chua."""
    lane = lane_of('O1-F05')
    t = np.arange(-4.0, lane.Tcmd + 3.0, 1.0 / FPS)
    jump = {k: (0.03, 0.0) for k, tk in enumerate(t) if tk > lane.Tcmd + 0.05 and (k % 18) == 0}
    out, _ = run(tmp_path, frames=make_frames('O1-F05', jump=jump))
    g = gate_ok(out)
    assert out['n_suspect_pos'] >= sum(1 for k in jump if t[k] < out['t_deadline']) - 1
    assert g['g1']['t_done'][1] is not None and g['g1']['state'] == 'chua_ket_luan'


def test_cs1_10_suspects_not_converging():
    """Codex CS1-10: 101 samples, half 2, min 3, one coordinate alternating 0.10 / 0.16 m: the passes cycle; the
    suspect set is not empty, flagged not converged. (Since CS2-10 a suspect without nmin non-suspect neighbours in
    the final set has no prediction: NaN, U_p,k not finite; test_cs2_10_*.)"""
    t = np.arange(101) / 60.0
    x = np.where(np.arange(101) % 2, 0.16, 0.10)
    sus, pred, passes, conv = cs.mark_suspects(t, x, np.zeros(101), 0.02, 2, 3)
    assert not conv and passes == cs.SUSPECT_PASSES
    assert sus.sum() > 50 and np.all(np.isnan(pred[sus]) | (pred[sus] > 0.02))


def test_cs1_11_alignment_frames_must_be_still_in_translation(tmp_path):
    """Codex CS1-11: before t = 0 the car drives straight (no rotation) at 0.2 m/s: no still frame, the yaw of the run
    is not compared (G4, G5 undefined) instead of aligning on a moving car."""
    fr = make_frames('O1-F05', measured=False)
    for r in fr:
        tk = r['t_s'] - T0 + OFFSET
        if tk < 0.0:
            r['x'] += 0.2 * tk
    cm.apply_gates(fr, 0.3)
    out, _ = run(tmp_path, frames=fr)
    gate_ok(out)
    assert not out['yaw_check']['compared'] and 'speed' in out['yaw_check']['reason']
    assert out['outcome'] == 'chua_ket_luan'


def test_cs1_12_imu_reset_before_the_alignment_part(tmp_path):
    """Codex CS1-12: an IMU reset 4 s before t = 0 (pre_s 3 s, the old cut from t0 - 5 s): the yaw after it is
    continuous, the run is compared."""
    out, _ = run(tmp_path, log_kw=dict(reset_at=-4.0))
    gate_ok(out)
    assert out['yaw_check']['compared'] and out['outcome'] == 'dat' and out['yaw_check']['log_from_s'] > -4.1


def test_cs1_12_reset_inside_the_alignment_part_keeps_its_effect(tmp_path):
    """A reset inside [t0 - pre_s, t0): the cut stays before the alignment part; no frame after the reset has a yaw."""
    out, _ = run(tmp_path, log_kw=dict(reset_at=-1.0))
    g = gate_ok(out)
    assert out['yaw_check']['log_from_s'] < -3.0 and g['g5']['n_undef'] == g['g5']['n_frames']


def test_cs1_13_cs2_4_old_frames_are_refused(tmp_path):
    """Codex CS2-4: eight shift vectors of a frame spread over four directions, 13 px long: the frames.csv before
    17/11 holds shift_px = 13 (median of the lengths) while the length of the median vector is 0. The old scorer
    called the camera bumped (13 > 12 px); such a file is now refused (track again), as a track.json saying so."""
    sv = [(13.0, 0.0), (-13.0, 0.0), (0.0, 13.0), (0.0, -13.0)] * 2
    assert cm.shift_of(sv) == (0.0, 13.0)
    fr = make_frames('O1-F05')
    for r in fr:
        r['shift_px'], r['shift_mag_px'] = 13.0, float('nan')        # frames.csv before 17/11
    with pytest.raises(ValueError, match='track on the video again'):
        run(tmp_path, frames=fr)
    with pytest.raises(ValueError, match='CS2-4'):
        run(tmp_path, track_info=dict(shift_kind='median_magnitude'))
    out, _ = run(tmp_path)
    assert out['shift_kind'] == 'median_vector' and out['outcome'] == 'dat'


def test_cs1_14_locked_command_thresholds_are_applied(tmp_path, cmd_ok):
    """Codex CS1-14: a packet-age bound of 150 ms with a 50 ms threshold in the constants: lost to the command path
    (the old scorer only read the flags of o1_cmd_path); (4) blocked by the ultrasonic is not compared."""
    cmd_ok['cond'] = dict(cond2=dict(value_m=0.004), cond3=dict(value_ms=150.0),
                          cond4=dict(value_m=0.5, applies=False))
    th = dict(cond2_m=0.02, cond3_ms=50.0, cond4_m=0.03)
    out, _ = run(tmp_path, hs_kw=dict(duong_lenh=th))
    gate_ok(out)
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == 'duong_lenh'
    assert out['cmd_path']['threshold_flags'] == ['cond3_vuot_nguong: 150.0 ms > 50 ms']
    cmd_ok['cond'] = dict(cond2=dict(value_m=0.004), cond3=dict(value_ms=None), cond4=dict(value_m=0.5, applies=False))
    out, _ = run(tmp_path, hs_kw=dict(duong_lenh=th))
    assert out['outcome'] == 'hong_he_do' and 'cond3_thieu_bang_chung' in out['cmd_path']['threshold_flags'][0]
    cmd_ok['cond'] = dict(cond2=dict(value_m=0.004), cond3=dict(value_ms=30.0), cond4=dict(value_m=0.05, applies=True))
    out, _ = run(tmp_path, hs_kw=dict(duong_lenh=th))      # (4) applies (no ultrasonic block): compared
    assert out['outcome'] == 'hong_he_do' and out['cmd_path']['threshold_flags'] == ['cond4_vuot_nguong: 50.0 mm > 30 mm']
    cmd_ok['cond'] = {}
    out, _ = run(tmp_path)                                 # no thresholds in a de_xuat file: values only
    assert out['outcome'] == 'dat' and out['cmd_path']['threshold_flags'] == []


def test_real_cmd_path_end_to_end(tmp_path):
    """o1_cmd_path NOT stubbed: firmware-like command columns, '#us' / '#us_end' events, the sender '#test_end' with
    the send times (Codex 16/11 test gap). Clean: dat with the values reported; a locked packet-age threshold below
    the observed bound: lost to the command path."""
    log = make_log(tmp_path / 'esp32.csv', 'O1-F05', cmd_real=True)
    hs = hs_file(tmp_path)
    out, _ = cs.score(make_frames('O1-F05'), log, 'O1-F05', hs, offset_s=OFFSET, offset_reason='test',
                      sender=log + '_sender.txt', offset_anchor_s=0.0, offset_err_s=0.0)
    gate_ok(out)
    c = out['cmd_path']
    assert out['outcome'] == 'dat' and c['flags'] == [] and 20.0 < c['cond3']['value_ms'] < 60.0
    assert c['cond2']['value_m'] < 0.001 and c['cond4']['applies'] is True
    hs = hs_file(tmp_path, duong_lenh=dict(cond2_m=0.02, cond3_ms=25.0, cond4_m=0.03))
    out, _ = cs.score(make_frames('O1-F05'), log, 'O1-F05', hs, offset_s=OFFSET, offset_reason='test',
                      sender=log + '_sender.txt', offset_anchor_s=0.0, offset_err_s=0.0)
    gate_ok(out)
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == 'duong_lenh'


def test_real_cmd_path_without_sender_is_lost(tmp_path):
    log = make_log(tmp_path / 'esp32.csv', 'O1-F05', cmd_real=True)
    out, _ = cs.score(make_frames('O1-F05'), log, 'O1-F05', hs_file(tmp_path), offset_s=OFFSET, offset_reason='test',
                      offset_anchor_s=0.0, offset_err_s=0.0)
    assert out['outcome'] == 'hong_he_do' and out['cmd_path']['lost']


def test_cs1_15_frame_gates_must_match_the_constants(tmp_path):
    """Codex CS1-15: the constants hold the frame gates; frames.csv gated with another (2a) threshold is refused,
    without track.json it is refused, with the same gates it is scored."""
    gk = dict(cm.DEFAULT_GATES)
    with pytest.raises(ValueError, match='other thresholds'):
        run(tmp_path, hs_kw=dict(cong_khung=gk), track_info=dict(gates=dict(gk, car_res_sim_m=0.001)))
    with pytest.raises(ValueError, match='track.json'):
        run(tmp_path, hs_kw=dict(cong_khung=gk))
    out, _ = run(tmp_path, hs_kw=dict(cong_khung=gk), track_info=dict(gates=dict(gk, rule='x'), frames_sha256='a'),
                 frames_sha256='a')
    assert out['gate_check']['checked'] and out['outcome'] == 'dat'


def test_cs2_5_track_json_must_belong_to_the_frames_scored(tmp_path):
    """Codex CS2-5: track.json with the gates of the constants kept, frames.csv replaced by another file (gated with
    1.5 mm): the old scorer compared the six thresholds only. Now its frames_sha256 must be the hash of the frames
    scored: another hash is refused (also before C8), a missing one is refused when the constants hold the gates."""
    gk = dict(cm.DEFAULT_GATES, car_res_sim_m=0.001)
    ti = dict(gates=dict(gk), frames_sha256='hash_of_the_c8_frames')
    with pytest.raises(ValueError, match='frames_sha256'):
        run(tmp_path, hs_kw=dict(cong_khung=gk), track_info=ti, frames_sha256='hash_of_the_old_frames')
    with pytest.raises(ValueError, match='CS2-5'):
        run(tmp_path, hs_kw=dict(cong_khung=gk), track_info=dict(gates=dict(gk)), frames_sha256='x')
    with pytest.raises(ValueError, match='CS2-5'):
        run(tmp_path, hs_kw=dict(cong_khung=gk), track_info=ti)
    with pytest.raises(ValueError, match='frames_sha256'):
        run(tmp_path, track_info=ti, frames_sha256='hash_of_the_old_frames')
    out, _ = run(tmp_path, track_info=ti, frames_sha256='hash_of_the_c8_frames')
    assert out['gate_check']['frames_sha256_match'] is True


def test_cs2_5_cli_ties_track_json_to_the_file(tmp_path):
    """Through main: track.json beside frames.csv with the hash of another file is refused (exit 2)."""
    cm.write_frames(tmp_path / 'frames.csv', make_frames('O1-F05'))
    hs_file(tmp_path)
    log = make_log(tmp_path / 'esp32.csv', 'O1-F05')
    (tmp_path / 'track.json').write_text(json.dumps(dict(frames_sha256='0' * 64)), encoding='utf-8')
    args = ['--frames', str(tmp_path / 'frames.csv'), '--esp32', log, '--task', 'O1-F05', '--hang-so',
            str(tmp_path / 'hang_so.json'), '--offset-s', str(OFFSET), '--offset-reason', 't', '--offset-anchor-s', '0',
            '--offset-err-s', '0']
    assert cs.main(args) == 2
    (tmp_path / 'track.json').write_text(json.dumps(dict(frames_sha256=cm.sha256(tmp_path / 'frames.csv'))),
                                         encoding='utf-8')
    assert cs.main(args + ['--out', str(tmp_path / 'score.json')]) == 0


def test_cs1_16_lines_without_log_seq_are_refused(tmp_path):
    """Codex CS1-16: two identical lines with an empty log_seq (the second marked duplicate) plus one UNMARKED copy
    of a normal line balanced the old count check (1 = 1)."""
    p = make_log(tmp_path / 'esp32.csv', 'O1-F05')
    rows = list(csv.DictReader(ln for ln in open(p, encoding='utf-8') if not ln.startswith('#')))
    empty = dict(rows[50], log_seq='')
    rows = rows[:300] + [empty, dict(empty), dict(rows[299])] + rows[300:]
    status = ['ok'] * 301 + ['duplicate', 'ok'] + ['ok'] * (len(rows) - 303)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        w.writeheader()
        w.writerows(rows)
    with open(p + '.segments.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['row', 'log_seq', 'timestamp_us', 'segment', 'status'])
        for i, (r, s) in enumerate(zip(rows, status), start=1):
            w.writerow([i, r['log_seq'], r['timestamp_us'], 0, s])
    with pytest.raises(ValueError, match='without log_seq'):
        cs.read_esp32(p)


def test_cs1_18_suspect_assumption_not_checked_clamps_lo(tmp_path):
    """Codex CS1-18 / O1 3.3: a suspect frame at 0.16 m among frames at 0.13 m has lo = 0.125 > D; without the C8 check
    of the assumption ('chua_kiem', 'khong_dung') its lo is clamped at D (no certain violation from it)."""
    lane = lane_of('O1-F05')
    k = (4 + 5) * 60 + 30                                  # t = 5.5 s
    _, _, _, hc = lane.command(5.5)
    fr = make_frames('O1-F05', jump={k: (-0.03 * math.sin(hc), 0.03 * math.cos(hc))},
                     lat_fn=lambda t: 0.13 if 5.0 <= t < 6.0 else 0.0)
    for gd, clamped in (('chua_kiem', True), ('khong_dung', True), ('dung', False)):
        out, rows = run(tmp_path, frames=fr, hs_kw=dict(gia_dinh_nghi_vi_tri=gd))
        r = rows[frame_at(rows, 5.5)]
        assert r['suspect'] == 1 and r['e_perp'] > 0.155
        assert (r['lo_g2'] <= 0.1) == clamped and (r['lo_g4'] <= 0.1 or not clamped)


def test_cs1_19_imu_range_exceeded_is_reported(tmp_path):
    md = {'O1-F05': dict(goc_quay_deg=1e5, wz_dps=5.0, alpha_dps2=1e6)}
    out, _ = run(tmp_path, hs_kw=dict(imu=dict(mien_dong=md)))
    y = out['yaw_check']
    assert y['mien_dong_vuot'] == 'wz_dps' and 0.0 <= y['t_mien_dong_vuot'] < 5.0 and y['n_after_mien_dong'] > 0
    assert out['outcome'] == 'dat'                         # reported only: the handling is Liam's (O1 3.3)
    md = {'O1-F05': dict(goc_quay_deg=1e5, wz_dps=4000.0, alpha_dps2=1e6)}
    out, _ = run(tmp_path, hs_kw=dict(imu=dict(mien_dong=md)))
    assert out['yaw_check']['t_mien_dong_vuot'] is None


def test_cs1_khoa_c8_needs_every_locked_block(tmp_path):
    full = dict(trang_thai='khoa_c8', duong_lenh=dict(cond2_m=0.02, cond3_ms=50.0, cond4_m=0.03),
                cong_khung=dict(cm.DEFAULT_GATES), gia_dinh_nghi_vi_tri='dung', nhanh_tick='giu',
                imu=dict(mien_dong={m: dict(goc_quay_deg=720.0, wz_dps=200.0, alpha_dps2=2000.0)
                                    for m in ('O1-F02', 'O1-F05', 'O1-F08', 'O1-K1', 'O1-K2')}))
    hs = hs_file(tmp_path, **json.loads(json.dumps(full)))
    assert hs['trang_thai'] == 'khoa_c8' and hs['gia_dinh_nghi_vi_tri'] == 'dung'
    for drop in ('duong_lenh', 'cong_khung', 'nhanh_tick'):          # nhanh_tick: O1 v11.33 OG3-4
        bad = json.loads(json.dumps(full))
        del bad[drop]
        with pytest.raises(ValueError, match=drop):
            hs_file(tmp_path, **bad)
    for gd in ('chua_kiem', 'x'):
        bad = json.loads(json.dumps(full))
        bad['gia_dinh_nghi_vi_tri'] = gd
        with pytest.raises(ValueError, match='gia_dinh'):
            hs_file(tmp_path, **bad)
    bad = json.loads(json.dumps(full))
    del bad['imu']['mien_dong']['O1-K2']
    with pytest.raises(ValueError, match='mien_dong'):
        hs_file(tmp_path, **bad)


def test_cs1_20_suspect_beyond_the_reopened_window_is_tracking_error(tmp_path):
    """Codex CS1-20 / O1 W11-4: a suspect frame 1.6 m ahead along the lane still hits the end of the window reopened
    to 1.23 m: a tracking error (the old code exempted suspect frames)."""
    lane = lane_of('O1-F05')
    k = (4 + 5) * 60
    s5, x5, y5, _ = lane.command(5.0)
    p = lane.path
    xa, ya = np.interp(s5 + 1.6, p.s, p.x), np.interp(s5 + 1.6, p.s, p.y)
    out, rows = run(tmp_path, frames=make_frames('O1-F05', jump={k: (xa - x5, ya - y5)}))
    gate_ok(out)
    j = frame_at(rows, 5.0)
    assert rows[j]['suspect'] == 1 and rows[j]['reopen'] == 1 and rows[j]['track_err'] == 1
    assert out['outcome'] == 'hong_he_do' and abs(out['t_first_track_err'] - 5.0) < 1e-6


# ---------------------------------------------------------------- other test gaps Codex listed
def test_violation_exactly_on_the_last_frame_before_a_gap(tmp_path, monkeypatch):
    """R3V-8: 'before t1' of a long gap includes the last valid frame before it (not a suspect here, stub)."""
    suspects_stub(monkeypatch)
    fr = make_frames('O1-F05', drop=[(8.0, 8.3)], lat_fn=lambda t: 0.13 if 7.98 <= t < 8.0 else 0.0)
    out, rows = run(tmp_path, frames=fr)
    gate_ok(out)
    assert out['t1_kind'] == 'khoang_trong' and out['outcome'] == 'khong_dat' and out['cat_t1']


def test_window_without_any_valid_frame(tmp_path):
    lane = lane_of('O1-F05')
    out, _ = run(tmp_path, frames=make_frames('O1-F05', drop=[(0.0, lane.Tcmd + 3.0)]))
    assert out['outcome'] == 'hong_he_do' and out['gate_v3'] is None


def test_certain_violation_only_in_the_union_domain(tmp_path, monkeypatch):
    """Two domains: way 1 done at t_done[0], way 2 never (suspect blocks of 8 frames every 0.4 s, stub); a lateral
    0.13 m after t_done[0] is in the union only: the condition is chua_ket_luan, not khong_dat (O1 3.3 R2V-3)."""
    lane = lane_of('O1-F05')
    blk = lambda tk: tk > lane.Tcmd - 0.5 and ((tk - lane.Tcmd) % 0.4) < 8.5 / FPS   # noqa: E731
    suspects_stub(monkeypatch, blk)
    base = run(tmp_path, frames=make_frames('O1-F05', measured=False))[0]
    td0 = base['gate_v3']['g1']['t_done'][0]
    fr = make_frames('O1-F05', measured=False, lat_fn=lambda tt: 0.13 if td0 + 0.3 <= tt < td0 + 0.6 else 0.0)
    out, _ = run(tmp_path, frames=fr)
    g = gate_ok(out)
    assert g['g1']['t_done'][0] == pytest.approx(td0) and g['g1']['t_done'][1] is None
    assert g['g2']['chung']['lo'] <= 0.1 < g['g2']['lo'] and g['g2']['state'] == 'chua_ket_luan'


# ---------------------------------------------------------------- Codex vong 2 (17/11, CS2-1..CS2-10), each failing on
# hil 769e62c; review 03_CODE/docs/review/2026-11-17_Tham_dinh_c7_score_o1_vong2.md
def test_cs2_1_offset_needs_an_anchor_and_the_drift_runs_from_it(tmp_path):
    """Codex CS2-1: an offset established at t = -100 s with a clock-rate difference at the bound (1000 ppm) is
    already 100 ms off at t = 0; the old drift term was troi_ppm x t_deadline (7 ms on K1). Now --offset-s needs its
    anchor and error, and the drift runs from the anchor to the far end of [0, t_deadline]."""
    with pytest.raises(ValueError, match='offset-anchor-s'):
        run(tmp_path, offset_anchor_s=None)
    with pytest.raises(ValueError, match='offset-anchor-s'):
        run(tmp_path, offset_err_s=None)
    out, _ = run(tmp_path, hs_kw=dict(troi_ppm=1000.0), offset_anchor_s=-100.0, offset_err_s=0.004)
    dl = out['t_deadline']
    p = out['U_t_parts']
    assert p['drift_s'] == pytest.approx(1e-3 * (100.0 + dl))
    assert out['U_t_s'] == pytest.approx(p['ghep_s'] + 0.004 + p['drift_s'] + p['tick_s'])
    assert list(out['offset_anchor_s']) == [-100.0, -100.0] and out['offset_err_s'] == 0.004
    out, _ = run(tmp_path, hs_kw=dict(troi_ppm=1000.0), offset_anchor_s=2.0, offset_err_s=0.0)
    assert out['U_t_parts']['drift_s'] == pytest.approx(1e-3 * max(2.0, dl - 2.0))


def test_cs2_2_log_ending_after_the_run_closed_but_before_the_deadline(tmp_path):
    """Codex CS2-2: done about 1 s after T_cmd, deadline T_cmd + 2 s, the log stops at T_cmd + 1.5 s: no tick is
    missing over the domain scored [0, t_done], so the run is scored (the old caller asked for ticks up to the deadline:
    U_t infinite, hong_he_do). A log ending before t_done still gives no finite U_t. The drift still runs to the
    deadline (SC-12)."""
    out, _ = run(tmp_path, log_kw=dict(post=1.5))
    g = gate_ok(out)
    assert out['outcome'] == 'dat' and g['g1']['state'] == 'dat'
    assert out['U_t_parts']['tick_to_s'] == pytest.approx(out['t_end_union'])
    assert out['U_t_parts']['drift_s'] == pytest.approx(200e-6 * out['t_deadline'])
    out, _ = run(tmp_path, log_kw=dict(post=0.5))
    assert out['outcome'] == 'hong_he_do' and 'U_t not bounded' in out['outcome_reasons'][0]


def test_cs2_3_no_suspect_frame_the_two_ways_of_g1_are_the_same(tmp_path):
    """Codex CS2-3: after T_cmd the car slides sideways at 0.00999949 m/s (just under v_d), no frame is suspect.
    frames.csv rounds the speed to 0.009999 (way 1 used it: done) and the poses to 1 um (way 2 rebuilt the speed
    from them: about 0.0100002, not held): G1 chua_ket_luan with two identical sets of frames. Now both ways take
    the speeds from the same poses and derivative: the two ways and the two domains coincide."""
    lane = lane_of('O1-F05')
    fr = make_frames('O1-F05', lat_fn=lambda t: 0.00999949 * (t - lane.Tcmd) if t > lane.Tcmd else 0.0)
    cm.write_frames(tmp_path / 'frames.csv', fr)
    fr = cm.read_frames(str(tmp_path / 'frames.csv'))
    out, rows = run(tmp_path, frames=fr, hs_kw=dict(v_d=0.01))
    g = gate_ok(out)
    assert out['n_suspect_pos'] == 0
    assert g['g1']['t_done'][0] == g['g1']['t_done'][1] and g['g1']['state'] in ('dat', 'khong_dat')
    assert out['t_end_common'] == out['t_end_union']
    assert all(r['speed'] == r['speed_way2'] for r in rows)


def test_cs2_8_suspect_assumption_rejected_never_gives_dat(tmp_path):
    """Codex CS2-8: K1 straight side, one frame mid-run 30 mm sideways (a position suspect, U_p,k about 35 mm): G2 hi
    65 mm, G4 by the formula about 73 mm, all below D; with the C8 check 'khong_dung' (or not done) the old scorer
    still gave dat. The frame now gives chua_ket_luan only (hi not finite), and dat stays when the check held."""
    ks = (0.3, -0.4, math.radians(30.0))
    lane = lane_of('O1-K1', ks)
    k = int(round((4.0 + 2.5) * FPS))
    _, _, _, hc = lane.command(2.5)
    h = hc + ks[2]
    fr = make_frames('O1-K1', k_start=ks, jump={k: (-0.03 * math.sin(h), 0.03 * math.cos(h))})
    for gd in ('khong_dung', 'chua_kiem', 'dung'):
        out, rows = run(tmp_path, 'O1-K1', frames=fr, k_start=ks, allow_final_set=True,
                        hs_kw=dict(gia_dinh_nghi_vi_tri=gd))
        g = gate_ok(out)
        r = rows[frame_at(rows, 2.5)]
        assert out['n_suspect_pos'] == 1 and r['suspect'] == 1 and abs(r['e_perp'] - 0.03) < 2e-3
        if gd == 'dung':
            assert out['outcome'] == 'dat' and r['hi_g2'] < 0.1 and r['hi_g4'] < 0.1
        else:
            assert out['outcome'] == 'chua_ket_luan'
            assert all(math.isinf(r[f'hi_{c}']) and r[f'lo_{c}'] <= 0.1 for c in ('g2', 'g3', 'g4'))
            assert all(g[c]['n_undef'] >= 1 and g[c]['state'] == 'chua_ket_luan' for c in ('g2', 'g3', 'g4'))


def _dyn_S(ta, psi, tg=(), gz=()):
    ta = np.asarray(ta, float)
    return dict(t=ta, psi=np.asarray(psi, float), t_all=ta, psi_all=np.asarray(psi, float),
                n_all=np.arange(len(ta)), t_recv=ta.copy(), keep=np.ones(len(ta), bool), tg=np.asarray(tg, float),
                gz=np.asarray(gz, float), period_fit_s=0.05, h_lim=0.105, t_break=math.inf, resets=[])


def test_cs2_9_turn_across_t0_and_gyro_peaks_are_reported():
    """Codex CS2-9: yaw reports at -0.025, 0.025, 0.075, 0.125 s with yaw 0, 0.1, 0.1, 0.1 rad: the pair across t = 0
    turns 0.1 rad (2.865 deg from t = 0 if linear), above a 2 deg limit; the old mask from t = 0 saw no turn. A gyro
    peak between two flat yaw reports exceeds the rate limit; a gyro step the acceleration limit."""
    big = dict(goc_quay_deg=1e5, wz_dps=5000.0, alpha_dps2=1e6)
    S = _dyn_S([-0.025, 0.025, 0.075, 0.125], [0.0, 0.1, 0.1, 0.1])
    r = cs.dyn_exceed(S, 0.0, 0.12, dict(big, goc_quay_deg=2.0))
    assert r[0] is not None and r[1] == 'goc_quay_deg' and 0.0 <= r[0] <= 0.03
    tg = np.arange(-0.1, 0.5, 0.01)
    gz = np.where(np.abs(tg - 0.2) < 1e-9, 1.0, 0.0)               # 57 deg/s for one report
    S = _dyn_S(np.arange(-0.1, 0.5, 0.05), np.zeros(12), tg, gz)
    r = cs.dyn_exceed(S, 0.0, 0.45, dict(big, wz_dps=30.0))
    assert r[1] == 'wz_dps' and r[0] == pytest.approx(0.2)
    r = cs.dyn_exceed(S, 0.0, 0.45, dict(big, alpha_dps2=3000.0))   # 57 deg/s within 10 ms: 5730 deg/s^2
    assert r[1] == 'alpha_dps2' and r[0] == pytest.approx(0.2)
    assert cs.dyn_exceed(S, 0.0, 0.45, big)[0] is None


def test_cs2_9_range_checked_without_the_yaw_alignment(tmp_path):
    """The U_IMU range is reported even when the camera yaw cannot be aligned (car moving before t = 0, CS1-11)."""
    fr = make_frames('O1-F05', measured=False)
    for r in fr:
        tk = r['t_s'] - T0 + OFFSET
        if tk < 0.0:
            r['x'] += 0.2 * tk
    cm.apply_gates(fr, 0.3)
    md = {'O1-F05': dict(goc_quay_deg=1e5, wz_dps=5.0, alpha_dps2=1e6)}
    out, _ = run(tmp_path, frames=fr, hs_kw=dict(imu=dict(mien_dong=md)))
    y = out['yaw_check']
    assert not y['compared'] and y['mien_dong_vuot'] == 'wz_dps' and y['t_mien_dong_vuot'] is not None


def _ref_pred(t, x, y, sus, half, nmin):
    """Independent prediction (O1 3.3: from the non-suspect frames only)."""
    out = np.full(len(t), np.nan)
    for k in range(len(t)):
        nb = [j for j in range(max(0, k - half), min(len(t), k + half + 1)) if j != k and not sus[j]]
        if len(nb) >= nmin:
            out[k] = math.hypot(x[k] - np.polyval(np.polyfit(t[nb] - t[k], x[nb], 2), 0.0),
                                y[k] - np.polyval(np.polyfit(t[nb] - t[k], y[nb], 2), 0.0))
    return out


def test_cs2_10_not_converging_predicts_from_the_final_set():
    """Codex CS2-10: the 101 alternating samples of CS1-10: the final set leaves almost no non-suspect frame, so most
    suspects have no supported prediction (NaN, U_p,k not finite); the old code kept |p - p^| from earlier passes
    that used frames now suspect. Every finite |p - p^| is the prediction from the FINAL non-suspect set, and no frame
    outside the set is predicted above pred_m."""
    t = np.arange(101) / 60.0
    x = np.where(np.arange(101) % 2, 0.16, 0.10)
    y = np.zeros(101)
    sus, pred, passes, conv = cs.mark_suspects(t, x, y, 0.02, 2, 3)
    ref = _ref_pred(t, x, y, sus, 2, 3)
    ref = np.where(sus & ~(ref > 0.02), np.nan, ref)            # CS3-6: in the set, not above thr: unclassified
    assert not conv and np.allclose(pred, ref, equal_nan=True, atol=1e-12)
    assert np.isnan(pred[sus]).sum() > 50
    assert not np.any(~sus & (np.nan_to_num(pred, nan=0.0) > 0.02))


def test_cs2_10_union_of_the_last_two_passes_is_grown_until_closed():
    """A 14-sample series (found by search, 17/11) that does not converge: the union of the last two passes {3, 4, 5, 7}
    leaves sample 9 predicted 0.03 m off from the samples outside it, so the set grows to {3, 4, 5, 7, 9}; starting
    from the last pass alone would end on {3, 4, 5, 6, 7, 9} (mutants 'no growth', 'last pass only' of 17/11)."""
    x = np.array([-0.0001, 0.0306, 0.0294, -0.0004, 0.031, -0.0003, 0.0297, 0.0308, 0.0002, 0.0, 0.0302, 0.0287,
                  0.0304, 0.0613])
    t = np.arange(len(x)) / 60.0
    sus, pred, passes, conv = cs.mark_suspects(t, x, np.zeros(len(x)), 0.02, 2, 3)
    assert not conv and np.nonzero(sus)[0].tolist() == [3, 4, 5, 7, 9]
    assert not np.any(~sus & (np.nan_to_num(pred, nan=0.0) > 0.02))
    ref = _ref_pred(t, x, np.zeros(len(x)), sus, 2, 3)
    assert np.allclose(pred, np.where(sus & ~(ref > 0.02), np.nan, ref), equal_nan=True, atol=1e-12)


def test_cs2_3_g1_does_not_read_the_speed_column(tmp_path):
    """CS2-3 (mutant 17/11 'way 1 from the CSV speed' survived): the speed and wz columns of frames.csv replaced by
    1 m/s and 0: both t_done stay those of the poses (the column is reported only)."""
    base, _ = run(tmp_path)
    fr = make_frames('O1-F05')
    for r in fr:
        r['speed'], r['wz'] = 1.0, 0.0
    out, rows = run(tmp_path, frames=fr)
    assert base['gate_v3']['g1']['t_done'][0] is not None
    assert out['gate_v3']['g1']['t_done'] == base['gate_v3']['g1']['t_done']
    assert all(r['speed_csv'] == 1.0 for r in rows)


def test_cs2_10_suspect_without_a_prediction_is_undefined(tmp_path, monkeypatch):
    """A suspect frame with no supported prediction: U_p,k not finite, G2-G4 of the frame undefined (n_undef), even
    when the C8 check of the assumption held (the old code used U_p alone)."""
    def fake(t, x, y, thr, half, nmin):
        sus = np.abs(t - 5.0) < 0.5 / FPS
        return sus, np.where(sus, np.nan, 0.001), 10, False
    monkeypatch.setattr(cs, 'mark_suspects', fake)
    out, rows = run(tmp_path, hs_kw=dict(gia_dinh_nghi_vi_tri='dung'))
    g = gate_ok(out)
    r = rows[frame_at(rows, 5.0)]
    assert r['suspect'] == 1 and math.isinf(r['U_p_k']) and out['n_suspect_no_pred'] == 1
    assert all(math.isinf(r[f'hi_{c}']) and g[c]['n_undef'] == 1 for c in ('g2', 'g3', 'g4'))
    assert out['outcome'] == 'chua_ket_luan'


# ---------------------------------------------------------------- test gaps Codex listed in round 2
def test_cs1_14_only_condition_2_above_and_exactly_on_the_thresholds(tmp_path, cmd_ok):
    th = dict(cond2_m=0.02, cond3_ms=50.0, cond4_m=0.03)
    cmd_ok['cond'] = dict(cond2=dict(value_m=0.025), cond3=dict(value_ms=30.0), cond4=dict(value_m=0.01, applies=True))
    out, _ = run(tmp_path, hs_kw=dict(duong_lenh=th))
    assert out['outcome'] == 'hong_he_do'
    assert out['cmd_path']['threshold_flags'] == ['cond2_vuot_nguong: 25.0 mm > 20 mm']
    cmd_ok['cond'] = dict(cond2=dict(value_m=0.02), cond3=dict(value_ms=50.0), cond4=dict(value_m=0.03, applies=True))
    out, _ = run(tmp_path, hs_kw=dict(duong_lenh=th))       # equal to the thresholds: not above
    assert out['outcome'] == 'dat' and out['cmd_path']['threshold_flags'] == []


def _seg_dist(p, a, b):
    d = b - a
    u = max(0.0, min(1.0, float(np.dot(p - a, d) / np.dot(d, d))))
    return float(np.hypot(*(p - a - u * d)))


def _window_dist(P, p, w_lo, w_hi):
    """Exact distance from p to the part [w_lo, w_hi] (cut at 0 and S) of the polyline P, by its own arc lengths."""
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    lo, hi = max(w_lo, 0.0), min(w_hi, cum[-1])
    best = math.inf
    for i in range(len(P) - 1):
        a, b = max(lo, cum[i]), min(hi, cum[i + 1])
        if a > b:
            continue
        u = (P[i + 1] - P[i]) / (cum[i + 1] - cum[i])
        best = min(best, _seg_dist(p, P[i] + (a - cum[i]) * u, P[i] + (b - cum[i]) * u))
    return best


def test_cs1_1_k2_e_perp_through_the_scorer_at_every_corner(tmp_path):
    """K2 through score (not the helpers alone): each frame's e_perp is the exact distance to the sides inside its
    two-sided window (the car 20 mm off the lane, around every corner), and the last frames project onto the end S
    with at_end."""
    ks = (0.0, 0.0, 0.0)
    lane = lane_of('O1-K2', ks)
    out, rows = run(tmp_path, 'O1-K2', frames=make_frames('O1-K2', k_start=ks, lateral=0.02), k_start=ks,
                    allow_final_set=True)
    gate_ok(out)
    P = lane.poly.p
    for r in rows:
        exact = _window_dist(P, np.array([r['x'], r['y']]), r['w_lo'], r['w_hi'])
        assert abs(r['e_perp'] - exact) < 1e-9, (r['t'], r['e_perp'], exact)
    end = [r for r in rows if r['t'] > lane.Tcmd + 0.1]
    assert end and all(r['s_perp'] == pytest.approx(lane.S) and r['at_end'] == 1 for r in end)


def test_cs1_12_unbridged_imu_gaps_just_before_the_alignment_part(tmp_path):
    """Two IMU gaps the gyro cannot bridge, the second ending just before the alignment part [t0 - 3 s, t0): the cut
    moves past both and the run is compared."""
    out, _ = run(tmp_path, log_kw=dict(imu_gaps=[(-4.6, -4.2), (-3.6, -3.05)]))
    gate_ok(out)
    y = out['yaw_check']
    assert y['compared'] and out['outcome'] == 'dat' and -3.1 < y['log_from_s'] < -3.0


def test_cs1_16_unmarked_copy_with_segments_is_refused(tmp_path):
    """A repeated log_seq the segments file marks 'ok' (not duplicate) is refused (L92), as a marked duplicate whose
    content differs from the earlier line."""
    p = make_log(tmp_path / 'esp32.csv', 'O1-F05')
    rows = list(csv.DictReader(ln for ln in open(p, encoding='utf-8') if not ln.startswith('#')))
    for status, change in (('ok', {}), ('duplicate', dict(vx='9.9'))):
        out_rows = rows[:300] + [dict(rows[299], **change)] + rows[300:]
        st = ['ok'] * 300 + [status] + ['ok'] * (len(out_rows) - 301)
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLS)
            w.writeheader()
            w.writerows(out_rows)
        with open(p + '.segments.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['row', 'log_seq', 'timestamp_us', 'segment', 'status'])
            for i, (r, st_i) in enumerate(zip(out_rows, st), start=1):
                w.writerow([i, r['log_seq'], r['timestamp_us'], 0, st_i])
        with pytest.raises(ValueError):
            cs.read_esp32(p)


# ---------------------------------------------------------------- Codex vong 3 (17/11, CS3-1..CS3-6), each failing on
# hil 4a5f9ee; review 03_CODE/docs/review/2026-11-17_Tham_dinh_c7_score_o1_vong3.md
K1S = (0.0, 0.0, 0.0)


def test_cs3_1_log_missing_after_t1_keeps_the_certain_violation_before(tmp_path):
    """Codex CS3-1: K1 (T_cmd 5 s, deadline 7 s), the car 0.13 m off the lane all along (G2 lo 0.125 m > D, never
    done), a camera gap [5.2, 5.5] s (t1 = 5.2), the log up to 6 s. The old scorer asked for ticks up to the deadline
    first: U_t infinite, hong_he_do. U_t over [0, t1] bounds the frames before t1: khong_dat (R2V-4). Without a
    violation the run is still lost to the measurement system."""
    kw = dict(k_start=K1S, allow_final_set=True, log_kw=dict(post=1.0))
    fr = make_frames('O1-K1', k_start=K1S, lateral=0.13, drop=[(5.2 + 0.5 / FPS, 5.5)])
    out, _ = run(tmp_path, 'O1-K1', frames=fr, **kw)
    gate_ok(out)
    assert out['outcome'] == 'khong_dat' and out['fail_cause'] == cs.FAIL_VIOL and out['cat_t1']
    assert out['U_t_scope'] == 'truoc_t1' and out['U_t_parts']['tick_to_s'] == pytest.approx(out['t1'])
    assert out['U_t_parts']['drift_s'] == pytest.approx(200e-6 * out['t_deadline'])     # drift still to the deadline
    fr = make_frames('O1-K1', k_start=K1S, drop=[(5.2 + 0.5 / FPS, 5.5)])
    out, _ = run(tmp_path, 'O1-K1', frames=fr, **kw)
    assert out['outcome'] == 'hong_he_do' and 'no certain violation before t1' in out['outcome_reasons'][0]


def test_cs3_2_suspect_outside_the_window_does_not_split_g1(tmp_path):
    """Codex CS3-2: K1, after 5 s the car slides 0.02 m/s until 5.91 s; one frame 30 mm off at 7.016667 s, AFTER the
    deadline (7 s): the only suspect, none in the run window. The old way 1 kept the jump in its derivative (0.059 m/s
    at 6.92 s, not done) while way 2 dropped it (done): G1 chua_ket_luan with no suspect in the run. Frames outside
    the window are now context common to both ways."""
    k = int(round((4.0 + 7.016667) * FPS))
    fr = make_frames('O1-K1', k_start=K1S, jump={k: (0.0, 0.03)},
                     lat_fn=lambda t: 0.02 * (min(t, 5.91) - 5.0) if t > 5.0 else 0.0)
    out, _ = run(tmp_path, 'O1-K1', frames=fr, k_start=K1S, allow_final_set=True)
    g = gate_ok(out)
    assert out['n_suspect_pos'] == 0
    assert g['g1']['t_done'][0] == g['g1']['t_done'][1] and out['t_end_common'] == out['t_end_union']


def test_cs3_4_turn_counts_pairs_by_both_time_bases():
    """Codex CS3-4: receive times 0.1 s after the sample times (tau_hat 0.1 s): the yaw step between the samples at
    -0.075 and -0.025 s has a receive span meeting [0, 0.12]; turn_totals of C8 counts it (0.1 rad > 2 deg), the old
    cut from the last sample at or before t = 0 did not."""
    S = _dyn_S([-0.125, -0.075, -0.025, 0.025, 0.075, 0.125], [0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
    S['t_recv'] = S['t_all'] + 0.1
    big = dict(goc_quay_deg=2.0, wz_dps=1e4, alpha_dps2=1e9)
    assert sum(cs.cu.turn_totals(S, 0.0, 0.12)) == pytest.approx(0.1)
    r = cs.dyn_exceed(S, 0.0, 0.12, big)
    assert r[1] == 'goc_quay_deg' and r[0] is not None


def test_cs3_5_gyro_without_a_report_before_t0_or_with_a_gap():
    """Codex CS3-5: gyro reports from 0.01 s only, a 57 deg/s peak at 0.02 s: reported (the old code skipped every
    gyro check without a report at or before t = 0). A 0.5 s gyro gap (h_lim 0.105 s) with flat values: no excess
    seen, but the note says the gap; a value not finite is skipped and said."""
    lim = dict(goc_quay_deg=1e5, wz_dps=30.0, alpha_dps2=1e9)
    S = _dyn_S(np.arange(-0.1, 0.2, 0.05), np.zeros(6), [0.01, 0.02, 0.03, 0.04], [0.0, 1.0, 0.0, 0.0])
    r = cs.dyn_exceed(S, 0.0, 0.04, lim)
    assert r[1] == 'wz_dps' and r[0] == pytest.approx(0.02) and 'start' in r[2]
    tg = np.r_[np.arange(-0.1, 0.1, 0.01), np.arange(0.6, 0.8, 0.01)]
    S = _dyn_S(np.arange(-0.1, 0.8, 0.05), np.zeros(18), tg, np.zeros(len(tg)))
    r = cs.dyn_exceed(S, 0.0, 0.75, lim)
    assert r[0] is None and 'h_lim' in r[2]
    gz = np.zeros(len(tg))
    gz[15] = np.nan                                         # t = 0.05 s, inside the window
    S = _dyn_S(np.arange(-0.1, 0.8, 0.05), np.zeros(18), tg, gz)
    assert 'not finite' in cs.dyn_exceed(S, 0.0, 0.75, lim)[2]


def test_cs3_6_unclassified_frame_of_a_cycle_has_no_bound():
    """Codex CS3-6: x = [0.0432, 0.0080, 0.0439, 0.0226, 0.0167, 0.0536, 0.0575] does not converge; the set {1, 2, 5}
    holds frame 2 whose final prediction is 14.9 mm, below 20 mm: unclassified (Liam 17/11: 'Coi la chua ket luan'),
    so no prediction (NaN, U_p,k not finite) instead of a finite bound from a flag the final prediction does not
    support."""
    x = np.array([0.0432, 0.0080, 0.0439, 0.0226, 0.0167, 0.0536, 0.0575])
    t = np.arange(7) / 60.0
    sus, pred, _, conv = cs.mark_suspects(t, x, np.zeros(7), 0.02, 2, 3)
    assert not conv and np.nonzero(sus)[0].tolist() == [1, 2, 5]
    assert math.isnan(pred[2]) and pred[5] > 0.02
    assert _ref_pred(t, x, np.zeros(7), sus, 2, 3)[2] == pytest.approx(0.014916667, abs=1e-6)


# ---------------------------------------------------------------- test gaps Codex listed in round 3
def test_cs2_1_offset_error_must_be_finite_and_not_negative(tmp_path):
    for bad in (-0.001, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='offset-err-s'):
            run(tmp_path, offset_err_s=bad)
    with pytest.raises(ValueError, match='offset-anchor-s'):
        run(tmp_path, offset_anchor_s=float('nan'))


def test_cs2_4_one_old_row_inside_a_new_file_is_refused(tmp_path):
    fr = make_frames('O1-F05')
    fr[200]['shift_mag_px'] = float('nan')                  # one row of the old version, shift_px kept
    with pytest.raises(ValueError, match='CS2-4'):
        run(tmp_path, frames=fr)


def test_cs2_8_undefined_suspect_does_not_hide_a_certain_violation_before_t1(tmp_path, monkeypatch):
    """A suspect frame at 3 s (stub; assumption not checked: hi infinite) and a certain violation on the frames just
    before a long gap at 8 s: the run is still khong_dat by R2V-4."""
    suspects_stub(monkeypatch, lambda tk: abs(tk - 3.0) < 0.5 / FPS)
    fr = make_frames('O1-F05', drop=[(8.0, 8.3)], lat_fn=lambda t: 0.13 if 7.9 <= t < 8.0 else 0.0)
    out, rows = run(tmp_path, frames=fr)
    gate_ok(out)
    r = rows[frame_at(rows, 3.0)]
    assert r['suspect'] == 1 and math.isinf(r['hi_g2'])
    assert out['outcome'] == 'khong_dat' and out['cat_t1']


# ---------------------------------------------------------------- Codex vong 4 (17/11, CS4-1, CS4-3..CS4-6), each failing
# on hil cc375d4; review 03_CODE/docs/review/2026-11-17_Tham_dinh_c7_score_o1_vong4.md
def test_cs4_1_rows_after_t1_carry_no_bound_when_u_t_holds_before_t1(tmp_path):
    """Codex CS4-1: the CS3-1 run (U_t over [0, t1] only): the frames after t1 were written with finite bounds from a
    U_t that holds before t1 only (G3 [0.117, 0.143] m at 6.5 s). They now carry no bound, flagged mien_U_t = 0."""
    fr = make_frames('O1-K1', k_start=K1S, lateral=0.13, drop=[(5.2 + 0.5 / FPS, 5.5)])
    out, rows = run(tmp_path, 'O1-K1', frames=fr, k_start=K1S, allow_final_set=True, log_kw=dict(post=1.0))
    assert out['outcome'] == 'khong_dat' and out['U_t_scope'] == 'truoc_t1'
    after = [r for r in rows if r['t'] > out['t1'] + 1e-9]
    before = [r for r in rows if r['t'] <= out['t1'] + 1e-9]
    assert after and all(r['mien_U_t'] == 0 and math.isnan(r['hi_g3']) and math.isnan(r['lo_g2']) for r in after)
    assert all(r['mien_U_t'] == 1 and math.isfinite(r['hi_g3']) for r in before)


def test_cs4_3_no_turn_counted_after_t_break():
    """Codex CS4-3: a reset at 0.075 s (t_break 0.05 s) and a 1 rad step after it: the old span widening of CS3-4 took
    the pair [0.05, 0.10] and reported 2 deg exceeded at 0.05 s; the yaw up to t_break turned 0 rad."""
    S = _dyn_S([-0.05, 0.0, 0.05, 0.10, 0.15], [0, 0, 0, 1, 1], [-0.05, 0.0, 0.05, 0.10, 0.15], [0.0] * 5)
    S['t_break'] = 0.05
    r = cs.dyn_exceed(S, 0.0, 0.15, dict(goc_quay_deg=2.0, wz_dps=1e4, alpha_dps2=1e9))
    assert r[0] is None and 't_break' in r[2]


def test_cs4_4_acceleration_between_finite_reports_across_a_nan():
    """Codex CS4-4: gz 0, NaN, 1 rad/s at 0, 0.05, 0.10 s: the two finite reports give 573 deg/s^2 > 500: reported
    (the old code paired the NaN with each neighbour and saw nothing)."""
    S = _dyn_S([-0.05, 0.0, 0.05, 0.10, 0.15], [0.0] * 5, [0.0, 0.05, 0.10], [0.0, float('nan'), 1.0])
    r = cs.dyn_exceed(S, 0.0, 0.10, dict(goc_quay_deg=1e5, wz_dps=1e4, alpha_dps2=500.0))
    assert r[1] == 'alpha_dps2' and r[0] == pytest.approx(0.10) and 'not finite' in r[2]


def test_cs4_5_gyro_only_after_the_window_is_no_excess():
    """Codex CS4-5: window [0, 1], gyro reports at 2 and 3 s only (1 rad/s at 2 s): the old code reported an excess
    at 2 s; it is outside the run: no excess, the coverage is noted."""
    S = _dyn_S(np.arange(-0.05, 1.1, 0.05), np.zeros(23), [2.0, 3.0], [1.0, 0.0])
    r = cs.dyn_exceed(S, 0.0, 1.0, dict(goc_quay_deg=1e5, wz_dps=30.0, alpha_dps2=1e9))
    assert r[0] is None and 'no gyro report inside the window' in r[2]


def test_cs4_6_last_yaw_report_on_the_window_end_is_coverage():
    S = _dyn_S([-0.05, 0.0, 0.05, 0.10], [0.0] * 4, [-0.05, 0.0, 0.05, 0.10], [0.0] * 4)
    r = cs.dyn_exceed(S, 0.0, 0.10, dict(goc_quay_deg=2.0, wz_dps=1e4, alpha_dps2=1e9))
    assert r == (None, '', '')


def test_cs3_5_acceleration_across_a_gyro_gap_is_an_excess():
    """The author's choice checked by Codex round 4: across a gap above h_lim the difference quotient still bounds the
    true peak from below (1 rad/s in 0.5 s = 115 deg/s^2 > 100): reported, and the gap is noted."""
    S = _dyn_S(np.arange(-0.05, 0.6, 0.05), np.zeros(13), [0.0, 0.5], [0.0, 1.0])
    r = cs.dyn_exceed(S, 0.0, 0.5, dict(goc_quay_deg=1e5, wz_dps=1e4, alpha_dps2=100.0))
    assert r[1] == 'alpha_dps2' and r[0] == pytest.approx(0.5) and 'h_lim' in r[2]


def test_cs3_2_suspects_inside_and_outside_the_window(tmp_path):
    """A suspect inside the window (2.5 s) and one after the deadline (7.0167 s): way 2 drops the one inside only, so
    near the window end (no suspect inside around it) both ways have the same speeds (the frame after the deadline is
    context of both)."""
    lane = lane_of('O1-K1', K1S)
    _, _, _, hc = lane.command(2.5)
    jump = {int(round((4.0 + 2.5) * FPS)): (-0.03 * math.sin(hc), 0.03 * math.cos(hc)),
            int(round((4.0 + 7.016667) * FPS)): (0.0, 0.03)}
    fr = make_frames('O1-K1', k_start=K1S, jump=jump, lat_fn=lambda t: 0.02 * (min(t, 5.91) - 5.0) if t > 5.0 else 0.0)
    out, rows = run(tmp_path, 'O1-K1', frames=fr, k_start=K1S, allow_final_set=True)
    assert rows[frame_at(rows, 2.5)]['suspect'] == 1
    end = [r for r in rows if 6.8 <= r['t'] < 7.0]
    assert end and all(r['speed_way2'] == pytest.approx(r['speed'], abs=1e-12) for r in end)


# ---------------------------------------------------------------- Codex vong 5 (17/11, CS5-1, CS5-2), each failing on
# hil 05ccfc4; review 03_CODE/docs/review/2026-11-17_Tham_dinh_c7_score_o1_vong5.md
BIGL = dict(goc_quay_deg=1e5, wz_dps=1e5, alpha_dps2=1e9)


def test_cs5_1_no_acceleration_across_an_imu_reset():
    """Codex CS5-1: gz 0, NaN, 0.01 rad/s at 0, 0.05, 1 s and a reset at 0.075 s: the two finite reports belong to two
    sensor starts, so their difference is no motion (05ccfc4 reported 0.573 deg/s^2 > 0.5 at 1 s); the note says the
    acceleration across the reset was not checked."""
    ta = np.arange(0.0, 1.0001, 0.05)
    S = _dyn_S(ta, np.zeros_like(ta), [0.0, 0.05, 1.0], [0.0, float('nan'), 0.01])
    S['resets'], S['t_break'] = [0.075], 0.05
    r = cs.dyn_exceed(S, 0.0, 1.0, dict(BIGL, alpha_dps2=0.5))
    assert r[0] is None and 'reset' in r[2] and 'acceleration' in r[2]


def test_cs5_1_true_excess_inside_one_segment_is_still_reported():
    """Gyro 0, 0, 0.5, 1.5 rad/s at 0, 0.05, 0.5, 0.55 s, reset at 0.3 s, limit 50 deg/s^2: the pair across the reset
    (0.5 rad/s in 0.45 s, 64 deg/s^2) is not used (05ccfc4: excess at 0.5 s); the pair after it (1 rad/s in 0.05 s,
    1146 deg/s^2) is an excess at 0.55 s."""
    ta = np.arange(0.0, 0.6001, 0.05)
    S = _dyn_S(ta, np.zeros_like(ta), [0.0, 0.05, 0.5, 0.55], [0.0, 0.0, 0.5, 1.5])
    S['resets'], S['t_break'] = [0.3], 0.25
    r = cs.dyn_exceed(S, 0.0, 0.6, dict(BIGL, alpha_dps2=50.0))
    assert r[1] == 'alpha_dps2' and r[0] == pytest.approx(0.55)


def test_cs5_2_edges_of_the_reports_used_are_noted():
    """Codex CS5-2: window [0, 7], gyro every 50 ms from -0.025 s: the reports used (inside the window) run from 0.025
    to 6.975 s, the 25 ms at each edge is not checked; 05ccfc4 gave an empty note (it looked at the whole series)."""
    tg = -0.025 + 0.05 * np.arange(142)
    ta = np.arange(-0.05, 7.05, 0.05)
    r = cs.dyn_exceed(_dyn_S(ta, np.zeros_like(ta), tg, np.zeros_like(tg)), 0.0, 7.0, BIGL)
    assert r[0] is None and 'start 0.025 s after t = 0' in r[2] and 'end 0.025 s before the window end' in r[2]


def test_cs5_2_one_finite_report_inside_says_acceleration_not_checked():
    ta = np.arange(-0.05, 1.0, 0.05)
    r = cs.dyn_exceed(_dyn_S(ta, np.zeros_like(ta), [-0.5, 0.5, 1.5], [0.0, 0.0, 0.0]), 0.0, 0.9, BIGL)
    assert r[0] is None and 'acceleration not checked' in r[2]


def test_cs5_2_all_reports_inside_not_finite_says_rate_not_checked():
    ta = np.arange(-0.05, 1.0, 0.05)
    S = _dyn_S(ta, np.zeros_like(ta), [-0.05, 0.4, 0.8, 0.95], [0.0, float('nan'), float('nan'), 0.0])
    r = cs.dyn_exceed(S, 0.0, 0.9, BIGL)
    assert r[0] is None and 'rate and acceleration not checked' in r[2]


# ---------------------------------------------------------------- test gaps Codex listed in round 5 (guards: pass on
# 05ccfc4, each fails on a narrower fix)
def test_cs4_1_all_eight_bounds_after_t1_and_one_key_set(tmp_path):
    """CS4-1: every lo_/hi_ of g2..g5 is NaN after t1 (not only lo_g2, hi_g3), and every row has the same keys so the
    rows.csv writer (csv.DictWriter on rows[0]) takes them all."""
    fr = make_frames('O1-K1', k_start=K1S, lateral=0.13, drop=[(5.2 + 0.5 / FPS, 5.5)])
    out, rows = run(tmp_path, 'O1-K1', frames=fr, k_start=K1S, allow_final_set=True, log_kw=dict(post=1.0))
    assert out['U_t_scope'] == 'truoc_t1'
    cols = [f'{s}_{c}' for c in cs.o1_gate.MEASURED for s in ('lo', 'hi')]
    after = [r for r in rows if r['t'] > out['t1'] + 1e-9]
    assert len(cols) == 8 and after and all(math.isnan(r[c]) for r in after for c in cols)
    assert all(set(r) == set(rows[0]) for r in rows)


def test_cs4_3_turn_before_t_break_is_still_an_excess():
    """CS4-3 guard: 0.1 rad (5.7 deg) between 0 and 0.05 s, t_break at 0.05 s, then a 1 rad step: the turn before the
    break is an excess (a fix that dropped the whole angle check at t_break would pass the CS4-3 test)."""
    S = _dyn_S([-0.05, 0.0, 0.05, 0.10, 0.15], [0, 0, 0.1, 1.1, 1.1], [-0.05, 0.0, 0.05, 0.10, 0.15], [0.0] * 5)
    S['t_break'] = 0.05
    r = cs.dyn_exceed(S, 0.0, 0.15, dict(BIGL, goc_quay_deg=2.0))
    assert r[1] == 'goc_quay_deg' and r[0] <= 0.05 + 1e-9


@pytest.mark.parametrize('k', [0, 2])
def test_cs4_5_reports_on_t0_and_t_end_count(k):
    """CS4-5 guard: a 1 rad/s report exactly at t0 (k = 0) or at t_end (k = 2) is inside the window."""
    gz = [0.0, 0.0, 0.0]
    gz[k] = 1.0
    ta = np.arange(-0.05, 1.1, 0.05)
    r = cs.dyn_exceed(_dyn_S(ta, np.zeros_like(ta), [0.0, 0.05, 0.10], gz), 0.0, 0.10, dict(BIGL, wz_dps=30.0))
    assert r[1] == 'wz_dps' and r[0] == pytest.approx(0.05 * k)


def test_cs4_6_last_yaw_report_before_the_window_end_is_noted():
    """CS4-6 guard: the last yaw report at 0.05 s, window end 0.10 s: the 0.050 s not covered is noted."""
    S = _dyn_S([-0.05, 0.0, 0.05], [0.0] * 3, [-0.05, 0.0, 0.05, 0.10], [0.0] * 4)
    r = cs.dyn_exceed(S, 0.0, 0.10, dict(BIGL, goc_quay_deg=2.0))
    assert 'IMU yaw reports end 0.050 s before the window end' in r[2]


@pytest.mark.parametrize('t_reset, excess', [(0.05, False), (0.0, True)])
def test_cs5_1_reset_on_a_receive_time_follows_the_yaw_rule(t_reset, excess):
    """As c8_u_imu.rv_samples: a report received at the reset row time is after the reset. Reset at 0.05 s (the second
    report): the pair crosses it, no acceleration; reset at 0.0 s (the first report): both after it, 1146 deg/s^2."""
    ta = np.arange(-0.05, 0.11, 0.05)
    S = _dyn_S(ta, np.zeros_like(ta), [0.0, 0.05], [0.0, 1.0])
    S['resets'] = [t_reset]
    r = cs.dyn_exceed(S, 0.0, 0.05, dict(BIGL, alpha_dps2=500.0))
    assert (r[1] == 'alpha_dps2') == excess


# ---------------------------------------------------------------- O1 v11.33 (17/11): G3, G5 bounds per frame (V_k, W_k)
def _brute_sup(lane, a, b, n=20001):
    tt = np.linspace(a, b, n)
    r = np.array([lane.rate(float(x)) for x in tt])
    return float(r[:, 0].max()), float(r[:, 1].max())


@pytest.mark.parametrize('task', ['O1-F02', 'O1-F05', 'O1-F08'])
def test_v11_33_cmd_rates_bound_the_sup_from_above_and_tightly(task):
    """V_k, W_k: never below the sup of a fine sampling (20001 points) of the exact rates, at most 1 % of the peak above
    it; the peaks are those of o1_gate.fig8_cmd_max (the schedule has tau_dot <= w0)."""
    lane = lane_of(task)
    cr = cs.CmdRates(lane)
    vmax, wzmax = lane.cmd_max
    rng = np.random.default_rng(3)
    for a in list(rng.uniform(-0.5, lane.Tcmd + 0.5, 25)) + [0.0, lane.Tcmd - 0.03]:
        b = a + 0.06
        v, w = cr.sup(a, b)
        bv, bw = _brute_sup(lane, max(a, 0.0), min(b, lane.Tcmd)) if b > 0 and a < lane.Tcmd else (0.0, 0.0)
        assert bv - 1e-12 <= v <= bv + 0.01 * vmax and bw - 1e-12 <= w <= bw + 0.01 * wzmax, (a, v, bv, w, bw)
    v, w = cr.sup(0.0, lane.Tcmd)
    assert v == pytest.approx(vmax, rel=0.01) and w == pytest.approx(wzmax, rel=0.01)


def test_v11_33_rates_are_zero_outside_the_command():
    """OG2-2: the reference holds its end values before t = 0 and after T_cmd: V = W = 0 on an interval wholly outside."""
    for task in ('O1-F05', 'O1-K1'):
        lane = lane_of(task, K1S) if task == 'O1-K1' else lane_of(task)
        cr = cs.CmdRates(lane)
        assert cr.sup(-0.2, -0.01) == (0.0, 0.0) and cr.sup(lane.Tcmd + 0.01, lane.Tcmd + 0.3) == (0.0, 0.0)


def test_v11_33_k_task_has_no_heading_rate():
    """K1: G5 against the start heading (constant): W_k = 0 everywhere; V_k at most the task speed K_V."""
    lane = lane_of('O1-K1', K1S)
    cr = cs.CmdRates(lane)
    v, w = cr.sup(0.0, lane.Tcmd)
    assert w == 0.0 and v <= lane.vmax * 1.01


# ---------------------------------------------------------------- O1 v11.33 SC-12: tick part kept / dropped (OG2-1, OG3-4)
def test_v11_33_tick_kept_by_default_before_c8(tmp_path):
    out, _ = run(tmp_path)
    p = out['U_t_parts']
    assert p['nhanh_tick'] == 'giu' and p['tick_s'] > 0
    assert out['U_t_s'] == pytest.approx(p['ghep_s'] + p['offset_err_s'] + p['drift_s'] + p['tick_s'])


def test_v11_33_drop_tick_needs_the_sample_mark_match(tmp_path, monkeypatch):
    """Branch 'bo' with the clock match of o1_time_sync (no sample marks) or with --offset-s: refused, never a U_t
    without the tick part on a match that does not carry it."""
    with pytest.raises(ValueError, match='nhanh_tick'):
        run(tmp_path, hs_kw=dict(nhanh_tick='bo'))
    lane = lane_of('O1-F05')
    sync, _ = _sync_fake(lane, OFFSET, 0.0)
    monkeypatch.setattr(cs, 'clock_match', lambda frames, rows, task: (sync, None))
    with pytest.raises(ValueError, match='nhanh_tick'):
        run(tmp_path, offset_s=None, offset_reason='', hs_kw=dict(nhanh_tick='bo'))


def test_v11_33_drop_tick_with_the_sample_mark_match(tmp_path):
    """Branch 'bo' at a node of the C8b grid (match on sample marks): U_t without the tick part, the tick still
    measured (coverage of both ends, OG2-8)."""
    out, _ = run_gio(tmp_path, hs_kw=dict(nhanh_tick='bo'))
    p = out['U_t_parts']
    assert p['nhanh_tick'] == 'bo' and math.isfinite(p['tick_s']) and p['tick_s'] > 0
    assert out['U_t_s'] == pytest.approx(p['luoi_s'] + p['drift_s'] + p['man_trap_s'])


def test_v11_33_tick_branch_value_checked(tmp_path):
    with pytest.raises(ValueError, match='nhanh_tick'):
        hs_file(tmp_path, nhanh_tick='x')


# ---------------------------------------------------------------- O1 v11.33 3.5 (OG2-3): report-only conditions and R2V-4
def test_v11_33_report_only_condition_does_not_make_a_violation_before_t1(tmp_path):
    """A certain violation of G3 only (lo 0.12 > D) before t1: the exception R2V-4 sees it; with G3 report-only at the
    level of the run it does not; G4 is never report-only."""
    n = 4
    lo = {c: np.zeros(n) for c in cs.o1_gate.MEASURED}
    lo['g3'][1] = 0.12
    mask = np.ones(n, bool)
    assert cs.certain_violations(lo, mask) == ['g3']
    assert cs.certain_violations(lo, mask, frozenset({'g3'})) == []
    lo['g4'][2] = 0.12
    assert cs.certain_violations(lo, mask, frozenset({'g2', 'g3', 'g5'})) == ['g4']
    with pytest.raises(ValueError, match='report_only'):
        run(tmp_path, report_only=frozenset({'g4'}))


def test_v11_33_window_narrower_than_the_grid_uses_its_ends():
    """A window of 0.4 ms between two grid points (step 1 ms): the bound is at least the rate at its ends."""
    lane = lane_of('O1-F05')
    cr = cs.CmdRates(lane)
    a = 5.0003
    v, w = cr.sup(a, a + 0.0004)
    ra, rb = lane.rate(a), lane.rate(a + 0.0004)
    assert v >= max(ra[0], rb[0]) and w >= max(ra[1], rb[1]) and v > 0.1


# ---------------------------------------------------------------- C8b v4 3 (17/11): U_t at a node (--gio-c8b)
NODES = [OFFSET - 0.002, OFFSET - 0.0015, OFFSET, OFFSET + 0.001]       # gaps 0.5, 1.5, 1.0 ms: step 1.5 ms


def gio_file(tmp_path, **over):
    d = dict(format='c8b_gio_v1', moc_mau=True, ok=True, nodes_s=NODES, cua_so_ghep_log_s=[T0 + 3.0, T0 + 9.0],
             sha256={'esp32': 'e' * 64})
    d.update(over)
    p = tmp_path / 'gio.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    return str(p)


def run_gio(tmp_path, nut=2, gio_kw=None, **kw):
    return run(tmp_path, offset_s=None, offset_reason='', gio=cs.load_gio(gio_file(tmp_path, **(gio_kw or {})), nut),
               **kw)


def test_c8b_node_u_t_table(tmp_path):
    """At node 2 (D = OFFSET): U_t = whole grid step (largest gap, 1.5 ms) + troi_ppm x the farthest distance from the
    match window [3, 9] s (command time) to [0, t_deadline] + rolling shutter (one frame) + tick; u_t_ghep_s and
    --offset-err-s not added; o1_time_sync not run (offset_src c8b_gio)."""
    out, _ = run_gio(tmp_path, hs_kw=dict(troi_ppm=1000.0, u_t_ghep_s=0.05))
    p, dl = out['U_t_parts'], out['t_deadline']
    assert out['offset_src'] == 'c8b_gio' and out['offset_s'] == OFFSET and out['gio_c8b']['nut'] == 2
    assert p['luoi_s'] == pytest.approx(0.0015) and p['ghep_s'] == 0.0 and p['offset_err_s'] == 0.0
    assert p['drift_s'] == pytest.approx(1e-3 * max(9.0, dl - 3.0))
    assert p['man_trap_s'] == pytest.approx(1.0 / FPS)
    assert out['U_t_s'] == pytest.approx(0.0015 + p['drift_s'] + p['man_trap_s'] + p['tick_s'])


def test_c8b_node_scores_with_the_node_offset(tmp_path):
    out, _ = run_gio(tmp_path, nut=0)
    assert out['offset_s'] == NODES[0]


@pytest.mark.parametrize('bad', [dict(moc_mau=False), dict(ok=False), dict(nodes_s=[]),
                                 dict(nodes_s=[1.0, 1.0]), dict(cua_so_ghep_log_s=[2.0, 1.0]),
                                 dict(cua_so_ghep_log_s=[1.0]), dict(format='x')])
def test_c8b_bad_grid_file_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        cs.load_gio(gio_file(tmp_path, **bad), 0)


def test_c8b_node_index_and_one_clock_source(tmp_path):
    with pytest.raises(ValueError, match='nut'):
        cs.load_gio(gio_file(tmp_path), 4)
    with pytest.raises(ValueError, match='one clock source'):
        run(tmp_path, gio=cs.load_gio(gio_file(tmp_path), 0))         # run() gives --offset-s by default


def test_c8b_one_node_has_no_grid_step(tmp_path):
    out, _ = run_gio(tmp_path, nut=0, gio_kw=dict(nodes_s=[OFFSET]))
    assert out['U_t_parts']['luoi_s'] == 0.0


def test_c8b_shutter_by_row_and_column_spans(tmp_path):
    """man_trap alpha_r 8e-6 s/px, alpha_c -2e-6 s/px (absolute value, C8B4R4-1): the term is |a_r| x row span +
    |a_c| x column span of the car tag over the frames seen in [min(window, 0), max(window, t_deadline)]."""
    fr = make_frames('O1-F05')
    for k, r in enumerate(fr):
        r['row_px'], r['col_px'] = 1000.0 + (k % 7), 2000.0 + 3 * (k % 5)
    mt = dict(alpha_r_s_px=8e-6, alpha_c_s_px=-2e-6, nguon='test')
    out, _ = run_gio(tmp_path, frames=fr, hs_kw=dict(man_trap=mt))
    assert out['U_t_parts']['man_trap_s'] == pytest.approx(8e-6 * 6 + 2e-6 * 12)
    for r in fr:
        r['col_px'] = float('nan')
    with pytest.raises(ValueError, match='col_px'):
        run_gio(tmp_path, frames=fr, hs_kw=dict(man_trap=mt))
    with pytest.raises(ValueError, match='man_trap'):
        hs_file(tmp_path, man_trap=dict(alpha_r_s_px=8e-6, alpha_c_s_px=0.0))          # no nguon


def test_c8b_one_frame_is_the_median_frame_interval(tmp_path):
    """Every third frame 3 ms late: intervals 13.7 / 16.7 / 19.7 ms; one frame is the median, 16.7 ms."""
    fr = make_frames('O1-F05')
    for k, r in enumerate(fr):
        if k % 3 == 0:
            r['t_s'] += 0.003
    out, _ = run_gio(tmp_path, frames=fr)
    assert out['U_t_parts']['man_trap_s'] == pytest.approx(1.0 / FPS, abs=1e-6)


def test_c8b_rows_carry_camera_time_and_domains(tmp_path):
    """--rows for the grid merge: t_cam_s is the frame's own t_s (the join key across nodes), mien_chung <= mien_hop,
    and the union-domain frames are the frames the gate counted."""
    fr = make_frames('O1-F05')
    out, rows = run_gio(tmp_path, frames=fr)
    ts = {r['t_s'] for r in fr}
    assert rows and all(r['t_cam_s'] in ts and 'iv' not in r for r in rows)
    assert all(r['t_cam_s'] + OFFSET - T0 == pytest.approx(r['t'], abs=1e-9) for r in rows)
    assert all(r['mien_chung'] <= r['mien_hop'] for r in rows)
    assert sum(r['mien_hop'] for r in rows) == out['gate_v3']['g2']['n_frames']


def test_c8b_rows_common_and_union_domains_differ(tmp_path, monkeypatch):
    """As test_two_ways_of_using_suspects (common domain to way 1's t_done, union to the deadline): mien_chung marks
    the scored frames up to t_end_common, mien_hop those up to t_end_union, so some frames are in the union only."""
    lane = lane_of('O1-F05')
    suspects_stub(monkeypatch, lambda tk: tk > lane.Tcmd - 0.5 and ((tk - lane.Tcmd) % 0.4) < 8.5 / FPS)
    out, rows = run_gio(tmp_path, frames=make_frames('O1-F05'), hs_kw=dict(pred_half=30, gia_dinh_nghi_vi_tri='dung'))
    assert out['t_end_common'] < out['t_end_union']
    assert all(r['mien_chung'] == int(bool(r['scored']) and r['t'] <= out['t_end_common'] + 1e-12) for r in rows)
    assert any(r['mien_hop'] and not r['mien_chung'] for r in rows)


# ---------------------------------------------------------------- test gaps Codex listed in round 6 (guards)
def test_cs5_1_rate_still_checked_at_a_report_of_a_pair_across_a_reset():
    """Only the acceleration across a reset is dropped: a rate excess at a report of that pair is still reported."""
    ta = np.arange(0.0, 1.0001, 0.05)
    S = _dyn_S(ta, np.zeros_like(ta), [0.0, 1.0], [0.0, 1.0])
    S['resets'], S['t_break'] = [0.5], 0.45
    r = cs.dyn_exceed(S, 0.0, 1.0, dict(BIGL, wz_dps=30.0, alpha_dps2=0.5))
    assert r[1] == 'wz_dps' and r[0] == pytest.approx(1.0) and 'reset' in r[2]


def test_cs5_2_edges_from_the_finite_reports_not_all_reports_inside():
    """NaN at the first and last report inside the window, finite in the middle: the edges noted are those of the
    finite reports (tw would give none)."""
    ta = np.arange(-0.05, 1.05, 0.05)
    tg = [0.0, 0.2, 0.4, 0.6, 1.0]
    S = _dyn_S(ta, np.zeros_like(ta), tg, [float('nan'), 0.0, 0.0, 0.0, float('nan')])
    r = cs.dyn_exceed(S, 0.0, 1.0, BIGL)
    assert 'used start 0.200 s after t = 0' in r[2] and 'used end 0.400 s before the window end' in r[2]


def test_cs5_2_one_finite_report_still_checks_the_rate():
    ta = np.arange(-0.05, 1.0, 0.05)
    r = cs.dyn_exceed(_dyn_S(ta, np.zeros_like(ta), [-0.5, 0.5, 1.5], [0.0, 1.0, 0.0]), 0.0, 0.9, dict(BIGL, wz_dps=30.0))
    assert r[1] == 'wz_dps' and r[0] == pytest.approx(0.5) and 'acceleration not checked' in r[2]


def test_cc1_3_cli_records_the_events_file_it_read(tmp_path):
    """Codex CC1-3: the events file actually read (default <log>.events.txt) by name and sha256; none: null."""
    import c7_measure as cm
    cm.write_frames(tmp_path / 'frames.csv', make_frames('O1-F05'))
    log = make_log(tmp_path / 'esp32.csv', 'O1-F05')
    hs_file(tmp_path)
    base = ['--frames', str(tmp_path / 'frames.csv'), '--esp32', log, '--task', 'O1-F05', '--hang-so',
            str(tmp_path / 'hang_so.json'), '--offset-s', str(OFFSET), '--offset-reason', 'test',
            '--offset-anchor-s', '0', '--offset-err-s', '0']
    assert cs.main(base + ['--out', str(tmp_path / 's1.json')]) == 0
    s = json.loads((tmp_path / 's1.json').read_text(encoding='utf-8'))
    assert s['events_file'] is None and s['events_sha256'] is None
    ev = tmp_path / 'esp32.csv.events.txt'
    ev.write_text('2026-11-17T10:00:00 #us 1\n', encoding='utf-8')
    assert cs.main(base + ['--out', str(tmp_path / 's2.json')]) == 0
    s = json.loads((tmp_path / 's2.json').read_text(encoding='utf-8'))
    assert s['events_file'] == 'esp32.csv.events.txt' and s['events_sha256'] == cm.sha256(str(ev))



def test_cc2_t4_explicit_events_wins_over_the_default(tmp_path):
    """Codex round 2 T4: --events given and the default <log>.events.txt both exist: the given one is read and hashed."""
    import c7_measure as cm
    cm.write_frames(tmp_path / 'frames.csv', make_frames('O1-F05'))
    log = make_log(tmp_path / 'esp32.csv', 'O1-F05')
    hs_file(tmp_path)
    (tmp_path / 'esp32.csv.events.txt').write_text('2026-11-17T10:00:00 #us 1\n', encoding='utf-8')
    ev = tmp_path / 'other_events.txt'
    ev.write_text('2026-11-17T10:00:00 #us 2\n', encoding='utf-8')
    args = ['--frames', str(tmp_path / 'frames.csv'), '--esp32', log, '--task', 'O1-F05', '--hang-so',
            str(tmp_path / 'hang_so.json'), '--offset-s', str(OFFSET), '--offset-reason', 'test',
            '--offset-anchor-s', '0', '--offset-err-s', '0', '--events', str(ev), '--out', str(tmp_path / 's.json')]
    assert cs.main(args) == 0
    s = json.loads((tmp_path / 's.json').read_text(encoding='utf-8'))
    assert s['events_file'] == 'other_events.txt' and s['events_sha256'] == cm.sha256(str(ev))
