"""Tests for tools/c7_score.py on SYNTHETIC pose sequences (built here from the command; NOT a
measurement). Frames are what c7_measure.read_frames returns; the ESP32 log is a list of dicts."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_score as cs  # noqa: E402

FPS = 60.0
T0 = 5.0          # s, command start on the log clock
OFFSET = 0.25     # s, t_log = t_cam + OFFSET


def lane_of(task, k_start=None):
    return cs.Fig8Task(task) if task.startswith('O1-F') else cs.KTask(task, k_start)


def make_frames(task, lateral=0.0, k_start=(0.0, 0.0, 0.0), pre=1.0, post=3.0, drop=(), shift=None,
                back=None, lag=0.0):
    """Car exactly on the command (shifted by `lateral` along the lane normal, delayed by `lag` s),
    still at the end point after T_cmd. drop: (t0, t1) command-time spans with no valid frame.
    back: (t_a, t_b, b) the car runs on the lane with a time warp t - b sin(pi (t - t_a) / (t_b - t_a)):
    with b pi / (t_b - t_a) > 1 it goes backward along the lane for a while (continuous frames)."""
    lane = lane_of(task, k_start)
    t = np.arange(-pre, lane.Tcmd + post, 1.0 / FPS)
    xs, ys, ps = [], [], []
    for tk in t:
        tw = tk - lag
        if back and back[0] <= tk <= back[1]:
            tw -= back[2] * math.sin(math.pi * (tk - back[0]) / (back[1] - back[0]))
        _, px, py, hc = lane.command(max(0.0, tw))
        xs.append(px - lateral * math.sin(hc))
        ys.append(py + lateral * math.cos(hc))
        ps.append(hc if task.startswith('O1-F') else 0.0)
    xs, ys, ps = np.array(xs), np.array(ys), np.array(ps)
    if not task.startswith('O1-F'):                            # lane frame -> floor frame
        x0, y0, h0 = k_start
        c, s = math.cos(h0), math.sin(h0)
        xs, ys = x0 + c * xs - s * ys, y0 + s * xs + c * ys
        ps = ps + h0
    vx, vy = np.gradient(xs, t), np.gradient(ys, t)
    wz = np.gradient(np.unwrap(ps), t)
    rows = []
    for k, tk in enumerate(t):
        ok = not any(a <= tk < b for a, b in drop)
        rows.append(dict(idx=k, t_s=tk + T0 - OFFSET, valid=ok, why=[] if ok else ['khong_thay_xe'], n_marks=24,
                         rms_px=0.3, shift_px=(shift.get(k, 0.2) if shift else 0.2), cam_z=2.6, car_seen=ok,
                         x=xs[k], y=ys[k], psi=cs.wrap(ps[k]), tag_x=xs[k], tag_y=ys[k], res_m=0.0002, scale=1.0,
                         pred_m=0.0, row_px=500.0, vx=vx[k], vy=vy[k], wz=wz[k], speed=math.hypot(vx[k], vy[k])))
    return rows


def make_log(task, k_start=(0.0, 0.0, 0.0), edges=1, pre=2.0, post=4.0):
    lane = lane_of(task, k_start)
    rows, seq = [], 0
    for tk in np.arange(-pre, lane.Tcmd + post, 0.01):
        _, px, py, hc = lane.command(max(0.0, tk))
        s1, *_ = lane.command(max(0.0, tk + 0.005))
        s0, *_ = lane.command(max(0.0, tk - 0.005))
        ra = 1 if tk >= 0 else 0
        if edges == 2 and lane.Tcmd / 2 <= tk < lane.Tcmd / 2 + 0.05:
            ra = 0
        rows.append(dict(timestamp_us=str(int(round((tk + T0) * 1e6))), run_active=str(ra), vx=f'{(s1 - s0) / 0.01:.5f}',
                         vy='0', log_seq=str(seq), pos_reset_n='1'))
        seq += 1
    return rows


def test_on_lane_passes():
    out, rows = cs.score(make_frames('O1-F05'), make_log('O1-F05'), 'O1-F05', 0.005, offset_s=OFFSET)
    assert out['completed'] == 'true'
    assert out['e_perp_max'] < 1e-3
    assert out['outcome'] == 'dat' and out['track_err_frames'] == 0 and out['n_back_frames'] == 0
    assert abs(out['t_end'] - (out['T_cmd'] + 1.0)) < 0.05           # still for t_d after T_cmd
    assert out['placement_pos_mm'] < 1.0 and abs(out['placement_heading_deg']) < 0.1
    for k in ('e_theta_lane_max_deg', 'e_theta_cmd_max_deg', 'e_s_absmax', 'e_trk_max'):   # s05_final_table keys
        assert k in out


@pytest.mark.parametrize('lat,U,want', [(0.12, 0.005, 'khong_dat'), (0.098, 0.005, 'chua_ket_luan'),
                                        (0.09, 0.005, 'dat')])
def test_lateral_offset_against_D_with_U(lat, U, want):
    # the lateral offset stays to the end, so the end point is lat from the origin: completion needs lat <= D
    out, _ = cs.score(make_frames('O1-F02', lateral=lat), make_log('O1-F02'), 'O1-F02', U, offset_s=OFFSET)
    assert abs(out['e_perp_max'] - lat) < 2e-3
    assert out['outcome'] == want, out['outcome_reasons']


def test_gap_above_t_gap_spoils_the_run():
    out, _ = cs.score(make_frames('O1-F05', drop=[(4.0, 4.2)]), make_log('O1-F05'), 'O1-F05', 0.005, offset_s=OFFSET)
    assert out['outcome'] == 'hong_he_do' and 'khoang trong' in out['outcome_reasons'][0]
    assert out['gap_max'] > 0.19


def test_short_gap_is_reported_not_spoiling():
    out, _ = cs.score(make_frames('O1-F05', drop=[(4.0, 4.06)]), make_log('O1-F05'), 'O1-F05', 0.005, offset_s=OFFSET)
    assert out['outcome'] == 'dat'
    assert out['n_gaps_over_frame'] == 1 and 0.06 < out['gaps_total_s'] < 0.09


def test_gap_before_first_window_frame_counts():
    out, _ = cs.score(make_frames('O1-F05', drop=[(-0.1, 0.1)]), make_log('O1-F05'), 'O1-F05', 0.005, offset_s=OFFSET)
    assert out['outcome'] == 'hong_he_do'


def test_camera_bumped_spoils_the_run():
    fr = make_frames('O1-F05', shift={300: 15.0})
    out, _ = cs.score(fr, make_log('O1-F05'), 'O1-F05', 0.005, offset_s=OFFSET)
    assert out['bumped'] and out['outcome'] == 'hong_he_do'


def test_long_gap_is_tracking_error():
    out, _ = cs.score(make_frames('O1-F02', drop=[(5.0, 5.8)]), make_log('O1-F02'), 'O1-F02', 0.005, offset_s=OFFSET)
    assert out['n_miss_frames'] == 1 and out['track_err_frames'] >= 1 and out['outcome'] == 'hong_he_do'


def test_backward_on_lane_is_a_flag_not_an_error():
    out, _ = cs.score(make_frames('O1-F02', back=(10.0, 11.0, 0.8)), make_log('O1-F02'), 'O1-F02', 0.005,
                      offset_s=OFFSET)
    assert out['n_back_frames'] > 0 and out['track_err_frames'] == 0
    assert out['outcome'] == 'dat'


def test_upper_end_reopens_the_window():
    # the car is 0.3 s ahead of the command on frames 5..9 s: some frames jump ahead along the lane
    lane = lane_of('O1-F08')
    t = np.arange(-1.0, lane.Tcmd + 3.0, 1.0 / FPS)
    x, y, psi, sp, wz = [], [], [], [], []
    for tk in t:
        lead = 0.3 if 5.0 <= tk < 5.5 else 0.0
        _, px, py, hc = lane.command(max(0.0, tk + lead))
        x.append(px); y.append(py); psi.append(hc); sp.append(0.0 if tk > lane.Tcmd else 1.0); wz.append(0.0)
    out, rows, _ = cs.evaluate_camera(t, x, y, psi, sp, wz, lane)
    assert out['n_reopen_frames'] >= 1 and out['n_upper_frames'] == 0 and out['track_err_frames'] == 0


def test_k1_on_a_rotated_start_pose():
    ks = (0.3, -0.4, math.radians(30.0))
    out, _ = cs.score(make_frames('O1-K1', k_start=ks), make_log('O1-K1', ks), 'O1-K1', 0.005, offset_s=OFFSET,
                      k_start=ks, allow_final_set=True)
    assert out['completed'] == 'true' and out['e_perp_max'] < 1e-3 and out['outcome'] == 'dat'
    assert out['e_theta_start_max_deg'] < 0.01 and 'e_theta_lane_max_deg' not in out


def test_k2_needs_flag_and_start():
    with pytest.raises(ValueError):
        cs.score(make_frames('O1-K2'), make_log('O1-K2'), 'O1-K2', 0.005, offset_s=OFFSET)
    with pytest.raises(ValueError):
        cs.score(make_frames('O1-K2'), make_log('O1-K2'), 'O1-K2', 0.005, offset_s=OFFSET, allow_final_set=True)


def test_two_run_edges_refused():
    with pytest.raises(ValueError, match='rising edges'):
        cs.score(make_frames('O1-F05'), make_log('O1-F05', edges=2), 'O1-F05', 0.005, offset_s=OFFSET)


def test_clock_from_speed_matching():
    out, _ = cs.score(make_frames('O1-F05'), make_log('O1-F05'), 'O1-F05', 0.005)
    assert out['offset_src'] == 'o1_time_sync'
    assert abs(out['offset_s'] - OFFSET) < 0.01
    assert out['outcome'] == 'dat'


def test_late_car_misses_completion():
    # 2.5 s behind the command: cannot be still at the end before T_cmd + 2 s
    out, _ = cs.score(make_frames('O1-F05', lag=2.5, post=6.0), make_log('O1-F05', post=7.0), 'O1-F05', 0.005,
                      offset_s=OFFSET)
    assert out['completed'] == 'false' and out['outcome'] == 'khong_dat'


def test_cli_writes_score_json(tmp_path):
    import c7_measure as cm
    fr = make_frames('O1-F05')
    cm.write_frames(tmp_path / 'frames.csv', fr)
    log = make_log('O1-F05')
    with open(tmp_path / 'esp32.csv', 'w', newline='', encoding='utf-8') as f:
        f.write('# header comment\n' + ','.join(log[0]) + '\n')
        for r in log:
            f.write(','.join(r.values()) + '\n')
    rc = cs.main(['--frames', str(tmp_path / 'frames.csv'), '--esp32', str(tmp_path / 'esp32.csv'), '--task', 'O1-F05',
                  '--u-m', '0.005', '--u-src', 'test', '--offset-s', str(OFFSET), '--out', str(tmp_path / 'score.json')])
    assert rc == 0
    s = json.loads((tmp_path / 'score.json').read_text(encoding='utf-8'))
    assert s['outcome'] == 'dat' and s['U_src'] == 'test' and len(s['frames_sha256']) == 64
