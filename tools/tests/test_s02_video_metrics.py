"""tools/s02_video_metrics.py and the car's family R in s02_metrics (Viec CODE 5, design section 34; Liam 17/11
CODE_S02 local_bc2d1c16). Made-up runs with a known truth: a car that follows r(t) through a first-order lag (TAU,
smooth like a real car: a speed step sampled at 10 ms or 33 ms would move its edge by up to half a period), its
true motion integrated on a 1 ms grid, placed at any pose on the floor. The ESP32 log (test_s02_metrics_v3.mk) gets
that motion as its encoder speeds and firmware pose; the camera frames see the same motion on a video clock
t_cam = t_log - D with D known, with lost frames, gaps, noise at rest. The family R the tool and score_run give must
equal the one computed straight from the truth."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import c7_measure  # noqa: E402
import s02_metrics as M  # noqa: E402
import s02_ref as R  # noqa: E402
import s02_video_metrics as V  # noqa: E402
from agv_logger import COLUMNS  # noqa: E402
from test_s02_metrics_v3 import T0, mk  # noqa: E402

T_MAT, T_TRONG = 0.68, 0.10        # s, test values of t_mat_s, t_trong_s (the O1 draft numbers; the tool reads them
                                   # only from the C8 constants file, fix/s02-video-hang-so)
D_TRUE = 1_234.5678                  # s: t_log = t_cam + D (video clock far from the ESP32 clock)
OFF = (1.30, -0.75, 2.40)            # world pose of the test frame origin (x0, y0, heading at the start)
TAU = 0.12                           # s, the car's lag behind r(t)
DT = 0.001


def truth(test, delay_s=0.0, lateral=0.0, off=OFF, prep_s=3.0, tail_s=1.5):
    """The car's true motion on a 1 ms grid t (s from the sender t = 0, T0): body speeds of r(t - delay) through the
    lag, heading and position integrated from rest at the test frame origin (xt, yt, tht), shifted by lateral (m,
    left of the start heading) from t = 0.5 s on; world pose X, Y, PSI with the origin at off; speed, wz."""
    t = np.arange(-prep_s - 2.0, R.t_cmd(test) + tail_s + 2.0, DT)
    r = R.reference(test, t - delay_s)
    a = 1.0 - math.exp(-DT / TAU)
    v = np.zeros_like(r)
    for k in range(1, len(t)):
        v[k] = v[k - 1] + a * (r[k] - v[k - 1])
    psi0 = R.start_heading(test)
    th = psi0 + np.r_[0.0, np.cumsum(0.5 * (v[1:, 2] + v[:-1, 2]) * DT)]
    vxw = np.cos(th) * v[:, 0] - np.sin(th) * v[:, 1]
    vyw = np.sin(th) * v[:, 0] + np.cos(th) * v[:, 1]
    xt = np.r_[0.0, np.cumsum(0.5 * (vxw[1:] + vxw[:-1]) * DT)]
    yt = np.r_[0.0, np.cumsum(0.5 * (vyw[1:] + vyw[:-1]) * DT)]
    lat = np.where(t >= 0.5, lateral, 0.0)
    xt, yt = xt - lat * math.sin(psi0), yt + lat * math.cos(psi0)
    x0, y0, th0 = off
    b = th0 - psi0
    return dict(t=t, xt=xt, yt=yt, tht=th, X=x0 + math.cos(b) * xt - math.sin(b) * yt,
                Y=y0 + math.sin(b) * xt + math.cos(b) * yt, PSI=th + b, vx=v[:, 0], vy=v[:, 1], wz=v[:, 2],
                speed=np.hypot(v[:, 0], v[:, 1]))


def run_of(test, **kw):
    """(log, truth): the log of mk(test) with the true motion as its encoder speeds, gyro and firmware pose."""
    tr = truth(test, **kw)
    log = mk(test)
    t = (log['timestamp_us'] - T0) * 1e-6
    at = lambda a: np.interp(t, tr['t'], a)                          # noqa: E731
    log['vx'], log['vy'] = at(tr['vx']), at(tr['vy'])
    log['wz'] = log['wz_meas'] = log['gz_fused'] = at(tr['wz'])
    log['pos_x'], log['pos_y'], log['pos_th'] = at(tr['xt']), at(tr['yt']), at(tr['tht'])
    return log, tr


def expected(test, log, tr, t0_us):
    """e_p, e_psi (rms, max) straight from the truth on the window lines of score_run (t = 0 at t0_us)."""
    ts = log['timestamp_us']
    tm = (ts - t0_us) * 1e-6
    win = (tm >= M.WINDOW_SKIP_S - M.WINDOW_EPS_S) & (tm <= R.t_cmd(test) + M.WINDOW_EPS_S)
    tt = (ts[win] - T0) * 1e-6
    x, y, th = (np.interp(tt, tr['t'], tr[k]) for k in ('xt', 'yt', 'tht'))
    ref = R.pose_ref(test, tm[win])
    ep = np.hypot(x - ref[:, 0], y - ref[:, 1])
    eps = np.abs(R.wrap(th - ref[:, 2]))
    return dict(e_p_rms=math.sqrt(np.mean(ep ** 2)), e_p_max=ep.max(), e_psi_rms=math.sqrt(np.mean(eps ** 2)),
                e_psi_max=eps.max())


def frames_of(log, tr, D=D_TRUE, fps=30.0, seed=3, lost=None, rest_noise=0.0, t_end=None, speed=None,
              lost_pose=False, rate=0.0):
    """frames.csv rows (c7_measure.read_frames layout): frames every 1/fps (+- 2 ms) from 1 s before the first log
    line to t_end (s from T0; default 1 s after the last line), pose, speed, wz of the truth at the frame's true time;
    lost(t) -> True for an invalid frame (no pose; lost_pose: a wrong but finite pose, 0.5 m off, as a frame refused
    by c7_measure's gates keeps it); rest_noise: +rest_noise m on x over [-0.5, -0.25) s and -rest_noise over
    [-0.25, 0) (mean 0 over the 0.5 s rest only); speed(t) replaces the camera speed; rate: the video clock runs
    fast by this fraction (t_cam = t_log - D + rate * t)."""
    rng = np.random.default_rng(seed)
    ts = log['timestamp_us'] * 1e-6
    t0, t1 = ts[0] - 1.0, (ts[-1] + 1.0 if t_end is None else T0 * 1e-6 + t_end)
    base = np.arange(t0, t1, 1.0 / fps)
    t_log = base + rng.uniform(-0.002, 0.002, len(base))
    t_rel = t_log - T0 * 1e-6
    at = lambda a: np.interp(t_rel, tr['t'], a)                       # noqa: E731
    X, Y, PSI, SP, WZ = at(tr['X']), at(tr['Y']), at(tr['PSI']), at(tr['speed']), at(tr['wz'])
    sgn = np.where(t_rel < -0.25, 1.0, -1.0)
    X = X + np.where((t_rel >= -0.5) & (t_rel < 0.0), rest_noise * sgn, 0.0)
    if speed is not None:
        SP = speed(t_rel)
    rows = []
    nan = float('nan')
    for k in range(len(t_log)):
        bad = bool(lost(t_rel[k])) if lost is not None else False
        no = bad and not lost_pose
        rows.append(dict(idx=k, t_s=t_log[k] - D + rate * t_rel[k], valid=not bad, why=['lost'] if bad else [], n_marks=4,
                         rms_px=0.3, shift_px=0.1, cam_z=2.5, car_seen=not bad,
                         x=nan if no else X[k] + (0.5 if bad else 0.0), y=nan if no else Y[k],
                         psi=nan if no else float(R.wrap(PSI[k])),
                         tag_x=nan, tag_y=nan, res_m=nan, scale=1.0, pred_m=None, row_px=nan,
                         vx=nan, vy=nan, wz=nan if bad else WZ[k], speed=nan if bad else SP[k]))
    return rows


def _write_log(path, log):
    """agv_logger CSV of a made-up log: whole numbers as integers (run_active '1', as the logger writes them)."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(len(log['timestamp_us'])):
            row = []
            for c in COLUMNS:
                x = float(log[c][i]) if c in log else 0.0
                row.append(str(int(x)) if x.is_integer() else f'{x:.9f}')
            w.writerow(row)


def hang_so(tmp_path, name='hang_so.json', **over):
    """A valid C8 constants file (c7_hang_so_v1, the keys tools/c7_score.load_hang_so checks) with t_mat_s, t_trong_s
    = T_MAT, T_TRONG unless overridden; returns its path."""
    d = dict(version='c7_hang_so_v1', trang_thai='de_xuat', nguon='test', U_p_m=0.005, U_psi_deg=2.0,
             u_t_ghep_s=0.005, troi_ppm=200.0, t_trong_s=T_TRONG, t_mat_s=T_MAT, bump_px=12.0, drift_px=1.5,
             pred_m=0.02, pred_half=6, pred_min=4, yaw_nghi_deg=3.0, v_d=0.01,
             imu=dict(period_s=0.05, tau_hat_s=0.0, s_plus=0.0, s_minus=0.0, j_lim_s=0.005, h_lim_s=0.105, pre_s=3.0,
                      wz_still=0.05, n_min=30,
                      U_IMU_deg={'O1-F02': 1.0, 'O1-F05': 1.0, 'O1-F08': 1.0, 'O1-K1': 1.0, 'O1-K2': 1.0}))
    d.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(d), encoding='utf-8')
    return str(path)


def log_rows(log, tmp_path, name='log.csv'):
    p = tmp_path / name
    _write_log(p, log)
    return str(p), V.read_log_rows(str(p))


def _table(rows, cols):
    data = np.array([[float(c) if c != '' else np.nan for c in r] for r in rows]).reshape(len(rows), len(cols))
    t = {c: data[:, i] for i, c in enumerate(cols)}
    t['_row'] = {int(x): i for i, x in enumerate(t['timestamp_us'])}
    return t


def plant_dict(out, cam=None, info=None):
    """What s02_metrics._plant_inputs gives for the tool's rows (cells as written): the plant table with the camera
    frames under '_cam' (and the camera t_miss)."""
    pl = _table(out, V.COLUMNS)
    if cam is not None:
        pl['_cam'] = _table(cam, V.CAM_COLUMNS)
        pl['_cam']['t_miss_s'] = info['t_miss_s'] if info else T_MAT
    return pl


def scored(test, tmp_path, frames_kw=None, drop=None, **kw):
    """(log, truth, info, plant, metrics) of a made-up run through the tool; drop: mask of log lines lost (taken
    out of the log before the tool and the scoring see it)."""
    log, tr = run_of(test, **kw)
    if drop is not None:
        log = drop_lines(log, drop)
    _, rows = log_rows(log, tmp_path)
    info, out, cam = V.map_video(frames_of(log, tr, **(frames_kw or {})), rows, T_MAT, T_TRONG)
    pl = plant_dict(out, cam, info) if out else None
    m = M.score_run(log, test, 'real', plant=pl) if pl is not None else None
    return log, tr, info, pl, m


def drop_lines(log, drop):
    out = {c: x[~drop] for c, x in log.items() if not c.startswith('_')}
    out['_all'] = out['_every'] = out
    return out


@pytest.mark.parametrize('test', ['A-L2', 'A-L1', 'A-C2', 'A-C1', 'A-S2', 'A-F2'])
def test_offset_pose_and_family_r_match_the_truth(tmp_path, test):
    """D found within 2 ms; the pose on every log line is the true pose there (linear between frames); psi
    unwrapped (A-C2, A-C1 turn 2 pi); e_p, e_psi as computed from the truth; e_est near 0 (firmware pose = truth)."""
    log, tr, info, pl, m = scored(test, tmp_path)
    assert info['ok'], info['reason']
    assert info['offset_s'] == pytest.approx(D_TRUE, abs=0.002)
    w0, w1 = info['sync']['window_cam_s']
    assert info['sync']['short'] == (w1 - w0 < 9.0)
    t_rel = (pl['timestamp_us'] - T0) * 1e-6
    X, Y, PSI = (np.interp(t_rel, tr['t'], tr[k]) for k in ('X', 'Y', 'PSI'))
    assert np.all(np.isfinite(pl['x'])) and np.all(pl['cam_lost'] == 0)
    assert np.max(np.hypot(pl['x'] - X, pl['y'] - Y)) < 0.003
    assert np.max(np.abs(pl['theta'] - PSI - 2 * math.pi * np.round((pl['theta'] - PSI) / (2 * math.pi)))) < 0.005
    assert np.max(np.abs(np.diff(pl['theta']))) < 0.1                 # unwrapped: no 2 pi jump
    assert m['r_reason'] == '' and m['e_est_reason'] == ''
    ex = expected(test, log, tr, m['t0_metric_us'])
    for k, v in ex.items():
        assert m[k] == pytest.approx(v, abs=0.002 if k.startswith('e_p') else 0.004), (k, m[k], v)
    assert m['e_est_p_max'] < 0.003 and m['e_est_psi_max'] < 0.005
    for k in ('e_vx_true', 'e_vy_true', 'e_wz_true'):                  # not on the car (spec P2-3 table)
        assert math.isnan(m[k])


def test_lateral_offset_and_delay_are_measured(tmp_path):
    log, tr, info, pl, m = scored('A-L2', tmp_path, lateral=0.03)
    ex = expected('A-L2', log, tr, m['t0_metric_us'])
    assert ex['e_p_max'] > 0.03 and m['e_p_max'] == pytest.approx(ex['e_p_max'], abs=0.002)
    assert m['e_p_rms'] == pytest.approx(ex['e_p_rms'], abs=0.002)
    assert m['e_est_p_max'] < 0.003                                    # the firmware pose has the same offset
    (tmp_path / 'd').mkdir()
    log, tr, info, pl, m = scored('A-L2', tmp_path / 'd', delay_s=0.1)
    ex = expected('A-L2', log, tr, m['t0_metric_us'])
    assert ex['e_p_max'] > 0.05 and m['e_p_max'] == pytest.approx(ex['e_p_max'], abs=0.002)


def test_rest_pose_is_the_mean_before_t0_not_the_value_at_t0(tmp_path):
    """+2 cm on x over [-0.5, -0.25) s, -2 cm over [-0.25, 0): the mean over the 0.5 s rest is the true start, so
    the family R matches the truth; the pose at t = 0, or a mean over a shorter rest, would be 2 cm off."""
    log, tr, info, pl, m = scored('A-L2', tmp_path, frames_kw=dict(rest_noise=0.02))
    t0 = m['t0_metric_us']
    t = (pl['timestamp_us'] - t0) * 1e-6
    rest = (t >= -M.REST_S) & (t < 0)
    assert np.ptp(pl['x'][rest]) > 0.01                               # the noise is on the rest lines
    ex = expected('A-L2', log, tr, t0)
    assert m['e_p_max'] == pytest.approx(ex['e_p_max'], abs=0.003)
    pl2 = dict(pl)
    pl2['x'] = pl['x'].copy()
    pl2['x'][np.flatnonzero(rest)[3]] = np.nan                        # a rest log line without a pose: not used
    assert M.score_run(log, 'A-L2', 'real', plant=pl2)['e_p_max'] == pytest.approx(m['e_p_max'], abs=1e-12)
    cam = pl['_cam']
    tc = (cam['timestamp_us'] - t0) * 1e-6
    keep = ~((tc >= -0.45) & (tc < 0.0))                              # camera frames: rest no longer covered
    pl3 = dict(pl, _cam=dict({k: v[keep] for k, v in cam.items() if k in V.CAM_COLUMNS}, t_miss_s=0.4))
    m3 = M.score_run(log, 'A-L2', 'real', plant=pl3)
    assert m3['r_reason'] == 'no_rest_pose' and math.isnan(m3['e_p_rms'])
    assert M.score_run(log, 'A-L2', 'real', plant=dict(pl, _cam=None))['r_reason'] == 'no_rest_pose'


def test_a_camera_gap_across_the_rest_is_no_rest_pose(tmp_path):
    """S2V2-1 (Codex round 2): frames lost over [-1.0, -0.26) s, a gap of about 0.77 s (> T_MAT 0.68 s) that meets
    the rest [-0.5, 0) though only about 0.25 s of it lies inside: no rest pose, with the default t_miss (the clock
    itself is fine: the start span begins about 0.18 s before t = 0, after the gap)."""
    log, tr, info, pl, m = scored('A-L1', tmp_path, frames_kw=dict(lost=lambda t: -1.0 <= t < -0.26))
    assert info['ok'], info['reason']
    assert pl['_cam']['t_miss_s'] == T_MAT
    assert m['r_reason'] == 'no_rest_pose' and math.isnan(m['e_p_rms'])


def test_lost_log_lines_in_the_rest_do_not_move_the_rest_pose(tmp_path):
    """S2V1-4 (Codex round 1): half of the log lines in [-0.5, -0.25) s lost (the +2 cm half of the rest noise): the
    rest pose comes from the camera frames, so the family R is the same as with every line (a mean over the log
    lines would move by about 1 cm)."""
    log, tr = run_of('A-L2')
    t = (log['timestamp_us'] - T0) * 1e-6
    drop = np.zeros(len(t), bool)
    drop[np.flatnonzero((t >= -0.5) & (t < -0.25))[::2]] = True          # every other line (log_seq jumps of 2)
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    _, _, _, _, m_all = scored('A-L2', tmp_path / 'a', frames_kw=dict(rest_noise=0.02))
    _, _, _, _, m_drop = scored('A-L2', tmp_path / 'b', frames_kw=dict(rest_noise=0.02), drop=drop)
    assert m_drop['e_p_max'] == pytest.approx(m_all['e_p_max'], abs=1e-4)
    assert m_drop['e_p_rms'] == pytest.approx(m_all['e_p_rms'], abs=1e-4)


@pytest.mark.parametrize('gap_s,lost,lost_pose', [(0.5, False, False), (0.9, True, False), (0.5, False, True),
                                                   (0.9, True, True)])
def test_a_camera_gap_longer_than_t_miss_is_camera_lost(tmp_path, gap_s, lost, lost_pose):
    """Frames lost over gap_s in the window: below T_MAT (0.68 s) the pose is interpolated across (cam_lost 0);
    above it the lines have no pose, cam_lost 1 and the family R is refused (camera_lost). Invalid frames that keep a
    (wrong) pose are not used either."""
    log, tr, info, pl, m = scored('A-L1', tmp_path, frames_kw=dict(lost=lambda t: 5.0 <= t < 5.0 + gap_s,
                                                                   lost_pose=lost_pose))
    assert info['ok'] and info['largest_gap_s'] == pytest.approx(gap_s, abs=0.04)
    t_rel = (pl['timestamp_us'] - T0) * 1e-6
    inside = (t_rel > 5.05) & (t_rel < 5.0 + gap_s - 0.05)
    assert np.all(pl['cam_lost'][inside] == (1 if lost else 0))
    assert np.all(np.isnan(pl['x'][inside]) == lost)
    if lost:
        assert m['r_reason'] == 'camera_lost' and m['e_est_reason'] == 'camera_lost' and math.isnan(m['e_p_rms'])
        assert info['n_lost'] == int(np.sum(pl['cam_lost'] == 1)) > 0
        assert math.isfinite(m['e_vx'])                                # family E kept
    else:
        ex = expected('A-L1', log, tr, m['t0_metric_us'])
        assert m['r_reason'] == '' and m['e_p_max'] == pytest.approx(ex['e_p_max'], abs=0.003)


def test_a_video_that_ends_before_the_stop_is_not_scored(tmp_path):
    """The video ends at t = 4 s, before the stop of A-L2 (6 s): the stop edge is not covered (S2V1-2), the clock is
    not ok, no pose rows (S2V1-7: the uncovered end is a camera loss, not a silent gap)."""
    log, tr, info, pl, m = scored('A-L2', tmp_path, frames_kw=dict(t_end=4.0))
    assert not info['ok'] and info['reason'].startswith('clock_sync_failed (edges_not_covered: stop')
    assert pl is None and info['edge_gaps_s']['start'] < T_TRONG


def test_frames_lost_at_both_edges_are_not_scored_with_a_wrong_offset(tmp_path):
    """S2V1-2 (Codex round 1, reproduced: D 46 ms off, corr 0.999, ok): frames lost over [-0.15, 0.35) s and
    [5.85, 6.35) s (gaps 0.54 s < T_MAT, a short window) -> clock_sync_failed (edges_not_covered); the same loss in
    the middle of the run (3.0-3.5 s) is scored."""
    log, tr, info, pl, m = scored('A-L2', tmp_path, frames_kw=dict(
        lost=lambda t: (-0.15 <= t < 0.35) or (5.85 <= t < 6.35)))
    assert not info['ok'] and 'edges_not_covered' in info['reason'] and pl is None
    assert info['edge_gaps_s']['start'] > T_TRONG and info['edge_gaps_s']['stop'] > T_TRONG
    (tmp_path / 'mid').mkdir()
    log, tr, info, pl, m = scored('A-L2', tmp_path / 'mid', frames_kw=dict(lost=lambda t: 3.0 <= t < 3.5))
    assert info['ok'] and info['offset_s'] == pytest.approx(D_TRUE, abs=0.002)
    assert max(info['edge_gaps_s'].values()) < 0.05


def test_frames_lost_just_inside_the_edge_span_are_refused(tmp_path):
    """A gap of 0.15 s (> T_TRONG 0.10 s) 0.5 s after the start, inside [edge - 0.2, edge + 1.0] s: refused; the same
    gap 1.2 s after the start (outside the span) is scored."""
    log, tr, info, pl, m = scored('A-L1', tmp_path, frames_kw=dict(lost=lambda t: 0.5 <= t < 0.65))
    assert not info['ok'] and info['reason'].startswith('clock_sync_failed (edges_not_covered: start')
    (tmp_path / 'b').mkdir()
    log, tr, info, pl, m = scored('A-L1', tmp_path / 'b', frames_kw=dict(lost=lambda t: 1.25 <= t < 1.4))
    assert info['ok'], info['reason']


def test_a_gap_across_the_edge_span_counts_whole(tmp_path):
    """S2V2-2 (Codex round 2): frames lost over 0.14 s straddling the start of the span [first command - 0.2 s, ...]:
    the part inside is only 0.07 s, the gap between the real frames 0.14 s > T_TRONG: refused."""
    log, tr = run_of('A-L1')
    _, rows = log_rows(log, tmp_path)
    a = V.run_commands(rows)[0] - T0 * 1e-6 - V.EDGE_BEFORE_S
    info, out, cam = V.map_video(frames_of(log, tr, lost=lambda t: a - 0.07 <= t < a + 0.07), rows, T_MAT, T_TRONG)
    assert not info['ok'] and info['reason'].startswith('clock_sync_failed (edges_not_covered: start')
    assert info['edge_gaps_s']['start'] > 0.13


def test_largest_gap_and_too_wide_by_hand():
    """S2V2-1, -2, -4: real points only, the time beyond the first / last point infinite, a gap of exactly the
    limit passes."""
    t = np.array([-1.0, -0.2, -0.1, 0.0, 0.1])
    assert M.largest_gap(t, -0.5, 0.0) == pytest.approx(0.8)          # (-1.0, -0.2) meets (-0.5, 0)
    assert M.largest_gap(t, -0.15, 0.05) == pytest.approx(0.1)
    assert M.largest_gap(t, 0.05, 0.3) == math.inf                     # after the last point
    assert M.largest_gap(t, -3.0, -2.0) == math.inf
    assert M.largest_gap(np.array([]), 0.0, 1.0) == math.inf
    assert not M.too_wide(1.1 - 1.0, 0.1) and M.too_wide(0.1 + 2e-6, 0.1)
    assert not M.too_wide(0.68, 0.68) and M.too_wide(0.69, 0.68)


def test_a_start_end_difference_over_50_ms_is_clock_sync_failed(tmp_path):
    """S2V1-1 (Codex round 1): spec P2-3 refuses a start/end difference over 50 ms; o1_time_sync only checks
    |D - mean(start, end)|, which a steady drift keeps near 0. A video clock 1 % fast over A-L1 (thirds about 5 s
    apart: about 50-60 ms) is refused; 0.1 % fast is scored."""
    log, tr, info, pl, m = scored('A-L1', tmp_path, frames_kw=dict(rate=0.012))
    assert not info['ok'] and info['reason'].startswith('clock_sync_failed (start/end'), info['reason']
    assert abs(info['sync']['start_end_diff_s']) > 0.05 and info['sync']['ok']       # o1_time_sync alone: ok
    (tmp_path / 'b').mkdir()
    log, tr, info, pl, m = scored('A-L1', tmp_path / 'b', frames_kw=dict(rate=0.001))
    assert info['ok'], info['reason']


def test_pose_at_by_hand():
    t = np.array([0.0, 0.1, 0.2, 1.2, 1.3])
    x = np.array([0.0, 1.0, 2.0, 12.0, 13.0])
    X, Y, P, L = V.pose_at(np.array([-0.01, 0.0, 0.05, 0.2, 0.7, 1.2, 1.25, 1.3, 1.31]), t, x, x, x, t_miss=0.68)
    np.testing.assert_allclose(X[[1, 2, 3, 5, 6, 7]], [0.0, 0.5, 2.0, 12.0, 12.5, 13.0])
    assert np.isnan(X[0]) and np.isnan(L[0]) and np.isnan(X[8]) and np.isnan(L[8])     # just outside: no pose, no flag
    X2, _, _, L2 = V.pose_at(np.array([-0.69, -0.67, 1.97, 1.99]), t, x, x, x, t_miss=0.68)
    assert np.all(np.isnan(X2)) and L2[0] == 1 and np.isnan(L2[1]) and np.isnan(L2[2]) and L2[3] == 1  # S2V1-7
    assert np.isnan(X[4]) and L[4] == 1                                                 # inside a 1 s gap
    assert L[3] == 0 and L[5] == 0                                                      # on the frames at its ends
    X, Y, P, L = V.pose_at(np.array([0.7]), t, x, x, x, t_miss=1.0)
    assert X[0] == pytest.approx(7.0) and L[0] == 0                                     # 1 s <= t_miss: linear


def test_the_window_starts_at_the_first_command_not_at_run_active(tmp_path):
    """The S02 preparation (zero commands, run_active up from about -2.9 s) is not in the fit window; a stray
    non-zero command before the rising edge, or on a motor-off line, is not the run's first command (S2V1-6)."""
    log, _ = run_of('A-L2')
    _, rows = log_rows(log, tmp_path)
    ta, tb = V.s02_window(rows)
    t_first = (log['timestamp_us'][np.flatnonzero(log['sp_vx'] != 0)[0]]) * 1e-6
    assert ta == pytest.approx(t_first - 0.5) and ta > (T0 * 1e-6 - 1.0)
    log2 = {c: (x.copy() if isinstance(x, np.ndarray) else x) for c, x in log.items()}
    log2['sp_vx'][3] = 0.2                                            # before the rising edge (k = 10)
    k = 50
    log2['sp_vy'][k] = 0.3                                            # inside the run, motor off
    log2['motor_on'][k] = 0.0
    log2['_all'] = log2['_every'] = log2
    _, rows2 = log_rows(log2, tmp_path, 'stray.csv')
    assert V.s02_window(rows2) == (ta, tb)
    assert V.run_commands(rows2)[0] == pytest.approx(t_first)


def test_a_camera_speed_that_does_not_match_is_clock_sync_failed(tmp_path):
    rng = np.random.default_rng(5)
    log, tr, info, pl, m = scored('A-L1', tmp_path, frames_kw=dict(speed=lambda t: rng.uniform(0, 0.3, len(t))))
    assert not info['ok'] and pl is None and info['offset_s'] is None
    assert info['sync']['ok'] is False and info['reason'] == f"clock_sync_failed ({info['sync']['why']})"


def test_gyro_cross_check_is_reported_only(tmp_path):
    log, tr = run_of('A-C1')
    _, rows = log_rows(log, tmp_path)
    info, out, cam = V.map_video(frames_of(log, tr), rows, T_MAT, T_TRONG)
    assert info['ok'] and info['gyro'] is not None and info['gyro']['channel'] == 'gyro'
    assert abs(info['gyro_minus_speed_s']) < 0.01
    fr = frames_of(log, tr)
    for f in fr:
        f['wz'] = 0.3                                                  # a gyro channel that cannot be fitted
    info2, out2, cam2 = V.map_video(fr, rows, T_MAT, T_TRONG)
    assert info2['ok'] and info2['offset_s'] == info['offset_s'] and out2 == out and cam2 == cam


def test_duplicate_log_lines_are_dropped_and_time_must_increase(tmp_path):
    log, _ = run_of('A-L2')
    p, rows = log_rows(log, tmp_path)
    lines = open(p, encoding='utf-8').read().splitlines()
    (tmp_path / 'dup.csv').write_text('\n'.join(lines[:200] + [lines[199]] + lines[200:]) + '\n', encoding='utf-8')
    assert len(V.read_log_rows(str(tmp_path / 'dup.csv'))) == len(rows)
    sw = lines[:200] + [lines[201], lines[200]] + lines[202:]          # two lines swapped (UDP): put back in order
    (tmp_path / 'swap.csv').write_text('\n'.join(sw) + '\n', encoding='utf-8')
    assert V.read_log_rows(str(tmp_path / 'swap.csv')) == rows
    head = lines[0].split(',')
    back = lines[150].split(',')
    back[head.index('vx')] = '9.5'                                     # same time, other content: refused
    (tmp_path / 'back.csv').write_text('\n'.join(lines[:200] + [','.join(back)] + lines[200:]) + '\n',
                                       encoding='utf-8')
    with pytest.raises(ValueError, match='conflicting lines'):            # same log_seq, other content
        V.read_log_rows(str(tmp_path / 'back.csv'))
    # S2V2-3 (Codex round 2): lines of another boot (log_seq goes on, time goes back) are refused as the scorer does
    seq, ts = head.index('log_seq'), head.index('timestamp_us')
    last = lines[-1].split(',')
    boot = []
    for k in range(5):
        c = list(last)
        c[seq], c[ts] = str(int(last[seq]) + 1 + k), str(100_000 + 10_000 * k)
        boot.append(','.join(c))
    (tmp_path / 'boot.csv').write_text('\n'.join(lines + boot) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='DUT restart'):
        V.read_log_rows(str(tmp_path / 'boot.csv'))
    with pytest.raises(ValueError, match='DUT restart'):
        M.load_log(str(tmp_path / 'boot.csv'))


def test_command_line_and_run_dir_metrics_score_the_car(tmp_path, capsys):
    """frames.csv + log -> <run>_plant.csv, <run>_plant_video.json (never overwritten); run_dir_metrics scores the
    car from them; a video that is not ok or missing gives r_reason, no family R, family E kept."""
    log, tr = run_of('A-L2', lateral=0.02)
    lp, _ = log_rows(log, tmp_path, 'r1.csv')
    c7_measure.write_frames(str(tmp_path / 'frames.csv'), frames_of(log, tr))
    (tmp_path / 'track.json').write_text('{}', encoding='utf-8')
    out = str(tmp_path / 'r1_plant.csv')
    hs = hang_so(tmp_path)
    assert V.main(['--frames', str(tmp_path / 'frames.csv'), '--log', lp, '--hang-so', hs, '--out', out]) == 0
    info = json.loads((tmp_path / 'r1_plant_video.json').read_text(encoding='utf-8'))
    assert info['ok'] and info['inputs']['frames_sha256'] == V.sha256(str(tmp_path / 'frames.csv'))
    assert info['inputs']['track_json_sha256'] == V.sha256(str(tmp_path / 'track.json'))
    assert info['inputs']['log_sha256'] == V.sha256(lp)
    assert info['plant_sha256'] == V.sha256(out) and info['cam_sha256'] == V.sha256(str(tmp_path / 'r1_plant_cam.csv'))
    assert V.main(['--frames', str(tmp_path / 'frames.csv'), '--log', lp, '--hang-so', hs, '--out', out]) == 2
    assert '#ket_qua,ma=2' in capsys.readouterr().out
    pl = M.load_plant(out)
    assert set(V.COLUMNS) <= set(pl) and np.all(np.isnan(pl['vx_b']))
    rng = np.random.default_rng(9)                                     # r2: camera speed unrelated -> exit 1
    lp2, _ = log_rows(log, tmp_path, 'r2.csv')
    c7_measure.write_frames(str(tmp_path / 'f2.csv'), frames_of(log, tr, speed=lambda t: rng.uniform(0, 0.3, len(t))))
    assert V.main(['--frames', str(tmp_path / 'f2.csv'), '--log', lp2, '--hang-so', hs, '--out', str(tmp_path / 'r2_plant.csv')]) == 1
    log_rows(log, tmp_path, 'r3.csv')                                  # r3: no video at all
    (tmp_path / 'runs.csv').write_text(
        'run_id,platform,controller,test_id,speed_mps,rep,file,aborted,block_status,result,reason\n'
        'r1,real,PID,A-L2,0.5,1,r1.csv,,,completed,\n'
        'r2,real,PID,A-L2,0.5,2,r2.csv,,,completed,\n'
        'r3,real,PID,A-L2,0.5,3,r3.csv,,,completed,\n', encoding='utf-8')
    o = M.run_dir_metrics(str(tmp_path))
    ex = expected('A-L2', log, tr, o[0]['t0_metric_us'])
    assert o[0]['r_reason'] == '' and o[0]['e_p_max'] == pytest.approx(ex['e_p_max'], abs=0.002)
    assert o[0]['t0_metric_method'] == 'two_anchors'
    assert o[1]['r_reason'].startswith('clock_sync_failed') and math.isnan(o[1]['e_p_rms'])
    assert math.isfinite(o[1]['e_vx'])
    assert o[2]['r_reason'] == 'no_video_file'
    # S2V1-3: the json vouches for these files only
    import shutil
    for what in ('plant', 'cam', 'log'):
        d = tmp_path / f'mm_{what}'
        d.mkdir()
        for f in ('r1.csv', 'r1_plant.csv', 'r1_plant_cam.csv', 'r1_plant_video.json'):
            shutil.copy(tmp_path / f, d / f)
        f = d / ('r1.csv' if what == 'log' else f'r1_plant{"_cam" if what == "cam" else ""}.csv')
        txt = f.read_text(encoding='utf-8').splitlines()
        cells = txt[60].split(',')
        cells[1] = cells[1] + '1'                                      # one digit more in one cell
        txt[60] = ','.join(cells)
        f.write_text('\n'.join(txt) + '\n', encoding='utf-8')
        (d / 'runs.csv').write_text('run_id,platform,controller,test_id,speed_mps,rep,file\n'
                                    'r1,real,PID,A-L2,0.5,1,r1.csv\n', encoding='utf-8')
        assert M.run_dir_metrics(str(d))[0]['r_reason'] == f'video_mismatch ({what})', what
    os.remove(tmp_path / 'r1_plant_cam.csv')
    assert M.run_dir_metrics(str(tmp_path), 'm3.csv')[0]['r_reason'] == 'no_cam_file'


def test_the_car_on_stands_has_no_family_r():
    m = M.score_run(mk('A-K1'), 'A-K1', 'real')
    assert m['r_reason'] == 'real_lifted' and 'rms_w_err_true' not in m and 'e_p_rms' not in m


def test_an_older_log_without_log_seq_is_deduplicated_and_sorted(tmp_path):
    """A 37-column log (no log_seq, no motor_on): byte-for-byte duplicates dropped, lines sorted by timestamp_us;
    two different lines at one time refused."""
    log, _ = run_of('A-L2')
    head = COLUMNS[:37]
    assert 'log_seq' not in head
    lines = [','.join(head)]
    for i in range(300):
        lines.append(','.join(str(int(x)) if float(x).is_integer() else f'{float(x):.9f}'
                              for x in (log[c][i] if c in log else 0.0 for c in head)))
    body = lines[1:]
    mixed = body[:100] + [body[99], body[99]] + [body[101], body[100]] + body[102:]
    (tmp_path / 'old.csv').write_text('\n'.join([lines[0]] + mixed) + '\n', encoding='utf-8')
    rows = V.read_log_rows(str(tmp_path / 'old.csv'))
    assert len(rows) == 300 and [r['timestamp_us'] for r in rows] == [b.split(',')[0] for b in body]
    bad = body[50].split(',')
    bad[head.index('vx')] = '9.5'
    (tmp_path / 'old_bad.csv').write_text('\n'.join([lines[0]] + body + [','.join(bad)]) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='does not increase'):
        V.read_log_rows(str(tmp_path / 'old_bad.csv'))


def test_span_ends_a_float_step_off_a_frame_do_not_flip_the_gap():
    """S2V3-1 (Codex round 3): a span end computed as e - D + margin sits 1e-15 s past the last frame (or past a
    frame that ends a long gap): the frame bounds the span, neither the infinite time after the last frame nor the
    long gap ending on the start end meets it."""
    t = np.r_[np.arange(0.0, 8.02, 1 / 30.0)[:-1], 8.02]
    b = 56.02 - 49 + 1                                                # 8.020000000000003 > 8.02
    assert b > t[-1]
    assert M.largest_gap(t, b - 1.2, b) < 0.06                        # not inf
    t2 = np.r_[np.arange(2.0, 2.96, 1 / 30.0), 3.1, 3.1 + np.arange(1, 30) / 30.0]   # a 0.14 s gap ending at 3.1
    a = 3.3 - 0.2                                                     # 3.0999999999999996 < 3.1
    assert a < 3.1
    assert M.largest_gap(t2, a, a + 0.5) < 0.05
    assert M.largest_gap(t2, a - 0.01, a + 0.5) > 0.13                 # the gap really inside the span: counted
    assert M.largest_gap(t, 8.03, 9.0) == math.inf                     # after the last frame by more: still inf


def test_read_log_rows_takes_a_path_object(tmp_path):
    """S2V3-2 (Codex round 3): a pathlib.Path works for a DUT log (dut_rows joins strings)."""
    log, _ = run_of('A-L2')
    p, rows = log_rows(log, tmp_path)
    import pathlib
    assert V.read_log_rows(pathlib.Path(p)) == rows


def test_the_thresholds_come_only_from_the_c8_constants_file(tmp_path, capsys):
    """fix/s02-video-hang-so (CODE 17/11): no number kept in the tool (the c7_score constants T_MISS, T_GAP are gone);
    t_mat_s, t_trong_s of --hang-so decide. A 0.5 s camera gap mid-run: interpolated with t_mat_s 0.68, camera_lost
    with 0.4; a 0.15 s gap 0.5 s after the start: edges_not_covered with t_trong_s 0.10, covered with 0.20. The json
    binds the constants file (SHA-256, version, trang_thai)."""
    assert not hasattr(V, 'T_MISS') and not hasattr(V, 'T_GAP')
    log, tr = run_of('A-L2')
    lp, _ = log_rows(log, tmp_path, 'r.csv')

    def run(name, lost, **hs_over):
        d = tmp_path / name
        d.mkdir()
        c7_measure.write_frames(str(d / 'frames.csv'), frames_of(log, tr, lost=lost))
        hs = hang_so(d, **hs_over)
        code = V.main(['--frames', str(d / 'frames.csv'), '--log', lp, '--hang-so', hs, '--out', str(d / 'p.csv')])
        info = json.loads((d / 'p_video.json').read_text(encoding='utf-8'))
        return code, info, hs

    mid = lambda t: 3.0 <= t < 3.5                                     # noqa: E731
    code, info, hs = run('a', mid)
    assert code == 0 and info['ok'] and info['n_lost'] == 0 and info['t_miss_s'] == 0.68
    assert info['inputs']['hang_so_sha256'] == V.sha256(hs) and info['inputs']['hang_so_trang_thai'] == 'de_xuat'
    assert info['inputs']['hang_so_version'] == 'c7_hang_so_v1'
    code, info, _ = run('b', mid, t_mat_s=0.4)
    assert code == 0 and info['n_lost'] > 0 and info['t_miss_s'] == 0.4
    edge = lambda t: 0.5 <= t < 0.65                                   # noqa: E731
    code, info, _ = run('c', edge)
    assert code == 1 and 'edges_not_covered' in info['reason'] and info['t_gap_s'] == 0.10
    code, info, _ = run('d', edge, t_trong_s=0.20)
    assert code == 0 and info['ok'] and info['t_gap_s'] == 0.20


def test_the_constants_file_is_required_and_checked(tmp_path, capsys):
    """--hang-so is required; a file c7_score.load_hang_so refuses (t_trong_s not below t_mat_s, another version) is
    an error (exit 2), nothing written."""
    log, tr = run_of('A-L1')
    lp, _ = log_rows(log, tmp_path, 'r.csv')
    c7_measure.write_frames(str(tmp_path / 'frames.csv'), frames_of(log, tr))
    base = ['--frames', str(tmp_path / 'frames.csv'), '--log', lp, '--out', str(tmp_path / 'p.csv')]
    with pytest.raises(SystemExit):
        V.main(base)
    for k, over in enumerate((dict(t_trong_s=0.68), dict(version='x'), dict(duong_lenh=[]), dict(imu=[]),
                              dict(imu=dict(U_IMU_deg=[1.0])))):            # S2H1-1: wrong structure too
        hs = hang_so(tmp_path, name=f'bad{k}.json', **over)
        assert V.main(base + ['--hang-so', hs]) == 2
        assert '#ket_qua,ma=2' in capsys.readouterr().out
        assert not (tmp_path / 'p.csv').exists() and not (tmp_path / 'p_video.json').exists()
        assert not (tmp_path / 'p_cam.csv').exists()
