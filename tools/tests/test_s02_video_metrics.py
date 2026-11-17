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
              lost_pose=False):
    """frames.csv rows (c7_measure.read_frames layout): frames every 1/fps (+- 2 ms) from 1 s before the first log
    line to t_end (s from T0; default 1 s after the last line), pose, speed, wz of the truth at the frame's true time;
    lost(t) -> True for an invalid frame (no pose; lost_pose: a wrong but finite pose, 0.5 m off, as a frame refused
    by c7_measure's gates keeps it); rest_noise: +rest_noise m on x over [-0.5, -0.25) s and -rest_noise over
    [-0.25, 0) (mean 0 over the 0.5 s rest only); speed(t) replaces the camera speed."""
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
        rows.append(dict(idx=k, t_s=t_log[k] - D, valid=not bad, why=['lost'] if bad else [], n_marks=4,
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


def log_rows(log, tmp_path, name='log.csv'):
    p = tmp_path / name
    _write_log(p, log)
    return str(p), V.read_log_rows(str(p))


def plant_dict(out):
    """s02_metrics.load_plant of the tool's rows (cells as written)."""
    data = np.array([[float(c) if c != '' else np.nan for c in r] for r in out]).reshape(len(out), len(V.COLUMNS))
    pl = {c: data[:, i] for i, c in enumerate(V.COLUMNS)}
    pl['_row'] = {int(t): i for i, t in enumerate(pl['timestamp_us'])}
    return pl


def scored(test, tmp_path, frames_kw=None, **kw):
    log, tr = run_of(test, **kw)
    _, rows = log_rows(log, tmp_path)
    info, out = V.map_video(frames_of(log, tr, **(frames_kw or {})), rows)
    pl = plant_dict(out) if out else None
    m = M.score_run(log, test, 'real', plant=pl) if pl is not None else None
    return log, tr, info, pl, m


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
    pl2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pl.items()}
    pl2['x'][np.flatnonzero(rest)[3]] = np.nan                        # one rest line without a pose
    m2 = M.score_run(log, 'A-L2', 'real', plant=pl2)
    assert m2['r_reason'] == 'no_rest_pose' and math.isnan(m2['e_p_rms'])


@pytest.mark.parametrize('gap_s,lost,lost_pose', [(0.5, False, False), (0.9, True, False), (0.5, False, True),
                                                   (0.9, True, True)])
def test_a_camera_gap_longer_than_t_miss_is_camera_lost(tmp_path, gap_s, lost, lost_pose):
    """Frames lost over gap_s in the window: below T_MISS (0.68 s) the pose is interpolated across (cam_lost 0);
    above it the lines have no pose, cam_lost 1 and the family R is refused (camera_lost). Invalid frames that keep a
    (wrong) pose are not used either."""
    assert V.T_MISS == 0.68
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


def test_no_extrapolation_past_the_last_frame(tmp_path):
    """The video ends at t = 4 s (inside the window [1, 6] s of A-L2; the fit window is cut there): later lines have
    no pose and no cam_lost (the camera does not cover them, not a gap); the family R has a gap (plant_pose_gap), not
    camera_lost. (A-L1 cut at 9 s: its last third has no camera, the clock is not ok.)"""
    log, tr, info, pl, m = scored('A-L2', tmp_path, frames_kw=dict(t_end=4.0))
    assert info['ok'], info['reason']
    t_rel = (pl['timestamp_us'] - T0) * 1e-6
    after = t_rel > 4.05
    assert np.all(np.isnan(pl['x'][after])) and np.all(np.isnan(pl['cam_lost'][after]))
    assert info['n_outside'] == int(np.sum(np.isnan(pl['cam_lost'])))
    assert m['r_reason'] == 'plant_pose_gap' and math.isnan(m['e_p_rms'])


def test_pose_at_by_hand():
    t = np.array([0.0, 0.1, 0.2, 1.2, 1.3])
    x = np.array([0.0, 1.0, 2.0, 12.0, 13.0])
    X, Y, P, L = V.pose_at(np.array([-0.01, 0.0, 0.05, 0.2, 0.7, 1.2, 1.25, 1.3, 1.31]), t, x, x, x, t_miss=0.68)
    np.testing.assert_allclose(X[[1, 2, 3, 5, 6, 7]], [0.0, 0.5, 2.0, 12.0, 12.5, 13.0])
    assert np.isnan(X[0]) and np.isnan(L[0]) and np.isnan(X[8]) and np.isnan(L[8])     # outside: no pose, no flag
    assert np.isnan(X[4]) and L[4] == 1                                                 # inside a 1 s gap
    assert L[3] == 0 and L[5] == 0                                                      # on the frames at its ends
    X, Y, P, L = V.pose_at(np.array([0.7]), t, x, x, x, t_miss=1.0)
    assert X[0] == pytest.approx(7.0) and L[0] == 0                                     # 1 s <= t_miss: linear


def test_the_window_starts_at_the_first_command_not_at_run_active(tmp_path):
    """The S02 preparation (zero commands, run_active up from about -2.9 s) is not in the fit window."""
    log, _ = run_of('A-L2')
    _, rows = log_rows(log, tmp_path)
    ta, tb = V.s02_window(rows)
    t_first = (log['timestamp_us'][np.flatnonzero(log['sp_vx'] != 0)[0]]) * 1e-6
    assert ta == pytest.approx(t_first - 0.5) and ta > (T0 * 1e-6 - 1.0)


def test_a_camera_speed_that_does_not_match_is_clock_sync_failed(tmp_path):
    rng = np.random.default_rng(5)
    log, tr, info, pl, m = scored('A-L1', tmp_path, frames_kw=dict(speed=lambda t: rng.uniform(0, 0.3, len(t))))
    assert not info['ok'] and info['reason'].startswith('clock_sync_failed') and pl is None
    assert info['offset_s'] is None


def test_gyro_cross_check_is_reported_only(tmp_path):
    log, tr = run_of('A-C1')
    _, rows = log_rows(log, tmp_path)
    info, out = V.map_video(frames_of(log, tr), rows)
    assert info['ok'] and info['gyro'] is not None and info['gyro']['channel'] == 'gyro'
    assert abs(info['gyro_minus_speed_s']) < 0.01
    fr = frames_of(log, tr)
    for f in fr:
        f['wz'] = 0.3                                                  # a gyro channel that cannot be fitted
    info2, out2 = V.map_video(fr, rows)
    assert info2['ok'] and info2['offset_s'] == info['offset_s'] and out2 == out


def test_duplicate_log_lines_are_dropped_and_time_must_increase(tmp_path):
    log, _ = run_of('A-L2')
    p, rows = log_rows(log, tmp_path)
    lines = open(p, encoding='utf-8').read().splitlines()
    (tmp_path / 'dup.csv').write_text('\n'.join(lines[:200] + [lines[199]] + lines[200:]) + '\n', encoding='utf-8')
    assert len(V.read_log_rows(str(tmp_path / 'dup.csv'))) == len(rows)
    head = lines[0].split(',')
    back = lines[150].split(',')
    back[head.index('vx')] = '9.5'                                     # not a byte-for-byte duplicate
    (tmp_path / 'back.csv').write_text('\n'.join(lines[:200] + [','.join(back)] + lines[200:]) + '\n',
                                       encoding='utf-8')
    with pytest.raises(ValueError, match='does not increase'):
        V.read_log_rows(str(tmp_path / 'back.csv'))


def test_command_line_and_run_dir_metrics_score_the_car(tmp_path, capsys):
    """frames.csv + log -> <run>_plant.csv, <run>_plant_video.json (never overwritten); run_dir_metrics scores the
    car from them; a video that is not ok or missing gives r_reason, no family R, family E kept."""
    log, tr = run_of('A-L2', lateral=0.02)
    lp, _ = log_rows(log, tmp_path, 'r1.csv')
    c7_measure.write_frames(str(tmp_path / 'frames.csv'), frames_of(log, tr))
    (tmp_path / 'track.json').write_text('{}', encoding='utf-8')
    out = str(tmp_path / 'r1_plant.csv')
    assert V.main(['--frames', str(tmp_path / 'frames.csv'), '--log', lp, '--out', out]) == 0
    info = json.loads((tmp_path / 'r1_plant_video.json').read_text(encoding='utf-8'))
    assert info['ok'] and info['inputs']['frames_sha256'] == V.sha256(str(tmp_path / 'frames.csv'))
    assert info['inputs']['track_json_sha256'] == V.sha256(str(tmp_path / 'track.json'))
    assert info['inputs']['log_sha256'] == V.sha256(lp)
    assert V.main(['--frames', str(tmp_path / 'frames.csv'), '--log', lp, '--out', out]) == 2
    assert '#ket_qua,ma=2' in capsys.readouterr().out
    pl = M.load_plant(out)
    assert set(V.COLUMNS) <= set(pl) and np.all(np.isnan(pl['vx_b']))
    rng = np.random.default_rng(9)                                     # r2: camera speed unrelated -> exit 1
    lp2, _ = log_rows(log, tmp_path, 'r2.csv')
    c7_measure.write_frames(str(tmp_path / 'f2.csv'), frames_of(log, tr, speed=lambda t: rng.uniform(0, 0.3, len(t))))
    assert V.main(['--frames', str(tmp_path / 'f2.csv'), '--log', lp2, '--out', str(tmp_path / 'r2_plant.csv')]) == 1
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


def test_the_car_on_stands_has_no_family_r():
    m = M.score_run(mk('A-K1'), 'A-K1', 'real')
    assert m['r_reason'] == 'real_lifted' and 'rms_w_err_true' not in m and 'e_p_rms' not in m
