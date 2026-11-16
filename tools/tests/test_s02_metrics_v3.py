"""s02_metrics.score_run / run_dir_metrics (CODE_S02 design v3 sections 18-21; spec S02 v1.13 P2-2b, P2-3, P2-5,
P2-6): t = 0 per platform, window [1 s, T_cmd], family E and R, step quantities, loss on the tick stream, rows of
runs that are not completed, metrics_columns.json. Logs are dicts as s02_metrics.load_log returns them (and CSV
files for run_dir_metrics), built from a model of a DUT that follows r(t) with a chosen delay."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import s02_metrics as M  # noqa: E402
import s02_ref as R  # noqa: E402
from agv_logger import COLUMNS  # noqa: E402

T0 = 50_000_000.0                         # sender t = 0 on the DUT clock (us)


def mk(test='A-L2', delay_s=0.0, lag_us=15_000.0, prep_s=3.0, tail_s=1.5, jitter_us=0.0, seed=1, wz_off=0.0,
       pose=None, drop=None):
    """Motor-on log of one run: ticks every 10 ms (+- jitter) from t = -prep_s to T_cmd + tail_s; the firmware
    command (sp_*) is r(t - lag) (the anchors), the firmware speeds are r(t - delay); counters grow 10 per tick.
    pose(t) -> (x, y, th) of the firmware pose (default: the command pose of r(t - delay)). drop: mask of lines
    to drop (lost lines)."""
    rng = np.random.default_rng(seed)
    T = R.t_cmd(test)
    n = int(round((prep_s + T + tail_s) / 0.01))
    k = np.arange(n)
    ts = T0 - prep_s * 1e6 + k * 10_000.0 + np.round(rng.uniform(-jitter_us, jitter_us, n))
    t = (ts - T0) * 1e-6
    v = R.reference(test, t - delay_s)
    sp = R.reference(test, t - lag_us * 1e-6)
    w = R.ik(*v.T) * 60 / (2 * math.pi)
    p = R.pose_ref(test, t - delay_s) if pose is None else np.array([pose(x) for x in t])
    log = dict(timestamp_us=ts, tick=k + 100.0, log_seq=k + 1000.0, motor_on=np.ones(n),
               run_active=np.where(k < 10, 0.0, 1.0), pos_reset_n=np.ones(n),
               vx=v[:, 0], vy=v[:, 1], wz=v[:, 2], wz_meas=v[:, 2] + wz_off, gz_fused=v[:, 2], wz_src=np.ones(n),
               sp_vx=sp[:, 0], sp_vy=sp[:, 1], sp_wz=sp[:, 2], traj_age_ms=np.full(n, 8.0),
               g_valid=np.ones(n), bias_gz=np.zeros(n), bias_updates=np.zeros(n),
               pos_x=p[:, 0], pos_y=p[:, 1], pos_th=p[:, 2])
    for c, x in zip(M.WHEEL_RPM, w.T):
        log[c] = x
    for c in ('fl', 'fr', 'rl', 'rr'):
        log[f'dpwm_{c}'] = 10.0 * k
        log[f'sat_{c}'] = np.zeros(n)
    if drop is not None:
        log = {c: x[~drop] for c, x in log.items()}
    log['_all'] = log
    log['_every'] = log
    return log


def plant_of(log, test='A-L2', delay_s=0.0, off=(0.0, 0.0, 0.0), lateral=0.0, gap=None):
    """Plant table on the log lines: the command pose of r(t - delay) placed at world pose off (x0, y0, th0),
    shifted by lateral (m, left of the start heading) from t = 0.5 s on (a constant offset from before t = 0 is
    removed by taking each pose relative to its value at t = 0, P2-3); wheel speeds of r(t - delay). gap: mask of
    empty rows."""
    ts = log['timestamp_us']
    t = (ts - T0) * 1e-6
    p = R.pose_ref(test, t - delay_s)
    psi0 = R.start_heading(test)
    lat = np.where(t >= 0.5, lateral, 0.0)
    p[:, 0] -= lat * math.sin(psi0)
    p[:, 1] += lat * math.cos(psi0)
    x0, y0, th0 = off
    a = th0 - psi0
    X = x0 + math.cos(a) * p[:, 0] - math.sin(a) * p[:, 1]
    Y = y0 + math.sin(a) * p[:, 0] + math.cos(a) * p[:, 1]
    TH = R.wrap(th0 + (p[:, 2] - psi0))
    v = R.reference(test, t - delay_s)
    w = R.ik(*v.T)
    pl = dict(timestamp_us=ts, x=X, y=Y, theta=TH, vx_b=v[:, 0], vy_b=v[:, 1], wz=v[:, 2],
              w1=w[:, 0], w2=w[:, 1], w3=w[:, 2], w4=w[:, 3])
    if gap is not None:
        for c in pl:
            if c != 'timestamp_us':
                pl[c] = np.where(gap, np.nan, pl[c])
    pl['_row'] = {int(x): i for i, x in enumerate(ts)}
    return pl


def hil(log, test='A-L2', plant=None, sender=T0):
    return M.score_run(log, test, 'hil', plant=plant, sender_t0_us=sender)


def test_perfect_tracking_scores_zero_and_window_is_1s_to_T():
    log = mk()
    m = hil(log, plant=plant_of(log))
    assert m['t0_metric_method'] == 'sender' and m['metric_reason'] == ''
    for k in ('e_vx', 'e_vy', 'e_wz', 'rms_w_err', 'e_yaw_imu', 'e_vx_true', 'e_p_rms', 'e_psi_max', 'e_est_p_rms'):
        assert m[k] == pytest.approx(0.0, abs=1e-9), k
    assert m['window_rows'] == 501 and m['window_s'] == pytest.approx(5.0)        # A-L2: [1, 6] s, 10 ms
    assert m['t0_lag_ms'] == pytest.approx(20.0, abs=0.01)     # command 15 ms late: first tick showing it at 20 ms
    assert m['pwm_tv'] == pytest.approx(10 / 1000 / 0.01)


@pytest.mark.parametrize('shift_us', [-30_000.0, 30_000.0])
def test_moving_t0_moves_the_window_by_whole_lines(shift_us):
    log = mk()
    m = hil(log, sender=T0 + shift_us)
    t = (log['timestamp_us'] - (T0 + shift_us)) * 1e-6
    assert m['window_rows'] == int(np.sum((t >= 1.0) & (t <= R.t_cmd('A-L2'))))
    assert m['t0_lag_ms'] == pytest.approx(20.0 - shift_us / 1000, abs=0.01)


def test_ak1_window_starts_at_1s_not_after_the_first_nonzero_command():
    m = hil(mk('A-K1'), 'A-K1')
    assert m['window_rows'] == 801 and m['window_s'] == pytest.approx(8.0)


def test_delay_50ms_gives_the_rms_of_the_shifted_steps():
    log = mk('A-K1', delay_s=0.05)
    m = hil(log, 'A-K1')
    t = (log['timestamp_us'] - T0) * 1e-6
    win = (t >= 1) & (t <= 9)
    err = R.reference('A-K1', t[win])[:, 0] - R.reference('A-K1', t[win] - 0.05)[:, 0]
    assert m['e_vx'] == pytest.approx(math.sqrt(np.mean(err ** 2)), rel=1e-9) and m['e_vx'] > 0.03
    assert m['rise_s_seg2'] == pytest.approx(0.0) and m['rise_s_seg3'] == pytest.approx(0.0)


def test_e_wz_follows_wz_meas_not_encoder_wz():
    m = hil(mk('A-C2', wz_off=0.1), 'A-C2')
    assert m['e_wz'] == pytest.approx(0.1) and m['e_vx'] == pytest.approx(0.0, abs=1e-12)
    assert m['wz_src_enc_frac'] == 0.0
    assert m['e_yaw_imu'] == pytest.approx(0.0, abs=1e-12)              # gz_fused, not wz_meas


def test_pwm_tv_uses_the_dut_clock_with_jittered_ticks():
    log = mk(jitter_us=3000.0)
    m = hil(log)
    t = (log['timestamp_us'] - T0) * 1e-6
    i = np.flatnonzero((t >= 1) & (t <= 6))
    want = (log['dpwm_fl'][i[-1]] - log['dpwm_fl'][i[0]]) / 1000 / ((log['timestamp_us'][i[-1]] - log['timestamp_us'][i[0]]) * 1e-6)
    assert m['pwm_tv'] == pytest.approx(want, rel=1e-12)
    assert m['pwm_tv'] != pytest.approx(10 / 1000 / 0.01, rel=1e-6)


def test_no_sender_t0_on_hil_never_falls_back_to_the_anchors():
    m = M.score_run(mk(), 'A-L2', 'hil', sender_reason='map_not_certified')
    assert m['t0_metric_method'] == 'none' and m['metric_reason'] == 'no_sender_t0 (map_not_certified)'
    assert math.isnan(m['e_vx']) and math.isfinite(m['t0_us']) and m['r_reason'] == 'no_t0'


def test_car_uses_the_anchors_and_sim_the_exact_prep():
    log = mk(lag_us=20_000.0)
    m = M.score_run(log, 'A-L2', 'real')
    assert m['t0_metric_method'] == 'two_anchors' and m['t0_metric_us'] == pytest.approx(T0 + 20_000, abs=1)
    assert m['r_reason'] == 'real_needs_video' and 'e_p_rms' not in m
    sim = mk(lag_us=0.0)
    sim['timestamp_us'] = sim['timestamp_us'] - T0 + 3e6               # fwsim clock: t = 0 at prep_s
    m = M.score_run(sim, 'A-L2', 'sim', prep_s=3.0, plant=None)
    assert m['t0_metric_method'] == 'sim_exact' and m['t0_metric_us'] == 3e6 and m['e_vx'] == pytest.approx(0, abs=1e-12)
    assert m['r_reason'] == 'no_plant_file'
    assert M.score_run(sim, 'A-L2', 'sim')['metric_reason'] == 'no_prep_s'


def test_lateral_offset_and_start_pose():
    """Figure 8 (start heading pi/4) with the plant starting at (1, 2, -2.5 rad): theta crosses +-pi; a 5 cm lateral
    offset gives e_p 5 cm; the firmware pose (command pose) differs from the plant by the same 5 cm."""
    log = mk('A-F1')
    pl = plant_of(log, 'A-F1', off=(1.0, 2.0, -2.5))
    assert np.any(np.abs(np.diff(pl['theta'])) > 6)                   # heading turns 3 pi / 2 clockwise: wraps
    m = hil(log, 'A-F1', pl)
    assert m['e_p_max'] == pytest.approx(0.0, abs=1e-9) and m['e_psi_max'] == pytest.approx(0.0, abs=1e-9)
    m = hil(log, 'A-F1', plant_of(log, 'A-F1', off=(1.0, 2.0, -2.5), lateral=0.05))
    assert m['e_p_rms'] == pytest.approx(0.05, abs=1e-9) and m['e_p_max'] == pytest.approx(0.05, abs=1e-9)
    assert m['e_est_p_rms'] == pytest.approx(0.05, abs=1e-9)         # firmware pose = command pose
    m = hil(log, 'A-F1', plant_of(log, 'A-F1', off=(1.0, 2.0, -2.5), lateral=-0.05))
    assert m['e_p_max'] == pytest.approx(0.05, abs=1e-9)


def test_plant_delay_gives_position_lag():
    log = mk('A-L2')
    m = hil(log, plant=plant_of(log, delay_s=0.02))
    assert m['e_p_max'] == pytest.approx(0.5 * 0.02, abs=1e-9) and m['e_est_p_max'] == pytest.approx(0.01, abs=1e-9)


def test_empty_plant_cell_in_window_makes_family_r_undefined():
    log = mk()
    t = (log['timestamp_us'] - T0) * 1e-6
    m = hil(log, plant=plant_of(log, gap=np.abs(t - 3.0) < 0.005))
    assert math.isnan(m['e_p_rms']) and math.isnan(m['e_vx_true']) and 'plant_gap' in m['r_reason']
    m = hil(log, plant=plant_of(log, gap=t < 0.5))                     # no plant pose at t = 0
    assert math.isnan(m['e_p_rms']) and 'no_plant_pose_at_t0' in m['r_reason'] and math.isfinite(m['e_vx_true'])


def test_e_est_undefined_after_a_reset_in_the_window_or_edge_after_t0():
    log = mk()
    t = (log['timestamp_us'] - T0) * 1e-6
    log['pos_reset_n'] = np.where(t > 2.0, 2.0, 1.0)
    m = hil(log, plant=plant_of(log))
    assert math.isnan(m['e_est_p_rms']) and m['e_est_reason'] == 'pos_reset_in_window'
    assert m['e_p_rms'] == pytest.approx(0, abs=1e-9)
    log = mk()
    log['run_active'] = np.where((log['timestamp_us'] - T0) * 1e-6 < 0.5, 0.0, 1.0)   # motors on after t = 0
    m = hil(log, plant=plant_of(log))
    assert 'edge_after_t0' in m['prep_flags'] and m['e_est_reason'] == 'edge_after_t0'


def test_lifted_family_r_uses_true_wheel_speeds():
    log = mk('A-K2')
    pl = plant_of(log, 'A-K2', delay_s=0.03)
    for c in ('vx_b', 'vy_b', 'wz', 'x', 'y', 'theta'):
        pl[c] = np.zeros(len(pl[c]))                                     # kê bánh: body velocity 0 in the plant
    m = hil(log, 'A-K2', pl)
    assert m['rms_w_err_true'] > 0 and 'e_p_rms' not in m
    assert m['rise_s_true_seg2'] == pytest.approx(0.0) and m['e_xl_true_seg2'] == pytest.approx(0.0, abs=1e-9)


def test_loss_on_the_tick_stream_and_generation_pause():
    n = len(mk()['tick'])
    drop = np.zeros(n, bool)
    drop[np.arange(100, n, 40)] = True                                 # 2.5 % of the lines
    m = hil(mk(drop=drop))
    assert m['missing_pct'] == pytest.approx(100 * drop.sum() / n, abs=0.2) and m['missing_over'] == 1
    log = mk()
    log['timestamp_us'] = log['timestamp_us'] + np.where(np.arange(n) >= 400, 30_000.0, 0.0)
    m = hil(log)
    assert m['gen_pause_n'] == 1 and m['missing_pct'] == 0.0 and m['missing_over'] == 0


def test_age_before_the_anchors():
    log = mk()
    t = (log['timestamp_us'] - T0) * 1e-6
    log['traj_age_ms'] = np.where((t > -0.08) & (t < 0.0), 45.0, 8.0)  # first non-zero packet lost
    m = hil(log)
    assert m['age_pre_a_ms'] == 45.0 and m['age_pre_flags'] == 'age_pre_a_over'
    assert m['age_pre_b_ms'] == 8.0


def _write_log(path, log):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        n = len(log['timestamp_us'])
        for i in range(n):
            row = []
            for c in COLUMNS:
                if c in log:
                    x = log[c][i]
                    row.append(int(x) if c in ('timestamp_us', 'tick', 'log_seq', 'motor_on') else f'{x:.6f}')
                else:
                    row.append('0')
            w.writerow(row)


def test_run_dir_rows_for_runs_not_completed_and_columns_json(tmp_path):
    log = mk()
    _write_log(tmp_path / 'ok.csv', log)
    pl = plant_of(log)
    with open(tmp_path / 'ok_plant.csv', 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['timestamp_us', 'x', 'y', 'theta', 'vx_w', 'vy_w', 'vx_b', 'vy_b', 'wz', 'w1', 'w2', 'w3', 'w4']
        w.writerow(cols)
        for i in range(len(pl['timestamp_us'])):
            w.writerow([int(pl['timestamp_us'][i])] + [f'{pl[c][i]:.9f}' if c in pl else '0' for c in cols[1:]])
    (tmp_path / 'ok_plant_map.json').write_text(json.dumps(dict(ok=True, sender_t0_us=T0)), encoding='utf-8')
    (tmp_path / 'runs.csv').write_text(
        'run_id,platform,controller,test_id,speed_mps,rep,file,aborted,block_status,result,reason\n'
        'a,hil,PID,A-L2,0.5,1,missing_a.csv,1,,,estop\n'
        'b,hil,PID,A-L2,0.5,1,missing_b.csv,,battery_rerun,completed,\n'
        'c,hil,PID,A-L2,0.5,1,missing_c.csv,,,replaced,camera\n'
        'd,hil,PID,A-L2,0.5,1,missing_d.csv,,,broken,controller\n'
        'e,hil,PID,A-L2,0.5,1,ok.csv,,,completed,\n', encoding='utf-8')
    out = M.run_dir_metrics(str(tmp_path))
    assert [(o['run_id'], o['status'], o['status_reason']) for o in out] == [
        ('a', 'aborted', 'estop'), ('b', 'battery_rerun', ''), ('c', 'replaced', 'camera'),
        ('d', 'broken', 'controller'), ('e', 'ok', '')]
    assert 'e_vx' not in out[0] and out[4]['e_p_rms'] == pytest.approx(0.0, abs=1e-6)
    assert out[4]['t0_metric_method'] == 'sender'
    with open(tmp_path / 'metrics.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['e_vx'] == '' and rows[4]['status'] == 'ok'
    info = json.loads((tmp_path / 'metrics_columns.json').read_text(encoding='utf-8'))['columns']
    assert info['e_p_rms']['family'] == 'R' and info['e_p_rms']['unit'] == 'm'
    assert 'LOWER BOUND' in info['cmd_age_max_ms']['source'] and info['e_vx']['nonneg'] is True
    assert M.cell_counts(out)[('A-L2', 'hil', 'PID')] == dict(aborted=1, battery_rerun=1, replaced=1, broken=1, ok=1)


def test_hil_map_not_ok_gives_no_sender_t0(tmp_path):
    log = mk()
    _write_log(tmp_path / 'r.csv', log)
    (tmp_path / 'r_plant_map.json').write_text(json.dumps(dict(ok=False, reason='not_certified')), encoding='utf-8')
    (tmp_path / 'runs.csv').write_text('run_id,platform,controller,test_id,speed_mps,rep,file\n'
                                       'r,hil,PID,A-L2,0.5,1,r.csv\n', encoding='utf-8')
    o = M.run_dir_metrics(str(tmp_path))[0]
    assert o['metric_reason'] == 'no_sender_t0 (map_not_certified)' and math.isnan(o['e_vx'])
    os.remove(tmp_path / 'r_plant_map.json')
    o = M.run_dir_metrics(str(tmp_path), 'm2.csv')[0]
    assert o['metric_reason'] == 'no_sender_t0 (no_map_file)'


def test_column_info_of_step_columns():
    i = M.column_info('rise_s_true_seg2')
    assert i['family'] == 'R' and i['unit'] == 's' and i['kind'] == 'metric' and i['nonneg']
    assert M.column_info('no90_seg3')['kind'] == 'flag' and M.column_info('frac90_seg2')['nonneg'] is False


def test_sat_pct_is_the_share_of_saturated_ticks():
    log = mk()
    log['sat_fl'] = 0.5 * (log['tick'] - log['tick'][0])                # half the ticks of FL saturated
    m = hil(log)
    assert m['sat_pct'] == pytest.approx(50.0 / 4)


def test_loss_just_over_one_percent_and_repeated_ticks():
    n = len(mk()['tick'])
    drop = np.zeros(n, bool)
    drop[np.arange(100, n, 67)] = True                                 # about 1.5 % of the lines
    m = hil(mk(drop=drop))
    assert 1.0 < m['missing_pct'] < 2.0 and m['missing_over'] == 1
    log = mk()
    rep = np.r_[np.arange(n), np.arange(200, 210)]                     # 10 ticks logged twice: not a loss
    for c in list(log):
        if not c.startswith('_'):
            log[c] = log[c][np.sort(rep)]
    m = hil(log)
    assert m['missing_pct'] == 0.0


@pytest.mark.parametrize('who', ['plant', 'firmware'])
def test_heading_errors_are_wrapped(who):
    """A heading that turns 0.8 rad/s more than the command (plant), or than the plant (firmware pose): the error
    passes pi inside the window (A-L2: 4.8 rad at 6 s); e_psi, e_est_psi are |wrap(.)| <= pi."""
    extra = lambda x: 0.8 * np.clip(x, 0, None)                      # noqa: E731
    if who == 'firmware':
        log = mk(pose=lambda x: tuple(R.pose_ref('A-L2', [x])[0][:2]) + (float(extra(x)),))
        pl = plant_of(log)
    else:
        log = mk()
        pl = plant_of(log)
        pl['theta'] = R.wrap(pl['theta'] + extra((log['timestamp_us'] - T0) * 1e-6))
    m = hil(log, plant=pl)
    t = (log['timestamp_us'] - T0) * 1e-6
    e = np.abs(R.wrap(extra(t[(t >= 1) & (t <= 6)])))
    k = 'e_psi' if who == 'plant' else 'e_est_psi'
    assert m[f'{k}_max'] <= math.pi + 1e-9 and m[f'{k}_max'] == pytest.approx(e.max(), abs=1e-9)
    assert m[f'{k}_rms'] == pytest.approx(math.sqrt(np.mean(e ** 2)), abs=1e-9)


def test_e_yaw_imu_from_gz_fused_offset_and_empty_cell():
    """S2M1-12 (Codex round 1 of v3): gz_fused 0.07 rad/s off the command gives e_yaw_imu 0.07 (encoder wz and
    wz_meas unchanged); one empty gz_fused cell in the window: e_yaw_imu undefined, the other metrics kept."""
    log = mk('A-C2')
    log['gz_fused'] = log['gz_fused'] + 0.07
    m = hil(log, 'A-C2')
    assert m['e_yaw_imu'] == pytest.approx(0.07) and m['e_wz'] == pytest.approx(0.0, abs=1e-12)
    t = (log['timestamp_us'] - T0) * 1e-6
    log['gz_fused'] = np.where(np.abs(t - 2.0) < 0.005, np.nan, log['gz_fused'])
    m = hil(log, 'A-C2')
    assert math.isnan(m['e_yaw_imu']) and m['e_wz'] == pytest.approx(0.0, abs=1e-12)


def test_reset_on_the_line_at_t0_counts():
    """S2M1-8 (Codex round 1 of v3): the baseline of pos_reset_n is the last line before t = 0, so a reset on the
    line at t = 0 itself makes e_est undefined; a reset before t = 0 does not; no line before t = 0: unchecked."""
    log = mk()
    t = (log['timestamp_us'] - T0) * 1e-6
    assert np.any(t == 0.0)
    log['pos_reset_n'] = np.where(t >= 0.0, 2.0, 1.0)
    m = hil(log, plant=plant_of(log))
    assert m['e_est_reason'] == 'pos_reset_in_window' and math.isnan(m['e_est_p_rms'])
    log['pos_reset_n'] = np.where(t >= -0.01, 2.0, 1.0)                 # on the last line before t = 0
    m = hil(log, plant=plant_of(log))
    assert m['e_est_reason'] == '' and m['e_est_p_rms'] == pytest.approx(0.0, abs=1e-9)
    log['pos_reset_n'] = np.where(t >= 0.0, 2.0, 1.0)
    log['pos_reset_n'][np.flatnonzero(t < 0.0)[-1]] = np.nan          # S2M2-5: empty cell on the last line before 0
    m = hil(log, plant=plant_of(log))
    assert m['e_est_reason'] == 'pos_reset_in_window'
    log['pos_reset_n'] = np.where(t >= 0.0, 2.0, np.nan)                # no value before t = 0 at all
    m = hil(log, plant=plant_of(log))
    assert m['e_est_reason'] == 'pos_reset_unchecked'
    late = {c: (x[t >= 0.0] if not c.startswith('_') else x) for c, x in mk().items()}
    late['_all'] = late['_every'] = late
    m = hil(late, plant=plant_of(late))
    assert m['e_est_reason'] == 'pos_reset_unchecked'


def _runs_dir(tmp_path, platform, logs):
    """A run folder: one log per (run_id, log, plant), runs.csv of completed runs, plant and map files (HIL)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    lines = ['run_id,platform,controller,test_id,speed_mps,rep,file,aborted,block_status,result,reason,prep_s']
    for i, (rid, log) in enumerate(logs, start=1):
        _write_log(tmp_path / f'{rid}.csv', log)
        pl = plant_of(log)
        cols = ['timestamp_us', 'x', 'y', 'theta', 'vx_w', 'vy_w', 'vx_b', 'vy_b', 'wz', 'w1', 'w2', 'w3', 'w4']
        with open(tmp_path / f'{rid}_plant.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for k in range(len(pl['timestamp_us'])):
                w.writerow([int(pl['timestamp_us'][k])] + [f'{pl[c][k]:.9f}' if c in pl else '0' for c in cols[1:]])
        (tmp_path / f'{rid}_plant_map.json').write_text(json.dumps(dict(ok=True, sender_t0_us=T0)),
                                                        encoding='utf-8')
        lines.append(f'{rid},{platform},PID,A-L2,0.5,{i},{rid}.csv,,,completed,,{3.0}')
    (tmp_path / 'runs.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(tmp_path)


def test_loss_over_one_percent_or_a_reset_mid_run_is_broken_through_compare(tmp_path):
    """S2M1-1 (Codex round 1 of v3): a completed run with missing_over = 1 (P2-5) or run_bad = 1 (pose reset
    mid-run, L37) is status broken in metrics.csv, with no metric and its loss / reset columns; s02_compare then
    leaves it out (runs.csv -> metrics.csv -> compare)."""
    import s02_compare as C
    n = len(mk()['tick'])
    drop = np.zeros(n, bool)
    drop[np.arange(100, n, 40)] = True                                 # 2.5 % lost
    lossy = mk(drop=drop)
    bad = mk()
    t = (bad['timestamp_us'] - T0) * 1e-6
    bad['pos_reset_n'] = np.where(t > 3.0, 2.0, 1.0)                     # reset mid-run
    good = [mk(seed=s, delay_s=0.01 * s) for s in (1, 2, 3)]
    h = _runs_dir(tmp_path / 'h', 'hil', [('g1', good[0]), ('g2', good[1]), ('g3', good[2]),
                                           ('lossy', lossy), ('bad', bad)])
    out = M.run_dir_metrics(h)
    st = {o['run_id']: o for o in out}
    assert [st[r]['status'] for r in ('g1', 'g2', 'g3')] == ['ok'] * 3
    assert st['lossy']['status'] == 'broken' and st['lossy']['status_reason'].startswith('missing_over')
    assert st['lossy']['missing_over'] == 1 and st['lossy']['missing_pct'] > 1.0 and 'e_vx' not in st['lossy']
    assert all(k in st['lossy'] for k in M.BROKEN_KEEP)                 # loss and reset columns kept (S2M2-T)
    assert st['bad']['status'] == 'broken' and 'mid-run' in st['bad']['status_reason']
    assert st['bad']['run_bad'] == 1 and 'e_vx' not in st['bad'] and 'e_p_rms' not in st['bad']
    with open(os.path.join(h, 'metrics.csv'), newline='') as f:
        rows = {r['run_id']: r for r in csv.DictReader(f)}
    assert rows['lossy']['e_vx'] == '' and rows['lossy']['missing_pct'] != '' and rows['bad']['e_p_rms'] == ''
    assert M.cell_counts(out)[('A-L2', 'hil', 'PID')] == dict(ok=3, broken=2)
    s = _runs_dir(tmp_path / 's', 'sim', [(f's{i}', mk(seed=i)) for i in (1, 2, 3)])
    M.run_dir_metrics(s)
    rows, info = C.load([s, h])
    assert sorted(r['run_id'] for r in rows if r['platform'] == 'hil') == ['g1', 'g2', 'g3']
    x = next(x for x in C.compare(rows, info) if x['metric'] == 'e_vx' and x['platform_b'] == 'hil')
    assert x['n_b'] == 3


def test_command_line_prints_the_cell_line(tmp_path, monkeypatch, capsys):
    """S2M2-T (Codex round 2 of v3): the command line prints cell_line (ok / 5, broken apart)."""
    import runpy
    n = len(mk()['tick'])
    drop = np.zeros(n, bool)
    drop[np.arange(100, n, 40)] = True
    h = _runs_dir(tmp_path / 'h', 'hil', [('g1', mk()), ('lossy', mk(drop=drop))])
    monkeypatch.setattr(sys, 'argv', ['s02_metrics.py', h, 'm.csv'])
    runpy.run_path(M.__file__, run_name='__main__')
    assert '  A-L2 hil PID: ok 1 / 5, broken 1' in capsys.readouterr().out


def test_cell_line_reports_ok_over_five_and_history_apart():
    """S2M1-9 (Codex round 1 of v3): ok / 5 (P2-4), broken and aborted in the cell, replaced and battery_rerun as
    history, not in the denominator."""
    c = dict(ok=3, broken=1, aborted=1, replaced=2, battery_rerun=6)
    assert M.cell_line(('A-L2', 'hil', 'PID'), c) == \
        '  A-L2 hil PID: ok 3 / 5, aborted 1, broken 1; history: replaced 2, battery_rerun 6'
    assert M.cell_line(('A-K1', 'real', 'LADRC'), dict(ok=5)) == '  A-K1 real LADRC: ok 5 / 5'
