"""Tests of fwsim/s03_doe.py (spec S03 v6) and o1_feasibility.evaluate_v7 (O1 v7 section 5.9,
S03 v6 section 4). Synthetic logs and responses with a known answer; the last tests need
fwsim/build/fwsim.dll (firmware with pos_ctrl.c). Run: python -m pytest fwsim/tests -q"""
import math
import struct
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim  # noqa: E402
import o1_feasibility as o1  # noqa: E402
import s03_doe as s3  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')


@pytest.fixture(scope='module')
def path():
    return o1.Path8()


def _ideal(sch, dt=1e-4):
    """Ideal car from the origin (heading 45 deg on the floor), plant frame, 100 Hz log."""
    n = int(round((sch.Tcmd + o1.T_LATE) / dt))
    x = y = th = 0.0
    t_, pose, vel = [], [], []
    for k in range(n + 1):
        t = k * dt
        vx, vy, wz = sch.setpoint(t)
        if k % 100 == 0 and k:
            t_.append(t)
            pose.append((x, y, th))
            vel.append((vx, vy, wz))
        x += vx * math.cos(th) * dt
        y += vx * math.sin(th) * dt
        th += wz * dt
    m = len(t_)
    return dict(t=np.array(t_), pose=np.array(pose), v_true=np.array(vel), pwm=np.zeros((m, 4)),
                duty=np.zeros((m, 4)))


def _plant_xy(fx, fy):
    """Floor point -> plant frame (inverse of the 45 deg start rotation of evaluate)."""
    c, s = math.cos(o1.HEAD0), math.sin(o1.HEAD0)
    return c * fx + s * fy, -s * fx + c * fy


class _Later:
    """A schedule whose T_cmd is later than the motion: tests condition 5 (t >= T_cmd)."""

    def __init__(self, sch, extra):
        self.__dict__.update(sch.__dict__)
        self.Tcmd = sch.Tcmd + extra
        self.tau = sch.tau


# ── evaluate_v7 ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('v', [0.2, 0.8])
def test_v7_ideal_car(path, v):
    sch = o1.Schedule(v)
    res, rows = o1.evaluate_v7(_ideal(sch), path, sch)
    assert res['completed'] == 'true' and res['status'] == 'ok' and res['track_err_frames'] == 0
    assert res['e_perp_max'] < 0.002 and res['e_trk_max'] < 0.01 and res['e_theta_cmd_max_deg'] < 0.5
    assert res['t_end'] == pytest.approx(sch.Tcmd + o1.T_D, abs=0.02)
    assert res['e_perp_global_max'] <= res['e_perp_max'] + 1e-9


def test_v7_condition5_waits_for_the_command_end(path):
    """The car is still at the origin 3 s before T_cmd: v5 ended there, v7 waits for T_cmd (O6-3)."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    later = _Later(sch, 3.0)
    log = {k: np.concatenate([v, np.repeat(v[-1:], 300, axis=0)]) for k, v in log.items() if k != 't'} | \
        dict(t=np.concatenate([log['t'], log['t'][-1] + 0.01 * np.arange(1, 301)]))
    res, _ = o1.evaluate_v7(log, path, later)
    assert res['completed'] == 'true'
    assert res['t_end'] == pytest.approx(later.Tcmd, abs=0.02)
    old, _ = o1.evaluate(log, path, later)
    assert old['t_end'] < later.Tcmd - 1.0


def test_v7_condition2_holds_over_the_whole_still_window(path):
    """L63 (2) (Liam 07/11, 'Suot cua so t_d'): the car is still from T_cmd but outside D of the
    origin for the first 0.5 s, then at the origin: the t_d window starts when it is near the
    origin, so t_end = T_cmd + 0.5 + t_d (before: T_cmd + t_d, condition 2 at the instant only)."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    off = (log['t'] >= sch.Tcmd - 1e-9) & (log['t'] < sch.Tcmd + 0.5 - 1e-9)
    nx, ny = -math.sqrt(0.5) * 0.15, math.sqrt(0.5) * 0.15                    # 0.15 m > D, across the lane
    px, py = _plant_xy(nx, ny)
    log['pose'][off, 0] += px
    log['pose'][off, 1] += py
    res, rows = o1.evaluate_v7(log, path, sch)
    assert max(r['dist_origin'] for r in rows if sch.Tcmd <= r['t'] < sch.Tcmd + 0.5) > o1.D
    assert res['completed'] == 'true' and res['status'] == 'ok'
    assert res['t_end'] == pytest.approx(sch.Tcmd + 0.5 + o1.T_D, abs=0.02)


def test_v7_one_frame_off_the_origin_restarts_the_window(path):
    """EV7-3: still at the origin from T_cmd, one frame 0.15 m off at T_cmd + 0.5 s: the chain
    restarts after that frame, t_end = (T_cmd + 0.5 + one frame) + t_d."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    k = int(np.searchsorted(log['t'], sch.Tcmd + 0.5 - 1e-9))
    px, py = _plant_xy(-math.sqrt(0.5) * 0.15, math.sqrt(0.5) * 0.15)
    log['pose'][k, 0] += px
    log['pose'][k, 1] += py
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['completed'] == 'true' and res['status'] == 'ok'
    # exact frame (EV7-3, Codex round 2: abs=0.015 > one frame let a one-frame-early end pass):
    # the first frame at least t_d after the restart frame k + 1
    t = log['t']
    j = int(np.argmax(t - t[k + 1] >= o1.T_D - 1e-9))
    assert t[j] - t[k + 1] >= o1.T_D - 1e-9 > t[j - 1] - t[k + 1]
    assert res['t_end'] == pytest.approx(t[j], abs=1e-9)


def test_v7_track_error_and_short_stays_at_origin_is_false(path):
    """EV7-3: tracking error before the progress; after T_cmd the car is still but leaves the
    D disc for 0.1 s every 0.6 s, so no still-and-near window lasts t_d: conditions 2-5 never
    hold together -> 'false', cond2_5_ever False, the window runs to the deadline."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    k = int(np.searchsorted(log['t'], 4.0))
    s_k = path.s_of_tau(sch.tau(log['t'][k])[0])
    j = int(np.searchsorted(path.s, s_k + 1.0))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    px, py = _plant_xy(-math.sqrt(0.5) * 0.15, math.sqrt(0.5) * 0.15)
    off = (log['t'] >= sch.Tcmd - 1e-9) & (np.mod(log['t'] - sch.Tcmd + 1e-9, 0.6) >= 0.5)
    log['pose'][off, 0] += px
    log['pose'][off, 1] += py
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['status'] == 'track_err' and res['completed'] == 'false' and not res['cond2_5_ever']
    assert res['t_end'] == pytest.approx(log['t'][log['t'] < sch.Tcmd + o1.T_LATE - 1e-9][-1])


def test_v7_turning_in_place_is_not_still(path):
    """Condition 3 also on |wz| < omega_d = v_d / (lx + ly) (O6-18)."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    log['v_true'][log['t'] > sch.Tcmd, 2] = 0.1
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['completed'] == 'false' and not res['cond2_5_ever']


def test_v7_track_error_before_progress_is_inconclusive(path):
    """One frame jumps 1 m ahead mid-run (upper end of the window): progress not yet reached,
    conditions 2-5 hold at the end -> inconclusive, status track_err (S03 4, S3-V-1)."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    k = int(np.searchsorted(log['t'], 4.0))
    s_k = path.s_of_tau(sch.tau(log['t'][k])[0])
    j = int(np.searchsorted(path.s, s_k + 1.0))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['status'] == 'track_err' and res['completed'] == 'inconclusive'
    assert res['t_first_track_err'] == pytest.approx(log['t'][k])


def test_v7_track_error_after_progress_keeps_completion(path):
    """A frame 0.5 m back on the lane after progress reached S - D_d (B8b-1): still 'true'."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    k = int(np.searchsorted(log['t'], sch.Tcmd + 0.3))
    j = int(np.searchsorted(path.s, path.S - 0.5))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['status'] == 'track_err' and res['completed'] == 'true'
    assert res['t_progress'] < res['t_first_track_err']


def test_v7_track_error_and_never_at_origin_is_false(path):
    """Car parked 0.5 m from the origin, one jump frame: conditions 2-5 never hold -> 'false'."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    log['pose'][:] = (0.5, 0.0, 0.0)
    log['v_true'][:] = 0.0
    k = 100
    log['pose'][k, :2] = _plant_xy(path.x[200000], path.y[200000])
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['status'] == 'track_err' and res['completed'] == 'false'


def test_v7_lag_sign_and_window_of_y3_y4(path):
    """e_s > 0 when the car is behind the command (O1 v7 S2); Y3, Y4 only on [0, t_end]."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    after = log['t'] > sch.Tcmd + o1.T_D + 0.05
    log['duty'][after, 0] = np.where(np.arange(after.sum()) % 2, 0.5, -0.5)   # duty chatter after t_end
    log['pwm'][after, 1] = 1000                     # saturated only after t_end (S3C-22: must not count)
    half = log['t'] <= sch.Tcmd / 2
    log['pwm'][half, 2] = -1000                     # saturated over the first half of the window
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['tv_duty'] == 0.0
    n_win = int(np.sum(log['t'] <= res['t_end'] + 1e-9))
    assert res['sat_frac'] == pytest.approx(half.sum() / (4 * n_win))
    assert res['sat_any_frac'] == pytest.approx(half.sum() / n_win)
    whole = (np.abs(log['pwm']) >= 1000).mean()
    assert res['sat_frac'] != pytest.approx(whole)  # the whole-log value would differ
    still = _ideal(sch)
    still['pose'][:] = 0.0
    still['v_true'][:] = 0.0
    _, rows = o1.evaluate_v7(still, path, sch)
    assert rows[100]['e_s'] > 0 and rows[100]['s_cmd'] - rows[100]['s_car'] > 0


def test_v7_censored_and_measurement_error(path):
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    log['pose'][:, 0] += 0.7                     # whole run 0.7 m off: censored at 0.5 m
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['e_perp_max'] > 0.5 and res['e_perp_max_cens'] == o1.C_CENS
    assert math.isnan(res['e_meas_perp_max'])    # no firmware pose columns


def _with_fw_pose(log, shift_left=0.0, reset_at=None):
    """Add fwsim POS_COLS: firmware pose = true pose moved shift_left to the right of the heading
    (so true - firmware = shift_left to the LEFT, on the lane normal), run_active 1."""
    P = {n: i for i, n in enumerate(fwsim.POS_COLS)}
    pos = np.zeros((len(log['t']), len(fwsim.POS_COLS)))
    th = log['pose'][:, 2]
    pos[:, P['pos_x']] = log['pose'][:, 0] + shift_left * np.sin(th)
    pos[:, P['pos_y']] = log['pose'][:, 1] - shift_left * np.cos(th)
    pos[:, P['run_active']] = 1
    pos[:, P['pos_reset_n']] = 1
    if reset_at is not None:
        pos[reset_at:, P['pos_reset_n']] = 2
    log['pos'] = pos
    return log


def test_v7_measurement_error_on_the_lane_normal(path):
    """S3C-22: a known 3 cm offset between true and firmware pose along the lane normal gives
    e_meas_perp_max 0.03 m; a second pose reset makes it NaN (O6-11, S3C-19)."""
    sch = o1.Schedule(0.5)
    res, _ = o1.evaluate_v7(_with_fw_pose(_ideal(sch), 0.03), path, sch)
    assert res['e_meas_perp_max'] == pytest.approx(0.03, abs=2e-3) and res['e_meas_note'] == ''
    res, _ = o1.evaluate_v7(_with_fw_pose(_ideal(sch), 0.03, reset_at=500), path, sch)
    assert math.isnan(res['e_meas_perp_max']) and 'O6-11' in res['e_meas_note']


def test_v7_pose_change_after_t_end_keeps_e_meas(path):
    """Codex S3D-1: run_active low or a second reset AFTER the run ended (t_end, before the
    deadline) is outside the run: e_meas_perp_max stays a number; inside the run it is NaN."""
    sch = o1.Schedule(0.5)
    base, _ = o1.evaluate_v7(_with_fw_pose(_ideal(sch), 0.03), path, sch)
    assert base['completed'] == 'true' and base['t_end'] < sch.Tcmd + o1.T_LATE - 0.2
    P = {n: i for i, n in enumerate(fwsim.POS_COLS)}
    for col, val in (('run_active', 0), ('pos_reset_n', 2)):
        log = _with_fw_pose(_ideal(sch), 0.03)
        log['pos'][log['t'] > base['t_end'] + 0.1, P[col]] = val
        res, _ = o1.evaluate_v7(log, path, sch)
        assert res['t_end'] == base['t_end'] and res['e_meas_note'] == ''
        assert res['e_meas_perp_max'] == pytest.approx(base['e_meas_perp_max'])
        log = _with_fw_pose(_ideal(sch), 0.03)
        log['pos'][log['t'] > base['t_end'] - 1.0, P[col]] = val
        res, _ = o1.evaluate_v7(log, path, sch)
        assert math.isnan(res['e_meas_perp_max']) and 'O6-11' in res['e_meas_note']


def test_v7_untrusted_progress_keeps_the_window_open(path):
    """S3C-4: tracking error before the progress -> y1 inconclusive and the window runs to the
    last frame before the deadline, so duty chatter after the stop is counted."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    k = int(np.searchsorted(log['t'], 4.0))
    s_k = path.s_of_tau(sch.tau(log['t'][k])[0])
    j = int(np.searchsorted(path.s, s_k + 1.0))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    late = log['t'] > sch.Tcmd + o1.T_D + 0.2
    log['duty'][late, 0] = np.where(np.arange(late.sum()) % 2, 0.5, -0.5)
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['completed'] == 'inconclusive'
    deadline = sch.Tcmd + o1.T_LATE
    assert res['t_end'] > deadline - 0.02 and res['t_end'] < deadline
    assert res['tv_duty'] > 0


def _still_from(sch, t_still):
    """Ideal log, the car reported moving (0.1 m/s) before t_still, plus a frame exactly at
    t_still and one exactly at the deadline (log in time order): the car is still from t_still
    on, so conditions 2-5 first hold together at t_still + t_d (S3C-23 round 3)."""
    log = _ideal(sch)
    deadline = sch.Tcmd + o1.T_LATE
    t = np.concatenate([log['t'], [t_still, deadline]])
    order = np.argsort(t, kind='stable')
    src = np.concatenate([np.arange(len(log['t'])), [len(log['t']) - 1, len(log['t']) - 1]])[order]
    log['t'] = t[order]
    for key in ('pose', 'v_true', 'pwm', 'duty'):
        log[key] = np.asarray(log[key])[src]
    log['v_true'] = log['v_true'].copy()
    log['v_true'][log['t'] < t_still - 1e-12, 0] = 0.1
    return log, deadline


def test_v7_deadline_is_strict(path):
    """S3C-23 (test of round 2): still for exactly t_d AT the deadline frame -> not completed;
    still for t_d one frame (10 ms) before the deadline -> completed. Both on the same log
    except where the stillness starts."""
    sch = o1.Schedule(0.8)
    log, deadline = _still_from(sch, sch.Tcmd + o1.T_LATE - o1.T_D)
    k = int(np.flatnonzero(log['t'] == deadline)[0])
    still = log['t'][np.flatnonzero(np.hypot(log['v_true'][:, 0], log['v_true'][:, 1]) < o1.V_D)[0]]
    assert log['t'][k] - still == pytest.approx(o1.T_D, abs=1e-12)    # still exactly t_d at the deadline frame
    res, rows = o1.evaluate_v7(log, path, sch)
    assert all(r['t'] < deadline for r in rows)
    assert res['completed'] == 'false'
    log, _ = _still_from(sch, sch.Tcmd + o1.T_LATE - o1.T_D - 0.02)
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['completed'] == 'true' and res['t_end'] < deadline


# ── S03 pieces ───────────────────────────────────────────────────────────────────────

def test_sweep_grid_rules():
    g = s3.sweep_grid('kp_xy', 1.5, 0.0, 6.0)               # a = 0.15, b = 15, plus 0
    assert g[0] == 0.0 and 1.5 in g and g == sorted(g)
    assert min(x for x in g if x > 0) == pytest.approx(0.15) and max(g) == pytest.approx(15.0)
    assert len(g) == 12                                     # 11 log points (1.5 is one of them) + 0
    g = s3.sweep_grid('omega_o_xy', 30.0, 5.0, 48.0)        # a = min(3, 5), b = min(max(300, 48), 48)
    assert 0.0 not in g and 30.0 in g and max(g) == 48.0 and g[0] == pytest.approx(3.0) and len(g) == 12
    g = s3.sweep_grid('omega_c_xy', 8.0, 1.0, 25.0)         # a = min(0.8, 1), b = max(80, 25)
    assert g[0] == pytest.approx(0.8) and max(g) == pytest.approx(80.0) and 8.0 in g


def test_level_rule():
    grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert s3.levels_from_sweep(grid, 2.0, [False, True, True, True, False, True]) == (1.0, 3.0)
    assert s3.levels_from_sweep(grid, 2.0, [True] * 6) == (0.0, 5.0)
    assert s3.levels_from_sweep(grid, 2.0, [True, False, True, False, True, True]) == (2.0, 2.0)  # L = H
    with pytest.raises(s3.S03Stop, match=r'3\.1 \(i\)'):
        s3.levels_from_sweep(grid, 2.0, [True, True, False, True, True, True])


def test_scale_and_decode():
    assert s3.scale_of(1.0, 4.0) == ('log', 2.0)
    assert s3.scale_of(0.0, 3.0) == ('lin', 1.5) and s3.scale_of(1.0, 3.0) == ('lin', 2.0)
    for L, H in ((1.0, 16.0), (0.0, 3.0)):
        sc = s3.scale_of(L, H)[0]
        assert s3.decode(-1, L, H, sc) == pytest.approx(L) and s3.decode(1, L, H, sc) == pytest.approx(H)
        assert s3.decode(0, L, H, sc) == pytest.approx(s3.scale_of(L, H)[1])
    assert s3.decode(0.5, 1.0, 16.0, 'log') == pytest.approx(8.0)     # sqrt(16) * 4^0.5


def test_design_is_full_factorial():
    pts = s3.design_points()
    codes = np.array([c for _, c in pts])
    assert [p for p, _ in pts] == list(range(1, 129)) and len({tuple(c) for c in codes}) == 128
    assert np.all(codes.sum(axis=0) == 0)
    for i in range(7):
        for j in range(i + 1, 7):
            assert (codes[:, i] * codes[:, j]).sum() == 0
    assert codes[0, 0] == -1 and codes[1, 0] == 1 and codes[1, 1] == -1     # X1 changes fastest
    inner = np.array([c for _, c in s3.design_points(0.5, 129)])
    assert set(np.unique(inner)) == {-0.5, 0.5}


def test_effects_known_model():
    codes = np.array([c for _, c in s3.design_points()])
    x = codes
    y = 2.0 + 0.5 * x[:, 0] + 0.25 * x[:, 0] * x[:, 1] - 0.1 * x[:, 2] * x[:, 3] * x[:, 4]
    e = s3.effects(codes, y)
    assert len(e) == 127
    assert e['X1'] == pytest.approx(1.0) and e['X1:X2'] == pytest.approx(0.5)
    assert e['X3:X4:X5'] == pytest.approx(-0.2)
    assert all(abs(v) < 1e-12 for k, v in e.items() if k not in ('X1', 'X1:X2', 'X3:X4:X5'))
    half = codes * 0.5                                    # inner box: effect at its own width (S3-R-3)
    e2 = s3.effects(half, 2.0 + 0.5 * half[:, 0])
    assert e2['X1'] == pytest.approx(0.5)


def _runs(model, mu_levels=(0.4, 0.5, 0.6), speeds=(0.5,)):
    pts = s3.design_points()
    runs = {}
    for pid, c in [(0, (0.0,) * 7)] + pts:
        for mu in mu_levels:
            for v in speeds:
                runs[(pid, mu, v)] = model(np.array(c), mu)
    return pts, runs


def _row(e, tv=1.0, sat=0.0, completed='true', status='ok'):
    return dict(e_perp_max=e, e_perp_max_cens=min(e, 0.5), tv_duty=tv, sat_frac=sat, completed=completed,
                status=status)


def test_analyse_worst_range_and_thresholds():
    """Y2 = 0.05 + 0.01 X1 + 0.02 mu X2: effect X1 0.02 (significant at 0.01), X2 2*0.02*mu;
    worst over mu takes mu = 0.6; Y3 threshold 20 % of the reference."""
    def model(c, mu):
        return _row(0.05 + 0.01 * c[0] + 0.02 * mu * c[1], tv=1.0 + 0.1 * c[2],
                    completed='false' if c[3] > 0 and c[4] > 0 else 'true')
    pts, runs = _runs(model)
    eff, curv, ver, fails = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): 1.0})
    get = {(r['response'], r['mu'], r['term']): r for r in eff}
    assert get[('e_perp_max', '0.5', 'X1')]['effect'] == pytest.approx(0.02)
    assert get[('e_perp_max', '0.5', 'X1')]['significant']
    # worst: X2 = +1 is worst at mu 0.6 (+0.012), X2 = -1 at mu 0.4 (-0.008): effect 0.020
    assert get[('e_perp_max', 'worst', 'X2')]['effect'] == pytest.approx(0.012 + 0.008)
    # range over mu: 0.004 for both levels of X2, so its effect is 0
    assert get[('e_perp_max', 'range', 'X2')]['effect'] == pytest.approx(0.0, abs=1e-12)
    assert get[('e_perp_max', 'range', 'X3')]['effect'] == pytest.approx(0.0, abs=1e-12)
    r3 = get[('tv_duty', '0.5', 'X3')]
    assert r3['threshold'] == pytest.approx(0.2) and r3['effect'] == pytest.approx(0.2) and not r3['significant']
    assert all(v['main'] == 'as_measured' for v in ver)
    nc = {(f['factors'], f['levels'], f['mu']): f['count'] for f in fails if f['kind'] == 'not_completed'}
    assert nc[('X4:X5', '+,+', 0.5)] == 32 and nc[('X4:X5', '-,+', 0.5)] == 0 and nc[('X4', '+', 0.5)] == 32
    c = {(r['response'], r['mu']): r for r in curv}
    assert c[('e_perp_max', '0.5')]['diff'] == pytest.approx(0.0)


def test_analyse_censoring_changes_the_main_version():
    """A lost run (Y2 3 m) at one corner flips a significant set between Y2 and censored Y2."""
    def model(c, mu):
        big = c[0] > 0 and c[1] > 0 and c[2] > 0 and c[3] > 0 and c[4] > 0 and c[5] > 0 and c[6] > 0
        return _row(3.0 if big else 0.05, completed='false' if big else 'true')
    pts, runs = _runs(model, mu_levels=(0.5,))
    _, _, ver, _ = s3.analyse('ADRC', 'outer', pts, runs, [0.5], [0.5], {('ADRC', 0.5): 1.0})
    assert any(v['main'] == 'both' for v in ver)


def test_analyse_counts_track_errors_and_inconclusive():
    def model(c, mu):
        if c[0] > 0 and c[1] > 0:
            return _row(0.05, completed='inconclusive', status='track_err')
        return _row(0.05)
    pts, runs = _runs(model, mu_levels=(0.5,))
    eff, _, _, fails = s3.analyse('PID', 'outer', pts, runs, [0.5], [0.5], {('PID', 0.5): 1.0})
    r = next(x for x in eff if x['mu'] == '0.5' and x['term'] == 'X1')
    assert r['n_track_err'] == 32 and r['n_inconclusive'] == 32
    inc = {(f['factors'], f['levels']): f['count'] for f in fails if f['kind'] == 'inconclusive'}
    assert inc[('X1:X2', '+,+')] == 32


def test_worst_cell_counts_are_points():
    """S3C-13 round 3: at 'worst' and 'range', n_inconclusive counts points whose worst Y1 is
    inconclusive (none false), n_track_err points with a track error at some mu."""
    def model(c, mu):
        if c[0] > 0:        # false at 0.4, inconclusive + track error at 0.5 and 0.6
            return _row(0.05, completed='false' if mu == 0.4 else 'inconclusive',
                        status='ok' if mu == 0.4 else 'track_err')
        return _row(0.05, completed='inconclusive' if mu == 0.6 else 'true',
                    status='track_err' if mu == 0.6 else 'ok')
    pts, runs = _runs(model)
    eff, _, _, _ = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): 1.0})
    for cond in ('worst', 'range'):
        r = next(x for x in eff if x['mu'] == cond and x['term'] == 'X1')
        assert r['n_inconclusive'] == 64 and r['n_track_err'] == 128
    r = next(x for x in eff if x['mu'] == '0.5' and x['term'] == 'X1')
    assert r['n_inconclusive'] == 64 and r['n_track_err'] == 64


def test_log_digest_detects_one_bit():
    a = dict(x=np.array([1.0, 2.0]))
    b = dict(x=np.array([1.0, np.nextafter(2.0, 3.0)]))
    assert s3.log_digest(a) == s3.log_digest(dict(x=np.array([1.0, 2.0])))
    assert s3.log_digest(a) != s3.log_digest(b)


# ── with the DLL ─────────────────────────────────────────────────────────────────────

needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')


@needs_dll
@pytest.mark.parametrize('ctrl', ['PID', 'ADRC'])
def test_defaults_and_gain_setting(ctrl):
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    d = s3.defaults(fw, ctrl)
    want = (dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05) if ctrl == 'PID' else
            dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0))
    want.update(pos_kp=6.0, pos_ki=3.0, k_psi=1.0)          # S03 v6 section 2.1 table
    assert d == pytest.approx(want)
    kw = s3.sim_kwargs(fw, ctrl, dict(d, **{s3.FACTORS[ctrl][2]: 2.5}))
    inner = kw['pid'] if ctrl == 'PID' else kw['adrc']
    assert inner[2][0] == 2.5 and inner[0][0] == pytest.approx(want[s3.FACTORS[ctrl][0]])
    assert kw['fw_params'] == pytest.approx(dict(pos_kp=6.0, pos_ki=3.0, k_psi=1.0))


@needs_dll
def test_run_one_and_new_process_rerun_are_identical(path, tmp_path):
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    g = s3.defaults(fw, 'PID')
    row, log = s3.run_one(fw, path, 'PID', g, 0.8, 0.5)
    assert set(row) >= {'status', 'completed', 'e_perp_max', 'tv_duty', 'sat_frac', 'pos_reset_mid'}
    assert row['pos_reset_mid'] == 0
    again = s3.run_in_new_process(_DLL, [('PID', g, 0.8, 0.5)], str(tmp_path))[0]
    assert again['digest'] == s3.log_digest(log) and s3.row_text(again['row']) == s3.row_text(row)


@needs_dll
def test_gains_reach_the_firmware_and_change_the_run(path):
    """S3C-21: seven non-default gains are what the firmware holds after the run, the run differs
    from the default one, and default - custom - default gives the first run back bit for bit."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    for ctrl in ('PID', 'ADRC'):
        d = s3.defaults(fw, ctrl)
        g = {n: (x * (1.3 + 0.2 * k) if x else 0.3 + 0.1 * k) for k, (n, x) in enumerate(d.items())}
        assert len({round(x, 9) for x in g.values()}) == len(g)      # S3C-21 round 3: all different
        r0, l0 = s3.run_one(fw, path, ctrl, d, 0.8, 0.5)
        r1, l1 = s3.run_one(fw, path, ctrl, g, 0.8, 0.5)
        f = s3.FACTORS[ctrl]
        if ctrl == 'PID':
            got = [fw.get_pid(0)['kp'], fw.get_pid(1)['ki'], fw.get_pid(2)['kp'], fw.get_pid(2)['ki']]
            assert fw.get_pid(0)['kd'] == 0.0
        else:
            got = [fw.get_adrc(0)['omega_c'], fw.get_adrc(1)['omega_o'], fw.get_adrc(2)['omega_c'],
                   fw.get_adrc(2)['omega_o']]
        assert got == pytest.approx([g[n] for n in f[:4]], rel=1e-6)
        # S3C-21 round 2: x and y share the xy gains; fixed parameters stay as after reset
        fw.reset()
        fw.set_mode(ctrl)
        base = [fw.get_pid(i) for i in range(3)] if ctrl == 'PID' else [fw.get_adrc(i) for i in range(3)]
        s3.run_one(fw, path, ctrl, g, 0.8, 0.5)
        now = [fw.get_pid(i) for i in range(3)] if ctrl == 'PID' else [fw.get_adrc(i) for i in range(3)]
        keys = (('kp', 'ki'), ('int_limit', 'out_limit', 'kd')) if ctrl == 'PID' else \
            (('omega_c', 'omega_o'), ('b0', 'out_limit'))
        for i in range(3):
            want = (g[f[0]], g[f[1]]) if i < 2 else (g[f[2]], g[f[3]])
            assert [now[i][k] for k in keys[0]] == pytest.approx(want, rel=1e-6)
            assert [now[i][k] for k in keys[1]] == [base[i][k] for k in keys[1]]
        pm = fwsim.pos_meta(fw)
        assert pm['kp'] == pytest.approx(g['pos_kp'], rel=1e-6) and pm['ki'] == pytest.approx(g['pos_ki'], rel=1e-6)
        assert s3.log_digest(l1) != s3.log_digest(l0)
        r2, l2 = s3.run_one(fw, path, ctrl, d, 0.8, 0.5)
        assert s3.log_digest(l2) == s3.log_digest(l0)
        g_psi = dict(d, k_psi=3.0)                       # k_psi alone changes the run (it is applied)
        _, l3 = s3.run_one(fw, path, ctrl, g_psi, 0.8, 0.5)
        assert s3.log_digest(l3) != s3.log_digest(l0)


FAKE_DEFAULTS = dict(PID=dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05, pos_kp=6.0, pos_ki=3.0, k_psi=1.0),
                     ADRC=dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0, pos_kp=6.0,
                               pos_ki=3.0, k_psi=1.0))


class _Calls(list):
    """The calls list of _fake, with what _run_main saw on the side."""


def _fake(design_fail=False, default_fails=False, error_in_design=False):
    """Instant stand-in for safe_run: sweep runs (one factor off its default) complete; design
    runs (several factors off) fail when X1 or X2 is above its default (96/128 corners)."""
    calls, works = _Calls(), []

    def fake_safe_run(fw, path, ctrl, g, v, mu):
        f = s3.FACTORS[ctrl]
        dflt = FAKE_DEFAULTS[ctrl]
        off = [n for n in f if not math.isclose(g[n], dflt[n])]
        calls.append(len(off))
        design = len(off) >= 2
        if error_in_design and design:
            return dict(status='run_error', completed='', error='FakeError: boom'), None
        fail = (design_fail and design and (g[f[0]] > dflt[f[0]] or g[f[1]] > dflt[f[1]])) or \
               (default_fails and not off and v == 0.8)
        x = np.array([g[n] for n in f])
        log = dict(t=np.arange(3.0) * 0.01, duty=np.zeros((3, 4)) + x.sum())
        return dict(status='ok', completed='false' if fail else 'true', e_perp_max=0.01 + 1e-3 * float(x[0]),
                    e_perp_max_cens=0.01, e_perp_global_max=0.01, tv_duty=np.float64(1.0 + float(x[1])),
                    sat_frac=0.0, sat_any_frac=0.0, e_theta_lane_max_deg=1.0, e_theta_cmd_max_deg=1.0,
                    ds_absmax=0.0, e_trk_max=0.0, t_end=np.float64(1.25), e_s_absmax=0.0, track_err_frames=0,
                    cond2_5_ever=np.bool_(True), pos_reset_mid=0, e_perp_rms=0.0, e_trk_rms=0.0), log

    def fake_new_process(dll, runs, work):
        works.append(work)
        out = []
        for j in runs:
            r, l = fake_safe_run(None, None, *j)
            out.append(dict(row=r, digest=s3.log_digest(l) if l is not None else None))
        return out
    return fake_safe_run, fake_new_process, calls, works


def _public_text(out):
    """Every file outside sealed/ as text (npz skipped)."""
    txt = {}
    for root, _, files in os.walk(out):
        rel = os.path.relpath(root, out)
        if rel.split(os.sep)[0] == 'sealed':
            continue
        for f in files:
            if not f.endswith('.npz'):
                txt[os.path.normpath(os.path.join(rel, f))] = open(os.path.join(root, f), encoding='utf-8').read()
    return txt


def _run_main(tmp_path, monkeypatch, capsys, **kw):
    fsr, fnp, calls, works = _fake(**kw)
    out = tmp_path / 'run'
    seen = dict(design_before_first_main_run=None, analyse=[])

    def fsr_watch(fw, path, ctrl, g, v, mu):
        dflt = FAKE_DEFAULTS[ctrl]
        if seen['design_before_first_main_run'] is None and \
                sum(not math.isclose(g[n], dflt[n]) for n in s3.FACTORS[ctrl]) >= 2:
            f = out / 'design.csv'
            seen['design_before_first_main_run'] = f.read_text() if f.is_file() else ''
        return fsr(fw, path, ctrl, g, v, mu)
    real_analyse = getattr(s3.analyse, 'real', s3.analyse)        # not a watcher of an earlier call

    def analyse_watch(ctrl, box, *a, **k):
        seen['analyse'].append((ctrl, box))
        return real_analyse(ctrl, box, *a, **k)
    analyse_watch.real = real_analyse
    monkeypatch.setattr(s3, 'analyse', analyse_watch)
    calls.seen = seen
    monkeypatch.setattr(s3, 'safe_run', fsr_watch)
    monkeypatch.setattr(s3, 'run_in_new_process', fnp)
    monkeypatch.setattr(s3, 'defaults', lambda fw, ctrl: dict(FAKE_DEFAULTS[ctrl]))
    monkeypatch.setattr(sys, 'argv', ['s03_doe.py', '--out', str(out), '--mu', '0.5', '--trial'])
    code = s3.main()
    return code, out, capsys.readouterr().out, calls, works


@needs_dll
def test_main_pipeline_clean(tmp_path, monkeypatch, capsys):
    import csv as _csv
    code, out, console, _, works = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    for f in ('sweep.csv', 'sweep_counts.csv', 'levels.csv', 'design.csv', 'meta.yaml', 'sealed/runs.csv',
              'sealed/effects.csv', 'sealed/fail_counts.csv', 'sealed/notes.json', 'sealed/manifest.json',
              'sealed/determinism_check.csv'):
        assert (out / f).is_file(), f
    assert not (out / 'determinism_check.csv').exists() and not (out / '_work').exists()
    assert works and all(os.path.relpath(w, out).split(os.sep)[0] == 'sealed' for w in works)   # S3C-2
    runs = list(_csv.DictReader(open(out / 'sealed' / 'runs.csv')))
    assert len(runs) == 2 * (129 + 128) * 3                        # L41: the inner box always runs
    assert sum(r['box'] == 'inner' for r in runs) == 2 * 128 * 3
    design = list(_csv.DictReader(open(out / 'design.csv')))
    assert len(design) == 2 * 257 and {r['box'] for r in design} == {'centre', 'outer', 'inner'}
    effects = list(_csv.DictReader(open(out / 'sealed' / 'effects.csv')))
    assert effects and {r['box'] for r in effects} == {'outer'}     # rule off: inner box not analysed
    assert all(float(r['t_end']) == 1.25 for r in runs)            # S3C-14: plain numbers
    assert {r['cond2_5_ever'] for r in runs} == {'true'}
    sweep = list(_csv.DictReader(open(out / 'sweep.csv')))
    assert all(float(r['t_end']) == 1.25 and float(r['tv_duty']) > 0 for r in sweep)
    det = list(_csv.DictReader(open(out / 'sealed' / 'determinism_check.csv')))
    assert len(det) == 2 * 4 * 3 and all(r['log_arrays_equal'] == 'true' for r in det)
    assert len(list((out / 'sealed' / 'logs').glob('*.csv'))) == 2 * 4 * 3    # S03 3.4: log.csv of the 4 points
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'trial: true' in meta and 'determinism: "passed' in meta and 'equal' not in meta
    sha_line = next(l for l in meta.splitlines() if l.startswith('file_sha256:'))
    assert 'runs.csv' not in sha_line and 'notes.json' not in sha_line          # S3C-1
    assert 'inner' not in console.lower()


@needs_dll
def test_main_inner_box_stays_sealed(tmp_path, monkeypatch, capsys):
    """96/128 corners 'false' -> the rule of S03 3.2 is on and the inner box is analysed; no
    public file or console line differs from a run where the rule is off (S3C-1..S3C-3), except
    the date and the timing lines, and both runs make the same number of runs (L41: the run
    time does not depend on the rule)."""
    import csv as _csv
    import json as _json
    code, out, console, calls1, _ = _run_main(tmp_path / 'a', monkeypatch, capsys, design_fail=True)
    assert code == 0
    notes = _json.load(open(out / 'sealed' / 'notes.json'))
    assert notes['inner_rule_speeds'] == {'PID': [0.2, 0.5, 0.8], 'ADRC': [0.2, 0.5, 0.8]}
    effects = list(_csv.DictReader(open(out / 'sealed' / 'effects.csv')))
    nc = notes['no_conclusion_speeds']
    inner_eff = {(r['controller'], float(r['v'])) for r in effects if r['box'] == 'inner'}
    assert inner_eff == {(c, v) for c in ('PID', 'ADRC') for v in (0.2, 0.5, 0.8) if v not in nc[c]}
    assert inner_eff                                      # the rule is on: the inner box is analysed
    code2, out2, console2, calls2, _ = _run_main(tmp_path / 'b', monkeypatch, capsys)
    assert code2 == 0
    assert len(calls1) == len(calls2)
    assert calls1.seen['analyse'] == calls2.seen['analyse'] == [(c, b) for c in ('PID', 'ADRC')
                                                                for b in ('outer', 'inner')]   # S3E-1
    for c in (calls1, calls2):                          # S3E-2: design.csv on disk before the first run
        assert c.seen['design_before_first_main_run'].count('\n') == 1 + 2 * 257
    pub1, pub2 = _public_text(out), _public_text(out2)
    assert set(pub1) == set(pub2)
    for name in pub1:
        a_, b_ = pub1[name], pub2[name]
        if name == 'meta.yaml':
            keep = lambda t: [l for l in t.splitlines() if not l.startswith(('elapsed_s', 'date', 'script'))]  # noqa
            assert keep(a_) == keep(b_)
        else:
            assert a_ == b_, name
    assert console.replace(str(out), '') == console2.replace(str(out2), '')


@needs_dll
def test_main_stops_everything_when_a_default_fails(tmp_path, monkeypatch, capsys):
    """S3C-9: a default that does not complete at a gate speed stops S03 for both controllers
    before any sweep (3.1 (i))."""
    import csv as _csv
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, default_fails=True)
    assert code == 2 and '3.1 (i)' in console
    assert len(calls) == 3                                   # PID defaults at the 3 speeds, then stop
    assert (out / 'design.csv').read_text().strip() == ''
    sweep = list(_csv.DictReader(open(out / 'sweep.csv')))   # S3D-4: the default check is recorded
    assert [(r['factor'], r['v'], r['completed']) for r in sweep] == [
        ('(default check)', '0.2', 'true'), ('(default check)', '0.5', 'true'), ('(default check)', '0.8', 'false')]
    assert 'stopped in the sweep' in (out / 'meta.yaml').read_text(encoding='utf-8')


@needs_dll
def test_main_run_error_in_the_design_is_recorded_and_sealed(tmp_path, monkeypatch, capsys):
    """S3C-12, S3C-3: the run_error row is in sealed/runs.csv; the console and meta.yaml only say
    that S03 stopped in the main design."""
    import csv as _csv
    code, out, console, _, _ = _run_main(tmp_path, monkeypatch, capsys, error_in_design=True)
    assert code == 2
    assert 'FakeError' not in console and 'sealed/stop.json' in console
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'FakeError' not in meta and 'stopped in the main design' in meta
    rows = list(_csv.DictReader(open(out / 'sealed' / 'runs.csv')))
    assert rows[-1]['status'] == 'run_error' and 'FakeError' in rows[-1]['error']
    assert 'FakeError' in (out / 'sealed' / 'stop.json').read_text()
    design = list(_csv.DictReader(open(out / 'design.csv')))         # S3C-3 (c): written before the run
    assert {r['controller'] for r in design} == {'PID', 'ADRC'} and len(design) == 2 * 257


def test_gate_speeds_must_be_task_speeds(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['s03_doe.py', '--out', str(tmp_path / 'x'), '--mu', '0.5', '--trial',
                                      '--gate-speeds', '0.3'])
    with pytest.raises(SystemExit, match='S3C-10'):
        s3.main()


def test_scale_tolerance_on_the_real_grid():
    """S3C-6: L from np.geomspace (12.000000000000002) with H = 48 is a ratio of 4 -> log."""
    g = s3.sweep_grid('omega_o_wz', 30.0, 5.0, 48.0)
    L = min(x for x in g if math.isclose(x, 12.0))
    assert L != 12.0                                  # the rounding is really there
    sc, centre = s3.scale_of(L, 48.0)
    assert sc == 'log' and centre == pytest.approx(24.0)
    assert s3.scale_of(12.1, 48.0)[0] == 'lin'
    assert s3.scale_of(1.0, 3.999999998)[0] == 'lin'           # S3D-7: a ratio really below 4
    assert s3.scale_of(1.0, 3.999999999998)[0] == 'lin'        # S3D-7 round 3


def test_fmt_numpy_scalars():
    assert s3.fmt(np.float64(32.07)) == '32.07' and s3.fmt(np.bool_(True)) == 'true'
    assert s3.fmt(np.int64(3)) == 3 and s3.fmt(float('nan')) == 'nan'


def test_row_text_is_bit_exact_with_ids():
    r = dict(status='ok', e_perp_max=float('nan'))
    r2 = dict(status='ok', e_perp_max=struct.unpack('<d', bytes.fromhex('010000000000f87f'))[0])  # other NaN payload
    ids = ('PID', 3, 'outer', 0.5, 0.8)
    assert s3.row_text(r, ids) == s3.row_text(dict(r), ids)
    assert s3.row_text(r, ids) != s3.row_text(r2, ids)
    assert s3.row_text(r, ids) != s3.row_text(r, ('PID', 4, 'outer', 0.5, 0.8))
    assert set(s3.row_text(r, ids)) == set(s3.RUN_COLS)


def test_analyse_no_conclusion_speed_has_no_effects():
    """S3C-11: a speed where the inner box is also mostly not completed gets no effect rows."""
    pts, runs = _runs(lambda c, mu: _row(0.05), mu_levels=(0.5,))
    eff, curv, ver, fails = s3.analyse('PID', 'outer', pts, runs, [0.5], [0.5], {('PID', 0.5): 1.0},
                                       no_conclusion=[0.5])
    assert eff == [] and curv == [] and ver == [] and fails


def test_analyse_y3_zero_reference_and_worst_counts():
    """S3C-13: Y3 reference 0 -> note; fail counts over mu ('worst': any mu not completed)."""
    def model(c, mu):
        return _row(0.05, tv=0.1 * c[2], completed='false' if (c[0] > 0 and mu == 0.6) else 'true')
    pts, runs = _runs(model)
    eff, _, _, fails = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): 0.0})
    r = next(x for x in eff if x['response'] == 'tv_duty' and x['term'] == 'X3' and x['mu'] == '0.5')
    assert r['significant'] and 'reference run is 0' in r['note']
    w = {(f['factors'], f['levels']): f['count'] for f in fails if f['kind'] == 'not_completed' and f['mu'] == 'worst'}
    assert w[('X1', '+')] == 64 and w[('X1', '-')] == 0


def test_worst_y1_is_three_valued():
    """S3C-13 round 2: at 'worst', a point false at one mu and inconclusive at another counts as
    not completed only; inconclusive at one mu and true elsewhere counts as inconclusive."""
    def model(c, mu):
        if c[0] > 0:
            return _row(0.05, completed={0.4: 'false', 0.5: 'inconclusive'}.get(mu, 'true'))
        return _row(0.05, completed='inconclusive' if mu == 0.6 else 'true')
    pts, runs = _runs(model)
    _, _, _, fails = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): 1.0})
    w = {(f['kind'], f['levels']): f['count'] for f in fails
         if f['mu'] == 'worst' and f['factors'] == 'X1'}
    assert w[('not_completed', '+')] == 64 and w[('inconclusive', '+')] == 0
    assert w[('not_completed', '-')] == 0 and w[('inconclusive', '-')] == 64


def test_sweep_counts():
    rows = [dict(controller='PID', factor='kp_xy', v=0.5, mu=0.5, completed=c, sat_frac=s)
            for c, s in (('true', 0.0), ('false', 0.1), ('inconclusive', 0.0))]
    (c,) = s3.sweep_counts(rows)
    assert (c['n'], c['not_completed'], c['inconclusive'], c['saturated']) == (3, 1, 1, 1)


def test_analyse_writes_the_box_notes():
    pts, runs = _runs(lambda c, mu: _row(0.05), mu_levels=(0.5,))
    eff, _, _, _ = s3.analyse('PID', 'outer', pts, runs, [0.5], [0.5], {('PID', 0.5): 1.0},
                              {('outer', 0.5): 'most corners not completed'})
    assert all(r['note'] == 'most corners not completed' for r in eff)
    eff, _, _, _ = s3.analyse('PID', 'outer', pts, runs, [0.5], [0.5], {('PID', 0.5): 1.0})
    assert all(r['note'] == '' for r in eff)
