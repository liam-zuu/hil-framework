"""Tests of fwsim/s03_doe.py (spec S03 v6) and o1_feasibility.evaluate_v7 (O1 v7 section 5.9,
S03 v6 section 4). Synthetic logs and responses with a known answer; the last tests need
fwsim/build/fwsim.dll (firmware with pos_ctrl.c). Run: python -m pytest fwsim/tests -q"""
import itertools
import math
import struct
import os
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim  # noqa: E402
import o1_feasibility as o1  # noqa: E402
import s03_doe as s3  # noqa: E402
import s03_sangloc as sl  # noqa: E402
import s04_ga as s4  # noqa: E402
import make_s03_screen_u as mk  # noqa: E402

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


def _counts_ok(res, rows):
    """S3V8-7: the two counts against track_err_frames and the frames kept (t_k <= t_end)."""
    nu, nb, nt = res['n_upper_frames'], res['n_back_frames'], res['track_err_frames']
    assert nt == sum(r['track_err'] for r in rows)
    assert max(nu, nb) <= nt <= nu + nb
    assert nu <= len(rows) and nb <= len(rows) and rows[-1]['t'] == res['t_end']


def test_v7_diagnostic_counts(path):
    """S03 v8 section 4 (S3V8-7): ideal car 0 / 0; a frame 1 m ahead counts in n_upper (and the
    frame after it in n_back); a frame 0.5 m back after the progress counts in n_back only; a jump
    after t_end is not counted."""
    sch = o1.Schedule(0.5)
    res, rows = o1.evaluate_v7(_ideal(sch), path, sch)
    assert res['n_upper_frames'] == 0 and res['n_back_frames'] == 0
    _counts_ok(res, rows)
    log = _ideal(sch)                                             # upper end of the window
    k = int(np.searchsorted(log['t'], 4.0))
    j = int(np.searchsorted(path.s, path.s_of_tau(sch.tau(log['t'][k])[0]) + 1.0))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    res, rows = o1.evaluate_v7(log, path, sch)
    # known answer: the jump frame is on the upper end (s only grows, s_prev moves up by D_k), the
    # next frame, back on the path, is below s_prev - delta: one upper, one back, two frames
    assert (res['n_upper_frames'], res['n_back_frames'], res['track_err_frames']) == (1, 1, 2)
    assert [r['t'] for r in rows if r['track_err']] == [log['t'][k], log['t'][k + 1]]
    _counts_ok(res, rows)
    log = _ideal(sch)                                             # back on the lane after the progress
    k = int(np.searchsorted(log['t'], sch.Tcmd + 0.3))
    j = int(np.searchsorted(path.s, path.S - 0.5))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    res, rows = o1.evaluate_v7(log, path, sch)
    assert (res['n_upper_frames'], res['n_back_frames'], res['track_err_frames']) == (0, 1, 1)   # back only
    _counts_ok(res, rows)



def _extend(log, n=300):
    """Log held still n more frames (100 Hz) after its end."""
    out = {k: np.concatenate([v, np.repeat(v[-1:], n, axis=0)]) for k, v in log.items() if k != 't'}
    out['t'] = np.concatenate([log['t'], log['t'][-1] + 0.01 * np.arange(1, n + 1)])
    return out


def test_v7_counts_skip_a_flagged_frame_after_t_end(path):
    """S3V8-7 round 2 (Codex: path[200000] is the origin, no flag): a frame 0.5 m back on the lane
    (s = S - 0.5) 0.2 s after t_end is not counted; control: with T_cmd 1 s later the window is
    still open at that frame and the same frame counts in n_back."""
    sch = o1.Schedule(0.5)
    log = _extend(_ideal(sch))
    t_end0 = o1.evaluate_v7(log, path, sch)[0]['t_end']
    k = int(np.searchsorted(log['t'], t_end0 + 0.2))
    j = int(np.searchsorted(path.s, path.S - 0.5))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    res, rows = o1.evaluate_v7(log, path, sch)
    assert res['t_end'] == t_end0 < log['t'][k] < sch.Tcmd + o1.T_LATE
    assert (res['n_upper_frames'], res['n_back_frames'], res['track_err_frames']) == (0, 0, 0)
    _counts_ok(res, rows)
    later = _Later(sch, 1.5)              # control: T_cmd after frame k (t_end0 + 0.2 = T_cmd + t_d + 0.2)
    assert later.Tcmd > log['t'][k]
    res2, rows2 = o1.evaluate_v7(log, path, later)
    assert res2['t_end'] > log['t'][k]
    assert (res2['n_upper_frames'], res2['n_back_frames'], res2['track_err_frames']) == (0, 1, 1)
    _counts_ok(res2, rows2)


def test_v7_counts_skip_a_flagged_frame_at_the_deadline(path):
    """S3V8-7 round 2: a car that tracks the path but never completes (turns in place after T_cmd,
    condition 3 never holds) is scored up to the deadline with no flag; a jump frame at the first
    instant >= deadline is not counted, the same jump one frame earlier is."""
    sch = o1.Schedule(0.8)
    base = _extend(_ideal(sch), 50)
    base['v_true'][base['t'] > sch.Tcmd, 2] = 0.1
    res0, rows0 = o1.evaluate_v7(base, path, sch)
    assert res0['completed'] == 'false' and res0['track_err_frames'] == 0
    dl = sch.Tcmd + o1.T_LATE
    kd = int(np.searchsorted(base['t'], dl - 1e-9))               # first frame not before the deadline
    assert base['t'][kd] >= dl - 1e-9 > base['t'][kd - 1] and rows0[-1]['t'] == base['t'][kd - 1]
    j = int(np.searchsorted(path.s, path.S - 0.5))                # 0.5 m back on the lane: a back flag
    jump = _plant_xy(path.x[j], path.y[j])
    got = {}
    for k in (kd, kd - 1):
        log = {n: v.copy() for n, v in base.items()}
        log['pose'][k, :2] = jump
        res, rows = o1.evaluate_v7(log, path, sch)
        _counts_ok(res, rows)
        got[k] = (res['n_upper_frames'], res['n_back_frames'], res['track_err_frames'])
    assert got[kd] == (0, 0, 0)
    assert got[kd - 1][2] == 1 and got[kd - 1][0] + got[kd - 1][1] >= 1


class _BothFlags:
    """Path8 whose tracker reports, at one frame (call pair), the upper end AND a two-sided
    nearest point behind s_prev - delta: one frame counted in both n_upper and n_back."""

    def __init__(self, path, frame):
        self._p, self._frame, self._calls = path, frame, 0

    def __getattr__(self, name):
        return getattr(self._p, name)

    def nearest(self, x, y, lo, hi):
        out = self._p.nearest(x, y, lo, hi)
        k, second = divmod(self._calls, 2)
        self._calls += 1
        if k != self._frame:
            return out
        if not second:                                            # narrow window: upper end
            return (hi,) + tuple(out[1:3]) + (True,)
        return (lo - 1.0,) + tuple(out[1:])                       # two-sided: behind s_prev - delta


def test_v7_one_frame_in_both_counts(path):
    """S3V8-7 round 2: one frame with both flags -> n_upper 1, n_back 1, track_err_frames 1
    (max(n_upper, n_back) <= track_err_frames <= n_upper + n_back, both bounds reached apart)."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    res, rows = o1.evaluate_v7(log, _BothFlags(path, 200), sch)
    assert res['n_upper_frames'] >= 1 and res['n_back_frames'] >= 1
    assert rows[200]['track_err'] == 1
    _counts_ok(res, rows)
    first = [i for i, r in enumerate(rows) if r['track_err']][0]
    assert first == 200


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
    worst over mu takes mu = 0.6; Y3 threshold the absolute 0.4 s^-1 (S03 v8.6 section 5.2)."""
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
    # S03 v8.6: the Y3 threshold is the absolute 0.4 s^-1 (before: 20 % of the default-gain run)
    assert r3['threshold'] == pytest.approx(0.4) and r3['effect'] == pytest.approx(0.2) and not r3['significant']
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
    assert set(s3.GATE_RUN_COLS) <= set(row) and set(s3.GATE_RUN_COLS) <= set(s3.row_text(row))   # v8.2 (T6)


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


def _fake_gate_fields(ctrl, g, v):
    """S3G-3: v8.2 gate columns of a fake run, different per gains and speed (known from the inputs)."""
    x = float(sum(g[n] * (k + 1) for k, n in enumerate(s3.FACTORS[ctrl])))
    bad = v == 0.8
    f = dict(e_bao_max=1e-3 * x + v + 0.5, gate_version='gate_fake', gate_decide='g1,g2,g3,g4,g5', gate_u_t=0.0,
             g1_q='true', g1_state='dat', gate_outcome='khong_dat' if bad else 'dat')
    for i, c in enumerate(('g2', 'g3', 'g4', 'g5')):
        q = 1e-3 * x + v + 0.01 * (i + 1)
        f.update({f'{c}_q': q, f'{c}_lo': q - 1e-6, f'{c}_hi': q + 1e-6, f'{c}_n_undef': i,
                  f'{c}_state': 'khong_dat' if bad and c == 'g5' else 'dat'})
        for j, x_ in enumerate(('q', 'lo', 'hi', 'n_undef')):                 # G1F-2: the 24 B81-3 columns too
            f[f'{c}_{x_}_chung'] = (i + 10 * j) if x_ == 'n_undef' else q + 1e-4 * (j + 1)
    f.update(g1_ways=1, g1_t_deadline=40.0 + v, g1_t_done_1=20.0 + 1e-3 * x, g1_t_done_2=None, g1_t1=None if bad else 30.0,
             g1_prog_before_t1=None if bad else True, g1_cond25_before_deadline=None if bad else (x > 5.0),
             g1_cat_t1=None)
    return f


def _fake(design_fail=False, default_fails=False, error_in_design=False, error_in_sweep=None, row_over=None,
          kep=False, design_warn=False):
    """Instant stand-in for safe_run: sweep runs (one factor off its default) complete; design
    runs (several factors off) fail when X1 or X2 is above its default (96/128 corners).
    error_in_sweep (S3G-2): 'default' -> run_error at the first default check; 'factor' -> at every
    sweep run with one factor off. row_over (S03 v8.4): f(ctrl, g, v) -> dict of row fields to replace, or None.
    kep (S03 v8.6 2.1): the log carries an 'en' channel (motor on 2 of 3 ticks) and a 'kep' channel
    (KEP_COLS, vx flagged on the two motor-on ticks). design_warn: a RuntimeWarning in every
    main-design run (the sealed diag.txt channel of section 3.1b item 13)."""
    calls, works = _Calls(), []

    def fake_safe_run(fw, path, ctrl, g, v, mu):
        f = s3.FACTORS[ctrl]
        dflt = FAKE_DEFAULTS[ctrl]
        off = [n for n in f if not math.isclose(g[n], dflt[n])]
        calls.append(len(off))
        design = len(off) >= 2
        if design_warn and design:
            warnings.warn('FakeDesignWarning', RuntimeWarning)
        if error_in_design and design:
            return dict(status='run_error', completed='', error='FakeError: boom'), None
        if (error_in_sweep == 'default' and not off) or (error_in_sweep == 'factor' and len(off) == 1):
            return dict(status='run_error', completed='', error='FakeSweepError: boom'), None
        fail = (design_fail and design and (g[f[0]] > dflt[f[0]] or g[f[1]] > dflt[f[1]])) or \
               (default_fails and not off and v == 0.8)
        x = np.array([g[n] for n in f])
        log = dict(t=np.arange(3.0) * 0.01, duty=np.zeros((3, 4)) + x.sum())
        if kep:
            k = np.zeros((3, len(fwsim.KEP_COLS)))
            k[0, 0] = k[1, 0] = 1.0                       # kep_vx flagged on the two motor-on ticks
            log.update(en=np.array([1.0, 1.0, 0.0]), kep=k)
        # S3V8-7: the two diagnostic counts carry known values (factors off the default; v = 0.8)
        row = dict(status='ok', completed='false' if fail else 'true', e_perp_max=0.01 + 1e-3 * float(x[0]),
                   e_perp_max_cens=0.01, e_perp_global_max=0.01, tv_duty=np.float64(1.0 + float(x[1])),
                   sat_frac=0.0, sat_any_frac=0.0, e_theta_lane_max_deg=1.0, e_theta_cmd_max_deg=1.0,
                   ds_absmax=0.0, e_trk_max=0.0, t_end=np.float64(1.25), e_s_absmax=0.0, track_err_frames=0,
                   n_upper_frames=len(off), n_back_frames=int(v == 0.8),
                   cond2_5_ever=np.bool_(True), pos_reset_mid=0, e_perp_rms=0.0, e_trk_rms=0.0,
                   **_fake_gate_fields(ctrl, g, v))
        if row_over is not None and len(off) <= 1:
            row.update(row_over(ctrl, g, v) or {})
        return row, log

    def fake_new_process(dll, runs, work):
        os.makedirs(work, exist_ok=True)              # S3V8-6: as run_in_new_process, the folder stays
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
        if rel.split(os.sep)[0] in ('sealed', 'sealed_test'):      # v8: two sealed folders
            continue
        for f in files:
            if not f.endswith('.npz'):
                txt[os.path.normpath(os.path.join(rel, f))] = open(os.path.join(root, f), encoding='utf-8').read()
    return txt


def _tree(d):
    """Every folder and file (bytes) under d: the state of sealed/ (S3V8-6)."""
    snap = {}
    for root, _, files in os.walk(d):
        rel = os.path.relpath(root, d)
        snap[('dir', rel)] = None
        for f in files:
            with open(os.path.join(root, f), 'rb') as fh:
                snap[('file', os.path.normpath(os.path.join(rel, f)))] = fh.read()
    return snap


def _test_gains(out):
    """{controller: set of gain tuples} of the 'test' rows of sealed/design.csv: the test-set runs are
    known from the input, not from the phase the script is in (S3V8-6; v8.5 section 7: the design is
    sealed now)."""
    import csv as _csv
    f = out / 'sealed' / 'design.csv'
    if not f.is_file() or not f.read_text().strip():
        return None
    res = {}
    for r in _csv.DictReader(open(f)):
        if r['box'] == 'test':
            res.setdefault(r['controller'], set()).add(tuple(float(r[f'{n}_value']) for n in s3.FACTORS[r['controller']]))
    return res


def _fake_load_screen_u(path=None, trial=False):
    """Stand-in for sl.load_screen_u (S03 v8.5 8.9; independent of the git state of the worktree):
    the sets and the info dict the meta.yaml 'sang_loc' plan line records."""
    screen = {s: np.zeros((sl.SET_ROWS[s], 7)) for s in sl.SETS}
    info = dict(file='fake_screen_u.csv', sha256='0' * 64, sha256_lf='1' * 64, sha256_crlf='2' * 64,
                git=dict(tracked=True, clean=True, commit='f4k3'), head_blob_sha256='0' * 64,
                matches_head_blob_lf=True, problems=[])
    return screen, info


def _fake_run_step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc,
                   tool_check=None):
    """Default fake of sl.run_step (S03 3.1b): every controller falls back, nhanh
    'du_phong_kiem'; a minimal sealed/sangloc/ket_qua.json is written; returns the run_step dict
    shape (mien / nhanh / ly_do / N_da_dung / pha)."""
    sl_dir = os.path.join(sealed, 'sangloc')
    os.makedirs(sl_dir, exist_ok=True)
    doc = dict(N=None, N_da_dung=0, m_min=sl.M_MIN)
    for c in controllers:
        doc[c] = dict(nhanh='du_phong_kiem', trang_thai='hoan_tat',
                      ly_do=['fake run_step: no accepted box'], nguon_mien='du_phong',
                      nguong={m: None for m in sl.MODES},
                      hop={b: None for b in ('B_giua', 'B_ba_mu', 'B1', 'B2')}, mien=None,
                      che_do={m: dict(kiem_duoc_nhan_o_lan=None, can=None, tap={})
                              for m in sl.MODES},
                      n_ngoai_bang={s: 0 for s in sl.SCREEN_SETS},
                      ti_le_kha_thi_trong_mien={m: None for m in sl.MODES},
                      n_cap_trung_float32=0, kiem_tat_dinh=None)
    sl.write_json(os.path.join(sl_dir, 'ket_qua.json'), doc)
    return dict(mien={c: None for c in controllers},
                nhanh={c: 'du_phong_kiem' for c in controllers},
                ly_do={c: ['fake run_step: no accepted box'] for c in controllers},
                N_da_dung=0, pha=[])


def _run_main(tmp_path, monkeypatch, capsys, error_in_test=False, argv=None, run_step=None,
              load_screen=None, patch_screen=True, **kw):
    """error_in_test: True -> run_error row; 'warn' -> a warning and a print in every test run;
    'raise' -> an OSError (not S03Stop) in the first test run.
    Non-trial runs (S03 3.1b wired in): s03_sangloc.load_screen_u and run_step are faked so the
    pipeline does not depend on the git state of s03_screen_u.csv (run_step / load_screen give
    other fakes; patch_screen=False keeps the real functions, e.g. the --sangloc-thu check)."""
    fsr, fnp, calls, works = _fake(**kw)
    argv = argv or ['--mu', '0.5', '--trial']
    if '--trial' not in argv and patch_screen:
        monkeypatch.setattr(sl, 'load_screen_u', load_screen or _fake_load_screen_u)
        monkeypatch.setattr(sl, 'run_step', run_step or _fake_run_step)
    out = tmp_path / 'run'
    seen = dict(design_before_first_main_run=None, analyse=[], after_sealed=0, other_after_test=0,
                manifest_at_first_test=None, sealed_before_test=None, sealed_after=None,
                meta_at_first_run=None)
    tg = {}

    def fsr_watch(fw, path, ctrl, g, v, mu):
        dflt = FAKE_DEFAULTS[ctrl]
        if seen['meta_at_first_run'] is None:
            m = out / 'meta.yaml'                        # S03 8.9: the plan meta at the first run
            seen['meta_at_first_run'] = m.read_text(encoding='utf-8') if m.is_file() else None
        if not tg.get('loaded'):
            got = _test_gains(out)
            if got is not None:
                tg.update(got, loaded=True)
        is_test = tg.get('loaded') and tuple(g[n] for n in s3.FACTORS[ctrl]) in tg.get(ctrl, ())
        if is_test:                                    # S3V8-6: a test-set run, known from sealed/design.csv
            if seen['after_sealed'] == 0:
                seen['manifest_at_first_test'] = (out / 'sealed' / 'manifest.json').is_file()
                seen['sealed_before_test'] = _tree(out / 'sealed')
            seen['after_sealed'] += 1
            if error_in_test is True:
                calls.append(-1)
                return dict(status='run_error', completed='', error='FakeTestError: boom'), None
            if error_in_test == 'warn':
                import warnings as _w
                _w.warn('FakeTestWarning p%d' % seen['after_sealed'], RuntimeWarning)
                print('FakeTestPrint')
            if error_in_test == 'raise':
                raise OSError('FakeTestOSError while writing a log of a test point')
        elif seen['after_sealed']:
            seen['other_after_test'] += 1
        if seen['design_before_first_main_run'] is None and \
                sum(not math.isclose(g[n], dflt[n]) for n in s3.FACTORS[ctrl]) >= 2:
            f = out / 'sealed' / 'design.csv'          # v8.5 section 7: the design is sealed
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
    monkeypatch.setattr(sys, 'argv', ['s03_doe.py', '--out', str(out)] + (argv or ['--mu', '0.5', '--trial']))
    code = s3.main()
    if (out / 'sealed').is_dir():
        seen['sealed_after'] = _tree(out / 'sealed')
    cap = capsys.readouterr()
    seen['stderr'] = cap.err
    return code, out, cap.out, calls, works


@needs_dll
def test_main_pipeline_clean(tmp_path, monkeypatch, capsys):
    import csv as _csv
    code, out, console, calls, works = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    for f in ('sweep.csv', 'sweep_counts.csv', 'levels.csv', 'meta.yaml', 'sealed/runs.csv',
              'sealed/effects.csv', 'sealed/fail_counts.csv', 'sealed/notes.json', 'sealed/manifest.json',
              'sealed/determinism_check.csv', 'sealed_test/diag.txt',
              # v8.5 section 7 [v8.5]: the design and the screening levels are sealed, with the
              # screening outputs and the sealed diagnostics
              'sealed/design.csv', 'sealed/levels_sangloc.csv', 'sealed/diag.txt',
              'sealed/sangloc/ket_qua.json'):
        assert (out / f).is_file(), f
    assert not (out / 'design.csv').exists()                 # v8.5 section 7: no public design.csv
    assert not (out / 'determinism_check.csv').exists() and not (out / '_work').exists()
    assert works and all(os.path.relpath(w, out).split(os.sep)[0] == 'sealed' for w in works)   # S3C-2
    assert not (out / 'sealed_test' / '_work').exists()
    # S3V8-1, S3V8-6: test runs known from design.csv; sealed/ (files, folders, bytes) as it was
    # just before the first test run, and closed (manifest) by then; no other run after it
    seen = calls.seen
    assert seen['after_sealed'] == 2 * 100 * 3 and seen['other_after_test'] == 0
    assert seen['manifest_at_first_test'] is True
    assert seen['sealed_before_test'] == seen['sealed_after']
    assert ('dir', '_work') not in seen['sealed_after']           # removed before the manifest
    runs = list(_csv.DictReader(open(out / 'sealed' / 'runs.csv')))
    assert len(runs) == 2 * (129 + 128 + 14) * 3                   # L41: the inner box always runs; v8 axial
    assert sum(r['box'] == 'inner' for r in runs) == 2 * 128 * 3
    assert sum(r['box'] == 'axial' for r in runs) == 2 * 14 * 3 and not any(r['box'] == 'test' for r in runs)
    design = list(_csv.DictReader(open(out / 'sealed' / 'design.csv')))      # v8.5 7: sealed
    assert len(design) == 2 * 371 and {r['box'] for r in design} == {'centre', 'outer', 'inner', 'axial', 'test'}
    test = list(_csv.DictReader(open(out / 'sealed_test' / 'runs_test.csv')))      # v8: separate seal
    assert len(test) == 2 * 100 * 3 and {r['box'] for r in test} == {'test'} and {r['mu'] for r in test} == {'0.5'}
    for f in ('notes.json', 'manifest.json'):
        assert (out / 'sealed_test' / f).is_file()
    sealed_txt = ''.join(open(os.path.join(r_, f_), encoding='utf-8', errors='replace').read()
                         for r_, _, fs in os.walk(out / 'sealed') for f_ in fs
                         if not f_.endswith('.npz') and f_ != 'design.csv')
    # sealed/ says nothing of the test set; design.csv is excluded: it lists the 100 test
    # points (an input of the phase, S03 section 7)
    assert 'test' not in sealed_txt.replace('determinism', '')
    # S3V8-7: the values of the two diagnostic counts reach the three CSV files (fake: n_upper =
    # factors off the default, n_back = 1 at v = 0.8)
    off = {(d['controller'], d['point_id']): sum(not math.isclose(float(d[f'{n}_value']), FAKE_DEFAULTS[d['controller']][n])
                                                 for n in s3.FACTORS[d['controller']]) for d in design}
    assert max(off.values()) == 7
    for r in runs + test:
        assert int(r['n_upper_frames']) == off[(r['controller'], r['point_id'])], (r['box'], r['point_id'])
        assert int(r['n_back_frames']) == (r['v'] == '0.8')
    for r in list(_csv.DictReader(open(out / 'sweep.csv'))):
        assert int(r['n_upper_frames']) == (0 if r['is_default'] == 'true' else 1)
        assert int(r['n_back_frames']) == (r['v'] == '0.8')
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
        assert c.seen['design_before_first_main_run'].count('\n') == 1 + 2 * 371
        assert c.seen['after_sealed'] == 2 * 100 * 3            # v8: only the test set runs after sealed/
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
    # v8.5 section 7: design.csv is sealed and is never made after a sweep stop
    assert not (out / 'design.csv').exists() and not (out / 'sealed' / 'design.csv').exists()
    sweep = list(_csv.DictReader(open(out / 'sweep.csv')))   # S3D-4: the default check is recorded
    assert [(r['factor'], r['v'], r['completed']) for r in sweep] == [
        ('(default check)', '0.2', 'true'), ('(default check)', '0.5', 'true'), ('(default check)', '0.8', 'false')]
    assert 'stopped in the sweep' in (out / 'meta.yaml').read_text(encoding='utf-8')
    assert next(_csv.reader(open(out / 'levels.csv', encoding='utf-8'))) == s3.LEVEL_COLS      # S83-1: header kept


@needs_dll
def test_main_run_error_in_the_design_is_recorded_and_sealed(tmp_path, monkeypatch, capsys):
    """S3C-12, S3C-3: the run_error row is in sealed/runs.csv; the console and meta.yaml only say
    that S03 stopped after the sweep (v8.5 3.1b item 13: the reason is in sealed/stop.json)."""
    import csv as _csv
    code, out, console, _, _ = _run_main(tmp_path, monkeypatch, capsys, error_in_design=True)
    assert code == 2
    assert 'FakeError' not in console and 'sealed/stop.json' in console
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'FakeError' not in meta and 'stopped after the sweep' in meta
    rows = list(_csv.DictReader(open(out / 'sealed' / 'runs.csv')))
    assert rows[-1]['status'] == 'run_error' and 'FakeError' in rows[-1]['error']
    assert 'FakeError' in (out / 'sealed' / 'stop.json').read_text()
    design = list(_csv.DictReader(open(out / 'sealed' / 'design.csv')))   # v8.5 7: sealed, before the run
    assert {r['controller'] for r in design} == {'PID', 'ADRC'} and len(design) == 2 * 371
    assert not (out / 'sealed_test').exists()                       # v8: no test set after a sealed stop


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
    """S3C-13 / S03 v8.6: with the reference 0 the tv_duty threshold is still the absolute 0.4
    (the X3 effect 0.2 is not significant, no 'reference run is 0' note); fail counts over mu."""
    def model(c, mu):
        return _row(0.05, tv=0.1 * c[2], completed='false' if (c[0] > 0 and mu == 0.6) else 'true')
    pts, runs = _runs(model)
    eff, _, _, fails = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): 0.0})
    r = next(x for x in eff if x['response'] == 'tv_duty' and x['term'] == 'X3' and x['mu'] == '0.5')
    # S03 v8.6: the Y3 threshold is the absolute 0.4 s^-1 whatever the reference holds
    assert r['threshold'] == 0.4 and not r['significant'] and 'reference run is 0' not in r['note']
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


# ── S03 v8 (L67): axial points, test set, checks of section 11 item 4 ────────────────

def test_v8_axial_points_ids_and_codes():
    pts = dict(s3.axial_points())
    assert sorted(pts) == list(range(257, 271))
    for k in range(1, 8):                       # X_k at -1: 255 + 2k, at +1: 256 + 2k, others 0
        assert pts[255 + 2 * k] == tuple(-1.0 if j == k - 1 else 0.0 for j in range(7))
        assert pts[256 + 2 * k] == tuple(1.0 if j == k - 1 else 0.0 for j in range(7))


def test_v8_committed_test_points_pass_condition_7():
    """The committed file reads, passes section 8.7, and is the LHS of section 3.2 bit for bit."""
    from scipy.stats import qmc
    pts = s3.check_test_points()
    assert [p for p, _ in pts] == list(range(271, 371))
    u = qmc.LatinHypercube(d=7, scramble=True, optimization=None, rng=np.random.default_rng(20261107)).random(100)
    assert all(struct.pack('<d', 2.0 * u[j, k] - 1.0) == struct.pack('<d', pts[j][1][k])
               for j in range(100) for k in range(7))


@pytest.mark.parametrize('edit, why', [
    (lambda rows: rows.__setitem__(0, rows[0].replace('X7', 'X8')), 'columns'),
    (lambda rows: rows.pop(), 'rows'),
    (lambda rows: rows.__setitem__(5, rows[5].replace('5,275,', '5,276,', 1)), 'point_id'),
    (lambda rows: rows.__setitem__(3, ','.join(rows[3].split(',')[:2] + rows[4].split(',')[2:])), 'bin'),
    (lambda rows: rows.__setitem__(9, ','.join(rows[9].split(',')[:2] + ['1.5'] + rows[9].split(',')[3:])),
     'finite'),
])
def test_v8_test_points_check_catches(tmp_path, edit, why):
    rows = open(s3.TEST_POINTS, encoding='utf-8').read().splitlines()
    edit(rows)
    p = tmp_path / 'tp.csv'
    p.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    with pytest.raises(s3.S03Stop, match=why):
        s3.check_test_points(str(p))


def test_v8_make_test_points_never_overwrites(tmp_path):
    with pytest.raises(s3.S03Stop, match='never generated again'):
        s3.make_test_points(s3.TEST_POINTS, str(tmp_path / 'g.json'))
    p, g = tmp_path / 'tp.csv', tmp_path / 'g.json'
    gen = s3.make_test_points(str(p), str(g))
    assert p.read_bytes() == open(s3.TEST_POINTS, 'rb').read()      # same file on this machine
    assert gen['seed'] == 20261107 and gen['bit_generator'] == 'PCG64' and not (tmp_path / 'tp.csv.tmp').exists()


def _design_rows():
    rows = []
    pts = ([(0, 'centre', (0.0,) * 7)] + [(p, 'outer', c) for p, c in s3.design_points(1.0, 1)] +
           [(p, 'inner', c) for p, c in s3.design_points(0.5, 129)] + [(p, 'axial', c) for p, c in s3.axial_points()] +
           [(p, 'test', c) for p, c in s3.check_test_points()])
    for pid, box, codes in pts:
        rows.append(dict(controller='PID', point_id=pid, box=box, **{f'X{k + 1}': c for k, c in enumerate(codes)}))
    return rows


def test_v8_design_map_check():
    rows, tp = _design_rows(), s3.check_test_points()
    s3.check_design_map(rows, tp)
    bad = [dict(r) for r in rows]
    bad[200]['box'] = 'outer'                                   # an inner point labelled outer
    with pytest.raises(s3.S03Stop, match='point 200'):
        s3.check_design_map(bad, tp)
    bad = [dict(r) for r in rows]
    bad[260]['X3'] = 1.0                                        # an axial point with a wrong code
    with pytest.raises(s3.S03Stop, match='point 260'):
        s3.check_design_map(bad, tp)
    with pytest.raises(s3.S03Stop, match='370 rows'):
        s3.check_design_map(rows[:-1], tp)


def test_v8_design_map_checks_the_test_codes():
    """S3V8-4: codes of points 271..370 compared with the test file, bit for bit: swapped codes,
    a NaN, or -0.0 for 0.0 are caught."""
    rows, tp = _design_rows(), s3.check_test_points()
    i271, i272 = (next(i for i, r in enumerate(rows) if r['point_id'] == p) for p in (271, 272))
    bad = [dict(r) for r in rows]
    for k in range(1, 8):
        bad[i271][f'X{k}'], bad[i272][f'X{k}'] = rows[i272][f'X{k}'], rows[i271][f'X{k}']
    with pytest.raises(s3.S03Stop, match='point 271'):
        s3.check_design_map(bad, tp)
    bad = [dict(r) for r in rows]
    bad[i272]['X1'] = float('nan')
    with pytest.raises(s3.S03Stop, match='point 272'):
        s3.check_design_map(bad, tp)
    i0 = next(i for i, r in enumerate(rows) if r['point_id'] == 0)
    bad = [dict(r) for r in rows]
    bad[i0]['X4'] = -0.0                                        # equal to 0.0 as a number, not as bits
    with pytest.raises(s3.S03Stop, match='point 0'):
        s3.check_design_map(bad, tp)


def test_v8_latin_bins_use_exact_decimal_bounds(tmp_path):
    """S3V8-5 (Codex counterexample): in X1, the points of bins 57 [0.12, 0.14) and 58 [0.14, 0.16)
    become 0.14 and 0.14000000000000012: both are in bin 58, bin 57 is empty. Float bounds
    (-1 + 0.02 * 57 = 0.14000000000000012) counted one point in each bin."""
    rows = open(s3.TEST_POINTS, encoding='utf-8').read().splitlines()
    head, body = rows[0], [r.split(',') for r in rows[1:]]
    x1 = [float(r[2]) for r in body]
    j57 = next(j for j, x in enumerate(x1) if 0.12 <= x < 0.14)
    j58 = next(j for j, x in enumerate(x1) if 0.14 < x < 0.16)
    assert -1.0 + 0.02 * 57 == 0.14000000000000012                # the rounding is really there
    body[j57][2], body[j58][2] = repr(0.14), repr(0.14000000000000012)
    text = '\n'.join([head] + [','.join(r) for r in body]) + '\n'
    with pytest.raises(s3.S03Stop, match='bin 57'):
        s3.check_test_points_text(text)
    assert s3._latin_bin(0.14) == 58 and s3._latin_bin(np.nextafter(0.14, -1)) == 57
    assert s3._latin_bin(-1.0) == 1 and s3._latin_bin(np.nextafter(1.0, -2)) == 100 and s3._latin_bin(1.0) == 101
    assert s3._latin_bin(-0.0) == s3._latin_bin(0.0) == 51


def _git(d, *args):
    import subprocess
    r = subprocess.run(['git', '-C', str(d), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'core.autocrlf=false']
                       + list(args), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _repo_with_test_file(tmp_path, gen_sha=None):
    """A git repo holding a copy of the test file (bytes as in this work tree) and its .gen.json."""
    import hashlib
    import json as _json
    d = tmp_path / 'repo'
    d.mkdir(parents=True)
    _git(d, 'init', '-q')
    raw = open(s3.TEST_POINTS, 'rb').read()
    (d / 'tp.csv').write_bytes(raw)
    gen = dict(seed=20261107, sha256=gen_sha or hashlib.sha256(raw).hexdigest())
    (d / 'tp.gen.json').write_text(_json.dumps(gen), encoding='utf-8')
    _git(d, 'add', 'tp.csv', 'tp.gen.json')
    _git(d, 'commit', '-q', '-m', 'test file')
    return d


def test_v8_load_test_points_committed_file():
    """S3V8-3: the committed file passes; the two SHA-256 are of the bytes read; the generation
    record is the one beside the file; content equal to HEAD after CRLF -> LF."""
    import hashlib
    pts, info = s3.load_test_points()
    raw = open(s3.TEST_POINTS, 'rb').read()
    assert pts == s3.check_test_points()
    assert info['sha256'] == hashlib.sha256(raw).hexdigest()
    assert info['sha256_lf'] == hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    assert info['matches_head_blob_lf'] and info['generation_sha256_matches'] and info['problems'] == []
    assert info['generation']['sha256'] in (info['sha256'], info['sha256_lf'])
    assert info['generation_file'].endswith('s03_test_points.gen.json')


def test_v8_load_test_points_assume_unchanged_is_caught(tmp_path):
    """S3V8-3 (Codex counterexample): tracked file marked assume-unchanged, one code moved inside
    the same Latin bin: git status is clean and condition 7 passes, the HEAD blob differs -> stop."""
    d = _repo_with_test_file(tmp_path)
    p = d / 'tp.csv'
    s3.load_test_points(str(p))                                  # as committed: passes
    _git(d, 'update-index', '--assume-unchanged', 'tp.csv')
    raw = p.read_bytes()
    first = raw.split(b'\n')[1].split(b',')[2]                   # X1 of point 271
    new = repr(float(first) + 1e-6).encode()
    assert s3._latin_bin(float(new)) == s3._latin_bin(float(first))
    p.write_bytes(raw.replace(first, new, 1))
    assert _git(d, 'status', '--porcelain').strip() == ''        # git status does not see it
    s3.check_test_points(str(p))                                 # condition 7 still passes
    with pytest.raises(s3.S03Stop, match='blob in HEAD'):
        s3.load_test_points(str(p))
    _, info = s3.load_test_points(str(p), trial=True)            # --trial: recorded, not a stop
    assert any('blob in HEAD' in x for x in info['problems'])


def test_v8_load_test_points_lf_checkout_with_the_committed_record(tmp_path):
    """R2-2: a clean LF checkout of the same content (Pi, core.autocrlf=false) passes with the
    generation record as committed (sha256 of the CRLF file written on Windows); a changed code
    in the LF file still fails the record."""
    import hashlib
    import json as _json
    d = _repo_with_test_file(tmp_path)                            # record: SHA-256 of the CRLF bytes
    p = d / 'tp.csv'
    crlf_sha = _json.loads((d / 'tp.gen.json').read_text(encoding='utf-8'))['sha256']
    assert crlf_sha == hashlib.sha256(open(s3.TEST_POINTS, 'rb').read()).hexdigest()
    lf = p.read_bytes().replace(b'\r\n', b'\n')
    assert lf != p.read_bytes()                                   # the work tree really holds CRLF
    p.write_bytes(lf)
    _git(d, 'add', 'tp.csv')
    _git(d, 'commit', '-q', '-m', 'lf checkout')
    _, info = s3.load_test_points(str(p))
    assert info['sha256'] == info['sha256_lf'] != crlf_sha == info['sha256_crlf']
    assert info['generation_sha256_matches'] and info['problems'] == []
    first = lf.split(b'\n')[1].split(b',')[2]
    p.write_bytes(lf.replace(first, repr(float(first) + 1e-6).encode(), 1))
    _git(d, 'commit', '-q', '-am', 'changed code')                # committed, but not the generated file
    with pytest.raises(s3.S03Stop, match='generation record'):
        s3.load_test_points(str(p))


def test_v8_load_test_points_unreadable_record_is_recorded(tmp_path, monkeypatch):
    """S3V8-3 round 2: an I/O error opening <name>.gen.json is recorded (trial) / a stop (main),
    not an exception out of the preflight."""
    d = _repo_with_test_file(tmp_path)
    p = d / 'tp.csv'
    real_open = open

    def fake_open(f, *a, **k):
        if str(f).endswith('tp.gen.json'):
            raise PermissionError('FakeLockedRecord')
        return real_open(f, *a, **k)
    monkeypatch.setattr('builtins.open', fake_open)
    _, info = s3.load_test_points(str(p), trial=True)
    assert 'FakeLockedRecord' in info['generation']['unreadable'] and info['problems']
    with pytest.raises(s3.S03Stop, match='generation record') as e:
        s3.load_test_points(str(p))
    assert e.value.tp_info['sha256'] == info['sha256']           # R2-4: what was read travels with the stop


def test_v8_load_test_points_eol_and_own_generation_record(tmp_path):
    """S3V8-3: LF bytes on disk (blob LF) pass when the record has the LF SHA-256; the record
    read is <name>.gen.json beside the file given, not the default one; a wrong sha256 stops."""
    import hashlib
    d = _repo_with_test_file(tmp_path)
    p = d / 'tp.csv'
    lf = p.read_bytes().replace(b'\r\n', b'\n')
    p.write_bytes(lf)
    (d / 'tp.gen.json').write_text('{"seed": 1, "sha256": "%s"}' % hashlib.sha256(lf).hexdigest(), encoding='utf-8')
    _git(d, 'add', 'tp.csv', 'tp.gen.json')
    _git(d, 'commit', '-q', '-m', 'lf')
    _, info = s3.load_test_points(str(p))
    assert info['generation']['seed'] == 1 and info['sha256'] == info['sha256_lf']
    d2 = _repo_with_test_file(tmp_path / 'b', gen_sha='0' * 64)
    with pytest.raises(s3.S03Stop, match='generation record'):
        s3.load_test_points(str(d2 / 'tp.csv'))
    (d2 / 'tp.gen.json').unlink()
    _git(d2, 'commit', '-q', '-am', 'no record')
    with pytest.raises(s3.S03Stop, match='generation record'):
        s3.load_test_points(str(d2 / 'tp.csv'))


def test_v8_bounds_check():
    lv = {n: (1.0, 8.0, 'log') for n in s3.FACTORS['ADRC']}
    lv['omega_o_xy'] = (12.0, 48.0, 'log')

    def gains_at(codes):
        return {n: s3.decode(c, *lv[n]) for n, c in zip(s3.FACTORS['ADRC'], codes)}
    pts = [(1, 'outer', (1.0,) * 7), (2, 'outer', (-1.0,) * 7)]
    s3.check_bounds('ADRC', lv, pts, gains_at)                 # on the bounds: passes
    with pytest.raises(s3.S03Stop, match=r'11.4 \(i\)'):
        s3.check_bounds('ADRC', lv, [(3, 'test', (1.0 + 1e-6,) + (0.0,) * 6)], gains_at)
    lv['omega_o_xy'] = (12.0, 48.00001, 'log')                # float32 of H is above 48
    with pytest.raises(s3.S03Stop, match=r'11.4 \(ii\)'):
        s3.check_bounds('ADRC', lv, [(4, 'outer', (0.0, 1.0) + (0.0,) * 5)], gains_at)


@needs_dll
def test_v8_run_error_in_the_test_set_stays_in_sealed_test(tmp_path, monkeypatch, capsys):
    """A stop in the test set: sealed/ is already closed and unchanged (no stop.json, nothing about
    the test set); the reason is only in sealed_test/stop.json; the console says only that S03
    stopped in the test set."""
    import json as _json
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, error_in_test=True)
    assert code == 2
    assert 'FakeTestError' not in console and 'sealed_test/stop.json' in console
    assert not (out / 'sealed' / 'stop.json').exists()
    assert 'FakeTestError' in (out / 'sealed_test' / 'stop.json').read_text()
    man = _json.load(open(out / 'sealed' / 'manifest.json'))
    for name, sha in man['files'].items():                     # sealed/ as closed before the test set
        assert s3.sha256_file(str(out / 'sealed' / name)) == sha
    assert calls.seen['sealed_before_test'] == calls.seen['sealed_after']      # S3V8-6: whole tree
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'stopped in the test set' in meta and 'FakeTestError' not in meta


@needs_dll
@pytest.mark.parametrize('mode', ['warn', 'raise'])
def test_v8_test_phase_diagnostics_stay_in_sealed_test(tmp_path, monkeypatch, capsys, mode):
    """S3V8-2: a warning or a print of a test run, or an exception that is not S03Stop, leaves
    nothing on the console or stderr; it is in sealed_test/diag.txt (and stop.json for the
    exception); sealed/ unchanged."""
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, error_in_test=mode)
    err = calls.seen['stderr']
    for word in ('FakeTest', 'p271', 'Traceback'):
        assert word not in console and word not in err
    diag = (out / 'sealed_test' / 'diag.txt').read_text(encoding='utf-8')
    man = (out / 'sealed_test' / 'manifest.json').read_text()
    assert 'diag.txt' in man
    if mode == 'warn':
        assert code == 0 and 'FakeTestWarning' in diag and 'FakeTestPrint' in diag
        assert diag.count('warning: RuntimeWarning: FakeTestWarning') >= 1     # policy in force (R2-3)
    else:
        assert code == 2 and 'sealed_test/stop.json' in console
        assert 'FakeTestOSError' in (out / 'sealed_test' / 'stop.json').read_text()
        assert 'Traceback' in diag and 'FakeTestOSError' in diag
        assert 'stopped in the test set' in (out / 'meta.yaml').read_text(encoding='utf-8')
    assert calls.seen['sealed_before_test'] == calls.seen['sealed_after']


@needs_dll
def test_v8_test_phase_keeps_the_warning_policy(tmp_path, monkeypatch, capsys):
    """R2-3: with RuntimeWarning made an error (as -W error::RuntimeWarning), a warning of a test
    run is still an error: the test phase stops (before the fix the phase forced 'always' and ran
    to the end); nothing on the console but the generic stop line."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('error', RuntimeWarning)
        code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, error_in_test='warn')
    assert code == 2 and 'sealed_test/stop.json' in console
    assert 'FakeTestWarning' not in console and 'FakeTestWarning' not in calls.seen['stderr']
    assert 'RuntimeWarning: FakeTestWarning' in (out / 'sealed_test' / 'stop.json').read_text()
    assert calls.seen['after_sealed'] == 1                        # stopped at the first test run


@needs_dll
def test_v8_output_error_of_the_test_phase_stays_sealed(tmp_path, monkeypatch, capsys):
    """R2-1: writing runs_test.csv fails (PermissionError): the stop is in sealed_test/stop.json
    (written before the other outputs) with the error, the traceback in diag.txt; console and
    stderr show only the generic stop line; sealed/ unchanged."""
    real = s3._write_csv_to

    def failing(f, rows, cols=None):
        if rows and rows[0].get('box') == 'test':
            raise PermissionError('FakeDiskError on runs_test.csv')
        return real(f, rows, cols)
    monkeypatch.setattr(s3, '_write_csv_to', failing)
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 2 and 'sealed_test/stop.json' in console
    for txt in (console, calls.seen['stderr']):
        assert 'FakeDiskError' not in txt and 'Traceback' not in txt
    stop = (out / 'sealed_test' / 'stop.json').read_text()
    assert 'writing the test-set outputs failed' in stop and 'FakeDiskError' in stop
    assert 'FakeDiskError' in (out / 'sealed_test' / 'diag.txt').read_text(encoding='utf-8')
    assert calls.seen['sealed_before_test'] == calls.seen['sealed_after']
    assert 'stopped in the test set' in (out / 'meta.yaml').read_text(encoding='utf-8')


@needs_dll
def test_v8_unwritable_sealed_test_still_says_only_stop(tmp_path, monkeypatch, capsys):
    """R2-1: sealed_test/ cannot even be created: nothing is written there, and the console and
    stderr still carry only the generic stop line (no traceback)."""
    real = os.makedirs

    def mk(p, *a, **k):
        if os.path.basename(os.path.normpath(p)) == 'sealed_test':
            raise PermissionError('FakeNoFolder')
        return real(p, *a, **k)
    monkeypatch.setattr(s3.os, 'makedirs', mk)
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 2 and 'sealed_test/stop.json' in console
    for txt in (console, calls.seen['stderr']):
        assert 'FakeNoFolder' not in txt and 'Traceback' not in txt
    assert not (out / 'sealed_test').exists() and calls.seen['after_sealed'] == 0


@needs_dll
def test_v8_stop_at_the_test_file_keeps_what_was_read(tmp_path, monkeypatch, capsys):
    """R2-4: a main run stops because the test file is not committed: meta.yaml still records the
    SHA-256 and the problems of the bytes read (not 'not read'). v8.5 section 7: the file is read
    once at the start, so the stop is now BEFORE the sweep and no run is made at all."""
    import hashlib
    p = tmp_path / 'tp.csv'
    raw = open(s3.TEST_POINTS, 'rb').read()
    p.write_bytes(raw)
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys,
                                             argv=['--mu', '0.4', '0.5', '0.6', '--test-points', str(p)])
    assert code == 2 and 'not committed' in console
    assert not calls                                            # the stop is before any run
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    line = next(l for l in meta.splitlines() if l.startswith('test_points:'))
    assert hashlib.sha256(raw).hexdigest() in line and 'not committed' in line and '"read": "no' not in line


@needs_dll
def test_v8_uncommitted_test_points_stop_a_main_run(tmp_path, monkeypatch, capsys):
    """Section 8.7: without --trial the test file must be tracked and unmodified in git; v8.5
    section 7: read at the start, so a failed check is a public stop before the sweep."""
    p = tmp_path / 'tp.csv'
    p.write_bytes(open(s3.TEST_POINTS, 'rb').read())
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys,
                                             argv=['--mu', '0.4', '0.5', '0.6', '--test-points', str(p)])
    assert code == 2 and 'not committed' in console
    assert not calls                                            # the stop is before any run
    assert not (out / 'design.csv').exists() and not (out / 'sealed_test').exists()


# ── S03 v8.2: e_bao_max and G1-G5 (design 03_CODE/docs/2026-11-14_Thiet_ke_S03_G1G5_v1.md) ──────────

import o1_gate as G  # noqa: E402
import o1_gate_eval as E  # noqa: E402

# the 27 runs.csv columns of S03 v8 (blob 2127c69), written out here so that a change to them fails
V8_COLS = ['controller', 'point_id', 'box', 'mu', 'v', 'status', 'completed', 'e_perp_max', 'e_perp_max_cens',
           'e_perp_global_max', 'tv_duty', 'sat_frac', 'sat_any_frac', 'e_theta_lane_max_deg',
           'e_theta_cmd_max_deg', 'ds_absmax', 'e_trk_max', 't_end', 'e_s_absmax', 'track_err_frames',
           'cond2_5_ever', 'pos_reset_mid', 'e_perp_rms', 'e_trk_rms', 'n_upper_frames', 'n_back_frames', 'error']
V82_COLS = (['e_bao_max', 'gate_version', 'gate_decide', 'gate_u_t', 'g1_q', 'g1_state']
            + [f'g{i}_{x}' for i in range(2, 6) for x in ('q', 'lo', 'hi', 'n_undef', 'state')] + ['gate_outcome'])
Q_OF = dict(g2='e_perp_max', g3='e_trk_max', g4='e_bao_max', g5='e_theta_cmd_max_deg')
# B81-3 G1 fields and the common domain (15/11), after gate_outcome; names of rq3_log GATE_COLS
G1C_COLS = ([f'g{i}_{x}_chung' for i in range(2, 6) for x in ('q', 'lo', 'hi', 'n_undef')]
            + ['g1_ways', 'g1_t_deadline', 'g1_t_done_1', 'g1_t_done_2', 'g1_t1', 'g1_prog_before_t1',
               'g1_cond25_before_deadline', 'g1_cat_t1'])
G1F = {'true': dict(t_done=[5.0], t_deadline=10.0, t1=None, prog_before_t1=None, cond25_before_deadline=None),
       'false': dict(t_done=[None], t_deadline=10.0, t1=None, prog_before_t1=None, cond25_before_deadline=None),
       'inconclusive': dict(t_done=[None], t_deadline=10.0, t1=2.0, prog_before_t1=False,
                            cond25_before_deadline=True)}


def test_v82_columns_appended_after_the_v8_columns():
    assert s3.OLD_COLS == V8_COLS and s3.RUN_COLS[:len(V8_COLS)] == V8_COLS
    assert s3.GATE_RUN_COLS == V82_COLS and s3.G1_RUN_COLS == G1C_COLS
    assert s3.RUN_COLS == V8_COLS + V82_COLS + G1C_COLS
    tools = os.path.join(HERE, '..', '..', 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import rq3_log                                       # B81-3: the same names as the shared gate columns
    assert set(V82_COLS) - {'e_bao_max', 'gate_outcome'} <= set(rq3_log.GATE_COLS)


def _gframes(e_perp, e_trk, e_bao, th):
    return dict(e_perp=np.array(e_perp, float), e_trk=np.array(e_trk, float), e_bao=np.array(e_bao, float),
                e_theta_cmd_deg=np.array(th, float))


@pytest.mark.parametrize('frames, completed, want, outcome', [
    (_gframes([0.05, 0.02], [0.03, 0.01], [0.06, 0.04], [1.0, 2.0]), 'true',
     dict(g1='dat', g2='dat', g3='dat', g4='dat', g5='dat'), 'dat'),
    (_gframes([0.05, 0.02], [0.12, 0.01], [0.06, np.nan], [1.0, 2.0]), 'true',
     dict(g1='dat', g2='dat', g3='khong_dat', g4='chua_ket_luan', g5='dat'), 'khong_dat'),
    (_gframes([0.05, 0.02], [0.03, 0.01], [0.06, 0.04], [1.0, 8.0]), 'inconclusive',
     dict(g1='chua_ket_luan', g2='dat', g3='dat', g4='dat', g5='khong_dat'), 'khong_dat'),
    (_gframes([0.05, 0.10], [0.03, 0.10], [0.06, 0.10], [1.0, 7.5]), 'inconclusive',
     dict(g1='chua_ket_luan', g2='dat', g3='dat', g4='dat', g5='dat'), 'chua_ket_luan'),
    (_gframes([0.05, 0.02], [0.03, 0.01], [0.06, 0.04], [1.0, 2.0]), 'false',
     dict(g1='khong_dat', g2='dat', g3='dat', g4='dat', g5='dat'), 'khong_dat'),
])
def test_v82_gate_fields_are_the_gate_dict(frames, completed, want, outcome):
    """T3: gate_fields copies the o1_gate.gate states and maxima (an edge equal to the threshold passes,
    O1 3.5)."""
    gt = G.gate(frames, completed, v_cmd_max=0.9, wz_cmd_max=1.1, u_t=s3.GATE_U_T, decide=G.DECIDE_ALL,
                g1_fields=G1F[completed])
    m = float(np.max(frames['e_bao']))
    f = s3.gate_fields(dict(e_bao_max=m), gt)
    assert set(f) == set(V82_COLS) | set(G1C_COLS)
    assert {c: f[f'{c}_state'] for c in want} == want and f['gate_outcome'] == outcome
    assert f['gate_version'] == G.GATE_VERSION and f['gate_decide'] == 'g1,g2,g3,g4,g5' and f['gate_u_t'] == 0.0
    assert f['g1_q'] == {'dat': 'true', 'khong_dat': 'false'}.get(want['g1'], 'inconclusive')
    assert f['g4_n_undef'] == int(np.count_nonzero(~np.isfinite(frames['e_bao'])))
    for c in ('g2', 'g3', 'g4', 'g5'):
        assert f[f'{c}_q'] == f[f'{c}_lo'] == f[f'{c}_hi'] == gt[c]['q']      # U = 0 on fwsim
    assert s3.fmt(f['e_bao_max']) == repr(m)
    # G1F-2: the 24 B81-3 columns are the fields of the gate dict, one by one (here prog != cond25 when inconclusive)
    g1 = G1F[completed]
    assert (f['g1_ways'], f['g1_t_deadline'], f['g1_t_done_1'], f['g1_t_done_2'], f['g1_t1'], f['g1_prog_before_t1'],
            f['g1_cond25_before_deadline'], f['g1_cat_t1']) == (1, g1['t_deadline'], g1['t_done'][0], None, g1['t1'],
                                                                 g1['prog_before_t1'], g1['cond25_before_deadline'], None)
    for c in ('g2', 'g3', 'g4', 'g5'):
        assert [f[f'{c}_{x}_chung'] for x in ('q', 'lo', 'hi', 'n_undef')] == [gt[c][x] for x in ('q', 'lo', 'hi', 'n_undef')]


def test_v82_run_error_has_no_gate_values(tmp_path):
    """T4: a run_error row (here: no DLL) has no gate value; the CSV cells are empty."""
    row, log = s3.safe_run(None, None, 'PID', dict(FAKE_DEFAULTS['PID']), 0.8, 0.5)
    assert row['status'] == 'run_error' and log is None and not (set(V82_COLS) | set(G1C_COLS)) & set(row)
    p = tmp_path / 'r.csv'
    s3.write_csv(str(p), [dict(controller='PID', point_id=0, box='centre', mu=0.5, v=0.8, **row)], s3.RUN_COLS)
    import csv as _csv
    r = next(_csv.DictReader(open(p)))
    assert all(r[c] == '' for c in V82_COLS + G1C_COLS)


@pytest.fixture(scope='module')
def v82_runs(path):
    """Default gains of both controllers at 0.5 and 0.8 m/s, and one detuned PID run (mu 0.5):
    [(ctrl, v, new row, log)]. The gate evaluator costs about 2-4 s a run."""
    if not os.path.isfile(_DLL):
        pytest.skip('fwsim.dll not built')
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    out = []
    for ctrl in ('PID', 'ADRC'):
        d = s3.defaults(fw, ctrl)
        for v in (0.5, 0.8):
            out.append((ctrl, v) + s3.run_one(fw, path, ctrl, d, v, 0.5))
    d = s3.defaults(fw, 'PID')
    out.append(('PID', 0.8) + s3.run_one(fw, path, 'PID', dict(d, kp_xy=0.15, pos_kp=0.6), 0.8, 0.5))
    return out


@needs_dll
def test_v82_v8_columns_keep_their_bits(path, v82_runs):
    """T1 (a): on the same log, the v8 scoring (evaluate_v7 alone, as blob 2127c69 did) gives every v8
    column of the new row bit for bit."""
    for ctrl, v, row, log in v82_runs:
        sch = o1.Schedule(v)
        p = fwsim.plant_params(s3.argparse.Namespace(ground='traction', mu=0.5))
        res, _ = o1.evaluate_v7(log, path, sch, lsum=p.lx + p.ly)
        old = {k: res[k] for k in V8_COLS[5:] if k in res}
        old.update(pos_reset_mid=fwsim.run_validity(log)['pos_reset_mid'])
        assert set(row) - set(V82_COLS) - set(G1C_COLS) == set(old), (ctrl, v)
        assert {k: s3._bits(row[k]) for k in old} == {k: s3._bits(old[k]) for k in old}, (ctrl, v)


@needs_dll
def test_v82_gate_maxima_are_the_v8_maxima(v82_runs):
    """T2: G2, G3, G5 q are the evaluate_v7 maxima and G4 q is e_bao_max, bit for bit; lo = hi = q;
    G1 follows completed; states from the thresholds."""
    states = set()
    for ctrl, v, row, _ in v82_runs:
        assert all(row[f'{c}_n_undef'] == 0 for c in Q_OF), (ctrl, v)
        for c, k in Q_OF.items():
            assert s3._bits(row[f'{c}_q']) == s3._bits(row[k]) == s3._bits(row[f'{c}_lo']) == \
                s3._bits(row[f'{c}_hi']), (ctrl, v, c)
            assert row[f'{c}_state'] == ('dat' if row[k] <= G.THRESHOLDS[c] else 'khong_dat'), (ctrl, v, c)
        assert row['g1_q'] == row['completed'] and row['g1_state'] == G.g1_state(row['completed'])
        assert row['gate_outcome'] == G.run_outcome(row[f'g{i}_state'] for i in range(1, 6))
        assert row['gate_u_t'] == 0.0 and row['gate_decide'] == 'g1,g2,g3,g4,g5'
        states |= {row[f'g{i}_state'] for i in range(1, 6)}
    assert {'dat', 'khong_dat'} <= states                 # the runs reach both sides of a threshold


@needs_dll
def test_v82_main_writes_the_gate_columns_and_meta(tmp_path, monkeypatch, capsys):
    """T5, T7: runs.csv, runs_test.csv headers are the v8 columns then the v8.2 columns; the sweep has
    them too (one scoring, Liam 14/11 'Co, mot cach cham'); meta.yaml records the gate and the evaluator."""
    import csv as _csv
    import hashlib as _h
    import json as _json
    code, out, _, _, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    for f in ('sealed/runs.csv', 'sealed_test/runs_test.csv'):
        # S03 v8.6: ngoai_tam_giac then the three kep_frac columns end the two sealed run files
        # only (2.1 last item, section 7, 11 item 6 (c))
        assert next(_csv.reader(open(out / f))) == \
            V8_COLS + V82_COLS + G1C_COLS + ['ngoai_tam_giac'] + s3.KEP_FRAC_COLS, f
    assert next(_csv.reader(open(out / 'sweep.csv'))) == list(s3.SWEEP_ID_COLS) + s3.RUN_COLS[5:]   # S3G2-3
    sweep_head = next(_csv.reader(open(out / 'sweep.csv')))
    assert 'ngoai_tam_giac' not in sweep_head and not set(s3.KEP_FRAC_COLS) & set(sweep_head)
    for f in ('sealed/runs.csv', 'sealed_test/runs_test.csv'):
        # S03 v8.6: the standard fake defaults never break the LADRC triangle
        assert {r['ngoai_tam_giac'] for r in _csv.DictReader(open(out / f))} == {'false'}
    # S3G-3: the VALUES reach the three files (fake gate fields differ per gains and speed)
    gains = {(d['controller'], d['point_id']): {n: float(d[f'{n}_value']) for n in s3.FACTORS[d['controller']]}
             for d in _csv.DictReader(open(out / 'sealed' / 'design.csv'))}        # v8.5 7: sealed
    n = 0
    for f in ('sealed/runs.csv', 'sealed_test/runs_test.csv'):
        for r in _csv.DictReader(open(out / f)):
            want = _fake_gate_fields(r['controller'], gains[(r['controller'], r['point_id'])], float(r['v']))
            cols = V82_COLS + G1C_COLS                                            # G1F-2: the B81-3 columns too
            assert {c: r[c] for c in cols} == {c: str(s3.fmt(want[c])) if want[c] is not None else '' for c in cols}, \
                (f, r['point_id'])
            n += 1
    for r in _csv.DictReader(open(out / 'sweep.csv')):
        g = dict(FAKE_DEFAULTS[r['controller']])
        if not r['factor'].startswith('('):
            g[r['factor']] = float(r['value'])
        want = _fake_gate_fields(r['controller'], g, float(r['v']))
        cols = V82_COLS + G1C_COLS
        assert {c: r[c] for c in cols} == {c: str(s3.fmt(want[c])) if want[c] is not None else '' for c in cols}, \
            (r['factor'], r['value'])
        n += 1
    assert n > 2 * (271 * 3 + 300)
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'source_check: "same at import, start and end of the run"' in meta   # S3G-1, S3G2-1
    assert meta.startswith('spec: S03 v8.6 ')                                # v8.6 (keeps the v8.2 gate)
    lines = meta.splitlines()
    gate = _json.loads(next(x for x in lines if x.startswith('gate: ')).split('   #')[0][len('gate: '):])
    assert gate['version'] == G.GATE_VERSION and gate['eval_version'] == E.EVAL_VERSION and gate['u_t'] == 0.0
    assert gate['decide'] == list(G.DECIDE_ALL) and gate['columns'] == V82_COLS
    ev = _json.loads(next(x for x in lines if x.startswith('evaluator_sha256: ')).split('   #')[0]
                     [len('evaluator_sha256: '):])
    assert set(ev) == {'fwsim/o1_feasibility.py', 'fwsim/o1_gate_eval.py', 'fwsim/o1_gate.py', 'tools/o1_lanes.py',
                       'rpi5/o1_tasks.py'}
    b = open(E.__file__, 'rb').read()
    assert ev['fwsim/o1_gate_eval.py'] == dict(sha256=_h.sha256(b).hexdigest(),
                                               sha256_lf=_h.sha256(b.replace(b'\r\n', b'\n')).hexdigest())


# ── after Codex round 1 (S3G-1..4; Liam 14/11 CODE_FEAS "Sua 5 muc + vong 2 (Recommended)") ──────────

@needs_dll
def test_v82_run_one_copies_the_gate_dict_it_was_given(path, monkeypatch):
    """S3G-4: through the real run_one: evaluate_fig8 gets the log, path, schedule and lsum; gate() gets the
    evaluator's frames and completed, the command maxima, u_t 0, DECIDE_ALL and the status; the row copies
    what gate() returned (marked values), never recomputing it."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    seen = {}
    real_eval, real_gate = E.evaluate_fig8, G.gate

    def spy_eval(log, p_, sch, lsum=0.24):
        out = real_eval(log, p_, sch, lsum=lsum)
        seen['eval'] = dict(path=p_, v=sch.v, lsum=lsum, frames=out[2], res=out[0])
        return out

    def spy_gate(frames, completed, **kw):
        seen['gate'] = dict(frames=frames, completed=completed, **kw)
        gt = real_gate(frames, completed, **kw)
        gt['g3'] = dict(gt['g3'], state='SPY_STATE', q=123.0)
        gt['outcome'] = 'SPY_OUTCOME'
        gt['version'] = 'SPY_VERSION'
        return gt
    monkeypatch.setattr(E, 'evaluate_fig8', spy_eval)
    monkeypatch.setattr(G, 'gate', spy_gate)
    g = s3.defaults(fw, 'PID')
    row, _ = s3.run_one(fw, path, 'PID', g, 0.8, 0.5)
    p = fwsim.plant_params(s3.argparse.Namespace(ground='traction', mu=0.5))
    assert seen['eval']['path'] is path and seen['eval']['v'] == 0.8 and seen['eval']['lsum'] == p.lx + p.ly
    sg = seen['gate']
    assert sg['frames'] is seen['eval']['frames'] and sg['completed'] == seen['eval']['res']['completed']
    assert sg['u_t'] == 0.0 and sg['decide'] == G.DECIDE_ALL and sg['status'] == seen['eval']['res']['status']
    assert (sg['v_cmd_max'], sg['wz_cmd_max']) == G.fig8_cmd_max(o1.Schedule(0.8))
    assert 'u_p' not in sg and 'u_psi_deg' not in sg                     # the defaults 0 of gate()
    assert row['g3_state'] == 'SPY_STATE' and row['g3_q'] == 123.0 and row['gate_outcome'] == 'SPY_OUTCOME'
    assert row['gate_version'] == 'SPY_VERSION' and row['e_bao_max'] == seen['eval']['res']['e_bao_max']


@needs_dll
@pytest.mark.parametrize('where', ['evaluate_fig8', 'fig8_cmd_max', 'gate'])
def test_v82_a_scoring_failure_is_a_run_error(path, monkeypatch, where):
    """S3G-4: a failure in the evaluator, the command maxima (cache emptied) or the gate is a run_error
    row with no gate value (S03 then stops, B8b-20)."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')

    def boom(*a, **k):
        raise ValueError(f'boom in {where}')
    monkeypatch.setattr(s3, '_CMD_MAX', {})
    monkeypatch.setattr(E if where == 'evaluate_fig8' else G, where, boom)
    row, log = s3.safe_run(fw, path, 'PID', s3.defaults(fw, 'PID'), 0.8, 0.5)
    assert row['status'] == 'run_error' and log is None and row['error'] == f'ValueError: boom in {where}'
    assert not set(V82_COLS) & set(row)


@needs_dll
@pytest.mark.parametrize('mode', ['default', 'factor'])
def test_v82_sweep_run_error_is_a_sweep_row(tmp_path, monkeypatch, capsys, mode):
    """S3G-2: a run_error in the sweep is a row of sweep.csv (gate cells empty) before S03 stops; the
    header holds every run column even when the first run fails."""
    import csv as _csv
    code, out, console, _, _ = _run_main(tmp_path, monkeypatch, capsys, error_in_sweep=mode)
    assert code == 2 and '[STOP]' in console
    head = next(_csv.reader(open(out / 'sweep.csv')))
    assert head == list(s3.SWEEP_ID_COLS) + s3.RUN_COLS[5:]                      # S3G2-3: one fixed header
    assert next(_csv.reader(open(out / 'levels.csv', encoding='utf-8'))) == s3.LEVEL_COLS   # S83-1
    rows = list(_csv.DictReader(open(out / 'sweep.csv')))
    err = [r for r in rows if r['status'] == 'run_error']
    assert len(err) == 1 and err[0]['error'] == 'FakeSweepError: boom' and err[-1] == rows[-1]
    assert all(err[0][c] == '' for c in V82_COLS)
    if mode == 'default':
        assert len(rows) == 1 and err[0]['factor'] == '(default check)' and err[0]['is_default'] == 'true'
    else:
        f0 = s3.FACTORS['PID'][0]
        assert err[0]['factor'] == f0 and err[0]['is_default'] == 'false' and err[0]['value'] != ''
        assert all(r['status'] == 'ok' for r in rows[:-1])


@needs_dll
def test_v82_sources_taken_at_the_start_and_checked_at_the_end(tmp_path, monkeypatch, capsys):
    """S3G-1: script and evaluator are recorded as they were before the first run; a change by the end
    is recorded in source_check, said on the console and ends with exit code 2."""
    calls = dict(git=0, ev=0)
    real_git, real_ev = s3.git_state, s3.evaluator_sha256

    def git_state(repo):
        calls['git'] += 1
        return dict(real_git(repo), commit=f'commit{calls["git"]}')

    def ev():
        calls['ev'] += 1
        return dict(real_ev(), marker=calls['ev'])
    monkeypatch.setattr(s3, 'git_state', git_state)
    monkeypatch.setattr(s3, 'evaluator_sha256', ev)
    code, out, console, _, _ = _run_main(tmp_path, monkeypatch, capsys)
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert calls == dict(git=2, ev=2) and code == 2
    assert '"commit": "commit1"' in next(x for x in meta.splitlines() if x.startswith('script: '))
    ev_line = next(x for x in meta.splitlines() if x.startswith('evaluator_sha256: '))
    import json as _json
    assert 'marker' not in ev_line and _json.dumps(s3.EVAL_SHA_AT_IMPORT) in ev_line  # S3G2-1: as imported
    sc = next(x for x in meta.splitlines() if x.startswith('source_check: '))
    assert 'CHANGED' in sc and '"commit": "commit2"' in sc and 'S3G-1' in meta and 'source_check' in console


@needs_dll
def test_v82_evaluator_changed_before_the_start_is_caught(tmp_path, monkeypatch, capsys):
    """S3G2-1: the evaluator hashed at import differs from the files at the start (changed in between):
    recorded as CHANGED, exit 2, although start and end agree."""
    monkeypatch.setattr(s3, 'EVAL_SHA_AT_IMPORT', dict(s3.EVAL_SHA_AT_IMPORT, marker='import'))
    code, out, console, _, _ = _run_main(tmp_path, monkeypatch, capsys)
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    sc = next(x for x in meta.splitlines() if x.startswith('source_check: '))
    assert code == 2 and 'CHANGED' in sc and '"marker": "import"' in sc and 'source_check' in console


def test_v82_untracked_files_are_not_a_source_change(tmp_path):
    """S3G2-2: an untracked file (e.g. an output folder inside the repo) leaves the snapshot key equal; a
    change of a tracked file does not."""
    import subprocess as _sp
    repo = str(tmp_path / 'repo')
    os.makedirs(repo)

    def git(*a):
        _sp.run(['git', '-C', repo, *a], check=True, capture_output=True)
    git('init', '-q')
    git('config', 'user.email', 't@t')
    git('config', 'user.name', 't')
    (tmp_path / 'repo' / 'a.py').write_text('x = 1\n')
    git('add', 'a.py')
    git('commit', '-q', '-m', 'a')
    k0 = s3.source_snapshot(repo)['key']
    os.makedirs(os.path.join(repo, 'out', 'sealed'))
    (tmp_path / 'repo' / 'out' / 'sealed' / 'runs.csv').write_text('a\n')
    s1 = s3.source_snapshot(repo)
    assert s1['key'] == k0 and s1['script']['dirty']          # meta still says untracked files are there
    (tmp_path / 'repo' / 'a.py').write_text('x = 2\n')
    k2 = s3.source_snapshot(repo)['key']
    assert k2 != k0 and k2['tracked'] == ['a.py'] and k2['commit'] == k0['commit']


# ── S03 v8.3: extended sweep grid (design 03_CODE/docs/2026-11-15_Thiet_ke_S03_v83_luoi_noi_v1.md) ─────

import tune_ga as _tg  # noqa: E402

GRID_CASES = [('kp_xy', 1.5, 0.0, 6.0), ('ki_xy', 0.05, 0.0, 10.0), ('pos_kp', 6.0, 0.0, 12.0),
              ('k_psi', 1.0, 0.0, 4.0), ('omega_c_xy', 8.0, 1.0, 25.0), ('omega_o_xy', 30.0, 5.0, 48.0),
              ('kp_wz', 0.37, 0.0, 6.0),        # default off the log points
              ('kp_xy', 0.6, 0.0, 6.0),         # b = hi = 10 x default: the default is the middle log point
              ('omega_c_wz', 0.5, 1.0, 25.0)]   # a = default / 10 < lo


# S83-3: sweep_grid of the v8 code (s03_doe.py at hil 30d8157, blob f98effa) on GRID_CASES, IEEE-754 bytes
# (little endian) of each point; written once from that code, never from the code under test
V8_GRID_BITS = {
    ('kp_xy', 1.5, 0.0, 6.0): '0000000000000000 333333333333c33f 310d3228116ece3f 7228cc4b361dd83f 87db97dff01be33f b53e722f3449ee3f 000000000000f83f 40481fb9ca040340 9132bfde83240e40 69d27d17ede21740 3167a79dc0ed2240 0000000000002e40',
    ('ki_xy', 0.05, 0.0, 10.0): '0000000000000000 7b14ae47e17a743f c8db1863dee5853f e6f57620fd69973f cd9cdbedfa08a93f 9a9999999999a93f eea82d10b4c4ba3f d9edbfc5259fcc3f 4779c18e709ade3f 9338124f6d5cf03f aa5ee43a697e0140 8da874d878b41240 0000000000002440',
    ('pos_kp', 6.0, 0.0, 12.0): '0000000000000000 333333333333e33f 2f0d3228116eee3f 7228cc4b361df83f 87db97dff01b0340 b53e722f34490e40 0000000000001840 40481fb9ca042340 9132bfde83242e40 69d27d17ede23740 3167a79dc0ed4240 0000000000004e40',
    ('k_psi', 1.0, 0.0, 4.0): '0000000000000000 9a9999999999b93f 75b376c56049c43f 4c70dd877913d03f b424757f967ad93f 79d4f674cd30e43f 000000000000f03f 5560d4f6b85bf93f 61ccd4e957180440 e16d521f3cd90f40 978934d2003d1940 0000000000002440',
    ('omega_c_xy', 8.0, 1.0, 25.0): '9a9999999999e93f 76b376c56049f43f 4c70dd8779130040 b524757f967a0940 79d4f674cd301440 0000000000002040 5560d4f6b85b2940 60ccd4e957183440 e06d521f3cd93f40 968934d2003d4940 0000000000005440',
    ('omega_o_xy', 30.0, 5.0, 48.0): '0000000000000840 23fd957d0eab0f40 d8e824a4a9e41440 b159184a9a911b40 c602c60248302240 0100000000002840 23fd957d0eab2f40 d8e824a4a9e43440 af59184a9a913b40 0000000000003e40 c502c60248304240 0000000000004840',
    ('kp_wz', 0.37, 0.0, 6.0): '0000000000000000 8b6ce7fba9f1a23f d55dc910e682af3f 606e10733635ba3f d1d7d39f07ccc53f 44f68162e420d23f ae47e17a14aed73f f7895103a127de3f a266b3bd6314e93f cba12a0cd1dbf43f c49cfb8c1b590140 a4cca5094fdb0c40 0000000000001840',
    ('kp_xy', 0.6, 0.0, 6.0): '0000000000000000 b81e85eb51b8ae3f 26a4c1b90d58b83f 28ed3cd6914ac33f 3f5f59ff1a93ce3f f7fef425903ad83f 333333333333e33f 330d3228116eee3f 7428cc4b361df83f 87db97dff01b0340 b53e722f34490e40 0000000000001840',
    ('omega_c_wz', 0.5, 1.0, 25.0): '9a9999999999a93f 2db3457d3dd4b73f c3c117183e2ec63f 4f01970270a5d43f 000000000000e03f 33072641ce37e33f a7f4979b77e3f13f 436bd723aca60040 0864bbd095ff0e40 7146e49e9fda1c40 ea50bf30a3db2a40 0000000000003940',
}


@pytest.mark.parametrize('name, d, lo, hi', GRID_CASES)
def test_v83_old_points_match_the_v8_code(name, d, lo, hi):
    """S83-3: sweep_grid and the first points of sweep_grid_n are the bytes of the v8 code (30d8157)."""
    want = V8_GRID_BITS[(name, d, lo, hi)].split()
    grid, info = s3.sweep_grid_n(name, d, lo, hi)
    assert [struct.pack('<d', x).hex() for x in s3.sweep_grid(name, d, lo, hi)] == want
    assert [struct.pack('<d', x).hex() for x in grid[:len(want)]] == want and len(grid) == len(want) + info['K']


@pytest.mark.parametrize('d, K', [
    (0.3162277664121227, 4),     # r^4 = 9.999999995: at or above 10 (1 - 1e-9) -> K = 4 (no tolerance would give 5)
    (0.3162277699696851, 5),     # r^4 = 9.99999995: below 10 (1 - 1e-9) -> K = 5 (a tolerance 1e-8 would give 4)
])
def test_v83_K_on_both_sides_of_the_tolerance(d, K):
    """S83-4 (Codex counterexample): PID ki_xy with hi 10, a = d/10, b = 10, r^4 = (100/d)^0.4 near 10."""
    _, info = s3.sweep_grid_n('ki_xy', d, 0.0, 10.0)
    assert info['K'] == K and (info['r'] ** 4 >= 10 * (1 - 1e-9)) == (K == 4)


@pytest.mark.parametrize('name, d, lo, hi', GRID_CASES)
def test_v83_old_points_kept_bit_for_bit(name, d, lo, hi):
    """T1: the first points of sweep_grid_n are sweep_grid, bit for bit (11 log points, default, 0)."""
    base = s3.sweep_grid(name, d, lo, hi)
    grid, info = s3.sweep_grid_n(name, d, lo, hi)
    assert [s3._bits(x) for x in grid[:len(base)]] == [s3._bits(x) for x in base]
    assert grid[len(base):] == info['noi'] and info['n_points'] == len(grid) and grid == sorted(set(grid))
    assert d in grid and (0.0 in grid) == (lo == 0.0)
    a = d / 10.0 if lo == 0.0 else min(d / 10.0, lo)
    assert info['a'] == a and info['b'] == max(base) and info['a'] == min(x for x in base if x > 0)


@pytest.mark.parametrize('name, d, lo, hi', GRID_CASES)
def test_v83_extension_definition(name, d, lo, hi):
    """T2: r = (b/a)^(1/10); points b r^k, k = 1..K, K the smallest with r^K >= 10 (1 - 1e-9); omega_o not
    extended."""
    _, info = s3.sweep_grid_n(name, d, lo, hi)
    r = (info['b'] / info['a']) ** 0.1
    assert info['r'] == r
    if name.startswith('omega_o'):
        assert info['K'] == 0 and info['b_n'] is None and info['noi'] == []
        return
    K = info['K']
    assert K >= 1 and r ** K >= 10 * (1 - 1e-9) and not r ** (K - 1) >= 10 * (1 - 1e-9)
    assert info['noi'] == [info['b'] * r ** k for k in range(1, K + 1)] and info['b_n'] == info['noi'][-1]
    assert info['b_n'] >= 10 * info['b'] * (1 - 1e-9) and info['b_n'] < 10 * info['b'] * r


def test_v83_spec_example_pid_ki_xy():
    """Spec S03 v8.3 3.1: PID ki_xy, a 0.005, b 10, K = 4, last point about 209."""
    _, info = s3.sweep_grid_n('ki_xy', 0.05, 0.0, 10.0)
    assert (info['a'], info['b'], info['K']) == (0.005, 10.0, 4) and 205 < info['b_n'] < 213


@pytest.mark.parametrize('ok_lo, ok_hi, want', [
    (None, None, ('0', 'hết lưới')),                                       # everything completes (v8.4: H = b)
    (2, None, ('điểm kế tiếp không hoàn thành', 'hết lưới')),
    (None, -3, ('0', 'điểm kế tiếp không hoàn thành')),
])
def test_v84_level_sources_lo_zero(ok_lo, ok_hi, want):
    """T3 (v8.4): sources on the original grid with the 0 point (lo = 0); ok_lo: first index that completes;
    ok_hi: index past the last that completes."""
    grid, info = s3.sweep_grid_n('kp_xy', 1.5, 0.0, 6.0)
    base = s3.base_grid(grid, info)
    assert base == s3.sweep_grid('kp_xy', 1.5, 0.0, 6.0) and len(grid) - len(base) == info['K'] > 0
    ok = [True] * len(base)
    if ok_lo:
        ok[:ok_lo] = [False] * ok_lo
    if ok_hi:
        ok[ok_hi:] = [False] * -ok_hi
    L, H = s3.levels_from_sweep(base, 1.5, ok)
    assert s3.level_sources(base, L, H, info, 'kp_xy') == want
    assert H <= info['b']


def test_v84_level_sources_lower_end_and_omega_cap():
    """T3: 'hết lưới dưới' (L = a > 0, no lower point), 'trần ω_o' (omega_o at b = 48, first over 'hết lưới');
    an omega_o grid that does not reach 48 ends with 'hết lưới'."""
    grid, info = s3.sweep_grid_n('omega_o_xy', 30.0, 5.0, 48.0)
    base = s3.base_grid(grid, info)
    assert base == grid
    L, H = s3.levels_from_sweep(base, 30.0, [True] * len(base))
    assert L == base[0] > 0 and H == 48.0
    assert s3.level_sources(base, L, H, info, 'omega_o_xy') == ('hết lưới dưới', 'trần ω_o')
    ok = [True] * len(base)
    ok[-1] = False
    L, H = s3.levels_from_sweep(base, 30.0, ok)
    assert s3.level_sources(base, L, H, info, 'omega_o_xy')[1] == 'điểm kế tiếp không hoàn thành'
    grid, info = s3.sweep_grid_n('omega_o_wz', 3.0, 1.0, 20.0)            # b = 30 < 48: no cut
    assert info['b'] < s3.OMEGA_O_MAX
    L, H = s3.levels_from_sweep(grid, 3.0, [True] * len(grid))
    assert s3.level_sources(grid, L, H, info, 'omega_o_wz')[1] == 'hết lưới'


def test_v84_levels_ignore_the_extension():
    """S03 v8.4: extension points that complete never move H above b (the level rule runs on the original grid)."""
    grid, info = s3.sweep_grid_n('ki_xy', 0.05, 0.0, 10.0)
    base = s3.base_grid(grid, info)
    assert grid[len(base):] == info['noi'] and min(info['noi']) > info['b'] == base[-1]
    L, H = s3.levels_from_sweep(base, 0.05, [True] * len(base))
    assert H == info['b'] < info['b_n']
    with pytest.raises(s3.S03Stop) as e:
        s3.levels_from_sweep(base, 0.05, [False] * len(base))
    assert e.value.case == '(i)'


# ── S03 v8.4 out-of-range check ───────────────────────────────────────────────────

def _kr(completed='true', status='ok', e=0.05, tv=1.0):
    return dict(completed=completed, status=status, e_perp_max=e, tv_duty=tv)


GATE3 = (0.2, 0.5, 0.8)
THR = {0.2: 0.2, 0.5: 0.2, 0.8: 0.2}


def _w(base, upd):
    return {**base, **upd}


def _all(**k):
    return {v: _kr(**k) for v in GATE3}


@pytest.mark.parametrize('pt, bd, gate, thr, want', [
    # step 1: not completing (concluded, status ok) at one gate speed, even if another speed is not comparable
    (_w(_all(), {0.2: _kr(completed='false'), 0.8: _kr(status='track_err')}), _all(), GATE3, THR,
     ('không cờ', 1, False)),
    # a 'false' with track_err is not step 1: not comparable, step 2
    (_w(_all(), {0.2: _kr(completed='false', status='track_err')}), _all(), GATE3, THR, ('chưa kết luận', 2, False)),
    # step 2: inconclusive y1, missing run of the level, non-finite threshold, non-finite e
    (_w(_all(), {0.5: _kr(completed='inconclusive')}), _all(), GATE3, THR, ('chưa kết luận', 2, False)),
    (_all(), _w(_all(), {0.8: None}), GATE3, THR, ('chưa kết luận', 2, False)),
    (_all(), _all(), GATE3, {0.2: 0.2, 0.5: float('nan'), 0.8: 0.2}, ('chưa kết luận', 2, False)),
    (_w(_all(), {0.5: _kr(e=float('nan'))}), _all(), GATE3, THR, ('chưa kết luận', 2, False)),
    # a speed out of the gate does not count: its 'false' is not step 1, its bad level run is not step 2
    (_w(_all(e=0.0), {0.2: _kr(completed='false')}), _w(_all(e=0.01), {0.2: None}), (0.5, 0.8), THR,
     ('cờ', 3, False)),
    # step 3 by e: lower by exactly 0.01 at one speed, not higher elsewhere
    (_w(_all(e=0.01), {0.5: _kr(e=0.0)}), _all(e=0.01), GATE3, THR, ('cờ', 3, False)),
    # step 3 by Y3: tv lower by the threshold, e the same
    (_w(_all(), {0.8: _kr(tv=0.75)}), _all(), GATE3, THR, ('cờ', 3, False)),
    # threshold 0: a lower tv does not flag; a higher tv is worse
    (_w(_all(), {0.8: _kr(tv=0.5)}), _all(), GATE3, {0.2: 0.0, 0.5: 0.0, 0.8: 0.0}, ('không cờ', 4, False)),
    # higher by exactly 0.01 m is not worse: the e lower elsewhere flags
    (_w(_all(e=0.0), {0.2: _kr(e=0.01)}), _w(_all(e=0.01), {0.2: _kr(e=0.0)}), GATE3, THR, ('cờ', 3, False)),
    # step 4: better at one speed but worse (tv above the threshold) at another; bao_rieng
    (_w(_all(), {0.5: _kr(e=0.03), 0.8: _kr(tv=1.5)}), _all(), GATE3, THR, ('không cờ', 4, True)),
    # step 4: nothing better
    (_all(), _all(), GATE3, THR, ('không cờ', 4, False)),
    # S84-1: inclusive bounds between two non-zero binary64 numbers (0.03 - 0.02 = 0.009999999999999998)
    (_w(_all(e=0.03), {0.5: _kr(e=0.02)}), _all(e=0.03), GATE3, THR, ('cờ', 3, False)),
    (_w(_all(e=0.03), {0.2: _kr(e=0.04), 0.5: _kr(e=0.02)}), _all(e=0.03), GATE3, THR, ('cờ', 3, False)),
    (_w(_all(), {0.8: _kr(tv=0.8)}), _all(), GATE3, THR, ('cờ', 3, False)),             # 1.0 - 0.8 against 0.2
    (_w(_all(e=0.03), {0.5: _kr(e=0.0201)}), _all(e=0.03), GATE3, THR, ('không cờ', 4, False)),   # just short
    # S84V2-1: the rounding allowance is one subtraction's, not relative: 1e-14 m short of 0.01 is not 'lower'
    (_w(_all(e=0.03), {0.5: _kr(e=0.02000000000001)}), _all(e=0.03), GATE3, THR, ('không cờ', 4, False)),
    # S84V2-1: equal responses with a tiny positive Y3 threshold are not 'lower by the threshold'
    (_all(), _all(), GATE3, {0.2: 1e-13, 0.5: 1e-13, 0.8: 1e-13}, ('không cờ', 4, False)),
    (_w(_all(e=0.03), {0.5: _kr(e=0.02), 0.8: _kr(e=0.03, tv=1.5)}), _all(e=0.03), GATE3, THR,
     ('không cờ', 4, True)),
])
def test_v84_classify_point_steps(pt, bd, gate, thr, want):
    assert s3.classify_point(pt, bd, gate, thr) == want


@pytest.mark.parametrize('fn, x, ref, thr, want', [
    # S84V2-1: inclusive bounds within one subtraction's rounding, no wider
    ('lower', 1.0, 1.0, 1e-13, False),                  # equal: not lower by a positive threshold
    ('lower', 0.02000000000001, 0.03, 0.01, False),     # 1e-14 m short
    ('lower', 0.02, 0.03, 0.01, True),                  # 0.03 - 0.02 = 0.009999999999999998
    ('lower', 0.8, 1.0, 0.2, True),                     # Y3: 1.0 - 0.8 = 0.19999999999999996
    ('not_above', 0.04, 0.03, 0.01, True),              # 0.04 - 0.03 = 0.010000000000000002
    ('not_above', 0.04000000000001, 0.03, 0.01, False),  # 1e-14 m over
    ('not_above', 1.0 + 1e-13, 1.0, 0.0, False),        # threshold 0: higher is higher
    # S84V3-1: exact, no tolerance at all
    ('lower', 1.0, 1.0, 0.2 * 5e-16, False),            # equal, threshold about 1e-16
    ('lower', 0.020000000000000007, 0.03, 0.01, False),  # 2 ULP short
    ('not_above', 0.040000000000000015, 0.03, 0.01, False),  # 2 ULP over
    ('not_above', 1.0000000000000002, 1.0, 0.0, False),  # 1 ULP over, threshold 0
    ('lower', 0.0, 0.0, 5e-324, False),                 # zeros, the smallest positive threshold
    ('lower', 0.0, 0.01, 0.01, True),
    ('not_above', 0.0, 0.0, 0.0, True),
])
def test_v84_bound_helpers(fn, x, ref, thr, want):
    f = s3.lower_by_at_least if fn == 'lower' else s3.not_above_by
    assert f(x, ref, thr) is want


def test_v86_y3_threshold_is_absolute():
    """S03 v8.6 section 5.2: the Y3 threshold is the exact decimal 0.4 s^-1 at every speed, mu,
    worst and range (before: the exact 0.2 x tv_duty of the default-gain run); Y3_REL is gone."""
    from fractions import Fraction
    assert s3.Y3_ABS == 0.4 and s3.THRESH['tv_duty'] == 0.4 and not hasattr(s3, 'Y3_REL')
    assert s3.y3_threshold() == Fraction(2, 5)
    thr = {v: s3.y3_threshold() for v in GATE3}
    # tv lower by exactly 0.4 at one speed: flag (binary64 0.7 - 0.3 is 0.39999999999999997; exact: 0.4)
    assert s3.classify_point(_w(_all(tv=0.7), {0.8: _kr(tv=0.3)}), _all(tv=0.7), GATE3, thr) == ('cờ', 3, False)
    # tv lower by 0.39: no flag
    assert s3.classify_point(_w(_all(tv=0.7), {0.8: _kr(tv=0.31)}), _all(tv=0.7), GATE3,
                             thr) == ('không cờ', 4, False)
    # tv higher by exactly 0.4 at 0.2 is not worse (binary64 1.1 - 0.7 is 0.40000000000000013, exact 0.4):
    # the e lower at 0.5 flags
    assert s3.classify_point(_w(_all(tv=0.7), {0.2: _kr(tv=1.1), 0.5: _kr(e=0.0, tv=0.7)}), _all(tv=0.7), GATE3,
                             thr) == ('cờ', 3, False)
    # tv higher by 0.41 at 0.2 is worse: no flag; the e lower at 0.5 is bao_rieng
    assert s3.classify_point(_w(_all(tv=0.7), {0.2: _kr(tv=1.11), 0.5: _kr(e=0.0, tv=0.7)}), _all(tv=0.7), GATE3,
                             thr) == ('không cờ', 4, True)


def _kiem_rows(ctrl, grids, fn, mu=0.5):
    """Sweep rows of every grid point at every speed, fields from fn(name, x, v)."""
    out = []
    for (c, name), (grid, _) in grids.items():
        if c != ctrl:
            continue
        for x in grid:
            for v in s3.SPEEDS:
                out.append(dict(controller=c, factor=name, value=x, v=v, mu=mu, **fn(name, x, v)))
    return out


def _kiem_setup(ctrl='PID'):
    lo_b, hi_b = _tg.BOUNDS[ctrl]
    grids = {(ctrl, n): s3.sweep_grid_n(n, FAKE_DEFAULTS[ctrl][n], lo, hi)
             for n, lo, hi in zip(s3.FACTORS[ctrl], lo_b, hi_b)}
    levels = {ctrl: {}}
    for (c, n), (grid, info) in grids.items():
        base = s3.base_grid(grid, info)
        levels[c][n] = (base[0], base[-1], 'lin')
    y3 = {(ctrl, v): 1.0 for v in s3.SPEEDS}
    return grids, levels, y3


def test_v84_check_totals_and_rows():
    """Every extension point is outside [L, H] = the original grid ends and compared with H; a factor whose L is
    above the lowest point has its lower points compared with L; the totals count them."""
    grids, levels, y3 = _kiem_setup()
    grid, info = grids[('PID', 'kp_xy')]
    base = s3.base_grid(grid, info)
    levels['PID']['kp_xy'] = (base[2], base[-1], 'lin')               # two original points below L
    ki_noi = grids[('PID', 'ki_xy')][1]['noi']

    def fn(name, x, v):
        if name == 'ki_xy' and x == ki_noi[0]:
            return _kr(e=0.03)                                        # better by 0.02 m: flag
        if name == 'ki_xy' and x == ki_noi[1]:
            return _kr(status='track_err')                            # not comparable
        return _kr()
    rows, summ = s3.out_of_range_check(_kiem_rows('PID', grids, fn), levels, grids, FAKE_DEFAULTS, ['PID'],
                                       list(s3.SPEEDS), 0.5, y3, None)
    pts = [r for r in rows if r['dong'] == 'diem']
    tot = [r for r in rows if r['dong'] == 'tong']
    n_noi = sum(i['n_noi'] for _, i in grids.values())
    assert len(pts) == n_noi + 2 and len(tot) == 1
    low = [r for r in pts if r['factor'] == 'kp_xy' and r['can'] == 'L']
    assert [r['value'] for r in low] == base[:2] and all(r['can_value'] == base[2] for r in low)
    assert {r['luoi'] for r in low} == {'0', 'log'}
    assert all(r['can'] == 'H' and r['luoi'] == 'noi' for r in pts if r not in low)
    by = {(r['factor'], r['value']): r for r in pts}
    assert by[('ki_xy', ki_noi[0])]['ket_qua'] == 'cờ' and by[('ki_xy', ki_noi[0])]['buoc'] == 3
    assert by[('ki_xy', ki_noi[1])]['ket_qua'] == 'chưa kết luận'
    assert by[('ki_xy', ki_noi[0])]['e_diem_v0.5'] == 0.03 and by[('ki_xy', ki_noi[0])]['e_can_v0.5'] == 0.05
    assert by[('ki_xy', ki_noi[0])]['nguong_tv_v0.5'] == pytest.approx(0.4)  # S03 v8.6: absolute 0.4 s^-1
    t = tot[0]
    assert t['ket_qua'] == 'đã kiểm đủ' and t['n_diem_ngoai'] == t['n_da_kiem'] == n_noi + 2
    assert (t['n_co'], t['n_chua_ket_luan'], t['n_khong_co'], t['n_chua_kiem']) == (1, 1, n_noi, 0)
    assert summ['PID']['co'] == [f'ki_xy={ki_noi[0]!r}']
    assert set(s3.KIEM_COLS) >= set(k for r in rows for k in r)


def test_v86_check_y3_threshold_absolute():
    """S03 v8.6 section 5.2: out_of_range_check takes the Y3 threshold as the exact decimal 0.4
    s^-1 at every speed whatever y3_ref holds (binary64 0.7 - 0.3 is 0.39999999999999997, the
    exact decimals give 0.4); the CSV column holds it as a float."""
    grids, levels, y3 = _kiem_setup()
    ki_noi = grids[('PID', 'ki_xy')][1]['noi']

    def fn(name, x, v):
        if name == 'ki_xy' and x == ki_noi[0] and v == 0.8:
            return _kr(tv=0.3)                              # 0.4 below the level's 0.7: flag
        return _kr(tv=0.7)
    args = (_kiem_rows('PID', grids, fn), levels, grids, FAKE_DEFAULTS, ['PID'], list(s3.SPEEDS), 0.5)
    rows, _ = s3.out_of_range_check(*args, {k: 0.9 for k in y3}, None)
    rows2, _ = s3.out_of_range_check(*args, {k: 123.0 for k in y3}, None)
    rows3, _ = s3.out_of_range_check(*args, {}, None)
    assert rows == rows2 == rows3                           # S03 v8.6: the reference is not used any more
    by = {(r['factor'], r['value']): r for r in rows if r['dong'] == 'diem'}
    assert (by[('ki_xy', ki_noi[0])]['ket_qua'], by[('ki_xy', ki_noi[0])]['buoc']) == ('cờ', 3)
    assert by[('ki_xy', ki_noi[0])]['nguong_tv_v0.8'] == 0.4
    assert type(by[('ki_xy', ki_noi[0])]['nguong_tv_v0.8']) is float
    assert by[('ki_xy', ki_noi[1])]['ket_qua'] == 'không cờ'


def test_v84_check_gate_speeds_only_and_bao_rieng():
    grids, levels, y3 = _kiem_setup()
    k_noi = grids[('PID', 'k_psi')][1]['noi']

    def fn(name, x, v):
        if name == 'k_psi' and x == k_noi[0]:
            return _kr(e=0.02) if v == 0.5 else _kr(tv=2.0) if v == 0.8 else _kr(completed='false')
        return _kr()
    rows, summ = s3.out_of_range_check(_kiem_rows('PID', grids, fn), levels, grids, FAKE_DEFAULTS, ['PID'],
                                       [0.5, 0.8], 0.5, y3, None)
    r = next(x for x in rows if x['dong'] == 'diem' and x['factor'] == 'k_psi' and x['value'] == k_noi[0])
    assert (r['ket_qua'], r['buoc'], r['bao_rieng']) == ('không cờ', 4, True)     # 0.2 is out of the gate
    assert r['trong_cong_v0.2'] is False and r['trong_cong_v0.5'] is True
    assert summ['PID']['bao_rieng'] == [f'k_psi={k_noi[0]!r}'] and summ['PID']['n_bao_rieng'] == 1


def test_v84_check_part_way_and_not_checked():
    grids, levels, y3 = _kiem_setup()
    del levels['PID']['pos_ki']
    rows, summ = s3.out_of_range_check(_kiem_rows('PID', grids, lambda *a: _kr()), levels, grids, FAKE_DEFAULTS,
                                       ['PID', 'ADRC'], list(s3.SPEEDS), 0.5, y3, None)
    tot = {r['controller']: r for r in rows if r['dong'] == 'tong'}
    assert tot['PID']['ket_qua'] == 'kiểm dở' and tot['PID']['n_gain_chua_kiem'] == 1
    assert tot['PID']['n_chua_kiem'] == len(grids[('PID', 'pos_ki')][0])
    assert tot['ADRC']['ket_qua'] == 'kiểm dở' and tot['ADRC']['n_gain_chua_kiem'] == 7      # sweep not reached
    assert 'grid not known' in tot['ADRC']['ghi_chu'] and tot['ADRC']['n_da_kiem'] == 0
    for case in ('(i)', '(ii)'):
        rows, summ = s3.out_of_range_check(_kiem_rows('PID', grids, lambda *a: _kr()), levels, grids,
                                           FAKE_DEFAULTS, ['PID', 'ADRC'], list(s3.SPEEDS), 0.5, y3, case)
        assert [r['ket_qua'] for r in rows] == ['không kiểm', 'không kiểm']
        assert summ['PID']['n_da_kiem'] == 0


def test_v83_grid_labels():
    grid, info = s3.sweep_grid_n('kp_xy', 0.6, 0.0, 6.0)        # the default replaced the middle log point
    lab = [s3.grid_label(x, 0.6, info['noi']) for x in grid]
    assert lab.count('mac_dinh') == 1 and lab.count('0') == 1 and lab.count('noi') == info['K']
    assert lab.count('log') == len(grid) - 2 - info['K'] == 10       # 11 log points, one of them the default


@needs_dll
def test_v83_main_writes_levels_sources_luoi_and_grid_meta(tmp_path, monkeypatch, capsys):
    """T4: levels.csv a, b, b_n, L_nguon, H_nguon after the old columns; sweep.csv luoi per row and one row
    per grid point and speed; design decoded within the new levels; meta spec v8.3 and the grid block."""
    import csv as _csv
    import json as _json
    code, out, _, _, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    head = next(_csv.reader(open(out / 'levels.csv', encoding='utf-8')))
    assert head == ['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note', 'a', 'b', 'b_n',
                    'L_nguon', 'H_nguon']
    lv = {(r['controller'], r['factor']): r for r in _csv.DictReader(open(out / 'levels.csv', encoding='utf-8'))}
    sweep = list(_csv.DictReader(open(out / 'sweep.csv', encoding='utf-8')))
    for ctrl in s3.CONTROLLERS:
        lo_b, hi_b = _tg.BOUNDS[ctrl]
        for name, lo, hi in zip(s3.FACTORS[ctrl], lo_b, hi_b):
            d = FAKE_DEFAULTS[ctrl][name]
            grid, info = s3.sweep_grid_cut(ctrl, name, d, lo, hi)      # S03 v8.6: cut at the ceiling
            base = s3.base_grid(grid, info)
            r = lv[(ctrl, name)]
            assert float(r['a']) == info['a'] and float(r['b']) == info['b']
            assert r['b_n'] == ('' if info['b_n'] is None else repr(info['b_n']))
            # the fake completes every sweep run: L, H are the ends of the cut original grid
            assert float(r['L']) == base[0] and float(r['H']) == base[-1] == min(info['b'], info['T'])
            assert (r['L_nguon'], r['H_nguon']) == s3.level_sources(base, base[0], base[-1], info, name)
            rows = [x for x in sweep if x['controller'] == ctrl and x['factor'] == name]
            assert len(rows) == len(grid) * len(s3.SPEEDS)
            for x in rows:
                assert x['luoi'] == s3.grid_label(float(x['value']), d, info['noi'])
                # S03 v8.6: tran is true at the point equal to the ceiling T
                assert x['tran'] == ('true' if float(x['value']) == info['T'] else 'false')
    assert {x['luoi'] for x in sweep if x['factor'].startswith('(')} == {'mac_dinh'}
    assert {x['tran'] for x in sweep if x['factor'].startswith('(')} == {'false'}
    # S83-5: every design value (centre, outer, inner, axial, test) is the decoding of its code with the
    # levels; v8.5 section 7: the design is sealed/design.csv and decodes from levels_sangloc.csv,
    # which in a --trial run holds the same (fallback) levels as the public levels.csv
    design = list(_csv.DictReader(open(out / 'sealed' / 'design.csv', encoding='utf-8')))
    assert {r['box'] for r in design} == {'centre', 'outer', 'inner', 'axial', 'test'}
    for r in design:
        c = r['controller']
        for k, n in enumerate(s3.FACTORS[c]):
            L_, H_, sc = float(lv[(c, n)]['L']), float(lv[(c, n)]['H']), lv[(c, n)]['scale']
            assert r[f'{n}_value'] == repr(s3.decode(float(r[f'X{k + 1}']), L_, H_, sc)), (c, r['point_id'], n)
    # S03 v8.6: the cut inside the original grid of PID kp_wz ends at the ceiling T = 4.6
    assert (lv[('PID', 'kp_wz')]['H'], lv[('PID', 'kp_wz')]['H_nguon'], lv[('PID', 'kp_wz')]['b'],
            lv[('PID', 'kp_wz')]['b_n']) == ('4.6', 'trần bảng chặn', '10.0', '4.6')
    assert lv[('PID', 'kp_xy')]['b_n'] == '41.0'
    assert {r['H_nguon'] for r in lv.values()} == {'hết lưới', 'trần ω_o', 'trần bảng chặn'}
    assert {r['L_nguon'] for r in lv.values()} == {'0', 'hết lưới dưới'}
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert meta.startswith('spec: S03 v8.6 ')
    assert 'reference only' not in meta                                     # S03 v8.6: b_n is used
    grid_meta = _json.loads(next(x for x in meta.splitlines() if x.startswith('grid: ')).split('   #')[0][6:])
    assert set(grid_meta) == {f'{c} {n}' for c in s3.CONTROLLERS for n in s3.FACTORS[c]}
    assert grid_meta['PID kp_wz']['b_n'] == grid_meta['PID kp_wz']['T'] == 4.6
    assert grid_meta['PID kp_wz']['n_noi'] == 0
    assert grid_meta['PID ki_xy']['K'] == 4 and grid_meta['PID ki_xy']['n_bo'] == 0
    assert grid_meta['ADRC omega_o_xy']['b_n'] == 48.0 and lv[('ADRC', 'omega_o_xy')]['b_n'] == '48.0'   # Codex C3-5
    # v8.4: the check file, fixed header; with the fake every point outside [L, H] is 'không cờ'
    kiem = list(_csv.DictReader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8')))
    assert next(_csv.reader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8'))) == s3.KIEM_COLS
    pts = [r for r in kiem if r['dong'] == 'diem']
    n_noi = sum(i['n_noi'] for i in (s3.sweep_grid_cut(c, n, FAKE_DEFAULTS[c][n], lo, hi)[1]
                                     for c in s3.CONTROLLERS
                                     for n, lo, hi in zip(s3.FACTORS[c], *_tg.BOUNDS[c])))
    assert len(pts) == n_noi and {r['luoi'] for r in pts} == {'noi'} and {r['ket_qua'] for r in pts} == {'không cờ'}
    assert [(r['controller'], r['ket_qua']) for r in kiem if r['dong'] == 'tong'] == \
        [('PID', 'đã kiểm đủ'), ('ADRC', 'đã kiểm đủ')]
    assert 'kiem_ngoai_mien.csv' in _json.loads(next(x for x in meta.splitlines()
                                                     if x.startswith('file_sha256: '))[13:])
    km = _json.loads(next(x for x in meta.splitlines() if x.startswith('kiem_ngoai_mien: ')).split('   #')[0][17:])
    assert km['PID']['ket_qua'] == 'đã kiểm đủ' and km['ADRC']['co'] == []
    # S03 v8.6 section 5.2: the thresholds as a JSON dict, tv_duty the absolute 0.4 s^-1
    th = _json.loads(next(x for x in meta.splitlines() if x.startswith('thresholds: ')).split('   #')[0][12:])
    assert th['tv_duty'] == 0.4
    y3 = _json.loads(next(x for x in meta.splitlines() if x.startswith('y3_reference: '))
                     [len('y3_reference: '):])
    assert set(y3) == {f'{c} v={v}' for c in s3.CONTROLLERS for v in s3.SPEEDS}
    import yaml as _yaml
    _yaml.safe_load(meta)                                   # the whole meta.yaml is valid YAML


def _over_k_psi(ctrl_want, fields):
    """row_over of _fake: the given fields for the k_psi extension points of one controller (above b)."""
    lo_b, hi_b = _tg.BOUNDS[ctrl_want]
    j = s3.FACTORS[ctrl_want].index('k_psi')
    b = s3.sweep_grid_n('k_psi', 1.0, lo_b[j], hi_b[j])[1]['b']

    def over(ctrl, g, v):
        return fields if ctrl == ctrl_want and g['k_psi'] > b else None
    return over


@needs_dll
def test_v85_main_flag_is_reported_and_s03_goes_on(tmp_path, monkeypatch, capsys):
    """S03 v8.5 K6: a flagged point (ADRC k_psi extension better on e) no longer stops S03; the PID
    k_psi extension points stay inconclusive; exit 0, the main design ran, both are reported in
    kiem_ngoai_mien.csv, on the console and in meta.yaml."""
    import csv as _csv
    import json as _json
    over_a = _over_k_psi('ADRC', dict(e_perp_max=0.0))
    over_p = _over_k_psi('PID', dict(status='track_err'))
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys,
                                             row_over=lambda c, g, v: over_a(c, g, v) or over_p(c, g, v))
    assert code == 0
    assert max(calls) == 7                                          # the main design ran
    assert len(list(_csv.DictReader(open(out / 'sealed' / 'runs.csv')))) == 2 * 271 * 3
    assert len(list(_csv.DictReader(open(out / 'sealed' / 'design.csv')))) == 2 * 371   # v8.5 7: sealed
    kiem = list(_csv.DictReader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8')))
    co = [r for r in kiem if r['ket_qua'] == 'cờ']
    assert co and {(r['controller'], r['factor'], r['can']) for r in co} == {('ADRC', 'k_psi', 'H')}
    assert {r['ket_qua'] for r in kiem if r['dong'] == 'diem' and r['controller'] == 'PID'
            and r['factor'] == 'k_psi'} == {'chưa kết luận'}
    assert '[STOP]' not in console
    assert '[out-of-range check] flagged points' in console
    assert 'WARNING: out-of-range check has inconclusive points' in console
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'status: "ran to the end"' in meta
    km = _json.loads(next(x for x in meta.splitlines() if x.startswith('kiem_ngoai_mien: ')).split('   #')[0][17:])
    assert km['ADRC']['co']


@needs_dll
def test_v84_main_inconclusive_warns_and_goes_on(tmp_path, monkeypatch, capsys):
    import csv as _csv
    code, out, console, _, _ = _run_main(tmp_path, monkeypatch, capsys,
                                         row_over=_over_k_psi('PID', dict(status='track_err')))
    assert code == 0 and (out / 'sealed' / 'runs.csv').exists()
    assert 'WARNING: out-of-range check has inconclusive points' in console
    kiem = list(_csv.DictReader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8')))
    assert {r['ket_qua'] for r in kiem if r['dong'] == 'diem' and r['factor'] == 'k_psi' and
            r['controller'] == 'PID'} == {'chưa kết luận'}


@needs_dll
def test_v84_main_check_totals_when_s03_stops_in_the_sweep(tmp_path, monkeypatch, capsys):
    """'kiểm dở' when a run_error stops the sweep (every grid point counted as not checked), 'không kiểm' at 3.1 (i)."""
    import csv as _csv
    code, out, _, _, _ = _run_main(tmp_path, monkeypatch, capsys, error_in_sweep='factor')
    assert code == 2
    tot = {r['controller']: r for r in _csv.DictReader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8'))
           if r['dong'] == 'tong'}
    # S03 v8.6: the points not checked are those of the grid after the cut at the ceiling
    n_all = {c: sum(len(s3.sweep_grid_cut(c, n, FAKE_DEFAULTS[c][n], lo, hi)[0])
                    for n, lo, hi in zip(s3.FACTORS[c], *_tg.BOUNDS[c])) for c in s3.CONTROLLERS}
    for c in s3.CONTROLLERS:
        assert tot[c]['ket_qua'] == 'kiểm dở' and tot[c]['n_gain_chua_kiem'] == '7'
        assert tot[c]['n_chua_kiem'] == str(n_all[c]) and tot[c]['n_da_kiem'] == '0'
    tmp2 = tmp_path / 'b'
    tmp2.mkdir()
    code, out, _, _, _ = _run_main(tmp2, monkeypatch, capsys, default_fails=True)
    assert code == 2
    assert [r['ket_qua'] for r in _csv.DictReader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8'))] == \
        ['không kiểm', 'không kiểm']


@needs_dll
def test_v84_main_case_ii_after_one_controller_is_not_checked(tmp_path, monkeypatch, capsys):
    """S84-6: PID sweeps fully, ADRC has L = H (omega_c_xy completes only at its default): S03 stops at 3.1 (ii)
    raised in main, both totals 'không kiểm', no point row, no main-design run."""
    import csv as _csv

    def over(ctrl, g, v):
        return dict(completed='false') if ctrl == 'ADRC' and g['omega_c_xy'] != 8.0 else None
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, row_over=over)
    assert code == 2 and '3.1 (ii)' in console and max(calls) <= 1
    kiem = list(_csv.DictReader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8')))
    assert [(r['dong'], r['controller'], r['ket_qua']) for r in kiem] == [('tong', 'PID', 'không kiểm'),
                                                                          ('tong', 'ADRC', 'không kiểm')]
    assert 'S03 stopped at 3.1 (ii)' in kiem[0]['ghi_chu']


@needs_dll
def test_v84_main_case_i_from_the_level_rule_keeps_its_mark(tmp_path, monkeypatch, capsys):
    """S84-6: the default point of the PID sweep fails at 0.8 although the default check passed: levels_from_sweep
    raises (i), main raises it again with the factor name, the mark '(i)' survives: both totals 'không kiểm'."""
    import csv as _csv
    seen = dict(n=0)

    def over(ctrl, g, v):
        if ctrl == 'PID' and all(g[n] == FAKE_DEFAULTS['PID'][n] for n in s3.FACTORS['PID']):
            seen['n'] += 1
            if seen['n'] > 3 and v == 0.8:                 # after the 3 runs of the default check
                return dict(completed='false')
        return None
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, row_over=over)
    assert code == 2 and 'PID kp_xy: default gains do not complete' in console
    assert [r['ket_qua'] for r in _csv.DictReader(open(out / 'kiem_ngoai_mien.csv', encoding='utf-8'))] == \
        ['không kiểm', 'không kiểm']


@needs_dll
def test_b813_g1_fields_on_real_runs(v82_runs):
    """B81-3 (15/11): on real DLL runs, the G1 fields agree with completed and the run, the common domain is the
    union one, and G1 recomputed from the fields is g1_state."""
    for ctrl, v, row, _ in v82_runs:
        assert row['g1_ways'] == 1 and row['g1_t_done_2'] is None and row['g1_cat_t1'] is None
        assert row['g1_t_deadline'] == o1.Schedule(v).Tcmd + o1.T_LATE        # G1F-2: the deadline itself
        if row['completed'] == 'true':
            assert row['g1_t_done_1'] == row['t_end'] < row['g1_t_deadline']
        else:
            assert row['g1_t_done_1'] is None
        assert (row['g1_t1'] is None) == (row['status'] == 'ok')
        f = dict(t_done=[row['g1_t_done_1']], t_deadline=row['g1_t_deadline'], t1=row['g1_t1'],
                 prog_before_t1=row['g1_prog_before_t1'], cond25_before_deadline=row['g1_cond25_before_deadline'])
        assert G.g1_state_from_fields(f) == row['g1_state'], (ctrl, v)
        for c in Q_OF:
            for x in ('q', 'lo', 'hi', 'n_undef'):
                assert s3._bits(row[f'{c}_{x}_chung']) == s3._bits(row[f'{c}_{x}'])


# ── S03 v8.6: the sweep grid cut at the gain-table ceiling, W nodes, absolute Y3, triangle ──────

HI_T = {'PID': dict(kp_xy=41.0, ki_xy=1500.0, kp_wz=4.6, ki_wz=230.0, pos_kp=200.0, pos_ki=5500.0,
                    k_psi=50.0),
        'ADRC': dict(omega_c_xy=200.0, omega_o_xy=100.0, omega_c_wz=200.0, omega_o_wz=100.0,
                     pos_kp=200.0, pos_ki=40000.0, k_psi=50.0)}


def _f32v(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def test_v86_factor_names_are_the_gain_table_names():
    """S03 v8.6 section 3.1: the factors are the names of tools/gain_table.py and ceiling() is its
    upper bound; a name that is not a gain of the controller raises S03Stop."""
    assert s3.FACTORS == {c: s3.gain_table.NAMES[c] for c in s3.CONTROLLERS}
    for c in s3.CONTROLLERS:
        for n in s3.FACTORS[c]:
            assert s3.ceiling(c, n) == HI_T[c][n]
    with pytest.raises(s3.S03Stop, match='ceiling'):
        s3.ceiling('PID', 'omega_c_xy')


def test_v86_cut_at_ceiling_rules():
    """S03 v8.6 section 3.1 rules (1), (2): points above T dropped; a remaining point within 1e-9
    becomes T, else T is appended."""
    assert s3.cut_at_ceiling([1.0, 2.0, 4.0, 8.0], 5.0) == ([1.0, 2.0, 4.0, 5.0], 1, True)
    assert s3.cut_at_ceiling([1.0, 2.0, 4.0], 5.0) == ([1.0, 2.0, 4.0], 0, False)
    assert s3.cut_at_ceiling([1.0, 2.0, 4.0, 8.0], 4.0) == ([1.0, 2.0, 4.0], 1, False)
    x = 4.0 * (1 - 5e-10)                             # within 1e-9: becomes T, same bits
    pts, n_bo, them = s3.cut_at_ceiling([1.0, x, 8.0], 4.0)
    assert pts == [1.0, 4.0] and s3._bits(pts[-1]) == s3._bits(4.0) and (n_bo, them) == (1, False)
    x = 4.0 * (1 - 2e-9)                              # just outside: T appended
    pts, n_bo, them = s3.cut_at_ceiling([1.0, x, 8.0], 4.0)
    assert pts == [1.0, x, 4.0] and (n_bo, them) == (1, True)
    # a point within 1e-9 ABOVE T is dropped like any point above
    assert s3.cut_at_ceiling([1.0, 4.0 * (1 + 5e-10)], 4.0) == ([1.0, 4.0], 1, True)
    # nothing dropped: the points stay as they are even within 1e-9 below T
    pts, n_bo, them = s3.cut_at_ceiling([1.0, 4.0 * (1 - 5e-10)], 4.0)
    assert pts == [1.0, 4.0 * (1 - 5e-10)] and (n_bo, them) == (0, False)
    # two remaining points within 1e-9 with one dropped: error
    with pytest.raises(s3.S03Stop, match='ceiling'):
        s3.cut_at_ceiling([1.0, 4.0 * (1 - 7e-10), 4.0 * (1 - 5e-10), 8.0], 4.0)


def test_v86_spec_example_pid_kp_wz_is_cut_in_the_original_grid():
    """S03 v8.6 section 3.1 example: PID kp_wz (default 1.0, bounds 0..6, T = 4.6 < b = 10): the
    cut is inside the original grid, T is appended and the extension is gone entirely."""
    grid, info = s3.sweep_grid_cut('PID', 'kp_wz', 1.0, 0.0, 6.0)
    old = s3.sweep_grid('kp_wz', 1.0, 0.0, 6.0)
    kept = [x for x in old if x <= 4.6]
    assert [s3._bits(x) for x in grid] == [s3._bits(x) for x in kept + [4.6]]
    assert kept[-1] == pytest.approx(3.98, abs=0.01)
    assert min(x for x in old if x > 4.6) == pytest.approx(6.31, abs=0.01)
    assert s3.base_grid(grid, info) == grid
    assert info['noi'] == [] and info['n_noi'] == 0
    assert info['b'] == 10.0 and info['a'] == 0.1 and info['T'] == 4.6 and info['b_n'] == 4.6 == grid[-1]
    i0 = s3.sweep_grid_n('kp_wz', 1.0, 0.0, 6.0)[1]
    assert info['n_bo'] == len(s3.sweep_grid_n('kp_wz', 1.0, 0.0, 6.0)[0]) - len(kept)
    assert info['n_points'] == len(grid)
    assert info['K'] == i0['K'] and info['r'] == i0['r']
    assert s3.grid_label(4.6, 1.0, info['noi']) == 'log'
    assert s3.levels_from_sweep(grid, 1.0, [True] * len(grid)) == (0.0, 4.6)
    assert s3.level_sources(grid, 0.0, 4.6, info, 'kp_wz') == ('0', 'trần bảng chặn')
    ok = [True] * (len(grid) - 1) + [False]
    L, H = s3.levels_from_sweep(grid, 1.0, ok)
    assert H == kept[-1] and s3.level_sources(grid, L, H, info, 'kp_wz')[1] == \
        'điểm kế tiếp không hoàn thành'


@pytest.mark.parametrize('ctrl, name', [('PID', 'kp_xy'), ('PID', 'pos_kp'), ('PID', 'k_psi'),
                                        ('ADRC', 'omega_c_xy'), ('ADRC', 'omega_c_wz'),
                                        ('ADRC', 'pos_kp'), ('ADRC', 'k_psi')])
def test_v86_cut_in_the_extension(ctrl, name):
    """b < T < b_n: the original grid is kept bit for bit, the extension is cut at T."""
    lo, hi = dict(zip(s3.FACTORS[ctrl], zip(*_tg.BOUNDS[ctrl])))[name]
    d = FAKE_DEFAULTS[ctrl][name]
    T = HI_T[ctrl][name]
    g0, i0 = s3.sweep_grid_n(name, d, lo, hi)
    g, i = s3.sweep_grid_cut(ctrl, name, d, lo, hi)
    assert i0['b'] < T < i0['b_n']
    assert [s3._bits(x) for x in s3.base_grid(g, i)] == [s3._bits(x) for x in s3.sweep_grid(name, d, lo, hi)]
    assert i['noi'] == [x for x in i0['noi'] if x <= T] + [T]
    assert g[-1] == T == i['b_n']
    assert i['n_bo'] == len([x for x in i0['noi'] if x > T]) >= 1
    for k in ('a', 'b', 'r', 'K', 'n_log'):
        assert i[k] == i0[k]
    assert s3.grid_label(T, d, i['noi']) == 'noi'
    base = s3.base_grid(g, i)
    assert s3.levels_from_sweep(base, d, [True] * len(base))[1] == i['b']
    assert s3.level_sources(base, base[0], i['b'], i, name)[1] == 'hết lưới'


@pytest.mark.parametrize('ctrl, name', [('PID', 'ki_xy'), ('PID', 'ki_wz'), ('PID', 'pos_ki'),
                                        ('ADRC', 'omega_o_xy'), ('ADRC', 'omega_o_wz'),
                                        ('ADRC', 'pos_ki')])
def test_v86_uncut_gains_keep_the_old_grid_bit_for_bit(ctrl, name):
    """T >= b_n: the grid and every old info key are exactly those of sweep_grid_n."""
    lo, hi = dict(zip(s3.FACTORS[ctrl], zip(*_tg.BOUNDS[ctrl])))[name]
    d = FAKE_DEFAULTS[ctrl][name]
    g0, i0 = s3.sweep_grid_n(name, d, lo, hi)
    g, i = s3.sweep_grid_cut(ctrl, name, d, lo, hi)
    assert [s3._bits(x) for x in g] == [s3._bits(x) for x in g0]
    assert i['n_bo'] == 0 and i['T'] == HI_T[ctrl][name]
    if name.startswith('omega_o'):                  # Codex C3-5: b_n of omega_o is the grid end b, not None
        assert i0['b_n'] is None and i['b_n'] == i['b'] == 48.0 == g[-1]
        assert {k: v for k, v in i.items() if k not in ('T', 'n_bo', 'b_n')} == \
            {k: v for k, v in i0.items() if k != 'b_n'}
    else:
        assert {k: v for k, v in i.items() if k not in ('T', 'n_bo')} == i0
    if (ctrl, name) == ('PID', 'ki_xy'):
        want = V8_GRID_BITS[('ki_xy', 0.05, 0.0, 10.0)].split()
        assert [struct.pack('<d', x).hex() for x in g[:len(want)]] == want
    if (ctrl, name) == ('ADRC', 'omega_o_xy'):
        want = V8_GRID_BITS[('omega_o_xy', 30.0, 5.0, 48.0)].split()
        assert [struct.pack('<d', x).hex() for x in g[:len(want)]] == want


def test_v86_point_within_1e9_of_the_ceiling_becomes_the_ceiling():
    """(a) b itself within 1e-9 of T (here b == T == 50): the base end becomes T, no extension.
    (b) cut_grid: T a hair above b keeps b and adds T to the extension; a hair below drops b and
    appends T to the original grid."""
    g, i = s3.sweep_grid_cut('PID', 'k_psi', 5.0, 0.0, 4.0)      # b = 50.0 = T
    _, i_uncut = s3.sweep_grid_n('k_psi', 5.0, 0.0, 4.0)
    assert g == s3.sweep_grid('k_psi', 5.0, 0.0, 4.0)
    assert i['noi'] == [] and i['b_n'] == 50.0 == g[-1] and i['n_bo'] == i_uncut['K']
    assert s3.level_sources(s3.base_grid(g, i), g[0], 50.0, i, 'k_psi')[1] == 'trần bảng chặn'

    g0, i0 = s3.sweep_grid_n('ki_xy', 0.05, 0.0, 10.0)           # b = 10.0
    base0 = s3.base_grid(g0, i0)
    pts, info = s3.cut_grid(g0, i0, 10.0 * (1 + 5e-10))          # b within 1e-9 below T: becomes T
    assert len(s3.base_grid(pts, info)) == len(base0)
    assert s3.base_grid(pts, info)[-1] == 10.0 * (1 + 5e-10) and info['noi'] == []
    pts, info = s3.cut_grid(g0, i0, 10.0 * (1 + 2e-9))           # just above: b kept, T in noi
    assert s3.base_grid(pts, info)[-1] == 10.0 and info['noi'] == [10.0 * (1 + 2e-9)]
    pts, info = s3.cut_grid(g0, i0, 10.0 * (1 - 5e-10))          # just below: b dropped, T appended
    assert s3.base_grid(pts, info)[-1] == 10.0 * (1 - 5e-10) and info['noi'] == []
    for T in (10.0 * (1 + 5e-10), 10.0 * (1 + 2e-9), 10.0 * (1 - 5e-10)):
        _, info = s3.cut_grid(g0, i0, T)
        assert info['b_n'] == T and info['b'] == 10.0


def test_v86_default_above_the_ceiling_stops():
    """S03 v8.6 section 3.1: a firmware default above its ceiling is not a grid point: S03 stops."""
    with pytest.raises(s3.S03Stop, match='ceiling'):
        s3.sweep_grid_cut('PID', 'kp_wz', 5.0, 0.0, 6.0)


W_NODE_CASES = ([(c, n, FAKE_DEFAULTS[c][n], lo, hi)
                 for c in s3.CONTROLLERS
                 for n, lo, hi in zip(s3.FACTORS[c], *_tg.BOUNDS[c])]
                + [('PID', 'kp_wz', 0.37, 0.0, 6.0), ('PID', 'kp_xy', 0.6, 0.0, 6.0),
                   ('ADRC', 'omega_c_wz', 0.5, 1.0, 25.0)])


def _w_nodes_want(name, d, lo, hi, T):
    """The spec expectation of w_nodes (S03 v8.6 section 3.1b item 2), written here from the spec:
    the 11 values of geomspace(a, b, 11) (the one within 1e-9 of the default is the default) plus
    b r^k, then rules (1), (2): drop > T; if dropped, a point within 1e-9 becomes T else T appended."""
    a = d / 10.0 if lo == 0.0 else min(d / 10.0, lo)
    b = max(10.0 * d, hi)
    if name.startswith('omega_o'):
        b = min(b, 48.0)
    logs = [d if math.isclose(x, d, rel_tol=1e-9) else float(x) for x in np.geomspace(a, b, 11)]
    r = (b / a) ** 0.1
    K = 0
    if not name.startswith('omega_o'):
        K = 1
        while r ** K < 10 * (1 - 1e-9):
            K += 1
    ext = [float(b * r ** k) for k in range(1, K + 1)]
    pts = [x for x in logs + ext if not x > T]
    if len(pts) < len(logs) + len(ext):
        near = [i for i, x in enumerate(pts) if math.isclose(x, T, rel_tol=1e-9)]
        if near:
            pts[near[0]] = T
        else:
            pts.append(T)
    return pts, a, b


@pytest.mark.parametrize('ctrl, name, d, lo, hi', W_NODE_CASES)
def test_v86_w_nodes_definition(ctrl, name, d, lo, hi):
    """The W nodes are the cut log-plus-extension lists of the grid, never the luoi labels."""
    want, a, b = _w_nodes_want(name, d, lo, hi, HI_T[ctrl][name])
    nodes = s3.w_nodes(ctrl, name, d, lo, hi)
    assert [s3._bits(x) for x in nodes] == [s3._bits(x) for x in want]
    grid, info = s3.sweep_grid_cut(ctrl, name, d, lo, hi)
    gbits = [s3._bits(x) for x in grid]
    nbits = [s3._bits(x) for x in nodes]
    assert all(x in gbits for x in nbits)
    assert 0.0 not in nodes and nodes[0] == a
    assert nodes[-1] == (info['b_n'] if info['b_n'] is not None else info['b'])
    assert set(x for x in grid if s3._bits(x) not in nbits) <= {0.0, d}
    assert (d in nodes) == any(math.isclose(x, d, rel_tol=1e-9) for x in np.geomspace(a, b, 11))


def test_v86_w_nodes_do_not_follow_the_luoi_label():
    """PID kp_wz: 4.6 is a node although grid_label calls it 'log' and the default 1.0 is a node
    although its label is 'mac_dinh'; with a default off the log points the default is no node."""
    nodes = s3.w_nodes('PID', 'kp_wz', 1.0, 0.0, 6.0)
    grid, info = s3.sweep_grid_cut('PID', 'kp_wz', 1.0, 0.0, 6.0)
    assert 4.6 in nodes and s3.grid_label(4.6, 1.0, info['noi']) == 'log'
    assert 1.0 in nodes and s3.grid_label(1.0, 1.0, info['noi']) == 'mac_dinh'
    assert nodes != [x for x in grid if s3.grid_label(x, 1.0, info['noi']) in ('log', 'noi')]
    old = s3.sweep_grid('kp_wz', 1.0, 0.0, 6.0)
    assert len(nodes) == len([x for x in old if x != 0.0 and x <= 4.6]) + 1
    nodes2 = s3.w_nodes('PID', 'kp_wz', 0.37, 0.0, 6.0)
    assert 0.37 not in nodes2 and s3.grid_label(0.37, 0.37, []) == 'mac_dinh'


def test_v86_analyse_threshold_does_not_depend_on_the_reference():
    """S03 v8.6 section 5.2: the analysis threshold is the absolute 0.4 s^-1 at every speed, mu,
    worst and range; tv = 0.2 X3 gives the effect exactly 0.4 -> NOT significant (strictly above)."""
    all_eff = []
    for ref in (0.0, 1.0, 100.0):
        pts, runs = _runs(lambda c, mu: _row(0.05, tv=0.2 * c[2]))
        eff, _, _, _ = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): ref})
        all_eff.append(eff)
    assert all_eff[0] == all_eff[1] == all_eff[2]
    eff = all_eff[0]
    tv_rows = [r for r in eff if r['response'] == 'tv_duty']
    assert {r['mu'] for r in tv_rows} == {'0.4', '0.5', '0.6', 'worst', 'range'}
    assert all(r['threshold'] == 0.4 for r in tv_rows)
    x3 = next(r for r in tv_rows if r['mu'] == '0.5' and r['term'] == 'X3')
    assert x3['effect'] == pytest.approx(0.4) and not x3['significant']
    assert not any('reference' in r['note'] for r in eff)
    pts, runs = _runs(lambda c, mu: _row(0.05, tv=0.25 * c[2]))
    eff, _, _, _ = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): 1.0})
    x3 = next(r for r in eff if r['response'] == 'tv_duty' and r['mu'] == '0.5' and r['term'] == 'X3')
    assert x3['effect'] == pytest.approx(0.5) and x3['significant']


def test_v86_bounds_check_gain_table():
    """S03 v8.6 section 3.1b item 4: check_table compares the float32 of every gain with the
    gain table (bounds themselves pass, 0 accepted where the lower bound is 0); the triangle is
    NOT checked there. check_bounds keeps its v8 behaviour: a gain outside the table but inside
    [L, H] does not raise."""
    lv = {n: (0.0, 1.0, 'lin') for n in s3.FACTORS['PID']}
    lv['kp_wz'] = (0.0, 4.6, 'lin')

    def gains_at(codes):
        return {n: s3.decode(c, *lv[n]) for n, c in zip(s3.FACTORS['PID'], codes)}
    pts = [(1, 'outer', (1.0,) * 7), (2, 'outer', (-1.0,) * 7)]
    s3.check_table('PID', pts, gains_at)                    # 4.6 and 0 are in the table
    lv['kp_wz'] = (0.0, 4.60000001, 'lin')                  # float32 of the corner is f32(4.6): passes
    s3.check_table('PID', pts, gains_at)
    lv['kp_wz'] = (0.0, 4.6000002, 'lin')                   # float32 above the table bound
    s3.check_bounds('PID', lv, pts, gains_at)               # inside [L, H]: the v8 check does not raise
    with pytest.raises(s3.S03Stop, match='gain table'):
        s3.check_table('PID', pts, gains_at)

    lv = {n: (1.0, 8.0, 'log') for n in s3.FACTORS['ADRC']}
    lv['omega_c_xy'] = (0.4, 8.0, 'log')

    def gains_at_a(codes):
        return {n: s3.decode(c, *lv[n]) for n, c in zip(s3.FACTORS['ADRC'], codes)}
    s3.check_bounds('ADRC', lv, [(2, 'outer', (-1.0,) * 7)], gains_at_a)      # inside [L, H]: no raise
    with pytest.raises(s3.S03Stop, match='gain table'):
        s3.check_table('ADRC', [(2, 'outer', (-1.0,) * 7)], gains_at_a)       # 0.4 < lower bound 0.5
    lv['omega_c_xy'] = (1.0, 8.0, 'log')
    lv['pos_kp'] = (0.0, 6.0, 'lin')                        # 0 where the lower bound is 0: passes
    s3.check_table('ADRC', [(2, 'outer', (-1.0,) * 7)], gains_at_a)
    lv['omega_c_xy'] = (1.0, 150.0, 'log')
    lv['omega_o_xy'] = (12.0, 48.0, 'log')
    # (150, 48): outside the triangle but inside the table: no raise
    s3.check_table('ADRC', [(1, 'outer', (1.0,) * 7)], gains_at_a)


def test_v86_candidate_check_and_outside_triangle():
    """S03 v8.6 section 3.1b item 4 (Codex C3-2): check_candidate(ctrl, g, w) checks, in order,
    (1) finite binary64 and a_i <= x <= w_i of the reference box W (exact, no tolerance),
    (2) the float32 of every omega_o* <= 48, (3) the float32 of every gain inside the gain table,
    (4) the LADRC triangle. A failure of 1-3 raises S03Stop naming the step; a broken triangle is
    the only classifiable failure ('ngoai_bang'), else 'ok'. outside_triangle is
    'not trong_tam_giac' (PID always False); the sum is binary64 on the float32 values, strictly
    below 200."""
    base = dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0, pos_kp=6.0,
                pos_ki=3.0, k_psi=1.0)

    def w_adrc(**kw):
        w = {n: (0.5, 200.0) for n in s3.FACTORS['ADRC']}
        w.update(omega_o_xy=(5.0, 50.0), omega_o_wz=(5.0, 50.0), pos_kp=(1.0, 6.0))
        w.update(kw)
        return w

    W = w_adrc()
    assert s3.check_candidate('ADRC', base, W) == 'ok' and s3.outside_triangle('ADRC', base) is False
    # Codex C3-2: omega_o 49 inside a W ending at 50 passes the table (49 <= 100) and the
    # triangle (8 + 2*49 < 200) but is above the omega_o ceiling 48 -> step 2
    with pytest.raises(s3.S03Stop, match='step 2'):
        s3.check_candidate('ADRC', dict(base, omega_o_xy=49.0), W)
    # step 1: outside the W box
    with pytest.raises(s3.S03Stop, match='step 1'):
        s3.check_candidate('ADRC', dict(base, pos_kp=7.0), W)
    # a set failing steps 1 and 2 reports step 1
    with pytest.raises(s3.S03Stop, match='step 1'):
        s3.check_candidate('ADRC', dict(base, omega_o_xy=51.0), W)
    # the two ends of W are inside (exact comparison)
    assert s3.check_candidate('ADRC', dict(base, pos_kp=1.0), W) == 'ok'
    assert s3.check_candidate('ADRC', dict(base, pos_kp=6.0), W) == 'ok'
    # not finite -> step 1
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(s3.S03Stop, match='step 1'):
            s3.check_candidate('ADRC', dict(base, pos_kp=bad), W)
    # step 2 is on the float32: float32(48.0000001) is 48.0 and passes; float32(48.00001) > 48
    assert s3.check_candidate('ADRC', dict(base, omega_o_xy=48.0000001), W) == 'ok'
    with pytest.raises(s3.S03Stop, match='step 2'):
        s3.check_candidate('ADRC', dict(base, omega_o_xy=48.00001), W)
    # step 3: the float32 must be inside the gain table (0 accepted where the lower bound is 0)
    with pytest.raises(s3.S03Stop, match='step 3'):
        s3.check_candidate('ADRC', dict(base, omega_c_xy=0.4), w_adrc(omega_c_xy=(0.3, 200.0)))
    # step 4: a broken triangle is the only classifiable failure, on either axis
    g = dict(base, omega_c_xy=104.0, omega_o_xy=48.0)            # sum exactly 200
    assert s3.check_candidate('ADRC', g, W) == 'ngoai_bang' and s3.outside_triangle('ADRC', g) is True
    g = dict(base, omega_c_wz=104.0, omega_o_wz=48.0)
    assert s3.check_candidate('ADRC', g, W) == 'ngoai_bang' and s3.outside_triangle('ADRC', g) is True
    g = dict(base, omega_c_xy=103.9999999, omega_o_xy=48.0)      # float32 is 104.0 -> outside
    assert s3.check_candidate('ADRC', g, W) == 'ngoai_bang' and s3.outside_triangle('ADRC', g) is True
    g = dict(base, omega_c_xy=103.99999, omega_o_xy=48.0)        # float32 stays below 104.0
    assert s3.check_candidate('ADRC', g, W) == 'ok' and s3.outside_triangle('ADRC', g) is False
    # PID: inside -> 'ok' (never 'ngoai_bang'); a float32 above the table bound -> step 3
    gp = dict(FAKE_DEFAULTS['PID'])
    Wp = {n: (0.0, 10.0) for n in s3.FACTORS['PID']}
    Wp['kp_wz'] = (0.0, 5.0)
    assert s3.check_candidate('PID', gp, Wp) == 'ok' and s3.outside_triangle('PID', gp) is False
    assert s3.check_candidate('PID', dict(gp, kp_wz=4.6), Wp) == 'ok'         # 4.6 is the table bound
    with pytest.raises(s3.S03Stop, match='step 3'):
        s3.check_candidate('PID', dict(gp, kp_wz=4.6000002), Wp)  # float32 is 4.6000004 > 4.6
    with pytest.raises(s3.S03Stop, match='step 3'):
        s3.check_candidate('PID', dict(gp, kp_xy=41.1), dict(Wp, kp_xy=(0.0, 50.0)))  # 41.1 > 41.0
    # malformed input: a wrong key set of g or w -> ValueError
    with pytest.raises(ValueError):
        s3.check_candidate('PID', {k: x for k, x in gp.items() if k != 'k_psi'}, Wp)
    with pytest.raises(ValueError):
        s3.check_candidate('ADRC', base, {k: v for k, v in W.items() if k != 'k_psi'})


@needs_dll
def test_v86_main_ngoai_tam_giac_columns_and_count(tmp_path, monkeypatch, capsys):
    """S03 v8.6 sections 3.1b/3.2: with the ADRC omega_c_xy default 12.0 the outer corners with
    X1 = X2 = +1 break the triangle; they run like any point and carry ngoai_tam_giac in design.csv,
    runs.csv, runs_test.csv; the per-controller counts are in the notes files."""
    import csv as _csv
    import json as _json
    monkeypatch.setitem(FAKE_DEFAULTS['ADRC'], 'omega_c_xy', 12.0)
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    # v8.5 section 7: design.csv is sealed/design.csv (not public)
    header = next(_csv.reader(open(out / 'sealed' / 'design.csv', encoding='utf-8')))
    assert header[-1] == 'ngoai_tam_giac'
    # S03 v8.6: the value columns are the factor names, shared names deduplicated (pos_*, k_psi)
    assert header[:-1] == (['controller', 'point_id', 'box'] + [f'X{k}' for k in range(1, 8)]
                           + [f'{n}_value' for n in s3.FACTORS['PID']]
                           + [f'{n}_value' for n in s3.FACTORS['ADRC'] if n not in s3.FACTORS['PID']])
    design = list(_csv.DictReader(open(out / 'sealed' / 'design.csv', encoding='utf-8')))

    def want_flag(r):
        if r['controller'] == 'PID':
            return 'false'
        g = {n: float(r[f'{n}_value']) for n in s3.FACTORS['ADRC']}
        bad = (_f32v(g['omega_c_xy']) + 2.0 * _f32v(g['omega_o_xy']) >= 200.0
               or _f32v(g['omega_c_wz']) + 2.0 * _f32v(g['omega_o_wz']) >= 200.0)
        return 'true' if bad else 'false'
    for r in design:
        assert r['ngoai_tam_giac'] == want_flag(r), (r['controller'], r['point_id'])
    outer_flag = [r for r in design if r['controller'] == 'ADRC' and r['ngoai_tam_giac'] == 'true'
                  and r['box'] == 'outer']
    assert len(outer_flag) == 32
    assert all(r['X1'] == '1.0' and r['X2'] == '1.0' for r in outer_flag)
    assert not [r for r in design if r['ngoai_tam_giac'] == 'true' and r['box'] != 'test'
                and not (r['box'] == 'outer' and r['X1'] == '1.0' and r['X2'] == '1.0')]
    assert not any(r['ngoai_tam_giac'] == 'true' for r in design if r['controller'] == 'PID')
    flags = {(r['controller'], r['point_id']): r['ngoai_tam_giac'] for r in design}
    for f in ('sealed/runs.csv', 'sealed_test/runs_test.csv'):
        with open(out / f, encoding='utf-8') as fh:
            rd = list(_csv.DictReader(fh))
        for r in rd:
            assert r['ngoai_tam_giac'] == flags[(r['controller'], r['point_id'])], (f, r['point_id'])
    runs = list(_csv.DictReader(open(out / 'sealed' / 'runs.csv', encoding='utf-8')))
    assert len(runs) == 2 * 271 * 3 and {r['status'] for r in runs} == {'ok'}
    notes = _json.load(open(out / 'sealed' / 'notes.json'))
    assert notes['ngoai_tam_giac_points'] == {'PID': 0, 'ADRC': 32}
    n_test = sum(1 for r in design if r['box'] == 'test' and r['ngoai_tam_giac'] == 'true')
    notes_t = _json.load(open(out / 'sealed_test' / 'notes.json'))
    assert notes_t['ngoai_tam_giac_points'] == {'PID': 0, 'ADRC': n_test}
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    # S03 v8.6: the counts are in the notes files only, not a meta.yaml field (the files: block
    # only documents the notes key)
    assert 'ngoai_tam_giac_points:' not in meta
    assert 'ngoai_tam_giac' not in console.replace(str(out), '')  # nothing of it on the console
    # (the tmp path itself contains the test name)
    assert 'ngoai_tam_giac' not in next(_csv.reader(open(out / 'sweep.csv', encoding='utf-8')))


@needs_dll
def test_v86_main_gain_table_failure_stops_before_the_main_design(tmp_path, monkeypatch, capsys):
    """S03 v8.6 section 3.1b item 4 (C3-1 stop channel): an ADRC omega_c_xy design corner below the
    (raised) table lower bound stops S03 inside the sealed window, phase 'thiet_ke': the console
    and meta.yaml say only 'stopped after the sweep', the reason and the phase are in
    sealed/stop.json, no main-design run, no design.csv anywhere, stop.json in the manifest."""
    import csv as _csv
    import json as _json
    monkeypatch.setitem(s3.gain_table.LO, 'ADRC', (1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0))
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 2 and 'sealed/stop.json' in console
    # the stop is sealed (3.1b item 13): the reason never reaches the console
    assert 'gain table' not in console and '[STOP]' not in console
    assert max(calls) <= 1                                          # no run with two factors off
    stop = _json.load(open(out / 'sealed' / 'stop.json', encoding='utf-8'))
    assert stop['pha'] == 'thiet_ke' and 'outside the gain table' in stop['stop']
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'stopped after the sweep (the reason is in sealed/stop.json)' in meta
    # no gain value of the reason escapes into a public file
    assert stop['stop'] not in meta and 'outside the gain table' not in meta
    # the design is sealed (v8.5 section 7): nothing of it is public, and the stop hit during its
    # construction, before the file was written
    assert not (out / 'design.csv').exists() and not (out / 'sealed' / 'design.csv').exists()
    man = _json.load(open(out / 'sealed' / 'manifest.json', encoding='utf-8'))
    assert 'stop.json' in man['files'] and 'design.csv' not in man['files']
    assert not (out / 'sealed_test').exists()
    runs = out / 'sealed' / 'runs.csv'
    assert not runs.exists() or len(list(_csv.DictReader(open(runs)))) == 0


# ── Codex C3 round 1 (2026-11-22): C3-3 (the S03 side), C3-4, C3-5 ───────────────────────────────

C34_D = 41.0 * (1 - 5e-10)        # Codex C3-4: PID kp_xy, T = 41, temporary bounds 0 .. 41


def _bl(xs):
    return [s3._bits(x) for x in xs]


def test_c3_3_b_within_1e9_below_the_ceiling_is_the_end_of_the_original_grid():
    """Codex C3-3, the S03 side (tools/tests/test_chep_luot_A.py c33_run holds the same numbers): PID
    pos_kp with b = 199.9999999 within 1e-9 below T = 200 and the first extension point above T. The
    extension is dropped, the point b becomes T and keeps its part of the grid: the original grid
    ends at T (luoi log, tran true), info b keeps its definition, b_n = T, and H can be T > b."""
    b, T = 199.9999999, 200.0
    assert b < T and math.isclose(b, T, rel_tol=1e-9)
    g0, i0 = s3.sweep_grid_n('pos_kp', 6.0, 0.0, b)
    base0 = s3.base_grid(g0, i0)
    assert i0['b'] == b == base0[-1] and i0['noi'][0] > T
    grid, info = s3.sweep_grid_cut('PID', 'pos_kp', 6.0, 0.0, b)
    assert s3.base_grid(grid, info) == grid and info['noi'] == [] and info['n_bo'] == i0['K']
    assert _bl(grid) == _bl(base0[:-1] + [T])
    assert (info['b'], info['b_n'], info['T']) == (b, T, T)
    assert s3.grid_label(T, 6.0, info['noi']) == 'log' and [x == info['T'] for x in grid].count(True) == 1
    L, H = s3.levels_from_sweep(grid, 6.0, [True] * len(grid))
    assert H == T > info['b'] and s3.level_sources(grid, L, H, info, 'pos_kp') == ('0', 'trần bảng chặn')


def test_c3_4_default_on_a_log_point_merged_into_the_ceiling():
    """Codex C3-4 (kp_xy, T = 41, default 41 (1 - 5e-10), bounds 0 .. 41): the default is the 6th log
    point and the only remaining point within 1e-9 of T, so the cut makes it T. Both are the float32
    41.0: the point T stands for the default run (default_node) in the level rule and the labels,
    and the W nodes are the log points below it plus T."""
    d, T = C34_D, 41.0
    assert d < T and math.isclose(d, T, rel_tol=1e-9) and _f32v(d) == _f32v(T) == 41.0
    g0, i0 = s3.sweep_grid_n('kp_xy', d, 0.0, 41.0)
    assert d in g0 and len(s3.base_grid(g0, i0)) == 12             # 0 and 11 log points, the default one of them
    grid, info = s3.sweep_grid_cut('PID', 'kp_xy', d, 0.0, 41.0)   # S03Stop before the fix
    assert _bl(grid) == _bl([x for x in g0 if x < d] + [T]) and d not in grid
    assert info['b_n'] == T == grid[-1] and info['noi'] == [] and info['n_bo'] == len(g0) - len(grid)
    assert s3.base_grid(grid, info) == grid
    node = s3.default_node(grid, info, d)
    assert s3._bits(node) == s3._bits(T)
    L, H = s3.levels_from_sweep(grid, node, [True] * len(grid))
    assert (L, H) == (0.0, T) and s3.level_sources(grid, L, H, info, 'kp_xy') == ('0', 'trần bảng chặn')
    assert s3.levels_from_sweep(grid, node, [False] * (len(grid) - 1) + [True]) == (T, T)   # only the default run
    with pytest.raises(s3.S03Stop) as e:                           # the default run (at T) does not complete
        s3.levels_from_sweep(grid, node, [True] * (len(grid) - 1) + [False])
    assert e.value.case == '(i)'
    assert s3.grid_label(T, node, info['noi']) == 'mac_dinh'
    nodes = s3.w_nodes('PID', 'kp_xy', d, 0.0, 41.0)
    want, a, _ = _w_nodes_want('kp_xy', d, 0.0, 41.0, T)
    assert _bl(nodes) == _bl(want) == _bl([x for x in grid if x != 0.0])
    assert nodes[0] == a == d / 10.0 and nodes[-1] == T == info['b_n'] and len(nodes) == 6 and d not in nodes


def test_c3_4_default_added_beside_the_log_points_merged_into_the_ceiling():
    """Codex C3-4, the other case: PID kp_wz, T = 4.6, default 4.6 (1 - 5e-10), bounds 0 .. 100, so
    b = 100 and the default is no log point but a point added on its own. The cut makes it T; the W
    nodes (11 log points and the extension, cut, T appended) are the log points below T plus T."""
    T = 4.6
    d = 4.6 * (1 - 5e-10)
    assert d < T and math.isclose(d, T, rel_tol=1e-9) and _f32v(d) == _f32v(T)
    g0, i0 = s3.sweep_grid_n('kp_wz', d, 0.0, 100.0)
    assert len(s3.base_grid(g0, i0)) == 13 and i0['b'] == 100.0    # 0, 11 log points, the default beside them
    grid, info = s3.sweep_grid_cut('PID', 'kp_wz', d, 0.0, 100.0)  # S03Stop before the fix
    assert _bl(grid) == _bl([x for x in g0 if x < d] + [T]) and d not in grid
    assert info['b_n'] == T and info['noi'] == [] and s3.base_grid(grid, info) == grid
    node = s3.default_node(grid, info, d)
    assert s3._bits(node) == s3._bits(T)
    assert s3.levels_from_sweep(grid, node, [True] * len(grid)) == (0.0, T)
    assert s3.grid_label(T, node, info['noi']) == 'mac_dinh'
    nodes = s3.w_nodes('PID', 'kp_wz', d, 0.0, 100.0)
    want, a, _ = _w_nodes_want('kp_wz', d, 0.0, 100.0, T)
    assert _bl(nodes) == _bl(want) == _bl([x for x in grid if x != 0.0])
    assert nodes[0] == a and nodes[-1] == T and len(nodes) == 6 and d not in nodes


def test_c3_4_default_node_boundaries():
    """default_node at the edges: the default exactly T (its own point), at the edge of the 1e-9
    tolerance (merged, inclusive as math.isclose), just outside it (its own point below the added
    T), far from T (itself), and within 1e-9 ABOVE T (dropped like any point above: S03 stops)."""
    T = 41.0
    grid, info = s3.sweep_grid_cut('PID', 'kp_xy', T, 0.0, 41.0)
    assert grid[-1] == T and s3._bits(s3.default_node(grid, info, T)) == s3._bits(T)
    d = T * (1 - 1e-9)
    assert math.isclose(d, T, rel_tol=1e-9) and _f32v(d) == 41.0
    grid, info = s3.sweep_grid_cut('PID', 'kp_xy', d, 0.0, 41.0)
    assert d not in grid and grid[-1] == T and s3.default_node(grid, info, d) == T
    d = T * (1 - 2e-9)
    assert not math.isclose(d, T, rel_tol=1e-9) and _f32v(d) == 41.0
    grid, info = s3.sweep_grid_cut('PID', 'kp_xy', d, 0.0, 41.0)
    assert grid[-2:] == [d, T] and s3._bits(s3.default_node(grid, info, d)) == s3._bits(d)
    grid, info = s3.sweep_grid_cut('PID', 'kp_xy', 1.5, 0.0, 6.0)
    assert s3.default_node(grid, info, 1.5) == 1.5
    with pytest.raises(s3.S03Stop, match='ceiling'):
        s3.sweep_grid_cut('PID', 'kp_xy', T * (1 + 5e-10), 0.0, 41.0)


def test_c3_4_default_another_float32_than_the_ceiling_stops(monkeypatch):
    """The point T stands for the default only when the firmware receives one value for both: a
    default within 1e-9 below T that rounds to another float32 than T still stops S03."""
    T = (4.0 + 2.0 ** -22) * (1 + 2e-10)          # just above the middle of two float32 neighbours
    d = T * (1 - 5e-10)                           # just below it
    assert d < T and math.isclose(d, T, rel_tol=1e-9)
    assert _f32v(T) == 4.0 + 2.0 ** -21 and _f32v(d) == 4.0
    g0, i0 = s3.sweep_grid_n('kp_wz', d, 0.0, 6.0)
    pts, info = s3.cut_grid(g0, i0, T)
    assert d not in pts and pts[-1] == T
    with pytest.raises(s3.S03Stop, match='float32'):
        s3.default_node(pts, info, d)
    monkeypatch.setattr(s3, 'ceiling', lambda ctrl, name: T)
    with pytest.raises(s3.S03Stop, match='PID kp_wz: .*float32'):
        s3.sweep_grid_cut('PID', 'kp_wz', d, 0.0, 6.0)


@pytest.mark.parametrize('name, d, lo, hi, want', [('omega_o_xy', 30.0, 5.0, 48.0, 48.0),
                                                   ('omega_o_wz', 30.0, 5.0, 48.0, 48.0),
                                                   ('omega_o_wz', 3.0, 1.0, 20.0, 30.0)])
def test_c3_5_omega_o_b_n_is_the_largest_grid_point(name, d, lo, hi, want):
    """Codex C3-5 (S03 v8.6 section 3.1: b_n is the largest point of the grid after the cut): omega_o
    has no extension, so b_n of sweep_grid_cut is the end b of its grid (48 with the campaign
    default); sweep_grid_n itself is unchanged (None)."""
    g0, i0 = s3.sweep_grid_n(name, d, lo, hi)
    assert i0['b_n'] is None
    g, i = s3.sweep_grid_cut('ADRC', name, d, lo, hi)
    assert _bl(g) == _bl(g0) and i['n_bo'] == 0 and i['noi'] == [] and i['n_noi'] == 0
    assert s3._bits(i['b_n']) == s3._bits(want) and i['b_n'] == g[-1] == i['b']
    assert {k: v for k, v in i.items() if k not in ('T', 'n_bo', 'b_n')} == {k: v for k, v in i0.items() if k != 'b_n'}


@needs_dll
def test_c3_4_main_runs_the_default_at_the_ceiling(tmp_path, monkeypatch, capsys):
    """Codex C3-4 through main(): with the PID kp_xy default 41 (1 - 5e-10) S03 no longer stops in the
    sweep; the grid point 41.0 is the default run (is_default, luoi mac_dinh, tran true), the level
    rule starts from it, levels.csv keeps the firmware default."""
    import csv as _csv
    d = C34_D
    monkeypatch.setitem(FAKE_DEFAULTS['PID'], 'kp_xy', d)
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0 and '[STOP]' not in console
    sweep = [r for r in _csv.DictReader(open(out / 'sweep.csv', encoding='utf-8'))
             if r['controller'] == 'PID' and r['factor'] == 'kp_xy']
    top = [r for r in sweep if r['value'] == '41.0']
    assert len(top) == len(s3.SPEEDS)
    assert {(r['is_default'], r['luoi'], r['tran']) for r in top} == {('true', 'mac_dinh', 'true')}
    rest = [r for r in sweep if r['value'] != '41.0']
    assert rest and all(float(r['value']) < d for r in rest)
    assert {(r['is_default'], r['tran']) for r in rest} == {('false', 'false')}
    assert 'mac_dinh' not in {r['luoi'] for r in rest}
    lv = {(r['controller'], r['factor']): r for r in _csv.DictReader(open(out / 'levels.csv', encoding='utf-8'))}
    r = lv[('PID', 'kp_xy')]
    assert (r['L'], r['H'], r['H_nguon'], r['b_n'], r['default']) == ('0.0', '41.0', 'trần bảng chặn', '41.0', repr(d))
    assert float(r['b']) == 10.0 * d
    assert max(calls) == 7                                          # the main design ran


# ── S03 v8.5/v8.6: the screening step wired into main() (C3-1, section 7, kep_frac) ─────────────

MU3 = ['--mu', '0.4', '0.5', '0.6']                       # three mu levels -> the step runs


@needs_dll
def test_v86_main_no_public_design_or_sealed_hash(tmp_path, monkeypatch, capsys):
    """C3-1: after a clean run the root has no design.csv, no public file has a point_id
    column, meta.yaml file_sha256 has no design entry, and the SHA-256 of sealed/design.csv and
    sealed/levels_sangloc.csv appears in no public file and not on the console."""
    import csv as _csv
    import json as _json
    code, out, console, _, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert not (out / 'design.csv').exists()
    for name in ('sweep.csv', 'sweep_counts.csv', 'levels.csv', 'kiem_ngoai_mien.csv'):
        assert 'point_id' not in next(_csv.reader(open(out / name, encoding='utf-8')))
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    sha = _json.loads(next(x for x in meta.splitlines() if x.startswith('file_sha256: '))
                      [len('file_sha256: '):])
    assert not any('design' in k for k in sha)
    pub = _public_text(out)
    for s_ in ('design.csv', 'levels_sangloc.csv'):
        h = s3.sha256_file(str(out / 'sealed' / s_))
        assert not any(h in t for t in pub.values()) and h not in console
    header = next(_csv.reader(open(out / 'sealed' / 'design.csv', encoding='utf-8')))
    assert header[-1] == 'ngoai_tam_giac'
    assert len(list(_csv.DictReader(open(out / 'sealed' / 'design.csv', encoding='utf-8')))) == 2 * 371


@needs_dll
def test_v86_main_sealed_manifest_lists_the_screening_files(tmp_path, monkeypatch, capsys):
    """v8.5 section 7: sealed/manifest.json lists levels_sangloc.csv, design.csv, diag.txt and
    every file under sangloc/ (stop.json only when stopped, see the C3-1 stop-channel test);
    every hash is the file's."""
    import json as _json
    code, out, _, _, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    man = _json.load(open(out / 'sealed' / 'manifest.json', encoding='utf-8'))
    for name in ('runs.csv', 'effects.csv', 'levels_sangloc.csv', 'design.csv', 'diag.txt',
                 'notes.json', 'sangloc/ket_qua.json'):
        assert name in man['files'], name
    for name, sha in man['files'].items():
        assert s3.sha256_file(str(out / 'sealed' / name)) == sha
    assert 'stop.json' not in man['files']


@needs_dll
def test_v86_main_trial_levels_sangloc_is_the_fallback(tmp_path, monkeypatch, capsys):
    """--trial (one mu level): the step cannot run (it needs three mu levels, S03 2.2):
    sealed/levels_sangloc.csv holds the public levels rows plus nguon_mien 'du_phong' and the
    reason (columns LEVEL_SL_COLS), sangloc/ket_qua.json says khong_chay_sang_loc, and the
    design decodes from the sealed file read back (S03 3.2 Nguon muc)."""
    import csv as _csv
    import json as _json
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert next(_csv.reader(open(out / 'sealed' / 'levels_sangloc.csv', encoding='utf-8'))) == \
        sl.LEVEL_SL_COLS
    pub = {(r['controller'], r['factor']): r
           for r in _csv.DictReader(open(out / 'levels.csv', encoding='utf-8'))}
    slv = list(_csv.DictReader(open(out / 'sealed' / 'levels_sangloc.csv', encoding='utf-8')))
    assert len(slv) == 14
    for r in slv:
        p = pub[(r['controller'], r['factor'])]
        assert all(r[k] == p[k] for k in s3.LEVEL_COLS)
        assert r['nguon_mien'] == 'du_phong' and 'three mu levels' in r['ly_do']
    doc = _json.load(open(out / 'sealed' / 'sangloc' / 'ket_qua.json', encoding='utf-8'))
    for c in s3.CONTROLLERS:
        assert doc[c]['nhanh'] == 'khong_chay_sang_loc' and doc[c]['nguon_mien'] == 'du_phong'
        assert 'three mu levels' in doc[c]['ly_do'][0]
    design = list(_csv.DictReader(open(out / 'sealed' / 'design.csv', encoding='utf-8')))
    for c in s3.CONTROLLERS:
        lv = sl.read_levels(str(out / 'sealed' / 'levels_sangloc.csv'), c)
        for r in [r for r in design if r['controller'] == c]:
            for k, n in enumerate(s3.FACTORS[c]):
                assert r[f'{n}_value'] == repr(s3.decode(float(r[f'X{k + 1}']), *lv[n]))


@needs_dll
def test_v86_main_accepted_box_feeds_the_design(tmp_path, monkeypatch, capsys):
    """A screening box strictly inside W (S03 3.1b): the PID rows of sealed/levels_sangloc.csv
    are 'sang_loc' with L, H = the box and its labels; every PID design value is
    decode(code, L, H, scale) of THAT file and differs from the public-levels decode; ADRC falls
    back; public levels.csv is unchanged; meta.yaml and the console carry no result word or box
    value of the step (spec 11.5 (e))."""
    import csv as _csv
    lo_b, hi_b = _tg.BOUNDS['PID']
    nodes = {n: s3.w_nodes('PID', n, FAKE_DEFAULTS['PID'][n], lo_b[j], hi_b[j])
             for j, n in enumerate(s3.FACTORS['PID'])}
    box = {n: (ns[1], ns[2]) for n, ns in nodes.items()}          # strictly inside W

    def step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc, tool_check=None):
        res = _fake_run_step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc)
        res['mien'] = dict(res['mien'], PID=box)
        res['nhanh'] = dict(res['nhanh'], PID='nhan')
        res['ly_do'] = dict(res['ly_do'], PID=[])
        return res

    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, run_step=step,
                                             argv=MU3)
    assert code == 0
    slv = list(_csv.DictReader(open(out / 'sealed' / 'levels_sangloc.csv', encoding='utf-8')))
    pid = {r['factor']: r for r in slv if r['controller'] == 'PID'}
    for n, r in pid.items():
        assert r['nguon_mien'] == 'sang_loc'
        assert float(r['L']) == box[n][0] and float(r['H']) == box[n][1]
    assert {r['nguon_mien'] for r in slv if r['controller'] == 'ADRC'} == {'du_phong'}
    pub = {r['factor']: r for r in _csv.DictReader(open(out / 'levels.csv', encoding='utf-8'))
           if r['controller'] == 'PID'}
    design = [r for r in _csv.DictReader(open(out / 'sealed' / 'design.csv', encoding='utf-8'))
              if r['controller'] == 'PID']
    differ = False
    for r in design:
        for k, n in enumerate(s3.FACTORS['PID']):
            want = s3.decode(float(r[f'X{k + 1}']), box[n][0], box[n][1], pid[n]['scale'])
            assert r[f'{n}_value'] == repr(want)
            differ |= want != s3.decode(float(r[f'X{k + 1}']), float(pub[n]['L']),
                                        float(pub[n]['H']), pub[n]['scale'])
    assert differ                                                 # the box changed the domain
    assert all(pub[n]['L'] != repr(box[n][0]) or pub[n]['H'] != repr(box[n][1]) for n in pid)
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    # the 'interpretations' block legitimately explains du_phong/khong_chay_sang_loc; the RESULT
    # words and the box values must not appear anywhere else (spec 11.5 (e))
    meta_res = '\n'.join(l for l in meta.splitlines() if not l.startswith('  - '))
    assert 'du_phong' not in meta_res and 'nguon_mien' not in meta_res
    assert 'sang_loc' not in console and 'du_phong' not in console
    for n in box:
        assert repr(box[n][0]) not in meta and repr(box[n][1]) not in meta


@needs_dll
def test_v86_main_run_order_and_plan_meta(tmp_path, monkeypatch, capsys):
    """Section 7 run order and 8.9: at the FIRST firmware run (the default check) the public
    meta.yaml already says 'running: started; the sweep is not finished' and holds the
    'sang_loc' plan line (m_min 86, T_max_gio 24.0, the label, the numbers-file SHA-256); when
    the step is called the status is 'running: the sweep is done ...', the four public sweep
    files exist, sealed/levels_sangloc.csv and sealed/design.csv do not, and the sweep of BOTH
    controllers is complete (items 10, 13)."""
    import csv as _csv
    seen = {}

    def step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc, tool_check=None):
        root = os.path.dirname(sealed)
        meta = open(os.path.join(root, 'meta.yaml'), encoding='utf-8').read()
        seen['meta_running'] = 'status: "running: the sweep is done' in meta
        seen['plan'] = next((l for l in meta.splitlines() if l.startswith('sang_loc: ')), '')
        seen['public'] = {n: os.path.isfile(os.path.join(root, n))
                          for n in ('sweep.csv', 'sweep_counts.csv', 'levels.csv',
                                    'kiem_ngoai_mien.csv')}
        seen['not_yet'] = not os.path.isfile(os.path.join(sealed, 'levels_sangloc.csv')) \
            and not os.path.isfile(os.path.join(sealed, 'design.csv'))
        seen['sweep_done'] = {r['controller'] for r in _csv.DictReader(
            open(os.path.join(root, 'sweep.csv'), encoding='utf-8'))} == set(s3.CONTROLLERS)
        return _fake_run_step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc)

    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, run_step=step,
                                             argv=MU3)
    assert code == 0 and seen['meta_running'] and seen['sweep_done'] and seen['not_yet']
    assert all(seen['public'].values())
    for frag in ('"m_min": 86', '"T_max_gio": 24.0', sl.T_MAX_LABEL, '0' * 64):
        assert frag in seen['plan']
    # S03 8 condition 9: already at the first firmware run (the default check) the public
    # meta.yaml holds the numbers-file SHA-256 and the plan of section 3.1b
    first = calls.seen['meta_at_first_run']
    assert 'status: "running: started; the sweep is not finished"' in first
    plan0 = next(l for l in first.splitlines() if l.startswith('sang_loc: '))
    for frag in ('"m_min": 86', '"T_max_gio": 24.0', sl.T_MAX_LABEL, '0' * 64):
        assert frag in plan0
    # sealed/design.csv existed before the first main-design run (_run_main records its content)
    assert calls.seen['design_before_first_main_run'].count('\n') == 1 + 2 * 371
    assert calls.seen['manifest_at_first_test'] is True          # sealed/ closed before the test set


@needs_dll
@pytest.mark.parametrize('kind', ['stop', 'runtime'])
def test_v86_main_screening_stop_stays_sealed(tmp_path, monkeypatch, capsys, kind):
    """3.1b item 13: a failure inside the screening step is a sealed stop: 'SECRET 12.345' is
    on neither the console nor any public file, sealed/stop.json holds the reason and
    pha 'sang_loc', status 'stopped after the sweep', no main-design run, no sealed_test/; a
    non-S03Stop also leaves its traceback in sealed/diag.txt, never on stderr."""
    import json as _json

    def step(*a, **k):
        if kind == 'stop':
            raise s3.S03Stop('SECRET 12.345')
        raise RuntimeError('SECRET 12.345')

    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, run_step=step,
                                             argv=MU3)
    assert code == 2
    for t in [console, calls.seen['stderr']] + list(_public_text(out).values()):
        assert 'SECRET' not in t and '12.345' not in t and 'Traceback' not in t
    stop = _json.load(open(out / 'sealed' / 'stop.json', encoding='utf-8'))
    assert stop['pha'] == 'sang_loc' and 'SECRET 12.345' in stop['stop']
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'stopped after the sweep (the reason is in sealed/stop.json)' in meta
    assert max(calls) <= 1 and not (out / 'sealed_test').exists()
    diag = (out / 'sealed' / 'diag.txt').read_text(encoding='utf-8')
    if kind == 'runtime':
        assert 'RuntimeError: SECRET 12.345' in diag and 'Traceback' in diag


@needs_dll
def test_v86_main_sealed_phase_output_goes_to_diag(tmp_path, monkeypatch, capsys):
    """3.1b item 13: prints and warnings raised inside the sealed phases (here in the fake
    step and in a main-design run) land in sealed/diag.txt, not on the console or stderr."""
    def step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc, tool_check=None):
        print('SANG_PRINT')
        warnings.warn('SANG_WARN', RuntimeWarning)
        return _fake_run_step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc)

    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, run_step=step,
                                             design_warn=True, argv=MU3)
    assert code == 0
    for word in ('SANG_PRINT', 'SANG_WARN', 'FakeDesignWarning'):
        assert word not in console and word not in calls.seen['stderr']
    diag = (out / 'sealed' / 'diag.txt').read_text(encoding='utf-8')
    assert 'SANG_PRINT' in diag and 'warning: RuntimeWarning: SANG_WARN' in diag
    assert 'warning: RuntimeWarning: FakeDesignWarning' in diag


@needs_dll
def test_v86_main_tool_check_public_invariance(tmp_path, monkeypatch, capsys):
    """spec 11.5 (e) and 8.11: --sangloc-thu end to end with the REAL sl.run_step (fake
    safe_run_gate, _firmware and an in-process run_jobs): two runs with different fake scorings
    (every set feasible / none feasible) produce identical console text and identical public
    files except the meta.yaml date / elapsed_s / script lines; trial: true; the sealed
    screening files exist, ket_qua says hoan_tat, and the two runs differ inside sealed/."""
    import json as _json
    ufile = str(tmp_path / 'u_thu.csv')
    mk.write(ufile, base=sl.TRIAL_SEED_BASE)                        # the tool-check file (8.11)

    def gate_all(fw, path, ctrl, g, v, mu):                         # every set feasible
        return dict(status='ok', completed='true', e_perp_max=0.05, tv_duty=0.4, sat_frac=0.0,
                    t_end=1.0, e_theta_lane_max_deg=0.5, e_theta_cmd_max_deg=0.5, ds_absmax=1.0,
                    e_trk_max=0.05, e_bao_max=0.02), {'t': np.arange(3.0)}

    def gate_none(fw, path, ctrl, g, v, mu):                        # no set feasible
        row, log = gate_all(fw, path, ctrl, g, v, mu)
        return dict(row, e_perp_max=99.0), log

    def jobs(jobs_, work, n, lp):                                   # the worker pool in-process
        return [sl.do_job(j) for j in jobs_]

    outs = []
    for gate in (gate_all, gate_none):
        monkeypatch.setattr(s4, 'safe_run_gate', gate)
        monkeypatch.setattr(sl, '_firmware', lambda dll: object())
        monkeypatch.setattr(sl, 'run_jobs', jobs)
        monkeypatch.setattr(sl, '_clock', itertools.count(1).__next__)
        code, out, console, _, _ = _run_main(
            tmp_path / ('a' if gate is gate_all else 'b'), monkeypatch, capsys,
            patch_screen=False,
            argv=['--mu', '0.4', '0.5', '0.6', '--sangloc-thu', '--screen-u', ufile,
                  '--sangloc-thu-n0', '20', '--sangloc-thu-n', '30'])
        outs.append((code, out, console))
    (code_a, out_a, con_a), (code_b, out_b, con_b) = outs
    assert code_a == code_b == 0
    assert con_a.replace(str(out_a), '') == con_b.replace(str(out_b), '')
    pub_a, pub_b = _public_text(out_a), _public_text(out_b)
    assert set(pub_a) == set(pub_b)
    for name in pub_a:
        if name.endswith('meta.yaml'):
            keep = [l for l in pub_a[name].splitlines()             # volatile meta fields only
                    if not l.startswith(('date', 'elapsed_s', 'script'))]
            want = [l for l in pub_b[name].splitlines()
                    if not l.startswith(('date', 'elapsed_s', 'script'))]
            assert keep == want
        else:
            assert pub_a[name] == pub_b[name], name
    for out in (out_a, out_b):
        assert 'trial: true' in (out / 'meta.yaml').read_text(encoding='utf-8')
        assert (out / 'sealed' / 'sangloc_thu' / 'so_dem.json').is_file()
        doc = _json.load(open(out / 'sealed' / 'sangloc' / 'ket_qua.json', encoding='utf-8'))
        for c in s3.CONTROLLERS:
            assert doc[c]['trang_thai'] == 'hoan_tat'
    assert _tree(out_a / 'sealed') != _tree(out_b / 'sealed')       # they differ inside sealed/


def test_v86_flag_validation(tmp_path, monkeypatch):
    """8.11 and the flag rules: --sangloc-thu not with --trial, needs a --screen-u whose
    basename is not the reference name, n0 in 1..SET_ROWS['S0'], n in 1..SET_ROWS['S1'],
    --screen-r 0; the -n0/-n flags are refused without it; --screen-r >= 0; a --screen-r >= 1
    rerun (item 3: a NEW numbers file) needs --screen-u under another name; --jobs >= 1;
    --trial refuses --screen-u and --screen-r != 0."""
    base = ['s03_doe.py', '--out', str(tmp_path / 'x'), '--mu', '0.4', '0.5', '0.6']
    bad = [['s03_doe.py', '--out', str(tmp_path / 'x'), '--mu', '0.5', '--trial', '--sangloc-thu'],
           base + ['--sangloc-thu'],
           base + ['--sangloc-thu', '--screen-u', sl.SCREEN_U],
           base + ['--sangloc-thu', '--screen-u', str(tmp_path / 'u.csv'),
                   '--sangloc-thu-n0', '0', '--sangloc-thu-n', '30'],
           base + ['--sangloc-thu', '--screen-u', str(tmp_path / 'u.csv'),
                   '--sangloc-thu-n0', '20', '--sangloc-thu-n', str(sl.SET_ROWS['S1'] + 1)],
           base + ['--sangloc-thu', '--screen-u', str(tmp_path / 'u.csv'),
                   '--sangloc-thu-n0', '20', '--sangloc-thu-n', '30', '--screen-r', '1'],
           base + ['--sangloc-thu-n0', '20'],
           base + ['--screen-r', '-1'],
           base + ['--screen-r', '1'],                    # a rerun needs its own numbers file
           base + ['--screen-r', '1', '--screen-u', str(tmp_path / 's03_screen_u.csv')],
           base + ['--jobs', '0'],
           ['s03_doe.py', '--out', str(tmp_path / 'y'), '--mu', '0.5', '--trial',
            '--screen-u', 'x'],
           ['s03_doe.py', '--out', str(tmp_path / 'y'), '--mu', '0.5', '--trial',
            '--screen-r', '1']]
    for av in bad:
        monkeypatch.setattr(sys, 'argv', av)
        with pytest.raises(SystemExit):
            s3.main()


@needs_dll
def test_v86_main_tool_check_same_content_stops_before_the_sweep(tmp_path, monkeypatch, capsys):
    """8.11: the tool-check numbers file must also differ in CONTENT from the committed
    reference — a byte copy of fwsim/s03_screen_u.csv is a public stop 'before the sweep'
    whose console line names the same-content rule (exit 2), no run is made and no sealed
    screening folder appears. Runs the real load_screen_u path (patch_screen=False)."""
    u = tmp_path / 'u_thu.csv'
    u.write_bytes(open(sl.SCREEN_U, 'rb').read())
    code, out, console, calls, _ = _run_main(
        tmp_path, monkeypatch, capsys, patch_screen=False,
        argv=['--mu', '0.4', '0.5', '0.6', '--sangloc-thu', '--screen-u', str(u),
              '--sangloc-thu-n0', '20', '--sangloc-thu-n', '30'])
    assert code == 2 and 'before the sweep' in console and 'same content' in console
    assert not calls
    for sub in ('sangloc', 'sangloc_thu'):
        d = out / 'sealed' / sub
        assert not d.is_dir() or not list(d.iterdir())


@needs_dll
def test_v86_main_screen_file_failure_stops_before_the_sweep(tmp_path, monkeypatch, capsys):
    """8.9 and item 13 first item: a numbers-file failure is a public stop BEFORE the sweep
    (no run at all); the 'sang_loc' plan line of meta.yaml still holds the file info read
    (.screen_info); exit code 2."""
    def bad_load(path=None, trial=False):
        exc = s3.S03Stop('screen numbers: fake_screen_u.csv: broken (S03 v8.5 8.9)')
        exc.screen_info = dict(file='fake_screen_u.csv', sha256='a' * 64,
                               problems=['broken on purpose'])
        raise exc

    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys,
                                             load_screen=bad_load, argv=MU3)
    assert code == 2 and 'before the sweep' in console and 'broken' in console
    assert not calls
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    plan = next(l for l in meta.splitlines() if l.startswith('sang_loc: '))
    assert 'fake_screen_u.csv' in plan and 'broken on purpose' in plan
    assert 'stopped before the sweep' in meta


@needs_dll
def test_v86_main_notes_pha(tmp_path, monkeypatch, capsys):
    """Section 7 [v8.5]: sealed/notes.json 'pha' holds the 'quet' entry (actual = the rows of
    sweep.csv, planned = what the grids imply, one process), the entries the step returned, and
    per controller 'tam_va_hai_hop', 'truc', 'kiem_tat_dinh_3_4' whose actual counts equal the
    runs.csv rows and the determinism rows; the old keys are still there."""
    import csv as _csv
    import json as _json
    pha_in = [dict(pha='S0', controller='all', n_bo_gain={'PID': 5}, n_luot_ke_hoach=None,
                   n_luot_thuc=5, n_luot_thuc_theo_controller=None, n_chay_lai_chan_doan=0,
                   giay_may=1.0, so_tien_trinh=2)]

    def step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc, tool_check=None):
        res = _fake_run_step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc)
        res['pha'] = pha_in
        return res

    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, run_step=step,
                                             argv=MU3)
    assert code == 0
    notes = _json.load(open(out / 'sealed' / 'notes.json', encoding='utf-8'))
    pha = notes['pha']
    n_sweep = len(list(_csv.DictReader(open(out / 'sweep.csv', encoding='utf-8'))))
    q = next(p for p in pha if p['pha'] == 'quet')
    planned = sum(3 + 3 * sum(len(s3.sweep_grid_cut(c, n, FAKE_DEFAULTS[c][n], lo, hi)[0])
                              for n, lo, hi in zip(s3.FACTORS[c], *_tg.BOUNDS[c]))
                  + 3 * (3 - 1) for c in s3.CONTROLLERS)
    assert (q['controller'], q['n_luot_thuc'], q['n_luot_ke_hoach'], q['so_tien_trinh']) == \
        ('all', n_sweep, planned, 1)
    assert any(p.get('n_bo_gain') == {'PID': 5} for p in pha if p['pha'] == 'S0')
    runs = list(_csv.DictReader(open(out / 'sealed' / 'runs.csv', encoding='utf-8')))
    det = list(_csv.DictReader(open(out / 'sealed' / 'determinism_check.csv', encoding='utf-8')))
    for c in s3.CONTROLLERS:
        th = next(p for p in pha if p['pha'] == 'tam_va_hai_hop' and p['controller'] == c)
        tr = next(p for p in pha if p['pha'] == 'truc' and p['controller'] == c)
        kt = next(p for p in pha if p['pha'] == 'kiem_tat_dinh_3_4' and p['controller'] == c)
        assert th['n_luot_thuc'] + tr['n_luot_thuc'] == sum(r['controller'] == c for r in runs)
        assert tr['n_luot_thuc'] == sum(r['controller'] == c and r['box'] == 'axial'
                                      for r in runs)
        assert kt['n_luot_thuc'] == sum(r['controller'] == c for r in det)
    for k in ('inner_rule_speeds', 'no_conclusion_speeds', 'ngoai_tam_giac_points',
              'elapsed_s_total', 'elapsed_s_main_design', 'kep_frac', 'run_error_reruns',
              'stop'):
        assert k in notes


def test_v86_kep_frac_fields():
    """S03 v8.6 2.1 last item: PID with a 'kep' log channel -> the fwsim.kep_indicator
    fractions; ADRC, a run_error row (log None) and a DLL without the channel -> the three
    columns empty; a kep_lech flag -> S03Stop."""
    t = np.arange(5) * 0.01
    en = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    kep = np.zeros((5, len(fwsim.KEP_COLS)))
    kep[0, 0] = kep[1, 0] = kep[4, 0] = 1.0            # vx flagged on 3 of the 4 motor-on ticks
    kep[2, 1] = 1.0                                  # vy on 1
    log = dict(t=t, en=en, kep=kep)
    row = dict(t_end=0.05)
    want = fwsim.kep_indicator(log, row['t_end'])
    got = s3.kep_frac_fields('PID', row, log)
    assert got == {k: want[k] for k in s3.KEP_FRAC_COLS}
    assert (got['kep_frac_vx'], got['kep_frac_vy'], got['kep_frac_wz']) == (0.75, 0.25, 0.0)
    empty = {k: '' for k in s3.KEP_FRAC_COLS}
    assert s3.kep_frac_fields('ADRC', row, log) == empty
    assert s3.kep_frac_fields('PID', row, None) == empty
    assert s3.kep_frac_fields('PID', row, dict(t=t, en=en)) == empty
    bad = dict(t=t, en=en, kep=np.array(kep, copy=True))
    bad['kep'][0, fwsim.KEP_COLS.index('kep_lech')] = 1.0
    with pytest.raises(s3.S03Stop):
        s3.kep_frac_fields('PID', row, bad)


@needs_dll
def test_v86_main_kep_frac_columns(tmp_path, monkeypatch, capsys):
    """Through main(): a fake log with a kep channel -> the PID rows of runs.csv and
    runs_test.csv hold the kep_indicator fractions (vx 1.0 here, vy/wz 0.0), the ADRC rows are
    empty, sweep.csv has no kep column, and meta.yaml records kep_channel."""
    import csv as _csv
    code, out, console, calls, _ = _run_main(tmp_path, monkeypatch, capsys, kep=True)
    assert code == 0
    for f in ('sealed/runs.csv', 'sealed_test/runs_test.csv'):
        rows = list(_csv.DictReader(open(out / f)))
        assert {r['kep_frac_vx'] for r in rows if r['controller'] == 'PID'} == {'1.0'}
        assert {r['kep_frac_vy'] for r in rows if r['controller'] == 'PID'} == {'0.0'}
        assert {r['kep_frac_vx'] for r in rows if r['controller'] == 'ADRC'} == {''}
    assert not set(s3.KEP_FRAC_COLS) & set(next(_csv.reader(open(out / 'sweep.csv'))))
    assert 'kep_channel: ' in (out / 'meta.yaml').read_text(encoding='utf-8')


@needs_dll
def test_v86_kep_frac_real_run(path):
    """kep_frac_fields on a real PID run equals fwsim.kep_indicator of the log at the row's
    t_end (skipped when the DLL has no kep channel)."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.kep_cols:
        pytest.skip('the DLL has no kep channel')
    row, log = s3.run_one(fw, path, 'PID', s3.defaults(fw, 'PID'), 0.8, 0.5)
    got = s3.kep_frac_fields('PID', row, log)
    want = fwsim.kep_indicator(log, row['t_end'])
    assert got == {k: want[k] for k in s3.KEP_FRAC_COLS}

