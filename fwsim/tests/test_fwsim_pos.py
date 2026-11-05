"""Position loop on fwsim: pos_ctrl.c of the ESP32 firmware (AGV feat/pos-loop, O1 PA3,
Liam approved A1 + B1 in the CODE session 04/11/2026). A1: the command pose is integrated on
the ESP32 from the body-velocity command; B1: heading stays with heading.c, the loop acts on
x, y only. Logic checks on the rigid, noise-free plant, not performance numbers.
Run: python -m pytest fwsim/tests  (needs fwsim/build/fwsim.dll built from firmware with pos_ctrl.c)"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import o1_feasibility as o1  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
pytestmark = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')

P = {n: i for i, n in enumerate(fwsim.POS_COLS)}
ON = dict(pos_on=1, pos_kp=6.0, pos_ki=3.0)
# Firmware defaults since 04/11 (Liam, CODE session: "làm cách 1 ban khuyến nghị đi"): loop, (a), (d) ON.
# Tests of one part alone turn the others off by name.
OFF = dict(pos_on=0, pos_e_win=0.0, still_on=0)
DEFAULTS = (1, 6.0, 3.0, 0.5, 0.005, 0.010, 1, 0.05, 500000)   # pos_params(): on .. t_still_us


@pytest.fixture(scope='module')
def fw():
    f = fwsim.FirmwareSim()
    if not f.pos_cols:
        pytest.skip('fwsim.dll built from firmware without pos_ctrl.c')
    return f


class _RigidParams(fwsim.MecanumParams):
    def __init__(self):
        super().__init__()
        self.ground_model = 'rigid'


@pytest.fixture(autouse=True)
def rigid_plant(monkeypatch):
    monkeypatch.setattr(fwsim, 'MecanumParams', _RigidParams)


def col(log, name):
    return log['pos'][:, P[name]]


def const(vx=0.0, vy=0.0, wz=0.0):
    return lambda t: (vx, vy, wz)


def test_defaults_on(fw):
    """Firmware defaults (campaign firmware): position loop, (a) window, (d) hold all on."""
    fwsim.simulate(fw, const(vx=0.3), 0.5, imu={})
    assert fw.pos_params() == pytest.approx(DEFAULTS, rel=1e-6)


def test_off_no_correction(fw):
    """Turned off by name (pos_on = 0): no correction, but the poses are still tracked."""
    log = fwsim.simulate(fw, const(vx=0.3, wz=0.3), 3.0, imu={}, fw_params=OFF)
    assert fw.pos_params()[0] == 0
    assert np.all(col(log, 'pos_on') == 0)
    assert np.all(col(log, 'pos_corr_vx') == 0) and np.all(col(log, 'pos_corr_vy') == 0)
    assert col(log, 'pos_x_ref')[-1] > 0.5


def test_command_pose_follows_planned_figure8(fw):
    """A1: the pose integrated on the ESP32 from the 50 Hz held command stays on the planned
    figure 8 (plant frame: floor path rotated by -45 deg) and ends at the start."""
    sch, path = o1.Schedule(0.5), o1.Path8()
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + 1.0, imu={}, fw_params=ON)
    c, s = math.cos(o1.HEAD0), math.sin(o1.HEAD0)
    xr, yr = col(log, 'pos_x_ref'), col(log, 'pos_y_ref')
    fx, fy = c * xr - s * yr, s * xr + c * yr               # to the floor frame
    d = [path.nearest(x, y, 0.0, path.S)[1] for x, y in zip(fx[::10], fy[::10])]
    assert max(d) < 0.02
    assert math.hypot(xr[-1], yr[-1]) < 0.02


def test_estimate_close_to_true_pose_on_rigid_plant(fw):
    """Rigid plant, no noise: encoder odometry + IMU yaw stays within 2 cm of the true pose."""
    sch = o1.Schedule(0.5)
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + 1.0, imu={}, fw_params=ON)
    err = np.hypot(col(log, 'pos_x') - log['pose'][:, 0], col(log, 'pos_y') - log['pose'][:, 1])
    assert err.max() < 0.02
    assert np.mean(col(log, 'pos_th_src')[log['t'] > 0.5] == 1) > 0.99


def test_heading_estimate_takes_imu_yaw(fw):
    """While heading.c runs, theta of the estimate is the IMU yaw (psi_imu), not the encoder
    heading: a yaw that drifts 0.1 rad/s away from the plant (hook on the game RV reports)
    shows up in pos_th. The offset is constant: the encoder heading of the ticks before the
    first IMU report (50 ms at 20 Hz), kept when the source switches to the IMU."""
    drift = lambda t, kind, v: v + math.degrees(0.1 * t) if kind == 'rv' else v  # noqa: E731
    log = fwsim.simulate(fw, const(vx=0.3), 3.0, imu=dict(hook=drift), fw_params=ON)
    head_on = log['imu'][:, fwsim.IMU_COLS.index('head_on')] == 1
    psi_imu = log['imu'][:, fwsim.IMU_COLS.index('psi_imu')]
    assert head_on[log['t'] > 0.5].all()
    d = np.angle(np.exp(1j * (col(log, 'pos_th') - psi_imu)))
    assert np.ptp(d[head_on]) < 1e-5 and np.abs(d[head_on]).max() < 0.01
    assert abs(psi_imu[-1] - log['pose'][-1, 2]) > 0.2         # the drift is really there


def test_loop_cuts_estimated_error_on_straight_line(fw):
    """Straight 0.3 m/s for 6 s: the per-wheel plant parameters (S01) drift the car; the loop
    keeps the estimated pose closer to the command pose than without it."""
    off = fwsim.simulate(fw, const(vx=0.3), 6.0, imu={}, fw_params=OFF)
    on = fwsim.simulate(fw, const(vx=0.3), 6.0, imu={}, fw_params=ON)
    e = lambda g: np.hypot(col(g, 'pos_x_ref') - col(g, 'pos_x'), col(g, 'pos_y_ref') - col(g, 'pos_y'))  # noqa: E731
    assert e(on)[-1] < 0.5 * e(off)[-1]


def test_correction_saturates_and_integral_does_not_wind_up(fw):
    """Phase 1: wheels never turn (counts stay 0) while the command asks 0.3 m/s for 3 s: the
    error grows to 0.9 m and the correction holds at v_corr_max. Phase 2: the command pose
    stops and the wheels turn forward at 0.6 m/s, so the estimate passes the command pose.
    Without anti-windup the integral of phase 1 (about 1.3 m s) keeps the correction at
    +v_corr_max long after the estimate has passed; with it, the correction turns negative
    once the estimate is 0.1 m past. The (a) window and (d) hold are off: the window clears the
    integral when the estimate passes the command pose, which would pass this test without
    anti-windup too (Codex round 2 Q2)."""
    fw.reset()
    fw.set_param('pos_on', 1)
    fw.set_param('pos_e_win', 0.0)
    fw.set_param('still_on', 0)
    fw.set_run(True, False)
    k = 0
    for _ in range(300):                          # phase 1, encoders stuck at 0
        k += 1
        fw.tick(k * fwsim.FW_DT_US, [0, 0, 0, 0], (0.3, 0.0, 0.0))
    vmax = fw.pos_params()[3]
    assert fw.pos_last()[P['pos_corr_vx']] == pytest.approx(vmax, abs=1e-6)
    r = fwsim.MecanumParams().r
    per_tick = 0.6 * fwsim.FW_DT_US * 1e-6 / r * fwsim.COUNTS_PER_REV / (2 * math.pi)
    n, seen = 0.0, None
    for _ in range(400):                          # phase 2, all wheels forward
        k += 1
        n += per_tick
        c = int(math.floor(n))
        fw.tick(k * fwsim.FW_DT_US, [c, c, c, c], (0.0, 0.0, 0.0))
        p = fw.pos_last()
        if p[P['pos_x']] - p[P['pos_x_ref']] > 0.1:
            seen = p[P['pos_corr_vx']]
            break
    assert seen is not None, 'estimate never passed the command pose: check the count direction'
    assert seen < 0.0


def test_run_restart_resets_poses(fw):
    """run_active low then high: command and estimated poses restart from 0."""
    fw.reset()
    fw.set_param('pos_on', 1)
    fw.set_run(True, False)
    for k in range(1, 101):
        fw.tick(k * fwsim.FW_DT_US, [0, 0, 0, 0], (0.3, 0.0, 0.0))
    assert fw.pos_last()[P['pos_x_ref']] > 0.25
    fw.set_run(False, False)
    fw.tick(101 * fwsim.FW_DT_US, [0, 0, 0, 0], (0.0, 0.0, 0.0))
    assert fw.pos_last()[P['pos_corr_vx']] == 0.0
    n0 = fw.pos_last()[P['pos_reset_n']]
    assert fw.pos_last()[P['run_active']] == 0
    fw.set_run(True, False)
    fw.tick(102 * fwsim.FW_DT_US, [0, 0, 0, 0], (0.3, 0.0, 0.0))
    assert fw.pos_last()[P['pos_x_ref']] == pytest.approx(0.003, abs=1e-4)
    assert fw.pos_last()[P['pos_reset_n']] == n0 + 1 and fw.pos_last()[P['run_active']] == 1   # Codex P6


def test_heading_wraps_without_jump(fw):
    """Two full turns in place then a straight leg: theta estimate wraps, no pose jump; the angle
    stays in [-pi, pi] and its wrapped step is small (Codex P16)."""
    sp = lambda t: (0.0, 0.0, 2.0) if t < 2 * math.pi else (0.3, 0.0, 0.0)  # noqa: E731
    log = fwsim.simulate(fw, sp, 2 * math.pi + 2.0, imu={}, fw_params=ON)
    for name in ('pos_x', 'pos_y', 'pos_x_ref', 'pos_y_ref'):
        assert np.abs(np.diff(col(log, name))).max() < 0.01
    for name in ('pos_th', 'pos_th_ref'):
        th = col(log, name)
        assert np.abs(th).max() <= math.pi + 1e-6
        # pos_th follows the 20 Hz yaw: steps of 2 rad/s x 50 ms = 0.1 rad between reports
        assert np.abs(np.angle(np.exp(1j * np.diff(th)))).max() < 0.15
        assert (np.abs(np.diff(th)) > math.pi).any()                     # it did wrap


# ── Fixes after Codex round 1 (05/11, Liam: "Duyệt theo khuyến nghị") ─────────────────────────
class _Ticks:
    """Direct firmware ticks: command sp, all four wheels driven at wheel_v [m/s] (counts)."""
    def __init__(self, fw, **params):
        fw.reset()
        for k, v in params.items():
            fw.set_param(k, v)
        fw.set_run(True, False)
        self.fw, self.k, self.n = fw, 0, 0.0
        self.per_mps = fwsim.FW_DT_US * 1e-6 / fwsim.MecanumParams().r * fwsim.COUNTS_PER_REV / (2 * math.pi)

    def step(self, sp, wheel_v=0.0, n=1):
        for _ in range(n):
            self.k += 1
            self.n += wheel_v * self.per_mps
            c = int(math.floor(self.n))
            out = self.fw.tick(self.k * fwsim.FW_DT_US, [c, c, c, c], sp)
        return self.fw.pos_last(), out


def _err(p):
    return math.hypot(p[P['pos_ex_b']], p[P['pos_ey_b']])


def test_block_stops_correction_toward_obstacle(fw):
    """P1: command 0 with the estimate behind the command pose; front blocked: no forward
    correction, and the integral does not grow while blocked (same output after unblocking)."""
    t = _Ticks(fw, pos_e_win=0.0, still_on=0)
    t.step((0.1, 0.0, 0.0), n=50)                       # wheels stuck: about 5 cm behind
    free, _ = t.step((0.0, 0.0, 0.0))
    assert free[P['pos_corr_vx']] > 0.1
    fw.set_param('block_fwd', 1)
    for _ in range(50):
        p, _ = t.step((0.0, 0.0, 0.0))
        assert p[P['pos_corr_vx']] == 0.0
    fw.set_param('block_fwd', 0)
    p, _ = t.step((0.0, 0.0, 0.0))
    assert p[P['pos_corr_vx']] == pytest.approx(free[P['pos_corr_vx']], abs=0.002)


def test_block_other_direction_still_corrects(fw):
    """P1: only the blocked direction is cut: back blocked, the forward correction stays."""
    t = _Ticks(fw, pos_e_win=0.0, still_on=0)
    t.step((0.1, 0.0, 0.0), n=50)
    fw.set_param('block_bwd', 1)
    p, _ = t.step((0.0, 0.0, 0.0))
    assert p[P['pos_corr_vx']] > 0.1


# ── Fixes after Codex round 2 (05/11, Liam: "bạn cứ sửa và kiểm bằng test trước đi") ──────────
@pytest.mark.parametrize('mode', ['PID', 'ADRC'])
def test_inner_loop_output_not_toward_obstacle(fw, mode):
    """Round 2 P1: the inner loop charges (wheels stuck under a forward command), then the command
    drops to 0 with the front blocked. The PID integral / ESO left over must not drive the wheels
    forward (main.c hard cutoff only sees the intent and v_cmd, both 0 here). Position loop,
    (a), (d) off to isolate the inner loop."""
    t = _Ticks(fw, pos_on=0, pos_e_win=0.0, still_on=0)
    fw.set_mode(mode)
    t.step((0.3, 0.0, 0.0), n=100)                      # wheels stuck: integral / ESO charge
    fw.set_param('block_fwd', 1)
    for _ in range(20):
        _, out = t.step((0.0, 0.0, 0.0))
        assert np.all(out[2] <= 0), out[2]              # no forward PWM on any wheel


def test_strafe_still_runs_with_front_blocked(fw):
    """Round 2 P1: the clamp is per axis, so a sideways command still drives the wheels."""
    t = _Ticks(fw, pos_on=0, pos_e_win=0.0, still_on=0)
    fw.set_param('block_fwd', 1)
    _, out = t.step((0.0, 0.3, 0.0), n=20)
    assert np.abs(out[2]).max() > 0


def test_blocked_integral_discharges(fw):
    """Round 2 P1: the integral built while behind stays from growing toward the blocked side but
    may discharge. The estimate passes the command pose while the front is blocked; once the
    discharge turns the correction negative (allowed, backward), unblocking gives no forward step
    from an old integral."""
    t = _Ticks(fw, pos_e_win=0.0, still_on=0)
    t.step((0.02, 0.0, 0.0), n=50)                      # command pose 1 cm ahead, wheels stuck
    t.step((0.0, 0.0, 0.0), n=500)                      # 5 s behind by 1 cm: integral about 0.05 m s
    p, _ = t.step((0.0, 0.0, 0.0))
    assert 0.1 < p[P['pos_corr_vx']] < 0.5              # unsaturated, integral part dominant
    fw.set_param('block_fwd', 1)
    t.step((0.0, 0.0, 0.0), wheel_v=0.1, n=20)          # estimate moves about 2 cm: 1 cm past
    for _ in range(600):
        p, _ = t.step((0.0, 0.0, 0.0))
        if p[P['pos_corr_vx']] < 0.0:
            break
    assert p[P['pos_ex_b']] < 0.0 and p[P['pos_corr_vx']] < 0.0
    fw.set_param('block_fwd', 0)
    p, _ = t.step((0.0, 0.0, 0.0))
    assert p[P['pos_corr_vx']] < 0.0


_HUGE_WZ = """
import sys, math
sys.path.insert(0, sys.argv[1])
import fwsim
f = fwsim.FirmwareSim()
f.reset()
f.set_run(True, False)
for k in range(1, 6):
    f.tick(k * fwsim.FW_DT_US, [0, 0, 0, 0], (0.0, 0.0, 1e12))
th = f.pos_last()[fwsim.POS_COLS.index('pos_th_ref')]
assert math.isfinite(th) and abs(th) <= math.pi + 1e-6, th
print('ok')
"""


def test_huge_finite_command_does_not_hang(fw):
    """Round 2 P18: a finite but huge wz (1e12) went into wrap_pi; in float32 1e10 - 2 pi == 1e10, so
    the while loop never ended. Run in a subprocess with a timeout (a hang would freeze pytest).
    The receiver range check (traj_receiver.c) is not in fwsim; this checks wrap_pi."""
    import subprocess
    r = subprocess.run([sys.executable, '-c', _HUGE_WZ, fwsim.HERE], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr


def test_yaw_dropouts_keep_imu_offset(fw):
    """P2: yaw drifts 0.1 rad/s away from the plant (the encoders do not see it) and drops out
    twice for 0.5 s. While the heading loop runs, theta = psi_imu + the same offset each time:
    the offset is not reset on every return of the yaw, only when psi0 changes."""
    def hook(t, kind, v):
        if kind == 'rv' and (1.0 <= t < 1.5 or 2.0 <= t < 2.5):
            return None
        return v + math.degrees(0.1 * t) if kind == 'rv' else v
    log = fwsim.simulate(fw, const(vx=0.3), 3.5, imu=dict(hook=hook), fw_params=ON)
    head_on = log['imu'][:, fwsim.IMU_COLS.index('head_on')] == 1
    psi_imu = log['imu'][:, fwsim.IMU_COLS.index('psi_imu')]
    assert (np.diff(head_on.astype(int)) == 1).sum() >= 3           # on at start + after each gap
    sel = head_on & (log['t'] > 0.5)
    d = np.angle(np.exp(1j * (col(log, 'pos_th') - psi_imu)))
    assert np.ptp(d[sel]) < 1e-4


def test_integral_cleared_in_window(fw):
    """P5: the integral built while catching up is cleared in the (a) window, so a new small
    command right after starts with the proportional part only."""
    t = _Ticks(fw, still_on=0)
    t.step((0.05, 0.0, 0.0), n=50)                      # 2.5 cm behind, unsaturated
    for _ in range(300):                                # wheels catch up, command 0
        p, _ = t.step((0.0, 0.0, 0.0), wheel_v=0.05)
        if p[P['pos_in_pos']] == 1:
            break
    assert p[P['pos_in_pos']] == 1
    p, _ = t.step((0.01, 0.0, 0.0))
    assert p[P['pos_corr_vx']] == pytest.approx(6.0 * p[P['pos_ex_b']], abs=0.003)


def test_integral_acts_below_saturation(fw):
    """P16: unsaturated, the output is Kp e + Ki integral on both axes (dropping the integral
    would give Kp e only)."""
    t = _Ticks(fw, pos_e_win=0.0, still_on=0)
    p, _ = t.step((0.02, 0.02, 0.0), n=100)             # error grows to 2 cm per axis, integral 0.01 m s
    for ax, e in (('pos_corr_vx', 'pos_ex_b'), ('pos_corr_vy', 'pos_ey_b')):
        assert p[P[ax]] < 0.5
        assert p[P[ax]] > 6.0 * p[P[e]] + 0.5 * 3.0 * 0.01


def test_window_hysteresis(fw):
    """P16: the (a) window is entered below e_win = 5 mm only, kept between 5 and 10 mm, left
    above e_exit = 10 mm."""
    t = _Ticks(fw, still_on=0)
    t.step((0.008, 0.0, 0.0), n=100)                    # 8 mm behind
    for _ in range(20):
        p, _ = t.step((0.0, 0.0, 0.0))
        assert p[P['pos_in_pos']] == 0                  # 8 mm: not entered
    for _ in range(300):
        p, _ = t.step((0.0, 0.0, 0.0), wheel_v=0.01)
        if p[P['pos_in_pos']] == 1:
            break
    assert p[P['pos_in_pos']] == 1 and _err(p) < 0.005
    seen_mid = False
    for _ in range(400):                                # back away slowly
        p, _ = t.step((0.0, 0.0, 0.0), wheel_v=-0.01)
        e = _err(p)
        if e <= 0.0098:
            assert p[P['pos_in_pos']] == 1
            seen_mid |= e > 0.0055
        if p[P['pos_in_pos']] == 0:
            break
    assert seen_mid and p[P['pos_in_pos']] == 0 and _err(p) > 0.010


# ── (a) in-position window, (d) standstill hold (Liam 04/11, CODE session: "đồng ý làm a+d") ──
AD = dict(ON, pos_e_win=0.005, still_on=1)


def _traction():
    p = fwsim.MecanumParams()
    p.ground_model = 'traction'
    p.mu_hd = np.full(4, 0.5)
    return p


def test_a_d_off_by_name(fw):
    log = fwsim.simulate(fw, const(vx=0.3), 2.0, imu={}, fw_params=dict(ON, pos_e_win=0.0, still_on=0))
    pp = fw.pos_params()
    assert pp[4] == 0 and pp[6] == 0                      # e_win 0, still_on 0
    assert np.all(col(log, 'pos_in_pos') == 0) and np.all(col(log, 'still') == 0)


@pytest.mark.parametrize('v', [0.2, 0.5, 0.8])
def test_ladrc_completes_o1_with_a_d(fw, v):
    """Traction plant (S01 motor dead zone and friction): LADRC + position loop hunts at standstill
    without a, d (O1_FEAS run02, run05_diag); with a + d the O1 stop condition is met in time and
    the hold keeps the PWM at 0."""
    sch, path = o1.Schedule(v), o1.Path8()
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, _traction(), seed=1, mode='ADRC', imu={},
                         fw_params=AD)
    res, _ = o1.evaluate(log, path, sch)
    assert res['completed']
    held = col(log, 'still') == 1
    assert held.any() and np.all(log['pwm'][held] == 0)
    assert not held[log['t'] <= sch.Tcmd].any()           # never while the command runs
    assert np.all(col(log, 'pos_corr_vx')[held] == 0) and np.all(col(log, 'pos_corr_vy')[held] == 0)


def test_hold_waits_for_slow_and_exits_on_command(fw):
    """Brake from 0.8 m/s: no hold while faster than v_still; hold after t_still once slow; a new
    command releases it at once and the car moves again."""
    sp = lambda t: (0.8, 0.0, 0.0) if t < 1.5 else ((0.0, 0.0, 0.0) if t < 4.0 else (0.3, 0.0, 0.0))  # noqa: E731
    log = fwsim.simulate(fw, sp, 5.5, _traction(), seed=1, mode='ADRC', imu={}, fw_params=AD)
    t, held = log['t'], col(log, 'still') == 1
    speed = np.hypot(log['v_fw'][:, 0], log['v_fw'][:, 1])
    t_hold = t[np.argmax(held)]
    assert 1.5 + 0.5 <= t_hold < 4.0
    k = np.argmax(held)
    assert np.all(speed[k - 50:k] < 0.05)                # slow for the whole t_still before the hold
    assert not held[(t > 4.0 + 0.011)].any()              # released by the new command
    assert log['v_true'][t > 5.0, 0].min() > 0.2


def test_window_only_with_zero_command(fw):
    """(a) alone: no window while the command runs; after it stops, inside the window the loop
    gives no correction."""
    sp = lambda t: (0.3, 0.0, 0.0) if t < 2.0 else (0.0, 0.0, 0.0)  # noqa: E731
    log = fwsim.simulate(fw, sp, 5.0, _traction(), seed=1, mode='PID', imu={},
                         fw_params=dict(ON, pos_e_win=0.005, still_on=0))
    t, inpos = log['t'], col(log, 'pos_in_pos') == 1
    assert not inpos[t < 2.0].any()
    assert inpos.any()
    assert np.all(col(log, 'pos_corr_vx')[inpos] == 0) and np.all(col(log, 'pos_corr_vy')[inpos] == 0)



def test_hold_stops_position_correction(fw):
    """(d) alone (no window): while held the position loop gives no correction, even though the
    estimated error after the hunting stops is several mm."""
    sch = o1.Schedule(0.5)
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, _traction(), seed=1, mode='ADRC', imu={},
                         fw_params=dict(ON, pos_e_win=0.0, still_on=1))
    held = col(log, 'still') == 1
    assert held.any()
    e = np.hypot(col(log, 'pos_ex_b'), col(log, 'pos_ey_b'))
    assert e[held].max() > 0.003
    assert np.all(col(log, 'pos_corr_vx')[held] == 0) and np.all(col(log, 'pos_corr_vy')[held] == 0)
