"""IMU in the loop on fwsim: imu_fusion.c + heading.c of the ESP32 firmware against
the plant's IMU model (spec controller-IMU v2.2 section 10, AGV_MECANUM docs/).
All thresholds are the spec's [DE XUAT] defaults; sigma_gz is not measured yet (K1),
so r_th = k_r * q_wz. The IMU model is noise-free (G2), so these are logic checks,
not performance numbers.
Run: python -m pytest fwsim/tests  (needs fwsim/build/fwsim.dll: python fwsim/build_fwsim.py)"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
from imu_model import GyroInjection  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')

C = {n: i for i, n in enumerate(fwsim.IMU_COLS)}
P_REPORT = 0.05      # 20 Hz (IMU_REPORT_PERIOD_US)
TICK = 0.01
T_AGE = 0.125        # 2.5 report periods
T_SLIP = 0.2
T_SETTLE = 0.5
K_SW = 1.5           # section 10 switching criterion [DE XUAT]
DPWM_MIN = 20        # [DE XUAT, not locked: QD 21] floor of the switching criterion, 2 % of PWM_MAX


@pytest.fixture(scope='module')
def fw():
    return fwsim.FirmwareSim()


class _RigidParams(fwsim.MecanumParams):
    """Plant for these logic checks: ground_model 'rigid' (Liam 28/10, CODE session: pin the
    section 10 logic tests to rigid). Their [DE XUAT] thresholds were set on the rigid plant;
    on 'traction' (default since 28/10) two cases differ by plant dynamics, not by slip
    (r_true < 0.001 rad/s): switch-back PWM jump 36 > 33 in the gyro-loss case, slip flag off
    at 3.5 s > 3.26 s after the injection (design note docs/2026-10-28_Thiet_ke_cach_A_mo_hinh_truot.md)."""
    def __init__(self):
        super().__init__()
        self.ground_model = 'rigid'


@pytest.fixture(autouse=True)
def rigid_plant(monkeypatch):
    monkeypatch.setattr(fwsim, 'MecanumParams', _RigidParams)


def const(vx=0.0, vy=0.0, wz=0.0):
    return lambda t: (vx, vy, wz)


def col(log, name):
    return log['imu'][:, C[name]]


def edges(log, name):
    """Times where a column changes, and the new values."""
    v = col(log, name)
    i = np.nonzero(np.diff(v))[0] + 1
    return log['t'][i], v[i]


def max_dpwm_near(log, t_ev, half=0.2):
    d = np.abs(np.diff(log['pwm'], axis=0)).max(axis=1)
    w = np.abs(log['t'][1:] - t_ev) <= half
    return d[w].max()


def test_no_imu_keeps_encoder_and_no_heading(fw):
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 2.0)
    assert np.all(col(log, 'wz_src') == 0)
    assert np.all(col(log, 'head_on') == 0)
    np.testing.assert_array_equal(col(log, 'wz_cmd'), log['sp'][:, 2])


def test_valid_throughout_circle(fw):
    """IMU valid all run (nominal plant, no slip model): wz from IMU on >= 99 % of ticks after
    the first n_back samples, no slip flag; r_true = 0 because the plant has no slip."""
    sp = fwsim.make_setpoint(type('A', (), dict(traj='circle', speed=0.3, radius=0.5, period=8.0))())
    log = fwsim.simulate(fw, sp, 10.0, imu={})
    late = log['t'] > 3 * P_REPORT + TICK
    assert np.mean(col(log, 'wz_src')[late] == 1) >= 0.99
    assert np.all(np.abs(log['r_true']) < 0.5 * col(log, 'r_th')[-1])
    assert col(log, 'slip_on_count')[-1] == 0
    assert col(log, 'head_on')[late].all()


def test_20hz_sample_held_between_ticks(fw):
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 3.0, imu={})
    after_first = log['t'] >= P_REPORT
    assert col(log, 'g_valid')[after_first].all()
    assert col(log, 'r_valid')[after_first].all()
    assert col(log, 'age_g_ms')[after_first].max() < P_REPORT * 1e3 + 1e-6


def test_heading_loop_reduces_drift_on_straight_line(fw):
    """The plant's per-wheel parameters (S01) make the vehicle turn on a straight command;
    the heading loop holds the start heading. Both runs take wz from the IMU, only k_psi
    differs, so the gain comes from the heading loop alone (Codex ctrl-imu N1)."""
    a = fwsim.simulate(fw, const(vx=0.3), 6.0, imu={}, fw_params=dict(k_psi=0.0))
    b = fwsim.simulate(fw, const(vx=0.3), 6.0, imu={})
    assert col(a, 'wz_src')[a['t'] > 0.5].all() and col(b, 'wz_src')[b['t'] > 0.5].all()
    assert abs(b['pose'][-1, 2]) < 0.5 * abs(a['pose'][-1, 2])


def test_gyro_loss_switches_to_encoder_and_back(fw):
    drop = lambda t, k, v: None if (k == 'gyro' and 2.0 <= t < 3.0) else v   # noqa: E731
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 5.0, imu=dict(hook=drop))
    ref = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 5.0, imu={})
    t_sw, v_sw = edges(log, 'wz_src')
    last_good = 2.0 - P_REPORT
    lost = t_sw[(v_sw == 0) & (t_sw > 1.0)][0]
    assert last_good < lost <= last_good + T_AGE + TICK + 1e-9
    back = t_sw[(v_sw == 1) & (t_sw > 2.5)][0]
    assert back == pytest.approx(3.0 + 2 * P_REPORT)          # third good sample (n_back = 3)
    for t_ev in (lost, back):
        assert max_dpwm_near(log, t_ev) <= max(K_SW * max_dpwm_near(ref, t_ev), DPWM_MIN)
    # heading loop stays on: yaw is still there
    assert col(log, 'head_on')[log['t'] > 0.5].all()


def test_nan_gyro_sample(fw):
    nan = lambda t, k, v: math.nan if (k == 'gyro' and abs(t - 2.0) < 1e-9) else v   # noqa: E731
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 3.0, imu=dict(hook=nan))
    t = log['t']
    bad = (t >= 2.0 - 1e-9) & (t < 2.0 + P_REPORT - 1e-9)
    assert not col(log, 'g_valid')[bad].any()
    assert col(log, 'g_valid')[(t >= 2.0 + P_REPORT - 1e-9)].all()
    assert np.isfinite(log['imu']).all()
    t_sw, v_sw = edges(log, 'wz_src')
    assert t_sw[(v_sw == 0) & (t_sw > 1.0)][0] == pytest.approx(2.0)
    assert t_sw[(v_sw == 1) & (t_sw > 1.0)][0] == pytest.approx(2.0 + 3 * P_REPORT)


def test_gyro_loss_at_100hz_uses_t_age_of_that_rate(fw):
    """--imu-rate 100: T_age = 2.5 x 10 ms, as the firmware built for 100 Hz (Codex ctrl-imu #9)."""
    p, t_age = 0.01, 0.025
    drop = lambda t, k, v: None if (k == 'gyro' and 2.0 <= t < 3.0) else v   # noqa: E731
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 4.0, imu=dict(rate_hz=100.0, hook=drop))
    t_sw, v_sw = edges(log, 'wz_src')
    lost = t_sw[(v_sw == 0) & (t_sw > 1.0)][0]
    assert 2.0 - p < lost <= 2.0 - p + t_age + TICK + 1e-9


def test_accel_nan_breaks_standstill_window(fw):
    """A non-finite accel sample makes the standstill window invalid until a good one
    (Codex ctrl-imu #7): one NaN every 2 s window gives no bias update."""
    hook = lambda t, k, v: ((math.nan,) * 3 if k == 'accel' and abs(t % 1.0 - 0.5) < 1e-9   # noqa: E731
                            else v + 0.01 if k == 'gyro' else v)
    prm = dict(sigma_gz=0.01, sigma_a=0.05)
    log = fwsim.simulate(fw, const(), 9.0, imu=dict(hook=hook), fw_params=prm, motor_on=False)
    assert col(log, 'bias_updates')[-1] == 0


def test_gyro_saturation_threshold(fw):
    """0.9 x 2000 deg/s = 31.4 rad/s (Codex ctrl-imu N2): a sample at 32 rad/s is not valid."""
    sat = lambda t, k, v: 32.0 if (k == 'gyro' and abs(t - 2.0) < 1e-9) else v   # noqa: E731
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 3.0, imu=dict(hook=sat))
    t = log['t']
    assert not col(log, 'g_valid')[(t >= 2.0 - 1e-9) & (t < 2.0 + P_REPORT - 1e-9)].any()


def test_imu_init_fail_motors_stay_off(fw):
    """imu_init() failed (spec section 7): EN stays low and no PWM reaches the plant, even
    with a setpoint and SWA turned on during the run."""
    log = fwsim.simulate(fw, const(vx=0.3), 2.0, imu=dict(init_fail=True))
    assert not log['en'].any()
    assert not np.abs(log['duty']).any()
    fw.motor_enable(True)                 # SWA on: main.c keeps swa_on = false
    _, _, _, _, en = fw.tick(2_010_000, [0, 0, 0, 0], (0.3, 0.0, 0.0))
    assert en == 0
    fw.reset()


@pytest.mark.parametrize('mode', ['PID', 'ADRC'])
def test_controller_reset_when_motor_reenabled(fw, mode):
    """Codex ctrl-imu #1, Liam 28/10: wheels held (counts stay 0) with a setpoint winds up the
    PID integral / ESO; SWA off, setpoint 0, SWA on again: the first PWM is 0, not the old state."""
    fw.reset()
    fw.set_mode(mode)
    t = 0
    for _ in range(100):
        t += 10_000
        _, _, pwm, _, _ = fw.tick(t, [0, 0, 0, 0], (0.3, 0.0, 0.0))
    assert np.abs(pwm).max() > 0
    fw.motor_enable(False)
    for _ in range(5):
        t += 10_000
        fw.tick(t, [0, 0, 0, 0], (0.0, 0.0, 0.0))
    fw.motor_enable(True)
    t += 10_000
    _, _, pwm, _, en = fw.tick(t, [0, 0, 0, 0], (0.0, 0.0, 0.0))
    assert en == 1
    assert not np.abs(pwm).any()
    fw.reset()


def test_slip_not_evaluable_column(fw):
    """Codex ctrl-imu #8 = (c), Liam 28/10: slip_eval = 0 while the wheels stand or the gyro is
    invalid; a flag that is on goes off T_slip after the check stops (spec section 13.1)."""
    still = fwsim.simulate(fw, const(), 2.0, imu={})
    assert not col(still, 'slip_eval').any()
    inj = GyroInjection(t0=1.0, t_inj=3.0, s=0.3)
    drop = lambda t, k, v: None if (k == 'gyro' and t >= 2.0) else v   # noqa: E731
    log = fwsim.simulate(fw, const(wz=1.0), 3.0, imu=dict(injection=inj, hook=drop))
    t = log['t']
    assert col(log, 'slip_eval')[(t > 0.5) & (t < 2.0)].all()
    assert col(log, 'slip_flag')[(t > 1.5) & (t < 2.0)].all()
    t_off = edges(log, 'slip_eval')[0][-1]                 # gyro too old: check stops
    assert not col(log, 'slip_eval')[t >= t_off].any()
    t_flag = edges(log, 'slip_flag')[0][-1]
    assert t_flag == pytest.approx(t_off + T_SLIP)
    assert col(log, 'slip_on_count')[-1] == 1


def test_yaw_loss_turns_heading_off_gyro_stays(fw):
    drop = lambda t, k, v: None if (k == 'rv' and 2.0 <= t < 3.0) else v   # noqa: E731
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 5.0, imu=dict(hook=drop))
    t = log['t']
    off = (t > 2.0 - P_REPORT + T_AGE + TICK) & (t < 3.0)
    assert not col(log, 'head_on')[off].any()
    assert col(log, 'wz_src')[t > 0.5].all()                   # gyro unaffected
    assert col(log, 'head_on')[t >= 3.0 + TICK].all()          # back on with the first rv sample
    assert col(log, 'psi_reset_count')[-1] == 0                # small error: table row 5


def test_chip_reset_mid_run(fw):
    log = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 5.0, imu=dict(resets=[2.0]))
    ref = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 5.0, imu={})
    t = log['t']
    assert col(log, 'psi_reset_count')[-1] == 1
    settle = (t >= 2.0) & (t < 2.0 + T_SETTLE - 1e-9)
    assert not col(log, 'head_on')[settle].any()
    assert not col(log, 'wz_src')[settle].any()
    assert col(log, 'head_on')[t >= 2.0 + T_SETTLE + TICK].all()
    # new psi0: no turn to the old heading (error after re-enable stays small)
    assert np.abs(col(log, 'e_psi')[t > 2.6]).max() < 0.1
    for t_ev in edges(log, 'head_on')[0][edges(log, 'head_on')[0] > 1.0]:
        assert max_dpwm_near(log, t_ev) <= max(K_SW * max_dpwm_near(ref, t_ev), DPWM_MIN)


def test_mode_change_resets_heading(fw):
    """Section 5 priority 1: PID <-> LADRC switch turns the heading loop off, new psi0."""
    fw.reset()
    fw.set_run(True, False)
    fw.imu_reset(-1_000_000)
    for k in range(1, 301):
        t_us = k * 10_000
        if t_us % 50_000 == 0:
            fw.imu_report('gyro', t_us, 0.0)
            fw.imu_report('rv', t_us, 10.0)
            fw.imu_report('accel', t_us, (0.0, 0.0, 9.81))
        if k == 200:
            fw.set_mode('ADRC')
        fw.tick(t_us, [0, 0, 0, 0], (0.0, 0.0, 0.0))
        m = fw.imu_last()
        if k == 199:
            assert m[C['head_on']] == 1
        if k == 200:
            assert m[C['head_on']] == 0 and m[C['psi_reset_count']] == 1
    assert fw.imu_last()[C['head_on']] == 1


def test_injection_above_threshold(fw):
    """Method C on the gyro, residual >= 2 r_th, rotate in place: flag on within
    T_slip + 1 report period + 1 tick after t0, off within the same after t0 + T_inj,
    wz source unchanged."""
    inj = GyroInjection(t0=2.0, t_inj=1.0, s=0.3)
    log = fwsim.simulate(fw, const(wz=1.0), 5.0, imu=dict(injection=inj))
    t = log['t']
    r_th = col(log, 'r_th')[-1]
    assert np.abs(col(log, 'r')[(t > 2.3) & (t < 2.9)]).min() >= 2 * r_th
    t_on, v_on = edges(log, 'slip_flag')
    assert len(t_on) == 2 and v_on[0] == 1 and v_on[1] == 0
    assert 2.0 <= t_on[0] <= 2.0 + T_SLIP + P_REPORT + TICK + 1e-9
    assert 3.0 <= t_on[1] <= 3.0 + T_SLIP + P_REPORT + TICK + 1e-9
    assert col(log, 'wz_src')[t > 0.5].all()


def test_injection_below_threshold(fw):
    """Residual about 0.5 r_th: the flag stays off (threshold check, not sensitivity to real slip)."""
    inj = GyroInjection(t0=2.0, t_inj=1.0, s=0.03)
    log = fwsim.simulate(fw, const(wz=1.0), 5.0, imu=dict(injection=inj))
    t = log['t']
    r = np.abs(col(log, 'r')[(t > 2.3) & (t < 2.9)])
    assert 0.3 < r.mean() / col(log, 'r_th')[-1] < 0.7
    assert col(log, 'slip_on_count')[-1] == 0


def test_bias_off_until_noise_measured(fw):
    """sigma_gz = sigma_a = 0 (not measured): the standstill bias estimate never runs."""
    log = fwsim.simulate(fw, const(), 6.0, imu={})
    assert col(log, 'bias_updates')[-1] == 0


def test_bias_estimated_at_standstill(fw):
    """With a sigma set, a constant gyro offset at standstill (setpoint 0, motor disabled)
    is learnt in steps of 0.1 sigma per T_still window, capped at b_max. Not while moving,
    not with the wheels lifted."""
    off = 0.01
    hook = lambda t, k, v: v + off if k == 'gyro' else v   # noqa: E731
    prm = dict(sigma_gz=0.01, sigma_a=0.05)
    log = fwsim.simulate(fw, const(), 9.0, imu=dict(hook=hook), fw_params=prm, motor_on=False)
    n = col(log, 'bias_updates')[-1]
    assert n == 4                                    # windows of 2 s in 9 s
    assert col(log, 'bias_gz')[-1] == pytest.approx(n * 0.1 * 0.01, rel=1e-4)
    capped = fwsim.simulate(fw, const(), 9.0, imu=dict(hook=hook), motor_on=False,
                            fw_params=dict(prm, b_max=0.0015))
    assert col(capped, 'bias_gz')[-1] == pytest.approx(0.0015, rel=1e-4)
    moving = fwsim.simulate(fw, const(vx=0.1), 5.0, imu=dict(hook=hook), fw_params=prm)
    assert col(moving, 'bias_updates')[-1] == 0
    lifted = fwsim.simulate(fw, const(), 5.0, imu=dict(hook=hook), fw_params=prm, motor_on=False,
                            params=_lifted_params())
    assert col(lifted, 'bias_updates')[-1] == 0


def test_no_bias_with_motor_on_and_gyro_offset(fw):
    """Motor enabled, setpoint 0: the gyro offset drives the wz loop, PWM is not 0, so the
    standstill condition (motor disabled or PWM = 0) is not met: no bias update."""
    hook = lambda t, k, v: v + 0.01 if k == 'gyro' else v   # noqa: E731
    log = fwsim.simulate(fw, const(), 6.0, imu=dict(hook=hook), fw_params=dict(sigma_gz=0.01, sigma_a=0.05))
    assert np.abs(log['pwm']).max() > 0
    assert col(log, 'bias_updates')[-1] == 0


def _lifted_params():
    p = fwsim.MecanumParams()
    p.wheels_lifted = True
    return p
