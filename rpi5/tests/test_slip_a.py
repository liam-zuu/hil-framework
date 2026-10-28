"""Ground contact 'traction' (QĐ 25 method A), plant-level checks of
docs/2026-10-28_Thiet_ke_cach_A_mo_hinh_truot.md section 6 (thresholds fixed in
the note before running, Liam approved 28/10): T-A1, T-A3, T-A4, T-A5, T-A6.
T-A2 and T-A7 (fwsim, group A metrics vs qrms) live with fwsim. These tests check
the implementation against the model equations, not the model against the vehicle."""
import math

import numpy as np
import pytest

from params import MecanumParams
from plant_step import plant_step, traction_nsub, _traction_step, rotating_frame_torque
from imu_model import imu_from_plant

DT = 0.001


def ground(mu=0.5, **kw):
    p = MecanumParams()
    p.ground_model = 'traction'
    p.mu_hd = np.full(4, float(mu))
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def push_cmd(p, direction, duty=1.0):
    """Duty command (before motor polarity) that turns every wheel the way that pushes
    the body along direction = (vx, vy, wz)."""
    sgn = np.sign(p.H_inv @ np.asarray(direction, dtype=float))
    return duty * sgn * p.motor_polarity


def body_vel(x):
    c, s = math.cos(x[2]), math.sin(x[2])
    return np.array([c * x[3] + s * x[4], -s * x[3] + c * x[4], x[5]])


def kinetic_energy(p, x):
    return 0.5 * (p.M * (x[3] ** 2 + x[4] ** 2) + p.Iz * x[5] ** 2 + float(np.sum(p.J_wheels * x[6:10] ** 2)))


# T-A1: wheels lifted, 'traction' changes nothing (bit for bit)
def test_lifted_bit_identical_to_rigid():
    rng_u = np.random.default_rng(1)
    u_seq = rng_u.uniform(-1, 1, size=(3000, 4))
    u_seq[1000:1500] = 0.0                         # coast and stop too
    out = {}
    for model in ('rigid', 'traction'):
        p = MecanumParams()
        p.ground_model = model
        p.update_physical(wheels_lifted=True)
        x, rng = np.zeros(10), np.random.default_rng(0)
        for u in u_seq:
            x = plant_step(x, u, p, DT, rng)
        out[model] = x
    assert np.array_equal(out['rigid'], out['traction'])


# T-A3: virtual force gauge (spring + damper holding the body), full duty after slip:
# static spring force / (M g) = mu_hd within 1 %, pushing straight and sideways
@pytest.mark.parametrize('mu', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('direction', [(1, 0, 0), (0, 1, 0)])
def test_force_gauge_reads_mu_hd(mu, direction):
    p = ground(mu)
    k = 2000.0                                     # [N/m]
    c = 2 * 0.7 * math.sqrt(k * p.M)               # damping ratio 0.7

    def gauge(xw, yw, vxw, vyw):
        return -k * xw - c * vxw, -k * yw - c * vyw

    u = push_cmd(p, direction)
    x = np.zeros(10)
    for _ in range(3000):
        x = _traction_step(x, u, p, DT, f_ext=gauge)
    f_spring = k * math.hypot(x[0], x[1])
    assert f_spring / (p.M * p.g) == pytest.approx(mu, rel=0.01)
    # Yaw is not checked: the four motors differ (S01), so the first ms before every wheel
    # saturates leaves a small yaw rate (about 0.003 rad/s at mu 0.8), and with all wheels
    # sliding nothing damps it (the gauge holds x, y only). That is the model, not an error.


# T-A4: motors open (EN = 0), arbitrary start: kinetic energy never increases
def test_energy_never_increases_when_coasting():
    p = ground(0.4, motor_enable=0.0)
    x = np.zeros(10)
    x[3:6] = [0.5, -0.3, 1.0]
    x[6:10] = [5.0, -3.0, 2.0, -4.0]
    rng = np.random.default_rng(0)
    e = kinetic_energy(p, x)
    for _ in range(2000):
        x = plant_step(x, np.zeros(4), p, DT, rng)
        e_new = kinetic_energy(p, x)
        assert e_new <= e * (1 + 1e-12) + 1e-15
        e = e_new
    assert e < 1e-6                                # everything has stopped


# T-A5: drive beyond the grip: wheels slip, body accelerates at mu_hd g (± 5 %),
# and the IMU model's accelerometer shows the same
def test_slip_onset_body_accel_is_mu_g():
    mu = 0.3
    p = ground(mu)
    u = push_cmd(p, (1, 0, 0))
    x, rng = np.zeros(10), np.random.default_rng(0)
    for _ in range(300):
        x = plant_step(x, u, p, DT, rng)
    v0 = body_vel(x)[0]
    x_prev = x
    for _ in range(50):
        x_prev, x = x, plant_step(x, u, p, DT, rng)
    a = (body_vel(x)[0] - v0) / 0.050
    assert a == pytest.approx(mu * p.g, rel=0.05)
    slip = x[6:10] - p.H_inv @ body_vel(x)
    assert np.all(slip > p.slip_omega_s)            # every wheel past the smoothing band
    ax = imu_from_plant(x, x_prev, DT, p.g)[1]
    assert ax == pytest.approx(mu * p.g, rel=0.05)


def test_gentle_drive_does_not_slip():
    p = ground(0.5)
    u = push_cmd(p, (1, 0, 0), duty=0.15)
    x, rng = np.zeros(10), np.random.default_rng(0)
    for _ in range(2000):
        x = plant_step(x, u, p, DT, rng)
    slip = x[6:10] - p.H_inv @ body_vel(x)
    assert np.all(np.abs(slip) < p.slip_omega_s)    # inside the band: wheels grip
    assert body_vel(x)[0] > 0.05


# T-A6: deterministic, and the plant RNG is not touched
def test_deterministic_and_no_rng_use():
    p = ground(0.3)
    u_seq = np.random.default_rng(2).uniform(-1, 1, size=(1000, 4))
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(0)
        state0 = rng.bit_generator.state
        x = np.zeros(10)
        for u in u_seq:
            x = plant_step(x, u, p, DT, rng)
        assert rng.bit_generator.state == state0
        runs.append(x)
    assert np.array_equal(runs[0], runs[1])


# Sanity (T-A2 proper is on fwsim): almost no slip (mu_hd = 5) gives the rigid plant's
# body velocity within 2 % of the forward speed on every axis. Not equal: the rigid plant
# lets the wheels "fight" (pattern [-1, -1, +1, +1], design note §3.6), traction does not.
def test_large_mu_close_to_rigid():
    u = push_cmd(MecanumParams(), (1, 0, 0), duty=0.3)
    v = {}
    for model in ('rigid', 'traction'):
        p = ground(5.0) if model == 'traction' else MecanumParams()
        x, rng = np.zeros(10), np.random.default_rng(0)
        for _ in range(1000):
            x = plant_step(x, u, p, DT, rng)
        v[model] = body_vel(x)
    assert np.all(np.abs(v['traction'] - v['rigid']) < 0.02 * abs(v['rigid'][0]))


# Bug HIL-PLANT-ROTFRAME (28/10): turning while moving, the rigid plant must need the same
# lateral force as the traction plant (which integrates the body in the world frame).
# Without the rotating-frame term the rigid body vy was off by far more than 2 %.
def test_turning_rigid_matches_traction():
    p0 = MecanumParams()
    sgn = np.sign(p0.H_inv @ np.array([1.0, 0.0, 0.0])) + 0.5 * np.sign(p0.H_inv @ np.array([0.0, 0.0, 1.0]))
    u = 0.4 * sgn * p0.motor_polarity               # forward + yaw: an arc
    v = {}
    for model in ('rigid', 'traction'):
        p = ground(5.0) if model == 'traction' else MecanumParams()
        x, rng = np.zeros(10), np.random.default_rng(0)
        for _ in range(1500):
            x = plant_step(x, u, p, DT, rng)
        v[model] = body_vel(x)
    assert abs(v['rigid'][2]) > 0.5                   # it does turn
    assert np.all(np.abs(v['traction'] - v['rigid']) < 0.02 * abs(v['rigid'][0]))


def test_rotating_frame_term_does_no_work():
    p = MecanumParams()
    w = np.array([8.0, 3.0, 5.0, -2.0])
    assert float(w @ rotating_frame_torque(w, p)) == pytest.approx(0.0, abs=1e-12)


# Sanity (T-A7 proper is on fwsim): doubling the sub-steps changes the result by < 0.5 %
def test_substep_convergence():
    p1 = ground(0.3)
    p2 = ground(0.3, slip_nsub=2 * traction_nsub(ground(0.3)))
    rng_u = np.random.default_rng(3)
    u_seq = np.repeat(rng_u.uniform(-1, 1, size=(20, 4)), 100, axis=0)
    xs = []
    for p in (p1, p2):
        x, rng = np.zeros(10), np.random.default_rng(0)
        for u in u_seq:
            x = plant_step(x, u, p, DT, rng)
        xs.append(x)
    scale = np.maximum(np.abs(xs[1][3:10]), 0.05)
    assert np.all(np.abs(xs[0][3:10] - xs[1][3:10]) / scale < 0.005)


def test_nsub_from_stability_bound():
    p = ground(0.5)
    k = 0.5 * p.M * p.g / 4 * p.r / p.slip_omega_s
    h = DT / traction_nsub(p)
    assert h * k * p.traction_lam_max <= 2.0 / 2.0 + 1e-12      # TRACTION_SAFETY = 2
    assert traction_nsub(ground(0.5, slip_nsub=3)) == 3
