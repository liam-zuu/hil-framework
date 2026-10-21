"""DC motor wheel model (motor_model='dc'), checked against the model equations.
Parameter values come from params.py (spec S01 run01); these tests check the
implementation, not the identification."""
import numpy as np
import pytest

from params import MecanumParams
from plant_step import plant_step, motor_torque


def lifted(**kw):
    p = MecanumParams()
    p.motor_polarity = np.ones(4)
    for k, v in kw.items():
        setattr(p, k, v)
    p.update_physical(wheels_lifted=True)
    return p


def run(p, u, steps, x=None):
    x = np.zeros(10) if x is None else x
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x = plant_step(x, np.asarray(u, dtype=float), p, 0.001, rng)
    return x


def test_steady_state_matches_closed_form():
    p = lifted()
    u = 0.6
    w = run(p, [u] * 4, 3000)[6:10]
    k_r = p.motor_K / p.motor_R
    expected = (k_r * p.V_batt * (u - p.motor_u0) - p.motor_Tc) / (p.motor_b + k_r * p.motor_K)
    assert np.allclose(w, expected, rtol=1e-3)


def test_step_time_constant_is_electrical():
    p = lifted()
    w_ss = run(p, [0.8] * 4, 3000)[6:10]
    x = run(p, [0.8] * 4, 80)                      # 80 ms after the step
    tau_model = p.J_wheels / (p.motor_b + p.motor_K ** 2 / p.motor_R)
    frac = x[6:10] / w_ss
    # first-order response with constant offset terms: 63 % near t = tau
    assert np.all(np.abs(tau_model - 0.08) < 0.03)
    assert np.all((frac > 0.5) & (frac < 0.8))


def test_below_breakaway_wheel_stays_still():
    p = lifted()
    u_small = 0.05 + p.motor_u0                    # torque below Ts for every wheel
    assert np.all(np.abs(motor_torque(u_small, np.zeros(4), p)) < p.motor_Ts)
    w = run(p, u_small, 2000)[6:10]
    assert np.allclose(w, 0.0)


def test_free_coast_is_slower_than_brake():
    p = lifted()
    x0 = run(p, [0.7] * 4, 2000)
    p.motor_enable = 0.0                            # EN low: free coast
    coast = run(p, [0.0] * 4, 300, x0.copy())[6:10]
    p.motor_enable = 1.0                            # PWM 0, EN high: brake
    brake = run(p, [0.0] * 4, 300, x0.copy())[6:10]
    assert np.all(np.abs(coast) > np.abs(brake))
    assert np.all(np.abs(brake) < 0.05 * np.abs(x0[6:10]))      # brake: almost stopped after 0.3 s


def test_coast_ends_at_rest_without_sign_reversal():
    p = lifted(motor_enable=0.0)
    x0 = run(lifted(), [0.7] * 4, 2000)
    w = run(p, [0.0] * 4, 6000, x0)[6:10]
    assert np.allclose(w, 0.0)


def test_ground_mode_moves_the_body():
    p = MecanumParams()
    p.motor_polarity = np.ones(4)
    p.update_physical()
    x = run(p, [0.5] * 4, 2000)
    # wheels are not identical (per-wheel S01 parameters), so allow some drift
    assert x[0] > 0.1 and abs(x[1]) < 0.2 * x[0]
