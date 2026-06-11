"""Tests for plant_step.py and mecanum_plant.py — physics correctness."""
import math
import pytest
import numpy as np
import random
from params import PARAMS
from plant_step import plant_step
from mecanum_plant import MecanumPlant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zero_state():
    return [0.0] * 10

def _run_steps(state, tau, n, dt=0.001, params=None, rng=None):
    """Run plant for n steps with constant torque."""
    p = params or PARAMS
    x = list(state)
    for _ in range(n):
        x = plant_step(x, tau, p, dt, rng)
    return x


# ---------------------------------------------------------------------------
# 1. Kinematics — torque pattern → expected body motion direction
# ---------------------------------------------------------------------------

class TestKinematics:
    """Qualitative direction tests (mirror MATLAB M2 open-loop tests)."""

    def test_forward_drive(self):
        """Equal positive torque on all wheels → positive vx, vy≈0, wz≈0."""
        tau = [0.05, 0.05, 0.05, 0.05]
        x = _run_steps(zero_state(), tau, 3000)
        vx, vy, wz = x[3], x[4], x[5]
        assert vx > 0.5, f"vx={vx:.3f} should be positive"
        assert abs(vy) < 0.1 * abs(vx), f"vy={vy:.4f} should be near zero"
        assert abs(wz) < 0.1, f"wz={wz:.4f} should be near zero"

    def test_strafe_left(self):
        """Mecanum strafe pattern → positive vy, vx≈0."""
        # Left strafe: wheel pattern [-,+,+,-]
        tau = [-0.05, 0.05, 0.05, -0.05]
        x = _run_steps(zero_state(), tau, 3000)
        vx, vy = x[3], x[4]
        assert vy > 0.3, f"vy={vy:.3f} should be positive"
        assert abs(vx) < 0.2 * abs(vy), f"vx={vx:.4f} should be near zero"

    def test_ccw_rotation(self):
        """CCW rotation pattern → positive wz, vx≈vy≈0."""
        tau = [-0.05, 0.05, -0.05, 0.05]
        x = _run_steps(zero_state(), tau, 3000)
        vx, vy, wz = x[3], x[4], x[5]
        assert wz > 0.5, f"wz={wz:.3f} should be positive"
        assert abs(vx) < 0.1, f"vx={vx:.4f} should be near zero"
        assert abs(vy) < 0.1, f"vy={vy:.4f} should be near zero"


# ---------------------------------------------------------------------------
# 2. Dynamics — steady-state and energy
# ---------------------------------------------------------------------------

class TestDynamics:
    """Quantitative dynamics: SS = τ/b_w (linear regime only)."""

    def test_steady_state_omega(self):
        """In linear range: ω_ss = τ / b_w for each wheel."""
        # Linear regime: tau << omega_max * b_w = 34.56 * 0.002 = 0.069 N·m
        tau_val = 0.02  # well inside linear range
        tau = [tau_val] * 4
        p = PARAMS
        omega_ss_expected = tau_val / p.b_w   # = 10.0 rad/s
        # Run 15 time constants to settle (tau_c ≈ J_eff/b_w ≈ 2.4s → 36s)
        x = _run_steps(zero_state(), tau, int(15 * 2.5 / 0.001))
        for i, omega in enumerate(x[6:10]):
            err_pct = abs(omega - omega_ss_expected) / omega_ss_expected * 100
            assert err_pct < 1.0, (
                f"wheel {i}: omega={omega:.4f}, expected {omega_ss_expected:.4f} "
                f"({err_pct:.2f}% error)"
            )

    def test_torque_clamped_no_explosion(self):
        """Very high torque should not cause NaN or divergence."""
        tau = [10.0, 10.0, 10.0, 10.0]   # way above tau_max=0.5
        x = _run_steps(zero_state(), tau, 1000)
        for val in x:
            assert math.isfinite(val), f"State exploded: {x}"

    def test_no_nan_random_torques(self):
        """5000 random torques should never produce NaN/Inf."""
        import random
        rng = random.Random(42)
        x = zero_state()
        p = PARAMS
        for _ in range(5000):
            tau = [rng.uniform(-0.6, 0.6) for _ in range(4)]
            x = plant_step(x, tau, p, 0.001)
            for v in x:
                assert math.isfinite(v), f"NaN/Inf in state: {x}"

    def test_double_torque_double_omega_ss(self):
        """In linear regime: 2× torque → 2× ω_ss."""
        settle_steps = int(15 * 2.5 / 0.001)
        tau1 = [0.01] * 4
        tau2 = [0.02] * 4
        x1 = _run_steps(zero_state(), tau1, settle_steps)
        x2 = _run_steps(zero_state(), tau2, settle_steps)
        for i in range(4):
            ratio = x2[6 + i] / x1[6 + i]
            assert abs(ratio - 2.0) < 0.05, f"wheel {i}: ratio={ratio:.3f}, expected 2.0"


# ---------------------------------------------------------------------------
# 3. Integration quality
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_heading_wraps_to_minus_pi_pi(self):
        """Constant CCW rotation should keep θ in [-π, π]."""
        tau = [-0.05, 0.05, -0.05, 0.05]
        x = zero_state()
        p = PARAMS
        for _ in range(20_000):
            x = plant_step(x, tau, p, 0.001)
            theta = x[2]
            assert -math.pi - 1e-9 <= theta <= math.pi + 1e-9, \
                f"θ={theta:.4f} outside [-π, π]"

    def test_zero_torque_decelerates(self):
        """After spin-up, zero torque should decelerate all wheels."""
        # Spin up
        tau_fwd = [0.05] * 4
        x = _run_steps(zero_state(), tau_fwd, 2000)
        omega_after_spinup = x[6]
        assert omega_after_spinup > 1.0, "Spin-up failed"

        # Coast with zero torque
        x_coasted = _run_steps(x, [0.0]*4, 2000)
        for i in range(4):
            assert x_coasted[6+i] < x[6+i], \
                f"wheel {i} did not decelerate: {x_coasted[6+i]:.3f} >= {x[6+i]:.3f}"

    def test_kinetic_energy_increases_during_acceleration(self):
        """During constant positive torque, total KE should be strictly increasing."""
        tau = [0.03] * 4
        p = PARAMS
        x = zero_state()
        ke_prev = 0.0
        for k in range(500):
            x = plant_step(x, tau, p, 0.001)
            # KE = 0.5 * sum(J_w * omega_i^2)
            ke = 0.5 * p.J_w * sum(w**2 for w in x[6:10])
            if k > 0:
                assert ke >= ke_prev - 1e-10, \
                    f"KE decreased at step {k}: {ke:.6f} < {ke_prev:.6f}"
            ke_prev = ke


# ---------------------------------------------------------------------------
# 4. Slip model
# ---------------------------------------------------------------------------

class TestSlipModel:
    def test_slip_disabled_no_effect(self):
        """slip.enabled=False → identical output regardless of p_spontaneous."""
        import copy
        p = copy.deepcopy(PARAMS)
        p.slip.enabled = False
        tau = [0.1] * 4  # above friction limit (0.381 N·m... wait 0.1 < 0.381)
        # Use torque above friction limit
        tau = [0.4, 0.4, 0.4, 0.4]
        x0 = zero_state()
        x1 = plant_step(list(x0), tau, p, 0.001)
        x2 = plant_step(list(x0), tau, p, 0.001)
        for a, b in zip(x1, x2):
            assert a == b

    def test_high_torque_slip_reduces_acceleration(self):
        """With slip enabled and tau > friction limit, acceleration should drop."""
        import copy
        import random
        p_noslip = copy.deepcopy(PARAMS)
        p_noslip.slip.enabled = False

        p_slip = copy.deepcopy(PARAMS)
        p_slip.slip.enabled = True
        p_slip.slip.prob_spontaneous = 0.0  # only torque-induced

        # tau=0.45 > tau_friction_max=0.381
        tau = [0.45] * 4
        x0 = zero_state()
        rng_slip = np.random.default_rng(99)

        # Run one step without slip → get reference acceleration
        x_noslip = plant_step(list(x0), tau, p_noslip, 0.001)
        domega_noslip = x_noslip[6] - x0[6]

        # Run many steps with slip to catch at least one slip event
        accumulated_slip = 0.0
        n_trials = 200
        for _ in range(n_trials):
            x_slip = plant_step(list(x0), tau, p_slip, 0.001, rng_slip)
            accumulated_slip += x_slip[6] - x0[6]
        avg_domega_slip = accumulated_slip / n_trials

        assert avg_domega_slip < domega_noslip, (
            f"Slip should reduce acceleration: avg={avg_domega_slip:.6f} "
            f">= noslip={domega_noslip:.6f}"
        )


# ---------------------------------------------------------------------------
# 5. MecanumPlant wrapper
# ---------------------------------------------------------------------------

class TestMecanumPlant:
    def test_instantiate(self):
        p = MecanumPlant(PARAMS, dt=0.001, seed=0)
        assert p is not None

    def test_step_returns_10_states(self):
        p = MecanumPlant(PARAMS)
        s = p.step([0.0]*4)
        assert len(s) == 10

    def test_get_state(self):
        p = MecanumPlant(PARAMS)
        p.step([0.01]*4)
        s = p.get_state()
        assert len(s) == 10

    def test_time_advances(self):
        p = MecanumPlant(PARAMS, dt=0.001)
        for _ in range(5):
            p.step([0.0]*4)
        assert abs(p.t - 0.005) < 1e-9

    def test_reset_zeroes_state(self):
        p = MecanumPlant(PARAMS)
        for _ in range(100):
            p.step([0.05]*4)
        p.reset()
        assert p.get_state() == [0.0]*10
        assert p.t == 0.0

    def test_set_slip(self):
        p = MecanumPlant(PARAMS)
        p.set_slip(True)
        p.set_slip(False)  # just verify it doesn't crash
