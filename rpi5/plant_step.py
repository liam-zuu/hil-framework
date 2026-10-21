"""
Mecanum AGV plant model — Python port of plant_step.m (M6 version)

Implements Lagrangian coupled dynamics with optional wheel slip model.
Called every 1 ms from HILNode (Process 1).

State vector (10 elements):
  [0]  x       [m]       body position X (world frame)
  [1]  y       [m]       body position Y (world frame)
  [2]  theta   [rad]     heading, wrapped to [-π, π]
  [3]  vx      [m/s]     body velocity X (world frame)
  [4]  vy      [m/s]     body velocity Y (world frame)
  [5]  wz      [rad/s]   yaw rate
  [6]  omega1  [rad/s]   wheel 1 speed (FL)
  [7]  omega2  [rad/s]   wheel 2 speed (FR)
  [8]  omega3  [rad/s]   wheel 3 speed (RL)
  [9]  omega4  [rad/s]   wheel 4 speed (RR)

Integrator: semi-implicit Euler with midpoint rotation
  - velocity (omega) updated first, then pose
  - midpoint angle avoids drift in constant-curvature turns
"""
import numpy as np
from params import MecanumParams, PARAMS

_DEFAULT_RNG = np.random.default_rng(0)


def _wrap_angle(theta: float) -> float:
    """Wrap angle to [-π, π] — equivalent to MATLAB mod(θ+π, 2π)-π."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def plant_step(
    x: np.ndarray,
    tau: np.ndarray,
    params: MecanumParams = PARAMS,
    dt: float = 0.001,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Advance plant state by one timestep (semi-implicit Euler).

    Args:
        x:      State vector (10,), will be copied (not mutated).
        tau:    Wheel command (4,). params.motor_model 'dc': PWM duty in [-1, 1]
                (H7 sends duty in the SPI torque field). 'torque': torque [N·m],
                clipped to ±tau_max.
        params: MecanumParams instance. Defaults to module singleton.
        dt:     Timestep [s]. Must match HIL loop period.
        rng:    numpy Generator for wheel slip randomness.
                Pass a fixed-seed Generator for reproducibility.

    Returns:
        x_new:  Updated state vector (10,), new array.
    """
    if rng is None:
        rng = _DEFAULT_RNG

    x     = np.asarray(x, dtype=float)
    omega = x[6:10].copy()   # current wheel speeds

    if getattr(params, 'motor_model', 'torque') == 'dc':
        omega_new = _dc_wheel_step(omega, np.asarray(tau, dtype=float), params, rng, dt)
    else:
        tau = np.clip(np.asarray(tau, dtype=float), -params.tau_max, params.tau_max)
        tau = tau * getattr(params, 'motor_polarity', 1.0)   # motor wiring polarity

        # ── 1. Effective torque (with optional slip) ──────────────────────
        tau_eff = _apply_slip(tau, omega, params, rng) if params.slip.enabled else tau

        # ── 2. Wheel dynamics: M_eff × dω/dt = τ_eff − b_w × ω ──────────
        d_omega = params.M_eff_inv @ (tau_eff - params.b_w * omega)

        # Semi-implicit: update omega first
        omega_new = np.clip(omega + d_omega * dt, -params.omega_max, params.omega_max)

    if getattr(params, 'wheels_lifted', False):
        # Wheels in the air: the body does not move.
        x_new = x.copy()
        x_new[3:6] = 0.0
        x_new[6:10] = omega_new
        return x_new

    # ── 3. Body velocities (no-slip kinematic constraint) ─────────────────
    v_body = params.H_fwd @ omega_new   # [vx_body, vy_body, wz]

    # ── 4. Pose integration (midpoint rotation) ───────────────────────────
    theta     = x[2]
    theta_mid = theta + 0.5 * v_body[2] * dt   # midpoint angle
    cos_m, sin_m = np.cos(theta_mid), np.sin(theta_mid)

    # Body → world frame velocity rotation
    vx_w = cos_m * v_body[0] - sin_m * v_body[1]
    vy_w = sin_m * v_body[0] + cos_m * v_body[1]

    # ── 5. Assemble new state ─────────────────────────────────────────────
    x_new = np.empty(10)
    x_new[0] = x[0] + vx_w * dt
    x_new[1] = x[1] + vy_w * dt
    x_new[2] = _wrap_angle(theta + v_body[2] * dt)
    x_new[3] = vx_w
    x_new[4] = vy_w
    x_new[5] = v_body[2]
    x_new[6:10] = omega_new

    return x_new


OMEGA_EPS = 1e-3   # [rad/s] below this a wheel counts as standing still


def motor_torque(u, omega, params):
    """DC motor torque at the wheel shaft for duty u (already polarity-corrected)."""
    u = np.clip(u, -1.0, 1.0)
    u_eff = np.sign(u) * np.maximum(np.abs(u) - params.motor_u0, 0.0)
    tau = params.motor_K / params.motor_R * (params.V_batt * u_eff - params.motor_K * omega)
    return tau * params.motor_enable      # EN = 0: open circuit, no torque, no braking


def _dc_wheel_step(omega, u_cmd, params, rng, dt):
    """One step of the 'dc' wheel model: DC motor, viscous + Coulomb + static friction."""
    u = np.clip(u_cmd, -1.0, 1.0) * getattr(params, 'motor_polarity', 1.0)
    tau_m = motor_torque(u, omega, params)
    if params.slip.enabled and not getattr(params, 'wheels_lifted', False):
        tau_m = _apply_slip(tau_m, omega, params, rng)

    moving = np.abs(omega) > OMEGA_EPS
    stuck = ~moving & (np.abs(tau_m) <= params.motor_Ts)       # static friction holds
    tau_f = params.motor_b * omega + params.motor_Tc * np.sign(omega)
    # Breaking away from rest: Coulomb opposes the drive torque
    tau_f = np.where(~moving & ~stuck, params.motor_Tc * np.sign(tau_m), tau_f)
    tau_net = np.where(stuck, 0.0, tau_m - tau_f)

    omega_new = omega + (params.M_eff_inv @ tau_net) * dt
    omega_new = np.where(stuck, 0.0, omega_new)
    # Friction alone must not reverse a wheel within one step: stop it instead.
    crossed = moving & (np.sign(omega_new) != np.sign(omega)) & (np.abs(tau_m) <= params.motor_Ts)
    return np.where(crossed, 0.0, omega_new)


def _apply_slip(
    tau: np.ndarray,
    omega: np.ndarray,
    params: MecanumParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stochastic wheel slip model (M6 physics).

    Two trigger conditions (per wheel):
      1. Torque-induced: |τᵢ| > μ_static × (Mg/4) × r
      2. Spontaneous:    random surface imperfection (prob_spontaneous)

    During slip, effective torque = μ_kinetic × F_N × r × noise_factor.
    noise_factor ∈ [0.5, 1.5] models surface irregularity.
    """
    F_N             = params.M * params.g / 4.0              # N per wheel
    tau_friction_max = params.slip.mu_static  * F_N * params.r  # 0.381 N·m
    tau_kinetic_base = params.slip.mu_kinetic * F_N * params.r  # 0.238 N·m

    tau_eff = tau.copy()
    # Draw the same number of random numbers every step (2 per wheel), whatever the
    # torque: the random stream, and so a seeded run, does not depend on the controller.
    u_spont = rng.random(4)
    z_noise = rng.standard_normal(4)
    for i in range(4):
        slip = (
            abs(tau[i]) > tau_friction_max
            or u_spont[i] < params.slip.prob_spontaneous
        )
        if slip:
            noise = 1.0 + params.slip.noise_sigma * z_noise[i]
            noise = float(np.clip(noise, 0.5, 1.5))
            tau_eff[i] = np.sign(tau[i]) * tau_kinetic_base * noise

    return tau_eff
