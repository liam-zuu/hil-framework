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

Ground contact (params.ground_model, motor model 'dc', wheels on the ground):
  'rigid'    body velocity = H_fwd @ omega (no-slip constraint); with the body's
             rotating-frame term since 28/10 (rotating_frame_torque, bug HIL-PLANT-ROTFRAME)
  'traction' QĐ 25 method A (docs/2026-10-28_Thiet_ke_cach_A_mo_hinh_truot.md): the
             body velocity [3:6] is its own state, wheels and body are coupled only by
             the traction torque, which saturates at mu_hd * N * r per wheel, so the
             wheels can slip. Explicit sub-steps (traction_nsub), no randomness.
"""
import math

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

    if (getattr(params, 'motor_model', 'torque') == 'dc'
            and getattr(params, 'ground_model', 'rigid') == 'traction'
            and not getattr(params, 'wheels_lifted', False)):
        return _traction_step(x, np.asarray(tau, dtype=float), params, dt)

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


def rotating_frame_torque(omega, params):
    """Generalized wheel torque of the body's rotating-frame term, ground_model 'rigid'.

    The body velocity v = H_fwd @ omega is in the body frame, so Newton reads
    M (dv/dt + wz x v) = F, i.e. M (dvx/dt - wz vy) = Fx, M (dvy/dt + wz vx) = Fy
    (the yaw row has no such term). M_eff covers M dv/dt only; the wz x v part
    (centripetal / Coriolis) maps to the wheels as H_fwd^T [-M wz vy, M wz vx, 0].
    Left out before 28/10 (M6 port): turning while moving needed no lateral force.
    It does no work (v . (wz x v) = 0)."""
    v = params.H_fwd @ omega
    return params.H_fwd.T @ np.array([-params.M * v[2] * v[1], params.M * v[2] * v[0], 0.0])


def _dc_wheel_step(omega, u_cmd, params, rng, dt):
    """One step of the 'dc' wheel model: DC motor, viscous + Coulomb + static friction."""
    u = np.clip(u_cmd, -1.0, 1.0) * getattr(params, 'motor_polarity', 1.0)
    tau_m = motor_torque(u, omega, params)
    if not getattr(params, 'wheels_lifted', False):
        if params.slip.enabled:
            tau_m = _apply_slip(tau_m, omega, params, rng)
        # Drive torque left after the body's rotating-frame term (bug HIL-PLANT-ROTFRAME):
        # folded into tau_m so the static-friction test sees the net drive.
        tau_m = tau_m - rotating_frame_torque(omega, params)

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


TRACTION_SAFETY = 2.0   # sub-steps per dt: this factor over the explicit-Euler stability bound


def traction_nsub(params: MecanumParams, dt: float = 0.001) -> int:
    """Sub-steps per dt for ground_model 'traction'.

    params.slip_nsub if set; otherwise from the stiffest slip mode: the traction law
    is linear with slope k = C / omega_s inside the band, so explicit Euler is stable for
    h < 2 / (k * lam_max), lam_max = params.traction_lam_max. TRACTION_SAFETY x that.
    """
    if params.slip_nsub:
        return int(params.slip_nsub)
    c_max = float(np.max(params.mu_hd)) * params.M * params.g / 4.0 * params.r
    k = c_max / params.slip_omega_s
    return max(1, math.ceil(TRACTION_SAFETY * dt * k * params.traction_lam_max / 2.0))


def _traction_step(x, u_cmd, params, dt, f_ext=None):
    """One dt of ground_model 'traction' (method A), plain floats in the inner loop.

    Per wheel i, at every sub-step h = dt / traction_nsub:
      omega_c,i = H_inv[i] . v_body                     contact speed seen by wheel i
      tau_c,i   = C_i * sat((omega_i - omega_c,i) / omega_s),  C_i = mu_hd,i * (M g / 4) * r
      J_i domega_i/dt = tau_m,i - tau_f,i - tau_c,i      motor and friction as _dc_wheel_step,
                                                        static friction holds when |tau_m - tau_c| <= Ts
    Body: [Fx, Fy, Mz] = H_inv^T tau_c (body frame), integrated in the world frame
    (no Coriolis term needed), then the pose. Power check: the wheels lose tau_c.omega,
    the body gains tau_c.omega_c, the rest tau_c.(omega - omega_c) >= 0 is dissipated.

    f_ext: tests only (virtual force gauge): f(x, y, vx_w, vy_w) -> (Fx_w, Fy_w) [N].
    """
    p = params
    pol = p.motor_polarity.tolist()
    u = [min(1.0, max(-1.0, float(u_cmd[i]))) * pol[i] for i in range(4)]
    n_load = p.M * p.g / 4.0
    cap = [float(m) * n_load * p.r for m in np.broadcast_to(p.mu_hd, (4,))]
    inv_ws = 1.0 / p.slip_omega_s
    rows = p.H_inv_rows
    kr = (p.motor_K / p.motor_R).tolist()
    kk, u0 = p.motor_K.tolist(), p.motor_u0.tolist()
    tc_f, ts, bv, jw = p.motor_Tc.tolist(), p.motor_Ts.tolist(), p.motor_b.tolist(), p.J_wheels.tolist()
    vb, en = p.V_batt, p.motor_enable
    m, iz = p.M, p.Iz

    xw, yw, th, vxw, vyw, wz = (float(v) for v in x[0:6])
    w = [float(v) for v in x[6:10]]
    n = traction_nsub(p, dt)
    h = dt / n
    for _ in range(n):
        c, s = math.cos(th), math.sin(th)
        vbx = c * vxw + s * vyw            # world -> body
        vby = -s * vxw + c * vyw
        fx = fy = mz = 0.0
        for i in range(4):
            wi = w[i]
            hx, hy, hz = rows[i]
            z = (wi - (hx * vbx + hy * vby + hz * wz)) * inv_ws
            tc = cap[i] * (1.0 if z > 1.0 else -1.0 if z < -1.0 else z)
            ui = u[i]
            a = abs(ui) - u0[i]
            ue = (a if ui > 0.0 else -a) if a > 0.0 else 0.0
            drive = kr[i] * (vb * ue - kk[i] * wi) * en - tc
            if abs(wi) > OMEGA_EPS:
                wn = wi + h * (drive - bv[i] * wi - (tc_f[i] if wi > 0.0 else -tc_f[i])) / jw[i]
                if (wn > 0.0) != (wi > 0.0) and abs(drive) <= ts[i]:
                    wn = 0.0               # friction alone must not reverse the wheel
            elif abs(drive) <= ts[i]:
                wn = 0.0                   # static friction holds
            else:
                wn = wi + h * (drive - (tc_f[i] if drive > 0.0 else -tc_f[i])) / jw[i]
            w[i] = wn
            fx += hx * tc
            fy += hy * tc
            mz += hz * tc
        axw = (c * fx - s * fy) / m        # body -> world
        ayw = (s * fx + c * fy) / m
        if f_ext is not None:
            gx, gy = f_ext(xw, yw, vxw, vyw)
            axw += gx / m
            ayw += gy / m
        vxw += h * axw
        vyw += h * ayw
        wz += h * mz / iz
        xw += h * vxw
        yw += h * vyw
        th += h * wz
    return np.array([xw, yw, _wrap_angle(th), vxw, vyw, wz, w[0], w[1], w[2], w[3]])


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
