"""
HIL Mecanum AGV Parameters — Python port of params_mecanum.m (M5.2 / M6)

Single source of truth for all physical and simulation constants.
Import PARAMS singleton or instantiate MecanumParams() directly.

Usage:
    from params import PARAMS
    dt = 0.001
    tau_max = PARAMS.tau_max
"""
import numpy as np
from dataclasses import dataclass, field


# ── Sub-configs (nested structs) ─────────────────────────────────────────

@dataclass
class SpiParams:
    tau_range:    float       = 1.0          # full-scale torque ±[N·m]
    float_bits:   int         = 16
    # Per-state full-scale range — must match MATLAB params.spi.state_ranges
    # [x, y, θ, vx, vy, wz, ω1, ω2, ω3, ω4]
    state_ranges: np.ndarray  = field(
        default_factory=lambda: np.array(
            [5.0, 5.0, np.pi, 3.0, 3.0, 10.0, 40.0, 40.0, 40.0, 40.0]
        )
    )


@dataclass
class SlipParams:
    enabled:           bool  = False
    mu_static:         float = 0.8     # dry concrete
    mu_kinetic:        float = 0.5     # ≈ 63% of static
    prob_spontaneous:  float = 0.002   # per wheel per step
    noise_sigma:       float = 0.15    # kinetic friction variation σ
    detect_threshold:  float = 0.15    # slip ratio detection threshold
    imu_wz_threshold:  float = 0.5     # IMU yaw rate mismatch [rad/s]


@dataclass
class DisturbanceParams:
    enabled:    bool  = False
    type:       str   = 'none'   # 'step'|'ramp'|'random'|'combined'
    magnitude:  float = 0.05     # [N·m]
    start_time: float = 3.0      # [s]
    ramp_rate:  float = 0.02     # [N·m/s]
    random_sigma: float = 0.03   # [N·m]


# ── Main parameter class ──────────────────────────────────────────────────

class MecanumParams:
    """
    All parameters for the Mecanum AGV HIL simulation.

    Derived matrices (H_fwd, H_inv, M_eff, M_eff_inv) are computed
    once from the physical constants in __init__.  If you update r,
    lx, ly, M, Iz, or J_w, call _compute_kinematics() again.
    """

    def __init__(self):
        # ── Physical parameters ──────────────────────────────────────────
        self.r         = 0.0485    # wheel radius [m]
        self.lx        = 0.12     # longitudinal half-distance [m]
        self.ly        = 0.12     # lateral half-distance [m]
        self.M         = 4.0      # total mass [kg]
        self.Iz        = 0.0384   # yaw inertia [kg·m²]  (computed from geometry)
        self.J_w       = 0.00247  # wheel inertia [kg·m²]
        self.b_w       = 0.002    # viscous friction [N·m·s/rad]
        self.tau_max   = 0.5      # max torque per wheel [N·m]
        self.omega_max = 34.56    # max wheel speed [rad/s]
        self.g         = 9.81     # gravity [m/s²]

        # ── Encoder parameters ───────────────────────────────────────────
        self.enc_ppr          = 1024    # pulses per revolution
        self.enc_noise_sigma  = 0.02    # Gaussian noise σ on counts
        self.enc_filter_tau   = 0.005   # IIR filter τ [s]

        # ── IMU parameters ───────────────────────────────────────────────
        self.imu_accel_noise       = 0.05          # [m/s²]
        self.imu_gyro_noise        = 0.002         # [rad/s]
        self.imu_accel_bias_drift  = 0.001
        self.imu_gyro_bias_drift   = 0.0001
        self.imu_filter_tau        = 0.003         # [s]
        self.imu_outlier_accel     = 50.0          # [m/s²]
        self.imu_outlier_gyro      = 20.0          # [rad/s]
        self.imu_adc_bits          = 16
        self.imu_accel_range       = 4 * 9.81      # ±4 g
        self.imu_gyro_range        = 2000 * np.pi / 180   # ±2000 deg/s

        # ── PWM parameters ───────────────────────────────────────────────
        self.pwm_deadband      = 0.02    # fraction of full-scale
        self.pwm_jitter_sigma  = 0.001
        self.pwm_res_bits      = 10

        # ── Sync timing ──────────────────────────────────────────────────
        self.sync_jitter_us   = 5
        self.sync_timeout_us  = 50

        # ── Controller gains (M5.2 optimised) ────────────────────────────
        # Inner PID
        self.pid = dict(Kp=0.04, Ki=0.5, Kd=0.0004)
        # Inner ADRC
        self.adrc = dict(kp=20.0, omega_o=100.0, beta1=200.0, beta2=10000.0,
                         b0=None)          # b0 filled after J_w known
        # Outer position loop
        self.pos_ctrl = dict(
            Kp_pos=6.0, Ki_pos=3.0,
            Kp_theta=8.0, Ki_theta=6.0,
            vx_max=1.5, vy_max=1.5, wz_max=5.0,
        )

        # ── Sub-configs ──────────────────────────────────────────────────
        self.spi         = SpiParams()
        self.slip        = SlipParams()
        self.disturbance = DisturbanceParams()

        # ── Derived matrices (computed once) ─────────────────────────────
        self._compute_kinematics()

    # ─────────────────────────────────────────────────────────────────────

    def _compute_kinematics(self) -> None:
        """
        Compute H_inv, H_fwd, M_eff, M_eff_inv from physical constants.

        Must be called again if r, lx, ly, M, Iz, or J_w is changed.
        Formulation: Taheri 2015 / Muir-Neuman 1987, X-config mecanum.
        """
        r, lx, ly = self.r, self.lx, self.ly
        L = lx + ly   # 0.24 m

        # Inverse kinematics: ω = H_inv @ [vx, vy, wz]  — shape (4, 3)
        # Wheel numbering: 1=FL, 2=FR, 3=RL, 4=RR
        self.H_inv = (1.0 / r) * np.array([
            [1.0, -1.0, -L],
            [1.0,  1.0,  L],
            [1.0,  1.0, -L],
            [1.0, -1.0,  L],
        ])

        # Forward kinematics: [vx, vy, wz] = H_fwd @ ω  — shape (3, 4)
        self.H_fwd = np.linalg.pinv(self.H_inv)

        # Effective inertia (Lagrangian, coupled):
        #   M_eff = H_fwd.T @ diag(M, M, Iz) @ H_fwd + J_w * I₄
        # Off-diagonal coupling ≈ 29% of diagonal (body mass via kinematics)
        M_body = np.diag([self.M, self.M, self.Iz])
        self.M_eff     = self.H_fwd.T @ M_body @ self.H_fwd + self.J_w * np.eye(4)
        self.M_eff_inv = np.linalg.inv(self.M_eff)

        # ADRC b0 = 1 / J_eff (effective inertia in forward mode)
        self.adrc['b0'] = 1.0 / (self.M_eff[0, 0])

    def update_physical(self, **kwargs) -> None:
        """
        Update one or more physical parameters and recompute derived matrices.

        Example:
            params.update_physical(M=4.5, b_w=0.003)
        """
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown parameter: {key}")
            setattr(self, key, val)
        self._compute_kinematics()

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def tau_friction_max(self) -> float:
        """Max static friction torque per wheel [N·m]."""
        return self.slip.mu_static * self.M * self.g / 4.0 * self.r

    @property
    def linear_range_tau(self) -> float:
        """Max torque still in linear (non-saturating) plant regime."""
        return self.omega_max * self.b_w   # ≈ 0.069 N·m  (14% of tau_max)


# ── Module-level singleton ────────────────────────────────────────────────
# Import this in all modules:
#   from params import PARAMS
PARAMS = MecanumParams()
