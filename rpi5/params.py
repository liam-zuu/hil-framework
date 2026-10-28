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
        # Motor wiring polarity per wheel [FL, FR, RL, RR]: sign of the real wheel
        # rotation for a positive command (duty as H7 reads it: rpwm - lpwm).
        # S01 run01 (05_DATA/results/S01/2026-10-20_run01, 20/10/2026): all four
        # wheels turn backward for +PWM with ESP32 motor_dir = +1, so -1 here and
        # ESP32 motor_dir = -1 (motor.c). Not yet checked on the HIL bench (wiring
        # ESP32 -> H7): if the closed loop runs away on HIL, flip this sign.
        self.motor_polarity   = np.array([-1.0, -1.0, -1.0, -1.0])

        # ── Motor model ──────────────────────────────────────────────────
        # 'dc'    : command = PWM duty u in [-1, 1] (what H7 sends in the SPI
        #           torque field). Brushed DC motor + gearbox at the wheel shaft:
        #             tau_m = (K/R) * (V * u_eff - K * omega),
        #             u_eff = sign(u) * max(|u| - u0, 0)
        #           friction  b*omega + Tc*sign(omega), static breakaway Ts.
        # 'torque': legacy M6 model, command = torque [N.m], viscous b_w only.
        # Values: spec S01 run01 identification, direction-averaged
        # (05_DATA/results/S01/2026-10-20_run01/identification/plant_params_from_S01.csv,
        # script matlab/identification/s01_derive_plant_params.py). R measured at the
        # driver terminals with a multimeter, about +-15-25 % uncertain; J and b scale with 1/R.
        self.motor_model  = 'dc'
        self.motor_K      = np.array([0.30394, 0.32210, 0.30917, 0.28811])   # [V.s/rad] = [N.m/A]
        self.motor_R      = np.array([3.31, 3.38, 3.73, 3.95])               # [ohm]
        self.motor_u0     = np.array([0.05223, 0.03170, 0.03574, 0.05413])   # duty offset (brush drop etc.)
        self.motor_Tc     = np.array([0.02394, 0.02738, 0.02347, 0.02433])   # Coulomb friction [N.m]
        self.motor_Ts     = np.array([0.11878, 0.14760, 0.10489, 0.08129])   # static breakaway torque [N.m]
        self.motor_b      = np.array([0.00062, 0.00081, 0.00074, 0.00175])   # viscous [N.m.s/rad]
        self.J_wheels     = np.array([0.00209, 0.00236, 0.00204, 0.00249])   # wheel+gear+rotor [kg.m^2]
        self.V_batt       = 12.4      # [V] (S01 run01 mean 12.435 V)
        self.motor_enable = 1.0       # 1 = driver enabled; 0 = EN low, motor open (free coast)
        self.wheels_lifted = False    # True: wheels off the ground, no body coupling (S01 bench)
        self.g         = 9.81     # gravity [m/s²]

        # ── Ground contact (QĐ 25 method A, docs/2026-10-28_Thiet_ke_cach_A_mo_hinh_truot.md) ──
        # 'rigid'   : body follows the wheels (no-slip kinematic constraint), the plant used up to 28/10
        #             (with the rotating-frame term since 28/10, bug HIL-PLANT-ROTFRAME).
        # 'traction': body has its own dynamics (M, Iz); traction torque per wheel
        #             tau_c = C * sat((omega - omega_contact) / omega_s), C = mu_hd * (M g / 4) * r,
        #             omega_contact = row of H_inv times the body velocity.
        # Only for motor_model 'dc' with the wheels on the ground (wheels_lifted is unchanged).
        # 'traction' ignores slip.enabled (the stochastic _apply_slip model is not used with it).
        # Default 'traction' since 28/10 (design note Q3 (a), Liam approved): T-A1..T-A7 passed,
        # 05_DATA/results/SLIP_A/2026-10-28_run03 + rpi5/tests/test_slip_a.py. On the Pi the 1 kHz
        # budget for it is not measured yet (plant_timing.py): run_hil.py --ground rigid falls back.
        self.ground_model = 'traction'
        # Effective wheel-floor mu as the force-gauge test measures it: mu_hd = F_max / (m g),
        # pushing straight or sideways (spec controller-IMU v2.2 §1b.4). Not the material mu:
        # with ideal 45° rollers mu_hd ≈ mu_material / sqrt(2).
        self.mu_hd        = np.full(4, 0.5)   # [ASSUMPTION — pending MU_FLOOR measurement]
        self.slip_omega_s = 0.1               # [rad/s] traction smoothing band [ASSUMPTION — numerical smoothing]
        self.slip_nsub    = None              # sub-steps per dt; None = from the stability bound [calculated]

        # ── Encoder parameters ───────────────────────────────────────────
        self.enc_ppr          = 1320    # counts/rev, 4x (GB37-520: 11 PPR x 30:1 = 330/ch; x4). Not used by HIL path: H7 ENC_PPR=330 generates pulses
        self.enc_noise_sigma  = 0.02    # Gaussian noise σ on counts
        self.enc_filter_tau   = 0.005   # IIR filter τ [s]

        # ── IMU parameters ───────────────────────────────────────────────
        # Same values as params_mecanum.m: BNO085 measured 11/06/2026 (session log only,
        # raw run pending: tools/imu_capture.py). Not used by the HIL plant: there is no
        # IMU emulation, the ESP32 reads its real BNO085 on the bench.
        self.imu_accel_noise       = 0.01254       # [m/s²]  (was 0.05 before 21/10)
        self.imu_gyro_noise        = 0.00031       # [rad/s] (was 0.002 before 21/10)
        self.imu_rate_hz           = 20            # BNO085 report rate in imu.cpp (50 ms)
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
        J_diag = (np.diag(self.J_wheels) if getattr(self, 'motor_model', 'torque') == 'dc'
                  else self.J_w * np.eye(4))
        if getattr(self, 'wheels_lifted', False):
            self.M_eff = J_diag                      # wheels in the air: no body coupling
        else:
            self.M_eff = self.H_fwd.T @ M_body @ self.H_fwd + J_diag
        self.M_eff_inv = np.linalg.inv(self.M_eff)

        # Ground contact 'traction': rows of H_inv as plain floats (inner loop), and the
        # largest eigenvalue of J^-1 + H_inv diag(M, M, Iz)^-1 H_inv^T [1/(kg m^2)], which
        # sets the fastest slip mode and so the explicit sub-step bound (plant_step.traction_nsub).
        self.H_inv_rows = tuple(tuple(float(v) for v in row) for row in self.H_inv)
        A_slip = (np.linalg.inv(J_diag)
                  + self.H_inv @ np.diag([1.0 / self.M, 1.0 / self.M, 1.0 / self.Iz]) @ self.H_inv.T)
        self.traction_lam_max = float(np.linalg.eigvalsh(A_slip).max())

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
