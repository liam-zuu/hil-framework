"""
IMU values for the BNO085 emulator on the H7 (G2, IMU option A).

The RPi5 computes what the IMU on the vehicle would measure and sends it in
the SPI frame (protocol v3: gz, ax, ay, az, yaw, imu_seq); the H7 only packs
it as SH-2 reports. G2 (Liam 27/10, Q3): values without noise or bias, taken
straight from the plant. Noise + output steps measured on the vehicle (K1,
10/11/2026): ImuNoise below, shared by the HIL Pi and fwsim (L71).

Frame: body frame x forward, y left, z up. The real BNO085 mounting axes on
the vehicle are not used yet (to be derived from the 10/06 logs at G3).
Accelerometer convention as the BNO085 accelerometer report: specific force
including gravity, so a vehicle at rest on flat ground reads az = +g.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# MecanumPlant / plant_step state layout (see plant_step.py)
IDX_THETA = 2
IDX_VX_W  = 3      # world frame
IDX_VY_W  = 4      # world frame
IDX_WZ    = 5


def imu_from_plant(state: Sequence[float], prev_state: Optional[Sequence[float]],
                   dt: float, g: float) -> tuple:
    """
    Noise-free IMU sample from two consecutive MecanumPlant states.

    Args:
        state:      current plant state [x, y, θ, vx_w, vy_w, wz, ω1..ω4].
        prev_state: state one step earlier (None on the first step → a = 0).
        dt:         plant step [s].
        g:          gravity [m/s²].

    Returns:
        (gz, ax, ay, az, yaw) [rad/s, m/s², m/s², m/s², rad], the order of
        protocol.encode_state(imu=...).
    """
    theta = state[IDX_THETA]
    if prev_state is None:
        a_wx = a_wy = 0.0
    else:
        a_wx = (state[IDX_VX_W] - prev_state[IDX_VX_W]) / dt
        a_wy = (state[IDX_VY_W] - prev_state[IDX_VY_W]) / dt
    # world → body: rotate by -θ
    c, s = math.cos(theta), math.sin(theta)
    ax = c * a_wx + s * a_wy
    ay = -s * a_wx + c * a_wy
    return (state[IDX_WZ], ax, ay, g, theta)


@dataclass(frozen=True)
class GyroInjection:
    """
    Controlled IMU–encoder inconsistency injection (QĐ 25 method C; spec
    controller–IMU v2.2 §6.1, V2 (a)): during [t0, t0 + t_inj) the gyro model
    reports (1 − s)·wz_body. The plant state is untouched, so the residual
    r = wz_enc − gz the firmware sees is known exactly (s·wz_body) and does not
    depend on any slip model. A test signal, not a physical slip model.
    """
    t0: float      # [s]
    t_inj: float   # [s]
    s: float       # fraction of the body yaw rate the gyro misses

    def active(self, t: float) -> bool:
        return self.t0 <= t < self.t0 + self.t_inj

    def apply(self, gz: float, t: float) -> float:
        return (1.0 - self.s) * gz if self.active(t) else gz


def parse_imu_static(text: str) -> tuple:
    """'gz,ax,ay,az,yaw' → tuple of 5 floats (run_hil.py --imu-static)."""
    vals = [float(v) for v in text.split(',')]
    if len(vals) != 5:
        raise ValueError(f'--imu-static needs 5 comma-separated values '
                         f'(gz,ax,ay,az,yaw), got {len(vals)}')
    return tuple(vals)


# ── Noise + output-step model (Liam 10/11/2026, CODE local_69fa8ade, L71) ──────
#
# Measured on the BNO085 of the vehicle, 300 s static, motors off
# (05_DATA/results/IMU_NOISE/2026-11-10_run01_20hz, run02_100hz; analysis
# 05_DATA/results/IMU_NOISE/2026-11-10_analysis/ket_qua.txt):
#   - the gyro comes out in steps of 2^-9 rad/s (SH-2 Q9); 90-93 % of the samples
#     are exactly 0 at 20 Hz, 99 % at 100 Hz. sigma below = maximum-likelihood
#     sigma of "Gaussian, then rounded to the step" for gz [measured];
#   - gyro bias 3.3e-5 rad/s (20 Hz) / 2.5e-6 (100 Hz), 40x below one step: not
#     modelled (bias 0);
#   - game RV yaw at rest: std 1.35e-4 rad (20 Hz) / 4.85e-5 rad (100 Hz), slow
#     (lag-1 autocorrelation 0.975 / 0.969): first-order Gauss-Markov with
#     tau = -dt / ln(lag1) [measured], no drift (< 0.002 deg/min measured);
#   - the accelerometer comes out on a grid of 10/256 m/s^2 (0.039); its noise
#     under that grid is not identifiable: rounded only, no noise added. The
#     real grid is offset from 0 and differs per axis (limitation: here centred
#     on 0). The controller uses the accelerometer only in the still test of
#     the bias estimator, which is off (sigma = 0 in firmware, L71).
# One run per rate, one place, vehicle at rest: vibration with motors running
# is not in these numbers.

GYRO_STEP_RADPS = 2.0 ** -9          # SH-2 Q9 output step [measured: 0.001953]
ACCEL_STEP_MPS2 = 10.0 / 256.0       # observed output grid [measured: 0.039062]


@dataclass(frozen=True)
class ImuNoiseSpec:
    """Noise of one report rate (see the block comment above)."""
    rate_hz: float
    sigma_gz: float          # [rad/s] Gaussian sigma before rounding to GYRO_STEP
    sigma_yaw: float         # [rad] stationary sigma of the yaw noise
    tau_yaw: float           # [s] Gauss-Markov time constant of the yaw noise


NOISE_20HZ = ImuNoiseSpec(20.0, sigma_gz=5.89e-4, sigma_yaw=1.354e-4,
                          tau_yaw=-0.0499 / math.log(0.975))      # ~1.97 s
NOISE_100HZ = ImuNoiseSpec(100.0, sigma_gz=3.99e-4, sigma_yaw=4.85e-5,
                           tau_yaw=-0.0100 / math.log(0.969))     # ~0.318 s


def noise_spec_for_rate(rate_hz: float) -> ImuNoiseSpec:
    """Measured spec for 20 or 100 Hz; other rates were not measured."""
    for s in (NOISE_20HZ, NOISE_100HZ):
        if abs(rate_hz - s.rate_hz) < 1e-6:
            return s
    raise ValueError(f'IMU noise measured only at 20 and 100 Hz, not {rate_hz} Hz')


def _round_to(v: float, q: float) -> float:
    return q * math.floor(v / q + 0.5)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ImuNoise:
    """
    Adds the measured noise and output steps to an IMU sample
    (gz, ax, ay, az, yaw), the tuple of imu_from_plant(). One instance per run,
    with its own seed. apply(sample, dt): dt = time since the previous call
    [s] (report period in fwsim, frame period on the HIL Pi); the yaw noise is
    a discrete first-order Gauss-Markov process with phi = exp(-dt/tau), started
    from its stationary distribution, so its statistics do not depend on dt.
    """

    def __init__(self, spec: ImuNoiseSpec, seed: int):
        self.spec = spec
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._n_yaw = None

    def apply(self, sample: Sequence[float], dt: float) -> tuple:
        gz, ax, ay, az, yaw = (float(v) for v in sample)
        s = self.spec
        e_g, e_y = self._rng.standard_normal(2)
        gz = _round_to(gz + s.sigma_gz * e_g, GYRO_STEP_RADPS)
        if self._n_yaw is None:
            self._n_yaw = s.sigma_yaw * e_y
        else:
            phi = math.exp(-max(dt, 0.0) / s.tau_yaw)
            self._n_yaw = phi * self._n_yaw + s.sigma_yaw * math.sqrt(1.0 - phi * phi) * e_y
        ax, ay, az = (_round_to(v, ACCEL_STEP_MPS2) for v in (ax, ay, az))
        return (gz, ax, ay, az, _wrap_pi(yaw + self._n_yaw))
