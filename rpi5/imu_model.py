"""
IMU values for the BNO085 emulator on the H7 (G2, IMU option A).

The RPi5 computes what the IMU on the vehicle would measure and sends it in
the SPI frame (protocol v3: gz, ax, ay, az, yaw, imu_seq); the H7 only packs
it as SH-2 reports. G2 (Liam 27/10, Q3): values without noise or bias, taken
straight from the plant. Measured noise (K1, 10/11/2026): ImuNoise below
replays it, shared by the HIL Pi and fwsim (Liam "Phát lại nhiễu đo", L72);
firmware_view() gives what the firmware receives after the SPI frame, the H7
SH-2 packing and the ESP32 library, so fwsim feeds the same values as HIL.

Frame: body frame x forward, y left, z up. The real BNO085 mounting axes on
the vehicle are not used yet (to be derived from the 10/06 logs at G3).
Accelerometer convention as the BNO085 accelerometer report: specific force
including gravity, so a vehicle at rest on flat ground reads az = +g.
"""
from __future__ import annotations

import hashlib
import math
import os
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


# ── Measured noise, replayed (Liam 10/11/2026, CODE, "Phát lại nhiễu đo", L72) ─
#
# Replaces the Gaussian model of 4089f17 (L71 (1); Codex round 1 IMN-3/-5: the
# measured gyro has tails and clusters, the yaw a long correlation, that a
# rounded Gaussian / first-order Gauss-Markov model does not give).
# Source: BNO085 on the vehicle, 300 s at rest, motors off, one run per rate
# (05_DATA/results/IMU_NOISE/2026-11-10_run01_20hz, run02_100hz, raw_static.csv).
# rpi5/data/imu_noise_replay_<rate>hz.csv (tools/make_imu_noise_replay.py; the
# header holds the source path and its SHA-256) keeps, per report, the gyro z
# and yaw outputs as whole output steps:
#   k_gz  [2^-9 rad/s]  gyro z (the true rate is 0; bias 40x below one step);
#   k_yaw [2^-13 rad]   game-RV yaw minus its most common level (one Q14 step of
#                       the quaternion k at yaw 0).
# Each run starts at a place chosen by its seed and plays the reports in order,
# wrapping round at the end; report m of a run uses row (start + m) mod N.
# ImuNoise.apply() adds them to the plant truth: gz is put on the chip's output
# grid first (round to 2^-9, then + k_gz steps); yaw gets + k_yaw * 2^-13 and is
# rounded later through the Q14 quaternion (H7 on HIL, firmware_view in fwsim).
# Accelerometer: rounded to the observed 10/256 m/s^2 grid only, no noise
# (IMN-11 DEFERRED: real grid offset and per-axis steps, accel rate 32/128 Hz,
# not modelled; the controller uses it only in the still test of the bias
# estimator, which is off, sigma = 0 in firmware, L71).
# Limits: vehicle at rest, motors off (no vibration); 300 s per rate, so runs
# of different seeds can share stretches of the record; roll = pitch = 0 (the
# ~0.001 deg yaw coupling of the real roll/pitch quantisation is not kept).

GYRO_STEP_RADPS = 2.0 ** -9          # SH-2 Q9 output step [measured: 0.001953]
ACCEL_STEP_MPS2 = 10.0 / 256.0       # observed output grid [measured: 0.039062]
YAW_STEP_RAD = 2.0 ** -13            # yaw of one Q14 step of qk at yaw 0
REPLAY_RATES = (20.0, 100.0)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _round_to(v: float, q: float) -> float:
    return q * math.floor(v / q + 0.5)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class NoiseReplay:
    rate_hz: float
    k_gz: np.ndarray
    k_yaw: np.ndarray
    path: str
    sha256: str        # of the replay file with LF line ends (same on Windows and the Pi)
    header: dict       # '# key: value' lines of the file (source, source_sha256, ...)


_REPLAY_CACHE: dict = {}


def replay_path(rate_hz: float) -> str:
    return os.path.join(DATA_DIR, f'imu_noise_replay_{int(round(rate_hz))}hz.csv')


def load_replay(rate_hz: float) -> NoiseReplay:
    """The measured replay record of 20 or 100 Hz; other rates were not measured."""
    if not any(abs(rate_hz - r) < 1e-6 for r in REPLAY_RATES):
        raise ValueError(f'IMU noise measured only at 20 and 100 Hz, not {rate_hz} Hz')
    key = float(round(rate_hz))
    if key not in _REPLAY_CACHE:
        path = replay_path(rate_hz)
        raw = open(path, 'rb').read()
        header, rows = {}, []
        for line in raw.decode('utf-8').splitlines():
            if line.startswith('#'):
                k, _, v = line[1:].partition(':')
                header[k.strip()] = v.strip()
            elif line and not line.startswith('i,'):
                i, g, y = line.split(',')
                if int(i) != len(rows):
                    raise ValueError(f'{path}: row index {i} out of order')
                rows.append((int(g), int(y)))
        if int(header.get('n', -1)) != len(rows) or float(header.get('rate_hz', 0)) != key:
            raise ValueError(f'{path}: header n/rate_hz do not match the rows')
        a = np.asarray(rows, dtype=np.int64)
        _REPLAY_CACHE[key] = NoiseReplay(key, a[:, 0], a[:, 1], path,
                                         hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(), header)
    return _REPLAY_CACHE[key]


class ImuNoise:
    """
    Replays the measured noise of one report rate onto IMU samples
    (gz, ax, ay, az, yaw), the tuple of imu_from_plant(). One instance per run,
    with its own seed (start row = numpy default_rng(seed).integers(N)).

    apply(sample, t_us): t_us = time since the start of the run [us]. The row is
    chosen by m = t_us // period_us: fwsim calls once per report at
    t_us = m * period_us; the HIL Pi calls every 1 ms frame (t_us from the Pi's
    own start), so the value is held for one report period. Same (rate, seed,
    t_us) → same output.
    HIL limit (Codex round 2 IMN2-1, DEFERRED): the H7 schedules its reports
    from the Set Feature it received, not from the Pi's start, so on HIL the
    rows are the same sequence shifted by an unknown fixed number of rows
    against fwsim with the same seed, and the gyro and game-RV reports (two H7
    schedules) can take rows one apart. The statistics per rate are the same;
    sample-by-sample comparison of HIL and fwsim noise would need the H7
    report counter.
    """

    def __init__(self, rate_hz: float, seed: int):
        self.replay = load_replay(rate_hz)
        self.rate_hz = self.replay.rate_hz
        self.period_us = int(round(1e6 / self.rate_hz))
        self.seed = int(seed)
        self.start = int(np.random.default_rng(self.seed).integers(len(self.replay.k_gz)))

    def row(self, t_us: int) -> int:
        return (self.start + int(t_us) // self.period_us) % len(self.replay.k_gz)

    def apply(self, sample: Sequence[float], t_us: int) -> tuple:
        gz, ax, ay, az, yaw = (float(v) for v in sample)
        i = self.row(t_us)
        gz = _round_to(gz, GYRO_STEP_RADPS) + float(self.replay.k_gz[i]) * GYRO_STEP_RADPS
        yaw = _wrap_pi(yaw + float(self.replay.k_yaw[i]) * YAW_STEP_RAD)
        ax, ay, az = (_round_to(v, ACCEL_STEP_MPS2) for v in (ax, ay, az))
        return (gz, ax, ay, az, yaw)

    def describe(self) -> dict:
        h = self.replay.header
        return dict(model='replay', rate_hz=self.rate_hz, seed=self.seed, start_row=self.start,
                    n_rows=len(self.replay.k_gz), file=os.path.basename(self.replay.path),
                    file_sha256=self.replay.sha256, source=h.get('source'),
                    source_sha256=h.get('source_sha256'))


# ── What the firmware receives (Codex round 1 IMN-2) ─────────────────────────
#
# HIL path of one IMU sample: Pi protocol.encode_state (int16, scale/32767) →
# H7 spi1_handler (float32 raw * (MAX / 32767.0f)) → bno_emu_port
# _sample_from_imu (qk = sinf(yaw/2), qr = cosf(yaw/2), qi = qj = 0) →
# bno_emu build_report to_fixed (accel Q8, gyro Q9, quaternion Q14, float32,
# round half away from zero, clamp to the chip range) → ESP32 SH-2 library
# (raw * SCALE_Q(n), float32) → BNO08x euler (atan2 of the quaternion, float32,
# times RAD_2_DEG). firmware_view() repeats it in float32 so fwsim gives the
# firmware the same values; bnoemu_pc/tests/test_firmware_view.py checks it
# against the C code (bno_emu.c and the SH-2 library, compiled on the PC).

_F = np.float32
_IMU_FRAME_MAX = dict(gz=_F(34.906585), acc=_F(39.24), yaw=_F(3.14159265))   # hil_protocol.h
_INT16_MAXF = _F(32767.0)
_GYRO_RANGE = _F(_F(2000.0) * _F(3.14159265) / _F(180.0))        # bno_emu_default_cfg
_ACCEL_RANGE = _F(_F(8.0) * _F(9.80665))
_RV_RANGE = _F(1.9999)
_RAD_2_DEG = _F(180.0 / math.pi)                                  # (180.0f / M_PI) → float


def _frame(value: float, scale_name: str, scale: float) -> np.float32:
    from protocol import _to_int16       # the Pi encoder itself
    raw = _to_int16(value, scale)
    return _F(_F(raw) * _F(_IMU_FRAME_MAX[scale_name] / _INT16_MAXF))


def _to_fixed(x: np.float32, q: int, rng: np.float32) -> int:
    x = _F(min(max(x, -rng), rng))
    v = _F(x * _F(1 << q))
    r = int(_F(v + _F(0.5))) if v >= 0 else int(_F(v - _F(0.5)))     # C cast truncates
    return max(-32768, min(32767, r))


def firmware_view(sample: Sequence[float]) -> tuple:
    """
    (gz, ax, ay, az, yaw) as sent by the Pi → (gz [rad/s], (ax, ay, az) [m/s²],
    yaw [deg]) as the firmware stores them from the IMU reports (imu.cpp).
    """
    from protocol import IMU_RANGES
    gz, ax, ay, az, yaw = sample
    gz_h = _frame(gz, 'gz', IMU_RANGES[0])
    acc_h = [_frame(v, 'acc', IMU_RANGES[1]) for v in (ax, ay, az)]
    yaw_h = _frame(yaw, 'yaw', IMU_RANGES[4])
    with np.errstate(all='ignore'):
        h = _F(_F(0.5) * yaw_h)
        qk_raw = _to_fixed(_F(np.sin(h)), 14, _RV_RANGE)
        qr_raw = _to_fixed(_F(np.cos(h)), 14, _RV_RANGE)
        sq14, sq9, sq8 = _F(1.0 / (1 << 14)), _F(1.0 / (1 << 9)), _F(1.0 / (1 << 8))
        qk, qr = _F(_F(qk_raw) * sq14), _F(_F(qr_raw) * sq14)
        z = np.arctan2(_F(_F(2.0) * _F(qr * qk)), _F(_F(1.0) - _F(_F(2.0) * _F(qk * qk))))
        yaw_deg = _F(_F(z) * _RAD_2_DEG)
        g = _F(_F(_to_fixed(gz_h, 9, _GYRO_RANGE)) * sq9)
        acc = tuple(float(_F(_F(_to_fixed(v, 8, _ACCEL_RANGE)) * sq8)) for v in acc_h)
    return float(g), acc, float(yaw_deg)
