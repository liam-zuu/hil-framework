"""
SPI Wire Protocol — must match H7 hil_protocol.h byte-for-byte.

Frame layout v3 (40 bytes, full-duplex), all int16 big-endian:

  MOSI (RPi5 → H7)
    0–19   10 × int16  plant state [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz]
    20     uint8       fault_flags
    21     uint8       seq (frame counter, wraps)
    22–23  int16       gz_imu   (v3)
    24–29  3 × int16   trajectory reference [vx_ref, vy_ref, wz_ref] (body frame)
    30     uint8       traj_flags (bit0 = VALID, bit1 = NEW)
    31–36  3 × int16   ax_imu, ay_imu, az_imu (v3)
    37–38  int16       yaw_imu  (v3)
    39     uint8       imu_seq  (v3, +1 for every new plant IMU sample)

  The IMU values feed the BNO085 emulator on the H7 (G0 v1 §5). The H7 stops
  sending IMU reports when imu_seq stops changing, so imu=None (seq 0 forever)
  means "no IMU" to the ESP32.

  MISO (H7 → RPi5)
    0–7    4 × int16   torque[0..3]  (H7 currently sends PWM duty here)
    8–15   4 × int16   pwm[0..3]     duty × 32767
    16     uint8       status
    17     uint8       seq (echo of MOSI seq)
    18–39  reserved    0x00

Encoding formula:
  int16 = round(clamp(v, ±range) / range × 32767)
  float = int16 / 32767 × range

Ranges (hil_protocol.h defines):
    ω1-4   : HIL_OMEGA_MAX      = 40.0  rad/s
    x, y   : HIL_POS_MAX        =  5.0  m
    θ      : HIL_THETA_MAX      = π     rad
    vx, vy : HIL_VEL_MAX        =  3.0  m/s
    wz     : HIL_OMEGA_BODY_MAX = 10.0  rad/s
    vx_ref, vy_ref : HIL_REF_VEL_MAX   = 3.0  m/s
    wz_ref         : HIL_REF_OMEGA_MAX = 10.0 rad/s
    torque : HIL_TORQUE_MAX     =  1.0  N·m
    gz_imu         : HIL_IMU_GYRO_MAX  = 34.906585 rad/s (±2000 deg/s)
    ax, ay, az_imu : HIL_IMU_ACCEL_MAX = 39.24 m/s² (±4 g)
    yaw_imu        : HIL_IMU_YAW_MAX   = π rad

The H7 relays the trajectory reference to the ESP32 over UART when
traj_flags has both VALID and NEW set (uart_relay.c, non-blocking through the
USART2 TX FIFO since branch h7-relay-wd). One packet takes ~1.2 ms on the wire,
so set NEW only on frames that carry a new setpoint (~50 Hz), never every 1 ms;
a packet that finds the FIFO still busy is dropped and counted on the H7.

CRITICAL: If you change anything here, update the H7 firmware decode
          in spi1_handler.c accordingly. A 1-byte offset here → all states
          wrong → controller gets garbage → system fails silently.
"""
import struct
import numpy as np

# ── Wire constants ────────────────────────────────────────────────────────
FRAME_SIZE   = 40           # bytes per SPI transaction (both directions)
N_STATES     = 10
N_TORQUES    = 4
N_REFS       = 3

# MOSI byte offsets
OFS_FAULT     = 20
OFS_SEQ       = 21
OFS_TRAJ_REF  = 24
OFS_TRAJ_FLAGS = 30
OFS_IMU_GZ     = 22
OFS_IMU_ACC    = 31         # ax, ay, az
OFS_IMU_YAW    = 37
OFS_IMU_SEQ    = 39

# MISO byte offsets
OFS_PWM       = 8
OFS_STATUS    = 16
OFS_MISO_SEQ  = 17

# traj_flags bits (HIL_TRAJ_VALID / HIL_TRAJ_NEW)
TRAJ_VALID = 1 << 0
TRAJ_NEW   = 1 << 1

# Must match hil_protocol.h defines exactly — field order: [ω1,ω2,ω3,ω4, x,y,θ, vx,vy,wz]
STATE_RANGES: list[float] = [40.0, 40.0, 40.0, 40.0,   # omega[0..3]  HIL_OMEGA_MAX
                              5.0, 5.0,                  # pos_x, pos_y HIL_POS_MAX
                              np.pi,                     # theta        HIL_THETA_MAX
                              3.0, 3.0,                  # vx, vy       HIL_VEL_MAX
                              10.0]                      # omega_body   HIL_OMEGA_BODY_MAX
REF_RANGES: list[float] = [3.0, 3.0,                    # vx_ref, vy_ref HIL_REF_VEL_MAX
                           10.0]                         # wz_ref         HIL_REF_OMEGA_MAX
IMU_RANGES: list[float] = [34.906585,                # gz   HIL_IMU_GYRO_MAX (±2000 deg/s)
                           39.24, 39.24, 39.24,      # ax, ay, az HIL_IMU_ACCEL_MAX (±4 g)
                           np.pi]                    # yaw  HIL_IMU_YAW_MAX
N_IMU = 5
TORQUE_RANGE: float = 1.0   # ±1.0 N·m
PWM_RANGE:    float = 1.0   # duty ±1.0


# ── Core conversion helpers ───────────────────────────────────────────────

def _to_int16(value: float, scale: float) -> int:
    """Clamp to ±scale then quantise to int16 range [-32767, 32767]."""
    clamped = max(-scale, min(scale, float(value)))
    return int(round(clamped / scale * 32767))


def _from_int16(raw: int, scale: float) -> float:
    """Recover float from int16 raw value."""
    return int(raw) / 32767.0 * scale


def _check_len(frame: bytes, need: int) -> None:
    if len(frame) < need:
        raise ValueError(f"Frame too short: {len(frame)} < {need}")


# ── MOSI: RPi5 → H7 ───────────────────────────────────────────────────────

def encode_state(state, traj_ref=None, traj_flags: int = 0,
                 seq: int = 0, fault_flags: int = 0,
                 imu=None, imu_seq: int = 0) -> bytes:
    """
    Pack plant state (+ optional trajectory reference, IMU) → 40-byte MOSI frame.

    Args:
        state:       Iterable of 10 floats [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz].
        traj_ref:    Optional (vx_ref, vy_ref, wz_ref), body frame [m/s, m/s, rad/s].
                     None → zeros.
        traj_flags:  TRAJ_VALID | TRAJ_NEW bits. 0 → H7 does not relay.
        seq:         Frame counter, stored modulo 256.
        fault_flags: Plant fault bits (HIL_FAULT_*).
        imu:         Optional (gz, ax, ay, az, yaw) [rad/s, m/s², m/s², m/s², rad]
                     for the BNO085 emulator. None → zeros.
        imu_seq:     Stored modulo 256; change it whenever imu is a new sample.

    Returns:
        40-byte bytes object ready to send via spidev.xfer2().
    """
    state = list(state)
    if len(state) != N_STATES:
        raise ValueError(f"Expected {N_STATES} states, got {len(state)}")

    buf = bytearray(FRAME_SIZE)   # initialised to 0x00
    for i, (val, rng) in enumerate(zip(state, STATE_RANGES)):
        struct.pack_into('>h', buf, i * 2, _to_int16(val, rng))
    buf[OFS_FAULT] = fault_flags & 0xFF
    buf[OFS_SEQ]   = seq & 0xFF

    if traj_ref is not None:
        traj_ref = list(traj_ref)
        if len(traj_ref) != N_REFS:
            raise ValueError(f"Expected {N_REFS} trajectory refs, got {len(traj_ref)}")
        for i, (val, rng) in enumerate(zip(traj_ref, REF_RANGES)):
            struct.pack_into('>h', buf, OFS_TRAJ_REF + i * 2, _to_int16(val, rng))
    buf[OFS_TRAJ_FLAGS] = traj_flags & 0xFF

    if imu is not None:
        imu = list(imu)
        if len(imu) != N_IMU:
            raise ValueError(f"Expected {N_IMU} IMU values, got {len(imu)}")
        ofs = [OFS_IMU_GZ, OFS_IMU_ACC, OFS_IMU_ACC + 2, OFS_IMU_ACC + 4, OFS_IMU_YAW]
        for o, val, rng in zip(ofs, imu, IMU_RANGES):
            struct.pack_into('>h', buf, o, _to_int16(val, rng))
    buf[OFS_IMU_SEQ] = imu_seq & 0xFF
    return bytes(buf)


def decode_state(frame: bytes) -> list:
    """
    Unpack MOSI frame → 10-element float list.

    Returns: [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz]
    Typically used for loopback testing.
    """
    _check_len(frame, N_STATES * 2)
    return [
        _from_int16(struct.unpack_from('>h', frame, i * 2)[0], rng)
        for i, rng in enumerate(STATE_RANGES)
    ]


def decode_traj_ref(frame: bytes) -> tuple:
    """
    Unpack trajectory part of a MOSI frame (mirror of H7 _decode()).

    Returns: (vx_ref, vy_ref, wz_ref, traj_flags)
    """
    _check_len(frame, OFS_TRAJ_FLAGS + 1)
    refs = [
        _from_int16(struct.unpack_from('>h', frame, OFS_TRAJ_REF + i * 2)[0], rng)
        for i, rng in enumerate(REF_RANGES)
    ]
    return refs[0], refs[1], refs[2], frame[OFS_TRAJ_FLAGS]


def decode_imu(frame: bytes) -> tuple:
    """
    Unpack the IMU part of a MOSI frame (mirror of H7 _decode(), v3).

    Returns: (gz, ax, ay, az, yaw, imu_seq)
    """
    _check_len(frame, OFS_IMU_SEQ + 1)
    ofs = [OFS_IMU_GZ, OFS_IMU_ACC, OFS_IMU_ACC + 2, OFS_IMU_ACC + 4, OFS_IMU_YAW]
    vals = [_from_int16(struct.unpack_from('>h', frame, o)[0], rng) for o, rng in zip(ofs, IMU_RANGES)]
    return (*vals, frame[OFS_IMU_SEQ])


# ── MISO: H7 → RPi5 ───────────────────────────────────────────────────────

def encode_torques(torques, pwm=None, status: int = 0, seq: int = 0) -> bytes:
    """
    Pack 4 torques (+ optional 4 PWM duties) → 40-byte MISO frame.

    Used for loopback testing (H7 normally encodes this).
    """
    torques = list(torques)
    if len(torques) != N_TORQUES:
        raise ValueError(f"Expected {N_TORQUES} torques, got {len(torques)}")

    buf = bytearray(FRAME_SIZE)
    for i, val in enumerate(torques):
        struct.pack_into('>h', buf, i * 2, _to_int16(val, TORQUE_RANGE))
    if pwm is not None:
        pwm = list(pwm)
        if len(pwm) != N_TORQUES:
            raise ValueError(f"Expected {N_TORQUES} pwm values, got {len(pwm)}")
        for i, val in enumerate(pwm):
            struct.pack_into('>h', buf, OFS_PWM + i * 2, _to_int16(val, PWM_RANGE))
    buf[OFS_STATUS]   = status & 0xFF
    buf[OFS_MISO_SEQ] = seq & 0xFF
    return bytes(buf)


def decode_torques(frame: bytes) -> list:
    """
    Unpack MISO frame → 4 torque floats [N·m].

    Called by HILNode after every SPI transaction to extract τ from H7.
    """
    _check_len(frame, N_TORQUES * 2)
    return [
        _from_int16(struct.unpack_from('>h', frame, i * 2)[0], TORQUE_RANGE)
        for i in range(N_TORQUES)
    ]


def decode_output(frame: bytes) -> dict:
    """
    Unpack full MISO frame (mirror of H7 _encode()).

    Returns: {'torque': [4], 'pwm': [4], 'status': int, 'seq': int}
    """
    _check_len(frame, OFS_MISO_SEQ + 1)
    return {
        'torque': decode_torques(frame),
        'pwm': [
            _from_int16(struct.unpack_from('>h', frame, OFS_PWM + i * 2)[0], PWM_RANGE)
            for i in range(N_TORQUES)
        ],
        'status': frame[OFS_STATUS],
        'seq':    frame[OFS_MISO_SEQ],
    }


# ── LSB reference values (for documentation / verification) ──────────────
LSB_TORQUE  = TORQUE_RANGE  / 32767   # ≈ 3.05e-5  N·m
LSB_STATE   = [rng / 32767 for rng in STATE_RANGES]
# [1.22e-3 rad/s (ω), 1.53e-4 m (pos), 9.59e-5 rad (θ), 9.16e-5 m/s (vx/vy), 3.05e-4 rad/s (wz)]
LSB_REF     = [rng / 32767 for rng in REF_RANGES]
