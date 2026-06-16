"""
SPI Wire Protocol — must match H7 hil_protocol.h byte-for-byte.

Frame layout (24 bytes, full-duplex):
  MOSI (RPi5 → H7): 10 × int16 big-endian  (states)  + 4 bytes padding
  MISO (H7 → RPi5):  4 × int16 big-endian  (torques) + 16 bytes padding

Encoding formula:
  int16 = round(clamp(v, ±range) / range × 32767)
  float = int16 / 32767 × range

MOSI field order [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz]:
  State ranges must match hil_protocol.h defines:
    ω1-4  : HIL_OMEGA_MAX      = 40.0  rad/s
    x, y  : HIL_POS_MAX        = 10.0  m
    θ     : HIL_THETA_MAX      = π     rad
    vx, vy: HIL_VEL_MAX        =  3.0  m/s
    wz    : HIL_OMEGA_BODY_MAX = 10.0  rad/s

Torque range: ±1.0 N·m (HIL_TORQUE_MAX)

CRITICAL: If you change anything here, update the H7 firmware decode
          in spi1_handler.c accordingly. A 1-byte offset here → all states
          wrong → controller gets garbage → system fails silently.
"""
import struct
import numpy as np

# ── Wire constants ────────────────────────────────────────────────────────
FRAME_SIZE   = 24           # bytes per SPI transaction (both directions)
N_STATES     = 10
N_TORQUES    = 4

# Must match hil_protocol.h defines exactly — field order: [ω1,ω2,ω3,ω4, x,y,θ, vx,vy,wz]
STATE_RANGES: list[float] = [40.0, 40.0, 40.0, 40.0,   # omega[0..3]  HIL_OMEGA_MAX
                              10.0, 10.0,                # pos_x, pos_y HIL_POS_MAX
                              np.pi,                     # theta        HIL_THETA_MAX
                              3.0, 3.0,                  # vx, vy       HIL_VEL_MAX
                              10.0]                      # omega_body   HIL_OMEGA_BODY_MAX
TORQUE_RANGE: float = 1.0   # ±1.0 N·m


# ── Core conversion helpers ───────────────────────────────────────────────

def _to_int16(value: float, scale: float) -> int:
    """Clamp to ±scale then quantise to int16 range [-32767, 32767]."""
    clamped = max(-scale, min(scale, float(value)))
    return int(round(clamped / scale * 32767))


def _from_int16(raw: int, scale: float) -> float:
    """Recover float from int16 raw value."""
    return int(raw) / 32767.0 * scale


# ── Public encode / decode API ────────────────────────────────────────────

def encode_state(state) -> bytes:
    """
    Pack 10-element state vector → 24-byte MOSI frame.

    Bytes 20-23 are zero-padding (unused, reserved for future states).
    Big-endian int16 to match H7 SPI slave DMA byte order.

    Args:
        state: Iterable of 10 floats [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz].

    Returns:
        24-byte bytes object ready to send via spidev.xfer2().
    """
    state = list(state)
    if len(state) != N_STATES:
        raise ValueError(f"Expected {N_STATES} states, got {len(state)}")

    buf = bytearray(FRAME_SIZE)   # initialised to 0x00
    for i, (val, rng) in enumerate(zip(state, STATE_RANGES)):
        struct.pack_into('>h', buf, i * 2, _to_int16(val, rng))
    return bytes(buf)


def decode_state(frame: bytes) -> list:
    """
    Unpack 24-byte frame → 10-element float list.

    Returns: [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz]
    Typically used for loopback testing.
    """
    if len(frame) < N_STATES * 2:
        raise ValueError(f"Frame too short: {len(frame)} < {N_STATES * 2}")
    return [
        _from_int16(struct.unpack_from('>h', frame, i * 2)[0], rng)
        for i, rng in enumerate(STATE_RANGES)
    ]


def encode_torques(torques) -> bytes:
    """
    Pack 4 torques → 24-byte frame.

    Used for loopback testing (H7 normally encodes this).
    Bytes 8-23 are zero-padding.
    """
    torques = list(torques)
    if len(torques) != N_TORQUES:
        raise ValueError(f"Expected {N_TORQUES} torques, got {len(torques)}")

    buf = bytearray(FRAME_SIZE)
    for i, val in enumerate(torques):
        struct.pack_into('>h', buf, i * 2, _to_int16(val, TORQUE_RANGE))
    return bytes(buf)


def decode_torques(frame: bytes) -> list:
    """
    Unpack 24-byte MISO frame → 4 torque floats [N·m].

    Called by HILNode after every SPI transaction to extract τ from H7.
    """
    if len(frame) < N_TORQUES * 2:
        raise ValueError(f"Frame too short: {len(frame)} < {N_TORQUES * 2}")
    return [
        _from_int16(struct.unpack_from('>h', frame, i * 2)[0], TORQUE_RANGE)
        for i in range(N_TORQUES)
    ]


# ── LSB reference values (for documentation / verification) ──────────────
LSB_TORQUE  = TORQUE_RANGE  / 32767   # ≈ 3.05e-5  N·m
LSB_STATE   = [rng / 32767 for rng in STATE_RANGES]
# [1.22e-3 rad/s (ω), 3.05e-4 m (pos), 9.59e-5 rad (θ), 9.16e-5 m/s (vx/vy), 3.05e-4 rad/s (wz)]