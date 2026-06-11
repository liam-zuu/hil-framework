"""
Trajectory Publisher — Process 2 (Mission computer, 50 Hz)

Responsibility:
  - Compute trajectory reference at time t  (50 Hz)
  - Pack into 18-byte UART packet
  - Send to ESP32 via UART

This process is completely independent of Process 1 (plant loop).
It uses wall-clock time — no synchronisation with the plant tick needed.

UART Packet (18 bytes):
  Offset  Field      Type        Description
  ─────── ─────────  ──────────  ──────────────────────────
  0       header     uint8       0xBB (sync byte)
  1–4     x_ref      float32 LE  world-frame X reference [m]
  5–8     y_ref      float32 LE  world-frame Y reference [m]
  9–12    theta_ref  float32 LE  heading reference [rad]
  13–16   vx_ref     float32 LE  forward speed reference [m/s]
  17      checksum   uint8       XOR of bytes 1..16

ESP32 fail-safe: if no packet received in 100 ms → hold last reference.

Supported trajectories (matching MATLAB trajectory_generator.m M5.2):
  idle     — hold origin, zero velocity
  line     — straight line along +X
  circle   — constant-radius circle starting at origin
  figure8  — figure-8 with curvature-based parametric form (no gradient spike)
"""
from __future__ import annotations

import logging
import math
import signal
import struct
import time
from dataclasses import dataclass
from typing import Tuple

try:
    import serial as _serial
    _SERIAL_OK = True
except ImportError:
    _serial    = None
    _SERIAL_OK = False

log = logging.getLogger('hil.traj')

UART_HEADER   = 0xBB
PACKET_SIZE   = 18
UART_BAUDRATE = 115_200


# ── Trajectory specification ──────────────────────────────────────────────

@dataclass
class TrajectorySpec:
    """Defines a reference trajectory for the run."""
    kind:     str   = 'idle'   # 'idle'|'line'|'circle'|'figure8'
    v:        float = 0.3      # nominal linear speed [m/s]
    radius:   float = 0.5      # circle or figure-8 amplitude [m]
    period:   float = 8.0      # figure-8 full period [s]
    duration: float = 0.0      # 0 = run forever


# ── Reference trajectory functions ───────────────────────────────────────
# Each returns (x_ref, y_ref, theta_ref, vx_ref) at time t

def _traj_idle(t: float, s: TrajectorySpec) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, 0.0, 0.0


def _traj_line(t: float, s: TrajectorySpec) -> Tuple[float, float, float, float]:
    """Straight line along +X at constant speed."""
    return s.v * t, 0.0, 0.0, s.v


def _traj_circle(t: float, s: TrajectorySpec) -> Tuple[float, float, float, float]:
    """
    Constant-radius circle starting at origin.
    The robot travels counter-clockwise at speed v.
    """
    R  = s.radius
    wz = s.v / R                    # angular rate [rad/s]
    x  = R * math.sin(wz * t)
    y  = R * (1.0 - math.cos(wz * t))
    theta = math.atan2(math.cos(wz * t), math.sin(wz * t))  # tangent direction
    return x, y, theta, s.v


def _traj_figure8(t: float, s: TrajectorySpec) -> Tuple[float, float, float, float]:
    """
    Figure-8 using curvature-based parametric equations.
    No gradient spike at crossover point (fixed MATLAB M5.1 bug).

    Path: x = A*sin(τ),  y = (A/2)*sin(2τ)  where τ = 2π*t/T
    Speed-normalised so |velocity| ≈ s.v.
    """
    A   = s.radius
    T   = s.period
    tau = 2.0 * math.pi * t / T

    x    = A * math.sin(tau)
    y    = 0.5 * A * math.sin(2.0 * tau)

    # Velocity components for heading and speed normalisation
    dxdt = A * (2.0 * math.pi / T) * math.cos(tau)
    dydt = A * (2.0 * math.pi / T) * math.cos(2.0 * tau)
    speed = math.hypot(dxdt, dydt) + 1e-9

    theta = math.atan2(dydt, dxdt)
    vx    = s.v * dxdt / speed
    return x, y, theta, vx


_TRAJ_FNS = {
    'idle':    _traj_idle,
    'line':    _traj_line,
    'circle':  _traj_circle,
    'figure8': _traj_figure8,
}


def get_reference(t: float, spec: TrajectorySpec) -> Tuple[float, float, float, float]:
    """
    Compute reference (x, y, theta, vx) at time t for the given spec.

    Returns (0, 0, 0, 0) for unknown trajectory kinds.
    """
    fn = _TRAJ_FNS.get(spec.kind, _traj_idle)
    return fn(t, spec)


# ── UART packet encoding ──────────────────────────────────────────────────

def _xor8(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result & 0xFF


def encode_packet(x: float, y: float, theta: float, vx: float) -> bytes:
    """
    Pack reference into 18-byte UART packet with XOR checksum.

    Float layout: little-endian float32 (standard for ARM Cortex-M).
    ESP32 decoder must use same endianness.
    """
    payload  = struct.pack('<ffff', x, y, theta, vx)   # 16 bytes
    checksum = _xor8(payload)
    return bytes([UART_HEADER]) + payload + bytes([checksum])


def decode_packet(pkt: bytes) -> Tuple[bool, float, float, float, float]:
    """
    Decode 18-byte UART packet (for testing / loopback verification).

    Returns:
        (valid, x, y, theta, vx)
        valid = False if header or checksum mismatch.
    """
    if len(pkt) != PACKET_SIZE:
        return False, 0.0, 0.0, 0.0, 0.0
    if pkt[0] != UART_HEADER:
        return False, 0.0, 0.0, 0.0, 0.0
    payload  = pkt[1:17]
    checksum = pkt[17]
    if _xor8(payload) != checksum:
        return False, 0.0, 0.0, 0.0, 0.0
    x, y, theta, vx = struct.unpack('<ffff', payload)
    return True, x, y, theta, vx


# ── Publisher process ─────────────────────────────────────────────────────

class TrajectoryPublisher:
    """
    Sends trajectory references to ESP32 over UART at 50 Hz.

    Designed to run in its own process (Process 2).  Uses wall-clock time
    so it does not need to coordinate with the 1 kHz plant loop.
    """

    def __init__(
        self,
        spec:     TrajectorySpec,
        port:     str   = '/dev/ttyAMA0',
        rate_hz:  int   = 50,
        simulate: bool  = False,
    ):
        """
        Args:
            spec:     Trajectory specification.
            port:     UART device (check: ls /dev/ttyAMA* /dev/serial*).
            rate_hz:  Publish rate [Hz]. 50 Hz is ample for the outer loop.
            simulate: Skip actual UART (for dev / CI testing).
        """
        self.spec      = spec
        self.port      = port
        self.rate_hz   = rate_hz
        self.simulate  = simulate

        self._uart     = None
        self._running  = False
        self._n_sent   = 0
        self._n_errors = 0

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block and publish.  Call as multiprocessing.Process target."""
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT,  self._on_signal)

        if not self.simulate:
            self._open_uart()

        period_s  = 1.0 / self.rate_hz
        t_start   = time.monotonic()
        next_tick = t_start + period_s
        self._running = True

        log.info(
            "Trajectory publisher start — kind=%s  v=%.2f m/s  "
            "r=%.2f m  @ %d Hz  port=%s",
            self.spec.kind, self.spec.v, self.spec.radius,
            self.rate_hz, self.port,
        )

        try:
            while self._running:
                now = time.monotonic()
                t   = now - t_start

                if self.spec.duration > 0 and t >= self.spec.duration:
                    log.info("Trajectory duration %.1f s reached.", self.spec.duration)
                    break

                x, y, theta, vx = get_reference(t, self.spec)
                pkt = encode_packet(x, y, theta, vx)
                self._send(pkt)
                self._n_sent += 1

                # Coarse sleep to next tick (50 Hz — jitter OK here)
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_tick += period_s

        finally:
            log.info(
                "Trajectory publisher stopped — sent=%d  errors=%d",
                self._n_sent, self._n_errors,
            )
            if self._uart is not None:
                self._uart.close()

    # ── Internals ─────────────────────────────────────────────────────────

    def _open_uart(self) -> None:
        if not _SERIAL_OK:
            raise ImportError(
                "pyserial not installed.  Run: pip install pyserial"
            )
        self._uart = _serial.Serial(
            self.port,
            baudrate=UART_BAUDRATE,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=0.1,
        )
        log.info("UART opened: %s @ %d baud", self.port, UART_BAUDRATE)

    def _send(self, pkt: bytes) -> None:
        if self.simulate:
            return
        try:
            self._uart.write(pkt)
        except Exception as exc:
            self._n_errors += 1
            log.warning("UART send error: %s", exc)

    def _on_signal(self, signum: int, _frame) -> None:
        log.info("Trajectory publisher signal %d — stopping.", signum)
        self._running = False
