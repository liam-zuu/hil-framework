"""
Trajectory Publisher — Process 2 (Mission computer, 50 Hz)

Responsibility:
  - Compute trajectory reference at time t  (50 Hz)
  - Default (HIL v2): write body-frame velocity setpoint (vx, vy, wz) into a
    shared array. Process 1 packs it into the 40-byte SPI frame, the H7
    relays it to the ESP32 as the 14-byte 0xBB UART packet.
  - Legacy (uart_direct=True): pack the 18-byte packet below and send it to
    the ESP32 over UART directly. The current ESP32 traj_receiver expects the
    14-byte packet, so this path is kept only for old tests.

This process is completely independent of Process 1 (plant loop).
It uses wall-clock time — no synchronisation with the plant tick needed.

Shared setpoint array (mp.Array('d', 4)):
  [0] vx_ref  [m/s]    body frame
  [1] vy_ref  [m/s]    body frame (0 for these trajectories)
  [2] wz_ref  [rad/s]
  [3] counter          incremented on every new setpoint (Process 1 sets
                       TRAJ_VALID | TRAJ_NEW only when it changes)

Legacy UART Packet (18 bytes):
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
Body-velocity only (spec S02 group A, get_body_velocity):
  strafe        — vy = v, vx = 0, heading held
  lift_step     — wheels lifted: vx 0 -> v -> 0, 3 s each
  lift_reverse  — wheels lifted: vx +v -> -v -> +v, 3 s each
  lift_ramp     — wheels lifted: vx ramp -v -> +v over 10 s
GROUP_A maps the spec S02 test IDs (A-L1 ... A-K3) to a TrajectorySpec.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import signal
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

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
    kind:     str   = 'idle'   # 'idle'|'line'|'circle'|'figure8'|'strafe'|'lift_*'
    v:        float = 0.3      # nominal linear speed [m/s]
    radius:   float = 0.5      # circle or figure-8 amplitude [m]
    period:   float = 8.0      # figure-8 full period [s]
    duration: float = 0.0      # 0 = run forever
    loop:     bool  = False    # S06 V7: at duration, start the trajectory again


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


def get_body_velocity(t: float, spec: TrajectorySpec) -> Tuple[float, float, float]:
    """
    Body-frame velocity setpoint (vx, vy, wz) at time t, for the ESP32.

    The robot keeps its heading tangent to the path, so vy = 0 and
    wz = path curvature × path speed.  Time scaling matches get_reference(),
    so integrating these setpoints follows the same (x, y) path.
    """
    if spec.kind == 'line':
        return spec.v, 0.0, 0.0
    if spec.kind == 'strafe':
        return 0.0, spec.v, 0.0
    if spec.kind == 'lift_step':
        return (spec.v if 3.0 <= t < 6.0 else 0.0), 0.0, 0.0
    if spec.kind == 'lift_reverse':
        return (-spec.v if 3.0 <= t < 6.0 else spec.v) if t < 9.0 else 0.0, 0.0, 0.0
    if spec.kind == 'lift_ramp':
        return (-spec.v + 2.0 * spec.v * min(t, LIFT_RAMP_S) / LIFT_RAMP_S), 0.0, 0.0
    if spec.kind == 'circle':
        return spec.v, 0.0, spec.v / spec.radius
    if spec.kind == 'figure8':
        A   = spec.radius
        w   = 2.0 * math.pi / spec.period
        tau = w * t
        dx  = A * w * math.cos(tau)
        dy  = A * w * math.cos(2.0 * tau)
        ddx = -A * w * w * math.sin(tau)
        ddy = -2.0 * A * w * w * math.sin(2.0 * tau)
        speed2 = dx * dx + dy * dy + 1e-12
        return math.sqrt(speed2), 0.0, (dx * ddy - dy * ddx) / speed2
    return 0.0, 0.0, 0.0


# ── Spec S02 group A ──────────────────────────────────────────────────────

LIFT_RAMP_S = 10.0


def figure8_period(A: float, v: float, n: int = 20000) -> float:
    """Period giving mean path speed v on x = A sin(tau), y = (A/2) sin(2 tau)."""
    tau = (np.arange(n) + 0.5) * (2.0 * math.pi / n)
    ds = np.hypot(A * np.cos(tau), A * np.cos(2.0 * tau)) * (2.0 * math.pi / n)
    return float(np.sum(ds)) / v


def _group_a():
    g = {}
    for i, v in ((1, 0.2), (2, 0.5)):
        g[f'A-L{i}'] = TrajectorySpec('line', v=v, duration=3.0 / v)                       # 3 m
        g[f'A-C{i}'] = TrajectorySpec('circle', v=v, radius=1.0, duration=2 * math.pi * 1.0 / v)   # 1 lap
        T8 = figure8_period(1.0, v)
        g[f'A-F{i}'] = TrajectorySpec('figure8', v=v, radius=1.0, period=T8, duration=T8)  # 1 period
        g[f'A-S{i}'] = TrajectorySpec('strafe', v=v, duration=2.0 / v)                     # 2 m
    g['A-K1'] = TrajectorySpec('lift_step', v=0.3, duration=9.0)
    g['A-K2'] = TrajectorySpec('lift_reverse', v=0.3, duration=9.0)
    g['A-K3'] = TrajectorySpec('lift_ramp', v=0.2, duration=LIFT_RAMP_S)
    return g


GROUP_A = _group_a()          # test ID -> TrajectorySpec (spec S02 P2-1)
LIFTED_TESTS = ('A-K1', 'A-K2', 'A-K3')


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
        spec:        TrajectorySpec,
        port:        str   = '/dev/ttyAMA0',
        rate_hz:     int   = 50,
        simulate:    bool  = False,
        shared_ref                 = None,
        uart_direct: Optional[bool] = None,
    ):
        """
        Args:
            spec:        Trajectory specification.
            port:        UART device (check: ls /dev/ttyAMA* /dev/serial*).
            rate_hz:     Publish rate [Hz]. 50 Hz is ample for the outer loop.
            simulate:    Skip actual UART (for dev / CI testing).
            shared_ref:  mp.Array('d', 4) read by Process 1 (HIL v2 path).
            uart_direct: Also send the legacy 18-byte packet over UART.
                         Default: True only when shared_ref is None.
        """
        self.spec        = spec
        self.port        = port
        self.rate_hz     = rate_hz
        self.simulate    = simulate
        self.shared_ref  = shared_ref
        self.uart_direct = (shared_ref is None) if uart_direct is None else uart_direct

        self._uart     = None
        self._running  = False
        self._n_sent   = 0
        self._n_errors = 0

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block and publish.  Call as multiprocessing.Process target."""
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT,  self._on_signal)

        if self.uart_direct and not self.simulate:
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
                    if self.spec.loop:            # S06 V7: repeat for the whole run
                        t_start += self.spec.duration
                        t = now - t_start
                    else:
                        log.info("Trajectory duration %.1f s reached.", self.spec.duration)
                        break

                if self.shared_ref is not None:
                    vx_b, vy_b, wz_b = get_body_velocity(t, self.spec)
                    with self.shared_ref.get_lock():
                        self.shared_ref[0] = vx_b
                        self.shared_ref[1] = vy_b
                        self.shared_ref[2] = wz_b
                        self.shared_ref[3] += 1.0
                if self.uart_direct:
                    x, y, theta, vx = get_reference(t, self.spec)
                    self._send(encode_packet(x, y, theta, vx))
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
