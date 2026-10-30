"""
HIL Node — Process 1 (Plant simulation, 1 kHz)

Responsibility:
  - Wait for 1 ms tick (PrecisionTimer)
  - GPIO pulse → H7 (alert: SPI frame imminent)
  - SPI full-duplex: send encoded state + trajectory setpoint, receive torques
  - Decode torques → plant.step() → update state
  - Expose state via shared multiprocessing.Array for Process 2
  - Read the setpoint written by Process 2 (shared_ref); flag it VALID|NEW
    only on the tick after it changes, so the H7 relays it at ~50 Hz

Real-time requirements:
  - Must complete each tick in < 1 ms (1 ms period)
  - Typical work: GPIO pulse (~10 μs) + SPI 40 bytes (320 μs at 1 MHz,
    72 μs at 4 MHz) + spidev overhead + plant (~50 μs)
  - Total ≈ 400 μs at 1 MHz → still inside 1 ms; raise spi_speed_hz to
    2–4 MHz after the link is validated to regain margin

CPU allocation:
  - This process: core 0, SCHED_FIFO priority 90
  - Trajectory publisher: core 2, normal scheduling
  - OS / kernel: cores 1, 3
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass
from typing import Optional

from timing import PrecisionTimer, set_realtime, pin_cpu
from gpio_sync import GPIOSync
from spi_master import SPIMaster
from protocol import encode_state, decode_torques, TRAJ_VALID, TRAJ_NEW
from plant_interface import PlantInterface, MockPlant
from mecanum_plant import MecanumPlant
from imu_model import imu_from_plant
import t4_imu_profile
from params import PARAMS

log = logging.getLogger('hil.node')


def plant_to_frame_state(state: list) -> list:
    """
    MecanumPlant state [x, y, θ, vx, vy, wz, ω1..ω4] (plant_step.py) → the
    order protocol.encode_state expects [ω1..ω4, x, y, θ, vx, vy, wz].
    Bug HIL-STATE-ORDER (27/10): the plant state went into the frame as is,
    so the H7 took x, y, θ, vx as the four wheel speeds.
    """
    return list(state[6:10]) + list(state[0:6])


@dataclass
class HILConfig:
    """Runtime configuration for the HIL node."""
    # Timing
    period_us:         int   = 1_000       # 1 kHz
    # Real-time
    rt_priority:       int   = 90          # SCHED_FIFO priority [1-99]
    cpu_core:          int   = 0           # plant loop CPU core
    # GPIO
    gpio_pin:          int   = 17          # BCM sync pin
    gpio_chip:         int   = 4           # 4 for RPi5 (RP1), 0 for RPi4
    # SPI
    spi_bus:           int   = 0
    spi_device:        int   = 0
    spi_speed_hz:      int   = 1_000_000
    # Loop control
    max_steps:         int   = 0           # 0 = run indefinitely
    simulate_hw:       bool  = False       # True = no GPIO/SPI hardware
    # IMU for the BNO085 emulator (G2): (gz, ax, ay, az, yaw) sent in every
    # frame instead of the plant values; None = plant values (MecanumPlant)
    imu_static:        Optional[tuple] = None
    # S06 T4 (4): 't4' = locked IMU sequence of t4_imu_profile.py by frame time
    imu_profile:       Optional[str]   = None
    # Monitoring
    # S06 (29/10): raw MOSI + MISO of every frame and the send time, kept in
    # preallocated arrays inside the loop (one index write per step) and
    # saved to this .npz after the loop. '' = off. frame_log_max caps the
    # memory when max_steps is 0 (88 bytes per frame).
    frame_log:         str   = ''
    frame_log_max:     int   = 1_200_000   # 20 min at 1 kHz
    jitter_warn_us:    float = 500.0       # warn if jitter exceeds this
    stats_interval:    int   = 10_000      # print stats every N steps


class HILNode:
    """
    Orchestrates the 1 kHz plant simulation loop.

    Designed to run inside a dedicated OS process (via multiprocessing).
    Call run() from the process target function.

    Example (inside Process 1 target):
        plant  = MecanumPlant(dt=0.001)
        shared = mp.Array('d', 10)
        node   = HILNode(plant, config=HILConfig(), shared_state=shared)
        node.run()
    """

    def __init__(
        self,
        plant:         PlantInterface,
        config:        HILConfig          = None,
        shared_state:  Optional[mp.Array] = None,
        shared_ref:    Optional[mp.Array] = None,
    ):
        """
        Args:
            plant:        Plant implementation.
            config:       HILConfig.  Defaults to simulate_hw=True, 1 kHz.
            shared_state: mp.Array('d', 10) for IPC with trajectory publisher.
                          Write-locked on each tick.  Optional.
            shared_ref:   mp.Array('d', 4) [vx_ref, vy_ref, wz_ref, counter]
                          written by the trajectory publisher.  Optional;
                          None → no setpoint sent (traj_flags = 0).
        """
        self.plant        = plant
        self.cfg          = config or HILConfig(simulate_hw=True)
        self.shared_state = shared_state
        self.shared_ref   = shared_ref
        self._ref         = (0.0, 0.0, 0.0)
        self._ref_count   = 0.0

        self._gpio  = GPIOSync(
            pin=self.cfg.gpio_pin,
            gpiochip=self.cfg.gpio_chip,
            simulate=self.cfg.simulate_hw,
        )
        self._spi   = SPIMaster(
            bus=self.cfg.spi_bus,
            device=self.cfg.spi_device,
            speed_hz=self.cfg.spi_speed_hz,
            simulate=self.cfg.simulate_hw,
        )
        self._timer = PrecisionTimer(
            period_us=self.cfg.period_us,
            history_len=max(self.cfg.stats_interval, 5_000),
        )

        self._running   = False
        self._step      = 0
        self._imu_seq   = 0
        self._prev_state: Optional[list] = None
        # MecanumPlant keeps its own state order; MockPlant (--fixed-omega)
        # already uses the frame order
        self._mecanum   = isinstance(plant, MecanumPlant)
        self._t_start:  Optional[float] = None
        self._stop_reason = 'max_steps'
        self._flog = None
        if self.cfg.frame_log:
            import numpy as np
            n = self.cfg.max_steps or self.cfg.frame_log_max
            # t_pulse0/1_ns: just before / just after raising the sync pulse (S06 spec line 111:
            # P_send := t_pulse_rise, the same physical edge as d_1w; tools/s06_frames_decode.py)
            self._flog = {
                't_send_ns':   np.zeros(n, dtype=np.int64),
                't_pulse0_ns': np.zeros(n, dtype=np.int64),
                't_pulse1_ns': np.zeros(n, dtype=np.int64),
                'mosi':        np.zeros((n, 40), dtype=np.uint8),
                'miso':        np.zeros((n, 40), dtype=np.uint8),
            }
            self._flog_n = n

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Block and run the HIL loop.

        Sets SCHED_FIFO and pins to cpu_core before entering the loop.
        Installs SIGTERM / SIGINT for clean shutdown.
        """
        set_realtime(priority=self.cfg.rt_priority)
        pin_cpu(core=self.cfg.cpu_core)

        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT,  self._on_signal)

        self._running = True
        self._t_start = time.monotonic()
        # S06 W1/Y6: run start on the clock of t_send_ns (monotonic, us)
        log.info('#run_start,timestamp_us=%d,max_steps=%d,frame_log=%s',
                 time.monotonic_ns() // 1000, self.cfg.max_steps, self.cfg.frame_log or '-')

        log.info(
            "Plant loop start — %d Hz  simulate=%s  max_steps=%d",
            1_000_000 // self.cfg.period_us,
            self.cfg.simulate_hw,
            self.cfg.max_steps,
        )

        try:
            while self._running:
                if self.cfg.max_steps and self._step >= self.cfg.max_steps:
                    break
                if self._flog is not None and self._step >= self._flog_n:
                    self._stop_reason = 'frame_log_full'
                    break
                jitter_ns = self._timer.wait_next_tick()
                self._tick(jitter_ns)
        finally:
            self._shutdown()

    # ── Single tick ───────────────────────────────────────────────────────

    def _tick(self, jitter_ns: int) -> None:
        """
        Execute one 1 ms step:
          1. GPIO pulse  →  H7 (alert)
          2. SPI tx(state) / rx(torques)
          3. plant.step(torques)
          4. Write shared state
        """
        # 1. Sync pulse — H7 EXTI fires, enables DMA receive
        if self._flog is not None:
            t_p0 = time.monotonic_ns()
            self._gpio.pulse()
            t_p1 = time.monotonic_ns()
        else:
            self._gpio.pulse()

        # 2. Encode current state + setpoint, send MOSI, receive MISO (torques from H7)
        traj_flags = 0
        if self.shared_ref is not None:
            with self.shared_ref.get_lock():
                vx_r, vy_r, wz_r, count = self.shared_ref[:]
            if count != self._ref_count:
                self._ref_count = count
                self._ref       = (vx_r, vy_r, wz_r)
                traj_flags      = TRAJ_VALID | TRAJ_NEW
        state  = self.plant.get_state()
        imu, imu_seq = self._imu_sample(state)
        frame_state = plant_to_frame_state(state) if self._mecanum else state
        mosi   = encode_state(frame_state, traj_ref=self._ref, traj_flags=traj_flags,
                              seq=self._step, imu=imu, imu_seq=imu_seq)
        t_send = time.monotonic_ns()
        miso   = self._spi.transfer(mosi)
        torques = decode_torques(miso)
        if self._flog is not None:
            k = self._step
            self._flog['t_send_ns'][k] = t_send
            self._flog['t_pulse0_ns'][k] = t_p0
            self._flog['t_pulse1_ns'][k] = t_p1
            self._flog['mosi'][k] = memoryview(bytes(mosi))
            self._flog['miso'][k] = memoryview(bytes(miso))

        # 3. Advance plant dynamics
        new_state = self.plant.step(torques)

        # 4. Publish state to Process 2 (trajectory publisher)
        if self.shared_state is not None:
            with self.shared_state.get_lock():
                self.shared_state[:] = new_state

        self._step += 1

        # 5. Jitter monitoring
        if jitter_ns > self.cfg.jitter_warn_us * 1_000:
            log.warning(
                "Jitter overrun: step=%d  jitter=%.0f μs",
                self._step, jitter_ns / 1_000,
            )

        # 6. Periodic stats log
        if self._step % self.cfg.stats_interval == 0:
            self._log_stats()

    def _imu_sample(self, state: list):
        """
        IMU values for the BNO085 emulator (G2, Liam 27/10 Q3): the static
        values if set, else noise-free values from a MecanumPlant. Other plants
        (MockPlant, --fixed-omega) send no IMU (imu_seq stays 0).
        """
        if self.cfg.imu_static is not None:
            imu = self.cfg.imu_static
        elif self.cfg.imu_profile == 't4':
            imu = t4_imu_profile.sample(self._step * self.cfg.period_us / 1e6)
        elif self._mecanum:
            imu = imu_from_plant(state, self._prev_state,
                                 self.cfg.period_us / 1e6, PARAMS.g)
            self._prev_state = state
        else:
            return None, 0
        # a new sample every frame: 1..255, skipping 0 (before its first
        # sample the H7 takes imu_seq 0 as "no IMU yet")
        self._imu_seq = self._imu_seq % 255 + 1
        return imu, self._imu_seq

    # ── Cleanup & diagnostics ─────────────────────────────────────────────

    def _save_frame_log(self) -> None:
        import numpy as np
        n = self._step
        np.savez(self.cfg.frame_log, **{k: v[:n] for k, v in self._flog.items()})
        log.info('Frame log: %d frames -> %s', n, self.cfg.frame_log)

    def _on_signal(self, signum: int, _frame) -> None:
        log.info("Signal %d received — stopping plant loop.", signum)
        self._stop_reason = f'signal_{signum}'
        self._running = False

    def _shutdown(self) -> None:
        log.info('#run_end,timestamp_us=%d,steps=%d,reason=%s',
                 time.monotonic_ns() // 1000, self._step, self._stop_reason)
        if self._flog is not None:
            self._save_frame_log()
        elapsed = time.monotonic() - (self._t_start or time.monotonic())
        log.info(
            "Plant loop stopped — steps=%d  elapsed=%.1f s",
            self._step, elapsed,
        )
        self._timer.print_jitter_report()

        spi_stats = self._spi.stats
        if spi_stats['errors']:
            log.warning("SPI errors: %d / %d (%.1f%%)",
                        spi_stats['errors'], spi_stats['transfers'],
                        spi_stats['error_pct'])

        self._gpio.close()
        self._spi.close()

    def _log_stats(self) -> None:
        elapsed = time.monotonic() - (self._t_start or time.monotonic())
        rate    = self._step / elapsed if elapsed > 0 else 0
        s       = self._timer.jitter_stats
        log.info(
            "step=%-8d  t=%5.1f s  rate=%6.0f Hz  jitter p99=%.0f μs",
            self._step, elapsed, rate,
            s.get('p99_us', -1) if s else -1,
        )
