"""
HIL Node — Process 1 (Plant simulation, 1 kHz)

Responsibility:
  - Wait for 1 ms tick (PrecisionTimer)
  - GPIO pulse → H7 (alert: SPI frame imminent)
  - SPI full-duplex: send encoded state, receive encoded torques
  - Decode torques → plant.step() → update state
  - Expose state via shared multiprocessing.Array for Process 2

Real-time requirements:
  - Must complete each tick in < 1 ms (1 ms period)
  - Typical work: GPIO pulse (~10 μs) + SPI (~24 μs at 1 MHz) + plant (~50 μs)
  - Total ≈ 100–150 μs → comfortable margin

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
from protocol import encode_state, decode_torques
from plant_interface import PlantInterface, MockPlant

log = logging.getLogger('hil.node')


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
    # Monitoring
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
    ):
        """
        Args:
            plant:        Plant implementation.
            config:       HILConfig.  Defaults to simulate_hw=True, 1 kHz.
            shared_state: mp.Array('d', 10) for IPC with trajectory publisher.
                          Write-locked on each tick.  Optional.
        """
        self.plant        = plant
        self.cfg          = config or HILConfig(simulate_hw=True)
        self.shared_state = shared_state

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
        self._t_start:  Optional[float] = None

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
        self._gpio.pulse()

        # 2. Encode current state, send MOSI, receive MISO (torques from H7)
        state  = self.plant.get_state()
        mosi   = encode_state(state)
        miso   = self._spi.transfer(mosi)
        torques = decode_torques(miso)

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

    # ── Cleanup & diagnostics ─────────────────────────────────────────────

    def _on_signal(self, signum: int, _frame) -> None:
        log.info("Signal %d received — stopping plant loop.", signum)
        self._running = False

    def _shutdown(self) -> None:
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
