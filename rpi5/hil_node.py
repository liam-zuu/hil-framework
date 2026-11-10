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

import gc
import json
import logging
import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass
from typing import Optional

from timing import PrecisionTimer, set_realtime, pin_cpu, lock_memory
from gpio_sync import GPIOSync
from spi_master import SPIMaster
from protocol import (encode_state, decode_torques, TRAJ_VALID, TRAJ_NEW, OFS_MISO_SEQ, CMD_SWA_ON,
                      miso_check, MISO_OK, MISO_EMPTY)
from plant_interface import PlantInterface, MockPlant
from mecanum_plant import MecanumPlant
from imu_model import imu_from_plant
import t4_imu_profile
from params import PARAMS
from rt_monitor import RTMonitor
from s06_load_schedule import HeavyBlock
from step_log import StepLog
from run_lifecycle import ControlBox, LifecycleConfig, RunLifecycle, StepIn, END, RUN, STOPPING

# I8 (Liam 09/11, "Duyet ca 5"): who holds the RC switch SWA of the iBUS frames the H7 sends.
# 'phase' = on in RUN and STOPPING of --finite-run (needs cfg.lifecycle), 'on' / 'off' = fixed.
SWA_MODES = ('off', 'on', 'phase')
SWA_PHASES_ON = (RUN, STOPPING)   # STOPPING only after the run reached RUN (I8-1)
# I8-5: frames sent after the lifecycle reached END (SWA off, setpoint 0, no publisher), so the H7
# surely gets the off command and the frame log shows the H7 status follow it (filter 3 frames +
# iBUS period 7 ms + answer 2 frames, about 13 ms; 30 frames = 30 ms)
END_TAIL_FRAMES = 30
# I8-R2-2 (Codex round 2): 30 frames are not 30 ms when the timer catches up on late deadlines (it sends
# them back to back), so the tail also lasts END_TAIL_MS on time.monotonic_ns, from the END frame on.
END_TAIL_MS = 30
# A3: frames j < 2 are not checked: the H7 answers frame N in frame N + 2 and the first two frames still
# carry the reply staged in the previous link period (as s06_frames_decode `start_leftover`, L50: "không
# tính frame rỗng ở đầu log trước câu trả lời đầu tiên (j < 2)")
MISO_SKIP_FRAMES = 2

log =logging.getLogger('hil.node')


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
    # imu_model.ImuNoise (replay of the measured noise, L72) applied to the plant
    # IMU values every frame; None = noise-free plant values. Not applied to
    # imu_static / imu_profile (those are locked test sequences).
    imu_noise:         Optional[object] = None
    # Timing of the 1 kHz loop: lock memory, stop the cyclic GC and touch the
    # frame log before the loop. False = the run without them, to compare
    rt_tune:           bool  = True
    # Monitoring
    # S06 (29/10): raw MOSI + MISO of every frame and the send time, kept in
    # preallocated arrays inside the loop (one index write per step) and
    # saved to this .npz after the loop. '' = off. frame_log_max caps the
    # memory when max_steps is 0 (88 bytes per frame).
    frame_log:         str   = ''
    frame_log_max:     int   = 1_200_000   # 20 min at 1 kHz
    # S06 Z8: (first step, end step) of the heavy block of s06_load_schedule.py, None = no block;
    # load_schedule = 'path sha256' of the schedule file, kept in the run manifest
    heavy_block:       Optional[tuple] = None
    load_schedule:     str   = ''
    # --finite-run: phases of run_lifecycle.py; None = the loop as before
    lifecycle:         Optional[LifecycleConfig] = None
    # I8: SWA command bit (MOSI byte 20 bit 3), see SWA_MODES. Default 'off': a node
    # built without a choice never turns the DUT motors on (run_hil.py resolves --swa)
    swa_mode:          str   = 'off'
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
        monitor:       Optional[RTMonitor] = None,
        step_log:      Optional[StepLog]   = None,
        control:       Optional[ControlBox] = None,
        responder                           = None,
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
            monitor:      RTMonitor shared with a process outside the loop
                          (run_hil.py). Given: the loop only records lateness
                          counters, no logging and no statistics inside the
                          1 kHz loop (THESIS H2). None: the old in-loop log.
            step_log:     StepLog (S06 T2): t_sched, t_start, t_done, t_end of
                          every step into a shared ring, read outside the loop.
            control:      ControlBox of the run owner; needed with cfg.lifecycle.
            responder:    simulate mode only: fake_h7.FakeH7 instead of the echo.
        """
        self.plant        = plant
        self.cfg          = config or HILConfig(simulate_hw=True)
        self.shared_state = shared_state
        self.shared_ref   = shared_ref
        self.monitor      = monitor
        self.step_log     = step_log
        self._t_done_ns   = 0
        self.control      = control
        self._lc          = RunLifecycle(self.cfg.lifecycle) if self.cfg.lifecycle else None
        if self._lc is not None and control is None:
            raise ValueError('cfg.lifecycle needs a ControlBox')
        if self.cfg.swa_mode not in SWA_MODES:
            raise ValueError(f'swa_mode {self.cfg.swa_mode!r} not in {SWA_MODES}')
        if self.cfg.swa_mode == 'phase' and self._lc is None:
            raise ValueError("swa_mode 'phase' needs cfg.lifecycle (--finite-run)")
        self._swa       = None               # last bit sent; None before the first frame
        self._swa_events: list = []          # (step, 0/1) at every change, for the manifest
        self._run_seen  = False              # I8-1: the lifecycle reached RUN
        self._end_seen  = False              # I8-5: the lifecycle reached END, the tail frames follow
        self._tail_t0_ns: Optional[int] = None  # I8-R2-2: send time of the END frame
        self._tail_sent = 0                  # I8-R2-3: tail frames sent after the END frame
        self._tail_complete = False          # I8-R2-3: the tail ran its frames and its time
        # A3 (Liam 09/11, CODE: "ok tôi đồng ý hết"; STATUS_CODE L50, L61 b): MISO frames with a bad CRC
        # or all 0 from frame MISO_SKIP_FRAMES on, counted in the loop, reported at the end (#miso line,
        # manifest). Report only: the loop still takes the duty from every frame (L50 rule: a run with
        # any of them is not a main number, run again)
        self._miso_crc_bad = 0
        self._miso_empty   = 0
        self._miso_first_crc_bad: Optional[int] = None
        self._miso_first_empty: Optional[int] = None
        self._h7_seq      = -1
        self._h7_new      = False
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
            responder=responder,
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
        # L34 (Fable F-ND #7): git blobs of the plant_step.py / params.py this process imported,
        # taken at start; a SIM and a HIL run are compared only with the same blobs
        from plant_blob import plant_source
        self._plant_src = plant_source()
        log.info('Plant source blobs: %s', self._plant_src)
        self._heavy     = HeavyBlock(*self.cfg.heavy_block) if self.cfg.heavy_block else None
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
            # np.zeros maps pages lazily: without this, a page fault every ~100 frames inside the loop
            if self.cfg.rt_tune:
                for a in self._flog.values():
                    a.fill(0)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Block and run the HIL loop.

        Sets SCHED_FIFO and pins to cpu_core before entering the loop.
        Installs SIGTERM / SIGINT for clean shutdown.
        """
        self._rt = dict(sched_fifo=set_realtime(priority=self.cfg.rt_priority),
                        cpu_pinned=pin_cpu(core=self.cfg.cpu_core),
                        mem_locked=self.cfg.rt_tune and lock_memory(), gc_off=self.cfg.rt_tune)
        if self.cfg.rt_tune:
            # no GC pass inside the loop; freeze keeps what is alive now out of any later one
            gc.collect()
            gc.freeze()
            gc.disable()

        signal.signal(signal.SIGTERM, self._on_signal)
        # --finite-run: Ctrl+C reaches the whole process group; the owner turns it into STOP,
        # so the loop goes on to send the zero setpoints. SIGTERM still stops at once.
        signal.signal(signal.SIGINT, signal.SIG_IGN if self._lc is not None else self._on_signal)

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

        # S06 Y16: schedule from here, not from when the timer was built (before,
        # the first deadlines were already past and the first steps ran back to back)
        self._timer.reset()
        if self._lc is not None:
            self._lc.begin(time.monotonic())
        try:
            while self._running:
                if self.cfg.max_steps and self._step >= self.cfg.max_steps:
                    break
                if self._flog is not None and self._step >= self._flog_n:
                    self._stop_reason = 'frame_log_full'
                    break
                jitter_ns = self._timer.wait_next_tick()
                self._tick(jitter_ns)
                if self.step_log is not None:
                    # S06 T2 (Z9): t_end is taken inside record, after its fields
                    self.step_log.record(self._step - 1, self._timer.last_deadline_ns,
                                         self._timer.last_wake_ns, self._t_done_ns)
        finally:
            self._shutdown()
            if self.cfg.rt_tune:
                gc.enable()
                gc.unfreeze()

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
        tail = self._end_seen                # I8-5: after END, no lifecycle step, no publisher
        lc = self._lifecycle_step() if (self._lc is not None and not tail) else None
        if tail:
            pass                                 # keep the last setpoint (0 from STOPPING), flags 0
        elif lc is not None and lc.send_zero:
            self._ref  = (0.0, 0.0, 0.0)         # section 9.1: VALID|NEW, not through the publisher
            traj_flags = TRAJ_VALID | TRAJ_NEW
        elif (lc is None or lc.publish) and self.shared_ref is not None:
            with self.shared_ref.get_lock():
                vx_r, vy_r, wz_r, count = self.shared_ref[:]
            if count != self._ref_count:
                self._ref_count = count
                self._ref       = (vx_r, vy_r, wz_r)
                traj_flags      = TRAJ_VALID | TRAJ_NEW
        state  = self.plant.get_state()
        imu, imu_seq = self._imu_sample(state)
        frame_state = plant_to_frame_state(state) if self._mecanum else state
        if self._heavy is not None:
            self._heavy.apply(self._step, frame_state)   # S06 Z8: omega to the H7 only, the plant goes on
        swa = self._swa_bit()
        mosi   = encode_state(frame_state, traj_ref=self._ref, traj_flags=traj_flags,
                              seq=self._step, imu=imu, imu_seq=imu_seq,
                              fault_flags=CMD_SWA_ON if swa else 0)
        t_send = time.monotonic_ns()
        miso   = self._spi.transfer(mosi)
        if swa != self._swa:                 # I8-11: recorded once the frame went out
            self._swa = swa
            self._swa_events.append((self._step, swa))
        if tail:
            # I8-R2-2: both END_TAIL_FRAMES frames and END_TAIL_MS since the END frame went out
            self._tail_sent += 1
            if (self._tail_sent >= END_TAIL_FRAMES
                    and t_send - self._tail_t0_ns >= END_TAIL_MS * 1_000_000):
                self._tail_complete = True
                self._running = False
        elif self._end_seen:
            # this step reached END: the tail time starts once its frame is out (Codex round 3: not
            # t_send, taken before a transfer that may stall), so the H7 has had the off command
            self._tail_t0_ns = time.monotonic_ns()
        torques = decode_torques(miso)
        if self._step >= MISO_SKIP_FRAMES:
            chk = miso_check(miso)
            if chk != MISO_OK:
                if chk == MISO_EMPTY:
                    self._miso_empty += 1
                    if self._miso_first_empty is None:
                        self._miso_first_empty = self._step
                else:
                    self._miso_crc_bad += 1
                    if self._miso_first_crc_bad is None:
                        self._miso_first_crc_bad = self._step
        if self._lc is not None:
            self._h7_new = miso[OFS_MISO_SEQ] != self._h7_seq
            self._h7_seq = miso[OFS_MISO_SEQ]
        if self._flog is not None:
            k = self._step
            self._flog['t_send_ns'][k] = t_send
            self._flog['t_pulse0_ns'][k] = t_p0
            self._flog['t_pulse1_ns'][k] = t_p1
            self._flog['mosi'][k] = memoryview(bytes(mosi))
            self._flog['miso'][k] = memoryview(bytes(miso))

        # 3. Advance plant dynamics
        # (ARM holds the plant at its initial state: no integration of the old DUT's PWM)
        new_state = state if lc is not None and lc.hold_plant else self.plant.step(torques)

        # 4. Publish state to Process 2 (trajectory publisher)
        if self.shared_state is not None:
            with self.shared_state.get_lock():
                self.shared_state[:] = new_state
        self._t_done_ns = time.monotonic_ns()      # S06 T2 t_done (Z9)

        self._step += 1

        # 5. Jitter monitoring
        if self.monitor is not None:
            # a few integer writes; the stats are printed by another process
            self.monitor.record(self._step, jitter_ns, self._timer.overruns)
            return
        if jitter_ns > self.cfg.jitter_warn_us * 1_000:
            log.warning(
                "Jitter overrun: step=%d  jitter=%.0f μs",
                self._step, jitter_ns / 1_000,
            )

        # 6. Periodic stats log
        if self._step % self.cfg.stats_interval == 0:
            self._log_stats()

    def _swa_bit(self) -> int:
        """I8: SWA command for this frame. 'phase' reads the phase after this step's lifecycle step."""
        m = self.cfg.swa_mode
        if m == 'on':
            return 1
        if m == 'phase':
            ph = self._lc.phase
            if ph == RUN:
                return 1
            # I8-1: STOP or an error in ARM goes straight to STOPPING; the motors stay off then
            return 1 if (ph == STOPPING and self._run_seen) else 0
        return 0

    def _lifecycle_step(self):
        """One phase step. Reads the owner cells (one try of the lock), no blocking call."""
        box = self.control
        box.rt_poll(self._lc.phase)
        now = time.monotonic_ns()
        out = self._lc.step(StepIn(t=now / 1e9, h7_reply_new=self._h7_new,
                                   publisher_done=box.pub_done, publisher_alive=not box.pub_dead,
                                   stop_requested=box.stop,
                                   heartbeat_age_s=(now - box.heartbeat_ns) / 1e9))
        if self._lc.phase == RUN:
            self._run_seen = True
        if out.done:
            self._stop_reason = self._lc.outcome
            self._end_seen = True                # I8-5: the loop stops after the tail (I8-R2-2)
        return out

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
            if self.cfg.imu_noise is not None:
                # replay row by frame time // report period, held over the report period;
                # the H7 takes the latest frame at each of its reports, on its own schedule
                # (fixed row shift against fwsim, IMN2-1: see imu_model.ImuNoise)
                imu = self.cfg.imu_noise.apply(imu, self._step * self.cfg.period_us)
        else:
            return None, 0
        # a new sample every frame: 1..255, skipping 0 (before its first
        # sample the H7 takes imu_seq 0 as "no IMU yet")
        self._imu_seq = self._imu_seq % 255 + 1
        return imu, self._imu_seq

    # ── Cleanup & diagnostics ─────────────────────────────────────────────

    def _save_frame_log(self) -> None:
        import platform
        import numpy as np
        n = self._step
        # Run manifest (S06 tools, Codex Sol round 3 S57/S64): what the Pi itself ran, saved with the frames
        # so the scoring tools can tell a cut or partial log and which IMU source fed the MOSI.
        # imu_source: the source _imu_sample really uses, in its order of priority (S73, Codex Sol round 4)
        imu_source = ('static' if self.cfg.imu_static is not None else
                      self.cfg.imu_profile if self.cfg.imu_profile == 't4' else
                      'plant' if self._mecanum else 'none')
        manifest = dict(step_first=0, step_last=n - 1, steps=n, period_us=self.cfg.period_us,
                        max_steps=self.cfg.max_steps, stop_reason=self._stop_reason,
                        imu_source=imu_source, imu_profile=self.cfg.imu_profile, imu_static=self.cfg.imu_static,
                        imu_noise=(None if self.cfg.imu_noise is None or imu_source != 'plant' else
                                   self.cfg.imu_noise.describe()),
                        plant=type(self.plant).__name__, plant_src=self._plant_src,     # L34
                        heavy_block=self.cfg.heavy_block,
                        load_schedule=self.cfg.load_schedule, rt=self._rt, uname=' '.join(platform.uname()),     # X8: the kernel the run ran on
                        lifecycle=self.lifecycle_summary(),
                        swa_mode=self.cfg.swa_mode, swa_events=self.swa_events(),     # I8
                        miso_check=self.miso_summary(),                                # A3
                        **self.swa_tail())                                             # I8-R2-3
        np.savez(self.cfg.frame_log, manifest=np.array(json.dumps(manifest)),
                 **{k: v[:n] for k, v in self._flog.items()})
        log.info('Frame log: %d frames -> %s', n, self.cfg.frame_log)

    def _on_signal(self, signum: int, _frame) -> None:
        log.info("Signal %d received — stopping plant loop.", signum)
        self._stop_reason = f'signal_{signum}'
        self._running = False

    def swa_events(self) -> list:
        """I8: [[step, 0/1], ...] at every change of the SWA bit sent (first frame included)."""
        return [[int(k), int(v)] for k, v in self._swa_events]

    def miso_summary(self) -> dict:
        """A3: MISO frames checked (from frame MISO_SKIP_FRAMES on), how many had a bad CRC or were all 0,
        and the step of the first of each (None: none). Meaningful only against the real H7."""
        return dict(checked=max(0, self._step - MISO_SKIP_FRAMES), crc_bad=self._miso_crc_bad,
                    empty=self._miso_empty, first_crc_bad=self._miso_first_crc_bad,
                    first_empty=self._miso_first_empty)

    def swa_tail(self) -> dict:
        """I8-R2-3: the END tail. swa_tail_sent = tail frames sent after the END frame; swa_tail_complete =
        it ran END_TAIL_FRAMES frames and END_TAIL_MS (False when max_steps, a full frame log, a signal or
        an error cut it, or END was never reached); None without --finite-run (no tail)."""
        return dict(swa_tail_sent=self._tail_sent,
                    swa_tail_complete=self._tail_complete if self._lc is not None else None)

    def lifecycle_summary(self) -> Optional[dict]:
        """Phases and outcome of a --finite-run, with the END tail (I8-R2-3); None without it."""
        if self._lc is None:
            return None
        return dict(self._lc.summary(), lock_misses=self.control.lock_misses, **self.swa_tail())

    @property
    def run_failed(self) -> bool:
        """--finite-run ended in error:<reason>, or without reaching END, or (I8-R2-3, Codex round 3)
        reached END but its tail was cut (max_steps, a full frame log, a signal), so run_hil.py exits 3."""
        if self._lc is None:
            return False
        return (self._lc.phase != END or (self._lc.outcome or '').startswith('error:')
                or not self._tail_complete)

    def _shutdown(self) -> None:
        if self._lc is not None:
            self.control.set_phase(self._lc.phase)
            log.info('#run_summary,%s', json.dumps(self.lifecycle_summary()))
        log.info('#swa,mode=%s,events=%s,tail=%s', self.cfg.swa_mode, json.dumps(self.swa_events()),
                 json.dumps(self.swa_tail()))                                           # I8, I8-R2-3
        m = self.miso_summary()                                                         # A3
        log.info('#miso,checked=%d,crc_bad=%d,empty=%d,first_crc_bad=%s,first_empty=%s',
                 m['checked'], m['crc_bad'], m['empty'], m['first_crc_bad'], m['first_empty'])
        if m['crc_bad'] or m['empty']:
            log.warning('MISO: %d frame CRC sai, %d frame rỗng trong lượt (L50: không làm số chính)',
                        m['crc_bad'], m['empty'])
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
