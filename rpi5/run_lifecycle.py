"""
Finite run lifecycle for one HIL run: PREPARE -> ARM -> RUN -> STOPPING -> END.

Stage B of the HIL workflow (03_CODE/docs/2026-10-28_Luong_du_lieu_HIL_2_che_do.md,
section 9.1; Liam approved writing it on the PC before G2, 29/10). RunLifecycle
is a pure state machine: no SPI, no files, no processes. The 1 kHz loop calls
step() once per real-time step with what it observed and does what the result
says. Wired into hil_node.py / run_hil.py behind --finite-run (Liam 02/11);
without that flag nothing here runs. ControlBox is the small shared block between
the run owner (run_hil.py), the 1 kHz loop and the publisher (section 9.1).

Rules taken from section 9.1 (numbers marked [ĐỀ XUẤT] there are defaults here,
to be measured on the HIL bench before a run is judged with them):
  * PREPARE: the owner makes the run folder, meta.yaml, SPI lock; begin() -> ARM,
    fail_prepare() -> END with no STOPPING (nothing has run).
  * ARM: plant held at its initial state (hold_plant), setpoint 0 with VALID|NEW
    every zero_every_steps (50 Hz at 1 kHz). Leaves ARM when the H7 has answered
    with new MISO frames continuously for arm_h7_fresh_s [ĐỀ XUẤT 1 s] (a gap
    longer than h7_gap_max_s restarts the count) and, if need_dut_ready, the
    DUT is READY. dut_ready is recorded 'yes' or 'unverified', never inferred.
  * RUN: publisher setpoints pass through (publish).
  * Any phase before STOPPING goes to STOPPING on: heartbeat older than
    heartbeat_max_s (error:heartbeat), publisher dead (error:publisher), STOP
    (user_stop), publisher DONE in RUN (completed), phase over its limit
    (error:arm_timeout / error:run_timeout). The FIRST reason is kept; later
    ones do not overwrite it (STOP has no clear). Order inside one step:
    heartbeat, publisher dead, STOP, DONE, timeouts.
  * STOPPING: the real-time loop itself sends setpoint 0 with VALID|NEW on the
    first STOPPING step and every zero_every_steps (send_zero), without the
    publisher; the step of the first one is recorded. Plant keeps running.
    After stop_s [ĐỀ XUẤT 1.0 s] -> END.
  * END: done. summary() gives outcome, the step of every phase change, the
    first zero step, dut_ready. Whether the DUT applied the zero is NOT known
    here (stop contract flags come from the DUT log, section 9.1).

Usage (one real-time step):
    lc = RunLifecycle(LifecycleConfig(run_s=30.0))
    lc.begin(t0)
    out = lc.step(StepIn(t=t, h7_reply_new=miso_seq_changed, publisher_done=done,
                         stop_requested=stop, heartbeat_age_s=age))
    if out.send_zero:   frame traj = (0, 0, 0), flags VALID|NEW
    elif out.publish:   frame traj = publisher setpoint
    if out.hold_plant:  do not integrate the plant
    if out.done:        leave the loop, write summary()
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional

PREPARE, ARM, RUN, STOPPING, END = 'PREPARE', 'ARM', 'RUN', 'STOPPING', 'END'
_PHASES = (PREPARE, ARM, RUN, STOPPING, END)

COMPLETED = 'completed'
USER_STOP = 'user_stop'
ERR_PREPARE = 'error:prepare'
ERR_HEARTBEAT = 'error:heartbeat'
ERR_PUBLISHER = 'error:publisher'
ERR_ARM_TIMEOUT = 'error:arm_timeout'
ERR_RUN_TIMEOUT = 'error:run_timeout'


@dataclass
class LifecycleConfig:
    run_s: float                      # RUN length of the profile
    arm_h7_fresh_s: float = 1.0       # [ĐỀ XUẤT] H7 answering continuously before RUN
    h7_gap_max_s: float = 0.010       # [ĐỀ XUẤT] longer gap restarts the ARM count
    arm_timeout_s: float = 30.0       # [ĐỀ XUẤT] ARM limit
    run_grace_s: float = 5.0          # [ĐỀ XUẤT] RUN limit = run_s + this
    stop_s: float = 1.0               # [ĐỀ XUẤT] t_stop: zero setpoints before END
    heartbeat_max_s: float = 2.0      # [ĐỀ XUẤT] owner heartbeat age limit
    zero_every_steps: int = 20        # 50 Hz at 1 kHz
    need_dut_ready: bool = False      # ARM also waits for DUT READY

    def __post_init__(self):
        if self.run_s <= 0:
            raise ValueError("run_s must be > 0")
        if self.zero_every_steps < 1:
            raise ValueError("zero_every_steps must be >= 1")
        for name in ('arm_h7_fresh_s', 'h7_gap_max_s', 'arm_timeout_s', 'run_grace_s',
                     'stop_s', 'heartbeat_max_s'):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass
class StepIn:
    t: float                          # owner's monotonic clock [s]
    h7_reply_new: bool = False        # MISO seq changed in this step's frame
    publisher_done: bool = False
    publisher_alive: bool = True
    stop_requested: bool = False
    heartbeat_age_s: float = 0.0
    dut_ready: Optional[bool] = None  # None = not known (no DUT log path)


@dataclass
class StepOut:
    phase: str                        # phase AFTER this step
    hold_plant: bool = False
    publish: bool = False
    send_zero: bool = False
    done: bool = False


@dataclass
class RunLifecycle:
    cfg: LifecycleConfig
    phase: str = PREPARE
    outcome: Optional[str] = None
    detail: Optional[str] = None
    steps: int = 0
    transitions: list = field(default_factory=list)   # (step, t, phase)
    first_zero_step: Optional[int] = None
    zeros_sent: int = 0
    dut_ready: str = 'unverified'
    _t_phase: float = 0.0
    _h7_streak_start: Optional[float] = None
    _h7_last: Optional[float] = None
    _stop_steps: int = 0

    # ── owner calls ────────────────────────────────────────────────────
    def begin(self, t: float) -> None:
        """PREPARE done (folder, meta.yaml, SPI lock): start ARM."""
        if self.phase != PREPARE:
            raise RuntimeError(f"begin() in phase {self.phase}")
        self._enter(ARM, t)

    def fail_prepare(self, t: float, detail: str) -> None:
        """PREPARE failed: nothing ran, so END without STOPPING."""
        if self.phase != PREPARE:
            raise RuntimeError(f"fail_prepare() in phase {self.phase}")
        self._latch(ERR_PREPARE, detail)
        self._enter(END, t)

    def step(self, s: StepIn) -> StepOut:
        if self.phase in (PREPARE, END):
            raise RuntimeError(f"step() in phase {self.phase}")
        n = self.steps            # index of this step; transitions in it are stamped n
        try:
            return self._step(s, n)
        finally:
            self.steps = n + 1

    def _step(self, s: StepIn, n: int) -> StepOut:
        if self.phase in (ARM, RUN):
            self._check_stop_reasons(s)
        if self.phase == ARM:
            self._arm(s)
        elif self.phase == RUN and s.t - self._t_phase > self.cfg.run_s + self.cfg.run_grace_s:
            self._to_stopping(ERR_RUN_TIMEOUT, None, s.t)

        out = StepOut(phase=self.phase)
        if self.phase == ARM:
            out.hold_plant = True
            out.send_zero = (n - self._phase_step()) % self.cfg.zero_every_steps == 0
        elif self.phase == RUN:
            out.publish = True
        elif self.phase == STOPPING:
            if s.t - self._t_phase >= self.cfg.stop_s and self._stop_steps > 0:
                self._enter(END, s.t)
                out.phase = END
                out.done = True
            else:
                out.send_zero = self._stop_steps % self.cfg.zero_every_steps == 0
                if out.send_zero and self.first_zero_step is None:
                    self.first_zero_step = n
                self._stop_steps += 1
        if out.send_zero:
            self.zeros_sent += 1
        return out

    def summary(self) -> dict:
        return {
            'outcome': self.outcome,
            'detail': self.detail,
            'steps': self.steps,
            'transitions': [{'step': n, 't': t, 'phase': p} for n, t, p in self.transitions],
            'first_zero_step': self.first_zero_step,
            'zeros_sent': self.zeros_sent,
            'dut_ready': self.dut_ready,
        }

    # ── internals ──────────────────────────────────────────────────────
    def _phase_step(self) -> int:
        return self.transitions[-1][0]

    def _enter(self, phase: str, t: float) -> None:
        self.phase = phase
        self._t_phase = t
        self.transitions.append((self.steps, t, phase))

    def _latch(self, outcome: str, detail: Optional[str]) -> None:
        if self.outcome is None:
            self.outcome, self.detail = outcome, detail

    def _to_stopping(self, outcome: str, detail: Optional[str], t: float) -> None:
        self._latch(outcome, detail)
        self._stop_steps = 0
        self._enter(STOPPING, t)

    def _check_stop_reasons(self, s: StepIn) -> None:
        if s.heartbeat_age_s > self.cfg.heartbeat_max_s:
            self._to_stopping(ERR_HEARTBEAT, f"age {s.heartbeat_age_s:.3f} s", s.t)
        elif not s.publisher_alive:
            self._to_stopping(ERR_PUBLISHER, None, s.t)
        elif s.stop_requested:
            self._to_stopping(USER_STOP, None, s.t)
        elif self.phase == RUN and s.publisher_done:
            self._to_stopping(COMPLETED, None, s.t)

    def _arm(self, s: StepIn) -> None:
        if self.phase != ARM:
            return
        if s.h7_reply_new:
            if self._h7_last is None or s.t - self._h7_last > self.cfg.h7_gap_max_s:
                self._h7_streak_start = s.t
            self._h7_last = s.t
        elif self._h7_last is not None and s.t - self._h7_last > self.cfg.h7_gap_max_s:
            self._h7_streak_start = None
        h7_ok = (self._h7_streak_start is not None
                 and s.t - self._h7_streak_start >= self.cfg.arm_h7_fresh_s)
        if s.dut_ready:
            self.dut_ready = 'yes'
        dut_ok = (not self.cfg.need_dut_ready) or bool(s.dut_ready)
        if h7_ok and dut_ok:
            self._enter(RUN, s.t)
        elif s.t - self._t_phase > self.cfg.arm_timeout_s:
            self._to_stopping(ERR_ARM_TIMEOUT, 'no_h7' if not h7_ok else 'dut_not_ready', s.t)


# Cells of the control box. Only the owner writes stop, heartbeat, done, dead (stop
# has no clear); only the 1 kHz loop writes phase (section 9.1, B21).
_STOP, _HEARTBEAT_NS, _PUB_DONE, _PUB_DEAD, _PHASE = range(5)


class ControlBox:
    """
    Shared cells of one run (section 9.1 "hop dieu khien"). The 1 kHz loop tries the
    lock once per step and keeps its last values when it misses, so a stale heartbeat
    still ages and a stuck lock ends in error:heartbeat. The publisher only reads the
    phase, without the lock.
    """

    def __init__(self):
        self._a = mp.Array('q', 5)
        self.stop = False                 # last values the 1 kHz loop read
        self.heartbeat_ns = 0
        self.pub_done = False
        self.pub_dead = False
        self.lock_misses = 0

    # owner
    def beat(self, now_ns: int) -> None:
        with self._a.get_lock():
            self._a[_HEARTBEAT_NS] = now_ns

    def request_stop(self) -> None:
        with self._a.get_lock():
            self._a[_STOP] = 1

    def publisher_done(self) -> None:
        with self._a.get_lock():
            self._a[_PUB_DONE] = 1

    def publisher_dead(self) -> None:
        with self._a.get_lock():
            self._a[_PUB_DEAD] = 1

    def stop_requested(self) -> bool:
        return bool(self._a.get_obj()[_STOP])

    # 1 kHz loop
    def rt_poll(self, phase: str) -> bool:
        """One try of the lock: publish phase, read the owner cells. False = missed."""
        lock = self._a.get_lock()
        if not lock.acquire(False):
            self.lock_misses += 1
            return False
        try:
            a = self._a.get_obj()
            a[_PHASE] = _PHASES.index(phase)
            self.stop = bool(a[_STOP])
            self.heartbeat_ns = a[_HEARTBEAT_NS]
            self.pub_done = bool(a[_PUB_DONE])
            self.pub_dead = bool(a[_PUB_DEAD])
        finally:
            lock.release()
        return True

    def set_phase(self, phase: str) -> None:
        """Outside the loop only: blocks on the lock."""
        with self._a.get_lock():
            self._a[_PHASE] = _PHASES.index(phase)

    # publisher
    def phase(self) -> str:
        return _PHASES[self._a.get_obj()[_PHASE]]

