"""
Precision Real-Time Timing — RPi5 HIL Plant Loop

Strategy: sleep + busy-wait hybrid
  - Phase 1: time.sleep() until BUSY_WAIT_NS before deadline
  - Phase 2: busy-wait (spin) for the final 200 μs
  Avoids wakeup jitter of pure sleep AND 100% CPU of pure busy-wait.

Real-time setup:
  - SCHED_FIFO via sched_setscheduler() syscall (requires root/CAP_SYS_NICE)
  - CPU affinity via os.sched_setaffinity()
  Both act on the calling thread (pid 0 on Linux), i.e. the process's main thread when
  called from it.

Jitter on RPi5 with SCHED_FIFO: not measured with this code yet (19/10 setup log: max
0.5 μs from the terminal, no saved file). The numbers once written here (mean 10–30 μs,
max 50–150 μs) had no measurement behind them (B0 review round 3, question 5).

Usage:
    from timing import set_realtime, pin_cpu, PrecisionTimer

    set_realtime(priority=90)
    pin_cpu(core=0)
    timer = PrecisionTimer(period_us=1000)
    while True:
        jitter_ns = timer.wait_next_tick()
        # ... do plant work here (must finish < 1ms) ...

Standalone jitter test:
    python timing.py [n_ticks]
"""
from __future__ import annotations

import collections
import ctypes
import os
import time
import sys
from typing import Optional


# ── Linux scheduling constants ────────────────────────────────────────────
_SCHED_FIFO  = 1
_SCHED_OTHER = 0

class _sched_param(ctypes.Structure):
    _fields_ = [('sched_priority', ctypes.c_int)]

try:
    _libc      = ctypes.CDLL('libc.so.6', use_errno=True)
    _HAS_LIBC  = True
except OSError:
    _libc      = None
    _HAS_LIBC  = False

# Busy-wait tail: spin for this many ns before the deadline
BUSY_WAIT_NS: int = 200_000   # 200 μs


# ── Real-time scheduling ──────────────────────────────────────────────────

def set_realtime(priority: int = 90) -> bool:
    """
    Set SCHED_FIFO real-time scheduling for the calling process.

    Requires root or CAP_SYS_NICE capability:
        sudo setcap cap_sys_nice+ep $(which python3)
    Or simply run with sudo.

    Args:
        priority: FIFO priority [1..99].  90 leaves room for kernel threads.

    Returns:
        True on success; False if running without privileges (logs a warning).
    """
    if not _HAS_LIBC:
        print("[timing] libc unavailable — SCHED_FIFO skipped", flush=True)
        return False

    priority = int(max(1, min(99, priority)))
    param = _sched_param(priority)
    ret   = _libc.sched_setscheduler(0, _SCHED_FIFO, ctypes.byref(param))
    if ret != 0:
        errno = ctypes.get_errno()
        print(
            f"[timing] sched_setscheduler FAILED (errno={errno}). "
            f"Run as root or: sudo setcap cap_sys_nice+ep $(which python3)",
            flush=True,
        )
        return False

    print(f"[timing] SCHED_FIFO priority={priority} OK", flush=True)
    return True


def restore_normal_scheduling() -> None:
    """Restore SCHED_OTHER (default Linux scheduling)."""
    if not _HAS_LIBC:
        return
    param = _sched_param(0)
    _libc.sched_setscheduler(0, _SCHED_OTHER, ctypes.byref(param))


def pin_cpu(core: int) -> bool:
    """
    Pin the calling process to a single CPU core.

    RPi5 has cores 0-3.
      core 0 → plant loop  (SCHED_FIFO)
      core 2 → trajectory publisher (normal)
      cores 1, 3 → OS / kernel threads

    Args:
        core: 0-based CPU index.

    Returns:
        True on success.
    """
    try:
        os.sched_setaffinity(0, {core})
        print(f"[timing] Process pinned to CPU core {core}", flush=True)
        return True
    except (AttributeError, OSError) as exc:
        print(f"[timing] CPU pin unavailable: {exc}", flush=True)
        return False


# ── Precision timer ───────────────────────────────────────────────────────

class PrecisionTimer:
    """
    Fixed-rate timer using sleep + busy-wait hybrid.

    All time values are in nanoseconds internally (time.monotonic_ns).
    External API exposes microseconds for readability.

    Attributes:
        period_ns:   Nominal period [ns].
        overruns:    Number of ticks where actual time exceeded one full period.
    """

    def __init__(self, period_us: int = 1_000, history_len: int = 5_000,
                 clock=time.monotonic_ns, sleep=time.sleep):
        """
        Args:
            period_us:   Tick period [μs]. 1000 = 1 kHz.
            history_len: Size of the rolling jitter ring buffer.
            clock, sleep: monotonic ns clock and sleep(seconds); tests pass fakes to check
                         deadlines on any OS (B0 review round 3 N18).

        Fixed rate: each deadline is the previous one plus the period, never "now + period",
        so a late tick does not shift later ones; after an overrun the following ticks
        return at once until the timer has caught up.
        """
        self._clock = clock
        self._sleep = sleep
        self.period_ns:  int = period_us * 1_000
        self._next_ns:   int = clock() + self.period_ns
        self._jitter:    collections.deque = collections.deque(maxlen=history_len)
        self._overruns:  int = 0

    # ── Core method ───────────────────────────────────────────────────────

    def wait_next_tick(self) -> int:
        """
        Block until the next scheduled tick deadline.

        Returns:
            Jitter in nanoseconds.  Positive = we woke up late.
        """
        deadline_ns   = self._next_ns
        busy_start_ns = deadline_ns - BUSY_WAIT_NS

        # Phase 1: coarse sleep (OS can preempt here)
        now_ns = self._clock()
        if now_ns < busy_start_ns:
            self._sleep((busy_start_ns - now_ns) * 1e-9)

        # Phase 2: busy-wait — burns CPU but gives precise wake
        while self._clock() < deadline_ns:
            pass

        actual_ns  = self._clock()
        jitter_ns  = actual_ns - deadline_ns
        self._next_ns += self.period_ns

        if jitter_ns > self.period_ns:
            self._overruns += 1

        self._jitter.append(jitter_ns)
        return jitter_ns

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Re-arm timer from now. Clears jitter history."""
        self._next_ns  = self._clock() + self.period_ns
        self._jitter.clear()
        self._overruns = 0

    # ── Statistics ────────────────────────────────────────────────────────

    @property
    def jitter_stats(self) -> dict:
        """
        Return jitter statistics dict over the history buffer.

        Keys: n, mean_us, max_us, p99_us, p999_us (if n≥1000), overruns.
        Returns empty dict if no data yet.
        """
        if not self._jitter:
            return {}
        data = list(self._jitter)
        n    = len(data)
        arr  = sorted(data)
        return {
            'n':        n,
            'mean_us':  sum(data) / n / 1_000.0,
            'max_us':   arr[-1]   / 1_000.0,
            'p99_us':   arr[min(n - 1, int(0.99 * n))] / 1_000.0,
            'p999_us':  arr[min(n - 1, int(0.999 * n))] / 1_000.0 if n >= 1_000 else None,
            'overruns': self._overruns,
        }

    def print_jitter_report(self) -> None:
        """Human-readable jitter summary to stdout."""
        s = self.jitter_stats
        if not s:
            print("[timing] No jitter data yet.", flush=True)
            return
        print(f"[timing] Jitter report (n={s['n']}):", flush=True)
        print(f"  mean   = {s['mean_us']:7.1f} μs")
        print(f"  max    = {s['max_us']:7.1f} μs")
        print(f"  p99    = {s['p99_us']:7.1f} μs")
        if s['p999_us'] is not None:
            print(f"  p99.9  = {s['p999_us']:7.1f} μs")
        print(f"  overruns = {s['overruns']}")
        flush = True

    @property
    def overruns(self) -> int:
        return self._overruns


# ── Standalone jitter measurement ─────────────────────────────────────────

def _save_sched() -> dict:
    """Scheduler (policy, priority) and CPU affinity of the calling thread; None for the part
    the OS cannot report (then that part is not changed, B0 review round 3 N17)."""
    def _try(fn):
        try:
            return fn()
        except (AttributeError, OSError):
            return None
    policy = _try(lambda: os.sched_getscheduler(0))
    prio = _try(lambda: os.sched_getparam(0).sched_priority)
    return dict(scheduler=None if policy is None or prio is None else (policy, prio),
                affinity=_try(lambda: os.sched_getaffinity(0)))


def _restore_sched(saved: dict, tried: dict) -> list:
    """Put back each part whose change was tried (also when the helper raised after changing
    it, round 4 V8), on its own (a failed scheduler restore does not skip the affinity one),
    and read it back. Returns the errors, empty if all went back."""
    errors = []
    if tried.get('scheduler'):
        policy, prio = saved['scheduler']
        try:
            os.sched_setscheduler(0, policy, os.sched_param(prio))
            back = (os.sched_getscheduler(0), os.sched_getparam(0).sched_priority)
            if back != (policy, prio):
                errors.append(f'scheduler: {back} after restore, expected {(policy, prio)}')
        except (AttributeError, OSError) as exc:
            errors.append(f'scheduler: {exc}')
    if tried.get('affinity'):
        try:
            os.sched_setaffinity(0, saved['affinity'])
            back = os.sched_getaffinity(0)
            if back != saved['affinity']:
                errors.append(f'affinity: {sorted(back)} after restore, expected {sorted(saved["affinity"])}')
        except (AttributeError, OSError) as exc:
            errors.append(f'affinity: {exc}')
    for e in errors:
        print(f"[timing] could not restore {e}", flush=True)
    return errors


def measure_jitter(
    n_ticks:   int = 2_000,
    period_us: int = 1_000,
    rt:        bool = True,
) -> dict:
    """
    Run the timer in isolation and report jitter statistics.

    Run this BEFORE connecting any hardware to verify RPi5 timing budget:
        python timing.py 2000
    Budget: max jitter < 500 μs (half the 1 ms period); the jitter on the Pi has not been
    measured with this code yet.

    Args:
        n_ticks:   Number of ticks to collect.
        period_us: Timer period [μs].
        rt:        If True, attempt to set SCHED_FIFO and pin to core 0 first.

    Returns:
        Jitter statistics dict (same keys as PrecisionTimer.jitter_stats) plus
          'rt'             {'scheduler': bool, 'affinity': bool}: what was really applied
                           (False: could not read the old value, or no privilege)
          'restore_errors' list of parts that could not be put back (empty = all back)

    With rt=True the calling thread's scheduler and CPU affinity are changed for the
    measurement only (B0 review B41: a pytest run on the Pi must not leave the test process
    on SCHED_FIFO pinned to core 0). A part whose old value cannot be read is not changed;
    each part whose change was tried is restored on its own, also when the measurement or
    the setup helper raises (round 3 N17, round 4 V8).
    """
    saved = _save_sched() if rt else None
    applied = {'scheduler': False, 'affinity': False}     # what the helpers report as done
    tried = {'scheduler': False, 'affinity': False}       # restored even if a helper raised (V8)
    errors = []
    try:
        if rt:
            if saved['scheduler'] is not None:
                tried['scheduler'] = True
                applied['scheduler'] = bool(set_realtime(priority=90))
            else:
                print("[timing] scheduler not readable: SCHED_FIFO not set", flush=True)
            if saved['affinity'] is not None:
                tried['affinity'] = True
                applied['affinity'] = bool(pin_cpu(core=0))
            else:
                print("[timing] affinity not readable: CPU pin not set", flush=True)

        hz = 1_000_000 // period_us
        print(f"[timing] Measuring jitter: {n_ticks} ticks @ {period_us} μs ({hz} Hz) …",
              flush=True)

        timer = PrecisionTimer(period_us=period_us, history_len=n_ticks + 100)
        for _ in range(n_ticks):
            timer.wait_next_tick()
    finally:
        if saved is not None:
            errors = _restore_sched(saved, tried)

    timer.print_jitter_report()
    s = timer.jitter_stats
    s['rt'] = dict(applied)
    s['restore_errors'] = errors

    # HIL budget check: max jitter should be < 500 μs (50% of 1ms period)
    budget_ok = s.get('max_us', 9999) < 500
    print(f"[timing] Budget check (max < 500 μs): {'PASS ✓' if budget_ok else 'FAIL ✗'}"
          f"  rt={s['rt']}  restore_errors={errors or 'none'}", flush=True)
    return s


def jitter_ok(stats: dict) -> bool:
    """Exit criterion of --jitter-only / this script: within budget AND everything restored."""
    return stats.get('max_us', 9999) < 500 and not stats.get('restore_errors')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000
    stats = measure_jitter(n_ticks=n, period_us=1_000, rt=True)
    sys.exit(0 if jitter_ok(stats) else 1)
