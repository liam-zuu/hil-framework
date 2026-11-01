"""
Plant-loop statistics without work inside the 1 kHz loop (THESIS H2, 28/10).

Before: every stats_interval steps the plant loop itself called
PrecisionTimer.jitter_stats (sorted() over 10 000 samples) and log.info, and
logged a warning on each late tick. On the Pi the tick right after that print
started 641-860 us late (H7_LINK/2026-10-27_run01, H7_CLOCK/2026-10-27_run02).

Now the loop only writes a few integers per tick into a shared array (single
writer, no lock): step count, timer overruns, all-time max lateness, and a
1 us histogram of lateness. A process outside the real-time loop (run_hil.py's
main process) reads it and prints the stats; percentiles of a window come from
the histogram difference between two reads.

"Lateness" is what PrecisionTimer.wait_next_tick returns: how far after the
deadline the busy-wait exits (not SPI + plant time).
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Optional

N_BINS = 2_000                 # 1 us bins 0..1999 us; the last bin also holds >= 1999 us

# Header slots of the shared array
I_STEP, I_OVERRUNS, I_MAX_NS, I_WARN = 0, 1, 2, 3
N_HEAD = 4


class RTMonitor:
    """Shared lateness counters. Create in the parent, pass to the plant process."""

    def __init__(self, warn_us: float = 500.0, arr=None):
        self.warn_ns = int(warn_us * 1_000)
        # 'q' = int64; RawArray: no lock, the plant loop is the only writer
        self.arr = arr if arr is not None else mp.RawArray('q', N_HEAD + N_BINS)

    def __getstate__(self):
        return {'warn_ns': self.warn_ns, 'arr': self.arr}

    def __setstate__(self, s):
        self.warn_ns, self.arr = s['warn_ns'], s['arr']

    # ── writer: plant loop, once per tick ─────────────────────────────────
    def record(self, step: int, late_ns: int, overruns: int) -> None:
        a = self.arr
        a[I_STEP] = step
        a[I_OVERRUNS] = overruns
        if late_ns > a[I_MAX_NS]:
            a[I_MAX_NS] = late_ns
        if late_ns > self.warn_ns:
            a[I_WARN] += 1
        b = late_ns // 1_000
        a[N_HEAD + (b if b < N_BINS else N_BINS - 1)] += 1

    # ── reader: any other process ─────────────────────────────────────────
    def snapshot(self) -> list:
        return list(self.arr)

    @staticmethod
    def window_stats(new: list, old: Optional[list] = None) -> dict:
        """Stats of the ticks between two snapshots (old None: since start)."""
        hist = [n - (old[N_HEAD + i] if old else 0) for i, n in enumerate(new[N_HEAD:])]
        n = sum(hist)
        out = dict(step=new[I_STEP], n=n, overruns=new[I_OVERRUNS],
                   warn=new[I_WARN] - (old[I_WARN] if old else 0),
                   max_all_us=new[I_MAX_NS] / 1_000)
        if n == 0:
            return out
        acc, p99, p999 = 0, None, None
        for i, c in enumerate(hist):
            acc += c
            if p99 is None and acc >= 0.99 * n:
                p99 = i
            if p999 is None and acc >= 0.999 * n:
                p999 = i
        top = max(i for i, c in enumerate(hist) if c)
        out.update(p99_us=p99, p999_us=p999, max_us=top,
                   max_over_scale=top == N_BINS - 1)
        return out
