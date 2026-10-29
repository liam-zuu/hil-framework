"""
Per-step timestamps of the 1 kHz plant loop for S06 T2 (H2), written into a
shared ring buffer by the loop and drained to CSV by a process outside it.

Spec S06 v3.4 T2 (V1, W7, Z9, Y16):
  t_sched  deadline of step k (PrecisionTimer), t_start  wake-up after it,
  t_done   after SPI transfer + plant.step + shared_state write,
  t_end    after every other job of the loop (monitor.record and this ring
           write, except the commit store), all time.monotonic_ns().
W7: every slot carries its step number; the writer counts steps it could not
store because the reader had not freed the slot (overflow, the run is not
usable when > 0); a slot is read only when committed, never half written.

Layout (one multiprocessing.RawArray of int64, no lock, one writer, one reader):
  header  [W_NEXT, R_NEXT, OVERFLOW]   next step to write / to read, overflow
  slot i  [COMMIT, STEP, T_SCHED, T_START, T_DONE, T_END]
The writer fills the slot, then stores COMMIT = step + 1 last; the reader takes
a slot only if COMMIT == STEP + 1 == expected step + 1, read before and after
the fields. Python gives no memory barrier between processes, so the reader
also stays READ_LAG steps behind W_NEXT (1 ms each): a slot is read at least
READ_LAG ms after its commit.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import List, Optional, Tuple

W_NEXT, R_NEXT, OVERFLOW = 0, 1, 2
N_HEAD = 3
C_COMMIT, C_STEP, C_SCHED, C_START, C_DONE, C_END = range(6)
SLOT = 6
DEFAULT_SLOTS = 16_384          # 16 s at 1 kHz; the reader drains every 0.2 s
READ_LAG = 2

CSV_HEADER = 'step,t_sched_us,t_start_us,t_done_us,t_end_us,miss'


class StepLog:
    """Shared ring of per-step timestamps. Create in the parent, pass to the plant process."""

    def __init__(self, n_slots: int = DEFAULT_SLOTS, arr=None):
        self.n = n_slots
        self.arr = arr if arr is not None else mp.RawArray('q', N_HEAD + n_slots * SLOT)

    def __getstate__(self):
        return {'n': self.n, 'arr': self.arr}

    def __setstate__(self, s):
        self.n, self.arr = s['n'], s['arr']

    # ── writer (plant loop) ──
    def record(self, step: int, t_sched: int, t_start: int, t_done: int) -> None:
        """Store step k; t_end is taken here, after the fields are written."""
        a = self.arr
        if step - a[R_NEXT] >= self.n:          # slot not read yet: drop, count
            a[OVERFLOW] += 1
            a[W_NEXT] = step + 1
            return
        b = N_HEAD + (step % self.n) * SLOT
        a[b + C_COMMIT] = 0
        a[b + C_STEP] = step
        a[b + C_SCHED] = t_sched
        a[b + C_START] = t_start
        a[b + C_DONE] = t_done
        a[b + C_END] = time.monotonic_ns()
        a[b + C_COMMIT] = step + 1
        a[W_NEXT] = step + 1

    # ── reader (outside the loop) ──
    def drain(self, final: bool = False) -> Tuple[List[Tuple[int, int, int, int, int]], int]:
        """Committed rows from R_NEXT on (up to READ_LAG behind the writer, or all
        when final), and the number of steps found missing or torn."""
        a = self.arr
        w = a[W_NEXT]
        r = a[R_NEXT]
        end = w if final else w - READ_LAG
        rows, lost = [], 0
        while r < end:
            b = N_HEAD + (r % self.n) * SLOT
            c1 = a[b + C_COMMIT]
            row = (a[b + C_STEP], a[b + C_SCHED], a[b + C_START], a[b + C_DONE], a[b + C_END])
            c2 = a[b + C_COMMIT]
            if c1 == c2 == r + 1 and row[0] == r:
                rows.append(row)
            else:
                lost += 1                         # overflowed (not stored) or torn
            r += 1
        a[R_NEXT] = r
        return rows, lost

    def counters(self) -> dict:
        a = self.arr
        return {'written_next': a[W_NEXT], 'read_next': a[R_NEXT], 'overflow': a[OVERFLOW]}


def format_row(row, period_ns: int = 1_000_000) -> str:
    step, ts, t0, td, te = row
    miss = 1 if td > ts + period_ns else 0
    return f'{step},{ts / 1e3:.3f},{t0 / 1e3:.3f},{td / 1e3:.3f},{te / 1e3:.3f},{miss}'


class StepLogCSV:
    """Reader side: drain the ring into a CSV file and keep the counts for the run meta."""

    def __init__(self, log: StepLog, path: str, period_ns: int = 1_000_000):
        self.log, self.period_ns = log, period_ns
        self.f = open(path, 'w', newline='\n')
        self.f.write(CSV_HEADER + '\n')
        self.rows = 0
        self.lost = 0
        self.first_step: Optional[int] = None

    def drain(self, final: bool = False) -> None:
        rows, lost = self.log.drain(final)
        self.lost += lost
        if rows and self.first_step is None:
            self.first_step = rows[0][0]
        for row in rows:
            self.f.write(format_row(row, self.period_ns) + '\n')
        self.rows += len(rows)

    def close(self) -> dict:
        self.drain(final=True)
        self.f.close()
        c = self.log.counters()
        info = time.get_clock_info('monotonic')
        return dict(rows=self.rows, lost_or_torn=self.lost, overflow=c['overflow'],
                    steps_written=c['written_next'], first_step=self.first_step,
                    monotonic_resolution_s=info.resolution, monotonic_impl=info.implementation,
                    ring_slots=self.log.n, read_lag_steps=READ_LAG)
