"""
s06_load_schedule.py — S06 load schedule of a T1/T2/T10 run (spec section 2.2: V7 and Z8).

One locked file, saved in the run folder, says what a run plays:
  * `test` rows: group A setpoints one after the other, contiguous from t = 0 to the end of the run
    (V7: order locked before the run). The trajectory publisher plays them, so the relay runs all along;
  * one `heavy` row (Z8): the 60 s reference block at the start of the scored part, after the 10 s T2
    leaves out. Inside it the frame sent to the H7 carries the omega of `heavy_omegas` instead of the
    plant wheel speeds. The plant still runs its full dynamics every 1 ms (no MockPlant, spec Z8); only
    the omega going down to the H7 changes.

[ĐỀ XUẤT CODE, spec chưa ghi] thứ tự bài = thứ tự GROUP_A, phần lệch pha PHASE_MS, ghi trong file lịch.

Usage (run_hil.py --load-schedule plays the file):
    python3 s06_load_schedule.py --total-s 610 --out S06_load.csv
    python3 s06_load_schedule.py --total-s 610 --no-heavy --skip A-K1,A-K2,A-K3 --out S02_cfg.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass, fields
from typing import List, Optional, Sequence, Tuple

from trajectory_pub import GROUP_A, TrajectorySpec

SKIP_S = 10.0                    # spec T2: the first 10 s of a run are not scored
HEAVY_S = 60.0                   # Z8
SWAP_MS = 50                     # Z8: omega changes every 50 ms
W_LOW, W_HIGH = 30.0, 39.2       # Z8, rad/s, same direction on the four wheels
PHASE_MS = (0, 13, 25, 38)       # [ĐỀ XUẤT CODE] Z8 "lệch pha": no two wheels change in the same frame


@dataclass
class Row:
    idx: int
    kind: str                    # 'test' | 'heavy'
    test: str                    # group A id, '' for heavy
    start_ms: int
    end_ms: int


def make_schedule(total_s: float, heavy_s: float = HEAVY_S, skip_s: float = SKIP_S,
                  order: Optional[Sequence[str]] = None) -> List[Row]:
    order = list(order) if order is not None else list(GROUP_A)
    if not order:
        raise ValueError('the test order is empty')
    unknown = [t for t in order if t not in GROUP_A]
    if unknown:
        raise ValueError(f'not group A tests: {unknown}')
    total_ms = int(round(total_s * 1000.0))
    if total_ms <= 0:
        raise ValueError('total_s must be > 0')
    rows, t, i = [], 0, 0
    while t < total_ms:
        test = order[i % len(order)]
        d = max(1, int(round(GROUP_A[test].duration * 1000.0)))
        rows.append(Row(0, 'test', test, t, min(t + d, total_ms)))
        t += d
        i += 1
    if heavy_s > 0:
        a, b = int(round(skip_s * 1000.0)), int(round((skip_s + heavy_s) * 1000.0))
        if b > total_ms:
            raise ValueError(f'heavy block ends at {b} ms, after the run ({total_ms} ms)')
        rows.append(Row(0, 'heavy', '', a, b))
    for k, r in enumerate(rows):
        r.idx = k
    return rows


def write_csv(path: str, rows: Sequence[Row]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(Row)])
        w.writeheader()
        for r in rows:
            w.writerow(vars(r))


def read_csv(path: str) -> List[Row]:
    out = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out.append(Row(int(row['idx']), row['kind'], row['test'], int(row['start_ms']), int(row['end_ms'])))
    return out


def file_sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check(rows: Sequence[Row]) -> None:
    """A file edited by hand must not play something else than it says: test rows contiguous from 0,
    known tests, at most one heavy block, inside the run. Raises ValueError."""
    tests = [r for r in rows if r.kind == 'test']
    heavy = [r for r in rows if r.kind == 'heavy']
    if not tests or len(tests) + len(heavy) != len(rows) or len(heavy) > 1:
        raise ValueError('schedule: need test rows, at most one heavy row, no other kind')
    t = 0
    for r in tests:
        if r.test not in GROUP_A or r.start_ms != t or r.end_ms <= r.start_ms:
            raise ValueError(f'schedule: bad test row {r}')
        t = r.end_ms
    if heavy and not 0 <= heavy[0].start_ms < heavy[0].end_ms <= t:
        raise ValueError(f'schedule: heavy block {heavy[0]} is outside the run')


def total_steps(rows: Sequence[Row]) -> int:
    """Steps of the run at 1 kHz."""
    return max(r.end_ms for r in rows)


def heavy_span(rows: Sequence[Row]) -> Optional[Tuple[int, int]]:
    """(first step, end step) of the heavy block, None without one."""
    for r in rows:
        if r.kind == 'heavy':
            return r.start_ms, r.end_ms
    return None


def trajectory_spec(rows: Sequence[Row]) -> TrajectorySpec:
    """The publisher plays the test rows; duration 0: it goes on sending setpoint 0 after the last row until
    the plant loop ends, so the relay does not stop before the run does."""
    return TrajectorySpec(kind='load', sequence=tuple((r.start_ms / 1000.0, r.end_ms / 1000.0, GROUP_A[r.test])
                                                      for r in rows if r.kind == 'test'))


def heavy_omegas(k: int) -> List[float]:
    """Omega of the 4 wheels, frame order, k steps after the start of the block: each wheel alternates
    W_LOW, W_HIGH every SWAP_MS, shifted by PHASE_MS."""
    return [W_HIGH if ((k + p) // SWAP_MS) & 1 else W_LOW for p in PHASE_MS]


class HeavyBlock:
    """Replaces the wheel speeds of the frame state inside the block (hil_node.py, one comparison per step
    outside it)."""

    def __init__(self, first: int, end: int):
        self.first, self.end = first, end

    def apply(self, step: int, frame_state: list) -> None:
        if self.first <= step < self.end:
            frame_state[0:4] = heavy_omegas(step - self.first)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--total-s', type=float, required=True, help='run length [s] (T2: 610 = 10 + 600 scored)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-heavy', action='store_true', help='no Z8 block (T2 run with the S02 configuration)')
    ap.add_argument('--skip', default='', help='group A tests left out, comma separated')
    a = ap.parse_args()
    skip = {t for t in a.skip.split(',') if t}
    rows = make_schedule(a.total_s, 0.0 if a.no_heavy else HEAVY_S, order=[t for t in GROUP_A if t not in skip])
    write_csv(a.out, rows)
    print(f'{sum(r.kind == "test" for r in rows)} test rows, {total_steps(rows)} steps, '
          f'heavy block {heavy_span(rows)} -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
