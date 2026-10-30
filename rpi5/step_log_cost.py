"""
Cost of one StepLog.record() call on the Pi, spec S06 X7 / C1-12 (§4.1, T2): measured directly, in a run of
its own, not from the start latency of two runs (the start latency does not see what happens after the start).

Each call of record() is timed with time.monotonic_ns() on the core and priority of the plant loop, paced at
1 kHz like the loop, with the reader kept up (not timed) so every call takes the normal path, not the overflow
one. Reports median, p99.9 and max in µs, and the cost of the pair of clock reads around the call (included in
the figures, not subtracted). Not measured: contention with the reader process of a real run.

    sudo python3 step_log_cost.py --n 60000 --out step_log_cost.json
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time

import numpy as np

from step_log import R_NEXT, StepLog
from timing import PrecisionTimer, pin_cpu, set_realtime


def measure(n: int, pace: bool = True, period_us: int = 1_000) -> dict:
    log = StepLog()
    cost, pair = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    cost.fill(0)
    pair.fill(0)
    timer = PrecisionTimer(period_us=period_us) if pace else None
    clock = time.monotonic_ns
    arr = log.arr
    for k in range(n):
        if timer is not None:
            timer.wait_next_tick()
        arr[R_NEXT] = k                         # the reader has taken everything: no overflow path
        t = clock()
        log.record(k, t, t, t)
        t1 = clock()
        t2 = clock()
        cost[k], pair[k] = t1 - t, t2 - t1
    q = lambda x, p: float(np.percentile(x, p)) / 1e3  # noqa: E731
    return dict(n=n, paced=pace, median_us=q(cost, 50), p999_us=q(cost, 99.9), max_us=float(cost.max()) / 1e3,
                clock_pair_median_us=q(pair, 50), overflow=log.counters()['overflow'],
                uname=' '.join(platform.uname()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Thời gian mỗi lần gọi StepLog.record() (S06 X7)')
    ap.add_argument('--n', type=int, default=60_000)
    ap.add_argument('--rt-priority', type=int, default=90)
    ap.add_argument('--cpu-core', type=int, default=0)
    ap.add_argument('--no-pace', action='store_true', help='không giữ nhịp 1 kHz (chỉ để thử nhanh)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    rt = dict(sched_fifo=set_realtime(priority=a.rt_priority), cpu_pinned=pin_cpu(core=a.cpu_core))
    gc.collect()
    gc.disable()
    try:
        res = measure(a.n, pace=not a.no_pace)
    finally:
        gc.enable()
    res['rt'] = rt
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    return 0


if __name__ == '__main__':
    sys.exit(main())
