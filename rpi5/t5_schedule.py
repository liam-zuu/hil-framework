"""
t5_schedule.py — wheel-speed schedule of S06 T5 (H5, encoder frequency error; spec S06 v3.4, W15, Y16, Z1).

Points: omega in {0.32, 0.5, 1, 2, 10, 20, 39.2} rad/s, both directions, scored; 0.29 rad/s is the
indicator point (omega_d = 0.29054 <= 0.3, expect 0 counts, not scored). Pass 1 sweeps up, pass 2
sweeps down; at each omega the direction alternates (+ then -). W15: one visit per pass where the
4 wheels get different speeds, so a swapped wheel shows up (H4).

Each visit holds the speeds for VISIT_WINDOWS x 10 s. The scorer (tools/s06_t5_score.py) tiles the
visit into 10 s windows on the ESP32 clock and drops the first and the last (they hold the change
of speed and the alignment error), so 5 full windows are scored per visit.
CODE choices [ĐỀ XUẤT], not in the spec: 7 windows per visit (70 s), the order "+ then -", the W15
speeds (+1, -2, +10, -20 rad/s, all scored points, mixed signs) and the W15 visit at the end of
each pass. No randomness: the schedule is fixed, there is no seed.

    python t5_schedule.py --pass 1 --out T5_pass1.csv     # then run_hil.py --omega-steps @T5_pass1.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict, fields
from typing import List, Sequence

POINTS = (0.32, 0.5, 1.0, 2.0, 10.0, 20.0, 39.2)    # scored, rad/s (Z1, Q5)
INDICATOR = 0.29                                    # omega_d 0.29054 <= deadband 0.3: expect 0 counts
W15_OMEGAS = (1.0, -2.0, 10.0, -20.0)               # FL, FR, RL, RR [ĐỀ XUẤT]
WINDOW_S = 10.0
VISIT_WINDOWS = 7                                   # first and last dropped -> 5 scored [ĐỀ XUẤT]


@dataclass
class Visit:
    idx: int
    run_pass: int         # 1 (up, within 2 min of power-on) or 2 (down, after >= 30 min)
    visit: int            # 0.. within the pass
    kind: str             # point | indicator | w15
    w_fl: float
    w_fr: float
    w_rl: float
    w_rr: float
    t_start_ms: int       # from the first frame of the run (RPi5 clock)
    t_end_ms: int

    @property
    def omegas(self) -> List[float]:
        return [self.w_fl, self.w_fr, self.w_rl, self.w_rr]


def make_schedule(run_pass: int) -> List[Visit]:
    if run_pass not in (1, 2):
        raise ValueError('run_pass must be 1 or 2')
    mags = [INDICATOR] + list(POINTS)
    if run_pass == 2:
        mags.reverse()
    plan = []
    for m in mags:
        kind = 'indicator' if m == INDICATOR else 'point'
        plan += [(kind, [m] * 4), (kind, [-m] * 4)]
    plan.append(('w15', list(W15_OMEGAS)))
    hold_ms = int(round(VISIT_WINDOWS * WINDOW_S * 1000))
    out, t = [], 0
    for i, (kind, w) in enumerate(plan):
        out.append(Visit(i, run_pass, i, kind, *w, t_start_ms=t, t_end_ms=t + hold_ms))
        t += hold_ms
    return out


def segments(sched: Sequence[Visit]) -> list:
    """run_hil.py --omega-steps segments: (omegas, seconds, ramp=False), contiguous from 0."""
    t = 0
    segs = []
    for v in sched:
        if v.t_start_ms != t or v.t_end_ms <= v.t_start_ms:
            raise ValueError(f'T5 schedule row {v.idx}: visits must be contiguous from 0 ms')
        segs.append((v.omegas, (v.t_end_ms - v.t_start_ms) / 1000.0, False))
        t = v.t_end_ms
    return segs


def write_csv(path: str, sched: Sequence[Visit]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([fl.name for fl in fields(Visit)])
        for v in sched:
            w.writerow(list(asdict(v).values()))


def read_csv(path: str) -> List[Visit]:
    types = {fl.name: fl.type for fl in fields(Visit)}
    conv = {'int': int, 'float': float, 'str': str}
    out = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out.append(Visit(**{k: conv[types[k]](row[k]) for k in types}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='S06 T5 wheel-speed schedule')
    ap.add_argument('--pass', dest='run_pass', type=int, required=True, choices=[1, 2])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sched = make_schedule(a.run_pass)
    write_csv(a.out, sched)
    print(f'{a.out}: pass {a.run_pass}, {len(sched)} visits, {sched[-1].t_end_ms / 1000:.0f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
