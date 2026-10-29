"""
t3b_schedule.py — S06 T3b wheel-speed schedule, its playback and its tolerances.

Spec S06 v3.4, T3b (W4, Y3, Y5, Z1, Z2, V2-6, V2-7, V3-6, W15):
  * transitions per wheel: step up x2 or more (2 -> 39.2, 5 -> 12.5), ramp
    0.4 -> 10 rad/s in 0.3 s, step down, reversal, stop, start from rest;
    each x 5 per wheel; exponential start-ups w_to*(1 - exp(-t/tau)) with
    tau in {0.05, 0.1, 0.2, 0.4} s x w_to in {10, 30} rad/s, 10 times each per
    wheel (80 per wheel);
  * every start from rest has an integer random delay in [0, 100] ms (seed);
  * the 4 wheels run at the same time, each in its own shuffled order
    (one wheel, one profile at a time: W15);
  * scoring window: 0.1 s before the transition start to 0.3 s after its end
    (exponential: end = 5 tau after the start); one transition per window.

Slot of one transition on one wheel (ms from the slot start S):
  S          step to the 'from' level (preparation, not scored)
  S+600      t0 = transition start (+ random delay for a start from rest)
  t_end      t0 (step), t0+300 (ramp), t0 + 5 tau (exponential)
  t_end+400  next slot
so the window [t0-100, t_end+300] holds only this transition.

The tolerance N_tol and the time-uncertainty limit of each window follow the
spec formulas on the omega actually framed (int16 with range 40 rad/s, as
protocol.encode_state), with omega_eff = 0 when |omega_d| <= 0.3 rad/s.

CLI: python t3b_schedule.py --seed 1 --out T3b_schedule.csv [--tol T3b_tolerance.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, asdict, fields
from typing import List, Sequence

OMEGA_RANGE = 40.0                 # rad/s, protocol.py / hil_protocol.h HIL_OMEGA_MAX
K_COUNT = 1320.0 / (2.0 * math.pi)  # counts per rad (330 lines x 4 edges)
DEADBAND = 0.3                     # rad/s: |omega_d| <= this -> omega_eff = 0 (Z1)
H5_LOW = 0.31983                   # rad/s: lowest omega of the H5 acceptance range
H5_TOL = 0.006                     # relative static frequency error allowed (H5)
D_MAX_S = 0.001                    # HIL one-way delay bound of H3 (i)

PREP_MS, POST_MS = 600, 400
WIN_BEFORE_MS, WIN_AFTER_MS = 100, 300
DELAY_MAX_MS = 100

# kind: (from, to, shape, duration_ms) -- shape step | ramp
KINDS = {
    'step_up_2_39.2':    (2.0, 39.2, 'step', 0),
    'step_up_5_12.5':    (5.0, 12.5, 'step', 0),
    'ramp_0.4_10':       (0.4, 10.0, 'ramp', 300),
    'step_down_39.2_2':  (39.2, 2.0, 'step', 0),
    'step_down_12.5_5':  (12.5, 5.0, 'step', 0),
    'reverse_10_-10':    (10.0, -10.0, 'step', 0),
    'stop_2_0':          (2.0, 0.0, 'step', 0),
    'start_0_5':         (0.0, 5.0, 'step', 0),
}
EXP_TAUS = (0.05, 0.1, 0.2, 0.4)
EXP_TARGETS = (10.0, 30.0)
REPS = 5
EXP_REPS = 10


@dataclass
class Transition:
    id: int
    wheel: int            # 0..3 (FL, FR, RL, RR as in the frame)
    kind: str
    w_from: float
    w_to: float
    shape: str            # step | ramp | exp
    tau_s: float
    delay_ms: int
    t_prep_ms: int
    t0_ms: int
    t_end_ms: int
    win_start_ms: int
    win_end_ms: int
    slot_end_ms: int


def quantize(w: float) -> float:
    """omega as it travels in the frame: protocol._to_int16 then back."""
    c = max(-OMEGA_RANGE, min(OMEGA_RANGE, w))
    return int(round(c / OMEGA_RANGE * 32767)) / 32767.0 * OMEGA_RANGE


def make_schedule(seed: int = 1, reps: int = REPS, exp_reps: int = EXP_REPS) -> List[Transition]:
    rng = random.Random(seed)
    out: List[Transition] = []
    for wheel in range(4):
        items = [(k,) for k in KINDS for _ in range(reps)]
        items += [('exp', tau, wt) for tau in EXP_TAUS for wt in EXP_TARGETS for _ in range(exp_reps)]
        rng.shuffle(items)                       # each wheel its own order (W15)
        t = 0
        for it in items:
            if it[0] == 'exp':
                _, tau, wt = it
                kind, w_from, w_to, shape, dur = f'exp_tau{tau}_to{wt:g}', 0.0, wt, 'exp', int(round(5 * tau * 1000))
            else:
                kind = it[0]
                w_from, w_to, shape, dur = KINDS[kind]
                tau = 0.0
            delay = rng.randint(0, DELAY_MAX_MS) if w_from == 0.0 else 0
            t0 = t + PREP_MS + delay
            t_end = t0 + dur
            out.append(Transition(0, wheel, kind, w_from, w_to, shape, tau, delay, t, t0, t_end,
                                  t0 - WIN_BEFORE_MS, t_end + WIN_AFTER_MS, t_end + POST_MS))
            t = t_end + POST_MS
    out.sort(key=lambda r: (r.t0_ms, r.wheel))
    for i, r in enumerate(out):
        r.id = i
    return out


def omega_table(sched: Sequence[Transition], n_ms: int = 0) -> List[List[float]]:
    """Commanded omega of each wheel at every ms (not quantized); 0 after the last slot."""
    n = n_ms or max(r.slot_end_ms for r in sched)
    w = [[0.0] * n for _ in range(4)]
    for r in sched:
        row = w[r.wheel]
        for k in range(r.t_prep_ms, min(r.slot_end_ms, n)):
            if k < r.t0_ms:
                v = r.w_from
            elif r.shape == 'ramp':
                f = min(1.0, (k - r.t0_ms) / max(1, r.t_end_ms - r.t0_ms))
                v = r.w_from + (r.w_to - r.w_from) * f
            elif r.shape == 'exp':
                v = r.w_to * (1.0 - math.exp(-(k - r.t0_ms) / 1000.0 / r.tau_s))
            else:
                v = r.w_to
            row[k] = v
    return w


def _events(eff: Sequence[float]):
    """Start events and stop intervals of one wheel over the whole run (V2-6,
    V3-6): stop event = omega_eff from non-zero to 0 or a sign change; start
    event = from 0 to non-zero or a sign change; standstill from k = 0 is a
    stop interval; a stop interval lasts until the next start event."""
    def sgn(v):
        return (v > 0) - (v < 0)
    stops = [0]                                  # standstill from init ...
    starts = [0] if eff and sgn(eff[0]) != 0 else []   # ... ends at once if frame 0 already moves
    for k in range(1, len(eff)):
        a, b = sgn(eff[k - 1]), sgn(eff[k])
        if a == b:
            continue
        if a != 0:
            stops.append(k)
        if b != 0:
            starts.append(k)
    intervals = []
    j = 0
    for s in stops:
        while j < len(starts) and starts[j] < s:
            j += 1
        intervals.append((s, starts[j] if j < len(starts) else len(eff)))
    return starts, intervals


def tolerance(sched_row: Transition, w_d: Sequence[float], ev=None) -> dict:
    """N_tol, its parts and the time-uncertainty limit of one window (spec T3b).

    w_d: omega of that wheel as framed, one value per ms from 0 (the whole
    run, so a stop interval that started before the window is seen).
    ev: _events() of that wheel, to reuse over many windows."""
    ws, we = sched_row.win_start_ms, sched_row.win_end_ms
    eff = [0.0 if abs(v) <= DEADBAND else v for v in w_d]
    s1 = sum(abs(eff[k]) for k in range(ws, we + 1) if abs(eff[k]) >= H5_LOW) * 0.001
    s2 = sum(abs(eff[k]) for k in range(ws, we + 1) if 0.0 < abs(eff[k]) < H5_LOW) * 0.001
    tv = sum(abs(eff[k] - eff[k - 1]) for k in range(ws + 1, we + 1))

    starts, intervals = ev if ev is not None else _events(eff)
    n_start = sum(1 for k in starts if ws <= k <= we)
    n_stop_iv = sum(1 for a, b in intervals if a <= we and b >= ws)   # [a, b] meets the window
    fixed = 1 + n_start + 2 * n_stop_iv
    n_tol = K_COUNT * (H5_TOL * s1 + 1.0 * s2 + D_MAX_S * tv) + fixed
    # u_t <= 1/4 (N_tol - fixed); u_t = K * sum_ends (read span + sync) * max|eff| near the end
    limit = 0.25 * (n_tol - fixed)
    m = [max(abs(eff[k]) for k in range(max(0, e - 1), min(len(eff), e + 2))) for e in (ws, we)]
    t_allow_ms = (limit / (K_COUNT * sum(m)) * 1000.0) if sum(m) > 0 else float('inf')
    return {'id': sched_row.id, 'wheel': sched_row.wheel, 'kind': sched_row.kind,
            'n_tol': n_tol, 'h5_part': K_COUNT * H5_TOL * s1, 'low_part': K_COUNT * s2,
            'delay_part': K_COUNT * D_MAX_S * tv, 'n_start': n_start, 'n_stop_intervals': n_stop_iv,
            'fixed': fixed, 'u_t_limit': limit, 't_allow_ms_per_end': t_allow_ms}


def tolerances(sched: Sequence[Transition]) -> List[dict]:
    table = omega_table(sched)
    wd = [[quantize(v) for v in row] for row in table]
    ev = [_events([0.0 if abs(v) <= DEADBAND else v for v in row]) for row in wd]
    return [tolerance(r, wd[r.wheel], ev[r.wheel]) for r in sched]


def write_csv(path: str, sched: Sequence[Transition]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([fl.name for fl in fields(Transition)])
        for r in sched:
            w.writerow(list(asdict(r).values()))


def read_csv(path: str) -> List[Transition]:
    types = {fl.name: fl.type for fl in fields(Transition)}
    out = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            kw = {}
            for k, v in row.items():
                t = types[k]
                kw[k] = int(v) if t in (int, 'int') else float(v) if t in (float, 'float') else v
            out.append(Transition(**kw))
    return out


class ScheduleOmegaPlant:
    """Plant stand-in for run_hil.py --omega-schedule: wheel speeds from the
    T3b schedule, one value per 1 ms step, frame order [w1..w4, 0 x 6]."""

    def __init__(self, sched: Sequence[Transition]):
        self._w = omega_table(sched)
        self._n = len(self._w[0])
        self._k = 0
        self._state = [0.0] * 10

    @property
    def total_steps(self) -> int:
        return self._n

    def reset(self, state=None) -> list:
        self._k = 0
        self._state = [0.0] * 10
        return list(self._state)

    def get_state(self) -> list:
        return list(self._state)

    def step(self, torques: list) -> list:
        k = self._k
        self._k += 1
        self._state[0:4] = [row[k] if k < self._n else 0.0 for row in self._w]
        return list(self._state)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tol', default=None, help='also write N_tol and u_t limit per window')
    a = ap.parse_args()
    sched = make_schedule(a.seed)
    write_csv(a.out, sched)
    n = max(r.slot_end_ms for r in sched)
    print(f'{len(sched)} transitions, {n / 1000:.1f} s, seed {a.seed} -> {a.out}')
    if a.tol:
        tol = tolerances(sched)
        with open(a.tol, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(tol[0]))
            w.writeheader()
            w.writerows(tol)
        by = {}
        for t in tol:
            k = t['kind']
            b = by.setdefault(k, [math.inf, math.inf])
            b[0] = min(b[0], t['n_tol'])
            b[1] = min(b[1], t['t_allow_ms_per_end'])
        print(f'{"kind":22s} {"min N_tol":>10s} {"min t_allow ms/end":>19s}')
        for k, (nt, ta) in sorted(by.items()):
            print(f'{k:22s} {nt:10.2f} {ta:19.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
