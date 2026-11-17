"""
t3_schedule.py — S06 T3 (i), (ii) wheel-speed step schedule, its checks and the H7 half period.

Spec S06 T3 (V2, Z7, L7-10): 100 steps generated beforehand with a fixed seed and saved as
T3_schedule.csv in the run folder; each step changes the speed of ONE wheel, the wheels in turn
(step j changes wheel j mod 4), 25 steps per wheel; the steps of each wheel alternate up and down
(>= 12 up, >= 12 down) and include a step 2 -> 39.2 rad/s; omega covers 2 to 39.2 rad/s; two
consecutive steps are >= 50 ms apart, so the matching window of a step
[mark j; mark j + 2 hp_old + 1 ms] (largest 10.5 ms at 2 rad/s) holds no event of another step.
The other three wheels keep their speed (no retime, so no H7 debug edge outside a window).
Design: 03_CODE/docs/2026-11-15_Thiet_ke_S06_T3_danh_dau_v1.md section 4 item 2 (Liam 15/11
"Duyệt, code luôn").

Layout (CODE choices [ĐỀ XUẤT], not in the spec):
  * every wheel spins the whole run, same direction (+), so each step is an EncoderGen retime
    (same direction, new speed; encoder_gen.c EncoderGen_SetOmega), the only path that toggles PF3;
  * lead-in LEAD_MS at the first level (2 rad/s: the H7 starts the wheels, no step, no marker);
  * gaps between steps: seeded integers in [GAP_MIN_MS, GAP_MAX_MS] (phase of the step against the
    encoder edges not locked to one value);
  * levels: each wheel starts at 2.0, its first step is 2 -> 39.2, then down to a level in
    [2, cur / STEP_MIN_RATIO], up to [cur * STEP_MIN_RATIO, 39.2], ...; rounded to 0.1 rad/s;
  * tail TAIL_MS at the last levels after the last step, then the schedule ends (run_hil sends 0:
    a stop, no marker).

The file has the columns of t3b_schedule.Transition, one row per step, so
`run_hil.py --omega-schedule T3_schedule.csv --marker t3` plays it with the existing playback
(t3b_schedule.ScheduleOmegaPlant): the row of a step covers [t_prep_ms, slot_end_ms) of its wheel
(from the previous step of that wheel, or 0, to its next step, or the end), w_from before t0_ms,
w_to from t0_ms. kind = 't3_up' / 't3_down', shape 'step'. win_start_ms / win_end_ms: the nominal
window on the schedule clock (scoring uses the marker edge on the ESP32 clock, s06_t3_score.py).

hp_ticks() mirrors the H7: encoder_gen.c _half_period (float32, ENC_TICKS_PER_S = 1e6 x
ENC_CLOCK_CAL 1.0, ENC_PPR 330, + 0.5 then truncate, 1..65535) on the omega the H7 decodes
(spi1_handler.c: int16 * (40 / 32767) in float32). check() requires every step to change hp by
>= HP_MIN_CHANGE ticks, so a float rounding difference between Pi and H7 cannot hide a retime.

CLI: python rpi5/t3_schedule.py --seed 1 --out T3_schedule.csv
     (writes T3_schedule_check.json next to it; exit 2 if a check fails)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import List, Sequence

import numpy as np

from t3b_schedule import Transition, write_csv, read_csv, omega_table   # noqa: F401 (read_csv re-exported)

N_STEPS = 100
N_WHEELS = 4
W_LOW, W_HIGH = 2.0, 39.2           # rad/s, spec T3
LEAD_MS = 1000
TAIL_MS = 500
GAP_MIN_MS, GAP_MAX_MS = 60, 120    # >= 50 ms (spec L7-10)
GAP_SPEC_MS = 50
STEP_MIN_RATIO = 1.15               # each step changes omega by at least 15 %
HP_MIN_CHANGE = 2                   # ticks (1 us)
OMEGA_RANGE = 40.0                  # HIL_OMEGA_MAX
ENC_OMEGA_MIN = 0.3                 # encoder_gen.c
ENC_TICKS_PER_S = 1_000_000.0       # ENC_TIMER_FREQ x ENC_CLOCK_CAL (1.0)
ENC_PPR = 330
LOCKED_SEED = 1                     # [ĐỀ XUẤT CODE_S06]: seed of the acceptance schedule


def frame_int16(w: float) -> int:
    """protocol._to_int16 of the wheel speed."""
    c = max(-OMEGA_RANGE, min(OMEGA_RANGE, float(w)))
    return int(round(c / OMEGA_RANGE * 32767))


def h7_omega(raw: int) -> float:
    """The omega the H7 decodes from the int16 (float32 arithmetic of spi1_handler.c)."""
    return float(np.float32(raw) * (np.float32(OMEGA_RANGE) / np.float32(32767.0)))


def hp_ticks(w: float) -> int:
    """EncoderGen half period in timer ticks (1 us) for the frame speed w, as the H7 computes it."""
    a = abs(np.float32(h7_omega(frame_int16(w))))
    if a <= ENC_OMEGA_MIN:
        return 0
    hp = np.float32(ENC_TICKS_PER_S) * np.float32(math.pi) / (a * np.float32(ENC_PPR))
    if hp > 65535.0:
        return 65535
    if hp < 1.0:
        return 1
    return int(np.float32(hp + np.float32(0.5)))


def window_ms(w_from: float) -> float:
    """Width of the matching window of a step, ms: 2 hp_old + 1 ms (spec L7-10)."""
    return 2.0 * hp_ticks(w_from) / 1000.0 + 1.0


def _levels(rng: random.Random, n: int) -> List[float]:
    """n + 1 levels of one wheel: 2.0, 39.2, then alternately down and up."""
    lv = [W_LOW, W_HIGH]
    while len(lv) < n + 1:
        cur = lv[-1]
        if len(lv) % 2 == 0:       # down
            hi = cur / STEP_MIN_RATIO
            v = rng.uniform(W_LOW, hi)
        else:                      # up
            lo = cur * STEP_MIN_RATIO
            v = rng.uniform(lo, W_HIGH)
        v = min(W_HIGH, max(W_LOW, round(v, 1)))
        lv.append(v)
    return lv


def make_schedule(seed: int = LOCKED_SEED) -> List[Transition]:
    rng = random.Random(seed)
    per = N_STEPS // N_WHEELS
    levels = [_levels(rng, per) for _ in range(N_WHEELS)]
    t_steps, t = [], LEAD_MS
    for j in range(N_STEPS):
        if j:
            t += rng.randint(GAP_MIN_MS, GAP_MAX_MS)
        t_steps.append(t)
    t_stop = t_steps[-1] + TAIL_MS
    rows: List[Transition] = []
    for j in range(N_STEPS):
        w = j % N_WHEELS
        k = j // N_WHEELS                      # k-th step of wheel w
        w_from, w_to = levels[w][k], levels[w][k + 1]
        t0 = t_steps[j]
        t_prep = 0 if k == 0 else t_steps[j - N_WHEELS]
        nxt = j + N_WHEELS
        slot_end = t_steps[nxt] if nxt < N_STEPS else t_stop
        rows.append(Transition(
            id=j, wheel=w, kind='t3_up' if w_to > w_from else 't3_down', w_from=w_from, w_to=w_to,
            shape='step', tau_s=0.0, delay_ms=0, t_prep_ms=t_prep, t0_ms=t0, t_end_ms=t0,
            win_start_ms=t0, win_end_ms=t0 + int(math.ceil(window_ms(w_from))), slot_end_ms=slot_end))
    return rows


def check(sched: Sequence[Transition]) -> List[str]:
    """Spec conditions of the T3 schedule; [] = all hold."""
    bad: List[str] = []
    if len(sched) != N_STEPS:
        bad.append(f'{len(sched)} bậc, cần {N_STEPS}')
    rows = sorted(sched, key=lambda r: r.t0_ms)
    for j, r in enumerate(rows):
        if r.wheel != j % N_WHEELS:
            bad.append(f'bậc {j}: bánh {r.wheel}, cần {j % N_WHEELS} (xoay vòng)')
        if not (W_LOW <= r.w_from <= W_HIGH and W_LOW <= r.w_to <= W_HIGH):
            bad.append(f'bậc {j}: ω ngoài [{W_LOW}; {W_HIGH}]')
        if hp_ticks(r.w_from) == 0 or hp_ticks(r.w_to) == 0:
            bad.append(f'bậc {j}: ω dưới ngưỡng dừng của H7')
        elif abs(hp_ticks(r.w_to) - hp_ticks(r.w_from)) < HP_MIN_CHANGE:
            bad.append(f'bậc {j}: hp đổi < {HP_MIN_CHANGE} tick (H7 có thể không hẹn lại)')
        if frame_int16(r.w_to) == frame_int16(r.w_from):
            bad.append(f'bậc {j}: int16 của ω không đổi (Pi không nâng chân đánh dấu)')
        if j and r.t0_ms - rows[j - 1].t0_ms < GAP_SPEC_MS:
            bad.append(f'bậc {j}: cách bậc trước {r.t0_ms - rows[j - 1].t0_ms} ms < {GAP_SPEC_MS} ms')
        if j + 1 < len(rows) and window_ms(r.w_from) >= rows[j + 1].t0_ms - r.t0_ms:
            bad.append(f'bậc {j}: cửa sổ {window_ms(r.w_from):.2f} ms chạm bậc sau')
    for w in range(N_WHEELS):
        mine = [r for r in rows if r.wheel == w]
        up = sum(1 for r in mine if r.w_to > r.w_from)
        down = sum(1 for r in mine if r.w_to < r.w_from)
        if len(mine) != N_STEPS // N_WHEELS:
            bad.append(f'bánh {w}: {len(mine)} bậc, cần {N_STEPS // N_WHEELS}')
        if up < 12 or down < 12:
            bad.append(f'bánh {w}: {up} tăng, {down} giảm (cần ≥ 12 mỗi loại)')
        if not any(r.w_from == W_LOW and r.w_to == W_HIGH for r in mine):
            bad.append(f'bánh {w}: không có bậc {W_LOW} → {W_HIGH}')
        if any((a.w_to > a.w_from) == (b.w_to > b.w_from) for a, b in zip(mine, mine[1:])):
            bad.append(f'bánh {w}: bậc không xen kẽ tăng, giảm')
        for a, b in zip(mine, mine[1:]):
            if a.w_to != b.w_from or a.slot_end_ms != b.t0_ms or b.t_prep_ms != a.t0_ms:
                bad.append(f'bánh {w}: bậc {a.id} → {b.id} không nối liền (bánh phải quay suốt lượt)')
        if mine and mine[0].t_prep_ms != 0:
            bad.append(f'bánh {w}: không quay từ đầu lượt')
    # background wheels hold their speed: between two steps only the stepping wheel changes
    tab = omega_table(rows)
    n = len(tab[0])
    t0s = {r.t0_ms: r.wheel for r in rows}
    for k in range(1, n):
        for w in range(N_WHEELS):
            if tab[w][k] != tab[w][k - 1] and t0s.get(k) != w and k != n:
                bad.append(f'ms {k}: bánh {w} đổi ω ngoài bậc của nó')
                break
    return bad


def summary(sched: Sequence[Transition], seed: int) -> dict:
    rows = sorted(sched, key=lambda r: r.t0_ms)
    gaps = [b.t0_ms - a.t0_ms for a, b in zip(rows, rows[1:])]
    return dict(seed=seed, steps=len(rows), duration_ms=max(r.slot_end_ms for r in rows),
                gap_min_ms=min(gaps) if gaps else None, gap_max_ms=max(gaps) if gaps else None,
                window_max_ms=max(window_ms(r.w_from) for r in rows),
                w_min=min(min(r.w_from, r.w_to) for r in rows), w_max=max(max(r.w_from, r.w_to) for r in rows),
                up=sum(1 for r in rows if r.w_to > r.w_from), down=sum(1 for r in rows if r.w_to < r.w_from))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Lịch bậc ω của S06 T3 (spec S06 T3, Z7, L7-10)')
    ap.add_argument('--seed', type=int, default=LOCKED_SEED)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    sched = make_schedule(a.seed)
    bad = check(sched)
    write_csv(a.out, sched)
    info = dict(summary(sched, a.seed), checks_failed=bad, ok=not bad)
    side = os.path.splitext(a.out)[0] + '_check.json'
    with open(side, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"{info['steps']} bậc, {info['duration_ms'] / 1000:.1f} s, seed {a.seed} -> {a.out} "
          f"(khoảng cách {info['gap_min_ms']}–{info['gap_max_ms']} ms, cửa sổ lớn nhất {info['window_max_ms']:.2f} ms)")
    for b in bad:
        print('LỖI LỊCH:', b, file=sys.stderr)
    return 2 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
