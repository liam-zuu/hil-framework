"""
t5_schedule.py — wheel-speed schedule of S06 T5 (H5, encoder frequency error; spec S06 v3.4, W15, Y16, Z1).

Points: omega in {0.32, 0.5, 1, 2, 10, 20, 39.2} rad/s plus the four off-grid points {9.9942, 19.979,
31.68, 38.78} rad/s (spec S06 v3.5, T5, câu 7: half period 9519.98/omega ticks with a fraction near 0.5;
checked on the decoded omega_d and the encoder_gen.c rounding in ket_qua_e3/bang.csv; the spec's 9.995 is
moved to 9.9942 = int16 8187, frac 0.550 instead of 0.434, "chỉnh ω đặt nếu cần", CC1-10, Codex C3-18;
OFF_GRID_LOCK keeps int16, omega_d, half period before / after rounding, checked before writing), both directions,
scored; 0.29 rad/s is the indicator point (omega_d = 0.29054 <= 0.3, expect 0 counts, not scored). Pass 1 sweeps up, pass 2
sweeps down; at each omega the direction alternates (+ then -). W15: one visit per pass where the
4 wheels get different speeds, so a swapped wheel shows up (H4).

Each visit holds the speeds for VISIT_WINDOWS x 10 s. The scorer (tools/s06_t5_score.py) tiles the
visit into 10 s windows on the ESP32 clock and drops the first and the last (they hold the change
of speed and the alignment error), so 5 full windows are scored per visit.
CODE choices [ĐỀ XUẤT], not in the spec: 7 windows per visit (70 s), the order "+ then -", the W15
speeds (+1, -2, +10, -20 rad/s, all scored points, mixed signs) and the W15 visit at the end of
each pass. No randomness: the schedule is fixed, there is no seed.

    python t5_schedule.py --pass 1 --out T5_pass1.csv     # then run_hil.py --omega-steps @T5_pass1.csv
It also prints the s06_logger.py command (profile heavy, duration = schedule + 60 s).
"""
from __future__ import annotations

import argparse
import csv
import math
import struct
from dataclasses import dataclass, asdict, fields
from typing import List, Sequence

OFF_GRID_POINTS = (9.9942, 19.979, 31.68, 38.78)   # v3.5 câu 7 (9.995 -> 9.9942, C3-18), frac in [0.48; 0.56]
POINTS = (0.32, 0.5, 1.0, 2.0, 9.9942, 10.0, 19.979, 20.0, 31.68, 38.78, 39.2)  # scored, rad/s (Z1, Q5, v3.5)
# C3-18: per off-grid |omega|: (int16 in the frame, omega_d of the H7 in float32, half period before rounding
# (float32), timer value after (uint32_t)(hp + 0.5f)); the - direction is the same with int16, omega_d negated
OFF_GRID_LOCK = {
    9.9942: (8187, 9.99420166015625, 952.5501708984375, 953),
    19.979: (16366, 19.9786376953125, 476.50787353515625, 477),
    31.68: (25951, 31.679433822631836, 300.5097351074219, 301),
    38.78: (31768, 38.780479431152344, 245.48374938964844, 245),
}
INDICATOR = 0.29                                    # omega_d 0.29054 <= deadband 0.3: expect 0 counts
W15_OMEGAS = (1.0, -2.0, 10.0, -20.0)               # FL, FR, RL, RR [ĐỀ XUẤT]
WINDOW_S = 10.0
VISIT_WINDOWS = 7                                   # first and last dropped -> 5 scored [ĐỀ XUẤT]
LOGGER_MARGIN_S = 60                                # s06_load run outlasts the schedule [ĐỀ XUẤT]


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


def _f32(x: float) -> float:
    """Round to IEEE single (one float32 operation = its exact binary64 result rounded once)."""
    return struct.unpack('<f', struct.pack('<f', x))[0]


H7_OMEGA_LSB = _f32(40.0 / 32767.0)                        # HIL_OMEGA_MAX / INT16_MAXF (spi1_handler.c:116)
H7_HP_NUM = _f32(_f32(1e6 * 1.0) * _f32(math.pi))           # ENC_TICKS_PER_S * (float)M_PI (encoder_gen.c:45)


def h7_timer(hp: float) -> int:
    """encoder_gen.c:47-49: clamp to [1, 65535], then (uint32_t)(hp + 0.5f)."""
    if hp > 65535.0:
        return 65535
    if hp < 1.0:
        return 1
    return int(math.floor(_f32(hp + 0.5)))


def h7_off_grid(omega_set: float) -> tuple:
    """(int16 in bytes 0-1 of protocol.encode_state, omega_d = (float)int16 * (40.0f / 32767.0f), half period
    1e6f * (float)M_PI / (|omega_d| * 330.0f) in float32, timer value after rounding)."""
    import protocol
    raw = struct.unpack_from('>h', protocol.encode_state([omega_set] * 4 + [0.0] * 6))[0]
    wd = _f32(raw * H7_OMEGA_LSB)
    hp = _f32(H7_HP_NUM / _f32(abs(wd) * 330.0))
    return raw, wd, hp, h7_timer(hp)


def check_off_grid_lock() -> None:
    """C3-18: OFF_GRID_LOCK must hold every off-grid point and match the frame and the H7 timer in both
    directions; ValueError otherwise (the schedule is not written)."""
    if tuple(OFF_GRID_LOCK) != OFF_GRID_POINTS:
        raise ValueError(f'OFF_GRID_LOCK {tuple(OFF_GRID_LOCK)} != OFF_GRID_POINTS {OFF_GRID_POINTS}')
    for m, (raw, wd, hp, timer) in OFF_GRID_LOCK.items():
        for s in (1, -1):
            got, want = h7_off_grid(s * m), (s * raw, s * wd, hp, timer)
            if got != want:
                raise ValueError(f'off-grid {s * m}: frame/H7 give {got}, OFF_GRID_LOCK has {want}')


def main() -> int:
    ap = argparse.ArgumentParser(description='S06 T5 wheel-speed schedule')
    ap.add_argument('--pass', dest='run_pass', type=int, required=True, choices=[1, 2])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    check_off_grid_lock()
    sched = make_schedule(a.run_pass)
    write_csv(a.out, sched)
    print(f'{a.out}: pass {a.run_pass}, {len(sched)} visits, {sched[-1].t_end_ms / 1000:.0f} s')
    print('ESP32 run (start before run_hil.py):')
    print('  ' + logger_command(sched))
    return 0


def logger_duration_s(sched: Sequence[Visit]) -> int:
    """s06_load run length: the whole schedule + LOGGER_MARGIN_S, so the run outlasts run_hil.py."""
    return int(math.ceil(sched[-1].t_end_ms / 1000.0 + LOGGER_MARGIN_S))


def logger_command(sched: Sequence[Visit]) -> str:
    return (f'python tools/s06_logger.py --esp32 <ESP32_IP> --profile heavy '
            f'--duration {logger_duration_s(sched)} --out <run dir>')


if __name__ == '__main__':
    raise SystemExit(main())
