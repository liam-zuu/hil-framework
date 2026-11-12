"""
hil_edge_map.py: RS-4, the ESP32 run_active rising edge of a HIL run mapped to the plant step of the Pi
frame log, so t = 0 of the O1 schedule on HIL is the tick where the firmware starts to apply the command
(O1 v11.9 3.4, dossier v3.2 5.2), as on the floor. Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi").
No new data path: the ESP32 log of agv_logger.py and the frame log of run_hil.py --frame-log, both written
by every bench session (hil_bench_session.py --run).

Method (count matching, no firmware change). The H7 makes the encoder edges from the plant wheel speeds
(EncoderGen, ENC_PPR edges per turn on channel A, wheels below ENC_OMEGA_MIN emit nothing); the ESP32
counts them (quadrature, COUNTS_PER_REV per turn) and logs the counts with timestamp_us of its tick. So the
count of a log line is the plant wheel angle at the plant step of that tick:
    count_i - count_edge = K (A(a + (ts_i - ts_edge) / period_us) - A(a)),   K = COUNTS_PER_REV / 2 pi
with A(k) = sum_{j<k} omega_j [|omega_j| > OMEGA_MIN] x period (linear between steps) and a the plant step
of the edge line. a is the least-squares fit over the four wheels, on the lines from the edge to
WINDOWS_S after the first line whose counts moved (search a in [k_run + SEARCH[0], k_run + SEARCH[1]],
grid then refinement). Clock rate taken as equal (ESP32 and Pi): the error is in the bound.
Result: offset_ms = (a - k_run) x period, the difference to the plant RUN step used before (the "chênh"
of O1 3.4), and the bound U_t of t = 0 [ĐỀ XUẤT, not yet reviewed by THESIS]:
    U_t = period x (1 + spread) + DRIFT_PPM x 1e-6 x (ts_window_end - ts_edge)
  1 step: the encoder shows step k from the frame transfer on (the H7 takes it within the step; estimate);
  spread: largest |a_W - a| over the windows (fit stability); drift: clock-rate difference bound.
The mapping is refused (ok False, reason) when: the log has no run_active rising edge or more than one;
fewer than two windows have MIN_MOVE_COUNTS of count change; the RMS count residual is above RMS_MAX;
the spread is above SPREAD_MAX_STEPS; the fit sits on the edge of the search range.
Thresholds RMS_MAX, SPREAD_MAX_STEPS, DRIFT_PPM, MIN_MOVE_COUNTS: [ĐỀ XUẤT], set from the bench run
CLOSED_LOOP/2026-11-11_run01 (F02, F05, F08; RMS 0.7-2.0 counts on 1-3 s windows, a stable to 1-2 ms) and
CLOSED_LOOP/2026-11-13_run08 (K2), to be confirmed.

Log reading: the CSV of agv_logger.py by column name (any column count that has the columns used); with
<log>.segments.csv, lines marked 'duplicate' are dropped after checking they equal an earlier kept line
(L92), 'conflict' / 'ambiguous' or more than one segment refuse the log; without it, equal lines are
dropped. Lines sorted by log_seq.

  python tools/hil_edge_map.py --frame-log <run>.npz --esp-log <run>.csv [--json out]
Last line: #ket_qua,ma=0 mapped / 3 not mapped (reason) / 2 error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ENC_PPR = 330                    # stm32h7/Core/Inc/encoder_gen.h ENC_PPR (a test checks)
COUNTS_PER_REV = 4 * ENC_PPR     # quadrature on the ESP32 (AGV encoder.h PPR 1320)
OMEGA_MIN = 0.3                  # rad/s, stm32h7/Core/Src/encoder_gen.c ENC_OMEGA_MIN (a test checks)
WINDOWS_S = (1.0, 2.0, 3.0)      # s after the first count change
SEARCH = (-100.0, 400.0)         # steps around the RUN step
COARSE, FINE = 0.25, 0.01        # steps
MIN_MOVE_COUNTS = 200            # [ĐỀ XUẤT] sum over the wheels of |count change| in a window
RMS_MAX = 3.0                    # [ĐỀ XUẤT] counts
SPREAD_MAX_STEPS = 3.0           # [ĐỀ XUẤT]
DRIFT_PPM = 100.0                # [ĐỀ XUẤT] clock-rate difference bound ESP32 vs Pi
WHEELS = ('fl_count', 'fr_count', 'rl_count', 'rr_count')
NEED = ('timestamp_us', 'log_seq', 'motor_on', 'run_active') + WHEELS
SEG_COLS = ['row', 'log_seq', 'timestamp_us', 'segment', 'status']


def read_esp_log(path: str) -> dict:
    """The columns NEED of an agv_logger CSV as int arrays, duplicates dropped, sorted by log_seq."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f'{path}: empty')
    head, body = rows[0], rows[1:]
    miss = [c for c in NEED if c not in head]
    if miss:
        raise ValueError(f'{path}: columns {miss} missing')
    if any(len(r) != len(head) for r in body):
        raise ValueError(f'{path}: lines with another column count')
    seg = path + '.segments.csv'
    keep = []
    if os.path.isfile(seg):
        with open(seg, newline='', encoding='utf-8') as f:
            marks = list(csv.reader(f))
        if not marks or marks[0] != SEG_COLS or len(marks) - 1 != len(body):
            raise ValueError(f'{seg}: header or row count does not match {path}')
        seen, segs = set(), set()
        s, t = head.index('log_seq'), head.index('timestamp_us')
        for i, (r, m) in enumerate(zip(body, marks[1:]), start=1):
            if m[0] != str(i) or m[1] != r[s] or m[2] != r[t]:
                raise ValueError(f'{seg} row {i} does not match the CSV line')
            if m[4] in ('conflict', 'ambiguous'):
                raise ValueError(f'{seg}: {m[4]} lines, log not usable as is')
            if m[4] == 'duplicate':
                if tuple(r) not in seen:
                    raise ValueError(f'{seg} row {i}: marked duplicate, no earlier equal line')
                continue
            seen.add(tuple(r))
            segs.add(m[3])
            keep.append(r)
        if len(segs) > 1:
            raise ValueError(f'{path}: more than one logger segment (DUT restart)')
    else:
        seen = set()
        for r in body:
            if tuple(r) not in seen:
                seen.add(tuple(r))
                keep.append(r)
    try:
        cols = {c: np.array([int(r[head.index(c)]) for r in keep], dtype=np.int64) for c in NEED}
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from None
    order = np.argsort(cols['log_seq'], kind='stable')
    out = {c: v[order] for c, v in cols.items()}
    if len(np.unique(out['log_seq'])) != len(out['log_seq']):
        raise ValueError(f'{path}: two different lines with the same log_seq')
    out['counts'] = np.column_stack([out[w] for w in WHEELS]).astype(float)
    return out


def rising_edges(run_active) -> np.ndarray:
    ra = np.asarray(run_active)
    return np.nonzero((ra[1:] == 1) & (ra[:-1] == 0))[0] + 1


def plant_angle(omega, period_s) -> np.ndarray:
    """A[k] = encoder angle at the start of step k (k = 0..n), wheels below OMEGA_MIN emit nothing."""
    w = np.asarray(omega, dtype=float).copy()
    w[np.abs(w) <= OMEGA_MIN] = 0.0
    return np.vstack([np.zeros(w.shape[1]), np.cumsum(w * period_s, axis=0)])


def _interp(A, k):
    kk = np.clip(k, 0.0, len(A) - 1.0)
    i = np.minimum(np.floor(kk).astype(int), len(A) - 2)
    f = (kk - i)[:, None]
    return A[i] * (1.0 - f) + A[i + 1] * f


def _cost(A, a, dk, dc):
    K = COUNTS_PER_REV / (2.0 * math.pi)
    pred = (_interp(A, a + dk) - _interp(A, np.array([a]))) * K
    return float(np.mean((dc - pred) ** 2))


def fit_offset(A, dk, dc, lo, hi):
    """a in [lo, hi] minimising the count residual: coarse grid, then a fine grid around the best."""
    grid = np.arange(lo, hi + COARSE / 2, COARSE)
    c = np.array([_cost(A, a, dk, dc) for a in grid])
    a0 = grid[int(np.argmin(c))]
    fine = np.arange(a0 - COARSE, a0 + COARSE + FINE / 2, FINE)
    cf = np.array([_cost(A, a, dk, dc) for a in fine])
    a = float(fine[int(np.argmin(cf))])
    return a, math.sqrt(float(cf.min()))


def map_edge(omega, k_run: int, esp: dict, period_s: float) -> dict:
    """Plant step of the run_active edge of `esp` (read_esp_log) on a frame log with wheel speeds
    `omega` (n, 4) [rad/s] and RUN step k_run. Returns a dict, ok True / False with reason."""
    period_us = period_s * 1e6
    out = dict(ok=False, reason='', method='encoder_count_match', k_run=int(k_run), period_s=period_s,
               counts_per_rev=COUNTS_PER_REV, omega_min=OMEGA_MIN, windows_s=list(WINDOWS_S),
               thresholds=dict(rms_max=RMS_MAX, spread_max_steps=SPREAD_MAX_STEPS, drift_ppm=DRIFT_PPM,
                               min_move_counts=MIN_MOVE_COUNTS))
    edges = rising_edges(esp['run_active'])
    out['n_edges'] = int(len(edges))
    if len(edges) != 1:
        out['reason'] = f'{len(edges)} run_active rising edges in the log (need exactly 1)'
        return out
    ie = int(edges[0])
    ts, cnt = esp['timestamp_us'].astype(float), esp['counts']
    out.update(edge_log_seq=int(esp['log_seq'][ie]), edge_timestamp_us=int(ts[ie]))
    after = np.arange(ie, len(ts))
    after = after[esp['run_active'][after] == 1]
    moved = after[np.any(cnt[after] != cnt[ie], axis=1)]
    if not len(moved):
        out['reason'] = 'counts never move after the edge'
        return out
    t_on = ts[moved[0]]
    A = plant_angle(omega, period_s)
    lo, hi = k_run + SEARCH[0], k_run + SEARCH[1]
    fits = {}
    for W in WINDOWS_S:
        sel = after[ts[after] <= t_on + W * 1e6]
        dc = cnt[sel] - cnt[ie]
        if np.abs(np.diff(cnt[sel], axis=0)).sum() < MIN_MOVE_COUNTS:
            continue
        a, rms = fit_offset(A, (ts[sel] - ts[ie]) / period_us, dc, lo, hi)
        fits[W] = dict(a=a, rms_counts=rms, n_lines=int(len(sel)), t_end_us=float(ts[sel][-1]))
    out['fits'] = {str(W): f for W, f in fits.items()}
    if len(fits) < 2:
        out['reason'] = f'{len(fits)} window(s) with {MIN_MOVE_COUNTS} counts of motion (need 2)'
        return out
    W = max(fits)
    a, rms = fits[W]['a'], fits[W]['rms_counts']
    spread = max(abs(f['a'] - a) for f in fits.values())
    drift_s = DRIFT_PPM * 1e-6 * (fits[W]['t_end_us'] - ts[ie]) * 1e-6
    out.update(edge_step=a, offset_ms=(a - k_run) * period_s * 1e3, rms_counts=rms, spread_steps=spread,
               u_t_s=period_s * (1.0 + spread) + drift_s, u_t_parts_s=dict(step=period_s, spread=period_s * spread,
                                                                          drift=drift_s))
    if a - lo < 2 * COARSE or hi - a < 2 * COARSE:
        out['reason'] = f'fit on the edge of the search range [{lo}, {hi}] steps'
    elif rms > RMS_MAX:
        out['reason'] = f'RMS count residual {rms:.2f} > {RMS_MAX}'
    elif spread > SPREAD_MAX_STEPS:
        out['reason'] = f'windows disagree by {spread:.2f} steps > {SPREAD_MAX_STEPS}'
    else:
        out['ok'] = True
    return out


def map_run(frame_log: str, esp_log: str) -> dict:
    """map_edge on the files (frame log of run_hil.py, ESP32 CSV of agv_logger.py)."""
    import hil_o1_score as H
    manifest, mosi, miso = H.load(frame_log)
    state, ref, flags, _ = H.decode_frames(mosi, miso)
    k_run, src = H.command_start(manifest, ref, flags)
    out = map_edge(state[:, 0:4], k_run, read_esp_log(esp_log), float(manifest.get('period_us', 1000)) * 1e-6)
    out.update(k_run_src=src, frame_log=os.path.basename(frame_log), esp_log=os.path.basename(esp_log))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame-log', required=True)
    ap.add_argument('--esp-log', required=True)
    ap.add_argument('--json', default='')
    a = ap.parse_args(argv)
    try:
        out = map_run(a.frame_log, a.esp_log)
    except (ValueError, KeyError, OSError) as exc:
        print('hil_edge_map:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    print(text)
    print(f'#ket_qua,ma={0 if out["ok"] else 3}')
    return 0 if out['ok'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
