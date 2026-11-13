"""
hil_edge_map.py: RS-4, the ESP32 run_active rising edge of a HIL run mapped to the plant step of the Pi
frame log, so t = 0 of the O1 schedule on HIL is the tick where the firmware starts to apply the command
(O1 v11.9 3.4, dossier v3.2 5.2), as on the floor. Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi");
after Codex Sol round 1 (G32-2, -3, -4, -6, -10, -12) the method is a two-sided bracket, Liam 13/11 CODE
local_7f7b0ef4 ("Kẹp hai phía (Recommended)", L97). No firmware change, no new data path: the ESP32 log of
agv_logger.py and the frame log of run_hil.py --frame-log, both written by every bench session.

Each log line has timestamp_us = the start of its control tick (t_now). Two fits put the ESP32 clock on the
plant steps, with errors of opposite sign, both from causality:
  count side (EARLY bound)  the logged encoder counts are the plant wheel angle at the plant step where they
      were sampled: count_i - count_edge = K (A(a + (ts_i - ts_edge) / period_us) - A(a)), K = 1320 / 2 pi,
      A(k) = sum_{j<k} omega_j [|omega_j| > ENC_OMEGA_MIN] period. The counts are never newer than the
      tick: encoder_task samples them on its own 10 ms loop (AGV cfc892b main.c 228-234; push_log reads
      them, 246-250), and EncoderGen's discrete edges come after the ideal angle (slow start); so the fitted
      a is early by the sample age. Early bound: a_count - COUNT_MARGIN (EncoderGen retiming may put a count
      up to about one count ahead [ĐỀ XUẤT]).
  PWM side (LATE bound)     the pwm the firmware writes in a tick (log columns pwm_*) is measured by the H7
      and reaches the plant in the MISO duty of a later frame (H7 answers frame N in frame N + 2; at least
      MISO_LAG_MIN frames is taken [CÁCH HIỂU]); the duty cannot appear before it is written. Fitting the
      ESP32 pwm steps to the MISO duty gives a late by the compute time in the tick plus the capture and frame
      lag. Late bound: a_pwm - MISO_LAG_MIN + PWM_MARGIN.
The edge lies in [early, late]: edge_step = the middle, U_t = half the width x period + DRIFT_PPM over the fit
span (the clock rates of ESP32 and Pi are taken as equal near the edge). The O1 schedule and the plant
steps both run on the Pi clock, so no drift to t_deadline enters on HIL (unlike the floor, where the camera
clock is mapped over the whole run). [ĐỀ XUẤT, not reviewed by THESIS: the spacing of the firmware ticks is
not added; the edge tick itself is inside the bracket.]
Each side is fitted on the lines from the edge to WINDOWS_S after the first motion (count side) or the first
non-zero pwm (PWM side); count side: smallest a over the windows, PWM side: largest (the windows' spread
widens the bracket). Refused (ok False, reason) when: not exactly one run_active rising edge; the edge line
does not follow a run_active = 0 line with the next log_seq (lost lines at the edge, G32-4); fewer than two
distinct windows with motion; RMS above RMS_MAX / PWM_COST_MAX; a cost minimum that is not unique (another
local minimum at >= UNIQUE_SEP steps about as low as the best and behind a ridge, G32-6); a pwm-duty correlation below
MIN_CORR on any wheel; fit on the edge of the search range; early > late; width above BRACKET_MAX_STEPS.
Thresholds [ĐỀ XUẤT], set from CLOSED_LOOP/2026-11-11_run01 (F02, F05, F08) and 2026-11-13_run08 (K1a, K2).

Log reading: the CSV of agv_logger.py by column name; with <log>.segments.csv, lines marked 'duplicate' are
dropped after checking they equal an earlier kept line (L92), 'conflict' / 'ambiguous' or more than one
segment refuse the log; without it, a duplicate line refuses the log (S05 6.3, G32-10). Sorted by log_seq.

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
WINDOWS_S = (1.0, 2.0, 3.0)      # s after the first motion / first non-zero pwm
SEARCH = (-100.0, 400.0)         # steps around the RUN step
COARSE, FINE = 0.25, 0.01        # steps
MISO_LAG_MIN = 1.0               # frames [CÁCH HIỂU] (H7 answers frame N in N + 2; 1 taken, wider bracket)
COUNT_MARGIN = 1.0               # steps [ĐỀ XUẤT]
PWM_MARGIN = 0.5                 # steps [ĐỀ XUẤT]
MIN_MOVE_COUNTS = 200            # [ĐỀ XUẤT] sum over the wheels of |count change| in a window
RMS_MAX = 3.0                    # [ĐỀ XUẤT] counts
PWM_COST_MAX = 100.0             # [ĐỀ XUẤT] capped mean square (pwm units^2)
PWM_CAP = 400.0                  # pwm units^2: one frame's square error is capped (capture noise)
GLITCH = 990.0                   # |duty x 1000| >= this: a capture glitch frame, not used
MIN_CORR = 0.8                   # [ĐỀ XUẤT] pwm vs duty, per wheel
UNIQUE_SEP = 5.0                 # steps
UNIQUE_RATIO = 1.5               # [ĐỀ XUẤT] a real alias costs about the same as the best (ratio ~1);
                                 # the 1 ms frame ripple of the PWM basin gives about 2 (2026-11-11_run01)
BRACKET_MAX_STEPS = 20.0         # [ĐỀ XUẤT]
RIDGE_FRAC = 0.25                # [ĐỀ XUẤT] alias test (_unique)
DRIFT_PPM = 100.0                # [ĐỀ XUẤT] clock-rate difference bound ESP32 vs Pi, over the fit span
WHEELS = ('fl_count', 'fr_count', 'rl_count', 'rr_count')
PWMS = ('pwm_fl', 'pwm_fr', 'pwm_rl', 'pwm_rr')
NEED = ('timestamp_us', 'log_seq', 'motor_on', 'run_active') + WHEELS + PWMS
SEG_COLS = ['row', 'log_seq', 'timestamp_us', 'segment', 'status']


def read_esp_log(path: str) -> dict:
    """The columns NEED of an agv_logger CSV as arrays, duplicates handled (docstring), sorted by log_seq."""
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
    keep, seen = [], set()
    if os.path.isfile(seg):
        with open(seg, newline='', encoding='utf-8') as f:
            marks = list(csv.reader(f))
        if not marks or marks[0] != SEG_COLS or len(marks) - 1 != len(body):
            raise ValueError(f'{seg}: header or row count does not match {path}')
        segs = set()
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
        for i, r in enumerate(body, start=1):
            if tuple(r) in seen:
                raise ValueError(f'{path} line {i + 1}: duplicate line and no {os.path.basename(seg)} to '
                                 'confirm dropping it (S05 6.3)')
            seen.add(tuple(r))
            keep.append(r)
    try:
        cols = {c: np.array([int(r[head.index(c)]) for r in keep], dtype=np.int64)
                for c in NEED if c not in PWMS}
        pw = np.array([[float(r[head.index(c)]) for c in PWMS] for r in keep], dtype=float).reshape(-1, 4)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from None
    order = np.argsort(cols['log_seq'], kind='stable')
    out = {c: v[order] for c, v in cols.items()}
    if len(np.unique(out['log_seq'])) != len(out['log_seq']):
        raise ValueError(f'{path}: two different lines with the same log_seq')
    out['counts'] = np.column_stack([out[w] for w in WHEELS]).astype(float)
    out['pwm'] = pw[order]
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


def _minimise(cost, lo, hi):
    """(a, best cost, coarse grid, coarse costs): coarse grid then a fine grid around the best."""
    grid = np.arange(lo, hi + COARSE / 2, COARSE)
    c = np.array([cost(a) for a in grid])
    a0 = grid[int(np.argmin(c))]
    fine = np.arange(a0 - COARSE, a0 + COARSE + FINE / 2, FINE)
    cf = np.array([cost(a) for a in fine])
    return float(fine[int(np.argmin(cf))]), float(cf.min()), grid, c


def _unique(grid, c, a, floor):
    """True unless another local minimum m of the coarse cost, UNIQUE_SEP steps or more from a, is an alias:
    about as low as the best (c_m < UNIQUE_RATIO x best + floor) AND behind a ridge (c_m - best <= RIDGE_FRAC
    x (highest cost between the two - best)). A ripple of the 1 ms frames on the wall of the one basin has
    no such ridge; a repeated pattern (G32-6) has both."""
    ib = int(np.argmin(c))
    best = float(c[ib])
    loc = np.nonzero((c <= np.r_[np.inf, c[:-1]]) & (c <= np.r_[c[1:], np.inf]) & (np.abs(grid - a) >= UNIQUE_SEP))[0]
    for j in loc:
        ridge = float(np.max(c[min(ib, j):max(ib, j) + 1]))
        if c[j] < UNIQUE_RATIO * best + floor and c[j] - best <= RIDGE_FRAC * (ridge - best):
            return False
    return True


def _windows(ts, idx, t_from):
    """Lines of `idx` up to t_from + W, one set per window; a window with the same lines as the one
    before is dropped (a short log, G32-6)."""
    out, last = [], None
    for W in WINDOWS_S:
        sel = idx[ts[idx] <= t_from + W * 1e6]
        if last is not None and len(sel) == len(last):
            continue
        out.append((W, sel))
        last = sel
    return out


def count_side(A, k_run, esp, ie, period_us):
    ts, cnt = esp['timestamp_us'].astype(float), esp['counts']
    after = np.arange(ie, len(ts))
    after = after[esp['run_active'][after] == 1]
    moved = after[np.any(cnt[after] != cnt[ie], axis=1)]
    if not len(moved):
        return dict(ok=False, reason='counts never move after the edge')
    K = COUNTS_PER_REV / (2.0 * math.pi)
    lo, hi = k_run + SEARCH[0], k_run + SEARCH[1]
    fits = {}
    for W, sel in _windows(ts, after, ts[moved[0]]):
        if np.abs(np.diff(cnt[sel], axis=0)).sum() < MIN_MOVE_COUNTS:
            continue
        dk, dc = (ts[sel] - ts[ie]) / period_us, cnt[sel] - cnt[ie]

        def cost(a, dk=dk, dc=dc):
            pred = (_interp(A, a + dk) - _interp(A, np.array([a]))) * K
            return float(np.mean((dc - pred) ** 2))
        a, c, grid, cs = _minimise(cost, lo, hi)
        fits[str(W)] = dict(a=a, rms_counts=math.sqrt(c), n_lines=int(len(sel)), t_end_us=float(ts[sel][-1]),
                            unique=_unique(grid, cs, a, 0.25), on_edge=bool(a - lo < 2 * COARSE or hi - a < 2 * COARSE))
    out = dict(fits=fits)
    if len(fits) < 2:
        return dict(out, ok=False, reason=f'count side: {len(fits)} distinct window(s) with motion (need 2)')
    bad = [w for w, f in fits.items() if f['on_edge'] or not f['unique'] or f['rms_counts'] > RMS_MAX]
    if bad:
        f = fits[bad[0]]
        why = ('edge of the search range' if f['on_edge'] else 'cost minimum not unique' if not f['unique']
               else f'RMS {f["rms_counts"]:.2f} > {RMS_MAX}')
        return dict(out, ok=False, reason=f'count side, window {bad[0]} s: {why}')
    a = min(f['a'] for f in fits.values())
    return dict(out, ok=True, a=a, spread=max(f['a'] for f in fits.values()) - a,
                t_end_us=max(f['t_end_us'] for f in fits.values()))


def pwm_side(duty, k_run, esp, ie, period_us):
    ts, pw = esp['timestamp_us'].astype(float), esp['pwm']
    D = np.asarray(duty, dtype=float) * 1000.0
    n = len(D)
    after = np.arange(ie, len(ts))
    after = after[esp['run_active'][after] == 1]
    on = after[np.any(pw[after] != 0, axis=1)]
    if not len(on):
        return dict(ok=False, reason='pwm never leaves 0 after the edge')
    lo, hi = k_run + SEARCH[0], k_run + SEARCH[1]
    good = np.all(np.abs(D) < GLITCH, axis=1)
    fits = {}
    for W, sel in _windows(ts, after, ts[on[0]]):
        t_last = ts[sel][-1]
        k_hi = int(min(n, hi + (t_last - ts[ie]) / period_us + 1))
        frames = np.arange(max(0, int(lo) - 5), k_hi)
        frames = frames[good[frames]]
        lines = np.r_[ie - 1, sel] if ie > 0 else sel        # the 0 before the edge

        def pred(a, frames=frames, lines=lines):
            tick_k = a + (ts[lines] - ts[ie]) / period_us
            j = np.searchsorted(tick_k, frames, side='right') - 1
            P = pw[lines][np.clip(j, 0, len(lines) - 1)].copy()
            P[(j < 0) | (lines[np.clip(j, 0, len(lines) - 1)] < ie)] = 0.0
            return P
        P0 = pred(k_run + 15.0)
        sgn = np.sign(np.sum(P0 * D[frames], axis=0))
        sgn[sgn == 0] = 1.0

        def cost(a):
            e = D[frames] - pred(a) * sgn
            return float(np.mean(np.minimum(e ** 2, PWM_CAP)))
        a, c, grid, cs = _minimise(cost, lo, hi)
        P = pred(a) * sgn
        corr = [float(np.corrcoef(P[:, w], D[frames, w])[0, 1]) if np.std(P[:, w]) > 0 and np.std(D[frames, w]) > 0
                else float('nan') for w in range(4)]
        fits[str(W)] = dict(a=a, cost=c, corr=corr, sign=[float(s) for s in sgn], n_frames=int(len(frames)),
                            n_glitch=int(np.count_nonzero(~good[max(0, int(lo) - 5):k_hi])), t_end_us=float(t_last),
                            unique=_unique(grid, cs, a, 10.0), on_edge=bool(a - lo < 2 * COARSE or hi - a < 2 * COARSE))
    out = dict(fits=fits)
    if len(fits) < 2:
        return dict(out, ok=False, reason=f'PWM side: {len(fits)} distinct window(s) (need 2)')
    for w, f in fits.items():
        why = ('edge of the search range' if f['on_edge'] else 'cost minimum not unique' if not f['unique']
               else f'cost {f["cost"]:.1f} > {PWM_COST_MAX}' if f['cost'] > PWM_COST_MAX
               else f'pwm-duty correlation {min(f["corr"]):.2f} < {MIN_CORR}'
               if not all(np.isfinite(f['corr'])) or min(f['corr']) < MIN_CORR else '')
        if why:
            return dict(out, ok=False, reason=f'PWM side, window {w} s: {why}')
    a = max(f['a'] for f in fits.values())
    return dict(out, ok=True, a=a, spread=a - min(f['a'] for f in fits.values()),
                t_end_us=max(f['t_end_us'] for f in fits.values()))


def map_edge(omega, duty, k_run: int, esp: dict, period_s: float) -> dict:
    """Plant step of the run_active edge of `esp` (read_esp_log) on a frame log with wheel speeds `omega`
    (n, 4) [rad/s], MISO duty `duty` (n, 4) and RUN step k_run. Returns a dict, ok True / False with reason."""
    period_us = period_s * 1e6
    out = dict(ok=False, reason='', method='two_sided_bracket', k_run=int(k_run), period_s=period_s,
               counts_per_rev=COUNTS_PER_REV, omega_min=OMEGA_MIN, windows_s=list(WINDOWS_S),
               constants=dict(miso_lag_min=MISO_LAG_MIN, count_margin=COUNT_MARGIN, pwm_margin=PWM_MARGIN,
                              rms_max=RMS_MAX, pwm_cost_max=PWM_COST_MAX, min_corr=MIN_CORR,
                              unique_ratio=UNIQUE_RATIO, bracket_max_steps=BRACKET_MAX_STEPS, drift_ppm=DRIFT_PPM,
                              min_move_counts=MIN_MOVE_COUNTS))
    edges = rising_edges(esp['run_active'])
    out['n_edges'] = int(len(edges))
    if len(edges) != 1:
        out['reason'] = f'{len(edges)} run_active rising edges in the log (need exactly 1)'
        return out
    ie = int(edges[0])
    out.update(edge_log_seq=int(esp['log_seq'][ie]), edge_timestamp_us=int(esp['timestamp_us'][ie]))
    if int(esp['log_seq'][ie]) != int(esp['log_seq'][ie - 1]) + 1:
        out['reason'] = (f'log_seq {int(esp["log_seq"][ie - 1])} -> {int(esp["log_seq"][ie])} at the edge: lines lost, '
                         'the first run_active line may not be the edge tick (G32-4)')
        return out
    c = count_side(plant_angle(omega, period_s), k_run, esp, ie, period_us)
    p = pwm_side(duty, k_run, esp, ie, period_us)
    out.update(count_side=c, pwm_side=p)
    if not c['ok'] or not p['ok']:
        out['reason'] = c.get('reason') or p.get('reason')
        return out
    early = c['a'] - COUNT_MARGIN
    late = p['a'] - MISO_LAG_MIN + PWM_MARGIN
    span_s = (max(c['t_end_us'], p['t_end_us']) - esp['timestamp_us'][ie]) * 1e-6
    drift_s = DRIFT_PPM * 1e-6 * span_s
    a = 0.5 * (early + late)
    out.update(early_step=early, late_step=late, edge_step=a, offset_ms=(a - k_run) * period_s * 1e3,
               u_t_s=0.5 * (late - early) * period_s + drift_s,
               u_t_parts_s=dict(half_bracket=0.5 * (late - early) * period_s, drift=drift_s))
    if early > late:
        out['reason'] = f'no bracket: early bound {early:.2f} > late bound {late:.2f} steps'
    elif late - early > BRACKET_MAX_STEPS:
        out['reason'] = f'bracket {late - early:.2f} steps > {BRACKET_MAX_STEPS}'
    else:
        out['ok'] = True
    return out


def map_run(frame_log: str, esp_log: str) -> dict:
    """map_edge on the files (frame log of run_hil.py, ESP32 CSV of agv_logger.py)."""
    import hil_o1_score as H
    manifest, mosi, miso = H.load(frame_log)
    state, ref, flags, duty = H.decode_frames(mosi, miso)
    k_run, src = H.command_start(manifest, ref, flags)
    out = map_edge(state[:, 0:4], duty, k_run, read_esp_log(esp_log), float(manifest.get('period_us', 1000)) * 1e-6)
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
