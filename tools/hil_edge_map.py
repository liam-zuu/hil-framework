"""
hil_edge_map.py: RS-4, the ESP32 run_active rising edge of a HIL run mapped to the plant step of the Pi
frame log, so t = 0 of the O1 schedule on HIL is the tick where the firmware starts to apply the command
(O1 v11.9 3.4, dossier v3.2 5.2), as on the floor. Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi");
CODE local_7f7b0ef4 ("Kẹp hai phía", L97; after Codex Sol round 2/2, "Sửa firmware (Recommended)", L98).
Inputs: the ESP32 log of agv_logger.py (with its .events.txt) and the frame log of run_hil.py --frame-log.

L98 (AGV fix/enc-log-tick 2c02641): control_task reads the four PCNT counts right after t_now, with the
scheduler of its core suspended, and logs that pair (timestamp_us, *_count). Before, the counts were
encoder_task's copy from its own 10 ms loop, older or newer than the tick by an unknown amount (Codex
G32-R2-1 (b)); logs of such firmware are refused (FW_COUNTS_AT_TICK, #fw lines of the events file).

Count side (the bracket). Pi time on the frame log: frame k is sent at T_k (t_send_ns; k x period when the
frame log has no times). The H7 applies omega_k from T_k + d, d in [D_MIN_US, D_MAX_US] (transfer and main
loop; H7 'Frame: ... age_max=750us', CLOSED_LOOP/2026-11-13_run08 h7_COM3.txt) [ĐỀ XUẤT]. Theta(t) = the
ideal encoder angle for omega applied at T_k (wheels at or below ENC_OMEGA_MIN emit nothing), K = 1320 / 2 pi.
For an unknown s (Pi time), line i (ESP32 time ts_i, D_i = ts_i - ts_edge) fits when, on every wheel,
    | (count_i - count_edge) - K (Theta(s + D_i) - Theta(s)) | <= TOL_COUNTS + K W_i (DRIFT_PPM D_i + J_ESP_US)
W_i = the largest |omega| near s + D_i (Lipschitz bound for the clock-rate difference and the time between
t_now and the PCNT reads). TOL_COUNTS bounds how far the H7 EncoderGen's emitted count strays from K Theta
between two times: measured with the unchanged encoder_gen.c on the PC (stm32h7/pc_tests/encgen, profile
mode; tools/tests/encgen_oracle.py) on the wheel speeds of the six frame logs below and on stop/start,
reverse, slow-ramp, step profiles, with the H7 apply time random in 0..750 us: largest range 7.36 counts
(13/11, 03_CODE/docs/2026-11-13_RS4_L98_dem_tai_tick.md); TOL_COUNTS = 10 [ĐỀ XUẤT]. Every s on a 10 us grid
over the search range is tested on every line from the edge to FIT_S; the feasible s must form ONE
connected interval [s_lo, s_hi] (no alias, G32-6, G32-R2-3) away from the search range ends; the edge tick is
then at Pi time t_e in [s_lo + D_MIN_US - GRID_US, s_hi + D_MAX_US + GRID_US], given as plant steps (T_k
interpolated): early_step, late_step, edge_step the middle, U_t = half the width. The clock-rate difference
is inside the test, so U_t has no separate drift term. The O1 schedule and the plant steps run on the Pi, so
no drift to t_deadline enters on HIL. [ĐỀ XUẤT, not reviewed by THESIS: the spacing of the firmware ticks is
not added; the edge tick itself is bracketed.]

PWM side (cross-check only, not in the bracket). The pwm written in a tick (pwm_*) reaches the plant in the
MISO duty of a later frame, never an earlier one. The fit of the pwm steps to the duty is a step function of
the shift (whole frames), so its minimum is an interval; its right end p_hi (G32-R2-2) cannot be before the
edge: early_step <= p_hi is required, and p_lo - late_step <= PWM_LAG_MAX_STEPS (write within the 10 ms tick,
H7 capture, frame answer) [ĐỀ XUẤT]. The PWM checks of round 1 (cost, correlation, alias) still refuse.

Refused (ok False, reason) when: firmware before L98 or no events file; not exactly one run_active rising
edge; the edge line does not follow a run_active = 0 line with the next log_seq (G32-4); counts never move;
no feasible s, more than one feasible interval, or one on the search range end; bracket wider than
BRACKET_MAX_STEPS; the PWM fit fails or disagrees with the bracket.
Frame logs used for the thresholds: CLOSED_LOOP/2026-11-11_run01 (F02, F05, F08), 2026-11-13_run08 (K1a, K1b,
K2) (counts of the old firmware there are not used, only the plant wheel speeds).

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
WINDOWS_S = (1.0, 2.0, 3.0)      # s after the first non-zero pwm (PWM side)
SEARCH = (-100.0, 400.0)         # steps around the RUN step
COARSE, FINE = 0.25, 0.01        # steps (PWM side)
# count side (L98)
FW_COUNTS_AT_TICK = ('2c026415dfc6',)   # AGV firmware whose log counts are read at t_now (fix/enc-log-tick)
FIT_S = 3.0                      # s of log lines after the edge
TOL_COUNTS = 10.0                # [ĐỀ XUẤT] largest EncoderGen range measured 7.36 counts (docstring)
D_MIN_US, D_MAX_US = 0.0, 1000.0  # [ĐỀ XUẤT] H7 apply time after t_send (age_max 750 us + transfer)
J_ESP_US = 100.0                 # [ĐỀ XUẤT] t_now to the PCNT reads (a few us; interrupts may come in between)
GRID_US = 10.0                   # grid of s
PWM_LAG_MAX_STEPS = 15.0         # [ĐỀ XUẤT] tick compute (< 10 ms) + capture + frame answer
PWM_SLACK_STEPS = 0.5            # [ĐỀ XUẤT] the PWM fit puts ticks on k x period, frames move <= 0.15 step
PWM_COST_MAX = 100.0             # [ĐỀ XUẤT] capped mean square (pwm units^2)
PWM_CAP = 400.0                  # pwm units^2: one frame's square error is capped (capture noise)
GLITCH = 990.0                   # |duty x 1000| >= this: a capture glitch frame, not used
MIN_CORR = 0.8                   # [ĐỀ XUẤT] pwm vs duty, per wheel
UNIQUE_SEP = 5.0                 # steps
UNIQUE_RATIO = 1.5               # [ĐỀ XUẤT] a real alias costs about the same as the best (ratio ~1);
                                 # the 1 ms frame ripple of the PWM basin gives about 2 (2026-11-11_run01)
BRACKET_MAX_STEPS = 20.0         # [ĐỀ XUẤT]
RIDGE_FRAC = 0.25                # [ĐỀ XUẤT] alias test (_unique, PWM side)
DRIFT_PPM = 100.0                # [ĐỀ XUẤT] clock-rate difference bound ESP32 vs Pi
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
    ev = path + '.events.txt'
    out['fw'] = read_fw(ev) if os.path.isfile(ev) else None
    return out


def read_fw(path: str) -> list:
    """The '#fw,sha=..,dirty=..,env=..' lines of an agv_logger events file, as dicts (in file order)."""
    out = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            i = line.find('#fw,')
            if i < 0:
                continue
            d = dict(kv.split('=', 1) for kv in line[i + 4:].strip().split(',') if '=' in kv)
            out.append(d)
    return out


def fw_check(fw) -> str:
    """'' when every #fw line names firmware that logs counts read at t_now (L98), else the reason."""
    if fw is None:
        return 'no events file with #fw lines next to the ESP32 log: firmware not known'
    if not fw:
        return 'no #fw line in the events file: firmware not known'
    for d in fw:
        sha, dirty = d.get('sha', ''), d.get('dirty', '')
        if dirty != '0' or len(sha) < 12 or sha[:12] not in {s[:12] for s in FW_COUNTS_AT_TICK}:
            return (f'firmware sha={sha} dirty={dirty}: its log counts are not known to be read at t_now (L98, '
                    f'{FW_COUNTS_AT_TICK})')
    return ''


def rising_edges(run_active) -> np.ndarray:
    ra = np.asarray(run_active)
    return np.nonzero((ra[1:] == 1) & (ra[:-1] == 0))[0] + 1


def frame_times_us(n, period_s, t_send_ns=None) -> np.ndarray:
    """Pi time [us] of frames 0..n-1 from the first: t_send_ns of the frame log, else k x period."""
    if t_send_ns is None:
        return np.arange(n, dtype=float) * period_s * 1e6
    t = np.asarray(t_send_ns, dtype=np.int64)
    if len(t) != n or n < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('frame log t_send_ns: not one increasing time per frame')
    return (t - t[0]).astype(float) / 1e3


class Theta:
    """Ideal encoder angle [rad] of the four wheels at Pi time t [us] when omega_k is applied from T_k until
    T_k+1 (wheels at or below OMEGA_MIN emit nothing, as EncoderGen); 0 before T_0. wmax[k]: largest |omega|
    over frames k - 2 .. k + 2 (Lipschitz bound near frame k)."""

    def __init__(self, T_us, omega):
        w = np.asarray(omega, dtype=float).copy()
        w[np.abs(w) <= OMEGA_MIN] = 0.0
        self.T, self.w = np.asarray(T_us, dtype=float), w
        self.cum = np.vstack([np.zeros(w.shape[1]), np.cumsum(w[:-1] * (np.diff(self.T) * 1e-6)[:, None], axis=0)])
        a = np.abs(w)
        m = a.copy()
        for sh in (1, 2):
            m[sh:] = np.maximum(m[sh:], a[:-sh])
            m[:-sh] = np.maximum(m[:-sh], a[sh:])
        self.wmax = m

    def at(self, t):
        """(angle (..., 4), frame index (...)) at times t (any shape)."""
        t = np.asarray(t, dtype=float)
        j = np.clip(np.searchsorted(self.T, t, side='right') - 1, 0, len(self.T) - 1)
        th = self.cum[j] + self.w[j] * ((t - self.T[j]) * 1e-6)[..., None]
        th[t < self.T[0]] = 0.0
        return th, j


def _minimise(cost, lo, hi):
    """(a, best cost, coarse grid, coarse costs, fine grid, fine costs): coarse grid then a fine grid around
    the best."""
    grid = np.arange(lo, hi + COARSE / 2, COARSE)
    c = np.array([cost(a) for a in grid])
    a0 = grid[int(np.argmin(c))]
    fine = np.arange(a0 - 1.0, a0 + 1.0 + FINE / 2, FINE)      # 1 step each side: the plateau is <= 1 step
    cf = np.array([cost(a) for a in fine])
    return float(fine[int(np.argmin(cf))]), float(cf.min()), grid, c, fine, cf


def _plateau(fine, cf):
    """(lo, hi): the run of fine-grid points around the minimum whose cost equals it (the PWM cost is a
    step function of the shift: whole frames, G32-R2-2), hi extended by one fine step."""
    i = int(np.argmin(cf))
    flat = cf <= cf[i] + 1e-9 * max(1.0, abs(cf[i]))
    lo = hi = i
    while lo > 0 and flat[lo - 1]:
        lo -= 1
    while hi < len(cf) - 1 and flat[hi + 1]:
        hi += 1
    return float(fine[lo]), float(fine[hi] + FINE)


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


def count_side(theta: Theta, k_run, esp, ie):
    """Feasible set of s (Pi time [us] whose ideal angle matches the counts of the edge line, docstring) on a
    GRID_US grid over the search range: ok with s_lo, s_hi when it is one interval away from the range ends."""
    ts, cnt = esp['timestamp_us'].astype(float), esp['counts']
    T = theta.T
    kk = np.arange(len(T), dtype=float)
    g_lo = float(np.interp(k_run + SEARCH[0], kk, T, left=T[0] + (k_run + SEARCH[0]) * (T[1] - T[0])))
    g_hi = float(np.interp(k_run + SEARCH[1], kk, T))
    grid = np.arange(g_lo, g_hi + GRID_US / 2, GRID_US)
    after = np.arange(ie, len(ts))
    after = after[esp['run_active'][after] == 1]
    D = ts[after] - ts[ie]
    moved = np.nonzero(np.any(cnt[after] != cnt[ie], axis=1))[0]
    d_move = D[moved[0]] if len(moved) else 0.0                      # first line with a count change
    lim = min(d_move + FIT_S * 1e6, T[-1] - grid[-1] - 1000.0)
    keep = D <= lim
    lines, D = after[keep], D[keep]
    out = dict(n_lines=int(len(lines)), fit_span_s=float(D[-1] * 1e-6) if len(D) else 0.0, grid_us=GRID_US,
               search_us=[g_lo, g_hi])
    if lim <= 0:
        return dict(out, ok=False, reason='count side: the frame log ends before the search range does')
    dc = cnt[lines] - cnt[ie]
    if not np.any(dc != 0):
        return dict(out, ok=False, reason='counts never move after the edge (within the fit span)')
    K = COUNTS_PER_REV / (2.0 * math.pi)
    lip = (DRIFT_PPM * 1e-6 * D + J_ESP_US) * 1e-6                     # s, per line
    ratio = np.empty(len(grid))
    for c0 in range(0, len(grid), 1000):
        s = grid[c0:c0 + 1000]
        th0, _ = theta.at(s)                                             # (G, 4)
        th, j = theta.at(s[:, None] + D[None, :])                       # (G, L, 4), (G, L)
        pred = K * (th - th0[:, None, :])
        tol = TOL_COUNTS + K * theta.wmax[j] * lip[None, :, None]
        ratio[c0:c0 + 1000] = np.max(np.abs(dc[None] - pred) / tol, axis=(1, 2))
    feas = ratio <= 1.0
    out.update(best_ratio=float(ratio.min()), s_best=float(grid[int(np.argmin(ratio))]))
    idx = np.nonzero(feas)[0]
    if not len(idx):
        return dict(out, ok=False, reason=f'count side: no Pi time fits every count within the tolerance (best '
                                          f'{ratio.min():.2f} x tolerance)')
    comps = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    out['components_us'] = [[float(grid[c[0]]), float(grid[c[-1]])] for c in comps]
    if len(comps) > 1:
        return dict(out, ok=False, reason=f'count side: {len(comps)} separate feasible intervals (alias)')
    if idx[0] == 0 or idx[-1] == len(grid) - 1:
        return dict(out, ok=False, reason='count side: feasible interval on the edge of the search range')
    return dict(out, ok=True, s_lo=float(grid[idx[0]]), s_hi=float(grid[idx[-1]]))


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
        a, c, grid, cs, fine, cf = _minimise(cost, lo, hi)
        a_lo, a_hi = _plateau(fine, cf)
        P = pred(a) * sgn
        corr = [float(np.corrcoef(P[:, w], D[frames, w])[0, 1]) if np.std(P[:, w]) > 0 and np.std(D[frames, w]) > 0
                else float('nan') for w in range(4)]
        fits[str(W)] = dict(a=a, a_lo=a_lo, a_hi=a_hi, cost=c, corr=corr, sign=[float(s) for s in sgn],
                            n_frames=int(len(frames)),
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
    return dict(out, ok=True, a_lo=min(f['a_lo'] for f in fits.values()), a_hi=max(f['a_hi'] for f in fits.values()),
                t_end_us=max(f['t_end_us'] for f in fits.values()))


def map_edge(omega, duty, k_run: int, esp: dict, period_s: float, t_send_ns=None) -> dict:
    """Plant step of the run_active edge of `esp` (read_esp_log) on a frame log with wheel speeds `omega`
    (n, 4) [rad/s], MISO duty `duty` (n, 4), RUN step k_run and frame send times t_send_ns (None: k x period).
    Returns a dict, ok True / False with reason."""
    period_us = period_s * 1e6
    out = dict(ok=False, reason='', method='count_feasible_set_L98', k_run=int(k_run), period_s=period_s,
               counts_per_rev=COUNTS_PER_REV, omega_min=OMEGA_MIN, frame_times='t_send_ns' if t_send_ns is not None
               else 'k x period', fw=esp.get('fw'),
               constants=dict(fw_counts_at_tick=list(FW_COUNTS_AT_TICK), fit_s=FIT_S, tol_counts=TOL_COUNTS,
                              d_min_us=D_MIN_US, d_max_us=D_MAX_US, j_esp_us=J_ESP_US, grid_us=GRID_US,
                              drift_ppm=DRIFT_PPM, bracket_max_steps=BRACKET_MAX_STEPS,
                              pwm_lag_max_steps=PWM_LAG_MAX_STEPS, pwm_windows_s=list(WINDOWS_S),
                              pwm_cost_max=PWM_COST_MAX, min_corr=MIN_CORR, unique_ratio=UNIQUE_RATIO))
    why = fw_check(esp.get('fw'))
    if why:
        out['reason'] = why
        return out
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
    n = len(omega)
    try:
        T = frame_times_us(n, period_s, t_send_ns)
    except ValueError as exc:
        out['reason'] = str(exc)
        return out
    c = count_side(Theta(T, omega), k_run, esp, ie)
    p = pwm_side(duty, k_run, esp, ie, period_us)
    out.update(count_side=c, pwm_side=p)
    if not c['ok']:
        out['reason'] = c['reason']
        return out
    kk = np.arange(n, dtype=float)
    early_t = c['s_lo'] + D_MIN_US - GRID_US
    late_t = c['s_hi'] + D_MAX_US + GRID_US
    mid_t = 0.5 * (early_t + late_t)
    early, late, a = (float(np.interp(t, T, kk)) for t in (early_t, late_t, mid_t))
    half = 0.5 * (late_t - early_t) * 1e-6
    out.update(early_step=early, late_step=late, edge_step=a, offset_ms=(a - k_run) * period_s * 1e3,
               edge_pi_time_us=mid_t, u_t_s=half, u_t_parts_s=dict(half_bracket=half))
    if late - early > BRACKET_MAX_STEPS:
        out['reason'] = f'bracket {late - early:.2f} steps > {BRACKET_MAX_STEPS}'
    elif not p['ok']:
        out['reason'] = f'PWM cross-check failed: {p["reason"]}'
    elif early > p['a_hi'] + PWM_SLACK_STEPS:
        out['reason'] = (f'PWM cross-check: duty follows the pwm at shift <= {p["a_hi"]:.2f}, before the early '
                         f'bound {early:.2f} (a duty cannot precede its write)')
    elif p['a_lo'] - late > PWM_LAG_MAX_STEPS:
        out['reason'] = (f'PWM cross-check: duty follows the pwm {p["a_lo"] - late:.2f} steps after the late '
                         f'bound (> {PWM_LAG_MAX_STEPS})')
    else:
        out['ok'] = True
    return out


def map_run(frame_log: str, esp_log: str) -> dict:
    """map_edge on the files (frame log of run_hil.py, ESP32 CSV of agv_logger.py)."""
    import hil_o1_score as H
    manifest, mosi, miso = H.load(frame_log)
    state, ref, flags, duty = H.decode_frames(mosi, miso)
    k_run, src = H.command_start(manifest, ref, flags)
    out = map_edge(state[:, 0:4], duty, k_run, read_esp_log(esp_log), float(manifest.get('period_us', 1000)) * 1e-6,
                   frame_send_times(frame_log))
    out.update(k_run_src=src, frame_log=os.path.basename(frame_log), esp_log=os.path.basename(esp_log))
    return out


def frame_send_times(frame_log: str):
    """t_send_ns of a run_hil.py frame log, None when the file has none."""
    with np.load(frame_log, allow_pickle=False) as z:
        return np.array(z['t_send_ns']) if 't_send_ns' in z.files else None


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
