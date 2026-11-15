#!/usr/bin/env python3
"""
c8_u_imu.py: the IMU dynamic part of U_IMU (O1 v11.25 lines 204-206, 247; section 4 "largest observed at C8
plus a fixed margin"), one number per task level. Design CLAUDE_THESIS
03_CODE/docs/2026-11-15_Thiet_ke_C8_U_IMU_v2.md (Liam 15/11 CODE_C13: "Duyet ca 9", "Camera-IMU o C8",
"Them 4 luot RC", "Doi sang 105 ms", "+25 %", "Chi log C8"). Codex Sol round 1 CU-1..CU-23 fixed
(Liam 15/11 CODE_C13 "Sua ca 23 + vong 2").

  U_IMU,dong(m) = U_g + (1 + le) * [ E_thang + beta_max * T_win + (dtau_fit + j_lim) * omega_max
                                     + alpha_max * h_lim^2 / 8 + q ]
  E_thang       = max over runs of (delta_plus * P + delta_minus * N)   (P, N: total CCW, CW turn of the window,
                                                                         every report, scale-corrected yaw)
CODE_CAM adds omega_max * dt_sync (C7b) and writes U_IMU into the C8 gate file.

IMU yaw used here and by the camera scorer (rv_samples, interp_yaw): the game rotation vector reports of the
ESP32 log (rv_n changes; for each report the first row with a finite age >= 0); the firmware time of a report
is the ESP32 RECEIVE time (AGV 5bde4c5 src/imu/imu.cpp 149, 187; imu_fusion.c 226-228; timestamp_us = now_us,
main.c 254, 608). The sensor samples on an even grid of period T0 (50 ms, the fitted period must stay within
+-2 %), so the receive times are put on that grid: the log is cut at gaps above SPLIT_S (1 s), each segment of
at least MIN_SEG reports gets its own index (period searched in +-2 %, chosen by the final fit) and line fit
r = a_s + b_s n (CU5-1, CU6); a report more than j_lim off its grid, or in a segment too short to fit, keeps its
yaw (turn totals) but not its time. Sample time = a_s + b_s n - tau_hat (the yaw at frame time t is read from
this shifted series). Unwrapping: steps between two
reports at most h_lim apart unwrap normally; across a longer gap the branch is chosen by the integral of gz
over the gap (gyro reports at most h_lim apart there), else (or more than pi/2 away) the yaw after the gap is
not usable: interp_yaw gives NaN from there. Each step divided by (1 + s_plus) when positive, by (1 + s_minus)
when negative (F9). A change of reset_n (IMU reset) in a window is refused.
Scale: s = dpsi_raw / (+-N 2 pi) - 1 (signed denominator, design 2.1 after CU-22).

Sign: positive = the IMU yaw increases = 'ccw' (the scale run checks the sign the operator gives).

  python tools/c8_u_imu.py thang --log L1.csv --n-vong 5 --chieu ccw --cach rc --toc-do cham --out L1_thang.json
  python tools/c8_u_imu.py thang-gop --in L1_thang.json ... L8_thang.json --delta-gach-deg 2 --out thang.json
  python tools/c8_u_imu.py troi --log <lifted / static / still logs> ... [--doan all|run_active] [--thang thang.json]
                                --out troi.json
  python tools/c8_u_imu.py tong --muc O1-F08 --c8-log a.csv [--c8-log b.csv] --thang thang.json --troi troi.json
                                (--tre tre.json | --tre-thu TAU_S,DTAU_S) --u-g-deg X --le 0.25 --out u_imu_dong.json
  python tools/c8_u_imu.py tre --cap frames.csv,log.csv,D [--cap ...] --thang thang.json --out tre.json
  python tools/c8_u_imu.py kiem --frames frames.csv --log log.csv --offset-s D --u-imu u_imu_dong.json --muc O1-F08
                                --u-psi-deg X [--u-ghep-deg Y] --out kiem.json
frames.csv of c7_measure.py (CODE_CAM 15/11 through CODE): t_s (camera clock, s), psi (camera yaw, floor frame,
wrapped), wz, valid, car_seen; t_log = t_s + D (o1_time_sync or --offset-s). `tre` (F3): tau_hat from the
camera-IMU yaw on the C8 runs, a time constant only. `kiem` (design 5): back check and closure; not OK means
U_IMU is not locked. Before the camera fit, `tong --tre-thu` marks the file "thu": true (not lockable).
The camera scorer uses S = rv_samples(load_log(path), tau_hat=, s_plus=, s_minus=, j_lim=, h_lim=) and
interp_yaw(S, t_log) with the numbers of u_imu_dong.json.
A C8 log for a level: one contiguous run_active part, at least T_cmd of the level long (rpi5/o1_tasks.py), IMU
and gyro reports covering the window [t_on - pre_s, t_off + tail_s]. Paths (inputs, outputs, temporary files)
with a part sealed, sealed_test or O1_FEAS are refused. Outputs are never overwritten (`tong` adds a level to an
existing file only when every shared parameter is equal).
Last line: #ket_qua,ma=0 done / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VERSION = 'u_imu_dong_v2'
PERIOD_S = 0.05            # report period of the campaign firmware: BNO085 20 Hz (C4, L73)
PERIOD_TOL = 0.02          # CU-1: the fitted period must stay within +-2 % of PERIOD_S
J_LIM_S = 0.005            # design 3 step 2 [DE XUAT]
H_LIM_S = 0.105            # Liam 15/11 "Doi sang 105 ms"
LE = 0.25                  # Liam 15/11 "+25 %"
MIN_STILL_S = 3.0          # design 2.1 / 9: >= 3 s still, on the gyro report times (CU-16)
MAX_LATE = 0.2             # CU-1: at most 20 % of the reports off the grid
SPLIT_S = 1.0              # CU6: no grid index across a gap of more than 1 s (20 reports)
MIN_SEG = 5                # CU6: a segment needs at least 5 reports for a fitted grid
MIN_PERIOD_SEG = 20        # CU7-2: the period is measured only on segments of at least 20 reports
END_STILL_S = 3.0          # kiem closure: still frames in [t_off + tail, t_off + tail + 3 s] (CU-14)
T_CMD_TOL_S = 0.1          # CU-8: run_active at least T_cmd - 0.1 s
FORBIDDEN = {'sealed', 'sealed_test', 'o1_feas'}
LOG_COLS = ('timestamp_us', 'yaw_rad', 'rv_n', 'age_r_ms', 'gz', 'gyro_n', 'age_g_ms', 'reset_n')
SCALE_RUNS = {('rc', 'cham'), ('rc', 'nhanh'), ('tay', 'cham'), ('tay', 'nhanh')}     # design 2.1 (CU-15)
HAND_RUNS = {('tay', 'cham'), ('tay', 'nhanh')}


class InputError(Exception):
    pass


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def _finite(x, what, lo=None, hi=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise InputError(f'{what}: {x!r} is not a finite number')
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise InputError(f'{what}: {x!r} outside [{lo}, {hi}]')
    return float(x)


# ── input ───────────────────────────────────────────────────────────────────────────────

def check_path(path):
    parts = [p.lower() for p in os.path.normpath(os.path.abspath(path)).replace('\\', '/').split('/')]
    bad = FORBIDDEN.intersection(parts)
    if bad:
        raise InputError(f'{path}: path part {sorted(bad)} is behind the fence (S05 2.1, O1 3.9): refused')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _drop_duplicates(path, rows):
    """Same rule as tools/c5_lifted_check.py drop_duplicates (S05 6.3): drop rows the logger marked
    'duplicate' in <log>.segments.csv after checking each equals an earlier kept row."""
    seg = path + '.segments.csv'
    if not os.path.isfile(seg):
        return rows, None
    with open(seg, newline='', encoding='utf-8') as f:
        marks = list(csv.DictReader(f))
    if len(marks) != len(rows):
        raise InputError(f'{seg}: {len(marks)} rows for {len(rows)} data lines')
    keep, seen, n = [], set(), 0
    for i, (r, m) in enumerate(zip(rows, marks), start=1):
        if m.get('row') != str(i) or m.get('log_seq') != r.get('log_seq'):
            raise InputError(f'{seg} row {i} does not match CSV data line {i}')
        key = tuple(r.items())
        if m.get('status') == 'duplicate':
            if key not in seen:
                raise InputError(f'{seg} row {i}: marked duplicate but no earlier equal line')
            n += 1
            continue
        seen.add(key)
        keep.append(r)
    return keep, n


def load_log(path, need=LOG_COLS):
    """An agv_logger CSV as float arrays (rows sorted by timestamp_us, duplicates dropped)."""
    check_path(path)
    if not os.path.isfile(path):
        raise InputError(f'no log {path}')
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    if not rows:
        raise InputError(f'{path}: no data line')
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise InputError(f'{path}: no column {miss}')
    rows, n_dup = _drop_duplicates(path, rows)
    cols = set(need) | ({'run_active', 'roll', 'pitch', 'r_valid', 'g_valid'} & set(rows[0]))
    try:
        L = {c: np.array([float(r[c]) if r[c] not in ('', None) else np.nan for r in rows]) for c in cols}
    except ValueError as exc:
        raise InputError(f'{path}: {exc}')
    order = np.argsort(L['timestamp_us'], kind='stable')
    for c in cols:
        L[c] = L[c][order]
    if not np.all(np.isfinite(L['timestamp_us'])) or np.any(np.diff(L['timestamp_us']) <= 0):
        raise InputError(f'{path}: timestamp_us not finite or two rows with the same timestamp_us')
    L['n_dup'] = n_dup or 0
    L['path'] = path
    return L


# ── IMU samples (shared with the camera scorer) ─────────────────────────────────────────────────

def report_rows(n, age):
    """CU-7: one row per report: the first row of each new count value with a finite age >= 0."""
    idx, cur, want = [], None, False
    for i in range(len(n)):
        if not np.isfinite(n[i]) or n[i] <= 0:
            continue
        if n[i] != cur:
            if want:
                raise InputError(f'report count {cur:.0f}: no row with a finite age >= 0')
            cur, want = n[i], True
        if want and np.isfinite(age[i]) and age[i] >= 0:
            idx.append(i)
            want = False
    if want:
        raise InputError(f'report count {cur:.0f}: no row with a finite age >= 0')
    return np.array(idx, dtype=int)


def _index(r, b):
    """Grid index of each receive time for period b, stepping from the last report that sat on the grid."""
    n = np.zeros(len(r), dtype=int)
    i_lt = 0
    for k in range(1, len(r)):
        m = max(1, int(round((r[k] - r[i_lt]) / b)))
        n[k] = n[i_lt] + m
        if abs(r[k] - (r[i_lt] + m * b)) <= 0.25 * b:
            i_lt = k
    return n


def _fit(r, n, j_lim):
    """Iterative line fit r = a + b n on the reports within j_lim (as before CU6): a, b, residual j, inliers."""
    inl = np.ones(len(r), dtype=bool)
    a = b = float('nan')
    j = np.full(len(r), np.nan)
    for _ in range(10):
        if inl.sum() < 2:
            return a, b, j, np.zeros(len(r), dtype=bool)
        b, a = np.polyfit(n[inl], r[inl], 1)
        j = r - (a + b * n)
        new = np.abs(j) <= j_lim
        if np.array_equal(new, inl):
            break
        inl = new
    return a, b, j, inl


def _grid_segment(r, period_s, j_lim):
    """One segment (no gap above SPLIT_S): try every indexing period in +-PERIOD_TOL of T0 (step 0.01 ms, CU5-1)
    and keep the one whose FINAL fit (CU6-2) has its period inside +-PERIOD_TOL (CU6-3), then the fewest reports
    off the grid, then the smallest RMS of the others."""
    best = None
    for b_try in np.arange(period_s * (1 - PERIOD_TOL), period_s * (1 + PERIOD_TOL) + 5e-6, 1e-5):
        n_try = _index(r, b_try)
        a, b, j, inl = _fit(r, n_try, j_lim)
        bad_period = not (abs(b / period_s - 1.0) <= PERIOD_TOL)
        rms = float(np.sqrt(np.mean(j[inl] ** 2))) if inl.any() else math.inf
        key = (bad_period, int((~inl).sum()), rms)
        if best is None or key < best[0]:
            best = (key, n_try, a, b, j, inl)
    return best[1:]


def _grid(r, period_s, j_lim):
    """Sample times of the reports from their receive times. CU6: the log is cut into segments at gaps above
    SPLIT_S (no grid index is carried across such a gap: with the period known only to +-PERIOD_TOL the index step
    over a long gap is not determined).
    - The period comes from segments of at least MIN_PERIOD_SEG reports (_grid_segment, own line fit, period
      checked); b = the period of the longest of them (of the longest segment of at least MIN_SEG reports when
      none is that long).
    - A segment of MIN_SEG .. MIN_PERIOD_SEG - 1 reports takes that period and fits only its offset (CU7-2: a
      period from a few jittered reports is not trusted nor checked).
    - A segment under MIN_SEG reports has no fit: its reports are off the grid (kept only for turn totals), but
      it is still indexed with that period so missing reports inside it show as steps > 1 (CU7-1).
    Returns n (index, increasing, a step > 1 marks missing reports; across a cut at least 2), t_fit (fitted sample
    time, or the receive time when not fitted), b, j (residual, NaN when not fitted), inl. At most MAX_LATE of all
    reports may be off the grid (CU-1)."""
    cuts = np.nonzero(np.diff(r) > SPLIT_S)[0] + 1
    segs = list(zip(np.r_[0, cuts], np.r_[cuts, len(r)]))
    n = np.zeros(len(r), dtype=int)
    t_fit = r.astype(float).copy()
    j = np.full(len(r), np.nan)
    inl = np.zeros(len(r), dtype=bool)
    lens = [e - s0 for s0, e in segs]
    src = [k for k, L in enumerate(lens) if L >= MIN_PERIOD_SEG]
    if not src:
        src = [int(np.argmax(lens))] if max(lens) >= MIN_SEG else []
    if not src:
        raise InputError(f'no run of at least {MIN_SEG} IMU reports without a gap above {SPLIT_S:.1f} s (CU-1)')
    full, b_main = {}, None
    for k in src:
        s0, e = segs[k]
        ns, a, bk, js, ok = _grid_segment(r[s0:e], period_s, j_lim)
        if not (abs(bk / period_s - 1.0) <= PERIOD_TOL):
            raise InputError(f'fitted report period {bk * 1e3:.2f} ms outside {period_s * 1e3:.0f} ms '
                             f'+-{PERIOD_TOL:.0%} (CU-1)')
        full[k] = (ns, a + bk * ns, js, ok)
    b_main = None
    for k in sorted(src, key=lambda k: -lens[k]):
        s0, e = segs[k]
        b_main = float(np.polyfit(full[k][0][full[k][3]], r[s0:e][full[k][3]], 1)[0]) if full[k][3].sum() >= 2 else None
        if b_main is not None:
            break
    if b_main is None:
        raise InputError('no segment with a usable period (CU-1)')
    offset = 0
    for k, (s0, e) in enumerate(segs):
        rs = r[s0:e]
        if k in full:
            ns, tf, js, ok = full[k]
            t_fit[s0:e], j[s0:e], inl[s0:e] = tf, js, ok
        else:
            ns = _index(rs, b_main)                              # CU7-1: missing reports inside show as steps > 1
            if len(rs) >= MIN_SEG:                               # CU7-2: offset only, period from the long segments
                ok = np.ones(len(rs), dtype=bool)
                for _ in range(10):
                    a = float(np.mean(rs[ok] - b_main * ns[ok]))
                    js = rs - (a + b_main * ns)
                    new = np.abs(js) <= j_lim
                    if np.array_equal(new, ok) or new.sum() == 0:
                        break
                    ok = new
                ok = np.abs(js) <= j_lim
                tf = a + b_main * ns
                if not ok.all():                                 # CU8-2: its own grid if that fits more reports
                    n2, a2, b2, j2, ok2 = _grid_segment(rs, period_s, j_lim)
                    if abs(b2 / period_s - 1.0) <= PERIOD_TOL and ok2.sum() > ok.sum():
                        ns, tf, js, ok = n2, a2 + b2 * n2, j2, ok2
                t_fit[s0:e], j[s0:e], inl[s0:e] = tf, js, ok
        n[s0:e] = offset + ns
        if e < len(r):
            offset = n[e - 1] + max(2, int(round((r[e] - r[e - 1]) / b_main)))
    if (~inl).sum() > MAX_LATE * len(r):
        raise InputError(f'{(~inl).sum()} of {len(r)} reports more than {j_lim * 1e3:.0f} ms off the grid (CU-1)')
    return n, t_fit, b_main, j, inl


def gyro_samples(L):
    """Gyro reports (receive times, gz); with a g_valid column, reports at rows with g_valid != 1 are dropped
    (firmware: age, settle time after a reset, CU-4)."""
    idx = report_rows(L['gyro_n'], L['age_g_ms'])
    if 'g_valid' in L:
        idx = idx[L['g_valid'][idx] == 1]
    t = L['timestamp_us'][idx] * 1e-6 - L['age_g_ms'][idx] * 1e-3
    if np.any(np.diff(t) <= 0):
        raise InputError(f'{L.get("path")}: gyro receive times not increasing')
    return t, L['gz'][idx]


def _gz_integral(tg, gz, a, b, h_lim):
    """Integral of gz over [a, b] (trapezoid, interpolated ends); None without gyro reports at most h_lim apart."""
    k = np.nonzero((tg > a) & (tg < b))[0]
    if tg.size == 0 or tg[0] > a or tg[-1] < b:
        return None
    t = np.concatenate([[a], tg[k], [b]])
    g = np.concatenate([[np.interp(a, tg, gz)], gz[k], [np.interp(b, tg, gz)]])
    i0, i1 = np.searchsorted(tg, a, 'right') - 1, np.searchsorted(tg, b, 'left')
    if np.max(np.diff(tg[max(i0, 0):i1 + 1])) > h_lim:
        return None
    return float(np.sum((g[1:] + g[:-1]) / 2 * np.diff(t)))


def rv_samples(L, period_s=PERIOD_S, tau_hat=0.0, s_plus=0.0, s_minus=0.0, j_lim=J_LIM_S, h_lim=H_LIM_S):
    """Game rotation vector samples of a log. Every report: t_all (grid sample time minus tau_hat, ESP32 clock),
    psi_all (unwrapped, scale-corrected yaw), n_all (grid index), t_recv; kept reports (on the grid): t, psi.
    t_break (same clock as t): the yaw is not usable from the last report before a gap that could not be
    unwrapped (CU-3, CU2-2) or before an IMU reset (CU-4); resets: row times where reset_n changed.
    Also tg, gz: the gyro reports (receive times), for turn_totals."""
    for v, w in ((tau_hat, 'tau_hat'), (s_plus, 's_plus'), (s_minus, 's_minus'), (j_lim, 'j_lim'), (h_lim, 'h_lim')):
        _finite(v, w)
    if not (-0.1 < s_plus < 0.1 and -0.1 < s_minus < 0.1):
        raise InputError('scale correction outside +-10 %')
    idx = report_rows(L['rv_n'], L['age_r_ms'])
    if idx.size < 2:
        raise InputError(f'{L.get("path")}: fewer than two rotation vector reports')
    r = L['timestamp_us'][idx] * 1e-6 - L['age_r_ms'][idx] * 1e-3
    if np.any(np.diff(r) <= 0):
        raise InputError(f'{L.get("path")}: rotation vector receive times not increasing')
    y = L['yaw_rad'][idx]
    if not np.all(np.isfinite(y)):
        raise InputError(f'{L.get("path")}: yaw_rad not finite')
    n, t_fit, b, j, keep = _grid(r, period_s, j_lim)
    valid = (L['r_valid'][idx] == 1) if 'r_valid' in L else np.ones(len(idx), dtype=bool)
    keep = keep & valid                                    # CU-4: firmware settle / age flag
    t_all = t_fit - tau_hat
    tg, gz = gyro_samples(L) if np.any(L['gyro_n'] > 0) else (np.array([]), np.array([]))
    d0 = (np.diff(y) + math.pi) % (2 * math.pi) - math.pi
    d = d0.copy()
    t_break = math.inf
    for k in np.nonzero(np.diff(r) > h_lim)[0]:              # CU-3: branch over a long gap from the gyro
        ig = _gz_integral(tg, gz, r[k], r[k + 1], h_lim)
        if ig is not None:
            m = round((ig - d0[k]) / (2 * math.pi))
            d[k] = d0[k] + 2 * math.pi * m
        if ig is None or abs(d[k] - ig) > math.pi / 2:
            t_break = min(t_break, t_all[k])                  # CU2-2: from the report BEFORE the gap
    rs = L['reset_n']
    ch = np.nonzero(np.diff(rs) != 0)[0] + 1
    resets = [float(L['timestamp_us'][i] * 1e-6) for i in ch]
    for tr in resets:                                        # CU-4: no yaw from the last report before a reset
        k = int(np.searchsorted(r, tr, 'left')) - 1
        t_break = min(t_break, t_all[max(k, 0)])
    d = np.where(d > 0, d / (1.0 + s_plus), d / (1.0 + s_minus))
    psi = y[0] + np.concatenate([[0.0], np.cumsum(d)])
    return dict(t=t_all[keep], psi=psi[keep], t_all=t_all, psi_all=psi, n_all=n, t_recv=r, j=j, keep=keep,
                n_samples=int(idx.size), n_late=int((~keep & valid).sum()), n_invalid=int((~valid).sum()),
                valid=valid, period_fit_s=float(b), rows=idx, h_lim=h_lim,
                t_break=t_break, resets=resets, tg=tg, gz=gz)


def interp_yaw(S, t, h_lim=None):
    """IMU yaw at times t (same clock as S['t']): linear between two kept reports at most h_lim apart, both at or
    before S['t_break'] (the last usable report); NaN otherwise."""
    h_lim = S.get('h_lim', H_LIM_S) if h_lim is None else h_lim
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ts, ps = S['t'], S['psi']
    tb = S.get('t_break', math.inf)
    out = np.full(t.shape, np.nan)
    i = np.searchsorted(ts, t, side='right')
    ok = (i > 0) & (i < len(ts)) & (t <= tb)
    ok[ok] = ts[i[ok]] <= tb                                  # CU-4: the later end too
    i0, i1 = i[ok] - 1, i[ok]
    gap = ts[i1] - ts[i0]
    w = (t[ok] - ts[i0]) / gap
    v = np.where(gap <= h_lim + 1e-9, ps[i0] + w * (ps[i1] - ps[i0]), np.nan)
    out[ok] = v
    exact = np.isin(t, ts) & (t <= tb)
    if exact.any():
        out[exact] = np.interp(t[exact], ts, ps)
    return out


def turn_totals(S, w0, w1):
    """Total CCW (P) and CW (N) turn over [w0, w1], from the yaw of EVERY report (CU-2). CU9-1: the grid is not
    trusted here. Each report has a time span [min(sample, receive), max(sample, receive)] (both bases, CU2-1);
    a pair of consecutive reports counts when its span from the first report's start to the second report's end
    meets the window; the sample times of the reports must cover the window (CU10-1: a receive time after the
    window end does not stand for a sample). A pair is suspect, and then both P and
    N get max|gz| over its span times the span length (a turn back and forth can hide there, CU-2), when: the grid
    index steps by more than 1, the receive or sample gap is above 1.5 periods (CU8-1), or either report is off
    the grid (not kept: its sample time is not trusted, CU9-1). That needs gyro reports at most h_lim apart over
    the span (else refused); the padding of one period only searches the largest |gz| (CU4-2)."""
    ta, ps, na, tr = S['t_all'], S['psi_all'], S['n_all'], S['t_recv']
    keep = S.get('keep', np.ones(len(ta), dtype=bool))
    lo, hi = np.minimum(ta, tr), np.maximum(ta, tr)
    if ta[0] > w0 or ta[-1] < w1:                          # CU10-1: covered on the SAMPLE-time base (the yaw's)
        raise InputError('IMU samples do not cover the window')
    tg, gz = S.get('tg', np.array([])), S.get('gz', np.array([]))
    per = S.get('period_fit_s', PERIOD_S)
    P = N = 0.0
    for k in range(len(ta) - 1):
        a, b = lo[k], hi[k + 1]
        if b < w0 or a > w1:
            continue
        d = ps[k + 1] - ps[k]
        if d > 0:
            P += d
        else:
            N -= d
        suspect = (na[k + 1] - na[k] > 1 or max(tr[k + 1] - tr[k], ta[k + 1] - ta[k]) > 1.5 * per
                   or not keep[k] or not keep[k + 1])
        if suspect:
            if tg.size < 2 or tg[0] > a or tg[-1] < b:
                raise InputError('suspect IMU reports in the window without gyro reports over the gap (CU-2)')
            j0, j1 = np.searchsorted(tg, a, 'right') - 1, np.searchsorted(tg, b, 'left')
            if np.max(np.diff(tg[j0:j1 + 1])) > S.get('h_lim', H_LIM_S):
                raise InputError('suspect IMU reports in the window and gyro reports too far apart (CU-2)')
            sel = (tg >= a - per) & (tg <= b + per)
            sel[j0:j1 + 1] = True
            extra = float(np.max(np.abs(gz[sel]))) * (b - a)
            P += extra
            N += extra
    return P, N


def alpha_max(tg, gz):
    """CU-5: largest |dgz / dt| between consecutive gyro reports (receive times, no floor)."""
    dt = np.diff(tg)
    if dt.size == 0:
        return 0.0
    if np.any(dt <= 0):
        raise InputError('gyro receive times not increasing')
    return float(np.max(np.abs(np.diff(gz)) / dt))


def _check_no_reset(S, a, b, what):
    bad = [x for x in S['resets'] if a <= x <= b]
    if bad:
        raise InputError(f'{what}: IMU reset (reset_n changed) at {bad[0]:.3f} s inside the window (CU-4)')


def run_window(L):
    """CU-8: the one contiguous run_active part (t_on, t_off)."""
    if 'run_active' not in L or not (L['run_active'] == 1).any():
        raise InputError(f'{L.get("path")}: no run_active row (t = 0 of the run)')
    on = np.nonzero(L['run_active'] == 1)[0]
    if np.any(np.diff(on) != 1):
        raise InputError(f'{L.get("path")}: more than one run_active part: one run per file (CU-8)')
    if on[-1] == len(L['run_active']) - 1:
        raise InputError(f'{L.get("path")}: the log ends while run_active is 1: no end edge, log cut? (CU-8)')
    return L['timestamp_us'][on[0]] * 1e-6, L['timestamp_us'][on[-1]] * 1e-6


def t_cmd(muc):
    import o1_tasks                     # rpi5/o1_tasks.py, read only
    try:
        return float(o1_tasks.task(muc, allow_final_set=True)[0])
    except ValueError as exc:
        raise InputError(f'--muc {muc!r}: {exc}')


# ── scale run (design 2.1) ───────────────────────────────────────────────────────────────

def _still_ends(tg, gz, g_still, min_still_s):
    still = np.abs(gz) < g_still
    if not still[0] or not still[-1]:
        raise InputError('the log does not start and end with the car still (|gz| < %.3f rad/s)' % g_still)
    i = 0
    while i < len(still) and still[i]:
        i += 1
    k = len(still) - 1
    while k >= 0 and still[k]:
        k -= 1
    if i >= len(still) or k < 0:
        raise InputError('no rotation in the log')
    lead, trail = (tg[0], tg[i - 1]), (tg[k + 1], tg[-1])
    for name, (a, b) in (('start', lead), ('end', trail)):
        if b - a < min_still_s:
            raise InputError(f'still part at the {name} is {b - a:.2f} s, need >= {min_still_s} s (design 9)')
    return lead, trail, (tg[i], tg[k])


def thang(path, n_vong, chieu, period_s=PERIOD_S, g_still=0.02, min_still_s=MIN_STILL_S, edge_s=0.2):
    """Scale error of one run of n_vong turns between two still parts: s = dpsi / (+-n 2 pi) - 1."""
    if chieu not in ('ccw', 'cw'):
        raise InputError(f'--chieu {chieu!r}: ccw or cw')
    if not isinstance(n_vong, int) or n_vong < 1:
        raise InputError('--n-vong: an integer >= 1')
    L = load_log(path)
    S = rv_samples(L, period_s=period_s)
    if S['resets']:
        raise InputError(f'IMU reset in the scale run at {S["resets"][0]:.3f} s (CU-4)')
    if math.isfinite(S['t_break']):
        raise InputError('a gap in the IMU reports could not be unwrapped (CU-3)')
    tg, gz = gyro_samples(L)
    lead, trail, rot = _still_ends(tg, gz, g_still, min_still_s)
    t_all = S['t_recv']
    a = (t_all >= lead[0]) & (t_all <= lead[1] - edge_s)
    b = (t_all >= trail[0] + edge_s) & (t_all <= trail[1])
    if a.sum() < 5 or b.sum() < 5:
        raise InputError('too few IMU reports in a still part')
    dpsi = float(np.median(S['psi_all'][b]) - np.median(S['psi_all'][a]))
    exp = (1 if chieu == 'ccw' else -1) * n_vong * 2 * math.pi
    if not math.isfinite(dpsi) or dpsi * exp <= 0:
        raise InputError(f'IMU yaw changed by {math.degrees(dpsi):+.1f} deg: not the direction {chieu}')
    s = dpsi / exp - 1.0
    if not abs(s) <= 0.1:
        raise InputError(f'scale {s:+.3f}: more than 10 %, turn count wrong? (dpsi {math.degrees(dpsi):.1f} deg)')
    rot_mask = (tg >= rot[0]) & (tg <= rot[1])
    return dict(s=s, chieu=chieu, n_vong=n_vong, dpsi_rad=dpsi, dpsi_deg=math.degrees(dpsi),
                still_start_s=list(lead), still_end_s=list(trail),
                omega_median=float(np.median(np.abs(gz[rot_mask]))) if rot_mask.any() else None,
                n_reports=S['n_samples'], n_late=S['n_late'], n_dup=L['n_dup'])


def thang_gop(runs, delta_gach_deg=2.0):
    """CU-15: each direction needs RC slow, RC fast, hand slow, hand fast (F4 closed), or hand slow and hand fast
    only (F4 left open). s_hat = mean of the RC runs (of the hand runs when there is none); delta = alignment
    error / (n 2 pi) + largest |s_i - s_hat| over every run of that direction."""
    _finite(delta_gach_deg, '--delta-gach-deg', 0.0, 45.0)
    out = dict(delta_gach_deg=delta_gach_deg, f4_mo=False)
    for chieu, key in (('ccw', 'plus'), ('cw', 'minus')):
        rr = [r for r in runs if r.get('chieu') == chieu]
        for r in rr:
            _finite(r.get('s'), f'scale run s ({chieu})', -0.1, 0.1)
            if not isinstance(r.get('n_vong'), int) or r['n_vong'] < 1:
                raise InputError(f'scale run n_vong {r.get("n_vong")!r}')
        have = {(r.get('cach'), r.get('toc_do')) for r in rr}
        if have == SCALE_RUNS:
            base = [r for r in rr if r['cach'] == 'rc']
        elif have == HAND_RUNS:
            base = rr
            out['f4_mo'] = True
        else:
            raise InputError(f'direction {chieu}: scale runs {sorted(have)}; need {sorted(SCALE_RUNS)} or, without RC, '
                             f'{sorted(HAND_RUNS)} (design 2.1, CU-15)')
        s_hat = float(np.mean([r['s'] for r in base]))
        g = max(math.radians(delta_gach_deg) / (r['n_vong'] * 2 * math.pi) for r in rr)
        out['s_' + key] = s_hat
        out['delta_' + key] = g + max(abs(r['s'] - s_hat) for r in rr)
        out['n_' + key] = dict(rc=sum(r['cach'] == 'rc' for r in rr), tay=sum(r['cach'] == 'tay' for r in rr))
    return out


# ── drift and resolution (design 2.2) ──────────────────────────────────────────────────────

def _segments(mask):
    k = np.nonzero(mask)[0]
    if k.size == 0:
        return []
    cut = np.nonzero(np.diff(k) != 1)[0]
    starts = np.concatenate([[k[0]], k[cut + 1]])
    ends = np.concatenate([k[cut], [k[-1]]])
    return list(zip(starts, ends))


def _yaw_parts(path, doan, period_s, s_plus, s_minus):  # -> [(t, psi, n_late)]
    """Parts (t, psi) of a file to measure drift on: whole file, or each contiguous run_active part (CU-9)."""
    check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        head = f.readline()
    if head.startswith('kind,'):                       # tools/imu_capture.py raw_<phase>.csv
        if doan != 'all':
            raise InputError(f'{path}: an imu_capture raw file has no run_active')
        with open(path, newline='', encoding='utf-8') as f:
            R = [r for r in csv.DictReader(f) if r['kind'] == 'R']
        if len(R) < 2:
            raise InputError(f'{path}: no R (game RV) rows')
        t = np.array([float(r['t_us']) for r in R]) * 1e-6
        y = np.radians([float(r['v2']) for r in R])
        d = np.diff(np.unwrap(y))
        d = np.where(d > 0, d / (1.0 + s_plus), d / (1.0 + s_minus))          # CU-10
        return [(t, y[0] + np.concatenate([[0.0], np.cumsum(d)]), None)]
    L = load_log(path)
    S = rv_samples(L, period_s=period_s, s_plus=s_plus, s_minus=s_minus)
    if S['resets'] or math.isfinite(S['t_break']):
        raise InputError(f'{path}: IMU reset or a gap that could not be unwrapped')
    t, psi = S['t_recv'], S['psi_all']
    if doan == 'all':
        return [(t, psi, S['n_late'])]
    if 'run_active' not in L:
        raise InputError(f'{path}: no run_active column')
    parts = [(t[a:b + 1], psi[a:b + 1], S['n_late']) for a, b in _segments(L['run_active'][S['rows']] == 1) if b > a]
    if not parts:
        raise InputError(f'{path}: no run_active part')
    return parts


def troi(paths, period_s=PERIOD_S, min_dt=2.0, doan='all', block_s=1.0, s_plus=0.0, s_minus=0.0):
    """beta = largest |psi(t) - psi(t0)| / (t - t0) with t - t0 >= min_dt (t0 = start of each part);
    q = largest |yaw - median of its 1 s block| (blocks of >= 2 reports, CU-11). Over every file and part."""
    _finite(min_dt, '--min-dt', 0.5)
    per, beta, q = [], 0.0, 0.0
    for p in paths:
        for t, psi, n_late in _yaw_parts(p, doan, period_s, s_plus, s_minus):
            dt = t - t[0]
            m = dt >= min_dt
            b = float(np.max(np.abs(psi[m] - psi[0]) / dt[m])) if m.any() else float('nan')
            blk = np.floor(dt / block_s).astype(int)
            qq = 0.0
            for k in np.unique(blk):
                v = psi[blk == k]
                if v.size >= 2:
                    qq = max(qq, float(np.max(np.abs(v - np.median(v)))))
            per.append(dict(file=os.path.basename(p), duration_s=float(dt[-1]), beta=b, q=qq, n_late=n_late))
            if np.isfinite(b):
                beta = max(beta, b)
            q = max(q, qq)
    if not any(np.isfinite(x['beta']) for x in per):
        raise InputError(f'no part longer than {min_dt} s')
    for x in per:
        if not np.isfinite(x['beta']):
            x['beta'] = None
    return dict(beta_max=beta, q=q, min_dt_s=min_dt, doan=doan, s_plus=s_plus, s_minus=s_minus, files=per)


# ── total for one level (design 1) ─────────────────────────────────────────────────────────

SHARED = ('version', 'le', 'U_g', 'period_s', 'j_lim_s', 'h_lim_s', 'tau_hat_s', 'dtau_fit_s', 's_plus', 's_minus',
          'delta_plus', 'delta_minus', 'beta_max', 'q', 'thu', 'f4_mo', 'pre_s', 'tail_s')


def _check_thang(d):
    for k in ('s_plus', 's_minus'):
        _finite(d.get(k), f'thang {k}', -0.1, 0.1)
    for k in ('delta_plus', 'delta_minus'):
        _finite(d.get(k), f'thang {k}', 0.0, 0.1)


def _level_numbers(path, muc, tre, thang_d, period_s, j_lim, h_lim, pre_s, tail_s):
    L = load_log(path)
    t_on, t_off = run_window(L)
    need = t_cmd(muc) - T_CMD_TOL_S
    if t_off - t_on < need:
        raise InputError(f'{path}: run_active {t_off - t_on:.2f} s shorter than T_cmd of {muc} ({need:.2f} s): '
                         f'log cut? (CU-8)')
    w0, w1 = t_on - pre_s, t_off + tail_s
    S = rv_samples(L, period_s=period_s, tau_hat=tre['tau_hat_s'], s_plus=thang_d['s_plus'],
                   s_minus=thang_d['s_minus'], j_lim=j_lim, h_lim=h_lim)
    _check_no_reset(S, w0, w1, path)
    inv = S['t_all'][~S['valid']]
    if np.any((inv >= w0) & (inv <= w1)):
        raise InputError(f'{path}: IMU reports flagged not valid by the firmware (r_valid) inside the window (CU-4)')
    if S['t_break'] < w1:
        raise InputError(f'{path}: a gap in the IMU reports could not be unwrapped inside the window (CU-3)')
    P, N = turn_totals(S, w0, w1)
    tg, gz = gyro_samples(L) if np.any(L['gyro_n'] > 0) else (np.array([]), np.array([]))
    if tg.size < 2 or tg[0] > w0 or tg[-1] < w1:
        raise InputError(f'{path}: gyro reports do not cover the window (CU-6)')
    i0, i1 = np.searchsorted(tg, w0, 'right') - 1, np.searchsorted(tg, w1, 'left')
    tw, gw = tg[i0:i1 + 1], gz[i0:i1 + 1]
    if not np.all(np.isfinite(gw)) or np.max(np.diff(tw)) > h_lim:
        raise InputError(f'{path}: gyro reports in the window not finite or more than h_lim apart (CU-6)')
    eps = None
    if 'roll' in L and 'pitch' in L:                    # CU-21: tilt change against the still part before t = 0
        tt = L['timestamp_us'] * 1e-6
        pre = (tt >= w0) & (tt < t_on)
        win = (tt >= w0) & (tt <= w1)
        if pre.sum() >= 3:
            eps = float(max(np.nanmax(np.abs(L['roll'][win] - np.nanmedian(L['roll'][pre]))),
                            np.nanmax(np.abs(L['pitch'][win] - np.nanmedian(L['pitch'][pre])))))
    return dict(P=P, N=N, E=thang_d['delta_plus'] * P + thang_d['delta_minus'] * N, T_win=float(w1 - w0),
                omega=float(np.max(np.abs(gw))), alpha=alpha_max(tw, gw), eps_max_deg=eps,
                n_late=S['n_late'], n_reports=S['n_samples'], file=os.path.basename(path), sha256=sha256(path))


def tong(muc, logs, thang_d, troi_d, tre, u_g, le, out, period_s=PERIOD_S, j_lim=J_LIM_S, h_lim=H_LIM_S,
         pre_s=2.0, tail_s=None):
    import o1_tasks
    tail_s = o1_tasks.O1_TAIL_S if tail_s is None else tail_s
    check_path(out)
    _check_thang(thang_d)
    for k in ('beta_max', 'q'):
        _finite(troi_d.get(k), f'troi {k}', 0.0, 0.1)
    if (troi_d.get('s_plus', 0.0), troi_d.get('s_minus', 0.0)) != (thang_d['s_plus'], thang_d['s_minus']):
        raise InputError('troi was computed with another scale correction than thang (CU-10)')
    _finite(tre.get('tau_hat_s'), 'tau_hat_s', -0.2, 0.2)
    _finite(tre.get('dtau_fit_s'), 'dtau_fit_s', 0.0, 0.2)
    for v, w in ((u_g, 'U_g'), (le, 'le'), (pre_s, 'pre_s')):
        _finite(v, w, 0.0)
    _finite(tail_s, 'tail_s', 0.5)                                       # CU-8
    if not logs:
        raise InputError('no C8 log for the level')
    nums = [_level_numbers(p, muc, tre, thang_d, period_s, j_lim, h_lim, pre_s, tail_s) for p in logs]
    E = max(x['E'] for x in nums)                                      # CU-20: per run, then the largest
    T = max(x['T_win'] for x in nums)
    om = max(x['omega'] for x in nums)
    al = max(x['alpha'] for x in nums)
    phan = dict(thang=E, troi=troi_d['beta_max'] * T, thoi_gian=(tre['dtau_fit_s'] + j_lim) * om,
                noi_suy=al * h_lim ** 2 / 8.0, q=troi_d['q'])
    u = u_g + (1.0 + le) * sum(phan.values())
    shared = dict(version=VERSION, le=le, U_g=u_g, period_s=period_s, j_lim_s=j_lim, h_lim_s=h_lim,
                  tau_hat_s=tre['tau_hat_s'], dtau_fit_s=tre['dtau_fit_s'], s_plus=thang_d['s_plus'],
                  s_minus=thang_d['s_minus'], delta_plus=thang_d['delta_plus'], delta_minus=thang_d['delta_minus'],
                  beta_max=troi_d['beta_max'], q=troi_d['q'], thu=bool(tre.get('thu')),
                  f4_mo=bool(thang_d.get('f4_mo')), pre_s=pre_s, tail_s=tail_s)
    eps = [x['eps_max_deg'] for x in nums if x['eps_max_deg'] is not None]
    level = dict(E_thang=E, P_max=max(x['P'] for x in nums), N_max=max(x['N'] for x in nums), T_win=T,
                 omega_max=om, alpha_max=al, eps_max_deg=max(eps) if eps else None, phan=phan,
                 U_IMU_dong_rad=u, U_IMU_dong_deg=math.degrees(u), luc_chay=now_iso(), logs=nums)
    if os.path.exists(out):
        d = _read_json(out)
        diff = [k for k in SHARED if d.get(k) != shared[k]]
        if diff:
            raise InputError(f'{out}: shared parameters differ {diff}: one file per parameter set')
        if muc in d.get('muc', {}):
            raise InputError(f'{out}: level {muc} already there (never overwritten)')
    else:
        d = dict(shared, muc={}, nguon=dict(thang=thang_d.get('nguon', []), troi=troi_d.get('nguon', []),
                                            tre=tre.get('nguon', [])), script=script_state())
    d['muc'][muc] = level
    _replace_json(out, d)
    return d


# ── camera - IMU (design 2.3, 3 step 5, 5) ─────────────────────────────────────────────────────

FRAME_NEED = ('t_s', 'psi', 'valid', 'car_seen', 'wz')    # c7_measure.py FRAME_COLS (CODE_CAM 15/11 through CODE)


def read_frames(path):
    """Valid frames with the car seen: t_s (camera clock), psi (unwrapped, rad), wz (rad/s)."""
    check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise InputError(f'{path}: no frame')
    miss = [c for c in FRAME_NEED if c not in rows[0]]
    if miss:
        raise InputError(f'{path}: no column {miss}')
    rows = [r for r in rows if r['valid'] == '1' and r['car_seen'] == '1' and r['psi'] not in ('', 'nan')]
    if len(rows) < 2:
        raise InputError(f'{path}: fewer than two valid frames with the car seen')
    try:
        t = np.array([float(r['t_s']) for r in rows])
        psi = np.array([float(r['psi']) for r in rows])
        wz = np.array([float(r['wz']) if r['wz'] not in ('', 'nan') else np.nan for r in rows])
    except ValueError as exc:
        raise InputError(f'{path}: {exc}')
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(psi))) or np.any(np.diff(t) <= 0):
        raise InputError(f'{path}: t_s or psi not finite, or t_s not increasing')
    return dict(t=t, psi=np.unwrap(psi), wz=wz, path=path)


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def _still(F, tl, a, b, wz_still):
    return (tl >= a) & (tl < b) & (np.abs(np.nan_to_num(F['wz'], nan=np.inf)) < wz_still)


def _align(F, tl, psi_i, mask):
    """Alignment constant c = median of psi_cam - psi_IMU (both unwrapped) on the still frames of mask; spread."""
    e = F['psi'][mask] - psi_i[mask]
    return float(np.median(e)), float(e.max() - e.min())


def tre_one(frames, log, offset_s, s_plus=0.0, s_minus=0.0, period_s=PERIOD_S, j_lim=J_LIM_S, h_lim=H_LIM_S,
            pre_s=3.0, tail_s=1.0, n_min=30, wz_still=0.05, half_s=0.06, step_s=0.001):
    """Delay tau of the IMU yaw against the camera yaw on one run: RMS of psi_cam - psi_IMU - c smallest, on ONE
    set of frames compared at every tau of the grid (CU-12), then a parabola through the three points."""
    _finite(offset_s, 'offset_s')
    F = read_frames(frames)
    L = load_log(log)
    S0 = rv_samples(L, period_s=period_s, s_plus=s_plus, s_minus=s_minus, j_lim=j_lim, h_lim=h_lim)
    t0, t1 = run_window(L)
    _check_no_reset(S0, t0 - pre_s, t1 + tail_s, log)
    tl = F['t'] + offset_s
    win = (tl >= t0) & (tl <= t1 + tail_s)
    still = _still(F, tl, t0 - pre_s, t0, wz_still)
    taus = np.arange(-half_s, half_s + step_s / 2, step_s)
    P = np.array([interp_yaw(dict(S0, t=S0['t'] - tau, t_break=S0['t_break'] - tau), tl, h_lim) for tau in taus])
    common = np.all(np.isfinite(P), axis=0)
    if (still & common).sum() < n_min or (win & common).sum() < n_min:
        raise InputError(f'{frames}: fewer than {n_min} still frames before t = 0 or frames in the run compared at '
                         f'every delay (CU-12)')
    rms, n_used = [], set()
    for k in range(len(taus)):
        c, _ = _align(F, tl, P[k], still & common)
        r = F['psi'][win & common] - P[k][win & common] - c
        n_used.add(int(r.size))
        rms.append(float(np.sqrt(np.mean(r ** 2))))
    rms = np.array(rms)
    k = int(np.argmin(rms))
    if k == 0 or k == len(taus) - 1:
        raise InputError(f'{frames}: delay at the edge of +-{half_s * 1e3:.0f} ms (clock offset wrong?)')
    y0, y1, y2 = rms[k - 1], rms[k], rms[k + 1]
    den = y0 - 2 * y1 + y2
    tau = taus[k] + (0.5 * step_s * (y0 - y2) / den if den > 0 else 0.0)
    if not (math.isfinite(tau) and math.isfinite(y1)):
        raise InputError(f'{frames}: delay not determined (CU-12)')
    return dict(tau_s=float(tau), rms_rad=float(y1), n_frames=int((win & common).sum()),
                n_frames_some=int((win & np.any(np.isfinite(P), axis=0)).sum()), n_used=sorted(n_used),
                offset_s=offset_s,
                n_late=S0['n_late'], frames=os.path.basename(frames), frames_sha256=sha256(frames),
                log=os.path.basename(log), log_sha256=sha256(log))


def tre(caps, thang_d, step_s=0.001, **kw):
    """tau_hat = median over the C8 runs; dtau_fit = largest |tau_i - tau_hat| + one grid step."""
    if not caps:
        raise InputError('no C8 run')
    for k in ('s_plus', 's_minus'):
        _finite(thang_d.get(k), f'thang {k}', -0.1, 0.1)
    runs = [tre_one(fr, lg, off, thang_d['s_plus'], thang_d['s_minus'], step_s=step_s, **kw) for fr, lg, off in caps]
    t = np.array([r['tau_s'] for r in runs])
    hat = float(np.median(t))
    return dict(tau_hat_s=hat, dtau_fit_s=float(np.max(np.abs(t - hat)) + step_s), thu=False, runs=runs,
                luc_chay=now_iso())


def check_u_file(u):
    """CU-17 (round 2): version and every number kiem uses, finite and in range."""
    if u.get('version') != VERSION:
        raise InputError(f'U_IMU file version {u.get("version")!r}, expected {VERSION}')
    if not isinstance(u.get('muc'), dict):
        raise InputError('U_IMU file has no muc table')
    _finite(u.get('period_s'), 'period_s', 0.01, 0.2)
    _finite(u.get('tau_hat_s'), 'tau_hat_s', -0.2, 0.2)
    _finite(u.get('j_lim_s'), 'j_lim_s', 0.0, 0.05)
    _finite(u.get('h_lim_s'), 'h_lim_s', 0.0, 1.0)
    _finite(u.get('pre_s'), 'pre_s', 0.0)
    _finite(u.get('tail_s'), 'tail_s', 0.5)
    _finite(u.get('le'), 'le', 0.0)
    _check_thang(u)
    for k in ('beta_max', 'q'):
        _finite(u.get(k), k, 0.0, 0.1)
    for m, lev in u['muc'].items():
        if not isinstance(lev, dict):
            raise InputError(f'level {m}: not an object')
        _finite(lev.get('U_IMU_dong_rad'), f'{m} U_IMU_dong_rad', 0.0, math.pi)


def kiem(frames, log, offset_s, u, muc, u_psi, u_ghep=0.0, n_min=30, wz_still=0.05):
    """Design 5: back check |psi_cam - psi_IMU| <= U_psi + U_IMU(m) on every compared frame, and the closure of
    the IMU yaw against the camera between the still parts before and after the run (unwrapped, CU-13; end part
    [t_off + tail, + END_STILL_S], CU-14)."""
    check_u_file(u)
    if muc not in u['muc']:
        raise InputError(f'level {muc} not in the U_IMU file')
    for v, w in ((offset_s, 'offset_s'), (u_psi, 'U_psi'), (u_ghep, 'u_ghep')):
        _finite(v, w, None if w == 'offset_s' else 0.0)
    lev = u['muc'][muc]
    F = read_frames(frames)
    L = load_log(log)
    S = rv_samples(L, period_s=u['period_s'], tau_hat=u['tau_hat_s'], s_plus=u['s_plus'], s_minus=u['s_minus'],
                   j_lim=u['j_lim_s'], h_lim=u['h_lim_s'])
    t0, t1 = run_window(L)
    te = t1 + u['tail_s']
    _check_no_reset(S, t0 - u['pre_s'], te + END_STILL_S, log)
    tl = F['t'] + offset_s
    psi_i = interp_yaw(S, tl)
    start = _still(F, tl, t0 - u['pre_s'], t0, wz_still) & np.isfinite(psi_i)
    if start.sum() < n_min:
        raise InputError(f'{start.sum()} still frames with an IMU yaw before t = 0, need {n_min} (design 3 step 5)')
    c, spread = _align(F, tl, psi_i, start)
    res = _wrap(F['psi'] - psi_i - c)
    win = (tl >= t0) & (tl <= te)
    comp = win & np.isfinite(res)
    lim = u_psi + lev['U_IMU_dong_rad'] + u_ghep
    over = comp & (np.abs(res) > lim)
    end = _still(F, tl, te, te + END_STILL_S, wz_still) & np.isfinite(psi_i)
    out = dict(muc=muc, align_c_rad=c, align_spread_rad=spread, align_ok=bool(spread <= u_psi),
               n_win=int(win.sum()), n_compared=int(comp.sum()), n_not_compared=int((win & ~comp).sum()),
               limit_rad=lim, n_over=int(over.sum()), t_over_s=[float(x) for x in (tl[over] - t0)[:20]],
               max_abs_rad=float(np.max(np.abs(res[comp]))) if comp.any() else None, n_late=S['n_late'],
               luc_chay=now_iso())
    if end.sum() >= n_min:
        r = float(np.median(F['psi'][end] - psi_i[end]) - c)
        t_end = float(tl[end].max())
        P, N = turn_totals(S, t0 - u['pre_s'], t_end)
        T = t_end - (t0 - u['pre_s'])
        b = 2 * u_psi + (1 + u['le']) * (u['delta_plus'] * P + u['delta_minus'] * N + u['beta_max'] * T + u['q'])
        out.update(closure_rad=r, closure_bound_rad=float(b), closure_ok=bool(abs(r) <= b), closure_P=P, closure_N=N,
                   closure_T_s=float(T))
    else:
        out.update(closure_rad=None, closure_ok=None,
                   closure_note=f'fewer than {n_min} still frames in the {END_STILL_S:.0f} s after the window')
    out['ok'] = bool(out['align_ok'] and out['n_over'] == 0 and out['closure_ok'] is True)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────────────

def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', os.path.basename(__file__)],
                           capture_output=True, text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), luc_chay=now_iso())
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, luc_chay=now_iso())


def _read_json(path):
    check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f, parse_constant=lambda x: (_ for _ in ()).throw(InputError(f'{path}: {x} in JSON')))
    if not isinstance(d, dict):
        raise InputError(f'{path}: JSON top level is not an object (CU-17)')
    d.setdefault('nguon', [{'file': os.path.basename(path), 'sha256': sha256(path)}])
    return d


def _dumps(d):
    return json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False)         # CU-12: NaN refused


def _write_new(path, d):
    check_path(path)
    text = _dumps(d)
    with open(path, 'x', encoding='utf-8') as f:
        f.write(text)


def _replace_json(path, d):
    """CU-19: a private temporary file in the same folder, then an atomic replace."""
    check_path(path)
    text = _dumps(d)
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.', suffix='.tmp', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('thang')
    a1.add_argument('--log', required=True)
    a1.add_argument('--n-vong', type=int, required=True)
    a1.add_argument('--chieu', required=True, choices=('ccw', 'cw'))
    a1.add_argument('--cach', required=True, choices=('rc', 'tay'))
    a1.add_argument('--toc-do', required=True, choices=('cham', 'nhanh'))
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('thang-gop')
    a2.add_argument('--in', dest='inp', nargs='+', required=True)
    a2.add_argument('--delta-gach-deg', type=float, default=2.0)
    a2.add_argument('--out', required=True)
    a3 = sub.add_parser('troi')
    a3.add_argument('--log', nargs='+', required=True)
    a3.add_argument('--doan', default='all', choices=('all', 'run_active'))
    a3.add_argument('--min-dt', type=float, default=2.0)
    a3.add_argument('--thang', help='scale correction of thang-gop (CU-10)')
    a3.add_argument('--out', required=True)
    a4 = sub.add_parser('tong')
    a4.add_argument('--muc', required=True)
    a4.add_argument('--c8-log', action='append', required=True)
    a4.add_argument('--thang', required=True)
    a4.add_argument('--troi', required=True)
    g = a4.add_mutually_exclusive_group(required=True)
    g.add_argument('--tre')
    g.add_argument('--tre-thu', help='TAU_S,DTAU_S trial values before the camera fit (file marked thu)')
    a4.add_argument('--u-g-deg', type=float, required=True)
    a4.add_argument('--le', type=float, default=LE)
    a4.add_argument('--pre-s', type=float, default=2.0)
    a4.add_argument('--tail-s', type=float, default=None, help='default rpi5/o1_tasks.py O1_TAIL_S')
    a4.add_argument('--out', required=True)
    a5 = sub.add_parser('tre')
    a5.add_argument('--cap', action='append', required=True, help='FRAMES.csv,LOG.csv,OFFSET_S (t_log = t_s + D)')
    a5.add_argument('--thang', required=True)
    a5.add_argument('--out', required=True)
    a6 = sub.add_parser('kiem')
    a6.add_argument('--frames', required=True)
    a6.add_argument('--log', required=True)
    a6.add_argument('--offset-s', type=float, required=True)
    a6.add_argument('--u-imu', required=True)
    a6.add_argument('--muc', required=True)
    a6.add_argument('--u-psi-deg', type=float, required=True)
    a6.add_argument('--u-ghep-deg', type=float, default=0.0, help='omega_max * dt_sync of the level (CODE_CAM)')
    a6.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        check_path(a.out)                                                  # CU-18: before any read or write
        if os.path.exists(a.out) and a.cmd != 'tong':
            raise InputError(f'{a.out} exists: never overwritten')
        if a.cmd == 'thang':
            r = thang(a.log, a.n_vong, a.chieu)
            r.update(cach=a.cach, toc_do=a.toc_do, nguon=[{'file': os.path.basename(a.log), 'sha256': sha256(a.log)}],
                     script=script_state())
            _write_new(a.out, r)
            print(f'{a.log}: s = {r["s"]:+.5f} ({r["dpsi_deg"]:.2f} deg for {a.n_vong} turns {a.chieu}), '
                  f'{r["n_late"]} late of {r["n_reports"]} reports')
        elif a.cmd == 'thang-gop':
            runs = [_read_json(p) for p in a.inp]
            r = thang_gop(runs, a.delta_gach_deg)
            r.update(nguon=[{'file': os.path.basename(p), 'sha256': sha256(p)} for p in a.inp], script=script_state())
            _write_new(a.out, r)
            print(f's_plus {r["s_plus"]:+.5f} +- {r["delta_plus"]:.5f}, s_minus {r["s_minus"]:+.5f} +- '
                  f'{r["delta_minus"]:.5f}' + (' (no RC run: F4 open)' if r['f4_mo'] else ''))
        elif a.cmd == 'troi':
            sp = sm = 0.0
            if a.thang:
                th = _read_json(a.thang)
                _check_thang(th)
                sp, sm = th['s_plus'], th['s_minus']
            r = troi(a.log, min_dt=a.min_dt, doan=a.doan, s_plus=sp, s_minus=sm)
            r.update(nguon=[{'file': os.path.basename(p), 'sha256': sha256(p)} for p in a.log], script=script_state())
            _write_new(a.out, r)
            print(f'beta_max {math.degrees(r["beta_max"]):.4f} deg/s, q {math.degrees(r["q"]):.4f} deg')
        elif a.cmd == 'tre':
            caps = []
            for c in a.cap:
                x = c.split(',')
                if len(x) != 3:
                    raise InputError(f'--cap {c!r}: FRAMES,LOG,OFFSET_S')
                caps.append((x[0], x[1], float(x[2])))
            r = tre(caps, _read_json(a.thang))
            r.update(nguon=[{'file': x[k], 'sha256': x[k + '_sha256']} for x in r['runs'] for k in ('frames', 'log')],
                     script=script_state())
            _write_new(a.out, r)
            print(f'tau_hat {r["tau_hat_s"] * 1e3:.1f} ms, dtau_fit {r["dtau_fit_s"] * 1e3:.1f} ms over {len(caps)} runs')
        elif a.cmd == 'kiem':
            r = kiem(a.frames, a.log, a.offset_s, _read_json(a.u_imu), a.muc, math.radians(a.u_psi_deg),
                     math.radians(a.u_ghep_deg))
            r.update(nguon=[{'file': os.path.basename(p), 'sha256': sha256(p)} for p in (a.frames, a.log, a.u_imu)],
                     script=script_state())
            _write_new(a.out, r)
            print(f'{a.muc}: {r["n_over"]} of {r["n_compared"]} frames over {math.degrees(r["limit_rad"]):.2f} deg; '
                  f'closure {r.get("closure_ok")}; alignment spread {math.degrees(r["align_spread_rad"]):.2f} deg; '
                  + ('OK' if r['ok'] else 'NOT OK: do not lock U_IMU (design 5)'))
        else:
            if a.tre:
                tre_d = _read_json(a.tre)
            else:
                x = [float(v) for v in a.tre_thu.split(',')]
                if len(x) != 2:
                    raise InputError('--tre-thu TAU_S,DTAU_S')
                tre_d = dict(tau_hat_s=x[0], dtau_fit_s=x[1], thu=True, nguon=[])
            d = tong(a.muc, a.c8_log, _read_json(a.thang), _read_json(a.troi), tre_d, math.radians(a.u_g_deg),
                     a.le, a.out, pre_s=a.pre_s, tail_s=a.tail_s)
            m = d['muc'][a.muc]
            print(f'{a.muc}: U_IMU,dong {m["U_IMU_dong_deg"]:.2f} deg; parts (deg) '
                  + ', '.join(f'{k} {math.degrees(v):.2f}' for k, v in m['phan'].items())
                  + (' [THU: not lockable]' if d['thu'] else ''))
    except (InputError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print('c8_u_imu:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
