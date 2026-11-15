#!/usr/bin/env python3
"""
c8_u_imu.py: the IMU dynamic part of U_IMU (O1 v11.25 lines 204-206, 247; section 4 "largest observed at C8
plus a fixed margin"), one number per task level. Design CLAUDE_THESIS
03_CODE/docs/2026-11-15_Thiet_ke_C8_U_IMU_v2.md (Liam 15/11 CODE_C13: "Duyet ca 9", "Camera-IMU o C8",
"Them 4 luot RC", "Doi sang 105 ms", "+25 %", "Chi log C8").

  U_IMU,dong(m) = U_g + (1 + le) * [ E_thang + beta_max * T_win + (dtau_fit + j_lim) * omega_max
                                     + alpha_max * h_lim^2 / 8 + q ]
  E_thang       = delta_plus * P + delta_minus * N   (P, N: total CCW, CW turn since alignment; the yaw is
                                                      scale-corrected with s_plus, s_minus first)
CODE_CAM adds omega_max * dt_sync (C7b) and writes U_IMU into the C8 gate file.

IMU yaw used here and by the camera scorer (rv_samples, interp_yaw): the game rotation vector reports of the
ESP32 log (a row where rv_n changes); the firmware time of a report is the ESP32 RECEIVE time
(AGV 5bde4c5 src/imu/imu.cpp 149, 187: rv_t_us = esp_timer_get_time() in the callback;
imu_fusion.c 226-228: age_r_ms = now_us - rv_t_us; now_us = timestamp_us of the row, main.c 254, 608).
The sensor samples on an even grid, so the receive times are put on a grid (index from the last trusted
report, line fit r = a + b n on the inliers), a report more than j_lim off the grid is dropped (late), and
the sample time is a + b n - tau_hat. Unwrapped; each step divided by (1 + s_plus) when positive, by
(1 + s_minus) when negative (scale correction, F9). Linear interpolation only between two kept reports at
most h_lim apart; else NaN (frame cannot be compared).

Sign: positive = the IMU yaw increases = 'ccw' (the scale run checks the sign the operator gives).

  python tools/c8_u_imu.py thang --log L1.csv --n-vong 5 --chieu ccw --cach rc --toc-do cham --out L1_thang.json
  python tools/c8_u_imu.py thang-gop --in L1_thang.json ... L8_thang.json --delta-gach-deg 2 --out thang.json
  python tools/c8_u_imu.py troi --log <lifted / static / still logs> ... [--doan all|run_active] --out troi.json
  python tools/c8_u_imu.py tong --muc O1-F08 --c8-log a.csv [--c8-log b.csv] --thang thang.json --troi troi.json
                                (--tre tre.json | --tre-thu TAU_S,DTAU_S) --u-g-deg X --le 0.25 --out u_imu_dong.json
`tre` (tau_hat from camera-IMU yaw, F3) and `kiem` (closure and back check, design 5) wait for the frame format
of CODE_CAM (through CODE); until then `tong` takes --tre-thu and marks the file "thu": true (not lockable).
Paths with a part sealed, sealed_test or O1_FEAS are refused. Outputs are never overwritten (`tong` adds a
level to an existing file only when every shared parameter is equal).
Last line: #ket_qua,ma=0 done / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = 'u_imu_dong_v2'
PERIOD_S = 0.05            # report period of the campaign firmware: BNO085 20 Hz (C4, L73)
J_LIM_S = 0.005            # design 3 step 2 [DE XUAT]
H_LIM_S = 0.105            # Liam 15/11 "Doi sang 105 ms"
LE = 0.25                  # Liam 15/11 "+25 %"
FORBIDDEN = {'sealed', 'sealed_test', 'o1_feas'}
LOG_COLS = ('timestamp_us', 'yaw_rad', 'rv_n', 'age_r_ms', 'gz', 'gyro_n', 'age_g_ms')


class InputError(Exception):
    pass


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
    cols = set(need) | ({'run_active', 'roll', 'pitch'} & set(rows[0]))
    L = {c: np.array([float(r[c]) if r[c] not in ('', None) else np.nan for r in rows]) for c in cols}
    order = np.argsort(L['timestamp_us'], kind='stable')
    for c in cols:
        L[c] = L[c][order]
    if np.any(np.diff(L['timestamp_us']) <= 0):
        raise InputError(f'{path}: two data rows with the same timestamp_us')
    L['n_dup'] = n_dup or 0
    L['path'] = path
    return L


# ── IMU yaw samples (shared with the camera scorer) ───────────────────────────────────────────────

def _report_rows(n, age):
    idx = [i for i in range(len(n)) if n[i] > 0 and (i == 0 or n[i] != n[i - 1])]
    idx = [i for i in idx if np.isfinite(age[i]) and age[i] >= 0]
    return np.array(idx, dtype=int)


def _grid(r, period_s, j_lim):
    """Index n of each receive time on the report grid and the residual j to the line fit."""
    d = np.diff(r)
    good = d[(d > 0.5 * period_s) & (d < 1.5 * period_s)]
    b_hat = float(np.median(good)) if good.size else period_s
    n = np.zeros(len(r), dtype=int)
    i_lt = 0                                   # last trusted report
    for k in range(1, len(r)):
        m = max(1, int(round((r[k] - r[i_lt]) / b_hat)))
        n[k] = n[i_lt] + m
        if abs(r[k] - (r[i_lt] + m * b_hat)) <= 0.25 * b_hat:
            i_lt = k
    inl = np.ones(len(r), dtype=bool)
    for _ in range(10):
        if inl.sum() < 2:
            raise InputError('fewer than two IMU reports on the grid')
        b, a = np.polyfit(n[inl], r[inl], 1)
        j = r - (a + b * n)
        new = np.abs(j) <= j_lim
        if np.array_equal(new, inl):
            break
        inl = new
    return n, a, b, j, inl


def rv_samples(L, period_s=PERIOD_S, tau_hat=0.0, s_plus=0.0, s_minus=0.0, j_lim=J_LIM_S):
    """Game rotation vector samples of a log: t (sample time, ESP32 clock, s) and psi (unwrapped,
    scale-corrected yaw, rad) of the kept reports; j, keep over every report."""
    idx = _report_rows(L['rv_n'], L['age_r_ms'])
    if idx.size < 2:
        raise InputError(f'{L.get("path")}: fewer than two rotation vector reports')
    r = L['timestamp_us'][idx] * 1e-6 - L['age_r_ms'][idx] * 1e-3
    if np.any(np.diff(r) <= 0):
        raise InputError(f'{L.get("path")}: rotation vector receive times not increasing')
    n, a, b, j, keep = _grid(r, period_s, j_lim)
    raw = np.unwrap(L['yaw_rad'][idx])
    d = np.diff(raw)
    d = np.where(d > 0, d / (1.0 + s_plus), d / (1.0 + s_minus))
    psi = raw[0] + np.concatenate([[0.0], np.cumsum(d)])
    t = a + b * n - tau_hat
    return dict(t=t[keep], psi=psi[keep], psi_all=psi, t_recv=r, j=j, keep=keep, n_samples=int(idx.size),
                n_late=int((~keep).sum()), period_fit_s=float(b), rows=idx)


def gyro_samples(L):
    idx = _report_rows(L['gyro_n'], L['age_g_ms'])
    t = L['timestamp_us'][idx] * 1e-6 - L['age_g_ms'][idx] * 1e-3
    return t, L['gz'][idx]


def interp_yaw(S, t, h_lim=H_LIM_S):
    """IMU yaw at times t (same clock as S['t']): linear between two kept reports at most h_lim apart."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ts, ps = S['t'], S['psi']
    i = np.searchsorted(ts, t, side='right')
    out = np.full(t.shape, np.nan)
    ok = (i > 0) & (i < len(ts))
    i0, i1 = i[ok] - 1, i[ok]
    gap = ts[i1] - ts[i0]
    good = gap <= h_lim + 1e-9
    w = (t[ok] - ts[i0]) / gap
    v = ps[i0] + w * (ps[i1] - ps[i0])
    sub = np.full(ok.sum(), np.nan)
    sub[good] = v[good]
    out[ok] = sub
    exact = np.isin(t, ts)
    if exact.any():
        out[exact] = np.interp(t[exact], ts, ps)
    return out


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


def thang(path, n_vong, chieu, period_s=PERIOD_S, g_still=0.02, min_still_s=2.0, edge_s=0.2):
    """Scale error of one run of n_vong turns between two still parts: s = dpsi / (+-n 2 pi) - 1."""
    if chieu not in ('ccw', 'cw'):
        raise InputError(f'--chieu {chieu!r}: ccw or cw')
    if n_vong < 1:
        raise InputError('--n-vong >= 1')
    L = load_log(path)
    S = rv_samples(L, period_s=period_s)
    tg, gz = gyro_samples(L)
    lead, trail, rot = _still_ends(tg, gz, g_still, min_still_s)
    t_all = S['t_recv']
    a = (t_all >= lead[0]) & (t_all <= lead[1] - edge_s)
    b = (t_all >= trail[0] + edge_s) & (t_all <= trail[1])
    if a.sum() < 5 or b.sum() < 5:
        raise InputError('too few IMU reports in a still part')
    dpsi = float(np.median(S['psi_all'][b]) - np.median(S['psi_all'][a]))
    exp = (1 if chieu == 'ccw' else -1) * n_vong * 2 * math.pi
    if dpsi * exp <= 0:
        raise InputError(f'IMU yaw changed by {math.degrees(dpsi):+.1f} deg: not the direction {chieu}')
    s = dpsi / exp - 1.0
    if abs(s) > 0.1:
        raise InputError(f'scale {s:+.3f}: more than 10 %, turn count wrong? (dpsi {math.degrees(dpsi):.1f} deg)')
    rot_mask = (tg >= rot[0]) & (tg <= rot[1])
    return dict(s=s, chieu=chieu, n_vong=n_vong, dpsi_rad=dpsi, dpsi_deg=math.degrees(dpsi),
                still_start_s=list(lead), still_end_s=list(trail),
                omega_median=float(np.median(np.abs(gz[rot_mask]))) if rot_mask.any() else None,
                n_reports=S['n_samples'], n_late=S['n_late'], n_dup=L['n_dup'])


def thang_gop(runs, delta_gach_deg=2.0):
    """s_hat per direction = mean of the RC runs (of the hand runs when there is no RC run: F4 left open);
    delta = alignment error / (n 2 pi) + largest |s_i - s_hat| over every run of that direction."""
    out = dict(delta_gach_deg=delta_gach_deg, f4_mo=False)
    for chieu, key in (('ccw', 'plus'), ('cw', 'minus')):
        rr = [r for r in runs if r['chieu'] == chieu]
        if not rr:
            raise InputError(f'no scale run in direction {chieu}')
        rc = [r for r in rr if r.get('cach') == 'rc']
        base = rc or rr
        if not rc:
            out['f4_mo'] = True
        s_hat = float(np.mean([r['s'] for r in base]))
        g = max(math.radians(delta_gach_deg) / (r['n_vong'] * 2 * math.pi) for r in rr)
        out['s_' + key] = s_hat
        out['delta_' + key] = g + max(abs(r['s'] - s_hat) for r in rr)
        out['n_' + key] = dict(rc=len(rc), tay=len(rr) - len(rc))
    return out


# ── drift and resolution (design 2.2) ──────────────────────────────────────────────────────

def _yaw_series(path, doan, period_s):
    check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        head = f.readline()
    if head.startswith('kind,'):                       # tools/imu_capture.py raw_<phase>.csv
        with open(path, newline='', encoding='utf-8') as f:
            R = [r for r in csv.DictReader(f) if r['kind'] == 'R']
        if len(R) < 2:
            raise InputError(f'{path}: no R (game RV) rows')
        t = np.array([float(r['t_us']) for r in R]) * 1e-6
        return t, np.unwrap(np.radians([float(r['v2']) for r in R]))
    L = load_log(path)
    S = rv_samples(L, period_s=period_s)
    t, psi = S['t_recv'], S['psi_all']
    if doan == 'run_active':
        if 'run_active' not in L:
            raise InputError(f'{path}: no run_active column')
        on = L['run_active'][S['rows']] == 1
        if not on.any():
            raise InputError(f'{path}: no run_active row')
        k = np.nonzero(on)[0]
        t, psi = t[k[0]:k[-1] + 1], psi[k[0]:k[-1] + 1]
    return t, psi


def troi(paths, period_s=PERIOD_S, min_dt=2.0, doan='all', block_s=1.0):
    """beta = largest |psi(t) - psi(t0)| / (t - t0) with t - t0 >= min_dt (t0 = start of the part);
    q = largest |yaw - median of its 1 s block|. Over every file."""
    per, beta, q = [], 0.0, 0.0
    for p in paths:
        t, psi = _yaw_series(p, doan, period_s)
        dt = t - t[0]
        m = dt >= min_dt
        b = float(np.max(np.abs(psi[m] - psi[0]) / dt[m])) if m.any() else float('nan')
        blk = np.floor(dt / block_s).astype(int)
        qq = 0.0
        for k in np.unique(blk):
            v = psi[blk == k]
            if v.size >= 3:
                qq = max(qq, float(np.max(np.abs(v - np.median(v)))))
        per.append(dict(file=os.path.basename(p), sha256=sha256(p), duration_s=float(dt[-1]), beta=b, q=qq))
        if np.isfinite(b):
            beta = max(beta, b)
        q = max(q, qq)
    if not any(np.isfinite(x['beta']) for x in per):
        raise InputError(f'no part longer than {min_dt} s')
    return dict(beta_max=beta, q=q, min_dt_s=min_dt, doan=doan, files=per)


# ── total for one level (design 1) ─────────────────────────────────────────────────────────

SHARED = ('version', 'le', 'U_g', 'period_s', 'j_lim_s', 'h_lim_s', 'tau_hat_s', 'dtau_fit_s', 's_plus', 's_minus',
          'delta_plus', 'delta_minus', 'beta_max', 'q', 'thu', 'f4_mo', 'pre_s', 'tail_s')


def _level_numbers(path, tre, thang_d, period_s, j_lim, pre_s, tail_s):
    L = load_log(path)
    if 'run_active' not in L or not (L['run_active'] == 1).any():
        raise InputError(f'{path}: no run_active row (t = 0 of the run)')
    on = np.nonzero(L['run_active'] == 1)[0]
    t_on, t_off = L['timestamp_us'][on[0]] * 1e-6, L['timestamp_us'][on[-1]] * 1e-6
    w0, w1 = t_on - pre_s, t_off + tail_s
    S = rv_samples(L, period_s=period_s, tau_hat=tre['tau_hat_s'], s_plus=thang_d['s_plus'],
                   s_minus=thang_d['s_minus'], j_lim=j_lim)
    m = (S['t'] >= w0) & (S['t'] <= w1)
    d = np.diff(S['psi'][m])
    tg, gz = gyro_samples(L)
    g = (tg >= w0) & (tg <= w1)
    tw, gw = tg[g], gz[g]
    dt = np.maximum(np.diff(tw), 0.5 * period_s)
    alpha = float(np.max(np.abs(np.diff(gw)) / dt)) if gw.size > 1 else 0.0
    return dict(P=float(d[d > 0].sum()), N=float(-d[d < 0].sum()), T_win=float(w1 - w0),
                omega=float(np.max(np.abs(gw))) if gw.size else 0.0, alpha=alpha,
                n_late=S['n_late'], n_reports=S['n_samples'], file=os.path.basename(path), sha256=sha256(path))


def tong(muc, logs, thang_d, troi_d, tre, u_g, le, out, period_s=PERIOD_S, j_lim=J_LIM_S, h_lim=H_LIM_S,
         pre_s=2.0, tail_s=1.0):
    if not logs:
        raise InputError('no C8 log for the level')
    nums = [_level_numbers(p, tre, thang_d, period_s, j_lim, pre_s, tail_s) for p in logs]
    P = max(x['P'] for x in nums)
    N = max(x['N'] for x in nums)
    T = max(x['T_win'] for x in nums)
    om = max(x['omega'] for x in nums)
    al = max(x['alpha'] for x in nums)
    phan = dict(thang=thang_d['delta_plus'] * P + thang_d['delta_minus'] * N,
                troi=troi_d['beta_max'] * T,
                thoi_gian=(tre['dtau_fit_s'] + j_lim) * om,
                noi_suy=al * h_lim ** 2 / 8.0,
                q=troi_d['q'])
    u = u_g + (1.0 + le) * sum(phan.values())
    shared = dict(version=VERSION, le=le, U_g=u_g, period_s=period_s, j_lim_s=j_lim, h_lim_s=h_lim,
                  tau_hat_s=tre['tau_hat_s'], dtau_fit_s=tre['dtau_fit_s'], s_plus=thang_d['s_plus'],
                  s_minus=thang_d['s_minus'], delta_plus=thang_d['delta_plus'], delta_minus=thang_d['delta_minus'],
                  beta_max=troi_d['beta_max'], q=troi_d['q'], thu=bool(tre.get('thu')),
                  f4_mo=bool(thang_d.get('f4_mo')), pre_s=pre_s, tail_s=tail_s)
    level = dict(P_max=P, N_max=N, T_win=T, omega_max=om, alpha_max=al, phan=phan, U_IMU_dong_rad=u,
                 U_IMU_dong_deg=math.degrees(u), logs=nums)
    if os.path.exists(out):
        with open(out, encoding='utf-8') as f:
            d = json.load(f)
        diff = [k for k in SHARED if d.get(k) != shared[k]]
        if diff:
            raise InputError(f'{out}: shared parameters differ {diff}: one file per parameter set')
        if muc in d['muc']:
            raise InputError(f'{out}: level {muc} already there (never overwritten)')
    else:
        d = dict(shared, muc={}, nguon=dict(thang=thang_d.get('nguon', []), troi=troi_d.get('nguon', []),
                                            tre=tre.get('nguon', [])), script=script_state())
    d['muc'][muc] = level
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(tmp, out)
    return d


# ── CLI ────────────────────────────────────────────────────────────────────────────────

def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', os.path.basename(__file__)],
                           capture_output=True, text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None)


def _read_json(path):
    check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    d.setdefault('nguon', [{'file': os.path.basename(path), 'sha256': sha256(path)}])
    return d


def _write_new(path, d):
    if os.path.exists(path):
        raise InputError(f'{path} exists: never overwritten')
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(d, f, indent=1, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('thang')
    a1.add_argument('--log', required=True)
    a1.add_argument('--n-vong', type=int, required=True)
    a1.add_argument('--chieu', required=True, choices=('ccw', 'cw'))
    a1.add_argument('--cach', required=True, choices=('rc', 'tay'))
    a1.add_argument('--toc-do', default='', choices=('', 'cham', 'nhanh'))
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('thang-gop')
    a2.add_argument('--in', dest='inp', nargs='+', required=True)
    a2.add_argument('--delta-gach-deg', type=float, default=2.0)
    a2.add_argument('--out', required=True)
    a3 = sub.add_parser('troi')
    a3.add_argument('--log', nargs='+', required=True)
    a3.add_argument('--doan', default='all', choices=('all', 'run_active'))
    a3.add_argument('--min-dt', type=float, default=2.0)
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
    a4.add_argument('--tail-s', type=float, default=1.0)
    a4.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
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
            r = troi(a.log, min_dt=a.min_dt, doan=a.doan)
            r.update(nguon=[{'file': x['file'], 'sha256': x['sha256']} for x in r['files']], script=script_state())
            _write_new(a.out, r)
            print(f'beta_max {math.degrees(r["beta_max"]):.4f} deg/s, q {math.degrees(r["q"]):.4f} deg')
        else:
            if a.tre:
                tre = _read_json(a.tre)
            else:
                x = [float(v) for v in a.tre_thu.split(',')]
                if len(x) != 2:
                    raise InputError('--tre-thu TAU_S,DTAU_S')
                tre = dict(tau_hat_s=x[0], dtau_fit_s=x[1], thu=True, nguon=[])
            d = tong(a.muc, a.c8_log, _read_json(a.thang), _read_json(a.troi), tre, math.radians(a.u_g_deg),
                     a.le, a.out, pre_s=a.pre_s, tail_s=a.tail_s)
            m = d['muc'][a.muc]
            print(f'{a.muc}: U_IMU,dong {m["U_IMU_dong_deg"]:.2f} deg; parts (deg) '
                  + ', '.join(f'{k} {math.degrees(v):.2f}' for k, v in m['phan'].items())
                  + (' [THU: not lockable]' if d['thu'] else ''))
    except (InputError, OSError, ValueError, KeyError) as exc:
        print('c8_u_imu:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
