#!/usr/bin/env python3
"""
c8b_gio.py: clock interval of one O1 floor run between the camera (frames.csv of c7_measure.py) and the ESP32 log,
by two estimates on sample marks (C8b design v4 section 3, approved: Liam 17/11, CODE_C8B local_cccce6cb,
"Duyet (Recommended)"; CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_C8b_v4.md sections 2, 3, 7). Written by
CODE_C8B local_21b7debc on feat/c8b-u. Writes the clock file c8b_gio_v1 that c8b_luoi.py (and the camera scorer,
per node) read.

Convention t_log = t_cam + D (o1_time_sync). Four channels, each matched to the camera by the Pearson correlation on
a 2 ms grid (the o1_time_sync method, re-implemented here so that NOTHING is bridged: no interpolation across a
lost or rejected camera frame, a log hole, a non-consecutive or invalid count pair, or an IMU report gap above
2 periods + 10 ms; section 7):
  speed_cnt  |v| from the wheel counts of two log lines consecutive by log_seq, both valid (not INT32_MIN), at the
             middle of the two timestamp_us (O1 v11.33 SC-12 OG3-1, OG3-2), fixed firmware kinematics
             (C8B4-2: AGV 5bde4c5 encoder.h PPR 1320, kinematics.h r 0.0485 m, lx + ly 0.24 m)
  wz_cnt     wz from the same counts
  gyro_rx    gz_fused of each gyro report at its ESP32 RECEIVE time timestamp_us - age_g_ms (design 2: OG3-3 of O1)
  rv_rx      yaw rate of consecutive game-rotation-vector reports (yaw_rad) at the middle of their receive times
The camera side is the frame speed / wz of frames.csv on the frame PTS.
Interval (section 3): L = larger D of the two encoder channels (assumption: the body does not lead the wheels);
U~ = smaller D of the two IMU channels (assumption: the IMU gives no apparent lead, F2). Centre (L + U~)/2,
half width LE x (U~ - L)/2 (LE = 1.25, author's convention). An estimate with assumptions, not a proven bound.
Grid nodes (F3; C8B4R2-1, -2): spacing <= 1 ms, both ends included; zero width (L = U~): one node (C8B4R5-2).

Checks that BLOCK (section 7, C8B4-13; O1 v11.34 3.5, Liam 17/11 at THESIS Q2 "Hong he do, chay lai <=3
(Recommended)"). Lost to the measurement system (ket_luan hong_he_do: run again with the same gains; the count of
reruns, at most 3 per gain set and level, is kept outside this script): fewer than PAIR_MIN (95 % [DE XUAT]) of the
consecutive line pairs of the log window valid, a log hole above LOG_HOLE_S (100 ms [DE XUAT]), and the per-run
checks of section 3: a channel fit not ok (correlation, start / end thirds, o1_time_sync limits), L > U~, the
start-up estimate not meeting the scoring interval, and, with a C8 lock file, the width U~ - L or the gyro - RV
difference outside the range of the C8 runs of the same level (at C8 itself only the first two). No finite clock
bound (ket_luan chua_ket_luan: G1-G5 not concluded; design 7): reset_n not constant, g_valid or r_valid 0 in the log
window, an IMU age below AGE_NEG_LIM_MS (-2 ms [DE XUAT]), a lock file without the range of the level. Refused
(exit 2): duplicate log lines not marked by the logger segments file (a repeated log_seq, L92), a log_seq jump in the
run window (o1_time_sync.log_window), times not increasing.
Start-up check (F5; C8B4R2-3; a consistency check only, never narrows the interval): from standstill, over the
first STARTUP_S after the car leaves its place, a lower estimate from the wheel path (D >= t_enc(s_k) - t_cam,k:
the tag does not go farther than the wheels rolled) and an upper estimate from the RV yaw (D <= t_rv - t_cam(dpsi):
the IMU does not overshoot the true angle turned one way); each one only when its condition holds (path at least
STARTUP_MIN_S_M, yaw turned one way at least STARTUP_MIN_PSI_DEG), else 'khong_xac_dinh'. Brought to the matching
window by the drift rate x distance, then checked against the scoring interval.
Reported for the camera scorer: the matching window on the log clock (union of the channel windows), the largest
distance from it to the scoring domain [t0, t0 + t_deadline] (drift part of U_t: rate bound x this distance,
C8B4R2-4), the row / column span of the car tag in the matching window and the scoring domain (rolling shutter,
C8B4R2-6, -9), the frame period, the PTS interval spread (design 3: PTS assumed constant offset; uneven frame
intervals reported), and the drift rate of this run on the two bound channels (|D_end - D_start| / distance of the
third centres; the bound used is the largest over the C8 runs x 1.25, c8b_u.py).

  python tools/c8b_gio.py --frames <run>/frames.csv --esp32 <run>.csv --task O1-F08 --out gio.json \
      [--khoa c8b_khoa.json (troi_ppm, U_p_m, U_psi_deg, dai)] [--startup-s 0.5]
Last line: #ket_qua,ma=0,ok=1|0 written (ok = 0: written, the run has no finite clock bound) / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402
import o1_time_sync as ts                                                # noqa: E402

FORMAT = 'c8b_gio_v1'
INT32_MIN = -2147483648
PPR = 1320.0               # counts per wheel turn, AGV 5bde4c5 src/encoder/encoder.h
R_W = 0.0485               # m, wheel radius, AGV 5bde4c5 kinematics.h
LXY = 0.24                 # m, lx + ly, AGV 5bde4c5 kinematics.h
IMU_P = 0.050              # s, IMU_REPORT_PERIOD_US 50000 (imu.h), campaign 20 Hz
IMU_GAP_S = 2 * IMU_P + 0.010   # section 7: no bridging across a larger IMU report gap
LOG_P = 0.010              # s, control / log tick of the campaign firmware (10 ms)
LOG_GAP_S = 2 * LOG_P + 0.010   # section 7: a count pair over a larger time gap is not used (C8BS-2), even with
                                # consecutive log_seq; the whole run is lost only above LOG_HOLE_S
AGE_NEG_LIM_MS = -2.0      # [DE XUAT] section 7: a negative age down to -2 ms is valid (design 2), below is an error
PAIR_MIN = 0.95            # [DE XUAT] section 7
LOG_HOLE_S = 0.100         # [DE XUAT] section 7
LE = 1.25                  # half-width margin, author's convention (design 3, as U_IMU v11.26)
STEP_S = 0.001             # grid spacing (F3)
CAM_GAP_FACTOR = 1.5       # two valid frames further apart than 1.5 median frame periods: not bridged
STARTUP_S = 0.5            # F5: first 0.3-0.5 s after the car leaves its place (Fable)
STARTUP_LEAVE_M = 0.005    # [DE XUAT] the car has left when the tag is 5 mm from its standing place
STARTUP_MIN_S_M = 0.030    # [DE XUAT] lower estimate only with at least 30 mm of tag path in the window
STARTUP_USE_S_M = 0.010    # [DE XUAT] frames with less tag displacement are below the camera noise: not used
STARTUP_MIN_PSI_DEG = 1.0  # [DE XUAT] upper estimate only with at least 1 deg turned one way
STARTUP_E_P_M = 0.005      # [DE XUAT] O1 v11.36: allowed error of a camera displacement before C8 (2 U_p after C8)
STARTUP_E_PSI_DEG = 0.5    # [DE XUAT] O1 v11.36: allowed error of a camera angle change before C8 (2 U_psi after C8)
PRE_S = (1.0, 0.1)         # s before t0: standing place from the frames in [t0 - 1.0, t0 - 0.1] (camera clock)
CHANNELS = ('speed_cnt', 'wz_cnt', 'gyro_rx', 'rv_rx')
LOWER, UPPER = ('speed_cnt', 'wz_cnt'), ('gyro_rx', 'rv_rx')


class GioError(Exception):
    pass


C8B_FILES = ('tools/c8b_gio.py', 'tools/c8b_luoi.py', 'tools/c8b_u.py', 'tools/c8b_3b.py', 'tools/c7_score.py',
             'tools/o1_time_sync.py', 'tools/c8_u_imu.py', 'tools/gate_c8.py', 'fwsim/o1_gate.py')


def repo_state(path_in_repo=None):
    """C8BS-16: commit of the repo of path_in_repo (default this file) and whether ANY file of the C8b chain
    (C8B_FILES: the C8b scripts, the scorer, the clock match, gate files) has a change not committed there."""
    here = os.path.dirname(os.path.abspath(path_in_repo or __file__))
    try:
        top = subprocess.run(['git', '-C', here, 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        c = subprocess.run(['git', '-C', top, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', top, 'status', '--porcelain', '--', *C8B_FILES], capture_output=True,
                           text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()) or c.returncode != 0,
                    luc_chay=cu.now_iso())
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, luc_chay=cu.now_iso())


# ── log ──────────────────────────────────────────────────────────────────────────────────────

def read_log(path):
    """Data lines of an agv_logger CSV (string dicts), duplicates the logger marked in <log>.segments.csv dropped
    (c8_u_imu._drop_duplicates, the c5_lifted_check rule); a repeated log_seq left after that is refused (L92)."""
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith('#'))]
    rows = [r for r in rows if r.get('timestamp_us') not in (None, '')]
    if not rows:
        raise GioError(f'{path}: no data line')
    need = ('timestamp_us', 'log_seq', 'fl_count', 'fr_count', 'rl_count', 'rr_count', 'gz_fused', 'age_g_ms',
            'gyro_n', 'yaw_rad', 'age_r_ms', 'rv_n', 'reset_n', 'g_valid', 'r_valid', 'run_active',
            'sp_vx', 'sp_vy', 'sp_wz')
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise GioError(f'{path}: no column {miss}')
    try:
        rows, n_dup = cu._drop_duplicates(path, rows)
    except cu.InputError as exc:
        raise GioError(str(exc))
    seq = [int(float(r['log_seq'])) for r in rows]
    if len(set(seq)) != len(seq):
        raise GioError(f'{path}: repeated log_seq without the logger segments file marking it (L92): run lost')
    t = np.array([float(r['timestamp_us']) for r in rows])
    if np.any(np.diff(t) <= 0):
        raise GioError(f'{path}: timestamp_us does not increase')
    return rows, int(n_dup or 0)


def controller_of(log_path):
    """C8BS-14: the controller of the run from the '#gain ... ctrl=X' lines of the logger events file
    (<log>.events.txt); None when the file or the lines are missing or name more than one controller."""
    ev = log_path + '.events.txt'
    if not os.path.isfile(ev):
        return None, None
    found = set()
    with open(ev, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '#gain,' in line:
                for part in line.strip().split(','):
                    if part.startswith('ctrl='):
                        found.add(part[5:])
    return (found.pop() if len(found) == 1 else None), ev


def _col(rows, c):
    return np.array([float(r[c]) if r[c] not in ('', None) else np.nan for r in rows])


def fk(rate_cps):
    """Counts/s (fl, fr, rl, rr) -> (vx, vy, wz): kinematics_forward of the firmware (fixed, C8B4-2)."""
    w = np.asarray(rate_cps, float) * 2.0 * math.pi / PPR
    fl, fr, rl, rr = w.T
    r4 = R_W / 4.0
    return np.column_stack([r4 * (fl + fr + rl + rr), r4 * (-fl + fr + rl - rr), r4 / LXY * (-fl + fr - rl + rr)])


def _gapped(t, v, breaks):
    """Series (t, v) with a NaN sample at the middle of every break (index i: between samples i and i+1), so that
    linear interpolation gives NaN across it (nothing bridged)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    b = np.flatnonzero(breaks)
    if not len(b):
        return t, v
    tn = 0.5 * (t[b] + t[b + 1])
    tt = np.insert(t, b + 1, tn)
    vv = np.insert(v, b + 1, np.nan)
    return tt, vv


def count_pairs(rows, tl):
    """Encoder pairs: lines consecutive by log_seq, both with four valid counts, at most LOG_GAP_S apart. Returns
    (mid times, rate (n, 4), breaks (bool, between pair k and k+1: not adjacent pairs), valid flags of every
    consecutive line pair, line index of each pair)."""
    seq = np.array([int(float(r['log_seq'])) for r in rows])
    C = np.array([[int(float(r[c])) for c in ('fl_count', 'fr_count', 'rl_count', 'rr_count')] for r in rows], float)
    good = np.all(C != INT32_MIN, axis=1)
    adj = np.diff(seq) == 1
    pair_ok = adj & good[1:] & good[:-1] & (np.diff(tl) <= LOG_GAP_S)
    i = np.flatnonzero(pair_ok)
    rate = (C[i + 1] - C[i]) / (tl[i + 1] - tl[i])[:, None]
    tm = 0.5 * (tl[i + 1] + tl[i])
    breaks = np.diff(i) != 1                        # pairs k, k+1 must share a line, else a break
    return tm, rate, breaks, pair_ok, i


def imu_reports(rows, tl, ncol, agecol, valcol):
    """One sample per report: the first line of each new count value with a finite age; receive time
    timestamp_us - age (negative ages are valid, design 2). Returns (t, value, ages of those lines [ms])."""
    n = _col(rows, ncol)
    age = _col(rows, agecol)
    val = _col(rows, valcol)
    idx, cur = [], None
    for k in range(len(n)):
        if not np.isfinite(n[k]) or n[k] <= 0:
            continue
        if n[k] != cur and np.isfinite(age[k]):
            idx.append(k)
            cur = n[k]
    idx = np.array(idx, int)
    tr = tl[idx] - age[idx] * 1e-3
    o = np.argsort(tr, kind='stable')
    tr, vr, ar = tr[o], val[idx][o], age[idx][o]
    keep = np.r_[True, np.diff(tr) > 0]
    return tr[keep], vr[keep], ar[keep]


# ── matching (o1_time_sync method, nothing bridged) ─────────────────────────────────────────────────

def _onset(t, v, thr=ts.MOVE_V, hold=0.2):
    fin = np.isfinite(v)
    t, v = t[fin], v[fin]
    above = v > thr
    for i in np.flatnonzero(above):
        j = np.searchsorted(t, t[i] + hold)
        if j >= len(t):
            break
        if above[i:j + 1].all():
            return float(t[i])
    raise GioError('the profile never stays above the moving threshold')


def cam_grid(t_cam, v_cam, valid, t0, t1, step=ts.GRID_DT):
    """Camera values on a grid over [t0, t1): linear between two valid frames that are next to each other in the
    file AND at most CAM_GAP_FACTOR frame periods apart; NaN elsewhere (no bridging over a rejected or lost frame)."""
    tg = np.arange(t0, t1, step)
    t_cam, v_cam = np.asarray(t_cam, float), np.asarray(v_cam, float)
    ok = np.asarray(valid, bool) & np.isfinite(v_cam)
    per = float(np.median(np.diff(t_cam)))
    seg_ok = ok[:-1] & ok[1:] & (np.diff(t_cam) <= CAM_GAP_FACTOR * per)
    j = np.searchsorted(t_cam, tg, side='right') - 1           # frame at or before
    inside = (j >= 0) & (j < len(t_cam) - 1)
    jj = np.clip(j, 0, len(t_cam) - 2)
    w = (tg - t_cam[jj]) / (t_cam[jj + 1] - t_cam[jj])
    vg = v_cam[jj] * (1 - w) + v_cam[jj + 1] * w
    exact = inside & (w == 0.0) & ok[jj]
    good = inside & (seg_ok[jj] | exact)
    vg = np.where(good, vg, np.nan)
    return tg, vg


def _corr_at(D, tg, vg, t_log, v_log):
    x = np.interp(tg + D, t_log, v_log, left=np.nan, right=np.nan)
    ok = np.isfinite(x) & np.isfinite(vg)
    if ok.sum() < ts.MIN_OVERLAP_S / ts.GRID_DT:
        return -np.inf
    a, b = vg[ok] - vg[ok].mean(), x[ok] - x[ok].mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else -np.inf


def _fit(t_cam, v_cam, valid, t_log, v_log, t0, t1, guess, half=ts.SEARCH_S, widen=ts.WIDEN):
    tg, vg = cam_grid(t_cam, v_cam, valid, t0, t1)
    for n in range(widen + 1):
        Ds = guess + np.arange(-half, half + ts.GRID_DT / 2, ts.GRID_DT)
        cs = np.array([_corr_at(D, tg, vg, t_log, v_log) for D in Ds])
        k = int(np.argmax(cs))
        if not np.isfinite(cs[k]):
            raise GioError('no overlap between the camera window and the log')
        if 0 < k < len(cs) - 1:
            break
        guess, half = float(Ds[k]), 2.0 * half
    else:
        raise GioError(f'correlation peak on the edge of the search range after widening {widen} times')
    D = float(Ds[k])
    if np.all(np.isfinite(cs[k - 1:k + 2])):
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:
            D += 0.5 * (y0 - y2) / den * ts.GRID_DT
    return D, float(cs[k])


def match(t_cam, v_cam, valid, t_log, v_log, window_log, coarse):
    """D (t_log = t_cam + D) over the window, the start and end thirds, ok by the o1_time_sync limits."""
    w0 = max(window_log[0] - coarse, float(t_cam[0]))
    w1 = min(window_log[1] - coarse, float(t_cam[-1]))
    if w1 - w0 < ts.MIN_OVERLAP_S:
        raise GioError(f'window {w1 - w0:.1f} s too short')
    D, c = _fit(t_cam, v_cam, valid, t_log, v_log, w0, w1, coarse)
    out = dict(D_s=D, corr=c, window_cam_s=[w0, w1], window_log_s=[w0 + D, w1 + D], short=False,
               D_start_s=None, D_end_s=None, corr_start=None, corr_end=None, rate=None, ok=True, why='')
    why = []
    if w1 - w0 < 3 * ts.MIN_OVERLAP_S:
        out['short'] = True                           # design 7: short runs handled apart (no thirds, no rate)
    else:
        third = (w1 - w0) / 3.0
        try:
            Ds_, cs_ = _fit(t_cam, v_cam, valid, t_log, v_log, w0, w0 + third, D, min(ts.SEARCH_S, 0.5), widen=0)
            De_, ce_ = _fit(t_cam, v_cam, valid, t_log, v_log, w1 - third, w1, D, min(ts.SEARCH_S, 0.5), widen=0)
        except GioError as exc:
            out.update(ok=False, why=f'start/end third: {exc}')
            return out
        out.update(D_start_s=Ds_, D_end_s=De_, corr_start=cs_, corr_end=ce_,
                   rate=abs(De_ - Ds_) / (w1 - w0 - third))            # distance of the third centres
        if abs(D - 0.5 * (Ds_ + De_)) > ts.SPLIT_TOL_S:
            why.append(f'|D - mean(start, end)| > {ts.SPLIT_TOL_S * 1e3:.0f} ms')
        for nm, cc in (('start', cs_), ('end', ce_)):         # design 7: each third well correlated (C8BS-1)
            if cc < ts.MIN_CORR:
                why.append(f'corr of the {nm} third {cc:.3f} < {ts.MIN_CORR}')
    if c < ts.MIN_CORR:
        why.append(f'corr {c:.3f} < {ts.MIN_CORR}')
    out.update(ok=not why, why='; '.join(why))
    return out


# ── frames ───────────────────────────────────────────────────────────────────────────────────

def read_frames(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        fr = list(csv.DictReader(f))
    if not fr:
        raise GioError(f'{path}: no frame')
    need = ('t_s', 'valid', 'x', 'y', 'psi', 'speed', 'wz', 'row_px')
    miss = [c for c in need if c not in fr[0]]
    if miss:
        raise GioError(f'{path}: no column {miss}')

    def g(c):
        return np.array([float(r[c]) if r.get(c) not in ('', 'nan', None) else np.nan for r in fr])
    F = {c: g(c) for c in ('t_s', 'x', 'y', 'psi', 'speed', 'wz', 'row_px')}
    F['col_px'] = g('col_px') if 'col_px' in fr[0] else np.full(len(fr), np.nan)
    F['valid'] = np.array([r['valid'] == '1' for r in fr])
    if np.any(np.diff(F['t_s']) <= 0):
        raise GioError(f'{path}: t_s does not increase')
    for c in ('x', 'y', 'psi', 'speed', 'wz', 'row_px', 'col_px'):
        F[c] = np.where(F['valid'], F[c], np.nan)
    return F


def t_deadline(task):
    """t_deadline = T_cmd + T_LATE of the task (O1 3.4)."""
    from o1_feasibility import T_LATE, Schedule
    from o1_tasks import FIG8, FINAL_SET
    if task in FIG8:
        return Schedule(FIG8[task]).Tcmd + T_LATE
    if task in FINAL_SET:
        import o1_lanes
        return o1_lanes.KSchedule(task).Tcmd + T_LATE
    raise GioError(f'task {task!r}')


# ── start-up check (F5) ─────────────────────────────────────────────────────────────────────────

def startup(F, t0_log, D_guess, tm, vel, br, t_rv, yaw_rv, startup_s=STARTUP_S, e_p=STARTUP_E_P_M,
            e_psi=math.radians(STARTUP_E_PSI_DEG)):
    """Lower / upper estimate of D from the standstill start (see the module text). D_guess only places the
    standing-place window and the RV search window; the estimates do not use it otherwise.
    [CACH HIEU; O1 v11.36] the camera error is allowed for, so that the check only flags a real disagreement (trial
    16/11: camera and RV yaw differ by up to 0.2 deg at about 3 deg/s just after the start, which alone moves a yaw
    crossing by about 60 ms): lower D >= t_enc(d_k - e_p) - t_k (the true tag displacement is at least the measured
    one minus e_p); upper D <= t_rv - t_cam(a - e_psi) (the true camera angle change reaches a no earlier than the
    measured one reaches a - e_psi). e_p, e_psi bound the error of a DIFFERENCE of two camera poses (the standing place
    and the frame; C8BS-5): before C8 the [DE XUAT] 5 mm, 0.5 deg of O1 v11.36, after C8 2 U_p, 2 U_psi.
    Nothing bridged (C8BS-3): the wheel path only over count pairs without a break from t0 - 0.5 s to the end of the
    window; the camera crossing only over valid frames next to each other; else 'khong_xac_dinh' for that side."""
    t = F['t_s']
    v = F['valid'] & np.isfinite(F['x']) & np.isfinite(F['y'])
    t0c = t0_log - D_guess
    pre = v & (t >= t0c - PRE_S[0]) & (t <= t0c - PRE_S[1])
    res = dict(lo_s=None, hi_s=None, lo_why='', hi_why='', t_start_log_s=None, e_p_m=e_p,
               e_psi_deg=math.degrees(e_psi))
    if pre.sum() < 5:
        res.update(lo_why='fewer than 5 standing frames before t0', hi_why='fewer than 5 standing frames before t0')
        return res
    x0, y0 = float(np.median(F['x'][pre])), float(np.median(F['y'][pre]))
    d = np.hypot(F['x'] - x0, F['y'] - y0)
    after = v & (t > t0c - PRE_S[1])
    left = np.flatnonzero(after & (d > STARTUP_LEAVE_M))
    if not len(left):
        res.update(lo_why='the car never leaves its place', hi_why='the car never leaves its place')
        return res
    tl0 = float(t[left[0]])
    win = v & (t >= tl0) & (t <= tl0 + startup_s)
    res['t_start_log_s'] = tl0 + D_guess
    # lower: wheel path length (sum of |FK translation| over the count pairs from standstill) against the tag chord
    vxy = np.hypot(vel[:, 0], vel[:, 1])
    k0 = int(np.searchsorted(tm, t0_log - 0.5))
    k1 = int(np.searchsorted(tm, tl0 + startup_s + D_guess + 0.2))
    tt, vv = tm[k0:k1 + 1], vxy[k0:k1 + 1]
    if len(tt) < 2:
        res['lo_why'] = 'no count pair after t0 - 0.5 s'
    elif np.any(br[k0:min(k1, len(br))]):
        res['lo_why'] = 'a break in the count pairs between t0 - 0.5 s and the end of the start-up window'
    else:
        s_enc = np.r_[0.0, np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(tt))]
        dd = d - e_p
        use = win & (dd >= STARTUP_USE_S_M)
        if np.nanmax(d[win]) >= STARTUP_MIN_S_M and use.any():
            te = np.interp(dd[use], s_enc, tt, left=np.nan, right=np.nan)
            ok = np.isfinite(te)
            if ok.any():
                res['lo_s'] = float(np.max(te[ok] - t[use][ok]))
            else:
                res['lo_why'] = 'the wheel path does not reach the tag displacement'
        else:
            res['lo_why'] = (f'tag path below {STARTUP_MIN_S_M * 1e3:.0f} mm (or minus e_p below '
                             f'{STARTUP_USE_S_M * 1e3:.0f} mm) in the start-up window')
    # upper: the RV yaw change against the camera yaw change, one way only
    psi = F['psi']
    m = v & np.isfinite(psi)
    pu = np.full_like(psi, np.nan)
    pu[m] = np.unwrap(psi[m])
    psi0 = float(np.median(pu[pre & m])) if (pre & m).sum() >= 5 else None
    rv_pre = (t_rv >= t0_log - PRE_S[0]) & (t_rv <= t0_log - PRE_S[1])
    if psi0 is None or rv_pre.sum() < 3:
        res['hi_why'] = 'no standing yaw before t0 (camera or RV)'
        return res
    span = (t > t0c - PRE_S[1]) & (t <= tl0 + startup_s)          # every frame of the span, valid or not
    per = float(np.median(np.diff(t)))
    ts_ = t[span]
    if not np.all(m[span]) or (len(ts_) > 1 and np.max(np.diff(ts_)) > CAM_GAP_FACTOR * per):
        res['hi_why'] = 'a rejected or lost camera frame in the start-up span: no crossing time without bridging'
        return res
    yw = np.unwrap(yaw_rv)
    y0 = float(np.median(yw[rv_pre]))
    dc = pu - psi0
    wc = span
    lim = math.radians(STARTUP_MIN_PSI_DEG)
    if not (wc & win).any() or np.nanmax(np.abs(dc[wc & win])) < lim + e_psi:
        res['hi_why'] = f'camera yaw turns less than {STARTUP_MIN_PSI_DEG:g} deg + e_psi in the start-up window'
        return res
    sel = wc & win
    sgn = np.sign(dc[sel][np.argmax(np.abs(dc[sel]))])
    sc = sgn * dc[wc]
    tc = t[wc]
    if np.any(sc < -(lim + e_psi)):
        res['hi_why'] = 'camera yaw does not turn one way in the start-up window'
        return res
    reach = np.maximum.accumulate(sc)                       # first time the camera reaches a given angle
    rv_w = (t_rv >= tl0 + D_guess - 0.2) & (t_rv <= tl0 + startup_s + D_guess + 0.2)
    di = sgn * (yw[rv_w] - y0)
    tr = t_rv[rv_w]
    if np.any(di < -lim):
        res['hi_why'] = 'RV yaw does not turn one way in the start-up window'
        return res
    est = []
    for a, b in zip(di, tr):
        aa = a - e_psi
        if a < lim or aa > reach[-1] or reach[0] >= aa:     # camera already there at the first frame: no bound
            continue
        j = int(np.argmax(reach >= aa))
        tcam = tc[j - 1] + (aa - reach[j - 1]) / (reach[j] - reach[j - 1]) * (tc[j] - tc[j - 1])
        est.append(b - tcam)
    if est:
        res['hi_s'] = float(min(est))
    else:
        res['hi_why'] = 'no RV report between the angle limit and the camera range'
    return res


# ── one run ──────────────────────────────────────────────────────────────────────────────────

def nodes(c, h, step=STEP_S):
    if h <= 0.0:
        return [float(c)]
    k = max(1, int(math.ceil(2.0 * h / step - 1e-9)))
    while True:                                   # spacing never above step, even by rounding (CAM check, CODE 17/11)
        nd = np.linspace(c - h, c + h, k + 1)
        if np.max(np.diff(nd)) <= step:
            return [float(x) for x in nd]
        k += 1


def analyse(frames_path, log_path, task, khoa=None, startup_s=STARTUP_S):
    F = read_frames(frames_path)
    rows, n_dup = read_log(log_path)
    tl = np.array([float(r['timestamp_us']) for r in rows]) * 1e-6
    try:
        win = ts.log_window(rows)
    except ValueError as exc:
        raise GioError(f'log window: {exc}')
    if win is None:
        raise GioError('log without run_active')
    ra = [r['run_active'] == '1' for r in rows]
    i0 = [i for i in range(1, len(ra)) if ra[i] and not ra[i - 1]][0]
    t0 = float(tl[i0])
    inw = (tl >= win[0]) & (tl <= win[1])
    fails = []          # the run has no finite clock bound: G1-G5 not concluded (design 3, 7)
    hong = []           # run lost to the measurement system, run again (design 7; O1 v11.34 3.5 Q2: the per-run checks)
    kiem = {}
    # section 7 checks on the log window
    kiem['n_dup_dropped'] = n_dup
    rn = {rows[k]['reset_n'] for k in np.flatnonzero(inw)}
    kiem['reset_n_const'] = len(rn) == 1
    kiem['g_valid_all'] = all(rows[k]['g_valid'] == '1' for k in np.flatnonzero(inw))
    kiem['r_valid_all'] = all(rows[k]['r_valid'] == '1' for k in np.flatnonzero(inw))
    for k_, what in (('reset_n_const', 'reset_n changes in the log window'),
                     ('g_valid_all', 'g_valid = 0 in the log window'), ('r_valid_all', 'r_valid = 0 in the log window')):
        if not kiem[k_]:
            fails.append(what)
    ag = np.r_[_col(rows, 'age_g_ms')[inw], _col(rows, 'age_r_ms')[inw]]
    ag = ag[np.isfinite(ag)]
    kiem['age_min_ms'] = float(ag.min()) if len(ag) else None
    kiem['n_age_neg'] = int(np.sum(ag < 0))
    if len(ag) and ag.min() < AGE_NEG_LIM_MS:
        fails.append(f'IMU age {ag.min():.2f} ms < {AGE_NEG_LIM_MS} ms')
    tm, rate, br, pair_ok, ip = count_pairs(rows, tl)
    pw = inw[1:] & inw[:-1]
    kiem['pairs_window'] = int(pw.sum())
    kiem['pairs_valid_share'] = float(pair_ok[pw].mean()) if pw.any() else 0.0
    if kiem['pairs_valid_share'] < PAIR_MIN:
        hong.append(f'valid count pairs {kiem["pairs_valid_share"]:.3f} < {PAIR_MIN} in the log window')
    hole = float(np.max(np.diff(tl[inw]))) if inw.sum() > 1 else math.inf
    kiem['log_hole_max_s'] = hole
    if hole > LOG_HOLE_S:
        hong.append(f'log hole {hole * 1e3:.1f} ms > {LOG_HOLE_S * 1e3:.0f} ms in the log window')
    vel = fk(rate)
    # channels
    tg, gz, _ = imu_reports(rows, tl, 'gyro_n', 'age_g_ms', 'gz_fused')
    tr, yr, _ = imu_reports(rows, tl, 'rv_n', 'age_r_ms', 'yaw_rad')
    gw = (tg >= win[0]) & (tg <= win[1])
    rw = (tr >= win[0]) & (tr <= win[1])
    kiem['gyro_gap_max_s'] = float(np.max(np.diff(tg[gw]))) if gw.sum() > 1 else math.inf
    kiem['rv_gap_max_s'] = float(np.max(np.diff(tr[rw]))) if rw.sum() > 1 else math.inf
    yru = np.unwrap(yr)
    rv_rate = np.diff(yru) / np.diff(tr)
    rv_t = 0.5 * (tr[1:] + tr[:-1])
    rv_brk_inner = np.diff(tr) > IMU_GAP_S                 # a rate over a gap is not used
    rv_rate = np.where(rv_brk_inner, np.nan, rv_rate)
    series = {
        'speed_cnt': _gapped(tm, np.hypot(vel[:, 0], vel[:, 1]), br),
        'wz_cnt': _gapped(tm, vel[:, 2], br),
        'gyro_rx': _gapped(tg, gz, np.diff(tg) > IMU_GAP_S),
        'rv_rx': _gapped(rv_t, rv_rate, np.r_[np.diff(rv_t) > IMU_GAP_S] if len(rv_t) > 1 else np.zeros(0, bool)),
    }
    cam = {'speed_cnt': F['speed'], 'wz_cnt': F['wz'], 'gyro_rx': F['wz'], 'rv_rx': F['wz']}
    try:
        coarse = _onset(tm, np.hypot(vel[:, 0], vel[:, 1])) - _onset(F['t_s'], F['speed'])
    except GioError as exc:
        raise GioError(f'coarse guess: {exc}')
    kenh = {}
    for ch in CHANNELS:
        try:
            kenh[ch] = match(F['t_s'], cam[ch], F['valid'], series[ch][0], series[ch][1], win, coarse)
        except GioError as exc:
            kenh[ch] = dict(D_s=None, ok=False, why=str(exc))
        if not kenh[ch]['ok']:
            hong.append(f'{ch}: {kenh[ch]["why"]}')
    out = dict(format=FORMAT, task=task, t0_log_s=t0, window_log_s=list(win), coarse_s=coarse, kenh=kenh, kiem=kiem,
               moc_mau=True, step_s=STEP_S, le=LE)
    if any(kenh[c]['D_s'] is None for c in CHANNELS):
        out.update(ok=False, ly_do=hong + fails, ly_do_hong=hong, ly_do_chua=fails, nodes_s=[],
                   ket_luan='hong_he_do' if hong else 'chua_ket_luan')
        return out, F, None
    L_ch = max(LOWER, key=lambda c: kenh[c]['D_s'])
    U_ch = min(UPPER, key=lambda c: kenh[c]['D_s'])
    L, U = kenh[L_ch]['D_s'], kenh[U_ch]['D_s']
    c, h = 0.5 * (L + U), LE * 0.5 * (U - L)
    out.update(L_s=L, U_s=U, L_kenh=L_ch, U_kenh=U_ch, width_s=U - L, center_s=c, half_s=h,
               gyro_minus_rv_s=kenh['gyro_rx']['D_s'] - kenh['rv_rx']['D_s'])
    if L > U:
        hong.append(f'L {L * 1e3:.1f} ms > U~ {U * 1e3:.1f} ms')
        out['nodes_s'] = []
    else:
        out['nodes_s'] = nodes(c, h)
    # matching window on the log clock: union over the four channels
    wl = [kenh[ch]['window_log_s'] for ch in CHANNELS]
    cw = [min(w[0] for w in wl), max(w[1] for w in wl)]
    td = t_deadline(task)
    out['t_deadline_s'] = td
    out['cua_so_ghep_log_s'] = cw
    out['troi_khoang_xa_s'] = max(abs(t0 + td - cw[0]), abs(cw[1] - t0))      # farthest pair of points
    rates = [kenh[ch]['rate'] for ch in (L_ch, U_ch)]
    out['rate_luot'] = None if any(r is None for r in rates) else max(rates)
    # rolling shutter spans and frame period (camera clock: matching window and scoring domain, all nodes)
    nd = out['nodes_s'] or [c]
    a_cam = min(cw[0] - max(nd), t0 - max(nd))
    b_cam = max(cw[1] - min(nd), t0 + td - min(nd))
    sel = F['valid'] & (F['t_s'] >= a_cam) & (F['t_s'] <= b_cam)
    dts = np.diff(F['t_s'])
    per = float(np.median(dts))
    out['man_trap'] = dict(
        frame_s=per, cam_span_s=[a_cam, b_cam],
        row_px=[float(np.nanmin(F['row_px'][sel])), float(np.nanmax(F['row_px'][sel]))] if sel.any() else None,
        col_px=([float(np.nanmin(F['col_px'][sel])), float(np.nanmax(F['col_px'][sel]))]
                if sel.any() and np.isfinite(F['col_px'][sel]).any() else None))
    out['pts'] = dict(median_s=per, min_s=float(dts.min()), max_s=float(dts.max()),
                      n_over_1_5=int(np.sum(dts > 1.5 * per)), n_under_0_5=int(np.sum(dts < 0.5 * per)))
    # start-up check
    kk = khoa or {}
    # C8BS-5: the allowance bounds the DIFFERENCE of two poses (standing place and frame): 2 U_p, 2 U_psi after C8
    e_p = 2.0 * kk['U_p_m'] if 'U_p_m' in kk else STARTUP_E_P_M
    e_psi = math.radians(2.0 * kk['U_psi_deg'] if 'U_psi_deg' in kk else STARTUP_E_PSI_DEG)
    st = startup(F, t0, c, tm, vel, br, tr, yr, startup_s, e_p, e_psi)
    rate_b = ((khoa or {}).get('troi_ppm') or 0.0) * 1e-6 or out['rate_luot'] or 0.0
    wid = 0.0
    if st['t_start_log_s'] is not None:
        mid = 0.5 * (cw[0] + cw[1])
        wid = rate_b * max(abs(st['t_start_log_s'] - cw[0]), abs(st['t_start_log_s'] - cw[1]), abs(st['t_start_log_s'] - mid))
    st['troi_s'] = wid
    # C8BS-4: one interval meeting the scoring interval AND both start-up bounds (a missing bound is unbounded)
    lo_b = -math.inf if st['lo_s'] is None else st['lo_s'] - wid
    hi_b = math.inf if st['hi_s'] is None else st['hi_s'] + wid
    if st['lo_s'] is None and st['hi_s'] is None:
        st['ket_qua'] = 'khong_xac_dinh'
    else:
        meet = max(c - h, lo_b) <= min(c + h, hi_b) + 1e-12
        st['ket_qua'] = 'nhat_quan' if meet else 'khong_nhat_quan'
        if not meet:
            hong.append('start-up estimate does not meet the scoring interval')
    out['khoi_dong'] = st
    # C8 range of the same level (campaign runs)
    if khoa is not None:
        rg = (khoa.get('dai') or {}).get(task)
        if rg is None:
            fails.append(f'khoa file: no C8 range for {task}')
        else:
            for k_, v_ in (('width_s', out['width_s']), ('gyro_minus_rv_s', out['gyro_minus_rv_s'])):
                lo_, hi_ = rg[k_]
                inr = lo_ - 1e-12 <= v_ <= hi_ + 1e-12
                kiem[f'{k_}_in_c8_range'] = inr
                if not inr:
                    hong.append(f'{k_} {v_ * 1e3:.1f} ms outside the C8 range [{lo_ * 1e3:.1f}, {hi_ * 1e3:.1f}] ms')
    else:
        kiem['c8_range'] = 'khong kiem (luot C8 hay chua khoa)'
    out.update(ok=not (fails or hong), ly_do=hong + fails, ly_do_hong=hong, ly_do_chua=fails,
               ket_luan='hong_he_do' if hong else ('chua_ket_luan' if fails else 'ok'))
    return out, F, dict(tm=tm, vel=vel, tg=tg, gz=gz, tr=tr, yr=yr)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--esp32', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--khoa', default=None, help='C8 lock file of c8b_u.py (drift rate, C8 ranges per level)')
    ap.add_argument('--startup-s', type=float, default=STARTUP_S)
    a = ap.parse_args(argv)
    try:
        khoa = None
        if a.khoa:
            with open(a.khoa, encoding='utf-8') as f:
                khoa = json.load(f)
        out, _, _ = analyse(a.frames, a.esp32, a.task, khoa, a.startup_s)
        ctrl, ev = controller_of(a.esp32)
        out['controller'] = ctrl
        # contract of c7_score --gio-c8b (CODE 17/11, VG1-1): the clock file names the log and frames of ITS run
        out['sha256'] = dict(esp32=cu.sha256(a.esp32), frames=cu.sha256(a.frames))
        out['nguon'] = [dict(file=os.path.basename(p), sha256=cu.sha256(p)) for p in (a.frames, a.esp32)
                        + ((a.khoa,) if a.khoa else ()) + ((ev,) if ev else ())]
        out['script'] = repo_state()
        cu.check_path(a.out)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(_clean(out), indent=1, ensure_ascii=False, allow_nan=False))
    except (GioError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_gio: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print(f'L {out.get("L_s", float("nan")) * 1e3:.1f} ms ({out.get("L_kenh")}), '
          f'U~ {out.get("U_s", float("nan")) * 1e3:.1f} ms ({out.get("U_kenh")}), '
          f'{len(out["nodes_s"])} nut; ok {out["ok"]}; {"; ".join(out["ly_do"])}')
    print(f'#ket_qua,ma=0,ok={int(out["ok"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
