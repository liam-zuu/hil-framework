#!/usr/bin/env python3
"""
c7_score.py: score one O1 floor run from the camera (frames.csv of c7_measure.py), the ESP32 log and the
sender log, with the camera rules of O1 v11.28 sections 3.2-3.5, 3.7. Writes score.json with the gate block
'gate_v3' in the B81-3 floor format of tools/rq3_log.py (two ways of using the frames, two frame domains),
checked with rq3_log.validate_gate(gate, 'HW') and check_outcome_vs_gate before it is written, plus the report
keys of the earlier version. Task b of CODE_CAM (CODE schedule 4b #4): design CLAUDE_THESIS 03_CODE/docs/
2026-11-16_Thiet_ke_viec_b_bo_cham_camera_v1.md section 3 (follows the approved O1; the frame gates of Liam 16/11
CODE_CAM local_a806ae7e are in c7_measure.py). Earlier version (O1 v11.4): L83, Liam 11/11 CODE local_7787e22f,
local_5e745270.

Inputs
  frames.csv   VALID frames of c7_measure (gates O1 3.3 (1), (2)); invalid frames are lost frames.
  ESP32 log    agv_logger CSV of ONE run. Duplicate lines (L92, S05 6.3; CODE 16/11 to CODE_CAM): lines the
               logger marked 'duplicate' in <log>.segments.csv are dropped after checking each equals an earlier
               line (c8_u_imu._drop_duplicates, the rule of c5_lifted_check); a repeated log_seq WITHOUT the
               segments file is refused (exit 2: the run is lost, L92). A line without log_seq is refused; the
               lines o1_cmd_path keeps must be, line by line, the lines left after the marked duplicates (CS1-16).
  --hang-so    the constants (the C8 gate file values; before C8 a 'de_xuat' file, the output says which):
               U_p, U_psi, the C7b residual of U_t, troi_ppm (bound of the camera / ESP32 clock-rate difference),
               t_trong, t_mat, bump, drift, the position-suspect rule, the yaw-suspect threshold, v_d, and the IMU
               block (c8_u_imu parameters, U_IMU per level). Required when trang_thai = khoa_c8, optional before
               (Codex 16/11 vong 1): duong_lenh (thresholds of the command path (2)-(4), CS1-14), cong_khung (the
               frame-gate thresholds; frames.csv must have been gated with them, track.json beside it with the frames_sha256 of the
               frames.csv scored, CS1-15, CS2-5),
               gia_dinh_nghi_vi_tri ('dung' / 'khong_dung', the C8 check of the position-suspect assumption;
               'chua_kiem' before C8, CS1-18), imu.mien_dong (the locked range of U_IMU per level: angle turned,
               rate, angular acceleration, CS1-19). Schema: load_hang_so.
  --sender     pc_sender log of the run (#test_end: send window of the command path, o1_cmd_path).

Clock (O1 3.4): t = 0 is the run_active rising edge of the log by o1_cmd_path.find_edge (first line with
run_active = 1 whose line just before by log_seq was received with 0; that line lost: run lost to the
measurement system). The video clock is matched by o1_time_sync.sync (C7b), or --offset-s with a reason; not
matched: run lost to the measurement system. Command time of a frame: t = t_cam + offset - t0.
U_t (O1 3.5 SC-12) = u_t_ghep_s (C7b residual, constants) [+ --offset-err-s] + drift + tick (drift_bound,
tick_max_s):
  drift  the clock-rate part up to t_deadline (CS1-4): rate = max(C7b |offset_end - offset_start| / distance of the
         third centres, troi_ppm) x the largest distance from the centre of the C7b window to [0, t_deadline], plus
         |offset used - mean of the thirds|; C7b short (K1): troi_ppm alone (CS1-6); --offset-s: troi_ppm x the
         largest distance from --offset-anchor-s (the command time where the offset was established) to
         [0, t_deadline], and --offset-err-s added; no anchor or error: refused (CS2-1).
         e_max (S05 4.7 (i), Pi / ESP32) is not used: it does not bound the camera clock.
  tick   the largest interval between consecutive RECEIVED log lines meeting [t0, t0 + end of the union domain]
         (CS2-2: the domain scored, not t_deadline; a lost line may hide a long tick; the pairs across the window
         ends count; CS1-5); the log not reaching both ends: no finite U_t, the run is lost to the measurement system.
         O1 v11.33 SC-12 (OG2-1, OG3-4): hang_so nhanh_tick 'giu' keeps this part (default before C8), 'bo' drops it
         and needs the clock match on sample marks at their true times (C8b; a match without it is refused); the
         branch is required in a locked file. The coverage check above holds on both branches (OG2-8).

Frames (O1 3.3)
  position-suspect  a valid frame whose centre is more than pred_m from a quadratic over +-pred_half valid
                    NON-suspect frames (both sides, the frame excluded, at least pred_min) [DE XUAT, locked at
                    C8]; iterated until the set does not change (at most 10 passes, reported). It keeps its value;
                    U_p,k = |p_k - p^_k| + U_p; it does not push the progress; the tracker's dt runs from the last
                    non-suspect frame; a suspect frame more than t_mat after the last non-suspect frame is a
                    TRACKING ERROR (R2V-1) and the later suspects of that chain are not scored. A suspect frame
                    whose search still hits the end of the window reopened to the cap is a tracking error (W11-4,
                    as every frame; CS1-20). Not converged after 10 passes: the union of the last two passes grown
                    until closed, |p - p^| predicted again from the frames outside that final set; a suspect frame
                    without nmin such neighbours has no finite U_p,k (G2-G4 undefined, n_undef; CS1-10, CS2-10).
                    C8 check of the assumption not 'dung': for G2-G4 of a suspect frame lo_k is clamped at the
                    threshold and hi_k is not finite: it gives chua_ket_luan only (O1 3.3; CS1-18, CS2-8).
  yaw check (IMU)   IMU yaw at the frame time from c8_u_imu.rv_samples / interp_yaw (period, tau_hat, scale, j_lim,
                    h_lim of the constants), aligned by c = median of wrap(psi_cam - psi_IMU) over >= n_min still
                    frames (|wz| < wz_still AND speed < v_d, CS1-11) in [t0 - pre_s, t0); the IMU yaw is built
                    from after the last reset or unbridged gap before that part (CS1-12); spread above U_psi: no
                    frame of the run is
                    compared. A frame with d = |wrap(psi_cam - psi_IMU - c)| above yaw_nghi_deg is yaw-suspect:
                    U_psi,k = d + U_IMU(level); not compared (no IMU yaw: gap, reset, run not aligned): U_psi,k
                    infinite, the frame is in n_undef of G4 and G5.
  gaps              time between consecutive valid frames in [0, end of the union domain], both ends included
                    (from t = 0 to the first valid frame, from the last to the window end, RB-2); suspect frames
                    are valid frames (they make no gap).
Tracker (O1 3.3, camera): window [s_prev - delta, s_prev + D_k], D_k = 1.5 v_max dt + delta, dt capped at t_mat;
dt above t_mat: tracking error; the upper end: search again to s_prev + 1.23 m, still the upper end: tracking
error; e_perp, s_perp in the two-sided window [s_prev - D_k, hi] (K1, K2: exact projection, heading of
Polyline.head_at, CS1-1); backward on the lane: report flag; s only grows; cut at s = S. Tracking errors and gaps
count only up to the end of the union domain (CS1-7); a tracking error at the frame that starts a long gap wins
(that frame is not before t1, CS1-8).
G1 two ways (O1 3.3 RS2-1, R2V-3): way 1 (t_done[0]) the suspect frames are used like every frame (they push the
progress), way 2 (t_done[1]) they are gaps, with speeds from the chain without them (CS1-9). The speeds of both ways
come from the frame poses of frames.csv by the derivative of c7_measure (way 1 over every valid frame), so without a
suspect frame the two ways are the same (CS2-3); the speed column of frames.csv is reported only. Completion at a
frame of the way: s >= S - D, t >= T_cmd,
t < t_deadline, and over an unbroken trailing t_d of frames of the way (a gap above t_trong breaks it): speed < v_d,
|wz| < v_d / (lx + ly), centre within D of the end point. Domains: common [0, earlier t_done], union [0, later
t_done] (no t_done: t_deadline). G1: both ways done dat, neither khong_dat, else chua_ket_luan.
G2-G5 per frame (O1 3.4, 3.5): q_k with lo_k, hi_k
  G2 e_perp +- U_p,k (figure 8: lo also - h/2, the grid distance is above the true one by up to h/2, CS1-2);
  G3 e_trk +- (U_p,k + U_t V_k);  G5 |e_theta_cmd| +- (U_psi,k + U_t W_k) (deg) (O1 v11.33 2, OG-12): V_k, W_k the
     sup of |dp_cmd/dt|, |dpsi_cmd/dt| over [t_k - U_t, t_k + U_t] on the true schedule of the level (CmdRates); the
     command holds its end values outside [0, T_cmd] (rate 0 there, OG2-2); psi_cmd on its continuous branch
     (OG2-9; figure 8: the tangent, K1, K2: the start heading, W_k = 0);
  G4 e_bao +- (U_p,k + r_c (U_psi,k + kappa_eff U_p,k)) (figure 8: + r_c kappa_max h/2, lo - h/2) when the four
     conditions hold (one projection solution within the smallest distance + 2 U_p,k, not at an end of the window
     cut at 0 and S (CS1-3), not a re-searched frame, e_perp + U_p,k <= D; K1, K2 also: one lane heading over
     J_k, no corner, CS1-1), else the bounds over J_k (O1 3.4: figure 8 on the lane grid of step h plus the two
     window ends, with h/2 and kappa_max h/2; K1, K2 exact per straight side, a corner point with the outgoing
     heading). U_psi,k infinite: hi_k infinite (n_undef), lo_k from J_k.
  K1, K2: G5 uses |psi - start heading| (as o1_gate_eval). v_cmd_max, wz_cmd_max (o1_gate.fig8_cmd_max /
  k_cmd_max, the peaks of the schedule) stay in the gate block as report and as assumption (i) of O1 3.5.
  States (rq3_log.cond_state2): khong_dat when lo of the common domain > threshold; else the three states on the
  union.
Run outcome (O1 3.5):
  hong_he_do   camera bumped (shift above bump_px: the length of the median shift vector, CS1-13; a frames.csv of the
               version before 17/11, which holds the median of the lengths, is refused: track again, CS2-4), command path
               lost (o1_cmd_path flags, and (2)-(4) above the thresholds of the constants, CS1-14; fail cause
               'duong_lenh' when it is the only cause), edge unknown, clock not matched, U_t not bounded, no valid
               frame in the window, or t1
               (first tracking error, or the start of the first gap above t_trong) without a certain violation
               before it (fail cause 'he_do');
  khong_dat    fail cause vi_pham_truoc_loi_he_do when a scored frame before t1 has lo_k > threshold for a
               condition G2-G5 (R2V-4; only when the run is not lost for another cause): numbers on the frames
               before t1 only, G1 chua_ket_luan, cat_t1 true;
  else         the gate: one khong_dat -> khong_dat; else one chua_ket_luan -> chua_ket_luan; else dat.
Reported, not deciding: placement before t = 0, pos_reset_n values, mark shift drift (design 3.6), the command path
numbers (when the constants hold no thresholds), suspect counts, frames without a prediction, yaw check numbers, the
first time the IMU motion leaves the locked U_IMU range (angle turned, gyro rate and acceleration as at C8, checked
whether or not the yaw is aligned, CS2-9; handling: Liam decides when it happens, O1 3.3).

  python tools/c7_score.py --frames <dir>/frames.csv --esp32 <run>.csv --sender <run>_sender.txt --task O1-F05 \
      --hang-so hang_so.json --out score.json [--offset-s D --offset-reason TEXT --offset-anchor-s T --offset-err-s E]
      [--k-start X,Y,DEG] [--allow-final-set] [--events <run>.csv.events.txt] [--rows rows.csv]
Last line: #ket_qua,ma=0,outcome=... scored (hong_he_do included) / 2 refused or any error (CS1-6).
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
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c7_measure as cm                                                  # noqa: E402
import c8_u_imu as cu                                                    # noqa: E402  (read only, CODE_C13)
import o1_cmd_path as ocp                                                # noqa: E402
import o1_gate                                                           # noqa: E402
import o1_gate_eval                                                      # noqa: E402
import o1_lanes                                                          # noqa: E402
import o1_time_sync                                                      # noqa: E402
import rq3_log                                                           # noqa: E402
from o1_feasibility import A as A8                                      # noqa: E402
from o1_feasibility import D, DELTA, HEAD0, T_D, T_LATE, Path8, Schedule, path_point, wrap  # noqa: E402
from o1_tasks import FIG8, FINAL_SET                                     # noqa: E402

VERSION = 'c7_score_v2'
LSUM = 0.24            # lx + ly [m], omega_d = v_d / LSUM (O1 3.4, params.py)
CAP = 1.23             # m, upper bound of the reopened window (O1 3.3 W11-1, W11-4)
R_C = 0.223            # m, O1 3.5 G4
KAPPA_EFF_FIG8 = 9.2   # 1/m, O1 3.5 G4 (0 on K1, K2)
KAPPA_MAX = 4.80       # 1/m, O1 3.4 bound of the figure-8 curvature
PRE_START_S = 0.5      # s, placement error from the valid frames of the last 0.5 s before t = 0
LOG_LEAD_S = 2.0       # s, IMU yaw built from t0 - pre_s - LOG_LEAD_S on at the earliest (imu_from)
SUSPECT_PASSES = 10
HANG_SO_VERSION = 'c7_hang_so_v1'
FAIL_CMD = 'duong_lenh'
FAIL_MEAS = 'he_do'
FAIL_VIOL = rq3_log.VIOL_BEFORE

# constants file: key -> closed range
HS_NUM = {'U_p_m': (0.0, 0.05), 'U_psi_deg': (0.0, 20.0), 'u_t_ghep_s': (0.0, 0.2), 'troi_ppm': (0.0, 5000.0),
          't_trong_s': (0.0, 0.68), 't_mat_s': (0.0, 0.68), 'bump_px': (0.0, 100.0), 'drift_px': (0.0, 100.0),
          'pred_m': (0.0, 0.2), 'yaw_nghi_deg': (0.0, 45.0), 'v_d': (0.0, 1.0)}
HS_INT = {'pred_half': (2, 60), 'pred_min': (3, 60)}
HS_IMU_NUM = {'period_s': (0.01, 0.2), 'tau_hat_s': (-0.2, 0.2), 's_plus': (-0.1, 0.1), 's_minus': (-0.1, 0.1),
              'j_lim_s': (0.0, 0.05), 'h_lim_s': (0.0, 1.0), 'pre_s': (0.5, 30.0), 'wz_still': (0.0, 1.0)}
HS_STATES = ('de_xuat', 'khoa_c8')
HS_CMD = {'cond2_m': (0.0, 1.0), 'cond3_ms': (0.0, 10000.0), 'cond4_m': (0.0, 1.0)}     # O1 3.5 command path (2)-(4)
HS_DYN = {'goc_quay_deg': (0.0, 100000.0), 'wz_dps': (0.0, 5000.0), 'alpha_dps2': (0.0, 1e6)}  # O1 3.3 U_IMU range
GIA_DINH = ('dung', 'khong_dung', 'chua_kiem')       # O1 3.3 position-suspect assumption checked at C8 (CS1-18)
REPORT_ONLY_OK = frozenset(('g2', 'g3', 'g5'))     # O1 v11.33 3.9 rule 2b: G1, G4 always decide
NHANH_TICK = ('giu', 'bo')       # O1 v11.33 SC-12 (OG2-1, OG3-4): the tick part of U_t kept / dropped, locked before C8


# ---------------------------------------------------------------- constants
def _num(d, k, lo, hi, what):
    v = d.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= v <= hi:
        raise ValueError(f'{what} {k}: {v!r} is not a number in [{lo}, {hi}]')
    return float(v)


def load_hang_so(path):
    """The constants file (module docstring), checked key by key; returns a dict with 'sha256' and 'file'."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, dict) or d.get('version') != HANG_SO_VERSION:
        raise ValueError(f'{path}: version {d.get("version") if isinstance(d, dict) else d!r}, '
                         f'expected {HANG_SO_VERSION}')
    if d.get('trang_thai') not in HS_STATES:
        raise ValueError(f'{path}: trang_thai {d.get("trang_thai")!r}, one of {HS_STATES}')
    if not isinstance(d.get('nguon'), str) or not d['nguon'].strip():
        raise ValueError(f'{path}: nguon (where the numbers come from) missing')
    out = dict(version=HANG_SO_VERSION, trang_thai=d['trang_thai'], nguon=d['nguon'], file=os.path.basename(path),
               sha256=cm.sha256(path))
    for k, (lo, hi) in HS_NUM.items():
        out[k] = _num(d, k, lo, hi, 'hang_so')
    if not out['t_trong_s'] < out['t_mat_s']:
        raise ValueError('hang_so: t_trong_s must be below t_mat_s (O1 3.3)')
    for k, (lo, hi) in HS_INT.items():
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise ValueError(f'hang_so {k}: {v!r} is not a whole number in [{lo}, {hi}]')
        out[k] = v
    if out['pred_min'] > 2 * out['pred_half']:
        raise ValueError('hang_so: pred_min above the number of neighbours 2 pred_half')
    imu = d.get('imu')
    if not isinstance(imu, dict):
        raise ValueError('hang_so imu: missing (the yaw check is part of the gate, O1 3.3)')
    oi = {k: _num(imu, k, lo, hi, 'hang_so imu') for k, (lo, hi) in HS_IMU_NUM.items()}
    n_min = imu.get('n_min')
    if isinstance(n_min, bool) or not isinstance(n_min, int) or not 5 <= n_min <= 10000:
        raise ValueError(f'hang_so imu n_min: {n_min!r} is not a whole number in [5, 10000]')
    oi['n_min'] = n_min
    u = imu.get('U_IMU_deg')
    if not isinstance(u, dict) or not u:
        raise ValueError('hang_so imu U_IMU_deg: a table level -> degrees')
    oi['U_IMU_deg'] = {m: _num(u, m, 0.0, 90.0, 'hang_so imu U_IMU_deg') for m in u}
    # blocks that only a locked file must hold (Codex 16/11 CS1-14, -15, -18, -19); before C8 they are optional
    locked = out['trang_thai'] == 'khoa_c8'
    cmd = d.get('duong_lenh')
    if cmd is None and locked:
        raise ValueError('hang_so duong_lenh: the command-path thresholds (2)-(4) are required when khoa_c8 (O1 3.5)')
    out['duong_lenh'] = None if cmd is None else {k: _num(cmd, k, lo, hi, 'hang_so duong_lenh')
                                                  for k, (lo, hi) in HS_CMD.items()}
    gk = d.get('cong_khung')
    if gk is None and locked:
        raise ValueError('hang_so cong_khung: the frame-gate thresholds are required when khoa_c8 (O1 3.3)')
    out['cong_khung'] = None if gk is None else cm.check_gates(gk, 'hang_so cong_khung')
    gd = d.get('gia_dinh_nghi_vi_tri', 'chua_kiem')
    if gd not in GIA_DINH or (locked and gd == 'chua_kiem'):
        raise ValueError(f'hang_so gia_dinh_nghi_vi_tri: {gd!r}, one of {GIA_DINH[:2] if locked else GIA_DINH} '
                         '(the C8 check of the position-suspect assumption, O1 3.3)')
    out['gia_dinh_nghi_vi_tri'] = gd
    nt = d.get('nhanh_tick', None if locked else 'giu')
    if nt not in NHANH_TICK:
        raise ValueError(f'hang_so nhanh_tick: {nt!r}, one of {NHANH_TICK} (O1 v11.33 SC-12: the tick part of U_t '
                         'kept or dropped, chosen before the C8 runs are scored; required when khoa_c8)')
    out['nhanh_tick'] = nt
    md = imu.get('mien_dong')
    if md is None and locked:
        raise ValueError('hang_so imu mien_dong: the locked range of U_IMU per level is required when khoa_c8')
    if md is not None:
        if not isinstance(md, dict) or (locked and set(md) != set(oi['U_IMU_deg'])):
            raise ValueError('hang_so imu mien_dong: a table level -> {goc_quay_deg, wz_dps, alpha_dps2}, '
                             'the same levels as U_IMU_deg')
        md = {m: {k: _num(v, k, lo, hi, f'hang_so imu mien_dong {m}') for k, (lo, hi) in HS_DYN.items()}
              if isinstance(v, dict) else _num({}, m, 0, 0, 'hang_so imu mien_dong') for m, v in md.items()}
    oi['mien_dong'] = md
    out['imu'] = oi
    return out


# ---------------------------------------------------------------- ESP32 log
def read_esp32(path):
    """(rows of o1_cmd_path.read_log sorted by log_seq, '#' lines, duplicates dropped, has traj_age_ms) with the
    L92 rule (module docstring)."""
    with open(path, newline='', encoding='utf-8') as f:
        raw = list(csv.DictReader(ln for ln in f if not ln.startswith('#')))
    if not raw:
        raise ValueError(f'{path}: no data line')
    empty = [i for i, r in enumerate(raw, start=1) if r.get('log_seq') in ('', None)]
    if empty:                                            # CS1-16: o1_cmd_path.read_log would skip them silently
        raise ValueError(f'{path}: data line {empty[0]} without log_seq ({len(empty)} lines): the log is refused (L92)')
    try:
        kept, n_seg = cu._drop_duplicates(path, raw)
    except cu.InputError as exc:
        raise ValueError(str(exc))
    if n_seg is None:
        kept, n_seg = raw, 0
    seqs = [int(r['log_seq']) for r in kept]
    if len(set(seqs)) != len(seqs):
        raise ValueError(f'{path}: repeated log_seq not marked duplicate in {os.path.basename(path)}.segments.csv: '
                         'the run is lost (L92, S05 6.3: only lines the logger marked duplicate are dropped)')
    rows, events, dup, has_age = ocp.read_log(path)
    if [r['_seq'] for r in rows] != sorted(seqs) or dup != n_seg:
        raise ValueError(f'{path}: the lines kept by o1_cmd_path ({len(rows)}, {dup} dropped) are not the lines left '
                         f'after the marked duplicates ({len(kept)}, {n_seg} dropped) (L92)')
    by_seq = {int(r['log_seq']): r for r in kept}
    # line by line, not only the counts (CS1-16). A guard: with the unique log_seq and the equal sequence lists above it
    # cannot fire (a marked duplicate equals an earlier kept line; o1_cmd_path keeps the first); mutation 17/11 confirms
    for r in rows:
        k = by_seq[r['_seq']]
        if any(r.get(c) != v for c, v in k.items()):
            raise ValueError(f'{path}: log_seq {r["_seq"]}: o1_cmd_path kept another line than the segments rule (L92)')
    return rows, events, n_seg, has_age


def clock_match(frames, log_rows, task):
    """o1_time_sync on the camera frames: path speed for the figure 8, signed body velocities for K1, K2
    (F-6); window from the log (F-4). Also the gyro channel as an independent second estimate (Y-1,
    reported only; None when it cannot be fitted)."""
    t_cam = np.array([r['t_s'] for r in frames])
    val = np.array([r['valid'] for r in frames])
    sp = np.where(val, [r['speed'] for r in frames], np.nan)
    win = o1_time_sync.log_window(log_rows)
    coarse = dict(coarse_cam=(t_cam, sp), coarse_log=o1_time_sync.log_speed(log_rows))
    if task in FIG8:
        tl, vl = o1_time_sync.log_channel(log_rows, 'speed')
        r = o1_time_sync.sync(t_cam, sp, tl, vl, window_log=win, channel='speed', **coarse)
    else:
        psi = np.array([r['psi'] for r in frames])
        vx, vy = np.array([r['vx'] for r in frames]), np.array([r['vy'] for r in frames])
        bx = np.cos(psi) * vx + np.sin(psi) * vy                 # floor velocity -> body frame
        by = -np.sin(psi) * vx + np.cos(psi) * vy
        vb = np.where(val[:, None], np.column_stack([bx, by]), np.nan)
        tl, vl = o1_time_sync.log_channel(log_rows, 'body')
        r = o1_time_sync.sync(t_cam, vb, tl, vl, window_log=win, channel='body', **coarse)
    g = None
    if 'gz_fused' in log_rows[0]:
        try:
            wz = np.where(val, [f['wz'] for f in frames], np.nan)
            tg, gz = o1_time_sync.log_channel(log_rows, 'gyro')
            g = o1_time_sync.sync(t_cam, wz, tg, gz, window_log=win, channel='gyro', **coarse)
        except ValueError:
            g = None
    return r, g


def tick_max_s(rows, t0_us, t_end_us):
    """Upper bound of the firmware tick interval over [t0, t_end] (U_t tick term, O1 3.5 SC-12): the largest
    timestamp interval between consecutive RECEIVED lines (by log_seq; a lost line between them may hide a long tick,
    and the interval of the two received lines bounds it, CS1-5) whose interval meets [t0, t_end], the pairs across
    the two window ends included. inf when the log does not reach both ends (no evidence: no finite bound)."""
    if not rows or rows[0]['_t'] > t0_us or rows[-1]['_t'] < t_end_us:
        return math.inf
    best = 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        if b['_t'] >= t0_us and a['_t'] <= t_end_us:
            best = max(best, (b['_t'] - a['_t']) * 1e-6)
    return best


# ---------------------------------------------------------------- command rates (O1 v11.33 2: bounds per frame)
class CmdRates:
    """V(t) = |dp_cmd/dt|, W(t) = |dpsi_cmd/dt| of a lane on its true schedule, and their sup over an interval.
    rate(t) -> (V, W) is exact at t (figure 8: |p'(tau)| tau_dot, |dpsi/dtau| tau_dot; K: the task speed, W = 0); it is
    continuous on [0, T_cmd] and 0 outside (the command holds its end values, OG2-2). sup(a, b): the largest value on
    a grid of step DT plus the two ends of [a, b] cut to [0, T_cmd], plus L DT / 2, L = 2 x the largest slope seen
    between grid points (a Lipschitz bound with margin: between two grid points the function cannot rise above the
    larger end by more than L DT / 2)."""
    DT = 1e-3

    def __init__(self, lane):
        self.lane, self.T = lane, float(lane.Tcmd)
        n = max(int(math.ceil(self.T / self.DT)), 1)
        self.t = np.linspace(0.0, self.T, n + 1)
        r = np.array([lane.rate(float(x)) for x in self.t])
        self.v, self.w = r[:, 0], r[:, 1]
        # L DT / 2 with L = 2 max|diff| / dt is max|diff| itself
        self.pad = [float(np.max(np.abs(np.diff(c)))) if c.size > 1 else 0.0 for c in (self.v, self.w)]

    def sup(self, a, b):
        a, b = max(a, 0.0), min(b, self.T)
        if not (math.isfinite(a) and math.isfinite(b)) or b < a:
            return 0.0, 0.0                                   # the whole interval outside the command
        i0, i1 = int(np.searchsorted(self.t, a, 'left')), int(np.searchsorted(self.t, b, 'right'))
        ea, eb = self.lane.rate(a), self.lane.rate(b)
        out = []
        for c, (ra, rb), p in zip((self.v, self.w), ((ea[0], eb[0]), (ea[1], eb[1])), self.pad):
            m = max(ra, rb, float(np.max(c[i0:i1])) if i1 > i0 else 0.0)
            out.append(m + p if m > 0.0 or p > 0.0 else 0.0)
        return out[0], out[1]


# ---------------------------------------------------------------- lanes
class Fig8Task:
    def __init__(self, task):
        self.task, self.path, self.sch = task, Path8(), Schedule(FIG8[task])
        self.Tcmd, self.vmax, self.v, self.S = self.sch.Tcmd, self.sch.vmax, self.sch.v, self.path.S
        self.end = (0.0, 0.0)
        self.start = (0.0, 0.0, HEAD0)
        self.kappa_eff = KAPPA_EFF_FIG8
        self.cmd_max = o1_gate.fig8_cmd_max(self.sch)
        self.grid = float(np.max(np.diff(self.path.s)))       # sample step of the lane (nearest, J_k)
        self.h = self.grid
        self.end_tol = 2.0 * self.grid + 1e-9                 # a grid projection at a window end
        self.head_u = np.unwrap(self.path.head)               # heading at the interpolated window ends of J_k

    def project(self, x, y, s_lo, s_hi):
        """(s_perp, e_perp, lane heading) of the nearest point among the lane samples in [s_lo, s_hi] (grid of step h)
        and the two window ends (cut at 0 and S, interpolated): the first sample inside may lie almost h from an end,
        the ends make the grid distance at most h/2 above the true one (CS1-2)."""
        s2, e, hd, _ = self.path.nearest(x, y, s_lo, s_hi)
        for se in (max(s_lo, 0.0), min(s_hi, self.S)):
            xe, ye = np.interp(se, self.path.s, self.path.x), np.interp(se, self.path.s, self.path.y)
            de = math.hypot(xe - x, ye - y)
            if de < e:
                s2, e, hd = se, de, float(wrap(np.interp(se, self.path.s, self.head_u)))
        return float(s2), float(e), float(hd)

    def to_lane(self, x, y, psi):
        return x, y, psi                                    # the floor frame is the lane frame (O1 3.2)

    def command(self, t):
        tau, _ = self.sch.tau(t)
        px, py, hc = path_point(tau)
        return float(self.path.s_of_tau(tau)), px, py, hc

    def rate(self, t):
        """(|dp_cmd/dt|, |dpsi_cmd/dt|) at t: p = (A sin tau, A/2 sin 2 tau), psi on the tangent, both times tau_dot
        (0 before 0 and after T_cmd: the schedule holds tau there)."""
        tau, td = self.sch.tau(t)
        dx, dy = A8 * math.cos(tau), A8 * math.cos(2.0 * tau)
        ddx, ddy = -A8 * math.sin(tau), -2.0 * A8 * math.sin(2.0 * tau)
        return math.hypot(dx, dy) * td, abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) * td

    def theta_err(self, psi, hc):
        return wrap(psi - hc)

    def nominal(self, head):
        return o1_gate_eval.fig8_nominal(head)


class KTask:
    def __init__(self, task, start):
        self.task, self.path, self.sch = task, o1_lanes.Lane(o1_lanes.k_corners(task)), o1_lanes.KSchedule(task)
        self.Tcmd, self.vmax, self.v, self.S = self.sch.Tcmd, self.sch.vmax, self.sch.v, self.path.S
        self.end = self.path.end
        self.start = tuple(start)
        self.kappa_eff = 0.0
        self.cmd_max = o1_gate.k_cmd_max(self.sch)
        self.poly = o1_gate_eval.Polyline(o1_lanes.k_corners(task))
        self.grid = float(np.max(np.diff(self.path.s)))
        self.h = 0.0                                        # J_k exact per straight side
        self.end_tol = 1e-9                                 # exact projection

    def project(self, x, y, s_lo, s_hi):
        """(s_perp, e_perp, lane heading) by the EXACT projection on the sides (Polyline.solutions), the heading as
        Polyline.head_at: a corner takes its outgoing side, the closing corner side 0 (CS1-1; the grid Lane keeps the
        incoming side at a corner)."""
        s2, e, hd = self.poly.solutions(x, y, s_lo, s_hi)[0]
        return s2, e, hd

    def to_lane(self, x, y, psi):
        x0, y0, h0 = self.start
        c, s = math.cos(h0), math.sin(h0)
        dx, dy = x - x0, y - y0
        return c * dx + s * dy, -s * dx + c * dy, wrap(psi - h0)

    def command(self, t):
        s_c = self.sch.s_cmd(t)
        px, py, hc = self.path.point_at(s_c)
        return s_c, px, py, hc

    def rate(self, t):
        """(|dp_cmd/dt|, 0): the speed of the task command (rpi5/o1_tasks, body frame, no rotation); the G5 reference is
        the start heading, constant (O1 3.7), so W = 0. 0 outside [0, T_cmd]."""
        if not 0.0 <= t <= self.Tcmd:
            return 0.0, 0.0
        vx, vy = self.sch.f(t)[:2]
        return math.hypot(vx, vy), 0.0

    def theta_err(self, psi, hc):
        return wrap(psi)                                   # K1, K2: against the start heading (O1 3.4)

    def nominal(self, head):
        return o1_gate_eval.k_nominal(head)


# ---------------------------------------------------------------- G4 helpers
def g_range(theta_lo, theta_hi, a, b):
    """(min, max) of g(th) = a |sin th| + b (|cos th| - 1) on [theta_lo, theta_hi]: g has period pi, kinks at
    k pi / 2 and the maxima of a |sin| + b |cos| at +-atan2(a, b) + k pi; the extremes are at those points or the
    ends. An infinite or >= pi interval gives the whole range [min(a, b) - b, hypot(a, b) - b]."""
    if not (math.isfinite(theta_lo) and math.isfinite(theta_hi)) or theta_hi - theta_lo >= math.pi:
        return min(a, b) - b, math.hypot(a, b) - b

    def g(th):
        return a * abs(math.sin(th)) + b * (abs(math.cos(th)) - 1.0)
    c = [theta_lo, theta_hi]
    phi = math.atan2(a, b)
    for base, step in ((0.0, math.pi / 2), (phi, math.pi), (-phi, math.pi)):
        th = base + math.ceil((theta_lo - base) / step) * step
        while th <= theta_hi:
            c.append(th)
            th += step
    v = [g(th) for th in c]
    return min(v), max(v)


def k_jk(lane, x, y, w_lo, w_hi, upk):
    """J_k of a K lane (O1 3.4, exact per straight side): [(e_min, e_max, lane heading)] of the parts of the sides in
    the clipped two-sided window within the smallest distance + 2 U_p,k. A J_k part that reaches the END corner of its
    side adds that corner point with the heading Polyline.head_at gives there (outgoing side; closing corner of a
    closed lane: side 0), since a true centre may project exactly onto the corner (CS1-1). The start corner of a side
    already has that side's heading."""
    P = lane.poly
    lo_s, hi_s = max(w_lo, 0.0), min(w_hi, float(P.cum[-1]))
    segs = []
    for i in range(len(P.len)):
        u_lo, u_hi = max(lo_s, P.cum[i]) - P.cum[i], min(hi_s, P.cum[i + 1]) - P.cum[i]
        if u_lo > u_hi + 1e-15:
            continue
        rel = np.array([x, y]) - P.p[i]
        uf = float(np.dot(rel, P.u[i]))
        perp = abs(float(P.u[i][0] * rel[1] - P.u[i][1] * rel[0]))
        segs.append((i, u_lo, u_hi, uf, perp))

    def dist(uf, perp, u):
        return math.hypot(u - uf, perp)
    dmin = min(dist(uf, p, min(max(uf, a), b)) for _, a, b, uf, p in segs)
    thr = dmin + 2.0 * upk
    out = []
    for i, a, b, uf, p in segs:
        u_star = min(max(uf, a), b)
        e_min = dist(uf, p, u_star)
        if e_min > thr + 1e-15:
            continue
        half = math.sqrt(max(0.0, thr * thr - p * p))      # part of the side within thr: around the foot
        ia, ib = max(a, uf - half), min(b, uf + half)
        if ia > ib:
            ia = ib = u_star
        out.append((e_min, max(dist(uf, p, ia), dist(uf, p, ib), e_min), float(P.head[i])))
        if ib >= float(P.len[i]) - 1e-12:                  # the end corner of side i is in J_k
            h_c = P.head_at(i, float(P.cum[i + 1]))
            if abs(float(wrap(h_c - P.head[i]))) > 1e-12:
                e_c = dist(uf, p, float(P.len[i]))
                out.append((e_c, e_c, h_c))
    return out


def g4_bounds_jk(lane, x, y, psi, w_lo, w_hi, upk, upsik):
    """(lo, hi) of G4 over J_k (O1 3.4 'Can G4 khi khong dung cong thuc U'): the lane points of the two-sided
    window (clipped at 0 and S) within the smallest distance + 2 U_p,k (+ h on the grid); per point j the distance e_j
    and the heading band [theta_j - U_psi,k, theta_j + U_psi,k] (+ kappa_max h / 2 on the grid). Figure 8: the grid
    of the lane plus the two clipped window ends (R2V-5, R3V-3; CS1-2)."""
    hi_v, lo_v = -math.inf, math.inf
    if isinstance(lane, KTask):
        for e_min, e_max, head in k_jk(lane, x, y, w_lo, w_hi, upk):
            psi0, al, bl = lane.nominal(head)
            th = float(wrap(psi - head - psi0))
            gmin, gmax = g_range(th - upsik, th + upsik, al, bl)
            hi_v = max(hi_v, e_max + upk + gmax)
            lo_v = min(lo_v, max(0.0, e_min - upk) + gmin)
        return lo_v, hi_v
    path = lane.path
    i0, i1 = o1_gate_eval.window_index(path, w_lo, w_hi)
    s_ends = np.array([max(w_lo, 0.0), min(w_hi, path.S)])          # the two window ends, interpolated
    ix = np.interp(s_ends, path.s, path.x)
    iy = np.interp(s_ends, path.s, path.y)
    ih = np.interp(s_ends, path.s, lane.head_u)
    px = np.r_[path.x[i0:i1], ix]
    py = np.r_[path.y[i0:i1], iy]
    ph = np.r_[path.head[i0:i1], ih]
    d = np.hypot(px - x, py - y)
    h = lane.h
    sel = np.nonzero(d <= float(d.min()) + 2.0 * upk + h)[0]
    th = wrap(psi - ph[sel])                               # psi_0 = 0 on the figure 8
    widen = upsik + KAPPA_MAX * h / 2.0
    _, al, bl = lane.nominal(0.0)
    for e, t in zip(d[sel], th):
        gmin, gmax = g_range(float(t) - widen, float(t) + widen, al, bl)
        hi_v = max(hi_v, float(e) + upk + h / 2.0 + gmax)
        lo_v = min(lo_v, max(0.0, float(e) - upk - h / 2.0) + gmin)
    return lo_v, hi_v


def one_heading(lane, x, y, w_lo, w_hi, upk):
    """K1, K2: True when every point of J_k has the same lane heading (no corner in the uncertainty region): the U
    formula of G4 (kappa_eff = 0) assumes one heading (CS1-1). Figure 8: True (kappa_eff covers the bend)."""
    if not isinstance(lane, KTask):
        return True
    hs = [h for _, _, h in k_jk(lane, x, y, w_lo, w_hi, upk)]
    return all(abs(float(wrap(h - hs[0]))) <= 1e-12 for h in hs)


def one_solution(lane, x, y, w_lo, w_hi, upk):
    """True when the projection has one solution within the smallest distance + 2 U_p,k (O1 3.5 G4 condition 1)."""
    if isinstance(lane, KTask):
        return len(lane.poly.solutions(x, y, w_lo, w_hi, tie_tol=2.0 * upk)) == 1
    idx, _, _ = o1_gate_eval.grid_projections(lane.path, x, y, w_lo, w_hi, tie_tol=2.0 * upk)
    return len(idx) == 1


# ---------------------------------------------------------------- frames
def _predict(t, x, y, sus, half, nmin):
    """|p_k - p^_k| of every frame from a quadratic over +-half frames that are NOT in sus (the frame excluded);
    NaN when fewer than nmin such frames."""
    n = len(t)
    pred = np.full(n, np.nan)
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        idx = [j for j in range(lo, hi) if j != k and not sus[j]]
        if len(idx) < nmin:
            continue
        tt = t[idx] - t[k]
        px = np.polyfit(tt, x[idx], 2)[-1]
        py = np.polyfit(tt, y[idx], 2)[-1]
        pred[k] = math.hypot(x[k] - px, y[k] - py)
    return pred


def mark_suspects(t, x, y, thr, half, nmin):
    """Position-suspect frames (module docstring): (suspect bool array, |p - p^| array (NaN without a prediction),
    passes used, converged). Converged: the prediction of every frame uses the non-suspect frames of the final set.
    Not converged after SUSPECT_PASSES (the set cycles, CS1-10, CS2-10): start from the UNION of the last two passes,
    add every frame outside the set whose prediction from the frames outside the set is above thr, until none is
    added (the set only grows, so it ends); |p - p^| of every frame is then predicted again from the frames outside
    that FINAL set only (O1 3.3: the prediction uses non-suspect frames only). A suspect frame left without nmin
    such neighbours, or whose final prediction is not above thr (unclassified, CS3-6), has NaN: U_p,k is not finite
    (evaluate). The flag is reported; the rule is locked at C8."""
    n = len(t)
    sus = np.zeros(n, bool)
    hist = []
    for it in range(1, SUSPECT_PASSES + 1):
        pred = _predict(t, x, y, sus, half, nmin)
        new = pred > thr                                   # NaN > thr is False
        hist.append(new)
        if np.array_equal(new, sus):
            return sus, pred, it, True
        sus = new
    sus = hist[-1] | hist[-2]
    for _ in range(n + 1):
        pred = _predict(t, x, y, sus, half, nmin)
        grow = ~sus & (pred > thr)
        if not grow.any():
            break
        sus = sus | grow
    # CS3-6 (Liam 17/11 CODE_CAM local_bab5dc33: 'Coi la chua ket luan (Recommended)'): a frame of the final set whose
    # final prediction is NOT above thr is unclassified (the cycle kept its flag): it stays in the set (G1 two ways and
    # the tracker treat it as a suspect, the two ways cover both cases) and gets no prediction (NaN): U_p,k not finite,
    # G2-G4 undefined, so it never gives dat or khong dat
    pred = np.where(sus & ~(pred > thr), np.nan, pred)
    return sus, pred, SUSPECT_PASSES, False


def run_tracker(t, x, y, lane, push, t_mat, t_start=0.0):
    """Camera tracker over the window frames (time order). push: frames that move the progress; the others
    (suspects) get e_perp only (module docstring). Returns per-frame dicts. at_end (a G4 condition, SC-4): the
    projection of the two-sided window sits on an end of that window AFTER the cut at 0 and S (CS1-3; the cut only
    exempts the run start and end from the TRACKING-error rule, not from the G4 formula conditions)."""
    path = lane.path
    s_prev, t_last = 0.0, t_start
    out, chain_err = [], False
    for k in range(len(t)):
        dt = float(t[k] - t_last)
        miss = dt > t_mat
        dk = 1.5 * lane.vmax * min(dt, t_mat) + DELTA
        hi = s_prev + dk
        s_star, _, _, upper = path.nearest(x[k], y[k], s_prev - DELTA, hi)
        reopen = False
        if upper:
            reopen, hi = True, s_prev + CAP
            s_star, _, _, upper = path.nearest(x[k], y[k], s_prev - DELTA, hi)
        w_lo = s_prev - dk
        s_two, e_perp, head = lane.project(x[k], y[k], w_lo, hi)
        c_lo, c_hi = max(w_lo, 0.0), min(hi, lane.S)
        at_end = bool(s_two <= c_lo + lane.end_tol or s_two >= c_hi - lane.end_tol)
        r = dict(e_perp=float(e_perp), s_perp=float(s_two), head=float(head), w_lo=w_lo, w_hi=hi, reopen=reopen,
                 at_end=at_end, back=False, trk=False, miss=bool(miss), scored=True)
        if push[k]:
            chain_err = False
            r['back'] = bool(s_two < s_prev - DELTA - 1e-12)
            r['trk'] = bool(upper or miss)
            s_prev = max(s_prev, s_star)
            t_last = float(t[k])
        elif chain_err:
            r['scored'] = False
        elif miss:                                        # R2V-1: a suspect chain longer than t_mat
            r['trk'], r['scored'], chain_err = True, False, True
        elif upper:                                       # W11-4: still on the end of the reopened window (CS1-20)
            r['trk'] = True
        r['s_car'] = s_prev
        out.append(r)
    return out


def completion_time(t, s_car, dist_end, speed, wz, way, lane, t_gap, v_d, t_deadline):
    """Earliest t of a frame of the way where the run is complete (module docstring), or None."""
    start, prev_t = None, None
    for k in np.nonzero(way)[0]:
        still = bool(np.nan_to_num(speed[k], nan=np.inf) < v_d
                     and abs(np.nan_to_num(wz[k], nan=np.inf)) < v_d / LSUM and dist_end[k] <= D)
        if not still or (prev_t is not None and t[k] - prev_t > t_gap + 1e-12):
            start = None
        if still and start is None:
            start = float(t[k])
        prev_t = float(t[k])
        held = still and t[k] - start >= T_D - 1e-9
        if held and s_car[k] >= lane.S - D and t[k] >= lane.Tcmd - 1e-9 and t[k] < t_deadline - 1e-12:
            return float(t[k])
    return None


def slice_log(L, t_from_us):
    """The c8_u_imu log dict from timestamp_us >= t_from_us on: the IMU yaw of a run is unwrapped from shortly
    before its alignment part only, so a report gap or reset long before the run (one log with several runs,
    trial 16/11 r2: a gap 77 s before t = 0) does not make the run's yaw unusable (c8_u_imu.rv_samples gives no
    yaw after an unbridged gap). The alignment constant takes up any 2 pi branch."""
    m = L['timestamp_us'] >= t_from_us
    out = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in L.items()}
    if int(m.sum()) < 2:
        raise cu.InputError(f'{L.get("path")}: fewer than two log lines from the alignment part on')
    return out


def imu_from(L_full, t_from_us, t_align_s, imu):
    """rv_samples of the log from t_from_us on, moved past every IMU reset or unbridged report gap that lies before
    the alignment part (t_align_s = t0 - pre_s): such a break would leave no yaw from it on (c8_u_imu.rv_samples
    t_break), although the yaw after it is continuous (CS1-12). The cut never moves into the alignment part; a reset
    or gap inside the alignment part or the run keeps its effect. Returns (samples, start of the slice [s])."""
    t_from = float(t_from_us)
    for _ in range(50):
        L = slice_log(L_full, t_from)
        S = cu.rv_samples(L, period_s=imu['period_s'], tau_hat=imu['tau_hat_s'], s_plus=imu['s_plus'],
                          s_minus=imu['s_minus'], j_lim=imu['j_lim_s'], h_lim=imu['h_lim_s'])
        tb = S['t_break']
        if not (math.isfinite(tb) and tb < t_align_s):
            break
        k = int(np.argmin(np.abs(S['t_all'] - tb)))
        if k + 1 >= len(S['rows']):
            break
        t_new = float(L['timestamp_us'][S['rows'][k + 1]])       # the line of the first report after the break
        if t_new <= t_from or t_new * 1e-6 > t_align_s:
            break
        t_from = t_new
    return S, t_from * 1e-6


def dyn_exceed(S, t0, t_end, lim):
    """First time [s, ESP32 clock] in [t0, t_end] where the IMU motion leaves the locked range of U_IMU (O1 3.3
    v11.26), with the quantities of the C8 range (c8_u_imu.tong / u_imu_run; CS2-9):
      angle turned  P + N of c8_u_imu.turn_totals over [t0, t]: every report pair whose span (sample and receive
                    times, both bases, CS3-4) meets the window counts in full, a suspect pair adds max|gz| x its span;
                    yaw up to t_break;
      rate          |gz| of every finite gyro report inside [t0, t_end] (a peak between two yaw reports is in the
                    gyro, not in the yaw difference; a report outside the window is coverage only, CS4-5);
      acceleration  |dgz / dt| between consecutive FINITE gyro reports inside the window (c8_u_imu.alpha_max; a NaN
                    between them is skipped, CS4-4; across a gap the difference quotient still bounds the true peak from
                    below, so an excess is certain; never across an IMU reset, CS5-1).
    The yaw pairs stop at t_break (CS4-3: a step after a reset is not a turn).
    Every excess seen is reported, even when the evidence does not cover the whole window (CS3-5); what could not be
    checked (the edges before the first and after the last finite gyro report used, CS5-2; fewer than two of them; a
    gyro gap above h_lim; values not finite; a pair across a reset; yaw reports not covering the window) is said in
    the note. Returns (time, limit key, note): (None, '', note) when nothing is seen
    above the limits (CS1-19: reported, the handling is Liam's when it happens)."""
    cand, notes = [], []
    tg, gz = np.asarray(S.get('tg', np.array([])), float), np.asarray(S.get('gz', np.array([])), float)
    if tg.size:
        h_lim = S.get('h_lim', cu.H_LIM_S)
        c0 = max(int(np.searchsorted(tg, t0, 'right')) - 1, 0)       # coverage: the reports bracketing the window
        c1 = min(int(np.searchsorted(tg, t_end, 'left')), tg.size - 1)
        tc = tg[c0:c1 + 1]
        if tc.size >= 2 and np.max(np.diff(tc)) > h_lim:
            notes.append(f'gyro gap {np.max(np.diff(tc)):.3f} s above h_lim {h_lim:.3f} s (a peak may be missed)')
        inw = (tg >= t0) & (tg <= t_end)                               # CS4-5: an excess counts inside the window only
        tw, gw = tg[inw], gz[inw]
        fin = np.isfinite(gw)
        if not fin.all():
            notes.append(f'{int((~fin).sum())} gyro values not finite (not checked)')
        tf, gf = tw[fin], gw[fin]                                      # CS4-4: consecutive FINITE reports
        if not tw.size:
            notes.append('no gyro report inside the window: rate and acceleration not checked')
        elif not tf.size:
            notes.append('no finite gyro report inside the window: rate and acceleration not checked')
        else:                                                          # CS5-2: the edges of the reports USED
            if tf[0] > t0:
                notes.append(f'gyro reports used start {tf[0] - t0:.3f} s after t = 0 (not checked before)')
            if tf[-1] < t_end:
                notes.append(f'gyro reports used end {t_end - tf[-1]:.3f} s before the window end (not checked after)')
            if tf.size < 2:
                notes.append('one finite gyro report inside the window: acceleration not checked')
        bad = np.nonzero(np.degrees(np.abs(gf)) > lim['wz_dps'])[0]
        if len(bad):
            cand.append((float(tf[bad[0]]), 'wz_dps'))
        if tf.size >= 2:
            acc = np.degrees(np.abs(np.diff(gf)) / np.diff(tf))
            # CS5-1: no difference across an IMU reset (two sensor starts). Same rule as the yaw (c8_u_imu.rv_samples):
            # a report received before the reset row is before it. The part across a reset is not checked.
            rs = np.asarray(S.get('resets', []), float)
            cross = np.array([bool(np.any((rs > a) & (rs <= b))) for a, b in zip(tf[:-1], tf[1:])], bool)
            if cross.any():
                r_in = rs[(rs > tf[0]) & (rs <= tf[-1])]
                notes.append(f'IMU reset at {r_in[0] - t0:.3f} s: acceleration across it not checked')
            bad = np.nonzero(~cross & (acc > lim['alpha_dps2']))[0]
            if len(bad):
                cand.append((float(tf[bad[0] + 1]), 'alpha_dps2'))
    else:
        notes.append('no gyro report: rate and acceleration not checked')
    ta, tr = S['t_all'], S['t_recv']
    t_br = S.get('t_break', math.inf)
    t_hi = min(t_end, t_br)
    if t_br < t_end:
        notes.append(f'IMU yaw not usable after t_break ({t_br - t0:.2f} s): angle turned checked up to it')
    if ta[0] > t0:
        notes.append('IMU yaw reports start after t = 0: angle turned not checked')
    elif t_hi >= t0:
        lo_s, hi_s = np.minimum(ta, tr), np.maximum(ta, tr)
        # the pairs turn_totals counts for [t0, te]: span from lo of the first to hi of the second meets the window
        meet = np.nonzero(hi_s[1:] >= t0)[0]
        k_cov = int(np.searchsorted(ta, t0, 'right')) - 1             # turn_totals needs a sample at or before t0
        i0 = max(0, min(int(meet[0]) if len(meet) else ta.size - 1, k_cov))
        last = np.nonzero(lo_s <= t_hi)[0]
        i1 = max(int(np.searchsorted(ta, t_hi, 'left')), int(last[-1]) + 1 if len(last) else 0)
        cap = int(np.searchsorted(ta, t_br, 'right')) - 1 if math.isfinite(t_br) else ta.size - 1
        i1 = min(i1, cap, ta.size - 1)                                 # CS4-3: no report after t_break
        if ta[i1] < t_hi:                                              # CS4-6: by time, not by array index
            notes.append(f'IMU yaw reports end {t_hi - ta[i1]:.3f} s before '
                         f'{"t_break" if t_hi < t_end else "the window end"}')
        sub = dict(S, **{k: S[k][i0:i1 + 1] for k in ('t_all', 'psi_all', 'n_all', 't_recv', 'keep')})
        lim_rad = math.radians(lim['goc_quay_deg'])
        for k in range(i0, i1 + 1):
            te = min(float(ta[k]), t_hi)
            if te < t0:
                continue
            try:
                p, n = cu.turn_totals(sub, t0, te)
            except cu.InputError as exc:
                notes.append(f'angle turned checked up to {te - t0:.2f} s: {exc}')
                break
            if p + n > lim_rad:
                cand.append((max(t0, te), 'goc_quay_deg'))
                break
    tv, key = min(cand) if cand else (None, '')
    return tv, key, '; '.join(notes)


def yaw_bounds(t_log, psi_cam, wz, speed, t0, hs, level, log_path, t_end):
    """Per-frame U_psi,k [rad] (inf: not compared) and the yaw check report (module docstring). t_log: frame times
    on the ESP32 clock [s]; psi_cam: camera yaw in the floor frame; wz, speed: camera rates of the frames (the
    alignment frames must be still in rotation AND translation, O1 3.3 'luc xe dung yen'; CS1-11). t_end: end of the
    run window on the ESP32 clock (range check of U_IMU)."""
    imu = hs['imu']
    n = len(psi_cam)
    rep = dict(compared=False, reason='', align_c_deg=None, align_spread_deg=None, n_align=0, n_suspect=0,
               n_undef=n, d_max_deg=None, U_IMU_deg=None, t_mien_dong_vuot=None, mien_dong_vuot='',
               n_after_mien_dong=0, mien_dong_note='')
    if level not in imu['U_IMU_deg']:
        rep['reason'] = f'no U_IMU for {level} in the constants'
        return np.full(n, np.inf), rep
    u_imu = math.radians(imu['U_IMU_deg'][level])
    rep['U_IMU_deg'] = imu['U_IMU_deg'][level]
    try:
        S, t_from = imu_from(cu.load_log(log_path), (t0 - imu['pre_s'] - LOG_LEAD_S) * 1e6, t0 - imu['pre_s'], imu)
        rep['log_from_s'] = t_from - t0
    except cu.InputError as exc:
        rep['reason'] = f'IMU yaw: {exc}'
        rep['mien_dong_note'] = 'no IMU samples: the U_IMU range not checked'
        return np.full(n, np.inf), rep
    # the U_IMU range is checked whether or not the camera yaw can be aligned (CS2-9)
    md = (imu.get('mien_dong') or {}).get(level)
    if md is None:
        rep['mien_dong_note'] = 'no locked U_IMU range for this level in the constants (before C8): not checked'
    else:
        tv, key, why = dyn_exceed(S, t0, t_end, md)
        rep['mien_dong_note'] = why
        if tv is not None:
            rep.update(t_mien_dong_vuot=tv - t0, mien_dong_vuot=key,
                       n_after_mien_dong=int(np.sum(np.asarray(t_log) >= tv)),
                       mien_dong_note='; '.join(x for x in ('U_IMU no longer guaranteed after this time (O1 3.3); '
                                                            'handling: Liam decides', why) if x))
    psi_i = cu.interp_yaw(S, np.asarray(t_log, float))      # NaN after an IMU reset or a gap (c8_u_imu)
    e = wrap(psi_cam - psi_i)
    still = ((t_log >= t0 - imu['pre_s']) & (t_log < t0) & np.isfinite(e)
             & (np.abs(np.nan_to_num(wz, nan=np.inf)) < imu['wz_still'])
             & (np.nan_to_num(speed, nan=np.inf) < hs['v_d']))
    rep['n_align'] = int(still.sum())
    if still.sum() < imu['n_min']:
        rep['reason'] = (f'{int(still.sum())} still frames (|wz| < wz_still, speed < v_d) with an IMU yaw before '
                         f't = 0, need {imu["n_min"]}')
        return np.full(n, np.inf), rep
    ea = e[still]
    ea = wrap(ea - ea[0]) + ea[0]                         # one branch around the first value
    c = float(np.median(ea))
    spread = float(ea.max() - ea.min())
    rep.update(align_c_deg=math.degrees(c), align_spread_deg=math.degrees(spread))
    u_psi = math.radians(hs['U_psi_deg'])
    if spread > u_psi:
        rep['reason'] = 'alignment spread above U_psi: the run is not compared (O1 3.3)'
        return np.full(n, np.inf), rep
    d = np.abs(wrap(psi_cam - psi_i - c))
    upk = np.where(np.isfinite(d), u_psi, np.inf)
    sus = np.isfinite(d) & (d > math.radians(hs['yaw_nghi_deg']))
    upk[sus] = d[sus] + u_imu
    rep.update(compared=True, n_suspect=int(sus.sum()), n_undef=int((~np.isfinite(d)).sum()),
               d_max_deg=float(np.degrees(np.nanmax(d))) if np.isfinite(d).any() else None)
    return upk, rep


def certain_violations(lo, mask, report_only=frozenset()):
    """Conditions G2-G5 with a certain violation (finite lo_k above the threshold) on the frames of mask (R2V-4);
    O1 v11.33 3.5 (OG2-3): a condition only reported at the level of the run (3.9 rule 2b) is not looked at."""
    return [c for c in o1_gate.MEASURED if c not in report_only
            and bool(np.any(np.isfinite(lo[c][mask]) & (lo[c][mask] > o1_gate.THRESHOLDS[c])))]


# ---------------------------------------------------------------- gate
def build_gate(per, dom_common, dom_union, lane, u_t, hs, g1, note):
    """gate_v3 block in the B81-3 floor format (rq3_log.validate_gate 'HW')."""
    vmax, wzmax = lane.cmd_max
    g = dict(g1=g1)
    for c in o1_gate.MEASURED:
        q, lo, hi = per[c]
        thr = o1_gate.THRESHOLDS[c]
        ac = o1_gate.aggregate(q[dom_common], lo[dom_common], hi[dom_common], thr)
        au = o1_gate.aggregate(q[dom_union], lo[dom_union], hi[dom_union], thr)
        st = rq3_log.cond_state2(ac['lo'], au['lo'], au['hi'], au['n_undef'], thr)
        g[c] = dict({k: au[k] for k in ('q', 'lo', 'hi', 'n_undef', 'n_frames', 'ratio')}, threshold=thr, state=st,
                    chung={k: ac[k] for k in ('q', 'lo', 'hi', 'n_undef')})
    ratios = [g[k]['ratio'] for k in o1_gate.MEASURED if o1_gate._finite(g[k]['ratio'])]
    return dict(version=o1_gate.GATE_VERSION, D=o1_gate.D, theta_max_deg=o1_gate.THETA_MAX_DEG, v_cmd_max=vmax,
                wz_cmd_max=wzmax, u_p=hs['U_p_m'], u_psi_deg=hs['U_psi_deg'], u_t=u_t, **g,
                **o1_gate.evaluate_outcomes(g, o1_gate.DECIDE_ALL), decide=list(o1_gate.DECIDE_ALL),
                conditions='G1,G2,G3,G4,G5', max_ratio=max(ratios) if ratios else float('nan'),
                track_err=False, note=note)


def g1_block(t_done, t_deadline, t1, cat):
    if cat:
        st = 'chua_ket_luan'
    else:
        done = [x is not None for x in t_done]
        st = 'dat' if all(done) else ('khong_dat' if not any(done) else 'chua_ket_luan')
    q = {'dat': 'true', 'khong_dat': 'false'}.get(st, 'inconclusive')
    return dict(q=q, state=st, t_done=list(t_done), t_deadline=t_deadline, t1=t1, prog_before_t1=None,
                cond25_before_deadline=None, cat_t1=bool(cat))


# ---------------------------------------------------------------- scoring
def _speeds(t, x, y, psi, idx):
    """(speed, wz) of the frames idx from the chain of those frames only, by the derivative of c7_measure (NaN
    elsewhere and with fewer than 5 frames)."""
    sp, w = np.full(len(t), np.nan), np.full(len(t), np.nan)
    if len(idx) >= 5:
        vx, vy = cm.sg_derivative(t[idx], x[idx]), cm.sg_derivative(t[idx], y[idx])
        sp[idx], w[idx] = np.hypot(vx, vy), cm.sg_derivative(t[idx], np.unwrap(psi[idx]))
    return sp, w


def evaluate(t, x, y, psi, speed, wz, lane, hs, upsi_k, u_t):
    """Camera rules on the VALID frames (command time t, increasing; pose in the lane frame); upsi_k [rad] per
    frame; speed, wz: those of frames.csv (reported per frame only, CS2-3). u_t: a number, or a function of the end
    of the union domain returning U_t (CS2-2: the tick evidence is needed over the domain scored, not after the run
    ended). Returns (summary, per-frame rows, gate inputs); None without a frame in the window; (summary, None, None)
    when U_t is not finite."""
    t, x, y, psi = (np.asarray(a, float) for a in (t, x, y, psi))
    speed, wz, upsi_k = np.asarray(speed, float), np.asarray(wz, float), np.asarray(upsi_k, float)
    deadline = lane.Tcmd + T_LATE
    win = np.nonzero((t >= 0.0) & (t < deadline - 1e-12))[0]
    if not len(win):
        return None
    pre = np.nonzero(t < 0.0)[0]
    t_start = float(t[pre[-1]]) if len(pre) else 0.0
    sus_all, pred_all, passes, conv = mark_suspects(t, x, y, hs['pred_m'], hs['pred_half'], hs['pred_min'])
    # G1 speeds of BOTH ways from the same numbers and the same derivative (CS2-3): the frame poses given here (from
    # frames.csv) and the derivative of c7_measure; way 1 over every valid frame, way 2 over the chain WITHOUT the
    # suspect frames OF THE RUN WINDOW (CS1-9: a suspect frame disturbs its neighbours; CS3-2: frames outside the
    # window are context common to both ways). No suspect frame in the window: the same speeds, so the two ways and
    # the two domains coincide (O1 3.3).
    sus_win = sus_all & (t >= 0.0) & (t < deadline - 1e-12)
    sp1, wz1 = _speeds(t, x, y, psi, np.arange(len(t)))
    sp2, wz2 = (sp1, wz1) if not sus_win.any() else _speeds(t, x, y, psi, np.nonzero(~sus_win)[0])
    tw, xw, yw, pw = t[win], x[win], y[win], psi[win]
    spw, wzw, usw = sp1[win], wz1[win], upsi_k[win]
    sus, pred = sus_all[win], pred_all[win]
    n = len(win)
    main = run_tracker(tw, xw, yw, lane, ~sus, hs['t_mat_s'], t_start)
    way1 = run_tracker(tw, xw, yw, lane, np.ones(n, bool), hs['t_mat_s'], t_start)
    d_end = np.hypot(xw - lane.end[0], yw - lane.end[1])
    td1 = completion_time(tw, np.array([r['s_car'] for r in way1]), d_end, spw, wzw, np.ones(n, bool), lane,
                          hs['t_trong_s'], hs['v_d'], deadline)
    td2 = completion_time(tw, np.array([r['s_car'] for r in main]), d_end, sp2[win], wz2[win], ~sus, lane,
                          hs['t_trong_s'], hs['v_d'], deadline)
    ends = [deadline if v is None else v for v in (td1, td2)]
    end_c, end_u = min(ends), max(ends)
    # gaps over [0, end of the union domain], both ends included (RB-2); suspects are valid frames
    gaps, prev = [], 0.0
    for j in range(n):
        if tw[j] > end_u + 1e-9:
            break
        gaps.append((prev, float(tw[j])))
        prev = float(tw[j])
    gaps.append((prev, end_u))
    glen = [b - a for a, b in gaps]
    long_gap = next((a for a, b in gaps if b - a > hs['t_trong_s'] + 1e-12), None)
    # tracking errors only inside the union domain (CS1-7: frames after the run window closed do not spoil the run)
    trk_t = next((float(tw[j]) for j in range(n) if main[j]['trk'] and tw[j] <= end_u + 1e-9), None)
    # t1 and its kind (CS1-8): on a tie the tracking error wins, so the error frame is not 'before t1'
    if trk_t is not None and (long_gap is None or trk_t <= long_gap):
        t1, t1_kind = trk_t, 'loi_theo_doi'
    elif long_gap is not None:
        t1, t1_kind = long_gap, 'khoang_trong'
    else:
        t1, t1_kind = None, ''
    # U_t (CS2-2, CS3-1): the tick evidence over the domain scored; when the log does not cover the union domain but
    # covers [0, t1], U_t over [0, t1] still bounds the frames before t1, so a certain violation before t1 (R2V-4)
    # is decided (U_t_scope 'truoc_t1'; the caller uses it for the frames before t1 only)
    u_scope = 'mien_hop'
    if callable(u_t):
        fn, u_t = u_t, u_t(end_u)
        if not math.isfinite(u_t) and t1 is not None:
            u_t1 = fn(max(0.0, t1))
            if math.isfinite(u_t1):
                u_t, u_scope = u_t1, 'truoc_t1'
    if not math.isfinite(u_t):
        return dict(t_end_union=end_u, t_done_way1=td1, t_done_way2=td2, t1=t1, t1_kind=t1_kind), None, None
    q = {c: np.full(n, np.nan) for c in o1_gate.MEASURED}
    lo = {c: np.full(n, np.nan) for c in o1_gate.MEASURED}
    hi = {c: np.full(n, np.nan) for c in o1_gate.MEASURED}
    cr = CmdRates(lane)                 # O1 v11.33 2: V_k, W_k over [t_k - U_t, t_k + U_t] (command time)
    # U_p,k: a suspect frame without a prediction from the final non-suspect set has no bound (CS2-10)
    upk_p = hs['U_p_m'] + np.where(sus, np.nan_to_num(pred, nan=np.inf), 0.0)
    hg = lane.h                     # figure 8: grid step; the grid distance is >= the true one by up to h / 2 (CS1-2)
    clamp = hs.get('gia_dinh_nghi_vi_tri', 'chua_kiem') != 'dung'
    rows, n_formula = [], 0
    for j in range(n):
        r = main[j]
        s_c, px, py, hc = lane.command(float(tw[j]))
        e_trk = math.hypot(xw[j] - px, yw[j] - py)
        eth = float(lane.theta_err(pw[j], hc))
        psi0, al, bl = lane.nominal(r['head'])
        e_bao = o1_gate_eval.g4_value(r['e_perp'], pw[j], r['head'], psi0, al, bl)
        up, us = float(upk_p[j]), float(usw[j])
        q['g2'][j], q['g3'][j], q['g4'][j], q['g5'][j] = r['e_perp'], e_trk, e_bao, abs(math.degrees(eth))
        lo['g2'][j], hi['g2'][j] = r['e_perp'] - up - hg / 2.0, r['e_perp'] + up
        vk, wk = cr.sup(float(tw[j]) - u_t, float(tw[j]) + u_t)
        u3 = up + vk * u_t
        lo['g3'][j], hi['g3'][j] = e_trk - u3, e_trk + u3
        u5 = math.degrees(us + wk * u_t) if math.isfinite(us) else math.inf
        lo['g5'][j], hi['g5'][j] = q['g5'][j] - u5, q['g5'][j] + u5
        formula = bool(math.isfinite(us) and math.isfinite(up) and not r['at_end'] and not r['reopen']
                       and r['e_perp'] + up <= D
                       and one_solution(lane, xw[j], yw[j], r['w_lo'], r['w_hi'], up)
                       and one_heading(lane, xw[j], yw[j], r['w_lo'], r['w_hi'], up))
        if not math.isfinite(up):       # CS2-10: no supported prediction, no position bound: G2-G4 undefined
            for c in ('g2', 'g3', 'g4'):
                lo[c][j], hi[c][j] = -math.inf, math.inf
        elif formula:
            u4 = up + R_C * (us + lane.kappa_eff * up + KAPPA_MAX * hg / 2.0)
            lo['g4'][j], hi['g4'][j] = e_bao - u4 - hg / 2.0, e_bao + u4
            n_formula += 1
        else:
            l4, h4 = g4_bounds_jk(lane, xw[j], yw[j], pw[j], r['w_lo'], r['w_hi'], up, us)
            lo['g4'][j] = min(l4, e_bao)
            hi['g4'][j] = max(h4, e_bao) if math.isfinite(us) else math.inf
        if clamp and sus[j]:
            # O1 3.3 (B81-7): the C8 check of the suspect assumption failed or is missing: the frame gives chua ket
            # luan only, never khong dat (lo clamped at the threshold, CS1-18) and never dat (hi not finite: the frame
            # is in n_undef, CS2-8)
            for c in ('g2', 'g3', 'g4'):
                lo[c][j] = min(lo[c][j], o1_gate.THRESHOLDS[c])
                hi[c][j] = math.inf
        e_s = -((xw[j] - px) * math.cos(hc) + (yw[j] - py) * math.sin(hc))
        rows.append(dict(t=float(tw[j]), x=float(xw[j]), y=float(yw[j]), psi=float(pw[j]), s_cmd=s_c,
                         s_car=r['s_car'], s_car_way1=way1[j]['s_car'], s_perp=r['s_perp'], e_perp=r['e_perp'],
                         e_trk=e_trk, e_bao=e_bao, e_theta_cmd_deg=math.degrees(eth),
                         e_theta_lane_deg=math.degrees(wrap(pw[j] - r['head'])), e_s=e_s, dist_end=float(d_end[j]),
                         speed=float(spw[j]), wz=float(wzw[j]), speed_csv=float(speed[win[j]]),
                         speed_way2=float(sp2[win[j]]), suspect=int(sus[j]),
                         pred_m=float(pred[j]), U_p_k=up, U_psi_k_deg=math.degrees(us),
                         w_lo=float(r['w_lo']), w_hi=float(r['w_hi']),
                         track_err=int(r['trk']), miss=int(r['miss']), back=int(r['back']), reopen=int(r['reopen']),
                         at_end=int(r['at_end']), scored=int(r['scored']), g4_formula=int(formula),
                         **{f'{b}_{c}': d[c][j] for c in o1_gate.MEASURED for b, d in (('lo', lo), ('hi', hi))}))
    summ = dict(task=lane.task, v=lane.v, T_cmd=lane.Tcmd, t_deadline=deadline, t_done_way1=td1, t_done_way2=td2,
                t_end_common=end_c, t_end_union=end_u, n_suspect_pos=int(sus.sum()),
                suspect_passes=passes, suspect_converged=conv,
                n_track_err=int(sum(r['trk'] for r in main)), t_first_track_err=trk_t,
                n_miss_frames=int(sum(r['miss'] for r in main)), n_back_frames=int(sum(r['back'] for r in main)),
                n_reopen_frames=int(sum(r['reopen'] for r in main)),
                n_unscored=int(sum(not r['scored'] for r in main)), n_g4_formula=n_formula,
                gap_max=float(max(glen)), t_gap_long=long_gap, t1=t1, t1_kind=t1_kind, frames_window=n,
                gia_dinh_nghi_vi_tri=hs.get('gia_dinh_nghi_vi_tri', 'chua_kiem'),
                n_no_pred=int(np.sum(~np.isfinite(pred))), n_suspect_no_pred=int(np.sum(sus & ~np.isfinite(pred))),
                U_t_used_s=float(u_t), U_t_scope=u_scope)
    return summ, rows, dict(q=q, lo=lo, hi=hi, t=tw, scored=np.array([r['scored'] for r in main]), gaps=glen)


def lost_output(task, reason, cause, **extra):
    return dict(task=task, version=VERSION, outcome='hong_he_do', fail_cause=cause, outcome_reasons=[reason],
                gate_v3=None, rule=f'O1 v11.28 3.3-3.5 (camera); {VERSION}', **extra)


def drift_bound(sync, offset_s, t0_s, deadline, troi_ppm, anchor_s=None):
    """(drift term of U_t [s], how) over the whole run window [0, t_deadline] (O1 3.5 SC-12; CS1-4, CS1-6). With C7b:
    rate = max(|offset_end - offset_start| / distance of the two third centres, troi_ppm) (troi_ppm alone when C7b is
    short, K1), times the largest distance from the centre of the C7b window to [0, t_deadline], plus the gap between
    the offset used (the fit of the whole window) and the offset interpolated at that centre (mean of the thirds).
    With --offset-s (no C7b; CS2-1): the offset holds (within --offset-err-s, added to U_t by the caller) at the
    command time anchor_s (--offset-anchor-s) where it was established; drift = troi_ppm x the largest distance from
    anchor_s to [0, t_deadline]. No anchor: refused (the bound needs the time the offset was right at)."""
    lim = troi_ppm * 1e-6
    if sync is None:
        if anchor_s is None or not math.isfinite(anchor_s):
            raise ValueError('--offset-s needs --offset-anchor-s (command time where the offset was established) and '
                             '--offset-err-s (its error): the drift bound runs from that time (CS2-1)')
        far = max(abs(anchor_s), abs(deadline - anchor_s))
        return lim * far, (f'troi_ppm x {far:.3f} s from the offset anchor t = {anchor_s:.3f} s (--offset-s: no C7b '
                           'evidence for the camera / ESP32 clocks)')
    w0, w1 = sync.window_cam_s
    tc = 0.5 * (w0 + w1) + offset_s - t0_s
    far = max(abs(tc), abs(deadline - tc))
    if sync.short:
        return lim * far, f'troi_ppm x {far:.3f} s (C7b short: no start / end thirds)'
    span = (w1 - w0) * 2.0 / 3.0
    rate_m = abs(sync.start_end_diff_s) / span
    mid = abs(sync.offset_s - 0.5 * (sync.offset_start_s + sync.offset_end_s))
    rate = max(rate_m, lim)
    return rate * far + mid, (f'max(C7b {rate_m * 1e6:.0f} ppm, troi_ppm) x {far:.3f} s + |offset - mean of the '
                              f'thirds| {mid * 1e3:.2f} ms')


def cmd_thresholds(cmd, hs):
    """Flags of the command-path conditions (2)-(4) against the thresholds of the constants (O1 3.5; CS1-14);
    [] when the constants hold none (before C8: values reported only). A condition that applies with no value is
    missing evidence."""
    th = hs.get('duong_lenh')
    if th is None:
        return []
    out = []
    for key, name, unit, scale in (('cond2', 'cond2_m', 'mm', 1e3), ('cond3', 'cond3_ms', 'ms', 1.0),
                                   ('cond4', 'cond4_m', 'mm', 1e3)):
        c = cmd.get(key) or {}
        if key == 'cond4' and c.get('applies') is not True:
            continue                    # ultrasonic block (False) or unknown (None, flagged by o1_cmd_path)
        v = c.get('value_ms' if key == 'cond3' else 'value_m')
        if v is None:
            out.append(f'{key}_thieu_bang_chung: no value to compare with the locked threshold')
        elif v > th[name]:
            out.append(f'{key}_vuot_nguong: {v * scale:.1f} {unit} > {th[name] * scale:g} {unit}')
    return out


def check_track_gates(hs, track_info, frames_sha256=None):
    """CS1-15: the frame gates of frames.csv (track.json beside it) must be the ones of the constants file, and
    (CS2-5) track.json must belong to the frames.csv scored: its frames_sha256 equal to the file's. Returns the report
    dict; raises ValueError on a mismatch, or when the constants hold the gates and track.json, its hash or the hash
    of the frames scored is missing. A hash mismatch is refused in every case (track.json of another file)."""
    ti = track_info or {}
    tg = ti.get('gates')
    gk = hs.get('cong_khung')
    rep = dict(track_gates=None if tg is None else {k: tg.get(k) for k in cm.GATE_KEYS}, constants_gates=gk,
               checked=False, frames_sha256_match=None)
    if ti.get('frames_sha256') and frames_sha256:
        rep['frames_sha256_match'] = ti['frames_sha256'] == frames_sha256
        if not rep['frames_sha256_match']:
            raise ValueError('track.json is not the one of the frames.csv scored (frames_sha256 differs, CS2-5): '
                             'run c7_measure track or regate again')
    if gk is None:
        rep['note'] = 'constants without cong_khung (before C8): the gates of frames.csv are reported, not checked'
        return rep
    if tg is None:
        raise ValueError('track.json beside frames.csv with the gates used is needed: the constants file holds the '
                         'frame-gate thresholds (CS1-15)')
    diff = [k for k in cm.GATE_KEYS if not (isinstance(tg.get(k), (int, float)) and float(tg[k]) == float(gk[k]))]
    if diff:
        raise ValueError(f'frames.csv was gated with other thresholds than the constants file: {diff} '
                         '(run c7_measure track or regate with --hang-so)')
    if not rep['frames_sha256_match']:
        raise ValueError('track.json without frames_sha256, or the hash of the frames.csv scored not given: the gates '
                         'cannot be tied to the file scored (CS2-5)')
    rep['checked'] = True
    return rep


def old_shift_frames(frames):
    """True when frames.csv holds the shift of the version before 17/11: shift_px (median of the lengths) without
    shift_mag_px; track always writes both together (CS1-13, CS2-4)."""
    return any(np.isfinite(r.get('shift_px', float('nan'))) and not np.isfinite(r.get('shift_mag_px', float('nan')))
               for r in frames)


def score(frames, esp32_path, task, hs, offset_s=None, offset_reason='', k_start=None, allow_final_set=False,
          sender=None, events=None, track_info=None, offset_anchor_s=None, offset_err_s=None, frames_sha256=None,
          report_only=frozenset()):
    if task not in FIG8 and task not in FINAL_SET:
        raise ValueError(f'task {task!r}: one of {tuple(FIG8) + FINAL_SET}')
    # O1 v11.33 3.9 rule 2b: conditions only reported at the level of the run (from the C8 gate file; G1, G4 always
    # decide). Used here only by the exception "violation before a measurement-system fault" (OG2-3); the deciding
    # set of the gate block follows the C8 gate file format (not yet fixed: CODE_GATE)
    report_only = frozenset(report_only)
    if not report_only <= REPORT_ONLY_OK:
        raise ValueError(f'report_only {sorted(report_only)}: only {sorted(REPORT_ONLY_OK)} can be report-only (O1 3.9 2b)')
    if task in FINAL_SET and not allow_final_set:
        raise ValueError(f'{task} is the final test set (O1 3.7): only with --allow-final-set')
    if task in FINAL_SET and k_start is None:
        raise ValueError(f'{task}: give --k-start X,Y,HEAD_DEG (nominal start pose, locked at C8)')
    if offset_s is not None:
        if not offset_reason.strip():
            raise ValueError('--offset-s needs --offset-reason (C7b skipped)')
        if offset_anchor_s is None or offset_err_s is None or not math.isfinite(offset_anchor_s) \
                or not (math.isfinite(offset_err_s) and offset_err_s >= 0.0):
            raise ValueError('--offset-s needs --offset-anchor-s (command time [s] where the offset was established) '
                             'and --offset-err-s (its error [s], >= 0) (CS2-1)')
    if old_shift_frames(frames) or (track_info or {}).get('shift_kind') == 'median_magnitude':
        raise ValueError('frames.csv holds the mark shift of the version before 17/11 (median of the lengths): the '
                         'camera-bumped decision needs the length of the median vector, which cannot be rebuilt from '
                         'it: run c7_measure track on the video again (CS2-4)')
    lane = Fig8Task(task) if task in FIG8 else KTask(task, k_start)
    gate_check = check_track_gates(hs, track_info, frames_sha256)
    if hs['trang_thai'] == 'khoa_c8' and task not in (hs['imu'].get('mien_dong') or {}):
        raise ValueError(f'hang_so imu mien_dong: no locked U_IMU range for {task}')
    rows, ev_in, n_dup, has_age = read_esp32(esp32_path)
    base = dict(hang_so=dict(file=hs['file'], sha256=hs['sha256'], trang_thai=hs['trang_thai'], nguon=hs['nguon']),
                n_duplicates=n_dup)
    i_edge, later, probs = ocp.find_edge(rows)
    if i_edge is None:
        if any(p.startswith('edge_unknown') for p in probs):
            return lost_output(task, probs[0], FAIL_MEAS, **base), []
        raise ValueError('; '.join(probs))
    t0_us = float(rows[i_edge]['_t'])
    if not any(r['valid'] for r in frames):
        return lost_output(task, 'no valid camera frame in the clip', FAIL_MEAS, **base), []
    t_cam = np.array([r['t_s'] for r in frames])
    sync, sync_gyro = None, None
    if offset_s is None:
        sync, sync_gyro = clock_match(frames, rows, task)
        if not sync.ok:
            return lost_output(task, f'clock not matched (C7b): {sync.why}', FAIL_MEAS, **base), []
        offset_s = sync.offset_s
    deadline = lane.Tcmd + T_LATE
    # drift up to t_deadline (SC-12); the --offset-s error at its anchor (CS2-1) adds to the C7b residual of the
    # constants [CACH HIEU: kept both, the larger bound]
    drift, drift_src = drift_bound(sync, offset_s, t0_us / 1e6, deadline, hs['troi_ppm'], offset_anchor_s)
    off_err = float(offset_err_s) if sync is None else 0.0
    u_t_parts = dict(ghep_s=hs['u_t_ghep_s'], offset_err_s=off_err, drift_s=drift, drift_src=drift_src)

    # O1 v11.33 SC-12 (OG2-1, OG3-4): the tick part leaves U_t only on the branch 'bo', which needs the clock match
    # on sample marks at their true times (C8b); without it that branch cannot score the run
    drop_tick = hs['nhanh_tick'] == 'bo'
    if drop_tick and not (sync is not None and getattr(sync, 'moc_mau', False)):
        raise ValueError('hang_so nhanh_tick bo: the tick part leaves U_t only with the clock match on sample marks at '
                         'their true times (C8b, O1 v11.33 SC-12); this match is not ' +
                         ('used (--offset-s)' if sync is None else 'available in o1_time_sync'))
    u_t_parts['nhanh_tick'] = hs['nhanh_tick']

    def u_t_of(end_u):
        # tick evidence over the domain scored, [t0, t0 + end of the union domain] (CS2-2: a log ending after the
        # run closed but before t_deadline does not spoil it); the drift still runs to t_deadline (SC-12). The
        # check that the log covers both ends (finite tick) holds on both branches (OG2-8).
        tick = tick_max_s(rows, t0_us, t0_us + end_u * 1e6)
        u_t_parts.update(tick_s=tick, tick_to_s=end_u)
        if not math.isfinite(tick):
            return math.inf
        return hs['u_t_ghep_s'] + off_err + drift + (0.0 if drop_tick else tick)
    vi = [k for k, r in enumerate(frames) if r['valid']]
    t_log = t_cam[vi] + offset_s
    tv = t_log - t0_us / 1e6
    pose = [lane.to_lane(frames[k]['x'], frames[k]['y'], frames[k]['psi']) for k in vi]
    x = np.array([p[0] for p in pose])
    y = np.array([p[1] for p in pose])
    psi = np.array([p[2] for p in pose])
    sp = np.array([frames[k]['speed'] for k in vi])
    wzv = np.array([frames[k]['wz'] for k in vi])
    upsi_k, yaw = yaw_bounds(t_log, np.array([frames[k]['psi'] for k in vi]), wzv, sp, t0_us / 1e6, hs, task,
                             esp32_path, t0_us / 1e6 + deadline)
    res = evaluate(tv, x, y, psi, sp, wzv, lane, hs, upsi_k, u_t_of)
    if res is None:
        return lost_output(task, 'no valid camera frame in the run window', FAIL_MEAS, **base), []
    out, prow, gi = res
    if gi is None:                     # CS1-6: no finite U_t is no evidence; never a gate with an unknown U_t
        return lost_output(task, f'U_t not bounded (the log does not cover [t0, t0 + {out["t_end_union"]:.3f} s], '
                                 f'the end of the union domain: tick {u_t_parts.get("tick_s")})',
                           FAIL_MEAS, U_t_parts=u_t_parts, **base), []
    u_t = out['U_t_used_s']
    t_end_u = out['t_end_union']
    # camera bumped: largest mark shift of ANY frame inside [0, end of the union domain]; shift_px is the length of
    # the median vector (old files refused above, CS2-4)
    t_all = t_cam + offset_s - t0_us / 1e6
    sh = np.array([r['shift_px'] for r in frames])[(t_all >= 0.0) & (t_all <= t_end_u + 1e-9)]
    shift_kind = 'median_vector'
    sh = sh[np.isfinite(sh)]
    shift_max = float(sh.max()) if len(sh) else float('nan')
    bumped = bool(len(sh) and shift_max > hs['bump_px'])
    # command path (O1 3.5): o1_cmd_path on the same log
    ev = ev_in + ocp.read_events(events or (esp32_path + '.events.txt'))
    cmd = ocp.analyse(rows, ocp.Task(task, allow_final_set), ev, has_age, ocp.read_sender(sender, task),
                      ocp.read_sender_bao(sender, task), ocp.read_sender_complete(sender, task))
    cmd_th = cmd_thresholds(cmd, hs)
    cmd_flags = list(cmd['flags']) + cmd_th
    cmd_lost = bool(cmd_flags)
    # outcome (O1 3.5)
    t1 = out['t1']
    sc = gi['scored']
    before = None
    viol_before = False
    if t1 is not None:
        gap_case = out['t1_kind'] == 'khoang_trong'
        before = sc & ((gi['t'] <= t1 + 1e-12) if gap_case else (gi['t'] < t1 - 1e-12))
        viol_before = bool(certain_violations(gi['lo'], before, report_only))
    reasons, fail_cause = [], ''
    if bumped:
        reasons.append(f'may quay bi va (dich moc {shift_max:.1f} px > {hs["bump_px"]:g} px)')
    if cmd_lost:
        reasons.append('duong lenh: ' + '; '.join(cmd_flags))
    per = {c: (gi['q'][c], gi['lo'][c], gi['hi'][c]) for c in o1_gate.MEASURED}
    cat = bool(t1 is not None and viol_before and not bumped and not cmd_lost)
    if out['U_t_scope'] == 'truoc_t1' and not cat:
        # CS3-1: U_t holds before t1 only; without a certain violation before t1 the run is lost as before
        return lost_output(task, '; '.join(reasons + [
            f'U_t not bounded over the union domain (the log does not cover [t0, t0 + {t_end_u:.3f} s]); no certain '
            f'violation before t1 = {t1:.3f} s']), FAIL_MEAS, U_t_parts=u_t_parts, **base), []
    if cat:
        dom_c = dom_u = before
        g1 = g1_block([None, None], deadline, max(0.0, t1), True)
        if out['U_t_scope'] == 'truoc_t1':
            # CS4-1: U_t holds before t1 only: the rows after it carry no bound (not finite), flagged mien_U_t = 0
            for r, m in zip(prow, before):
                r['mien_U_t'] = int(bool(m))
                if not m:
                    for c in o1_gate.MEASURED:
                        r[f'lo_{c}'] = r[f'hi_{c}'] = float('nan')
        reasons.append(f'vi pham chac chan truoc t1 = {t1:.3f} s (R2V-4)'
                       + (' (U_t tren [0, t1]: log khong phu mien hop, CS3-1)' if out['U_t_scope'] == 'truoc_t1'
                          else ''))
    else:
        dom_c = sc & (gi['t'] <= out['t_end_common'] + 1e-12)
        dom_u = sc & (gi['t'] <= t_end_u + 1e-12)
        g1 = g1_block([out['t_done_way1'], out['t_done_way2']], deadline, None if t1 is None else max(0.0, t1),
                      False)
        if out['t_gap_long'] is not None:
            reasons.append(f'khoang trong > t_trong {hs["t_trong_s"]:g} s tu t = {out["t_gap_long"]:.3f} s')
        if out['t_first_track_err'] is not None:
            reasons.append(f'loi theo doi tai t = {out["t_first_track_err"]:.3f} s')
    note = (f'camera O1 v11.28; constants {hs["trang_thai"]} ({hs["file"]}); U_t = {u_t * 1e3:.1f} ms; '
            f'yaw check {"on" if yaw["compared"] else "off: " + yaw["reason"]}')
    gate = build_gate(per, dom_c, dom_u, lane, u_t, hs, g1, note)
    if cat:
        outcome, fail_cause = 'khong_dat', FAIL_VIOL
    elif reasons:
        outcome = 'hong_he_do'
        fail_cause = FAIL_CMD if cmd_lost and not bumped and t1 is None else FAIL_MEAS
    else:
        outcome = gate['outcome_decide']
    try:
        rq3_log.validate_gate(gate, 'HW')
        rq3_log.check_outcome_vs_gate(outcome, gate, 'HW', fail_cause)
    except rq3_log.LogError as exc:
        raise RuntimeError(f'gate block refused by rq3_log (bug of this scorer): {exc}')
    # report keys of the earlier version (s05_final_table, people): on the union domain (cat: before t1)
    pr = [r for r, m in zip(prow, dom_u) if m]
    if pr:
        ep = np.array([r['e_perp'] for r in pr])
        out.update(e_perp_max=float(ep.max()), e_perp_rms=float(np.sqrt(np.mean(ep ** 2))),
                   e_trk_max=float(max(r['e_trk'] for r in pr)), e_bao_max=float(max(r['e_bao'] for r in pr)),
                   e_s_absmax=float(max(abs(r['e_s']) for r in pr)),
                   ds_absmax=float(max(abs(r['s_cmd'] - r['s_car']) for r in pr)),
                   e_theta_cmd_max_deg=float(max(abs(r['e_theta_cmd_deg']) for r in pr)),
                   stop_x=pr[-1]['x'], stop_y=pr[-1]['y'], stop_dist_end=pr[-1]['dist_end'])
        if isinstance(lane, Fig8Task):
            out.update(e_theta_lane_max_deg=float(max(abs(r['e_theta_lane_deg']) for r in pr)),
                       stop_heading_err_deg=math.degrees(wrap(pr[-1]['psi'] - HEAD0)))
        else:
            out.update(e_theta_start_max_deg=out['e_theta_cmd_max_deg'],
                       stop_heading_err_deg=math.degrees(wrap(pr[-1]['psi'])))
    frame_dt = float(np.median(np.diff(t_cam))) if len(t_cam) > 1 else 1 / 60
    big = [g for g in gi['gaps'] if g > 1.5 * frame_dt]
    out.update(n_gaps_over_frame=len(big), gaps_total_s=float(sum(big)))
    m_sw = (tv >= 0) & (tv <= t_end_u)
    sw, tsw = np.array([frames[k]['shift_px'] for k in vi])[m_sw], tv[m_sw]
    drift = None
    if len(tsw) and tsw[-1] - tsw[0] >= 2.0:
        drift = float(abs(np.nanmean(sw[tsw >= tsw[-1] - 1.0]) - np.nanmean(sw[tsw <= tsw[0] + 1.0])))
    resets = sorted({r.get('pos_reset_n') for r in rows[i_edge:] if r['_t'] <= t0_us + t_end_u * 1e6 + 1e-3
                     and r.get('pos_reset_n') not in (None, '')})
    out.update(placement(tv, x, y, psi, lane))
    out.update(base, version=VERSION, outcome=outcome, fail_cause=fail_cause, outcome_reasons=reasons,
               completed=g1['q'], gate_v3=gate, cat_t1=cat, U_t_s=u_t, U_t_parts=u_t_parts, yaw_check=yaw,
               shift_max_px=shift_max, bumped=bumped, shift_drift_px=drift,
               shift_drift_note=('quay lai clip H truoc luot sau (design 3.6)'
                                 if drift is not None and drift > hs['drift_px'] else ''),
               t0_log_us=t0_us, edge_log_seq=rows[i_edge]['_seq'], later_rises=later,
               offset_s=offset_s, offset_src='--offset-s' if sync is None else 'o1_time_sync',
               offset_reason=offset_reason if sync is None else '',
               offset_anchor_s=offset_anchor_s if sync is None else None,
               offset_err_s=offset_err_s if sync is None else None,
               sync=None if sync is None else dict(corr=sync.corr, offset_start_s=sync.offset_start_s,
                                                    offset_end_s=sync.offset_end_s,
                                                    start_end_diff_s=sync.start_end_diff_s, short=sync.short,
                                                    channel=sync.channel, window_src=sync.window_src,
                                                    widened=sync.widened),
               sync_gyro=None if sync_gyro is None else dict(offset_s=sync_gyro.offset_s, corr=sync_gyro.corr,
                                                             ok=sync_gyro.ok, why=sync_gyro.why),
               cmd_path=dict(lost=cmd_lost, flags=cmd_flags, notes=cmd['notes'],
                             cond2=cmd.get('cond2'), cond3=cmd.get('cond3'), cond4=cmd.get('cond4'),
                             thresholds=hs.get('duong_lenh') or 'none in the constants (before C8): values reported '
                                                                'only', threshold_flags=cmd_th),
               gate_check=gate_check, shift_kind=shift_kind,
               pos_reset_n_values=resets, pos_reset_mid=len(resets) > 1,
               frames_total=len(frames), frames_valid=len(vi), k_start=list(lane.start),
               thresholds={k: hs[k] for k in ('U_p_m', 'U_psi_deg', 'u_t_ghep_s', 'troi_ppm', 't_trong_s', 't_mat_s',
                                              'bump_px', 'pred_m', 'pred_half', 'pred_min', 'yaw_nghi_deg', 'v_d')},
               rule=f'O1 v11.28 3.3-3.5, 3.7 (camera); {VERSION}')
    return out, prow


def placement(t, x, y, psi, lane):
    """Mean pose of the valid frames in [-PRE_START_S, 0) against the nominal start (O1 3.2), lane frame."""
    m = (t >= -PRE_START_S) & (t < 0.0)
    if not m.any():
        return dict(placement_n=0, placement_pos_mm=None, placement_heading_deg=None)
    x0, y0, h0 = lane.start if isinstance(lane, Fig8Task) else (0.0, 0.0, 0.0)
    mx, my = float(np.mean(x[m])), float(np.mean(y[m]))
    mpsi = float(np.angle(np.mean(np.exp(1j * psi[m]))))
    return dict(placement_n=int(m.sum()), placement_pos_mm=math.hypot(mx - x0, my - y0) * 1000,
                placement_heading_deg=math.degrees(wrap(mpsi - h0)), placement_xy=[mx, my])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--esp32', required=True, help='agv_logger CSV of the run (segments file beside it)')
    ap.add_argument('--task', required=True)
    ap.add_argument('--hang-so', required=True, help='constants file (c7_hang_so_v1)')
    ap.add_argument('--sender', default=None, help='pc_sender log of the run (#test_end)')
    ap.add_argument('--events', default=None, help='default <esp32>.events.txt when it exists')
    ap.add_argument('--offset-s', type=float, default=None, help='t_log = t_cam + D, skips o1_time_sync')
    ap.add_argument('--offset-reason', default='', help='why --offset-s is given')
    ap.add_argument('--offset-anchor-s', type=float, default=None,
                    help='command time [s] where --offset-s was established (drift bound from it, CS2-1)')
    ap.add_argument('--offset-err-s', type=float, default=None, help='error of --offset-s at its anchor [s]')
    ap.add_argument('--k-start', default=None, help='X,Y,HEAD_DEG nominal start pose of K1/K2 (floor frame)')
    ap.add_argument('--allow-final-set', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--rows', default='', help='optional per-frame CSV of the window frames')
    a = ap.parse_args(argv)
    try:
        for p in (a.out, a.rows):
            if p and os.path.exists(p):
                raise ValueError(f'{p} exists: use a new file')
        hs = load_hang_so(a.hang_so)
        ks = None
        if a.k_start:
            x0, y0, h = cm._pair(a.k_start, 3, '--k-start')
            ks = (x0, y0, math.radians(h))
        tj = os.path.join(os.path.dirname(os.path.abspath(a.frames)), 'track.json')
        track_info = None
        if os.path.isfile(tj):
            with open(tj, encoding='utf-8') as f:
                track_info = json.load(f)
        fsha = cm.sha256(a.frames)
        out, rows = score(cm.read_frames(a.frames), a.esp32, a.task, hs, a.offset_s, a.offset_reason, ks,
                          a.allow_final_set, a.sender, a.events, track_info, a.offset_anchor_s, a.offset_err_s, fsha)
        out.update(frames_file=os.path.basename(a.frames), frames_sha256=fsha,
                   esp32_file=os.path.basename(a.esp32), esp32_sha256=cm.sha256(a.esp32),
                   sender_file=os.path.basename(a.sender) if a.sender else None,
                   sender_sha256=cm.sha256(a.sender) if a.sender and os.path.exists(a.sender) else None)
    except Exception as exc:            # CS1-6: any error ends with exit 2, never a traceback without a verdict
        print(f'c7_score: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    text = json.dumps(cm._json(out), indent=1, ensure_ascii=False)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    if a.rows and rows:
        with open(a.rows, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f'{out["task"]}: {out["outcome"]} {"(" + out["fail_cause"] + ")" if out.get("fail_cause") else ""} '
          f'{"; ".join(out.get("outcome_reasons", []))}')
    print(f'#ket_qua,ma=0,outcome={out["outcome"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
