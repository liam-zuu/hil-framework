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

Relative frame (J v6 2, 5; design CODE_CAM v5 CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_CAM_cham_tuong_doi_v5.md)
  c7_score.py cua-so ...      F2 cua_so_v1.jsonl (pre, post) of a run (c7_tuong_doi.cua_so)
  c7_score.py tham-chieu ...  F12 tham_chieu_v1.json (c7_tuong_doi.tham_chieu)
  c7_score.py --tuong-doi --cua-so F --tham-chieu F --truc-xe F --lan-lap F --a6 F --xoay F [--c8b-khoa F] ...
      every valid frame goes through c7_chuoi_khung (A6, centre after q) and the rigid change of frame of
      c7_tuong_doi.to_lane (lane = p_tc + R(phi) lane of the task), then the rules below run unchanged; a failed
      reference window: run lost to the reference window (hong_cua_so_tc); no position basis (phi, q, plate check):
      the t_som case (evaluate_t_som); G5 of O1-F08 report only (J v6 6). No --k-start.

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
U_t (O1 3.5 SC-12) = parts in U_t_parts (drift_bound, shutter_bound, tick_max_s). The budget of C8b v4 3 holds ONLY
at a node of the C8b time grid; without --gio-c8b the earlier budget is kept (Codex round 1 of v11.33):
  C7b match (o1_time_sync)  u_t_ghep_s (its residual, constants) + drift + tick; no shutter part.
  --offset-s                u_t_ghep_s + --offset-err-s + drift + tick; no shutter part.
  --gio-c8b FILE --nut K    D = nodes_s[K]: luoi_s (the whole grid step, largest gap of adjacent nodes, each at most
                            1 ms) + drift + shutter + tick by branch; u_t_ghep_s and --offset-err-s not added. FILE must
                            name, by sha256 {esp32, frames}, the log and frames.csv scored (Codex VG1-1).
  shutter  (--gio-c8b only) hang_so man_trap absent: one frame (median frame interval); else |alpha_r| row span +
         |alpha_c| column span of the car tag over the frames in the match window UNION [0, t_deadline] (C8B4R2-6,
         -9, C8B4R4-1; Codex VG1-2).
  drift  the clock-rate part up to t_deadline (CS1-4): rate = max(C7b |offset_end - offset_start| / distance of the
         third centres, troi_ppm) x the largest distance from the centre of the C7b window to [0, t_deadline], plus
         |offset used - mean of the thirds|; C7b short (K1): troi_ppm alone (CS1-6); --offset-s: troi_ppm x the
         largest distance from any point of --offset-anchor-s (T, or the match window A,B, C8B4R2-4) to any point of
         [0, t_deadline], and --offset-err-s added; no anchor or error: refused (CS2-1); --gio-c8b: troi_ppm x the
         largest distance from any point of the match window cua_so_ghep_log_s (command time) to any point of
         [0, t_deadline] (C8b v4 3).
         e_max (S05 4.7 (i), Pi / ESP32) is not used: it does not bound the camera clock.
  tick   the largest interval between consecutive RECEIVED log lines meeting [t0, t0 + end of the union domain]
         (CS2-2: the domain scored, not t_deadline; a lost line may hide a long tick; the pairs across the window
         ends count; CS1-5); the log not reaching both ends: no finite U_t, the run is lost to the measurement system.
         O1 v11.33 SC-12 (OG2-1, OG3-4): hang_so nhanh_tick 'giu' keeps this part (default before C8), 'bo' drops it
         and needs the clock match on sample marks at their true times (C8b, --gio-c8b; a match without it is refused,
         and the log must still cover both ends, else the run is lost as on 'giu', KT6); the
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
      --hang-so hang_so.json --out score.json [--offset-s D --offset-reason TEXT --offset-anchor-s T|A,B --offset-err-s E
      | --gio-c8b gio.json --nut K]
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
import c7_chuoi_khung as ck                                              # noqa: E402  (F1, CAM v5)
import c7_measure as cm                                                  # noqa: E402
import c7_tuong_doi as ctd                                               # noqa: E402  (relative frame, CAM v5)
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
THRESH_KEYS = ('U_p_m', 'U_psi_deg', 'u_t_ghep_s', 'troi_ppm', 't_trong_s', 't_mat_s', 'bump_px', 'pred_m',
               'pred_half', 'pred_min', 'yaw_nghi_deg', 'v_d')      # the 'thresholds' block of every output
REPORT_ONLY_OK = frozenset(('g2', 'g3', 'g5'))     # O1 v11.33 3.9 rule 2b: G1, G4 always decide
NHANH_TICK = ('giu', 'bo')       # O1 v11.33 SC-12 (OG2-1, OG3-4): the tick part of U_t kept / dropped, locked before C8
MAN_TRAP_NHANH = ('alpha', 'mot_khung')    # J v6 6: the shutter branch of a level (no row compensation, Liam 19/11)
# J v6 (= appendix O1 v11.41, 06_SPECS/2026-11-19_Spec_O1_v11.41_Phu_luc_J_v6.md): the source of the report-only G5 of
# F08 in the relative mode (J v6 6, Liam 19/11), written as gate_file_sha256 of that gate (no C8 gate file decides it)
J_V6_SHA256 = '98767aa41ac176363795ca87d37f8a7efb4944e6d8d2925d4da293ceafd29d00'


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
    mt = d.get('man_trap')                         # C8b v4 3: rolling shutter by row, column (phep 3b); none: one frame
    if mt is not None:
        if not isinstance(mt, dict) or not isinstance(mt.get('nguon'), str) or not mt['nguon'].strip():
            raise ValueError('hang_so man_trap: {alpha_r_s_px, alpha_c_s_px, nguon} (C8b v4 3, phep 3b)')
        nh = mt.get('nhanh')
        # J v6 6 (J4-13, contract F14; Codex CT1-11): a branch table by level, 'alpha' (measured, inside the 3b range)
        # or 'mot_khung', covering EVERY level (F02, F05, F08 and the K tasks: 'O1-K1', 'O1-K2' or one key 'K'); the
        # alpha numbers are needed only when a level takes 'alpha'; without a table the earlier single branch (alpha
        # for every level; the relative mode then takes one frame, man_trap_of)
        if nh is not None:
            if not isinstance(nh, dict) or not all(v in MAN_TRAP_NHANH for v in nh.values()):
                raise ValueError(f'hang_so man_trap nhanh: a table level -> one of {MAN_TRAP_NHANH}')
            bad = sorted(set(nh) - set(MAN_TRAP_KEYS))
            miss = [m for m in tuple(FIG8) + tuple(FINAL_SET) if _nhanh_of(nh, m) is None]
            if bad or miss:
                raise ValueError(f'hang_so man_trap nhanh: keys {bad} unknown, levels {miss} missing (one branch per '
                                 f'level, J v6 6)')
        need_alpha = nh is None or 'alpha' in nh.values()
        mt = {k: (_num(mt, k, -1e-3, 1e-3, 'hang_so man_trap') if need_alpha or k in mt else None)
              for k in ('alpha_r_s_px', 'alpha_c_s_px')}
        if nh is not None:
            mt['nhanh'] = dict(nh)
    out['man_trap'] = mt
    # J v6 2, 4; contract F14 (CAM v5 3.2): checked when present; required only by the relative mode (khung_tc_of,
    # Codex CT1-15: a locked file of the absolute mode without it still loads)
    out['khung_tc'] = None if d.get('khung_tc') is None else ctd.check_khung_tc(d['khung_tc'], locked)
    # F14 (CODE 19/11 after T9): the yaw-latch age threshold of J v6 5 (1), relative mode only (tuoi_yaw_max_of)
    out['tuoi_yaw_max_ms'] = (None if d.get('tuoi_yaw_max_ms') is None
                              else _num(d, 'tuoi_yaw_max_ms', 1.0, 1000.0, 'hang_so'))
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


MAN_TRAP_KEYS = ('O1-F02', 'O1-F05', 'O1-F08', 'O1-K1', 'O1-K2', 'K')


def _nhanh_of(nh, task):
    """The shutter branch of a level in the table (a K task: its own key, else 'K'); None when absent."""
    v = nh.get(task)
    if v is None and task in ('O1-K1', 'O1-K2'):
        v = nh.get('K')
    return v


def khung_tc_of(hs):
    """The reference-window gates of the relative mode: the khung_tc block, or the proposal before C8; a locked file
    without the block is refused here (contract F14)."""
    if hs.get('khung_tc') is not None:
        return hs['khung_tc']
    return ctd.check_khung_tc(None, hs['trang_thai'] == 'khoa_c8')


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
    a grid of step DT plus the two ends of [a, b] cut to [0, T_cmd], plus a margin pad = the largest change of the
    sampled rate between two adjacent grid points over the whole schedule (pad = L DT / 2 with L = 2 x that change /
    DT). This is a SAMPLED bound with a margin, not a proven one (Codex round 1 of v11.33, P3): the margin covers a
    rise between two grid points of up to twice the largest sampled slope, which holds for these schedules (smooth
    between their few breakpoints, checked by test_v11_33_cmd_rates_against_a_finite_difference_oracle, an oracle
    built from the command positions and headings, not from rate()), but is not derived from a bound on the second
    derivative."""
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
    def __init__(self, task, start, end_incoming=False):
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
        # J v6 0 (f), 3 (Codex CT1-8): at the last corner of K2 (s = S, back at the start corner) the INCOMING side,
        # for the projection and for every point j = S of J_k; the absolute mode keeps O1 3.7 (side 0)
        self.end_incoming = bool(end_incoming)

    def project(self, x, y, s_lo, s_hi):
        """(s_perp, e_perp, lane heading) by the EXACT projection on the sides (Polyline.solutions), the heading as
        Polyline.head_at: a corner takes its outgoing side, the closing corner side 0 (CS1-1; the grid Lane keeps the
        incoming side at a corner)."""
        s2, e, hd = self.poly.solutions(x, y, s_lo, s_hi)[0]
        if self.end_incoming and self.poly.closed and s2 >= float(self.poly.cum[-1]) - 1e-12:
            hd = float(self.poly.head[-1])
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
            if getattr(lane, 'end_incoming', False) and P.closed and i == len(P.len) - 1:
                continue                                    # J v6 3: the closing corner keeps its incoming side i
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


def run_tracker(t, x, y, lane, push, t_mat, t_start=0.0, cam_cover=None):
    """Camera tracker over the window frames (time order). push: frames that move the progress; the others
    (suspects) get e_perp only (module docstring). Returns per-frame dicts. at_end (a G4 condition, SC-4): the
    projection of the two-sided window sits on an end of that window AFTER the cut at 0 and S (CS1-3; the cut only
    exempts the run start and end from the TRACKING-error rule, not from the G4 formula conditions).
    cam_cover(a, b) (relative mode, Codex CT2-4; CODE 19/11 after round 4: J wins over the O1 tracker there): True
    when the CAMERA frames cover [a, b] with no hole above t_trong; a stretch longer than t_mat between two frames with
    a centre that the camera covers (frames without A6 or q: undefined frames, J v6 0, 6) is no loss of the car and no
    tracking error: the window grows with the whole time; when it passes CAP the progress is ambiguous (the lane can
    cross itself): from that frame on 'prog_undef' (progress not determined; the caller leaves G2, G4 undefined and
    G1 not concluded unless completed before). A tracking error there only on its own evidence: the projection on
    the upper end of that grown window (more progress than the car could make)."""
    path = lane.path
    s_prev, t_last = 0.0, t_start
    out, chain_err, undef = [], False, False
    for k in range(len(t)):
        dt = float(t[k] - t_last)
        covered = bool(push[k] and dt > t_mat and cam_cover is not None and cam_cover(t_last, float(t[k])))
        miss = dt > t_mat and not covered
        dk = 1.5 * lane.vmax * (dt if covered else min(dt, t_mat)) + DELTA
        undef = undef or bool(covered and dk > CAP)
        hi = s_prev + dk
        s_star, _, _, upper = path.nearest(x[k], y[k], s_prev - DELTA, hi)
        reopen = False
        if upper and not (covered and dk >= CAP):
            reopen, hi = True, s_prev + CAP
            s_star, _, _, upper = path.nearest(x[k], y[k], s_prev - DELTA, hi)
        w_lo = s_prev - dk
        s_two, e_perp, head = lane.project(x[k], y[k], w_lo, hi)
        c_lo, c_hi = max(w_lo, 0.0), min(hi, lane.S)
        at_end = bool(s_two <= c_lo + lane.end_tol or s_two >= c_hi - lane.end_tol)
        r = dict(e_perp=float(e_perp), s_perp=float(s_two), head=float(head), w_lo=w_lo, w_hi=hi, reopen=reopen,
                 at_end=at_end, back=False, trk=False, miss=bool(miss), scored=True, prog_undef=undef,
                 noi_khong_tam=covered)
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


def yaw_bounds(t_log, psi_cam, wz, speed, t0, hs, level, log_path, t_end, mode='thuong', align_mask=None,
               align_ref=None):
    """Per-frame U_psi,k [rad] (inf: not compared) and the yaw check report (module docstring). t_log: frame times
    on the ESP32 clock [s]; psi_cam: camera yaw in the floor frame; wz, speed: camera rates of the frames (the
    alignment frames must be still in rotation AND translation, O1 3.3 'luc xe dung yen'; CS1-11). t_end: end of the
    run window on the ESP32 clock (range check of U_IMU). mode 'khong_xac_dinh' (plate check undecided, J v6 5): every
    compared frame takes the bound |dpsi_cam - dpsi_IMU| + U_IMU (U_IMU of the constants carries U_IMU,D of C8B).
    rep['_psi_imu_al'] (removed by the caller before output): the IMU yaw aligned to the camera, per frame.
    align_mask (relative frame, J v6 2, Codex CT1-4): the frames of the reference set (tham_chieu_v1 ids) align the
    two yaws with the SAME frames and the SAME merge as the reference heading (trung_binh_go_vong of each, then the
    difference), every one of them needing an IMU yaw. rep['_so_duoc'], rep['_nghi_ngo'] (removed by the caller):
    per frame, compared with the IMU, and yaw-suspect (Codex CT1-6: the flags themselves, not read back from U_psi,k).
    align_ref (Codex CT2-3): (t_log, psi') of EXACTLY the frames of tham_chieu_v1 ids (times and headings from the
    chain, so a frame of F12 outside frames.csv 'valid' is not dropped): the alignment uses those frames, every one
    needing an IMU yaw; it replaces align_mask."""
    imu = hs['imu']
    n = len(psi_cam)
    rep = dict(compared=False, reason='', align_c_deg=None, align_spread_deg=None, n_align=0, n_suspect=0,
               n_undef=n, d_max_deg=None, U_IMU_deg=None, t_mien_dong_vuot=None, mien_dong_vuot='',
               n_after_mien_dong=0, mien_dong_note='',
               # Codex CT3-8: the alignment source set before any early exit
               align_set=('tham_chieu_ids' if align_ref is not None else
                          'tham_chieu' if align_mask is not None else 'dung_yen'),
               _so_duoc=np.zeros(n, bool), _nghi_ngo=np.zeros(n, bool))
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
    if align_ref is not None:
        rep['align_set'] = 'tham_chieu_ids'
        tr, pr = (np.asarray(v, float) for v in align_ref)
        pi_r = cu.interp_yaw(S, tr)
        ok_r = np.isfinite(pi_r) & np.isfinite(pr)
        rep['n_align'] = int(ok_r.sum())
        if not len(tr) or not ok_r.all() or ok_r.sum() < imu['n_min']:
            rep['reason'] = (f'{int(ok_r.sum())} of {len(tr)} reference frames (tham_chieu_v1 ids) with an IMU yaw '
                             f'(every one needed, at least {imu["n_min"]}; J v6 2)')
            return np.full(n, np.inf), rep
        c = float(wrap(ck.trung_binh_go_vong(pr) - ck.trung_binh_go_vong(pi_r)))
        ea = wrap(wrap(pr - pi_r) - c)
        spread = float(ea.max() - ea.min())
    elif align_mask is not None:
        am = np.asarray(align_mask, bool)
        still = am & np.isfinite(e)
        rep['n_align'] = int(still.sum())
        if not am.any() or still.sum() != am.sum() or still.sum() < imu['n_min']:
            rep['reason'] = (f'{int(still.sum())} of {int(am.sum())} reference frames with an IMU yaw (every one '
                             f'needed, at least {imu["n_min"]}; J v6 2)')
            return np.full(n, np.inf), rep
        c = float(wrap(ck.trung_binh_go_vong(psi_cam[am]) - ck.trung_binh_go_vong(psi_i[am])))
        ea = wrap(e[am] - c)
        spread = float(ea.max() - ea.min())
    else:
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
    if mode == 'khong_xac_dinh':
        upk = np.where(np.isfinite(d), d + u_imu, np.inf)
    rep['_psi_imu_al'] = psi_i + c
    rep['_so_duoc'], rep['_nghi_ngo'] = np.isfinite(d), sus
    rep.update(compared=True, n_suspect=int(sus.sum()), n_undef=int((~np.isfinite(d)).sum()),
               d_max_deg=float(np.degrees(np.nanmax(d))) if np.isfinite(d).any() else None)
    return upk, rep


def certain_violations(lo, mask, report_only=frozenset()):
    """Conditions G2-G5 with a certain violation (finite lo_k above the threshold) on the frames of mask (R2V-4);
    O1 v11.33 3.5 (OG2-3): a condition only reported at the level of the run (3.9 rule 2b) is not looked at."""
    return [c for c in o1_gate.MEASURED if c not in report_only
            and bool(np.any(np.isfinite(lo[c][mask]) & (lo[c][mask] > o1_gate.THRESHOLDS[c])))]


# ---------------------------------------------------------------- gate
def build_gate(per, dom_common, dom_union, lane, u_t, hs, g1, note, report_only=(), ro_sha=None):
    """gate_v3 block in the B81-3 floor format (rq3_log.validate_gate 'HW'). report_only (Codex CT1-1): conditions
    measured and reported but not deciding the run; then a gate_v3.3 block (report_only, required, gate_file_sha256
    = ro_sha, the source of the exemption) whose decide set leaves them out."""
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
    ro = [c for c in o1_gate.CONDS if c in set(report_only)]
    if ro:
        req = o1_gate.required_of(ro)
        dec = [c for c in o1_gate.DECIDE_ALL if c in set(req)]
        ver, extra = o1_gate.GATE_VERSION_V1133, dict(report_only=ro, required=list(req), gate_file_sha256=ro_sha)
    else:
        req, dec, ver, extra = None, list(o1_gate.DECIDE_ALL), o1_gate.GATE_VERSION, {}
    return dict(version=ver, D=o1_gate.D, theta_max_deg=o1_gate.THETA_MAX_DEG, v_cmd_max=vmax,
                wz_cmd_max=wzmax, u_p=hs['U_p_m'], u_psi_deg=hs['U_psi_deg'], u_t=u_t, **g,
                **o1_gate.evaluate_outcomes(g, dec, req), decide=dec, **extra,
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


def evaluate(t, x, y, psi, speed, wz, lane, hs, upsi_k, u_t, upk_add=None, rel_mode=False, gap_t=None):
    """Camera rules on the VALID frames (command time t, increasing; pose in the lane frame); upsi_k [rad] per
    frame; speed, wz: those of frames.csv (reported per frame only, CS2-3). u_t: a number, or a function of the end
    of the union domain returning U_t (CS2-2: the tick evidence is needed over the domain scored, not after the run
    ended). upk_add [m] per frame (relative frame, CAM v5 3.5: the lever arm of the phi bound; inf: the position of
    that frame is not defined) adds to U_p,k. rel_mode (J v6 6 line 166, 0b; Codex CT1-9): a frame whose yaw is not
    compared is undefined for G4 (lo not finite; q kept as a diagnostic). gap_t (relative mode, Codex CT2-4): command
    times of the CAMERA frames (every valid frame, also one without a defined centre); the gaps of the measurement
    system and t1 of a gap are taken over them, the tracker and G1 over the frames given. Returns (summary, per-frame
    rows, gate inputs); None without a frame in the window; (summary, None, None) when U_t is not finite."""
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
    cover = None
    if gap_t is not None:
        tc_all = np.sort(np.asarray(gap_t, float))

        def cover(a, b):
            seg = np.concatenate([[a], tc_all[(tc_all > a) & (tc_all < b)], [b]])
            return bool(float(np.max(np.diff(seg))) <= hs['t_trong_s'] + 1e-12)
    main = run_tracker(tw, xw, yw, lane, ~sus, hs['t_mat_s'], t_start, cam_cover=cover)
    way1 = run_tracker(tw, xw, yw, lane, np.ones(n, bool), hs['t_mat_s'], t_start, cam_cover=cover)
    d_end = np.hypot(xw - lane.end[0], yw - lane.end[1])
    td1 = completion_time(tw, np.array([r['s_car'] for r in way1]), d_end, spw, wzw, np.ones(n, bool), lane,
                          hs['t_trong_s'], hs['v_d'], deadline)
    td2 = completion_time(tw, np.array([r['s_car'] for r in main]), d_end, sp2[win], wz2[win], ~sus, lane,
                          hs['t_trong_s'], hs['v_d'], deadline)
    ends = [deadline if v is None else v for v in (td1, td2)]
    end_c, end_u = min(ends), max(ends)
    # gaps over [0, end of the union domain], both ends included (RB-2); suspects are valid frames; relative mode:
    # over the camera frames (gap_t), a frame without a defined centre is an undefined frame, not a gap (CT2-4)
    tg = tw if gap_t is None else np.sort(np.asarray(gap_t, float))
    tg = tg[(tg >= 0.0) & (tg < deadline - 1e-12)]
    gaps, prev = [], 0.0
    for tj in tg:
        if tj > end_u + 1e-9:
            break
        gaps.append((prev, float(tj)))
        prev = float(tj)
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
    if upk_add is not None:
        upk_p = upk_p + np.asarray(upk_add, float)[win]
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
            lo['g4'][j] = min(l4, e_bao) if (math.isfinite(us) or not rel_mode) else -math.inf
            hi['g4'][j] = max(h4, e_bao) if math.isfinite(us) else math.inf
        pu = bool(rel_mode and (r['prog_undef'] or way1[j]['prog_undef']))
        if pu:
            # CODE 19/11 after Codex round 4 (CT2-4): progress not determined after a long centre-less stretch: the
            # projection (G2, G4) is undefined; G3 (distance to the commanded point) does not use the progress
            for c in ('g2', 'g4'):
                lo[c][j], hi[c][j] = -math.inf, math.inf
        if clamp and sus[j]:
            # O1 3.3 (B81-7): the C8 check of the suspect assumption failed or is missing: the frame gives chua ket
            # luan only, never khong dat (lo clamped at the threshold, CS1-18) and never dat (hi not finite: the frame
            # is in n_undef, CS2-8)
            for c in ('g2', 'g3', 'g4'):
                lo[c][j] = min(lo[c][j], o1_gate.THRESHOLDS[c])
                hi[c][j] = math.inf
        e_s = -((xw[j] - px) * math.cos(hc) + (yw[j] - py) * math.sin(hc))
        rows.append(dict(t=float(tw[j]), iv=int(win[j]), x=float(xw[j]), y=float(yw[j]), psi=float(pw[j]), s_cmd=s_c,
                         s_car=r['s_car'], s_car_way1=way1[j]['s_car'], s_perp=r['s_perp'], e_perp=r['e_perp'],
                         e_trk=e_trk, e_bao=e_bao, e_theta_cmd_deg=math.degrees(eth),
                         e_theta_lane_deg=math.degrees(wrap(pw[j] - r['head'])), e_s=e_s, dist_end=float(d_end[j]),
                         speed=float(spw[j]), wz=float(wzw[j]), speed_csv=float(speed[win[j]]),
                         speed_way2=float(sp2[win[j]]), suspect=int(sus[j]),
                         pred_m=float(pred[j]), U_p_k=up, U_psi_k_deg=math.degrees(us),
                         w_lo=float(r['w_lo']), w_hi=float(r['w_hi']),
                         track_err=int(r['trk']), miss=int(r['miss']), back=int(r['back']), reopen=int(r['reopen']),
                         at_end=int(r['at_end']), scored=int(r['scored']), g4_formula=int(formula),
                         tien_do_chua_xd=int(pu), noi_khong_tam=int(bool(r['noi_khong_tam'])),
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
                U_t_used_s=float(u_t), U_t_scope=u_scope,
                t_tien_do_chua_xd=next((float(tw[j]) for j in range(n) if rel_mode
                                        and (main[j]['prog_undef'] or way1[j]['prog_undef'])), None))
    return summ, rows, dict(q=q, lo=lo, hi=hi, t=tw, scored=np.array([r['scored'] for r in main]), gaps=glen)


# ---------------------------------------------------------------- relative frame (J v6 2, 5; CAM v5 3)
def tam_doi_of(lan_lap):
    """Plate check of the run (lan_lap_v1 tam_doi_trong_luot, J v6 5): khong_doi / doi / khong_xac_dinh; missing
    check (no lan_lap_v1, null) -> khong_xac_dinh."""
    if lan_lap is None:
        return 'khong_xac_dinh'
    v = lan_lap.get('tam_doi_trong_luot')
    if isinstance(v, dict):
        v = v.get('ket_qua', v.get('gia_tri'))
    if v is False or v == 'khong_doi':
        return 'khong_doi'
    if v is True or v == 'doi':
        return 'doi'
    return 'khong_xac_dinh'


def build_rel(task, chain, tc, truc_xe, lan_lap, u_beta_cam_deg=None, sources=None):
    """The relative-frame inputs of score(): chain rows (c7_chuoi_khung), the reference tc (c7_tuong_doi.tham_chieu),
    phi by the beta path, the plate check, and whether the position basis holds (else the t_som case, J v6 5)."""
    phi, u_phi, phi_ok, phi_why = ctd.phi_beta(task, truc_xe, lan_lap, tc, u_beta_cam_deg)
    tam = tam_doi_of(lan_lap)
    why = []
    if not phi_ok:
        why += ['phi: ' + w for w in phi_why]
    if not tc['p_ok']:
        why.append('vi tri tham chieu qua q khong xac dinh')
    if not tc['psi_ok']:
        why.append('huong tham chieu sau A6 khong xac dinh')
    if tam != 'khong_doi':
        why.append(f'tam doi giua luot: {tam}')
    if tc['trang_thai'] != 'dat':
        why.append(f'tham chieu {tc["trang_thai"]}')
    return dict(task=task, chain=chain, tc=tc, phi=phi, u_phi=u_phi, phi_ok=phi_ok, phi_why=phi_why, tam_doi=tam,
                can_cu=not why, ly_do_can_cu=why, sources=sources or {})


def rel_pose(frames, vi, rel, lane):
    """Per valid frame: the pose in the nominal lane frame by the rigid change of frame (CAM v5 0.1), the floor
    heading psi' for the yaw check, the extra position bound (lever arm of u_phi; inf where the position is not
    defined) and the definition masks. No pose from the raw tag (Codex CT1-7): a frame without A6 has no heading and
    no centre (NaN), a frame without q no centre. tam_ok: the centre of the frame is defined (A6 and q; speeds need
    only that, Codex CT1-5); speed_def: tam_ok and the plate did not move in the run (a moved plate is no car motion);
    pos_def: tam_ok with the reference and phi (positions on the lane); yaw_def: A6 and a reference heading on a set of
    J v6 2 (giao or ma_mat_neo)."""
    ch = {c['frame_id']: c for c in rel['chain']}
    rows = [ch.get(int(frames[k]['idx'])) for k in vi]
    if any(r is None for r in rows):
        raise ValueError('the chain (c7_chuoi_khung) lacks valid frames of frames.csv: run it on this frames.csv')
    a6 = np.array([r['a6_ok'] for r in rows], bool)
    qok = np.array([r['q_ok'] for r in rows], bool)
    psi_p = np.array([r['psi_ma_p'] if r['a6_ok'] else np.nan for r in rows], float)
    px = np.array([r['p_x'] if r['q_ok'] else np.nan for r in rows], float)
    py = np.array([r['p_y'] if r['q_ok'] else np.nan for r in rows], float)
    tam_ok = a6 & qok & np.isfinite(px) & np.isfinite(py) & np.isfinite(psi_p)
    tc = rel['tc']
    # without a reference position or heading the speeds still hold (a rigid change of frame keeps them): any
    # defined frame stands in, flagged by pos_def / yaw_def false
    f0 = np.nonzero(tam_ok)[0]
    p_tc = tc['p_tc_m'] if tc['p_ok'] else ([float(px[f0[0]]), float(py[f0[0]])] if len(f0) else [0.0, 0.0])
    fy = np.nonzero(np.isfinite(psi_p))[0]
    psi_tc = tc['psi_tc_rad'] if tc['psi_ok'] else (float(psi_p[fy[0]]) if len(fy) else 0.0)
    psi_dd = HEAD0 if isinstance(lane, Fig8Task) else 0.0
    x, y, psi, dpsi, bx, by = ctd.to_lane(px, py, psi_p, p_tc, psi_tc, rel['phi'], psi_dd)
    r_k = np.hypot(bx, by)
    pos_def = tam_ok & bool(tc['p_ok']) & bool(rel['phi_ok'])
    u = rel['u_phi']
    arm = np.where(np.nan_to_num(r_k) > 0.0, (2.0 * r_k * math.sin(min(u, math.pi) / 2.0)) if math.isfinite(u)
                   else np.inf, 0.0)
    upk_add = np.where(pos_def, arm, np.inf)
    yaw_def = a6 & bool(tc['psi_ok']) & (tc['tap'] in ('giao', 'ma_mat_neo'))
    speed_def = tam_ok & (rel['tam_doi'] != 'doi')
    return dict(x=x, y=y, psi=psi, psi_cam=psi_p, upk_add=upk_add, pos_def=pos_def, yaw_def=yaw_def, dpsi=dpsi, q_ok=qok,
                tam_ok=tam_ok, speed_def=speed_def, bx=bx, by=by, r_k=r_k, px=px, py=py, psi_tc=psi_tc, psi_dd=psi_dd,
                tam_doi=rel['tam_doi'], can_cu=rel['can_cu'], arm=arm)


def _fin(v):
    """A float for JSON / F13: None when not finite (Codex CT1-16: no Infinity, NaN in the files)."""
    v = float(v)
    return v if math.isfinite(v) else None


def rel_row(R, iv):
    """Relative-frame columns of one frame row (F13 rows): floor centre, change of the tag heading, body frame at
    start (firmware frame), lever arm and its bound (None when not defined)."""
    return dict(x_san=_fin(R['px'][iv]), y_san=_fin(R['py'][iv]), dpsi_ma=_fin(R['dpsi'][iv]),
                x_fw=_fin(R['bx'][iv]), y_fw=_fin(R['by'][iv]), r_k=_fin(R['r_k'][iv]),
                U_phi_k=_fin(R['arm'][iv]) if R['pos_def'][iv] else None, vi_tri_xac_dinh=int(bool(R['pos_def'][iv])),
                huong_xac_dinh=int(bool(R['yaw_def'][iv])))


def rel_block(rel, R):
    """The khung_tuong_doi_v1 block of score.json (F13)."""
    tc = rel['tc']
    return dict(schema=ctd.KTD_SCHEMA, phien_ban=ctd.PHIEN_BAN, chuoi_khung=ck.PHIEN_BAN,
                nguon=rel.get('sources', {}), p_tc_m=tc['p_tc_m'], psi_tc_rad=tc['psi_tc_rad'], tap=tc['tap'],
                mat_neo=tc['mat_neo'], phi_rad=rel['phi'],
                u_phi_rad=None if not math.isfinite(rel['u_phi']) else float(rel['u_phi']),
                u_phi_trang_thai=('dat' if math.isfinite(rel['u_phi']) else
                                  ('thieu_phi' if not rel['phi_ok'] else 'thieu_u_beta_cam')),
                phi_ok=rel['phi_ok'],
                duong_phi='beta', phi_ly_do=rel['phi_why'], tam_doi=rel['tam_doi'], ca_t_som=rel['ly_do_can_cu'],
                can_cu_vi_tri=rel['can_cu'], tham_chieu_trang_thai=tc['trang_thai'],
                bien_doi=None if R is None else dict(p_dd=[0.0, 0.0], psi_dd=R['psi_dd']))


def evaluate_t_som(t, x, y, psi, R, lane, hs, upsi_k, u_t, psi_imu_al, level, yaw_ok, t_dyn=None):
    """The t_som case (J v6 5 'Thieu can cu vi tri'): G1-G4 not concluded on every frame; G5 on the common domain
    [0, t_som] and the union [0, t_deadline]; t1 = start of the first gap above t_trong (no position tracker); the
    plate 'doi': G5 only from the IMU heading (a certain violation, never 'dat'; only before t_dyn, the time the U_IMU
    range is exceeded, Codex CT1-4). yaw_ok: per frame, compared with the IMU and not yaw-suspect (yaw_bounds flags,
    Codex CT1-6). Speeds over the frames with a defined centre only (R['speed_def'], Codex CT1-5); way 2 without the
    position suspects (holes of that way); wz from the tag heading over the frames that have one, usable when every
    sample of its stencil passed the yaw check (J5-4). Same return shape as evaluate."""
    t, x, y, psi = (np.asarray(a, float) for a in (t, x, y, psi))
    upsi_k = np.asarray(upsi_k, float)
    yaw_ok = np.asarray(yaw_ok, bool)
    deadline = lane.Tcmd + T_LATE
    win = np.nonzero((t >= 0.0) & (t < deadline - 1e-12))[0]
    if not len(win):
        return None
    sd = np.asarray(R['speed_def'], bool)
    ids = np.nonzero(sd)[0]
    sp1 = _speeds(t, x, y, psi, ids)[0]
    sus = np.zeros(len(t), bool)
    if len(ids) > 2 * hs['pred_half']:
        sus[ids] = mark_suspects(t[ids], x[ids], y[ids], hs['pred_m'], hs['pred_half'], hs['pred_min'])[0]
    sus = sus & (t >= 0.0) & (t < deadline - 1e-12)
    sp2 = sp1 if not sus.any() else _speeds(t, x, y, psi, np.nonzero(sd & ~sus)[0])[0]
    yi = np.nonzero(np.isfinite(psi))[0]
    wz1 = np.full(len(t), np.nan)
    wz_ok = np.zeros(len(t), bool)
    if len(yi) >= 5:
        wz1[yi] = cm.sg_derivative(t[yi], np.unwrap(psi[yi]))
        wz_ok[yi] = ctd.wz_usable(yaw_ok[yi], cm.SG_HALF)             # J5-4 on the stencil of that derivative
    w_d = hs['v_d'] / LSUM
    # Codex CT2-2: the coverage of the window [t - t_d, t] is that of the CAMERA frames (every valid frame: a frame
    # without q still shows the tag, its speed is only undefined, which cannot rule completion out, J v6 5); the
    # speed has its own mask; way 2 treats the position-suspect frames as holes (like G1)
    ts1 = ctd.t_som(t, sd & np.isfinite(sp1), sp1, wz_ok, wz1, hs['v_d'], w_d, T_D, hs['t_trong_s'],
                    lane.Tcmd, deadline, present=None)
    ts2 = ctd.t_som(t, sd & ~sus & np.isfinite(sp2), sp2, wz_ok, wz1, hs['v_d'], w_d, T_D, hs['t_trong_s'],
                    lane.Tcmd, deadline, present=~sus)
    tsom = min(ts1, ts2)
    tw = t[win]
    n = len(win)
    end_u = deadline
    gaps, prev = [], 0.0
    for j in range(n):
        gaps.append((prev, float(tw[j])))
        prev = float(tw[j])
    gaps.append((prev, end_u))
    glen = [b - a for a, b in gaps]
    long_gap = next((a for a, b in gaps if b - a > hs['t_trong_s'] + 1e-12), None)
    t1, t1_kind = (long_gap, 'khoang_trong') if long_gap is not None else (None, '')
    u_scope = 'mien_hop'
    if callable(u_t):
        fn, u_t = u_t, u_t(end_u)
        if not math.isfinite(u_t) and t1 is not None:
            u_t1 = fn(max(0.0, t1))
            if math.isfinite(u_t1):
                u_t, u_scope = u_t1, 'truoc_t1'
    if not math.isfinite(u_t):
        return dict(t_end_union=end_u, t_done_way1=None, t_done_way2=None, t1=t1, t1_kind=t1_kind,
                    t_som_s=tsom), None, None
    q = {c: np.full(n, np.nan) for c in o1_gate.MEASURED}
    lo = {c: np.full(n, -np.inf) for c in o1_gate.MEASURED}
    hi = {c: np.full(n, np.inf) for c in o1_gate.MEASURED}
    cr = CmdRates(lane)
    doi = R['tam_doi'] == 'doi'
    u_imu = math.radians(hs['imu']['U_IMU_deg'].get(level, math.inf)) if doi else 0.0
    psi_imu_lane = None
    if doi and psi_imu_al is not None:
        psi_imu_lane = ctd.rel_heading(psi_imu_al, R['psi_tc'], R['psi_dd'])[0]
    rows = []
    for j in range(n):
        k = int(win[j])
        _, _, _, hc = lane.command(float(tw[j]))
        vk, wk = cr.sup(float(tw[j]) - u_t, float(tw[j]) + u_t)
        eth = float(lane.theta_err(psi[k], hc)) if math.isfinite(psi[k]) else math.nan
        q5 = abs(math.degrees(eth))
        us = float(upsi_k[k])
        if doi:
            pi_ = np.nan if psi_imu_lane is None else float(psi_imu_lane[k])
            before_dyn = t_dyn is None or float(tw[j]) < t_dyn - 1e-12
            if math.isfinite(pi_) and math.isfinite(u_imu) and before_dyn and R['yaw_def'][k]:
                qi = abs(math.degrees(float(lane.theta_err(pi_, hc))))
                q['g5'][j], lo['g5'][j] = qi, qi - math.degrees(u_imu + wk * u_t)
        elif math.isfinite(q5) and math.isfinite(us):
            q['g5'][j] = q5
            u5 = math.degrees(us + wk * u_t)
            lo['g5'][j], hi['g5'][j] = q5 - u5, q5 + u5
        else:
            q['g5'][j] = q5                                          # undefined: kept as a diagnostic only
        rows.append(dict(t=float(tw[j]), iv=k, x=float(x[k]), y=float(y[k]), psi=float(psi[k]),
                         e_theta_cmd_deg=math.degrees(eth), speed=float(sp1[k]), wz=float(wz1[k]),
                         wz_dung_duoc=int(bool(wz_ok[k])), U_psi_k_deg=math.degrees(us), scored=1,
                         suspect=int(sus[k]), pred_m=math.nan,
                         **{f'{b}_{c}': d[c][j] for c in o1_gate.MEASURED for b, d in (('lo', lo), ('hi', hi))}))
    summ = dict(task=lane.task, v=lane.v, T_cmd=lane.Tcmd, t_deadline=deadline, t_done_way1=None, t_done_way2=None,
                t_end_common=tsom, t_end_union=end_u, t_som_s=tsom, t_som_way1_s=ts1, t_som_way2_s=ts2,
                n_suspect_pos=int(sus.sum()), n_track_err=0, t_first_track_err=None, gap_max=float(max(glen)),
                t_gap_long=long_gap, t1=t1, t1_kind=t1_kind, frames_window=n, U_t_used_s=float(u_t),
                U_t_scope=u_scope, ca_t_som=True)
    return summ, rows, dict(q=q, lo=lo, hi=hi, t=tw, scored=np.ones(n, bool), gaps=glen)


def add_undefined(prow, gi, tv, idx, deadline, R):
    """Frames of the run window left out of the position rules (no centre, Codex CT1-7: A6 or q missing on that
    frame) put back as UNDEFINED frames of every condition (n_undef, J v6 0b), rows kept for the report (F13: their
    raw tag and state); the evaluated frames keep their order. idx: indices (in the valid frames) of the frames left
    out. Returns (rows, gi) sorted by command time."""
    add = [int(k) for k in idx if 0.0 <= tv[k] < deadline - 1e-12]
    if not add:
        return prow, gi
    new = []
    for k in add:
        r = dict(t=float(tv[k]), iv=k, x=math.nan, y=math.nan, psi=math.nan, e_perp=math.nan, e_trk=math.nan,
                 e_bao=math.nan, suspect=0, pred_m=math.nan, scored=1, U_p_k=math.inf, U_psi_k_deg=math.inf,
                 khung_thieu_tam=1)
        for c in o1_gate.MEASURED:
            r[f'lo_{c}'], r[f'hi_{c}'] = -math.inf, math.inf
        new.append(r)
    allr = prow + new
    order = sorted(range(len(allr)), key=lambda i: allr[i]['t'])
    n0 = len(prow)
    out = dict(gi)
    for key in ('q', 'lo', 'hi'):
        out[key] = {c: np.r_[gi[key][c], np.full(len(new), np.nan if key == 'q' else (-np.inf if key == 'lo'
                                                                                         else np.inf))][order]
                    for c in o1_gate.MEASURED}
    out['t'] = np.r_[gi['t'], [r['t'] for r in new]][order]
    out['scored'] = np.r_[gi['scored'], np.ones(len(new), bool)][order]
    for r in prow:
        r.setdefault('khung_thieu_tam', 0)
    del n0
    return [allr[i] for i in order], out


TUOI_YAW_MAX_DE_XUAT_MS = 125.0     # CODE 19/11 (T9): the firmware's own validity limit at 20 Hz [DE XUAT, khoa o C8]
T_AGE_US = 125000                   # imu_fusion t_age_us at 20 Hz: the proven domain of the logged age (T9 doc 2.1)


def tuoi_yaw_max_of(hs):
    """The age threshold of the yaw latch [ms] (contract F14 tuoi_yaw_max_ms): the constants, or the proposal before
    C8; a locked file without it is refused here (relative mode only)."""
    v = hs.get('tuoi_yaw_max_ms')
    if v is None:
        if hs['trang_thai'] == 'khoa_c8':
            raise ValueError('hang_so tuoi_yaw_max_ms: the yaw-latch age threshold is required when khoa_c8 (F14)')
        return TUOI_YAW_MAX_DE_XUAT_MS
    return float(v)


def _cell(r, k):
    try:
        v = float(r.get(k))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def yaw_latch_flags(rows, i_edge, age_max_ms, seq_max=None):
    """J v6 5 (1), heading latch at the command edge (Codex CT1-19; CODE 19/11 after T9, reference
    fwsim/t9_heading_cases.ref_cond1 of hil d400759, T9 doc 2.1, 4, 7): the edge row has r_valid = 1, rv_n > 0,
    head_on = 1, psi_ref = 0; the upper bound of the yaw age at the latch, integer us, = age_r_ms as printed + 50 us
    (half print step) + gap to the next row + 3 us, needs the logged age inside [-gap - 0.05 ms, 125.05 ms]; the next
    row is log_seq + 1 (dong_ke_lien) or, when that one is missing, the next later row of the log with a larger
    log_seq and time (a looser bound, dong_ke_lien false, CODE 19/11); no such row, or a cell missing: not
    determinable. Codex CT2-9: the next row must be of the boot of the edge: log_seq <= seq_max, the last data line
    of the run's '#us_end' of the same boot (run_us; log_seq counts again from 0 at each boot, logger.c); seq_max
    None (boot of the run not confirmed): not determinable. A bound above age_max_ms (exact decimal comparison), not
    latched or not determinable: lost to the command path. Returns (flags, info)."""
    from decimal import Decimal
    e = rows[i_edge]
    info = dict(tuoi_yaw_max_ms=age_max_ms)
    flags = []
    latched = (_cell(e, 'r_valid') == 1 and (_cell(e, 'rv_n') or 0) > 0 and _cell(e, 'head_on') == 1
               and _cell(e, 'psi_ref') == 0.0)
    info['chot_dung'] = bool(latched)
    if not latched:
        flags.append('vong huong khong chot dung o canh (r_valid, rv_n, head_on, psi_ref; J v6 5 (1))')
    seq0, t0 = _cell(e, 'log_seq'), _cell(e, 'timestamp_us')
    nxt, lien = None, False
    info['seq_max_cung_boot'] = seq_max
    for r in rows[i_edge + 1:] if seq_max is not None else []:
        sq, tt = _cell(r, 'log_seq'), _cell(r, 'timestamp_us')
        if sq is not None and seq0 is not None and tt is not None and t0 is not None and seq0 < sq <= seq_max                 and tt > t0:
            nxt, lien = r, sq == seq0 + 1
            break
    age = _cell(e, 'age_r_ms')
    age_us = None if age is None else int(round(age * 10)) * 100
    gap_us = None if nxt is None else int(_cell(nxt, 'timestamp_us') - t0)
    info.update(dong_ke_lien=lien if nxt is not None else None, age_r_ms=age,
                gap_ms=None if gap_us is None else gap_us / 1000)
    if not latched or age_us is None or gap_us is None or not (-gap_us - 50 <= age_us <= T_AGE_US + 50):
        info['can_tuoi_ms'] = None
        flags.append('can tren tuoi yaw luc chot khong xac dinh (J v6 5 (1))')
    else:
        bound = age_us + 50 + gap_us + 3
        info['can_tuoi_ms'] = bound / 1000
        if Decimal(bound) > Decimal(str(age_max_ms)) * 1000:
            flags.append(f'can tren tuoi yaw luc chot {bound / 1000:.3f} ms > {age_max_ms:g} ms (J v6 5 (1))')
    return flags, info


def psi_reset_flags(rows, i_edge, t0_us, t_end_s):
    """J v6 5 (1) (Codex CT1-19, part): psi_reset_count or reset_n changing inside the run window [t0, t0 + t_end]
    (the heading reference reset, or the IMU chip reset, mid-run) is a run lost to the command path; neither column
    in the log: no evidence, also lost (J v6 5: missing evidence of the command path is a lost run)."""
    cols = [c for c in ('psi_reset_count', 'reset_n') if any(c in r for r in rows[i_edge:i_edge + 1])]
    if not cols:
        return ['khong co cot psi_reset_count / reset_n (J v6 5 (1))']
    out = []
    for c in cols:
        # Codex CT2-9 (T9 ref_cond1): every row of the window compared with the value AT THE EDGE; an empty or
        # unreadable cell (edge included) is missing evidence of the command path, not a skipped row
        win = [r for r in rows[i_edge:] if r['_t'] <= t0_us + t_end_s * 1e6 + 1e-3]
        v0 = _cell(rows[i_edge], c)
        miss = [r['_seq'] for r in win if _cell(r, c) is None]
        if miss:
            out.append(f'{c} trong hay khong doc duoc o {len(miss)} dong (log_seq {miss[0]}...; J v6 5 (1))')
        ch = sorted({_cell(r, c) for r in win if _cell(r, c) is not None and _cell(r, c) != v0})
        if ch:
            out.append(f'{c} doi trong luot tu {v0} sang {ch} (J v6 5 (1))')
    return out


FAIL_OP = 'thao_tac'                # J v6 5 'do thao tac'; CODE 19/11 after Codex CAM round 2 (CT1-13)


def run_us(rows, events, i_edge):
    """The ultrasonic block of the motor-on run holding the edge, tied as O1 3.1 lines 162-167 ask (Codex CT3-5):
    dict(ok, why, boot, start, end, last). One tie for the operator check and the T9 next-row bound:
    o1_cmd_path.us_for_edge with off_seq (first motor-off data line after the edge; a summary past it belongs to a
    later run), the '#us,ev=start' copies of the latest first_seq <= edge (one boot), the first '#us_end' of that
    boot with last_data_seq >= edge; first_seq = the first data line of the motor-on stretch (a lost line before it
    allowed: first_seq after the line before the stretch), last_data_seq inside [last motor-on line, off_seq),
    ticks = last_data_seq - first_seq + 1. Anything missing or contradicting: ok False with the reasons."""
    uniq = list(dict.fromkeys(events))
    seq = rows[i_edge]['_seq']
    off_seq = next((r['_seq'] for r in rows[i_edge:] if not r.get('_mo', True)), None)
    lo = i_edge
    while lo > 0 and rows[lo - 1].get('_mo', True):
        lo -= 1
    hi = i_edge
    while hi + 1 < len(rows) and rows[hi + 1].get('_mo', True):
        hi += 1
    why = []
    tied = ocp.us_for_edge(uniq, seq, off_seq)
    if tied is None:
        why.append('khong gan duoc #us_end cua luot (thieu, mau thuan, mo ho hay qua bien motor)')
    starts = [ocp._kv(e) for e in uniq if e.startswith('#us,') and ocp._kv(e).get('ev') == 'start']
    f0 = max((int(k['first_seq']) for k in starts if int(k.get('first_seq', -1)) <= seq), default=None)
    st = [k for k in starts if f0 is not None and int(k['first_seq']) == f0]
    boots = {k.get('boot') for k in st}
    boot = boots.pop() if len(boots) == 1 else None
    if boot is None:
        why.append('#us,ev=start cua luot thieu hay hai boot')
    ends = sorted((int(ocp._kv(e)['last_data_seq']), i, ocp._kv(e)) for i, e in enumerate(uniq)
                  if e.startswith('#us_end') and 'last_data_seq' in ocp._kv(e))
    end = next((k for L, _, k in ends if L >= seq and boot is not None and k.get('boot') == boot), None)
    last = None if end is None else int(end['last_data_seq'])
    # Codex CT3-5 (round 4): copies of one event differing in content are a contradiction (O1 line 163: the logger
    # sends identical copies); every field the rule reads must be there; the run's '#gain' of the same first_seq
    # (and boot when it carries one) must exist (O1 line 163: first_seq equal to the '#gain' line of the run)
    if len(st) > 1:
        why.append(f'{len(st)} ban #us,ev=start khac noi dung cho first_seq {f0}')
    if end is not None:
        same = [k for L, _, k in ends if L == last and k.get('boot') == boot]
        if len(same) > 1:
            why.append(f'{len(same)} ban #us_end khac noi dung cho last_data_seq {last}')
        miss_e = [f for f in ('on', 'changes', 'blk_ticks', 'ticks', 'last_data_seq', 'boot') if f not in end]
        if miss_e:
            why.append(f'#us_end thieu truong {miss_e}')
    for k in st:
        miss_s = [f for f in ('on', 'boot', 'first_seq') if f not in k]
        if miss_s:
            why.append(f'#us,ev=start thieu truong {miss_s}')
            break
    if f0 is not None:
        gains = [ocp._kv(e) for e in uniq if e.startswith('#gain,')]
        gk = [k for k in gains if k.get('first_seq') == str(f0) and k.get('boot', boot) == boot]
        if not gk:
            why.append(f'khong co #gain cua luot (first_seq {f0}, boot {boot})')
    if f0 is not None:
        prev = rows[lo - 1]['_seq'] if lo > 0 else -1
        if not prev < f0 <= rows[lo]['_seq']:
            why.append(f'first_seq {f0} khong la dong dau doan motor bat (log_seq {rows[lo]["_seq"]})')
    if last is not None:
        if last < rows[hi]['_seq'] or (off_seq is not None and last >= off_seq):
            why.append(f'last_data_seq {last} ngoai doan motor bat [{rows[hi]["_seq"]}, {off_seq})')
        try:
            ticks = int(end.get('ticks'))
        except (TypeError, ValueError):
            ticks = None
        if f0 is None or ticks is None or ticks <= 0 or ticks != last - f0 + 1:
            why.append(f'ticks {end.get("ticks")} != last_data_seq - first_seq + 1')
    return dict(ok=not why and tied is not None and end is not None, why=why, boot=boot, start=st, end=end,
                last=last, blk=None if tied is None else tied[0])


def thao_tac_of(rows, events, i_edge):
    """(kinds, reasons) of the operator causes of a run (J v6 5 'do thao tac'; O1 3.1 'Thao tac moi luot tren san'
    lines 156-167; CODE 19/11: this scorer is the one source; labels of J v6 5, CODE 19/11 after Codex round 3
    CT3-6), from the log and its '#' events only:
      boot          more than one firmware boot id in the events of the file ('#gain' with boot, '#us', '#us_end',
                    '#gain_ack'; boot is a 32-bit random number per start, gain_rx.c; one file = one run, CG7);
      khoa_sieu_am  the ultrasonic summary of the run (O1 'Luot khong bi chan', run_us): not tied (missing,
                    contradicting, ambiguous, past the motor-off line, ticks not matching), blk_ticks > 0, on or
                    changes not 0 at '#us_end' or on at the start, or a '#us,ev=change' line (O1 line 167); not
                    decided without an edge;
      loi_nap_gain  a '#gain_ack' answer with ok = 0 among the events (gain_rx.c: ok = err == GAIN_OK).
    The log column reset_n is the IMU chip reset count (logger.c 147, main.c 299 imu->reset_n), not a firmware
    boot: it stays with the command path (J v6 5 (1), psi_reset_flags). Not checked here: the '#gain' content
    against the gain file and the boot queries before / after the run (gain_line_check, S05 6.3, 4.7 (iv))."""
    uniq = list(dict.fromkeys(events))
    kinds, why = [], []
    boots = sorted({ocp._kv(e).get('boot', '').lower() for e in uniq
                    if e.startswith(('#gain,', '#us,', '#us_end', '#gain_ack')) and ocp._kv(e).get('boot')})
    if len(boots) > 1:
        kinds.append('boot')
        why.append(f'boot: {len(boots)} boot id trong log ({", ".join(boots)})')
    if i_edge is not None:
        ru = run_us(rows, uniq, i_edge)
        sw = list(ru['why'])
        if ru['blk'] is not None and ru['blk'] > 0:
            sw.append(f'blk_ticks {ru["blk"]} > 0')
        if any(k.get('on') not in ('0', None) for k in ru['start']):
            sw.append('on != 0 o #us,ev=start')
        end = ru['end']
        if end is not None and (end.get('on') != '0' or end.get('changes') != '0'):
            sw.append(f'#us_end on {end.get("on")}, changes {end.get("changes")}')
        if any(e.startswith('#us,') and ocp._kv(e).get('ev') == 'change' for e in uniq):
            sw.append('co dong #us,ev=change trong luot')
        if sw:
            kinds.append('khoa_sieu_am')
            why.append('khoa sieu am: ' + '; '.join(dict.fromkeys(sw)))
    bad = [e for e in uniq if e.startswith('#gain_ack') and ocp._kv(e).get('ok') == '0']
    if bad:
        kinds.append('loi_nap_gain')
        why.append(f'loi nap gain: {len(bad)} #gain_ack ok=0 ({ocp._kv(bad[0]).get("err")})')
    return kinds, why


def cham_c8_why(rel, gio, hs, task):
    """Reasons (empty: a C8 result) why a relative scoring is not a C8 result (CODE 19/11 after Codex round 2, CT2-7;
    Codex CT3-3 rounds 3, 4): time only from a FINAL c8b_gio_v1 (it names its preliminary grid and the window file),
    the window the one built from that preliminary grid (rel['chuoi_gio_c8'] from bind_clock), constants locked for C8
    with every locked part the relative mode reads (shutter branch table, khung_tc F14, tuoi_yaw_max_ms; J v6 7)."""
    why_c8 = []
    if gio is None:
        why_c8.append('gio khong tu c8b_gio_v1: chi chan doan (CODE 19/11)')
    elif not (_hex64(gio.get('luoi_so_bo_sha256')) and _hex64(gio.get('cua_so_v1_sha256'))):
        why_c8.append('c8b_gio_v1 khong phai file gio cuoi (thieu luoi_so_bo_sha256 / cua_so_v1_sha256)')
    elif rel.get('chuoi_gio_c8') is not True:
        # Codex CT3-3 (round 4): the window must be the one the final grid names, built from its preliminary grid
        why_c8.append('cua_so_v1 khong dung tu luoi so bo ma file gio cuoi ghi (chuoi so bo -> F2 -> gio cuoi)')
    if hs['trang_thai'] != 'khoa_c8':
        why_c8.append(f'hang so {hs["trang_thai"]!r}, chua khoa C8 (J v6 7)')
    else:
        # the locked parts of the scorer checked BEFORE the flag (Codex CT3-3: also on an early exit)
        for chk in (lambda: man_trap_of(hs, task, rel=True), lambda: khung_tc_of(hs),
                    lambda: tuoi_yaw_max_of(hs)):
            try:
                chk()
            except ValueError as exc:
                why_c8.append(f'khoa C8 thieu: {exc}')
    return why_c8


def operator_loss(esp32_path, events, task, other):
    """score.json of a relative run lost to an operator cause found in the log alone (Codex CT3-4), or None when the
    log shows none: used when the relative inputs cannot be read (no edge, so no cua_so_v1, ...)."""
    rows, ev_in, _, _ = read_esp32(esp32_path)
    i_edge, _, probs = ocp.find_edge(rows)
    ev = ev_in + ocp.read_events(events or (esp32_path + '.events.txt'))
    thao, why = thao_tac_of(rows, ev, i_edge)
    if not thao:
        return None
    return lost_output(task, 'thao tac: ' + '; '.join(why), FAIL_OP, thao_tac=thao,
                       ly_do_khac=[other] + (['; '.join(probs)] if i_edge is None else []), cham_c8=False,
                       cham_c8_ly_do='luot hong do thao tac truoc khi dung duoc dau vao tuong doi')


def lost_output(task, reason, cause, **extra):
    return dict(task=task, version=VERSION, car_pose_version=cm.CAR_POSE_VERSION, outcome='hong_he_do',
                fail_cause=cause, outcome_reasons=[reason], gate_v3=None,
                rule=f'O1 v11.28 3.3-3.5 (camera); {VERSION}', **extra)


def anchor_window(anchor_s):
    """--offset-anchor-s as (a0, a1): one command time T (the offset established at T, CS2-1) or a window A,B (C8b v4
    3: the match window, the offset right at an unknown point of it); refused when missing, not finite or B < A."""
    if isinstance(anchor_s, str):
        try:
            v = tuple(float(x) for x in anchor_s.split(','))
        except ValueError:
            v = ()
    elif isinstance(anchor_s, (tuple, list)):
        v = tuple(float(x) for x in anchor_s)
    elif anchor_s is None:
        v = ()
    else:
        v = (float(anchor_s),)
    if len(v) == 1:
        v = (v[0], v[0])
    if len(v) != 2 or not all(math.isfinite(x) for x in v) or v[1] < v[0]:
        raise ValueError(f'--offset-anchor-s {anchor_s!r}: T or A,B (command time [s] where --offset-s was established, '
                         'finite, B >= A) (CS2-1, C8b v4 3)')
    return v


def man_trap_of(hs, task, rel=False):
    """The shutter constants of a level (J v6 6): with the branch table 'nhanh' a level takes 'alpha' or one frame;
    no table: man_trap as it is (earlier format) in the absolute mode, one frame in the relative mode (J v6 6 needs
    the table locked before C8; Codex CT1-11)."""
    mt = hs.get('man_trap')
    if rel and hs.get('trang_thai') == 'khoa_c8' and not (isinstance(mt, dict) and 'nhanh' in mt):
        # Codex CT1-11 (round 2): J v6 6, 7: the branch table is locked before C8; a locked file without it is refused
        raise ValueError('hang_so man_trap.nhanh: the shutter branch table per level is required when khoa_c8 in the '
                         'relative mode (J v6 6, 7)')
    if mt is None:
        return None
    if 'nhanh' not in mt:
        return None if rel else mt
    return mt if _nhanh_of(mt['nhanh'], task) == 'alpha' else None


def shutter_bound(frames, mt, t_cmd, spans):
    """(rolling-shutter term of U_t [s], how) (C8b v4 3, C8B4R2-6, -9): without hang_so man_trap one frame (the
    median frame interval of frames.csv); with man_trap {alpha_r_s_px, alpha_c_s_px} (measured by phep 3b, margin
    already in) |alpha_r| x (row span) + |alpha_c| x (column span) of the car tag over the frames with the car seen
    whose command time t_cmd lies in the UNION of the closed intervals spans (the match window and the scored domain
    [0, t_deadline]; Codex VG1-2: the union, not the hull, so a frame between two disjoint parts does not count; the
    row, column the offset belongs to is not known, C8B4-5). A column term without col_px in frames.csv: refused."""
    if mt is None:
        t = np.array([r['t_s'] for r in frames], float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if not dt.size:
            raise ValueError('frames.csv: no frame interval for the rolling-shutter term (one frame)')
        f = float(np.median(dt))
        return f, f'one frame ({f * 1e3:.2f} ms, no man_trap in hang_so)'
    spans = [(float(a), float(b)) for a, b in spans]
    sel = [r for r, tc in zip(frames, t_cmd) if r.get('car_seen') and any(a <= tc <= b for a, b in spans)]
    out = 0.0
    parts = []
    for key, col in (('alpha_r_s_px', 'row_px'), ('alpha_c_s_px', 'col_px')):
        a = abs(mt[key])
        if a == 0.0:
            continue
        v = np.array([r.get(col, float('nan')) for r in sel], float)
        if not v.size or not np.all(np.isfinite(v)):
            where = ' U '.join(f'[{a0:.3f}, {b0:.3f}]' for a0, b0 in spans)
            raise ValueError(f'hang_so man_trap {key} != 0 needs {col} on every frame with the car seen in '
                             f'{where} s (track again with c7_measure at this commit)')
        span = float(v.max() - v.min())
        out += a * span
        parts.append(f'{a:g} s/px x {span:.1f} px ({col})')
    return out, 'man_trap: ' + (' + '.join(parts) if parts else '0')


GIO_FORMAT = 'c8b_gio_v1'
GIO_STEP_MAX_S = 1e-3          # C8b v4 3: adjacent nodes at most 1 ms apart (Codex VG1-3)
GIO_STEP_TOL_S = 1e-9          # float tolerance on that limit
GIO_SHA_KEYS = ('esp32', 'frames')     # c8b_gio writes sha256 {esp32, frames} of the files it matched (Codex VG1-1)


def _hex64(v):
    return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)


def load_gio(path, nut, esp32_sha256, frames_sha256):
    """The time-grid file of the C8b script (c8b_gio_v1, CODE_C8B; interface 1 of CODE 17/11) at node nut: D =
    nodes_s[nut] (t_log = t_cam + D), the match window cua_so_ghep_log_s [a, b] (ESP32 clock, s), the grid step (the
    largest gap between adjacent nodes, 0 for one node). Refused unless moc_mau and ok are true, the nodes finite and
    increasing with adjacent nodes at most 1 ms apart (C8b v4 3; Codex VG1-3), the window finite with a <= b, nut a
    node index, and (Codex VG1-1) the file's sha256 {esp32, frames} (64 lowercase hex each) equal to the sha256 of the
    ESP32 log and the frames.csv being scored: a grid file of another run, or one without that evidence, never gives
    D, the match window or the tick branch 'bo' to this run."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f'{c} in {path}')))
    if not isinstance(d, dict) or d.get('format', GIO_FORMAT) != GIO_FORMAT:
        raise ValueError(f'{path}: not a {GIO_FORMAT} file')
    if d.get('moc_mau') is not True or d.get('ok') is not True:
        raise ValueError(f'{path}: moc_mau and ok must be true (C8b v4 3, 7: the match on sample marks, every check '
                         f'passed): moc_mau {d.get("moc_mau")!r}, ok {d.get("ok")!r}')
    sha = d.get('sha256')
    if not isinstance(sha, dict) or not all(_hex64(sha.get(k)) for k in GIO_SHA_KEYS):
        raise ValueError(f'{path}: sha256 {{esp32, frames}} (64 lowercase hex each) of the files the grid was made from '
                         f'is required (Codex VG1-1): {sha!r}')
    for k, want in zip(GIO_SHA_KEYS, (esp32_sha256, frames_sha256)):
        if sha[k] != want:
            raise ValueError(f'{path}: made from another {k} file (sha256 {sha[k]} != {want} of the file scored; '
                             'Codex VG1-1)')
    nodes = d.get('nodes_s')
    if not isinstance(nodes, list) or not nodes or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in nodes) \
            or any(b <= a for a, b in zip(nodes[:-1], nodes[1:])):
        raise ValueError(f'{path}: nodes_s must be finite numbers, increasing, not empty')
    wide = [(i, b - a) for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])) if b - a > GIO_STEP_MAX_S + GIO_STEP_TOL_S]
    if wide:
        i, g = wide[0]
        raise ValueError(f'{path}: nodes_s {i}, {i + 1} are {g * 1e3:.6f} ms apart; adjacent nodes at most 1 ms '
                         '(C8b v4 3; Codex VG1-3)')
    w = d.get('cua_so_ghep_log_s')
    if not isinstance(w, list) or len(w) != 2 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in w) or w[1] < w[0]:
        raise ValueError(f'{path}: cua_so_ghep_log_s [a, b] finite, a <= b')
    if isinstance(nut, bool) or not isinstance(nut, int) or not 0 <= nut < len(nodes):
        raise ValueError(f'--nut {nut!r}: a node index 0..{len(nodes) - 1}')
    step = max((b - a for a, b in zip(nodes[:-1], nodes[1:])), default=0.0)
    return dict(file=os.path.basename(path), sha256=cm.sha256(path), nut=nut, D=float(nodes[nut]),
                win_log_s=[float(w[0]), float(w[1])], step_s=float(step), n_nodes=len(nodes),
                vao={k: sha[k] for k in GIO_SHA_KEYS},
                # Codex CT3-3: a final grid names its preliminary grid and the window file built from it (c8b_gio
                # --so-bo / --cua-so, hil faa8160); a preliminary grid has neither
                luoi_so_bo_sha256=d.get('luoi_so_bo_sha256'), cua_so_v1_sha256=d.get('cua_so_v1_sha256'))


def drift_bound(sync, offset_s, t0_s, deadline, troi_ppm, anchor_s=None):
    """(drift term of U_t [s], how) over the whole run window [0, t_deadline] (O1 3.5 SC-12; CS1-4, CS1-6). With C7b:
    rate = max(|offset_end - offset_start| / distance of the two third centres, troi_ppm) (troi_ppm alone when C7b is
    short, K1), times the largest distance from the centre of the C7b window to [0, t_deadline], plus the gap between
    the offset used (the fit of the whole window) and the offset interpolated at that centre (mean of the thirds).
    With --offset-s (no C7b; CS2-1): the offset holds (within --offset-err-s, added to U_t by the caller) at a point
    of anchor_s (--offset-anchor-s: one command time, or the match window A,B of C8b v4 3); drift = troi_ppm x the
    largest distance from any point of it to any point of [0, t_deadline]. No anchor: refused."""
    lim = troi_ppm * 1e-6
    if sync is None:
        a0, a1 = anchor_window(anchor_s)
        # C8b v4 3 (C8B4R2-4): the offset is right at an unknown point of the match window [a0, a1]: the largest
        # distance from any point of it to any point of [0, t_deadline]
        far = max(abs(x - y) for x in (a0, a1) for y in (0.0, deadline))
        return lim * far, (f'troi_ppm x {far:.3f} s from the offset window [{a0:.3f}, {a1:.3f}] s (--offset-s: no C7b '
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
          report_only=frozenset(), gio=None, rel=None):
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
    if task in FINAL_SET and k_start is None and rel is None:
        raise ValueError(f'{task}: give --k-start X,Y,HEAD_DEG (nominal start pose, locked at C8)')
    if report_only and task not in FIG8:
        raise ValueError(f'report_only {sorted(report_only)} on {task}: K1, K2 decide on all five conditions '
                         '(CODE_GATE 4; Codex VG1-4)')
    if gio is not None and offset_s is not None:
        raise ValueError('--gio-c8b and --offset-s: one clock source only')
    if gio is not None:
        # Codex VG1-1: the grid file must name the log and the frames scored (load_gio checks it at the CLI; checked
        # again here for a grid dict handed to score() directly)
        vao = gio.get('vao') or {}
        if frames_sha256 is None or vao.get('frames') != frames_sha256 or vao.get('esp32') != cm.sha256(esp32_path):
            raise ValueError('--gio-c8b: the grid file was not made from the ESP32 log and frames.csv scored '
                             '(sha256 esp32 / frames differ or not given; Codex VG1-1)')
    if offset_s is not None:
        if not offset_reason.strip():
            raise ValueError('--offset-s needs --offset-reason (C7b skipped)')
        if offset_anchor_s is None or offset_err_s is None or not (math.isfinite(offset_err_s) and offset_err_s >= 0.0):
            raise ValueError('--offset-s needs --offset-anchor-s (command time [s] where the offset was established) '
                             'and --offset-err-s (its error [s], >= 0) (CS2-1)')
        offset_anchor_s = anchor_window(offset_anchor_s)
    if old_shift_frames(frames) or (track_info or {}).get('shift_kind') == 'median_magnitude':
        raise ValueError('frames.csv holds the mark shift of the version before 17/11 (median of the lengths): the '
                         'camera-bumped decision needs the length of the median vector, which cannot be rebuilt from '
                         'it: run c7_measure track on the video again (CS2-4)')
    if rel is not None and k_start is not None:
        raise ValueError('--tuong-doi: no --k-start (J v6 2: the start pose is the measured reference)')
    # relative frame (J v6 2; CAM v5 0.1): no nominal start pose; the lane frame of K is the body frame at start; the
    # closing corner of K2 takes its incoming side (J v6 0 (f), 3)
    lane = (Fig8Task(task) if task in FIG8 else
            (KTask(task, k_start) if rel is None else KTask(task, (0.0, 0.0, 0.0), end_incoming=True)))
    ro_sha = None
    if rel is not None and report_only - ({'g5'} if task == 'O1-F08' else set()):
        # Codex CT3-2: in the relative mode the only exemption is G5 of F08 from appendix J (J v6 6); a further mask
        # from the caller has no locked 2b source here and would carry the J sha: refused
        raise ValueError(f'report_only {sorted(report_only)} in the relative mode: only g5 of O1-F08 (J v6 6)')
    if rel is not None and task == 'O1-F08':
        # J v6 6: G5 of F08 report only (Liam 19/11). Codex CT2-5: the source (J v6) is set by the mode and the task,
        # whatever the caller passed, so the exception, the gate and its check use ONE mask
        report_only = report_only | {'g5'}
        ro_sha = J_V6_SHA256
    gate_check = check_track_gates(hs, track_info, frames_sha256)
    if hs['trang_thai'] == 'khoa_c8' and task not in (hs['imu'].get('mien_dong') or {}):
        raise ValueError(f'hang_so imu mien_dong: no locked U_IMU range for {task}')
    rows, ev_in, n_dup, has_age = read_esp32(esp32_path)
    # Codex CC1-1 (17/11): a run lost early carries the scoring set-up known so far (constants, start pose, t = 0 and
    # clock offset when known, else null), so c7_calib_drift gop can apply O1 4 (Q3) to it
    base = dict(hang_so=dict(file=hs['file'], sha256=hs['sha256'], trang_thai=hs['trang_thai'], nguon=hs['nguon']),
                n_duplicates=n_dup, thresholds={k: hs[k] for k in THRESH_KEYS}, k_start=list(lane.start),
                t0_log_us=None,
                # Codex CC2-1: an offset given (--offset-s) or a grid node (gio D) is known from the start; null only
                # while the C7b match has not run or when it failed
                offset_s=gio['D'] if gio is not None else offset_s,
                offset_src='c8b_gio' if gio is not None else ('--offset-s' if offset_s is not None else None))
    if rel is not None:
        # CODE 19/11 after Codex round 2 (CT2-7): the relative scoring of C8 takes its time ONLY from the C8b grid
        # (c8b_gio_v1); C7b and --offset-s are diagnostics. Codex CT3-3: only a FINAL grid (it names its preliminary
        # grid and the window file built from it) and a scorer locked before C8 (J v6 7: khoa_c8, F14) give a C8 result
        why_c8 = cham_c8_why(rel, gio, hs, task)
        base.update(cham_c8=not why_c8, cham_c8_ly_do='; '.join(why_c8) or None)
    i_edge, later, probs = ocp.find_edge(rows)
    # Codex CT3-7: the absolute mode reads the events after its early exits, as at 921e839
    ev = ev_in + ocp.read_events(events or (esp32_path + '.events.txt')) if rel is not None else None
    # J v6 5, CODE 19/11 after Codex round 2 (CT1-13): the operator causes first, from this scorer only (the log and
    # its events): thao_tac [kinds] in score.json, fail_cause 'thao_tac' (kinds in outcome_reasons); relative mode
    thao, thao_why = thao_tac_of(rows, ev, i_edge) if rel is not None else ([], [])
    if thao:
        base['thao_tac'] = thao

    def lost(reason, cause, **kw):
        # every loss of a run with an operator cause is that loss (J v6 5 order); the other reasons are kept
        if thao:
            kw['ly_do_khac'] = [reason] + list(kw.get('ly_do_khac') or [])
            return lost_output(task, 'thao tac: ' + '; '.join(thao_why), FAIL_OP, **kw, **base)
        return lost_output(task, reason, cause, **kw, **base)
    if i_edge is None:
        if any(p.startswith('edge_unknown') for p in probs) or thao:
            # Codex CT3-4: with operator evidence the run is that loss, whatever stops the edge
            return lost('; '.join(probs), FAIL_MEAS), []
        raise ValueError('; '.join(probs))
    t0_us = float(rows[i_edge]['_t'])
    base['t0_log_us'] = t0_us
    if rel is None and not any(r['valid'] for r in frames):
        # Codex CT2-8: the absolute mode keeps its order of 921e839 (no sender read for a clip without a frame)
        return lost_output(task, 'no valid camera frame in the clip', FAIL_MEAS, **base), []
    # command path (O1 3.5): o1_cmd_path on the same log, read before any loss is decided so every reason of the run
    # is kept (J v6 5: the operator causes first, then the others; Codex CT1-13)
    if ev is None:
        ev = ev_in + ocp.read_events(events or (esp32_path + '.events.txt'))
    try:
        cmd = ocp.analyse(rows, ocp.Task(task, allow_final_set), ev, has_age, ocp.read_sender(sender, task),
                          ocp.read_sender_bao(sender, task), ocp.read_sender_complete(sender, task))
    except ValueError as exc:
        if not thao:
            raise
        # Codex CT3-4: an unreadable sender log does not hide the operator loss; its error is kept beside it
        return lost(f'duong lenh khong doc duoc: {exc}', FAIL_MEAS), []
    cmd_th = cmd_thresholds(cmd, hs)
    cmd_flags = list(cmd['flags']) + cmd_th
    if rel is not None and rel['tc']['trang_thai'] == 'hong_cua_so_tc':
        # J v6 5: run lost to the reference window (a kind of measurement-system loss), before the other rules of
        # the camera; the command-path reasons are kept beside it (ly_do_khac) for the run history. With an
        # operator cause the run is that loss and does not count as a window loss (J v6 5 J2-11)
        return lost('cua so tham chieu: ' + '; '.join(rel['tc']['ly_do']), FAIL_MEAS,
                    hong_cua_so_tc=not thao, ly_do_khac=['duong lenh: ' + f for f in cmd_flags],
                    khung_tuong_doi_v1=_clean(rel_block(rel, None))), []
    if not any(r['valid'] for r in frames):
        return lost('no valid camera frame in the clip', FAIL_MEAS), []
    t_cam = np.array([r['t_s'] for r in frames])
    sync, sync_gyro = None, None
    if gio is not None:                                # C8b v4 3: a node of the C8b time grid, no o1_time_sync
        offset_s = gio['D']
    elif offset_s is None:
        sync, sync_gyro = clock_match(frames, rows, task)
        if not sync.ok:
            return lost(f'clock not matched (C7b): {sync.why}', FAIL_MEAS), []
        offset_s = sync.offset_s
    base.update(offset_s=offset_s,
                offset_src='c8b_gio' if gio is not None else ('--offset-s' if sync is None else 'o1_time_sync'))
    deadline = lane.Tcmd + T_LATE
    # drift up to t_deadline (SC-12); the --offset-s error at its anchor (CS2-1) adds to the C7b residual of the
    # constants [CACH HIEU: kept both, the larger bound]
    if gio is None:
        drift, drift_src = drift_bound(sync, offset_s, t0_us / 1e6, deadline, hs['troi_ppm'], offset_anchor_s)
        off_err = float(offset_err_s) if sync is None else 0.0
        ghep, luoi, shut = hs['u_t_ghep_s'], 0.0, 0.0
        u_t_parts = dict(ghep_s=ghep, offset_err_s=off_err, drift_s=drift, drift_src=drift_src)
        if rel is not None:
            # J v6 6 (Codex CT1-11): the shutter term on every time source of the relative mode (--offset-s, C7b),
            # by the branch of the level; the frames of the scored domain (and the offset anchor when given)
            spans = [(0.0, deadline)] + ([tuple(offset_anchor_s)] if offset_anchor_s is not None else [])
            mt_l = man_trap_of(hs, task, rel=True)
            shut, shut_src = shutter_bound(frames, mt_l, t_cam + offset_s - t0_us / 1e6, spans)
            u_t_parts.update(man_trap_s=shut, man_trap_src=shut_src,
                             man_trap_nhanh='alpha' if mt_l is not None else 'mot_khung')
    else:
        # C8b v4 3, table of U_t at a node (CODE 17/11, interface 1): grid step (whole step), drift from the match
        # window (log clock) to the scored domain incl. t_deadline, rolling shutter; no C7b residual, no offset error
        win = (gio['win_log_s'][0] - t0_us / 1e6, gio['win_log_s'][1] - t0_us / 1e6)     # command time
        lim = hs['troi_ppm'] * 1e-6
        far = max(abs(x - y) for x in win for y in (0.0, deadline))
        drift, drift_src = lim * far, f'troi_ppm x {far:.3f} s from the match window [{win[0]:.3f}, {win[1]:.3f}] s'
        ghep, off_err, luoi = 0.0, 0.0, gio['step_s']
        mt_l = man_trap_of(hs, task, rel=rel is not None)
        shut, shut_src = shutter_bound(frames, mt_l, t_cam + offset_s - t0_us / 1e6, [win, (0.0, deadline)])
        u_t_parts = dict(ghep_s=0.0, offset_err_s=0.0, luoi_s=luoi, drift_s=drift, drift_src=drift_src,
                         man_trap_s=shut, man_trap_src=shut_src,
                         man_trap_nhanh='alpha' if mt_l is not None else 'mot_khung')

    # O1 v11.33 SC-12 (OG2-1, OG3-4): the tick part leaves U_t only on the branch 'bo', which needs the clock match
    # on sample marks at their true times (C8b); without it that branch cannot score the run
    drop_tick = hs['nhanh_tick'] == 'bo'
    if drop_tick and gio is None:
        raise ValueError('hang_so nhanh_tick bo: the tick part leaves U_t only with the clock match on sample marks at '
                         'their true times (C8b, O1 v11.33 SC-12): score at the nodes of the C8b grid (--gio-c8b); '
                         'o1_time_sync and --offset-s do not carry it')
    u_t_parts['nhanh_tick'] = hs['nhanh_tick']

    def u_t_of(end_u):
        # tick evidence over the domain scored, [t0, t0 + end of the union domain] (CS2-2: a log ending after the
        # run closed but before t_deadline does not spoil it); the drift still runs to t_deadline (SC-12). The
        # check that the log covers both ends (finite tick) holds on both branches (OG2-8).
        tick = tick_max_s(rows, t0_us, t0_us + end_u * 1e6)
        u_t_parts.update(tick_s=tick, tick_to_s=end_u)
        if not math.isfinite(tick):
            return math.inf
        return ghep + off_err + luoi + drift + shut + (0.0 if drop_tick else tick)
    vi = [k for k, r in enumerate(frames) if r['valid']]
    t_log = t_cam[vi] + offset_s
    tv = t_log - t0_us / 1e6
    sp = np.array([frames[k]['speed'] for k in vi])
    wzv = np.array([frames[k]['wz'] for k in vi])
    R = None
    if rel is None:
        pose = [lane.to_lane(frames[k]['x'], frames[k]['y'], frames[k]['psi']) for k in vi]
        x = np.array([p[0] for p in pose])
        y = np.array([p[1] for p in pose])
        psi = np.array([p[2] for p in pose])
        psi_cam, upk_add, ymode = np.array([frames[k]['psi'] for k in vi]), None, 'thuong'
    else:
        R = rel_pose(frames, vi, rel, lane)
        x, y, psi, psi_cam, upk_add = R['x'], R['y'], R['psi'], R['psi_cam'], R['upk_add']
        ymode = 'khong_xac_dinh' if R['tam_doi'] == 'khong_xac_dinh' else 'thuong'
    # relative frame (J v6 2, Codex CT1-4): the camera and IMU yaws aligned on the frames and the merge of the
    # reference heading (tham_chieu_v1 ids) when that reference is on a set of J v6 2
    ar = None
    if R is not None and rel['tc']['psi_ok'] and rel['tc']['tap'] in ('giao', 'ma_mat_neo'):
        # Codex CT2-3: exactly the ids of tham_chieu_v1 (F12), times and psi' from the chain
        chm = {c['frame_id']: c for c in rel['chain']}
        rr = [chm[int(i)] for i in rel['tc']['ids']]
        ar = (np.array([float(c['t_cam']) for c in rr]) + offset_s, np.array([float(c['psi_ma_p']) for c in rr]))
    upsi_k, yaw = yaw_bounds(t_log, psi_cam, wzv, sp, t0_us / 1e6, hs, task, esp32_path, t0_us / 1e6 + deadline,
                             mode=ymode, align_ref=ar)
    psi_imu_al = yaw.pop('_psi_imu_al', None)
    so_duoc, nghi_yaw = yaw.pop('_so_duoc'), yaw.pop('_nghi_ngo')
    # CODE 19/11 (CT2-4): no frame with a centre in the run window (A6 or q lost over the whole domain, video there):
    # G1-G4 not concluded (the path of the t_som case, G5 where the heading is known), never 'no frame' lost
    no_centre = R is not None and not np.any(R['pos_def'] & (tv >= 0.0) & (tv < lane.Tcmd + T_LATE - 1e-12))
    tsom_case = R is not None and (not R['can_cu'] or no_centre)
    if R is not None:
        upsi_k = np.where(R['yaw_def'], upsi_k, np.inf)
        yaw_ok = R['yaw_def'] & so_duoc & ~nghi_yaw
    use = np.arange(len(vi))
    if tsom_case:
        t_dyn = yaw.get('t_mien_dong_vuot')
        res = evaluate_t_som(tv, x, y, psi, R, lane, hs, upsi_k, u_t_of, psi_imu_al, task, yaw_ok, t_dyn)
    elif R is not None:
        # Codex CT1-7: only frames with a defined centre enter the tracker, the speeds and G1; the others come back as
        # undefined frames (gaps for G1, n_undef for G2-G5)
        use = np.nonzero(R['pos_def'])[0]
        res = evaluate(tv[use], x[use], y[use], psi[use], sp[use], wzv[use], lane, hs, upsi_k[use], u_t_of,
                       upk_add=upk_add[use], rel_mode=True, gap_t=tv)
    else:
        res = evaluate(tv, x, y, psi, sp, wzv, lane, hs, upsi_k, u_t_of, upk_add=upk_add)
    if res is None:
        return lost('no valid camera frame in the run window', FAIL_MEAS), []
    out, prow, gi = res
    if prow is not None and len(use) != len(vi):
        for r in prow:
            r['iv'] = int(use[r['iv']])                 # back to the index among the valid frames
        prow, gi = add_undefined(prow, gi, tv, np.setdiff1d(np.arange(len(vi)), use), lane.Tcmd + T_LATE, R)
    if gi is None:                     # CS1-6: no finite U_t is no evidence; never a gate with an unknown U_t
        return lost(f'U_t not bounded (the log does not cover [t0, t0 + {out["t_end_union"]:.3f} s], '
                    f'the end of the union domain: tick {u_t_parts.get("tick_s")})', FAIL_MEAS,
                    U_t_parts=_clean(u_t_parts) if rel is not None else u_t_parts), []
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
    chot_yaw = None
    if R is not None:
        cmd_flags += psi_reset_flags(rows, i_edge, t0_us, t_end_u)
        ru = run_us(rows, ev, i_edge)              # Codex CT3-5: the same tie as the operator check
        fl_y, chot_yaw = yaw_latch_flags(rows, i_edge, tuoi_yaw_max_of(hs), ru['last'] if ru['ok'] else None)
        cmd_flags += fl_y
    cmd_lost = bool(cmd_flags)
    # outcome (O1 3.5)
    t1 = out['t1']
    sc = gi['scored']
    before = None
    viol_before = False
    if t1 is not None:
        gap_case = out['t1_kind'] == 'khoang_trong'
        before = sc & ((gi['t'] <= t1 + 1e-12) if gap_case else (gi['t'] < t1 - 1e-12))
        if tsom_case:                    # J v6 5 (J3-3): the evidence must also lie in the common domain [0, t_som]
            before = before & (gi['t'] <= out['t_som_s'] + 1e-12)
        viol_before = bool(certain_violations(gi['lo'], before, report_only))
    reasons, fail_cause = (['thao tac: ' + '; '.join(thao_why)] if thao else []), ''
    if bumped:
        reasons.append(f'may quay bi va (dich moc {shift_max:.1f} px > {hs["bump_px"]:g} px)')
    if cmd_lost:
        reasons.append('duong lenh: ' + '; '.join(cmd_flags))
    per = {c: (gi['q'][c], gi['lo'][c], gi['hi'][c]) for c in o1_gate.MEASURED}
    # O1 3.5 line 377: no exception for a run lost to the camera bump, the command path or an operator cause
    cat = bool(t1 is not None and viol_before and not bumped and not cmd_lost and not thao)
    if out['U_t_scope'] == 'truoc_t1' and not cat:
        # CS3-1: U_t holds before t1 only; without a certain violation before t1 the run is lost as before
        return lost('; '.join(reasons + [
            f'U_t not bounded over the union domain (the log does not cover [t0, t0 + {t_end_u:.3f} s]); no certain '
            f'violation before t1 = {t1:.3f} s']), FAIL_MEAS, U_t_parts=_clean(u_t_parts) if rel is not None else u_t_parts), []
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
        if tsom_case:                    # J v6 5: G1-G4 not concluded without the position basis
            g1 = dict(g1, state='chua_ket_luan', q='inconclusive')
        t_pu = out.get('t_tien_do_chua_xd')
        if t_pu is not None and any(td is None or td > t_pu + 1e-12
                                    for td in (out['t_done_way1'], out['t_done_way2'])):
            # CODE 19/11 (CT2-4): the progress is not determined from t_pu: a completion not shown before it is not
            # concluded either way; gate_v3.g1.tien_do_chua_xd_tu_s names it (rq3_log cannot recompute this state
            # from t_done yet: not checked by rq3_log, gate_rq3 says so, as in the t_som case)
            g1 = dict(g1, state='chua_ket_luan', q='inconclusive', tien_do_chua_xd_tu_s=float(t_pu))
        if out['t_gap_long'] is not None:
            reasons.append(f'khoang trong > t_trong {hs["t_trong_s"]:g} s tu t = {out["t_gap_long"]:.3f} s')
        if out['t_first_track_err'] is not None:
            reasons.append(f'loi theo doi tai t = {out["t_first_track_err"]:.3f} s')
    note = (('relative frame J v6 (CAM v5); ' if R is not None else '') + f'camera O1 v11.28; constants {hs["trang_thai"]} ({hs["file"]}); U_t = {u_t * 1e3:.1f} ms; '
            f'yaw check {"on" if yaw["compared"] else "off: " + yaw["reason"]}')
    gate = build_gate(per, dom_c, dom_u, lane, u_t, hs, g1, note, report_only if ro_sha else (), ro_sha)
    # C8b v4 3 (CODE 17/11, interface 1): per frame its camera time (the key that joins a frame across the nodes of
    # the C8b grid) and whether it is in the common / union domain at this node (cat_t1: the part before t1)
    tcv = t_cam[vi]
    for r, mc, mu in zip(prow, dom_c, dom_u):
        iv = r.pop('iv')
        r['t_cam_s'] = float(tcv[iv])
        if R is not None:
            # F13 rows (C8B 19/11): frame identity, q, the suspect flags and the widening of a position suspect
            sus_p = bool(r.get('suspect', 0))
            r.update(rel_row(R, iv), frame_id=int(frames[vi[iv]]['idx']), q_ok=int(bool(R['q_ok'][iv])),
                     nghi_ngo_vi_tri=int(sus_p), U_p_noi_m=_fin(r['pred_m']) if sus_p else 0.0,
                     so_duoc_yaw=int(bool(so_duoc[iv])), nghi_ngo_yaw=int(bool(nghi_yaw[iv])))
        r['mien_chung'], r['mien_hop'] = int(bool(mc)), int(bool(mu))
    if cat:
        outcome, fail_cause = 'khong_dat', FAIL_VIOL
    elif reasons:
        outcome = 'hong_he_do'
        fail_cause = (FAIL_OP if thao else
                      FAIL_CMD if cmd_lost and not bumped and t1 is None else FAIL_MEAS)
    else:
        outcome = gate['outcome_decide']
    if 'tien_do_chua_xd_tu_s' in gate['g1'] and not tsom_case:
        gate_rq3 = 'chua_kiem: g1 tien_do_chua_xd_tu_s (CT2-4, CODE 19/11) chua co trong rq3_log'
    elif tsom_case:
        # J v6 5 t_som case: G1 not concluded without the position basis. Fields fixed by CODE_GATE 19/11 in
        # gate_v3.g1: ca_t_som true, t_som_s (end of the common domain of G5), t_done [null, null]; rq3_log reads them
        # from feat/rq3-tuong-doi (not merged here yet): the gate is not checked by rq3_log here, gate_rq3 says so
        gate['g1']['ca_t_som'] = True
        gate['g1']['t_som_s'] = float(out['t_som_s'])
        gate_rq3 = 'chua_kiem: rq3_log cua feat/rq3-tuong-doi (ca_t_som) chua gop vao nhanh nay'
    else:
        gate_rq3 = 'dat'
        try:
            req = gate.get('required')
            rq3_log.validate_gate(gate, 'HW', req)
            rq3_log.check_outcome_vs_gate(outcome, gate, 'HW', fail_cause, req)
        except rq3_log.LogError as exc:
            raise RuntimeError(f'gate block refused by rq3_log (bug of this scorer): {exc}')
    # report keys of the earlier version (s05_final_table, people): on the union domain (cat: before t1)
    pr = [r for r, m in zip(prow, dom_u) if m and math.isfinite(r.get('e_perp', math.nan))]
    if pr and not tsom_case:
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
    if R is None:
        out.update(placement(tv, x, y, psi, lane))
    else:
        # CODE 19/11 (GATE: compared with mien_da_tinh of u_moc_manifest_v1): the frames this node relies on, every
        # frame of the union domain (cat: before t1), by frame_id and the hash of that list (c7_tuong_doi.ids_sha)
        ids_hop = [int(r['frame_id']) for r in prow if r['mien_hop']]
        mpp = dict(nut=None if gio is None else gio['nut'], D_s=float(offset_s), id_khung=ids_hop,
                   id_khung_sha256=ctd.ids_sha(ids_hop))
        out.update(khung_tuong_doi_v1=_clean(dict(rel_block(rel, R), t_som_s=out.pop('t_som_s', None),
                                               gio_sha256=None if gio is None else gio['sha256'],
                                               esp32_sha256=cm.sha256(esp32_path), mien_phai_phu=mpp,
                                               chot_yaw=chot_yaw)),
                   placement_n=None)
    out.update(base, version=VERSION, car_pose_version=cm.CAR_POSE_VERSION, outcome=outcome, fail_cause=fail_cause,
               outcome_reasons=reasons,
               completed=g1['q'], gate_v3=gate, gate_rq3=gate_rq3, cat_t1=cat, report_only=sorted(report_only), U_t_s=u_t, U_t_parts=u_t_parts, yaw_check=yaw,
               shift_max_px=shift_max, bumped=bumped, shift_drift_px=drift,
               shift_drift_note=('quay lai clip H truoc luot sau (design 3.6)'
                                 if drift is not None and drift > hs['drift_px'] else ''),
               t0_log_us=t0_us, edge_log_seq=rows[i_edge]['_seq'], later_rises=later,
               offset_s=offset_s,
               offset_src='c8b_gio' if gio is not None else ('--offset-s' if sync is None else 'o1_time_sync'),
               offset_reason=offset_reason if sync is None and gio is None else '',
               offset_anchor_s=offset_anchor_s if sync is None and gio is None else None,
               offset_err_s=offset_err_s if sync is None and gio is None else None,
               gio_c8b=None if gio is None else {k: gio[k] for k in ('file', 'sha256', 'nut', 'D', 'win_log_s', 'step_s',
                                                                    'vao')},
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
               thresholds={k: hs[k] for k in THRESH_KEYS},
               rule=(f'J v6 (relative frame, CAM v5) on O1 v11.28 3.3-3.5, 3.7 (camera); {VERSION}' if R is not None
                     else f'O1 v11.28 3.3-3.5, 3.7 (camera); {VERSION}'))
    return out, prow


def _clean(o):
    """o with every non-finite float replaced by None (JSON without Infinity / NaN; Codex CT1-16)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


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


def _clock_nodes(a, rows, t0_us, frames, hs, w_s):
    """(nodes D_i, u_ref,i, sha of the grid or None, reasons) for the reference window (CAM v5 3.2): the C8b grid
    (every node) or --offset-s; the time part u = grid step or offset error + C7b residual, drift to the window, one
    frame, tick. A log that does not cover the window (no tick bound) is a reason of the window (Codex CT1-13: a run
    lost to the reference window), not a tool error; the tick term is then left out of u."""
    fdt = ctd._frame_dt([r['t_s'] for r in frames])
    t0 = t0_us / 1e6
    span = (-w_s - 1.0, 1.0)                                   # command time around the edge
    tick = tick_max_s(rows, t0_us + span[0] * 1e6, t0_us + span[1] * 1e6)
    why = []
    if not math.isfinite(tick):
        why.append('log khong phu cua so tham chieu (khong co can tick)')
        tick = 0.0
    lim = hs['troi_ppm'] * 1e-6
    if a.gio_c8b:
        g = load_gio(a.gio_c8b, 0, cm.sha256(a.esp32), cm.sha256(a.frames))
        with open(a.gio_c8b, encoding='utf-8') as f:
            nodes = [float(v) for v in json.load(f)['nodes_s']]
        far = max(abs(w - (t0 + x)) for w in g['win_log_s'] for x in span)
        u = g['step_s'] + lim * far + fdt + tick
        return nodes, [u] * len(nodes), g['sha256'], why
    if a.offset_s is None or a.offset_err_s is None or a.offset_anchor_s is None:
        raise ValueError('cua-so needs --gio-c8b, or --offset-s with --offset-err-s and --offset-anchor-s')
    anc = anchor_window(a.offset_anchor_s)
    far = max(abs(w - x) for w in anc for x in span)
    return [float(a.offset_s)], [float(a.offset_err_s) + hs['u_t_ghep_s'] + lim * far + fdt + tick], None, why


def window_gates(hs, track, fsha):
    """(frame gates, source) of the reference window (contract F2 cong_khung, cong_khung_nguon): track.json beside
    frames.csv when it is the one of this frames.csv (check_track_gates; Codex CT1-14 round 3: a track.json with gates
    must carry frames_sha256 equal to the file's, also before C8), else the locked cong_khung of the constants, else
    the c7_measure defaults ('mac_dinh')."""
    check_track_gates(hs, track, fsha)
    tg = (track or {}).get('gates')
    if tg:
        if (track or {}).get('frames_sha256') != fsha:
            raise ValueError('track.json gates without the frames_sha256 of this frames.csv: not used for the window '
                             '(Codex CT1-14)')
        return dict(tg), 'track.json'
    if hs.get('cong_khung'):
        return dict(hs['cong_khung']), 'hang_so'
    return dict(cm.DEFAULT_GATES), 'mac_dinh'


def bind_gates(pre, hs, track, fsha):
    """Codex CT1-3 (round 3): the window was gated with the frame gates of THIS scoring (window_gates)."""
    gates = json.loads(json.dumps(window_gates(hs, track, fsha)[0]))
    if pre.get('cong_khung') != gates:
        raise ValueError(f'cua_so_v1 cong_khung {pre.get("cong_khung")!r} differs from the frame gates of the scoring '
                         f'{gates!r}')


def bind_session(pre, track):
    """Several camera segments (CODE 20/11 khuya, chain A; c7_ghep_cua_so --doan): the camera of a run is chosen by the
    session_sha256 of ITS F2 row, so the frames scored must be tracked with that session: the track.json beside
    frames.csv must name the same session_sha256. A row without session_sha256 (older F2) is not checked."""
    want = pre.get('session_sha256')
    if want is None:
        return
    have = (track or {}).get('session_sha256')
    if have is None:
        raise ValueError(f'cua_so_v1 row of {pre.get("luot_id")} names session {want[:12]}, but there is no '
                         'track.json with session_sha256 beside frames.csv')
    if have != want:
        raise ValueError(f'cua_so_v1 row of {pre.get("luot_id")} is of session {want[:12]}, frames.csv was tracked '
                         f'with session {have[:12]} (another camera segment)')


def _track_beside(frames_path):
    tj = os.path.join(os.path.dirname(os.path.abspath(frames_path)), 'track.json')
    if not os.path.isfile(tj):
        return None
    with open(tj, encoding='utf-8') as f:
        return json.load(f)


def _ref_from(pre, chain, lan_lap, hs, frames, extra):
    return ctd.tham_chieu(pre, chain, lan_lap, khung_tc_of(hs), hs['t_trong_s'],
                          ctd._frame_dt([r['t_s'] for r in frames]), extra)


def chain_and_ref(frames, a6, xoay, ll, track, task, truc, pre, hs, extra):
    """(chain, tham_chieu_v1) of a run. Another mounting (lan_lap_v1 cung_lan_lap not true): the C8B q composition
    needs u_psi_w_rad, the noise of the mean tag heading over the run window (F16): the largest deviation of psi'
    (after A6, no q needed) from its mean over the reference set, psi_lech_max_rad of a first pass (a bound by the
    data, not a guess); then the chain again with it."""
    cho = ctd.cho_of(task)
    chain = ck.chuoi_khung(frames, a6, xoay, ll, track=track, cho=cho, truc_xe=truc)
    tc = _ref_from(pre, chain, ll, hs, frames, extra)
    if ll is not None and ll.get('cung_lan_lap') is not True:
        u_w = tc.get('psi_lech_max_rad')
        chain = ck.chuoi_khung(frames, a6, xoay, ll, track=track, cho=cho, truc_xe=truc, u_psi_w_rad=u_w)
        tc = _ref_from(pre, chain, ll, hs, frames, extra)
        tc['u_psi_w_rad'] = u_w
    return chain, tc


def _pre_of(cua_rows, luot_id=None):
    """The 'pre' row of a run in a cua_so_v1 file: the only one, or the one of luot_id (a session file)."""
    pres = [r for r in cua_rows if r.get('loai') == 'pre']
    if luot_id is not None:
        pres = [r for r in pres if r.get('luot_id') == luot_id]
    if len(pres) != 1:
        raise ValueError(f'cua_so_v1: {len(pres)} pre rows' + (f' of luot_id {luot_id!r}' if luot_id else '') +
                         ' (need exactly one; a session file needs --luot-id)')
    return pres[0]


def _post_of(cua_rows, luot_id):
    """The 'post' row of luot_id in a cua_so_v1 file (None when there is none)."""
    posts = [r for r in cua_rows if r.get('loai') == 'post' and r.get('luot_id') == luot_id]
    if len(posts) > 1:
        raise ValueError(f'cua_so_v1: {len(posts)} post rows of luot_id {luot_id!r}')
    return posts[0] if posts else None


def _bind_sets(ll, end, row):
    """Every F8 frame set of one end (the anchor-tag intersection read by ctd.neo_of, and khung_chung when written as
    {id, pts_s}) inside the F2 row of that end WITH its PTS (Codex CT2-1, CT3-1)."""
    sets = []
    d = (ll.get('dau') or {}).get(end) if isinstance(ll.get('dau'), dict) else None
    top = ll.get('khung_chung_sha256')
    if isinstance(d, dict) and isinstance(top, dict) and end in top and 'khung_chung_sha256' in d             and d['khung_chung_sha256'] != top[end]:
        # Codex CT4-1: the two copies of the khung_chung sha (top level and dau.<end>) must agree
        raise ValueError(f'lan_lap_v1 khung_chung_sha256.{end} differs from dau.{end}.khung_chung_sha256')
    if isinstance(d, dict) and row is not None:
        # Codex CT2-1 (round 4): the end is of the window row of that end (frames, clip), when both name one
        for k in ('frames_sha256', 'clip_sha256'):
            if d.get(k) and row.get(k) and d[k] != row[k]:
                raise ValueError(f'lan_lap_v1 dau.{end}.{k} differs from cua_so_v1 {end} {k}')
    nb = ctd.neo_of(ll, end)
    if nb is not None:
        sets.append((nb['nguon'], nb['ids'], nb['pts']))
    kc = (ll.get('khung_chung') or {}).get(end) if isinstance(ll.get('khung_chung'), dict) else None
    if isinstance(kc, dict) and (nb is None or nb['nguon'] != 'khung_chung'):
        ids, pts = ctd._id_pts(kc, f'khung_chung.{end}')
        top = ll.get('khung_chung_sha256')
        ctd._check_sha(ids, top.get(end) if isinstance(top, dict) else None, f'khung_chung_sha256.{end}')
        sets.append(('khung_chung', ids, pts))
    if not sets:
        return
    if row is None or not row.get('khung'):
        if any(ids for _, ids, _ in sets):
            raise ValueError(f'lan_lap_v1 {end} frames without a {end} window in cua_so_v1')
        return
    win = {int(f['id']): float(f['pts_s']) for f in row['khung']}
    for what, ids, pts in sets:
        if not set(ids) <= set(win):
            raise ValueError(f'lan_lap_v1 {what}.{end} has frames outside the {end} window of cua_so_v1')
        if pts is not None and any(abs(p - win[i]) > 1e-9 for i, p in zip(ids, pts)):
            raise ValueError(f'lan_lap_v1 {what}.{end} pts_s differ from the PTS of those frames in cua_so_v1 {end}')


def bind_lan_lap(ll, pre, cua_sha, task, post=None):
    """Codex CT1-3: lan_lap_v1 must be the one of THIS run and of THIS window file: same luot_id and place as the
    'pre' row, cua_so_v1_sha256 = the window file given, its frame sets of both ends inside the windows of cua_so_v1
    with their PTS (pre; post when the post row is given: Codex CT2-1, CT3-1), the clip of its 'pre' end the clip of
    the window (when both name one). Any mismatch: ValueError (inputs of another run)."""
    if ll is None:
        return
    k = ctd.cho_of(task)
    if pre.get('luot_id') is None or ll.get('luot_id') != pre.get('luot_id'):
        raise ValueError(f'lan_lap_v1 luot_id {ll.get("luot_id")!r} != cua_so_v1 pre luot_id {pre.get("luot_id")!r}')
    for what, v in (('lan_lap_v1', ll.get('cho')), ('cua_so_v1 pre', pre.get('cho'))):
        if v != k:
            raise ValueError(f'{what} cho {v!r}: the task {task} needs {k!r}')
    if ll.get('cua_so_v1_sha256') != cua_sha:
        raise ValueError('lan_lap_v1 cua_so_v1_sha256 is not the window file given (NHACTAM read another file)')
    _bind_sets(ll, 'pre', pre)
    # Codex CT2-1 (round 4): the post end always (no post row: a non-empty post set of F8 is refused)
    _bind_sets(ll, 'post', post)


HANG_SO_F2 = ('u_t_ghep_s', 'troi_ppm', 't_trong_s')    # constants that enter cua_so_v1 (u_ref,i, sample gate)


def bind_clock(pre, a, hs):
    """Codex CT1-3: the window was built with the time source and the gates the run is scored with (the C8b grid file
    by sha256, or the same --offset-s, --offset-err-s, --offset-anchor-s; nguong = khung_tc of the constants)."""
    chuoi = False
    if a.gio_c8b:
        # Codex CT2-7 (CODE 19/11; C8B c8b_gio --so-bo / --cua-so, hil feat/c8b-3b-ga faa8160): the window is built
        # from the preliminary grid; the final grid names it (luoi_so_bo_sha256) and the window file built from it
        # (cua_so_v1_sha256). Same file also accepted (the window built from the final grid itself)
        g = ctd.read_json(a.gio_c8b)
        cua = getattr(a, 'cua_so', None)
        same = pre.get('gio_sha256') == cm.sha256(a.gio_c8b)
        chain_ok = (pre.get('gio_sha256') is not None and g.get('luoi_so_bo_sha256') == pre.get('gio_sha256')
                    and cua is not None and g.get('cua_so_v1_sha256') == cm.sha256(cua))
        if not (same or chain_ok):
            raise ValueError('cua_so_v1 was built with another time grid: gio_sha256 is neither --gio-c8b nor its '
                             'luoi_so_bo_sha256 (with cua_so_v1_sha256 = this window file)')
        chuoi = bool(chain_ok)
    else:
        want = dict(offset_s=a.offset_s, offset_err_s=a.offset_err_s, offset_anchor_s=a.offset_anchor_s)
        if pre.get('gio_sha256') is not None or pre.get('offset_cfg') != want:
            raise ValueError(f'cua_so_v1 time source {pre.get("offset_cfg") or "gio " + str(pre.get("gio_sha256"))} '
                             f'differs from the scoring {want}')
    kt = khung_tc_of(hs)
    if {k: pre.get('nguong', {}).get(k) for k in ctd.KHUNG_TC_DE_XUAT} != {k: kt[k] for k in ctd.KHUNG_TC_DE_XUAT}:
        raise ValueError('cua_so_v1 nguong differ from khung_tc of the constants')
    # Codex CT1-3 (round 2): the constants the window was built with (u_ref of the nodes: u_t_ghep_s, troi_ppm; the
    # sample gate: t_trong_s) must be those of the scoring
    used = pre.get('hang_so_dung')
    if not isinstance(used, dict) or any(used.get(k) != hs[k] for k in HANG_SO_F2):
        raise ValueError(f'cua_so_v1 hang_so_dung {used!r} differ from the constants of the scoring '
                         f'{ {k: hs[k] for k in HANG_SO_F2} } (or missing)')
    # Codex CT3-3 (round 4): True only for the C8 chain preliminary grid -> F2 (this file) -> final grid
    return chuoi


def main_rel(argv):
    """Sub-commands cua-so and tham-chieu (module docstring, relative frame)."""
    cmd = argv[0]
    ap = argparse.ArgumentParser(prog=f'c7_score.py {cmd}')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--hang-so', required=True)
    ap.add_argument('--out', required=True)
    if cmd == 'cua-so':
        ap.add_argument('--esp32', required=True)
        ap.add_argument('--gio-c8b', default=None)
        ap.add_argument('--offset-s', type=float, default=None)
        ap.add_argument('--offset-err-s', type=float, default=None)
        ap.add_argument('--offset-anchor-s', default=None)
        for k in ('--luot-id', '--clip', '--session', '--dets'):
            ap.add_argument(k, default=None)
    else:
        ap.add_argument('--cua-so', required=True)
        ap.add_argument('--luot-id', default=None, help='run id (a cua_so_v1 / lan_lap_v1 file of several runs)')
        ap.add_argument('--truc-xe', default=None, help='truc_xe_v1 (for the C8B q of another mounting, F16)')
        ap.add_argument('--a6', default=None)
        ap.add_argument('--xoay', default=None)
        ap.add_argument('--lan-lap', default=None)
    a = ap.parse_args(argv[1:])
    try:
        if os.path.exists(a.out):
            raise ValueError(f'{a.out} exists: use a new file')
        hs = load_hang_so(a.hang_so)
        frames = cm.read_frames(a.frames)
        track = _track_beside(a.frames)
        fsha = cm.sha256(a.frames)
        lane_T = (Fig8Task(a.task) if a.task in FIG8 else KTask(a.task, (0.0, 0.0, 0.0))).Tcmd
        if cmd == 'cua-so':
            chain = ck.chuoi_khung(frames, None, None, track=track)       # the tag gate and tag pose only
            rows, _, _, _ = read_esp32(a.esp32)
            i_edge, _, probs = ocp.find_edge(rows)
            if i_edge is None:
                raise ValueError('; '.join(probs))
            t0_us = float(rows[i_edge]['_t'])
            kt = khung_tc_of(hs)
            nodes, u, gsha, cwhy = _clock_nodes(a, rows, t0_us, frames, hs, kt['w_s'])
            # Codex CT1-14: the gates of track.json only when track.json is the one of this frames.csv (sha256) and
            # equals the locked cong_khung of the constants (check_track_gates); named among the sources
            tj = os.path.join(os.path.dirname(os.path.abspath(a.frames)), 'track.json')
            gates, gsrc = window_gates(hs, track, fsha)
            fit_ids = None
            if a.session:
                with open(a.session, encoding='utf-8') as f:
                    fit_ids = json.load(f).get('fit_ids')
            nguon = [dict(vai_tro=k, duong_dan=os.path.abspath(v), sha256=cm.sha256(v))
                     for k, v in (('frames', a.frames), ('esp32', a.esp32), ('gio_c8b', a.gio_c8b), ('clip', a.clip),
                                  ('session', a.session), ('dets', a.dets), ('hang_so', a.hang_so),
                                  ('track', tj if track is not None else None)) if v]
            extra = dict(luot_id=a.luot_id, clip=a.clip and os.path.basename(a.clip),
                         clip_sha256=a.clip and cm.sha256(a.clip), session_sha256=a.session and cm.sha256(a.session),
                         dets_sha256=a.dets and cm.sha256(a.dets), frames_sha256=fsha,
                         esp32_sha256=cm.sha256(a.esp32), gio_sha256=gsha,
                         offset_cfg=None if gsha else dict(offset_s=a.offset_s, offset_err_s=a.offset_err_s,
                                                           offset_anchor_s=a.offset_anchor_s), nguon=nguon,
                         fit_ids_nguon=fit_ids, cong_khung=dict(gates), cong_khung_nguon=gsrc,
                         hang_so_sha256=cm.sha256(a.hang_so),
                         hang_so_dung={k: hs[k] for k in HANG_SO_F2})
            out = ctd.cua_so(frames, chain, rows, t0_us, nodes, u, kt, hs['t_trong_s'], lane_T, a.task, extra,
                             gates=gates, extra_why=cwhy)
            ctd.write_jsonl(a.out, out)
            pre = out[0]['ket_luan']
            print(f'pre: {"dat" if pre["dat"] else "khong dat: " + "; ".join(pre["ly_do"])}')
        else:
            cua_rows = ctd.read_jsonl(a.cua_so)
            pre = _pre_of(cua_rows, a.luot_id)
            if pre.get('frames_sha256') != fsha:
                raise ValueError('cua_so_v1 was made from another frames.csv')
            bind_session(pre, track)
            ll = ck.doc_lan_lap(a.lan_lap, pre.get('luot_id'))
            bind_lan_lap(ll, pre, cm.sha256(a.cua_so), a.task, _post_of(cua_rows, pre.get('luot_id')))
            a6 = ck.doc_a6(a.a6)
            truc = ctd.read_json(a.truc_xe) if a.truc_xe else None
            extra = dict(cua_so_v1_sha256=cm.sha256(a.cua_so), frames_sha256=fsha,
                         a6_sha256=None if a6 is None else a6['sha256'],
                         xoay_sha256=a.xoay and cm.sha256(a.xoay), nguon=ck.nguon(a.frames, a6, a.xoay, ll),
                         truc_xe_sha256=a.truc_xe and cm.sha256(a.truc_xe))
            chain, tc = chain_and_ref(frames, a6, a.xoay, ll, track, a.task, truc, pre, hs, extra)
            with open(a.out, 'x', encoding='utf-8') as f:
                f.write(json.dumps(cm._json(tc), ensure_ascii=False, indent=1) + '\n')
            print(f'tham chieu: {tc["trang_thai"]} tap {tc["tap"]} n {tc["n"]} {"; ".join(tc["ly_do"])}')
    except Exception as exc:
        print(f'c7_score {cmd}: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


def load_rel(a, frames, track, fsha, esha, task, hs):
    """The relative-frame inputs of a scoring run (--tuong-doi), every file bound to the frames and log scored and
    to each other by sha256; the reference recomputed from them must equal tham_chieu_v1."""
    cua = ctd.read_jsonl(a.cua_so)
    pre = _pre_of(cua, a.luot_id)
    if pre.get('frames_sha256') != fsha or pre.get('esp32_sha256') != esha:
        raise ValueError('cua_so_v1 was made from another frames.csv or ESP32 log')
    chuoi_gio = bind_clock(pre, a, hs)
    bind_gates(pre, hs, track, fsha)
    bind_session(pre, track)
    tcf = ctd.read_json(a.tham_chieu)
    if tcf.get('schema') != ctd.THAM_CHIEU_SCHEMA or tcf.get('cua_so_v1_sha256') != cm.sha256(a.cua_so):
        raise ValueError('tham_chieu_v1 does not name this cua_so_v1 (sha256)')
    ll = ck.doc_lan_lap(a.lan_lap, pre.get('luot_id'))
    bind_lan_lap(ll, pre, cm.sha256(a.cua_so), task, _post_of(cua, pre.get('luot_id')))
    a6 = ck.doc_a6(a.a6)
    for k, v in (('a6_sha256', None if a6 is None else a6['sha256']), ('xoay_sha256', a.xoay and cm.sha256(a.xoay)),
                 ('lan_lap_v1_sha256', None if ll is None else ll['_sha256'])):
        if tcf.get(k) != v:
            raise ValueError(f'tham_chieu_v1 {k} differs from the file given')
    cho = ctd.cho_of(task)
    truc = ctd.read_json(a.truc_xe) if a.truc_xe else None
    if tcf.get('truc_xe_sha256') not in (None, a.truc_xe and cm.sha256(a.truc_xe)):
        raise ValueError('tham_chieu_v1 truc_xe_sha256 differs from --truc-xe')
    chain, tc = chain_and_ref(frames, a6, a.xoay, ll, track, task, truc, pre, hs,
                              {k: tcf.get(k) for k in ('cua_so_v1_sha256', 'frames_sha256', 'a6_sha256', 'xoay_sha256')})
    for k in ('ids_sha256', 'tap', 'trang_thai', 'p_ok', 'psi_ok'):
        if tc[k] != tcf.get(k):
            raise ValueError(f'tham_chieu_v1 {k} {tcf.get(k)!r} != recomputed {tc[k]!r}')
    for k in ('p_tc_m', 'psi_tc_rad'):
        x, y = tc[k], tcf.get(k)
        if (x is None) != (y is None) or (x is not None and np.max(np.abs(np.subtract(x, y))) > 1e-12):
            raise ValueError(f'tham_chieu_v1 {k} differs from the recomputed value')
    ubc = None
    if a.c8b_khoa:
        kh = ctd.read_json(a.c8b_khoa)
        st = kh.get('u_beta_cam_trang_thai')
        st = st.get(cho) if isinstance(st, dict) else st
        v = (kh.get('u_beta_cam_deg') or {}).get(cho)
        if st == 'dat' and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0.0:
            ubc = float(v)                                 # already times 1.25 (contract F16)
    src = {k: dict(duong_dan=os.path.abspath(v), sha256=cm.sha256(v)) for k, v in
           (('cua_so_v1', a.cua_so), ('tham_chieu_v1', a.tham_chieu), ('truc_xe_v1', a.truc_xe),
            ('lan_lap_v1', a.lan_lap), ('c8b_a6_v2', a.a6), ('xoay_v1', a.xoay), ('c8b_khoa_v5', a.c8b_khoa)) if v}
    rel = build_rel(task, chain, tc, truc, ll, ubc, src)
    rel['chuoi_gio_c8'] = chuoi_gio
    return rel


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ('cua-so', 'tham-chieu'):
        return main_rel(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--esp32', required=True, help='agv_logger CSV of the run (segments file beside it)')
    ap.add_argument('--task', required=True)
    ap.add_argument('--hang-so', required=True, help='constants file (c7_hang_so_v1)')
    ap.add_argument('--sender', default=None, help='pc_sender log of the run (#test_end)')
    ap.add_argument('--events', default=None, help='default <esp32>.events.txt when it exists')
    ap.add_argument('--offset-s', type=float, default=None, help='t_log = t_cam + D, skips o1_time_sync')
    ap.add_argument('--offset-reason', default='', help='why --offset-s is given')
    ap.add_argument('--offset-anchor-s', default=None,
                    help='T or A,B: command time [s], or the match window, where --offset-s was established (drift '
                         'bound from it, CS2-1; C8b v4 3)')
    ap.add_argument('--offset-err-s', type=float, default=None, help='error of --offset-s at its anchor [s]')
    ap.add_argument('--gio-c8b', default=None, help='time-grid file of the C8b script (c8b_gio_v1): score at --nut')
    ap.add_argument('--nut', type=int, default=None, help='node index of --gio-c8b (D = nodes_s[nut])')
    ap.add_argument('--k-start', default=None, help='X,Y,HEAD_DEG nominal start pose of K1/K2 (floor frame)')
    ap.add_argument('--allow-final-set', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--rows', default='', help='optional per-frame CSV of the window frames')
    ap.add_argument('--tuong-doi', action='store_true', help='relative frame (J v6; CAM v5)')
    for k in ('--cua-so', '--tham-chieu', '--truc-xe', '--lan-lap', '--a6', '--xoay', '--c8b-khoa', '--luot-id'):
        ap.add_argument(k, default=None)
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
        if (a.gio_c8b is None) != (a.nut is None):
            raise ValueError('--gio-c8b and --nut go together')
        out_op = None
        try:
            gio = load_gio(a.gio_c8b, a.nut, cm.sha256(a.esp32), fsha) if a.gio_c8b else None
        except (ValueError, OSError) as exc:
            # Codex CT3-4 (round 4): a grid file refused (ok false, ...) does not hide an operator loss
            if not a.tuong_doi:
                raise
            gio, out_op = None, operator_loss(a.esp32, a.events, a.task, f'file gio: {exc}')
            if out_op is None:
                raise
        # Codex CC1-3: the events file actually read (given, or the default beside the log), by name and sha256
        evp = a.events or (a.esp32 + '.events.txt')
        evp = evp if os.path.exists(evp) else None
        frames = cm.read_frames(a.frames)
        rel = None
        if a.tuong_doi and out_op is None:
            if not (a.out and a.rows):
                raise ValueError('--tuong-doi needs --out and --rows (F13: the frame rows go with score.json, named '
                                 'there by sha256; Codex CT1-16)')
            try:
                if not (a.cua_so and a.tham_chieu):
                    raise ValueError('--tuong-doi needs --cua-so and --tham-chieu')
                rel = load_rel(a, frames, track_info, fsha, cm.sha256(a.esp32), a.task, hs)
            except (ValueError, OSError) as exc:
                # Codex CT3-4: a run with operator evidence (no edge, so no cua_so_v1, ...) is recorded as that loss;
                # a window file never written (FileNotFoundError) too (round 4)
                out_op = operator_loss(a.esp32, a.events, a.task, f'dau vao tuong doi: {exc}')
                if out_op is None:
                    raise
        elif any((a.cua_so, a.tham_chieu, a.truc_xe, a.lan_lap, a.a6, a.xoay, a.c8b_khoa, a.luot_id)):
            raise ValueError('relative-frame inputs without --tuong-doi')
        if out_op is not None:
            out, rows = out_op, []
        else:
            out, rows = score(frames, a.esp32, a.task, hs, a.offset_s, a.offset_reason, ks,
                              a.allow_final_set, a.sender, a.events, track_info, a.offset_anchor_s, a.offset_err_s,
                              fsha, gio=gio, rel=rel)
        out.update(frames_file=os.path.basename(a.frames), frames_sha256=fsha,
                   esp32_file=os.path.basename(a.esp32), esp32_sha256=cm.sha256(a.esp32),
                   sender_file=os.path.basename(a.sender) if a.sender else None,
                   sender_sha256=cm.sha256(a.sender) if a.sender and os.path.exists(a.sender) else None,
                   events_file=os.path.basename(evp) if evp else None, events_sha256=cm.sha256(evp) if evp else None)
    except Exception as exc:            # CS1-6: any error ends with exit 2, never a traceback without a verdict
        print(f'c7_score: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    if a.rows and rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))      # every key (undefined frames have fewer)
        with open(a.rows, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        if isinstance(out.get('khung_tuong_doi_v1'), dict):
            # Codex CT1-16: the full path (--rows and --out may be in different folders)
            out['khung_tuong_doi_v1'].update(rows_file=os.path.abspath(a.rows), rows_sha256=cm.sha256(a.rows))
    if a.tuong_doi:
        # Codex CT1-16: the whole score.json without Infinity / NaN in the relative mode (not finite -> null)
        text = json.dumps(_clean(cm._json(out)), indent=1, ensure_ascii=False, allow_nan=False)
    else:
        text = json.dumps(cm._json(out), indent=1, ensure_ascii=False)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    print(f'{out["task"]}: {out["outcome"]} {"(" + out["fail_cause"] + ")" if out.get("fail_cause") else ""} '
          f'{"; ".join(out.get("outcome_reasons", []))}')
    print(f'#ket_qua,ma=0,outcome={out["outcome"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
