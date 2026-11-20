#!/usr/bin/env python3
"""
c8_hc_a.py: C8 step 12b, angle a between the roof tag and the car axis (slow straight runs sent by the Pi), and the
checks of one attempt of rpi5/c8_hc_seq.py (CODE_HIEUCHUAN, 19/11/2026).

Design: CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_CODE_HIEUCHUAN_v4.md sections 2.6, 2.7, 3; data contract
03_CODE/docs/2026-11-19_Hop_dong_du_lieu_C8_tuong_doi_v1.md (F1 c7_chuoi_khung, F2 cua_so_v1, F5 doan_12b_v1,
F6 hieu_chuan_neo, F7 truc_xe_v1, F9 e_hieu_m). Estimator of THESIS O1 v11.41 v6 T3 (proposal): a of a place = circular
mean of the four long legs; u_a = largest |a_l - a| over every long leg of every attempt + map error on the baseline
(+ the conservative terms of the design: encoder, q, yaw inside the leg, camera heading between the leg start and
the reference window).

HCB4 (20/11/2026, session 2 data; design CLAUDE_THESIS 03_CODE/docs/2026-11-20_Thiet_ke_HIEUCHUAN_HCB4_truot_cheo_v1.md,
Liam 20/11 "Duyet cach sua (Recommended)"): per leg, the camera displacement C is summed in the co-rotating tag
frame with the CAMERA heading (no IMU in a_l) and compared with the plain body-frame encoder sum E; the car creeps
along the FL-RR diagonal (0.15-0.30 % of the distance, the same body direction forwards and backwards), measured per
attempt as sigma = half the backward - forward difference of |C|/|E| and taken off (E' = E + sigma |E_x| (-1, +1)) for
the repeatability, ratio-spread and straightness gates; a = circular mean of the RAW a_l, and the largest deviation of
u_a stays on the RAW a_l. The frames must be referenced to the floor marks ('neo_san').

  python tools/c8_hc_a.py kiem DIR [--age-r-max-ms 125]                    # at the floor, log only (~10 s)
  python tools/c8_hc_a.py kiem_video DIR --clip CLIP --session SESSION      # at the floor, sparse video (2-5 min)
  python tools/c8_hc_a.py cua_so DIR... --frames F --track T [--dets D]     # windows (F2) + legs (F5), per attempt
      run 2 (T9b): ... --a6 A6 --xoay X --lan-lap lan_lap.json --ra NEW_DIR  # F5 lan_lap_id from F8 run 1
  python tools/c8_hc_a.py tinh --cho goc|K1|K2 --cap DIR [--them DIR...] --frames-of DIR=frames.csv ...
                          [--neo hieu_chuan_neo.json] [--e-hieu e_hieu.json [--doan NEW_DIR/doan_12b_v1.json]]
                          [--xoay xoay.json] [--out truc_xe_v1.json]

DIR is one attempt folder of c8_hc_seq.py (log.csv, sender.txt, lich.csv, gain_query_*.txt). The video side reads
the frames CSV of c7_measure track run with tag_offset 0 (tag_x, tag_y, psi = tag heading). The common frame chain of
CAM (tools/c7_chuoi_khung.py: A6 -> psi', centre p after q, frame gates) is used when importable AND its inputs are
given; otherwise every output is 'chan_doan' (raw tag heading, tag centre, gate valid & car_seen) and must not be
used for scoring. Locked values need: F1, c8b_a6_v2, xoay_v1 (q, du_m), F6 psi_chuan_k, F9 e_hieu_m, and the C8B
two-end camera heading bound (u_psi_cuc; not yet agreed: always chan_doan until it is).
Every angle written to F7 is in degrees (_deg); every index is BEFORE the 1.25 margin.
Last line: #ket_qua,ma=0 pass / 1 checked and failed / 2 error or refused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
import c8_hc_seq as hs                                                    # noqa: E402

VERSION = 'c8_hc_a_v1'
R_WHEEL, PPR, LXY = 0.0485, 1320, 0.24          # firmware kinematics.h, encoder.h
COUNT_M = 2 * math.pi * R_WHEEL / PPR           # 0.2309 mm of wheel travel per count
INT32_MIN = -2147483648
SP_EPS = 1e-6
# thresholds [de xuat], locked before C8 session 2 (design 3.4)
SPREAD_MAX_DEG = 0.30           # largest spread of the four a_l (THESIS v6)
SPREAD_VIDEO_DEG = 0.25         # floor decision on the raw sparse video (design 2.7)
RATIO_LO, RATIO_HI, RATIO_SPREAD = 0.98, 1.02, 0.005
PERP_MAX_M = 0.005
EPS_YAW_MAX_DEG = 0.30
EPS_SMOOTH_S = 0.5             # smoothing of camera heading and IMU yaw in the dynamic yaw check (v4 3.2 step 6)
MIN_FRAMES_TINH, MIN_FRAMES_VIDEO = 120, 60
VIDEO_STRIDE = 2               # floor read of the whole clip (decode dominates: 1/10 0.9x, 1/2 1.33x real time, 19/11)
E_COVER_TURNS, E_GAP_MAX_S = 1.5, 2.0   # E coverage per turn direction (v4 2.7)
VET_MARGIN = 1.5               # on the encoder speed bounds of the real footprint (wheel slip at these speeds)
RV_REPEAT_W = 0.1              # rad/s: while the wheels turn faster than this an RV report must change the yaw
WIN_MIN_S, STILL_MIN_S = 2.5, 3.0
U_D_MAX_S = 0.10
EXPIRY_MS = 200.0               # firmware TRAJ_TIMEOUT_US (main.c 45): run_active ends when the newest packet is this old
EXPIRY_SLACK_S = 0.03
# HCB2 (session 2, 20/11, HC_goc_2: Wi-Fi lost ~140 ms of packets twice around t = 108-112 s while turning): the
# sender-domain boundary (last row with a fresh packet, HCC1-4) and the quality limit were one number, so raising the
# limit moved the boundary into the firmware wait and broke the tail-duration check. Two numbers now:
FRESH_AGE_MS = 60.0             # boundary of the sender domain (unchanged: 3 send periods)
TRAJ_AGE_MAX_MS = EXPIRY_MS - 2 * 1000.0 / hs.SEND_HZ   # 160 ms [de xuat]: the firmware never came within two
#                                 packets of its 200 ms stop; what late packets did to the command is bounded per block
RV_STALL_S = 0.15               # 3 report periods at 20 Hz
# HCB1 (session 2, 20/11): IMU vs encoder in a turn = linear fit of the IMU yaw (at RV updates) on the encoder heading;
# the correlation of rates was 0.40-0.58 on good data (97 % of a turn is at constant rate: no signal variance, only the
# 20 Hz RV staircase). Measured HC_goc_1/2: k 1.028-1.029, residual RMS 3.2-4.1 mrad, max 12.5-14.6 mrad.
IMU_E_K_LO, IMU_E_K_HI = 0.90, 1.10          # [de xuat] IMU yaw / encoder heading (encoder under-reads ~3 %)
IMU_E_RES_MAX_RAD = 0.030                    # [de xuat] ~2x the largest residual seen; a 0.15 s stall at 0.5 rad/s: 75
SP_INT_TOL_M, SP_INT_TOL_RAD = 0.003, math.radians(0.5)
REF_TOL_M, STOP_TOL_M = 0.003, 0.015
DUR_TOL_S = 0.10


class AError(Exception):
    pass


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def circmean(a):
    a = np.asarray(a, float)
    return math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))


def mean_unwrapped(a):
    """trung_binh_go_vong (NHACTAM/CAM): arithmetic mean after unwrapping each angle around the circular mean."""
    a = np.asarray(a, float)
    c = circmean(a)
    return c + float(np.mean([wrap(x - c) for x in a]))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ attempt folder
LOG_COLS = ('timestamp_us', 'fl_count', 'fr_count', 'rl_count', 'rr_count', 'sp_vx', 'sp_vy', 'sp_wz', 'yaw_rad',
            'run_active', 'motor_on', 'pos_reset_n', 'log_seq', 'r_valid', 'rv_n', 'age_r_ms', 'reset_n',
            'psi_reset_count', 'traj_age_ms', 'pos_x_ref', 'pos_y_ref')


def load_log(path):
    rows = []
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        r = csv.DictReader(ln for ln in f if not ln.startswith('#'))
        miss = [c for c in LOG_COLS if c not in (r.fieldnames or [])]
        if miss:
            raise AError(f'{path}: missing columns {miss}')
        for row in r:
            rows.append(row)
    if not rows:
        raise AError(f'{path}: no data rows')
    L = {c: np.array([float(x[c]) for x in rows]) for c in LOG_COLS}
    L['t'] = L['timestamp_us'] * 1e-6
    L['C'] = np.column_stack([L[c] for c in ('fl_count', 'fr_count', 'rl_count', 'rr_count')]).astype(np.int64)
    L['bad'] = np.any(L['C'] == INT32_MIN, axis=1)
    L['n'] = len(rows)
    return L


def parse_fields(line):
    out = {}
    for p in line.strip().split(',')[1:]:
        if '=' in p:
            k, v = p.split('=', 1)
            out[k] = v
    return out


def load_attempt(d):
    A = dict(dir=d, thu_id=os.path.basename(os.path.normpath(d)))
    p = os.path.join(d, 'sender.txt')
    if not os.path.isfile(p):
        raise AError(f'{d}: no sender.txt')
    lines = [ln.rstrip('\n') for ln in open(p, encoding='utf-8')]
    A['sender_lines'] = lines
    tag = lambda k: [parse_fields(x) for x in lines if x.startswith(k + ',')]     # noqa: E731
    A['seq_start'] = (tag('#seq_start') or [None])[0]
    A['seq_end'] = (tag('#seq_end') or [None])[0]
    A['huy'] = tag('#seq_huy')
    A['logger_dong'] = (tag('#logger_dong') or [{}])[0]
    A['ket_qua'] = (tag('#ket_qua') or [{}])[-1]
    if A['seq_start'] is None:
        raise AError(f'{d}: no #seq_start (nothing was sent)')
    A['segs'] = read_lich(os.path.join(d, 'lich.csv'))
    if hashlib.sha256(open(os.path.join(d, 'lich.csv'), 'rb').read()).hexdigest() != A['seq_start'].get('lich_sha256'):
        raise AError(f'{d}: lich.csv sha256 differs from #seq_start')
    A['cho'] = A['seq_start'].get('cho')
    A['phan'] = A['seq_start'].get('phan')
    A['log_path'] = os.path.join(d, 'log.csv')
    A['log'] = load_log(A['log_path'])
    A['events_path'] = A['log_path'] + '.events.txt'
    return A


def read_lich(path):
    segs = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            segs.append(dict(i=int(r['i']), loai=r['loai'], t0=float(r['t0_s']), t1=float(r['t1_s']), truc=r['truc'],
                             tu=float(r['tu']), toi=float(r['toi']), v=float(r['v']), a=float(r['a'])))
    return segs


# ------------------------------------------------------------------ log analysis
def run_range(L):
    """(i0, i1) of the single run_active block; AError when not exactly one rising edge."""
    ra = L['run_active'] > 0.5
    edges = np.flatnonzero(ra[1:] & ~ra[:-1]) + 1
    if ra[0]:
        edges = np.r_[0, edges]
    if len(edges) != 1:
        raise AError(f'{len(edges)} rising edges of run_active (need exactly 1)')
    i0 = int(edges[0])
    falls = np.flatnonzero(~ra[i0:])
    i1 = i0 + (int(falls[0]) - 1 if len(falls) else len(ra) - i0 - 1)
    return i0, i1


def sp_blocks(L, i0, i1):
    """Contiguous rows of non-zero sp in [i0, i1]: list of dict(j0, j1, truc, dau, t0, t1, tich_phan)."""
    nz = (np.abs(L['sp_vx']) > SP_EPS) | (np.abs(L['sp_vy']) > SP_EPS) | (np.abs(L['sp_wz']) > SP_EPS)
    out, j = [], i0
    while j <= i1:
        if nz[j]:
            k = j
            while k + 1 <= i1 and nz[k + 1]:
                k += 1
            sl = slice(j, k + 1)
            dt = np.diff(L['t'][j:k + 2]) if k + 1 < L['n'] else np.r_[np.diff(L['t'][j:k + 1]), 0.01]
            ivx = float(np.sum(L['sp_vx'][sl] * dt[:k + 1 - j]))
            iwz = float(np.sum(L['sp_wz'][sl] * dt[:k + 1 - j]))
            ivy = float(np.sum(np.abs(L['sp_vy'][sl])))
            truc = 'wz' if abs(iwz) * LXY > abs(ivx) else 'vx'
            out.append(dict(j0=j, j1=k, truc=truc, tp_vx=ivx, tp_wz=iwz, vy_any=ivy > 0, t0=float(L['t'][j]),
                            t1=float(L['t'][k]) + 0.01))
            j = k + 1
        else:
            j += 1
    return out


def _block_dt(L, b):
    j, k = b['j0'], b['j1']
    dt = np.diff(L['t'][j:k + 2]) if k + 1 < L['n'] else np.r_[np.diff(L['t'][j:k + 1]), 0.01]
    return dt[:k + 1 - j]


def stale_s(L, j):
    """Staleness of the command at row j beyond one send period [s] (traj_age_ms - 1/SEND_HZ, >= 0); 0 outside."""
    if j < 0 or j >= L['n']:
        return 0.0
    return max(float(L['traj_age_ms'][j]) / 1000.0 - 1.0 / hs.SEND_HZ, 0.0)


def stale_bound(L, b, main_key, v_top, a):
    """HCB2: bound of the change of the integral of one command block caused by late or lost packets (session 2,
    20/11, HC_goc_2: six packets lost on the down ramp of a turn, 0.386 rad/s held 140 ms -> +0.0095 rad = a age^2/2).
    A held value differs from the planned one only where the plan changes: on the ramps (logged |sp| below the plateau
    v_top), by at most a x staleness, staleness = traj_age - one send period. Late packets that reach the car in a
    burst carry an age that hides their delay: not covered, so such a block fails (conservative)."""
    sl = slice(b['j0'], b['j1'] + 1)
    ramp = np.abs(L[main_key][sl]) < v_top - SP_EPS
    exc = np.maximum(L['traj_age_ms'][sl] / 1000.0 - 1.0 / hs.SEND_HZ, 0.0)
    return float(a * np.sum((exc * _block_dt(L, b))[ramp]))


def body_increments(L):
    """Per row i>=1: body-frame (dx, dy) from the count increments (kinematics_forward), and wheel-rate wz."""
    dC = np.diff(L['C'], axis=0).astype(float) * COUNT_M
    dx = 0.25 * (dC[:, 0] + dC[:, 1] + dC[:, 2] + dC[:, 3])
    dy = 0.25 * (-dC[:, 0] + dC[:, 1] + dC[:, 2] - dC[:, 3])
    dth = 0.25 / LXY * (-dC[:, 0] + dC[:, 1] - dC[:, 2] + dC[:, 3])
    return np.r_[0.0, dx], np.r_[0.0, dy], np.r_[0.0, dth]


def odometry_from(L, j_ref):
    """Positions (n, 2) in the body frame at row j_ref, integrating the increments rotated by the IMU yaw change."""
    dx, dy, _ = body_increments(L)
    yaw = np.unwrap(L['yaw_rad'])
    mid = np.r_[yaw[0], 0.5 * (yaw[1:] + yaw[:-1])] - yaw[j_ref]
    c, s = np.cos(mid), np.sin(mid)
    X = np.cumsum(c * dx - s * dy)
    Y = np.cumsum(s * dx + c * dy)
    return np.column_stack([X - X[j_ref], Y - Y[j_ref]])


def still_windows(L, blocks, i0, i1):
    """Log windows of the stops: list of dict(ta, tb, ja, jb) in order (before the first block, between blocks,
    after the last). Start = counts at rest for good + 0.3 s; end = next sp edge - 0.1 s, last = end of run - tail."""
    moving = np.r_[False, np.any(np.diff(L['C'], axis=0) != 0, axis=1)]
    bounds = [(i0, blocks[0]['j0'])] + [(blocks[k]['j1'] + 1, blocks[k + 1]['j0']) for k in range(len(blocks) - 1)]
    bounds.append((blocks[-1]['j1'] + 1, i1 + 1))
    out = []
    for n, (a, b) in enumerate(bounds):
        mv = np.flatnonzero(moving[a:b])
        rest = a + (int(mv[-1]) + 1 if len(mv) else 0)
        if rest >= b:
            out.append(dict(ta=None, tb=None, ja=None, jb=None, ly_do='khong_dung'))
            continue
        ta = float(L['t'][rest]) + 0.3
        if n == len(bounds) - 1:
            tb = float(L['t'][min(b - 1, L['n'] - 1)]) - hs.TAIL_S - 0.1
        else:
            tb = float(L['t'][b]) - 0.1
        still_s = float(L['t'][min(b, L['n'] - 1)] - L['t'][rest])
        w = dict(ta=ta, tb=tb, ja=rest, jb=b - 1, still_s=still_s)
        seq = np.diff(L['log_seq'][rest:b])
        w['ho'] = bool(np.any(seq != 1)) or bool(np.any(np.diff(L['t'][rest:b]) > 0.03))
        out.append(w)
    return out


def plan_moves(segs):
    return [s for s in segs if s['loai'] in ('tien', 'xoay')]


def sender_domain(L, i0, i1):
    """(j_s, problems). j_s = last row of the run with a fresh packet (0 <= traj_age_ms <= FRESH_AGE_MS): up to
    there the sender was sending. Rows j_s+1..i1 are the firmware waiting for its 200 ms timeout after the last
    (zero) packet (main.c 524-525; HCC1-4): there no new packet may arrive (age only grows), the age stays under
    EXPIRY_MS, the wait is not longer than EXPIRY_MS (+ slack) and the command is 0 on every axis."""
    age = L['traj_age_ms']
    fresh = np.flatnonzero((age[i0:i1 + 1] >= 0) & (age[i0:i1 + 1] <= FRESH_AGE_MS))
    if not len(fresh):
        raise AError('no fresh trajectory packet inside the run')
    j_s = i0 + int(fresh[-1])
    problems = []
    # HCC2-2: the falling edge must be in the log, the rows continuous across it, the age there >= EXPIRY_MS and
    # the wait before it reaching EXPIRY_MS - EXPIRY_SLACK_S (a log cut while run_active is 1 is not a full run)
    if i1 >= L['n'] - 1:
        problems.append('khong thay canh ha run_active (log het khi run_active=1)')
    else:
        if L['log_seq'][i1 + 1] != L['log_seq'][i1] + 1:
            problems.append('ho log_seq o canh ha')
        dt_us = int(L['timestamp_us'][i1 + 1]) - int(L['timestamp_us'][i1])      # HCC4-4: integer microseconds
        if not 0 < dt_us <= 30000:                       # same tolerance as the windows (gap > 30 ms)
            problems.append(f'canh ha cach {dt_us} us (can 0 < dt <= 30000)')
        if abs((age[i1 + 1] - age[i1]) - dt_us / 1000.0) > 11.0:
            problems.append(f'tuoi goi qua canh ha {age[i1]:.0f}->{age[i1 + 1]:.0f} ms khong khop dt {dt_us} us')
        if L['traj_age_ms'][i1 + 1] < EXPIRY_MS:
            problems.append(f"canh ha khi traj_age {L['traj_age_ms'][i1 + 1]:.0f} ms < 200")
    if i1 <= j_s or age[i1] < EXPIRY_MS - EXPIRY_SLACK_S * 1000:
        problems.append(f'khong thay phan cho het han (age cuoi {age[i1]:.0f} ms)')
    if i1 > j_s:
        ex = slice(j_s + 1, i1 + 1)
        if np.any(np.diff(age[j_s:i1 + 1]) < 0):
            problems.append('goi moi trong phan het han')
        if age[i1] >= EXPIRY_MS:
            problems.append(f'traj_age {age[i1]:.0f} ms with run_active')
        if L['t'][i1] - L['t'][j_s] > EXPIRY_MS / 1000.0 + EXPIRY_SLACK_S:
            problems.append(f"het han {L['t'][i1] - L['t'][j_s]:.3f} s")
        if any(np.any(np.abs(L[c][ex]) > SP_EPS) for c in ('sp_vx', 'sp_vy', 'sp_wz')):
            problems.append('lenh khac 0 khi het han')
    return j_s, problems


def read_events_unique(path):
    """'#' texts of the logger events file, receive time dropped, identical copies (the firmware sends #us/#us_end
    3 times) kept once, in order; [] when there is no file."""
    import gain_line_check as glc
    ev = glc.read_events(path) or []
    seen, out = set(), []
    for line in ev:
        _, txt = glc.split_event(line)
        txt = txt.strip()
        if txt not in seen:
            seen.add(txt)
            out.append(txt)
    return out


def _int(v):
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def logger_summary_problem(log_path):
    """None, or why the agv_logger summary does not prove a closed run of this very CSV (HCC1-6)."""
    try:
        s = json.load(open(log_path + '.summary.json', encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return f'summary: {type(exc).__name__}'
    if s.get('status') != 'closed':
        return f"summary status {s.get('status')}"
    if ((s.get('files') or {}).get('csv') or {}).get('sha256') != sha256(log_path):
        return 'summary csv sha256 khac log.csv'
    return None


def us_evidence(texts, boot, L, i0, i1):
    """(changes inside the log, covering #us_end or None, why) from the #us / #us_end lines (us_state.h (c); HCC1-6).
    A #us_end covers the run when its boot is the boot of the gain questions, its lượt starts at or before the run
    (#us,ev=start of that boot with first_seq <= log_seq[i0], or no start line and motor_on = 1 from the first log
    row) and ends at or after it (log_seq[i1] <= last_data_seq <= last log_seq) with motor_on = 1 in between."""
    seq = L['log_seq']
    s0, s_i0, s_i1, s_end = seq[0], seq[i0], seq[i1], seq[-1]
    us = [(t.split(',', 1)[0], parse_fields(t)) for t in texts if t.startswith('#us,') or t.startswith('#us_end,')]
    changes = [f for k, f in us if k == '#us' and f.get('ev') == 'change'
               and _int(f.get('first_seq')) is not None and s0 <= _int(f.get('first_seq')) <= s_end]
    starts = [f for k, f in us if k == '#us' and f.get('ev') == 'start' and f.get('boot', '').lower() == boot]
    ends = [f for k, f in us if k == '#us_end' and f.get('boot', '').lower() == boot]
    cover, why = None, 'khong co #us_end cung boot trong log'
    for e in ends:
        last = _int(e.get('last_data_seq'))
        if last is None or not (s_i1 <= last <= s_end):
            why = f'#us_end last_data_seq {e.get("last_data_seq")} ngoai [{s_i1:.0f}, {s_end:.0f}]'
            continue
        jl = int(np.searchsorted(seq, last, side='right')) - 1
        st = [s for s in starts if _int(s.get('first_seq')) is not None and _int(s.get('first_seq')) <= s_i0
              and (_int(e.get('ticks')) is None or _int(e.get('ticks')) == last - _int(s.get('first_seq')) + 1)]
        if st:
            j0 = max(0, int(np.searchsorted(seq, _int(st[-1].get('first_seq')))))
        elif not starts and np.all(L['motor_on'][:i0 + 1] > 0.5):
            j0 = 0
        else:
            why = '#us,ev=start khong khop luot'
            continue
        if not np.all(L['motor_on'][j0:jl + 1] > 0.5):
            why = 'motor tat giua luot #us'
            continue
        if cover is not None and cover != e:                 # HCC2-3: two different footers both fit
            return changes, None, '#us_end mau thuan (nhieu ban khac noi dung cung phu luot)'
        cover, why = e, ''
    return changes, cover, why


def kiem(A, age_r_max_ms=125.0):
    """Log-side checks of one attempt (design 2.6; Codex code review round 1 HCC1-4..6). Returns (checks list of
    (name, ok, detail), info)."""
    L, segs = A['log'], A['segs']
    ck = []

    def add(name, ok, detail=''):
        ck.append((name, bool(ok), str(detail)))
    se = A['seq_end'] or {}
    add('sender_hoan_tat', se.get('complete') == '1' and se.get('n_bo') == '0' and not A['huy']
        and A['ket_qua'].get('ma') == '0',
        f"complete={se.get('complete')} n_bo={se.get('n_bo')} huy={len(A['huy'])} ket_qua={A['ket_qua'].get('ma')}")
    add('logger_dong_sach', A['logger_dong'].get('ma') == '0', f"ma={A['logger_dong'].get('ma')}")
    why = logger_summary_problem(A['log_path'])
    add('logger_summary', why is None, why or 'closed, sha256 khop')
    info = dict(cat_pwm='chua_xac_minh')
    try:
        i0, i1 = run_range(L)
    except AError as exc:
        add('mot_canh_run_active', False, exc)
        return ck, info
    add('mot_canh_run_active', True, f'rows {i0}-{i1}')
    seq = L['log_seq']
    try:
        import gain_line_check as glc
        qb = glc.read_query(os.path.join(A['dir'], 'gain_query_truoc.txt'))
        qa = glc.read_query(os.path.join(A['dir'], 'gain_query_sau.txt'))
        why = glc.query_problem(qb, qa, check_run_boot=False)
        same = why is None and all((qb or {}).get(k) == (qa or {}).get(k) for k in ('sha', 'env', 'applied_n'))
        add('dinh_danh_gain', same, why or '')
        boot = (qb or {}).get('boot', '').lower()
        texts = read_events_unique(A['events_path'])
        fw = [parse_fields(t) for t in texts if t.startswith('#fw,')]
        gn = [parse_fields(t) for t in texts if t.startswith('#gain,')]
        okfw = all(f.get('sha', '')[:7] == (qb or {}).get('sha', '')[:7] and f.get('env') == (qb or {}).get('env')
                   for f in fw)
        bad_g = [g for g in gn if g.get('boot', '').lower() != boot or _int(g.get('first_seq')) is None
                 or not (seq[0] <= _int(g.get('first_seq')) <= seq[i0])]
        add('dinh_danh_fw_gain_log', okfw and not bad_g and bool(boot),
            f"#fw {len(fw)} #gain {len(gn)}" + (f"; #gain sai boot/first_seq: {bad_g[0]}" if bad_g else ''))
        changes, cover, why_us = us_evidence(texts, boot, L, i0, i1)
        info['blk_ticks'] = None if cover is None else _int(cover.get('blk_ticks'))   # whole motor run: info only
        add('khong_cat_sieu_am', not changes, f'{len(changes)} #us change; blk_ticks '
            f"{cover.get('blk_ticks') if cover else '-'} (chi bao, ca luot motor)")
        if cover is None:
            info['cat_pwm'], info['cat_pwm_ly_do'] = 'chua_xac_minh', why_us
        else:
            ct = _int(cover.get('cut_ticks'))
            info['cat_pwm'] = 'chua_xac_minh' if ct is None else ('khong' if ct == 0 else 'co')
            info['cat_pwm_ly_do'] = f"cut_ticks={cover.get('cut_ticks')}"
    except ImportError:
        add('dinh_danh_gain', False, 'gain_line_check not importable')
    rng = slice(i0, i1 + 1)
    add('motor_on', np.all(L['motor_on'][rng] > 0.5))
    add('pos_reset_tang_1', i0 > 0 and L['pos_reset_n'][i1] - L['pos_reset_n'][i0 - 1] == 1
        and L['pos_reset_n'][i0] == L['pos_reset_n'][i1],
        f"{L['pos_reset_n'][max(i0 - 1, 0)]:.0f}->{L['pos_reset_n'][i1]:.0f}")
    add('log_seq_lien', np.all(np.diff(seq[rng]) == 1))
    add('so_dem_hop_le', not np.any(L['bad'][rng]))
    try:
        j_s, exp_prob = sender_domain(L, i0, i1)
    except AError as exc:
        add('traj_age', False, exc)
        return ck, info
    age_max = float(np.nanmax(L['traj_age_ms'][i0:j_s + 1]))
    add('traj_age', age_max <= TRAJ_AGE_MAX_MS and age_max >= 0, f'max {age_max:.0f} ms (mien phat {i0}-{j_s})')
    add('het_han_firmware', not exp_prob, '; '.join(exp_prob) or f"{L['t'][i1] - L['t'][j_s]:.3f} s, lenh 0")
    # every non-zero command row of the WHOLE log (HCC1-5): inside the sender domain and matching the schedule
    blocks_all = sp_blocks(L, 0, L['n'] - 1)
    blocks = [b for b in blocks_all if i0 <= b['j0'] and b['j1'] <= j_s]
    out = [b for b in blocks_all if b not in blocks]
    add('khong_goi_muon', not out, '' if not out else
        f"{len(out)} khoi lenh khac 0 ngoai mien phat, dau tien dong {out[0]['j0']} (t {out[0]['t0']:.2f} s)")
    moves = plan_moves(segs)
    t_run = float(L['t'][i0])
    ok_b = len(blocks) == len(moves)
    detail = f'{len(blocks)} blocks vs {len(moves)} planned'
    if ok_b:
        ref_bound = 0.0                          # HCB2: pos_x_ref carries the late-packet changes of earlier legs
        tre = []
        for b, m in zip(blocks, moves):
            dur, want = m['t1'] - m['t0'], m['toi'] - m['tu']
            sl = slice(b['j0'], b['j1'] + 1)
            key = 'sp_vx' if m['loai'] == 'tien' else 'sp_wz'
            B = stale_bound(L, b, key, m['v'], m['a'])
            if B > 0.1 * (SP_INT_TOL_M if m['loai'] == 'tien' else SP_INT_TOL_RAD):
                tre.append(f"{m['loai']} {B:.4f}")
            if m['loai'] == 'tien':
                main, others, tp = L['sp_vx'][sl], ('sp_vy', 'sp_wz'), b['tp_vx']
                ref_bound += B
                good = (abs(tp - want) <= SP_INT_TOL_M + B
                        and abs(L['pos_x_ref'][b['j1']] - m['toi']) <= REF_TOL_M + ref_bound)
            else:
                main, others, tp = L['sp_wz'][sl], ('sp_vx', 'sp_vy'), b['tp_wz']
                good = abs(tp - want) <= SP_INT_TOL_RAD + B
            pure = all(np.all(np.abs(L[c][sl]) <= SP_EPS) for c in others)
            one_sign = np.all(np.sign(main[np.abs(main) > SP_EPS]) == math.copysign(1.0, want))
            # HCB2: a late first packet starts the block late (the zero packets before it aged), a late zero packet
            # ends it late (the last value held): the time checks widen by the staleness seen at those rows
            s0, s1 = stale_s(L, b['j0'] - 1), stale_s(L, b['j1'])
            t_ok = (abs((b['t0'] - t_run) - m['t0']) <= DUR_TOL_S + s0
                    and abs((b['t1'] - b['t0']) - dur) <= DUR_TOL_S + s0 + s1)
            if not (good and pure and one_sign and t_ok):
                ok_b = False
                detail = (f"block {b['j0']}: {m['loai']} tich phan {tp:.4f} vs {want:.4f} (can goi tre {B:.4f}), "
                          f"truc khac 0 {not pure}, doi dau {not one_sign}, bat dau {b['t0'] - t_run:.2f} vs "
                          f"{m['t0']:.2f} s, dai {b['t1'] - b['t0']:.2f} vs {dur:.2f} s")
                break
        if ok_b and tre:
            detail += f"; goi tre tren doan tang/giam toc, can cong them: {', '.join(tre)}"
        if ok_b:                                 # the zero tail: hold + TAIL_S sent whole up to the last packet
            n_pk = hs.n_samples(segs) + int(math.ceil(hs.TAIL_S * hs.SEND_HZ - 1e-9))
            span = float(L['t'][j_s]) - t_run
            if abs(span - (n_pk - 1) / hs.SEND_HZ) > DUR_TOL_S:
                ok_b = False
                detail = f'mien phat {span:.2f} s vs lich {(n_pk - 1) / hs.SEND_HZ:.2f} s (duoi goi 0)'
    add('lenh_nhan_khop_lich', ok_b, detail)
    # IMU
    ok_imu = np.all(L['r_valid'][rng] > 0.5) and np.nanmax(L['age_r_ms'][rng]) <= age_r_max_ms
    ok_imu = ok_imu and L['reset_n'][i0] == L['reset_n'][i1] and L['psi_reset_count'][i0] == L['psi_reset_count'][i1]
    rv = L['rv_n'][rng]
    tt = L['t'][rng]
    ch = np.flatnonzero(np.diff(rv) != 0)
    stall = float(np.max(np.diff(tt[np.r_[0, ch + 1, len(tt) - 1]]))) if len(tt) > 1 else 0.0
    add('imu_tuoi_khong_reset', ok_imu and stall <= RV_STALL_S, f'rv stall {stall:.3f} s')
    if ok_b:
        _, _, dth = body_increments(L)
        yaw = np.unwrap(L['yaw_rad'])
        for b in blocks:
            if b['truc'] != 'wz':
                continue
            sl = slice(b['j0'], b['j1'] + 1)
            t = L['t'][sl]
            if len(t) < 60:
                add('imu_e_tuong_quan', False, 'block too short')
                continue
            # HCB1: IMU yaw at the RV updates fitted on the encoder heading (same body turn, encoder scale ~3 % low);
            # a frozen, jumping or mis-scaled IMU leaves a slope off 1 or a residual (no need for rate variance)
            th = np.cumsum(dth[sl])
            upd = np.r_[True, np.diff(L['rv_n'][sl]) != 0]
            X = np.column_stack([th[upd] - th[0], np.ones(int(upd.sum()))])
            coef = np.linalg.lstsq(X, yaw[sl][upd], rcond=None)[0]
            res = yaw[sl][upd] - X @ coef
            kk, r_max, r_rms = float(coef[0]), float(np.max(np.abs(res))), float(np.sqrt(np.mean(res ** 2)))
            fit_ok = IMU_E_K_LO <= kk <= IMU_E_K_HI and r_max <= IMU_E_RES_MAX_RAD
            k = np.ones(25) / 25
            sign_ok = np.sign(yaw[b['j1']] - yaw[b['j0']]) == np.sign(b['tp_wz'])
            # HCC1-9: a repeated RV value while the wheels turn (a 0.15 s repeat kept corr ~0.95)
            rate = np.convolve(np.abs(dth[sl]) / np.maximum(np.r_[np.diff(t), 0.01], 1e-3), k, 'same')
            ch = np.flatnonzero(np.diff(L['rv_n'][sl]) != 0) + 1
            ysl = L['yaw_rad'][sl]
            same = [j for j0_, j in zip(ch[:-1], ch[1:]) if rate[j] > RV_REPEAT_W and abs(ysl[j] - ysl[j0_]) < 1e-6]
            add('imu_e_tuong_quan', fit_ok and sign_ok and not same,
                f'yaw IMU = k goc encoder: k {kk:.4f}, du max {1e3 * r_max:.1f} mrad (RMS {1e3 * r_rms:.1f}), '
                f'dung chieu {bool(sign_ok)}, RV lap luc quay {len(same)}')
        wins = still_windows(L, blocks, i0, j_s)
        add('cua_so_dung_du', all(w.get('still_s', 0) >= STILL_MIN_S and not w.get('ho') for w in wins),
            ', '.join(f"{w.get('still_s', 0):.2f}" for w in wins))
        # odometry stop vs plan (catches a PWM cut with a correct reference)
        odo = odometry_from(L, wins[0]['ja'])
        stops = [0.0] + [m['toi'] for m in moves if m['loai'] == 'tien']
        bad = []
        k = 0
        for w in wins:
            if k >= len(stops) or w['ja'] is None:
                break
            j = (w['ja'] + w['jb']) // 2
            if abs(odo[j, 0] - stops[k]) > STOP_TOL_M:
                bad.append(f'stop {k}: {odo[j, 0]:.4f} vs {stops[k]}')
            k += 1
        add('diem_dung_odo', not bad, '; '.join(bad))
        info.update(i0=i0, i1=i1, j_s=j_s, blocks=blocks, windows=wins)
    return ck, info


# ------------------------------------------------------------------ camera
def frames_basic(frames_csv, track_json):
    """Diagnostic chain: tag centre and raw tag heading of c7_measure track (tag_offset must be 0)."""
    tj = json.load(open(track_json, encoding='utf-8'))
    off = tj.get('tag_offset')                       # HCC7-8: must be present and (0, 0, 0): psi is the tag heading
    if not isinstance(off, (list, tuple)) or len(off) != 3 or any(abs(float(v)) > 0 for v in off):
        raise AError(f'{track_json}: tag_offset {off} (track with tag_offset (0, 0, 0))')
    t, x, y, ps, ok, idx = [], [], [], [], [], []
    with open(frames_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                vals = [float(r[c]) for c in ('t_s', 'tag_x', 'tag_y', 'psi')]
            except (KeyError, ValueError):
                continue
            good = r.get('valid') == '1' and r.get('car_seen') in ('1', 'True') and all(math.isfinite(v) for v in vals)
            idx.append(int(r['idx']))
            t.append(vals[0]); x.append(vals[1]); y.append(vals[2]); ps.append(vals[3]); ok.append(good)
    F = dict(idx=np.array(idx), t=np.array(t), x=np.array(x), y=np.array(y), psi=np.array(ps), ok=np.array(ok, bool))
    F['chuoi'] = 'chan_doan'
    F['neo_san'] = True          # c7_measure track refits the camera pose on the floor marks in every frame
    return F


def a6_tam_id(a6_path):
    """The plate of an A6 file: c8b_a6_v2.tam_id (top-level key; CODE 20/11 A6C1-7: declared once in the A6 file,
    other sessions only compare). None without a path, a readable JSON object, or a non-empty string tam_id."""
    if not a6_path:
        return None
    try:
        with open(a6_path, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    t = d.get('tam_id') if isinstance(d, dict) else None
    return t if isinstance(t, str) and t else None


def _accepts(fn, name):
    try:
        import inspect
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def frames_chain(frames_csv, a6, xoay, lan_lap=None, track=None, cho=None):
    """F1 of CAM when importable: frame_id, t_cam, p_x, p_y, psi_ma_p, cong_ok, a6_ok, q_ok, trang_thai.
    None (caller falls back to frames_basic, chain 'chan_doan') when CAM or C8B (c8b_a6) is not merged yet, when
    the chain refuses its inputs (ValueError: schema, track tag_offset), or when no frame has A6 AND q applied
    (e.g. xoay_v1 without a6_sha256 or of another A6: CAM gives q None, p NaN). A frame is used only when its gate,
    A6 and q are all ok (cong_ok alone would let NaN centres through). 'khoa' needs every gated frame A6 + q ok.
    HCV3-1 (CODE 20/11, contract log): without lan_lap the plate is c8b_a6_v2.tam_id of the A6 file, passed as
    tam_id= to CAM chuoi_khung once CAM has that parameter (CAM adds it after session 2; before that the call is the
    old one and CAM's A6 plate check refuses, so the chain stays chan_doan). No lan_lap is ever built here."""
    try:
        import c7_chuoi_khung as ck                                       # CAM, after its merge
        kw = dict(lan_lap=lan_lap, track=track, cho=cho)
        if lan_lap is None and _accepts(ck.chuoi_khung, 'tam_id'):
            kw['tam_id'] = a6_tam_id(a6)
        rows = ck.chuoi_khung(frames_csv, ck.doc_a6(a6), xoay, **kw)
    except (ImportError, ValueError):
        return None
    if not rows:
        return None
    t = np.array([float(r['t_cam']) for r in rows], float)
    gated = np.array([bool(r['cong_ok']) for r in rows])
    full = gated & np.isfinite(t) & np.array([bool(r.get('a6_ok')) and bool(r.get('q_ok')) for r in rows])
    if not full.any():
        return None
    F = dict(idx=np.array([r['frame_id'] for r in rows]), t=t,
             x=np.array([r['p_x'] for r in rows]), y=np.array([r['p_y'] for r in rows]),
             psi=np.array([r['psi_ma_p'] for r in rows]), ok=full,
             tx=np.array([float(r.get('tag_x', np.nan)) for r in rows]),     # HCH1-2: tag centre (A6 residual)
             ty=np.array([float(r.get('tag_y', np.nan)) for r in rows]))
    # HCC7-7: 'khoa' only when every gated frame has a finite PTS, A6 and q, and the PTS strictly increase
    F['chuoi'] = 'khoa' if bool(np.all(full[gated])) and bool(np.all(np.diff(t[full]) > 0)) else 'chan_doan'
    F['neo_san'] = True          # rows of c7_measure track: camera pose refitted on the floor marks per frame
    return F


def _rate(t, x, y, p, half):
    """Centred time-difference speeds over +-half (s): translation (m/s) and heading rate (rad/s) per sample."""
    a = np.searchsorted(t, t - half)
    b = np.minimum(np.searchsorted(t, t + half, side='right') - 1, len(t) - 1)
    dt = t[b] - t[a]
    ok = (b > a) & (dt > 0)
    dt = np.where(ok, dt, 1.0)
    v = np.where(ok, np.hypot(x[b] - x[a], y[b] - y[a]) / dt, 0.0)
    w = np.where(ok, np.abs(p[b] - p[a]) / dt, 0.0)
    return v, w


def _intervals(t, s, thr, min_s=0.5, max_hole_s=0.2):
    """[(start, end)] where s > thr, edges by linear interpolation of the crossing between samples; holes shorter
    than max_hole_s are bridged, intervals shorter than min_s dropped."""
    mv = s > thr
    out, i, n = [], 0, len(t)
    while i < n:
        if not mv[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mv[j + 1]:
            j += 1

        def cross(k0, k1):
            if k0 < 0 or k1 >= n or s[k1] == s[k0]:
                return float(t[max(min(k1, n - 1), 0)])
            f = (thr - s[k0]) / (s[k1] - s[k0])
            return float(t[k0] + f * (t[k1] - t[k0]))
        st, en = cross(i - 1, i), cross(j, j + 1) if j + 1 < n else float(t[j])
        if out and st - out[-1][1] < max_hole_s:
            out[-1] = (out[-1][0], en)
        else:
            out.append((st, en))
        i = j + 1
    return [(a, b) for a, b in out if b - a > min_s]


def cam_motion_events(F, thr_v=0.02, thr_w=math.radians(5.0), half=0.25, rot_guard_s=0.5):
    """[(start, end, kind)] camera motion intervals in time order (HCC1-8): 'wz' where the tag heading turns,
    'vx' where the tag centre moves while the heading does NOT turn (within rot_guard_s of a turn): the centre of a
    tag off the turning axis moves during E and must not count as a translation."""
    m = F['ok']
    t, x, y, p = F['t'][m], F['x'][m], F['y'][m], np.unwrap(F['psi'][m])
    if len(t) < 10:
        return []
    v, w = _rate(t, x, y, p, half)
    rot = w > thr_w
    near_rot = np.zeros(len(t), bool)
    for a, b in _intervals(t, w, thr_w, min_s=0.0, max_hole_s=0.0):
        near_rot |= (t >= a - rot_guard_s) & (t <= b + rot_guard_s)
    ev = []
    for a, b in _intervals(t, np.where(rot, w, 0.0), thr_w):        # HCC2-4: kind carries the turn direction
        ia, ib = np.searchsorted(t, a), min(np.searchsorted(t, b), len(t) - 1)
        ev.append((a, b, 'wz+' if p[ib] - p[ia] > 0 else 'wz-'))
    ev += [(a, b, 'vx') for a, b in _intervals(t, np.where(near_rot, 0.0, v), thr_v)]
    return sorted(ev)


def log_motion_events(L, blocks, thr_v=0.02, thr_w=math.radians(5.0), half=0.25):
    """(start, end) log times of motion of each block, with the SAME speed thresholds, window and interpolated
    edges as the camera: translation speed from the encoder odometry, turn rate from the wheel increments."""
    dx, dy, dth = body_increments(L)
    t = L['t']
    X, Y, TH = np.cumsum(dx), np.cumsum(dy), np.cumsum(dth)
    v, w = _rate(t, X, Y, TH, half)
    out = []
    for k, b in enumerate(blocks):
        j0 = blocks[k - 1]['j1'] + 1 if k else 0
        j1 = blocks[k + 1]['j0'] if k + 1 < len(blocks) else L['n']
        sl = slice(j0, j1)
        s = v[sl] if b['truc'] == 'vx' else w[sl]
        iv = _intervals(t[sl], s, thr_v if b['truc'] == 'vx' else thr_w)
        inside = [e for e in iv if e[1] >= L['t'][b['j0']] - 0.5 and e[0] <= L['t'][b['j1']] + 2.0]
        out.append(None if not inside else (inside[0][0], inside[-1][1]))
    return out


def align(L, blocks, F):
    """D = t_log - t_cam from the motion edges (HCC1-8): the camera intervals are matched to the log blocks IN
    ORDER and BY KIND (translation of 12b, turns of E); extra camera intervals before or after (the car placed by
    hand) are skipped by choosing the consecutive run of camera intervals with the same kinds and the smallest
    spread. Returns (D, u_D, diffs)."""
    lev = log_motion_events(L, blocks)
    if any(e is None for e in lev):
        raise AError('a schedule block without motion in the encoder log')
    lk = [(s, e, b['truc'] if b['truc'] == 'vx' else ('wz+' if b['tp_wz'] > 0 else 'wz-'))
          for (s, e), b in zip(lev, blocks)]
    cev = cam_motion_events(F)
    kinds = [k for _, _, k in lk]
    best = None
    for o in range(len(cev) - len(lk) + 1):
        cand = cev[o:o + len(lk)]
        if [k for _, _, k in cand] != kinds:
            continue
        diffs = []
        for (cs, ce, _), (ls, le, _) in zip(cand, lk):
            diffs += [ls - cs, le - ce]
        D = float(np.median(diffs))
        u = float(np.max(np.abs(np.array(diffs) - D)))
        if best is None or u < best[1]:
            best = (D, u, diffs)
    if best is None:
        raise AError(f"camera intervals {[k for _, _, k in cev]} do not contain the log sequence {kinds}")
    return best


def rv_samples(L):
    """(t, yaw) of the rotation-vector reports: rows where rv_n changes, timed at row time - age_r_ms (HCC1-9)."""
    rv = L['rv_n']
    j = np.r_[0, np.flatnonzero(np.diff(rv) != 0) + 1]
    t = L['t'][j] - L['age_r_ms'][j] * 1e-3
    yaw = np.unwrap(L['yaw_rad'][j])
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], yaw[keep]


def leg_cam(F, L, ja, jb, D, wa, wb, smooth_s=None, rv=None):
    """HCB4: the camera side of one leg between the centres of the stop windows wa and wb (log rows ja, jb).
    C: displacement of the centre in the CO-ROTATING tag frame = sum over the frames of R(-psi_mid) dp, from the
    mean of window a through every gated frame to the mean of window b (camera heading only: no IMU, no exact
    camera-log time). eps: largest |camera heading - IMU yaw| over the leg after taking off the straight line
    through the two end windows (a steady IMU drift does not count; the encoder is no referee: session 2 K2 showed
    its heading slips), both averaged over +-smooth_s/2 on the camera frame times, the IMU interpolated in the RV
    report times (HCC1-9); eps_tb: the path-weighted mean of the same difference; cam_imu: the camera - IMU
    heading change between the two windows (diagnostic: IMU drift). eps, eps_tb, cam_imu are NaN (no evidence) with
    an RV gap > RV_STALL_S, a camera gap > smooth_s, or fewer than 80 % of the leg covered. None: < 10 frames."""
    smooth_s = EPS_SMOOTH_S if smooth_s is None else smooth_s
    t_rv, y_rv = rv if rv is not None else rv_samples(L)
    ta, tb = float(L['t'][ja]), float(L['t'][jb])
    mk = F['ok'] & (F['t'] + D >= ta) & (F['t'] + D <= tb)
    if mk.sum() < 10:
        return None
    tc = F['t'][mk] + D
    p = np.unwrap(F['psi'][mk])
    pa = p[0] + wrap(wa['psi'] - p[0])
    pb = p[-1] + wrap(wb['psi'] - p[-1])
    X = np.r_[wa['p'][0], F['x'][mk], wb['p'][0]]
    Y = np.r_[wa['p'][1], F['y'][mk], wb['p'][1]]
    pm = 0.5 * (np.r_[pa, p][:len(p) + 1] + np.r_[p, pb])              # heading at the middle of each step
    ddx, ddy = np.diff(X), np.diff(Y)
    c, s = np.cos(pm), np.sin(pm)
    C = np.array([float(np.sum(c * ddx + s * ddy)), float(np.sum(-s * ddx + c * ddy))])   # R(-pm) (ddx, ddy)
    nan = float('nan')
    out = dict(C=C, dpsi=float(pb - pa), gap=float(np.max(np.diff(np.r_[ta, tc, tb]))), eps=nan, eps_tb=nan,
               cam_imu=nan)
    if out['gap'] > smooth_s:
        return out
    r0, r1 = np.searchsorted(t_rv, ta), np.searchsorted(t_rv, tb)
    seg = t_rv[max(r0 - 1, 0):min(r1 + 1, len(t_rv))]
    if len(seg) < 2 or np.max(np.diff(seg)) > RV_STALL_S:
        return out
    yi = np.interp(tc, t_rv, y_rv)
    d_b = wrap((pb - pa) - (wb['yaw'] - wa['yaw']))
    lin = (p - pa) - (yi - wa['yaw']) - d_b * (tc - ta) / (tb - ta)
    lin = (lin + math.pi) % (2 * math.pi) - math.pi
    h = smooth_s / 2
    a = np.searchsorted(tc, tc - h)
    b = np.searchsorted(tc, tc + h, side='right')
    cs = np.r_[0.0, np.cumsum(lin)]
    n = b - a
    full = (tc - h >= ta) & (tc + h <= tb) & (n >= 3)
    if full.sum() < 3 or (tc[full][-1] - tc[full][0]) < 0.8 * (tb - ta - smooth_s):
        return out
    sm = (cs[b] - cs[a]) / np.maximum(n, 1)
    w = np.hypot(np.diff(F['x'][mk]), np.diff(F['y'][mk]))
    out['eps'] = float(np.max(np.abs(sm[full])))
    out['eps_tb'] = float(np.sum(0.5 * (lin[1:] + lin[:-1]) * w) / np.sum(w)) if np.sum(w) > 0 else 0.0
    out['cam_imu'] = float(d_b)
    return out


def e_coverage(F, L, blocks, D):
    """Per turn block of E (HCC1-7): useful turns seen by the camera (sum of |heading change| between consecutive
    tag samples no further apart than E_GAP_MAX_S) and the largest gap without the tag, over the turning interval
    of the log mapped to the camera."""
    lev = log_motion_events(L, blocks)
    out = []
    for b, e in zip(blocks, lev):
        if b['truc'] != 'wz':
            continue
        if e is None:
            out.append(dict(vong=0.0, ho_s=float('inf')))
            continue
        a, c = e[0] - D, e[1] - D
        m = F['ok'] & (F['t'] >= a) & (F['t'] <= c)
        ts = F['t'][m]
        if len(ts) < 2:
            out.append(dict(vong=0.0, ho_s=c - a))
            continue
        gaps = np.diff(np.r_[a, ts, c])
        p = np.unwrap(F['psi'][m])
        dp = math.copysign(1.0, b['tp_wz']) * np.diff(p)          # HCC2-4: signed progress in the log direction
        good = np.diff(ts) <= E_GAP_MAX_S
        out.append(dict(vong=float(np.sum(dp[good]) / (2 * math.pi)), ho_s=float(np.max(gaps))))
    return out


def cam_window(F, w, D, u_D, fps=60.0, gated=True):
    """Frame mask of the camera image of log window w, shrunk by u_D and one frame each side; gated=False keeps
    every decoded frame of that time range (F2 'khung', HCC1-11), True only the frames passing the tag gate."""
    if w['ta'] is None:
        return np.zeros(len(F['t']), bool)
    a = w['ta'] - D + u_D + 1.0 / fps
    b = w['tb'] - D - u_D - 1.0 / fps
    m = (F['t'] >= a) & (F['t'] <= b)
    return F['ok'] & m if gated else m


# ------------------------------------------------------------------ legs and a
LONG_LEGS = ((1, 2), (2, 3), (3, 4), (4, 5))
SHORT_LEGS = ((0, 1), (5, 6))
STOPS_12B = 7                   # windows 0..6: the stops of 12b (ref_12b, dung_12b)
TROI_MIN_WIN = 4                # HCB3: at least 4 stop windows for the 2-parameter drift line
CAM_TROI_MIN_DEG_S = 0.001      # HCB3 [de xuat]: a camera-vs-encoder slope under this is not called a camera drift
#                                 (<= 0.005 deg on a over a half leg); it still enters u_yaw (|slope| x half leg)


def imu_drift_fit(W, L, skip=(), trong_tai=True):
    """DIAGNOSTIC ONLY since HCB4 (a_l does not use the IMU; analyse calls it with trong_tai=False, so the
    IMU - camera line is reported whatever the encoder says: session 2 K2 showed the encoder heading slips while
    the camera, referenced to the floor marks, does not turn). Kept as written for HCB3:
    HCB3 cach A (Liam 20/11 at CODE, button: "Cach A"; session 2 K1/K2: the IMU yaw drifted 0.03-0.05 deg/s
    against the camera, the encoder agreeing with the camera). Over the 12b stop windows with a camera mean, leaving
    out the windows in skip (the two ends of the leg being checked: no self-fit), at least TROI_MIN_WIN:
      - line (2 parameters) of the IMU drift d_k = (yaw_k - yaw_0) - wrap(psi_k - psi_0) [rad] against the window
        centre time (camera as reference: residuals ~0.1 deg);
      - camera stability against the encoder: slope of e_k = wrap(psi_k - psi_0) - (th_k - th_0); |slope| > 2 SE ->
        the camera itself may drift (mount): no correction (a camera drift moved into the IMU would be hidden);
        a slope under CAM_TROI_MIN_DEG_S is not a drift (it enters u_yaw instead).
    Returns None (fewer windows), dict(ly_do='camera_troi_so_encoder', ...) or dict(pf, n, du_max (in-fit),
    du_ngoai {k: |residual|} of the skipped windows, se_ce [rad/s], doc_ce [rad/s])."""
    ks = [k for k in range(min(STOPS_12B, len(W))) if 'psi' in W[k]]
    if not ks:
        return None
    k0 = ks[0]
    t = {k: float(L['t'][W[k]['jc']]) for k in ks}
    cam = {k: wrap(W[k]['psi'] - W[k0]['psi']) for k in ks}
    d = {k: (W[k]['yaw'] - W[k0]['yaw']) - cam[k] for k in ks}
    e = {k: cam[k] - (W[k]['th_enc'] - W[k0]['th_enc']) for k in ks}
    fit = [k for k in ks if k not in skip]
    if len(fit) < TROI_MIN_WIN:
        return None
    tt = np.array([t[k] for k in fit])
    pe = np.polyfit(tt, [e[k] for k in fit], 1)
    re = np.array([e[k] for k in fit]) - np.polyval(pe, tt)
    se = float(np.sqrt(np.sum(re ** 2) / (len(fit) - 2)) / np.sqrt(np.sum((tt - tt.mean()) ** 2)))
    if trong_tai and abs(pe[0]) > 2 * se and abs(pe[0]) > math.radians(CAM_TROI_MIN_DEG_S):
        return dict(ly_do='camera_troi_so_encoder', n=len(fit), doc_ce=float(pe[0]), se_ce=se)
    pf = np.polyfit(tt, [d[k] for k in fit], 1)
    res = {k: abs(d[k] - float(np.polyval(pf, t[k]))) for k in ks}
    return dict(pf=pf, n=len(fit), du_max=max(res[k] for k in fit),
                du_ngoai={k: res[k] for k in ks if k in skip}, se_ce=se, doc_ce=float(pe[0]))


def analyse(A, F, k_info, min_frames=MIN_FRAMES_TINH):
    """Windows, legs, a_l of one attempt (12b part). Returns dict; raises AError when not computable."""
    L = A['log']
    blocks, wins = k_info['blocks'], k_info['windows']
    D, u_D, diffs = align(L, blocks, F)
    if u_D > U_D_MAX_S:
        raise AError(f'u_D {u_D:.3f} s > {U_D_MAX_S}')
    n12 = sum(1 for s in plan_moves(A['segs']) if s['loai'] == 'tien')
    if n12 not in (0, 6):
        raise AError(f'{n12} translations in the schedule (12b has 6)')
    W = []
    yaw = np.unwrap(L['yaw_rad'])
    dx, dy, dth_ = body_increments(L)
    th_enc = np.cumsum(dth_)                                # encoder heading (diagnostic doc_ce of imu_drift_fit)
    for k, w in enumerate(wins):
        m = cam_window(F, w, D, u_D, F.get('fps', 60.0))
        lw = (L['t'] >= w['ta']) & (L['t'] <= w['tb']) if w['ta'] is not None else np.zeros(L['n'], bool)
        rec = dict(chi_so=k, n=int(m.sum()), mask=m, mask_all=cam_window(F, w, D, u_D, F.get('fps', 60.0), False),
                   ta=w['ta'], tb=w['tb'])
        if rec['n'] >= min_frames:
            rec['p'] = np.array([F['x'][m].mean(), F['y'][m].mean()])
            # HCH1-2: the tag centre (frames_basic: x, y are the tag centre) for the C8B two-end bound
            rec['p_tag'] = np.array([F.get('tx', F['x'])[m].mean(), F.get('ty', F['y'])[m].mean()])
            rec['psi'] = mean_unwrapped(F['psi'][m])
            rec['cs'] = (float(np.mean(np.cos(F['psi'][m]))), float(np.mean(np.sin(F['psi'][m]))))
            rec['yaw'] = float(np.mean(yaw[lw]))
            rec['th_enc'] = float(np.mean(th_enc[lw]))
            rec['jc'] = int(np.flatnonzero(lw)[len(np.flatnonzero(lw)) // 2])
            h = int(m.sum()) // 2
            idx = np.flatnonzero(m)
            rec['psi_nua'] = abs(wrap(mean_unwrapped(F['psi'][idx[:h]]) - mean_unwrapped(F['psi'][idx[h:]])))
        W.append(rec)
    rvs = rv_samples(L)
    legs = []
    for (ia, ib), dai in ([] if n12 == 0 else [(lg, True) for lg in LONG_LEGS] + [(lg, False) for lg in SHORT_LEGS]):
        wa, wb = W[ia], W[ib]
        leg = dict(i=len(legs), dai=dai, a=ia, b=ib)
        if 'p' not in wa or 'p' not in wb:
            leg.update(trang_thai='thieu', ly_do=f'window {ia if "p" not in wa else ib} < {min_frames} frames')
            legs.append(leg)
            continue
        ja, jb = wa['jc'], wb['jc']
        cam = leg_cam(F, L, ja, jb, D, wa, wb, rv=rvs)
        if cam is None:
            leg.update(trang_thai='thieu', ly_do=f'fewer than 10 frames between windows {ia} and {ib}')
            legs.append(leg)
            continue
        # HCB4: both displacements in the body (co-rotating) frame. Encoder: plain sum of the firmware body
        # increments (no yaw needed); camera: leg_cam (camera heading). a_l_tho = arg E - arg C
        dp = wb['p'] - wa['p']
        ds = np.array([float(np.sum(dx[ja + 1:jb + 1])), float(np.sum(dy[ja + 1:jb + 1]))])
        dc = cam['C']
        a_tho = wrap(math.atan2(ds[1], ds[0]) - math.atan2(dc[1], dc[0]))
        r_tho = float(np.hypot(*dc) / np.hypot(*ds))
        path = float(np.sum(np.hypot(dx[ja + 1:jb + 1], dy[ja + 1:jb + 1])))
        leg.update(trang_thai='hop_le', dp=dp, ds=ds, dc=dc, L_m=float(np.hypot(*dp)), Ls_m=float(np.hypot(*ds)),
                   ti_le_tho=r_tho, ti_le=r_tho, a_l_tho=a_tho, a_l=a_tho, theta_a=wrap(wa['psi'] - a_tho),
                   eps=cam['eps'], eps_tb=cam['eps_tb'], cam_imu=cam['cam_imu'], dpsi=cam['dpsi'],
                   ho_khung_s=cam['gap'], path=path,
                   chieu='toi' if ds[0] > 0 else 'lui', R_a=wa['psi'], R_b=wb['psi'], cs_a=wa['cs'], cs_b=wb['cs'],
                   nua_a=wa['psi_nua'], nua_b=wb['psi_nua'], p_a=wa['p'], p_b=wb['p'],
                   p_tag_a=wa['p_tag'], p_tag_b=wb['p_tag'])
        legs.append(leg)
    tc = slip_compensate(legs)
    th = stops_straightness(W, legs, tc.get('sigma')) if n12 else dict(tho=float('nan'), bu=float('nan'))
    tr_all = imu_drift_fit(W, L, trong_tai=False) if n12 else None     # diagnostic: IMU - camera line over the stops
    troi = None if tr_all is None else (dict(ly_do=tr_all['ly_do']) if 'pf' not in tr_all else dict(
        toc_do_deg_s=math.degrees(float(tr_all['pf'][0])), n_cua_so=tr_all['n'],
        du_khop_max_deg=math.degrees(tr_all['du_max']), doc_ce_deg_s=math.degrees(tr_all['doc_ce']),
        se_ce_deg_s=math.degrees(tr_all['se_ce'])))
    return dict(D=D, u_D=u_D, diffs=diffs, windows=W, legs=legs, chuoi=F['chuoi'], troi_imu=troi, truot_cheo=tc,
                thang_hang_m=th, neo_san=bool(F.get('neo_san', False)))


def slip_compensate(legs):
    """HCB4 (design 2026-11-20_Thiet_ke_HIEUCHUAN_HCB4_truot_cheo_v1.md; Liam 20/11): the car creeps along the
    FL-RR diagonal of the body, about 0.15-0.30 % of the distance, the SAME body direction forwards and backwards
    (session 2: 5 attempts, direction of the even camera-encoder difference 47.0 +- 4.2 deg; the encoders do not
    see it). sigma = (mean raw ratio |C|/|E| of the backward long legs - mean of the forward ones) / 2; every leg
    gets E' = E + sigma*|E_x|*(-1, +1), a_l = arg E' - arg C, ti_le = |C|/|E'| (long and short legs: the creep is
    proportional to the distance). Fewer than one long leg in each direction -> no compensation (a_l = a_l_tho)
    and the reason. Returns dict(sigma, r_lui, r_toi, ly_do)."""
    longs = [lg for lg in legs if lg.get('dai') and lg['trang_thai'] == 'hop_le']
    lui = [lg['ti_le_tho'] for lg in longs if lg['chieu'] == 'lui']
    toi = [lg['ti_le_tho'] for lg in longs if lg['chieu'] == 'toi']
    if not lui or not toi:
        return dict(sigma=None, r_lui=lui, r_toi=toi, ly_do=f'doan dai hop le: {len(lui)} lui, {len(toi)} toi')
    sigma = 0.5 * (float(np.mean(lui)) - float(np.mean(toi)))
    for lg in legs:
        if lg['trang_thai'] != 'hop_le':
            continue
        s = sigma * abs(lg['ds'][0])
        e = np.array([lg['ds'][0] - s, lg['ds'][1] + s])
        lg['a_l'] = wrap(math.atan2(e[1], e[0]) - math.atan2(lg['dc'][1], lg['dc'][0]))
        lg['ti_le'] = float(np.hypot(*lg['dc']) / np.hypot(*e))
    return dict(sigma=sigma, r_lui=lui, r_toi=toi, ly_do=None)


def stops_straightness(W, legs, sigma):
    """Largest distance of the 12b stops from their common line [m]: 'tho' as measured, 'bu' after taking off the
    slip drift predicted by sigma (each leg moves the car sigma*|E_x|*(-1, +1) in the body frame; cumulated along
    the legs in time order, turned to the floor by the body heading at the leg start = tag heading - mean raw a)."""
    ks = [k for k in range(min(STOPS_12B, len(W))) if 'p' in W[k]]
    tho = perp_offsets([W[k] for k in ks])
    good = [lg for lg in legs if lg['trang_thai'] == 'hop_le']
    longs = [lg for lg in good if lg['dai']]
    if sigma is None or not longs:
        return dict(tho=tho, bu=tho)
    a = circmean([lg['a_l_tho'] for lg in longs])
    drift, acc = {}, np.zeros(2)
    for lg in sorted(good, key=lambda g: g['a']):
        hd = lg['R_a'] - a
        s = sigma * abs(lg['ds'][0])
        acc = acc + s * np.array([-math.cos(hd) - math.sin(hd), -math.sin(hd) + math.cos(hd)])
        drift[lg['b']] = acc.copy()
    pts = [dict(p=W[k]['p'] - drift.get(k, np.zeros(2))) for k in ks]
    return dict(tho=tho, bu=perp_offsets(pts))


def perp_offsets(W):
    pts = np.array([w['p'] for w in W if 'p' in w])
    if len(pts) < 3:
        return float('nan')
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c)
    return float(np.max(np.abs((pts - c) @ vt[1])))


def _deg_leg(lg):
    """One leg for F7: angles in degrees with _deg names (HCC1-12), arrays as lists, internal fields dropped."""
    out = {}
    for k, v in lg.items():
        if k in ('R_a', 'R_b', 'cs_a', 'cs_b', 'nua_a', 'nua_b', 'p_tag_a', 'p_tag_b'):
            continue
        if k in ('a_l', 'a_l_tho', 'theta_a', 'eps', 'eps_tb', 'cam_imu', 'dpsi'):
            out[k + '_deg'] = None if v is None or not math.isfinite(v) else math.degrees(v)
        else:
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def place_result(cho, attempts, cap_id, e_hieu=None, q=None, du_m=None, neo=None, psi_cuc=None, cap_ok=True,
                 cap_why=(), bao_q=False, vet=None, psi_cuc_chot=False, need_windows=None, psi_hai_dau=None,
                 can_hai_dau=None, p_ref_tag=None):
    for v in (psi_cuc, psi_hai_dau):                       # HCC3-6: never let max()/sin() hide a bad bound
        if v is not None and not (math.isfinite(v) and 0.0 <= v <= math.pi):
            raise AError(f'can ngoai mien [0, pi]: {v}')
    can_why = []

    def can(p1, p2):
        """Two-end camera heading bound of C8B (c8b_a6.u_psi_hai_dau_rad, design v4 3.3) for two window tag
        positions, or None (its reasons kept: the term is then missing, never a bound of 0; HCH1-4: a bound out of
        [0, pi] is missing too, with its reason)."""
        u, why = can_hai_dau(p1, p2)
        if u is not None and not (isinstance(u, (int, float)) and math.isfinite(u) and 0.0 <= u <= math.pi):
            u, why = None, [f'can ngoai mien [0, pi]: {u}']
        if u is None:
            can_why.extend(w for w in why if w not in can_why)
            return None
        return float(u)
    """Combine the analysed attempts of one place (F7 block). attempts = list of (thu_id, analysis or None, why).

    cap_ok False (the issued attempt failed kiem) -> chua_ket_luan: a is never issued from such an attempt; its legs
    and those of every other attempt still count in the largest deviation (HCC1-10: issuing a is separate from the
    legs of a failed attempt having enough evidence). e_hieu: {leg i: e [m]} of F9 for the issued attempt (each long
    leg needs 0 <= e < L, HCC1-12). need_windows: window indices that must hold >= MIN_FRAMES_TINH frames (HCC1-11).
    Every early stop returns the same keys (values None) instead of raising."""
    attempts = [(x[0], x[1], x[2] if len(x) > 2 else ()) for x in attempts]
    res = dict(cho=cho, trang_thai='khoa', ly_do=[], luot_thu=[], a_k_deg=None, u_a_k_deg=None, do_tan_deg=None,
               u_hang_deg={}, cong={}, giu_ngoai=[], troi_imu=None)
    for tid, an, why in attempts:
        res['luot_thu'].append(dict(thu_id=tid, cap=tid == cap_id, trang_thai='thieu' if an is None else 'tinh_duoc',
                                    ly_do=list(why) if why else [],
                                    D_s=None if an is None else an['D'], u_D_s=None if an is None else an['u_D'],
                                    doan=[] if an is None else [_deg_leg(lg) for lg in an['legs']]))
    cap = {tid: an for tid, an, _ in attempts}.get(cap_id)

    def stop(why):
        return dict(res, trang_thai='chua_ket_luan', ly_do=list(why))
    if not cap_ok:
        return stop(['luot cap truot kiem: ' + ', '.join(cap_why)])
    if cap is None:
        return stop(['luot cap khong tinh duoc'])
    if need_windows:
        few = [(k, cap['windows'][k]['n']) for k in need_windows if cap['windows'][k]['n'] < MIN_FRAMES_TINH]
        if few:
            return stop([f'cua so thieu khung (< {MIN_FRAMES_TINH}): {few}'])
    longs = [lg for lg in cap['legs'] if lg['dai'] and lg['trang_thai'] == 'hop_le']
    if len(longs) != 4:
        return stop(['cap: fewer than 4 long legs'])
    # HCB4 (Liam 20/11, CODE_HIEUCHUAN p5: "Duyet cach sua (Recommended)"): a of the place = circular mean of the
    # four RAW a_l (the spec estimator; the diagonal-slip compensation cancels in it); the repeatability gate runs
    # on the compensated a_l; the largest deviation of u_a stays on the RAW a_l of every long leg of every attempt
    # (the whole compensation counted as uncertain: covers a car that drifts in one direction only)
    a = circmean([lg['a_l_tho'] for lg in longs])
    spread = max(abs(wrap(x['a_l'] - y['a_l'])) for x in longs for y in longs)
    spread_tho = max(abs(wrap(x['a_l_tho'] - y['a_l_tho'])) for x in longs for y in longs)
    tc = cap.get('truot_cheo') or {}
    all_long = [lg for _, an, _ in attempts if an is not None for lg in an['legs']
                if lg['dai'] and lg['trang_thai'] == 'hop_le']
    dev = max(abs(wrap(lg['a_l_tho'] - a)) for lg in all_long)
    Lmin = min(lg['L_m'] for lg in longs)
    Lsmin = min(lg['Ls_m'] for lg in longs)
    bad = []
    if e_hieu is None:
        res['ly_do'].append('thieu e_hieu_m (F9)')
        e = 0.0
    else:
        es = []
        for lg in longs:
            v = e_hieu.get(lg['i'])
            if v is None or not math.isfinite(v) or v < 0 or v >= lg['L_m']:
                bad.append(f"e_hieu_m doan {lg['i']} = {v} (can 0 <= e < L {lg['L_m']:.3f})")
            else:
                es.append(v)
        e = max(es) if es else 0.0
    u_map = math.asin(e / Lmin) if e < Lmin else math.pi / 2
    r_enc = 2 * COUNT_M / Lsmin                            # HCB4: E is a plain body-frame sum (no rotation terms)
    u_enc = math.asin(r_enc) if r_enc < 1 else math.pi / 2
    hd_thieu = False
    thieu_hang = []                                        # HCH1-4: terms without a bound: exported null, not 0
    # HCH1-3: with the C8B function the max runs over the long legs of every attempt of the place (as the
    # deviation term), not only the issued one
    legs_u = all_long if can_hai_dau is not None else longs
    cap_legs = {id(lg) for lg in longs}                    # HCH2-3: the F9 replay (bao_q) covers the issued run only
    if q is None or du_m is None:
        res['ly_do'].append('thieu q, du_m (xoay_v1)')
        thieu_hang.append('u_q')
        u_q = 0.0
    else:
        qn = math.hypot(*q)
        u_q = 0.0
        for lg in legs_u:
            bao = bao_q and id(lg) in cap_legs
            if can_hai_dau is not None and not bao:        # HCH1-7: F9 replay holds the q turn term: not needed
                v = can(lg['p_tag_a'], lg['p_tag_b'])
                hd_thieu = hd_thieu or v is None
                psi_hai_dau = None if v is None else v
            # ||mean R_b - mean R_a||_2 of the window rotation matrices (HCC1-12), and the turn error of q from the
            # error of the DIFFERENCE of the two end headings (both half-split errors, opposite signs, added)
            dR = math.hypot(lg['cs_b'][0] - lg['cs_a'][0], lg['cs_b'][1] - lg['cs_a'][1])
            # HCC2-8: the bound of the end-heading difference comes from C8B (u_psi_hai_dau_rad); the half-split
            # spread of the two windows only enters as a floor (window noise), never as the bound itself
            u_hd = max(lg['nua_a'] + lg['nua_b'], psi_hai_dau or 0.0)
            term = dR * du_m + (0.0 if (bao_q and id(lg) in cap_legs) else qn * 2 * math.sin(u_hd / 2))
            u_q = max(u_q, term / lg['L_m'])
    if q is not None and du_m is not None and (not bao_q or len(legs_u) > len(longs)) \
            and (psi_hai_dau is None or hd_thieu):
        res['ly_do'].append('thieu can sai hieu huong hai dau (C8B u_psi_hai_dau_rad) cho u_q')
        thieu_hang.append('u_q')
    eps = [lg['eps'] for lg in longs]
    eps_thieu = any(not math.isfinite(x) for x in eps)            # HCC1-9: no evidence -> gate fails, not 0
    eps = [x if math.isfinite(x) else 0.0 for x in eps]
    # HCB4: a_l no longer uses the IMU. A camera heading error in motion turns C by its PATH-WEIGHTED MEAN, so
    # u_yaw = largest |eps_tb| (camera - IMU, the straight line through the two end windows taken off); the static
    # part by position belongs to the C8B two-end bound (u_psi_cuc)
    u_yaw = max((abs(lg['eps_tb']) for lg in longs if math.isfinite(lg['eps_tb'])), default=0.0)
    if not cap.get('neo_san'):
        res['ly_do'].append('khung khong neo moc san (huong camera co the troi)')
    if tc.get('sigma') is None:
        res['ly_do'].append('khong bu truot cheo: ' + str(tc.get('ly_do')))
    res['troi_imu'] = dict(cap.get('troi_imu') or {},
                           cam_imu_hai_dau_deg=[math.degrees(lg['cam_imu']) if math.isfinite(lg['cam_imu']) else None
                                                for lg in longs],
                           cach='chi thong tin (HCB4): a_l khong dung IMU; yaw_dong neo hai dau doan nen IMU troi deu '
                                'khong anh huong')
    if can_hai_dau is not None:
        # design v4 3.3: u_psi_cuc = max over the long legs of the bound between the start window of the leg (at
        # the extreme, where a takes psi') and the reference window (where psi_chuan_k is taken)
        # HCH2-4: the reference end is the tag centre of the F6 frame set (khung_chung) when given
        if isinstance(p_ref_tag, str):                     # HCV3-4: F6 frames not all found with a tag centre
            can_why.append('dau chuan: khong dung lai du moi khung F6 (tam ma huu han)')
            p_ref = None
        else:
            p_ref = p_ref_tag if p_ref_tag is not None else cap['windows'][0].get('p_tag')
        vals = [None if p_ref is None else can(lg['p_tag_a'], p_ref) for lg in legs_u]
        if any(v is None for v in vals):
            psi_cuc = None
        else:
            psi_cuc, psi_cuc_chot = max(vals), True
    if psi_cuc is None:
        res['ly_do'].append('thieu can huong camera hai dau (C8B)')
        thieu_hang.append('u_psi_cuc')
        u_cuc = 0.0
    else:
        u_cuc = float(psi_cuc)
        if not psi_cuc_chot:
            res['ly_do'].append('can C8B: ten truong chua chot (hop dong muc 4.5)')
    u_a = dev + u_map + u_enc + u_q + u_yaw + u_cuc
    gates = dict(
        do_tan=spread <= math.radians(SPREAD_MAX_DEG),
        # HCB4: range on the raw and on the compensated ratio, spread on the compensated one
        ti_le=all(RATIO_LO <= lg[k] <= RATIO_HI for lg in longs for k in ('ti_le', 'ti_le_tho'))
        and (max(lg['ti_le'] for lg in longs) - min(lg['ti_le'] for lg in longs)) <= RATIO_SPREAD,
        thang_hang=cap['thang_hang_m']['bu'] <= PERP_MAX_M,      # the 7 stops of 12b, slip drift taken off
        yaw_dong=not eps_thieu and all(x <= math.radians(EPS_YAW_MAX_DEG) for x in eps),
        e_hieu_hop_le=not bad,
    )
    shorts = [lg for lg in cap['legs'] if not lg['dai']]
    giu_ngoai = []
    for lg in shorts:
        if lg['trang_thai'] != 'hop_le':                       # HCC1-11: a lost held-out leg is not skipped
            giu_ngoai.append(dict(i=lg['i'], dat=False, ly_do=lg.get('ly_do', 'thieu')))
            continue
        u_short = math.asin(min(e / lg['L_m'], 1.0)) + math.asin(min(2 * COUNT_M / lg['Ls_m'], 1.0))
        giu_ngoai.append(dict(i=lg['i'], a_l_deg=math.degrees(lg['a_l']), lech_deg=math.degrees(abs(wrap(lg['a_l'] - a))),
                              nguong_deg=math.degrees(u_a + u_short),
                              dat=abs(wrap(lg['a_l'] - a)) <= u_a + u_short))
    gates['giu_ngoai'] = len(shorts) == len(SHORT_LEGS) and all(g['dat'] for g in giu_ngoai)
    if vet is None:
        res['ly_do'].append('thieu vet thuc (--moc)')
    else:
        gates['vet_thuc'] = vet['khe_m'] > 0.0
    if cap['chuoi'] != 'khoa':
        res['ly_do'].append('chuoi khung chua khoa (F1, A6, q)')
    if neo is None:
        res['ly_do'].append('thieu psi_chuan_k (F6)')
    failed = [k for k, v in gates.items() if not v]
    res['ly_do'] += bad + [f'can C8B: {w}' for w in can_why]
    if failed:
        res['trang_thai'] = 'chua_ket_luan'
        res['ly_do'] += failed
    elif res['ly_do']:
        res['trang_thai'] = 'chan_doan'
    hang = dict(lech_lon_nhat=math.degrees(dev), u_map=math.degrees(u_map), u_enc=math.degrees(u_enc),
                u_q=math.degrees(u_q), u_yaw=math.degrees(u_yaw), u_psi_cuc=math.degrees(u_cuc))
    # HCH1-4: with the C8B bound function a missing bound is null, and so is the sum (the gates above used the
    # partial sum: smaller threshold, stricter); without it the earlier output is kept (listed in u_hang_thieu)
    rong = thieu_hang if can_hai_dau is not None else []
    for k in rong:
        hang[k] = None
    out = dict(res, a_k_deg=math.degrees(a), u_a_k_deg=None if rong else math.degrees(u_a),
               do_tan_deg=math.degrees(spread), do_tan_tho_deg=math.degrees(spread_tho),
               truot_cheo=dict(tc, cach='HCB4: sigma = (tb ti_le_tho doan lui - tb ti_le_tho doan toi)/2 tren bon doan dai; '
                                        'E bu = E + sigma*|E_x|*(-1, +1) (duong cheo FL-RR); a_k = tb vong a_l_tho; '
                                        'lech_lon_nhat tren a_l_tho; do_tan, ti_le (tan), thang_hang tren so da bu'),
               thang_hang_mm=dict(tho=1e3 * cap['thang_hang_m']['tho'], bu=1e3 * cap['thang_hang_m']['bu']),
               u_hang_deg=hang, u_hang_thieu=sorted(set(thieu_hang)),
               u_q_gom_xoay_q=(not bao_q) or len(legs_u) > len(longs),       # HCV3-5: some leg carries it
               u_q_bao_boi_f9=sorted({tid for tid, an_, _ in attempts if tid == cap_id}) if bao_q else [],
               cong=gates, giu_ngoai=giu_ngoai, vet_thuc=vet)
    if neo is not None:
        out['phi_hat_chuan_k_deg'] = math.degrees(wrap(math.radians(neo) - a))
        out['u_phi_hat_chuan_k_deg'] = out['u_a_k_deg']
    return out


# ------------------------------------------------------------------ CLI
F2_FIELDS = ('schema', 'luot_id', 'thu_id', 'cho', 'loai', 'chi_so', 'clip', 'clip_sha256', 'session_sha256',
             'fit_ids_nguon', 'frames_sha256', 'khung', 'khung_sha256', 'khung_ma_hop_le', 'khung_ma_hop_le_sha256',
             'phep_gop', 't_canh_s', 'bien_cuoi', 'vi_tri_lenh_m', 'D_s', 'u_D_s', 't0_log_us', 'esp32_sha256', 'chuoi')
F5_HEAD = ('schema', 'thu_id', 'cho', 'lan_lap_id', 'clip', 'clip_sha256', 'session_sha256', 'frames_sha256',
           'track_sha256', 'cua_so_v1_sha256', 'esp32_sha256', 'commit_hil', 'car_pose_cfg', 'xoay_sha256',
           'a6_sha256', 'chuoi', 'dets_sha256', 'dets_fp', 'nguon', 'doan', 'trang_thai')
F5_WIN = ('loai', 'chi_so', 'khung_id', 'khung_pts_s', 'khung_sha256', 'p_m', 'psi_rad')
F5_LEG = ('i', 'dai', 'chieu', 'vi_tri_lenh_m', 'cua_so_a', 'cua_so_b', 'dp_m', 'L_m', 'trang_thai', 'ly_do')


def f2_problems(row):
    """Missing or inconsistent fields of one F2 cua_so_v1 row (contract 2.A + HIEUCHUAN keys; HCC1-11). There is
    no reader of the receivers yet (NHACTAM, C8B, KIEMCHEO): this is the writer-side check."""
    p = [f'thieu {k}' for k in F2_FIELDS if k not in row]
    if p:
        return p
    if row['schema'] != 'cua_so_v1':
        p.append('schema')
    ids = [k['id'] for k in row['khung']]
    if any(b <= a for a, b in zip([k['pts_s'] for k in row['khung']][:-1], [k['pts_s'] for k in row['khung']][1:])):
        p.append('khung: PTS khong tang')
    if not set(row['khung_ma_hop_le']) <= set(ids):
        p.append('khung_ma_hop_le khong la tap con cua khung')
    if row['bien_cuoi'] == 'duoi_log' and row['t_canh_s'] is not None:
        p.append('cua so cuoi: t_canh_s phai null')
    if row['bien_cuoi'] == 'canh_lenh' and row['t_canh_s'] is None:
        p.append('t_canh_s null khi bien_cuoi canh_lenh')
    if row['khung_sha256'] != _ids_sha(ids) or row['khung_ma_hop_le_sha256'] != _ids_sha(row['khung_ma_hop_le']):
        p.append('khung sha256')
    return p


def f5_problems(doan):
    """Missing fields of F5 doan_12b_v1 (contract 2.D + KIEMCHEO; HCC1-11) and its content sha256."""
    p = [f'thieu {k}' for k in F5_HEAD if k not in doan]
    for d in doan.get('doan', []):
        p += [f"doan {d.get('i')}: thieu {k}" for k in F5_LEG if k not in d]
        for nm in ('cua_so_a', 'cua_so_b'):
            p += [f"doan {d.get('i')} {nm}: thieu {k}" for k in F5_WIN if k not in (d.get(nm) or {})]
    for k in ('side_m', 'h_m', 'tag_offset', 'q_m', 'du_m'):
        if k not in (doan.get('car_pose_cfg') or {}):
            p.append(f'car_pose_cfg thieu {k}')
    body = {k: v for k, v in doan.items() if k != 'sha256'}
    if doan.get('sha256') != hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest():
        p.append('sha256 noi dung')
    return p


def _ids_sha(ids):
    return hashlib.sha256(','.join(str(int(i)) for i in ids).encode()).hexdigest()


def commit_hil():
    """'<HEAD sha>[+dirty]' of the repository holding this tool, or None."""
    import subprocess
    try:
        h = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        d = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', '.'], capture_output=True, text=True,
                           timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None if h.returncode else h.stdout.strip() + ('+dirty' if d.stdout.strip() else '')


def dets_fingerprint(path):
    """frames_fingerprint(load_dets(path)[0]) of c7_measure (CAM; contract 2.D) or None when that version is not here."""
    if not path:
        return None
    try:
        import c7_measure as cm
        return cm.frames_fingerprint(cm.load_dets(path)[0])
    except (ImportError, AttributeError):
        return None


def stop_positions(segs):
    """Commanded position along the car axis of every stop window (window k = the stop before move k)."""
    x, out = 0.0, [0.0]
    for m in plan_moves(segs):
        if m['loai'] == 'tien':
            x = m['toi']
        out.append(x)
    return out


def report(ck):
    for name, ok, detail in ck:
        print(f"{'DAT ' if ok else 'HONG'} {name:24s} {detail}")


def cmd_kiem(a):
    A = load_attempt(a.dir)
    ck, info = kiem(A, a.age_r_max_ms)
    report(ck)
    print(f"cat_pwm={info['cat_pwm']}")
    return 0 if all(ok for _, ok, _ in ck) else 1


def frames_of(a, A):
    F = None
    if a.a6 and a.xoay:
        F = frames_chain(a.frames, a.a6, a.xoay, track=a.track,
                         cho=A['cho'] if A else getattr(a, 'cho', None))     # HCC7-6, CT2-10
    return F or frames_basic(a.frames, a.track)


def window_roles(segs):
    """(window index, loai of F2) in order: ref_12b, six dung_12b, then E_dau (the stop just before the first turn,
    also the last dung_12b when run together), E_cuoi; the stop between the two turns is not an F2 row."""
    moves = plan_moves(segs)
    roles, k = [], 0                       # window k = the stop before move k (k = len(moves): the last stop)
    n_t = sum(1 for m in moves if m['loai'] == 'tien')
    if n_t:
        roles.append((0, 'ref_12b'))
        roles += [(i, 'dung_12b') for i in range(1, n_t + 1)]
    turns = [i for i, m in enumerate(moves) if m['loai'] == 'xoay']
    if turns:
        roles.append((turns[0], 'E_dau'))
        roles.append((len(moves), 'E_cuoi'))
    return roles


def cmd_cua_so(a):
    A = load_attempt(a.dir)
    ck, info = kiem(A)
    if 'windows' not in info:
        raise AError('kiem: no windows (log checks failed before the windows)')
    F = frames_of(a, A)
    an = analyse(A, F, info)
    clip_sha = sha256(a.clip) if a.clip else None
    tj0 = json.load(open(a.track, encoding='utf-8'))
    for k_t, v in (('video_sha256', clip_sha), ('session_sha256', sha256(a.session) if a.session else None),
                   ('frames_sha256', sha256(a.frames))):
        if tj0.get(k_t) and v and tj0[k_t] != v:          # HCC2-5 (round 3): the track was made from another file
            raise AError(f'{a.track}: {k_t} khac file dua vao cua_so')
    S = json.load(open(a.session, encoding='utf-8')) if a.session else {}
    ses_sha = sha256(a.session) if a.session else a.session_sha256
    # F8 (T9b, CODE 20/11): lan_lap_id of this 12b from NHACTAM lan_lap_v1 run 1; read before any file is written
    ll_id, ll_why, ll_sha = None, 'chua co --lan-lap (F8 lan_lap_v1 cua NHACTAM)', None
    lan_lap, ra_dir = getattr(a, 'lan_lap', None), getattr(a, 'ra', None)
    if lan_lap:
        ll_id, ll_why, ll_sha = read_lan_lap(lan_lap, A['thu_id'], A['cho'], a.a6, clip_sha, sha256(a.frames))
    ra = ra_dir or a.dir                     # run 2 (with --a6 --xoay --lan-lap) writes to a new folder (mode x)
    if ra_dir:
        os.makedirs(ra, exist_ok=True)
    pos = stop_positions(A['segs'])
    rows = []
    roles = window_roles(A['segs'])
    last_k = len(an['windows']) - 1
    for k_w, lo in roles:
        w = an['windows'][k_w]
        ids_all = F['idx'][w['mask_all']]
        ts_all = F['t'][w['mask_all']]
        ids_ok = F['idx'][w['mask']]
        cuoi = k_w == last_k
        rows.append(dict(schema='cua_so_v1', luot_id=A['thu_id'], thu_id=A['thu_id'], cho=A['cho'], loai=lo,
                         chi_so=w['chi_so'], clip=a.clip and os.path.basename(a.clip), clip_sha256=clip_sha,
                         session_sha256=ses_sha, fit_ids_nguon=S.get('fit_ids'), frames_sha256=sha256(a.frames),
                         khung=[dict(id=int(i), pts_s=float(t)) for i, t in zip(ids_all, ts_all)],
                         khung_sha256=_ids_sha(ids_all), khung_ma_hop_le=[int(i) for i in ids_ok],
                         khung_ma_hop_le_sha256=_ids_sha(ids_ok), phep_gop='trung_binh_go_vong',
                         t_canh_s=None if (cuoi or w['tb'] is None) else w['tb'] + 0.1 - an['D'],
                         bien_cuoi='duoi_log' if cuoi else 'canh_lenh', vi_tri_lenh_m=pos[k_w],
                         D_s=an['D'], u_D_s=an['u_D'], t0_log_us=int(A['log']['timestamp_us'][0]),
                         esp32_sha256=sha256(A['log_path']), chuoi=an['chuoi']))
    for r in rows:
        pr = f2_problems(r)
        if pr:
            raise AError(f"F2 {r['loai']} {r['chi_so']}: {pr}")
    out = os.path.join(ra, 'cua_so_v1.jsonl')
    with open(out, 'x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tj = json.load(open(a.track, encoding='utf-8'))
    xo = json.load(open(a.xoay, encoding='utf-8')) if a.xoay else {}
    xb = (xo.get('cho', {}).get(A['cho']) or xo) if xo else {}
    role_of = {k: lo for k, lo in roles if lo in ('ref_12b', 'dung_12b')}     # HCC2-7: not the E_dau alias
    nguon = [dict(vai_tro=k, duong_dan=os.path.abspath(v), sha256=sha256(v))
             for k, v in (('clip', a.clip), ('session', a.session), ('frames', a.frames), ('track', a.track),
                          ('dets', a.dets), ('log', A['log_path']), ('sender', os.path.join(a.dir, 'sender.txt')),
                          ('lich', os.path.join(a.dir, 'lich.csv')), ('cua_so_v1', out), ('a6', a.a6),
                          ('xoay', a.xoay), ('lan_lap', lan_lap)) if v]
    # HCC2-7: 'khoa' needs every source file AND the keys made from them (dets_fp; lan_lap_id of NHACTAM from F8)
    src_ok = (all((a.clip, a.session, a.dets, a.a6, a.xoay)) and dets_fingerprint(a.dets) is not None
              and ll_id is not None)
    doan = dict(schema='c8_hc_doan_12b_v1', thu_id=A['thu_id'], cho=A['cho'], lan_lap_id=ll_id,
                lan_lap_v1_sha256=ll_sha, lan_lap_id_ly_do=ll_why,
                trang_thai=an['chuoi'] if src_ok else 'chan_doan', clip=a.clip, clip_sha256=clip_sha, session_sha256=ses_sha,
                frames_sha256=sha256(a.frames), track_sha256=sha256(a.track), cua_so_v1_sha256=sha256(out),
                esp32_sha256=sha256(A['log_path']), commit_hil=commit_hil(),
                car_pose_cfg=dict(side_m=S.get('car_side_m'), h_m=S.get('tag_h_m'), tag_offset=tj.get('tag_offset'),
                                  q_m=xb.get('q_tag_m'), du_m=xb.get('du_m')),
                xoay_sha256=sha256(a.xoay) if a.xoay else None, a6_sha256=sha256(a.a6) if a.a6 else None,
                chuoi=an['chuoi'], dets_sha256=sha256(a.dets) if a.dets else None, dets_fp=dets_fingerprint(a.dets),
                nguon=nguon, doan=[])
    for lg in an['legs']:
        d = dict(i=lg['i'], dai=lg['dai'], trang_thai=lg['trang_thai'], ly_do=lg.get('ly_do'),
                 chieu=lg.get('chieu') or ('toi' if pos[lg['b']] > pos[lg['a']] else 'lui'),
                 vi_tri_lenh_m=[pos[lg['a']], pos[lg['b']]], dp_m=None, L_m=None)
        for nm, k_w in (('cua_so_a', lg['a']), ('cua_so_b', lg['b'])):
            w = an['windows'][k_w]
            ids = F['idx'][w['mask']]
            d[nm] = dict(loai=role_of.get(k_w), chi_so=w['chi_so'], khung_id=[int(i) for i in ids],
                         khung_pts_s=[float(t) for t in F['t'][w['mask']]], khung_sha256=_ids_sha(ids),
                         p_m=w['p'].tolist() if 'p' in w else None, psi_rad=w.get('psi'))
        if lg['trang_thai'] == 'hop_le':
            d.update(dp_m=lg['dp'].tolist(), L_m=lg['L_m'])
        doan['doan'].append(d)
    body = json.dumps(doan, ensure_ascii=False, sort_keys=True)
    doan['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    pr = f5_problems(doan)
    if pr:
        raise AError(f'F5: {pr}')
    with open(os.path.join(ra, 'doan_12b_v1.json'), 'x', encoding='utf-8') as f:
        json.dump(doan, f, ensure_ascii=False, indent=1)
    print(f"D={an['D']:.3f} s u_D={an['u_D']:.3f} s chuoi={an['chuoi']}")
    for lg in an['legs']:
        if lg['trang_thai'] == 'hop_le':
            print(f"doan {lg['i']} {'dai ' if lg['dai'] else 'ngan'} {lg['chieu']:3s} L={lg['L_m']:.3f} m "
                  f"a_l={math.degrees(lg['a_l']):+.3f} deg ti_le={lg['ti_le']:.4f} eps="
                  + ('thieu' if not math.isfinite(lg['eps']) else f"{math.degrees(lg['eps']):.3f} deg"))
        else:
            print(f"doan {lg['i']} {lg['trang_thai']}: {lg.get('ly_do')}")
    return 0


def _block(j, cho):
    """The block of one place in a file of another session: j['cho'][cho] when present, else j."""
    return (j.get('cho') or {}).get(cho) or j


_KHAC = object()


def _meta(m, k):
    """A key of the window meta of NHACTAM (its internal name, e.g. clip_sha) or the contract name (clip_sha256);
    HCV3-3: both present and different -> a value that matches nothing (the window is refused)."""
    if k in m and k + '256' in m and m[k] != m[k + '256']:
        return _KHAC
    return m.get(k, m.get(k + '256'))


def read_neo(path, cho, ref_row, want_ids=False):
    """psi_chuan_k_deg of F6 (NHACTAM hieu_chuan_neo_v1) after checking it was made on the ref_12b window of the
    issued attempt (contract F6; HCC1-10; NHACTAM 19/11 via CODE; Codex HCG1-1..6). khung_chung of NHACTAM is
    (frames through the anchor gate) & khung_ma_hop_le & (A6 valid), so it equals neither khung_sha256 nor
    khung_ma_hop_le_sha256 of the window. The check: the file trang_thai is 'dat'; the standard of THIS run is
    chuan_12b['<cho>|<luot_id of the ref_12b row>'] (or cho[cho] when it names that run), 'dat', with cho and
    luot_id of the run; its clip_sha256 and frames_sha256 equal those of the ref_12b row; exactly one ref_12b window
    in cua_so[*] with meta cho, luot, clip_sha, frames_sha of the run carries the same khung_chung_sha256; its
    khung_chung.id are distinct integers inside khung_ma_hop_le of the ref_12b row; sha256 of the sorted IDs joined
    by commas ("1,2,...") = khung_chung_sha256; psi finite. Anything else, malformed files included: AError (code 2)."""
    if ref_row is None:
        raise AError('F6 needs the ref_12b row of cua_so_v1.jsonl of the issued attempt (run cua_so first)')
    try:
        v, ids = _read_neo(path, cho, ref_row)
        return (v, ids) if want_ids else v
    except AError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError, KeyError, RecursionError) as exc:  # HCG1-6
        raise AError(f'{path}: F6 sai dang ({type(exc).__name__}: {exc})')


def _read_neo(path, cho, ref_row):
    j = json.load(open(path, encoding='utf-8'))
    if not isinstance(j, dict) or j.get('trang_thai') != 'dat':                          # HCG1-4
        raise AError(f"{path}: trang_thai toan file {j.get('trang_thai') if isinstance(j, dict) else None!r}")
    luot = ref_row.get('luot_id')
    if not luot:
        raise AError('ref_12b row without luot_id')
    b = (j.get('chuan_12b') or {}).get(f'{cho}|{luot}')                                    # HCG1-1
    if b is None:
        c = (j.get('cho') or {}).get(cho)
        b = c if isinstance(c, dict) and c.get('luot_id') == luot else None
    if not isinstance(b, dict):
        raise AError(f'{path}: khong co ban ghi chuan cua {cho}|{luot}')
    if b.get('trang_thai') != 'dat':
        raise AError(f"{path}: {cho}|{luot} trang_thai {b.get('trang_thai')!r} ({b.get('ly_do', '')})")
    if b.get('cho') != cho or b.get('luot_id') != luot:                                     # HCG1-2
        raise AError(f"{path}: ban ghi chuan cua {b.get('cho')}|{b.get('luot_id')}, can {cho}|{luot}")
    c = (j.get('cho') or {}).get(cho)
    n_cho = sum(1 for k in (j.get('chuan_12b') or {}) if str(k).split('|', 1)[0] == cho)
    keys = ('trang_thai', 'cho', 'luot_id', 'clip_sha256', 'frames_sha256', 'khung_chung_sha256', 'psi_chuan_k_deg')
    if n_cho == 1 and c is not b:
        # HCG2-1, HCV3-3: one 12b run at the place: the writer puts the same record in cho[cho] and chuan_12b; both
        # must be there and agree on every key (a block without them is not that record)
        if not isinstance(c, dict) or any(c.get(k) != b.get(k) for k in keys):
            raise AError(f'{path}: cho[{cho}] va chuan_12b[{cho}|{luot}] khac nhau')
    elif n_cho > 1 and isinstance(c, dict) and (c.get('trang_thai') == 'dat' or 'psi_chuan_k_deg' in c):
        # several runs: the writer leaves an aggregate block (chua_ket_luan, no psi); a record there contradicts it
        raise AError(f'{path}: cho[{cho}] la ban ghi trong khi cho co {n_cho} luot 12b')
    for k in ('psi_chuan_k_deg', 'clip_sha256', 'khung_chung_sha256', 'frames_sha256'):
        if b.get(k) is None or b.get(k) == '':
            raise AError(f'{path}: {cho}|{luot} thieu {k}')
    if b['clip_sha256'] != ref_row.get('clip_sha256'):
        raise AError(f'{path}: clip_sha256 khac cua so ref_12b')
    if b['frames_sha256'] != ref_row.get('frames_sha256'):
        raise AError(f'{path}: frames_sha256 khac cua so ref_12b')
    cs = j.get('cua_so')
    if not isinstance(cs, list) or not all(isinstance(c, dict) and isinstance(c.get('meta'), dict) for c in cs):
        raise AError(f'{path}: cua_so phai la danh sach cua so co meta')                  # HCG2-3
    # HCG2-2: the ref_12b window of this place and run must be unique BEFORE its sources are compared
    wins = [c for c in cs if c['meta'].get('loai') == 'ref_12b' and c['meta'].get('cho') == cho
            and c['meta'].get('luot') == luot]                                                # HCG1-2
    if len(wins) != 1:
        raise AError(f'{path}: {len(wins)} cua so ref_12b cua {cho}|{luot} (can dung 1)')
    w0 = wins[0]
    if _meta(w0['meta'], 'clip_sha') != ref_row.get('clip_sha256') \
            or _meta(w0['meta'], 'frames_sha') != ref_row.get('frames_sha256') \
            or w0.get('khung_chung_sha256') != b['khung_chung_sha256']:                       # HCG1-3
        raise AError(f'{path}: cua so ref_12b cua {cho}|{luot}: clip, frames hay khung_chung_sha256 khac')
    ids = (wins[0].get('khung_chung') or {}).get('id')
    if (not isinstance(ids, list) or not ids
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids)       # HCG1-5
            or len(set(ids)) != len(ids)):
        raise AError(f'{path}: khung_chung.id phai la danh sach so nguyen phan biet, khong rong')
    ok_ids = ref_row.get('khung_ma_hop_le') or []
    if not set(ids) <= {int(i) for i in ok_ids}:
        raise AError(f'{path}: khung_chung khong nam trong khung_ma_hop_le cua ref_12b')
    if hashlib.sha256(','.join(str(i) for i in sorted(ids)).encode()).hexdigest() != b['khung_chung_sha256']:
        raise AError(f'{path}: sha256 cua khung_chung.id khac khung_chung_sha256')
    v = float(b['psi_chuan_k_deg'])
    if not math.isfinite(v):
        raise AError(f'{path}: psi_chuan_k_deg khong huu han')
    return v, sorted(ids)


def _f2_rows(path):
    """Window rows of a cua_so_v1 (F2) file: JSON lines, a JSON list, or {schema, cua_so|dong|windows: [...]}."""
    txt = open(path, encoding='utf-8').read()
    try:
        o = json.loads(txt)
    except ValueError:
        o = [json.loads(s) for s in txt.splitlines() if s.strip()]
    if isinstance(o, dict):
        o = next((o[k] for k in ('cua_so', 'dong', 'windows') if isinstance(o.get(k), list)), None)
    if not isinstance(o, list) or not all(isinstance(r, dict) for r in o):
        raise AError(f'{path}: khong doc duoc danh sach cua so F2')
    return o


def read_lan_lap(path, thu_id, cho, a6_path, clip_sha, frames_sha):
    """(lan_lap_id or None, reason or None, sha256 of the file) of a 12b attempt from F8 lan_lap_v1 of NHACTAM (run 1
    of lan_lap, without --f15 / --truc-xe; CODE 20/11, sequence T9, T9b). F8 has one luot[] record per car run
    (windows pre/post), none for a 12b attempt: the placement of the 12b is the lan_lap_id that F8 gives to the runs of
    the same place it verified against this 12b as standard (luot[].chuan_12b = thu_id; a record with luot_id = thu_id
    counts too). Exactly one value -> that id; none, or several -> None with the reason (F5 stays chan_doan).
    A file that is not this session's F8 is refused (AError, code 2): schema lan_lap_v1; tam_id of the file and of every
    record = c8b_a6_v2.tam_id of --a6; a6.sha256 = the --a6 file; its F2 file present with the sha256 it names and
    holding the ref_12b window of this attempt (same luot_id, cho, clip_sha256, frames_sha256)."""
    tam = a6_tam_id(a6_path)
    if tam is None:
        raise AError('--lan-lap can --a6 co tam_id (c8b_a6_v2.tam_id)')
    j = json.load(open(path, encoding='utf-8'))
    if not isinstance(j, dict) or j.get('schema') != 'lan_lap_v1':
        raise AError(f'{path}: khong phai lan_lap_v1')
    if j.get('tam_id') != tam:
        raise AError(f"{path}: tam_id {j.get('tam_id')!r} khac tam_id cua A6 {tam!r}")
    if (j.get('a6') or {}).get('sha256') != sha256(a6_path):
        raise AError(f'{path}: a6.sha256 khac file --a6')
    f2, f2_sha = j.get('cua_so_v1_file'), j.get('cua_so_v1_sha256')
    if not f2 or not os.path.isfile(f2) or sha256(f2) != f2_sha:
        raise AError(f'{path}: file F2 cua F8 ({f2}) khong co hay sha256 khac cua_so_v1_sha256')
    if not any(r.get('luot_id') == thu_id and r.get('cho') == cho and r.get('loai') == 'ref_12b'
               and r.get('clip_sha256') == clip_sha and r.get('frames_sha256') == frames_sha for r in _f2_rows(f2)):
        raise AError(f'{path}: F2 cua F8 khong co cua so ref_12b cua {thu_id} ({cho}) voi cung clip, frames')
    luot = j.get('luot')
    if not isinstance(luot, list) or not all(isinstance(r, dict) for r in luot):
        raise AError(f'{path}: luot phai la danh sach ban ghi')
    if any(r.get('tam_id') != tam for r in luot):
        raise AError(f'{path}: co luot cua tam khac {tam!r}')
    ids = {r['lan_lap_id'] for r in luot if r.get('cho') == cho and thu_id in (r.get('chuan_12b'), r.get('luot_id'))
           and isinstance(r.get('lan_lap_id'), str) and r['lan_lap_id']}
    f_sha = sha256(path)
    if len(ids) == 1:
        return ids.pop(), None, f_sha
    if not ids:
        return None, f'F8 khong co luot nao cua {cho} xac minh lan lap voi 12b {thu_id} (chuan_12b)', f_sha
    return None, f'F8 cho 12b {thu_id} nhieu lan_lap_id {sorted(ids)}', f_sha


def read_e_hieu(path, doan_path):
    """{leg i: e_hieu_m} of F9 (KIEMCHEO) after checking doan_12b_sha256 against the F5 of the issued attempt (its
    content sha256 or the file sha256; HCC1-10); else AError (code 2)."""
    j = json.load(open(path, encoding='utf-8'))
    d = json.load(open(doan_path, encoding='utf-8'))
    body = {k: v for k, v in d.items() if k != 'sha256'}
    if d.get('sha256') != hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest():
        raise AError(f'{doan_path}: sha256 noi dung khong khop (file da bi sua)')       # HCC2-5
    if j.get('doan_12b_sha256') not in (d.get('sha256'), sha256(doan_path)):
        raise AError(f'{path}: doan_12b_sha256 khac {doan_path}')
    st = j.get('trang_thai')
    if st is not None and st != 'dat':          # KCJ1-7: a diagnostic F9 never unlocks u_map
        raise AError(f'{path}: F9 trang_thai {st!r} (chi nhan dat)')
    out = {}
    for x in j.get('doan', []):
        if 'e_hieu_m_chan_doan' in x:
            raise AError(f'{path}: F9 chi co e_hieu_m_chan_doan (chan doan, khong mo khoa u_map)')
        if 'i' not in x or 'e_hieu_m' not in x:
            raise AError(f'{path}: dong thieu i / e_hieu_m')
        out[int(x['i'])] = float(x['e_hieu_m'])
    return out, bool(j.get('bao_q_xoay', False))


def _phan_ok(r):
    """HCH2-5: the parts of a C8B bound are finite, >= 0 and add up to u_rad, for the two ways C8B combines them:
    cong_rieng (A3 and A6 residual at both ends) and replay_chung_theo_ca (worst H32 case + A6 residual without
    U_meo_E at both ends, design 2b). Any other form: not accepted."""
    ph = r.get('phan') if isinstance(r.get('phan'), dict) else {}
    g = r.get('ghep')
    if g == 'cong_rieng':
        parts = list(ph.get('A3_rad') or []) + list(ph.get('A6du_rad') or [])
        n = 4
    elif g == 'replay_chung_theo_ca':
        parts = [ph.get('ca_rad')] + list(ph.get('A6du_khong_meo_rad') or [])
        n = 3
    else:
        return False
    if len(parts) != n or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                  and v >= 0 for v in parts):
        return False
    u = float(r['u_rad'])
    return abs(sum(parts) - u) <= 1e-12 * max(1.0, abs(u))


def can_hai_dau_c8b(a6_path, kc_path, tam_id, nguon):
    """(p1, p2) -> (u [rad] or None, reasons) from c8b_a6.u_psi_hai_dau_rad (C8B hil feat/c8b-3b-ga 2539ee8; CODE
    20/11 local_c29a2de2): kc = KIEMCHEO kiem_cheo.json (c7_kiem_cheo_v7, --xuat-phat); C8B checks A3 INSIDE the
    function (version, quy_ve_ma_noc, ma_ao, du_lieu_du, plumb, per place mot_phuong / ngoai_suy / cap_hong) and
    returns thieu with reasons, so there is no A3 gate here (CODE 20/11; replaces HCH1-5, HCH2-2). p1, p2 = tag
    centres on the floor (car_pose frame, m). luot, dai = None: the largest U_chuyen.dong over runs and bands
    (the 12b windows are still windows of calibration attempts, not C8 target runs; [CACH HIEU], conservative).
    Needs --a6 and --kc together, non-empty, and an A6 file with tam_id (else AError, code 2); tam_id = c8b_a6_v2.tam_id
    (HCV3-1, CODE 20/11), --tam-id optional and only compared to it. c8b_a6 not merged yet, the A6 file refused,
    an error inside the function or parts that do not add up: every call gives None with the reason (the result stays
    chan_doan, the term null)."""
    if not (a6_path and kc_path and tam_id):
        raise AError('--kc can di cung --a6 co tam_id (ham can hai dau cua C8B), khong rong')
    kc = json.load(open(kc_path, encoding='utf-8'))
    nguon.append(dict(vai_tro='kiem_cheo', duong_dan=os.path.abspath(kc_path), sha256=sha256(kc_path)))
    try:
        import c8b_a6 as c8a                                     # CODE_C8B, after the common merge
        a6o = c8a.load_a6(a6_path)
    except ImportError:
        return lambda p1, p2: (None, ['c8b_a6 chua gop'])
    except Exception as exc:                                     # noqa: BLE001  A6Error / bad file: no bound
        why = f'A6 khong doc duoc ({type(exc).__name__}: {exc})'
        return lambda p1, p2: (None, [why])
    nguon.append(dict(vai_tro='c8b_a6_ham', duong_dan=os.path.abspath(c8a.__file__), sha256=sha256(c8a.__file__)))

    def fn(p1, p2):
        try:
            r = c8a.u_psi_hai_dau_rad(a6o, kc, [float(v) for v in p1], [float(v) for v in p2], tam_id,
                                      luot=None, dai=None)
        except Exception as exc:                                 # noqa: BLE001  HCH1-6: a bad A6 file is no bound
            return None, [f'ham C8B loi ({type(exc).__name__}: {exc})']
        if not isinstance(r, dict) or r.get('trang_thai') != 'dat' or r.get('u_rad') is None:
            return None, [str(w) for w in ((r or {}).get('ly_do') or ['trang_thai khac dat'])]
        if not _phan_ok(r):
            return None, [f"thanh phan can C8B khong hop le ({r.get('ghep')}): {r.get('phan')}"]
        return float(r['u_rad']), []
    return fn


def read_can_c8b(path, cho):
    """(u_psi_cuc [rad], u_psi_hai_dau [rad] or None, agreed) from a C8B file (names not yet agreed, contract 4.5):
    u_psi_cuc_rad (camera heading, leg start vs reference window), u_psi_hai_dau_rad (error of the DIFFERENCE of the
    two end headings of a leg, for the q turn term of u_q; HCC2-8)."""
    b = _block(json.load(open(path, encoding='utf-8')), cho)
    if 'u_psi_cuc_rad' not in b:
        raise AError(f'{path}: {cho} thieu u_psi_cuc_rad')
    hd = b.get('u_psi_hai_dau_rad')
    vals = [float(b['u_psi_cuc_rad'])] + ([] if hd is None else [float(hd)])
    if not all(math.isfinite(v) and 0.0 <= v <= math.pi for v in vals):   # HCC3-6
        raise AError(f'{path}: can C8B ngoai mien [0, pi]: {vals}')
    return vals[0], None if hd is None else vals[1], bool(b.get('chot', False))


def check_cap_sources(d, frames, track, log_path, a6, xoay, need):
    """HCC2-5: the F2 rows and the F5 of the issued attempt must have been made from the very frames, track and log
    tinh analyses now (and the same A6, xoay files); need=True (F6/F9 given) and a file missing -> AError."""
    p2, p5 = os.path.join(d, 'cua_so_v1.jsonl'), os.path.join(d, 'doan_12b_v1.json')
    if not need:                                          # HCC3-4: diagnosis from the current sources only
        return
    if not (os.path.isfile(p2) and os.path.isfile(p5)):
        raise AError(f'{d}: F6/F9 need cua_so_v1.jsonl and doan_12b_v1.json of this attempt (run cua_so)')
    fs, ls = sha256(frames), sha256(log_path)
    for r in (json.loads(x) for x in open(p2, encoding='utf-8')):
        if r.get('frames_sha256') != fs or r.get('esp32_sha256') != ls:
            raise AError(f'{p2}: {r.get("loai")} lam tu frames/log khac file dang tinh')
    d5 = json.load(open(p5, encoding='utf-8'))
    pr = f5_problems(d5)
    if pr:
        raise AError(f'{p5}: {pr}')
    want = dict(frames_sha256=fs, track_sha256=sha256(track), esp32_sha256=ls, cua_so_v1_sha256=sha256(p2),
                a6_sha256=sha256(a6) if a6 else None, xoay_sha256=sha256(xoay) if xoay else None)
    bad = [k for k, v in want.items() if d5.get(k) != v]
    if bad:
        raise AError(f'{p5}: {bad} khac nguon dang tinh')


def cmd_tinh(a):
    attempts, cap_ok, cap_why, cap_ref, cap_F = [], True, [], None, None
    cap_id = os.path.basename(os.path.normpath(a.cap))
    nguon = []
    for d in [a.cap] + list(a.them or []):
        A = load_attempt(d)
        if A['cho'] != a.cho:
            raise AError(f'{d}: cho {A["cho"]} != {a.cho}')
        fr = a.frames_of.get(d) if a.frames_of else None
        if not fr:
            raise AError(f'{d}: --frames-of {d}=FRAMES,TRACK missing')
        frames, track = fr.split(',')
        ck, info = kiem(A)
        failed = [n for n, ok, _ in ck if not ok]
        is_cap = A['thu_id'] == cap_id
        if is_cap and failed:
            cap_ok, cap_why = False, failed
        nguon += [dict(vai_tro=f'{A["thu_id"]}:{k}', duong_dan=os.path.abspath(v), sha256=sha256(v))
                  for k, v in (('log', A['log_path']), ('sender', os.path.join(d, 'sender.txt')),
                               ('lich', os.path.join(d, 'lich.csv')), ('frames', frames), ('track', track))]
        if 'windows' not in info:
            attempts.append((A['thu_id'], None, failed or ['khong co cua so']))
            continue
        F = None
        if a.a6 and a.xoay:
            F = frames_chain(frames, a.a6, a.xoay, track=track, cho=A['cho'])          # HCC7-6, CT2-10
        F = F or frames_basic(frames, track)
        try:
            an = analyse(A, F, info)
        except AError as exc:
            attempts.append((A['thu_id'], None, failed + [str(exc)]))
            continue
        attempts.append((A['thu_id'], an, failed))
        if is_cap:
            check_cap_sources(d, frames, track, A['log_path'], a.a6, a.xoay, need=bool(a.neo or a.e_hieu))
            cap_F, cap_A, cap_an = F, A, an
            p = os.path.join(d, 'cua_so_v1.jsonl')
            if os.path.isfile(p):
                rows = [json.loads(x) for x in open(p, encoding='utf-8')]
                cap_ref = next((r for r in rows if r.get('loai') == 'ref_12b'), None)
                nguon.append(dict(vai_tro='cua_so_v1', duong_dan=os.path.abspath(p), sha256=sha256(p)))
    e = None
    bao_q = False
    ll_f5 = None
    if a.e_hieu:
        dp = a.doan or os.path.join(a.cap, 'doan_12b_v1.json')
        e, bao_q = read_e_hieu(a.e_hieu, dp)
        f5 = json.load(open(dp, encoding='utf-8'))
        if f5.get('thu_id') != cap_id or f5.get('cho') != a.cho:
            raise AError(f'{dp}: F5 cua {f5.get("thu_id")} ({f5.get("cho")}), khong phai luot cap {cap_id} ({a.cho})')
        ll_f5 = f5.get('lan_lap_id')
        nguon += [dict(vai_tro='e_hieu_m', duong_dan=os.path.abspath(a.e_hieu), sha256=sha256(a.e_hieu)),
                  dict(vai_tro='doan_12b_v1', duong_dan=os.path.abspath(dp), sha256=sha256(dp))]
    q = du = None
    if a.xoay:
        x = _block(json.load(open(a.xoay, encoding='utf-8')), a.cho)
        q, du = x.get('q_tag_m'), x.get('du_m')
        nguon.append(dict(vai_tro='xoay_v1', duong_dan=os.path.abspath(a.xoay), sha256=sha256(a.xoay)))
    if a.a6:
        nguon.append(dict(vai_tro='c8b_a6_v2', duong_dan=os.path.abspath(a.a6), sha256=sha256(a.a6)))
    neo = None
    p_ref_tag = None
    if a.neo:
        neo, neo_ids = read_neo(a.neo, a.cho, cap_ref, want_ids=True)
        p_ref_tag = 'thieu'                                # HCV3-4: no fallback to window 0 once F6 is given
        if cap_F is not None:                              # HCH2-4: tag centre over the F6 common frames
            idx = np.asarray(cap_F['idx'])
            tx, ty = cap_F.get('tx', cap_F['x']), cap_F.get('ty', cap_F['y'])
            pos = [np.flatnonzero(idx == i) for i in neo_ids]
            if all(len(k) == 1 and np.isfinite(tx[k[0]]) and np.isfinite(ty[k[0]]) for k in pos):
                sel = np.array([k[0] for k in pos])
                p_ref_tag = np.array([tx[sel].mean(), ty[sel].mean()])
        nguon.append(dict(vai_tro='hieu_chuan_neo', duong_dan=os.path.abspath(a.neo), sha256=sha256(a.neo)))
    psi_cuc, psi_hd, chot = None, None, False
    if a.can_c8b:
        psi_cuc, psi_hd, chot = read_can_c8b(a.can_c8b, a.cho)
        nguon.append(dict(vai_tro='can_c8b', duong_dan=os.path.abspath(a.can_c8b), sha256=sha256(a.can_c8b)))
    can_fn = None
    if a.kc is not None or a.tam_id is not None:                 # HCH1-8: an empty value is not 'absent'
        tam = a6_tam_id(a.a6)                                    # HCV3-1: the plate is c8b_a6_v2.tam_id
        if a.tam_id is not None and a.tam_id != tam:
            raise AError(f'--tam-id {a.tam_id!r} khac tam_id cua A6 {tam!r} (tam_id khai mot lan trong c8b_a6_v2)')
        can_fn = can_hai_dau_c8b(a.a6, a.kc, tam, nguon)
    vet = None
    if a.moc and cap_F is not None:
        import c8_hc_vet as cv
        tags = cv.load_tags(a.moc, a.vien_mm / 1000)
        L = cap_A['log']
        i0, i1 = run_range(L)
        D = cap_an['D']
        m = cap_F['ok'] & (cap_F['t'] + D >= L['t'][i0]) & (cap_F['t'] + D <= L['t'][i1])
        a_cap = circmean([lg['a_l'] for lg in cap_an['legs'] if lg['dai'] and lg['trang_thai'] == 'hop_le'] or [0.0])
        x0, y0, _ = cv.START[a.cho]
        rim = np.abs(np.diff(L['C'][i0:i1 + 2], axis=0)) * COUNT_M / np.maximum(np.diff(L['t'][i0:i1 + 2]), 1e-3)[:, None]
        v_b = VET_MARGIN * math.sqrt(2) * float(np.max(rim))     # mecanum: |vx|, |vy| <= max wheel rim speed
        w_b = VET_MARGIN * float(np.max(rim)) / LXY              # |wz| <= max rim speed / (lx + ly)
        vet = cv.vet_thuc_camera(cap_F['t'][m], cap_F['x'][m], cap_F['y'][m], cap_F['psi'][m], a_cap, tags,
                                 a.d_vet_mm / 1000, v_bound=v_b, w_bound=w_b,
                                 nguon_can=f'encoder luot cap x {VET_MARGIN} (khong truot)')
        nguon.append(dict(vai_tro='moc_32', duong_dan=os.path.abspath(a.moc), sha256=sha256(a.moc)))
    roles = dict((lo, k) for k, lo in window_roles(load_attempt(a.cap)['segs']))
    need = sorted(k for k, lo in window_roles(load_attempt(a.cap)['segs']) if lo in ('ref_12b', 'dung_12b'))
    del roles
    r = place_result(a.cho, attempts, cap_id, e_hieu=e, q=q, du_m=du, neo=neo, psi_cuc=psi_cuc, cap_ok=cap_ok,
                     cap_why=cap_why, bao_q=bao_q, vet=vet, psi_cuc_chot=chot, need_windows=need, psi_hai_dau=psi_hd,
                     can_hai_dau=can_fn, p_ref_tag=p_ref_tag)
    r = dict(schema='truc_xe_v1', version=VERSION, commit_hil=commit_hil(), nguon=nguon,
             mien_hieu_luc=dict(cho=a.cho, thu_id_cap=cap_id, lan_lap_id=ll_f5,
                                ghi_chu='cung buoi, cung lan lap tam' + (
                                    ' (lan_lap_id cua F5, tu F8 lan_lap_v1 luot 1 cua NHACTAM)' if ll_f5 else
                                    ' (khong co lan_lap_id: F5 chua noi F8, chua xac minh)')),
             **r)
    txt = json.dumps(r, ensure_ascii=False, indent=1, default=float)
    if a.out:
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(txt)
    if r['a_k_deg'] is None:
        print(f"{a.cho}: trang_thai {r['trang_thai']} {r['ly_do']}")
    else:
        ua = (f"{r['u_a_k_deg']:.3f}" if r['u_a_k_deg'] is not None
              else 'null (' + ','.join(r.get('u_hang_thieu') or []) + ')')      # HCH2-1
        print(f"{a.cho}: a = {r['a_k_deg']:+.3f} deg, u_a = {ua} deg, do tan {r['do_tan_deg']:.3f} deg, "
              f"trang_thai {r['trang_thai']} {r['ly_do']}")
    return 0 if r['trang_thai'] in ('khoa', 'chan_doan') and all(r['cong'].values()) else 1


def video_checks(A, F, info):
    """Floor decision on the whole-clip read (design 2.7; HCC1-7): phases from the attempt manifest (lich.csv of
    '#seq_start'), alignment, >= MIN_FRAMES_VIDEO samples in every window that is used, 12b: four long legs,
    raw spread <= SPREAD_VIDEO_DEG, dynamic yaw with evidence; E: each turn direction >= E_COVER_TURNS useful
    turns and no tag gap > E_GAP_MAX_S. Returns (checks, analysis or None)."""
    ck = []

    def add(name, ok, detail=''):
        ck.append((name, bool(ok), str(detail)))
    moves = plan_moves(A['segs'])
    has12 = any(m['loai'] == 'tien' for m in moves)
    hasE = any(m['loai'] == 'xoay' for m in moves)
    try:
        an = analyse(A, F, info, min_frames=MIN_FRAMES_VIDEO)
    except AError as exc:
        add('giong_camera_log', False, exc)
        return ck, None
    add('giong_camera_log', an['u_D'] <= U_D_MAX_S, f"D {an['D']:.3f} s, u_D {an['u_D']:.3f} s")
    need = [k for k, _ in window_roles(A['segs'])]
    few = [(k, an['windows'][k]['n']) for k in need if an['windows'][k]['n'] < MIN_FRAMES_VIDEO]
    add('mau_moi_cua_so', not few, f'>= {MIN_FRAMES_VIDEO}; thieu {few}' if few else
        f"min {min(an['windows'][k]['n'] for k in need)}")
    if has12:
        longs = [lg for lg in an['legs'] if lg['dai'] and lg['trang_thai'] == 'hop_le']
        ok4 = len(longs) == 4
        spread = max(abs(wrap(x['a_l'] - y['a_l'])) for x in longs for y in longs) if ok4 else float('nan')
        add('bon_doan_dai', ok4, f'{len(longs)}/4')
        add('do_tan_tho', ok4 and spread <= math.radians(SPREAD_VIDEO_DEG), f'{math.degrees(spread):.3f} deg')
        eps = [lg['eps'] for lg in longs]
        f3 = lambda x: 'thieu' if not math.isfinite(x) else f'{math.degrees(x):.3f}'          # noqa: E731
        tc = an.get('truot_cheo') or {}
        add('yaw_dong', ok4 and all(math.isfinite(x) and x <= math.radians(EPS_YAW_MAX_DEG) for x in eps),
            ', '.join(f3(x) for x in eps) + ' (camera - IMU, neo hai dau doan; lech hai dau '
            + ', '.join(f3(lg['cam_imu']) for lg in longs) + ' deg)')
        add('neo_moc_san', an.get('neo_san', False), 'khung khop lai camera theo moc san' if an.get('neo_san')
            else 'khung KHONG neo moc san: huong camera co the troi')
        add('thang_hang', an['thang_hang_m']['bu'] <= PERP_MAX_M,
            f"{1e3 * an['thang_hang_m']['bu']:.2f} mm sau bu (tho {1e3 * an['thang_hang_m']['tho']:.2f})")
        add('truot_cheo', True, (f"sigma {100 * tc['sigma']:+.3f} %, do tan chua bu "
            f"{math.degrees(max(abs(wrap(x['a_l_tho'] - y['a_l_tho'])) for x in longs for y in longs)):.3f} deg"
            if ok4 and tc.get('sigma') is not None else f"khong bu: {tc.get('ly_do', 'thieu doan')}"))
    if hasE:
        cov = e_coverage(F, A['log'], info['blocks'], an['D'])
        add('phu_E', len(cov) == 2 and all(c['vong'] >= E_COVER_TURNS and c['ho_s'] <= E_GAP_MAX_S for c in cov),
            '; '.join(f"{c['vong']:.2f} vong, ho {c['ho_s']:.2f} s" for c in cov))
    return ck, an


def cmd_kiem_video(a):
    import c8_hc_video as hv                       # whole-clip reader (separate module, cv2)
    A = load_attempt(a.dir)
    ck, info = kiem(A)
    if 'windows' not in info:
        report(ck)
        raise AError('kiem: no windows')
    F = hv.sparse_frames(a.clip, a.session, stride=a.stride)
    vk, an = video_checks(A, F, info)
    bad_log = [c for c in ck if not c[1]]
    report(bad_log + vk)                                   # the log checks are part of the floor decision
    good = not bad_log and all(ok for _, ok, _ in vk)
    print('DAT' if good else 'LAM_LAI')
    return 0 if good else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('kiem')
    k.add_argument('dir')
    k.add_argument('--age-r-max-ms', type=float, default=125.0)
    v = sub.add_parser('kiem_video')
    v.add_argument('dir')
    v.add_argument('--clip', required=True)
    v.add_argument('--session', required=True)
    v.add_argument('--stride', type=int, default=VIDEO_STRIDE)
    c = sub.add_parser('cua_so')
    c.add_argument('dir')
    c.add_argument('--frames', required=True)
    c.add_argument('--track', required=True)
    c.add_argument('--clip', default=None)
    c.add_argument('--dets', default=None)
    c.add_argument('--session-sha256', default=None)
    c.add_argument('--a6', default=None)
    c.add_argument('--xoay', default=None)
    c.add_argument('--session', default=None, help='session JSON of c7 (fit_ids, car_side_m, tag_h_m, sha256)')
    c.add_argument('--lan-lap', default=None, help='lan_lap.json of NHACTAM (F8, run 1): lan_lap_id of this 12b; '
                                                   'needs --a6 with tam_id')
    c.add_argument('--ra', default=None, help='output folder (default DIR); run 2 with --lan-lap: a new folder')
    t = sub.add_parser('tinh')
    t.add_argument('--cho', required=True, choices=sorted(hs.PLACES))
    t.add_argument('--cap', required=True, help='attempt folder whose a is issued')
    t.add_argument('--them', nargs='*', help='other attempt folders of this place in the session (failed ones too)')
    t.add_argument('--frames-of', nargs='+', required=True, help='DIR=FRAMES_CSV,TRACK_JSON for each attempt')
    t.add_argument('--a6', default=None)
    t.add_argument('--xoay', default=None)
    t.add_argument('--neo', default=None, help='hieu_chuan_neo.json of NHACTAM (F6)')
    t.add_argument('--e-hieu', default=None, help='e_hieu_m of KIEMCHEO (F9)')
    t.add_argument('--doan', default=None, help='F5 doan_12b_v1.json the F9 was made from (default CAP/doan_12b_v1.json;'
                                                ' run 2 of cua_so writes it to its --ra folder)')
    t.add_argument('--can-c8b', default=None, help='C8B file with u_psi_cuc_rad (two-end camera heading bound)')
    t.add_argument('--kc', default=None, help='KIEMCHEO kiem_cheo.json (c7_kiem_cheo_v7) for c8b_a6.u_psi_hai_dau_rad')
    t.add_argument('--tam-id', default=None, help='optional check: must equal c8b_a6_v2.tam_id of --a6 (the plate '
                                                  'used for CAM and c8b_a6.u_psi_hai_dau_rad comes from the A6 file)')
    t.add_argument('--moc', default=None, help='moc_san_do_32.csv: real swept footprint (vet thuc)')
    t.add_argument('--vien-mm', type=float, default=10.0)
    t.add_argument('--d-vet-mm', type=float, default=20.0, help='tag centre to body box centre bound [de xuat]')
    t.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'tinh':
        a.frames_of = dict(s.split('=', 1) for s in a.frames_of)
    try:
        code = dict(kiem=cmd_kiem, kiem_video=cmd_kiem_video, cua_so=cmd_cua_so, tinh=cmd_tinh)[a.cmd](a)
    except (AError, OSError, ValueError, KeyError) as exc:
        print(f'loi: {exc}')
        print('#ket_qua,ma=2')
        return 2
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
