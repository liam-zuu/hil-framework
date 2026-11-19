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

  python tools/c8_hc_a.py kiem DIR [--age-r-max-ms 125]                    # at the floor, log only (~10 s)
  python tools/c8_hc_a.py kiem_video DIR --clip CLIP --session SESSION      # at the floor, sparse video (2-5 min)
  python tools/c8_hc_a.py cua_so DIR... --frames F --track T [--dets D]     # windows (F2) + legs (F5), per attempt
  python tools/c8_hc_a.py tinh --cho goc|K1|K2 --cap DIR [--them DIR...] --frames-of DIR=frames.csv ...
                          [--neo hieu_chuan_neo.json] [--e-hieu e_hieu.json] [--xoay xoay.json] [--out truc_xe_v1.json]

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
RV_REPEAT_W = 0.1              # rad/s: while the wheels turn faster than this an RV report must change the yaw
WIN_MIN_S, STILL_MIN_S = 2.5, 3.0
U_D_MAX_S = 0.10
TRAJ_AGE_MAX_MS = 60.0           # while the sender sends (50 Hz); not over the firmware timeout wait (HCC1-4)
EXPIRY_MS = 200.0               # firmware TRAJ_TIMEOUT_US (main.c 45): run_active ends when the newest packet is this old
EXPIRY_SLACK_S = 0.03
RV_STALL_S = 0.15               # 3 report periods at 20 Hz
E_CORR_MIN = 0.9
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
    """(j_s, problems). j_s = last row of the run with a fresh packet (0 <= traj_age_ms <= TRAJ_AGE_MAX_MS): up to
    there the sender was sending. Rows j_s+1..i1 are the firmware waiting for its 200 ms timeout after the last
    (zero) packet (main.c 524-525; HCC1-4): there no new packet may arrive (age only grows), the age stays under
    EXPIRY_MS, the wait is not longer than EXPIRY_MS (+ slack) and the command is 0 on every axis."""
    age = L['traj_age_ms']
    fresh = np.flatnonzero((age[i0:i1 + 1] >= 0) & (age[i0:i1 + 1] <= TRAJ_AGE_MAX_MS))
    if not len(fresh):
        raise AError('no fresh trajectory packet inside the run')
    j_s = i0 + int(fresh[-1])
    problems = []
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
        blk = cover is not None and (_int(cover.get('blk_ticks')) or 0) > 0
        add('khong_cat_sieu_am', not changes and not blk, f'{len(changes)} #us change; blk_ticks '
            f"{cover.get('blk_ticks') if cover else '-'}")
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
        for b, m in zip(blocks, moves):
            dur, want = m['t1'] - m['t0'], m['toi'] - m['tu']
            sl = slice(b['j0'], b['j1'] + 1)
            if m['loai'] == 'tien':
                main, others, tp = L['sp_vx'][sl], ('sp_vy', 'sp_wz'), b['tp_vx']
                good = abs(tp - want) <= SP_INT_TOL_M and abs(L['pos_x_ref'][b['j1']] - m['toi']) <= REF_TOL_M
            else:
                main, others, tp = L['sp_wz'][sl], ('sp_vx', 'sp_vy'), b['tp_wz']
                good = abs(tp - want) <= SP_INT_TOL_RAD
            pure = all(np.all(np.abs(L[c][sl]) <= SP_EPS) for c in others)
            one_sign = np.all(np.sign(main[np.abs(main) > SP_EPS]) == math.copysign(1.0, want))
            t_ok = abs((b['t0'] - t_run) - m['t0']) <= DUR_TOL_S and abs((b['t1'] - b['t0']) - dur) <= DUR_TOL_S
            if not (good and pure and one_sign and t_ok):
                ok_b = False
                detail = (f"block {b['j0']}: {m['loai']} tich phan {tp:.4f} vs {want:.4f}, truc khac 0 "
                          f"{not pure}, doi dau {not one_sign}, bat dau {b['t0'] - t_run:.2f} vs {m['t0']:.2f} s, "
                          f"dai {b['t1'] - b['t0']:.2f} vs {dur:.2f} s")
                break
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
            w_imu = np.gradient(yaw[sl], t)
            w_enc = np.gradient(np.cumsum(dth[sl]), t)
            k = np.ones(25) / 25
            a1, a2 = np.convolve(w_imu, k, 'valid'), np.convolve(w_enc, k, 'valid')
            corr = float(np.corrcoef(a1, a2)[0, 1]) if np.std(a1) > 0 and np.std(a2) > 0 else 0.0
            sign_ok = np.sign(yaw[b['j1']] - yaw[b['j0']]) == np.sign(b['tp_wz'])
            # HCC1-9: a repeated RV value while the wheels turn (a 0.15 s repeat kept corr ~0.95)
            rate = np.convolve(np.abs(dth[sl]) / np.maximum(np.r_[np.diff(t), 0.01], 1e-3), k, 'same')
            ch = np.flatnonzero(np.diff(L['rv_n'][sl]) != 0) + 1
            ysl = L['yaw_rad'][sl]
            same = [j for j0_, j in zip(ch[:-1], ch[1:]) if rate[j] > RV_REPEAT_W and abs(ysl[j] - ysl[j0_]) < 1e-6]
            add('imu_e_tuong_quan', corr >= E_CORR_MIN and sign_ok and not same,
                f'corr {corr:.3f}, RV lap luc quay {len(same)}')
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
    if any(abs(float(v)) > 0 for v in tj.get('tag_offset', [0, 0, 0])):
        raise AError(f'{track_json}: tag_offset {tj.get("tag_offset")} (track with tag_offset 0)')
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
    return F


def frames_chain(frames_csv, a6, xoay, lan_lap=None):
    """F1 of CAM when importable: frame_id, t_cam, p_x, p_y, psi_ma_p, cong_ok, trang_thai."""
    try:
        import c7_chuoi_khung as ck                                       # CAM, after its merge
    except ImportError:
        return None
    rows = ck.chuoi_khung(frames_csv, ck.doc_a6(a6), xoay, lan_lap=lan_lap)
    F = dict(idx=np.array([r['frame_id'] for r in rows]), t=np.array([r['t_cam'] for r in rows]),
             x=np.array([r['p_x'] for r in rows]), y=np.array([r['p_y'] for r in rows]),
             psi=np.array([r['psi_ma_p'] for r in rows]), ok=np.array([bool(r['cong_ok']) for r in rows]))
    F['chuoi'] = 'khoa' if all(r.get('trang_thai') == 'khoa' for r in rows) else 'chan_doan'
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
    ev = [(a, b, 'wz') for a, b in _intervals(t, np.where(rot, w, 0.0), thr_w)]
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
    lk = [(s, e, b['truc']) for (s, e), b in zip(lev, blocks)]
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


def leg_eps(F, L, ja, jb, D, psi_a, yaw_a, smooth_s=None, rv=None):
    """Largest |camera heading change - IMU yaw change| over the leg (HCC1-9): both relative to window a (window
    means psi_a, yaw_a), both averaged over the same +-smooth_s/2 time window on the camera frame times, the IMU
    interpolated in the RV report times (not the log row times). NaN (no evidence) when an RV gap > RV_STALL_S, a
    camera gap > smooth_s, or fewer than 80 % of the leg time is covered."""
    smooth_s = EPS_SMOOTH_S if smooth_s is None else smooth_s
    t_rv, y_rv = rv if rv is not None else rv_samples(L)
    ta, tb = float(L['t'][ja]), float(L['t'][jb])
    mk = F['ok'] & (F['t'] + D >= ta) & (F['t'] + D <= tb)
    if mk.sum() < 10:
        return float('nan')
    tc = F['t'][mk] + D
    if np.max(np.diff(np.r_[ta, tc, tb])) > smooth_s:
        return float('nan')
    r0, r1 = np.searchsorted(t_rv, ta), np.searchsorted(t_rv, tb)
    seg = t_rv[max(r0 - 1, 0):min(r1 + 1, len(t_rv))]
    if len(seg) < 2 or np.max(np.diff(seg)) > RV_STALL_S:
        return float('nan')
    pc = psi_a + np.vectorize(wrap)(np.unwrap(F['psi'][mk]) - psi_a)
    yi = np.interp(tc, t_rv, y_rv)
    h = smooth_s / 2
    a = np.searchsorted(tc, tc - h)
    b = np.searchsorted(tc, tc + h, side='right')
    cs_p, cs_y = np.r_[0.0, np.cumsum(pc)], np.r_[0.0, np.cumsum(yi)]
    n = b - a
    full = (tc - h >= ta) & (tc + h <= tb) & (n >= 3)
    if full.sum() < 3 or (tc[full][-1] - tc[full][0]) < 0.8 * (tb - ta - smooth_s):
        return float('nan')
    ps = (cs_p[b] - cs_p[a]) / np.maximum(n, 1)
    ys = (cs_y[b] - cs_y[a]) / np.maximum(n, 1)
    dev = (ps - psi_a) - (ys - yaw_a)
    return float(np.max(np.abs(np.vectorize(wrap)(dev[full]))))


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
        dp = np.abs(np.diff(p))
        good = np.diff(ts) <= E_GAP_MAX_S
        out.append(dict(vong=float(np.sum(dp[good]) / (2 * math.pi)), ho_s=float(np.max(gaps))))
    return out


def cam_window(F, w, D, u_D, fps=60.0):
    """Frame mask of the camera image of log window w, shrunk by u_D and one frame each side."""
    if w['ta'] is None:
        return np.zeros(len(F['t']), bool)
    a = w['ta'] - D + u_D + 1.0 / fps
    b = w['tb'] - D - u_D - 1.0 / fps
    return F['ok'] & (F['t'] >= a) & (F['t'] <= b)


# ------------------------------------------------------------------ legs and a
LONG_LEGS = ((1, 2), (2, 3), (3, 4), (4, 5))
SHORT_LEGS = ((0, 1), (5, 6))


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
    for k, w in enumerate(wins):
        m = cam_window(F, w, D, u_D, F.get('fps', 60.0))
        lw = (L['t'] >= w['ta']) & (L['t'] <= w['tb']) if w['ta'] is not None else np.zeros(L['n'], bool)
        rec = dict(chi_so=k, n=int(m.sum()), mask=m, ta=w['ta'], tb=w['tb'])
        if rec['n'] >= min_frames:
            rec['p'] = np.array([F['x'][m].mean(), F['y'][m].mean()])
            rec['psi'] = mean_unwrapped(F['psi'][m])
            rec['yaw'] = float(np.mean(yaw[lw]))
            rec['jc'] = int(np.flatnonzero(lw)[len(np.flatnonzero(lw)) // 2])
            h = int(m.sum()) // 2
            idx = np.flatnonzero(m)
            rec['psi_nua'] = abs(wrap(mean_unwrapped(F['psi'][idx[:h]]) - mean_unwrapped(F['psi'][idx[h:]])))
        W.append(rec)
    dx, dy, _ = body_increments(L)
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
        mid = 0.5 * (yaw[ja + 1:jb + 1] + yaw[ja:jb]) - wa['yaw']           # HCC1-12: window mean, as v4 3.2 step 3
        c, s = np.cos(mid), np.sin(mid)
        ds = np.array([np.sum(c * dx[ja + 1:jb + 1] - s * dy[ja + 1:jb + 1]),
                       np.sum(s * dx[ja + 1:jb + 1] + c * dy[ja + 1:jb + 1])])
        dp = wb['p'] - wa['p']
        th_a = wrap(math.atan2(dp[1], dp[0]) - math.atan2(ds[1], ds[0]))
        a_l = wrap(wa['psi'] - th_a)
        # dynamic yaw check over the leg (HCC1-9): 0.5 s, RV report times, gaps -> NaN (no evidence)
        eps = leg_eps(F, L, ja, jb, D, wa['psi'], wa['yaw'], rv=rvs)
        path = float(np.sum(np.hypot(dx[ja + 1:jb + 1], dy[ja + 1:jb + 1])))
        dth = float(np.sum(np.abs(np.diff(yaw[ja:jb + 1]))))
        leg.update(trang_thai='hop_le', dp=dp, ds=ds, L_m=float(np.hypot(*dp)), Ls_m=float(np.hypot(*ds)),
                   ti_le=float(np.hypot(*dp) / np.hypot(*ds)), theta_a=th_a, a_l=a_l, eps=eps, path=path, sum_dth=dth,
                   chieu='toi' if ds[0] > 0 else 'lui', R_a=wa['psi'], R_b=wb['psi'])
        legs.append(leg)
    return dict(D=D, u_D=u_D, diffs=diffs, windows=W, legs=legs, chuoi=F['chuoi'])


def perp_offsets(W):
    pts = np.array([w['p'] for w in W if 'p' in w])
    if len(pts) < 3:
        return float('nan')
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c)
    return float(np.max(np.abs((pts - c) @ vt[1])))


def place_result(cho, attempts, cap_id, e_hieu=None, q=None, du_m=None, neo=None, psi_cuc=None):
    """Combine the analysed attempts of one place (F7 block). attempts = list of (thu_id, analysis)."""
    res = dict(cho=cho, trang_thai='khoa', ly_do=[], luot_thu=[])
    cap = dict(attempts)[cap_id]
    longs = [lg for lg in cap['legs'] if lg['dai'] and lg['trang_thai'] == 'hop_le']
    if len(longs) != 4:
        return dict(res, trang_thai='chua_ket_luan', ly_do=['cap: fewer than 4 long legs'])
    a = circmean([lg['a_l'] for lg in longs])
    spread = max(abs(wrap(x['a_l'] - y['a_l'])) for x in longs for y in longs)
    all_long = [(tid, lg) for tid, an in attempts for lg in an['legs'] if lg['dai'] and lg['trang_thai'] == 'hop_le']
    dev = max(abs(wrap(lg['a_l'] - a)) for _, lg in all_long)
    Lmin = min(lg['L_m'] for lg in longs)
    Lsmin = min(lg['Ls_m'] for lg in longs)
    if e_hieu is None:
        res['ly_do'].append('thieu e_hieu_m (F9)')
        e = 0.0
    else:
        e = max(e_hieu)
    u_map = math.asin(min(e / Lmin, 1.0))
    u_enc = math.asin(min(COUNT_M * (2 + max(lg['sum_dth'] for lg in longs)) / Lsmin, 1.0))
    if q is None or du_m is None:
        res['ly_do'].append('thieu q, du_m (xoay_v1)')
        u_q = 0.0
    else:
        u_q = max((abs(2 * math.sin(wrap(lg['R_b'] - lg['R_a']) / 2)) * du_m) / lg['L_m'] for lg in longs)
    eps = [lg['eps'] for lg in longs]
    eps_thieu = any(not math.isfinite(x) for x in eps)            # HCC1-9: no evidence -> gate fails, not 0
    eps = [x if math.isfinite(x) else 0.0 for x in eps]
    u_yaw = max(math.asin(min(2 * math.sin(e_ / 2) * lg['path'] / lg['Ls_m'], 1.0)) for e_, lg in zip(eps, longs))
    if psi_cuc is None:
        res['ly_do'].append('thieu can huong camera hai dau (C8B)')
        u_cuc = 0.0
    else:
        u_cuc = float(psi_cuc)
    u_a = dev + u_map + u_enc + u_q + u_yaw + u_cuc
    gates = dict(
        do_tan=spread <= math.radians(SPREAD_MAX_DEG),
        ti_le=all(RATIO_LO <= lg['ti_le'] <= RATIO_HI for lg in longs)
        and (max(lg['ti_le'] for lg in longs) - min(lg['ti_le'] for lg in longs)) <= RATIO_SPREAD,
        thang_hang=perp_offsets(cap['windows'][:7]) <= PERP_MAX_M,      # the 7 stops of 12b only
        yaw_dong=not eps_thieu and all(x <= math.radians(EPS_YAW_MAX_DEG) for x in eps),
    )
    shorts = [lg for lg in cap['legs'] if not lg['dai'] and lg['trang_thai'] == 'hop_le']
    giu_ngoai = []
    for lg in shorts:
        u_short = math.asin(min(e / lg['L_m'], 1.0)) + math.asin(min(2 * COUNT_M / lg['Ls_m'], 1.0))
        giu_ngoai.append(dict(i=lg['i'], a_l_deg=math.degrees(lg['a_l']), lech_deg=math.degrees(abs(wrap(lg['a_l'] - a))),
                              nguong_deg=math.degrees(u_a + u_short),
                              dat=abs(wrap(lg['a_l'] - a)) <= u_a + u_short))
    gates['giu_ngoai'] = all(g['dat'] for g in giu_ngoai)
    if cap['chuoi'] != 'khoa':
        res['ly_do'].append('chuoi khung chua khoa (F1, A6, q)')
    if neo is None:
        res['ly_do'].append('thieu psi_chuan_k (F6)')
    failed = [k for k, v in gates.items() if not v]
    if failed:
        res['trang_thai'] = 'chua_ket_luan'
        res['ly_do'] += failed
    elif res['ly_do']:
        res['trang_thai'] = 'chan_doan'
    out = dict(res, a_k_deg=math.degrees(a), u_a_k_deg=math.degrees(u_a), do_tan_deg=math.degrees(spread),
               u_hang_deg=dict(lech_lon_nhat=math.degrees(dev), u_map=math.degrees(u_map), u_enc=math.degrees(u_enc),
                               u_q=math.degrees(u_q), u_yaw=math.degrees(u_yaw), u_psi_cuc=math.degrees(u_cuc)),
               cong=gates, giu_ngoai=giu_ngoai)
    if neo is not None:
        out['phi_hat_chuan_k_deg'] = math.degrees(wrap(math.radians(neo) - a))
        out['u_phi_hat_chuan_k_deg'] = out['u_a_k_deg']
    for tid, an in attempts:
        out['luot_thu'].append(dict(thu_id=tid, cap=tid == cap_id, D_s=an['D'], u_D_s=an['u_D'],
                                    doan=[{k: (v.tolist() if isinstance(v, np.ndarray) else
                                               (math.degrees(v) if k in ('a_l', 'theta_a', 'eps') and isinstance(v, float) else v))
                                           for k, v in lg.items() if k not in ('R_a', 'R_b')} for lg in an['legs']]))
    return out


# ------------------------------------------------------------------ CLI
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
        F = frames_chain(a.frames, a.a6, a.xoay)
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
    rows = []
    roles = window_roles(A['segs'])
    for k_w, lo in roles:
        w = an['windows'][k_w]
        idx = F['idx'][w['mask']]
        ts = F['t'][w['mask']]
        k_sha = hashlib.sha256(','.join(map(str, idx.tolist())).encode()).hexdigest()
        rows.append(dict(schema='cua_so_v1', luot_id=A['thu_id'], thu_id=A['thu_id'], cho=A['cho'], loai=lo,
                         chi_so=w['chi_so'], clip=a.clip and os.path.basename(a.clip), clip_sha256=clip_sha,
                         session_sha256=a.session_sha256, frames_sha256=sha256(a.frames),
                         khung=[dict(id=int(i), pts_s=float(t)) for i, t in zip(idx, ts)], khung_sha256=k_sha,
                         phep_gop='trung_binh_go_vong', t_canh_s=None if w['tb'] is None else w['tb'] + 0.1 - an['D'],
                         D_s=an['D'], u_D_s=an['u_D'], t0_log_us=int(A['log']['timestamp_us'][0]),
                         esp32_sha256=sha256(A['log_path']), chuoi=an['chuoi']))
    out = os.path.join(a.dir, 'cua_so_v1.jsonl')
    with open(out, 'x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    doan = dict(schema='c8_hc_doan_12b_v1', thu_id=A['thu_id'], cho=A['cho'], trang_thai=an['chuoi'],
                clip=a.clip, clip_sha256=clip_sha, session_sha256=a.session_sha256, frames_sha256=sha256(a.frames),
                track_sha256=sha256(a.track), cua_so_v1_sha256=sha256(out), esp32_sha256=sha256(A['log_path']),
                dets_sha256=sha256(a.dets) if a.dets else None, chuoi=an['chuoi'],
                nguon=[dict(vai_tro=k, duong_dan=os.path.abspath(v), sha256=sha256(v))
                       for k, v in (('frames', a.frames), ('track', a.track), ('log', A['log_path']),
                                    ('sender', os.path.join(a.dir, 'sender.txt')), ('lich', os.path.join(a.dir, 'lich.csv')),
                                    ('cua_so_v1', out)) if v],
                doan=[])
    for lg in an['legs']:
        wa, wb = an['windows'][lg['a']], an['windows'][lg['b']]
        d = dict(i=lg['i'], dai=lg['dai'], trang_thai=lg['trang_thai'], ly_do=lg.get('ly_do'))
        for nm, w in (('cua_so_a', wa), ('cua_so_b', wb)):
            d[nm] = dict(chi_so=w['chi_so'], khung_id=F['idx'][w['mask']].tolist(),
                         p_m=w['p'].tolist() if 'p' in w else None, psi_rad=w.get('psi'))
        if lg['trang_thai'] == 'hop_le':
            d.update(chieu=lg['chieu'], dp_m=lg['dp'].tolist(), L_m=lg['L_m'])
        doan['doan'].append(d)
    body = json.dumps(doan, ensure_ascii=False, sort_keys=True)
    doan['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    with open(os.path.join(a.dir, 'doan_12b_v1.json'), 'x', encoding='utf-8') as f:
        json.dump(doan, f, ensure_ascii=False, indent=1)
    print(f"D={an['D']:.3f} s u_D={an['u_D']:.3f} s chuoi={an['chuoi']}")
    for lg in an['legs']:
        if lg['trang_thai'] == 'hop_le':
            print(f"doan {lg['i']} {'dai ' if lg['dai'] else 'ngan'} {lg['chieu']:3s} L={lg['L_m']:.3f} m "
                  f"a_l={math.degrees(lg['a_l']):+.3f} deg ti_le={lg['ti_le']:.4f} eps={math.degrees(lg['eps']):.3f} deg")
        else:
            print(f"doan {lg['i']} {lg['trang_thai']}: {lg.get('ly_do')}")
    return 0


def cmd_tinh(a):
    attempts = []
    for spec in [a.cap] + list(a.them or []):
        d = spec
        A = load_attempt(d)
        if A['cho'] != a.cho:
            raise AError(f'{d}: cho {A["cho"]} != {a.cho}')
        fr = a.frames_of.get(d) if a.frames_of else None
        if not fr:
            raise AError(f'{d}: --frames-of {d}=FRAMES,TRACK missing')
        frames, track = fr.split(',')
        ck, info = kiem(A)
        if 'windows' not in info:
            raise AError(f'{d}: kiem gave no windows')
        F = None
        if a.a6 and a.xoay:
            F = frames_chain(frames, a.a6, a.xoay)
        F = F or frames_basic(frames, track)
        attempts.append((A['thu_id'], analyse(A, F, info)))
    e = None
    if a.e_hieu:
        j = json.load(open(a.e_hieu, encoding='utf-8'))
        e = [float(x['e_hieu_m']) for x in j.get('doan', []) if x.get('dai')]
    q = du = None
    if a.xoay:
        x = json.load(open(a.xoay, encoding='utf-8'))
        q, du = x.get('q_tag_m'), x.get('du_m')
    neo = None
    if a.neo:
        nj = json.load(open(a.neo, encoding='utf-8'))
        neo = nj.get('cho', {}).get(a.cho, {}).get('psi_chuan_k_deg', nj.get('psi_chuan_k_deg'))
    cap_id = os.path.basename(os.path.normpath(a.cap))
    r = place_result(a.cho, attempts, cap_id, e_hieu=e, q=q, du_m=du, neo=neo, psi_cuc=a.u_psi_cuc_rad)
    r = dict(schema='truc_xe_v1', version=VERSION, **r)
    txt = json.dumps(r, ensure_ascii=False, indent=1, default=float)
    if a.out:
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(txt)
    print(f"{a.cho}: a = {r['a_k_deg']:+.3f} deg, u_a = {r['u_a_k_deg']:.3f} deg, do tan {r['do_tan_deg']:.3f} deg, "
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
        add('yaw_dong', ok4 and all(math.isfinite(x) and x <= math.radians(EPS_YAW_MAX_DEG) for x in eps),
            ', '.join('thieu' if not math.isfinite(x) else f'{math.degrees(x):.3f}' for x in eps))
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
    report(vk)
    good = all(ok for _, ok, _ in vk)
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
    t = sub.add_parser('tinh')
    t.add_argument('--cho', required=True, choices=sorted(hs.PLACES))
    t.add_argument('--cap', required=True, help='attempt folder whose a is issued')
    t.add_argument('--them', nargs='*', help='other attempt folders of this place in the session (failed ones too)')
    t.add_argument('--frames-of', nargs='+', required=True, help='DIR=FRAMES_CSV,TRACK_JSON for each attempt')
    t.add_argument('--a6', default=None)
    t.add_argument('--xoay', default=None)
    t.add_argument('--neo', default=None, help='hieu_chuan_neo.json of NHACTAM (F6)')
    t.add_argument('--e-hieu', default=None, help='e_hieu_m of KIEMCHEO (F9)')
    t.add_argument('--u-psi-cuc-rad', type=float, default=None, help='C8B two-end camera heading bound (pending)')
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
