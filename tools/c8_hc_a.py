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
EPS_SMOOTH_S = 1.0             # smoothing of the camera heading in the dynamic yaw check
MIN_FRAMES_TINH, MIN_FRAMES_VIDEO = 120, 30
WIN_MIN_S, STILL_MIN_S = 2.5, 3.0
U_D_MAX_S = 0.10
TRAJ_AGE_MAX_MS = 60.0
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


def kiem(A, age_r_max_ms=125.0):
    """Log-side checks of one attempt (design 2.6). Returns (checks list of (name, ok, detail), info)."""
    L, segs = A['log'], A['segs']
    ck = []

    def add(name, ok, detail=''):
        ck.append((name, bool(ok), str(detail)))
    se = A['seq_end'] or {}
    add('sender_hoan_tat', se.get('complete') == '1' and se.get('n_bo') == '0' and not A['huy'],
        f"complete={se.get('complete')} n_bo={se.get('n_bo')} huy={len(A['huy'])}")
    add('logger_dong_sach', A['logger_dong'].get('ma') == '0', f"ma={A['logger_dong'].get('ma')}")
    try:
        import gain_line_check as glc
        qb = glc.read_query(os.path.join(A['dir'], 'gain_query_truoc.txt'))
        qa = glc.read_query(os.path.join(A['dir'], 'gain_query_sau.txt'))
        why = glc.query_problem(qb, qa, check_run_boot=False)
        same = why is None and all((qb or {}).get(k) == (qa or {}).get(k) for k in ('sha', 'env', 'applied_n'))
        add('dinh_danh_gain', same, why or '')
        ev = glc.read_events(A['events_path']) or []
        fw = [parse_fields(t) for _, t in map(glc.split_event, ev) if t.startswith('#fw,')]
        gn = [parse_fields(t) for _, t in map(glc.split_event, ev) if t.startswith('#gain,')]
        okfw = all(f.get('sha', '')[:7] == (qb or {}).get('sha', '')[:7] and f.get('env') == (qb or {}).get('env') for f in fw)
        okgn = all((g.get('boot') in (None, '') or g.get('boot', '').lower() == (qb or {}).get('boot', '').lower()) for g in gn)
        add('dinh_danh_fw_gain_log', okfw and okgn, f'#fw {len(fw)} #gain {len(gn)}')
        us = [t for _, t in map(glc.split_event, ev) if t.startswith('#us,') and 'ev=change' in t]
        add('khong_cat_sieu_am', not us, f'{len(us)} #us change')
        cat = [t for _, t in map(glc.split_event, ev) if t.startswith('#us_end')]
        cat_pwm = 'chua_xac_minh' if not cat else ('khong' if all(parse_fields(t).get('cut', '0') in ('0', '') for t in cat) else 'co')
    except ImportError:
        add('dinh_danh_gain', False, 'gain_line_check not importable')
        cat_pwm = 'chua_xac_minh'
    info = dict(cat_pwm=cat_pwm)
    try:
        i0, i1 = run_range(L)
    except AError as exc:
        add('mot_canh_run_active', False, exc)
        return ck, info
    add('mot_canh_run_active', True, f'rows {i0}-{i1}')
    rng = slice(i0, i1 + 1)
    add('motor_on', np.all(L['motor_on'][rng] > 0.5))
    add('pos_reset_tang_1', i0 > 0 and L['pos_reset_n'][i1] - L['pos_reset_n'][i0 - 1] == 1
        and L['pos_reset_n'][i0] == L['pos_reset_n'][i1],
        f"{L['pos_reset_n'][max(i0 - 1, 0)]:.0f}->{L['pos_reset_n'][i1]:.0f}")
    add('log_seq_lien', np.all(np.diff(L['log_seq'][rng]) == 1))
    add('so_dem_hop_le', not np.any(L['bad'][rng]))
    add('traj_age', np.nanmax(L['traj_age_ms'][rng]) <= TRAJ_AGE_MAX_MS, f"max {np.nanmax(L['traj_age_ms'][rng]):.0f} ms")
    blocks = sp_blocks(L, i0, i1)
    moves = plan_moves(segs)
    ok_b = len(blocks) == len(moves)
    detail = f'{len(blocks)} blocks vs {len(moves)} planned'
    if ok_b:
        x = th = 0.0
        for b, m in zip(blocks, moves):
            dur = m['t1'] - m['t0']
            if m['loai'] == 'tien':
                x = m['toi']
                good = (b['truc'] == 'vx' and not b['vy_any'] and abs(b['tp_vx'] - (m['toi'] - m['tu'])) <= SP_INT_TOL_M
                        and abs(b['tp_wz']) <= SP_INT_TOL_RAD)
                jr = b['j1']
                good = good and abs(L['pos_x_ref'][jr] - x) <= REF_TOL_M
            else:
                th = m['toi']
                good = b['truc'] == 'wz' and abs(b['tp_wz'] - (m['toi'] - m['tu'])) <= SP_INT_TOL_RAD and abs(b['tp_vx']) <= SP_INT_TOL_M
            good = good and abs((b['t1'] - b['t0']) - dur) <= DUR_TOL_S
            if not good:
                ok_b = False
                detail = f"block {b['j0']}: truc {b['truc']} vx {b['tp_vx']:.4f} wz {b['tp_wz']:.4f} dur {b['t1'] - b['t0']:.2f} vs {m}"
                break
    add('lenh_nhan_khop_lich', ok_b, detail)
    if ok_b:
        nz_after = (np.abs(L['sp_vx'][blocks[-1]['j1'] + 1:i1 + 1]) > SP_EPS) | (np.abs(L['sp_wz'][blocks[-1]['j1'] + 1:i1 + 1]) > SP_EPS)
        add('khong_goi_muon', not np.any(nz_after))
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
            add('imu_e_tuong_quan', corr >= E_CORR_MIN and sign_ok, f'corr {corr:.3f}')
        wins = still_windows(L, blocks, i0, i1)
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
        info.update(i0=i0, i1=i1, blocks=blocks, windows=wins)
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


def cam_motion_events(F, kind, thr_v=0.02, thr_w=math.radians(5.0), half=0.25):
    """(start, end) camera times of motion intervals: kind 'vx' from the tag centre speed, 'wz' from the heading rate."""
    m = F['ok']
    t, x, y, p = F['t'][m], F['x'][m], F['y'][m], np.unwrap(F['psi'][m])
    if len(t) < 10:
        return []
    sp = np.full(len(t), 0.0)
    for i in range(len(t)):
        a = np.searchsorted(t, t[i] - half)
        b = min(np.searchsorted(t, t[i] + half), len(t) - 1)
        if b > a and t[b] > t[a]:
            sp[i] = (math.hypot(x[b] - x[a], y[b] - y[a]) / (t[b] - t[a]) if kind == 'vx'
                     else abs(p[b] - p[a]) / (t[b] - t[a]))
    mv = sp > (thr_v if kind == 'vx' else thr_w)
    ev, i = [], 0
    while i < len(t):
        if mv[i]:
            j = i
            while j + 1 < len(t) and mv[j + 1]:
                j += 1
            if t[j] - t[i] > 0.5:
                ev.append((float(t[i]), float(t[j])))
            i = j + 1
        else:
            i += 1
    return ev


def log_motion_events(L, blocks, thr_v=0.02, thr_w=math.radians(5.0), half=0.25):
    """(start, end) log times of motion of each block, with the SAME speed thresholds and window as the camera
    (cam_motion_events): translation speed from the encoder odometry, turn rate from the wheel increments. A slow
    creep after the stop (ADRC) stays under the threshold on both sides."""
    dx, dy, dth = body_increments(L)
    t = L['t']
    X, Y, TH = np.cumsum(dx), np.cumsum(dy), np.cumsum(dth)
    out = []
    for k, b in enumerate(blocks):
        j0 = b['j0']
        j1 = blocks[k + 1]['j0'] if k + 1 < len(blocks) else L['n']
        idx = np.arange(j0, j1)
        a = np.searchsorted(t, t[idx] - half)
        c = np.minimum(np.searchsorted(t, t[idx] + half), L['n'] - 1)
        dt = np.maximum(t[c] - t[a], 1e-6)
        sp = (np.hypot(X[c] - X[a], Y[c] - Y[a]) / dt) if b['truc'] == 'vx' else (np.abs(TH[c] - TH[a]) / dt)
        mv = np.flatnonzero(sp > (thr_v if b['truc'] == 'vx' else thr_w))
        out.append(None if not len(mv) else (float(t[idx[mv[0]]]), float(t[idx[mv[-1]]])))
    return out


def align(L, blocks, F):
    """D = t_log - t_cam from the motion edges of translation and turning; returns (D, u_D, pairs)."""
    lev = log_motion_events(L, blocks)
    diffs = []
    for kind in ('vx', 'wz'):
        cev = cam_motion_events(F, kind)
        lk = [e for e, b in zip(lev, blocks) if b['truc'] == kind and e is not None]
        if not lk:
            continue
        if len(cev) != len(lk):
            raise AError(f'{kind}: {len(cev)} camera motion intervals vs {len(lk)} in the log')
        for (cs, ce), (ls, le) in zip(cev, lk):
            diffs += [ls - cs, le - ce]
    if not diffs:
        raise AError('no motion edge to align camera and log')
    D = float(np.median(diffs))
    return D, float(np.max(np.abs(np.array(diffs) - D))), diffs


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
    legs = []
    for (ia, ib), dai in ([] if n12 == 0 else [(lg, True) for lg in LONG_LEGS] + [(lg, False) for lg in SHORT_LEGS]):
        wa, wb = W[ia], W[ib]
        leg = dict(i=len(legs), dai=dai, a=ia, b=ib)
        if 'p' not in wa or 'p' not in wb:
            leg.update(trang_thai='thieu', ly_do=f'window {ia if "p" not in wa else ib} < {min_frames} frames')
            legs.append(leg)
            continue
        ja, jb = wa['jc'], wb['jc']
        ref = yaw[ja]
        mid = 0.5 * (yaw[ja + 1:jb + 1] + yaw[ja:jb]) - ref
        c, s = np.cos(mid), np.sin(mid)
        ds = np.array([np.sum(c * dx[ja + 1:jb + 1] - s * dy[ja + 1:jb + 1]),
                       np.sum(s * dx[ja + 1:jb + 1] + c * dy[ja + 1:jb + 1])])
        dp = wb['p'] - wa['p']
        th_a = wrap(math.atan2(dp[1], dp[0]) - math.atan2(ds[1], ds[0]))
        a_l = wrap(wa['psi'] - th_a)
        # dynamic yaw check over the leg: camera heading (smoothed) vs IMU, both relative to window a
        mk = F['ok'] & (F['t'] + D >= L['t'][ja]) & (F['t'] + D <= L['t'][jb])
        eps = float('nan')
        if mk.sum() >= 10:
            tc = F['t'][mk] + D
            pc = np.unwrap(F['psi'][mk])
            ker = max(1, int(round(EPS_SMOOTH_S * F.get('fps', 60.0))))
            pcs = np.convolve(pc, np.ones(ker) / ker, 'same')
            yi = np.convolve(np.interp(tc, L['t'], yaw), np.ones(ker) / ker, 'same')   # same smoothing, same grid
            dev = (pcs - wa['psi']) - (yi - yaw[ja])
            core = slice(ker // 2, len(dev) - ker // 2) if len(dev) > ker else slice(0, len(dev))
            eps = float(np.max(np.abs(np.vectorize(wrap)(dev[core])))) if len(dev[core]) else float('nan')
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
    if any(not math.isfinite(x) for x in eps):
        res['ly_do'].append('thieu kiem yaw dong')
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
        yaw_dong=all(x <= math.radians(EPS_YAW_MAX_DEG) for x in eps),
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


def cmd_kiem_video(a):
    import c8_hc_video as hv                       # sparse video reader (separate module, cv2)
    A = load_attempt(a.dir)
    ck, info = kiem(A)
    if 'windows' not in info:
        report(ck)
        raise AError('kiem: no windows')
    F = hv.sparse_frames(a.clip, a.session, stride=a.stride)
    an = analyse(A, F, info, min_frames=MIN_FRAMES_VIDEO)
    longs = [lg for lg in an['legs'] if lg['dai'] and lg['trang_thai'] == 'hop_le']
    ok = len(longs) == 4
    spread = max(abs(wrap(x['a_l'] - y['a_l'])) for x in longs for y in longs) if ok else float('nan')
    eps_ok = ok and all(math.isfinite(lg['eps']) and lg['eps'] <= math.radians(EPS_YAW_MAX_DEG) for lg in longs)
    print(f"D={an['D']:.3f} s; {len(longs)}/4 doan dai; do tan tho {math.degrees(spread):.3f} deg; yaw dong {'dat' if eps_ok else 'HONG'}")
    good = ok and spread <= math.radians(SPREAD_VIDEO_DEG) and eps_ok
    print('LAM_LAI' if not good else 'DAT')
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
    v.add_argument('--stride', type=int, default=3)
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
