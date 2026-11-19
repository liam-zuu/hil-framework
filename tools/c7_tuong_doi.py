#!/usr/bin/env python3
"""
c7_tuong_doi.py: the relative-frame part of the camera scorer (design CODE_CAM v5, CLAUDE_THESIS
03_CODE/docs/2026-11-19_Thiet_ke_CAM_cham_tuong_doi_v5.md section 3; rules J v6 03_CODE/docs/
2026-11-19_J_Quy_tac_cham_O1_v6.md sections 2, 5; formats: the C8 data contract 03_CODE/docs/
2026-11-19_Hop_dong_du_lieu_C8_tuong_doi_v1.md F2, F7, F8, F12, F13). c7_score imports it for --tuong-doi.

  cua_so      the still windows of a run (F2 cua_so_v1, rows 'pre' and 'post'): frames, frames that pass the tag gate
              of c7_chuoi_khung, the sample gate, the still gate (tag centre and tag heading of the frames, so a
              missing q or A6 never makes a still car look moving), the log coverage; a failed 'pre' window is a run
              lost to the reference window (J 5), decided before any other scoring rule.
  tham_chieu  the start pose (F12 tham_chieu_v1): over the INTERSECTION of the tag-gate frames of 'pre' and the body
              anchor frames of NHACTAM (lan_lap_v1 khung_chung.pre); an intersection with samples that fails the
              sample gate: run lost to the reference window; an EMPTY intersection (all anchors lost): the tag-gate
              frames of the same window, flag mat_neo (J 2). p_tc = mean centre p (after q), psi'_tc =
              trung_binh_go_vong of psi' (after A6) on that set.
  phi_beta    the lane angle by the beta path of the same session only (J 2): phi = phi_hat_chuan_k (truc_xe_v1 of
              HIEUCHUAN, place k of the task) + beta (lan_lap_v1 of NHACTAM); u_phi = 1.25 (u_phi_hat + u_beta) +
              u_beta_cam (contract 2.C; u_beta_cam of CODE_C8B, not yet agreed: None -> u_phi not finite).
  to_lane     the rigid change of frame that lets the absolute scorer run unchanged (design v5 0.1): body frame at
              the start b = R(-phi)(p - p_tc), then the nominal lane frame p' = p_dd + R(psi_dd) b, psi' = psi_dd +
              (psi'_ma - psi'_tc); (p_dd, psi_dd) = ((0, 0), HEAD0) for the figure 8, ((0, 0), 0) for K1, K2.
  t_som       the earliest time completion cannot be ruled out without the position (J 5 'Thieu can cu vi tri').
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c7_chuoi_khung as ck                                              # noqa: E402

PHIEN_BAN = 'c7_tuong_doi_v1'
CUA_SO_SCHEMA = 'cua_so_v1'
THAM_CHIEU_SCHEMA = 'tham_chieu_v1'
KTD_SCHEMA = 'khung_tuong_doi_v1'
TRUC_XE_SCHEMA = 'truc_xe_v1'
LE = 1.25                       # margin, once (C8B multiplies the indices; here only for u_phi, contract 2.C)
W_S = 3.0                       # J v6 2: the reference window is 3 s (not a constant of the file)
CHO_LOG_S = 6.0                 # J v6 2: the car still >= 6 s after the logger has a line (CODE 19/11)
KTC_F14 = ('n_min', 'frac_min', 'p_dung_m', 'psi_dung_deg')      # contract F14: the block locked at C8
INT32_MIN = -2 ** 31
COUNT_COLS = ('fl_count', 'fr_count', 'rl_count', 'rr_count')
# the gates of the reference window [DE XUAT, locked at C8] (J 4 'Nguong he do'; design v5 3.2; contract F14)
KHUNG_TC_DE_XUAT = dict(n_min=120, frac_min=0.6, p_dung_m=0.003, psi_dung_deg=0.3, w_s=3.0, bien_s=0.1,
                        dau_cuoi_s=0.25, log_ho_s=0.2)
KTC_RANGES = dict(n_min=(10, 10000), frac_min=(0.0, 1.0), p_dung_m=(0.0, 0.05), psi_dung_deg=(0.0, 5.0),
                  w_s=(0.5, 10.0), bien_s=(0.0, 1.0), dau_cuoi_s=(0.05, 1.0), log_ho_s=(0.01, 2.0))
TAM_DOI = ('khong_doi', 'doi', 'khong_xac_dinh')


def cho_of(task):
    """Place k of the 12b calibration for a task (J 2): figure 8 -> goc, O1-K1 -> K1, O1-K2 -> K2."""
    if task.startswith('O1-F'):
        return 'goc'
    if task in ('O1-K1', 'O1-K2'):
        return task[3:]
    raise ValueError(f'task {task!r}: no 12b place')


def ids_sha(ids):
    """sha256 of an ID list, the hash of HIEUCHUAN (c8_hc_a._ids_sha)."""
    return hashlib.sha256(','.join(str(int(i)) for i in ids).encode()).hexdigest()


def check_khung_tc(d, locked):
    """The khung_tc block of the constants file (F14), or the proposed numbers when absent before C8. Only the four
    F14 keys are required when locked (khoa_c8); w_s is fixed at 3 s by J v6 2 (another value refused); bien_s,
    dau_cuoi_s, log_ho_s are code proposals (a file may set them inside their ranges). Called for the relative mode
    only (Codex CT1-15: a locked file of the absolute mode needs no khung_tc)."""
    if d is None:
        if locked:
            raise ValueError(f'hang_so khung_tc: the reference-window gates {KTC_F14} are required when khoa_c8 '
                             '(J 4, contract F14)')
        return dict(KHUNG_TC_DE_XUAT, nguon='de_xuat_mac_dinh (c7_tuong_doi, khoa o C8)')
    if not isinstance(d, dict):
        raise ValueError('hang_so khung_tc: a table')
    extra = sorted(set(d) - set(KTC_RANGES) - {'nguon'})
    if extra:
        raise ValueError(f'hang_so khung_tc: unknown keys {extra}')
    out = {}
    for k, (lo, hi) in KTC_RANGES.items():
        if k in d:
            v = d[k]
        elif locked and k in KTC_F14:
            raise ValueError(f'hang_so khung_tc {k}: required when khoa_c8 (contract F14)')
        else:
            v = KHUNG_TC_DE_XUAT[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'hang_so khung_tc {k}: {v!r} not in [{lo}, {hi}]')
        out[k] = int(v) if k == 'n_min' else float(v)
    if out['w_s'] != W_S:
        raise ValueError(f'hang_so khung_tc w_s {out["w_s"]!r}: J v6 2 fixes the reference window at {W_S:g} s')
    out['nguon'] = str(d.get('nguon', 'hang_so'))
    return out


# ---------------------------------------------------------------- windows (F2)
def _frame_dt(t):
    dt = np.diff(np.asarray(t, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not dt.size:
        raise ValueError('frames.csv: no frame interval')
    return float(np.median(dt))


def _stillness(t, x, y, psi, kt):
    """(still, numbers) of a set of tag-gate frames: start-end shift (dau_cuoi_s means), largest deviation from the
    mean, heading spread after unwrapping around the circular mean."""
    if len(t) < 2:
        return False, dict(n=int(len(t)))
    a, b = t <= t[0] + kt['dau_cuoi_s'], t >= t[-1] - kt['dau_cuoi_s']
    shift = math.hypot(float(x[b].mean() - x[a].mean()), float(y[b].mean() - y[a].mean()))
    dev = float(np.max(np.hypot(x - x.mean(), y - y.mean())))
    c = ck.trung_binh_go_vong(psi)
    u = ck.wrap(psi - c)
    spread = float(np.max(u) - np.min(u))
    ok = shift <= kt['p_dung_m'] and dev <= kt['p_dung_m'] and math.degrees(spread) <= kt['psi_dung_deg']
    return bool(ok), dict(n=int(len(t)), dich_dau_cuoi_m=shift, lech_tb_max_m=dev, tan_huong_deg=math.degrees(spread))


def _samples(t_ok, a, b, dt, kt, t_trong):
    """Sample gate on [a, b) of the tag-gate frame times t_ok (J 2): count, share of the nominal frames, no hole above
    t_trong (from a, between frames, to the capture end of the last), a frame in the first and the last bien_s."""
    n = int(len(t_ok))
    nominal = (b - a) / dt
    why = []
    if n < kt['n_min']:
        why.append(f'{n} khung < n_min {kt["n_min"]}')
    if n < kt['frac_min'] * nominal:
        why.append(f'{n} khung < {kt["frac_min"]:g} x {nominal:.0f} danh dinh')
    if n:
        edges = np.concatenate([[a], t_ok, [b - dt]])
        hole = float(np.max(np.diff(edges)))
        if hole > t_trong + 1e-12:
            why.append(f'ho {hole:.3f} s > t_trong {t_trong:g} s')
        if t_ok[0] > a + kt['bien_s'] or t_ok[-1] + dt < b - kt['bien_s']:
            why.append('thieu khung o bien cua so')
    return not why, why


def _uncovered(t_ok, dt, a, b):
    """Length of [a, b] not covered by any capture interval [t, t + dt] of the tag-gate frames (ho_noi_s)."""
    if b <= a:
        return 0.0
    iv = sorted((max(a, float(t)), min(b, float(t) + dt)) for t in t_ok if t + dt > a and t < b)
    cov, end = 0.0, a
    for s, e in iv:
        if e <= end:
            continue
        cov += e - max(s, end)
        end = e
    return float(b - a - cov)


def _max_hole(t_ok, dt, a, b):
    """Longest part of [a, b] not covered by the capture intervals [t, t + dt] of the frames t_ok (b - a when none)."""
    if b <= a:
        return 0.0
    iv = sorted((max(a, float(t)), min(b, float(t) + dt)) for t in t_ok if t + dt > a and t < b)
    hole, end = 0.0, a
    for s, e in iv:
        hole = max(hole, s - end)
        end = max(end, e)
    return float(max(hole, b - end))


RATIO_WHY = ('ti_le', 'ti_le_it_lan_can')       # the (2b) verdicts of c7_measure.apply_gates (two-sided)


def mot_phia(frames, chain, gates, quant=5e-7):
    """Tag-gate frames of a reference window with the ONE-SIDED scale gate (contract F2; design v4 1.2; Codex CT1-14):
    a frame whose only failures are the two-sided (2b) verdicts, or none, passes when its scale is within scale_rel
    of the median scale of at least scale_min earlier frames of that kind within scale_win_s (the frame excluded);
    rounded scales (quant, c7_measure.CSV_QUANT) must pass for every value in the interval. Frames after a frame never
    change its verdict. Also needs the chain's tag gate except (2b) (car seen, finite tag)."""
    g = gates
    t = np.array([float(r['t_s']) for r in frames], float)
    sc = np.array([ck._f(r.get('scale')) for r in frames], float)
    base = np.array([bool(r.get('car_seen')) and all(w in RATIO_WHY for w in (r.get('why') or []))
                     and math.isfinite(c['tag_x']) and math.isfinite(c['tag_y']) and math.isfinite(c['tag_psi'])
                     and math.isfinite(s_) for r, c, s_ in zip(frames, chain, sc)], bool)
    out = np.zeros(len(frames), bool)
    ib = np.nonzero(base)[0]
    for j, k in enumerate(ib):
        prev = ib[:j]
        nb = sc[prev][t[prev] >= t[k] - g['scale_win_s'] - 1e-9]
        if len(nb) < g['scale_min']:
            continue
        ref = float(np.median(nb))
        r_lo = (sc[k] - quant) / (ref + quant) - 1.0
        r_hi = (sc[k] + quant) / (ref - quant) - 1.0
        out[k] = max(abs(r_lo), abs(r_hi)) <= g['scale_rel']
    return out


def _log_cover(log_rows, left_s, right_s, gap_s):
    """(covered, why, encoder changed) of the log over [left_s, right_s] (ESP32 clock): a received line at or before
    the left end and one at or after the right end, no hole above gap_s between them; the wheel counts (lines with
    four valid counts, INT32_MIN dropped first) only reported (J 5: no cause assigned)."""
    t = np.array([float(r['_t']) / 1e6 for r in log_rows])
    why = []
    if not len(t) or t[0] > left_s or t[-1] < right_s:
        why.append('log khong phu hai bien')
    sel = np.nonzero((t >= left_s - gap_s) & (t <= right_s + gap_s))[0]
    inside = t[(t >= left_s) & (t <= right_s)]
    pts = np.concatenate([[t[t <= left_s].max()] if np.any(t <= left_s) else [],
                          inside, [t[t >= right_s].min()] if np.any(t >= right_s) else []])
    if len(pts) > 1 and float(np.max(np.diff(pts))) > gap_s + 1e-9:
        why.append(f'log ho {float(np.max(np.diff(pts))):.3f} s > {gap_s:g} s')
    enc = None
    if len(sel) and all(c in log_rows[0] for c in COUNT_COLS):
        C = []
        for i in sel:
            try:
                c = [int(float(log_rows[i][k])) for k in COUNT_COLS]
            except (TypeError, ValueError):
                continue
            if INT32_MIN not in c:
                C.append(c)
        enc = bool(len(C) > 1 and np.any(np.array(C) != np.array(C[0])))
    return not why, why, enc


def cua_so(frames, chain, log_rows, t0_us, nodes, u_nodes, kt, t_trong, t_cmd_end, task, extra=None, gates=None,
           extra_why=()):
    """F2 rows (pre, post) of one run. frames: rows of c7_measure.read_frames; chain: rows of c7_chuoi_khung (same
    order, same idx); nodes: D_i (t_log = t_cam + D_i); u_nodes: u_ref,i [s] (the time part: grid or match error,
    drift, one frame, tick), symmetric. t_cmd_end: T_cmd (command time). extra: identity keys (luot_id, clip...).
    gates: frame-gate thresholds of the track (scale_min, scale_rel, scale_win_s; for the one-sided gate of 'pre');
    None: the two-sided valid of frames.csv. extra_why: reasons found by the caller (log not covering the window for
    the tick bound, Codex CT1-13): the 'pre' row fails with them (a run lost to the reference window, not exit 2)."""
    t = np.array([r['t_s'] for r in frames], float)
    ids = np.array([int(r['idx']) for r in frames])
    if [c['frame_id'] for c in chain] != [int(i) for i in ids]:
        raise ValueError('chain and frames.csv: different frames (run c7_chuoi_khung on this frames.csv)')
    ok = np.array([bool(c['cong_ok']) for c in chain])
    ok1 = mot_phia(frames, chain, gates) if gates is not None else ok
    tx = np.array([c['tag_x'] for c in chain], float)
    ty = np.array([c['tag_y'] for c in chain], float)
    tp = np.array([c['tag_psi'] for c in chain], float)
    dt = _frame_dt(t)
    t0 = t0_us / 1e6
    nodes, u_nodes = [float(v) for v in nodes], [float(v) for v in u_nodes]
    t_ref = min(t0 - D - u for D, u in zip(nodes, u_nodes))             # camera clock, earliest possible edge
    t_max = max(t0 - D + u for D, u in zip(nodes, u_nodes))             # latest possible edge
    base = dict(schema=CUA_SO_SCHEMA, cho=cho_of(task), phep_gop='trung_binh_go_vong', t0_log_us=t0_us,
                D_s=nodes, u_ref_minus_s=u_nodes, u_ref_plus_s=u_nodes, phien_ban=PHIEN_BAN,
                M=[t0 - max(nodes) - 6.0, t0 - min(nodes) + 1.0], nguong={k: kt[k] for k in KHUNG_TC_DE_XUAT},
                nguong_nguon=kt['nguon'], chuoi=ck.PHIEN_BAN, **(extra or {}))
    out = []
    # pre: every decoded PTS in [t_ref - w_s, t_ref) (khung); a tag frame is used when it passes the one-sided gate and
    # its latest capture time (PTS + one frame) is before t_ref (khung_ma_hop_le; Codex CT1-14)
    a = t_ref - kt['w_s']
    m = (t >= a) & (t < t_ref)
    mo = m & ok1 & (t + dt < t_ref)
    s_ok, s_why = _samples(t[mo], a, t_ref, dt, kt, t_trong)
    ms = (t >= a) & (t <= t_max) & ok1                                 # still over the window AND the joint to t_max
    d_ok, d_num = _stillness(t[ms], tx[ms], ty[ms], tp[ms], kt)
    l_ok, l_why, enc = _log_cover(log_rows, a + min(nodes), t_max + max(nodes), kt['log_ho_s'])
    ho = _uncovered(t[ok1], dt, t_ref, t_max)
    # J v6 2 'to the latest edge' (Codex CT1-12): no hole above t_trong in the tag coverage over [t_ref, t_max]
    ho_max = _max_hole(t[ok1], dt, t_ref, t_max)
    noi_ok = bool(ho_max <= t_trong + 1e-12)
    # J v6 2 'waiting time': the earliest edge >= 6 s after the first log line (log clock: the edge is t0)
    t_log0 = min(float(r['_t']) for r in log_rows) / 1e6 if log_rows else math.nan
    cho = t0 - t_log0
    cho_ok = bool(math.isfinite(cho) and cho >= CHO_LOG_S - 1e-9)
    why = (s_why + ([] if d_ok else ['xe khong dung yen trong cua so hay toi canh muon nhat']) + l_why
           + ([] if noi_ok else [f'doan noi toi canh muon nhat ho {ho_max:.3f} s > t_trong {t_trong:g} s'])
           + ([] if cho_ok else [f'cho {cho:.2f} s < {CHO_LOG_S:g} s sau dong log dau']) + list(extra_why))
    out.append(dict(base, loai='pre', t_ref=t_ref, t_max=t_max, t_canh_s=t_ref,
                    khung=[dict(id=int(i), pts_s=float(x)) for i, x in zip(ids[m], t[m])],
                    khung_sha256=ids_sha(ids[m]), khung_ma_hop_le=[int(i) for i in ids[mo]],
                    khung_ma_hop_le_sha256=ids_sha(ids[mo]), cong_ti_le='mot_phia' if gates is not None else 'hai_phia',
                    ket_luan=dict(du_mau=s_ok, dung_yen=d_ok, log_phu=l_ok, ho_noi_s=ho, ho_noi_max_s=ho_max,
                                  noi_phu=noi_ok, cho_log_s=cho, cho_du=cho_ok, dat=not why, ly_do=why,
                                  so_dung_yen=d_num, encoder_doi=enc)))
    # post: the last w_s of tag-gate frames, all after the latest possible end of the command
    t_end = max(t0 - D + u for D, u in zip(nodes, u_nodes)) + t_cmd_end
    tl = t[ok]
    if len(tl) and tl[-1] - kt['w_s'] >= t_end:
        b = float(tl[-1]) + dt
        a2 = b - kt['w_s']
        m2 = (t >= a2) & (t + dt <= b + 1e-9)
        mo2 = m2 & ok
        s2, w2 = _samples(t[mo2], a2, b, dt, kt, t_trong)
        d2, n2 = _stillness(t[mo2], tx[mo2], ty[mo2], tp[mo2], kt)
        why2 = w2 + ([] if d2 else ['xe khong dung yen'])
        out.append(dict(base, loai='post', t_canh_s=t_end,
                        khung=[dict(id=int(i), pts_s=float(x)) for i, x in zip(ids[m2], t[m2])],
                        khung_sha256=ids_sha(ids[m2]), khung_ma_hop_le=[int(i) for i in ids[mo2]],
                        khung_ma_hop_le_sha256=ids_sha(ids[mo2]),
                        ket_luan=dict(du_mau=s2, dung_yen=d2, dat=not why2, ly_do=why2, so_dung_yen=n2)))
    else:
        out.append(dict(base, loai='post', t_canh_s=t_end, khung=None, khung_ma_hop_le=None,
                        ket_luan=dict(dat=False, ly_do=['clip khong co du w_s khung sau khi het lenh'])))
    return out


# ---------------------------------------------------------------- reference (F12)
def tham_chieu(pre, chain, lan_lap, kt, t_trong, dt, extra=None):
    """F12 tham_chieu_v1 from the 'pre' row, the chain rows and lan_lap_v1 (or None). trang_thai: 'dat' (usable),
    'hong_cua_so_tc' (run lost to the reference window), 'chua_ket_luan' (no pose: missing q or A6 on the set)."""
    by = {c['frame_id']: c for c in chain}
    ma = [int(i) for i in (pre.get('khung_ma_hop_le') or [])]
    why, tap = [], 'giao'
    neo = None if lan_lap is None else ((lan_lap.get('khung_chung') or {}).get('pre'))
    if neo is None:
        # Codex CT1-3: a MISSING anchor list is not a confirmed loss of every anchor (mat_neo needs an empty list):
        # no set of J v6 2 can be formed, the reference is not concluded (G1-G5)
        ids, tap = ma, 'thieu_neo'
        why.append('thieu_lan_lap' if lan_lap is None else 'lan_lap_thieu_khung_chung_pre')
    else:
        s = set(int(i) for i in neo)
        ids = [i for i in ma if i in s]
        if not ids:
            ids, tap = ma, 'ma_mat_neo'
            why.append('mat_neo')
    trang = 'dat'
    if not pre['ket_luan']['dat']:
        trang = 'hong_cua_so_tc'
        why += ['pre: ' + w for w in pre['ket_luan']['ly_do']]
    tt = np.array([by[i]['t_cam'] for i in ids], float)
    s_ok, s_why = _samples(tt, pre['t_ref'] - kt['w_s'], pre['t_ref'], dt, kt, t_trong) if len(ids) else (False, ['rong'])
    if not s_ok and trang == 'dat':
        trang = 'hong_cua_so_tc'                       # J 2 (J5-1): an intersection with samples, not enough
        why += ['tap tham chieu: ' + w for w in s_why]
    rows = [by[i] for i in ids]
    p_ok = bool(rows) and all(r['q_ok'] for r in rows)
    y_ok = bool(rows) and all(r['a6_ok'] for r in rows)
    p_tc = psi_tc = None
    lp = lpsi = tan = None
    if p_ok:
        P = np.array([[r['p_x'], r['p_y']] for r in rows])
        p_tc = P.mean(axis=0).tolist()
        lp = float(np.max(np.hypot(P[:, 0] - p_tc[0], P[:, 1] - p_tc[1])))
    elif rows:
        why.append('vi_tri_qua_q_khong_xac_dinh: ' + (next((r['ly_do'] for r in rows if not r['q_ok']), '')))
    if y_ok:
        ps = np.array([r['psi_ma_p'] for r in rows])
        psi_tc = ck.trung_binh_go_vong(ps)
        dv = ck.wrap(ps - psi_tc)
        lpsi, tan = float(np.max(np.abs(dv))), float(np.max(dv) - np.min(dv))
    elif rows:
        why.append('huong_sau_A6_khong_xac_dinh')
    if y_ok and trang == 'dat' and math.degrees(tan) > kt['psi_dung_deg'] + 1e-12:
        # J v6 2 (Codex CT1-12): the heading spread gate on the set actually used (psi' after A6)
        trang = 'hong_cua_so_tc'
        why.append(f'tan huong sau A6 tren tap tham chieu {math.degrees(tan):.3f} deg > {kt["psi_dung_deg"]:g} deg')
    if trang == 'dat' and (tap == 'thieu_neo' or not (p_ok or y_ok)):
        trang = 'chua_ket_luan'
    return dict(schema=THAM_CHIEU_SCHEMA, phien_ban=PHIEN_BAN, chuoi_khung=ck.PHIEN_BAN, phep_gop='trung_binh_go_vong',
                tap=tap, mat_neo=tap == 'ma_mat_neo', ids=ids, ids_sha256=ids_sha(ids), n=len(ids),
                t_tc_s=float(tt.mean()) if len(tt) else None, p_tc_m=p_tc, psi_tc_rad=psi_tc, psi_tan_rad=tan,
                p_lech_max_m=lp, psi_lech_max_rad=lpsi, p_ok=p_ok, psi_ok=y_ok, trang_thai=trang, ly_do=why,
                lan_lap_v1_sha256=None if lan_lap is None else lan_lap.get('_sha256'), **(extra or {}))


# ---------------------------------------------------------------- phi (J 2, beta path)
DINH_DANH_12B = ('frames_sha256', 'clip_sha256', 'session_sha256')    # keys that tie truc_xe_v1 to the 12b of a run


def _truc_ids(truc_xe):
    """(12b run id, {identity key: sha}, lan_lap_id) of truc_xe_v1 (HIEUCHUAN c8_hc_a: mien_hieu_luc.thu_id_cap; the
    frames sha of that run from nguon '<thu_id>:frames'; clip / session / frames sha when the file has them)."""
    m = truc_xe.get('mien_hieu_luc') if isinstance(truc_xe.get('mien_hieu_luc'), dict) else {}
    thu = m.get('thu_id_cap', truc_xe.get('thu_id_cap'))
    sh = {k: m.get(k, truc_xe.get(k)) for k in DINH_DANH_12B}
    if not sh['frames_sha256'] and thu is not None:
        for n in truc_xe.get('nguon') or []:
            if isinstance(n, dict) and n.get('vai_tro') == f'{thu}:frames' and n.get('sha256'):
                sh['frames_sha256'] = n['sha256']
    return thu, {k: v for k, v in sh.items() if v}, m.get('lan_lap_id')


def rang_buoc_12b(truc_xe, lan_lap, k):
    """Reasons (empty: bound) why truc_xe_v1 is not the 12b of this run's placement (J v6 2: the latest 12b of the
    same session and the same placement; CODE 19/11 after the round-1 review: NHACTAM's lan_lap_v1 names it in
    chuan_12b (12b run id) with lan_lap_id; the session is checked on every identity key present on BOTH sides, the
    12b end of lan_lap_v1 dau.chuan against truc_xe_v1; no common key: not bound)."""
    why = []
    lid, ch = lan_lap.get('lan_lap_id'), lan_lap.get('chuan_12b')
    if lid is None:
        why.append('lan_lap_v1 thieu lan_lap_id' + (f" ({lan_lap['lan_lap_id_ly_do']})"
                                                     if lan_lap.get('lan_lap_id_ly_do') else ''))
    if ch is None:
        why.append('lan_lap_v1 thieu chuan_12b (khong co 12b cung lan lap o cho nay)')
    if lan_lap.get('cho') not in (None, k):
        raise ValueError(f'lan_lap_v1 of place {lan_lap.get("cho")!r}, the task needs {k!r}')
    thu, sh, t_lid = _truc_ids(truc_xe)
    if ch is not None and thu != ch:
        why.append(f'truc_xe_v1 cua 12b {thu!r} khac chuan_12b {ch!r} cua luot')
    if t_lid is not None and lid is not None and t_lid != lid:
        why.append(f'truc_xe_v1 lan_lap_id {t_lid!r} khac {lid!r}')
    dau = (lan_lap.get('dau') or {}).get('chuan') if isinstance(lan_lap.get('dau'), dict) else None
    dau = dau if isinstance(dau, dict) else {}
    common = [x for x in DINH_DANH_12B if sh.get(x) and dau.get(x)]
    if not common:
        why.append('chua buoc cung buoi: truc_xe_v1 va lan_lap_v1 dau.chuan khong co dinh danh chung '
                   f'({", ".join(DINH_DANH_12B)})')
    for x in common:
        if sh[x] != dau[x]:
            why.append(f'truc_xe_v1 {x} khac lan_lap_v1 dau.chuan {x} (12b khac buoi hay khac clip)')
    return why


def phi_beta(task, truc_xe, lan_lap, tc, u_beta_cam_deg=None):
    """(phi [rad] or None, u_phi [rad] (inf when a part is missing), ok, reasons). Same session and same placement
    only (rang_buoc_12b); no Delta-a path."""
    why = []
    k = cho_of(task)
    if truc_xe is None:
        return None, math.inf, False, ['thieu_truc_xe_v1']
    if truc_xe.get('schema') != TRUC_XE_SCHEMA:
        raise ValueError(f'truc_xe: schema {truc_xe.get("schema")!r}, expected {TRUC_XE_SCHEMA}')
    place = truc_xe.get('cho') or (truc_xe.get('mien_hieu_luc') or {}).get('cho')
    if place != k:
        raise ValueError(f'truc_xe_v1 of place {place!r}, the task {task} needs {k!r} (J 2; design v4 1.5)')
    if truc_xe.get('trang_thai') != 'khoa':
        why.append(f'truc_xe_v1 trang_thai {truc_xe.get("trang_thai")!r} (khong khoa)')
    ph, uph = truc_xe.get('phi_hat_chuan_k_deg'), truc_xe.get('u_phi_hat_chuan_k_deg')
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (ph, uph)):
        why.append('truc_xe_v1 thieu phi_hat_chuan_k_deg / u_phi_hat_chuan_k_deg')
        ph = uph = None
    if lan_lap is None:
        return None, math.inf, False, why + ['thieu_lan_lap_v1 (beta)']
    why += rang_buoc_12b(truc_xe, lan_lap, k)
    tt = lan_lap.get('trang_thai')
    if not (isinstance(tt, dict) and tt.get('beta') == 'dat'):
        why.append(f'beta khong hop le (trang_thai.beta {tt.get("beta") if isinstance(tt, dict) else tt!r})')
    b, ub = lan_lap.get('beta_deg'), lan_lap.get('u_beta_deg')
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (b, ub)):
        why.append('lan_lap_v1 thieu beta_deg / u_beta_deg')
        b = ub = None
    if tc['tap'] != 'giao':
        why.append('mat_neo: beta khong do duoc tren neo (J 2)')
    if why or ph is None or b is None:
        return None, math.inf, False, why
    phi = math.radians(ph + b)
    if u_beta_cam_deg is None:
        return phi, math.inf, True, ['thieu u_beta_cam (C8B): u_phi khong huu han']
    return phi, math.radians(LE * (uph + ub) + float(u_beta_cam_deg)), True, []


# ---------------------------------------------------------------- change of frame
def to_lane(px, py, psi_p, p_tc, psi_tc, phi, psi_dd):
    """(x', y', psi') in the nominal lane frame (module docstring). phi None (t_som case): positions with phi = 0,
    only their magnitudes of change (speeds) may be used."""
    ph = 0.0 if phi is None else phi
    c, s = math.cos(-ph), math.sin(-ph)
    dx, dy = np.asarray(px, float) - p_tc[0], np.asarray(py, float) - p_tc[1]
    bx, by = c * dx - s * dy, s * dx + c * dy
    cd, sd = math.cos(psi_dd), math.sin(psi_dd)
    x, y = cd * bx - sd * by, sd * bx + cd * by
    return (x, y) + tuple(rel_heading(psi_p, psi_tc, psi_dd)) + (bx, by)


def rel_heading(psi_p, psi_tc, psi_dd):
    """(psi' = wrap(psi_dd + dpsi), dpsi): dpsi = change of the tag heading from psi_tc on ONE continuous branch over
    the finite headings in time order (a NaN heading, A6 not usable, is skipped and stays NaN; the branch across it is
    the wrapped difference of its two sides), starting from wrap(first - psi_tc)."""
    p = np.asarray(psi_p, float)
    dpsi = np.full(len(p), np.nan)
    f = np.nonzero(np.isfinite(p))[0]
    if len(f) and psi_tc is not None and math.isfinite(psi_tc):
        u = np.unwrap(p[f])
        dpsi[f] = ck.wrap(p[f[0]] - psi_tc) + (u - u[0])
    return ck.wrap(psi_dd + dpsi), dpsi


def t_som(t, speed_ok, speed, wz_ok, wz, v_d, w_d, t_d, t_trong, t_cmd, t_deadline, present=None):
    """Earliest frame time t, t_cmd <= t < t_deadline, of the frames of one way of using the frames (present; None:
    every frame), whose window [t - t_d, t] does not rule completion out (J 5): no frame with a DEFINED speed >= v_d,
    no frame (of any way: wz comes from the tag heading, J4-10) with a USABLE |wz| >= w_d, no hole above t_trong
    among the frames of that way (Codex CT1-5: way 2 treats the position-suspect frames as holes, like G1), a frame of
    that way at the window start; t_deadline when there is none."""
    t = np.asarray(t, float)
    pres = np.ones(len(t), bool) if present is None else np.asarray(present, bool)
    bad = (speed_ok & (np.nan_to_num(speed, nan=np.inf) >= v_d)) | (wz_ok & (np.abs(np.nan_to_num(wz)) >= w_d))
    for k in np.nonzero(pres & (t >= t_cmd - 1e-9) & (t < t_deadline - 1e-12))[0]:
        lo = t[k] - t_d
        inside = (t >= lo - 1e-9) & (t <= t[k] + 1e-12)
        j = np.nonzero(inside & pres)[0]
        if not len(j) or t[j[0]] > lo + t_trong + 1e-12:
            continue
        seg = t[j]
        if len(seg) > 1 and float(np.max(np.diff(seg))) > t_trong + 1e-12:
            continue
        if bad[inside].any():
            continue
        return float(t[k])
    return float(t_deadline)


def wz_usable(yaw_ok, half):
    """J5-4: the rate of a frame is usable only when every yaw sample of its derivative stencil (+-half frames of
    c7_measure.sg_derivative, over the same frame list) passed the yaw check."""
    yaw_ok = np.asarray(yaw_ok, bool)
    n = len(yaw_ok)
    out = np.zeros(n, bool)
    for k in range(n):
        a, b = max(0, k - half), min(n, k + half + 1)
        out[k] = bool(b - a >= 5 and yaw_ok[a:b].all())
    return out


# ---------------------------------------------------------------- files
def write_jsonl(path, rows):
    with open(path, 'x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
