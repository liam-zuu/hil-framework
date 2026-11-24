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


def _check_sha(ids, sh, what):
    """Codex CT1-3, CT2-1: the sha written beside an ID set must hash exactly the IDs read; one hash form, ids_sha
    ('1,2'; CODE 19/11, NHACTAM sha_ids since hil e7b8428)."""
    if sh is not None and sh != ids_sha(ids):
        raise ValueError(f'lan_lap_v1 {what}: the sha256 is not ids_sha of its ids (contract F8)')


def _id_pts(v, what):
    """(ids, pts) of an F8 frame set {id: [...], pts_s: [...]}; ValueError when malformed."""
    ids, pts = v.get('id'), v.get('pts_s')
    if not isinstance(ids, list) or not isinstance(pts, list) or len(ids) != len(pts):
        raise ValueError(f'lan_lap_v1 {what}: id and pts_s lists of the same length (contract F8)')
    try:
        ids = [int(i) for i in ids]
        pts = [float(p) for p in pts]
    except (TypeError, ValueError):
        raise ValueError(f'lan_lap_v1 {what}: integer ids, numeric pts_s (contract F8)')
    if not all(math.isfinite(p) for p in pts):
        raise ValueError(f'lan_lap_v1 {what}: pts_s not finite')
    return ids, pts


def neo_of(lan_lap, end='pre'):
    """The anchor frames of one end of lan_lap_v1, normalised: dict(ids, pts, sha, nguon, mat_neo) or None when the
    end is missing. ValueError on a malformed set or a sha that does not hash its ids.
      nguon 'giao_ma_neo' (Codex CT3-1): dau[end].khung_giao_ma_neo {id, pts_s} + khung_giao_sha256, the set of J v6 2
            (CAM tag gate AND body anchors, c7_nhac_tam.tom_cua_so, hil e7b8428); dau[end].mat_neo must say whether
            it is empty (an empty set with mat_neo not true, or a non-empty one with mat_neo true: ValueError);
      nguon 'khung_chung' (Codex CT2-1; when the run has no khung_giao_ma_neo): khung_chung[end] {id, pts_s} + its sha
            (khung_chung_sha256[end] or dau[end].khung_chung_sha256); NHACTAM leaves it empty with a null sha when its
            own gate on that set failed, so an empty set with a null sha is NOT a loss of every anchor (mat_neo None);
      nguon 'danh_sach': a bare ID list (earlier synthetic form, no PTS)."""
    if lan_lap is None:
        return None
    d = (lan_lap.get('dau') or {}).get(end) if isinstance(lan_lap.get('dau'), dict) else None
    if isinstance(d, dict) and d.get('khung_giao_ma_neo') is not None:
        g = d['khung_giao_ma_neo']
        if not isinstance(g, dict):
            raise ValueError(f'lan_lap_v1 dau.{end}.khung_giao_ma_neo: {{id, pts_s}} expected (contract F8)')
        ids, pts = _id_pts(g, f'dau.{end}.khung_giao_ma_neo')
        sh = d.get('khung_giao_sha256')
        if sh is None:
            raise ValueError(f'lan_lap_v1 dau.{end}.khung_giao_sha256 missing (contract F8)')
        _check_sha(ids, sh, f'dau.{end}.khung_giao_sha256')
        mn = d.get('mat_neo')
        if (mn is True) != (not ids):
            raise ValueError(f'lan_lap_v1 dau.{end}: mat_neo {mn!r} with {len(ids)} intersection frames (contract F8)')
        return dict(ids=ids, pts=pts, sha=sh, nguon='giao_ma_neo', mat_neo=not ids)
    kc = lan_lap.get('khung_chung')
    v = kc.get(end) if isinstance(kc, dict) else None
    if v is None:
        return None
    top = lan_lap.get('khung_chung_sha256')
    if isinstance(top, dict) and end in top:
        sh = top[end]
        if isinstance(d, dict) and 'khung_chung_sha256' in d and d['khung_chung_sha256'] != sh:
            raise ValueError(f'lan_lap_v1 khung_chung_sha256.{end} differs from dau.{end}.khung_chung_sha256')
    else:
        sh = d.get('khung_chung_sha256') if isinstance(d, dict) else None
    if isinstance(v, dict):
        ids, pts = _id_pts(v, f'khung_chung.{end}')
        _check_sha(ids, sh, f'khung_chung_sha256.{end}')
        return dict(ids=ids, pts=pts, sha=sh, nguon='khung_chung', mat_neo=None if sh is None else not ids)
    if isinstance(v, list):
        ids = [int(i) for i in v]
        _check_sha(ids, sh, f'khung_chung_sha256.{end}')
        return dict(ids=ids, pts=None, sha=sh, nguon='danh_sach', mat_neo=not ids)
    raise ValueError(f'lan_lap_v1 khung_chung.{end}: {type(v).__name__}, expected {{id, pts_s}} (contract F8)')


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


def mot_phia(frames, chain, gates, t_ref, quant=5e-7):
    """Tag-gate frames of a reference window with the ONE-SIDED scale gate (contract F2; design v4 1.2; Codex CT1-14,
    round 2 'sua sai'): the eligible frames (every failure, if any, a two-sided (2b) verdict; car seen, finite tag and
    scale) give ONE reference scale_ref,tc = median scale over [t_ref - 4 s, t_ref) (the window and 1 s before it);
    an eligible frame passes when its scale is within scale_rel of it; rounded scales (quant, c7_measure.CSV_QUANT)
    must pass for every value in the interval. Fewer than scale_min eligible frames in that span: none passes."""
    g = gates
    t = np.array([float(r['t_s']) for r in frames], float)
    sc = np.array([ck._f(r.get('scale')) for r in frames], float)
    base = np.array([bool(r.get('car_seen')) and all(w in RATIO_WHY for w in (r.get('why') or []))
                     and math.isfinite(c['tag_x']) and math.isfinite(c['tag_y']) and math.isfinite(c['tag_psi'])
                     and math.isfinite(s_) for r, c, s_ in zip(frames, chain, sc)], bool)
    out = np.zeros(len(frames), bool)
    span = base & (t >= t_ref - W_S - 1.0 - 1e-12) & (t < t_ref)
    if int(span.sum()) < g['scale_min']:
        return out
    ref = float(np.median(sc[span]))
    r_lo = (sc - quant) / (ref + quant) - 1.0
    r_hi = (sc + quant) / (ref - quant) - 1.0
    out = base & (np.maximum(np.abs(r_lo), np.abs(r_hi)) <= g['scale_rel'])
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
    tx = np.array([c['tag_x'] for c in chain], float)
    ty = np.array([c['tag_y'] for c in chain], float)
    tp = np.array([c['tag_psi'] for c in chain], float)
    dt = _frame_dt(t)
    t0 = t0_us / 1e6
    nodes, u_nodes = [float(v) for v in nodes], [float(v) for v in u_nodes]
    t_ref = min(t0 - D - u for D, u in zip(nodes, u_nodes))             # camera clock, earliest possible edge
    ok1 = mot_phia(frames, chain, gates, t_ref) if gates is not None else ok
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
    # Codex CT1-12 (round 2): the car STILL for 6 s before the edge, not only 6 s of log: the still gate (tag frames
    # of the window gate) over [t_ref - 6 s, t_max] (6 s before the earliest edge to the latest), no hole above t_trong
    a6 = t_ref - CHO_LOG_S
    m6 = (t >= a6) & (t <= t_max) & ok1
    d6_ok, d6_num = _stillness(t[m6], tx[m6], ty[m6], tp[m6], kt)
    ho6 = _max_hole(t[ok1], dt, a6, t_max)
    d6_num['ho_max_s'] = ho6
    d6_ok = bool(d6_ok and ho6 <= t_trong + 1e-12)
    why = (s_why + ([] if d_ok else ['xe khong dung yen trong cua so hay toi canh muon nhat']) + l_why
           + ([] if noi_ok else [f'doan noi toi canh muon nhat ho {ho_max:.3f} s > t_trong {t_trong:g} s'])
           + ([] if cho_ok else [f'cho {cho:.2f} s < {CHO_LOG_S:g} s sau dong log dau'])
           + ([] if d6_ok else [f'xe khong dung yen du {CHO_LOG_S:g} s truoc canh (ho {ho6:.3f} s)'])
           + list(extra_why))
    out.append(dict(base, loai='pre', t_ref=t_ref, t_max=t_max, t_canh_s=t_ref,
                    khung=[dict(id=int(i), pts_s=float(x)) for i, x in zip(ids[m], t[m])],
                    khung_sha256=ids_sha(ids[m]), khung_ma_hop_le=[int(i) for i in ids[mo]],
                    khung_ma_hop_le_sha256=ids_sha(ids[mo]), cong_ti_le='mot_phia' if gates is not None else 'hai_phia',
                    ket_luan=dict(du_mau=s_ok, dung_yen=d_ok, log_phu=l_ok, ho_noi_s=ho, ho_noi_max_s=ho_max,
                                  noi_phu=noi_ok, cho_log_s=cho, cho_du=cho_ok, dung_yen_6s=d6_ok,
                                  so_dung_yen_6s=d6_num, dat=not why, ly_do=why,
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
        # CODE 19/11 after Codex round 4 (CT4-3): an empty list with the reason, never null (NHACTAM reads [] as a
        # failed window and still writes F8 for the run)
        out.append(dict(base, loai='post', t_canh_s=t_end, khung=[], khung_sha256=ids_sha([]), khung_ma_hop_le=[],
                        khung_ma_hop_le_sha256=ids_sha([]), ly_do=['clip khong co du w_s khung sau khi het lenh'],
                        ket_luan=dict(dat=False, ly_do=['clip khong co du w_s khung sau khi het lenh'])))
    return out


# ---------------------------------------------------------------- reference (F12)
TAP_MA_CACH_A = 'ma_cach_A'
W_NHAN_CACH_A = 'w_theo_ma_cach_A'


W_CANH_L_M = 0.24                 # the omega_d = v_d / 0.24 m of the G1 still rule (c7_score, J v6 G1)


def w_c8b(rows, p_tc, psi_tc, chain, pre, dt, v_d):
    """The four W numbers of the C8B lock T12 (C8B 23/11, local_0b12a551, design C8b v7 6a, 6b; before the margin):
      w_dao_dong_xy_m / _psi_rad: the W ids split in time into two adjacent halves, the mean of each (centre after q;
        heading trung_binh_go_vong): |p1 - p2|, |wrap(psi1 - psi2)| (repeatability of two adjacent windows, v4 6b;
        session 2 has one ~3 s W per run, so the two halves of that W: about 2x the error of the full mean, safe side).
      w_canh_xy_m / _psi_rad (v7 6a): the largest deviation from p_tc / psi'_tc over W plus the joint [t_ref, t_max]
        (chain frames with cong_ok, a6_ok, q_ok there) + the frame error (the largest deviation over W alone, still
        car: frame noise) + the capture interval (v_d dt / omega_d dt) + v_d ho_noi_max_s / omega_d ho_noi_max_s of the
        F2 pre row, omega_d = v_d / W_CANH_L_M; the (|q_hat| + dq) omega_d ho_noi_s part is added by C8B.
    Every part is also written (w_c8b). None when the position or heading of W is not defined, v_d is None, or W has
    fewer than two frames. Limit: the joint uses the two-sided frame gate of the chain (cong_ok), not the one-sided
    gate of F2 (mot_phia); the one-sided gate keeps more frames there."""
    out = dict(w_dao_dong_xy_m=None, w_dao_dong_psi_rad=None, w_canh_xy_m=None, w_canh_psi_rad=None)
    ho = (pre.get('ket_luan') or {}).get('ho_noi_max_s')
    if v_d is None or len(rows) < 2 or not (isinstance(ho, (int, float)) and math.isfinite(ho)):
        out['w_c8b'] = dict(ly_do='thieu v_d, ho_noi_max_s hay W < 2 khung')
        return out
    w_d = float(v_d) / W_CANH_L_M
    rs = sorted(rows, key=lambda r: r['t_cam'])
    h1, h2 = rs[:len(rs) // 2], rs[len(rs) // 2:]
    lo, hi = float(pre['t_ref']), float(pre['t_max'])
    w_ids = {r['frame_id'] for r in rows}
    noi = [c for c in chain if c['frame_id'] not in w_ids and lo <= c['t_cam'] <= hi and c['cong_ok']]
    part = dict(v_d=float(v_d), omega_d=w_d, L_m=W_CANH_L_M, dt_s=float(dt), ho_noi_max_s=float(ho),
                n_W=len(rows), n_nua=[len(h1), len(h2)], n_noi=len(noi), cong_noi='cong_ok (hai phia, chuoi)')
    if p_tc is not None:
        P1 = np.array([[r['p_x'], r['p_y']] for r in h1]).mean(axis=0)
        P2 = np.array([[r['p_x'], r['p_y']] for r in h2]).mean(axis=0)
        out['w_dao_dong_xy_m'] = float(np.hypot(*(P1 - P2)))
        PW = np.array([[r['p_x'], r['p_y']] for r in rows])
        lw = float(np.max(np.hypot(PW[:, 0] - p_tc[0], PW[:, 1] - p_tc[1])))
        nq = [c for c in noi if c['q_ok']]
        ln = max([lw] + [float(math.hypot(c['p_x'] - p_tc[0], c['p_y'] - p_tc[1])) for c in nq])
        out['w_canh_xy_m'] = ln + lw + float(v_d) * float(dt) + float(v_d) * float(ho)
        part.update(lech_max_m=ln, lech_max_W_m=lw, n_noi_q=len(nq))
    if psi_tc is not None:
        a1 = ck.trung_binh_go_vong(np.array([r['psi_ma_p'] for r in h1]))
        a2 = ck.trung_binh_go_vong(np.array([r['psi_ma_p'] for r in h2]))
        out['w_dao_dong_psi_rad'] = float(abs(ck.wrap(a1 - a2)))
        lw = float(np.max(np.abs(ck.wrap(np.array([r['psi_ma_p'] for r in rows]) - psi_tc))))
        na = [c for c in noi if c['a6_ok']]
        ln = max([lw] + [float(abs(ck.wrap(c['psi_ma_p'] - psi_tc))) for c in na])
        out['w_canh_psi_rad'] = ln + lw + w_d * float(dt) + w_d * float(ho)
        part.update(lech_max_rad=ln, lech_max_W_rad=lw, n_noi_a6=len(na))
    out['w_c8b'] = part
    return out


def tham_chieu(pre, chain, lan_lap, kt, t_trong, dt, extra=None, f8_rong_ok=False, v_d=None):
    """F12 tham_chieu_v1 from the 'pre' row, the chain rows and lan_lap_v1 (or None). trang_thai: 'dat' (usable),
    'hong_cua_so_tc' (run lost to the reference window), 'chua_ket_luan' (no pose: missing q or A6 on the set).
    f8_rong_ok (f8_rong(...)[0], CODE 23/11 chot T12 muc 4, goc 6812dcf6): W = the tag-gate frames of 'pre' (tap
    TAP_MA_CACH_A, w_nhan W_NHAN_CACH_A), not the tag-anchor intersection of J v6 2: J takes the intersection because
    beta is measured on the anchors and p_tc, psi(0) must share its set; cach A measures beta by the tag yaw, so the
    consistent set is the tag gate. The sample gate (n_min, frac_min, t_trong) and the spread gates are unchanged.
    v_d (constants file): the four W numbers of the C8B lock (w_c8b), None without it."""
    by = {c['frame_id']: c for c in chain}
    ma = [int(i) for i in (pre.get('khung_ma_hop_le') or [])]
    why, tap = [], 'giao'
    nb = None if f8_rong_ok else neo_of(lan_lap, 'pre')
    if f8_rong_ok:
        ids, tap = ma, TAP_MA_CACH_A
    elif nb is None:
        # Codex CT1-3: a MISSING anchor list is not a confirmed loss of every anchor (mat_neo needs an empty list):
        # no set of J v6 2 can be formed, the reference is not concluded (G1-G5)
        ids, tap = ma, 'thieu_neo'
        why.append('thieu_lan_lap' if lan_lap is None else 'lan_lap_thieu_khung_chung_pre')
    elif nb['mat_neo'] is None:
        # Codex CT2-1, CT3-1: an empty khung_chung with a null sha is NHACTAM's own gate failing, not a loss of every
        # anchor (that needs the intersection itself, khung_giao_ma_neo, empty): no set of J v6 2
        ids, tap = ma, 'thieu_neo'
        why.append('khung_chung.pre rong, sha null: neo chua xac nhan (NHACTAM), khong co khung_giao_ma_neo')
    else:
        s = set(nb['ids'])
        ids = [i for i in ma if i in s]
        if not ids:
            # J v6 2: only an EMPTY intersection (every anchor lost) falls back to the tag-gate frames, flag mat_neo;
            # an intersection with frames but too few is a window loss (sample gate below; Codex CT3-1)
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
                w_nhan=W_NHAN_CACH_A if tap == TAP_MA_CACH_A else None,
                **w_c8b(rows, p_tc, psi_tc, chain, pre, dt, v_d),
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


# F8 rong (Liam 23/11, phien CODE local_483c9ce2, hop chon "phi buoi 2", nguyen van: "Cach A (Recommended)"; CODE
# 03_CODE/docs/2026-11-23_Chot_T12_khoa_C8B_v1.md muc 4, goc 5a7d428b): an F8 that passed its identity checks but gives
# no lan_lap_id to any run (NHACTAM stopped the beta path for session 2) -> beta = 0, u_beta = the plate heading shift
# bound of session 1 (lift and put back, A2: 05_DATA/results/C8/2026-11-18_run01/C2/do_05b994c/c8_phan_nhac_tam.json
# goc_deg), plate unchanged in the run (one mounting assumed: Liam 23/11 at CODE "Chua thao, con nguyen"); labelled
# F8_RONG_NHAN. beta: CODE erratum 23/11 (goc 77991725), not 0: the body turn from the 12b window to W, = the tag turn
# (phi_beta). u_beta_cam is still read from the C8B lock (C8B computes it: camera bound of the tag-yaw difference).
F8_RONG_NHAN = ck.F8_RONG_NHAN
U_BETA_F8_RONG_DEG = 0.234
F8_RONG_GIA_DINH = ('mot lan lap tam (khong thao tam giua cac luot): beta = wrap(psi_tc - psi_chuan_k) (than xoay bang '
                    'ma xoay), u_beta = can lech tam buoi 1 (A2)')


REF_12B_CUA_SO = 'cua_so_v1'
REF_12B_TU_F7 = 'ref_12b_tu_F7_buoi_khac'


def ref_12b_nguon(co_ref_12b, truc_xe=None, a6_sha256=None, cho=None):
    """Where the 12b reference of the place comes from, for the identity check of f8_rong: REF_12B_CUA_SO when the
    window file has a ref_12b window of the place (co_ref_12b True, as before); REF_12B_TU_F7 when it has none but
    truc_xe_v1 (F7) is 'khoa' with leg-start windows dau_doan: in cach A every number of the 12b reference comes from
    that F7 (beta = psi'_tc - psi_chuan_k of F7; the run's own ref_12b window is not read), so a floor session
    without a 12b clip scores on the F7 of another session (Liam 24/11 tonight at CODE local_42161302, box choice,
    verbatim: "Sua 1 dieu kien (Recommended)"; plate "Chua thao, chua cham" from session 2 to run05; contract C8
    log 24/11). None otherwise (no 12b reference: not the F8 rong case).
    The F7 used instead of the window (Codex F8R-1, F8R-2, round 1 24/11): schema truc_xe_v1; its place (cho and
    mien_hieu_luc.cho, the same where both are written) = cho (the place of the F8 record) when given; its A6 source
    (nguon vai_tro c8b_a6_v2) = a6_sha256, the A6 the run is scored with; every dau_doan entry a leg-start window
    (thu_id, frames_sha256 of 64 hex, a non-empty ids list). The plate and mounting from the F7 session to the run
    are not checked by data here: the one-mounting assumption of cach A (F8_RONG_GIA_DINH), Liam's word for run05."""
    if co_ref_12b is True:
        return REF_12B_CUA_SO
    return REF_12B_TU_F7 if f7_thay_ref_12b(truc_xe, a6_sha256, cho) else None


def _hex64(v):
    return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)


def f7_thay_ref_12b(truc_xe, a6_sha256=None, cho=None):
    """True when truc_xe_v1 may stand for the missing ref_12b window (see ref_12b_nguon)."""
    if not isinstance(truc_xe, dict) or truc_xe.get('schema') != TRUC_XE_SCHEMA or truc_xe.get('trang_thai') != 'khoa':
        return False
    m = truc_xe.get('mien_hieu_luc') if isinstance(truc_xe.get('mien_hieu_luc'), dict) else {}
    places = {v for v in (truc_xe.get('cho'), m.get('cho')) if v is not None}
    if len(places) != 1 or (cho is not None and places != {cho}):
        return False
    a6s = [n.get('sha256') for n in truc_xe.get('nguon') or [] if isinstance(n, dict) and n.get('vai_tro') == 'c8b_a6_v2']
    if not a6_sha256 or a6s != [a6_sha256]:
        return False
    dd = truc_xe.get('dau_doan')
    return isinstance(dd, list) and bool(dd) and all(
        isinstance(e, dict) and isinstance(e.get('thu_id'), str) and e['thu_id'] and _hex64(e.get('frames_sha256'))
        and isinstance(e.get('ids'), list) and len(e['ids']) > 0 for e in dd)


def f8_rong(lan_lap, a6_sha256, co_ref_12b, truc_xe=None):
    """(True, []) when lan_lap (doc_lan_lap of THIS run, bound to the window file by c7_score.bind_lan_lap) is the
    F8 rong case of cach A, else (False, reasons). Same identity checks as the F5 gate HCC2-7 of CODE 23/11 (Chot F8
    rong muc 1b): schema (doc_lan_lap), tam_id given, a6.sha256 = the A6 used (a6_sha256), a 12b reference of this
    place (ref_12b_nguon: a ref_12b window in the window file, co_ref_12b, or, without one, a 'khoa' F7 with
    dau_doan, Liam 24/11), and no run of the file has a lan_lap_id (doc_lan_lap _moi_lan_lap_id_null). truc_xe_v1,
    when given, must not name a mounting either (its lan_lap_id null; its lan_lap_v1_sha256, when written, this F8
    file)."""
    if lan_lap is None:
        return False, ['thieu_lan_lap_v1']
    why = []
    if lan_lap.get('_moi_lan_lap_id_null') is not True or lan_lap.get('lan_lap_id') is not None:
        why.append('F8 co lan_lap_id (khong phai F8 rong)')
    if not (isinstance(lan_lap.get('tam_id'), str) and lan_lap['tam_id']):
        why.append('F8 thieu tam_id')
    a6 = lan_lap.get('a6') if isinstance(lan_lap.get('a6'), dict) else {}
    if not a6_sha256 or a6.get('sha256') != a6_sha256:
        why.append('F8 a6.sha256 khac A6 dang dung')
    if ref_12b_nguon(co_ref_12b, truc_xe, a6_sha256, lan_lap.get('cho')) is None:
        why.append('cua_so_v1 khong co cua so ref_12b cua cho nay')
    if truc_xe is not None:
        m = truc_xe.get('mien_hieu_luc') if isinstance(truc_xe.get('mien_hieu_luc'), dict) else {}
        # Codex F8R-3 (round 1 24/11): both levels read, a null in mien_hieu_luc does not hide the top level
        if any(d.get('lan_lap_id') is not None for d in (m, truc_xe)):
            why.append('truc_xe_v1 co lan_lap_id ma F8 khong co')
        if any(d.get('lan_lap_v1_sha256') is not None and d.get('lan_lap_v1_sha256') != lan_lap.get('_sha256')
               for d in (m, truc_xe)):
            why.append('truc_xe_v1 lan_lap_v1_sha256 khac F8 dang dung')
    return not why, why


def phi_beta(task, truc_xe, lan_lap, tc, u_beta_cam_deg=None, f8_rong_ok=False):
    """(phi [rad] or None, u_phi [rad] (inf when a part is missing), ok, reasons). Same session and same placement
    only (rang_buoc_12b); no Delta-a path. f8_rong_ok (f8_rong(...)[0] of the caller, cach A): beta = wrap(psi_tc_rad
    of tc - psi_chuan_k_deg of truc_xe_v1), u_beta = U_BETA_F8_RONG_DEG, no 12b binding through lan_lap_v1 (it names
    none) and no trang_thai.beta 'dat'; W must be the tag-gate set (tc tap TAP_MA_CACH_A, CODE 6812dcf6) instead of
    the intersection 'giao' of J 2; the truc_xe_v1 checks are kept."""
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
    if lan_lap.get('cho') not in (None, k):
        raise ValueError(f'lan_lap_v1 of place {lan_lap.get("cho")!r}, the task needs {k!r}')
    if f8_rong_ok:
        # cach A, CODE erratum 23/11 (chot T12 muc 4, goc 77991725; design lan lap v4 3: beta = wrap(alpha_dich -
        # alpha_chuan,k), the turn of the BODY from the 12b window to W): the plate does not turn on the car, so the body
        # turns as the tag: beta = wrap(psi'_tc of the run - psi_chuan_k of truc_xe_v1 (F7, same place, same session));
        # the run's beta_deg of NHACTAM is not used
        pc, pt = truc_xe.get('psi_chuan_k_deg'), tc.get('psi_tc_rad')
        if not (isinstance(pc, (int, float)) and math.isfinite(pc)):
            why.append('truc_xe_v1 thieu psi_chuan_k_deg (beta cach A)')
            b = None
        elif not (tc.get('psi_ok', True) and isinstance(pt, (int, float)) and math.isfinite(pt)):
            why.append('tham chieu thieu psi_tc_rad (beta cach A)')
            b = None
        else:
            b = math.degrees(ck.wrap(pt - math.radians(pc)))
        ub = U_BETA_F8_RONG_DEG
    else:
        why += rang_buoc_12b(truc_xe, lan_lap, k)
        tt = lan_lap.get('trang_thai')
        if not (isinstance(tt, dict) and tt.get('beta') == 'dat'):
            why.append(f'beta khong hop le (trang_thai.beta {tt.get("beta") if isinstance(tt, dict) else tt!r})')
        b, ub = lan_lap.get('beta_deg'), lan_lap.get('u_beta_deg')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (b, ub)):
            why.append('lan_lap_v1 thieu beta_deg / u_beta_deg')
            b = ub = None
    if f8_rong_ok:
        if tc['tap'] != TAP_MA_CACH_A:                 # cach A: W of the tag gate (tham_chieu f8_rong_ok)
            why.append(f"W khong theo cong ma (tap {tc['tap']!r}, cach A can {TAP_MA_CACH_A!r})")
    elif tc['tap'] != 'giao':
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
