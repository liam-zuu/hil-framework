#!/usr/bin/env python3
"""
c8b_mat_na_d2.py: the D2 cell mask applied to the node results of c8b_luoi (design C8B v7 section 7c line 166:
"o vuot hay o n_id < 10, n_thieu > 0 -> moi khung cua o: n_undef cho G4, G5 va moi dai luong vi tri qua q (G1-G4)";
CODE 24/11 khuya: the mask is applied at the grid combination, c8b_luoi --gop-only --mat-na-d2, c7_score unchanged, no
scoring again; design note side-session-notes 2026-11-24_C8B_thiet_ke_D2_hang.md section 5). Written by CODE_C8B
local_a56ba287 on fix/c8b-n1-khoa-gop, 24/11/2026.

Per scored node, from its rows file (c7_score --rows: frame_id, mien_chung, mien_hop, lo_g2..hi_g5 and the measured
values e_perp, e_trk, e_bao, e_theta_cmd_deg):
  1. Self-check (required, design note 5): the per-frame q, lo, hi rebuilt from the rows, aggregated as the scorer
     does (o1_gate.aggregate on the common and union domains), must give the node's gate_v3 numbers (q, lo, hi,
     n_undef of both domains, n_frames; q_so_do when the block has it). Not equal -> refused (the mask would not be
     applied on the scorer's own numbers). q per frame as in c7_score.evaluate: G2 e_perp, G3 e_trk, G4 e_bao,
     G5 |e_theta_cmd_deg| (the t_som branch with the IMU heading gives another G5 q: the self-check refuses it).
  2. A frame of the run whose (run, frame_id) is not in the mask's khung_mo (cell not in o_dat, outside the u bands,
     not in the D2 rows at all, or no frame_id) is masked: lo = -inf, hi = +inf for G2-G5 (q kept, as the scorer keeps
     the q of an undefined frame as a diagnostic). A finite violation of another frame stays (design 7c).
  3. The node numbers and its three-branch states are recomputed (cond_state2 on the new numbers); q_so_do (defined
     frames only, SD1-1) recomputed when the node has it.
  4. G1 [CACH HIEU C8B, in the list for CODE/Liam]: the scorer turns an undefined frame into a gap of the tracker; the
     gap rules cannot be rerun here, so a node with a frame newly undefined in its union domain gets G1 chua_ket_luan
     (the safe side: never dat, never khong_dat on a masked frame).
  5. cat_t1 node (violation before t1, the domain is the part before t1): the certain violation must stand on the
     unmasked frames; when the mask removes it, the node is not scored (kind loi: the scorer's other branch would
     apply, it cannot be rerun here).
Library: doc_mat_na(path, luot), ap_dung(nodes, out_dir, mn, state_fn) -> (nodes, summary).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402
import c8b_a6 as a6m                                                     # noqa: E402
import o1_gate                                                           # noqa: E402

VERSION = 'c8b_mat_na_d2_v1'
MEASURED = ('g2', 'g3', 'g4', 'g5')
Q_COL = {'g2': 'e_perp', 'g3': 'e_trk', 'g4': 'e_bao', 'g5': 'e_theta_cmd_deg'}     # c7_score.evaluate q per frame
REL_TOL = 1e-9


class MatNaError(ValueError):
    pass


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def doc_mat_na(path, luot):
    """The D2 mask of a c8b_a6_o d2 file for run `luot`: {file, sha256, sha_noi_dung, luot, mo (frame ids)}."""
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if d.get('schema') != a6m.SCHEMA_O or d.get('muc_dich') != 'd2':
        raise MatNaError(f'{path}: not a {a6m.SCHEMA_O} file of muc_dich d2')
    try:
        a6m._check_sha(d, path)
    except a6m.A6Error as exc:
        raise MatNaError(str(exc))
    if d.get('trang_thai') != 'dat':
        raise MatNaError(f'{path}: trang_thai {d.get("trang_thai")!r} (no cell built)')
    mn = d.get('mat_na')
    if not isinstance(mn, dict) or not isinstance(mn.get('khung_mo'), list):
        raise MatNaError(f'{path}: mat_na without khung_mo (c8b_a6_o before 24/11 khuya)')
    if not luot or luot not in ((d.get('tap') or {}).get('dich') or []):
        raise MatNaError(f'{path}: run {luot!r} is not in the target set (tap.dich) of the mask')
    mo = set()
    for e in mn['khung_mo']:
        if not (isinstance(e, list) and len(e) == 2 and isinstance(e[1], int) and not isinstance(e[1], bool)):
            raise MatNaError(f'{path}: khung_mo entry {e!r}')
        if e[0] == luot:
            mo.add(e[1])
    return dict(file=os.path.abspath(path), sha256=cu.sha256(path), sha_noi_dung=d['sha256'], luot=luot, mo=mo,
                n_o_dat=len(mn.get('o_dat') or []), n_o_thieu=len(mn.get('o_thieu') or []))


def doc_hang(path):
    """Per-frame arrays of a node rows file."""
    with open(path, newline='', encoding='utf-8') as f:
        rr = list(csv.DictReader(f))
    need = ('t_cam_s', 'frame_id', 'mien_chung', 'mien_hop') + tuple(f'{b}_{c}' for c in MEASURED
                                                                     for b in ('lo', 'hi'))
    if rr and any(k not in rr[0] for k in need):
        raise MatNaError(f'{path}: rows without {[k for k in need if k not in rr[0]]}')
    fid = []
    for r in rr:
        v = _f(r['frame_id'])
        fid.append(int(v) if math.isfinite(v) and v == int(v) else None)
    H = dict(fid=fid, t_cam=np.array([_f(r['t_cam_s']) for r in rr]),
             mc=np.array([r['mien_chung'] == '1' for r in rr], bool),
             mu=np.array([r['mien_hop'] == '1' for r in rr], bool),
             lo={c: np.array([_f(r[f'lo_{c}']) for r in rr]) for c in MEASURED},
             hi={c: np.array([_f(r[f'hi_{c}']) for r in rr]) for c in MEASURED},
             q={c: np.array([_f(r.get(Q_COL[c])) for r in rr]) for c in MEASURED})
    H['q']['g5'] = np.abs(H['q']['g5'])
    return H


def _top_defined(q, lo, hi, m):
    ok = m & np.isfinite(q) & np.isfinite(lo) & np.isfinite(hi)
    return float(q[ok].max()) if ok.any() else math.nan


def so_lieu(H):
    """The node numbers of G2-G5 as c8b_luoi.node_from_files keeps them (plus q_sd, q_sd_chung: SD1-1)."""
    out = {}
    for c in MEASURED:
        thr = o1_gate.THRESHOLDS[c]
        q, lo, hi = H['q'][c], H['lo'][c], H['hi'][c]
        au = o1_gate.aggregate(q[H['mu']], lo[H['mu']], hi[H['mu']], thr)
        ac = o1_gate.aggregate(q[H['mc']], lo[H['mc']], hi[H['mc']], thr)
        out[c] = dict(q=au['q'], lo=au['lo'], hi=au['hi'], n_undef=au['n_undef'], n_frames=au['n_frames'],
                      q_chung=ac['q'], lo_chung=ac['lo'], hi_chung=ac['hi'], n_undef_chung=ac['n_undef'],
                      q_sd=_top_defined(q, lo, hi, H['mu']), q_sd_chung=_top_defined(q, lo, hi, H['mc']))
    return out


def _same(a, b):
    if isinstance(a, int) and not isinstance(a, bool) or isinstance(b, int) and not isinstance(b, bool):
        return a == b
    a, b = _f(a), _f(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def tu_kiem(node, H, k):
    """Step 1: the rebuilt numbers equal the node's gate numbers, else MatNaError."""
    got = so_lieu(H)
    bad = []
    for c in MEASURED:
        want = node['nums'][c]
        keys = ['q', 'lo', 'hi', 'n_undef', 'q_chung', 'lo_chung', 'hi_chung', 'n_undef_chung', 'n_frames']
        keys += [x for x in ('q_sd', 'q_sd_chung') if x in want]
        for x in keys:
            if not _same(got[c][x], want.get(x)):
                bad.append(f'{c}.{x} {got[c][x]!r} vs gate {want.get(x)!r}')
    if bad:
        raise MatNaError(f'node {k}: the rows do not rebuild the gate numbers (self-check, mask not applied): '
                         + '; '.join(bad[:6]))
    return got


def _certain_before(H, so_do):
    """cat_t1 check on the domain before t1 (mien_chung on that branch): a finite lo above the threshold, or for a
    so_do condition q above it on a defined frame (c7_score.certain_violations, SD1-1)."""
    for c in MEASURED:
        thr = o1_gate.THRESHOLDS[c]
        m = H['mc']
        if c in so_do:
            v = np.where(np.isfinite(H['lo'][c]) & np.isfinite(H['hi'][c]), H['q'][c], np.nan)[m]
        else:
            v = H['lo'][c][m]
        if np.any(np.isfinite(v) & (v > thr)):
            return True
    return False


def ap_dung(nodes, out_dir, mn, state_fn, key_dec=6):
    """Steps 1-5 on every scored node (list from c8b_luoi.load_nodes, changed in place); returns (nodes, summary)."""
    tom = []
    for k, x in enumerate(nodes):
        if not x.get('ok'):
            tom.append(dict(nut=k, cham=False))
            continue
        H = doc_hang(os.path.join(out_dir, f'nut_{k:03d}_rows.csv'))
        tu_kiem(x, H, k)
        che = np.array([f is None or f not in mn['mo'] for f in H['fid']], bool)
        dinh = np.zeros(len(che), bool)
        for c in MEASURED:
            dinh |= np.isfinite(H['lo'][c]) & np.isfinite(H['hi'][c])
        moi = che & dinh                                 # frames newly undefined by the mask
        for c in MEASURED:
            H['lo'][c] = np.where(che, -math.inf, H['lo'][c])
            H['hi'][c] = np.where(che, math.inf, H['hi'][c])
        truoc = dict(x['states'])
        e = dict(nut=k, cham=True, n_khung=len(che), n_che=int(che.sum()), n_che_moi=int(moi.sum()),
                 n_che_moi_mien_hop=int((moi & H['mu']).sum()), trang_thai_truoc=truoc)
        if x.get('cat') and np.any(moi & H['mc']) and not _certain_before(H, set(x.get('so_do') or [])):
            why = 'mat_na_d2: the violation before t1 (cat_t1) stands only on masked frames: score the node again'
            nodes[k] = dict(x, ok=False, kind='loi', why=why)
            tom.append(dict(e, trang_thai_sau=None, ly_do=why))
            continue
        nums = so_lieu(H)
        st = dict(truoc)
        for c in MEASURED:
            nm = x['nums'][c]
            nm.update({kk: v for kk, v in nums[c].items() if kk in nm or kk not in ('q_sd', 'q_sd_chung')})
            st[c] = state_fn(nm['lo_chung'], nm['lo'], nm['hi'], nm['n_undef'], o1_gate.THRESHOLDS[c])
        if np.any(moi & H['mu']):
            st['g1'] = 'chua_ket_luan'                   # step 4 [CACH HIEU]
        x['states'] = st
        for i, t in enumerate(H['t_cam']):
            r = x['rows'].get(round(float(t), key_dec))
            if r is None:
                raise MatNaError(f'node {k}: rows frame t_cam_s {t!r} not in the node rows (a bug)')
            for c in MEASURED:
                r[c] = float(H['lo'][c][i])
        tom.append(dict(e, trang_thai_sau=st, g1_che=bool(np.any(moi & H['mu']))))
    return nodes, dict(version=VERSION, file=mn['file'], sha256=mn['sha256'], sha_noi_dung=mn['sha_noi_dung'],
                       luot=mn['luot'], n_khung_mo=len(mn['mo']), n_o_dat=mn['n_o_dat'], n_o_thieu=mn['n_o_thieu'],
                       nut=tom,
                       quy_tac='khung (luot, frame_id) khong trong khung_mo -> lo -inf, hi +inf G2-G5 (q giu); so lieu '
                               'nut va trang thai ba nhanh tinh lai; G1 chua_ket_luan khi nut co khung moi khong xac '
                               'dinh trong mien hop [CACH HIEU]; cat_t1 mat vi pham chac chan -> nut khong cham')
