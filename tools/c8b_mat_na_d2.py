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
     n_undef of both domains, n_frames; q_so_do when the block has it) AND, before any masking, the node's own
     three-branch states (Codex D2-6: numbers equal within the tolerance may lie on the two sides of a threshold).
     Not equal -> refused (the mask would not be applied on the scorer's own numbers). q per frame as in
     c7_score.evaluate: G2 e_perp, G3 e_trk, G4 e_bao, G5 |e_theta_cmd_deg|. The rows file must be the one the
     scorer wrote for this node (khung_tuong_doi_v1.rows_sha256 of the node score; D2-6).
     Refused, because the rows do not carry what the scorer decided with (the scorer would have to run again):
     a node of the plate-changed case (tam_doi 'doi': G5 q from the IMU heading, not in the rows; D2-7).
     A node of the t_som case (g1.ca_t_som: the end of its common domain comes from the frame speeds) with a
     masked frame (D2-5, D2v2-3) cannot be masked without the scorer either: the WHOLE run is then not concluded
     (L166, Liam 25/11 at CODE, box: "Chua ket luan (Recommended)"; before: refused): every frame of every
     scored node undefined for G2-G5, G1 chua_ket_luan at every node, no cat_t1 branch kept; the masked grid is
     written with ca_luot_chua_ket_luan in its mat_na_d2 block (real data: every node of a K run is t_som).
  2. A frame of the run whose (run, frame_id) is not in the mask's khung_mo (cell not in o_dat, outside the u bands,
     not in the D2 rows at all) is masked (rows with an empty or repeated frame_id are refused): lo = -inf, hi = +inf for G2-G5 (q kept, as the scorer keeps
     the q of an undefined frame as a diagnostic). A finite violation of another frame stays (design 7c).
  3. The node numbers and its three-branch states are recomputed (cond_state2 on the new numbers); q_so_do (defined
     frames only, SD1-1) recomputed when the node has it.
  4. G1 (L156, Liam 25/11 at CODE, box: "Co khung bi che thi G1 chua ket luan (Recommended)"): the scorer turns an undefined frame into a gap of the tracker; the
     gap rules cannot be rerun here, so a node with a MASKED frame in its union domain gets G1 chua_ket_luan (the
     safe side: never dat, never khong_dat on a masked frame). Any masked frame, not only one that had a finite
     bound (Codex D2-4: G1 follows the progress by the frame centres, which a frame can have with every G2-G5
     bound already infinite). Limits (in the assessment): G1 may be chua_ket_luan where a rerun would still
     conclude; the frame-dependent ends of the domains (t_done) stay those of the scoring without the mask.
  5. cat_t1 node (violation before t1, the domain is the part before t1): with a masked frame before t1 the certain
     violation must stand on the unmasked frames (the scorer's rule certain_violations: a finite lo above the
     threshold; a so_do condition: q above it on a frame with lo and hi finite, SD1-1); when the mask removes it,
     the node is not scored (kind loi: the scorer's other branch would apply, it cannot be rerun here). Checked
     for any masked frame before t1 (Codex D2-3). Limit: t1 itself is not found again.
  6. goc_luoi: the masked grid file is the grid of luoi.json plus the mask; the chain fields of luoi.json (clip,
     file_so_h, the H1 pair, clock and gate file SHA, task, the scorer commit) must be the same in the new file
     (Codex D2-1: flags not passed again would write null), and the mask may only take conclusions away: a
     condition or the run that is dat or khong_dat with the mask and was not so without it -> refused (the mask
     never makes a dat or a khong_dat; e.g. the middle node of a so_do condition replaced, D2-2).
  0. One scoring (Codex round 2 D2v2-4; CODE 25/11 03:07, no rule changed, a refusal only): the mask goes by
     (run, frame_id), so the mask built from one scoring of a run would fit the grid of another scoring of the same
     run (e.g. a scoring again on the H1 plates). The D2 rows carry cham_id (ma_lan_cham: the node score SHA of the
     scored nodes, in order), c8b_a6_o writes it per run in mat_na.lan_cham, and ap_dung computes it again from the
     node files it is about to mask: a mask without the id of the run, or with another id, is refused. No mask of
     its own for an H1 scoring is written: none of the 8 camera drift files of the four C8 sessions has vuot true.
Library: doc_mat_na(path, luot), ap_dung(nodes, out_dir, mn, state_fn) -> (nodes, summary), goc_luoi(path, res),
ma_lan_cham(pairs).
"""
from __future__ import annotations

import csv
import hashlib
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


def ma_lan_cham(cap):
    """The id of one scoring of a run (step 0): 16 hex of the sha256 of 'k:score_sha256;' over its scored nodes
    (node index, SHA of nut_KKK.json), in order."""
    return hashlib.sha256(''.join(f'{int(k)}:{s};' for k, s in cap).encode('ascii')).hexdigest()[:16]


def doc_mat_na(path, luot):
    """The D2 mask of a c8b_a6_o d2 file for run `luot`: {file, sha256, sha_noi_dung, luot, mo (frame ids),
    lan_cham (the scoring id of the run, None when the file has none)}."""
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
    lc = mn.get('lan_cham')
    if lc is not None and not (isinstance(lc, dict) and all(isinstance(v, str) and v for v in lc.values())):
        raise MatNaError(f'{path}: mat_na.lan_cham {lc!r}')
    return dict(file=os.path.abspath(path), sha256=cu.sha256(path), sha_noi_dung=d['sha256'], luot=luot, mo=mo,
                lan_cham=(lc or {}).get(luot),
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


def _co_q_so_do(node, c):
    """The node's gate block of condition c carries q_so_do (a so_do condition, SD1-1). c8b_luoi.node_from_files of
    GATE 2ae87ad gives q_sd, q_sd_chung to every condition, NaN where the block has none: those are not compared and
    not recomputed (found before the merge: the self-check would refuse every node)."""
    return isinstance(((node.get('gate') or {}).get(c) or {}).get('q_so_do'), dict)


def tu_kiem(node, H, k):
    """Step 1: the rebuilt numbers equal the node's gate numbers, else MatNaError."""
    got = so_lieu(H)
    bad = []
    for c in MEASURED:
        want = node['nums'][c]
        keys = ['q', 'lo', 'hi', 'n_undef', 'q_chung', 'lo_chung', 'hi_chung', 'n_undef_chung', 'n_frames']
        keys += [x for x in ('q_sd', 'q_sd_chung') if x in want and _co_q_so_do(node, c)]
        for x in keys:
            if not _same(got[c][x], want.get(x)):
                bad.append(f'{c}.{x} {got[c][x]!r} vs gate {want.get(x)!r}')
    if bad:
        raise MatNaError(f'node {k}: the rows do not rebuild the gate numbers (self-check, mask not applied): '
                         + '; '.join(bad[:6]))
    return got


def _diem_nut(score_path, rows_path, k):
    """khung_tuong_doi_v1 of the node score; the rows file must be the one the scorer wrote (rows_sha256)."""
    try:
        with open(score_path, encoding='utf-8') as f:
            sc = json.load(f)
    except (OSError, ValueError) as exc:
        raise MatNaError(f'node {k}: score file: {exc}')
    ktd = sc.get('khung_tuong_doi_v1') if isinstance(sc, dict) else None
    if not isinstance(ktd, dict) or not ktd.get('rows_sha256'):
        raise MatNaError(f'node {k}: the node score names no rows_sha256 (not a relative C8 scoring with --rows)')
    if ktd['rows_sha256'] != cu.sha256(rows_path):
        raise MatNaError(f'node {k}: {rows_path} is not the rows file the scorer wrote (rows_sha256)')
    return ktd


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
    """Steps 0-5 on every scored node (list from c8b_luoi.load_nodes, changed in place); returns (nodes, summary)."""
    # step 0 (D2v2-4): the mask is of THIS scoring of the run (the node score files about to be masked)
    ma = ma_lan_cham((k, cu.sha256(os.path.join(out_dir, f'nut_{k:03d}.json')))
                     for k, x in enumerate(nodes) if x.get('ok'))
    if mn.get('lan_cham') != ma:
        raise MatNaError(f'the mask is not of this scoring of run {mn.get("luot")!r}: mat_na.lan_cham '
                         f'{mn.get("lan_cham")!r}, the node files here give {ma!r} (D2 rows of another scoring, e.g. '
                         'H0 / H1, or written before cham_id): build the D2 rows and the cell file from these nodes')
    # pass 1: every scored node is checked and its masked frames found BEFORE any node is changed
    pre = {}
    for k, x in enumerate(nodes):
        if not x.get('ok'):
            continue
        rp = os.path.join(out_dir, f'nut_{k:03d}_rows.csv')
        ktd = _diem_nut(os.path.join(out_dir, f'nut_{k:03d}.json'), rp, k)
        H = doc_hang(rp)
        got = tu_kiem(x, H, k)
        for c in MEASURED:                               # D2-6: the rebuilt numbers give the node's own states
            s0 = state_fn(got[c]['lo_chung'], got[c]['lo'], got[c]['hi'], got[c]['n_undef'], o1_gate.THRESHOLDS[c])
            if s0 != x['states'][c]:
                raise MatNaError(f'node {k}: the rows rebuild {c} as {s0}, the node says {x["states"][c]} '
                                 '(self-check, mask not applied)')
        if ktd.get('tam_doi') == 'doi':                  # D2-7
            raise MatNaError(f'node {k}: plate-changed case (tam_doi doi): G5 is scored on the IMU heading, which '
                             'the rows do not carry; the mask cannot be applied without the scorer')
        if any(f is None for f in H['fid']) or len(set(H['fid'])) != len(H['fid']):
            # the relative scoring (C8) writes frame_id on every row, once (GATE 24/11: an empty id is an error)
            raise MatNaError(f'node {k}: rows with an empty or repeated frame_id (not a relative C8 scoring)')
        che = np.array([f not in mn['mo'] for f in H['fid']], bool)
        dinh = np.zeros(len(che), bool)
        for c in MEASURED:
            dinh |= np.isfinite(H['lo'][c]) & np.isfinite(H['hi'][c])
        pre[k] = dict(rp=rp, H=H, che=che, moi=che & dinh)   # moi: frames newly undefined by the mask (reported)
    # D2-5; Codex round 2 D2v2-3: a t_som node with ANY masked frame (not only one in its union domain: a frame
    # outside the domain the scorer cut can still be a sample of the turn rate that places t_som) cannot be masked
    # without the scorer. L166 (Liam 25/11 ~14:20 at CODE local_4932073c, box, verbatim: "Chua ket luan
    # (Recommended)"; before: refused): the WHOLE run is then not concluded: every frame of every scored node is
    # undefined for G2-G5 (as a mask with no open frame), G1 chua_ket_luan at every node, no node keeps the cat_t1
    # branch (its violation would stand on frames that are all masked); the masked grid is still written
    vuong = [k for k, d in pre.items() if (nodes[k]['gate'].get('g1') or {}).get('ca_t_som') and d['che'].any()]
    tom = []
    for k, x in enumerate(nodes):
        if not x.get('ok'):
            tom.append(dict(nut=k, cham=False))
            continue
        rp, H, che, moi = pre[k]['rp'], pre[k]['H'], pre[k]['che'], pre[k]['moi']
        ap = np.ones(len(che), bool) if vuong else che   # the frames made undefined at this node
        for c in MEASURED:
            H['lo'][c] = np.where(ap, -math.inf, H['lo'][c])
            H['hi'][c] = np.where(ap, math.inf, H['hi'][c])
        truoc = dict(x['states'])
        # rows: the frame file the mask was applied on (GATE 24/11: the mask goes by frame_id, the node score names
        # only its own sha); n_che ...: the frames outside khung_mo (also when the whole run is not concluded)
        e = dict(nut=k, cham=True, rows=dict(duong_dan=os.path.abspath(rp), sha256=cu.sha256(rp)),
                 n_khung=len(che), n_che=int(che.sum()), n_che_moi=int(moi.sum()),
                 n_che_mien_hop=int((che & H['mu']).sum()),
                 n_che_moi_mien_hop=int((moi & H['mu']).sum()), trang_thai_truoc=truoc)
        if vuong:
            e['che_ca_luot'] = True
            x['cat'] = False
        elif x.get('cat') and np.any(che & H['mc']) and not _certain_before(H, set(x.get('so_do') or [])):
            why = 'mat_na_d2: the violation before t1 (cat_t1) stands only on masked frames: score the node again'
            nodes[k] = dict(x, ok=False, kind='loi', why=why)
            tom.append(dict(e, trang_thai_sau=None, ly_do=why))
            continue
        nums = so_lieu(H)
        st = dict(truoc)
        for c in MEASURED:
            nm = x['nums'][c]
            nm.update({kk: v for kk, v in nums[c].items()
                       if kk not in ('q_sd', 'q_sd_chung') or (kk in nm and _co_q_so_do(x, c))})
            st[c] = state_fn(nm['lo_chung'], nm['lo'], nm['hi'], nm['n_undef'], o1_gate.THRESHOLDS[c])
        if vuong or np.any(che & H['mu']):
            st['g1'] = 'chua_ket_luan'                   # step 4 (L156); any masked frame (D2-4); L166
        if vuong and any(v != 'chua_ket_luan' for v in st.values()):
            raise MatNaError(f'node {k}: every frame masked (L166) but the states are {st} (a bug)')
        x['states'] = st
        for i, t in enumerate(H['t_cam']):
            r = x['rows'].get(round(float(t), key_dec))
            if r is None:
                raise MatNaError(f'node {k}: rows frame t_cam_s {t!r} not in the node rows (a bug)')
            for c in MEASURED:
                r[c] = float(H['lo'][c][i])
        tom.append(dict(e, trang_thai_sau=st, g1_che=bool(vuong or np.any(che & H['mu']))))
    return nodes, dict(version=VERSION, file=mn['file'], sha256=mn['sha256'], sha_noi_dung=mn['sha_noi_dung'],
                       luot=mn['luot'], lan_cham=ma, n_khung_mo=len(mn['mo']), n_o_dat=mn['n_o_dat'],
                       n_o_thieu=mn['n_o_thieu'], nut=tom,
                       ca_luot_chua_ket_luan=(dict(nut_t_som_co_khung_che=vuong, luat=LIAM_T_SOM) if vuong else None),
                       quy_tac='khung (luot, frame_id) khong trong khung_mo -> lo -inf, hi +inf G2-G5 (q giu); so lieu '
                               'nut va trang thai ba nhanh tinh lai; G1 chua_ket_luan khi nut co khung bi che trong '
                               'mien hop (L156, Liam 25/11); cat_t1 mat vi pham chac chan -> nut khong cham; nut tam_doi '
                               'doi -> tu choi; co nut t_som co khung bi che -> CA LUOT chua ket luan: moi khung cua moi '
                               'nut khong xac dinh, G1-G5 chua_ket_luan (L166, Liam 25/11)')


LIAM_T_SOM = ('L166, Liam 25/11 ~14:20 o CODE local_4932073c, hop chon, nguyen van: "Chua ket luan (Recommended)": '
              'luot co nut t_som co khung bi che thi ca luot chua ket luan, van ghi luoi_d2.json')


LUOI_GOC = ('task', 'gio_sha256', 'gate_file_sha256', 'clip', 'file_so_h', 'file_cham_h1', 'file_gio_h1')


def goc_luoi(path, res, khong_che=None):
    """Step 6 (Codex D2-1, D2-2): `res` (the grid result about to be written as luoi_d2.json) against the luoi.json
    of the same folder. Returns {duong_dan, sha256} of luoi.json for the mat_na_d2 block; MatNaError when a chain
    field differs or the mask made a conclusion."""
    if not os.path.isfile(path):
        raise MatNaError(f'{path}: the grid file without the mask must be there first (c8b_luoi --gop-only)')
    with open(path, encoding='utf-8') as f:
        g = json.load(f)
    if not isinstance(g, dict) or 'mat_na_d2' in g or not isinstance(g.get('states'), dict):
        raise MatNaError(f'{path}: not a grid file without a mask')

    def js(v):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    bad = [k for k in LUOI_GOC if js(g.get(k)) != js(res.get(k))]
    if js((g.get('scorer') or {}).get('commit')) != js((res.get('scorer') or {}).get('commit')):
        bad.append('scorer.commit')
    if bad:
        raise MatNaError(f'{path}: {bad} differ from the grid being written with the mask: pass the same --gate-file, '
                         '--clip, --clip-sha256, --file-so-h, --file-cham-h1, --file-gio-h1 again')
    if khong_che is not None:
        # Codex round 2 D2v2-1: luoi.json must be the grid of THESE node files as this code combines them without the
        # mask (the node score SHA in order, the states and the scored nodes of a fresh combine): a luoi.json left
        # from other node files could carry the conclusion the mask is not allowed to make
        sg = [x.get('sha256') for x in g.get('nguon_nut') or [] if isinstance(x, dict)]
        sr = [x.get('sha256') for x in res.get('nguon_nut') or [] if isinstance(x, dict)]
        cu_ = [k for k in ('states', 'nodes_ok') if js(g.get(k)) != js(khong_che.get(k))]
        if sg != sr or cu_:
            raise MatNaError(f'{path}: not the grid of the node files of this folder without the mask '
                             f'({"node score SHA differ; " if sg != sr else ""}{cu_}): combine again '
                             '(c8b_luoi --gop-only) before the mask')
    made =[c for c, v in (res.get('states') or {}).items()
            if v in ('dat', 'khong_dat') and g['states'].get(c) != v]
    if res.get('outcome') in ('dat', 'khong_dat') and g.get('outcome') != res.get('outcome'):
        made.append('outcome')
    if made:
        raise MatNaError(f'the mask made a conclusion for {made} (without the mask: states {g["states"]}, outcome '
                         f'{g.get("outcome")!r}): a mask only takes conclusions away; the run needs the scorer')
    return dict(duong_dan=os.path.abspath(path), sha256=cu.sha256(path))
