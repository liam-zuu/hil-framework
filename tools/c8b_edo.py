#!/usr/bin/env python3
"""
c8b_edo.py: reader of e_do,perp (diagnostic, not a run condition) for the relative scoring, J v6 (O1 v11.41 appendix
06_SPECS/2026-11-19_Spec_O1_v11.41_Phu_luc_J_v6.md) section 3 "e_do", section 4 "Loai muc theo e_do", section 5
(i)-(ii), item 12(a), (d); design C8B v7 section 9 item 6. Written by CODE_C8B local_07e5654d on feat/c8b-3b-ga,
19/11/2026 (CODE work order 19/11; Liam at CODE: "duyet ca 8, giao viec cac phien phu theo cach nhanh",
"Code ngay (Recommended)").

e_do,perp of a frame = n_Gamma(s)^T [b - z_f(t_log)], both in the lane frame of the task:
  b     the camera centre of the frame, the rows x, y of the scorer node (lane frame after the rigid change of
        c7_score relative mode, CAM v5 section 0.1, data contract F13);
  z_f   the firmware pose pos_x, pos_y of EVERY log row as logged (the edge row too, never set to 0), changed to the lane
        frame by the same rigid change: p_dd + R(psi_dd) z (F13 bien_doi);
  s     the projection candidates: the scorer's s_perp, both ends of its search window [w_lo, w_hi], and the projection
        of every firmware sample inside the window (several branches, ends; J v6 design section 9.4).
Clock (per node i of the c8b_gio file): a camera frame at t_cam maps to the log interval
  [t_cam + D_i - u_plus, t_cam + D_i + u_minus]  (log = camera + D; u_minus / u_plus of the node: u_cham_minus_s,
  u_cham_plus_s when the clock file has them, else half the node step each side [CACH HIEU, stated]);
z_f is interpolated at both ends and taken at every log row inside; an interval reaching a log hole > LOG_HOLE_S or a
firmware restart in the log is undefined for that frame.
Bounds of a frame at node i: lo = max(0, min over (s, z) |e| - U_p_k), hi = max |e| + U_p_k (U_p_k of the row: the
position bound of the difference; a sign change inside the set gives min |e| = 0).
Run: "vuot moi nut" (the lower bound above D at every node) by the cell rule of J v6 section 5: every cell (two
adjacent nodes) has ONE frame with lo > D at both nodes of the cell (the frame may differ between cells); one node:
any frame. "can_xac_dinh_moi_nut": every node scored and every scored frame of every node bounded.
Not concluded for the run (no evidence): khung_tuong_doi_v1 missing; phi_ok false; ca_t_som (q, delta_c_B, beta,
plate changed or not known); mat_neo; log without the rising edge of run_active, a second reset (firmware restart:
timestamp_us falls back; not reset_n, the IMU chip reset count), a log
hole in the run window.

D2 cell mask (design C8B v7 section 7c line 166: a frame of a cell over the bound, or of a cell short of frames, is
n_undef for e_do too; CODE 25/11, after merge 6; no rule, threshold or scoring changed): `luot --mat-na-d2 O_D2.json
--luot ID` reads the mask through the masked grid of the same folder, DIR/luoi_d2.json (c8b_luoi --gop-only
--mat-na-d2, block mat_na_d2): the mask file, the run, the scoring id (lan_cham), the clock file, the node score files
and the rows files must be the ones that grid was made from, and the frames masked here must be the frames masked
there (n_khung, n_che per node), else refused. A frame whose frame_id is not in khung_mo of the run is undefined
(counted in n_undef and n_che of its node): it is never in khung_vuot, never in the cell rule, and its node is not
bounded, so a masked frame makes no conclusion; a finite lower bound of another frame stays. A node the mask turned
into "not scored" (cat_t1 whose certain violation stood only on masked frames) is not scored here either. A masked
grid without a node used (clock file not bounded): chua_ket_luan. The file written carries mat_na_d2 {sha256,
sha_noi_dung, luot, lan_cham, n_khung_mo, nut_khong_cham} and luoi_d2_sha256 (both null without the mask).
`chung-cu`: a masked grid (a file with the key mat_na_d2) goes with an e_do file made with that grid's mask, and a
grid without the mask with an e_do file without it; mixed pairs are refused.

  python tools/c8b_edo.py luot --gio gio.json --luoi-dir DIR --esp32 log.csv --out edo.json
      [--mat-na-d2 o_d2.json --luot RUN_ID]
  python tools/c8b_edo.py chung-cu --luoi luoi.json [--luoi luoi_H1.json] --edo edo.json [--edo ...] \n      --ban-cham-cuoi true|false --out chung_cu.json   (F17 chung_cu_c8)
  python tools/c8b_edo.py muc --luot edo1.json [--luot ...] --out edo_muc.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_edo_v1'
VERSION_MUC = 'c8b_edo_muc_v1'
D_M = 0.10                 # J v6 section 4: D
LOG_HOLE_S = 0.100         # as c8b_gio [DE XUAT]


class EdoError(Exception):
    pass


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def read_log(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith('#'))]
    rows = [r for r in rows if r.get('timestamp_us') not in (None, '')]
    need = ('timestamp_us', 'pos_x', 'pos_y', 'run_active')
    if not rows or any(c not in rows[0] for c in need):
        raise EdoError(f'{path}: no data line or no column {[c for c in need if not rows or c not in rows[0]]}')
    t = np.array([_f(r['timestamp_us']) for r in rows]) * 1e-6
    d = np.diff(t)
    if np.any(d == 0) or not np.all(np.isfinite(t)):
        raise EdoError(f'{path}: timestamp_us repeated or not a number')
    # a firmware restart (boot) in the file: timestamp_us counts again from the boot (esp_timer), so it falls back;
    # the log is kept and the run's e_do is not concluded (J v6 section 3 'reset thu hai'; CODE 19/11: reset_n is the
    # IMU chip reset count, logger.c 147, main.c 299, not a firmware boot)
    return dict(t=t, x=np.array([_f(r['pos_x']) for r in rows]), y=np.array([_f(r['pos_y']) for r in rows]),
                run=np.array([r['run_active'] == '1' for r in rows]),
                restart=[int(i) + 1 for i in np.flatnonzero(d < 0)])


def log_checks(L):
    """Reasons the run's e_do is not concluded from the log (J v6 section 3): a second reset (a firmware restart in the
    log: timestamp_us falls back), no rising edge, more than one, a hole in the run window. The log column reset_n
    (IMU chip resets, logger.c 147) is not a firmware reset and is not a reason here (the run is lost to the command
    path by the scorer, J v6 section 5 (1))."""
    why = []
    if L.get('restart'):
        why.append(f'firmware restart in the log (timestamp_us falls back at row {L["restart"][0]}): a second reset')
        return why, None
    ra = L['run']
    edges = [i for i in range(1, len(ra)) if ra[i] and not ra[i - 1]]
    if len(edges) != 1:
        why.append(f'{len(edges)} rising edges of run_active (need exactly 1)')
        return why, None
    i0 = edges[0]
    i1 = i0 + int(np.argmax(~ra[i0:])) if (~ra[i0:]).any() else len(ra)
    tw = L['t'][i0:i1]
    if len(tw) > 1 and float(np.max(np.diff(tw))) > LOG_HOLE_S:
        why.append(f'log hole {float(np.max(np.diff(tw))) * 1e3:.1f} ms > {LOG_HOLE_S * 1e3:.0f} ms in the run')
    if not (np.isfinite(L['x'][i0:i1]).all() and np.isfinite(L['y'][i0:i1]).all()):
        why.append('pos_x / pos_y not finite in the run window')
    return why, i0


def fw_samples(L, a, b):
    """Firmware poses over the log interval [a, b]: both ends interpolated and every row inside; None when the
    interval leaves the log, meets a hole > LOG_HOLE_S, or the log has a firmware restart (times not one clock)."""
    t = L['t']
    if L.get('restart') or a < t[0] or b > t[-1] or b < a:
        return None
    i = int(np.searchsorted(t, a, side='right')) - 1
    j = int(np.searchsorted(t, b, side='left'))
    j = min(j, len(t) - 1)
    seg = slice(max(i, 0), j + 1)
    ts = t[seg]
    if len(ts) > 1 and float(np.max(np.diff(ts))) > LOG_HOLE_S:
        return None
    xs = [float(np.interp(a, t, L['x'])), float(np.interp(b, t, L['x']))]
    ys = [float(np.interp(a, t, L['y'])), float(np.interp(b, t, L['y']))]
    inside = (t > a) & (t < b)
    xs += list(L['x'][inside])
    ys += list(L['y'][inside])
    return np.array(xs), np.array(ys)


def to_lane(zx, zy, p_dd, psi_dd):
    c, s = math.cos(psi_dd), math.sin(psi_dd)
    return p_dd[0] + c * zx - s * zy, p_dd[1] + s * zx + c * zy


def heading_at(lane, s):
    """Lane heading at arc length s (the sampled lane of c7_score: Path8 or o1_lanes.Lane, unwrapped)."""
    P = lane.path
    if not hasattr(lane, '_edo_hu'):
        lane._edo_hu = np.unwrap(np.asarray(P.head, float))
    return float(np.interp(s, P.s, lane._edo_hu))


def frame_bounds(lane, bx, by, zx, zy, s_perp, w_lo, w_hi, up):
    """(lo, hi) of |e_do,perp| of one frame over the projection candidates and firmware samples."""
    ss = [s_perp, w_lo, w_hi]
    for x_, y_ in zip(zx, zy):
        ss.append(lane.project(float(x_), float(y_), w_lo, w_hi)[0])
    es = []
    for s_ in ss:
        hd = heading_at(lane, min(max(s_, 0.0), lane.S))
        nx, ny = -math.sin(hd), math.cos(hd)
        es.extend(nx * (bx - zx) + ny * (by - zy))
    es = np.array(es)
    if not np.isfinite(es).all():
        return 0.0, math.inf
    amin = 0.0 if (es.min() <= 0.0 <= es.max()) else float(np.min(np.abs(es)))
    return max(0.0, amin - up), float(np.max(np.abs(es))) + up


def lane_of(task):
    import c7_score
    return c7_score.Fig8Task(task) if task in c7_score.FIG8 else c7_score.KTask(task, (0.0, 0.0, 0.0))


def node_frames(score, rows):
    """Frames of one node: key (frame_id, else t_cam_s), x, y, s_perp, w_lo, w_hi, U_p_k, t_cam_s; scored only."""
    out = {}
    for r in rows:
        if str(r.get('scored', '1')) not in ('1', 'True', 'true'):
            continue
        k = r.get('frame_id') or r.get('t_cam_s')
        out[str(k)] = {c: _f(r.get(c)) for c in ('x', 'y', 's_perp', 'w_lo', 'w_hi', 'U_p_k', 't_cam_s')}
        out[str(k)]['fid'] = _fid(r)
    return out


def _fid(r):
    """frame_id of a rows line as the D2 mask keeps it (an integer), else None (c8b_mat_na_d2.doc_hang)."""
    v = _f(r.get('frame_id'))
    return int(v) if math.isfinite(v) and v == int(v) else None


def run_reasons(kt):
    why = []
    if not isinstance(kt, dict) or kt.get('schema') != 'khung_tuong_doi_v1':
        return ['no khung_tuong_doi_v1 block (F13)']
    if not kt.get('phi_ok'):
        why.append('phi_ok false (beta path not valid)')
    if kt.get('t_som_s') is not None or kt.get('ca_t_som'):
        why.append(f'ca t_som: {kt.get("ca_t_som")}')
    if kt.get('tam_doi') not in (None, 'khong_doi'):
        why.append(f'tam_doi {kt.get("tam_doi")}')
    if kt.get('mat_neo'):
        why.append('mat_neo')
    return why


def cell_rule(per_node, thr=D_M):
    """J v6 section 5: every cell (adjacent nodes) has one frame with a finite lo > thr at both nodes; one node: any."""
    n = len(per_node)
    if n == 0:
        return False
    if n == 1:
        return any(math.isfinite(v[0]) and v[0] > thr for v in per_node[0].values())
    for a, b in zip(per_node[:-1], per_node[1:]):
        if not any(k in b and math.isfinite(a[k][0]) and math.isfinite(b[k][0]) and a[k][0] > thr and b[k][0] > thr
                   for k in a):
            return False
    return True


def analyse_run(gio, nodes, L, task, thr=D_M, mat_na=None):
    """nodes: list of (score dict, rows list) per node in the order of gio['nodes_s'] (None: node not scored).
    mat_na (the D2 cell mask, from doc_mat_na_luoi; None: no mask): mo (set of the frame ids of the run left open),
    nut_khong_cham ({node index: reason}: nodes the mask turned into not scored), ly_do_luot (reasons the run is not
    concluded under the mask). A scored frame whose frame_id is not in mo is undefined."""
    why = []
    lw, _ = log_checks(L)
    why += lw
    mo = None if mat_na is None else mat_na['mo']
    bo = {} if mat_na is None else dict(mat_na.get('nut_khong_cham') or {})
    if mat_na is not None:
        why += list(mat_na.get('ly_do_luot') or [])
    lane = lane_of(task)
    step = float(gio.get('step_s') or 0.0)
    u_m = gio.get('u_cham_minus_s') or [step / 2] * len(gio['nodes_s'])
    u_p = gio.get('u_cham_plus_s') or [step / 2] * len(gio['nodes_s'])
    per_node, theo_nut, all_bounded = [], [], True
    for i, D in enumerate(gio['nodes_s']):
        nd = nodes[i] if i < len(nodes) else None
        if nd is None:
            all_bounded = False
            per_node.append({})
            theo_nut.append(dict(i=i, cham=False))
            continue
        if i in bo:                              # not scored in the masked grid (the mask took its violation away)
            all_bounded = False
            per_node.append({})
            theo_nut.append(dict(i=i, cham=False, ly_do=[f'mat_na_d2: {bo[i]}']))
            continue
        sc, rows = nd
        kt = sc.get('khung_tuong_doi_v1')       # F13 key of the scorer (data contract; Codex CAM CT5-3: the old name
        rr = run_reasons(kt)                     # khung_tuong_doi is not taken)
        why += [f'node {i}: {w}' for w in rr if f'node {i}: {w}' not in why]
        bd = kt.get('bien_doi') if isinstance(kt, dict) else None
        if rr or not isinstance(bd, dict):
            all_bounded = False
            per_node.append({})
            theo_nut.append(dict(i=i, cham=False, ly_do=rr or ['no bien_doi']))
            continue
        p_dd, psi_dd = bd.get('p_dd') or [0.0, 0.0], float(bd.get('psi_dd') or 0.0)
        res, n_undef, n_che = {}, 0, 0
        frs = node_frames(sc, rows)
        if mo is not None:
            # from the rows, not from frs: node_frames keeps one frame per key, a repeated id would not show
            ids = [_fid(r) for r in rows if str(r.get('scored', '1')) in ('1', 'True', 'true')]
            if any(f is None for f in ids) or len(set(ids)) != len(ids):
                raise EdoError(f'node {i}: scored rows with an empty or repeated frame_id: the D2 mask goes by '
                               'frame_id (not a relative C8 scoring)')
        for k, fr in frs.items():
            if mo is not None and fr['fid'] not in mo:
                # design 7c: a frame of a masked cell is undefined for e_do (no bound taken from it, whatever it is)
                res[k] = (math.nan, math.inf)
                n_undef += 1
                n_che += 1
                continue
            if not all(math.isfinite(fr[c]) for c in ('x', 'y', 's_perp', 'w_lo', 'w_hi', 'U_p_k', 't_cam_s')):
                res[k] = (math.nan, math.inf)
                n_undef += 1
                continue
            zs = fw_samples(L, fr['t_cam_s'] + D - u_p[i], fr['t_cam_s'] + D + u_m[i])
            if zs is None:
                res[k] = (math.nan, math.inf)
                n_undef += 1
                continue
            zx, zy = to_lane(zs[0], zs[1], p_dd, psi_dd)
            res[k] = frame_bounds(lane, fr['x'], fr['y'], np.asarray(zx), np.asarray(zy), fr['s_perp'], fr['w_lo'],
                                  fr['w_hi'], fr['U_p_k'])
        if n_undef or not res:
            all_bounded = False
        per_node.append(res)
        los = [v[0] for v in res.values() if math.isfinite(v[0])]
        his = [v[1] for v in res.values()]
        theo_nut.append(dict(i=i, D_s=D, cham=True, n_khung=len(res), n_undef=n_undef,
                             lo_max_m=max(los) if los else None, hi_max_m=max(his) if his else None,
                             khung_vuot=sorted(k for k, v in res.items() if math.isfinite(v[0]) and v[0] > thr)))
        if mo is not None:
            theo_nut[-1]['n_che'] = n_che
    if why:
        return dict(version=VERSION, task=task, trang_thai='chua_ket_luan', ly_do=why, theo_nut=theo_nut,
                    vuot_moi_nut=False, can_xac_dinh_moi_nut=False, nguong_m=thr)
    return dict(version=VERSION, task=task, trang_thai='dat', ly_do=[], theo_nut=theo_nut,
                vuot_moi_nut=cell_rule(per_node, thr), can_xac_dinh_moi_nut=bool(all_bounded), nguong_m=thr)


def muc(runs):
    """c8b_edo_muc_v1, J v6 section 4 and section 5 (i)-(ii), item 12(a), (d), J5-7. runs: list of dicts, each an
    c8b_edo_v1 run file plus the run facts the gate supplies: muc, controller, danh_sach (bool: in the list fixed before
    C8, 2b), hong (bool: lost, any kind), ban_cham ('cuoi' or 'tam'), lan_cham (list of the run's edo results: one, or
    two when scored again with the two calibrations)."""
    out = {}
    for r in runs:
        key = (r['muc'], r['controller'])
        e = out.setdefault(key, dict(muc=r['muc'], controller=r['controller'], luot_chung_cu=[], luot_du_can=[],
                                     n_luot=0))
        if not r.get('danh_sach') or r.get('hong') or r.get('ban_cham') != 'cuoi':
            continue                                     # not evidence (J v6: list, not lost, final scoring)
        e['n_luot'] += 1
        lc = r.get('lan_cham') or []
        ok = [x for x in lc if x.get('trang_thai') == 'dat']
        if lc and len(ok) == len(lc) and all(x['vuot_moi_nut'] for x in lc):
            e['luot_chung_cu'].append(r['luot_id'])      # over D at every node at every scoring (12(d))
        if lc and len(ok) == len(lc) and all(x['can_xac_dinh_moi_nut'] for x in lc):
            e['luot_du_can'].append(r['luot_id'])
    levels = {}
    for (m, c), e in out.items():
        lv = levels.setdefault(m, dict(muc=m, loai_muc=False, controller={}))
        lv['controller'][c] = dict(luot_chung_cu=e['luot_chung_cu'], du_bang_chung=bool(e['luot_du_can']),
                                   n_luot=e['n_luot'])
        if e['luot_chung_cu']:
            lv['loai_muc'] = True                        # one run of any controller is enough (12(a))
    for lv in levels.values():
        if lv['loai_muc']:
            lv['trang_thai'] = 'loai_muc'
        elif all(v['du_bang_chung'] for v in lv['controller'].values()):
            lv['trang_thai'] = 'giu_muc'
        else:
            lv['trang_thai'] = 'giu_muc_rui_ro'
            lv['c8_chua_xong'] = [c for c, v in lv['controller'].items() if not v['du_bang_chung'] and v['n_luot'] > 0
                                  ] + [c for c, v in lv['controller'].items() if v['n_luot'] == 0]
    return dict(version=VERSION_MUC, muc=sorted(levels.values(), key=lambda v: v['muc']), nguong_m=D_M)


def g4_finite_every_node(luoi):
    """G4 bound finite at every frame of the union domain at every node (J v6 section 5 "Khoa muc", J3-10): every
    node scored and the combined G4 numbers finite with no undefined frame."""
    nums = (luoi.get('nums') or {}).get('g4') if isinstance(luoi.get('nums'), dict) else None
    ok = luoi.get('nodes_ok') or []
    if not ok or not all(ok) or not isinstance(nums, dict):
        return False
    hi = nums.get('hi')
    return isinstance(hi, (int, float)) and math.isfinite(hi) and nums.get('n_undef') == 0


def kiem_cap_mat_na(luoi, edo, luoi_sha256=None, w=''):
    """The grid file and the e_do file of one scoring are both with the D2 mask (the same mask) or both without it;
    EdoError otherwise. A masked grid is a file with the key mat_na_d2 (luoi_d2.json; the block is None when the
    clock file is not bounded). luoi_sha256: SHA of the grid file, compared with luoi_d2_sha256 of the e_do file."""
    che = isinstance(luoi, dict) and 'mat_na_d2' in luoi
    m = edo.get('mat_na_d2') if isinstance(edo, dict) else None
    if che != (m is not None):
        raise EdoError(f'{w}the grid file is {"with" if che else "without"} the D2 mask, the e_do file '
                       f'{"with" if m is not None else "without"} it: both from the same masked grid, or neither')
    if not che:
        return
    blk = luoi['mat_na_d2']
    bad = [] if blk is None else [k for k in ('sha256', 'sha_noi_dung', 'luot', 'lan_cham')
                                  if not isinstance(m, dict) or m.get(k) != blk.get(k) or m.get(k) is None]
    if luoi_sha256 is not None and edo.get('luoi_d2_sha256') != luoi_sha256:
        bad.append('luoi_d2_sha256')
    if bad:
        raise EdoError(f'{w}the e_do file was not made with the mask of this masked grid ({bad} differ)')


def chung_cu(luois, edos, ban_cham_cuoi, luoi_sha256=None):
    """chung_cu_c8 of a run (data contract F17, GATE design v6_C8 section 7): one bo_so per scoring (two when scored
    again with the two calibrations), each {e_do_lo_vuot_D_moi_nut, e_do_xac_dinh_moi_nut,
    g4_can_huu_han_mien_hop_moi_nut}; phan_loai_du: G1-G5 classified at every scoring (before the 2b mask);
    ban_cham_cuoi: from the calibration drift check of CAM (false when thieu_H1). luoi_sha256: SHA of each grid
    file (the CLI gives them), for the D2 mask pair check (kiem_cap_mat_na)."""
    if len(luois) != len(edos) or not 1 <= len(luois) <= 2:
        raise EdoError('one or two scorings, the same number of luoi and edo files')
    for j, (L, e) in enumerate(zip(luois, edos)):
        kiem_cap_mat_na(L, e, None if luoi_sha256 is None else luoi_sha256[j], f'scoring {j + 1}: ')
    bo, phan = [], True
    for L, e in zip(luois, edos):
        st = L.get('states') or {}
        phan = phan and L.get('trang_thai_luot') == 'cham' and all(c in st for c in ('g1', 'g2', 'g3', 'g4', 'g5'))
        okd = e.get('trang_thai') == 'dat'
        bo.append(dict(e_do_lo_vuot_D_moi_nut=bool(okd and e.get('vuot_moi_nut')),
                       e_do_xac_dinh_moi_nut=bool(okd and e.get('can_xac_dinh_moi_nut')),
                       g4_can_huu_han_mien_hop_moi_nut=g4_finite_every_node(L)))
    return dict(phan_loai_du=bool(phan), ban_cham_cuoi=bool(ban_cham_cuoi), bo_so=bo)


def _load_nodes(gio, d):
    nodes = []
    for k in range(len(gio['nodes_s'])):
        sp, rp = os.path.join(d, f'nut_{k:03d}.json'), os.path.join(d, f'nut_{k:03d}_rows.csv')
        try:
            with open(sp, encoding='utf-8') as f:
                sc = json.load(f)
            with open(rp, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except (OSError, ValueError):
            nodes.append(None)
            continue
        nodes.append((sc, rows))
    return nodes


LUOI_D2 = 'luoi_d2.json'


def doc_mat_na_luoi(gio, gio_path, luoi_dir, mat_na_path, luot, nodes):
    """The D2 cell mask for `luot` as the masked grid luoi_dir/luoi_d2.json applied it. Returns (mat_na for
    analyse_run, block for the e_do file, SHA of luoi_d2.json). EdoError when the mask file, the run, the scoring id,
    the clock file, a node score file or a rows file is not the one of that grid, or the frames masked here are not
    the frames masked there. nodes: the list of _load_nodes (the rows this e_do run reads)."""
    import c8b_mat_na_d2 as md2                              # MatNaError is a ValueError (refused by main)
    mn = md2.doc_mat_na(mat_na_path, luot)
    lp = os.path.join(luoi_dir, LUOI_D2)
    if not os.path.isfile(lp):
        raise EdoError(f'{lp}: the masked grid must be there first (c8b_luoi --gop-only --mat-na-d2 --luot)')
    with open(lp, encoding='utf-8') as f:
        g = json.load(f)
    if not isinstance(g, dict) or 'mat_na_d2' not in g:
        raise EdoError(f'{lp}: not a masked grid file (no mat_na_d2)')
    if g.get('gio_sha256') != cu.sha256(gio_path):
        raise EdoError(f'{lp}: made for another clock file (gio_sha256)')
    lsha = cu.sha256(lp)
    blk = g['mat_na_d2']
    n = len(gio['nodes_s'])
    tin = dict(sha256=mn['sha256'], sha_noi_dung=mn['sha_noi_dung'], luot=luot, lan_cham=mn['lan_cham'],
               n_khung_mo=len(mn['mo']), nut_khong_cham=[])
    if blk is None:                                          # clock file not bounded: no node used by the grid
        return (dict(mo=mn['mo'], nut_khong_cham={},
                     ly_do_luot=['mat_na_d2: the masked grid has no node used (clock file not bounded)']), tin, lsha)
    bad = [k for k in ('sha256', 'sha_noi_dung', 'luot', 'lan_cham') if blk.get(k) != tin[k] or tin[k] is None]
    if bad:
        raise EdoError(f'{lp}: its mask is not {mat_na_path} for run {luot!r} ({bad} differ)')
    nut = blk.get('nut')
    if not isinstance(nut, list) or len(nut) != n or any(not isinstance(e, dict) or e.get('nut') != k
                                                         or not isinstance(e.get('cham'), bool)
                                                         for k, e in enumerate(nut)):
        raise EdoError(f'{lp}: mat_na_d2.nut is not one entry per node of the clock file ({n})')

    def sha_nut(k):
        return cu.sha256(os.path.join(luoi_dir, f'nut_{k:03d}.json'))
    src = {x.get('nut'): x.get('sha256') for x in g.get('nguon_nut') or [] if isinstance(x, dict)}
    co = {k for k in range(n) if os.path.isfile(os.path.join(luoi_dir, f'nut_{k:03d}.json'))}
    if set(src) != co or any(src[k] != sha_nut(k) for k in co):
        raise EdoError(f'{lp}: the node score files of {luoi_dir} are not the ones the masked grid was made from '
                       '(nguon_nut)')
    if md2.ma_lan_cham((k, sha_nut(k)) for k, e in enumerate(nut) if e['cham']) != blk['lan_cham']:
        raise EdoError(f'{lp}: the node score files here do not give the scoring id of the mask (lan_cham)')
    bo = {}
    for k, e in enumerate(nut):
        if not e['cham']:
            continue
        rp = os.path.join(luoi_dir, f'nut_{k:03d}_rows.csv')
        if (e.get('rows') or {}).get('sha256') != cu.sha256(rp) or nodes[k] is None:
            raise EdoError(f'{lp}: node {k}: {rp} is not the rows file the mask was applied on')
        ids = [_fid(r) for r in nodes[k][1]]
        che = sum(1 for f in ids if f not in mn['mo'])
        if len(ids) != e.get('n_khung') or che != e.get('n_che'):
            raise EdoError(f'{lp}: node {k}: {che} of {len(ids)} frames masked here, the masked grid says '
                           f'{e.get("n_che")} of {e.get("n_khung")}')
        if e.get('trang_thai_sau') is None:
            bo[k] = str(e.get('ly_do') or 'node not scored in the masked grid')
    tin['nut_khong_cham'] = sorted(bo)
    return dict(mo=mn['mo'], nut_khong_cham=bo, ly_do_luot=[]), tin, lsha


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('luot')
    a1.add_argument('--gio', required=True)
    a1.add_argument('--luoi-dir', required=True)
    a1.add_argument('--esp32', required=True)
    a1.add_argument('--out', required=True)
    a1.add_argument('--mat-na-d2', default=None,
                    help='c8b_a6_o d2 file: the D2 cell mask, read through --luoi-dir/luoi_d2.json (with --luot)')
    a1.add_argument('--luot', default=None, help='run id of this scoring in the D2 mask (target set)')
    a3 = sub.add_parser('chung-cu')
    a3.add_argument('--luoi', action='append', required=True)
    a3.add_argument('--edo', action='append', required=True)
    a3.add_argument('--ban-cham-cuoi', choices=('true', 'false'), required=True)
    a3.add_argument('--out', required=True)
    a2 = sub.add_parser('muc')
    a2.add_argument('--luot', action='append', required=True, help='json: run facts + lan_cham (see muc())')
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        cu.check_path(a.out)
        if a.cmd == 'luot':
            with open(a.gio, encoding='utf-8') as f:
                gio = json.load(f)
            if gio.get('format') != 'c8b_gio_v1':
                raise EdoError(f'{a.gio}: not a c8b_gio_v1 file')
            if bool(a.mat_na_d2) != bool(a.luot):
                raise EdoError('--mat-na-d2 goes with --luot (the run id of this scoring in the mask)')
            nodes = _load_nodes(gio, a.luoi_dir)
            mat_na, tin, lsha = (None, None, None) if not a.mat_na_d2 else doc_mat_na_luoi(
                gio, a.gio, a.luoi_dir, a.mat_na_d2, a.luot, nodes)
            res = analyse_run(gio, nodes, read_log(a.esp32), gio['task'], mat_na=mat_na)
            res.update(gio_sha256=cu.sha256(a.gio), esp32_sha256=cu.sha256(a.esp32), mat_na_d2=tin,
                       luoi_d2_sha256=lsha)
        elif a.cmd == 'chung-cu':
            def _j(p):
                with open(p, encoding='utf-8') as f:
                    return json.load(f)
            res = dict(chung_cu_c8=chung_cu([_j(p) for p in a.luoi], [_j(p) for p in a.edo], a.ban_cham_cuoi == 'true',
                                            luoi_sha256=[cu.sha256(p) for p in a.luoi]),
                       nguon=[dict(file=os.path.basename(p), sha256=cu.sha256(p)) for p in a.luoi + a.edo])
        else:
            runs = []
            for p in a.luot:
                with open(p, encoding='utf-8') as f:
                    runs.append(json.load(f))
            res = muc(runs)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(_clean(res), indent=1, ensure_ascii=False, allow_nan=False))
    except (EdoError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_edo: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
