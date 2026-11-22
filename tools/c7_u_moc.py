#!/usr/bin/env python3
"""
c7_u_moc.py: bound of the shifted-anchor error U_moc (CODE_CAM; design CLAUDE_THESIS
03_CODE/docs/2026-11-19_Thiet_ke_CAM_cham_tuong_doi_v4.md section 5, algorithm 03_CODE/docs/2026-11-18_De_xuat_C8b_can_moc_dich_v3.md
section 2, gate contract 03_CODE/docs/2026-11-19_Thiet_ke_rq3_tuong_doi_v6_C8.md sections 10, 23).

Step 'khung' (this file, the heavy part; one run, every valid frame in [tu_s, den_s]):
  the anchors of the frame = moc_neo_xe of the final frames.csv (chain xe, converged schedule, cham_lai 2); the pose is
  refitted in FULL (Huber, refit_pose_checked from cam_H[xe], the very path of track) with
    (a) each anchor j moved by D in 8 directions 45 deg apart from +x floor (D = c7_measure.undetected_bound(u_so):
        max(T_loc + u_loc, T_tho + u_tho) of chain xe, CAM v4 section 4), and
    (b) 200 combinations, every anchor moved: uniform direction, radius D with probability 1/2 else D sqrt(u),
        seeded by SHA-256 of (dets sha256, lich sha256, run, frame idx, SAMPLER) (C8BM-R3-1: no file holding the
        result seeds it);
  then car_pose (tag_offset (0, 0, 0), the session's side, kappa, plumb) and the A6 of CODE_C8B (c8b_a6.delta_psi_a6,
  plumb / zc of 16/11 as c7_chuoi_khung): tag centre t and heading psi' = tag_psi - dpsi_A6 for each refit.
  Per frame:  S_t = sum_j max_dir |dt|, S_psi = sum_j max_dir |dpsi'|, S_sin = sum_j sin(max_dir |dpsi'| / 2),
              C_t = max over (b) |dt|, C_psi = max over (b) |dpsi'|, and the perturbed pose of the worst refit for xy
              and for psi (dx, dy, dpsi', res_sim, scale, rms) for the dependence check of step 'manifest'.
  The bound on the centre p = t - R(psi') q (step 4 of the chain, q known only at scoring time: lan_lap of NHACTAM)
  is taken in 'manifest' by |dp| <= |dt| + 2 sin(|dpsi'| / 2) |q| (exact identity |(R(a+d) - R(a)) q| = 2|sin(d/2)||q|
  plus the triangle inequality: an upper bound of the design quantity, never below it).
  Gate flips (TC3-12): only gate (2a) (res_sim of the car tag, a function of the camera pose the anchors give) is
  re-evaluated on the perturbed pose; flips of the worst refits ('doi_cong_cuc_dai') make the manifest ok=false, flips
  of any refit are reported ('doi_cong_bat_ky'). Gate (1) (floor-fit rms) is NOT a dependence: the shift models a true
  mark position off the map while the image and the map the scorer uses stay the same, so the observed rms does not
  move (the refit rms under the alternative map is reported only, rms_px of the worst poses; on C8 session 2 it rose
  above rms_H + 0.5 px in 2458 of 4676 worst poses of C8-P-F02-1_lai1 while res_sim and scale never flipped). Gate
  (2b) (scale vs the median of the neighbours) is reported only (doi_ti_le_bat_ky): an anchor shift is common to the
  neighbour frames, so it moves their scales together and the ratio to the median to first order not at all.
  A frame is blocked (loi) when the base refit does not reproduce frames.csv (tag_x, tag_y, tag_psi to 1e-5), a refit
  is not ok (refit_pose_checked flags) or a number is not finite: the run has no U (design: 'chan, khong xuat so').

Step 'manifest' (u_moc_manifest_v1, one per scoring = one luoi.json of c8b_luoi, H0 or H1; gate_c8 GC6-6, v6_C8
sections 10, 23, 24; CAM v4 section 5):
  scored nodes = the nodes of luoi.json nguon_nut kept by the grid (nodes_ok true, hong_nut false), in node order, as
  gate_c8._node_scores reads them; mien_da_tinh = their khung_tuong_doi_v1.mien_phai_phu, exactly (gate: equal).
  Domain of the bound = union of their id_khung, widened on each side by the largest grid step (camera time; every
  valid frame with the car seen in it), plus the reference window W (tham_chieu_v1 ids). Adjacent scored nodes whose
  domains start or end more than one grid step + one frame apart: ok false (domain not stable, v4 section 5).
  Every frame of the domain must be in the khung file without loi; any (2a) flip of its worst refits: ok false.
  q of the run: the chain of the scoring (c7_chuoi_khung.chuoi_khung with the very sources of the node score:
  frames, A6, xoay, lan_lap, truc_xe, cho, u_psi_w_rad of tham_chieu_v1), recovered from its rows
  q = R(-psi') (t - p); no q: ok false. The A6 of the khung file vs the A6 of the scoring: dpsi_a6 of the chain
  equals dpsi_a6 of the khung file on EVERY frame of the khung file (1e-12 rad), else ok false.
  Per frame: U_xy = max(S_t + 2|q| S_sin, C_t + 2|q| sin(C_psi / 2)), U_psi = max(S_psi, C_psi).
  U_moc_dong_* = max over the domain; U_moc_tham_xy_m = mean of U_xy over W; U_moc_tham_psi_rad: z = mean e^{i psi'}
  over W, R = |z|, S = mean 2 sin(min(U_psi, pi) / 2), asin(S / R) when S < R, else ok false; U_moc_hieu_xy_m =
  dong + tham. Dependence (TC3-12): the scoring run again by c8b_luoi on the frames moved to the worst refit of each
  frame (c7_score --khung-nhieu, once 'xy', once 'psi'); for every scored node: the reference window state (tap,
  tham_chieu_trang_thai, hong_cua_so_tc), t1 and t_done (each way, within one frame) the same, else ok false; not
  given: ok false (chua_kiem_phu_thuoc).

Step 'nhieu' (used by c7_score --khung-nhieu): frames.csv rows moved to the worst refit of the khung file.

Units: m and rad in the files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c7_clip_probe as cp                                               # noqa: E402
import c7_measure as cm                                                  # noqa: E402
import c7_chuoi_khung as ck                                              # noqa: E402

FORMAT_KHUNG = 'u_moc_khung_v1'
SAMPLER = 'c7_u_moc_bo_lay_mau_v1'       # 8 directions from +x + 200 combinations (v3 section 2); recorded in files
N_DIR = 8
N_COMB = 200
TOL_REPRO = 1e-5                          # m / rad: base refit vs frames.csv (6 decimals written)
CAR_ID = cm.CAR_ID


def sha256(path):
    return ck.sha256(path)


def seed_of(dets_sha, lich_sha, run, idx):
    s = '|'.join([str(dets_sha), str(lich_sha), str(run), str(int(idx)), SAMPLER])
    return int.from_bytes(hashlib.sha256(s.encode('utf-8')).digest()[:8], 'big')


def combos(rng, n_anchor, D):
    """(N_COMB, n_anchor, 2) shifts: uniform direction, radius D w.p. 1/2 else D sqrt(u) (v3 section 2 step 3)."""
    ang = rng.uniform(0.0, 2.0 * math.pi, (N_COMB, n_anchor))
    edge = rng.random((N_COMB, n_anchor)) < 0.5
    rad = np.where(edge, D, D * np.sqrt(rng.random((N_COMB, n_anchor))))
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=-1)


# ---------------------------------------------------------------- worker state (one per process)
_W = {}


def _init(cfg):
    sess = cfg['session_obj']
    _W.update(cfg)
    _W['ctx'] = cm.ChainCtx(sess)
    _W['frames'], _, _ = cm.load_dets(cfg['dets'])
    _W['a6'] = ck.doc_a6(cfg['a6']) if cfg.get('a6') else None


def _gate_2a(res_sim, g):
    """Verdict of gate (2a) (c7_measure.apply_gates: res_sim_m <= car_res_sim_m) of one pose (module docstring)."""
    return bool(res_sim <= g['car_res_sim_m'])


def _scale_ok(scale, scale_ref, g):
    """Gate (2b) against the frame's own scale_ref (reported only, module docstring)."""
    return (not math.isfinite(scale_ref)) or abs(scale / scale_ref - 1.0) <= g['scale_rel']


def _a6_of(xy, psi):
    a6 = _W['a6']
    if a6 is None:
        return np.zeros(len(psi)), np.ones(len(psi), bool)
    d, ok, _ = a6['mod'].delta_psi_a6(a6['obj'], xy, psi, ck.PLUMB_16_11, ck.ZC_16_11)
    return np.asarray(d, float), np.asarray(ok, bool) & np.isfinite(d)


def khung_mot(task):
    """One frame: task = (k, row dict subset). Returns the per-frame record (module docstring)."""
    k, row = task
    ctx, sess = _W['ctx'], _W['session_obj']
    side, kappa, plumb = sess['car_side_m'], sess['kappa'], sess['plumb_xy_m']
    off = (0.0, 0.0, 0.0)
    D, g = _W['D'], _W['gates']
    idx, t, d = _W['frames'][k]
    out = dict(k=k, idx=int(idx), t_s=float(t), loi='')
    anchors = sorted(int(x) for x in str(row['moc_neo_xe']).split(';') if x)
    if CAR_ID not in d or any(i not in d for i in anchors) or len(anchors) < cm.MIN_TRAIN:
        out['loi'] = 'thieu_phat_hien'
        return out
    obj = np.vstack([cp.mark_corners_floor(*ctx.marks[i]) for i in anchors])
    img = np.vstack([d[i] for i in anchors])
    cam_h = ctx.cams[cm.CHAIN_XE]

    def fit(ob):
        c, rms, ok, ly, _ = cm.refit_pose_checked(cam_h, ob, img, None, ctx.cond_max)
        if not ok:
            return None
        p = cm.car_pose(c, d[CAR_ID], side, kappa, plumb, off)
        return (p['tag_x'], p['tag_y'], p['tag_psi'], p['res_sim_m'], p['scale'], rms)

    base = fit(obj)
    if base is None:
        out['loi'] = 'khop_goc_khong_tin'
        return out
    ref = (float(row['tag_x']), float(row['tag_y']), float(row['tag_psi']))
    if max(abs(base[0] - ref[0]), abs(base[1] - ref[1]), abs(cm.cp_wrap(base[2] - ref[2]))) > TOL_REPRO:
        out['loi'] = 'khop_goc_khac_frames'
        return out
    fits = []
    dirs = [(math.cos(a * math.pi / 4), math.sin(a * math.pi / 4)) for a in range(N_DIR)]
    for m in range(len(anchors)):
        for cx, cy in dirs:
            ob = obj.copy()
            ob[4 * m:4 * m + 4] += (D * cx, D * cy)
            fits.append(fit(ob))
    rng = np.random.default_rng(seed_of(_W['dets_sha'], _W['lich_sha'], _W['run'], idx))
    for sh in combos(rng, len(anchors), D):
        ob = obj + np.repeat(sh, 4, axis=0)
        fits.append(fit(ob))
    if any(f is None for f in fits):
        out['loi'] = 'khop_nhieu_khong_tin'
        return out
    F = np.array([base] + fits, float)                       # (1 + 8n + 200, 6)
    if not np.all(np.isfinite(F)):
        out['loi'] = 'khong_huu_han'
        return out
    dps, okA = _a6_of(F[:, :2], F[:, 2])
    if not okA.all():
        out['loi'] = 'a6_khong_dat'
        return out
    psi_p = F[:, 2] - dps
    dt = np.hypot(F[1:, 0] - F[0, 0], F[1:, 1] - F[0, 1])
    dpsi = np.abs([cm.cp_wrap(v) for v in psi_p[1:] - psi_p[0]])
    n8 = N_DIR * len(anchors)
    t8, p8 = dt[:n8].reshape(-1, N_DIR), dpsi[:n8].reshape(-1, N_DIR)
    mt, mp = t8.max(axis=1), p8.max(axis=1)
    sref = float(row['scale_ref'])
    verdict = [_gate_2a(f[3], g) for f in F]
    base_ok = verdict[0]
    flips = [i for i in range(1, len(F)) if verdict[i] != base_ok]
    sc = [_scale_ok(f[4], sref, g) for f in F]
    flips_sc = sum(1 for i in range(1, len(F)) if sc[i] != sc[0])
    i_xy = 1 + int(np.argmax(dt))                       # worst refits (xy, psi) over (a) and (b)
    i_psi = 1 + int(np.argmax(dpsi))
    worst_dir_xy = [1 + m * N_DIR + int(np.argmax(t8[m])) for m in range(len(anchors))]
    worst_dir_psi = [1 + m * N_DIR + int(np.argmax(p8[m])) for m in range(len(anchors))]
    cuc_dai = set([i_xy, i_psi] + worst_dir_xy + worst_dir_psi)

    def pose(i):
        return dict(dx=float(F[i, 0] - F[0, 0]), dy=float(F[i, 1] - F[0, 1]), dpsi_p=float(cm.cp_wrap(psi_p[i] - psi_p[0])),
                    res_sim_m=float(F[i, 3]), scale=float(F[i, 4]), rms_px=float(F[i, 5]))

    out.update(n_neo=len(anchors), cong_goc=bool(base_ok), psi_p=float(psi_p[0]), dpsi_a6=float(dps[0]),
               S_t_m=float(mt.sum()), S_psi_rad=float(mp.sum()), S_sin=float(np.sin(mp / 2.0).sum()),
               C_t_m=float(dt[n8:].max()), C_psi_rad=float(dpsi[n8:].max()),
               s_t_max_m=float(mt.max()), s_psi_max_rad=float(mp.max()),
               xau_xy=pose(i_xy), xau_psi=pose(i_psi),
               doi_cong_cuc_dai=sorted(i for i in flips if i in cuc_dai), doi_cong_bat_ky=len(flips),
               doi_ti_le_bat_ky=flips_sc, rms_goc_px=float(F[0, 5]), rms_max_px=float(F[1:, 5].max()))
    return out


# ---------------------------------------------------------------- command khung
def run_khung(a):
    t_start = time.time()
    with open(a.session, encoding='utf-8') as f:
        sess = json.load(f)
    with open(a.u_so, encoding='utf-8') as f:
        u_so = json.load(f)
    with open(a.lich, encoding='utf-8') as f:
        lich = json.load(f)
    gates, gsha, _ = cm.load_gates(a.hang_so)
    sess_sha, dets_sha = sha256(a.session), sha256(a.dets)
    rows = cm.read_frames(a.frames)
    frames, _, _ = cm.load_dets(a.dets)
    why = cm.verify_final(lich, u_so, sha256(a.u_so), sess, sess_sha, run=a.run, dets_sha=dets_sha, frames=frames)
    if why:
        raise ValueError(f'lich.json is not the converged schedule of this run: {why}')
    if len(rows) != len(frames) or any(int(r['idx']) != int(f[0]) for r, f in zip(rows, frames)):
        raise ValueError('frames.csv and dets.npz of different frames')
    D = cm.undetected_bound(u_so)
    if not (isinstance(D, float) and math.isfinite(D) and D > 0.0):
        raise ValueError(f'D from u_so not a positive number: {D!r}')
    src = ck._tag_heading_source(rows, a.frames, None)     # 'tag_psi', or 'psi' of a tag_offset (0, 0, 0) track
    tasks = []
    for k, r in enumerate(rows):
        t = float(r['t_s'])
        if not (a.tu_s <= t <= a.den_s) or not r['valid'] or not r['car_seen']:
            continue
        tasks.append((k, dict(tag_x=r['tag_x'], tag_y=r['tag_y'], tag_psi=r[src], moc_neo_xe=r['moc_neo_xe'],
                              scale_ref=r['scale_ref'])))
    if not tasks:
        raise ValueError('no valid frame in [tu_s, den_s]')
    cfg = dict(session_obj=sess, dets=a.dets, a6=a.a6, D=D, gates=gates, dets_sha=dets_sha,
               lich_sha=lich['lich_sha256'], run=a.run)
    res = []
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs, initializer=_init, initargs=(cfg,)) as pool:
            for n, r in enumerate(pool.imap(khung_mot, tasks, chunksize=4)):
                res.append(r)
                if n % 50 == 0:
                    print(f'... {n}/{len(tasks)} {time.time() - t_start:.0f} s', file=sys.stderr, flush=True)
    else:
        _init(cfg)
        res = [khung_mot(x) for x in tasks]
    loi = [r for r in res if r['loi']]
    ok = [r for r in res if not r['loi']]
    info = dict(format=FORMAT_KHUNG, bo_lay_mau=SAMPLER, n_huong=N_DIR, n_to_hop=N_COMB, D_m=D, run=a.run,
                luot_id=a.luot_id, tu_s=a.tu_s, den_s=a.den_s, n_khung=len(res), n_loi=len(loi),
                loi_mau=[dict(idx=r['idx'], loi=r['loi']) for r in loi[:20]],
                max_S_t_m=max((r['S_t_m'] for r in ok), default=None),
                max_C_t_m=max((r['C_t_m'] for r in ok), default=None),
                max_S_psi_rad=max((r['S_psi_rad'] for r in ok), default=None),
                max_C_psi_rad=max((r['C_psi_rad'] for r in ok), default=None),
                n_doi_cong_cuc_dai=sum(bool(r['doi_cong_cuc_dai']) for r in ok),
                n_doi_cong_bat_ky=sum(bool(r['doi_cong_bat_ky']) for r in ok),
                n_doi_ti_le_bat_ky=sum(bool(r['doi_ti_le_bat_ky']) for r in ok), cong_phu_thuoc='(2a) res_sim',
                car_pose=dict(tag_offset=[0.0, 0.0, 0.0], side_m=sess['car_side_m'], kappa=sess['kappa'],
                              plumb_xy_m=sess['plumb_xy_m'], phien_ban=sess.get('car_pose_version')),
                a6_chuoi=dict(plumb=list(ck.PLUMB_16_11), zc=ck.ZC_16_11),
                nguon=dict(frames=dict(duong_dan=os.path.abspath(a.frames), sha256=sha256(a.frames)),
                           dets=dict(duong_dan=os.path.abspath(a.dets), sha256=dets_sha),
                           session=dict(duong_dan=os.path.abspath(a.session), sha256=sess_sha),
                           u_so=dict(duong_dan=os.path.abspath(a.u_so), sha256=sha256(a.u_so)),
                           lich=dict(duong_dan=os.path.abspath(a.lich), sha256=sha256(a.lich),
                                     lich_sha256=lich['lich_sha256'], tap_moc_sha256=lich.get('tap_moc_sha256')),
                           a6=None if not a.a6 else dict(duong_dan=os.path.abspath(a.a6), sha256=sha256(a.a6)),
                           hang_so=dict(duong_dan=os.path.abspath(a.hang_so), sha256=gsha)),
                script=cm.script_state(('tools/c7_u_moc.py', 'tools/c7_measure.py', 'tools/c7_chuoi_khung.py',
                                        'tools/c7_clip_probe.py', 'tools/c8b_a6.py')),
                huong_ma=src, giay=round(time.time() - t_start, 1), jobs=a.jobs)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'u_moc_khung.jsonl'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(cm._json(info), ensure_ascii=False) + '\n')
        for r in sorted(res, key=lambda r: r['k']):
            f.write(json.dumps(cm._json(r), ensure_ascii=False) + '\n')
    print(f"{a.run}: {len(res)} khung, {len(loi)} loi, max S_t {info['max_S_t_m']}, max C_t {info['max_C_t_m']}, "
          f"max S_psi {info['max_S_psi_rad']}, doi cong cuc dai {info['n_doi_cong_cuc_dai']}, {info['giay']} s")
    return 0 if not loi else 3


# ---------------------------------------------------------------- khung file, step 'nhieu'
FORMAT_MANIFEST = 'u_moc_manifest_v1'
NHIEU_LOAI = {'xy': 'xau_xy', 'psi': 'xau_psi'}
TOL_A6 = 1e-12                            # rad: dpsi_a6 of the khung A6 vs the scoring A6, same inputs
TOL_Q = 1e-9                              # m: q recovered from every chain row agrees
U_KEYS = ('U_moc_dong_xy_m', 'U_moc_tham_xy_m', 'U_moc_dong_psi_rad', 'U_moc_tham_psi_rad')


def doc_khung(path):
    """(header, {idx: record}) of a u_moc_khung_v1 file."""
    with open(path, encoding='utf-8') as f:
        lines = [json.loads(x) for x in f if x.strip()]
    if not lines or lines[0].get('format') != FORMAT_KHUNG:
        raise ValueError(f'{path}: not a {FORMAT_KHUNG} file')
    recs = {}
    for r in lines[1:]:
        i = int(r['idx'])
        if i in recs:
            raise ValueError(f'{path}: frame idx {i} twice')
        recs[i] = r
    return lines[0], recs


def khung_nhieu(rows, khung_path, loai, frames_sha):
    """(rows, info): frames.csv rows (c7_measure.read_frames) with every frame of the khung file (no loi) moved to its
    worst refit for xy ('xy': xau_xy) or psi ('psi': xau_psi): x, tag_x += dx; y, tag_y += dy; psi, tag_psi +=
    dpsi_p (finite cells only). For the dependence check of 'manifest' (c7_score --khung-nhieu); never a scoring."""
    if loai not in NHIEU_LOAI:
        raise ValueError(f'--khung-nhieu-loai {loai!r}: one of {sorted(NHIEU_LOAI)}')
    head, recs = doc_khung(khung_path)
    if head['nguon']['frames']['sha256'] != frames_sha:
        raise ValueError('the khung file was made from another frames.csv (sha256)')
    out, n = [], 0
    for r in rows:
        rec = recs.get(int(r['idx']))
        if rec is None or rec.get('loi'):
            out.append(r)
            continue
        p = rec[NHIEU_LOAI[loai]]
        r2 = dict(r)
        for cols, v in ((('x', 'tag_x'), p['dx']), (('y', 'tag_y'), p['dy']), (('psi', 'tag_psi'), p['dpsi_p'])):
            for c in cols:
                if c in r2 and math.isfinite(ck._f(r2[c])):
                    r2[c] = ck._f(r2[c]) + float(v)
        out.append(r2)
        n += 1
    return out, dict(duong_dan=os.path.abspath(khung_path), sha256=sha256(khung_path), loai=loai, n_khung_doi=n,
                     gioi_han='vx, vy, wz, speed of frames.csv not moved; the raw heading moved by dpsi_p (the change '
                              'of psi\' after A6), the A6 change between the two poses neglected')


# ---------------------------------------------------------------- step 'manifest'
def _ref(path):
    return dict(duong_dan=os.path.abspath(path), sha256=sha256(path))


def _path(p, goc):
    return p if os.path.isabs(p) else os.path.join(goc, p)


def _doc_ref(ref, goc, what):
    """Path of a {duong_dan, sha256} reference, hashed again."""
    p = _path(ref['duong_dan'], goc)
    if sha256(p) != ref['sha256']:
        raise ValueError(f'{what}: {p} has another sha256 than the one named')
    return p


def _json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def nut_cham(luoi, goc):
    """[(k, node score, its reference)] of the nodes the scoring relies on, in node order (gate_c8._node_scores: a
    node score named by nguon_nut, nodes_ok true, hong_nut false)."""
    nn, nd, hn, ok = (luoi.get(k) for k in ('nguon_nut', 'nodes_s', 'hong_nut', 'nodes_ok'))
    if not (isinstance(nn, list) and isinstance(nd, list) and isinstance(hn, list) and isinstance(ok, list)
            and nn and len(nd) == len(hn) == len(ok)):
        raise ValueError('luoi.json: nguon_nut, nodes_s, hong_nut, nodes_ok (one per node) needed')
    by = {}
    for e in nn:
        k = e.get('nut')
        if type(k) is not int or not 0 <= k < len(nd) or k in by:
            raise ValueError(f'luoi.json nguon_nut: node {k!r} not a node index, or twice')
        by[k] = e
    out = []
    for k in range(len(nd)):
        if k not in by or hn[k] or not ok[k]:
            continue
        p = _doc_ref(by[k], goc, f'node {k}')
        out.append((k, _json(p), dict(nut=k, duong_dan=os.path.abspath(p), sha256=by[k]['sha256'])))
    return out


def _mpp(sc, k):
    import c7_tuong_doi as ctd
    m = (sc.get('khung_tuong_doi_v1') or {}).get('mien_phai_phu')
    if not isinstance(m, dict) or m.get('nut') != k or not m.get('id_khung') \
            or ctd.ids_sha(m['id_khung']) != m.get('id_khung_sha256'):
        raise ValueError(f'node {k}: no mien_phai_phu {{nut, D_s, id_khung, id_khung_sha256}} (c7_score >= 0857f02)')
    return m


def _same_sources(kept):
    """khung_tuong_doi_v1.nguon of the scored nodes: one set of sources for the scoring."""
    srcs = [json.dumps((sc.get('khung_tuong_doi_v1') or {}).get('nguon'), sort_keys=True) for _, sc, _ in kept]
    if len(set(srcs)) != 1:
        raise ValueError('the scored nodes name different sources (khung_tuong_doi_v1.nguon)')
    return kept[0][1]['khung_tuong_doi_v1']['nguon']


def q_of_chain(chain):
    """(q [x, y] or None, ly_do): q = R(-psi') (t - p) on every row with q_ok, all within TOL_Q."""
    qs = []
    for r in chain:
        if not r['q_ok']:
            continue
        c, s = math.cos(r['psi_ma_p']), math.sin(r['psi_ma_p'])
        dx, dy = r['tag_x'] - r['p_x'], r['tag_y'] - r['p_y']
        qs.append((c * dx + s * dy, -s * dx + c * dy))
    if not qs:
        why = sorted({r['ly_do'] for r in chain if r['ly_do']})
        return None, 'thieu_q: ' + '; '.join(why[:3])
    Q = np.array(qs)
    if np.max(np.abs(Q - Q[0])) > TOL_Q:
        return None, 'q_khong_mot_gia_tri'
    return [float(Q[0, 0]), float(Q[0, 1])], ''


def u_khung(rec, aq):
    """(U_xy, U_psi) of one frame record, arm |q| = aq."""
    uxy = max(rec['S_t_m'] + 2.0 * aq * rec['S_sin'], rec['C_t_m'] + 2.0 * aq * math.sin(rec['C_psi_rad'] / 2.0))
    return uxy, max(rec['S_psi_rad'], rec['C_psi_rad'])


def u_tham_psi(psi_p, upsi):
    """asin(S / R) (v4 section 5) or None when S >= R."""
    z = np.mean(np.exp(1j * np.asarray(psi_p, float)))
    R = float(abs(z))
    S = float(np.mean(2.0 * np.sin(np.minimum(np.asarray(upsi, float), math.pi) / 2.0)))
    return (math.asin(S / R) if S < R else None), R, S


def _t_eq(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def phu_thuoc(kept, luoi, nhieu, khung_sha, dt, goc):
    """(record, reasons) of the dependence check (module docstring): the scoring again on the moved frames."""
    rec, bad = {}, []
    for loai in sorted(NHIEU_LOAI):
        if loai not in nhieu:
            bad.append(f'chua_kiem_phu_thuoc_{loai}')
            continue
        p = nhieu[loai]
        L = _json(p)
        if L.get('gio_sha256') != luoi.get('gio_sha256') or L.get('nodes_s') != luoi.get('nodes_s'):
            bad.append(f'{loai}: luoi nhieu of another grid (gio_sha256, nodes_s)')
            continue
        by = {e['nut']: e for e in L.get('nguon_nut') or []}
        n_doi = []
        for k, sc, _ in kept:
            if k not in by:
                n_doi.append(f'nut {k}: khong co diem nhieu')
                continue
            s2 = _json(_doc_ref(by[k], goc, f'{loai} node {k}'))
            kn = s2.get('khung_nhieu') or {}
            if kn.get('sha256') != khung_sha or kn.get('loai') != loai or s2.get('frames_sha256') != sc.get('frames_sha256'):
                n_doi.append(f'nut {k}: diem nhieu khong cua file khung nay ({loai}) hay frames khac')
                continue
            k1, k2 = sc.get('khung_tuong_doi_v1') or {}, s2.get('khung_tuong_doi_v1') or {}
            for f in ('tap', 'tham_chieu_trang_thai'):
                if k1.get(f) != k2.get(f):
                    n_doi.append(f'nut {k}: {f} {k1.get(f)!r} -> {k2.get(f)!r}')
            if bool(sc.get('hong_cua_so_tc')) != bool(s2.get('hong_cua_so_tc')):
                n_doi.append(f'nut {k}: hong_cua_so_tc doi')
            g1, h1 = (sc.get('gate_v3') or {}).get('g1') or {}, (s2.get('gate_v3') or {}).get('g1') or {}
            td1, td2 = list(g1.get('t_done') or [None, None]), list(h1.get('t_done') or [None, None])
            for w in range(2):
                a_, b_ = (td1 + [None, None])[w], (td2 + [None, None])[w]
                if not _t_eq(a_, b_, dt + 1e-9):
                    n_doi.append(f'nut {k}: t_done cach {w + 1} {a_!r} -> {b_!r}')
            if not _t_eq(g1.get('t1'), h1.get('t1'), dt + 1e-9):
                n_doi.append(f'nut {k}: t1 {g1.get("t1")!r} -> {h1.get("t1")!r}')
        rec[loai] = dict(luoi=_ref(p), n_nut=len(kept), doi=n_doi)
        bad += [f'{loai}: {x}' for x in n_doi]
    return rec, bad


def run_manifest(a):
    import c7_tuong_doi as ctd
    t_start = time.time()
    goc = a.goc or os.path.dirname(os.path.abspath(a.luoi))
    head, recs = doc_khung(a.khung)
    khung_sha = sha256(a.khung)
    ng = head['nguon']
    fpath = _doc_ref(ng['frames'], goc, 'khung nguon.frames')
    for k in ('session', 'u_so', 'lich', 'hang_so'):
        _doc_ref(ng[k], goc, f'khung nguon.{k}')
    luoi = _json(a.luoi)
    kept = nut_cham(luoi, goc)
    if not kept:
        raise ValueError('luoi.json: no node the scoring relies on (no manifest for a scoring without one)')
    bad = []
    mien = [_mpp(sc, k) for k, sc, _ in kept]
    for k, sc, _ in kept:
        if sc.get('frames_sha256') != ng['frames']['sha256']:
            raise ValueError(f'node {k} scored another frames.csv than the khung file')
    src = _same_sources(kept)
    task = kept[0][1].get('task')
    rows = cm.read_frames(fpath)
    t_of = {int(r['idx']): float(r['t_s']) for r in rows}
    dt = float(np.median(np.diff(sorted(t_of.values()))))
    nodes_s = sorted(float(x) for x in luoi['nodes_s'])
    step = float(np.max(np.diff(nodes_s))) if len(nodes_s) > 1 else 0.0
    # domain: union of the node domains, widened by the largest grid step, plus W
    spans = []
    for m in mien:
        miss = [i for i in m['id_khung'] if int(i) not in t_of]
        if miss:
            raise ValueError(f'node {m["nut"]}: frame ids not in frames.csv ({miss[:3]})')
        ts = [t_of[int(i)] for i in m['id_khung']]
        spans.append((min(ts), max(ts)))
    for (s0, e0), (s1, e1), m0, m1 in zip(spans, spans[1:], mien, mien[1:]):
        if abs(s1 - s0) > step + dt + 1e-9 or abs(e1 - e0) > step + dt + 1e-9:
            bad.append(f'mien_chua_on_dinh: nut {m0["nut"]} -> {m1["nut"]} ({s0:.3f}-{e0:.3f} -> {s1:.3f}-{e1:.3f} s)')
    lo, hi = min(s for s, _ in spans) - step, max(e for _, e in spans) + step
    dom = {int(i) for m in mien for i in m['id_khung']}
    dom |= {int(r['idx']) for r in rows if r['valid'] and r['car_seen'] and lo - 1e-9 <= float(r['t_s']) <= hi + 1e-9}
    tc_path = _doc_ref(src['tham_chieu_v1'], goc, 'tham_chieu_v1')
    tc = _json(tc_path)
    W = [int(i) for i in tc.get('ids') or []]
    if not W or ctd.ids_sha(W) != tc.get('ids_sha256'):
        raise ValueError('tham_chieu_v1: no reference ids, or ids_sha256 not of its ids')
    # every frame of the domain and W in the khung file, without loi, no (2a) flip of its worst refits
    need = sorted(dom | set(W))
    thieu = [i for i in need if i not in recs]
    loi = [i for i in need if i in recs and recs[i].get('loi')]
    flip = [i for i in need if i in recs and not recs[i].get('loi') and recs[i].get('doi_cong_cuc_dai')]
    if thieu:
        bad.append(f'thieu_khung: {len(thieu)} khung cua mien khong co trong file khung (dau {thieu[:5]})')
    if loi:
        bad.append(f'khung_loi: {len(loi)} khung (dau {loi[:5]})')
    if flip:
        bad.append(f'doi_cong_2a: {len(flip)} khung (dau {flip[:5]})')
    # the chain of the scoring: q, and the A6 of the khung file vs the A6 of the scoring
    a6c_p = _doc_ref(src['c8b_a6_v2'], goc, 'c8b_a6_v2 of the scoring')
    a6c = ck.doc_a6(a6c_p)
    if ng.get('a6') is None:
        raise ValueError('the khung file has no A6 (A6 = 0 is a test run)')
    a6k_p = _doc_ref(ng['a6'], goc, 'khung nguon.a6')
    xo_p = _doc_ref(src['xoay_v1'], goc, 'xoay_v1') if src.get('xoay_v1') else None
    ll_p = _doc_ref(src['lan_lap_v1'], goc, 'lan_lap_v1') if src.get('lan_lap_v1') else None
    tr_p = _doc_ref(src['truc_xe_v1'], goc, 'truc_xe_v1') if src.get('truc_xe_v1') else None
    cs_p = _doc_ref(src['cua_so_v1'], goc, 'cua_so_v1')
    pres = [r for r in ctd.read_jsonl(cs_p) if r.get('loai') == 'pre' and r.get('luot_id') == head.get('luot_id')]
    if len(pres) != 1 or pres[0].get('frames_sha256') != ng['frames']['sha256']:
        raise ValueError(f'cua_so_v1: not one pre row of {head.get("luot_id")!r} with the frames of the khung file')
    ll = ck.doc_lan_lap(ll_p, head.get('luot_id')) if ll_p else None
    tj = os.path.join(os.path.dirname(fpath), 'track.json')
    track = _json(tj) if os.path.isfile(tj) else None
    chain = ck.chuoi_khung(fpath, a6c, xo_p, ll, track=track, cho=ctd.cho_of(task),
                           truc_xe=_json(tr_p) if tr_p else None, u_psi_w_rad=tc.get('u_psi_w_rad'))
    q, why_q = q_of_chain(chain)
    if q is None:
        bad.append(why_q)
    kiem_a6 = dict(a6_khung=_ref(a6k_p), a6_cham=_ref(a6c_p), n_khung=0, n_khac=0, max_lech_rad=0.0)
    if sha256(a6k_p) != sha256(a6c_p):
        a6k = ck.doc_a6(a6k_p)
        by_id = {int(r['idx']): r for r in rows}
        ids = sorted(i for i in recs if not recs[i].get('loi'))
        src_h = ck._tag_heading_source(rows, fpath, track)
        xy = np.array([[ck._f(by_id[i]['tag_x']), ck._f(by_id[i]['tag_y'])] for i in ids], float)
        ps = np.array([ck._f(by_id[i][src_h]) for i in ids], float)
        dk, okk, _ = a6k['mod'].delta_psi_a6(a6k['obj'], xy, ps, ck.PLUMB_16_11, ck.ZC_16_11)
        dc, okc, _ = a6c['mod'].delta_psi_a6(a6c['obj'], xy, ps, ck.PLUMB_16_11, ck.ZC_16_11)
        dk, dc = np.asarray(dk, float), np.asarray(dc, float)
        same = (np.asarray(okk, bool) == np.asarray(okc, bool)) & (np.abs(dk - dc) <= TOL_A6)
        lech = np.abs(dk - dc)
        kiem_a6.update(n_khung=len(ids), n_khac=int((~same).sum()),
                       max_lech_rad=float(np.nanmax(lech)) if np.isfinite(lech).any() else None)
        if (~same).any():
            bad.append(f'a6_khung_khac_a6_cham: {int((~same).sum())} khung')
    else:
        kiem_a6.update(n_khung=sum(1 for i in recs if not recs[i].get('loi')), cung_file=True)
    # the bound
    U = dict.fromkeys(U_KEYS)
    U['U_moc_hieu_xy_m'] = None
    if q is not None and not thieu and not loi:
        aq = math.hypot(q[0], q[1])
        ud = [u_khung(recs[i], aq) for i in sorted(dom)]
        uw = [u_khung(recs[i], aq) for i in W]
        U['U_moc_dong_xy_m'] = max(x for x, _ in ud)
        U['U_moc_dong_psi_rad'] = max(p for _, p in ud)
        U['U_moc_tham_xy_m'] = float(np.mean([x for x, _ in uw]))
        ut, R, S = u_tham_psi([recs[i]['psi_p'] for i in W], [p for _, p in uw])
        U['U_moc_tham_psi_rad'] = ut
        if ut is None:
            bad.append(f'tham_psi_S_khong_nho_hon_R: S {S} R {R}')
        U['U_moc_hieu_xy_m'] = U['U_moc_dong_xy_m'] + U['U_moc_tham_xy_m']
    if not all(isinstance(U[k], float) and math.isfinite(U[k]) and U[k] >= 0.0 for k in U_KEYS) and not bad:
        bad.append('U_khong_huu_han')
    nhieu = dict(x.split('=', 1) for x in (a.nhieu or []))
    pt, bad_pt = phu_thuoc(kept, luoi, nhieu, khung_sha, dt, goc)
    bad += bad_pt
    ses = _json(_path(ng['session']['duong_dan'], goc))
    if not isinstance(ses.get('tag_h_m'), (int, float)):
        raise ValueError('session.json: no tag_h_m (car_pose h_m)')
    cp = head['car_pose']
    lich_p = _path(ng['lich']['duong_dan'], goc)
    lich = _json(lich_p)
    if lich.get('version') != 'c7_lich_v3' or lich.get('hoi_tu') is not True or not lich.get('tap_moc_file_sha256'):
        raise ValueError('lich.json: not a converged c7_lich_v3 with tap_moc_file_sha256')
    gh = []
    if ll is not None and ll.get('cung_lan_lap') is not True:
        gh.append('q qua delta_c_B tho (NHACTAM cach (i), Liam chot 23/11 o NHACTAM: gia dinh tam khong doi cao): vi tri '
                  'lech them khoang dh*|P - F|/(H - h), 0,2*dh o goc toi 0,6*dh o K1; khong cong vao U')
    gh.append('can thuc nghiem (lon nhat quan sat duoc tren 8 huong va 200 to hop), khong phai can chung minh (v3 2)')
    man = dict(format=FORMAT_MANIFEST, ok=not bad, ly_do=bad, file_cham_sha256=sha256(a.luoi), file_cham=_ref(a.luoi),
               lan_cham=a.lan_cham, luot_id=head.get('luot_id'), task=task, mien_da_tinh=mien, **U,
               nguon=dict(bo_lay_mau=head['bo_lay_mau'],
                          car_pose=dict(tag_offset=[0.0, 0.0, 0.0], side_m=cp['side_m'], h_m=float(ses['tag_h_m']),
                                        kappa=cp['kappa'], phien_ban=cp['phien_ban']),
                          tap_moc=lich['tap_moc_file_sha256'], lich=_ref(lich_p)),
               a6_sha256=sha256(a6c_p), kiem_a6=kiem_a6,
               q=dict(q_m=q, abs_q_m=None if q is None else math.hypot(*q), nguon='c7_chuoi_khung (q_luot) cua lan cham'),
               mien=dict(n_khung=len(dom), n_W=len(W), buoc_luoi_s=step, dt_khung_s=dt, tu_s=lo, den_s=hi,
                         n_nut=len(kept)),
               phu_thuoc=pt, gioi_han=gh,
               dau_vao=dict(khung=_ref(a.khung), nut=[r for _, _, r in kept], tham_chieu_v1=_ref(tc_path),
                            cua_so_v1=_ref(cs_p), xoay_v1=xo_p and _ref(xo_p), lan_lap_v1=ll_p and _ref(ll_p),
                            truc_xe_v1=tr_p and _ref(tr_p), **{k: ng[k] for k in ('frames', 'dets', 'session', 'u_so',
                                                                                    'hang_so')}),
               script=cm.script_state(('tools/c7_u_moc.py', 'tools/c7_chuoi_khung.py', 'tools/c7_tuong_doi.py',
                                       'tools/c7_measure.py', 'tools/c8b_a6.py')),
               giay=round(time.time() - t_start, 1))
    text = json.dumps(cm._json(man), ensure_ascii=False, indent=1, allow_nan=False)
    with open(a.out, 'x', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f"{head.get('luot_id')} {a.lan_cham}: ok {man['ok']}, U dong/tham xy {U['U_moc_dong_xy_m']} / "
          f"{U['U_moc_tham_xy_m']} m, psi {U['U_moc_dong_psi_rad']} / {U['U_moc_tham_psi_rad']} rad"
          + ('' if not bad else '; ' + '; '.join(bad[:4])))
    return 0 if man['ok'] else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('khung', help='per-frame bound of one run (heavy; module docstring)')
    k.add_argument('--frames', required=True, help='final frames.csv (track --lich, cham_lai 2)')
    k.add_argument('--dets', required=True)
    k.add_argument('--session', required=True)
    k.add_argument('--u-so', dest='u_so', required=True)
    k.add_argument('--lich', required=True, help='converged lich.json the final track used')
    k.add_argument('--run', required=True, help='name of the clip in lich.json')
    k.add_argument('--luot-id', dest='luot_id', default=None)
    k.add_argument('--a6', default=None, help='c8b_a6_v2 file of CODE_C8B (psi\' after A6); none: A6 = 0 (test only)')
    k.add_argument('--hang-so', dest='hang_so', required=True, help='constants file (cong_khung)')
    k.add_argument('--tu-s', dest='tu_s', type=float, default=-math.inf)
    k.add_argument('--den-s', dest='den_s', type=float, default=math.inf)
    k.add_argument('--jobs', type=int, default=1)
    k.add_argument('--out', required=True)
    m = sub.add_parser('manifest', help='u_moc_manifest_v1 of one scoring (module docstring)')
    m.add_argument('--khung', required=True, help='u_moc_khung.jsonl of the run (step khung)')
    m.add_argument('--luoi', required=True, help='luoi.json of the scoring (c8b_luoi; H0 or H1)')
    m.add_argument('--lan-cham', dest='lan_cham', required=True, help='H0, H1_so_bo or H1 (recorded)')
    m.add_argument('--nhieu', action='append', metavar='LOAI=LUOI',
                   help='luoi.json of the scoring again with c7_score --khung-nhieu (xy=..., psi=...)')
    m.add_argument('--goc', default=None, help='folder the relative paths of the score files are relative to '
                                               '(default: the folder of --luoi)')
    m.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'manifest':
            return run_manifest(a)
        return run_khung(a)
    except (OSError, ValueError, KeyError) as e:
        print(f'c7_u_moc: {e}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    sys.exit(main())
