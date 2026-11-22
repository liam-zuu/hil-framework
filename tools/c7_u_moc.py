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
  Gate flips (TC3-12): for every refit the frame gates (1) rms, (2a) res_sim, (2b) scale vs the frame's scale_ref are
  evaluated on the perturbed pose; flips of the worst refits ('doi_cong_cuc_dai') make the manifest ok=false, flips of
  any refit are reported ('doi_cong_bat_ky').
  A frame is blocked (loi) when the base refit does not reproduce frames.csv (tag_x, tag_y, tag_psi to 1e-5), a refit
  is not ok (refit_pose_checked flags) or a number is not finite: the run has no U (design: 'chan, khong xuat so').

Step 'manifest' (u_moc_manifest_v1 per scoring): later, needs the score files of C8B / NHACTAM (mien_phai_phu, W).

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


def _gates(rms, res_sim, scale, scale_ref, g, rms_h):
    """Verdict of gates (1) rms, (2a) res_sim, (2b) scale of one pose (c7_measure.apply_gates, without the moc_it and
    neighbour-count parts that do not move with an anchor shift)."""
    if not rms <= rms_h + g['rms_extra_px']:
        return False
    if not res_sim <= g['car_res_sim_m']:
        return False
    if math.isfinite(scale_ref) and not abs(scale / scale_ref - 1.0) <= g['scale_rel']:
        return False
    return True


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
    D, g, rms_h = _W['D'], _W['gates'], sess['rms_px_H']
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
    verdict = [_gates(f[5], f[3], f[4], sref, g, rms_h) for f in F]
    base_ok = verdict[0]
    flips = [i for i in range(1, len(F)) if verdict[i] != base_ok]
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
               doi_cong_cuc_dai=sorted(i for i in flips if i in cuc_dai), doi_cong_bat_ky=len(flips))
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
    a = ap.parse_args(argv)
    try:
        return run_khung(a)
    except (OSError, ValueError, KeyError) as e:
        print(f'c7_u_moc: {e}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    sys.exit(main())
