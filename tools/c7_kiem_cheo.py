#!/usr/bin/env python3
"""
c7_kiem_cheo.py: independent checks of the C7 camera calibration for the C8b error budget.

CODE_KIEMCHEO (local_d677d6fa), 18/11/2026. Design CLAUDE_THESIS 03_CODE/docs/2026-11-18_Thiet_ke_c7_kiem_cheo_v1.md
(page C8b v10, approved: Liam 18/11 at CODE_C8B, "Duyet v10, Codex ra bo cham CAM (Recommended)"; Codex round 6
R6-1, R6-2). Starting point: the trial of CODE_CAM, 05_DATA/results/C7_VIDEO/2026-11-16_buoi_thu/kiem_cheo_A3_v1/
thu_kiem_cheo_A3.py. The camera fit, the floor corners and the pose refit are those of c7_measure / c7_clip_probe
(CODE_CAM); this file only calls them.

Independence (R6-2, design 2): a held-out mark (the pair of A3, the group k of P1 and u_so) enters no estimation
step: not the intrinsics, distortion or pose, not the second pass of fit_two_pass (its sigma is the residual of the
training marks), and not the model choice: the model is LOCKED before C8 (--model) and a session whose `chosen`
differs is refused. zc, the plumb point and the tape coordinates are external numbers (not fitted to C8 images).

  a3      every clip H of the session: every pair of floor marks with tape distance <= L_max seen in the clip, both
          left out, intrinsics + pose fitted on the rest; dtheta = signed angle from the tape vector to the camera
          vector [deg]. huong_vung.max_deg = largest |dtheta| over all pairs and clips (tape error NOT subtracted).
  p1      every clip H: per group k, intrinsics + pose fitted without k, the centres of k predicted; error = predicted
          - tape [m]. kiem_cheo.max_m = largest error over marks and clips (frame-by-frame part: see the design 5).
  u_so    the two consecutive clips H at the start (a, b): per group k the camera of a without k; at b the pose only
          is refitted on the marks outside k (intrinsics of a), the centres of k predicted; d_i = |pred b - pred a|.
          u_so = max d_i; threshold of the mark-shift rule = 1 mm + u_so (page v10 7a).
  xuat    c8_cam_u_vao.json for c8b_u (CODE_C8B; version c8_cam_u_vao_v10): blocks kiem_cheo, huong_vung from this
          file's result, the other blocks from part files (--phan NAME=file.json), every number finite and >= 0,
          no default, sha256 of every source in `nguon`; an existing file is never replaced.

  python tools/c7_kiem_cheo.py phan_tich --model B1 --H-a <dir>/session.json --H-b <dir>/session.json \
      [--H-z <dir>/session.json] [--sigma moc_san_do.csv ...] [--groups 4] [--L-max 0.8] --out <dir>
      -> <dir>/kiem_cheo.json, <dir>/u_so.json, <dir>/a3_cap.csv
  python tools/c7_kiem_cheo.py xuat --kiem-cheo <dir>/kiem_cheo.json --phan nhac_tam=nt.json ... --out c8_cam_u_vao.json
Last line: #ket_qua,ma=0 done / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7_clip_probe as cp                                               # noqa: E402
import c7_measure as cm                                                  # noqa: E402

VERSION = 'c7_kiem_cheo_v1'
U_SO_VERSION = 'c7_u_so_v1'
CAM_U_VERSION = 'c8_cam_u_vao_v10'          # c8b_u.CAM_VERSION (CODE_C8B)
NGUONG_CO_DINH_M = 0.001                     # page v10 7a: 1 mm + the uncertainty of the comparison
L_MAX_M = 0.8                                # [DE XUAT] pair length of A3, locked before C8
GROUPS = 4                                   # [DE XUAT] CODE_CAM design 2: ids[k::4]
MIN_TRAIN = 4                                # 4 marks = 16 points: pose determined (as c7_measure.track)
# blocks of c8_cam_u_vao and their required numeric keys (c8b_u.py docstring, CODE_C8B 18/11)
CAM_BLOCKS = {
    'kiem_cheo': ('max_m', 'moc_dich_m'),
    'xoay_tai_cho': ('du_tam_m', 'chenh_goc_xa_m'),
    'nhac_tam': ('vi_tri_m', 'goc_deg'),
    'day_doi': ('dn_m', 'dzc_m', 'kappa'),
    'do_cao': ('rho_max_m', 'dh_m', 'H_m', 'h_m'),
    'nhieu_khung': ('lech_max_m', 'lech_max_deg'),
    'goc_dan': ('thuoc_kep_deg', 'tam_than_deg'),
    'huong_vung': ('max_deg',),
}
OWN_BLOCKS = ('kiem_cheo', 'huong_vung')


class KCError(Exception):
    pass


# ---------------------------------------------------------------- inputs
def clip_from_session(s, model):
    """(img {id: (4, 2) mean corners px}, marks {id: (x, y, side)}, w, h, zc) of the floor marks of a session of
    c7_measure calib. Refuses a session fitted with another model than the locked one (design 2)."""
    if s.get('kind') != 'c7_session':
        raise KCError('not a c7_measure session')
    if s.get('chosen') != model:
        raise KCError(f'session model {s.get("chosen")!r} is not the locked model {model!r} (design 2: the model '
                      'choice used every mark, a held-out mark would have entered it)')
    marks = {int(i): tuple(float(v) for v in m) for i, m in s['marks'].items()}
    ids = [int(i) for i in s['fit_ids']]
    img = {i: np.asarray(s['ref_corners_px'][str(i)], float) for i in ids}
    if any(c.shape != (4, 2) for c in img.values()):
        raise KCError('ref_corners_px: not 4 corners per mark')
    return img, {i: marks[i] for i in ids}, int(s['width']), int(s['height']), s.get('zc_tape_m')


def load_sigma(paths):
    """{id: (sigma_x_m, sigma_y_m)} from marker CSVs with sigma columns (reporting only)."""
    sig = {}
    for p in paths or ():
        with open(p, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('sigma_x_mm') and r.get('sigma_y_mm'):
                    sig[int(r['id'])] = (float(r['sigma_x_mm']) / 1000, float(r['sigma_y_mm']) / 1000)
    return sig


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- fits
def fit_without(img, marks, train, model, w, h, zc):
    """Intrinsics + pose on the marks `train` only (c7_measure.fit_two_pass, as the main calibration)."""
    train = sorted(train)
    if len(train) < MIN_TRAIN:
        raise KCError(f'only {len(train)} training marks')
    obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in train])
    im = np.vstack([img[i] for i in train])
    cam, _ = cm.fit_two_pass(model, obj, im, w, h, zc)
    return cam


def centre_floor(cam, corners):
    """Mark centre on the floor: mean of the 4 corners projected to z = 0 (as the trial and c7_measure)."""
    return cam.pix2floor(np.asarray(corners, float), 0.0).mean(axis=0)


def signed_angle_deg(v_ref, v):
    """Signed angle from v_ref to v [deg], in (-180, 180]."""
    return math.degrees(math.atan2(v_ref[0] * v[1] - v_ref[1] * v[0], float(np.dot(v_ref, v))))


def make_groups(ids, n=GROUPS):
    """CODE_CAM design 2: ids[k::n] in ID order (the session's own grouping is used when it has one)."""
    ids = sorted(ids)
    return [ids[k::n] for k in range(n) if ids[k::n]]


def session_groups(s, ids, n=GROUPS):
    g = (s.get('kiem_cheo') or {}).get('nhom')
    if g is None:
        return make_groups(ids, n)
    g = [sorted(int(i) for i in grp) for grp in g]
    flat = [i for grp in g for i in grp]
    if len(flat) != len(set(flat)):
        raise KCError('session kiem_cheo.nhom: a mark in two groups')
    return g


# ---------------------------------------------------------------- A3
def a3_clip(img, marks, model, w, h, zc, l_max=L_MAX_M, sigma=None):
    """Rows per pair (i < j, tape distance <= l_max), both left out of the fit."""
    ids = sorted(img)
    rows = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            vt = np.array(marks[j][:2]) - np.array(marks[i][:2])
            L = float(np.linalg.norm(vt))
            if L > l_max:
                continue
            cam = fit_without(img, marks, [k for k in ids if k not in (i, j)], model, w, h, zc)
            vc = centre_floor(cam, img[j]) - centre_floor(cam, img[i])
            r = dict(i=i, j=j, L_m=L, dtheta_deg=signed_angle_deg(vt, vc),
                     mid_x=float((marks[i][0] + marks[j][0]) / 2), mid_y=float((marks[i][1] + marks[j][1]) / 2))
            if sigma and i in sigma and j in sigma:
                n = np.array([-vt[1], vt[0]]) / L
                sp = math.sqrt(sum((sigma[m][0] * n[0]) ** 2 + (sigma[m][1] * n[1]) ** 2 for m in (i, j)))
                r['sigma_thuoc_deg'] = math.degrees(sp / L)
            rows.append(r)
    return rows


def a3_summary(rows_by_clip, new_ids=()):
    allr = [r for rows in rows_by_clip.values() for r in rows]
    if not allr:
        raise KCError('A3: no pair')
    d = np.array([r['dtheta_deg'] for r in allr])
    st = [r['sigma_thuoc_deg'] for r in allr if 'sigma_thuoc_deg' in r]
    worst = allr[int(np.argmax(np.abs(d)))]
    out = dict(max_deg=float(np.max(np.abs(d))), rms_deg=float(np.sqrt(np.mean(d ** 2))), n_cap=len(allr),
               L_m=[float(min(r['L_m'] for r in allr)), float(max(r['L_m'] for r in allr))],
               cap_lon_nhat=dict(clip=next(c for c, rows in rows_by_clip.items() if worst in rows),
                                 i=worst['i'], j=worst['j'], dtheta_deg=worst['dtheta_deg']),
               theo_clip={c: dict(n_cap=len(rows), max_deg=float(max(abs(r['dtheta_deg']) for r in rows)),
                                  rms_deg=float(np.sqrt(np.mean([r['dtheta_deg'] ** 2 for r in rows]))))
                          for c, rows in rows_by_clip.items() if rows})
    new = set(new_ids)
    for name, sel in (('co_moc_giua', lambda r: r['i'] in new or r['j'] in new),
                      ('chi_moc_cu', lambda r: r['i'] not in new and r['j'] not in new)):
        sub = [abs(r['dtheta_deg']) for r in allr if sel(r)]
        out[f'max_deg_{name}'] = float(max(sub)) if sub else None
    if st and len(st) == len(allr):
        rs = float(np.sqrt(np.mean(np.square(st))))
        out['rms_sigma_thuoc_deg'] = rs
        out['rms_tru_thuoc_deg_tham_khao'] = float(math.sqrt(max(0.0, out['rms_deg'] ** 2 - rs ** 2)))
    out['gioi_han'] = ('lon nhat |dtheta| khong tru sai thuoc (bao thu); doi sang ma 0,14 m gia dinh truong sai tron, '
                       'sai o thang duoi L nho nhat khong bao; chi lay mau tai trung diem cap; do tren mat san, ma '
                       'noc o do cao 0,12 m (thiet ke c7_kiem_cheo v1 muc 3)')
    return out


# ---------------------------------------------------------------- P1, u_so
def group_cams(img, marks, groups, model, w, h, zc):
    """{k: camera fitted (intrinsics + pose) without group k}, only marks seen in this clip."""
    ids = set(img)
    return {k: fit_without(img, marks, sorted(ids - set(g)), model, w, h, zc) for k, g in enumerate(groups)}


def p1_clip(img, marks, groups, cams):
    """{id: error vector [m]} of every mark of every group, predicted by the camera fitted without its group."""
    out = {}
    for k, g in enumerate(groups):
        for i in g:
            if i in img:
                out[i] = centre_floor(cams[k], img[i]) - np.array(marks[i][:2])
    return out


def pos_refit(cam_k, img_b, marks, groups, k):
    """Positions of the marks of group k in clip b: pose refitted on b's marks outside k, intrinsics of cam_k
    (c7_measure.refit_pose, as track does frame by frame)."""
    out_ids = sorted(i for i in img_b if i not in groups[k])
    if len(out_ids) < MIN_TRAIN:
        raise KCError(f'group {k}: only {len(out_ids)} marks outside it in clip b')
    obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in out_ids])
    im = np.vstack([img_b[i] for i in out_ids])
    cam, _ = cm.refit_pose(cam_k, obj, im)
    return {i: centre_floor(cam, img_b[i]) for i in groups[k] if i in img_b}


def u_so(img_a, img_b, marks, groups, cams_a):
    """{id: |pred b - pred a| [m]} (design 4). Marks in only one clip are reported missing."""
    d, thieu = {}, []
    for k, g in enumerate(groups):
        pb = pos_refit(cams_a[k], img_b, marks, groups, k)
        for i in g:
            if i in img_a and i in pb:
                d[i] = float(np.linalg.norm(pb[i] - centre_floor(cams_a[k], img_a[i])))
            else:
                thieu.append(i)
    return d, sorted(thieu)


# ---------------------------------------------------------------- export
def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise KCError(f'{where}: {v!r} is not a finite number >= 0')
    return float(v)


def check_block(name, blk):
    if name not in CAM_BLOCKS:
        raise KCError(f'unknown block {name!r}')
    if not isinstance(blk, dict):
        raise KCError(f'{name}: not an object')
    miss = [k for k in CAM_BLOCKS[name] if k not in blk]
    if miss:
        raise KCError(f'{name}: missing {miss} (no default)')
    for k in CAM_BLOCKS[name]:
        _num(blk[k], f'{name}.{k}')
    return blk


def build_cam_u(own, parts, ghi_chu=None):
    """own: {'kiem_cheo': blk, 'huong_vung': blk, '_src': (file, sha)}; parts: {name: (blk, file, sha)}."""
    out = dict(version=CAM_U_VERSION)
    nguon = []
    for name in OWN_BLOCKS:
        out[name] = check_block(name, own[name])
        nguon.append(dict(khoi=name, file=own['_src'][0], sha256=own['_src'][1]))
    for name, (blk, f, sha) in sorted(parts.items()):
        if name in OWN_BLOCKS:
            raise KCError(f'{name} comes from the kiem_cheo result, not from --phan')
        out[name] = check_block(name, blk)
        nguon.append(dict(khoi=name, file=f, sha256=sha))
    miss = [b for b in CAM_BLOCKS if b not in out]
    if miss:
        raise KCError(f'missing blocks {miss} (no default)')
    out['nguon'] = nguon
    if ghi_chu:
        out['ghi_chu'] = ghi_chu
    return out


def write_new(path, obj):
    if os.path.exists(path):
        raise KCError(f'{path} exists: not replaced')
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- CLI
def phan_tich(a):
    sig = load_sigma(a.sigma)
    sessions = {'H_a': a.H_a, 'H_b': a.H_b}
    if a.H_z:
        sessions['H_z'] = a.H_z
    clips, raw = {}, {}
    for name, p in sessions.items():
        with open(p, encoding='utf-8') as f:
            s = json.load(f)
        raw[name] = s
        clips[name] = clip_from_session(s, a.model)
    ids_a = sorted(clips['H_a'][0])
    groups = session_groups(raw['H_a'], ids_a, a.groups)
    # A3
    a3 = {n: a3_clip(img, mk, a.model, w, h, zc, a.L_max, sig) for n, (img, mk, w, h, zc) in clips.items()}
    new_ids = [i for i in ids_a if (raw['H_a'].get('vai_tro') or {}).get(str(i)) == 'moc_giua']
    A3 = a3_summary(a3, new_ids)
    # P1 (clip H) and u_so
    cams, p1 = {}, {}
    for n, (img, mk, w, h, zc) in clips.items():
        cams[n] = group_cams(img, mk, groups, a.model, w, h, zc)
        p1[n] = p1_clip(img, mk, groups, cams[n])
    e = [(n, i, float(np.linalg.norm(v))) for n, d in p1.items() for i, v in d.items()]
    worst = max(e, key=lambda t: t[2])
    img_a, mk_a = clips['H_a'][0], clips['H_a'][1]
    d_so, thieu = u_so(img_a, clips['H_b'][0], mk_a, groups, cams['H_a'])
    if thieu:
        raise KCError(f'u_so: marks {thieu} not in both clips H_a, H_b')
    uso = max(d_so.values())
    res = dict(version=VERSION, model=a.model, L_max_m=a.L_max, nhom=groups,
               sessions={n: dict(file=p, sha256=sha256_file(p)) for n, p in sessions.items()},
               huong_vung=dict(A3, max_deg=A3['max_deg']),
               kiem_cheo=dict(max_m=worst[2], moc_dich_m=NGUONG_CO_DINH_M + uso, max_clip=worst[0], max_id=worst[1],
                              phan_khung='chua gop (thiet ke muc 5: doc cot kiem cheo tung khung cua track)',
                              theo_clip={n: {str(i): [float(v[0]), float(v[1])] for i, v in d.items()}
                                         for n, d in p1.items()}),
               u_so=dict(u_so_m=uso, d_m={str(i): v for i, v in d_so.items()}))
    if 'H_z' in clips:
        d_z, _ = u_so(img_a, clips['H_z'][0], mk_a, groups, cams['H_a'])
        res['dau_cuoi'] = dict(max_m=max(d_z.values()) if d_z else None, d_m={str(i): v for i, v in d_z.items()})
    os.makedirs(a.out, exist_ok=True)
    write_new(os.path.join(a.out, 'kiem_cheo.json'), res)
    write_new(os.path.join(a.out, 'u_so.json'),
              dict(version=U_SO_VERSION, u_so_m=uso, nguong_m=NGUONG_CO_DINH_M + uso, nhom=groups,
                   d_m={str(i): v for i, v in d_so.items()}, sessions=res['sessions'],
                   tao_luc=datetime.datetime.now().isoformat(timespec='seconds')))
    with open(os.path.join(a.out, 'a3_cap.csv'), 'x', newline='', encoding='utf-8') as f:
        cols = ['clip', 'i', 'j', 'L_m', 'dtheta_deg', 'sigma_thuoc_deg', 'mid_x', 'mid_y']
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for n, rows in a3.items():
            for r in rows:
                wr.writerow(dict({k: r.get(k, '') for k in cols[1:]}, clip=n))
    print(f"A3: {A3['n_cap']} cap, max {A3['max_deg']:.3f} deg, RMS {A3['rms_deg']:.3f} deg; "
          f"P1 clip H: max {worst[2] * 1000:.2f} mm (moc {worst[1]}, {worst[0]}); u_so {uso * 1000:.2f} mm")


def xuat(a):
    with open(a.kiem_cheo, encoding='utf-8') as f:
        kc = json.load(f)
    if kc.get('version') != VERSION:
        raise KCError(f'{a.kiem_cheo}: not a {VERSION} result')
    own = dict(kiem_cheo=kc['kiem_cheo'], huong_vung=kc['huong_vung'], _src=(a.kiem_cheo, sha256_file(a.kiem_cheo)))
    parts = {}
    for spec in a.phan or ():
        name, _, path = spec.partition('=')
        if not path or name in parts:
            raise KCError(f'--phan {spec!r}: NAME=file.json, each block once')
        with open(path, encoding='utf-8') as f:
            parts[name] = (json.load(f), path, sha256_file(path))
    write_new(a.out, build_cam_u(own, parts, a.ghi_chu))
    print(f'{a.out}: {CAM_U_VERSION}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('phan_tich')
    p.add_argument('--model', required=True)
    p.add_argument('--H-a', dest='H_a', required=True)
    p.add_argument('--H-b', dest='H_b', required=True)
    p.add_argument('--H-z', dest='H_z')
    p.add_argument('--sigma', nargs='*')
    p.add_argument('--groups', type=int, default=GROUPS)
    p.add_argument('--L-max', dest='L_max', type=float, default=L_MAX_M)
    p.add_argument('--out', required=True)
    x = sub.add_parser('xuat')
    x.add_argument('--kiem-cheo', dest='kiem_cheo', required=True)
    x.add_argument('--phan', action='append')
    x.add_argument('--ghi-chu', dest='ghi_chu')
    x.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        (phan_tich if a.cmd == 'phan_tich' else xuat)(a)
    except (KCError, OSError, ValueError, KeyError) as e:
        print(f'loi: {e}')
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
