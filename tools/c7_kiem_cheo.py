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
          vector [deg]. A3.max_deg = largest |dtheta| over all pairs and clips (tape error NOT subtracted).
  P1      every clip H: per group k, intrinsics + pose fitted without k, the centres of k predicted; error = predicted
          - tape [m]. P1.bo_nhom_max_m = largest error over marks and clips; P1.tung_khung_max_m = the frame-by-frame
          part from the track output of CODE_CAM (design 5; not there yet: xuat refuses).
  u_so    the two consecutive clips H at the start (a, b): per group k the camera of a without k; at b the pose only
          is refitted on the marks outside k (intrinsics of a), the centres of k predicted; d_i = |pred b - pred a|.
          moc_dich.u_so_m = max d_i, moc_dich.nguong_m = 1 mm (page v10 7a; c8b_u adds both to P1).
  P7, A5  every frame of every clip H: pose refitted on the marks outside the group of mark i (intrinsics of the
          camera without the group), floor centre and heading of i; P7.lech_max_m, A5.lech_max_deg = largest deviation
          from the mark's own mean over the clip (not sigma; CODE_C8B, O1 v11.39 J1).
  xuat    c8_cam_u_vao.json for c8b_u (CODE_C8B; version c8_cam_u_vao_v2): blocks P1, moc_dich, A3, P7, A5 from this
          file's result, the other blocks from part files (--phan NAME=file.json), every number finite and >= 0,
          no default, sha256 of every source in `nguon`; an existing file is never replaced.

  python tools/c7_kiem_cheo.py phan_tich --model B1 --H-a <dir>/session.json --H-b <dir>/session.json \
      --video-a <clip H_a> --video-b <clip H_b> [--H-z <dir>/session.json --video-z <clip H_z>] \
      [--sigma moc_san_do.csv ...] [--groups 4] [--L-max 0.8] --out <dir>
      -> <dir>/kiem_cheo.json, <dir>/u_so.json, <dir>/a3_cap.csv
  python tools/c7_kiem_cheo.py xuat --kiem-cheo <dir>/kiem_cheo.json --phan P3=p3.json --phan A2=a2.json \
      --phan than_dich=td.json --phan day_doi=dd.json --phan do_cao=dc.json --phan bo_cham=bc.json --out c8_cam_u_vao.json
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

VERSION = 'c7_kiem_cheo_v2'                  # v2: design v2 (Codex round 1, KC1-1..11)
U_SO_VERSION = 'c7_u_so_v2'
TUNG_KHUNG_VERSION = 'c7_tung_khung_v1'
CAM_U_VERSION = 'c8_cam_u_vao_v2'           # c8b_u.CAM_VERSION (CODE_C8B, format agreed 18/11 with CODE_CAM)
NGUONG_CO_DINH_M = 0.001                     # page v10 7a: 1 mm + the uncertainty of the comparison (u_so)
L_MAX_M = 1.05                               # [DE XUAT] pair length of A3, locked before C8 (v2, KC1-7: the
                                             # centre pairs 25-26, 27-28 at 1.0 m; 84 pairs with 32 marks)
WIN_FRAMES = 30                              # [DE XUAT] window of the mark-shift median: 0.5 s at 60 fps (CODE_CAM)
N_SMALL_SETS = 3                             # per group: the smallest run mark sets also get the window term
CLIPS_REQUIRED = ('H_a', 'H_b', 'H_m', 'H_z')  # design v2 7: two at the start, one mid-session, one after the last run
GROUPS = 4                                   # [DE XUAT] CODE_CAM design 2: ids[k::4]
MIN_TRAIN = 4                                # 4 marks = 16 points: pose determined (as c7_measure.track)
# blocks of c8_cam_u_vao_v2 and the numbers c8b_u.u_p_u_psi reads (CODE_C8B 18/11); other keys in a block are
# information; 'than_dich' is information only; bo_cham.tag_offset is checked apart (null or 3 signed numbers)
CAM_BLOCKS = {
    'P1': ('bo_nhom_max_m', 'tung_khung_max_m'),
    'moc_dich': ('nguong_m', 'u_so_m'),
    'P3': ('vi_tri_max_m',),
    'A2': ('goc_max_deg',),
    'than_dich': (),
    'P7': ('lech_max_m',),
    'A3': ('max_deg',),
    'A5': ('lech_max_deg',),
    'day_doi': ('dn_m', 'dzc_m', 'kappa'),
    'do_cao': ('rho_max_m', 'h_m', 'H_m'),
    'bo_cham': ('side_m',),
}
OWN_BLOCKS = ('P1', 'moc_dich', 'A3', 'P7', 'A5')


class KCError(Exception):
    pass


# ---------------------------------------------------------------- inputs
def clip_from_session(s, model, thu=False):
    """(img {id: (4, 2) mean corners px}, marks {id: (x, y, side)}, w, h, zc) of the floor marks of a session of
    c7_measure calib. The session must have been calibrated with the model forced (calib --model, key model_khoa;
    KC1-1: a model chosen by cross-validation over every mark let the held-out marks into the choice); thu=True
    (trial on old sessions) only asks chosen == model, and its results cannot be exported."""
    if s.get('kind') != 'c7_session':
        raise KCError('not a c7_measure session')
    if not thu and s.get('model_khoa') != model:
        raise KCError(f'session model_khoa {s.get("model_khoa")!r} is not the locked model {model!r} (calib must '
                      'force the model, KC1-1)')
    if s.get('chosen') != model:
        raise KCError(f'session model {s.get("chosen")!r} is not the locked model {model!r} (design 2: the model '
                      'choice used every mark, a held-out mark would have entered it)')
    marks = {int(i): tuple(float(v) for v in m) for i, m in s['marks'].items()}
    ids = [int(i) for i in s['fit_ids']]
    img = {i: np.asarray(s['ref_corners_px'][str(i)], float) for i in ids}
    if any(c.shape != (4, 2) for c in img.values()):
        raise KCError('ref_corners_px: not 4 corners per mark')
    return img, {i: marks[i] for i in ids}, int(s['width']), int(s['height']), s.get('zc_tape_m')


def load_moc(paths):
    """{id: row dict} of measured marker CSVs (c7_marks_from_tape output: x_cm, y_cm, canh_den_mm, sigma_x_mm,
    sigma_y_mm, canh_nguon)."""
    rows = {}
    for p in paths or ():
        with open(p, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('id', '').strip().isdigit():
                    rows[int(r['id'])] = r
    return rows


def check_moc(rows, marks, ids, thu=False):
    """KC1-9: every mark of the fit must come from a MEASURED marker file: sigma present, black side measured
    (canh_nguon = do), centre and side equal to the session's (1e-9 m). A layout CSV (design positions, side
    100.0, no sigma) is refused. Returns {id: (sigma_x_m, sigma_y_m)} (only reported, never subtracted)."""
    sig, bad = {}, []
    for i in ids:
        r = rows.get(i)
        try:
            ok = (r is not None and r.get('sigma_x_mm') not in ('', None) and r.get('sigma_y_mm') not in ('', None)
                  and r.get('canh_nguon', '') == 'do'
                  and abs(float(r['x_cm']) / 100 - marks[i][0]) <= 1e-9 and abs(float(r['y_cm']) / 100 - marks[i][1]) <= 1e-9
                  and abs(float(r['canh_den_mm']) / 1000 - marks[i][2]) <= 1e-9)
        except (KeyError, ValueError):
            ok = False
        if ok:
            sig[i] = (float(r['sigma_x_mm']) / 1000, float(r['sigma_y_mm']) / 1000)
        else:
            bad.append(i)
    if bad and not thu:
        raise KCError(f'marks {bad}: not in a measured marker file (sigma, canh_nguon = do, same centre and side '
                      'as the session; KC1-9)')
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


def pos_with(cam_k, corners, marks, fit_ids, targets):
    """Positions {i: (2,)} of the marks `targets` from one set of corners (a frame or the clip mean): the pose is
    refitted from cam_k (the clip-H camera of the group, never a previous frame: KC1-1) on the marks fit_ids seen,
    intrinsics of cam_k. None when fewer than MIN_TRAIN marks."""
    use = sorted(i for i in fit_ids if i in corners)
    if len(use) < MIN_TRAIN:
        return None
    obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in use])
    im = np.vstack([corners[i] for i in use])
    cam, _ = cm.refit_pose(cam_k, obj, im)
    return {i: centre_floor(cam, corners[i]) for i in targets if i in corners}


def run_sets(groups, all_ids, tap_moc):
    """Per group k: the distinct fit sets S minus k that occur in the runs (track's marks used per frame), plus the
    full set; sets with fewer than MIN_TRAIN marks are dropped (no check value there, KC1-6 counts them)."""
    out = []
    for g in groups:
        sets = {frozenset(set(all_ids) - set(g))}
        for S in tap_moc or ():
            T = frozenset(set(S) - set(g)) & frozenset(all_ids)
            if len(T) >= MIN_TRAIN:
                sets.add(T)
        out.append(sorted(sets, key=lambda t: (len(t), sorted(t))))
    return out


def vec_median_windows(seq, win):
    """Component-wise median of each run of `win` consecutive positions (a window with a missing frame is
    skipped). seq: list of (2,) or None."""
    out = []
    for a in range(0, len(seq) - win + 1):
        w = seq[a:a + win]
        if any(v is None for v in w):
            continue
        out.append(np.median(np.array(w), axis=0))
    return out


def u_so_v2(img_a, img_b, dets_b, marks, groups, cams_a, tap_moc=None, win=WIN_FRAMES, n_small=N_SMALL_SETS):
    """u_so by the comparison of track (design v2 4). Baseline of mark i in group k: the pose of the clip-H_a camera
    of group k refitted on the H_a mean corners of the marks outside k, then i predicted (symmetric with the rest:
    two equal clips give 0, KC1-8). At H_b:
      static term  max over the run sets S (outside k) of |pos_S(H_b mean) - baseline|   (sets of the runs)
      window term  max over windows of |median_win pos_S(H_b frames) - pos_S(H_b mean)| for the full set and the
                   n_small smallest sets of each group (fewer marks: noisier pose)
    u_so = static + window (triangle inequality on pos_S(window) - baseline). Returns (u_so, details)."""
    all_ids = sorted(img_a)
    sets = run_sets(groups, all_ids, tap_moc)
    stat, wind, per = 0.0, 0.0, {}
    for k, g in enumerate(groups):
        base = pos_with(cams_a[k], img_a, marks, frozenset(set(all_ids) - set(g)), g)
        if base is None:
            raise KCError(f'group {k}: fewer than {MIN_TRAIN} marks outside it in H_a')
        small = set(sets[k][:n_small]) | {frozenset(set(all_ids) - set(g))}
        for S in sets[k]:
            pb = pos_with(cams_a[k], img_b, marks, S, g)
            if pb is None:
                continue
            for i, v in pb.items():
                if i in base:
                    d = float(np.linalg.norm(v - base[i]))
                    stat = max(stat, d)
                    per[i] = max(per.get(i, 0.0), d)
            if S in small and dets_b:
                seq = {i: [] for i in g}
                for fr in dets_b:
                    pf = pos_with(cams_a[k], fr, marks, S, g)
                    for i in g:
                        seq[i].append(None if pf is None or i not in pf else pf[i])
                for i in g:
                    if i not in pb:
                        continue
                    for m in vec_median_windows(seq[i], win):
                        wind = max(wind, float(np.linalg.norm(m - pb[i])))
    return stat + wind, dict(tinh_m=stat, cua_so_m=wind, d_tinh_m={str(i): v for i, v in per.items()},
                             n_tap={str(k): len(s) for k, s in enumerate(sets)}, cua_so_khung=win,
                             co_tap_luot=bool(tap_moc))


def time_check(img_a, img_t, marks, groups, cams_a):
    """Full-set positions at a later clip H against the H_a baseline: largest |d|, the common median vector (camera
    change) and the largest part of a mark after removing it (information; design v2 4)."""
    all_ids = sorted(img_a)
    d = {}
    for k, g in enumerate(groups):
        fit = frozenset(set(all_ids) - set(g))
        base = pos_with(cams_a[k], img_a, marks, fit, g)
        pt = pos_with(cams_a[k], img_t, marks, fit, g)
        if base is None or pt is None:
            continue
        for i in g:
            if i in base and i in pt:
                d[i] = pt[i] - base[i]
    if not d:
        return dict(max_m=None)
    v = np.array(list(d.values()))
    med = np.median(v, axis=0)
    rest = {i: float(np.linalg.norm(x - med)) for i, x in d.items()}
    return dict(max_m=float(np.max(np.linalg.norm(v, axis=1))), chung_m=[float(med[0]), float(med[1])],
                rieng_max_m=max(rest.values()), rieng_max_id=max(rest, key=rest.get),
                d_m={str(i): [float(x[0]), float(x[1])] for i, x in d.items()})


# ---------------------------------------------------------------- P7, A5 (frame noise of the static marks)
def clip_dets(video, ids, max_frames=None):
    """Per-frame refined corners {id: (4, 2)} of the floor marks `ids` in a clip (c7_measure detection)."""
    det = cm.make_detector()
    out = []
    for _, _, g in cm.iter_frames(video, max_frames):
        d = cm.detect_refined(det, g)
        out.append({i: np.asarray(c, float) for i, c in d.items() if i in ids})
    if not out:
        raise KCError(f'{video}: no frame read')
    return out


def mark_heading(floor_corners, side):
    """Heading [rad] of a mark from its 4 floor corners (rigid fit to the square, as car_pose)."""
    return cm.rigid_fit(cp.mark_corners_floor(0.0, 0.0, side), floor_corners)[0]


def noise_clip(dets, marks, groups, cams):
    """Per mark: largest deviation from its own mean over the frames, centre [m] and heading [deg]. Each frame, the
    pose is refitted on the marks outside the mark's group (intrinsics of the camera without the group), so the
    mark does not steady its own pose. {id: dict(n, lech_max_m, lech_max_deg, sd_m, sd_deg)}."""
    cen, ang = {}, {}
    for d in dets:
        for k, g in enumerate(groups):
            seen_k = [i for i in g if i in d]
            out_ids = sorted(i for i in d if i not in g and i in marks)
            if not seen_k or len(out_ids) < MIN_TRAIN:
                continue
            obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in out_ids])
            im = np.vstack([d[i] for i in out_ids])
            cam, _ = cm.refit_pose(cams[k], obj, im)
            for i in seen_k:
                fc = cam.pix2floor(d[i], 0.0)
                cen.setdefault(i, []).append(fc.mean(axis=0))
                ang.setdefault(i, []).append(mark_heading(fc, marks[i][2]))
    out = {}
    for i in cen:
        c = np.array(cen[i])
        a = np.degrees(np.unwrap(np.array(ang[i])))
        dc = np.linalg.norm(c - c.mean(axis=0), axis=1)
        da = np.abs(a - a.mean())
        out[i] = dict(n=len(c), lech_max_m=float(dc.max()), lech_max_deg=float(da.max()),
                      sd_m=float(np.sqrt(np.mean(dc ** 2))), sd_deg=float(np.sqrt(np.mean(da ** 2))))
    return out


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
    if name == 'bo_cham':
        if 'tag_offset' not in blk:
            raise KCError('bo_cham: missing tag_offset (null = no offset)')
        to = blk['tag_offset']
        if to is not None and (not isinstance(to, list) or len(to) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in to)):
            raise KCError(f'bo_cham.tag_offset {to!r}: null or [dx_m, dy_m, dpsi_rad]')
        if not blk['side_m'] > 0:
            raise KCError('bo_cham.side_m must be > 0')
    return blk


def build_cam_u(own, parts, ghi_chu=None):
    """own: {block: blk for OWN_BLOCKS, '_src': (file, sha)}; parts: {name: (blk, file, sha)}. Every block gets its
    `nguon` (CODE_C8B 18/11: every block has one) and the top-level list."""
    out = dict(version=CAM_U_VERSION)
    nguon = []
    for name in OWN_BLOCKS:
        blk = dict(check_block(name, own[name]))
        blk['nguon'] = dict(file=own['_src'][0], sha256=own['_src'][1])
        out[name] = blk
        nguon.append(dict(khoi=name, file=own['_src'][0], sha256=own['_src'][1]))
    for name, (blk, f, sha) in sorted(parts.items()):
        if name in OWN_BLOCKS:
            raise KCError(f'{name} comes from the kiem_cheo result, not from --phan')
        blk = dict(check_block(name, blk))
        blk.setdefault('nguon', dict(file=f, sha256=sha))
        out[name] = blk
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
def _load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def phan_tich(a):
    clips_in = {'H_a': (a.H_a, a.video_a), 'H_b': (a.H_b, a.video_b), 'H_m': (a.H_m, a.video_m),
                'H_z': (a.H_z, a.video_z)}
    clips_in = {n: v for n, v in clips_in.items() if v[0]}
    missing = [n for n in CLIPS_REQUIRED if n not in clips_in]
    if missing and not a.thu:
        raise KCError(f'clips H missing: {missing} (design v2 7: H_a, H_b, H_m, H_z)')
    clips, raw = {}, {}
    for name, (p, _) in clips_in.items():
        raw[name] = _load_json(p)
        clips[name] = clip_from_session(raw[name], a.model, a.thu)
    img_a, mk_a = clips['H_a'][0], clips['H_a'][1]
    ids_a = sorted(img_a)
    for n, (img, mk, *_r) in clips.items():
        if any(mk[i] != mk_a[i] for i in mk if i in mk_a):
            raise KCError(f'{n}: mark coordinates differ from H_a (one marker file per session)')
    moc_rows = load_moc(a.moc)
    sig = check_moc(moc_rows, mk_a, sorted(set().union(*[set(c[0]) for c in clips.values()])), a.thu)
    groups = session_groups(raw['H_a'], ids_a, a.groups)
    tap = None
    if a.tap_moc:
        tap = [sorted(int(i) for i in S) for S in _load_json(a.tap_moc)['tap']]
    # A3 over every clip H
    a3 = {n: a3_clip(img, mk, a.model, w, h, zc, a.L_max, sig) for n, (img, mk, w, h, zc) in clips.items()}
    roles = {i: r.get('vai_tro', '') for i, r in moc_rows.items()}
    A3 = a3_summary(a3, [i for i in ids_a if roles.get(i) == 'moc_giua'])
    # P1 at clip H; group cameras of every clip
    cams, p1 = {}, {}
    for n, (img, mk, w, h, zc) in clips.items():
        cams[n] = group_cams(img, mk, groups, a.model, w, h, zc)
        p1[n] = p1_clip(img, mk, groups, cams[n])
    e = [(n, i, float(np.linalg.norm(v))) for n, d in p1.items() for i, v in d.items()]
    worst = max(e, key=lambda t: t[2])
    # frames of every clip H (P7, A5; window term of u_so)
    dets = {}
    for n, (p, v) in clips_in.items():
        if v is None:
            raise KCError(f'{n}: video needed (P7, A5, u_so)')
        if sha256_file(v) != raw[n].get('video_sha256'):
            raise KCError(f'{n}: {v} is not the video of the session (sha256)')
        dets[n] = clip_dets(v, set(clips[n][0]), a.max_frames)
    uso, uso_info = u_so_v2(img_a, clips['H_b'][0], dets['H_b'], mk_a, groups, cams['H_a'], tap, a.win)
    nz = {n: noise_clip(dets[n], clips[n][1], groups, cams[n]) for n in clips}
    allz = [(n, i, r) for n, d in nz.items() for i, r in d.items()]
    zp = max(allz, key=lambda t: t[2]['lech_max_m'])
    za = max(allz, key=lambda t: t[2]['lech_max_deg'])
    sess = {n: dict(file=p, sha256=sha256_file(p)) for n, (p, _) in clips_in.items()}
    common = dict(buoi=a.buoi, model=a.model, nhom=groups, sessions=sess, thu=bool(a.thu),
                  moc=[dict(file=p, sha256=sha256_file(p)) for p in a.moc or ()])
    res = dict(version=VERSION, **common, L_max_m=a.L_max,
               A3=dict(A3, buoi=a.buoi),
               P1=dict(bo_nhom_max_m=worst[2], tung_khung_max_m=None, max_clip=worst[0], max_id=worst[1], buoi=a.buoi,
                       tung_khung='tu file tung_khung (lenh tung_khung, track sau cham lai cua CODE_CAM)',
                       theo_clip={n: {str(i): [float(v[0]), float(v[1])] for i, v in d.items()} for n, d in p1.items()}),
               moc_dich=dict(nguong_m=NGUONG_CO_DINH_M, u_so_m=uso, buoi=a.buoi, **uso_info,
                             nghia='nguong quyet dinh T = nguong_m + u_so_m; dich chua phat hien <= T + u_so_m (KC1-3)'),
               P7=dict(lech_max_m=zp[2]['lech_max_m'], max_clip=zp[0], max_id=zp[1], buoi=a.buoi,
                       sd_max_m=max(r['sd_m'] for _, _, r in allz)),
               A5=dict(lech_max_deg=za[2]['lech_max_deg'], max_clip=za[0], max_id=za[1], buoi=a.buoi,
                       sd_max_deg=max(r['sd_deg'] for _, _, r in allz),
                       gioi_han='moc san 96,4 mm, ma noc 144,5 mm: nhieu goc moc san lon hon; so khung = cac clip H, '
                                'lech lon nhat tang theo so mau; moc dung yen, khong gom nhoe (P4, A4)'),
               nhieu_theo_clip={n: {str(i): r for i, r in d.items()} for n, d in nz.items()},
               theo_thoi_gian={n: time_check(img_a, clips[n][0], mk_a, groups, cams['H_a'])
                               for n in clips if n != 'H_a'})
    os.makedirs(a.out, exist_ok=True)
    write_new(os.path.join(a.out, 'kiem_cheo.json'), res)
    write_new(os.path.join(a.out, 'u_so.json'),
              dict(version=U_SO_VERSION, buoi=a.buoi, u_so_m=uso, nguong_m=NGUONG_CO_DINH_M + uso, nhom=groups,
                   chi_tiet=uso_info, sessions=sess, tao_luc=datetime.datetime.now().isoformat(timespec='seconds')))
    with open(os.path.join(a.out, 'a3_cap.csv'), 'x', newline='', encoding='utf-8') as f:
        cols = ['clip', 'i', 'j', 'L_m', 'dtheta_deg', 'sigma_thuoc_deg', 'mid_x', 'mid_y']
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for n, rows in a3.items():
            for r in rows:
                wr.writerow(dict({k: r.get(k, '') for k in cols[1:]}, clip=n))
    print(f"A3: {A3['n_cap']} cap, max {A3['max_deg']:.3f} deg, RMS {A3['rms_deg']:.3f} deg; "
          f"P1 clip H: max {worst[2] * 1000:.2f} mm (moc {worst[1]}, {worst[0]}); u_so {uso * 1000:.2f} mm "
          f"(tinh {uso_info['tinh_m'] * 1000:.2f} + cua so {uso_info['cua_so_m'] * 1000:.2f}); "
          f"P7 {zp[2]['lech_max_m'] * 1000:.2f} mm; A5 {za[2]['lech_max_deg']:.3f} deg"
          + ('  [THU: khong xuat duoc]' if a.thu else ''))


def check_compat(kc, tk):
    """KC1-10: the frame-by-frame file belongs to this kiem_cheo result (same session, groups, model, sha)."""
    if kc.get('thu'):
        raise KCError('kiem_cheo result made with --thu: not exported')
    miss = [n for n in CLIPS_REQUIRED if n not in (kc.get('sessions') or {})]
    if miss:
        raise KCError(f'kiem_cheo result without clips {miss}')
    if tk.get('version') != TUNG_KHUNG_VERSION:
        raise KCError(f'frame-by-frame file: not {TUNG_KHUNG_VERSION}')
    for k in ('buoi', 'model', 'nhom'):
        if tk.get(k) != kc.get(k):
            raise KCError(f'frame-by-frame file: {k} {tk.get(k)!r} differs from the kiem_cheo result {kc.get(k)!r}')
    if not tk.get('cham_lai'):
        raise KCError('frame-by-frame file: track not rescored after the mark-shift rule (cham_lai missing)')
    v = tk.get('tung_khung_max_m')
    _num(v, 'tung_khung_max_m')
    return float(v)


def xuat(a):
    kc = _load_json(a.kiem_cheo)
    if kc.get('version') != VERSION:
        raise KCError(f'{a.kiem_cheo}: not a {VERSION} result')
    if kc.get('buoi') != a.buoi:
        raise KCError(f'{a.kiem_cheo}: buoi {kc.get("buoi")!r} is not {a.buoi!r}')
    tk = _load_json(a.tung_khung)
    if tk.get('kiem_cheo_sha256') != sha256_file(a.kiem_cheo):
        raise KCError('frame-by-frame file: made from another kiem_cheo result (sha256)')
    own = {b: dict(kc[b]) for b in OWN_BLOCKS}
    own['P1']['tung_khung_max_m'] = check_compat(kc, tk)
    own['P1']['tung_khung_nguon'] = dict(file=a.tung_khung, sha256=sha256_file(a.tung_khung))
    own['_src'] = (a.kiem_cheo, sha256_file(a.kiem_cheo))
    parts = {}
    for spec in a.phan or ():
        name, _, path = spec.partition('=')
        if not path or name in parts:
            raise KCError(f'--phan {spec!r}: NAME=file.json, each block once')
        blk = _load_json(path)
        if not isinstance(blk, dict) or blk.get('buoi') != a.buoi:
            raise KCError(f'--phan {name}: buoi {blk.get("buoi") if isinstance(blk, dict) else None!r} is not '
                          f'{a.buoi!r} (KC1-10)')
        parts[name] = (blk, path, sha256_file(path))
    out = build_cam_u(own, parts, a.ghi_chu)
    out['ghi_chu'] = (out.get('ghi_chu', '') + f' buoi={a.buoi}').strip()
    write_new(a.out, out)
    print(f'{a.out}: {CAM_U_VERSION}, buoi {a.buoi}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('phan_tich')
    p.add_argument('--buoi', required=True)
    p.add_argument('--model', required=True)
    for n in ('a', 'b', 'm', 'z'):
        p.add_argument(f'--H-{n}', dest=f'H_{n}', required=n in 'ab')
        p.add_argument(f'--video-{n}', dest=f'video_{n}')
    p.add_argument('--moc', nargs='+', required=True, help='measured marker CSVs (c7_marks_from_tape)')
    p.add_argument('--tap-moc', dest='tap_moc', help='JSON {"tap": [[ids], ...]}: mark sets used by track in the runs')
    p.add_argument('--win', type=int, default=WIN_FRAMES)
    p.add_argument('--max-frames', dest='max_frames', type=int)
    p.add_argument('--groups', type=int, default=GROUPS)
    p.add_argument('--L-max', dest='L_max', type=float, default=L_MAX_M)
    p.add_argument('--thu', action='store_true', help='trial on old sessions: no model lock, no H_m/H_z; not exported')
    p.add_argument('--out', required=True)
    x = sub.add_parser('xuat')
    x.add_argument('--buoi', required=True)
    x.add_argument('--kiem-cheo', dest='kiem_cheo', required=True)
    x.add_argument('--tung-khung', dest='tung_khung', required=True)
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
