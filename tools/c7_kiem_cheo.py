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

VERSION = 'c7_kiem_cheo_v6'                  # v6: A3 carried to the car tag (design v5 S5-1); v5: Codex code review round 1
U_SO_VERSION = 'c7_u_so_v4'
TUNG_KHUNG_VERSION = 'c7_tung_khung_v3'
CAM_U_VERSION = 'c8_cam_u_vao_v2'           # c8b_u.CAM_VERSION (CODE_C8B, format agreed 18/11 with CODE_CAM)
NGUONG_CO_DINH_M = 0.001                     # page v10 7a: 1 mm + the uncertainty of the comparison (u_so)
L_MAX_M = 1.05                               # [DE XUAT] pair length of A3, locked before C8 (v2, KC1-7: the
                                             # centre pairs 25-26, 27-28 at 1.0 m; 84 pairs with 32 marks)
# A3 carried to the car tag (design v5 S5-1): virtual car tags (session car_side_m, tag_h_m) at these points of every
# pair segment and these headings (a square repeats every 90 deg), seen through the full camera, measured through
# the camera without the pair
A3_TAG_T = (0.0, 0.25, 0.5, 0.75, 1.0)       # [DE XUAT] fractions along i -> j (CODE_C8B 18/11: several points)
A3_TAG_PHI_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)  # [DE XUAT]
A3_VUONG_R_M = 0.7                           # [DE XUAT] information only: a pair has a perpendicular partner when
A3_VUONG_DELTA_DEG = 2.0                     #   a pair of the same clip has its midpoint within R, angle 90 +- delta
# Approved: Liam 18/11/2026 at CODE_KIEMCHEO (local_bba4e18e), choice box on the method of design v5 S5 (A3 an empirical
# cross-check index carried to the car tag, limits (a)-(m)), verbatim: "Duyet (Recommended)". c8b_u refuses A3 unless
# the export says quy_ve_ma_noc true, which only this constant allows (check_a3_export).
A3_QUY_DUYET = True
WIN_S = 0.5                                 # [DE XUAT] window of the mark-shift median (CODE_CAM design v2 3)
WIN_MIN_N = 10                               #   at least 10 samples
WIN_GAP_S = 0.1                              #   a gap above 0.1 s restarts the window
COVER_MAX_MISS = 0.10                        # [DE XUAT] design v4 5: valid frames without a check value, at most
COVER_MAX_GAP_S = 1.0                        #   and no gap without check longer than this
N_SMALL_SETS = 3                             # per group: the smallest run mark sets also get the window term
KIEM_H_MIN = 0.95                            # [DE XUAT] design v3 7 (KC2-2): share of frames with every mark at once
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
    """Mark centre on the floor: the MEAN OF THE 4 IMAGE CORNERS back-projected to z = 0, as c7_measure.calibrate
    and CODE_CAM design v2 1 (KCC1-1; the mean of the projected corners differs under perspective and distortion)."""
    return cam.pix2floor(np.asarray(corners, float).mean(axis=0, keepdims=True), 0.0)[0]


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
    if set(flat) != set(ids):
        raise KCError(f'session kiem_cheo.nhom does not partition the fit marks: missing {sorted(set(ids) - set(flat))}, '
                      f'extra {sorted(set(flat) - set(ids))} (KCC1-14)')
    return g


# ---------------------------------------------------------------- A3
def pair_angle_diff(cam_see, cam_meas, mi, mj):
    """Angle [rad] of the pair i -> j measured through cam_meas minus through cam_see, both on the image that
    cam_see makes of the two floor marks mi, mj = (x, y, side) (centre = back-projected mean of the image corners, as
    centre_floor). 0 when the two cameras are the same."""
    pix = [cam_see.project(np.column_stack([cp.mark_corners_floor(*m), np.zeros(4)])) for m in (mi, mj)]
    v_see = centre_floor(cam_see, pix[1]) - centre_floor(cam_see, pix[0])
    v_meas = centre_floor(cam_meas, pix[1]) - centre_floor(cam_meas, pix[0])
    return math.radians(signed_angle_deg(v_see, v_meas))


def tag_heading_diff(cam_see, cam_meas, x, y, phi, side, tag_h):
    """Kabsch heading [rad] of a car tag (side, top at height tag_h, centre (x, y), heading phi) measured through
    cam_meas minus through cam_see, both on the image cam_see makes of it. As c7_measure.car_pose: the corners go to
    z = 0, then a homothety about the plumb point (factor 1 - kappa > 0: c8b_u needs H > h), which keeps the Kabsch
    angle, so the heading depends neither on kappa nor on the plumb point (design v5 S5-1)."""
    body = cm.car_corners_body(side)
    c, s = math.cos(phi), math.sin(phi)
    P = body @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    pix = cam_see.project(np.column_stack([P, np.full(4, float(tag_h))]))
    a_see = cm.rigid_fit(body, cam_see.pix2floor(pix, 0.0))[0]
    a_meas = cm.rigid_fit(body, cam_meas.pix2floor(pix, 0.0))[0]
    return cm.cp_wrap(a_meas - a_see)


def a3_tag_gap(cam_full, cam_ij, mi, mj, side, tag_h, ts=A3_TAG_T, phis=A3_TAG_PHI_DEG):
    """Design v5 S5-1 (b): over virtual car tags along the segment i -> j and the headings phis, the largest
    |tag heading difference - pair angle difference| [deg] between the full camera and the camera without the pair,
    in BOTH directions (seen through one, measured through the other; Codex KCA1-1, v5b), the largest |tag heading
    difference| [deg] (information) and the pair angle difference full -> without pair [deg]."""
    pi, pj = np.array(mi[:2], float), np.array(mj[:2], float)
    gap = dpsi = 0.0
    for see, meas in ((cam_full, cam_ij), (cam_ij, cam_full)):
        dp = pair_angle_diff(see, meas, mi, mj)
        for t in ts:
            x, y = (1.0 - t) * pi + t * pj
            for ph in phis:
                d = tag_heading_diff(see, meas, x, y, math.radians(ph), side, tag_h)
                gap, dpsi = max(gap, abs(d - dp)), max(dpsi, abs(d))
    return math.degrees(gap), math.degrees(dpsi), math.degrees(pair_angle_diff(cam_full, cam_ij, mi, mj))


def session_tag(raws):
    """(car_side_m, tag_h_m) of the car tag, the same in every session (design v5 S5-1); refused when missing, not a
    finite number > 0 (side) / >= 0 (height), or different between the clips H."""
    vals = set()
    for n, s in raws.items():
        side, th = s.get('car_side_m'), s.get('tag_h_m')
        for k, v, lo_ok in (('car_side_m', side, lambda x: x > 0), ('tag_h_m', th, lambda x: x >= 0)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo_ok(v):
                raise KCError(f'{n}: session {k} {v!r} (A3 is carried to the car tag, design v5 S5-1)')
        vals.add((float(side), float(th)))
    if len(vals) != 1:
        raise KCError(f'car tag differs between the clips H: {sorted(vals)}')
    return vals.pop()


def a3_clip(img, marks, model, w, h, zc, l_max=L_MAX_M, sigma=None, tag=None):
    """Rows per pair (i < j, tape distance <= l_max), both left out of the fit. tag = (car_side_m, tag_h_m): each
    row also gets the conversion to the car tag (design v5 S5-1): gap_deg (b), quy_deg = |dtheta| + gap_deg."""
    ids = sorted(img)
    rows = []
    cam_full = fit_without(img, marks, ids, model, w, h, zc) if tag is not None else None
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
                     mid_x=float((marks[i][0] + marks[j][0]) / 2), mid_y=float((marks[i][1] + marks[j][1]) / 2),
                     huong_deg=math.degrees(math.atan2(vt[1], vt[0])))
            if tag is not None:
                g, dpsi, dp = a3_tag_gap(cam_full, cam, marks[i], marks[j], *tag)
                r.update(gap_deg=g, dpsi_ma_max_deg=dpsi, dtheta_hai_cam_deg=dp, quy_deg=abs(r['dtheta_deg']) + g)
            if sigma and i in sigma and j in sigma:
                n = np.array([-vt[1], vt[0]]) / L
                sp = math.sqrt(sum((sigma[m][0] * n[0]) ** 2 + (sigma[m][1] * n[1]) ** 2 for m in (i, j)))
                r['sigma_thuoc_deg'] = math.degrees(sp / L)
            rows.append(r)
    return rows


def a3_perpendicular(rows, r_m=A3_VUONG_R_M, delta_deg=A3_VUONG_DELTA_DEG):
    """Information for design v5 S5-1 (a): the pairs of one clip that have a partner pair with its midpoint within
    r_m and a direction 90 +- delta_deg away (the local-linear argument needs two perpendicular pairs in one area)."""
    n = 0
    for p in rows:
        for q in rows:
            if q is p or math.hypot(q['mid_x'] - p['mid_x'], q['mid_y'] - p['mid_y']) > r_m:
                continue
            d = abs(((q['huong_deg'] - p['huong_deg']) % 180.0) - 90.0)
            if d <= delta_deg:
                n += 1
                break
    return n


def a3_summary(rows_by_clip, new_ids=(), tag=None):
    """A3 over every clip H. When every row carries the conversion to the car tag (a3_clip tag=...), max_deg is the
    empirical cross-check index carried to the car tag, max(|dtheta| + gap) (design v5 S5-1; not a bound, it can be
    below the true tag error at a sample: limits (k), (l)) and max_deg_cap the pair angle alone; else max_deg is the
    pair angle and the block says it is not converted. quy_ve_ma_noc = A3_QUY_DUYET (Liam's approval) only then."""
    allr = [r for rows in rows_by_clip.values() for r in rows]
    if not allr:
        raise KCError('A3: no pair')
    d = np.array([r['dtheta_deg'] for r in allr])
    st = [r['sigma_thuoc_deg'] for r in allr if 'sigma_thuoc_deg' in r]
    worst = allr[int(np.argmax(np.abs(d)))]
    out = dict(max_deg=float(np.max(np.abs(d))), rms_deg=float(np.sqrt(np.mean(d ** 2))), n_cap=len(allr),
               L_m=[float(min(r['L_m'] for r in allr)), float(max(r['L_m'] for r in allr))],
               cap_lon_nhat=dict(clip=next(c for c, rows in rows_by_clip.items() if any(worst is r for r in rows)),
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
    conv = all('quy_deg' in r for r in allr)
    if conv:
        q = allr[int(np.argmax([r['quy_deg'] for r in allr]))]
        out.update(max_deg_cap=out['max_deg'], max_deg=float(q['quy_deg']),
                   gap_max_deg=float(max(r['gap_deg'] for r in allr)),
                   dpsi_ma_max_deg=float(max(r['dpsi_ma_max_deg'] for r in allr)),
                   cap_quy_lon_nhat=dict(clip=next(c for c, rows in rows_by_clip.items() if any(q is r for r in rows)), i=q['i'],
                                         j=q['j'], dtheta_deg=q['dtheta_deg'], gap_deg=q['gap_deg']),
                   ma_ao=dict(t=list(A3_TAG_T), phi_deg=list(A3_TAG_PHI_DEG), hai_chieu=True,
                              canh_m=None if tag is None else float(tag[0]), cao_m=None if tag is None else float(tag[1])),
                   vuong_goc=dict(R_m=A3_VUONG_R_M, delta_deg=A3_VUONG_DELTA_DEG, thong_tin=True,
                                  theo_clip={c: dict(n_cap=len(rows), n_co_doi=a3_perpendicular(rows))
                                             for c, rows in rows_by_clip.items() if rows}),
                   quy_ve_ma_noc=bool(A3_QUY_DUYET))
        out['loai'] = ('cuc dai thuc nghiem quy ve huong Kabsch ma noc (thiet ke v5 S5-1), KHONG phai can chung minh '
                       'cho camera day du: max qua cap, clip cua |dtheta| + gap; gap = max |dpsi_ma - dtheta_cap| giua '
                       'camera day du va camera bo cap, hai chieu, ma ao doc doan noi cap'
                       + ('' if A3_QUY_DUYET else '; CACH QUY CHUA DUOC LIAM DUYET: quy_ve_ma_noc false'))
    else:
        out['loai'] = 'cuc dai thuc nghiem goc cap moc, chua quy ve huong ma noc (khong ghi quy_ve_ma_noc)'
    out['gioi_han'] = [
        '(a) khong tru sai thuoc, nhung sai thuoc co the triet tieu mot phan sai camera: khong chung minh bao thu',
        '(b) goc cap tren doan L chi bang goc cuc bo cua ma khi dao ham truong sai bien thien it trong doan va voi '
        'moi phuong (bien dang cat giu huong duong cheo, doi huong ngang)',
        '(c) chi lay mau tai cac cap, phuong, thoi diem da co',
        '(d) do tren mat san; ma noc o do cao 0,12 m',
        '(e) camera clip H, khong phai camera tung khung khi track',
        '(f) so tru thuoc chi tham khao, gia dinh sai thuoc doc lap voi sai camera']
    if conv:
        out['gioi_han'] += [
            '(g) gap mo phong chenh lech giua cac camera da khop trong ho mo hinh (model khoa), camera day du va '
            'camera bo cap; dung no thay cho chenh lech giua camera that va camera bo cap la gia dinh; sai ngoai ho '
            '(meo khong mo hinh, san khong phang) o thang nho hon doan noi cap khong bi chan',
            '(h) phan |dtheta| dua tren lap luan tuyen tinh cuc bo (a): hai cap vuong goc cung vung cho tong 2 omega (bac nhat); '
            'dieu kien nay chi la thong tin (vuong_goc), khong phai cong',
            '(i) ma ao chi dat doc doan noi cap (A3_TAG_T) va cac huong A3_TAG_PHI_DEG; vung ngoai cac doan khong '
            'lay mau',
            '(j) camera bo cap va camera day du khop tren clip H; track khop lai tu the tung khung',
            '(k) sai chung cua moi phep do goc anh moc hieu chuan (vi du lech do tim goc theo vi tri anh ma ma noc '
            'khong chiu) khong lo ra o cap giu ra: moi camera khop cung lech, dtheta va gap bang 0 trong khi ma noc lech '
            '(Codex KCA1-1: yaw 0,01 do, lech goc anh <= 0,2 px)',
            '(l) |dtheta| + gap la chi so kiem cheo quy doi tu goc cap (so thuoc) va chenh lech giua cac camera da '
            'khop, khong phai phep do truc tiep hay can bao dam cua sai huong that, ca cua camera bo cap lan camera '
            'day du; co the thap hon sai that ngay tai mau (k); dung cho camera day du giua cac moc la gia dinh kiem '
            'cheo; chua gan voi muc bao phu xac suat',
            '(m) phep vi tu cua car_pose giu goc khi 1 - kappa > 0 (H > h); lap luan (a) can cung mot anh xa cuc bo '
            'cho hai cap vuong goc, hai camera bo cap khac nhau chua tu thoa']
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
    if not set(fit_ids) <= set(corners):
        return None                                   # KC2-2: the set must be seen whole, never shrunk to what is seen
    use = sorted(fit_ids)
    if len(use) < MIN_TRAIN:
        return None
    obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in use])
    im = np.vstack([corners[i] for i in use])
    cam, _ = cm.refit_pose(cam_k, obj, im)
    return {i: centre_floor(cam, corners[i]) for i in targets if i in corners}


def run_sets(all_ids, tap_moc):
    """The distinct mark sets of the runs (track `moc_dung`, after each exclusion pass: KC2-1) plus the full set."""
    sets = {frozenset(all_ids)}
    for S in tap_moc or ():
        T = frozenset(S) & frozenset(all_ids)
        if T:
            sets.add(T)
    return sorted(sets, key=lambda t: (len(t), sorted(t)))


def comparators(groups):
    """design v4 2: {chain: {target group t: excluded groups}}. Chain 'xe' (the car camera's mark-shift rule): mark j
    is compared by the camera without its own group. Chain k (the check camera k, run on data without group k,
    KC1-1): mark j outside k by the camera without k and without the group of j (CODE_CAM T33-4)."""
    G = len(groups)
    out = {'xe': {t: frozenset([t]) for t in range(G)}}
    for k in range(G):
        out[str(k)] = {t: frozenset([k, t]) for t in range(G) if t != k}
    return out


def windows_by_time(times, seq, win_s=WIN_S, min_n=WIN_MIN_N, gap_s=WIN_GAP_S):
    """Medians of the window of the mark-shift rule (CODE_CAM design v2 3): samples of the last continuous sighting
    within win_s before each frame, a gap above gap_s restarts it, at least min_n samples. seq: (2,) or None."""
    out, start = [], None
    last_t = None
    for e, (t, v) in enumerate(zip(times, seq)):
        if v is None:
            continue
        if last_t is None or t - last_t > gap_s:
            start = e
        last_t = t
        w = [seq[i] for i in range(start, e + 1) if seq[i] is not None and t - times[i] < win_s]
        if len(w) >= min_n:
            out.append(np.median(np.array(w), axis=0))
    return out


class GroupCams:
    """Cameras (intrinsics + pose) fitted on the H_a mean corners without a set of groups; cached."""

    def __init__(self, img, marks, groups, model, w, h, zc):
        self.img, self.marks, self.groups, self.args = img, marks, groups, (model, w, h, zc)
        self.cache = {}

    def ids_without(self, E):
        out = set().union(*[set(self.groups[e]) for e in E]) if E else set()
        return frozenset(i for i in self.img if i not in out)

    def __call__(self, E):
        E = frozenset(E)
        if E not in self.cache:
            self.cache[E] = fit_without(self.img, self.marks, sorted(self.ids_without(E)), *self.args)
        return self.cache[E]


def u_so_family(times_a, dets_a, img_b, times_b, dets_b, marks, groups, gcams, tap_moc=None, n_small=N_SMALL_SETS,
                win_s=WIN_S, min_n=WIN_MIN_N, gap_s=WIN_GAP_S):
    """u_so of every chain by its own comparators (design v4 4; KC1-1, KC1-8, KC3-1):
      baseline   of target j: MEAN over the H_a frames of the per-frame position of j (pose refitted from the camera
                 without E on every mark outside E, the set seen whole), as the track (CODE_CAM design v2 3);
      static     max over the chain's run sets S of |pos(H_b mean corners, fit S minus E) - baseline|;
      noise      for the full set and the n_small smallest sets: per frame of H_b, |pos - pos(H_b mean)| (raw) and
                 |window median - pos(H_b mean)| (filtered, the window of the rule);
      u_so = static + filtered noise, u_so_tho = static + raw noise (triangle inequality).
    tap_moc: {chain: [[ids], ...]} (KCC1-2: a chain uses only the sets of its own exclusion history), or None.
    Every target mark is checked on its own (KCC1-5): a (chain, target group, set, mark) with no static position or,
    for the noise sets, no raw sample or no window, is listed in khong_tinh_duoc; the sets fully computed are listed
    in tap_tinh (KCC1-3: tung_khung refuses track sets outside it). No H_b frames: KCError. A mark without a baseline
    in H_a: KCError (the whole session). Returns ({chain: dict}, info)."""
    if not dets_b:
        raise KCError('u_so: no H_b frames (the noise terms cannot be computed, KCC1-5)')
    all_ids = sorted(gcams.img)
    res, cache = {}, {}

    def frame_pos(E, fit, frames, targets):
        key = (id(frames), E, fit, tuple(targets))   # E = {k, t} serves chain k target t and chain t target k
        if key not in cache:
            cam = gcams(E)
            cache[key] = [pos_with(cam, fr, marks, fit, targets) for fr in frames]
        return cache[key]

    n_sets = {}
    for chain, comp in comparators(groups).items():
        sets = run_sets(all_ids, (tap_moc or {}).get(chain))
        n_sets[chain] = len(sets)
        stat = raw = filt = 0.0
        khong, tinh = [], []
        for t, E in comp.items():
            targets = groups[t]
            out_E = gcams.ids_without(E)
            cam = gcams(E)
            fa = frame_pos(E, out_E, dets_a, targets)
            base = {}
            for j in targets:
                v = [p[j] for p in fa if p is not None and j in p]
                if not v:
                    raise KCError(f'chain {chain}: mark {j} has no baseline in H_a (KC3-2: the whole session)')
                base[j] = np.mean(np.array(v), axis=0)
            fits = sorted({S & out_E for S in sets}, key=lambda s: (len(s), sorted(s)))
            small = set(fits[:n_small]) | {out_E}
            for fit in fits:
                ok = True
                pb = pos_with(cam, img_b, marks, fit, targets) or {}
                for j in targets:
                    if j not in pb:
                        khong.append(dict(nhom_dich=t, tap=sorted(fit), moc=j, vi='tinh'))
                        ok = False
                        continue
                    stat = max(stat, float(np.linalg.norm(pb[j] - base[j])))
                if fit in small:
                    fb = frame_pos(E, fit, dets_b, targets)
                    for j in targets:
                        if j not in pb:
                            continue
                        seq = [None if p is None or j not in p else p[j] for p in fb]
                        got = [v for v in seq if v is not None]
                        if got:                              # KCC2-2: the raw maximum counts even without a window
                            raw = max(raw, max(float(np.linalg.norm(v - pb[j])) for v in got))
                        wins = windows_by_time(times_b, seq, win_s, min_n, gap_s)
                        if not got or not wins:
                            khong.append(dict(nhom_dich=t, tap=sorted(fit), moc=j, vi='tho' if not got else 'cua_so'))
                            ok = False
                            continue
                        filt = max(filt, max(float(np.linalg.norm(m - pb[j])) for m in wins))
                if ok:
                    tinh.append(dict(nhom_dich=t, tap=sorted(fit)))
        res[chain] = dict(u_so_m=stat + filt, u_so_tho_m=stat + raw, tinh_m=stat, cua_so_m=filt, tho_m=raw,
                          nguong_m=NGUONG_CO_DINH_M + stat + filt, nguong_tho_m=NGUONG_CO_DINH_M + stat + raw,
                          khong_tinh_duoc=khong, tap_tinh=tinh)
    return res, dict(n_tap=n_sets, co_tap_luot=bool(tap_moc), cua_so_s=win_s, toi_thieu_mau=min_n, mat_dau_s=gap_s)


def time_check(dets_a, img_t, marks, groups, gcams):
    """Chain-xe comparison at a later clip H (H_m, H_z) with the SAME baseline and comparators as u_so (KCC1-12):
    baseline = mean over the H_a frames of the per-frame position (camera without the mark's group, full set outside
    it); position at the clip = from its mean corners. Returns the common median vector (camera change), each mark's
    own part after removing it, and the largest of both; tung_khung compares the own parts with the threshold."""
    d = {}
    for t, E in comparators(groups)['xe'].items():
        fit = gcams.ids_without(E)
        cam = gcams(E)
        fa = [pos_with(cam, fr, marks, fit, groups[t]) for fr in dets_a]
        pt = pos_with(cam, img_t, marks, fit, groups[t]) or {}
        for j in groups[t]:
            v = [p[j] for p in fa if p is not None and j in p]
            if v and j in pt:
                d[j] = pt[j] - np.mean(np.array(v), axis=0)
    if not d:
        return dict(max_m=None, rieng_m={})
    v = np.array(list(d.values()))
    med = np.median(v, axis=0)
    rest = {i: float(np.linalg.norm(x - med)) for i, x in d.items()}
    return dict(max_m=float(np.max(np.linalg.norm(v, axis=1))), chung_m=[float(med[0]), float(med[1])],
                rieng_max_m=max(rest.values()), rieng_max_id=max(rest, key=rest.get),
                rieng_m={str(i): r for i, r in rest.items()},
                d_m={str(i): [float(x[0]), float(x[1])] for i, x in d.items()})


# ---------------------------------------------------------------- P7, A5 (frame noise of the static marks)
def clip_dets(video, ids, max_frames=None):
    """(times [s, from the presentation time], per-frame refined corners {id: (4, 2)}) of the floor marks `ids` in a
    clip (c7_measure detection)."""
    det = cm.make_detector()
    times, out = [], []
    for _, t, g in cm.iter_frames(video, max_frames):
        d = cm.detect_refined(det, g)
        times.append(t)
        out.append({i: np.asarray(c, float) for i, c in d.items() if i in ids})
    if not out:
        raise KCError(f'{video}: no frame read')
    return times, out


def kiem_h(dets, ids):
    """KC2-2: share of the frames of a clip H in which EVERY mark of ids is detected at once (each mark seen in
    half the frames is not enough for the run mark sets of u_so)."""
    ids = set(ids)
    n_all = sum(1 for d in dets if ids <= set(d))
    per = {i: sum(1 for d in dets if i in d) / len(dets) for i in sorted(ids)}
    return dict(n_khung=len(dets), ti_le_cung_luc=n_all / len(dets), ti_le_moc=per)


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


A3_GEOM_TOL_M = 1e-6


def check_a3_export(out):
    """Codex KCA1-2, KCA1-3 (v5b): A3.quy_ve_ma_noc true is refused while A3_QUY_DUYET is False (whatever the input
    file says); a converted A3 must have been carried with the tag the scorer uses: ma_ao.canh_m = bo_cham.side_m,
    ma_ao.cao_m = do_cao.h_m (within A3_GEOM_TOL_M)."""
    a3 = out['A3']
    if a3.get('quy_ve_ma_noc') is True and not A3_QUY_DUYET:
        raise KCError('A3.quy_ve_ma_noc true but the method is not approved (A3_QUY_DUYET False, design v5 S5)')
    if a3.get('quy_ve_ma_noc') is True or 'max_deg_cap' in a3:
        ma = a3.get('ma_ao') or {}
        for k, blk, key in (('canh_m', 'bo_cham', 'side_m'), ('cao_m', 'do_cao', 'h_m')):
            v = ma.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise KCError(f'A3.ma_ao.{k} {v!r}: the tag A3 was carried to is not recorded (KCA1-3)')
            if abs(float(v) - float(out[blk][key])) > A3_GEOM_TOL_M:
                raise KCError(f'A3 carried to a tag with {k} {v} m, the scorer uses {blk}.{key} {out[blk][key]} m '
                              '(KCA1-3)')


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
    check_a3_export(out)
    out['nguon'] = nguon
    if ghi_chu:
        out['ghi_chu'] = ghi_chu
    return out


def write_new(path, obj):
    if os.path.exists(path):
        raise KCError(f'{path} exists: not replaced')
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- frame-by-frame P1 and coverage (tung_khung)
def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _ids(cell):
    return frozenset(int(x) for x in str(cell).split(';') if x.strip())


def _true(cell):
    return str(cell).strip() in ('1', 'True', 'true')


def run_frame_check(frames, khung, track, groups, u_so_chains, max_thieu=COVER_MAX_MISS, max_gap_s=COVER_MAX_GAP_S):
    """One run (design v4 5; Codex code round 1). frames: rows of frames.csv (idx, t_s, valid, moc_neo_<chain> for
    every chain: the anchors of that chain in that frame, every frame); khung: rows of kiem_cheo_khung.csv (khung,
    moc, nhom, ex, ey); track: track.json (moc_loai with chuoi, id, t_bd); u_so_chains: the 'chuoi' block of u_so.json.
      P1 frame part: the RAW check error |(ex, ey)| on the valid frames, no median (KC1-4); a mark stops being a target
      at the t_bd of the chain-xe schedule (KCC1-7, order-free). Every check row must be finite, of a known frame,
      mark and its own group (KCC1-6); no accepted row on a valid frame = no check value there.
      coverage: counted from the accepted rows, not from flags (KCC1-6): short of checks when more than max_thieu of
      the valid frames have none, or a gap without check is longer than max_gap_s, or no accepted row at all.
      sets: in EVERY frame (valid or not, KCC1-4), each comparator with a target of its group SEEN in the frame
      (moc_thay, KCC3-1: a seen mark left out of the anchors is still a target) must have its fit set (anchors of the
      chain minus the excluded groups), when large enough to fit, among the sets u_so computed (tap_tinh, KCC1-3);
      else short of checks.
    Returns dict(max_m (None without rows), n_valid, ti_le_thieu, khoang_mat_max_s, thieu, ly_do)."""
    G = len(groups)
    comp = comparators(groups)
    need = {'idx', 't_s', 'valid', 'moc_thay'} | {f'moc_neo_{c}' for c in comp}
    if not frames or not need <= set(frames[0]):
        raise KCError(f'frames.csv: missing columns {sorted(need - set(frames[0] if frames else ()))}')
    gmap = {i: k for k, g in enumerate(groups) for i in g}
    t_of = {int(r['idx']): float(r['t_s']) for r in frames}
    t_bd = {}
    for m in track.get('moc_loai') or []:
        if m.get('chuoi') is None:
            raise KCError('moc_loai: a record without chuoi (KCC1-7)')
        if str(m['chuoi']) == 'xe':
            j = int(m['id'])
            t_bd[j] = min(t_bd.get(j, math.inf), float(m['t_bd']))
    vset = {int(r['idx']) for r in frames if _true(r['valid'])}
    e_max, has = None, set()
    for r in khung:
        try:
            f, j, k, ex, ey = int(r['khung']), int(r['moc']), int(r['nhom']), float(r['ex']), float(r['ey'])
        except (KeyError, ValueError, TypeError):
            raise KCError(f'kiem_cheo_khung.csv: bad row {r} (KCC1-6)')
        if not (math.isfinite(ex) and math.isfinite(ey)) or f not in t_of or gmap.get(j) != k or k >= G:
            raise KCError(f'kiem_cheo_khung.csv: row not finite or of an unknown frame / mark / group: {r} (KCC1-6)')
        if f not in vset or (j in t_bd and t_of[f] >= t_bd[j]):
            continue
        e = math.hypot(ex, ey)
        e_max = e if e_max is None else max(e_max, e)
        has.add(f)
    done = {(c, x['nhom_dich'], frozenset(x['tap'])) for c, v in u_so_chains.items() for x in v.get('tap_tinh', [])}
    lost_sets = set()
    for r in frames:
        seen = _ids(r['moc_thay'])                 # KCC3-1: floor marks detected in the frame (targets), not the anchors
        for c, cmp_ in comp.items():
            A = _ids(r[f'moc_neo_{c}'])
            for tg, E in cmp_.items():
                if not any(gmap.get(i) == tg for i in seen):
                    continue                       # KCC1-4: no mark of group tg seen in this frame, the comparator is idle
                fit = frozenset(i for i in A if gmap.get(i) not in E)
                if len(fit) >= MIN_TRAIN and (c, tg, fit) not in done:
                    lost_sets.add((c, tg, tuple(sorted(fit))))
    valid = sorted((float(r['t_s']), int(r['idx'])) for r in frames if _true(r['valid']))
    miss, gap, gap_max = 0, None, 0.0
    for t, f in valid:
        if f not in has:
            miss += 1
            gap = t if gap is None else gap
            gap_max = max(gap_max, t - gap)
        else:
            if gap is not None:
                gap_max = max(gap_max, t - gap)
            gap = None
    frac = miss / len(valid) if valid else 1.0
    ly_do = []
    if not valid:
        ly_do.append('khong co khung cham xe')
    if e_max is None:
        ly_do.append('khong co so kiem nao')
    if frac > max_thieu:
        ly_do.append(f'{frac * 100:.1f} % khung cham xe khong co so kiem (> {max_thieu * 100:.0f} %)')
    if gap_max > max_gap_s:
        ly_do.append(f'khoang mat kiem {gap_max:.2f} s (> {max_gap_s} s)')
    if lost_sets:
        ly_do.append(f'{len(lost_sets)} tap moc cua bo so chua tinh u_so (KCC1-3, KCC1-4)')
    return dict(max_m=e_max, n_valid=len(valid), ti_le_thieu=frac, khoang_mat_max_s=gap_max, thieu=bool(ly_do),
                ly_do=ly_do, tap_chua_tinh=[dict(chuoi=c, nhom_dich=t, tap=list(s)) for c, t, s in sorted(lost_sets)][:50])


def read_plan(path):
    """Run ids of the session plan, one per line (or the first CSV column), '#' lines ignored (KCC1-11)."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip().split(',')[0].strip()
            if s and not s.startswith('#') and s.lower() not in ('luot', 'slot', 'run'):
                out.append(s)
    if not out or len(out) != len(set(out)):
        raise KCError(f'{path}: empty or repeated run ids')
    return out


def tung_khung(a):
    kc = _load_json(a.kiem_cheo)
    if kc.get('version') != VERSION or kc.get('buoi') != a.buoi:
        raise KCError(f'{a.kiem_cheo}: not a {VERSION} result of buoi {a.buoi}')
    if kc.get('thu'):
        raise KCError('kiem_cheo result made with --thu or --max-frames')
    us = _load_json(a.u_so)
    sha_u = sha256_file(a.u_so)
    if sha_u != kc.get('u_so_sha256'):
        raise KCError('u_so.json is not the one of the kiem_cheo result (sha256)')
    groups = kc['nhom']
    chains = us['chuoi']
    sha_ha = kc['sessions']['H_a']['sha256']
    want_ng = {c: dict(nguong_m=v['nguong_m'], nguong_tho_m=v['nguong_tho_m']) for c, v in chains.items()}
    plan = read_plan(a.ke_hoach)
    luot, e_all, loai = {}, None, set()
    for spec in a.luot:
        rid, _, d = spec.partition('=')
        if not d or rid in luot:
            raise KCError(f'--luot {spec!r}: RUN=dir, each run once')
        if rid not in plan:
            raise KCError(f'{rid}: not in the plan {a.ke_hoach}')
        tr = _load_json(os.path.join(d, 'track.json'))
        fp, kp = os.path.join(d, 'frames.csv'), os.path.join(d, 'kiem_cheo_khung.csv')
        for k, want in (('u_so_sha256', sha_u), ('session_sha256', sha_ha), ('model_khoa', kc['model']),
                        ('nhom', groups), ('hang_so_sha256', kc.get('hang_so_sha256')), ('nguong_dung', want_ng),
                        ('frames_sha256', sha256_file(fp)), ('kiem_cheo_khung_sha256', sha256_file(kp))):
            if tr.get(k) != want:
                raise KCError(f'{rid}: track {k} {tr.get(k)!r} is not {want!r} (KC1-10, KCC1-8, KCC1-9)')
        if tr.get('moc_loai') and tr.get('cham_lai') != 2:
            raise KCError(f'{rid}: marks excluded but the track was not rescored (cham_lai 2)')
        r = run_frame_check(_read_csv(fp), _read_csv(kp), tr, groups, chains)
        r['nguon'] = dict(track_sha256=sha256_file(os.path.join(d, 'track.json')), frames_sha256=tr['frames_sha256'],
                          kiem_cheo_khung_sha256=tr['kiem_cheo_khung_sha256'])
        luot[rid] = r
        if r['max_m'] is not None:
            e_all = r['max_m'] if e_all is None else max(e_all, r['max_m'])
        loai |= {int(m['id']) for m in tr.get('moc_loai') or [] if str(m.get('chuoi')) == 'xe'}   # KCC1-12
    thieu_luot = sorted(set(plan) - set(luot))
    # KCC1-12: a mark whose own shift at H_m / H_z is above the chain-xe threshold but that no run excluded
    ng = chains['xe']['nguong_m']
    mau = sorted({(n, int(i)) for n, tc in (kc.get('theo_thoi_gian') or {}).items() if n in ('H_m', 'H_z')  # KCC2-1
                  for i, r in (tc.get('rieng_m') or {}).items() if r > ng and int(i) not in loai})
    out = dict(version=TUNG_KHUNG_VERSION, buoi=a.buoi, model=kc['model'], nhom=groups, moc=kc.get('moc'),
               hang_so_sha256=kc.get('hang_so_sha256'), kiem_cheo_sha256=sha256_file(a.kiem_cheo), u_so_sha256=sha_u,
               session_sha256=sha_ha, cham_lai=True, ke_hoach=dict(file=a.ke_hoach, sha256=sha256_file(a.ke_hoach),
                                                                   luot=plan),
               tung_khung_max_m=e_all, luot=luot, luot_thieu=sorted(r for r, v in luot.items() if v['thieu']),
               luot_chua_co=thieu_luot, mau_thuan=[dict(clip=n, moc=i) for n, i in mau],
               hoan_tat=not thieu_luot and not mau and e_all is not None,
               luat=dict(max_thieu=COVER_MAX_MISS, max_gap_s=COVER_MAX_GAP_S))
    write_new(a.out, out)
    print(f"tung_khung: {len(luot)}/{len(plan)} luot, max "
          f"{'-' if e_all is None else f'{e_all * 1000:.2f} mm'}; thieu kiem: {out['luot_thieu']}; chua co: "
          f"{thieu_luot}; mau thuan: {out['mau_thuan']}; hoan tat: {out['hoan_tat']}")


# ---------------------------------------------------------------- CLI
def _load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def thu_flag(thu, max_frames):
    """KCC1-10: a result made from a cut clip (--max-frames) is a trial, never exported."""
    return bool(thu or max_frames)


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
    need = {i for i in moc_rows if i != 0}
    for n, (img, *_r) in clips.items():
        lack = sorted(need - set(img))
        if lack and not a.thu:
            raise KCError(f'{n}: measured floor marks {lack} not in the session fit (no baseline, KCC1-14)')
    sig = check_moc(moc_rows, mk_a, sorted(set().union(*[set(c[0]) for c in clips.values()])), a.thu)
    groups = session_groups(raw['H_a'], ids_a, a.groups)
    tap = None
    if a.tap_moc:                                    # {chain: [[ids]...]}, every pass of KC2-1 merged (KCC1-2)
        tap = {str(c): [sorted(int(i) for i in S) for S in v] for c, v in _load_json(a.tap_moc).items()}
    a.thu = thu_flag(a.thu, a.max_frames)
    if not a.thu and not a.hang_so:
        raise KCError('--hang-so needed (KCC1-9)')
    # A3 over every clip H, carried to the car tag of the sessions (design v5 S5-1)
    tag = session_tag(raw)
    a3 = {n: a3_clip(img, mk, a.model, w, h, zc, a.L_max, sig, tag=tag) for n, (img, mk, w, h, zc) in clips.items()}
    roles = {i: r.get('vai_tro', '') for i, r in moc_rows.items()}
    A3 = a3_summary(a3, [i for i in ids_a if roles.get(i) == 'moc_giua'], tag=tag)
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
    gc = GroupCams(img_a, mk_a, groups, a.model, *clips['H_a'][2:])
    chuoi, uso_info = u_so_family(dets['H_a'][0], dets['H_a'][1], clips['H_b'][0], dets['H_b'][0], dets['H_b'][1],
                                  mk_a, groups, gc, tap)
    xe = chuoi['xe']
    uso = max(xe['u_so_m'], xe['u_so_tho_m'])        # the car chain detects by both thresholds (CODE_CAM v2 3)
    nz = {n: noise_clip(dets[n][1], clips[n][1], groups, cams[n]) for n in clips}
    allz = [(n, i, r) for n, d in nz.items() for i, r in d.items()]
    zp = max(allz, key=lambda t: t[2]['lech_max_m'])
    za = max(allz, key=lambda t: t[2]['lech_max_deg'])
    sess = {n: dict(file=p, sha256=sha256_file(p)) for n, (p, _) in clips_in.items()}
    common = dict(buoi=a.buoi, model=a.model, nhom=groups, sessions=sess, thu=bool(a.thu),
                  moc=[dict(file=p, sha256=sha256_file(p)) for p in a.moc or ()],
                  hang_so_sha256=sha256_file(a.hang_so) if a.hang_so else None)
    res = dict(version=VERSION, **common, L_max_m=a.L_max,
               A3=dict(A3, buoi=a.buoi),
               P1=dict(bo_nhom_max_m=worst[2], tung_khung_max_m=None, max_clip=worst[0], max_id=worst[1], buoi=a.buoi,
                       tung_khung='tu file tung_khung (lenh tung_khung, track sau cham lai cua CODE_CAM)',
                       theo_clip={n: {str(i): [float(v[0]), float(v[1])] for i, v in d.items()} for n, d in p1.items()}),
               moc_dich=dict(nguong_m=NGUONG_CO_DINH_M, u_so_m=uso, buoi=a.buoi, chuoi_xe=xe, **uso_info,
                             nghia='u_so_m = max(u_so, u_so_tho) cua chuoi xe; nguong T = nguong_m + u_so; neu u_so '
                                   'chan sai phep so thi dich chua phat hien <= T + u_so (KC1-3)'),
               P7=dict(lech_max_m=zp[2]['lech_max_m'], max_clip=zp[0], max_id=zp[1], buoi=a.buoi,
                       sd_max_m=max(r['sd_m'] for _, _, r in allz)),
               A5=dict(lech_max_deg=za[2]['lech_max_deg'], max_clip=za[0], max_id=za[1], buoi=a.buoi,
                       sd_max_deg=max(r['sd_deg'] for _, _, r in allz),
                       gioi_han='moc san 96,4 mm, ma noc 144,5 mm: nhieu goc moc san lon hon; so khung = cac clip H, '
                                'lech lon nhat tang theo so mau; moc dung yen, khong gom nhoe (P4, A4)'),
               nhieu_theo_clip={n: {str(i): r for i, r in d.items()} for n, d in nz.items()},
               theo_thoi_gian={n: time_check(dets['H_a'][1], clips[n][0], mk_a, groups, gc)
                               for n in clips if n != 'H_a'})
    os.makedirs(a.out, exist_ok=True)
    up = os.path.join(a.out, 'u_so.json')
    write_new(up, dict(version=U_SO_VERSION, buoi=a.buoi, model=a.model, nhom=groups, chuoi=chuoi, chi_tiet=uso_info,
                       sessions=sess, tao_luc=datetime.datetime.now().isoformat(timespec='seconds')))
    res['u_so_sha256'] = sha256_file(up)              # KC1-10: the final track must say it used this file
    res['kiem_h'] = {n: kiem_h(dets[n][1], clips[n][0]) for n in clips}
    write_new(os.path.join(a.out, 'kiem_cheo.json'), res)
    with open(os.path.join(a.out, 'a3_cap.csv'), 'x', newline='', encoding='utf-8') as f:
        cols = ['clip', 'i', 'j', 'L_m', 'dtheta_deg', 'sigma_thuoc_deg', 'mid_x', 'mid_y', 'huong_deg', 'gap_deg',
                'quy_deg', 'dpsi_ma_max_deg', 'dtheta_hai_cam_deg']
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for n, rows in a3.items():
            for r in rows:
                wr.writerow(dict({k: r.get(k, '') for k in cols[1:]}, clip=n))
    print(f"A3: {A3['n_cap']} cap, quy ve ma noc max {A3['max_deg']:.3f} deg (cap {A3['max_deg_cap']:.3f}, gap "
          f"{A3['gap_max_deg']:.3f}; quy_ve_ma_noc {A3['quy_ve_ma_noc']}), RMS cap {A3['rms_deg']:.3f} deg; "
          f"P1 clip H: max {worst[2] * 1000:.2f} mm (moc {worst[1]}, {worst[0]}); u_so xe {uso * 1000:.2f} mm "
          f"(tinh {xe['tinh_m'] * 1000:.2f}, loc {xe['cua_so_m'] * 1000:.2f}, tho {xe['tho_m'] * 1000:.2f}); "
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
    for k, want in (('u_so_sha256', kc.get('u_so_sha256')), ('session_sha256', kc['sessions']['H_a']['sha256'])):
        if tk.get(k) != want:
            raise KCError(f'frame-by-frame file: {k} {tk.get(k)!r} is not the one of the kiem_cheo result (KC1-10)')
    if sorted(m['sha256'] for m in tk.get('moc') or []) != sorted(m['sha256'] for m in kc.get('moc') or []):
        raise KCError('frame-by-frame file: another marker file (KC1-10)')
    if tk.get('hang_so_sha256') != kc.get('hang_so_sha256'):
        raise KCError('frame-by-frame file: another constants file (KCC1-9)')
    if not tk.get('cham_lai'):
        raise KCError('frame-by-frame file: track not rescored after the mark-shift rule (cham_lai missing)')
    if not tk.get('hoan_tat') or tk.get('luot_chua_co') or tk.get('mau_thuan'):
        raise KCError(f"frame-by-frame file not complete: runs missing {tk.get('luot_chua_co')}, contradictions "
                      f"{tk.get('mau_thuan')} (KCC1-11, KCC1-12)")
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
    own['P1']['luot_thieu_kiem'] = tk.get('luot_thieu', [])          # KCC1-11: these runs stay not concluded
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


def kiem_h_cli(a):
    """In-session check right after a clip H (design v3 7): refuses below the share (quay lai clip H)."""
    s = _load_json(a.session)
    ids = sorted(i for i in load_moc(a.moc) if i != 0)  # KCC1-14: every measured floor mark, not the session's fit_ids
    if not ids:
        raise KCError('no mark in the marker file')
    missing = sorted(set(ids) - {int(i) for i in s['fit_ids']})
    r = kiem_h(clip_dets(a.video, set(ids))[1], ids)
    if missing:
        raise KCError(f'clip H khong dat: moc {missing} khong co trong session (KCC1-14), quay lai clip H')
    low = [i for i, v in r['ti_le_moc'].items() if v < a.nguong]
    print(f"kiem_h: {r['n_khung']} khung, cung luc moi moc {r['ti_le_cung_luc'] * 100:.1f} % (can >= "
          f"{a.nguong * 100:.0f} %); moc duoi nguong: {low}")
    if r['ti_le_cung_luc'] < a.nguong:
        raise KCError('clip H khong dat: quay lai clip H (KC2-2)')


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
    p.add_argument('--hang-so', dest='hang_so', help='constants file of the session (c7_score), sha recorded')
    p.add_argument('--tap-moc', dest='tap_moc', help='JSON {"tap": [[ids], ...]}: mark sets used by track in the runs')
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
    t = sub.add_parser('tung_khung')
    t.add_argument('--buoi', required=True)
    t.add_argument('--kiem-cheo', dest='kiem_cheo', required=True)
    t.add_argument('--u-so', dest='u_so', required=True)
    t.add_argument('--ke-hoach', dest='ke_hoach', required=True, help='run ids of the session plan (KCC1-11)')
    t.add_argument('--luot', action='append', required=True, help='RUN=<track dir> (frames.csv, kiem_cheo_khung.csv, '
                   'track.json of CODE_CAM)')
    t.add_argument('--out', required=True)
    h = sub.add_parser('kiem_h')
    h.add_argument('--video', required=True)
    h.add_argument('--session', required=True)
    h.add_argument('--nguong', type=float, default=KIEM_H_MIN)
    h.add_argument('--moc', nargs='+', required=True, help='measured marker CSVs: the marks that must be seen')
    a = ap.parse_args(argv)
    try:
        {'phan_tich': phan_tich, 'xuat': xuat, 'kiem_h': kiem_h_cli, 'tung_khung': tung_khung}[a.cmd](a)
    except (KCError, OSError, ValueError, KeyError) as e:
        print(f'loi: {e}')
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
