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
  xuat    c8_cam_u_vao.json for c8b_u (CODE_C8B; version c8_cam_u_vao_v3, c8b_u 198d2e9): blocks P1, A3, P7, A5 from
          this file's result, moc_dich = the U_moc block of CODE_CAM (--moc-dich) + tien_de_buoi (--tien-de), the
          other blocks from part files (--phan NAME=file.json), every number finite and >= 0, no default, sha256 of
          every source in `nguon`; an existing file is never replaced. (v4 with the S7 blocks: after C8B, NHACTAM
          confirm F15.)
  S7      (design v6 S8-4; with --xuat-phat goc|K1|K2=X,Y and --plumb X,Y) P1_tuong_doi (d_sj = measured - tape,
          d_sj_he_dau, chuyen_sj = held - full; signed per pair in cap_tuong_doi.csv), A3_tuong_doi (per end), and
          always meo_cuc_bo (diagnostic), thang_chung; hang_so_s7 and commit_hil recorded (S7-16 lock).
  tung_khung  per run: frame-by-frame P1, coverage, contradictions per run (TD1-2), thieu_du_lieu_neo (TD2-1).
  tien_de tien_de_buoi_v1 of a session (design S6, approved): dat / thieu_du_lieu / khong_dat with reasons.
  e_hieu  F9 e_hieu_m per leg of a doan_12b_v1 (F5) for CODE_HIEUCHUAN: parts (i), (ii); (iii) after the CAM merge.

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

VERSION = 'c7_kiem_cheo_v7'                  # v7: relative scoring S7, video paths, ID lock (design v6 S8); v6: A3 to the tag
U_SO_VERSION = 'c7_u_so_v4'
TUNG_KHUNG_VERSION = 'c7_tung_khung_v4'      # v4: contradictions per run, unique frames, thieu_du_lieu_neo (S6-9, S6-10b)
CAM_U_VERSION = 'c8_cam_u_vao_v3'           # c8b_u v3 198d2e9 (moc_dich = U_moc of CODE_CAM + tien_de_buoi)
TIEN_DE_FORMAT = 'tien_de_buoi_v1'           # design S6 (approved: Liam 18/11 at CODE_KIEMCHEO, "Duyet (Recommended)")
E_HIEU_SCHEMA = 'c7_e_hieu_v1'               # F9 (contract 2026-11-19 v1): read by c8_hc_a.read_e_hieu (CODE_HIEUCHUAN)
DOAN_SCHEMA = 'c8_hc_doan_12b_v1'            # F5 (CODE_HIEUCHUAN), confirmed 19/11
CHO = ('goc', 'K1', 'K2')                    # place names of every file (CODE 19/11)
# S7 constants (design v6 S8-4; [DE XUAT], locked by commit + SHA before phan_tich runs on the real clips H, S7-16)
R_XP_M = 0.6                                 # marks of a start region: within R_XP of the start point
R_TRONG_M = 0.55                             # every sample of the start domain within R_TRONG of a region mark
#   0.35 -> 0.55: Liam 20/11/2026 at CODE_KIEMCHEO (local_599d3188), choice box, verbatim: "Nang len 0,55 m
#   (Recommended)" -- on the locked 32-mark floor (grid ~0.7 m) the largest distance to the nearest mark is 0.47-0.51 m
#   at goc / K1 / K2 (start discs and 12b corridors), so 0.35 always flagged khoang_trong (HCV3-2, CODE 20/11)
# 12b corridor of every place (HCV3-2, CODE 20/11): the car moves along its axis between these stops [m] from the start
# point, nose at this floor heading [deg]; source rpi5/c8_hc_seq.py PLACES and its docstring (hil 8c2aa2e)
DUONG_12B = {'goc': (45.0, -0.50, 0.50), 'K1': (90.0, -0.25, 0.25), 'K2': (0.0, -0.35, 0.15)}
R_XP_DUONG_M = 0.8                           # [DE XUAT] marks of a 12b corridor: within this of the 12b segment (the
                                             #   smallest 0.05 step with every corridor inside its region hull on the
                                             #   32-mark floor; 0.6 / 0.7 leave goc or K1 outside)
MIEN_XP_M = 0.10                             # start domain before the session: disc about the nominal point (by eye)
MIEN_XP_BUOC_M = 0.005                       #   grid step of the coverage test (KCJ1-11: 1-Lipschitz margin added)
ID_KHOA = tuple(range(1, 33))                # KS7-7, KCJ1-10: the floor marks of the official path (0 = car tag)
HAI_PHUONG_DEG = 60.0                        # two A3 pair directions (mod 180) at least this apart, else mot_phuong
DUNG_SAI_IN_GIAN = 0.002                     # print tolerance of the mark square: stretch
DUNG_SAI_IN_DU_MM = 0.3                      #   non-affine residual [mm]
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
# Camera segments (Liam 20/11/2026 at CODE_KIEMCHEO local_599d3188, choice box, verbatim: "Theo doan, bo H giua
# (Recommended)"): a session whose iPhone was taken off and put back is checked per camera SEGMENT (phan_tich --doan):
# the two clips H at the start of the segment are H_a, H_b, the clip H after the last run of the segment is H_z (the
# drift check), no H_m (a drift that happens and comes back inside the segment is not seen: limit); a segment without
# an end clip (--khong-H-cuoi) has no drift check at all (limit 'khong co H cuoi').
DOAN_GIOI_HAN = ('doan camera (Liam 20/11 "Theo doan, bo H giua"): khong co clip H giua; troi xay ra roi tu ve giua '
                 'doan khong thay')
DOAN_KHONG_CUOI = 'doan camera khong co H cuoi: khong kiem troi trong doan'


def vai_tro(kc):
    """Roles of the clips H a kiem_cheo result must have: the four of design v2 7, or those of its camera segment."""
    d = (kc or {}).get('doan')
    if not d:
        return CLIPS_REQUIRED
    return tuple(n for n in CLIPS_REQUIRED if n in (d.get('vai_tro_co') or ()))
GROUPS = 4                                   # [DE XUAT] CODE_CAM design 2: ids[k::4]
MIN_TRAIN = 4                                # 4 marks = 16 points: pose determined (as c7_measure.track)
# blocks of c8_cam_u_vao_v2 and the numbers c8b_u.u_p_u_psi reads (CODE_C8B 18/11); other keys in a block are
# information; 'than_dich' is information only; bo_cham.tag_offset is checked apart (null or 3 signed numbers)
CAM_BLOCKS = {
    'P1': ('bo_nhom_max_m', 'tung_khung_max_m'),
    'moc_dich': ('U_moc_xy_m', 'U_moc_psi_deg'),     # v3: the U_moc block of CODE_CAM (--moc-dich) + tien_de_buoi
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
OWN_BLOCKS = ('P1', 'A3', 'P7', 'A5')
MOC_NGUON_KEYS = ('frames', 'lich', 'tap_moc', 'u_so', 'session', 'car_pose', 'bo_lay_mau')   # c8b_u v3 moc_dich.nguon


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
    if len(ids) != len(set(ids)):
        raise KCError(f'session fit_ids repeat an id: {sorted(i for i in set(ids) if ids.count(i) > 1)} (KCJ1-10)')
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
                    if int(r['id']) in rows:
                        raise KCError(f'{p}: mark id {r["id"]} repeated (KCJ1-10)')
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


def centre_floor4(cam, corners):
    """S7 centre (KS7-7, KCJ1-9; the definition C33-6 brings to every path with cm.mark_centre_floor): the MEAN of the
    4 corners back-projected to z = 0 (the square's centre for an exact camera, whatever the perspective)."""
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
                if not (math.isfinite(d) and math.isfinite(dp)):        # KCJ1-13: each sample before the max
                    raise ValueError('non-finite tag / pair angle sample')
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


class PairCams:
    """Cameras of one clip fitted without a set of marks (the pair of A3 / S7, the mark of the local distortion, the
    full set for the scoring camera), cached so A3 and S7 share the fits (design v6 S8-4 (2))."""

    def __init__(self, img, marks, model, w, h, zc):
        self.img, self.marks, self.args = img, marks, (model, w, h, zc)
        self.ids = sorted(img)
        self.cache = {}

    def __call__(self, out=()):
        key = frozenset(out)
        if key not in self.cache:
            self.cache[key] = fit_without(self.img, self.marks, [k for k in self.ids if k not in key], *self.args)
        return self.cache[key]


def a3_clip(img, marks, model, w, h, zc, l_max=L_MAX_M, sigma=None, tag=None, pc=None, hong=None):
    """Rows per pair (i < j, tape distance <= l_max), both left out of the fit. tag = (car_side_m, tag_h_m): each
    row also gets the conversion to the car tag (design v5 S5-1): gap_deg (b), quy_deg = |dtheta| + gap_deg.
    pc: a PairCams of the same clip (shared fits), or None. hong: a list; a pair whose fit or conversion fails is
    appended to it (KCJ1-13) instead of stopping the run; without it the error is raised."""
    ids = sorted(img)
    rows = []
    pc = pc or PairCams(img, marks, model, w, h, zc)
    cam_full = pc() if tag is not None else None
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            vt = np.array(marks[j][:2]) - np.array(marks[i][:2])
            L = float(np.linalg.norm(vt))
            if L > l_max:
                continue
            try:                                            # KCJ1-13: fit, angle and tag conversion of the pair
                cam = pc((i, j))
                vc = centre_floor(cam, img[j]) - centre_floor(cam, img[i])
                r = dict(i=i, j=j, L_m=L, dtheta_deg=signed_angle_deg(vt, vc),
                         mid_x=float((marks[i][0] + marks[j][0]) / 2), mid_y=float((marks[i][1] + marks[j][1]) / 2),
                         huong_deg=math.degrees(math.atan2(vt[1], vt[0])))
                if tag is not None:
                    g, dpsi, dp = a3_tag_gap(cam_full, cam, marks[i], marks[j], *tag)
                    r.update(gap_deg=g, dpsi_ma_max_deg=dpsi, dtheta_hai_cam_deg=dp, quy_deg=abs(r['dtheta_deg']) + g)
                if not all(math.isfinite(v) for v in r.values()):
                    raise KCError('non-finite result')
            except Exception as e:
                if hong is None:
                    raise
                hong.append(dict(i=i, j=j, L_m=L, ly_do=str(e)[:200]))
                continue
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
    d, base, now = {}, {}, {}
    for t, E in comparators(groups)['xe'].items():
        fit = gcams.ids_without(E)
        cam = gcams(E)
        fa = [pos_with(cam, fr, marks, fit, groups[t]) for fr in dets_a]
        pt = pos_with(cam, img_t, marks, fit, groups[t]) or {}
        for j in groups[t]:
            v = [p[j] for p in fa if p is not None and j in p]
            if v and j in pt:
                base[j], now[j] = np.mean(np.array(v), axis=0), pt[j]
                d[j] = pt[j] - base[j]
    if not d:
        return dict(max_m=None, rieng_m={})
    v = np.array(list(d.values()))
    med = np.median(v, axis=0)
    rest = {i: float(np.linalg.norm(x - med)) for i, x in d.items()}
    out = dict(max_m=float(np.max(np.linalg.norm(v, axis=1))), chung_m=[float(med[0]), float(med[1])],
               rieng_max_m=max(rest.values()), rieng_max_id=max(rest, key=rest.get),
               rieng_m={str(i): r for i, r in rest.items()},
               d_m={str(i): [float(x[0]), float(x[1])] for i, x in d.items()})
    if len(d) >= 2:   # C8BX-06 (information): turn and stretch of the targets left AFTER the pose refit (a rigid camera
        #   move is absorbed by the refit, a floor change under a still camera is not); not the eps_H,psi test of J v6
        ids = sorted(d)
        ang, _, sc, _ = cm.rigid_fit(np.array([base[i] for i in ids]), np.array([now[i] for i in ids]), with_scale=True)
        out.update(xoay_chung_deg=math.degrees(ang), gian_chung=float(sc - 1.0))
    return out


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
    return dict(n_khung=len(dets), ti_le_cung_luc=n_all / len(dets), ti_le_moc=per,
                khung_loai=dict(n=len(dets) - n_all, ly_do='thieu_moc'))      # C8BX-02


def mark_heading(floor_corners, side):
    """Heading [rad] of a mark from its 4 floor corners (rigid fit to the square, as car_pose)."""
    return cm.rigid_fit(cp.mark_corners_floor(0.0, 0.0, side), floor_corners)[0]


def noise_clip(dets, marks, groups, cams):
    """Per mark: largest deviation from its own mean over the frames, centre [m] and heading [deg]. Each frame, the
    pose is refitted on the marks outside the mark's group (intrinsics of the camera without the group), so the
    mark does not steady its own pose. {id: dict(n, lech_max_m, lech_max_deg, sd_m, sd_deg)}."""
    cen, ang, pix, edge = {}, {}, {}, {}
    for d in dets:
        for i, c in d.items():                        # C8BX-01: image position and edge angle of every mark
            pix.setdefault(i, []).append(np.asarray(c, float).mean(axis=0))
            e = np.asarray(c, float)[1] - np.asarray(c, float)[0]
            edge.setdefault(i, []).append(math.atan2(e[1], e[0]))
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
        uv = np.mean(np.array(pix[i]), axis=0)
        out[i] = dict(n=len(c), lech_max_m=float(dc.max()), lech_max_deg=float(da.max()),
                      sd_m=float(np.sqrt(np.mean(dc ** 2))), sd_deg=float(np.sqrt(np.mean(da ** 2))),
                      u_px=float(uv[0]), v_px=float(uv[1]),
                      goc_canh_deg=float(np.degrees(np.mean(np.unwrap(np.array(edge[i]))))),
                      canh_m=float(marks[i][2]))
    return out


def noise_by_size(allz):
    """C8BX-01: largest P7, A5 deviation per mark size [mm of the black side]."""
    co = {}
    for _, _, r in allz:
        k = f"{r['canh_m'] * 1000:.1f}"
        b = co.setdefault(k, dict(lech_max_m=0.0, lech_max_deg=0.0, n_moc=0))
        b.update(lech_max_m=max(b['lech_max_m'], r['lech_max_m']), lech_max_deg=max(b['lech_max_deg'], r['lech_max_deg']),
                 n_moc=b['n_moc'] + 1)
    return co


# ---------------------------------------------------------------- S7: relative scoring (design v5 S7, v6 S8-4)
def _hull(pts):
    """Convex hull (counter-clockwise, Andrew's monotone chain) of 2D points; fewer than 3 distinct: the points."""
    P = sorted(set(map(tuple, np.asarray(pts, float).tolist())))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def disc_in_hull(c, r, hull):
    """KCJ1-11: the WHOLE disc (centre c, radius r) inside the convex polygon hull (counter-clockwise): the centre is at
    least r inside every edge line. A hull of fewer than 3 points contains nothing."""
    H = np.asarray(hull, float).reshape(-1, 2) - np.asarray(c, float)   # KCJ2: relative coordinates
    if len(H) < 3 or abs(sum(H[k, 0] * H[(k + 1) % len(H), 1] - H[(k + 1) % len(H), 0] * H[k, 1]
                             for k in range(len(H)))) <= 1e-18:
        return False                                  # degenerate polygon (repeated or aligned vertices)
    n = len(hull)
    for k in range(n):
        a, b = np.array(hull[k]), np.array(hull[(k + 1) % n])
        e = b - a
        L = float(np.linalg.norm(e))
        if L == 0.0:
            continue
        if (e[0] * (c[1] - a[1]) - e[1] * (c[0] - a[0])) / L < r - 1e-12:
            return False
    return True


def disc_max_gap(c, r, pts, step=MIEN_XP_BUOC_M):
    """KCJ1-11: an upper bound of max over the disc (centre c, radius r) of the distance to the nearest point of pts:
    the largest distance over a square grid of the disc (step) and 720 rim points, plus step * sqrt(2) + pi r / 720
    (the distance function is 1-Lipschitz; a disc point lies in a grid cell: a corner of it inside the disc is within
    step * sqrt(2), and when no corner is inside, the rim crosses the cell, so a rim sample is within step * sqrt(2) +
    half the rim spacing)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts) or not step > 0 or not r >= 0:
        raise KCError('disc_max_gap: no mark, or a step / radius out of range')
    pts = pts - np.asarray(c, float)                  # relative coordinates (no cancellation far from the origin)
    c = np.zeros(2)
    g = np.arange(-r, r + step / 2, step)
    X, Y = np.meshgrid(g, g)
    P = np.column_stack([X.ravel(), Y.ravel()])
    P = P[np.hypot(P[:, 0], P[:, 1]) <= r]
    ang = np.linspace(0.0, 2 * math.pi, 720, endpoint=False)
    P = np.vstack([P, r * np.column_stack([np.cos(ang), np.sin(ang)])]) + c
    d = np.min(np.linalg.norm(P[:, None, :] - np.asarray(pts, float)[None, :, :], axis=2), axis=1)
    return float(d.max()) + step * math.sqrt(2) + math.pi * r / 720


def start_regions(marks, xp, r_xp=R_XP_M):
    """{k: dict(moc, vung_mo_rong)} of every start point xp {k: (x, y)}: the marks within r_xp; fewer than 2 -> the
    3 nearest (vung_mo_rong). marks {id: (x, y, side)}."""
    out = {}
    for k, c in xp.items():
        d = {i: math.hypot(m[0] - c[0], m[1] - c[1]) for i, m in marks.items()}
        z = sorted(i for i, v in d.items() if v <= r_xp)
        wide = len(z) < 2
        if wide:
            z = sorted(sorted(d, key=lambda i: (d[i], i))[:3])
        out[k] = dict(moc=z, vung_mo_rong=wide)
    return out


def seg_dist(p, a, b):
    """Distance from the point p to the segment a-b (a == b: to the point)."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0.0 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def duong_12b(xp, spec):
    """(a, b) floor ends of the 12b segment of a place: start point xp, spec = (nose heading deg, lo, hi) of
    DUONG_12B."""
    ang, lo, hi = spec
    u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    c = np.asarray(xp, float)
    return c + lo * u, c + hi * u


def capsule_in_hull(a, b, r, hull):
    """The whole capsule (segment a-b widened by r) inside the convex polygon hull: the capsule is the convex hull of
    its two end discs, so it is inside a convex polygon exactly when both end discs are (disc_in_hull)."""
    return disc_in_hull(a, r, hull) and disc_in_hull(b, r, hull)


def capsule_max_gap(a, b, r, pts, step=MIEN_XP_BUOC_M):
    """Upper bound of max over the capsule (segment a-b widened by r) of the distance to the nearest point of pts, as
    disc_max_gap: largest distance over a square grid of the capsule (step) and its rim (two half circles and two
    sides, spacing h <= step), plus step * sqrt(2) + h / 2 (1-Lipschitz; a capsule point lies in a grid cell whose
    corner is inside, or the rim crosses the cell)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts) or not step > 0 or not r > 0:
        raise KCError('capsule_max_gap: no mark, or a step / radius out of range')
    a, b = np.asarray(a, float), np.asarray(b, float)
    o = (a + b) / 2                                   # relative coordinates
    pts, a, b = pts - o, a - o, b - o
    lo, hi = np.minimum(a, b) - r, np.maximum(a, b) + r
    gx, gy = np.arange(lo[0], hi[0] + step / 2, step), np.arange(lo[1], hi[1] + step / 2, step)
    X, Y = np.meshgrid(gx, gy)
    P = np.column_stack([X.ravel(), Y.ravel()])
    ab = b - a
    L = float(np.linalg.norm(ab))
    t = np.clip(((P - a) @ ab) / (L * L), 0.0, 1.0) if L > 0 else np.zeros(len(P))
    P = P[np.linalg.norm(P - (a + t[:, None] * ab), axis=1) <= r]
    n_arc = max(720, int(math.ceil(math.pi * r / step)))
    h = math.pi * r / n_arc
    u = ab / L if L > 0 else np.array([1.0, 0.0])
    nrm = np.array([-u[1], u[0]])
    th = np.linspace(-math.pi / 2, math.pi / 2, n_arc + 1)
    rim = [b + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), nrm)),
           a - r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), nrm))]
    if L > 0:
        s = np.linspace(0.0, 1.0, max(2, int(math.ceil(L / h)) + 1))[:, None]
        rim += [a + r * nrm + s * ab, a - r * nrm + s * ab]
    P = np.vstack([P] + rim)
    d = np.min(np.linalg.norm(P[:, None, :] - pts[None, :, :], axis=2), axis=1)
    return float(d.max()) + step * math.sqrt(2) + h / 2


def duong_regions(marks, xp, r=R_XP_DUONG_M, duong=None):
    """{k: dict(doan_m, moc, vung_mo_rong)} of every start point xp {k: (x, y)} with a 12b segment (DUONG_12B): the
    marks within r of the segment; fewer than 2 -> the 3 nearest (vung_mo_rong), as start_regions."""
    duong = DUONG_12B if duong is None else duong
    out = {}
    for k, c in xp.items():
        if k not in duong:
            continue
        a, b = duong_12b(c, duong[k])
        d = {i: seg_dist(m[:2], a, b) for i, m in marks.items()}
        z = sorted(i for i, v in d.items() if v <= r)
        wide = len(z) < 2
        if wide:
            z = sorted(sorted(d, key=lambda i: (d[i], i))[:3])
        out[k] = dict(doan_m=[[float(v) for v in a], [float(v) for v in b]], moc=z, vung_mo_rong=wide)
    return out


def _a3_dir_flags(region_ids, a3_rows, two_deg, l_max):
    """mot_phuong reasons of a region (S7-14.3, KCJ1-12): no two A3 pair directions (mod 180) two_deg apart among the
    valid pairs with an end in the region, and region marks in no A3 pair of the clip."""
    zs = set(region_ids)
    use = [r for r in a3_rows if (r['i'] in zs or r['j'] in zs) and r['L_m'] <= l_max]
    dirs = [r['huong_deg'] % 180.0 for r in use]
    two = any(min(abs(a - b), 180.0 - abs(a - b)) >= two_deg for a in dirs for b in dirs)
    held = {m for r in use for m in (r['i'], r['j'])}
    lone = sorted(m for m in zs if m not in held)
    return (['mot_phuong'] if not two else []) + ([f'moc_khong_co_A3loc:{lone}'] if lone else [])


def duong_cover(region_ids, marks, a, b, a3_rows, r_trong=R_TRONG_M, two_deg=HAI_PHUONG_DEG, l_max=L_MAX_M):
    """region_cover for a 12b corridor (segment a-b widened by MIEN_XP_M): ngoai_suy when the capsule is not inside
    the hull of the region marks or a capsule point is farther than r_trong from the nearest of them (guaranteed
    upper bound); mot_phuong as region_cover. Also returns the bound itself (khoang_trong_max_m)."""
    pts = np.array([marks[i][:2] for i in region_ids], float)
    why = []
    if not capsule_in_hull(a, b, MIEN_XP_M, _hull(pts)):
        why.append('mien_ngoai_bao_loi')
    gap = capsule_max_gap(a, b, MIEN_XP_M, pts)
    if gap > r_trong:
        why.append('khoang_trong')
    mp = _a3_dir_flags(region_ids, a3_rows, two_deg, l_max)
    return dict(ngoai_suy=bool(why), mot_phuong=bool(mp), ly_do=why + mp, khoang_trong_max_m=gap)


def region_cover(region_ids, marks, c, a3_rows, r_trong=R_TRONG_M, two_deg=HAI_PHUONG_DEG, l_max=L_MAX_M):
    """S7-12.3, S7-14.3: ngoai_suy when a sample of the start domain is outside the convex hull of the region marks
    or farther than r_trong from the nearest of them; mot_phuong when the valid A3 pairs of the clip (L <= L_MAX_M,
    at least one end in the region) do not have two directions (mod 180) at least two_deg apart."""
    pts = np.array([marks[i][:2] for i in region_ids], float)
    hull = _hull(pts)
    why = []
    if not disc_in_hull(c, MIEN_XP_M, hull):                     # KCJ1-11: the whole disc, not samples
        why.append('mien_ngoai_bao_loi')
    if disc_max_gap(c, MIEN_XP_M, pts) > r_trong:                # KCJ1-11: guaranteed upper bound over the disc
        why.append('khoang_trong')
    zs = set(region_ids)
    use = [r for r in a3_rows if (r['i'] in zs or r['j'] in zs) and r['L_m'] <= l_max]   # KCJ1-12: the A3 pairs
    dirs = [r['huong_deg'] % 180.0 for r in use]
    two = any(min(abs(a - b), 180.0 - abs(a - b)) >= two_deg for a in dirs for b in dirs)
    held = {m for r in use for m in (r['i'], r['j'])}
    lone = sorted(m for m in zs if m not in held)              # KCJ1-12: a region mark in no A3 pair of the clip
    mp = []
    if not two:
        mp.append('mot_phuong')
    if lone:
        mp.append(f'moc_khong_co_A3loc:{lone}')
    return dict(ngoai_suy=bool(why), mot_phuong=bool(mp), ly_do=why + mp)


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def tag_pose(cam_see, cams_meas, x, y, phi, side, tag_h, kappa, plumb, tag_offset=(0.0, 0.0, 0.0)):
    """A virtual car tag (side, top at tag_h, centre (x, y), heading phi) imaged by cam_see and measured through each
    camera of cams_meas by c7_measure.car_pose (same kappa, plumb, offset): [(x, y, psi)] of the measured pose."""
    body = cm.car_corners_body(side)
    P = body @ _rot(phi).T + np.array([x, y])
    pix = cam_see.project(np.column_stack([P, np.full(4, float(tag_h))]))
    out = []
    for c in cams_meas:
        r = cm.car_pose(c, pix, side, kappa, plumb, tag_offset)
        out.append(np.array([r['x'], r['y'], r['psi']]))
    return out


def chuyen_pair(cam_full, cam_held, pa, pb, psi_a, psi_b, side, tag_h, kappa, plumb, tag_offset=(0.0, 0.0, 0.0)):
    """S8-4 (2), S8-5 (ii): difference of the measured tag DIFFERENCE b - a between the held camera and the full
    (scoring) camera, held - full, for the tag imaged by each of the two cameras in turn (KCA1-1). Returns
    [(nhin, dx, dy, dpsi)]: nhin = 'full' (the image made by the full camera) or 'held'."""
    out = []
    for nhin, see in (('full', cam_full), ('held', cam_held)):
        ha, fa = tag_pose(see, (cam_held, cam_full), pa[0], pa[1], psi_a, side, tag_h, kappa, plumb, tag_offset)
        hb, fb = tag_pose(see, (cam_held, cam_full), pb[0], pb[1], psi_b, side, tag_h, kappa, plumb, tag_offset)
        d = (hb - ha) - (fb - fa)
        if not np.all(np.isfinite(d)):                # KCJ1-13: each sample finite before any maximum
            raise ValueError('non-finite virtual tag sample')
        out.append((nhin, float(d[0]), float(d[1]), float(cm.cp_wrap(d[2]))))
    return out


def p1_rel_clip(clip, pc, marks, regions, tag, kappa, plumb, l_max=L_MAX_M, phis=A3_TAG_PHI_DEG):
    """P1 relative of one clip H (S7-3, S7-12.1, S7-14.4). For every region k, mark s of the region and mark j != s:
    camera without s and j (pc((s, j))); d_sj = (c(j) - c(s)) - (tape_j - tape_s) [m, measured - tape, floor frame];
    d_sj_he_dau = R (c(j) - c(s)) - (tape_j - tape_s), R the Procrustes rotation of the training centres to the tape;
    chuyen_sj = held - full of the virtual tag difference j - s (tag at the two mark positions, headings phis, both
    imaging cameras). Returns (rows (one per pair, heading, imaging camera), pair summaries, broken pairs)."""
    rows, pairs, hong = [], [], []
    full = pc()
    side, tag_h = tag
    for k, reg in regions.items():
        for s in reg['moc']:
            for j in pc.ids:
                if j == s:
                    continue
                try:                                        # KS7-7, KCJ1-13: the WHOLE pair, listed when it fails
                    cam = pc((s, j))
                    xs, xj = np.array(marks[s][:2]), np.array(marks[j][:2])
                    cs, cj = centre_floor4(cam, pc.img[s]), centre_floor4(cam, pc.img[j])     # KCJ1-9
                    d = (cj - cs) - (xj - xs)
                    train = [i for i in pc.ids if i not in (s, j)]
                    src = np.array([centre_floor4(cam, pc.img[i]) for i in train])
                    dst = np.array([marks[i][:2] for i in train], float)
                    ang = cm.rigid_fit(src, dst)[0]
                    dh = _rot(ang) @ (cj - cs) - (xj - xs)
                    L = float(np.linalg.norm(xj - xs))
                    ch, rr = [], []
                    for ph in phis:
                        a = math.radians(ph)
                        for nhin, dx, dy, dpsi in chuyen_pair(full, cam, xs, xj, a, a, side, tag_h, kappa, plumb):
                            ch.append((dx, dy, dpsi))
                            rr.append(dict(clip=clip, vung=k, s=s, j=j, L_m=L, d_x=float(d[0]), d_y=float(d[1]),
                                           dh_x=float(dh[0]), dh_y=float(dh[1]), phi_deg=ph, nhin=nhin, chuyen_x=dx,
                                           chuyen_y=dy, chuyen_psi_deg=math.degrees(dpsi)))
                    pr = dict(clip=clip, vung=k, s=s, j=j, L_m=L, trong_vung=j in reg['moc'] and L <= l_max,
                              d_m=float(np.linalg.norm(d)), dh_m=float(np.linalg.norm(dh)),
                              chuyen_m=max(math.hypot(x, y) for x, y, _ in ch),
                              chuyen_psi_deg=max(abs(math.degrees(p)) for _, _, p in ch))
                    if not all(math.isfinite(pr[x]) for x in ('d_m', 'dh_m', 'chuyen_m', 'chuyen_psi_deg')):
                        raise KCError('non-finite result')
                except Exception as e:
                    hong.append(dict(clip=clip, vung=k, s=s, j=j, ly_do=str(e)[:200]))
                    continue
                rows += rr
                pairs.append(pr)
    return rows, pairs, hong


def a3_rel_clip(a3_rows, regions):
    """A3 relative of one clip (S7-4, KS7-3, S8-4 (3)): A3loc(m) = largest quy_deg over the clip's pairs holding m;
    A3_xp(k) = largest A3loc over the region (None when no region mark is in a pair); max_deg = largest over k of
    A3_xp(k) + largest A3loc (both ends in the SAME clip)."""
    loc = {}
    for r in a3_rows:
        for m in (r['i'], r['j']):
            loc[m] = max(loc.get(m, 0.0), r['quy_deg'])
    xp = {}
    for k, reg in regions.items():
        v = [loc[m] for m in reg['moc'] if m in loc]
        xp[k] = max(v) if v else None
    far = max(loc.values()) if loc else None
    tot = [x + far for x in xp.values() if x is not None and far is not None]
    return dict(A3loc=loc, A3_xp=xp, A3loc_max=far, max_deg=max(tot) if tot else None)


def affine_mark(cam, corners_px, side):
    """Local distortion of one mark (S7-5, KS7-4): its 4 image corners to the floor (z = 0), affine fit from the
    model square. Returns dict(gian [s1 - 1, s2 - 1] (singular values), cat_deg (angle of the two mapped axes - 90),
    du_mm (largest non-affine residual of a corner))."""
    fl = cam.pix2floor(np.asarray(corners_px, float), 0.0)
    src = cp.mark_corners_floor(0.0, 0.0, side)
    X = np.column_stack([src, np.ones(4)])
    M = np.linalg.lstsq(X, fl, rcond=None)[0]
    A = M[:2].T
    res = fl - X @ M
    sv = np.linalg.svd(A, compute_uv=False)
    e1, e2 = A[:, 0], A[:, 1]
    cat = math.degrees(math.atan2(e1[0] * e2[1] - e1[1] * e2[0], float(e1 @ e2))) - 90.0
    return dict(gian=[float(sv[0] - 1.0), float(sv[1] - 1.0)], cat_deg=float(cat),
                du_mm=float(np.max(np.linalg.norm(res, axis=1)) * 1000.0))


def meo_clip(pc, marks):
    """{id: affine_mark with the camera without the mark, plus canh_m}."""
    out = {}
    for i in pc.ids:
        r = affine_mark(pc((i,)), pc.img[i], marks[i][2])
        r['canh_m'] = float(marks[i][2])
        out[i] = r
    return out


def meo_summary(by_clip):
    """S7-12.2, S7-14.2: thresholds per mark size = largest |q(H_a) - q(H_b)| of the same mark + print tolerance, for
    the stretch and the non-affine residual; bien_dang_hinh (DIAGNOSTIC only, never a lock condition) when a mark of
    any clip exceeds them. Without H_a and H_b the thresholds and the flag are None."""
    allv = [(c, i, r) for c, d in by_clip.items() for i, r in d.items()]
    out = dict(phan_du_max_mm=max(r['du_mm'] for _, _, r in allv),
               gian_chinh_max=max(max(abs(g) for g in r['gian']) for _, _, r in allv),
               cat_max_deg=max(abs(r['cat_deg']) for _, _, r in allv), chi_chan_doan=True)
    def key(r):                                       # KCJ1-15: one size rule for grouping, thresholds, flag, JSON
        return f"{r['canh_m'] * 1000:.1f}"
    sizes = sorted({key(r) for _, _, r in allv})
    ng, flag, co = {}, None, {}
    for sd in sizes:
        sel = [(c, i, r) for c, i, r in allv if key(r) == sd]
        co[sd] = dict(phan_du_max_mm=max(r['du_mm'] for _, _, r in sel),
                                     gian_chinh_max=max(max(abs(g) for g in r['gian']) for _, _, r in sel),
                                     cat_max_deg=max(abs(r['cat_deg']) for _, _, r in sel))
    if 'H_a' in by_clip and 'H_b' in by_clip:
        flag = False
        for sd in sizes:
            com = [i for i in by_clip['H_a'] if i in by_clip['H_b'] and key(by_clip['H_a'][i]) == sd]
            if not com:
                continue
            lg = max(max(abs(x - y) for x, y in zip(by_clip['H_a'][i]['gian'], by_clip['H_b'][i]['gian'])) for i in com)
            ld = max(abs(by_clip['H_a'][i]['du_mm'] - by_clip['H_b'][i]['du_mm']) for i in com)
            n = dict(gian=lg + DUNG_SAI_IN_GIAN, du_mm=ld + DUNG_SAI_IN_DU_MM)
            ng[sd] = n
            for c, i, r in allv:
                if key(r) == sd and (max(abs(g) for g in r['gian']) > n['gian'] or r['du_mm'] > n['du_mm']):
                    flag = True
    out.update(nguong_lap_HaHb=ng or None, bien_dang_hinh=flag, theo_co=co,
               dung_sai_in=dict(gian=DUNG_SAI_IN_GIAN, du_mm=DUNG_SAI_IN_DU_MM),
               gioi_han='chi phan doi xung cua meo o thang ma; xoay cuc bo khong thay; co tat chi nghia la kiem hinh '
                        'dang chua phat hien bat thuong (KS7-4, S7-14.2); nhieu do goc moc san vao ca ba so')
    return out


def thang_clip(p1, marks):
    """C8BX-03: similarity fit (with scale) of the tape to the held-out (per group) centres of the clip:
    (ti_le - 1, xoay_deg)."""
    ids = sorted(p1)
    src = np.array([marks[i][:2] for i in ids], float)
    dst = src + np.array([p1[i] for i in ids])
    ang, _, sc, _ = cm.rigid_fit(src, dst, with_scale=True)
    return dict(ti_le_tru_1=float(sc - 1.0), xoay_deg=math.degrees(ang),
                chieu='thuoc -> tam giu ngoai (ti_le = camera / thuoc; C8BX-03, KCJ1-16)')


def s7_summary(rel, a3rel, cover, regions, a3_hong=None):
    """Blocks P1_tuong_doi, A3_tuong_doi of kiem_cheo.json (S8-4 (2), (3)) from the per-clip results. KCJ1-13: the
    expected pairs are counted against the computed ones; a broken pair (P1 or A3) touching a region sets its
    cap_hong flag and du_lieu_du false: the maxima are then OBSERVED maxima, not a full-data result."""
    pairs = [p for c in rel.values() for p in c['pairs']]
    if not pairs:
        raise KCError('S7: no pair of a start region')
    worst = max(pairs, key=lambda p: p['d_m'])
    hong_p1 = [h for c in rel.values() for h in c['hong']]
    hong_a3 = [dict(h, clip=c) for c, v in (a3_hong or {}).items() for h in v]

    def mx(v):
        v = list(v)
        return max(v) if v else None
    tv = {}
    for k, reg in regions.items():
        pk = [p for p in pairs if p['vung'] == k]
        pin = [p for p in pk if p['trong_vung']]
        zs = set(reg['moc'])
        hk = [h for h in hong_p1 if h['vung'] == k] + [h for h in hong_a3 if h['i'] in zs or h['j'] in zs]
        tv[k] = dict(max_m=mx(p['d_m'] for p in pk), d_sj_he_dau_max_m=mx(p['dh_m'] for p in pk),
                     chuyen_max_m=mx(p['chuyen_m'] for p in pk), cap_hong=bool(hk),
                     chuyen_psi_max_deg=mx(p['chuyen_psi_deg'] for p in pk),
                     trong_vung_max_m=max([max(p['d_m'], p['dh_m']) for p in pin], default=None),
                     trong_vung_n_cap=len(pin), moc_vung=reg['moc'], n_cap=len(pk), vung_mo_rong=reg['vung_mo_rong'],
                     ngoai_suy=any(cover[c][k]['ngoai_suy'] for c in cover),
                     mot_phuong=any(cover[c][k]['mot_phuong'] for c in cover),
                     ly_do={c: cover[c][k]['ly_do'] for c in cover if cover[c][k]['ly_do']})
    P = dict(max_m=max(p['d_m'] for p in pairs), d_sj_he_dau_max_m=max(p['dh_m'] for p in pairs),
             chuyen_max_m=max(p['chuyen_m'] for p in pairs),
             chuyen_psi_max_deg=max(p['chuyen_psi_deg'] for p in pairs), theo_vung=tv,
             cap_lon_nhat=dict(clip=worst['clip'], vung=worst['vung'], s=worst['s'], j=worst['j'], d_m=worst['d_m']),
             cap_hong=hong_p1, cap_hong_A3=hong_a3,
             n_cap={c: dict(du_kien=v['n_du_kien'], tinh=len(v['pairs'])) for c, v in rel.items()},
             du_lieu_du=not hong_p1 and not hong_a3 and all(len(v['pairs']) == v['n_du_kien'] for v in rel.values()),
             he_truc=('d_sj: he san; d_sj_he_dau: sau khi bo xoay chung uoc tu ID huan luyen; chuyen_sj: hieu ma ao '
                      'j - s do qua car_pose, he san'),
             chieu_tru='d_sj = do - thuoc; chuyen_sj = held - full (camera bo cap tru camera cham)',
             loai='chi so thuc nghiem truoc le (Liam 19/11 "Uoc luong thuc nghiem"), khong phai can',
             gioi_han=['(a) thuoc cua s, j vao d_sj o ca hai dau', '(b) chi lay mau tai moc',
                       '(c) sai chung cua moi goc anh hieu chuan khong lo ra', '(d) camera clip H, khong phai tung khung',
                       '(e) A3 tuong doi la tong hai chi so thuc nghiem tung dau (truoc le), khong phai can',
                       '(f) diem xuat phat danh dinh, mien bat dinh MIEN_XP_M',
                       '(g) truong sai giua mau, o do cao ma, ngoai ho mo hinh khong bi chan; khong tuyen bo '
                       'd_sj + chuyen_sj la sai ma noc so chuan'])
    ax = {}
    for k in regions:
        v = [a3rel[c]['A3_xp'][k] for c in a3rel if a3rel[c]['A3_xp'][k] is not None]
        ax[k] = max(v) if v else None
    loc = {}
    for c in a3rel:
        for m, v in a3rel[c]['A3loc'].items():
            loc[m] = max(loc.get(m, 0.0), v)
    tot = [a3rel[c]['max_deg'] for c in a3rel if a3rel[c]['max_deg'] is not None]
    A = dict(max_deg=max(tot) if tot else None, A3_xp_deg=ax, A3loc_deg={str(m): v for m, v in sorted(loc.items())},
             A3loc_max_deg=max(loc.values()) if loc else None,
             mot_phuong={k: tv[k]['mot_phuong'] or ax[k] is None for k in regions},
             ngoai_suy={k: tv[k]['ngoai_suy'] for k in regions},
             cap_hong={k: tv[k]['cap_hong'] for k in regions}, du_lieu_du=P['du_lieu_du'],
             theo_clip={c: dict(max_deg=a3rel[c]['max_deg'], A3_xp_deg=a3rel[c]['A3_xp']) for c in a3rel},
             loai='chi so kiem cheo gop hai dau (KS7-3), trong tung clip; truoc le')
    return P, A


def duong_summary(dregs, dcover, a3rel, a3_hong=None):
    """Block A3_tuong_doi.duong_12b (HCV3-2, CODE 20/11): A3 by position over the 12b corridor of each place (segment
    doan_m widened by ban_kinh_m), keyed like A3_tuong_doi: A3_deg[k] = largest A3loc of the corridor marks over the
    clips (None when no corridor mark is in a pair: mot_phuong), with the flags ngoai_suy (capsule outside the hull
    of the corridor marks or farther than R_TRONG_M from them), mot_phuong, cap_hong (a broken A3 pair touching a
    corridor mark). Only A3: no P1 pair is computed for the corridor."""
    hong = [dict(h, clip=c) for c, v in (a3_hong or {}).items() for h in v]
    A, mp, ns, ch, ly, gap, moc, doan, wide, tc = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for k, r in dregs.items():
        per = {}
        for c in a3rel:
            v = [a3rel[c]['A3loc'][m] for m in r['moc'] if m in a3rel[c]['A3loc']]
            per[c] = max(v) if v else None
        vals = [v for v in per.values() if v is not None]
        A[k] = max(vals) if vals else None
        ns[k] = any(dcover[c][k]['ngoai_suy'] for c in dcover)
        mp[k] = any(dcover[c][k]['mot_phuong'] for c in dcover) or A[k] is None
        zs = set(r['moc'])
        ch[k] = any(h['i'] in zs or h['j'] in zs for h in hong)
        ly[k] = {c: dcover[c][k]['ly_do'] for c in dcover if dcover[c][k]['ly_do']}
        gap[k] = max(dcover[c][k]['khoang_trong_max_m'] for c in dcover)
        moc[k], doan[k], wide[k], tc[k] = r['moc'], r['doan_m'], r['vung_mo_rong'], per
    return dict(A3_deg=A, mot_phuong=mp, ngoai_suy=ns, cap_hong=ch, ly_do=ly, khoang_trong_max_m=gap, moc=moc,
                doan_m=doan, ban_kinh_m=MIEN_XP_M, vung_mo_rong=wide, theo_clip=tc, R_TRONG_M=R_TRONG_M,
                R_XP_DUONG_M=R_XP_DUONG_M,
                nguon_doan='rpi5/c8_hc_seq.py PLACES (hil 8c2aa2e): diem dung theo truc xe tu diem xuat phat, huong mui',
                loai='chi so kiem cheo A3 theo vi tri tren hanh lang 12b (doan 12b noi rong ban_kinh_m), trong tung '
                     'clip roi lay lon nhat; truoc le; chi A3, khong co P1 cho hanh lang',
                gioi_han=['(a) A3 lay tai moc, khong tai diem cua doan; moc cach diem xa nhat toi khoang_trong_max_m',
                          '(b) doan 12b theo lich Pi danh dinh; xe lech khoi doan hon ban_kinh_m thi ra ngoai mien',
                          '(c) tam ma noc lay bang diem xuat phat (lech tam ma bo qua, nhu dia xuat phat)'])


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
    check_cam_v3(out)
    return out


def _hex64(x, what):
    if not isinstance(x, str) or len(x) != 64 or any(ch not in '0123456789abcdef' for ch in x):
        raise KCError(f'{what}: {x!r} is not a SHA-256 (64 lowercase hex)')


def _same_num(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(map(_same_num, a, b))
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)


def check_cam_v3(out):
    """KCJ1-4: the writer refuses what the pinned reader (c8b_u v3 198d2e9 check_cam) refuses: top-level keys, the
    numbers, A3 carried to the tag (quy_ve_ma_noc true), H > h, H - dzc > h, kappa = h / H, the moc_dich block
    (approval and premise strings, nguon: frames {run: sha} non-empty, lich, tap_moc, u_so, session sha, bo_lay_mau,
    car_pose exactly {tag_offset, side_m, h_m} equal to bo_cham / do_cao)."""
    if out['A3'].get('cap_hong') or out['A3'].get('du_lieu_du') is False:      # KCJ2-1: A3 over a partial pair set
        raise KCError(f"A3: broken pairs {out['A3'].get('cap_hong')}: the maximum is not over the planned pairs")
    extra = sorted(set(out) - set(CAM_BLOCKS) - {'version', 'nguon', 'ghi_chu'})
    if out.get('version') != CAM_U_VERSION or extra:
        raise KCError(f'camera file: version {out.get("version")!r} / unknown top-level keys {extra}')
    if out['A3'].get('quy_ve_ma_noc') is not True:
        raise KCError('A3.quy_ve_ma_noc must be true for c8b_u v3 (A3 carried to the car tag and approved)')
    H, h, dzc, kap = out['do_cao']['H_m'], out['do_cao']['h_m'], out['day_doi']['dzc_m'], out['day_doi']['kappa']
    if not H > h or not H - dzc > h:
        raise KCError(f'do_cao / day_doi: H {H} m, dzc {dzc} m, h {h} m (H - dzc must be above h)')
    if abs(kap - h / H) > 1e-6 * (h / H):
        raise KCError(f'day_doi.kappa {kap} is not do_cao.h_m / do_cao.H_m = {h / H}')
    md = out['moc_dich']
    for k in ('phuong_phap_duyet', 'tien_de_buoi'):
        if not isinstance(md.get(k), str):
            raise KCError(f'moc_dich.{k} missing or not a string')
    ng = md.get('nguon')
    if not isinstance(ng, dict) or set(MOC_NGUON_KEYS) - set(ng):
        raise KCError('moc_dich.nguon incomplete')
    if not isinstance(ng['frames'], dict) or not ng['frames']:
        raise KCError('moc_dich.nguon.frames: {RUN_ID: sha256} of every run needed')
    for rid, x in ng['frames'].items():
        _hex64(x, f'moc_dich.nguon.frames.{rid}')
    for k in ('lich', 'tap_moc', 'u_so', 'session'):
        _hex64(ng[k], f'moc_dich.nguon.{k}')
    if not isinstance(ng['bo_lay_mau'], str) or not ng['bo_lay_mau'].strip():
        raise KCError('moc_dich.nguon.bo_lay_mau empty')
    cp_ = ng['car_pose']
    if not isinstance(cp_, dict) or set(cp_) != {'tag_offset', 'side_m', 'h_m'}:
        raise KCError('moc_dich.nguon.car_pose needs exactly tag_offset, side_m, h_m')
    for k, w in (('tag_offset', out['bo_cham']['tag_offset']), ('side_m', out['bo_cham']['side_m']), ('h_m', h)):
        if not _same_num(cp_[k], w):
            raise KCError(f'moc_dich.nguon.car_pose.{k} {cp_[k]!r} is not the one of this camera file {w!r}')


def publish_new(path, write):
    """KCJ1-6: write(f) fills a temporary file in the same directory; the result is published under path only when
    complete, never over an existing file (rename on Windows refuses an existing target; link elsewhere); on any error
    the temporary file is removed and nothing new stays."""
    if os.path.exists(path):
        raise KCError(f'{path} exists: not replaced')
    tmp = f'{path}.tmp{os.getpid()}'
    try:
        with open(tmp, 'x', newline='', encoding='utf-8') as f:
            write(f)
        if os.name == 'nt':
            os.rename(tmp, path)
        else:
            os.link(tmp, path)
            os.unlink(tmp)
    except FileExistsError:
        raise KCError(f'{path} exists: not replaced')
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_new(path, obj):
    publish_new(path, lambda f: json.dump(obj, f, indent=1, ensure_ascii=False))


def load_json_sha(path):
    """(object, sha256) from ONE read of the bytes (KCJ1-6: the sha is that of the bytes used)."""
    with open(path, 'rb') as f:
        b = f.read()
    return json.loads(b.decode('utf-8')), hashlib.sha256(b).hexdigest()


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
    Returns dict(max_m (None without rows), n_valid, ti_le_thieu, khoang_mat_max_s, thieu, ly_do, thieu_du_lieu_neo).
    TD2-1: frames.csv without the anchor columns (moc_thay, moc_neo_<chain>) is not refused: the run is marked
    thieu_du_lieu_neo (the anchor sets it used, and its contradictions, cannot be known; tien_de: the session lacks
    data). Without idx, t_s, valid the file is not a frames.csv: refused."""
    G = len(groups)
    comp = comparators(groups)
    base = {'idx', 't_s', 'valid'}
    if not frames or not base <= set(frames[0]):
        raise KCError(f'frames.csv: missing columns {sorted(base - set(frames[0] if frames else ()))}')
    need = {'moc_thay'} | {f'moc_neo_{c}' for c in comp}
    if not need <= set(frames[0]) or any(r.get(k) is None for r in frames for k in need):   # KCJ1-5: short rows
        return dict(max_m=None, n_valid=sum(1 for r in frames if _true(r['valid'])), ti_le_thieu=1.0,
                    khoang_mat_max_s=None, thieu=True, thieu_du_lieu_neo=True, tap_chua_tinh=[],
                    ly_do=[f'frames.csv thieu cot neo {sorted(need - set(frames[0]))} (TD2-1)'])
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
                thieu_du_lieu_neo=False, ly_do=ly_do,
                tap_chua_tinh=[dict(chuoi=c, nhom_dich=t, tap=list(s)) for c, t, s in sorted(lost_sets)][:50])


def run_contradictions(frames, track, marks_over):
    """TD1-2: marks of `marks_over` (own shift at H_m / H_z above the chain-xe threshold) that THIS run used as a
    chain-xe anchor in a frame not preceded by the run's own chain-xe exclusion of the mark (t_s < t_bd, or never
    excluded). The time of the shift is unknown, so every such use counts (also before H_m). Sorted ids."""
    t_bd = {}
    for m in track.get('moc_loai') or []:
        if str(m.get('chuoi')) == 'xe':
            j = int(m['id'])
            t_bd[j] = min(t_bd.get(j, math.inf), float(m['t_bd']))
    used = set()
    for r in frames:                                  # KCJ1-5: rows without a usable chain-xe anchor cell are skipped
        try:                                          #   here (the run is thieu_du_lieu_neo anyway)
            t, A = float(r['t_s']), _ids(r['moc_neo_xe'])
        except (KeyError, ValueError, TypeError):
            continue
        for i in A:
            if i in marks_over and t < t_bd.get(i, math.inf):
                used.add(i)
    return sorted(used)


def read_plan(path):
    """(events, runs) of the session plan (TD2-2): one id per line (or the first CSV column), '#' lines and a header
    word ignored (KCC1-11). events = every line in order, clips H included (H_a, H_b, H_m, H_z); runs = the lines
    that are not clip H names. Empty, repeated ids, or no run: refused."""
    return read_plan_sha(path)[:2]


def read_plan_sha(path):
    """read_plan plus the sha256 of the very bytes parsed (KCJ1-2, KCJ1-6: no second read between parse and hash)."""
    with open(path, 'rb') as f:
        b = f.read()
    ev = []
    for line in b.decode('utf-8').splitlines():
        s = line.strip().split(',')[0].strip()
        if s and not s.startswith('#') and s.lower() not in ('luot', 'slot', 'run'):
            ev.append(s)
    runs = [s for s in ev if s not in CLIPS_REQUIRED]
    if not runs or len(ev) != len(set(ev)):
        raise KCError(f'{path}: empty, repeated ids or no run')
    return ev, runs, hashlib.sha256(b).hexdigest()


def plan_order_problems(ev, roles=CLIPS_REQUIRED):
    """TD1-3, TD2-2: the declared order must be H_a, H_b first and in a row, H_m with at least one run before and
    after it, H_z last. Returns the list of problems (empty = right shape); the real filming order is not checked
    (limit (e)). roles: the clips H of the result (vai_tro: a camera segment has no H_m); a clip H of another role in
    the plan is a problem too."""
    p = [f'thieu {n}' for n in roles if n not in ev]
    p += [f'{n} khong thuoc doan' for n in CLIPS_REQUIRED if n in ev and n not in roles]
    if p:
        return p
    if ev[:2] != ['H_a', 'H_b']:
        p.append('H_a, H_b khong lien nhau o dau')
    if 'H_z' in roles and ev[-1] != 'H_z':
        p.append('H_z khong o cuoi')
    if 'H_m' in roles:
        k = ev.index('H_m')
        if not any(s not in CLIPS_REQUIRED for s in ev[:k]) or not any(s not in CLIPS_REQUIRED for s in ev[k + 1:]):
            p.append('H_m khong o giua cac luot')
    return p


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
    ev, plan, plan_sha = read_plan_sha(a.ke_hoach)
    # TD1-2 (KCC1-12 per run): marks whose own shift at H_m / H_z is above the chain-xe threshold (KCC2-1: H_b not)
    ng = chains['xe']['nguong_m']
    over = {}
    for n, tc in (kc.get('theo_thoi_gian') or {}).items():
        if n in ('H_m', 'H_z'):
            for i, r in (tc.get('rieng_m') or {}).items():
                if r > ng:
                    over.setdefault(int(i), []).append(n)
    luot, e_all, mau, dirs, frames_sha = {}, None, [], set(), {}
    for spec in a.luot:
        rid, _, d = spec.partition('=')
        if not d or rid in luot:
            raise KCError(f'--luot {spec!r}: RUN=dir, each run once')
        if rid not in plan:
            raise KCError(f'{rid}: not in the plan {a.ke_hoach}')
        dn = os.path.normcase(os.path.abspath(d))
        if dn in dirs:
            raise KCError(f'--luot {rid}: directory {d} given for two runs (TD1-6)')
        dirs.add(dn)
        tr = _load_json(os.path.join(d, 'track.json'))
        fp, kp = os.path.join(d, 'frames.csv'), os.path.join(d, 'kiem_cheo_khung.csv')
        fs = sha256_file(fp)
        if fs in frames_sha:
            raise KCError(f'--luot {rid}: same frames.csv as {frames_sha[fs]} (TD1-6)')
        frames_sha[fs] = rid
        for k, want in (('u_so_sha256', sha_u), ('session_sha256', sha_ha), ('model_khoa', kc['model']),
                        ('nhom', groups), ('hang_so_sha256', kc.get('hang_so_sha256')), ('nguong_dung', want_ng),
                        ('frames_sha256', sha256_file(fp)), ('kiem_cheo_khung_sha256', sha256_file(kp))):
            if tr.get(k) != want:
                raise KCError(f'{rid}: track {k} {tr.get(k)!r} is not {want!r} (KC1-10, KCC1-8, KCC1-9)')
        if tr.get('moc_loai') and tr.get('cham_lai') != 2:
            raise KCError(f'{rid}: marks excluded but the track was not rescored (cham_lai 2)')
        frows = _read_csv(fp)
        r = run_frame_check(frows, _read_csv(kp), tr, groups, chains)
        tp = os.path.join(d, 'track.json')
        r['nguon'] = dict(track_sha256=sha256_file(tp), frames_sha256=tr['frames_sha256'],
                          kiem_cheo_khung_sha256=tr['kiem_cheo_khung_sha256'])
        r['duong_dan'] = dict(track=os.path.abspath(tp), frames=os.path.abspath(fp),       # TD3-2
                              kiem_cheo_khung=os.path.abspath(kp))
        if frows and 'moc_neo_xe' in frows[0]:      # KCJ1-5: known contradictions kept even when other anchors lack
            for i in run_contradictions(frows, tr, set(over)):
                mau += [dict(luot=rid, clip=n, moc=i) for n in over[i]]
        luot[rid] = r
        if r['max_m'] is not None:
            e_all = r['max_m'] if e_all is None else max(e_all, r['max_m'])
    thieu_luot = sorted(set(plan) - set(luot))
    mau.sort(key=lambda x: (x['luot'], x['clip'], x['moc']))
    thieu_neo = sorted(k for k, v in luot.items() if v['thieu_du_lieu_neo'])
    out = dict(version=TUNG_KHUNG_VERSION, buoi=a.buoi, model=kc['model'], nhom=groups, moc=kc.get('moc'),
               hang_so_sha256=kc.get('hang_so_sha256'), kiem_cheo_sha256=sha256_file(a.kiem_cheo), u_so_sha256=sha_u,
               session_sha256=sha_ha, cham_lai=True,
               ke_hoach=dict(file=os.path.abspath(a.ke_hoach), sha256=plan_sha, luot=plan, su_kien=ev),
               tung_khung_max_m=e_all, luot=luot, luot_thieu=sorted(r for r, v in luot.items() if v['thieu']),
               luot_chua_co=thieu_luot, luot_thieu_neo=thieu_neo, mau_thuan=mau,
               moc_vuot_sau=[dict(moc=i, clip=c) for i in sorted(over) for c in over[i]],
               hoan_tat=not thieu_luot and not mau and not thieu_neo and e_all is not None,
               c33_12=False,                      # design v6 S8-3: refusal of trial tracks etc. after the CAM merge
               luat=dict(max_thieu=COVER_MAX_MISS, max_gap_s=COVER_MAX_GAP_S))
    write_new(a.out, out)
    _say(f"tung_khung: {len(luot)}/{len(plan)} luot, max "
          f"{'-' if e_all is None else f'{e_all * 1000:.2f} mm'}; thieu kiem: {out['luot_thieu']}; chua co: "
          f"{thieu_luot}; thieu neo: {thieu_neo}; mau thuan: {mau}; hoan tat: {out['hoan_tat']}")


# ---------------------------------------------------------------- F15 (a) meo_ma_ao_E, (b) meo_dau_chan_bang
# Contract F15 (CODE 19/11, C8B confirmed the names; KIEMCHEO sets the fields): map residual field held - full of the
# H32 floor marks, carried to points in the car_pose lifted convention (p = pix2floor(uv, 0), P = p + kappa (C - p)).
# KCF1-1 (Codex F15 round 1): the measured points come through the camera REFITTED per frame. Full version (CODE
# 20/11 after the CAM merge 3625211, Liam at CODE "Giao ngay"): per-frame cameras from the dets.npz of CODE_CAM
# (c7_measure.load_dets), the scoring camera as the scoring makes it, the held camera of an H32 case refitted on the
# same frame from the clip-H camera without the case, on the same anchors without the case. The diagnostic versions
# (pixels through the session camera, schema ..._chan_doan) are gone.
MEO_E_SCHEMA = 'c7_meo_ma_ao_E_v1'
MEO_BANG_SCHEMA = 'c7_meo_dau_chan_bang_v1'
MEO_TRANG_THAI = 'dat'
TOL_REPLAY_M = 1e-6                 # replay of the scoring pose vs frames.csv tag_x / tag_y (10 decimals written)


def _ids_of(cell):
    """IDs of a c7_measure _ids_cell ('1;2;3'; '' = none)."""
    s = '' if cell is None else str(cell).strip()
    return [] if not s else sorted(int(v) for v in s.split(';'))


def load_dets_checked(path, want_sha=None, want_fp=None, video_sha=None, what='dets.npz'):
    """{idx: (t_s, {id: (4, 2) px})} of a CODE_CAM dets.npz (c7_measure.load_dets) with its file sha256, checked
    against the sha / frames_fingerprint / video sha the sources name (any mismatch: KCError, code 2)."""
    if not path:
        raise KCError(f'{what}: no file (per-frame cameras need the detections of CODE_CAM)')
    sha = sha256_file(path)
    if want_sha is not None and sha != want_sha:
        raise KCError(f'{what} {path}: sha256 {sha} is not the one its source names {want_sha}')
    frames, vsha, _ = cm.load_dets(path)
    if want_fp is not None and cm.frames_fingerprint(frames) != want_fp:
        raise KCError(f'{what} {path}: frames_fingerprint is not the dets_fp its source names')
    if video_sha is not None and vsha != video_sha:
        raise KCError(f'{what} {path}: detections of another clip (video_sha256 {vsha})')
    by = {}
    for idx, t, d in frames:
        if idx in by:
            raise KCError(f'{what} {path}: frame {idx} twice')
        by[idx] = (t, d)
    return by, sha, vsha


class FramePoses:
    """Per-frame cameras of one session (c7_measure 33-mark chain, design v3 2): the scoring camera = pose refitted
    from the clip-H camera cam_H[xe] on the anchors (ChainCtx.pose, trust flags); the held camera of an H32 case rm =
    pose refitted on the same frame from the clip-H camera fitted WITHOUT rm (PairCams of the session, the same
    fit_two_pass) on the same anchors without rm (same trust flags). A refit that is not trusted: KCError (KCF1-5)."""

    def __init__(self, session, pc):
        self.ctx = cm.ChainCtx(session)
        self.pc = pc
        self.marks = self.ctx.marks

    def full(self, anchors, d, what=''):
        anchors = sorted(anchors)
        if not set(anchors) <= set(d):
            raise KCError(f'{what}: anchors {sorted(set(anchors) - set(d))} not detected in the frame')
        cam, rms, ok, ly, cond = self.ctx.pose(cm.CHAIN_XE, anchors, d)
        if not ok:
            raise KCError(f'{what}: scoring pose refit not trusted ({ly}, cond {cond:.3g})')
        return cam

    def held(self, rm, anchors, d, what=''):
        use = [i for i in sorted(anchors) if i not in set(rm)]
        if len(use) < MIN_TRAIN:
            raise KCError(f'{what}: case {_ca_ten(rm)} leaves {len(use)} anchors (< {MIN_TRAIN}) (KCF1-5)')
        obj = np.vstack([cp.mark_corners_floor(*self.marks[i]) for i in use])
        cam, rms, ok, ly, cond = cm.refit_pose_checked(self.pc(rm), obj, np.vstack([d[i] for i in use]), None,
                                                       self.ctx.cond_max)
        if not ok:
            raise KCError(f'{what}: held pose refit of case {_ca_ten(rm)} not trusted ({ly}, cond {cond:.3g}) (KCF1-5)')
        return cam


def lift(p, kappa, C):
    return np.asarray(p, float) + kappa * (np.asarray(C, float) - np.asarray(p, float))


def unlift(P, kappa, C):
    return (np.asarray(P, float) - kappa * np.asarray(C, float)) / (1.0 - kappa)


def map_residual(cam_full, cam_held, P, kappa, C):
    """Lifted points P (n, 2) of the scoring convention: the pixel the scoring (full) camera gives them, measured back
    through the held camera, lifted the same way: P_held - P [m] (held - full)."""
    P = np.asarray(P, float).reshape(-1, 2)
    uv = cam_full.project(np.column_stack([unlift(P, kappa, C), np.zeros(len(P))]))
    d = lift(cam_held.pix2floor(uv, 0.0), kappa, C) - P
    if not np.all(np.isfinite(d)):
        raise KCError('non-finite map residual')
    return d


def h32_ids(marks, foot, r_xp):
    """KCF1-4: the H32 hold-out set of a footprint: every mark within r_xp of ANY footprint point (floor points, not
    lifted), padded to at least the 3 marks nearest to the footprint."""
    foot = np.asarray(foot, float).reshape(-1, 2)
    dist = {i: float(np.min(np.linalg.norm(foot - np.array(m[:2]), axis=1))) for i, m in marks.items()}
    ids = {i for i, d in dist.items() if d <= r_xp}
    for i in sorted(dist, key=lambda i: (dist[i], i)):
        if len(ids) >= 3:
            break
        ids.add(i)
    return sorted(ids)


def _ca_ten(ids):
    """KCF1-3: a hold-out case is named by its left-out id set ('bo:18-19-30'), the same name meaning the same camera
    in every place and file."""
    return 'bo:' + '-'.join(str(i) for i in sorted(ids))


def held_cases(pc, ids):
    """Hold-out cases: 'bo_vung' (every id of the set) and 'bo_cap:s-j' for each pair of the set. KCF1-5: every
    planned case must fit, else KCError (no partial maximum). Returns [(name, ids left out, camera)]."""
    plan = [(_ca_ten(ids), tuple(ids))] + [(_ca_ten((a, b)), (a, b)) for k, a in enumerate(ids) for b in ids[k + 1:]]
    out = []
    for name, rm in plan:
        try:
            out.append((name, sorted(rm), pc(rm)))
        except Exception as e:
            raise KCError(f'H32 case {name} (left out {sorted(rm)}) failed: {str(e)[:200]} (KCF1-5)')
    return out


def a3_held_case(name, img, marks, cam_full, cam_ca, rm, tag, l_max=L_MAX_M):
    """A3 of one H32 hold-out case (CODE 20/11, F15 a): every pair (i < j) of the left-out set rm with tape distance
    <= l_max, measured on the session image through the camera of THE CASE (fitted without all of rm, not only without
    the pair): dtheta_deg = pair angle through cam_ca minus the tape angle, gap_deg = a3_tag_gap(cam_full, cam_ca)
    (both directions), quy_deg = |dtheta| + gap (as a3_clip). cam_full = the scoring camera (session.camera, the one
    of du_m). A non-finite sample or a failed conversion fails the case: KCError (KCF1-5, code 2)."""
    rows = []
    rm = sorted(rm)
    for a in range(len(rm)):
        for b in range(a + 1, len(rm)):
            i, j = rm[a], rm[b]
            vt = np.array(marks[j][:2], float) - np.array(marks[i][:2], float)
            L = float(np.linalg.norm(vt))
            if L > l_max:
                continue
            try:
                vc = centre_floor(cam_ca, img[j]) - centre_floor(cam_ca, img[i])
                g, dpsi, dp = a3_tag_gap(cam_full, cam_ca, marks[i], marks[j], *tag)
                r = dict(i=i, j=j, L_m=L, dtheta_deg=signed_angle_deg(vt, vc), gap_deg=g, dpsi_ma_max_deg=dpsi,
                         dtheta_hai_cam_deg=dp)
                r['quy_deg'] = abs(r['dtheta_deg']) + g
                if not all(math.isfinite(v) for v in r.values()):
                    raise ValueError('non-finite result')
            except Exception as e:
                raise KCError(f'H32 case {name}: A3 of pair {i}-{j} failed: {str(e)[:200]} (KCF1-5)')
            rows.append(r)
    out = dict(cam_full='session.camera', cam_ca=name, L_max_m=float(l_max), n_cap=len(rows), cap=rows,
               quy_max_deg=max((r['quy_deg'] for r in rows), default=None))
    if not rows:
        out['ly_do'] = 'khong co cap moc bo nao trong L_max'
    return out


def _hex(x, what):
    if not isinstance(x, str) or len(x) != 64 or any(ch not in '0123456789abcdef' for ch in x):
        raise KCError(f'{what}: {x!r} is not a sha256 (KCF1-7)')
    return x


def _session_ctx(path, want_sha, model):
    """(session dict, sha, marks, PairCams, scoring camera, kappa, C) from the bytes of the session file; the sha
    must be want_sha; KCF1-4: the 32 locked floor marks (ids 1-32) only."""
    s, sha = load_json_sha(path)
    if sha != _hex(want_sha, 'session sha256 named by the source'):
        raise KCError(f'{path}: sha256 {sha} is not the one the source names {want_sha}')
    img, mk, w, h, zc = clip_from_session(s, model)
    if tuple(sorted(img)) != ID_KHOA:
        raise KCError(f'{path}: fit ids are not the locked ids 1-32 (KCF1-4)')
    for k in ('camera', 'kappa', 'plumb_xy_m', 'tag_h_m', 'car_side_m'):
        if k not in s:
            raise KCError(f'{path}: session without {k} (car_pose convention)')
    nums = [s['kappa'], s['tag_h_m'], s['car_side_m']] + list(s['plumb_xy_m'] or [])
    if len(s['plumb_xy_m'] or []) != 2 or not all(_fin_ge0(abs(v)) if isinstance(v, (int, float)) and
                                                  not isinstance(v, bool) else False for v in nums) \
            or not 0 < s['kappa'] < 1 or not s['tag_h_m'] > 0 or not s['car_side_m'] > 0:
        raise KCError(f'{path}: kappa / tag_h_m / car_side_m / plumb_xy_m not finite or out of range (KCF1-6)')
    return s, sha, mk, PairCams(img, mk, model, w, h, zc), cm.cam_from_dict(s['camera']), float(s['kappa']), \
        np.asarray(s['plumb_xy_m'], float)


def _with_sha(obj):
    obj['sha256'] = hashlib.sha256(json.dumps({k: v for k, v in obj.items() if k != 'sha256'}, ensure_ascii=False,
                                              sort_keys=True).encode()).hexdigest()
    return obj


def _ids_sha_forms(ids):
    """sha256 of a frame-ID list in the two forms the F8 writers used (c7_nhac_tam sha_ids: '1,2' since 1cfe26e,
    compact JSON '[1,2]' at f9c4b3d)."""
    v = sorted(int(i) for i in ids)
    return {hashlib.sha256(','.join(str(i) for i in v).encode()).hexdigest(),
            hashlib.sha256(json.dumps(v, separators=(',', ':')).encode()).hexdigest()}


def _check_h_bar(path, h_used, C=None, Zc=None):
    """KCF1-6: the tag height of the session against h-bar of the plate (c8b_a6_v2 field h_m), when given."""
    if not path:
        return dict(da_kiem=False, ly_do='khong co --a6: h_m cua session chua doi chieu voi h-bar cua tam')
    d, sha = load_json_sha(path)
    if d.get('schema') != 'c8b_a6_v2':
        raise KCError(f'--a6: schema {d.get("schema")!r} is not c8b_a6_v2 (KCF1-6)')
    hb = d.get('h_m')
    if isinstance(hb, bool) or not isinstance(hb, (int, float)) or not math.isfinite(hb) or abs(float(hb) - h_used) > 1e-6:
        raise KCError(f'--a6: h_m {hb!r} is not the session tag height {h_used} (KCF1-6)')
    pl, H = d.get('plumb_xy_m'), d.get('H_m')
    if not isinstance(pl, list) or len(pl) != 2 or not isinstance(H, (int, float)) or isinstance(H, bool) or \
            any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in pl) or \
            (C is not None and max(abs(float(pl[0]) - C[0]), abs(float(pl[1]) - C[1])) > 1e-9) or \
            (Zc is not None and abs(float(H) - Zc) > 1e-9):
        raise KCError('--a6: plumb_xy_m / H_m are not the car_pose set-up of the session (KCF1-6)')
    return dict(da_kiem=True, file=os.path.abspath(path), sha256=sha, h_m=float(hb))


def meo_E(a):
    """F15 (a): for the E frames of one place (one E_dau and one E_cuoi window of the clip in cua_so_v1; the moving
    frames strictly between them = E_doan, valid with a tag pose), the 4 virtual tag corners (ArUco order,
    c7_measure.car_corners_body) of every E_doan frame, map residual held - full per H32 case through the PER-FRAME
    cameras of the dets.npz (FramePoses, full version CODE 20/11). Each case also carries its A3 (a3_held_case) on the
    session of the E clip."""
    if a.cho not in ('goc', 'K1'):
        raise KCError(f'--cho {a.cho!r}: goc|K1 (E places, KCF1-8)')
    l_max = L_MAX_M if getattr(a, 'L_max', None) is None else a.L_max
    if not (math.isfinite(l_max) and l_max > 0):
        raise KCError(f'--L-max {l_max!r}: a finite length > 0')
    tp, fp = os.path.join(a.track, 'track.json'), os.path.join(a.track, 'frames.csv')
    tr, tr_sha = load_json_sha(tp)
    with open(fp, 'rb') as f:
        fb = f.read()
    fr_sha = hashlib.sha256(fb).hexdigest()
    clip = _hex(tr.get('video_sha256'), 'track.video_sha256')
    if _hex(tr.get('frames_sha256'), 'track.frames_sha256') != fr_sha:
        raise KCError('E track: frames.csv is not the one of track.json')
    s, ses_sha, mk, pc, full, kappa, C = _session_ctx(a.session, tr.get('session_sha256'), a.model)
    with open(a.cua_so, 'rb') as f:
        cb = f.read()
    cs_sha = hashlib.sha256(cb).hexdigest()                     # KCF1-7: hash of the bytes parsed
    try:
        rows = [json.loads(x) for x in cb.decode('utf-8').splitlines() if x.strip()]
    except ValueError as e:
        raise KCError(f'cua_so_v1: {e}')
    ew = [r for r in rows if isinstance(r, dict) and r.get('cho') == a.cho and r.get('loai') in ('E_dau', 'E_cuoi')
          and r.get('clip_sha256') == clip]
    by = {k: [r for r in ew if r['loai'] == k] for k in ('E_dau', 'E_cuoi')}
    if len(by['E_dau']) != 1 or len(by['E_cuoi']) != 1:
        raise KCError(f'cua_so_v1: need exactly one E_dau and one E_cuoi window of {a.cho} for this clip (KCF1-8)')
    lid = by['E_dau'][0].get('luot_id')
    if not isinstance(lid, str) or not lid or by['E_cuoi'][0].get('luot_id') != lid:
        raise KCError('cua_so_v1: E_dau and E_cuoi without the same run id (KCF1-8)')
    import io
    try:
        t_of = {int(r['idx']): float(r['t_s']) for r in csv.DictReader(io.StringIO(fb.decode('utf-8')))}
    except (KeyError, ValueError, TypeError) as e:
        raise KCError(f'frames.csv: {e} (KCF1-8)')
    ids = {}
    for k, (r,) in by.items():
        if r.get('frames_sha256') != fr_sha or r.get('session_sha256') != ses_sha:
            raise KCError(f'cua_so_v1 {k}: another frames / session (KCF1-7)')
        kh = r.get('khung')
        if not isinstance(kh, list) or not kh or not all(
                isinstance(x, dict) and isinstance(x.get('id'), int) and not isinstance(x.get('id'), bool)
                and x['id'] in t_of and isinstance(x.get('pts_s'), (int, float))
                and abs(float(x['pts_s']) - t_of[x['id']]) <= 1e-6 for x in kh):
            raise KCError(f'cua_so_v1 {k}: khung missing, malformed or not the frames of the track (KCF1-8)')
        ids[k] = [x['id'] for x in kh]
        if any(b <= a_ for a_, b in zip(ids[k], ids[k][1:])):
            raise KCError(f'cua_so_v1 {k}: frame ids repeated or out of order (KCF1-8)')
    lo, hi = max(ids['E_dau']), min(ids['E_cuoi'])
    if not lo < hi:
        raise KCError('cua_so_v1: E_dau does not end before E_cuoi starts (KCF1-8)')
    try:
        fr = [r for r in csv.DictReader(io.StringIO(fb.decode('utf-8'))) if _true(r.get('valid'))
              and lo < int(r['idx']) < hi and r.get('tag_x') not in ('', None)]
        poses = [(int(r['idx']), float(r['t_s']), float(r['tag_x']), float(r['tag_y']), _ids_of(r['moc_dung']))
                 for r in fr]
    except (KeyError, ValueError, TypeError) as e:
        raise KCError(f'frames.csv: {e} (KCF1-8; the 33-mark track writes moc_dung)')
    if not poses:
        raise KCError('no valid moving E frame between E_dau and E_cuoi (E_doan)')
    to = tr.get('tag_offset')
    to = [0.0, 0.0, 0.0] if to is None else to
    if not isinstance(to, list) or len(to) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in to):
        raise KCError(f'track.tag_offset {to!r}: null or [dx_m, dy_m, dpsi_rad] (KCF1-8)')
    side = float(s['car_side_m'])
    # full version (CODE 20/11): the detections of the E clip (dets.npz named by the track, sha checked)
    dp_ = a.dets or tr.get('dets_file')
    dets, dets_sha, _ = load_dets_checked(dp_, _hex(tr.get('dets_sha256'), 'track.dets_sha256'), None, clip,
                                          'E dets.npz')
    fp_ = FramePoses(s, pc)
    rx = R_XP_M if a.r_xp is None else a.r_xp
    per, dev = [], 0.0
    for fid, t, x, y, anc in poses:
        if fid not in dets or cm.CAR_ID not in dets[fid][1]:
            raise KCError(f'E frame {fid}: valid with a tag in frames.csv but not in dets.npz with the car tag')
        d = dets[fid][1]
        cam = fp_.full(anc, d, f'E frame {fid}')
        r = cm.car_pose(cam, d[cm.CAR_ID], side, kappa, C, to)
        dev = max(dev, abs(r['tag_x'] - x), abs(r['tag_y'] - y))
        if not dev <= TOL_REPLAY_M:
            raise KCError(f'E frame {fid}: the replayed scoring pose differs from frames.csv by {dev:.3g} m '
                          f'(> {TOL_REPLAY_M}: another session, detections or anchors)')
        uv = np.asarray(d[cm.CAR_ID], float)
        per.append((fid, t, anc, d, uv, lift(cam.pix2floor(uv, 0.0), kappa, C)))
    foot = np.vstack([unlift(P, kappa, C) for *_r, P in per])             # KCF1-4: floor footprint, not lifted
    hid = h32_ids(mk, foot, rx)
    cases = held_cases(pc, hid)
    a3_ca = {n: a3_held_case(n, pc.img, mk, full, cam, ids_, (side, float(s['tag_h_m'])), l_max)
             for n, ids_, cam in cases}
    khung = []
    for fid, t, anc, d, uv, P in per:
        du = {}
        for n, ids_, _ in cases:
            ch = fp_.held(ids_, anc, d, f'E frame {fid}')
            res = lift(ch.pix2floor(uv, 0.0), kappa, C) - P
            if not np.all(np.isfinite(res)):
                raise KCError(f'E frame {fid}, case {n}: non-finite residual (KCF1-5)')
            du[n] = [[float(v) for v in xx] for xx in res]
        khung.append(dict(frame_id=fid, t_s=t, goc_m=[[float(v) for v in xx] for xx in P], du_m=du))
    out = _with_sha(dict(
        schema=MEO_E_SCHEMA, cho=a.cho, trang_thai=MEO_TRANG_THAI, ly_do=[], chieu='held_minus_full',
        bo_H32=True, truoc_le=True,
        thu_tu_goc='ArUco 0..3 (goc anh that cua ma noc trong dets.npz, thu tu ArUco)',
        clip_sha256=clip, nguon=[dict(vai_tro='track', duong_dan=os.path.abspath(tp), sha256=tr_sha),
                                 dict(vai_tro='frames', duong_dan=os.path.abspath(fp), sha256=fr_sha),
                                 dict(vai_tro='dets', duong_dan=os.path.abspath(dp_), sha256=dets_sha),
                                 dict(vai_tro='cua_so_v1', duong_dan=os.path.abspath(a.cua_so), sha256=cs_sha),
                                 dict(vai_tro='session', duong_dan=os.path.abspath(a.session), sha256=ses_sha)],
        mien_E=dict(sau_E_dau=lo, truoc_E_cuoi=hi, khung_dong=len(khung)),
        ban_do=dict(phien_ban=VERSION, session_sha256=ses_sha, full='tung_khung: ChainCtx.pose(xe, moc_dung)',
                    held='tung_khung: refit_pose_checked tu camera clip H khop khong co ca, tren moc_dung tru ca',
                    H32_ids=hid, R_XP_M=rx, lech_phat_lai_max_m=dev,
                    ca=[dict(ten=n, bo_ids=ids_, A3=a3_ca[n]) for n, ids_, _ in cases]),
        h_m=float(s['tag_h_m']), h_bar=_check_h_bar(a.a6, float(s['tag_h_m']), C, float(s['tag_h_m']) / kappa), kappa=kappa,
        plumb_xy_m=[float(x) for x in C], Zc_m=float(s['tag_h_m']) / kappa, tag_offset=[float(v) for v in to],
        khung=khung, commit_hil=commit_hil(),
        gioi_han=['chi so thuc nghiem truoc le; truong du ban do giua camera bo moc H32 va camera cham, trong ho mo hinh',
                  'camera tung khung: camera cham phat lai dung chuoi xe cua track (moc_dung, cam_H[xe]), lech so voi '
                  'frames.csv <= TOL_REPLAY_M; camera bo ca khop lai tren cung khung, cung neo tru ca; goc_m = 4 goc '
                  'anh that cua ma noc qua camera cham, nang car_pose',
                  'ban_do.ca[*].A3 (CODE 20/11): moi cap moc trong tap bo cua ca, L <= L_max, do tren anh session qua '
                  'camera cua ca (khop khong co ca tap bo), gap qua a3_tag_gap voi camera cham session.camera; '
                  'quy_deg = |dtheta| + gap_deg nhu a3_clip; quy_max_deg = max cac cap (null khi khong co cap)']))
    write_new(a.out, out)
    _say(f'meo_ma_ao_E {a.cho}: {len(khung)} khung, {len(cases)} ca H32 (camera tung khung), lech phat lai '
         f'{dev:.2e} m, A3 quy_max_deg theo ca: '
         + ', '.join(f'{n}={v["quy_max_deg"]:.4f}' if v['quy_max_deg'] is not None else f'{n}=khong_cap'
                     for n, v in a3_ca.items()))


def _ll_session(ll, a):
    """KCF1-2: the session of a single-session F8 file (no --session): F8 session_file, else F6 (hieu_chuan.file, sha
    checked) session.file; its sha must be F8 session_sha256."""
    if ll.get('session_file'):
        return ll['session_file']
    hc = ll.get('hieu_chuan') or {}
    if not hc.get('file'):
        raise KCError('lan_lap.json: no session path (session_file, hieu_chuan.file) and no --session (KCF1-2)')
    f6, sha6 = load_json_sha(hc['file'])
    if sha6 != _hex(hc.get('sha256'), 'lan_lap.hieu_chuan.sha256'):
        raise KCError('hieu_chuan_neo.json: sha256 is not the one lan_lap.json names (KCF1-7)')
    path = (f6.get('session') or {}).get('file')
    if not path:
        raise KCError('hieu_chuan_neo.json: no session.file (KCF1-2)')
    return path


def meo_bang(a):
    """F15 (b): every anchor (neo_ten: A, B, oc*) of every run end of lan_lap.json and of every window of the anchor
    calibration file (hieu_chuan_neo.json cua_so[*].Ph, KCF1-3), lifted with kappa_n = h_n / Z_c as c7_nhac_tam, per
    place, map residual held - full per H32 case through the PER-FRAME cameras of the anchor frames (khung_neo) of
    the clip (dets.npz of CODE_CAM, --dets; full version CODE 20/11): on every anchor frame the scoring camera is the
    pose refitted on the session marks seen (as c7_nhac_tam do_khung), the anchor pixel is the lifted point brought
    back through it, the held camera of a case is refitted on the same frame without the case; the residual of an item
    is the mean over its anchor frames (Ph is the mean over the same frames)."""
    ll, ll_sha = load_json_sha(a.lan_lap)
    if ll.get('schema') != 'lan_lap_v1':
        raise KCError(f'lan_lap.json: schema {ll.get("schema")!r} is not lan_lap_v1 (KCF1-7)')
    ten = list(ll.get('neo_ten') or [])
    if ten[:2] != ['A', 'B'] or any(not (t in ('A', 'B') or (isinstance(t, str) and t.startswith('oc')))
                                    for t in ten) or len(set(ten)) != len(ten):
        raise KCError(f'lan_lap.json: neo_ten {ten!r} is not A, B, oc* (F15, KCF2-1)')
    thieu_oc = not any(t.startswith('oc') for t in ten)
    # sessions (CODE 20/11 khuya, chain A): every item is measured through the cameras of ITS session (run end
    # session_sha256, F6 window meta.session_sha); --session repeated gives the files; without it the single session of
    # the F8 (KCF1-2). The car_pose set-up (kappa_n, Z_c, plumb) and the floor marks must be the same in every session.
    lls = [x for x in (ll.get('sessions') or []) if isinstance(x, dict)]   # F8 of several sessions (NHACTAM 2bf72b3)
    paths = list(a.session or []) or ([x.get('file') for x in lls] if lls else [_ll_session(ll, a)])
    if not all(paths):
        raise KCError('lan_lap.json sessions: a session without file (and no --session)')
    ses = {}
    for sp_ in paths:
        sd_, sha_ = load_json_sha(sp_)
        if sha_ in ses:
            raise KCError(f'--session: {sp_} given twice (sha {sha_})')
        s_, _, mk_, pc_, _, kt_, C_ = _session_ctx(sp_, sha_, a.model)
        ses[sha_] = dict(file=os.path.abspath(sp_), s=s_, mk=mk_, pc=pc_, kappa_t=kt_, C=C_)
    top = ll.get('session_sha256')
    if lls:
        allowed = {_hex(x.get('sha256'), 'lan_lap.sessions[].sha256') for x in lls}
        if top is not None and top not in allowed:
            raise KCError('lan_lap.json: session_sha256 not in sessions (KCF1-7)')
    elif top:
        allowed = set(top if isinstance(top, list) else [top])
    else:
        raise KCError('lan_lap.json: no session_sha256 / sessions (KCF1-7)')
    if not set(ses) <= allowed:
        raise KCError(f'--session {sorted(set(ses) - allowed)} not a session of lan_lap.json (KCF1-7)')
    s0 = next(iter(ses.values()))
    s, mk, C, kappa_t = s0['s'], s0['mk'], s0['C'], s0['kappa_t']
    for v in ses.values():
        if v['mk'] != mk or np.max(np.abs(v['C'] - C)) > 1e-12 or abs(float(v['s']['tag_h_m']) / v['kappa_t']
                                                                       - float(s['tag_h_m']) / kappa_t) > 1e-9:
            raise KCError('sessions with other floor marks / plumb point / Z_c: one car_pose set-up needed')
    ses_sha = next(iter(ses)) if len(ses) == 1 else None
    Zc = float(s['tag_h_m']) / kappa_t
    h_n = float(ll['h_n_m'])
    kappa_n = h_n / Zc
    items = []
    for r in ll.get('luot') or []:
        for dau, x in sorted((r.get('dau') or {}).items()):
            if dau == 'dich' or not isinstance(x, dict) or x.get('Ph_neo_m') is None:
                continue
            if x.get('session_sha256') not in ses:
                raise KCError(f'lan_lap.json {r.get("luot_id")} {dau}: session sha missing or not given (KCF1-7)')
            _hex(x.get('khung_neo_sha256'), f'lan_lap.json {r.get("luot_id")} {dau} khung_neo_sha256')
            kn = x.get('khung_neo') or {}
            _hex(x.get('clip_sha256'), f'lan_lap.json {r.get("luot_id")} {dau} clip_sha256')
            if not isinstance(kn, dict) or not isinstance(kn.get('id'), list) \
                    or not all(isinstance(v, int) and not isinstance(v, bool) for v in kn['id']) \
                    or x['khung_neo_sha256'] not in _ids_sha_forms(kn['id']):
                raise KCError(f'lan_lap.json {r.get("luot_id")} {dau}: khung_neo_sha256 is not its frame set (KCF1-7)')
            items.append(dict(nguon='lan_lap', luot_id=r.get('luot_id'), dau=dau, cho=x.get('cho') or r.get('cho'),
                              clip_sha256=x.get('clip_sha256'), khung_neo_sha256=x.get('khung_neo_sha256'),
                              P=x['Ph_neo_m'], _khung=list(kn['id']), _ses=x['session_sha256']))
    hc = ll.get('hieu_chuan') or {}
    hc_rec = None
    if hc.get('file'):                                 # KCF1-3: every window of the anchor-shape calibration too
        f6, sha6 = load_json_sha(hc['file'])
        if sha6 != _hex(hc.get('sha256'), 'lan_lap.hieu_chuan.sha256'):
            raise KCError('hieu_chuan_neo.json: sha256 is not the one lan_lap.json names (KCF1-7)')
        f6s = f6.get('sessions') or [f6.get('session') or {}]
        if not all(isinstance(x, dict) and x.get('sha256') in allowed for x in f6s):
            raise KCError('hieu_chuan_neo.json: a session sha256 that is not one of lan_lap.json (KCF1-7)')
        hc_rec = dict(file=os.path.abspath(hc['file']), sha256=sha6)
        for k, w in enumerate(f6.get('cua_so') or []):
            if not isinstance(w, dict) or w.get('Ph') is None:
                continue
            m = w.get('meta') or {}
            sw = m.get('session_sha') or (f6s[0].get('sha256') if len(f6s) == 1 else None)   # single-session F6
            if sw not in ses:
                raise KCError(f'hieu_chuan_neo.json cua_so[{k}]: session {sw!r} not given (--session)')
            kn = w.get('khung_neo') or {}
            if not isinstance(kn, dict) or not isinstance(kn.get('id'), list) or not kn['id'] or not all(
                    isinstance(v, int) and not isinstance(v, bool) for v in kn['id']):
                raise KCError(f'hieu_chuan_neo.json cua_so[{k}]: khung_neo.id missing (per-frame cameras need it)')
            items.append(dict(nguon='hieu_chuan_neo', chi_so=k, loai=m.get('loai'), luot_id=m.get('luot'),
                              cho=m.get('cho'), clip_sha256=_hex(m.get('clip_sha'), f'cua_so[{k}].meta.clip_sha'),
                              khung_neo_sha256=w.get('khung_neo_sha256'), P=w['Ph'], _khung=list(kn['id']),
                              _ses=sw))
    if not items:
        raise KCError('lan_lap.json: no anchor point (Ph_neo_m / cua_so Ph)')
    by_cho = {}
    for it in items:
        if it['cho'] not in CHO:
            raise KCError(f'anchor point of place {it["cho"]!r}')
        P = np.asarray(it['P'], float).reshape(-1, 2) if np.size(it['P']) else np.zeros((0, 2))
        if len(P) != len(ten) or not np.all(np.isfinite(P)):
            raise KCError(f'{it["nguon"]} {it.get("luot_id")} {it.get("dau", it.get("chi_so"))}: {len(P)} anchors, '
                          f'neo_ten has {len(ten)}')
        it['P'] = P
        by_cho.setdefault(it['cho'], []).append(it)
    rx = R_XP_M if a.r_xp is None else a.r_xp
    hids = {cho: h32_ids(mk, np.vstack([unlift(it['P'], kappa_n, C) for it in its]), rx)    # KCF1-4: floor
            for cho, its in by_cho.items()}
    cases, seen = [], set()                           # KCF1-3: the cases of every place, applied to every place
    used = sorted({it['_ses'] for its in by_cho.values() for it in its})
    for cho in sorted(hids):
        for sh in used:                               # KCF1-5: every case fits in every session that has items
            for n, ids_, cam in held_cases(ses[sh]['pc'], hids[cho]):
                if n not in seen:
                    seen.add(n)
                    cases.append((n, ids_, cam))
    # full version: the detections of every clip of the items (--dets, matched by the video sha inside the file)
    dets_by, dets_rec = {}, []
    for pth in a.dets or []:
        by, sha_d, vsha = load_dets_checked(pth, what='--dets')
        if vsha in dets_by:
            raise KCError(f'--dets: two files of clip {vsha}')
        dets_by[vsha] = by
        dets_rec.append(dict(duong_dan=os.path.abspath(pth), sha256=sha_d, clip_sha256=vsha))
    fps = {sh: FramePoses(ses[sh]['s'], ses[sh]['pc']) for sh in used}
    fits = {sh: set(int(i) for i in ses[sh]['s']['fit_ids']) for sh in used}
    theo, n_khung, trai = {}, 0, 0.0
    for cho, its in sorted(by_cho.items()):
        hid = hids[cho]
        muc, muc_hc = [], []                            # run ends (keyed luot_id, dau) apart from calibration windows
        for it in its:
            where = f"{it['nguon']} {it.get('luot_id')} {it.get('dau', it.get('chi_so'))}"
            by = dets_by.get(it['clip_sha256'])
            if by is None:
                raise KCError(f'{where}: no --dets of clip {it["clip_sha256"]} (per-frame cameras)')
            Pf = np.column_stack([unlift(it['P'], kappa_n, C), np.zeros(len(it['P']))])
            fp_, fit_ids = fps[it['_ses']], fits[it['_ses']]
            acc = {n: [] for n, _, _ in cases}
            for fid in it['_khung']:
                if fid not in by:
                    raise KCError(f'{where}: anchor frame {fid} not in the dets.npz of the clip')
                d = by[fid][1]
                anc = sorted(i for i in d if i in fit_ids)
                cam = fp_.full(anc, d, f'{where} frame {fid}')
                uv = cam.project(Pf)
                P0 = lift(cam.pix2floor(uv, 0.0), kappa_n, C)
                for n, ids_, _ in cases:
                    ch = fp_.held(ids_, anc, d, f'{where} frame {fid}')
                    r_ = lift(ch.pix2floor(uv, 0.0), kappa_n, C) - P0
                    if not np.all(np.isfinite(r_)):
                        raise KCError(f'{where} frame {fid}, case {n}: non-finite residual (KCF1-5)')
                    acc[n].append(r_)
                n_khung += 1
            res = {}
            for n, v in acc.items():
                v = np.array(v)
                res[n] = v.mean(axis=0)
                trai = max(trai, float(np.max(np.linalg.norm(v - res[n][None], axis=2))))
            rec = {k: v for k, v in it.items() if k not in ('P', 'cho', 'nguon', '_khung', '_ses')}
            rec['session_sha256'] = it['_ses']
            rec['n_khung_neo'] = len(it['_khung'])
            rec['neo'] = {t: {n: [float(v) for v in res[n][k]] for n in res} for k, t in enumerate(ten)}
            (muc if it['nguon'] == 'lan_lap' else muc_hc).append(rec)
        theo[cho] = dict(H32_ids=hid, ca=[dict(ten=n, bo_ids=ids_) for n, ids_, _ in cases], muc=muc,
                         muc_hieu_chuan=muc_hc)
    # every place carries the same case list (names = left-out id sets); H32_ids is the place's own set
    out = _with_sha(dict(
        schema=MEO_BANG_SCHEMA, trang_thai=MEO_TRANG_THAI, ly_do=[], chieu='held_minus_full', bo_H32=True,
        truoc_le=True, neo_ten=ten, h_neo_m=ll.get('h_neo_m'), h_n_m=h_n, kappa_n=kappa_n, Zc_m=Zc,
        plumb_xy_m=[float(c) for c in C], thieu_oc=thieu_oc,
        quy_uoc='diem neo da nang car_pose (p = pix2floor(uv, 0), P = p + kappa_n (C - p)), m',
        lan_lap=dict(file=os.path.abspath(a.lan_lap), sha256=ll_sha, hieu_chuan_sha256=hc.get('sha256')),
        hieu_chuan_neo=hc_rec,
        ban_do=dict(phien_ban=VERSION, session_sha256=ses_sha,
                    sessions=[dict(file=v['file'], sha256=sh) for sh, v in ses.items()],
                    full='tung_khung: pose khop lai tren moc session thay duoc (nhu c7_nhac_tam do_khung)',
                    held='tung_khung: refit_pose_checked tu camera clip H khop khong co ca, cung moc tru ca',
                    R_XP_M=rx, n_khung=n_khung, trai_trong_muc_max_m=trai),
        dets=dets_rec,
        theo_cho=theo, commit_hil=commit_hil(),
        gioi_han=['chi so thuc nghiem truoc le; trong ho mo hinh camera',
                  'pixel neo tung khung = diem neo da nang (trung binh tren khung neo) dua nguoc qua camera cham cua '
                  'khung; du cua muc = trung binh tren khung neo (trai_trong_muc_max_m: lech lon nhat quanh trung binh)',
                  'camera cham tung khung khop tu cam_H[xe] co co tin cay, khong noi tiep khung truoc nhu do_khung',
                  'moi neo nang bang kappa_n = h_n / Z_c nhu c7_nhac_tam; do cao rieng tung neo (h_neo_m) chi ghi lai',
                  'NHACTAM chay lai khop hinh neo (cua_so hieu_chuan_neo) va giai beta bang cac du nay (V4-06)']))
    write_new(a.out, out)
    _say(f"meo_dau_chan_bang: {sum(len(v['muc']) for v in theo.values())} muc, cho {sorted(theo)}, thieu_oc {thieu_oc}"
         f', {n_khung} khung neo (camera tung khung)')


# ---------------------------------------------------------------- CLI
def _load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def parse_xy(s, what):
    try:
        x, y = (float(v) for v in str(s).split(','))
    except ValueError:
        raise KCError(f'{what} {s!r}: X,Y in metres')
    if not (math.isfinite(x) and math.isfinite(y)):
        raise KCError(f'{what} {s!r}: not finite')
    return (x, y)


def parse_xp(specs):
    """--xuat-phat NAME=X,Y (NAME in goc, K1, K2; each once)."""
    out = {}
    for spec in specs or ():
        k, _, v = spec.partition('=')
        if k not in CHO or k in out or not v:
            raise KCError(f'--xuat-phat {spec!r}: goc|K1|K2=X,Y, each once')
        out[k] = parse_xy(v, f'--xuat-phat {k}')
    return out


def commit_hil():
    """(S7-16) git HEAD of this tool's repository and whether its working tree is dirty; None outside git."""
    import subprocess
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=HERE,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return dict(sha=sha, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return None


KHOA_S7_SCHEMA = 'c7_khoa_s7_v1'


def manifest_git(path):
    """dict(commit, time, sach) of the last git commit touching path in its own repository; sach = the file has no
    change against that commit; None outside git or when never committed."""
    import subprocess
    d, f = os.path.split(os.path.abspath(path))
    try:
        log = subprocess.run(['git', 'log', '-1', '--format=%H %cI', '--', f], cwd=d, capture_output=True, text=True,
                             check=True).stdout.strip()
        st = subprocess.run(['git', 'status', '--porcelain', '--', f], cwd=d, capture_output=True, text=True,
                            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not log:
        return None
    c, t = log.split(' ', 1)
    return dict(commit=c, time=t, sach=not st)


def check_khoa_s7(path, hang_so_s7, ch):
    """KCJ1-14 (S7-16, J v6 6 (1)): the manifest {schema c7_khoa_s7_v1, hang_so_s7, commit_hil} must equal the
    constants in force and the code commit of this run (clean tree), and be itself committed unchanged in git (its
    commit time is the lock time C8B compares with the data). Returns the record written into kiem_cheo.json."""
    m, sha = load_json_sha(path)
    want = json.loads(json.dumps(hang_so_s7))
    got = m.get('hang_so_s7') or {}
    if m.get('schema') != KHOA_S7_SCHEMA:
        raise KCError(f'--khoa-s7: schema is not {KHOA_S7_SCHEMA}')
    if got != want:
        diff = sorted(k for k in set(want) | set(got) if got.get(k) != want.get(k))
        raise KCError(f'--khoa-s7: constants differ from those in force {diff}; in force: {json.dumps(want)}')
    if ch is None or ch.get('dirty') or m.get('commit_hil') != ch.get('sha'):
        raise KCError(f'--khoa-s7: commit_hil {m.get("commit_hil")!r} is not the clean code commit of this run {ch!r}')
    g = manifest_git(path)
    if g is None or not g['sach']:
        raise KCError('--khoa-s7: the manifest is not committed unchanged in git (lock before the data, S7-16)')
    return dict(file=os.path.abspath(path), sha256=sha, git_commit=g['commit'], git_time=g['time'])


def _say(msg):
    """KCJ1-6: the summary line after a file is published never turns the result into code 2."""
    try:
        print(msg)
    except (OSError, ValueError):
        pass


def thu_flag(thu, max_frames):
    """KCC1-10: a result made from a cut clip (--max-frames) is a trial, never exported."""
    return bool(thu or max_frames)


def phan_tich(a):
    clips_in = {'H_a': (a.H_a, a.video_a), 'H_b': (a.H_b, a.video_b), 'H_m': (a.H_m, a.video_m),
                'H_z': (a.H_z, a.video_z)}
    clips_in = {n: v for n, v in clips_in.items() if v[0]}
    doan = None
    if getattr(a, 'doan', None):                     # camera segment (Liam 20/11 "Theo doan, bo H giua")
        if a.H_m:
            raise KCError('--doan: a camera segment has no H_m (Liam 20/11 "Theo doan, bo H giua")')
        if a.khong_H_cuoi and a.H_z:
            raise KCError('--doan --khong-H-cuoi with --H-z')
        co = ['H_a', 'H_b'] + ([] if a.khong_H_cuoi else ['H_z'])
        doan = dict(ten=a.doan, vai_tro_co=co, vai_tro_vang=[n for n in CLIPS_REQUIRED if n not in co],
                    nguon='Liam 20/11, CODE_KIEMCHEO local_599d3188, hop chon: "Theo doan, bo H giua (Recommended)"',
                    gioi_han=[DOAN_GIOI_HAN] + ([DOAN_KHONG_CUOI] if a.khong_H_cuoi else []))
    elif getattr(a, 'khong_H_cuoi', False):
        raise KCError('--khong-H-cuoi only with --doan')
    need = vai_tro(dict(doan=doan))
    missing = [n for n in need if n not in clips_in]
    if missing and not a.thu:
        raise KCError(f'clips H missing: {missing} (design v2 7: H_a, H_b, H_m, H_z; segment: {list(need)})')
    clips, raw, ses_sha = {}, {}, {}
    for name, (p, _) in clips_in.items():
        raw[name], ses_sha[name] = load_json_sha(p)            # KCJ1-6: the sha of the bytes used
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
    # KS7-7: one ID set over every clip H, equal to the measured floor marks (no repeat: ids are dict keys)
    if not a.thu:
        diff = {n: sorted(set(c[0]) ^ set(ids_a)) for n, c in clips.items() if set(c[0]) != set(ids_a)}
        if diff:
            raise KCError(f'clips H fit different mark ids than H_a: {diff} (KS7-7)')
        if need and set(ids_a) != need:
            raise KCError(f'fit ids {ids_a} are not the measured floor marks {sorted(need)} (KS7-7)')
        if tuple(ids_a) != ID_KHOA:                   # KCJ1-10: the official path is the 32-mark floor, ids 1-32
            raise KCError(f'fit ids {ids_a} are not the locked ids 1-32 (KS7-7; old 24-mark sessions: --thu)')
    xp = parse_xp(a.xuat_phat)
    plumb = parse_xy(a.plumb, '--plumb') if a.plumb else None
    if xp and plumb is None:
        raise KCError('--xuat-phat needs --plumb (the car_pose plumb point of the scoring, S7 chuyen_sj)')
    groups = session_groups(raw['H_a'], ids_a, a.groups)
    # KCJ1-14 (S7-16, J v6 6 (1)): the S7 constants in force, checked against a manifest locked (committed) BEFORE
    # the held-out data are looked at; recorded with its sha and commit
    tag = session_tag(raw)
    hang_so_s7 = dict(R_XP_M=R_XP_M, R_TRONG_M=R_TRONG_M, MIEN_XP_M=MIEN_XP_M, MIEN_XP_BUOC_M=MIEN_XP_BUOC_M,
                      HAI_PHUONG_DEG=HAI_PHUONG_DEG, DUNG_SAI_IN_GIAN=DUNG_SAI_IN_GIAN,
                      DUNG_SAI_IN_DU_MM=DUNG_SAI_IN_DU_MM, L_MAX_M=a.L_max, A3_TAG_T=list(A3_TAG_T),
                      A3_TAG_PHI_DEG=list(A3_TAG_PHI_DEG), R_XP_DUONG_M=R_XP_DUONG_M,
                      DUONG_12B={k: list(v) for k, v in sorted(DUONG_12B.items())},
                      xuat_phat={k: list(v) for k, v in sorted(xp.items())},
                      plumb=None if plumb is None else list(plumb), car_side_m=tag[0], tag_h_m=tag[1], model=a.model,
                      nhom=groups, id_khoa=ids_a)
    ch = commit_hil()
    khoa = None
    if a.khoa_s7:
        khoa = check_khoa_s7(a.khoa_s7, hang_so_s7, ch)
    elif xp and not a.thu:
        raise KCError('--khoa-s7 needed with --xuat-phat (S7-16: constants locked before phan_tich, KCJ1-14)')
    tap = None
    if a.tap_moc:                                    # {chain: [[ids]...]}, every pass of KC2-1 merged (KCC1-2)
        tap = {str(c): [sorted(int(i) for i in S) for S in v] for c, v in _load_json(a.tap_moc).items()}
    a.thu = thu_flag(a.thu, a.max_frames)
    if not a.thu and not a.hang_so:
        raise KCError('--hang-so needed (KCC1-9)')
    # KCJ1-6: every output name is free before anything is computed or written
    outs = ['u_so.json', 'kiem_cheo.json', 'a3_cap.csv'] + (['cap_tuong_doi.csv'] if xp else [])
    busy = [n for n in outs if os.path.exists(os.path.join(a.out, n))]
    if busy:
        raise KCError(f'{a.out}: {busy} exist: not replaced')
    # A3 over every clip H, carried to the car tag of the sessions (design v5 S5-1)
    pcs = {n: PairCams(img, mk, a.model, w, h, zc) for n, (img, mk, w, h, zc) in clips.items()}
    a3_hong = {}
    a3 = {n: a3_clip(img, mk, a.model, w, h, zc, a.L_max, sig, tag=tag, pc=pcs[n], hong=a3_hong.setdefault(n, []))
          for n, (img, mk, w, h, zc) in clips.items()}
    roles = {i: r.get('vai_tro', '') for i, r in moc_rows.items()}
    A3 = a3_summary(a3, [i for i in ids_a if roles.get(i) == 'moc_giua'], tag=tag)
    A3['cap_hong'] = [dict(h, clip=n) for n, v in a3_hong.items() for h in v]      # KCJ1-13: never dropped silently
    A3['du_lieu_du'] = not A3['cap_hong']
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
    # S7 (design v5 S7, v6 S8-4): relative P1 / A3, local distortion, common scale
    s7, cap_rows = {}, []
    meo = {n: meo_clip(pcs[n], clips[n][1]) for n in clips}
    s7['meo_cuc_bo'] = meo_summary(meo)
    s7['thang_chung'] = dict(theo_clip={n: thang_clip(p1[n], clips[n][1]) for n in clips})
    tl = [v['ti_le_tru_1'] for v in s7['thang_chung']['theo_clip'].values()]
    s7['thang_chung'].update(lech_max=float(max(tl) - min(tl)), loai='thong tin (C8BX-03)')
    if xp:
        regions = start_regions(mk_a, xp, R_XP_M)
        rel, a3rel, cover = {}, {}, {}
        for n, (img, mk, w, h, zc) in clips.items():
            if zc is None:
                raise KCError(f'{n}: session without zc_tape_m (kappa of car_pose)')
            rows, pairs, hong = p1_rel_clip(n, pcs[n], mk, regions, tag, tag[1] / zc, plumb, a.L_max)
            cap_rows += rows
            rel[n] = dict(pairs=pairs, hong=hong, n_du_kien=sum(len(r['moc']) * (len(pcs[n].ids) - 1)
                                                                for r in regions.values()))
            a3rel[n] = a3_rel_clip(a3[n], regions)
            cover[n] = {k: region_cover(regions[k]['moc'], mk, xp[k], a3[n], R_TRONG_M, HAI_PHUONG_DEG, a.L_max)
                        for k in regions}
        s7['P1_tuong_doi'], s7['A3_tuong_doi'] = s7_summary(rel, a3rel, cover, regions, a3_hong)
        dregs = duong_regions(mk_a, xp, R_XP_DUONG_M)                       # HCV3-2: the 12b corridor of each place
        dcover = {n: {k: duong_cover(r['moc'], clips[n][1], *r['doan_m'], a3[n], R_TRONG_M, HAI_PHUONG_DEG, a.L_max)
                      for k, r in dregs.items()} for n in clips}
        s7['A3_tuong_doi']['duong_12b'] = duong_summary(dregs, dcover, a3rel, a3_hong)
    else:
        s7['P1_tuong_doi'] = s7['A3_tuong_doi'] = None
    sess = {n: dict(file=os.path.abspath(p), sha256=ses_sha[n], video=os.path.abspath(clips_in[n][1]),
                    video_sha256=raw[n].get('video_sha256')) for n, (p, _) in clips_in.items()}   # TD3-2, TD1-3
    common = dict(buoi=a.buoi, model=a.model, nhom=groups, sessions=sess, thu=bool(a.thu),
                  moc=[dict(file=os.path.abspath(p), sha256=sha256_file(p)) for p in a.moc or ()],
                  hang_so_sha256=sha256_file(a.hang_so) if a.hang_so else None)
    res = dict(version=VERSION, **common, doan=doan, L_max_m=a.L_max, id_khoa=ids_a, hang_so_s7=hang_so_s7,
               commit_hil=ch, khoa_s7=khoa, **s7,
               A3=dict(A3, buoi=a.buoi),
               P1=dict(bo_nhom_max_m=worst[2], tung_khung_max_m=None, max_clip=worst[0], max_id=worst[1], buoi=a.buoi,
                       tung_khung='tu file tung_khung (lenh tung_khung, track sau cham lai cua CODE_CAM)',
                       theo_clip={n: {str(i): [float(v[0]), float(v[1])] for i, v in d.items()} for n, d in p1.items()}),
               moc_dich=dict(nguong_m=NGUONG_CO_DINH_M, u_so_m=uso, buoi=a.buoi, chuoi_xe=xe, **uso_info,
                             nghia='u_so_m = max(u_so, u_so_tho) cua chuoi xe; nguong T = nguong_m + u_so; neu u_so '
                                   'chan sai phep so thi dich chua phat hien <= T + u_so (KC1-3)'),
               P7=dict(lech_max_m=zp[2]['lech_max_m'], max_clip=zp[0], max_id=zp[1], buoi=a.buoi,
                       sd_max_m=max(r['sd_m'] for _, _, r in allz), theo_co=noise_by_size(allz)),
               A5=dict(lech_max_deg=za[2]['lech_max_deg'], max_clip=za[0], max_id=za[1], buoi=a.buoi,
                       sd_max_deg=max(r['sd_deg'] for _, _, r in allz), theo_co=noise_by_size(allz),
                       gioi_han_c8bx01='dich / xoay cung chung cua tu giac do do canh khong lo o du 4 goc; chuyen moc '
                                       'san sang ma noc la gioi han (g)',
                       gioi_han='moc san 96,4 mm, ma noc 144,5 mm: nhieu goc moc san lon hon; so khung = cac clip H, '
                                'lech lon nhat tang theo so mau; moc dung yen, khong gom nhoe (P4, A4)'),
               nhieu_theo_clip={n: {str(i): r for i, r in d.items()} for n, d in nz.items()},
               theo_thoi_gian={n: time_check(dets['H_a'][1], clips[n][0], mk_a, groups, gc)
                               for n in clips if n != 'H_a'})
    os.makedirs(a.out, exist_ok=True)
    # CAM 21/11 (c7_measure.check_u_so_source, Codex C33-5a): the u_so file itself says whether it is a trial
    us_obj = dict(version=U_SO_VERSION, buoi=a.buoi, model=a.model, thu=bool(a.thu), nhom=groups, chuoi=chuoi,
                  chi_tiet=uso_info,
                  sessions=sess, tao_luc=datetime.datetime.now().isoformat(timespec='seconds'))
    us_bytes = json.dumps(us_obj, indent=1, ensure_ascii=False)
    res['u_so_sha256'] = hashlib.sha256(us_bytes.encode('utf-8')).hexdigest()   # KC1-10: the track must name it
    res['kiem_h'] = {n: kiem_h(dets[n][1], clips[n][0]) for n in clips}

    def csv_text(cols, rows):
        import io
        f = io.StringIO(newline='')
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
        return f.getvalue()
    files = {'u_so.json': us_bytes}
    if res['P1_tuong_doi'] is not None:               # KS7B-2: the signed numbers per pair, beside the JSON
        cols = ['clip', 'vung', 's', 'j', 'L_m', 'd_x', 'd_y', 'dh_x', 'dh_y', 'phi_deg', 'nhin', 'chuyen_x', 'chuyen_y',
                'chuyen_psi_deg']
        txt = csv_text(cols, [{k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()} for r in cap_rows])
        files['cap_tuong_doi.csv'] = txt
        res['P1_tuong_doi']['cap_csv'] = dict(file=os.path.abspath(os.path.join(a.out, 'cap_tuong_doi.csv')),
                                              sha256=hashlib.sha256(txt.encode('utf-8')).hexdigest())
    cols = ['clip', 'i', 'j', 'L_m', 'dtheta_deg', 'sigma_thuoc_deg', 'mid_x', 'mid_y', 'huong_deg', 'gap_deg',
            'quy_deg', 'dpsi_ma_max_deg', 'dtheta_hai_cam_deg']
    files['a3_cap.csv'] = csv_text(cols, [dict({k: r.get(k, '') for k in cols[1:]}, clip=n)
                                          for n, rows in a3.items() for r in rows])
    files['kiem_cheo.json'] = json.dumps(res, indent=1, ensure_ascii=False)
    done = []                                         # KCJ1-6: all or nothing
    try:
        for name in ('u_so.json', 'cap_tuong_doi.csv', 'a3_cap.csv', 'kiem_cheo.json'):
            if name in files:
                p = os.path.join(a.out, name)
                publish_new(p, lambda f, t=files[name]: f.write(t))
                done.append(p)
    except BaseException:
        for p in done:
            os.unlink(p)
        raise
    _say(f"A3: {A3['n_cap']} cap, quy ve ma noc max {A3['max_deg']:.3f} deg (cap {A3['max_deg_cap']:.3f}, gap "
          f"{A3['gap_max_deg']:.3f}; quy_ve_ma_noc {A3['quy_ve_ma_noc']}), RMS cap {A3['rms_deg']:.3f} deg; "
          f"P1 clip H: max {worst[2] * 1000:.2f} mm (moc {worst[1]}, {worst[0]}); u_so xe {uso * 1000:.2f} mm "
          f"(tinh {xe['tinh_m'] * 1000:.2f}, loc {xe['cua_so_m'] * 1000:.2f}, tho {xe['tho_m'] * 1000:.2f}); "
          f"P7 {zp[2]['lech_max_m'] * 1000:.2f} mm; A5 {za[2]['lech_max_deg']:.3f} deg"
          + ('  [THU: khong xuat duoc]' if a.thu else ''))


def check_ident(kc, tk):
    """S6-3: identity part (shared by tien_de and xuat, code 2): the frame-by-frame file belongs to this kiem_cheo
    result (version, session, groups, model, u_so, H_a session, marker and constants files, rescored)."""
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


def check_compat(kc, tk):
    """KC1-10: the frame-by-frame file belongs to this kiem_cheo result (check_ident) and is complete (xuat only:
    refused here, graded in tien_de)."""
    if kc.get('thu'):
        raise KCError('kiem_cheo result made with --thu: not exported')
    miss = [n for n in vai_tro(kc) if n not in (kc.get('sessions') or {})]
    if miss:
        raise KCError(f'kiem_cheo result without clips {miss}')
    check_ident(kc, tk)
    if not tk.get('hoan_tat') or tk.get('luot_chua_co') or tk.get('mau_thuan') or tk.get('luot_thieu_neo'):
        raise KCError(f"frame-by-frame file not complete: runs missing {tk.get('luot_chua_co')}, contradictions "
                      f"{tk.get('mau_thuan')}, anchors missing {tk.get('luot_thieu_neo')} (KCC1-11, TD1-2, TD2-1)")
    v = tk.get('tung_khung_max_m')
    _num(v, 'tung_khung_max_m')
    return float(v)


def moc_dich_block(a, kc, tk):
    """Block moc_dich of c8_cam_u_vao_v3 (c8b_u v3 198d2e9; design v6 S8-6): the U_moc block of CODE_CAM (--moc-dich:
    U_moc_xy_m, U_moc_psi_deg, phuong_phap_duyet, nguon {frames, lich, tap_moc, u_so, session, car_pose, bo_lay_mau})
    plus tien_de_buoi = ket_qua of this session's tien_de_buoi_v1 (--tien-de). Both must be of this kiem_cheo result:
    tien_de session_id = buoi, sha256.kiem_cheo = this file, sha256.u_so = kc.u_so_sha256 = moc_dich.nguon.u_so.
    The u_so of this file goes in as information (u_so_kiem_cheo). Returns (blk, file, sha)."""
    td = _load_json(a.tien_de)
    if td.get('format') != TIEN_DE_FORMAT or td.get('session_id') != a.buoi:
        raise KCError(f'--tien-de: not a {TIEN_DE_FORMAT} of session {a.buoi}')
    sh = td.get('sha256') or {}
    if sh.get('kiem_cheo') != sha256_file(a.kiem_cheo) or sh.get('u_so') != kc.get('u_so_sha256'):
        raise KCError('--tien-de: made from another kiem_cheo / u_so result (sha256)')
    if sh.get('tung_khung') is None or sh.get('tung_khung') != sha256_file(a.tung_khung):   # KCJ1-3
        raise KCError('--tien-de: made from another tung_khung (another scoring) than --tung-khung (sha256)')
    md = _load_json(a.moc_dich)
    if not isinstance(md, dict) or md.get('buoi') != a.buoi:
        raise KCError(f'--moc-dich: buoi {md.get("buoi") if isinstance(md, dict) else None!r} is not {a.buoi!r}')
    ng = md.get('nguon')
    if not isinstance(ng, dict) or sorted(set(MOC_NGUON_KEYS) - set(ng)):
        raise KCError(f'--moc-dich: nguon lacks {sorted(set(MOC_NGUON_KEYS) - set(ng or {}))} (c8b_u v3)')
    if ng.get('u_so') != kc.get('u_so_sha256'):
        raise KCError('--moc-dich: nguon.u_so is not the u_so.json of this kiem_cheo result')
    if ng.get('session') != kc['sessions']['H_a']['sha256']:                       # KCJ1-3: same scoring chain
        raise KCError('--moc-dich: nguon.session is not the H_a session of this kiem_cheo result')
    fr = ng.get('frames')                             # KCJ1-3, KCJ2-2: run id -> sha, each declared run its own frames;
    if not isinstance(fr, dict) or not fr:            #   a run of tung_khung absent here is allowed (clock-failed run,
        raise KCError('--moc-dich: nguon.frames empty')   # c8b_u v3 checks completeness against the clock files)
    bad = sorted(r for r, x in fr.items() if r not in tk['luot'] or tk['luot'][r]['nguon']['frames_sha256'] != x)
    if bad:
        raise KCError(f'--moc-dich: nguon.frames of runs {bad} are not the frames of these runs in --tung-khung')
    if not isinstance(md.get('phuong_phap_duyet'), str):
        raise KCError('--moc-dich: phuong_phap_duyet missing (c8b_u v3)')
    if 'tien_de_buoi' in md and md['tien_de_buoi'] != td.get('ket_qua'):
        raise KCError(f"--moc-dich: tien_de_buoi {md['tien_de_buoi']!r} is not the tien_de result {td.get('ket_qua')!r}")
    blk = dict(md)
    blk['tien_de_buoi'] = td.get('ket_qua')
    blk['tien_de_nguon'] = dict(file=os.path.abspath(a.tien_de), sha256=sha256_file(a.tien_de))
    blk['luot_cham'] = {r: v['nguon']['frames_sha256'] for r, v in sorted(tk['luot'].items())}    # KCJ3-1
    blk['luot_vang_u_moc'] = sorted(set(tk['luot']) - set(fr))   # must be clock-failed runs: khoa checks (C8B)
    blk['u_so_kiem_cheo'] = dict(u_so_m=kc['moc_dich'].get('u_so_m'), nguong_m=kc['moc_dich'].get('nguong_m'))
    return blk, a.moc_dich, sha256_file(a.moc_dich)


# ---------------------------------------------------------------- tien_de (design S6, S6-9, S6-10, S6-10b; v6 S8-3)
def _rehash(path, want, what):
    """TD3-2: the file at path must still have the SHA-256 of the evidence (else code 2)."""
    if not path or not os.path.isfile(path):
        raise KCError(f'{what}: file {path!r} not found (TD3-2)')
    got = sha256_file(path)
    if got != want:
        raise KCError(f'{what}: {path} has sha256 {got}, the evidence says {want} (TD3-2)')
    return dict(vai_tro=what, duong_dan=os.path.abspath(path), sha256=want)


def _fin_ge0(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def _fin01(v):
    return _fin_ge0(v) and v <= 1.0


def full_sets_ok(chain, tap_tinh, kc):
    """KCJ1-2: for every comparator of the chain, the full-set fit (every locked mark outside its excluded groups)
    must be among the computed sets (the set the runs use when every anchor is seen)."""
    groups = kc.get('nhom') or []
    ids = set(int(i) for i in kc.get('id_khoa') or [])
    got = {(x.get('nhom_dich'), tuple(sorted(x.get('tap') or []))) for x in tap_tinh or []}
    cmp_ = comparators(groups).get(chain)
    if not cmp_ or not ids:
        return False
    for t, E in cmp_.items():
        fit = tuple(sorted(i for i in ids if not any(i in groups[e] for e in E)))
        if (t, fit) not in got:
            return False
    return True


def tien_de_eval(kc, us, tk, sha_kc, sha_us, sha_tk):
    """S6-4 grading of one session (pure: no file read). kc, us, tk: the three results (tk None without
    --tung-khung). Returns the tien_de_buoi_v1 dict without nguon / ngay. Raises KCError on identity faults (code 2)."""
    sid = kc.get('buoi')
    if us.get('version') != U_SO_VERSION or us.get('buoi') != sid:
        raise KCError(f'u_so.json: not a {U_SO_VERSION} of session {sid}')
    if sha_us != kc.get('u_so_sha256'):
        raise KCError('u_so.json is not the one of the kiem_cheo result (sha256)')
    sess = kc.get('sessions') or {}
    vids = [s.get('video_sha256') for s in sess.values()]
    if len(set(vids)) != len(vids) or None in vids or len({s['sha256'] for s in sess.values()}) != len(sess):
        raise KCError('clips H: two clips with the same video or session sha256, or a video sha256 missing (TD1-3)')
    if tk is not None:
        check_ident(kc, tk)
        if tk.get('kiem_cheo_sha256') != sha_kc:
            raise KCError('tung_khung.json: made from another kiem_cheo result (sha256)')
        fr = [v['nguon']['frames_sha256'] for v in tk['luot'].values()]
        if len(set(fr)) != len(fr):
            raise KCError('tung_khung.json: two runs with the same frames (TD1-6)')
        plan = tk['ke_hoach'].get('luot') or []      # KCJ1-2: the runs written + the runs missing = the plan
        ev_runs = [x for x in tk['ke_hoach'].get('su_kien') or [] if x not in CLIPS_REQUIRED]
        if sorted(ev_runs) != sorted(plan):
            raise KCError('tung_khung.json: ke_hoach.luot is not the runs of ke_hoach.su_kien (KCJ1-2)')
        if sorted(set(tk['luot']) | set(tk.get('luot_chua_co') or [])) != sorted(plan) or \
                set(tk['luot']) & set(tk.get('luot_chua_co') or []):
            raise KCError('tung_khung.json: runs and missing runs do not make up the plan (KCJ1-2)')
    why, worst = [], 'dat'

    def bad(level, reason):
        nonlocal worst
        why.append(reason)
        order = ('dat', 'thieu_du_lieu', 'khong_dat')
        if order.index(level) > order.index(worst):
            worst = level
    if kc.get('thu'):
        bad('thieu_du_lieu', 'thu')
    roles = vai_tro(kc)
    for n in roles:
        if n not in sess:
            bad('thieu_du_lieu', f'thieu_clip:{n}')
    if not any(n in sess for n in ('H_m', 'H_z')):                  # camera segment without an end clip
        bad('thieu_du_lieu', 'khong_co_clip_kiem_troi')
    for n, r in (kc.get('kiem_h') or {}).items():
        v = r.get('ti_le_cung_luc')                 # KCJ1-2: NaN, inf, missing or not a number fails too
        if not _fin01(v) or not v >= KIEM_H_MIN:
            bad('thieu_du_lieu', f'kiem_h:{n}')
    for n in sess:
        if n not in (kc.get('kiem_h') or {}):
            bad('thieu_du_lieu', f'kiem_h:{n}')
    chains = us.get('chuoi') or {}
    khong = sorted(c for c, v in chains.items() if v.get('khong_tinh_duoc'))
    # KCJ1-2: every comparator chain of the groups, with finite numbers and computed sets
    need_c = set(comparators(kc.get('nhom') or []))
    for c in sorted(need_c | set(chains)):
        v = chains.get(c)
        nums = ('u_so_m', 'u_so_tho_m', 'nguong_m', 'nguong_tho_m')
        if c not in need_c or v is None or any(not _fin_ge0(v.get(k)) for k in nums) or \
                not full_sets_ok(c, v.get('tap_tinh'), kc):                    # KCJ1-2: the required sets
            if c not in khong:
                khong.append(c)
    khong.sort()
    for c in khong:
        bad('thieu_du_lieu', f'u_so_khong_tinh:{c}')
    # KCJ1-1 (J v6 J4-12, M3): the comparison at H_m and H_z must exist for EVERY locked mark with a finite own part;
    # a missing or partial comparison is missing data for the premise (not 'no contradiction seen')
    ids = [str(i) for i in kc.get('id_khoa') or []]
    if not kc.get('thu') and tuple(kc.get('id_khoa') or ()) != ID_KHOA:            # KCJ1-10 at the premise
        bad('thieu_du_lieu', 'id_khoa_khong_phai_1_32')
    for n in ('H_m', 'H_z'):
        if n not in sess:
            continue
        rm = ((kc.get('theo_thoi_gian') or {}).get(n) or {}).get('rieng_m') or {}
        lack = [i for i in ids if not _fin_ge0(rm.get(i))]                           # KCJ1-1: no bool, >= 0
        if not ids or lack:
            bad('thieu_du_lieu', f'so_{n}_thieu:{lack[:8] if ids else "id_khoa"}')
    luot, thieu_kiem, mau = [], [], []
    if tk is None:
        bad('thieu_du_lieu', 'khong_co_tung_khung')
    else:
        for p in plan_order_problems(tk['ke_hoach'].get('su_kien') or [], roles):
            bad('thieu_du_lieu', f'ke_hoach:{p}')
        for rid in tk.get('luot_chua_co') or []:
            bad('thieu_du_lieu', f'luot_chua_co:{rid}')
        luot = sorted(tk['luot'])
        for rid in luot:
            r = tk['luot'][rid]
            if r.get('thieu_du_lieu_neo'):
                bad('thieu_du_lieu', f'thieu_neo:{rid}')
            elif r.get('tap_chua_tinh'):
                bad('thieu_du_lieu', f'tap_neo_chua_tinh:{rid}')
            elif r.get('thieu'):
                thieu_kiem.append(rid)                # TD1-5 (B, Liam 18/11): only that run goes down
        mau = list(tk.get('mau_thuan') or [])
        for m in mau:
            bad('khong_dat', f"mau_thuan:{m['luot']}:{m['clip']}:{m['moc']}")
    xe = chains.get('xe')
    num_ok = xe is not None and not khong and set(chains) == need_c
    clip_h = {n: (dict(session=sess[n]['file'], sha256=sess[n]['sha256'], video=sess[n].get('video'),
                       video_sha256=sess[n].get('video_sha256'), kiem_h=(kc.get('kiem_h') or {}).get(n))
                  if n in sess else None) for n in roles}
    sha = dict(u_so=sha_us, kiem_cheo=sha_kc, tung_khung=sha_tk,
               ke_hoach=None if tk is None else tk['ke_hoach']['sha256'],
               sessions={n: s['sha256'] for n, s in sess.items()},
               clip_h={n: s.get('video_sha256') for n, s in sess.items()},
               luot=None if tk is None else {rid: dict(track=v['nguon']['track_sha256'], frames=v['nguon']['frames_sha256'],
                                                       kiem_cheo_khung=v['nguon']['kiem_cheo_khung_sha256'])
                                            for rid, v in tk['luot'].items()})
    return dict(format=TIEN_DE_FORMAT, session_id=sid, ket_qua=worst, ly_do=why, nguong_m=NGUONG_CO_DINH_M,
                u_so_m=max(xe['u_so_m'], xe['u_so_tho_m']) if num_ok else None,
                u_so_chuoi={c: dict(u_so_m=v['u_so_m'], u_so_tho_m=v['u_so_tho_m'], nguong_m=v['nguong_m'],
                                    nguong_tho_m=v['nguong_tho_m']) for c, v in chains.items()} if num_ok else None,
                tap_neo={c: v.get('tap_tinh', []) for c, v in chains.items()}, clip_h=clip_h, luot=luot,
                luot_thieu_kiem=thieu_kiem, mau_thuan=mau, model=kc.get('model'), nhom=kc.get('nhom'), sha256=sha,
                vai_tro_clip_h={n: r for n, r in (('H_a', 'dau1'), ('H_b', 'dau2'), ('H_m', 'giua'), ('H_z', 'cuoi'))
                                if n in roles},                                   # F18 (GATE GC6-4); segment: no giua
                doan=kc.get('doan'),
                h1_ho_tro=False, version_cong_cu=VERSION,
                dung_cho=('buoi ke thua U: ket_qua != dat -> moi luot ca buoi chua ket luan (J v6 §5); buoi C8: '
                          'ket_qua = dat la dieu kien khoa C8 (J v6 §5 J3-5), khong la ket qua luot'),
                gioi_han=['(a) u_so do o hai clip H dau buoi; dung cho luc chay la gia dinh',
                          '(b) u_so = tong cac cuc dai thanh phan thuc nghiem (tinh + nhieu) theo thuat toan, khong '
                          'phai can chung minh, khong bao dam moi thoi diem',
                          '(c) sai chung cua moi moc khong lo ra', '(d) tin phan_tich, tung_khung da chay dung (kiem SHA)',
                          '(e) thu tu clip H / luot la khai bao cua nguoi chay (ke hoach), khong kiem bang gio quay',
                          '(f) kiem nguon nhat quan va cung dinh danh buoi, khong chung minh nguon thuoc dung buoi vat ly',
                          '(g) C33-12 (tu choi track thu, cham_lai 2 moi luot, lich_sha256, ket_luan) chua co: ban nay '
                          'khong dung cho buoi that (thiet ke v6 S8-3)']
                + ([DOAN_GIOI_HAN] if kc.get('doan') else [])
                + ([DOAN_KHONG_CUOI] if kc.get('doan') and 'H_z' not in roles else []))


def tien_de(a):
    if not re_date(a.ngay):
        raise KCError(f'--ngay {a.ngay!r}: YYYY-MM-DD (S6-10)')
    kc, sha_kc = load_json_sha(a.kiem_cheo)         # KCJ1-6: the sha of the bytes read
    if kc.get('version') != VERSION or kc.get('buoi') != a.session_id:
        raise KCError(f'{a.kiem_cheo}: not a {VERSION} result of session {a.session_id} (TD1-4)')
    us, sha_us = load_json_sha(a.u_so)
    tk, sha_tk = load_json_sha(a.tung_khung) if a.tung_khung else (None, None)
    if tk is not None and tk.get('buoi') != a.session_id:
        raise KCError(f'{a.tung_khung}: buoi {tk.get("buoi")!r} is not {a.session_id!r} (TD1-4)')
    out = tien_de_eval(kc, us, tk, sha_kc, sha_us, sha_tk)
    # TD3-2: every hashed source by its path, SHA from the evidence that made the result, re-hashed to check
    nguon = [dict(vai_tro='kiem_cheo', duong_dan=os.path.abspath(a.kiem_cheo), sha256=sha_kc),
             _rehash(a.u_so, kc['u_so_sha256'], 'u_so')]
    for n, s in sorted((kc.get('sessions') or {}).items()):
        nguon.append(_rehash(s['file'], s['sha256'], f'session_{n}'))
        nguon.append(_rehash(s.get('video'), s.get('video_sha256'), f'clip_{n}'))
    if tk is not None:
        nguon.append(dict(vai_tro='tung_khung', duong_dan=os.path.abspath(a.tung_khung), sha256=sha_tk))
        ev, runs, sh = read_plan_sha(tk['ke_hoach']['file'])                     # KCJ1-2: the verified content
        if sh != tk['ke_hoach']['sha256'] or ev != tk['ke_hoach'].get('su_kien') or runs != tk['ke_hoach'].get('luot'):
            raise KCError('ke_hoach: the file is not the plan tung_khung read (sha256 / events, KCJ1-2)')
        nguon.append(dict(vai_tro='ke_hoach', duong_dan=os.path.abspath(tk['ke_hoach']['file']), sha256=sh))
        for rid, v in sorted(tk['luot'].items()):
            dd = v.get('duong_dan') or {}
            nguon.append(_rehash(dd.get('track'), v['nguon']['track_sha256'], f'track_{rid}'))
            nguon.append(_rehash(dd.get('frames'), v['nguon']['frames_sha256'], f'frames_{rid}'))
            nguon.append(_rehash(dd.get('kiem_cheo_khung'), v['nguon']['kiem_cheo_khung_sha256'], f'kiem_cheo_khung_{rid}'))
    out.update(ngay=a.ngay, buoi=a.session_id, nguon=nguon)
    write_new(a.out, out)
    _say(f"tien_de_buoi {a.session_id}: {out['ket_qua']} ({', '.join(out['ly_do']) or 'khong ly do'}); "
          f"luot {len(out['luot'])}, thieu kiem {out['luot_thieu_kiem']}")


def re_date(s):
    try:
        return datetime.date.fromisoformat(str(s)).isoformat() == s
    except ValueError:
        return False


# ---------------------------------------------------------------- e_hieu_m (S7-17, F9; design v6 S8-5)
def doan_body_sha(d):
    """Content sha256 of an F5 file as c8_hc_a writes and checks it (sort_keys, without the key sha256)."""
    body = {k: v for k, v in d.items() if k != 'sha256'}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def iii_leg(fp_, rows, dets, ids_a, ids_b, side, kappa, plumb, to):
    """F9 (iii) (S7-17, KS7B-3 5; CODE 20/11): replay of the anchor sets REALLY used by the two windows of one 12b leg,
    on the real frames of the leg (dets.npz) through the scoring chain: per frame f, the tag pose through the scoring
    camera on its own anchors S_f (moc_dung) minus through the camera refitted on the anchors common to EVERY frame of
    both windows (the reference run, defined here: same start cam_H[xe], same trust flags); (iii) = |mean over b - mean
    over a| of that difference (position, heading). 0 when no anchor changes. Returns (m, psi_rad, reason, info)."""
    fr = list(ids_a) + list(ids_b)
    if not ids_a or not ids_b:
        return None, None, 'cua so khong co khung', None
    S = {}
    for f in fr:
        if f not in rows or f not in dets or cm.CAR_ID not in dets[f][1]:
            return None, None, f'khung {f} khong co trong frames.csv / dets.npz (hay khong thay ma noc)', None
        S[f] = _ids_of(rows[f].get('moc_dung'))
    ref = sorted(set.intersection(*[set(v) for v in S.values()]))
    if len(ref) < MIN_TRAIN:
        return None, None, f'tap neo chung {len(ref)} moc (< {MIN_TRAIN})', None
    e, dev = {}, 0.0
    try:
        for f in fr:
            d = dets[f][1]
            pS = cm.car_pose(fp_.full(S[f], d, f'khung {f}'), d[cm.CAR_ID], side, kappa, plumb, to)
            pR = cm.car_pose(fp_.full(ref, d, f'khung {f} (tap chung)'), d[cm.CAR_ID], side, kappa, plumb, to)
            tx, ty = rows[f].get('tag_x'), rows[f].get('tag_y')
            if tx not in ('', None) and ty not in ('', None):
                dev = max(dev, abs(pS['tag_x'] - float(tx)), abs(pS['tag_y'] - float(ty)))
            e[f] = np.array([pS['x'] - pR['x'], pS['y'] - pR['y'], float(cm.cp_wrap(pS['psi'] - pR['psi']))])
    except KCError as ex:
        return None, None, str(ex)[:200], None
    if not dev <= TOL_REPLAY_M:
        raise KCError(f'(iii): the replayed scoring pose differs from frames.csv by {dev:.3g} m (another session, '
                      'detections or anchors)')
    ea = np.mean([e[f] for f in ids_a], axis=0)
    eb = np.mean([e[f] for f in ids_b], axis=0)
    dd = eb - ea
    if not np.all(np.isfinite(dd)):
        return None, None, 'khong huu han', None
    info = dict(tap_chung=ref, n_khung_a=len(ids_a), n_khung_b=len(ids_b), lech_phat_lai_max_m=dev,
                doi_tap=any(list(v) != ref for v in S.values()))
    return float(math.hypot(dd[0], dd[1])), abs(float(cm.cp_wrap(dd[2]))), '', info


def e_hieu(a):
    kc = _load_json(a.kiem_cheo)
    if kc.get('version') != VERSION or kc.get('thu'):
        raise KCError(f'{a.kiem_cheo}: not a {VERSION} result (or a trial)')
    hs = kc.get('hang_so_s7') or {}
    d = _load_json(a.doan)
    if d.get('schema') != DOAN_SCHEMA:
        raise KCError(f'{a.doan}: schema {d.get("schema")!r} is not {DOAN_SCHEMA}')
    if d.get('sha256') != doan_body_sha(d):
        raise KCError(f'{a.doan}: content sha256 does not match (file edited)')
    cho = d.get('cho')
    if cho not in CHO or cho not in (hs.get('xuat_phat') or {}):
        raise KCError(f'{a.doan}: cho {cho!r} has no start region in the kiem_cheo result (--xuat-phat)')
    tv = ((kc.get('P1_tuong_doi') or {}).get('theo_vung') or {}).get(cho)
    if tv is None:
        raise KCError('kiem_cheo result without P1_tuong_doi for this place')
    part_i = tv.get('trong_vung_max_m')
    cfg = d.get('car_pose_cfg') or {}
    side, h_m = cfg.get('side_m'), cfg.get('h_m')
    to = cfg.get('tag_offset') or (0.0, 0.0, 0.0)
    plumb = hs.get('plumb')
    # (ii) cameras of the session of the F5 file (the scoring session): the pairs of the 12b CORRIDOR marks of the
    # place, L <= L_max. (ii) is the error of the difference between the two windows of a 12b leg, both on the 12b
    # segment, so its marks are those of A3_tuong_doi.duong_12b (CODE 20/11, HCV3-2), not of the start disc
    ses = next((x for x in d.get('nguon') or [] if x.get('vai_tro') == 'session'), None)
    cams, why_ii, reg = [], [], None
    why_iii, dets_iii, rows_iii, fp_iii, set_iii = [], None, None, None, None
    duong = hs.get('DUONG_12B')
    if ses is None or not ses.get('duong_dan'):
        why_ii.append('F5 khong co nguon session')
    elif side is None or h_m is None:
        why_ii.append('car_pose_cfg thieu side_m / h_m')
    elif not isinstance(duong, dict) or cho not in duong or not isinstance(hs.get('R_XP_DUONG_M'), (int, float)):
        why_ii.append('kiem_cheo khong co hanh lang 12b (hang_so_s7.DUONG_12B, R_XP_DUONG_M)')
    else:
        if ses.get('sha256') != d.get('session_sha256'):
            raise KCError('F5: nguon session sha256 is not session_sha256 (KCJ1-8)')
        if d.get('session_sha256') != ((kc.get('sessions') or {}).get('H_a') or {}).get('sha256'):
            raise KCError('F5: its session is not the H_a session of the kiem_cheo result (another camera segment: '
                          'run e_hieu with the kiem_cheo of the segment of the 12b clip)')
        _rehash(ses['duong_dan'], d.get('session_sha256'), 'session_F5')
        # KCJ1-8: the leg ends come from frames / track of the SAME session: chain F5 -> track -> session, frames
        src = {x.get('vai_tro'): x for x in d.get('nguon') or []}
        if 'track' not in src or 'frames' not in src:
            raise KCError('F5: nguon without track / frames (the leg ends cannot be tied to the session, KCJ1-8)')
        for role, key in (('track', 'track_sha256'), ('frames', 'frames_sha256')):
            if src[role].get('sha256') != d.get(key):
                raise KCError(f'F5: nguon {role} sha256 is not {key} (KCJ1-8)')
        _rehash(src['frames'].get('duong_dan'), d.get('frames_sha256'), 'frames_F5')
        tr, tr_sha = load_json_sha(src['track']['duong_dan'])
        if tr_sha != d.get('track_sha256'):
            raise KCError('F5: track file changed (sha256, KCJ1-8)')
        if tr.get('session_sha256') != d.get('session_sha256') or tr.get('frames_sha256') != d.get('frames_sha256'):
            raise KCError('F5: the track was not made from this session / these frames (KCJ1-8)')
        s, s_sha = load_json_sha(ses['duong_dan'])
        if s_sha != d.get('session_sha256'):
            raise KCError('F5: session file changed (sha256, KCJ1-8)')
        img, mk, w, h, zc = clip_from_session(s, kc['model'])
        pc = PairCams(img, mk, kc['model'], w, h, zc)
        reg = duong_regions(mk, {cho: hs['xuat_phat'][cho]}, float(hs['R_XP_DUONG_M']),
                            {cho: tuple(duong[cho])})[cho]['moc']
        L_max = hs.get('L_MAX_M', L_MAX_M)
        for x in range(len(reg)):
            for y in range(x + 1, len(reg)):
                s_, j_ = reg[x], reg[y]
                if math.dist(mk[s_][:2], mk[j_][:2]) <= L_max:
                    cams.append(((s_, j_), pc((s_, j_))))
        full = pc()
        kappa = float(h_m) / zc
        if not cams:
            why_ii.append('vung khong co cap trong L_max')
        # (iii): the detections of the 12b clip (F5 nguon dets, dets_sha256, dets_fp) and the frames.csv rows
        if 'dets' not in src or not src['dets'].get('duong_dan'):
            why_iii.append('F5 khong co nguon dets (dets.npz cua CAM)')
        elif d.get('dets_sha256') is None or d.get('dets_fp') is None:
            why_iii.append('F5 khong co dets_sha256 / dets_fp')
        else:
            if src['dets'].get('sha256') not in (None, d['dets_sha256']):
                raise KCError('F5: nguon dets sha256 is not dets_sha256')
            dets_iii, _, _ = load_dets_checked(src['dets']['duong_dan'], d['dets_sha256'], d['dets_fp'],
                                               tr.get('video_sha256'), 'dets_F5')
            rows_iii = {}
            for r_ in _read_csv(src['frames']['duong_dan']):
                rows_iii[int(r_['idx'])] = r_
            fp_iii = FramePoses(s, pc)
            # the scoring chain of the track: session kappa / plumb, the track's tag offset (replay checked)
            set_iii = (float(s['kappa']), np.asarray(s['plumb_xy_m'], float),
                       tuple(tr.get('tag_offset') or s.get('tag_offset') or (0.0, 0.0, 0.0)))
    if not d.get('doan'):
        raise KCError('F5 without legs: nothing to write (KCJ1-7)')
    legs = []
    for g in d.get('doan') or []:
        why = []
        if g.get('trang_thai') != 'hop_le':
            why.append(f"doan F5 {g.get('trang_thai')}")
        if d.get('trang_thai') == 'chan_doan':
            why.append('F5 chan_doan')
        ca, cb = g.get('cua_so_a') or {}, g.get('cua_so_b') or {}
        ii_m = ii_psi = None
        if why_ii:
            why += why_ii
        elif ca.get('p_m') is None or cb.get('p_m') is None or ca.get('psi_rad') is None or cb.get('psi_rad') is None:
            why.append('thieu p_m / psi_rad: chi (i)')
        else:
            ii_m = ii_psi = 0.0
            for _, cam in cams:
                for _, dx, dy, dpsi in chuyen_pair(full, cam, ca['p_m'], cb['p_m'], ca['psi_rad'], cb['psi_rad'],
                                                    float(side), float(h_m), kappa, plumb, tuple(to)):
                    ii_m, ii_psi = max(ii_m, math.hypot(dx, dy)), max(ii_psi, abs(dpsi))
        iii_m = iii_psi = iii_info = None
        if fp_iii is None:
            why += why_iii or ['(iii) khong tinh: chuoi session / track / frames cua F5 chua dung (xem ly do (ii))']
        else:
            iii_m, iii_psi, w3, iii_info = iii_leg(fp_iii, rows_iii, dets_iii, list(ca.get('khung_id') or []),
                                                   list(cb.get('khung_id') or []), float(s['car_side_m']), *set_iii)
            if w3:
                why.append(f'(iii) {w3}')
        vals = [v for v in (part_i, ii_m, iii_m) if v is not None]
        if not vals:
            raise KCError('e_hieu_m: a leg with neither (i) nor (ii) (start region without in-region pairs)')
        psis = [v for v in (ii_psi, iii_psi) if v is not None]
        rec = dict(i=g.get('i'), e_hieu_psi_rad=max(psis) if psis else None, phan_i_m=part_i, phan_ii_m=ii_m,
                   phan_iii_m=iii_m, phan_iii_psi_rad=iii_psi, phan_iii=iii_info)
        # KCJ1-7: a diagnostic number is NOT written under e_hieu_m (the key the reader turns into a usable value);
        # e_hieu_m appears only when (i), (ii), (iii) are all there and neither the leg nor F5 is chan_doan
        if not why and part_i is not None and ii_m is not None and iii_m is not None:
            rec.update(e_hieu_m=max(vals), trang_thai='dat', ly_do=[])
        else:
            rec.update(e_hieu_m_chan_doan=max(vals), trang_thai='chan_doan', ly_do=why or ['thieu mot phan'])
        legs.append(rec)
    all_dat = all(x['trang_thai'] == 'dat' for x in legs)
    out = dict(schema=E_HIEU_SCHEMA, cho=cho, doan_12b_sha256=d['sha256'], bao_q_xoay=False,
               mien_ii=dict(loai='duong_12b', moc=reg),
               trang_thai='dat' if all_dat else 'chan_doan', doan=legs, hang_so_s7=hs,
               loai='chi so thuc nghiem truoc le (S7-17): max (i), (ii), (iii) theo doan',
               nguon=[dict(vai_tro='kiem_cheo', duong_dan=os.path.abspath(a.kiem_cheo), sha256=sha256_file(a.kiem_cheo)),
                      dict(vai_tro='doan_12b_v1', duong_dan=os.path.abspath(a.doan), sha256=sha256_file(a.doan))]
               + ([dict(vai_tro='session_ii', duong_dan=ses['duong_dan'], sha256=d.get('session_sha256'))]
                  if ses and not why_ii else []),
               gioi_han=['(i) cap moc dai hon doan: khong thay meo o thang doan ngan hon khoang cach moc',
                         '(ii) chi trong ho mo hinh; tam ma dat tai p_m, lech q bo qua; cap moc cua hanh lang 12b '
                         '(A3_tuong_doi.duong_12b), (i) van la cap trong dia xuat phat (P1_tuong_doi.theo_vung)',
                         '(iii) phat lai tap neo that cua hai cua so tren khung that (dets.npz) qua chuoi cham, so '
                         'voi tap neo chung cua moi khung hai cua so (doi chung dinh nghia o iii_leg); chi thay phan '
                         'do doi tap neo, khong thay sai chung cua ca tap'],
               commit_hil=commit_hil())
    write_new(a.out, out)
    top = max(x.get('e_hieu_m', x.get('e_hieu_m_chan_doan')) for x in legs)
    _say(f"e_hieu {cho}: {len(legs)} doan, max {top * 1000:.2f} mm, "
         f"{sum(x['trang_thai'] == 'dat' for x in legs)} doan dat; trang_thai {out['trang_thai']}")


NEO_LUOT_SCHEMA = 'c7_neo_luot_v1'            # P_neo / A_neo (S7-13 KS7B-3, S7-14.5; CODE 23/11 T12 chot section 5)


def neo_run(fp_, rows, dets, ids_w, ids_post, side, kappa, plumb, to):
    """P_neo / A_neo of one C8 run (S7-14.5; CODE 23/11: the bound of hang_m covers EVERY frame t of the run): the
    reference window W = pre. For every valid frame t of the run (frames.csv valid with a tag pose, from the first W
    frame to the last post frame), the reference anchor set R_t = (anchors common to every W frame) & S_t; the replay
    deviation of a frame f on R_t is e_f = pose on its own anchors S_f - pose on R_t (scoring chain, same as iii_leg);
    value_t = |e_t - mean over W of e_f| (position, heading). A frame whose replay cannot be made (no detection of the
    car tag, R_t below MIN_TRAIN, refit not trusted) is not valid (S7-14.5) and counted. Returns (dict, reason):
    P_neo_m, A_neo_rad = max over the valid frames, the frames of the max, counts."""
    if not ids_w or not ids_post:
        return None, 'cua so khong co khung'
    for f in list(ids_w) + list(ids_post):
        if f not in rows:
            return None, f'khung {f} cua cua so khong co trong frames.csv'
    S = {}

    def anchors(f):
        if f not in S:
            S[f] = _ids_of(rows[f].get('moc_dung'))
        return S[f]

    for f in ids_w:
        if f not in dets or cm.CAR_ID not in dets[f][1]:
            return None, f'khung W {f} khong co trong dets.npz (hay khong thay ma noc)'
    W = set.intersection(*[set(anchors(f)) for f in ids_w])

    def dev_on(f, ref):
        d = dets[f][1]
        pS = cm.car_pose(fp_.full(anchors(f), d, f'khung {f}'), d[cm.CAR_ID], side, kappa, plumb, to)
        pR = cm.car_pose(fp_.full(ref, d, f'khung {f} (tap chung)'), d[cm.CAR_ID], side, kappa, plumb, to)
        return pS, np.array([pS['x'] - pR['x'], pS['y'] - pR['y'], float(cm.cp_wrap(pS['psi'] - pR['psi']))])

    lo, hi = min(ids_w), max(ids_post)
    run = [f for f in sorted(rows) if lo <= f <= hi and _true(rows[f].get('valid'))
           and rows[f].get('tag_x') not in ('', None)]
    cache, bad, dev = {}, {}, 0.0
    best_p, best_a, f_p, f_a, n_ok = 0.0, 0.0, None, None, 0
    for t in run:
        if t not in dets or cm.CAR_ID not in dets[t][1]:
            bad[t] = 'khong co dets / ma noc'
            continue
        ref = tuple(sorted(W & set(anchors(t))))
        if len(ref) < MIN_TRAIN:
            bad[t] = f'tap neo chung {len(ref)} moc (< {MIN_TRAIN})'
            continue
        if ref not in cache:
            try:
                cache[ref] = np.mean([dev_on(f, list(ref))[1] for f in ids_w], axis=0)
            except KCError as ex:
                cache[ref] = str(ex)[:160]
        m = cache[ref]
        if isinstance(m, str):
            bad[t] = f'W tren tap chung: {m}'
            continue
        try:
            pS, et = dev_on(t, list(ref))
        except KCError as ex:
            bad[t] = str(ex)[:160]
            continue
        dev = max(dev, abs(pS['tag_x'] - float(rows[t]['tag_x'])), abs(pS['tag_y'] - float(rows[t]['tag_y'])))
        dd = et - m
        p, ang = float(math.hypot(dd[0], dd[1])), abs(float(cm.cp_wrap(dd[2])))
        if not (math.isfinite(p) and math.isfinite(ang)):
            bad[t] = 'khong huu han'
            continue
        n_ok += 1
        if f_p is None or p > best_p:
            best_p, f_p = p, t
        if f_a is None or ang > best_a:
            best_a, f_a = ang, t
    if not dev <= TOL_REPLAY_M:
        raise KCError(f'neo: the replayed scoring pose differs from frames.csv by {dev:.3g} m (another session, '
                      'detections or anchors)')
    if not n_ok:
        return None, 'khong khung nao phat lai duoc'
    return dict(P_neo_m=best_p, A_neo_rad=best_a, khung_max_P=f_p, khung_max_A=f_a, n_khung=len(run),
                n_khung_hop_le=n_ok, n_khung_khong_hop_le=len(bad),
                khong_hop_le_mau=[dict(khung=k, ly_do=v) for k, v in sorted(bad.items())[:20]],
                n_tap_chung=len(cache), tap_W=sorted(W), lech_phat_lai_max_m=dev,
                doi_tap=any(set(anchors(t)) != W for t in run if t not in bad)), ''


def neo_luot(a):
    """P_neo / A_neo (S7-13 KS7B-3, S7-14.5; CODE 23/11 T12: names c7_neo_luot_v1, the bound over EVERY frame of the
    run): for every C8 run with a pre and a post window in the merged F2 (cua_so_v1.jsonl, khung_ma_hop_le = the frames
    the scorer averages), replay of the anchor sets REALLY used by each frame (frames.csv moc_dung) through the scoring
    chain of the final track (track.json, frames.csv, dets.npz of CODE_CAM, session of the window), neo_run: P_neo_m /
    A_neo_rad = max over the valid frames t of the run of |replay deviation at t - mean replay deviation over W = pre|
    (position / heading), before margin; info.hai_dau keeps post - pre (iii_leg). A run whose replay cannot be made:
    chua_ket_luan + reason. theo_cho: max over the runs of each place; not dat when a run of the place is not."""
    with open(a.cua_so, 'rb') as f:
        cb = f.read()
    cs_sha = hashlib.sha256(cb).hexdigest()                     # KCF1-7: hash of the bytes parsed
    try:
        rows_cs = [json.loads(x) for x in cb.decode('utf-8').splitlines() if x.strip()]
    except ValueError as e:
        raise KCError(f'cua_so_v1: {e}')
    win = {}
    for r in rows_cs:
        if isinstance(r, dict) and r.get('loai') in ('pre', 'post'):
            lid = r.get('luot_id')
            if not isinstance(lid, str) or not lid:
                raise KCError('cua_so_v1: a pre/post window without luot_id')
            if r['loai'] in win.setdefault(lid, {}):
                raise KCError(f'cua_so_v1: two {r["loai"]} windows of {lid}')
            win[lid][r['loai']] = r
    if not win:
        raise KCError('cua_so_v1: no pre/post window (the merged F2 of the C8 runs is needed)')
    tracks = {}                                                  # frames sha -> (dir, track, track sha)
    for d in a.track:
        tr, tr_sha = load_json_sha(os.path.join(d, 'track.json'))
        fs = _hex(tr.get('frames_sha256'), f'{d} track.frames_sha256')
        if sha256_file(os.path.join(d, 'frames.csv')) != fs:
            raise KCError(f'{d}: frames.csv is not the one of track.json')
        if fs in tracks:
            raise KCError(f'--track {d}: same frames.csv as {tracks[fs][0]}')
        tracks[fs] = (d, tr, tr_sha)
    sessions = {}
    for sp_ in a.session:
        s_, sha_ = load_json_sha(sp_)
        if sha_ in sessions:
            raise KCError(f'--session {sp_} given twice')
        sessions[sha_] = sp_
    out_l = {}
    for lid in sorted(win):
        w = win[lid]
        rec = dict(cho=next(iter(w.values())).get('cho'), P_neo_m=None, A_neo_rad=None, n_pre=None, n_post=None,
                   info=None)          # cho of the window there is: a run without post still counts at its place
        why = []
        if set(w) != {'pre', 'post'}:
            why.append(f'thieu cua so {sorted({"pre", "post"} - set(w))}')
        else:
            pre, post = w['pre'], w['post']
            rec['cho'] = pre.get('cho')
            for k in ('cho', 'clip_sha256', 'session_sha256', 'frames_sha256'):
                if pre.get(k) != post.get(k):
                    raise KCError(f'cua_so_v1 {lid}: pre and post with another {k}')
            ids = {}
            for nm, r in (('pre', pre), ('post', post)):
                # khung_ma_hop_le: the frame ids the scorer averages (ints, c7_score cua-so); else khung [{id, pts_s}]
                kh = r.get('khung_ma_hop_le') if r.get('khung_ma_hop_le') is not None else r.get('khung')
                kh = [x.get('id') if isinstance(x, dict) else x for x in kh] if isinstance(kh, list) else None
                if not kh or not all(isinstance(x, int) and not isinstance(x, bool) for x in kh) \
                        or len(set(kh)) != len(kh):
                    raise KCError(f'cua_so_v1 {lid} {nm}: khung_ma_hop_le / khung empty, malformed or repeated')
                ids[nm] = kh
            rec.update(n_pre=len(ids['pre']), n_post=len(ids['post']))
            fs = pre.get('frames_sha256')
            if fs not in tracks:
                why.append('khong co --track co frames.csv cua cua so')
            elif pre.get('session_sha256') not in sessions:
                why.append('khong co --session cua cua so')
            else:
                d, tr, tr_sha = tracks[fs]
                if tr.get('session_sha256') != pre['session_sha256'] or tr.get('video_sha256') != pre['clip_sha256']:
                    raise KCError(f'{lid}: the track of the window was made from another session / clip')
                s, ses_sha, mk, pc, full, kappa, C = _session_ctx(sessions[pre['session_sha256']],
                                                                  tr.get('session_sha256'), a.model)
                dp_ = tr.get('dets_file') or os.path.join(d, 'dets.npz')
                dets, dets_sha, _ = load_dets_checked(dp_, _hex(tr.get('dets_sha256'), f'{lid} track.dets_sha256'),
                                                      None, tr.get('video_sha256'), f'{lid} dets.npz')
                rows = {int(r_['idx']): r_ for r_ in _read_csv(os.path.join(d, 'frames.csv'))}
                to = tuple(tr.get('tag_offset') or s.get('tag_offset') or (0.0, 0.0, 0.0))
                cfg = (float(s['car_side_m']), float(s['kappa']), np.asarray(s['plumb_xy_m'], float), to)
                fp_ = FramePoses(s, pc)
                res, w3 = neo_run(fp_, rows, dets, ids['pre'], ids['post'], *cfg)
                if w3:
                    why.append(w3)
                else:
                    # kept in info (CODE 23/11): the two-end number post - pre, as e_hieu (iii)
                    m2, psi2, w2, i2 = iii_leg(fp_, rows, dets, ids['pre'], ids['post'], *cfg)
                    res['hai_dau'] = dict(P_m=m2, A_rad=psi2, ly_do=w2 or None, tap_chung=(i2 or {}).get('tap_chung'))
                    rec.update(P_neo_m=res.pop('P_neo_m'), A_neo_rad=res.pop('A_neo_rad'), info=res)
                rec['nguon'] = dict(track=dict(duong_dan=os.path.abspath(os.path.join(d, 'track.json')),
                                               sha256=tr_sha),
                                    frames=dict(duong_dan=os.path.abspath(os.path.join(d, 'frames.csv')), sha256=fs),
                                    dets=dict(duong_dan=os.path.abspath(dp_), sha256=dets_sha),
                                    session=dict(duong_dan=os.path.abspath(sessions[ses_sha]), sha256=ses_sha))
        rec.update(trang_thai='dat' if not why else 'chua_ket_luan', ly_do=why)
        out_l[lid] = rec
    theo_cho = {}                     # CODE 23/11: {P_neo_m, A_neo_rad, n_luot, trang_thai, ly_do}; C8B reads goc
    for lid, r in sorted(out_l.items()):
        c = theo_cho.setdefault(r['cho'], dict(P_neo_m=None, A_neo_rad=None, n_luot=0, luot_max_P=None,
                                               luot_max_A=None, ly_do=[]))
        if r['trang_thai'] != 'dat':
            c['ly_do'].append(f'{lid} chua_ket_luan')
            continue
        if c['P_neo_m'] is None or r['P_neo_m'] > c['P_neo_m']:
            c.update(P_neo_m=r['P_neo_m'], luot_max_P=lid)
        if c['A_neo_rad'] is None or r['A_neo_rad'] > c['A_neo_rad']:
            c.update(A_neo_rad=r['A_neo_rad'], luot_max_A=lid)
        c['n_luot'] += 1
    for c in theo_cho.values():
        c['trang_thai'] = 'dat' if not c['ly_do'] and c['n_luot'] else 'chua_ket_luan'
    out = dict(schema=NEO_LUOT_SCHEMA, model=a.model, luot=out_l, theo_cho=theo_cho,
               luot_chua_ket_luan=sorted(k for k, v in out_l.items() if v['trang_thai'] != 'dat'),
               nguon=dict(cua_so=dict(duong_dan=os.path.abspath(a.cua_so), sha256=cs_sha)),
               loai='chi so thuc nghiem truoc le (S7-13 KS7B-3, S7-14.5): max tren moi khung hop le cua luot '
                    '|lech phat lai khung t - trung binh lech phat lai tren W (pre)|',
               gioi_han=['phat lai tap neo that tung khung (frames.csv moc_dung) tren khung that (dets.npz) qua chuoi '
                         'cham, so voi tap tham chieu R_t = (neo chung moi khung W) giao S_t (doi chung dinh nghia o '
                         'neo_run); chi thay phan do doi tap neo, khong thay sai chung cua ca tap',
                         'khung luot: hop le trong frames.csv, co tu the ma, tu khung W dau toi khung post cuoi; khung '
                         'phat lai khong duoc (thieu dets, R_t < MIN_TRAIN, refit khong tin) khong hop le, dem rieng',
                         'dem trung mot phan voi P7 / A5 (lech khung theo thoi gian): phia than trong (CODE 23/11)',
                         'info.hai_dau: so hai dau post - pre nhu e_hieu (iii), chi de doi chieu'],
               commit_hil=commit_hil())
    write_new(a.out, out)
    _say(f"neo_luot: {len(out_l)} luot, {sum(v['trang_thai'] == 'dat' for v in out_l.values())} dat; theo_cho "
         + ', '.join(f"{c} {v['trang_thai']}: P "
                     f"{'-' if v['P_neo_m'] is None else round(v['P_neo_m'] * 1000, 3)} mm, A "
                     f"{'-' if v['A_neo_rad'] is None else round(math.degrees(v['A_neo_rad']), 4)} deg"
                     for c, v in sorted(theo_cho.items()))
         + f"; chua ket luan {out['luot_chua_ket_luan']}")


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
        if name == 'moc_dich':
            raise KCError('moc_dich comes from --moc-dich (U_moc of CODE_CAM) and --tien-de, not from --phan')
        blk = _load_json(path)
        if not isinstance(blk, dict) or blk.get('buoi') != a.buoi:
            raise KCError(f'--phan {name}: buoi {blk.get("buoi") if isinstance(blk, dict) else None!r} is not '
                          f'{a.buoi!r} (KC1-10)')
        parts[name] = (blk, path, sha256_file(path))
    parts['moc_dich'] = moc_dich_block(a, kc, tk)
    out = build_cam_u(own, parts, a.ghi_chu)
    out['ghi_chu'] = (out.get('ghi_chu', '') + f' buoi={a.buoi}').strip()
    write_new(a.out, out)
    _say(f'{a.out}: {CAM_U_VERSION}, buoi {a.buoi}')


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
    p.add_argument('--xuat-phat', dest='xuat_phat', action='append',
                   help='goc|K1|K2=X,Y start point of the place (floor m, from the plan; S7, design v6 S8-4)')
    p.add_argument('--plumb', help='X,Y plumb point of car_pose (the scoring one, L125), needed with --xuat-phat')
    p.add_argument('--khoa-s7', dest='khoa_s7', help='c7_khoa_s7_v1 manifest committed before the run (KCJ1-14)')
    p.add_argument('--doan', help='camera segment name (Liam 20/11 "Theo doan, bo H giua"): clips H_a, H_b, H_z of the '
                   'segment, no H_m')
    p.add_argument('--khong-H-cuoi', dest='khong_H_cuoi', action='store_true',
                   help='with --doan: the segment has no clip H after its last run (no drift check; limit)')
    p.add_argument('--out', required=True)
    x = sub.add_parser('xuat')
    x.add_argument('--buoi', required=True)
    x.add_argument('--kiem-cheo', dest='kiem_cheo', required=True)
    x.add_argument('--tung-khung', dest='tung_khung', required=True)
    x.add_argument('--moc-dich', dest='moc_dich', required=True, help='U_moc block of CODE_CAM (c8b_u v3 moc_dich)')
    x.add_argument('--tien-de', dest='tien_de', required=True, help='tien_de_buoi.json of this session')
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
    d = sub.add_parser('tien_de')
    d.add_argument('--session-id', dest='session_id', required=True, help='session_id of rq3_log = buoi of the results')
    d.add_argument('--ngay', required=True, help='YYYY-MM-DD of the session (S6-10)')
    d.add_argument('--kiem-cheo', dest='kiem_cheo', required=True)
    d.add_argument('--u-so', dest='u_so', required=True)
    d.add_argument('--tung-khung', dest='tung_khung')
    d.add_argument('--out', required=True)
    e = sub.add_parser('e_hieu')
    e.add_argument('--kiem-cheo', dest='kiem_cheo', required=True)
    e.add_argument('--doan', required=True, help='doan_12b_v1.json of CODE_HIEUCHUAN (F5)')
    e.add_argument('--out', required=True)
    m = sub.add_parser('meo_E', help='F15 (a) meo_ma_ao_E of one place')
    m.add_argument('--model', required=True)
    m.add_argument('--cho', required=True)
    m.add_argument('--session', required=True, help='scoring session of the E clip (sha = track.session_sha256)')
    m.add_argument('--track', required=True, help='E track dir (track.json, frames.csv)')
    m.add_argument('--cua-so', dest='cua_so', required=True, help='cua_so_v1.jsonl with E_dau, E_cuoi')
    m.add_argument('--r-xp', dest='r_xp', type=float)
    m.add_argument('--a6', help='c8b_a6_v2 of the plate: h_m must be the session tag height (KCF1-6)')
    m.add_argument('--dets', help='dets.npz of the E clip (default: track.json dets_file; sha = track.dets_sha256)')
    m.add_argument('--L-max', dest='L_max', type=float, default=L_MAX_M, help='pair length of the per-case A3')
    m.add_argument('--out', required=True)
    b = sub.add_parser('meo_bang', help='F15 (b) meo_dau_chan_bang from lan_lap.json')
    b.add_argument('--model', required=True)
    b.add_argument('--lan-lap', dest='lan_lap', required=True)
    b.add_argument('--r-xp', dest='r_xp', type=float)
    b.add_argument('--session', action='append',
                   help='session(s) of the F8 items, repeated (each item uses the one of its session_sha256; KCF1-2)')
    b.add_argument('--dets', action='append', required=True,
                   help='dets.npz of every clip of the anchor frames (run ends, calibration windows), repeated')
    b.add_argument('--out', required=True)
    n = sub.add_parser('neo_luot', help='P_neo / A_neo of every C8 run (pre -> post anchor-set replay)')
    n.add_argument('--model', required=True)
    n.add_argument('--cua-so', dest='cua_so', required=True, help='merged F2 cua_so_v1.jsonl with pre/post windows')
    n.add_argument('--track', action='append', required=True,
                   help='final track dir of a run (track.json, frames.csv; dets_file of track.json), repeated')
    n.add_argument('--session', action='append', required=True, help='scoring session(s) of the windows, repeated')
    n.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        {'phan_tich': phan_tich, 'xuat': xuat, 'kiem_h': kiem_h_cli, 'tung_khung': tung_khung, 'tien_de': tien_de,
         'e_hieu': e_hieu, 'meo_E': meo_E, 'meo_bang': meo_bang, 'neo_luot': neo_luot}[a.cmd](a)
    except (KCError, OSError, ValueError, KeyError) as e:
        print(f'loi: {e}')
        print('#ket_qua,ma=2')
        return 2
    try:                                              # KCJ1-6: a closed stdout after publication keeps code 0
        print('#ket_qua,ma=0')
        sys.stdout.flush()
    except (OSError, ValueError):
        try:                                          # no second flush attempt at interpreter exit (exit code 120)
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
