#!/usr/bin/env python3
"""
c7_measure.py: camera measurement chain of C7, video -> car pose on the floor, frame by frame.

L83 (Liam 11/11 CODE local_7787e22f, "duyet" all recommendations of the Fable C7 design, review
CLAUDE_THESIS 03_CODE/docs/review/2026-11-11_Tham_dinh_Fable_C7_va_bom_loi.md); Liam 11/11 CODE
local_5e745270: "oke viet luon di". Rewritten to the repo standard from the Fable trial code
(03_CODE/docs/review/raw/2026-11-11_Fable_C7/c7lib.py: refine_edges, fit_camera, car_meas), design
sections cited as "design 3.x" = 2026-11-11_Fable_C7_thiet_ke_do_v1.md. O1 sections = O1 v11.4.

Chain (design 0, 3.1-3.10):
  read      OpenCV FFmpeg, auto rotation OFF (parameters belong to the sensor pixels), frame time =
            presentation time CAP_PROP_POS_MSEC (not index / fps), grey 8 bit, no resize, no filter
  detect    ArUco DICT_4X4_50 with OpenCV corner refinement OFF, then corners by EDGE FITTING
            (refine_edges): 24 profiles per side of the black square, edge = centroid of the
            positive derivative (black -> white, outward), total least squares line, corner =
            intersection (design 3.2: cornerSubPix pulls blurred corners 0.2-0.5 px inward)
  calib     clip H (static, no car, floor marks ID 1-24 + check tags ID 25-33): mean refined corners,
            camera models B1 / B2 / KB2 of c7_clip_probe (pinhole, one focal length, centre fixed),
            Z_c from the tape as a prior (sigma 20 mm), two passes (second with sigma = the real
            residual, design 3.3), model by 4-fold cross-validation over the marks (simplest within
            15 % of the best); check tags NOT in the fit, their error reported (design 4.2 (a))
  per frame the camera POSE (6 dof) is refitted to the floor marks seen in that frame, intrinsics and
            distortion from clip H (design 3.6, vibration compensation); shift_px = length of the
            median VECTOR of the mark-centre shifts from clip H (O1 3.5 'dich_moc'; CS1-13, Codex 16/11;
            shift_mag_px = median of the lengths, reported; frames.csv before 17/11 holds only the
            median of the lengths in shift_px), rms_px = reprojection RMS of the marks after the refit
  car       the 4 corners of the car tag (ID 0) -> floor plane z = 0 with the frame camera -> parallax
            P = P' + kappa (C - P'), kappa = h / Z_c from the tape, C = PLUMB POINT measured with a
            plumb line (not the camera centre of the fit, design 3.4) -> rigid fit (Procrustes) to the
            square of the measured side: tag centre, heading, fit residual, apparent scale -> wheel
            centre = tag centre - R(psi) (dx, dy), heading - dpsi (tag offset, from the static test,
            design 3.5; 0 until measured, the session says which)
  gates     frame VALID (O1 3.3 (1), (2)) only if (1) at least 8 floor marks and rms_px <= rms of
            clip H + 0.5 px; (2a) residual of the car tag fit to a square WITH SCALE (the uniform
            shrink removed) <= 1.5 mm; (2b) apparent scale within 0.5 % of the median scale of the
            NEIGHBOUR frames (frames that passed (1) and (2a) within +-0.25 s, the frame itself excluded;
            fewer than 8 neighbours: the frame fails; O1 v11.30 3.3 (2)). Invalid frames are lost frames (no interpolation,
            design 3.7, 3.8). Design 2026-11-16_Thiet_ke_viec_b_bo_cham_camera_v1.md; Liam 16/11 CODE_CAM
            local_a806ae7e: Q1 "Bo co deu roi kiem (Recommended)", Q2 "Trung vi truot +-0,25 s
            (Recommended)" (trial 16/11: the edge fit sits about 1.2 px inside the tag on the real
            iPhone video, more when the car moves fast; a uniform shrink moves neither centre nor
            heading). The 20 mm jump from a quadratic prediction is NOT a gate any more (O1 v11.10
            RS2-1: a position-suspect frame keeps its value); c7_score.py marks those frames.
  speed     quadratic fit over +-6 valid frames on the real frame times (design 3.8), for the stop
            condition and the clock matching (C7b); NaN on invalid frames
All thresholds are [DE XUAT] and are locked at the measurement-system trial C8 (O1 3.3, 3.9 item 4); at C8
they come from the constants file (--hang-so), and c7_score refuses frames gated with other thresholds.
The run rules (gaps, tracker, completion, outcome) are in c7_score.py.

  python tools/c7_measure.py calib --video <clip H> --markers <csv...> --zc 2.60 --plumb X,Y \
      --tag-h 0.120 --out <dir>      -> <dir>/session.json, calib_marks.csv, report.txt
  python tools/c7_measure.py track --video <run clip> --session <dir>/session.json --out <dir2>
                                     -> <dir2>/frames.csv, track.json
  python tools/c7_measure.py regate --frames <old frames.csv> --session <dir>/session.json --out <dir3>
                                     -> the gates of this version on a frames.csv written before, no video:
                                        res_sim_m from res_m and scale (exact for a square: rigid residual^2
                                        = residual with scale^2 + ((1 - scale) side / sqrt 2)^2 at the same
                                        rotation), then apply_gates and the speeds again; a verdict that the
                                        rounding of the CSV (6 decimals) could flip fails as 'can_track_lai'
                                        (CS1-17). Not a substitute for track when the pose measurement changed.
  [--hang-so FILE]   track, regate: frame-gate thresholds from the 'cong_khung' block of the c7_score
                     constants file (CS1-15), written with its sha256 to track.json; else the module defaults
  [--tag-offset DX,DY,DPSI_DEG]  tag centre minus wheel centre in the body frame (m, m, deg) and tag
                                 heading minus body heading, from the static test (C8 clip T)
  33 marks (session of calib --model): track writes dets.npz, frames.csv (+ moc_thay, moc_dung, moc_neo_<chain>,
  kc_*_k, nen_doi, bat_dinh_moc, chua_ket_luan_xe), kiem_cheo_khung.csv, track.json; lich builds the session schedule
  (one fixed-point step; c7_kiem_cheo phan_tich --tap-moc between steps); track --lich --run --u-so --dets = final pass.
Last line: #ket_qua,ma=0 done / 2 error / 3 final 33-mark pass with frames or check rows not concluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7_clip_probe as cp                                               # noqa: E402
import c7_moc_dich as md                                                 # noqa: E402

CAR_ID = 0
# 33 marks, check by rotation, mark-shift rule (design 2026-11-18_Thiet_ke_33_moc_kiem_cheo_A3_v3.md + v4; a session
# calibrated with calib --model carries the block 'kiem_cheo' and track runs the chains on it) [DE XUAT, locked at C8]
KC_GROUPS = 4                # groups ids[k::4] in ID order (v3 2)
MIN_TRAIN = 4                # 4 marks = 16 points: the pose is determined (as before)
CAMZ_TOL_M = 0.2             # fit flag: camera height within this of the clip-H fit (v3 2)
COND_MAX = 1.0e4             # fit flag: condition number of the column-normalised pose Jacobian (v3 2; the H_a
                             # distribution is written to the session to lock it at C8)
NONLIN_EVERY = 30            # do_nhay: the box-vertex refit (v4 4) on every 30th valid frame, its worst ratio to the
                             # linear sensitivity scales every frame (cost: 4 refits per mark per frame otherwise)
CHAIN_XE = 'xe'
FLOOR_IDS = tuple(range(1, 25))     # floor marks of a session without 'floor_ids' (before 18/11)
CHECK_IDS = tuple(range(25, 34))    # print IDs of the check tags; since 18/11 taped for good as floor marks (vai_tro
                                    # moc_giua, c7_clip_probe.FLOOR_ROLES), then they enter the fit
HALF_FLOOR_PX = 6.0          # design 3.2: half width of the edge profiles, floor marks
HALF_CAR_PX = 9.0            #             car tag (room for blur up to about 6 px)
SG_HALF = 6                  # design 3.8: +-6 frames (0.2 s at 60 fps)
# frame gates, O1 3.3 v11.4 (O-2) [DE XUAT, locked at C8]
GATE_MIN_MARKS = 8
GATE_RMS_EXTRA_PX = 0.5
GATE_CAR_RES_M = 0.0015     # (2a) residual of the fit WITH scale
GATE_SCALE_REL = 0.005       # (2b) against the median of the neighbours
SCALE_WIN_S = 0.25           # (2b) neighbours: frames that passed (1), (2a) within +-0.25 s (O1 v11.30: by time, the
                             #      clip may be 60 or 120 fps; +-15 frames at 60 fps)
SCALE_MIN = 8                # (2b) fewer neighbours: the frame fails
GATE_KEYS = ('min_marks', 'rms_extra_px', 'car_res_sim_m', 'scale_rel', 'scale_win_s', 'scale_min')
DEFAULT_GATES = dict(min_marks=GATE_MIN_MARKS, rms_extra_px=GATE_RMS_EXTRA_PX, car_res_sim_m=GATE_CAR_RES_M,
                     scale_rel=GATE_SCALE_REL, scale_win_s=SCALE_WIN_S, scale_min=SCALE_MIN)
GATE_RANGES = dict(min_marks=(4, 24), rms_extra_px=(0.0, 5.0), car_res_sim_m=(0.0, 0.01), scale_rel=(0.0, 0.05),
                   scale_win_s=(0.05, 2.0), scale_min=(1, 240))
CSV_QUANT = 5e-7             # half a unit of the 6th decimal: rounding of res_m / scale in a frames.csv (CS1-17)


# ---------------------------------------------------------------- detection and edge refinement
def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 5, 45, 10
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cp.DICT), p)


def refine_edges(img, corners, half=6.0, skip=0.18, n_samp=24, iters=2):
    """Refined (4, 2) corners of one tag by fitting a line to each side of the black square, or None.

    Per side: n_samp profiles across the side (skip a share at both ends, away from the corners);
    the edge on a profile is the centroid of the positive derivative going outward (black -> white;
    values below 20 % of the profile's peak dropped); a total-least-squares line through the edge
    points, points beyond 2.5 sigma dropped once; corner k = intersection of sides k-1 and k.
    ArUco corners run clockwise in the image (top-left, top-right, bottom-right, bottom-left of the
    tag), so the outward normal of side p0 -> p1 is (d_y, -d_x). With a symmetric blur (lens, even
    motion) the centroid of the derivative sits on the edge at mid-exposure (design 3.2)."""
    c = np.asarray(corners, float).copy()
    f32 = img if img.dtype == np.float32 else img.astype(np.float32)
    ss = np.arange(-half, half + 0.25, 0.5)
    sm = (ss[:-1] + ss[1:]) / 2.0
    for _ in range(iters):
        lines = []
        for k in range(4):
            p0, p1 = c[k], c[(k + 1) % 4]
            d = p1 - p0
            length = float(np.hypot(*d))
            if length < 8.0:
                return None
            d = d / length
            nrm = np.array([d[1], -d[0]])
            fr = np.linspace(skip, 1.0 - skip, n_samp)
            base = p0[None, :] + fr[:, None] * length * d[None, :]
            px = (base[:, None, 0] + ss[None, :] * nrm[0]).astype(np.float32)
            py = (base[:, None, 1] + ss[None, :] * nrm[1]).astype(np.float32)
            prof = cv2.remap(f32, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g = np.clip(np.diff(prof, axis=1), 0.0, None)
            g = np.where(g >= 0.2 * g.max(axis=1, keepdims=True), g, 0.0)
            wsum = g.sum(axis=1)
            ok = wsum > 1e-6
            if ok.sum() < 6:
                return None
            sstar = (g[ok] * sm[None, :]).sum(axis=1) / wsum[ok]
            pts = base[ok] + sstar[:, None] * nrm[None, :]
            for _ in range(2):
                m = pts.mean(axis=0)
                _, _, vt = np.linalg.svd(pts - m)
                dirv, nv = vt[0], vt[1]
                r = (pts - m) @ nv
                keep = np.abs(r) <= max(2.5 * r.std(), 0.05)
                if keep.all() or keep.sum() < 6:
                    break
                pts = pts[keep]
            lines.append((m, dirv))
        new = np.zeros((4, 2))
        for k in range(4):
            (m1, d1), (m2, d2) = lines[k - 1], lines[k]
            a = np.column_stack([d1, -d2])
            if abs(np.linalg.det(a)) < 1e-9:
                return None
            tt = np.linalg.solve(a, m2 - m1)
            new[k] = m1 + tt[0] * d1
        if np.max(np.linalg.norm(new - c, axis=1)) > 4.0 + half:
            return None
        c = new
    return c


def detect_refined(det, gray):
    """{id: (4, 2) corners} with edge refinement; a tag the refinement rejects is dropped (its corners
    would carry the inward bias of an unrefined detection; counted by the caller as not seen)."""
    out = {}
    for i, c in cp.detect_markers(det, gray).items():           # near-square, unique IDs only
        side_px = float(np.mean(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)))
        half = HALF_CAR_PX if int(i) == CAR_ID else HALF_FLOOR_PX
        half = min(half, 0.75 * side_px / 6.0)               # at most 3/4 of the black border (6 cells)
        r = refine_edges(gray, c, half=half)
        if r is not None:
            out[int(i)] = r
    return out


# ---------------------------------------------------------------- video
def sha256(path):
    return cp.sha256(path)


def iter_frames(path, max_frames=None):
    """Yields (index, t_s from the PTS, grey frame). Auto rotation off (design 3.1)."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise OSError(f'OpenCV cannot open {path}')
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    k = 0
    try:
        while max_frames is None or k < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            yield k, t, g
            k += 1
    finally:
        cap.release()


# ---------------------------------------------------------------- camera
def cam_to_dict(cam):
    return dict(model=cam.model, f=cam.f, cx=cam.cx, cy=cam.cy, k=[float(v) for v in cam.k],
                rvec=[float(v) for v in cam.rvec], tvec=[float(v) for v in cam.tvec])


def cam_from_dict(d):
    return cp.Camera(d['model'], d['f'], d['cx'], d['cy'], d['k'], d['rvec'], d['tvec'])


def fit_two_pass(model, obj, img, w, h, zc):
    """c7_clip_probe.fit_camera, then again with sigma = the real residual per axis when the residual
    is well above the detection noise (design 3.3: taping errors swamp the Z_c prior otherwise)."""
    cam, rms = cp.fit_camera(model, obj, img, w, h, zc)
    if zc is not None and rms / math.sqrt(2.0) > 1.5 * 0.3:
        cam, rms = cp.fit_camera(model, obj, img, w, h, zc, sigma_px=rms / math.sqrt(2.0))
    return cam, rms


def refit_pose(cam, obj_xy, img):
    """Camera pose only (rvec, tvec), intrinsics and distortion of cam kept (design 3.6).
    Returns (camera, reprojection RMS px)."""
    obj = np.column_stack([obj_xy, np.zeros(len(obj_xy))])

    def make(x):
        return cp.Camera(cam.model, cam.f, cam.cx, cam.cy, cam.k, x[:3], x[3:])

    def res(x):
        return (make(x).project(obj) - img).ravel()

    sol = least_squares(res, np.concatenate([cam.rvec, cam.tvec]), loss='huber', f_scale=1.0,
                        x_scale='jac', max_nfev=200)
    c = make(sol.x)
    r = c.project(obj) - img
    return c, float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def refit_pose_checked(cam_h, obj_xy, img, zc_h=None, cond_max=COND_MAX, camz_tol=CAMZ_TOL_M):
    """Pose only, STARTED FROM cam_h (a clip-H camera, never a previous frame: v3 2 T33-8), with the trust flags of
    v3 2 (T33-3, T33-17) on the training marks only: success, cost not above the start value, |camera height - zc_h|
    <= camz_tol, condition number of the column-normalised Jacobian <= cond_max. obj_xy (4n, 2) floor corners, img
    (4n, 2) pixels. Returns (camera, rms px, ok, ly_do, cond); ly_do '' when ok, else khop_khong_tin / dieu_kien.
    zc_h None: the height of cam_h ITSELF (Codex C33-3: a reference from the camera with every mark would let the data of
    a held-out group into the flag of a check chain)."""
    if zc_h is None:
        zc_h = float(abs(cam_h.centre[2]))
    obj = np.column_stack([obj_xy, np.zeros(len(obj_xy))])

    def make(x):
        return cp.Camera(cam_h.model, cam_h.f, cam_h.cx, cam_h.cy, cam_h.k, x[:3], x[3:])

    def res(x):
        return (make(x).project(obj) - img).ravel()

    x0 = np.concatenate([cam_h.rvec, cam_h.tvec])
    cost0 = cp.huber_cost(res(x0))
    sol = least_squares(res, x0, loss='huber', f_scale=1.0, x_scale='jac', max_nfev=200)
    c = make(sol.x)
    r = c.project(obj) - img
    rms = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
    jac = np.asarray(sol.jac, float)
    nrm = np.linalg.norm(jac, axis=0)
    if np.all(nrm > 0) and np.all(np.isfinite(jac)):
        sv = np.linalg.svd(jac / nrm, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    else:
        cond = math.inf
    ly_do = ''
    if not (sol.success and float(sol.cost) <= cost0 * (1 + 1e-12) and np.all(np.isfinite(sol.x))
            and abs(abs(c.centre[2]) - zc_h) <= camz_tol):
        ly_do = 'khop_khong_tin'
    elif not cond <= cond_max:
        ly_do = 'dieu_kien'
    return c, rms, not ly_do, ly_do, cond


def mark_centre_floor(cam, corners):
    """Centre of a floor mark (v3 1, T33-15): the MEAN OF ITS 4 CORNERS back-projected to the floor z = 0 (the centre of
    the rigid 4-corner fit; right for a square on the floor when the camera is right). Used for every check, baseline,
    shift and report of the 33-mark chain; d59197a (and the sessions without 'kiem_cheo') back-projected the mean of
    the image corners instead (calibrate reports the difference once)."""
    return cam.pix2floor(np.asarray(corners, float).reshape(4, 2), 0.0).mean(axis=0)


def make_groups(ids, n=KC_GROUPS):
    """v3 2: G groups ids[k::G] in ID order (empty groups dropped)."""
    ids = sorted(int(i) for i in ids)
    return [ids[k::n] for k in range(n) if ids[k::n]]


def cam_key(excl):
    """Key of the clip-H camera fitted without the groups excl: 'xe' (none), 'k', 'a,b' (a < b)."""
    e = sorted(set(int(v) for v in excl))
    return CHAIN_XE if not e else ','.join(str(v) for v in e)


def chain_names(groups):
    return [CHAIN_XE] + [str(k) for k in range(len(groups))]


def comparator_excl(chain, t):
    """Groups left out of the comparator of a target in group t for the chain (v3 2): chain xe -> {t}; chain k ->
    {k, t} (t != k)."""
    return frozenset([t]) if chain == CHAIN_XE else frozenset([int(chain), t])


def session_floor_ids(session):
    """Floor-mark IDs of a session: its 'floor_ids' (written from 18/11, by vai_tro), else ID 1-24."""
    return tuple(int(i) for i in session.get('floor_ids', FLOOR_IDS))


def floor_obs(dets, marks, ids=FLOOR_IDS):
    """(obj (4n, 2) floor corners, img (4n, 2) pixels, ids used) of the floor marks in dets."""
    use = [i for i in sorted(dets) if i in marks and i in ids]
    if not use:
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in use])
    img = np.vstack([dets[i] for i in use])
    return obj, img, use


# ---------------------------------------------------------------- car pose
def car_corners_body(side):
    """Car tag corners in the body frame (x forward, y left), ArUco order, top edge forward (the
    'DAU XE' arrow of the print points to the front, guide 2026-11-10 item 5)."""
    h = side / 2.0
    return np.array([[h, h], [h, -h], [-h, -h], [-h, h]])


def rigid_fit(src, dst, with_scale=False):
    """2D rigid (Kabsch) src -> dst: (angle, translation, scale, rms of the fit)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ms, dst - md
    hm = a.T @ b
    ang = math.atan2(hm[0, 1] - hm[1, 0], hm[0, 0] + hm[1, 1])
    c, s = math.cos(ang), math.sin(ang)
    rot = np.array([[c, -s], [s, c]])
    sc = float(np.sum(b * (a @ rot.T)) / np.sum(a * a)) if with_scale else 1.0
    t = md - sc * (rot @ ms)
    rms = float(np.sqrt(np.mean(np.sum((sc * a @ rot.T + md - dst) ** 2, axis=1))))
    return ang, t, sc, rms


def car_pose(cam, corners_px, side, kappa, plumb, tag_offset=(0.0, 0.0, 0.0)):
    """(x, y, psi) of the WHEEL centre on the floor, tag centre (x, y), fit residual [m], apparent
    scale of the tag after the parallax correction, from the 4 corners of the car tag (design 3.4,
    3.5). kappa = h / Z_c; plumb = (x, y) of the plumb point of the lens on the floor. tag_offset =
    (dx, dy) tag centre minus wheel centre in the BODY frame, dpsi tag heading minus body heading (as
    track --tag-offset says): the offset is turned by the body heading ang - dpsi, not by the tag
    heading (found 16/11 with c7_calib_drift CD3-2; the old code was off by about |offset| x dpsi)."""
    p_floor = cam.pix2floor(np.asarray(corners_px, float), 0.0)
    C = np.asarray(plumb, float)
    P = p_floor + kappa * (C - p_floor)
    body = car_corners_body(side)
    ang, t, _, res = rigid_fit(body, P)
    _, _, sc, res_sim = rigid_fit(body, P, with_scale=True)
    dx, dy, dpsi = tag_offset
    c, s = math.cos(ang - dpsi), math.sin(ang - dpsi)
    xw, yw = t[0] - (c * dx - s * dy), t[1] - (s * dx + c * dy)
    return dict(x=float(xw), y=float(yw), psi=float(cp_wrap(ang - dpsi)), tag_x=float(t[0]), tag_y=float(t[1]),
                tag_psi=float(ang), res_m=res, res_sim_m=res_sim, scale=sc)


def res_sim_from(res_m, scale, side):
    """Residual of the fit WITH scale from the rigid residual and the scale of the same 4 corners: the
    rotation of the two Procrustes fits is the same and the scale is the least-squares one, so
    res_rigid^2 = res_sim^2 + (1 - scale)^2 * side^2 / 2 (each corner side / sqrt 2 from the centre)."""
    return math.sqrt(max(0.0, res_m * res_m - ((1.0 - scale) * side) ** 2 / 2.0))


def cp_wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------- gates and speed
def sg_derivative(t, x, half=SG_HALF):
    """Derivative by a quadratic fit over +-half samples on the real times (design 3.8)."""
    n = len(t)
    v = np.full(n, np.nan)
    for k in range(n):
        a, b = max(0, k - half), min(n, k + half + 1)
        if b - a < 5:
            continue
        c = np.polyfit(np.asarray(t[a:b]) - t[k], np.asarray(x[a:b]), 2)
        v[k] = c[1]
    return v


def check_gates(g, what='cong_khung'):
    """The frame-gate thresholds (GATE_KEYS) as a checked dict (CS1-15: they come from the constants file)."""
    if not isinstance(g, dict):
        raise ValueError(f'{what}: a table {GATE_KEYS}')
    out = {}
    for k in GATE_KEYS:
        v = g.get(k)
        lo, hi = GATE_RANGES[k]
        whole = k in ('min_marks', 'scale_min')
        if isinstance(v, bool) or not isinstance(v, int if whole else (int, float)) or not math.isfinite(v) \
                or not lo <= v <= hi:
            raise ValueError(f'{what} {k}: {v!r} is not {"a whole number" if whole else "a number"} in [{lo}, {hi}]')
        out[k] = v if whole else float(v)
    return out


def load_gates(path):
    """(gates, sha256, trang_thai) from the 'cong_khung' block of a constants file (c7_score --hang-so, CS1-15)."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, dict) or 'cong_khung' not in d:
        raise ValueError(f'{path}: no cong_khung block (frame-gate thresholds)')
    return check_gates(d['cong_khung'], f'{os.path.basename(path)} cong_khung'), sha256(path), d.get('trang_thai')


def apply_gates(rows, rms_h, gates=None, quant=0.0):
    """Sets valid / why on every frame row in place (O1 3.3 (1), (2); module docstring) and the speeds on the
    valid frames. Order: floor gate (1), shape (2a) on the residual WITH scale, then scale (2b) against the median
    of the neighbours within +-scale_win_s (frame times, the frame excluded) among the frames that passed (1) and
    (2a) (scale_ref, NaN when fewer than scale_min). Frame times must increase. gates: the thresholds (default
    DEFAULT_GATES; the constants file at C8, CS1-15). quant > 0 (regate of a rounded frames.csv, CS1-17): res_sim_m
    lies in [_res_sim_lo, _res_sim_hi] of the row (else +-quant) and every scale is off by up to quant; a frame whose
    (2a) or (2b) verdict changes inside those intervals fails with why 'can_track_lai' (decide it from the video
    with track). A neighbour whose (2a) is undecided may or may not be a neighbour (CS2-6): the median runs over
    the certain neighbours plus every subset of those (keeping scale_min), and a frame whose verdict changes over
    that range is 'can_track_lai' too. Returns the median scale of the frames that passed (1), (2a) (reported
    only)."""
    g = DEFAULT_GATES if gates is None else check_gates(gates)
    for r in rows:
        why = []
        if r['n_marks'] < g['min_marks']:
            why.append('moc_it')
        elif not (r['rms_px'] <= rms_h + g['rms_extra_px']):
            why.append('moc_du_thua')
        if r.get('khop_ok') is False:                     # 33-mark chain: the main fit is not trusted (v3 2, why khop)
            why.append('khop')
        if not r['car_seen']:
            why.append('khong_thay_xe')
        elif quant > 0.0:
            lo = r.get('_res_sim_lo', r['res_sim_m'] - quant)
            hi = r.get('_res_sim_hi', r['res_sim_m'] + quant)
            if not (hi <= g['car_res_sim_m']):
                why.append('can_track_lai' if lo <= g['car_res_sim_m'] else 'xe_meo')
        elif not (r['res_sim_m'] <= g['car_res_sim_m']):
            why.append('xe_meo')
        r['why'] = why
        r['scale_ref'] = float('nan')
        for k in ('vx', 'vy', 'wz', 'speed'):
            r[k] = float('nan')
    p12 = [k for k, r in enumerate(rows) if not r['why']]
    sc = np.array([rows[k]['scale'] for k in p12])
    # neighbours of (2b): the frames that passed (1) and (2a). With quant > 0 a frame whose ONLY failure is an
    # undecided (2a) ('can_track_lai') may be in that set or not (CS2-6): pool them, flagged
    pool = sorted(p12 + ([k for k, r in enumerate(rows) if r['why'] == ['can_track_lai']] if quant > 0.0 else []))
    amb = np.array([bool(rows[k]['why']) for k in pool], bool)
    sp = np.array([rows[k]['scale'] for k in pool])
    tp = np.array([rows[k]['t_s'] for k in pool])
    smin, rel = g['scale_min'], g['scale_rel']
    for j, k in enumerate(pool):
        if amb[j]:
            continue
        a = int(np.searchsorted(tp, tp[j] - g['scale_win_s'] - 1e-9, side='left'))
        b = int(np.searchsorted(tp, tp[j] + g['scale_win_s'] + 1e-9, side='right'))
        idx = np.r_[np.arange(a, j), np.arange(j + 1, b)].astype(int)
        nb, nu = sp[idx][~amb[idx]], np.sort(sp[idx][amb[idx]])
        if len(nb) + len(nu) < smin:
            rows[k]['why'].append('ti_le_it_lan_can')
            continue
        if not len(nu):
            ref = float(np.median(nb))
            mmin = mmax = ref
        else:
            # the median over the certain neighbours plus any subset of the undecided ones that keeps at least
            # scale_min: for a subset size the smallest (largest) undecided scales give the lowest (highest) median
            ref = float(np.median(np.r_[nb, nu]))
            sizes = range(max(0, smin - len(nb)), len(nu) + 1)
            mmin = min(float(np.median(np.r_[nb, nu[:m]])) for m in sizes)
            mmax = max(float(np.median(np.r_[nb, nu[len(nu) - m:]])) for m in sizes)
        rows[k]['scale_ref'] = ref
        few = len(nb) < smin                # the frame may also fail for too few neighbours
        if quant > 0.0:                    # the scale and every neighbour off by up to quant: an interval of ratios
            r_lo = (rows[k]['scale'] - quant) / (mmax + quant) - 1.0
            r_hi = (rows[k]['scale'] + quant) / (mmin - quant) - 1.0
            dmax = max(abs(r_lo), abs(r_hi))
            dmin = 0.0 if r_lo <= 0.0 <= r_hi else min(abs(r_lo), abs(r_hi))
            if not dmin <= rel:
                rows[k]['why'].append('ti_le')
            elif not dmax <= rel or few:
                rows[k]['why'].append('can_track_lai')
        elif not abs(rows[k]['scale'] / ref - 1.0) <= rel:
            rows[k]['why'].append('ti_le')
    for r in rows:
        r['valid'] = not r['why']
    vi = [k for k, r in enumerate(rows) if r['valid']]
    if len(vi) >= 5:
        tv = np.array([rows[k]['t_s'] for k in vi])
        vx = sg_derivative(tv, np.array([rows[k]['x'] for k in vi]))
        vy = sg_derivative(tv, np.array([rows[k]['y'] for k in vi]))
        wz = sg_derivative(tv, np.unwrap(np.array([rows[k]['psi'] for k in vi])))
        for j, k in enumerate(vi):
            rows[k].update(vx=float(vx[j]), vy=float(vy[j]), wz=float(wz[j]),
                           speed=float(math.hypot(vx[j], vy[j])) if np.isfinite(vx[j]) else float('nan'))
    return float(np.median(sc)) if len(sc) else float('nan')


def res_sim_interval(res_m, scale, side, q=CSV_QUANT):
    """[lo, hi] of res_sim_from when res_m and scale are each known to +-q only (rounded CSV, CS1-17): res_sim grows
    with res_m and falls with |1 - scale|."""
    d = abs(1.0 - scale)
    lo = res_sim_from(max(0.0, res_m - q), 1.0 - (d + q), side)
    hi = res_sim_from(res_m + q, 1.0 - max(0.0, d - q), side)
    return lo, hi


def cell_quant(row, col):
    """Half a unit of the last decimal written in the frames.csv cell (read_frames '_dec'); CSV_QUANT (6 decimals)
    for a row without that record."""
    d = (row.get('_dec') or {}).get(col)
    return CSV_QUANT if d is None else 0.5 * 10.0 ** (-d)


def regate(rows, session, gates=None):
    """The gates of this version on rows read from a frames.csv written before (read_frames): res_sim_m is ALWAYS
    rebuilt from res_m and scale (res_sim_from: an exact identity of the two Procrustes fits) with the rounding of
    THOSE cells (CS1-17, CS2-7: a res_sim_m written by an earlier regate is an estimate, not a measurement, and is
    not read), then apply_gates with that rounding: a frame whose verdict of (2a) or (2b) changes inside the interval
    fails with why 'can_track_lai' (counted by the caller; decide it with track on the video). The rounding of a cell
    is its number of decimals (cell_quant); write_frames keeps the decimals read, so a regate of a regate gives the
    same verdicts. In place; returns the median scale. regate does NOT replace track when the pose measurement itself
    has changed since the file was written (for example the tag-offset rotation of car_pose, 16/11): it only
    re-applies the gates to the stored residuals and scales; the stored centres are used as they are."""
    side = session['car_side_m']
    q_scale = 0.0
    for r in rows:
        r.pop('_res_sim_lo', None)
        r.pop('_res_sim_hi', None)
        if not r['car_seen']:
            continue
        dec = r.setdefault('_dec', {})
        for c in ('res_m', 'scale'):                     # CS4-2: the fallback rounding (6 decimals) needs values on that
            v = r[c]                                     # grid, else writing 6 decimals would move the interval
            if c not in dec and np.isfinite(v) and abs(v * 1e6 - round(v * 1e6)) > 1e-6:
                raise ValueError(f'frame {r.get("idx")}: {c} {v!r} is not on the 6-decimal grid and its rounding is not '
                                 "given ('_dec'): regate cannot bound it (CS4-2)")
        for c in ('res_m', 'scale', 'res_sim_m'):        # CS3-3: a row on the fallback rounding is written back with
            dec.setdefault(c, 6)                         # 6 decimals, never with digits it does not hold
        q = max(cell_quant(r, 'res_m'), cell_quant(r, 'scale'))
        q_scale = max(q_scale, cell_quant(r, 'scale'))
        if np.isfinite(r['res_m']) and np.isfinite(r['scale']):
            r['res_sim_m'] = res_sim_from(r['res_m'], r['scale'], side)
            r['_res_sim_lo'], r['_res_sim_hi'] = res_sim_interval(r['res_m'], r['scale'], side, q)
        else:
            r['res_sim_m'] = float('nan')
            r['_res_sim_lo'] = r['_res_sim_hi'] = float('nan')
    out = apply_gates(rows, session['rms_px_H'], gates, quant=q_scale or CSV_QUANT)
    for r in rows:
        r.pop('_res_sim_lo', None)
        r.pop('_res_sim_hi', None)
    return out


# ---------------------------------------------------------------- calibration (clip H)
def calibrate(video, marks, zc, plumb, tag_h, car_side, stride=1, max_frames=None, min_frac=0.5, floor=None,
              model=None, n_groups=KC_GROUPS, roles=None, dets_in=None):
    """Session camera from the static clip H (design 3.0, 3.3, 4.2 (a)). Returns the session dict.
    floor: IDs of the floor marks (c7_clip_probe.floor_ids, by vai_tro); None: ID 1-24.
    model: the LOCKED camera model (design 33 marks v3 1: calib --model; written as model_khoa, the cross-validation is
    then a report only) and the block 'kiem_cheo' (groups, the clip-H cameras without each group and each pair,
    baselines of every chain, kiem_cheo_block). None: the model chosen by cross-validation, no block (as d59197a).
    dets_in: (detections per frame, (h, w), frames) instead of reading the video (tests on synthetic detections)."""
    if model is not None and model not in cp.MODELS:
        raise ValueError(f'--model {model!r}: one of {cp.MODELS}')
    floor = tuple(sorted(FLOOR_IDS if floor is None else floor))
    dets, shape, n = [], None, 0
    if dets_in is not None:
        dets, shape, n = list(dets_in[0]), tuple(dets_in[1]), int(dets_in[2])
    else:
        det = make_detector()
        for k, t, g in iter_frames(video, max_frames):
            n += 1
            if k % stride:
                continue
            shape = g.shape
            dets.append(detect_refined(det, g))
    if shape is None:
        raise ValueError(f'{video}: no frame read')
    h_px, w_px = shape
    acc = {}
    for d in dets:
        for i, c in d.items():
            acc.setdefault(i, []).append(c)
    mean = {i: np.mean(np.array(v), axis=0) for i, v in acc.items() if len(v) >= min_frac * len(dets)}
    spread = {i: float(np.max(np.linalg.norm(np.array(v).mean(axis=1) - np.array(v).mean(axis=(0, 1)), axis=1)))
              for i, v in acc.items() if i in mean}
    ids = [i for i in sorted(mean) if i in marks and i in floor]
    if len(ids) < 6:
        raise ValueError(f'only {len(ids)} floor marks seen in at least half of the frames: no calibration')
    obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in ids])
    img = np.vstack([mean[i] for i in ids])
    fits, cv = {}, {}
    for m in cp.MODELS:
        cam, rms = fit_two_pass(m, obj, img, w_px, h_px, zc)
        fits[m] = dict(rms_px=rms, f_px=cam.f, k=cam.k.tolist(), Zc_m=float(abs(cam.centre[2])),
                       corner_distortion=cam.corner_distortion(w_px, h_px))
        cv[m] = cp.cross_validate(m, ids, mean, marks, w_px, h_px, zc)
    chosen = cp.choose_model(cv) if model is None else model
    if chosen is None:
        raise ValueError('no camera model could be cross-validated')
    cam, rms = fit_two_pass(chosen, obj, img, w_px, h_px, zc)
    if model is not None and not cam.fit_info['ok']:
        raise ValueError(f'clip H camera fit did not converge: {cam.fit_info} (v3 2, T33-17)')
    per_mark = []
    for i in ids:
        r = cam.project(np.column_stack([cp.mark_corners_floor(*marks[i]), np.zeros(4)])) - mean[i]
        ctr = cam.pix2floor(mean[i].mean(axis=0, keepdims=True))[0]
        per_mark.append(dict(id=i, ring='giua' if i >= 25 else ('ngoai' if i >= 13 else 'trong'), rms_px=float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))),
                             err_mm=float(np.hypot(*(ctr - np.array(marks[i][:2]))) * 1000), spread_px=spread[i]))
    check = {i: float(np.hypot(*(cam.pix2floor(mean[i].mean(axis=0, keepdims=True))[0] - np.array(marks[i][:2]))) * 1000)
             for i in sorted(mean) if i in marks and i not in floor and i != CAR_ID}
    zc_used = zc if zc is not None else float(abs(cam.centre[2]))
    extra = {}
    if model is not None:
        extra = dict(model_khoa=model, vai_tro={str(i): (roles or {}).get(i, '') for i in sorted(marks)},
                     kiem_cheo=kiem_cheo_block(dets, mean, marks, ids, cam, model, w_px, h_px, zc, n_groups,
                                               floor_all=[i for i in floor if i in marks and i != CAR_ID]))
    return dict(
        **extra,
        kind='c7_session', video=os.path.basename(video), video_sha256=sha256(video) if os.path.isfile(video) else None,
        frames=n, stride=stride,
        width=w_px, height=h_px, camera=cam_to_dict(cam), rms_px_H=rms, model_fits=fits, cross_validation_mm=cv,
        chosen=chosen, fit_ids=ids, floor_ids=list(floor), ref_centres_px={str(i): mean[i].mean(axis=0).tolist() for i in ids},
        ref_corners_px={str(i): mean[i].tolist() for i in ids},   # (4, 2) mean refined corners (c7_calib_drift CD1-3)
        per_mark=per_mark, check_tags_err_mm=check, zc_tape_m=zc, zc_fit_m=float(abs(cam.centre[2])),
        plumb_xy_m=list(plumb), tag_h_m=tag_h, car_side_m=car_side, kappa=tag_h / zc_used,
        kappa_src='thuoc: h / Z_c (design 3.4 cach 1)' if zc is not None else 'h / Z_c cua phep khop (KHONG dung cho so chinh)',
        tag_offset=[0.0, 0.0, 0.0], tag_offset_src='chua do (thu tinh clip T, C8): 0',
        marks={str(i): list(v) for i, v in marks.items()},
        gates=gates_dict())


def gates_dict(gates=None, src=None):
    """The thresholds applied (written to session.json / track.json; c7_score compares them with its constants
    file, CS1-15). src: dict(file, sha256, trang_thai) of the constants file, None for the module defaults."""
    g = DEFAULT_GATES if gates is None else check_gates(gates)
    return dict(g, rule='(2a) residual with scale, (2b) scale vs median of the neighbours (Liam 16/11 CODE_CAM)',
                status='DE XUAT, khoa o C8' if src is None else f'tu file hang so ({src.get("trang_thai")})',
                hang_so=src)


# ---------------------------------------------------------------- 33 marks: chains, comparators (design v3 2, 3)
def kiem_cheo_block(dets, mean, marks, ids, cam, model, w, h, zc, n_groups=KC_GROUPS, cond_max=COND_MAX, floor_all=None):
    """Block 'kiem_cheo' of a session calibrated with a locked model (v3 1, 2; v4 3): groups; clip-H cameras (intrinsics
    + pose, fit_two_pass as the main one, model locked) without each group k and each pair {a, b}; their convergence;
    the BASELINE of every chain c and target j = mean over the clip-H frames of the position of j by its comparator
    (pose refitted from the comparator camera on every mark outside the comparator's groups, that set SEEN WHOLE in the
    frame, v4 3 / KC4); marks of a chain without a baseline (nen_thieu: track refuses the session); the condition numbers
    of the main per-frame refit on H (to lock COND_MAX at C8); the centre-definition difference with d59197a.
    floor_all: every floor mark of the marker files (Codex C33-15: the groups and universes are those of the locked
    list, not of the marks seen in clip H; a floor mark missing from H is a mark without baseline, it blocks the
    session, it does not leave the universe)."""
    floor_all = sorted(set(ids) | set(floor_all or ()))
    groups = make_groups(floor_all, n_groups)
    G = len(groups)
    gmap = {i: k for k, g in enumerate(groups) for i in g}
    cams, khop = {CHAIN_XE: cam}, {CHAIN_XE: cam.fit_info}
    excls = [frozenset([k]) for k in range(G)] + [frozenset([a, b]) for a in range(G) for b in range(a + 1, G)]
    for ex in excls:
        train = [i for i in ids if gmap[i] not in ex]
        if len(train) < MIN_TRAIN:
            raise ValueError(f'camera without groups {sorted(ex)}: only {len(train)} marks')
        c, _ = fit_two_pass(model, np.vstack([cp.mark_corners_floor(*marks[i]) for i in train]),
                            np.vstack([mean[i] for i in train]), w, h, zc)
        cams[cam_key(ex)], khop[cam_key(ex)] = c, c.fit_info
    bad = {k: v for k, v in khop.items() if not v['ok']}
    if bad:
        raise ValueError(f'clip H cameras did not converge: {bad} (v3 2, T33-17)')
    zc_h = float(abs(cam.centre[2]))
    chains = chain_names(groups)
    acc = {c: {} for c in chains}
    conds = []
    for d in dets:
        fits = {}
        for c in chains:
            for t in range(G):
                if c != CHAIN_XE and t == int(c):
                    continue
                ex = comparator_excl(c, t)
                anchors = [i for i in ids if gmap[i] not in ex]
                if not set(anchors) <= set(d):
                    continue                                   # KC4: the anchor set seen whole
                key = cam_key(ex)
                if key not in fits:
                    obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in anchors])
                    fits[key] = refit_pose_checked(cams[key], obj, np.vstack([d[i] for i in anchors]), None, cond_max)
                fc, _, ok, _, _ = fits[key]
                if not ok:
                    continue
                for j in groups[t]:
                    if j in d:
                        acc[c].setdefault(j, []).append(mark_centre_floor(fc, d[j]))
        if set(ids) <= set(d):
            obj = np.vstack([cp.mark_corners_floor(*marks[i]) for i in ids])
            conds.append(refit_pose_checked(cam, obj, np.vstack([d[i] for i in ids]), None, math.inf)[4])
    nen, nen_n, thieu = {}, {}, []
    for c in chains:
        uc = [i for i in floor_all if c == CHAIN_XE or gmap[i] != int(c)]
        nen[c] = {str(j): np.mean(np.array(acc[c][j]), axis=0).tolist() for j in uc if acc[c].get(j)}
        nen_n[c] = {str(j): len(acc[c].get(j, [])) for j in uc}
        thieu += [[c, j] for j in uc if not acc[c].get(j)]
    lech = max(float(np.linalg.norm(mark_centre_floor(cam, mean[i]) - cam.pix2floor(mean[i].mean(axis=0, keepdims=True))[0]))
               for i in ids) * 1000.0
    return dict(nhom=groups, cam_H={k: cam_to_dict(v) for k, v in cams.items()}, khop_H=khop, nen=nen, nen_n=nen_n,
                nen_thieu=thieu, cond_max=cond_max, camz_tol_m=CAMZ_TOL_M,
                cond_H=dict(n=len(conds), max=float(max(conds)) if conds else None,
                            p95=float(np.percentile(conds, 95)) if conds else None),
                tam='trung binh 4 goc sau nghich chieu xuong san (v3 1, T33-15)', lech_tam_d59197a_mm_max=lech,
                luat=dict(win_s=md.WIN_S, win_min_n=md.WIN_MIN_N, win_gap_s=md.WIN_GAP_S, nen_win_s=md.NEN_WIN_S,
                          nen_raw_frac=md.NEN_RAW_FRAC, nen_raw_min=md.NEN_RAW_MIN, thong_ke='trung vi do lon (v4 1)'))


class ChainCtx:
    """The chains of a session with a 'kiem_cheo' block: chain xe (universe = every fit mark, camera cam_H[xe]) and
    chain k (universe without group k, camera cam_H[k]); comparator of a target of group t in chain c: camera without
    comparator_excl(c, t), anchors = the chain's anchors outside those groups (v3 2)."""

    def __init__(self, session):
        kc = session.get('kiem_cheo')
        if not kc or not session.get('model_khoa'):
            raise ValueError('session without model_khoa / kiem_cheo: calib --model first (v3 1)')
        if kc.get('nen_thieu'):
            raise ValueError(f'marks without a clip-H baseline {kc["nen_thieu"]}: the whole session is blocked (v3 3)')
        self.groups = [[int(i) for i in g] for g in kc['nhom']]
        self.G = len(self.groups)
        self.gmap = {i: k for k, g in enumerate(self.groups) for i in g}
        self.chains = chain_names(self.groups)
        self.cams = {k: cam_from_dict(v) for k, v in kc['cam_H'].items()}
        self.marks = {int(i): tuple(v) for i, v in session['marks'].items()}
        self.zc_h = float(session['zc_fit_m'])
        self.cond_max = float(kc.get('cond_max', COND_MAX))
        self.nen = {c: {int(j): np.asarray(v, float) for j, v in d.items()} for c, d in kc['nen'].items()}
        allm = set(self.gmap)
        self.universe = {CHAIN_XE: allm}
        self.universe.update({str(k): allm - set(g) for k, g in enumerate(self.groups)})

    def pose(self, key, anchors, d, cache=None, fkey=None):
        """(camera, rms, ok, ly_do, cond) of the pose refit from cam_H[key] on the anchors (sorted IDs) seen in d."""
        anchors = tuple(sorted(anchors))
        if len(anchors) < MIN_TRAIN:
            return None, float('nan'), False, 'thieu_moc', float('nan')
        ck = (fkey, key, anchors)
        if cache is not None and fkey is not None and ck in cache:
            x, rms, ok, ly, cond = cache[ck]
            c0 = self.cams[key]
            return cp.Camera(c0.model, c0.f, c0.cx, c0.cy, c0.k, x[:3], x[3:]), rms, ok, ly, cond
        obj = np.vstack([cp.mark_corners_floor(*self.marks[i]) for i in anchors])
        out = refit_pose_checked(self.cams[key], obj, np.vstack([d[i] for i in anchors]), None, self.cond_max)
        if cache is not None and fkey is not None:
            cache[ck] = (np.r_[out[0].rvec, out[0].tvec], out[1], out[2], out[3], out[4])
        return out

    def frame(self, d, excl=None, deltas=True, cache=None, fkey=None):
        """One frame: detections d {id: (4, 2) px}; excl {chain: marks out of its fit in this frame (exclusions +
        moc_nghi)}. Returns dict(seen, neo {chain: anchors}, main (pose tuple), kc {k: check of group k}, delta
        {chain: {j: (2,) comparator position - baseline}} (deltas=True))."""
        excl = excl or {}
        seen = sorted(i for i in d if i in self.gmap)
        neo = {c: [i for i in seen if i in self.universe[c] and i not in excl.get(c, ())] for c in self.chains}
        main = self.pose(CHAIN_XE, neo[CHAIN_XE], d, cache, fkey)
        kc = {}
        for k in range(self.G):
            c = str(k)
            pc = self.pose(cam_key([k]), neo[c], d, cache, fkey)
            targets = [j for j in seen if self.gmap[j] == k]
            rows = []
            if pc[2]:
                hull = None
                if len(neo[c]) >= 3:
                    pts = np.array([self.marks[i][:2] for i in neo[c]], np.float32) * 1000.0
                    hull = cv2.convexHull(pts)
                for j in targets:
                    p = mark_centre_floor(pc[0], d[j])
                    e = p - np.array(self.marks[j][:2])
                    out = 0.0
                    if hull is not None:
                        dist = cv2.pointPolygonTest(hull, (float(self.marks[j][0] * 1000), float(self.marks[j][1] * 1000)), True)
                        out = max(0.0, -dist) / 1000.0
                    rows.append(dict(moc=j, nhom=k, chuoi=c, ex=float(e[0]), ey=float(e[1]), x_doan=float(p[0]),
                                     y_doan=float(p[1]), ngoai_suy_m=out))
            ly = pc[3] if not pc[2] else ('' if rows else 'khong_thay_moc_kiem')
            kc[k] = dict(ok=bool(rows), n=len(rows), err_max=max((math.hypot(r['ex'], r['ey']) for r in rows),
                                                                  default=float('nan')), ly_do=ly, rows=rows)
        delta = {}
        if deltas:
            for c in self.chains:
                dc = {}
                for t in range(self.G):
                    if c != CHAIN_XE and t == int(c):
                        continue
                    targets = [j for j in seen if self.gmap[j] == t]
                    if not targets:
                        continue
                    ex = comparator_excl(c, t)
                    pc = self.pose(cam_key(ex), [i for i in neo[c] if self.gmap[i] not in ex], d, cache, fkey)
                    if not pc[2]:
                        continue
                    for j in targets:
                        dc[j] = mark_centre_floor(pc[0], d[j]) - self.nen[c][j]
                delta[c] = dc
        return dict(seen=seen, neo=neo, main=main, kc=kc, delta=delta)


# ---------------------------------------------------------------- detections cache (dets.npz)
def save_dets(path, frames, video_sha, shape):
    """frames: [(idx, t_s, {id: (4, 2)})]. Written with numbers and plain strings only (np.load without pickle)."""
    ids, fo, cs = [], [], []
    for n, (_, _, d) in enumerate(frames):
        for i in sorted(d):
            ids.append(i), fo.append(n), cs.append(np.asarray(d[i], float))
    np.savez_compressed(path, idx=np.array([f[0] for f in frames], int), t=np.array([f[1] for f in frames], float),
                        ids=np.array(ids, int), frame_of=np.array(fo, int),
                        corners=np.array(cs, float).reshape(-1, 4, 2), shape=np.array(shape, int),
                        video_sha256=np.array(video_sha))


def load_dets(path):
    """(frames [(idx, t_s, {id: (4, 2)})], video_sha256, shape) of save_dets."""
    z = np.load(path, allow_pickle=False)
    frames = [(int(i), float(t), {}) for i, t in zip(z['idx'], z['t'])]
    for i, n, c in zip(z['ids'], z['frame_of'], z['corners']):
        frames[int(n)][2][int(i)] = c
    return frames, str(z['video_sha256']), tuple(int(v) for v in z['shape'])


def detect_video(video, max_frames=None, progress=None):
    det = make_detector()
    out, shapes = [], set()
    for k, t, g in iter_frames(video, max_frames):
        shapes.add(g.shape)
        out.append((k, t, detect_refined(det, g)))
        if progress and k % 100 == 0:
            progress(k)
    return out, shapes


# ---------------------------------------------------------------- tracking a run clip
def shift_of(vectors):
    """(shift_px, shift_mag_px) of the mark-centre shift vectors of one frame [px]: the length of the component-wise
    median vector (O1 3.5 'do lon cua trung vi vec-to', CS1-13) and the median of the lengths (reported)."""
    sv = np.asarray(vectors, float).reshape(-1, 2)
    return float(np.hypot(*np.median(sv, axis=0))), float(np.median(np.linalg.norm(sv, axis=1)))


def track(video, session, tag_offset=None, max_frames=None, progress=None, gates=None, gates_src=None):
    """Per-frame rows of a run clip with the session camera (design 3.5-3.8). Returns (rows, info).
    shift_px (camera bumped, O1 3.5): the LENGTH OF THE MEDIAN VECTOR of the mark-centre shifts from clip H
    (component-wise median, CS1-13); shift_mag_px: the median of the lengths (reported only)."""
    cam0 = cam_from_dict(session['camera'])
    marks = {int(i): tuple(v) for i, v in session['marks'].items()}
    ref = {int(i): np.array(v) for i, v in session['ref_centres_px'].items()}
    side, kappa, plumb = session['car_side_m'], session['kappa'], session['plumb_xy_m']
    off = tag_offset if tag_offset is not None else session['tag_offset']
    fl_ids = session_floor_ids(session)
    det = make_detector()
    rows, cam_prev, shapes = [], cam0, set()
    for k, t, g in iter_frames(video, max_frames):
        shapes.add(g.shape)
        d = detect_refined(det, g)
        obj, img, used = floor_obs(d, marks, fl_ids)
        r = dict(idx=k, t_s=t, n_marks=len(used), rms_px=float('nan'), shift_px=float('nan'), cam_z=float('nan'),
                 shift_mag_px=float('nan'), car_seen=False, x=float('nan'), y=float('nan'), psi=float('nan'), tag_x=float('nan'),
                 tag_y=float('nan'), res_m=float('nan'), res_sim_m=float('nan'), scale=float('nan'),
                 scale_ref=float('nan'), row_px=float('nan'), col_px=float('nan'),
                 vx=float('nan'), vy=float('nan'), wz=float('nan'), speed=float('nan'))
        sh = [d[i].mean(axis=0) - ref[i] for i in used if i in ref]
        if sh:
            r['shift_px'], r['shift_mag_px'] = shift_of(sh)
        cam = None
        if len(used) >= 4:                                 # 4 marks = 16 points: the pose is determined
            cam, r['rms_px'] = refit_pose(cam_prev, obj, img)
            r['cam_z'] = float(cam.centre[2])
            cam_prev = cam
        if cam is not None and CAR_ID in d:
            r.update(car_pose(cam, d[CAR_ID], side, kappa, plumb, off))
            r['car_seen'] = True
            r['row_px'] = float(d[CAR_ID][:, 1].mean())
            r['col_px'] = float(d[CAR_ID][:, 0].mean())      # C8b v4 3: rolling shutter by row and column
        rows.append(r)
        if progress and k % 100 == 0:
            progress(k)
    med_scale = apply_gates(rows, session['rms_px_H'], gates)
    info = dict(video=os.path.basename(video), video_sha256=sha256(video), frames=len(rows),
                decoded_shapes=sorted(list(s) for s in shapes), median_scale=med_scale, tag_offset=list(off),
                gates=gates_dict(gates, gates_src), shift_kind='median_vector',
                n_valid=int(sum(r['valid'] for r in rows)),
                zc_session_m=session.get('zc_fit_m'),
                cam_z_drift_rel=_cam_z_drift(rows, session.get('zc_fit_m')))
    return rows, info


def _cam_z_drift(rows, zc_h):
    """Largest |camera height of a frame / that of clip H - 1| (design 7 item 8: warning above 1 %)."""
    z = [abs(r['cam_z']) for r in rows if np.isfinite(r['cam_z'])]
    if not z or not zc_h:
        return None
    return float(max(abs(v / zc_h - 1.0) for v in z))


def u_so_thresholds(u_so, chains):
    """{chain: {nguong_m, nguong_tho_m}} from u_so.json of c7_kiem_cheo (block 'chuoi'), the values as written (KIEMCHEO
    tung_khung compares them with track.json nguong_dung)."""
    ch = (u_so or {}).get('chuoi') or {}
    miss = [c for c in chains if c not in ch]
    if miss:
        raise ValueError(f'u_so.json: chains {miss} missing (c7_kiem_cheo phan_tich of the same session)')
    out = {}
    for c in chains:
        v = ch[c]
        for k in ('nguong_m', 'nguong_tho_m', 'u_so_m', 'u_so_tho_m'):
            x = v.get(k)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 and k[0] == 'n':
                raise ValueError(f'u_so.json chuoi {c} {k}: {x!r}')
        out[c] = dict(nguong_m=v['nguong_m'], nguong_tho_m=v['nguong_tho_m'])
    return out


U_SO_NGUONG_CO_DINH_M = 0.001          # c7_kiem_cheo NGUONG_CO_DINH_M: T = 1 mm + u_so (page v10 7a)


def check_u_so_source(u_so, session, session_sha, ctx):
    """u_so.json must be of THIS session and setup (Codex C33-5): H_a session sha, locked model, groups, not a trial;
    every chain's numbers >= 0 and T = 1 mm + u_so, T_raw = 1 mm + u_so_raw."""
    ha = ((u_so.get('sessions') or {}).get('H_a') or {}).get('sha256')
    if ha != session_sha:
        raise ValueError(f'u_so.json is of another H_a session ({ha!r}, not {session_sha!r})')
    if u_so.get('model') != session.get('model_khoa'):
        raise ValueError(f"u_so.json model {u_so.get('model')!r} is not the session's {session.get('model_khoa')!r}")
    if [sorted(int(i) for i in g) for g in (u_so.get('nhom') or [])] != ctx.groups:
        raise ValueError('u_so.json groups are not the session groups')
    if u_so.get('thu'):
        raise ValueError('u_so.json is a trial (--thu)')
    for c in ctx.chains:
        v = (u_so.get('chuoi') or {}).get(c) or {}
        for k in ('u_so_m', 'u_so_tho_m'):
            x = v.get(k)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f'u_so.json chuoi {c} {k}: {x!r} (finite, >= 0)')
        for t_, u_ in (('nguong_m', 'u_so_m'), ('nguong_tho_m', 'u_so_tho_m')):
            if not abs(v.get(t_, math.nan) - (U_SO_NGUONG_CO_DINH_M + v[u_])) <= 1e-12:
                raise ValueError(f'u_so.json chuoi {c}: {t_} is not 1 mm + {u_}')


def undetected_bound(u_so):
    """D (m) of chain xe (v3 3, v4 4): the larger of T + u_so (a shift lasting a window or more) and T_raw + u_so_raw (a
    shorter one, the frames over T_raw were left out). Conditional: a bound only if u_so bounds the comparison error."""
    v = u_so['chuoi'][CHAIN_XE]
    return max(v['nguong_m'] + v['u_so_m'], v['nguong_tho_m'] + v['u_so_tho_m'])


def car_sensitivity(cam, anchors, d, marks, car_px, side, kappa, plumb, off, D, vertices=False):
    """v4 4: how much the car output (wheel centre x, y and heading psi of car_pose) moves when the tape centre of an
    anchor mark j is off by up to D, through the pose refit of the frame. Linear part at the solution: d pose / d m_j =
    -(Jx' W Jx)^-1 Jx' W Jm_j (Huber weights at the solution), times d car / d pose (central differences); s_xy_j, s_psi_j
    = the spectral norms (m per m, rad per m). Returns dict(U_xy = sum_j s_xy_j D, U_psi = sum_j s_psi_j D, s_xy_max,
    s_psi_max, ratio_xy, ratio_psi): with vertices=True every mark is also moved to the 4 vertices (+-D, +-D) and the pose
    refitted (warm start), ratio = largest |change| / (s |vertex|) (1 for a linear problem; c7_measure scales U by it)."""
    anchors = sorted(anchors)
    P = np.column_stack([np.vstack([cp.mark_corners_floor(*marks[i]) for i in anchors]), np.zeros(4 * len(anchors))])
    img = np.vstack([d[i] for i in anchors])
    x = np.r_[cam.rvec, cam.tvec]

    def mk(xv):
        return cp.Camera(cam.model, cam.f, cam.cx, cam.cy, cam.k, xv[:3], xv[3:])

    def car(c):
        p = car_pose(c, car_px, side, kappa, plumb, off)
        return np.array([p['x'], p['y'], p['psi']])

    r = (mk(x).project(P) - img).ravel()
    w = np.where(np.abs(r) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(r), 1e-12))
    h = 1e-6
    Jx = np.zeros((len(r), 6))
    Dc = np.zeros((3, 6))
    for a in range(6):
        e = np.zeros(6)
        e[a] = h
        Jx[:, a] = ((mk(x + e).project(P) - mk(x - e).project(P)).ravel()) / (2 * h)
        dc = car(mk(x + e)) - car(mk(x - e))
        dc[2] = cp_wrap(dc[2])
        Dc[:, a] = dc / (2 * h)
    dP = []
    for a in range(2):
        e = np.zeros(3)
        e[a] = h
        dP.append((mk(x).project(P + e) - mk(x).project(P - e)) / (2 * h))      # (4n, 2): d(u, v) / d X_a per point
    A = Jx.T @ (w[:, None] * Jx)
    Ainv = np.linalg.inv(A)
    c0 = car(mk(x))
    U_xy = U_psi = s_xy_max = s_psi_max = 0.0
    ratio_xy = ratio_psi = 0.0
    for m, j in enumerate(anchors):
        rows = slice(8 * m, 8 * m + 8)
        Jm = np.zeros((8, 2))
        for p in range(4):
            Jm[2 * p:2 * p + 2, 0] = dP[0][4 * m + p]
            Jm[2 * p:2 * p + 2, 1] = dP[1][4 * m + p]
        g = Jx[rows].T @ (w[rows, None] * Jm)
        S = Dc @ (-Ainv @ g)                                                   # (3, 2): car per metre of mark shift
        s_xy, s_psi = float(np.linalg.norm(S[:2], 2)), float(np.linalg.norm(S[2:3], 2))
        U_xy += s_xy * D
        U_psi += s_psi * D
        s_xy_max, s_psi_max = max(s_xy_max, s_xy), max(s_psi_max, s_psi)
        if vertices:
            for vx_, vy_ in ((D, D), (D, -D), (-D, D), (-D, -D)):
                Pv = P.copy()
                Pv[4 * m:4 * m + 4, 0] += vx_
                Pv[4 * m:4 * m + 4, 1] += vy_
                cv_, _, okv, _, _ = refit_pose_checked(mk(x), Pv[:, :2], img, abs(cam.centre[2]), math.inf, math.inf)
                if not okv:                                  # Codex C33-14: an untrusted vertex fit voids the ratio
                    ratio_xy = ratio_psi = math.inf
                    continue
                dlt = car(cv_) - c0
                dlt[2] = cp_wrap(dlt[2])
                L = math.hypot(vx_, vy_)
                if s_xy * L > 1e-12:
                    ratio_xy = max(ratio_xy, float(np.hypot(dlt[0], dlt[1])) / (s_xy * L))
                if s_psi * L > 1e-15:
                    ratio_psi = max(ratio_psi, abs(float(dlt[2])) / (s_psi * L))
    return dict(U_xy=U_xy, U_psi=U_psi, s_xy_max=s_xy_max, s_psi_max=s_psi_max,
                ratio_xy=ratio_xy if vertices else None, ratio_psi=ratio_psi if vertices else None)


def _ids_cell(ids):
    return ';'.join(str(int(i)) for i in sorted(ids))


def track_chains(frames, video_name, video_sha, session, tag_offset=None, gates=None, gates_src=None, lich=None,
                 run=None, u_so=None, u_so_sha=None, cache=None, shapes=None, thu=False):
    """track for a session with a 'kiem_cheo' block (design 33 marks v3 + v4). frames: [(idx, t_s, detections)] of the
    clip (detect_video or load_dets). Every pose is refitted from a clip-H camera (stateless, T33-8) with the trust flags;
    the main fit (chain xe) leaves out the marks the schedule excludes at that frame (lich: the converged schedule of the
    session, run: this clip's name in it); chain k checks group k on its own anchors (kiem_cheo_khung rows). With lich and
    u_so (the final pass, cham_lai 2): do_nhay (U_moc_xy_m, U_moc_psi_rad) on the valid frames. Returns (rows, khung,
    info)."""
    ctx = ChainCtx(session)
    marks = ctx.marks
    ref = {int(i): np.array(v) for i, v in session['ref_centres_px'].items()}
    side, kappa, plumb = session['car_side_m'], session['kappa'], session['plumb_xy_m']
    off = tag_offset if tag_offset is not None else session['tag_offset']
    sched, q = None, None
    if lich is not None:
        names = [c['name'] for c in lich['clips']]
        if run not in names:
            raise ValueError(f'run {run!r} is not a clip of the schedule ({names})')
        q = names.index(run)
        sched = lich['chuoi']
    rows, khung, keep = [], [], {}
    for idx, t, d in frames:
        excl = {c: md.excluded_at(sched[c], q, idx) for c in ctx.chains} if sched else {}
        fr = ctx.frame(d, excl, deltas=False, cache=cache, fkey=idx)
        cam, rms, ok, ly, cond = fr['main']
        used = fr['neo'][CHAIN_XE]
        r = dict(idx=idx, t_s=t, n_marks=len(used), rms_px=rms, shift_px=float('nan'), cam_z=float('nan'),
                 shift_mag_px=float('nan'), car_seen=False, x=float('nan'), y=float('nan'), psi=float('nan'),
                 tag_x=float('nan'), tag_y=float('nan'), res_m=float('nan'), res_sim_m=float('nan'), scale=float('nan'),
                 scale_ref=float('nan'), row_px=float('nan'), col_px=float('nan'),
                 vx=float('nan'), vy=float('nan'), wz=float('nan'), speed=float('nan'),
                 khop_ok=bool(ok), khop_ly_do=ly, cond=cond, moc_thay=_ids_cell(fr['seen']), moc_dung=_ids_cell(used))
        for c in ctx.chains:
            r[f'moc_neo_{c}'] = _ids_cell(fr['neo'][c])
        n_khung0 = len(khung)
        for k, v in fr['kc'].items():
            r[f'kc_ok_{k}'], r[f'kc_n_{k}'], r[f'kc_err_max_{k}'], r[f'kc_ly_do_{k}'] = int(v['ok']), v['n'], \
                v['err_max'], v['ly_do']
            for row in v['rows']:
                khung.append(dict(clip=run or video_name, khung=idx, **row))
        nen_c, bd = [], []
        if sched:
            for c in ctx.chains:
                a, b = md.inconclusive_at(sched[c], q, idx)
                if a:
                    nen_c.append(c)
                bd += [f'{c}:{j}' for j in b]
        inc = set(nen_c) | {x.split(':')[0] for x in bd}           # chains not concluded at this frame (Codex C33-1)
        for row in khung[n_khung0:]:
            row['chua_ket_luan'] = int(row['chuoi'] in inc)
        r['nen_doi'] = ';'.join(nen_c)
        r['bat_dinh_moc'] = ';'.join(bd)
        r['chua_ket_luan_xe'] = int(CHAIN_XE in nen_c or any(s.startswith(CHAIN_XE + ':') for s in bd))
        sh = [d[i].mean(axis=0) - ref[i] for i in used if i in ref]
        if sh:
            r['shift_px'], r['shift_mag_px'] = shift_of(sh)
        if cam is not None:
            r['cam_z'] = float(cam.centre[2])
        if cam is not None and CAR_ID in d:
            r.update(car_pose(cam, d[CAR_ID], side, kappa, plumb, off))
            r['car_seen'] = True
            r['row_px'] = float(d[CAR_ID][:, 1].mean())
            r['col_px'] = float(d[CAR_ID][:, 0].mean())
            keep[len(rows)] = (cam, used, d)
        rows.append(r)
    med_scale = apply_gates(rows, session['rms_px_H'], gates)
    nl = None
    if u_so is not None and lich is not None:
        D = undetected_bound(u_so)
        vi = [k for k, r in enumerate(rows) if r['valid'] and k in keep]
        out, rx, rp = {}, 0.0, 0.0
        for n_, k in enumerate(vi):
            cam, used, d = keep[k]
            s = car_sensitivity(cam, used, d, marks, d[CAR_ID], side, kappa, plumb, off, D, vertices=n_ % NONLIN_EVERY == 0)
            out[k] = s
            if s['ratio_xy'] is not None:
                rx, rp = max(rx, s['ratio_xy']), max(rp, s['ratio_psi'])
        fx, fp = max(1.0, rx), max(1.0, rp)
        for k, s in out.items():
            rows[k].update(U_moc_xy_m=s['U_xy'] * fx, U_moc_psi_rad=s['U_psi'] * fp, do_nhay_xy=s['s_xy_max'],
                           do_nhay_psi=s['s_psi_max'])
        nl = dict(D_m=D, D_dieu_kien='chi la can neu u_so chan sai phep so luc chay (v4 4)', ratio_xy_max=rx,
                  ratio_psi_max=rp, he_so_xy=fx, he_so_psi=fp, moi_n_khung=NONLIN_EVERY, n_khung=len(out),
                  gioi_han='can thuc nghiem: tuyen tinh tai nghiem, dinh hop +-D tren mau moi 30 khung hop le',
                  vai_tro='THAM DO, khong phai can da xac nhan (Codex C33-14: he so day cung, IRLS co dinh, moi mau '
                          '30 khung; C33-13: cach truyen U sang xe do C8B chot)')
    info = dict(video=video_name, video_sha256=video_sha, frames=len(rows),
                decoded_shapes=sorted(list(s) for s in (shapes or ())), median_scale=med_scale, tag_offset=list(off),
                gates=gates_dict(gates, gates_src), hang_so_sha256=(gates_src or {}).get('sha256'),
                shift_kind='median_vector', n_valid=int(sum(r['valid'] for r in rows)),
                n_khop_khong_tin=int(sum(not r['khop_ok'] for r in rows)),
                n_chua_ket_luan_xe=int(sum(r['chua_ket_luan_xe'] for r in rows)),
                n_chua_ket_luan_xe_hop_le=int(sum(r['chua_ket_luan_xe'] and r['valid'] for r in rows)),
                n_hang_kiem_chua_ket_luan=int(sum(k_['chua_ket_luan'] for k_ in khung)),
                zc_session_m=session.get('zc_fit_m'), cam_z_drift_rel=_cam_z_drift(rows, session.get('zc_fit_m')),
                model_khoa=session['model_khoa'], nhom=ctx.groups, chuoi=ctx.chains, run=run, do_nhay=nl,
                u_so_sha256=u_so_sha, nguong_dung=u_so_thresholds(u_so, ctx.chains) if u_so is not None else None,
                cham_lai=2 if (lich is not None and lich.get('hoi_tu') and not thu) else 1, thu=bool(thu))
    if lich is not None:
        info.update(lich_sha256=lich['lich_sha256'], lap_E=lich.get('lap'), **schedule_for_run(lich, q, frames))
    else:
        info.update(lich_sha256=None, moc_loai=[], moc_nghi=[], nen_doi=[], bat_dinh_moc=[], lap_E=None)
    return rows, khung, info


def schedule_for_run(lich, q, frames):
    """moc_loai / moc_nghi / nen_doi / bat_dinh_moc of the schedule that touch clip q, for track.json. moc_loai t_bd is
    in THIS clip's time (KIEMCHEO compares it with frames.csv t_s): the frame time when the exclusion starts in this clip,
    the first frame's time when it started in an earlier clip (tu_clip_truoc)."""
    t_of = {idx: t for idx, t, _ in frames}
    t0 = frames[0][1] if frames else 0.0
    loai, nghi, nen, bat = [], [], [], []
    for c, s in lich['chuoi'].items():
        for j, v in s.get('loai', {}).items():
            if v['tu'][0] > q:
                continue
            before = v['tu'][0] < q
            loai.append(dict(chuoi=c, id=int(j), t_bd=t0 if before else t_of.get(v['tu'][1], t0),
                             tu_clip_truoc=before, tu=v['tu'], bao_dong=v['bao_dong'], bat_dinh=v['bat_dinh']))
        nghi += [dict(chuoi=c, id=j, khung=f) for j, qq, f in s.get('moc_nghi', []) if qq == q]
        nen += [dict(chuoi=c, **v) for v in s.get('nen_doi', []) if v['tu'][0] <= q and
                (v['den'] is None or v['den'][0] >= q)]
        bat += [dict(chuoi=c, **v) for v in s.get('bat_dinh_moc', []) if v['tu'][0] <= q <= v['den'][0]]
    return dict(moc_loai=loai, moc_nghi=nghi, nen_doi=nen, bat_dinh_moc=bat)


def _tap_cover_missing(tap_prev, u_so, groups):
    """Comparator fit sets of the previous schedule's anchor sets that u_so.json does not list as computed (tap_tinh)."""
    gmap = {i: k for k, g in enumerate(groups) for i in g}
    done = {(c, int(x['nhom_dich']), frozenset(int(i) for i in x['tap']))
            for c, v in (u_so.get('chuoi') or {}).items() for x in v.get('tap_tinh', [])}
    miss = set()
    for c, sets in tap_prev.items():
        for S in sets:
            for t in range(len(groups)):
                if c != CHAIN_XE and t == int(c):
                    continue
                ex = comparator_excl(c, t)
                fit = frozenset(i for i in S if gmap.get(i) not in ex)
                if len(fit) >= MIN_TRAIN and (c, t, fit) not in done:
                    miss.add((c, t, tuple(sorted(fit))))
    return sorted(miss)


def build_lich(session, session_sha, runs, u_so=None, u_so_sha=None, prev=None, cache_dir=None, progress=None):
    """One step of the fixed point of the session schedule (v3 3 order, v4 3): runs = [(name, kind 'luot'|'H', frames,
    dets_sha256)] in session order (thu_tu_clip.csv). Deltas of every chain under the exclusions of prev (the schedule
    of the previous step; none = empty), thresholds of u_so (none: the first step, an empty schedule and the anchor sets
    only), the rule of c7_moc_dich per chain -> the new schedule. tap_moc = every anchor set of every chain under prev
    and under the new schedule, merged with prev's. Converged (hoi_tu) when the new schedule (with thresholds and the
    tap_moc sha) equals prev's; a repeat of an older step = cycle; more than 10 steps = not converged. Returns lich."""
    ctx = ChainCtx(session)
    empty = dict(loai={}, moc_nghi=[], nen_doi=[], bat_dinh_moc=[])
    clips = [dict(name=n, kind=k, dets_sha256=s, n=len(f)) for n, k, f, s in runs]
    if len({c['name'] for c in clips}) != len(clips):
        raise ValueError('run names must be unique')
    if prev is not None:
        if prev.get('session_sha256') != session_sha or prev.get('clips') != clips:
            raise ValueError('previous schedule: other session or other clips (names, kinds, dets sha, order)')
        if u_so is None:
            raise ValueError('a step after the first needs --u-so (c7_kiem_cheo phan_tich --tap-moc <prev tap_moc.json>)')
    old = prev['chuoi'] if prev else {c: empty for c in ctx.chains}
    if u_so is not None:
        check_u_so_source(u_so, session, session_sha, ctx)
    nguong = u_so_thresholds(u_so, ctx.chains) if u_so is not None else None
    tap = {c: {tuple(S) for S in (prev or {}).get('tap_moc', {}).get(c, [])} for c in ctx.chains}
    canh_bao = []
    if prev is not None:
        if not (u_so.get('chi_tiet') or {}).get('co_tap_luot'):
            raise ValueError('u_so.json was computed without --tap-moc: run c7_kiem_cheo phan_tich with the previous '
                             'step\'s tap_moc.json (v3 3 order)')
        miss = _tap_cover_missing(prev.get('tap_moc', {}), u_so, ctx.groups)
        if miss:
            canh_bao.append(dict(loai='u_so_chua_tinh_tap', n=len(miss), mau=[list(m[:2]) + [list(m[2])] for m in miss[:20]]))
    per_chain = {c: [] for c in ctx.chains}
    caches = {}
    for q, (name, kind, frames, _) in enumerate(runs):
        cache = caches[name] = _load_cache(cache_dir, name, session_sha, runs[q][3]) if cache_dir else {}
        n = len(frames)
        dd = {c: {j: np.full((n, 2), np.nan) for j in ctx.universe[c]} for c in ctx.chains}
        for e, (idx, t, d) in enumerate(frames):
            excl = {c: md.excluded_at(old[c], q, idx) for c in ctx.chains}
            fr = ctx.frame(d, excl, deltas=u_so is not None, cache=cache, fkey=idx)
            for c in ctx.chains:
                tap[c].add(tuple(fr['neo'][c]))
                for j, v in fr['delta'].get(c, {}).items():
                    dd[c][j][e] = v
            if progress and e % 500 == 0:
                progress(f'{name} {e}/{n}')
        tt = np.array([f[1] for f in frames])
        ii = np.array([f[0] for f in frames], int)
        for c in ctx.chains:
            per_chain[c].append(md.Clip(name, kind, tt, dd[c], ii))
    giai_thich = []
    if u_so is not None:
        new = {}
        for c in ctx.chains:
            T_c, Tr_c = nguong[c]['nguong_m'], nguong[c]['nguong_tho_m']
            S = md.schedule(per_chain[c], T_c, Tr_c, ctx.universe[c])
            _explain_base_changes(ctx, runs, old, c, S, per_chain[c], T_c, Tr_c, caches, giai_thich)
            new[c] = S
    else:
        new = {c: empty for c in ctx.chains}
    if cache_dir:
        for name, cache in caches.items():
            _save_cache(cache_dir, name, session_sha, cache, _dets_sha_of(runs, name))
    for q, (name, kind, frames, _) in enumerate(runs):           # anchor sets the new schedule will use (no fit)
        for idx, t, d in frames:
            seen = [i for i in d if i in ctx.gmap]
            for c in ctx.chains:
                ex = md.excluded_at(new[c], q, idx)
                tap[c].add(tuple(sorted(i for i in seen if i in ctx.universe[c] and i not in ex)))
    tap_json = {c: sorted([list(S) for S in tap[c] if S], key=lambda s: (len(s), s)) for c in ctx.chains}
    tap_sha = md.schedule_sha256(tap_json)
    body = dict(chuoi=new, nguong_dung=nguong, tap_moc_sha256=tap_sha)
    sha = md.schedule_sha256(body)
    lap = list((prev or {}).get('lap', [])) + [dict(lich_sha256=sha, loai={c: sorted(new[c]['loai']) for c in ctx.chains},
                                                    n_moc_nghi={c: len(new[c]['moc_nghi']) for c in ctx.chains},
                                                    n_nen_doi={c: len(new[c]['nen_doi']) for c in ctx.chains},
                                                    u_so_sha256=u_so_sha)]
    # Codex C33-5: a repeated schedule is FINAL only when u_so computed every comparator set the schedule uses
    thieu_tap = _tap_cover_missing(tap_json, u_so, ctx.groups) if u_so is not None else []
    if thieu_tap:
        canh_bao.append(dict(loai='u_so_chua_tinh_tap_lich_moi', n=len(thieu_tap),
                             mau=[list(m[:2]) + [list(m[2])] for m in thieu_tap[:20]]))
    lap_lai = prev is not None and u_so is not None and prev.get('lich_sha256') == sha
    hoi_tu = lap_lai and not thieu_tap
    chu_trinh = not hoi_tu and sha in [h['lich_sha256'] for h in lap[:-2]]
    return dict(version='c7_lich_v1', session_sha256=session_sha, u_so_sha256=u_so_sha, clips=clips, chuoi=new,
                giai_thich_nen_doi=giai_thich,
                nguong_dung=nguong, tap_moc=tap_json, tap_moc_sha256=tap_sha, lich_sha256=sha, lap=lap, hoi_tu=hoi_tu,
                chu_trinh=chu_trinh, qua_10=(len(lap) > 10 and not hoi_tu), canh_bao=canh_bao,
                lap_lai=lap_lai, thieu_tap_u_so=len(thieu_tap),
                ket_luan=('hoi tu' if hoi_tu else 'lich lap lai nhung u_so thieu tap: chay c7_kiem_cheo phan_tich '
                          '--tap-moc <tap_moc.json nay> roi lich tiep' if lap_lai else
                          'chu trinh: chua ket luan' if chu_trinh else
                          'qua 10 lan: chua ket luan' if len(lap) > 10 else 'chua hoi tu: chay tiep'),
                luat=dict(win_s=md.WIN_S, win_min_n=md.WIN_MIN_N, win_gap_s=md.WIN_GAP_S, nen_win_s=md.NEN_WIN_S,
                          nen_raw_frac=md.NEN_RAW_FRAC, nen_raw_min=md.NEN_RAW_MIN, kiem_h_min=md.KIEM_H_MIN,
                          version=md.SCHEDULE_VERSION))


def _chain_clips(ctx, runs, old, c, extra, caches):
    """[md.Clip] of chain c: deltas with the exclusions of the schedule old, plus extra {mark: (q, f) from} out of the
    anchors of chain c."""
    out = []
    for q, (name, kind, frames, _) in enumerate(runs):
        n = len(frames)
        dd = {j: np.full((n, 2), np.nan) for j in ctx.universe[c]}
        for e, (idx, t, d) in enumerate(frames):
            excl = {cc: md.excluded_at(old[cc], q, idx) for cc in ctx.chains}
            excl[c] = excl[c] | {a for a, p in extra.items() if md._le(p, (q, idx))}
            for j, v in ctx.frame(d, excl, deltas=True, cache=caches.get(name), fkey=idx)['delta'].get(c, {}).items():
                dd[j][e] = v
        out.append(md.Clip(name, kind, np.array([f[1] for f in frames]), dd, np.array([f[0] for f in frames], int)))
    return out


def _peak(clips, a, ep):
    """Largest |delta| of mark a inside the base-change episode ep."""
    best = 0.0
    for q, cl in enumerate(clips):
        if a not in cl.d:
            continue
        for e, f in enumerate(cl.idx):
            p = (q, int(f))
            if md._le(ep['tu'], p) and (ep['den'] is None or md._key(p) < md._key(ep['den'])):
                v = float(np.linalg.norm(cl.d[a][e]))
                if v == v:
                    best = max(best, v)
    return best


def _explain_base_changes(ctx, runs, old, c, S, clips, T_c, Tr_c, caches, log):
    """INFORMATION ONLY (Codex C33-9: 'largest |delta|' and 'fewer episodes' do not prove which mark moved; a good mark
    can be picked). One shifted ANCHOR moves the comparators of other marks too (20 mm on one mark: 1.7 mm on another,
    synthetic, test (ii)), so a single shift can look like two alarms: the episode stays a nen_doi (not concluded).
    For every base-change episode of chain c the alarmed mark with the largest |delta| inside it is taken out of the
    chain's anchors from the episode start and the deltas recomputed; the result (would the rule then exclude that mark
    with fewer episodes?) is logged in giai_thich_nen_doi as a hint for re-measuring that mark's tape position. The
    schedule is NOT changed."""
    tried = set()
    while True:
        cand = None
        for ep in S['nen_doi']:
            ms = [a for a in ep['moc'] if a not in tried]
            if ms:
                cand = (ep, ms)
                break
        if cand is None:
            return S, clips
        ep, ms = cand
        a = max(ms, key=lambda m: (_peak(clips, m, ep), -m))
        tried.add(a)
        clips2 = _chain_clips(ctx, runs, old, c, {a: tuple(ep['tu'])}, caches)
        S2 = md.schedule(clips2, T_c, Tr_c, ctx.universe[c])
        ok = str(a) in S2['loai'] and len(S2['nen_doi']) < len(S['nen_doi'])
        log.append(dict(chuoi=c, moc=int(a), tu=ep['tu'], dinh_m=_peak(clips, a, ep), mot_moc_giai_thich=bool(ok),
                        nghia='chi thong tin, lich khong doi (Codex C33-9)'))


def _cache_path(cache_dir, name):
    return os.path.join(cache_dir, f'pose_cache_{name}.pkl')


CACHE_VERSION = 'c7_pose_cache_v2'


def _cache_tag(session_sha, dets_sha):
    """What a cached pose depends on (Codex C33-4): the session, the detections of the clip, the fit rule."""
    return dict(version=CACHE_VERSION, session_sha256=session_sha, dets_sha256=dets_sha, cond_max=COND_MAX,
                camz_tol_m=CAMZ_TOL_M, min_train=MIN_TRAIN)


def _dets_sha_of(runs, name):
    return {r[0]: r[3] for r in runs}[name]


def _load_cache(cache_dir, name, session_sha, dets_sha):
    """The pose cache of a clip, or {} when anything it depends on differs (never another clip's poses)."""
    import pickle
    p = _cache_path(cache_dir, name)
    if not os.path.isfile(p) or not dets_sha:
        return {}
    with open(p, 'rb') as f:
        obj = pickle.load(f)                                  # our own file (written by _save_cache)
    return obj.get('cache', {}) if obj.get('tag') == _cache_tag(session_sha, dets_sha) else {}


def _save_cache(cache_dir, name, session_sha, cache, dets_sha):
    import pickle
    if not dets_sha:
        return
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, name), 'wb') as f:
        pickle.dump(dict(tag=_cache_tag(session_sha, dets_sha), cache=cache), f, protocol=4)


FRAME_COLS = ('idx', 't_s', 'valid', 'why', 'n_marks', 'rms_px', 'shift_px', 'cam_z', 'car_seen', 'x', 'y', 'psi',
              'tag_x', 'tag_y', 'res_m', 'res_sim_m', 'scale', 'scale_ref', 'row_px', 'vx', 'vy', 'wz', 'speed',
              'shift_mag_px', 'col_px')      # col_px: C8b v4 3 (17/11); a file without it reads NaN
FINE_COLS = ('res_m', 'res_sim_m', 'scale', 'scale_ref')     # written with 10 decimals (CS1-17); older files: 6
# columns of a 33-mark track (session with 'kiem_cheo'), after FRAME_COLS; per chain / group columns follow the session
EXTRA_FIXED = ('khop_ok', 'khop_ly_do', 'cond', 'moc_thay', 'moc_dung', 'nen_doi', 'bat_dinh_moc', 'chua_ket_luan_xe',
               'U_moc_xy_m', 'U_moc_psi_rad', 'do_nhay_xy', 'do_nhay_psi')
TEXT_COLS = ('khop_ly_do', 'moc_thay', 'moc_dung', 'nen_doi', 'bat_dinh_moc')


def extra_cols(rows):
    """The 33-mark columns present in rows (none for a track of an old session): EXTRA_FIXED, moc_neo_<chain>, kc_*_k."""
    if not rows or 'moc_thay' not in rows[0]:
        return ()
    keys = rows[0].keys()
    neo = [k for k in keys if k.startswith('moc_neo_')]
    ks = sorted({int(k.rsplit('_', 1)[1]) for k in keys if k.startswith('kc_ok_')})
    kc = [f'kc_{n}_{k}' for k in ks for n in ('ok', 'n', 'err_max', 'ly_do')]
    return tuple(EXTRA_FIXED) + tuple(neo) + tuple(kc)


def write_frames(path, rows):
    """FINE_COLS with 10 decimals, or with the decimals the row was read with (read_frames '_dec'; CS2-7: a regate
    must not write a rounded value with more digits than it holds). The 33-mark columns (extra_cols) follow FRAME_COLS."""
    cols = FRAME_COLS + extra_cols(rows)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            out = []
            dec = r.get('_dec') or {}
            for c in cols:
                v = r.get(c)
                if isinstance(v, str):
                    pass
                elif c == 'why':
                    v = '|'.join(v)
                elif isinstance(v, bool):
                    v = int(v)
                elif v is None or (isinstance(v, float) and math.isnan(v)):
                    v = ''
                elif isinstance(v, float):
                    v = f'{v:.{dec.get(c, 10)}f}' if c in FINE_COLS else f'{v:.6f}'
                out.append(v)
            w.writerow(out)


def _decimals(s):
    """Decimal places a number cell resolves: the digits after the point, minus the exponent of a scientific form
    ('3.410e-3' resolves 6 places, CS3-3); never below 0."""
    s = s.strip().lower()
    mant, _, exp = s.partition('e')
    d = len(mant.split('.', 1)[1]) if '.' in mant else 0
    if exp:
        try:
            d -= int(exp)
        except ValueError:
            return None
    return max(d, 0)


def read_frames(path):
    """Rows of frames.csv as dicts with numbers (NaN for empty), valid as bool; '_dec': the decimals of the
    non-empty FINE_COLS cells (regate rounding, CS2-7)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            o = {}
            dec = {}
            for c in FRAME_COLS:
                v = r.get(c, '')
                if c == 'why':
                    o[c] = [x for x in v.split('|') if x]
                elif c in ('valid', 'car_seen'):
                    o[c] = v == '1'
                elif c in ('idx', 'n_marks'):
                    o[c] = int(v)
                else:
                    o[c] = float(v) if v != '' else float('nan')
                    if c in FINE_COLS and v != '' and _decimals(v) is not None:
                        dec[c] = _decimals(v)
            for c in r:                                   # 33-mark columns: kept as text, written back as read
                if c not in FRAME_COLS and c is not None:
                    o[c] = (r[c] == '1') if c == 'khop_ok' else r[c]     # khop_ok drives the gate 'khop' (regate)
            o['_dec'] = dec
            rows.append(o)
    return rows


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _pair(s, n, name):
    try:
        v = [float(x) for x in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f'{name}: {n} numbers separated by commas')
    if len(v) != n:
        raise argparse.ArgumentTypeError(f'{name}: {n} numbers separated by commas')
    return v


def _new_dir(path):
    if os.path.isdir(path) and os.listdir(path):
        raise ValueError(f'{path} already holds files: use a new folder')
    os.makedirs(path, exist_ok=True)


def calib_report(s):
    L = [f'Hiệu chuẩn C7 từ clip H {s["video"]} ({s["width"]}x{s["height"]}, {s["frames"]} khung)',
         f'Mô hình chọn: {s["chosen"]}; chiếu lại {s["rms_px_H"]:.2f} px RMS; Z_c thước {s["zc_tape_m"]} m, khớp '
         f'{s["zc_fit_m"]:.3f} m; κ = {s["kappa"]:.5f} ({s["kappa_src"]})']
    for m in cp.MODELS:
        fi, cv = s['model_fits'][m], s['cross_validation_mm'][m]
        cvs = f'kiểm chéo {cv["rms_mm"]:.1f} mm RMS (lớn nhất {cv["max_mm"]:.1f})' if cv else 'kiểm chéo: không đủ mốc'
        L.append(f'  {m}: chiếu lại {fi["rms_px"]:.2f} px, f {fi["f_px"]:.0f} px, méo ở góc khung '
                 f'{fi["corner_distortion"]:.1%}; {cvs}')
    rings = {}
    for p in s['per_mark']:
        rings.setdefault(p['ring'], []).append(p['rms_px'])
    L.append('Dư thừa theo vòng mốc (px RMS, design 7 mục 7): ' +
             ', '.join(f'{k} {float(np.sqrt(np.mean(np.square(v)))):.2f}' for k, v in rings.items()))
    L.append(f"Mốc sàn vào khớp: {len(s['fit_ids'])} ({', '.join(str(i) for i in s['fit_ids'])})")
    if s['check_tags_err_mm']:
        L.append('Mã có tọa độ nhưng ngoài khớp (vai_tro khác mốc sàn): ' +
                 ', '.join(f'{k}: {v:.1f}' for k, v in s['check_tags_err_mm'].items()) + ' mm')
    L.append(f'Lệch mã nóc xe: {s["tag_offset"]} ({s["tag_offset_src"]})')
    return '\n'.join(L) + '\n'


KHUNG_COLS = ('clip', 'khung', 'moc', 'nhom', 'chuoi', 'ex', 'ey', 'x_doan', 'y_doan', 'ngoai_suy_m', 'chua_ket_luan')


def track_cli_chains(a, sess, off, gates, gsrc):
    """track of a session with 'kiem_cheo' (33 marks): dets.npz, frames.csv, kiem_cheo_khung.csv, track.json."""
    sess_sha = sha256(a.session)
    lich = u_so = u_sha = None
    if a.lich:
        if not a.run or not a.u_so or not a.dets:
            raise ValueError('--lich needs --run, --u-so and --dets (the dets.npz the schedule was built on)')
        with open(a.lich, encoding='utf-8') as f:
            lich = json.load(f)
        if lich.get('session_sha256') != sess_sha:
            raise ValueError('lich.json is of another session')
        if not lich.get('hoi_tu') and not a.thu:
            raise ValueError(f"lich.json has not converged ({lich.get('ket_luan')}): run the lich steps on, or --thu")
    elif a.u_so:
        raise ValueError('--u-so only with --lich (final pass)')
    if a.u_so:
        with open(a.u_so, encoding='utf-8') as f:
            u_so = json.load(f)
        u_sha = sha256(a.u_so)
        if lich is not None and lich.get('u_so_sha256') != u_sha:
            raise ValueError('u_so.json is not the one the schedule was built with (sha256)')
    shapes = None
    if a.dets:
        frames, vsha, shape = load_dets(a.dets)
        shapes = [shape]
        if a.video and sha256(a.video) != vsha:
            raise ValueError('--video is not the clip of --dets (sha256)')
        if lich is not None:
            names = [c['name'] for c in lich['clips']]
            if a.run in names and lich['clips'][names.index(a.run)]['dets_sha256'] != sha256(a.dets):
                raise ValueError(f'--dets is not the detections run {a.run} of the schedule was built on (sha256)')
        vname = os.path.basename(a.video) if a.video else None
    else:
        if not a.video:
            raise ValueError('--video or --dets needed')
        frames, sh = detect_video(a.video, a.max_frames, progress=lambda k: print(f'... khung {k}', file=sys.stderr))
        shapes = sorted(sh)
        vsha, vname = sha256(a.video), os.path.basename(a.video)
        shape = shapes[0] if shapes else (0, 0)
    if a.max_frames is not None:
        frames = frames[:a.max_frames]
    if a.dets:
        dets_sha = sha256(a.dets)                       # the schedule names this file (the zip is not byte-stable)
    else:
        save_dets(os.path.join(a.out, 'dets.npz'), frames, vsha, shape)
        dets_sha = sha256(os.path.join(a.out, 'dets.npz'))
    cache = _load_cache(a.cache, a.run or vname or 'clip', sess_sha, dets_sha) if a.cache else None
    rows, khung, info = track_chains(frames, vname, vsha, sess, off, gates, gsrc, lich, a.run, u_so, u_sha, cache,
                                     shapes, a.thu or a.max_frames is not None)
    if a.cache:
        _save_cache(a.cache, a.run or vname or 'clip', sess_sha, cache, dets_sha)
    fp, kp = os.path.join(a.out, 'frames.csv'), os.path.join(a.out, 'kiem_cheo_khung.csv')
    write_frames(fp, rows)
    with open(kp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KHUNG_COLS)
        w.writeheader()
        for r in khung:
            if all(math.isfinite(r[k]) for k in ('ex', 'ey', 'x_doan', 'y_doan')):
                w.writerow({k: (f'{r[k]:.9f}' if isinstance(r[k], float) else r[k]) for k in KHUNG_COLS})
    inc = info['n_chua_ket_luan_xe_hop_le'] + info['n_hang_kiem_chua_ket_luan']
    info['ket_luan'] = ('lan_dau (cham_lai 1, chua co lich)' if lich is None else
                        'chua_ket_luan (nen_doi / bat_dinh_moc tren khung hop le hay hang kiem; Codex C33-1)' if inc else
                        'thu (lich chua hoi tu)' if info['cham_lai'] != 2 else 'ket_luan')
    info.update(session=os.path.abspath(a.session), session_sha256=sess_sha, frames_sha256=sha256(fp),
                kiem_cheo_khung_sha256=sha256(kp), dets_sha256=dets_sha, dets_file=os.path.abspath(a.dets) if a.dets
                else os.path.abspath(os.path.join(a.out, 'dets.npz')),
                lich_file=os.path.abspath(a.lich) if a.lich else None)
    with open(os.path.join(a.out, 'track.json'), 'w', encoding='utf-8') as f:
        json.dump(_json(info), f, ensure_ascii=False, indent=1)
    print(f'{info["frames"]} khung, {info["n_valid"]} hợp lệ, {info["n_khop_khong_tin"]} khớp không tin, '
          f'{info["n_chua_ket_luan_xe"]} chưa kết luận (chuỗi xe), {info["n_hang_kiem_chua_ket_luan"]} hàng kiểm chưa '
          f'kết luận; cham_lai {info["cham_lai"]}; {info["ket_luan"]}')
    return 3 if (lich is not None and inc) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('calib')
    a1.add_argument('--video', required=True)
    a1.add_argument('--markers', nargs='+', required=True, help='CSVs id, x_cm, y_cm, canh_den_mm (taped centres)')
    a1.add_argument('--zc', type=float, required=True, help='lens height above the floor from the tape [m]')
    a1.add_argument('--plumb', required=True, help='X,Y of the plumb point of the 0.5x lens on the floor [m]')
    a1.add_argument('--tag-h', type=float, required=True, help='height of the car tag face above the floor [m]')
    a1.add_argument('--car-side', type=float, default=None, help='black side of the car tag [m] (default: CSV ID 0)')
    a1.add_argument('--stride', type=int, default=1)
    a1.add_argument('--max-frames', type=int, default=None)
    a1.add_argument('--model', default=None, choices=cp.MODELS,
                    help='LOCKED camera model (33-mark chain, v3 1): session model_khoa + block kiem_cheo')
    a1.add_argument('--groups', type=int, default=KC_GROUPS, help='groups of the check by rotation (v3 2)')
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('track')
    a2.add_argument('--video', default=None)
    a2.add_argument('--dets', default=None, help='dets.npz of an earlier track of the same clip (no decoding)')
    a2.add_argument('--session', required=True)
    a2.add_argument('--tag-offset', default=None, help='DX,DY,DPSI_DEG (m, m, deg), from the static test')
    a2.add_argument('--max-frames', type=int, default=None)
    a2.add_argument('--hang-so', default=None, help='constants file of c7_score: frame-gate thresholds (cong_khung)')
    a2.add_argument('--lich', default=None, help='lich.json of the lich command (converged): the final pass')
    a2.add_argument('--run', default=None, help='name of this clip in lich.json')
    a2.add_argument('--u-so', dest='u_so', default=None, help='u_so.json of c7_kiem_cheo (the one lich.json names)')
    a2.add_argument('--cache', default=None, help='folder of the pose cache (shared with the lich steps)')
    a2.add_argument('--thu', action='store_true', help='trial: accept a schedule that has not converged (cham_lai 1)')
    a2.add_argument('--out', required=True)
    a4 = sub.add_parser('lich')
    a4.add_argument('--session', required=True)
    a4.add_argument('--luot', action='append', required=True,
                    help='NAME=<track dir with dets.npz>[:H], every clip of the session IN ORDER (thu_tu_clip.csv); :H '
                         'for a clip H (it can close a nen_doi)')
    a4.add_argument('--u-so', dest='u_so', default=None, help='u_so.json of c7_kiem_cheo (from the previous tap_moc)')
    a4.add_argument('--truoc', default=None, help='lich.json of the previous step')
    a4.add_argument('--cache', default=None, help='folder of the pose cache')
    a4.add_argument('--out', required=True)
    a3 = sub.add_parser('regate')
    a3.add_argument('--frames', required=True, help='frames.csv written by track (this or an older version)')
    a3.add_argument('--session', required=True)
    a3.add_argument('--hang-so', default=None, help='constants file of c7_score: frame-gate thresholds (cong_khung)')
    a3.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        gates, gsrc = None, None
        if getattr(a, 'hang_so', None):
            gates, gsha, gstate = load_gates(a.hang_so)
            gsrc = dict(file=os.path.basename(a.hang_so), sha256=gsha, trang_thai=gstate)
        _new_dir(a.out)
        if a.cmd == 'calib':
            plumb = _pair(a.plumb, 2, '--plumb')
            marks = cp.load_marks(a.markers)
            floor = cp.floor_ids(marks, cp.load_roles(a.markers))
            side = a.car_side
            if side is None:
                for p in a.markers:
                    with open(p, newline='', encoding='utf-8') as f:
                        for r in csv.DictReader(f):
                            if r.get('id') == str(CAR_ID):
                                side = float(r['canh_den_mm']) / 1000
            if side is None:
                raise ValueError('car tag side: give --car-side or a CSV with ID 0')
            s = calibrate(a.video, marks, a.zc, plumb, a.tag_h, side, a.stride, a.max_frames, floor=floor,
                          model=a.model, n_groups=a.groups, roles=cp.load_roles(a.markers))
            s['markers_csv'] = [dict(file=os.path.basename(p), sha256=sha256(p)) for p in a.markers]
            with open(os.path.join(a.out, 'session.json'), 'w', encoding='utf-8') as f:
                json.dump(_json(s), f, ensure_ascii=False, indent=1)
            with open(os.path.join(a.out, 'calib_marks.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(s['per_mark'][0]))
                w.writeheader()
                w.writerows(s['per_mark'])
            txt = calib_report(s)
            open(os.path.join(a.out, 'report.txt'), 'w', encoding='utf-8').write(txt)
            print(txt)
        elif a.cmd == 'regate':
            with open(a.session, encoding='utf-8') as f:
                sess = json.load(f)
            rows = read_frames(a.frames)
            old_shift = all(not np.isfinite(r['shift_mag_px']) for r in rows)
            med = regate(rows, sess, gates)
            write_frames(os.path.join(a.out, 'frames.csv'), rows)
            info = dict(regate_of=os.path.abspath(a.frames), regate_of_sha256=sha256(a.frames),
                        session=os.path.abspath(a.session), session_sha256=sha256(a.session), frames=len(rows),
                        median_scale=med, gates=gates_dict(gates, gsrc), n_valid=int(sum(r['valid'] for r in rows)),
                        n_can_track_lai=int(sum('can_track_lai' in r['why'] for r in rows)),
                        shift_kind='median_magnitude' if old_shift else 'median_vector',
                        frames_sha256=sha256(os.path.join(a.out, 'frames.csv')))
            with open(os.path.join(a.out, 'track.json'), 'w', encoding='utf-8') as f:
                json.dump(_json(info), f, ensure_ascii=False, indent=1)
            print(f'{info["frames"]} khung, {info["n_valid"]} hợp lệ, {info["n_can_track_lai"]} cần track lại '
                  f'(cổng chấm lại, không đọc video)')
        elif a.cmd == 'lich':
            sess_sha = sha256(a.session)
            with open(a.session, encoding='utf-8') as f:
                sess = json.load(f)
            runs = []
            for spec in a.luot:
                kind = 'luot'
                if spec.endswith(':H'):
                    spec, kind = spec[:-2], 'H'
                name, _, d = spec.partition('=')
                if not name or not d:
                    raise ValueError(f'--luot {spec!r}: NAME=<track dir>[:H]')
                p = os.path.join(d, 'dets.npz')
                frames, _, _ = load_dets(p)
                runs.append((name, kind, frames, sha256(p)))
            u_so = u_sha = prev = None
            if a.u_so:
                with open(a.u_so, encoding='utf-8') as f:
                    u_so = json.load(f)
                u_sha = sha256(a.u_so)
            if a.truoc:
                with open(a.truoc, encoding='utf-8') as f:
                    prev = json.load(f)
            lich = build_lich(sess, sess_sha, runs, u_so, u_sha, prev, a.cache,
                              progress=lambda s: print('...', s, file=sys.stderr))
            with open(os.path.join(a.out, 'tap_moc.json'), 'w', encoding='utf-8') as f:
                json.dump(lich['tap_moc'], f, ensure_ascii=False)
            lich['tap_moc_file_sha256'] = sha256(os.path.join(a.out, 'tap_moc.json'))
            with open(os.path.join(a.out, 'lich.json'), 'w', encoding='utf-8') as f:
                json.dump(_json(lich), f, ensure_ascii=False, indent=1)
            print(f"lich buoc {len(lich['lap'])}: {lich['ket_luan']}; loai " +
                  ', '.join(f"{c}: {sorted(v['loai'])}" for c, v in lich['chuoi'].items()) +
                  f"; canh bao {lich['canh_bao']}")
        else:
            with open(a.session, encoding='utf-8') as f:
                sess = json.load(f)
            off = None
            if a.tag_offset:
                dx, dy, dpsi = _pair(a.tag_offset, 3, '--tag-offset')
                off = (dx, dy, math.radians(dpsi))
            if sess.get('kiem_cheo'):
                code = track_cli_chains(a, sess, off, gates, gsrc)
                print(f'#ket_qua,ma={code}')               # 3: final pass with frames not concluded (Codex C33-1)
                return code
            if a.video is None:
                raise ValueError('--video needed (a session without kiem_cheo reads the video)')
            rows, info = track(a.video, sess, off, a.max_frames,
                               progress=lambda k: print(f'... khung {k}', file=sys.stderr), gates=gates, gates_src=gsrc)
            info.update(session=os.path.abspath(a.session), session_sha256=sha256(a.session))
            write_frames(os.path.join(a.out, 'frames.csv'), rows)
            info['frames_sha256'] = sha256(os.path.join(a.out, 'frames.csv'))   # c7_calib_drift CD2-3
            with open(os.path.join(a.out, 'track.json'), 'w', encoding='utf-8') as f:
                json.dump(_json(info), f, ensure_ascii=False, indent=1)
            print(f'{info["frames"]} khung, {info["n_valid"]} hợp lệ; lệch độ cao camera lớn nhất '
                  f'{info["cam_z_drift_rel"]}')
    except (ValueError, OSError, KeyError, argparse.ArgumentTypeError) as exc:
        print('c7_measure:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
