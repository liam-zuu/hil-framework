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
            distortion from clip H (design 3.6, vibration compensation); shift_px = median distance
            of the mark centres from clip H, rms_px = reprojection RMS of the marks after the refit
  car       the 4 corners of the car tag (ID 0) -> floor plane z = 0 with the frame camera -> parallax
            P = P' + kappa (C - P'), kappa = h / Z_c from the tape, C = PLUMB POINT measured with a
            plumb line (not the camera centre of the fit, design 3.4) -> rigid fit (Procrustes) to the
            square of the measured side: tag centre, heading, fit residual, apparent scale -> wheel
            centre = tag centre - R(psi) (dx, dy), heading - dpsi (tag offset, from the static test,
            design 3.5; 0 until measured, the session says which)
  gates     frame VALID (O1 3.3 v11.4 O-2) only if (1) at least 8 floor marks and rms_px <= rms of
            clip H + 0.5 px; (2) car tag fit residual <= 1.5 mm, apparent scale within 0.5 % of the
            median of the clip, centre within 20 mm of a quadratic predicted from the valid frames
            around it. Invalid frames are lost frames (no interpolation, design 3.7, 3.8)
  speed     quadratic fit over +-6 valid frames on the real frame times (design 3.8), for the stop
            condition and the clock matching (C7b); NaN on invalid frames
All thresholds are [DE XUAT] and are locked at the measurement-system trial C8 (O1 3.3, 3.9 item 4).
The run rules (gaps, tracker, completion, outcome) are in c7_score.py.

  python tools/c7_measure.py calib --video <clip H> --markers <csv...> --zc 2.60 --plumb X,Y \
      --tag-h 0.120 --out <dir>      -> <dir>/session.json, calib_marks.csv, report.txt
  python tools/c7_measure.py track --video <run clip> --session <dir>/session.json --out <dir2>
                                     -> <dir2>/frames.csv, track.json
  [--tag-offset DX,DY,DPSI_DEG]  tag centre minus wheel centre in the body frame (m, m, deg) and tag
                                 heading minus body heading, from the static test (C8 clip T)
Last line: #ket_qua,ma=0 done / 2 error.
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

CAR_ID = 0
FLOOR_IDS = tuple(range(1, 25))
CHECK_IDS = tuple(range(25, 34))
HALF_FLOOR_PX = 6.0          # design 3.2: half width of the edge profiles, floor marks
HALF_CAR_PX = 9.0            #             car tag (room for blur up to about 6 px)
SG_HALF = 6                  # design 3.8: +-6 frames (0.2 s at 60 fps)
# frame gates, O1 3.3 v11.4 (O-2) [DE XUAT, locked at C8]
GATE_MIN_MARKS = 8
GATE_RMS_EXTRA_PX = 0.5
GATE_CAR_RES_M = 0.0015
GATE_SCALE_REL = 0.005
GATE_PRED_M = 0.020
PRED_HALF = 6                # frames each side for the quadratic prediction
PRED_MIN = 4                 # valid neighbours needed, else the prediction gate is not applied


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
    corners, ids, _ = det.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for cc, i in zip(corners, ids.ravel()):
        c = cc.reshape(4, 2).astype(float)
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
    3.5). kappa = h / Z_c; plumb = (x, y) of the plumb point of the lens on the floor."""
    p_floor = cam.pix2floor(np.asarray(corners_px, float), 0.0)
    C = np.asarray(plumb, float)
    P = p_floor + kappa * (C - p_floor)
    body = car_corners_body(side)
    ang, t, _, res = rigid_fit(body, P)
    _, _, sc, _ = rigid_fit(body, P, with_scale=True)
    dx, dy, dpsi = tag_offset
    c, s = math.cos(ang), math.sin(ang)
    xw, yw = t[0] - (c * dx - s * dy), t[1] - (s * dx + c * dy)
    return dict(x=float(xw), y=float(yw), psi=float(cp_wrap(ang - dpsi)), tag_x=float(t[0]), tag_y=float(t[1]),
                tag_psi=float(ang), res_m=res, scale=sc)


def cp_wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------- gates and speed
def quad_predict(t, x, k, valid, half=PRED_HALF, nmin=PRED_MIN):
    """Value at t[k] of a quadratic fitted to the valid samples within +-half indices of k (k
    excluded); None when fewer than nmin."""
    lo, hi = max(0, k - half), min(len(t), k + half + 1)
    idx = [j for j in range(lo, hi) if j != k and valid[j]]
    if len(idx) < nmin:
        return None
    tt = np.asarray([t[j] for j in idx]) - t[k]
    c = np.polyfit(tt, np.asarray([x[j] for j in idx]), 2)
    return float(c[-1])


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


def apply_gates(rows, rms_h):
    """Sets valid / why on every frame row in place (O1 3.3 v11.4 O-2) and the speeds on the valid
    frames. Order: floor gate, car fit residual, scale against the clip median (on the frames that
    passed the first two), prediction against the valid neighbours (one pass)."""
    for r in rows:
        why = []
        if r['n_marks'] < GATE_MIN_MARKS:
            why.append('moc_it')
        elif not (r['rms_px'] <= rms_h + GATE_RMS_EXTRA_PX):
            why.append('moc_du_thua')
        if not r['car_seen']:
            why.append('khong_thay_xe')
        elif r['res_m'] > GATE_CAR_RES_M:
            why.append('xe_du_thua')
        r['why'] = why
    sc = [r['scale'] for r in rows if not r['why']]
    med = float(np.median(sc)) if sc else float('nan')
    for r in rows:
        if not r['why'] and abs(r['scale'] / med - 1.0) > GATE_SCALE_REL:
            r['why'].append('ti_le')
    ok = [not r['why'] for r in rows]
    t = [r['t_s'] for r in rows]
    xs, ys = [r['x'] for r in rows], [r['y'] for r in rows]
    for k, r in enumerate(rows):
        if not ok[k]:
            continue
        px, py = quad_predict(t, xs, k, ok), quad_predict(t, ys, k, ok)
        r['pred_m'] = None if px is None else math.hypot(r['x'] - px, r['y'] - py)
        if r['pred_m'] is not None and r['pred_m'] > GATE_PRED_M:
            r['why'].append('nhay_vi_tri')
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
    return med


# ---------------------------------------------------------------- calibration (clip H)
def calibrate(video, marks, zc, plumb, tag_h, car_side, stride=1, max_frames=None, min_frac=0.5):
    """Session camera from the static clip H (design 3.0, 3.3, 4.2 (a)). Returns the session dict."""
    det = make_detector()
    dets, shape, n = [], None, 0
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
    ids = [i for i in sorted(mean) if i in marks and i in FLOOR_IDS]
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
    chosen = cp.choose_model(cv)
    if chosen is None:
        raise ValueError('no camera model could be cross-validated')
    cam, rms = fit_two_pass(chosen, obj, img, w_px, h_px, zc)
    per_mark = []
    for i in ids:
        r = cam.project(np.column_stack([cp.mark_corners_floor(*marks[i]), np.zeros(4)])) - mean[i]
        ctr = cam.pix2floor(mean[i].mean(axis=0, keepdims=True))[0]
        per_mark.append(dict(id=i, ring='ngoai' if i >= 13 else 'trong', rms_px=float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))),
                             err_mm=float(np.hypot(*(ctr - np.array(marks[i][:2]))) * 1000), spread_px=spread[i]))
    check = {i: float(np.hypot(*(cam.pix2floor(mean[i].mean(axis=0, keepdims=True))[0] - np.array(marks[i][:2]))) * 1000)
             for i in sorted(mean) if i in marks and i in CHECK_IDS}
    zc_used = zc if zc is not None else float(abs(cam.centre[2]))
    return dict(
        kind='c7_session', video=os.path.basename(video), video_sha256=sha256(video), frames=n, stride=stride,
        width=w_px, height=h_px, camera=cam_to_dict(cam), rms_px_H=rms, model_fits=fits, cross_validation_mm=cv,
        chosen=chosen, fit_ids=ids, ref_centres_px={str(i): mean[i].mean(axis=0).tolist() for i in ids},
        per_mark=per_mark, check_tags_err_mm=check, zc_tape_m=zc, zc_fit_m=float(abs(cam.centre[2])),
        plumb_xy_m=list(plumb), tag_h_m=tag_h, car_side_m=car_side, kappa=tag_h / zc_used,
        kappa_src='thuoc: h / Z_c (design 3.4 cach 1)' if zc is not None else 'h / Z_c cua phep khop (KHONG dung cho so chinh)',
        tag_offset=[0.0, 0.0, 0.0], tag_offset_src='chua do (thu tinh clip T, C8): 0',
        marks={str(i): list(v) for i, v in marks.items()},
        gates=dict(min_marks=GATE_MIN_MARKS, rms_extra_px=GATE_RMS_EXTRA_PX, car_res_m=GATE_CAR_RES_M,
                   scale_rel=GATE_SCALE_REL, pred_m=GATE_PRED_M, status='DE XUAT, khoa o C8'))


# ---------------------------------------------------------------- tracking a run clip
def track(video, session, tag_offset=None, max_frames=None, progress=None):
    """Per-frame rows of a run clip with the session camera (design 3.5-3.8). Returns (rows, info)."""
    cam0 = cam_from_dict(session['camera'])
    marks = {int(i): tuple(v) for i, v in session['marks'].items()}
    ref = {int(i): np.array(v) for i, v in session['ref_centres_px'].items()}
    side, kappa, plumb = session['car_side_m'], session['kappa'], session['plumb_xy_m']
    off = tag_offset if tag_offset is not None else session['tag_offset']
    det = make_detector()
    rows, cam_prev, shapes = [], cam0, set()
    for k, t, g in iter_frames(video, max_frames):
        shapes.add(g.shape)
        d = detect_refined(det, g)
        obj, img, used = floor_obs(d, marks)
        r = dict(idx=k, t_s=t, n_marks=len(used), rms_px=float('nan'), shift_px=float('nan'), cam_z=float('nan'),
                 car_seen=False, x=float('nan'), y=float('nan'), psi=float('nan'), tag_x=float('nan'),
                 tag_y=float('nan'), res_m=float('nan'), scale=float('nan'), row_px=float('nan'),
                 pred_m=None, vx=float('nan'), vy=float('nan'), wz=float('nan'), speed=float('nan'))
        sh = [np.linalg.norm(d[i].mean(axis=0) - ref[i]) for i in used if i in ref]
        if sh:
            r['shift_px'] = float(np.median(sh))
        cam = None
        if len(used) >= 4:                                 # 4 marks = 16 points: the pose is determined
            cam, r['rms_px'] = refit_pose(cam_prev, obj, img)
            r['cam_z'] = float(cam.centre[2])
            cam_prev = cam
        if cam is not None and CAR_ID in d:
            r.update(car_pose(cam, d[CAR_ID], side, kappa, plumb, off))
            r['car_seen'] = True
            r['row_px'] = float(d[CAR_ID][:, 1].mean())
        rows.append(r)
        if progress and k % 100 == 0:
            progress(k)
    med_scale = apply_gates(rows, session['rms_px_H'])
    info = dict(video=os.path.basename(video), video_sha256=sha256(video), frames=len(rows),
                decoded_shapes=sorted(list(s) for s in shapes), median_scale=med_scale, tag_offset=list(off),
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


FRAME_COLS = ('idx', 't_s', 'valid', 'why', 'n_marks', 'rms_px', 'shift_px', 'cam_z', 'car_seen', 'x', 'y', 'psi',
              'tag_x', 'tag_y', 'res_m', 'scale', 'pred_m', 'row_px', 'vx', 'vy', 'wz', 'speed')


def write_frames(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(FRAME_COLS)
        for r in rows:
            out = []
            for c in FRAME_COLS:
                v = r.get(c)
                if c == 'why':
                    v = '|'.join(v)
                elif isinstance(v, bool):
                    v = int(v)
                elif v is None or (isinstance(v, float) and math.isnan(v)):
                    v = ''
                elif isinstance(v, float):
                    v = f'{v:.6f}'
                out.append(v)
            w.writerow(out)


def read_frames(path):
    """Rows of frames.csv as dicts with numbers (NaN for empty), valid as bool."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            o = {}
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
    if s['check_tags_err_mm']:
        L.append('Mã kiểm 25–33 (không vào khớp; tâm DANH ĐỊNH tới khi đo thước): ' +
                 ', '.join(f'{k}: {v:.1f}' for k, v in s['check_tags_err_mm'].items()) + ' mm')
    else:
        L.append('Mã kiểm 25–33: không thấy trong clip')
    L.append(f'Lệch mã nóc xe: {s["tag_offset"]} ({s["tag_offset_src"]})')
    return '\n'.join(L) + '\n'


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
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('track')
    a2.add_argument('--video', required=True)
    a2.add_argument('--session', required=True)
    a2.add_argument('--tag-offset', default=None, help='DX,DY,DPSI_DEG (m, m, deg), from the static test')
    a2.add_argument('--max-frames', type=int, default=None)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        _new_dir(a.out)
        if a.cmd == 'calib':
            plumb = _pair(a.plumb, 2, '--plumb')
            marks = cp.load_marks(a.markers)
            side = a.car_side
            if side is None:
                for p in a.markers:
                    with open(p, newline='', encoding='utf-8') as f:
                        for r in csv.DictReader(f):
                            if r.get('id') == str(CAR_ID):
                                side = float(r['canh_den_mm']) / 1000
            if side is None:
                raise ValueError('car tag side: give --car-side or a CSV with ID 0')
            s = calibrate(a.video, marks, a.zc, plumb, a.tag_h, side, a.stride, a.max_frames)
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
        else:
            with open(a.session, encoding='utf-8') as f:
                sess = json.load(f)
            off = None
            if a.tag_offset:
                dx, dy, dpsi = _pair(a.tag_offset, 3, '--tag-offset')
                off = (dx, dy, math.radians(dpsi))
            rows, info = track(a.video, sess, off, a.max_frames,
                               progress=lambda k: print(f'... khung {k}', file=sys.stderr))
            info.update(session=os.path.abspath(a.session), session_sha256=sha256(a.session))
            write_frames(os.path.join(a.out, 'frames.csv'), rows)
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
