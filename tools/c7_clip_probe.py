#!/usr/bin/env python3
"""
c7_clip_probe.py: diagnosis of the camera trial clip (C7, before the measurement-system trial C8).

L83 (Liam 11/11 CODE local_7787e22f, "duyet" the review recommendations of the Fable C7 design):
CLAUDE_THESIS 03_CODE/docs/review/2026-11-11_Fable_C7_thiet_ke_do_v1.md section 6.2 asks one clip
of 10-20 s (4K 60, 0.5x lens, focus and exposure locked, the phone hung where it will be, no car,
floor marks taped) and six answers from it. Liam 11/11 CODE local_b905f2ba: "viet script di".
This script gives them; it measures nothing for the thesis by itself.

  (1) marks seen: per ID the share of frames it is detected in, its mean centre, the distance of
      its corners to the frame edge (marks near the edge are flagged);
  (2) lens: fits the camera to the floor marks (corners, floor coordinates from the print CSV) with
      the models of design section 3.3: H (homography, no distortion), B1 (k1), B2 (k1, k2) and KB2
      (OpenCV fisheye, 2 coefficients); pinhole, one focal length, image centre fixed at the frame
      centre, 6-dof pose, the lens height Z_c from the tape as a prior (sigma 20 mm). Each model is
      also scored by 4-fold cross-validation over the marks (error in mm on the floor at the marks
      left out). The simplest model within 15 % of the best is chosen (order B1, B2, KB2) and the
      distortion at the frame corner is reported. Checkerboard rule (section 3.3): B1 or B2 chosen
      and distortion at the corner below about 10 % -> not needed; KB2 chosen -> needed;
  (3) does the frame drift while the phone hangs still (electronic stabilisation, swing): the
      median displacement of all mark centres against their median position, per frame;
  (4) frame times: presentation time (PTS) of every frame from CAP_PROP_POS_MSEC, steps against
      1/fps, dropped frames (a step of 1.5 periods or more);
  (5) can OpenCV read the file, size, codec, rotation flag (auto rotation is turned off: the
      parameters belong to the sensor pixels, design section 3.1);
  (6) exposure time: NOT read (no ffprobe or exiftool on the laptop, 11/11); stated as such.
It also writes the clip's size and SHA-256 (L88: video stays out of git, git keeps this table).

Corner refinement here is OpenCV CORNER_REFINE_SUBPIX: section 3.2 found it pulls corners about
0.2-0.5 px inward on blurred marks; enough for this diagnosis, NOT for the measurement pipeline.
Mark corners on the floor: centre from the CSV, black side from the CSV, top edge (ArUco corners
0 -> 1) along +x with the arrow to +y, as taped (guide 2026-11-10_Huong_dan_Dan_so8_va_moc_san.md).
The CSV centres are the NOMINAL ones until the taped centres are measured (design 6.1 item 6);
floor errors in mm then include the taping error (rule of thumb section 3.3: max ~0.75 sigma).

  python tools/c7_clip_probe.py --video <clip.MOV> --zc 2.60 \
      --markers <..>/2026-11-10_ArUco_in_C8.csv <..>/2026-11-10_ArUco_in_C8_vung_noi.csv \
      --out <clip folder>/probe_<clip name>
  [--stride N]  detect on every N-th frame (all frames are still read for their times)
  [--max-frames N]
Output in --out: report.txt (Vietnamese summary), report.json, marks.csv, frames.csv
(index, t_ms, n_marks, drift_px), clip_manifest.csv. Exit code 0 when the file was read.
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

DICT = cv2.aruco.DICT_4X4_50
EDGE_WARN_PX = 50.0          # corner closer than this to the frame edge: flagged (section 6.2 (1))
ZC_SIGMA_M = 0.02            # tape measurement of the lens height (section 3.3)
CV_FOLDS = 4
CV_TOL = 0.15                # simplest model within 15 % of the best cross-validated error
MODELS = ('B1', 'B2', 'KB2')
CORNER_DIST_LIMIT = 0.10     # distortion at the frame corner below this: no checkerboard (3.3 rule 1)


# ---------------------------------------------------------------- floor marks
def load_marks(paths):
    """{id: (x_m, y_m, side_m)} for rows with a centre (the car tag ID 0 has none)."""
    marks = {}
    for p in paths:
        with open(p, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('x_cm', '') in ('', None) or r.get('y_cm', '') in ('', None):
                    continue
                marks[int(r['id'])] = (float(r['x_cm']) / 100, float(r['y_cm']) / 100, float(r['canh_den_mm']) / 1000)
    return marks


def mark_corners_floor(x, y, side):
    """ArUco corner order 0..3 (top-left, top-right, bottom-right, bottom-left of the tag itself), the
    tag's top edge towards +y: floor coordinates (4, 2), metres."""
    h = side / 2
    return np.array([[x - h, y + h], [x + h, y + h], [x + h, y - h], [x - h, y - h]])


# ---------------------------------------------------------------- camera models
def _rot(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def distort(model, xn, yn, k):
    """Normalised pinhole coordinates -> distorted normalised coordinates."""
    if model == 'H' or model == 'P':
        return xn, yn
    r2 = xn * xn + yn * yn
    if model == 'B1':
        s = 1 + k[0] * r2
    elif model == 'B2':
        s = 1 + k[0] * r2 + k[1] * r2 * r2
    elif model == 'KB2':
        r = np.sqrt(r2)
        th = np.arctan(r)
        thd = th * (1 + k[0] * th ** 2 + k[1] * th ** 4)
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
    else:
        raise ValueError(model)
    return xn * s, yn * s


def undistort(model, xd, yd, k, iters=30):
    """Inverse of distort() by fixed-point iteration (the distortion of a phone lens is mild enough)."""
    xn, yn = np.array(xd, float), np.array(yd, float)
    for _ in range(iters):
        xx, yy = distort(model, xn, yn, k)
        xn = xn - (xx - xd)
        yn = yn - (yy - yd)
    return xn, yn


NK = {'P': 0, 'B1': 1, 'B2': 2, 'KB2': 2}


class Camera:
    """f [px], centre (cx, cy) fixed, model, k, rvec/tvec world -> camera (world z up, floor z = 0)."""

    def __init__(self, model, f, cx, cy, k, rvec, tvec):
        self.model, self.f, self.cx, self.cy = model, float(f), float(cx), float(cy)
        self.k, self.rvec, self.tvec = np.asarray(k, float), np.asarray(rvec, float), np.asarray(tvec, float)
        self.R = _rot(self.rvec)

    @property
    def centre(self):
        return -self.R.T @ self.tvec

    def project(self, P):
        Pc = (self.R @ np.asarray(P, float).T).T + self.tvec
        xn, yn = Pc[:, 0] / Pc[:, 2], Pc[:, 1] / Pc[:, 2]
        xd, yd = distort(self.model, xn, yn, self.k)
        return np.column_stack([self.f * xd + self.cx, self.f * yd + self.cy])

    def pix2floor(self, uv, z=0.0):
        uv = np.asarray(uv, float).reshape(-1, 2)
        xn, yn = undistort(self.model, (uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, self.k)
        d = (self.R.T @ np.column_stack([xn, yn, np.ones_like(xn)]).T).T
        C = self.centre
        s = (z - C[2]) / d[:, 2]
        return C[:2] + d[:, :2] * s[:, None]

    def corner_distortion(self, w, h):
        """Relative radial displacement of the frame-corner pixel: |r_d - r_u| / r_u, r_d its (distorted)
        normalised radius, r_u the pinhole radius of the same ray."""
        r_d = math.hypot(w / 2, h / 2) / self.f
        r_u, _ = undistort(self.model, np.array([r_d]), np.array([0.0]), self.k)
        return abs(r_d - r_u[0]) / r_u[0]


def initial_pose(obj_xy, img, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    obj = np.column_stack([obj_xy, np.zeros(len(obj_xy))]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError('solvePnP failed')
    return rvec.ravel(), tvec.ravel()


def fit_camera(model, obj_xy, img, w, h, zc=None, f0=None):
    """Least squares on the reprojection error (Huber), centre fixed, Z_c prior (sigma 20 mm) when given."""
    cx, cy = w / 2, h / 2
    if f0 is None:
        f0 = 0.36 * w                                 # about a 13 mm-equivalent lens on a 16:9 frame [GUESS]
    rv0, tv0 = initial_pose(obj_xy, img, f0, cx, cy)
    if zc is not None:                                # scale the pose so that the camera height is zc
        cam = Camera('P', f0, cx, cy, [], rv0, tv0)
        Z = cam.centre[2]
        if abs(Z) > 1e-6:
            f0 = f0 * zc / abs(Z)
            rv0, tv0 = initial_pose(obj_xy, img, f0, cx, cy)
    nk = NK[model]
    x0 = np.concatenate([[f0], np.zeros(nk), rv0, tv0])
    obj = np.column_stack([obj_xy, np.zeros(len(obj_xy))])

    def unpack(x):
        return Camera(model, x[0], cx, cy, x[1:1 + nk], x[1 + nk:4 + nk], x[4 + nk:7 + nk])

    def res(x):
        c = unpack(x)
        r = (c.project(obj) - img).ravel()
        if zc is not None:
            sigma_px = 0.3
            r = np.concatenate([r, [(abs(c.centre[2]) - zc) / ZC_SIGMA_M * sigma_px]])
        return r

    sol = least_squares(res, x0, loss='huber', f_scale=1.0, max_nfev=4000)
    cam = unpack(sol.x)
    rep = cam.project(obj) - img
    return cam, float(np.sqrt(np.mean(np.sum(rep ** 2, axis=1))))


def homography_check(obj_xy, img):
    """Plain homography floor -> image: reprojection RMS [px] and floor error at the points [mm]."""
    Hm, _ = cv2.findHomography(obj_xy.astype(np.float64), img.astype(np.float64), 0)
    pr = cv2.perspectiveTransform(obj_xy.reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)
    back = cv2.perspectiveTransform(img.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(Hm)).reshape(-1, 2)
    return (float(np.sqrt(np.mean(np.sum((pr - img) ** 2, axis=1)))),
            float(np.max(np.linalg.norm(back - obj_xy, axis=1)) * 1000))


def cross_validate(model, mark_ids, corners_img, marks, w, h, zc, folds=CV_FOLDS):
    """Leave a quarter of the marks out, fit on the rest, floor error [mm] of the left-out mark centres."""
    ids = sorted(mark_ids)
    errs = []
    for k in range(folds):
        test = ids[k::folds]
        train = [i for i in ids if i not in test]
        if len(train) < 4 or not test:
            continue
        obj = np.vstack([mark_corners_floor(*marks[i]) for i in train])
        img = np.vstack([corners_img[i] for i in train])
        cam, _ = fit_camera(model, obj, img, w, h, zc)
        for i in test:
            c = cam.pix2floor(corners_img[i].mean(axis=0, keepdims=True))[0]
            errs.append(float(np.hypot(*(c - np.array(marks[i][:2]))) * 1000))
    if not errs:
        return None
    return dict(rms_mm=float(np.sqrt(np.mean(np.square(errs)))), max_mm=float(np.max(errs)), n=len(errs))


def choose_model(cv):
    ok = {m: v for m, v in cv.items() if v is not None}
    if not ok:
        return None
    best = min(v['rms_mm'] for v in ok.values())
    for m in MODELS:
        if m in ok and ok[m]['rms_mm'] <= best * (1 + CV_TOL):
            return m
    return None


# ---------------------------------------------------------------- video
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_clip(path, stride=1, max_frames=None):
    """Reads every frame (times), detects marks on every stride-th. Returns info, times, detections."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return dict(opened=False), [], {}
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    info = dict(opened=True, backend=cap.getBackendName(), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps_nominal=float(cap.get(cv2.CAP_PROP_FPS)),
                frame_count_header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                fourcc=''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
                orientation_meta=float(cap.get(cv2.CAP_PROP_ORIENTATION_META)))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)
    times, dets, shapes = [], {}, set()
    k = 0
    while max_frames is None or k < max_frames:
        if k % stride == 0:
            ok, frame = cap.read()
        else:
            ok = cap.grab()
            frame = None
        if not ok:
            break
        times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
        if frame is not None:
            shapes.add(frame.shape[:2])
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            corners, ids, _ = det.detectMarkers(g)
            dets[k] = {} if ids is None else {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}
        k += 1
    cap.release()
    info['frames_read'] = len(times)
    info['decoded_shapes'] = sorted(list(s) for s in shapes)
    return info, times, dets


def frame_times(times, fps):
    t = np.asarray(times, float)
    if len(t) < 3:
        return dict(n=len(t))
    dt = np.diff(t)
    per = 1000.0 / fps if fps > 0 else float(np.median(dt))
    return dict(n=len(t), duration_s=float((t[-1] - t[0]) / 1000), dt_median_ms=float(np.median(dt)),
                dt_mean_ms=float(np.mean(dt)), dt_std_ms=float(np.std(dt)), dt_min_ms=float(np.min(dt)),
                dt_max_ms=float(np.max(dt)), period_ms=per,
                n_dropped=int(np.sum(dt >= 1.5 * per)), n_nonmonotonic=int(np.sum(dt <= 0)),
                n_off_25pct=int(np.sum(np.abs(dt - per) > 0.25 * per)))


def drift(dets):
    """Per frame: median over marks of |centre - that mark's median centre| [px]."""
    cen = {}
    for k, d in dets.items():
        for i, c in d.items():
            cen.setdefault(i, {})[k] = c.mean(axis=0)
    ref = {i: np.median(np.array(list(v.values())), axis=0) for i, v in cen.items()}
    per = {}
    for k, d in dets.items():
        disp = [np.linalg.norm(c.mean(axis=0) - ref[i]) for i, c in d.items()]
        per[k] = float(np.median(disp)) if disp else float('nan')
    v = np.array([x for x in per.values() if not math.isnan(x)])
    s = dict(rms_px=float(np.sqrt(np.mean(v ** 2))) if len(v) else None, max_px=float(np.max(v)) if len(v) else None)
    return per, s


def mark_table(dets, w, h, marks):
    n = len(dets)
    rows = {}
    for d in dets.values():
        for i, c in d.items():
            rows.setdefault(i, []).append(c)
    out = []
    for i in sorted(set(rows) | set(marks)):
        cs = rows.get(i, [])
        if cs:
            a = np.array(cs)
            m = a.mean(axis=0)
            edge = float(min(m[:, 0].min(), m[:, 1].min(), w - 1 - m[:, 0].max(), h - 1 - m[:, 1].max()))
            side = float(np.mean(np.linalg.norm(np.diff(np.vstack([m, m[:1]]), axis=0), axis=1)))
            out.append(dict(id=i, known=i in marks, seen_frac=len(cs) / n, u=float(m[:, 0].mean()), v=float(m[:, 1].mean()),
                            side_px=side, edge_px=edge, near_edge=edge < EDGE_WARN_PX))
        else:
            out.append(dict(id=i, known=i in marks, seen_frac=0.0, u=None, v=None, side_px=None, edge_px=None,
                            near_edge=None))
    return out


def mean_corners(dets, min_frac=0.5):
    n = len(dets)
    acc = {}
    for d in dets.values():
        for i, c in d.items():
            acc.setdefault(i, []).append(c)
    return {i: np.mean(np.array(v), axis=0) for i, v in acc.items() if len(v) >= min_frac * n}


def lens_analysis(dets, marks, w, h, zc):
    mc = mean_corners(dets)
    ids = [i for i in mc if i in marks and 1 <= i <= 24]       # fit only on the floor marks (3.3)
    out = dict(fit_ids=sorted(ids))
    if len(ids) < 6:
        out['note'] = f'only {len(ids)} floor marks seen in at least half of the frames; no fit'
        return out
    obj = np.vstack([mark_corners_floor(*marks[i]) for i in ids])
    img = np.vstack([mc[i] for i in ids])
    out['homography'] = dict(zip(('rms_px', 'max_floor_err_mm'), homography_check(obj, img)))
    fits, cv = {}, {}
    for m in MODELS:
        cam, rms = fit_camera(m, obj, img, w, h, zc)
        fits[m] = dict(rms_px=rms, f_px=cam.f, k=cam.k.tolist(), Zc_m=float(abs(cam.centre[2])),
                       centre_xy_m=cam.centre[:2].tolist(), corner_distortion=cam.corner_distortion(w, h),
                       tilt_deg=float(math.degrees(math.acos(min(1.0, abs((cam.R.T @ [0, 0, 1])[2]))))))
        cv[m] = cross_validate(m, ids, mc, marks, w, h, zc)
    out['fits'] = fits
    out['cross_validation_mm'] = cv
    ch = choose_model(cv)
    out['chosen'] = ch
    check = [i for i in mc if i in marks and 25 <= i <= 33]
    if ch and check:
        cam, _ = fit_camera(ch, obj, img, w, h, zc)
        e = {i: float(np.hypot(*(cam.pix2floor(mc[i].mean(axis=0, keepdims=True))[0] - np.array(marks[i][:2]))) * 1000)
             for i in check}
        out['check_marks_err_mm'] = e
    if ch:
        cd = fits[ch]['corner_distortion']
        if ch == 'KB2':
            out['checkerboard'] = 'can (KB2 chon theo kiem cheo; muc 3.3 tieu chi 2)'
        elif cd < CORNER_DIST_LIMIT:
            out['checkerboard'] = 'khong can (B1/B2, meo o goc khung {:.1%} < 10 %; muc 3.3 tieu chi 1)'.format(cd)
        else:
            out['checkerboard'] = 'xem lai: {} nhung meo o goc khung {:.1%} >= 10 %'.format(ch, cd)
    return out


def report_text(clip, man, info, ft, table, dr, lens):
    L = [f'Chẩn đoán clip thử C7: {os.path.basename(clip)}', '',
         f'(5) File: mở được = {info.get("opened")}; {info.get("width")}x{info.get("height")}, codec {info.get("fourcc")}, '
         f'fps danh định {info.get("fps_nominal")}, cờ xoay {info.get("orientation_meta")} (đã tắt tự xoay); '
         f'cỡ khung giải mã {info.get("decoded_shapes")}; SHA-256 {man["sha256"][:16]}…, {man["size_bytes"] / 1e6:.1f} MB']
    if ft.get('n', 0) >= 3:
        L.append(f'(4) Giờ khung: {ft["n"]} khung, {ft["duration_s"]:.2f} s; bước trung vị {ft["dt_median_ms"]:.3f} ms '
                 f'(chu kỳ {ft["period_ms"]:.3f}), lệch chuẩn {ft["dt_std_ms"]:.3f}, nhỏ nhất {ft["dt_min_ms"]:.3f}, lớn nhất '
                 f'{ft["dt_max_ms"]:.3f}; bỏ khung (bước >= 1,5 chu kỳ): {ft["n_dropped"]}; không tăng: {ft["n_nonmonotonic"]}')
    seen = [r for r in table if r['seen_frac'] > 0]
    miss = [r['id'] for r in table if r['known'] and r['seen_frac'] == 0]
    edge = [r['id'] for r in table if r.get('near_edge')]
    L.append(f'(1) Mốc: thấy {len(seen)} ID {[r["id"] for r in seen]}; mốc biết mà không thấy: {miss}; '
             f'sát mép khung (< {EDGE_WARN_PX:g} px): {edge}')
    low = [(r['id'], round(r['seen_frac'], 2)) for r in seen if r['seen_frac'] < 0.95]
    if low:
        L.append(f'    thấy dưới 95 % số khung: {low}')
    L.append(f'(3) Trôi khung khi máy treo yên: RMS {dr["rms_px"]} px, lớn nhất {dr["max_px"]} px (median dịch tâm các mốc)')
    if 'fits' in lens:
        h = lens['homography']
        L.append(f'(2) Ống kính (khớp {len(lens["fit_ids"])} mốc sàn, tâm mốc DANH ĐỊNH nên sai số mm gồm cả sai lệch dán):')
        L.append(f'    homography: chiếu lại {h["rms_px"]:.2f} px RMS, sai số sàn lớn nhất tại mốc {h["max_floor_err_mm"]:.1f} mm')
        for m in MODELS:
            fi, cv = lens['fits'][m], lens['cross_validation_mm'][m]
            cvs = f'kiểm chéo {cv["rms_mm"]:.1f} mm RMS (lớn nhất {cv["max_mm"]:.1f})' if cv else 'kiểm chéo: không đủ mốc'
            L.append(f'    {m}: chiếu lại {fi["rms_px"]:.2f} px, f {fi["f_px"]:.0f} px, k {[round(x, 4) for x in fi["k"]]}, '
                     f'Z_c {fi["Zc_m"]:.3f} m, nghiêng {fi["tilt_deg"]:.1f}°, méo ở góc khung {fi["corner_distortion"]:.1%}; {cvs}')
        L.append(f'    chọn: {lens["chosen"]}; bàn cờ: {lens.get("checkerboard")}')
        if lens.get('check_marks_err_mm'):
            L.append(f'    mã kiểm 25-33 (không vào khớp): {{{", ".join(f"{k}: {v:.1f}" for k, v in lens["check_marks_err_mm"].items())}}} mm')
    else:
        L.append(f'(2) Ống kính: {lens.get("note")}')
    L.append('(6) Thời gian phơi sáng: CHƯA ĐỌC (laptop không có ffprobe hay exiftool; thêm công cụ phải hỏi Liam)')
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--markers', nargs='+', required=True, help='print CSVs (id, x_cm, y_cm, canh_den_mm)')
    ap.add_argument('--zc', type=float, default=None, help='lens height above the floor from the tape [m]')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--max-frames', type=int, default=None)
    a = ap.parse_args(argv)
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds files: use a new folder')
    os.makedirs(a.out, exist_ok=True)
    marks = load_marks(a.markers)
    man = dict(file=os.path.basename(a.video), size_bytes=os.path.getsize(a.video), sha256=sha256(a.video))
    with open(os.path.join(a.out, 'clip_manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(man)); w.writeheader(); w.writerow(man)
    info, times, dets = read_clip(a.video, a.stride, a.max_frames)
    if not info.get('opened') or not times:
        txt = f'OpenCV KHÔNG mở được {a.video} (backend FFmpeg). Thử chép lại file gốc, hay đổi Formats sang Most Compatible.\n'
        open(os.path.join(a.out, 'report.txt'), 'w', encoding='utf-8').write(txt)
        print(txt)
        return 2
    W, H = info['width'], info['height']
    ft = frame_times(times, info['fps_nominal'])
    per, dr = drift(dets)
    table = mark_table(dets, W, H, marks)
    lens = lens_analysis(dets, marks, W, H, a.zc)
    with open(os.path.join(a.out, 'marks.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    with open(os.path.join(a.out, 'frames.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['index', 't_ms', 'n_marks', 'drift_px'])
        for k, t in enumerate(times):
            w.writerow([k, f'{t:.3f}', len(dets[k]) if k in dets else '', f'{per[k]:.4f}' if k in per else ''])
    rep = dict(clip=man, video=info, frame_times=ft, drift=dr, marks=table, lens=lens, zc_m=a.zc, stride=a.stride,
               corner_refine='CORNER_REFINE_SUBPIX (diagnosis only)', exposure='not read: no ffprobe/exiftool')
    with open(os.path.join(a.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    txt = report_text(a.video, man, info, ft, table, dr, lens)
    open(os.path.join(a.out, 'report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
