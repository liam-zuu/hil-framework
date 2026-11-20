#!/usr/bin/env python3
"""
c8_hc_video.py: fast sparse reading of a C8 12b/E clip for the floor decision of tools/c8_hc_a.py kiem_video
(CODE_HIEUCHUAN, 19/11/2026; design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_CODE_HIEUCHUAN_v4.md 2.7).

Every frame is grabbed, one in `stride` is decoded; the roof tag (c7_measure.CAR_ID) is found in a square around its
last position (full frame when lost), its corners refined and turned into the tag centre and the tag heading by
c7_measure.car_pose (c7_measure is imported, not changed). tag_offset 0, so psi is the tag heading. Output: the
frames dict read by c8_hc_a.analyse (chuoi 'chan_doan': a quick check, not a measurement).

HCB4 (CODE_HIEUCHUAN p5, 20/11/2026): the camera POSE is refitted on the floor marks (c7_measure.refit_pose, the
whole frame searched) every `refit_every` decoded frames and kept in between, so the tag pose is referenced to the
floor ('neo_san'): a camera that turns during the clip cannot pass as a heading change of the car (session 2: the
iPhone was re-mounted between the calibration clip and the HC clips; inside the clips the pose moved <= 0.08 deg).
Frames before the first successful refit are not used.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sparse_frames(clip, session_path, stride=3, roi=360, max_frames=None, progress=None, refit_every=30):
    import cv2
    import c7_measure as cm
    S = json.load(open(session_path, encoding='utf-8'))
    cam = cm.cam_from_dict(S['camera'])
    marks = {int(i): tuple(v) for i, v in S['marks'].items()}
    side, kappa, plumb = S['car_side_m'], S['kappa'], S['plumb_xy_m']
    det = cm.make_detector()
    cap = cv2.VideoCapture(clip, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise OSError(f'OpenCV cannot open {clip}')
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    idx, ts, xs, ys, ps, oks = [], [], [], [], [], []
    k, prev, n_dec, n_refit, rms_max = 0, None, 0, 0, 0.0
    try:
        while max_frames is None or k < max_frames:
            if not cap.grab():
                break
            if k % stride:
                k += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            corners, full = None, None
            if n_dec % refit_every == 0:                       # HCB4: camera pose on the floor marks of this frame
                full = cm.detect_refined(det, g)
                obj, img, used = cm.floor_obs(full, marks)
                if len(used) >= 4:
                    cam, rms = cm.refit_pose(cam, obj, img)
                    n_refit, rms_max = n_refit + 1, max(rms_max, rms)
                corners = full.get(cm.CAR_ID)
            n_dec += 1
            if corners is None and prev is not None:
                h, w = g.shape
                x0, y0 = max(0, int(prev[0]) - roi), max(0, int(prev[1]) - roi)
                x1, y1 = min(w, int(prev[0]) + roi), min(h, int(prev[1]) + roi)
                d = cm.detect_refined(det, np.ascontiguousarray(g[y0:y1, x0:x1]))
                if cm.CAR_ID in d:
                    corners = d[cm.CAR_ID] + np.array([x0, y0], float)
            if corners is None and full is None:
                d = cm.detect_refined(det, g)
                corners = d.get(cm.CAR_ID)
            if corners is not None and n_refit:
                r = cm.car_pose(cam, corners, side, kappa, plumb, (0.0, 0.0, 0.0))
                prev = corners.mean(axis=0)
                idx.append(k); ts.append(t); xs.append(r['tag_x']); ys.append(r['tag_y']); ps.append(r['tag_psi'])
                oks.append(True)
            else:
                prev = None
                idx.append(k); ts.append(t); xs.append(np.nan); ys.append(np.nan); ps.append(np.nan); oks.append(False)
            if progress and len(idx) % 200 == 0:
                progress(k)
            k += 1
    finally:
        cap.release()
    return dict(idx=np.array(idx), t=np.array(ts), x=np.array(xs), y=np.array(ys), psi=np.array(ps),
                ok=np.array(oks, bool), chuoi='chan_doan', fps=60.0 / stride, neo_san=n_refit > 0,
                refit=dict(n=n_refit, rms_px_max=rms_max, moi_khung=refit_every))
