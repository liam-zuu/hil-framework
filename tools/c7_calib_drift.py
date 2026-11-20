#!/usr/bin/env python3
"""
c7_calib_drift.py: clip H at the start and at the end of every floor session, compare the two
calibrations, and when they differ score the runs twice (O1 v11.28 section 4, gate "Clip hieu chuan
(clip H) dau va cuoi moi buoi do san"; Liam 15/11 THESIS local_ec05edf3 "Cham lai hai lan
(Recommended)"). Design: CLAUDE_THESIS 03_CODE/docs/2026-11-15_Thiet_ke_so_clip_H_dau_cuoi_v1.md
(approved Liam 15/11 CODE_CAM local_3b0bfad7: Q1 "Duyet D_H, 2 mm (Recommended)", Q2 "Chua ket luan
(Recommended)", Q3 "Tinh hong he do (Recommended)", Q4 "Co, quay them (Recommended)"). The spec names
the threshold eps_H (O1 v11.28; T_H is the labour time of S05). Codex Sol round 1 (CD1-2..11, Liam
16/11 CODE_CAM "Sua 10 muc + vong 2 (Recommended)"); round 2 (CD2-1..7, Liam 16/11 CODE_CAM "Sua 7 muc +
vong 3 (Recommended)"); round 3 (CD3-2..6, Liam 16/11 CODE_CAM "Sua 5 muc + Codex vong 4"); round 4
(CD4-1, CD4-2, Liam 16/11 CODE_CAM "Sua 2 muc + Codex vong 5").

Why: c7_measure.py track refits the camera POSE on every frame from the 4 corners of the floor marks
(design 3.6, Huber loss), so a rigid camera move is compensated in the floor mapping; the parallax step
(plumb point, kappa = h / Z_c) stays at the values of the session (CD1-2). Not compensated: a change of
the intrinsics (focus, lens, heat), and a height change that a plane of marks reads as a focal change
(CD2-1). The end-of-session clip H catches those. Out of scope (CD1-1): an error common to all mark
coordinates; that is the absolute accuracy, U_p and the check tags of C8.
ASSUMPTION (CD3-1): focus and exposure locked for the whole session (C8 trial on paper, step 0d), so the
physical focal length does not change. A plane of marks cannot tell (Z, f) from (Z (1 + e), f (1 + e)):
a rise of the camera together with the same relative focal increase leaves every mark pixel unchanged,
both hypotheses below give D_H = 0, and the parallax of the car tag is wrong (2.3 mm at 2.7 m from the
plumb point for 6 cm). Only an independent observation (a target at a known height in clip H, the tape
height measured again at the end) would remove this; none is in the workflow.

so   --s0 <calib0>/session.json [--s1 <calib1>/session.json] --tasks O1-F02,O1-F05[,O1-K1 --k-start X,Y,DEG]
     --out drift.json
     D_H [mm] = the largest error of the car (wheel centre) position, at poses every 5 cm along the lanes
     of the tasks run in the session (body heading: tangent on the figure 8, the k_start heading on K1, K2,
     CD3-3), with the tag offset the runs are tracked with (--tag-offset, default the tag_offset of --s0,
     CD3-2), if a frame of one clip H is scored with
     the calibration of the other clip H, through the measuring chain itself (CD2-2, one_way):
        lens C at the time of b: plumb_a + camera shift of the two fits, height Z_c + dZ ("fit"), or the
              focal change of the fits read as a height change, Z = (Z_c + dZ) f_a / f_b
              ("focal_as_height", CD2-1);
        the 4 corners of the car tag at height h (wheel pose + tag offset) -> their rays from C meet the
              floor at P' -> pixels =
              cam_b.project(P') (cam_b maps the floor as the camera of that time);
        a' = cam_a, pose refitted to the mark corners of clip H b (c7_measure.refit_pose);
        c7_measure.car_pose(a', pixels, side, kappa_a, plumb_a, tag offset) - true wheel pose.
     both directions (a, b) = (0, 1) and (1, 0), both lens hypotheses; D_H = the largest. vuot = D_H > eps_H,
     eps_H = max(--eps-mm, 2 x --d-lap-mm) (2 mm [DE XUAT], locked at C8; D_lap = D_H of the two clips H
     shot back to back at the start of C8, Q4). No --s1 (clip H at the end missing): thieu_H1, vuot.
     The two sessions must share the mark file (sha256), zc_tape, plumb, tag height, car tag side, image
     size (CD1-5). Reported, not deciding: shift of the mark centres H1 - H0 (median, largest, per mark),
     RMS of the pose refits, marks whose own corner residual grows above --mark-px (2 px [DE XUAT]),
     focal length, distortion, camera height and centre of the two fits.
gop  --drift drift.json --s0-score score.json --s0-track <dir0>/track.json
     [--s1-score score_h1.json --s1-track <dir1>/track.json] --out score_final.json
     Provenance (CD1-4, CD2-3): each track.json names the session it used (sha256, = the one in
     drift.json) and the frames.csv it wrote (frames_sha256 = the file next to it = the frames_sha256 of
     the score). Same run (CD1-6, CD2-4): both tracks have the same video, tag offset, frame count and
     decoded size, and that tag offset is the one of drift.json (CD3-2); both scores of c7_score_v2, the
     same task, esp32 log, t = 0 (t0_log_us), k_start, thresholds, constants file (hang_so: file, sha256), sender
     log and events file (sha256, CC1-3); both tracks the same frame gates (CC1-4). A score lost to the measurement
     system early (c7_score lost_output) carries the same fields, offset and t = 0 null when not known (CC1-1).
     Clock (Liam 17/11 CODE_CAM local_4a972073, box, verbatim: "Moi lan tu ghep gio (Recommended)"): each
     score matches the clocks on its own frames (C7b, or --offset-s as for any run), so the two scorings
     differ by the calibration and what follows from it, U_t included; the two offsets are kept in the
     result (calib_drift.offset_s_h0, offset_s_h1), not required equal (the old rule of CD1-6).
     Clip H shot again inside a session (O1 v11.31 4, B8S-9): the new clip closes the part before it:
     so --s0 <clip H in use> --s1 <new clip H> --tasks <tasks run in that part>, then gop for every run of
     that part; the runs after it are scored with the new clip H (it is the s0 of the next part).
     Scope (CD2-5): the task of the score is in drift.tasks and, for K1/K2, its k_start is that of
     drift.json, below eps_H too. All these fields must be present and valid: frame count and decoded
     sizes positive integers, thresholds numbers >= 0 (text only in 'status'), no null (CD2-6, CD3-5, CD4-1).
     not vuot: the H0 score unchanged. vuot: same outcome -> kept, flag calib_drift; different ->
     chua_ket_luan (calib_drift_split); hong_he_do in either -> hong_he_do (Q3, rerun, O1 3.5).
     thieu_H1: chua_ket_luan with flag thieu_H1 (Q2); hong_he_do of H0 stays hong_he_do. The whole H1
     score is kept in the result; nothing is averaged.
Last line: #ket_qua,ma=0 done / 2 error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c7_clip_probe as cp                                               # noqa: E402
import c7_measure as cm                                                  # noqa: E402
import o1_lanes                                                          # noqa: E402
from o1_feasibility import Path8                                         # noqa: E402
from o1_tasks import FIG8, FINAL_SET                                     # noqa: E402

EPS_MM = 2.0           # eps_H [DE XUAT], locked at C8 (design Q1; O1 v11.28 4)
MARK_PX = 2.0          # own corner residual growth of one mark that names it [DE XUAT]
STEP_M = 0.05          # spacing of the lane samples (arc length)
MIN_MARKS = 6          # common floor marks needed for the pose refit (calibrate needs 6 too)
OUTCOMES = ('dat', 'khong_dat', 'chua_ket_luan', 'hong_he_do')
SAME_KEYS = ('zc_tape_m', 'plumb_xy_m', 'tag_h_m', 'car_side_m', 'width', 'height')     # CD1-5
SCORE_SAME = ('version', 'task', 'hang_so', 'k_start', 'thresholds', 'esp32_sha256', 't0_log_us',  # CD1-4, CD2-4
              'sender_sha256', 'events_sha256')        # Codex CC1-3 (17/11): the same command-path inputs
SCORE_VERSION = 'c7_score_v2'      # the scorer whose output this combines (U_m, D of the version before are gone)
TRACK_SAME = ('video_sha256', 'tag_offset', 'frames', 'decoded_shapes')       # CD2-4: track options (gates: CC1-4)
THRESH_TEXT = ('status',)          # the only text field of c7_score thresholds (metadata); the rest are numbers >= 0


def _finite(x, name, lo=None, strict=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f'{name}: not a number ({x!r})') from None
    if not math.isfinite(v):
        raise ValueError(f'{name}: not finite ({x!r})')
    if lo is not None and (v <= lo if strict else v < lo):
        raise ValueError(f'{name}: {v} must be {">" if strict else ">="} {lo}')
    return v


# ---------------------------------------------------------------- lanes
def parse_tasks(s):
    tasks = [t.strip() for t in s.split(',') if t.strip()]
    if not tasks:
        raise ValueError('--tasks: no task given')
    bad = [t for t in tasks if t not in FIG8 and t not in FINAL_SET]
    if bad:
        raise ValueError(f'--tasks: unknown {bad} (one of {tuple(FIG8) + FINAL_SET})')
    return sorted(set(tasks), key=tasks.index)


def lane_poses(tasks, k_start=None, step=STEP_M):
    """(n, 3) WHEEL centre poses (x, y, body heading rad) every `step` m of arc along the lanes of the
    tasks run in the session: the figure 8 when any O1-Fxx is in tasks, body heading on the tangent
    (o1_tasks: "heading on the tangent", Path8.head); K1, K2 placed on the nominal start pose k_start
    (x, y, heading rad; the inverse of c7_score.KTask.to_lane), which is then required (CD1-8), body
    heading held at k_start's heading (o1_tasks: K1 strafes, K2 "no body rotation"; CD3-3)."""
    pts, names = [], []
    if any(t in FIG8 for t in tasks):
        p8 = Path8(n=40001)
        q = _resample(p8.s, p8.x, p8.y, step)
        ss = _arc_samples(p8.s, step)
        pts.append(np.column_stack([q, np.interp(ss, p8.s, np.unwrap(p8.head))]))
        names.append('O1-F')
    ks = [t for t in tasks if t in FINAL_SET]
    if ks:
        if k_start is None:
            raise ValueError(f'{ks} in --tasks: give --k-start X,Y,HEAD_DEG (nominal start pose)')
        x0, y0, h0 = k_start
        c, s = math.cos(h0), math.sin(h0)
        for task in ks:
            lane = o1_lanes.Lane(o1_lanes.k_corners(task))
            q = _resample(lane.s, lane.x, lane.y, step)
            pts.append(np.column_stack([x0 + c * q[:, 0] - s * q[:, 1], y0 + s * q[:, 0] + c * q[:, 1],
                                        np.full(len(q), float(h0))]))
            names.append(task)
    return np.vstack(pts), names


def lane_points(tasks, k_start=None, step=STEP_M):
    """(n, 2) floor points of lane_poses, and the lane names."""
    p, names = lane_poses(tasks, k_start, step)
    return p[:, :2], names


def _arc_samples(s, step):
    """Arc lengths 0, step, 2 step, ... and the end (CD1-9)."""
    s = np.asarray(s, float)
    ss = np.arange(0.0, s[-1], step)
    if s[-1] - ss[-1] > 1e-9:
        ss = np.append(ss, s[-1])
    return ss


def _resample(s, x, y, step):
    """Points at arc length 0, step, 2 step, ... and the end point (CD1-9)."""
    ss = _arc_samples(s, step)
    return np.column_stack([np.interp(ss, s, x), np.interp(ss, s, y)])


# ---------------------------------------------------------------- one session
def load_session(path):
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    if s.get('kind') != 'c7_session':
        raise ValueError(f'{path}: not a c7_measure calib session (kind {s.get("kind")!r})')
    for k in ('camera', 'marks', 'ref_corners_px', 'ref_centres_px', 'kappa', 'tag_h_m', 'tag_offset') + SAME_KEYS:
        if k not in s:
            raise ValueError(f'{path}: no {k!r} (calibrate again with c7_measure.py at this commit)')
    check_camera(s, path)
    s['_sha256'] = cm.sha256(path)
    return s


def check_camera(s, name):
    """Focal length finite and positive, numbers of the set-up finite (CD3-4: f = 0 divided by zero)."""
    c = s['camera']
    if not isinstance(c, dict):
        raise ValueError(f'{name}: camera is not an object')
    _finite(c.get('f'), f'{name} camera f', 0.0, strict=True)
    for k in ('cx', 'cy'):
        _finite(c.get(k), f'{name} camera {k}')
    for k in ('k', 'rvec', 'tvec'):
        v = c.get(k)
        if not isinstance(v, list) or not v:
            raise ValueError(f'{name}: camera {k} {v!r}')
        for x in v:
            _finite(x, f'{name} camera {k}')
    _finite(s['zc_tape_m'], f'{name} zc_tape_m', 0.0, strict=True)
    _finite(s['tag_h_m'], f'{name} tag_h_m', 0.0)
    _finite(s['car_side_m'], f'{name} car_side_m', 0.0, strict=True)
    _finite(s['kappa'], f'{name} kappa', 0.0)


def check_same(s0, s1):
    """The two calibrations must be of the same set-up (CD1-5)."""
    if s0['marks'] != s1['marks']:
        raise ValueError('the two sessions use different mark coordinates: calibrate both clips H with the same --markers')
    h0 = sorted((m['file'], m['sha256']) for m in s0.get('markers_csv') or [])
    h1 = sorted((m['file'], m['sha256']) for m in s1.get('markers_csv') or [])
    if not h0 or h0 != h1:
        raise ValueError(f'mark files differ or are not recorded (markers_csv {h0} / {h1})')
    for k in SAME_KEYS:
        if s0.get(k) != s1.get(k):
            raise ValueError(f'the two sessions differ in {k}: {s0.get(k)!r} / {s1.get(k)!r}')
    if s0['zc_tape_m'] is None:
        raise ValueError('zc_tape_m missing: calibrate with --zc (kappa from the tape, design 3.4)')


def _corners(s, ids):
    return np.vstack([np.asarray(s['ref_corners_px'][str(i)], float) for i in ids])


def _obj(marks, ids):
    return np.vstack([cp.mark_corners_floor(*marks[i]) for i in ids])


def _per_mark(cam, marks, ids, img):
    """Corner RMS [px] of each mark under cam."""
    r = cam.project(np.column_stack([_obj(marks, ids), np.zeros(4 * len(ids))])) - img
    return np.sqrt(np.mean(np.sum(r.reshape(len(ids), 4, 2) ** 2, axis=2), axis=1))


def tag_corners(x, y, psi, side, tag_offset=(0.0, 0.0, 0.0)):
    """(4, 2) floor xy of the car tag corners, ArUco order (c7_measure.car_corners_body), for the wheel
    centre pose (x, y, psi): tag centre = wheel centre + R(psi) (dx, dy), tag heading = psi + dpsi, the
    inverse of what c7_measure.car_pose does with the tag offset (CD3-2)."""
    dx, dy, dpsi = tag_offset
    c, s = math.cos(psi), math.sin(psi)
    tx, ty = x + c * dx - s * dy, y + s * dx + c * dy
    a = psi + dpsi
    c, s = math.cos(a), math.sin(a)
    b = cm.car_corners_body(side)
    return np.column_stack([tx + c * b[:, 0] - s * b[:, 1], ty + s * b[:, 0] + c * b[:, 1]])


def physical_centres(sa, cam_a, cam_b):
    """Lens positions assumed at the time of clip H b, against the scoring session a (whose plumb
    point and Z_c from the tape are taken as right at the time of a):
      fit              plumb_a + the camera shift of the two fits, Z_c + their height change (CD1-2);
      focal_as_height  the same plumb, but the focal change of the two fits read as a height change: a
                       plane of marks cannot tell one from the other and both fits hold the height to
                       the tape (CD2-1): Z = (Z_c + dZ) f_a / f_b.
    D_H takes the worse of the two."""
    dC = cam_b.centre - cam_a.centre
    cxy = np.asarray(sa['plumb_xy_m'], float) + dC[:2]
    z_fit = float(sa['zc_tape_m']) + float(dC[2])
    return dict(fit=np.array([cxy[0], cxy[1], z_fit]),
                focal_as_height=np.array([cxy[0], cxy[1], z_fit * cam_a.f / cam_b.f])), dC


def one_way(sa, sb, poses, tag_offset=(0.0, 0.0, 0.0)):
    """Car position errors [m] (dict hypothesis -> (n, 2) vectors) at the poses when frames of clip H b
    are scored with the calibration of clip H a, the way track and car_pose do it (CD2-2):
      1. the 4 corners of the car tag at its height h, along the lane, seen from the lens C of the time
         of clip H b (physical_centres): their rays meet the floor at P' = C + (corner - C) Z / (Z - h);
         cam_b maps the floor as the camera of that time does, so the corner pixels = cam_b.project(P');
      2. a' = cam_a, pose refitted to the corners of the floor marks of clip H b (c7_measure.refit_pose);
      3. c7_measure.car_pose(a', pixels, side, kappa_a, plumb_a, tag_offset): the wheel centre track
         would report (kappa and plumb point of the scoring session a, the tag offset track uses);
      4. error = that - the true pose.
    Same session (a = b): 0 by construction (the parallax step is exact for a pinhole over a plane)."""
    marks = {int(i): tuple(v) for i, v in sa['marks'].items()}
    fl = set(cm.session_floor_ids(sa)) & set(cm.session_floor_ids(sb))
    ids = [int(i) for i in sorted(sb['ref_corners_px'], key=int) if int(i) in marks and int(i) in fl]
    if len(ids) < MIN_MARKS:
        raise ValueError(f'only {len(ids)} common floor marks between the two clips H (need {MIN_MARKS})')
    cam_a, cam_b = cm.cam_from_dict(sa['camera']), cm.cam_from_dict(sb['camera'])
    a_ref, rms = cm.refit_pose(cam_a, _obj(marks, ids), _corners(sb, ids))      # what track does (CD1-3)
    ka, h, side = float(sa['kappa']), float(sa['tag_h_m']), float(sa['car_side_m'])
    plumb = [float(v) for v in sa['plumb_xy_m']]
    cents, dC = physical_centres(sa, cam_a, cam_b)
    errs = {}
    for name, C in cents.items():
        if not C[2] > h:
            raise ValueError(f'lens height {C[2]:.3f} m ({name}) not above the tag height {h} m')
        e = np.zeros((len(poses), 2))
        for k, (x, y, psi) in enumerate(poses):
            q = tag_corners(x, y, psi, side, tag_offset)
            floor = C[:2] + (q - C[:2]) * (C[2] / (C[2] - h))
            px = cam_b.project(np.column_stack([floor, np.zeros(4)]))
            r = cm.car_pose(a_ref, px, side, ka, plumb, tag_offset)
            e[k] = (r['x'] - x, r['y'] - y)
        errs[name] = e
    return errs, dict(ids=ids, rms_px=rms, camera_shift_m=dC.tolist(),
                      lens_z_m={k: float(v[2]) for k, v in cents.items()})


def compare(s0, s1, tasks, k_start=None, eps_mm=EPS_MM, d_lap_mm=None, mark_px=MARK_PX, tag_offset=None):
    """drift dict of design section 2 (s1 None: clip H at the end missing). tag_offset (dx, dy, dpsi
    rad): the one the runs are tracked with; None: the tag_offset of session H0 (CD3-2)."""
    eps_mm = _finite(eps_mm, '--eps-mm', 0.0, strict=True)
    mark_px = _finite(mark_px, '--mark-px', 0.0, strict=True)
    if d_lap_mm is not None:
        d_lap_mm = _finite(d_lap_mm, '--d-lap-mm', 0.0)
    eps = max(eps_mm, 2.0 * d_lap_mm) if d_lap_mm is not None else eps_mm
    poses, names = lane_poses(tasks, k_start)                        # also checks --k-start for K1/K2
    off = _triple(s0.get('tag_offset') if tag_offset is None else list(tag_offset), 'tag offset')
    pts = poses[:, :2]
    base = dict(kind='c7_calib_drift', rule='O1 v11.28 4 (clip H dau va cuoi buoi); c7_calib_drift.py',
                h0_video=s0.get('video'), h0_video_sha256=s0.get('video_sha256'), s0_session_sha256=s0['_sha256'],
                tasks=list(tasks), lanes=names, n_points=int(len(pts)), step_m=STEP_M,
                k_start=None if k_start is None else list(k_start), tag_offset=off,
                eps_H_mm=eps, eps_mm=eps_mm, d_lap_mm=d_lap_mm, mark_px=mark_px,
                status='eps_H, mark_px: DE XUAT, khoa o C8')
    if s1 is None:
        return dict(base, thieu_H1=True, vuot=True, D_H_mm=None, s1_session_sha256=None,
                    note='thieu clip H cuoi: cac luot tu clip H dang dung toi cuoi buoi chua ket luan (Q2)')
    check_same(s0, s1)
    check_camera(s0, 'session H0')
    check_camera(s1, 'session H1')
    e01, d01 = one_way(s0, s1, poses, off)
    e10, d10 = one_way(s1, s0, poses, off)
    by = {}
    for way, ee in (('h0_to_h1', e01), ('h1_to_h0', e10)):
        for hyp, e in ee.items():
            n = 1000.0 * np.linalg.norm(e, axis=1)
            k = int(np.argmax(n))
            by[(way, hyp)] = (float(n[k]), float(np.median(n)), k)
    (w_way, w_hyp), (d_h, _, w_k) = max(by.items(), key=lambda kv: kv[1][0])
    worst = pts[w_k]
    ref0 = {int(i): np.array(v, float) for i, v in s0['ref_centres_px'].items()}
    ref1 = {int(i): np.array(v, float) for i, v in s1['ref_centres_px'].items()}
    fl = set(cm.session_floor_ids(s0)) & set(cm.session_floor_ids(s1))
    common = [i for i in sorted(ref0) if i in ref1 and i in fl and str(i) in s0['ref_corners_px']
              and str(i) in s1['ref_corners_px']]
    shift = {i: float(np.linalg.norm(ref1[i] - ref0[i])) for i in common}
    # a mark that moved on the floor: its own corner residual, camera of H0 refitted to H1, against the
    # same camera refitted to H0 itself (both calibrations trust the same mark file)
    marks0 = {int(i): tuple(v) for i, v in s0['marks'].items()}
    cam0 = cm.cam_from_dict(s0['camera'])
    c00, rms00 = cm.refit_pose(cam0, _obj(marks0, common), _corners(s0, common))
    c01, rms01 = cm.refit_pose(cam0, _obj(marks0, common), _corners(s1, common))
    r00, r01 = _per_mark(c00, marks0, common, _corners(s0, common)), _per_mark(c01, marks0, common, _corners(s1, common))
    grow = [dict(id=i, px=round(float(b - a), 3)) for i, a, b in zip(common, r00, r01) if b - a > mark_px]
    c0, c1 = s0['camera'], s1['camera']
    return dict(base, thieu_H1=False, s1_session_sha256=s1['_sha256'],
                h1_video=s1.get('video'), h1_video_sha256=s1.get('video_sha256'),
                D_H_mm=d_h, vuot=bool(d_h > eps),
                D_mm={way: {hyp: dict(max=by[(way, hyp)][0], median=by[(way, hyp)][1]) for hyp in e01}
                      for way in ('h0_to_h1', 'h1_to_h0')},
                worst=dict(direction=w_way, hypothesis=w_hyp),
                worst_xy_m=[float(worst[0]), float(worst[1])],
                camera_shift_m=d10['camera_shift_m'], lens_z_m=dict(h0_to_h1=d01['lens_z_m'], h1_to_h0=d10['lens_z_m']),
                shift_px=dict(median=float(np.median(list(shift.values()))) if shift else None,
                              max=max(shift.values()) if shift else None,
                              max_id=max(shift, key=shift.get) if shift else None,
                              per_mark={str(i): round(v, 3) for i, v in shift.items()}),
                missing_in_h1=[i for i in sorted(ref0) if i in fl and i not in ref1],
                missing_in_h0=[i for i in sorted(ref1) if i in fl and i not in ref0],
                refit_rms_px=dict(h0_cam_on_h0=rms00, h0_cam_on_h1=rms01, h0_cam_on_h1_lanes=d01['rms_px'],
                                  h1_cam_on_h0_lanes=d10['rms_px']),
                marks_grown=grow,
                focal_px=[c0['f'], c1['f']], focal_rel=c1['f'] / c0['f'] - 1.0, k=[c0['k'], c1['k']],
                model=[c0['model'], c1['model']], zc_fit_m=[s0.get('zc_fit_m'), s1.get('zc_fit_m')])


# ---------------------------------------------------------------- combine the two scores of a run
def _dict(x, name):
    if not isinstance(x, dict):
        raise ValueError(f'{name}: not a JSON object')
    return x


def _need(d, keys, name):
    miss = [k for k in keys if k not in d]
    if miss:
        raise ValueError(f'{name}: no {miss}')


def _sha(x, name):
    if not isinstance(x, str) or not x:
        raise ValueError(f'{name}: missing or empty')
    return x


def _triple(x, name):
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise ValueError(f'{name}: 3 numbers expected ({x!r})')
    return [_finite(v, name) for v in x]


def check_drift(d):
    """Schema and consistency of a drift.json (CD1-7, CD2-6)."""
    _dict(d, 'drift.json')
    if d.get('kind') != 'c7_calib_drift':
        raise ValueError('not a c7_calib_drift file')
    _need(d, ('vuot', 'thieu_H1', 'eps_H_mm', 's0_session_sha256', 's1_session_sha256', 'D_H_mm', 'tasks', 'k_start',
              'tag_offset'), 'drift.json')
    for k in ('vuot', 'thieu_H1'):
        if not isinstance(d[k], bool):
            raise ValueError(f'drift.json: {k!r} not true/false')
    eps = _finite(d['eps_H_mm'], 'drift.json eps_H_mm', 0.0, strict=True)
    _sha(d['s0_session_sha256'], 'drift.json s0_session_sha256')
    tasks = d['tasks']
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f'drift.json: tasks {tasks!r}')
    parse_tasks(','.join(tasks))
    if any(t in FINAL_SET for t in tasks) or d['k_start'] is not None:
        _triple(d['k_start'], 'drift.json k_start')
    _triple(d['tag_offset'], 'drift.json tag_offset')
    if d['thieu_H1']:
        if not d['vuot'] or d['D_H_mm'] is not None or d['s1_session_sha256'] is not None:
            raise ValueError('drift.json: thieu_H1 must come with vuot true, no D_H_mm, no s1 session')
    else:
        dh = _finite(d['D_H_mm'], 'drift.json D_H_mm', 0.0)
        _sha(d['s1_session_sha256'], 'drift.json s1_session_sha256')
        if d['vuot'] != (dh > eps):
            raise ValueError(f'drift.json: vuot {d["vuot"]} does not match D_H {dh} / eps_H {eps}')


def check_score(sc, tr, frames_sha, session_sha, drift, name):
    """One score with its track.json (CD1-4, CD2-3, CD2-5, CD2-6)."""
    _dict(sc, f'{name} score')
    _dict(tr, f'{name} track.json')
    _need(sc, ('outcome', 'offset_s', 'frames_sha256') + SCORE_SAME, f'{name} score')
    _need(tr, ('session_sha256', 'frames_sha256', 'gates') + TRACK_SAME, f'{name} track.json')
    # Codex CC1-4, CC2-2: the frame gates the track applied, checked (null, empty or a bad threshold refused); the two
    # tracks are compared on these values in combine
    cm.check_gates(tr['gates'], f'{name} track.json gates')
    if sc['outcome'] not in OUTCOMES:
        raise ValueError(f'{name} score: outcome {sc["outcome"]!r}')
    if sc['version'] != SCORE_VERSION:
        raise ValueError(f'{name} score: version {sc["version"]!r}, expected {SCORE_VERSION} (score again with '
                         'c7_score.py at this commit)')
    # Codex CC1-1: a run lost to the measurement system early has no clock offset (C7b failed) or no t = 0 (edge
    # lost): null there, never a number made up; any judged run has both
    lost = sc['outcome'] == 'hong_he_do'
    for k in ('offset_s', 't0_log_us'):
        if not (lost and sc[k] is None):
            _finite(sc[k], f'{name} score {k}')
    hs = _dict(sc['hang_so'], f'{name} score hang_so')
    _need(hs, ('file', 'sha256'), f'{name} score hang_so')
    _sha(hs['sha256'], f'{name} score hang_so.sha256')
    th = sc['thresholds']
    if not isinstance(th, dict) or not [k for k in th if k not in THRESH_TEXT]:
        raise ValueError(f'{name} score: thresholds {th!r}')
    for k, v in th.items():                                    # CD3-5, CD4-1: checked key by key, no null
        if k in THRESH_TEXT:
            if not isinstance(v, str):
                raise ValueError(f'{name} score thresholds.{k}: text expected ({v!r})')
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f'{name} score thresholds.{k}: number expected ({v!r})')
        else:
            _finite(v, f'{name} score thresholds.{k}', 0.0)
    _sha(sc['esp32_sha256'], f'{name} score esp32_sha256')
    _sha(sc['frames_sha256'], f'{name} score frames_sha256')
    ks = _triple(sc['k_start'], f'{name} score k_start')
    # CD2-5: the run is one the drift was computed for, on the same nominal start pose
    if sc['task'] not in drift['tasks']:
        raise ValueError(f'{name} score is task {sc["task"]!r}, drift.json covers {drift["tasks"]}')
    if sc['task'] in FINAL_SET:
        dk = _triple(drift['k_start'], 'drift.json k_start')
        if max(abs(a - b) for a, b in zip(ks, dk)) > 1e-9:
            raise ValueError(f'{name} score k_start {ks} is not the k_start of drift.json {dk}')
    # CD1-4, CD2-3: score -> frames.csv -> track.json -> session
    if tr['session_sha256'] != session_sha:
        raise ValueError(f'{name} track.json used session {tr["session_sha256"]!r}, drift.json has {session_sha!r}')
    _sha(tr['video_sha256'], f'{name} track.json video_sha256')
    _sha(tr['frames_sha256'], f'{name} track.json frames_sha256 (track with c7_measure.py at this commit)')
    toff = _triple(tr['tag_offset'], f'{name} track.json tag_offset')
    doff = _triple(drift['tag_offset'], 'drift.json tag_offset')
    if max(abs(a - b) for a, b in zip(toff, doff)) > 1e-9:                     # CD3-2
        raise ValueError(f'{name} track.json tag_offset {toff} is not the one drift.json was computed with {doff}')
    # CD3-5: values of the compared fields, not only their presence
    if isinstance(tr['frames'], bool) or not isinstance(tr['frames'], int) or tr['frames'] < 1:
        raise ValueError(f'{name} track.json frames {tr["frames"]!r}')
    shp = tr['decoded_shapes']
    if (not isinstance(shp, list) or not shp or
            not all(isinstance(x, list) and len(x) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                                                               for v in x) for x in shp)):
        raise ValueError(f'{name} track.json decoded_shapes {shp!r}')
    if tr['frames_sha256'] != frames_sha:
        raise ValueError(f'{name}: the frames.csv next to track.json is not the one that track wrote')
    if sc['frames_sha256'] != tr['frames_sha256']:
        raise ValueError(f'{name} score is not of the frames.csv of its track.json')


def combine(drift, sc0, tr0, fr0_sha, sc1=None, tr1=None, fr1_sha=None):
    """score_final dict of design section 3."""
    check_drift(drift)
    check_score(sc0, tr0, fr0_sha, drift['s0_session_sha256'], drift, 'H0')
    o0 = sc0['outcome']
    out = dict(sc0)
    info = dict(eps_H_mm=drift['eps_H_mm'], D_H_mm=drift.get('D_H_mm'), vuot=drift['vuot'],
                thieu_H1=drift['thieu_H1'], outcome_h0=o0, outcome_h1=None, flags=[],
                s0_session_sha256=drift['s0_session_sha256'], s1_session_sha256=drift.get('s1_session_sha256'))
    if not drift['vuot'] or drift['thieu_H1']:
        if sc1 is not None or tr1 is not None:
            raise ValueError('drift.json: below eps_H or clip H at the end missing, but a second score was given')
        if drift['thieu_H1']:
            info['flags'] = ['thieu_H1']
            out['outcome'] = 'hong_he_do' if o0 == 'hong_he_do' else 'chua_ket_luan'
        out['calib_drift'] = info
        return out
    if sc1 is None or tr1 is None or fr1_sha is None:
        raise ValueError('drift above eps_H: give --s1-score and --s1-track (the run scored with the clip H at the end)')
    check_score(sc1, tr1, fr1_sha, drift['s1_session_sha256'], drift, 'H1')
    for k in TRACK_SAME:
        if tr0[k] != tr1[k]:
            raise ValueError(f'the two track.json differ in {k}: {tr0[k]!r} / {tr1[k]!r} (same video, same options)')
    g0, g1 = cm.check_gates(tr0['gates']), cm.check_gates(tr1['gates'])
    if g0 != g1:
        raise ValueError(f'the two track.json applied other frame gates: {g0!r} / {g1!r} (CC1-4)')
    for k in SCORE_SAME:
        if sc0[k] != sc1[k]:
            raise ValueError(f'the two scores differ in {k}: {sc0[k]!r} / {sc1[k]!r}')
    o1 = sc1['outcome']
    info['outcome_h1'] = o1
    # Liam 17/11 "Moi lan tu ghep gio": each score matched the clocks on its own frames; both kept, not averaged
    info.update(offset_s_h0=sc0['offset_s'], offset_s_h1=sc1['offset_s'],          # null: that scoring lost it
                offset_src_h0=sc0.get('offset_src'), offset_src_h1=sc1.get('offset_src'))
    info['flags'] = ['calib_drift']
    if 'hong_he_do' in (o0, o1):
        final = 'hong_he_do'                                          # Q3
    elif o0 == o1:
        final = o0
    else:
        final = 'chua_ket_luan'
        info['flags'].append('calib_drift_split')
    info['h1_score'] = sc1                                            # CD1-6: kept whole
    out['outcome'] = final
    out['calib_drift'] = info
    return out


# ---------------------------------------------------------------- CLI
def _no_nan(c):
    raise ValueError(f'{c} in JSON: not accepted (CD1-7)')


def _read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f, parse_constant=_no_nan)                        # object: check_drift, check_score


def _write(path, obj):
    if os.path.exists(path):
        raise ValueError(f'{path} exists: not overwritten')
    text = json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False)      # CD1-7
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def _frames_sha(track_path):
    p = os.path.join(os.path.dirname(os.path.abspath(track_path)), 'frames.csv')
    if not os.path.exists(p):
        raise ValueError(f'no frames.csv next to {track_path}')
    return cm.sha256(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('so', help='compare the two clips H of a session')
    a1.add_argument('--s0', required=True, help='session.json of clip H in use (start, or the last re-shoot)')
    a1.add_argument('--s1', default=None, help='session.json of clip H at the end (absent: thieu_H1)')
    a1.add_argument('--tasks', required=True, help='tasks run in the session, e.g. O1-F02,O1-F05,O1-F08')
    a1.add_argument('--k-start', default=None, help='X,Y,HEAD_DEG nominal start pose of K1/K2 (C8, final test)')
    a1.add_argument('--eps-mm', type=float, default=EPS_MM)
    a1.add_argument('--d-lap-mm', type=float, default=None, help='D_H of the two back-to-back clips H of C8 (Q4)')
    a1.add_argument('--mark-px', type=float, default=MARK_PX)
    a1.add_argument('--tag-offset', default=None,
                    help='DX,DY,DPSI_DEG the runs are tracked with (track --tag-offset); default: tag_offset of --s0')
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('gop', help='combine the H0 and H1 scores of one run')
    a2.add_argument('--drift', required=True)
    a2.add_argument('--s0-score', required=True)
    a2.add_argument('--s0-track', required=True, help='track.json of the H0 frames (session sha256, video)')
    a2.add_argument('--s1-score', default=None)
    a2.add_argument('--s1-track', default=None)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'so':
            ks = None
            if a.k_start:
                x0, y0, h = cm._pair(a.k_start, 3, '--k-start')
                ks = (_finite(x0, '--k-start x'), _finite(y0, '--k-start y'), math.radians(_finite(h, '--k-start deg')))
            tasks = parse_tasks(a.tasks)
            s0 = load_session(a.s0)
            s1 = load_session(a.s1) if a.s1 else None
            off = None
            if a.tag_offset:
                dx, dy, dp = cm._pair(a.tag_offset, 3, '--tag-offset')
                off = (_finite(dx, '--tag-offset dx'), _finite(dy, '--tag-offset dy'),
                       math.radians(_finite(dp, '--tag-offset deg')))
            d = compare(s0, s1, tasks, ks, a.eps_mm, a.d_lap_mm, a.mark_px, off)
            _write(a.out, d)
            if d['thieu_H1']:
                print('THIEU clip H cuoi: cac luot chua ket luan (Q2)')
            else:
                print(f"D_H {d['D_H_mm']:.2f} mm (eps_H {d['eps_H_mm']:.2f}): {'VUOT, cham lai hai lan' if d['vuot'] else 'trong nguong'}; "
                      f"dich tam moc trung vi {d['shift_px']['median']:.2f} px, lon nhat {d['shift_px']['max']:.2f} px "
                      f"(ID {d['shift_px']['max_id']}); moc du tang: {[m['id'] for m in d['marks_grown']] or 'khong'}")
            print(f"#ket_qua,ma=0,vuot={int(d['vuot'])}")
        else:
            two = (a.s1_score is not None, a.s1_track is not None)
            if two[0] != two[1]:
                raise ValueError('give both --s1-score and --s1-track, or neither')
            r = combine(_read(a.drift), _read(a.s0_score), _read(a.s0_track), _frames_sha(a.s0_track),
                        _read(a.s1_score) if two[0] else None, _read(a.s1_track) if two[0] else None,
                        _frames_sha(a.s1_track) if two[0] else None)
            _write(a.out, r)
            print(f"outcome {r['outcome']} (H0 {r['calib_drift']['outcome_h0']}, H1 {r['calib_drift']['outcome_h1']}), "
                  f"co {r['calib_drift']['flags']}")
            print(f"#ket_qua,ma=0,outcome={r['outcome']}")
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError, argparse.ArgumentTypeError) as e:
        print(f'LOI: {e}')
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    sys.exit(main())
