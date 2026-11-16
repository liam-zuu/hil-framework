#!/usr/bin/env python3
"""
c7_calib_drift.py: clip H at the start and at the end of every floor session, compare the two
calibrations, and when they differ score the runs twice (O1 v11.28 section 4, gate "Clip hieu chuan
(clip H) dau va cuoi moi buoi do san"; Liam 15/11 THESIS local_ec05edf3 "Cham lai hai lan
(Recommended)"). Design: CLAUDE_THESIS 03_CODE/docs/2026-11-15_Thiet_ke_so_clip_H_dau_cuoi_v1.md
(approved Liam 15/11 CODE_CAM local_3b0bfad7: Q1 "Duyet D_H, 2 mm (Recommended)", Q2 "Chua ket luan
(Recommended)", Q3 "Tinh hong he do (Recommended)", Q4 "Co, quay them (Recommended)"). The spec names
the threshold eps_H (O1 v11.28; T_H is the labour time of S05). Codex Sol round 1 (CD1-2..11, Liam
16/11 CODE_CAM "Sua 10 muc + vong 2 (Recommended)").

Why: c7_measure.py track refits the camera POSE on every frame from the 4 corners of the floor marks
(design 3.6, Huber loss), so a rigid camera move is compensated in the floor mapping; the parallax step
(plumb point, kappa = h / Z_c) stays at the values of the session (CD1-2). Not compensated: a change of
the intrinsics (focus, lens, heat). The end-of-session clip H catches those. Out of scope (CD1-1): an
error common to all mark coordinates; that is the absolute accuracy, U_p and the check tags of C8.

so   --s0 <calib0>/session.json [--s1 <calib1>/session.json] --tasks O1-F02,O1-F05[,O1-K1 --k-start X,Y,DEG]
     --out drift.json
     D_H [mm] = the largest error of the car position, at points every 5 cm along the lanes of the tasks
     run in the session, if a frame of one clip H is scored with the calibration of the other clip H,
     the way track does it (pose refit on the 4 corners of every mark, then the parallax step of the
     scoring session):
        p   = cam_b.project(g)                     g seen in the image at the time of clip H b
        a'  = cam_a, pose refitted to the corners of clip H b (c7_measure.refit_pose)
        g0  = a'.pix2floor(p)                      floor mapping error  g0 - g
        car = (1 - kappa_a)(g0 - g) + kappa_a (C_a - g) - kappa_b (C_b - g)
              C = plumb point; C_b = C_a + camera shift between the two fits, kappa_b = h / (Z_c + dZ)
     both directions (a, b) = (0, 1) and (1, 0); D_H = the larger maximum. vuot = D_H > eps_H,
     eps_H = max(--eps-mm, 2 x --d-lap-mm) (2 mm [DE XUAT], locked at C8; D_lap = D_H of the two clips H
     shot back to back at the start of C8, Q4). No --s1 (clip H at the end missing): thieu_H1, vuot.
     The two sessions must share the mark file (sha256), zc_tape, plumb, tag height, car tag side, image
     size (CD1-5). Reported, not deciding: shift of the mark centres H1 - H0 (median, largest, per mark),
     RMS of the pose refits, marks whose own corner residual grows above --mark-px (2 px [DE XUAT]),
     focal length, distortion, camera height and centre of the two fits.
gop  --drift drift.json --s0-score score.json --s0-track <dir0>/track.json
     [--s1-score score_h1.json --s1-track <dir1>/track.json] --out score_final.json
     Provenance (CD1-4): each track.json names the session it used (sha256, = the one in drift.json), both
     tracks are of the same video, each score's frames_sha256 = frames.csv next to its track.json, both
     scores have the same esp32 log, U_m, D, k_start, thresholds and clock offset (CD1-6).
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
SCORE_SAME = ('U_m', 'D', 'k_start', 'thresholds', 'esp32_sha256', 't0_log_us')         # CD1-4, CD1-6


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


def lane_points(tasks, k_start=None, step=STEP_M):
    """(n, 2) floor points every `step` m of arc along the lanes of the tasks run in the session: the
    figure 8 when any O1-Fxx is in tasks; K1, K2 placed on the nominal start pose k_start (x, y, heading
    rad; the inverse of c7_score.KTask.to_lane), which is then required (CD1-8)."""
    pts, names = [], []
    if any(t in FIG8 for t in tasks):
        p8 = Path8(n=40001)
        pts.append(_resample(p8.s, p8.x, p8.y, step))
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
            pts.append(np.column_stack([x0 + c * q[:, 0] - s * q[:, 1], y0 + s * q[:, 0] + c * q[:, 1]]))
            names.append(task)
    return np.vstack(pts), names


def _resample(s, x, y, step):
    """Points at arc length 0, step, 2 step, ... and the end point (CD1-9)."""
    s = np.asarray(s, float)
    ss = np.arange(0.0, s[-1], step)
    if s[-1] - ss[-1] > 1e-9:
        ss = np.append(ss, s[-1])
    return np.column_stack([np.interp(ss, s, x), np.interp(ss, s, y)])


# ---------------------------------------------------------------- one session
def load_session(path):
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    if s.get('kind') != 'c7_session':
        raise ValueError(f'{path}: not a c7_measure calib session (kind {s.get("kind")!r})')
    for k in ('camera', 'marks', 'ref_corners_px', 'ref_centres_px', 'kappa', 'tag_h_m') + SAME_KEYS:
        if k not in s:
            raise ValueError(f'{path}: no {k!r} (calibrate again with c7_measure.py at this commit)')
    s['_sha256'] = cm.sha256(path)
    return s


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


def one_way(sa, sb, pts):
    """Car position error [m] (vectors, (n, 2)) at pts when frames of clip H b are scored with the
    calibration of clip H a, as track and car_pose do; plus the refit details."""
    marks = {int(i): tuple(v) for i, v in sa['marks'].items()}
    ids = [int(i) for i in sorted(sb['ref_corners_px'], key=int) if int(i) in marks and int(i) in cm.FLOOR_IDS]
    if len(ids) < MIN_MARKS:
        raise ValueError(f'only {len(ids)} common floor marks between the two clips H (need {MIN_MARKS})')
    cam_a, cam_b = cm.cam_from_dict(sa['camera']), cm.cam_from_dict(sb['camera'])
    img_b = _corners(sb, ids)
    a_ref, rms = cm.refit_pose(cam_a, _obj(marks, ids), img_b)      # what track does (CD1-3)
    p = cam_b.project(np.column_stack([pts, np.zeros(len(pts))]))
    e_floor = a_ref.pix2floor(p, 0.0) - pts
    # parallax step (CD1-2): car_pose uses the plumb point and kappa of the scoring session a
    ka, h = float(sa['kappa']), float(sa['tag_h_m'])
    ca = np.asarray(sa['plumb_xy_m'], float)
    dC = cam_b.centre - cam_a.centre                                 # camera shift between the two fits
    zc_a = float(sa['zc_tape_m'])
    kb = h / (zc_a + dC[2])
    cb = ca + dC[:2]
    e = (1.0 - ka) * e_floor + ka * (ca - pts) - kb * (cb - pts)
    return e, dict(ids=ids, rms_px=rms, camera_shift_m=dC.tolist(), kappa=[ka, kb])


def compare(s0, s1, tasks, k_start=None, eps_mm=EPS_MM, d_lap_mm=None, mark_px=MARK_PX):
    """drift dict of design section 2 (s1 None: clip H at the end missing)."""
    eps_mm = _finite(eps_mm, '--eps-mm', 0.0, strict=True)
    mark_px = _finite(mark_px, '--mark-px', 0.0, strict=True)
    if d_lap_mm is not None:
        d_lap_mm = _finite(d_lap_mm, '--d-lap-mm', 0.0)
    eps = max(eps_mm, 2.0 * d_lap_mm) if d_lap_mm is not None else eps_mm
    pts, names = lane_points(tasks, k_start)                         # also checks --k-start for K1/K2
    base = dict(kind='c7_calib_drift', rule='O1 v11.28 4 (clip H dau va cuoi buoi); c7_calib_drift.py',
                h0_video=s0.get('video'), h0_video_sha256=s0.get('video_sha256'), s0_session_sha256=s0['_sha256'],
                tasks=list(tasks), lanes=names, n_points=int(len(pts)), step_m=STEP_M,
                k_start=None if k_start is None else list(k_start),
                eps_H_mm=eps, eps_mm=eps_mm, d_lap_mm=d_lap_mm, mark_px=mark_px,
                status='eps_H, mark_px: DE XUAT, khoa o C8')
    if s1 is None:
        return dict(base, thieu_H1=True, vuot=True, D_H_mm=None, s1_session_sha256=None,
                    note='thieu clip H cuoi: cac luot tu clip H dang dung toi cuoi buoi chua ket luan (Q2)')
    check_same(s0, s1)
    e01, d01 = one_way(s0, s1, pts)
    e10, d10 = one_way(s1, s0, pts)
    n01, n10 = np.linalg.norm(e01, axis=1), np.linalg.norm(e10, axis=1)
    k01, k10 = int(np.argmax(n01)), int(np.argmax(n10))
    d_h = 1000.0 * max(n01[k01], n10[k10])
    worst = pts[k01] if n01[k01] >= n10[k10] else pts[k10]
    ref0 = {int(i): np.array(v, float) for i, v in s0['ref_centres_px'].items()}
    ref1 = {int(i): np.array(v, float) for i, v in s1['ref_centres_px'].items()}
    common = [i for i in sorted(ref0) if i in ref1 and i in cm.FLOOR_IDS and str(i) in s0['ref_corners_px']
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
                D_mm={'h0_to_h1': dict(max=1000 * float(n01[k01]), median=1000 * float(np.median(n01))),
                      'h1_to_h0': dict(max=1000 * float(n10[k10]), median=1000 * float(np.median(n10)))},
                worst_xy_m=[float(worst[0]), float(worst[1])],
                camera_shift_m=d10['camera_shift_m'], kappa=d01['kappa'],
                shift_px=dict(median=float(np.median(list(shift.values()))) if shift else None,
                              max=max(shift.values()) if shift else None,
                              max_id=max(shift, key=shift.get) if shift else None,
                              per_mark={str(i): round(v, 3) for i, v in shift.items()}),
                missing_in_h1=[i for i in sorted(ref0) if i in cm.FLOOR_IDS and i not in ref1],
                missing_in_h0=[i for i in sorted(ref1) if i in cm.FLOOR_IDS and i not in ref0],
                refit_rms_px=dict(h0_cam_on_h0=rms00, h0_cam_on_h1=rms01, h0_cam_on_h1_lanes=d01['rms_px'],
                                  h1_cam_on_h0_lanes=d10['rms_px']),
                marks_grown=grow,
                focal_px=[c0['f'], c1['f']], focal_rel=c1['f'] / c0['f'] - 1.0, k=[c0['k'], c1['k']],
                model=[c0['model'], c1['model']], zc_fit_m=[s0.get('zc_fit_m'), s1.get('zc_fit_m')])


# ---------------------------------------------------------------- combine the two scores of a run
def check_drift(d):
    """Schema and consistency of a drift.json (CD1-7)."""
    if d.get('kind') != 'c7_calib_drift':
        raise ValueError('not a c7_calib_drift file')
    for k in ('vuot', 'thieu_H1'):
        if not isinstance(d.get(k), bool):
            raise ValueError(f'drift.json: {k!r} missing or not true/false')
    eps = _finite(d.get('eps_H_mm'), 'drift.json eps_H_mm', 0.0, strict=True)
    if not d.get('s0_session_sha256'):
        raise ValueError('drift.json: no s0_session_sha256')
    if d['thieu_H1']:
        if not d['vuot'] or d.get('D_H_mm') is not None or d.get('s1_session_sha256') is not None:
            raise ValueError('drift.json: thieu_H1 must come with vuot true, no D_H_mm, no s1 session')
    else:
        dh = _finite(d.get('D_H_mm'), 'drift.json D_H_mm', 0.0)
        if not d.get('s1_session_sha256'):
            raise ValueError('drift.json: no s1_session_sha256')
        if d['vuot'] != (dh > eps):
            raise ValueError(f'drift.json: vuot {d["vuot"]} does not match D_H {dh} / eps_H {eps}')


def check_score(sc, tr, frames_sha, session_sha, name):
    """One score with its track.json (CD1-4)."""
    if sc.get('outcome') not in OUTCOMES:
        raise ValueError(f'{name} score: outcome {sc.get("outcome")!r}')
    _finite(sc.get('offset_s'), f'{name} score offset_s')
    _finite(sc.get('U_m'), f'{name} score U_m', 0.0)
    if not sc.get('esp32_sha256') or not sc.get('frames_sha256'):
        raise ValueError(f'{name} score: no esp32_sha256 / frames_sha256 (score with c7_score.py at this commit)')
    if tr.get('session_sha256') != session_sha:
        raise ValueError(f'{name} track.json used session {tr.get("session_sha256")!r}, drift.json has {session_sha!r}')
    if sc['frames_sha256'] != frames_sha:
        raise ValueError(f'{name} score is not of the frames.csv next to its track.json')


def combine(drift, sc0, tr0, fr0_sha, sc1=None, tr1=None, fr1_sha=None):
    """score_final dict of design section 3."""
    check_drift(drift)
    check_score(sc0, tr0, fr0_sha, drift['s0_session_sha256'], 'H0')
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
    check_score(sc1, tr1, fr1_sha, drift['s1_session_sha256'], 'H1')
    if not tr0.get('video_sha256') or tr0.get('video_sha256') != tr1.get('video_sha256'):
        raise ValueError('the two track.json are not of the same run video')
    for k in SCORE_SAME:
        if sc0.get(k) != sc1.get(k):
            raise ValueError(f'the two scores differ in {k}: {sc0.get(k)!r} / {sc1.get(k)!r}')
    if abs(float(sc0['offset_s']) - float(sc1['offset_s'])) > 1e-9:
        raise ValueError('the two scores use different clock offsets: score H1 with --offset-s of the H0 score')
    o1 = sc1['outcome']
    info['outcome_h1'] = o1
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
        return json.load(f, parse_constant=_no_nan)


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
            d = compare(s0, s1, tasks, ks, a.eps_mm, a.d_lap_mm, a.mark_px)
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
