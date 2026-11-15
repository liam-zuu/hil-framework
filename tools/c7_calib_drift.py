#!/usr/bin/env python3
"""
c7_calib_drift.py: clip H at the start and at the end of every floor session, compare the two
calibrations, and when they differ score the runs twice (O1 v11.27 section 4, gate "Clip hieu chuan
(clip H) dau va cuoi moi buoi do san"; Liam 15/11 THESIS local_ec05edf3 "Cham lai hai lan
(Recommended)"). Design: CLAUDE_THESIS 03_CODE/docs/2026-11-15_Thiet_ke_so_clip_H_dau_cuoi_v1.md
(approved Liam 15/11 CODE_CAM local_3b0bfad7: Q1 "Duyet D_H, 2 mm (Recommended)", Q2 "Chua ket luan
(Recommended)", Q3 "Tinh hong he do (Recommended)", Q4 "Co, quay them (Recommended)").

Why: c7_measure.py track refits the camera POSE on every frame from the floor marks (design 3.6), so a
rigid camera move is compensated. Not compensated: a floor mark that moved (the pose fit trusts the
mark file) and a change of the intrinsics (focus, lens, heat). The end-of-session clip H catches those.

so   --s0 <calib0>/session.json [--s1 <calib1>/session.json] [--k-start X,Y,DEG] --out drift.json
     D_H [mm] = the largest floor error, at points every 5 cm along the reference lanes of the session
     (figure 8; K1, K2 with --k-start), if a frame of one clip H is scored with the calibration of the
     other clip H AFTER the pose refit that track does:
        p  = cam_b.project(g)                      g seen in the image at the time of clip H b
        a' = cam_a with its pose refitted to the mark centres of clip H b
        D  = |a'.pix2floor(p) - g|
     both directions (a, b) = (0, 1) and (1, 0); D_H = the larger maximum. vuot = D_H > T_H,
     T_H = max(--th-mm, 2 x --d-lap-mm) (2 mm [DE XUAT], locked at C8; D_lap = D_H of the two clip H
     shot back to back at the start of C8, Q4). No --s1 (clip H at the end missing): thieu_H1, vuot.
     Reported, not deciding: shift of the mark centres H1 - H0 (median, largest, per mark), RMS of the
     pose refit against the RMS of the session itself, marks whose own residual grows above
     --mark-px (2 px [DE XUAT]), focal length, distortion, camera height of the two fits.
gop  --drift drift.json --s0-score score.json [--s1-score score_h1.json] --out score_final.json
     not vuot: the H0 score unchanged. vuot: both scores of the run, the same outcome -> kept with flag
     calib_drift; different -> chua_ket_luan (calib_drift_split); hong_he_do in either -> hong_he_do
     (Q3, the run is rerun, O1 3.5). thieu_H1: chua_ket_luan with flag thieu_H1 (Q2), hong_he_do of
     H0 kept. Both sets of numbers are written; nothing is averaged.
The H1 score must come from the same run log (t0_log_us equal) and the same clock offset (give
c7_score.py --offset-s with the offset of the H0 score): checked.
Last line: #ket_qua,ma=0 done / 2 error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c7_clip_probe as cp                                               # noqa: E402
import c7_measure as cm                                                  # noqa: E402
import o1_lanes                                                          # noqa: E402
from o1_feasibility import Path8                                         # noqa: E402
from o1_tasks import FINAL_SET                                           # noqa: E402

TH_MM = 2.0            # T_H [DE XUAT], locked at C8 (design Q1)
MARK_PX = 2.0          # own residual growth of one mark that names it [DE XUAT]
STEP_M = 0.05          # spacing of the lane samples
MIN_MARKS = 6          # common floor marks needed for the pose refit (calibrate needs 6 too)
OUTCOMES = ('dat', 'khong_dat', 'chua_ket_luan', 'hong_he_do')


# ---------------------------------------------------------------- lanes
def lane_points(k_start=None, step=STEP_M):
    """(n, 2) floor points every `step` m along the figure 8 and, with k_start (x, y, heading rad), the
    K1 and K2 lanes placed on that nominal start pose (as c7_score.KTask.to_lane, inverted)."""
    p8 = Path8(n=40001)
    pts = [_resample(p8.s, p8.x, p8.y, step)]
    names = ['O1-F']
    if k_start is not None:
        x0, y0, h0 = k_start
        c, s = math.cos(h0), math.sin(h0)
        for task in FINAL_SET:
            lane = o1_lanes.Lane(o1_lanes.k_corners(task))
            q = _resample(lane.s, lane.x, lane.y, step)
            pts.append(np.column_stack([x0 + c * q[:, 0] - s * q[:, 1], y0 + s * q[:, 0] + c * q[:, 1]]))
            names.append(task)
    return np.vstack(pts), names


def _resample(s, x, y, step):
    s = np.asarray(s, float)
    n = max(2, int(math.floor(s[-1] / step)) + 1)
    ss = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(ss, s, x), np.interp(ss, s, y)])


# ---------------------------------------------------------------- one session
def load_session(path):
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    if s.get('kind') != 'c7_session':
        raise ValueError(f'{path}: not a c7_measure calib session (kind {s.get("kind")!r})')
    return s


def centre_model(cam, marks, ids):
    """(n, 2) model pixel centre of each mark = mean of its 4 projected corners, the same definition as
    the session ref_centres_px (mean of the refined corners)."""
    out = []
    for i in ids:
        c = cp.mark_corners_floor(*marks[i])
        out.append(cam.project(np.column_stack([c, np.zeros(4)])).mean(axis=0))
    return np.array(out)


def refit_pose_centres(cam, marks, ids, centres):
    """Pose of cam (intrinsics and distortion kept, as c7_measure.refit_pose) refitted to the observed
    mark centres. Returns (camera, RMS px, per-mark residual px)."""
    def make(x):
        return cp.Camera(cam.model, cam.f, cam.cx, cam.cy, cam.k, x[:3], x[3:])

    def res(x):
        return (centre_model(make(x), marks, ids) - centres).ravel()

    sol = least_squares(res, np.concatenate([cam.rvec, cam.tvec]), loss='huber', f_scale=1.0,
                        x_scale='jac', max_nfev=400)
    c = make(sol.x)
    r = np.linalg.norm(centre_model(c, marks, ids) - centres, axis=1)
    return c, float(np.sqrt(np.mean(r ** 2))), r


def one_way(sa, sb, pts):
    """Floor error [m] at pts when frames of clip H b are scored with the calibration of clip H a after
    the pose refit; plus the refit details."""
    marks = {int(i): tuple(v) for i, v in sa['marks'].items()}
    ref_b = {int(i): np.array(v, float) for i, v in sb['ref_centres_px'].items()}
    ids = [i for i in sorted(ref_b) if i in marks and i in cm.FLOOR_IDS]
    if len(ids) < MIN_MARKS:
        raise ValueError(f'only {len(ids)} common floor marks between the two clips H (need {MIN_MARKS})')
    cam_a, cam_b = cm.cam_from_dict(sa['camera']), cm.cam_from_dict(sb['camera'])
    centres_b = np.array([ref_b[i] for i in ids])
    a_ref, rms, per = refit_pose_centres(cam_a, marks, ids, centres_b)
    # the same refit of cam_b to its own clip: the residual level the comparison starts from
    _, rms_self, per_self = refit_pose_centres(cam_b, {int(i): tuple(v) for i, v in sb['marks'].items()}, ids,
                                               centres_b)
    p = cam_b.project(np.column_stack([pts, np.zeros(len(pts))]))
    g = a_ref.pix2floor(p, 0.0)
    err = np.linalg.norm(g[:, :2] - pts, axis=1)
    return err, dict(ids=ids, rms_px=rms, rms_self_px=rms_self, per_mark_px=per.tolist(),
                     per_mark_self_px=per_self.tolist())


def compare(s0, s1, k_start=None, th_mm=TH_MM, d_lap_mm=None, mark_px=MARK_PX):
    """drift dict of design section 2 (s1 None: clip H at the end missing)."""
    th = max(th_mm, 2.0 * d_lap_mm) if d_lap_mm is not None else th_mm
    base = dict(kind='c7_calib_drift', rule='O1 v11.27 4 (clip H dau va cuoi buoi); c7_calib_drift.py',
                h0_video=s0.get('video'), h0_video_sha256=s0.get('video_sha256'),
                T_H_mm=th, th_mm=th_mm, d_lap_mm=d_lap_mm, mark_px=mark_px,
                status='T_H, mark_px: DE XUAT, khoa o C8')
    if s1 is None:
        return dict(base, thieu_H1=True, vuot=True, D_H_mm=None,
                    note='thieu clip H cuoi: cac luot tu clip H dang dung toi cuoi buoi chua ket luan (Q2)')
    if s0['marks'] != s1['marks']:
        raise ValueError('the two sessions use different mark files: calibrate both clips H with the same --markers')
    pts, names = lane_points(k_start)
    e01, d01 = one_way(s0, s1, pts)
    e10, d10 = one_way(s1, s0, pts)
    k01, k10 = int(np.argmax(e01)), int(np.argmax(e10))
    d_h = 1000.0 * max(e01[k01], e10[k10])
    worst = pts[k01] if e01[k01] >= e10[k10] else pts[k10]
    ref0 = {int(i): np.array(v, float) for i, v in s0['ref_centres_px'].items()}
    ref1 = {int(i): np.array(v, float) for i, v in s1['ref_centres_px'].items()}
    common = [i for i in sorted(ref0) if i in ref1 and i in cm.FLOOR_IDS]
    shift = {i: float(np.linalg.norm(ref1[i] - ref0[i])) for i in common}
    # a mark that moved on the floor: its own residual, camera of H0 refitted to H1, against the same
    # camera refitted to H0 itself (both calibrations trust the same mark file, so D_H alone may miss it)
    marks0 = {int(i): tuple(v) for i, v in s0['marks'].items()}
    cam0 = cm.cam_from_dict(s0['camera'])
    _, rms00, r00 = refit_pose_centres(cam0, marks0, common, np.array([ref0[i] for i in common]))
    _, rms01, r01 = refit_pose_centres(cam0, marks0, common, np.array([ref1[i] for i in common]))
    grow = [dict(id=i, px=round(float(b - a), 3)) for i, a, b in zip(common, r00, r01) if b - a > mark_px]
    c0, c1 = s0['camera'], s1['camera']
    return dict(base, thieu_H1=False, h1_video=s1.get('video'), h1_video_sha256=s1.get('video_sha256'),
                D_H_mm=d_h, vuot=bool(d_h > th),
                D_mm={'h0_to_h1': dict(max=1000 * float(e01[k01]), median=1000 * float(np.median(e01))),
                      'h1_to_h0': dict(max=1000 * float(e10[k10]), median=1000 * float(np.median(e10)))},
                worst_xy_m=[float(worst[0]), float(worst[1])], lanes=names, n_points=int(len(pts)),
                step_m=STEP_M, k_start=None if k_start is None else list(k_start),
                shift_px=dict(median=float(np.median(list(shift.values()))) if shift else None,
                              max=max(shift.values()) if shift else None,
                              max_id=max(shift, key=shift.get) if shift else None,
                              per_mark={str(i): round(v, 3) for i, v in shift.items()}),
                missing_in_h1=[i for i in sorted(ref0) if i in cm.FLOOR_IDS and i not in ref1],
                missing_in_h0=[i for i in sorted(ref1) if i in cm.FLOOR_IDS and i not in ref0],
                refit_rms_px=dict(h0_cam_on_h0=rms00, h0_cam_on_h1=rms01, h1_cam_on_h1=d01['rms_self_px'],
                                  h1_cam_on_h0=d10['rms_px']),
                marks_grown=grow,
                focal_px=[c0['f'], c1['f']], focal_rel=c1['f'] / c0['f'] - 1.0, k=[c0['k'], c1['k']],
                model=[c0['model'], c1['model']],
                zc_fit_m=[s0.get('zc_fit_m'), s1.get('zc_fit_m')])


# ---------------------------------------------------------------- combine the two scores of a run
def combine(drift, sc0, sc1=None):
    """score_final dict of design section 3."""
    o0 = sc0.get('outcome')
    if o0 not in OUTCOMES:
        raise ValueError(f'H0 score: outcome {o0!r}')
    out = dict(sc0)
    info = dict(T_H_mm=drift.get('T_H_mm'), D_H_mm=drift.get('D_H_mm'), vuot=drift.get('vuot'),
                thieu_H1=drift.get('thieu_H1'), outcome_h0=o0, outcome_h1=None, flags=[])
    if not drift.get('vuot'):
        if sc1 is not None:
            raise ValueError('drift below T_H: no second score is used (give only --s0-score)')
        out['calib_drift'] = info
        return out
    if drift.get('thieu_H1'):
        if sc1 is not None:
            raise ValueError('drift.json says clip H at the end is missing, but a second score was given')
        final = 'hong_he_do' if o0 == 'hong_he_do' else 'chua_ket_luan'
        info['flags'] = ['thieu_H1']
    else:
        if sc1 is None:
            raise ValueError('drift above T_H: give --s1-score (the run scored with the clip H at the end)')
        o1 = sc1.get('outcome')
        if o1 not in OUTCOMES:
            raise ValueError(f'H1 score: outcome {o1!r}')
        if sc0.get('t0_log_us') != sc1.get('t0_log_us'):
            raise ValueError('the two scores are not of the same run log (t0_log_us differ)')
        if sc0.get('offset_s') is None or sc1.get('offset_s') is None or \
                abs(float(sc0['offset_s']) - float(sc1['offset_s'])) > 1e-9:
            raise ValueError('the two scores use different clock offsets: score H1 with --offset-s of the H0 score')
        info['outcome_h1'] = o1
        info['flags'] = ['calib_drift']
        if 'hong_he_do' in (o0, o1):
            final = 'hong_he_do'                                      # Q3
        elif o0 == o1:
            final = o0
        else:
            final = 'chua_ket_luan'
            info['flags'].append('calib_drift_split')
        info['h1'] = {k: sc1.get(k) for k in ('outcome', 'outcome_reasons', 'conditions', 'e_perp_max', 'completed',
                                               'frames_valid', 'gap_max', 'shift_max_px')}
    info['h0'] = {k: sc0.get(k) for k in ('outcome', 'outcome_reasons', 'conditions', 'e_perp_max', 'completed',
                                          'frames_valid', 'gap_max', 'shift_max_px')}
    out['outcome'] = final
    out['calib_drift'] = info
    return out


# ---------------------------------------------------------------- CLI
def _read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write(path, obj):
    if os.path.exists(path):
        raise ValueError(f'{path} exists: not overwritten')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('so', help='compare the two clips H of a session')
    a1.add_argument('--s0', required=True, help='session.json of clip H in use (start, or the last re-shoot)')
    a1.add_argument('--s1', default=None, help='session.json of clip H at the end (absent: thieu_H1)')
    a1.add_argument('--k-start', default=None, help='X,Y,HEAD_DEG nominal start pose of K1/K2 (C8, final test)')
    a1.add_argument('--th-mm', type=float, default=TH_MM)
    a1.add_argument('--d-lap-mm', type=float, default=None, help='D_H of the two back-to-back clips H of C8 (Q4)')
    a1.add_argument('--mark-px', type=float, default=MARK_PX)
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('gop', help='combine the H0 and H1 scores of one run')
    a2.add_argument('--drift', required=True)
    a2.add_argument('--s0-score', required=True)
    a2.add_argument('--s1-score', default=None)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'so':
            ks = None
            if a.k_start:
                x0, y0, h = cm._pair(a.k_start, 3, '--k-start')
                ks = (x0, y0, math.radians(h))
            s0 = load_session(a.s0)
            s1 = load_session(a.s1) if a.s1 else None
            d = compare(s0, s1, ks, a.th_mm, a.d_lap_mm, a.mark_px)
            _write(a.out, d)
            if d['thieu_H1']:
                print('THIEU clip H cuoi: cac luot chua ket luan (Q2)')
            else:
                print(f"D_H {d['D_H_mm']:.2f} mm (T_H {d['T_H_mm']:.2f}): {'VUOT, cham lai hai lan' if d['vuot'] else 'trong nguong'}; "
                      f"dich tam moc trung vi {d['shift_px']['median']:.2f} px, lon nhat {d['shift_px']['max']:.2f} px "
                      f"(ID {d['shift_px']['max_id']}); moc du tang: {[m['id'] for m in d['marks_grown']] or 'khong'}")
            print(f"#ket_qua,ma=0,vuot={int(d['vuot'])}")
        else:
            d = _read(a.drift)
            if d.get('kind') != 'c7_calib_drift':
                raise ValueError(f'{a.drift}: not a c7_calib_drift file')
            r = combine(d, _read(a.s0_score), _read(a.s1_score) if a.s1_score else None)
            _write(a.out, r)
            print(f"outcome {r['outcome']} (H0 {r['calib_drift']['outcome_h0']}, H1 {r['calib_drift']['outcome_h1']}), "
                  f"co {r['calib_drift']['flags']}")
            print(f"#ket_qua,ma=0,outcome={r['outcome']}")
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f'LOI: {e}')
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    sys.exit(main())
