#!/usr/bin/env python3
"""
c7_nhac_tam.py: plate lift test of C8 (trang C8b v10 DA DUYET, P3 / A2, section 3 'Nhac tam') -> repeatability of the
car tag position and heading RELATIVE TO THE CAR BODY over the placements of the tag plate; the body is watched through
two textured tape patches stuck on the body.

Design: CLAUDE_THESIS 03_CODE/docs/2026-11-18_Thiet_ke_c7_nhac_tam_v2.md (CODE_NHACTAM local_f8bebd94, 18/11/2026;
v2 after Codex Sol round 1, findings NT-1..NT-14). Measurement (design 0, 5): car still at O, control power off;
N0 = plate fitted on its pins, N1.. = lifted and put back, the last one after the calliper heights (B1); one clip of
10 s per placement, nobody in the frame; clip R = plate lying on the floor >= 0.5 m from the car (5 s).

Per frame (2.1): marks of the session's own fit (session['fit_ids']) -> camera pose refitted (c7_measure.refit_pose);
the 4 corners of the car tag (ID 0) AND the body boxes are taken to their planes with the SAME frame camera
(pix2floor, z = tag_h for the tag, z = h_than for the boxes; NT-6); tag = rigid fit to the square of car_side_m.
Box: template of a gated frame of N0 nearest its middle, matched (TM_CCOEFF_NORMED, parabolic peak, then ECC
translation). Frame gates (NT-9): finite, >= 8 marks, mark rms <= rms of clip H + 0.5 px, tag fit residual with
scale <= 1.5 mm, every box correlation >= 0.9.
Per clip (2.2): 0.5 s dropped at both ends, every stride-th frame; >= ceil(300 / stride) gated frames and >= 60 % of
the frames considered; medians; still if the 95th percentile of the deviations is within the thresholds.
Body (2.3): rigid 2D fit of the box points N0 -> Nj; clip R proves the boxes belong to the body (boxes found where they
were in N0 while the tag is away or gone); kiem_than = R passed, >= 2 boxes, fit residual within the threshold.
Repeatability (2.4): tag mapped back to the N0 body frame p' = R^T (p - t), psi' = psi - angle(R); largest over pairs
of |p'_j - p'_k| + eps_j + eps_k (eps_j = (|t_j| + |dC_j|) dh / (H - h), dh = height bound of the boxes) and of
|wrap(psi'_j - psi'_k)|. Value for U: body removed when kiem_than, else raw (limitation of v10 section 3).

  python tools/c7_nhac_tam.py khung --clip N0.MOV --session S/session.json --out D
        -> D/khung_N0.png (crop around the car, pixel grid, proposed boxes), D/khung_N0.json
  python tools/c7_nhac_tam.py do --clip N0.MOV N1.MOV N2.MOV N3.MOV N4.MOV --rong R.MOV --session S/session.json \
        --vung-than "u,v,w,h;u,v,w,h" [--h-than M] [--stride K] --out D
        -> D/nhac_tam.json, D/frames_nhac_tam.csv, D/bao_cao.txt and, without a blocking error,
           D/c8_phan_nhac_tam.json (block for c7_kiem_cheo xuat --phan nhac_tam=...; trang_thai dat / dat_gioi_han)
Boxes are pixel boxes (left, top, width, height) in the frames of N0, each inside one tape patch. Thresholds [DE XUAT].
Last line: #ket_qua,ma=0 dat / 1 dat_gioi_han (no body check: raw value with its limitation) / 2 blocking error
(no block written).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7_measure as cm                                                  # noqa: E402

VERSION = 'c7_nhac_tam_v2'
MIN_PLACEMENTS = 4
DEFAULTS = dict(
    trim_s=0.5,            # 2.2: dropped at both ends of a clip
    min_marks=8,           # 2.1: floor marks for the frame camera
    rms_extra_px=0.5,      # 2.1: mark rms <= rms of clip H + this
    tag_res_sim_m=0.0015,  # 2.1: residual of the tag fit WITH scale (gate (2a) of the scorer)
    corr_min=0.9,          # 2.1: every body box must correlate at least this
    min_frames_s1=300,     # 2.2: gated frames at stride 1 (ceil(300 / stride) at stride k)
    min_frac=0.6,          #      and share of the frames considered
    still_tag_m=0.0005,    # 2.2: 95th percentile of the deviation from the clip median, tag centre
    still_tag_deg=0.3,     #      tag heading
    still_box_m=0.0005,    #      each body box
    rigid_res_m=0.0003,    # 2.3: rms of the rigid fit of the box points N0 -> Nj
    r_tag_away_m=0.2,      # 2.3: clip R, tag at least this far from its N0 place (or not seen)
    r_min_frames_s1=100,   #      gated R frames at stride 1
    dh_m=0.03,             # 2.4: height bound of the boxes [uoc luong]
    search_px=25,          # 2.1: template search half-window
)
THRESH_STATUS = 'DE XUAT (thiet ke c7_nhac_tam v2 muc 2)'
NAN = float('nan')


# ---------------------------------------------------------------- small helpers
def parse_boxes(s):
    """'u,v,w,h;u,v,w,h' -> [(u, v, w, h)] ints, w, h >= 8."""
    out = []
    for part in [p for p in s.replace(' ', '').split(';') if p]:
        v = [int(round(float(x))) for x in part.split(',')]
        if len(v) != 4 or v[2] < 8 or v[3] < 8 or v[0] < 0 or v[1] < 0:
            raise ValueError(f'box {part!r}: need u,v,w,h with w, h >= 8')
        out.append(tuple(v))
    if not out:
        raise ValueError('no body box')
    return out


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def circ_median(angles):
    """Median of angles (rad) unwrapped around their circular mean."""
    a = np.asarray(angles, float)
    m = math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
    return wrap(m + float(np.median([wrap(x - m) for x in a])))


def subpix(r, x, y):
    """Parabolic sub-pixel peak of a correlation map around the integer peak (x, y)."""
    def p(a, b, c):
        den = a - 2.0 * b + c
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    sx = p(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0.0
    sy = p(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0.0
    return x + sx, y + sy


class Box:
    """One body box: template from the reference frame, matched in later frames (design 2.1)."""

    def __init__(self, ref_gray, box, search):
        u, v, w, h = box
        H, W = ref_gray.shape
        if u - search < 0 or v - search < 0 or u + w + search > W or v + h + search > H:
            raise ValueError(f'box {box}: too close to the frame edge for a search of {search} px')
        self.box, self.search = box, search
        self.tpl = ref_gray[v:v + h, u:u + w].astype(np.float32).copy()
        if float(self.tpl.std()) < 2.0:
            raise ValueError(f'box {box}: no texture (grey std {self.tpl.std():.2f})')
        self.u0, self.v0 = u + (w - 1) / 2.0, v + (h - 1) / 2.0      # centre of the box in pixel coordinates

    def match(self, gray):
        """(u, v) of the box centre in this frame, correlation at the peak."""
        u, v, w, h = self.box
        s = self.search
        win = gray[v - s:v + h + s, u - s:u + w + s].astype(np.float32)
        r = cv2.matchTemplate(win, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        x, y = subpix(r, *loc)
        x, y = self.ecc(win, x, y)
        return self.u0 + x - s, self.v0 + y - s, float(mx)

    def ecc(self, win, x, y):
        """Sub-pixel translation by ECC (cv2.findTransformECC) started at the correlation peak; the parabolic peak is
        kept when ECC fails or moves more than 1 px from it."""
        warp = np.array([[1.0, 0.0, x], [0.0, 1.0, y]], np.float32)
        try:
            _, warp = cv2.findTransformECC(self.tpl, win, warp, cv2.MOTION_TRANSLATION,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 1)
        except cv2.error:
            return x, y
        ex, ey = float(warp[0, 2]), float(warp[1, 2])
        if not (math.isfinite(ex) and math.isfinite(ey)) or math.hypot(ex - x, ey - y) > 1.0:
            return x, y
        return ex, ey


def frames_of(video, wanted):
    """{index: grey frame} for the wanted indices, one decoding pass."""
    out, wanted = {}, set(wanted)
    for k, t, g in cm.iter_frames(video):
        if k in wanted:
            out[k] = g
        if len(out) == len(wanted):
            break
    return out


def frame_count(video):
    n = 0
    for _ in cm.iter_frames(video):
        n += 1
    return n


def git_head():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        d = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', os.path.basename(__file__)],
                           capture_output=True, text=True, timeout=10)
        return dict(commit=c.stdout.strip() or None, file_dirty=bool(d.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, file_dirty=None)


# ---------------------------------------------------------------- per frame
class Ctx:
    """Session values used per frame."""

    def __init__(self, session, h_than, cfg):
        self.cam0 = cm.cam_from_dict(session['camera'])
        self.marks = {int(i): tuple(v) for i, v in session['marks'].items()}
        self.fit_ids = [int(i) for i in session['fit_ids']]
        self.side = float(session['car_side_m'])
        self.tag_h = float(session['tag_h_m'])
        self.rms_h = float(session['rms_px_H'])
        self.h_than = float(h_than)
        self.cfg = cfg
        self.det = cm.make_detector()


def tag_pose(cam, corners_px, side, tag_h):
    """Tag centre, heading, fit residual with scale, from the 4 corners taken to z = tag_h with the frame camera
    (NT-6: the same projection as the boxes)."""
    P = cam.pix2floor(np.asarray(corners_px, float), tag_h)
    body = cm.car_corners_body(side)
    ang, t, _, _ = cm.rigid_fit(body, P)
    _, _, _, res_sim = cm.rigid_fit(body, P, with_scale=True)
    return float(t[0]), float(t[1]), float(ang), float(res_sim)


def measure_frame(ctx, cam_prev, k, t, g, boxes):
    """One row (2.1) and the camera to start the next refit from."""
    d = cm.detect_refined(ctx.det, g)
    obj, img, used = cm.floor_obs(d, ctx.marks, ids=ctx.fit_ids)
    r = dict(idx=k, t_s=t, n_marks=len(used), rms_px=NAN, cam_x=NAN, cam_y=NAN, cam_z=NAN, tag_seen=False,
             tag_x=NAN, tag_y=NAN, tag_psi=NAN, tag_res_sim_m=NAN, box=[], gate_marks=False)
    cam = cam_prev
    if len(used) >= 4:
        cam, r['rms_px'] = cm.refit_pose(cam_prev, obj, img)
        r['cam_x'], r['cam_y'], r['cam_z'] = (float(v) for v in cam.centre)
        r['gate_marks'] = bool(len(used) >= ctx.cfg['min_marks'] and math.isfinite(r['rms_px'])
                               and r['rms_px'] <= ctx.rms_h + ctx.cfg['rms_extra_px'])
        if cm.CAR_ID in d:
            x, y, a, res = tag_pose(cam, d[cm.CAR_ID], ctx.side, ctx.tag_h)
            r.update(tag_seen=True, tag_x=x, tag_y=y, tag_psi=a, tag_res_sim_m=res)
        for b in boxes:
            u, v, c = b.match(g)
            xy = cam.pix2floor(np.array([[u, v]]), ctx.h_than)[0]
            r['box'].append((float(xy[0]), float(xy[1]), c, u, v))
    return r, cam


def frame_ok(r, n_boxes, cfg):
    """Full frame gate of a placement clip (2.1)."""
    vals = [r['tag_x'], r['tag_y'], r['tag_psi'], r['tag_res_sim_m']] + [v for b in r['box'] for v in b[:3]]
    return bool(r['gate_marks'] and r['tag_seen'] and len(r['box']) == n_boxes and all(math.isfinite(v) for v in vals)
                and r['tag_res_sim_m'] <= cfg['tag_res_sim_m'] and all(b[2] >= cfg['corr_min'] for b in r['box']))


def measure_clip(video, ctx, boxes, stride=1, max_frames=None):
    """Per-frame rows of one clip, every stride-th frame."""
    cam, rows = ctx.cam0, []
    for k, t, g in cm.iter_frames(video, max_frames):
        if k % stride:
            continue
        r, cam = measure_frame(ctx, cam, k, t, g, boxes)
        rows.append(r)
    return rows


def pick_template_frame(video, ctx, n, tries=12, step=15):
    """Index and grey image of the gated frame of N0 nearest its middle (NT-9): middle, middle +- step, ..."""
    mid = n // 2
    order = [mid] + [mid + s * step * q for q in range(1, tries) for s in (-1, 1)]
    order = [i for i in order if 0 <= i < n][:tries]
    frames = frames_of(video, order)
    for i in order:
        g = frames.get(i)
        if g is None:
            continue
        r, _ = measure_frame(ctx, ctx.cam0, i, 0.0, g, [])
        if r['gate_marks'] and r['tag_seen'] and r['tag_res_sim_m'] <= ctx.cfg['tag_res_sim_m']:
            return i, g
    raise ValueError(f'{video}: no gated frame near the middle for the box templates')


# ---------------------------------------------------------------- per clip
def _trimmed(rows, cfg):
    if not rows:
        return []
    t0, t1 = rows[0]['t_s'], rows[-1]['t_s']
    return [r for r in rows if t0 + cfg['trim_s'] <= r['t_s'] <= t1 - cfg['trim_s']]


def clip_summary(rows, n_boxes, cfg, stride=1):
    """Medians of one placement clip, stillness, used frames (design 2.2)."""
    out = dict(n_frames=len(rows), flags=[])
    kept = _trimmed(rows, cfg)
    used = [r for r in kept if frame_ok(r, n_boxes, cfg)]
    need = int(math.ceil(cfg['min_frames_s1'] / stride))
    out.update(n_considered=len(kept), n_used=len(used), n_needed=need)
    if not kept or len(used) < need or len(used) < cfg['min_frac'] * len(kept):
        out.update(ok=False, flags=['thieu_khung'])
        return out
    x = np.array([r['tag_x'] for r in used])
    y = np.array([r['tag_y'] for r in used])
    psi = [r['tag_psi'] for r in used]
    mx, my, mp = float(np.median(x)), float(np.median(y)), circ_median(psi)
    dpos = np.hypot(x - mx, y - my)
    dang = np.array([abs(math.degrees(wrap(a - mp))) for a in psi])
    bxy = np.array([[b[:2] for b in r['box']] for r in used])          # (n, boxes, 2)
    bmed = np.median(bxy, axis=0)
    bdev = np.linalg.norm(bxy - bmed[None], axis=2)                   # (n, boxes)
    p95 = lambda a: float(np.percentile(a, 95))                        # noqa: E731
    box95 = [p95(bdev[:, i]) for i in range(n_boxes)]
    still = (p95(dpos) <= cfg['still_tag_m'] and p95(dang) <= cfg['still_tag_deg']
             and all(v <= cfg['still_box_m'] for v in box95))
    if not still:
        out['flags'].append('clip_co_dong')
    cam = [float(np.median([r[c] for r in used])) for c in ('cam_x', 'cam_y', 'cam_z')]
    out.update(ok=still, still=still, tag=[mx, my, mp], tag_p95_m=p95(dpos), tag_p95_deg=p95(dang),
               tag_max_m=float(dpos.max()), tag_max_deg=float(dang.max()), box=bmed.tolist(), box_p95_m=box95,
               corr_min=[float(min(r['box'][i][2] for r in used)) for i in range(n_boxes)], cam=cam,
               rms_px_median=float(np.median([r['rms_px'] for r in used])))
    return out


def rong_summary(rows, ref_box, ref_tag, n_boxes, cfg, stride=1):
    """Clip R (2.3): the boxes are found where they were in N0 while the tag is away or gone -> they are body."""
    kept = _trimmed(rows, cfg)
    ok_rows = []
    for r in kept:
        if not r['gate_marks'] or len(r['box']) != n_boxes:
            continue
        away = (not r['tag_seen']) or math.hypot(r['tag_x'] - ref_tag[0], r['tag_y'] - ref_tag[1]) >= cfg['r_tag_away_m']
        found = all(b[2] >= cfg['corr_min'] and math.isfinite(b[0]) for b in r['box'])
        if away and found:
            ok_rows.append(r)
    need = int(math.ceil(cfg['r_min_frames_s1'] / stride))
    passed = bool(kept) and len(ok_rows) >= need and len(ok_rows) >= cfg['min_frac'] * len(kept)
    out = dict(n_considered=len(kept), n_ok=len(ok_rows), n_needed=need, dat=passed)
    if ok_rows:
        bmed = np.median(np.array([[b[:2] for b in r['box']] for r in ok_rows]), axis=0)
        out['box'] = bmed.tolist()
        out['lech_vung_so_N0_m'] = [float(np.hypot(*(bmed[i] - np.array(ref_box[i])))) for i in range(n_boxes)]
    return out


# ---------------------------------------------------------------- body motion, repeatability
def body_motion(ref_pts, pts):
    """Rigid 2D motion of the body N0 -> Nj from the box points: (angle rad, t (2,), rms m) with pts ~ R ref + t."""
    ang, t, _, rms = cm.rigid_fit(np.asarray(ref_pts, float), np.asarray(pts, float))
    return ang, np.asarray(t, float), rms


def to_body0(tag, motion):
    """Tag pose of placement j mapped back to the body frame of N0: p' = R^T (p - t), psi' = psi - ang (NT-3)."""
    ang, t, _ = motion
    c, s = math.cos(ang), math.sin(ang)
    p = np.array(tag[:2], float) - t
    return [c * p[0] + s * p[1], -s * p[0] + c * p[1], wrap(tag[2] - ang)]


def pairwise(poses, eps=None):
    """Largest |p_j - p_k| + eps_j + eps_k (m) and |wrap(psi_j - psi_k)| (deg) over all pairs, with the pairs."""
    eps = eps or [0.0] * len(poses)
    best_p, best_a, pp, pa = 0.0, 0.0, None, None
    for j, k in itertools.combinations(range(len(poses)), 2):
        dp = math.hypot(poses[j][0] - poses[k][0], poses[j][1] - poses[k][1]) + eps[j] + eps[k]
        da = abs(math.degrees(wrap(poses[j][2] - poses[k][2])))
        if dp >= best_p:
            best_p, pp = dp, [j, k]
        if da >= best_a:
            best_a, pa = da, [j, k]
    return dict(vi_tri_m=best_p, cap_vi_tri=pp, goc_deg=best_a, cap_goc=pa)


def analyse(summaries, cfg, rong=None, tag_h=0.12):
    """Design 2.3, 2.4 on the placement summaries (all ok). rong: summary of clip R (None = no clip R)."""
    raw = [s['tag'] for s in summaries]
    nb = len(summaries[0]['box'])
    why = []
    if nb < 2:
        why.append('it_hon_2_vung')
    if rong is None:
        why.append('khong_co_clip_R')
    elif not rong['dat']:
        why.append('clip_R_khong_dat')
    motions = []
    for j, s in enumerate(summaries):
        m = body_motion(summaries[0]['box'], s['box']) if nb >= 2 else (0.0, np.zeros(2), NAN)
        motions.append(m)
        if nb >= 2 and not m[2] <= cfg['rigid_res_m']:
            why.append(f'N{j}_du_khop_{m[2] * 1000:.2f}mm')
    kiem_than = not why
    c0 = np.array(summaries[0]['cam'][:2])
    H = summaries[0]['cam'][2]
    eps, body = [], []
    for j, (s, m) in enumerate(zip(summaries, motions)):
        dc = float(np.hypot(*(np.array(s['cam'][:2]) - c0)))
        tj = float(np.hypot(*m[1])) if nb >= 2 else NAN
        eps.append((tj + dc) * cfg['dh_m'] / (H - tag_h) if nb >= 2 else NAN)
        body.append(dict(lan=f'N{j}', dich_m=tj, quay_deg=math.degrees(m[0]) if nb >= 2 else NAN,
                         du_khop_m=m[2], camera_dich_m=dc, eps_m=eps[-1]))
    res = dict(tho=pairwise(raw), kiem_than=kiem_than, ly_do_khong_kiem_than=why, than=body)
    if nb >= 2:
        corr = [to_body0(raw[j], motions[j]) for j in range(len(raw))]
        res['tru_than'] = dict(pairwise(corr, eps), tu_the=corr)
        res['than_dich_max_m'] = max(b['dich_m'] for b in body)
        res['than_dich_max_deg'] = max(abs(b['quay_deg']) for b in body)
    else:
        res.update(tru_than=None, than_dich_max_m=None, than_dich_max_deg=None)
    use = res['tru_than'] if kiem_than else res['tho']
    res.update(vi_tri_m=use['vi_tri_m'], goc_deg=use['goc_deg'], so_dung='tru_than' if kiem_than else 'tho')
    return res


GIOI_HAN = ('Do lap QUAN SAT DUOC cua {n} lan dat (mau nho; khong phai can moi lan dat sau, khong nhan he so; le 1,25 do '
            'c8b_u; so dung cho chien dich do CODE_C8B/Liam quyet khi khoa U). Tam cao/nghieng on dinh tren chot: bao boi P6 '
            '(B1 tren lan dat cuoi). Chot mon qua cac buoi chua bao. Do tai mot cho (goc O). Can sai do cao vung than dh '
            '{dh} m [uoc luong]; goc khong cong eps.')
GIOI_HAN_THO = (' KHONG kiem duoc than ({why}): so tho gom ca than dich khi nhac tam; than dich cung luc tam dich nguoc '
                'chieu co the triet tieu (trang v10 muc 3).')


def block(res, clips, rong, session_path, session_sha, cfg):
    """Block nhac_tam for c8_cam_u_vao (C8B o U v10: nhac_tam.vi_tri_m, nhac_tam.goc_deg; other keys information)."""
    t = res['tru_than']
    g = GIOI_HAN.format(n=len(clips), dh=cfg['dh_m'])
    if not res['kiem_than']:
        g += GIOI_HAN_THO.format(why=', '.join(res['ly_do_khong_kiem_than']))
    src = [dict(file=c['file'], sha256=c['sha256']) for c in clips]
    if rong is not None:
        src.append(dict(file=rong['file'], sha256=rong['sha256']))
    src.append(dict(file=session_path, sha256=session_sha))
    return dict(
        vi_tri_m=res['vi_tri_m'], goc_deg=res['goc_deg'],
        trang_thai='dat' if res['kiem_than'] else 'dat_gioi_han',
        vi_tri_tho_m=res['tho']['vi_tri_m'], goc_tho_deg=res['tho']['goc_deg'],
        vi_tri_tru_than_m=None if t is None else t['vi_tri_m'], goc_tru_than_deg=None if t is None else t['goc_deg'],
        kiem_than=res['kiem_than'], than_dich_max_m=res['than_dich_max_m'], than_dich_max_deg=res['than_dich_max_deg'],
        n_lan_dat=len(clips),
        cach=('lon nhat qua moi cap cua cac lan dat; ma va vung than chieu cung camera tung khung; da tru than '
              'p\'=R^T(p-t) cong eps_j+eps_k (thiet ke c7_nhac_tam v2 muc 2.4)' if res['kiem_than'] else
              'lon nhat qua moi cap cua cac lan dat, so THO (thiet ke c7_nhac_tam v2 muc 2.4)'),
        gioi_han=g, nguon=src)


# ---------------------------------------------------------------- report, frames file
def report(out):
    L = [f'Nhấc tấm C8 ({out["version"]}): {out["ket_luan"]}']
    for c in out['clips']:
        s = c['tom_tat']
        if not s.get('ok'):
            L.append(f'  {c["lan"]} {c["file"]}: HỎNG ({", ".join(s["flags"])}; dùng {s.get("n_used", 0)}/'
                     f'{s.get("n_needed", "?")} khung) → quay lại clip này, KHÔNG nhấc tấm')
            continue
        L.append(f'  {c["lan"]} {c["file"]}: {s["n_used"]} khung; mã lệch (phân vị 95 %) {s["tag_p95_m"] * 1000:.2f} mm, '
                 f'{s["tag_p95_deg"]:.3f}°; tương quan vùng nhỏ nhất {min(s["corr_min"]):.3f}')
    rg = out.get('rong')
    if rg is not None:
        L.append(f'  R {rg["file"]}: {"đạt" if rg["tom_tat"]["dat"] else "KHÔNG ĐẠT"} '
                 f'({rg["tom_tat"]["n_ok"]}/{rg["tom_tat"]["n_needed"]} khung)')
    r = out.get('ket_qua')
    if r:
        L.append(f'Thô (mã trên sàn): vị trí {r["tho"]["vi_tri_m"] * 1000:.2f} mm, góc {r["tho"]["goc_deg"]:.3f}°')
        if r['tru_than']:
            L.append(f'Đã trừ thân (+ε): vị trí {r["tru_than"]["vi_tri_m"] * 1000:.2f} mm, góc '
                     f'{r["tru_than"]["goc_deg"]:.3f}°; thân dịch lớn nhất {r["than_dich_max_m"] * 1000:.2f} mm, '
                     f'{r["than_dich_max_deg"]:.3f}°')
        if not r['kiem_than']:
            L.append(f'Không kiểm được thân ({", ".join(r["ly_do_khong_kiem_than"])}): dùng số thô, có giới hạn')
        L.append(f'Đưa vào U: P3 = {r["vi_tri_m"] * 1000:.2f} mm, A2 = {r["goc_deg"]:.3f}° ({r["so_dung"]})')
    if out['co']:
        L.append('Cờ: ' + ', '.join(out['co']))
    return '\n'.join(L) + '\n'


FRAME_COLS = ('lan', 'idx', 't_s', 'n_marks', 'rms_px', 'cam_x', 'cam_y', 'cam_z', 'tag_seen', 'tag_x', 'tag_y',
              'tag_psi', 'tag_res_sim_m')


def write_frames(path, named_rows):
    nb = max((len(r['box']) for _, rows in named_rows for r in rows), default=0)
    cols = list(FRAME_COLS) + [f'vung{i}_{q}' for i in range(nb) for q in ('x', 'y', 'corr', 'u', 'v')]

    def f(v, n):
        return '' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f'{v:.{n}f}'
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, rows in named_rows:
            for r in rows:
                v = [name, r['idx'], f(r['t_s'], 6), r['n_marks'], f(r['rms_px'], 4), f(r['cam_x'], 7), f(r['cam_y'], 7),
                     f(r['cam_z'], 7), int(r['tag_seen']), f(r['tag_x'], 7), f(r['tag_y'], 7), f(r['tag_psi'], 7),
                     f(r['tag_res_sim_m'], 7)]
                for b in r['box']:
                    v += [f(b[0], 7), f(b[1], 7), f(b[2], 5), f(b[3], 4), f(b[4], 4)]
                v += [''] * (len(cols) - len(v))
                w.writerow(v)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return cm._json(o)


# ---------------------------------------------------------------- commands
def run_do(clips, session_path, boxes_spec, out_dir, rong=None, h_than=None, stride=1, max_frames=None, cfg=None):
    """Whole analysis; writes the outputs; returns the exit code (0 dat, 1 dat_gioi_han, 2 blocking error)."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    if len(clips) < MIN_PLACEMENTS:
        raise ValueError(f'need at least {MIN_PLACEMENTS} placement clips in order (N0 N1 ...), got {len(clips)}')
    if stride < 1:
        raise ValueError('stride >= 1')
    cm._new_dir(out_dir)
    with open(session_path, encoding='utf-8') as f:
        session = json.load(f)
    ctx = Ctx(session, session['tag_h_m'] if h_than is None else h_than, cfg)
    boxes_px = parse_boxes(boxes_spec)
    n0 = frame_count(clips[0])
    tidx, tframe = pick_template_frame(clips[0], ctx, n0)
    boxes = [Box(tframe, b, cfg['search_px']) for b in boxes_px]
    named_rows, clip_out = [], []
    for j, v in enumerate(clips):
        rows = measure_clip(v, ctx, boxes, stride, max_frames)
        named_rows.append((f'N{j}', rows))
        clip_out.append(dict(lan=f'N{j}', file=os.path.basename(v), sha256=cm.sha256(v),
                             tom_tat=clip_summary(rows, len(boxes), cfg, stride)))
    rong_out = None
    flags = [f'{c["lan"]}_{x}' for c in clip_out for x in c['tom_tat']['flags']]
    if rong is not None:
        rows = measure_clip(rong, ctx, boxes, stride, max_frames)
        named_rows.append(('R', rows))
        s0 = clip_out[0]['tom_tat']
        rs = rong_summary(rows, s0.get('box', []), s0.get('tag', [NAN, NAN]), len(boxes), cfg, stride) \
            if s0.get('ok') else dict(dat=False, n_ok=0, n_needed=0, n_considered=0)
        rong_out = dict(file=os.path.basename(rong), sha256=cm.sha256(rong), tom_tat=rs)
    out = dict(version=VERSION, session=os.path.abspath(session_path), session_sha256=cm.sha256(session_path),
               hil=git_head(), h_than_m=ctx.h_than, h_than_src='tag_h_m cua session' if h_than is None else '--h-than',
               vung_than_px=[list(b) for b in boxes_px], khung_mau=dict(clip='N0', idx=tidx), stride=stride,
               nguong=dict(cfg, trang_thai=THRESH_STATUS), clips=clip_out, rong=rong_out, ket_qua=None)
    write_frames(os.path.join(out_dir, 'frames_nhac_tam.csv'), named_rows)
    if not all(c['tom_tat']['ok'] for c in clip_out):
        out.update(co=flags, ket_luan='LỖI CHẶN: có clip hỏng (không ghi khối U)', ma=2)
    else:
        res = analyse([c['tom_tat'] for c in clip_out], cfg, None if rong_out is None else rong_out['tom_tat'],
                      ctx.tag_h)
        out['ket_qua'] = res
        out['co'] = flags + ([] if res['kiem_than'] else ['khong_kiem_than'])
        out['ma'] = 0 if res['kiem_than'] else 1
        out['ket_luan'] = 'đạt' if res['kiem_than'] else 'đạt có giới hạn (không kiểm được thân)'
        blk = block(res, clip_out, rong_out, out['session'], out['session_sha256'], cfg)
        with open(os.path.join(out_dir, 'c8_phan_nhac_tam.json'), 'w', encoding='utf-8') as f:
            json.dump(_jsonable(blk), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'nhac_tam.json'), 'w', encoding='utf-8') as f:
        json.dump(_jsonable(out), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'bao_cao.txt'), 'w', encoding='utf-8') as f:
        f.write(report(out))
    return out['ma']


def propose_boxes(gray, tag_c, tag_side_px, n=4, size=40, exclude=0.75, reach=3.0):
    """Boxes (u, v, w, h) of strong texture around the car tag, outside a circle of exclude x tag side and within
    reach x tag side; greedy, at least one box size apart. Suggestions only: the scorer picks boxes inside the tape."""
    g = gray.astype(np.float32)
    ev = cv2.cornerMinEigenVal(g, 3)
    score = cv2.boxFilter(ev, -1, (size, size))
    H, W = g.shape
    vv, uu = np.mgrid[0:H, 0:W]
    rr = np.hypot(uu - tag_c[0], vv - tag_c[1])
    mask = (rr > exclude * tag_side_px + size * 0.71) & (rr < reach * tag_side_px)
    mask &= (uu > size) & (uu < W - size) & (vv > size) & (vv < H - size)
    score = np.where(mask, score, -1.0)
    out = []
    for _ in range(n):
        k = int(np.argmax(score))
        v, u = divmod(k, W)
        if score[v, u] <= 0:
            break
        out.append((int(u - size // 2), int(v - size // 2), size, size))
        score[max(0, v - size):v + size, max(0, u - size):u + size] = -1.0
    return out


def run_khung(clip, session_path, out_dir):
    cm._new_dir(out_dir)
    mid = frame_count(clip) // 2
    g = frames_of(clip, [mid])[mid]
    d = cm.detect_refined(cm.make_detector(), g)
    if cm.CAR_ID not in d:
        raise ValueError(f'{clip}: car tag not seen in frame {mid}')
    c = d[cm.CAR_ID]
    ctr = c.mean(axis=0)
    side = float(np.mean(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)))
    half = int(3.2 * side)
    u0, v0 = max(0, int(ctr[0]) - half), max(0, int(ctr[1]) - half)
    u1, v1 = min(g.shape[1], int(ctr[0]) + half), min(g.shape[0], int(ctr[1]) + half)
    crop = g[v0:v1, u0:u1]
    boxes = [(u + u0, v + v0, w, h) for u, v, w, h in propose_boxes(crop, ctr - [u0, v0], side)]
    img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for x in range((u0 // 50 + 1) * 50, u1, 50):
        cv2.line(img, (2 * (x - u0), 0), (2 * (x - u0), img.shape[0]), (0, 160, 255), 1)
        cv2.putText(img, str(x), (2 * (x - u0) + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)
    for y in range((v0 // 50 + 1) * 50, v1, 50):
        cv2.line(img, (0, 2 * (y - v0)), (img.shape[1], 2 * (y - v0)), (0, 160, 255), 1)
        cv2.putText(img, str(y), (2, 2 * (y - v0) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)
    for i, (u, v, w, h) in enumerate(boxes):
        cv2.rectangle(img, (2 * (u - u0), 2 * (v - v0)), (2 * (u + w - u0), 2 * (v + h - v0)), (0, 255, 0), 2)
        cv2.putText(img, f'de xuat {i}', (2 * (u - u0), 2 * (v - v0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, 'khung_N0.png'), img)
    info = dict(version=VERSION, clip=os.path.basename(clip), sha256=cm.sha256(clip), idx=mid,
                tag_centre_px=ctr.tolist(), tag_side_px=side, crop_px=[u0, v0, u1, v1], anh_phong=2,
                o_de_xuat=[list(b) for b in boxes],
                ghi_chu='O de xuat chi la goi y (ket cau manh); chon o nam tron trong mieng bang dinh tren than')
    with open(os.path.join(out_dir, 'khung_N0.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('khung')
    a1.add_argument('--clip', required=True)
    a1.add_argument('--session', required=True)
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('do')
    a2.add_argument('--clip', nargs='+', required=True)
    a2.add_argument('--rong', help='clip R: plate off the car (design 2.3); without it kiem_than is false')
    a2.add_argument('--session', required=True)
    a2.add_argument('--vung-than', required=True)
    a2.add_argument('--h-than', type=float)
    a2.add_argument('--stride', type=int, default=1)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'khung':
            print(json.dumps(run_khung(a.clip, a.session, a.out), ensure_ascii=False))
            code = 0
        else:
            code = run_do(a.clip, a.session, a.vung_than, a.out, a.rong, a.h_than, a.stride)
            with open(os.path.join(a.out, 'bao_cao.txt'), encoding='utf-8') as f:
                sys.stdout.write(f.read())
    except (OSError, ValueError, KeyError) as e:
        print(f'LOI: {e}', file=sys.stderr)
        code = 2
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
