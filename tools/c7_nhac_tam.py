#!/usr/bin/env python3
"""
c7_nhac_tam.py: plate lift test of C8 (trang C8b v10 DA DUYET, P3 / A2, section 3 'Nhac tam') -> repeatability of the
car tag position and heading RELATIVE TO THE CAR BODY over the placements of the tag plate; the body is watched through
two textured tape patches stuck on the body.

Design: CLAUDE_THESIS 03_CODE/docs/2026-11-18_Thiet_ke_c7_nhac_tam_v4.md (CODE_NHACTAM local_f8bebd94, 18/11/2026;
v2 after Codex Sol round 1, NT-1..NT-14; v3 after round 2, NT2-1..NT2-4; v4 after round 3, NT3-1..NT3-4; v5 after Codex code round 1, NC-1..NC-9; v6 after code round 2, NC2-1..NC2-4; v7 after code round 3, NC3-1..NC3-2). Measurement (design 0, 5): car still at O, control power off;
N0 = plate fitted on its pins, N1.. = lifted and put back, the last one after the calliper heights (B1); one clip of
10 s per placement, nobody in the frame; clip R = plate lying on the floor >= 0.5 m from the car (5 s).

Per frame (2.1): marks of the session's own fit (session['fit_ids']) -> camera pose refitted (c7_measure.refit_pose);
the 4 corners of the car tag (ID 0) AND the body boxes are taken to their planes with the SAME frame camera
(pix2floor, z = tag_h for the tag, z = h_than for the boxes; NT-6); tag = rigid fit to the square of car_side_m.
Box: template of a gated frame of N0 nearest its middle, matched (TM_CCOEFF_NORMED, parabolic peak, then ECC
translation). Frame gates (NT-9): finite, >= 8 marks, mark rms <= rms of clip H + 0.5 px, tag fit residual with
scale <= 1.5 mm, every box correlation >= 0.9.
Per clip (2.2): 0.5 s dropped at both ends, every stride-th frame; >= max(100, ceil(300 / stride)) gated frames (U block only at stride <= 3) and >= 60 % of
the frames considered; medians; still if the 95th percentile of the deviations is within the thresholds.
Body (2.3): rigid 2D fit of the box points N0 -> Nj; clip R proves the boxes belong to the body (boxes found where they
were in N0 while the tag is SEEN >= 0.2 m away, NT2-3); kiem_than = R passed, two boxes >= 0.10 m apart, fit residual ok.
Repeatability (2.4): tag mapped back to the N0 body frame p' = R^T (p - t), psi' = psi - angle(R); largest over pairs
of |p'_j - p'_k| + eps_j + eps_k (eps_j = (|t_j| + |dC_xy,j| + |(R_j - I) C0|) dh / (H - h) + dh |dH_j| / (H - h)^2
|qbar_0 - C0| + max|1 - 1/r| |p_j - C0| over r in the scale interval; NT2-1, NT3-3, NT-7, NT3-1, NT3-2) and of
|wrap(psi'_j - psi'_k)|. Value for U: body removed when kiem_than, else raw (limitation of v10 section 3).

  python tools/c7_nhac_tam.py khung --clip N0.MOV --session S/session.json --out D
        -> D/khung_N0.png (crop around the car, pixel grid, proposed boxes), D/khung_N0.json
  python tools/c7_nhac_tam.py kiem --clip X.MOV --mau N0.MOV --mau-idx IDX [--rong] --session S/session.json \
        --vung-than "u,v,w,h;u,v,w,h" [--stride 3] --out D
        -> one clip checked at the floor BEFORE the plate is lifted again (NT2-4): prints DUOC / QUAY LAI; ma=0 / 2
  python tools/c7_nhac_tam.py do --clip N0.MOV N1.MOV N2.MOV N3.MOV N4.MOV --rong R.MOV --session S/session.json \
        --vung-than "u,v,w,h;u,v,w,h" --mau-idx IDX [--h-than M] [--stride K<=3] [--loai FILE=REASON ...] --out D
  IDX = 'idx' of khung_N0.json: the boxes are picked on that very frame (NC2-2)
        -> D/nhac_tam.json, D/frames_nhac_tam.csv, D/bao_cao.txt and, without a blocking error,
           D/c8_phan_nhac_tam.json (block for c7_kiem_cheo xuat --phan nhac_tam=...; trang_thai dat / dat_gioi_han)
Boxes are pixel boxes (left, top, width, height) in the frames of N0, each inside one tape patch. Thresholds [DE XUAT].

v4 (placement of the plate per run; design 2026-11-19_Thiet_ke_nhac_tam_lan_lap_v4.md, DA DUYET; CODE_NHACTAM
local_68e19299, 19/11/2026); khung, kiem, do above unchanged:
  python tools/c7_nhac_tam.py hieu_chuan --cua-so W.json --session S/session.json --neo "A:u,v,w,h;B:...;oc1:..." \
        --mau CLIP SESSION IDX --kiem-r R/kiem.json ... --a6 A6.json --tam-id T [--h-neo M] --out D
        -> D/hieu_chuan_neo.json (F6), fit windows only (12b, E), locked before the held-out windows
  python tools/c7_nhac_tam.py lan_lap --hieu-chuan D/hieu_chuan_neo.json --cua-so W.json --session S/session.json \
        --a6 A6.json --tam-id T [--truc-xe T.json] [--xoay X.json] [--u-beta-cam-deg U] --out D2
        -> D2/lan_lap.json (lan_lap_v1, F8)
Anchor boxes are found in a window around the place predicted from the window's tag pose (median; carried from the
previous window of the clip when the tag is lost), the template turned by the image rotation.
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

VERSION = 'c7_nhac_tam_v7'
MIN_PLACEMENTS = 4
DEFAULTS = dict(
    trim_s=0.5,            # 2.2: dropped at both ends of a clip
    min_marks=8,           # 2.1: floor marks for the frame camera
    rms_extra_px=0.5,      # 2.1: mark rms <= rms of clip H + this
    tag_res_sim_m=0.0015,  # 2.1: residual of the tag fit WITH scale (gate (2a) of the scorer)
    corr_min=0.9,          # 2.1: every body box must correlate at least this
    peak2_ratio=0.9,       # 2.1: second peak (outside +-3 px) below this x the main peak (unambiguous match)
    min_frames_s1=300,     # 2.2: gated frames: max(min_frames_abs, ceil(300 / stride))
    min_frames_abs=100,
    max_stride=3,          # 2.2: the U block only at stride <= 3 (NT2-2)
    min_box_sep_m=0.10,    # 2.3: the two boxes on two patches (NT-1)
    min_frac=0.6,          #      and share of the frames considered
    still_tag_m=0.0005,    # 2.2: 95th percentile of the deviation from the clip median, tag centre
    still_tag_deg=0.3,     #      tag heading
    still_box_m=0.0005,    #      each body box
    rigid_res_m=0.0003,    # 2.3: rms of the rigid fit of the box points N0 -> Nj
    r_tag_away_m=0.2,      # 2.3: clip R, tag at least this far from its N0 place (or not seen)
    r_min_frames_s1=100,   #      gated R frames: max(r_min_frames_abs, ceil(100 / stride))
    r_min_frames_abs=40,
    dh_m=0.03,             # 2.4: height bound of the boxes [uoc luong]
    scale_k=3.0,           # 2.4: scale interval = median +- k x standard error of the median (NT3-1)
    search_px=60,          # 2.1: template search half-window (about 9 cm); a peak on its border is rejected (NC-1)
    still_scale_rel=0.003, # 2.2: 95th percentile of |scale / median - 1| (NC-3: a scattered scale is not a still clip)
    a0_min=0.996,          # 2.4: lower bound of the true scale of N0, |height error| <= 1 cm [uoc luong] (NC-2)
    dz_tag_m=0.01,         # 2.4: the same 1 cm bound on the tag height, used in eps with dh (NC3-1)
)
THRESH_STATUS = 'DE XUAT (thiet ke c7_nhac_tam v4 muc 2)'
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
        """(u, v) of the box centre in this frame, correlation at the peak, ratio second peak / peak (second peak =
        best value outside +-3 px of the main one; NT-9 ambiguous match)."""
        u, v, w, h = self.box
        s = self.search
        win = gray[v - s:v + h + s, u - s:u + w + s].astype(np.float32)
        r = cv2.matchTemplate(win, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if loc[0] in (0, r.shape[1] - 1) or loc[1] in (0, r.shape[0] - 1):
            return NAN, NAN, NAN, 1.0             # NC-1: peak on the window border, the true one may lie outside
        r2 = r.copy()
        r2[max(0, loc[1] - 3):loc[1] + 4, max(0, loc[0] - 3):loc[0] + 4] = -1.0
        ratio = float(r2.max() / mx) if mx > 0 else 1.0
        x, y = subpix(r, *loc)
        e = self.ecc(win, x, y)
        if e is None:
            return NAN, NAN, NAN, 1.0             # NC-1: ECC failed or moved more than 1 px: no trusted sub-pixel peak
        x, y = e
        return self.u0 + x - s, self.v0 + y - s, float(mx), ratio

    def ecc(self, win, x, y):
        """Sub-pixel translation by ECC (cv2.findTransformECC) started at the correlation peak; None when ECC fails or
        moves more than 1 px from it (NC-1: the match is then not used)."""
        warp = np.array([[1.0, 0.0, x], [0.0, 1.0, y]], np.float32)
        try:
            _, warp = cv2.findTransformECC(self.tpl, win, warp, cv2.MOTION_TRANSLATION,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 1)
        except cv2.error:
            return None
        ex, ey = float(warp[0, 2]), float(warp[1, 2])
        if not (math.isfinite(ex) and math.isfinite(ey)) or math.hypot(ex - x, ey - y) > 1.0:
            return None
        return ex, ey


def frames_of(video, wanted, with_time=False):
    """{index: grey frame} (or {index: (t_s, grey frame)} with with_time) for the wanted indices, one decoding pass."""
    out, wanted = {}, set(wanted)
    for k, t, g in cm.iter_frames(video):
        if k in wanted:
            out[k] = (t, g) if with_time else g
        if len(out) == len(wanted):
            break
    return out


def check_template(g, box, cfg):
    """A box is usable on frame g only if it builds (edge margin, texture) and its own match passes the box gate:
    correlation, unambiguous peak (a periodic pattern fails), ECC (NC2-2)."""
    b = Box(g, box, cfg['search_px'])
    u, v, c, q = b.match(g)
    if not box_ok((0.0, 0.0, c, u, v, q), cfg):
        raise ValueError(f'box {box}: not usable on the template frame (corr {c}, second peak ratio {q:.3f})')
    return b


def template_at(video, ctx, idx, boxes_px):
    """The template frame given by its index (from khung_N0.json): every box must be usable there (NC2-2)."""
    tg = frames_of(video, [idx], with_time=True).get(idx)
    if tg is None:
        raise ValueError(f'{video}: no frame {idx}')
    t, g = tg
    r, _ = measure_frame(ctx, ctx.cam0, idx, t, g, [])                    # real PTS (NC3-2)
    if not (core_finite(r) and r['gate_marks'] and r['tag_seen'] and r['tag_res_sim_m'] <= ctx.cfg['tag_res_sim_m']):
        raise ValueError(f'{video}: frame {idx} does not pass the gates')
    return idx, g, [check_template(g, b, ctx.cfg) for b in boxes_px]


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
        self.shape = (int(session['height']), int(session['width']))
        self.h_than = float(h_than)
        self.cfg = cfg
        self.det = cm.make_detector()


def tag_pose(cam, corners_px, side, tag_h):
    """Tag centre, heading, fit residual with scale, from the 4 corners taken to z = tag_h with the frame camera
    (NT-6: the same projection as the boxes)."""
    P = cam.pix2floor(np.asarray(corners_px, float), tag_h)
    body = cm.car_corners_body(side)
    ang, t, _, _ = cm.rigid_fit(body, P)
    _, _, sc, res_sim = cm.rigid_fit(body, P, with_scale=True)
    return float(t[0]), float(t[1]), float(ang), float(res_sim), float(sc)       # sc: apparent scale (NT-7)


def measure_frame(ctx, cam_prev, k, t, g, boxes):
    """One row (2.1) and the camera to start the next refit from."""
    if g.shape != ctx.shape:
        raise ValueError(f'frame {k}: size {g.shape[::-1]} differs from the session {ctx.shape[::-1]}')
    d = cm.detect_refined(ctx.det, g)
    obj, img, used = cm.floor_obs(d, ctx.marks, ids=ctx.fit_ids)
    r = dict(idx=k, t_s=t, n_marks=len(used), rms_px=NAN, cam_x=NAN, cam_y=NAN, cam_z=NAN, tag_seen=False,
             tag_x=NAN, tag_y=NAN, tag_psi=NAN, tag_res_sim_m=NAN, tag_scale=NAN, box=[], gate_marks=False)
    cam = cam_prev
    if len(used) >= 4:
        cam, r['rms_px'] = cm.refit_pose(cam_prev, obj, img)
        r['cam_x'], r['cam_y'], r['cam_z'] = (float(v) for v in cam.centre)
        r['gate_marks'] = bool(len(used) >= ctx.cfg['min_marks'] and math.isfinite(r['rms_px'])
                               and r['rms_px'] <= ctx.rms_h + ctx.cfg['rms_extra_px'])
        if cm.CAR_ID in d:
            x, y, a, res, sc = tag_pose(cam, d[cm.CAR_ID], ctx.side, ctx.tag_h)
            r.update(tag_seen=True, tag_x=x, tag_y=y, tag_psi=a, tag_res_sim_m=res, tag_scale=sc)
        for b in boxes:
            u, v, c, q = b.match(g)
            xy = cam.pix2floor(np.array([[u, v]]), ctx.h_than)[0]
            r['box'].append((float(xy[0]), float(xy[1]), c, u, v, q))
    return r, cam


def core_finite(r):
    """Finite time and camera centre: shared by every gate (placement frames, clip R, template frame; NC2-4)."""
    return all(math.isfinite(r[k]) for k in ('t_s', 'cam_x', 'cam_y', 'cam_z', 'rms_px'))


def box_ok(b, cfg):
    """Box match gate (2.1): finite (floor point, correlation, pixel), correlation, unambiguous peak."""
    return bool(all(math.isfinite(v) for v in b[:6]) and b[2] >= cfg['corr_min'] and b[5] < cfg['peak2_ratio'])


def frame_ok(r, n_boxes, cfg):
    """Full frame gate of a placement clip (2.1); every number used later must be finite (NC-4)."""
    vals = [r['tag_x'], r['tag_y'], r['tag_psi'], r['tag_res_sim_m'], r['tag_scale']]
    return bool(core_finite(r) and r['gate_marks'] and r['tag_seen'] and len(r['box']) == n_boxes and all(math.isfinite(v) for v in vals)
                and r['tag_res_sim_m'] <= cfg['tag_res_sim_m'] and all(box_ok(b, cfg) for b in r['box']))


def measure_clip(video, ctx, boxes, stride=1, max_frames=None):
    """Per-frame rows of one clip, every stride-th frame."""
    cam, rows = ctx.cam0, []
    for k, t, g in cm.iter_frames(video, max_frames):
        if k % stride:
            continue
        r, cam = measure_frame(ctx, cam, k, t, g, boxes)
        rows.append(r)
    return rows


def pick_template_frame(video, ctx, n, boxes_px=None, batch=16):
    """Index and grey image of the gated frame of N0 nearest its middle (NT-9, NC-5): frames in order of distance to
    the middle, in batches of `batch` decoded frames, until one passes the mark and tag gates and gives a usable
    template for every box."""
    mid = n // 2
    order = sorted(range(n), key=lambda i: (abs(i - mid), i))
    for b0 in range(0, len(order), batch):
        part = order[b0:b0 + batch]
        frames = frames_of(video, part, with_time=True)
        for i in part:
            tg = frames.get(i)
            if tg is None:
                continue
            t, g = tg
            r, _ = measure_frame(ctx, ctx.cam0, i, t, g, [])                    # real PTS (NC3-2)
            if not (core_finite(r) and r['gate_marks'] and r['tag_seen']
                    and r['tag_res_sim_m'] <= ctx.cfg['tag_res_sim_m']):              # NC2-4
                continue
            try:
                for b in boxes_px or []:
                    check_template(g, b, ctx.cfg)
            except ValueError:
                continue
            return i, g
    raise ValueError(f'{video}: no gated frame with usable box templates')


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
    need = max(cfg['min_frames_abs'], int(math.ceil(cfg['min_frames_s1'] / stride)))
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
    sc = np.array([r['tag_scale'] for r in used])
    sc95 = p95(np.abs(sc / np.median(sc) - 1.0))
    still = (p95(dpos) <= cfg['still_tag_m'] and p95(dang) <= cfg['still_tag_deg']
             and all(v <= cfg['still_box_m'] for v in box95) and sc95 <= cfg['still_scale_rel'])
    if not still:
        out['flags'].append('clip_co_dong')
    cam = [float(np.median([r[c] for r in used])) for c in ('cam_x', 'cam_y', 'cam_z')]
    scale, scale_u = float(np.median(sc)), (float(cfg['scale_k'] * 1.2533 * sc.std(ddof=1) / math.sqrt(len(sc)))
                                            if len(sc) > 1 else NAN)
    if not (math.isfinite(scale) and math.isfinite(scale_u) and scale > 0 and scale - scale_u > 0):   # NC2-3
        out['flags'].append('thang_ngoai_mien')
        still = False
    out.update(ok=still, still=still, tag=[mx, my, mp], scale=float(np.median(sc)), scale_p95_rel=sc95,
               scale_u=float(cfg['scale_k'] * 1.2533 * sc.std(ddof=1) / math.sqrt(len(sc))) if len(sc) > 1 else NAN,
               tag_p95_m=p95(dpos), tag_p95_deg=p95(dang),
               tag_max_m=float(dpos.max()), tag_max_deg=float(dang.max()), box=bmed.tolist(), box_p95_m=box95,
               corr_min=[float(min(r['box'][i][2] for r in used)) for i in range(n_boxes)], cam=cam,
               rms_px_median=float(np.median([r['rms_px'] for r in used])))
    return out


def rong_summary(rows, ref_box, ref_tag, n_boxes, cfg, stride=1):
    """Clip R (2.3): the boxes are found where they were in N0 while the tag is SEEN away (NT2-3) -> they are body."""
    kept = _trimmed(rows, cfg)
    ok_rows = []
    for r in kept:
        if not (core_finite(r) and r['gate_marks']) or len(r['box']) != n_boxes:      # NC2-4
            continue
        away = bool(r['tag_seen'] and math.isfinite(r['tag_x']) and math.isfinite(r['tag_y'])
                    and math.hypot(r['tag_x'] - ref_tag[0], r['tag_y'] - ref_tag[1]) >= cfg['r_tag_away_m'])  # NT2-3
        if away and all(box_ok(b, cfg) for b in r['box']):
            ok_rows.append(r)
    need = max(cfg['r_min_frames_abs'], int(math.ceil(cfg['r_min_frames_s1'] / stride)))
    passed = bool(kept) and len(ok_rows) >= need and len(ok_rows) >= cfg['min_frac'] * len(kept)
    out = dict(n_considered=len(kept), n_ok=len(ok_rows), n_needed=need, dat=passed,
               idx_ok=[r['idx'] for r in ok_rows])
    if ok_rows:
        bmed = np.median(np.array([[b[:2] for b in r['box']] for r in ok_rows]), axis=0)
        out['box'] = bmed.tolist()
        out['lech_vung_so_N0_m'] = [float(np.hypot(*(bmed[i] - np.array(ref_box[i])))) for i in range(n_boxes)]
    return out


# ---------------------------------------------------------------- body motion, repeatability
def scale_interval(sj, uj, s0, u0, same):
    """Interval of r = s_j / s_0 with the statistical uncertainties of both medians (NT3-1); N0 against itself is 1.
    Systematic edge bias between clips is NOT inside (limitation)."""
    if same:
        return 1.0, 1.0
    vals = (sj, uj, s0, u0)
    if not all(math.isfinite(v) for v in vals) or uj < 0 or u0 < 0 or sj - uj <= 0 or s0 - u0 <= 0:
        raise ValueError(f'thang ngoai mien: s_j {sj}, u_j {uj}, s_0 {s0}, u_0 {u0} (NC-3)')
    return (sj - uj) / (s0 + u0), (sj + uj) / (s0 - u0)


def scale_abs(rep_, cfg):
    """NC2-1: distances on the plane z = tag_h are true distances x a (a = (H - h)/(H - z) of the plate); with every
    placement at the same wrong height r = 1 and eps_h = 0, so the distance itself is divided by a0_min."""
    rep_ = dict(rep_)
    rep_['vi_tri_m'] = rep_['vi_tri_m'] / cfg['a0_min']
    return rep_


def rot2(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


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
    if not all(math.isfinite(v) for p in poses for v in p[:3]) or not all(math.isfinite(e) for e in eps):
        raise ValueError('pairwise: non-finite pose or eps (NC-4)')
    best_p, best_a, pp, pa = 0.0, 0.0, None, None
    for j, k in itertools.combinations(range(len(poses)), 2):
        dp = math.hypot(poses[j][0] - poses[k][0], poses[j][1] - poses[k][1]) + eps[j] + eps[k]
        da = abs(math.degrees(wrap(poses[j][2] - poses[k][2])))
        if dp >= best_p:
            best_p, pp = dp, [j, k]
        if da >= best_a:
            best_a, pa = da, [j, k]
    return dict(vi_tri_m=best_p, cap_vi_tri=pp, goc_deg=best_a, cap_goc=pa)


def analyse(summaries, cfg, rong=None, tag_h=0.12, boxes_px=None):
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
    if boxes_px is not None:
        for (i, a), (k, b) in itertools.combinations(enumerate(boxes_px), 2):
            if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
                why.append(f'o_{i}_{k}_chong_nhau')                            # NC-6
    if nb >= 2:
        b0 = np.array(summaries[0]['box'])
        sep = min(float(np.hypot(*(b0[i] - b0[k]))) for i in range(nb) for k in range(i + 1, nb))
        if sep < cfg['min_box_sep_m']:
            why.append(f'vung_gan_nhau_{sep * 1000:.0f}mm')                    # NT-1: two boxes on two patches
    motions = []
    for j, s in enumerate(summaries):
        m = body_motion(summaries[0]['box'], s['box']) if nb >= 2 else (0.0, np.zeros(2), NAN)
        motions.append(m)
        if nb >= 2 and not m[2] <= cfg['rigid_res_m']:
            why.append(f'N{j}_du_khop_{m[2] * 1000:.2f}mm')
    cam0 = np.array(summaries[0]['cam'], float)
    c0, H = cam0[:2], float(cam0[2])
    s0, u0 = summaries[0].get('scale', 1.0), summaries[0].get('scale_u', 0.0)
    qbar0 = np.mean(np.array(summaries[0]['box'], float), axis=0) if nb >= 1 else c0
    eps, eps_h, body = [], [], []
    for j, (s, m) in enumerate(zip(summaries, motions)):
        cj = np.array(s['cam'], float)
        dc = float(np.hypot(*(cj[:2] - c0)))                                   # x, y (NT3-3)
        dH = abs(float(cj[2]) - H)
        r_lo, r_hi = scale_interval(s.get('scale', 1.0), s.get('scale_u', 0.0), s0, u0, j == 0)
        arm = max(float(np.hypot(*(np.array(s['tag'][:2]) - cj[:2]))), float(np.hypot(*(np.array(summaries[0]['tag'][:2]) - c0))))
        # NC-2: |1/a_j - 1/a_0| = |1 - 1/r| / a_0, a_0 the true scale of N0 (>= a0_min); arm from each clip's camera
        eh = max(abs(1.0 - 1.0 / r_lo), abs(1.0 - 1.0 / r_hi)) / cfg['a0_min'] * arm
        eps_h.append(eh)                                                       # NT-7, NT3-1, NT3-2, NC-2
        if nb >= 2:
            tj = float(np.hypot(*m[1]))
            rot_c = float(np.hypot(*((rot2(m[0]) - np.eye(2)) @ c0)))         # |(R - I) C0| (NT2-1)
            # NC3-1: the tag (|dz| <= dz_tag) and the boxes (|dz| <= dh) may both be off their planes; the body removal
            # then leaves |a - b| / b x (motion) with |a - b| / b <= (dh + dz_tag) / (H - |dH| - h - dz_tag)
            dz = cfg['dh_m'] + cfg['dz_tag_m']
            den = H - dH - tag_h - cfg['dz_tag_m']   # the lower camera height and the higher plane: an upper bound
            if not den > 0:
                raise ValueError(f'H - |dH| - h - dz_tag = {den} <= 0 (NC-3)')
            e_H = dz * dH / den ** 2 * float(np.hypot(*(qbar0 - c0)))                 # NT3-3
            eps.append((tj + dc + rot_c) * dz / den + e_H + eh)
        else:
            tj, rot_c, e_H = NAN, NAN, NAN
            eps.append(NAN)
        body.append(dict(lan=f'N{j}', dich_m=tj, quay_deg=math.degrees(m[0]) if nb >= 2 else NAN,
                         du_khop_m=m[2], camera_dich_xy_m=dc, camera_dich_z_m=dH, quay_quanh_camera_m=rot_c,
                         eps_H_m=e_H, eps_m=eps[-1], eps_h_m=eh, thang_so_N0=s.get('scale', 1.0) / s0,
                         thang_khoang=[r_lo, r_hi]))
    kiem_than = not why
    res = dict(tho=scale_abs(pairwise(raw, eps_h), cfg), kiem_than=kiem_than, ly_do_khong_kiem_than=why, than=body)
    if nb >= 2:
        corr = [to_body0(raw[j], motions[j]) for j in range(len(raw))]
        res['tru_than'] = dict(scale_abs(pairwise(corr, eps), cfg), tu_the=corr)
        res['than_dich_max_m'] = max(b['dich_m'] for b in body)
        res['than_dich_max_deg'] = max(abs(b['quay_deg']) for b in body)
    else:
        res.update(tru_than=None, than_dich_max_m=None, than_dich_max_deg=None)
    use = res['tru_than'] if kiem_than else res['tho']
    res.update(vi_tri_m=use['vi_tri_m'], goc_deg=use['goc_deg'], so_dung='tru_than' if kiem_than else 'tho')
    return res


GIOI_HAN = ('Do lap QUAN SAT DUOC cua {n} lan dat (mau nho; khong phai can moi lan dat sau, khong nhan he so; le 1,25 do '
            'c8b_u; so dung cho chien dich do CODE_C8B/Liam quyet khi khoa U). Tam ngoi chot cao thap giua cac lan: '
            'eps_h tu thang bieu kien cua ma voi bat dinh thong ke; LECH HE THONG cua do canh giua hai clip (co deu '
            'khoang 0,06 px) co the che tam cao 3 mm (~0,6 mm) va CHUA bao (NT3-1). Tam nghieng, hai o khac do cao chua '
            'bao. Chot mon qua cac buoi chua bao. Do tai mot cho (goc O). Can sai do cao vung than dh {dh} m [uoc '
            'luong]; goc khong cong eps.')
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
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):
        return None                       # LLC2-6: strict JSON (no NaN / Infinity literals); the reason is in ly_do
    return cm._json(o)


# ---------------------------------------------------------------- commands
def _setup(session_path, boxes_spec, n0_clip, h_than, cfg, mau_idx=None):
    """mau_idx: the template frame chosen by khung (khung_N0.json 'idx'): the boxes were picked on that image, so it
    is used as is (NC2-2); None = nearest gated frame with usable boxes."""
    with open(session_path, encoding='utf-8') as f:
        session = json.load(f)
    ctx = Ctx(session, session['tag_h_m'] if h_than is None else h_than, cfg)
    boxes_px = parse_boxes(boxes_spec)
    if mau_idx is not None:
        tidx, _, boxes = template_at(n0_clip, ctx, int(mau_idx), boxes_px)
    else:
        tidx, tframe = pick_template_frame(n0_clip, ctx, frame_count(n0_clip), boxes_px)
        boxes = [check_template(tframe, b, cfg) for b in boxes_px]
    return ctx, boxes_px, tidx, boxes


def _save_r_frame(video, idx, out_dir):
    """khung_R.png: one frame of clip R that passed (NT2-3 evidence: plate off the car, boxes on the body)."""
    g = frames_of(video, [idx]).get(idx)
    if g is not None:
        cv2.imwrite(os.path.join(out_dir, 'khung_R.png'), g)


def parse_rejected(items):
    """['FILE=REASON', ...] -> [{file, ly_do, sha256}] (NT2-4: every clip shot is kept and accounted for)."""
    out = []
    for it in items or []:
        f, sep, why = it.partition('=')
        if not sep or not f or not why:
            raise ValueError(f'--loai {it!r}: need FILE=REASON')
        out.append(dict(file=os.path.basename(f), ly_do=why, sha256=cm.sha256(f)))
    return out


def run_do(clips, session_path, boxes_spec, out_dir, rong=None, h_than=None, stride=1, max_frames=None, cfg=None,
           loai=None, mau_idx=None):
    """Whole analysis; writes the outputs; returns the exit code (0 dat, 1 dat_gioi_han, 2 blocking error)."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    if len(clips) < MIN_PLACEMENTS:
        raise ValueError(f'need at least {MIN_PLACEMENTS} placement clips in order (N0 N1 ...), got {len(clips)}')
    if stride < 1:
        raise ValueError('stride >= 1')
    rejected = parse_rejected(loai)
    shas = [cm.sha256(v) for v in clips] + ([cm.sha256(rong)] if rong is not None else [])
    if len(set(shas)) != len(shas):
        raise ValueError('the same clip is given twice (placements / R must be different recordings; NC-8)')
    if {r['sha256'] for r in rejected} & set(shas):
        raise ValueError('a rejected clip (--loai) is also used (NC-8)')
    cm._new_dir(out_dir)
    ctx, boxes_px, tidx, boxes = _setup(session_path, boxes_spec, clips[0], h_than, cfg, mau_idx)
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
            if s0.get('ok') else dict(dat=False, n_ok=0, n_needed=0, n_considered=0, idx_ok=[])
        rong_out = dict(file=os.path.basename(rong), sha256=cm.sha256(rong), tom_tat=rs)
        if rs['idx_ok']:
            _save_r_frame(rong, rs['idx_ok'][len(rs['idx_ok']) // 2], out_dir)
    out = dict(version=VERSION, session=os.path.abspath(session_path), session_sha256=cm.sha256(session_path),
               hil=git_head(), h_than_m=ctx.h_than, h_than_src='tag_h_m cua session' if h_than is None else '--h-than',
               vung_than_px=[list(b) for b in boxes_px], khung_mau=dict(clip='N0', idx=tidx), stride=stride,
               nguong=dict(cfg, trang_thai=THRESH_STATUS), clips=clip_out, rong=rong_out, clip_bi_loai=rejected,
               ket_qua=None)
    write_frames(os.path.join(out_dir, 'frames_nhac_tam.csv'), named_rows)
    out.update(co=list(flags), ma=2)
    blk = None
    if not all(c['tom_tat']['ok'] for c in clip_out):
        out['ket_luan'] = 'LỖI CHẶN: có clip hỏng (không ghi khối U)'
    else:
        try:                                     # NC-3, NC-4: out of the numeric domain -> blocking, no U block
            res = analyse([c['tom_tat'] for c in clip_out], cfg, None if rong_out is None else rong_out['tom_tat'],
                          ctx.tag_h, boxes_px)
            out['ket_qua'] = res
            out['co'] += [] if res['kiem_than'] else ['khong_kiem_than']
            blk = block(res, clip_out, rong_out, out['session'], out['session_sha256'], cfg)
            if not all(math.isfinite(float(blk[k])) and float(blk[k]) >= 0 for k in ('vi_tri_m', 'goc_deg')):
                raise ValueError(f'U block not finite: {blk["vi_tri_m"]}, {blk["goc_deg"]}')
        except ValueError as e:
            blk = None
            out['co'].append('loi_mien_so')
            out['ket_luan'] = f'LỖI CHẶN: {e} (không ghi khối U)'
        if blk is not None and stride > cfg['max_stride']:          # NT2-2: quick look only, no U block
            blk = None
            out['co'].append('stride_xem_nhanh')
            out['ket_luan'] = f'XEM NHANH (stride {stride} > {cfg["max_stride"]}): không ghi khối U'
        elif blk is not None:
            out['ma'] = 0 if out['ket_qua']['kiem_than'] else 1
            out['ket_luan'] = 'đạt' if out['ma'] == 0 else 'đạt có giới hạn (không kiểm được thân)'
    if blk is not None:
        with open(os.path.join(out_dir, 'c8_phan_nhac_tam.json'), 'w', encoding='utf-8') as f:
            json.dump(_jsonable(blk), f, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(out_dir, 'nhac_tam.json'), 'w', encoding='utf-8') as f:
        json.dump(_jsonable(out), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'bao_cao.txt'), 'w', encoding='utf-8') as f:
        f.write(report(out))
    return out['ma']


def run_kiem(clip, session_path, boxes_spec, n0_clip, out_dir, rong=False, h_than=None, stride=3, cfg=None,
             mau_idx=None):
    """Check ONE clip at the floor, before the plate is lifted to the next placement (NT2-4). Placement clip: enough
    gated frames, still. Clip R: the R conditions against N0 (n0_clip must already be checked). Returns (ok, text);
    writes kiem.json in out_dir. No U block."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    if not 1 <= stride <= cfg['max_stride']:
        raise ValueError(f'kiem: stride must be 1..{cfg["max_stride"]} (NC-7)')
    cm._new_dir(out_dir)
    ctx, boxes_px, tidx, boxes = _setup(session_path, boxes_spec, n0_clip, h_than, cfg, mau_idx)
    rows = measure_clip(clip, ctx, boxes, stride)
    info = dict(version=VERSION, kind='kiem', clip=os.path.basename(clip), sha256=cm.sha256(clip),
                n0=os.path.basename(n0_clip), stride=stride, rong=bool(rong), vung_than_px=[list(b) for b in boxes_px],
                n0_sha256=cm.sha256(n0_clip), mau_idx=int(tidx))       # LLC1-9 (added keys; v7 behaviour unchanged)
    if rong:
        s0 = clip_summary(measure_clip(n0_clip, ctx, boxes, stride), len(boxes), cfg, stride)
        if not s0.get('ok'):
            ok, why = False, 'N0 chua dat'
            info['tom_tat'] = None
        else:
            rs = rong_summary(rows, s0['box'], s0['tag'], len(boxes), cfg, stride)
            info['tom_tat'] = rs
            ok = rs['dat']
            why = '' if ok else f'R: {rs["n_ok"]}/{rs["n_needed"]} khung dat (ma phai thay o xa >= 0,2 m, vung than phai thay)'
            if rs['idx_ok']:
                _save_r_frame(clip, rs['idx_ok'][len(rs['idx_ok']) // 2], out_dir)
    else:
        s = clip_summary(rows, len(boxes), cfg, stride)
        info['tom_tat'] = s
        ok = bool(s.get('ok'))
        why = '' if ok else ', '.join(s['flags']) + f' ({s.get("n_used", 0)}/{s.get("n_needed", "?")} khung)'
    info.update(duoc=ok, ly_do=why)
    with open(os.path.join(out_dir, 'kiem.json'), 'w', encoding='utf-8') as f:
        json.dump(_jsonable(info), f, ensure_ascii=False, indent=1)
    txt = (f'{os.path.basename(clip)}: ĐƯỢC' if ok else
           f'{os.path.basename(clip)}: QUAY LẠI, KHÔNG nhấc tấm ({why})')
    return ok, txt


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
    """Crop around the car on the SAME template frame that do / kiem use (NC-5)."""
    cm._new_dir(out_dir)
    with open(session_path, encoding='utf-8') as f:
        ses = json.load(f)
    ctx = Ctx(ses, ses['tag_h_m'], dict(DEFAULTS))
    mid, g = pick_template_frame(clip, ctx, frame_count(clip))
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
    m = DEFAULTS['search_px']
    boxes = [(u + u0, v + v0, w, h) for u, v, w, h in propose_boxes(crop, ctr - [u0, v0], side)
             if u + u0 - m >= 0 and v + v0 - m >= 0 and u + u0 + w + m <= g.shape[1] and v + v0 + h + m <= g.shape[0]]
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
                cach_dung='dua idx nay vao kiem / do bang --mau-idx (cung khung mau, NC2-2)',
                tag_centre_px=ctr.tolist(), tag_side_px=side, crop_px=[u0, v0, u1, v1], anh_phong=2,
                o_de_xuat=[list(b) for b in boxes],
                ghi_chu='O de xuat chi la goi y (ket cau manh); chon o nam tron trong mieng bang dinh tren than')
    with open(os.path.join(out_dir, 'khung_N0.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    return info


# ======================================================================== v4: placement of the plate per run
# Design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_nhac_tam_lan_lap_v4.md (DA DUYET, Liam 19/11 "Nhan, ghi gioi
# han"; section 6b delta_c_B, Liam "Them dc_B, Codex hep", fixed after Codex 4b LL5-1..6). Sections cited as "v4 x".
#   hieu_chuan: anchors on the body (tape A, B, bolts) fitted on the FIT windows only (12b, E), locked with time and
#               sha256 before any held-out window is read (v4 1, D1) -> hieu_chuan_neo.json (F6)
#   lan_lap:    pre / post windows of every run -> lan_lap.json (lan_lap_v1, F8): beta, u_beta, delta_c_B (dat when
#               the gates hold), diagnostics (delta_a, q, delta_h_goc: always chua_ket_luan), tam_doi_trong_luot
# Every number is an empirical index BEFORE the 1.25 margin (C8B multiplies once).
import datetime                                                          # noqa: E402
import hashlib                                                           # noqa: E402
import importlib                                                         # noqa: E402

from scipy.optimize import least_squares, linprog, minimize_scalar        # noqa: E402

LL_VERSION = 'c7_nhac_tam_lan_lap_v1'
# Names of formats and keys owned by other sessions (hop dong du lieu C8 tuong doi v1): the ONE place to change them.
TEN = dict(
    cua_so='cua_so_v1', a6='c8b_a6_v2', a6_module='c8b_a6', xoay='xoay_v1', truc_xe='truc_xe_v1',
    chuoi_khung='c7_chuoi_khung_v1', doan_12b='doan_12b_v1', hieu_chuan='hieu_chuan_neo_v1', lan_lap='lan_lap_v1',
    cs_list=('cua_so', 'dong', 'windows'), schema='schema', luot='luot_id', cho='cho', loai='loai', clip='clip',
    clip_sha='clip_sha256', session_sha='session_sha256', khung='khung', k_id='id', k_pts='pts_s',
    khung_ma='khung_ma_hop_le', t_canh='t_canh_s', frames_sha='frames_sha256',
    xoay_r='r_floor_m', trang_thai='trang_thai',
    tx_phi='phi_hat_chuan_k_deg', tx_u_phi='u_phi_hat_chuan_k_deg', tx_a='a_k_deg',
    tx_mien='mien_hieu_luc', tx_thu='thu_id_cap', tx_nguon='nguon', tx_dat=('khoa',),
    f15='c7_meo_dau_chan_bang_v1', k_sha='khung_sha256', ma_sha='khung_ma_hop_le_sha256',
    tx_f6='hieu_chuan_neo', tx_a6='c8b_a6_v2',
    a6_g='g_b', a6_eps_g='eps_g',
)
CHO = ('goc', 'K1', 'K2')
LOAI_KHOP = ('ref_12b', 'dung_12b', 'E_dau', 'E_cuoi')        # v4 1: fit set
LOAI_GIU = ('pre', 'post')                                     #        held-out set
LOAI_CHUAN, LOAI_DICH = 'ref_12b', 'pre'                       # standard window of a place, target window of a run
LOAI_12B = ('ref_12b', 'dung_12b')
LL_DEFAULTS = dict(
    min_marks=8, rms_extra_px=0.5, corr_min=0.9, peak2_ratio=0.9, search_px=60,
    still_neo_m=0.0005,      # v4 3: 95th percentile of the anchor deviation in a window
    min_khung=120,           # v4 3: >= 120 frames (3 s) ...
    min_frac=0.6,            #       ... and >= 60 % of the window
    mo_ho_deg=5.0,           # v4 5: a frame off the circular mean by more than this: ambiguous window
    hang_rel=1e-3,           # v4 3: weak direction, relative singular value
    suy_bien_rel=1e-9,       #       numerically null (no finite standard error): the rank is missing
    hang_k=3.0,              #       replay at +-3 standard errors / normalised units
    replay_deg=0.01,         #       change of the difference above this: thieu_hang
    s_min_m=1e-7,            #       floor of the fit standard deviation (noise-free synthetic data)
    eps_tong_deg=0.03,       # v4 3: r_max locked so that 2 eps_neo <= this
    n_huong=72,              #       directions of the single-anchor displacement test
    min_cap=4,               # v4 4: held-out pairs
    u_beta_max_deg=0.05,     # v4 4
    tich_hop_deg=1e-6,       # v4 0: integration check
    dh_max_m=0.005,          # v4 6b
    min_12b=3,               # v4 6b: 12b windows of the place for the indices
    min_giu_vi_tri=1,        # v4 6b LL5-5: same-place held-out pairs in the session
    doi_theta_deg=0.05,      # v4 8: floors of the 'changed' thresholds (the 12b spread of the place is used when larger)
    doi_c_m=0.0002,
    doi_s=0.0005,
)
LL_THRESH_STATUS = 'DE XUAT (thiet ke lan lap v4 muc 3, 4, 6b, 8)'


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def gop_goc(a):
    """trung_binh_go_vong (rad): circular mean, then the mean of the deviations unwrapped around it."""
    a = np.asarray(a, float)
    m = math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
    return wrap(m + float(np.mean([wrap(x - m) for x in a])))


def mo_ho(a, lim_deg):
    """v4 5 (LL3-8): a frame further than lim_deg from the circular mean -> ambiguous branch."""
    a = np.asarray(a, float)
    if not len(a):
        return False
    m = math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
    return bool(max(abs(math.degrees(wrap(x - m))) for x in a) > lim_deg)


def sha_ids(ids):
    """sha256 of a frame-ID list: sorted ints joined by commas, no brackets ("1,2"; the one form of every session,
    CODE 19/11, as ids_sha of HIEUCHUAN, CAM)."""
    return hashlib.sha256(','.join(str(i) for i in sorted(int(i) for i in ids)).encode()).hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(_jsonable(o), sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- v4 2: one projection contract for every point
def lift(p, kappa, C):
    """car_pose lift of a floor point p (pix2floor at z = 0): P = p + kappa (C - p)."""
    return np.asarray(p, float) + kappa * (np.asarray(C, float) - np.asarray(p, float))


def unlift(P, kappa, C):
    return (np.asarray(P, float) - kappa * np.asarray(C, float)) / (1.0 - kappa)


def thuan(G, h, alpha, o, F, H, C, a):
    """v4 2 (LL4-4), exact: observed P^ of body points G (n, 2) at heights h (n) for the body pose (alpha, o), frame
    camera centre (F, H), lift centre C and a = 1 - h_lift / Z_c: P^ - C = a lam (P - C) + a (1 - lam) (F - C)."""
    G, h, F, C = np.asarray(G, float), np.asarray(h, float), np.asarray(F, float), np.asarray(C, float)
    P = G @ rot2(alpha).T + np.asarray(o, float)
    lam = H / (H - h)
    return C + a * (lam[:, None] * (P - C) + (1.0 - lam)[:, None] * (F - C))


def nghich(Ph, h, F, H, C, a):
    """Inverse of thuan for points of known heights h: the true floor positions."""
    Ph, F, C = np.atleast_2d(np.asarray(Ph, float)), np.asarray(F, float), np.asarray(C, float)
    lam = H / (H - np.broadcast_to(np.asarray(h, float), (len(Ph),)))
    return C + ((Ph - C) / a - (1.0 - lam)[:, None] * (F - C)) / lam[:, None]


def tu_the(G, h, Ph, F, H, C, a):
    """Body pose (alpha, o) of one window / frame with G, h locked: rigid fit of G to the true points; residual per
    anchor in the observed plane (m)."""
    P = nghich(Ph, h, F, H, C, a)
    ang, t, _, _ = cm.rigid_fit(np.asarray(G, float), P)
    res = np.linalg.norm(thuan(G, h, ang, t, F, H, C, a) - Ph, axis=1)
    return float(ang), np.asarray(t, float), res


# ---------------------------------------------------------------- v4 3: anchor shape fit, rank, replay
def _pack_info(n, fix_h):
    nG = 1 + 2 * (n - 2)                     # L of B on the x axis, the others free; A at the origin (LL3-2)
    nh = 0 if fix_h else n - 1               # h_A = h_n
    return nG, nh


def _unpack(x, n, W, h_n, fix_h):
    nG, nh = _pack_info(n, fix_h is not None)
    G = np.zeros((n, 2))
    G[1, 0] = x[0]
    G[2:] = x[1:nG].reshape(-1, 2)
    h = np.full(n, h_n) if fix_h is None else np.array(fix_h, float)
    if fix_h is None:
        h[1:] = x[nG:nG + nh]
    poses = x[nG + nh:].reshape(W, 3)
    return G, h, poses


def khop_hinh(obs, h_n, C, a, fix_h=None, x0=None):
    """v4 3: least squares of G, h (h_A = h_n), (alpha_w, o_w) on the windows obs = [dict(Ph (n,2), F, H)].
    fix_h: heights held fixed (replay). Returns dict(G, h, poses, res (W, n), jac, x, idx_h, s)."""
    n, W = len(obs[0]['Ph']), len(obs)
    nG, nh = _pack_info(n, fix_h is not None)
    if x0 is None:
        P0 = nghich(obs[0]['Ph'], h_n, obs[0]['F'], obs[0]['H'], C, a)
        a0 = math.atan2(*(P0[1] - P0[0])[::-1])
        G0 = (P0 - P0[0]) @ rot2(a0)                                        # R(-a0) (P - P_A)
        x0 = [G0[1, 0]] + list(G0[2:].ravel()) + ([h_n] * nh)
        for w in obs:
            ang, t, _ = tu_the(G0, np.full(n, h_n), w['Ph'], w['F'], w['H'], C, a)
            x0 += [ang, t[0], t[1]]
        x0 = np.array(x0, float)

    def resid(x):
        G, h, poses = _unpack(x, n, W, h_n, fix_h)
        return np.concatenate([(thuan(G, h, p[0], p[1:], w['F'], w['H'], C, a) - w['Ph']).ravel()
                               for p, w in zip(poses, obs)])
    sol = least_squares(resid, x0, x_scale='jac', ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=300)
    G, h, poses = _unpack(sol.x, n, W, h_n, fix_h)
    r = sol.fun.reshape(W, n, 2)
    return dict(G=G, h=h, poses=poses, res=np.linalg.norm(r, axis=2), jac=sol.jac, x=sol.x, hoi_tu=bool(sol.status > 0),
                idx_h=list(range(nG, nG + nh)), m=len(sol.fun), p=len(sol.x), ssr=float(np.sum(sol.fun ** 2)))


def hang(fit, cfg):
    """v4 3 (LL3-1, LL4-5): Jacobian in h AFTER removing G and the poses, columns scaled by the fit standard errors
    of h; weak directions = relative singular values < hang_rel. Returns dict(sigma_h, s_rel, yeu (list of dh vectors
    of one normalised unit), suy_bien)."""
    J, ih = fit['jac'], fit['idx_h']
    if not ih:
        return dict(sigma_h=[], s_rel=[], yeu=[], suy_bien=False)
    dof = fit['m'] - fit['p']
    s = max(math.sqrt(fit['ssr'] / dof) if dof > 0 else NAN, cfg['s_min_m'])
    if not math.isfinite(s):
        s = cfg['s_min_m']
    ir = [i for i in range(J.shape[1]) if i not in ih]
    Jh, Jr = J[:, ih], J[:, ir]
    Jp = Jh - Jr @ np.linalg.lstsq(Jr, Jh, rcond=None)[0]           # h columns after removing G and the poses
    _, S0, V0 = np.linalg.svd(Jp, full_matrices=False)
    ref = float(np.linalg.svd(Jh, compute_uv=False).max())          # size of the h columns before the removal
    suy = bool(ref <= 0 or not fit.get('hoi_tu', True) or S0.min() < cfg['suy_bien_rel'] * ref)
    if suy:                                                         # LLC1-10: no finite standard error
        n = len(ih)
        return dict(sigma_h=np.full(n, math.inf), cov_h=np.full((n, n), math.inf), s_rel=np.zeros(n), yeu=[],
                    suy_bien=True, s_fit_m=s)
    cov = s * s * (V0.T @ np.diag(1.0 / S0 ** 2) @ V0)              # marginal covariance of h (Schur complement)
    sig = np.sqrt(np.diag(cov))
    _, S, Vt = np.linalg.svd(Jp * sig[None, :], full_matrices=False)
    rel = S / S.max()
    yeu = [sig * Vt[k] for k in range(len(S)) if rel[k] < cfg['hang_rel']]
    return dict(sigma_h=sig, cov_h=cov, s_rel=rel, yeu=yeu, suy_bien=False, s_fit_m=s)


def eps_neo_bang(G, h, F, H, C, a, cfg=None):
    """v4 3 (LL3-3), 6b (LL5-4): single-anchor displacement with G, h locked, linearised (LLC1-11): for every anchor i,
    dalpha = a_i . d, do = B_i d, residual of anchor j = R_ij d (d the displacement of the observed point)."""
    G, h = np.asarray(G, float), np.asarray(h, float)
    base = thuan(G, h, 0.0, (0.0, 0.0), F, H, C, a)
    e, out = 1e-6, []
    for i in range(len(G)):
        cols = []
        for u in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            Ph = base.copy()
            Ph[i] += e * u
            ang, t, _ = tu_the(G, h, Ph, F, H, C, a)
            r = thuan(G, h, ang, t, F, H, C, a) - Ph
            cols.append((wrap(ang) / e, t / e, r / e))
        out.append(dict(a=np.array([cols[0][0], cols[1][0]]), B=np.column_stack([cols[0][1], cols[1][1]]),
                        R=np.stack([cols[0][2], cols[1][2]], axis=-1)))          # R: (n, 2, 2)
    return out


def _max_ti_so(b, num):
    """Largest num(u) / (largest anchor residual R_j u) over the unit direction u (LLC1-11): dense grid (vectorised),
    then a bounded refinement around the best grid points. num maps (2, N) directions to (N,)."""
    th = np.linspace(0.0, 2.0 * math.pi, 3601)[:-1]

    def ratio(t):
        U = np.vstack([np.cos(t), np.sin(t)])
        den = np.max(np.linalg.norm(np.einsum('jab,bn->jan', b['R'], U), axis=1), axis=0)
        return np.where(den > 0, num(U) / np.where(den > 0, den, 1.0), 0.0)
    v = ratio(th)
    best = float(v.max())
    for k in np.argsort(v)[-6:]:
        r = minimize_scalar(lambda x: -float(ratio(np.array([x]))[0]), bounds=(th[k] - 0.002, th[k] + 0.002),
                            method='bounded', options=dict(xatol=1e-10))
        best = max(best, -float(r.fun))
    return best


def k_goc(bang):
    """Largest |dalpha| / largest anchor residual over anchors and directions."""
    return max(_max_ti_so(b, lambda U, b=b: np.abs(b['a'] @ U)) for b in bang)


def r_max_tu(bang, cfg):
    """r_max with 2 eps_neo = eps_tong_deg (v4 3) and eps_neo (deg, one end). bang: one table or a list of tables (one
    per window geometry: the largest ratio is used)."""
    tabs = bang if bang and isinstance(bang[0], list) else [bang]
    k = max(k_goc(b) for b in tabs)
    r = math.radians(cfg['eps_tong_deg'] / 2.0) / k
    return r, math.degrees(r * k)


def eps_goc(bang, r_max):
    """eps_neo (deg) of one end with its own geometry and the locked r_max."""
    return math.degrees(r_max * k_goc(bang))


def eps_vi_tri(bang, r_max, c_B):
    """LL5-4: largest error of R(-alpha)(t - o) for a tag point at c_B (body frame) from one anchor displaced with
    residual <= r_max (origin and angle through the lever arm), one end, continuous over the direction (LLC1-11)."""
    Jc = np.array([[0.0, 1.0], [-1.0, 0.0]]) @ np.asarray(c_B, float)    # d/dalpha of R(-alpha) x at alpha = 0

    def num(U, b):
        return np.linalg.norm(-(b['B'] @ U) + (b['a'] @ U)[None, :] * Jc[:, None], axis=0)
    return r_max * max(_max_ti_so(b, lambda U, b=b: num(U, b)) for b in bang)


def trong_mien(x, pts, tol=1e-9):
    """v4 4: x inside the convex hull of pts (LP: lambda >= 0, sum 1, sum lambda p = x, slack <= tol)."""
    P = np.asarray(pts, float)
    m, d = P.shape
    if m == 0:
        return False
    # variables: lambda (m), slack+ (d), slack- (d); minimise the slack
    c = np.concatenate([np.zeros(m), np.ones(2 * d)])
    A = np.zeros((d + 1, m + 2 * d))
    A[:d, :m] = P.T
    A[:d, m:m + d] = np.eye(d)
    A[:d, m + d:] = -np.eye(d)
    A[d, :m] = 1.0
    b = np.concatenate([np.asarray(x, float), [1.0]])
    r = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * (m + 2 * d), method='highs')
    return bool(r.status == 0 and r.fun <= tol)


# ---------------------------------------------------------------- inputs owned by other sessions (adapters)
class A6:
    """Adapter of c8b_a6_v2 (C8B, hop dong F3): load_a6, delta_psi_a6, a6du_rad, ap_dung. mod = injected module
    (tests); else tools/c8b_a6.py. NHACTAM never computes A6 itself."""

    def __init__(self, path, mod=None):
        self.mod = mod if mod is not None else importlib.import_module(TEN['a6_module'])
        self.path, self.sha256 = path, cm.sha256(path)
        self.obj = self.mod.load_a6(path)

    def _get(self, k):
        return self.obj[k] if isinstance(self.obj, dict) else getattr(self.obj, k)

    def dpsi(self, xy, psi, plumb, zc):
        d, ok, _ = self.mod.delta_psi_a6(self.obj, np.asarray(xy, float).reshape(-1, 2), np.asarray(psi, float).ravel(),
                                         plumb, zc)
        return np.asarray(d, float), np.asarray(ok, bool)

    def g_b(self):
        return np.asarray(self._get(TEN['a6_g']), float)

    def eps_g(self):
        return float(self._get(TEN['a6_eps_g']))

    def ap_dung(self, tam_id):
        return self.mod.ap_dung(self.obj, tam_id)


def doc_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def doc_cua_so(path):
    """cua_so_v1 (F2): JSON {schema, <list key>: [...]}, a JSON list, or JSON lines. Returns the window dicts."""
    with open(path, encoding='utf-8') as f:
        txt = f.read()
    try:
        o = json.loads(txt)
    except json.JSONDecodeError:
        o = [json.loads(s) for s in txt.splitlines() if s.strip()]
    if isinstance(o, dict):
        if o.get(TEN['schema']) not in (None, TEN['cua_so']):
            raise ValueError(f'{path}: schema {o.get(TEN["schema"])} is not {TEN["cua_so"]}')
        rows = next((o[k] for k in TEN['cs_list'] if k in o), None)
        if rows is None:
            raise ValueError(f'{path}: no window list ({TEN["cs_list"]})')
    else:
        rows = o
    for w in rows:
        if w.get(TEN['schema'], TEN['cua_so']) != TEN['cua_so']:
            raise ValueError(f'{path}: window with schema {w.get(TEN["schema"])}')
        if w[TEN['cho']] not in CHO or w[TEN['loai']] not in LOAI_KHOP + LOAI_GIU:
            raise ValueError(f'{path}: window cho={w[TEN["cho"]]} loai={w[TEN["loai"]]}')
    return rows


def khung_cua(w):
    """[(id, pts_s)] of a window; an empty or null list (CAM: clip shorter than the post window, CT4-3) -> []."""
    return [(int(k[TEN['k_id']]), float(k[TEN['k_pts']])) for k in (w.get(TEN['khung']) or [])]


def khung_ma(w):
    """IDs of the frames whose tag passed the CAM gate (khung_ma_hop_le: IDs or {id, ...})."""
    return {int(k[TEN['k_id']]) if isinstance(k, dict) else int(k) for k in w.get(TEN['khung_ma'], [])}


def kiem_canh(w, pts):
    """v4 1: the window against its command edge by type: pre / 12b / E end at or before the edge, post starts at or
    after it; no edge (last window) -> not checked."""
    tc = w.get(TEN['t_canh'])
    if tc is None or not len(pts):
        return True
    return bool(min(pts) >= tc) if w[TEN['loai']] == 'post' else bool(max(pts) <= tc)


def doc_theo_cho(path, schema):
    """xoay_v1 / truc_xe_v1: {cho: {...}} (directly, or under a key 'cho') -> {cho: dict}."""
    o = doc_json(path)
    if o.get(TEN['schema'], schema) != schema:
        raise ValueError(f'{path}: schema {o.get(TEN["schema"])} is not {schema}')
    m = o.get('cho', o)
    return {k: v for k, v in m.items() if k in CHO and isinstance(v, dict)}, cm.sha256(path)


def doc_truc_xe(paths):
    """truc_xe_v1 of HIEUCHUAN (one file per place and issued 12b attempt: 'cho', mien_hieu_luc.thu_id_cap, nguon) ->
    {(cho, thu_id_cap): block}, [{file, sha256}]. LLC2-1: phi is read for the 12b run that gave the standard, not
    per place."""
    out, src = {}, []
    for p in paths or []:
        o = doc_json(p)
        if o.get(TEN['schema']) != TEN['truc_xe']:
            raise ValueError(f'{p}: schema {o.get(TEN["schema"])} is not {TEN["truc_xe"]}')
        mh = o.get(TEN['tx_mien']) or {}
        key = (o.get(TEN['cho']) or mh.get(TEN['cho']), mh.get(TEN['tx_thu']))
        if key[0] not in CHO or not key[1]:
            raise ValueError(f'{p}: needs cho and {TEN["tx_mien"]}.{TEN["tx_thu"]}')
        if key in out:
            raise ValueError(f'{p}: a second truc_xe_v1 for {key}')
        out[key] = o
        src.append(dict(file=os.path.abspath(p), sha256=cm.sha256(p)))
    return out, src


def tx_nguon_sha(block, vai_tro):
    """sha256 of the source with this vai_tro in the nguon list of a truc_xe_v1 block ('<thu_id>:frames' CT1-2,
    'hieu_chuan_neo', 'c8b_a6_v2'), or None."""
    for n in block.get(TEN['tx_nguon']) or []:
        if isinstance(n, dict) and n.get('vai_tro') == vai_tro:
            return n.get('sha256')
    return None


# ---------------------------------------------------------------- pixel layer: frames -> records
def parse_neo(s):
    """'A:u,v,w,h;A:u,v,w,h;B:...;oc1:...' -> ordered {name: [box, ...]}; A, B first (gauge LL3-2). Several boxes of
    one name are one physical point (their mean)."""
    out = {}
    for part in [p for p in s.replace(' ', '').split(';') if p]:
        name, sep, b = part.partition(':')
        if not sep or not name:
            raise ValueError(f'--neo {part!r}: need NAME:u,v,w,h')
        out.setdefault(name, []).extend(parse_boxes(b))
    if 'A' not in out or 'B' not in out:
        raise ValueError('--neo needs anchors A and B (tape patches; gauge of v4 3)')
    for (n1, b1), (n2, b2) in itertools.combinations(out.items(), 2):            # LLC1-9: one physical point, one name
        for a, b in itertools.product(b1, b2):
            if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
                raise ValueError(f'--neo: box {a} of {n1} overlaps box {b} of {n2}')
    for n, bs in out.items():
        if len(set(bs)) != len(bs):
            raise ValueError(f'--neo: anchor {n} lists one box twice')
    return {k: out[k] for k in ['A', 'B'] + [k for k in out if k not in ('A', 'B')]}


def kiem_r_neo(neo, kiem_files, mau_clip, mau_idx):
    """v4 3: every anchor box must be in the vung_than_px of a v7 'kiem --rong' result that passed (clip R proves the
    box is on the body) AND that was checked against the template clip itself: same content (n0_sha256) and the same
    template frame (mau_idx), not only the same file name (LLC1-9). Returns ({name: ok}, evidence)."""
    passed, ev = [], []
    mau_sha = cm.sha256(mau_clip)
    for f in kiem_files or []:
        k = doc_json(f)
        dung_mau = k.get('n0_sha256') == mau_sha and k.get('mau_idx') is not None and int(k['mau_idx']) == int(mau_idx)
        ev.append(dict(file=os.path.basename(f), sha256=cm.sha256(f), duoc=bool(k.get('duoc')), rong=bool(k.get('rong')),
                       n0=k.get('n0'), n0_sha256=k.get('n0_sha256'), mau_idx=k.get('mau_idx'), dung_mau=dung_mau))
        if k.get('rong') and k.get('duoc') and dung_mau:
            passed += [tuple(int(x) for x in b) for b in k.get('vung_than_px', [])]
    return {n: all(tuple(b) in passed for b in bs) for n, bs in neo.items()}, ev


class NeoMau:
    """Anchor templates on one frame (v4 3): per box, the lifted floor point in the TAG frame of that frame (to predict
    where the box is in another window from the tag pose) and a patch wide enough to be turned."""

    def __init__(self, gray, cam, tag, neo, kappa_n, C, cfg):
        self.neo, self.kappa_n, self.C, self.cfg = neo, kappa_n, np.asarray(C, float), cfg
        self.t0, self.psi0 = np.array([tag['tag_x'], tag['tag_y']]), tag['tag_psi']
        self.img_ang0 = self._img_ang(cam, self.t0, self.psi0)
        self.items = []
        for name, boxes in neo.items():
            for (u, v, w, hh) in boxes:
                uc, vc = u + (w - 1) / 2.0, v + (hh - 1) / 2.0
                Ph = lift(cam.pix2floor(np.array([[uc, vc]]), 0.0)[0], kappa_n, self.C)
                d = rot2(-self.psi0) @ (Ph - self.t0)
                r = int(math.ceil(math.hypot(w, hh) / 2.0)) + 3
                u_lo, v_lo = int(math.floor(uc)) - r, int(math.floor(vc)) - r
                if u_lo < 0 or v_lo < 0 or u_lo + 2 * r + 2 > gray.shape[1] or v_lo + 2 * r + 2 > gray.shape[0]:
                    raise ValueError(f'anchor box {(u, v, w, hh)}: too close to the template frame edge')
                patch = gray[v_lo:v_lo + 2 * r + 2, u_lo:u_lo + 2 * r + 2].astype(np.float32)
                self.items.append(dict(name=name, box=(u, v, w, hh), d=d, patch=patch, pc=(uc - u_lo, vc - v_lo),
                                       wh=(w, hh)))
        self._cache = {}

    def _img_ang(self, cam, t, psi):
        a = unlift(np.vstack([t, t + rot2(psi) @ [0.05, 0.0]]), self.kappa_n, self.C)
        uv = cam.project(np.column_stack([a, np.zeros(2)]))
        return math.atan2(uv[1, 1] - uv[0, 1], uv[1, 0] - uv[0, 0])

    def tpl(self, j, rot):
        """Template of box j turned by rot (rad, image (u, v) frame), rounded to 0.05 deg (cache)."""
        key = (j, round(math.degrees(rot) / 0.05))
        if key not in self._cache:
            it = self.items[j]
            pu, pv = it['pc']
            dr = math.radians(key[1] * 0.05)
            c, s = math.cos(dr), math.sin(dr)
            M = np.float32([[c, -s, pu - c * pu + s * pv], [s, c, pv - s * pu - c * pv]])
            rp = cv2.warpAffine(it['patch'], M, it['patch'].shape[::-1], flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
            w, hh = it['wh']
            u0, v0 = pu - (w - 1) / 2.0, pv - (hh - 1) / 2.0
            Mt = np.float32([[1, 0, u0], [0, 1, v0]])
            t = cv2.warpAffine(rp, Mt, (w, hh), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
            if float(t.std()) < 2.0:
                raise ValueError(f'anchor box {it["box"]}: no texture')
            self._cache[key] = t
        return self._cache[key]

    def du_doan(self, cam, t, psi):
        """Predicted pixel centre of every box and the image rotation, from a tag pose (t, psi) and a frame camera."""
        rot = self._img_ang(cam, np.asarray(t, float), psi) - self.img_ang0
        uv = []
        for it in self.items:
            Ph = np.asarray(t, float) + rot2(psi) @ it['d']
            uv.append(cam.project(np.array([[*unlift(Ph, self.kappa_n, self.C), 0.0]]))[0])
        return np.array(uv), rot


def khop_mau(gray, tpl, uc, vc, search):
    """Template matched around the predicted centre (uc, vc): (u, v, corr, second-peak ratio); NaN when the peak lies
    on the window border or ECC moves more than 1 px (as Box.match, NC-1)."""
    hh, w = tpl.shape
    u0, v0 = int(round(uc - (w - 1) / 2.0)) - search, int(round(vc - (hh - 1) / 2.0)) - search
    if u0 < 0 or v0 < 0 or u0 + w + 2 * search > gray.shape[1] or v0 + hh + 2 * search > gray.shape[0]:
        return NAN, NAN, NAN, 1.0
    win = gray[v0:v0 + hh + 2 * search, u0:u0 + w + 2 * search].astype(np.float32)
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    if loc[0] in (0, r.shape[1] - 1) or loc[1] in (0, r.shape[0] - 1):
        return NAN, NAN, NAN, 1.0
    r2 = r.copy()
    r2[max(0, loc[1] - 3):loc[1] + 4, max(0, loc[0] - 3):loc[0] + 4] = -1.0
    ratio = float(r2.max() / mx) if mx > 0 else 1.0
    x, y = subpix(r, *loc)
    warp = np.array([[1.0, 0.0, x], [0.0, 1.0, y]], np.float32)
    try:
        _, warp = cv2.findTransformECC(tpl, win, warp, cv2.MOTION_TRANSLATION,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 1)
    except cv2.error:
        return NAN, NAN, NAN, 1.0
    ex, ey = float(warp[0, 2]), float(warp[1, 2])
    if not (math.isfinite(ex) and math.isfinite(ey)) or math.hypot(ex - x, ey - y) > 1.0:
        return NAN, NAN, NAN, 1.0
    return u0 + ex + (w - 1) / 2.0, v0 + ey + (hh - 1) / 2.0, float(mx), ratio


class LLCtx:
    """Session values of the v4 commands (v4 2: car_pose contract, plumb C, Z_c of the session)."""

    def __init__(self, session, h_n, cfg):
        self.cam0 = cm.cam_from_dict(session['camera'])
        self.marks = {int(i): tuple(v) for i, v in session['marks'].items()}
        self.fit_ids = [int(i) for i in session['fit_ids']]
        self.side, self.tag_h = float(session['car_side_m']), float(session['tag_h_m'])
        self.kappa_t = float(session['kappa'])
        self.Zc = self.tag_h / self.kappa_t
        self.C = np.asarray(session['plumb_xy_m'], float)
        self.rms_h = float(session['rms_px_H'])
        self.h_n = float(h_n)
        self.kappa_n = self.h_n / self.Zc
        self.a_n, self.a_t = 1.0 - self.kappa_n, 1.0 - self.kappa_t
        self.cfg = cfg
        self.det = cm.make_detector()


def do_khung(ctx, cam_prev, g):
    """Pass 1 of a frame: refitted camera (session fit_ids), mark gate, tag by car_pose (v4 2)."""
    d = cm.detect_refined(ctx.det, g)
    obj, img, used = cm.floor_obs(d, ctx.marks, ids=ctx.fit_ids)
    r = dict(cam=None, cam_ok=False, tag=None)
    if len(used) >= 4:
        cam, rms = cm.refit_pose(cam_prev, obj, img)
        ok = bool(len(used) >= ctx.cfg['min_marks'] and math.isfinite(rms) and rms <= ctx.rms_h + ctx.cfg['rms_extra_px']
                  and np.all(np.isfinite(cam.centre)))
        r.update(cam=cam, cam_ok=ok)
        if cm.CAR_ID in d:
            r['tag'] = cm.car_pose(cam, d[cm.CAR_ID], ctx.side, ctx.kappa_t, ctx.C)
    return r


def do_clip(clip, wins, ctx, mau):
    """Records of every window frame of one clip (two decoding passes: cameras and tags, then the anchors with the
    prediction of the window). wins: [(key, [(id, pts)])]. Returns {key: [record]}, flags {key: [why]}."""
    want = {}
    for key, fr in wins:
        for i, p in fr:
            want.setdefault(i, []).append((key, p))
    p1, cam = {}, ctx.cam0
    for k, t, g in cm.iter_frames(clip):
        if k in want:
            for key, p in want[k]:
                if abs(t - p) > 1e-3:
                    raise ValueError(f'{clip}: frame {k} PTS {t:.6f} s differs from the window ({p:.6f} s)')
            r = do_khung(ctx, cam, g)
            if r['cam'] is not None:
                cam = r['cam']
            p1[k] = r
    miss = sorted(set(want) - set(p1))
    if miss:
        raise ValueError(f'{clip}: frames {miss[:5]}... not in the clip')
    own, why = {}, {}
    for key, fr in wins:
        tags = [p1[i]['tag'] for i, _ in fr if p1[i]['tag'] is not None and p1[i]['cam_ok']]
        own[key] = (np.median([[x['tag_x'], x['tag_y']] for x in tags], axis=0),
                    circ_median([x['tag_psi'] for x in tags])) if tags else None
    pred, keys = {}, [key for key, _ in wins]                 # windows in time order
    for j, key in enumerate(keys):
        if own[key] is not None:
            pred[key] = own[key]
            continue
        near = sorted((abs(j2 - j), j2) for j2, k2 in enumerate(keys) if own[k2] is not None)   # LLC1-16: before or after
        pred[key] = own[keys[near[0][1]]] if near else None
        why.setdefault(key, []).append('du_doan_tu_cua_so_gan_nhat' if near else 'khong_du_doan_neo')
    out = {key: [] for key, _ in wins}
    for k, t, g in cm.iter_frames(clip):
        if k not in want:
            continue
        r1 = p1[k]
        for key, p in want[k]:
            rec = dict(id=k, t=p, cam_ok=r1['cam_ok'], F=NAN * np.ones(2), H=NAN, Ph=None, neo_ok=False, tag=None)
            if r1['cam'] is not None:
                rec['F'], rec['H'] = np.asarray(r1['cam'].centre[:2], float), float(r1['cam'].centre[2])
            if r1['tag'] is not None:
                rec['tag'] = dict(x=r1['tag']['tag_x'], y=r1['tag']['tag_y'], psi=r1['tag']['tag_psi'],
                                  s=r1['tag']['scale'], res_sim=r1['tag']['res_sim_m'])
            if r1['cam_ok'] and pred[key] is not None:
                uvp, rot = mau.du_doan(r1['cam'], *pred[key])
                pts, ok = {}, True
                for j, it in enumerate(mau.items):
                    u, v, c, q = khop_mau(g, mau.tpl(j, rot), uvp[j, 0], uvp[j, 1], ctx.cfg['search_px'])
                    if not (math.isfinite(u) and c >= ctx.cfg['corr_min'] and q < ctx.cfg['peak2_ratio']):
                        ok = False
                        break
                    P = lift(r1['cam'].pix2floor(np.array([[u, v]]), 0.0)[0], ctx.kappa_n, ctx.C)
                    pts.setdefault(it['name'], []).append(P)
                if ok:
                    rec['Ph'] = np.array([np.mean(pts[n], axis=0) for n in mau.neo])
                    rec['neo_ok'] = True
            out[key].append(rec)
    return out, why


# ---------------------------------------------------------------- windows (geometry layer, records -> summaries)
def tom_neo(recs, n_khung, cfg):
    """v4 3 (LL3-7): anchor frame set of a window: camera and anchor gates; still (95th percentile of each anchor's
    deviation <= still_neo_m); >= min_khung frames and >= min_frac of the window. The tag is NOT needed."""
    use = [r for r in recs if r['cam_ok'] and r['neo_ok']]
    out = dict(n=len(use), n_cua_so=n_khung, ids=[r['id'] for r in use], pts=[r['t'] for r in use], ok=False, ly_do=[])
    if len(use) < cfg['min_khung'] or len(use) < cfg['min_frac'] * n_khung:
        out['ly_do'].append('thieu_khung_neo')
        return out
    Ph = np.array([r['Ph'] for r in use])
    med = np.median(Ph, axis=0)
    p95 = np.percentile(np.linalg.norm(Ph - med[None], axis=2), 95, axis=0)
    out['p95_m'] = p95.tolist()
    if not np.all(p95 <= cfg['still_neo_m']):
        out['ly_do'].append('neo_khong_dung_yen')
        return out
    out.update(ok=True, Ph=Ph.mean(axis=0), F=np.mean([r['F'] for r in use], axis=0), H=float(np.mean([r['H'] for r in use])))
    return out


def tom_chung(recs, ma_ids, a6, a6_ok, ctx, cfg, neo_ids):
    """Tag sets of a window (v4 5, 8; LLC1-13): 'ma' = anchor set AND CAM tag gate AND tag seen (no A6: centre and scale
    comparisons); 'chung' = 'ma' AND A6 ok (psi' = tag_psi - dpsi_A6; LL3-8 ambiguous branch dropped)."""
    ids = set(neo_ids)
    use = [r for r in recs if r['id'] in ids and r['id'] in ma_ids and r['tag'] is not None]
    out = dict(ok_ma=False, ok=False, ly_do=[], ids_ma=[], pts_ma=[], ids=[], pts=[])
    if len(use) < cfg['min_khung']:
        out['ly_do'].append('thieu_khung_ma')
        return out

    def arrs(rs):
        return dict(Ph=np.array([r['Ph'] for r in rs]), F=np.array([r['F'] for r in rs]), H=np.array([r['H'] for r in rs]),
                    t=np.array([[r['tag']['x'], r['tag']['y']] for r in rs]), s=np.array([r['tag']['s'] for r in rs]))
    out.update(ok_ma=True, ids_ma=[r['id'] for r in use], pts_ma=[r['t'] for r in use], ma=arrs(use),
               psi_tho=np.array([r['tag']['psi'] for r in use]))
    if a6 is None or not a6_ok:
        out['ly_do'].append('a6_khong_ap_dung' if a6 is not None else 'thieu_a6')
        return out
    xy = np.array([[r['tag']['x'], r['tag']['y']] for r in use])
    dp, ok = a6.dpsi(xy, [r['tag']['psi'] for r in use], ctx.C, ctx.Zc)
    u2 = [(r, d) for r, d, k in zip(use, dp, ok) if k and math.isfinite(d)]
    if len(u2) < cfg['min_khung']:
        out['ly_do'].append('thieu_khung_chung')
        return out
    psi = np.array([wrap(r['tag']['psi'] - d) for r, d in u2])
    if mo_ho(psi, cfg['mo_ho_deg']):
        out['ly_do'].append('mo_ho_nhanh')
        return out
    out.update(ok=True, ids=[r['id'] for r, _ in u2], pts=[r['t'] for r, _ in u2], psi=psi, **arrs([r for r, _ in u2]))
    return out


def tinh_chung(ch, G, h, ctx, psi=None):
    """Per-frame body poses (G, h locked) -> merged angles and means: alpha (gop), psi' (gop) and theta when psi is
    given, o, tag centre, scale, camera."""
    al, oo = [], []
    for Ph, F, H in zip(ch['Ph'], ch['F'], ch['H']):
        a_, o_, _ = tu_the(G, h, Ph, F, H, ctx.C, ctx.a_n)
        al.append(a_)
        oo.append(o_)
    alpha = gop_goc(al)
    out = dict(alpha=alpha, o=np.mean(oo, axis=0), t=ch['t'].mean(axis=0), s=float(ch['s'].mean()),
               F=ch['F'].mean(axis=0), H=float(ch['H'].mean()), n=len(al), psi=None, theta=None)
    if psi is not None:
        p = gop_goc(psi)
        out.update(psi=p, theta=wrap(p - alpha))
    return out


# ---------------------------------------------------------------- v4 6b: delta_c_B (LL5-1..6)
def tam_ma_that(t, z, F, H, ctx, g_san=None):
    """LL5-1, LL5-3: tag centre (car_pose, before the offset) inverted by the forward equation of v4 2 at height z,
    after removing the Kabsch-centre shift of a tilted tag L^2 g_floor / (4 (H - h)) (H: refitted camera height of the
    window; the design's Z_c - h read as the H_w - h of the forward equation, LL5-1)."""
    t = np.asarray(t, float)
    if g_san is not None:
        t = t - ctx.side ** 2 * np.asarray(g_san, float) / (4.0 * (H - ctx.tag_h))
    return nghich(t, z, F, H, ctx.C, ctx.a_t)[0]


def _psi_w(w):
    """Tag yaw used to turn g_b to the floor: psi' when A6 applied, else the raw merged yaw."""
    return w['psi'] if w.get('psi') is not None else w['psi_tho']


def thang_sua(s, t, g_san, ctx, H, F):
    """v4 6b: apparent tag scale with the first-order tilt part g_floor . (p - F) / (2 (H - h)) removed (ray centre
    F, H of the forward equation, LL5-1; the design's (p - C) / (Z_c - h) is the case F = C, H = Z_c)."""
    return s - float(np.dot(g_san, np.asarray(t, float) - np.asarray(F, float))) / (2.0 * (H - ctx.tag_h))


def _s_sua(w, ctx, g_b):
    return thang_sua(w['s'], w['t'], rot2(_psi_w(w)) @ g_b, ctx, w['H'], w['F'])


def c_B_cua(w, ctx, g_b, dz=0.0):
    """Tag centre in the anchor frame of a window summary (v4 6, 6b): R(-alpha)(P_t - o)."""
    Pt = tam_ma_that(w['t'], ctx.tag_h + dz, w['F'], w['H'], ctx, rot2(_psi_w(w)) @ g_b)
    return rot2(-w['alpha']) @ (Pt - w['o'])


def _kiem_mien_so(*v):
    if not all(math.isfinite(float(x)) for x in v):
        raise ValueError('6b: non-finite value')


def delta_h_uoc(ws, wd, ctx, g_b):
    """Dh estimate (v4 6b) from the scale ratio, tilt part of both scales removed (one g_b, LL5-6). With the forward
    equation (LL5-1) s = k H / (H - z): Dh = H_d - h - (s_s / s_d) H_d (H_s - h) / H_s (k, the edge bias, cancels);
    = (H - h)(1 - s_s / s_d) when H_s = H_d (the design's (Z_c - h)(1 - s_s / s_d) with Z_c read as H)."""
    ss, sd, h = _s_sua(ws, ctx, g_b), _s_sua(wd, ctx, g_b), ctx.tag_h
    if not (ss > 0 and sd > 0 and ws['H'] > h and wd['H'] > h):
        raise ValueError('6b: scale or H - h not positive')
    return wd['H'] - h - (ss / sd) * wd['H'] * (ws['H'] - h) / ws['H']


def delta_c_B(ws, wd, ctx, g_b):
    """(delta_c_B corrected, raw, Dh estimate): target minus standard, anchor frame."""
    dh = delta_h_uoc(ws, wd, ctx, g_b)
    cs = c_B_cua(ws, ctx, g_b)
    return c_B_cua(wd, ctx, g_b, dh) - cs, c_B_cua(wd, ctx, g_b) - cs, dh


def trai_nhom(ws12, ctx, g_b):
    """Spreads of a group of 12b windows of one placement (v4 6, 6b, 8): theta (deg; None without A6), c_B (largest
    pair distance), scale ratio max |s_i / s_j - 1| (tilt removed, LL5-2)."""
    cs = [c_B_cua(w, ctx, g_b) for w in ws12]
    ss = [_s_sua(w, ctx, g_b) for w in ws12]
    th = [w['theta'] for w in ws12]
    return dict(theta=(max((abs(math.degrees(wrap(a - b))) for a, b in itertools.combinations(th, 2)), default=NAN)
                       if all(x is not None for x in th) else NAN),
                c_B=max((float(np.linalg.norm(a - b)) for a, b in itertools.combinations(cs, 2)), default=NAN),
                s=max((abs(a / b - 1.0) for a, b in itertools.permutations(ss, 2)), default=NAN), n=len(ws12))


def u_delta_c_B(ws, wd, ctx, g_b, eps_g, trai_c, trai_s, eps_pos, du_giu):
    """u_delta_c_B_m before the margin (v4 6b, LL5-1..5, LLC1-12): window noise + Dh index propagated through the
    ratio formula (12b ratio repeatability, eps_g of each end) times the arm |P_t - F_d| from the corrected centre +
    tilt centre index of each end with its own H + Dh x place offset + anchor position error (both ends) + held-out."""
    h = ctx.tag_h
    hs, hd = ws['H'] - h, wd['H'] - h
    ss, sd = _s_sua(ws, ctx, g_b), _s_sua(wd, ctx, g_b)
    _, _, dh = delta_c_B(ws, wd, ctx, g_b)
    rho_s = float(np.linalg.norm(np.asarray(ws['t']) - ws['F']))
    rho_d = float(np.linalg.norm(np.asarray(wd['t']) - wd['F']))
    K = wd['H'] * hs / ws['H']
    # LLC1-12 (round 2): the relative repeatability scales the ratio; never below the design's (H - h) x spread form
    u_ratio = max(ss / sd, 1.0) * (trai_s + eps_g * rho_s / (2.0 * hs * ss) + eps_g * rho_d / (2.0 * hd * sd))
    u_dh = K * u_ratio
    Pt = tam_ma_that(wd['t'], h + dh, wd['F'], wd['H'], ctx, rot2(_psi_w(wd)) @ g_b)
    # place offset term with the ray arm of each end, (p - F_w) (LL5-1 reading of Delta(p - C)), not t_d - t_s
    arm = (np.asarray(wd['t'], float) - wd['F']) - (np.asarray(ws['t'], float) - ws['F'])
    parts = dict(nhieu=trai_c, dh=u_dh / hd * float(np.linalg.norm(Pt - wd['F'])),
                 nghieng=ctx.side ** 2 * eps_g / 4.0 * (1.0 / hs + 1.0 / hd),
                 lech_cho=abs(dh) * float(np.linalg.norm(arm)) / hd,
                 neo=eps_pos, giu_ngoai=du_giu)
    _kiem_mien_so(u_dh, *parts.values())
    if any(v < 0 for v in parts.values()):
        raise ValueError('6b: negative index part')
    return float(sum(parts.values())), parts, u_dh


GIOI_HAN_V4 = [
    'Moi so la chi so thuc nghiem TRUOC le; le 1,25 do C8B nhan mot lan (C8BX-24).',
    'LL4-1 (Liam nhan 19/11): bam neo khong co sai cung phu thuoc cho/clip qua u_beta (cap truoc-sau khong thay).',
    'LL4-3 (Liam nhan 19/11): hai mieng bang dinh khong cung xoay quanh cum oc.',
    'v4 6b: lech do canh cua ma nhu nhau giua hai lan lap o cung cho; mat neo khong doi do cao giua hai dau.',
    'LL5-4 (gui CODE gom trinh Liam): ca cum neo khong dich cung chung phu thuoc cho/clip qua chi so.',
    'LL5-5 (gui CODE gom trinh Liam): chi so giu ngoai vi tri lay tu cap cung cho co trong buoi, dung cho moi cho.',
    'u_beta la chi so thuc nghiem trong mien da kiem (v4 11); delta_a, q, delta_h_goc chi chan doan.',
    'Gop tam theo trung binh cua so (khong theo tung khung): lech bac hai theo dao dong goc (LLC1-18 DEFERRED).',
    'LLC2-2: cac 12b lien nhau o cac cho khac nhau la mot lan lap (khong so duoc khac cho); 12b lien nhau cung cho '
    'phai so khong_doi.',
    'LLC1-11: cuc dai phep doi mot neo tinh tuyen tinh (phi tuyen con lai co 1e-7 do o hinh test, P3 DEFERRED).',
    'LLC1-13, LLC2-3: chua co phep so thang chi bang ma (khong neo); mat neo -> so thang khong_xac_dinh (DEFERRED, '
    'gui CODE).',
    'LLC1-16: cua so mat ma chi du doan neo tu cua so gan nhat cung clip; khong du doan -> cua so hong (an toan).',
]


# ---------------------------------------------------------------- F2 checks (LLC1-7)
def kiem_f2(rows):
    """Structure of the window file before any measurement: unique IDs and finite increasing PTS in a window, one pre
    and one post per run, no window given twice, no frame shared by the fit and held-out roles."""
    seen, roles, luot, cung_loai = {}, {}, {}, {}
    for i, w in enumerate(rows):
        fr = khung_cua(w)
        ids, pts = [k for k, _ in fr], [p for _, p in fr]
        if len(set(ids)) != len(ids):
            raise ValueError(f'window {i}: frame IDs repeated')
        if not all(math.isfinite(p) for p in pts) or any(b <= a for a, b in zip(pts, pts[1:])) \
                or any(b <= a for a, b in zip(ids, ids[1:])):
            raise ValueError(f'window {i}: PTS / IDs not finite and increasing')
        for kk, lst in ((TEN['k_sha'], ids), (TEN['ma_sha'], sorted(khung_ma(w)))):
            if w.get(kk) is not None and w[kk] != sha_ids(lst):
                raise ValueError(f'window {i}: {kk} is not sha_ids of its frame list')
        key = (w[TEN['clip_sha']], tuple(ids))
        if ids and key in seen:                        # an empty window (CT4-3) is a failed window, not a duplicate
            raise ValueError(f'window {i}: the same frames as window {seen[key]}')
        seen[key] = i
        role = 'khop' if w[TEN['loai']] in LOAI_KHOP else 'giu'
        for k in ids:
            r0 = roles.setdefault((w[TEN['clip_sha']], k), role)
            if r0 != role:
                raise ValueError(f'window {i}: frame {k} in both the fit and the held-out sets (D1)')
            i0 = cung_loai.setdefault((w[TEN['clip_sha']], w[TEN['loai']], k), i)
            if i0 != i:                    # LLC1-7: overlapping windows of one type are not independent ends / pairs
                raise ValueError(f'window {i}: frame {k} also in window {i0} of the same type {w[TEN["loai"]]}')
        if role == 'giu':
            d = luot.setdefault(w[TEN['luot']], {})
            if w[TEN['loai']] in d:
                raise ValueError(f'run {w[TEN["luot"]]}: two {w[TEN["loai"]]} windows')
            d[w[TEN['loai']]] = i


# ---------------------------------------------------------------- hieu_chuan (v4 1, 3, 9; F6)
def _measure(cs_rows, keys, session_path, mau_args, ctx, neo, clip_dir):
    """Records of the windows `keys` (indices into cs_rows), clip by clip."""
    mau = _mau(mau_args, ctx, neo)
    by_clip, recs, flags = {}, {}, {}
    for i in keys:
        w = cs_rows[i]
        fr = khung_cua(w)
        if not fr:                                     # CT4-3: no frame -> no record (the window fails, run kept)
            recs[i] = []
            continue
        by_clip.setdefault(w[TEN['clip']], []).append((i, fr))
    for clip, wins in by_clip.items():
        path = clip if os.path.isabs(clip) or clip_dir is None else os.path.join(clip_dir, clip)
        sha = cm.sha256(path)
        for i, _ in wins:
            if cs_rows[i][TEN['clip_sha']] != sha:
                raise ValueError(f'{clip}: sha256 differs from the window file')
        wins.sort(key=lambda kw: min(p for _, p in kw[1]))
        r, f = do_clip(path, wins, ctx, mau)
        recs.update(r)
        flags.update(f)
    return recs, flags, mau


def _mau(mau_args, ctx, neo):
    """Template frame of the anchors: (clip, session, idx) -> NeoMau (camera of that session)."""
    clip, ses_path, idx = mau_args
    ses = doc_json(ses_path)
    mctx = LLCtx(ses, ctx.h_n, ctx.cfg)
    fr = frames_of(clip, [idx], with_time=True).get(idx)
    if fr is None:
        raise ValueError(f'{clip}: no frame {idx}')
    r = do_khung(mctx, mctx.cam0, fr[1])
    if not r['cam_ok'] or r['tag'] is None:
        raise ValueError(f'{clip}: template frame {idx} fails the camera / tag gate')
    return NeoMau(fr[1], r['cam'], r['tag'], neo, mctx.kappa_n, mctx.C, ctx.cfg)


def tom_cua_so(w, recs, a6, a6_ok, ctx, cfg):
    fr = khung_cua(w)
    pts = [p for _, p in fr]
    nb = tom_neo(recs, len(fr), cfg)
    ch = tom_chung(recs, khung_ma(w), a6, a6_ok, ctx, cfg, nb['ids']) if nb['ok'] else \
        dict(ok=False, ok_ma=False, ly_do=['thieu_neo'], ids=[], pts=[], ids_ma=[], pts_ma=[])
    canh = kiem_canh(w, pts)
    ma = khung_ma(w)
    giao = [r for r in recs if r['cam_ok'] and r['neo_ok'] and r['id'] in ma and r['tag'] is not None]
    kl = w.get('ket_luan') if isinstance(w.get('ket_luan'), dict) else {}
    if w[TEN['loai']] in LOAI_GIU:          # LLC1-7 (round 2): CAM's stillness verdict is required, not assumed
        cam_dung, ly_kl = kl.get('dung_yen') is True, ('cam_khong_dung_yen' if 'dung_yen' in kl else 'thieu_ket_luan_dung_yen')
    else:
        cam_dung, ly_kl = kl.get('dung_yen', True) is not False, 'cam_khong_dung_yen'
    ly = nb['ly_do'] + ([] if canh else ['canh_lenh']) + ([] if cam_dung else [ly_kl])
    return dict(meta={k: w.get(TEN[k]) for k in ('luot', 'cho', 'loai', 'clip', 'clip_sha', 'session_sha', 'frames_sha')},
                t0=min(pts) if pts else NAN, neo=nb, chung=ch, canh_ok=canh, ok=bool(nb['ok'] and canh and cam_dung),
                ly_do=ly, giao=dict(id=[r['id'] for r in giao], pts_s=[r['t'] for r in giao]), mat_neo=not giao)


def _du_tap(arr, G, h, ctx):
    """LLC1-2 (round 2): largest anchor residual of the mean anchor points of a frame subset (the subset that a summary
    is built on), same pose fit as the anchor set."""
    _, _, res = tu_the(G, h, np.mean(arr['Ph'], axis=0), np.mean(arr['F'], axis=0), float(np.mean(arr['H'])), ctx.C,
                       ctx.a_n)
    return float(np.max(res))


def _tom_hai(s, G, h, ctx, r_max):
    """'ma' and 'chung' summaries of a window (JSON-able dicts or None); a subset whose own anchor residual exceeds
    r_max is dropped (LLC1-2: the anchor gate is checked on the frames that the position uses). Returns
    (ma, chung, reasons)."""
    ch, ly = s['chung'], []
    ma = chung = None
    if ch.get('ok_ma'):
        if _du_tap(ch['ma'], G, h, ctx) <= r_max:
            ma = tinh_chung(ch['ma'], G, h, ctx)
            ma['psi_tho'] = gop_goc(ch['psi_tho'])
        else:
            ly.append('du_neo_tap_ma_vuot_r_max')
    if ch.get('ok'):
        if _du_tap(ch, G, h, ctx) <= r_max:
            chung = tinh_chung(ch, G, h, ctx, ch['psi'])
        else:
            ly.append('du_neo_tap_chung_vuot_r_max')
    return ma, chung, ly


def nhom_12b(cua, luot12, cho):
    """LLC1-4: 12b windows of one 12b run (luot_id; one run, the plate is not touched) at the place of its standard
    (round 2: same place only), with a common set."""
    return [c for c in cua if c['meta']['loai'] in LOAI_12B and c['meta']['luot'] == luot12 and c['meta']['cho'] == cho
            and c['chung']]


def khoa_hieu_chuan(wsum, ctx, cfg, a6):
    """v4 3, 4 on the fit windows: shape fit, rank, r_max / eps_neo, fit residual, window noise, per-place standard.
    wsum: list of (window dict, summary) of the fit set."""
    use = [(w, s) for w, s in wsum if s['ok']]
    out = dict(ly_do=[])
    if len(use) < 2:
        out.update(trang_thai='chua_ket_luan', ly_do=['thieu_cua_so_khop'])
        return out
    obs = [dict(Ph=s['neo']['Ph'], F=s['neo']['F'], H=s['neo']['H']) for _, s in use]
    fit = khop_hinh(obs, ctx.h_n, ctx.C, ctx.a_n)
    hg = hang(fit, cfg)
    bangs = [eps_neo_bang(fit['G'], fit['h'], o['F'], o['H'], ctx.C, ctx.a_n) for o in obs]   # LLC1-11: every window
    r_max, eps1 = r_max_tu(bangs, cfg)
    du = float(fit['res'].max())
    tt = 'dat'
    if du > r_max:
        tt, out['ly_do'] = 'chua_ket_luan', ['neo_khong_kha_thi']
    elif hg['suy_bien']:
        tt, out['ly_do'] = 'chua_ket_luan', ['thieu_hang']
    g_b = a6.g_b()
    cua = []
    for (w, s), p, res in zip(use, fit['poses'], fit['res']):
        ma, chung, ly_du = _tom_hai(s, fit['G'], fit['h'], ctx, r_max)
        c = dict(meta=s['meta'], t0=s['t0'], thu_tu=s['thu_tu'], alpha=float(p[0]), o=p[1:].tolist(),
                 Ph=s['neo']['Ph'].tolist(),
                 F=s['neo']['F'].tolist(), H=s['neo']['H'], du_m=res.tolist(), n_neo=s['neo']['n'],
                 khung_neo=dict(id=s['neo']['ids'], pts_s=s['neo']['pts']), khung_neo_sha256=sha_ids(s['neo']['ids']),
                 ma=None if ma is None else _jsonable(ma), chung=None if chung is None else _jsonable(chung),
                 ly_do_chung=s['chung']['ly_do'] + ly_du)
        if chung is not None:
            c['khung_chung'] = dict(id=s['chung']['ids'], pts_s=s['chung']['pts'])
            c['khung_chung_sha256'] = sha_ids(s['chung']['ids'])
        cua.append(c)
    # window noise (v4 4): adjacent fit windows of a clip, |wrap(dalpha - dpsi')| (alpha of the anchor set)
    nh = []
    for clip in {c['meta']['clip'] for c in cua}:
        cc = sorted([c for c in cua if c['meta']['clip'] == clip and c['chung']], key=lambda c: c['t0'])
        nh += [abs(math.degrees(wrap(wrap(b['alpha'] - a['alpha']) - wrap(b['chung']['psi'] - a['chung']['psi']))))
               for a, b in zip(cc, cc[1:])]
    nhieu = max(nh) if nh else NAN
    chuan = {}                     # J v6 2: one standard per 12b run, key 'cho|luot_id'
    for r in [c for c in cua if c['meta']['loai'] == LOAI_CHUAN]:
        key = f"{r['meta']['cho']}|{r['meta']['luot']}"
        if r['chung'] is None:
            chuan[key] = dict(trang_thai='chua_ket_luan', ly_do='thieu_khung_chung', cho=r['meta']['cho'],
                              luot_id=r['meta']['luot'])
            continue
        cw = _w_from_hc(r)
        chuan[key] = dict(trang_thai='dat' if math.isfinite(nhieu) else 'chua_ket_luan',
                          ly_do='' if math.isfinite(nhieu) else 'thieu_cap_nhieu', cho=r['meta']['cho'],
                          luot_id=r['meta']['luot'], clip_sha256=r['meta']['clip_sha'],
                          khung_chung_sha256=r['khung_chung_sha256'], psi_chuan_k_deg=math.degrees(r['chung']['psi']),
                          u_psi_chuan_k_cua_so_deg=nhieu, alpha_chuan_k_deg=math.degrees(r['alpha']),
                          theta_chuan_k_deg=math.degrees(r['chung']['theta']),
                          c_B_chuan_k_m=c_B_cua(cw, ctx, g_b).tolist(),
                          n_12b_cung_lan_lap=len(nhom_12b(cua, r['meta']['luot'], r['meta']['cho'])),
                          frames_sha256=r['meta'].get('frames_sha'))
    cho = {}                       # per place when the place has exactly one 12b run (F6 as agreed); else see chuan_12b
    for k in CHO:
        ks = [key for key, v in chuan.items() if v['cho'] == k]
        cho[k] = chuan[ks[0]] if len(ks) == 1 else dict(
            trang_thai='chua_ket_luan', ly_do='nhieu_12b_xem_chuan_12b' if ks else 'khong_co_cua_so_chuan', khoa=ks)
    a_ref = cua[0]['alpha']
    mien = [[*(rot2(-c['alpha']) @ (np.asarray(c['o']) - ctx.C)), a_ref + wrap(c['alpha'] - a_ref)] for c in cua]
    out.update(trang_thai=tt, G=fit['G'], h=fit['h'], sigma_h=hg['sigma_h'], cov_h=hg.get('cov_h'), s_rel=hg['s_rel'],
               huong_yeu=hg['yeu'], suy_bien=hg['suy_bien'], s_fit_m=hg.get('s_fit_m'), hoi_tu=fit['hoi_tu'],
               r_max_m=r_max, eps_neo_deg=eps1, du_lon_nhat_m=du, nhieu_cua_so_deg=nhieu, n_cap_nhieu=len(nh), cua_so=cua,
               cho=cho, chuan_12b=chuan, x=fit['x'], idx_h=fit['idx_h'], mien_khop=dict(dinh=mien, a_ref=a_ref,
                                                                         cach='(R(-alpha)(o - C), alpha) moi cua so khop'))
    return out


def run_hieu_chuan(cua_so_path, session_path, neo_spec, mau, out_dir, a6_path, tam_id, kiem_r=None, h_neo=None,
                   clip_dir=None, cfg=None, a6_mod=None, _records=None):
    """v4 1, 3, 9: fit windows only -> hieu_chuan_neo.json, locked (time, sha256) before any held-out window is read.
    mau = (clip, session, idx) of the template frame of the anchor boxes. Returns the exit code (0 dat, 1 written with
    chua_ket_luan, 2 error)."""
    cfg = dict(LL_DEFAULTS, **(cfg or {}))
    rows = doc_cua_so(cua_so_path)
    kiem_f2(rows)
    ses = doc_json(session_path)
    ses_sha = cm.sha256(session_path)
    neo = parse_neo(neo_spec)
    r_ok, r_ev = kiem_r_neo(neo, kiem_r, mau[0], mau[2])
    neo_dung = {k: v for k, v in neo.items() if r_ok[k]}
    ctx = LLCtx(ses, ses['tag_h_m'] if h_neo is None else h_neo, cfg)
    a6 = A6(a6_path, a6_mod)
    a6_tt = a6.ap_dung(tam_id)
    a6_ok = (a6_tt[0] if isinstance(a6_tt, (list, tuple)) else a6_tt) != 'chua_ket_luan'
    keys = [i for i, w in enumerate(rows) if w[TEN['loai']] in LOAI_KHOP]
    for i in keys:
        if rows[i][TEN['session_sha']] != ses_sha:
            raise ValueError(f'window {i}: session sha256 differs from {session_path}')
    mau_sha = cm.sha256(mau[0])
    for i, w in enumerate(rows):                         # LLC1-6: the template frame is not a held-out frame
        if w[TEN['loai']] in LOAI_GIU and w[TEN['clip_sha']] == mau_sha and int(mau[2]) in {k for k, _ in khung_cua(w)}:
            raise ValueError(f'template frame {mau[2]} belongs to held-out window {i}')
    cm._new_dir(out_dir)
    out = dict(schema=TEN['hieu_chuan'], version=LL_VERSION, hil=git_head(), thiet_ke='2026-11-19_Thiet_ke_nhac_tam_lan_lap_v4',
               cua_so_file=dict(file=os.path.abspath(cua_so_path), sha256=cm.sha256(cua_so_path)),
               session=dict(file=os.path.abspath(session_path), sha256=ses_sha, fit_ids=ctx.fit_ids,
                            plumb_xy_m=ctx.C.tolist(), Z_c_m=ctx.Zc, tag_h_m=ctx.tag_h),
               a6=dict(file=os.path.abspath(a6_path), sha256=a6.sha256, schema=TEN['a6'], tam_id=tam_id, ap_dung=a6_tt),
               neo=[dict(ten=k, o_px=[list(b) for b in v], kiem_r=r_ok[k]) for k, v in neo.items()], kiem_r=r_ev,
               mau=dict(clip=os.path.abspath(mau[0]), sha256=mau_sha, session=os.path.abspath(mau[1]),
                        session_sha256=cm.sha256(mau[1]), idx=int(mau[2])),
               h_n_m=ctx.h_n, quy_uoc='goc G o neo A, truc x theo A->B; h_A = h_n (v4 3, LL3-2)',
               phep_gop='trung_binh_go_vong', vai_tro=dict(khop=list(LOAI_KHOP), giu_ngoai=list(LOAI_GIU),
                                                             chuan=LOAI_CHUAN, dich=LOAI_DICH),
               gioi_han=GIOI_HAN_V4, nguong=dict(cfg, trang_thai=LL_THRESH_STATUS))
    if len(neo_dung) < 3 or 'A' not in neo_dung or 'B' not in neo_dung:
        out.update(trang_thai='chua_ket_luan', ly_do=['thieu_neo'])
    else:
        if _records is None:
            recs, flags, _ = _measure(rows, keys, session_path, mau, ctx, neo_dung, clip_dir)
        else:
            recs, flags = _records, {}
        wsum = [(rows[i], dict(tom_cua_so(rows[i], recs[i], a6, a6_ok, ctx, cfg), thu_tu=i)) for i in keys]
        kh = khoa_hieu_chuan(wsum, ctx, cfg, a6)
        out.update(trang_thai=kh['trang_thai'], ly_do=kh['ly_do'], co_cua_so={str(k): v for k, v in flags.items()},
                   cua_so_loi=[dict(meta=s['meta'], ly_do=s['ly_do'] + s['chung']['ly_do']) for _, s in wsum if not s['ok']])
        out.update({k: v for k, v in kh.items() if k not in ('trang_thai', 'ly_do')})
        out['neo_ten'] = list(neo_dung)
    out['khoa'] = dict(thoi_diem=datetime.datetime.now().astimezone().isoformat(), noi_dung_sha256=sha_obj(out),
                       ghi_chu='khoa truoc khi doc cua so giu ngoai (v4 1, D1); lan_lap kiem lai sha256 nay')
    with open(os.path.join(out_dir, 'hieu_chuan_neo.json'), 'w', encoding='utf-8') as f:
        json.dump(_jsonable(out), f, ensure_ascii=False, indent=1, allow_nan=False)
    return 0 if out['trang_thai'] == 'dat' else 1


def doc_hieu_chuan(path):
    """hieu_chuan_neo.json with its lock checked (the content sha256 must match: nothing changed after the lock)."""
    o = doc_json(path)
    if o.get('schema') != TEN['hieu_chuan']:
        raise ValueError(f'{path}: schema {o.get("schema")} is not {TEN["hieu_chuan"]}')
    k = o.pop('khoa', None)
    if k is None or sha_obj(o) != k['noi_dung_sha256']:
        raise ValueError(f'{path}: lock sha256 does not match the content (changed after hieu_chuan)')
    o['khoa'] = k
    return o


# ---------------------------------------------------------------- lan_lap (v4 0, 3-9; F8)
def _w_from_hc(c, which='chung'):
    """Window summary of the fit set back from hieu_chuan_neo.json (means of the 'chung' or 'ma' set)."""
    ch = c[which]
    if ch is None:
        return None
    return dict(alpha=ch['alpha'], psi=ch.get('psi'), psi_tho=ch.get('psi_tho', ch.get('psi')), theta=ch.get('theta'),
                o=np.array(ch['o']), t=np.array(ch['t']), s=ch['s'], F=np.array(ch['F']), H=ch['H'], n=ch['n'])


def so_sanh(wa, wb, ctx, g_b, nguong):
    """v4 8 (C8BX-18, LL4-2, LLC1-13): two windows (summaries with or without A6): theta (needs A6 at both ends),
    c_B (Dh-corrected delta_c_B), scale (Dh / (H - h) from the ratio), each doi / khong_doi / khong_xac_dinh.
    g_b None (A6 does not apply to this plate, LLC2-4): the tilt of the plate is unknown, so c_B and the scale are
    khong_xac_dinh (no correction with another plate's g_b)."""
    def st(x, lim):
        if x is None or not (math.isfinite(x) and math.isfinite(lim)):
            return dict(gia_tri=x, nguong=lim, trang_thai='khong_xac_dinh')
        return dict(gia_tri=x, nguong=lim, trang_thai='doi' if x > lim else 'khong_doi')
    th = cb = sc = None
    if wa is not None and wb is not None:
        if wa.get('theta') is not None and wb.get('theta') is not None:
            th = abs(math.degrees(wrap(wb['theta'] - wa['theta'])))
        if g_b is not None:
            try:
                dcb, _, dh = delta_c_B(wa, wb, ctx, g_b)
                cb, sc = float(np.linalg.norm(dcb)), abs(dh) / (wb['H'] - ctx.tag_h)
            except ValueError:
                pass
    return dict(theta=st(th, nguong['theta']), c_B=st(cb, nguong['c_B']), s=st(sc, nguong['s']))


def tong_doi(ss):
    st = [x['trang_thai'] for x in ss.values()]
    return True if 'doi' in st else (None if 'khong_xac_dinh' in st else False)


def _hai(x):
    """Best summary of a held-out window for 8: 'chung' (A6) when there, else 'ma'."""
    if x is None:
        return None
    return x.get('chung') or x.get('ma')


def doc_f15(f15_path, hc, ctx):
    """c7_meo_dau_chan_bang_v1 with its source checked (v4 13; F15C-2): schema, content sha256, made from a lan_lap
    of THIS hieu_chuan_neo.json, neo_ten, chieu held_minus_full, bo_H32 and truoc_le true, kappa_n = h_n / Z_c.
    Returns (object, file sha256); a wrong source raises ValueError (code 2)."""
    with open(f15_path, 'rb') as fh:
        raw = fh.read()
    o, f_sha = json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()
    if o.get('schema') != TEN['f15']:
        raise ValueError(f'{f15_path}: schema {o.get("schema")} is not {TEN["f15"]}')
    body = {k: v for k, v in o.items() if k != 'sha256'}
    if hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != o.get('sha256'):
        raise ValueError(f'{f15_path}: content sha256 does not match')
    if (o.get('lan_lap') or {}).get('hieu_chuan_sha256') != hc['_file_sha256']:
        raise ValueError(f'{f15_path}: made from a lan_lap of another hieu_chuan_neo.json')
    if o.get('neo_ten') != hc.get('neo_ten') or o.get('chieu') != 'held_minus_full':
        raise ValueError(f'{f15_path}: neo_ten / chieu differ')
    if o.get('bo_H32') is not True or o.get('truoc_le') is not True:
        raise ValueError(f'{f15_path}: bo_H32 and truoc_le must be true (held-out cases, before the margin)')
    if not abs(float(o.get('kappa_n', NAN)) - ctx.kappa_n) <= 1e-12:
        raise ValueError(f'{f15_path}: kappa_n differs from h_n / Z_c of the session')
    for cho in o.get('theo_cho') or {}:
        if cho not in CHO:
            raise ValueError(f'{f15_path}: place {cho!r}')
    return o, f_sha


def _vec2(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def beta_cam_replay(f15_path, hc, out_l, luot, hw, fitw, doan, G, h, ctx):
    """v4 13 (F15 b, F16): largest change of beta per place when the anchor points of BOTH ends (standard, pre) move
    by the map residual of each held-out camera case of KIEMCHEO (c7_meo_dau_chan_bang_v1). Only runs whose standard
    is verified (chuan_12b, F15C-4). A duplicated or wrong-source entry, a case vector missing / not a finite 2-vector
    at any anchor of either end, or a place with runs but no block -> that place chua_ket_luan (F15C-1, 3, 5).
    Returns {cho: {max_abs_dbeta_rad, n, trang_thai, ly_do, f15_sha256}}."""
    o, f_sha = doc_f15(f15_path, hc, ctx)
    ten = o['neo_ten']
    can = {}
    for r in out_l:
        lid = r['luot_id']
        j, i = doan.get(lid, {}).get('j'), luot.get(lid, {}).get('pre')
        if r.get('beta_deg') is not None and r.get('chuan_12b') is not None and j is not None and i is not None:
            can.setdefault(r['cho'], []).append((lid, j, i))
    theo = o.get('theo_cho') or {}
    out = {}
    for cho in sorted(set(theo) | set(can)):
        blk, dbs, ly = theo.get(cho), [], []
        if blk is None:
            ly.append('thieu_khoi_cho')
            blk = {}
        hong = set(blk.get('ca_hong') or [])
        muc = {}
        for m in blk.get('muc') or []:
            k = (m.get('luot_id'), m.get('dau'))
            if k in muc:
                ly.append(f'{k[0]}:{k[1]}:muc_trung')
            muc[k] = m
        for lid, j, i in can.get(cho, []) if theo.get(cho) is not None else []:
            ends = []
            for dau, src in (('chuan', fitw[j]), ('pre', hw[i])):
                m = muc.get((lid, dau))
                if m is None:
                    ly.append(f'{lid}:{dau}:thieu_muc')
                elif m.get('clip_sha256') != src['meta']['clip_sha'] or m.get('khung_neo_sha256') != src['khung_neo_sha256']:
                    ly.append(f'{lid}:{dau}:sai_nguon')
                    m = None
                ends.append((m, src))
            if any(m is None for m, _ in ends):
                continue
            keys = [set((((m.get('neo') or {}).get(t)) or {}).keys()) for m, _ in ends for t in ten]
            chung = set.intersection(*keys) - hong
            for n in sorted(set.union(*keys) - hong - chung):
                ly.append(f'{lid}:{n}:ca_thieu')
            for n in sorted(chung):
                if not all(_vec2(m['neo'][t][n]) for m, _ in ends for t in ten):
                    ly.append(f'{lid}:{n}:vec_sai')
                    continue
                da = []
                for m, src in ends:
                    P0 = np.asarray(src['Ph'], float)
                    F, H = np.asarray(src['F'], float), float(src['H'])
                    a1, _, _ = tu_the(G, h, P0 + np.array([m['neo'][t][n] for t in ten], float), F, H, ctx.C, ctx.a_n)
                    a0, _, _ = tu_the(G, h, P0, F, H, ctx.C, ctx.a_n)
                    da.append(wrap(a1 - a0))
                dbs.append(abs(wrap(da[1] - da[0])))
        ok = bool(dbs) and not ly and all(math.isfinite(x) for x in dbs)
        out[cho] = dict(max_abs_dbeta_rad=max(dbs) if dbs else None, n=len(dbs),
                        trang_thai='dat' if ok else 'chua_ket_luan',
                        ly_do=ly + ([] if dbs else ['khong_co_cap_luot_ca']), f15_sha256=f_sha)
    return out


def run_lan_lap(hc_path, cua_so_path, session_path, out_dir, a6_path, tam_id, truc_xe=None, xoay=None,
                u_beta_cam_deg=None, clip_dir=None, mau_clip=None, cfg=None, a6_mod=None, _records=None, f15=None):
    """v4 0, 3-9: held-out windows -> lan_lap.json (lan_lap_v1). Exit code 0 written / 2 error."""
    hc = doc_hieu_chuan(hc_path)
    khoa_ng = {k: v for k, v in hc['nguong'].items() if k in LL_DEFAULTS}
    doi = sorted(k for k, v in (cfg or {}).items() if k in LL_DEFAULTS and khoa_ng.get(k, LL_DEFAULTS[k]) != v)
    if doi:                                                    # LLC1-6: thresholds are locked at hieu_chuan
        raise ValueError(f'thresholds differ from the locked ones: {doi}')
    cfg = dict(LL_DEFAULTS, **khoa_ng)
    if u_beta_cam_deg is not None and not (math.isfinite(u_beta_cam_deg) and u_beta_cam_deg >= 0):
        raise ValueError('u_beta_cam must be finite and >= 0')
    ses = doc_json(session_path)
    ses_sha = cm.sha256(session_path)
    if ses_sha != hc['session']['sha256']:
        raise ValueError('session sha256 differs from hieu_chuan_neo.json')
    a6 = A6(a6_path, a6_mod)
    if a6.sha256 != hc['a6']['sha256']:
        raise ValueError('A6 file sha256 differs from hieu_chuan_neo.json')
    if tam_id != hc['a6'].get('tam_id'):
        raise ValueError(f'tam_id {tam_id} differs from hieu_chuan ({hc["a6"].get("tam_id")})')
    a6_tt = a6.ap_dung(tam_id)
    a6_ok = (a6_tt[0] if isinstance(a6_tt, (list, tuple)) else a6_tt) != 'chua_ket_luan'
    mau_path = mau_clip or hc['mau']['clip']                    # LLC1-6 (round 2): the clip actually used, every time
    if cm.sha256(mau_path) != hc['mau']['sha256']:
        raise ValueError(f'{"--mau-clip" if mau_clip else "template clip"} sha256 differs from the locked template clip')
    if cm.sha256(hc['mau']['session']) != hc['mau']['session_sha256']:
        raise ValueError('template session sha256 differs from the locked one')
    if cm.sha256(cua_so_path) != hc['cua_so_file']['sha256']:     # LLC1-6, LLC1-15 (round 2): one window file, one order
        raise ValueError('window file sha256 differs from the one locked at hieu_chuan (positions and roles are locked)')
    ctx = LLCtx(ses, hc['h_n_m'], cfg)
    rows = doc_cua_so(cua_so_path)
    kiem_f2(rows)
    keys = [i for i, w in enumerate(rows) if w[TEN['loai']] in LOAI_GIU]
    fit_frames = {(c['meta']['clip_sha'], k) for c in hc.get('cua_so', []) for k in c['khung_neo']['id']}
    for i in keys:
        if rows[i][TEN['session_sha']] != ses_sha:
            raise ValueError(f'window {i}: session sha256 differs (each end checked against its source)')
        if any((rows[i][TEN['clip_sha']], k) in fit_frames for k, _ in khung_cua(rows[i])):
            raise ValueError(f'window {i}: a held-out frame is a locked fit frame (D1)')
    cm._new_dir(out_dir)
    g_b = a6.g_b()
    g_b_so = g_b if a6_ok else None                            # LLC2-4: no other plate's tilt in the comparisons of 8
    eps_g = a6.eps_g()
    if isinstance(truc_xe, str):
        truc_xe = [truc_xe]
    tx, tx_src = doc_truc_xe(truc_xe)
    xo, xo_sha = doc_theo_cho(xoay, TEN['xoay']) if xoay else ({}, None)
    base = dict(schema=TEN['lan_lap'], version=LL_VERSION, hil=git_head(), tam_id=tam_id,
                neo_ten=hc.get('neo_ten'), h_neo_m=hc.get('h'), h_n_m=hc['h_n_m'],
                Ph_neo_cach='diem neo da nang (car_pose: p = pix2floor(uv, 0), P = p + kappa_n (C - p)), m, thu tu neo_ten',
                hieu_chuan=dict(file=os.path.abspath(hc_path), sha256=cm.sha256(hc_path),
                                khoa_sha256=hc['khoa']['noi_dung_sha256']),
                cua_so_v1_file=os.path.abspath(cua_so_path), cua_so_v1_sha256=cm.sha256(cua_so_path),
                session_file=os.path.abspath(session_path), session_sha256=ses_sha, fit_ids=ctx.fit_ids,
                a6=dict(file=os.path.abspath(a6_path), sha256=a6.sha256, tam_id=tam_id, ap_dung=a6_tt),
                truc_xe=tx_src, xoay_sha256=xo_sha, xoay_file=os.path.abspath(xoay) if xoay else None,
                phep_gop='trung_binh_go_vong', gioi_han=GIOI_HAN_V4, nguong=dict(cfg, trang_thai=LL_THRESH_STATUS))
    luot = {}
    for i in keys:
        luot.setdefault(rows[i][TEN['luot']], {})[rows[i][TEN['loai']]] = i
    if 'G' not in hc:
        # LLC2-3: no anchor shape -> still one record per run (J v6 5: the plate check runs for every run); no anchor
        # frame was measured, so the tag-anchor intersection is empty (mat_neo) and every comparison khong_xac_dinh
        hw = {i: _hw_rong(rows[i], i) for i in keys}
        env = dict(hw=hw, hc=hc, fitw=[], ctx=ctx, cfg=cfg, g_b=g_b, doan={}, tam_id=tam_id, a6_sha=a6.sha256,
                   hc_sha=base['hieu_chuan']['sha256'], rows=rows)
        so0 = so_sanh(None, None, ctx, g_b_so, dict(theta=NAN, c_B=NAN, s=NAN))
        out_l = []
        for lid, dd in sorted(luot.items()):
            r = _mot_luot(lid, dd, so0, env)
            for k in r['ly_do']:
                r['ly_do'][k].append('thieu_hinh_neo_hieu_chuan')
            out_l.append(r)
        base.update(trang_thai_buoi='chua_ket_luan', ly_do=hc.get('ly_do', []), luot=out_l)
        if f15 is not None:                                     # F15C-5: the source is checked on every path
            o15, f_sha = doc_f15(f15, dict(hc, _file_sha256=base['hieu_chuan']['sha256']), ctx)
            base['beta_cam_replay'] = {c: dict(max_abs_dbeta_rad=None, n=0, trang_thai='chua_ket_luan',
                                               ly_do=['thieu_hinh_neo_hieu_chuan'], f15_sha256=f_sha)
                                       for c in sorted(set(o15.get('theo_cho') or {}) | {r['cho'] for r in out_l
                                                                                        if r['cho'] in CHO})}
            base['f15'] = dict(file=os.path.abspath(f15), sha256=f_sha)
        _write_ll(out_dir, base)
        return 0
    G, h = np.array(hc['G'], float), np.array(hc['h'], float)
    fitw = hc['cua_so']
    obs_fit = [dict(Ph=np.array(c['Ph']), F=np.array(c['F']), H=c['H']) for c in fitw]
    r_max = _f(hc['r_max_m'])
    eps_hc = _f(hc['eps_neo_deg'])
    # ---- held-out windows
    if _records is None:
        neo = {d['ten']: [tuple(b) for b in d['o_px']] for d in hc['neo'] if d['ten'] in hc['neo_ten']}
        m = hc['mau']
        recs, flags, _ = _measure(rows, keys, session_path, (mau_clip or m['clip'], m['session'], m['idx']), ctx, neo,
                                  clip_dir)
    else:
        recs, flags = _records, {}
    hw = {}
    for i in keys:
        s = tom_cua_so(rows[i], recs[i], a6, a6_ok, ctx, cfg)
        d = dict(meta=s['meta'], t0=s['t0'], ok=s['ok'], ly_do=list(s['ly_do']), chung=None, ma=None,
                 khung_neo=dict(id=s['neo']['ids'], pts_s=s['neo']['pts']), khung_neo_sha256=sha_ids(s['neo']['ids']),
                 khung_chung=dict(id=s['chung']['ids'], pts_s=s['chung']['pts']),
                 khung_chung_sha256=sha_ids(s['chung']['ids']) if s['chung'].get('ok') else None, du_ok=False,
                 khung_giao_ma_neo=s['giao'], khung_giao_sha256=sha_ids(s['giao']['id']), mat_neo=s['mat_neo'],
                 thu_tu=i)
        if s['ok']:
            a_, o_, res = tu_the(G, h, s['neo']['Ph'], s['neo']['F'], s['neo']['H'], ctx.C, ctx.a_n)
            d.update(alpha=a_, o=o_, du_m=res.tolist(), Ph=s['neo']['Ph'], F=s['neo']['F'], H=s['neo']['H'],
                     du_ok=bool(np.all(res <= r_max)))
            d['ma'], d['chung'], ly_du = _tom_hai(s, G, h, ctx, r_max)
            d['ly_do'] += s['chung']['ly_do'] + ly_du
            if d['chung'] is None:                              # LLC3-3: the exported common set is the one used
                d['khung_chung'], d['khung_chung_sha256'] = dict(id=[], pts_s=[]), None
            if not d['du_ok']:
                d['ly_do'].append('du_neo_vuot_r_max')
        hw[i] = d
    # ---- spreads of the standard 12b group of each place (LLC1-4); thresholds of 8 within each place (LLC1-13)
    runs12 = sorted({c['meta']['luot'] for c in fitw if c['meta']['loai'] == LOAI_CHUAN})
    cho12 = {l12: next(c['meta']['cho'] for c in fitw if c['meta']['luot'] == l12 and c['meta']['loai'] == LOAI_CHUAN)
             for l12 in runs12}
    nhom = {l12: [_w_from_hc(c) for c in nhom_12b(fitw, l12, cho12[l12])] for l12 in runs12}
    trai = {l12: trai_nhom(v, ctx, g_b) if len(v) >= 2 else None for l12, v in nhom.items()}

    def nguong_cho(ks):
        def mx(key, floor):
            v = [trai[l][key] for l in runs12 if cho12[l] in ks and trai.get(l) is not None and math.isfinite(trai[l][key])]
            return max([floor] + v)
        return dict(theta=mx('theta', cfg['doi_theta_deg']), c_B=mx('c_B', cfg['doi_c_m']), s=mx('s', cfg['doi_s']))
    so = {}
    for lid, dd in luot.items():
        pre, post = hw.get(dd.get('pre')), hw.get(dd.get('post'))
        ca = {x['meta']['cho'] for x in (pre, post) if x is not None}
        so[lid] = so_sanh(_hai(pre) if pre and pre['ok'] and pre['du_ok'] else None,      # LLC2-4: anchor gate too
                          _hai(post) if post and post['ok'] and post['du_ok'] else None, ctx, g_b_so, nguong_cho(ca))
    # ---- u_beta (v4 4): held-out pairs, |wrap(dalpha - dpsi')|; exclusion only on independent evidence (LL4-2, LLC1-8)
    cap, loai_cap = [], []
    for lid, dd in sorted(luot.items()):
        pre, post = hw.get(dd.get('pre')), hw.get(dd.get('post'))
        if pre is None or post is None or not (pre['ok'] and post['ok']):
            loai_cap.append(dict(luot_id=lid, ly_do='thieu_khung'))
        elif not (pre['du_ok'] and post['du_ok']):
            loai_cap.append(dict(luot_id=lid, ly_do='du_neo_vuot_r_max'))
        elif pre['chung'] is None or post['chung'] is None:
            loai_cap.append(dict(luot_id=lid, ly_do='thieu_a6_hoac_khung_chung'))
        elif any(so[lid][k]['trang_thai'] == 'doi' for k in ('c_B', 's')):
            loai_cap.append(dict(luot_id=lid, ly_do='tam_doi_c_B_hoac_thang'))
        else:
            dd_ = abs(math.degrees(wrap(wrap(post['alpha'] - pre['alpha'])
                                        - wrap(post['chung']['psi'] - pre['chung']['psi']))))
            cap.append(dict(luot_id=lid, lech_deg=dd_))
    nhieu = hc['nhieu_cua_so_deg']
    eps_end = {}
    for i, x in hw.items():                                     # LLC1-11: eps_neo of each end with its own geometry
        if x['ok']:
            eps_end[i] = max(eps_hc, eps_goc(eps_neo_bang(G, h, x['F'], x['H'], ctx.C, ctx.a_n), r_max))
    eps_cap = max([2.0 * max(eps_end[luot[c['luot_id']]['pre']], eps_end[luot[c['luot_id']]['post']]) for c in cap],
                  default=2.0 * eps_hc)
    u_beta_goc = (max(c['lech_deg'] for c in cap) + (nhieu if nhieu is not None else NAN)) if cap else NAN
    u_beta = u_beta_goc + eps_cap
    u_beta_tt = []
    if len(cap) < cfg['min_cap']:
        u_beta_tt.append('thieu_giu_ngoai')
    if not (math.isfinite(u_beta) and u_beta <= cfg['u_beta_max_deg']):
        u_beta_tt.append('u_beta_vuot' if math.isfinite(u_beta) else 'u_beta_khong_tinh_duoc')
    # ---- domain (v4 4): fit windows and the used held-out ends (anchor gate passed, LLC1-8)
    a_ref = hc['mien_khop']['a_ref']

    def diem(a_, o_):
        return [*(rot2(-a_) @ (np.asarray(o_, float) - ctx.C)), a_ref + wrap(a_ - a_ref)]
    mien = [list(p) for p in hc['mien_khop']['dinh']]
    mien_nguon = [dict(vai_tro='khop', meta=c['meta']) for c in fitw]
    for c in cap:
        for k in ('pre', 'post'):
            x = hw[luot[c['luot_id']][k]]
            mien.append(diem(x['alpha'], x['o']))
            mien_nguon.append(dict(vai_tro='giu_ngoai', luot_id=c['luot_id'], loai=k))
    # ---- 6b held-out residual for the position (LL5-5, LLC1-3): every same-place pre / post pair with data
    giu_vt = []
    for lid, dd in sorted(luot.items()):
        pre, post = hw.get(dd.get('pre')), hw.get(dd.get('post'))
        if pre is None or post is None or pre['meta']['cho'] != post['meta']['cho']:
            continue
        if pre['ok'] and post['ok'] and pre['du_ok'] and post['du_ok'] and pre['chung'] and post['chung']:
            try:
                dc, _, _ = delta_c_B(pre['chung'], post['chung'], ctx, g_b)
                giu_vt.append(dict(luot_id=lid, lech_m=float(np.linalg.norm(dc))))
            except ValueError:
                pass
    du_giu = max((g['lech_m'] for g in giu_vt), default=NAN)
    # ---- runs
    doan = phan_doan(rows, luot, hw, fitw, so, ctx, g_b_so, nguong_cho)
    env = dict(hw=hw, hc=hc, fitw=fitw, obs_fit=obs_fit, doan=doan, G=G, h=h, ctx=ctx, cfg=cfg, g_b=g_b,
               eps_g=eps_g, u_beta=u_beta, u_beta_goc=u_beta_goc, eps_cap=eps_cap, eps_end=eps_end, eps_hc=eps_hc,
               u_beta_tt=u_beta_tt, mien=mien, diem=diem, r_max=r_max, nhom=nhom,
               trai=trai, du_giu=du_giu, giu_vt=giu_vt, tx=tx, xo=xo, u_beta_cam_deg=u_beta_cam_deg,
               nguong_cho=nguong_cho, tam_id=tam_id, a6_sha=a6.sha256, hc_sha=base['hieu_chuan']['sha256'], rows=rows)
    out_l = [_mot_luot(lid, dd, so[lid], env) for lid, dd in sorted(luot.items())]
    if f15 is not None:                                          # v4 13: second run with KIEMCHEO's F15 (b)
        base['beta_cam_replay'] = beta_cam_replay(f15, dict(hc, _file_sha256=base['hieu_chuan']['sha256']), out_l,
                                                  luot, hw, fitw, doan, G, h, ctx)
        base['f15'] = dict(file=os.path.abspath(f15), sha256=cm.sha256(f15))
    base.update(trang_thai_buoi=hc['trang_thai'], ly_do=hc.get('ly_do', []), u_beta_deg=u_beta,
                u_beta_cach='buoi: lech lon nhat + 2 eps_neo cap + nhieu; moi luot: eps_neo cap thay bang max(eps cap, '
                            'eps chuan + eps dich) (LLC2-5)',
                trang_thai_u_beta='dat' if not u_beta_tt else 'chua_ket_luan', ly_do_u_beta=u_beta_tt,
                cap_giu_ngoai=cap, cap_loai=loai_cap, eps_neo_deg=hc['eps_neo_deg'], eps_cap_deg=eps_cap, r_max_m=r_max,
                nhieu_cua_so_deg=nhieu, giu_ngoai_vi_tri=giu_vt, mien=dict(dinh=mien, nguon=mien_nguon),
                cua_so_loi=[dict(meta=x['meta'], ly_do=x['ly_do']) for x in hw.values() if not x['ok']],
                co_cua_so={str(k): v for k, v in flags.items()}, luot=out_l)
    _write_ll(out_dir, base)
    return 0


def _write_ll(out_dir, o):
    with open(os.path.join(out_dir, 'lan_lap.json'), 'w', encoding='utf-8') as f:
        json.dump(_jsonable(o), f, ensure_ascii=False, indent=1, allow_nan=False)


def _f(x):
    """A number read back from hieu_chuan_neo.json (null = not finite, LLC2-6)."""
    return NAN if x is None else float(x)


def _replay(G, h, hg, obs_fit, j_chuan, obs_d, ctx, cfg, beta, x_fit=None, idx_h=None):
    """v4 3 (LL4-5): largest change of beta when the shape and poses are refitted with every h_i at +-k sigma and
    +-k normalised units along each weak direction; inf when a refit does not converge or a height reaches the camera."""
    steps = []
    for i, s in enumerate(hg['sigma_h']):
        e = np.zeros(len(hg['sigma_h']))
        e[i] = s
        steps.append(e)
    steps += [np.asarray(v, float) for v in hg['yeu']]
    worst = 0.0
    h_tran = min([w['H'] for w in obs_fit] + [obs_d['H']])
    for e in steps:
        if not np.all(np.isfinite(e)):
            return math.inf
        for sg in (1.0, -1.0):
            hh = h.copy()
            hh[1:] += sg * cfg['hang_k'] * e
            if not np.all(hh < h_tran):                       # a height at or above the camera: no finite replay
                return math.inf
            x0 = None if x_fit is None else np.delete(np.asarray(x_fit, float), idx_h)
            fit = khop_hinh(obs_fit, ctx.h_n, ctx.C, ctx.a_n, fix_h=hh, x0=x0)
            if not fit['hoi_tu']:
                return math.inf
            a_s = fit['poses'][j_chuan][0]
            a_d, _, _ = tu_the(fit['G'], hh, obs_d['Ph'], obs_d['F'], obs_d['H'], ctx.C, ctx.a_n)
            worst = max(worst, abs(math.degrees(wrap(wrap(a_d - a_s) - beta))))
    return worst


def _dau(x, env, loai=None, cho=None):
    """F8 block of one end (LLC1-14)."""
    return dict(loai=loai or x['meta']['loai'], cho=cho or x['meta']['cho'], clip=x['meta']['clip'],
                clip_sha256=x['meta']['clip_sha'], session_sha256=x['meta']['session_sha'],
                khung_neo=x['khung_neo'], khung_neo_sha256=x['khung_neo_sha256'],
                khung_chung=x.get('khung_chung'), khung_chung_sha256=x.get('khung_chung_sha256'),
                khung_giao_ma_neo=x.get('khung_giao_ma_neo'), khung_giao_sha256=x.get('khung_giao_sha256'),
                mat_neo=x.get('mat_neo'), Ph_neo_m=x.get('Ph'), fit_ids=env['ctx'].fit_ids, a6_sha256=env['a6_sha'],
                hieu_chuan_sha256=env['hc_sha'], tam_id=env['tam_id'], phep_gop='trung_binh_go_vong',
                frames_sha256=x['meta'].get('frames_sha'), ly_do=list(x.get('ly_do', [])))


def _hw_rong(w, i):
    """Held-out end with nothing measured (no anchor shape, LLC2-3): empty frame sets, mat_neo."""
    pts = [p for _, p in khung_cua(w)]
    e = dict(id=[], pts_s=[])
    return dict(meta={k: w.get(TEN[k]) for k in ('luot', 'cho', 'loai', 'clip', 'clip_sha', 'session_sha', 'frames_sha')},
                t0=min(pts) if pts else NAN, ok=False, ly_do=['thieu_hinh_neo_hieu_chuan'], chung=None, ma=None,
                khung_neo=e, khung_neo_sha256=sha_ids([]), khung_chung=e, khung_chung_sha256=None, du_ok=False,
                khung_giao_ma_neo=e, khung_giao_sha256=sha_ids([]), mat_neo=True, thu_tu=i)


def _mot_luot(lid, d, so, env):
    hw, hc, fitw, ctx, cfg, g_b = env['hw'], env['hc'], env['fitw'], env['ctx'], env['cfg'], env['g_b']
    pre, post = hw.get(d.get('pre')), hw.get(d.get('post'))
    dich = pre
    cho = dich['meta']['cho'] if dich is not None else None
    r = dict(schema=TEN['lan_lap'], luot_id=lid, cho=cho, tam_id=env['tam_id'], beta_deg=None, u_beta_deg=None,
             delta_a_deg=None, q_m=None, q_B_m=None, delta_c_B_m=None, u_delta_c_B_m=None, delta_h_goc_m=None,
             tam_doi_trong_luot=tong_doi(so), so_sanh_trong_luot=so, cung_lan_lap=None, lan_lap_id=None,
             chuan_12b=None, chuan_12b_ung_vien=None, doi_so_chuan=None,
             kiem_tich_hop_deg=None, phi_deg=None, u_phi_truoc_le_deg=None,
             trang_thai=dict(beta='chua_ket_luan', delta_a='chua_ket_luan', q='chua_ket_luan', h='chua_ket_luan',
                             delta_c_B='chua_ket_luan', phi='chua_ket_luan'),
             ly_do=dict(beta=[], delta_a=['yaw_ma_khong_chi_so'], q=['chi_chan_doan_v4'], h=['chi_chan_doan_v4'],
                        delta_c_B=[], phi=[]),
             dau={}, khung_chung={}, khung_chung_sha256={}, tap_khung_sha256={})
    for nm, x in (('pre', pre), ('post', post)):
        if x is not None:
            r['dau'][nm] = _dau(x, env)
            r['khung_chung'][nm] = x.get('khung_chung')
            r['khung_chung_sha256'][nm] = x.get('khung_chung_sha256')
            r['tap_khung_sha256'][nm] = dict(neo=x['khung_neo_sha256'], chung=x.get('khung_chung_sha256'))
    if 'pre' in r['dau']:
        r['dau']['dich'] = r['dau']['pre']          # LLC2-7: key of the handed-over v1 kept (same block as 'pre')
    lneo = []                                                   # anchor-path gate shared by beta and delta_c_B
    if dich is None or not dich['ok']:
        lneo.append('cua_so_dich_loi')
        if env['doan'].get(lid, {}).get('ly_do'):               # CT4-3: e.g. lan_lap_khong_xac_dinh, record kept
            r['lan_lap_id_ly_do'] = env['doan'][lid]['ly_do']
            lneo.append(env['doan'][lid]['ly_do'])
        r['ly_do']['beta'] += lneo
        r['ly_do']['delta_c_B'] += lneo
        return r
    if hc['trang_thai'] != 'dat':
        lneo += hc.get('ly_do', []) or ['hieu_chuan_chua_dat']
    if hc.get('suy_bien') or not hc.get('hoi_tu', True):
        lneo.append('thieu_hang')
    if not dich['du_ok']:
        lneo.append('du_neo_vuot_r_max')
    dd = env['doan'].get(lid, {})
    r['lan_lap_id'], r['chuan_12b'] = dd.get('lan_lap_id'), dd.get('chuan_12b')
    r['chuan_12b_ung_vien'], r['doi_so_chuan'] = dd.get('ung_vien'), dd.get('doi_so_chuan')
    if dd.get('ly_do'):
        r['lan_lap_id_ly_do'] = dd['ly_do']
    j = dd.get('j')
    if j is None:                       # J v6 2: no 12b of this placement at this place (or order contradicted)
        lneo.append(dd.get('ly_do') or 'khong_co_cua_so_chuan')
        r['ly_do']['beta'] += lneo
        r['ly_do']['delta_c_B'] += lneo
        r['ly_do']['phi'] += lneo
        return r
    if dd.get('ly_do'):                 # LLC2-2: the standard of the place is not verified for this run (computed only)
        lneo.append(dd['ly_do'])
    cs = fitw[j]
    fr_sha = env['rows'][cs['thu_tu']].get(TEN['frames_sha'])  # CT1-2: frames.csv of the ref_12b row (F2, HIEUCHUAN)
    r['dau']['chuan'] = dict(loai=LOAI_CHUAN, cho=cho, clip=cs['meta']['clip'], clip_sha256=cs['meta']['clip_sha'],
                             session_sha256=cs['meta']['session_sha'], frames_sha256=fr_sha, khung_neo=cs['khung_neo'],
                             khung_neo_sha256=cs['khung_neo_sha256'], khung_chung=cs.get('khung_chung'),
                             khung_chung_sha256=cs.get('khung_chung_sha256'), Ph_neo_m=cs['Ph'], fit_ids=ctx.fit_ids,
                             a6_sha256=env['a6_sha'], hieu_chuan_sha256=env['hc_sha'], tam_id=env['tam_id'],
                             phep_gop='trung_binh_go_vong')
    beta = math.degrees(wrap(dich['alpha'] - cs['alpha']))
    # LLC2-5: the anchor part of u_beta for THIS standard-target pair (not only the pairs in cap)
    eps_s = max(env['eps_hc'], eps_goc(eps_neo_bang(env['G'], env['h'], np.array(cs['F']), cs['H'], ctx.C, ctx.a_n),
                                       env['r_max']))
    eps_luot = max(env['eps_cap'], eps_s + env['eps_end'].get(d.get('pre'), NAN))
    u_beta_l = env['u_beta_goc'] + eps_luot
    r['beta_deg'], r['u_beta_deg'], r['eps_neo_luot_deg'] = beta, u_beta_l, eps_luot
    r['du_dich_m'] = dich['du_m']
    if max(cs['du_m']) > env['r_max'] and 'du_neo_vuot_r_max' not in lneo:
        lneo.append('du_neo_vuot_r_max')
    ub_luot = ['u_beta_luot_vuot'] if math.isfinite(u_beta_l) and u_beta_l > cfg['u_beta_max_deg'] else []
    hg = dict(sigma_h=np.array(hc['sigma_h'], float), yeu=[np.array(v, float) for v in hc['huong_yeu']])
    if 'thieu_hang' not in lneo:
        rp = _replay(env['G'], env['h'], hg, env['obs_fit'], j, dict(Ph=dich['Ph'], F=dich['F'], H=dich['H']), ctx, cfg,
                     math.radians(beta), hc.get('x'), hc.get('idx_h'))
        r['replay_deg'] = rp
        if not rp <= cfg['replay_deg']:
            lneo.append('thieu_hang')
    if not trong_mien(env['diem'](dich['alpha'], dich['o']), env['mien']):
        lneo.append('ngoai_mien')
    lb = r['ly_do']['beta']
    lb += lneo + env['u_beta_tt'] + ub_luot
    if not lb:
        r['trang_thai']['beta'] = 'dat'
    lc = r['ly_do']['delta_c_B']
    lc += lneo                                                  # LLC1-2: not the yaw budget u_beta
    sw, dc = _w_from_hc(cs), dich['chung']
    if sw is None or dc is None:
        lc.append('thieu_khung_chung_hoac_a6')
        r['ly_do']['phi'].append('thieu_khung_chung_hoac_a6')
        r['ly_do']['q'].append('thieu_khung_chung_hoac_a6')
        if dich.get('ma') is not None and _w_from_hc(cs, 'ma') is not None:
            r['cung_lan_lap'] = None
        return r
    # ---- delta_a, integration check (v4 0, 5)
    da = wrap(dc['theta'] - sw['theta'])
    r['delta_a_deg'] = math.degrees(da)
    r['beta_chung_deg'] = math.degrees(wrap(dc['alpha'] - sw['alpha']))
    r['beta_chenh_tap_deg'] = wrapd(r['beta_chung_deg'] - beta)
    # LLC2-1, CT1-2: F7 of the 12b run that gave THIS standard (same place, thu_id_cap = chuan_12b, frames.csv of its
    # ref_12b row); a F7 of another 12b run at the place is not used
    t = env['tx'].get((cho, r['chuan_12b_ung_vien']))
    tx_ly = []
    if t is not None:
        tf = tx_nguon_sha(t, f"{r['chuan_12b_ung_vien']}:frames")
        if fr_sha is None or tf is None:
            tx_ly.append('thieu_frames_sha256_chuan')
        elif tf != fr_sha:
            tx_ly.append('truc_xe_khac_cua_so_chuan')
        elif tx_nguon_sha(t, TEN['tx_f6']) != env['hc_sha'] or tx_nguon_sha(t, TEN['tx_a6']) != env['a6_sha']:
            tx_ly.append('truc_xe_khac_f6_hoac_a6')            # LLC2-1 round 3: the F6 / A6 actually used here
        elif t.get(TEN['trang_thai']) not in TEN['tx_dat']:
            tx_ly.append('truc_xe_chua_dat')                    # LLC3-2: not locked -> its angles are not read
        if tx_ly:
            t = None
    if t is not None:
        a_k, phi_k = math.radians(float(t[TEN['tx_a']])), math.radians(float(t[TEN['tx_phi']]))
    else:
        a_k, phi_k = 0.0, wrap(sw['psi'])
    # v4 0 (LL4-9): algebraic identity on the common sets; a phi_hat not built from this psi_chuan_k, or a delta_a /
    # alpha from another frame set, breaks it
    r['kiem_tich_hop_deg'] = abs(math.degrees(wrap(phi_k + wrap(dc['alpha'] - sw['alpha']) - (dc['psi'] - a_k - da))))
    r['kiem_tich_hop_nguon'] = 'truc_xe_v1' if t is not None else 'noi_bo (khong co truc_xe_v1)'
    # ---- phi (v4 0, LL4-10, LLC1-1): every term finite and >= 0, F7 dat, identity finite and within tol
    lp = r['ly_do']['phi']
    if r['trang_thai']['beta'] != 'dat':
        lp.append('beta_chua_dat')
    if not r['kiem_tich_hop_deg'] <= cfg['tich_hop_deg']:
        lp.append('loi_ghep')
    u_ph = None
    if t is None:
        lp += tx_ly or ['thieu_truc_xe']
    else:
        if t.get(TEN['trang_thai']) not in TEN['tx_dat']:
            lp.append('truc_xe_chua_dat')
        u_ph = t.get(TEN['tx_u_phi'])
        if u_ph is None or not (math.isfinite(float(u_ph)) and float(u_ph) >= 0):
            lp.append('thieu_u_phi_hat')
            u_ph = None
        r['phi_deg'] = math.degrees(phi_k) + beta
    ucam = env['u_beta_cam_deg']
    if ucam is None:
        lp.append('thieu_u_beta_cam')
    if not math.isfinite(u_beta_l):
        lp.append('u_beta_khong_tinh_duoc')
    if u_ph is not None and ucam is not None and math.isfinite(u_beta_l):
        tong = float(u_ph) + u_beta_l + float(ucam)
        if math.isfinite(tong):                                 # LLC1-1 (round 2): the sum itself finite
            r['u_phi_truoc_le_deg'] = tong
        else:
            lp.append('u_phi_khong_huu_han')
    if not lp and r['u_phi_truoc_le_deg'] is not None:
        r['trang_thai']['phi'] = 'dat'
    # ---- q (v4 6, diagnostic; LLC1-17: both E ends, xoay_v1 dat, E ends not changed)
    x = env['xo'].get(cho)
    ed = [c for c in fitw if c['meta']['cho'] == cho and c['meta']['loai'] in ('E_dau', 'E_cuoi') and c['chung']]
    if x is None or x.get(TEN['xoay_r']) is None or x.get(TEN['trang_thai']) != 'dat':
        r['ly_do']['q'].append('thieu_xoay_v1_dat')
    elif sorted(c['meta']['loai'] for c in ed) != ['E_cuoi', 'E_dau']:
        r['ly_do']['q'].append('thieu_hai_dau_E')
    elif len({c['meta']['clip_sha'] for c in ed}) != 1:        # LLC1-17 (round 2): both E ends of one E clip
        r['ly_do']['q'].append('hai_dau_E_khac_clip')
    else:
        e1, e2 = (_w_from_hc(c) for c in ed)
        if tong_doi(so_sanh(e1, e2, ctx, g_b, env['nguong_cho']({cho}))) is not False:
            r['ly_do']['q'].append('E_tam_doi_hoac_chua_xac_dinh')
        else:
            rb = [rot2(-ew['alpha']) @ (nghich(np.array(x[TEN['xoay_r']], float), ctx.tag_h, ew['F'], ew['H'], ctx.C,
                                               ctx.a_t)[0] - ew['o']) for ew in (e1, e2)]
            qB = c_B_cua(dc, ctx, g_b) - np.mean(rb, axis=0)
            r['q_B_m'], r['q_m'] = qB.tolist(), (rot2(-dc['theta']) @ qB).tolist()
            r['q_B_chuan_k_m'] = (c_B_cua(sw, ctx, g_b) - np.mean(rb, axis=0)).tolist()   # F16 name, diagnostic
    r['theta_luot_rad'], r['theta_chuan_rad'] = dc['theta'], sw['theta']        # F16: tag yaw minus body (psi' - alpha)
    # ---- 4 corner heights (v4 7, diagnostic): g . r_i (tag frame, rotation-free) target minus standard; one g_b
    #      per plate for both ends (LL5-6) so the tilt part is 0 here; the mean change needs kappa (none in v4)
    g_d = g_s = g_b
    r['delta_h_goc_m'] = [float(np.dot(g_d, ci) - np.dot(g_s, ci)) for ci in cm.car_corners_body(ctx.side)]
    r['ly_do']['h'].append('khong_co_kappa')
    # ---- cung_lan_lap (v4 6, information only; LLC1-4: within the spread of the standard 12b group, no floor)
    tr = env['trai'].get(cs['meta']['luot'])
    if tr is not None and tr['n'] >= 2:
        cl = so_sanh(sw, dc, ctx, g_b, dict(theta=tr['theta'], c_B=tr['c_B'], s=tr['s']))
        r['cung_lan_lap'] = tong_doi(cl) is False if tong_doi(cl) is not None else None
        r['cung_lan_lap_chi_so'] = cl
    # ---- delta_c_B (v4 6b, LL5-1..6, LLC1-2, 3, 12)
    try:
        dcb, raw, dh = delta_c_B(sw, dc, ctx, g_b)
    except ValueError as e:
        lc.append(f'mien_so: {e}')
        return r
    r['delta_c_B_m'], r['delta_c_B_tho_m'], r['delta_h_uoc_m'] = dcb.tolist(), raw.tolist(), dh
    n12 = 0 if tr is None else tr['n']
    if n12 < cfg['min_12b']:
        lc.append('thieu_12b_cung_lan_lap')
    else:
        b_s = eps_neo_bang(env['G'], env['h'], np.array(cs['F']), cs['H'], ctx.C, ctx.a_n)
        b_d = eps_neo_bang(env['G'], env['h'], dich['F'], dich['H'], ctx.C, ctx.a_n)
        e_pos = (eps_vi_tri(b_s, env['r_max'], c_B_cua(sw, ctx, g_b))
                 + eps_vi_tri(b_d, env['r_max'], c_B_cua(dc, ctx, g_b, dh)))
        try:
            u, parts, u_dh = u_delta_c_B(sw, dc, ctx, g_b, env['eps_g'], tr['c_B'], tr['s'], e_pos,
                                         env['du_giu'] if math.isfinite(env['du_giu']) else 0.0)
            r['u_delta_c_B_m'], r['u_delta_c_B_phan'], r['u_delta_h_m'] = u, parts, u_dh
        except ValueError as e:
            lc.append(f'mien_so: {e}')
    if abs(dh) > cfg['dh_max_m']:
        lc.append('dh_vuot_ngoi_chot')
    if len(env['giu_vt']) < cfg['min_giu_vi_tri']:
        lc.append('thieu_giu_ngoai_vi_tri')
    if dc['n'] < cfg['min_khung']:
        lc.append('thieu_khung_chung')
    if not lc and r['u_delta_c_B_m'] is not None:
        r['trang_thai']['delta_c_B'] = 'dat'
    return r


def _w_bat_ky(c):
    """Summary of a fit window for the placement check: 'chung' (A6) when there, else 'ma'."""
    return _w_from_hc(c) or _w_from_hc(c, 'ma')


def phan_doan(rows, luot, hw, fitw, so, ctx, g_b, nguong_cho):
    """J v6 2, F8 (CODE 19/11; round 2 LLC2-2, LLC1-15): placement of each run by machine, on the ONE window file
    locked at hieu_chuan (positions = file order; CAM writes the windows in recording order).
    12b groups: every 12b run of the file is an item (a 12b whose windows all failed too: J v6 2 '12b of that placement
    not valid'); a 12b run after a car run opens a new group; consecutive 12b runs are one group (12b of the places in
    a row) unless two of them are at the SAME place and their comparison (8) is not khong_doi.
    A car run uses the latest group before it: the standard of its place in that group is used only when the run is
    VERIFIED to be on the same placement: its pre compared (8) with that standard window says khong_doi (only the pre
    links the placement to the target window, LLC3-1). doi -> 'tam_doi_so_chuan', khong_xac_dinh ->
    'lan_lap_khong_xac_dinh': this run has no standard and no lan_lap_id. A valid doi in the group (a run whose plate
    check says doi, or a run off its standard) breaks the group: every later run until the next 12b has no standard
    ('lan_lap_sau_doi'; the plate may have been re-seated back, LLC3-1). No group / no 12b of the place / its 12b not
    valid -> no standard. File order = recording order per clip, checked on the raw rows (LLC1-15).
    Returns {luot_id: dict(lan_lap_id, chuan_12b (verified), ung_vien, j (candidate), ly_do, doi_so_chuan)}."""
    j_of = {c['thu_tu']: j for j, c in enumerate(fitw)}
    g12 = {}
    for i, w in enumerate(rows):
        if w[TEN['loai']] in LOAI_12B:
            g12.setdefault(w[TEN['luot']], []).append(i)
    items = []
    for l12, idx in g12.items():
        ok = [j_of[i] for i in idx if i in j_of]
        ref_i = [i for i in idx if rows[i][TEN['loai']] == LOAI_CHUAN]
        items.append(dict(kind='12b', id=l12, pos=idx[0], win=idx, cho=rows[(ref_i or idx)[0]][TEN['cho']],
                          first=ok[0] if ok else None, last=ok[-1] if ok else None))
    for lid, d in luot.items():
        items.append(dict(kind='luot', id=lid, pos=min(d.values()), win=[d[k] for k in LOAI_GIU if k in d],
                          pre=d.get('pre'), post=d.get('post')))
    items.sort(key=lambda x: x['pos'])
    by_clip = {}                       # LLC1-15: PTS order inside each clip follows the RAW row order (every window)
    t0 = {}
    for i, w in enumerate(rows):
        pts = [p for _, p in khung_cua(w)]
        if pts:
            t0[i] = min(pts)
            by_clip.setdefault(w[TEN['clip_sha']], []).append((i, t0[i]))
    sai = any(any(b[1] <= a[1] for a, b in zip(v, v[1:])) for v in by_clip.values())
    sai = sai or any(d.get('pre') in t0 and d.get('post') in t0 and t0[d['post']] <= t0[d['pre']]       # same clip only
                     and rows[d['pre']][TEN['clip_sha']] == rows[d['post']][TEN['clip_sha']] for d in luot.values())
    if sai:
        return {lid: dict(lan_lap_id=None, chuan_12b=None, j=None, ly_do='thu_tu_mau_thuan') for lid in luot}
    ref = {(c['meta']['cho'], c['meta']['luot']): j for j, c in enumerate(fitw) if c['meta']['loai'] == LOAI_CHUAN}
    out, seg, seg12, prev, gay = {}, -1, {}, None, False
    for it in items:
        if it['kind'] == '12b':
            w_dau = _w_bat_ky(fitw[it['first']]) if it['first'] is not None else None
            new = prev is None or prev['kind'] != '12b'
            if not new and prev['cho'] == it['cho']:          # the same place twice in a row: the plate is checked
                new = tong_doi(so_sanh(prev['w'], w_dau, ctx, g_b, nguong_cho({it['cho']}))) is not False
            if new:
                seg += 1
                seg12[seg] = {}
                gay = False
            seg12[seg][it['cho']] = it['id'] if (it['cho'], it['id']) in ref else None
            w_cuoi = _w_bat_ky(fitw[it['last']]) if it['last'] is not None else None
            prev = dict(kind='12b', cho=it['cho'], w=w_cuoi if w_cuoi is not None else (None if new else prev['w']))
            continue
        prev = dict(kind='luot')
        pre = hw.get(it['pre']) if it['pre'] is not None else None
        post = hw.get(it['post']) if it['post'] is not None else None
        cho = pre['meta']['cho'] if pre is not None else None
        o = dict(lan_lap_id=None, chuan_12b=None, ung_vien=None, j=None, ly_do=None, doi_so_chuan=None)
        out[it['id']] = o
        if seg < 0:
            o['ly_do'] = 'khong_co_12b_cung_lan_lap'
            continue
        if cho not in seg12[seg]:
            o['ly_do'] = 'khong_co_cua_so_chuan'
            continue
        l12 = seg12[seg][cho]
        if l12 is None:
            o['ly_do'] = '12b_cung_lan_lap_khong_hop_le'
            continue
        o['ung_vien'], o['j'] = l12, ref[(cho, l12)]
        sw = _w_bat_ky(fitw[o['j']])
        wx = _hai(pre) if pre['ok'] and pre['du_ok'] else None
        d = tong_doi(so_sanh(sw, wx, ctx, g_b, nguong_cho({cho})))
        o['doi_so_chuan'] = d
        if gay:
            o['ly_do'] = 'lan_lap_sau_doi'
        elif d is False:
            o['lan_lap_id'], o['chuan_12b'] = f'L{seg}', l12
        else:
            o['ly_do'] = 'tam_doi_so_chuan' if d is True else 'lan_lap_khong_xac_dinh'
        if d is True or tong_doi(so[it['id']]) is True:         # LLC3-1: a valid doi breaks the group
            gay = True
    return out


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
    a2.add_argument('--loai', nargs='*', default=[], help='FILE=REASON of every clip shot and rejected (NT2-4)')
    a2.add_argument('--mau-idx', type=int, help='template frame index of N0 from khung_N0.json (NC2-2)')
    a2.add_argument('--out', required=True)
    a3 = sub.add_parser('kiem')
    a3.add_argument('--clip', required=True)
    a3.add_argument('--mau', required=True, help='clip N0 (template of the boxes; for N0 itself give the same file)')
    a3.add_argument('--rong', action='store_true', help='the clip is clip R')
    a3.add_argument('--session', required=True)
    a3.add_argument('--vung-than', required=True)
    a3.add_argument('--h-than', type=float)
    a3.add_argument('--stride', type=int, default=3)
    a3.add_argument('--mau-idx', type=int, help='template frame index of N0 from khung_N0.json (NC2-2)')
    a3.add_argument('--out', required=True)
    a4 = sub.add_parser('hieu_chuan', help='v4: anchors fitted on the 12b / E windows, locked (hieu_chuan_neo.json)')
    a4.add_argument('--cua-so', required=True, help='cua_so_v1 file (CAM, HIEUCHUAN)')
    a4.add_argument('--session', required=True)
    a4.add_argument('--neo', required=True, help='"A:u,v,w,h;B:u,v,w,h;oc1:u,v,w,h" boxes on the template frame')
    a4.add_argument('--mau', required=True, nargs=3, metavar=('CLIP', 'SESSION', 'IDX'), help='template frame of the boxes')
    a4.add_argument('--kiem-r', nargs='*', default=[], help='kiem.json of v7 "kiem --rong" (clip R) covering the boxes')
    a4.add_argument('--a6', required=True, help='c8b_a6_v2 file (C8B)')
    a4.add_argument('--tam-id', required=True, help='plate id of the session (CAM) for ap_dung of c8b_a6_v2')
    a4.add_argument('--h-neo', type=float, help='nominal anchor height h_n (default tag_h_m of the session)')
    a4.add_argument('--thu-muc-clip')
    a4.add_argument('--out', required=True)
    a5 = sub.add_parser('lan_lap', help='v4: pre / post of every run -> lan_lap.json (lan_lap_v1)')
    a5.add_argument('--hieu-chuan', required=True)
    a5.add_argument('--cua-so', required=True)
    a5.add_argument('--session', required=True)
    a5.add_argument('--a6', required=True)
    a5.add_argument('--truc-xe', nargs='*', default=[],
                    help='truc_xe_v1 files (HIEUCHUAN; one per place and 12b attempt, matched to the standard, LLC2-1)')
    a5.add_argument('--xoay', help='xoay_v1 (C8B)')
    a5.add_argument('--u-beta-cam-deg', type=float, help='u_beta_cam (C8B), before the margin')
    a5.add_argument('--tam-id', required=True, help='plate id of the session (CAM), as in hieu_chuan')
    a5.add_argument('--mau-clip', help='path of the template clip when moved')
    a5.add_argument('--f15', help='c7_meo_dau_chan_bang_v1 of KIEMCHEO (F15 b) -> block beta_cam_replay (v4 13)')
    a5.add_argument('--thu-muc-clip')
    a5.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'hieu_chuan':
            code = run_hieu_chuan(a.cua_so, a.session, a.neo, (a.mau[0], a.mau[1], int(a.mau[2])), a.out, a.a6,
                                  a.tam_id, a.kiem_r, a.h_neo, a.thu_muc_clip)
            o = doc_json(os.path.join(a.out, 'hieu_chuan_neo.json'))
            print(f'hieu_chuan: {o["trang_thai"]} {", ".join(o.get("ly_do", []))}')
        elif a.cmd == 'lan_lap':
            code = run_lan_lap(a.hieu_chuan, a.cua_so, a.session, a.out, a.a6, a.tam_id, a.truc_xe, a.xoay,
                               a.u_beta_cam_deg, a.thu_muc_clip, a.mau_clip, f15=a.f15)
            o = doc_json(os.path.join(a.out, 'lan_lap.json'))
            for x in o['luot']:
                print(f'{x["luot_id"]}: beta {x["trang_thai"]["beta"]} {x["beta_deg"]}, delta_c_B '
                      f'{x["trang_thai"]["delta_c_B"]}, lan_lap_id {x["lan_lap_id"]}')
        elif a.cmd == 'khung':
            print(json.dumps(run_khung(a.clip, a.session, a.out), ensure_ascii=False))
            code = 0
        elif a.cmd == 'kiem':
            if a.stride < 1:
                raise ValueError('--stride >= 1')
            ok, txt = run_kiem(a.clip, a.session, a.vung_than, a.mau, a.out, a.rong, a.h_than, a.stride,
                               mau_idx=a.mau_idx)
            print(txt)
            code = 0 if ok else 2                  # NC-7: a failed clip is a blocking result (contract of section 3)
        else:
            code = run_do(a.clip, a.session, a.vung_than, a.out, a.rong, a.h_than, a.stride, loai=a.loai,
                          mau_idx=a.mau_idx)
            with open(os.path.join(a.out, 'bao_cao.txt'), encoding='utf-8') as f:
                sys.stdout.write(f.read())
    except (OSError, ValueError, KeyError) as e:
        print(f'LOI: {e}', file=sys.stderr)
        code = 2
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
