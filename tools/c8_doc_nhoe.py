#!/usr/bin/env python3
"""
c8_doc_nhoe.py: read the blurred frames of the 3b B clips (tag on a spinning wheel) that the frame gate (2a) of
c7_measure rejects, with a corner-by-corner shape check from the image; frames.csv in the c7_measure track format
for c8b_3b.

Design: CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_CODE_DOCNHOE_v3.md (Codex Sol rounds 1-3; reviews
03_CODE/docs/review/2026-11-19_Tham_dinh_CODE_DOCNHOE_vong*.md). Written by CODE_DOCNHOE local_8bbf0c0b, 19/11/2026,
on feat/c8-doc-nhoe (base feat/c8-33-moc 33b444f). Imports c7_measure, c7_clip_probe, c8b_3b; changes none of them.

Why: on a spinning tag the residual of the 4-corner fit to a square WITH scale (res_sim_m) grows with the speed,
0.6 mm still -> about 2.2 mm at 1.3 m/s, and the gate (2a) of 1.5 mm, set for a still tag, rejects the fast part,
while centre and heading stay on the known circle. A residual alone cannot tell this from a wrong corner (Codex
DN1-01, DN2-01). The tag passes the same image places every turn, so the shape of the tag read from a fast frame is
compared with the shape read from the SLOW frames at the same phase (72 cells of 5 deg); the change of shape with
speed is a fixed pattern in the tag frame (same in both spin directions, design v3 1), learnt on the OTHER clip and
taken off; what is left (e_hinh, largest corner, mm) must stay within e_max. A corner wrong in the image itself
(occlusion, edge on the wrong line) shows in e_hinh (design v3 1.4: from 5 mm every frame caught at every speed; the
old gate on still frames catches 76-96 % at 5 mm).

Steps (design v3 2):
  cat    one pass over the video: a ROI_HALF crop around the tag centre of every frame where the track saw the tag
         (col_px, row_px), with its PTS (must equal t_s of frames.csv) -> <cache>.npz + .json (kept outside the repo)
  khoa   from both clips: corners re-detected on the crops (centre must equal the track); v_t (leave-one-out); image
         axis; phase reference shapes (old valid frames, v_t <= 0.3, same clip); speed pattern of each clip (Huber);
         e_hinh with the pattern of the OTHER clip; e_max = max(1.5 x p99.9 on the old valid frames <= 0.8 m/s, 1 mm);
         image injection (one corner of the tag moved d mm IN THE CROP, re-detected) for the new gate by speed band
         and for the old gate (2a) on slow frames; refuses (du_nhay false) when not sensitive enough -> khoa.json,
         written OUTSIDE the result tree, before any frame is taken back
  cong   needs khoa.json (hashes checked): a frame rejected by (2a) is taken back by (2a''') when re-detection,
         v_t, phase reference, pattern range, e_hinh <= e_max and res_sim_m <= 5 mm all hold; c7_measure.apply_gates
         twice (A: nothing taken = the track; B: taken frames res_sim 0): old frames keep run A, taken frames run B;
         frames.csv, doc_nhoe_rows.csv, doc_nhoe.json; khoa.json copied in; the clips joined with C1/ghep_3b.py
         (checked row by row); c8b_3b (CLI, unchanged) on the join
  kiem   accuracy report against the OLD 3b result (fixed circle, R0, phi_sl recomputed as c8b_3b does): signed dr,
         hp by clip x speed band x group, by phase quadrant and image rows, leave-one-out tangential residual, funnel
         from every PTS, the alpha fit sample of c8b_3b, extremes without clusters (sample sensitivity). Selects
         no frame.
All thresholds are [DE XUAT].

  python tools/c8_doc_nhoe.py cat  --video V.MOV --frames F/frames.csv --out CACHE_PREFIX
  python tools/c8_doc_nhoe.py khoa --c1 F1/frames.csv CACHE1 --c2 F2/frames.csv CACHE2 --session S/session.json \
                                   --cu C1/c8b_3b_A2_B.json --out C1_docnhoe_khoa.json
  python tools/c8_doc_nhoe.py cong --khoa C1_docnhoe_khoa.json --c1 ... --c2 ... --session ... --cu ... \
                                   --ghep C1/ghep_3b.py --frames-a C1/A2_ghep/frames.csv --out OUT_DIR
  python tools/c8_doc_nhoe.py kiem --out OUT_DIR --cu C1/c8b_3b_A2_B.json --cu-ghep C1/B_ghep/frames.csv [--rs CSV]
Last line: #ket_qua,ma=0 done / 2 refused or error.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7_clip_probe as cp                                               # noqa: E402,F401
import c7_measure as cm                                                  # noqa: E402

VERSION = 'c8_doc_nhoe_v3'
CAR_ID = cm.CAR_ID
ROI_HALF = 110              # crop half size, px (tag about 92 px, blur up to about 15 px)
MATCH_TOL_PX = 0.01         # the crop re-detection must give the centre of the track within this [DE XUAT]
VT_WIN_S = 0.1              # v_t: frames within +-0.1 s, the frame itself left out BEFORE unwrapping [DE XUAT]
VT_SIDE_MIN = 2             #      at least 2 frames on each side
VT_GAP_S = 0.05             #      no gap above 0.05 s between consecutive frames of the window
VT_STEP_MAX = 0.6           #      no phase step above 0.6 rad between consecutive frames of the window
VT_SIDE_TOL = 0.2           #      left and right linear rates agree within 20 % of their mean
SLOW_VT = 0.3               # phase reference: old valid frames up to 0.3 m/s
N_PHASE_REF = 72            # 5 deg cells
PHASE_MIN = 5               # frames per cell
HUBER_IT, HUBER_K = 10, 1.5
PAT_RANGE_Q = 99.0          # the pattern is used up to the 99th percentile of v_t of the clip it was learnt on
KHOA_VT_MAX = 0.8           # e_max from the old valid frames up to 0.8 m/s
E_MAX_K, E_Q = 1.5, 99.9
E_MAX_FLOOR_M, E_MAX_CEIL_M = 0.001, 0.003
RES_SIM_CAP_M = 0.005       # coarse cap on res_sim_m of a frame taken back
INJ_MM = (1.0, 2.0, 3.0, 5.0, 8.0)
INJ_BANDS = ((0.0, 0.3), (0.6, 0.9), (1.0, 1.4))
INJ_N = 25
INJ_SEED = 20261119
INJ_CATCH_AT_MM, INJ_CATCH_MIN = 5.0, 0.95
OLD_GATE_MM = (2.0, 3.0, 4.0, 5.0, 8.0)
QUANT = 5e-11               # half a unit of the 10th decimal of res_sim_m, scale (frames.csv of track)
BANDS_VT = (0.0, 0.3, 0.8, 1.0, 1.166, 1.283, 99.0)
N_PHASE = 4                 # report: phase quadrants
MIN_CELL = 30


class DnError(Exception):
    pass


# ── small helpers ────────────────────────────────────────────────────────────────────────────────

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', os.path.basename(__file__)],
                           capture_output=True, text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), version=VERSION,
                    c7_measure_sha256=sha256(cm.__file__))
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, version=VERSION, c7_measure_sha256=sha256(cm.__file__))


def low_priority():
    """Below-normal priority for this process and its children (does not slow the other sessions' tracks)."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:                                   # noqa: BLE001 - best effort
        pass


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_json_new(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        f.write(json.dumps(_clean(obj), indent=1, ensure_ascii=False, allow_nan=False))


def kasa(x, y):
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (cx, cy, c), *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    return float(cx), float(cy), float(math.sqrt(c + cx ** 2 + cy ** 2))


# ── clips ────────────────────────────────────────────────────────────────────────────────────────

def load_clip(path):
    """frames.csv of c7_measure track (FRAME_COLS, no 33-mark columns), as a dict of arrays + the rows."""
    with open(path, newline='', encoding='utf-8') as f:
        hdr = next(csv.reader(f))
    if tuple(hdr) != tuple(cm.FRAME_COLS):
        raise DnError(f'{path}: header is not the c7_measure track header {cm.FRAME_COLS}')
    rows = cm.read_frames(path)
    if len(rows) < 100:
        raise DnError(f'{path}: fewer than 100 frames')
    g = lambda c: np.array([r[c] for r in rows], float)             # noqa: E731
    t = g('t_s')
    if np.any(np.diff(t) <= 0):
        raise DnError(f'{path}: t_s does not increase')
    if [r['idx'] for r in rows] != list(range(len(rows))):
        raise DnError(f'{path}: idx is not 0..n-1')
    seen = np.array([bool(r['car_seen']) for r in rows])
    col, row = g('col_px'), g('row_px')
    if not np.all(np.isfinite(col[seen]) & np.isfinite(row[seen])):
        raise DnError(f'{path}: a frame with the tag seen has no col_px / row_px')
    return dict(path=path, sha256=sha256(path), rows=rows, t=t, seen=seen, col=col, row=row,
                res_sim=g('res_sim_m'), valid_old=np.array([bool(r['valid']) for r in rows]),
                why_old=[list(r['why']) for r in rows], x=g('tag_x'), y=g('tag_y'), psi=g('psi'))


def image_axis(C):
    """Image point of the wheel axis: circle through the tag centre pixels of the old valid frames."""
    m = C['valid_old']
    ax, ay, rp = kasa(C['col'][m], C['row'][m])
    return ax, ay, rp


def image_angle(C, axis):
    ang = np.full(len(C['t']), np.nan)
    s = C['seen']
    ang[s] = np.unwrap(np.arctan2(C['row'][s] - axis[1], C['col'][s] - axis[0]))
    return ang



def omega_loo(t, ang, seen, win=VT_WIN_S, side_min=VT_SIDE_MIN, gap=VT_GAP_S):
    """Rotation rate at every seen frame from the OTHER seen frames within +-win (design v3 2.2, DN2-04): the frame
    itself is left out BEFORE the phase is unwrapped (the raw angles ang may be wrapped or unwrapped: only differences
    modulo 2 pi are used); at least side_min frames on each side; no gap above `gap` and no phase step above
    VT_STEP_MAX between consecutive frames of the window; the linear rates of the left and right halves agree within
    VT_SIDE_TOL; quadratic fit for the rate. NaN otherwise."""
    idx = np.flatnonzero(seen)
    ts, an = t[idx], ang[idx]
    om = np.full(len(t), np.nan)
    for j, k in enumerate(idx):
        a = int(np.searchsorted(ts, ts[j] - win - 1e-9, 'left'))
        b = int(np.searchsorted(ts, ts[j] + win + 1e-9, 'right'))
        if j - a < side_min or b - j - 1 < side_min:
            continue
        sel = np.r_[np.arange(a, j), np.arange(j + 1, b)]
        tt = ts[sel] - ts[j]
        if np.max(np.diff(ts[a:b])) > gap + 1e-9:
            continue
        st = (np.diff(an[sel]) + math.pi) % (2 * math.pi) - math.pi        # wrapped steps, frame left out
        if np.max(np.abs(st)) > VT_STEP_MAX:
            continue
        aa = np.r_[0.0, np.cumsum(st)]
        nl = j - a
        wl = np.polyfit(tt[:nl], aa[:nl], 1)[0]
        wr = np.polyfit(tt[nl:], aa[nl:], 1)[0]
        m = 0.5 * (abs(wl) + abs(wr))
        if np.sign(wl) != np.sign(wr) or (m > 0 and abs(wl - wr) > VT_SIDE_TOL * m):
            continue
        om[k] = np.polyfit(tt, aa, 2)[1]
    return om


def side_px(q):
    return float(np.mean(np.linalg.norm(np.diff(np.vstack([q, q[:1]]), axis=0), axis=1)))


def quad_heading(q):
    v = (q[1] - q[0]) + (q[2] - q[3])
    return math.atan2(v[1], v[0])


def norm_shape(q):
    """The 4 corners in the tag's own frame (centre, heading of the quad), divided by the mean side."""
    c = q.mean(axis=0)
    h = quad_heading(q)
    R = np.array([[math.cos(-h), -math.sin(-h)], [math.sin(-h), math.cos(-h)]])
    return ((q - c) @ R.T) / side_px(q)


def sim_res(a, b):
    """b minus the least-squares similarity image of a (4 x 2, centred): the shape difference."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    hm = a.T @ b
    ang = math.atan2(hm[0, 1] - hm[1, 0], hm[0, 0] + hm[1, 1])
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    sc = float(np.sum(b * (a @ R.T)) / np.sum(a * a))
    return b - sc * (a @ R.T)


def phase_cell(ang):
    out = np.full(len(ang), -1, int)
    ok = np.isfinite(ang)
    out[ok] = np.floor(np.mod(ang[ok], 2 * math.pi) / (2 * math.pi) * N_PHASE_REF).astype(int) % N_PHASE_REF
    return out


def phase_refs(shapes, cell, use):
    """Median normalised shape per phase cell over the frames `use` (cells with fewer than PHASE_MIN: NaN)."""
    ref = np.full((N_PHASE_REF, 4, 2), np.nan)
    n = np.zeros(N_PHASE_REF, int)
    for c in range(N_PHASE_REF):
        m = use & (cell == c)
        n[c] = int(m.sum())
        if n[c] >= PHASE_MIN:
            ref[c] = np.median(shapes[m], axis=0)
    return ref, n


def shape_dev(shape, ref_cell, side_m):
    """Shape difference (8,) in m on the tag plane: similarity residual from the phase reference to the frame."""
    return (sim_res(ref_cell, shape) * side_m).ravel()


def huber_pattern(v, dev):
    """dev ~ v P1 + v^2 P2 (8 components each), Huber weights on the corner-vector norm of the residual."""
    A = np.column_stack([v, v ** 2])
    w = np.ones(len(v))
    B = None
    for _ in range(HUBER_IT):
        B, *_ = np.linalg.lstsq(A * w[:, None], dev * w[:, None], rcond=None)
        r = np.linalg.norm(dev - A @ B, axis=1)
        s = 1.4826 * float(np.median(r)) + 1e-12
        w = np.sqrt(np.minimum(1.0, HUBER_K * s / np.maximum(r, 1e-15)))
    return B


def e_hinh(dev, v, B):
    """Largest corner of what is left after the speed pattern: max_i |dev_i - (v P1 + v^2 P2)_i| (m)."""
    E = np.asarray(dev).reshape(-1, 8) - np.column_stack([np.atleast_1d(v), np.atleast_1d(v) ** 2]) @ B
    return np.max(np.linalg.norm(E.reshape(-1, 4, 2), axis=2), axis=1)


# ── crops (cat) ──────────────────────────────────────────────────────────────────────────────────

def cmd_cat(a):
    low_priority()
    C = load_clip(a.frames)
    want = set(np.flatnonzero(C['seen']).tolist())
    idx, x0s, y0s, pts, crops = [], [], [], [], []
    n = 0
    for k, tk, g in cm.iter_frames(a.video):
        n = k + 1
        if k >= len(C['t']):
            raise DnError(f'video has more frames than {a.frames}')
        if abs(tk - C['t'][k]) > 1e-6:
            raise DnError(f'frame {k}: PTS {tk:.6f} != t_s {C["t"][k]:.6f}')
        if k not in want:
            continue
        cx, cy = int(round(C['col'][k])), int(round(C['row'][k]))
        x0, y0 = cx - ROI_HALF, cy - ROI_HALF
        if x0 < 0 or y0 < 0 or x0 + 2 * ROI_HALF > g.shape[1] or y0 + 2 * ROI_HALF > g.shape[0]:
            continue                                     # crop off the picture: no image check (counted)
        idx.append(k); x0s.append(x0); y0s.append(y0); pts.append(tk)
        crops.append(np.ascontiguousarray(g[y0:y0 + 2 * ROI_HALF, x0:x0 + 2 * ROI_HALF]))
    if n != len(C['t']):
        raise DnError(f'video has {n} frames, {a.frames} {len(C["t"])}')
    if os.path.exists(a.out + '.npz') or os.path.exists(a.out + '.json'):
        raise DnError(f'{a.out}.npz/.json exists')
    np.savez_compressed(a.out + '.npz', idx=np.array(idx, int), x0=np.array(x0s, int), y0=np.array(y0s, int),
                        pts=np.array(pts), crops=np.array(crops, np.uint8))
    write_json_new(a.out + '.json', dict(video=os.path.basename(a.video), video_sha256=sha256(a.video),
                                         frames=a.frames, frames_sha256=C['sha256'], roi_half=ROI_HALF,
                                         n_frames=n, n_seen=len(want), n_crops=len(idx),
                                         npz_sha256=sha256(a.out + '.npz'), script=script_state()))
    print(f'cat: {len(idx)} crops of {len(want)} seen frames, {n} frames')


def load_cache(prefix, C):
    with open(prefix + '.json', encoding='utf-8') as f:
        meta = json.load(f)
    if meta['frames_sha256'] != C['sha256']:
        raise DnError(f'{prefix}: made from another frames.csv')
    if sha256(prefix + '.npz') != meta['npz_sha256']:
        raise DnError(f'{prefix}.npz: sha256 differs from its .json')
    z = np.load(prefix + '.npz')
    d = {int(k): j for j, k in enumerate(z['idx'])}
    return dict(meta=meta, pos=d, x0=z['x0'], y0=z['y0'], crops=z['crops'], pts=z['pts'])


# ── image: re-detection ─────────────────────────────────────────────────────

_DET = None


def detector():
    global _DET
    if _DET is None:
        _DET = cm.make_detector()
    return _DET


def redetect(cache, k, C):
    """Corners (4, 2) in full-image pixels re-detected in the crop, or (None, why)."""
    j = cache['pos'].get(k)
    if j is None:
        return None, 'khong_cat'
    crop = cache['crops'][j]
    d = cm.detect_refined(detector(), crop)
    if CAR_ID not in d:
        return None, 'khong_do_lai'
    q = d[CAR_ID] + [cache['x0'][j], cache['y0'][j]]
    c = q.mean(axis=0)
    if abs(c[0] - C['col'][k]) > MATCH_TOL_PX or abs(c[1] - C['row'][k]) > MATCH_TOL_PX:
        return None, 'lech_track'
    return q, ''



def read_cu(path):
    with open(path, encoding='utf-8') as f:
        J = json.load(f)
    B = J['B']
    return dict(path=path, sha256=sha256(path), centre=B['centre'], R0_m=float(B['R0_m']))


def load_session(path):
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    return s, sha256(path)


def analyse_clip(C, cache, cu, side_m):
    """Corners re-detected on every crop, image axis, phase, v_t (leave-one-out), normalised shapes, phase cells."""
    n = len(C['t'])
    Q = np.full((n, 4, 2), np.nan)
    why = [''] * n
    for k in np.flatnonzero(C['seen']):
        q, w = redetect(cache, k, C)
        if q is None:
            why[k] = w
        else:
            Q[k] = q
    qok = np.isfinite(Q.reshape(n, -1)).all(axis=1)
    axis = image_axis(C)
    ang = image_angle(C, axis)
    om = omega_loo(C['t'], ang, C['seen'])
    vt = np.abs(om) * cu['R0_m']
    shapes = np.full((n, 4, 2), np.nan)
    for k in np.flatnonzero(qok):
        shapes[k] = norm_shape(Q[k])
    return dict(Q=Q, qok=qok, why_det=why, axis=axis, ang=ang, om=om, vt=vt, shapes=shapes, cell=phase_cell(ang))


def devs(A, ref, side_m):
    n = len(A['vt'])
    dev = np.full((n, 8), np.nan)
    for k in np.flatnonzero(A['qok']):
        c = A['cell'][k]
        if c >= 0 and np.isfinite(ref[c]).all():
            dev[k] = shape_dev(A['shapes'][k], ref[c], side_m)
    return dev


def _rng_pick(rng, idx, n):
    idx = np.asarray(idx)
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def inject_new(C, cache, A, ref, B_other, e_max, side_m, rng):
    """Image injection for the new gate (design v3 1.4, 2.5): one corner of the tag moved d mm IN THE CROP (projective
    warp fixing the other three), corners re-detected, e_hinh against the phase reference and the other clip's
    pattern; caught when the tag is no longer found or e_hinh > e_max."""
    det = detector()
    out = []
    ok = A['qok'] & np.isfinite(A['vt']) & np.array([A['cell'][k] >= 0 and np.isfinite(ref[A['cell'][k]]).all()
                                                     for k in range(len(A['vt']))])
    for lo, hi in INJ_BANDS:
        for k in _rng_pick(rng, np.flatnonzero(ok & (A['vt'] >= lo) & (A['vt'] < hi)), INJ_N):
            j = cache['pos'][k]
            crop = cache['crops'][j]
            off = np.array([cache['x0'][j], cache['y0'][j]], float)
            q = A['Q'][k]
            ppm = side_px(q) / side_m
            corner = int(rng.integers(4))
            for mm in INJ_MM:
                th = float(rng.uniform(0.0, 2 * math.pi))
                q2 = q.copy()
                q2[corner] += mm * 1e-3 * ppm * np.array([math.cos(th), math.sin(th)])
                H = cv2.getPerspectiveTransform((q - off).astype(np.float32), (q2 - off).astype(np.float32))
                c2 = cv2.warpPerspective(crop, H, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
                d = cm.detect_refined(det, c2)
                if CAR_ID not in d:
                    out.append(dict(idx=int(k), dai=[lo, hi], mm=mm, goc=corner, e_m=None, bat=True))
                    continue
                dv = shape_dev(norm_shape(d[CAR_ID] + off), ref[A['cell'][k]], side_m)
                e = float(e_hinh(dv, A['vt'][k], B_other)[0])
                out.append(dict(idx=int(k), dai=[lo, hi], mm=mm, goc=corner, e_m=e, bat=bool(e > e_max)))
    return out


def inject_old(C, cache, A, sess, side_m, gate_m, rng):
    """The same image injection on slow old valid frames for the OLD gate (2a): res_sim_m of car_pose with the
    session camera (the track refits the camera per frame; the static camera gives the same residual within a few
    um on these clips, design v3 1.4) above the gate = caught."""
    det = detector()
    cam = cm.cam_from_dict(sess['camera'])
    out = []
    ok = A['qok'] & C['valid_old'] & np.isfinite(A['vt']) & (A['vt'] < INJ_BANDS[0][1])
    for k in _rng_pick(rng, np.flatnonzero(ok), INJ_N):
        j = cache['pos'][k]
        crop = cache['crops'][j]
        off = np.array([cache['x0'][j], cache['y0'][j]], float)
        q = A['Q'][k]
        ppm = side_px(q) / side_m
        corner = int(rng.integers(4))
        base = cm.car_pose(cam, q, side_m, sess['kappa'], sess['plumb_xy_m'])['res_sim_m']
        for mm in OLD_GATE_MM:
            th = float(rng.uniform(0.0, 2 * math.pi))
            q2 = q.copy()
            q2[corner] += mm * 1e-3 * ppm * np.array([math.cos(th), math.sin(th)])
            H = cv2.getPerspectiveTransform((q - off).astype(np.float32), (q2 - off).astype(np.float32))
            c2 = cv2.warpPerspective(crop, H, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            d = cm.detect_refined(det, c2)
            if CAR_ID not in d:
                out.append(dict(idx=int(k), mm=mm, res_sim_m=None, bat=True, res_sim_goc_m=base))
                continue
            rs = cm.car_pose(cam, d[CAR_ID] + off, side_m, sess['kappa'], sess['plumb_xy_m'])['res_sim_m']
            out.append(dict(idx=int(k), mm=mm, res_sim_m=rs, bat=bool(rs > gate_m), res_sim_goc_m=base,
                            res_sim_track_m=float(C['res_sim'][k])))
    return out


def _rate(rows, key, val, extra=None):
    sel = [r['bat'] for r in rows if r[key] == val and (extra is None or extra(r))]
    return float(np.mean(sel)) if sel else None


def cmd_khoa(a):
    low_priority()
    cu = read_cu(a.cu)
    sess, ssha = load_session(a.session)
    side = float(sess['car_side_m'])
    G = cm.DEFAULT_GATES['car_res_sim_m']
    clips, An, caches = {}, {}, {}
    for name, (fp, cache_p) in (('c1', a.c1), ('c2', a.c2)):
        C = load_clip(fp)
        cache = load_cache(cache_p, C)
        clips[name], caches[name] = C, cache
        An[name] = analyse_clip(C, cache, cu, side)
    refs, pats, rng_v, D = {}, {}, {}, {}
    for name, C in clips.items():
        A = An[name]
        use = A['qok'] & C['valid_old'] & np.isfinite(A['vt']) & (A['vt'] <= SLOW_VT)
        refs[name] = phase_refs(A['shapes'], A['cell'], use)
        D[name] = devs(A, refs[name][0], side)
        ok = np.isfinite(D[name]).all(axis=1) & np.isfinite(A['vt'])
        if ok.sum() < 1000:
            raise DnError(f'{name}: {int(ok.sum())} frames with a shape and v_t, need 1000')
        pats[name] = huber_pattern(A['vt'][ok], D[name][ok])
        rng_v[name] = float(np.percentile(A['vt'][ok], PAT_RANGE_Q))
    out = dict(version=VERSION, session=a.session, session_sha256=ssha, cu=a.cu, cu_sha256=cu['sha256'],
               gate_2a_m=G, hang_so=dict(ROI_HALF=ROI_HALF, MATCH_TOL_PX=MATCH_TOL_PX, VT_WIN_S=VT_WIN_S,
                                         VT_SIDE_MIN=VT_SIDE_MIN, VT_GAP_S=VT_GAP_S, VT_STEP_MAX=VT_STEP_MAX,
                                         VT_SIDE_TOL=VT_SIDE_TOL, SLOW_VT=SLOW_VT, N_PHASE_REF=N_PHASE_REF,
                                         PHASE_MIN=PHASE_MIN, HUBER=[HUBER_IT, HUBER_K], PAT_RANGE_Q=PAT_RANGE_Q,
                                         KHOA_VT_MAX=KHOA_VT_MAX, E_MAX=[E_MAX_K, E_Q, E_MAX_FLOOR_M, E_MAX_CEIL_M],
                                         RES_SIM_CAP_M=RES_SIM_CAP_M, INJ=[INJ_MM, INJ_BANDS, INJ_N, INJ_SEED],
                                         INJ_CATCH=[INJ_CATCH_AT_MM, INJ_CATCH_MIN], OLD_GATE_MM=OLD_GATE_MM),
               clips={})
    rng = np.random.default_rng(INJ_SEED)
    du = True
    for name, C in clips.items():
        other = 'c2' if name == 'c1' else 'c1'
        A = An[name]
        ok = np.isfinite(D[name]).all(axis=1) & np.isfinite(A['vt'])
        e = np.full(len(A['vt']), np.nan)
        e[ok] = e_hinh(D[name][ok], A['vt'][ok], pats[other])
        ref_set = ok & C['valid_old'] & (A['vt'] <= KHOA_VT_MAX)
        if ref_set.sum() < 1000:
            raise DnError(f'{name}: {int(ref_set.sum())} old valid frames for e_max, need 1000')
        e_max = max(E_MAX_K * float(np.percentile(e[ref_set], E_Q)), E_MAX_FLOOR_M)
        e_max = math.ceil(e_max * 1e5 - 1e-9) / 1e5
        inj = inject_new(C, caches[name], A, refs[name][0], pats[other], e_max, side, rng)
        inj_old = inject_old(C, caches[name], A, sess, side, G, rng)
        tab = {f'{lo}-{hi}': {str(mm): _rate(inj, 'mm', mm, lambda r, lo=lo: r['dai'][0] == lo) for mm in INJ_MM}
               for lo, hi in INJ_BANDS}
        tab_old = {str(mm): _rate(inj_old, 'mm', mm) for mm in OLD_GATE_MM}
        fast = f'{INJ_BANDS[-1][0]}-{INJ_BANDS[-1][1]}'
        why = []
        if e_max > E_MAX_CEIL_M:
            why.append(f'e_max {e_max * 1e3:.2f} mm > {E_MAX_CEIL_M * 1e3:.0f} mm')
        for b, r in tab.items():
            if r[str(INJ_CATCH_AT_MM)] is None or r[str(INJ_CATCH_AT_MM)] < INJ_CATCH_MIN:
                why.append(f'band {b}: caught at {INJ_CATCH_AT_MM} mm {r[str(INJ_CATCH_AT_MM)]}')
        if (tab[fast]['3.0'] or 0.0) < (tab_old['3.0'] or 0.0):
            why.append(f'fast band caught at 3 mm {tab[fast]["3.0"]} < old gate on slow frames {tab_old["3.0"]}')
        du = du and not why
        cm_meta = caches[name]['meta']
        out['clips'][name] = dict(
            frames=C['path'], frames_sha256=C['sha256'], cache_npz_sha256=cm_meta['npz_sha256'],
            video_sha256=cm_meta['video_sha256'], axis_px=list(A['axis']),
            n_do_lai=int(A['qok'].sum()), do_lai_loi=dict(Counter(w for w in A['why_det'] if w)),
            chuan_pha=refs[name][0].tolist(), chuan_pha_n=refs[name][1].tolist(),
            kieu_lech=pats[name].tolist(), kieu_lech_mien_vt=rng_v[name], kieu_lech_dung_cho=other,
            e_max_m=e_max, e_ref_n=int(ref_set.sum()), e_ref_p50_m=float(np.median(e[ref_set])),
            e_ref_p999_m=float(np.percentile(e[ref_set], E_Q)),
            bom_loi_moi=dict(ti_le_bat=tab, chi_tiet=inj), bom_loi_cong_cu=dict(ti_le_bat=tab_old, chi_tiet=inj_old),
            du_nhay=not why, ly_do=why)
        print(f'khoa {name}: e_max {e_max * 1e3:.2f} mm; bat (moi) {tab}; bat (cu, cham) {tab_old}; '
              f'du_nhay {not why} {why}')
    out['du_nhay'] = du
    out['script'] = script_state()
    write_json_new(a.out, out)
    if not du:
        print('khoa: kiem hinh KHONG du nhay; cong se tu choi')


# ── cong ─────────────────────────────────────────────────────────────────────────────────────────

def regate_with(rows, take, rms_h, gates):
    """c7_measure.apply_gates on a copy where the frames taken back carry res_sim_m 0 (they pass (2a) and join the
    neighbours of (2b)); the others keep their residual. valid, why, scale_ref and the speeds are copied back to a
    copy of the rows; res_sim_m stays the measured one."""
    eff = copy.deepcopy(rows)
    for k, r in enumerate(eff):
        if take[k]:
            r['res_sim_m'] = 0.0
    cm.apply_gates(eff, rms_h, gates, quant=QUANT)
    out = copy.deepcopy(rows)
    for r, e in zip(out, eff):
        for c in ('valid', 'why', 'scale_ref', 'vx', 'vy', 'wz', 'speed'):
            r[c] = e[c]
    return out


def combine(rows, take, rms_h, gates):
    """Design v3 2.4 (DN2-06): run A takes nothing (= the track), run B takes the frames; a taken frame gets run B,
    every other frame run A. Speeds: run B for every frame (the valid set of the output), as apply_gates would give
    them on the final valid set when no old frame changes. Returns (rows, run A, run B, old frames run B flips)."""
    ra = regate_with(rows, np.zeros(len(rows), bool), rms_h, gates)
    rb = regate_with(rows, take, rms_h, gates)
    out = copy.deepcopy(rows)
    flips = []
    for k, r in enumerate(out):
        src = rb[k] if take[k] else ra[k]
        for c in ('valid', 'why', 'scale_ref'):
            r[c] = copy.deepcopy(src[c])
        if not take[k] and ra[k]['valid'] != rb[k]['valid']:
            flips.append(k)
    # speeds on the final valid set (c7_measure.sg_derivative, as apply_gates does)
    for r in out:
        for c in ('vx', 'vy', 'wz', 'speed'):
            r[c] = float('nan')
    vi = [k for k, r in enumerate(out) if r['valid']]
    if len(vi) >= 5:
        tv = np.array([out[k]['t_s'] for k in vi])
        vx = cm.sg_derivative(tv, np.array([out[k]['x'] for k in vi]))
        vy = cm.sg_derivative(tv, np.array([out[k]['y'] for k in vi]))
        wz = cm.sg_derivative(tv, np.unwrap(np.array([out[k]['psi'] for k in vi])))
        for j, k in enumerate(vi):
            out[k].update(vx=float(vx[j]), vy=float(vy[j]), wz=float(wz[j]),
                          speed=float(math.hypot(vx[j], vy[j])) if np.isfinite(vx[j]) else float('nan'))
    return out, ra, rb, flips


def decide(C, A, K, Kother, side_m, gate_m):
    """Per frame: candidate (fails the old (2a)), reason it is not taken, e_hinh; returns (take, info)."""
    n = len(C['t'])
    ref = np.array(K['chuan_pha'])
    B = np.array(Kother['kieu_lech'])
    vmax = Kother['kieu_lech_mien_vt']
    take = np.zeros(n, bool)
    info = [dict() for _ in range(n)]
    for k in np.flatnonzero(C['seen']):
        d = info[k]
        d['v_t'] = float(A['vt'][k]) if np.isfinite(A['vt'][k]) else None
        d['o_pha'] = int(A['cell'][k])
        cand = not (C['res_sim'][k] <= gate_m)
        d['cong'] = 'ung_vien' if cand else 'cu'
        e = None
        if A['qok'][k] and np.isfinite(A['vt'][k]) and A['cell'][k] >= 0 and np.isfinite(ref[A['cell'][k]]).all():
            e = float(e_hinh(shape_dev(A['shapes'][k], ref[A['cell'][k]], side_m), A['vt'][k], B)[0])
        d['e_hinh_m'] = e
        if not cand:
            continue
        if not A['qok'][k]:
            d['ly_do'] = A['why_det'][k] or 'khong_do_lai'
        elif not np.isfinite(A['vt'][k]):
            d['ly_do'] = 'khong_vt'
        elif e is None:
            d['ly_do'] = 'khong_chuan_pha'
        elif A['vt'][k] > vmax:
            d['ly_do'] = 'ngoai_mien_kieu_lech'
        elif not e <= K['e_max_m']:
            d['ly_do'] = 'e_hinh_vuot'
        elif not C['res_sim'][k] <= RES_SIM_CAP_M:
            d['ly_do'] = 'res_sim_vuot_tran'
        else:
            d['ly_do'] = ''
            take[k] = True
    return take, info


def rows_csv(path, C, out, info, take, ra, rb, e_max):
    cols = ('clip_idx', 't_s', 'v_t', 'o_pha', 'cong', 'ly_do', 'e_hinh_m', 'e_max_m', 'nhan_lai', 'valid_cu',
            'why_cu', 'valid_A', 'valid_B', 'valid_moi', 'why_moi')
    with open(path, 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(len(out)):
            if not C['seen'][k]:
                continue
            d = info[k]
            vals = [k, f'{C["t"][k]:.6f}', d.get('v_t'), d.get('o_pha'), d.get('cong'), d.get('ly_do'),
                    d.get('e_hinh_m'), e_max, int(take[k]), int(C['valid_old'][k]), '|'.join(C['why_old'][k]),
                    int(ra[k]['valid']), int(rb[k]['valid']), int(out[k]['valid']), '|'.join(out[k]['why'])]
            w.writerow(['' if v is None else (f'{v:.9g}' if isinstance(v, float) else v) for v in vals])


def cmd_cong(a):
    low_priority()
    with open(a.khoa, encoding='utf-8') as f:
        KH = json.load(f)
    if KH.get('version') != VERSION:
        raise DnError(f'{a.khoa}: version {KH.get("version")} != {VERSION}')
    if not KH.get('du_nhay'):
        raise DnError(f'{a.khoa}: the shape check is not sensitive enough (du_nhay false): refused')
    cu = read_cu(a.cu)
    if KH['cu_sha256'] != cu['sha256']:
        raise DnError('khoa.json was made with another old 3b result')
    sess, ssha = load_session(a.session)
    if KH['session_sha256'] != ssha:
        raise DnError('khoa.json was made with another session')
    side = float(sess['car_side_m'])
    gates = dict(cm.DEFAULT_GATES)
    G = gates['car_res_sim_m']
    if abs(G - KH['gate_2a_m']) > 1e-12:
        raise DnError('gate (2a) differs from khoa.json')
    clips = {}
    for name, (fp, cache_p) in (('c1', a.c1), ('c2', a.c2)):
        C = load_clip(fp)
        K = KH['clips'][name]
        if C['sha256'] != K['frames_sha256']:
            raise DnError(f'{fp}: not the frames.csv of khoa.json {name}')
        cache = load_cache(cache_p, C)
        if cache['meta']['npz_sha256'] != K['cache_npz_sha256']:
            raise DnError(f'{cache_p}: not the crops of khoa.json {name}')
        clips[name] = (C, cache)
    os.makedirs(a.out, exist_ok=False)
    shutil.copyfile(a.khoa, os.path.join(a.out, 'khoa.json'))
    summary = dict(version=VERSION, khoa=a.khoa, khoa_sha256=sha256(a.khoa), session=a.session, session_sha256=ssha,
                   cu=a.cu, cu_sha256=cu['sha256'], gates=gates, quant=QUANT, clips={})
    outs = {}
    for name, (C, cache) in clips.items():
        other = 'c2' if name == 'c1' else 'c1'
        K, Ko = KH['clips'][name], KH['clips'][other]
        A = analyse_clip(C, cache, cu, side)
        if np.max(np.abs(np.array(A['axis']) - np.array(K['axis_px']))) > 1e-6:
            raise DnError(f'{name}: image axis differs from khoa.json')
        take, info = decide(C, A, K, Ko, side, G)
        out, ra, rb, flips = combine(C['rows'], take, sess['rms_px_H'], gates)
        same_a = sum(1 for x, y in zip(ra, C['rows']) if x['valid'] != y['valid'] or x['why'] != y['why'])
        d = os.path.join(a.out, f'B_{name}_3b')
        os.makedirs(d, exist_ok=False)
        fpath = os.path.join(d, 'frames.csv')
        cm.write_frames(fpath, out)
        rows_csv(os.path.join(d, 'doc_nhoe_rows.csv'), C, out, info, take, ra, rb, K['e_max_m'])
        vnew = np.array([r['valid'] for r in out])
        cand = [info[k] for k in range(len(out)) if info[k].get('cong') == 'ung_vien']
        js = dict(clip=name, frames_in=C['path'], frames_in_sha256=C['sha256'], frames_out_sha256=sha256(fpath),
                  kieu_lech_tu=other, e_max_m=K['e_max_m'], n_frames=len(out), n_seen=int(C['seen'].sum()),
                  n_valid_cu=int(C['valid_old'].sum()), n_valid_moi=int(vnew.sum()),
                  chuyen=dict(Counter(f'{int(C["valid_old"][k])}->{int(vnew[k])}' for k in range(len(out)))),
                  why_moi=dict(Counter('|'.join(r['why']) for r in out)),
                  ung_vien=len(cand), ly_do_khong_nhan=dict(Counter(x.get('ly_do') for x in cand if x.get('ly_do'))),
                  nhan_lai=int(take.sum()), nhan_lai_hop_le=int((take & vnew).sum()),
                  lan_A_khac_track=same_a, khung_cu_lan_B_doi=len(flips), khung_cu_lan_B_doi_idx=flips[:200])
        write_json_new(os.path.join(d, 'doc_nhoe.json'), dict(js, script=script_state()))
        summary['clips'][name] = js
        outs[name] = fpath
        print(f'cong {name}: hop le {js["n_valid_cu"]} -> {js["n_valid_moi"]}; nhan lai {js["nhan_lai"]} '
              f'({js["nhan_lai_hop_le"]} hop le); khong nhan {js["ly_do_khong_nhan"]}; lan A khac track {same_a}; '
              f'khung cu lan B doi {len(flips)}')
        if same_a:
            raise DnError(f'{name}: run A differs from the track on {same_a} frames')
    gdir = os.path.join(a.out, 'B_ghep')
    subprocess.run([sys.executable, '-I', a.ghep, '--c1', outs['c1'], '--c2', outs['c2'], '--out', gdir,
                    '--ten', 'B doc nhoe (c8_doc_nhoe v3) IMG_3601 roi IMG_3602'], check=True)
    _check_join(outs['c1'], outs['c2'], os.path.join(gdir, 'frames.csv'))
    tb_out = os.path.join(a.out, 'c8b_3b_A2_B.json')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'c8b_3b.py'), '--frames-a', a.frames_a, '--frames-b',
                        os.path.join(gdir, 'frames.csv'), '--out', tb_out], capture_output=True, text=True)
    with open(os.path.join(a.out, 'c8b_3b_stdout.txt'), 'x', encoding='utf-8') as f:
        f.write(p.stdout + p.stderr)
    summary['c8b_3b'] = dict(ma=p.returncode, cuoi=p.stdout.strip().splitlines()[-1] if p.stdout.strip() else '')
    summary['script'] = script_state()
    write_json_new(os.path.join(a.out, 'doc_nhoe_tong.json'), summary)
    print(f'c8b_3b: {summary["c8b_3b"]["cuoi"]}')


def _check_join(f1, f2, fg):
    """Every row of the join equals its source row (t_s of clip 2 shifted, idx of clip 2 offset; other cells equal)."""
    def rd(p):
        with open(p, newline='', encoding='utf-8') as f:
            return list(csv.reader(f))
    a, b, g = rd(f1), rd(f2), rd(fg)
    if g[0] != a[0] or a[0] != b[0] or len(g) != len(a) + len(b) - 1:
        raise DnError('join: header or row count differs')
    ii, it = a[0].index('idx'), a[0].index('t_s')
    n1 = len(a) - 1
    off = None
    for j, src in enumerate(a[1:] + b[1:]):
        out = g[1 + j]
        for c, (u, v) in enumerate(zip(src, out)):
            if c == ii:
                if int(v) != int(u) + (n1 if j >= n1 else 0):
                    raise DnError(f'join row {j}: idx')
            elif c == it:
                dt = float(v) - float(u)
                if j < n1 and abs(dt) > 1e-9:
                    raise DnError(f'join row {j}: t_s')
                if j >= n1:
                    off = dt if off is None else off
                    if abs(dt - off) > 2e-6:
                        raise DnError(f'join row {j}: t_s shift not constant')
            elif u != v:
                raise DnError(f'join row {j} column {a[0][c]}: {u!r} != {v!r}')




# ── kiem (report) ────────────────────────────────────────────────────────────────────────────────

def phi_reference(cu_ghep, centre):
    """phi_sl of the old result, the steps of c8b_3b.analyse_b on the old join: read -> rate -> slowest (on the angle
    about the median point) -> heading minus the angle about the OLD centre, median over the slowest part."""
    import c8b_3b as tb
    F = tb.read(cu_ghep)
    th0 = np.unwrap(np.arctan2(F['y'] - np.median(F['y']), F['x'] - np.median(F['x'])))
    sl = tb.slowest(tb.rate(F['t'], th0))
    th = np.unwrap(np.arctan2(F['y'] - centre[1], F['x'] - centre[0]))
    return float(np.median((F['psi'] - th)[sl]))


def cmd_kiem(a):
    cu = read_cu(a.cu)
    cx, cy = cu['centre']
    R0 = cu['R0_m']
    phi_sl = phi_reference(a.cu_ghep, (cx, cy))
    rep = dict(version=VERSION, moc=dict(centre=[cx, cy], R0_m=R0, phi_sl_rad=phi_sl, tu=a.cu, sha256=cu['sha256'],
                                         phi_tu=a.cu_ghep, phi_tu_sha256=sha256(a.cu_ghep)), clips={})
    md = ['# Kiem do chinh xac c8_doc_nhoe v3', '',
          f'Moc co dinh: tam ({cx:.4f}; {cy:.4f}) m, R0 {R0:.4f} m tu `{a.cu}`; phi_sl {phi_sl:.6f} rad tinh lai tu '
          f'`{a.cu_ghep}`. Bao cao, khong chon khung. Tieu chi la [hau nghiem, DE XUAT].', '']
    rng = np.random.default_rng(INJ_SEED + 1)
    for name in ('c1', 'c2'):
        d = os.path.join(a.out, f'B_{name}_3b')
        C = load_clip(os.path.join(d, 'frames.csv'))
        with open(os.path.join(d, 'doc_nhoe_rows.csv'), newline='', encoding='utf-8') as f:
            R = {int(r['clip_idx']): r for r in csv.DictReader(f)}
        n = len(C['t'])
        vnew = np.array([r['valid'] for r in C['rows']])
        vold = np.array([R[k]['valid_cu'] == '1' if k in R else False for k in range(n)])
        s = C['seen']
        th = np.full(n, np.nan)
        th[s] = np.unwrap(np.arctan2(C['y'][s] - cy, C['x'][s] - cx))
        dr = np.hypot(C['x'] - cx, C['y'] - cy) - R0
        hp = (C['psi'] - th - phi_sl + math.pi) % (2 * math.pi) - math.pi
        vt = np.abs(omega_loo(C['t'], th, vnew)) * R0
        vt_seen = np.abs(omega_loo(C['t'], th, s)) * R0
        tang = np.full(n, np.nan)                         # leave-one-out: quadratic over +-7 valid frames without it
        vi = np.flatnonzero(vnew)
        per = float(np.median(np.diff(C['t'])))
        for j in range(7, len(vi) - 7):
            w = vi[j - 7:j + 8]
            if C['t'][w[-1]] - C['t'][w[0]] > 14.5 * per:
                continue
            sel = np.r_[w[:7], w[8:]]
            c = np.polyfit(C['t'][sel] - C['t'][vi[j]], th[sel], 2)
            tang[vi[j]] = (th[vi[j]] - c[2]) * R0
        e = np.array([float(R[k]['e_hinh_m']) if k in R and R[k]['e_hinh_m'] != '' else np.nan for k in range(n)])
        quad = np.where(np.isfinite(th), np.floor(np.mod(np.nan_to_num(th), 2 * math.pi) / (math.pi / 2)), -1)
        grp = {'cu': vnew & vold, 'moi': vnew & ~vold}
        cells, by_q, by_row = [], [], []
        bands = list(zip(BANDS_VT[:-1], BANDS_VT[1:]))
        for lo, hi in bands:
            for g, gm in grp.items():
                m = gm & np.isfinite(vt) & (vt >= lo) & (vt < hi)
                cells.append(_cell(name, f'{lo}-{hi}', g, m, dr, hp, tang, e, th))
                for qd in range(N_PHASE):
                    by_q.append(_cell(name, f'{lo}-{hi}/q{qd}', g, m & (quad == qd), dr, hp, tang, e, th))
        rp = C['row']
        q3 = np.nanpercentile(rp[vnew], [33.3, 66.7])
        for lab, m in (('hang_tren', rp < q3[0]), ('hang_giua', (rp >= q3[0]) & (rp < q3[1])), ('hang_duoi', rp >= q3[1])):
            for lo, hi in bands:
                for g, gm in grp.items():
                    mm = gm & m & np.isfinite(vt) & (vt >= lo) & (vt < hi)
                    by_row.append(_cell(name, f'{lab}/{lo}-{hi}', g, mm, dr, hp, tang, e, th))
        # funnel from every PTS: unseen frames and frames without v_t are a class of their own
        funnel = dict(tong=n, khong_thay_ma=int((~s).sum()), thay_ma_khong_vt=int((s & ~np.isfinite(vt_seen)).sum()),
                      theo_dai=[])
        for lo, hi in bands:
            m = s & np.isfinite(vt_seen) & (vt_seen >= lo) & (vt_seen < hi)
            funnel['theo_dai'].append(dict(dai=f'{lo}-{hi}', thay_ma=int(m.sum()), hop_le_cu=int((m & vold).sum()),
                                           hop_le_moi=int((m & vnew).sum()),
                                           co_vt_tren_hop_le=int((m & vnew & np.isfinite(vt)).sum())))
        gaps = _runs(~s)
        funnel['lo_mat_ma'] = dict(so_lo=len(gaps), dai_nhat=max((b - a for a, b in gaps), default=0),
                                   khung_trong_lo_dai=int(sum(b - a for a, b in gaps if (b - a) * per > VT_GAP_S)))
        crit = _criteria(cells)
        sens = _cluster_sensitivity(dr, hp, vt, vnew, gaps, rng)
        rep['clips'][name] = dict(o=cells, theo_pha=by_q, theo_hang=by_row, pheu=funnel, tieu_chi=crit,
                                  do_nhay_cuc_tri=sens)
        md += _md_clip(name, cells, crit, funnel, sens)
    rep['alpha'] = _alpha_report(os.path.join(a.out, 'B_ghep', 'frames.csv'))
    if a.rs:
        rep['doi_chieu_rs'] = _rs_check(a.rs, os.path.join(a.out, 'B_c1_3b'))
    rep['script'] = script_state()
    write_json_new(os.path.join(a.out, 'kiem_do_chinh_xac.json'), rep)
    md += ['', '## Mau alpha (c8b_3b)', '', '```', json.dumps(_clean(rep['alpha']['tom_tat']), ensure_ascii=False,
                                                            indent=1), '```']
    if a.rs:
        md += ['', '## Doi chieu doc doc lap CODE_UT (rs_3601), lech tam px', '', '```',
               json.dumps(_clean(rep['doi_chieu_rs']), ensure_ascii=False, indent=1), '```']
    with open(os.path.join(a.out, 'kiem_do_chinh_xac.md'), 'x', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('kiem: xong')


def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _cluster_sensitivity(dr, hp, vt, valid, gaps, rng, draws=200):
    """Extremes of |dr|, |hp| over the valid frames with v_t in F08..CAP when random runs of frames, with the lengths
    of the real gaps of lost tags, are also lost (sample sensitivity, DN1-08)."""
    m = valid & np.isfinite(vt) & (vt >= 1.166) & (vt < 1.283)
    if m.sum() < MIN_CELL:
        return None
    lens = [b - a for a, b in gaps] or [1]
    base = (float(np.max(np.abs(dr[m]))), float(np.max(np.abs(hp[m]))))
    res = []
    n = len(dr)
    for _ in range(draws):
        mm = m.copy()
        for _ in range(max(1, len(lens) // 10)):
            L = int(rng.choice(lens))
            a0 = int(rng.integers(0, max(1, n - L)))
            mm[a0:a0 + L] = False
        if mm.sum():
            res.append((float(np.max(np.abs(dr[mm]))), float(np.max(np.abs(hp[mm])))))
    r = np.array(res)
    return dict(n=int(m.sum()), dr_max_m=base[0], hp_max_rad=base[1],
                dr_max_khoang_m=[float(r[:, 0].min()), float(r[:, 0].max())],
                hp_max_khoang_rad=[float(r[:, 1].min()), float(r[:, 1].max())])


def _cell(clip, band, grp, m, dr, hp, tang, e, th):
    turns = float(np.ptp(th[m]) / (2 * math.pi)) if m.sum() > 1 else 0.0
    return dict(clip=clip, dai=band, nhom=grp, n=int(m.sum()), vong=turns, dr_m=_stat(dr[m]), hp_rad=_stat(hp[m]),
                tiep_loo_m=_stat(tang[m]), e_hinh_m=_stat(e[m]))


def _criteria(cells):
    """Design v3 3 [hau nghiem, DE XUAT], consistency of the reading only: per band dat / truot / thieu."""
    out = []
    for c0 in cells:
        if c0['nhom'] != 'moi':
            continue
        b = c0['dai']
        if float(b.split('-')[0]) >= 1.283:
            continue
        cu = next(c for c in cells if c['dai'] == b and c['nhom'] == 'cu')
        mo = c0
        r = dict(dai=b)
        if mo['n'] < MIN_CELL or mo['vong'] < 1.0:
            r['moi'] = 'thieu'
        else:
            ok = mo['dr_m']['sd'] <= 0.15e-3 and mo['hp_rad']['sd'] <= math.radians(0.3) and \
                (mo['tiep_loo_m'] is None or mo['tiep_loo_m']['p95_abs'] <= 0.6e-3)
            r['moi'] = 'dat' if ok else 'truot'
        if mo['n'] >= MIN_CELL and cu['n'] >= MIN_CELL:
            ddr = abs(mo['dr_m']['trung_vi'] - cu['dr_m']['trung_vi'])
            dhp = abs(mo['hp_rad']['trung_vi'] - cu['hp_rad']['trung_vi'])
            r['chong'] = 'dat' if ddr <= 0.1e-3 and dhp <= math.radians(0.15) else 'truot'
            r['chong_so'] = dict(d_dr_m=ddr, d_hp_deg=math.degrees(dhp))
        else:
            r['chong'] = 'thieu'
        out.append(r)
    return out


def _md_clip(name, cells, crit, funnel, sens):
    f = lambda s, k, sc=1e3: '-' if s is None else f'{s[k] * sc:.2f}'          # noqa: E731
    dg = 180 / math.pi
    L = [f'## {name}', '', '| dai v_t | nhom | n | vong | dr tv/sd mm | hp tv/sd deg | tiep LOO p95 mm | e_hinh p95 mm |',
         '|---|---|---|---|---|---|---|---|']
    for c in cells:
        if c['n'] == 0:
            continue
        L.append(f'| {c["dai"]} | {c["nhom"]} | {c["n"]} | {c["vong"]:.1f} | {f(c["dr_m"], "trung_vi")}/{f(c["dr_m"], "sd")} '
                 f'| {f(c["hp_rad"], "trung_vi", dg)}/{f(c["hp_rad"], "sd", dg)} | {f(c["tiep_loo_m"], "p95_abs")} '
                 f'| {f(c["e_hinh_m"], "p95_abs")} |')
    L += ['', 'Tieu chi [hau nghiem]: ' + '; '.join(f'{r["dai"]}: moi {r["moi"]}, chong {r["chong"]}' for r in crit),
          '', f'Pheu: tong {funnel["tong"]}, khong thay ma {funnel["khong_thay_ma"]}, thay ma khong v_t '
          f'{funnel["thay_ma_khong_vt"]}; ' + '; '.join(f'{p["dai"]}: thay {p["thay_ma"]}, hop le cu {p["hop_le_cu"]}, moi '
                                                        f'{p["hop_le_moi"]}' for p in funnel['theo_dai']),
          f'Lo mat ma: {funnel["lo_mat_ma"]}', f'Do nhay cuc tri (bo cum nhu lo mat ma): {sens}', '']
    return L


def _stat(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(n=int(len(v)), trung_vi=float(np.median(v)), sd=float(np.std(v)),
                p95_abs=float(np.percentile(np.abs(v), 95)), max_abs=float(np.max(np.abs(v))))




def _alpha_report(fg):
    """The sample c8b_3b's shutter fit actually uses on the new join, alpha without each piece, and a synthetic
    recovery: known alpha added to a smooth fit of the real angle on the real mask."""
    import c8b_3b as tb
    F = tb.read(fg)
    t, x, y = F['t'], F['x'], F['y']
    th0 = np.unwrap(np.arctan2(y - np.median(y), x - np.median(x)))
    sl = tb.slowest(tb.rate(t, th0))
    cx, cy, R0 = tb.circle_fit(x[sl], y[sl])
    th = np.unwrap(np.arctan2(y - cy, x - cx))
    om = tb.rate(t, th)
    spin = tb.trim_runs(t, om, np.isfinite(om) & (np.abs(om) > tb.SHUTTER_MIN_RAD_S))
    row, col = F['row'], F['col']
    dr = row - np.nanmean(row[spin])
    dc = col - np.nanmean(col[spin])
    ok = spin & np.isfinite(dr) & np.isfinite(dc) & np.isfinite(om)
    pieces = tb.make_pieces(t, th, om, ok)
    pc = []
    for p in pieces:
        pc.append(dict(t0=float(t[p][0]), t1=float(t[p][-1]), n=int(p.sum()), vong=float(np.ptp(th[p]) / (2 * math.pi)),
                       v_t=[float(np.min(np.abs(om[p])) * R0), float(np.max(np.abs(om[p])) * R0)],
                       hang=[float(np.min(row[p])), float(np.max(row[p]))], chieu=int(np.sign(np.median(om[p])))))
    whole = tb.fit_shutter(t, th, om, row, F['col'], spin)
    loo = []
    for i, p in enumerate(pieces):
        f = tb.fit_shutter(t, th, om, row, F['col'], spin & ~p)
        if f:
            loo.append(dict(bo_doan=i, alpha_r=f['groups'][0]['alpha_r_s_px'], alpha_c=f['groups'][0]['alpha_c_s_px']))
    # synthetic: smooth part from the whole fit without the shutter term, plus a known alpha and the fit residual
    syn = None
    if whole and pieces:
        m = np.any(np.vstack(pieces), axis=0)
        X = tb.shutter_design(t, th, om, dr, dc, pieces, [ok], True)[m]
        beta, *_ = np.linalg.lstsq(X, th[m], rcond=None)
        resid = th[m] - X @ beta
        base = X[:, :-2] @ beta[:-2]
        out = []
        for ar, ac in ((3.0e-6, -0.6e-6), (0.0, 0.0), (5.0e-6, 1.0e-6)):
            th_s = th.copy()
            th_s[m] = base + om[m] * (ar * dr[m] + ac * dc[m]) + resid
            f = tb.fit_shutter(t, th_s, om, row, F['col'], spin)
            out.append(dict(alpha_r_that=ar, alpha_c_that=ac, alpha_r_uoc=f['groups'][0]['alpha_r_s_px'],
                            alpha_c_uoc=f['groups'][0]['alpha_c_s_px']))
        syn = out
    tom = dict(n_doan=len(pieces), n_khung=int(np.sum([p['n'] for p in pc])) if pc else 0,
               v_t_max=max((p['v_t'][1] for p in pc), default=None),
               alpha_r_chung=whole['groups'][0]['alpha_r_s_px'] if whole else None,
               alpha_c_chung=whole['groups'][0]['alpha_c_s_px'] if whole else None,
               alpha_r_bo_doan=[min((d['alpha_r'] for d in loo), default=None), max((d['alpha_r'] for d in loo), default=None)],
               tong_hop=syn)
    return dict(tom_tat=tom, doan=pc, bo_doan=loo)


def _rs_check(rs_path, d):
    """CODE_UT's independent reading of IMG_3601 (corners per frame), by idx: tag-centre difference in px."""
    with open(os.path.join(d, 'doc_nhoe_rows.csv'), newline='', encoding='utf-8') as f:
        R = {int(r['clip_idx']): r for r in csv.DictReader(f)}
    C = load_clip(os.path.join(d, 'frames.csv'))
    diffs = {'cu': [], 'moi': []}
    with open(rs_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            k = int(r['idx'])
            if r['ok'] != '1' or k >= len(C['t']) or not C['rows'][k]['valid'] or abs(float(r['t_s']) - C['t'][k]) > 1e-5:
                continue
            g = 'cu' if R.get(k, {}).get('valid_cu') == '1' else 'moi'
            diffs[g].append(math.hypot(float(r['col']) - C['col'][k], float(r['row']) - C['row'][k]))
    return {g: _stat(np.array(v)) for g, v in diffs.items()}



# ── main ─────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cat')
    p.add_argument('--video', required=True)
    p.add_argument('--frames', required=True)
    p.add_argument('--out', required=True, help='prefix: <out>.npz, <out>.json (outside the repo)')
    for nm in ('khoa', 'cong'):
        p = sub.add_parser(nm)
        p.add_argument('--c1', nargs=2, required=True, metavar=('FRAMES', 'CACHE'))
        p.add_argument('--c2', nargs=2, required=True, metavar=('FRAMES', 'CACHE'))
        p.add_argument('--session', required=True)
        p.add_argument('--cu', required=True, help='old c8b_3b json (B centre, R0)')
        p.add_argument('--out', required=True)
        if nm == 'cong':
            p.add_argument('--khoa', required=True)
            p.add_argument('--ghep', required=True, help='C1/ghep_3b.py')
            p.add_argument('--frames-a', required=True, help='A clip join for c8b_3b (C1/A2_ghep/frames.csv)')
    p = sub.add_parser('kiem')
    p.add_argument('--out', required=True)
    p.add_argument('--cu', required=True)
    p.add_argument('--cu-ghep', required=True, help='old join C1/B_ghep/frames.csv (heading reference)')
    p.add_argument('--rs', default=None)
    a = ap.parse_args(argv)
    try:
        dict(cat=cmd_cat, khoa=cmd_khoa, cong=cmd_cong, kiem=cmd_kiem)[a.cmd](a)
    except (DnError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'c8_doc_nhoe: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
