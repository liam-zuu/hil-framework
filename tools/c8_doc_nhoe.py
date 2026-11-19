#!/usr/bin/env python3
"""
c8_doc_nhoe.py: read the blurred frames of the 3b B clips (tag on a spinning wheel) that the frame gate (2a) of
c7_measure rejects, with a per-frame check of the tag shape (phase reference, speed pattern) and of every side of the
tag (edge evidence); frames.csv in the c7_measure track format for c8b_3b.

Design: CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_CODE_DOCNHOE_v5.md section 6 (the locked version), built on v4
sections 2-3 and v3 sections 2-6 (Codex Sol rounds 1-5; reviews 03_CODE/docs/review/2026-11-19_Tham_dinh_CODE_DOCNHOE_
vong*.md). Liam 19/11 (CODE_DOCNHOE local_8bbf0c0b, choice box): "Dung, ghi gioi han (Recommended)". Written by
CODE_DOCNHOE local_2b44e87e, 19/11/2026, on feat/c8-doc-nhoe (base feat/c8-33-moc 33b444f). Imports c7_measure,
c7_clip_probe, c8b_3b; changes none of them.

Why: on a spinning tag the residual of the 4-corner fit to a square WITH scale (res_sim_m) grows with the speed and
the gate (2a) of 1.5 mm, set for a still tag, rejects the fast part, while centre and heading stay on the known
circle. The tag passes the same image places every turn, so the shape read from a fast frame is compared with the
shape read from the SLOW frames of OTHER turns at the same phase; the change of shape with speed is a fixed pattern
in the tag frame, learnt on the OTHER clip and taken off (e_hinh). Each side of the tag carries its own evidence
(points kept, line RMS, contrast, width); contrast and width change with speed by a pattern learnt on the other clip,
taken off (r_canh). The temporal check (i) (residual against the neighbours in time) is computed and written per
frame for the REPORT only and drives a side variant (bien_the_thoi_gian/), not the main output (design v5 6, DN5-02).

Main gate for a frame that fails the old (2a) (design v4 2):
  (a) re-detected on the crop, ID 0, centre equal to the track within MATCH_TOL_PX;
  (b) v_t defined (leave-one-out, design v3 2.2) and the verdict unchanged when each neighbour in turn is left out;
  (c) a phase reference: the cell from the angle PREDICTED without the frame, the median of the slow old valid
      frames of that cell from OTHER passes (|unwrapped angle difference| >= pi), at least PHASE_MIN;
  (d) v_t within the p99 of v_t of the clip the patterns were learnt on;
  (e) e_hinh <= e_max;  (f) every side kept >= KEPT_MIN points, line RMS <= RMS_MAX_PX;  (g) r_canh <= r_max;
  (h) res_sim_m + QUANT <= RES_SIM_CAP_M.
Old frames keep every field of the track; frames taken back get (1), (2b), speeds of run B (design v3 2.4).

Commands:
  cat    one pass over the video: a crop around the tag centre of every frame where the track saw the tag, with its
         PTS (must equal t_s of frames.csv) -> <cache>.npz + .json (outside the repo)
  khoa   from both clips: references, patterns, e_max, r_max, temporal thresholds; the criterion (2) checks (design
         v4 3 with v5 2): largest pose error a gate admits under a one-corner move (fast frames, new gate vs slow
         frames, old gate), paired image injection by phase cell (families P, Od, Ob, E, B, S, PS) scored through
         the decision path of cong, health of the old valid frames, the cluster families (report: limitation
         DN5-01), the template parameters and the locked sample of the second reading; refuses (du_nhay false)
         when a condition fails -> khoa.json OUTSIDE the result tree, before any frame is taken back
  cong   needs khoa.json (hashes, version, recomputed numbers checked): the main output, the temporal columns, the
         side variant, the second reading on the locked sample, the joins, c8b_3b (CLI, unchanged) on both joins
  kiem   accuracy report against the OLD 3b result (fixed circle, R0, phi_sl recomputed as c8b_3b does): signed dr,
         hp by clip x speed band x group, phase quadrant, image rows; funnel; alpha sample; cluster sensitivity of
         the extremes; second reading; scale of new frames vs old frames of the same phase; rs_3601. Selects nothing.
All thresholds are [DE XUAT].

  python tools/c8_doc_nhoe.py cat  --video V.MOV --frames F/frames.csv --out CACHE_PREFIX
  python tools/c8_doc_nhoe.py khoa --c1 F1/frames.csv CACHE1 --c2 F2/frames.csv CACHE2 --session S/session.json \\
                                   --cu C1/c8b_3b_A2_B.json --out C1_docnhoe_khoa.json
  python tools/c8_doc_nhoe.py cong --khoa C1_docnhoe_khoa.json --c1 ... --c2 ... --session ... --cu ... \\
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
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7_clip_probe as cp                                               # noqa: E402
import c7_measure as cm                                                  # noqa: E402

VERSION = 'c8_doc_nhoe_v5'
CAR_ID = cm.CAR_ID
ROI_HALF = 110              # crop half size, px (tag about 92 px, blur up to about 15 px)
MATCH_TOL_PX = 0.01         # (a) the crop re-detection gives the centre of the track within this [DE XUAT]
VT_WIN_S = 0.1              # (b) v_t: frames within +-0.1 s, the frame itself left out BEFORE unwrapping [DE XUAT]
VT_SIDE_MIN = 2             #     at least 2 frames on each side
VT_GAP_S = 0.05             #     no gap above 0.05 s between consecutive frames of the window
VT_STEP_MAX = 0.6           #     no phase step above 0.6 rad between consecutive frames of the window
VT_SIDE_TOL = 0.2           #     left and right linear rates agree within 20 % of their mean
SLOW_VT = 0.3               # (c) phase reference: old valid frames up to 0.3 m/s
N_PHASE_REF = 72            #     5 deg cells
PHASE_MIN = 5               #     frames per cell, other passes only
HUBER_IT, HUBER_K = 10, 1.5
PAT_RANGE_Q = 99.0          # (d) the patterns are used up to the 99th percentile of v_t of the clip they were learnt on
KHOA_VT_MAX = 0.8           # e_max, r_max from the old valid frames up to 0.8 m/s
E_MAX_K, E_Q = 1.5, 99.9    # e_max = max(1.5 x p99.9, floor), raw (DN3-10)
E_MAX_FLOOR_M, E_MAX_CEIL_M = 0.001, 0.003
KEPT_MIN = 20               # (f) per side: points kept of 24 profiles
RMS_MAX_PX = 0.5            #     per side: RMS of the edge points about the line
R_MAX_CEIL = 0.6            # (g) r_max = 1.5 x p99.9 (old valid, <= 0.8 m/s), refused above 0.6
RES_SIM_CAP_M = 0.005       # (h)
QUANT = 5e-11               # half a unit of the 10th decimal of res_sim_m, scale (frames.csv of track)
# temporal check (i), design v5 1 (report only, side variant)
T_WIN_S, T_SIDE, T_GAP_S = 0.1, 2, 0.05
T_K, T_Q = 1.5, 99.9
T_BANDS = ((0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.05))
T_V_CAP = 1.283             # the tangential threshold is extrapolated linearly up to CAP (held above)
# criterion (2) in khoa (design v4 3, v5 2)
INJ_SEED = 20261119
INJ_PAIRS = 30              # pairs (slow old valid, fast taken) per clip, same phase cell
FAST_BAND = (1.0, 1.3)
FAMILIES = (('P', 2.0), ('P', 3.0), ('P', 4.0), ('P', 5.0), ('Od', 6), ('Od', 10), ('Od', 15), ('Ob', 6), ('Ob', 10),
            ('Ob', 15), ('E', 3), ('E', 6), ('B', 1), ('S', 0.5), ('S', 1.0), ('S', 1.5), ('S', 2.0), ('S', 3.0),
            ('S', 4.0), ('PS', (2.0, 1.0)), ('PS', (2.5, 2.0)), ('PS', (3.0, 3.0)))
REPORT_FAMS = ('S', 'PS')  # whole-tag shift: no one-frame gate sees it (old gate neither, DN4-01): report only (design v5
#                            7, Liam 19/11 "Sua, ghi ro (Recommended)", changed AFTER the trial run of khoa)
BAD_C_M, BAD_P_DEG = 0.5e-3, 0.25    # "accepted wrong": centre > 0.5 mm or heading > 0.25 deg vs the original read
# hard condition per family (design v5 8, Liam 19/11 "Dem loi sai (Recommended)", changed AFTER the run of 784fad7):
# accepted-wrong of the new gate <= of the old gate, and in pairs: wrong accepted only by the new gate <= only by the
# old gate. The largest accepted errors (centre, heading) are reported only (the envelope measures the largest
# error a gate admits). No tolerance anywhere (DNC1-05).
ENV_N, ENV_DIRS, ENV_BISECT, ENV_MAX_MM = 60, 24, 14, 15.0
HEALTH_MAX = 0.001          # v5 2.5: old valid frames failing the new gate <= 0.1 % of those that can be scored
CLUSTER_L = (6, 15, 29)     # cluster families (DN5-01, report only)
CLUSTER_MM = (1.0, 2.0, 3.0, 4.4)
CLUSTER_N = 10
# second reading: blurred template fit (design v2 2.2), report only
SIG_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
SIG_N = 20
T_GRID_S = tuple(k * 0.002 for k in range(9))      # 0-16 ms
T_N, T_BAND = 30, (0.6, 1.1)
N_BLUR = 7
MASK_DILATE_PX = 4
TPL_CELL = 16
DOC2_BAND = (1.166, 1.283)  # every DOC2_STRIDE-th frame (by idx) with v_t in F08..CAP, locked in khoa
DOC2_STRIDE = 10
# kiem
BANDS_VT = (0.0, 0.3, 0.8, 1.0, 1.166, 1.283, 99.0)
N_PHASE = 4
MIN_CELL = 30
RUN_LONG = 6                # DN4-07: runs of taken frames longer than this without an old frame


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
                    sha256=sha256(__file__), c7_measure_sha256=sha256(cm.__file__))
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, version=VERSION, sha256=sha256(__file__),
                    c7_measure_sha256=sha256(cm.__file__))


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


def wrap(a):
    return (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi


def kasa(x, y):
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (cx, cy, c), *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    return float(cx), float(cy), float(math.sqrt(c + cx ** 2 + cy ** 2))


def _close(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol * (1.0 + np.abs(b))))


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
                res_sim=g('res_sim_m'), scale=g('scale'), scale_ref=g('scale_ref'),
                valid_old=np.array([bool(r['valid']) for r in rows]),
                why_old=[list(r['why']) for r in rows], x=g('tag_x'), y=g('tag_y'), psi=g('psi'))


def image_axis(C):
    """Image point of the wheel axis: circle through the tag centre pixels of the old valid frames."""
    m = C['valid_old']
    return kasa(C['col'][m], C['row'][m])


def image_angle(C, axis):
    ang = np.full(len(C['t']), np.nan)
    s = C['seen']
    ang[s] = np.unwrap(np.arctan2(C['row'][s] - axis[1], C['col'][s] - axis[0]))
    return ang


# ── v_t, leave-one-out (design v3 2.2, v4 2 (b), (c)) ────────────────────────────────────────────

def _windows(ts, win=VT_WIN_S):
    a = np.searchsorted(ts, ts - win - 1e-9, 'left')
    b = np.searchsorted(ts, ts + win + 1e-9, 'right')
    return a, b


def loo_fit(ts, an, j, a, b, drop=-1, side_min=VT_SIDE_MIN, gap=VT_GAP_S):
    """Rate and predicted angle at position j from the positions a..b-1 other than j (and `drop`): the frame is left
    out BEFORE the phase is unwrapped (only wrapped differences of `an` are used, DN2-04); at least side_min frames on
    each side; no gap above `gap` between consecutive NEIGHBOURS (the frame itself not counted, DNC1-02); no phase step
    above VT_STEP_MAX; the linear rates of the two sides agree within VT_SIDE_TOL; quadratic fit. Returns
    (omega rad/s, predicted angle rad, absolute in the frame of `an`) or None."""
    pos = [i for i in range(a, b) if i != j and i != drop]
    left = sum(1 for i in pos if i < j)
    if left < side_min or len(pos) - left < side_min:
        return None
    sel = np.array(pos)
    if np.max(np.diff(ts[sel])) > gap + 1e-9:
        return None
    st = wrap(np.diff(an[sel]))
    if np.max(np.abs(st)) > VT_STEP_MAX:
        return None
    aa = an[sel[0]] + np.r_[0.0, np.cumsum(st)]
    tt = ts[sel] - ts[j]
    wl = np.polyfit(tt[:left], aa[:left], 1)[0]
    wr = np.polyfit(tt[left:], aa[left:], 1)[0]
    m = 0.5 * (abs(wl) + abs(wr))
    if np.sign(wl) != np.sign(wr) or (m > 0 and abs(wl - wr) > VT_SIDE_TOL * m):
        return None
    c = np.polyfit(tt, aa, 2)
    return float(c[1]), float(c[2])


def loo_all(t, ang, seen):
    """loo_fit at every seen frame: (omega, predicted angle, window (j, a, b) per frame index)."""
    idx = np.flatnonzero(seen)
    ts, an = t[idx], ang[idx]
    om = np.full(len(t), np.nan)
    pred = np.full(len(t), np.nan)
    win = {}
    A, B = _windows(ts)
    for j, k in enumerate(idx):
        win[int(k)] = (j, int(A[j]), int(B[j]))
        r = loo_fit(ts, an, j, int(A[j]), int(B[j]))
        if r is not None:
            om[k], pred[k] = r
    return om, pred, win, ts, an


def omega_loo(t, ang, seen):
    """Rotation rate at every seen frame from the OTHER seen frames within +-VT_WIN_S (NaN where not defined)."""
    return loo_all(t, ang, seen)[0]


def phase_cell(ang):
    ang = np.atleast_1d(np.asarray(ang, float))
    out = np.full(len(ang), -1, int)
    ok = np.isfinite(ang)
    out[ok] = np.floor(np.mod(ang[ok], 2 * math.pi) / (2 * math.pi) * N_PHASE_REF).astype(int) % N_PHASE_REF
    return out


# ── shape (design v3 2.3) ────────────────────────────────────────────────────────────────────────

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


def shape_dev(shape, ref_cell, side_m):
    """Shape difference (8,) in m on the tag plane: similarity residual from the phase reference to the frame."""
    return (sim_res(ref_cell, shape) * side_m).ravel()


def huber_fit(A, Y):
    """Y ~ A P, Huber weights on the norm of the residual row (HUBER_IT passes, HUBER_K x MAD scale)."""
    w = np.ones(len(A))
    P = None
    for _ in range(HUBER_IT):
        P, *_ = np.linalg.lstsq(A * w[:, None], Y * w[:, None], rcond=None)
        r = np.linalg.norm(Y - A @ P, axis=1)
        s = 1.4826 * float(np.median(r)) + 1e-12
        w = np.sqrt(np.minimum(1.0, HUBER_K * s / np.maximum(r, 1e-15)))
    return P


def huber_pattern(v, dev):
    """dev ~ v P1 + v^2 P2 (8 components each)."""
    return huber_fit(np.column_stack([v, v ** 2]), dev)


def e_hinh(dev, v, B):
    """Largest corner of what is left after the speed pattern: max_i |dev_i - (v P1 + v^2 P2)_i| (m)."""
    v = np.atleast_1d(np.asarray(v, float))
    E = np.asarray(dev).reshape(-1, 8) - np.column_stack([v, v ** 2]) @ B
    return np.max(np.linalg.norm(E.reshape(-1, 4, 2), axis=2), axis=1)


class RefSet:
    """Phase references (design v4 2 (c), DN3-04): per cell the normalised shapes of the slow old valid frames with
    their unwrapped image angle; the reference for a frame is the median over the frames of its cell from OTHER
    passes, |angle difference| >= pi (the frame's own pass, itself included, is left out)."""

    def __init__(self, idx, cells, angs, shapes):
        self.idx = np.asarray(idx, int)
        self.by = {}
        for c in np.unique(cells):
            m = cells == c
            self.by[int(c)] = (np.asarray(angs)[m], np.asarray(shapes)[m])

    def ref(self, cell, ang_k):
        if cell < 0 or cell not in self.by or not np.isfinite(ang_k):
            return None
        a, s = self.by[cell]
        m = np.abs(a - ang_k) >= math.pi
        if m.sum() < PHASE_MIN:
            return None
        return np.median(s[m], axis=0)


# ── edges (design v4 1.1, 2 (f), (g)) ────────────────────────────────────────────────────────────

def refine_edges_diag(img, corners, half=6.0, skip=0.18, n_samp=24, iters=2):
    """c7_measure.refine_edges, line for line, plus the evidence of each side at the last iteration: profiles kept
    (ok), points kept after the 2.5 sigma rejection, RMS of the kept points about the line (px), median edge contrast
    (peak of the outward derivative), median spread of the derivative (px, edge width). Returns (corners, diag) or
    (None, None) exactly when refine_edges returns None (test: same corners)."""
    c = np.asarray(corners, float).copy()
    f32 = img if img.dtype == np.float32 else img.astype(np.float32)
    ss = np.arange(-half, half + 0.25, 0.5)
    sm = (ss[:-1] + ss[1:]) / 2.0
    diag = None
    for _ in range(iters):
        lines, diag = [], []
        for k in range(4):
            p0, p1 = c[k], c[(k + 1) % 4]
            d = p1 - p0
            length = float(np.hypot(*d))
            if length < 8.0:
                return None, None
            d = d / length
            nrm = np.array([d[1], -d[0]])
            fr = np.linspace(skip, 1.0 - skip, n_samp)
            base = p0[None, :] + fr[:, None] * length * d[None, :]
            px = (base[:, None, 0] + ss[None, :] * nrm[0]).astype(np.float32)
            py = (base[:, None, 1] + ss[None, :] * nrm[1]).astype(np.float32)
            prof = cv2.remap(f32, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g = np.clip(np.diff(prof, axis=1), 0.0, None)
            peak = g.max(axis=1)
            g = np.where(g >= 0.2 * g.max(axis=1, keepdims=True), g, 0.0)
            wsum = g.sum(axis=1)
            ok = wsum > 1e-6
            if ok.sum() < 6:
                return None, None
            sstar = (g[ok] * sm[None, :]).sum(axis=1) / wsum[ok]
            spread = np.sqrt(np.maximum((g[ok] * (sm[None, :] - sstar[:, None]) ** 2).sum(axis=1) / wsum[ok], 0.0))
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
            r = (pts - m) @ nv
            lines.append((m, dirv))
            diag.append(dict(ok=int(ok.sum()), kept=int(len(pts)), rms_px=float(np.sqrt(np.mean(r ** 2))),
                             contrast=float(np.median(peak[ok])), width_px=float(np.median(spread))))
        new = np.zeros((4, 2))
        for k in range(4):
            (m1, d1), (m2, d2) = lines[k - 1], lines[k]
            a = np.column_stack([d1, -d2])
            if abs(np.linalg.det(a)) < 1e-9:
                return None, None
            tt = np.linalg.solve(a, m2 - m1)
            new[k] = m1 + tt[0] * d1
        if np.max(np.linalg.norm(new - c, axis=1)) > 4.0 + half:
            return None, None
        c = new
    return c, diag


def detect_diag(det, gray):
    """c7_measure.detect_refined for the car tag only, with the per-side evidence: (corners, diag) or (None, why)."""
    found = cp.detect_markers(det, gray)
    if CAR_ID not in found:
        return None, 'khong_do_lai'
    c0 = found[CAR_ID]
    side = float(np.mean(np.linalg.norm(np.diff(np.vstack([c0, c0[:1]]), axis=0), axis=1)))
    half = min(cm.HALF_CAR_PX, 0.75 * side / 6.0)
    q, dg = refine_edges_diag(gray, c0, half=half)
    if q is None:
        return None, 'refine_none'
    return q, dg


def edge_vec(dg):
    """8 photometric numbers of a frame, sides in the tag's own order: log(contrast_j / median contrast) and
    log(width_j / median width), j = 0..3 (ArUco side order is fixed on the tag)."""
    con = np.array([d['contrast'] for d in dg], float)
    wid = np.array([d['width_px'] for d in dg], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.r_[np.log(con / np.median(con)), np.log(wid / np.median(wid))]


def edge_basic_ok(dg):
    return min(d['kept'] for d in dg) >= KEPT_MIN and max(d['rms_px'] for d in dg) <= RMS_MAX_PX


def edge_pattern(v, X):
    """X ~ [1, v, v^2] P (3 x 8)."""
    return huber_fit(np.column_stack([np.ones_like(v), v, v ** 2]), X)


def r_canh(x, v, P):
    """Largest photometric residual of a side after the speed pattern."""
    x = np.asarray(x).reshape(-1, 8)
    v = np.atleast_1d(np.asarray(v, float))
    return np.max(np.abs(x - np.column_stack([np.ones_like(v), v, v ** 2]) @ P), axis=1)


# ── crops (cat) ──────────────────────────────────────────────────────────────────────────────────

def cmd_cat(a):
    low_priority()
    C = load_clip(a.frames)
    want = set(np.flatnonzero(C['seen']).tolist())
    if os.path.exists(a.out + '.npz') or os.path.exists(a.out + '.json'):
        raise DnError(f'{a.out}.npz/.json exists')
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
    if meta.get('roi_half') != ROI_HALF:
        raise DnError(f'{prefix}: roi_half {meta.get("roi_half")} != {ROI_HALF}')
    if sha256(prefix + '.npz') != meta['npz_sha256']:
        raise DnError(f'{prefix}.npz: sha256 differs from its .json')
    z = np.load(prefix + '.npz')
    d = {int(k): j for j, k in enumerate(z['idx'])}
    for j, k in enumerate(z['idx']):
        if abs(float(z['pts'][j]) - C['t'][int(k)]) > 1e-6:
            raise DnError(f'{prefix}: crop {int(k)} PTS differs from t_s')
    return dict(meta=meta, pos=d, x0=z['x0'], y0=z['y0'], crops=z['crops'], pts=z['pts'])


# ── analysis of a clip ───────────────────────────────────────────────────────────────────────────

_DET = None


def detector():
    global _DET
    if _DET is None:
        _DET = cm.make_detector()
    return _DET


def redetect(cache, k, C):
    """(corners (4, 2) in full-image px, side evidence, '') re-detected in the crop, or (None, None, why)."""
    j = cache['pos'].get(int(k))
    if j is None:
        return None, None, 'khong_cat'
    q, dg = detect_diag(detector(), cache['crops'][j])
    if q is None:
        return None, None, dg
    q = q + [cache['x0'][j], cache['y0'][j]]
    c = q.mean(axis=0)
    if not math.hypot(c[0] - C['col'][k], c[1] - C['row'][k]) <= MATCH_TOL_PX:      # Euclidean (DNC1-14)
        return None, None, 'lech_track'
    return q, dg, ''


def read_cu(path):
    with open(path, encoding='utf-8') as f:
        J = json.load(f)
    B = J['B']
    return dict(path=path, sha256=sha256(path), centre=B['centre'], R0_m=float(B['R0_m']))


def load_session(path):
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    return s, sha256(path)


def analyse_clip(C, cache, R0):
    """Corners and side evidence re-detected on every crop, image axis, unwrapped image angle, leave-one-out rate and
    predicted angle, v_t, normalised shapes, edge vectors, phase cells (measured, predicted)."""
    n = len(C['t'])
    Q = np.full((n, 4, 2), np.nan)
    F = np.full((n, 4, 4), np.nan)                     # side x (kept, rms_px, contrast, width_px)
    why = [''] * n
    for k in np.flatnonzero(C['seen']):
        q, dg, w = redetect(cache, k, C)
        if q is None:
            why[k] = w
        else:
            Q[k] = q
            F[k] = [[d['kept'], d['rms_px'], d['contrast'], d['width_px']] for d in dg]
    qok = np.isfinite(Q.reshape(n, -1)).all(axis=1)
    axis = image_axis(C)
    ang = image_angle(C, axis)
    om, pred, win, ts, an = loo_all(C['t'], ang, C['seen'])
    shapes = np.full((n, 4, 2), np.nan)
    X = np.full((n, 8), np.nan)
    for k in np.flatnonzero(qok):
        shapes[k] = norm_shape(Q[k])
        X[k] = edge_vec(_dg(F[k]))
    return dict(Q=Q, F=F, qok=qok, why_det=why, axis=axis, ang=ang, om=om, pred=pred, vt=np.abs(om) * R0,
                win=win, ts=ts, an=an, R0=R0, shapes=shapes, X=X, cell=phase_cell(pred), cell_meas=phase_cell(ang))


def _dg(f):
    return [dict(kept=f[i, 0], rms_px=f[i, 1], contrast=f[i, 2], width_px=f[i, 3]) for i in range(4)]


def ref_idx(C, A):
    return np.flatnonzero(A['qok'] & C['valid_old'] & np.isfinite(A['vt']) & (A['vt'] <= SLOW_VT) & (A['cell'] >= 0))


def make_refset(A, idx):
    """Reference frames placed in the cell of their PREDICTED angle, as the target (DNC1-14); pass identity by the
    measured unwrapped angle."""
    idx = np.asarray(idx, int)
    return RefSet(idx, A['cell'][idx], A['ang'][idx], A['shapes'][idx])


def devs(A, rs, side_m):
    """Shape difference of every re-detected frame with a v_t from its phase reference (NaN without one)."""
    n = len(A['vt'])
    dev = np.full((n, 8), np.nan)
    for k in np.flatnonzero(A['qok'] & np.isfinite(A['vt'])):
        r = rs.ref(int(A['cell'][k]), A['ang'][k])
        if r is not None:
            dev[k] = shape_dev(A['shapes'][k], r, side_m)
    return dev


def learn(A, dev):
    """Shape pattern, edge pattern and their v_t range (p99) learnt on this clip (used for the other clip)."""
    ok = np.isfinite(dev).all(axis=1) & np.isfinite(A['vt'])
    if ok.sum() < 1000:
        raise DnError(f'{int(ok.sum())} frames with a shape and v_t, need 1000')
    oke = np.isfinite(A['X']).all(axis=1) & np.isfinite(A['vt'])
    return dict(kieu_lech=huber_pattern(A['vt'][ok], dev[ok]),
                kieu_canh=edge_pattern(A['vt'][oke], A['X'][oke]),
                mien_vt=float(min(np.percentile(A['vt'][ok], PAT_RANGE_Q), np.percentile(A['vt'][oke], PAT_RANGE_Q))),
                n_hoc=int(ok.sum()), n_hoc_canh=int(oke.sum()))


# ── the main gate (design v4 2) ──────────────────────────────────────────────────────────────────

class Gate:
    """Thresholds and models of one clip: its RefSet, the other clip's patterns and range, e_max, r_max."""

    def __init__(self, A, rs, B, P, vmax, e_max, r_max, side_m):
        self.A, self.rs, self.B, self.P, self.vmax = A, rs, np.asarray(B), np.asarray(P), vmax
        self.e_max, self.r_max, self.side = e_max, r_max, side_m
        self._drops = {}

    def shape_part(self, k, sh, xv, vt, cell):
        """(c), (d), (e), (g) for a frame at index k with shape sh, edge vector xv under (vt, cell): (why, e, r)."""
        ref = self.rs.ref(int(cell), self.A['ang'][k])
        if ref is None:
            return 'khong_chuan_pha', None, None
        if not vt <= self.vmax:
            return 'ngoai_mien', None, None
        e = float(e_hinh(shape_dev(sh, ref, self.side), vt, self.B)[0])
        r = float(r_canh(xv, vt, self.P)[0]) if np.all(np.isfinite(xv)) else math.inf
        if not e <= self.e_max:
            return 'e_hinh_vuot', e, r
        if not r <= self.r_max:
            return 'r_canh_vuot', e, r
        return '', e, r

    def drops(self, k):
        """(vt, cell) of frame k with each neighbour of its window left out in turn; None where not defined."""
        if k not in self._drops:
            j, a, b = self.A['win'][k]
            out = []
            for i in range(a, b):
                if i == j:
                    continue
                r = loo_fit(self.A['ts'], self.A['an'], j, a, b, drop=i)
                out.append(None if r is None else (abs(r[0]) * self.A['R0'], int(phase_cell(r[1])[0])))
            self._drops[k] = out
        return self._drops[k]

    def judge(self, k, sh, xv, F4, res_sim):
        """(b)-(h) for frame k with shape sh, edge vector xv, side evidence F4 (4 x 4) and res_sim: (why, e, r)."""
        vt = self.A['vt'][k]
        if not np.isfinite(vt):
            return 'khong_vt', None, None
        why, e, r = self.shape_part(k, sh, xv, vt, self.A['cell'][k])
        if why:
            return why, e, r
        if not edge_basic_ok(_dg(F4)):
            return 'canh_co_ban', e, r
        if not res_sim + QUANT <= RES_SIM_CAP_M:
            return 'res_sim_vuot_tran', e, r
        dr = self.drops(k)                                   # (b) the verdict holds with every neighbour left out
        for d in dr:
            if d is not None and self.shape_part(k, sh, xv, d[0], d[1])[0]:
                return 'vt_khong_ben', e, r                  # a defined v_t flips the verdict (checked first, DNC1-08)
        if any(d is None for d in dr):
            return 'vt_mat_khi_bo', e, r                     # v_t undefined with a neighbour left out (design v5 7)
        return '', e, r


def decide(C, A, G, gate_m):
    """Per seen frame: candidate (fails the old (2a)), why it is not taken, e_hinh, r_canh; returns (take, info)."""
    n = len(C['t'])
    take = np.zeros(n, bool)
    info = [dict() for _ in range(n)]
    for k in np.flatnonzero(C['seen']):
        d = info[k]
        d['v_t'] = float(A['vt'][k]) if np.isfinite(A['vt'][k]) else None
        d['o_pha'] = int(A['cell'][k])
        cand = not (C['res_sim'][k] <= gate_m)
        d['cong'] = 'ung_vien' if cand else 'cu'
        if not A['qok'][k]:
            d['ly_do'] = A['why_det'][k] or 'khong_do_lai'
            d['e_hinh_m'] = d['r_canh'] = None
            continue
        if not cand:                                         # old frame: the numbers, for the report only
            vt = A['vt'][k]
            if np.isfinite(vt):
                _, e, r = G.shape_part(k, A['shapes'][k], A['X'][k], vt, A['cell'][k])
            else:
                e = r = None
            d['e_hinh_m'], d['r_canh'], d['ly_do'] = e, r, ''
            continue
        why, e, r = G.judge(k, A['shapes'][k], A['X'][k], A['F'][k], C['res_sim'][k])
        d['e_hinh_m'], d['r_canh'], d['ly_do'] = e, r, why
        take[k] = not why
    return take, info


# ── run A / run B (design v3 2.4) ────────────────────────────────────────────────────────────────

def regate_with(rows, take, rms_h, gates):
    """c7_measure.apply_gates on a copy where the frames taken back carry res_sim_m 0 (they pass (2a) and join the
    neighbours of (2b)); the others keep their residual. valid, why, scale_ref and the speeds are copied back to a
    copy of the rows; res_sim_m stays the measured one."""
    eff = copy.deepcopy(rows)
    for k, r in enumerate(eff):
        if take[k]:
            r['res_sim_m'] = 0.0
            r.pop('_res_sim_lo', None)
            r.pop('_res_sim_hi', None)
    cm.apply_gates(eff, rms_h, gates, quant=QUANT)
    out = copy.deepcopy(rows)
    for r, e in zip(out, eff):
        for c in ('valid', 'why', 'scale_ref', 'vx', 'vy', 'wz', 'speed'):
            r[c] = e[c]
    return out


def combine(rows, take, rms_h, gates):
    """Run A takes nothing (= the track, checked by cong), run B takes the frames. A frame not taken keeps its track
    row unchanged, every field (design v4 2, DNC1-01); a taken frame gets valid, why, scale_ref and the speeds of run
    B (design v3 2.4); res_sim_m stays the measured one. Returns (rows, run A, run B, old frames run B flips)."""
    ra = regate_with(rows, np.zeros(len(rows), bool), rms_h, gates)
    rb = regate_with(rows, take, rms_h, gates)
    out = copy.deepcopy(rows)
    flips = []
    for k, r in enumerate(out):
        if take[k]:
            for c in ('valid', 'why', 'scale_ref', 'vx', 'vy', 'wz', 'speed'):
                r[c] = copy.deepcopy(rb[k][c])
        elif ra[k]['valid'] != rb[k]['valid']:
            flips.append(k)
    return out, ra, rb, flips


# ── temporal check (i), design v5 1 (report only) ────────────────────────────────────────────────

def temporal_one(t, x, y, psi, nidx, k, centre, R0):
    """Residuals of frame k against its neighbours in time: nidx sorted frame indices of the neighbour set N (k is
    taken out of it); +-T_WIN_S, >= T_SIDE each side, no gap above T_GAP_S between neighbours (k not counted,
    DNC1-02). Tangential: (theta_k - theta
    hat) R0, theta hat from a quadratic over the neighbours' angles about the fixed old centre, unwrapped WITHOUT the
    frame (DN5-04); radial: r_k - median r of the neighbours; heading: hp_k minus a linear fit of hp = psi - theta of
    the neighbours. Returns (tangential m, radial m, heading rad, v of the fit m/s) or None."""
    cx, cy = centre
    a = int(np.searchsorted(t[nidx], t[k] - T_WIN_S - 1e-9, 'left'))
    b = int(np.searchsorted(t[nidx], t[k] + T_WIN_S + 1e-9, 'right'))
    nb = np.array([i for i in nidx[a:b] if i != k], int)
    if (nb < k).sum() < T_SIDE or (nb > k).sum() < T_SIDE:
        return None
    if np.max(np.diff(t[nb])) > T_GAP_S + 1e-9:
        return None
    th_n = np.unwrap(np.arctan2(y[nb] - cy, x[nb] - cx))
    tt = t[nb] - t[k]
    c = np.polyfit(tt, th_n, 2)
    th_k = math.atan2(y[k] - cy, x[k] - cx)
    du_tiep = float(wrap(th_k - c[2])) * R0
    r_n = np.hypot(x[nb] - cx, y[nb] - cy)
    du_phap = float(math.hypot(x[k] - cx, y[k] - cy) - np.median(r_n))
    hp_n = np.unwrap(wrap(psi[nb] - th_n))
    ch = np.polyfit(tt, hp_n, 1)
    du_huong = float(wrap(psi[k] - th_k - ch[1]))
    return du_tiep, du_phap, du_huong, abs(float(c[1])) * R0


def temporal_pass(C, N, centre, R0, targets):
    n = len(C['t'])
    res = np.full((n, 4), np.nan)
    nidx = np.flatnonzero(N)
    for k in targets:
        r = temporal_one(C['t'], C['x'], C['y'], C['psi'], nidx, int(k), centre, R0)
        if r is not None:
            res[k] = r
    return res


def temporal_thresholds(res, old):
    """T_tiep(v) = T_K (a + b v) from a line through (median v, p99.9 |tangential|) of the old valid frames in the 4
    bands; T_phap, T_huong = T_K p99.9 over the old valid frames (design v5 1.2)."""
    xs, ys = [], []
    for lo, hi in T_BANDS:
        m = old & np.isfinite(res[:, 0]) & (res[:, 3] >= lo) & (res[:, 3] < hi)
        if m.sum() < MIN_CELL:
            raise DnError(f'temporal threshold: {int(m.sum())} old valid frames in {lo}-{hi} m/s')
        xs.append(float(np.median(res[m, 3])))
        ys.append(float(np.percentile(np.abs(res[m, 0]), T_Q)))
    b, a = np.polyfit(xs, ys, 1)
    mo = old & np.isfinite(res[:, 1])
    return dict(a_m=float(a), b_s=float(b), diem=[xs, ys], T_phap_m=T_K * float(np.percentile(np.abs(res[mo, 1]), T_Q)),
                T_huong_rad=T_K * float(np.percentile(np.abs(res[mo, 2]), T_Q)), n_cu=int(mo.sum()))


def t_tiep(TH, v):
    return T_K * (TH['a_m'] + TH['b_s'] * np.minimum(v, T_V_CAP))


def temporal_fail(res, TH):
    """True where the frame fails (i) (residuals defined), False where it passes, NaN-less mask of defined rows."""
    ok = np.isfinite(res[:, 0])
    f = np.zeros(len(res), bool)
    f[ok] = (np.abs(res[ok, 0]) > t_tiep(TH, res[ok, 3])) | (np.abs(res[ok, 1]) > TH['T_phap_m']) | \
        (np.abs(res[ok, 2]) > TH['T_huong_rad'])
    return f, ok


def temporal_check(C, valid_old, take, centre, R0, TH=None):
    """Two passes (design v5 1.3, DN5-04): N = old valid frames + frames taken by the main gate; pass 1 marks the
    frames of N that fail; pass 2 recomputes every frame with N minus those, and decides. Thresholds (when TH is None)
    from the residuals of the old valid frames on the chain of old valid frames ONLY, so the frames taken back do not
    move them (DNC1-07). Returns (pass-2 residuals, passes (i) for the frames whose residuals are defined, defined,
    TH, N of pass 2)."""
    N = valid_old | take
    targets = np.flatnonzero(C['seen'])
    if TH is None:
        TH = temporal_thresholds(temporal_pass(C, valid_old, centre, R0, np.flatnonzero(valid_old)), valid_old)
    r1 = temporal_pass(C, N, centre, R0, targets)
    f1, _ = temporal_fail(r1, TH)
    N2 = N & ~f1
    r2 = temporal_pass(C, N2, centre, R0, targets)
    f2, ok2 = temporal_fail(r2, TH)
    return r2, ok2 & ~f2, ok2, TH, N2


# ── image injection (design v4 1.4, v5 2.1) ──────────────────────────────────────────────────────

_MK = None


def _marker():
    global _MK
    if _MK is None:
        s = TPL_CELL
        mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cp.DICT), CAR_ID, 6 * s)
        can = np.array([[s, s], [7 * s, s], [7 * s, 7 * s], [s, 7 * s]], np.float32) - 0.5 - s
        _MK = (mk, can)
    return _MK


def draw_params(fam, mag, rng):
    """The random choices of one injection, drawn once and applied to both frames of a pair."""
    p = dict(fam=fam, mag=mag, goc=int(rng.integers(4)), huong=float(rng.uniform(0.0, 2 * math.pi)),
             giua=bool(rng.random() < 0.5), o=(int(rng.integers(1, 5)), int(rng.integers(1, 5))),
             huong2=float(rng.uniform(0.0, 2 * math.pi)))
    return p


def _blend(img, mask, val):
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)
    return np.clip(img * (1 - m) + val * m, 0, 255).astype(np.uint8)


def _move_corner(crop, qc, i, vec):
    q2 = qc.copy()
    q2[i] += vec
    H = cv2.getPerspectiveTransform(qc.astype(np.float32), q2.astype(np.float32))
    return cv2.warpPerspective(crop, H, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _shift(crop, vec):
    M = np.float32([[1, 0, vec[0]], [0, 1, vec[1]]])
    return cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def inject(crop, qc, ppm, p):
    """The crop with one fault of family p['fam'] (qc corners in crop px, ppm px per m on the tag):
    P one corner moved mag mm (projective warp fixing the other three); Od / Ob dark / bright disc of radius mag px
    at a corner or a side middle; E a dark line parallel to a side, mag px outward; B one inner cell of the code
    flipped; S the whole tag shifted mag mm; PS one corner moved mag[0] mm then the whole tag shifted mag[1] px."""
    fam, mag = p['fam'], p['mag']
    u = np.array([math.cos(p['huong']), math.sin(p['huong'])])
    i = p['goc']
    if fam == 'P':
        return _move_corner(crop, qc, i, mag * 1e-3 * ppm * u)
    if fam == 'S':
        return _shift(crop, mag * 1e-3 * ppm * u)
    if fam == 'PS':
        u2 = np.array([math.cos(p['huong2']), math.sin(p['huong2'])])
        return _shift(_move_corner(crop, qc, i, mag[0] * 1e-3 * ppm * u), mag[1] * u2)
    img = crop.astype(np.float32)
    mask = np.zeros(crop.shape, np.uint8)
    if fam in ('Od', 'Ob'):
        pt = 0.5 * (qc[i] + qc[(i + 1) % 4]) if p['giua'] else qc[i]
        cv2.circle(mask, tuple(int(v) for v in np.round(pt)), int(mag), 1, -1)
        return _blend(img, mask, 20.0 if fam == 'Od' else 235.0)
    if fam == 'E':
        p0, p1 = qc[i], qc[(i + 1) % 4]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([d[1], -d[0]])
        a = p0 + mag * nrm - 0.1 * (p1 - p0)
        b = p1 + mag * nrm + 0.1 * (p1 - p0)
        cv2.line(mask, tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b)), 1, 2)
        return _blend(img, mask, 20.0)
    if fam == 'B':
        mk, can = _marker()
        s = TPL_CELL
        H = cv2.getPerspectiveTransform(can, qc.astype(np.float32))
        r_, c_ = p['o']
        cell = np.array([[c_, r_], [c_ + 1, r_], [c_ + 1, r_ + 1], [c_, r_ + 1]], np.float32) * s - 0.5
        poly = cv2.perspectiveTransform(cell[None], H)[0]
        val = 235.0 if mk[r_ * s + s // 2, c_ * s + s // 2] < 128 else 20.0
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 1)
        return _blend(img, mask, val)
    raise DnError(f'family {fam}')


class Pose:
    """car_pose with the static session camera; injected poses are the track pose plus the static-camera change."""

    def __init__(self, sess):
        self.cam = cm.cam_from_dict(sess['camera'])
        self.side, self.kappa, self.plumb = float(sess['car_side_m']), sess['kappa'], sess['plumb_xy_m']

    def __call__(self, q):
        p = cm.car_pose(self.cam, q, self.side, self.kappa, self.plumb)
        return np.array([p['tag_x'], p['tag_y']]), p['tag_psi'], p['res_sim_m'], p['scale']


def _err(pose, p0, p1):
    return float(np.linalg.norm(p1[0] - p0[0])), abs(math.degrees(float(wrap(p1[1] - p0[1]))))


# ── khoa ─────────────────────────────────────────────────────────────────────────────────────────

def build(clips, An, side):
    """References, patterns (each clip's own), the gates (other clip's patterns), e_max, r_max; deterministic."""
    rs, dv, L = {}, {}, {}
    for name, C in clips.items():
        A = An[name]
        rs[name] = make_refset(A, ref_idx(C, A))
        dv[name] = devs(A, rs[name], side)
        try:
            L[name] = learn(A, dv[name])
        except DnError as e:
            raise DnError(f'{name}: {e}') from None
    G, th = {}, {}
    for name, C in clips.items():
        o = _other(name)
        A = An[name]
        ok = np.isfinite(dv[name]).all(axis=1) & np.isfinite(A['vt'])
        e = np.full(len(A['vt']), np.nan)
        e[ok] = e_hinh(dv[name][ok], A['vt'][ok], L[o]['kieu_lech'])
        ref_set = ok & C['valid_old'] & (A['vt'] <= KHOA_VT_MAX)
        if ref_set.sum() < 1000:
            raise DnError(f'{name}: {int(ref_set.sum())} old valid frames for e_max, need 1000')
        e_max = max(E_MAX_K * float(np.percentile(e[ref_set], E_Q)), E_MAX_FLOOR_M)
        oke = np.isfinite(A['X']).all(axis=1) & np.isfinite(A['vt'])
        r = np.full(len(A['vt']), np.nan)
        r[oke] = r_canh(A['X'][oke], A['vt'][oke], L[o]['kieu_canh'])
        rset = oke & C['valid_old'] & (A['vt'] <= KHOA_VT_MAX)
        r_max = E_MAX_K * float(np.percentile(r[rset], E_Q))
        # DN3-04 report: e_hinh with the pattern of the SAME clip
        e_own = np.full(len(A['vt']), np.nan)
        e_own[ok] = e_hinh(dv[name][ok], A['vt'][ok], L[name]['kieu_lech'])
        G[name] = Gate(A, rs[name], L[o]['kieu_lech'], L[o]['kieu_canh'], L[o]['mien_vt'], e_max, r_max, side)
        th[name] = dict(e_max_m=e_max, e_ref_n=int(ref_set.sum()), e_ref_p50_m=float(np.median(e[ref_set])),
                        e_ref_p999_m=float(np.percentile(e[ref_set], E_Q)), r_max=r_max, r_ref_n=int(rset.sum()),
                        r_ref_p999=float(np.percentile(r[rset], E_Q)), e_own=e_own)
    return rs, L, G, th


def _other(name):
    return 'c2' if name == 'c1' else 'c1'


def health(C, A, G):
    """v5 2.5: every old valid frame through (a)-(h): passes / fails ((a) included, DNC1-08) / cannot be scored (no
    v_t, v_t undefined with a neighbour left out (v5 7), no phase reference, outside the pattern range)."""
    cnt = Counter()
    fails = []
    for k in np.flatnonzero(C['valid_old']):
        if not A['qok'][k]:                                  # (a) fails on an old frame: counted as failing (DNC1-08)
            cnt['truot:' + (A['why_det'][k] or 'khong_do_lai')] += 1
            fails.append(int(k))
            continue
        why, _, _ = G.judge(k, A['shapes'][k], A['X'][k], A['F'][k], C['res_sim'][k])
        if why in ('khong_vt', 'khong_chuan_pha', 'ngoai_mien', 'vt_mat_khi_bo'):    # no v_t = cannot score (v5 7)
            cnt['khong_cham:' + why] += 1
        elif why:
            cnt['truot:' + why] += 1
            fails.append(int(k))
        else:
            cnt['qua'] += 1
    n_ch = sum(v for k, v in cnt.items() if not k.startswith('khong_cham'))
    n_tr = sum(v for k, v in cnt.items() if k.startswith('truot'))
    return dict(dem=dict(cnt), cham_duoc=n_ch, truot=n_tr, ti_le_truot=n_tr / n_ch if n_ch else None,
                dat=bool(n_ch and n_tr <= HEALTH_MAX * n_ch), truot_idx=fails[:200])


_P12_OK = {'ti_le', 'ti_le_it_lan_can'}


def p12_mask(rows):
    """Frames that passed (1) and (2a) in a gate run: the neighbour pool of (2b) (c7_measure.apply_gates, quant 0)."""
    return np.array([bool(r['valid']) or (bool(r['why']) and set(r['why']) <= _P12_OK) for r in rows])


def scale_check(C, p12, k, sc, gates):
    """(2b) of c7_measure.apply_gates (quant 0) for frame k with scale sc against the median scale of the frames of
    p12 within +-scale_win_s, k excluded. Returns (passes, why)."""
    t = C['t']
    m = p12 & (np.abs(t - t[k]) <= gates['scale_win_s'] + 1e-9)
    m[k] = False
    if m.sum() < gates['scale_min']:
        return False, 'ti_le_it_lan_can'
    return bool(abs(sc / float(np.median(C['scale'][m])) - 1.0) <= gates['scale_rel']), 'ti_le'


def scorable(C, A, G, gate_m):
    """Candidates of the old gate (fail (2a)) with a re-detection, a v_t in the fast band, a phase reference and the
    pattern range: the fast sample of the criterion (2) is drawn from these, independently of the verdict of the new
    gate (DNC1-04)."""
    ok = np.zeros(len(C['t']), bool)
    for k in np.flatnonzero(C['seen'] & A['qok'] & ~(C['res_sim'] <= gate_m)):
        v = A['vt'][k]
        if np.isfinite(v) and FAST_BAND[0] <= v <= FAST_BAND[1] and v <= G.vmax and \
                G.rs.ref(int(A['cell'][k]), A['ang'][k]) is not None:
            ok[k] = True
    return np.flatnonzero(ok)


def envelope(C, A, G, pose, idx, gate, gate_m, rng):
    """Largest centre / heading change a gate admits when one corner moves (24 directions x 4 corners, bisection up
    to ENV_MAX_MM), per frame (design v4 1.3; corners moved directly, no re-detection: this measures the shape part
    of the gate, (c)-(e) for the new one, res_sim for the old one). The sample is drawn before scoring; frames the
    gate rejects unmoved are counted and left out of the percentiles (DNC1-04)."""
    out_c, out_p, rej = [], [], []
    side = G.side
    samp = _pick(rng, idx, ENV_N)
    for k in samp:
        q0 = A['Q'][k]
        p0 = pose(q0)
        ppm = side_px(q0) / side

        def passes(q):
            if gate == 'old':
                return pose(q)[2] <= gate_m
            return G.shape_part(k, norm_shape(q), A['X'][k], A['vt'][k], A['cell'][k])[0] in ('', 'r_canh_vuot')

        if not passes(q0):
            rej.append(int(k))
            continue
        bc = bp = 0.0
        for i in range(4):
            for th in np.linspace(0, 2 * math.pi, ENV_DIRS, endpoint=False):
                u = np.array([math.cos(th), math.sin(th)])
                lo, hi = 0.0, ENV_MAX_MM
                for _ in range(ENV_BISECT):
                    mid = 0.5 * (lo + hi)
                    q = q0.copy()
                    q[i] += mid * 1e-3 * ppm * u
                    if passes(q):
                        lo = mid
                    else:
                        hi = mid
                q = q0.copy()
                q[i] += lo * 1e-3 * ppm * u
                ec, ep = _err(pose, p0, pose(q))
                bc, bp = max(bc, ec), max(bp, ep)
        out_c.append(bc)
        out_p.append(bp)
    c, p = np.array(out_c), np.array(out_p)
    if not len(c):
        return None
    return dict(n=len(c), n_mau=len(samp), khong_nhan_goc=rej, mau_idx=[int(k) for k in samp],
                tam_m=[float(np.percentile(c, 50)), float(np.percentile(c, 90)), float(c.max())],
                huong_deg=[float(np.percentile(p, 50)), float(np.percentile(p, 90)), float(p.max())])


def _pick(rng, idx, n):
    idx = np.asarray(idx)
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def _injected(src, pose, p):
    crop, off, q0, p0 = src
    ppm = side_px(q0) / pose.side
    q, dg = detect_diag(detector(), inject(crop, q0 - off, ppm, p))
    if q is None:
        return None, None, None
    q = q + off
    return q, dg, pose(q)


def score_old(C, k, src, pose, p, gates, p12a):
    """Old gate on a slow old valid frame, as cong treats a frame that is not taken: (2a) on the track residual
    moved by the static-camera change, (2b) against the run-A pool. lost ID / (2a) / (2b) / accepted."""
    p0 = src[3]
    q, _, p1 = _injected(src, pose, p)
    if q is None:
        return dict(tang='mat_id')
    ec, ep = _err(pose, p0, p1)
    out = dict(loi_tam_m=ec, loi_huong_deg=ep)
    if not C['res_sim'][k] + (p1[2] - p0[2]) <= gates['car_res_sim_m']:
        out['tang'] = 'truot_2a'
        return out
    ok, w = scale_check(C, p12a, k, C['scale'][k] + (p1[3] - p0[3]), gates)
    out['tang'] = 'nhan' if ok else 'truot_2b'
    return out


def score_new(C, A, G, k, src, pose, p, gates, p12a, p12b, tctx):
    """A fast candidate through the decision path of cong (DNC1-03): the injected residual and scale are the track
    values moved by the static-camera change; a residual that passes (2a) routes the frame as an old one ((2b)
    against the run-A pool); otherwise (b)-(h) of the new gate, then (2b) against the run-B pool. (a) does not apply
    (the injected image differs from the track by construction). The temporal check (i) is recorded on its own
    (qua_i, one pass with the pass-2 neighbour set N2)."""
    p0 = src[3]
    q, dg, p1 = _injected(src, pose, p)
    if q is None:
        return dict(tang='mat_id')
    ec, ep = _err(pose, p0, p1)
    rs = C['res_sim'][k] + (p1[2] - p0[2])
    sc = C['scale'][k] + (p1[3] - p0[3])
    out = dict(loi_tam_m=ec, loi_huong_deg=ep)
    cx, cy, R0, TH, N2 = tctx
    x, y, ps = C['x'].copy(), C['y'].copy(), C['psi'].copy()
    x[k] += p1[0][0] - p0[0][0]
    y[k] += p1[0][1] - p0[0][1]
    ps[k] += float(wrap(p1[1] - p0[1]))
    tr = temporal_one(C['t'], x, y, ps, np.flatnonzero(N2), k, (cx, cy), R0)
    out['qua_i'] = None if tr is None else not bool(temporal_fail(np.array([tr]), TH)[0][0])
    if rs <= gates['car_res_sim_m']:
        out['tuyen'] = 'cu'
        ok, _ = scale_check(C, p12a, k, sc, gates)
        out['tang'] = 'nhan' if ok else 'truot_2b'
        return out
    out['tuyen'] = 'moi'
    F4 = np.array([[d['kept'], d['rms_px'], d['contrast'], d['width_px']] for d in dg])
    why, e, r = G.judge(k, norm_shape(q), edge_vec(dg), F4, rs)
    out.update(e_hinh_m=e, r_canh=r)
    if why:
        out['tang'] = {'canh_co_ban': 'truot_canh', 'r_canh_vuot': 'truot_canh', 'vt_khong_ben': 'truot_vt',
                       'vt_mat_khi_bo': 'truot_vt', 'res_sim_vuot_tran': 'truot_tran'}.get(why, 'truot_hinh')
        out['ly_do'] = why
        return out
    ok, _ = scale_check(C, p12b, k, sc, gates)
    out['tang'] = 'nhan' if ok else 'truot_2b'
    return out


def _bad(r, kc=1.0, kp=1.0):
    return r.get('loi_tam_m') is not None and (r['loi_tam_m'] > kc * BAD_C_M or r['loi_huong_deg'] > kp * BAD_P_DEG)


def injection(C, A, G, cache, pose, fast_pool, gates, p12a, p12b, tctx, rng):
    """Pairs (slow old valid frame, fast scorable candidate) of the same predicted phase cell, each slow frame used
    once; the same fault (family, corner, direction, size) on both; old gate on the slow frame, decision path of cong
    on the fast one. The fast pool does not depend on the verdict of the new gate (DNC1-04)."""
    slow = np.flatnonzero(C['valid_old'] & A['qok'] & np.isfinite(A['vt']) & (A['vt'] <= SLOW_VT) & (A['cell'] >= 0))
    used = set()
    pairs = []
    for k in rng.permutation(fast_pool):
        s = [int(v) for v in slow[A['cell'][slow] == A['cell'][k]] if int(v) not in used]
        if s:
            ks = int(rng.choice(s))
            used.add(ks)
            pairs.append((ks, int(k)))
        if len(pairs) >= INJ_PAIRS:
            break
    rows = []
    for ks, kf in pairs:
        src = {}
        for kk in (ks, kf):
            j = cache['pos'][kk]
            off = np.array([cache['x0'][j], cache['y0'][j]], float)
            src[kk] = (cache['crops'][j], off, A['Q'][kk], pose(A['Q'][kk]))
        for fam, mag in FAMILIES:
            p = draw_params(fam, mag, rng)
            o = score_old(C, ks, src[ks], pose, p, gates, p12a)
            nw = score_new(C, A, G, kf, src[kf], pose, p, gates, p12a, p12b, tctx)
            rows.append(dict(ho=fam, do_lon=mag, cham=ks, nhanh=kf, o_pha=int(A['cell'][kf]),
                             goc=p['goc'], huong_rad=p['huong'], cu=o, moi=nw))
    return pairs, rows


def inj_table(rows):
    """Per family x size: accepted-wrong counts, largest accepted errors, rates over the faults that really do harm,
    sensitivity to the 0.5 mm / 0.25 deg convention (x 0.5, x 1.5), the layers, the paired view. dieu_kien false for
    REPORT_FAMS (report only, design v5 7)."""
    tab = {}
    for fam, mag in FAMILIES:
        R = [r for r in rows if r['ho'] == fam and r['do_lon'] == mag]
        key = f'{fam}_{mag if not isinstance(mag, tuple) else "_".join(str(v) for v in mag)}'
        d = dict(n=len(R))
        for g in ('cu', 'moi'):
            acc = [r[g] for r in R if r[g]['tang'] == 'nhan']
            bad = [r[g] for r in R if _bad(r[g])]
            d[g] = dict(nhan=len(acc), nhan_sai=sum(1 for x in acc if _bad(x)),
                        nhan_sai_k05=sum(1 for x in acc if _bad(x, 0.5, 0.5)),
                        nhan_sai_k15=sum(1 for x in acc if _bad(x, 1.5, 1.5)),
                        sai_that=len(bad), ti_le_nhan_tren_sai_that=(sum(1 for x in bad if x['tang'] == 'nhan') /
                                                                       len(bad)) if bad else None,
                        max_tam_nhan_m=max((x['loi_tam_m'] for x in acc), default=0.0),
                        max_huong_nhan_deg=max((x['loi_huong_deg'] for x in acc), default=0.0),
                        tang=dict(Counter(r[g]['tang'] for r in R)))
        acc_i = [r['moi'] for r in R if r['moi']['tang'] == 'nhan' and r['moi'].get('qua_i')]
        d['moi_co_i'] = dict(nhan=len(acc_i), nhan_sai=sum(1 for x in acc_i if _bad(x)),
                             max_tam_nhan_m=max((x['loi_tam_m'] for x in acc_i), default=0.0),
                             max_huong_nhan_deg=max((x['loi_huong_deg'] for x in acc_i), default=0.0))
        why = []
        d['dieu_kien'] = fam not in REPORT_FAMS
        if d['moi']['nhan_sai'] > d['cu']['nhan_sai']:
            why.append('nhan_sai')
        # paired view: pairs both gates accept, centre error new minus old; wrong faults only one gate accepts
        both = [(r['cu']['loi_tam_m'], r['moi']['loi_tam_m']) for r in R
                if r['cu']['tang'] == 'nhan' and r['moi']['tang'] == 'nhan']
        d['theo_cap'] = dict(ca_hai_nhan=len(both), hieu_tam_m=_stat(np.array([b - a for a, b in both])),
                             chi_moi_nhan_sai=sum(1 for r in R if r['moi']['tang'] == 'nhan' and
                                                  r['cu']['tang'] != 'nhan' and _bad(r['moi'])),
                             chi_cu_nhan_sai=sum(1 for r in R if r['cu']['tang'] == 'nhan' and
                                                 r['moi']['tang'] != 'nhan' and _bad(r['cu'])))
        if d['theo_cap']['chi_moi_nhan_sai'] > d['theo_cap']['chi_cu_nhan_sai']:
            why.append('chi_moi_nhan_sai')
        d['bao_cao_max'] = dict(tam_vuot=d['moi']['max_tam_nhan_m'] > d['cu']['max_tam_nhan_m'],
                                huong_vuot=d['moi']['max_huong_nhan_deg'] > d['cu']['max_huong_nhan_deg'])
        d['dat'], d['ly_do'] = not why, why
        tab[key] = d
    return tab


def cluster_report(C, A, take, centre, R0, TH, N2, rng):
    """DN5-01 (report only, a limitation): a run of L frames of N with consecutive idx (no lost frame inside, DNC1-11)
    shifted together by d mm (one direction) on the pose chain; share of the run frames the temporal check (i)
    catches."""
    nidx = np.flatnonzero(N2)
    starts = np.flatnonzero(take & (A['vt'] >= FAST_BAND[0]) & (A['vt'] <= FAST_BAND[1]))
    out = []
    for L in CLUSTER_L:
        for d in CLUSTER_MM:
            caught, tot = 0, 0
            for k0 in _pick(rng, starts, CLUSTER_N):
                j0 = int(np.searchsorted(nidx, k0))
                run = nidx[j0:j0 + L]
                if len(run) < L or run[-1] - run[0] != L - 1:
                    continue
                th = float(rng.uniform(0, 2 * math.pi))
                x, y = C['x'].copy(), C['y'].copy()
                x[run] += d * 1e-3 * math.cos(th)
                y[run] += d * 1e-3 * math.sin(th)
                for k in run:
                    r = temporal_one(C['t'], x, y, C['psi'], nidx, int(k), centre, R0)
                    if r is None:
                        continue
                    tot += 1
                    caught += int(temporal_fail(np.array([r]), TH)[0][0])
            out.append(dict(L=L, mm=d, khung=tot, bi_bat=caught, ti_le=caught / tot if tot else None))
    return out


# ── second reading: blurred template fit (design v2 2.2; report) ─────────────────────────────────

_TPL = None


def template():
    """(canvas float32, canonical corners): marker ID 0, 6 x 6 cells, white border of one cell; corners of the black
    square in ArUco order, pixel centres."""
    global _TPL
    if _TPL is None:
        s = TPL_CELL
        mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cp.DICT), CAR_ID, 6 * s)
        canvas = np.full((8 * s, 8 * s), 255, np.uint8)
        canvas[s:7 * s, s:7 * s] = mk
        can = np.array([[s, s], [7 * s, s], [7 * s, 7 * s], [s, 7 * s]], np.float32) - 0.5
        _TPL = (canvas.astype(np.float32), can)
    return _TPL


def _rot(q, axis, th):
    c, s = math.cos(th), math.sin(th)
    d = q - np.asarray(axis[:2])
    return np.column_stack([axis[0] + c * d[:, 0] - s * d[:, 1], axis[1] + s * d[:, 0] + c * d[:, 1]])


def render(quad, axis, om, T, sig, shape, off):
    """Template seen through the quad, blurred: mean of N_BLUR copies turned about the image axis by om T (k/(N-1) -
    1/2), then a Gaussian of sigma px."""
    canvas, can = template()
    acc = np.zeros(shape, np.float32)
    n = N_BLUR if om * T != 0.0 else 1
    for i in range(n):
        th = om * T * (i / (n - 1) - 0.5) if n > 1 else 0.0
        q = (_rot(quad, axis, th) - off).astype(np.float32)
        H = cv2.getPerspectiveTransform(can, q)
        acc += cv2.warpPerspective(canvas, H, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=255.0)
    acc /= n
    return cv2.GaussianBlur(acc, (0, 0), sig) if sig > 0 else acc


def fit_template(crop, off, q0, axis, om, T, sig):
    """Blurred-template fit of the 4 corners started from q0 (gain and offset solved linearly at every step); mask:
    the canvas area of q0 widened by MASK_DILATE_PX. Returns (corners, corr = sqrt(R^2), success)."""
    canvas, can = template()
    patch = crop.astype(np.float32)
    H0 = cv2.getPerspectiveTransform(can, (q0 - off).astype(np.float32))
    m = cv2.warpPerspective(np.ones_like(canvas), H0, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_NEAREST)
    m = cv2.dilate((m > 0.5).astype(np.uint8), np.ones((2 * MASK_DILATE_PX + 1,) * 2, np.uint8)) > 0
    p = patch[m]
    pc = p - p.mean()
    if pc @ pc <= 0:
        return None, 0.0, False

    def res(z):
        r = render(q0 + z.reshape(4, 2), axis, om, T, sig, patch.shape, off)[m]
        A = np.column_stack([r, np.ones_like(r)])
        coef, *_ = np.linalg.lstsq(A, p, rcond=None)
        return A @ coef - p
    try:
        s = least_squares(res, np.zeros(8), diff_step=0.05, method='lm', max_nfev=400)
    except ValueError:
        return None, 0.0, False
    r2 = 1.0 - float(s.fun @ s.fun) / float(pc @ pc)
    return q0 + s.x.reshape(4, 2), math.sqrt(max(r2, 0.0)), bool(s.success)


def compare_reads(q_e, q_t, axis, om, side_m):
    """e_goc (m, largest corner gap) and template minus edge reading: centre tangential (along the motion), radial
    (m), heading (rad, image)."""
    sc = side_m / side_px(q_e)
    e = float(np.max(np.linalg.norm(q_t - q_e, axis=1))) * sc
    c_e = q_e.mean(axis=0)
    r = c_e - np.asarray(axis[:2])
    r = r / np.linalg.norm(r)
    tg = np.array([-r[1], r[0]]) * (1.0 if om >= 0 else -1.0)
    d = q_t.mean(axis=0) - c_e
    dh = float(wrap(quad_heading(q_t) - quad_heading(q_e)))
    return e, float(d @ tg) * sc, float(d @ r) * sc, dh


def read_two(cache, A, k, T, sig, side_m):
    j = cache['pos'][int(k)]
    off = np.array([cache['x0'][j], cache['y0'][j]], float)
    qt, corr, ok = fit_template(cache['crops'][j], off, A['Q'][k], A['axis'], A['om'][k], T, sig)
    if qt is None:
        return dict(ok=False)
    e, dt, dr, dh = compare_reads(A['Q'][k], qt, A['axis'], A['om'][k], side_m)
    return dict(ok=ok, corr=corr, e_goc_m=e, d_tiep_m=dt, d_phap_m=dr, d_huong_rad=dh)


def _spread(idx, n):
    idx = np.asarray(idx)
    return idx if len(idx) <= n else idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]


def template_params(C, A, cache, side_m):
    """(T, sigma) on the joint grid T_GRID_S x SIG_GRID maximising the mean corr over one sample of old valid frames
    (design v2 2.2, DNC1-13): SIG_N slowest plus T_N spread over T_BAND; the locked sample of the second reading."""
    ov = C['valid_old'] & A['qok'] & np.isfinite(A['vt'])
    slow = np.flatnonzero(ov)[np.argsort(A['vt'][ov])][:SIG_N]
    tb = _spread(np.flatnonzero(ov & (A['vt'] >= T_BAND[0]) & (A['vt'] <= T_BAND[1])), T_N)
    mau = np.unique(np.r_[slow, tb]).astype(int)
    tab = []
    for T in T_GRID_S:
        for sg in SIG_GRID:
            cs = [read_two(cache, A, k, T, sg, side_m).get('corr', np.nan) for k in mau]
            tab.append((T, sg, float(np.nanmean(cs))))
    T, sig, _ = max(tab, key=lambda p: p[2])
    samp = np.flatnonzero(A['qok'] & np.isfinite(A['vt']) & (A['vt'] >= DOC2_BAND[0]) &
                          (A['vt'] < DOC2_BAND[1]))[::DOC2_STRIDE]
    return dict(sigma_px=sig, T_s=T, bang_T_sigma=tab, mau_tham_so_idx=mau.tolist(), mau_idx=samp.tolist())


def load_all(a, need_hash=None):
    cu = read_cu(a.cu)
    sess, ssha = load_session(a.session)
    clips, caches, An = {}, {}, {}
    for name, (fp, cache_p) in (('c1', a.c1), ('c2', a.c2)):
        C = load_clip(fp)
        if need_hash is not None:
            K = need_hash['clips'][name]
            if C['sha256'] != K['frames_sha256']:
                raise DnError(f'{fp}: not the frames.csv of khoa.json {name}')
        cache = load_cache(cache_p, C)
        if need_hash is not None and cache['meta']['npz_sha256'] != need_hash['clips'][name]['cache_npz_sha256']:
            raise DnError(f'{cache_p}: not the crops of khoa.json {name}')
        clips[name], caches[name] = C, cache
    side = float(sess['car_side_m'])
    for name, C in clips.items():
        An[name] = analyse_clip(C, caches[name], cu['R0_m'])
    return cu, sess, ssha, side, clips, caches, An


def cmd_khoa(a):
    low_priority()
    cu, sess, ssha, side, clips, caches, An = load_all(a)
    gates = dict(cm.DEFAULT_GATES)
    G0 = gates['car_res_sim_m']
    rs, L, G, th = build(clips, An, side)
    pose = Pose(sess)
    out = dict(version=VERSION, session=a.session, session_sha256=ssha, cu=a.cu, cu_sha256=cu['sha256'],
               gate_2a_m=G0, gates=gates, hang_so=_constants(), clips={})
    du = True
    for ci, (name, C) in enumerate(clips.items()):
        A, Gn = An[name], G[name]
        take, info = decide(C, A, Gn, G0)
        _, ra, rb, _ = combine(C['rows'], take, sess['rms_px_H'], gates)
        p12a, p12b = p12_mask(ra), p12_mask(rb)
        r2, qua, ok2, TH, N2 = temporal_check(C, C['valid_old'], take, cu['centre'], cu['R0_m'])
        tctx = (cu['centre'][0], cu['centre'][1], cu['R0_m'], TH, N2)
        H = health(C, A, Gn)
        # one generator per use, so a report-only change cannot move a hard sample (DNC1-04)
        rng_env, rng_inj, rng_clu = (np.random.default_rng([INJ_SEED, ci, u]) for u in (1, 2, 3))
        slow_idx = np.flatnonzero(C['valid_old'] & A['qok'] & np.isfinite(A['vt']) & (A['vt'] <= SLOW_VT))
        fast_pool = scorable(C, A, Gn, G0)
        env_old = envelope(C, A, Gn, pose, slow_idx, 'old', G0, rng_env)
        env_new = envelope(C, A, Gn, pose, fast_pool, 'new', G0, rng_env)
        pairs, inj = injection(C, A, Gn, caches[name], pose, fast_pool, gates, p12a, p12b, tctx, rng_inj)
        tab = inj_table(inj)
        clus = cluster_report(C, A, take, cu['centre'], cu['R0_m'], TH, N2, rng_clu)
        tp = template_params(C, A, caches[name], side)
        why = []
        t = th[name]
        if t['e_max_m'] > E_MAX_CEIL_M:
            why.append(f'e_max {t["e_max_m"] * 1e3:.3f} mm > {E_MAX_CEIL_M * 1e3:.0f} mm')
        if t['r_max'] > R_MAX_CEIL:
            why.append(f'r_max {t["r_max"]:.3f} > {R_MAX_CEIL}')
        if env_old is None or env_new is None:
            why.append('bao: khong du khung')
        else:
            for key in ('tam_m', 'huong_deg'):
                if any(n > o for n, o in zip(env_new[key], env_old[key])):
                    why.append(f'bao {key}: moi {env_new[key]} > cu {env_old[key]}')
        if len(pairs) < INJ_PAIRS:
            why.append(f'bom loi: {len(pairs)} cap < {INJ_PAIRS}')
        for key, d in tab.items():
            if d['dieu_kien'] and not d['dat']:
                why.append(f'bom loi {key}: {d["ly_do"]}')
        if not H['dat']:
            why.append(f'suc khoe: truot {H["truot"]} / {H["cham_duoc"]}')
        du = du and not why
        cm_meta = caches[name]['meta']
        out['clips'][name] = dict(
            frames=C['path'], frames_sha256=C['sha256'], cache_npz_sha256=cm_meta['npz_sha256'],
            video_sha256=cm_meta['video_sha256'], axis_px=list(A['axis']),
            n_do_lai=int(A['qok'].sum()), do_lai_loi=dict(Counter(w for w in A['why_det'] if w)),
            chuan_pha_idx=rs[name].idx.tolist(), kieu_lech=L[name]['kieu_lech'], kieu_canh=L[name]['kieu_canh'],
            mien_vt=L[name]['mien_vt'], n_hoc=L[name]['n_hoc'], n_hoc_canh=L[name]['n_hoc_canh'],
            dung_kieu_cua=_other(name),
            **{k: v for k, v in t.items() if k != 'e_own'},
            nguong_thoi_gian=TH, suc_khoe=H, bao_cu_cham=env_old, bao_moi_nhanh=env_new,
            bom_loi=dict(cap=pairs, n_ung_vien_nhanh=len(fast_pool), bang=tab, chi_tiet=inj), ho_cum=clus,
            doc_hai=tp,
            du_doan_nhan=int(take.sum()), du_doan_ly_do=dict(Counter(d.get('ly_do') for d in info
                                                                       if d.get('cong') == 'ung_vien')),
            du_nhay=not why, ly_do=why)
        print(f'khoa {name}: e_max {t["e_max_m"] * 1e3:.3f} mm, r_max {t["r_max"]:.3f}; bao cu {env_old} moi {env_new}; '
              f'suc khoe {H["truot"]}/{H["cham_duoc"]}; cap {len(pairs)}; du_nhay {not why} {why}')
    out['du_nhay'] = du
    out['script'] = script_state()
    write_json_new(a.out, out)
    if not du:
        print('khoa: tieu chi (2) KHONG dat; cong se tu choi')


def _constants():
    return dict(ROI_HALF=ROI_HALF, MATCH_TOL_PX=MATCH_TOL_PX, VT=[VT_WIN_S, VT_SIDE_MIN, VT_GAP_S, VT_STEP_MAX,
                                                                   VT_SIDE_TOL],
                SLOW_VT=SLOW_VT, N_PHASE_REF=N_PHASE_REF, PHASE_MIN=PHASE_MIN, HUBER=[HUBER_IT, HUBER_K],
                PAT_RANGE_Q=PAT_RANGE_Q, KHOA_VT_MAX=KHOA_VT_MAX, E_MAX=[E_MAX_K, E_Q, E_MAX_FLOOR_M, E_MAX_CEIL_M],
                KEPT_MIN=KEPT_MIN, RMS_MAX_PX=RMS_MAX_PX, R_MAX_CEIL=R_MAX_CEIL, RES_SIM_CAP_M=RES_SIM_CAP_M,
                QUANT=QUANT, T=[T_WIN_S, T_SIDE, T_GAP_S, T_K, T_Q, T_BANDS, T_V_CAP], INJ_SEED=INJ_SEED,
                INJ_PAIRS=INJ_PAIRS, FAST_BAND=FAST_BAND, FAMILIES=FAMILIES, BAD=[BAD_C_M, BAD_P_DEG],
                ENV=[ENV_N, ENV_DIRS, ENV_BISECT, ENV_MAX_MM],
                HEALTH_MAX=HEALTH_MAX, CLUSTER=[CLUSTER_L, CLUSTER_MM, CLUSTER_N],
                DOC2=[SIG_GRID, SIG_N, T_GRID_S, T_N, T_BAND, N_BLUR, MASK_DILATE_PX, TPL_CELL, DOC2_BAND,
                      DOC2_STRIDE])


# ── cong ─────────────────────────────────────────────────────────────────────────────────────────

def check_khoa(KH, name, rs, L, th, A):
    """The numbers cong recomputes must equal those locked in khoa.json (the same code on the same inputs)."""
    K = KH['clips'][name]
    bad = []
    if not _close(A['axis'], K['axis_px'], 1e-9):
        bad.append('axis')
    if rs.idx.tolist() != K['chuan_pha_idx']:
        bad.append('chuan_pha_idx')
    for k in ('kieu_lech', 'kieu_canh', 'mien_vt'):
        if not _close(L[k], K[k]):
            bad.append(k)
    for k in ('e_max_m', 'r_max'):
        if not _close(th[k], K[k]):
            bad.append(k)
    if bad:
        raise DnError(f'{name}: recomputed {bad} differ from khoa.json')


def scale_nb_report(C, take, rb, gates):
    """DN3-07 (report): for every frame taken back, its (2b) neighbours in the run-B pool (DNC1-11) split into old
    (the track pool) and new (taken back), binned 0 / 1-7 / >= 8 old; (2b) with the old neighbours only where >= 8."""
    pa = p12_mask(C['rows'])
    pb = p12_mask(rb)
    t = C['t']
    cnt, bins = Counter(), Counter()
    n_old, n_new = [], []
    for k in np.flatnonzero(take):
        m = pb & (np.abs(t - t[k]) <= gates['scale_win_s'] + 1e-9)
        m[k] = False
        o, w = np.flatnonzero(m & pa), np.flatnonzero(m & ~pa)
        n_old.append(len(o))
        n_new.append(len(w))
        bins['0' if not len(o) else ('1-7' if len(o) < 8 else '>=8')] += 1
        if len(o) >= 8:
            ok = abs(C['scale'][k] / float(np.median(C['scale'][o])) - 1.0) <= gates['scale_rel']
            cnt['chi_cu_dat' if ok else 'chi_cu_truot'] += 1
    return dict(lang_gieng_cu=_stat(np.array(n_old, float)), lang_gieng_moi=_stat(np.array(n_new, float)),
                so_lang_gieng_cu=dict(bins), ti_le_chi_lang_gieng_cu=dict(cnt))


def same_clip_report(C, A, info, e_own, e_max):
    """DN3-04 (report, DNC1-10): candidates scored by both patterns (the other clip's, used, and this clip's), by v_t
    band: (e) verdict with each pattern and how many differ; candidates outside the range of the other clip's pattern
    counted apart. Only the shape condition (e) is compared, not the whole gate."""
    out = dict(ngoai_mien=0, theo_dai=[])
    rows = []
    for k in np.flatnonzero(C['seen']):
        d = info[k]
        if d.get('cong') != 'ung_vien':
            continue
        if d.get('ly_do') == 'ngoai_mien':
            out['ngoai_mien'] += 1
            continue
        if d.get('e_hinh_m') is None or not np.isfinite(e_own[k]):
            continue
        rows.append((A['vt'][k], d['e_hinh_m'] <= e_max, e_own[k] <= e_max))
    for lo, hi in zip(BANDS_VT[:-1], BANDS_VT[1:]):
        r = [x for x in rows if lo <= x[0] < hi]
        if r:
            out['theo_dai'].append(dict(dai=f'{lo}-{hi}', n=len(r), qua_kieu_clip_kia=sum(x[1] for x in r),
                                        qua_kieu_cung_clip=sum(x[2] for x in r), khac=sum(x[1] != x[2] for x in r)))
    return out


def long_runs(C, take):
    """DN4-07: runs of frames taken back with consecutive idx (a lost or old frame ends a run, DNC1-11), longer than
    RUN_LONG."""
    runs = _runs(take)
    long_ = [(a, b - 1, b - a) for a, b in runs if b - a > RUN_LONG]
    return dict(so_cum=len(long_), so_khung=int(sum(r[2] for r in long_)), dai_nhat=max((r[2] for r in long_), default=0),
                cum=long_[:100])


def rows_csv(path, C, out, info, take, ra, rb, K, tres, tqua, tok, doc2):
    cols = ('clip_idx', 't_s', 'v_t', 'o_pha', 'cong', 'ly_do', 'e_hinh_m', 'e_max_m', 'r_canh', 'r_max', 'nhan_lai',
            'valid_cu', 'why_cu', 'valid_A', 'valid_B', 'valid_moi', 'why_moi', 'du_tiep_m', 'du_phap_m',
            'du_huong_rad', 'v_thoi_gian', 'qua_thoi_gian', 'doc_hai')
    with open(path, 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(len(out)):
            if not C['seen'][k]:
                continue
            d = info[k]
            q = '' if not tok[k] else int(tqua[k])
            vals = [k, f'{C["t"][k]:.6f}', d.get('v_t'), d.get('o_pha'), d.get('cong'), d.get('ly_do'),
                    d.get('e_hinh_m'), K['e_max_m'], d.get('r_canh'), K['r_max'], int(take[k]),
                    int(C['valid_old'][k]), '|'.join(C['why_old'][k]), int(ra[k]['valid']), int(rb[k]['valid']),
                    int(out[k]['valid']), '|'.join(out[k]['why']),
                    *(float(v) if np.isfinite(v) else None for v in tres[k]), q, int(k in doc2)]
            w.writerow(['' if v is None else (f'{v:.9g}' if isinstance(v, float) else v) for v in vals])


def _c8b(a, out_dir, fg):
    tb_out = os.path.join(out_dir, 'c8b_3b_A2_B.json')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'c8b_3b.py'), '--frames-a', a.frames_a, '--frames-b', fg,
                        '--out', tb_out], capture_output=True, text=True)
    with open(os.path.join(out_dir, 'c8b_3b_stdout.txt'), 'x', encoding='utf-8') as f:
        f.write(p.stdout + p.stderr)
    res = dict(ma=p.returncode, cuoi=p.stdout.strip().splitlines()[-1] if p.stdout.strip() else '')
    if p.returncode != 0 or not os.path.exists(tb_out):
        raise DnError(f'c8b_3b failed on {fg} (code {p.returncode}); see {out_dir}/c8b_3b_stdout.txt')
    with open(tb_out, encoding='utf-8') as f:
        J = json.load(f)
    B = J['B']
    res.update(du_phu=B.get('du_phu'), phu=B.get('phu'), rad_dev_m=B.get('rad_dev_m'),
               hp_dev_deg=B.get('hp_dev_deg'), ga_dev_m=B.get('ga_dev_m'), ga_ok=J.get('ga_ok'),
               thieu=J.get('thieu'), alpha=B.get('shutter', {}).get('use_alpha'),
               alpha_why=B.get('shutter', {}).get('why'), v_t_max_m_s=B.get('v_t_max_m_s'))
    return res


def _join(a, d, outs, ten):
    gdir = os.path.join(d, 'B_ghep')
    subprocess.run([sys.executable, '-I', a.ghep, '--c1', outs['c1'], '--c2', outs['c2'], '--out', gdir,
                    '--ten', ten], check=True)
    _check_join(outs['c1'], outs['c2'], os.path.join(gdir, 'frames.csv'))
    return os.path.join(gdir, 'frames.csv')


def cmd_cong(a):
    low_priority()
    with open(a.khoa, encoding='utf-8') as f:
        KH = json.load(f)
    if KH.get('version') != VERSION:
        raise DnError(f'{a.khoa}: version {KH.get("version")} != {VERSION}')
    if not KH.get('du_nhay'):
        raise DnError(f'{a.khoa}: criterion (2) not met (du_nhay false): refused')
    # DNC1-09: the flag must agree with the clip verdicts and the code that scored them must be this code
    if sorted(KH.get('clips', {})) != ['c1', 'c2'] or not all(
            c.get('du_nhay') is True and c.get('ly_do') == [] for c in KH['clips'].values()):
        raise DnError(f'{a.khoa}: du_nhay true but a clip verdict is not met: refused')
    me = script_state()
    ks = KH.get('script') or {}
    if ks.get('sha256') != me['sha256'] or ks.get('c7_measure_sha256') != me['c7_measure_sha256'] or ks.get('dirty'):
        raise DnError(f'{a.khoa}: made by another code (sha256 of c8_doc_nhoe / c7_measure) or a dirty tree: refused')
    cu = read_cu(a.cu)
    if KH['cu_sha256'] != cu['sha256']:
        raise DnError('khoa.json was made with another old 3b result')
    sess, ssha = load_session(a.session)
    if KH['session_sha256'] != ssha:
        raise DnError('khoa.json was made with another session')
    gates = dict(cm.DEFAULT_GATES)
    G0 = gates['car_res_sim_m']
    if abs(G0 - KH['gate_2a_m']) > 1e-12 or _clean(gates) != KH['gates']:
        raise DnError('gates differ from khoa.json')
    if os.path.exists(a.out):
        raise DnError(f'{a.out} exists')
    cu, sess, ssha, side, clips, caches, An = load_all(a, need_hash=KH)
    rs, L, G, th = build(clips, An, side)
    for name in clips:
        check_khoa(KH, name, rs[name], L[name], th[name], An[name])
    os.makedirs(a.out, exist_ok=False)
    shutil.copyfile(a.khoa, os.path.join(a.out, 'khoa.json'))
    vdir = os.path.join(a.out, 'bien_the_thoi_gian')
    os.makedirs(vdir, exist_ok=False)
    summary = dict(version=VERSION, khoa=a.khoa, khoa_sha256=sha256(a.khoa), session=a.session, session_sha256=ssha,
                   cu=a.cu, cu_sha256=cu['sha256'], gates=gates, quant=QUANT, clips={})
    outs, outs_v = {}, {}
    for name, C in clips.items():
        A, K = An[name], KH['clips'][name]
        take, info = decide(C, A, G[name], G0)
        out, ra, rb, flips = combine(C['rows'], take, sess['rms_px_H'], gates)
        same_a = sum(1 for x, y in zip(ra, C['rows']) if x['valid'] != y['valid'] or x['why'] != y['why'])
        if same_a:
            raise DnError(f'{name}: run A differs from the track on {same_a} frames')
        tres, tqua, tok, TH, N2 = temporal_check(C, C['valid_old'], take, cu['centre'], cu['R0_m'])
        if not _close([TH['a_m'], TH['b_s'], TH['T_phap_m'], TH['T_huong_rad']],
                      [K['nguong_thoi_gian'][k] for k in ('a_m', 'b_s', 'T_phap_m', 'T_huong_rad')]):
            raise DnError(f'{name}: temporal thresholds differ from khoa.json')
        # second reading on the locked sample
        samp = K['doc_hai']['mau_idx']
        doc2 = {}
        for k in samp:
            doc2[int(k)] = read_two(caches[name], A, int(k), K['doc_hai']['T_s'], K['doc_hai']['sigma_px'], side)
        d = os.path.join(a.out, f'B_{name}_3b')
        os.makedirs(d, exist_ok=False)
        fpath = os.path.join(d, 'frames.csv')
        cm.write_frames(fpath, out)
        rows_csv(os.path.join(d, 'doc_nhoe_rows.csv'), C, out, info, take, ra, rb, K, tres, tqua, tok, doc2)
        with open(os.path.join(d, 'doc_hai_rows.csv'), 'x', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(('clip_idx', 'v_t', 'nhom', 'ok', 'corr', 'e_goc_m', 'd_tiep_m', 'd_phap_m', 'd_huong_rad'))
            for k, r in sorted(doc2.items()):
                g = 'cu' if C['valid_old'][k] else ('moi' if take[k] else 'khong_nhan')
                w.writerow([k, f'{A["vt"][k]:.6f}', g, int(r.get('ok', False))] +
                           [('' if r.get(c) is None else f'{r[c]:.9g}') for c in
                            ('corr', 'e_goc_m', 'd_tiep_m', 'd_phap_m', 'd_huong_rad')])
        # variant with (i) as a gate (report)
        take_i = take & tok & tqua
        out_i, _, _, _ = combine(C['rows'], take_i, sess['rms_px_H'], gates)
        dv = os.path.join(vdir, f'B_{name}_3b')
        os.makedirs(dv, exist_ok=False)
        cm.write_frames(os.path.join(dv, 'frames.csv'), out_i)
        vnew = np.array([r['valid'] for r in out])
        vnew_i = np.array([r['valid'] for r in out_i])
        cand = [info[k] for k in range(len(out)) if info[k].get('cong') == 'ung_vien']
        old_t = C['valid_old'] & tok
        e_own = th[name]['e_own']
        js = dict(clip=name, frames_in=C['path'], frames_in_sha256=C['sha256'], frames_out_sha256=sha256(fpath),
                  kieu_lech_tu=_other(name), e_max_m=K['e_max_m'], r_max=K['r_max'], n_frames=len(out),
                  n_seen=int(C['seen'].sum()), n_valid_cu=int(C['valid_old'].sum()), n_valid_moi=int(vnew.sum()),
                  chuyen=dict(Counter(f'{int(C["valid_old"][k])}->{int(vnew[k])}' for k in range(len(out)))),
                  why_moi=dict(Counter('|'.join(r['why']) for r in out)),
                  ung_vien=len(cand), ly_do_khong_nhan=dict(Counter(x.get('ly_do') for x in cand if x.get('ly_do'))),
                  nhan_lai=int(take.sum()), nhan_lai_hop_le=int((take & vnew).sum()),
                  lan_A_khac_track=same_a, khung_cu_lan_B_doi=len(flips), khung_cu_lan_B_doi_idx=flips[:200],
                  thoi_gian=dict(nguong=TH, nhan_lai_co_du=int((take & tok).sum()),
                                 nhan_lai_truot_i=int((take & tok & ~tqua).sum()),
                                 nhan_lai_khong_tinh_i=int((take & ~tok).sum()),
                                 cu_truot_i_chi_bao=int((old_t & ~tqua).sum()), cu_co_du=int(old_t.sum()),
                                 cu_truot_i_idx=np.flatnonzero(old_t & ~tqua).tolist()[:200],
                                 bien_the_n_valid=int(vnew_i.sum())),
                  lang_gieng_2b=scale_nb_report(C, take, rb, gates), cum_dai=long_runs(C, take),
                  kieu_cung_clip=same_clip_report(C, A, info, e_own, K['e_max_m']),
                  doc_hai=dict(n=len(doc2), ok=sum(1 for r in doc2.values() if r.get('ok'))))
        write_json_new(os.path.join(d, 'doc_nhoe.json'), dict(js, script=script_state()))
        summary['clips'][name] = js
        outs[name], outs_v[name] = fpath, os.path.join(dv, 'frames.csv')
        print(f'cong {name}: hop le {js["n_valid_cu"]} -> {js["n_valid_moi"]} (bien the co (i) {int(vnew_i.sum())}); '
              f'nhan lai {js["nhan_lai"]}; khong nhan {js["ly_do_khong_nhan"]}; khung cu lan B doi {len(flips)}; '
              f'(i): nhan lai truot {js["thoi_gian"]["nhan_lai_truot_i"]}, cu truot (chi bao) '
              f'{js["thoi_gian"]["cu_truot_i_chi_bao"]}')
    fg = _join(a, a.out, outs, 'B doc nhoe (c8_doc_nhoe v5) IMG_3601 roi IMG_3602')
    summary['c8b_3b'] = _c8b(a, a.out, fg)
    fgv = _join(a, vdir, outs_v, 'B doc nhoe (c8_doc_nhoe v5, bien the co kiem thoi gian (i)) IMG_3601 roi IMG_3602')
    summary['c8b_3b_bien_the_thoi_gian'] = _c8b(a, vdir, fgv)
    summary['script'] = script_state()
    write_json_new(os.path.join(a.out, 'doc_nhoe_tong.json'), summary)
    for k in ('c8b_3b', 'c8b_3b_bien_the_thoi_gian'):
        s = summary[k]
        print(f'{k}: du_phu {s.get("du_phu")}, rad_dev {s.get("rad_dev_m")}, hp_dev {s.get("hp_dev_deg")}; {s["cuoi"]}')


JOIN_GAP_S = 10.0          # C1/ghep_3b.py default gap between the clips


def _check_join(f1, f2, fg):
    """Every row of the join equals its source row (t_s of clip 2 shifted, idx of clip 2 offset; other cells equal);
    the first frame of clip 2 comes JOIN_GAP_S after the last frame of clip 1 (DNC1-15)."""
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
        if len(out) != len(src):
            raise DnError(f'join row {j}: cell count')
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
    if abs(float(g[1 + n1][it]) - float(g[n1][it]) - JOIN_GAP_S) > 2e-6:
        raise DnError('join: the gap between the clips is not JOIN_GAP_S')


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


def _num(s):
    return float(s) if s not in ('', None) else np.nan


def cmd_kiem(a):
    cu = read_cu(a.cu)
    cx, cy = cu['centre']
    R0 = cu['R0_m']
    phi_sl = phi_reference(a.cu_ghep, (cx, cy))
    rep = dict(version=VERSION, moc=dict(centre=[cx, cy], R0_m=R0, phi_sl_rad=phi_sl, tu=a.cu, sha256=cu['sha256'],
                                         phi_tu=a.cu_ghep, phi_tu_sha256=sha256(a.cu_ghep)), clips={})
    md = ['# Kiem do chinh xac c8_doc_nhoe v5', '',
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
        hp = wrap(C['psi'] - th - phi_sl)
        vt = np.abs(omega_loo(C['t'], th, vnew)) * R0
        vt_seen = np.abs(omega_loo(C['t'], th, s)) * R0
        tang = np.array([_num(R[k]['du_tiep_m']) if k in R else np.nan for k in range(n)])
        e = np.array([_num(R[k]['e_hinh_m']) if k in R else np.nan for k in range(n)])
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
        funnel = dict(tong=n, khong_thay_ma=int((~s).sum()), thay_ma_khong_vt=int((s & ~np.isfinite(vt_seen)).sum()),
                      ung_vien=sum(1 for r in R.values() if r['cong'] == 'ung_vien'),
                      quyet_dinh=dict(Counter((r['ly_do'] or 'nhan') for r in R.values() if r['cong'] == 'ung_vien')),
                      theo_dai=[],
                      ghi_chu='use_p cua c8b_3b chua dung lai o day (DNC1-12, DEFERRED); v_t theo khung hop le moi')
        for lo, hi in bands:
            m = s & np.isfinite(vt_seen) & (vt_seen >= lo) & (vt_seen < hi)
            funnel['theo_dai'].append(dict(dai=f'{lo}-{hi}', thay_ma=int(m.sum()), hop_le_cu=int((m & vold).sum()),
                                           hop_le_moi=int((m & vnew).sum()),
                                           co_vt_tren_hop_le=int((m & vnew & np.isfinite(vt)).sum())))
        per = float(np.median(np.diff(C['t'])))
        gaps = _runs(~s)
        funnel['lo_mat_ma'] = dict(so_lo=len(gaps), dai_nhat=max((b - a for a, b in gaps), default=0),
                                   khung_trong_lo_dai=int(sum(b - a for a, b in gaps if (b - a) * per > VT_GAP_S)))
        crit = _criteria(cells)
        sens = _cluster_sensitivity(dr, hp, vt, vnew, gaps, rng)
        doc2 = _doc2_report(os.path.join(d, 'doc_hai_rows.csv'))
        scl = _scale_report(C, R, vold, vnew, vt)
        rep['clips'][name] = dict(o=cells, theo_pha=by_q, theo_hang=by_row, pheu=funnel, tieu_chi=crit,
                                  do_nhay_cuc_tri=sens, doc_hai=doc2, ti_le_scale=scl)
        md += _md_clip(name, cells, crit, funnel, sens, doc2, scl)
    rep['alpha'] = _alpha_report(os.path.join(a.out, 'B_ghep', 'frames.csv'))
    if a.rs:
        with open(os.path.join(a.out, 'khoa.json'), encoding='utf-8') as f:
            ax = json.load(f)['clips']['c1']['axis_px']
        rep['doi_chieu_rs'] = _rs_check(a.rs, os.path.join(a.out, 'B_c1_3b'), ax)
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


def _doc2_report(path):
    """Second reading (template minus edge reading), signed, by band and group (report, DN4-09)."""
    with open(path, newline='', encoding='utf-8') as f:
        R = list(csv.DictReader(f))
    out = []
    for g in ('cu', 'moi', 'khong_nhan'):
        rr = [r for r in R if r['nhom'] == g and r['ok'] == '1']
        if not rr:
            continue
        row = dict(nhom=g, n=len(rr), n_loi=sum(1 for r in R if r['nhom'] == g and r['ok'] != '1'))
        for c in ('d_tiep_m', 'd_phap_m', 'd_huong_rad', 'e_goc_m', 'corr'):
            v = np.array([_num(r[c]) for r in rr])
            row[c] = _stat(v)
        out.append(row)
    return out


def _scale_report(C, R, vold, vnew, vt):
    """DN4-08: scale of the new frames vs the median scale of the old valid frames of the same phase cell with v_t
    0.8-1.05 (report); F08-CAP has no old frame to compare (2b) with."""
    cell = np.array([int(R[k]['o_pha']) if k in R and R[k]['o_pha'] != '' else -1 for k in range(len(C['t']))])
    ref = {}
    mo = vold & np.isfinite(vt) & (vt >= 0.8) & (vt < 1.05)
    for c in np.unique(cell[mo]):
        ref[int(c)] = float(np.median(C['scale'][mo & (cell == c)]))
    out = {}
    for lab, lo, hi in (('0.8-1.05', 0.8, 1.05), ('F08-CAP', 1.166, 1.283)):
        m = vnew & ~vold & np.isfinite(vt) & (vt >= lo) & (vt < hi)
        r = np.array([C['scale'][k] / ref[int(cell[k])] - 1.0 for k in np.flatnonzero(m) if int(cell[k]) in ref])
        out[lab] = _stat(r)
    out['ghi_chu'] = 'F08-CAP: thieu doi chung ti le (2b) (DN4-08)'
    return out


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
            if mo['tiep_loo_m'] is None:
                r['moi'] = 'thieu'                       # the tangential residual is required (DNC1-12)
            else:
                ok = mo['dr_m']['sd'] <= 0.15e-3 and mo['hp_rad']['sd'] <= math.radians(0.3) and \
                    mo['tiep_loo_m']['p95_abs'] <= 0.6e-3
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


def _md_clip(name, cells, crit, funnel, sens, doc2, scl):
    f = lambda s, k, sc=1e3: '-' if s is None else f'{s[k] * sc:.2f}'          # noqa: E731
    dg = 180 / math.pi
    L = [f'## {name}', '', '| dai v_t | nhom | n | vong | dr tv/sd mm | hp tv/sd deg | du tiep (i) p95 mm | '
         'e_hinh p95 mm |', '|---|---|---|---|---|---|---|---|']
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
          f'Lo mat ma: {funnel["lo_mat_ma"]}', f'Do nhay cuc tri (bo cum nhu lo mat ma): {sens}', '',
          'Doc thu hai (khop mau tru doc canh, F08-CAP, mau khoa truoc), trung vi mm / deg:']
    for r in doc2:
        L.append(f'- {r["nhom"]} n {r["n"]} (loi {r["n_loi"]}): tiep {f(r["d_tiep_m"], "trung_vi")}, phap '
                 f'{f(r["d_phap_m"], "trung_vi")}, huong {f(r["d_huong_rad"], "trung_vi", dg)}, e_goc p95 '
                 f'{f(r["e_goc_m"], "p95_abs")}')
    L += ['', f'Ti le scale khung moi / khung cu cung pha (0.8-1.05): {scl}', '']
    return L


def _stat(v):
    v = np.asarray(v, float)
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


def _rs_check(rs_path, d, axis):
    """CODE_UT's independent reading of IMG_3601 (corners per frame), by idx: tag-centre difference in px, its size
    and, signed, along the motion (tangential about the image axis, sign of the rotation) and radial, by v_t band."""
    with open(os.path.join(d, 'doc_nhoe_rows.csv'), newline='', encoding='utf-8') as f:
        R = {int(r['clip_idx']): r for r in csv.DictReader(f)}
    C = load_clip(os.path.join(d, 'frames.csv'))
    ang = image_angle(C, axis)
    om = omega_loo(C['t'], ang, C['seen'])
    diffs = {'cu': [], 'moi': []}
    signed = []
    with open(rs_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            k = int(r['idx'])
            if r['ok'] != '1' or k >= len(C['t']) or not C['rows'][k]['valid'] or abs(float(r['t_s']) - C['t'][k]) > 1e-5:
                continue
            g = 'cu' if R.get(k, {}).get('valid_cu') == '1' else 'moi'
            dc, dr_ = float(r['col']) - C['col'][k], float(r['row']) - C['row'][k]
            diffs[g].append(math.hypot(dc, dr_))
            ru = np.array([C['col'][k] - axis[0], C['row'][k] - axis[1]])
            ru = ru / np.linalg.norm(ru)
            sg = 1.0 if not np.isfinite(om[k]) or om[k] >= 0 else -1.0
            v = _num(R.get(k, {}).get('v_t', ''))
            signed.append((g, v, float(np.array([dc, dr_]) @ np.array([-ru[1], ru[0]])) * sg,
                           float(np.array([dc, dr_]) @ ru)))
    out = {g: _stat(np.array(v)) for g, v in diffs.items()}
    out['co_dau_theo_dai'] = []
    for lo, hi in zip(BANDS_VT[:-1], BANDS_VT[1:]):
        for g in ('cu', 'moi'):
            x = [(a, b) for gg, v, a, b in signed if gg == g and np.isfinite(v) and lo <= v < hi]
            if x:
                x = np.array(x)
                out['co_dau_theo_dai'].append(dict(dai=f'{lo}-{hi}', nhom=g, tiep_px=_stat(x[:, 0]),
                                                   phap_px=_stat(x[:, 1])))
    return out


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
    if a.cmd == 'khoa' and os.path.exists(a.out):
        print(f'c8_doc_nhoe: {a.out} exists', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
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
