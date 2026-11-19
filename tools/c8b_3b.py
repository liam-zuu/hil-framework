#!/usr/bin/env python3
"""
c8b_3b.py: test 3b of C8 (spinning wheel), C8b design v4 section 4b (approved: Liam 17/11, CODE_C8B local_cccce6cb,
"Duyet (Recommended)"; CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_C8b_v4.md). Written by CODE_C8B
local_21b7debc on feat/c8b-u.

Set-up (design 4b): a 60 cm motorcycle wheel lying flat, a stiff flat base taped across the rim, the car's own tag
plate (3D-printed, on its pins) on the base, level; (A) tag centre on the axis, (B) about 25 cm from the axis; the
car out of the picture; spun by hand both ways, slowing down. Each clip is tracked by c7_measure.py track with
--tag-h = the measured tag height and no tag offset, so frames.csv x, y = the tag centre (tag_x, tag_y), psi = the
tag heading, row_px / col_px = the tag centre in sensor pixels (3840 x 2160, auto rotation off).

What it gives (names as in design 4b, C8B4R2-5, C8B4R3-3):
  (A) heading oscillation from spinning: the heading residual after a smooth (quadratic) fit over every 0.5 s
      window; position error from spinning: the tag centre against the fixed point of the slowest part.
  (B) speed-dependent heading-position error: (tag heading - angle of the tag centre about the axis) against the
      slowest part, by speed and direction; radial position error: radius against the slowest part.
  d_psi_quay, d_p_quay = the largest of these parts up to the F08 rates, taken over the data up to CAP (1.1) x the
      peak (heading: |Omega| vs wz_max of O1-F08 for (A); position and the (B) heading-position part: tangential speed
      |Omega| R vs v_max of O1-F08); each part must reach the peak in each direction (C8BS-10), else no number.
  Rolling shutter (F8; C8B4R2-6, R4-1): at (B) the angle of the centre is moved by Omega (alpha_r dr + alpha_c dc)
      (dr, dc: pixel offset of the tag from its mean over a turn), once per turn, proportional to the rotation rate,
      sign flipping with the direction. Model: per piece (one direction, at most PIECE_TURNS turns) a cubic in time,
      plus first and second harmonics of the angle (wobble, eccentricity, perspective; NOT proportional to the rate),
      plus Omega (alpha_r dr + alpha_c dc); linear least squares. alpha is used only when the shutter shift
      |alpha_r| DR + |alpha_c| DC at a reference span (DR_REF rows, DC_REF columns) estimated separately on the
      speed bands x two directions spreads at most SPREAD_MAX (30 % [DE XUAT]) around its mean; else one frame
      (design table "Man trap"). The value used by U_t is alpha x 1.25 (le).
Not seen (limits, design 4b): smooth errors absorbed by the fits, a common phase error in heading and tangential
position, errors that do not change with speed (test 2), vibration and tilt of the real car.

Checks (design 4b "Kiem ga"; v2: after a geometric model of the slowest part, C8B design C8b tuong doi v6 section 5a,
Liam 19/11 at CODE local_6653c962 "keu C8b lam luon di"; Codex round 1 G3B-1..8): per CLIP (the joined clips are
cut at a time gap > CLIP_GAP_S: the set-up may be moved between clips when re-spun by hand, each clip has its own
geometry), the residual of the clip's slowest part after the model within GA_TOL_M (2 mm), in the fit and held out
(the slowest part split in two by time, each half fitted and checked on the other; (A) also split by speed); every
clip must have a slowest part; else ga_ok false, numbers not used.
  (A) the tag centre is c0 + R(psi) q (c0: the axis on the floor, q: the tag centre from the axis in the tag frame;
      the tag cannot be placed on the axis to better than about 1.5 mm, so v1, which asked a fixed point, failed on
      geometry). q must be identified: the headings of the slowest part spread (1 - |mean e^(i psi)| >= Q_SPREAD_MIN).
      A camera error locked to the rotation angle is taken into q (J v6 section 6): a part constant in the tag frame,
      the same at every speed and both ways, is a tag-frame offset, the same on the car (q measured there by E), not
      a spinning error; the part that changes with speed (q of the slower vs faster half of the slowest part) or with
      the direction (q of the two directions) is added to the position part p_dev; not estimable -> p_dev null.
      A centre moving in step with the rotation is the same rigid motion as a larger q (no error for P4).
  (B) the radius is R0 + first and second harmonics of the angle (centre offset, ellipse) + a linear drift in time
      in the fit; held out: no drift term (the radius must stay put between the halves), harmonics only when the
      half covers a turn, the circle refitted on the half when it covers a turn. The radial part rad_dev is
      |r - R0| about the clip's circle (harmonics and drift NOT removed). The tangential part is not measured (no
      independent phase reference): a limit.
  The v1 numbers are kept as *_tho_m. An optional clip of the tag on the axis (--frames-a-cheo, "A cu") is analysed
  the same way as a cross-check of the axis model only (its heading oscillation is not used: jerky hand spin).

  python tools/c8b_3b.py --frames-a A/frames.csv --frames-b B/frames.csv [--frames-a-cheo Acu/frames.csv] --out 3b.json
Last line: #ket_qua,ma=0,ga_ok=1|0,alpha_dung=1|0 / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_3b_v2'
LE = 1.25
GA_TOL_M = 0.002            # design 4b "Kiem ga" (kept, CODE 19/11)
Q_SPREAD_MIN = 0.10         # [DE XUAT] (A) q is identified when 1 - |mean e^(i psi)| over the fitted frames >= 0.10
CLIP_GAP_S = 8.0            # a gap > 8 s between ROWS (valid or not) separates two joined clips (ghep_3b.py leaves 10 s);
                            # gaps of valid frames are not used (B IMG_3602: 88 s of rejected fast frames, run 9dd8b22)
WIN_S = 0.5                 # design 4b (A): smooth fit over 0.5 s
SLOW_SHARE = 0.10           # the slowest part: the 10 % of the moving frames with the smallest |Omega| [DE XUAT]
SLOW_MIN = 30               # at least 30 frames in the slowest part
MOVE_RAD_S = 0.3            # a frame is spinning when |Omega| > 0.3 rad/s [DE XUAT]
PIECE_TURNS = 3.0           # design 4b "vai vong" per smooth piece
TRIM_START_S, TRIM_END_S = 0.5, 0.25   # [DE XUAT] see trim_runs
MIN_PIECE_TURNS = 1.0       # [DE XUAT] a piece shorter than one turn is not used (a cubic over part of a turn
                            # absorbs the once-per-turn terms)
SPREAD_MAX = 0.30           # [DE XUAT] design 4b
DR_REF, DC_REF = 700.0, 700.0   # reference span (rows 700-1450 of the car runs; columns of the same size)
SHUTTER_MIN_RAD_S = 2.0     # [DE XUAT] shutter fit only above 2 rad/s: below, one smooth piece of >= 1 turn lasts
                            # too long for a cubic, and the shutter term is small (synthetic test: bias at 0.3-1.5 rad/s)
PHU_FRAC = 1.0              # C8BS-10 round 3: a part counts as measured up to the F08 peak only when its used data
                            # reach the peak itself in each direction (design 4b: (A) spin above wz peak 2.61 rad/s,
                            # 0.42 turn/s; (B) above v peak / R, 0.74 turn/s at 25 cm)
RUN_JUMP_RAD_S = 0.5        # [DE XUAT] a rise of the smoothed rate this far above the run's lowest starts a new spin
RUN_GAP_JUMP_RAD_S = 0.1    # [DE XUAT] across such a stretch any rise above the rate noise is a re-spin (Codex G3B-1 r3)
TAM_TOL_M = 0.001            # [DE XUAT] centres of two parts within 1 mm: the held-out gap is a radius change
RUN_GAP_S = 2.0             # [DE XUAT] a rise across a stretch this long without spinning frames is a re-spin
RUN_SMOOTH = 31            # spinning frames in the running median of the rate for the runs (0.5 s at 60 frames/s)
TURN_BINS = 12              # [DE XUAT] a full turn: every 30 degree sector ...
TURN_BIN_MIN = 3            # ... has at least 3 frames
CAP = 1.1                   # [DE XUAT] each part uses the data up to 1.1 x the F08 peak (a superset of the range up to
                            # the peak: the largest error over it is not below the largest up to the peak), so the
                            # hand spin, which slows down continuously, passes through the whole range up to the peak
SUBSET_MIN = 120            # frames per check fit (design 4b consistency)
DERIV_HALF = 3              # frames each side for the rate (central quadratic fit)


class TbError(Exception):
    pass


def read(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        fr = list(csv.DictReader(f))
    if not fr:
        raise TbError(f'{path}: no frame')
    need = ('t_s', 'valid', 'psi', 'row_px')
    miss = [c for c in need if c not in fr[0]]
    if miss:
        raise TbError(f'{path}: no column {miss}')
    xs, ys = ('tag_x', 'tag_y') if 'tag_x' in fr[0] else ('x', 'y')

    def g(c):
        return np.array([float(r[c]) if c in r and r[c] not in ('', 'nan', None) else np.nan for r in fr])
    v = np.array([r['valid'] == '1' for r in fr])
    F = dict(t=g('t_s'), x=g(xs), y=g(ys), psi=g('psi'), row=g('row_px'), col=g('col_px'))
    # clip of every row from the gaps of ALL rows (valid or not): a long stretch of rejected frames inside a clip is
    # not a joint between clips (ghep_3b.py leaves CLIP_GAP_S+ between the last row of a clip and the next)
    ta = F['t']
    F['clip'] = np.concatenate([[0], np.cumsum(np.diff(ta) > CLIP_GAP_S)]) if len(ta) else np.zeros(0, int)
    k = v & np.isfinite(F['x']) & np.isfinite(F['y']) & np.isfinite(F['psi'])
    F = {c: a[k] for c, a in F.items()}
    if len(F['t']) < 50:
        raise TbError(f'{path}: fewer than 50 valid frames')
    if np.any(np.diff(F['t']) <= 0):
        raise TbError(f'{path}: t_s does not increase')
    F['psi'] = _unwrap_gapped(F['t'], F['psi'])
    F['path'] = path
    return F


def _unwrap_gapped(t, a):
    return np.unwrap(a)


def rate(t, a, half=DERIV_HALF):
    """d a / d t by a quadratic over +-half frames (NaN at the ends and where the frames are not contiguous)."""
    n = len(t)
    out = np.full(n, np.nan)
    per = float(np.median(np.diff(t)))
    for i in range(half, n - half):
        tt = t[i - half:i + half + 1] - t[i]
        if tt[-1] - tt[0] > (2 * half + 0.5) * per:
            continue
        c = np.polyfit(tt, a[i - half:i + half + 1], 2)
        out[i] = c[1]
    return out


def slowest(om):
    mv = np.isfinite(om) & (np.abs(om) > MOVE_RAD_S)
    idx = np.flatnonzero(mv)
    if len(idx) < SLOW_MIN:
        raise TbError(f'fewer than {SLOW_MIN} spinning frames')
    k = max(SLOW_MIN, int(SLOW_SHARE * len(idx)))
    return idx[np.argsort(np.abs(om[idx]))[:k]]


# ── geometric models of the slowest part (v2) ─────────────────────────────────────────────────────

def halves(t, idx):
    """The fitted frames split in two by time (first half, second half of idx sorted by t)."""
    s = idx[np.argsort(t[idx])]
    h = len(s) // 2
    return s[:h], s[h:]


def axis_fit(x, y, psi, idx):
    """(A) tag centre = c0 + R(psi) q, linear least squares on the frames idx. Returns c0, q, spread
    (1 - |mean e^(i psi)|) and whether q is identified."""
    c, s = np.cos(psi[idx]), np.sin(psi[idx])
    A = np.zeros((2 * len(idx), 4))
    A[0::2, 0] = 1
    A[1::2, 1] = 1
    A[0::2, 2], A[0::2, 3] = c, -s
    A[1::2, 2], A[1::2, 3] = s, c
    b = np.empty(2 * len(idx))
    b[0::2], b[1::2] = x[idx], y[idx]
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    spread = float(1.0 - abs(np.mean(np.exp(1j * psi[idx]))))
    return th[:2], th[2:], spread, bool(spread >= Q_SPREAD_MIN)


def axis_res(x, y, psi, c0, q):
    """|tag centre - (c0 + R(psi) q)| per frame."""
    c, s = np.cos(psi), np.sin(psi)
    return np.hypot(x - (c0[0] + c * q[0] - s * q[1]), y - (c0[1] + s * q[0] + c * q[1]))


def radius_design(th, t, tm, drift=True):
    cols = [np.ones_like(th), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)]
    if drift:
        cols.append(t - tm)
    return np.column_stack(cols)


def radius_fit(r, th, t, idx):
    """(B) radius = R0 + harmonics 1, 2 of the angle + a linear drift in time, on the frames idx."""
    tm = float(np.mean(t[idx]))
    co, *_ = np.linalg.lstsq(radius_design(th[idx], t[idx], tm), r[idx], rcond=None)
    return co, tm


def radius_res(r, th, t, co, tm, drift=True):
    if drift:
        return r - radius_design(th, t, tm) @ co
    return r - radius_design(th, t, tm, drift=False) @ co[:5]


def held_out(fit, res, t, idx, split=None):
    """Largest residual of each half of idx when the model is fitted on the other half; None when a half is too
    small (< SLOW_MIN / 2) or (A) its q is not identified. split: the two halves (default: by time)."""
    a, b = split if split is not None else halves(t, idx)
    if min(len(a), len(b)) < SLOW_MIN // 2:
        return None, 'a half of the slowest part has too few frames'
    worst = 0.0
    for f_, e_ in ((a, b), (b, a)):
        m = fit(f_)
        if m is None:
            return None, 'q not identified on a half of the slowest part'
        worst = max(worst, float(np.max(res(m, e_))))
    return worst, ''


def clips(t, clip=None):
    """Frame index arrays of the clips joined in one frames.csv (ghep_3b.py leaves a gap > CLIP_GAP_S): the set-up
    may be moved between clips (re-spun by hand), so each clip has its own geometry (Codex G3B-1). clip: the clip
    of each frame from read() (all rows); without it, the gaps of t."""
    if clip is None:
        cut = np.flatnonzero(np.diff(t) > CLIP_GAP_S) + 1
        return np.split(np.arange(len(t)), cut)
    return [np.flatnonzero(clip == c) for c in np.unique(clip)]


def slowest_in(om, idx):
    """slowest() within one clip; None when the clip has fewer than SLOW_MIN spinning frames."""
    sub = np.full(len(om), np.nan)
    sub[idx] = om[idx]
    try:
        return slowest(sub)
    except TbError:
        return None


def spin_runs(t, om):
    """Run label of every frame (-1: not spinning). A run is one spin of the set-up: a new run starts when the
    direction changes or when the rate rises by more than RUN_JUMP_RAD_S above the lowest rate of the current run
    (a re-spin by hand, during which the set-up may be touched; Codex G3B-1 round 2). The rate is smoothed by a
    running median over RUN_SMOOTH spinning frames so that noise does not cut a run."""
    lab = np.full(len(om), -1)
    sp = np.flatnonzero(np.isfinite(om) & (np.abs(om) > MOVE_RAD_S))
    if not len(sp):
        return lab
    sp = sp[np.argsort(t[sp])]
    a = np.abs(om[sp])
    h = RUN_SMOOTH // 2
    seg = np.concatenate([[0], np.cumsum(np.diff(t[sp]) > RUN_GAP_S)])   # the median does not reach across a gap
    sm = np.empty(len(a))
    for i in range(len(a)):
        w_ = slice(max(0, i - h), i + h + 1)
        sm[i] = np.median(a[w_][seg[w_] == seg[i]])
    sg = np.sign(om[sp])
    # after a start the rate may still rise (spin-up): the lowest rate is tracked only once it falls again
    run, lo, s0, rising = 0, math.inf, sg[0], True
    for j, i in enumerate(sp):
        # a rise across a stretch without spinning frames (> RUN_GAP_S) is a re-spin too, even when the frames
        # before it were still rising (a slow start, then the set-up spun by hand while the frames were lost)
        regap = j > 0 and t[i] - t[sp[j - 1]] > RUN_GAP_S and sm[j] > sm[j - 1] + RUN_GAP_JUMP_RAD_S
        if sg[j] != s0 or regap or (not rising and sm[j] > lo + RUN_JUMP_RAD_S):
            run, lo, s0, rising = run + 1, math.inf, sg[j], True
        if rising and j > 0 and lab[sp[j - 1]] == run and sm[j] < sm[j - 1]:
            rising = False
        if not rising:
            lo = min(lo, sm[j])
        lab[i] = run
    return lab


def held_out_runs(fit, res, slk, runs_ci, run):
    """Codex G3B-1 round 2: a set-up moved between two spins of one clip is not seen by the time halves when the slow
    frames come from one spin. For every run of the clip that has slow frames: fit on the slow frames of the other runs,
    take the largest residual on its own slow frames. Returns (worst or None, reasons, runs without slow frames: their
    geometry is not checked, so their frames are not used)."""
    why, worst, unchecked = [], None, []
    if len(runs_ci) < 2:
        return None, why, unchecked
    unchecked = [int(r_) for r_ in runs_ci if not np.any(run[slk] == r_)]
    with_slow = [r_ for r_ in runs_ci if int(r_) not in unchecked]
    if len(with_slow) < 2:
        # Codex G3B-1 round 3: one spin with slow frames, the others not used: nothing to hold out between spins (the
        # time halves still check the one spin); not a failure
        return None, why, unchecked
    for r_ in with_slow:
        own = slk[run[slk] == r_]
        other = slk[run[slk] != r_]
        if len(other) < SLOW_MIN // 2:
            why.append(f'run {r_}: held-out by run: fewer than {SLOW_MIN // 2} slow frames in the other runs')
            continue
        m = fit(other)
        if m is None:
            why.append(f'run {r_}: held-out by run: q not identified on the other runs')
            continue
        e = float(np.max(res(m, own)))
        worst = e if worst is None else max(worst, e)
    return worst, why, unchecked


def split_kind(x, y, ia, ib):
    """Codex G3B3-1: a failed held-out comparison between two parts of the slow frames (ia, ib): circle of each part
    (identified when it covers a full turn about its own circle); centre shift and radius change between them; kind
    'ban_kinh' (radius change, centres within TAM_TOL_M), 'tam' (centre moved) or 'khong_nhan_dang' (a part under a
    turn: the circle is not fixed, the cause not told)."""
    out = []
    for i in (ia, ib):
        cx_, cy_, R_ = circle_fit(x[i], y[i])
        out.append((cx_, cy_, R_, full_turn(np.arctan2(y[i] - cy_, x[i] - cx_))))
    (ax, ay, aR, aok), (bx, by, bR, bok) = out
    dc, dr = float(math.hypot(bx - ax, by - ay)), float(bR - aR)
    kind = 'khong_nhan_dang' if not (aok and bok) else ('ban_kinh' if dc <= TAM_TOL_M else 'tam')
    return dict(kieu=kind, dich_tam_m=dc, doi_ban_kinh_m=dr, n=[int(len(ia)), int(len(ib))])


def full_turn(ang):
    """The angles cover every one of TURN_BINS sectors of a turn with at least TURN_BIN_MIN frames (a part of a turn
    does not fix a circle; Codex G3B-5 round 2)."""
    if not len(ang):
        return False
    k = np.floor(np.mod(ang, 2 * math.pi) / (2 * math.pi) * TURN_BINS).astype(int) % TURN_BINS
    return bool(np.all(np.bincount(k, minlength=TURN_BINS) >= TURN_BIN_MIN))


def slowest_turn(om, idx, x, y):
    """slowest_in() grown by steps of 5 % of the spinning frames until the slow part covers a full turn about the
    circle fitted to it; None when the clip never does."""
    sp = idx[np.isfinite(om[idx]) & (np.abs(om[idx]) > MOVE_RAD_S)]
    if len(sp) < SLOW_MIN:
        return None
    s = sp[np.argsort(np.abs(om[sp]))]
    k = max(SLOW_MIN, int(SLOW_SHARE * len(s)))
    step = max(1, len(s) // 20)
    while True:
        sel = s[:k]
        cx_, cy_, _ = circle_fit(x[sel], y[sel])
        if full_turn(np.arctan2(y[sel] - cy_, x[sel] - cx_)):
            return sel
        if k >= len(s):
            return None
        k = min(len(s), k + step)


def speed_halves(om, idx):
    """The slowest part split by |Omega| (slower half, faster half): a geometry fitted on one speed half must
    predict the other, else a speed-dependent measurement error is being taken for geometry (Codex G3B-3)."""
    s = idx[np.argsort(np.abs(om[idx]))]
    h = len(s) // 2
    return s[:h], s[h:]


# ── (A) tag centre on the axis ────────────────────────────────────────────────────────────────────

def analyse_a(F, wz_max):
    t, x, y, psi = F['t'], F['x'], F['y'], F['psi']
    om = rate(t, psi)
    sl = slowest(om)
    px, py = float(np.median(x[sl])), float(np.median(y[sl]))
    dev_tho = float(np.max(np.hypot(x[sl] - px, y[sl] - py)))          # v1 check (fixed point), kept for reference
    dev_tho_all = np.hypot(x - px, y - py)
    use = np.isfinite(om) & (np.abs(om) > MOVE_RAD_S) & (np.abs(om) <= CAP * wz_max)

    def _fit(i):
        c_, q_, _, ok_ = axis_fit(x, y, psi, i)
        return (c_, q_) if ok_ else None

    def _res(m, i):
        return axis_res(x[i], y[i], psi[i], *m)
    why, p4_why, per_clip = [], [], []
    dev = np.full(len(t), np.nan)                     # residual of each frame against the geometry of its own clip
    run = spin_runs(t, om)
    for k, ci in enumerate(clips(t, F.get('clip'))):
        slk = slowest_in(om, ci)
        if slk is None:
            if (use[ci]).any():
                why.append(f'clip {k}: fewer than {SLOW_MIN} spinning frames, its geometry not checked')
            per_clip.append(dict(clip=k, slow_n=0))
            continue
        c0, q, spread, q_ok = axis_fit(x, y, psi, slk)
        dk = float(np.max(_res((c0, q), slk)))
        ho_t, why_t = held_out(_fit, _res, t, slk)
        if not q_ok:
            why.append(f'clip {k}: q not identified, heading spread {spread:.3f} < {Q_SPREAD_MIN}')
        if dk > GA_TOL_M:
            why.append(f'clip {k}: residual {dk * 1e3:.2f} mm > {GA_TOL_M * 1e3:.0f} mm')
        if ho_t is None:
            why.append(f'clip {k}: held-out by time: {why_t}')
        elif ho_t > GA_TOL_M:
            why.append(f'clip {k}: held-out by time residual {ho_t * 1e3:.2f} mm > {GA_TOL_M * 1e3:.0f} mm')
        # set-up moved between two spins of the clip (Codex G3B-1 round 2)
        runs_ci = [int(r_) for r_ in np.unique(run[ci]) if r_ >= 0]
        ho_r, why_r, unchecked = held_out_runs(_fit, _res, slk, runs_ci, run)
        why += [f'clip {k}: {w}' for w in why_r]
        if ho_r is not None and ho_r > GA_TOL_M:
            why.append(f'clip {k}: held-out by run residual {ho_r * 1e3:.2f} mm > {GA_TOL_M * 1e3:.0f} mm')
        not_checked = ci[np.isin(run[ci], unchecked)]
        # speed part of an error the model takes into q (Codex G3B-3, J v6 section 6): q fitted on the slower and on
        # the faster half of the slowest part; the cross residual is a set-up check, the change of q is added to the
        # position part
        a_, b_ = speed_halves(om, slk)
        fa_, fb_ = _fit(a_), _fit(b_)
        dq_v, ho_v = None, None
        if fa_ is None or fb_ is None:
            p4_why.append(f'clip {k}: q not identified on a speed half of the slowest part')
        else:
            dq_v = float(np.hypot(*(fa_[1] - fb_[1])))
            ho_v = max(float(np.max(_res(fa_, b_))), float(np.max(_res(fb_, a_))))
            if ho_v > GA_TOL_M:
                why.append(f'clip {k}: held-out by speed residual {ho_v * 1e3:.2f} mm > {GA_TOL_M * 1e3:.0f} mm')
        # direction part within the clip (both directions in its slowest part)
        pos_, neg_ = slk[om[slk] > 0], slk[om[slk] < 0]
        dq_d = None
        if min(len(pos_), len(neg_)) >= SLOW_MIN // 2:
            fp_, fn_ = _fit(pos_), _fit(neg_)
            if fp_ is not None and fn_ is not None:
                dq_d = float(np.hypot(*(fp_[1] - fn_[1])))
        dev[ci] = _res((c0, q), ci) + (dq_v if dq_v is not None else 0.0)
        dev[not_checked] = np.nan                       # a spin without slow frames: its set-up not checked, not used
        use[not_checked] = False
        per_clip.append(dict(clip=k, slow_n=int(len(slk)), c0=[float(c0[0]), float(c0[1])],
                             q_m=[float(q[0]), float(q[1])], q_abs_m=float(np.hypot(*q)), q_spread=spread,
                             q_nhan_dang=q_ok, ga_dev_m=dk, ga_giu_ngoai_thoi_gian_m=ho_t,
                             ga_giu_ngoai_toc_do_m=ho_v, dq_toc_do_m=dq_v, dq_chieu_trong_clip_m=dq_d,
                             n_luot_quay=len(runs_ci), ga_giu_ngoai_luot_m=ho_r,
                             luot_khong_kiem=unchecked, n_khung_khong_kiem=int(len(not_checked)),
                             chieu=sorted({int(np.sign(v)) for v in om[slk]}),
                             slow_rate_rad_s=[float(np.min(np.abs(om[slk]))), float(np.max(np.abs(om[slk])))]))
    ok_clips = [d for d in per_clip if d['slow_n']]
    if not ok_clips:
        why.append('no clip with a slowest part')
    ga_ok = not why
    ga_dev = max((d['ga_dev_m'] for d in ok_clips), default=None)
    hos = [v for d in ok_clips for v in (d['ga_giu_ngoai_thoi_gian_m'], d['ga_giu_ngoai_toc_do_m'],
                                         d['ga_giu_ngoai_luot_m']) if v is not None]
    doi = []
    for d1, d2 in zip(ok_clips[:-1], ok_clips[1:]):
        doi.append(dict(clip=[d1['clip'], d2['clip']],
                        dc0_m=float(np.hypot(d2['c0'][0] - d1['c0'][0], d2['c0'][1] - d1['c0'][1])),
                        dq_m=float(np.hypot(d2['q_m'][0] - d1['q_m'][0], d2['q_m'][1] - d1['q_m'][1]))))
    # camera error locked to the rotation angle that the model takes into q (J v6 section 6, Codex J5-11): a part
    # constant in the tag frame and the same at every speed and both ways is a tag-frame offset, the same on the car
    # (q measured there by the clip E with the same mechanism), so it is not a spinning error; the part that changes
    # with the speed (dq_toc_do, per clip) or with the direction (q of the two directions in one clip, or of clips
    # spun the other way) is, and is added to the position part. Not estimable -> position part not finite.
    dq_dirs = [d['dq_chieu_trong_clip_m'] for d in ok_clips if d['dq_chieu_trong_clip_m'] is not None]
    # between two clips only when each is one-way and the ways differ (Codex G3B2-1: a clip with a few slow frames the
    # other way is not a clip spun the other way; its direction part needs its own fit of both ways)
    dq_dirs += [dd['dq_m'] for dd, (d1, d2) in zip(doi, zip(ok_clips[:-1], ok_clips[1:]))
                if len(d1['chieu']) == 1 and len(d2['chieu']) == 1 and d1['chieu'] != d2['chieu']]
    dq_chieu = max(dq_dirs, default=None)
    if dq_chieu is None:
        p4_why.append('the direction part of the absorbed camera error not estimated (no slowest part of both '
                      'directions)')
    if p4_why:
        dev[:] = np.nan                                 # position part not finite
    else:
        dev = dev + dq_chieu
    use_d = use & np.isfinite(dev)
    c0, q = (ok_clips[0]['c0'], ok_clips[0]['q_m']) if ok_clips else ([math.nan] * 2, [math.nan] * 2)
    # heading residual after a quadratic over every 0.5 s window (step a quarter window)
    res_max, bands, n_win, win_rates = 0.0, [], 0, []
    t0 = t[0]
    while t0 + WIN_S <= t[-1]:
        m = (t >= t0) & (t < t0 + WIN_S)
        per = float(np.median(np.diff(t)))
        om_m = om[m]
        one_way = np.isfinite(om_m).all() and (np.all(om_m > MOVE_RAD_S) or np.all(om_m < -MOVE_RAD_S))
        contiguous = m.sum() >= 10 and np.max(np.diff(t[m])) <= 3 * per
        if one_way and contiguous:                # a window over a stop or a lost stretch is not used
            tt = t[m] - t[m].mean()
            r = psi[m] - np.polyval(np.polyfit(tt, psi[m], 2), tt)
            w = float(np.nanmean(np.abs(om[m]))) if np.isfinite(om[m]).any() else math.nan
            if math.isfinite(w) and MOVE_RAD_S < w <= CAP * wz_max:
                res_max = max(res_max, float(np.max(np.abs(r))))
                n_win += 1
                win_rates.append(float(np.sign(om_m[0])) * w)
            bands.append((w, float(np.max(np.abs(r)))))
        t0 += WIN_S / 4
    cov_h = coverage(np.array(win_rates), wz_max)           # the heading windows used, each direction
    cov_p = coverage(om[use], wz_max)                        # the position frames used, each direction
    return dict(fixed_point=[float(c0[0]), float(c0[1])], q_m=[float(q[0]), float(q[1])],
                q_abs_m=max((d['q_abs_m'] for d in ok_clips), default=None),
                q_nhan_dang=bool(ok_clips) and all(d['q_nhan_dang'] for d in ok_clips),
                theo_clip=per_clip, doi_giua_clip=doi,
                p4_why=p4_why,
                hap_thu=dict(dq_toc_do_max_m=max((d['dq_toc_do_m'] for d in ok_clips if d['dq_toc_do_m'] is not None),
                                                 default=None),
                             dq_chieu_m=dq_chieu,
                             cach=('p_dev = du sau mo hinh cua clip + dq_toc_do cua clip + dq theo chieu (trong clip hay giua clip nguoc chieu); '
                                   'lech co dinh trong he ma khong doi theo toc do nam trong q (tren xe q do bang E '
                                   'cung co che), khong thuoc P4; mot phan khong uoc duoc -> p_dev null')),
                slow_n=int(len(sl)), slow_rate_rad_s=float(np.median(np.abs(om[sl]))),
                ga_dev_m=ga_dev, ga_dev_giu_ngoai_m=max(hos) if hos else None, ga_ok=ga_ok, ga_why=why,
                ga_dev_tho_m=dev_tho, fixed_point_tho=[px, py],
                du_phu=cov_h['ok'] and cov_p['ok'],
                phu=dict(huong=cov_h, vi_tri=cov_p), n_win=n_win,
                psi_osc_deg=math.degrees(res_max) if n_win else None,
                p_dev_m=float(np.max(dev[use_d])) if use_d.any() and not p4_why else None,
                p_dev_tho_m=float(np.max(dev_tho_all[use])) if use.any() else None,
                rate_max_rad_s=float(np.nanmax(np.abs(om))), n_used=int(use.sum()),
                by_rate=_binned(bands))


def _binned(pairs, nb=6):
    pairs = [(w, r) for w, r in pairs if math.isfinite(w)]
    if not pairs:
        return []
    w = np.array([p[0] for p in pairs])
    r = np.array([p[1] for p in pairs])
    edges = np.linspace(0, w.max() + 1e-9, nb + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (w >= a) & (w < b)
        if m.any():
            out.append(dict(rate_rad_s=[float(a), float(b)], n=int(m.sum()), max_deg=math.degrees(float(r[m].max()))))
    return out


# ── (B) tag off the axis ─────────────────────────────────────────────────────────────────────────

def coverage(rates, need):
    """C8BS-10: the samples (or windows) a part actually uses reach PHU_FRAC x need in EACH direction (signed
    rates or speeds; need: the F08 peak) [DE XUAT]."""
    r = np.asarray(rates, float)
    r = r[np.isfinite(r)]
    pos = float(np.max(r[r > 0])) if (r > 0).any() else 0.0
    neg = float(-np.min(r[r < 0])) if (r < 0).any() else 0.0
    return dict(ok=bool(pos >= PHU_FRAC * need and neg >= PHU_FRAC * need), can=need, phan=PHU_FRAC,
                max_duong=pos, max_am=neg)


def circle_fit(x, y):
    """Algebraic (Kasa) circle fit: centre, radius."""
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(cx), float(cy), float(math.sqrt(c + cx ** 2 + cy ** 2))


def shutter_design(t, th, om, dr, dc, pieces, groups, has_c):
    """Columns: per piece a cubic in time; harmonics 1, 2 of the angle (shared by all pieces: wobble, eccentricity,
    perspective, not proportional to the rate); per group g: Omega dr 1[g] (and Omega dc 1[g] with col_px)."""
    cols = []
    for p in pieces:
        tt = np.where(p, t - t[p].mean(), 0.0)
        for k in range(4):
            cols.append(np.where(p, tt ** k, 0.0))
    for k in (1, 2):
        cols += [np.cos(k * th), np.sin(k * th)]
    for g in groups:
        cols.append(np.where(g, om * dr, 0.0))
        if has_c:
            cols.append(np.where(g, om * dc, 0.0))
    return np.column_stack(cols)


def trim_runs(t, om, sel):
    """Drop TRIM_START_S at the start and TRIM_END_S at the end of every run of one direction (the push by hand,
    the turn-around: the rate changes too fast for the smooth pieces) [DE XUAT]."""
    out = sel.copy()
    idx = np.flatnonzero(sel)
    if not len(idx):
        return out
    per = float(np.median(np.diff(t)))
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1 or t[b] - t[a] > 3 * per or np.sign(om[b]) != np.sign(om[a]):
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    for a, b in runs:
        out[a:b + 1] &= (t[a:b + 1] >= t[a] + TRIM_START_S) & (t[a:b + 1] <= t[b] - TRIM_END_S)
    return out


def make_pieces(t, th, om, sel):
    """Pieces of one direction and at most PIECE_TURNS turns, contiguous frames, within sel."""
    pieces = []
    idx = np.flatnonzero(sel)
    if not len(idx):
        return pieces
    per = float(np.median(np.diff(t)))
    start = idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        brk = (b != a + 1) or (t[b] - t[a] > 3 * per) or np.sign(om[b]) != np.sign(om[a]) \
            or abs(th[b] - th[start]) > PIECE_TURNS * 2 * math.pi
        if brk:
            pieces.append((start, a))
            start = b
    pieces.append((start, idx[-1]))
    out = []
    for a, b in pieces:
        if b - a + 1 >= 20 and abs(th[b] - th[a]) >= MIN_PIECE_TURNS * 2 * math.pi:
            m = np.zeros(len(t), bool)
            m[a:b + 1] = True
            out.append(m)
    return out


def fit_shutter(t, th, om, row, col, sel, groups=None):
    """alpha_r, alpha_c [s/px] for each group (default: one group = sel); smooth pieces and harmonics shared."""
    has_c = bool(np.isfinite(col[sel]).any())
    if not has_c:
        col = np.zeros_like(row)                       # no col_px: alpha_c not estimated (0)
    dr = row - np.nanmean(row[sel])
    dc = col - np.nanmean(col[sel])
    ok = sel & np.isfinite(dr) & np.isfinite(dc) & np.isfinite(om)
    pieces = make_pieces(t, th, om, ok)
    if not pieces:
        return None
    m = np.any(np.vstack(pieces), axis=0)
    groups = [ok] if groups is None else [g & ok for g in groups]
    X = shutter_design(t, th, om, dr, dc, pieces, groups, has_c)[m]
    beta, *_ = np.linalg.lstsq(X, th[m], rcond=None)
    k = 2 if has_c else 1
    tail = beta[-k * len(groups):].reshape(len(groups), k)
    out = []
    for g, b in zip(groups, tail):
        ar = float(b[0])
        ac = float(b[1]) if has_c else 0.0
        out.append(dict(alpha_r_s_px=ar, alpha_c_s_px=ac, n=int((g & m).sum()), has_col=has_c,
                        shift_ref_s=abs(ar) * DR_REF + abs(ac) * DC_REF))
    res = dict(groups=out, pieces=len(pieces), n=int(m.sum()),
               rms_rad=float(np.sqrt(np.mean((th[m] - X @ beta) ** 2))))
    return res


def analyse_b(F, v_max, wz_max, frame_s):
    t, x, y, psi = F['t'], F['x'], F['y'], F['psi']
    th0 = np.unwrap(np.arctan2(y - np.median(y), x - np.median(x)))
    om = rate(t, th0)
    sl = slowest(om)
    cx, cy, R0 = circle_fit(x[sl], y[sl])
    th = np.unwrap(np.arctan2(y - cy, x - cx))
    om = rate(t, th)
    r = np.hypot(x - cx, y - cy)
    ga_dev_tho = float(np.max(np.abs(r[sl] - R0)))                     # v1 check, kept for reference
    # per clip (Codex G3B2-2): centre, angle, rate and tangential speed from the clip's own circle; filled below
    th = np.full(len(t), np.nan)
    om = np.full(len(t), np.nan)
    v_t = np.full(len(t), np.nan)
    use_p = np.zeros(len(t), bool)
    hp_err = np.full(len(t), np.nan)

    ctx = {}

    def _bfit(i):
        # the whole geometry (circle centre, radius model) from the frames i only (Codex G3B-4), when they cover at
        # least one turn about the centre of the clip's slowest part; else that centre is kept (a part of a turn does
        # not fix a circle) and the case is recorded
        cx_, cy_, _ = circle_fit(x[i], y[i])
        if 'full' in ctx:
            fx, fy = ctx['full']
            s_ = i[np.argsort(t[i])]
            span = np.ptp(np.unwrap(np.arctan2(y[s_] - fy, x[s_] - fx)))
            if span < 2 * math.pi:
                cx_, cy_ = fx, fy
                ctx['tam_chung'] = True
        r_ = np.hypot(x - cx_, y - cy_)
        th_ = np.unwrap(np.arctan2(y - cy_, x - cx_))
        if 'full' not in ctx:
            co_, tm_ = radius_fit(r_, th_, t, i)
            return cx_, cy_, co_, tm_
        # held out: no drift term (a drift fitted on one half is not extrapolated: the check is that the radius
        # stays put between the halves), harmonics only when the half covers one turn (Codex G3B-4 rank)
        s_ = i[np.argsort(t[i])]
        turn = np.ptp(np.unwrap(np.arctan2(y[s_] - cy_, x[s_] - cx_))) >= 2 * math.pi
        X = radius_design(th_[i], t[i], 0.0, drift=False)
        if not turn:
            X = X[:, :1]
            ctx['khong_hai_hoa'] = True
        c_, *_ = np.linalg.lstsq(X, r_[i], rcond=None)
        co_ = np.zeros(6)
        co_[:len(c_)] = c_
        return cx_, cy_, co_, 0.0

    def _bres(m, i):
        cx_, cy_, co_, tm_ = m
        r_ = np.hypot(x[i] - cx_, y[i] - cy_)
        th_ = np.arctan2(y[i] - cy_, x[i] - cx_)
        return np.abs(radius_res(r_, th_, t[i], co_, tm_))
    ga_why, per_clip = [], []
    rad_err = np.full(len(t), np.nan)
    for k, ci in enumerate(clips(t, F.get('clip'))):
        # rough angle about the clip's median point: only to rank the slow frames and count the sectors of a turn
        th_r = np.unwrap(np.arctan2(y[ci] - np.median(y[ci]), x[ci] - np.median(x[ci])))
        om_r = np.full(len(t), np.nan)
        om_r[ci] = rate(t[ci], th_r)
        # the slow part grown until it covers a full turn (Codex G3B-5 round 2: a free circle on a part of a turn
        # takes a drift of the radius away)
        slk = slowest_turn(om_r, ci, x, y)
        if slk is None:
            if (np.isfinite(om_r[ci]) & (np.abs(om_r[ci]) > MOVE_RAD_S)).any():
                ga_why.append(f'clip {k}: no slow part covering a full turn ({TURN_BINS} sectors x {TURN_BIN_MIN} '
                              f'frames) among {SLOW_MIN}+ spinning frames, its geometry not checked')
            per_clip.append(dict(clip=k, slow_n=0))
            continue
        cxk, cyk, Rk = circle_fit(x[slk], y[slk])
        th[ci] = np.unwrap(np.arctan2(y[ci] - cyk, x[ci] - cxk))
        om[ci] = rate(t[ci], th[ci])
        v_t[ci] = np.abs(om[ci]) * Rk
        use_p[ci] = np.isfinite(om[ci]) & (np.abs(om[ci]) > MOVE_RAD_S) & (v_t[ci] <= CAP * v_max)
        ctx.clear()
        m = _bfit(slk)
        dk = float(np.max(_bres(m, slk)))
        ctx['full'] = (m[0], m[1])
        ho_t, why_t = held_out(_bfit, _bres, t, slk)
        tam_chung = bool(ctx.get('tam_chung'))
        khong_hai_hoa = bool(ctx.get('khong_hai_hoa'))
        kieu = []                       # Codex G3B3-1: what a failed held-out check saw (radius change or centre move)
        if ho_t is None:
            ga_why.append(f'clip {k}: held-out by time: {why_t}')
        elif ho_t > GA_TOL_M:
            ga_why.append(f'clip {k}: held-out by time residual {ho_t * 1e3:.2f} mm > {GA_TOL_M * 1e3:.0f} mm')
        if dk > GA_TOL_M:
            ga_why.append(f'clip {k}: residual {dk * 1e3:.2f} mm > {GA_TOL_M * 1e3:.0f} mm')
        # set-up moved between two spins of the clip (Codex G3B-1 round 2)
        run = np.full(len(t), -1)
        run[ci] = spin_runs(t[ci], om[ci])
        runs_ci = [int(r_) for r_ in np.unique(run[ci]) if r_ >= 0]
        ho_r, why_r, unchecked = held_out_runs(_bfit, _bres, slk, runs_ci, run)
        ga_why += [f'clip {k}: {w}' for w in why_r]
        if ho_r is not None and ho_r > GA_TOL_M:
            ga_why.append(f'clip {k}: held-out by run residual {ho_r * 1e3:.2f} mm > {GA_TOL_M * 1e3:.0f} mm')
        not_checked = ci[np.isin(run[ci], unchecked)]
        use_p[not_checked] = False
        # Codex G3B3-1, G3B4-1: a failed held-out check (by time or by run) is classified on EVERY split of the clip,
        # on the slow frames and on the frames used (up to CAP x v_max, the spins checked): two time halves, and each
        # checked spin against the other checked spins. A split whose two parts each cover a full turn fixes both
        # circles, so a moved centre is identified ('tam') whenever any split can see it; 'khong_nhan_dang' is left
        # only when every split has a part under a turn
        if (ho_t is not None and ho_t > GA_TOL_M) or (ho_r is not None and ho_r > GA_TOL_M):
            used_k = ci[use_p[ci]]
            for nm, sel in (('', slk), (' khung_dung', used_k)):
                if len(sel) >= 2:
                    kieu.append(dict(phep='thoi_gian' + nm, **split_kind(x, y, *halves(t, sel))))
                for r_ in runs_ci:
                    if r_ in unchecked:
                        continue
                    own, oth = sel[run[sel] == r_], sel[(run[sel] != r_) & ~np.isin(run[sel], unchecked)]
                    if len(own) and len(oth):
                        kieu.append(dict(phep=f'luot {r_}{nm}', **split_kind(x, y, own, oth)))
        # radial part (Codex G3B-5 round 3): the true radius of the set-up is one unknown constant within the clip, so
        # the error of a frame can be as large as the PEAK-TO-PEAK of the measured radius over the frames used (a drift
        # is not halved by taking the deviation from the mean); harmonics and drift not removed
        rk = np.hypot(x[ci] - cxk, y[ci] - cyk)
        rad_err[ci] = np.abs(rk - Rk)
        uk = use_p[ci] & np.isfinite(rk)
        rad_ptp_k = float(np.ptp(rk[uk])) if uk.any() else None
        # heading against the radius: the tag may sit at another angle on the set-up in each clip
        phi_k = psi[ci] - th[ci]
        hp_err[ci] = np.abs(np.angle(np.exp(1j * (phi_k - np.median(psi[slk] - th[slk])))))
        co_k = m[2]
        per_clip.append(dict(clip=k, slow_n=int(len(slk)), centre=[cxk, cyk], R0_m=Rk, ga_dev_m=dk,
                             ga_giu_ngoai_thoi_gian_m=ho_t, giu_ngoai_tam_chung=tam_chung,
                             giu_ngoai_khong_hai_hoa=khong_hai_hoa,
                             n_luot_quay=len(runs_ci), ga_giu_ngoai_luot_m=ho_r, luot_khong_kiem=unchecked,
                             n_khung_khong_kiem=int(len(not_checked)), kieu_lech=kieu, ban_kinh_dinh_dinh_m=rad_ptp_k,
                             v_t_max_m_s=float(np.nanmax(v_t[ci])) if np.isfinite(v_t[ci]).any() else None,
                             mo_hinh_ban_kinh=dict(bac1_m=float(np.hypot(co_k[1], co_k[2])),
                                                   elip_m=float(np.hypot(co_k[3], co_k[4])), troi_m_s=float(co_k[5]))))
    ok_clips = [d for d in per_clip if d['slow_n']]
    if not ok_clips:
        ga_why.append('no clip with a slowest part')
    ga_dev = max((d['ga_dev_m'] for d in ok_clips), default=None)
    hos = [v for d in ok_clips for v in (d['ga_giu_ngoai_thoi_gian_m'], d['ga_giu_ngoai_luot_m']) if v is not None]
    # the frames used: tangential speed of each frame from its own clip's radius, up to v_max, each direction
    cov = coverage(np.sign(om[use_p]) * v_t[use_p], v_max)
    use_r = use_p & np.isfinite(rad_err)
    rad_lech_tb = float(np.max(rad_err[use_r])) if use_r.any() else None      # deviation from the mean (information)
    # Codex G3B4-2: a clip with no frame in the band used gives no frame to the position part, so it adds no radius
    # (its None does not void the clips that have frames used); None only when no clip has a frame used
    ptps = [d['ban_kinh_dinh_dinh_m'] for d in ok_clips if d['ban_kinh_dinh_dinh_m'] is not None]
    rad_dev = max(ptps) if ptps else None
    rad_dev_tho = float(np.max(np.abs(r[use_p] - R0))) if use_p.any() else None
    use_h = use_p & np.isfinite(hp_err)
    hp_dev = float(np.max(hp_err[use_h])) if use_h.any() else None
    # rolling shutter: one alpha over all spinning frames, and one per speed band x direction (pieces and
    # harmonics shared: within one band the rate hardly changes, so a band alone cannot tell the shutter term from
    # the first harmonic)
    spin = trim_runs(t, om, np.isfinite(om) & (np.abs(om) > SHUTTER_MIN_RAD_S))
    w = fit_shutter(t, th, om, F['row'], F['col'], spin)
    whole = dict(w['groups'][0], pieces=w['pieces'], rms_rad=w['rms_rad']) if w else None
    # consistency (design 4b: the same alpha over the speed bands and the two directions): the shutter term is told
    # from the harmonics only by a change of the rate (speed or sign), so each check fit takes a pair with that lever
    # and its own harmonics: low band (both directions), high band (both directions), +direction (both bands),
    # -direction (both bands) [CACH HIEU]
    subsets = []
    if spin.any():
        med = float(np.median(np.abs(om[spin])))
        lo_b, hi_b = spin & (np.abs(om) <= med), spin & (np.abs(om) > med)
        for name, s_ in (('dai_thap', lo_b), ('dai_cao', hi_b), ('chieu_duong', spin & (om > 0)),
                         ('chieu_am', spin & (om < 0))):
            if s_.sum() >= SUBSET_MIN and (s_ & (om > 0)).any() and (s_ & (om < 0)).any()                     or name.startswith('chieu') and s_.sum() >= SUBSET_MIN:
                f = fit_shutter(t, th, om, F['row'], F['col'], s_)
                if f:
                    subsets.append(dict(f['groups'][0], ten=name, n_pieces=f['pieces']))
    subsets = [d for d in subsets if 'alpha_r_s_px' in d]
    use_alpha, why = False, ''
    if whole is None:
        why = 'no fit'
    elif len(subsets) < 4:
        why = f'{len(subsets)} of 4 check fits (two bands, two directions) with {SUBSET_MIN} frames'
    elif not (whole['has_col'] and all(d['has_col'] for d in subsets)):
        why = 'no col_px in the whole fit or a check fit: alpha_c not measured, one frame kept (C8BS-11)'
    else:
        # C8BS-12: both coefficients, not only one summary number: alpha_r within SPREAD_MAX of its mean and one sign;
        # alpha_c within SPREAD_MAX x |mean alpha_r| of its mean (alpha_c may be near 0) [DE XUAT]; and the whole fit
        # inside the spread of the check fits
        ar = np.array([d['alpha_r_s_px'] for d in subsets])
        ac = np.array([d['alpha_c_s_px'] for d in subsets])
        m_r = float(np.mean(ar))
        spread = float((ar.max() - ar.min()) / abs(m_r)) if m_r != 0 else math.inf
        spread_c = float((ac.max() - ac.min()) / abs(m_r)) if m_r != 0 else math.inf
        whole.update(spread=spread, spread_c=spread_c)
        if len({int(np.sign(v)) for v in ar}) != 1:
            why = 'alpha_r changes sign between the check fits'
        elif spread > SPREAD_MAX:
            why = f'alpha_r spread {spread:.2f} > {SPREAD_MAX}'
        elif spread_c > SPREAD_MAX:
            why = f'alpha_c spread {spread_c:.2f} x |alpha_r| > {SPREAD_MAX}'
        elif not (ar.min() - 1e-15 <= whole['alpha_r_s_px'] <= ar.max() + 1e-15
                  and ac.min() - 1e-15 <= whole['alpha_c_s_px'] <= ac.max() + 1e-15):
            why = 'the whole fit is outside the check fits'
        else:
            use_alpha = True
    # C8BS-12: the value used covers every check fit: the envelope of |alpha_r|, |alpha_c| over the whole fit and
    # the check fits (each coefficient on its own), so |a_r| Dr + |a_c| Dc is not below any fit for any span
    env_r = max([abs(whole['alpha_r_s_px'])] + [abs(d['alpha_r_s_px']) for d in subsets]) if whole else None
    env_c = max([abs(whole['alpha_c_s_px'])] + [abs(d['alpha_c_s_px']) for d in subsets]) if whole else None
    rows = F['row'][spin]
    return dict(centre=[cx, cy], R0_m=R0, slow_n=int(len(sl)), slow_rate_rad_s=float(np.median(np.abs(om[sl]))),
                ga_dev_m=ga_dev, ga_dev_giu_ngoai_m=max(hos) if hos else None, ga_ok=not ga_why, ga_why=ga_why,
                ga_dev_tho_m=ga_dev_tho, theo_clip=per_clip,
                du_phu=cov['ok'], phu=cov, rad_dev_m=rad_dev, rad_dev_tho_m=rad_dev_tho, rad_lech_tb_m=rad_lech_tb,
                rad_kieu='dinh-dinh ban kinh tren khung dung, tung clip (Codex G3B-5 vong 3)',
                tiep_tuyen='khong do: thieu chuan pha doc lap (Codex G3B-6); P4 cua B chi gom ban kinh',
                hp_dev_deg=math.degrees(hp_dev) if hp_dev is not None else None,
                rate_max_rad_s=float(np.nanmax(np.abs(om))), v_t_max_m_s=float(np.nanmax(v_t)),
                row_span_px=[float(np.nanmin(rows)), float(np.nanmax(rows))] if len(rows) else None,
                shutter=dict(whole=whole, subsets=subsets, use_alpha=use_alpha, why=why,
                             alpha_r_s_px=env_r if use_alpha else None,
                             alpha_c_s_px=env_c if use_alpha else None,
                             alpha_r_le_s_px=LE * env_r if use_alpha else None,
                             alpha_c_le_s_px=LE * env_c if use_alpha else None,
                             alpha_kieu='bao |alpha| cua fit chung va cac fit kiem (C8BS-12)',
                             mot_khung_s=frame_s))


def cross_check(fc, a, wz_max):
    """The tag on the axis ("A cu"): the same axis model, as a check of the model only (q small, residual within
    GA_TOL_M); its heading oscillation and position parts are not used. Axis distance to A: information only (the
    set-up may have moved between the clips)."""
    c = analyse_a(read(fc), wz_max)
    return dict(file=os.path.basename(fc), q_abs_m=c['q_abs_m'], q_nhan_dang=c['q_nhan_dang'], ga_dev_m=c['ga_dev_m'],
                ga_dev_giu_ngoai_m=c['ga_dev_giu_ngoai_m'], ga_dev_tho_m=c['ga_dev_tho_m'],
                khop_mo_hinh_ok=c['ga_ok'], ly_do=c['ga_why'], theo_clip=c['theo_clip'],
                ghi_chu=('khop_mo_hinh_ok chi xac nhan mo hinh truc khop clip nay (Codex G3B-7); khong xac nhan ma gan '
                         'truc, khong anh huong ket qua A/B, khong dung lam can huong hay vi tri'))


def b_troi_only(b):
    """L132: every reason B failed the set-up check is a held-out residual by time or by spin, not a residual in the
    fit, a part of a turn, too few frames or q not identified; AND no failed held-out comparison IDENTIFIES a moved
    centre (both parts' circles fixed by a full turn and their centres more than TAM_TOL_M apart: the set-up moving,
    Codex G3B3-1; every split of the clip is tried, on the slow frames and on the frames used, Codex G3B4-1). A
    comparison that cannot tell (a part under a turn, 'khong_nhan_dang') does not block: the radial
    part is the peak-to-peak radius over the frames used (G3B-5 round 3), so the unexplained gap is counted in it as an
    error (the safe side; 2 slow frames at the start of B IMG_3602 of session 1 are this case)."""
    w = b['ga_why']
    if not w or not all(re.search(r'held-out by (time|run) residual [0-9.]+ mm > ', x) for x in w):
        return False
    ks = [kk for d in b.get('theo_clip', []) for kk in d.get('kieu_lech') or []]
    return bool(ks) and not any(kk['kieu'] == 'tam' for kk in ks)


def analyse(fa, fb, task='O1-F08', fc=None, b_ke_ca_troi=False):
    from o1_feasibility import Schedule
    from o1_tasks import FIG8
    import o1_gate
    v_max, wz_max = o1_gate.fig8_cmd_max(Schedule(FIG8[task]))
    A = read(fa)
    B = read(fb)
    frame_s = float(np.median(np.diff(B['t'])))
    a = analyse_a(A, wz_max)
    b = analyse_b(B, v_max, wz_max, frame_s)
    cheo = cross_check(fc, a, wz_max) if fc else None
    ga_ok = a['ga_ok'] and b['ga_ok']
    # C8BS-10: every part measured over the required rates and both ways, else no number (nothing replaced by 0)
    parts_psi = [a['psi_osc_deg'], b['hp_dev_deg']]
    parts_p = [a['p_dev_m'], b['rad_dev_m']]
    thieu = [nm for nm, ok_ in (('A phu toc do / hai chieu', a['du_phu']), ('B phu toc do / hai chieu', b['du_phu']),
                                ('A dao dong huong', parts_psi[0] is not None), ('B huong-vi tri', parts_psi[1] is not None),
                                ('A vi tri', parts_p[0] is not None), ('B ban kinh', parts_p[1] is not None)) if not ok_]
    # L132 (Liam 19/11 at CODE, "Dung B buoi 1 ke ca troi (Recommended)"): B of session 1 used although its set-up
    # check failed by the drift of the radius only; the drift is in rad_dev (not removed), so it counts as an error
    b_troi = bool(b_ke_ca_troi and a['ga_ok'] and not b['ga_ok'] and b_troi_only(b))
    usable = (ga_ok or b_troi) and not thieu
    return dict(version=VERSION, task_rates=task, v_max=v_max, wz_max=wz_max, A=a, B=b, A_cheo=cheo,
                ga_ok=ga_ok, thieu=thieu,
                b_ke_ca_troi_yeu_cau=bool(b_ke_ca_troi), b_dung_ke_ca_troi=b_troi,
                b_ke_ca_troi_nguon=('L132, Liam 19/11 phien CODE local_e6d36dbe: "Dung B buoi 1 ke ca troi '
                                    '(Recommended)"; gioi han: khong tach troi ga voi sai camera') if b_troi else None,
                dung_duoc=usable,
                d_psi_quay_deg=max(parts_psi) if usable else None,
                d_p_quay_m=max(parts_p) if usable else None,
                gioi_han=['sai tron bi khop hap thu', 'B: sai tiep tuyen khong do (thieu chuan pha)', 'sai pha chung huong va vi tri tiep tuyen', 'lech khong doi '
                          'theo toc do (phep 2)', 'rung, nghieng xe that',
                          'doi ga trong mot luot quay khong tach duoc voi sai do theo toc do (nam trong p_dev, rad_dev)'])


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames-a', required=True)
    ap.add_argument('--frames-b', required=True)
    ap.add_argument('--frames-a-cheo', default=None, help='tag on the axis ("A cu"): cross-check of the axis model')
    ap.add_argument('--task', default='O1-F08', help='level whose command peaks limit the rates (design 4b: F08)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--b-ke-ca-troi', action='store_true',
                    help='L132: use B although its set-up check failed by the drift of the radius only (session 1)')
    a = ap.parse_args(argv)
    try:
        out = analyse(a.frames_a, a.frames_b, a.task, a.frames_a_cheo, b_ke_ca_troi=a.b_ke_ca_troi)
        out['nguon'] = [dict(vai=v, file=os.path.basename(p), sha256=cu.sha256(p))
                        for v, p in (('A', a.frames_a), ('B', a.frames_b), ('A_cheo', a.frames_a_cheo)) if p]
        out['script'] = cu.script_state()
        cu.check_path(a.out)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(_clean(out), indent=1, ensure_ascii=False, allow_nan=False))
    except (TbError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_3b: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    sh = out['B']['shutter']
    print(f'ga_ok {out["ga_ok"]} (A {out["A"]["ga_why"] or "dat"}; B {out["B"]["ga_why"] or "dat"}); '
          f'thieu {out["thieu"]}; b_dung_ke_ca_troi {out["b_dung_ke_ca_troi"]}; '
          f'd_psi_quay {out["d_psi_quay_deg"]} deg; d_p_quay {out["d_p_quay_m"]} m; '
          f'alpha {"dung" if sh["use_alpha"] else "khong dung: " + sh["why"]}')
    if out['A_cheo'] is not None:
        c = out['A_cheo']
        def _mm(v):                                     # Codex G3B-7: a null field prints as such, no TypeError
            return 'null' if v is None else f'{v * 1e3:.2f} mm'
        print(f'A cheo (chi kiem mo hinh): khop {c["khop_mo_hinh_ok"]}, |q| {_mm(c["q_abs_m"])}, '
              f'ga_dev {_mm(c["ga_dev_m"])}')
    print(f'#ket_qua,ma=0,ga_ok={int(out["ga_ok"])},alpha_dung={int(sh["use_alpha"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
