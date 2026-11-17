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
  d_psi_quay, d_p_quay = the largest of these parts up to the F08 rates (heading: |Omega| <= wz_max of O1-F08 for
      (A); position and the (B) heading-position part: tangential speed |Omega| R <= v_max of O1-F08).
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

Checks (design 4b "Kiem ga"): the slowest part: (A) the centre still within GA_TOL_M (2 mm), (B) the radius constant
within 2 mm; else the set-up must be redone (ga_ok false, numbers not used).

  python tools/c8b_3b.py --frames-a A/frames.csv --frames-b B/frames.csv --out 3b.json
Last line: #ket_qua,ma=0,ga_ok=1|0,alpha_dung=1|0 / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_3b_v1'
LE = 1.25
GA_TOL_M = 0.002            # design 4b "Kiem ga"
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
PHU_FRAC = 0.8              # [DE XUAT] C8BS-10: a part counts as measured up to the F08 peak when its used data reach
                            # 80 % of the peak in each direction
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


# ── (A) tag centre on the axis ────────────────────────────────────────────────────────────────────

def analyse_a(F, wz_max):
    t, x, y, psi = F['t'], F['x'], F['y'], F['psi']
    om = rate(t, psi)
    sl = slowest(om)
    px, py = float(np.median(x[sl])), float(np.median(y[sl]))
    dev_sl = float(np.max(np.hypot(x[sl] - px, y[sl] - py)))
    dev = np.hypot(x - px, y - py)
    use = np.isfinite(om) & (np.abs(om) > MOVE_RAD_S) & (np.abs(om) <= wz_max)
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
            if math.isfinite(w) and MOVE_RAD_S < w <= wz_max:
                res_max = max(res_max, float(np.max(np.abs(r))))
                n_win += 1
                win_rates.append(float(np.sign(om_m[0])) * w)
            bands.append((w, float(np.max(np.abs(r)))))
        t0 += WIN_S / 4
    cov_h = coverage(np.array(win_rates), wz_max)           # the heading windows used, each direction
    cov_p = coverage(om[use], wz_max)                        # the position frames used, each direction
    return dict(fixed_point=[px, py], slow_n=int(len(sl)), slow_rate_rad_s=float(np.median(np.abs(om[sl]))),
                ga_dev_m=dev_sl, ga_ok=bool(dev_sl <= GA_TOL_M), du_phu=cov_h['ok'] and cov_p['ok'],
                phu=dict(huong=cov_h, vi_tri=cov_p), n_win=n_win,
                psi_osc_deg=math.degrees(res_max) if n_win else None,
                p_dev_m=float(np.max(dev[use])) if use.any() else None,
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
    ga_dev = float(np.max(np.abs(r[sl] - R0)))
    phi = psi - th
    phi_sl = float(np.median(phi[sl]))
    v_t = np.abs(om) * R0
    use_p = np.isfinite(om) & (np.abs(om) > MOVE_RAD_S) & (v_t <= v_max)
    cov = coverage(om[use_p] * R0, v_max)          # the frames used: tangential speed up to v_max, each direction
    rad_dev = float(np.max(np.abs(r[use_p] - R0))) if use_p.any() else None
    hp_dev = float(np.max(np.abs(phi[use_p] - phi_sl))) if use_p.any() else None
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
                ga_dev_m=ga_dev, ga_ok=bool(ga_dev <= GA_TOL_M), du_phu=cov['ok'], phu=cov, rad_dev_m=rad_dev,
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


def analyse(fa, fb, task='O1-F08'):
    from o1_feasibility import Schedule
    from o1_tasks import FIG8
    import o1_gate
    v_max, wz_max = o1_gate.fig8_cmd_max(Schedule(FIG8[task]))
    A = read(fa)
    B = read(fb)
    frame_s = float(np.median(np.diff(B['t'])))
    a = analyse_a(A, wz_max)
    b = analyse_b(B, v_max, wz_max, frame_s)
    ga_ok = a['ga_ok'] and b['ga_ok']
    # C8BS-10: every part measured over the required rates and both ways, else no number (nothing replaced by 0)
    parts_psi = [a['psi_osc_deg'], b['hp_dev_deg']]
    parts_p = [a['p_dev_m'], b['rad_dev_m']]
    thieu = [nm for nm, ok_ in (('A phu toc do / hai chieu', a['du_phu']), ('B phu toc do / hai chieu', b['du_phu']),
                                ('A dao dong huong', parts_psi[0] is not None), ('B huong-vi tri', parts_psi[1] is not None),
                                ('A vi tri', parts_p[0] is not None), ('B ban kinh', parts_p[1] is not None)) if not ok_]
    usable = ga_ok and not thieu
    return dict(version=VERSION, task_rates=task, v_max=v_max, wz_max=wz_max, A=a, B=b, ga_ok=ga_ok, thieu=thieu,
                dung_duoc=usable,
                d_psi_quay_deg=max(parts_psi) if usable else None,
                d_p_quay_m=max(parts_p) if usable else None,
                gioi_han=['sai tron bi khop hap thu', 'sai pha chung huong va vi tri tiep tuyen', 'lech khong doi '
                          'theo toc do (phep 2)', 'rung, nghieng xe that'])


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
    ap.add_argument('--task', default='O1-F08', help='level whose command peaks limit the rates (design 4b: F08)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        out = analyse(a.frames_a, a.frames_b, a.task)
        out['nguon'] = [dict(file=os.path.basename(p), sha256=cu.sha256(p)) for p in (a.frames_a, a.frames_b)]
        out['script'] = cu.script_state()
        cu.check_path(a.out)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(_clean(out), indent=1, ensure_ascii=False, allow_nan=False))
    except (TbError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_3b: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    sh = out['B']['shutter']
    print(f'ga_ok {out["ga_ok"]}; thieu {out["thieu"]}; d_psi_quay {out["d_psi_quay_deg"]} deg; d_p_quay {out["d_p_quay_m"]} m; '
          f'alpha {"dung" if sh["use_alpha"] else "khong dung: " + sh["why"]}')
    print(f'#ket_qua,ma=0,ga_ok={int(out["ga_ok"])},alpha_dung={int(sh["use_alpha"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
