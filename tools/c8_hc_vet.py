#!/usr/bin/env python3
"""
c8_hc_vet.py: swept footprint of the car body and wheels during C8 steps 12b and E against the black squares of the 32
floor tags (CODE_HIEUCHUAN, 19/11/2026; design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_CODE_HIEUCHUAN_v4.md 2.5).

    python tools/c8_hc_vet.py --moc moc_san_do_32.csv [--out vet.json] [--vien-mm 10] [--lech-mm 30] [--lech-deg 3]
                              [--qua-da-mm 10] [--ngang-mm 2] [--d-mm 20] [--tro-ve-mm 15] [--cong-mm 20]

Model (Codex HC1-5, HC2-7): body rectangle 340 x 290 mm under BOTH hypotheses (340 along the car x axis, or across),
plus four wheel boxes (centres +-120 mm, half length 48.5 mm along x, half width 20 mm along y; firmware LX = LY =
0.12, WHEEL_RADIUS 0.0485); tags = black square of side canh_den_mm grown by --vien-mm each side (paper, tape) and by
3 sigma of the tag position (sigma_x_mm, sigma_y_mm of the CSV). Placement error: grid of +-lech mm on each floor axis
and +-lech deg (5 points each); translation: overshoot +-qua_da along the axis and +-ngang across (fwsim); turn in
place (E): disc of the farthest body/wheel point about the turning centre, the body centre up to d from it (C8B, tag
centre to nominal body centre, [de xuat] 20 mm, or d_vet_m of xoay_v1 with --xoay), plus the return error after 12b
and the drift of the turning centre during E (--troi-e-mm, default 10 mm = 2 x the fwsim largest 5.0 mm, ADRC K1,
19/11). Gaps before any further margin.
Continuous placement error (Codex code review HCC1-13): 12b takes the smallest gap of the grid AND of a bounded local
minimisation (scipy, Powell) started from the five lowest grid points, so a gap between grid samples is not missed;
E is exact: the turning centre anywhere in the square of +-(lech + tro_ve + troi_e), so the gap is the distance from
the tag to that square minus the disc radius.
Places and stops: rpi5/c8_hc_seq.PLACES. Also vet_thuc(): the real footprint from measured body poses (used by
tools/c8_hc_a.py tinh).
Last line: #ket_qua,ma=0 every gap >= --cong-mm / 1 some gap smaller / 2 error.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
import c8_hc_seq as hs                                                    # noqa: E402

VERSION = 'c8_hc_vet_v1'
BODY_LONG, BODY_SHORT = 0.340, 0.290
WHEEL_C, WHEEL_HL, WHEEL_HW = 0.12, 0.0485, 0.020
START = {'goc': (0.0, 0.0, 45.0), 'K1': (1.0, 0.0, 90.0), 'K2': (-0.5, -0.5, 0.0)}   # floor x, y [m], nose [deg]
HYPS = ('340_doc', '340_ngang')


def rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def pieces(hyp):
    """Convex pieces of the car in the body frame (x forward), body rectangle first."""
    L, W = (BODY_LONG, BODY_SHORT) if hyp == '340_doc' else (BODY_SHORT, BODY_LONG)
    out = [rect(-L / 2, L / 2, -W / 2, W / 2)]
    for sx, sy in itertools.product((-1, 1), (-1, 1)):
        cx, cy = sx * WHEEL_C, sy * WHEEL_C
        out.append(rect(cx - WHEEL_HL, cx + WHEEL_HL, cy - WHEEL_HW, cy + WHEEL_HW))
    return out


def swept_x(poly, lo, hi, ngang):
    """Convex hull of the piece translated along body x over [lo, hi] and across by +-ngang (a box sweep)."""
    x0, x1 = poly[:, 0].min() + lo, poly[:, 0].max() + hi
    y0, y1 = poly[:, 1].min() - ngang, poly[:, 1].max() + ngang
    return rect(x0, x1, y0, y1)


def place(poly, x, y, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return poly @ np.array([[c, s], [-s, c]]) + [x, y]


def load_tags(path, vien):
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            x, y = float(r['x_cm']) / 100, float(r['y_cm']) / 100
            sx = 3 * float(r.get('sigma_x_mm') or 0) / 1000
            sy = 3 * float(r.get('sigma_y_mm') or 0) / 1000
            h = float(r['canh_den_mm']) / 2000 + vien
            out[int(r['id'])] = rect(x - h - sx, x + h + sx, y - h - sy, y + h + sy)
    if len(out) != 32:
        raise ValueError(f'{path}: {len(out)} tags, expected 32')
    return out


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _overlap(P, Q):
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = P @ n, Q @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def poly_dist(P, Q):
    """Distance between two convex polygons, 0 when they overlap."""
    if _overlap(P, Q):
        return 0.0
    d = math.inf
    for A, B in ((P, Q), (Q, P)):
        for p in A:
            for i in range(len(B)):
                d = min(d, _seg_dist(p, B[i], B[(i + 1) % len(B)]))
    return d


def disc_dist(c, r, Q):
    """Distance from a disc to a convex polygon, 0 when they overlap."""
    c = np.asarray(c, float)

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    inside = all(cross(Q[(i + 1) % len(Q)] - Q[i], c - Q[i]) >= 0 for i in range(len(Q)))
    if inside:
        return 0.0
    d = min(_seg_dist(c, Q[i], Q[(i + 1) % len(Q)]) for i in range(len(Q)))
    return max(0.0, d - r)


def gaps(shapes, tags):
    best = (math.inf, None)
    for tid, q in tags.items():
        for s in shapes:
            g = poly_dist(s, q)
            if g < best[0]:
                best = (g, tid)
    return best


def grid(e):
    return (-e, -e / 2, 0.0, e / 2, e)


def sweep_12b(cho, hyp, tags, lech, lech_deg, qua_da, ngang, refine=True):
    """(nominal (gap, tag), worst (gap, tag), worst of the grid alone). worst = min of the grid and of a bounded
    local minimisation from the five lowest grid points (HCC1-13)."""
    stops, _ = hs.PLACES[cho]
    lo, hi = min(stops) - qua_da, max(stops) + qua_da
    x0, y0, h0 = START[cho]
    shapes0 = [swept_x(p, lo, hi, ngang) for p in pieces(hyp)]
    nominal = gaps([place(s, x0, y0, h0) for s in shapes0], tags)
    # a placement error moves every point of the shapes by at most lech*sqrt(2) + R*lech_deg (R: farthest point from
    # the car centre), so a tag whose nominal gap exceeds the smallest one by twice that can never be the worst (exact
    # pruning, the gap is 1-Lipschitz in the displacement)
    R = max(float(np.max(np.hypot(s[:, 0], s[:, 1]))) for s in shapes0)
    move = lech * math.sqrt(2) + R * math.radians(lech_deg)
    nom = {k: gaps([place(s, x0, y0, h0) for s in shapes0], {k: q})[0] for k, q in tags.items()}
    near = {k: tags[k] for k, g in nom.items() if g <= nominal[0] + 2 * move}

    def at(v):
        return gaps([place(s, x0 + v[0], y0 + v[1], h0 + v[2]) for s in shapes0], near)
    pts = [(at((dx, dy, dh)), (dx, dy, dh)) for dx, dy, dh in itertools.product(grid(lech), grid(lech), grid(lech_deg))]
    pts.sort(key=lambda z: z[0][0])
    grid_worst = pts[0][0]
    worst = grid_worst
    if refine:
        from scipy.optimize import minimize
        bounds = [(-lech, lech), (-lech, lech), (-lech_deg, lech_deg)]
        for g0, v0 in pts[:5]:
            r = minimize(lambda v: at(v)[0], np.array(v0, float), method='Powell', bounds=bounds,
                         options=dict(xtol=1e-5, ftol=1e-7, maxfev=400))
            g = at(np.clip(r.x, [b[0] for b in bounds], [b[1] for b in bounds]))
            if g[0] < worst[0]:
                worst = g
    return nominal, worst, grid_worst


def reach(hyp):
    return max(float(np.max(np.hypot(p[:, 0], p[:, 1]))) for p in pieces(hyp))


def sweep_E(cho, hyp, tags, lech, d, tro_ve, troi_e=0.0):
    """Exact over the continuous square of centre positions (HCC1-13): gap = dist(tag, square) - (reach + 2d)."""
    x0, y0, _ = START[cho]
    r = reach(hyp)
    nominal = min((disc_dist((x0, y0), r, q), tid) for tid, q in tags.items())
    e = lech + tro_ve + troi_e
    rr = r + 2 * d
    sq = rect(x0 - e, x0 + e, y0 - e, y0 + e)
    worst = min((max(0.0, poly_dist(sq, q) - rr), tid) for tid, q in tags.items())
    return nominal, worst, r


def vet_thuc(poses, tags, hyp):
    """Smallest gap of the real footprint: poses = iterable of (x, y, heading rad) of the BODY centre (floor)."""
    best = (math.inf, None)
    ps = pieces(hyp)
    r = reach(hyp) + 0.05
    for x, y, th in poses:
        near = {k: q for k, q in tags.items() if np.min(np.hypot(q[:, 0] - x, q[:, 1] - y)) < r + 0.5}
        if not near:
            continue
        g = gaps([place(p, x, y, math.degrees(th)) for p in ps], near)
        if g[0] < best[0]:
            best = g
    return best


V_BOUND, W_BOUND = 0.6, 1.0     # m/s, rad/s: Pi limits 0.15 / 0.6 + firmware outer loop up to 0.5 / 0.5 (v4 2.3)


def vet_thuc_camera(t, x, y, psi_tag, a, tags, d, hyps=HYPS, every=1):
    """Real footprint from the camera (HCC1-13), every `every`-th valid frame: the body heading is the tag heading
    - a, the body centre is within d of the measured centre (the chain gives the tag centre after q, not the body
    box centre), and between two used frames the car moves at most the largest step of the frames (translation +
    reach x turn), half of which is taken off. Returns dict(khe_m, ma, buoc_m, d_m, hyp) of the worst hypothesis.
    HCC2-6: every valid frame is used (every=1) and the step between two used frames is bounded in TIME
    (V_BOUND dt + reach W_BOUND dt), not by the pose difference, so a touch between frames or across a gap without
    the tag is covered (a long gap makes the gap negative: no evidence)."""
    t, x, y, psi_tag = (np.asarray(v, float) for v in (t, x, y, psi_tag))
    if len(t) < 2:
        return None
    th = np.unwrap(psi_tag) - a
    use = np.arange(0, len(t), every)
    if use[-1] != len(t) - 1:
        use = np.r_[use, len(t) - 1]
    out = None
    for hyp in hyps:
        r = reach(hyp)
        dt = np.diff(t[use])
        step = np.maximum(np.hypot(np.diff(x[use]), np.diff(y[use])) + r * np.abs(np.diff(th[use])),
                          (V_BOUND + r * W_BOUND) * dt)
        g, tid = vet_thuc(zip(x[use], y[use], th[use]), tags, hyp)
        k = g - d - (float(np.max(step)) / 2 if len(step) else 0.0)
        if out is None or k < out['khe_m']:
            out = dict(khe_m=k, ma=tid, buoc_m=float(np.max(step)) if len(step) else 0.0, d_m=d, hyp=hyp)
    return out


def run(a):
    tags = load_tags(a.moc, a.vien_mm / 1000)
    rows = []
    for cho in ('goc', 'K1', 'K2'):
        for hyp in HYPS:
            n, w, gw = sweep_12b(cho, hyp, tags, a.lech_mm / 1000, a.lech_deg, a.qua_da_mm / 1000, a.ngang_mm / 1000)
            rows.append(dict(cho=cho, buoc='12b', gia_thuyet=hyp, khe_dung_dau_mm=n[0] * 1000, ma_dung_dau=n[1],
                             khe_xau_nhat_mm=w[0] * 1000, ma_xau_nhat=w[1], khe_luoi_mm=gw[0] * 1000))
            if hs.PLACES[cho][1]:
                d, d_src = d_of(a, cho)
                n, w, r = sweep_E(cho, hyp, tags, a.lech_mm / 1000, d, a.tro_ve_mm / 1000, a.troi_e_mm / 1000)
                rows.append(dict(cho=cho, buoc='E', gia_thuyet=hyp, ban_kinh_mm=r * 1000, khe_dung_dau_mm=n[0] * 1000,
                                 ma_dung_dau=n[1], khe_xau_nhat_mm=w[0] * 1000, ma_xau_nhat=w[1], d_mm=d * 1000,
                                 d_nguon=d_src))
    ok = all(r['khe_xau_nhat_mm'] >= a.cong_mm for r in rows)
    import hashlib
    sha = hashlib.sha256(open(a.moc, 'rb').read()).hexdigest()
    return dict(version=VERSION, moc=os.path.abspath(a.moc), moc_sha256=sha, tham_so=vars(a).copy(), dong=rows, dat=ok)


def d_of(a, cho):
    """(d [m], source) for the E disc: d_vet_m of xoay_v1 (C8B) for this place when --xoay is given, else --d-mm
    ([de xuat])."""
    if getattr(a, 'xoay', None):
        import hashlib
        x = json.load(open(a.xoay, encoding='utf-8'))
        v = (x.get('cho', {}).get(cho) or {}).get('d_vet_m', x.get('d_vet_m'))
        if v is not None:
            return float(v), f"xoay_v1 {os.path.abspath(a.xoay)} sha256 {hashlib.sha256(open(a.xoay, 'rb').read()).hexdigest()}"
    return a.d_mm / 1000, f'tham so --d-mm {a.d_mm:g} [de xuat]'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--moc', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--vien-mm', type=float, default=10.0)
    ap.add_argument('--lech-mm', type=float, default=30.0)
    ap.add_argument('--lech-deg', type=float, default=3.0)
    ap.add_argument('--qua-da-mm', type=float, default=10.0)
    ap.add_argument('--ngang-mm', type=float, default=2.0)
    ap.add_argument('--d-mm', type=float, default=20.0)
    ap.add_argument('--tro-ve-mm', type=float, default=15.0)
    ap.add_argument('--troi-e-mm', type=float, default=10.0, help='E centre drift (2 x fwsim 5.0 mm)')
    ap.add_argument('--xoay', default=None, help='xoay_v1 of C8B: d_vet_m per place')
    ap.add_argument('--cong-mm', type=float, default=20.0)
    a = ap.parse_args(argv)
    try:
        res = run(a)
    except (OSError, ValueError, KeyError) as exc:
        print(f'loi: {exc}')
        print('#ket_qua,ma=2')
        return 2
    for r in res['dong']:
        print(f"{r['cho']:4s} {r['buoc']:3s} {r['gia_thuyet']:9s} dung dau {r['khe_dung_dau_mm']:7.1f} mm (ma {r['ma_dung_dau']})"
              f"  xau nhat {r['khe_xau_nhat_mm']:7.1f} mm (ma {r['ma_xau_nhat']})")
    if a.out:
        with open(a.out, 'x', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"#ket_qua,ma={0 if res['dat'] else 1}")
    return 0 if res['dat'] else 1


if __name__ == '__main__':
    sys.exit(main())
