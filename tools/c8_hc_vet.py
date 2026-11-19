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
centre to nominal body centre, [de xuat] 20 mm), plus the return error after 12b. Gaps before any further margin.
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


def sweep_12b(cho, hyp, tags, lech, lech_deg, qua_da, ngang):
    stops, _ = hs.PLACES[cho]
    lo, hi = min(stops) - qua_da, max(stops) + qua_da
    x0, y0, h0 = START[cho]
    shapes0 = [swept_x(p, lo, hi, ngang) for p in pieces(hyp)]
    nominal = gaps([place(s, x0, y0, h0) for s in shapes0], tags)
    worst = (math.inf, None)
    for dx, dy, dh in itertools.product(grid(lech), grid(lech), grid(lech_deg)):
        g = gaps([place(s, x0 + dx, y0 + dy, h0 + dh) for s in shapes0], tags)
        if g[0] < worst[0]:
            worst = g
    return nominal, worst


def reach(hyp):
    return max(float(np.max(np.hypot(p[:, 0], p[:, 1]))) for p in pieces(hyp))


def sweep_E(cho, hyp, tags, lech, d, tro_ve):
    x0, y0, _ = START[cho]
    r = reach(hyp)
    nominal = min((disc_dist((x0, y0), r, q), tid) for tid, q in tags.items())
    e = lech + tro_ve
    rr = r + 2 * d
    worst = (math.inf, None)
    for dx, dy in itertools.product(grid(e), grid(e)):
        g = min((disc_dist((x0 + dx, y0 + dy), rr, q), tid) for tid, q in tags.items())
        if g[0] < worst[0]:
            worst = g
    return nominal, worst, r


def vet_thuc(poses, tags, hyp):
    """Smallest gap of the real footprint: poses = iterable of (x, y, heading rad) of the BODY centre (floor)."""
    best = (math.inf, None)
    ps = pieces(hyp)
    for x, y, th in poses:
        g = gaps([place(p, x, y, math.degrees(th)) for p in ps], tags)
        if g[0] < best[0]:
            best = g
    return best


def run(a):
    tags = load_tags(a.moc, a.vien_mm / 1000)
    rows = []
    for cho in ('goc', 'K1', 'K2'):
        for hyp in HYPS:
            n, w = sweep_12b(cho, hyp, tags, a.lech_mm / 1000, a.lech_deg, a.qua_da_mm / 1000, a.ngang_mm / 1000)
            rows.append(dict(cho=cho, buoc='12b', gia_thuyet=hyp, khe_dung_dau_mm=n[0] * 1000, ma_dung_dau=n[1],
                             khe_xau_nhat_mm=w[0] * 1000, ma_xau_nhat=w[1]))
            if hs.PLACES[cho][1]:
                n, w, r = sweep_E(cho, hyp, tags, a.lech_mm / 1000, a.d_mm / 1000, a.tro_ve_mm / 1000)
                rows.append(dict(cho=cho, buoc='E', gia_thuyet=hyp, ban_kinh_mm=r * 1000, khe_dung_dau_mm=n[0] * 1000,
                                 ma_dung_dau=n[1], khe_xau_nhat_mm=w[0] * 1000, ma_xau_nhat=w[1]))
    ok = all(r['khe_xau_nhat_mm'] >= a.cong_mm for r in rows)
    return dict(version=VERSION, moc=os.path.abspath(a.moc), tham_so=vars(a).copy(), dong=rows, dat=ok)


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
