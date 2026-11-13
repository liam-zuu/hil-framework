#!/usr/bin/env python3
"""
c7_marks_from_tape.py: L104 (Liam 14/11, CODE local_9951f4b7, "ok"; given to CODE_CAM 14/11, box
"Giao L104 cho CODE_CAM (Recommended)"). The true centre of each ArUco floor mark is measured with a
steel tape as distances to fixed reference points, O = (0, 0) and X = (150, 0) cm by default
(guide 00_OVERVIEW/Setup_Guides/2026-11-10_Huong_dan_Dan_so8_va_moc_san.md section 4e), and since
Liam 14/11 (CODE_CAM, "Them Y, X2 (Recommended)") also Y = (0, 100) and X2 = (-150, 0) cm. This script
turns the distances into floor coordinates and writes the marker CSV that
`c7_measure.py calib --markers` reads (id, vai_tro, x_cm, y_cm, canh_den_mm, dict).

Input (template 00_OVERVIEW/Setup_Guides/2026-11-14_Mau_do_tam_moc.csv), one row per point:
  id               1-33 (ArUco), or a name (day_doi = plumb point under the lens), or a reference:
                   X (its d_O is the O-X distance, replaces the design 1500 mm), Y (d_O, d_X: its true
                   position), X2 (d_O, d_X, d_Y: its true position); not measured = design position
  d_<R>_mm_1, d_<R>_mm_2   distance from reference R to the centre, two readings (mm); R = O, X, and
                   any reference added with --ref (e.g. Y). Readings more than 2 mm apart: DO_LAI
  phia             tren / duoi (y > 0 / y < 0); used when there is no design position
  canh_den_mm      black side measured on the print (empty: the design value of the design CSV)
  nguoi_do, ngay, ghi_chu   kept in the output
Design positions (side, deviation): the ArUco CSVs (2026-11-10_ArUco_in_C8*.csv, *_ma_kiem.csv).

Solution: least squares on the distances (Gauss-Newton), started from the design position (so with two
references the solution on the design side of the O-X line is taken), or from the closed form with
`phia` for a point without a design position. Checks per point:
  KHONG_CAT   with two references, the two circles do not meet (d_O + d_X < L or |d_O - d_X| > L):
              a reading is wrong
  DU_LON      least-squares residual RMS > 2 mm (three or more references)
  LECH        centre more than --warn-dev mm (default 20) from the design position
  KEM         1-sigma uncertainty of x or y above --warn-sigma mm (default 3), from a reading
              error of --sigma-d mm (default 1, an ESTIMATE of tape reading): points on or near the
              O-X line get a poor y (both circles cross almost tangentially); a third reference off
              the line (--ref Y=0,1000) fixes it
  PHIA        the solution and `phia` disagree
Rows with DO_LAI, KHONG_CAT or DU_LON are not written to the marker CSV (no centre). Thresholds are
[DE XUAT].

  python tools/c7_marks_from_tape.py --measured do_tam_moc.csv \
      --design <ArUco_in_C8.csv> <..._vung_noi.csv> <..._ma_kiem.csv> --out <dir> [--ref Y=0,1000]
  -> <dir>/moc_san_do.csv (for calib --markers), <dir>/bao_cao.txt; the plumb point is printed as
     the --plumb X,Y argument of calib (metres).
Last line: #ket_qua,ma=0 done / 1 done with warnings / 2 error.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

READ_TOL_MM = 2.0
# Reference points (mm). L104 (Liam 14/11 CODE "ok"): O, X. Y and X2 added by Liam 14/11 in CODE_CAM,
# box "Them Y, X2 (Recommended)": with O, X only, 14 of the 33 design positions had a 1-sigma error
# above 3 mm for a 1 mm reading error (estimate), the marks on the x axis had no usable y; with
# O, X, Y, X2 the largest is 2.5 mm, median 1.0 mm. Order matters: Y is located from O, X, then X2
# (on the x axis) from O, X, Y.
DEFAULT_REFS = {'O': (0.0, 0.0), 'X': (1500.0, 0.0), 'Y': (0.0, 1000.0), 'X2': (-1500.0, 0.0)}
RES_TOL_MM = 2.0
OUT_COLS = ['id', 'vai_tro', 'x_cm', 'y_cm', 'canh_den_mm', 'dict', 'lech_tk_mm', 'sigma_x_mm',
            'sigma_y_mm', 'canh_nguon', 'nguoi_do', 'ngay', 'ghi_chu']


def parse_ref(s):
    name, xy = s.split('=')
    x, y = (float(v) for v in xy.split(','))
    return name.strip(), (x * 10.0, y * 10.0)          # cm -> mm


def load_design(paths):
    """{id: dict(x_mm, y_mm, vai_tro, canh_den_mm, dict)} for rows with a centre."""
    out = {}
    for p in paths:
        with open(p, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('x_cm', '') in ('', None) or r.get('y_cm', '') in ('', None):
                    continue
                out[r['id'].strip()] = dict(x_mm=float(r['x_cm']) * 10, y_mm=float(r['y_cm']) * 10,
                                            vai_tro=r.get('vai_tro', ''), canh_den_mm=r.get('canh_den_mm', ''),
                                            dict=r.get('dict', 'DICT_4X4_50'))
    return out


def readings(row, ref):
    """Mean of the readings of reference ref (mm), their spread, or (None, None) when none given."""
    v = [float(row[k]) for k in (f'd_{ref}_mm_1', f'd_{ref}_mm_2', f'd_{ref}_mm')
         if row.get(k, '') not in ('', None)]
    if not v:
        return None, None
    return float(np.mean(v)), float(max(v) - min(v))


def two_circle(d0, d1, L):
    """Closed form for refs at (0, 0) and (L, 0): (x, |y|) or None when the circles do not meet."""
    if d0 + d1 < L or abs(d0 - d1) > L:
        return None
    x = (d0 * d0 - d1 * d1 + L * L) / (2 * L)
    return x, math.sqrt(max(d0 * d0 - x * x, 0.0))


def solve(refs, dists, p0, iters=50):
    """Least squares point from distances to refs (mm), started at p0. Returns (p, residuals, J)."""
    p = np.array(p0, float)
    R = np.array(refs, float)
    d = np.array(dists, float)
    for _ in range(iters):
        diff = p[None, :] - R
        r = np.linalg.norm(diff, axis=1)
        r = np.where(r < 1e-9, 1e-9, r)
        J = diff / r[:, None]
        step, *_ = np.linalg.lstsq(J, d - r, rcond=None)
        p = p + step
        if np.linalg.norm(step) < 1e-6:
            break
    diff = p[None, :] - R
    r = np.linalg.norm(diff, axis=1)
    J = diff / np.where(r < 1e-9, 1e-9, r)[:, None]
    return p, d - r, J


def sigma_xy(J, sigma_d):
    """1-sigma of (x, y) from independent reading errors sigma_d (mm)."""
    JtJ = J.T @ J
    if abs(np.linalg.det(JtJ)) < 1e-12:
        return float('inf'), float('inf')
    C = np.linalg.inv(JtJ) * sigma_d ** 2
    return float(math.sqrt(C[0, 0])), float(math.sqrt(C[1, 1]))


def process(rows, design, refs, sigma_d=1.0, warn_dev=20.0, warn_sigma=3.0):
    """rows: dicts of the measured CSV. refs: {name: (x_mm, y_mm)} (O and X first). Returns
    (points, report lines, n_errors, n_warnings, L used)."""
    refs = dict(refs)
    lines, n_err, n_warn = [], 0, 0
    xrow = [r for r in rows if r['id'].strip().upper() == 'X']
    if xrow:
        L, spread = readings(xrow[0], 'O')
        if L is not None:
            if spread > READ_TOL_MM:
                lines.append(f'X: O-X doc lech {spread:.1f} mm > {READ_TOL_MM:g}: DO_LAI (giu {refs["X"][0]:.0f} mm)')
                n_err += 1
            else:
                lines.append(f'O-X do lai {L:.1f} mm (thiet ke {refs["X"][0]:.0f} mm), dung so do')
                refs['X'] = (L, 0.0)
    # the other reference points (Y, X2, ...): true position from their distances to the references
    # fixed before them (in order), started at the design position; kept at design when not measured
    fixed = ['O', 'X']
    for name in [n for n in refs if n not in ('O', 'X')]:
        rr = [r for r in rows if r['id'].strip().upper() == name.upper()]
        use, dist, bad = [], [], False
        for n in fixed:
            m, sp = readings(rr[0], n) if rr else (None, None)
            if m is None:
                continue
            bad |= sp > READ_TOL_MM
            use.append(refs[n])
            dist.append(m)
        if len(use) >= 2 and not bad:
            p, res, J = solve(use, dist, refs[name])
            sx, sy = sigma_xy(J, sigma_d)
            ok = max(sx, sy) <= warn_sigma and (len(use) < 3 or float(np.sqrt(np.mean(res ** 2))) <= RES_TOL_MM)
            if ok:
                lines.append(f'{name}: do duoc ({p[0] / 10:.1f}, {p[1] / 10:.1f}) cm (thiet ke {refs[name][0] / 10:g}, '
                             f'{refs[name][1] / 10:g}), sigma {max(sx, sy):.1f} mm')
                refs[name] = (float(p[0]), float(p[1]))
            else:
                lines.append(f'{name}: so do khong dat (sigma {max(sx, sy):.1f} mm hoac du lon): giu vi tri thiet ke')
                n_warn += 1
        elif rr:
            lines.append(f'{name}: {"DO_LAI" if bad else "thieu khoang cach"}: giu vi tri thiet ke')
            n_err += bad
            n_warn += not bad
        fixed.append(name)
    names = list(refs)
    ref_ids = {n.upper() for n in refs}
    points = []
    for row in rows:
        pid = row['id'].strip()
        if not pid or pid.upper() in ref_ids:
            continue
        flags = []
        use, dist = [], []
        for n in names:
            m, sp = readings(row, n)
            if m is None:
                continue
            if sp > READ_TOL_MM:
                flags.append(f'DO_LAI (d_{n} hai lan lech {sp:.1f} mm)')
            use.append(refs[n])
            dist.append(m)
        if not use:
            continue
        des = design.get(pid)
        phia = (row.get('phia') or '').strip().lower()
        if len(use) < 2:
            flags.append('THIEU (can it nhat hai khoang cach)')
        p = None
        if not any(f.startswith(('DO_LAI', 'THIEU')) for f in flags):
            L = refs['X'][0] if refs['X'][1] == 0 and refs['O'] == (0.0, 0.0) else None
            if 'O' in names and 'X' in names and len(use) == 2 and use == [refs['O'], refs['X']] and L:
                if two_circle(dist[0], dist[1], L) is None:
                    flags.append(f'KHONG_CAT (d_O {dist[0]:.0f} + d_X {dist[1]:.0f} so voi O-X {L:.0f} mm)')
            if not any(f.startswith('KHONG_CAT') for f in flags):
                if des is not None:
                    p0 = (des['x_mm'], des['y_mm'] if abs(des['y_mm']) > 1 else (-1.0 if phia == 'duoi' else 1.0))
                else:
                    tc = two_circle(dist[0], dist[1], refs['X'][0]) if len(dist) >= 2 else None
                    p0 = (tc[0], -tc[1] if phia == 'duoi' else tc[1]) if tc else (0.0, -1.0 if phia == 'duoi' else 1.0)
                    if abs(p0[1]) < 1:
                        p0 = (p0[0], -1.0 if phia == 'duoi' else 1.0)
                p, res, J = solve(use, dist, p0)
                rms = float(np.sqrt(np.mean(res ** 2)))
                if len(use) >= 3 and rms > RES_TOL_MM:
                    flags.append(f'DU_LON (du {rms:.1f} mm)')
                sx, sy = sigma_xy(J, sigma_d)
                if max(sx, sy) > warn_sigma:
                    flags.append(f'KEM (sigma x {sx:.1f}, y {sy:.1f} mm voi sai so doc {sigma_d:g} mm)')
                dev = math.hypot(p[0] - des['x_mm'], p[1] - des['y_mm']) if des else None
                if dev is not None and dev > warn_dev:
                    flags.append(f'LECH ({dev:.0f} mm so voi thiet ke)')
                if phia in ('tren', 'duoi') and abs(p[1]) > 3 * max(sy, 1.0) and (p[1] > 0) != (phia == 'tren'):
                    flags.append('PHIA (nghiem khac phia ghi)')
        bad = any(f.startswith(('DO_LAI', 'THIEU', 'KHONG_CAT', 'DU_LON')) for f in flags)
        n_err += bad
        n_warn += (not bad) and bool(flags)
        pt = dict(id=pid, ok=not bad, flags=flags, row=row, des=des)
        if p is not None and not bad:
            canh = (row.get('canh_den_mm') or '').strip()
            pt.update(x_mm=float(p[0]), y_mm=float(p[1]), sigma_x=sx, sigma_y=sy, dev=dev,
                      canh=canh if canh else (des['canh_den_mm'] if des else ''),
                      canh_src='do' if canh else ('thiet_ke' if des else ''))
        points.append(pt)
        where = f'({p[0] / 10:.1f}, {p[1] / 10:.1f}) cm' if p is not None and not bad else '-'
        lines.append(f'{pid:>8}: {where:>22}  ' + ('; '.join(flags) if flags else 'OK'))
    return points, lines, n_err, n_warn, refs['X'][0]


def write_markers(path, points):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        for p in points:
            if not p['ok'] or not p['id'].isdigit():
                continue
            d, r = p['des'] or {}, p['row']
            w.writerow(dict(id=p['id'], vai_tro=d.get('vai_tro', 'moc_san'), x_cm=f'{p["x_mm"] / 10:.2f}',
                            y_cm=f'{p["y_mm"] / 10:.2f}', canh_den_mm=p['canh'], dict=d.get('dict', 'DICT_4X4_50'),
                            lech_tk_mm='' if p['dev'] is None else f'{p["dev"]:.1f}',
                            sigma_x_mm=f'{p["sigma_x"]:.2f}', sigma_y_mm=f'{p["sigma_y"]:.2f}',
                            canh_nguon=p['canh_src'], nguoi_do=r.get('nguoi_do', ''), ngay=r.get('ngay', ''),
                            ghi_chu=r.get('ghi_chu', '')))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--measured', required=True)
    ap.add_argument('--design', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref', action='append', default=[], help='extra reference NAME=x_cm,y_cm (column d_NAME_mm_*)')
    ap.add_argument('--sigma-d', type=float, default=1.0, help='tape reading error, mm (estimate)')
    ap.add_argument('--warn-dev', type=float, default=20.0)
    ap.add_argument('--warn-sigma', type=float, default=3.0)
    a = ap.parse_args(argv)
    try:
        refs = dict(DEFAULT_REFS)
        for s in a.ref:
            n, xy = parse_ref(s)
            refs[n] = xy
        design = load_design(a.design)
        with open(a.measured, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        points, lines, n_err, n_warn, L = process(rows, design, refs, a.sigma_d, a.warn_dev, a.warn_sigma)
        os.makedirs(a.out, exist_ok=True)
        out_csv = os.path.join(a.out, 'moc_san_do.csv')
        if os.path.exists(out_csv):
            raise FileExistsError(out_csv)
        write_markers(out_csv, points)
        plumb = [p for p in points if p['id'].lower() == 'day_doi' and p['ok']]
        head = [f'Do tam moc (L104): {len(points)} diem, loi {n_err}, canh bao {n_warn}; O-X = {L:.1f} mm; '
                f'diem tham chieu {", ".join(f"{k}=({v[0] / 10:g},{v[1] / 10:g}) cm" for k, v in refs.items())}; '
                f'sai so doc gia dinh {a.sigma_d:g} mm (uoc luong)']
        if plumb:
            head.append(f'Day doi: --plumb {plumb[0]["x_mm"] / 1000:.4f},{plumb[0]["y_mm"] / 1000:.4f}')
        txt = '\n'.join(head + lines)
        with open(os.path.join(a.out, 'bao_cao.txt'), 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
        print(txt)
        rc = 0 if n_err == 0 and n_warn == 0 else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f'LOI: {exc}')
        rc = 2
    print(f'#ket_qua,ma={rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
