#!/usr/bin/env python3
"""
c8b_d3_hang.py: the rows of D3 (separate position check, design C8B v7 section 8) for c8b_kiem_vi_tri.py, read from the
files of the sources; one CSV row per signed error of a source on its test set (tap kiem: made after the C8b lock).
CODE_C8B local_adeaeda8, 24/11/2026 (CODE local_42161302: "lam D3 truoc").

Columns (c8b_kiem_vi_tri COLS): nguon, tap, dai_luong, mien, sai_m, chi_so_m, co_chuan_dong. Every source file must be
one the C8b lock names (khoa_truoc.nguon, same sha256), so the index compared is the one the lock used.
Sources written here (the others of NGUON_CAN come when their owners deliver; c8b_kiem_vi_tri then lists them missing):
  do_nhay_cao       (5) c8b_u_h_v1 (c8b_u_h.py): every held-out row giu_ngoai[].sai_m (h predicted - h calibrated, m)
                    against u_h_chung_m; dai_luong h, mien chung; co_chuan_dong 0 (no dynamic height reference).
  troi_H            (7) c7_calib_drift files, one per camera segment (BUOI/doanN): sai_m = D_H_mm / 1000 against the
                    lock's eps_H.mm / 1000 (D_H only triggers a scoring again; design 8 (7)); a file without the end
                    clip H (thieu_H1) or without D_H is refused; co_chuan_dong 0.
  dau_cuoi_lan_lap  (6) F8 empty: the angle a between the tag axis and the car motion axis (F7 truc_xe_v1 of HIEUCHUAN)
                    of each later HC_goc against the first (session 2, F7 'khoa'): Liam 24/11 at CODE local_42161302,
                    box, verbatim: "Thay bang goc HC_goc (Recommended)". As a position at the largest lever arm of the A6
                    domain: sai_m = rho_max x rad(a_later - a_first), chi_so_m = rho_max x rad(u_a_noi_bo of the
                    first) (the same rho on both sides: the test is the angle test); dai_luong a_tam_xe, mien = place;
                    co_chuan_dong 0. Limit: heading only, not position; a later F7 'chan_doan' has no u of its own.

  python tools/c8b_d3_hang.py --khoa c8b_khoa_v5.json --a6 c8b_a6_v2.json --u-h c8b_u_h_chung_v1.json \\
      --drift 2026-11-20_run02/doan2=drift.json [--drift ...] --goc-a-dau F7_buoi2.json --goc-a 2026-11-23_run03=F7.json \\
      --out d3_rows.csv
Writes d3_rows.csv and d3_rows.csv.nguon.json (sources with sha256, rules, limits). Never overwrites.
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402
import c8b_kiem_vi_tri as kvt                                            # noqa: E402

VERSION = 'c8b_d3_hang_v1'
LIAM_DAU_CUOI = ('Liam 24/11 o CODE local_42161302, hop chon, nguyen van: "Thay bang goc HC_goc (Recommended)" '
                 '(nhat ky hop dong C8 24/11 khuya)')


class HangError(ValueError):
    pass


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _read(p):
    cu.check_path(p)
    with open(p, encoding='utf-8') as f:
        return json.load(f), cu.sha256(p)


def lock_shas(khoa, vai):
    kt = khoa.get('khoa_truoc') if isinstance(khoa.get('khoa_truoc'), dict) else {}
    return {n.get('sha256') for n in kt.get('nguon') or [] if isinstance(n, dict) and n.get('vai') == vai}


def _row(nguon, dl, mien, sai, cs, dong=False):
    return dict(nguon=nguon, tap='kiem', dai_luong=dl, mien=mien, sai_m=repr(float(sai)), chi_so_m=repr(float(cs)),
                co_chuan_dong='1' if dong else '0')


def rows_u_h(d, sha, khoa):
    if sha not in lock_shas(khoa, 'u_h_chung'):
        raise HangError(f'--u-h sha256 {sha[:12]} is not a u_h_chung source of the lock (index must be the one used)')
    if not isinstance(d, dict) or d.get('schema') != 'c8b_u_h_v1':
        raise HangError('--u-h: not a c8b_u_h_v1 file')
    cs = _num(d.get('u_h_chung_m'))
    g = d.get('giu_ngoai')
    if cs is None or cs < 0 or not isinstance(g, list) or not g:
        raise HangError('--u-h: u_h_chung_m / giu_ngoai missing')
    out = []
    for r in g:
        s = _num((r or {}).get('sai_m'))
        if s is None:
            raise HangError(f'--u-h: held-out row {r!r} without a finite sai_m')
        out.append(_row('do_nhay_cao', 'h', 'chung', s, cs))
    return out


def rows_drift(items, khoa):
    eps = _num((khoa.get('eps_H') or {}).get('mm')) if isinstance(khoa.get('eps_H'), dict) else None
    if eps is None or eps <= 0:
        raise HangError('the lock has no eps_H.mm')
    out, seen = [], set()
    for seg, d in items:
        if seg in seen:
            raise HangError(f'--drift {seg}: twice')
        seen.add(seg)
        if not isinstance(d, dict) or d.get('kind') != 'c7_calib_drift':
            raise HangError(f'--drift {seg}: not a c7_calib_drift file')
        dh = _num(d.get('D_H_mm'))
        if d.get('thieu_H1') or dh is None or dh < 0:
            raise HangError(f'--drift {seg}: no D_H (end clip H missing or D_H_mm not finite)')
        out.append(_row('troi_H', 'D_H', seg, dh / 1000.0, eps / 1000.0))
    return out


def rows_goc_a(f_dau, sha_dau, sau, khoa, a6):
    if sha_dau not in lock_shas(khoa, 'truc_xe'):
        raise HangError(f'--goc-a-dau sha256 {sha_dau[:12]} is not the F7 (truc_xe) of the lock')
    if not isinstance(f_dau, dict) or f_dau.get('schema') != 'truc_xe_v1' or f_dau.get('trang_thai') != 'khoa':
        raise HangError('--goc-a-dau: a truc_xe_v1 F7 with trang_thai khoa (the first HC_goc, session 2)')
    a0, u0, cho = _num(f_dau.get('a_k_deg')), _num(f_dau.get('u_a_noi_bo_deg')), f_dau.get('cho')
    if a0 is None or u0 is None or u0 <= 0:
        raise HangError('--goc-a-dau: a_k_deg / u_a_noi_bo_deg missing')
    rho = _num(((a6 or {}).get('mien') or {}).get('rho_max_m'))
    if rho is None or rho <= 0:
        raise HangError('--a6: mien.rho_max_m missing')
    out, gh = [], []
    for buoi, f, sha in sau:
        if not isinstance(f, dict) or f.get('schema') != 'truc_xe_v1' or f.get('cho') != cho:
            raise HangError(f'--goc-a {buoi}: not a truc_xe_v1 F7 of place {cho}')
        if f.get('session_sha256') == f_dau.get('session_sha256'):
            raise HangError(f'--goc-a {buoi}: the same camera session as the first HC_goc (not a later end)')
        a = _num(f.get('a_k_deg'))
        if a is None:
            raise HangError(f'--goc-a {buoi}: a_k_deg missing')
        out.append(_row('dau_cuoi_lan_lap', 'a_tam_xe', cho, rho * math.radians(a - a0), rho * math.radians(u0)))
        if f.get('trang_thai') != 'khoa':
            gh.append(f'{buoi}: F7 {f.get("trang_thai")!r} (no u of its own), sha256 {sha[:12]}')
    return out, dict(a_dau_deg=a0, u_a_noi_bo_dau_deg=u0, rho_max_m=rho, cho=cho, gioi_han_dau_sau=gh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--khoa', required=True, help='the C8b lock (c8b_khoa_v5; the combined one of the gate)')
    ap.add_argument('--a6', required=True, help='the A6 of the lock (c8b_a6_v2: mien.rho_max_m)')
    ap.add_argument('--u-h', default=None, help='c8b_u_h_v1 file (do_nhay_cao)')
    ap.add_argument('--drift', action='append', default=[], help='BUOI/doanN=c7_calib_drift json (troi_H), repeat')
    ap.add_argument('--goc-a-dau', default=None, help='F7 truc_xe_v1 of the first HC_goc (session 2, khoa)')
    ap.add_argument('--goc-a', action='append', default=[], help='BUOI=F7 truc_xe_v1 of a later HC_goc, repeat')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        khoa, ksha = _read(a.khoa)
        if not isinstance(khoa, dict) or khoa.get('version') != kvt.KHOA_VERSION:
            raise HangError(f'--khoa: not a {kvt.KHOA_VERSION} file')
        a6, a6sha = _read(a.a6)
        if a6sha not in lock_shas(khoa, 'a6'):
            raise HangError('--a6 is not the A6 of the lock')
        rows, src, info = [], [dict(vai='khoa', duong_dan=os.path.abspath(a.khoa), sha256=ksha),
                                dict(vai='a6', duong_dan=os.path.abspath(a.a6), sha256=a6sha)], {}
        if a.u_h:
            d, s = _read(a.u_h)
            rows += rows_u_h(d, s, khoa)
            src.append(dict(vai='do_nhay_cao', duong_dan=os.path.abspath(a.u_h), sha256=s))
        items = []
        for x in a.drift:
            seg, sep, p = x.partition('=')
            if not sep or not seg:
                raise HangError(f'--drift {x!r}: BUOI/doanN=path')
            d, s = _read(p)
            items.append((seg, d))
            src.append(dict(vai=f'troi_H {seg}', duong_dan=os.path.abspath(p), sha256=s))
        rows += rows_drift(items, khoa) if items else []
        if bool(a.goc_a_dau) != bool(a.goc_a):
            raise HangError('--goc-a-dau and --goc-a go together')
        if a.goc_a_dau:
            f0, s0 = _read(a.goc_a_dau)
            sau = []
            for x in a.goc_a:
                b, sep, p = x.partition('=')
                if not sep or not b:
                    raise HangError(f'--goc-a {x!r}: BUOI=path')
                f, s = _read(p)
                sau.append((b, f, s))
                src.append(dict(vai=f'goc_a {b}', duong_dan=os.path.abspath(p), sha256=s))
            src.append(dict(vai='goc_a_dau', duong_dan=os.path.abspath(a.goc_a_dau), sha256=s0))
            r, info['dau_cuoi_lan_lap'] = rows_goc_a(f0, s0, sau, khoa, a6)
            rows += r
        if not rows:
            raise HangError('no source given')
        with open(a.out, 'x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(kvt.COLS))
            w.writeheader()
            w.writerows(rows)
        meta = dict(version=VERSION, file=os.path.basename(a.out), sha256=cu.sha256(a.out), n_hang=len(rows),
                    nguon=src, thong_tin=info, liam_dau_cuoi_lan_lap=LIAM_DAU_CUOI if a.goc_a_dau else None,
                    nguon_co=sorted({r['nguon'] for r in rows}),
                    nguon_chua=[n for n in kvt.NGUON_CAN if n not in {r['nguon'] for r in rows}],
                    quy_tac='moi file nguon phai la nguon cua khoa (cung sha256); tap kiem (dung sau khoa)')
        with open(a.out + '.nguon.json', 'x', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    except (HangError, cu.InputError, OSError, ValueError) as exc:
        print(f'c8b_d3_hang: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print(f'{len(rows)} hang; nguon co {meta["nguon_co"]}; chua {meta["nguon_chua"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
