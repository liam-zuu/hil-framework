#!/usr/bin/env python3
"""
c8b_he_so_giu_ngoai.py: the leave-one-session-out ratios of the terms the C8b lock takes per session (P1, A3 of the
KIEMCHEO kiem_cheo.json; P_neo, A_neo of neo_luot) and the factors c8b_khoa_v5 --he-so-giu-ngoai applies.
CODE_C8B local_a56ba287, 24/11/2026 khuya (CODE local_577c658d: "them tham so he so, mac dinh 1"; the rule itself is
Liam's decision, pending: without the file the lock does not change).

Why (real data 24/11, side-session-notes 2026-11-24_C8B_D3_bang_phuong_an_cho_Liam.md): comparing the largest error
of a session with the largest error of the other sessions always fails the session holding the overall largest. Rule
proposed: ratio(term) = largest over the sessions s and the domains of (largest |error| of s) / (index of the OTHER
sessions, before the margin) (the rows of c8b_d3_hang.rows_p1a3 / rows_neo); ratio <= LE (the lock's margin 1.25):
the margin covers the transfer between sessions, factor 1; ratio > LE: the term is multiplied by ratio / LE in the
lock (U grows; no threshold is loosened).
Terms scaled by the lock: P1_m, P_neo_m, A_neo_rad. A3 also enters u_beta_cam, which the lock does not scale: an A3
ratio above LE gives trang_thai thieu (the lock refuses the file) [GIOI HAN].

  python tools/c8b_he_so_giu_ngoai.py --kiem-cheo BUOI/doanN=kiem_cheo.json [...] \
      --neo BUOI=neo_luot.json,neo_luot_khung.csv [...] --out he_so.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402
import c8b_a6 as a6m                                                     # noqa: E402
import c8b_d3_hang as d3                                                 # noqa: E402
import c8b_khoa_v5 as khoa                                               # noqa: E402

SCHEMA = khoa.HE_SO_SCHEMA
TEN = {'P1': 'P1_m', 'A3': 'A3_rad', 'P_neo': 'P_neo_m', 'A_neo': 'A_neo_rad'}


class HeSoError(ValueError):
    pass


def ti_le(rows):
    """{dai_luong: {ti_le, mien, sai_m, chi_so_m}}: the row with the largest |sai| / chi_so per quantity."""
    out = {}
    for r in rows:
        s, c = abs(float(r['sai_m'])), float(r['chi_so_m'])
        if c <= 0:
            if s > 0:
                raise HeSoError(f'{r["dai_luong"]} {r["mien"]}: the index of the other sessions is 0, no ratio')
            t = 0.0
        else:
            t = s / c
        o = out.get(r['dai_luong'])
        if o is None or t > o['ti_le']:
            out[r['dai_luong']] = dict(ti_le=t, mien=r['mien'], sai=s, chi_so=c)
    return out


def build(segs, neo, le=khoa.LE):
    """segs, neo as c8b_d3_hang.rows_p1a3 / rows_neo take them (rho 1: a ratio has no lever arm)."""
    tl = ti_le(d3.rows_p1a3(segs, 1.0))
    rn, thieu = d3.rows_neo(neo, 1.0)
    tl.update(ti_le(rn))
    miss = [k for k in TEN if k not in tl]
    if miss:
        raise HeSoError(f'no ratio for {miss}')
    hang = {TEN[k]: max(1.0, tl[k]['ti_le'] / le) for k in ('P1', 'P_neo', 'A_neo')}
    why = []
    if tl['A3']['ti_le'] > le:
        why.append(f'A3 ratio {tl["A3"]["ti_le"]:.4f} above the margin {le}: A3 is not scaled by the lock (it also '
                   'enters u_beta_cam)')
    return dict(schema=SCHEMA, le=le, ti_le={TEN[k]: v for k, v in tl.items()}, hang=hang,
                trang_thai='dat' if not why else 'thieu', ly_do=why, n_khung_neo_khong_so=len(thieu),
                quy_tac='ti le = max qua buoi s va mien cua (|sai| lon nhat cua s) / (chi so cua cac buoi KHAC, truoc '
                        'le); ti le <= le: he so 1; ti le > le: he so = ti le / le (U lon len, khong noi nguong)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kiem-cheo', action='append', required=True, help='BUOI/doanN=kiem_cheo.json, repeat')
    ap.add_argument('--neo', action='append', required=True, help='BUOI=neo_luot.json,khung.csv, repeat')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        segs, neo, src = [], [], []
        for x in a.kiem_cheo:
            seg, sep, p = x.partition('=')
            if not sep or '/' not in seg:
                raise HeSoError(f'--kiem-cheo {x!r}: BUOI/doanN=path')
            kc, s = d3._read(p)
            d = os.path.dirname(os.path.abspath(p))
            cap, cs = d3._csv(os.path.join(d, 'cap_tuong_doi.csv'))
            a3, as_ = d3._csv(os.path.join(d, 'a3_cap.csv'))
            segs.append((seg, kc, cap, a3))
            src += [dict(vai=f'kiem_cheo {seg}', duong_dan=os.path.abspath(p), sha256=s),
                    dict(vai=f'cap_tuong_doi {seg}', duong_dan=os.path.join(d, 'cap_tuong_doi.csv'), sha256=cs),
                    dict(vai=f'a3_cap {seg}', duong_dan=os.path.join(d, 'a3_cap.csv'), sha256=as_)]
        for x in a.neo:
            b, sep, ps = x.partition('=')
            pj, sep2, pc = ps.partition(',')
            if not sep or not sep2 or not b:
                raise HeSoError(f'--neo {x!r}: BUOI=neo_luot.json,khung.csv')
            n, s = d3._read(pj)
            kh, ks = d3._csv(pc)
            neo.append((b, n, kh))
            src += [dict(vai=f'neo {b}', duong_dan=os.path.abspath(pj), sha256=s),
                    dict(vai=f'neo_khung {b}', duong_dan=os.path.abspath(pc), sha256=ks)]
        out = build(segs, neo)
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True)
        st = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_he_so_giu_ngoai.py',
                             'c8b_d3_hang.py', 'c8b_khoa_v5.py'], capture_output=True, text=True)
        out.update(nguon=src, script=dict(commit=c.stdout.strip(), dirty=bool(st.stdout.strip()),
                                          luc_chay=datetime.datetime.now().astimezone().isoformat(timespec='seconds')))
        a6m.with_sha(out)
        cu.check_path(a.out)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False))
        print('; '.join(f'{k} ti le {v["ti_le"]:.4f}' for k, v in out['ti_le'].items()))
        print('he so ' + ', '.join(f'{k} {v:.4f}' for k, v in out['hang'].items()) + f'; trang_thai {out["trang_thai"]}')
    except (HeSoError, d3.HangError, khoa.KhoaError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_he_so_giu_ngoai: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
