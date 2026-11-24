#!/usr/bin/env python3
"""
c8b_kiem_vi_tri.py: D3, the separate position check of the C8b U standard (design C8B v7 section 0.5 and section 8),
file c8b_kiem_vi_tri_v1 {version, trang_thai, khoa_truoc_sha256, ...}, read by GATE gate_c8._check_bang_chung
(chuan_u.dieu_kien.kiem_vi_tri_rieng: version c8b_kiem_vi_tri_v1, trang_thai dat, khoa_truoc_sha256 = the sha256 of
the C8b lock c8b_khoa_v5 of the gate) and by the lock / scorer (mien_thieu).
Written by CODE_C8B local_f64f609e on feat/c8b-3b-ga, 20/11/2026 (task list of CODE_C8B: "c8b_kiem_vi_tri_v1 (D3)").

Input: one CSV row per signed error of a position source on a set (upstream: KIEMCHEO P1/A3 held-out H32, replay of
anchors / marks on real frames, 12b, E, the height sensitivity, the mounting ends of NHACTAM, the H drift; this tool
does not compute them):
  nguon, tap (khop | kiem), dai_luong, mien, sai_m, chi_so_m, co_chuan_dong
  sai_m signed (measured - reference); chi_so_m the index of that quantity in use (before the margin; for the
  sources held out by session, L155: the other sessions' index x the lock's margin x its held-out factor, written by
  c8b_d3_hang); co_chuan_dong
  1 when the source has an independent dynamic reference, 0 when not (P4 combined motion, dynamic height).
State rule (design 8): per (dai_luong, mien): every |sai| of the test set (kiem, after the lock) <= chi_so -> dat;
one above -> that quantity thieu in that domain (listed in mien_thieu; the index is NOT raised after the lock); on the
fit set (khop, before the lock) an excess raises the index to the largest |sai| seen (listed in chi_so_nang); a source
without an independent dynamic reference -> gioi_han (usable, the assumption written). File trang_thai: dat when the
lock is a c8b_khoa_v5 file with trang_thai khoa, every required source of design 8 (NGUON_CAN) has test rows, and every
value is finite; else thieu with the reasons. A dat file may still carry mien_thieu. LIMIT (found 24/11, L155): no
tool reads mien_thieu yet (lock, scorer and gate do not): a domain listed there is NOT made n_undef in the scoring.

  python tools/c8b_kiem_vi_tri.py --rows d3_rows.csv --khoa c8b_khoa_v5.json --out c8b_kiem_vi_tri.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
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

VERSION = 'c8b_kiem_vi_tri_v1'
KHOA_VERSION = 'c8b_khoa_v5'
# design 8 (1)-(7): P1/A3 held-out H32, replay anchors / marks, 12b, E, height sensitivity, mounting ends, H drift
# 12b dropped 24/11 khuya (CODE applying C8B's recommendation, contract log C8: no signed position error of 12b; it is
# in the F7 and in the angle a of dau_cuoi_lan_lap); dau_cuoi_lan_lap = the angle a between HC_goc (Liam 24/11 at CODE)
NGUON_CAN = ('P1_A3_H32', 'replay_neo', 'E', 'do_nhay_cao', 'dau_cuoi_lan_lap', 'troi_H')   # [DE XUAT] names
NGUON_DL = {'P1_A3_H32': ('P1', 'A3'), 'replay_neo': ('P_neo', 'A_neo')}   # the quantities c8b_d3_hang writes for them
COLS = ('nguon', 'tap', 'dai_luong', 'mien', 'sai_m', 'chi_so_m', 'co_chuan_dong')


class ViTriError(ValueError):
    pass


def read_rows(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in COLS):
        raise ViTriError(f'{path}: columns {COLS} needed')
    out = []
    for n, r in enumerate(rows, 2):
        w = f'{path}:{n}'
        if r['tap'] not in ('khop', 'kiem'):
            raise ViTriError(f'{w}: tap {r["tap"]!r}')
        if not r['nguon'] or not r['dai_luong'] or not r['mien']:
            raise ViTriError(f'{w}: nguon, dai_luong, mien needed')
        try:
            sai, cs = float(r['sai_m']), float(r['chi_so_m'])
        except ValueError:
            raise ViTriError(f'{w}: sai_m / chi_so_m not numbers')
        if not (math.isfinite(sai) and math.isfinite(cs)) or cs < 0:
            raise ViTriError(f'{w}: sai_m / chi_so_m not finite or index < 0')
        if r['co_chuan_dong'] not in ('0', '1'):
            raise ViTriError(f'{w}: co_chuan_dong 0 / 1')
        out.append(dict(nguon=r['nguon'], tap=r['tap'], dl=r['dai_luong'], mien=r['mien'], sai=sai, cs=cs,
                        dong=r['co_chuan_dong'] == '1'))
    return out


def build(rows, khoa, khoa_sha):
    why = []
    if not isinstance(khoa, dict) or khoa.get('version') != KHOA_VERSION or khoa.get('trang_thai') != 'khoa':
        why.append(f'the lock is not a {KHOA_VERSION} file with trang_thai khoa (D3 is made after the lock)')
    have = {r['nguon'] for r in rows if r['tap'] == 'kiem'}
    miss = [n for n in NGUON_CAN if n not in have]
    if miss:
        why.append(f'required sources without test rows: {miss}')
    # Codex D3-3: a source held out by session has two quantities; rows of one alone do not make the source
    for ng, dls in NGUON_DL.items():
        co = {r['dl'] for r in rows if r['tap'] == 'kiem' and r['nguon'] == ng}
        if ng in have and not set(dls) <= co:
            why.append(f'source {ng}: test rows of {sorted(set(dls) - co)} missing')
    groups = {}
    for r in rows:
        groups.setdefault((r['nguon'], r['dl'], r['mien']), []).append(r)
    bang, mien_thieu, nang, gh = [], [], [], []
    for (ng, dl, mien), rs in sorted(groups.items()):
        cs = {r['cs'] for r in rs}
        if len(cs) != 1:
            raise ViTriError(f'{ng} / {dl} / {mien}: more than one index in use {sorted(cs)}')
        c = cs.pop()
        kh = [r['sai'] for r in rs if r['tap'] == 'khop']
        ki = [r['sai'] for r in rs if r['tap'] == 'kiem']
        dong = all(r['dong'] for r in rs)
        c_moi = max([c] + [abs(x) for x in kh])
        if c_moi > c:
            nang.append(dict(nguon=ng, dai_luong=dl, mien=mien, chi_so_m=c, chi_so_nang_m=c_moi))
        vuot = [x for x in ki if abs(x) > c_moi]
        if not ki:
            st = 'thieu'
        elif vuot:
            st = 'thieu'
            mien_thieu.append(dict(nguon=ng, dai_luong=dl, mien=mien, sai_max_m=max(vuot, key=abs), chi_so_m=c_moi))
        else:
            st = 'dat' if dong else 'gioi_han'
        if st == 'gioi_han':
            gh.append(dict(nguon=ng, dai_luong=dl, mien=mien, ly_do='khong co chuan dong doc lap'))
        bang.append(dict(nguon=ng, dai_luong=dl, mien=mien, n_khop=len(kh), n_kiem=len(ki),
                         sai_kiem_min_m=min(ki) if ki else None, sai_kiem_max_m=max(ki) if ki else None,
                         chi_so_m=c, chi_so_dung_m=c_moi, trang_thai=st))
    return dict(version=VERSION, trang_thai='dat' if not why else 'thieu', ly_do=why, khoa_truoc_sha256=khoa_sha,
                bang=bang, mien_thieu=mien_thieu, chi_so_nang=nang, gioi_han=gh, nguon_can=list(NGUON_CAN),
                truoc_le=True,
                quy_tac='kiem sau khoa: |sai| <= chi so -> dat, vuot -> dai luong thieu o mien do (khong nang); khop '
                        'truoc khoa: vuot -> nang chi so toi cuc dai; khong chuan dong doc lap -> gioi_han')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--khoa', required=True, help='c8b_khoa_v5 file (the C8b lock of the gate)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        cu.check_path(a.khoa)
        with open(a.khoa, encoding='utf-8') as f:
            khoa = json.load(f)
        # Codex D3-3: the rows are those c8b_d3_hang wrote for THIS lock: its file <rows>.nguon.json names the rows
        # file (sha256) and the lock (vai khoa, sha256); rows made by hand or for another lock are refused
        try:
            with open(a.rows + '.nguon.json', encoding='utf-8') as f:
                ng = json.load(f)
        except (OSError, ValueError) as exc:
            raise ViTriError(f'{a.rows}.nguon.json (written by c8b_d3_hang beside the rows) cannot be read: {exc}')
        ks = {n.get('sha256') for n in (ng.get('nguon') if isinstance(ng, dict) else None) or []
              if isinstance(n, dict) and n.get('vai') == 'khoa'}
        if not isinstance(ng, dict) or ng.get('version') != 'c8b_d3_hang_v1' or ng.get('sha256') != cu.sha256(a.rows) \
                or ks != {cu.sha256(a.khoa)}:
            raise ViTriError(f'{a.rows}: not the rows c8b_d3_hang wrote for the lock {a.khoa} (its .nguon.json: '
                             'version, sha256 of the rows, sha256 of the lock)')
        out = build(read_rows(a.rows), khoa, cu.sha256(a.khoa))
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_kiem_vi_tri.py'],
                           capture_output=True, text=True)
        out.update(nguon=[dict(vai='rows', file=os.path.abspath(a.rows), sha256=cu.sha256(a.rows)),
                          dict(vai='khoa', file=os.path.abspath(a.khoa), sha256=out['khoa_truoc_sha256'])],
                   script=dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()),
                               luc_chay=datetime.datetime.now().astimezone().isoformat(timespec='seconds')))
        a6m.with_sha(out)
        cu.check_path(a.out)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False))
        print(f'D3 {out["trang_thai"]}; mien_thieu {len(out["mien_thieu"])}; {"; ".join(out["ly_do"])}')
    except (ViTriError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_kiem_vi_tri: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
