#!/usr/bin/env python3
"""
c8b_a6_o.py: the cells of the camera-IMU heading check (design C8B v7 section 2b "giu ngoai tuyet doi" and section 7c
"O D2"), file c8b_a6_o_v1, read by c8b_a6.kiem_o (A6 held-out check) and by the lock / scorer (D2 cells).
Written by CODE_C8B local_f64f609e on feat/c8b-3b-ga, 20/11/2026 (task list of CODE_C8B: "c8b_a6_o_v1 (o D2)").

Input: one CSV row per (run, frame id, clock node) from upstream (CAM camera heading after A6, c8_u_imu IMU heading,
the same pairing (k, W) of design 7b; this tool does not compute them):
  luot, tap (khop | kiem | dich), id, nut, x_m, y_m, psi_rad, u_mps, lan_lap_id, d_rad, U_psi_rad, U_imu_rad
  d_rad = (camera heading difference - IMU heading difference) of frame k against its reference W, signed, measured
  minus reference; empty when that node cannot compare the frame (no camera, no valid RV, outside the clock domain).
  U_psi_rad, U_imu_rad = U_psi(k), U_IMU,D(k) before the margin (design 7b).
Cell (design 7c [DE XUAT], locked before by the --khoa-truoc file): zone O_M x O_M of the floor x heading arc
CUNG_DEG x band of u (--dai-u edges) x mounting (lan_lap_id). Per frame id (design 7c, V5-05): comparable only when
EVERY node compares it; its worst node is the largest |d| / (U_psi + U_imu). Per cell: n_tho (ids seen), n_id
(comparable ids), n_thieu (ids not comparable), ty_le_max and the worst id with its d, U_psi, U_imu (so the reader
recomputes |d| <= U_psi + U_imu on the very numbers), signed d per controller (median, p95, max |d|). Cell dat:
n_id >= N_ID_MIN, n_thieu = 0, ty_le_max <= 1. File, A6 held-out check (giu_ngoai_a6): trang_thai dat (every cell of
the checked set dat) | truot (a cell with ty_le_max > 1) | thieu (otherwise). File, D2 (d2; design 7c line 166, CODE
24/11 khuya): a cell not dat masks only its frames: trang_thai dat once cells exist, with mat_na {o_dat, o_thieu
[{o_id, ly_do: vuot | thieu_mau | thieu_id}], ngoai_dai_u [[run, id]], khung_mo [[run, id]] (the frames whose cell
is in o_dat), lan_cham {run: cham_id}}: c8b_luoi --mat-na-d2 makes every other frame of a target run n_undef for
G2-G5 (G1 per its rule). lan_cham (optional column cham_id of the rows, c8b_d2_hang; Codex round 2 D2v2-4, CODE
25/11): the id of the scoring the rows of a target run come from; a run with rows of two scorings is refused; the mask
is applied only on the node files of that id; a d2 file is refused when a target run has rows without the id (Codex
round 3 V3-6: the cells are shared by the runs).
The three sets (khop, kiem, dich) must be disjoint by run.

  python tools/c8b_a6_o.py --rows d2_rows.csv --muc-dich giu_ngoai_a6 --giu-ngoai-theo cho --khoa-truoc khoa.json \
      --cho-file a6_goc.json [--cho-file a6_K1.json] [--dai-u 0,0.3,0.6,1.0,1.5] --out c8b_a6_o.json
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402
import c8b_a6 as a6m                                                     # noqa: E402

SCHEMA = a6m.SCHEMA_O
O_M = 0.5                                   # [DE XUAT] design 7c: zone side
CUNG_DEG = 45.0                             # [DE XUAT] design 7c: heading arc (30 deg at F08 is about 12 frames)
DAI_U = (0.0, 0.3, 0.6, 1.0, 1.5)           # [DE XUAT] band edges of u [m/s]; u outside -> the frame not in a cell
N_ID_MIN = a6m.O_N_ID_MIN
TAP = ('khop', 'kiem', 'dich')
MUC_DICH = {'giu_ngoai_a6': 'kiem', 'd2': 'dich'}       # which set the cells are built on
COLS = ('luot', 'tap', 'id', 'nut', 'x_m', 'y_m', 'psi_rad', 'u_mps', 'lan_lap_id', 'd_rad', 'U_psi_rad', 'U_imu_rad')


class OError(ValueError):
    pass


def _f(v, what, empty_ok=False):
    if v in ('', None):
        if empty_ok:
            return None
        raise OError(f'{what}: empty')
    try:
        x = float(v)
    except ValueError:
        raise OError(f'{what}: {v!r} not a number')
    if not math.isfinite(x):
        raise OError(f'{what}: {v!r} not finite')
    return x


def read_rows(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in COLS):
        raise OError(f'{path}: columns {COLS} needed')
    out, seen = [], set()
    for n, r in enumerate(rows, 2):
        w = f'{path}:{n}'
        if r['tap'] not in TAP:
            raise OError(f'{w}: tap {r["tap"]!r}')
        key = (r['luot'], r['id'], r['nut'])
        if key in seen:
            raise OError(f'{w}: run / id / node repeated')
        seen.add(key)
        if not r['luot'] or not r['lan_lap_id']:
            raise OError(f'{w}: luot and lan_lap_id needed')
        d = _f(r['d_rad'], f'{w} d_rad', empty_ok=True)
        up = _f(r['U_psi_rad'], f'{w} U_psi_rad', empty_ok=True)
        ui = _f(r['U_imu_rad'], f'{w} U_imu_rad', empty_ok=True)
        if (up is not None and up < 0) or (ui is not None and ui < 0):
            raise OError(f'{w}: negative bound')
        # mien_hop (optional column, c8b_d2_hang; Codex D2-8): 1 = the frame is in the deciding (union) domain of
        # that node; without the column every row counts as deciding (the A6 held-out rows)
        mh = r.get('mien_hop')
        if mh not in (None, '', '0', '1'):
            raise OError(f'{w}: mien_hop {mh!r}')
        out.append(dict(luot=r['luot'], tap=r['tap'], id=int(r['id']), nut=r['nut'], x=_f(r['x_m'], f'{w} x_m'),
                        y=_f(r['y_m'], f'{w} y_m'), psi=_f(r['psi_rad'], f'{w} psi_rad'),
                        u=_f(r['u_mps'], f'{w} u_mps'), lan=r['lan_lap_id'], d=d, up=up, ui=ui,
                        ctrl=r.get('controller') or '', mh=mh != '0', cham=r.get('cham_id') or ''))
    return out


def _so_duoc(x):
    return x['d'] is not None and x['up'] is not None and x['ui'] is not None and x['up'] + x['ui'] > 0


def cell_of(x, y, psi, u, lan, dai_u=DAI_U):
    """The cell key of a frame, or None when u is outside the bands (the consumer maps frames the same way)."""
    b = None
    for j in range(len(dai_u) - 1):
        if dai_u[j] <= u < dai_u[j + 1] or (j == len(dai_u) - 2 and u == dai_u[-1]):
            b = j
            break
    if b is None:
        return None
    arc = int(math.floor((psi % (2 * math.pi)) / math.radians(CUNG_DEG))) % int(round(360 / CUNG_DEG))
    return (int(math.floor(x / O_M)), int(math.floor(y / O_M)), arc, b, lan)


def build(rows, muc_dich, giu_ngoai_theo, khoa_sha, cho_sha, dai_u=DAI_U):
    if muc_dich not in MUC_DICH:
        raise OError(f'muc_dich {muc_dich!r}')
    if giu_ngoai_theo not in ('cho', 'chieu', 'lan_lap', 'khoi_thoi_gian'):
        raise OError(f'giu_ngoai_theo {giu_ngoai_theo!r}')
    if not khoa_sha:
        raise OError('khoa_truoc needed: the cells and sets are locked before the check (design 0.3)')
    tap = {t: sorted({r['luot'] for r in rows if r['tap'] == t}) for t in TAP}
    runs_tap = {}
    for r in rows:
        if runs_tap.setdefault(r['luot'], r['tap']) != r['tap']:
            raise OError(f'run {r["luot"]} in two sets')
    use = MUC_DICH[muc_dich]
    # per (run, id): every node must compare it
    per = {}
    for r in rows:
        if r['tap'] != use:
            continue
        per.setdefault((r['luot'], r['id']), []).append(r)
    cells, outside, ngoai_mien = {}, 0, 0
    for (luot, fid), rs in list(per.items()):
        c0 = rs[0]
        if any((x['x'], x['y'], x['psi'], x['u'], x['lan']) != (c0['x'], c0['y'], c0['psi'], c0['u'], c0['lan'])
               for x in rs):
            raise OError(f'run {luot} id {fid}: pose / u / mounting differ between nodes (one frame, one pose)')
        # D2 (design 0.4 "tren toan mien quyet", 7c "khung co the quyet"; Codex D2-8): the cells are built on the
        # frames that can decide at some node; such a frame is comparable only when EVERY node where it can decide
        # compares it (a node where it cannot decide need not); its worst node is over every node that compares it.
        # A frame that can decide nowhere is in no cell and not in the mask (masking it changes nothing).
        need = [x for x in rs if x['mh']] if muc_dich == 'd2' else rs
        if not need:
            ngoai_mien += 1
            del per[(luot, fid)]
            continue
        key = cell_of(c0['x'], c0['y'], c0['psi'], c0['u'], c0['lan'], dai_u)
        if key is None:
            outside += 1
            continue
        c = cells.setdefault(key, dict(ids=[], thieu=[], ctrl={}))
        if not all(_so_duoc(x) for x in need):
            c['thieu'].append([luot, fid])
            continue
        co = [x for x in rs if _so_duoc(x)]
        worst = max(co, key=lambda x: abs(x['d']) / (x['up'] + x['ui']))
        ratio = abs(worst['d']) / (worst['up'] + worst['ui'])
        c['ids'].append(dict(luot=luot, id=fid, ratio=ratio, d=worst['d'], up=worst['up'], ui=worst['ui'],
                             nut=worst['nut']))
        c['ctrl'].setdefault(c0['ctrl'], []).append(max((x['d'] for x in co), key=abs))
    o = []
    for key, c in sorted(cells.items(), key=lambda kv: str(kv[0])):
        w = max(c['ids'], key=lambda z: z['ratio']) if c['ids'] else None
        n_id = len(c['ids'])
        dat = bool(n_id >= N_ID_MIN and not c['thieu'] and w is not None and w['ratio'] <= 1.0)
        bao = {k: dict(trung_vi=float(np.median(v)), p95_abs=float(np.percentile(np.abs(v), 95)),
                       max_abs=float(np.max(np.abs(v))), n=len(v)) for k, v in c['ctrl'].items()}
        o.append(dict(o_id='x{}_y{}_h{}_u{}_{}'.format(*key), vung=[key[0], key[1]], cung=key[2], dai_u=key[3],
                      lan_lap_id=key[4], tap=use, n_tho=n_id + len(c['thieu']), n_id=n_id, n_thieu=len(c['thieu']),
                      thieu_ids=c['thieu'], ty_le_max=None if w is None else w['ratio'],
                      id_xau=None if w is None else [w['luot'], w['id'], w['nut']],
                      d_max_rad=None if w is None else abs(w['d']), d_co_dau_rad=None if w is None else w['d'],
                      U_psi_rad=None if w is None else w['up'], U_imu_rad=None if w is None else w['ui'],
                      dat=dat, bao_cao_controller=bao))
    mat_na = None
    if muc_dich == 'd2':
        # D2 (design v7 7c line 166; CODE 24/11 khuya: the code had the strict rule of the A6 held-out check): a cell
        # over the bound (one id) or short of samples (n_id < N_ID_MIN, n_thieu > 0) masks ITS frames only (n_undef for
        # G4, G5 and every position through q); a frame outside the u bands is in no cell and is masked too. The file
        # is dat once cells exist: the mask is the result, read by the scorer (o_dat: frames whose cell is not listed
        # there are n_undef).
        def ly_do(c):
            w_ = []
            if c['ty_le_max'] is not None and c['ty_le_max'] > 1.0:
                w_.append('vuot')
            if c['n_id'] < N_ID_MIN:
                w_.append('thieu_mau')
            if c['n_thieu']:
                w_.append('thieu_id')
            return w_ or ['khong_so_duoc']
        ngoai = sorted([luot, fid] for (luot, fid), rs in per.items()
                       if cell_of(rs[0]['x'], rs[0]['y'], rs[0]['psi'], rs[0]['u'], rs[0]['lan'], dai_u) is None)
        o_dat = {c['o_id'] for c in o if c['dat']}
        # khung_mo (C8B 24/11 khuya, for c8b_luoi --mat-na-d2): the frames (run, id) of the target set whose cell is
        # in o_dat, listed so the consumer never maps frames to cells again; a frame of the run not listed is masked
        mo = []
        for (luot, fid), rs in per.items():
            k = cell_of(rs[0]['x'], rs[0]['y'], rs[0]['psi'], rs[0]['u'], rs[0]['lan'], dai_u)
            if k is not None and 'x{}_y{}_h{}_u{}_{}'.format(*k) in o_dat:
                mo.append([luot, fid])
        mo.sort()
        # lan_cham (Codex round 2 D2v2-4; CODE 25/11): the scoring id of the rows of each target run; the mask goes
        # by (run, frame id), so it must not be applied on the node files of another scoring of the same run
        cham = {}
        for r in rows:
            if r['tap'] == use:
                cham.setdefault(r['luot'], set()).add(r.get('cham') or '')
        hai = sorted(k for k, v in cham.items() if len(v) > 1)
        if hai:
            raise OError(f'runs {hai[:5]}: rows of more than one scoring (cham_id), or some rows without it')
        # Codex round 3 V3-6: the cells are shared by the target runs, so the rows of a run without an id (its mask
        # would be refused) must not fill a cell that opens the frames of another run: every target run has an id
        khong = sorted(k for k, v in cham.items() if v == {''})
        if khong:
            raise OError(f'runs {khong[:5]}: D2 rows without cham_id (c8b_d2_hang before the scoring id): every '
                         'target run of a d2 cell file carries the id of its scoring')
        lan_cham = {k: next(iter(v)) for k, v in sorted(cham.items())}
        mat_na = dict(o_dat=[c['o_id'] for c in o if c['dat']], khung_mo=mo, lan_cham=lan_cham,
                      o_thieu=[dict(o_id=c['o_id'], ly_do=ly_do(c)) for c in o if not c['dat']],
                      ngoai_dai_u=ngoai, n_ngoai_mien_quyet=ngoai_mien,
                      quy_tac='khung co o khong nam trong o_dat (o vuot, thieu mau, thieu id) hay ngoai dai u -> '
                              'n_undef cho G4, G5 va moi dai luong vi tri qua q (thiet ke C8b v7 7c dong 166); o cua '
                              'khung theo cell_of (x, y, psi, u, lan_lap_id) va khoa_o; o dung tren khung nam trong '
                              'mien hop o it nhat mot nut (mien_hop), ID so duoc khi moi nut ma khung nam trong mien '
                              'hop deu so duoc (D2-8)')
        st = 'dat' if o else 'thieu'
    elif not o:
        st = 'thieu'
    elif any(c['ty_le_max'] is not None and c['ty_le_max'] > 1.0 for c in o):
        st = 'truot'
    elif all(c['dat'] for c in o):
        st = 'dat'
    else:
        st = 'thieu'
    for a, b in (('khop', 'kiem'), ('khop', 'dich'), ('kiem', 'dich')):
        if set(tap[a]) & set(tap[b]):
            raise OError(f'sets {a} and {b} share runs')
    return a6m.with_sha(dict(
        schema=SCHEMA, version=SCHEMA, muc_dich=muc_dich, giu_ngoai_theo=giu_ngoai_theo, truoc_le=True, khoa_truoc_sha256=khoa_sha,
        a6_khop=dict(cho_sha256=list(cho_sha)), tap=tap,
        khoa_o=dict(O_M=O_M, CUNG_DEG=CUNG_DEG, DAI_U=list(dai_u), N_ID_MIN=N_ID_MIN, lan_lap='lan_lap_id',
                    quy_tac='o dat: n_id >= N_ID_MIN, n_thieu 0, moi id |d| <= U_psi + U_IMU,D o moi nut'),
        n_ngoai_dai_u=outside, o=o, trang_thai=st, **({} if mat_na is None else dict(mat_na=mat_na))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--muc-dich', required=True, choices=sorted(MUC_DICH))
    ap.add_argument('--giu-ngoai-theo', required=True, choices=('cho', 'chieu', 'lan_lap', 'khoi_thoi_gian'))
    ap.add_argument('--khoa-truoc', required=True)
    ap.add_argument('--cho-file', action='append', default=[])
    ap.add_argument('--dai-u', default=None, help='band edges of u [m/s], comma separated, increasing')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        dai = DAI_U if a.dai_u is None else tuple(float(x) for x in a.dai_u.split(','))
        if len(dai) < 2 or any(b <= x for x, b in zip(dai, dai[1:])):
            raise OError(f'--dai-u {a.dai_u!r}: increasing edges')
        if a.muc_dich == 'd2' and tuple(dai) != DAI_U:
            # Codex D2-15 (D1: the cell rule locked before the data is seen): the D2 cells are those of this file at
            # the scoring commit (O_M, CUNG_DEG, DAI_U, N_ID_MIN in khoa_o), never chosen on the command line
            raise OError(f'--dai-u {a.dai_u!r} with d2: the D2 bands are {DAI_U} of the code at the scoring commit')
        cu.check_path(a.khoa_truoc)
        rows = read_rows(a.rows)
        out = build(rows, a.muc_dich, a.giu_ngoai_theo, cu.sha256(a.khoa_truoc), [cu.sha256(p) for p in a.cho_file],
                    dai)
        out['nguon'] = [dict(vai='rows', file=os.path.abspath(a.rows), sha256=cu.sha256(a.rows)),
                        dict(vai='khoa_truoc', file=os.path.abspath(a.khoa_truoc), sha256=cu.sha256(a.khoa_truoc))] \
            + [dict(vai='cho', file=os.path.abspath(p), sha256=cu.sha256(p)) for p in a.cho_file]
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_a6_o.py', 'c8b_a6.py'],
                           capture_output=True, text=True)
        out['script'] = dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()),
                             luc_chay=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
        a6m.with_sha(out)
        cu.check_path(a.out)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False))
        print(f'{a.muc_dich}: {len(out["o"])} cells, trang_thai {out["trang_thai"]}')
    except (OError, a6m.A6Error, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_a6_o: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
