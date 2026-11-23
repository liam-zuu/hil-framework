#!/usr/bin/env python3
"""
c8b_p6_dong.py: P6 rieng (ii) dong, the moving part of the per-end height error of the car tag (design C8B v7
section 3, 'cao theo dau' (ii)), by the rule locked BEFORE the computation in
05_DATA/results/C8/2026-11-20_run02/c8b/T12_khoa/khoa_P6_rieng_ii.json (schema c8b_P6_rieng_ii_khoa_v1; CODE
local_483c9ce2 23/11 agreed the bands). Written by CODE_C8B local_0b12a551, 23/11/2026.

Rule (items of the lock file):
  1  bands of u (m/s): B1 = [0, u_3b(F02)], B2 = (u_3b(F02), u_3b(F05)], B3 = (u_3b(F05), u_3b(F08)] (c8b_khoa_v5.u_3b,
     side a); B0 = the standing frames u < v_d (nguong_khoa_c8_v1 v_d_m_s).
  2  u of a frame = speed + |wz| a / sqrt 2 (camera columns), valid == 1, scale / speed / wz finite.
  3  runs: the tham_chieu_v1 of the C8 runs at goc; frames from their 'frames' source (SHA checked); s_W = mean scale
     over the W ids of that run.
  4  dh = (H - h)(1 - s_W / s), H = tag_h_m / kappa, h = tag_h_m of the scoring session.
  5  leave one run out: D(X, B) = max |dh| over the frames of the runs other than X with u in B, at least SLOW_MIN
     (c8b_3b) frames; D(B) = max over X (the common maximum); none -> thieu.
  6  blur of 3b: N(B) = (H_3b - h_3b) |median over the 3b B frames in B with u >= v_d of (s / s_ref) - 1|, s_ref = median scale of the
     3b B frames with u < v_d, at least SLOW_MIN frames each; N(B0) = 0.
  7  delta_h_dong(B) = D(B) + N(B) (added, never subtracted).
  8, 9, 10: how c8b_khoa_v5 uses it (rho_max delta_h_dong / (H - h_max) into P4_m(B) of every level holding B, B0
     into hang_m); no acceleration / half bands (CODE: the band maximum contains them, the safe side).

  python tools/c8b_p6_dong.py --khoa-quy-tac khoa_P6_rieng_ii.json --tham-chieu tc1.json [--tham-chieu ...] \
      --session H2a/calib33_v2/session.json --b3b-frames B_ghep/frames.csv --b3b-session calib_tagB/session.json \
      --nguong nguong_khoa_c8_v1.json --canh-ma-m 0.1445 --out c8b_P6_rieng_ii_v1.json
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
import c8b_3b as tb3                                                     # noqa: E402

VERSION = 'c8b_P6_rieng_ii_v1'
KHOA_SCHEMA = 'c8b_P6_rieng_ii_khoa_v1'
LEVELS = ('O1-F02', 'O1-F05', 'O1-F08')
N_MIN = tb3.SLOW_MIN                     # lock item 5, 6: the existing c8b_3b constant (30 frames)


class P6Error(Exception):
    pass


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def bands_of(a_m):
    """[(name, lo, hi, lo_open)] of lock item 1."""
    import c8b_khoa_v5 as K
    u = [K.u_3b(lv, a_m) for lv in LEVELS]
    return [('B1', 0.0, u[0], False), ('B2', u[0], u[1], True), ('B3', u[1], u[2], True)]


def in_band(u, b):
    _, lo, hi, op = b
    return (u > lo if op else u >= lo) and u <= hi


def read_frames(path, sha, a_m):
    """{idx: (u, scale)} of the valid frames with finite scale, speed, wz (lock item 2); SHA must be sha."""
    cu.check_path(path)
    if sha is not None and cu.sha256(path) != sha:
        raise P6Error(f'{path}: SHA-256 is not {sha}')
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('valid') != '1':
                continue
            try:
                i = int(row['idx'])
            except (KeyError, TypeError, ValueError):
                continue
            s, v, w = _num(row.get('scale')), _num(row.get('speed')), _num(row.get('wz'))
            if s is None or v is None or w is None or s <= 0:
                continue
            out[i] = (v + abs(w) * a_m / math.sqrt(2.0), s)
    return out


def session_Hh(ses):
    h, k = _num(ses.get('tag_h_m')), _num(ses.get('kappa'))
    if h is None or k is None or not (0 < h and 0 < k):
        raise P6Error('session without tag_h_m / kappa')
    return h / k, h


def run_dh(tc, a_m, H, h):
    """[(u, |dh|)] of one run (lock items 3, 4) from its tham_chieu_v1."""
    fr = [n for n in tc.get('nguon') or [] if n.get('vai_tro') == 'frames']
    if tc.get('schema') != 'tham_chieu_v1' or len(fr) != 1 or fr[0].get('sha256') != tc.get('frames_sha256') \
            or not isinstance(tc.get('ids'), list) or not tc['ids']:
        raise P6Error('tham_chieu_v1 without one frames source of its frames_sha256 or without ids')
    F = read_frames(fr[0]['duong_dan'], tc['frames_sha256'], a_m)
    miss = [i for i in tc['ids'] if int(i) not in F]
    if miss:
        raise P6Error(f'{fr[0]["duong_dan"]}: {len(miss)} W ids without a valid scale / speed / wz')
    sW = sum(F[int(i)][1] for i in tc['ids']) / len(tc['ids'])
    return [(u, abs((H - h) * (1.0 - sW / s))) for u, s in F.values()], sW, fr[0]['duong_dan']


def leave_one_out(runs, sel):
    """lock item 5: (D or None, {run: D(X) or None}, frames of the pool) for the frames picked by sel(u)."""
    per = {x: [d for u, d in rows if sel(u)] for x, rows in runs.items()}
    dx = {}
    for x in runs:
        pool = [d for y, ds in per.items() if y != x for d in ds]
        dx[x] = max(pool) if len(pool) >= N_MIN else None
    ok = [v for v in dx.values() if v is not None]
    return (max(ok) if ok else None), dx, sum(len(v) for v in per.values())


def blur_3b(F3, H3, h3, v_d, b):
    """lock item 6: (N or None, n frames in the band, n reference frames)."""
    ref = sorted(s for u, s in F3.values() if u < v_d)
    inb = sorted(s for u, s in F3.values() if in_band(u, b) and u >= v_d)   # lock item 6 (sua truoc khi tinh)
    if len(ref) < N_MIN or len(inb) < N_MIN:
        return None, len(inb), len(ref)
    med = lambda xs: (xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]))
    return (H3 - h3) * abs(med(inb) / med(ref) - 1.0), len(inb), len(ref)


def build(tcs, ses, F3, ses3, v_d, a_m):
    H, h = session_Hh(ses)
    H3, h3 = session_Hh(ses3)
    runs, srcs = {}, []
    for name, tc in tcs:
        rows, sW, path = run_dh(tc, a_m, H, h)
        runs[name] = rows
        srcs.append(dict(luot=name, frames=path, frames_sha256=tc['frames_sha256'], s_W=sW, n_khung=len(rows)))
    if len(runs) < 2:
        raise P6Error('fewer than two runs: leave one out impossible (lock item 5)')
    out, why = {}, []
    D0, dx0, n0 = leave_one_out(runs, lambda u: u < v_d)
    out['B0'] = dict(u_lo_m_s=0.0, u_hi_m_s=v_d, D_m=D0, N_m=0.0, delta_h_m=None if D0 is None else D0,
                     D_theo_luot=dx0, n_khung=n0, trang_thai='dat' if D0 is not None else 'thieu')
    if D0 is None:
        why.append('B0: thieu khung dung yen')
    for b in bands_of(a_m):
        D, dx, n = leave_one_out(runs, lambda u, b=b: in_band(u, b))
        N, n3, nr = blur_3b(F3, H3, h3, v_d, b)
        ok = D is not None and N is not None
        out[b[0]] = dict(u_lo_m_s=b[1], u_hi_m_s=b[2], mo_trai=b[3], D_m=D, N_m=N, delta_h_m=(D + N) if ok else None,
                         D_theo_luot=dx, n_khung=n, n_khung_3b=n3, n_ref_3b=nr, trang_thai='dat' if ok else 'thieu')
        if not ok:
            why.append(f'{b[0]}: ' + ('thieu khung luot C8' if D is None else '') + (' thieu khung 3b' if N is None else ''))
    return dict(schema=VERSION, H_m=H, h_m=h, H_3b_m=H3, h_3b_m=h3, v_d_m_s=v_d, a_m=a_m, N_min=N_MIN, dai=out,
                luot=srcs, trang_thai='dat' if not why else 'thieu', ly_do=why)


def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_p6_dong.py'], capture_output=True,
                           text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), file='tools/c8b_p6_dong.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_p6_dong.py')


def _load(p):
    cu.check_path(p)
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--khoa-quy-tac', required=True, help='the rule lock (c8b_P6_rieng_ii_khoa_v1), written first')
    ap.add_argument('--tham-chieu', action='append', required=True, help='tham_chieu_v1 of one C8 run at goc; repeat')
    ap.add_argument('--session', required=True, help='scoring session.json (tag_h_m, kappa)')
    ap.add_argument('--b3b-frames', required=True, help='3b B frames.csv of session 1')
    ap.add_argument('--b3b-session', required=True, help='session.json of the 3b B clip (tag_h_m, kappa)')
    ap.add_argument('--nguong', required=True, help='nguong_khoa_c8_v1.json (v_d_m_s)')
    ap.add_argument('--canh-ma-m', type=float, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        cu.check_path(a.out)
        if os.path.exists(a.out):
            raise P6Error(f'{a.out} exists')
        kq = _load(a.khoa_quy_tac)
        if kq.get('schema') != KHOA_SCHEMA:
            raise P6Error(f'--khoa-quy-tac: not a {KHOA_SCHEMA} file')
        ng = _load(a.nguong)
        v_d = _num(((ng.get('v_d_m_s') or {}).get('gia_tri')))
        if ng.get('format') != 'nguong_khoa_c8_v1' or v_d is None or v_d <= 0:
            raise P6Error('--nguong: not a nguong_khoa_c8_v1 with v_d_m_s.gia_tri')
        if not 0 < a.canh_ma_m < 1:
            raise P6Error('--canh-ma-m: the tag side in metres')
        tcs = []
        for p in a.tham_chieu:
            tc = _load(p)
            tcs.append((os.path.basename(os.path.dirname(os.path.abspath(p))), tc))
        if len({n for n, _ in tcs}) != len(tcs):
            raise P6Error('two tham_chieu of the same run')
        F3 = read_frames(a.b3b_frames, None, a.canh_ma_m)
        out = build(tcs, _load(a.session), F3, _load(a.b3b_session), v_d, a.canh_ma_m)
        out['nguon'] = [dict(vai=v, file=os.path.basename(p), duong_dan=os.path.abspath(p), sha256=cu.sha256(p))
                        for v, ps in (('khoa_quy_tac', [a.khoa_quy_tac]), ('tham_chieu', a.tham_chieu),
                                      ('session', [a.session]), ('b3b_frames', [a.b3b_frames]),
                                      ('b3b_session', [a.b3b_session]), ('nguong', [a.nguong])) for p in ps]
        out['script'] = script_state()
        out['luc'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False))
    except (P6Error, cu.InputError, OSError, ValueError) as exc:
        print(f'c8b_p6_dong: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    for k, d in out['dai'].items():
        print(f'{k}: delta_h {d["delta_h_m"]} m (D {d["D_m"]}, N {d["N_m"]}) {d["trang_thai"]}')
    print(f'trang_thai {out["trang_thai"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
