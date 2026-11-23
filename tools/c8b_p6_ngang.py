#!/usr/bin/env python3
"""
c8b_p6_ngang.py: P6 rieng (ii) dong by the lateral width of the car tag (rule c8b_P6_rieng_ii_khoa_v2, locked first in
05_DATA/results/C8/2026-11-20_run02/c8b/T12_khoa/khoa_P6_rieng_ii_v2.json; Liam 23/11 at CODE, verbatim: "Duyet cach
moi"; proposal 03_CODE/docs/2026-11-23_De_xuat_P6_rieng_ii_bien_dang_ngang_v1.md). A rule changed AFTER the numbers of
the scale rule (c8b_p6_dong.py, lock v1) were seen; those stay as the counter-reference. CODE_C8B local_0b12a551.

Why: the tag scale changes with motion blur (along the travel direction), with the local stretch of the image map and
with the tag tilt, none of which is a height change; a height change scales the tag in every direction. The width
ACROSS the travel direction is not moved by blur along it.

Rule (items of the lock):
  1  bands of u as lock v1: B0 (u < v_d), B1 [0, u_3b(F02)], B2, B3; u = speed + |wz| a / sqrt 2 (frames.csv).
  2  frames of a run: its c8b_a6_goc_v1 file (lifted corners, valid frames between the pre and post windows).
  3  lam_y = |midpoint of the +y edge - midpoint of the -y edge| / a (body square c8b_a6.body); pose = mean of the
     corners, heading of (midpoint +x edge - midpoint -x edge).
  4  reference of a frame = median lam_y of the frames of the OTHER runs of the same camera session within pred_m and
     yaw_nghi_deg (nguong_khoa_c8_v1 quy_uoc_khac), at least 3; else the frame is not measured.
  5  dh = (H - h) |lam_y / ref - 1|; D(B) = max over the measured frames of the band, at least SLOW_MIN of them.
  6  N(B) on jig B (fixed height): pixel corners (goc-ga: c7_measure.detect_video) joined to B_ghep/frames.csv by idx
     for the pose, u and valid; the width along the tag axis at the larger angle to the median travel direction;
     reference = median width of the jig frames with u < v_d at the same pose (item 4 thresholds, >= 3);
     N(B) = (H_3b - h_3b) |median over the measured jig frames with u >= v_d in B of (width / ref - 1)|, >= SLOW_MIN.
  7  delta_h_dong(B) = D(B) + N(B); N(B0) = 0.

  python tools/c8b_p6_ngang.py goc-ga --video clip.MOV --out clip_goc.npz
  python tools/c8b_p6_ngang.py tinh --khoa-quy-tac khoa_P6_rieng_ii_v2.json --luot goc_X.json=frames.csv [...] \
      --session session.json --nguong nguong_khoa_c8_v1.json --canh-ma-m 0.1445 \
      [--ga-goc clip1_goc.npz=0 --ga-goc clip2_goc.npz=7040 --ga-frames B_ghep/frames.csv --ga-session s.json] --out o.json
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
import c8b_3b as tb3                                                     # noqa: E402
import c8b_a6 as a6m                                                     # noqa: E402

VERSION = 'c8b_P6_rieng_ii_v2'
KHOA_SCHEMA = 'c8b_P6_rieng_ii_khoa_v2'
GOC_SCHEMA = 'c8b_a6_goc_v1'
GA_SCHEMA = 'c8b_goc_ga_v1'
LEVELS = ('O1-F02', 'O1-F05', 'O1-F08')
N_MIN = tb3.SLOW_MIN
N_REF = 3                                # lock item 4: a median of 3 stands one odd frame


class P6Error(Exception):
    pass


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def bands_of(a_m):
    import c8b_khoa_v5 as K
    u = [K.u_3b(lv, a_m) for lv in LEVELS]
    return [('B1', 0.0, u[0], False), ('B2', u[0], u[1], True), ('B3', u[1], u[2], True)]


def in_band(u, b):
    return (u > b[1] if b[3] else u >= b[1]) & (u <= b[2])


def widths(P, side):
    """P (n, 4, 2) corners in the order of c8b_a6.body: (lam_x, lam_y, centre (n, 2), heading (n))."""
    r = a6m.body(side)
    yp, ym, xp, xm = r[:, 1] > 0, r[:, 1] < 0, r[:, 0] > 0, r[:, 0] < 0
    dy = P[:, yp].mean(1) - P[:, ym].mean(1)
    dx = P[:, xp].mean(1) - P[:, xm].mean(1)
    return (np.hypot(dx[:, 0], dx[:, 1]) / side, np.hypot(dy[:, 0], dy[:, 1]) / side, P.mean(1),
            np.arctan2(dx[:, 1], dx[:, 0]))


def read_u(path, sha, a_m):
    """{idx: (u, x, y, psi)} of the valid frames of a frames.csv with finite scale, speed, wz (lock item 1)."""
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
            if s is None or v is None or w is None:
                continue
            out[i] = (v + abs(w) * a_m / math.sqrt(2.0), _num(row.get('x')), _num(row.get('y')), _num(row.get('psi')))
    return out


def read_run(goc_path, frames_path, a_m):
    """One run (lock items 2, 3): dict(luot, session, u, lam, T, psi) over the goc frames with a finite u."""
    cu.check_path(goc_path)
    with open(goc_path, encoding='utf-8') as f:
        g = json.load(f)
    if not isinstance(g, dict) or g.get('schema') != GOC_SCHEMA or not isinstance(g.get('khung'), list):
        raise P6Error(f'{goc_path}: not a {GOC_SCHEMA} file')
    U = read_u(frames_path, g.get('frames_sha256'), a_m)
    keep = [k for k in g['khung'] if int(k[0]) in U]
    if not keep:
        raise P6Error(f'{goc_path}: no frame with u')
    P = np.array([k[2] for k in keep], float).reshape(-1, 4, 2)
    if not np.all(np.isfinite(P)):
        raise P6Error(f'{goc_path}: non-finite corner')
    _, lam, T, psi = widths(P, a_m)
    return dict(luot=g.get('luot_id'), session=g.get('session_sha256'), u=np.array([U[int(k[0])][0] for k in keep]),
                lam=lam, T=T, psi=psi, goc_sha256=cu.sha256(goc_path), frames_sha256=g.get('frames_sha256'),
                n_goc=len(g['khung']))


def ref_median(T, psi, val, To, psio, valo, d_max, a_max, chunk=400):
    """Per query frame: median of valo over the other frames within d_max and a_max, NaN below N_REF of them."""
    out = np.full(len(T), np.nan)
    if not len(To):
        return out
    for s in range(0, len(T), chunk):
        e = min(len(T), s + chunk)
        d = np.hypot(T[s:e, None, 0] - To[None, :, 0], T[s:e, None, 1] - To[None, :, 1])
        da = np.abs(np.angle(np.exp(1j * (psi[s:e, None] - psio[None, :]))))
        m = (d <= d_max) & (da <= a_max)
        for i in range(e - s):
            v = valo[m[i]]
            if len(v) >= N_REF:
                out[s + i] = np.median(v)
    return out


def do_c8(runs, H, h, v_d, d_max, a_max, a_m):
    """Lock items 4, 5: per band {D_m, n_khung, n_do, D_theo_luot}."""
    rel = []
    for X in runs:
        oth = [r for r in runs if r is not X and r['session'] == X['session'] and r['session'] is not None]
        if oth:
            To, po, lo = (np.concatenate([r[k] for r in oth]) for k in ('T', 'psi', 'lam'))
            ref = ref_median(X['T'], X['psi'], X['lam'], To, po, lo, d_max, a_max)
        else:
            ref = np.full(len(X['lam']), np.nan)
        rel.append((H - h) * np.abs(X['lam'] / ref - 1.0))
    out = {}
    for nm, sel in [('B0', lambda u: u < v_d)] + [(b[0], lambda u, b=b: in_band(u, b)) for b in bands_of(a_m)]:
        n = n_ok = 0
        per, top = {}, []
        for X, dh in zip(runs, rel):
            m = sel(X['u'])
            ok = m & np.isfinite(dh)
            n += int(m.sum())
            n_ok += int(ok.sum())
            per[X['luot']] = float(dh[ok].max()) if ok.any() else None
            top.extend(dh[ok].tolist())
        out[nm] = dict(D_m=(max(top) if n_ok >= N_MIN else None), n_khung=n, n_do=n_ok, D_theo_luot=per)
    return out


def read_ga(pairs, frames_path, a_m):
    """Jig frames (lock item 6): pixel corners of the goc-ga files (idx offset per clip) joined to frames.csv."""
    U = read_u(frames_path, None, a_m)
    rows = []
    src = []
    for path, off in pairs:
        cu.check_path(path)
        z = np.load(path, allow_pickle=False)
        if str(z['schema']) != GA_SCHEMA:
            raise P6Error(f'{path}: not a {GA_SCHEMA} file')
        src.append(dict(file=os.path.basename(path), sha256=cu.sha256(path), idx_cong=off,
                        video_sha256=str(z['video_sha256'])))
        for k, px in zip(z['k'], z['px']):
            r = U.get(int(k) + off)
            if r is not None and None not in r[1:]:
                rows.append((r, np.asarray(px, float)))
    if not rows:
        raise P6Error('no jig frame joined to frames.csv')
    return rows, src


def do_ga(rows, a_m, H3, h3, v_d, d_max, a_max):
    """Lock item 6: per band N_m; the axis across the median travel direction; pose from frames.csv (x, y, psi)."""
    u = np.array([r[0][0] for r in rows])
    T = np.array([[r[0][1], r[0][2]] for r in rows])
    psi = np.array([r[0][3] for r in rows])
    lx, ly, C, hd = widths(np.array([r[1] for r in rows]), a_m)
    # travel direction of the tag centre in the image, in the tag frame of the image (pixels): from neighbours in time
    dC = np.gradient(C, axis=0)
    th = np.angle(np.exp(1j * (np.arctan2(dC[:, 1], dC[:, 0]) - hd)))
    mv = u >= v_d
    ang = np.abs(np.angle(np.exp(2j * th[mv]))) / 2.0            # angle to the tag x axis, modulo 180, in [0, 90] deg
    goc = float(np.degrees(np.median(ang))) if mv.any() else float('nan')
    lam = ly if goc <= 45.0 else lx                              # the axis at the larger angle to the travel
    slow = u < v_d
    ref = ref_median(T, psi, lam, T[slow], psi[slow], lam[slow], d_max, a_max)
    rel = lam / ref - 1.0
    out = {}
    for b in bands_of(a_m):
        m = in_band(u, b) & mv
        ok = m & np.isfinite(rel)
        out[b[0]] = dict(N_m=((H3 - h3) * abs(float(np.median(rel[ok]))) if ok.sum() >= N_MIN else None),
                         n_khung_ga=int(m.sum()), n_do_ga=int(ok.sum()))
    return out, dict(goc_huong_chay_truc_x_ma_deg=goc, truc_ngang='y' if goc <= 45.0 else 'x', n_khung=len(rows),
                     n_cham=int(slow.sum()))


def session_Hh(ses):
    h, k = _num(ses.get('tag_h_m')), _num(ses.get('kappa'))
    if h is None or k is None or not (0 < h and 0 < k):
        raise P6Error('session without tag_h_m / kappa')
    return h / k, h


def build(runs, ses, ng, a_m, ga=None, ses3=None):
    H, h = session_Hh(ses)
    v_d = _num((ng.get('v_d_m_s') or {}).get('gia_tri'))
    qk = ng.get('quy_uoc_khac') or {}
    d_max, a_deg = _num(qk.get('pred_m')), _num(qk.get('yaw_nghi_deg'))
    if ng.get('format') != 'nguong_khoa_c8_v1' or None in (v_d, d_max, a_deg) or min(v_d, d_max, a_deg) <= 0:
        raise P6Error('--nguong: nguong_khoa_c8_v1 with v_d_m_s, quy_uoc_khac.pred_m, yaw_nghi_deg')
    if len(runs) < 2 or len({r['luot'] for r in runs}) != len(runs):
        raise P6Error('at least two distinct runs (leave one out)')
    a_max = math.radians(a_deg)
    dai = do_c8(runs, H, h, v_d, d_max, a_max, a_m)
    gab, gai = ({}, None)
    if ga is not None:
        H3, h3 = session_Hh(ses3)
        gab, gai = do_ga(ga, a_m, H3, h3, v_d, d_max, a_max)
        gai.update(H_3b_m=H3, h_3b_m=h3)
    why = []
    edges = dict(B0=(0.0, v_d), **{b[0]: (b[1], b[2]) for b in bands_of(a_m)})
    for nm, d in dai.items():
        N = 0.0 if nm == 'B0' else (gab.get(nm) or {}).get('N_m')
        d.update(u_lo_m_s=edges[nm][0], u_hi_m_s=edges[nm][1], N_m=N, **{k: v for k, v in (gab.get(nm) or {}).items()
                                                                   if k != 'N_m'})
        ok = d['D_m'] is not None and N is not None
        d.update(delta_h_m=(d['D_m'] + N) if ok else None, trang_thai='dat' if ok else 'thieu')
        if not ok:
            why.append(f'{nm}: ' + ('thieu khung do duoc cua luot C8; ' if d['D_m'] is None else '')
                       + ('thieu N ga B' if N is None else ''))
    return dict(schema=VERSION, H_m=H, h_m=h, v_d_m_s=v_d, pred_m=d_max, yaw_nghi_deg=a_deg, a_m=a_m, N_min=N_MIN,
                N_ref=N_REF, dai=dai, ga_B=gai,
                luot=[dict(luot=r['luot'], session_sha256=r['session'], goc_sha256=r['goc_sha256'],
                           frames_sha256=r['frames_sha256'], n_khung=int(len(r['u'])), n_goc=r['n_goc']) for r in runs],
                trang_thai='dat' if not why else 'thieu', ly_do=why)


def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_p6_ngang.py'], capture_output=True,
                           text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), file='tools/c8b_p6_ngang.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_p6_ngang.py')


def _load(p):
    cu.check_path(p)
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def goc_ga(video, out):
    """Pixel corners of the car tag on every frame of a clip (c7_measure.detect_video, the detector of the scoring)."""
    import c7_measure as cm
    cu.check_path(out)
    if os.path.exists(out):
        raise P6Error(f'{out} exists')
    fr, shapes = cm.detect_video(video)
    keep = [(k, t, np.asarray(d[cm.CAR_ID], float)) for k, t, d in fr if cm.CAR_ID in d]
    if not keep:
        raise P6Error(f'{video}: no car tag')
    np.savez(out, schema=GA_SCHEMA, video=os.path.basename(video), video_sha256=cu.sha256(video),
             n_frames=len(fr), k=np.array([k for k, _, _ in keep]), t=np.array([t for _, t, _ in keep]),
             px=np.array([p for _, _, p in keep]))
    return len(fr), len(keep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('goc-ga')
    g.add_argument('--video', required=True)
    g.add_argument('--out', required=True)
    t = sub.add_parser('tinh')
    t.add_argument('--khoa-quy-tac', required=True)
    t.add_argument('--luot', action='append', required=True, help='goc_X.json=frames.csv of one run at goc; repeat')
    t.add_argument('--session', required=True)
    t.add_argument('--nguong', required=True)
    t.add_argument('--canh-ma-m', type=float, required=True)
    t.add_argument('--ga-goc', action='append', default=None, help='goc-ga npz=idx offset in --ga-frames; repeat')
    t.add_argument('--ga-frames', default=None)
    t.add_argument('--ga-session', default=None)
    t.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'goc-ga':
            n, m = goc_ga(a.video, a.out)
            print(f'{a.video}: {n} frames, car tag on {m}')
            print('#ket_qua,ma=0')
            return 0
        cu.check_path(a.out)
        if os.path.exists(a.out):
            raise P6Error(f'{a.out} exists')
        if _load(a.khoa_quy_tac).get('schema') != KHOA_SCHEMA:
            raise P6Error(f'--khoa-quy-tac: not a {KHOA_SCHEMA} file')
        if not 0 < a.canh_ma_m < 1:
            raise P6Error('--canh-ma-m: the tag side in metres')
        runs, paths = [], []
        for s in a.luot:
            gp, sep, fp = s.partition('=')
            if not sep:
                raise P6Error(f'--luot {s!r}: goc.json=frames.csv')
            runs.append(read_run(gp, fp, a.canh_ma_m))
            paths += [('goc', gp), ('frames', fp)]
        ga = ses3 = None
        ga_src = []
        if a.ga_goc or a.ga_frames or a.ga_session:
            if not (a.ga_goc and a.ga_frames and a.ga_session):
                raise P6Error('--ga-goc, --ga-frames and --ga-session go together')
            pairs = []
            for s in a.ga_goc:
                p, sep, off = s.partition('=')
                if not sep or not off.lstrip('-').isdigit():
                    raise P6Error(f'--ga-goc {s!r}: file.npz=idx offset')
                pairs.append((p, int(off)))
            ga, ga_src = read_ga(pairs, a.ga_frames, a.canh_ma_m)
            ses3 = _load(a.ga_session)
            paths += [('ga_frames', a.ga_frames), ('ga_session', a.ga_session)]
        out = build(runs, _load(a.session), _load(a.nguong), a.canh_ma_m, ga, ses3)
        out['nguon'] = [dict(vai=v, file=os.path.basename(p), duong_dan=os.path.abspath(p), sha256=cu.sha256(p))
                        for v, p in [('khoa_quy_tac', a.khoa_quy_tac), ('session', a.session), ('nguong', a.nguong)]
                        + paths] + [dict(vai='ga_goc', **s) for s in ga_src]
        out['script'] = script_state()
        out['luc'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False))
    except (P6Error, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_p6_ngang: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    for k, d in out['dai'].items():
        print(f'{k}: D {d["D_m"]} N {d["N_m"]} delta_h {d["delta_h_m"]} ({d["n_do"]}/{d["n_khung"]} frames) {d["trang_thai"]}')
    print(f'trang_thai {out["trang_thai"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
