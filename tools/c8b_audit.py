#!/usr/bin/env python3
"""
c8b_audit.py: audit of the frames left out by the camera (design C8B v7 section 11b, C8BX-02, V5-07), file
c8b_audit_v1, one per run or calibration clip. Written by CODE_C8B local_f64f609e on feat/c8b-3b-ga, 20/11/2026.

From the CAM frames.csv (every frame, rejected ones with their reason in 'why', c7_measure FRAME_COLS):
  dem        counts of frames by reason x band of u x turning direction x task x controller; u = |v| + |omega| a /
             sqrt 2 (design 10, a = tag side), from the frame's speed and wz (a frame without them: band 'khong_u').
  ho         EVERY hole: a maximal run of not-valid frames, or a time step above HO_FACTOR x the median frame step,
             with its ends [t_a, t_b] (the valid frames around it), length and reasons.
  theo_G     per condition G with a decision domain (--mien {G: [[t0, t1], ...]}, camera clock): every hole meeting
             the domain (design 11b V5-07: all of them, not only near the largest value seen); the upper bound is
             finite only when EVERY such hole is excluded by the motion index: value at the hole ends v_a, v_b
             (--gia-tri rows t_cam_s, G, hi), rate bound r of the u bands at both ends (--toc-do {G: [[u_lo, u_hi,
             rate per s]]}, held-out runs), the largest value possible inside (v_a + v_b + r dt) / 2 (Lipschitz from
             both ends) strictly below the threshold (--nguong {G: D}); else can_tren 'thieu' with the reason. The
             lower bound (max over valid frames) is the scorer's; not touched here.
Before the margin; a missing input -> 'thieu' (never 0).

  python tools/c8b_audit.py --frames frames.csv --luot C8-P-F08-1 --bai O1-F08 --controller PID --canh-m 0.1445 \
      [--mien mien.json --gia-tri hi.csv --toc-do toc_do.json --nguong nguong.json] --out c8b_audit.json
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

VERSION = 'c8b_audit_v1'
DAI_U = (0.0, 0.3, 0.6, 1.0, 1.5)            # [DE XUAT] bands of u [m/s] (same edges as c8b_a6_o)
HO_FACTOR = 1.5                              # [DE XUAT] a step above 1.5 x the median frame step is a hole
WZ_THANG = 0.05                              # [DE XUAT] |wz| below: 'thang' (no turning direction)
NEED = ('idx', 't_s', 'valid', 'why')


class AuditError(ValueError):
    pass


def _fl(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def read_frames(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in NEED):
        raise AuditError(f'{path}: columns {NEED} needed (CAM frames.csv with every frame)')
    t = np.array([_fl(r['t_s']) for r in rows])
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise AuditError(f'{path}: t_s not finite or not increasing')
    return [dict(idx=int(r['idx']), t=float(tt), valid=r['valid'] in ('1', 'True', 'true'),
                 why=(r.get('why') or '').strip() or ('' if r['valid'] in ('1', 'True', 'true') else 'khong_ro'),
                 speed=_fl(r.get('speed')), wz=_fl(r.get('wz'))) for r, tt in zip(rows, t)]


def u_of(fr, a_m):
    if not (math.isfinite(fr['speed']) and math.isfinite(fr['wz'])):
        return math.nan
    return abs(fr['speed']) + abs(fr['wz']) * a_m / math.sqrt(2.0)


def band(u, edges=DAI_U):
    if not math.isfinite(u):
        return 'khong_u'
    for j in range(len(edges) - 1):
        if edges[j] <= u < edges[j + 1]:
            return f'u{j}'
    return 'ngoai_dai'


def direction(fr):
    if not math.isfinite(fr['wz']):
        return 'khong_ro'
    return 'thang' if abs(fr['wz']) < WZ_THANG else ('ccw' if fr['wz'] > 0 else 'cw')


def holes(F):
    """Maximal runs of not-valid frames and time steps above HO_FACTOR x median: [{t_a, t_b, dt, n_khung, ly_do,
    canh_a, canh_b}] where t_a / t_b are the valid frames around (None at a clip end)."""
    t = np.array([f['t'] for f in F])
    med = float(np.median(np.diff(t))) if len(t) > 1 else math.nan
    out, k = [], 0
    n = len(F)
    while k < n:
        if not F[k]['valid']:
            j = k
            while j < n and not F[j]['valid']:
                j += 1
            a, b = (k - 1 if k > 0 else None), (j if j < n else None)
            out.append(dict(t_a=None if a is None else F[a]['t'], t_b=None if b is None else F[b]['t'],
                            canh_a=a, canh_b=b, n_khung=j - k,
                            ly_do=sorted({F[i]['why'] for i in range(k, j)}), loai='khung_bi_loai'))
            k = j
        else:
            if k + 1 < n and F[k + 1]['valid'] and F[k + 1]['t'] - F[k]['t'] > HO_FACTOR * med:
                out.append(dict(t_a=F[k]['t'], t_b=F[k + 1]['t'], canh_a=k, canh_b=k + 1, n_khung=0,
                                ly_do=['ho_thoi_gian'], loai='ho_thoi_gian'))
            k += 1
    for h in out:
        h['dt'] = None if h['t_a'] is None or h['t_b'] is None else h['t_b'] - h['t_a']
    return out


def _meets(h, dom):
    a = -math.inf if h['t_a'] is None else h['t_a']
    b = math.inf if h['t_b'] is None else h['t_b']
    return any(a < d1 and b > d0 for d0, d1 in dom)


def _rate(table, u):
    """Largest rate of the bands touching u (closed bands); None when u is outside every band or unknown."""
    if not math.isfinite(u):
        return None
    r = [x[2] for x in table if x[0] - 1e-9 <= u <= x[1] + 1e-9]
    return max(r) if r else None


def audit(F, a_m, meta, mien=None, gia_tri=None, toc_do=None, nguong=None):
    dem = {}
    for f in F:
        if f['valid']:
            continue
        key = '|'.join((f['why'], band(u_of(f, a_m)), direction(f), meta['bai'], meta['controller']))
        dem[key] = dem.get(key, 0) + 1
    H = holes(F)
    theo = {}
    for G, dom in (mien or {}).items():
        why, hs = [], []
        for h in H:
            if not _meets(h, dom):
                continue
            e = dict(h, ket_luan=None)
            if h['t_a'] is None or h['t_b'] is None:
                e['ket_luan'] = 'ho_o_dau_clip'
                why.append(f'hole at a clip end {h["t_a"]}..{h["t_b"]}')
            elif gia_tri is None or toc_do is None or nguong is None or G not in (toc_do or {}) \
                    or G not in (nguong or {}):
                e['ket_luan'] = 'thieu_chi_so'
                why.append('no motion index / values / threshold for this condition')
            else:
                va, vb = (gia_tri.get(G) or {}).get(h['t_a']), (gia_tri.get(G) or {}).get(h['t_b'])
                ra = _rate(toc_do[G], u_of(F[h['canh_a']], a_m))
                rb = _rate(toc_do[G], u_of(F[h['canh_b']], a_m))
                if va is None or vb is None or ra is None or rb is None:
                    e['ket_luan'] = 'thieu_gia_tri_canh'
                    why.append(f'hole {h["t_a"]:.3f}..{h["t_b"]:.3f}: value or rate missing at an end')
                else:
                    r = max(ra, rb)
                    vmax = (va + vb + r * h['dt']) / 2.0
                    e.update(v_a=va, v_b=vb, toc_do=r, gia_tri_kha_di_max=vmax, nguong=nguong[G])
                    if vmax < nguong[G]:
                        e['ket_luan'] = 'loai_duoc'
                    else:
                        e['ket_luan'] = 'co_the_quyet'
                        why.append(f'hole {h["t_a"]:.3f}..{h["t_b"]:.3f}: possible value {vmax:.6g} >= threshold')
            hs.append(e)
        theo[G] = dict(ho=hs, n_ho=len(hs), can_tren='huu_han' if not why else 'thieu', ly_do=why)
    return dict(version=VERSION, **meta, canh_m=a_m, dai_u=list(DAI_U), n_khung=len(F),
                n_bi_loai=sum(1 for f in F if not f['valid']), dem=dem, ho=H, theo_G=theo, truoc_le=True,
                quy_tac='can tren huu han chi khi moi ho trong mien quyet bi loai bang chi so chuyen dong: '
                        '(v_a + v_b + r dt)/2 < nguong (thiet ke v7 11b)')


def _read_gia_tri(path):
    cu.check_path(path)
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            v, t = _fl(r.get('hi')), _fl(r.get('t_cam_s'))
            if not (math.isfinite(v) and math.isfinite(t)) or not r.get('G'):
                raise AuditError(f'{path}: row {r} not finite')
            out.setdefault(r['G'], {})[t] = v
    return out


def _json(path):
    if path is None:
        return None
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--luot', required=True)
    ap.add_argument('--bai', required=True)
    ap.add_argument('--controller', required=True)
    ap.add_argument('--canh-m', type=float, required=True)
    ap.add_argument('--mien')
    ap.add_argument('--gia-tri')
    ap.add_argument('--toc-do')
    ap.add_argument('--nguong')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if not (math.isfinite(a.canh_m) and a.canh_m > 0):
            raise AuditError('--canh-m: a positive tag side')
        F = read_frames(a.frames)
        mien = _json(a.mien)
        for G, dom in (mien or {}).items():
            if not all(isinstance(d, list) and len(d) == 2 and d[0] < d[1] for d in dom):
                raise AuditError(f'--mien {G}: [[t0, t1], ...] with t0 < t1')
        out = audit(F, a.canh_m, dict(luot=a.luot, bai=a.bai, controller=a.controller), mien,
                    _read_gia_tri(a.gia_tri) if a.gia_tri else None, _json(a.toc_do), _json(a.nguong))
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_audit.py'], capture_output=True,
                           text=True)
        out.update(nguon=[dict(vai=k, file=os.path.abspath(p), sha256=cu.sha256(p))
                          for k, p in (('frames', a.frames), ('mien', a.mien), ('gia_tri', a.gia_tri),
                                       ('toc_do', a.toc_do), ('nguong', a.nguong)) if p],
                   script=dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()),
                               luc_chay=datetime.datetime.now().astimezone().isoformat(timespec='seconds')))
        cu.check_path(a.out)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False))
        print(f'{a.luot}: {out["n_bi_loai"]}/{out["n_khung"]} frames left out, {len(out["ho"])} holes; '
              + '; '.join(f'{G} {v["can_tren"]}' for G, v in out['theo_G'].items()))
    except (AuditError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_audit: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
