#!/usr/bin/env python3
"""
c8b_u_h.py: u_h_chung, the common part of the height error of the car tag, from the three 3b jigs of session 1 and
the roof tag at its caliper placement (C8B design 03_CODE/docs/2026-11-19_Thiet_ke_C8b_tuong_doi_tu_hieu_chuan_v3.md
section 3 "thu nhan dang da do cao", v7 section 3; reader: c8b_a6.kiem_u_h in c8b_a6 gop --u-h-chung). Written by
CODE_C8B local_31d56744 on fix/c8b-a6-dets, 22/11/2026 (CODE order 22/11 khuya: u_h_chung -> gop --u-h-chung -> T7b).
The rule is locked in a c8b_u_h_khoa_v1 file BEFORE the numbers are computed (root repo
05_DATA/results/C8/2026-11-20_run02/c8b/T7_u_h_chung/khoa_u_h_chung.json); this tool reads its inputs from that file
and refuses any input whose sha256 is not the locked one.

Model (car_pose of c7_measure, car_pose_v2): a corner at true height h_t lifted with kappa = h_a / H (h_a = tag_h_m of
the session, H = tag_h_m / kappa) gives
    scale = k (H - h_a) / (H - h_t),   so   h_t = H - k (H - h_a) / scale,
k = effective edge of the printed tag x edge detection bias (one plate, the same for every jig and the roof).
  1. each jig j: s_j = median scale over the slowest frames (c8b_3b.read, c8b_3b.rate, c8b_3b.slowest, unchanged);
     k_j = s_j (H_j - h_cal_j) / (H_j - h_a_j)
  2. held out: k_(-j) = mean of the other two; h_pred_j = H_j - k_(-j) (H_j - h_a_j) / s_j;
     do_trai_ba_ga_m = max_j |h_pred_j - h_cal_j|, n_ga_giu_ngoai = number of jigs held out (3)
  3. roof: s_noc = median scale over the still window of the c7_measure track of clip N0 (valid, car_seen, finite
     scale, t within [t_first + trim_s, t_last - trim_s]; trim_s, n_min from the lock = c7_nhac_tam trim_s,
     min_frames_abs); h_video = H - k_chung (H - h_a) / s_noc with k_chung = mean of the three k_j;
     lech_video_so_tay_m = |h_video - h_cal_noc|
  4. u_h_chung_m = do_trai_ba_ga_m + lech_video_so_tay_m + hang_m (0.1 mm, c8b_a6.U_H_HANG_M); h_m = the lock h_m,
     equal to tag_h_m of every --session-cham.
  Cross-checks (information, not in u_h_chung): the static clip of jig A (no blur), standard errors of the medians
  (1.2533 std / sqrt n), the spread of k.

  python tools/c8b_u_h.py --khoa KHOA.json --goc D:/CLAUDE_THESIS --session-cham S1 [S2 ...] --out u_h.json
Last line: #ket_qua,ma=0,trang_thai=dat|thieu / ma=2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402
import c8b_3b as tb                                                      # noqa: E402
import c8b_a6 as a6                                                      # noqa: E402

VERSION = 'c8b_u_h_v1'
SCHEMA_KHOA = 'c8b_u_h_khoa_v1'
SE_MED = 1.2533             # standard error of a median = 1.2533 std / sqrt(n) (normal data; as c7_nhac_tam)
TOL_H = 1e-9


class UhError(Exception):
    pass


def content_sha(d):
    return a6.content_sha(d)


def h_of(s, k, H, h_a):
    """True height from a scale measured with assumed height h_a: h_t = H - k (H - h_a) / s."""
    return H - k * (H - h_a) / s


def k_of(s, H, h_a, h_t):
    """k from a scale measured with assumed height h_a on a tag at true height h_t."""
    return s * (H - h_t) / (H - h_a)


def read_session(path, sha):
    cu.check_path(path)
    got = cu.sha256(path)
    if got != sha:
        raise UhError(f'{path}: sha256 {got} is not the locked {sha}')
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    try:
        kap, h = float(s['kappa']), float(s['tag_h_m'])
    except (KeyError, TypeError, ValueError) as exc:
        raise UhError(f'{path}: kappa / tag_h_m ({exc})')
    if not (math.isfinite(kap) and math.isfinite(h) and 0 < kap < 1 and h > 0):
        raise UhError(f'{path}: kappa {kap} / tag_h_m {h} not usable')
    return dict(file=os.path.abspath(path), sha256=got, H=h / kap, h_a=h)


def _scale_by_t(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'scale' not in rows[0]:
        raise UhError(f'{path}: no scale column')
    out = {}
    for r in rows:
        try:
            out[float(r['t_s'])] = float(r['scale']) if r['scale'] not in ('', 'nan') else math.nan
        except (TypeError, ValueError):
            continue
    return out


def jig_scale(path, sha):
    """s_j: median scale over c8b_3b.slowest of the frames of c8b_3b.read; n, standard error, |Omega| range."""
    cu.check_path(path)
    got = cu.sha256(path)
    if got != sha:
        raise UhError(f'{path}: sha256 {got} is not the locked {sha}')
    try:
        F = tb.read(path)
        om = tb.rate(F['t'], F['psi'])
        idx = tb.slowest(om)
    except tb.TbError as exc:
        raise UhError(f'{path}: {exc}')
    sc_t = _scale_by_t(path)
    sc = np.array([sc_t.get(float(F['t'][i]), math.nan) for i in idx])
    sc = sc[np.isfinite(sc) & (sc > 0)]
    if len(sc) < tb.SLOW_MIN:
        raise UhError(f'{path}: {len(sc)} slowest frames with a finite scale (< {tb.SLOW_MIN})')
    a = np.abs(om[idx])
    return dict(file=os.path.abspath(path), sha256=got, s=float(np.median(sc)), n=int(len(sc)),
                se=float(SE_MED * sc.std(ddof=1) / math.sqrt(len(sc))),
                omega_rad_s=[float(a.min()), float(a.max())])


def window_scale(frames, trim_s, n_min, need_car=True):
    """Median scale over the still window of a c7_measure track frames.csv."""
    cu.check_path(frames)
    with open(frames, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise UhError(f'{frames}: no frame')
    t = np.array([float(r['t_s']) for r in rows])
    t0, t1 = float(t.min()), float(t.max())
    sc = []
    for r, ti in zip(rows, t):
        if r.get('valid') != '1' or (need_car and r.get('car_seen') != '1'):
            continue
        if not (t0 + trim_s <= ti <= t1 - trim_s):
            continue
        try:
            v = float(r['scale'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            sc.append(v)
    sc = np.array(sc)
    if len(sc) < n_min:
        raise UhError(f'{frames}: {len(sc)} still frames with a finite scale (< {n_min})')
    return dict(file=os.path.abspath(frames), sha256=cu.sha256(frames), s=float(np.median(sc)), n=int(len(sc)),
                se=float(SE_MED * sc.std(ddof=1) / math.sqrt(len(sc))), t_cua_so_s=[t0 + trim_s, t1 - trim_s])


def tinh(ga, noc, h_m, hang_m=a6.U_H_HANG_M):
    """The numbers of the lock rule 2-7. ga: [{ten, s, H, h_a, h_cal}] (>= 3), noc: {s, H, h_a, h_cal}."""
    if len(ga) < 3:
        raise UhError(f'{len(ga)} jigs (< 3, design 3)')
    ks = np.array([k_of(g['s'], g['H'], g['h_a'], g['h_cal']) for g in ga])
    ho = []
    for j, g in enumerate(ga):
        k_o = float(np.mean(np.delete(ks, j)))
        hp = h_of(g['s'], k_o, g['H'], g['h_a'])
        ho.append(dict(ten=g['ten'], k=float(ks[j]), k_giu_ngoai=k_o, h_du_bao_m=float(hp),
                       h_cal_m=g['h_cal'], sai_m=float(hp - g['h_cal'])))
    do_trai = float(max(abs(x['sai_m']) for x in ho))
    k_chung = float(np.mean(ks))
    h_video = float(h_of(noc['s'], k_chung, noc['H'], noc['h_a']))
    lech = float(abs(h_video - noc['h_cal']))
    phan = dict(do_trai_ba_ga_m=do_trai, lech_video_so_tay_m=lech, hang_m=float(hang_m))
    return dict(phan=phan, u_h_chung_m=float(sum(phan.values())), n_ga_giu_ngoai=len(ga), k_chung=k_chung,
                k_trai=float(ks.max() - ks.min()), giu_ngoai=ho, h_video_m=h_video, h_m=float(h_m))


def load_khoa(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, dict) or d.get('schema') != SCHEMA_KHOA:
        raise UhError(f'{path}: not a {SCHEMA_KHOA} file')
    for k in ('tam_id', 'h_m', 'ga', 'noc'):
        if k not in d:
            raise UhError(f'{path}: no {k}')
    if not isinstance(d['ga'], list) or len(d['ga']) < 3:
        raise UhError(f'{path}: ga must list >= 3 jigs')
    if abs(float(d['noc']['h_cal_m']) - float(d['h_m'])) > TOL_H:
        raise UhError(f'{path}: noc h_cal_m {d["noc"]["h_cal_m"]} is not h_m {d["h_m"]}')
    return d


def code_info():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        b = subprocess.run(['git', '-C', HERE, 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True,
                           text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_u_h.py', 'c8b_3b.py',
                                     'c8b_a6.py', 'c8_u_imu.py'], capture_output=True, text=True).stdout.strip())
    except OSError:
        return dict(commit=None, branch=None, dirty=None)
    return dict(commit=c or None, branch=b or None, dirty=dirty)


def run(khoa_path, goc, session_cham, out_path):
    K = load_khoa(khoa_path)
    p = lambda rel: os.path.join(goc, rel)                                 # noqa: E731
    why, nguon = [], [dict(vai='khoa', file=os.path.abspath(khoa_path), sha256=cu.sha256(khoa_path))]
    hm = float(K['h_m'])
    for sp in session_cham:
        cu.check_path(sp)
        with open(sp, encoding='utf-8') as f:
            th = float(json.load(f).get('tag_h_m'))
        if abs(th - hm) > TOL_H:
            raise UhError(f'{sp}: tag_h_m {th} is not h_m {hm} of the lock')
        nguon.append(dict(vai='session_cham', file=os.path.abspath(sp), sha256=cu.sha256(sp)))
    if not session_cham:
        raise UhError('--session-cham: at least one scoring session (h_m = its tag_h_m)')
    ga, ga_out = [], []
    for g in K['ga']:
        ses = read_session(p(g['session']), g['session_sha256'])
        nguon.append(dict(vai=f'session ga {g["ten"]}', file=ses['file'], sha256=ses['sha256']))
        try:
            js = jig_scale(p(g['frames']), g['frames_sha256'])
        except UhError as exc:
            why.append(f'ga {g["ten"]}: {exc}')
            continue
        nguon.append(dict(vai=f'frames ga {g["ten"]}', file=js['file'], sha256=js['sha256']))
        ga.append(dict(ten=g['ten'], s=js['s'], H=ses['H'], h_a=ses['h_a'], h_cal=float(g['h_cal_m'])))
        ga_out.append(dict(ten=g['ten'], s=js['s'], n=js['n'], se=js['se'], omega_rad_s=js['omega_rad_s'],
                           H_m=ses['H'], h_a_m=ses['h_a'], h_cal_m=float(g['h_cal_m'])))
    n = K['noc']
    ses_n = read_session(p(n['session']), n['session_sha256'])
    nguon.append(dict(vai='session noc', file=ses_n['file'], sha256=ses_n['sha256']))
    noc = None
    tj = os.path.join(p(n['track']), 'track.json')
    try:
        cu.check_path(tj)
        with open(tj, encoding='utf-8') as f:
            tr = json.load(f)
        if tr.get('video_sha256') != n['video_sha256']:
            raise UhError(f'{tj}: video_sha256 {tr.get("video_sha256")} is not the locked {n["video_sha256"]}')
        if tr.get('session_sha256') != n['session_sha256']:
            raise UhError(f'{tj}: session_sha256 is not the locked {n["session_sha256"]}')
        ws = window_scale(os.path.join(p(n['track']), 'frames.csv'), float(n['trim_s']), int(n['n_min']))
        if tr.get('frames_sha256') not in (None, ws['sha256']):
            raise UhError(f'{tj}: frames_sha256 is not the sha256 of its frames.csv')
        nguon += [dict(vai='track noc', file=os.path.abspath(tj), sha256=cu.sha256(tj)),
                  dict(vai='frames noc', file=ws['file'], sha256=ws['sha256'])]
        noc = dict(s=ws['s'], H=ses_n['H'], h_a=ses_n['h_a'], h_cal=float(n['h_cal_m']))
        noc_out = dict(s=ws['s'], n=ws['n'], se=ws['se'], t_cua_so_s=ws['t_cua_so_s'], H_m=ses_n['H'],
                       h_a_m=ses_n['h_a'], h_cal_m=float(n['h_cal_m']), video_sha256=n['video_sha256'])
    except (UhError, OSError, ValueError) as exc:
        why.append(f'noc: {exc}')
        noc_out = None
    d = dict(schema=a6.SCHEMA_UH, version=VERSION, tam_id=K['tam_id'], h_m=hm, ga=ga_out, noc=noc_out,
             khoa_sha256_noi_dung=content_sha(K), code=code_info(), nguon=nguon)
    if not why and len(ga) >= 3 and noc is not None:
        r = tinh(ga, noc, hm)
        d.update(trang_thai='dat', phan=r['phan'], u_h_chung_m=r['u_h_chung_m'], n_ga_giu_ngoai=r['n_ga_giu_ngoai'],
                 k_chung=r['k_chung'], k_trai=r['k_trai'], giu_ngoai=r['giu_ngoai'], h_video_m=r['h_video_m'],
                 ly_do=[])
    else:
        d.update(trang_thai='thieu', phan=None, u_h_chung_m=None, n_ga_giu_ngoai=len(ga), ly_do=why or
                 ['fewer than 3 jigs'])
    d['kiem_cheo'] = kiem_cheo(K, p, ga, d)
    with_sha = a6.with_sha(d)
    if d['trang_thai'] == 'dat':
        v, w = a6.kiem_u_h(with_sha, K['tam_id'], hm)                  # the reader must accept what we write
        if v is None:
            raise UhError(f'own file refused by c8b_a6.kiem_u_h: {w}')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(with_sha, f, ensure_ascii=False, indent=1)
    return with_sha


def kiem_cheo(K, p, ga, d):
    """Information only: the static jig A clip (no blur), the spread of k (in d)."""
    out = {}
    kc = K.get('kiem_cheo_tinh_A')
    if kc and len(ga) >= 3:
        try:
            ses = read_session(p(kc['session']), kc['session_sha256'])
            fr = p(kc['frames'])
            got = cu.sha256(fr)
            if got != kc['frames_sha256']:
                raise UhError(f'{fr}: sha256 {got} is not the locked {kc["frames_sha256"]}')
            ws = window_scale(fr, 0.0, 1, need_car=True)
            ks = [k_of(g['s'], g['H'], g['h_a'], g['h_cal']) for g in ga]
            kA = float(np.mean([k for g, k in zip(ga, ks) if g['ten'] != 'A']))
            kall = float(np.mean(ks))
            hA, hall = h_of(ws['s'], kA, ses['H'], ses['h_a']), h_of(ws['s'], kall, ses['H'], ses['h_a'])
            sA = next((g['s'] for g in ga if g['ten'] == 'A'), None)
            out['tinh_A'] = dict(s=ws['s'], n=ws['n'], se=ws['se'], H_m=ses['H'], h_a_m=ses['h_a'],
                                 h_cal_m=float(kc['h_cal_m']), h_du_bao_k_giu_ngoai_m=float(hA),
                                 h_du_bao_k_chung_m=float(hall), sai_k_giu_ngoai_m=float(hA - kc['h_cal_m']),
                                 k_tinh_A=float(k_of(ws['s'], ses['H'], ses['h_a'], float(kc['h_cal_m']))),
                                 thang_tinh_tren_cham_A=(float(k_of(ws['s'], ses['H'], ses['h_a'],
                                                                    float(kc['h_cal_m'])) / sA) if sA else None),
                                 vai_tro='thong tin: clip tinh (khong nhoe) cua ga A; khong vao u_h_chung')
        except (UhError, OSError, ValueError, KeyError) as exc:
            out['tinh_A'] = dict(loi=str(exc))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--khoa', required=True)
    ap.add_argument('--goc', default=os.getcwd(), help='root the lock paths are relative to (CLAUDE_THESIS)')
    ap.add_argument('--session-cham', nargs='+', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        d = run(a.khoa, a.goc, a.session_cham, a.out)
    except (UhError, cu.InputError, OSError, ValueError) as exc:
        print(f'loi: {exc}')
        print('#ket_qua,ma=2')
        return 2
    if d['trang_thai'] == 'dat':
        ph = d['phan']
        print(f'u_h_chung {d["u_h_chung_m"] * 1e3:.2f} mm = do trai ba ga {ph["do_trai_ba_ga_m"] * 1e3:.2f} + lech '
              f'video-so tay {ph["lech_video_so_tay_m"] * 1e3:.2f} + {ph["hang_m"] * 1e3:.1f}; h_video '
              f'{d["h_video_m"] * 1e3:.2f} mm (so tay {d["h_m"] * 1e3:.3f})')
        for g in d['giu_ngoai']:
            print(f'  ga {g["ten"]}: k {g["k"]:.5f}, giu ngoai h {g["h_du_bao_m"] * 1e3:.2f} mm '
                  f'(so tay {g["h_cal_m"] * 1e3:.3f}, sai {g["sai_m"] * 1e3:+.2f})')
    else:
        print('thieu: ' + '; '.join(d['ly_do']))
    print(f'#ket_qua,ma=0,trang_thai={d["trang_thai"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
