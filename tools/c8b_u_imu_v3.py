#!/usr/bin/env python3
"""
c8b_u_imu_v3.py: U_IMU,D(k), the IMU part of the bound of a heading DIFFERENCE frame k minus reference window W, per
frame (design C8B v7 section 7b, V4-07, V5-09), file u_imu_dong_v3. Written by CODE_C8B local_f64f609e on
feat/c8b-3b-ga, 20/11/2026 (M6 of the C8B code order, design v7 section 13). Uses the IMU sample primitives of
c8_u_imu (rv_samples, interp_yaw, turn_totals: same yaw as the camera scorer); u_imu_dong_v2 stays as it is.

  U_IMU,D(k) = d+ Pbar_k + d- Nbar_k + rdot |t_k - tbar_W| + 2 q + E_noi(k) + Ebar_noi(W) + E_tre(k) + Ebar_tre(W)
               + E_chuyen(k)
  P(k, j), N(k, j)  total CCW / CW turn between t_j and t_k from every report (c8_u_imu.turn_totals), averaged over
                    j in W with the weights of the W combination (equal weights: trung_binh_go_vong); tbar_W the same
                    mean of t_j.
  d+, d-            scale bounds (thang-gop: delta_plus, delta_minus) on the scale-corrected yaw.
  rdot              drift rate bound (troi: beta_max), q the still-yaw block spread of troi (2 q stands for
                    2 (sigma_q + sigma_n) of the design: q is measured on the yaw itself, so it holds quantisation and
                    noise together [CACH HIEU, sent to CODE 20/11]).
  E_tre(t)          largest |psi(t + d) - psi(t)| over d in [-u_minus, +u_plus] (one side each way, design 7b), on the
                    RV samples and their interpolation; u = clock pairing (the node's u_cham-, u_cham+) + u_lag.
  E_noi(t)          the interpolation index of the band of t (table `noi`: on HELD-OUT camera blocks of other C8 runs,
                    raw 60 fps camera yaw sampled at the RV times of the log, interpolated as the RV, the largest
                    residual against the raw camera yaw per band of |omega|, |alpha| and RV gap; camera noise not
                    removed: the safe side); no held-out block in the band, or an RV gap > H_LIM_S -> n_undef.
  E_chuyen(k)       the spread of the leave-one-run-out fits turned into an angle of the pair (k, W):
                    dd+ Pbar + dd- Nbar + drdot |t_k - tbar_W| + max |psi(t + d) - psi(t)| over d in [-dtau, dtau].
  Freshness         every RV report used around t (the two bracketing the interval [t - u_minus, t + u_plus]) has
                    an age <= TUOI_MAX_MS; else n_undef (design 7b "do moi").
All numbers before the margin (C8B applies 1.25 once, in c8b_khoa_v5). A missing part -> None with the reason
(design 0.2: never 0).

  python tools/c8b_u_imu_v3.py noi --cap frames.csv,log.csv,D[,run] [--cap ...] --thang thang.json --tham-so P.json \
      --out noi.json
  python tools/c8b_u_imu_v3.py chuyen --loo loo1.json [--loo ...] --out chuyen.json
  python tools/c8b_u_imu_v3.py tham-so --thang thang.json --troi troi.json --tre tre.json --u-lag-s X --noi noi.json \
      --chuyen chuyen.json --out u_imu_dong_v3.json
Library: load_v3(path); u_imu_delta(P3, S, t_k, t_W, omega_k, alpha_k, omega_W, alpha_W, u_minus, u_plus, L=None).
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'u_imu_dong_v3'
NOI_VERSION = 'c8b_u_imu_noi_v1'
CHUYEN_VERSION = 'c8b_u_imu_chuyen_v1'
TUOI_MAX_MS = 15.0                          # [DE XUAT] design 7b: RV report age
DAI_OMEGA = (0.0, 0.5, 1.0, 2.0, 4.0)      # [DE XUAT] |omega| bands [rad/s] of E_noi
DAI_ALPHA = (0.0, 2.0, 5.0, 20.0)          # [DE XUAT] |alpha| bands [rad/s^2]
DAI_HO = (0.0, 0.055, cu.H_LIM_S)          # [DE XUAT] RV gap bands [s]; above H_LIM_S: n_undef
TRE_BUOC_S = 0.001                          # grid step of the d search of E_tre


class U3Error(ValueError):
    pass


def _band(v, edges):
    if not math.isfinite(v) or v < edges[0]:
        return None
    for j in range(len(edges) - 1):
        if edges[j] <= v < edges[j + 1] or (j == len(edges) - 2 and v == edges[-1]):
            return j
    return None


def band_key(omega, alpha, gap):
    b = (_band(abs(omega), DAI_OMEGA), _band(abs(alpha), DAI_ALPHA), _band(gap, DAI_HO))
    return None if None in b else '{}_{}_{}'.format(*b)


def rv_gap(S, t):
    """Length of the RV interval (kept reports) holding t, NaN outside."""
    ts = S['t']
    i = int(np.searchsorted(ts, t, side='right'))
    if i <= 0 or i >= len(ts):
        return math.nan
    return float(ts[i] - ts[i - 1])


def e_tre(S, t, u_minus, u_plus, step=TRE_BUOC_S):
    """Largest |psi(t + d) - psi(t)| over d in [-u_minus, u_plus] (one side each way) or None when the yaw is not
    readable somewhere there."""
    if not (u_minus >= 0 and u_plus >= 0):
        return None
    d = np.concatenate([-np.arange(0.0, u_minus + step / 2, step), np.arange(step, u_plus + step / 2, step)])
    d = np.unique(np.clip(np.concatenate([d, [-u_minus, u_plus]]), -u_minus, u_plus))
    y = cu.interp_yaw(S, t + d)
    y0 = cu.interp_yaw(S, np.array([t]))[0]
    if not (np.all(np.isfinite(y)) and math.isfinite(y0)):
        return None
    return float(np.max(np.abs(y - y0)))


def tuoi_ok(S, t, u_minus, u_plus):
    """Every kept RV report bracketing [t - u_minus, t + u_plus] has age <= TUOI_MAX_MS (needs S['age_ms'])."""
    if 'age_ms' not in S:
        return False
    ts, age = S['t'], S['age_ms']
    i0 = int(np.searchsorted(ts, t - u_minus, side='right')) - 1
    i1 = int(np.searchsorted(ts, t + u_plus, side='left'))
    if i0 < 0 or i1 >= len(ts):
        return False
    return bool(np.all(age[i0:i1 + 1] <= TUOI_MAX_MS))


def rv_samples_age(L, **kw):
    """c8_u_imu.rv_samples plus the age [ms] of every kept report (the freshness check)."""
    S = cu.rv_samples(L, **kw)
    S['age_ms'] = L['age_r_ms'][S['rows']][S['keep']]
    return S


def _noi_lookup(P3, key):
    if key is None:
        return None, 'outside the bands of E_noi'
    v = (P3['E_noi'].get('bang') or {}).get(key)
    if v is None or v.get('max_rad') is None:
        return None, f'E_noi band {key} without held-out block'
    return float(v['max_rad']), ''


def u_imu_delta(P3, S, t_k, t_W, omega_k, alpha_k, omega_W, alpha_W, u_minus, u_plus):
    """(U_IMU,D(k) [rad] or None, parts, reasons) for the frame at log time t_k against the W frames at t_W (equal
    weights), omega / alpha of the camera at those frames, the clock pairing u_minus, u_plus of the node (one side
    each way; u_lag of the file is added here)."""
    why = []
    t_W = np.asarray(t_W, float)
    om_W, al_W = np.asarray(omega_W, float), np.asarray(alpha_W, float)
    if t_W.size == 0 or om_W.shape != t_W.shape or al_W.shape != t_W.shape:
        raise U3Error('W frames, their omega and alpha: same non-empty length')
    um, up = u_minus + P3['u_lag_s'], u_plus + P3['u_lag_s']
    PN = []
    try:
        for tj in t_W:
            a, b = min(tj, t_k), max(tj, t_k)
            PN.append(cu.turn_totals(S, a, b))
    except cu.InputError as e:
        return None, {}, [f'turn totals: {e}']
    Pb = float(np.mean([p for p, _ in PN]))
    Nb = float(np.mean([n for _, n in PN]))
    tb = float(np.mean(t_W))
    parts = dict(thang=P3['delta_plus'] * Pb + P3['delta_minus'] * Nb, troi=P3['r_dot'] * abs(t_k - tb),
                 q=2.0 * P3['q'])
    ek = e_tre(S, t_k, um, up)
    ew = [e_tre(S, tj, um, up) for tj in t_W]
    if ek is None or any(x is None for x in ew):
        why.append('E_tre: IMU yaw not readable over the clock domain')
    else:
        parts['tre'] = ek
        parts['tre_W'] = float(np.mean(ew))
    nk, wk = _noi_lookup(P3, band_key(omega_k, alpha_k, rv_gap(S, t_k)))
    nw = [_noi_lookup(P3, band_key(o, a, rv_gap(S, tj))) for o, a, tj in zip(om_W, al_W, t_W)]
    if nk is None:
        why.append(f'E_noi(k): {wk}')
    if any(v is None for v, _ in nw):
        why.append('E_noi(W): ' + '; '.join(sorted({w for v, w in nw if v is None})))
    if nk is not None and all(v is not None for v, _ in nw):
        parts['noi'] = nk
        parts['noi_W'] = float(np.mean([v for v, _ in nw]))
    ch = P3['E_chuyen']
    dtau = e_tre(S, t_k, ch['d_tau_s'], ch['d_tau_s'])
    if dtau is None:
        why.append('E_chuyen: IMU yaw not readable over the lag spread')
    else:
        parts['chuyen'] = ch['d_delta_plus'] * Pb + ch['d_delta_minus'] * Nb + ch['d_r_dot'] * abs(t_k - tb) + dtau
    if not tuoi_ok(S, t_k, um, up) or not all(tuoi_ok(S, tj, um, up) for tj in t_W):
        why.append(f'RV report older than {TUOI_MAX_MS} ms (or not covered) over the clock domain')
    parts.update(P_bar=Pb, N_bar=Nb, t_bar_W=tb)
    if why:
        return None, parts, why
    u = sum(parts[k] for k in ('thang', 'troi', 'q', 'tre', 'tre_W', 'noi', 'noi_W', 'chuyen'))
    return float(u), parts, []


# ── producers ────────────────────────────────────────────────────────────────────────────────────

def noi_table(caps, P):
    """E_noi per band from held-out camera blocks: caps [(frames, L, offset_s, run)] (c7 frames.csv, the log of that
    run, t_log = t_s + offset); P the scale / tau parameters of thang / tre. The camera yaw (raw, every valid frame) is
    sampled at the RV report times, interpolated linearly as the RV, the residual at every camera frame inside an RV
    interval <= H_LIM_S kept by the band of that frame (|omega|, |alpha| from the camera, gap)."""
    bang, runs = {}, []
    for F, L, off, run in caps:
        S = cu.rv_samples(L, tau_hat=P['tau_hat_s'], s_plus=P['s_plus'], s_minus=P['s_minus'])
        tl = F['t'] + off
        psi = np.unwrap(F['psi'])
        ok = (tl >= S['t'][0]) & (tl <= S['t'][-1])
        if ok.sum() < 3:
            continue
        cam_at_rv = np.interp(S['t'], tl, psi, left=np.nan, right=np.nan)
        om = np.gradient(psi, F['t'])
        al = np.gradient(om, F['t'])
        for i in np.nonzero(ok)[0]:
            j = int(np.searchsorted(S['t'], tl[i], side='right'))
            if j <= 0 or j >= len(S['t']):
                continue
            a, b = S['t'][j - 1], S['t'][j]
            if b - a > cu.H_LIM_S or not (math.isfinite(cam_at_rv[j - 1]) and math.isfinite(cam_at_rv[j])):
                continue
            w = (tl[i] - a) / (b - a)
            res = abs(cam_at_rv[j - 1] + w * (cam_at_rv[j] - cam_at_rv[j - 1]) - psi[i])
            key = band_key(om[i], al[i], b - a)
            if key is None:
                continue
            e = bang.setdefault(key, dict(max_rad=0.0, n=0, luot=set()))
            e['max_rad'] = max(e['max_rad'], float(res))
            e['n'] += 1
            e['luot'].add(run)
        runs.append(run)
    for e in bang.values():
        e['luot'] = sorted(e['luot'])
    return dict(version=NOI_VERSION, bang=bang, luot=runs, dai=dict(omega=list(DAI_OMEGA), alpha=list(DAI_ALPHA),
                                                                     ho=list(DAI_HO)),
                loai='chi so do truc tiep tren khoi giu ngoai; nhieu mot khung camera khong tru (phia an toan)')


def chuyen_spread(loos):
    """E_chuyen parameters from the leave-one-run-out fits [{delta_plus, delta_minus, r_dot, tau_hat_s}]: the spread
    (max - min) of each."""
    if len(loos) < 2:
        raise U3Error('E_chuyen needs at least two leave-one-run-out fits')
    out = {}
    for k, name in (('delta_plus', 'd_delta_plus'), ('delta_minus', 'd_delta_minus'), ('r_dot', 'd_r_dot'),
                    ('tau_hat_s', 'd_tau_s')):
        v = [cu._finite(x.get(k), f'loo {k}') for x in loos]
        out[name] = float(max(v) - min(v))
    return dict(version=CHUYEN_VERSION, n_loo=len(loos), **out)


def build_v3(thang, troi, tre, u_lag_s, noi, chuyen):
    cu._check_thang(thang)
    if (troi.get('s_plus', 0.0), troi.get('s_minus', 0.0)) != (thang['s_plus'], thang['s_minus']):
        raise U3Error('troi with another scale correction than thang')
    if noi.get('version') != NOI_VERSION or chuyen.get('version') != CHUYEN_VERSION:
        raise U3Error('noi / chuyen files of another version')
    for v, w in ((u_lag_s, 'u_lag_s'), (troi.get('beta_max'), 'beta_max'), (troi.get('q'), 'q'),
                 (tre.get('tau_hat_s'), 'tau_hat_s')):
        if w == 'tau_hat_s':
            cu._finite(v, w, -0.2, 0.2)
        else:
            cu._finite(v, w, 0.0)
    if tre.get('thu'):
        raise U3Error('tre is a trial (thu): not lockable')
    return dict(version=VERSION, delta_plus=thang['delta_plus'], delta_minus=thang['delta_minus'],
                s_plus=thang['s_plus'], s_minus=thang['s_minus'], r_dot=troi['beta_max'], q=troi['q'],
                tau_hat_s=tre['tau_hat_s'], u_lag_s=float(u_lag_s), E_noi=noi, E_chuyen=chuyen,
                tuoi_max_ms=TUOI_MAX_MS, h_lim_s=cu.H_LIM_S, j_lim_s=cu.J_LIM_S, period_s=cu.PERIOD_S,
                truoc_le=True, cong_thuc='d+ Pbar + d- Nbar + rdot |t - tbar_W| + 2 q + E_noi + Ebar_noi(W) + '
                                         'E_tre + Ebar_tre(W) + E_chuyen (thiet ke C8B v7 7b)')


def load_v3(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if d.get('version') != VERSION:
        raise U3Error(f'{path}: not a {VERSION} file')
    for k in ('delta_plus', 'delta_minus', 'r_dot', 'q', 'u_lag_s'):
        cu._finite(d.get(k), k, 0.0)
    return d


def _read(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write(path, d):
    cu.check_path(path)
    c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_u_imu_v3.py', 'c8_u_imu.py'],
                       capture_output=True, text=True)
    d['script'] = dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()),
                       luc_chay=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
    with open(path, 'x', encoding='utf-8') as f:
        f.write(json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('noi')
    a1.add_argument('--cap', action='append', required=True, help='frames.csv,log.csv,offset_s[,run]')
    a1.add_argument('--thang', required=True)
    a1.add_argument('--tre', required=True)
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('chuyen')
    a2.add_argument('--loo', action='append', required=True)
    a2.add_argument('--out', required=True)
    a3 = sub.add_parser('tham-so')
    for k in ('--thang', '--troi', '--tre', '--noi', '--chuyen', '--out'):
        a3.add_argument(k, required=True)
    a3.add_argument('--u-lag-s', type=float, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'noi':
            th, tr = _read(a.thang), _read(a.tre)
            P = dict(tau_hat_s=tr['tau_hat_s'], s_plus=th['s_plus'], s_minus=th['s_minus'])
            caps, src = [], []
            for c in a.cap:
                parts = c.split(',')
                if len(parts) not in (3, 4):
                    raise U3Error(f'--cap {c!r}: frames,log,offset[,run]')
                fp, lp, off = parts[0], parts[1], float(parts[2])
                run = parts[3] if len(parts) == 4 else os.path.basename(os.path.dirname(os.path.abspath(fp)))
                caps.append((cu.read_frames(fp), cu.load_log(lp), off, run))
                src += [dict(vai='frames', file=os.path.abspath(fp), sha256=cu.sha256(fp)),
                        dict(vai='log', file=os.path.abspath(lp), sha256=cu.sha256(lp))]
            out = noi_table(caps, P)
            out['nguon'] = src
        elif a.cmd == 'chuyen':
            out = chuyen_spread([_read(p) for p in a.loo])
            out['nguon'] = [dict(file=os.path.abspath(p), sha256=cu.sha256(p)) for p in a.loo]
        else:
            out = build_v3(_read(a.thang), _read(a.troi), _read(a.tre), a.u_lag_s, _read(a.noi), _read(a.chuyen))
            out['nguon'] = [dict(vai=k, file=os.path.abspath(getattr(a, k)), sha256=cu.sha256(getattr(a, k)))
                            for k in ('thang', 'troi', 'tre', 'noi', 'chuyen')]
        _write(a.out, out)
        print(f'{a.cmd}: written {a.out}')
    except (U3Error, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_u_imu_v3: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
