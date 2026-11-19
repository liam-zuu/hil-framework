#!/usr/bin/env python3
"""
c8b_a6.py: A6, the heading error of the car tag from its tilt, measured on the E clips (turning in place at goc / K1)
for the relative scoring (design C8B v7 section 2, 3, 10: c8b_a6_v2; J v6 = 06_SPECS/2026-11-19_Spec_O1_v11.41_Phu_luc_J_v6.md).
Written by CODE_C8B local_2e6331c2 on feat/c8b-3b-ga, 19/11/2026 (M4; CODE order 19/11, Liam at CODE "duyet ca 8,
giao viec cac phien phu theo cach nhanh"; Liam at CODE_C8B 19/11: "Lam M4 o day").

Model (exact for the lifted points of car_pose): a corner at height h_i = hbar + g.r_i + v s_i (r_i the corner in the
tag frame, car_corners_body order ArUco 0..3, s = (+1, -1, +1, -1)) lifted with ONE height hbar lands at
    P_i = X_i + (g.r_i + v s_i) d_i,   d_i = (P_i - C) / (H - hbar),   X_i = T + R(psi) (1 + sigma) r_i
(C the plumb point, H = Z_c; sigma a print scale of the tag). v is fitted as a nuisance only: the s_i pattern is
orthogonal to the g pattern over a square, but d_i differs between the corners (about side / rho), so an unfitted v of
0.475 mm leaks about 0.0004 into g (found in the tests); its value is NOT reported as the plate warp (design 3: v is not
estimated from video, it mixes the print and camera distortion; |v| <= v_max stays a domain assumption).
g is fitted on the E frames of one place by least squares with the per-frame pose (T, psi) eliminated (design 2b (a),
the 4-corner distortion; the camera-IMU agreement is a check only).

  do    one place (goc | K1): g_E, SE_E (block jackknife over time), U_meo_E (the map residual held - full of KIEMCHEO
        F15 (a) c7_meo_ma_ao_E_v1 at the four corners of every frame used, refitted per H32 case; max change of each
        component), s_E = 2 SE_E + U_meo_E, identifiability (a change of g above |g_E| -> not identified). Frames: the
        moving E frames (F15 mien_E: after E_dau, before E_cuoi) with |omega| <= OMEGA_MAX_RAD_S; the rolling-shutter
        displacement alpha_r (row - mean row) v + alpha_c (col - mean col) v fitted as a nuisance with g (L133, Liam via
        CODE 20/11, "Khop alpha (Recommended)"; replaces the rule 'shutter effect on g below 10 % of SE_E' of design 2b).
        SE_E with and without alpha and the largest alpha-g correlation are reported.
  gop   the A6 file c8b_a6_v2 from the places: compare g_cal (4 caliper heights of the first mounting, design 2c),
        eps_g = max(|s_E|, |g_E - g_cal| + U_cal) + U_chuyen (lap + cho + dong, each once; vector norms, design 2c),
        e_bc (exact lift against the model over the domain, grid refined until the maximum changes < 10 %),
        h_goc_m, u_h_goc_m (design 3), trang_thai {a6, h} and every reason. A6 is 'dat' only when every part is there:
        F15 the full version (not the diagnostic c7_meo_ma_ao_E_v1_chan_doan, KIEMCHEO 02dea17: then 'thieu'), g
        identified, g_cal consistent, U_chuyen all three, the camera-IMU held-out check given and passed, the domain.

Reader functions (data contract F3, CAM / HIEUCHUAN / NHACTAM):
  load_a6(path)                                    the file, schema and SHA checked
  ap_dung(a6, tam_id)                              A6 applies to that plate (same plate as the E clips)
  delta_psi_a6(a6, tag_xy_m, tag_psi_rad, plumb_xy_m, zc_m)
                                                   correction [rad] = [R(psi) g] x (t - C) / (2 (H - hbar)); C, Z_c
                                                   must be the car_pose set-up of the A6 file
  a6du_rad(a6, rho_m)                              residual bound after the correction, eps_g rho / (2 (H - hbar)) + e_bc
  u_psi_hai_dau_rad(a6, kc, p1_xy_m, p2_xy_m, tam_id)
                                                   bound of the camera heading difference of two ends (HIEUCHUAN
                                                   u_psi_cuc): A3(p1) + A3(p2) + A6du(p1) + A6du(p2), before the margin;
                                                   any part missing -> trang_thai 'thieu'
Every number before the margin 1.25 (C8B applies it once, in c8b_khoa_v5). Constants marked [DE XUAT] are proposals.

  python tools/c8b_a6.py do --frames E/frames.csv --session session.json --cho goc [--f15 meo_E_goc.json] --out a6_goc.json
  python tools/c8b_a6.py gop --cho-file a6_goc.json [--cho-file a6_K1.json] --tam-id T1 [--u-lap G] [--u-dong G] \
      [--giu-ngoai held.json] [--u-h-chung M] --out c8b_a6_v2.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
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

SCHEMA = 'c8b_a6_v2'
SCHEMA_CHO = 'c8b_a6_cho_v1'
F15_FULL = 'c7_meo_ma_ao_E_v1'
F15_DIAG = 'c7_meo_ma_ao_E_v1_chan_doan'                   # KIEMCHEO 02dea17 (KCF1-1): never a usable bound
CHO = ('goc', 'K1')
S_PATTERN = np.array([1.0, -1.0, 1.0, -1.0])
# first mounting of the roof plate, caliper heights [mm] (Liam 19/11, nhat_ky_buoi_CAM.md line 27), ArUco order
# 0..3 = front-left, front-right, back-right, back-left; the plate side 144.5 mm
H4_CAL_MM = (124.3, 123.3, 123.1, 126.0)
S_CAL_M = 0.1445
E_CAL_M = 0.0001                                           # caliper error per corner (design 2c)
RHO_MAX_M = 1.6                                            # [DE XUAT] design 2a
NGHIENG_MAX_RAD = math.radians(1.5)                        # [DE XUAT]
V_MAX_M = 0.001                                            # [DE XUAT] design 3 (twice the first plate's 0.475 mm)
OMEGA_MAX_RAD_S = 1.0                                      # [DE XUAT] L133 (1): frames used for the fit with alpha
                                                           # (simulation: 0.3 rad/s g off by about 0.4 SE, 1.0 by 1.1 SE)
N_MIN = 200                                                # [DE XUAT] frames used at least
N_BLOCK = 10                                               # [DE XUAT] jackknife blocks in time
EBC_TOL = 0.10                                             # design 2a: refine until the maximum changes < 10 %


class A6Error(Exception):
    pass


def body(side):
    h = side / 2.0
    return np.array([[h, h], [h, -h], [-h, -h], [-h, h]])


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def g_cal(h4_mm=H4_CAL_MM, side=S_CAL_M):
    """Tilt (gx, gy) and warp v of the plane through four corner heights (least squares, exact for 4 corners)."""
    r = body(side)
    A = np.column_stack([np.ones(4), r[:, 0], r[:, 1], S_PATTERN])
    co = np.linalg.solve(A, np.asarray(h4_mm, float) * 1e-3)
    return np.array(co[1:3]), float(co[3]), float(co[0])


def u_cal(side=S_CAL_M, e=E_CAL_M):
    """Per component 2 e / s (two sides of opposite sign, design 2c), and the vector norm."""
    c = 2.0 * e / side
    return c, c * math.sqrt(2.0)


# ── simulation of the lift (tests, e_bc) ─────────────────────────────────────────────────────────

def lifted_corners(T, psi, g, v, C, H, hbar, side, sigma=0.0):
    """Exact lifted corners P (4, 2) of car_pose for a tag at T, heading psi, tilt g, warp v: the ray from the camera
    (C, H) through the corner at height h_i hits the floor and is lifted with hbar: P = C + (X - C)(H - hbar)/(H - h)."""
    r = body(side)
    X = np.asarray(T, float) + (1.0 + sigma) * r @ _rot(psi).T
    h = hbar + r @ np.asarray(g, float) + v * S_PATTERN
    return np.asarray(C, float) + (X - np.asarray(C, float)) * ((H - hbar) / (H - h))[:, None]


def kabsch_psi(P, side):
    """Heading of the lifted corners as car_pose takes it (rigid fit of the body template, no scale)."""
    r = body(side)
    a, b = r - r.mean(0), P - P.mean(0)
    hm = a.T @ b
    return math.atan2(hm[0, 1] - hm[1, 0], hm[0, 0] + hm[1, 1])


def dpsi_model(g, t_xy, psi, C, H, hbar):
    """design 2a: delta psi_A6 = [R(psi) g] x (t - C) / (2 (H - hbar)) (2-D cross product)."""
    w = _rot(psi) @ np.asarray(g, float)
    d = np.asarray(t_xy, float) - np.asarray(C, float)
    return float(w[0] * d[1] - w[1] * d[0]) / (2.0 * (H - hbar))


# ── fit of g on the E frames ─────────────────────────────────────────────────────────────────────

def fit_g(P, C, H, hbar, side, iters=20, fields=()):
    """g (2), sigma, per-frame (T, psi) and the corner residual RMS [m] from lifted corners P (n, 4, 2): Gauss-Newton on
    P_i = T + R(psi)(1 + sigma) r_i + (g.r_i + v s_i) d_i + sum_k alpha_k F_k,i, d_i = (P_i - C)/(H - hbar) (exact for
    the lift, given P), the per-frame pose eliminated in each step (Schur complement). v, sigma and alpha_k (fields F_k
    (n, 4, 2): the rolling-shutter displacement per unit alpha, L133) are nuisance parameters, alpha not bounded in sign
    (alpha_r was negative at slow rates, design v6 2). Returns also the normal matrix of (g, v, sigma, alpha)."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        raise A6Error('fewer than 3 frames')
    r = body(side)
    C = np.asarray(C, float)
    d = (P - C) / (H - hbar)                                          # (n, 4, 2), fixed: built from the measured P
    F = [np.asarray(f, float) for f in fields]
    K = len(F)
    g, sig, vv, al = np.zeros(2), 0.0, 0.0, np.zeros(K)
    psi = np.array([kabsch_psi(P[f], side) for f in range(n)])
    T = np.array([P[f].mean(0) - (r @ _rot(psi[f]).T).mean(0) for f in range(n)])
    Bg = np.stack([(r[:, 0][:, None] * d).reshape(n, 8), (r[:, 1][:, None] * d).reshape(n, 8),
                   (S_PATTERN[:, None] * d).reshape(n, 8)] + [x.reshape(n, 8) for x in F], axis=2)   # (n, 8, 3 + K)
    ix = [0, 1, 2] + [4 + k for k in range(K)]                        # columns of Bg in the parameter order
    M = None
    for _ in range(iters):
        M, b = np.zeros((4 + K, 4 + K)), np.zeros(4 + K)
        per = []
        for f in range(n):
            R = _rot(psi[f])
            Rr = r @ R.T
            e = (P[f] - T[f] - (1.0 + sig) * Rr - (r @ g + vv * S_PATTERN)[:, None] * d[f]
                 - sum(al[k] * F[k][f] for k in range(K))).reshape(-1)
            A = np.zeros((8, 3))
            A[0::2, 0], A[1::2, 1] = 1.0, 1.0
            A[:, 2] = ((1.0 + sig) * Rr @ np.array([[0.0, 1.0], [-1.0, 0.0]])).reshape(-1)   # d(R r)/dpsi = (-y, x)
            B = np.zeros((8, 4 + K))
            B[:, ix] = Bg[f]
            B[:, 3] = Rr.reshape(-1)
            AtA_inv = np.linalg.inv(A.T @ A)
            Q = np.eye(8) - A @ AtA_inv @ A.T
            M += B.T @ Q @ B
            b += B.T @ Q @ e
            per.append((A, AtA_inv, B, e))
        dgam = np.linalg.solve(M, b)
        for f, (A, AtA_inv, B, e) in enumerate(per):
            dn = AtA_inv @ A.T @ (e - B @ dgam)
            T[f] += dn[:2]
            psi[f] += dn[2]
        g = g + dgam[:2]
        vv += dgam[2]
        sig += dgam[3]
        al = al + dgam[4:]
        if np.max(np.abs(dgam[:4])) < 1e-12 and (K == 0 or np.max(np.abs(dgam[4:])) < 1e-15):
            break
    Rs = np.stack([_rot(a) for a in psi])
    res = P - T[:, None, :] - (1.0 + sig) * np.einsum('ij,fkj->fik', r, Rs) - (r @ g + vv * S_PATTERN)[None, :, None] * d
    for k in range(K):
        res = res - al[k] * F[k]
    return dict(g=g, sigma=float(sig), v_phu=float(vv), alpha=[float(x) for x in al], T=T, psi=psi, M=M,
                res_rms_m=float(np.sqrt(np.mean(np.sum(res ** 2, axis=2)))))


def corr_alpha_g(M):
    """Largest |correlation| between a component of g and an alpha in the least-squares covariance (M^-1, the noise
    scale cancels): the price of fitting alpha with g (L133 (2))."""
    if M is None or M.shape[0] <= 4:
        return None
    cov = np.linalg.inv(M)
    sd = np.sqrt(np.diag(cov))
    return float(max(abs(cov[i, j] / (sd[i] * sd[j])) for i in (0, 1) for j in range(4, M.shape[0])))


def se_jackknife(P, C, H, hbar, side, nb=N_BLOCK, fields=()):
    """Block jackknife SE of g over nb blocks of consecutive frames (frames are not independent in time)."""
    n = len(P)
    nb = min(nb, n // 3)
    if nb < 3:
        raise A6Error('too few frames for the jackknife')
    edges = np.linspace(0, n, nb + 1).astype(int)
    gs = []
    for k in range(nb):
        keep = np.r_[0:edges[k], edges[k + 1]:n]
        gs.append(fit_g(P[keep], C, H, hbar, side, iters=6, fields=[f[keep] for f in fields])['g'])
    gs = np.array(gs)
    return np.sqrt((nb - 1) / nb * np.sum((gs - gs.mean(0)) ** 2, axis=0))


# ── inputs ───────────────────────────────────────────────────────────────────────────────────────

CORNER_COLS = tuple(f'c{i}_{a}' for i in range(4) for a in ('x', 'y'))


def read_e_frames(path):
    """idx, t (s), lifted corners P (n, 4, 2) of the valid frames of a CAM frames.csv with the 8 corner columns
    c0_x .. c3_y (CAM design v2 section 7b, after the parallax, ArUco order); refused without them."""
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in CORNER_COLS + ('idx', 't_s', 'valid')):
        raise A6Error(f'{path}: no corner columns {CORNER_COLS} (CAM frames.csv design v2 7b) or idx/t_s/valid')
    keep = []
    for r in rows:
        if r['valid'] not in ('1', 'True', 'true') or any(r[c] in ('', None) for c in CORNER_COLS):
            continue
        keep.append((int(r['idx']), float(r['t_s']), [float(r[c]) for c in CORNER_COLS]))
    if not keep:
        raise A6Error(f'{path}: no valid frame with its corners')
    idx = np.array([k[0] for k in keep])
    t = np.array([k[1] for k in keep])
    P = np.array([k[2] for k in keep]).reshape(-1, 4, 2)
    if not np.all(np.isfinite(P)) or np.any(np.diff(t) <= 0):
        raise A6Error(f'{path}: corners not finite or times not increasing')
    return idx, t, P


def read_session(path):
    """car_pose set-up of the session: C (plumb), H = Z_c = tag_h / kappa, hbar = tag_h, side."""
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    try:
        k, h, side = float(s['kappa']), float(s['tag_h_m']), float(s['car_side_m'])
        C = np.array([float(v) for v in s['plumb_xy_m']])
    except (KeyError, TypeError, ValueError) as exc:
        raise A6Error(f'{path}: kappa / tag_h_m / car_side_m / plumb_xy_m ({exc})')
    if not (0 < k < 1 and h > 0 and side > 0 and C.shape == (2,) and np.all(np.isfinite(C))):
        raise A6Error(f'{path}: car_pose numbers out of range')
    return dict(C=C, H=h / k, hbar=h, side=side, kappa=k, sha256=cu.sha256(path), file=os.path.abspath(path),
                camera=s.get('camera'))


def read_f15(path, ses):
    """KIEMCHEO F15 (a): (schema, trang_thai, {frame_id: {case: (4, 2)}}, mien_E, sha) checked against the session
    (kappa, plumb, Z_c, h, held_minus_full, bo_H32, truoc_le)."""
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    sch = d.get('schema')
    if sch not in (F15_FULL, F15_DIAG):
        raise A6Error(f'{path}: schema {sch!r}, not {F15_FULL} / {F15_DIAG}')
    if d.get('chieu') != 'held_minus_full' or d.get('bo_H32') is not True or d.get('truoc_le') is not True:
        raise A6Error(f'{path}: chieu held_minus_full, bo_H32 and truoc_le true needed')
    for k, want in (('kappa', ses['kappa']), ('Zc_m', ses['H']), ('h_m', ses['hbar'])):
        if not isinstance(d.get(k), (int, float)) or abs(float(d[k]) - want) > 1e-9 * max(1.0, abs(want)):
            raise A6Error(f'{path}: {k} {d.get(k)!r} is not the car_pose set-up of the session ({want})')
    pl = d.get('plumb_xy_m')
    if not isinstance(pl, list) or len(pl) != 2 or np.max(np.abs(np.array(pl, float) - ses['C'])) > 1e-9:
        raise A6Error(f'{path}: plumb_xy_m is not the plumb point of the session')
    du = {}
    for k in d.get('khung') or []:
        cases = {}
        for name, v in (k.get('du_m') or {}).items():
            a = np.asarray(v, float)
            if a.shape != (4, 2) or not np.all(np.isfinite(a)):
                raise A6Error(f'{path}: frame {k.get("frame_id")} case {name}: du_m not 4 x 2 finite')
            cases[name] = a
        du[int(k['frame_id'])] = cases
    if not du:
        raise A6Error(f'{path}: no frame')
    return dict(schema=sch, trang_thai=d.get('trang_thai'), du=du, mien_E=d.get('mien_E') or {}, cho=d.get('cho'),
                sha256=cu.sha256(path), file=os.path.abspath(path), ly_do=d.get('ly_do') or [])


def omega_of(t, P, side):
    """Turn rate of the tag [rad/s] per frame from the Kabsch heading of the corners (central difference)."""
    psi = np.unwrap(np.array([kabsch_psi(p, side) for p in P]))
    return np.gradient(psi, t)


def shutter_fields(P, t, cam, C, kappa):
    """Rolling-shutter displacement per unit alpha of every corner: (row - mean row of the tag) x corner velocity and
    (col - mean col) x velocity [m per (s/px)]; rows / cols from the session camera at the unlifted floor points."""
    P = np.asarray(P, float)
    C = np.asarray(C, float)
    p = (P - kappa * C) / (1.0 - kappa)                                # unlift (car_pose: P = p + kappa (C - p))
    uv = np.asarray(cam.project(np.column_stack([p.reshape(-1, 2), np.zeros(p.shape[0] * 4)])), float)
    uv = uv.reshape(P.shape[0], 4, 2)
    vel = np.gradient(P, t, axis=0)
    fr = (uv[:, :, 1] - uv[:, :, 1].mean(1, keepdims=True))[:, :, None] * vel
    fc = (uv[:, :, 0] - uv[:, :, 0].mean(1, keepdims=True))[:, :, None] * vel
    return fr, fc


# ── do: one place ────────────────────────────────────────────────────────────────────────────────

def do_place(cho, idx, t, P, ses, f15=None, cam=None):
    """g_E, SE_E, U_meo_E, s_E of one place; trang_thai 'dat' / 'thieu' with the reasons."""
    why = []
    C, H, hb, side = ses['C'], ses['H'], ses['hbar'], ses['side']
    sel = np.ones(len(idx), bool)
    if f15 is not None:
        me = f15['mien_E']
        lo, hi = me.get('sau_E_dau'), me.get('truoc_E_cuoi')
        if isinstance(lo, int) and isinstance(hi, int):
            sel &= (idx > lo) & (idx < hi)                                 # the moving E frames of F15 (E_doan)
    if sel.sum() < N_MIN:
        raise A6Error(f'{cho}: {int(sel.sum())} E frames < {N_MIN}')
    if cam is None:
        raise A6Error('no session camera: the rolling-shutter fields cannot be built (L133)')
    n_e = int(sel.sum())
    w = np.abs(omega_of(t, P, side))
    sel &= w <= OMEGA_MAX_RAD_S                                        # L133 (1): the domain of the alpha fit
    i1 = np.flatnonzero(sel)
    if len(i1) < N_MIN:
        why.append(f'{len(i1)} E frames with |omega| <= {OMEGA_MAX_RAD_S} rad/s (< {N_MIN}): g not taken (L133)')
        return dict(schema=SCHEMA_CHO, cho=cho, g_E=None, SE_E=None, U_meo_E=None, s_E=None, s_E_norm=None,
                    n_khung=int(len(i1)), trang_thai='thieu', ly_do=why,
                    set_up=dict(C=list(map(float, C)), H_m=H, h_m=hb, side_m=side, session_sha256=ses['sha256']))
    fr, fc = shutter_fields(P, t, cam, C, ses['kappa'])
    fl = [fr[i1], fc[i1]]
    fit = fit_g(P[i1], C, H, hb, side, fields=fl)
    se = se_jackknife(P[i1], C, H, hb, side, fields=fl)
    fit0 = fit_g(P[i1], C, H, hb, side)                                # L133 (2): the same frames without alpha
    se0 = se_jackknife(P[i1], C, H, hb, side)
    g = fit['g']
    um = None
    if f15 is None:
        why.append('no F15 (a) meo_ma_ao_E: U_meo_E not computed')
    else:
        if f15.get('cho') not in (None, cho):
            raise A6Error(f'F15 is for {f15.get("cho")!r}, not {cho!r}')
        if f15['schema'] == F15_DIAG or f15['trang_thai'] != 'dat' and f15['schema'] != F15_FULL:
            why.append(f'F15 {f15["schema"]} trang_thai {f15["trang_thai"]!r}: diagnostic only (KIEMCHEO KCF1-1), '
                       f'U_meo_E not a usable bound')
        miss = [int(k) for k in idx[i1] if int(k) not in f15['du']]
        if miss:
            raise A6Error(f'F15 lacks {len(miss)} frames used (first {miss[:3]})')
        names = sorted({n for k in idx[i1] for n in f15['du'][int(k)]})
        if not names:
            raise A6Error('F15 without any H32 case')
        dmax = np.zeros(2)
        for nm in names:
            D = np.array([f15['du'][int(k)].get(nm, np.full((4, 2), np.nan)) for k in idx[i1]])
            if not np.all(np.isfinite(D)):
                raise A6Error(f'F15 case {nm} missing at some frame used')
            gk = fit_g(P[i1] + D, C, H, hb, side, iters=6, fields=fl)['g']
            dmax = np.maximum(dmax, np.abs(gk - g))
        um = dmax
        if np.linalg.norm(dmax) > np.linalg.norm(g):
            why.append(f'g not identified: map noise moves g by {np.linalg.norm(dmax):.5f} > |g_E| '
                       f'{np.linalg.norm(g):.5f} (design 2b)')
    sE = 2.0 * se + um if um is not None else None
    st = 'dat' if not why else 'thieu'
    return dict(schema=SCHEMA_CHO, cho=cho, g_E=list(map(float, g)), sigma=fit['sigma'], SE_E=list(map(float, se)),
                U_meo_E=None if um is None else list(map(float, um)),
                s_E=None if sE is None else list(map(float, sE)),
                s_E_norm=None if sE is None else float(np.linalg.norm(sE)),
                res_rms_m=fit['res_rms_m'], n_khung=int(len(i1)), n_khung_E=n_e,
                man_trap=dict(cach='khop alpha_r, alpha_c lam tham so phu cung g (L133)',
                              alpha_r_s_px=fit['alpha'][0], alpha_c_s_px=fit['alpha'][1],
                              omega_max_rad_s=OMEGA_MAX_RAD_S, tuong_quan_alpha_g_max=corr_alpha_g(fit['M']),
                              g_khong_alpha=list(map(float, fit0['g'])), SE_E_khong_alpha=list(map(float, se0))),
                rho_max_m=float(np.max(np.linalg.norm(P[i1].mean(1) - C, axis=1))),
                f15=None if f15 is None else dict(schema=f15['schema'], trang_thai=f15['trang_thai'],
                                                 file=f15['file'], sha256=f15['sha256']),
                set_up=dict(C=list(map(float, C)), H_m=H, h_m=hb, side_m=side, session_sha256=ses['sha256']),
                trang_thai=st, ly_do=why)


# ── e_bc ─────────────────────────────────────────────────────────────────────────────────────────

def _ebc_grid(g_hat, eps_g, C, H, hbar, side, rho_max, v_max, k):
    """Largest |exact - model| on a grid of k rho steps, 2k directions and tag headings, g_hat and 8 points of the
    circle eps_g around it, v in {-v_max, 0, v_max} (vectorised)."""
    rhos = np.linspace(0.0, rho_max, k + 1)
    angs = np.linspace(0.0, 2 * math.pi, 2 * k, endpoint=False)
    gd = np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
    gs = np.vstack([np.asarray(g_hat, float)[None, :],
                    np.asarray(g_hat, float)[None, :] + eps_g * np.column_stack([np.cos(gd), np.sin(gd)])])
    R_, A_, S_, G_, V_ = np.meshgrid(rhos, angs, angs, np.arange(len(gs)), np.array([-v_max, 0.0, v_max]),
                                     indexing='ij')
    R_, A_, S_, G_, V_ = (x.ravel() for x in (R_, A_, S_, G_, V_))
    g = gs[G_]                                                       # (N, 2)
    T = np.asarray(C, float)[None, :] + R_[:, None] * np.column_stack([np.cos(A_), np.sin(A_)])
    r = body(side)
    c, s_ = np.cos(S_), np.sin(S_)
    X = T[:, None, :] + np.stack([c[:, None] * r[None, :, 0] - s_[:, None] * r[None, :, 1],
                                  s_[:, None] * r[None, :, 0] + c[:, None] * r[None, :, 1]], axis=2)
    h = hbar + g @ r.T + V_[:, None] * S_PATTERN[None, :]            # (N, 4)
    P = np.asarray(C, float)[None, None, :] + (X - np.asarray(C, float)[None, None, :]) * ((H - hbar) / (H - h))[:, :, None]
    a = r - r.mean(0)
    b = P - P.mean(1, keepdims=True)
    hm = np.einsum('ij,nik->njk', a, b)                              # a^T b per configuration
    ex = np.arctan2(hm[:, 0, 1] - hm[:, 1, 0], hm[:, 0, 0] + hm[:, 1, 1])
    ex = np.remainder(ex - S_ + math.pi, 2 * math.pi) - math.pi
    w = np.column_stack([c * g[:, 0] - s_ * g[:, 1], s_ * g[:, 0] + c * g[:, 1]])
    dd = T - np.asarray(C, float)[None, :]
    model = (w[:, 0] * dd[:, 1] - w[:, 1] * dd[:, 0]) / (2.0 * (H - hbar))
    return float(np.max(np.abs(ex - model)))


def e_bc(g_hat, eps_g, C, H, hbar, side, rho_max=RHO_MAX_M, v_max=V_MAX_M, k=4):
    """Largest |exact heading error - model| over the domain: tag centre within rho_max of C, every direction and tag
    heading, g within eps_g of g_hat, |v| <= v_max; the grid is halved until the maximum changes by < EBC_TOL
    (design 2a; an index, no claim between the nodes). Returns (value, grid k)."""
    prev = _ebc_grid(g_hat, eps_g, C, H, hbar, side, rho_max, v_max, k)
    for _ in range(3):
        k *= 2
        cur = _ebc_grid(g_hat, eps_g, C, H, hbar, side, rho_max, v_max, k)
        if prev > 0 and abs(cur - prev) < EBC_TOL * prev:
            return cur, k
        prev = cur
    return prev, k


# ── gop: the A6 file ─────────────────────────────────────────────────────────────────────────────

def gop(places, tam_id, u_lap=None, u_dong=None, giu_ngoai=None, u_h_chung=None, nguon=(), with_ebc=True):
    """c8b_a6_v2 from the place results (design 2c, 3, 10)."""
    why = []
    if not places:
        raise A6Error('no place')
    su = {json.dumps(p['set_up'], sort_keys=True) for p in places}
    if len(su) != 1:
        raise A6Error('places measured with different car_pose set-ups')
    s0 = places[0]['set_up']
    C, H, hb, side = np.array(s0['C']), s0['H_m'], s0['h_m'], s0['side_m']
    for p in places:
        if p['trang_thai'] != 'dat':
            why += [f'{p["cho"]}: {w}' for w in p['ly_do']] or [f'{p["cho"]}: trang_thai {p["trang_thai"]}']
    ok = [p for p in places if p['g_E'] is not None and p['s_E'] is not None]
    gc, v_cal, h_cal = g_cal()
    uc, uc_norm = u_cal()
    g_hat = s_norm = None
    if ok:
        w = [1.0 / max(p['s_E_norm'], 1e-12) ** 2 for p in ok]
        g_hat = np.sum([wi * np.array(p['g_E']) for wi, p in zip(w, ok)], axis=0) / sum(w)
        s_norm = max(p['s_E_norm'] for p in ok)
    u_cho = None
    gp = {p['cho']: np.array(p['g_E']) for p in ok}
    if len(gp) == 2:
        u_cho = float(np.linalg.norm(gp['goc'] - gp['K1']))
    elif len(places) == 1:
        why.append('one place only: U_chuyen.cho (goc - K1) not measured')
    else:
        why.append('a place without g: U_chuyen.cho not measured')
    if u_lap is None:
        why.append('U_chuyen.lap missing (two windows of different mountings on the same image footprint)')
    if u_dong is None:
        why.append('U_chuyen.dong missing (leave-one-out over the C8 runs by acceleration band, design 2c)')
    uch = None if None in (u_lap, u_cho, u_dong) else float(u_lap + u_cho + u_dong)
    dist = None if g_hat is None else float(np.linalg.norm(g_hat - gc))
    nhat_quan = None
    if dist is not None:
        nhat_quan = bool(dist <= s_norm + uc_norm + (u_lap or 0.0))
        if not nhat_quan:
            why.append(f'g_cal_bat_nhat: |g_E - g_cal| {dist:.5f} > s_E {s_norm:.5f} + U_cal {uc_norm:.5f}'
                       f' + dg_lap {u_lap or 0.0:.5f} (design 2c)')
    eps = None if (dist is None or uch is None) else float(max(s_norm, dist + uc_norm) + uch)
    if giu_ngoai is None:
        why.append('no camera-IMU held-out check (design 2b): A6 not checked')
    elif giu_ngoai.get('trang_thai') != 'dat':
        why.append(f'camera-IMU held-out check {giu_ngoai.get("trang_thai")!r} (design 2b)')
    ebc, ebc_n = (None, None)
    if g_hat is not None and with_ebc:
        ebc, ebc_n = e_bc(g_hat, eps if eps is not None else (s_norm + uc_norm), C, H, hb, side)
    mien_ok = None
    if g_hat is not None:
        tilt = math.atan(float(np.linalg.norm(g_hat)) + (eps or 0.0))
        rho = max(p.get('rho_max_m') or 0.0 for p in ok)
        mien_ok = bool(tilt <= NGHIENG_MAX_RAD and rho <= RHO_MAX_M)
        if not mien_ok:
            why.append(f'outside the A6 domain: tilt {math.degrees(tilt):.2f} deg (<= 1.5) or rho {rho:.2f} m (<= 1.6)')
    r = body(side)
    h_goc = None if g_hat is None else [float(hb + ri @ g_hat) for ri in r]
    u_g = eps
    u_h_goc = None if (u_h_chung is None or u_g is None) else [float(u_h_chung + np.linalg.norm(ri) * u_g + V_MAX_M)
                                                               for ri in r]
    why_h = [] if u_h_goc is not None else ['u_h_chung (video multi-height, design 3) or eps_g missing']
    st_a6 = 'dat' if (not why and eps is not None and ebc is not None) else 'thieu'
    return dict(schema=SCHEMA, tam_id=tam_id, g_b=None if g_hat is None else list(map(float, g_hat)),
                g_E_cho={p['cho']: p['g_E'] for p in places}, eps_g=eps, s_E=s_norm,
                SE_E={p['cho']: p['SE_E'] for p in places}, U_meo_E={p['cho']: p['U_meo_E'] for p in places},
                U_chuyen=dict(lap=u_lap, cho=u_cho, dong=u_dong), g_cal=list(map(float, gc)), v_cal_m=v_cal,
                h_cal_m=h_cal, U_cal=uc, U_cal_norm=uc_norm, khoang_g_cal=dist, nhat_quan=nhat_quan,
                e_bc_rad=ebc, e_bc_luoi=ebc_n,
                a6du_rad_per_m=None if eps is None else float(eps / (2.0 * (H - hb))),
                H_m=float(H), h_m=float(hb), plumb_xy_m=list(map(float, C)), side_m=float(side),
                mien=dict(rho_max_m=RHO_MAX_M, nghieng_max_rad=NGHIENG_MAX_RAD, v_max_m=V_MAX_M, dat=mien_ok),
                h_goc_m=h_goc, u_h_goc_m=u_h_goc, u_h_chung_m=u_h_chung, u_h_rieng_m=None,
                giu_ngoai=giu_ngoai, trang_thai=dict(a6=st_a6, h='dat' if u_h_goc is not None else 'thieu'),
                ly_do=why, ly_do_h=why_h, nguon=list(nguon), truoc_le=True)


# ── readers ──────────────────────────────────────────────────────────────────────────────────────

def load_a6(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, dict) or d.get('schema') != SCHEMA:
        raise A6Error(f'{path}: not a {SCHEMA} file')
    d['_sha256'] = cu.sha256(path)
    return d


def ap_dung(a6, tam_id):
    """A6 applies only to the plate of the E clips (a new plate: A6 thieu, design 2c / T16.8)."""
    return isinstance(a6, dict) and a6.get('tam_id') is not None and a6.get('tam_id') == tam_id


def _same_setup(a6, plumb_xy_m, zc_m):
    if np.max(np.abs(np.asarray(plumb_xy_m, float) - np.asarray(a6['plumb_xy_m'], float))) > 1e-9 \
            or abs(float(zc_m) - float(a6['H_m'])) > 1e-9:
        raise A6Error('plumb point / Z_c are not the car_pose set-up of the A6 file')


def delta_psi_a6(a6, tag_xy_m, tag_psi_rad, plumb_xy_m, zc_m):
    """Heading correction [rad] of a tag pose (subtract from the car_pose heading), or None when A6 is not dat."""
    _same_setup(a6, plumb_xy_m, zc_m)
    if (a6.get('trang_thai') or {}).get('a6') != 'dat' or a6.get('g_b') is None:
        return None
    return dpsi_model(a6['g_b'], tag_xy_m, tag_psi_rad, a6['plumb_xy_m'], a6['H_m'], a6['h_m'])


def a6du_rad(a6, rho_m):
    """Residual bound after the correction at a distance rho from the plumb point, or None when A6 is not dat."""
    if (a6.get('trang_thai') or {}).get('a6') != 'dat' or a6.get('eps_g') is None or a6.get('e_bc_rad') is None:
        return None
    return float(a6['eps_g'] * rho_m / (2.0 * (a6['H_m'] - a6['h_m'])) + a6['e_bc_rad'])


def _a3_rad(kc, p_xy):
    """A3 of the camera at a floor point from KIEMCHEO's camera file (c8_cam_u_vao: A3.max_deg; by position when the
    file carries A3.theo_vi_tri). [CACH HIEU] the largest A3 of the file at every point until the by-position form is
    fixed with KIEMCHEO."""
    a3 = (kc or {}).get('A3') if isinstance(kc, dict) else None
    v = (a3 or {}).get('max_deg') if isinstance(a3, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        return None
    return math.radians(float(v))


def u_psi_hai_dau_rad(a6, kc, p1_xy_m, p2_xy_m, tam_id):
    """HIEUCHUAN u_psi_cuc (CODE 19/11): bound of the camera heading difference of two ends, before the margin."""
    why = []
    if not ap_dung(a6, tam_id):
        why.append(f'A6 not for plate {tam_id!r} (A6 of {(a6 or {}).get("tam_id")!r})')
    parts = []
    for nm, p in (('p1', p1_xy_m), ('p2', p2_xy_m)):
        a3 = _a3_rad(kc, p)
        rho = float(np.linalg.norm(np.asarray(p, float) - np.asarray((a6 or {}).get('plumb_xy_m', [np.nan] * 2))))
        du = a6du_rad(a6, rho) if isinstance(a6, dict) else None
        if a3 is None:
            why.append(f'{nm}: A3 missing in the camera file')
        if du is None:
            why.append(f'{nm}: A6 not dat ({((a6 or {}).get("trang_thai") or {}).get("a6")!r})')
        parts.append((a3, du))
    if why:
        return dict(u_rad=None, trang_thai='thieu', ly_do=why)
    return dict(u_rad=float(sum(a + d for a, d in parts)), trang_thai='dat', ly_do=[],
                phan=dict(A3_rad=[a for a, _ in parts], A6du_rad=[d for _, d in parts]))


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────

def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_a6.py'], capture_output=True,
                           text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), file='tools/c8b_a6.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_a6.py')


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write(path, d):
    cu.check_path(path)
    with open(path, 'x', encoding='utf-8') as f:
        f.write(json.dumps(_clean(d), indent=1, ensure_ascii=False, allow_nan=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('do')
    a1.add_argument('--frames', required=True)
    a1.add_argument('--session', required=True)
    a1.add_argument('--cho', required=True, choices=CHO)
    a1.add_argument('--f15', default=None)
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('gop')
    a2.add_argument('--cho-file', action='append', required=True)
    a2.add_argument('--tam-id', required=True)
    a2.add_argument('--u-lap', type=float, default=None)
    a2.add_argument('--u-dong', type=float, default=None)
    a2.add_argument('--giu-ngoai', default=None, help='camera-IMU held-out check file (trang_thai dat)')
    a2.add_argument('--u-h-chung', type=float, default=None)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'do':
            ses = read_session(a.session)
            idx, t, P = read_e_frames(a.frames)
            f15 = read_f15(a.f15, ses) if a.f15 else None
            import c7_clip_probe as cp
            cam = cp.Camera(**{k: ses['camera'][k] for k in ('model', 'f', 'cx', 'cy', 'k', 'rvec', 'tvec')})
            out = do_place(a.cho, idx, t, P, ses, f15, cam)
            out.update(nguon=[dict(vai='frames', file=os.path.abspath(a.frames), sha256=cu.sha256(a.frames)),
                              dict(vai='session', file=ses['file'], sha256=ses['sha256'])]
                       + ([dict(vai='f15', file=f15['file'], sha256=f15['sha256'])] if f15 else []),
                       script=script_state(), luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            _write(a.out, out)
            print(f'{a.cho}: g_E {out["g_E"]} SE {out["SE_E"]} U_meo {out["U_meo_E"]} n {out["n_khung"]}; '
                  f'{out["trang_thai"]} {"; ".join(out["ly_do"])}')
        else:
            places, nguon = [], []
            for p in a.cho_file:
                cu.check_path(p)
                with open(p, encoding='utf-8') as f:
                    d = json.load(f)
                if d.get('schema') != SCHEMA_CHO:
                    raise A6Error(f'{p}: not a {SCHEMA_CHO} file')
                places.append(d)
                nguon.append(dict(vai=f'cho {d["cho"]}', file=os.path.abspath(p), sha256=cu.sha256(p)))
            if len({p['cho'] for p in places}) != len(places):
                raise A6Error('a place given twice')
            gn = None
            if a.giu_ngoai:
                with open(a.giu_ngoai, encoding='utf-8') as f:
                    gn = json.load(f)
                nguon.append(dict(vai='giu_ngoai', file=os.path.abspath(a.giu_ngoai), sha256=cu.sha256(a.giu_ngoai)))
            out = gop(places, a.tam_id, a.u_lap, a.u_dong, gn, a.u_h_chung, nguon)
            out.update(script=script_state(), luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            _write(a.out, out)
            print(f'A6 {out["trang_thai"]}; g_b {out["g_b"]} eps_g {out["eps_g"]} e_bc {out["e_bc_rad"]}; '
                  f'{"; ".join(out["ly_do"])}')
    except (A6Error, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_a6: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
