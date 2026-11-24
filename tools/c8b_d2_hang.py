#!/usr/bin/env python3
"""
c8b_d2_hang.py: the rows of the D2 check (design C8B v7 section 7c, 7b; design note side-session-notes
2026-11-24_C8B_thiet_ke_D2_hang.md sections 2, 3, 5, CODE 24/11 khuya), input of c8b_a6_o --muc-dich d2, and the
per-run inputs of u_imu_dong_v3 (E_noi leaving the target run out, E_chuyen from leave-one-run-out lags). Written by
CODE_C8B local_a56ba287 on fix/c8b-n1-khoa-gop, 24/11/2026.

hang (one target run, every scored node of its grid folder):
  d_rad(k, node)  = wrap(dpsi_ma_k - (psi_IMU(t_k + D) - m_W)): the camera heading change of frame k from the
                    reference W (dpsi_ma of the scorer rows: one branch from psi_tc, the W combination) minus the IMU
                    yaw change over the same pair, m_W = trung_binh_go_vong of the IMU yaw at the W frames (the ids
                    of tham_chieu_v1, times t_cam + D), equal weights as the W combination. The IMU yaw is the
                    scorer's own (c7_score.imu_from with the constants' imu block); self-check: wrap(psi_tc - m_W)
                    equals the node's yaw_check.align_c_deg (else refused), every suspect frame's widened U_psi_k
                    equals |d| + U_IMU (else refused), the rows' command time t equals t_cam + D - t0.
  U_psi_rad       the camera heading bound of the frame: gate_v3.u_psi_deg where the heading is defined
                    (huong_xac_dinh 1), empty otherwise [CACH HIEU C8B: not U_psi_k_deg of the rows, which a suspect
                    frame widens to |d| + U_IMU: comparing d with it would pass by construction].
  U_imu_rad       U_IMU,D(k) of u_imu_dong_v3 (c8b_u_imu_v3 formula, the turn totals by prefix sums, checked
                    equal to c8b_u_imu_v3.u_imu_delta in the tests), omega / alpha from the camera yaw (frames.csv,
                    as the E_noi table), clock pairing u- = u+ = U_t_s of the node [CACH HIEU C8B: gio.json carries no
                    u_cham-/+; U_t bounds the camera-log time of a frame at that node, the safe side].
  x_m, y_m, psi_rad, u_mps  the lane pose and camera speed of the frame (rows x, y, psi, speed); a frame without a
                    finite pose has no cell: left out (counted), so c8b_luoi --mat-na-d2 masks it.
  luot, tap = dich, id = frame_id, nut, lan_lap_id (--lan-lap, L0 for the F8-empty sessions), controller (gio.json).
noi-loo: c8b_u_imu_v3.noi_table on every held-out capture once, then for each target run the band maxima over the
  OTHER captures (design 7b "khoi giu ngoai = luot C8 khac luot dang cham"; the max over captures splits, so this is
  the table c8b_u_imu_v3 noi would give without that run); one file per target, c8b_u_imu_noi_v1.
loo: the E_chuyen inputs: tau_hat of tre.json recomputed leaving each run out (median of the other runs' tau_s: tre
  fits tau per run, so this is the refit without that run); delta+-, r_dot of thang / troi kept (design note 3).
ghep: concatenate per-run row files (same columns) for c8b_a6_o.

  python tools/c8b_d2_hang.py hang --luot R --luoi-dir DIR --gio gio.json --frames frames.csv --esp32 log.csv \
      --hang-so hang_so.json --tham-chieu tham_chieu_v1.json --u-imu u_imu_dong_v3.json [--lan-lap L0] --out rows.csv
  python tools/c8b_d2_hang.py noi-loo --cap frames,log,offset_s,run [--cap ...] --dich R [--dich ...] \
      --thang thang.json --tre tre.json --out-dir DIR
  python tools/c8b_d2_hang.py loo --tre tre.json --thang thang.json --troi troi.json --out-dir DIR
  python tools/c8b_d2_hang.py ghep --rows a.csv --rows b.csv --out all.csv
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
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402
import c8b_u_imu_v3 as v3                                                # noqa: E402

VERSION = 'c8b_d2_hang_v1'
COLS = ('luot', 'tap', 'id', 'nut', 'x_m', 'y_m', 'psi_rad', 'u_mps', 'lan_lap_id', 'd_rad', 'U_psi_rad', 'U_imu_rad',
        'controller', 'ly_do')
TOL_C_RAD = 1e-9


class D2Error(ValueError):
    pass


def _wrap(a):
    return (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _read_json(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ── U_IMU,D with the turn totals by prefix sums ───────────────────────────────────────────────────

class TurnSums:
    """c8_u_imu.turn_totals(S, w0, w1) for many windows: per report pair its span [a, b], its signed step and its
    suspect extra (or the error turn_totals would raise there), summed over the pairs meeting the window with prefix
    sums when the span ends are non-decreasing (else the original function)."""

    def __init__(self, S):
        self.S = S
        ta, ps, na, tr = S['t_all'], S['psi_all'], S['n_all'], S['t_recv']
        keep = S.get('keep', np.ones(len(ta), dtype=bool))
        lo, hi = np.minimum(ta, tr), np.maximum(ta, tr)
        self.a, self.b = lo[:-1], hi[1:]
        d = np.diff(ps)
        tg, gz = S.get('tg', np.array([])), S.get('gz', np.array([]))
        per = S.get('period_fit_s', cu.PERIOD_S)
        extra = np.zeros(len(d))
        err = np.zeros(len(d), bool)
        for k in range(len(d)):
            sus = (na[k + 1] - na[k] > 1 or max(tr[k + 1] - tr[k], ta[k + 1] - ta[k]) > 1.5 * per
                   or not keep[k] or not keep[k + 1])
            if not sus:
                continue
            a, b = self.a[k], self.b[k]
            if tg.size < 2 or tg[0] > a or tg[-1] < b:
                err[k] = True
                continue
            j0, j1 = np.searchsorted(tg, a, 'right') - 1, np.searchsorted(tg, b, 'left')
            gaps = np.diff(tg[j0:j1 + 1])
            if not gaps.size or np.max(gaps) > S.get('h_lim', cu.H_LIM_S):     # turn_totals would raise there
                err[k] = True
                continue
            sel = (tg >= a - per) & (tg <= b + per)
            sel[j0:j1 + 1] = True
            extra[k] = float(np.max(np.abs(gz[sel]))) * (b - a)
        self.cP = np.concatenate([[0.0], np.cumsum(np.where(d > 0, d, 0.0) + extra)])
        self.cN = np.concatenate([[0.0], np.cumsum(np.where(d > 0, 0.0, -d) + extra)])
        self.cE = np.concatenate([[0], np.cumsum(err)])
        self.mono = bool(np.all(np.diff(self.a) >= 0) and np.all(np.diff(self.b) >= 0))
        self.t0, self.t1 = float(ta[0]), float(ta[-1])

    def __call__(self, w0, w1):
        if not self.mono:
            return cu.turn_totals(self.S, w0, w1)
        if self.t0 > w0 or self.t1 < w1:
            raise cu.InputError('IMU samples do not cover the window')
        i0 = int(np.searchsorted(self.b, w0, 'left'))          # first pair with b >= w0
        i1 = int(np.searchsorted(self.a, w1, 'right'))         # pairs with a <= w1: [0, i1)
        if i1 <= i0:
            return 0.0, 0.0
        if self.cE[i1] - self.cE[i0]:
            raise cu.InputError('suspect IMU reports in the window without usable gyro reports over the gap (CU-2)')
        return float(self.cP[i1] - self.cP[i0]), float(self.cN[i1] - self.cN[i0])


class UNode:
    """U_IMU,D(k) of one node: the W part (tre_W, noi_W, freshness of W) computed once."""

    def __init__(self, P3, S, t_W, omega_W, alpha_W, u_minus, u_plus, turns=None):
        self.P3, self.S = P3, S
        self.t_W = np.asarray(t_W, float)
        self.um, self.up = u_minus + P3['u_lag_s'], u_plus + P3['u_lag_s']
        self.turns = turns or TurnSums(S)
        self.why_W = []
        ew = [v3.e_tre(S, tj, self.um, self.up) for tj in self.t_W]
        self.tre_W = None if any(x is None for x in ew) else float(np.mean(ew))
        if self.tre_W is None:
            self.why_W.append('E_tre: IMU yaw not readable over the clock domain')
        nw = [v3._noi_lookup(P3, v3.band_key(o, a, v3.rv_gap(S, tj)))
              for o, a, tj in zip(omega_W, alpha_W, self.t_W)]
        self.noi_W = None if any(v is None for v, _ in nw) else float(np.mean([v for v, _ in nw]))
        if self.noi_W is None:
            self.why_W.append('E_noi(W): ' + '; '.join(sorted({w for v, w in nw if v is None})))
        self.tuoi_W = all(v3.tuoi_ok(S, tj, self.um, self.up) for tj in self.t_W)
        self.tb = float(np.mean(self.t_W))

    def __call__(self, t_k, omega_k, alpha_k):
        """(U or None, parts, reasons), as c8b_u_imu_v3.u_imu_delta."""
        P3, S = self.P3, self.S
        why = []
        try:
            PN = [self.turns(min(tj, t_k), max(tj, t_k)) for tj in self.t_W]
        except cu.InputError as e:
            return None, {}, [f'turn totals: {e}']
        Pb = float(np.mean([p for p, _ in PN]))
        Nb = float(np.mean([n for _, n in PN]))
        parts = dict(thang=P3['delta_plus'] * Pb + P3['delta_minus'] * Nb, troi=P3['r_dot'] * abs(t_k - self.tb),
                     q=2.0 * P3['q'])
        ek = v3.e_tre(S, t_k, self.um, self.up)
        if ek is None or self.tre_W is None:
            why.append('E_tre: IMU yaw not readable over the clock domain')
        else:
            parts['tre'], parts['tre_W'] = ek, self.tre_W
        nk, wk = v3._noi_lookup(P3, v3.band_key(omega_k, alpha_k, v3.rv_gap(S, t_k)))
        if nk is None:
            why.append(f'E_noi(k): {wk}')
        why += [w for w in self.why_W if w.startswith('E_noi')]
        if nk is not None and self.noi_W is not None:
            parts['noi'], parts['noi_W'] = nk, self.noi_W
        ch = P3['E_chuyen']
        dtau = v3.e_tre(S, t_k, ch['d_tau_s'], ch['d_tau_s'])
        if dtau is None:
            why.append('E_chuyen: IMU yaw not readable over the lag spread')
        else:
            parts['chuyen'] = (ch['d_delta_plus'] * Pb + ch['d_delta_minus'] * Nb + ch['d_r_dot'] * abs(t_k - self.tb)
                               + dtau)
        if not v3.tuoi_ok(S, t_k, self.um, self.up) or not self.tuoi_W:
            why.append(f'RV report older than {v3.TUOI_MAX_MS} ms (or not covered) over the clock domain')
        parts.update(P_bar=Pb, N_bar=Nb, t_bar_W=self.tb)
        if why:
            return None, parts, why
        return float(sum(parts[k] for k in ('thang', 'troi', 'q', 'tre', 'tre_W', 'noi', 'noi_W', 'chuyen'))), parts, []


# ── camera frames ────────────────────────────────────────────────────────────────────────────────

def read_cam(path):
    """Every frame idx -> t_s; omega, alpha of the camera yaw at the valid frames with the car seen (as the E_noi
    table: np.gradient of the unwrapped yaw over t_s)."""
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(k not in rows[0] for k in ('idx', 't_s', 'valid', 'car_seen', 'psi')):
        raise D2Error(f'{path}: frames.csv columns idx, t_s, valid, car_seen, psi needed')
    t_of = {int(r['idx']): float(r['t_s']) for r in rows}
    F = cu.read_frames(path)
    vidx = [int(r['idx']) for r in rows if r['valid'] == '1' and r['car_seen'] == '1' and r['psi'] not in ('', 'nan')]
    if len(vidx) != len(F['t']):
        raise D2Error(f'{path}: valid frames do not match c8_u_imu.read_frames (a bug)')
    om = np.gradient(F['psi'], F['t'])
    al = np.gradient(om, F['t'])
    return dict(t_of=t_of, om={i: float(o) for i, o in zip(vidx, om)}, al={i: float(a) for i, a in zip(vidx, al)})


# ── hang ─────────────────────────────────────────────────────────────────────────────────────────

def imu_samples(log_path, t0, imu):
    """The scorer's IMU samples (c7_score.imu_from on the constants' imu block) plus the report ages."""
    import c7_score
    L_full = cu.load_log(log_path)
    S, t_from = c7_score.imu_from(L_full, (t0 - imu['pre_s'] - c7_score.LOG_LEAD_S) * 1e6, t0 - imu['pre_s'], imu)
    L = c7_score.slice_log(L_full, t_from * 1e6)
    S['age_ms'] = L['age_r_ms'][S['rows']][S['keep']]
    return S


def check_params(imu, P3):
    """The U_IMU,D file must describe the yaw the scorer reads (same lag, scale correction, grid limits)."""
    bad = [k for k, j in (('tau_hat_s', 'tau_hat_s'), ('s_plus', 's_plus'), ('s_minus', 's_minus'),
                          ('period_s', 'period_s'), ('j_lim_s', 'j_lim_s'), ('h_lim_s', 'h_lim_s'))
           if not (math.isfinite(_f(imu.get(k))) and math.isfinite(_f(P3.get(j)))
                   and abs(_f(imu[k]) - _f(P3[j])) <= 1e-12)]
    if bad:
        raise D2Error(f'u_imu_dong_v3 and the constants imu block differ in {bad}: the bound would be for another yaw')


def hang(a):
    import c7_chuoi_khung as ck
    import c7_score
    hs = c7_score.load_hang_so(a.hang_so)
    imu = hs['imu']
    P3 = v3.load_v3(a.u_imu)
    check_params(imu, P3)
    if a.luot in ((P3.get('E_noi') or {}).get('luot') or []):
        raise D2Error(f'the E_noi table of {a.u_imu} holds run {a.luot}: not held out (design 7b)')
    gio = _read_json(a.gio)
    tc = _read_json(a.tham_chieu)
    if tc.get('schema') != 'tham_chieu_v1' or tc.get('trang_thai') != 'dat' or not tc.get('ids'):
        raise D2Error(f'{a.tham_chieu}: a tham_chieu_v1 file, trang_thai dat, with ids')
    W = [int(i) for i in tc['ids']]
    cam = read_cam(a.frames)
    miss = [i for i in W if i not in cam['t_of']]
    if miss:
        raise D2Error(f'W ids {miss[:5]} not in {a.frames}')
    sha = dict(frames=cu.sha256(a.frames), esp32=cu.sha256(a.esp32), gio=cu.sha256(a.gio),
               tham_chieu=cu.sha256(a.tham_chieu), hang_so=cu.sha256(a.hang_so), u_imu=cu.sha256(a.u_imu))
    S, turns, t0_seen = None, None, None
    out, nodes, n_left = [], [], 0
    pose_of = {}                       # frame id -> (x, y, psi, u) of the first node holding it (one frame, one pose)
    for k in range(len(gio.get('nodes_s') or [])):
        sp = os.path.join(a.luoi_dir, f'nut_{k:03d}.json')
        rp = os.path.join(a.luoi_dir, f'nut_{k:03d}_rows.csv')
        if not (os.path.isfile(sp) and os.path.isfile(rp)):
            nodes.append(dict(nut=k, dung=False, ly_do='no score or rows file'))
            continue
        sc = _read_json(sp)
        g = sc.get('gate_v3')
        if sc.get('outcome') == 'hong_he_do' or not isinstance(g, dict):
            nodes.append(dict(nut=k, dung=False, ly_do=f'node not scored ({sc.get("outcome")})'))
            continue
        if sc.get('frames_sha256') != sha['frames'] or sc.get('esp32_sha256') != sha['esp32'] \
                or (sc.get('gio_c8b') or {}).get('sha256') != sha['gio'] \
                or (sc.get('hang_so') or {}).get('sha256') != sha['hang_so']:
            raise D2Error(f'node {k}: scored on other frames / log / clock / constants files than given')
        ktd = sc.get('khung_tuong_doi_v1') or {}
        if ((ktd.get('nguon') or {}).get('tham_chieu_v1') or {}).get('sha256') != sha['tham_chieu']:
            raise D2Error(f'node {k}: scored with another tham_chieu_v1 than {a.tham_chieu}')
        D = _f(sc['gio_c8b'].get('D'))
        t0 = _f(sc.get('t0_log_us')) * 1e-6
        if not (math.isfinite(D) and math.isfinite(t0)):
            raise D2Error(f'node {k}: no D or t0_log_us')
        if t0_seen is None:
            t0_seen = t0
            S = imu_samples(a.esp32, t0, imu)
            turns = TurnSums(S)
        elif t0 != t0_seen:
            raise D2Error(f'node {k}: t0_log_us differs between nodes')
        yc = sc.get('yaw_check') or {}
        psi_tc = _f(ktd.get('psi_tc_rad'))
        t_W = np.array([cam['t_of'][i] for i in W]) + D
        pw = cu.interp_yaw(S, t_W)
        if not np.all(np.isfinite(pw)) or not yc.get('compared'):
            nodes.append(dict(nut=k, dung=False, ly_do='IMU yaw missing at a W frame or the yaw not compared'))
            continue
        m_W = ck.trung_binh_go_vong(pw)
        c = float(_wrap(psi_tc - m_W))
        c_sc = math.radians(_f(yc.get('align_c_deg')))
        if not abs(float(_wrap(c - c_sc))) <= TOL_C_RAD:
            raise D2Error(f'node {k}: wrap(psi_tc - m_W) {c!r} is not the scorer alignment {c_sc!r} (self-check)')
        u_imu_sc = math.radians(_f(yc.get('U_IMU_deg')))
        u_psi = math.radians(_f(g.get('u_psi_deg')))
        u_t = _f(sc.get('U_t_s'))
        if not (math.isfinite(u_psi) and math.isfinite(u_t) and u_t >= 0):
            raise D2Error(f'node {k}: gate u_psi_deg or U_t_s not finite')
        UN = UNode(P3, S, t_W, [cam['om'].get(i, math.nan) for i in W], [cam['al'].get(i, math.nan) for i in W],
                   u_t, u_t, turns)
        with open(rp, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        nk = dict(nut=k, dung=True, D_s=D, U_t_s=u_t, align_c_rad=c, n_hang=0, n_d=0, n_U=0)
        for r in rows:
            fid = _f(r.get('frame_id'))
            tcam = _f(r.get('t_cam_s'))
            x, y, psi, u = (_f(r.get(z)) for z in ('x', 'y', 'psi', 'speed'))
            if not (math.isfinite(fid) and all(math.isfinite(v) for v in (x, y, psi, u))):
                n_left += 1
                continue
            fid = int(fid)
            # real data 24/11: the scorer's speed of a frame differs between nodes at the 1e-13 level (its time base
            # moves with D); c8b_a6_o needs one pose per frame: the first node's values, the others within 1e-9
            p0 = pose_of.setdefault(fid, (x, y, psi, u))
            if max(abs(a_ - b_) for a_, b_ in zip(p0, (x, y, psi, u))) > 1e-9:
                raise D2Error(f'node {k} frame {fid}: pose / speed differ between nodes by more than 1e-9')
            x, y, psi, u = p0
            t_log = tcam + D
            if abs((t_log - t0) - _f(r.get('t'))) > 1e-6:
                raise D2Error(f'node {k} frame {fid}: rows t is not t_cam + D - t0')
            why = []
            pk = float(cu.interp_yaw(S, np.array([t_log]))[0])
            dm = _f(r.get('dpsi_ma'))
            d = float(_wrap(dm - (pk - m_W))) if math.isfinite(pk) and math.isfinite(dm) else None
            if d is None:
                why.append('no IMU yaw or camera heading change at the frame')
            if r.get('nghi_ngo_yaw') == '1' and d is not None:
                w = math.radians(_f(r.get('U_psi_k_deg')))
                if abs(w - (abs(d) + u_imu_sc)) > 1e-9 * max(1.0, w):
                    raise D2Error(f'node {k} frame {fid}: suspect U_psi_k is not |d| + U_IMU (self-check of d)')
            up_ = u_psi if r.get('huong_xac_dinh') == '1' else None
            if up_ is None:
                why.append('heading not defined at the frame')
            Ui, _, wu = UN(t_log, cam['om'].get(fid, math.nan), cam['al'].get(fid, math.nan))
            why += wu
            out.append(dict(luot=a.luot, tap='dich', id=fid, nut=k, x_m=repr(x), y_m=repr(y), psi_rad=repr(psi),
                            u_mps=repr(abs(u)), lan_lap_id=a.lan_lap, d_rad='' if d is None else repr(d),
                            U_psi_rad='' if up_ is None else repr(up_), U_imu_rad='' if Ui is None else repr(Ui),
                            controller=gio.get('controller') or '', ly_do='; '.join(why)))
            nk['n_hang'] += 1
            nk['n_d'] += d is not None
            nk['n_U'] += Ui is not None
        nodes.append(nk)
    if not out:
        raise D2Error('no row: no node of the grid could be used')
    return out, dict(version=VERSION, luot=a.luot, n_hang=len(out), n_khung_bo_vi_tri=n_left, nut=nodes, sha256=sha,
                     u_lag_s=P3['u_lag_s'], lan_lap_id=a.lan_lap,
                     cach_hieu=['U_psi_rad = gate_v3.u_psi_deg o khung co huong (khong lay U_psi_k da noi cua khung '
                                'nghi ngo)', 'u-/u+ = U_t_s cua nut (gio.json khong co u_cham-/+)'])


def write_rows(path, rows):
    cu.check_path(path)
    with open(path, 'x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(COLS))
        w.writeheader()
        w.writerows(rows)


# ── noi-loo, loo, ghep ───────────────────────────────────────────────────────────────────────────

def noi_loo(caps, dich, P):
    """Per target run the E_noi table over the captures of the OTHER runs: noi_table per capture, then band max."""
    per = {}
    for F, L, off, run in caps:
        if run in per:
            raise D2Error(f'capture of run {run} given twice')
        per[run] = v3.noi_table([(F, L, off, run)], P)
    out = {}
    for r in dich:
        if r not in per:
            raise D2Error(f'target run {r} has no capture: its table would not be "the others"')
        bang, runs = {}, []
        for run, tb in per.items():
            if run == r:
                continue
            runs += tb['luot']
            for key, e in tb['bang'].items():
                b = bang.setdefault(key, dict(max_rad=0.0, n=0, luot=[]))
                b['max_rad'] = max(b['max_rad'], e['max_rad'])
                b['n'] += e['n']
                b['luot'] = sorted(set(b['luot']) | set(e['luot']))
        one = next(iter(per.values()))
        out[r] = dict(version=v3.NOI_VERSION, bang=bang, luot=runs, dai=one['dai'], loai=one['loai'],
                      bo_luot=r)
    return out


def loo_files(tre, thang, troi):
    runs = tre.get('runs') or []
    taus = [_f(x.get('tau_s')) for x in runs]
    if len(taus) < 3 or not all(math.isfinite(t) for t in taus):
        raise D2Error('tre.json: at least three runs with tau_s (two leave-one-out fits)')
    if abs(float(np.median(taus)) - _f(tre.get('tau_hat_s'))) > 1e-12:
        raise D2Error('tre.json tau_hat_s is not the median of its runs tau_s')
    out = []
    for i, x in enumerate(runs):
        rest = taus[:i] + taus[i + 1:]
        out.append(dict(bo=dict(log=x.get('log'), log_sha256=x.get('log_sha256')),
                        tau_hat_s=float(np.median(rest)), delta_plus=thang['delta_plus'],
                        delta_minus=thang['delta_minus'], r_dot=troi['beta_max'],
                        cach='trung vi tau_s cac luot con lai (tre khop tau tung luot); thang, troi giu (ghi chu '
                             'thiet ke D2 muc 3)'))
    return out


def _script():
    c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_d2_hang.py', 'c8b_u_imu_v3.py',
                        'c8_u_imu.py', 'c7_score.py'], capture_output=True, text=True)
    return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()),
                luc_chay=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))


def _write_json(path, d):
    cu.check_path(path)
    with open(path, 'x', encoding='utf-8') as f:
        f.write(json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('hang')
    for k in ('--luot', '--luoi-dir', '--gio', '--frames', '--esp32', '--hang-so', '--tham-chieu', '--u-imu', '--out'):
        h.add_argument(k, required=True)
    h.add_argument('--lan-lap', default='L0')
    n = sub.add_parser('noi-loo')
    n.add_argument('--cap', action='append', required=True, help='frames.csv,log.csv,offset_s,run')
    n.add_argument('--dich', action='append', required=True)
    for k in ('--thang', '--tre', '--out-dir'):
        n.add_argument(k, required=True)
    lo = sub.add_parser('loo')
    for k in ('--tre', '--thang', '--troi', '--out-dir'):
        lo.add_argument(k, required=True)
    g = sub.add_parser('ghep')
    g.add_argument('--rows', action='append', required=True)
    g.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'hang':
            rows, meta = hang(a)
            write_rows(a.out, rows)
            meta.update(out=os.path.abspath(a.out), out_sha256=cu.sha256(a.out), script=_script())
            _write_json(os.path.splitext(a.out)[0] + '_meta.json', meta)
            print(f'hang {a.luot}: {len(rows)} rows, {meta["n_khung_bo_vi_tri"]} frames without a pose left out')
        elif a.cmd == 'noi-loo':
            th, tr = _read_json(a.thang), _read_json(a.tre)
            P = dict(tau_hat_s=tr['tau_hat_s'], s_plus=th['s_plus'], s_minus=th['s_minus'])
            caps, src = [], []
            for c in a.cap:
                parts = c.split(',')
                if len(parts) != 4:
                    raise D2Error(f'--cap {c!r}: frames,log,offset_s,run')
                caps.append((cu.read_frames(parts[0]), cu.load_log(parts[1]), float(parts[2]), parts[3]))
                src.append(dict(luot=parts[3], frames=os.path.abspath(parts[0]), frames_sha256=cu.sha256(parts[0]),
                                log=os.path.abspath(parts[1]), log_sha256=cu.sha256(parts[1]),
                                offset_s=float(parts[2])))
            os.makedirs(a.out_dir, exist_ok=True)
            for r, tb in noi_loo(caps, a.dich, P).items():
                tb.update(nguon=[x for x in src if x['luot'] != r],
                          tham_so=dict(thang=cu.sha256(a.thang), tre=cu.sha256(a.tre)), script=_script())
                _write_json(os.path.join(a.out_dir, f'noi_bo_{r}.json'), tb)
            print(f'noi-loo: {len(a.dich)} tables')
        elif a.cmd == 'loo':
            fs = loo_files(_read_json(a.tre), _read_json(a.thang), _read_json(a.troi))
            os.makedirs(a.out_dir, exist_ok=True)
            for i, d in enumerate(fs):
                d.update(nguon=dict(tre=cu.sha256(a.tre), thang=cu.sha256(a.thang), troi=cu.sha256(a.troi)))
                _write_json(os.path.join(a.out_dir, f'loo_{i:02d}.json'), d)
            print(f'loo: {len(fs)} files')
        else:
            allr, cols = [], None
            for p in a.rows:
                cu.check_path(p)
                with open(p, newline='', encoding='utf-8') as f:
                    rd = csv.DictReader(f)
                    if cols is None:
                        cols = rd.fieldnames
                    elif rd.fieldnames != cols:
                        raise D2Error(f'{p}: other columns')
                    allr += list(rd)
            write_rows(a.out, allr)
            print(f'ghep: {len(allr)} rows')
    except (D2Error, v3.U3Error, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_d2_hang: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
