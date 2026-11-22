#!/usr/bin/env python3
"""
c8b_xoay.py: tag-body centre offset (P2 of the C8b page v10) from the car turning in place, with an IMU check.
Written by CODE_C8B local_6db0bd09 on feat/c8b-u, 18/11/2026. Page v10 (approved, Liam 18/11 at CODE_C8B
local_8b35516c, "Duyet v10, Codex ra bo cham CAM (Recommended)"): CLAUDE_THESIS 03_CODE/docs/
2026-11-18_Trang_C8b_ngan_sach_khong_do_tay_v10.md, P2 and section 3; C8 guide part E (the car turns in place
slowly with the hand controller, about 2 turns one way, stop, 2 turns the other way, at O and at K1 (1; 0));
design: 03_CODE/docs/2026-11-18_Thiet_ke_C8b_o_U_v10_v2.md section 3.

Input per place: the frames CSV of c7_measure.track on the clip of that place, run WITH tag_offset 0 (columns t_s,
valid, car_seen, x, y, psi, tag_x, tag_y; with tag_offset 0, psi is the tag heading and x, y equal tag_x, tag_y, the
tag centre after the parallax step; a file whose x, y differ from tag_x, tag_y is refused. CODE_CAM 18/11: the frames
CSV has no tag_psi column).

Model (one turning interval = one place, one direction, at least one full turn):
  tag(t) = c0 + v (t - t_mid) + R(psi_tag(t)) q
  c0, v: centre of the turn and a slow linear drift of it (the car may creep while turning); q: the tag centre minus
  the turning centre in the TAG frame. Linear least squares in (c0, v, q) with psi_tag from the camera.
  By the design of Liam the turning centre is the centre of the four wheels and of the IMU, unless the wheels slip
  (page v10 P2: a slip repeated about an off-centre point still gives a clean circle; a stated limit).
  The body-frame offset (c7_measure tag_offset dx, dy) is R(dpsi) q, dpsi = tag heading minus body heading.

Outputs (c8b_xoay_v1):
  q_tag_m      mean of q over the intervals (at least 2 places, each with both directions)
  An interval is NOT used (C8BU-R2-2) when its quadratic centre path departs from its linear one by more than 1 mm,
  or its residual RMS (linear drift) exceeds 1 mm [DE XUAT]; each place must keep a usable interval in each direction,
  else refused (turn again there). A creep moving in step with the turn cannot be told from another q (page v10 P2):
  what passes these checks is a stated limit, not bounded.
  du_m         = spread + 3 x the largest standard error (least squares, white-noise convention, a minor term)
                 + the largest residual RMS of a used interval (motion outside the model leaves a residual).
                 spread = largest |q_est - q_mean| over EVERY estimate: per interval with a linear drift, per
                 interval with a quadratic drift, and per whole turn with a linear drift (C8BU-R1-2: a curved creep
                 biases q the same way in both directions, so the spread of the linear fits alone does not show it;
                 within one turn the creep is close to linear). It covers origin / far place and the two directions.
                 A creep that is neither close to quadratic over an interval nor close to linear over one turn is a
                 stated limit (the drift flag shows a large creep).
  per interval: t0, t1, direction, turns, frames, q, se, drift v and |v| x duration, residual RMS / max
  imu (optional, --log PLACE=log.csv): camera heading minus IMU yaw (c8_u_imu.rv_samples) after a clock offset
  found by cross-correlating the turn rates, then a + b (psi_imu - psi_imu(t0)) removed per interval (IMU offset and
  scale): largest residual [deg]. A check only, not a bound and not in U (page v10 A3: turning in place sees only the
  heading-dependent part).
c8b_u.py takes q_tag_m and du_m: P2 = |(dx, dy)_scorer - R(dpsi_scorer) q_tag| + du_m + |q_tag| x A1 [rad].

  python tools/c8b_xoay.py --cho goc=frames_goc.csv --cho K1=frames_K1.csv [--log goc=log_goc.csv ...] --out xoay.json
  (PLACE names as CHO = goc, K1, K2; trinh tu chuoi C8 muc 2.6: the old 'O' of this line was a docstring slip)
  The c8b_xoay_v1 file above is a DIAGNOSTIC before A6 (one q over all places), not the contract F4.

F4 xoay_v1 per place (contract 03_CODE/docs/2026-11-19_Hop_dong_du_lieu_C8_tuong_doi_v1.md row F4, 2.B; CODE_C8B
22/11, step T7b of the C8 chain: after c8b_a6 gop, before NHACTAM / HIEUCHUAN read --xoay): with --a6 the same model
is fitted per E place (goc, K1) on the frames of E_doan only (ids strictly between the one E_dau and the one E_cuoi
row of the F2 files for that frames file) with the heading psi' = psi - delta_psi_a6 (c8b_a6, F3). Per place:
q_tag_m, u_q_m (= du_m of the place: spread + 3 SE + residual as above, + |q| a6du_rad), r_floor_m (turning centre on
the floor) and u_r_m (an index), the two directions and their difference, cua_so_E, d_vet_m = |q| + u_q (check of the
E trace only, <= 0.020 m [DE XUAT]), trang_thai dat | chua_ket_luan with ly_do (A6 not dat for the plate, the E
window missing or twice, the session of the E clip not among the A6 sessions, frames without A6, a direction
missing). File: version = schema = 'xoay_v1', a6_sha256, tam_id, cho {goc, K1}; numbers before the margin.
  python tools/c8b_xoay.py --cho goc=frames_goc.csv --cho K1=frames_K1.csv --a6 c8b_a6_v2.json \
      --cua-so cua_so_v1.jsonl [--cua-so ...] --tam-id T [--log goc=log.csv ...] --out xoay_v1.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_xoay_v1'
W_MIN = 0.15          # rad/s: a frame belongs to a turning interval above this rate [DE XUAT]
SMOOTH_S = 0.5        # s: window of the rate estimate
GAP_S = 0.5           # s: an interval is not bridged over a gap of more frames than this
MIN_TURN = 2 * math.pi
DRIFT_WARN_M = 0.01   # m: drift of the centre over an interval above this is flagged (slip) [DE XUAT]
CURVE_MAX_M = 0.001   # m: an interval whose quadratic centre path departs from its linear one by more is not used
RES_MAX_M = 0.001     # m: an interval whose residual RMS (linear drift) is larger is not used (C8BU-R2-2) [DE XUAT]
GRID_S = 0.05         # s: grid of the cross-correlation
FRAME_NEED = ('t_s', 'valid', 'car_seen', 'x', 'y', 'psi', 'tag_x', 'tag_y')
OFFSET_TOL_M = 1e-6   # x, y must equal tag_x, tag_y: the clip was tracked with tag_offset 0


class XError(Exception):
    pass


def read_frames(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise XError(f'{path}: no frame')
    miss = [c for c in FRAME_NEED if c not in rows[0]]
    if miss:
        raise XError(f'{path}: no column {miss} (frames CSV of c7_measure.track)')
    # C8BU-R2-1: x = tag_x shows only dx = dy = 0; a dpsi of the tracking turns psi and so q. c7_measure.track writes
    # track.json (tag_offset, frames) next to frames.csv: it must say [0, 0, 0] and the same number of frames
    tj = os.path.join(os.path.dirname(os.path.abspath(path)), 'track.json')
    if not os.path.isfile(tj):
        raise XError(f'{path}: no track.json beside it (c7_measure.track output): the tag_offset of the tracking '
                     f'cannot be checked')
    with open(tj, encoding='utf-8') as f:
        info = json.load(f)
    off = info.get('tag_offset')
    if not isinstance(off, list) or len(off) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                         or x != 0 for x in off):
        raise XError(f'{tj}: tag_offset {off!r}: the clip must be tracked with tag_offset [0, 0, 0]')
    if info.get('frames') != len(rows):
        raise XError(f'{tj}: frames {info.get("frames")!r} but {path} has {len(rows)} rows (not its track.json)')
    if info.get('frames_sha256') != cu.sha256(path):          # C8BU-R3-1: another tracking of the same clip
        raise XError(f'{tj}: frames_sha256 is not the SHA-256 of {path} (not its track.json; a regate output has no '
                     f'tag_offset: use the track output)')
    keep, ids = [], []
    has_idx = 'idx' in rows[0]
    for r in rows:
        if r['valid'] != '1' or r['car_seen'] != '1':
            continue
        try:
            v = [float(r[c]) for c in ('t_s', 'tag_x', 'tag_y', 'psi', 'x', 'y')]
            k = int(r['idx']) if has_idx else -1
        except ValueError:
            continue
        if all(math.isfinite(x) for x in v):
            if abs(v[4] - v[1]) > OFFSET_TOL_M or abs(v[5] - v[2]) > OFFSET_TOL_M:
                raise XError(f'{path}: x, y differ from tag_x, tag_y at t_s {v[0]}: tracked with a tag_offset; '
                             f'track the clip with tag_offset 0')
            keep.append(v[:4])
            ids.append(k)
    if len(keep) < 10:
        raise XError(f'{path}: fewer than 10 valid frames with the tag seen')
    a = np.array(keep)
    if np.any(np.diff(a[:, 0]) <= 0):
        raise XError(f'{path}: t_s not increasing')
    return dict(t=a[:, 0], x=a[:, 1], y=a[:, 2], psi=np.unwrap(a[:, 3]), path=path,
                idx=np.array(ids, int) if has_idx else None, sha256=cu.sha256(path),
                video_sha256=info.get('video_sha256'), session_sha256=info.get('session_sha256'))


def turn_rate(t, psi, smooth_s=SMOOTH_S):
    """Rate by the slope of psi over +-smooth_s/2 (central difference on the nearest frames)."""
    lo = np.searchsorted(t, t - smooth_s / 2, 'left')
    hi = np.searchsorted(t, t + smooth_s / 2, 'right') - 1
    dt = t[hi] - t[lo]
    with np.errstate(invalid='ignore', divide='ignore'):
        w = np.where(dt > 0, (psi[hi] - psi[lo]) / dt, 0.0)
    return w


def intervals(F, w_min=W_MIN, gap_s=GAP_S, min_turn=MIN_TURN):
    """Turning intervals [(i0, i1, sign)] (frame indices, inclusive) with |rate| > w_min, one sign, no frame gap
    above gap_s, and at least min_turn of heading change. Shorter ones are dropped (listed in the second value)."""
    t, psi = F['t'], F['psi']
    w = turn_rate(t, psi)
    s = np.where(w > w_min, 1, np.where(w < -w_min, -1, 0))
    out, short, i, n = [], [], 0, len(t)
    while i < n:
        if s[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and s[j + 1] == s[i] and t[j + 1] - t[j] <= gap_s:
            j += 1
        turn = abs(psi[j] - psi[i])
        if turn >= min_turn:
            out.append((i, j, int(s[i])))
        else:
            short.append(dict(t0=float(t[i]), t1=float(t[j]), vong=turn / (2 * math.pi)))
        i = j + 1
    return out, short


def fit_interval(F, i0, i1, deg=1):
    """tag = sum_k c_k tau^k (k <= deg: the centre and its drift, polynomial in time) + R(psi) q; least squares."""
    t, x, y, psi = (F[k][i0:i1 + 1] for k in ('t', 'x', 'y', 'psi'))
    n = len(t)
    tau = t - t.mean()
    c, s = np.cos(psi), np.sin(psi)
    m = 2 * (deg + 1)
    A = np.zeros((2 * n, m + 2))
    for k in range(deg + 1):
        A[0::2, 2 * k], A[1::2, 2 * k + 1] = tau ** k, tau ** k
    A[0::2, m], A[0::2, m + 1] = c, -s
    A[1::2, m], A[1::2, m + 1] = s, c
    b = np.empty(2 * n)
    b[0::2], b[1::2] = x, y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ sol
    dof = 2 * n - (m + 2)
    if dof <= 0:
        raise XError('too few frames in a turning interval')
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = math.sqrt(max(np.linalg.eigvalsh(cov[m:, m:])))
    rr = np.hypot(r[0::2], r[1::2])
    dur = float(t[-1] - t[0])
    v = sol[2:4] if deg >= 1 else np.zeros(2)
    centre = sum(np.outer(tau ** k, sol[2 * k:2 * k + 2]) for k in range(deg + 1))   # fitted centre path
    return dict(t0=float(t[0]), t1=float(t[-1]), n=int(n), vong=float(abs(psi[-1] - psi[0]) / (2 * math.pi)),
                bac=deg, c0_m=[float(sol[0]), float(sol[1])], v_m_s=[float(v[0]), float(v[1])],
                troi_m=float(np.max(np.hypot(*(centre - centre[0]).T))), q_m=[float(sol[m]), float(sol[m + 1])],
                se_m=se, res_rms_m=float(np.sqrt(np.mean(rr ** 2))), res_max_m=float(rr.max()), _centre=centre)


def revolutions(F, i0, i1):
    """Index windows of each whole turn inside an interval (heading change 2 pi from the window start)."""
    psi = F['psi']
    out, a = [], i0
    while True:
        k = np.nonzero(np.abs(psi[a:i1 + 1] - psi[a]) >= 2 * math.pi)[0]
        if k.size == 0:
            return out
        b = a + int(k[0])
        out.append((a, b))
        a = b


def imu_check(F, L, segs):
    """Clock offset log - camera by cross-correlating the turn rates on a common grid, refined by the RMS of
    psi_cam - psi_imu - const over the turning frames; then per interval the largest |psi_cam - psi_imu - a - b dpsi|."""
    S = cu.rv_samples(L)
    tc, pc = F['t'], F['psi']
    g_c = np.arange(tc[0], tc[-1], GRID_S)
    g_i = np.arange(S['t'][0], S['t'][-1], GRID_S)
    if g_c.size < 20 or g_i.size < 20:
        raise XError('IMU check: clip or log shorter than 1 s')
    wc = np.gradient(np.interp(g_c, tc, pc), GRID_S)
    wi = np.gradient(np.interp(g_i, S['t'], S['psi']), GRID_S)
    nfft = 1 << int(math.ceil(math.log2(len(wc) + len(wi))))
    xc = np.fft.irfft(np.fft.rfft(wi - wi.mean(), nfft) * np.conj(np.fft.rfft(wc - wc.mean(), nfft)), nfft)
    k = int(np.argmax(xc))
    lag = k if k < nfft // 2 else k - nfft
    off0 = g_i[0] - g_c[0] + lag * GRID_S                    # log time = camera time + off
    turning = np.zeros(len(tc), dtype=bool)
    for (i0, i1, _) in segs:
        turning[i0:i1 + 1] = True
    best = None
    for off in np.arange(off0 - 2 * GRID_S, off0 + 2 * GRID_S + 5e-4, 1e-3):
        pi_ = cu.interp_yaw(S, tc + off)
        m = turning & np.isfinite(pi_)
        if m.sum() < 30:
            continue
        e = pc[m] - pi_[m]
        rms = float(np.sqrt(np.mean((e - e.mean()) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(off))
    if best is None:
        raise XError('IMU check: fewer than 30 turning frames with an IMU yaw at any offset')
    off = best[1]
    pi_ = cu.interp_yaw(S, tc + off)
    out = []
    for (i0, i1, _) in segs:
        sl = slice(i0, i1 + 1)
        # the ends of a turn (start, stop) are left out: a 20 Hz yaw interpolated across a sharp change of rate is
        # off by up to w dt / 4 there (0.6 deg at 0.9 rad/s), which is the IMU grid, not the camera
        m = np.isfinite(pi_[sl]) & (tc[sl] >= tc[i0] + SMOOTH_S) & (tc[sl] <= tc[i1] - SMOOTH_S)
        if m.sum() < 30:
            out.append(dict(t0=float(tc[i0]), du_max_deg=None, ly_do='fewer than 30 frames with an IMU yaw'))
            continue
        pi_s, pc_s = pi_[sl][m], pc[sl][m]
        X = np.column_stack([np.ones(m.sum()), pi_s - pi_s[0]])
        coef, *_ = np.linalg.lstsq(X, pc_s - pi_s, rcond=None)
        r = pc_s - pi_s - X @ coef
        out.append(dict(t0=float(tc[i0]), n=int(m.sum()), ty_le=float(coef[1]),
                        du_max_deg=float(np.degrees(np.abs(r).max())),
                        du_rms_deg=float(np.degrees(np.sqrt(np.mean(r ** 2))))))
    return dict(offset_s=off, rms_offset_deg=float(np.degrees(best[0])), doan=out, n_late=S['n_late'],
                log=os.path.basename(L['path']), log_sha256=cu.sha256(L['path']))


def _fit_place(F):
    """The turning intervals of one place and their fits (shared by analyse and theo_cho): doan (every interval),
    used (signs of the usable ones), qs / c0s (q, centre of the usable linear fits), alts / c0_alts (quadratic drift
    and per turn), ses, rms, troi (largest linear centre drift of a usable interval), short."""
    segs, short = intervals(F)
    out = dict(doan=[], used=[], qs=[], c0s=[], alts=[], c0_alts=[], ses=[], rms=[], troi=[], short=short, segs=segs)
    for (i0, i1, sg) in segs:
        d = dict(fit_interval(F, i0, i1), chieu=sg)
        d2 = fit_interval(F, i0, i1, deg=2)                 # C8BU-R1-2: a curved drift is not linear
        vong = [fit_interval(F, a, b) for (a, b) in revolutions(F, i0, i1)]   # drift ~linear within one turn
        # C8BU-R2-2: a creep moving in step with the turn cannot be told from another q (page v10 P2 limit), so
        # an interval whose centre path is visibly not linear, or whose residual is large, is not used
        cong = float(np.max(np.hypot(*(d2.pop('_centre') - d.pop('_centre')).T)))
        for v in vong:
            v.pop('_centre')
        d['bac2'] = dict(q_m=d2['q_m'], se_m=d2['se_m'], troi_m=d2['troi_m'], lech_tuyen_tinh_m=cong)
        d['tung_vong'] = [dict(t0=v['t0'], t1=v['t1'], q_m=v['q_m'], se_m=v['se_m']) for v in vong]
        d['troi_lon'] = max(d['troi_m'], d2['troi_m']) > DRIFT_WARN_M
        why = ([f'centre path curved by {cong * 1e3:.2f} mm > {CURVE_MAX_M * 1e3:.1f}'] if cong > CURVE_MAX_M
               else []) + ([f'residual RMS {d["res_rms_m"] * 1e3:.2f} mm > {RES_MAX_M * 1e3:.1f}']
                           if d['res_rms_m'] > RES_MAX_M else [])
        d['dung'], d['bo_vi'] = not why, why
        out['doan'].append(d)
        if why:
            continue
        out['used'].append(sg)
        out['qs'].append(d['q_m'])
        out['c0s'].append(d['c0_m'])
        out['rms'].append(d['res_rms_m'])
        out['troi'].append(d['troi_m'])
        out['alts'].extend([d2['q_m']] + [v['q_m'] for v in vong])
        out['c0_alts'].extend([d2['c0_m']] + [v['c0_m'] for v in vong])
        out['ses'].extend([d['se_m'], d2['se_m']] + [v['se_m'] for v in vong])
    return out


def analyse(places, logs=None):
    """places: {name: frames path}; logs: {name: log path}. Returns the c8b_xoay_v1 dict."""
    if len(places) < 2:
        raise XError('need at least two places (origin and a far place, page v10 P2)')
    res, qs, ses, alts, rms = {}, [], [], [], []
    for name, path in places.items():
        F = read_frames(path)
        P = _fit_place(F)
        doan, used, short = P['doan'], P['used'], P['short']
        qs += P['qs']
        alts += P['alts']
        ses += P['ses']
        rms += P['rms']
        if set(used) != {1, -1}:
            raise XError(f'place {name}: needs a usable turning interval of at least one full turn in each direction '
                         f'(usable {sorted(set(used))}; left out: {[(x["chieu"], x["bo_vi"]) for x in doan if not x["dung"]]}'
                         f'; shorter intervals: {short}): turn again there, slowly and evenly')
        entry = dict(file=os.path.basename(path), sha256=cu.sha256(path), doan=doan, ngan=short)
        if logs and name in logs:
            entry['imu'] = imu_check(F, cu.load_log(logs[name]), P['segs'])
        res[name] = entry
    Q = np.array(qs)
    qm = Q.mean(axis=0)
    E = np.array(qs + alts)                                  # every estimate: linear, quadratic drift, each turn
    spread = float(np.max(np.hypot(*(E - qm).T)))
    se3 = 3.0 * float(max(ses))
    res_term = float(max(rms))                               # motion outside the model leaves a residual (R2-2)
    return dict(version=VERSION, q_tag_m=[float(qm[0]), float(qm[1])], du_m=spread + se3 + res_term,
                spread_m=spread, se3_m=se3, res_m=res_term, n_doan=len(qs), n_uoc=len(E), cho=res,
                troi_lon=[f'{n}:{d["t0"]:.2f}' for n, e in res.items() for d in e['doan'] if d['troi_lon']],
                quy_uoc=dict(w_min=W_MIN, smooth_s=SMOOTH_S, gap_s=GAP_S, drift_warn_m=DRIFT_WARN_M,
                             curve_max_m=CURVE_MAX_M, res_max_m=RES_MAX_M),
                gioi_han=['turning centre = centre of the four wheels and of the IMU unless the wheels slip; a slip '
                          'repeated about an off-centre point is not seen (page v10 P2)',
                          'a creep of the centre moving in step with the turn cannot be told from another q; intervals '
                          'with a visibly curved centre path or a large residual are left out, the rest is not bounded '
                          '(C8BU-R2-2)',
                          'imu: a check only, not in U'],
                written_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))


CHO = ('goc', 'K1', 'K2')
TRUC_XE_OK = ('dat', 'khoa')          # truc_xe_v1 (F7, HIEUCHUAN c8_hc_a) writes 'khoa' for a usable place


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _vec2(v):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    a, b = _num(v[0]), _num(v[1])
    return None if a is None or b is None else np.array([a, b])


def ghep_q(lan_lap, cho, xoay, truc_xe=None, u_psi_w_rad=None):
    """F16, way (a) chosen by CODE 19/11 (data contract 03_CODE/docs/2026-11-19_Hop_dong_du_lieu_C8_tuong_doi_v1.md;
    design C8B v7 section 1 step 4): turning centre -> tag centre in the tag frame of the run, for a run on another
    mounting of the plate than the clip E.
      q_B   = R(theta_chuan_rad) q_tag          (q_tag, u_q from xoay_v1 at the place, fitted on psi' after A6;
                                                 theta_chuan_rad: tag minus anchor heading at the reference window, F8)
      q_m   = R(-theta_luot_rad) (q_B + delta_c_B_m)          (theta_luot_rad: same at the run window, F8)
      u_q_m = u_q + u_delta_c_B_m + (|q_m| + u_q) u_theta,   u_theta = u_a_k (F7, deg -> rad) + u_psi_w_rad
    u_psi_w_rad: noise of the mean tag heading over the run window (the caller's W frames). Before the 1.25 margin.
    Limit LL4-6: a heading error of the tag common to a whole clip has no index of its own.
    Any input missing, not finite or not 'dat' -> trang_thai 'chua_ket_luan' (the scorer leaves the position through
    q undecided), never a guess."""
    why = []
    if cho not in CHO:
        raise ValueError(f'cho {cho!r} not in {CHO}')
    L = lan_lap or {}
    if L.get('cho') not in (None, cho):
        why.append(f'lan_lap is for {L.get("cho")!r}, not {cho!r}')
    th_l, th_c = _num(L.get('theta_luot_rad')), _num(L.get('theta_chuan_rad'))
    if th_l is None:
        why.append('lan_lap: theta_luot_rad missing')
    if th_c is None:
        why.append('lan_lap: theta_chuan_rad missing')
    # NHACTAM e7b8428 / d1f6843 (F8, CODE 19/11): a run has a standard only when its placement is VERIFIED against the
    # 12b of its place (khong_doi -> chuan_12b, lan_lap_id); tam_doi_so_chuan, lan_lap_khong_xac_dinh, lan_lap_sau_doi
    # (after a valid doi until the next 12b) and the others leave lan_lap_id null: no q for that run
    if ('lan_lap_id' in L or 'chuan_12b' in L) and (L.get('lan_lap_id') is None or L.get('chuan_12b') is None):
        why.append(f'lan_lap: no verified 12b standard for this run (ly_do {L.get("ly_do")!r})')
    st = L.get('trang_thai') if isinstance(L.get('trang_thai'), dict) else {}
    dcb, u_dcb = _vec2(L.get('delta_c_B_m')), _num(L.get('u_delta_c_B_m'))
    if st.get('delta_c_B') != 'dat' or dcb is None or u_dcb is None:
        why.append(f'lan_lap: delta_c_B not dat ({st.get("delta_c_B")!r}) or missing')
    X = xoay or {}
    c = (X.get('cho') or {}).get(cho) if isinstance(X.get('cho'), dict) else None
    if X.get('version') != 'xoay_v1' or not isinstance(c, dict):
        why.append(f'xoay: not xoay_v1 with place {cho!r} (version {X.get("version")!r}; the q of c8b_xoay_v1 is fitted '
                   f'on the tag heading without A6)')
        c = {}
    q_tag, u_q = _vec2(c.get('q_tag_m')), _num(c.get('u_q_m'))
    if c and (c.get('trang_thai') != 'dat' or q_tag is None or u_q is None):
        why.append(f'xoay {cho}: q_tag_m/u_q_m missing or trang_thai {c.get("trang_thai")!r}')
    T = truc_xe or {}
    u_a = _num(T.get('u_a_k_deg'))
    if T.get('schema') != 'truc_xe_v1' or T.get('cho') != cho or T.get('trang_thai') not in TRUC_XE_OK or u_a is None:
        why.append(f'truc_xe: no usable truc_xe_v1 for {cho!r} (u_a_k_deg)')
    u_w = _num(u_psi_w_rad)
    if u_w is None or u_w < 0:
        why.append('u_psi_w_rad (noise of the mean tag heading over the run window) missing')
    if why:
        return dict(q_m=None, u_q_m=None, u_theta_rad=None, trang_thai='chua_ket_luan', ly_do=why)

    def R(a):
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q_b = R(th_c) @ q_tag
    q = R(-th_l) @ (q_b + dcb)
    u_th = math.radians(u_a) + u_w
    u = u_q + u_dcb + (float(np.hypot(*q)) + u_q) * u_th
    return dict(q_m=[float(q[0]), float(q[1])], u_q_m=float(u), u_theta_rad=float(u_th), q_B_m=[float(q_b[0]),
                float(q_b[1])], trang_thai='dat', ly_do=[],
                gioi_han=['LL4-6: sai yaw ma chung ca clip khong co chi so rieng'])


# ── F4 xoay_v1 per place (contract 03_CODE/docs/2026-11-19_Hop_dong_du_lieu_C8_tuong_doi_v1.md row F4, section 2.B;
#    CODE_C8B 22/11: no tool wrote it at 6c14b4f, the readers CAM c7_chuoi_khung.doc_xoay, NHACTAM c7_nhac_tam,
#    HIEUCHUAN c8_hc_a / c8_hc_vet and ghep_q above waited for it) ─────────────────────────────────────────────────

VERSION_F4 = 'xoay_v1'
CHO_E = ('goc', 'K1')          # the E clips (turning in place) are at goc and K1 only (C8 guide part E, design C8B v7 1)
D_VET_MAX_M = 0.020            # contract F4: d for the E trace <= 20 mm [DE XUAT], a check of the trace only, not in U


def read_cua_so(paths):
    """cua_so_v1 rows (JSON lines; F2 of HIEUCHUAN c8_hc_a cua_so or the joined file of CAM) -> list of dicts."""
    rows = []
    for p in paths or []:
        cu.check_path(p)
        with open(p, encoding='utf-8') as f:
            for k, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError as exc:
                    raise XError(f'{p}:{k}: not JSON ({exc})')
                if r.get('schema') != 'cua_so_v1':
                    raise XError(f'{p}:{k}: schema {r.get("schema")!r} is not cua_so_v1')
                rows.append(dict(r, _file=os.path.abspath(p), _sha256=cu.sha256(p)))
    return rows


def e_window(rows, cho, F):
    """The E window of the clip of F at place cho from the F2 rows: the ONE E_dau and the ONE E_cuoi row of that place
    whose frames_sha256 is the SHA-256 of the frames file (so the same tracking of the same clip); E_doan = the frame
    ids strictly between the last E_dau frame and the first E_cuoi frame. Returns (dict or None, reasons)."""
    why = []
    mine = [r for r in rows if r.get('cho') == cho and r.get('loai') in ('E_dau', 'E_cuoi')
            and r.get('frames_sha256') == F['sha256']]
    by = {k: [r for r in mine if r['loai'] == k] for k in ('E_dau', 'E_cuoi')}
    for k, v in by.items():
        if len(v) != 1:
            why.append(f'{len(v)} {k} rows of place {cho} for frames {F["sha256"][:12]} in the cua_so_v1 files (need 1)')
    if why:
        return None, why
    ed, ec = by['E_dau'][0], by['E_cuoi'][0]
    try:
        a = max(int(k['id']) for k in ed['khung'])
        b = min(int(k['id']) for k in ec['khung'])
    except (KeyError, TypeError, ValueError):
        return None, ['E_dau / E_cuoi khung without frame ids']
    if not a < b:
        return None, [f'E_dau ends at frame {a}, not before E_cuoi starts at {b}']
    for r in (ed, ec):
        if F.get('video_sha256') and r.get('clip_sha256') != F['video_sha256']:
            why.append(f'{r["loai"]} clip_sha256 is not the video of track.json')
        if F.get('session_sha256') and r.get('session_sha256') != F['session_sha256']:
            why.append(f'{r["loai"]} session_sha256 is not the session of track.json')
    if why:
        return None, why
    return dict(clip=ed.get('clip'), clip_sha256=ed.get('clip_sha256'), session_sha256=ed.get('session_sha256'),
                luot_id_E=ed.get('luot_id'), E_dau=dict(id=[int(ed['khung'][0]['id']), a], chi_so=ed.get('chi_so')),
                E_cuoi=dict(id=[b, int(ec['khung'][-1]['id'])], chi_so=ec.get('chi_so')), E_doan=[a + 1, b - 1],
                cua_so_v1=[dict(file=r.get('_file'), sha256=r.get('_sha256'))
                           for r in ((ed,) if ed.get('_file') == ec.get('_file') else (ed, ec))]), []


def _sub(F, keep):
    return dict(F, t=F['t'][keep], x=F['x'][keep], y=F['y'][keep], psi=F['psi'][keep],
                idx=None if F.get('idx') is None else F['idx'][keep])


def theo_cho_place(cho, F, rows, a6, tam_id):
    """F4 block of one place: the E frames (E_doan), heading psi' = psi - delta_psi_a6 (c8b_a6 contract F3), the same
    interval model and checks as analyse, per place. Never raises for the data: a place that cannot be concluded is
    'chua_ket_luan' with its reasons (contract F4 trang_thai dat | chua_ket_luan + ly_do)."""
    import c8b_a6 as A6
    why = []
    out = dict(trang_thai='chua_ket_luan', ly_do=why, q_tag_m=None, u_q_m=None, du_m=None, r_floor_m=None,
               u_r_m=None, d_vet_m=None, frames=os.path.basename(F['path']), frames_sha256=F['sha256'])
    st, lw = A6.ap_dung(a6, tam_id)
    if st != 'dat':
        why.append(f'A6: {lw}')
    if F.get('idx') is None:
        why.append('frames.csv without the idx column: the E window cannot be applied')
        return out
    win, ww = e_window(rows, cho, F)
    why += ww
    if win is None:
        return out
    out['cua_so_E'] = win
    ses_a6 = [s.get('session_sha256') for s in (a6 or {}).get('session') or []]
    if ses_a6 and win['session_sha256'] not in ses_a6:          # da session (A): e_bc of the A6 file covers it
        why.append('session of the E clip is not among the sessions of the A6 file (e_bc not taken there)')
    keep = (F['idx'] >= win['E_doan'][0]) & (F['idx'] <= win['E_doan'][1])
    if keep.sum() < 10:
        why.append(f'{int(keep.sum())} valid frames in E_doan')
        return out
    E = _sub(F, keep)
    if st != 'dat':
        return out
    xy = np.column_stack([E['x'], E['y']])
    dpsi, ok, wk = A6.delta_psi_a6(a6, xy, E['psi'], a6['plumb_xy_m'], a6['H_m'])
    out['n_khung_E'], out['n_khung_a6_khong_dat'] = int(len(ok)), int((~ok).sum())
    if (~ok).any():
        why.append(f'{int((~ok).sum())} E frames without an A6 correction ({sorted(set(w for w in wk if w))})')
        return out
    Ep = dict(E, psi=E['psi'] - dpsi)                          # psi' (contract F3: SUBTRACT from the car_pose heading)
    P = _fit_place(Ep)
    out.update(doan=P['doan'], ngan=P['short'])
    if set(P['used']) != {1, -1}:
        why.append(f'needs a usable turning interval of at least one full turn in each direction in E_doan (usable '
                   f'{sorted(set(P["used"]))}; left out {[(x["chieu"], x["bo_vi"]) for x in P["doan"] if not x["dung"]]})')
        return out
    Q, C0 = np.array(P['qs']), np.array(P['c0s'])
    qm, cm = Q.mean(axis=0), C0.mean(axis=0)
    Eq = np.array(P['qs'] + P['alts'])
    spread = float(np.max(np.hypot(*(Eq - qm).T)))
    se3 = 3.0 * float(max(P['ses']))
    res_t = float(max(P['rms']))
    rho = float(np.max(np.hypot(*(xy - np.asarray(a6['plumb_xy_m'], float)).T)))
    a6du = A6.a6du_rad(a6, rho)
    if a6du is None:
        why.append(f'A6 residual bound a6du_rad not available at rho {rho:.3f} m')
        return out
    u_a6 = float(np.hypot(*qm)) * a6du                        # a heading error of psi' moves R(psi') q by |q| a6du
    u_q = spread + se3 + res_t + u_a6
    Ec = np.array(P['c0s'] + P['c0_alts'])
    spread_r = float(np.max(np.hypot(*(Ec - cm).T)))
    u_r = spread_r + float(max(P['troi'])) + res_t
    sg = {s: np.array([q for q, x in zip(P['qs'], [d for d in P['doan'] if d['dung']]) if x['chieu'] == s]).mean(axis=0)
          for s in (1, -1)}
    d_vet = float(np.hypot(*qm)) + u_q
    out.update(q_tag_m=[float(qm[0]), float(qm[1])], u_q_m=u_q, du_m=u_q, r_floor_m=[float(cm[0]), float(cm[1])],
               u_r_m=u_r, d_vet_m=d_vet, d_vet_dat=bool(d_vet <= D_VET_MAX_M),
               hai_chieu={'+': [float(v) for v in sg[1]], '-': [float(v) for v in sg[-1]],
                          'lech_m': float(np.hypot(*(sg[1] - sg[-1])))},
               u_q_phan=dict(spread_m=spread, se3_m=se3, res_m=res_t, a6_m=u_a6, a6du_rad=a6du, rho_max_m=rho),
               u_r_phan=dict(spread_m=spread_r, troi_m=float(max(P['troi'])), res_m=res_t),
               n_doan=len(P['qs']), n_uoc=len(Eq), a6_sha256=a6.get('_sha256'))
    if not why:
        out['trang_thai'] = 'dat'
    return out


def theo_cho(places, a6, rows, tam_id, logs=None):
    """F4 xoay_v1: {version, schema, a6_sha256, tam_id, cho: {goc, K1: block}}; every number before the margin."""
    bad = sorted(set(places) - set(CHO_E))
    if bad:
        raise XError(f'places {bad}: F4 xoay_v1 is per E place {CHO_E}')
    if not isinstance(a6, dict) or not a6.get('_sha256'):
        raise XError('the A6 file (c8b_a6.load_a6) is needed: q is fitted on psi\' after A6 (contract F4)')
    cho = {}
    for name, path in places.items():
        F = read_frames(path)
        b = theo_cho_place(name, F, rows, a6, tam_id)
        if logs and name in logs and b.get('trang_thai') == 'dat':
            keep = (F['idx'] >= b['cua_so_E']['E_doan'][0]) & (F['idx'] <= b['cua_so_E']['E_doan'][1])
            E = _sub(F, keep)
            b['imu'] = imu_check(E, cu.load_log(logs[name]), intervals(E)[0])   # heading before A6: a check only
        cho[name] = b
    return dict(version=VERSION_F4, schema=VERSION_F4, a6_sha256=a6['_sha256'], tam_id=tam_id, cho=cho,
                truoc_le=True, d_vet_max_m=D_VET_MAX_M,
                quy_uoc=dict(w_min=W_MIN, smooth_s=SMOOTH_S, gap_s=GAP_S, drift_warn_m=DRIFT_WARN_M,
                             curve_max_m=CURVE_MAX_M, res_max_m=RES_MAX_M,
                             u_q='spread (linear, quadratic drift, each turn) + 3 x largest SE + largest residual RMS '
                                 '+ |q| x a6du_rad (page v10 P2 per place, plus the A6 residual)',
                             u_r='spread of the centre estimates + largest linear centre drift + largest residual RMS '
                                 '[DE XUAT], an index',
                             d_vet='|q| + u_q (tag centre to turning centre; the turning centre is taken as the '
                                   'nominal body centre, page v10 P2)'),
                gioi_han=['turning centre = centre of the four wheels and of the IMU unless the wheels slip; a slip '
                          'repeated about an off-centre point is not seen (page v10 P2)',
                          'a creep of the centre moving in step with the turn cannot be told from another q (C8BU-R2-2)',
                          'u_r_m is an index of the centre on the floor, not a bound',
                          'imu: a check only, on the heading before A6, not in U'],
                written_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))


def script_state():
    """Commit of the repo and whether this script or c8_u_imu.py has local changes."""
    import subprocess
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_xoay.py', 'c8_u_imu.py'],
                           capture_output=True, text=True, timeout=10)
        ok = c.returncode == 0 and s.returncode == 0
        return dict(commit=c.stdout.strip() if ok else '', dirty=bool(s.stdout.strip()) if ok else None,
                    file='tools/c8b_xoay.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_xoay.py')


def _pairs(items, what):
    out = {}
    for s in items or []:
        k, sep, p = s.partition('=')
        if not sep or not k or not p:
            raise XError(f'{what} {s!r}: NAME=path')
        if k in out:
            raise XError(f'{what} {k} twice')
        out[k] = p
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cho', action='append', required=True, help='PLACE=frames.csv (tag_offset 0)')
    ap.add_argument('--log', action='append', help='PLACE=agv_logger csv (IMU check)')
    ap.add_argument('--a6', default=None, help="c8b_a6_v2 file: write the F4 xoay_v1 per place (q on psi' after A6)")
    ap.add_argument('--cua-so', action='append', default=[], help='cua_so_v1 (F2) with the E_dau / E_cuoi rows (F4)')
    ap.add_argument('--tam-id', default=None, help='plate of the runs (F4: A6 must be dat for it)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        places, logs = _pairs(a.cho, '--cho'), _pairs(a.log, '--log')
        if set(logs) - set(places):
            raise XError(f'--log for places without --cho: {sorted(set(logs) - set(places))}')
        cu.check_path(a.out)
        if os.path.exists(a.out):
            raise XError(f'{a.out} exists')
        if a.a6:
            import c8b_a6 as A6
            if not a.cua_so or not a.tam_id:
                raise XError('F4 (--a6) needs --cua-so and --tam-id')
            try:
                a6 = A6.load_a6(a.a6)
            except A6.A6Error as exc:
                raise XError(f'--a6: {exc}')
            d = theo_cho(places, a6, read_cua_so(a.cua_so), a.tam_id, logs)
            d['nguon'] = ([dict(vai='a6', file=os.path.abspath(a.a6), sha256=a6['_sha256'])]
                          + [dict(vai='cua_so_v1', file=os.path.abspath(p), sha256=cu.sha256(p)) for p in a.cua_so]
                          + [dict(vai=f'frames {k}', file=os.path.abspath(v), sha256=cu.sha256(v))
                             for k, v in places.items()])
            d['script'] = script_state()
            with open(a.out, 'x', encoding='utf-8') as f:
                json.dump(d, f, indent=1, ensure_ascii=False, allow_nan=False)
            for k, b_ in d['cho'].items():
                q = b_['q_tag_m']
                print(f'{k}: {b_["trang_thai"]}' + (f' q_tag ({q[0] * 1e3:.2f}, {q[1] * 1e3:.2f}) mm, u_q '
                                                    f'{b_["u_q_m"] * 1e3:.2f} mm, d_vet {b_["d_vet_m"] * 1e3:.2f} mm'
                                                    if q else '') + (f'; {"; ".join(b_["ly_do"])}' if b_['ly_do'] else ''))
            print('#ket_qua,ma=0')
            return 0
        d = analyse(places, logs)
        d['script'] = script_state()
        with open(a.out, 'x', encoding='utf-8') as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
        print(f'q_tag = ({d["q_tag_m"][0] * 1e3:.2f}, {d["q_tag_m"][1] * 1e3:.2f}) mm, du = {d["du_m"] * 1e3:.2f} mm, '
              f'{d["n_doan"]} intervals; drift flagged: {d["troi_lon"] or "none"}')
        print('#ket_qua,ma=0')
        return 0
    except (XError, cu.InputError, OSError) as exc:
        print(f'error: {exc}')
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    sys.exit(main())
