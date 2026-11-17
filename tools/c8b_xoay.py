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
  du_m         = spread (largest |q_k - q_mean|, covers origin / far place and the two directions)
                 + 3 x the largest standard error of q_k (least squares, white-noise convention; the spread is the
                 main term)
  per interval: t0, t1, direction, turns, frames, q, se, drift v and |v| x duration, residual RMS / max
  imu (optional, --log PLACE=log.csv): camera heading minus IMU yaw (c8_u_imu.rv_samples) after a clock offset
  found by cross-correlating the turn rates, then a + b (psi_imu - psi_imu(t0)) removed per interval (IMU offset and
  scale): largest residual [deg]. A check only, not a bound and not in U (page v10 A3: turning in place sees only the
  heading-dependent part).
c8b_u.py takes q_tag_m and du_m: P2 = |(dx, dy)_scorer - R(dpsi_scorer) q_tag| + du_m + |q_tag| x A1 [rad].

  python tools/c8b_xoay.py --cho O=frames_O.csv --cho K1=frames_K1.csv [--log O=log_O.csv ...] --out xoay.json
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
    keep = []
    for r in rows:
        if r['valid'] != '1' or r['car_seen'] != '1':
            continue
        try:
            v = [float(r[c]) for c in ('t_s', 'tag_x', 'tag_y', 'psi', 'x', 'y')]
        except ValueError:
            continue
        if all(math.isfinite(x) for x in v):
            if abs(v[4] - v[1]) > OFFSET_TOL_M or abs(v[5] - v[2]) > OFFSET_TOL_M:
                raise XError(f'{path}: x, y differ from tag_x, tag_y at t_s {v[0]}: tracked with a tag_offset; '
                             f'track the clip with tag_offset 0')
            keep.append(v[:4])
    if len(keep) < 10:
        raise XError(f'{path}: fewer than 10 valid frames with the tag seen')
    a = np.array(keep)
    if np.any(np.diff(a[:, 0]) <= 0):
        raise XError(f'{path}: t_s not increasing')
    return dict(t=a[:, 0], x=a[:, 1], y=a[:, 2], psi=np.unwrap(a[:, 3]), path=path)


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


def fit_interval(F, i0, i1):
    t, x, y, psi = (F[k][i0:i1 + 1] for k in ('t', 'x', 'y', 'psi'))
    n = len(t)
    tau = t - t.mean()
    c, s = np.cos(psi), np.sin(psi)
    A = np.zeros((2 * n, 6))
    A[0::2, 0], A[0::2, 2], A[0::2, 4], A[0::2, 5] = 1, tau, c, -s
    A[1::2, 1], A[1::2, 3], A[1::2, 4], A[1::2, 5] = 1, tau, s, c
    b = np.empty(2 * n)
    b[0::2], b[1::2] = x, y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ sol
    dof = 2 * n - 6
    if dof <= 0:
        raise XError('too few frames in a turning interval')
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = math.sqrt(max(np.linalg.eigvalsh(cov[4:, 4:])))
    rr = np.hypot(r[0::2], r[1::2])
    dur = float(t[-1] - t[0])
    v = sol[2:4]
    return dict(t0=float(t[0]), t1=float(t[-1]), n=int(n), vong=float(abs(psi[-1] - psi[0]) / (2 * math.pi)),
                c0_m=[float(sol[0]), float(sol[1])], v_m_s=[float(v[0]), float(v[1])],
                troi_m=float(np.hypot(*v) * dur), q_m=[float(sol[4]), float(sol[5])], se_m=se,
                res_rms_m=float(np.sqrt(np.mean(rr ** 2))), res_max_m=float(rr.max()))


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


def analyse(places, logs=None):
    """places: {name: frames path}; logs: {name: log path}. Returns the c8b_xoay_v1 dict."""
    if len(places) < 2:
        raise XError('need at least two places (origin and a far place, page v10 P2)')
    res, qs, ses = {}, [], []
    for name, path in places.items():
        F = read_frames(path)
        segs, short = intervals(F)
        dirs = {s for (_, _, s) in segs}
        if dirs != {1, -1}:
            raise XError(f'place {name}: needs a turning interval of at least one full turn in each direction '
                         f'(has {sorted(dirs)}; shorter intervals: {short})')
        doan = []
        for (i0, i1, sg) in segs:
            d = dict(fit_interval(F, i0, i1), chieu=sg)
            d['troi_lon'] = d['troi_m'] > DRIFT_WARN_M
            doan.append(d)
            qs.append(d['q_m'])
            ses.append(d['se_m'])
        entry = dict(file=os.path.basename(path), sha256=cu.sha256(path), doan=doan, ngan=short)
        if logs and name in logs:
            entry['imu'] = imu_check(F, cu.load_log(logs[name]), segs)
        res[name] = entry
    Q = np.array(qs)
    qm = Q.mean(axis=0)
    spread = float(np.max(np.hypot(*(Q - qm).T)))
    se3 = 3.0 * float(max(ses))
    return dict(version=VERSION, q_tag_m=[float(qm[0]), float(qm[1])], du_m=spread + se3, spread_m=spread,
                se3_m=se3, n_doan=len(qs), cho=res,
                troi_lon=[f'{n}:{d["t0"]:.2f}' for n, e in res.items() for d in e['doan'] if d['troi_lon']],
                quy_uoc=dict(w_min=W_MIN, smooth_s=SMOOTH_S, gap_s=GAP_S, drift_warn_m=DRIFT_WARN_M),
                gioi_han=['turning centre = centre of the four wheels and of the IMU unless the wheels slip; a slip '
                          'repeated about an off-centre point is not seen (page v10 P2)',
                          'imu: a check only, not in U'],
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
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        places, logs = _pairs(a.cho, '--cho'), _pairs(a.log, '--log')
        if set(logs) - set(places):
            raise XError(f'--log for places without --cho: {sorted(set(logs) - set(places))}')
        cu.check_path(a.out)
        if os.path.exists(a.out):
            raise XError(f'{a.out} exists')
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
