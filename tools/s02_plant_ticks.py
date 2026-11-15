"""
s02_plant_ticks.py: spec S02 Viec CODE 3 on HIL: the plant state of the Pi frame log brought onto the ESP32
clock (every metric tick), by matching the ESP32 encoder counts with the wheel angle of the plant.

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van) section 10.2 (a); Liam 15/11, CODE_S02
local_ff3bd6e3, "Khop so dem encoder (Recommended)"; spec S02 v1.10 P2-2b (plant on the DUT clock, residual >
2 counts RMS [ĐỀ XUẤT]: no family R for the run).

    python tools/s02_plant_ticks.py --frame-log <run.npz> --esp-log <run.csv> --test A-L2 --out <run>_plant.csv
    -> <run>_plant.csv (PLANT_COLUMNS of fwsim/s02_groupA_sim.py, one row per motor-on ESP32 line inside the
       frame log, empty outside) and <run>_plant_map.json; last line '#ket_qua,ma=0 | 1 (not mapped) | 2 (error)'

Inputs. Frame log of run_hil.py --frame-log: t_send_ns (Pi CLOCK_MONOTONIC right before each transfer,
hil_node.py), mosi (mosi_true when the run injected faults: the true plant), decoded as protocol.decode_state
(omega1-4, x, y, theta, world vx, vy, wz; hil_o1_score.decode_frames). The state of frame k is placed at its
send time (the H7 applies its wheel speeds right after the receive, RS-4). ESP32 log of agv_logger.py, read by
s02_metrics.load_log (every check of the logger output): fl_count..rr_count read at the start of each tick
(AGV main.c 232; int32, cumulative), timestamp_us.

Map t_Pi = alpha + beta t_ESP (seconds), per run.
  Expected count of wheel w at Pi time t: C_w(t) = 1320 / (2 pi) x the integral of the frame wheel speed, held
  from each frame to the next (the EncoderGen rate; wheels w = MOSI order = fl, fr, rl, rr, sign as
  tools/hil_edge_map.py). Used ticks: wheel w turning one way at >= OMEGA_FIT rad/s at the tick, inside a
  segment of one sign, SEG_SKIP_S after its start (start events, OMEGA_MIN of the H7), so each (wheel,
  segment) has one unknown count offset (least squares).
  Start value: the command anchors, first frame with a non-zero command (Pi) and anchor a of
  s02_cmd_path (ESP32), lag 0..30 ms, slope 1 + the anchor drift. Offset found by a grid at that slope
  (GRID1 over [-DELTA_SPAN_S, +DELTA_SPAN_S], then GRID2), then offset and slope together by Gauss-Newton
  least squares on every used tick (offsets per (wheel, segment) projected out): the speed changes (starts,
  stops, steps, ripple) fix the offset, a wrong slope leaves a ramp of count error over a segment held at
  speed. |beta - 1| <= 2002 ppm (S05 4.7 (i)) else not mapped; fewer than MIN_TICKS used ticks: not mapped.
  Then the residual RMS (counts) over every used tick: > RESID_MAX [ĐỀ XUẤT 2] -> not mapped (family R of
  the run undefined). A low residual does not show the map is fixed (S2P1-1, Codex round 1): with only
  steady-speed ticks a time shift is a count constant, taken by the segment offsets. So the fit must
  converge with a well-conditioned normal matrix, and the standard error of the mapped time (from that
  matrix and the residual variance, largest at an end of the run) must be <= SIGMA_T_MAX_S [ĐỀ XUẤT 1 ms];
  else not mapped (not_identified, sigma_t_over_max). Precision about 1 count / (1320 / 2 pi x |omega|): 0.5-1.2 ms at 4-10
  rad/s [ƯỚC LƯỢNG]; the sensor path (Pi send -> H7 receive -> EncoderGen) adds well under 1 ms.
Plant at a tick: linear interpolation in Pi time between the two frames around alpha + beta t_ESP (theta
unwrapped first, wrapped back to (-pi, pi]); outside the frame log: empty (no extrapolation, P2-3).
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s02_cmd_path  # noqa: E402

COUNTS_PER_REV = 1320            # 4 x ENC_PPR 330 (hil_edge_map, encoder.h PPR 1320)
WHEELS = ('fl_count', 'fr_count', 'rl_count', 'rr_count')
OMEGA_FIT = 1.0                  # rad/s: ticks used for the fit (EncoderGen OMEGA_MIN 0.3)
SEG_SKIP_S = 0.05                # s dropped at the start of each one-sign segment
MIN_TICKS = 50                   # used (tick, wheel) pairs
DELTA_SPAN_S = 0.060             # search half width around the anchor start value
GRID1_S, GRID2_S = 2e-4, 1e-5
GN_ITER, GN_TOL_S = 20, 1e-7     # Gauss-Newton passes, stop when the move is under this
COND_MAX = 1e12                  # normal matrix condition number above which delta / beta are not identified
SIGMA_T_MAX_S = 1e-3             # [ĐỀ XUẤT] standard error of the mapped time (S2P1-1, Codex vong 1)
RESID_MAX = 2.0                  # [ĐỀ XUẤT] counts RMS (S02 v1.10)
PPM_MAX = 2002.0                 # S05 4.7 (i)
ANCHOR_LAG_S = 0.015             # middle of the 0..30 ms command anchor lag (s02_cmd_path)
PLANT_COLUMNS = ['timestamp_us', 'x', 'y', 'theta', 'vx_w', 'vy_w', 'vx_b', 'vy_b', 'wz', 'w1', 'w2', 'w3', 'w4']


def load_frames(path):
    """(t_Pi [s], state (n, 10), ref (n, 3), manifest) of a run_hil.py frame log; mosi_true when present."""
    import hil_o1_score as H
    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(str(z['manifest']))
        t = np.asarray(z['t_send_ns'], dtype=np.float64) * 1e-9
        mosi = np.asarray(z['mosi_true'] if 'mosi_true' in z.files else z['mosi'])
        miso = np.asarray(z['miso'])
    state, ref, _, _ = H.decode_frames(mosi, miso)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError(f'{path}: t_send_ns must increase over at least two frames')
    return t, state, ref, manifest


def expected_counts(tp, omega):
    """C (n, 4) at the frame send times: counts of the speed held from each frame to the next."""
    dphi = omega[:-1] * np.diff(tp)[:, None]
    phi = np.vstack([np.zeros((1, 4)), np.cumsum(dphi, axis=0)])
    return phi * COUNTS_PER_REV / (2 * math.pi)


def _segments(w_at):
    """Segment id per tick of one wheel (one sign, |omega| >= OMEGA_FIT), -1 where not used."""
    sgn = np.where(np.abs(w_at) >= OMEGA_FIT, np.sign(w_at), 0.0)
    seg = np.full(len(w_at), -1)
    k = -1
    for i in range(len(w_at)):
        if sgn[i] != 0 and (i == 0 or sgn[i] != sgn[i - 1]):
            k += 1
        if sgn[i] != 0:
            seg[i] = k
    return seg


def used_ticks(te, counts, tp, omega, delta0, beta0=1.0):
    """[(wheel, tick indices, segment ids)] of the ticks used for the fit (module docstring)."""
    out = []
    for w in range(4):
        w_at = np.interp(delta0 + beta0 * te, tp, omega[:, w], left=0.0, right=0.0)
        seg = _segments(w_at)
        keep = seg >= 0
        for k in np.unique(seg[keep]):
            m = seg == k
            t_start = te[m][0]
            keep &= ~(m & (te < t_start + SEG_SKIP_S))
        keep &= np.isfinite(counts[:, w])
        idx = np.flatnonzero(keep)
        if idx.size:
            out.append((w, idx, seg[idx]))
    return out


def _proj(x, seg):
    """x minus its mean in each segment (the per-(wheel, segment) count offset eliminated)."""
    out = np.array(x, dtype=float)
    for k in np.unique(seg):
        m = seg == k
        out[m] -= out[m].mean()
    return out


def _resid(delta, beta, te, counts, tp, C, use, sel=None):
    """[(residual after the segment offsets, wheel, idx, seg)] at t_Pi = delta + beta (t_ESP - t_ref)."""
    out = []
    for w, idx, seg in use:
        if sel is not None:
            keep = sel[idx]
            idx, seg = idx[keep], seg[keep]
        if idx.size == 0:
            continue
        r = counts[idx, w] - np.interp(delta + beta * te[idx], tp, C[:, w])
        out.append((_proj(r, seg), w, idx, seg))
    return out


def _cost(delta, beta, te, counts, tp, C, use, sel=None):
    """Sum of squared residuals (counts) and their number."""
    rs = _resid(delta, beta, te, counts, tp, C, use, sel)
    return sum(float(np.sum(r ** 2)) for r, _, _, _ in rs), sum(r.size for r, _, _, _ in rs)


def _grid_delta(center, beta, te, counts, tp, C, use):
    """Coarse start: delta on a grid at fixed beta (GRID1 over +-DELTA_SPAN_S, then GRID2)."""
    for span, step in ((DELTA_SPAN_S, GRID1_S), (3 * GRID1_S, GRID2_S)):
        grid = center + np.arange(-span, span + step / 2, step)
        center = min((_cost(d, beta, te, counts, tp, C, use)[0], d) for d in grid)[1]
    return center


def _gauss_newton(delta, beta, te, counts, tp, omega, C, use):
    """(delta, beta) of least squares, both free; the segment offsets are projected out. dC/dt is the held
    frame speed x 1320 / 2 pi, so the Jacobian of a count is -(that) x (1, t_ESP - t_ref)."""
    kc = COUNTS_PER_REV / (2 * math.pi)
    converged, A = False, np.zeros((2, 2))
    for _ in range(GN_ITER):
        A, g = _normal(delta, beta, te, counts, tp, omega, C, use, kc)
        if not np.all(np.isfinite(A)) or np.linalg.cond(A) > COND_MAX:
            return delta, beta, False, A                # S2P1-1: no information on delta or beta
        step = -np.linalg.solve(A, g)
        delta, beta = delta + step[0], beta + step[1]
        if abs(step[0]) + abs(step[1]) * (te.max() - te.min()) < GN_TOL_S:
            converged = True
            break
    A, _ = _normal(delta, beta, te, counts, tp, omega, C, use, kc)
    return delta, beta, converged, A


def _normal(delta, beta, te, counts, tp, omega, C, use, kc):
    """Normal matrix J^T J and gradient J^T r of the projected residuals at (delta, beta)."""
    A = np.zeros((2, 2))
    g = np.zeros(2)
    for r, w, idx, seg in _resid(delta, beta, te, counts, tp, C, use):
        q = delta + beta * te[idx]
        k = np.clip(np.searchsorted(tp, q, 'right') - 1, 0, len(tp) - 1)
        rate = omega[k, w] * kc
        J = np.stack([_proj(-rate, seg), _proj(-rate * te[idx], seg)], axis=1)
        A += J.T @ J
        g += J.T @ r
    return A, g


def fit_map(te, counts, tp, omega, delta0, beta0=1.0):
    """alpha, beta (t_Pi = alpha + beta t_ESP, s; t_ESP measured from the first tick) and the fit numbers; ok
    False with a reason when not mapped. delta0: start value of t_Pi at t_ESP = te[0] (beta0 slope)."""
    C = expected_counts(tp, omega)
    t_ref = float(te[0])
    u = te - t_ref                                       # conditioning: times from the first tick
    out = dict(ok=False, reason='', alpha=None, beta=None, drift_ppm=None, resid_rms=None, n_used=0,
               delta0=float(delta0))
    use = used_ticks(u, counts, tp, omega, delta0, beta0)
    if sum(idx.size for _, idx, _ in use) < MIN_TICKS:
        out['reason'] = 'too_few_turning_ticks'
        return out
    d = _grid_delta(delta0, beta0, u, counts, tp, C, use)
    d, beta, converged, A = _gauss_newton(d, beta0, u, counts, tp, omega, C, use)
    alpha = d - beta * t_ref
    ppm = (beta - 1.0) * 1e6
    tot, n = _cost(d, beta, u, counts, tp, C, use)
    n_seg = sum(len(np.unique(seg)) for _, _, seg in use)
    dof = n - 2 - n_seg
    sigma_t = float('inf')
    if converged and dof > 0 and np.linalg.cond(A) <= COND_MAX:
        # S2P1-1: standard error of the mapped time t_Pi = d + beta u over the run (largest at an end of u)
        cov = (tot / dof) * np.linalg.inv(A)
        sigma_t = max(math.sqrt(max(cov[0, 0] + 2 * x * cov[0, 1] + x * x * cov[1, 1], 0.0))
                      for x in (0.0, float(u.max())))
    out.update(alpha=float(alpha), beta=float(beta), drift_ppm=float(ppm), resid_rms=math.sqrt(tot / n),
               n_used=int(n), converged=bool(converged), sigma_t_s=sigma_t)
    if not converged or not math.isfinite(sigma_t):
        out['reason'] = 'not_identified'
    elif sigma_t > SIGMA_T_MAX_S:
        out['reason'] = 'sigma_t_over_max'
    elif out['resid_rms'] > RESID_MAX:
        out['reason'] = 'residual_over_max'
    elif abs(ppm) > PPM_MAX:
        out['reason'] = 'drift_out_of_bound'
    else:
        out['ok'] = True
    return out


def plant_rows(te_us, tp, state, alpha, beta):
    """Rows of PLANT_COLUMNS at the ESP32 times te_us (empty outside the frame log)."""
    th = np.unwrap(state[:, 6])
    rows = []
    for t in te_us:
        q = alpha + beta * t * 1e-6
        if not tp[0] <= q <= tp[-1]:
            rows.append([int(t)] + [''] * (len(PLANT_COLUMNS) - 1))
            continue
        v = [np.interp(q, tp, state[:, i]) for i in range(10)]
        theta = float(np.interp(q, tp, th))
        c, s = math.cos(theta), math.sin(theta)
        vxw, vyw, wz = v[7], v[8], v[9]
        vxb, vyb = c * vxw + s * vyw, -s * vxw + c * vyw
        thw = math.atan2(s, c)
        rows.append([int(t)] + [f'{x:.6f}' for x in (v[4], v[5], thw, vxw, vyw, vxb, vyb, wz)]
                    + [f'{x:.5f}' for x in v[0:4]])
    return rows


def pi_anchors(tp, ref):
    """Pi send times (a, b) of the first frame with a non-zero command and of the first zero frame after the
    last non-zero one (None when missing): the Pi side of the command anchors of s02_cmd_path."""
    nzm = np.any(np.abs(ref) > 0, axis=1)
    nz = np.flatnonzero(nzm)
    if nz.size == 0:
        return None, None
    after = np.flatnonzero(~nzm & (np.arange(len(tp)) > nz[-1]))
    return float(tp[nz[0]]), (float(tp[after[0]]) if after.size else None)


def map_run(frame_log, esp_log, test_id):
    """(result dict, rows) of one HIL run."""
    import s02_metrics
    tp, state, ref, manifest = load_frames(frame_log)
    log = s02_metrics.load_log(esp_log)['_all']
    on = log['motor_on'] == 1
    te_us = log['timestamp_us'][on]
    counts = np.stack([log[w] for w in WHEELS], axis=1)[on]
    counts[counts == -2 ** 31] = np.nan                     # AGV LOG_CNT_INVALID
    anc, _, _ = s02_cmd_path.anchors(log, test_id)
    pa, pb = pi_anchors(tp, ref)
    res = dict(test=test_id, frame_log=os.path.basename(frame_log), esp_log=os.path.basename(esp_log),
               method='encoder_counts', t0_us=anc['t0_us'], t0_method=anc['t0_method'])
    # S2P1-4: only an anchor s02_cmd_path kept (a dropped one is NaN there), with its own Pi side
    if math.isfinite(anc['anchor_a_us']) and pa is not None:
        ta, t_pi_a = anc['anchor_a_us'], pa
    elif math.isfinite(anc['anchor_b_us']) and pb is not None:
        ta, t_pi_a = anc['anchor_b_us'], pb
    else:
        res.update(ok=False, reason='no_start_value (command anchors)')
        return res, [[int(t)] + [''] * (len(PLANT_COLUMNS) - 1) for t in te_us]
    # S2P1-10: drift_ppm is the DUT span over the sender span, minus 1; t_Pi = alpha + beta t_ESP needs the inverse
    # (the anchors carry the 10 ms hand-over: a start value only, the fit sets beta)
    drift = anc['drift_ppm'] if math.isfinite(anc['drift_ppm']) and abs(anc['drift_ppm']) <= PPM_MAX else 0.0
    beta0 = 1.0 / (1.0 + drift * 1e-6)
    te = te_us * 1e-6
    delta0 = t_pi_a - beta0 * (ta * 1e-6 - ANCHOR_LAG_S - te[0])     # t_Pi at the first tick
    fit = fit_map(te, counts, tp, state[:, 0:4], delta0, beta0)
    res.update(fit)
    rows = (plant_rows(te_us, tp, state, fit['alpha'], fit['beta']) if fit['ok'] else
            [[int(t)] + [''] * (len(PLANT_COLUMNS) - 1) for t in te_us])
    return res, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame-log', required=True)
    ap.add_argument('--esp-log', required=True)
    ap.add_argument('--test', required=True, choices=sorted(s02_cmd_path.GROUP_A))
    ap.add_argument('--out', required=True, help='<run>_plant.csv (new file); the map goes to <run>_plant_map.json')
    a = ap.parse_args(argv)
    js = os.path.splitext(a.out)[0] + '_map.json'
    for p in (a.out, js):
        if os.path.exists(p):
            print(f'{p} already exists: outputs are not overwritten')
            print('#ket_qua,ma=2')
            return 2
    try:
        res, rows = map_run(a.frame_log, a.esp_log, a.test)
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}')
        print('#ket_qua,ma=2')
        return 2
    with open(a.out, 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(PLANT_COLUMNS)
        w.writerows(rows)
    with open(js, 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('ok', 'reason', 'drift_ppm', 'resid_rms', 'n_used')},
                     default=float))
    code = 0 if res.get('ok') else 1
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
