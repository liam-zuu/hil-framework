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
  the floor quantization is correlated, so that error can be optimistic: the local sensitivity scale
  1 count / max |speed - segment mean| (a scale, not a bound: S2P2-1, Codex round 2) must also be
  <= T_SENS_MAX_S [ĐỀ XUẤT 2 ms]; else not mapped (not_identified, sigma_t_over_max, weak_excitation).
  Other branches (S2P2-1, Codex round 2): a speed that repeats with a period P (ripple) gives the same cost at
  delta + P, delta + 2P, ..., so the best point of one search can be a wrong branch tens of ms away. The cost
  is scanned at the fitted slope over the whole range the command anchors allow (start value +-AMBIG_SPAN_S:
  the anchor lag is 0..+50 ms, spec S02 P2-2b, around the middle value used here, plus a margin); any local
  minimum farther than AMBIG_SEP_S from the fit whose residual RMS is below the fit's + AMBIG_MARGIN
  [ĐỀ XUẤT 0.5 count] makes the map ambiguous: not mapped (ambiguous). Precision about 1 count / (1320 / 2 pi
  x |omega|): 0.5-1.2 ms at 4-10 rad/s [ƯỚC LƯỢNG]; the sensor path (Pi send -> H7 apply in its main loop ->
  EncoderGen) adds well under 1 ms (a limit: the apply time is not the send time, Codex round 2).
  Count rate of a frame (O-1, Codex round 2): the EncoderGen rate, not the ideal 1320 |omega| / 2 pi: no edges
  at |omega| <= 0.3 rad/s (ENC_OMEGA_MIN), half period rounded to whole 1 us timer ticks in float32
  (encoder_gen.c _half_period, ENC_CLOCK_CAL 1.0 at hil b907d98), so 2e6 / half_period counts/s (up to about
  500 ppm off the ideal at 10 rad/s). Start, stop and retime transients are not modelled (SEG_SKIP_S drops
  segment starts). The H7 timer clock against the ESP32 clock is a common rate, taken by beta.
  Not mapped either: a frame log with mosi_true (a run with --faults: the H7 got patched frames, the count
  model would need them, O-3), and a run whose '#us_end' reports cut ticks (us_cut, S2P2-2: same rule as the
  t = 0 of s02_cmd_path, Liam 15/11 CODE_S02 "Moc t = 0 khong xac dinh").
Plant at a tick: linear interpolation in Pi time between the two frames around alpha + beta t_ESP (theta
unwrapped first, wrapped back to (-pi, pi]); outside the frame log: empty (no extrapolation, P2-3).

NOT CERTIFIED (Codex Sol round 3, 15/11; thẩm định 03_CODE/docs/review/2026-11-15_Tham_dinh_S02_prep_vong3.md): a
fixed-beta scan misses branches with another beta (S2P3-1), minima between grid points are not refined (S2P3-2),
the command anchors do not bound the absolute lag (S2P3-3: lost packets at both ends shift both anchors), and the
rate model ignores the EncoderGen retime of every half-period change (S2P3-7). Liam 15/11 CODE_S02, nguyen van:
"Khoa anh xa, gop phan on (Recommended)": until the certification is redesigned (design v3), map_run reports the
fit (fit_ok, fit_reason, alpha, beta, ...) but returns ok False, reason not_certified, and no plant rows, so no
family R number on HIL can come out of this tool. fit_map itself is unchanged and tested.
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
T_SENS_MAX_S = 2e-3              # [ĐỀ XUẤT] local sensitivity scale _sensitivity (CLOSED_LOOP/2026-11-14_run01: 0.96-1.03 ms)
AMBIG_SPAN_S = 0.100             # cost scanned over the start value +- this (anchor lag 0..50 ms + margin; S2P2-1)
AMBIG_SEP_S = 3e-3               # a local minimum farther than this from the fit is another branch
AMBIG_MARGIN = 0.5               # [ĐỀ XUẤT] counts RMS: another branch within this of the fit -> ambiguous
ENC_OMEGA_MIN = 0.3              # rad/s, encoder_gen.c: no edges at or below
ENC_TICKS_PER_S = 1.0e6          # encoder_gen.h ENC_TIMER_FREQ x ENC_CLOCK_CAL (1.0)
ENC_PPR = 330
RESID_MAX = 2.0                  # [ĐỀ XUẤT] counts RMS (S02 v1.10)
CERTIFIED = False                # Codex round 3 S2P3-1..3, -7 open: map_run never outputs a map (module docstring)
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
        manifest['_has_mosi_true'] = 'mosi_true' in z.files      # O-3: a run with --faults
    state, ref, _, _ = H.decode_frames(mosi, miso)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError(f'{path}: t_send_ns must increase over at least two frames')
    return t, state, ref, manifest


def encodergen_omega(omega):
    """The wheel speed the EncoderGen edges carry for a frame speed omega (O-1, Codex round 2): 0 at |omega| <=
    ENC_OMEGA_MIN, else sign(omega) x 2 pi / 1320 x 2 / half_period, with half_period = (uint32)(hp + 0.5f) of
    hp = ENC_TICKS_PER_S pi / (|omega| ENC_PPR) in float32, clamped to 1..65535 (encoder_gen.c _half_period)."""
    om = np.asarray(omega, dtype=np.float64)
    a = np.abs(om).astype(np.float32)
    with np.errstate(divide='ignore'):
        hp = np.float32(ENC_TICKS_PER_S) * np.float32(math.pi) / (a * np.float32(ENC_PPR))
    hp = np.where(hp > 65535.0, 65535.0, np.where(hp < 1.0, 1.0, np.floor(hp + np.float32(0.5))))
    rate = 2.0 * ENC_TICKS_PER_S / hp.astype(np.float64)                 # counts per second (4 edges per cycle)
    out = np.sign(om) * rate * 2 * math.pi / COUNTS_PER_REV
    return np.where(om > ENC_OMEGA_MIN, out, np.where(om < -ENC_OMEGA_MIN, out, 0.0))


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


def _sensitivity(delta, beta, te, tp, omega, use):
    """1 count / the largest |speed - its segment mean| (counts/s) over the used ticks, s: the local scale below
    which a small time shift changes no projected count by more than one, so a floor quantization pattern
    (correlated, not white: the standard error alone missed a 3.9 ms bias at 0.1 rad/s ripple) can hide it.
    Only a local scale, not a bound on an undetected shift (S2P2-1): other branches are checked by _branches."""
    kc = COUNTS_PER_REV / (2 * math.pi)
    m = 0.0
    for w, idx, seg in use:
        k = np.clip(np.searchsorted(tp, delta + beta * te[idx], 'right') - 1, 0, len(tp) - 1)
        m = max(m, float(np.max(np.abs(_proj(omega[k, w] * kc, seg)))))
    return 1.0 / m if m > 0 else float('inf')


def _branches(d, beta, delta0, best_rms, te, counts, tp, C, use):
    """Other local minima of the cost at slope beta over delta0 +- AMBIG_SPAN_S (GRID1_S steps), farther than
    AMBIG_SEP_S from d: [(delta, residual RMS)] sorted by RMS, and the RMS of the best of them (inf if none)."""
    grid = delta0 + np.arange(-AMBIG_SPAN_S, AMBIG_SPAN_S + GRID1_S / 2, GRID1_S)
    cost = []
    for g in grid:
        tot, n = _cost(g, beta, te, counts, tp, C, use)
        cost.append(math.sqrt(tot / n) if n else float('inf'))
    cost = np.array(cost)
    loc = [i for i in range(len(grid))
           if (i == 0 or cost[i] <= cost[i - 1]) and (i == len(grid) - 1 or cost[i] <= cost[i + 1])]
    other = sorted((float(cost[i]), float(grid[i])) for i in loc if abs(grid[i] - d) > AMBIG_SEP_S)
    return [(g, c) for c, g in other], (other[0][0] if other else float('inf'))


def fit_map(te, counts, tp, omega, delta0, beta0=1.0):
    """alpha, beta (t_Pi = alpha + beta t_ESP, s; t_ESP measured from the first tick) and the fit numbers; ok
    False with a reason when not mapped. delta0: start value of t_Pi at t_ESP = te[0] (beta0 slope). omega: the
    frame wheel speeds; the count model uses encodergen_omega(omega) (O-1)."""
    omega = encodergen_omega(omega)
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
    t_sens = _sensitivity(d, beta, u, tp, omega, use)
    rms = math.sqrt(tot / n)
    alts, alt_rms = _branches(d, beta, delta0, rms, u, counts, tp, C, use)
    out.update(alpha=float(alpha), beta=float(beta), drift_ppm=float(ppm), resid_rms=rms,
               n_used=int(n), converged=bool(converged), sigma_t_s=sigma_t, t_sens_s=t_sens,
               alt_rms=alt_rms if math.isfinite(alt_rms) else None,            # None: no other minimum in the scan
               alt_shift_s=(alts[0][0] - d) if alts else None,
               alt_n=sum(1 for _, c in alts if c < rms + AMBIG_MARGIN))   # other branches within the margin
    if not converged or not math.isfinite(sigma_t):
        out['reason'] = 'not_identified'
    elif sigma_t > SIGMA_T_MAX_S:
        out['reason'] = 'sigma_t_over_max'
    elif t_sens > T_SENS_MAX_S:
        out['reason'] = 'weak_excitation'
    elif alt_rms < rms + AMBIG_MARGIN:
        out['reason'] = 'ambiguous'                      # S2P2-1: another branch fits as well
    elif rms > RESID_MAX:
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
    # S2P2-2 (Codex round 2): the same '#us_end' evidence as s02_metrics -> s02_cmd_path.stats
    us_cut = s02_cmd_path.us_cut_ticks(esp_log + '.events.txt')
    anc, _, _ = s02_cmd_path.anchors(log, test_id, us_cut)
    pa, pb = pi_anchors(tp, ref)
    res = dict(test=test_id, frame_log=os.path.basename(frame_log), esp_log=os.path.basename(esp_log),
               method='encoder_counts', t0_us=anc['t0_us'], t0_method=anc['t0_method'],
               us_cut_ticks=us_cut)
    empty = [[int(t)] + [''] * (len(PLANT_COLUMNS) - 1) for t in te_us]
    if us_cut is not None and us_cut > 0:
        res.update(ok=False, reason='us_cut')           # Liam 15/11 CODE_S02: no t = 0, no family R
        return res, empty
    if manifest.get('_has_mosi_true'):
        res.update(ok=False, reason='faults_not_supported')   # O-3: the H7 got patched frames
        return res, empty
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
    res.update(fit_ok=fit['ok'], fit_reason=fit['reason'], certified=CERTIFIED)
    if fit['ok'] and not CERTIFIED:
        # Liam 15/11 CODE_S02 "Khoa anh xa, gop phan on": the fit is reported for study, never used (fail-closed)
        res.update(ok=False, reason='not_certified')
    rows = (plant_rows(te_us, tp, state, fit['alpha'], fit['beta']) if res['ok'] else empty)
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
