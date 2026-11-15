"""
s02_cmd_path.py: spec S02 t = 0 on the DUT clock from two command anchors, the check that every reset came
before t = 0 (the preparation phase), and the command path numbers from traj_age_ms.

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van), sections 4.1-4.3 (Liam 15/11,
CODE_S02 local_ff3bd6e3, "Neo theo lenh trong log (Recommended)", "3 s (Recommended)"); spec S02 v1.10
P2-2 (preparation, cold start of SIM), P2-2b (t = 0 by two anchors, command path counted per v1.8 B81c-9).
Called by tools/s02_metrics.py on every line of a log (motor-off lines included), as a dict of arrays
(columns of tools/agv_logger.COLUMNS), in log_seq order (DUT) or file order (SIM).

t = 0 (anchors). The test schedule u_k = get_body_velocity(k / 50) of rpi5/trajectory_pub.GROUP_A (read
only) has a first non-zero packet k_a and a first zero packet k_b after the last non-zero one (k_a = 0,
k_b = K for every test but A-K1: 150, 300). On the log (motor-on lines): t_a = timestamp_us of the first
line whose command (sp_vx, sp_vy, sp_wz) is non-zero, t_b = the first line with a zero command after the
last non-zero one. t_ESP(k) = t_a + (k - k_a) (t_b - t_a) / (k_b - k_a) (absorbs the sender / ESP32 clock
rate; drift_ppm printed). A line shows a packet only after it arrived (up to a tick, the 10 ms hand-over
of traj_task, the link): the mark is late by about 0..30 ms, the same at both anchors [ĐỀ XUẤT].
  An anchor whose line before (log_seq - 1) is missing is dropped (the real first line may be lost);
  anchor b is dropped when its line has traj_age_ms >= 200 or -1 (the zero came from the timeout, not
  from a zero packet: last packets lost). With both: their span must be (k_b - k_a) 20 ms within
  ANCHOR_PPM of it plus ANCHOR_SLACK_US; a shorter span means a came late (first packets lost, an
  ultrasonic cut, motors on after k = 0): a is dropped (anchor_a_late); a longer one means b came late
  (first tail zeros lost; on HIL also STOPPING up to about 0.1 s after the publisher ends: k2 of
  CLOSED_LOOP/2026-11-14_run01 +59 ms): b is dropped (anchor_b_late). The rate between two anchors is
  clamped to +-ANCHOR_PPM (drift_ppm keeps the raw value, t0_reason drift_clamped): the 10 ms hand-over of
  each anchor is thousands of ppm over a short span. One anchor: nominal 50 Hz from it (one_anchor,
  uncertainty +- ANCHOR_PPM over the test). None: t0_method none; every number below is NaN with the
  reason.

Preparation (P2-2 line 101). edge = the last rising edge of run_active at or before t_ESP(0): a line with
run_active 1 whose previous line (DUT: log_seq - 1, received) has run_active 0. SIM (platform sim in runs.csv):
the first line already 1 is a cold start (edge_kind cold_start; fwsim sets run_active from the first tick). Flags (only reported, the run is classified
by the analysis, P2-5):
  edge_unknown       no edge seen (previous line lost, or the log starts with run_active 1 on a DUT)
  edge_after_t0      the first edge comes after t_ESP(0): no preparation, or SWA on late
  prep_short         t_ESP(0) - edge < PREP_MIN_S [ĐỀ XUẤT 2.5 s]
  run_active_toggle  a line with run_active 0 between the edge and t_ESP(K - 1)
Also: prep_s_dut, g_valid_t0, bias_gz_t0 (line nearest t_ESP(0)), bias_updates_prep (bias_updates at
that line minus at the edge).

Command path (P2-2b, DUT only; SIM has no packets): lines with timestamp_us in [t_ESP(0), t_ESP(K - 1)]
(first to last packet of the standard schedule). cmd_age_max_ms = largest traj_age_ms; cmd_gap_n = number
of runs of consecutive received lines (log_seq order) with traj_age_ms > 200 ms, missing lines never
splitting a run (v1.8 B81c-9: two real gaps may count as one); cmd_age_unknown_n = lines with -1;
cmd_rows; cmd_seq_holes = lines missing by log_seq inside the window. traj_age_ms is read before the
packet snapshot and rounded down (AGV 5bde4c5 main.c 532-540): both numbers are LOWER BOUNDS, not the
absolute delay, loss, duplicates or reordering of packets.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rpi5'))
from trajectory_pub import GROUP_A, get_body_velocity  # noqa: E402

SEND_HZ = 50
PERIOD_US = 1e6 / SEND_HZ
TIMEOUT_MS = 200              # TRAJ_TIMEOUT_US of the firmware (main.c 45)
PREP_MIN_S = 2.5              # [ĐỀ XUẤT] design 4.2 (preparation 3 s, Liam 15/11)
ANCHOR_PPM = 2002.0           # S05 4.7 (i) sender / ESP32 clock rate bound
ANCHOR_SLACK_US = 40_000.0    # [ĐỀ XUẤT] 2 ticks + traj_task hand-over + link, between the two anchors
NAN = float('nan')

FIELDS = ('t0_us', 't0_method', 't0_reason', 'drift_ppm', 'anchor_a_us', 'anchor_b_us',
          'edge_kind', 'edge_us', 'prep_s_dut', 'prep_flags', 'run_active_drop',
          'g_valid_t0', 'bias_gz_t0', 'bias_updates_prep',
          'cmd_age_max_ms', 'cmd_gap_n', 'cmd_age_unknown_n', 'cmd_rows', 'cmd_seq_holes', 'cmd_reason')


def schedule(test_id):
    """(K, k_a, k_b) of a group A test: K packets, first non-zero, first zero after the last non-zero."""
    spec = GROUP_A[test_id]
    K = int(math.ceil(spec.duration * SEND_HZ - 1e-9))
    nz = [k for k in range(K) if any(abs(x) > 0.0 for x in get_body_velocity(k / SEND_HZ, spec))]
    if not nz:
        raise ValueError(f'{test_id}: the schedule has no non-zero packet')
    return K, nz[0], nz[-1] + 1


def _col(log, name):
    x = log.get(name)
    return None if x is None else np.asarray(x, dtype=float)


def _prev_received(seq, i):
    """True when the line before line i (log_seq - 1) is in the log (always True without log_seq)."""
    if seq is None:
        return i > 0
    return i > 0 and seq[i - 1] == seq[i] - 1


def anchors(log, test_id):
    """t0_us, method, reason, drift_ppm, t_a, t_b and the map k -> t_ESP(k) (None when unknown)."""
    K, ka, kb = schedule(test_id)
    t = _col(log, 'timestamp_us')
    seq = _col(log, 'log_seq')
    seq = None if seq is None or np.all(np.isnan(seq)) else seq
    mo = _col(log, 'motor_on')
    on = np.ones(len(t), bool) if mo is None or np.all(np.isnan(mo)) else (mo == 1)
    sp = np.stack([_col(log, c) for c in ('sp_vx', 'sp_vy', 'sp_wz')], axis=1)
    nzl = on & np.any(np.abs(np.nan_to_num(sp)) > 0.0, axis=1)
    idx = np.flatnonzero(nzl)
    out = dict(t0_us=NAN, t0_method='none', t0_reason='', drift_ppm=NAN, anchor_a_us=NAN, anchor_b_us=NAN)
    if idx.size == 0:
        out['t0_reason'] = 'no_nonzero_command'
        return out, None, K
    ia = int(idx[0])
    after = np.flatnonzero(on & ~nzl & (np.arange(len(t)) > idx[-1]))
    ib = int(after[0]) if after.size else None
    age = _col(log, 'traj_age_ms')
    a_ok = _prev_received(seq, ia)
    b_ok = ib is not None and _prev_received(seq, ib)
    why_b = 'no_tail' if ib is None else ('anchor_b_lost' if not b_ok else '')
    if b_ok and age is not None and math.isfinite(age[ib]) and not 0 <= age[ib] < TIMEOUT_MS:
        b_ok, why_b = False, 'anchor_b_timeout'      # command 0 from the 200 ms timeout, not a zero packet
    ta, tb = t[ia], (t[ib] if ib is not None else NAN)
    out['anchor_a_us'], out['anchor_b_us'] = (ta if a_ok else NAN), (tb if b_ok else NAN)
    reason = ''
    if a_ok and b_ok:
        span, nom = tb - ta, (kb - ka) * PERIOD_US
        if abs(span - nom) > ANCHOR_PPM * 1e-6 * nom + ANCHOR_SLACK_US:
            # one anchor moved: a later than its packet (first packets lost, an ultrasonic cut, motors
            # on late) makes the span short; b later (first tail zeros lost) makes it long
            if span < nom:
                a_ok, reason = False, 'anchor_a_late'
            else:
                b_ok, reason = False, 'anchor_b_late'
    if a_ok and b_ok:
        # the span carries the hand-over quantization of both anchors (10 ms) and, on HIL, a late tail
        # (STOPPING after the owner sees DONE); over a short span that is thousands of ppm (A-K1: 3 s,
        # CLOSED_LOOP/2026-11-14_run01 k1a -11 ms): the rate is clamped to the clock bound
        drift = ((tb - ta) / (kb - ka) / PERIOD_US - 1.0) * 1e6
        per = PERIOD_US * (1.0 + max(-ANCHOR_PPM, min(ANCHOR_PPM, drift)) * 1e-6)
        out.update(t0_method='two_anchors', drift_ppm=drift,
                   t0_reason='drift_clamped' if abs(drift) > ANCHOR_PPM else '')
        fmap = lambda k: ta + (k - ka) * per          # noqa: E731
    elif a_ok or b_ok:
        out.update(t0_method='one_anchor', t0_reason=reason or ('anchor_a_lost' if not a_ok else why_b))
        fmap = (lambda k: ta + (k - ka) * PERIOD_US) if a_ok else (lambda k: tb + (k - kb) * PERIOD_US)
    else:
        out['t0_reason'] = 'anchors_lost'
        return out, None, K
    out['t0_us'] = fmap(0)
    return out, fmap, K


def prep_check(log, t0, t_end, sim=False):
    """Preparation numbers and flags (module docstring). sim: the platform of the run is SIM (runs.csv):
    a log whose first line already has run_active 1 is a cold start (fwsim sets it from the first tick)."""
    t = _col(log, 'timestamp_us')
    ra = _col(log, 'run_active')
    out = dict(edge_kind='unknown', edge_us=NAN, prep_s_dut=NAN, prep_flags='', run_active_drop=NAN,
               g_valid_t0=NAN, bias_gz_t0=NAN, bias_updates_prep=NAN)
    if ra is None or np.all(np.isnan(ra)) or not math.isfinite(t0):
        out['prep_flags'] = 'no_run_active' if (ra is None or np.all(np.isnan(ra))) else 'no_t0'
        return out
    seq = _col(log, 'log_seq')
    seq = None if seq is None or np.all(np.isnan(seq)) else seq
    flags = []
    rises = [i for i in range(len(t)) if ra[i] == 1 and _prev_received(seq, i) and ra[i - 1] == 0]
    before = [i for i in rises if t[i] <= t0]
    ie = None
    if before:
        ie, out['edge_kind'] = before[-1], 'edge'
    elif sim and len(t) and ra[0] == 1 and not any(ra[:np.searchsorted(t, t0, 'right')] == 0):
        ie, out['edge_kind'] = 0, 'cold_start'
    elif rises:
        flags.append('edge_after_t0')
    else:
        flags.append('edge_unknown')
    if ie is not None:
        out['edge_us'] = t[ie]
        out['prep_s_dut'] = (t0 - t[ie]) * 1e-6
        if out['prep_s_dut'] < PREP_MIN_S:
            flags.append('prep_short')
        span = (t >= t[ie]) & (t <= t_end)
        out['run_active_drop'] = int(np.sum(ra[span] == 0))
        if out['run_active_drop'] > 0:
            flags.append('run_active_toggle')
        i0 = int(np.argmin(np.abs(t - t0)))
        for name, key in (('g_valid', 'g_valid_t0'), ('bias_gz', 'bias_gz_t0')):
            c = _col(log, name)
            out[key] = NAN if c is None else c[i0]
        bu = _col(log, 'bias_updates')
        out['bias_updates_prep'] = NAN if bu is None else bu[i0] - bu[ie]
    out['prep_flags'] = ';'.join(flags)
    return out


def cmd_path(log, t0, t_end):
    """Command path numbers over the lines with timestamp_us in [t0, t_end] (module docstring)."""
    out = dict(cmd_age_max_ms=NAN, cmd_gap_n=NAN, cmd_age_unknown_n=NAN, cmd_rows=NAN, cmd_seq_holes=NAN,
               cmd_reason='')
    age = _col(log, 'traj_age_ms')
    seq = _col(log, 'log_seq')
    if age is None or np.all(np.isnan(age)) or seq is None or np.all(np.isnan(seq)):
        out['cmd_reason'] = 'no_traj_age'            # SIM, or a log before traj_age_ms (12/11)
        return out
    if not (math.isfinite(t0) and math.isfinite(t_end)):
        out['cmd_reason'] = 'no_t0'
        return out
    t = _col(log, 'timestamp_us')
    w = np.flatnonzero((t >= t0) & (t <= t_end))
    if w.size == 0:
        out['cmd_reason'] = 'no_line_in_window'
        return out
    a, s = age[w], seq[w]
    known = a >= 0
    over = known & (a > TIMEOUT_MS)
    out['cmd_rows'] = int(w.size)
    out['cmd_age_unknown_n'] = int(np.sum(a == -1))
    out['cmd_age_max_ms'] = float(np.max(a[known])) if known.any() else NAN
    # a run starts at an over line whose previous received line in the window is not over;
    # lines lost in between never split a run (B81c-9)
    out['cmd_gap_n'] = int(over[0]) + int(np.sum(over[1:] & ~over[:-1]))
    out['cmd_seq_holes'] = int((s[-1] - s[0] + 1) - w.size)
    return out


def stats(log, test_id, platform=''):
    """Every field of FIELDS for one run (log: every line, dict of arrays; platform of runs.csv)."""
    res, fmap, K = anchors(log, test_id)
    t0 = res['t0_us']
    t_end = fmap(K - 1) if fmap is not None else NAN
    res.update(prep_check(log, t0, t_end, sim=str(platform).strip().lower() == 'sim'))
    res.update(cmd_path(log, t0, t_end))
    return {k: res[k] for k in FIELDS}
