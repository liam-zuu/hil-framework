"""
s02_metrics.py: spec S02 P2-3 metrics, one script for SIM (fwsim), HIL and REAL.

Input: logs in the ESP32 logger format (tools/agv_logger.py COLUMNS, firmware
AGV_MECANUM a6644a5 or later: 28 columns + tick, dpwm_*, sat_*). Every quantity
is the firmware's own: vx/vy/wz from encoder_get_rpm over 10 ms, pwm from
vel_ctrl, and the cumulative counters of vel_ctrl, so dropped log lines inside
the window do not change the PWM metrics; losing a line at either end of the
window moves the window itself (B0 review round 3, question 3).

Formats (B0 review round 3 B51/N9), told apart by the header and the rows:
  DUT     86 columns (85 AGV 54b7269 and older campaign firmware, 66 ctrl-imu, 80 feat/pos-loop: their header and rows as
          written by older loggers are accepted too), every row with log_seq and motor_on:
          log_seq U32, motor_on exactly 0 or 1, timestamp_us an integer
  SIM     86 (or 66, 80, 85) columns, log_seq and motor_on empty in every row (fwsim/s02_groupA_sim.py)
  legacy  the first 37 or 28 columns (older firmware)
Anything else, or rows of two formats, is refused. SIM and legacy rows stay in file
order and timestamp_us must be finite and never go back.
A DUT log next to the logger's <log>.segments.csv must also have <log>.summary.json
(status closed) whose 'files' manifest matches the CSV and the segment file byte for
byte; its segment must be 'consistent' in the summary, and only the '#run_end' groups
the logger assigned to that segment count (B47-B50, N5, N7, N8). A log with the
logger's summary but no segment file is refused (V6). A DUT log without any logger
file is checked from its rows and the '#run_end' lines of <log>.events.txt (same
parser as the logger); a footer whose slots hold data lines, or two footers with the
same seq and other counters (V10), are refused.

S02 scoring (v3, score_run; run_dir_metrics and the command line use it): design 03_CODE/docs/
2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van) sections 17-21 (Liam 15/11 CODE_S02 local_0eb4085a, "Lay gio
bo phat (Recommended)", "Duyet, code sau gop (Recommended)"); spec S02 v1.13 P2-2b, P2-3, P2-5, P2-6.
  t = 0 of the metrics (t0_metric_us, t0_metric_method): SIM exact (prep_s of runs.csv on the fwsim clock,
    sim_exact); HIL the sender time (the first non-zero MOSI frame of the frame log mapped to the ESP32 clock
    by tools/s02_plant_ticks.py, sender; no map -> no metric, metric_reason no_sender_t0, never the anchors),
    t0_lag_ms = anchor t = 0 - sender t = 0; car (real) and anything else: the two command anchors of
    tools/s02_cmd_path.py (two_anchors / one_anchor). No t = 0: every window metric NaN with metric_reason.
  Reference r(t) off the machine (tools/s02_ref.py); window: motor-on lines with 1 s <= t <= T_cmd (every test).
  Family E: e_vx, e_vy (r - encoder vx, vy), e_wz (r - wz_meas, the source the controller chose; wz_src_enc_frac
    = share of window lines on the encoder source), rms_w_err (IK(r) - wheel speeds), e_yaw_imu (r_wz -
    gz_fused), pwm_tv (dpwm counters over the window time on the DUT clock), sat_pct; A-K1, A-K2 step
    quantities per segment of r_vx (s02_ref.step_quantities: rise_s, overshoot_pct, frac90, no90,
    rise_uninterp, e_xl) on the lines with 0 <= t < T_cmd.
  Family R (SIM, HIL: <run>_plant.csv on the same timestamp_us; empty cells, a missing file or a map that is not
    ok -> NaN with r_reason): floor tests e_vx_true, e_vy_true, e_wz_true, e_p_rms/max, e_psi_rms/max (poses in
    the test frame, each pose taken relative to its own value at t = 0), e_est_p_rms/max, e_est_psi_rms/max
    (firmware pose vs plant; NaN when pos_reset_n rises from the last line before t = 0 to T_cmd, when there is no
    line before t = 0 (pos_reset_unchecked), or edge_after_t0); A-K: rms_w_err_true and
    the step quantities on fk(true wheel speeds) (<q>_true_segN).
  Family R of the car on the floor (real; Viec CODE 5, design section 34): <run>_plant.csv of tools/
    s02_video_metrics.py (camera pose on the log lines, cam_lost) with <run>_plant_video.json (not ok, e.g.
    clock_sync_failed -> no family R, r_reason from it): e_p, e_psi, e_est only (spec P2-3 table, no *_true speeds);
    the camera pose is taken relative to its mean over the valid camera frames (<run>_plant_cam.csv) in [-REST_S, 0)
    (spec line 155, Liam 17/11 "0,5 s truoc t = 0", "Tinh tren khung camera"; no frame or a gap over t_miss there ->
    no_rest_pose); a window line with cam_lost = 1 -> no family R, r_reason camera_lost (P2-5: measuring system
    failed); the json must match the log, plant and camera files by SHA-256 (else video_mismatch (what)). The car
    on stands (A-K) has no family R (real_lifted).
  Loss (P2-5): missing_pct over the motor-on tick stream (1 - distinct ticks / tick span), gen_pause_n (lines one
    tick apart but > 2 ticks apart on the DUT clock), missing_over (> 1 %); missing_seq_pct the older log_seq
    number. Runs not completed (runs.csv aborted != 0, result aborted, broken or replaced, block_status
    battery_rerun): one metrics.csv line with status and status_reason, no metric, the log is not read.
    A completed run with missing_over = 1 or run_bad = 1 is broken too (S2M1-1): status broken, no metric, its loss
    and reset columns kept. Other completed runs have status ok. The printed cell line is ok / 5 (P2-4) with
    broken, aborted, and the history rows (replaced, battery_rerun) apart. Next to metrics.csv: metrics_columns.json (unit, family, source per column).
  age_pre_a_ms, age_pre_b_ms: largest traj_age_ms in the 100 ms before each kept anchor line (design 25, THESIS
    15/11 hint); flags age_pre_a_over / age_pre_b_over past AGE_PRE_MAX_MS [ĐỀ XUẤT 40 ms]: a lost first
    non-zero packet (anchor a) or lost first tail zeros (anchor b) make the DUT hold the older packet longer.

compute() below is the older window (1 s after the first non-zero logged setpoint to the last one), kept for the
study tools and logger tests that call it (fwsim/slip_a_check.py, tools/tests/test_agv_logger.py,
test_run_validity.py, fwsim/tests/test_fwsim.py); S02 numbers come from score_run.

Per run, compute() (window: 1 s after the first non-zero setpoint to the last non-zero
setpoint, i.e. initial transient and final stop excluded):
  e_vx, e_vy, e_wz   RMS(setpoint - firmware body velocity)               m/s, m/s, rad/s
  rms_w_err          mean over wheels of RMS(IK(setpoint) - wheel speed)    rad/s
  pwm_tv             mean over wheels of (sum |du|) / T, u = pwm / 1000     1/s   (from dpwm counters)
  sat_pct            mean over wheels of saturated ticks / ticks * 100       %     (from sat counters)
  e_yaw_imu          RMS(wz setpoint - gz), only when gz is logged (REAL)   rad/s
  missing_pct        1 - logged lines / lines expected (whole log)           %
                     AGV ctrl-imu logs: from log_seq and the '#run_end' lines in
                     <log>.events.txt (log_loss); older logs: from the tick span
  missing_tail_known 1 when a '#run_end' line covers the end of the log, so a lost
                     end of run is counted; 0 when it cannot be seen
  pos_reset_mid      pose resets inside the run (pos_reset_n increases from the first
  run_bad            run_active = 1 line to the end of the log, every line before the motor-off
                     filter; tools/run_validity.py, STATUS_CODE L37, Codex S3C-7, S3C-8);
  run_bad_reason     run_bad = 1 marks a broken run to run again (spec controller-IMU
                     section 14.7.3 (A)); its metrics are still written but are not results.
                     NaN when the log has no pos_reset_n / run_active (firmware before 05/11)
Lifted tests A-K1, A-K2, per setpoint step whose new level lasts >= 2 s:
  rise_s_segN (10-90 % of the step), overshoot_pct_segN (beyond the new level,
  % of the step size), e_ss_segN (mean setpoint - vx over the last 1 s).

Quantization floor (THESIS 21/10): the firmware speed is a 10 ms count difference,
so one count on one wheel moves omega by Q_W = 2*pi/1320/0.010 = 0.476 rad/s,
vx and vy by r/4 * Q_W = 0.0058 m/s and wz by r/(4 L) * Q_W = 0.024 rad/s.
  step_<metric>: that one-count step.
  qrms_<metric>: RMS a perfect controller would still show from quantization
    alone, if each wheel's count error is the difference of two independent
    uniform phases (variance 1/6 count^2) and the 4 wheels are independent:
    vx, vy: r/4 * Q_W * sqrt(4/6); wz: r/(4 L) * Q_W * sqrt(4/6); rms_w_err: Q_W / sqrt(6).
  Metrics near qrms are measurement resolution, not control error. No pass/fail
  flag: deciding how close is "near" is left to the analysis.

  python tools/s02_metrics.py <run_dir> [out_name]   # reads runs.csv, writes metrics.csv (or out_name) and
                                                     # <out_name stem>_columns.json
"""
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agv_logger import (COLUMNS, MAX_US_PER_SEQ, N_IMU, OLD_LENGTHS, RATE_SLACK_US,  # noqa: E402
                        SEGMENT_COLUMNS, SEGMENT_STATUSES, file_digest, parse_run_end)
from run_validity import pos_reset_mid, run_bad_fields  # noqa: E402
import s02_cmd_path  # noqa: E402
import s02_ref  # noqa: E402
HEADS = [COLUMNS] + [COLUMNS[:n] for n in OLD_LENGTHS]     # 86, 28, 37, 66, 80, 85 columns

R_WHEEL = 0.0485
L_SUM = 0.24          # LX + LY (kinematics.h)
PWM_MAX = 1000.0
TICK_S = 0.01
WINDOW_SKIP_S = 1.0
SEG_MIN_S = 2.0
STEP_TESTS = ('A-K1', 'A-K2')
COUNTS_PER_REV = 1320
Q_W = 2 * math.pi / COUNTS_PER_REV / TICK_S                    # rad/s, one count in one window
STEPS = {'e_vx': R_WHEEL / 4 * Q_W, 'e_vy': R_WHEEL / 4 * Q_W,
         'e_wz': R_WHEEL / (4 * L_SUM) * Q_W, 'rms_w_err': Q_W}
QRMS = {'e_vx': STEPS['e_vx'] * math.sqrt(4 / 6), 'e_vy': STEPS['e_vy'] * math.sqrt(4 / 6),
        'e_wz': STEPS['e_wz'] * math.sqrt(4 / 6), 'rms_w_err': Q_W / math.sqrt(6)}


def load_log(path):
    path = os.fspath(path)
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    head = rows[0]
    body = rows[1:]
    if head not in HEADS:
        raise ValueError(f'{path}: header is not a logger format (agv_logger.COLUMNS, 86, 85, 80, 66, 37 or 28 columns)')
    bad = [i + 2 for i, r in enumerate(body) if len(r) != len(head)]
    if bad:
        raise ValueError(f'{path}: lines {bad[:5]} do not have {len(head)} columns (mixed formats)')
    t = head.index('timestamp_us')
    kind = 'legacy'
    if len(head) >= N_IMU:
        s, mo = head.index('log_seq'), head.index('motor_on')
        kinds = {('dut' if r[s] != '' and r[mo] != '' else
                  'sim' if r[s] == '' and r[mo] == '' else 'bad') for r in body}
        if 'bad' in kinds or len(kinds) > 1:
            raise ValueError(f'{path}: log mixes lines with and without log_seq / motor_on (mixed formats)')
        kind = kinds.pop() if kinds else 'sim'
    run_end_last = None
    if kind == 'dut':
        body, run_end_last = dut_rows(body, str(path))     # path may be a pathlib.Path
    else:
        body = _old_format(body, t)
    data = np.array([[float(x) if x != '' else np.nan for x in r] for r in body])
    log_seq_all = None
    every = {h: data[:, head.index(h)] for h in ('run_active', 'pos_reset_n', 'motor_on') if h in head}
    all_cols = {h: data[:, i] for i, h in enumerate(head)}       # CODE_S02: before the motor-off filter
    if kind == 'dut':
        log_seq_all = data[:, head.index('log_seq')]
        # 10 Hz lines while the motor is off carry no controller tick: drop them
        data = data[data[:, head.index('motor_on')] == 1]
    log = {h: data[:, i] for i, h in enumerate(head)}
    log['_every'] = every                                        # L37 check on every line (S3C-8)
    log['_all'] = all_cols                                       # s02_cmd_path: every column, every line
    if kind == 'dut':
        log['_seq_all'] = log_seq_all                            # every line, motor off ones too
        log['_run_end_last'] = run_end_last
    return log


def _check_dut_row(r, s, t, mo, where):
    """A DUT row as the firmware prints it (B36, B51/N9): log_seq U32, integer timestamp,
    motor_on exactly 0 or 1."""
    if not (r[s].isascii() and r[s].isdigit() and int(r[s]) < 2**32):
        raise ValueError(f'{where}: log_seq {r[s]!r} is not an unsigned 32-bit number')
    try:
        int(r[t])
    except ValueError:
        raise ValueError(f'{where}: timestamp_us {r[t]!r} is not an integer') from None
    if r[mo] not in ('0', '1'):
        raise ValueError(f'{where}: motor_on {r[mo]!r} outside {{0, 1}}')


def dut_rows(body, path):
    """Text rows of one DUT log in log_seq order, and the last_data_seq of the '#run_end'
    groups that belong to it (B0 review B4, B30-B33, B47-B51, N5, N7-N9). The logger keeps
    lines as received; drop duplicate lines and undo UDP reordering."""
    s, t, mo = COLUMNS.index('log_seq'), COLUMNS.index('timestamp_us'), COLUMNS.index('motor_on')
    for i, r in enumerate(body):
        _check_dut_row(r, s, t, mo, f'{path} line {i + 2}')
    seg_path = path + '.segments.csv'
    if os.path.isfile(seg_path):
        body, ends = _logger_rows(body, path, seg_path)
    elif os.path.isfile(path + '.summary.json'):
        # written by the logger but its segment file is missing: never fall back to the
        # weaker plain-CSV checks (V6)
        raise ValueError(f'{path}.summary.json exists but {seg_path} is missing: incomplete logger output')
    else:
        seen = set()
        uniq = []
        for r in body:                      # equal as text: NaN fields do not matter (B31)
            if tuple(r) not in seen:
                seen.add(tuple(r))
                uniq.append(r)
        body, ends = uniq, None
    body = sorted(body, key=lambda r: int(r[s]))
    seq = [int(r[s]) for r in body]
    ts = [int(r[t]) for r in body]
    if ends is None:
        # no logger segment file: the stricter sample-time rule, rate bound included (a stalled
        # control task is refused too), and the footers of the events file
        for (s0, t0), (s1, t1) in zip(zip(seq, ts), zip(seq[1:], ts[1:])):
            if s1 == s0 or not t0 <= t1 <= t0 + MAX_US_PER_SEQ * (s1 - s0) + RATE_SLACK_US:
                raise ValueError(f'log crosses a DUT restart or has conflicting lines (seq {s0} -> {s1}, '
                                 f'timestamp_us {t0} -> {t1}); split it per boot before computing metrics')
        ends = read_run_end(path + '.events.txt')
        data = set(seq)
        for L in ends:
            clash = [L + k for k in (1, 2, 3) if L + k in data]
            if clash and L >= seq[0]:
                raise ValueError(f'#run_end of last_data_seq {L}: data line(s) {clash} where its '
                                 'footers must be (another boot, or a bad footer); not usable as is')
    else:
        for (s0, t0), (s1, t1) in zip(zip(seq, ts), zip(seq[1:], ts[1:])):
            if s1 == s0 or t1 < t0:          # the logger's segment guarantees this; check anyway
                raise ValueError(f'segment rows out of order (seq {s0} -> {s1}, timestamp_us {t0} -> {t1})')
    return body, ends


def _logger_rows(body, path, seg_path):
    """Rows of a log written by agv_logger.py, checked against its segment file and summary."""
    summ_path = path + '.summary.json'
    if not os.path.isfile(summ_path):
        raise ValueError(f'{seg_path} without {summ_path}: cannot check the logger output')
    with open(summ_path, encoding='utf-8') as f:
        summ = json.load(f)
    if summ.get('status') != 'closed':
        raise ValueError(f'{summ_path}: status {summ.get("status")!r}, the logger did not close cleanly')
    files = summ.get('files') or {}
    for name, p in (('csv', path), ('segments', seg_path)):
        want = files.get(name) or {}
        got = file_digest(p)
        if (want.get('bytes'), want.get('sha256')) != (got['bytes'], got['sha256']):
            raise ValueError(f'{p} does not match files.{name} of {summ_path}: edited, cut, split '
                             'or from another run')
    with open(seg_path, newline='') as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != SEGMENT_COLUMNS:
        raise ValueError(f'{seg_path}: header is not {SEGMENT_COLUMNS}')
    marks = rows[1:]
    if len(marks) != len(body):
        raise ValueError(f'{seg_path}: {len(marks)} rows for {len(body)} data lines')
    s, t = COLUMNS.index('log_seq'), COLUMNS.index('timestamp_us')
    keep, seen = [], set()
    segs = set()
    for i, (r, m) in enumerate(zip(body, marks), start=1):
        if len(m) != len(SEGMENT_COLUMNS) or m[0] != str(i) or m[1] != r[s] or m[2] != r[t]:
            raise ValueError(f'{seg_path} row {i} does not match CSV data line {i}')
        status, g = m[4], m[3]
        if status not in SEGMENT_STATUSES or status == 'no_seq' or not (g.isascii() and g.isdigit()):
            raise ValueError(f'{seg_path} row {i}: segment {g!r} / status {status!r} not valid for a DUT line')
        if status in ('conflict', 'ambiguous'):
            raise ValueError(f'log has {status} lines (see {seg_path}); not usable as is')
        if status == 'duplicate':
            if tuple(r) not in seen:
                raise ValueError(f'{seg_path} row {i}: marked duplicate but no earlier equal line')
            continue
        seen.add(tuple(r))
        keep.append(r)
        segs.add(int(g))
    if len(segs) > 1:
        raise ValueError('log crosses a DUT restart (more than one logger segment); '
                         'compute metrics on one segment')
    ends = []
    if segs:
        k = segs.pop()
        segment = next((x for x in summ.get('segments', []) if x.get('segment') == k), None)
        if segment is None or segment.get('consistent') is not True:
            raise ValueError(f'segment {k} is not consistent in {summ_path} (ambiguous, conflicting '
                             'or unresolved footer lines); not usable as is')
        ends = sorted({grp['last_data_seq'] for grp in segment.get('run_end_groups', [])})
    return keep, ends


def _old_format(body, t):
    """SIM and legacy rows: file order, timestamps finite and never going back (B51/N9)."""
    try:
        ts = [float(r[t]) for r in body]
    except ValueError:
        raise ValueError('old-format log: timestamp_us not a number') from None
    if not all(math.isfinite(x) for x in ts):
        raise ValueError('old-format log: timestamp_us not finite')
    if any(b < a for a, b in zip(ts, ts[1:])):
        raise ValueError('old-format log: timestamp_us goes back in file order')
    return body


def read_run_end(events_path):
    """last_data_seq of every '#run_end' line of the events file that agv_logger.parse_run_end
    accepts (B50/N5: the same parser as the logger; empty if none). Two footers with the same
    (seq, last_data_seq) but other counters are refused, as the logger flags them (V10)."""
    seen = {}
    if os.path.isfile(events_path):
        with open(events_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                i = line.find('#run_end,')
                if i < 0:
                    continue
                kv = parse_run_end(line[i:].strip())
                if kv is None:
                    continue
                key = (kv['seq'], kv['last_data_seq'])
                val = (kv['log_qdrop'], kv['log_send_err'], kv.get('udp_drop'), kv.get('udp_take'))
                if seen.setdefault(key, val) != val:
                    raise ValueError(f'{events_path}: #run_end seq={key[0]} last_data_seq={key[1]} '
                                     'with two different counter values; not usable as is')
    return sorted({L for _, L in seen})


def check_no_udp_taken(events_path):
    """One binary (Outline QĐ 6 level 1): on the HIL bench the setpoints must come from the H7
    relay. A '#run_end' with udp_take > 0 means a UDP packet reached the controller during the
    run, so that run is not valid. Firmware that prints no udp_take is not checked."""
    if not os.path.isfile(events_path):
        return
    with open(events_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            i = line.find('#run_end,')
            if i < 0:
                continue
            kv = parse_run_end(line[i:].strip())
            if kv is not None and kv.get('udp_take', 0) > 0:
                raise ValueError(f"{events_path}: #run_end last_data_seq={kv['last_data_seq']} has "
                                 f"udp_take={kv['udp_take']}: a UDP packet reached the controller "
                                 'during this HIL run; not usable as is')


def log_loss(log):
    """missing_pct and whether the end of the log is covered (Codex ctrl-imu #5, Liam 28/10).
    New format: log_seq counts every logger_push from boot (data lines, 10 Hz motor-off lines,
    3 '#run_end' copies with seq L+1..L+3 after the last data line L). Expected lines run from
    the first received seq to max(last received seq, last_data_seq of the #run_end lines), less
    the #run_end copies; so losing the end of a run counts too when a #run_end line arrived.
    Lines lost before the first received one are not visible. Old format: tick span."""
    if '_seq_all' not in log:
        tick = log['tick']
        return float((1 - len(tick) / (tick[-1] - tick[0] + 1)) * 100), 0
    seq = np.unique(log['_seq_all'].astype(np.int64))
    ends = [L for L in log['_run_end_last'] if L >= seq[0]]
    s0, s1 = int(seq[0]), int(max([seq[-1]] + ends))
    run_end = {L + k for L in ends for k in (1, 2, 3)}
    expected = (s1 - s0 + 1) - sum(1 for s in run_end if s0 <= s <= s1)
    got = int(np.sum([s not in run_end for s in seq]))
    last_on = np.nanmax(log['log_seq']) if np.isfinite(log['log_seq']).any() else seq[-1]   # motor-on lines
    tail_known = int(bool(ends) and max(ends) >= last_on)
    return float((1 - got / expected) * 100), tail_known


def ik(vx, vy, wz):
    k = 1.0 / R_WHEEL
    return np.stack([k * (vx - vy - L_SUM * wz), k * (vx + vy + L_SUM * wz),
                     k * (vx + vy - L_SUM * wz), k * (vx - vy + L_SUM * wz)], axis=1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def step_segments(t, sp, v):
    """Setpoint steps (sp changes between consecutive lines) whose new level lasts >= SEG_MIN_S."""
    ch = np.flatnonzero(np.abs(np.diff(sp)) > 1e-6) + 1
    bounds = list(ch) + [len(sp)]
    out = []
    for i, k in enumerate(ch):
        end = bounds[i + 1]
        if t[end - 1] - t[k] < SEG_MIN_S - 2 * TICK_S:
            continue
        a, b = sp[k - 1], sp[k]
        seg_t, seg_v = t[k:end], v[k:end]
        frac = (seg_v - a) / (b - a)
        i10, i90 = np.flatnonzero(frac >= 0.1), np.flatnonzero(frac >= 0.9)
        rise = seg_t[i90[0]] - seg_t[i10[0]] if i10.size and i90.size else np.nan
        over = max(float(np.max((seg_v - b) * np.sign(b - a))), 0.0) / abs(b - a) * 100
        last = seg_t >= seg_t[-1] - 1.0 + 1e-9
        out.append(dict(from_=a, to=b, rise_s=rise, overshoot_pct=over, e_ss=float(np.mean(b - seg_v[last]))))
    return out


def compute(log, test_id):
    t = (log['timestamp_us'] - log['timestamp_us'][0]) * 1e-6
    sp = np.stack([log['sp_vx'], log['sp_vy'], log['sp_wz']], axis=1)
    v = np.stack([log['vx'], log['vy'], log['wz']], axis=1)
    w = np.stack([log[f'{c}_rpm'] for c in ('fl', 'fr', 'rl', 'rr')], axis=1) * 2 * math.pi / 60
    tick = log['tick']
    m = {}
    m['missing_pct'], m['missing_tail_known'] = log_loss(log)
    ev = log.get('_every', log)          # every logged line, before the motor-off filter (S3C-8)
    m.update(run_bad_fields(pos_reset_mid(ev.get('run_active'), ev.get('pos_reset_n'), ev.get('motor_on'))))
    if '_all' in log and test_id in s02_cmd_path.GROUP_A:
        m.update(s02_cmd_path.stats(log['_all'], test_id, log.get('_platform', ''), log.get('_us_cut'),
                                   log.get('_prep_s')))     # CODE_S02: t = 0, preparation, command path

    moving =np.flatnonzero(np.any(np.abs(sp) > 1e-9, axis=1))
    t0, t1 = t[moving[0]] + WINDOW_SKIP_S, t[moving[-1]]
    win = (t >= t0) & (t <= t1)
    e = sp[win] - v[win]
    m['e_vx'], m['e_vy'], m['e_wz'] = rms(e[:, 0]), rms(e[:, 1]), rms(e[:, 2])
    m['rms_w_err'] = float(np.mean([rms(c) for c in (ik(*sp[win].T) - w[win]).T]))
    i0, i1 = np.flatnonzero(win)[[0, -1]]
    # cumulative counters (B51/N9): finite and never going back inside the window, or the
    # differences below mean nothing (a reset, a bad line); refused, not computed
    for name in ['tick'] + [f'{k}_{c}' for k in ('dpwm', 'sat') for c in ('fl', 'fr', 'rl', 'rr')]:
        x = log[name][i0:i1 + 1]
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) < 0):
            raise ValueError(f'counter {name} not finite or going back inside the metric window')
    dticks = tick[i1] - tick[i0]
    if dticks <= 0:
        raise ValueError('tick does not advance inside the metric window')
    dp = np.array([log[f'dpwm_{c}'][i1] - log[f'dpwm_{c}'][i0] for c in ('fl', 'fr', 'rl', 'rr')])
    sat = np.array([log[f'sat_{c}'][i1] - log[f'sat_{c}'][i0] for c in ('fl', 'fr', 'rl', 'rr')])
    if np.any(sat > dticks):
        raise ValueError('saturated ticks exceed controller ticks inside the metric window')
    m['pwm_tv'] = float(np.mean(dp / PWM_MAX / (dticks * TICK_S)))
    m['sat_pct'] = float(np.mean(sat / dticks * 100))
    gz = log.get('gz')
    m['e_yaw_imu'] = rms(sp[win, 2] - gz[win]) if gz is not None and np.all(np.isfinite(gz[win])) else np.nan
    m['window_s'] = float(t1 - t0)
    for k in STEPS:
        m[f'step_{k}'] = STEPS[k]
        m[f'qrms_{k}'] = QRMS[k]
    if test_id in STEP_TESTS:
        for n, s in enumerate(step_segments(t, sp[:, 0], v[:, 0]), start=1):
            m[f'step_seg{n}'] = f"{s['from_']:+.2f}->{s['to']:+.2f}"
            m[f'rise_s_seg{n}'], m[f'overshoot_pct_seg{n}'], m[f'e_ss_seg{n}'] = s['rise_s'], s['overshoot_pct'], s['e_ss']
    return m


# ---------------------------------------------------------------------------------------------------------------
# S02 scoring v3 (module docstring "S02 scoring")

LIFTED = ('A-K1', 'A-K2', 'A-K3')
WHEEL_RPM = ('fl_rpm', 'fr_rpm', 'rl_rpm', 'rr_rpm')
WINDOW_EPS_S = 1e-9
MISSING_MAX_PCT = 1.0                 # P2-5 (v1.1): <= 1 % per run
GEN_PAUSE_US = 2 * TICK_S * 1e6       # P2-5: two lines one tick apart but more than 2 periods apart
AGE_PRE_S = 0.100                     # design 25 (THESIS hint): traj_age_ms over the 100 ms before an anchor
AGE_PRE_MAX_MS = 40.0                 # [ĐỀ XUẤT]
WZ_SRC_ENC = 0                        # AGV imu_fusion.h WZ_SRC_ENC
PLATFORMS_R = ('sim', 'hil', 'real')  # family R from <run>_plant.csv (the car: tools/s02_video_metrics.py)
REST_S = 0.5                          # s, the car's rest pose: mean over [-REST_S, 0) (spec P2-3 line 155)
STATUS_FIELDS = ('status', 'status_reason')
# columns a broken run keeps (S2M1-1): loss and mid-run resets, no metric
BROKEN_KEEP = ('missing_pct', 'gen_pause_n', 'missing_over', 'missing_seq_pct', 'missing_tail_known',
               'pos_reset_mid', 'run_bad', 'run_bad_reason')
N_REP = 5                             # spec S02 P2-4: 5 runs per cell, fixed before the runs
ID_FIELDS = ('run_id', 'platform', 'controller', 'test_id', 'speed_mps', 'rep')
NAN = float('nan')

E_METRICS = ('e_vx', 'e_vy', 'e_wz', 'rms_w_err', 'e_yaw_imu', 'pwm_tv', 'sat_pct')
R_FLOOR = ('e_vx_true', 'e_vy_true', 'e_wz_true', 'e_p_rms', 'e_p_max', 'e_psi_rms', 'e_psi_max',
           'e_est_p_rms', 'e_est_p_max', 'e_est_psi_rms', 'e_est_psi_max')
R_LIFTED = ('rms_w_err_true',)


def tick_loss(log):
    """missing_pct, gen_pause_n over the motor-on tick stream (P2-5): 1 - distinct ticks / (last - first + 1);
    a generation pause is a pair of lines one tick apart whose timestamp_us differ by more than 2 periods."""
    tick = np.asarray(log['tick'], dtype=float)
    ts = np.asarray(log['timestamp_us'], dtype=float)
    ok = np.isfinite(tick) & np.isfinite(ts)
    tick, ts = tick[ok], ts[ok]
    if tick.size == 0:
        return NAN, NAN
    u = np.unique(tick)
    missing = (1.0 - u.size / (u[-1] - u[0] + 1.0)) * 100.0
    o = np.argsort(tick, kind='stable')
    tk, tt = tick[o], ts[o]
    pause = int(np.sum((np.diff(tk) == 1) & (np.diff(tt) > GEN_PAUSE_US)))
    return float(missing), pause


def age_pre(all_lines, anchor_us):
    """Largest traj_age_ms (>= 0) on the motor-on lines with timestamp_us in [anchor - AGE_PRE_S, anchor) (NaN when
    the anchor is unknown or no such line has an age)."""
    if not math.isfinite(anchor_us):
        return NAN
    age = all_lines.get('traj_age_ms')
    if age is None:
        return NAN
    t = np.asarray(all_lines['timestamp_us'], dtype=float)
    mo = all_lines.get('motor_on')
    on = np.ones(len(t), bool) if mo is None or np.all(np.isnan(mo)) else (np.asarray(mo) == 1)
    a = np.asarray(age, dtype=float)
    m = on & (t >= anchor_us - AGE_PRE_S * 1e6) & (t < anchor_us) & np.isfinite(a) & (a >= 0)
    return float(np.max(a[m])) if m.any() else NAN


def load_plant(path):
    """<run>_plant.csv (PLANT_COLUMNS of fwsim/s02_groupA_sim.py, tools/s02_plant_ticks.py) as float arrays, empty
    cells NaN, and the row of each timestamp_us."""
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    if 'timestamp_us' not in head:
        raise ValueError(f'{path}: no timestamp_us column')
    if any(len(r) != len(head) for r in body):
        raise ValueError(f'{path}: a row does not have {len(head)} columns')
    data = (np.array([[float(x) if x != '' else np.nan for x in r] for r in body]) if body else
            np.zeros((0, len(head))))
    plant = {h: data[:, i] for i, h in enumerate(head)}
    ts = plant['timestamp_us']
    if not np.all(np.isfinite(ts)) or len(set(ts.tolist())) != len(ts):
        raise ValueError(f'{path}: timestamp_us missing or repeated')
    plant['_row'] = {int(t): i for i, t in enumerate(ts)}
    return plant


def _plant_cols(plant, prow, names, mask):
    """Plant columns at the lines of mask (n, len(names)), or None when a line has no plant row or an empty cell."""
    idx = prow[mask]
    if idx.size == 0 or np.any(idx < 0) or any(n not in plant for n in names):
        return None
    out = np.stack([plant[n][idx] for n in names], axis=1)
    return None if not np.all(np.isfinite(out)) else out


def _at_zero(t, vals):
    """Linear interpolation of the columns of vals (n, k) at t = 0 between the two lines around it (None when t = 0 is
    not between two lines with values)."""
    ok = np.all(np.isfinite(vals), axis=1)
    t, vals = t[ok], vals[ok]
    j = int(np.searchsorted(t, 0.0, 'right')) - 1
    if j < 0:
        return None
    if t[j] == 0.0:
        return vals[j]
    if j + 1 >= len(t):
        return None
    w = (0.0 - t[j]) / (t[j + 1] - t[j])
    return vals[j] + w * (vals[j + 1] - vals[j])


def _unwrap_valid(th):
    """np.unwrap over the finite entries, NaN kept."""
    out = np.array(th, dtype=float)
    ok = np.isfinite(out)
    out[ok] = np.unwrap(out[ok])
    return out


def _rel_pose(t, x, y, th, psi0):
    """Pose in the test frame relative to its own value at t = 0 (P2-3), or None when that value is unknown."""
    th = _unwrap_valid(th)
    z = _at_zero(t, np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float), th], axis=1))
    if z is None:
        return None
    return s02_ref.to_test_frame(x, y, th, z[0], z[1], z[2], psi0)


def _rest_rel_pose(x, y, th, psi0, cam, t0_us):
    """The car's camera pose (x, y, th on the log lines, th already unwrapped with the frames) in the test frame
    relative to the mean camera pose over the valid frames in [-REST_S, 0) (P2-3 line 155; S2V1-4, Codex round 1,
    Liam 17/11 "Tinh tren khung camera (Recommended)": the frames of <run>_plant_cam.csv, not the log lines, so lost
    log lines do not move it), or None when there is no frame there or a gap over the camera t_miss (the span ends
    counted) leaves the rest uncovered. cam: {'timestamp_us', 'x', 'y', 'theta', 't_miss_s'}."""
    if cam is None:
        return None
    tc = (np.asarray(cam['timestamp_us'], dtype=float) - t0_us) * 1e-6
    rest = (tc >= -REST_S - WINDOW_EPS_S) & (tc < 0.0)
    if not rest.any() or np.max(np.diff(np.r_[-REST_S, tc[rest], 0.0])) > cam['t_miss_s']:
        return None
    z = [float(np.mean(np.asarray(cam[k], dtype=float)[rest])) for k in ('x', 'y', 'theta')]
    return s02_ref.to_test_frame(x, y, np.asarray(th, dtype=float), z[0], z[1], z[2], psi0)


def _plant_series(plant, prow, name):
    """Plant column on every log line (NaN where the line has no plant row)."""
    if name not in plant:
        return np.full(len(prow), np.nan)
    return np.where(prow >= 0, plant[name][np.clip(prow, 0, None)], np.nan)


def _family_r(log, test_id, t, win, plant, prep_flags, platform='', t0_us=NAN):
    """Family R numbers, r_reason and e_est_reason (module docstring); platform real: the camera rows of
    tools/s02_video_metrics.py."""
    m = {}
    prow = np.array([plant['_row'].get(int(x), -1) for x in log['timestamp_us']], dtype=np.int64)
    T = s02_ref.t_cmd(test_id)
    reasons = []
    r = s02_ref.reference(test_id, t[win])
    if test_id in LIFTED:
        W = _plant_cols(plant, prow, ['w1', 'w2', 'w3', 'w4'], win)
        m['rms_w_err_true'] = (float(np.mean([rms(c) for c in (s02_ref.ik(*r.T) - W).T])) if W is not None else NAN)
        if W is None:
            reasons.append('plant_gap')
        if test_id in STEP_TESTS:
            seg = (t >= -WINDOW_EPS_S) & (t < T - WINDOW_EPS_S)
            W2 = _plant_cols(plant, prow, ['w1', 'w2', 'w3', 'w4'], seg)
            if W2 is None:
                reasons.append('plant_gap_steps')
            q = s02_ref.step_quantities(t[seg] if W2 is not None else np.zeros(0),
                                        s02_ref.fk(W2)[:, 0] if W2 is not None else np.zeros(0), test_id)
            for k, v in q.items():
                if not k.startswith('step_seg'):
                    m[k.replace('_seg', '_true_seg')] = v
        m['r_reason'] = ';'.join(dict.fromkeys(reasons))
        return m
    for k in R_FLOOR:
        m[k] = NAN
    m['e_est_reason'] = ''
    real = platform == 'real'
    if real:
        # the car: pose only (spec P2-3 table); a window line in a camera gap longer than t_miss: the measuring
        # system failed (P2-5), no family R
        if np.any(_plant_series(plant, prow, 'cam_lost')[win] == 1):
            m['e_est_reason'] = m['r_reason'] = 'camera_lost'
            return m
    else:
        V = _plant_cols(plant, prow, ['vx_b', 'vy_b', 'wz'], win)
        if V is None:
            reasons.append('plant_gap')
        else:
            e = r - V
            m['e_vx_true'], m['e_vy_true'], m['e_wz_true'] = rms(e[:, 0]), rms(e[:, 1]), rms(e[:, 2])
    psi0 = s02_ref.start_heading(test_id)
    xyt = [_plant_series(plant, prow, n) for n in ('x', 'y', 'theta')]
    true = _rest_rel_pose(*xyt, psi0, plant.get('_cam'), t0_us) if real else _rel_pose(t, *xyt, psi0)
    if true is None or not all(np.all(np.isfinite(c[win])) for c in true):
        reasons.append(('no_rest_pose' if real else 'no_plant_pose_at_t0') if true is None else 'plant_pose_gap')
        m['e_est_reason'] = 'no_plant_pose'
        m['r_reason'] = ';'.join(dict.fromkeys(reasons))
        return m
    X, Y, PSI = (c[win] for c in true)
    ref = s02_ref.pose_ref(test_id, t[win])
    ep = np.hypot(X - ref[:, 0], Y - ref[:, 1])
    eps = np.abs(s02_ref.wrap(PSI - ref[:, 2]))
    m['e_p_rms'], m['e_p_max'] = rms(ep), float(np.max(ep))
    m['e_psi_rms'], m['e_psi_max'] = rms(eps), float(np.max(eps))
    # e_est: firmware pose vs plant pose, both in the test frame; undefined after a reset in the window or with no
    # preparation (spec P2-2 "Trinh tu khoi tao")
    why = ''
    if 'edge_after_t0' in (prep_flags or '').split(';'):
        why = 'edge_after_t0'
    prn = log.get('pos_reset_n')
    if not why and prn is not None and np.any(np.isfinite(prn)):
        # S2M1-8 (Codex round 1 of v3): the baseline is the last line before t = 0 (t < 0), so a reset on the line
        # at t = 0 itself counts; no line before t = 0: a reset at t = 0 cannot be seen, e_est undefined
        # S2M2-5 (Codex round 2 of v3): the last line before t = 0 with a value (an empty cell is no baseline)
        fin = np.flatnonzero((t < 0.0) & np.isfinite(prn))
        j0 = int(fin[-1]) if fin.size else -1
        if j0 < 0:
            why = 'pos_reset_unchecked'
        else:
            span = (np.arange(len(t)) >= j0) & (t <= T + WINDOW_EPS_S)
            p = prn[span]
            p = p[np.isfinite(p)]
            if p.size and np.max(p) > p[0]:
                why = 'pos_reset_in_window'
    fw = None
    if not why:
        if not all(c in log for c in ('pos_x', 'pos_y', 'pos_th')):
            why = 'no_firmware_pose'
        else:
            fw = _rel_pose(t, log['pos_x'], log['pos_y'], log['pos_th'], psi0)
            if fw is None or not all(np.all(np.isfinite(c[win])) for c in fw):
                why = 'no_firmware_pose'
    m['e_est_reason'] = why
    if not why:
        FX, FY, FP = (c[win] for c in fw)
        ee = np.hypot(FX - X, FY - Y)
        ea = np.abs(s02_ref.wrap(FP - PSI))
        m['e_est_p_rms'], m['e_est_p_max'] = rms(ee), float(np.max(ee))
        m['e_est_psi_rms'], m['e_est_psi_max'] = rms(ea), float(np.max(ea))
    m['r_reason'] = ';'.join(dict.fromkeys(reasons))
    return m


def _metric_t0(platform, cmd, prep_s, sender_t0_us, sender_reason):
    """(t0 of the metrics in us or NaN, method, reason) per platform (module docstring)."""
    if platform == 'sim':
        if prep_s is None or not math.isfinite(prep_s):
            return NAN, 'none', 'no_prep_s'
        return float(prep_s) * 1e6, 'sim_exact', ''          # fwsim log time = simulation time (s02_cmd_path)
    if platform == 'hil':
        if sender_t0_us is None or not math.isfinite(sender_t0_us):
            return NAN, 'none', 'no_sender_t0' + (f' ({sender_reason})' if sender_reason else '')
        return float(sender_t0_us), 'sender', ''
    if math.isfinite(cmd['t0_us']):
        return float(cmd['t0_us']), cmd['t0_method'], ''
    return NAN, 'none', 'no_t0' + (f" ({cmd['t0_reason']})" if cmd['t0_reason'] else '')


def score_run(log, test_id, platform='', prep_s=None, us_cut=None, plant=None, plant_reason='',
              sender_t0_us=None, sender_reason=''):
    """Every S02 number of one completed run (module docstring "S02 scoring"). log: load_log output; platform sim,
    hil, real (runs.csv); plant: load_plant output or None (plant_reason why); sender_t0_us: HIL sender t = 0 on the
    ESP32 clock (s02_plant_ticks map), sender_reason why not."""
    platform = str(platform).strip().lower()
    if test_id not in s02_cmd_path.GROUP_A:
        raise ValueError(f'test {test_id!r} is not a group A test')
    m = {}
    m['missing_pct'], m['gen_pause_n'] = tick_loss(log)
    m['missing_over'] = int(m['missing_pct'] > MISSING_MAX_PCT) if math.isfinite(m['missing_pct']) else NAN
    m['missing_seq_pct'], m['missing_tail_known'] = log_loss(log)
    ev = log.get('_every', log)
    m.update(run_bad_fields(pos_reset_mid(ev.get('run_active'), ev.get('pos_reset_n'), ev.get('motor_on'))))
    allc = log.get('_all', log)
    cmd = s02_cmd_path.stats(allc, test_id, platform, us_cut, prep_s)
    m.update(cmd)
    m['age_pre_a_ms'] = age_pre(allc, cmd['anchor_a_us'])
    m['age_pre_b_ms'] = age_pre(allc, cmd['anchor_b_us'])
    m['age_pre_flags'] = ';'.join(f'age_pre_{x}_over' for x in ('a', 'b')
                                  if math.isfinite(m[f'age_pre_{x}_ms']) and m[f'age_pre_{x}_ms'] > AGE_PRE_MAX_MS)
    t0, meth, why = _metric_t0(platform, cmd, prep_s, sender_t0_us, sender_reason)
    m['t0_metric_us'], m['t0_metric_method'], m['metric_reason'] = t0, meth, why
    m['t0_lag_ms'] = ((cmd['t0_us'] - t0) / 1000.0 if platform == 'hil' and math.isfinite(t0)
                      and math.isfinite(cmd['t0_us']) else NAN)
    for k in E_METRICS + ('wz_src_enc_frac', 'window_s', 'window_rows'):
        m[k] = NAN
    for k in STEPS:
        m[f'step_{k}'] = STEPS[k]
        m[f'qrms_{k}'] = QRMS[k]
    r_family = platform in PLATFORMS_R and not (platform == 'real' and test_id in LIFTED)
    if r_family:
        for k in (R_LIFTED if test_id in LIFTED else R_FLOOR):
            m[k] = NAN
    m['r_reason'] = ('' if r_family else 'real_lifted' if platform in PLATFORMS_R else 'platform_unknown')
    if not math.isfinite(t0):
        if r_family:
            m['r_reason'] = 'no_t0'
        return m
    ts = np.asarray(log['timestamp_us'], dtype=float)
    t = (ts - t0) * 1e-6
    T = s02_ref.t_cmd(test_id)
    win = (t >= WINDOW_SKIP_S - WINDOW_EPS_S) & (t <= T + WINDOW_EPS_S)
    if np.count_nonzero(win) < 2:
        m['metric_reason'] = 'no_window_rows'
        if r_family:
            m['r_reason'] = 'no_window_rows'
        return m
    r = s02_ref.reference(test_id, t[win])
    m['e_vx'], m['e_vy'] = rms(r[:, 0] - log['vx'][win]), rms(r[:, 1] - log['vy'][win])
    wzm = log.get('wz_meas')
    m['e_wz'] = rms(r[:, 2] - wzm[win]) if wzm is not None and np.all(np.isfinite(wzm[win])) else NAN
    src = log.get('wz_src')
    m['wz_src_enc_frac'] = (float(np.mean(src[win] == WZ_SRC_ENC)) if src is not None and
                            np.all(np.isfinite(src[win])) else NAN)
    w = np.stack([log[c] for c in WHEEL_RPM], axis=1)[win] * 2 * math.pi / 60
    m['rms_w_err'] = float(np.mean([rms(c) for c in (s02_ref.ik(*r.T) - w).T]))
    gz = log.get('gz_fused')
    m['e_yaw_imu'] = rms(r[:, 2] - gz[win]) if gz is not None and np.all(np.isfinite(gz[win])) else NAN
    i0, i1 = np.flatnonzero(win)[[0, -1]]
    for name in ['tick'] + [f'{k}_{c}' for k in ('dpwm', 'sat') for c in ('fl', 'fr', 'rl', 'rr')]:
        x = log[name][i0:i1 + 1]
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) < 0):
            raise ValueError(f'counter {name} not finite or going back inside the metric window')
    dticks = log['tick'][i1] - log['tick'][i0]
    if dticks <= 0:
        raise ValueError('tick does not advance inside the metric window')
    dt_s = (ts[i1] - ts[i0]) * 1e-6
    if dt_s <= 0:
        raise ValueError('timestamp_us does not advance inside the metric window')
    dp = np.array([log[f'dpwm_{c}'][i1] - log[f'dpwm_{c}'][i0] for c in ('fl', 'fr', 'rl', 'rr')])
    sat = np.array([log[f'sat_{c}'][i1] - log[f'sat_{c}'][i0] for c in ('fl', 'fr', 'rl', 'rr')])
    if np.any(sat > dticks):
        raise ValueError('saturated ticks exceed controller ticks inside the metric window')
    m['pwm_tv'] = float(np.mean(dp / PWM_MAX / dt_s))          # P2-3: time on the DUT clock
    m['sat_pct'] = float(np.mean(sat / dticks * 100))
    m['window_s'], m['window_rows'] = float(dt_s), int(np.count_nonzero(win))
    if test_id in STEP_TESTS:
        seg = (t >= -WINDOW_EPS_S) & (t < T - WINDOW_EPS_S)
        m.update(s02_ref.step_quantities(t[seg], log['vx'][seg], test_id))
    if r_family:
        if plant is None:
            m['r_reason'] = plant_reason or 'no_plant_file'
        else:
            m.update(_family_r(log, test_id, t, win, plant, cmd['prep_flags'], platform, t0))
    return m


def run_status(r):
    """(status, reason) of a runs.csv row: ok, or a run that is not completed (module docstring)."""
    res = (r.get('result') or '').strip().lower()
    why = (r.get('reason') or '').strip()
    if (r.get('block_status') or '').strip() == 'battery_rerun':
        return 'battery_rerun', why
    if res == 'replaced':
        return 'replaced', why
    if (r.get('aborted') or '').strip() not in ('', '0') or res == 'aborted':
        return 'aborted', why
    if res == 'broken':
        return 'broken', why
    return 'ok', ''


_UNIT = [('_pct', '%'), ('_ms', 'ms'), ('_us', 'us'), ('_ppm', 'ppm'), ('_s', 's')]
COLUMN_INFO = {
    'e_vx': ('E', 'm/s', 'RMS r_vx - vx (encoder, firmware)'),
    'e_vy': ('E', 'm/s', 'RMS r_vy - vy (encoder, firmware)'),
    'e_wz': ('E', 'rad/s', 'RMS r_wz - wz_meas (source chosen by the controller)'),
    'rms_w_err': ('E', 'rad/s', 'mean over wheels of RMS IK(r) - wheel speed (*_rpm)'),
    'e_yaw_imu': ('E', 'rad/s', 'RMS r_wz - gz_fused (gyro after bias)'),
    'pwm_tv': ('E', '1/s', 'mean over wheels of sum |du| / window time (dpwm counters, DUT clock), u = pwm/1000'),
    'sat_pct': ('E', '%', 'mean over wheels of saturated ticks / ticks (sat counters)'),
    'wz_src_enc_frac': ('info', '1', 'share of window lines with wz_src = encoder'),
    'e_vx_true': ('R', 'm/s', 'RMS r_vx - plant body vx'), 'e_vy_true': ('R', 'm/s', 'RMS r_vy - plant body vy'),
    'e_wz_true': ('R', 'rad/s', 'RMS r_wz - plant wz'),
    'e_p_rms': ('R', 'm', 'RMS |p - p_ref|, test frame'), 'e_p_max': ('R', 'm', 'max |p - p_ref|'),
    'e_psi_rms': ('R', 'rad', 'RMS wrap(psi - psi_ref)'), 'e_psi_max': ('R', 'rad', 'max |wrap(psi - psi_ref)|'),
    'e_est_p_rms': ('R', 'm', 'RMS |firmware pose - plant pose|'), 'e_est_p_max': ('R', 'm', 'max of the same'),
    'e_est_psi_rms': ('R', 'rad', 'RMS wrap(firmware heading - plant heading)'),
    'e_est_psi_max': ('R', 'rad', 'max of the same'),
    'rms_w_err_true': ('R', 'rad/s', 'mean over wheels of RMS IK(r) - plant wheel speed (A-K)'),
    'cmd_age_max_ms': ('cmd', 'ms', 'largest traj_age_ms in the schedule window: LOWER BOUND, not the absolute delay, '
                                    'loss, duplicates or reordering of packets (P2-2b)'),
    'cmd_gap_n': ('cmd', 'count', 'runs of lines with traj_age_ms > 200 ms: LOWER BOUND (P2-2b)'),
    't0_lag_ms': ('time', 'ms', 'HIL: anchor t = 0 minus sender t = 0 (DUT clock)'),
    'missing_pct': ('loss', '%', 'motor-on tick stream: 1 - distinct ticks / tick span (P2-5)'),
    'missing_seq_pct': ('loss', '%', 'log_seq based loss, every line (older number)'),
    'gen_pause_n': ('loss', 'count', 'generation pauses (P2-5)'),
}
NONNEG = set(E_METRICS) | set(R_FLOOR) | set(R_LIFTED)
STEP_COLS = {'rise_s': ('s', 'rise 10 -> 90 % of the step (P2-3)'),
             'overshoot_pct': ('%', 'max(y) - 1, >= 0'), 'frac90': ('1', 'time share with y >= 0.9'),
             'no90': ('flag', 'y never reaches 0.9'), 'rise_uninterp': ('flag', 'e_xl >= 10 % |A|: rise not read'),
             'e_xl': ('m/s', '|mean v over the last 1 s of the segment - segment level|')}


def column_info(k):
    """{family, unit, kind, source, nonneg} of a metrics.csv column (metrics_columns.json)."""
    import re
    mm = re.match(r'^(rise_s|overshoot_pct|frac90|no90|rise_uninterp|e_xl)(_true)?_seg(\d+)$', k)
    if mm:
        unit, src = STEP_COLS[mm.group(1)]
        flag = unit == 'flag'
        return dict(family='R' if mm.group(2) else 'E', unit='1' if flag else unit,
                    kind='flag' if flag else 'metric',
                    source=('fk(plant wheel speeds) ' if mm.group(2) else 'firmware vx ') + src
                    + f', segment {mm.group(3)}',
                    nonneg=mm.group(1) in ('rise_s', 'overshoot_pct', 'e_xl'))
    if k in COLUMN_INFO:
        fam, unit, src = COLUMN_INFO[k]
        return dict(family=fam, unit=unit, kind='metric' if fam in ('E', 'R') else 'info', source=src,
                    nonneg=k in NONNEG)
    unit = next((u for suf, u in _UNIT if k.endswith(suf)), '')
    return dict(family='info', unit=unit, kind='info', source='', nonneg=False)


def _plant_inputs(d, r, path, platform):
    """(plant or None, plant_reason, sender_t0_us or None, sender_reason) of one runs.csv row."""
    if platform not in PLATFORMS_R:
        return None, '', None, ''
    pf = (r.get('plant_file') or '').strip()
    pp = os.path.join(d, pf) if pf else os.path.splitext(path)[0] + '_plant.csv'
    preason, sender, sreason = '', None, ''
    if platform == 'hil':
        mp = os.path.splitext(pp)[0] + '_map.json'
        if not os.path.isfile(mp):
            return None, 'no_map_file', None, 'no_map_file'
        with open(mp, encoding='utf-8') as f:
            mj = json.load(f)
        if mj.get('ok') is not True:
            why = f"map_{mj.get('reason') or 'not_ok'}"
            return None, why, None, why
        s = mj.get('sender_t0_us')
        sender = float(s) if isinstance(s, (int, float)) and math.isfinite(float(s)) else None
        sreason = '' if sender is not None else 'no_sender_frame'
    if platform == 'real':
        vj = os.path.splitext(pp)[0] + '_video.json'       # tools/s02_video_metrics.py
        if not os.path.isfile(vj):
            return None, 'no_video_file', None, ''
        with open(vj, encoding='utf-8') as f:
            vi = json.load(f)
        if vi.get('ok') is not True:
            return None, str(vi.get('reason') or 'video_not_ok'), None, ''
        # S2V1-3 (Codex round 1): the json vouches for this log and these two files only
        cp = os.path.splitext(pp)[0] + '_cam.csv'
        for what, f, key in (('log', path, ('inputs', 'log_sha256')), ('plant', pp, ('plant_sha256',)),
                             ('cam', cp, ('cam_sha256',))):
            want = vi
            for k in key:
                want = want.get(k) if isinstance(want, dict) else None
            if not os.path.isfile(f):
                return None, f'no_{what}_file', None, ''
            if not want or _sha256(f) != want:
                return None, f'video_mismatch ({what})', None, ''
        plant = load_plant(pp)
        cam = load_plant(cp)
        cam['t_miss_s'] = float(vi['t_miss_s'])
        plant['_cam'] = cam
        return plant, preason, sender, sreason
    if not os.path.isfile(pp):
        return None, 'no_plant_file', sender, sreason
    return load_plant(pp), preason, sender, sreason


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run_dir_metrics(d, out_name='metrics.csv'):
    """metrics.csv (one line per runs.csv row) and <stem>_columns.json of a run folder (module docstring)."""
    with open(os.path.join(d, 'runs.csv'), newline='') as f:
        runs = list(csv.DictReader(f))
    out = []
    for r in runs:
        ids = {k: r.get(k, '') for k in ID_FIELDS}
        st, why = run_status(r)
        if st != 'ok':
            out.append(dict(ids, status=st, status_reason=why))       # P2-5: never dropped silently
            continue
        platform = r['platform'].strip().lower()
        path = os.path.join(d, r['file'])
        if platform == 'hil':
            check_no_udp_taken(path + '.events.txt')
        log = load_log(path)
        ps = (r.get('prep_s') or '').strip()
        prep_s = float(ps) if ps else None
        plant, preason, sender, sreason = _plant_inputs(d, r, path, platform)
        m = score_run(log, r['test_id'], platform, prep_s, s02_cmd_path.us_cut_ticks(path + '.events.txt'),
                      plant, preason, sender, sreason)
        # S2M1-1 (Codex round 1 of v3): a completed run with loss over 1 % (P2-5) or a pose reset mid-run (run_bad,
        # L37) is a broken run: status broken, no metric; its loss and reset columns stay (P2-5 reports them)
        broken = []
        if m.get('missing_over') == 1:
            broken.append(f"missing_over ({m['missing_pct']:.3g} % > {MISSING_MAX_PCT:g} %, P2-5)")
        if m.get('run_bad') == 1:
            broken.append(m.get('run_bad_reason') or 'run_bad')
        if broken:
            out.append(dict(ids, status='broken', status_reason='; '.join(broken),
                            **{k: m[k] for k in BROKEN_KEEP if k in m}))
            continue
        out.append(dict(ids, status='ok', status_reason='', **m))
    keys = list(ID_FIELDS) + list(STATUS_FIELDS)
    for o in out:
        keys += [k for k in o if k not in keys]

    def fmt(k, x):
        # S2P1-13 (Codex round 1): a timestamp in us (t0_us, anchor_*_us, edge_us, sim_t0_err_us) keeps every
        # digit (6 significant digits lose tens of ms after hours of uptime)
        if isinstance(x, (float, np.floating)) and k.endswith('_us'):
            return f'{x:.0f}' if math.isfinite(x) else 'nan'
        return f'{x:.6g}' if isinstance(x, (float, np.floating)) else x
    with open(os.path.join(d, out_name), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows({k: fmt(k, o.get(k, '')) for k in keys} for o in out)
    with open(os.path.join(d, os.path.splitext(out_name)[0] + '_columns.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(version='s02_metrics_v3', columns={k: column_info(k) for k in keys}), f, indent=1)
    return out


def cell_counts(out):
    """{(test_id, platform, controller): {status: n}} of run_dir_metrics output (P2-5: the cell reports n_ok / 5)."""
    cells = {}
    for o in out:
        key = (o.get('test_id', ''), o.get('platform', ''), o.get('controller', ''))
        c = cells.setdefault(key, {})
        c[o['status']] = c.get(o['status'], 0) + 1
    return cells


HISTORY = ('replaced', 'battery_rerun')     # kept in the history, outside the 5 runs of the cell (P2-4, P2-5)


def map_refused_reason(o):
    """Short reason of a completed (ok) HIL run whose plant map gave no t = 0 (metric_reason no_sender_t0: map not
    certified, e.g. map_apply_window_too_wide, no map file, no sender frame); '' for any other row. Spec S02 v1.16
    S8 (P2-5): such a run is not an outside cause, it stays in the 5 and the cell reports how many, per reason."""
    if o.get('status') != 'ok' or str(o.get('platform', '')).strip().lower() != 'hil':
        return ''
    mr = str(o.get('metric_reason') or '')
    if not mr.startswith('no_sender_t0'):
        return ''
    inner = mr[len('no_sender_t0'):].strip()
    if inner.startswith('(') and inner.endswith(')'):
        inner = inner[1:-1]
    return inner.split(' ')[0] or 'no_sender_t0'


def cell_map_refused(out):
    """{(test_id, platform, controller): {reason: n}} of the ok HIL runs without a map t = 0 (map_refused_reason)."""
    cells = {}
    for o in out:
        why = map_refused_reason(o)
        if why:
            c = cells.setdefault((o.get('test_id', ''), o.get('platform', ''), o.get('controller', '')), {})
            c[why] = c.get(why, 0) + 1
    return cells


def cell_line(key, c, refused=None):
    """Printed line of one cell (S2M1-9, Codex round 1 of v3): completed runs over the fixed N_REP (P2-4, P2-5:
    'n_hoan_tat / 5'), the broken and aborted runs, the ok HIL runs whose map was refused, per reason (refused:
    {reason: n} of cell_map_refused; spec v1.16 S8), then the history rows (replaced, battery_rerun) apart."""
    tid, plat, ctrl = key
    in_cell = [k for k in sorted(c) if k not in HISTORY and k != 'ok']
    hist = [k for k in HISTORY if c.get(k)]
    s = f'  {tid} {plat} {ctrl}: ok {c.get("ok", 0)} / {N_REP}'
    if refused:
        s += (f' (map refused {sum(refused.values())}: '
              + ', '.join(f'{k} {refused[k]}' for k in sorted(refused)) + ')')
    s += ''.join(f', {k} {c[k]}' for k in in_cell)
    if hist:
        s += '; history: ' + ', '.join(f'{k} {c[k]}' for k in hist)
    return s


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    name = sys.argv[2] if len(sys.argv) == 3 else 'metrics.csv'
    for nm in (name, os.path.splitext(name)[0] + '_columns.json'):
        if os.path.exists(os.path.join(sys.argv[1], nm)):
            sys.exit(f'{nm} already exists in {sys.argv[1]}: choose a new name (outputs are not overwritten).')
    res = run_dir_metrics(sys.argv[1], name)
    print(f'{len(res)} runs -> {os.path.join(sys.argv[1], name)}')
    refused = cell_map_refused(res)
    for key, c in sorted(cell_counts(res).items()):
        print(cell_line(key, c, refused.get(key)))
    bad = [r['run_id'] for r in res if r.get('run_bad') == 1]
    if bad:
        print(f'BROKEN RUNS (run_bad = 1, pose reset mid-run, L37): broken run, inside the 5 of its cell, not run '
              f'again (P2-4): {", ".join(bad)}')
    over = [r['run_id'] for r in res if r.get('missing_over') == 1]
    if over:
        print(f'LOSS OVER {MISSING_MAX_PCT:g} % (P2-5, broken run unless the logger tool failed): {", ".join(over)}')
