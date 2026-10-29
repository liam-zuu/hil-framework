"""
s02_metrics.py: spec S02 P2-3 metrics, one script for SIM (fwsim), HIL and REAL.

Input: logs in the ESP32 logger format (tools/agv_logger.py COLUMNS, firmware
AGV_MECANUM a6644a5 or later: 28 columns + tick, dpwm_*, sat_*). Every quantity
is the firmware's own: vx/vy/wz from encoder_get_rpm over 10 ms, pwm from
vel_ctrl, and the cumulative counters of vel_ctrl, so dropped log lines inside
the window do not change the PWM metrics; losing a line at either end of the
window moves the window itself (B0 review round 3, question 3).

Formats (B0 review round 3 B51/N9), told apart by the header and the rows:
  DUT     66 columns, every row with log_seq and motor_on (AGV ctrl-imu firmware):
          log_seq U32, motor_on exactly 0 or 1, timestamp_us an integer
  SIM     66 columns, log_seq and motor_on empty in every row (fwsim/s02_groupA_sim.py)
  legacy  the first 37 or 28 columns (older firmware)
Anything else, or rows of two formats, is refused. SIM and legacy rows stay in file
order and timestamp_us must be finite and never go back.
A DUT log next to the logger's <log>.segments.csv must also have <log>.summary.json
(status closed) whose 'files' manifest matches the CSV and the segment file byte for
byte; its segment must be 'consistent' in the summary, and only the '#run_end' groups
the logger assigned to that segment count (B47-B50, N5, N7, N8). A DUT log without a
segment file is checked from its rows and the '#run_end' lines of <log>.events.txt
(same parser as the logger); a footer whose slots hold data lines is refused.

Per run (window: 1 s after the first non-zero setpoint to the last non-zero
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

  python tools/s02_metrics.py <run_dir> [out_name]   # reads runs.csv, writes metrics.csv (or out_name)
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agv_logger import (COLUMNS, MAX_US_PER_SEQ, N_OLD, N_S02, RATE_SLACK_US,  # noqa: E402
                        SEGMENT_COLUMNS, SEGMENT_STATUSES, file_digest, parse_run_end)

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
    if head not in (COLUMNS, COLUMNS[:N_S02], COLUMNS[:N_OLD]):
        raise ValueError(f'{path}: header is not a logger format (agv_logger.COLUMNS, 66, 37 or 28 columns)')
    bad = [i + 2 for i, r in enumerate(body) if len(r) != len(head)]
    if bad:
        raise ValueError(f'{path}: lines {bad[:5]} do not have {len(head)} columns (mixed formats)')
    t = head.index('timestamp_us')
    kind = 'legacy'
    if head == COLUMNS:
        s, mo = head.index('log_seq'), head.index('motor_on')
        kinds = {('dut' if r[s] != '' and r[mo] != '' else
                  'sim' if r[s] == '' and r[mo] == '' else 'bad') for r in body}
        if 'bad' in kinds or len(kinds) > 1:
            raise ValueError(f'{path}: log mixes lines with and without log_seq / motor_on (mixed formats)')
        kind = kinds.pop() if kinds else 'sim'
    run_end_last = None
    if kind == 'dut':
        body, run_end_last = dut_rows(body, path)
    else:
        body = _old_format(body, t)
    data = np.array([[float(x) if x != '' else np.nan for x in r] for r in body])
    log_seq_all = None
    if kind == 'dut':
        log_seq_all = data[:, head.index('log_seq')]
        # 10 Hz lines while the motor is off carry no controller tick: drop them
        data = data[data[:, head.index('motor_on')] == 1]
    log = {h: data[:, i] for i, h in enumerate(head)}
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
    accepts (B50/N5: the same parser as the logger; empty if none)."""
    out = set()
    if os.path.isfile(events_path):
        with open(events_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                i = line.find('#run_end,')
                if i < 0:
                    continue
                kv = parse_run_end(line[i:].strip())
                if kv is not None:
                    out.add(kv['last_data_seq'])
    return sorted(out)


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

    moving = np.flatnonzero(np.any(np.abs(sp) > 1e-9, axis=1))
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


def run_dir_metrics(d, out_name='metrics.csv'):
    with open(os.path.join(d, 'runs.csv'), newline='') as f:
        runs = list(csv.DictReader(f))
    out = []
    for r in runs:
        if r.get('aborted') not in (None, '', '0'):
            continue
        m = compute(load_log(os.path.join(d, r['file'])), r['test_id'])
        out.append(dict(run_id=r['run_id'], platform=r['platform'], controller=r['controller'],
                        test_id=r['test_id'], speed_mps=r['speed_mps'], rep=r['rep'], **m))
    keys = []
    for o in out:
        keys += [k for k in o if k not in keys]
    fmt = lambda x: f'{x:.6g}' if isinstance(x, float) else x   # noqa: E731
    with open(os.path.join(d, out_name), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows({k: fmt(o.get(k, '')) for k in keys} for o in out)
    return out


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    name = sys.argv[2] if len(sys.argv) == 3 else 'metrics.csv'
    if os.path.exists(os.path.join(sys.argv[1], name)):
        sys.exit(f'{name} already exists in {sys.argv[1]}: choose a new name (outputs are not overwritten).')
    res = run_dir_metrics(sys.argv[1], name)
    print(f'{len(res)} runs -> {os.path.join(sys.argv[1], name)}')
