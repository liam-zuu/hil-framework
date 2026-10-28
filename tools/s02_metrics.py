"""
s02_metrics.py: spec S02 P2-3 metrics, one script for SIM (fwsim), HIL and REAL.

Input: logs in the ESP32 logger format (tools/agv_logger.py COLUMNS, firmware
AGV_MECANUM a6644a5 or later: 28 columns + tick, dpwm_*, sat_*). Every quantity
is the firmware's own: vx/vy/wz from encoder_get_rpm over 10 ms, pwm from
vel_ctrl, and the cumulative counters of vel_ctrl, so dropped log lines do not
change the PWM metrics.

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
import math
import os
import sys

import numpy as np

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
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    head = rows[0]
    data = np.array([[float(x) if x != '' else np.nan for x in r] for r in rows[1:] if len(r) == len(head)])
    seq_all = seq_on = None
    if 'log_seq' in head:
        seq_all = data[:, head.index('log_seq')]
    if 'motor_on' in head:
        # AGV ctrl-imu: 10 Hz lines while the motor is off carry no controller tick: drop them
        mo = data[:, head.index('motor_on')]
        data = data[np.isnan(mo) | (mo == 1)]
    log = {h: data[:, i] for i, h in enumerate(head)}
    if seq_all is not None and np.isfinite(seq_all).any():
        log['_seq_all'] = seq_all[np.isfinite(seq_all)]          # every line, motor off ones too
        log['_run_end_last'] = read_run_end(str(path) + '.events.txt')   # path may be a pathlib.Path
    return log


def read_run_end(events_path):
    """last_data_seq of every '#run_end' line agv_logger.py kept next to the CSV (empty if none)."""
    out = []
    if os.path.isfile(events_path):
        with open(events_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                for part in line.strip().split(','):
                    if part.startswith('last_data_seq='):
                        out.append(int(part.split('=', 1)[1]))
    return sorted(set(out))


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
    dticks = tick[i1] - tick[i0]
    dp = np.array([log[f'dpwm_{c}'][i1] - log[f'dpwm_{c}'][i0] for c in ('fl', 'fr', 'rl', 'rr')])
    sat = np.array([log[f'sat_{c}'][i1] - log[f'sat_{c}'][i0] for c in ('fl', 'fr', 'rl', 'rr')])
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
