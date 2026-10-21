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
  missing_pct        1 - logged lines / ticks spanned (whole log)            %
Lifted tests A-K1, A-K2, per setpoint step whose new level lasts >= 2 s:
  rise_s_segN (10-90 % of the step), overshoot_pct_segN (beyond the new level,
  % of the step size), e_ss_segN (mean setpoint - vx over the last 1 s).

  python tools/s02_metrics.py <run_dir>        # reads runs.csv, writes metrics.csv
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


def load_log(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    head = rows[0]
    data = np.array([[float(x) if x != '' else np.nan for x in r] for r in rows[1:] if len(r) == len(head)])
    return {h: data[:, i] for i, h in enumerate(head)}


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
    m['missing_pct'] = float((1 - len(tick) / (tick[-1] - tick[0] + 1)) * 100)

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
    if test_id in STEP_TESTS:
        for n, s in enumerate(step_segments(t, sp[:, 0], v[:, 0]), start=1):
            m[f'step_seg{n}'] = f"{s['from_']:+.2f}->{s['to']:+.2f}"
            m[f'rise_s_seg{n}'], m[f'overshoot_pct_seg{n}'], m[f'e_ss_seg{n}'] = s['rise_s'], s['overshoot_pct'], s['e_ss']
    return m


def run_dir_metrics(d):
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
    with open(os.path.join(d, 'metrics.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows({k: fmt(o.get(k, '')) for k in keys} for o in out)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    res = run_dir_metrics(sys.argv[1])
    print(f'{len(res)} runs -> {os.path.join(sys.argv[1], "metrics.csv")}')
