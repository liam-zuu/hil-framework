#!/usr/bin/env python3
"""
fault_c7_table.py: the type II table of fault injection batch 1 (claim C7, record only, no pass/fail;
L82, design 03_CODE/docs/2026-11-14_Thiet_ke_bom_loi_dot1_v1.md of the thesis repo, section 8.3; spec S06 T8
"Loai II").

  python tools/fault_c7_table.py <run folder> [--out <run folder>/c7_table.csv]

The run folder holds one sub-folder per (controller, case), as fwsim/fault_run.py writes them: <MODE>_<CASE>,
with <MODE>_base the baseline (fault_run.py --baseline) of each controller. Each sub-folder has log.csv,
faults.json, faults_tick.csv, faults_imu.csv, meta.yaml.

One row per (controller, fault case), each against the baseline of the same controller:
  * the fault window [start, end) from faults.json (several rows: first start, last end) and complete;
  * the run was the same as its baseline but for the schedule: same task, mode, seed, IMU config, ground,
    firmware and hil commit of the DLL (meta.yaml config / firmware), else the folder cannot be tabled (exit 2);
  * t_src_enc: first firmware tick at or after start where wz_src = 0 (wz from the encoders), t_src_imu: the
    first tick after it where wz_src = 1 again; t_slip_on / t_slip_off, t_ws_on / t_ws_off: first rising edge
    of slip_flag / ws_flag at or after start (a flag already on at start counts as a rise there), then the
    first falling edge after it (None: no edge); base_t_src_enc, base_t_slip_on, base_t_ws_on: the same in the
    baseline (an edge there too is not the fault's);
  * age_g_max_ms: largest gyro sample age inside the window; r_max: largest |r| (IMU - encoder wz residual,
    the firmware's) inside the window, r_th the firmware's threshold;
  * err_max_<ax> and base_err_max_<ax>: largest |true body velocity - setpoint| over [start, end + 1 s), the
    fault run and the baseline (spec); err_rms_<ax>, base_err_rms_<ax>: its RMS over the same span (a
    setpoint step inside the span, as A-W1's +1 -> -1 rad/s at 4 s, sets the largest error of both runs
    alike; the RMS shows the fault); sat_frac, base_sat_frac: share of tick-wheel samples with |pwm| >= PWM_MAX
    (1000, motor.h) over the same span (as fwsim.metrics);
  * end_dx, end_dy, end_dtheta: pose at the last tick minus the baseline's (theta wrapped to [-pi, pi));
  * r_true_max: largest |r_true| (plant wz from wheels - body wz) over the whole run (controller-IMU spec 6.1:
    a gyro_scale case is read with |r_true| < r_th / 2);
  * ticks, imu_rows, imu_dropped, base_imu_rows: number of firmware ticks (log.csv rows), of IMU report rows
    (faults_imu.csv; one per report, written before an imu_loss window drops it), of those dropped (mask bit
    M_LOSS), and of the baseline's rows; ticks_expected = duration / 10 ms (design 10.1 step 5).
Exit 0 = table written, 2 = the folder cannot be tabled.

Assumptions (stated by the author, to be checked): log.csv columns as fwsim.write_run writes them (IMU_COLS,
then WS_COLS when the firmware has wheel_slip.c); the tick time column t is the time of the tick in s.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

PWM_MAX = 1000                      # AGV src/motor/motor.h PWM_MAX
TAIL_S = 1.0                        # errors over the window and 1 s after it (spec S06 T8, type II)
TICK_S = 0.010                      # firmware tick (FW_DT_US)
M_LOSS = 16                         # fwsim/fault_run.py mask bit of imu_loss (faults_imu.csv)
SAME_KEYS = ('task', 'mode', 'seed', 'imu_rate', 'imu_boot_reset', 'imu_noise', 'imu_noise_seed', 'ground', 'mu',
             'no_imu', 'duration')
AXES = ('vx', 'vy', 'wz')


class TableError(Exception):
    pass


def read_log(path):
    with open(path, newline='') as fh:
        head = fh.readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if data.shape[1] != len(head):
        raise TableError(f'{path}: {data.shape[1]} columns, header has {len(head)}')
    return {h: data[:, i] for i, h in enumerate(head)}


def read_meta(path):
    """config and firmware lines of a fwsim meta.yaml (each written as one JSON object by fwsim.write_run)."""
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            for key in ('config:', '  esp32:', '  hil_framework:', 'faults:'):
                if line.startswith(key):
                    out[key.strip(' :')] = json.loads(line[len(key):].strip())
    for key in ('config', 'esp32', 'hil_framework'):
        if key not in out:
            raise TableError(f'{path}: no {key} line')
    return out


def load_case(folder):
    log = read_log(os.path.join(folder, 'log.csv'))
    with open(os.path.join(folder, 'faults.json')) as fh:
        faults = json.load(fh)
    meta = read_meta(os.path.join(folder, 'meta.yaml'))
    with open(os.path.join(folder, 'faults_imu.csv'), newline='') as fh:
        masks = [int(float(r['mask'])) for r in csv.DictReader(fh)]
    # fault_run.M_LOSS: the report of this row was dropped (imu_loss window)
    return dict(log=log, faults=faults, meta=meta, imu_rows=len(masks), folder=folder,
                imu_dropped=sum(1 for m in masks if m & M_LOSS))


def same_run(a, b):
    """Why a and b are not the same run but for the schedule, or None."""
    ca, cb = a['meta']['config'], b['meta']['config']
    for k in SAME_KEYS:
        if ca.get(k) != cb.get(k):
            return f'config {k}: {ca.get(k)!r} != {cb.get(k)!r}'
    for k in ('esp32', 'hil_framework'):
        if (a['meta'][k] or {}).get('commit') != (b['meta'][k] or {}).get('commit'):
            return f'{k} commit {a["meta"][k]} != {b["meta"][k]}'
    return None


def _first(t, cond, after):
    k = np.nonzero(cond & (t >= after - 1e-9))[0]
    return float(t[k[0]]) if len(k) else None


def edges(t, flag, start):
    """(first rising edge at or after start, first falling edge after it); None where there is none. A flag
    already on at start counts as risen at the first tick at or after start."""
    i0 = np.nonzero(t >= start - 1e-9)[0]
    if not len(i0):
        return None, None
    tt, on = t[i0[0]:], flag[i0[0]:] >= 0.5
    rise = on & ~np.r_[False, on[:-1]]               # the first tick counts as a rise when the flag is on
    k_on = np.nonzero(rise)[0]
    if not len(k_on):
        return None, None
    k_off = np.nonzero(~on[k_on[0]:])[0]
    return float(tt[k_on[0]]), (float(tt[k_on[0] + k_off[0]]) if len(k_off) else None)


def window(faults):
    rows = faults.get('rows') or []
    if not rows:
        raise TableError('faults.json has no rows')
    return min(r['start_ms'] for r in rows) / 1000.0, max(r['end_ms'] for r in rows) / 1000.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def row_for(mode, case, f, base):
    lf, lb = f['log'], base['log']
    t = lf['t']
    if len(t) != len(lb['t']) or not np.allclose(t, lb['t']):
        raise TableError(f'{f["folder"]}: tick times differ from the baseline')
    start, end = window(f['faults'])
    inw = (t >= start - 1e-9) & (t < end - 1e-9)
    span = (t >= start - 1e-9) & (t < end + TAIL_S - 1e-9)
    if not inw.any():
        raise TableError(f'{f["folder"]}: no tick inside the window')
    t_enc = _first(t, lf['wz_src'] < 0.5, start)
    t_imu = _first(t, lf['wz_src'] >= 0.5, t_enc + 1e-9) if t_enc is not None else None
    s_on, s_off = edges(t, lf['slip_flag'], start)
    out = dict(controller=mode, case=case,
               faults=';'.join(f'{r["fault"]}:{r["target"]}:{r.get("params") or {}}' for r in f['faults']['rows']),
               win_start_s=start, win_end_s=end, complete=bool(f['faults'].get('complete')),
               t_src_enc=t_enc, t_src_imu=t_imu, t_slip_on=s_on, t_slip_off=s_off)
    if 'ws_flag' in lf:
        out['t_ws_on'], out['t_ws_off'] = edges(t, lf['ws_flag'], start)
    else:
        out['t_ws_on'] = out['t_ws_off'] = None
    # the same edges in the baseline: an edge there too is not the fault's
    out['base_t_src_enc'] = _first(t, lb['wz_src'] < 0.5, start)
    out['base_t_slip_on'] = edges(t, lb['slip_flag'], start)[0]
    out['base_t_ws_on'] = edges(t, lb['ws_flag'], start)[0] if 'ws_flag' in lb else None
    out.update(age_g_max_ms=float(lf['age_g_ms'][inw].max()), r_max=float(np.abs(lf['r'][inw]).max()),
               r_th=float(lf['r_th'][-1]))
    for ax in AXES:
        ef, eb = lf[f'v_{ax}'][span] - lf[f'sp_{ax}'][span], lb[f'v_{ax}'][span] - lb[f'sp_{ax}'][span]
        out[f'err_max_{ax}'] = float(np.abs(ef).max())
        out[f'base_err_max_{ax}'] = float(np.abs(eb).max())
        out[f'err_rms_{ax}'] = float(np.sqrt(np.mean(ef ** 2)))
        out[f'base_err_rms_{ax}'] = float(np.sqrt(np.mean(eb ** 2)))
    pw = lambda lg: np.abs(np.column_stack([lg[f'pwm_{w}'][span] for w in ('fl', 'fr', 'rl', 'rr')]))
    out['sat_frac'] = float(np.mean(pw(lf) >= PWM_MAX))
    out['base_sat_frac'] = float(np.mean(pw(lb) >= PWM_MAX))
    out.update(end_dx=float(lf['x'][-1] - lb['x'][-1]), end_dy=float(lf['y'][-1] - lb['y'][-1]),
               end_dtheta=float(wrap(lf['theta'][-1] - lb['theta'][-1])),
               r_true_max=float(np.abs(lf['r_true']).max()), ticks=int(len(t)), imu_rows=int(f['imu_rows']),
               imu_dropped=int(f['imu_dropped']), base_imu_rows=int(base['imu_rows']),
               ticks_expected=int(round(float(f['meta']['config']['duration']) / TICK_S)))
    return out


def table(run_dir):
    subs = sorted(d for d in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, d, 'log.csv')))
    if not subs:
        raise TableError(f'{run_dir}: no <MODE>_<CASE> sub-folder with log.csv')
    cases = {}
    for d in subs:
        mode, sep, case = d.partition('_')
        if not sep or not case:
            raise TableError(f'{d}: not <MODE>_<CASE>')
        cases[(mode, case)] = load_case(os.path.join(run_dir, d))
    rows = []
    for (mode, case), f in sorted(cases.items()):
        if case == 'base':
            if f['faults'].get('rows'):
                raise TableError(f'{f["folder"]}: the baseline has schedule rows')
            continue
        base = cases.get((mode, 'base'))
        if base is None:
            raise TableError(f'{mode}_{case}: no {mode}_base baseline')
        why = same_run(f, base)
        if why:
            raise TableError(f'{mode}_{case} vs {mode}_base: {why}')
        rows.append(row_for(mode, case, f, base))
    if not rows:
        raise TableError(f'{run_dir}: no fault case')
    return rows


def write_csv(rows, path):
    keys = list(rows[0])
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if v is None else round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None, help='default: <run_dir>/c7_table.csv')
    a = ap.parse_args(argv)
    try:
        rows = table(a.run_dir)
    except (TableError, OSError, ValueError, KeyError) as exc:
        print(f'fault_c7_table: cannot table: {exc}', file=sys.stderr)
        return 2
    out = a.out or os.path.join(a.run_dir, 'c7_table.csv')
    write_csv(rows, out)
    print(f'fault_c7_table: {len(rows)} rows -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
