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

Bench folders (HIL bench, a tools/hil_bench_session.py session: session.json at the top; design 10.2 step 6):
the runs are session.json runs, named <MODE>_<CASE> as above; each has frames/<run>_frames.npz (Pi frame
log: mosi, mosi_true when --faults, manifest with lifecycle and faults), pi/<run>.csv (ESP32 WiFi log,
agv_logger) and pi/<run>.csv.events.txt (#gain line); bench_summary.json and, for a fault run,
fault_check_<run>.json. The same columns as the fwsim table, from two clocks:
  * Pi time: t = (frame step - t0) / 1 kHz, t0 = the first RUN step (lifecycle; = manifest faults t0_step, the
    origin of the schedule). True plant state from mosi_true (mosi in the baseline, which has no faults):
    body velocity = world velocity rotated by -theta (fwsim.body_velocity), setpoint = the MOSI reference the
    Pi sent; err_*, end pose (frame t0 + duration, duration = the task's + TAIL_S, as fault_run.py), r_true_max
    over [0, duration) (params.MecanumParams H_fwd, as fwsim). The span is taken on the 1 ms frames, the fwsim
    table on 10 ms ticks.
  * ESP32 time: t = (tick - tick of the first row with run_active = 1) x 10 ms, the firmware's tick counter
    from its own start of the run (it follows the SWA bit the Pi sets at t0 through the H7 and iBUS: some ms
    late, not measured on a common clock; ref_first_s and esp_sp_first_s show both clocks at the task's first
    setpoint step); only the rows of [0, duration) are read, as the fwsim run ends at duration (the ESP32 log
    goes on through the motor-off switch after it). The tick, not timestamp_us (the log time, some ms of
    jitter), so the grid is the 10 ms of the fwsim table. wz_src, slip_flag, ws_flag edges, age_g_max_ms,
    r_max, sat_frac (|pwm| >= PWM_MAX of the ESP32 rows) and ticks (ESP32 rows in [0, duration)) come from it.
  * Left blank on the bench (column blank lists them): r_th (no column in the ESP32 CSV); imu_rows,
    imu_dropped, base_imu_rows (the H7 builds the IMU reports, the Pi logs no report; the firmware's count of
    gyro reports received in [0, duration) is esp_gyro_n, base_esp_gyro_n).
  * More columns: run_s (RUN length, lifecycle), tick_gaps (ticks of [0, duration) without an ESP32 row),
ref_first_s (first change of the MOSI reference, Pi time), esp_sp_first_s (first change
    of sp_vx, sp_vy, sp_wz in the ESP32 rows, ESP32 time).
  * Same run but for the schedule: session.json run args equal without --faults / --step-log, same ESP32 and
    H7 #fw, same manifest imu_source, imu_noise, plant_src, proto, same #gain line (without first_seq,
    line_drop, boot), else exit 2; a run that bench_summary.json does not call clean, or a fault run whose
    fault_check_<run>.json is missing or not ok, cannot be tabled either.

Assumptions (stated by the author, to be checked): log.csv columns as fwsim.write_run writes them (IMU_COLS,
then WS_COLS when the firmware has wheel_slip.c); the tick time column t is the time of the tick in s. Bench:
the frame log has one frame per step from manifest step_first; the ESP32 columns are those of the 86-column
agv_logger header used here (run_active, tick, wz_src, slip_flag, ws_flag, age_g_ms, r, gyro_n, pwm_*, sp_*).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
import protocol as P                # noqa: E402  (frame layout, decode_state / decode_traj_ref)

PWM_MAX = 1000                     # AGV src/motor/motor.h PWM_MAX
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
    if os.path.isfile(os.path.join(run_dir, 'session.json')):
        return bench_table(run_dir)
    subs =sorted(d for d in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, d, 'log.csv')))
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


# ── HIL bench folders (hil_bench_session.py sessions) ──

BENCH_BLANK = ('r_th', 'imu_rows', 'imu_dropped', 'base_imu_rows')    # no bench source (docstring)
ESP_COLS = ('timestamp_us', 'tick', 'run_active', 'wz_src', 'slip_flag', 'ws_flag', 'age_g_ms', 'r', 'gyro_n',
            'pwm_fl', 'pwm_fr', 'pwm_rl', 'pwm_rr', 'sp_vx', 'sp_vy', 'sp_wz')
GAIN_SKIP = ('first_seq', 'line_drop', 'boot')                        # differ run to run, not the gains
ARG_SKIP = ('--faults', '--step-log')                                 # the schedule and its own step log
MAN_SAME = ('imu_source', 'imu_noise', 'plant_src', 'proto', 'frame_len', 'plant')


def _run_args(args):
    """The run_hil.py arguments of a session run without ARG_SKIP and their values."""
    toks, out, i = args.split(), [], 0
    while i < len(toks):
        if toks[i] in ARG_SKIP:
            i += 2
            continue
        if not any(toks[i].startswith(a + '=') for a in ARG_SKIP):
            out.append(toks[i])
        i += 1
    return out


def task_duration(args):
    """(task id, duration [s]) of a bench run: the --test task of rpi5/fault_tasks.py + TAIL_S (= the RUN
    then STOPPING of --finite-run, and fault_run.py's task + TAIL_S)."""
    toks = args.split()
    if '--test' not in toks or toks.index('--test') + 1 >= len(toks):
        raise TableError(f'run args {args!r}: no --test task')
    task = toks[toks.index('--test') + 1]
    import fault_tasks
    if task not in fault_tasks.TASKS:
        raise TableError(f'--test {task}: not a fault task of rpi5/fault_tasks.py ({sorted(fault_tasks.TASKS)})')
    return task, fault_tasks.TASKS[task][0] + TAIL_S


def read_esp(path):
    """ESP32 CSV (agv_logger) as {column: float array}; every ESP_COLS column must be there."""
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise TableError(f'{path}: no data row')
    head, body = rows[0], rows[1:]
    bad = [i for i, r in enumerate(body) if len(r) != len(head)]
    if bad:
        raise TableError(f'{path}: row {bad[0] + 2} has {len(body[bad[0]])} fields, header {len(head)}')
    miss = [c for c in ESP_COLS if c not in head]
    if miss:
        raise TableError(f'{path}: no column {miss}')
    data = np.array(body, dtype=float)
    return {h: data[:, i] for i, h in enumerate(head)}


def read_gain(path):
    """The #gain line of an agv_logger events file as a dict without GAIN_SKIP; every copy must agree."""
    seen = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if '#gain,' in line:
                kv = dict(p.split('=', 1) for p in line.strip().split('#gain,', 1)[1].split(',') if '=' in p)
                kv = {k: v for k, v in kv.items() if k not in GAIN_SKIP}
                if kv not in seen:
                    seen.append(kv)
    if len(seen) != 1:
        raise TableError(f'{path}: {len(seen)} different #gain lines (want 1)')
    return seen[0]


def load_bench_run(run_dir, srun, clean):
    name = srun['name']
    if not clean.get(name):
        raise TableError(f'{name}: not clean in bench_summary.json')
    z = np.load(os.path.join(run_dir, 'frames', f'{name}_frames.npz'))
    man = json.loads(str(z['manifest']))
    trans = (man.get('lifecycle') or {}).get('transitions') or []
    steps = {}
    for tr in trans:
        steps.setdefault(tr['phase'], int(tr['step']))
    if 'RUN' not in steps or 'STOPPING' not in steps:
        raise TableError(f'{name}: no RUN / STOPPING transition in the manifest lifecycle')
    t0 = steps['RUN']
    faults = man.get('faults')
    mosi = z['mosi']
    if faults:
        if 'mosi_true' not in z.files:
            raise TableError(f'{name}: fault run without mosi_true')
        if int(faults.get('t0_step', -1)) != t0:
            raise TableError(f'{name}: faults t0_step {faults.get("t0_step")} != first RUN step {t0}')
        true = z['mosi_true']
    else:
        true = mosi
    if mosi.shape != true.shape or mosi.shape[0] != int(man['steps']):
        raise TableError(f'{name}: frame arrays {mosi.shape} / {true.shape}, manifest steps {man["steps"]}')
    S = np.array([P.decode_state(bytes(fr)) for fr in true])
    R = np.array([P.decode_traj_ref(bytes(fr))[:3] for fr in mosi])
    k = int(man['step_first']) + np.arange(len(S)) - t0                # ms since t0
    task, dur = task_duration(srun['args'])
    esp_all = read_esp(os.path.join(run_dir, 'pi', f'{name}.csv'))
    on = np.nonzero(esp_all['run_active'] >= 0.5)[0]
    if not len(on):
        raise TableError(f'{name}: no ESP32 row with run_active = 1')
    # ESP32 clock = the firmware tick counter from the first run_active row; rows of [0, duration) only
    # (the fwsim run ends at duration; the ESP32 log goes on through the motor-off switch)
    n_ticks = int(round(dur / TICK_S))
    tk = esp_all['tick'] - esp_all['tick'][on[0]]
    keep = np.arange(len(tk)) >= on[0]
    keep &= tk < n_ticks
    esp = {c: v[keep] for c, v in esp_all.items()}
    if np.any(np.diff(esp['tick']) <= 0) or np.any(np.diff(esp['timestamp_us']) <= 0):
        raise TableError(f'{name}: ESP32 tick / timestamp_us not increasing in [0, duration)')
    te = np.round(tk[keep] * TICK_S, 6)
    out = dict(name=name, args=srun['args'], task=task, dur=dur, man=man, faults=faults, S=S, R=R, k=k,
               esp=esp, te=te, gain=read_gain(os.path.join(run_dir, 'pi', f'{name}.csv.events.txt')),
               esp32_fw=srun.get('esp32_fw'), h7_fw=srun.get('h7_fw'),
               run_s=(steps['STOPPING'] - t0) / 1000.0)
    return out


def check_fault_check(run_dir, f):
    """The fault run f must have fault_check_<run>.json (design 8.1), ok, of its frame log and schedule."""
    name = f['name']
    fc_path = os.path.join(run_dir, f'fault_check_{name}.json')
    if not os.path.isfile(fc_path):
        raise TableError(f'{name}: no fault_check_{name}.json')
    with open(fc_path) as fh:
        fc = json.load(fh)
    if fc.get('ok') is not True or fc.get('sha256') != f['faults'].get('sha256') or \
            os.path.basename(str(fc.get('frames'))) != f'{name}_frames.npz':
        raise TableError(f'{name}: fault_check_{name}.json not ok or not of this frame log / schedule')


def same_bench_run(a, b):
    """Why bench runs a and b are not the same run but for the schedule, or None."""
    if _run_args(a['args']) != _run_args(b['args']):
        return f'run args {_run_args(a["args"])} != {_run_args(b["args"])}'
    for k in ('esp32_fw', 'h7_fw'):
        if a[k] != b[k] or not a[k]:
            return f'{k} {a[k]} != {b[k]}'
    for k in MAN_SAME:
        if a['man'].get(k) != b['man'].get(k):
            return f'manifest {k}: {a["man"].get(k)!r} != {b["man"].get(k)!r}'
    if a['gain'] != b['gain']:
        return f'#gain {a["gain"]} != {b["gain"]}'
    return None


def _body(S):
    """True body velocity [vx, vy, wz] of decoded frame states [w1..w4, x, y, theta, vx, vy, wz] (world
    velocity rotated by -theta, as fwsim.body_velocity)."""
    c, s = np.cos(S[:, 6]), np.sin(S[:, 6])
    return np.column_stack([c * S[:, 7] + s * S[:, 8], -s * S[:, 7] + c * S[:, 8], S[:, 9]])


def _first_change(t, cols, after=0.0):
    sel = np.nonzero(t >= after - 1e-9)[0]
    if not len(sel):
        return None
    ref = cols[sel[0]]
    d = np.nonzero(np.any(cols[sel] != ref, axis=1))[0]
    return float(t[sel[d[0]]]) if len(d) else None


def bench_row(mode, case, f, base, H_fwd):
    rows = f['faults'].get('rows') or []
    if not rows:
        raise TableError(f'{f["name"]}: no schedule rows')
    s_ms, e_ms = min(int(r['start_ms']) for r in rows), max(int(r['end_ms']) for r in rows)
    dur_ms = int(round(f['dur'] * 1000))
    start, end = s_ms / 1000.0, e_ms / 1000.0
    out = dict(controller=mode, case=case,
               faults=';'.join(f'{r["fault"]}:{r["target"]}:{r.get("params") or {}}' for r in rows),
               win_start_s=start, win_end_s=end, complete=bool(f['faults'].get('complete')))
    # firmware flags on the ESP32 clock
    te, e, tb, eb = f['te'], f['esp'], base['te'], base['esp']
    inw = (te >= start - 1e-9) & (te < end - 1e-9)
    if not inw.any():
        raise TableError(f'{f["name"]}: no ESP32 row inside the window')
    t_enc = _first(te, e['wz_src'] < 0.5, start)
    out['t_src_enc'] = t_enc
    out['t_src_imu'] = _first(te, e['wz_src'] >= 0.5, t_enc + 1e-9) if t_enc is not None else None
    out['t_slip_on'], out['t_slip_off'] = edges(te, e['slip_flag'], start)
    out['t_ws_on'], out['t_ws_off'] = edges(te, e['ws_flag'], start)
    out['base_t_src_enc'] = _first(tb, eb['wz_src'] < 0.5, start)
    out['base_t_slip_on'] = edges(tb, eb['slip_flag'], start)[0]
    out['base_t_ws_on'] = edges(tb, eb['ws_flag'], start)[0]
    out.update(age_g_max_ms=float(e['age_g_ms'][inw].max()), r_max=float(np.abs(e['r'][inw]).max()), r_th=None)
    # tracking errors on the Pi clock (true plant state vs the reference the Pi sent)
    n_span = e_ms + int(TAIL_S * 1000) - s_ms
    err = {}
    for tag, run in (('', f), ('base_', base)):
        sel = (run['k'] >= s_ms) & (run['k'] < e_ms + int(TAIL_S * 1000))
        if int(sel.sum()) != n_span:
            raise TableError(f'{run["name"]}: {int(sel.sum())} frames in the span, want {n_span}')
        err[tag] = _body(run['S'][sel]) - run['R'][sel]
    for i, ax in enumerate(AXES):
        ef, eb_ = err[''][:, i], err['base_'][:, i]
        out[f'err_max_{ax}'] = float(np.abs(ef).max())
        out[f'base_err_max_{ax}'] = float(np.abs(eb_).max())
        out[f'err_rms_{ax}'] = float(np.sqrt(np.mean(ef ** 2)))
        out[f'base_err_rms_{ax}'] = float(np.sqrt(np.mean(eb_ ** 2)))
    pw = lambda t, lg: np.abs(np.column_stack([lg[f'pwm_{w}'][(t >= start - 1e-9) & (t < end + TAIL_S - 1e-9)]
                                               for w in ('fl', 'fr', 'rl', 'rr')]))
    out['sat_frac'] = float(np.mean(pw(te, e) >= PWM_MAX))
    out['base_sat_frac'] = float(np.mean(pw(tb, eb) >= PWM_MAX))
    # pose at t0 + duration, r_true over [0, duration)
    pose = {}
    for tag, run in (('', f), ('base_', base)):
        at = np.nonzero(run['k'] == dur_ms)[0]
        if not len(at):
            raise TableError(f'{run["name"]}: no frame at t0 + {dur_ms} ms')
        pose[tag] = run['S'][at[0], 4:7]
    rs = (f['k'] >= 0) & (f['k'] < dur_ms)
    r_true = (f['S'][rs, :4] @ np.asarray(H_fwd)[2]) - f['S'][rs, 9]
    out.update(end_dx=float(pose[''][0] - pose['base_'][0]), end_dy=float(pose[''][1] - pose['base_'][1]),
               end_dtheta=float(wrap(pose[''][2] - pose['base_'][2])), r_true_max=float(np.abs(r_true).max()))
    # counts on the ESP32 clock
    cnt = {}
    for tag, run in (('', f), ('base_', base)):
        tick, gyro = run['esp']['tick'], run['esp']['gyro_n']          # rows of [0, duration), increasing
        cnt[tag] = dict(ticks=int(len(tick)), gaps=int(round(run['dur'] / TICK_S)) - len(tick),
                        gyro=int(gyro[-1] - gyro[0]))
    out.update(ticks=cnt['']['ticks'], imu_rows=None, imu_dropped=None, base_imu_rows=None,
               ticks_expected=int(round(f['dur'] / TICK_S)), run_s=f['run_s'], tick_gaps=cnt['']['gaps'],
               esp_gyro_n=cnt['']['gyro'], base_esp_gyro_n=cnt['base_']['gyro'],
               ref_first_s=_first_change(f['k'] / 1000.0, f['R']),
               esp_sp_first_s=_first_change(te, np.column_stack([e['sp_vx'], e['sp_vy'], e['sp_wz']])),
               blank=';'.join(BENCH_BLANK))
    return out


def bench_table(run_dir):
    with open(os.path.join(run_dir, 'session.json')) as fh:
        session = json.load(fh)
    with open(os.path.join(run_dir, 'bench_summary.json')) as fh:
        summary = json.load(fh)
    clean = {r.get('name'): r.get('clean') is True for r in summary}
    runs = {}
    for srun in session.get('runs') or []:
        mode, sep, case = srun['name'].partition('_')
        if not sep or not case:
            raise TableError(f'{srun["name"]}: not <MODE>_<CASE>')
        if (mode, case) in runs:
            raise TableError(f'{srun["name"]}: twice in session.json')
        runs[(mode, case)] = load_bench_run(run_dir, srun, clean)
    if not runs:
        raise TableError(f'{run_dir}: session.json has no run')
    import params as _params
    H_fwd = _params.MecanumParams().H_fwd
    rows = []
    for (mode, case), f in sorted(runs.items()):
        if case == 'base':
            if f['faults']:
                raise TableError(f'{f["name"]}: the baseline has schedule rows')
            continue
        if not f['faults']:
            raise TableError(f'{f["name"]}: a fault case without faults in the manifest')
        check_fault_check(run_dir, f)
        base = runs.get((mode, 'base'))
        if base is None:
            raise TableError(f'{mode}_{case}: no {mode}_base baseline')
        why = same_bench_run(f, base)
        if why:
            raise TableError(f'{mode}_{case} vs {mode}_base: {why}')
        rows.append(bench_row(mode, case, f, base, H_fwd))
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
