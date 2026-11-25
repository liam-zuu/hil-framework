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
    firmware and hil commit of the DLL (meta.yaml config / firmware), both builds dirty: false, the same plant
    line (meta.yaml plant: mu_hd, V_batt, source blobs, ...), else the folder cannot be tabled (exit 2);
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
    M_LOSS), and of the baseline's rows; ticks_expected = fwsim_ticks(duration) (design 10.1 step 5): the ticks
    at 10, 20, .. ms up to the last plant step, round(duration / 1 ms) // 10 (Codex FS1-1: a task length that is
    not a whole number of 10 ms, A-C1 32.416 s, has 3241 ticks, not round(duration / 10 ms) = 3242).
  * batch 2 plant faults (design 2026-11-15_Thiet_ke_bom_loi_dot2_v1.md section 7.2, record only), in every
    row, blank where the case has no such fault:
      sat_frac_<w>, base_sat_frac_<w> (w = fl, fr, rl, rr): sat_frac of one wheel over the same span (s11: the
      jammed wheel against the three others);
      yaw_dev_max: largest |theta - theta of the baseline| (wrapped) over the span (s15);
      ramp_end_s, err_rms_ramp_<ax>, err_rms_hold_<ax> and base_*: for a batt_sag row (the first), the RMS
      tracking error over [start, start + ramp_ms) and over [start + ramp_ms, end + 1 s) (s13; ramp blank when
      ramp_ms = 0);
      zone_in_s, zone_out_s: for a mu_zone row (the first), the first plant step at which a wheel hub is inside
      its rectangle, then the first step after it with no hub inside while the row is on (s15 by position, the
      step each side enters, design 11.4), from the plant record (fwsim faults_plant.csv, steps from 0; bench
      <frames>_plant_faults.npz, steps - t0) and the hub positions of rpi5/fault_plant.contact_points with the
      nominal lx, ly of rpi5/params.py;
      pwm_mean, base_pwm_mean: mean |pwm| of the four wheels over the same span as sat_frac (s14: the extra
      effort a body load asks for; firmware PWM units, PWM_MAX = 1000).
    The iBUS loss columns (t_motor_off / t_motor_on against the window) come with the H7 part of ibus_loss: the
    firmware sets run_active = motor on && AUTO && trajectory fresh (AGV 5bde4c5 src/main.c:611), so a loss
    longer than 50 ms drops run_active inside [0, duration) and this bench table refuses such a run today.
Exit 0 = table written, 2 = the folder cannot be tabled.

Bench folders (HIL bench, a tools/hil_bench_session.py session: session.json at the top; design 10.2 step 6):
the runs are session.json runs, named <MODE>_<CASE> as above; each has frames/<run>_frames.npz (Pi frame
log: mosi, mosi_true when --faults, manifest with lifecycle and faults), pi/<run>.csv (ESP32 WiFi log,
agv_logger) and pi/<run>.csv.events.txt (#gain line); bench_summary.json and, for a fault run,
fault_check_<run>.json. The same columns as the fwsim table, from two clocks:
  * Pi time: t = (frame step - t0) / 1 kHz, t0 = the first RUN step (lifecycle; = manifest faults t0_step, the
    origin of the schedule). True plant state from mosi_true (mosi in the baseline, which has no faults):
    body velocity = world velocity rotated by -theta (fwsim.body_velocity), setpoint = the MOSI reference the
    Pi sent; err_*, end pose (frame t0 + t_end, t_end = 10 ms x fwsim_ticks(duration), the time of the last
    fwsim tick, duration = the task's + TAIL_S, as fault_run.py; = duration when it is a whole number of 10 ms,
    as A-W1's 9 s; Codex FS1-1), r_true_max over (0, t_end] (params.MecanumParams H_fwd, as fwsim, whose ticks
    are 0.01 .. t_end s). The span is
    taken on the 1 ms frames, the fwsim table on 10 ms ticks. The frame log must hold one uint8 frame of
    protocol.FRAME_SIZE bytes per step: frame_no (MOSI bytes 44-47) = step_first + row, one run_id (40-43)
    != 0, step_last = step_first + steps - 1, period_us 1000, in the baseline too (C7T1-5).
  * RUN on the bench is not exactly the task (C7T1-1, Liam 15/11 "Sua 10, giu RUN"): the lifecycle ends RUN
    when the publisher reports the end of the task, 8.083-8.102 s in FAULT_C7 run02 / run03, and the reference
    changes 2-4 ms after the schedule's clock (2.002-2.004 s for the 2 s step). run_s and ref_first_s record
    both per run. After the last setpoint change of A-W1 (6 s) RUN and STOPPING both send 0, so the end pose at
    t0 + duration is read at rest in both runs.
  * ESP32 time: t = (tick - tick of the first row with run_active = 1) x 10 ms, the firmware's tick counter
    from its own start of the run (it follows the SWA bit the Pi sets at t0 through the H7 and iBUS: some ms
    late, not measured on a common clock; ref_first_s and esp_sp_first_s show both clocks at the task's first
    setpoint step, an indicator, not a mapping of one clock onto the other); only the rows of [0, duration)
    are read, as the fwsim run ends at duration (the ESP32 log goes on through the motor-off switch after it).
    The tick, not timestamp_us, so the grid is the 10 ms of the fwsim table. The start of the run must be in
    the log (C7T1-2): the row just before the first run_active row is there, with run_active 0, tick one less
    and log_seq one less, and that first row's log_seq is the #gain line's first_seq; the rows of [0, duration)
    are exactly bench_ticks(duration) rows (j x 10 ms < duration; ticks_expected) with consecutive tick and log_seq and run_active 1, every ESP_COLS value
    finite, flags 0 / 1, counters whole (C7T1-6), gyro_n not decreasing from the row before the run (C7T2-2). wz_src, slip_flag, ws_flag edges, age_g_max_ms, r_max,
    sat_frac (|pwm| >= PWM_MAX of the ESP32 rows) and ticks (ESP32 rows in [0, duration)) come from it.
     Before finding the run start, drop a line only if <log>.segments.csv marks it duplicate and its text
     matches an earlier kept line with the same log_seq (L92). Segment marks must match the CSV rows. Drops
     require the logger summary to confirm the segments SHA-256 and duplicate count. A named SHA-256 must
     match even without drops. esp_dup_dropped and base_esp_dup_dropped count drops across each whole log.
  * Left blank on the bench (column blank lists them): r_th (no column in the ESP32 CSV); imu_rows,
    imu_dropped, base_imu_rows (the H7 builds the IMU reports, the Pi logs no report). esp_gyro_n,
    base_esp_gyro_n (C7T1-7): the firmware's gyro report counter gyro_n at the last row of [0, duration) minus
    at the first, i.e. the reports it counted after its first tick of the run up to its last one (duration - 10
    ms of ticks, about 180 at 20 Hz over 8.99 s), not a count of reports in [0, duration) (the row before the
    run is an idle row up to 100 ms earlier, so it does not bound tick 0).
  * More columns: run_s (RUN length, lifecycle), tick_gaps (ticks of [0, duration) without an ESP32 row: 0, as
    a run with one is refused), ref_first_s (first change of the MOSI reference, Pi time), esp_sp_first_s
    (first change of sp_vx, sp_vy, sp_wz in the ESP32 rows, ESP32 time).
  * Same run but for the schedule: session.json run args equal as rpi5/run_hil.py's own parser reads them
    (run_options: abbreviations, the last of a repeated option, --opt=value, any order; C7T1-10, C7T2-1,
    C7T3-1; args with a shell character refused, C7T4-1, and a ~ outside --faults / --step-log, C7T5-1),
    without --faults / --step-log, same ESP32 and H7 #fw, same manifest
    imu_source, imu_noise, plant_src, proto, same #gain line (without first_seq, line_drop, boot), else exit 2;
    a run that bench_summary.json does not call clean cannot be tabled either. A fault run is tabled only if
    tools/fault_check.check() passes again on this folder's frame log, the schedule its fault_check_<run>.json
    names (with the manifest's SHA-256) and this folder's steps/<run>_steps.csv (C7T1-3).

Assumptions (stated by the author, to be checked): log.csv columns as fwsim.write_run writes them (IMU_COLS,
then WS_COLS when the firmware has wheel_slip.c); the tick time column t is the time of the tick in s. Bench:
the frame log has one frame per step from manifest step_first; the ESP32 columns are those of the 86-column
agv_logger header used here (run_active, tick, wz_src, slip_flag, ws_flag, age_g_ms, r, gyro_n, pwm_*, sp_*).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
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


PLANT_DT = 0.001                    # fwsim.simulate dt (fault_run.py passes the default)


def fwsim_ticks(duration) -> int:
    """Firmware ticks of a fwsim run of this duration [s]: fwsim.simulate runs int(round(duration / dt)) plant
    steps, dt = 0.001, the same division here (Codex FS2-1: duration * 1000 rounds otherwise at some half ms),
    and ticks at every 10th, so 10, 20, .. ms (Codex FS1-1)."""
    return int(round(float(duration) / PLANT_DT)) // 10


def bench_ticks(duration) -> int:
    """ESP32 rows of [0, duration) on the bench: the number of j = 0, 1, .. whose ESP32 time round(j x 10 ms, 6)
    (the te of load_bench_run) is < duration (Codex FS1-1: not round(duration / 10 ms), 13.194 s has 1320 of
    them; FS2-1: the duration is not rounded first, 9.000000001 s has 901). Takes 0 <= duration <= BENCH_MAX_S,
    ValueError otherwise (Codex FS3-1: there the whole numbers up to duration / 10 ms stay apart as floats, so
    the two loops below run a few times at most; a run of a fault task lasts under a minute)."""
    d = float(duration)
    if not 0.0 <= d <= BENCH_MAX_S:                  # NaN fails too
        raise ValueError(f'duration {duration!r} s: bench_ticks takes 0 .. {BENCH_MAX_S:g} s')
    n = max(0, math.ceil(d / TICK_S) - 1)
    while round(n * TICK_S, 6) < d:
        n += 1
    while n > 0 and round((n - 1) * TICK_S, 6) >= d:
        n -= 1
    return n


BENCH_MAX_S = 1e9                   # bench_ticks domain [s] (Codex FS3-1)
M_LOSS = 16                         # fwsim/fault_run.py mask bit of imu_loss (faults_imu.csv)
SAME_KEYS = ('task', 'mode', 'seed', 'imu_rate', 'imu_boot_reset', 'imu_noise', 'imu_noise_seed', 'ground', 'mu',
             'no_imu', 'duration')
AXES = ('vx', 'vy', 'wz')
WHEELS = ('fl', 'fr', 'rl', 'rr')


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
            for key in ('config:', '  esp32:', '  hil_framework:', 'faults:', 'plant:'):
                if line.startswith(key):
                    out[key.strip(' :')] = json.loads(line[len(key):].strip())
    for key in ('config', 'esp32', 'hil_framework', 'plant'):
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
    plant = None
    pp = os.path.join(folder, 'faults_plant.csv')                       # batch 2, only with plant rows
    if os.path.isfile(pp):
        rec = read_log(pp)
        plant = (rec['step'], np.column_stack([rec['mask'], rec['x'], rec['y'], rec['theta']]))
    return dict(log=log, faults=faults, meta=meta, imu_rows=len(masks), folder=folder,
                imu_dropped=sum(1 for m in masks if m & M_LOSS), plant=plant)


def same_run(a, b):
    """Why a and b are not the same run but for the schedule, or None."""
    ca, cb = a['meta']['config'], b['meta']['config']
    for k in SAME_KEYS:
        if ca.get(k) != cb.get(k):
            return f'config {k}: {ca.get(k)!r} != {cb.get(k)!r}'
    for k in ('esp32', 'hil_framework'):
        if (a['meta'][k] or {}).get('commit') != (b['meta'][k] or {}).get('commit'):
            return f'{k} commit {a["meta"][k]} != {b["meta"][k]}'
        # C7T1-4: a dirty or unknown build may differ under the same commit
        for m in (a, b):
            if (m['meta'][k] or {}).get('dirty') is not False:
                return f'{k} of {m["folder"]}: dirty or unknown ({m["meta"][k]})'
    # C7T1-4: the plant the run used (meta.yaml plant: mu_hd, V_batt, source blobs, ...), not only the config
    if a['meta']['plant'] != b['meta']['plant']:
        return f'plant {a["meta"]["plant"]} != {b["meta"]["plant"]}'
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


def wheel_sat(pw_f, pw_b):
    """Batch 2 (s11): sat_frac of each wheel; pw_* = |pwm| columns FL, FR, RL, RR of the span."""
    out = {}
    for i, w in enumerate(WHEELS):
        out[f'sat_frac_{w}'] = float(np.mean(pw_f[:, i] >= PWM_MAX))
        out[f'base_sat_frac_{w}'] = float(np.mean(pw_b[:, i] >= PWM_MAX))
    return out


def pwm_mean(pw_f, pw_b):
    """Batch 2 (s14): mean |pwm| of the four wheels over the span; pw_* = |pwm| columns FL, FR, RL, RR."""
    return dict(pwm_mean=float(np.mean(pw_f)), base_pwm_mean=float(np.mean(pw_b)))


def _first_row(rows, fault):
    for i, r in enumerate(rows):
        if r['fault'] == fault:
            return i, r
    return None, None


def ramp_split(rows, t, ef, eb):
    """Batch 2 (s13): RMS tracking error over the ramp and after it, for the first batt_sag row. t: times [s] of
    the span samples, ef / eb: (n, 3) errors of the fault run and the baseline. Blank without a batt_sag row."""
    keys = ['ramp_end_s'] + [f'{tag}err_rms_{part}_{ax}' for tag in ('', 'base_') for part in ('ramp', 'hold')
                             for ax in AXES]
    out = dict.fromkeys(keys)
    _, r = _first_row(rows, 'batt_sag')
    if r is None:
        return out
    start, end = r['start_ms'] / 1000.0, r['end_ms'] / 1000.0
    ramp_end = start + int(r['params']['ramp_ms']) / 1000.0
    out['ramp_end_s'] = ramp_end
    parts = (('ramp', (t >= start - 1e-9) & (t < ramp_end - 1e-9)),
             ('hold', (t >= ramp_end - 1e-9) & (t < end + TAIL_S - 1e-9)))
    for tag, e in (('', ef), ('base_', eb)):
        for part, sel in parts:
            for i, ax in enumerate(AXES):
                out[f'{tag}err_rms_{part}_{ax}'] = float(np.sqrt(np.mean(e[sel, i] ** 2))) if sel.any() else None
    return out


def zone_times(rows, plant):
    """Batch 2 (s15 by position): (zone_in_s, zone_out_s) of the first mu_zone row from the plant record. plant =
    (relative steps, (n, 4) [mask, x, y, theta]) or None; the hub positions as fault_plant.contact_points."""
    i, r = _first_row(rows, 'mu_zone')
    if r is None:
        return dict(zone_in_s=None, zone_out_s=None)
    if plant is None:
        raise TableError(f'mu_zone row {r["id"]!r}: no plant record')
    import fault_plant
    import params as _params
    nom = _params.MecanumParams()
    x0, x1, y0, y1 = (float(r['params'][k]) for k in ('x0', 'x1', 'y0', 'y1'))
    steps, rec = plant
    on = (rec[:, 0].astype(np.int64) >> i) & 1 == 1
    inside = np.array([any(x0 <= hx < x1 and y0 <= hy < y1 for hx, hy in fault_plant.contact_points(v[1:], nom))
                       for v in rec])
    k_in = np.nonzero(on & inside)[0]
    if not len(k_in):
        return dict(zone_in_s=None, zone_out_s=None)
    k_out = np.nonzero(on[k_in[0]:] & ~inside[k_in[0]:])[0]
    return dict(zone_in_s=float(steps[k_in[0]]) / 1000.0,
                zone_out_s=float(steps[k_in[0] + k_out[0]]) / 1000.0 if len(k_out) else None)


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
               ticks_expected=fwsim_ticks(f['meta']['config']['duration']))
    # batch 2 plant faults (design 2026-11-15 section 7.2)
    rows = f['faults']['rows']
    out.update(wheel_sat(pw(lf), pw(lb)))
    out['yaw_dev_max'] = float(np.abs(wrap(lf['theta'][span] - lb['theta'][span])).max())
    err = lambda lg: np.column_stack([lg[f'v_{ax}'] - lg[f'sp_{ax}'] for ax in AXES])
    out.update(ramp_split(rows, t, err(lf), err(lb)))
    plant = None if f['plant'] is None else (f['plant'][0], f['plant'][1])     # fwsim steps count from 0 = t0
    out.update(zone_times(rows, plant))
    out.update(pwm_mean(pw(lf), pw(lb)))
    return out


def table(run_dir, *, report_duplicates=None):
    if os.path.isfile(os.path.join(run_dir, 'session.json')):
        return bench_table(run_dir, report_duplicates=report_duplicates)
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
ESP_COLS = ('timestamp_us', 'tick', 'log_seq', 'run_active', 'wz_src', 'slip_flag', 'ws_flag', 'age_g_ms', 'r',
            'gyro_n', 'pwm_fl', 'pwm_fr', 'pwm_rl', 'pwm_rr', 'sp_vx', 'sp_vy', 'sp_wz')
ESP_FLAGS = ('run_active', 'wz_src', 'slip_flag', 'ws_flag')          # 0 / 1
ESP_COUNTERS = ('tick', 'log_seq', 'gyro_n')                          # whole numbers
GAIN_SKIP = ('first_seq', 'line_drop', 'boot')                        # differ run to run, not the gains
MAN_SAME = ('imu_source', 'imu_noise', 'plant_src', 'proto', 'frame_len', 'plant')


RUN_SKIP = ('faults', 'step_log')                                     # argparse dests of --faults, --step-log
# C7T4-1: characters bash on the Pi acts on before run_hil.py sees the words (tools/hil_bench_session.py puts
# the args of a run into a bash command line as they are): quotes, escapes, expansions, globs, comments,
# command separators. Without them bash splits on blanks, as shlex.split does here, but it still expands ~
# where shlex does not (C7T5-1, C7T6-1: '--uart-port ~/u' becomes <HOME>/u, '--uart-port=~/u' stays ~/u, and
# a word shaped like an assignment, 'a=~/u', is expanded after the '='): so a ~ anywhere in the value of any
# option but --faults / --step-log (left out of the comparison) is refused after run_hil.py's parser.
SHELL_CHARS = frozenset('{}[]$`*?()!;&|<>"\'\\#\n\r')


def run_options(args):
    """C7T3-1: the options of a session run as rpi5/run_hil.py's own parser reads them (run_hil._parse_args with
    sys.argv set for the call): abbreviations (argparse allow_abbrev, '--grou' = '--ground'), the last of a
    repeated option, '--opt=value' and the order are what run_hil.py makes of them. A value equal to a default
    and the default itself stay apart (None vs 'traction'): such a pair is refused, not matched. Args run_hil.py
    refuses are TableError, and so are args with SHELL_CHARS (C7T4-1: bash would change them before run_hil.py,
    shlex here would not) and a ~ in any option value but faults, step_log (C7T5-1)."""
    import contextlib
    import io
    import shlex
    import run_hil
    bad = sorted(set(args) & SHELL_CHARS)
    if bad:
        raise TableError(f'run args {args!r}: shell characters {bad} (bash on the Pi acts on them before '
                         f'run_hil.py reads the args, C7T4-1)')
    old, err = sys.argv, io.StringIO()
    sys.argv = ['run_hil.py'] + shlex.split(args)
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            ns = run_hil._parse_args()
    except SystemExit:
        raise TableError(f'run args {args!r}: run_hil.py refuses them ({err.getvalue().strip()[-300:]})')
    finally:
        sys.argv = old
    opts = vars(ns)
    for k, v in opts.items():                                          # C7T5-1
        vals = v if isinstance(v, (list, tuple)) else [v]
        if k not in RUN_SKIP and any(isinstance(x, str) and '~' in x for x in vals):
            raise TableError(f'run args {args!r}: ~ in --{k.replace("_", "-")} (bash expands ~ where shlex does '
                             f'not, C7T5-1)')
    return opts


def _run_args(args):
    """C7T1-10, C7T2-1, C7T3-1: the effective run_hil.py options of a session run without RUN_SKIP (the schedule
    and its step log), as sorted (dest, repr(value)) pairs."""
    return sorted((k, repr(v)) for k, v in run_options(args).items() if k not in RUN_SKIP)


def task_duration(args):
    """(task id, duration [s]) of a bench run: the --test task (as run_hil.py reads it) + TAIL_S (the task then
    STOPPING of --finite-run, as fault_run.py's task + TAIL_S; RUN itself runs a little longer, docstring). The
    tasks fwsim/fault_run.py task_setpoint takes (design 2026-11-15 14.9 G1): A-W1 of rpi5/fault_tasks.py, or a
    group A test on the ground (rpi5/trajectory_pub.GROUP_A, not LIFTED_TESTS) with its spec duration."""
    opts = run_options(args)
    task = opts.get('test')
    if not task:
        raise TableError(f'run args {args!r}: no --test task')
    if opts.get('s02_prep') is not None:           # Codex FS1-4: --s02-prep changes the task's length and timing
        raise TableError(f'run args {args!r}: --s02-prep runs another task than --test alone (not a fault task)')
    import fault_tasks
    if task in fault_tasks.TASKS:
        return task, fault_tasks.TASKS[task][0] + TAIL_S
    from trajectory_pub import GROUP_A, LIFTED_TESTS
    if task not in GROUP_A or task in LIFTED_TESTS:
        raise TableError(f'--test {task}: not a fault task ({sorted(fault_tasks.TASKS)} or a group A test on the '
                         f'ground)')
    return task, GROUP_A[task].duration + TAIL_S


def _esp_segment_summary(path):
    try:
        with open(f'{path}.summary.json', encoding='utf-8') as fh:
            summary = json.load(fh)
        digest = ((summary.get('files') or {}).get('segments') or {}).get('sha256')
        segments = summary.get('segments')
    except (OSError, ValueError, AttributeError):
        return None, None
    count = None
    if isinstance(segments, list) and all(isinstance(s, dict) and type(s.get('duplicates')) is int for s in segments):
        count = sum(s['duplicates'] for s in segments)
    return digest if isinstance(digest, str) else None, count


def _drop_esp_duplicates(path, head, body, lines):
    seg = f'{path}.segments.csv'
    try:
        with open(seg, 'rb') as fh:
            blob = fh.read()
        marks = list(csv.DictReader(io.StringIO(blob.decode('utf-8'), newline=''), strict=True))
    except FileNotFoundError:
        return body, []
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise TableError(f'{seg}: {exc}') from exc
    if len(lines) != len(body):
        raise TableError(f'{path}: {len(body)} CSV rows for {len(lines)} data lines')
    if len(marks) != len(body):
        raise TableError(f'{seg}: {len(marks)} rows for {len(body)} data lines')
    named, count = _esp_segment_summary(path)
    if named is not None and named != hashlib.sha256(blob).hexdigest():
        raise TableError(f'{seg}: SHA-256 differs from the logger summary')

    cols = {h: i for i, h in enumerate(head)}
    kept, seen, dropped = [], set(), []
    for i, (row, line, mark) in enumerate(zip(body, lines, marks), start=1):
        seq, ts = row[cols['log_seq']], row[cols['timestamp_us']]
        if mark.get('row') != str(i) or mark.get('log_seq') != seq or mark.get('timestamp_us') != ts:
            raise TableError(f'{seg} row {i} does not match CSV data line {i}')
        raw = line.rstrip('\r\n')
        if mark.get('status') == 'duplicate':
            if raw not in seen:
                raise TableError(f'{seg} row {i}: log_seq {seq} marked duplicate but no earlier identical kept line')
            dropped.append(seq)
        else:
            kept.append(row)
            seen.add(raw)
    if dropped and named is None:
        raise TableError(f'{seg}: logger summary must name its SHA-256 to drop duplicate lines')
    if dropped and count != len(dropped):
        raise TableError(f'{seg}: dropped {len(dropped)} lines, logger summary counts {count} duplicates')
    return kept, dropped


def read_esp(path):
    """ESP32 CSV (agv_logger) as ({column: float array}, log_seq of each dropped line)."""
    with open(path, newline='') as fh:
        lines = fh.readlines()
    rows = list(csv.reader(lines))
    if len(rows) < 2:
        raise TableError(f'{path}: no data row')
    head, body = rows[0], rows[1:]
    bad = [i for i, r in enumerate(body) if len(r) != len(head)]
    if bad:
        raise TableError(f'{path}: row {bad[0] + 2} has {len(body[bad[0]])} fields, header {len(head)}')
    miss = [c for c in ESP_COLS if c not in head]
    if miss:
        raise TableError(f'{path}: no column {miss}')
    body, dropped = _drop_esp_duplicates(path, head, body, lines[1:])
    data = np.array(body, dtype=float)
    return {h: data[:, i] for i, h in enumerate(head)}, dropped


def read_gain(path):
    """The #gain line of an agv_logger events file: (dict without GAIN_SKIP, first_seq); every copy agrees."""
    seen = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if '#gain,' in line:
                kv = dict(p.split('=', 1) for p in line.strip().split('#gain,', 1)[1].split(',') if '=' in p)
                if kv not in seen:
                    seen.append(kv)
    if len(seen) != 1:
        raise TableError(f'{path}: {len(seen)} different #gain lines (want 1)')
    first = seen[0].get('first_seq')
    if first is None or not first.isdigit():
        raise TableError(f'{path}: #gain line without first_seq')
    return {k: v for k, v in seen[0].items() if k not in GAIN_SKIP}, int(first)


def _u32(frames, ofs):
    """Little-endian uint32 at byte ofs of every frame."""
    b = frames[:, ofs:ofs + 4].astype(np.uint64)
    return b[:, 0] | (b[:, 1] << np.uint64(8)) | (b[:, 2] << np.uint64(16)) | (b[:, 3] << np.uint64(24))


def check_frames(name, man, mosi, true):
    """C7T1-5: one uint8 frame of protocol.FRAME_SIZE bytes per step, frame_no = step, one run_id != 0."""
    for nm, a in (('mosi', mosi), ('mosi_true', true)):
        if a.dtype != np.uint8 or a.ndim != 2 or a.shape[1] != P.FRAME_SIZE:
            raise TableError(f'{name}: {nm} {a.shape} {a.dtype}, want (n, {P.FRAME_SIZE}) uint8')
    n, first = int(man['steps']), int(man['step_first'])
    if (len(mosi) != n or true.shape != mosi.shape or int(man.get('step_last', -1)) != first + n - 1
            or man.get('period_us') != 1000):
        raise TableError(f'{name}: frames {mosi.shape} / {true.shape}, manifest step_first {first} step_last '
                         f'{man.get("step_last")} steps {n} period_us {man.get("period_us")}')
    want = first + np.arange(n, dtype=np.uint64)
    for nm, a in (('mosi', mosi), ('mosi_true', true)):
        rid = _u32(a, P.OFS_RUN_ID)
        if not np.array_equal(_u32(a, P.OFS_FRAME_NO), want):
            raise TableError(f'{name}: {nm} frame_no is not step_first + row')
        if rid[0] == 0 or np.any(rid != rid[0]):
            raise TableError(f'{name}: {nm} run_id 0 or not one value')
    if not np.array_equal(_u32(true, P.OFS_RUN_ID), _u32(mosi, P.OFS_RUN_ID)):
        raise TableError(f'{name}: mosi and mosi_true run_id differ')


def esp_run_rows(name, esp_all, n_ticks, first_seq):
    """C7T1-2, C7T1-6: (rows of [0, duration), the row just before the run) of the ESP32 log, or TableError
    when the log does not show the start of the run or holds a gap or a bad value in the span."""
    on = np.nonzero(esp_all['run_active'] >= 0.5)[0]
    if not len(on):
        raise TableError(f'{name}: no ESP32 row with run_active = 1')
    i0 = int(on[0])
    if i0 == 0:
        raise TableError(f'{name}: no ESP32 row before the first run_active row (start of the run not in the log)')
    sel = np.arange(i0 - 1, min(i0 + n_ticks, len(esp_all['tick'])))
    for c in ESP_COLS:
        v = esp_all[c][sel]
        if not np.all(np.isfinite(v)):
            raise TableError(f'{name}: ESP32 {c} not finite near the run')
        if c in ESP_FLAGS and not np.all((v == 0) | (v == 1)):
            raise TableError(f'{name}: ESP32 {c} not 0 / 1 near the run')
        if c in ESP_COUNTERS and not np.all(v == np.round(v)):
            raise TableError(f'{name}: ESP32 {c} not a whole number near the run')
    pre = {c: esp_all[c][i0 - 1] for c in ESP_COLS}
    rows = {c: v[i0:i0 + n_ticks] for c, v in esp_all.items()}
    if (pre['run_active'] != 0 or rows['tick'][0] != pre['tick'] + 1 or rows['log_seq'][0] != pre['log_seq'] + 1
            or int(rows['log_seq'][0]) != first_seq):
        raise TableError(f'{name}: start of the run not shown (row before: run_active {pre["run_active"]:g}, tick '
                         f'{pre["tick"]:g}, log_seq {pre["log_seq"]:g}; first run row: tick {rows["tick"][0]:g}, '
                         f'log_seq {rows["log_seq"][0]:g}; #gain first_seq {first_seq})')
    if (len(rows['tick']) != n_ticks or np.any(np.diff(rows['tick']) != 1) or np.any(np.diff(rows['log_seq']) != 1)
            or np.any(np.diff(rows['timestamp_us']) <= 0) or np.any(rows['run_active'] != 1)):
        raise TableError(f'{name}: ESP32 rows of [0, duration): {len(rows["tick"])} of {n_ticks}, or a tick / '
                         f'log_seq gap, or run_active 0 inside')
    if np.any(np.diff(np.r_[pre['gyro_n'], rows['gyro_n']]) < 0):        # C7T2-2: no wrap is handled
        raise TableError(f'{name}: ESP32 gyro_n decreases near the run')
    return rows, pre


def bare_name(name, f):
    """Codex FQ1-2: the plant record named in the manifest must be a bare file name of the frames folder (a
    string, no folder part, not absolute, not '', '.', '..'), checked before anything is opened."""
    if (not isinstance(f, str) or f in ('', '.', '..') or os.path.isabs(f) or os.path.basename(f) != f
            or '/' in f or '\\' in f or ':' in f):
        raise TableError(f'{name}: plant_record file {f!r} is not a bare file name of the frames folder')
    return f


def load_bench_run(run_dir, srun, clean):
    name = srun['name']
    if not clean.get(name):
        raise TableError(f'{name}: not clean in bench_summary.json')
    npz = os.path.join(run_dir, 'frames', f'{name}_frames.npz')
    z = np.load(npz)
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
    check_frames(name, man, mosi, true)
    S = np.array([P.decode_state(bytes(fr)) for fr in true])
    R = np.array([P.decode_traj_ref(bytes(fr))[:3] for fr in mosi])
    k = int(man['step_first']) + np.arange(len(S)) - t0                # ms since t0
    task, dur = task_duration(srun['args'])
    gain, first_seq = read_gain(os.path.join(run_dir, 'pi', f'{name}.csv.events.txt'))
    # ESP32 clock = the firmware tick counter from the first run_active row; rows of [0, duration) only
    # (the fwsim run ends at duration; the ESP32 log goes on through the motor-off switch)
    n_ticks = bench_ticks(dur)
    esp_all, dup_seqs = read_esp(os.path.join(run_dir, 'pi', f'{name}.csv'))
    esp, pre = esp_run_rows(name, esp_all, n_ticks, first_seq)
    te = np.round((esp['tick'] - esp['tick'][0]) * TICK_S, 6)
    plant = None
    pr = (faults or {}).get('plant_record')
    if isinstance(pr, dict) and pr.get('file') is not None:             # batch 2: <frames>_plant_faults.npz
        zr = np.load(os.path.join(run_dir, 'frames', bare_name(name, pr['file'])))
        cols = json.loads(str(zr['cols']))
        rec = zr['record']
        plant = (rec[:, cols.index('step')] - t0,
                 rec[:, [cols.index(c) for c in ('mask', 'x', 'y', 'theta')]])
    return dict(name=name, args=srun['args'], task=task, dur=dur, man=man, faults=faults, S=S, R=R, k=k,
                esp=esp, esp_pre=pre, esp_dup_seqs=dup_seqs, te=te, gain=gain, npz=npz, esp32_fw=srun.get('esp32_fw'),
                h7_fw=srun.get('h7_fw'), run_s=(steps['STOPPING'] - t0) / 1000.0, plant=plant)


def run_fault_check(npz, sched, step_log):
    """tools/fault_check.check (design 8.1), the byte-for-byte check of a fault run (tests replace this)."""
    import fault_check
    return fault_check.check(npz, sched, step_log)


def _thesis_path(run_dir, rel):
    """A path the fault_check report wrote relative to the thesis repo (05_DATA/...): joined to the folder that
    holds the 05_DATA of run_dir."""
    if os.path.isabs(rel):
        return rel
    parts = os.path.abspath(run_dir).split(os.sep)
    norm = [os.path.normcase(p) for p in parts]                         # C7T2-4: 05_data on Windows
    if os.path.normcase('05_DATA') not in norm:
        raise TableError(f'{run_dir}: not under a 05_DATA folder, cannot place {rel}')
    i = norm.index(os.path.normcase('05_DATA'))
    return os.path.join(os.sep.join(parts[:i]) + os.sep, *rel.replace('\\', '/').split('/'))


def check_fault_check(run_dir, f):
    """C7T1-3: the fault run f is tabled only if tools/fault_check.check() passes again on this folder's frame
    log, the schedule its fault_check_<run>.json names (SHA-256 = the manifest's) and this folder's step log."""
    name = f['name']
    fc_path = os.path.join(run_dir, f'fault_check_{name}.json')
    if not os.path.isfile(fc_path):
        raise TableError(f'{name}: no fault_check_{name}.json (it names the schedule file)')
    with open(fc_path) as fh:
        sched = json.load(fh).get('schedule')
    if not sched:
        raise TableError(f'{name}: fault_check_{name}.json names no schedule')
    sched = _thesis_path(run_dir, sched)
    step_log = os.path.join(run_dir, 'steps', f'{name}_steps.csv')
    if not os.path.isfile(sched) or not os.path.isfile(step_log):
        raise TableError(f'{name}: schedule {sched} or step log {step_log} not found')
    res = run_fault_check(f['npz'], sched, step_log)
    if res.get('ok') is not True or res.get('sha256') != f['faults'].get('sha256'):
        raise TableError(f'{name}: tools/fault_check.check not ok again on this frame log / schedule / step log '
                         f'({res.get("error") or res.get("fails")})')


def _man_value(man, k):
    """Manifest value compared between a fault run and its baseline. Batch 2 (design 2026-11-15 section 4.1): a
    run with plant faults has plant 'FaultPlant', a wrapper of the plant of its baseline; it counts as the
    class its plant record names (inner, Codex FP1-11), and stays 'FaultPlant' without one."""
    v = man.get(k)
    pr = (man.get('faults') or {}).get('plant_record')
    if k == 'plant' and v == 'FaultPlant' and isinstance(pr, dict) and isinstance(pr.get('inner'), str):
        return pr['inner']
    return v


def same_bench_run(a, b):
    """Why bench runs a and b are not the same run but for the schedule, or None."""
    if _run_args(a['args']) != _run_args(b['args']):
        return f'run args {_run_args(a["args"])} != {_run_args(b["args"])}'
    for k in ('esp32_fw', 'h7_fw'):
        if a[k] != b[k] or not a[k]:
            return f'{k} {a[k]} != {b[k]}'
    for k in MAN_SAME:
        va, vb = (_man_value(m['man'], k) for m in (a, b))
        if va != vb:
            return f'manifest {k}: {va!r} != {vb!r}'
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
    end_ms = 10 * fwsim_ticks(f['dur'])                    # the last fwsim tick (Codex FS1-1)
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
    err, th, tk = {}, {}, None
    for tag, run in (('', f), ('base_', base)):
        sel = (run['k'] >= s_ms) & (run['k'] < e_ms + int(TAIL_S * 1000))
        if int(sel.sum()) != n_span:
            raise TableError(f'{run["name"]}: {int(sel.sum())} frames in the span, want {n_span}')
        err[tag] = _body(run['S'][sel]) - run['R'][sel]
        th[tag] = run['S'][sel, 6]
        tk = run['k'][sel] / 1000.0
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
    # batch 2 plant faults (design 2026-11-15 section 7.2): wheel sat on the ESP32 clock, the rest on the Pi's;
    # placed after ticks_expected below, as in the fwsim table (the bench columns extend the fwsim ones)
    b2 = wheel_sat(pw(te, e), pw(tb, eb))
    b2['yaw_dev_max'] = float(np.abs(wrap(th[''] - th['base_'])).max())
    b2.update(ramp_split(rows, tk, err[''], err['base_']))
    b2.update(zone_times(rows, f['plant']))
    b2.update(pwm_mean(pw(te, e), pw(tb, eb)))
    # pose at t0 + end_ms, r_true over (0, end_ms] (end_ms = duration for a whole number of 10 ms)
    pose = {}
    for tag, run in (('', f), ('base_', base)):
        at = np.nonzero(run['k'] == end_ms)[0]
        if not len(at):
            raise TableError(f'{run["name"]}: no frame at t0 + {end_ms} ms')
        pose[tag] = run['S'][at[0], 4:7]
    rs = (f['k'] > 0) & (f['k'] <= end_ms)                         # (0, end_ms], as fwsim (C7T1-8)
    r_true = (f['S'][rs, :4] @ np.asarray(H_fwd)[2]) - f['S'][rs, 9]
    out.update(end_dx=float(pose[''][0] - pose['base_'][0]), end_dy=float(pose[''][1] - pose['base_'][1]),
               end_dtheta=float(wrap(pose[''][2] - pose['base_'][2])), r_true_max=float(np.abs(r_true).max()))
    # counts on the ESP32 clock
    cnt = {}
    for tag, run in (('', f), ('base_', base)):
        tick, gyro = run['esp']['tick'], run['esp']['gyro_n']          # rows of [0, duration), consecutive
        cnt[tag] = dict(ticks=int(len(tick)), gaps=bench_ticks(run['dur']) - len(tick),
                        gyro=int(gyro[-1] - gyro[0]))  # counted after the first tick, to the last (C7T1-7)
    out.update(ticks=cnt['']['ticks'], imu_rows=None, imu_dropped=None, base_imu_rows=None,
               ticks_expected=bench_ticks(f['dur']))
    out.update(b2)
    out.update(run_s=f['run_s'], tick_gaps=cnt['']['gaps'],
               esp_gyro_n=cnt['']['gyro'], base_esp_gyro_n=cnt['base_']['gyro'],
               ref_first_s=_first_change(f['k'] / 1000.0, f['R']),
               esp_sp_first_s=_first_change(te, np.column_stack([e['sp_vx'], e['sp_vy'], e['sp_wz']])),
               esp_dup_dropped=len(f['esp_dup_seqs']), base_esp_dup_dropped=len(base['esp_dup_seqs']),
               blank=';'.join(BENCH_BLANK))
    return out


def bench_table(run_dir, *, report_duplicates=None):
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
    if report_duplicates is not None:
        for run in runs.values():
            if run['esp_dup_seqs']:
                report_duplicates(run['name'], run['esp_dup_seqs'])
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

    def report_duplicates(name, seqs):
        print(f'fault_c7_table: {name}: dropped {len(seqs)} duplicate ESP32 line(s) (log_seq {", ".join(seqs)})')

    try:
        rows = table(a.run_dir, report_duplicates=report_duplicates)
    except (TableError, OSError, ValueError, KeyError) as exc:
        print(f'fault_c7_table: cannot table: {exc}', file=sys.stderr)
        return 2
    out = a.out or os.path.join(a.run_dir, 'c7_table.csv')
    write_csv(rows, out)
    print(f'fault_c7_table: {len(rows)} rows -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
