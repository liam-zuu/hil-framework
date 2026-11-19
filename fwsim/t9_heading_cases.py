"""T9 (O1 v11.41 appendix J v6 section 5 command-path condition (1), section 7): three test cases of the
heading-loop latch on fwsim, i.e. the ESP32 firmware's own heading.c / imu_fusion.c / pos_ctrl.c /
vel_ctrl.c (AGV 5bde4c5 when the DLL is built from it), plus a normal run as the control.

Cases (car standing still, as before every O1 run; the cases test the firmware's logic, not dynamics):
  ok          yaw valid at the rising edge of run_active: latched at the edge (control, must pass)
  mat_yaw     yaw lost at the edge (no IMU report, RV, gyro and accel alike, from 200 ms before the edge:
              older than t_age 125 ms), never back during the run
  chot_muon   late latch: (a) no report from 200 ms before to 300 ms after the edge, psi0 latched on the
              first tick after the next report (+310 ms); (b) no report for 90 ms before the edge: the last
              one is 96.3 ms old (< t_age), the firmware latches it on the edge tick
  reset_giua  reset in the middle of the run: (a) chip reset (SH2 reset, reset_n + 1);
              (b) no report for 1 s while wz_ref turns psi_ref (r_valid 0 from +1.08 s), back at
              +2.01 s with |e| > e_psi_reset; (c) run_active forced to 0 for 300 ms (as a trajectory
              timeout would): the controller's reaction
  canh_doi_mode  extra, injected: a controller-mode event (vel_ctrl_set_mode) on the tick before the edge;
              checks heading.c's priority of reset_event over the start of the run. Not reachable on the
              car at 5bde4c5: nothing outside vel_ctrl.c calls vel_ctrl_set_mode and gain_set.c refuses
              a gain set for the other controller (GAIN_ERR_CTRL)

Each case writes the log rows the firmware would push (main.c push_log, logger.c columns of the same
name and print format) and judges them with ref_cond1(): a reference reading of J section 5 (1) for
the CAM scorer, NOT the scorer. The age threshold is a parameter: J locks it at C8; the default here
is only for the demo and is not a proposal.

  python fwsim/t9_heading_cases.py --out <dir> [--age-max-ms 60]

The rows are built here from the controller outputs of the DLL (fw_sim.c does not compile main.c,
logger.c, imu.cpp); column names and print formats as logger.c. rv_n and reset_n are the counters the
firmware copies from imu_data_t (main.c push_log: imu->rv_n, imu->reset_n); fw_sim.c keeps them in the
same struct but does not export them, so this script counts the reports and resets it hands to the DLL
(same increments as fw_sim.c). Report n carries yaw YAW0_DEG + n * YAW_STEP_DEG, so a test can tell
which report psi0 took.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim  # noqa: E402

TICK_US = fwsim.FW_DT_US          # control_task 100 Hz
RV_PERIOD_US = 50_000             # IMU_REPORT_PERIOD_US of the main env and ladrc (20 Hz, Liam 10/11)
RV_PHASE_US = 3_700               # report phase against the tick grid (any value not on the grid)
T0_US = 1_000_000
YAW0_DEG = 30.0                   # yaw of the standing car (any value: psi0 takes it)
YAW_STEP_DEG = 0.01               # report n carries yaw YAW0_DEG + n * YAW_STEP_DEG: tells which report psi0 took
EDGE_US = T0_US + 2_000_000       # run_active rises on the first tick at or after this time
RUN_US = 3_000_000
T_AGE_US = RV_PERIOD_US * 5 // 2    # imu_fusion.h t_age_us at 20 Hz: the firmware's own validity limit
SP_RUN = (0.0, 0.0, 0.3)          # wz_ref during the run, so psi_ref integrates (counts stay 0: car held)

# Columns as logger.c names them; formats as logger.c prints them
COLS = ['timestamp_us', 'log_seq', 'motor_on', 'run_active', 'r_valid', 'age_r_ms', 'rv_n', 'reset_n',
        'head_on', 'psi_ref', 'e_psi', 'psi_reset_count', 'pos_reset_n', 'yaw_rad', 'wz_cmd']
FMT = {'age_r_ms': '{:.1f}', 'psi_ref': '{:.4f}', 'e_psi': '{:.4f}', 'yaw_rad': '{:.4f}', 'wz_cmd': '{:.4f}'}


class Case:
    """Schedule of one case: which game-RV reports are dropped, chip resets, run_active stretches."""

    def __init__(self, name, drop=(), resets=(), run=((EDGE_US, EDGE_US + RUN_US),), mode_at=(),
                 extra_rv=()):
        self.name = name
        self.drop = drop          # [(t0, t1)): no RV/gyro/accel report with t in the window
        self.resets = resets      # chip reset times (fwsim_imu_reset)
        self.run = run            # [(t0, t1)): run_active = 1 on ticks in a window
        self.mode_at = mode_at    # tick times after which vel_ctrl_set_mode() is injected (fwsim only)
        self.extra_rv = extra_rv  # extra report times besides the 20 Hz grid

    def run_active(self, t):
        return any(a <= t < b for a, b in self.run)

    def dropped(self, t):
        return any(a <= t < b for a, b in self.drop)


def cases():
    e = EDGE_US
    return [
        Case('ok'),
        Case('mat_yaw', drop=((e - 200_000, e + RUN_US),)),
        Case('chot_muon_a', drop=((e - 200_000, e + 300_000),)),
        Case('chot_muon_b', drop=((e - 90_000, e + 1),)),
        Case('reset_giua_a', resets=(e + 1_000_000,)),
        Case('reset_giua_b', drop=((e + 1_000_000, e + 2_000_000),)),
        Case('reset_giua_c', run=((e, e + 1_000_000), (e + 1_300_000, e + RUN_US))),
        # extra, injected: mode event on the tick just before the edge (not reachable on the car, see
        # the module docstring): reset_event on the edge tick, yaw valid, latch one tick late
        Case('canh_doi_mode', mode_at=(e - TICK_US,)),
    ]


def simulate(fw, case):
    """Run one case; return the log rows (dict per tick, every tick: the motor is on throughout,
    as on the car during a run; run_active follows the case)."""
    fw.reset()
    fw.set_run(False)
    rv_n = reset_n = 0
    ev = []                                   # (t_us, kind) of IMU events, applied before the tick
    t = T0_US - RV_PERIOD_US
    while t < EDGE_US + RUN_US + 500_000:
        t += RV_PERIOD_US
        ev.append((t + RV_PHASE_US, 'rv'))
    for tr in case.resets:
        ev.append((tr, 'reset'))
    for tx in case.extra_rv:
        ev.append((tx, 'rv'))
    ev.sort()
    k = 0
    rows = []
    seq = 0
    t = T0_US
    while t < EDGE_US + RUN_US + 300_000:
        while k < len(ev) and ev[k][0] <= t:
            te, kind = ev[k]
            k += 1
            if kind == 'reset':
                fw.imu_reset(te)
                reset_n += 1
            elif not case.dropped(te):
                fw.imu_report('gyro', te, 0.0)
                fw.imu_report('accel', te, (0.0, 0.0, 9.80665))
                rv_n += 1
                fw.imu_report('rv', te, YAW0_DEG + rv_n * YAW_STEP_DEG)
        ra = case.run_active(t)
        fw.set_run(ra)
        fw.tick(t, (0, 0, 0, 0), SP_RUN if ra else (0.0, 0.0, 0.0))
        if t in case.mode_at:                 # injected event, seen by vel_ctrl_step on the next tick
            fw.set_mode('ADRC' if fw.ctrl_mode == 'PID' else 'PID')
        m = dict(zip(fwsim.IMU_COLS, fw.imu_last()))
        p = dict(zip(fwsim.POS_COLS, fw.pos_last()))
        seq += 1
        rows.append({'timestamp_us': t, 'log_seq': seq, 'motor_on': 1, 'run_active': int(p['run_active']),
                     'r_valid': int(m['r_valid']), 'age_r_ms': m['age_r_ms'], 'rv_n': rv_n, 'reset_n': reset_n,
                     'head_on': int(m['head_on']), 'psi_ref': m['psi_ref'], 'e_psi': m['e_psi'],
                     'psi_reset_count': int(m['psi_reset_count']), 'pos_reset_n': int(p['pos_reset_n']),
                     'yaw_rad': m['yaw'], 'wz_cmd': m['wz_cmd'], '_psi_imu': m['psi_imu']})
        t += TICK_US
    return rows


def as_logged(rows):
    """The rows as the logger prints them (floats rounded by logger.c's format), then parsed back."""
    out = []
    for r in rows:
        out.append({c: (float(FMT[c].format(r[c])) if c in FMT else int(r[c])) for c in COLS})
    return out


def _num(r, k):
    """Field as a finite number, or None (missing, empty, not a number, NaN/inf)."""
    try:
        v = float(r.get(k))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def ref_cond1(rows, age_max_ms, end_seq=None, edge_seq=None):
    """Reference reading of J v6 section 5 command-path condition (1) on the log rows of one run (one boot,
    sorted by log_seq; the caller checks the boot and duplicate log_seq). Not the scorer: the scorer has its
    own edge, end anchor and failure order (J section 5: operator first; a boot inside the run is an
    operator failure before this check).
    Returns (bad, reasons, info); when bad, info['loai'] is 'he_do' (measurement-system failure: no edge
    evidence, i.e. the row just before the edge missing or unreadable, or the log not reaching end_seq;
    O1 v11.41 v8 line 94) or 'duong_lenh'.

    Edge: edge_seq if the caller gives it (verified edge), else the first row with run_active = 1 whose
    previous row has run_active = 0. The previous row must be log_seq - 1 with run_active exactly 0.
    Window: the edge to end_seq (the scorer's end anchor, must be a row of the log at or after the edge;
    default: the last run_active = 1 row), every row in it, run_active = 0 stretches included; counters
    compared with their value on the edge row.
    Edge row: r_valid = 1, rv_n > 0, head_on = 1, psi_ref = 0 (O1 v8 line 94; J line 127).
    Yaw age at the latch (J line 127), integer microseconds: age_r_ms as printed (0.1 ms) + 50 us (half
    print step) + gap to the next row (tick start -> latch <= gap, T9 doc section 2.1) + 3 us (1 us clock
    steps at three stamps, float conversion), only when the edge row meets every condition above, the next
    row is log_seq + 1 with a later timestamp, and the logged age lies in the proven domain
    [-gap - 0.05 ms, t_age + 0.05 ms] (t_age = 125 ms at 20 Hz; outside it the flags and the age disagree);
    otherwise not determinable, which J counts as bad. Bad when the bound exceeds age_max_ms (exact
    decimal comparison, no rounding of the threshold)."""
    reasons, info = [], {}
    seq = [_num(r, 'log_seq') for r in rows]
    ra = [_num(r, 'run_active') for r in rows]
    if edge_seq is not None:
        i_edge = next((i for i in range(len(rows)) if seq[i] == edge_seq), None)
        if i_edge is None or ra[i_edge] != 1:
            return True, ['khong_co_dong_canh'], {'edge_seq': edge_seq, 'loai': 'he_do'}
    else:
        i_edge = next((i for i in range(1, len(rows)) if ra[i] == 1 and ra[i - 1] == 0), None)
        if i_edge is None:
            first = next((i for i in range(len(rows)) if ra[i] == 1), None)
            if first is None:
                return True, ['no_edge'], {'loai': 'duong_lenh'}
            i_edge = first                    # active without a readable 0 -> 1 change: no edge evidence
    e = rows[i_edge]
    info['edge_seq'] = seq[i_edge]
    if (i_edge == 0 or seq[i_edge] is None or seq[i_edge - 1] != seq[i_edge] - 1
            or ra[i_edge - 1] != 0):
        return True, ['thieu_dong_truoc_canh'], dict(info, loai='he_do')
    if end_seq is None:
        i_end = max(i for i in range(len(rows)) if ra[i] == 1)
    else:
        if end_seq < seq[i_edge]:
            raise ValueError(f'end_seq {end_seq} before the edge {seq[i_edge]}')
        i_end = next((i for i in range(len(rows)) if seq[i] == end_seq), None)
        if i_end is None:
            return True, ['log_khong_phu_moc_cuoi'], dict(info, loai='he_do')
    latched = (_num(e, 'r_valid') == 1 and (_num(e, 'rv_n') or 0) > 0 and _num(e, 'head_on') == 1
               and _num(e, 'psi_ref') == 0.0)
    if not latched:
        reasons.append('edge_not_latched')
    win = rows[i_edge:i_end + 1]
    for c in ('psi_reset_count', 'reset_n', 'pos_reset_n'):
        base = _num(e, c)
        vals = {_num(r, c) for r in win}
        if base is None or vals != {base}:
            reasons.append(f'{c}_changed')
            info[c] = sorted(vals, key=lambda v: (v is None, v))
    age = _num(e, 'age_r_ms')
    nxt = rows[i_edge + 1] if i_edge + 1 < len(rows) else None
    gap_us = None
    if nxt is not None and seq[i_edge + 1] == seq[i_edge] + 1:
        t0, t1 = _num(e, 'timestamp_us'), _num(nxt, 'timestamp_us')
        if t0 is not None and t1 is not None and t1 > t0:
            gap_us = int(t1 - t0)
    age_us = None if age is None else int(round(age * 10)) * 100
    if (not latched or age_us is None or gap_us is None
            or not (-gap_us - 50 <= age_us <= T_AGE_US + 50)):
        reasons.append('age_bound_undetermined')
    else:
        bound_us = age_us + 50 + gap_us + 3
        info.update(age_r_ms=age, gap_ms=gap_us / 1000, age_bound_ms=bound_us / 1000)
        if Decimal(bound_us) > Decimal(str(age_max_ms)) * 1000:
            reasons.append('age_bound_over')
    if reasons:
        info['loai'] = 'duong_lenh'
    return bool(reasons), reasons, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--age-max-ms', type=float, default=60.0,
                    help='demo threshold for the age bound (J locks it at C8; not a proposal)')
    a = ap.parse_args(argv)
    fw = fwsim.FirmwareSim()
    if fw.pos_cols == 0:
        sys.exit('fwsim.dll has no position loop: build it from firmware with pos_ctrl.c')
    os.makedirs(a.out, exist_ok=True)
    summary = {'fwsim_build': fw.build_info, 'age_max_ms': a.age_max_ms, 'cases': {}}
    for c in cases():
        rows = as_logged(simulate(fw, c))
        with open(os.path.join(a.out, f'{c.name}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for r in rows:
                w.writerow([FMT[k].format(r[k]) if k in FMT else r[k] for k in COLS])
        bad, why, info = ref_cond1(rows, a.age_max_ms)
        summary['cases'][c.name] = {'hong_duong_lenh_1': bad, 'ly_do': why, **info}
        print(f'{c.name:14s} bad={bad!s:5s} {why} {info}')
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
