"""T9 (O1 v11.41 appendix J v6 section 5 command-path condition (1), section 7): three test cases of the
heading-loop latch on fwsim, i.e. the ESP32 firmware's own heading.c / imu_fusion.c / pos_ctrl.c /
vel_ctrl.c (AGV 5bde4c5 when the DLL is built from it), plus a normal run as the control.

Cases (car standing still, as before every O1 run; the cases test the firmware's logic, not dynamics):
  ok          yaw valid at the rising edge of run_active: latched at the edge (control, must pass)
  mat_yaw     yaw lost at the edge (no game-RV report for > t_age), never back during the run
  chot_muon   late latch: (a) yaw lost at the edge, back 300 ms later, psi0 latched then;
              (b) yaw valid but old at the edge (no report for 90 ms < t_age 125 ms)
  reset_giua  reset in the middle of the run: (a) chip reset (SH2 reset, reset_n + 1);
              (b) yaw lost for 1 s while wz_ref turns psi_ref, back with |e| > e_psi_reset;
              (c) run_active drops for 300 ms (trajectory timeout) and comes back
  canh_doi_mode  extra: PID <-> LADRC change on the tick before the edge (yaw valid, latch 1 tick late)

Each case writes the log rows the firmware would push (main.c push_log, logger.c columns of the same
name and print format) and judges them with ref_cond1(): a reference reading of J section 5 (1) for
the CAM scorer, NOT the scorer. The age threshold is a parameter: J locks it at C8; the default here
is only for the demo and is not a proposal.

  python fwsim/t9_heading_cases.py --out <dir> [--age-max-ms 60]

rv_n and reset_n are the counters the firmware copies from imu_data_t (main.c push_log: imu->rv_n,
imu->reset_n); fw_sim.c keeps them in the same struct but does not export them, so this script
counts the reports and resets it hands to the DLL (same increments as fw_sim.c).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim  # noqa: E402

TICK_US = fwsim.FW_DT_US          # control_task 100 Hz
RV_PERIOD_US = 50_000             # IMU_REPORT_PERIOD_US of the main env and ladrc (20 Hz, Liam 10/11)
RV_PHASE_US = 3_700               # report phase against the tick grid (any value not on the grid)
T0_US = 1_000_000
YAW0_DEG = 30.0                   # yaw of the standing car (any value: psi0 takes it)
EDGE_US = T0_US + 2_000_000       # run_active rises on the first tick at or after this time
RUN_US = 3_000_000
SP_RUN = (0.0, 0.0, 0.3)          # wz_ref during the run, so psi_ref integrates (counts stay 0: car held)

# Columns as logger.c names them; formats as logger.c prints them
COLS = ['timestamp_us', 'log_seq', 'motor_on', 'run_active', 'r_valid', 'age_r_ms', 'rv_n', 'reset_n',
        'head_on', 'psi_ref', 'e_psi', 'psi_reset_count', 'pos_reset_n', 'yaw_rad', 'wz_cmd']
FMT = {'age_r_ms': '{:.1f}', 'psi_ref': '{:.4f}', 'e_psi': '{:.4f}', 'yaw_rad': '{:.4f}', 'wz_cmd': '{:.4f}'}


class Case:
    """Schedule of one case: which game-RV reports are dropped, chip resets, run_active stretches."""

    def __init__(self, name, drop=(), resets=(), run=((EDGE_US, EDGE_US + RUN_US),), mode_at=()):
        self.name = name
        self.drop = drop          # [(t0, t1)): no RV/gyro report with t in the window
        self.resets = resets      # chip reset times (fwsim_imu_reset)
        self.run = run            # [(t0, t1)): run_active = 1 on ticks in a window
        self.mode_at = mode_at    # tick times after which vel_ctrl_set_mode() runs (gain set, main.c 10a)

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
        # extra: PID <-> LADRC change applied on the tick just before the edge (gain set, motor off):
        # reset_event on the edge tick, yaw valid, latch one tick late
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
                fw.imu_report('rv', te, YAW0_DEG)
                rv_n += 1
        ra = case.run_active(t)
        fw.set_run(ra)
        fw.tick(t, (0, 0, 0, 0), SP_RUN if ra else (0.0, 0.0, 0.0))
        if t in case.mode_at:                 # main.c: gain set after vel_ctrl_step, seen next tick
            fw.set_mode('ADRC' if fw.ctrl_mode == 'PID' else 'PID')
        m = dict(zip(fwsim.IMU_COLS, fw.imu_last()))
        p = dict(zip(fwsim.POS_COLS, fw.pos_last()))
        seq += 1
        rows.append({'timestamp_us': t, 'log_seq': seq, 'motor_on': 1, 'run_active': int(p['run_active']),
                     'r_valid': int(m['r_valid']), 'age_r_ms': m['age_r_ms'], 'rv_n': rv_n, 'reset_n': reset_n,
                     'head_on': int(m['head_on']), 'psi_ref': m['psi_ref'], 'e_psi': m['e_psi'],
                     'psi_reset_count': int(m['psi_reset_count']), 'pos_reset_n': int(p['pos_reset_n']),
                     'yaw_rad': m['yaw'], 'wz_cmd': m['wz_cmd']})
        t += TICK_US
    return rows


def as_logged(rows):
    """The rows as the logger prints them (floats rounded by logger.c's format), then parsed back."""
    out = []
    for r in rows:
        out.append({c: (float(FMT[c].format(r[c])) if c in FMT else int(r[c])) for c in COLS})
    return out


def ref_cond1(rows, age_max_ms):
    """Reference reading of J v6 section 5 command-path condition (1) on log rows of one run (sorted by
    log_seq). Returns (bad, reasons, info). The run window here: the first run_active rising edge to the
    last row with run_active = 1 (the scorer uses its own end anchor); a run_active drop inside it is a
    firmware pose reset (pos_reset_n changes on the re-edge).

    Edge row (O1 v11.41 v8, T9 rule): r_valid = 1, head_on = 1, psi_ref = 0.
    Whole window: psi_reset_count, reset_n, pos_reset_n constant.
    Yaw age at the latch (J section 5 (1)): upper bound = age_r_ms (log, 0.1 ms) + 0.05 ms + gap to the
    next row (rigorous bound of tick start -> latch, T9 doc section 2); missing next row (log_seq not +1),
    missing age or flags -> not determinable -> bad."""
    reasons = []
    info = {}
    i_edge = next((i for i in range(1, len(rows))
                   if rows[i]['run_active'] == 1 and rows[i - 1]['run_active'] != 1), None)
    if i_edge is None:
        return True, ['no_edge'], info
    e = rows[i_edge]
    info['edge_seq'] = e['log_seq']
    if not (e['r_valid'] == 1 and e['head_on'] == 1 and e['psi_ref'] == 0.0):
        reasons.append('edge_not_latched')
    i_end = max(i for i in range(len(rows)) if rows[i]['run_active'] == 1)
    win = rows[i_edge:i_end + 1]
    for c in ('psi_reset_count', 'reset_n', 'pos_reset_n'):
        vals = sorted({r[c] for r in win})
        if len(vals) > 1:
            reasons.append(f'{c}_changed')
            info[c] = vals
    nxt = rows[i_edge + 1] if i_edge + 1 < len(rows) else None
    if nxt is None or nxt['log_seq'] != e['log_seq'] + 1 or e['r_valid'] != 1 or not math.isfinite(e['age_r_ms']):
        reasons.append('age_bound_undetermined')
    else:
        gap_ms = (nxt['timestamp_us'] - e['timestamp_us']) / 1000.0
        bound = e['age_r_ms'] + 0.05 + gap_ms
        info.update(age_r_ms=e['age_r_ms'], gap_ms=gap_ms, age_bound_ms=round(bound, 3))
        if bound > age_max_ms:
            reasons.append('age_bound_over')
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
