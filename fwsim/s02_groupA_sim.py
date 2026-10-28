"""
s02_groupA_sim.py: spec S02 P2, group A on the SIM platform (fwsim).

Runs every group A test (rpi5/trajectory_pub.GROUP_A) N times with one
controller, writes each run as an ESP32-format log (tools/agv_logger.py
COLUMNS) so tools/s02_metrics.py treats SIM, HIL and REAL the same way.

  python fwsim/s02_groupA_sim.py --controller PID  --out <05_DATA/results/S02/<date>_runNN>
  python fwsim/s02_groupA_sim.py --controller ADRC --out <...>

Per spec P2-2/P2-4/P2-5: seed = repetition number (1..N); test order shuffled
within each repetition block (order seed recorded); setpoint from
get_body_velocity held at 50 Hz; after the test duration the setpoint is 0 for
1 s (final stop, excluded by the metric window); A-K tests use the plant's
wheels-lifted mode. IMU columns are empty (fwsim has no IMU).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
import fwsim                                            # noqa: E402
from params import MecanumParams                        # noqa: E402
from trajectory_pub import GROUP_A, LIFTED_TESTS, get_body_velocity   # noqa: E402
from agv_logger import COLUMNS                          # noqa: E402
import s02_metrics                                      # noqa: E402

STOP_TAIL_S = 1.0


def log_rows(log):
    n = len(log['t'])
    rpm = log['rpm']
    for i in range(n):
        yield ([int(round(log['t'][i] * 1e6))] + [int(c) for c in log['count'][i]] +
               [f'{x:.2f}' for x in rpm[i]] + [f'{x:.4f}' for x in log['v_fw'][i]] +
               [f'{x:.4f}' for x in log['sp'][i]] + [int(p) for p in log['pwm'][i]] +
               [''] * 9 + [int(log['tick'][i])] + [int(x) for x in log['dpwm'][i]] + [int(x) for x in log['sat'][i]] +
               [''] * (len(COLUMNS) - 37))   # IMU / log-loss columns (AGV ctrl-imu): not filled here yet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--controller', choices=['PID', 'ADRC'], required=True)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--order-seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    os.makedirs(os.path.join(a.out, 'raw'), exist_ok=True)

    fw = fwsim.FirmwareSim()
    if fw.build_info.get('esp32', {}).get('dirty', True):
        print('WARNING: fwsim.dll was built from a dirty firmware tree; not usable as a main result')
    order_rng = np.random.default_rng(a.order_seed)
    runs, t_start = [], time.time()
    for rep in range(1, a.reps + 1):
        order = list(GROUP_A)
        order_rng.shuffle(order)
        for test_id in order:
            spec = GROUP_A[test_id]
            params = MecanumParams()
            if test_id in LIFTED_TESTS:
                params.update_physical(wheels_lifted=True)
            sp = (lambda s: lambda t: get_body_velocity(t, s) if t < s.duration else (0.0, 0.0, 0.0))(spec)
            log = fwsim.simulate(fw, sp, spec.duration + STOP_TAIL_S, params, seed=rep, mode=a.controller)
            run_id = f'{test_id}_{a.controller}_r{rep}'
            path = os.path.join('raw', f'{run_id}.csv')
            with open(os.path.join(a.out, path), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(COLUMNS)
                w.writerows(log_rows(log))
            runs.append(dict(run_id=run_id, platform='sim', controller=a.controller, test_id=test_id,
                             speed_mps=spec.v, rep=rep, seed=rep, v_batt_start='', v_batt_end='',
                             file=path, aborted=0, reason='ok', missing_pct=0))
    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    s02_metrics.run_dir_metrics(a.out)

    b = fw.build_info
    p0 = MecanumParams()
    gains = ({ax: fw.get_adrc(i) for i, ax in enumerate(['vx', 'vy', 'wz'])} if a.controller == 'ADRC'
             else {ax: fw.get_pid(i) for i, ax in enumerate(['vx', 'vy', 'wz'])})
    fw.reset()
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: S02
part: groupA
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
controller: {a.controller}
gains: {json.dumps(gains)}
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
  build_flags: "fwsim shim (no FreeRTOS); main.c glue repeated in fwsim/fw_sim.c"
  h7: null
  rpi5: null
plant: {fwsim.plant_meta(p0)}
plant_extra: {{motor_polarity: {p0.motor_polarity.tolist()}, lifted_tests: {list(LIFTED_TESTS)}}}
reps: {a.reps}
seeds: "seed = rep (1..{a.reps})"
order_seed: {a.order_seed}
setpoint: "trajectory_pub.get_body_velocity held at 50 Hz, then 0 for {STOP_TAIL_S} s"
libraries: {json.dumps(fwsim.lib_versions())}
elapsed_s: {time.time() - t_start:.0f}
files:
  - raw/<run_id>.csv: "ESP32 logger format (tools/agv_logger.py COLUMNS), 100 Hz, IMU columns empty"
  - runs.csv: "spec P2-6 columns + test_id"
  - metrics.csv: "tools/s02_metrics.py, one line per run"
notes: "the plant has no random input (legacy slip model off; ground_model traction is deterministic), so the {a.reps} seeds give identical runs"
""")
    print(f'{len(runs)} runs in {time.time() - t_start:.0f} s -> {a.out}')


if __name__ == '__main__':
    main()
