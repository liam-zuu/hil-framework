"""
s01_replay_plant.py: replay the spec S01 bench tests on the HIL plant model.

Runs the same plan as s01_openloop_runner.py (same segments, seeds, levels) on
rpi5/plant_step.py with wheels lifted, samples the encoder count at 1 kHz with
1320 counts/rev like the ESP32, and writes the same files (raw/*.csv, runs.csv,
plan.csv, meta.yaml). The output can be fed to matlab/identification/s01_identify.m
and compared with the measured run (plant fidelity check, thesis 5.1).

Usage
  python tools/s01_replay_plant.py --out <dir> [--reps 5] [--wheels 0 1 2 3] [--measured <run dir>]

The command follows the real test: firmware motor_dir = +1 (as in S01 run01) and
the plant's motor_polarity, so the signs match the measured data.
"""
import argparse
import csv
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))

import numpy as np                                   # noqa: E402
import s01_openloop_runner as runner                 # noqa: E402
from params import MecanumParams                     # noqa: E402
from plant_step import plant_step                    # noqa: E402

COUNTS_PER_REV = runner.COUNTS_PER_REV


def simulate(params, wheel, segs, v_batt):
    """Return rows (t_us, count, pwm_cmd, en) like the ESP32 test mode."""
    p = params
    p.V_batt = v_batt
    x = np.zeros(10)
    theta = 0.0
    rows, t_ms = [], 0
    rng = np.random.default_rng(0)
    for pwm, en, dur in segs:
        p.motor_enable = float(en)
        u = np.zeros(4)
        u[wheel] = pwm / 1000.0 if en else 0.0
        for _ in range(dur):
            x = plant_step(x, u, p, 0.001, rng)
            theta += x[6 + wheel] * 0.001
            count = int(math.floor(theta / (2 * math.pi) * COUNTS_PER_REV + 0.5))
            rows.append((t_ms * 1000, count, pwm if en else 0, en))
            t_ms += 1
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wheels', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--v-batt', type=float, default=12.435, help='S01 run01 mean battery voltage')
    a = ap.parse_args()

    raw_dir = os.path.join(a.out, 'raw')
    if os.path.exists(os.path.join(a.out, 'runs.csv')):
        sys.exit(f'{a.out} already holds data; use a new folder.')
    os.makedirs(raw_dir, exist_ok=True)

    params = MecanumParams()
    params.update_physical(wheels_lifted=True)
    plan = runner.build_plan(a.wheels, a.reps, False, a.seed)
    with open(os.path.join(a.out, 'plan.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(plan[0].keys()))
        wr.writeheader(); wr.writerows(plan)

    fields = ['run_id', 'wheel', 'scenario', 'pwm_level', 'direction', 'rep', 'v_batt_start', 'v_batt_end',
              'file', 'order', 'n_samples', 'aborted', 'reason', 'missing_pct', 'steady_ok', 'omega_ss_rad_s', 'block_ok']
    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='') as fr:
        wr = csv.DictWriter(fr, fieldnames=fields)
        wr.writeheader()
        for p in plan:
            w = p['wheel']
            lv = p['pwm_level'] * (1 if p['direction'] == 'pos' else -1)
            run_id = f"{runner.WHEELS[w]}_{p['scenario']}_{p['direction']}{p['pwm_level']:04d}_r{p['rep']}"
            rows = simulate(params, w, runner.segments(p['scenario'], lv), a.v_batt)
            omega = runner.omega_series(rows)
            fname = os.path.join('raw', run_id + '.csv')
            with open(os.path.join(a.out, fname), 'w', newline='') as f:
                fw = csv.writer(f)
                fw.writerow(['t_us', 'wheel', 'pwm_cmd', 'en', 'count', 'omega_rad_s'])
                for r, om in zip(rows, omega):
                    fw.writerow([r[0], runner.WHEELS[w], r[2], r[3], r[1], f'{om:.4f}'])
            miss, ok, wss = runner.qc(rows, omega, p['scenario'])
            wr.writerow(dict(run_id=run_id, wheel=runner.WHEELS[w], scenario=p['scenario'],
                             pwm_level=p['pwm_level'], direction=p['direction'], rep=p['rep'],
                             v_batt_start=a.v_batt, v_batt_end=a.v_batt, file=fname, order=p['order'],
                             n_samples=len(rows), aborted=0, reason='ok', missing_pct=f'{miss:.2f}',
                             steady_ok=ok, omega_ss_rad_s=wss, block_ok=1))

    try:
        commit = subprocess.run(['git', '-C', os.path.join(HERE, '..'), 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', os.path.join(HERE, '..'), 'status', '--porcelain'],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'UNKNOWN', True
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: S01
run: 1
setup: plant_model_lifted     # simulation, NOT a measurement
operator: Claude (phiên CODE)
firmware:
  rpi5: {{repo: hil-framework, path: rpi5/plant_step.py + tools/s01_replay_plant.py, commit: {commit}, dirty: {str(dirty).lower()}}}
  h7: null
  esp32: null
config:
  plan: "same as S01 run01 (build_plan seed {a.seed}, {a.reps} reps, wheels {a.wheels})"
  v_batt: {a.v_batt}
  motor_model: {params.motor_model}
  motor_polarity: {params.motor_polarity.tolist()}
  plant_params_source: "rpi5/params.py (from 05_DATA/results/S01/2026-10-20_run01/identification/plant_params_from_S01.csv)"
notes: "Replay of the S01 bench tests on the plant model for the fidelity check (compare with run01)."
""")
    print(f'Done: {len(plan)} simulated runs in {a.out}')


if __name__ == '__main__':
    main()
