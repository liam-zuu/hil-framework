"""
tune_ga.py: multi-objective GA (pymoo NSGA-II) on the firmware's own gains,
evaluated with fwsim (firmware C code + rpi5 plant).

Decision variables (PID, the parameters pid.c actually has):
  kp_xy, ki_xy   applied to both vx and vy (same structure in pid.c)
  kp_wz, ki_wz
  kd = 0 and the integral/output limits stay at the pid.c defaults.

Objectives (minimise), averaged over --seeds plant seeds on the 'steps' trajectory:
  f1 = RMS body-velocity error, sum of rms_vx, rms_vy and L * rms_wz (L = lx + ly = 0.24 m)
  f2 = total variation of the duty per second (actuator effort / chatter)

  python fwsim/tune_ga.py --pop 24 --gen 20 --out <dir>
  python fwsim/tune_ga.py --pop 6 --gen 2 --out <dir>       # smoke test

Output in --out: pareto.csv, history.csv (every evaluation), meta.yaml
(firmware commit and dirty state from fwsim_build.json, library versions, GA settings).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fwsim  # noqa: E402

VARS = ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz']
L_ARM = 0.24


def make_problem(fw, seeds, setpoint, duration, bounds, history, pcnt_mode):
    from pymoo.core.problem import ElementwiseProblem

    class FirmwareGains(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(VARS), n_obj=2, xl=np.array(bounds[0]), xu=np.array(bounds[1]))

        def _evaluate(self, x, out, *args, **kwargs):
            kp_xy, ki_xy, kp_wz, ki_wz = map(float, x)
            pid = {0: (kp_xy, ki_xy, 0.0, 0.5, 1.5), 1: (kp_xy, ki_xy, 0.0, 0.5, 1.5),
                   2: (kp_wz, ki_wz, 0.0, 1.0, 2.0)}
            f1, f2 = [], []
            for s in seeds:
                m = fwsim.metrics(fwsim.simulate(fw, setpoint, duration, seed=s, pid=pid, pcnt_mode=pcnt_mode))
                f1.append(m['rms_vx'] + m['rms_vy'] + L_ARM * m['rms_wz'])
                f2.append(m['tv_duty'])
            out['F'] = [float(np.mean(f1)), float(np.mean(f2))]
            history.append(dict(zip(VARS, map(float, x)), f1=out['F'][0], f2=out['F'][1]))

    return FirmwareGains()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pop', type=int, default=24)
    ap.add_argument('--gen', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--ga-seed', type=int, default=1)
    ap.add_argument('--speed', type=float, default=0.3)
    ap.add_argument('--radius', type=float, default=0.5)
    ap.add_argument('--pcnt-mode', default='idf', choices=list(fwsim.PCNT_MODES))
    ap.add_argument('--lower', type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0])
    ap.add_argument('--upper', type=float, nargs=4, default=[6.0, 10.0, 6.0, 10.0])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    os.makedirs(a.out, exist_ok=True)

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    fw = fwsim.FirmwareSim()
    sp = fwsim.make_setpoint(SimpleNamespace(traj='steps', speed=a.speed, radius=a.radius, period=8.0))
    history = []
    problem = make_problem(fw, a.seeds, sp, sp.duration, (a.lower, a.upper), history, a.pcnt_mode)
    t0 = time.time()
    res = minimize(problem, NSGA2(pop_size=a.pop), ('n_gen', a.gen), seed=a.ga_seed, verbose=False)
    elapsed = time.time() - t0

    X, F = np.atleast_2d(res.X), np.atleast_2d(res.F)
    order = np.argsort(F[:, 0])
    with open(os.path.join(a.out, 'pareto.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(VARS + ['f1_rms_body', 'f2_tv_duty'])
        for i in order:
            w.writerow([f'{v:.6g}' for v in X[i]] + [f'{F[i, 0]:.6g}', f'{F[i, 1]:.6g}'])
    with open(os.path.join(a.out, 'history.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=VARS + ['f1', 'f2'])
        w.writeheader(); w.writerows(history)
    b = fw.build_info
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: null
date: {time.strftime('%Y-%m-%d')}
setup: fwsim GA (PC simulation running the ESP32 firmware C code), NOT hardware
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
  ctrl_mode: {fw.ctrl_mode}
ga: {{algorithm: NSGA2, pop: {a.pop}, gen: {a.gen}, ga_seed: {a.ga_seed}, evaluations: {len(history)}, elapsed_s: {elapsed:.1f}}}
variables: {json.dumps(dict(names=VARS, lower=a.lower, upper=a.upper, fixed='kd = 0, limits from pid.c'))}
objectives: "f1 = rms_vx + rms_vy + 0.24 * rms_wz (true plant body velocity vs setpoint); f2 = duty total variation per second; mean over plant seeds"
evaluation: {json.dumps(dict(traj='steps', speed=a.speed, radius=a.radius, seeds=a.seeds, pcnt_mode=a.pcnt_mode))}
libraries: {json.dumps(fwsim.lib_versions())}
""")
    print(f'{len(history)} evaluations in {elapsed:.0f} s; {len(order)} Pareto points -> {a.out}')
    for i in order[:5]:
        print('  ' + '  '.join(f'{n}={v:.3g}' for n, v in zip(VARS, X[i])) + f'  f1={F[i, 0]:.4f} f2={F[i, 1]:.3f}')


if __name__ == '__main__':
    main()
