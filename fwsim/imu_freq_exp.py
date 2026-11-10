"""
IMU report-rate experiment, 20 Hz vs 100 Hz (spec controller-IMU v2.6 section 10;
Liam 10/11/2026, CODE local_69fa8ade, STATUS_CODE L71;
03_CODE/docs/2026-11-10_De_xuat_mo_hinh_IMU_va_thi_nghiem_tan_so.md section 4).

Protocol (locked before the run):
  - fwsim with the firmware in build/fwsim.dll, plant default MecanumParams (traction),
    gains = firmware defaults (initial gains), controllers PID and ADRC;
  - IMU: imu_model.ImuNoise with the measured spec of each rate, T_age = 2.5 periods
    (fwsim.simulate sets it from rate_hz);
  - tests: group A (trajectory_pub.GROUP_A) without the lifted A-K tests, plus A-W1
    (turn on the spot: wz 0 / +1.0 / -1.0 / 0 rad/s, 2 s each, 8 s); setpoint
    get_body_velocity held at 50 Hz, then 0 for STOP_TAIL_S;
  - seeds 0..4: seed = plant seed and IMU-noise seed; the same seed at both rates;
  - metrics on t in [T0, test duration] against the plant truth:
      E_wz  = RMS(wz_true - wz_ref)                                  [rad/s]
      E_psi = RMS(wrap(theta_true - psi_ref)) over ticks with head_on [rad] (NaN if none)
      R_pwm = sum over wheels of sum |duty[k] - duty[k-1]| / window   [1/s]
  - pairs d_X = X(20 Hz) - X(100 Hz) per (test, seed), per controller;
    100 Hz only if (1) mean(d_wz) > max(delta_wz, 2 SE) or mean(d_psi) > max(delta_psi, 2 SE),
    (2) the other one mean(d) >= -max(delta, 2 SE), (3) mean R_pwm(100) <= 1.10 mean R_pwm(20);
    delta_wz = 1/512 rad/s, delta_psi = 1.2e-4 rad. Otherwise 20 Hz. SE = sd / sqrt(n).
  - development data (V9).

  python fwsim/imu_freq_exp.py --out <dir> [--trial] [--seeds 0 1 2 3 4]
Without --trial the script stops when the firmware DLL or this repo is dirty.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
import fwsim  # noqa: E402
from build_fwsim import git_state  # noqa: E402
from imu_model import ImuNoise, noise_spec_for_rate  # noqa: E402
from trajectory_pub import GROUP_A, LIFTED_TESTS, get_body_velocity  # noqa: E402

STOP_TAIL_S = 1.0
T0 = 0.5
RATES = (20.0, 100.0)
CONTROLLERS = ('PID', 'ADRC')
DELTA = dict(E_wz=1.0 / 512.0, E_psi=1.2e-4)
RPWM_MAX_INCREASE = 0.10
C = {n: i for i, n in enumerate(fwsim.IMU_COLS)}


def a_w1(t):
    """A-W1: turn on the spot, wz 0 / +1 / -1 / 0 rad/s, 2 s each."""
    wz = 0.0 if t < 2.0 else 1.0 if t < 4.0 else -1.0 if t < 6.0 else 0.0
    return 0.0, 0.0, wz


def tests():
    out = []
    for tid, spec in GROUP_A.items():
        if tid in LIFTED_TESTS:
            continue
        out.append((tid, spec.duration, (lambda s: lambda t: get_body_velocity(t, s))(spec)))
    out.append(('A-W1', 8.0, a_w1))
    return out


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def metrics(log, duration):
    t = np.asarray(log['t'])
    w = (t >= T0) & (t <= duration)
    v = np.asarray(log['v_true'])[w]
    sp = np.asarray(log['sp'])[w]
    e_wz = float(np.sqrt(np.mean((v[:, 2] - sp[:, 2]) ** 2)))
    imu = np.asarray(log['imu'])[w]
    th = np.asarray(log['pose'])[w][:, 2]
    head = imu[:, C['head_on']] > 0.5
    e_psi = float(np.sqrt(np.mean(wrap(th[head] - imu[head, C['psi_ref']]) ** 2))) if head.any() else float('nan')
    duty = np.asarray(log['duty'])[w]
    win = float(t[w][-1] - t[w][0])
    r_pwm = float(np.abs(np.diff(duty, axis=0)).sum() / win)
    return dict(E_wz=e_wz, E_psi=e_psi, R_pwm=r_pwm, head_frac=float(head.mean()))


def decide(rows):
    """rows: dicts with controller, test, seed, rate_hz and the metrics. Returns per controller."""
    out = {}
    for ctrl in CONTROLLERS:
        r = {(x['test'], x['seed'], x['rate_hz']): x for x in rows if x['controller'] == ctrl}
        keys = sorted({(k[0], k[1]) for k in r})
        res = dict(n_pairs=len(keys))
        for m in ('E_wz', 'E_psi'):
            d = np.array([r[(tid, s, 20.0)][m] - r[(tid, s, 100.0)][m] for tid, s in keys])
            d = d[np.isfinite(d)]
            n = len(d)
            mean = float(d.mean()) if n else float('nan')
            se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float('nan')
            thr = max(DELTA[m], 2 * se) if n > 1 else float('nan')
            res[m] = dict(n=n, mean_d=mean, se=se, threshold=thr,
                          improves=bool(n > 1 and mean > thr), not_worse=bool(n > 1 and mean >= -thr))
        r20 = float(np.mean([r[(tid, s, 20.0)]['R_pwm'] for tid, s in keys]))
        r100 = float(np.mean([r[(tid, s, 100.0)]['R_pwm'] for tid, s in keys]))
        res['R_pwm'] = dict(mean_20=r20, mean_100=r100, increase=r100 / r20 - 1.0,
                            ok=bool(r100 <= (1.0 + RPWM_MAX_INCREASE) * r20))
        a, b = res['E_wz'], res['E_psi']
        c1 = a['improves'] or b['improves']
        c2 = (a['not_worse'] if b['improves'] else True) and (b['not_worse'] if a['improves'] else True)
        res['rule'] = dict(c1=c1, c2=bool(c2), c3=res['R_pwm']['ok'])
        res['choice_hz'] = 100 if (c1 and c2 and res['R_pwm']['ok']) else 20
        out[ctrl] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--tests', nargs='+', default=None, help='subset (tool check only, with --trial)')
    ap.add_argument('--trial', action='store_true', help='tool check: not a result')
    a = ap.parse_args()
    if a.tests and not a.trial:
        sys.exit('--tests needs --trial')
    fw = fwsim.FirmwareSim()
    repo = git_state(os.path.join(HERE, '..'))
    if not a.trial and (fw.build_info.get('esp32', {}).get('dirty', True) or repo['dirty']):
        sys.exit(f'dirty firmware or repo ({repo}); use --trial for a tool check')
    os.makedirs(a.out, exist_ok=False)
    rows, t_start = [], time.time()
    tl = [x for x in tests() if not a.tests or x[0] in a.tests]
    for ctrl in CONTROLLERS:
        for tid, dur, sp_fn in tl:
            sp = (lambda f, d: lambda t: f(t) if t < d else (0.0, 0.0, 0.0))(sp_fn, dur)
            for seed in a.seeds:
                for rate in RATES:
                    noise = ImuNoise(noise_spec_for_rate(rate), seed=seed)
                    log = fwsim.simulate(fw, sp, dur + STOP_TAIL_S, seed=seed, mode=ctrl,
                                         imu=dict(rate_hz=rate, noise=noise))
                    rows.append(dict(controller=ctrl, test=tid, seed=seed, rate_hz=rate, **metrics(log, dur)))
            print(f'{ctrl} {tid} done ({time.time() - t_start:.0f} s)', flush=True)
    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dec = decide(rows)
    with open(os.path.join(a.out, 'decision.json'), 'w', encoding='utf-8') as f:
        json.dump(dec, f, indent=2)
    plant = fwsim.plant_meta(fwsim.MecanumParams())
    try:
        imu_blob = subprocess.run(['git', '-C', os.path.join(HERE, '..'), 'rev-parse', 'HEAD:rpi5/imu_model.py'],
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        imu_blob = 'UNKNOWN'
    meta = dict(spec='spec controller-IMU v2.6 section 10 (IMU rate); L71', trial=a.trial,
                date=time.strftime('%Y-%m-%d'), elapsed_s=round(time.time() - t_start, 1),
                hil_framework=repo, fwsim_build=fw.build_info, plant=plant, imu_model_blob=imu_blob,
                seeds=a.seeds, rates=RATES, controllers=CONTROLLERS, tests=[x[0] for x in tl],
                T0=T0, stop_tail_s=STOP_TAIL_S, delta=DELTA, rpwm_max_increase=RPWM_MAX_INCREASE,
                noise={str(r): str(noise_spec_for_rate(r)) for r in RATES})
    with open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    for ctrl, r in dec.items():
        print(f"#ket_qua,controller={ctrl},choice_hz={r['choice_hz']},d_wz={r['E_wz']['mean_d']:.3g},"
              f"d_psi={r['E_psi']['mean_d']:.3g},rpwm_inc={r['R_pwm']['increase']:+.3f}")


if __name__ == '__main__':
    main()
