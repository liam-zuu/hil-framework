"""
b0_sensitivity.py: sensitivity of the LADRC result to b0 on fwsim (O1 figure-8 task).

Liam 05/11, CODE session (STATUS_CODE L38): b0 is NOT a GA variable (fixed from S01); instead one
check on fwsim that the result does not depend strongly on b0, as evidence for the thesis. If it
is very sensitive, reconsider.

Method: LADRC with the given gains (default: the firmware defaults; after the GA, pass the chosen
gains with --adrc), b0 of all three axes scaled by each factor in --factors (default 0.7, 0.85,
1.0, 1.15, 1.3, i.e. +-30 %), O1 schedule at the three speeds of o1_feasibility, measures from
o1_feasibility.evaluate (true plant pose). Plant and IMU options as fwsim (--ground, --mu, ...).

Criterion written BEFORE the run (Claude's proposal, Liam may change it): "sensitive" if, at any
speed, a scaled b0 changes 'completed' versus factor 1.0, changes e_perp_max <= D versus factor
1.0, or changes e_perp_max by more than 50 % of the factor-1.0 value. Simulation numbers only.

  python fwsim/b0_sensitivity.py --ground traction --mu 0.5 --out <05_DATA/results/B0_SENS/<date>_runNN>
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
import fwsim                 # noqa: E402
import o1_feasibility as o1  # noqa: E402

FACTORS = (0.7, 0.85, 1.0, 1.15, 1.3)
REL_LIMIT = 0.5              # e_perp_max change > 50 % of the factor-1.0 value = sensitive


def judge(rows):
    """Apply the pre-declared criterion per speed. Returns (sensitive: bool, reasons: list)."""
    reasons = []
    for v in sorted({r['v'] for r in rows}):
        nom = next(r for r in rows if r['v'] == v and r['b0_factor'] == 1.0)
        for r in rows:
            if r['v'] != v or r['b0_factor'] == 1.0:
                continue
            tag = f"v={v} b0x{r['b0_factor']}"
            if r['completed'] != nom['completed']:
                reasons.append(f'{tag}: completed {nom["completed"]} -> {r["completed"]}')
            if r['e_perp_max_le_D'] != nom['e_perp_max_le_D']:
                reasons.append(f'{tag}: e_perp_max <= D {nom["e_perp_max_le_D"]} -> {r["e_perp_max_le_D"]}')
            rel = abs(r['e_perp_max'] - nom['e_perp_max']) / nom['e_perp_max']
            if rel > REL_LIMIT:
                reasons.append(f'{tag}: e_perp_max {nom["e_perp_max"]:.4f} -> {r["e_perp_max"]:.4f} ({rel:.0%})')
    return bool(reasons), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    ap.add_argument('--factors', type=float, nargs='+', default=list(FACTORS))
    ap.add_argument('--adrc', type=float, nargs=4, default=None, metavar=('WC_XY', 'WO_XY', 'WC_WZ', 'WO_WZ'),
                    help='LADRC gains to test (default: firmware defaults)')
    fwsim.add_plant_imu_args(ap)
    a = ap.parse_args()
    if 1.0 not in a.factors:
        sys.exit('--factors must include 1.0 (the reference)')
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    os.makedirs(a.out, exist_ok=True)
    fw = fwsim.FirmwareSim(a.dll)
    if fw.build_info.get('esp32', {}).get('dirty', True):
        print('WARNING: fwsim.dll was built from a dirty firmware tree; not usable as a main result')
    fw.reset()
    base = [fw.get_adrc(i) for i in range(3)]
    gains = ([(a.adrc[0], a.adrc[1]), (a.adrc[0], a.adrc[1]), (a.adrc[2], a.adrc[3])] if a.adrc
             else [(g['omega_c'], g['omega_o']) for g in base])
    imu, path = fwsim.imu_config(a), o1.Path8()
    rows, t0 = [], time.time()
    for v in o1.SPEEDS:
        sch = o1.Schedule(v)
        for k in a.factors:
            adrc = {i: (gains[i][0], gains[i][1], base[i]['b0'] * k, base[i]['out_limit']) for i in range(3)}
            log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, fwsim.plant_params(a), seed=1, mode='ADRC',
                                 imu=imu, adrc=adrc)
            res, _ = o1.evaluate(log, path, sch)
            m = fwsim.metrics(log)
            rows.append(dict(v=v, b0_factor=k, b0_xy=base[0]['b0'] * k, b0_wz=base[2]['b0'] * k,
                             completed=res['completed'], e_perp_max=res['e_perp_max'], e_perp_max_le_D=res['e_perp_max_le_D'],
                             e_perp_rms=res['e_perp_rms'], e_theta_max_deg=res['e_theta_max_deg'],
                             stop_dist_origin=res['stop_dist_origin'], t_end=res['t_end'], sat_frac=res['sat_frac'],
                             tv_duty=m['tv_duty']))
            r = rows[-1]
            print(f"v={v} b0x{k}: completed={r['completed']} e_perp_max={r['e_perp_max']:.4f} "
                  f"stop={r['stop_dist_origin']:.4f} tv_duty={r['tv_duty']:.2f}")
    sensitive, reasons = judge(rows)
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    b = fw.build_info
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: null                    # Liam 05/11, CODE session (STATUS_CODE L38): b0 sensitivity check, evidence for the thesis
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
purpose: "LADRC result vs b0 scaled on all three axes; simulation numbers, not main results"
gains: {json.dumps(dict(source='--adrc' if a.adrc else 'firmware defaults (GA not run yet: re-run with the chosen gains)', omega_c_omega_o=gains))}
b0_default: {json.dumps([g['b0'] for g in base])}     # adrc.c defaults (from S01), axes vx, vy, wz
factors: {json.dumps(a.factors)}
criterion: "written before the run (Claude's proposal): sensitive if any scaled b0 changes completed, or e_perp_max <= D (D = {o1.D} m), versus factor 1.0 at the same speed, or changes e_perp_max by more than {REL_LIMIT:.0%} of the factor-1.0 value"
result: {json.dumps(dict(sensitive=sensitive, reasons=reasons))}
pos_loop: {json.dumps(fwsim.pos_meta(fw))}     # firmware defaults read back (loop, (a), (d) on)
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework_dll: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
plant: {fwsim.plant_meta(fwsim.plant_params(a))}
imu: {json.dumps(imu)}
task: "O1 figure 8 as fwsim/o1_feasibility.py (Schedule, Path8, evaluate), speeds {list(o1.SPEEDS)} m/s"
seed: 1                       # plant and IMU noise-free: the seed has no effect
libraries: {json.dumps(fwsim.lib_versions())}
elapsed_s: {time.time() - t0:.0f}
files:
  - summary.csv: "one line per (speed, b0 factor): completed, e_perp_max, e_perp_max <= D, e_perp_rms, e_theta_max_deg, stop distance to origin, t_end, PWM saturation fraction, duty total variation per second"
""")
    print(f'sensitive={sensitive}' + ''.join(f'\n  {x}' for x in reasons))
    print(f'done in {time.time() - t0:.0f} s -> {a.out}')


if __name__ == '__main__':
    main()
