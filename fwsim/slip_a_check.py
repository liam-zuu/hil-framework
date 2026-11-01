"""
slip_a_check.py: checks T-A2 and T-A7 of the slip method A design note
(docs/2026-10-28_Thiet_ke_cach_A_mo_hinh_truot.md section 6; thresholds fixed in
the note before running, Liam approved 28/10) on fwsim, group A ground tests.

  T-A2  ground_model 'traction' with mu_hd = 5 (almost no slip) vs 'rigid':
        |metric difference| <= 1 x qrms of that metric
  T-A7  'traction' at the nominal mu_hd: sub-steps n vs 2n, and omega_s vs omega_s/2:
        |metric difference| <= 0.5 x qrms

Metrics with a quantization floor in tools/s02_metrics.py (e_vx, e_vy, e_wz,
rms_w_err) carry the threshold. pwm_tv and sat_pct have no floor: reported, no
pass/fail. Lifted tests (A-K*) are left out: T-A1 (rpi5/tests/test_slip_a.py)
shows the lifted plant is bit-identical. One run per test (the plant has no random
input, seeds give identical runs). Each case runs without IMU reports (as
s02_groupA_sim.py) and with 20 Hz IMU reports (wz from the gyro, heading loop on).

  python fwsim/slip_a_check.py --out D:\\CLAUDE_THESIS\\05_DATA\\results\\SLIP_A\\<date>_runNN
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
import fwsim                                            # noqa: E402
from params import MecanumParams                        # noqa: E402
from plant_step import traction_nsub                    # noqa: E402
from trajectory_pub import GROUP_A, LIFTED_TESTS, get_body_velocity   # noqa: E402
from agv_logger import COLUMNS                          # noqa: E402
from s02_groupA_sim import log_rows, STOP_TAIL_S        # noqa: E402
import s02_metrics                                      # noqa: E402

FLOOR_METRICS = ('e_vx', 'e_vy', 'e_wz', 'rms_w_err')
INFO_METRICS = ('pwm_tv', 'sat_pct')
MU_NOMINAL = 0.5            # params.py default mu_hd [ASSUMPTION — pending MU_FLOOR]
MU_NO_SLIP = 5.0


def variants():
    """name -> params factory. The T-A7 n vs 2n pair pins n to the automatic value."""
    n0 = traction_nsub(_traction(MU_NOMINAL))

    def make(**kw):
        def f():
            p = MecanumParams()
            for k, v in kw.items():
                setattr(p, k, v)
            return p
        return f
    return {
        'rigid': make(ground_model='rigid'),
        'tr_mu5': make(ground_model='traction', mu_hd=np.full(4, MU_NO_SLIP)),
        'tr_nom': make(ground_model='traction', mu_hd=np.full(4, MU_NOMINAL)),
        'tr_nom_2n': make(ground_model='traction', mu_hd=np.full(4, MU_NOMINAL), slip_nsub=2 * n0),
        'tr_nom_ws2': make(ground_model='traction', mu_hd=np.full(4, MU_NOMINAL), slip_omega_s=0.05),
    }, n0


def _traction(mu):
    p = MecanumParams()
    p.ground_model, p.mu_hd = 'traction', np.full(4, mu)
    return p


CHECKS = [('T-A2', 'tr_mu5', 'rigid', 1.0),
          ('T-A7_nsub', 'tr_nom_2n', 'tr_nom', 0.5),
          ('T-A7_omega_s', 'tr_nom_ws2', 'tr_nom', 0.5)]


def git_info():
    def g(*a):
        try:
            return subprocess.run(['git', *a], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return 'UNKNOWN'
    return g('rev-parse', '--short', 'HEAD'), g('rev-parse', '--abbrev-ref', 'HEAD'), bool(g('status', '--porcelain'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tests', default='', help='comma list of test IDs (default: every ground test of group A)')
    a = ap.parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    os.makedirs(os.path.join(a.out, 'raw'), exist_ok=True)
    commit, branch, dirty = git_info()

    fw = fwsim.FirmwareSim()
    tests = [t for t in (a.tests.split(',') if a.tests else GROUP_A) if t not in LIFTED_TESTS]
    vmap, n0 = variants()
    t_start = time.time()
    metrics = {}                     # (controller, imu, variant, test) -> metrics dict
    for ctrl in ('PID', 'ADRC'):
        for imu_name, imu in (('noimu', None), ('imu20', {'rate_hz': 20.0})):
            for vname, make in vmap.items():
                for test_id in tests:
                    spec = GROUP_A[test_id]
                    sp = (lambda s: lambda t: get_body_velocity(t, s) if t < s.duration else (0.0, 0.0, 0.0))(spec)
                    p = make()
                    log = fwsim.simulate(fw, sp, spec.duration + STOP_TAIL_S, p, seed=1, mode=ctrl, imu=imu)
                    path = os.path.join(a.out, 'raw', f'{test_id}_{ctrl}_{imu_name}_{vname}.csv')
                    with open(path, 'w', newline='') as f:
                        w = csv.writer(f)
                        w.writerow(COLUMNS)
                        w.writerows(log_rows(log))
                    m = s02_metrics.compute(s02_metrics.load_log(path), test_id)
                    m['r_true_rms'] = float(np.sqrt(np.mean(log['r_true'] ** 2)))
                    m['r_true_max'] = float(np.max(np.abs(log['r_true'])))
                    metrics[(ctrl, imu_name, vname, test_id)] = m
                print(f'{ctrl} {imu_name} {vname}: {len(tests)} tests, {time.time() - t_start:.0f} s', flush=True)

    rows, fails = [], []
    for check, var, base, k in CHECKS:
        for (ctrl, imu_name, vname, test_id), m in metrics.items():
            if vname != var:
                continue
            mb = metrics[(ctrl, imu_name, base, test_id)]
            for name in FLOOR_METRICS + INFO_METRICS:
                q = s02_metrics.QRMS.get(name)
                d = m[name] - mb[name]
                ok = '' if q is None else int(abs(d) <= k * q)
                rows.append(dict(check=check, controller=ctrl, imu=imu_name, test_id=test_id, metric=name,
                                 base=base, base_value=mb[name], variant=var, value=m[name], diff=d,
                                 qrms='' if q is None else q, limit='' if q is None else k * q,
                                 ratio='' if q is None else abs(d) / q, passed=ok))
                if ok == 0:
                    fails.append(rows[-1])
    fmt = lambda x: f'{x:.6g}' if isinstance(x, float) else x   # noqa: E731
    with open(os.path.join(a.out, 'checks.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({kk: fmt(v) for kk, v in r.items()} for r in rows)
    with open(os.path.join(a.out, 'metrics.csv'), 'w', newline='') as f:
        keys = ['controller', 'imu', 'variant', 'test_id', *FLOOR_METRICS, *INFO_METRICS, 'r_true_rms', 'r_true_max']
        w = csv.writer(f)
        w.writerow(keys)
        for (ctrl, imu_name, vname, test_id), m in metrics.items():
            w.writerow([ctrl, imu_name, vname, test_id] + [fmt(m[kk]) for kk in keys[4:]])

    lines = []
    for check, var, base, k in CHECKS:
        sel = [r for r in rows if r['check'] == check and r['passed'] != '']
        worst = max(sel, key=lambda r: r['ratio'])
        lines.append(f"{check} ({var} vs {base}, limit {k} x qrms): {sum(r['passed'] for r in sel)}/{len(sel)} pass; "
                     f"worst {worst['metric']} {worst['test_id']} {worst['controller']} {worst['imu']}: "
                     f"{worst['ratio']:.3f} x qrms")
    for check, var, base, k in CHECKS:
        for name in INFO_METRICS:
            sel = [r for r in rows if r['check'] == check and r['metric'] == name]
            lines.append(f"{check} {name} (no floor, info): max |diff| {max(abs(r['diff']) for r in sel):.4g}")
    r_nom = [m['r_true_max'] for (c, i, v, t), m in metrics.items() if v == 'tr_nom']
    lines.append(f"r_true max over tr_nom runs (info, T-A8 later): {max(r_nom):.4g} rad/s")
    summary = '\n'.join(lines)
    print(summary)
    with open(os.path.join(a.out, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

    b = fw.build_info
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: none            # design note checks, docs/2026-10-28_Thiet_ke_cach_A_mo_hinh_truot.md section 6
part: T-A2, T-A7
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
script: {{repo: hil-framework, path: fwsim/slip_a_check.py, commit: {commit}, branch: {branch}, dirty: {str(dirty).lower()}}}
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework_at_build: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
  h7: null
  rpi5: null
tests: {json.dumps(tests)}
controllers: [PID, ADRC]          # firmware default gains
imu: [none, 20 Hz reports noise-free (rpi5/imu_model.py)]
variants: {{rigid: "ground_model rigid", tr_mu5: "traction mu_hd {MU_NO_SLIP}", tr_nom: "traction mu_hd {MU_NOMINAL}, nsub auto = {n0}", tr_nom_2n: "traction mu_hd {MU_NOMINAL}, nsub {2 * n0}", tr_nom_ws2: "traction mu_hd {MU_NOMINAL}, omega_s 0.05 (nsub auto)"}}
checks: {json.dumps([dict(check=c, variant=v, base=bb, limit_x_qrms=k) for c, v, bb, k in CHECKS])}
seed: 1                           # plant has no random input
libraries: {json.dumps(fwsim.lib_versions())}
elapsed_s: {time.time() - t_start:.0f}
result: "{'PASS' if not fails else f'FAIL ({len(fails)} rows)'}"
files:
  - raw/<test>_<controller>_<imu>_<variant>.csv: "ESP32 logger format, 100 Hz"
  - metrics.csv: "tools/s02_metrics.py metrics per run + r_true rms/max (wz from wheels - body wz)"
  - checks.csv: "one row per check x run x metric: difference, qrms, limit, pass"
  - summary.txt: "pass counts and worst ratio per check"
""")
    print(f'-> {a.out} ({time.time() - t_start:.0f} s), result {"PASS" if not fails else "FAIL"}')
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
