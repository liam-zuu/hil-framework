"""
IMU report-rate experiment, 20 Hz vs 100 Hz (spec controller-IMU v2.6 section 10;
Liam 10/11/2026, CODE local_69fa8ade, STATUS_CODE L71, L72 / CG12;
03_CODE/docs/2026-11-10_De_xuat_mo_hinh_IMU_va_thi_nghiem_tan_so.md section 4;
fixes of Codex Sol round 1/2, IMN-1, -2, -3/-5, -6, -8, -9, -12, -13).

Protocol (locked before the main run):
  - fwsim with the firmware in build/fwsim.dll, which must be AGV FW_EXPECTED, clean, with
    the DLL and wrapper hashes of build/fwsim_build.json (IMN-8); plant default
    MecanumParams (traction); gains = firmware defaults (read back into meta.json);
    controllers PID and ADRC;
  - IMU: imu_model.ImuNoise (replay of the measured noise of that rate, IMN-3/-5) and
    imu_model.firmware_view (what the firmware gets on HIL, IMN-2); T_age = 2.5 periods
    (fwsim.simulate sets it from rate_hz);
  - tests TESTS_LOCKED: group A of trajectory_pub.GROUP_A without the lifted A-K tests
    (levels 1 and 2: A-L1, A-C1, A-F1, A-S1, A-L2, A-C2, A-F2, A-S2) plus A-W1 (turn on the
    spot: wz 0 / +1.0 / -1.0 / 0 rad/s, 2 s each, 8 s) (IMN-13);
  - each run: PRE_S of setpoint 0 (vehicle at rest; the IMU reports and the heading loop sets
    psi0 before any motion, IMN-1), then the test setpoint (get_body_velocity held at 50 Hz),
    then 0 for STOP_TAIL_S; checked: heading on and both IMU sources valid at the end of the
    rest, no psi_ref reset in the window (else the run stops the experiment);
  - seeds 0..4 (main run: exactly these, IMN-6): seed = plant seed and replay seed; the same
    seed at both rates;
  - metric window: t - PRE_S in [0, test duration) (STOP tick excluded, IMN-12), against
    the plant truth:
      E_wz  = RMS(wz_true - wz_ref)                                             [rad/s]
      E_psi = RMS(wrap(theta_true - theta_true(t_on) - psi_ref)) over the ticks where the
              heading loop is on in BOTH branches of the pair (t_on = first tick with the
              loop on; psi_ref counts from there, IMN-1, IMN-9); this common share must be
              >= COVERAGE_MIN, else the experiment stops                       [rad]
      R_pwm = sum over wheels of sum |duty[k] - duty[k-1]| / window             [1/s]
    any non-finite metric stops the experiment (no pair is dropped, IMN-9);
  - d_X = X(20 Hz) - X(100 Hz) per (test, seed); per controller and seed the mean of d over
    the tests (each test weight 1); mean and SE = sd / sqrt(n_seeds) over the seed means
    (the tests of one seed share one noise stretch, IMN-6);
    100 Hz only if (1) mean(d_wz) > max(delta_wz, 2 SE) or mean(d_psi) > max(delta_psi, 2 SE),
    (2) the other one mean(d) >= -max(delta, 2 SE), (3) mean R_pwm(100) <= 1.10 mean R_pwm(20);
    delta_wz = 1/512 rad/s, delta_psi = 1.2e-4 rad. Otherwise 20 Hz. Different choices of
    the two controllers: Liam decides;
  - development data (V9).

  python fwsim/imu_freq_exp.py --out <dir> [--trial] [--seeds 0 1 2 3 4] [--tests ...]
Without --trial the script stops when the identity checks fail or the seeds differ from 0..4.
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
from build_fwsim import git_state, file_sha256, wrapper_identity  # noqa: E402
from imu_model import ImuNoise, load_replay  # noqa: E402
from trajectory_pub import GROUP_A, LIFTED_TESTS, get_body_velocity  # noqa: E402

FW_EXPECTED = '54b7269'          # AGV fix/imu-init-timeout: the firmware the noise was measured with
PRE_S = 1.0
STOP_TAIL_S = 1.0
RATES = (20.0, 100.0)
CONTROLLERS = ('PID', 'ADRC')
SEEDS_MAIN = (0, 1, 2, 3, 4)
TESTS_LOCKED = ('A-L1', 'A-C1', 'A-F1', 'A-S1', 'A-L2', 'A-C2', 'A-F2', 'A-S2', 'A-W1')
DELTA = dict(E_wz=1.0 / 512.0, E_psi=1.2e-4)
RPWM_MAX_INCREASE = 0.10
COVERAGE_MIN = 0.95
C = {n: i for i, n in enumerate(fwsim.IMU_COLS)}


class ExperimentError(RuntimeError):
    """A run or pair the protocol cannot score: the experiment stops (no silent drop)."""


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
    if tuple(x[0] for x in out) != TESTS_LOCKED:
        raise ExperimentError(f'test set {[x[0] for x in out]} differs from TESTS_LOCKED')
    return out


def setpoint_of(sp_fn, dur):
    """Rest for PRE_S, the test for dur (t - PRE_S < dur), then 0."""
    def sp(t):
        u = t - PRE_S
        return sp_fn(u) if 0.0 <= u < dur else (0.0, 0.0, 0.0)
    return sp


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def run_arrays(log, dur):
    """Arrays of one run, with the checks of the protocol (ExperimentError if one fails)."""
    t = np.asarray(log['t'])
    imu = np.asarray(log['imu'])
    w = (t - PRE_S >= 0.0) & (t - PRE_S < dur)
    if not w.any():
        raise ExperimentError('empty metric window')
    pre = np.nonzero(t < PRE_S)[0]
    if len(pre) == 0:
        raise ExperimentError('no tick before the motion')
    k = pre[-1]
    if not (imu[k, C['head_on']] > 0.5 and imu[k, C['g_valid']] > 0.5 and imu[k, C['r_valid']] > 0.5):
        raise ExperimentError(f'IMU / heading not ready at the end of the rest (t={t[k]:.3f})')
    if np.any(imu[w | (t < PRE_S), C['psi_reset_count']] != 0):
        raise ExperimentError('psi_ref was reset (psi_reset_count > 0)')
    on = imu[:, C['head_on']] > 0.5
    k_on = int(np.argmax(on))
    th = np.asarray(log['pose'])[:, 2]
    return dict(t=t, w=w, on=on, theta_rel=th - th[k_on], psi_ref=imu[:, C['psi_ref']],
                v=np.asarray(log['v_true']), sp=np.asarray(log['sp']), duty=np.asarray(log['duty']),
                t_on=float(t[k_on]))


def pair_metrics(a20, a100):
    """Metrics of the two branches of one (controller, test, seed) on the common window."""
    if not (np.array_equal(a20['t'], a100['t']) and np.array_equal(a20['w'], a100['w'])):
        raise ExperimentError('the two branches have different tick grids')
    w = a20['w']
    common = w & a20['on'] & a100['on']
    cov = float(common.sum() / w.sum())
    if cov < COVERAGE_MIN:
        raise ExperimentError(f'common heading coverage {cov:.3f} < {COVERAGE_MIN}')
    out = {}
    for rate, a in ((20.0, a20), (100.0, a100)):
        e_wz = float(np.sqrt(np.mean((a['v'][w, 2] - a['sp'][w, 2]) ** 2)))
        e_psi = float(np.sqrt(np.mean(wrap(a['theta_rel'][common] - a['psi_ref'][common]) ** 2)))
        tw = a['t'][w]
        r_pwm = float(np.abs(np.diff(a['duty'][w], axis=0)).sum() / float(tw[-1] - tw[0]))
        m = dict(E_wz=e_wz, E_psi=e_psi, R_pwm=r_pwm, coverage=cov, t_on=a['t_on'])
        if not all(math.isfinite(v) for v in m.values()):
            raise ExperimentError(f'non-finite metric {m}')
        out[rate] = m
    return out


def decide(rows):
    """rows: dicts with controller, test, seed, rate_hz and the metrics. Returns per controller."""
    out = {}
    for ctrl in CONTROLLERS:
        r = {(x['test'], x['seed'], x['rate_hz']): x for x in rows if x['controller'] == ctrl}
        seeds = sorted({k[1] for k in r})
        tl = sorted({k[0] for k in r})
        missing = [(t, s, q) for t in tl for s in seeds for q in RATES if (t, s, q) not in r]
        if missing:
            raise ExperimentError(f'{ctrl}: missing runs {missing[:5]}')
        res = dict(n_seeds=len(seeds), n_tests=len(tl), seeds=seeds, tests=tl)
        for m in ('E_wz', 'E_psi'):
            per_seed = np.array([np.mean([r[(t, s, 20.0)][m] - r[(t, s, 100.0)][m] for t in tl]) for s in seeds])
            n = len(per_seed)
            if n < 2 or not np.all(np.isfinite(per_seed)):
                raise ExperimentError(f'{ctrl} {m}: need >= 2 finite seed means, got {per_seed}')
            mean = float(per_seed.mean())
            se = float(per_seed.std(ddof=1) / math.sqrt(n))
            thr = max(DELTA[m], 2 * se)
            res[m] = dict(n=n, per_seed_d=per_seed.tolist(), mean_d=mean, se=se, threshold=thr,
                          improves=bool(mean > thr), not_worse=bool(mean >= -thr))
        r20 = float(np.mean([r[(t, s, 20.0)]['R_pwm'] for t in tl for s in seeds]))
        r100 = float(np.mean([r[(t, s, 100.0)]['R_pwm'] for t in tl for s in seeds]))
        res['R_pwm'] = dict(mean_20=r20, mean_100=r100, increase=r100 / r20 - 1.0,
                            ok=bool(r100 <= (1.0 + RPWM_MAX_INCREASE) * r20))
        a, b = res['E_wz'], res['E_psi']
        c1 = a['improves'] or b['improves']
        c2 = (a['not_worse'] if b['improves'] else True) and (b['not_worse'] if a['improves'] else True)
        res['rule'] = dict(c1=c1, c2=bool(c2), c3=res['R_pwm']['ok'])
        res['choice_hz'] = 100 if (c1 and c2 and res['R_pwm']['ok']) else 20
        out[ctrl] = res
    return out


def identity_check(fw, repo):
    """Main-run gate (IMN-8): the DLL is the clean build of FW_EXPECTED this repo's wrapper made."""
    b = fw.build_info
    esp = b.get('esp32', {})
    problems = []
    if not str(esp.get('commit', '')).startswith(FW_EXPECTED) or esp.get('dirty', True):
        problems.append(f"firmware {esp.get('commit')} dirty={esp.get('dirty')} (need {FW_EXPECTED}, clean)")
    dll = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
    if b.get('dll_sha256') != file_sha256(dll):
        problems.append('fwsim.dll differs from the dll_sha256 of fwsim_build.json')
    if b.get('wrapper_sha256') != wrapper_identity():
        problems.append('fw_sim.c / shim differ from the build (rebuild fwsim)')
    if repo['dirty']:
        problems.append(f"hil-framework dirty: {repo['modified']}")
    return problems


def read_back(fw):
    """Gains and fixed parameters the firmware holds (after reset, per mode)."""
    out = {}
    for ctrl in CONTROLLERS:
        fw.reset()
        fw.set_mode(ctrl)
        out[ctrl] = dict(pid={ax: fw.get_pid(ax) for ax in range(3)},
                         adrc={ax: fw.get_adrc(ax) for ax in range(3)},
                         pos=fw.pos_params() if fw.pos_cols else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS_MAIN))
    ap.add_argument('--tests', nargs='+', default=None, help='subset (tool check only, with --trial)')
    ap.add_argument('--trial', action='store_true', help='tool check: not a result')
    a = ap.parse_args()
    if a.tests and not a.trial:
        sys.exit('--tests needs --trial')
    fw = fwsim.FirmwareSim()
    repo = git_state(os.path.join(HERE, '..'))
    if not a.trial:
        if tuple(a.seeds) != SEEDS_MAIN:
            sys.exit(f'main run needs seeds {SEEDS_MAIN}, got {a.seeds} (IMN-6)')
        problems = identity_check(fw, repo)
        if problems:
            sys.exit('identity check failed (IMN-8): ' + '; '.join(problems) + ' -- use --trial for a tool check')
    os.makedirs(a.out, exist_ok=False)
    gains = read_back(fw)
    rows, t_start = [], time.time()
    tl = [x for x in tests() if not a.tests or x[0] in a.tests]
    try:
        for ctrl in CONTROLLERS:
            for tid, dur, sp_fn in tl:
                sp = setpoint_of(sp_fn, dur)
                for seed in a.seeds:
                    arr = {}
                    for rate in RATES:
                        log = fwsim.simulate(fw, sp, PRE_S + dur + STOP_TAIL_S, seed=seed, mode=ctrl,
                                             imu=dict(rate_hz=rate, noise=ImuNoise(rate, seed=seed)))
                        try:
                            arr[rate] = run_arrays(log, dur)
                        except ExperimentError as e:
                            raise ExperimentError(f'{ctrl} {tid} seed {seed} {rate:g} Hz: {e}')
                    try:
                        pm = pair_metrics(arr[20.0], arr[100.0])
                    except ExperimentError as e:
                        raise ExperimentError(f'{ctrl} {tid} seed {seed}: {e}')
                    for rate in RATES:
                        rows.append(dict(controller=ctrl, test=tid, seed=seed, rate_hz=rate, **pm[rate]))
                print(f'{ctrl} {tid} done ({time.time() - t_start:.0f} s)', flush=True)
        dec = decide(rows)
    except ExperimentError as e:
        with open(os.path.join(a.out, 'stopped.txt'), 'w', encoding='utf-8') as f:
            f.write(str(e) + '\n')
        print(f'#ket_qua,ma=3,stopped={e}')
        sys.exit(3)
    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, 'decision.json'), 'w', encoding='utf-8') as f:
        json.dump(dec, f, indent=2)
    plant = fwsim.plant_meta(fwsim.MecanumParams())
    blobs = {}
    for f in ('rpi5/imu_model.py', 'fwsim/imu_freq_exp.py', 'fwsim/fwsim.py',
              'rpi5/data/imu_noise_replay_20hz.csv', 'rpi5/data/imu_noise_replay_100hz.csv'):
        try:
            blobs[f] = subprocess.run(['git', '-C', os.path.join(HERE, '..'), 'rev-parse', f'HEAD:{f}'],
                                      capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            blobs[f] = 'UNKNOWN'
    meta = dict(spec='spec controller-IMU v2.6 section 10 (IMU rate); L71, L72', trial=a.trial,
                date=time.strftime('%Y-%m-%d'), elapsed_s=round(time.time() - t_start, 1),
                hil_framework=repo, fwsim_build=fw.build_info, fw_expected=FW_EXPECTED,
                firmware_read_back=gains, plant=plant, blobs=blobs,
                replay={str(r): dict(file_sha256=load_replay(r).sha256, **load_replay(r).header) for r in RATES},
                seeds=a.seeds, rates=RATES, controllers=CONTROLLERS, tests=[x[0] for x in tl],
                pre_s=PRE_S, stop_tail_s=STOP_TAIL_S, delta=DELTA, rpwm_max_increase=RPWM_MAX_INCREASE,
                coverage_min=COVERAGE_MIN)
    with open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    for ctrl, r in dec.items():
        print(f"#ket_qua,controller={ctrl},choice_hz={r['choice_hz']},d_wz={r['E_wz']['mean_d']:.3g},"
              f"se_wz={r['E_wz']['se']:.3g},d_psi={r['E_psi']['mean_d']:.3g},se_psi={r['E_psi']['se']:.3g},"
              f"rpwm_inc={r['R_pwm']['increase']:+.3f}")


if __name__ == '__main__':
    main()
