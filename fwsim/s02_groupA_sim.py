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
wheels-lifted mode.

IMU and plant (28/10, before DOE/GA): the firmware gets BNO085 reports from the plant
(QD 23: every firmware has the IMU; --imu-rate 20 or 100, --no-imu only to re-run the
21/10 runs) and the plant ground model is params.py's default (slip method A
'traction' since plant-slip-a 0928e78; --ground rigid, --mu to change). The log columns
of the IMU fusion / heading loop (AGV ctrl-imu, spec v2.2 section 8) come from the
firmware's own imu_fusion/heading outputs; log_seq counts the lines (no loss in SIM),
motor_on is the firmware's EN. Left empty: the raw chip columns ax..roll (fwsim does not
log the raw report values) and gyro_n, rv_n, reset_n (not exposed by fwsim.dll).
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
from trajectory_pub import GROUP_A, LIFTED_TESTS, get_body_velocity   # noqa: E402
from agv_logger import COLUMNS                          # noqa: E402
import s02_metrics                                      # noqa: E402

STOP_TAIL_S = 1.0


def rep_imu(imu, rep):
    """IMU config of repetition `rep`: with the noise replay on, its seed (start row) is the
    repetition number, the same for every test and both controllers, and the HIL run of the
    same repetition uses the same seed (Liam 11/11 CODE local_7787e22f, as S05 v3.7 section 6.3:
    seed fixed in advance by condition and repetition, shared by every gain set). Before
    11/11 every repetition used seed 0, so the N runs got the same noise segment."""
    if not imu or imu.get('noise') != 'replay':
        return imu
    return dict(imu, noise_seed=int(rep))

# Logger column (agv_logger.COLUMNS after the 37 S02 columns) -> (fwsim.IMU_COLS name, format).
# None = not available in fwsim (left empty).
CTRL_IMU_COLS = {
    'wz_meas': ('wz_meas', '.4f'), 'wz_cmd': ('wz_cmd', '.4f'), 'gz_fused': ('gz', '.4f'),
    'yaw_rad': ('yaw', '.4f'), 'age_g_ms': ('age_g_ms', '.1f'), 'age_r_ms': ('age_r_ms', '.1f'),
    'gyro_n': None, 'rv_n': None, 'reset_n': None,
    'g_valid': ('g_valid', 'd'), 'r_valid': ('r_valid', 'd'), 'wz_src': ('wz_src', 'd'),
    'src_switch_count': ('src_switch_count', 'd'), 'psi_ref': ('psi_ref', '.4f'), 'e_psi': ('e_psi', '.4f'),
    'head_on': ('head_on', 'd'), 'psi_reset_count': ('psi_reset_count', 'd'),
    'slip_flag': ('slip_flag', 'd'), 'slip_on_count': ('slip_on_count', 'd'), 'r': ('r', '.4f'),
    'bias_gz': ('bias_gz', '.5f'), 'src_switch_dwz': ('src_switch_dwz', '.4f'),
    'bias_updates': ('bias_updates', 'd'), 'slip_eval': ('slip_eval', 'd'),
}
# Position-loop and wheel-slip logger columns (agv_logger.COLUMNS 67..85, campaign firmware:
# pos_* then ws_*) -> (fwsim.POS_COLS / fwsim.WS_COLS name, format as logger.c). Left empty
# when the DLL has no pos_ctrl.c / wheel_slip.c (Codex P17, merge int/post-g2 05/11).
POS_LOG_COLS = {
    'pos_on': ('pos_on', 'd'), 'pos_x': ('pos_x', '.4f'), 'pos_y': ('pos_y', '.4f'),
    'pos_th': ('pos_th', '.4f'), 'pos_x_ref': ('pos_x_ref', '.4f'), 'pos_y_ref': ('pos_y_ref', '.4f'),
    'pos_th_ref': ('pos_th_ref', '.4f'), 'pos_corr_vx': ('pos_corr_vx', '.4f'),
    'pos_corr_vy': ('pos_corr_vy', '.4f'), 'pos_th_src': ('pos_th_src', 'd'),
    'pos_in_pos': ('pos_in_pos', 'd'), 'still': ('still', 'd'), 'run_active': ('run_active', 'd'),
    'pos_reset_n': ('pos_reset_n', 'd'),
}
WS_LOG_COLS = {
    'ws_res': ('ws_res', '.0f'), 'ws_th': ('ws_th', '.1f'), 'ws_flag': ('ws_flag', 'd'),
    'ws_on_count': ('ws_on_count', 'd'), 'ws_eval': ('ws_eval', 'd'),
}


def _fmt(v, spec):
    return format(int(round(v)), 'd') if spec == 'd' else format(v, spec)


def log_rows(log, use_imu=False):
    """One logger line per firmware tick, columns as tools/agv_logger.COLUMNS (campaign firmware, 85)."""
    n = len(log['t'])
    rpm = log['rpm']
    tail = COLUMNS[37:]
    imu_idx = {name: fwsim.IMU_COLS.index(name) for name in fwsim.IMU_COLS}
    pos_idx = {name: k for k, name in enumerate(fwsim.POS_COLS)}
    ws_idx = {name: k for k, name in enumerate(fwsim.WS_COLS)}
    has_pos, has_ws = 'pos' in log, 'ws' in log
    for i in range(n):
        extra = {'log_seq': str(i + 1), 'log_qdrop': '0', 'log_send_err': '0',
                 'motor_on': str(int(log['en'][i])), 'imu_init': '0' if use_imu else ''}
        for col, src in CTRL_IMU_COLS.items():
            extra[col] = _fmt(log['imu'][i][imu_idx[src[0]]], src[1]) if (src and use_imu) else ''
        for col, src in POS_LOG_COLS.items():
            extra[col] = _fmt(log['pos'][i][pos_idx[src[0]]], src[1]) if has_pos else ''
        for col, src in WS_LOG_COLS.items():
            extra[col] = _fmt(log['ws'][i][ws_idx[src[0]]], src[1]) if has_ws else ''
        yield ([int(round(log['t'][i] * 1e6))] + [int(c) for c in log['count'][i]] +
               [f'{x:.2f}' for x in rpm[i]] + [f'{x:.4f}' for x in log['v_fw'][i]] +
               [f'{x:.4f}' for x in log['sp'][i]] + [int(p) for p in log['pwm'][i]] +
               [''] * 9 + [int(log['tick'][i])] + [int(x) for x in log['dpwm'][i]] + [int(x) for x in log['sat'][i]] +
               [extra[c] for c in tail])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--controller', choices=['PID', 'ADRC'], required=True)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--order-seed', type=int, default=1)
    fwsim.add_plant_imu_args(ap)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    imu = fwsim.imu_config(a)
    if imu and imu.get('noise') == 'replay' and a.imu_noise_seed != 0:
        sys.exit('--imu-noise-seed: S02 sets the noise seed to the repetition number (rep_imu)')
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
            params = fwsim.plant_params(a, wheels_lifted=test_id in LIFTED_TESTS)
            sp = (lambda s: lambda t: get_body_velocity(t, s) if t < s.duration else (0.0, 0.0, 0.0))(spec)
            imu_r = rep_imu(imu, rep)
            log = fwsim.simulate(fw, sp, spec.duration + STOP_TAIL_S, params, seed=rep, mode=a.controller, imu=imu_r)
            run_id = f'{test_id}_{a.controller}_r{rep}'
            path = os.path.join('raw', f'{run_id}.csv')
            with open(os.path.join(a.out, path), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(COLUMNS)
                w.writerows(log_rows(log, imu is not None))
            runs.append(dict(run_id=run_id, platform='sim', controller=a.controller, test_id=test_id,
                             speed_mps=spec.v, rep=rep, seed=rep,
                             noise_seed=(imu_r or {}).get('noise_seed', ''), v_batt_start='', v_batt_end='',
                             file=path, aborted=0, reason='ok', missing_pct=0))
    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    s02_metrics.run_dir_metrics(a.out)

    b = fw.build_info
    p0 = fwsim.plant_params(a)
    gains = ({ax: fw.get_adrc(i) for i, ax in enumerate(['vx', 'vy', 'wz'])} if a.controller == 'ADRC'
             else {ax: fw.get_pid(i) for i, ax in enumerate(['vx', 'vy', 'wz'])})
    pos_loop = fwsim.pos_meta(fw)
    NOTE_NOISE = ('the plant has no random input (legacy slip model off; ground_model traction is deterministic); '
                  'the IMU noise replay (rpi5/imu_model.ImuNoise) starts at the row given by the repetition seed, '
                  'so repetitions differ by the noise segment; a re-run with the same seed is the same run bit for bit '
                  '(S02 v1.3 A-R2-2)')
    NOTE_NO_NOISE = 'no random input (plant deterministic, IMU noise off), so the repetitions give identical runs'
    fw.reset()
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: S02
part: groupA
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
controller: {a.controller}
gains: {json.dumps(gains)}
pos_loop: {json.dumps(pos_loop)}     # pos_ctrl.c + standstill hold read back (on by default since 04/11); null = DLL without pos_ctrl.c
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
  build_flags: "fwsim shim (no FreeRTOS); main.c glue repeated in fwsim/fw_sim.c"
  h7: null
  rpi5: null
plant: {fwsim.plant_meta(p0)}
plant_extra: {{motor_polarity: {p0.motor_polarity.tolist()}, lifted_tests: {list(LIFTED_TESTS)}}}
imu: {json.dumps(imu)}             # null = no IMU reports (--no-imu, only for pre-28/10 re-runs); noise_seed is set per repetition (seeds)
reps: {a.reps}
seeds: "plant seed = rep (1..{a.reps}); IMU noise replay seed = rep (rep_imu, runs.csv noise_seed); the HIL run of the same repetition uses the same seed"
order_seed: {a.order_seed}
setpoint: "trajectory_pub.get_body_velocity held at 50 Hz, then 0 for {STOP_TAIL_S} s"
libraries: {json.dumps(fwsim.lib_versions())}
elapsed_s: {time.time() - t_start:.0f}
files:
  - raw/<run_id>.csv: "ESP32 logger format (tools/agv_logger.py COLUMNS, 85), 100 Hz; IMU fusion / heading columns from the firmware (empty with --no-imu); pos_* and ws_* from the firmware (empty with a DLL without pos_ctrl.c / wheel_slip.c); raw chip columns ax..roll and gyro_n, rv_n, reset_n empty (not in fwsim)"
  - runs.csv: "spec P2-6 columns + test_id"
  - metrics.csv: "tools/s02_metrics.py, one line per run"
notes: "{NOTE_NOISE if imu and imu.get('noise') == 'replay' else NOTE_NO_NOISE}"
""")
    print(f'{len(runs)} runs in {time.time() - t_start:.0f} s -> {a.out}')


if __name__ == '__main__':
    main()
