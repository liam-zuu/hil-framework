"""tools/fault_c7_table.py (L82, design 8.3, spec S06 T8 type II): edges of the firmware flags, the numbers
of one (controller, case) row against its baseline on synthetic logs whose answers are known, and the folders
that cannot be tabled (no baseline, a baseline with rows, a different seed / firmware / tick times); bench
folders (hil_bench_session.py: Pi frame log + ESP32 CSV) on a synthetic session and on FAULT_C7 run02, run03.
Run: python -m pytest tools/tests"""
import csv
import functools
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools')]
import fault_c7_table as ct                    # noqa: E402

N = 900                                        # 9 s of 10 ms ticks
T = np.round(np.arange(1, N + 1) * 0.01, 6)
COLS = (['t'] + [f'sp_{a}' for a in ('vx', 'vy', 'wz')] + [f'v_{a}' for a in ('vx', 'vy', 'wz')] +
        [f'pwm_{w}' for w in ('fl', 'fr', 'rl', 'rr')] + ['x', 'y', 'theta', 'wz_src', 'age_g_ms', 'r', 'r_th',
                                                          'slip_flag', 'r_true', 'ws_flag'])
CONFIG = dict(task='A-W1', mode='PID', seed=0, imu_rate=20.0, imu_boot_reset=-1.0, imu_noise='replay',
              imu_noise_seed=0, ground='traction', mu=None, no_imu=False, duration=9.0)
ROW = dict(id='E1', fault='enc_loss', target='FL', start_ms=3000, end_ms=5000, params={})


def base_log():
    lg = {c: np.zeros(N) for c in COLS}
    lg['t'] = T.copy()
    lg['sp_wz'] = np.where(T < 4.0, 1.0, -1.0)
    lg['v_wz'] = lg['sp_wz'] + 0.01
    lg['wz_src'][:] = 1
    lg['age_g_ms'][:] = 40
    lg['r_th'][:] = 0.0784
    lg['theta'][:] = 3.1
    return lg


PLANT = dict(motor_model='dc', V_batt=12.4, ground_model='traction', mu_hd=[0.436] * 4,
             src=dict(plant_step_py='e56a', params_py='501e'))


def write_case(folder, lg, rows=(), config=None, esp32='5bde4c5', hil='a2de855', imu_rows=180, dropped=(),
               plant=None, dirty=False):
    os.makedirs(folder)
    np.savetxt(os.path.join(folder, 'log.csv'), np.column_stack([lg[c] for c in COLS]), delimiter=',',
               header=','.join(COLS), comments='', fmt='%.6g')
    rows = [dict(r) for r in rows]
    with open(os.path.join(folder, 'faults.json'), 'w') as fh:
        json.dump(dict(rows=rows, complete=True, events=[], not_run=[], cut=False), fh)
    with open(os.path.join(folder, 'faults_imu.csv'), 'w') as fh:
        fh.write('t,mask\n' + ''.join(f'{k * 0.05},{16 if k in dropped else 0}\n' for k in range(imu_rows)))
    with open(os.path.join(folder, 'meta.yaml'), 'w', encoding='utf-8') as fh:
        fh.write('spec: null\nfirmware:\n'
                 f'  esp32: {json.dumps(dict(commit=esp32, dirty=dirty))}\n'
                 f'  hil_framework: {json.dumps(dict(commit=hil, dirty=False))}\n'
                 f'config: {json.dumps(config or CONFIG)}\n'
                 f'plant: {json.dumps(plant or PLANT)}\n'
                 f'faults: {json.dumps(dict(rows=rows))}\n')


def fault_log():
    lg = base_log()
    w = (T >= 3.0) & (T < 5.0)
    lg['wz_src'][(T >= 3.08) & (T < 5.10)] = 0           # encoder wz from 3.08 s, IMU again at 5.10 s
    lg['slip_flag'][(T >= 3.25) & (T < 5.5)] = 1
    lg['ws_flag'][(T >= 3.22) & (T < 5.3)] = 1
    lg['age_g_ms'][w] = 2040
    lg['r'][T == 4.5] = -0.9
    lg['v_vx'][T == 5.9] = 0.4                          # inside the window + 1 s
    lg['v_vx'][T == 6.0] = 9.0                          # just after it: not counted
    lg['pwm_fl'][(T >= 3.0) & (T < 3.5)] = -1000         # 50 of 300 ticks, one wheel of four
    lg['x'][:] = 0.25
    lg['theta'][:] = -3.1                               # vs 3.1: wrapped difference 0.0832
    lg['r_true'][T == 1.0] = -1.2
    return lg


def test_edges():
    t = np.round(np.arange(10) * 1.0, 6)
    assert ct.edges(t, np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1]), 2.0) == (3.0, 5.0)
    assert ct.edges(t, np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]), 2.0) == (2.0, 5.0)     # on at start
    assert ct.edges(t, np.zeros(10), 2.0) == (None, None)
    assert ct.edges(t, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]), 2.0) == (7.0, None)
    assert ct.edges(t, np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 2.0) == (None, None)   # fell before start
    assert ct.edges(t, np.ones(10), 20.0) == (None, None)                              # start after the run


def test_row_against_the_baseline(tmp_path):
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_case(str(tmp_path / 'PID_encFL'), fault_log(), rows=[ROW], dropped=range(60, 100))
    rows = ct.table(str(tmp_path))
    assert len(rows) == 1
    r = rows[0]
    assert (r['controller'], r['case'], r['win_start_s'], r['win_end_s'], r['complete']) == ('PID', 'encFL', 3.0,
                                                                                         5.0, True)
    assert r['t_src_enc'] == pytest.approx(3.08) and r['t_src_imu'] == pytest.approx(5.10)
    assert r['t_slip_on'] == pytest.approx(3.25) and r['t_slip_off'] == pytest.approx(5.5)
    assert r['t_ws_on'] == pytest.approx(3.22) and r['t_ws_off'] == pytest.approx(5.3)
    assert r['base_t_src_enc'] is None and r['base_t_slip_on'] is None and r['base_t_ws_on'] is None
    assert r['age_g_max_ms'] == 2040 and r['r_max'] == pytest.approx(0.9) and r['r_th'] == pytest.approx(0.0784)
    assert r['err_max_vx'] == pytest.approx(0.4) and r['base_err_max_vx'] == 0
    assert r['err_max_wz'] == pytest.approx(0.01, abs=1e-6)
    assert r['err_rms_vx'] == pytest.approx(0.4 / math.sqrt(300)) and r['base_err_rms_vx'] == 0
    assert r['err_rms_wz'] == pytest.approx(0.01, abs=1e-6) == r['base_err_rms_wz']
    assert r['sat_frac'] == pytest.approx(50 / (300 * 4)) and r['base_sat_frac'] == 0
    assert r['end_dx'] == pytest.approx(0.25) and r['end_dtheta'] == pytest.approx(2 * math.pi - 6.2, abs=1e-6)
    assert r['r_true_max'] == pytest.approx(1.2)
    assert (r['ticks'], r['ticks_expected'], r['imu_rows'], r['imu_dropped'], r['base_imu_rows']) == (900, 900, 180,
                                                                                                  40, 180)


@pytest.mark.parametrize('kind,msg', [
    ('no_base', 'no PID_base'),
    ('base_rows', 'baseline has schedule rows'),
    ('seed', 'config seed'),
    ('noise', 'config imu_noise'),
    ('esp32', 'esp32 commit'),
    ('ticks', 'tick times differ'),
    ('name', 'not <MODE>_<CASE>'),
    ('plant', 'plant'),                        # C7T1-4: same config, another mu_hd
    ('dirty', 'dirty or unknown'),             # C7T1-4: a dirty DLL build under the same commit
    ('dirty_none', 'dirty or unknown'),
])
def test_folders_that_cannot_be_tabled(tmp_path, kind, msg):
    if kind != 'no_base':
        lb = base_log()
        if kind == 'ticks':
            lb['t'] = lb['t'] + 0.005
        write_case(str(tmp_path / ('PIDbase' if kind == 'name' else 'PID_base')), lb,
                   rows=[ROW] if kind == 'base_rows' else ())
    cfg = dict(CONFIG, seed=1) if kind == 'seed' else dict(CONFIG, imu_noise='off') if kind == 'noise' else None
    write_case(str(tmp_path / 'PID_encFL'), fault_log(), rows=[ROW], config=cfg,
               esp32='16702ff' if kind == 'esp32' else '5bde4c5',
               plant=dict(PLANT, mu_hd=[0.3] * 4) if kind == 'plant' else None,
               dirty=True if kind == 'dirty' else None if kind == 'dirty_none' else False)
    with pytest.raises(ct.TableError, match=msg):
        ct.table(str(tmp_path))


# ── bench folders (hil_bench_session.py sessions) ──

sys.path[:0] = [os.path.join(ROOT, 'rpi5')]
import protocol as P                           # noqa: E402
import fault_tasks                             # noqa: E402
import params as PARAMS                        # noqa: E402

T0 = 1000                                      # first RUN step
NF = T0 + 9500                                 # frames: past t0 + 9 s
TH = 0.5                                       # heading: body != world velocity
RUN_ID = 0x1234ABCD
ARGS = '--test A-W1 --seed 0 --imu-noise 20 --ground traction'
FW = dict(esp32_fw=[dict(sha='5bde4c55bf3a', dirty=0, env='esp32-s3-devkitc-1')],
          h7_fw=[dict(sha='7dd5ab265933', dirty=0, env='Debug')])
GAIN = '#gain,id=0,applied_n=0,ctrl=PID,kp_xy=1.5,ki_xy=0.05,kp_wz=1,ki_wz=0.05,first_seq={},line_drop=0,boot=ab'
ESP_PRE, ESP_RUN, ESP_POST = 5, 920, 30        # rows: idle before, run_active, idle after (tick frozen)
SEQ0 = 1000                                    # log_seq of the first ESP32 row
FIRST_SEQ = SEQ0 + ESP_PRE                     # log_seq of the first run_active row (= #gain first_seq)
H_FWD = PARAMS.MecanumParams().H_fwd


@functools.lru_cache(maxsize=None)
def bench_frames(fault, endpoint=False):
    """(mosi, mosi_true) of a run: 9.5 s after t0, wheels 2 rad/s each, body wz = A-W1 + 0.01, heading TH;
    the fault run: body vx 0.4 at t0 + 5.9 s and 9.0 at 6.0 s, x 0.25 at t0 + 9 s and 1.0 after, theta -3.1
    at t0 + 9 s (base 3.1); mosi of the fault run has wheel FL = 0 in [3, 5) s (enc_loss). endpoint: wheels
    -10 / 10 / -10 / 10 at t0 + 9 s (r_true 2.01, inside (0, duration]) and -20 / 20 at t0 (outside)."""
    true, sent = [], []
    for step in range(NF):
        k = step - T0
        wz_ref = fault_tasks.a_w1(k / 1000.0)[2] if 0 <= k < 8000 else 0.0
        vb = 0.4 if fault and k == 5900 else 9.0 if fault and k == 6000 else 0.0
        x = (0.25 if k == 9000 else 1.0 if k > 9000 else 0.0) if fault else 0.0
        th = (-3.1 if fault else 3.1) if k == 9000 else TH
        w = 10.0 if endpoint and k == 9000 else 20.0 if endpoint and k == 0 else None
        om = [-w, w, -w, w] if w else [2.0, 2.0, 2.0, 2.0]
        st = om + [x, 0.0, th, math.cos(th) * vb, math.sin(th) * vb, wz_ref + 0.01]
        kw = dict(traj_ref=(0.0, 0.0, wz_ref), seq=step, frame_no=step, run_id=RUN_ID)
        true.append(np.frombuffer(P.encode_state(st, **kw), np.uint8))
        if fault and 3000 <= k < 5000:
            st = [0.0] + st[1:]
        sent.append(np.frombuffer(P.encode_state(st, **kw), np.uint8))
    return np.array(sent), np.array(true)


def esp_rows(fault):
    """ESP32 CSV rows (ESP_COLS); tick 100 before the run, 101.. while run_active, frozen after; log_seq one per
    row from SEQ0; timestamp_us with +-2 ms of jitter; wz_src 0 from 9.2 s in both runs (the motor-off switch
    after the run)."""
    out, gyro = [], 0
    for i in range(ESP_PRE + ESP_RUN + ESP_POST):
        j = i - ESP_PRE                                    # run tick index, t = j * 10 ms
        ra = 1 if 0 <= j < ESP_RUN else 0
        tick = 100 + min(max(j + 1, 0), ESP_RUN)
        t = round(j * 0.01, 6)
        if ra and j % 5 == 0 and not (fault and 3.0 <= t < 5.0):
            gyro += 1                                      # 20 Hz reports, none in the imu window of the fault run
        w = 1.0 if 3.0 <= t < 5.0 else 0.0
        out.append(dict(timestamp_us=1_000_000 + 10_000 * i + (2000 if i % 2 else -2000), tick=tick,
                        log_seq=SEQ0 + i, run_active=ra,
                        wz_src=0 if (fault and 3.08 <= t < 5.10) or t >= 9.2 else 1,
                        slip_flag=1 if fault and 3.25 <= t < 5.5 else 0,
                        ws_flag=1 if fault and 3.22 <= t < 5.3 else 0,
                        age_g_ms=2040 if fault and w else 40, r=-0.9 if fault and abs(t - 4.5) < 1e-9 else 0.0,
                        gyro_n=gyro, pwm_fl=-1000 if fault and 3.0 <= t < 3.5 else 100, pwm_fr=0, pwm_rl=0, pwm_rr=0,
                        sp_vx=0.0, sp_vy=0.0, sp_wz=fault_tasks.a_w1(t)[2] if ra else 0.0))
    return out


def manifest(fault, t0=T0, **over):
    m = dict(step_first=0, step_last=NF - 1, steps=NF, period_us=1000, imu_source='plant',
             imu_noise=dict(model='replay', seed=0, start_row=5114), plant='MecanumPlant',
             plant_src=dict(plant_step_py='e56a'), proto='mosi_v4_miso_v6', frame_len=48,
             lifecycle=dict(transitions=[dict(step=0, phase='ARM'), dict(step=T0, phase='RUN'),
                                         dict(step=T0 + 8090, phase='STOPPING'), dict(step=T0 + 9090, phase='END')]),
             faults=dict(file='enc_FL.csv', sha256='abc', rows=[ROW], t0_step=t0, events=[], not_run=[], cut=False,
                         complete=True) if fault else None)
    m.update(over)
    return m


ESP_BREAK = {   # kind -> (row index of the fault run's ESP32 log, column, value); C7T1-2, C7T1-6
    'nan_r': (ESP_PRE + 400, 'r', 'nan'),
    'flag_bad': (ESP_PRE + 400, 'slip_flag', 2),
    'run_gap': (ESP_PRE + 400, 'run_active', 0),
    'seq_gap': (ESP_PRE + 400, 'log_seq', SEQ0 + ESP_PRE + 401),
    'half_tick': (ESP_PRE + 400, 'gyro_n', 50.5),
    'gyro_back': (ESP_PRE + 899, 'gyro_n', 0),                  # C7T2-2: the last counter of the span back to 0
}


def write_bench(d, kind=None):
    """A bench folder with PID_base and PID_enc_FL; kind breaks one thing."""
    os.makedirs(os.path.join(d, 'frames'))
    os.makedirs(os.path.join(d, 'pi'))
    os.makedirs(os.path.join(d, 'steps'))
    runs, summ = [], []
    for name, fault in (('PID_base', False), ('PID_enc_FL', True)):
        args = ARGS + (' --faults ~/fault_c7/enc_FL.csv' if fault else '') + f' --step-log ~/{name}_steps.csv'
        if fault and kind == 'args':
            args = args.replace('--seed 0', '--seed 1')
        if fault and kind == 'args_bad':
            args += ' --ground mud'
        if kind == 'args_tilde':                               # C7T5-1: ~ expanded in one run only
            args += ' --uart-port=~/uart' if fault else ' --uart-port ~/uart'
        if kind == 'args_shell':                               # C7T4-1: bash expands the braces of one run only
            args += " --uart-port {unused,--ground,rigid}" if fault else " --uart-port '{unused,--ground,rigid}'"
        if kind == 'args_repeat':                              # C7T2-1: argparse keeps the last --ground
            args = args.replace('--ground traction', '--ground traction --ground rigid' if fault else
                                '--ground rigid --ground traction')
        if kind == 'args_abbrev':                              # C7T3-1: --grou is --ground for argparse
            args = args.replace('--ground traction', '--grou traction --ground rigid' if fault else
                                '--ground rigid --grou traction')
        if fault and kind == 'args_same_abbrev':               # same effective options, abbreviated
            args = args.replace('--ground traction', '--grou=traction').replace('--imu-noise', '--imu-n')
        if fault and kind == 'args_order':                     # same pairs, another order and --opt=value
            args = '--seed 0 --ground=traction --imu-noise 20 --test=A-W1 --faults ~/x.csv --step-log=~/s.csv'
        fw = dict(FW)
        if fault and kind == 'esp32':
            fw['esp32_fw'] = [dict(sha='16702ff00000', dirty=0, env='esp32-s3-devkitc-1')]
        runs.append(dict(name=name, args=args, **fw))
        summ.append(dict(name=name, clean=not (fault and kind == 'clean')))
        mosi, true = bench_frames(fault, endpoint=fault and kind == 'endpoint')
        mosi, true = mosi.copy(), true.copy()
        over = {}
        if fault and kind == 'noise':
            over['imu_noise'] = dict(model='replay', seed=1, start_row=5114)
        if not fault and kind == 'period':
            over['period_us'] = 2000
        man = manifest(fault or kind == 'base_rows', t0=T0 + 1 if fault and kind == 't0' else T0, **over)
        n = NF - 600 if fault and kind == 'short' else NF
        man['steps'], man['step_last'] = n, n - 1
        if not fault and kind == 'frame_no':                   # C7T1-5: a lost baseline frame, steps renumbered
            mosi = np.delete(mosi, T0 + 4500, axis=0)
            n -= 1
            man['steps'], man['step_last'] = n, n - 1
        if not fault and kind == 'run_id0':
            mosi[:, P.OFS_RUN_ID:P.OFS_RUN_ID + 4] = 0
        arrs = dict(manifest=np.array(json.dumps(man)), mosi=mosi[:n])
        if fault or kind == 'base_rows':
            arrs['mosi_true'] = true[:n]
        if not fault and kind == 'dtype':
            arrs['mosi'] = mosi[:n].astype(np.uint16)
        np.savez(os.path.join(d, 'frames', f'{name}_frames.npz'), **arrs)
        rows = esp_rows(fault)
        cols = list(ct.ESP_COLS)
        if fault and kind == 'col':
            cols.remove('ws_flag')
        if fault and kind == 'no_run':
            for r in rows:
                r['run_active'] = 0
        if fault and kind in ESP_BREAK:
            i, c, v = ESP_BREAK[kind]
            rows[i][c] = v
        if fault and kind == 'lost_head':                      # C7T1-2: the first 3 run rows lost
            del rows[ESP_PRE:ESP_PRE + 3]
        if fault and kind == 'no_pre':
            del rows[:ESP_PRE]
        with open(os.path.join(d, 'pi', f'{name}.csv'), 'w', newline='') as fh:
            fh.write(','.join(cols) + '\n' + ''.join(','.join(str(r[c]) for c in cols) + '\n' for r in rows))
        gain = GAIN.format(FIRST_SEQ + (1 if fault and kind == 'first_seq' else 0))
        if fault and kind == 'gain':
            gain = gain.replace('kp_xy=1.5', 'kp_xy=2')
        with open(os.path.join(d, 'pi', f'{name}.csv.events.txt'), 'w') as fh:
            fh.write(''.join(f'2026-11-14T22:53:{s} {gain}\n' for s in ('01', '02', '03')))
        if fault:
            sched = os.path.join(d, 'enc_FL.csv')
            with open(sched, 'w') as fh:
                fh.write('# hil-fault-schedule v1\n')
            if kind != 'no_check':
                with open(os.path.join(d, f'fault_check_{name}.json'), 'w') as fh:
                    json.dump(dict(ok=True, sha256='abc', schedule=None if kind == 'no_sched' else sched,
                                   frames=f'05_DATA/x/frames/{name}_frames.npz'), fh)
            if kind != 'no_steps':
                open(os.path.join(d, 'steps', f'{name}_steps.csv'), 'w').close()
    with open(os.path.join(d, 'session.json'), 'w') as fh:
        json.dump(dict(runs=runs), fh)
    with open(os.path.join(d, 'bench_summary.json'), 'w') as fh:
        json.dump(summ, fh)


class _Calls(list):
    pass


@pytest.fixture
def fake_check(monkeypatch):
    """Replaces tools/fault_check.check (it needs MISO, step log, schedule of a real run): records the calls,
    answers with calls.answer."""
    calls = _Calls()
    calls.answer = dict(ok=True, sha256='abc')

    def check(npz, sched, step_log):
        calls.append((npz, sched, step_log))
        return dict(calls.answer)
    monkeypatch.setattr(ct, 'run_fault_check', check)
    return calls


def test_bench_row(tmp_path, fake_check):
    write_bench(str(tmp_path))
    rows = ct.table(str(tmp_path))
    assert len(rows) == 1
    r = rows[0]
    assert (r['controller'], r['case'], r['win_start_s'], r['win_end_s'], r['complete']) == ('PID', 'enc_FL', 3.0,
                                                                                         5.0, True)
    # C7T1-3: fault_check run again on this folder's frame log, the schedule its report names, its step log
    assert fake_check == [(os.path.join(str(tmp_path), 'frames', 'PID_enc_FL_frames.npz'),
                           os.path.join(str(tmp_path), 'enc_FL.csv'),
                           os.path.join(str(tmp_path), 'steps', 'PID_enc_FL_steps.csv'))]
    # ESP32 clock = tick count from the first run_active row (the jittered timestamp_us would miss these)
    assert r['t_src_enc'] == pytest.approx(3.08) and r['t_src_imu'] == pytest.approx(5.10)
    assert r['t_slip_on'] == pytest.approx(3.25) and r['t_slip_off'] == pytest.approx(5.5)
    assert r['t_ws_on'] == pytest.approx(3.22) and r['t_ws_off'] == pytest.approx(5.3)
    # rows from t = duration on are not read: the motor-off switch of wz_src at 9.2 s is not an edge
    assert r['base_t_src_enc'] is None and r['base_t_slip_on'] is None and r['base_t_ws_on'] is None
    assert r['age_g_max_ms'] == 2040 and r['r_max'] == pytest.approx(0.9) and r['r_th'] is None
    q = 3.0 / 32767                                         # int16 step of a velocity in the frame
    assert r['err_max_vx'] == pytest.approx(0.4, abs=2 * q) and r['base_err_max_vx'] == pytest.approx(0, abs=q)
    assert r['err_rms_vx'] == pytest.approx(0.4 / math.sqrt(3000), rel=1e-3)
    assert r['err_max_wz'] == pytest.approx(0.01, abs=1e-3) and r['err_rms_wz'] == pytest.approx(0.01, abs=1e-3)
    assert r['sat_frac'] == pytest.approx(50 / (300 * 4)) and r['base_sat_frac'] == 0
    assert r['end_dx'] == pytest.approx(0.25, abs=1e-3)
    assert r['end_dtheta'] == pytest.approx(2 * math.pi - 6.2, abs=1e-3)
    # r_true from the true wheels (mosi_true), not the sent ones (FL = 0 in the window): |0 - wz| at most 1.01
    assert r['r_true_max'] == pytest.approx(1.01, abs=2e-3)
    assert (r['ticks'], r['ticks_expected'], r['tick_gaps']) == (900, 900, 0)
    assert (r['imu_rows'], r['imu_dropped'], r['base_imu_rows']) == (None, None, None)
    # C7T1-7: gyro_n at the last row of the span minus at the first (counted after the first tick)
    assert (r['esp_gyro_n'], r['base_esp_gyro_n']) == (139, 179)
    assert r['run_s'] == pytest.approx(8.09)
    assert r['ref_first_s'] == pytest.approx(2.0) and r['esp_sp_first_s'] == pytest.approx(2.0)
    assert r['blank'] == 'r_th;imu_rows;imu_dropped;base_imu_rows'


def test_bench_r_true_over_0_duration(tmp_path, fake_check):
    """C7T1-8: r_true over (0, duration] as fwsim's ticks 0.01 .. 9.00 s: the frame at t0 + 9 s counts, t0 not."""
    write_bench(str(tmp_path), 'endpoint')
    r = ct.table(str(tmp_path))[0]
    assert r['r_true_max'] == pytest.approx(4 * 0.0505208 * 10 - 0.01, abs=5e-3)


def test_bench_args_in_another_order(tmp_path, fake_check):
    """C7T1-10: the same (option, value) pairs in another order, --opt=value, are the same run."""
    write_bench(str(tmp_path), 'args_order')
    assert len(ct.table(str(tmp_path))) == 1


def test_bench_columns_extend_the_fwsim_ones(tmp_path, fake_check):
    write_case(str(tmp_path / 'fw' / 'PID_base'), base_log())
    write_case(str(tmp_path / 'fw' / 'PID_encFL'), fault_log(), rows=[ROW])
    write_bench(str(tmp_path / 'hil'))
    kf, kb = list(ct.table(str(tmp_path / 'fw'))[0]), list(ct.table(str(tmp_path / 'hil'))[0])
    assert kb[:len(kf)] == kf
    assert kb[len(kf):] == ['run_s', 'tick_gaps', 'esp_gyro_n', 'base_esp_gyro_n', 'ref_first_s', 'esp_sp_first_s',
                            'esp_dup_dropped', 'base_esp_dup_dropped', 'blank']


def insert_esp_copy(log, source, dest, r=None):
    """Copy one 1-based data line to another, with a different line ending."""
    lines = log.read_bytes().splitlines(keepends=True)
    copy = lines[source].rstrip(b'\r\n')
    if r is not None:
        fields = copy.split(b',')
        fields[lines[0].rstrip().split(b',').index(b'r')] = r.encode('ascii')
        copy = b','.join(fields)
    lines.insert(dest, copy + b'\r\n')
    log.write_bytes(b''.join(lines))


def write_esp_segments(log, duplicates=()):
    seg = log.with_suffix('.csv.segments.csv')
    with log.open(newline='') as fh, seg.open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['row', 'log_seq', 'timestamp_us', 'segment', 'status'])
        for i, row in enumerate(csv.DictReader(fh), start=1):
            writer.writerow([i, row['log_seq'], row['timestamp_us'], 0, 'duplicate' if i in duplicates else 'ok'])
    return seg


def write_esp_summary(log, counts):
    seg = log.with_suffix('.csv.segments.csv')
    summary = dict(files=dict(segments=dict(file=seg.name, sha256=hashlib.sha256(seg.read_bytes()).hexdigest())),
                   segments=[dict(segment=i, duplicates=count) for i, count in enumerate(counts)])
    log.with_suffix('.csv.summary.json').write_text(json.dumps(summary), encoding='utf-8')


@pytest.mark.parametrize('source,dest', [
    pytest.param(ESP_PRE + 400, ESP_PRE + 401, id='adjacent'),
    pytest.param(ESP_PRE + 400, ESP_PRE + 404, id='late'),
    pytest.param(ESP_PRE - 2, ESP_PRE + 1, id='before-first-seq'),
])
def test_bench_drops_marked_esp_copies(tmp_path, fake_check, capsys, source, dest):
    write_bench(str(tmp_path))
    log = tmp_path / 'pi' / 'PID_enc_FL.csv'
    out = tmp_path / 'c7_table.csv'
    write_esp_segments(log)
    write_esp_summary(log, [0])
    assert ct.main([str(tmp_path)]) == 0
    assert 'duplicate ESP32' not in capsys.readouterr().out
    with out.open(newline='') as fh:
        before, = csv.DictReader(fh)
    assert (before['esp_dup_dropped'], before['base_esp_dup_dropped']) == ('0', '0')

    insert_esp_copy(log, source, dest)
    write_esp_segments(log, [dest])
    write_esp_summary(log, [1])
    assert ct.main([str(tmp_path)]) == 0
    with out.open(newline='') as fh:
        after, = csv.DictReader(fh)
    assert after == dict(before, esp_dup_dropped='1')
    stdout = capsys.readouterr().out.splitlines()
    assert [line for line in stdout if 'duplicate ESP32' in line] == [
        f'fault_c7_table: PID_enc_FL: dropped 1 duplicate ESP32 line(s) (log_seq {SEQ0 + source - 1})'
    ]


def test_bench_counts_esp_copies_in_baseline_after_the_run(tmp_path, fake_check, capsys):
    write_bench(str(tmp_path))
    before, = ct.table(str(tmp_path))
    assert (before['esp_dup_dropped'], before['base_esp_dup_dropped']) == (0, 0)
    log = tmp_path / 'pi' / 'PID_base.csv'
    source = ESP_PRE + ESP_RUN + 1
    insert_esp_copy(log, source, source + 1)
    insert_esp_copy(log, source, source + 2)
    write_esp_segments(log, [source + 1, source + 2])
    write_esp_summary(log, [2])

    after, = ct.table(str(tmp_path))
    assert after == dict(before, base_esp_dup_dropped=2)
    assert capsys.readouterr().out == ''
    assert ct.main([str(tmp_path)]) == 0
    stdout = capsys.readouterr().out.splitlines()
    seq = SEQ0 + source - 1
    assert [line for line in stdout if 'duplicate ESP32' in line] == [
        f'fault_c7_table: PID_base: dropped 2 duplicate ESP32 line(s) (log_seq {seq}, {seq})'
    ]


@pytest.mark.parametrize('status,r,msg', [
    pytest.param('duplicate', '9.9', 'no earlier identical kept line', id='changed-marked'),
    pytest.param('duplicate', '"0.0"', 'no earlier identical kept line', id='same-field-different-text'),
    pytest.param('ok', '9.9', 'log_seq gap', id='changed-unmarked'),
    pytest.param('ok', None, 'log_seq gap', id='identical-unmarked'),
    pytest.param(None, None, 'log_seq gap', id='no-segments'),
])
def test_bench_refuses_esp_copies_without_both_conditions(tmp_path, fake_check, status, r, msg):
    write_bench(str(tmp_path))
    log = tmp_path / 'pi' / 'PID_enc_FL.csv'
    source = ESP_PRE + 400
    insert_esp_copy(log, source, source + 1, r=r)
    if status is not None:
        duplicates = [source + 1] if status == 'duplicate' else []
        write_esp_segments(log, duplicates)
        write_esp_summary(log, [len(duplicates)])
    with pytest.raises(ct.TableError, match=msg):
        ct.table(str(tmp_path))


def test_bench_refuses_esp_copy_marked_on_the_first_occurrence(tmp_path, fake_check):
    write_bench(str(tmp_path))
    log = tmp_path / 'pi' / 'PID_enc_FL.csv'
    source = ESP_PRE + 400
    insert_esp_copy(log, source, source + 1)
    write_esp_segments(log, [source])
    write_esp_summary(log, [1])
    with pytest.raises(ct.TableError, match='no earlier identical kept line'):
        ct.table(str(tmp_path))


@pytest.mark.parametrize('kind', ['count', 'row', 'log_seq', 'timestamp_us'])
def test_bench_refuses_esp_segments_that_do_not_match(tmp_path, fake_check, kind):
    write_bench(str(tmp_path))
    log = tmp_path / 'pi' / 'PID_enc_FL.csv'
    seg = write_esp_segments(log)
    with seg.open(newline='') as fh:
        marks = list(csv.reader(fh))
    if kind == 'count':
        marks.pop()
    else:
        col = marks[0].index(kind)
        marks[ESP_PRE + 400][col] = str(int(marks[ESP_PRE + 400][col]) + 1)
    with seg.open('w', newline='') as fh:
        csv.writer(fh).writerows(marks)
    write_esp_summary(log, [0])
    with pytest.raises(ct.TableError, match='rows for' if kind == 'count' else 'does not match'):
        ct.table(str(tmp_path))


@pytest.mark.parametrize('kind,duplicate,msg', [
    ('wrong_sha', True, 'SHA-256 differs'),
    ('wrong_sha', False, 'SHA-256 differs'),
    ('missing_sha', True, 'must name its SHA-256'),
    ('missing_summary', True, 'must name its SHA-256'),
    ('wrong_count', True, 'counts 2 duplicates'),
    ('missing_count', True, 'counts None duplicates'),
])
def test_bench_refuses_inconsistent_esp_summary(tmp_path, fake_check, kind, duplicate, msg):
    write_bench(str(tmp_path))
    log = tmp_path / 'pi' / 'PID_enc_FL.csv'
    dest = ESP_PRE + 401
    if duplicate:
        insert_esp_copy(log, dest - 1, dest)
    write_esp_segments(log, [dest] if duplicate else [])
    if kind != 'missing_summary':
        write_esp_summary(log, [1, 1] if kind == 'wrong_count' else [int(duplicate)])
        path = log.with_suffix('.csv.summary.json')
        summary = json.loads(path.read_text(encoding='utf-8'))
        if kind == 'wrong_sha':
            summary['files']['segments']['sha256'] = '0' * 64
        elif kind == 'missing_sha':
            del summary['files']['segments']['sha256']
        elif kind == 'missing_count':
            del summary['segments'][0]['duplicates']
        path.write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ct.TableError, match=msg):
        ct.table(str(tmp_path))


@pytest.mark.parametrize('col', ['log_seq', 'tick', 'run_active'])
def test_bench_esp_segments_do_not_hide_run_gaps(tmp_path, fake_check, col):
    write_bench(str(tmp_path))
    log = tmp_path / 'pi' / 'PID_enc_FL.csv'
    with log.open(newline='') as fh:
        head, *rows = csv.reader(fh)
    i = head.index(col)
    if col == 'run_active':
        rows[ESP_PRE + 400][i] = '0'
    else:
        for row in rows[ESP_PRE + 400:]:
            row[i] = str(int(row[i]) + 1)
    with log.open('w', newline='') as fh:
        csv.writer(fh).writerows([head, *rows])
    write_esp_segments(log)
    write_esp_summary(log, [0])
    with pytest.raises(ct.TableError, match='tick / log_seq gap, or run_active 0 inside'):
        ct.table(str(tmp_path))


@pytest.mark.parametrize('kind,msg', [
    ('clean', 'not clean'),
    ('no_check', 'no fault_check_PID_enc_FL.json'),
    ('no_sched', 'names no schedule'),
    ('no_steps', 'not found'),
    ('check_bad', 'not ok again'),
    ('check_sha', 'not ok again'),
    ('args', 'run args'),
    ('esp32', 'esp32_fw'),
    ('noise', 'manifest imu_noise'),
    ('gain', '#gain'),
    ('t0', 't0_step'),
    ('short', 'no frame at t0'),
    ('base_rows', 'baseline has schedule rows'),
    ('col', 'no column'),
    ('no_run', 'run_active'),
    ('lost_head', 'start of the run not shown'),
    ('no_pre', 'no ESP32 row before'),
    ('first_seq', 'start of the run not shown'),
    ('nan_r', 'r not finite'),
    ('flag_bad', 'slip_flag not 0 / 1'),
    ('run_gap', 'run_active 0 inside'),
    ('seq_gap', 'log_seq gap'),
    ('half_tick', 'gyro_n not a whole number'),
    ('gyro_back', 'gyro_n decreases'),
    ('args_repeat', 'run args'),
    ('args_abbrev', 'run args'),
    ('args_bad', 'run_hil.py refuses'),
    ('args_shell', 'shell characters'),
    ('frame_no', 'frame_no is not step_first'),
    ('run_id0', 'run_id 0'),
    ('dtype', 'uint8'),
    ('period', 'period_us'),
])
def test_bench_folders_that_cannot_be_tabled(tmp_path, fake_check, kind, msg):
    write_bench(str(tmp_path), kind)
    if kind == 'check_bad':
        fake_check.answer = dict(ok=False, sha256='abc', error='byte 3 of frame 4000')
    if kind == 'check_sha':
        fake_check.answer = dict(ok=True, sha256='abd')
    with pytest.raises(ct.TableError, match=msg):
        ct.table(str(tmp_path))


def test_run_args_without_the_schedule():
    """C7T1-10, C7T2-1, C7T3-1: run args compared as run_hil.py's parser reads them."""
    a = '--test A-W1 --seed 0 --faults ~/x.csv --step-log ~/s.csv --ground traction'
    same = ct._run_args('--ground traction --seed 0 --test A-W1')
    assert ct._run_args(a) == same                                              # order, schedule dropped
    assert ct._run_args('--test=A-W1 --seed=0 --ground=traction') == same       # --opt=value
    assert ct._run_args('--test A-W1 --seed 0 --grou traction') == same         # abbreviation
    assert ct._run_args('--test A-W1 --fault y.csv --seed 0 --ground traction') == same   # --faults abbreviated
    assert ct._run_args('--test A-W1 --ground rigid --ground traction') != \
        ct._run_args('--test A-W1 --ground traction --ground rigid')            # C7T2-1: the last wins
    assert ct._run_args('--test A-W1 --ground rigid --grou traction') != \
        ct._run_args('--test A-W1 --grou traction --ground rigid')              # C7T3-1
    assert ct._run_args('--test A-W1 --ground traction') != ct._run_args('--test A-W1')   # default None apart
    with pytest.raises(ct.TableError, match='run_hil.py refuses'):
        ct._run_args('--test A-W1 --ground mud')
    with pytest.raises(ct.TableError, match='run_hil.py refuses'):
        ct._run_args('--test A-W1 --no-such-option 1')
    assert ct.task_duration(a) == ('A-W1', 9.0) == ct.task_duration('--seed 0 --tes=A-W1')
    with pytest.raises(ct.TableError, match='not a fault task'):
        ct.task_duration('--test A-K1')
    with pytest.raises(ct.TableError, match='not a fault task'):
        ct.task_duration('--test A-W1 --test A-K1')                            # the last wins
    with pytest.raises(ct.TableError, match='no --test'):
        ct.task_duration('--seed 0')


BASH_CHARS = list('{}[]$`*?()!;&|<>"#') + ["'", '\\', '\n', '\r']    # written here, not taken from the tool


@pytest.mark.parametrize('ch', BASH_CHARS)
def test_run_args_with_a_shell_character(ch):
    """C7T4-1: every character bash acts on before run_hil.py is refused, wherever it is in the args."""
    with pytest.raises(ct.TableError, match='shell characters'):
        ct._run_args(f'--test A-W1 --uart-port x{ch}y')
    with pytest.raises(ct.TableError, match='shell characters'):
        ct.task_duration(f'--test A-W1 --faults ~/a{ch}.csv')


def test_run_args_tilde_and_blanks_kept():
    """A ~ in --faults / --step-log (left out of the comparison) and blanks (tabs too): still read."""
    assert ct._run_args('--test A-W1\t--seed  0 --faults ~/fault_c7/x.csv --step-log ~/s.csv') == \
        ct._run_args('--test A-W1 --seed 0')
    assert ct._run_args('--test A-W1 --faults=~/x.csv --step-log=~/s.csv') == ct._run_args('--test A-W1')


@pytest.mark.parametrize('args', ['--test A-W1 --uart-port ~/uart', '--test A-W1 --uart-port=~/uart',
                                  '--test A-W1 --uart-port /dev/tty~1'])
def test_run_args_tilde_in_a_compared_option(args):
    """C7T5-1: bash expands ~ at the start of a word only ('--uart-port ~/uart' -> <HOME>/uart, '=~/uart' kept),
    shlex never: a ~ in an option that is compared is refused, in either form."""
    with pytest.raises(ct.TableError, match='~ in --uart-port'):
        ct._run_args(args)


def test_bench_tilde_pair_refused(tmp_path, fake_check):
    """C7T5-1 through a bench folder: the pair of the round-5 counterexample."""
    write_bench(str(tmp_path), 'args_tilde')
    with pytest.raises(ct.TableError, match='~ in --uart-port'):
        ct.table(str(tmp_path))


def test_bench_same_options_abbreviated(tmp_path, fake_check):
    """C7T3-1: the fault run written with abbreviations and --opt=value, same effective options: tabled."""
    write_bench(str(tmp_path), 'args_same_abbrev')
    assert len(ct.table(str(tmp_path))) == 1


def test_thesis_path(tmp_path):
    d = tmp_path / '05_DATA' / 'results' / 'FAULT_C7' / 'r2'
    d.mkdir(parents=True)
    got = ct._thesis_path(str(d), '05_DATA/results/FAULT_C7/r1/schedules/enc_FL.csv')
    assert os.path.normpath(got) == os.path.normpath(str(tmp_path / '05_DATA/results/FAULT_C7/r1/schedules/enc_FL.csv'))
    with pytest.raises(ct.TableError, match='not under a 05_DATA'):
        ct._thesis_path(str(tmp_path), '05_DATA/x.csv')


@pytest.mark.skipif(os.path.normcase('A') != 'a', reason='case-insensitive paths only (Windows)')
def test_thesis_path_any_case(tmp_path):
    """C7T2-4: on Windows 05_data is the same folder as 05_DATA."""
    d = tmp_path / '05_data' / 'results' / 'r2'
    d.mkdir(parents=True)
    got = ct._thesis_path(str(d), '05_DATA/results/r1/s.csv')
    assert os.path.normcase(got) == os.path.normcase(str(tmp_path / '05_DATA' / 'results' / 'r1' / 's.csv'))


FAULT_C7 = r'D:\CLAUDE_THESIS\05_DATA\results\FAULT_C7'


@pytest.mark.parametrize('run,mode', [('2026-11-14_run02', 'ADRC'), ('2026-11-14_run03', 'PID')])
def test_real_bench_runs(run, mode):
    """The two bench sessions of 14/11, with the real tools/fault_check.check run again on each fault run."""
    d = os.path.join(FAULT_C7, run)
    if not os.path.isfile(os.path.join(d, 'session.json')):
        pytest.skip('bench runs of 14/11 not on this machine')
    rows = {r['case']: r for r in ct.table(d)}
    assert sorted(rows) == ['enc_FL', 'gyro_s005', 'gyro_s03', 'imu_delay100', 'imu_loss']
    for r in rows.values():
        assert r['controller'] == mode and r['complete'] is True
        assert (r['win_start_s'], r['win_end_s']) == (3.0, 5.0)
        assert (r['ticks'], r['ticks_expected'], r['tick_gaps']) == (900, 900, 0)
        assert 2.0 <= r['ref_first_s'] < 2.01 and abs(r['esp_sp_first_s'] - 2.0) <= 0.02
        assert 8.0 < r['run_s'] < 8.2                                  # C7T1-1: RUN a little over 8 s
    # the imu_loss window of 2 s at 20 Hz: 40 gyro reports fewer than the baseline, wz from the encoders
    il = rows['imu_loss']
    assert il['base_esp_gyro_n'] - il['esp_gyro_n'] == 40
    assert 3.0 <= il['t_src_enc'] < 3.2 and 5.0 <= il['t_src_imu'] < 5.3


def test_cli(tmp_path):
    tool = os.path.join(ROOT, 'tools', 'fault_c7_table.py')
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_case(str(tmp_path / 'PID_encFL'), fault_log(), rows=[ROW])
    r = subprocess.run([sys.executable, tool, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    with open(tmp_path / 'c7_table.csv') as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 2 and lines[0].startswith('controller,case,') and lines[1].startswith('PID,encFL,')
    os.remove(tmp_path / 'PID_base' / 'log.csv')
    r = subprocess.run([sys.executable, tool, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and 'cannot table' in r.stderr


def test_cli_bench_refuses_without_fault_check(tmp_path):
    """The CLI on a bench folder: exit 2 when the fault run has no fault_check report (the real check needs a
    real run, so the CLI is not asked for a table of the synthetic one)."""
    tool = os.path.join(ROOT, 'tools', 'fault_c7_table.py')
    write_bench(str(tmp_path), 'no_check')
    r = subprocess.run([sys.executable, tool, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and 'no fault_check' in r.stderr



# ── batch 2 plant faults (design 2026-11-15 section 7.2): per-wheel saturation, yaw deviation, ramp split,
# zone entry from the plant record ───────────────────────────────────────────
B_ROW = dict(id='B1', fault='batt_sag', target='batt', start_ms=3000, end_ms=5000, params=dict(v1=7.44, ramp_ms=1000))
Z_ROW = dict(id='Z1', fault='mu_zone', target='all', start_ms=3000, end_ms=5000,
             params=dict(x0=1.0, x1=2.0, y0=-0.5, y1=0.5, mu=0.2))
REC_COLS = ('step', 'mask', 'V_batt', 'mu_fl', 'mu_fr', 'mu_rl', 'mu_rr', 'b_fl', 'b_fr', 'b_rl', 'b_rr',
            'tc_fl', 'tc_fr', 'tc_rl', 'tc_rr', 'ts_fl', 'ts_fr', 'ts_rl', 'ts_rr', 'x', 'y', 'theta')


def plant_record(step0=0, mask=3, zone_on=True):
    """Steps 3000..4999 (+ step0): x = -5 (all hubs out), 1.5 from 4000 (front hubs in), 5 from 4500 (out)."""
    k = np.arange(3000, 5000)
    rec = np.zeros((len(k), len(REC_COLS)))
    rec[:, 0] = k + step0
    rec[:, 1] = mask if zone_on else 1
    rec[:, REC_COLS.index('x')] = np.where(k < 4000, -5.0, np.where(k < 4500, 1.5, 5.0))
    return rec


def plant_log():
    lg = base_log()
    lg['v_vx'] = lg['sp_vx'] + np.where((T >= 3.0) & (T < 4.0), 0.1, np.where((T >= 4.0) & (T < 6.0), 0.3, 0.0))
    lg['pwm_rr'][(T >= 3.0) & (T < 4.0)] = 1000             # 100 of 300 span ticks, wheel RR only
    lg['theta'][:] = -3.1
    lg['theta'][T == 4.0] = 2.9                             # |2.9 - 3.1| = 0.2 > the wrapped 0.0832 elsewhere
    return lg


def write_plant_case(d, rows, rec=None):
    write_case(d, plant_log(), rows=rows)
    if rec is not None:
        np.savetxt(os.path.join(d, 'faults_plant.csv'), rec, delimiter=',', header=','.join(REC_COLS),
                   comments='', fmt='%.17g')


def test_plant_columns_fwsim(tmp_path):
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_plant_case(str(tmp_path / 'PID_sagzone'), [B_ROW, Z_ROW], plant_record())
    r = ct.table(str(tmp_path))[0]
    assert r['sat_frac_rr'] == pytest.approx(100 / 300) and r['sat_frac_fl'] == 0 and r['base_sat_frac_rr'] == 0
    assert r['sat_frac'] == pytest.approx(100 / 1200)
    assert r['yaw_dev_max'] == pytest.approx(0.2, abs=1e-6)
    assert r['ramp_end_s'] == pytest.approx(4.0)
    assert r['err_rms_ramp_vx'] == pytest.approx(0.1, abs=1e-6) and r['err_rms_hold_vx'] == pytest.approx(0.3, abs=1e-6)
    assert r['base_err_rms_ramp_vx'] == 0 and r['base_err_rms_hold_vx'] == 0
    assert r['err_rms_ramp_wz'] == pytest.approx(0.01, abs=1e-6)
    assert (r['zone_in_s'], r['zone_out_s']) == (4.0, 4.5)


def test_plant_columns_blank_without_such_rows(tmp_path):
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_case(str(tmp_path / 'PID_encFL'), fault_log(), rows=[ROW])
    r = ct.table(str(tmp_path))[0]
    assert r['ramp_end_s'] is None and r['err_rms_ramp_vx'] is None and r['base_err_rms_hold_wz'] is None
    assert r['zone_in_s'] is None and r['zone_out_s'] is None
    assert r['sat_frac_fl'] == pytest.approx(50 / 300) and r['sat_frac_fr'] == 0          # always written
    assert r['yaw_dev_max'] == pytest.approx(2 * math.pi - 6.2, abs=1e-6)


@pytest.mark.parametrize('rec,want', [
    (lambda: plant_record(zone_on=False), (None, None)),        # the zone row is never on in the record
    (lambda: plant_record()[:1200], (4.0, None)),               # still inside when the record ends
])
def test_zone_times_edges(tmp_path, rec, want):
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_plant_case(str(tmp_path / 'PID_zone'), [B_ROW, Z_ROW], rec())
    r = ct.table(str(tmp_path))[0]
    assert (r['zone_in_s'], r['zone_out_s']) == want


def test_ramp_zero_has_no_ramp_part(tmp_path):
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_plant_case(str(tmp_path / 'PID_sag'), [dict(B_ROW, params=dict(v1=7.44, ramp_ms=0))])
    r = ct.table(str(tmp_path))[0]
    assert r['ramp_end_s'] == 3.0 and r['err_rms_ramp_vx'] is None
    assert r['err_rms_hold_vx'] == pytest.approx(math.sqrt((100 * 0.01 + 200 * 0.09) / 300), abs=1e-6)


def test_zone_row_without_record_cannot_be_tabled(tmp_path):
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_plant_case(str(tmp_path / 'PID_zone'), [Z_ROW])
    assert ct.main([str(tmp_path)]) == 2


YAW_STEP = 4000                                  # ms after t0: the fault run's true theta + YAW_RAW int16 steps
YAW_RAW = 2086                                   # x pi / 32767 = 0.2000 rad


def bench_plant(d, file='PID_enc_FL_frames_plant_faults.npz'):
    """write_bench with batt_sag + mu_zone rows and a plant record (steps from the absolute frame step) in the
    fault run; its true theta has one spike of YAW_RAW int16 steps at t0 + YAW_STEP."""
    write_bench(d)
    npz = os.path.join(d, 'frames', 'PID_enc_FL_frames.npz')
    z = np.load(npz)
    arrs = {k: z[k].copy() for k in z.files}
    man = json.loads(str(arrs['manifest']))
    man['faults']['rows'] = [ROW, B_ROW, Z_ROW]
    man['faults']['plant_record'] = dict(file=file, sha256='x', rows=2000, inner='MecanumPlant')
    man['plant'] = 'FaultPlant'
    arrs['manifest'] = np.array(json.dumps(man))
    fr = arrs['mosi_true'][T0 + YAW_STEP]
    th = int.from_bytes(bytes(fr[12:14]), 'big', signed=True) + YAW_RAW
    fr[12:14] = np.frombuffer(th.to_bytes(2, 'big', signed=True), np.uint8)
    np.savez(npz, **arrs)
    np.savez(os.path.join(d, 'frames', 'PID_enc_FL_frames_plant_faults.npz'), record=plant_record(step0=T0, mask=6),
             cols=np.array(json.dumps(list(REC_COLS))))


def test_bench_plant_record_steps_from_t0(tmp_path, fake_check):
    d = str(tmp_path)
    bench_plant(d)
    r = ct.table(d)[0]
    assert (r['zone_in_s'], r['zone_out_s']) == (4.0, 4.5)              # record steps - t0
    assert r['ramp_end_s'] == 4.0
    q = 3.0 / 32767
    assert r['err_rms_ramp_wz'] == pytest.approx(0.01, abs=1e-3) and r['base_err_rms_ramp_vx'] == pytest.approx(0, abs=q)
    # Codex FQ1-3: answers known in advance; vx 0.4 at t0 + 5.9 s is in the hold part [4, 6) s, not in the ramp
    assert r['err_rms_ramp_vx'] == pytest.approx(0, abs=q)
    assert r['err_rms_hold_vx'] == pytest.approx(0.4 / math.sqrt(2000), rel=1e-3)
    assert r['yaw_dev_max'] == pytest.approx(YAW_RAW * math.pi / 32767, abs=2 * math.pi / 32767)
    assert r['sat_frac_fl'] == pytest.approx(50 / 300) and r['sat_frac_rr'] == 0
    kf = list(r)
    assert kf.index('ticks_expected') + 1 == kf.index('sat_frac_fl') and kf[-1] == 'blank'


@pytest.mark.parametrize('bad', ['../outside.npz', 123, '', '.', '..', 'sub/x.npz', 'sub\\x.npz', 'C:x.npz',
                                 os.path.abspath('x.npz')])
def test_bench_plant_record_name_refused_before_opening(tmp_path, fake_check, monkeypatch, bad):
    """Codex FQ1-2: a plant_record file that is not a bare name of the frames folder is refused before any open
    of it: exit 2, and np.load only ever saw the frame logs."""
    d = str(tmp_path)
    bench_plant(d, file=bad)
    seen, real = [], np.load

    def load(path, *a, **k):
        seen.append(str(path))
        return real(path, *a, **k)
    monkeypatch.setattr(ct.np, 'load', load)
    assert ct.main([d]) == 2
    assert seen and all(x.endswith('_frames.npz') for x in seen), seen


# ── s14 (design 2026-11-15 sections 7.2, 14.9 G1, 14.10; Liam 17/11 CODE_FAULT2 "lam s14 ngay di") ──
def test_task_duration_takes_the_tasks_of_fault_run():
    """G1: the bench table takes the tasks fwsim/fault_run.py task_setpoint takes: A-W1, or a group A test on the
    ground with its spec duration, + TAIL_S; the lifted tests and unknown ids are refused."""
    sys.path.insert(0, os.path.join(ROOT, 'rpi5'))
    from trajectory_pub import GROUP_A, LIFTED_TESTS
    assert ct.task_duration('--test A-W1') == ('A-W1', 9.0)
    for task in ('A-C1', 'A-L1', 'A-F2', 'A-S1'):
        assert ct.task_duration(f'--test {task}') == (task, GROUP_A[task].duration + 1.0), task
    for task in LIFTED_TESTS + ('A-ZZ',):
        with pytest.raises(ct.TableError, match='not a fault task'):
            ct.task_duration(f'--test {task}')


def test_task_duration_agrees_with_fault_run():
    sys.path.insert(0, os.path.join(ROOT, 'fwsim'))
    from trajectory_pub import GROUP_A, LIFTED_TESTS
    try:
        import fault_run
    except Exception as exc:                          # the fwsim DLL is not built here
        pytest.skip(f'fwsim not importable: {exc}')
    for task in ['A-W1'] + [t for t in GROUP_A if t not in LIFTED_TESTS]:
        assert ct.task_duration(f'--test {task}')[1] == fault_run.task_setpoint(task)[1] + fault_run.TAIL_S, task


def test_pwm_mean_fwsim(tmp_path):
    write_case(str(tmp_path / 'PID_base'), base_log())
    write_plant_case(str(tmp_path / 'PID_sagzone'), [B_ROW, Z_ROW], plant_record())
    write_case(str(tmp_path / 'PID_encFL'), fault_log(), rows=[ROW])
    rs = {r['case']: r for r in ct.table(str(tmp_path))}
    assert rs['sagzone']['pwm_mean'] == pytest.approx(1000 * 100 / 1200) and rs['sagzone']['base_pwm_mean'] == 0
    assert rs['encFL']['pwm_mean'] == pytest.approx(1000 * 50 / 1200)
    k = list(rs['encFL'])
    assert k.index('zone_out_s') + 1 == k.index('pwm_mean') and k[-1] == 'base_pwm_mean'


def test_pwm_mean_bench(tmp_path, fake_check):
    d = str(tmp_path)
    bench_plant(d)
    r = ct.table(d)[0]
    # esp_rows: FL -1000 for 50 of the 300 span ticks and 100 for the rest (fault run), 100 throughout (baseline)
    assert r['pwm_mean'] == pytest.approx((50 * 1000 + 250 * 100) / 1200)
    assert r['base_pwm_mean'] == pytest.approx(300 * 100 / 1200)
    kf = list(r)
    assert kf.index('zone_out_s') + 1 == kf.index('pwm_mean') and kf.index('base_pwm_mean') + 1 == kf.index('run_s')


# -- Codex round 1 (FS1-1, FS1-4) --
def test_tick_counts_of_a_duration():
    # FS1-1: fwsim ticks at 10 .. round(d / 1 ms) // 10 x 10 ms; the bench rows j x 10 ms < d
    assert ct.fwsim_ticks(9.0) == 900 and ct.bench_ticks(9.0) == 900
    assert ct.fwsim_ticks(16.0) == 1600 and ct.bench_ticks(16.0) == 1600
    sys.path.insert(0, os.path.join(ROOT, 'rpi5'))
    from trajectory_pub import GROUP_A
    c1 = GROUP_A['A-C1'].duration + 1.0                 # 32.4159.. s
    assert ct.fwsim_ticks(c1) == 3241 and ct.bench_ticks(c1) == 3242
    f2 = GROUP_A['A-F2'].duration + 1.0
    assert ct.bench_ticks(f2) == math.ceil(f2 * 100) and ct.bench_ticks(f2) - 1 < f2 * 100 <= ct.bench_ticks(f2)


def test_s02_prep_is_not_a_fault_task():
    # FS1-4: --s02-prep runs a longer, shifted task than --test alone
    with pytest.raises(ct.TableError, match='s02-prep'):
        ct.task_duration('--test A-L1 --finite-run --s02-prep 3')
    assert ct.task_duration('--test A-L1 --finite-run') == ('A-L1', 16.0)


# -- Codex round 2 (FS2-1) --
def test_tick_counts_at_the_half_ms_and_just_after_a_tick():
    # fwsim_ticks: the division of fwsim.simulate (duration / 0.001), not duration * 1000
    assert 1.0695 / 0.001 < 1069.5 <= 1.0695 * 1000          # the two roundings differ here
    assert ct.fwsim_ticks(1.0695) == int(round(1.0695 / 0.001)) // 10 == 106
    # bench_ticks: rows j x 10 ms < duration, the duration not rounded first
    assert ct.bench_ticks(9.000000001) == 901 and ct.bench_ticks(8.999999999) == 900
    assert ct.bench_ticks(0.0) == 0 and ct.bench_ticks(0.005) == 1 and ct.bench_ticks(0.01) == 1
    for d in (0.3, 1.0695, 9.0, 13.19444694, 32.41592653589793):
        n = ct.bench_ticks(d)
        assert round((n - 1) * ct.TICK_S, 6) < d <= round(n * ct.TICK_S, 6), d


# -- Codex round 3 (FS3-1) --
def test_bench_ticks_domain():
    import time
    t = time.perf_counter()
    for d in (ct.BENCH_MAX_S, 123456789.123456, 1e6 + 0.005):        # the largest taken: a few loop turns only
        n = ct.bench_ticks(d)
        assert round((n - 1) * ct.TICK_S, 6) < d <= round(n * ct.TICK_S, 6), d
    assert time.perf_counter() - t < 1.0
    for d in (1e307, 2.0 ** 100, ct.BENCH_MAX_S * 1.0000001, -0.001, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            ct.bench_ticks(d)
