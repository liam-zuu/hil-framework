"""tools/fault_c7_table.py (L82, design 8.3, spec S06 T8 type II): edges of the firmware flags, the numbers
of one (controller, case) row against its baseline on synthetic logs whose answers are known, and the folders
that cannot be tabled (no baseline, a baseline with rows, a different seed / firmware / tick times).
Run: python -m pytest tools/tests"""
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


def write_case(folder, lg, rows=(), config=None, esp32='5bde4c5', hil='a2de855', imu_rows=180, dropped=()):
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
                 f'  esp32: {json.dumps(dict(commit=esp32, dirty=False))}\n'
                 f'  hil_framework: {json.dumps(dict(commit=hil, dirty=False))}\n'
                 f'config: {json.dumps(config or CONFIG)}\n'
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
               esp32='16702ff' if kind == 'esp32' else '5bde4c5')
    with pytest.raises(ct.TableError, match=msg):
        ct.table(str(tmp_path))


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
