"""fwsim/fault_run.py with the s14 body loads body_drag, load_dist (design 2026-11-15 sections 3, 6, 14.10; Liam
17/11 CODE_FAULT2 "lam s14 ngay di"): the run is its baseline until the window, the drag slows the body in closed
loop and the controller pushes harder, the record, the Pi wrapper gives the same plant states, the same seed gives
the same run, and the c7 table reads the output (pwm_mean).
Run: python -m pytest fwsim/tests"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))
import fwsim                                          # noqa: E402
import fault_plant as fp                              # noqa: E402
import fault_run                                      # noqa: E402
import fault_schedule as fs                           # noqa: E402
from mecanum_plant import MecanumPlant                # noqa: E402
from params import MecanumParams                      # noqa: E402
from protocol import FRAME_SIZE                       # noqa: E402
from test_fault_plant_fwsim import IMU, fw, run, sched   # noqa: E402,F401  (fw: module fixture)

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')
DIST = 'L1,load_dist,body,{a},{b},F0=2.5;rate=0.8;Fmax=1.5;sigma=1.6;seed=5;dir_deg={d}\n'


def test_baseline_until_the_window_then_the_drag_slows_the_body(fw):
    base = run(fw, fs.Schedule(()))
    log = run(fw, sched('D1,body_drag,body,1500,4000,F=3.3\n'))
    t = log['t']
    before = t < 1.5 - 1e-9
    for k in ('omega', 'pose', 'duty', 'count'):
        assert np.array_equal(base[k][before], log[k][before]), k
    win = (t > 2.0) & (t < 4.0)
    # closed loop at 0.3 m/s: the controller needs more duty against the drag
    assert np.mean(np.abs(log['duty'][win])) > 1.05 * np.mean(np.abs(base['duty'][win]))
    rec = log['fault_plant']
    cols = {c: i for i, c in enumerate(fp.RECORD_COLS)}
    assert list(rec[:, 0].astype(int)) == list(range(1500, 4000))
    assert np.all(rec[:, cols['f_drag']] == 3.3) and np.all(rec[:, cols['f_dist']] == 0.0)
    p = MecanumParams()
    assert np.all(rec[:, cols['V_batt']] == p.V_batt) and np.all(rec[:, cols['b_fl']] == p.motor_b[0])
    assert log['faults']['complete']


def test_load_record_and_same_seed_same_run(fw):
    s = sched(DIST.format(a=500, b=3500, d=90))
    a, b = run(fw, s), run(fw, s)
    for k in ('omega', 'pose', 'duty'):
        assert np.array_equal(a[k], b[k]), k
    rec = a['fault_plant']
    cols = {c: i for i, c in enumerate(fp.RECORD_COLS)}
    n = fp.dist_noise(s.rows[0])
    k = rec[:, 0].astype(int)
    assert list(k) == list(range(500, 3500))
    assert all(rec[i, cols['f_dist']] == fp.dist_value(s.rows[0], int(k[i]), n) for i in range(0, len(k), 97))
    base = run(fw, fs.Schedule(()))
    t = a['t']
    assert np.max(np.abs(a['pose'][t > 2.0, 1] - base['pose'][t > 2.0, 1])) > 1e-3    # pushed along y


def test_body_and_other_plant_rows_together(fw):
    """s14 combined (nightmare_config.m order): drag from 1 s, battery sag from 2 s, load from 3 s."""
    log = run(fw, sched('D1,body_drag,body,1000,4000,F=3.3\nB1,batt_sag,batt,2000,4000,v1=9;ramp_ms=1000\n'
                        + DIST.format(a=3000, b=4000, d=0)))
    rec = log['fault_plant']
    cols = {c: i for i, c in enumerate(fp.RECORD_COLS)}
    k = rec[:, 0].astype(int)
    assert list(k) == list(range(1000, 4000))
    assert set(rec[k < 2000, 1]) == {1.0} and set(rec[(k >= 2000) & (k < 3000), 1]) == {3.0}
    assert set(rec[k >= 3000, 1]) == {7.0}
    assert np.all(rec[k >= 3000, cols['f_dist']] != 0.0) and np.all(rec[k >= 3000, cols['V_batt']] == 9.0)


def test_pi_wrapper_and_fwsim_give_the_same_plant_with_body_rows(fw, monkeypatch):
    """The duty of every fwsim step replayed into a MecanumPlant wrapped by FaultPlant gives the same states, with
    body rows, a parameter row and steps without any row."""
    s = sched('D1,body_drag,body,200,1500,F=3.3\n' + DIST.format(a=700, b=1900, d=135) +
              'J1,wheel_jam,RL,1000,1200,b=0.05\n', run_ms=2000)
    calls = []
    real, real_body = fwsim.plant_step, fp.body_step

    def spy(x, u, p, dt, rng):
        x_new = real(x, u, p, dt, rng)
        calls.append((np.array(x, copy=True), np.array(u, copy=True), np.array(x_new, copy=True)))
        return x_new

    def spy_body(x, u, p, dt, f):
        x_new = real_body(x, u, p, dt, f)
        calls.append((np.array(x, copy=True), np.array(u, copy=True), np.array(x_new, copy=True)))
        return x_new

    monkeypatch.setattr(fwsim, 'plant_step', spy)
    monkeypatch.setattr(fault_run, '_REAL_PLANT_STEP', spy)
    monkeypatch.setattr(fp, 'body_step', spy_body)
    log = run(fw, s, duration=2.0)
    monkeypatch.undo()
    patcher = fs.FramePatcher(s)
    inner = MecanumPlant(dt=0.001, seed=0)
    inner.reset(list(calls[0][0]))
    plant = fp.FaultPlant(inner, patcher, None)
    patcher.start(0)
    for k, (x, u, x_new) in enumerate(calls):
        assert np.array_equal(plant.get_state(), x), k
        patcher.patch(k, bytes(FRAME_SIZE))
        plant.step(list(u))
        assert np.array_equal(plant.get_state(), x_new), k
    rec = log['fault_plant']
    assert len(calls) == 2000 and plant._n == len(rec) == 1700               # union of the windows [200, 1900)
    assert np.array_equal(plant._rec, rec)


def test_c7_table_reads_body_output(fw, tmp_path):
    import fault_c7_table as ct
    out = str(tmp_path)
    for case, body in (('base', None), ('drag', 'D1,body_drag,body,1000,3000,F=3.3\n')):
        d = os.path.join(out, f'PID_{case}')
        sch = fs.Schedule(()) if body is None else sched(body)
        log = run(fw, sch)
        met = fwsim.metrics(log)
        cfg = dict(task='T', mode='PID', seed=0, imu_rate=20.0, imu_boot_reset=-1.0, imu_noise='replay',
                   imu_noise_seed=0, ground='traction', mu=None, no_imu=False, duration=4.0)
        fwsim.write_run(d, log, met, fw, cfg, MecanumParams())
        fault_run.write_faults(d, log)
    rows = ct.table(out)
    assert len(rows) == 1 and rows[0]['case'] == 'drag'
    assert rows[0]['pwm_mean'] > rows[0]['base_pwm_mean']
    assert rows[0]['faults'].startswith('body_drag:body:')
