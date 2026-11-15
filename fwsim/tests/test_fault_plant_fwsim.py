"""Fault injection batch 2, plant faults in fwsim (fault_run.py + rpi5/fault_plant.py; design 2026-11-15 section 6):
the run equals its baseline before the window, the plant row changes the plant from its first step, the
record holds exactly the window steps with the values of fault_plant.PlantVariant, the nominal parameter object
is untouched, checks against the nominal values, the Pi wrapper and fwsim give the same plant.
Run: python -m pytest fwsim/tests (needs fwsim/build/fwsim.dll: python fwsim/build_fwsim.py)"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import fwsim                                          # noqa: E402
import fault_plant as fp                              # noqa: E402
import fault_run                                      # noqa: E402
import fault_schedule as fs                           # noqa: E402
from mecanum_plant import MecanumPlant                # noqa: E402
from params import MecanumParams                      # noqa: E402
from protocol import FRAME_SIZE                       # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')
HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
IMU = dict(rate_hz=20.0, noise='replay', noise_seed=0)


@pytest.fixture(scope='module')
def fw():
    return fwsim.FirmwareSim()


def sched(body, run_ms=4000):
    rows = fs.parse(HEAD + body)
    fs.check(rows, run_ms, 'plant', 'traction')
    return fs.Schedule(tuple(rows), 'x.csv', 'abc')


def _sp(t):
    return (0.3, 0.0, 0.0)


def run(fw, s, params=None, duration=4.0):
    return fault_run.FwsimFaults(s).run(fw, _sp, duration, params or MecanumParams(), imu=dict(IMU), mode='PID')


def test_baseline_until_the_window_then_the_controller_pushes_the_jammed_wheel(fw):
    base = run(fw, fs.Schedule(()))
    p = MecanumParams()
    b0 = np.array(p.motor_b)
    log = run(fw, sched('J1,wheel_jam,FL,1500,3500,b=0.1\n'), params=p)
    t = log['t']
    before = t < 1.5 - 1e-9
    for k in ('omega', 'pose', 'duty', 'count'):
        assert np.array_equal(base[k][before], log[k][before]), k
    assert not np.array_equal(base['omega'][t > 1.6], log['omega'][t > 1.6])
    # closed loop: the wheel is held near its speed by the ground and the controller, which needs more duty
    win = (t > 2.0) & (t < 3.5)
    assert np.mean(np.abs(log['duty'][win, 0])) > 1.2 * np.mean(np.abs(base['duty'][win, 0]))
    rec = log['fault_plant']
    assert list(rec[:, 0].astype(int)) == list(range(1500, 3500)) and set(rec[:, 1]) == {1.0}
    assert np.array_equal(rec[0, 7:11], b0 + [0.1, 0, 0, 0])       # b_fl..b_rr
    assert np.array_equal(p.motor_b, b0)                            # the nominal object is untouched
    assert log['faults']['complete']


def test_batt_and_zone_values_in_the_record(fw):
    p = MecanumParams()
    log = run(fw, sched('B1,batt_sag,batt,1000,3000,v1=8;ramp_ms=1000\n'
                        'Z1,mu_zone,all,0,4000,x0=0.3;x1=0.6;y0=-1;y1=1;mu=0.05\n'), params=p)
    rec = log['fault_plant']
    cols = {c: i for i, c in enumerate(fp.RECORD_COLS)}
    steps = rec[:, 0].astype(int)
    assert list(steps) == list(range(0, 4000))                      # the zone row is on all the run
    v = rec[:, cols['V_batt']]
    assert v[steps == 999][0] == p.V_batt and v[steps == 1000][0] == p.V_batt
    assert v[steps == 1500][0] == pytest.approx(p.V_batt + (8 - p.V_batt) * 0.5, abs=1e-12)
    assert np.all(v[(steps >= 2000) & (steps < 3000)] == 8.0) and np.all(v[steps >= 3000] == p.V_batt)
    # per step: mu of a wheel = 0.05 exactly when its hub is in the zone at the start of the step
    var_rows = log['faults']['rows']
    assert var_rows[1]['fault'] == 'mu_zone'
    for r in rec[::50]:
        hubs = fp.contact_points(r[cols['x']:cols['theta'] + 1], p)
        for w, (hx, hy) in enumerate(hubs):
            inside = 0.3 <= hx < 0.6 and -1 <= hy < 1
            assert r[cols['mu_fl'] + w] == (0.05 if inside else 0.436)
    assert np.any(rec[:, cols['mu_fl']] == 0.05) and np.any(rec[:, cols['mu_rr']] == 0.436)


def test_refuses_values_above_nominal(fw):
    with pytest.raises(fs.ScheduleError):
        run(fw, sched('M1,mu_time,FL,100,200,mu=0.5\n'))
    p = MecanumParams()
    p.ground_model = 'rigid'
    with pytest.raises(fs.ScheduleError):
        run(fw, sched('J1,wheel_jam,FL,100,200,b=0.1\n'), params=p)
    assert fwsim.plant_step is fault_run._REAL_PLANT_STEP


def test_pi_wrapper_and_fwsim_give_the_same_plant(fw, monkeypatch):
    """The same duty sequence through the Pi FaultPlant and through the fwsim wrapper gives the same states:
    duties of the fwsim run (held per step) replayed into a MecanumPlant wrapped by FaultPlant."""
    s = sched('J1,wheel_jam,RL,300,1500,b=0.05;tc=0.02\nM1,mu_time,FR,800,1800,mu=0.1\n'
              'B1,batt_sag,batt,500,1900,v1=9;ramp_ms=400\n', run_ms=2000)
    calls = []
    real = fwsim.plant_step

    def spy(x, u, p, dt, rng):
        calls.append((np.array(x, copy=True), np.array(u, copy=True)))
        return real(x, u, p, dt, rng)

    # under the fault_run wrapper: sees the duty of every step (the wrapper takes it as the real plant_step)
    monkeypatch.setattr(fwsim, 'plant_step', spy)
    monkeypatch.setattr(fault_run, '_REAL_PLANT_STEP', spy)
    log = run(fw, s, duration=2.0)
    monkeypatch.undo()
    patcher = fs.FramePatcher(s)
    inner = MecanumPlant(dt=0.001, seed=0)
    inner.reset(list(calls[0][0]))
    plant = fp.FaultPlant(inner, patcher, None)
    patcher.start(0)
    for k, (x, u) in enumerate(calls[:-1]):
        assert np.array_equal(plant.get_state(), x), k
        patcher.patch(k, bytes(FRAME_SIZE))
        plant.step(list(u))
    assert np.array_equal(plant._rec[:plant._n], log['fault_plant'][:plant._n])
