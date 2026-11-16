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
    s = sched('J1,wheel_jam,RL,300,1500,b=0.05;tc=0.02\nM1,mu_time,FR,100,300,mu=0.1\n'
              'Z1,mu_zone,all,300,2000,x0=0.15;x1=0.3;y0=-1;y1=1;mu=0.05\n'
              'B1,batt_sag,batt,500,1900,v1=9;ramp_ms=400\n', run_ms=2000)
    calls = []
    real = fwsim.plant_step

    def spy(x, u, p, dt, rng):
        x_new = real(x, u, p, dt, rng)
        calls.append((np.array(x, copy=True), np.array(u, copy=True), np.array(x_new, copy=True)))
        return x_new

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
    for k, (x, u, x_new) in enumerate(calls):                    # every step, the last one too (FP1-12)
        assert np.array_equal(plant.get_state(), x), k
        patcher.patch(k, bytes(FRAME_SIZE))
        plant.step(list(u))
        assert np.array_equal(plant.get_state(), x_new), k
    rec = log['fault_plant']
    assert len(calls) == 2000 and plant._n == len(rec) == 1900               # union of the windows [100, 2000)
    assert np.array_equal(plant._rec, rec)                                    # the whole record
    cols = {c: i for i, c in enumerate(fp.RECORD_COLS)}
    assert np.any(rec[:, cols['mu_fl']] == 0.05) and np.any(rec[rec[:, 0] >= 300, cols['mu_fl']] != 0.05)


def test_motor_off_follows_en_with_a_plant_row(fw):                 # Codex FP1-5
    """EN 0 for the whole run (SWA off): a plant row whose zone no wheel reaches must leave the plant exactly as
    the baseline with EN 0 (the variant copies the runtime EN of every step)."""
    far = sched('Z1,mu_zone,all,0,2000,x0=50;x1=60;y0=50;y1=60;mu=0.05\n', run_ms=2000)
    imu = dict(IMU)
    a = fault_run.FwsimFaults(fs.Schedule(())).run(fw, _sp, 2.0, MecanumParams(), imu=dict(imu), mode='PID',
                                                     motor_on=False)
    b = fault_run.FwsimFaults(far).run(fw, _sp, 2.0, MecanumParams(), imu=dict(imu), mode='PID', motor_on=False)
    for k in ('omega', 'pose', 'duty', 'count'):
        assert np.array_equal(a[k], b[k]), k
    assert len(b['fault_plant']) == 2000


def test_run_checks_the_schedule_itself(fw):                         # Codex FP1-8
    unchecked = fs.Schedule(tuple(fs.parse(HEAD + 'J1,wheel_jam,FL,0,20,b=0.1\nJ2,wheel_jam,FL,10,30,b=0.2\n')))
    with pytest.raises(fs.ScheduleError):
        run(fw, unchecked)
    assert fwsim.plant_step is fault_run._REAL_PLANT_STEP


def test_no_plant_row_no_plant_output(fw, tmp_path):                 # Codex FP1-10
    log = run(fw, fs.Schedule(()), duration=1.0)
    assert 'fault_plant' not in log
    fault_run.write_faults(str(tmp_path), log)
    assert not (tmp_path / 'faults_plant.csv').exists()
    assert 'faults_plant' not in (tmp_path / 'meta.yaml').read_text(encoding='utf-8')
    log = run(fw, sched('J1,wheel_jam,FL,0,500,b=0.1\n', run_ms=1000), duration=1.0)
    out = tmp_path / 'p'
    out.mkdir()
    fault_run.write_faults(str(out), log)
    assert (out / 'faults_plant.csv').exists() and 'faults_plant.csv' in (out / 'meta.yaml').read_text(encoding='utf-8')


def test_en_reaching_the_plant_is_the_runtime_en(fw, monkeypatch):   # Codex FP2-3
    """The EN of the parameter object that reaches plant_step at each step with a plant row on equals the EN
    simulate wrote on the nominal object for that step; the run sees EN change (1 before the first firmware
    tick, then the firmware's EN 0 with the motors off), so a variant that kept its EN would fail."""
    params = MecanumParams()
    seen = []
    real = fwsim.plant_step

    def spy(x, u, p, dt, rng):
        seen.append((p is params, p.motor_enable, params.motor_enable))
        return real(x, u, p, dt, rng)

    monkeypatch.setattr(fwsim, 'plant_step', spy)
    monkeypatch.setattr(fault_run, '_REAL_PLANT_STEP', spy)
    s = sched('Z1,mu_zone,all,0,1000,x0=50;x1=60;y0=50;y1=60;mu=0.05\n', run_ms=1000)
    fault_run.FwsimFaults(s).run(fw, _sp, 1.0, params, imu=dict(IMU), mode='PID', motor_on=False)
    monkeypatch.undo()
    assert len(seen) == 1000 and not any(is_base for is_base, _, _ in seen)      # the variant, every step
    assert all(got == want for _, got, want in seen)
    assert {want for _, _, want in seen} == {0.0, 1.0}                           # EN changed during the run



def test_c7_table_reads_real_plant_fault_output(fw, tmp_path):
    """tools/fault_c7_table.py (batch 2 columns, design 7.2) on folders written as fault_run.py writes them:
    faults_plant.csv is read, the ramp of batt_sag and the zone entry of mu_zone are in the row."""
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))
    import fault_c7_table as ct
    cfg = dict(task='A-W1', mode='PID', seed=0, imu_rate=20.0, imu_boot_reset=-1.0, imu_noise='replay',
               imu_noise_seed=0, ground=None, mu=None, no_imu=False, duration=4.0)
    body = ('B1,batt_sag,batt,1000,3000,v1=7.44;ramp_ms=1000\n'
            'Z1,mu_zone,all,1000,3000,x0=0.6;x1=5;y0=-1;y1=1;mu=0.2\n')
    for name, s in (('PID_base', fs.Schedule(())), ('PID_sagzone', sched(body))):
        log = run(fw, s)
        out = str(tmp_path / name)
        fwsim.write_run(out, log, fwsim.metrics(log), fw, dict(cfg), MecanumParams())
        fault_run.write_faults(out, log)
    assert os.path.isfile(str(tmp_path / 'PID_sagzone' / 'faults_plant.csv'))
    (r,) = ct.table(str(tmp_path))
    assert r['case'] == 'sagzone' and r['ramp_end_s'] == 2.0
    assert r['err_rms_ramp_vx'] is not None and r['err_rms_hold_vx'] is not None
    rec = np.loadtxt(str(tmp_path / 'PID_sagzone' / 'faults_plant.csv'), delimiter=',', skiprows=1, ndmin=2)
    nom = MecanumParams()
    x, y, th = rec[:, 19], rec[:, 20], rec[:, 21]
    hub_x = np.max([x + np.cos(th) * sx * nom.lx - np.sin(th) * sy * nom.ly           # largest hub x
                    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))], axis=0)
    k_in = rec[np.nonzero(hub_x >= 0.6)[0][0], 0] / 1000.0                          # |y| << 1: x decides
    assert r['zone_in_s'] == k_in and 1.0 < r['zone_in_s'] < 3.0
    assert r['zone_out_s'] is None                          # x1 = 5 m: never left while the row is on
