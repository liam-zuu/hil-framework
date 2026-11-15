"""s02_groupA_sim (spec S02 P2 on fwsim): the IMU noise replay seed is the repetition number
(Liam 11/11 CODE local_7787e22f), so the N repetitions get different noise segments and a re-run
of one repetition is the same run bit for bit (S02 v1.3 A-R2-2).
Run: python -m pytest fwsim/tests/test_s02_groupA_sim.py"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import s02_groupA_sim as s02  # noqa: E402

HAVE_DLL = os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll'))


def test_rep_imu_sets_the_noise_seed_to_the_repetition():
    imu = dict(rate_hz=20.0, boot_reset=False, noise='replay', noise_seed=0)
    assert s02.rep_imu(imu, 3)['noise_seed'] == 3 and imu['noise_seed'] == 0      # the input is not changed
    assert s02.rep_imu(dict(rate_hz=20.0, boot_reset=False), 3) == dict(rate_hz=20.0, boot_reset=False)
    assert s02.rep_imu(None, 3) is None


@pytest.mark.skipif(not HAVE_DLL, reason='fwsim.dll not built')
def test_repetitions_differ_by_noise_and_a_rerun_is_identical():
    fw = fwsim.FirmwareSim()
    imu = dict(rate_hz=20.0, boot_reset=False, noise='replay', noise_seed=0)

    def run(rep):
        fw.reset()
        log = fwsim.simulate(fw, lambda t: (0.0, 0.0, 0.5 if t > 1.0 else 0.0), 4.0, None,
                             seed=rep, mode='PID', imu=s02.rep_imu(imu, rep))
        return np.asarray(log['imu'])[:, fwsim.IMU_COLS.index('gz')]

    g1, g2, g1b = run(1), run(2), run(1)
    assert np.array_equal(g1, g1b)              # same seed: bit for bit
    assert not np.array_equal(g1, g2)           # another repetition: another noise segment


@pytest.mark.skipif(not HAVE_DLL, reason='fwsim.dll not built')
def test_preparation_and_plant_file(tmp_path):
    """CODE_S02 (design 10.1, 11): 3 s of setpoint 0 before k = 0; the plant file has one row per log line,
    same timestamp_us; the command anchors put t = 0 at 3 s, the run is a cold start with no flag."""
    import csv
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
    import s02_cmd_path
    import s02_metrics
    from trajectory_pub import GROUP_A
    fw = fwsim.FirmwareSim()
    spec = s02.prep_spec(GROUP_A['A-L2'], 3.0)
    sp = lambda t: s02.get_body_velocity(t, spec) if t < spec.duration else (0.0, 0.0, 0.0)  # noqa: E731
    fw.reset()
    log = fwsim.simulate(fw, sp, spec.duration + s02.STOP_TAIL_S, None, seed=1, mode='PID',
                         imu=dict(rate_hz=20.0, boot_reset=False))
    p_log, p_plant = tmp_path / 'r.csv', tmp_path / 'r_plant.csv'
    with open(p_log, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(s02.COLUMNS)
        w.writerows(s02.log_rows(log, True))
    with open(p_plant, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(s02.PLANT_COLUMNS)
        w.writerows(s02.plant_rows(log))
    L = s02_metrics.load_log(str(p_log))
    with open(p_plant, newline='') as f:
        rows = list(csv.DictReader(f))
    assert [int(r['timestamp_us']) for r in rows] == [int(x) for x in L['_all']['timestamp_us']]
    t = L['_all']['timestamp_us']
    assert np.all(L['_all']['sp_vx'][t < 3.0e6] == 0.0) and np.all(L['_all']['sp_vx'][(t >= 3.0e6) & (t < 9.0e6)] > 0)
    s = s02_cmd_path.stats(L['_all'], 'A-L2', 'sim')
    assert s['t0_method'] == 'two_anchors' and s['t0_us'] == pytest.approx(3.0e6, abs=1)
    assert s['edge_kind'] == 'cold_start' and s['prep_flags'] == '' and s['cmd_reason'] == 'no_traj_age'
    # plant columns agree with the simulation (body velocity, pose, wheel 1)
    i = len(rows) // 2
    assert float(rows[i]['vx_b']) == pytest.approx(log['v_true'][i][0], abs=1e-6)
    assert float(rows[i]['x']) == pytest.approx(log['pose'][i][0], abs=1e-6)
    assert float(rows[i]['w1']) == pytest.approx(log['omega'][i][0], abs=1e-5)
    th = float(rows[i]['theta'])
    assert float(rows[i]['vx_w']) == pytest.approx(np.cos(th) * float(rows[i]['vx_b']) - np.sin(th) * float(rows[i]['vy_b']), abs=1e-5)
