"""Tests for the DOE/GA tools on fwsim with the IMU and slip method A (28/10):
fwsim/s02_groupA_sim.py (logger rows) and fwsim/tune_ga.py (evaluation settings).
Run: python -m pytest fwsim/tests  (needs fwsim/build/fwsim.dll: python fwsim/build_fwsim.py)"""
import argparse
import csv
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))
import fwsim  # noqa: E402
import s02_groupA_sim  # noqa: E402
import s02_metrics  # noqa: E402
import tune_ga  # noqa: E402
from agv_logger import COLUMNS  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')


@pytest.fixture(scope='module')
def fw():
    return fwsim.FirmwareSim()


def sim_args(*argv):
    ap = argparse.ArgumentParser()
    fwsim.add_plant_imu_args(ap)
    return ap.parse_args(list(argv))


def turn(t):
    """1 s still, then forward 0.3 m/s while turning 0.6 rad/s (IMU source and heading loop busy)."""
    return (0.3, 0.0, 0.6) if t >= 1.0 else (0.0, 0.0, 0.0)


def test_plant_and_imu_options():
    a = sim_args()
    assert fwsim.imu_config(a) == dict(rate_hz=20.0, boot_reset=-1.0)
    assert fwsim.plant_params(a).ground_model == 'traction'          # params.py default since 0928e78
    b = sim_args('--no-imu', '--ground', 'rigid')
    assert fwsim.imu_config(b) is None and fwsim.plant_params(b).ground_model == 'rigid'
    c = sim_args('--mu', '0.3', '--imu-rate', '100')
    np.testing.assert_allclose(fwsim.plant_params(c).mu_hd, 0.3)
    assert fwsim.imu_config(c)['rate_hz'] == 100.0
    assert fwsim.plant_params(a, wheels_lifted=True).wheels_lifted


def test_logger_rows_with_imu(fw, tmp_path):
    a = sim_args()
    log = fwsim.simulate(fw, turn, 4.0, fwsim.plant_params(a), mode='PID', imu=fwsim.imu_config(a))
    rows = list(s02_groupA_sim.log_rows(log, True))
    assert all(len(r) == len(COLUMNS) == 66 for r in rows)
    col = {c: [r[i] for r in rows] for i, c in enumerate(COLUMNS)}
    assert col['log_seq'] == [str(i + 1) for i in range(len(rows))]
    assert set(col['imu_init']) == {'0'} and set(col['log_qdrop']) == {'0'}
    assert col['wz_src'][-1] == '1'                   # wz from the gyro once reports flow
    assert col['g_valid'][-1] == '1' and col['r_valid'][-1] == '1'
    assert float(col['wz_meas'][-1]) == pytest.approx(0.6, abs=0.1)
    assert set(col['gyro_n']) == {''} and set(col['ax']) == {''}   # not available in fwsim
    path = tmp_path / 'run.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    lg = s02_metrics.load_log(str(path))
    assert s02_metrics.log_loss(lg)[0] == 0.0


def test_logger_rows_without_imu(fw):
    a = sim_args('--no-imu')
    log = fwsim.simulate(fw, turn, 2.0, fwsim.plant_params(a), mode='PID', imu=None)
    rows = list(s02_groupA_sim.log_rows(log, False))
    i = COLUMNS.index('wz_src')
    assert all(r[i] == '' for r in rows) and all(r[COLUMNS.index('imu_init')] == '' for r in rows)


def test_ga_evaluation_passes_imu_and_fresh_params(fw, monkeypatch):
    seen = []
    real = fwsim.simulate

    def spy(fw_, sp, duration, params=None, **kw):
        seen.append((id(params), params.ground_model, kw.get('imu')))
        return real(fw_, sp, 1.0, params, **kw)      # short run: only the settings are checked

    monkeypatch.setattr(fwsim, 'simulate', spy)
    a = sim_args('--imu-rate', '100', '--ground', 'traction')
    history = []
    sp = fwsim.make_setpoint(argparse.Namespace(traj='steps', speed=0.3, radius=0.5, period=8.0))
    prob = tune_ga.make_problem(fw, 'PID', [0, 1], sp, sp.duration, tune_ga.BOUNDS['PID'], history, 'idf', a)
    out = {}
    prob._evaluate(np.array([1.5, 0.05, 1.0, 0.05]), out)
    assert len(seen) == 2 and seen[0][0] != seen[1][0]           # a new params object per run
    assert all(g == 'traction' and imu == dict(rate_hz=100.0, boot_reset=-1.0) for _, g, imu in seen)
    assert np.all(np.isfinite(out['F'])) and len(history) == 1
