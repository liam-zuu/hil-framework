"""fwsim with the replayed IMU noise (imu_model.ImuNoise, L72) and imu_model.firmware_view
(what the firmware gets on HIL, Codex round 1 IMN-2); noise off = the behaviour before."""
import hashlib
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
from imu_model import (ImuNoise, firmware_view, imu_from_plant, GYRO_STEP_RADPS,  # noqa: E402
                       ACCEL_STEP_MPS2, YAW_STEP_RAD)

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')

C = {n: i for i, n in enumerate(fwsim.IMU_COLS)}
# sha256 of duty + imu columns of _base_run() with fwsim.py of 6f93607 (before the noise model)
# and the DLL of AGV 54b7269, computed 10/11/2026 on the CODE PC (MinGW gcc 13.2): this
# machine and build only
BASELINE_6F93607 = '9e9082190f98c01e9386eb9ebd432b8035f665acc4ebd95f07cee73b26906082'


@pytest.fixture(scope='module')
def fw():
    return fwsim.FirmwareSim()


def _turn(t):
    return (0.0, 0.0, 0.8 if 0.5 <= t < 2.5 else 0.0)


def _run(fw, noise, rate_hz=20.0, seen=None, motor_on=True):
    def hook(t, kind, v):
        if seen is not None:
            seen.append((t, kind, v))
        return v
    return fwsim.simulate(fw, _turn, 3.0, seed=0, mode='PID', motor_on=motor_on,
                          imu=dict(rate_hz=rate_hz, noise=noise, hook=hook))


@pytest.mark.parametrize('rate', [20.0, 100.0])
def test_reports_are_the_firmware_view_of_the_replay(fw, rate):
    seen = []
    noise = ImuNoise(rate, seed=1)
    _run(fw, noise, rate_hz=rate, seen=seen, motor_on=False)   # vehicle stays at rest
    gyro = [v for _, k, v in seen if k == 'gyro']
    acc = [v for _, k, v in seen if k == 'accel']
    rv = [v for _, k, v in seen if k == 'rv']
    assert len(gyro) == len(acc) == len(rv) == pytest.approx(3.0 * rate, abs=2)
    assert all(abs(g / GYRO_STEP_RADPS - round(g / GYRO_STEP_RADPS)) < 1e-9 for g in gyro)
    assert all(abs(a / ACCEL_STEP_MPS2 - round(a / ACCEL_STEP_MPS2)) < 1e-9 for v in acc for a in v)
    # at rest (motors off, truth 0): each report is exactly the replay row of its report index
    rows = [noise.row(int(round(t * 1e6))) for t, k, _ in seen if k == 'gyro']
    assert gyro == [float(noise.replay.k_gz[i]) * GYRO_STEP_RADPS for i in rows]
    for v, i in zip(rv, rows):
        assert math.radians(v) == pytest.approx(float(noise.replay.k_yaw[i]) * YAW_STEP_RAD, abs=1e-7)


@pytest.mark.parametrize('rate', [20.0, 100.0])
def test_report_values_equal_firmware_view_of_noise_apply(fw, rate, monkeypatch):
    """While turning (truth yaw off the Q14 grid, truth gz and accel off their grids), every
    accel, gyro and game-RV value fed to the firmware is firmware_view(noise.apply(truth, t_us))
    with t_us = m * period (Codex round 2 IMN2-2: catches simulate() skipping firmware_view)."""
    truth = []
    real = fwsim.imu_from_plant
    monkeypatch.setattr(fwsim, 'imu_from_plant', lambda *a: truth.append(real(*a)) or truth[-1])
    seen = []
    _run(fw, ImuNoise(rate, seed=6), rate_hz=rate, seen=seen)
    ref = ImuNoise(rate, seed=6)
    reports = {}
    for t, k, v in seen:
        reports.setdefault(int(round(t * 1e6)), {})[k] = v
    times = sorted(reports)
    assert len(times) == len(truth) > 2 * rate
    off_grid = differs = 0
    for t_us, s in zip(times, truth):
        assert t_us % ref.period_us == 0
        noisy = ref.apply(s, t_us)
        g, acc, y = firmware_view(noisy)
        r = reports[t_us]
        assert r['gyro'] == g and tuple(r['accel']) == acc and r['rv'] == y
        k = s[4] / YAW_STEP_RAD
        off_grid += abs(k - round(k)) > 0.2
        differs += r['rv'] != math.degrees(noisy[4])      # what skipping firmware_view would feed
    assert off_grid > 0.3 * len(times) and differs > 0.3 * len(times)


def test_same_seed_same_run_other_seed_differs(fw):
    a = _run(fw, ImuNoise(20.0, seed=3))
    b = _run(fw, ImuNoise(20.0, seed=3))
    c = _run(fw, ImuNoise(20.0, seed=4))
    assert np.array_equal(np.asarray(a['duty']), np.asarray(b['duty']))
    assert np.array_equal(np.asarray(a['imu']), np.asarray(b['imu']))
    assert not np.array_equal(np.asarray(a['imu'])[:, C['yaw']], np.asarray(c['imu'])[:, C['yaw']])


def _base_run(fw):
    log = fwsim.simulate(fw, lambda t: (0.1, 0.0, 0.8 if 0.5 <= t < 2.5 else 0.0), 3.0, seed=0, mode='PID',
                         imu=dict(rate_hz=20.0))
    return hashlib.sha256(np.ascontiguousarray(np.asarray(log['duty'], dtype=np.float64)).tobytes() +
                          np.ascontiguousarray(np.asarray(log['imu'], dtype=np.float64)).tobytes()).hexdigest()


def test_no_noise_is_the_old_behaviour(fw):
    b = fw.build_info.get('esp32', {})
    if not (str(b.get('commit', '')).startswith('54b7269') and not b.get('dirty', True)):
        pytest.skip('baseline hash is for the DLL of AGV 54b7269')
    assert _base_run(fw) == BASELINE_6F93607
