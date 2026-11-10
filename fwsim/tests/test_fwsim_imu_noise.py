"""fwsim with the measured IMU noise + output steps (imu_model.ImuNoise, L71)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
from imu_model import (ImuNoise, NOISE_20HZ, NOISE_100HZ, GYRO_STEP_RADPS,  # noqa: E402
                       ACCEL_STEP_MPS2)

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')

C = {n: i for i, n in enumerate(fwsim.IMU_COLS)}


@pytest.fixture(scope='module')
def fw():
    return fwsim.FirmwareSim()


def _turn(t):
    return (0.0, 0.0, 0.8 if 0.5 <= t < 2.5 else 0.0)


def _run(fw, noise, rate_hz=20.0, seen=None):
    def hook(t, kind, v):
        if seen is not None:
            seen.append((kind, v))
        return v
    return fwsim.simulate(fw, _turn, 3.0, seed=0, mode='PID',
                          imu=dict(rate_hz=rate_hz, noise=noise, hook=hook))


@pytest.mark.parametrize('spec', [NOISE_20HZ, NOISE_100HZ])
def test_reports_carry_the_output_steps(fw, spec):
    seen = []
    _run(fw, ImuNoise(spec, seed=1), rate_hz=spec.rate_hz, seen=seen)
    gyro = [v for k, v in seen if k == 'gyro']
    acc = [v for k, v in seen if k == 'accel']
    assert len(gyro) == pytest.approx(3.0 * spec.rate_hz, abs=2)
    assert all(abs(g / GYRO_STEP_RADPS - round(g / GYRO_STEP_RADPS)) < 1e-9 for g in gyro)
    assert all(abs(a / ACCEL_STEP_MPS2 - round(a / ACCEL_STEP_MPS2)) < 1e-9 for v in acc for a in v)
    # at rest before the turn most gyro reports are exactly 0
    rest = [v for (k, v), i in zip([s for s in seen if s[0] == 'gyro'], range(len(gyro))) if i < 0.4 * spec.rate_hz]
    assert sum(1 for v in rest if v == 0.0) >= 0.7 * len(rest)


def test_same_seed_same_run_other_seed_differs(fw):
    a = _run(fw, ImuNoise(NOISE_20HZ, seed=3))
    b = _run(fw, ImuNoise(NOISE_20HZ, seed=3))
    c = _run(fw, ImuNoise(NOISE_20HZ, seed=4))
    assert np.array_equal(np.asarray(a['duty']), np.asarray(b['duty']))
    assert not np.array_equal(np.asarray(a['imu'])[:, C['gz']], np.asarray(c['imu'])[:, C['gz']])


def test_no_noise_is_the_old_behaviour(fw):
    seen = []
    _run(fw, None, seen=seen)
    gyro = [v for k, v in seen if k == 'gyro']
    assert any(abs(g / GYRO_STEP_RADPS - round(g / GYRO_STEP_RADPS)) > 1e-6 for g in gyro)   # not on the grid
