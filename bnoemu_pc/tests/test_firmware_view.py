"""imu_model.firmware_view (what fwsim feeds the firmware) against the HIL chain in C
(Codex round 1 IMN-2, 10/11/2026): Pi protocol.encode_state → bnoemu_pc/fwview.exe
(spi1_handler scaling, _sample_from_imu, bno_emu.c build_report, ESP32 SH-2 library,
BNO08x euler). Run python bnoemu_pc/build_bnoemu.py first."""
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import protocol as P  # noqa: E402
from imu_model import (firmware_view, ImuNoise, GYRO_STEP_RADPS, YAW_STEP_RAD,  # noqa: E402
                       ACCEL_STEP_MPS2)

EXE = os.path.join(HERE, '..', 'build', 'fwview.exe')
pytestmark = pytest.mark.skipif(not os.path.isfile(EXE), reason='fwview.exe not built')
OFS = [P.OFS_IMU_GZ, P.OFS_IMU_ACC, P.OFS_IMU_ACC + 2, P.OFS_IMU_ACC + 4, P.OFS_IMU_YAW]


def c_chain(samples):
    lines = []
    for s in samples:
        f = P.encode_state([0.0] * P.N_STATES, imu=s, imu_seq=1)
        lines.append(' '.join(str(struct.unpack_from('>h', f, o)[0]) for o in OFS))
    r = subprocess.run([EXE], input='\n'.join(lines) + '\n', capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-500:]
    out = np.array([[float(x) for x in l.split()] for l in r.stdout.splitlines()])
    assert out.shape == (len(samples), 5)
    return out


def py_chain(samples):
    rows = []
    for s in samples:
        g, acc, y = firmware_view(s)
        rows.append([g, *acc, y])
    return np.array(rows)


def _samples(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.column_stack([rng.uniform(-6, 6, n), rng.uniform(-15, 15, n), rng.uniform(-15, 15, n),
                         rng.uniform(9.0, 10.5, n), rng.uniform(-np.pi, np.pi, n)])
    # yaw near 0 (rest), near +-pi (wrap) and at the frame limits
    s[:500, 4] = rng.normal(0, 3e-4, 500)
    s[500:700, 4] = np.pi - np.abs(rng.normal(0, 1e-3, 200))
    s[700:900, 4] = -np.pi + np.abs(rng.normal(0, 1e-3, 200))
    s[900:910, 0] = [40.0, -40.0, 34.9, -34.9, 0, 0, 0, 0, 0, 0]          # gyro saturation
    return [tuple(map(float, r)) for r in s]


def _replayed(rate):
    """Samples as ImuNoise gives them: rest, then a turn at 1 rad/s (A-W1-like)."""
    noise = ImuNoise(rate, seed=2)
    per = noise.period_us
    out, yaw = [], 0.0
    for m in range(int(6 * rate)):
        t = m / rate
        wz = 0.0 if t < 2 else 1.0 if t < 4 else -1.0
        yaw += wz / rate
        out.append(noise.apply((wz, 0.0, 0.0, 9.80665, yaw), m * per))
    return out


def _compare(samples):
    c, p = c_chain(samples), py_chain(samples)
    assert np.array_equal(c[:, :4], p[:, :4]), 'gyro / accel differ'
    dy = np.abs(c[:, 4] - p[:, 4])
    dy = np.minimum(dy, 360.0 - dy)
    return dy


def test_random_samples_same_as_c():
    dy = _compare(_samples())
    one_step_deg = np.degrees(YAW_STEP_RAD)
    # float32 maths of two runtimes: allow a last-bit difference to move a value by
    # at most one Q14 step, in at most 0.1 % of the samples; all others equal to 1e-4 deg
    assert np.all(dy <= 1.01 * one_step_deg)
    assert np.mean(dy > 1e-4) <= 0.001, f'{np.sum(dy > 1e-4)} samples differ'


@pytest.mark.parametrize('rate', [20.0, 100.0])
def test_replayed_samples_same_as_c(rate):
    dy = _compare(_replayed(rate))
    assert np.mean(dy > 1e-4) <= 0.001


def test_rest_levels_are_q14_steps():
    """At yaw 0 + k steps the firmware sees exactly k Q14 steps (no second rounding)."""
    ks = np.arange(-6, 7)
    p = py_chain([(0.0, 0.0, 0.0, 9.80665, float(k) * YAW_STEP_RAD) for k in ks])
    assert np.allclose(np.radians(p[:, 4]), ks * YAW_STEP_RAD, atol=2e-7)
    g = py_chain([(float(k) * GYRO_STEP_RADPS, float(k) * ACCEL_STEP_MPS2, 0.0, 9.80665, 0.0) for k in ks])
    assert np.array_equal(g[:, 0], ks * GYRO_STEP_RADPS)
    assert np.array_equal(g[:, 1], ks * ACCEL_STEP_MPS2)
