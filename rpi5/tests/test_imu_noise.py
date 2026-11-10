"""Tests for imu_model.ImuNoise (measured BNO085 noise + output steps, L71) and its use
on the HIL Pi (hil_node._imu_sample, frame encoding to the H7)."""
import math
import multiprocessing as mp

import numpy as np
import pytest

from params import PARAMS
from mecanum_plant import MecanumPlant
from hil_node import HILConfig, HILNode
from imu_model import (ImuNoise, ImuNoiseSpec, NOISE_20HZ, NOISE_100HZ, GYRO_STEP_RADPS,
                       ACCEL_STEP_MPS2, noise_spec_for_rate)
from protocol import encode_state, decode_imu


def _is_multiple(v, q):
    return abs(v / q - round(v / q)) < 1e-9


def _ncdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class TestSpec:
    def test_rates(self):
        assert noise_spec_for_rate(20) is NOISE_20HZ
        assert noise_spec_for_rate(100.0) is NOISE_100HZ
        with pytest.raises(ValueError):
            noise_spec_for_rate(50)

    def test_measured_numbers(self):
        # 05_DATA/results/IMU_NOISE/2026-11-10_analysis/ket_qua.txt (gz MLE sigma, yaw std, lag1)
        assert NOISE_20HZ.sigma_gz == pytest.approx(5.89e-4)
        assert NOISE_100HZ.sigma_gz == pytest.approx(3.99e-4)
        assert NOISE_20HZ.tau_yaw == pytest.approx(1.97, abs=0.01)
        assert NOISE_100HZ.tau_yaw == pytest.approx(0.318, abs=0.002)
        assert GYRO_STEP_RADPS == 2.0 ** -9 and ACCEL_STEP_MPS2 == pytest.approx(0.0390625)


class TestNoise:
    def test_output_steps(self):
        n = ImuNoise(NOISE_20HZ, seed=1)
        for i in range(500):
            gz, ax, ay, az, yaw = n.apply((0.3 * math.sin(i), 0.12, -0.05, PARAMS.g, 0.1), 0.05)
            assert _is_multiple(gz, GYRO_STEP_RADPS)
            for a in (ax, ay, az):
                assert _is_multiple(a, ACCEL_STEP_MPS2)
            assert -math.pi <= yaw < math.pi

    @pytest.mark.parametrize('spec', [NOISE_20HZ, NOISE_100HZ])
    def test_share_of_zero_gyro_matches_model(self, spec):
        n = ImuNoise(spec, seed=2)
        g = [n.apply((0.0, 0.0, 0.0, PARAMS.g, 0.0), 1.0 / spec.rate_hz)[0] for _ in range(20000)]
        share0 = sum(1 for v in g if v == 0.0) / len(g)
        p0 = 2 * _ncdf(GYRO_STEP_RADPS / 2 / spec.sigma_gz) - 1
        assert share0 == pytest.approx(p0, abs=0.01)

    @pytest.mark.parametrize('dt', [0.001, 0.05])
    def test_yaw_noise_stationary_sigma_and_correlation(self, dt):
        spec = NOISE_20HZ
        n = ImuNoise(spec, seed=3)
        steps = int(round(400 * spec.tau_yaw / dt))
        e = np.array([n.apply((0.0, 0.0, 0.0, PARAMS.g, 0.0), dt)[4] for _ in range(steps)])
        assert e.std() == pytest.approx(spec.sigma_yaw, rel=0.12)
        lag1 = np.corrcoef(e[:-1], e[1:])[0, 1]
        assert lag1 == pytest.approx(math.exp(-dt / spec.tau_yaw), abs=0.01)

    def test_seed_reproducible_and_seeds_differ(self):
        s = (0.2, 0.1, 0.0, PARAMS.g, 0.5)
        a = [ImuNoise(NOISE_100HZ, seed=7).apply(s, 0.01) for _ in range(1)]
        b = [ImuNoise(NOISE_100HZ, seed=7).apply(s, 0.01) for _ in range(1)]
        c = [ImuNoise(NOISE_100HZ, seed=8).apply(s, 0.01) for _ in range(1)]
        assert a == b and a != c

    def test_yaw_wraps(self):
        n = ImuNoise(ImuNoiseSpec(20.0, 0.0, 0.01, 1.0), seed=0)
        yaws = [n.apply((0.0, 0.0, 0.0, PARAMS.g, math.pi - 1e-4), 0.05)[4] for _ in range(200)]
        assert all(-math.pi <= y < math.pi for y in yaws)
        assert any(y < 0 for y in yaws)      # crossed +pi, came out near -pi


def _node(plant, **cfg):
    config = HILConfig(simulate_hw=True, max_steps=5, period_us=1_000, **cfg)
    return HILNode(plant, config=config, shared_state=mp.Array('d', 10))


class TestHilNode:
    def test_plant_values_get_noise_and_steps(self):
        node = _node(MecanumPlant(PARAMS, dt=0.001), imu_noise=ImuNoise(NOISE_20HZ, seed=4))
        imu, seq = node._imu_sample(node.plant.get_state())
        assert seq == 1
        assert _is_multiple(imu[0], GYRO_STEP_RADPS)
        assert _is_multiple(imu[3], ACCEL_STEP_MPS2) and abs(imu[3] - PARAMS.g) <= ACCEL_STEP_MPS2 / 2
        assert imu[4] != 0.0                       # yaw noise

    def test_static_values_are_not_noised(self):
        st = (0.5, 0.2, -0.1, 9.81, 1.0)
        node = _node(MecanumPlant(PARAMS, dt=0.001), imu_static=st, imu_noise=ImuNoise(NOISE_20HZ, seed=4))
        assert node._imu_sample(node.plant.get_state())[0] == st

    def test_gyro_step_survives_the_frame(self):
        # frame LSB of gz is 34.9/32767 = 0.00107 rad/s < GYRO_STEP/2, so the H7 can round back to
        # the same multiple of 2^-9 (its own Q9 packing; checked here only on the Pi side)
        n = ImuNoise(NOISE_20HZ, seed=5)
        for i in range(300):
            imu = n.apply((0.4 * math.sin(0.1 * i), 0.0, 0.0, PARAMS.g, 0.0), 0.05)
            rec = decode_imu(encode_state([0.0] * 10, imu=imu, imu_seq=1))
            assert round(rec[0] / GYRO_STEP_RADPS) == round(imu[0] / GYRO_STEP_RADPS)
            assert round(rec[3] / ACCEL_STEP_MPS2) == round(imu[3] / ACCEL_STEP_MPS2)
