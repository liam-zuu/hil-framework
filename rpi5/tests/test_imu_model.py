"""Tests for imu_model.py and the IMU part of the HIL node frame (G2)."""
import math
import multiprocessing as mp

import pytest

from params import PARAMS
from plant_interface import MockPlant
from mecanum_plant import MecanumPlant
from hil_node import HILConfig, HILNode
from imu_model import imu_from_plant, parse_imu_static
from protocol import encode_state, decode_imu, IMU_RANGES


def _state(theta=0.0, vx_w=0.0, vy_w=0.0, wz=0.0):
    return [0.0, 0.0, theta, vx_w, vy_w, wz, 0.0, 0.0, 0.0, 0.0]


class TestImuFromPlant:
    def test_at_rest_reads_gravity_only(self):
        s = _state(theta=0.7)
        assert imu_from_plant(s, s, 0.001, 9.81) == (0.0, 0.0, 0.0, 9.81, 0.7)

    def test_first_step_has_no_acceleration(self):
        gz, ax, ay, az, yaw = imu_from_plant(_state(vx_w=1.0, wz=0.3), None, 0.001, 9.81)
        assert (gz, ax, ay, az) == (0.3, 0.0, 0.0, 9.81)

    def test_forward_acceleration_heading_zero(self):
        _, ax, ay, _, _ = imu_from_plant(_state(vx_w=0.002), _state(), 0.001, 9.81)
        assert ax == pytest.approx(2.0)
        assert ay == pytest.approx(0.0)

    def test_world_acceleration_rotated_into_body(self):
        # heading +90 deg: world +y is body +x (forward), world +x is body -y
        th = math.pi / 2
        _, ax, ay, _, _ = imu_from_plant(_state(theta=th, vy_w=0.001), _state(theta=th),
                                         0.001, 9.81)
        assert ax == pytest.approx(1.0)
        assert ay == pytest.approx(0.0, abs=1e-12)
        _, ax, ay, _, _ = imu_from_plant(_state(theta=th, vx_w=0.001), _state(theta=th),
                                         0.001, 9.81)
        assert ax == pytest.approx(0.0, abs=1e-12)
        assert ay == pytest.approx(-1.0)

    def test_parse_imu_static(self):
        assert parse_imu_static('0.5,0.2,-0.1,9.81,1.0') == (0.5, 0.2, -0.1, 9.81, 1.0)
        with pytest.raises(ValueError):
            parse_imu_static('1,2,3')


def _node(plant, **cfg):
    config = HILConfig(simulate_hw=True, max_steps=5, period_us=1_000, **cfg)
    return HILNode(plant, config=config, shared_state=mp.Array('d', 10))


class TestNodeImuSample:
    def test_static_values_and_seq_changes_every_frame(self):
        imu = (0.5, 0.2, -0.1, 9.81, 1.0)
        node = _node(MockPlant(), imu_static=imu)
        seqs = []
        for _ in range(3):
            got, seq = node._imu_sample(node.plant.get_state())
            assert got == imu
            seqs.append(seq)
        assert seqs == [1, 2, 3]
        # what the H7 decodes, within one LSB
        rec = decode_imu(encode_state([0.0] * 10, imu=imu, imu_seq=seqs[-1]))
        for r, v, rng in zip(rec[:5], imu, IMU_RANGES):
            assert abs(r - v) <= rng / 32767
        assert rec[5] == 3

    def test_seq_skips_zero_on_wrap(self):
        node = _node(MockPlant(), imu_static=(0.0, 0.0, 0.0, 9.81, 0.0))
        seqs = [node._imu_sample([0.0] * 10)[1] for _ in range(256)]
        assert 0 not in seqs
        assert seqs[254] == 255 and seqs[255] == 1

    def test_mock_plant_sends_no_imu(self):
        node = _node(MockPlant())
        assert node._imu_sample([0.0] * 10) == (None, 0)

    def test_mecanum_plant_at_rest(self):
        node = _node(MecanumPlant(PARAMS, dt=0.001))
        imu, seq = node._imu_sample(node.plant.get_state())
        assert imu == (0.0, 0.0, 0.0, PARAMS.g, 0.0)
        assert seq == 1
