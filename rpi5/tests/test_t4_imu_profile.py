"""t4_imu_profile.py (S06 T4 (4) IMU sequence) and run_hil.py --imu-profile t4 through HILNode."""
import json
import math
import multiprocessing as mp

import numpy as np
import pytest

import t4_imu_profile as p
from hil_node import HILConfig, HILNode
from plant_interface import MockPlant
from protocol import decode_imu, IMU_RANGES


def test_static_levels():
    seen = {}
    for i, (name, v) in enumerate(p.LEVELS):
        gz, ax, ay, az, yaw = p.sample(i * p.LEVEL_S + 1.0)
        got = {'gz': gz, 'ax': ax, 'ay': ay, 'yaw': yaw}
        assert got[name] == v and az == p.AZ
        assert all(got[k] == 0.0 for k in got if k != name)
        seen.setdefault(name, set()).add(v)
    assert seen == {'gz': {-5, -0.5, 0.5, 5}, 'ax': {-2, 2}, 'ay': {-2, 2}, 'yaw': {-2, 0, 2}}


def test_ramp_is_monotonic_and_yaw_is_its_integral():
    ts = np.arange(p.STATIC_S, p.CYCLE_S, 0.001)
    s = np.array([p.sample(t) for t in ts])
    assert np.all(np.diff(s[:, 0]) > 0) and abs(s[0, 0] + 5) < 1e-9 and abs(s[-1, 0] - 5) < 0.01
    assert np.all(np.diff(s[:, 1]) > 0) and abs(s[0, 1] + 2) < 1e-9 and abs(s[-1, 1] - 2) < 0.01
    # yaw step = wz dt, wrapped
    dyaw = np.array([(b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(s[:-1, 4], s[1:, 4])])
    assert np.allclose(dyaw, 0.5 * (s[:-1, 0] + s[1:, 0]) * 0.001, atol=1e-9)
    assert np.all(np.abs(s[:, 4]) <= math.pi)


def test_cycle_repeats():
    assert p.sample(3.3) == p.sample(3.3 + p.CYCLE_S)
    assert p.segment(0.5) == 'gz=-5' and p.segment(p.STATIC_S + 1) == 'ramp'


def test_hil_node_sends_the_profile(tmp_path):
    log = str(tmp_path / 'f.npz')
    cfg = HILConfig(simulate_hw=True, max_steps=3000, period_us=1_000, imu_profile='t4', frame_log=log)
    HILNode(MockPlant(), config=cfg, shared_state=mp.Array('d', 10)).run()
    z = np.load(log)
    steps = (0, 1500, 2999)
    for k in steps:
        got = decode_imu(bytes(z['mosi'][k]))[:5]
        want = p.sample(k * 0.001)
        for g, w, rng in zip(got, want, IMU_RANGES):
            assert abs(g - w) <= rng / 32767 / 2 + 1e-12, (k, got, want)
    assert json.loads(str(z['manifest']))['imu_source'] == 't4'


def test_manifest_names_the_source_really_used(tmp_path):
    # S73: imu_static has priority in _imu_sample, whatever imu_profile asks for
    log = str(tmp_path / 'f.npz')
    cfg = HILConfig(simulate_hw=True, max_steps=20, period_us=1_000, imu_profile='t4',
                    imu_static=(0.0, 0.0, 0.0, 9.81, 0.0), frame_log=log)
    HILNode(MockPlant(), config=cfg, shared_state=mp.Array('d', 10)).run()
    assert json.loads(str(np.load(log)['manifest']))['imu_source'] == 'static'
