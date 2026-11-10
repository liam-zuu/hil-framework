"""Tests for imu_model.ImuNoise (replay of the measured BNO085 noise, Liam 10/11 "Phát lại
nhiễu đo", L72) and its use on the HIL Pi (hil_node._imu_sample, frame to the H7).
Codex round 1: IMN-3/-5 (replay instead of the Gaussian model), IMN-10 (tests against the
measured data, correlation at the report interval after the 1 ms Pi calls)."""
import importlib.util
import math
import multiprocessing as mp
import os

import numpy as np
import pytest

from params import PARAMS
from mecanum_plant import MecanumPlant
from hil_node import HILConfig, HILNode
from imu_model import (ImuNoise, load_replay, replay_path, imu_from_plant, GYRO_STEP_RADPS,
                       ACCEL_STEP_MPS2, YAW_STEP_RAD)
from protocol import encode_state, decode_imu

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', '..', 'tools', 'make_imu_noise_replay.py')
RATES = (20.0, 100.0)


def _raw_csv(rate):
    """The raw log the replay file was made from, if this machine has it (05_DATA of the
    thesis repo; IMU_NOISE_ROOT overrides the root)."""
    label = load_replay(rate).header['source']                     # 05_DATA/results/IMU_NOISE/...
    roots = [os.environ.get('IMU_NOISE_ROOT', ''), r'D:\CLAUDE_THESIS',
             os.path.join(HERE, '..', '..', '..', '..')]
    for r in roots:
        p = os.path.join(r, *label.split('/')) if r else ''
        if p and os.path.isfile(p):
            return p
    return None


def _tool():
    spec = importlib.util.spec_from_file_location('make_imu_noise_replay', TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _acf(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-lag] * x[lag:]).mean() / x.var())


def _is_multiple(v, q):
    return abs(v / q - round(v / q)) < 1e-9


def _rest_series(rate, seed, n_reports, pi_calls=False):
    """(gz, yaw) the model gives at rest for n_reports reports; pi_calls: one call per 1 ms
    frame as on the HIL Pi, then the value at each report instant (what the H7 takes)."""
    n = ImuNoise(rate, seed=seed)
    s = (0.0, 0.0, 0.0, PARAMS.g, 0.0)
    if not pi_calls:
        out = [n.apply(s, m * n.period_us) for m in range(n_reports)]
    else:
        frames = [n.apply(s, k * 1000) for k in range(n_reports * n.period_us // 1000)]
        out = [frames[m * n.period_us // 1000] for m in range(n_reports)]
    return np.array([o[0] for o in out]), np.array([o[4] for o in out])


class TestReplayFile:
    @pytest.mark.parametrize('rate', RATES)
    def test_header_and_rows(self, rate):
        r = load_replay(rate)
        assert int(r.header['n']) == len(r.k_gz) == len(r.k_yaw) > 5000
        assert float(r.header['rate_hz']) == rate
        assert float(r.header['gyro_step']) == GYRO_STEP_RADPS
        assert float(r.header['yaw_step_rad']) == YAW_STEP_RAD
        assert r.header['source'].endswith('/raw_static.csv') and len(r.header['source_sha256']) == 64
        assert os.path.basename(replay_path(rate)) == f'imu_noise_replay_{int(rate)}hz.csv'

    def test_unmeasured_rate_refused(self):
        with pytest.raises(ValueError):
            load_replay(50.0)
        with pytest.raises(ValueError):
            ImuNoise(50.0, seed=0)

    @pytest.mark.parametrize('rate', RATES)
    def test_file_is_the_tool_output_of_the_raw_log(self, rate):
        raw = _raw_csv(rate)
        if raw is None:
            pytest.skip('raw IMU_NOISE log not on this machine')
        t = _tool()
        r = load_replay(rate)
        assert t.sha256(raw) == r.header['source_sha256']
        info, rows = t.build(raw, rate)
        assert [x[1] for x in rows] == r.k_gz.tolist() and [x[2] for x in rows] == r.k_yaw.tolist()

    @pytest.mark.parametrize('rate', RATES)
    def test_replay_keeps_the_measured_statistics(self, rate):
        """Against the raw CSV itself (not against constants): zero share, tails and clusters
        of gz (IMN-3), std and correlation at lags 1 and 100 reports of yaw (IMN-5)."""
        raw = _raw_csv(rate)
        if raw is None:
            pytest.skip('raw IMU_NOISE log not on this machine')
        ns, g, y, *_ = _tool().load_pairs(raw)
        gz_m = np.round(np.array([g[n] for n in ns]) / GYRO_STEP_RADPS) * GYRO_STEP_RADPS   # log: 6 decimals
        yaw_m = np.radians([y[n] for n in ns])
        n = len(ns)
        gz, yaw = _rest_series(rate, seed=0, n_reports=n)     # one full lap of the record
        assert np.sort(gz).tolist() == np.sort(gz_m).tolist()                     # same gz values
        assert np.sum(np.abs(gz) >= 2 * GYRO_STEP_RADPS) == np.sum(np.abs(gz_m) >= 2 * GYRO_STEP_RADPS)
        pairs = lambda v: int(np.sum((v[:-1] != 0) & (v[1:] != 0)))
        assert abs(pairs(gz) - pairs(gz_m)) <= 1                                  # wrap joins one pair
        assert np.std(yaw) == pytest.approx(np.std(yaw_m), rel=0.03)              # log has 0.001 deg print
        for lag in (1, 100):
            assert _acf(yaw, lag) == pytest.approx(_acf(yaw_m, lag), abs=0.02)


class TestNoise:
    def test_output_steps(self):
        n = ImuNoise(20.0, seed=1)
        for i in range(500):
            truth = (0.3 * math.sin(i), 0.12, -0.05, PARAMS.g, 0.1)
            gz, ax, ay, az, yaw = n.apply(truth, i * n.period_us)
            row = n.row(i * n.period_us)
            assert _is_multiple(gz, GYRO_STEP_RADPS)
            assert gz == pytest.approx(round(truth[0] / GYRO_STEP_RADPS) * GYRO_STEP_RADPS
                                       + n.replay.k_gz[row] * GYRO_STEP_RADPS, abs=1e-12)
            for a in (ax, ay, az):
                assert _is_multiple(a, ACCEL_STEP_MPS2)
            assert yaw == pytest.approx(0.1 + n.replay.k_yaw[row] * YAW_STEP_RAD, abs=1e-12)

    def test_seed_sets_the_start_and_is_reproducible(self):
        s = (0.2, 0.1, 0.0, PARAMS.g, 0.5)
        def run(seed):
            n = ImuNoise(100.0, seed=seed)
            return [n.apply(s, m * n.period_us) for m in range(3000)]
        assert run(7) == run(7) and run(7) != run(8)
        assert ImuNoise(100.0, seed=7).start == int(np.random.default_rng(7).integers(len(load_replay(100.0).k_gz)))

    def test_rows_follow_the_report_index_and_wrap(self):
        n = ImuNoise(20.0, seed=3)
        N = len(n.replay.k_gz)
        assert n.row(0) == n.start and n.row(49_999) == n.start
        assert n.row(50_000) == (n.start + 1) % N
        assert n.row(N * 50_000) == n.start                                       # wraps round

    def test_yaw_wraps(self):
        n = ImuNoise(20.0, seed=0)
        ys = [n.apply((0.0, 0.0, 0.0, PARAMS.g, math.pi - 1e-5), m * n.period_us)[4] for m in range(3000)]
        assert all(-math.pi <= v < math.pi for v in ys)
        assert any(v < 0 for v in ys) and any(v > 0 for v in ys)

    @pytest.mark.parametrize('rate', RATES)
    def test_pi_calls_give_the_fwsim_sequence(self, rate):
        """1 ms calls (HIL Pi) sampled at the report instants m * period give the per-report
        sequence (fwsim), so the report-interval statistics do not depend on dt (IMN-10). Only
        when the samples are taken at those instants: the real H7 schedule has its own phase,
        a fixed row shift on HIL (IMN2-1, DEFERRED, see ImuNoise)."""
        n_rep = int(60 * rate)
        a = _rest_series(rate, seed=4, n_reports=n_rep)
        b = _rest_series(rate, seed=4, n_reports=n_rep, pi_calls=True)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_value_held_within_a_report_period(self):
        n = ImuNoise(20.0, seed=5)
        s = (0.0, 0.0, 0.0, PARAMS.g, 0.0)
        for m in range(20):
            vals = {n.apply(s, m * 50_000 + k * 1000) for k in range(50)}
            assert len(vals) == 1


def _node(plant, **cfg):
    config = HILConfig(simulate_hw=True, max_steps=5, period_us=1_000, **cfg)
    return HILNode(plant, config=config, shared_state=mp.Array('d', 10))


class TestHilNode:
    def test_plant_values_get_the_replay(self):
        noise = ImuNoise(20.0, seed=4)
        node = _node(MecanumPlant(PARAMS, dt=0.001), imu_noise=noise)
        st = node.plant.get_state()
        imu, seq = node._imu_sample(st)
        assert seq == 1
        truth = imu_from_plant(st, None, 0.001, PARAMS.g)
        assert imu == noise.apply(truth, 0)
        assert _is_multiple(imu[0], GYRO_STEP_RADPS)
        assert _is_multiple(imu[3], ACCEL_STEP_MPS2) and abs(imu[3] - PARAMS.g) <= ACCEL_STEP_MPS2 / 2

    def test_no_noise_is_the_plant_value(self):
        node = _node(MecanumPlant(PARAMS, dt=0.001))
        st = node.plant.get_state()
        imu, _ = node._imu_sample(st)
        assert imu == imu_from_plant(st, None, 0.001, PARAMS.g)

    def test_static_values_are_not_noised(self):
        st = (0.5, 0.2, -0.1, 9.81, 1.0)
        node = _node(MecanumPlant(PARAMS, dt=0.001), imu_static=st, imu_noise=ImuNoise(20.0, seed=4))
        assert node._imu_sample(node.plant.get_state())[0] == st

    def test_gyro_step_survives_the_frame(self):
        # frame LSB of gz is 34.9/32767 = 0.00107 rad/s < GYRO_STEP/2, so the H7 rounds back to the
        # same multiple of 2^-9; the whole chain to the firmware: bnoemu_pc/tests/test_firmware_view.py
        n = ImuNoise(20.0, seed=5)
        for i in range(300):
            imu = n.apply((0.4 * math.sin(0.1 * i), 0.0, 0.0, PARAMS.g, 0.0), i * n.period_us)
            rec = decode_imu(encode_state([0.0] * 10, imu=imu, imu_seq=1))
            assert round(rec[0] / GYRO_STEP_RADPS) == round(imu[0] / GYRO_STEP_RADPS)
            assert round(rec[3] / ACCEL_STEP_MPS2) == round(imu[3] / ACCEL_STEP_MPS2)
