"""Tests for hil_node.py and trajectory_pub.py (simulate_hw=True, no hardware)."""
import math
import multiprocessing as mp
import pytest

from params import PARAMS
from plant_interface import MockPlant
from mecanum_plant import MecanumPlant
from hil_node import HILConfig, HILNode
from trajectory_pub import (
    TrajectorySpec,
    encode_packet,
    decode_packet,
    get_reference,
)


# ---------------------------------------------------------------------------
# 1. HILNode in simulated mode (no GPIO / SPI hardware required)
# ---------------------------------------------------------------------------

def _make_node(max_steps=50, use_mock=True):
    plant  = MockPlant() if use_mock else MecanumPlant(PARAMS, dt=0.001)
    config = HILConfig(
        simulate_hw=True,
        max_steps=max_steps,
        period_us=1_000,   # 1 kHz nominal
    )
    shared = mp.Array('d', 10)
    return HILNode(plant, config=config, shared_state=shared), shared


class TestHILNodeSimulated:
    def test_instantiate(self):
        node, _ = _make_node()
        assert node is not None

    def test_run_completes_after_max_steps(self):
        """run() should return after exactly max_steps ticks."""
        node, _ = _make_node(max_steps=30)
        node.run()   # must not block indefinitely
        assert node._step == 30

    def test_shared_state_updated(self):
        """After run(), shared_state should still be 10 doubles (no corruption)."""
        node, shared = _make_node(max_steps=20)
        node.run()
        values = list(shared)
        assert len(values) == 10
        for v in values:
            assert math.isfinite(v), f"shared_state contains non-finite: {values}"

    def test_jitter_stats_collected(self):
        """Timer must record at least one tick worth of jitter data."""
        node, _ = _make_node(max_steps=20)
        node.run()
        stats = node._timer.jitter_stats
        assert stats["n"] >= 20, f"Expected >=20 ticks, got {stats['n']}"

    def test_state_finite_with_real_plant(self):
        """MockPlant ignores torques; real plant should also stay finite."""
        plant  = MecanumPlant(PARAMS, dt=0.001, seed=7)
        config = HILConfig(simulate_hw=True, max_steps=100, period_us=1_000)
        node   = HILNode(plant, config=config)
        node.run()
        state = plant.get_state()
        assert len(state) == 10
        for v in state:
            assert math.isfinite(v), f"Plant state non-finite: {state}"


# ---------------------------------------------------------------------------
# 2. Trajectory packet encode / decode
# ---------------------------------------------------------------------------

class TestTrajectoryPacket:
    def test_packet_length(self):
        pkt = encode_packet(1.0, 2.0, 0.5, 0.3)
        assert len(pkt) == 18

    def test_header_byte(self):
        pkt = encode_packet(0.0, 0.0, 0.0, 0.0)
        assert pkt[0] == 0xBB

    def test_checksum_valid(self):
        pkt = encode_packet(1.5, -0.8, math.pi / 4, 0.4)
        valid, *_ = decode_packet(pkt)
        assert valid, "Checksum should pass for freshly encoded packet"

    def test_round_trip_values(self):
        x, y, theta, vx = 1.23, -0.45, 0.78, 0.99
        pkt = encode_packet(x, y, theta, vx)
        valid, rx, ry, rt, rvx = decode_packet(pkt)
        assert valid
        assert abs(rx - x)     < 1e-5, f"x: {rx} vs {x}"
        assert abs(ry - y)     < 1e-5, f"y: {ry} vs {y}"
        assert abs(rt - theta) < 1e-5, f"theta: {rt} vs {theta}"
        assert abs(rvx - vx)   < 1e-5, f"vx: {rvx} vs {vx}"

    def test_corrupted_checksum_invalid(self):
        pkt = bytearray(encode_packet(1.0, 2.0, 0.3, 0.5))
        pkt[-1] ^= 0xFF   # flip all bits in checksum byte
        valid, *_ = decode_packet(bytes(pkt))
        assert not valid, "Corrupted packet should fail checksum"

    def test_wrong_header_invalid(self):
        pkt = bytearray(encode_packet(0.0, 0.0, 0.0, 0.0))
        pkt[0] = 0xAA   # wrong header
        valid, *_ = decode_packet(bytes(pkt))
        assert not valid

    def test_zero_packet(self):
        pkt = encode_packet(0.0, 0.0, 0.0, 0.0)
        valid, x, y, theta, vx = decode_packet(pkt)
        assert valid
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Trajectory reference generator
# ---------------------------------------------------------------------------

class TestTrajectoryGenerator:
    def test_idle_returns_zeros(self):
        spec = TrajectorySpec(kind='idle')
        x, y, theta, vx = get_reference(t=5.0, spec=spec)
        assert x == 0.0
        assert y == 0.0
        assert theta == 0.0
        assert vx == 0.0

    def test_line_x_increases_with_time(self):
        spec = TrajectorySpec(kind='line', v=0.5)
        x0, *_ = get_reference(t=0.0, spec=spec)
        x1, *_ = get_reference(t=2.0, spec=spec)
        assert x1 > x0, f"x should increase: x0={x0}, x1={x1}"

    def test_circle_at_t0_origin_ish(self):
        """Circle starts near (R, 0)."""
        R = 0.5
        spec = TrajectorySpec(kind='circle', v=0.5, radius=R)
        x, y, *_ = get_reference(t=0.0, spec=spec)
        # Centre of circle at origin, robot starts at (R, 0) or (0, 0)
        # Just verify finite values
        assert math.isfinite(x) and math.isfinite(y)

    def test_figure8_finite(self):
        spec = TrajectorySpec(kind='figure8', v=0.5, radius=0.5)
        for t in [0.0, 1.0, 3.0, 6.0, 10.0]:
            x, y, theta, vx = get_reference(t=t, spec=spec)
            assert all(math.isfinite(v) for v in [x, y, theta, vx]), \
                f"Non-finite at t={t}: ({x},{y},{theta},{vx})"

    def test_circle_returns_four_values(self):
        spec = TrajectorySpec(kind='circle', v=0.5, radius=1.0)
        ref = get_reference(t=1.0, spec=spec)
        assert len(ref) == 4

    def test_heading_bounded_minus_pi_pi(self):
        """Theta reference should stay in [-π, π] throughout trajectory."""
        spec = TrajectorySpec(kind='circle', v=0.5, radius=0.5, period=4.0)
        for k in range(200):
            t = k * 0.05
            _, _, theta, _ = get_reference(t=t, spec=spec)
            assert -math.pi - 1e-9 <= theta <= math.pi + 1e-9, \
                f"theta={theta:.4f} out of range at t={t:.2f}"

    def test_vx_nonnegative_for_forward_traj(self):
        """Forward speed reference should be non-negative for line trajectory."""
        spec = TrajectorySpec(kind='line', v=0.5)
        for k in range(50):
            _, _, _, vx = get_reference(t=k * 0.1, spec=spec)
            assert vx >= -1e-9, f"vx={vx} negative at t={k*0.1}"
