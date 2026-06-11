"""
Tests: protocol.py — SPI encode/decode must match MATLAB spi_interface.m

Run: pytest tests/test_protocol.py -v
"""
import struct
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from protocol import (
    encode_state, decode_state,
    encode_torques, decode_torques,
    _to_int16, _from_int16,
    STATE_RANGES, TORQUE_RANGE, FRAME_SIZE, N_STATES, N_TORQUES,
    LSB_TORQUE,
)


# ── int16 helpers ──────────────────────────────────────────────────────────

class TestInt16:
    def test_zero_is_zero(self):
        assert _to_int16(0.0, 5.0) == 0

    def test_positive_full_scale(self):
        assert _to_int16(5.0, 5.0) == 32767

    def test_negative_full_scale(self):
        assert _to_int16(-5.0, 5.0) == -32767

    def test_clamp_above_range(self):
        assert _to_int16(100.0, 5.0) == 32767

    def test_clamp_below_range(self):
        assert _to_int16(-100.0, 5.0) == -32767

    def test_round_trip_within_1_lsb(self):
        for val in [1.23, -2.71, 0.001, 4.999]:
            raw = _to_int16(val, 5.0)
            rec = _from_int16(raw, 5.0)
            assert abs(rec - val) <= 5.0 / 32767 + 1e-12

    def test_midpoint(self):
        raw = _to_int16(2.5, 5.0)
        assert raw == round(2.5 / 5.0 * 32767)


# ── State encoding ─────────────────────────────────────────────────────────

class TestStateEncoding:
    def test_frame_size(self):
        assert len(encode_state([0.0] * 10)) == FRAME_SIZE

    def test_zeros_round_trip(self):
        state = [0.0] * N_STATES
        assert decode_state(encode_state(state)) == pytest.approx(state, abs=1e-4)

    def test_positive_full_scale_round_trip(self):
        state = list(STATE_RANGES)
        rec   = decode_state(encode_state(state))
        for orig, r, rng in zip(state, rec, STATE_RANGES):
            assert abs(r - orig) <= rng / 32767 + 1e-12

    def test_negative_full_scale_round_trip(self):
        state = [-r for r in STATE_RANGES]
        rec   = decode_state(encode_state(state))
        for orig, r, rng in zip(state, rec, STATE_RANGES):
            assert abs(r - orig) <= rng / 32767 + 1e-12

    def test_clamping_over_range(self):
        state = [rng * 2 for rng in STATE_RANGES]
        rec   = decode_state(encode_state(state))
        for r, rng in zip(rec, STATE_RANGES):
            assert abs(abs(r) - rng) <= rng / 32767 + 1e-12

    def test_padding_bytes_are_zero(self):
        frame = encode_state([0.0] * 10)
        # bytes 20-23 are padding (states use bytes 0-19)
        assert frame[20:24] == b'\x00\x00\x00\x00'

    def test_big_endian_byte_order(self):
        """Critical: H7 SPI DMA reads big-endian int16."""
        state = [0.0] * 10
        state[0] = STATE_RANGES[0]   # x = +5 m → raw = 32767
        frame = encode_state(state)
        raw   = struct.unpack_from('>h', frame, 0)[0]
        assert raw == 32767

    def test_theta_range_pi(self):
        """Heading uses ±π range."""
        import math
        state = [0.0] * 10
        state[2] = math.pi - 0.001   # just inside range
        rec = decode_state(encode_state(state))
        assert abs(rec[2] - state[2]) <= math.pi / 32767 + 1e-12

    def test_wrong_length_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            encode_state([0.0] * 9)

    def test_typical_moving_state(self):
        """Representative HIL state: forward 0.5 m/s, slight curve."""
        state = [1.5, 0.1, 0.05, 0.5, 0.02, 0.1, 10.3, 10.3, 10.3, 10.3]
        rec   = decode_state(encode_state(state))
        for orig, r, rng in zip(state, rec, STATE_RANGES):
            assert abs(r - orig) <= rng / 32767 * 1.1


# ── Torque encoding ────────────────────────────────────────────────────────

class TestTorqueEncoding:
    def test_frame_size(self):
        assert len(encode_torques([0.0] * 4)) == FRAME_SIZE

    def test_zeros(self):
        rec = decode_torques(encode_torques([0.0] * 4))
        assert rec == pytest.approx([0.0] * 4, abs=1e-4)

    def test_round_trip_nominal(self):
        torques = [0.3, -0.2, 0.1, -0.15]
        rec     = decode_torques(encode_torques(torques))
        for orig, r in zip(torques, rec):
            assert abs(r - orig) <= TORQUE_RANGE / 32767 + 1e-12

    def test_clamp_over_range(self):
        torques = [5.0, -5.0, 5.0, -5.0]
        rec     = decode_torques(encode_torques(torques))
        for r in rec:
            assert abs(abs(r) - TORQUE_RANGE) <= TORQUE_RANGE / 32767 + 1e-12

    def test_padding_16_bytes(self):
        frame = encode_torques([0.0] * 4)
        # torques occupy bytes 0-7, padding = bytes 8-23
        assert frame[8:] == b'\x00' * 16

    def test_lsb_value(self):
        """LSB ≈ 3.05e-5 N·m — matches MATLAB spi_interface.m validation."""
        assert abs(LSB_TORQUE - 3.05e-5) < 1e-7

    def test_wrong_length_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            encode_torques([0.0] * 3)


# ── Protocol-level round-trip ──────────────────────────────────────────────

class TestProtocolRoundTrip:
    """Simulate a full HIL tick: encode state → decode state (loopback)."""

    def test_full_loopback_10_states(self):
        """In SPI loopback (MOSI shorted to MISO), decode_state(encode_state) ≈ identity."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            state = [rng.uniform(-r * 0.9, r * 0.9) for r in STATE_RANGES]
            mosi  = encode_state(state)
            # loopback: MISO = MOSI
            miso  = mosi
            rec   = decode_state(miso)
            for orig, r, rng_ in zip(state, rec, STATE_RANGES):
                assert abs(r - orig) <= rng_ / 32767 + 1e-12


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
