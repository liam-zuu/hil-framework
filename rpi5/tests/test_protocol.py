"""
Tests: protocol.py — SPI encode/decode must match H7 hil_protocol.h

MOSI field order: [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz] + traj ref + IMU (v3, 40 bytes)
  index:           0    1    2    3   4  5  6   7   8   9

Run: pytest tests/test_protocol.py -v
"""
import struct
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import re
import struct
import numpy as np
from protocol import (
    encode_state, decode_state, decode_traj_ref, decode_imu, IMU_RANGES,
    encode_torques, decode_torques, decode_output,
    _to_int16, _from_int16,
    STATE_RANGES, REF_RANGES, TORQUE_RANGE, FRAME_SIZE, N_STATES, N_TORQUES,
    LSB_TORQUE, TRAJ_VALID, TRAJ_NEW,
)

H7_HEADER = os.path.join(os.path.dirname(__file__), '..', '..',
                         'stm32h7', 'Core', 'Inc', 'hil_protocol.h')


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

    def test_reserved_bytes_are_zero(self):
        frame = encode_state([0.0] * 10, traj_ref=(1.0, -1.0, 2.0),
                             traj_flags=TRAJ_VALID, seq=7, fault_flags=1)
        # without imu, the v3 IMU bytes (22-23, 31-39) stay zero
        assert frame[22:24] == b'\x00\x00'
        assert frame[31:40] == b'\x00' * 9

    def test_imu_roundtrip_and_offsets(self):
        imu = (1.25, -0.5, 0.3, 9.80665, -2.0)
        frame = encode_state([0.0] * 10, imu=imu, imu_seq=0x1A5)
        rec = decode_imu(frame)
        for got, want, rng in zip(rec[:5], imu, IMU_RANGES):
            assert abs(got - want) <= rng / 32767 + 1e-12
        assert rec[5] == 0xA5                       # stored modulo 256
        # offsets as in hil_protocol.h v3: gz 22, ax 31, ay 33, az 35, yaw 37, seq 39
        assert struct.unpack_from('>h', frame, 22)[0] == round(1.25 / IMU_RANGES[0] * 32767)
        assert struct.unpack_from('>h', frame, 35)[0] == round(9.80665 / IMU_RANGES[3] * 32767)
        assert struct.unpack_from('>h', frame, 37)[0] == round(-2.0 / np.pi * 32767)
        assert frame[39] == 0xA5

    def test_imu_does_not_touch_other_fields(self):
        state = [1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 1.0, 0.2, -0.2, 2.0]
        a = encode_state(state, traj_ref=(1.0, 0.5, -1.0), traj_flags=TRAJ_VALID, seq=9, fault_flags=1)
        b = encode_state(state, traj_ref=(1.0, 0.5, -1.0), traj_flags=TRAJ_VALID, seq=9, fault_flags=1,
                         imu=(30.0, 39.0, -39.0, 9.8, 3.1), imu_seq=255)
        same = [i for i in range(FRAME_SIZE) if i not in (22, 23) and not 31 <= i <= 39]
        assert [a[i] for i in same] == [b[i] for i in same]

    def test_imu_wrong_length_raises(self):
        with pytest.raises(ValueError):
            encode_state([0.0] * 10, imu=(0.0, 0.0, 0.0))

    def test_seq_and_fault_bytes(self):
        frame = encode_state([0.0] * 10, seq=0x1FF, fault_flags=0x02)
        assert frame[20] == 0x02
        assert frame[21] == 0xFF   # seq stored modulo 256

    def test_big_endian_byte_order(self):
        """Critical: H7 SPI DMA reads big-endian int16.
        Layout: index 0 = omega[0], range = HIL_OMEGA_MAX = 40.0 rad/s.
        Full-scale omega[0] = 40.0 → raw int16 = 32767 at byte 0-1.
        """
        state = [0.0] * 10
        state[0] = STATE_RANGES[0]   # omega[0] = 40.0 rad/s → raw = 32767
        frame = encode_state(state)
        raw   = struct.unpack_from('>h', frame, 0)[0]
        assert raw == 32767

    def test_theta_range_pi(self):
        """Heading uses ±π range. theta is at index 6 in new layout."""
        state = [0.0] * 10
        state[6] = math.pi - 0.001   # index 6 = theta, just inside range
        rec = decode_state(encode_state(state))
        assert abs(rec[6] - state[6]) <= math.pi / 32767 + 1e-12

    def test_wrong_length_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            encode_state([0.0] * 9)

    def test_typical_moving_state(self):
        """Representative HIL state: forward 0.5 m/s, slight curve.
        Layout: [ω1, ω2, ω3, ω4, x, y, θ, vx, vy, wz]
        """
        state = [10.3, 10.3, 10.3, 10.3, 1.5, 0.1, 0.05, 0.5, 0.02, 0.1]
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

    def test_unused_bytes_zero(self):
        frame = encode_torques([0.0] * 4)
        # torques occupy bytes 0-7; pwm/status/seq default 0; bytes 18-35 reserved
        assert frame[8:] == b'\x00' * (FRAME_SIZE - 8)

    def test_full_output_round_trip(self):
        frame = encode_torques([0.3, -0.2, 0.1, 0.0], pwm=[0.5, -0.5, 1.0, 0.0],
                               status=3, seq=200)
        out = decode_output(frame)
        assert out['torque'] == pytest.approx([0.3, -0.2, 0.1, 0.0], abs=1e-4)
        assert out['pwm'] == pytest.approx([0.5, -0.5, 1.0, 0.0], abs=1e-4)
        assert out['status'] == 3
        assert out['seq'] == 200
        assert frame[18:] == b'\x00' * (FRAME_SIZE - 18)

    def test_lsb_value(self):
        """LSB ≈ 3.05e-5 N·m — matches HIL_TORQUE_MAX/32767."""
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


# ── Trajectory reference (MOSI bytes 24-30) ────────────────────────────────

class TestTrajRef:
    def test_default_is_zero_and_not_valid(self):
        frame = encode_state([0.0] * 10)
        assert decode_traj_ref(frame) == (0.0, 0.0, 0.0, 0)

    def test_round_trip(self):
        ref   = (0.3, -0.15, 0.6)
        frame = encode_state([0.0] * 10, traj_ref=ref, traj_flags=TRAJ_VALID | TRAJ_NEW)
        vx, vy, wz, flags = decode_traj_ref(frame)
        for orig, r, rng in zip(ref, (vx, vy, wz), REF_RANGES):
            assert abs(r - orig) <= rng / 32767 + 1e-12
        assert flags == TRAJ_VALID | TRAJ_NEW

    def test_byte_offsets_match_h7_decode(self):
        """H7 _decode(): vx_ref at bytes 24-25, wz_ref at 28-29, flags at 30."""
        frame = encode_state([0.0] * 10, traj_ref=(3.0, 0.0, -10.0), traj_flags=TRAJ_VALID)
        assert struct.unpack_from('>h', frame, 24)[0] == 32767
        assert struct.unpack_from('>h', frame, 28)[0] == -32767
        assert frame[30] == TRAJ_VALID

    def test_state_unchanged_by_traj_ref(self):
        state = [10.0, -10.0, 5.0, -5.0, 1.0, -1.0, 0.5, 0.3, -0.3, 1.0]
        a = encode_state(state)
        b = encode_state(state, traj_ref=(1.0, 1.0, 1.0), traj_flags=TRAJ_VALID)
        assert a[:20] == b[:20]

    def test_wrong_ref_length_raises(self):
        with pytest.raises(ValueError):
            encode_state([0.0] * 10, traj_ref=(0.0, 0.0))


# ── Consistency with the H7 header ─────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(H7_HEADER), reason="stm32h7 sources not present")
class TestMatchesH7Header:
    """Fails when hil_protocol.h and protocol.py drift apart."""

    @staticmethod
    def _define(name: str) -> float:
        with open(H7_HEADER, encoding='utf-8') as f:
            text = f.read()
        m = re.search(rf'#define\s+{name}\s+([0-9.]+)f?', text)
        assert m, f"{name} not found in hil_protocol.h"
        return float(m.group(1))

    def test_frame_len(self):
        assert self._define('SPI1_FRAME_LEN') == FRAME_SIZE

    def test_state_ranges(self):
        expected = ([self._define('HIL_OMEGA_MAX')] * 4
                    + [self._define('HIL_POS_MAX')] * 2
                    + [self._define('HIL_THETA_MAX')]
                    + [self._define('HIL_VEL_MAX')] * 2
                    + [self._define('HIL_OMEGA_BODY_MAX')])
        assert STATE_RANGES == pytest.approx(expected, rel=1e-6)

    def test_ref_and_torque_ranges(self):
        assert REF_RANGES == pytest.approx([self._define('HIL_REF_VEL_MAX')] * 2
                                           + [self._define('HIL_REF_OMEGA_MAX')])
        assert TORQUE_RANGE == pytest.approx(self._define('HIL_TORQUE_MAX'))

    def test_imu_ranges(self):
        assert IMU_RANGES == pytest.approx([self._define('HIL_IMU_GYRO_MAX')]
                                           + [self._define('HIL_IMU_ACCEL_MAX')] * 3
                                           + [self._define('HIL_IMU_YAW_MAX')], rel=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])