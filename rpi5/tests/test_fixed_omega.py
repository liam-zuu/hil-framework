"""run_hil.py --fixed-omega: constant wheel speeds reach the SPI frame unchanged (EncoderGen check)."""
import pytest

from plant_interface import MockPlant
from protocol import encode_state, decode_state
from run_hil import _parse_fixed_omega


def test_parse_fixed_omega():
    assert _parse_fixed_omega('10,20,-10,-20') == [10.0, 20.0, -10.0, -20.0]
    with pytest.raises(ValueError):
        _parse_fixed_omega('10,20')


def test_fixed_omega_frame_carries_wheel_speeds():
    w = [10.0, 20.0, -10.0, -20.0]
    plant = MockPlant()
    plant.reset(w + [0.0] * 6)
    state = plant.step([0.0] * 4)          # MockPlant ignores torques
    got = decode_state(encode_state(state))
    assert got[:4] == pytest.approx(w, abs=40.0 / 32767 * 1.5)
    assert got[4:] == pytest.approx([0.0] * 6, abs=1e-9)


# --omega-steps (bug H7-ENC-RETIME, 28/10): speed changes inside one run
from plant_interface import OmegaProfilePlant      # noqa: E402
from run_hil import _parse_omega_steps             # noqa: E402


def test_parse_omega_steps():
    segs = _parse_omega_steps('2,2,2,2:20/~39.2,-39.2,39.2,-39.2:0.3/0,0,0,0:5')
    assert segs == [([2.0] * 4, 20.0, False), ([39.2, -39.2, 39.2, -39.2], 0.3, True), ([0.0] * 4, 5.0, False)]
    for bad in ('2,2,2,2', '2,2,2:1', '2,2,2,2:0', ''):
        with pytest.raises(ValueError):
            _parse_omega_steps(bad)


def test_omega_profile_hold_ramp_then_zero():
    p = OmegaProfilePlant([([2.0, -2.0, 2.0, -2.0], 0.003, False), ([10.0, -10.0, 10.0, -10.0], 0.004, True)])
    assert p.total_steps == 7
    w = [p.step([0.0] * 4)[:4] for _ in range(9)]
    assert w[0] == w[2] == [2.0, -2.0, 2.0, -2.0]
    assert w[3] == pytest.approx([4.0, -4.0, 4.0, -4.0])           # 1/4 of the way from 2 to 10
    assert w[6] == pytest.approx([10.0, -10.0, 10.0, -10.0])       # last ramp step reaches the target
    assert w[7] == w[8] == [0.0] * 4                               # after the profile: stopped


def test_omega_profile_frame_carries_wheel_speeds():
    p = OmegaProfilePlant([([39.2, -39.2, 12.5, -0.4], 0.001, False)])
    got = decode_state(encode_state(p.step([0.0] * 4)))
    assert got[:4] == pytest.approx([39.2, -39.2, 12.5, -0.4], abs=40.0 / 32767 * 1.5)
