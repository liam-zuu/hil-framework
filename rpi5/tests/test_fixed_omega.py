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
