"""Tests for the firmware-in-the-loop PC simulation (fwsim.dll + rpi5 plant).
Run: python -m pytest fwsim/tests  (needs fwsim/build/fwsim.dll: python fwsim/build_fwsim.py)"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')


@pytest.fixture(scope='module')
def fw():
    return fwsim.FirmwareSim()


def const(vx=0.0, vy=0.0, wz=0.0):
    return lambda t: (vx, vy, wz)


def test_motor_set_duty_seen_by_h7(fw):
    """motor_dir = -1, inverted compare and pin swap in motor.c: a positive command
    gives a negative rpwm - lpwm, which the plant (motor_polarity = -1) turns forward."""
    fw.reset()
    duty, _, pwm, _, en = fw.tick(10_000, [0, 0, 0, 0], (0.3, 0.0, 0.0))
    assert en == 1
    assert np.all(pwm > 0)
    np.testing.assert_allclose(duty, -pwm / 1000.0)


def test_pid_defaults_match_pid_c(fw):
    fw.reset()
    assert fw.get_pid(0) == pytest.approx(dict(kp=1.5, ki=0.05, kd=0.0, int_limit=0.5, out_limit=1.5))
    assert fw.get_pid(2) == pytest.approx(dict(kp=1.0, ki=0.05, kd=0.0, int_limit=1.0, out_limit=2.0))


def test_forward_step_moves_forward(fw):
    log = fwsim.simulate(fw, const(vx=0.3), 4.0)
    late = log['t'] > 3.0
    # Default pid.c gains leave a steady error (ki = 0.05, motor dead zone): about 0.27 m/s here
    assert 0.25 < np.mean(log['v_true'][late, 0]) <= 0.3
    assert np.all(np.abs(log['v_true'][late, 1:]) < 0.02)
    # firmware's encoder estimate agrees with the plant on average
    assert np.mean(log['v_fw'][late, 0]) == pytest.approx(np.mean(log['v_true'][late, 0]), abs=0.005)


def test_reproducible(fw):
    a = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 2.0, seed=3)
    b = fwsim.simulate(fw, const(vx=0.2, wz=0.5), 2.0, seed=3)
    np.testing.assert_array_equal(a['duty'], b['duty'])


def test_pid_gain_change_takes_effect(fw):
    a = fwsim.simulate(fw, const(vx=0.3), 1.0)
    b = fwsim.simulate(fw, const(vx=0.3), 1.0, pid={0: (4.0, 0.5, 0.0, 0.5, 1.5)})
    assert not np.allclose(a['duty'], b['duty'])
    fw.reset()
    assert fw.get_pid(0)['kp'] == pytest.approx(1.5)      # reset restores pid.c defaults


def test_pcnt_limit_spike(fw):
    """ESP-IDF PCNT resets to 0 at high_limit (32767); encoder.c corrects as if it
    wrapped by 65536, so one tick reads a huge speed. Seen in the real log
    05_DATA/AGV_real_logs/2026-06-10/2026-06-10_210603_AGV_log.csv (count 32716 -> 41).
    Plain int16 wrap (what encoder.c assumes) gives no spike."""
    sp = const(vx=1.0)     # ~20.6 rad/s: 32767 counts after ~7.6 s
    idf = fwsim.simulate(fw, sp, 9.0, pcnt_mode='idf')
    w16 = fwsim.simulate(fw, sp, 9.0, pcnt_mode='wrap16')
    assert np.max(np.abs(idf['v_fw'][:, 0] - idf['sp'][:, 0])) > 10.0
    assert np.max(np.abs(w16['v_fw'][100:, 0] - w16['sp'][100:, 0])) < 0.5
