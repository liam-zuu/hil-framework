"""Tests of fwsim/sens_tre_slip.py (task C2): patched_simulate puts io_delay_ms and
slip_omega_s into the run while s03_doe.run_one stays the evaluation unchanged.
DLL: ket_qua_c2/build/fwsim.dll, fallback fwsim/build/fwsim.dll (skip if neither).
Run: python -m pytest -q fwsim/tests/test_sens_tre_slip.py"""
import argparse
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim                                            # noqa: E402
import o1_feasibility as o1                             # noqa: E402
import s03_doe as s03                                   # noqa: E402
import sens_tre_slip as sens                            # noqa: E402

_DLL = os.path.join(fwsim.HERE, '..', 'ket_qua_c2', 'build', 'fwsim.dll')
if not os.path.isfile(_DLL):
    _DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')


@pytest.fixture(scope='module')
def path():
    return o1.Path8()


@pytest.fixture(scope='module')
def fw():
    f = fwsim.FirmwareSim(_DLL)
    if not f.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    return f


def _same(a, b):
    """Bit-exact equality of two row values (NaN == NaN; repr for floats)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return repr(a) == repr(b)


@needs_dll
@pytest.mark.parametrize('ctrl', ('PID', 'ADRC'))
def test_ref_equals_direct(ctrl, fw, path):
    """The reference combo (delay 0, omega_s 0.1 = params.py default) through run_case is
    s03_doe.run_one bit for bit: same row values, same log arrays; fwsim.simulate restored."""
    orig = fwsim.simulate
    row1, log1 = sens.run_case(fw, path, ctrl, 0.8, 0.436, 0, 0.1)
    assert fwsim.simulate is orig
    row2, log2 = s03.run_one(fw, path, ctrl, s03.defaults(fw, ctrl), 0.8, 0.436)
    for k in row2:
        assert _same(row1[k], row2[k]), f'{k}: {row1[k]!r} != {row2[k]!r}'
    assert set(log1) == set(log2)
    for k in log2:
        assert np.array_equal(log1[k], log2[k], equal_nan=True), k


@needs_dll
def test_slip_omega_reaches_plant(monkeypatch, fw):
    """patched_simulate sets slip_omega_s on the params of that run only: plant_step sees it
    at every call, slip_nsub follows traction_nsub, MecanumParams() keeps the 0.1 default."""
    seen = []
    orig_step = fwsim.plant_step

    def rec_step(x, u, p, dt, rng):
        seen.append(p.slip_omega_s)
        return orig_step(x, u, p, dt, rng)

    monkeypatch.setattr(fwsim, 'plant_step', rec_step)
    nsub = {}
    for w in (0.05, 0.2):
        seen.clear()
        rec = []
        p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.436))
        with sens.patched_simulate(0, w, rec):
            fwsim.simulate(fw, lambda t: (0.2, 0.0, 0.0), 0.05, p, seed=1, mode='PID',
                           imu=dict(s03.IMU))
        assert seen and all(x == w for x in seen)
        assert len(rec) == 1 and rec[0]['slip_omega_s'] == w
        p_w = fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.436))
        p_w.slip_omega_s = w
        nsub[w] = fwsim.traction_nsub(p_w)
        assert rec[0]['slip_nsub'] == nsub[w]
        assert p.slip_omega_s == w                       # in-memory on that params object
    assert nsub[0.05] != nsub[0.2]
    assert fwsim.MecanumParams().slip_omega_s == 0.1     # params.py default untouched


@needs_dll
def test_io_delay_2ms(monkeypatch, fw):
    """io_delay_ms is a pure 1 ms-per-slot duty pipeline: with delay 0 the duty of the tick
    at t_j reaches the plant in the step at t_j, with delay 2 in the step at t_j + 2."""
    orig_step = fwsim.plant_step
    runs = {}
    for delay in (0, 2):
        u_rec = []

        def rec_step(x, u, p, dt, rng, u_rec=u_rec):
            u_rec.append(np.array(u, dtype=float).copy())
            return orig_step(x, u, p, dt, rng)

        monkeypatch.setattr(fwsim, 'plant_step', rec_step)
        p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.436))
        with sens.patched_simulate(delay, 0.1):
            log = fwsim.simulate(fw, lambda t: (0.2, 0.0, 0.0), 0.05, p, seed=1, mode='PID',
                                 imu=dict(s03.IMU))
        runs[delay] = (u_rec, log)

    u0, log0 = runs[0]
    u2, log2 = runs[2]
    j = next(i for i, d in enumerate(log0['duty']) if np.any(d != 0))
    t_j = int(round(log0['t'][j] * 1000))                # plant call index of the tick
    duty_j = log0['duty'][j]
    prev = log0['duty'][j - 1] if j else np.zeros(4)     # all zeros: j is the first nonzero tick
    # plant input is zero before t_j in both runs -> the duty decided up to t_j is identical
    assert np.array_equal(log0['duty'][:j + 1], log2['duty'][:j + 1])
    # delay 0: the tick duty is applied in the step starting at t_j
    assert np.array_equal(u0[t_j], duty_j) and not np.array_equal(u0[t_j - 1], duty_j)
    # delay 2: the tick duty is applied two steps later
    assert np.array_equal(u2[t_j], prev) and np.array_equal(u2[t_j + 1], prev)
    assert np.array_equal(u2[t_j + 2], duty_j)
