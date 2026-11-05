"""tune_ga.py decision variables (Liam 05/11, CODE session, STATUS_CODE L38): inner gains plus
pos_kp, pos_ki (position loop) and k_psi (heading loop) for both controllers; kd = 0 and b0 fixed.
Checks the mapping only, not GA results."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import tune_ga  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
pytestmark = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')


@pytest.fixture(scope='module')
def fw():
    f = fwsim.FirmwareSim()
    if not f.pos_cols:
        pytest.skip('fwsim.dll built from firmware without pos_ctrl.c')
    return f


def test_vars_and_bounds_match():
    for mode, names in tune_ga.VARS_BY_MODE.items():
        assert names[-3:] == ['pos_kp', 'pos_ki', 'k_psi']
        lo, hi = tune_ga.BOUNDS[mode]
        assert len(lo) == len(hi) == len(names) == 7
        assert all(a < b for a, b in zip(lo, hi))


@pytest.mark.parametrize('mode', ['PID', 'ADRC'])
def test_outer_gains_reach_the_firmware(fw, mode):
    """The three outer variables go to the firmware by name and read back after the run;
    b0 stays at the adrc.c default and kd at 0."""
    fw.reset()
    b0_default = [fw.get_adrc(i)['b0'] for i in range(3)]
    x = [2.0, 1.0, 2.0, 1.0] if mode == 'PID' else [8.0, 30.0, 8.0, 30.0]
    g = tune_ga.gains_for(mode, x + [7.5, 2.5, 1.5], fw)
    assert g['fw_params'] == dict(pos_kp=7.5, pos_ki=2.5, k_psi=1.5)
    fwsim.simulate(fw, lambda t: (0.2, 0.0, 0.0), 0.3, mode=mode, imu={}, **g)
    pp = fw.pos_params()
    assert pp[1] == pytest.approx(7.5) and pp[2] == pytest.approx(2.5)
    if mode == 'ADRC':
        assert [fw.get_adrc(i)['b0'] for i in range(3)] == pytest.approx(b0_default)
    else:
        assert all(fw.get_pid(i)['kd'] == 0.0 for i in range(3))
