"""s02_groupA_sim (spec S02 P2 on fwsim): the IMU noise replay seed is the repetition number
(Liam 11/11 CODE local_7787e22f), so the N repetitions get different noise segments and a re-run
of one repetition is the same run bit for bit (S02 v1.3 A-R2-2).
Run: python -m pytest fwsim/tests/test_s02_groupA_sim.py"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import s02_groupA_sim as s02  # noqa: E402

HAVE_DLL = os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll'))


def test_rep_imu_sets_the_noise_seed_to_the_repetition():
    imu = dict(rate_hz=20.0, boot_reset=False, noise='replay', noise_seed=0)
    assert s02.rep_imu(imu, 3)['noise_seed'] == 3 and imu['noise_seed'] == 0      # the input is not changed
    assert s02.rep_imu(dict(rate_hz=20.0, boot_reset=False), 3) == dict(rate_hz=20.0, boot_reset=False)
    assert s02.rep_imu(None, 3) is None


@pytest.mark.skipif(not HAVE_DLL, reason='fwsim.dll not built')
def test_repetitions_differ_by_noise_and_a_rerun_is_identical():
    fw = fwsim.FirmwareSim()
    imu = dict(rate_hz=20.0, boot_reset=False, noise='replay', noise_seed=0)

    def run(rep):
        fw.reset()
        log = fwsim.simulate(fw, lambda t: (0.0, 0.0, 0.5 if t > 1.0 else 0.0), 4.0, None,
                             seed=rep, mode='PID', imu=s02.rep_imu(imu, rep))
        return np.asarray(log['imu'])[:, fwsim.IMU_COLS.index('gz')]

    g1, g2, g1b = run(1), run(2), run(1)
    assert np.array_equal(g1, g1b)              # same seed: bit for bit
    assert not np.array_equal(g1, g2)           # another repetition: another noise segment
