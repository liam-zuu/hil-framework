"""fwsim side of L34 (plant source blobs in meta.yaml 'plant') and L37 (run_bad when the pose
resets mid-run). Needs fwsim/build/fwsim.dll from firmware with pos_ctrl.c (AGV int/post-g2).
Run: python -m pytest fwsim/tests"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
from plant_blob import plant_source  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
P = {n: i for i, n in enumerate(fwsim.POS_COLS)}


def test_plant_meta_has_source_blobs():
    d = json.loads(fwsim.plant_meta(fwsim.MecanumParams()))
    assert d['src'] == fwsim.PLANT_SRC == plant_source()
    assert all(len(d['src'][k]) == 40 for k in ('plant_step_py', 'params_py')) and d['src']['method']


def test_run_validity_without_pos_columns():
    f = fwsim.run_validity({'pos': []})
    assert math.isnan(f['run_bad'])


@pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')
def test_simulate_run_is_clean():
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    log = fwsim.simulate(fw, lambda t: (0.3, 0.0, 0.0), 2.0, imu={})
    f = fwsim.run_validity(log)
    assert f['pos_reset_mid'] == 0 and f['run_bad'] == 0


@pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')
def test_stale_packets_mid_run_mark_the_run_bad():
    """run_active low for 5 ticks in the middle (stale packets on the bench): the firmware resets
    the pose when it comes back (spec controller-IMU 14.7.3 (A)); run_validity marks the run."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    fw.reset()
    rows = []
    for k in range(1, 201):
        fw.set_run(not 100 <= k < 105, False)
        fw.tick(k * fwsim.FW_DT_US, [0, 0, 0, 0], (0.3, 0.0, 0.0))
        rows.append(list(fw.pos_last()))
    pos = np.array(rows)
    assert pos[-1, P['pos_reset_n']] == pos[0, P['pos_reset_n']] + 1
    f = fwsim.run_validity({'pos': pos})
    assert f['pos_reset_mid'] == 1 and f['run_bad'] == 1
