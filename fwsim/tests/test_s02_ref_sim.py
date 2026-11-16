"""CODE_S02 design v3 section 20 on fwsim: the wheel order and signs of tools/s02_ref.ik / fk against the plant
(rigid ground: no slip, so fk(plant wheel speeds) is the plant body velocity; ik is the plant H_inv),
and the whole SIM path (s02_groupA_sim -> runs.csv -> s02_metrics.run_dir_metrics) with families E and R.
Run: python fwsim/build_fwsim.py, then python -m pytest fwsim/tests/test_s02_ref_sim.py"""
import csv
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
import fwsim  # noqa: E402
import s02_groupA_sim as g  # noqa: E402
import s02_ref  # noqa: E402

HAVE_DLL = os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll'))


@pytest.mark.skipif(not HAVE_DLL, reason='fwsim.dll not built')
def test_kinematics_signs_match_the_plant_on_rigid_ground():
    from params import MecanumParams
    fw = fwsim.FirmwareSim()
    p = MecanumParams()
    p.ground_model = 'rigid'
    sp = lambda t: (0.3, 0.15, 0.4) if t < 1.5 else (-0.2, 0.25, -0.6)   # noqa: E731  every axis, both signs
    log = fwsim.simulate(fw, sp, 3.0, p, seed=1, mode='PID', imu=dict(rate_hz=20.0, boot_reset=False))
    vb, om = log['v_true'], log['omega']
    moving = np.abs(vb).max(axis=1) > 0.05
    assert moving.sum() > 100
    # body velocity is the world velocity of the plant state turned by its heading (fwsim.body_velocity): agrees with
    # fk of the wheel speeds to about 1e-4 m/s (0.05 %); a wrong sign or order would be off by the whole speed
    assert np.allclose(s02_ref.fk(om[moving]), vb[moving], atol=5e-4)
    # ik(body velocity) is NOT the wheel speeds in general: four wheels, three body axes; the plant wheels carry a
    # component in the null space of fk (wheels working against each other), up to about 1 rad/s here. So the check
    # of design 20 "ik of the plant body velocity gives w1..w4" is replaced by fk (order and signs) and the matrix
    assert np.abs(vb[moving]).min(axis=0).max() > 0.01
    eye = np.eye(3)
    assert np.allclose(s02_ref.ik(eye[:, 0], eye[:, 1], eye[:, 2]).T, p.H_inv)     # same matrix as the plant


@pytest.mark.skipif(not HAVE_DLL, reason='fwsim.dll not built')
def test_sim_path_through_main_scores_families_e_and_r(tmp_path, monkeypatch):
    """One floor test and one lifted test through s02_groupA_sim.main: t = 0 is prep_s exactly, family R is
    computed from the plant file, the step quantities of A-K1 exist for both families, and metrics_columns.json
    is written next to metrics.csv."""
    monkeypatch.setattr(g, 'GROUP_A', {k: g.GROUP_A[k] for k in ('A-L2', 'A-K1')})
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['s02_groupA_sim.py', '--controller', 'PID', '--reps', '1', '--out', str(out)])
    g.main()
    with open(out / 'metrics.csv', newline='') as f:
        rows = {r['test_id']: r for r in csv.DictReader(f)}
    L, K = rows['A-L2'], rows['A-K1']
    for r in (L, K):
        assert r['status'] == 'ok' and r['t0_metric_method'] == 'sim_exact' and r['t0_metric_us'] == '3000000'
        assert r['metric_reason'] == '' and r['r_reason'] == '' and r['missing_pct'] == '0'
    # straight line at 0.5 m/s: firmware and plant agree on speed within a few cm/s, position within cm
    assert 0 < float(L['e_vx']) < 0.05 and 0 < float(L['e_p_rms']) < 0.05 and float(L['e_psi_max']) < 0.1
    assert math.isfinite(float(L['e_est_p_rms'])) and L['e_est_reason'] == ''
    assert float(K['rms_w_err_true']) > 0 and math.isfinite(float(K['rise_s_true_seg2']))
    assert K['step_seg2'] == '+0.00->+0.30' and K['no90_seg2'] == '0'
    assert (out / 'metrics_columns.json').is_file()
