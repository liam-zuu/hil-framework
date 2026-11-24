"""c8b_d2_hang: the fast U_IMU,D equals c8b_u_imu_v3.u_imu_delta, the leave-one-out inputs (design C8B v7 7b, 7c)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c8_u_imu as U  # noqa: E402
import c8b_d2_hang as DH  # noqa: E402
import c8b_u_imu_v3 as V  # noqa: E402
import test_c8b_u_imu_v3 as TV  # noqa: E402


def test_turn_sums_equal_turn_totals(tmp_path):
    S, _ = TV._S(tmp_path)
    ts = DH.TurnSums(S)
    rng = np.random.default_rng(1)
    for _ in range(300):
        a, b = sorted(rng.uniform(S['t_all'][0] - 0.5, S['t_all'][-1] + 0.5, 2))
        try:
            want = U.turn_totals(S, a, b)
        except U.InputError:
            with pytest.raises(U.InputError):
                ts(a, b)
            continue
        got = ts(a, b)
        assert got[0] == pytest.approx(want[0], abs=1e-12) and got[1] == pytest.approx(want[1], abs=1e-12)


def test_turn_sums_suspect_pairs(tmp_path):
    # a log with dropped reports (suspect pairs get the gyro extra on both P and N)
    p = str(tmp_path / 'd.csv')
    TV.T.make_log(p, TV._rate, TV.A + 9.0, run=(TV.A, TV.A + 8.0), miss=[100, 101, 150], gyro_miss=())
    S = V.rv_samples_age(U.load_log(p))
    ts = DH.TurnSums(S)
    for a, b in ((TV.A - 2.0, TV.A + 6.0), (TV.A + 1.0, TV.A + 1.2), (TV.A + 2.0, TV.A + 7.5)):
        want = U.turn_totals(S, a, b)
        got = ts(a, b)
        assert got == pytest.approx(want, abs=1e-12)


def test_unode_equals_u_imu_delta(tmp_path):
    S, _ = TV._S(tmp_path)
    P3 = TV._P3()
    t_W = np.array([TV.A - 1.0, TV.A - 0.95, TV.A - 0.9, TV.A - 0.85])
    om_W, al_W = np.zeros(4), np.zeros(4)
    un = DH.UNode(P3, S, t_W, om_W, al_W, 0.01, 0.01)
    for t_k, om, al in ((TV.A + 2.0, 1.0, 0.0), (TV.A + 4.0, -1.0, 0.5), (TV.A + 5.5, 0.0, 0.0),
                        (TV.A + 50.0, 0.0, 0.0), (TV.A + 2.0, 9.0, 0.0)):
        want = V.u_imu_delta(P3, S, t_k, t_W, om, al, om_W, al_W, 0.01, 0.01)
        got = un(t_k, om, al)
        if want[0] is None:
            assert got[0] is None
        else:
            assert got[0] == pytest.approx(want[0], rel=1e-12)
            for k in ('thang', 'troi', 'q', 'tre', 'tre_W', 'noi', 'noi_W', 'chuyen', 'P_bar', 'N_bar'):
                assert got[1][k] == pytest.approx(want[1][k], rel=1e-12, abs=1e-15)


def test_loo_lags_are_medians_of_the_others():
    tre = dict(tau_hat_s=0.009, runs=[dict(tau_s=t, log=f'r{i}.csv') for i, t in
                                      enumerate((0.007, 0.009, 0.011, 0.010, 0.008))])
    fs = DH.loo_files(tre, dict(delta_plus=0.002, delta_minus=0.003), dict(beta_max=1e-4))
    assert [f['tau_hat_s'] for f in fs] == pytest.approx([0.0095, 0.009, 0.0085, 0.0085, 0.0095])
    ch = V.chuyen_spread(fs)
    assert ch['d_tau_s'] == pytest.approx(0.001) and ch['d_delta_plus'] == 0.0 and ch['n_loo'] == 5
    with pytest.raises(DH.D2Error):
        DH.loo_files(dict(tre, tau_hat_s=0.0091), {'delta_plus': 0, 'delta_minus': 0}, {'beta_max': 0})


def test_noi_loo_leaves_the_target_out(tmp_path):
    caps = []
    for i, w in enumerate((0.5, 1.0, 1.5)):
        p = str(tmp_path / f'l{i}.csv')
        TV.T.make_log(p, lambda t, w=w: w if TV.A <= t < TV.A + 3.0 else 0.0, TV.A + 6.0, run=(TV.A, TV.A + 5.0))
        fp = str(tmp_path / f'f{i}.csv')
        TV.T.write_frames(fp, lambda t, w=w: w if TV.A <= t < TV.A + 3.0 else 0.0, 10.0, TV.A + 6.0, offset=0.5,
                          psi0=0.0)
        caps.append((U.read_frames(fp), U.load_log(p), 0.5, f'r{i}'))
    P = dict(tau_hat_s=0.0, s_plus=0.0, s_minus=0.0)
    out = DH.noi_loo(caps, ['r0', 'r2'], P)
    for r in ('r0', 'r2'):
        want = V.noi_table([c for c in caps if c[3] != r], P)
        assert out[r]['luot'] == want['luot'] and r not in out[r]['luot']
        assert {k: v['max_rad'] for k, v in out[r]['bang'].items()} == \
            pytest.approx({k: v['max_rad'] for k, v in want['bang'].items()})
    with pytest.raises(DH.D2Error):
        DH.noi_loo(caps, ['r7'], P)                                    # a target without its capture
