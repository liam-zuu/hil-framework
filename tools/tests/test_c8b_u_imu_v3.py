"""c8b_u_imu_v3 (U_IMU,D(k), design C8B v7 section 7b): synthetic logs and camera frames of test_c8_u_imu."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c8_u_imu as U  # noqa: E402
import c8b_u_imu_v3 as V  # noqa: E402
import test_c8_u_imu as T  # noqa: E402

A, W_RATE = 13.0, 1.0


def _rate(t):
    return W_RATE if A <= t < A + 3.0 else (-W_RATE if A + 3.0 <= t < A + 5.0 else 0.0)


def _S(tmp_path, name='r', **kw):
    p = str(tmp_path / f'{name}.csv')
    T.make_log(p, _rate, A + 9.0, run=(A, A + 8.0), **kw)
    return V.rv_samples_age(U.load_log(p)), p


def _P3(noi_max=0.002, **kw):
    bang = {f'{i}_{j}_{k}': dict(max_rad=noi_max, n=50, luot=['x']) for i in range(4) for j in range(3)
            for k in range(2)}
    d = dict(version=V.VERSION, delta_plus=0.002, delta_minus=0.003, s_plus=0.0, s_minus=0.0,
             r_dot=math.radians(0.016), q=math.radians(0.01), tau_hat_s=0.0, u_lag_s=0.002,
             E_noi=dict(version=V.NOI_VERSION, bang=bang),
             E_chuyen=dict(version=V.CHUYEN_VERSION, d_delta_plus=0.0005, d_delta_minus=0.0005, d_r_dot=1e-5,
                           d_tau_s=0.003))
    d.update(kw)
    return d


def test_formula_parts(tmp_path):
    S, _ = _S(tmp_path)
    t_W = np.array([A - 1.0, A - 0.9, A - 0.8])                          # still reference window before the turn
    t_k = A + 2.0                                                        # +2 rad turned since W
    u, parts, why = V.u_imu_delta(_P3(), S, t_k, t_W, 1.0, 0.0, np.zeros(3), np.zeros(3), 0.01, 0.01)
    assert why == [] and u is not None
    assert parts['P_bar'] == pytest.approx(2.0, abs=0.02) and parts['N_bar'] == pytest.approx(0.0, abs=1e-6)
    assert parts['thang'] == pytest.approx(0.002 * parts['P_bar'] + 0.003 * parts['N_bar'])
    assert parts['troi'] == pytest.approx(math.radians(0.016) * abs(t_k - t_W.mean()))
    assert parts['q'] == pytest.approx(2 * math.radians(0.01))
    # E_tre: one side each way, u + u_lag = 12 ms at 1 rad/s -> about 0.012 rad; W still -> about 0
    assert parts['tre'] == pytest.approx(0.012, abs=0.0015) and parts['tre_W'] < 1e-6
    assert parts['noi'] == 0.002 and parts['noi_W'] == 0.002
    # E_chuyen: dd+ Pbar + dd- Nbar + drdot |t - tbar| + lag spread 3 ms at 1 rad/s (about 0.003 rad)
    lag = parts['chuyen'] - 0.0005 * parts['P_bar'] - 0.0005 * parts['N_bar'] - 1e-5 * abs(t_k - t_W.mean())
    assert lag == pytest.approx(0.003, abs=4e-4)
    assert u == pytest.approx(sum(parts[k] for k in ('thang', 'troi', 'q', 'tre', 'tre_W', 'noi', 'noi_W', 'chuyen')))


def test_turn_back_and_forth_counts_both_ways(tmp_path):
    # +3 rad then -2 rad: the net turn is 1 rad but P = 3, N = 2 enter (design 7b: P, N, not the net)
    S, _ = _S(tmp_path)
    u, parts, why = V.u_imu_delta(_P3(), S, A + 5.5, np.array([A - 1.0]), 0.0, 0.0, np.zeros(1), np.zeros(1),
                                  0.01, 0.01)
    assert parts['P_bar'] == pytest.approx(3.0, abs=0.03) and parts['N_bar'] == pytest.approx(2.0, abs=0.03)


def test_missing_parts_are_undefined(tmp_path):
    S, _ = _S(tmp_path)
    t_W = np.array([A - 1.0])
    P = _P3()
    P['E_noi'] = dict(version=V.NOI_VERSION, bang={})                     # no held-out block in any band
    u, _, why = V.u_imu_delta(P, S, A + 2.0, t_W, 1.0, 0.0, np.zeros(1), np.zeros(1), 0.01, 0.01)
    assert u is None and any('E_noi' in w for w in why)
    u, _, why = V.u_imu_delta(_P3(), S, A + 2.0, t_W, 9.0, 0.0, np.zeros(1), np.zeros(1), 0.01, 0.01)
    assert u is None and any('outside the bands' in w for w in why)      # |omega| above the table
    u, _, why = V.u_imu_delta(_P3(), S, A + 50.0, t_W, 1.0, 0.0, np.zeros(1), np.zeros(1), 0.01, 0.01)
    assert u is None                                                     # after the log: not covered


def test_freshness(tmp_path):
    S, _ = _S(tmp_path)
    S['age_ms'] = S['age_ms'].copy()
    i = int(np.searchsorted(S['t'], A + 2.0))
    S['age_ms'][i] = 30.0                                                 # one report around t_k too old
    u, _, why = V.u_imu_delta(_P3(), S, A + 2.0, np.array([A - 1.0]), 1.0, 0.0, np.zeros(1), np.zeros(1),
                              0.01, 0.01)
    assert u is None and any('older' in w for w in why)
    assert V.tuoi_ok(S, A - 1.0, 0.01, 0.01)


def test_noi_table_from_held_out_camera(tmp_path):
    # camera yaw = true yaw; an RV interpolation residual appears only where the rate changes inside an RV interval
    fr, lg = str(tmp_path / 'c.csv'), str(tmp_path / 'l.csv')
    T.make_log(lg, _rate, A + 9.0, run=(A, A + 8.0))
    T.write_frames(fr, _rate, 10.0, A + 9.0, offset=0.5, psi0=0.0)
    tab = V.noi_table([(U.read_frames(fr), U.load_log(lg), 0.5, 'r1')], dict(tau_hat_s=0.004, s_plus=0.0,
                                                                             s_minus=0.0))
    assert tab['version'] == V.NOI_VERSION and tab['bang']
    mx = max(e['max_rad'] for e in tab['bang'].values())
    assert 0 < mx < 0.05 and all(e['luot'] == ['r1'] for e in tab['bang'].values())


def test_chuyen_and_build(tmp_path):
    ch = V.chuyen_spread([dict(delta_plus=0.002, delta_minus=0.003, r_dot=1e-4, tau_hat_s=0.004),
                          dict(delta_plus=0.0025, delta_minus=0.002, r_dot=2e-4, tau_hat_s=0.006)])
    assert ch['d_delta_plus'] == pytest.approx(0.0005) and ch['d_tau_s'] == pytest.approx(0.002)
    with pytest.raises(V.U3Error):
        V.chuyen_spread([dict(delta_plus=0.0, delta_minus=0.0, r_dot=0.0, tau_hat_s=0.0)])
    th = dict(s_plus=0.0, s_minus=0.0, delta_plus=0.002, delta_minus=0.003)
    tr = dict(beta_max=1e-4, q=2e-4, s_plus=0.0, s_minus=0.0)
    noi = dict(version=V.NOI_VERSION, bang={})
    d = V.build_v3(th, tr, dict(tau_hat_s=0.004), 0.002, noi, ch)
    assert d['version'] == 'u_imu_dong_v3' and d['r_dot'] == 1e-4 and d['truoc_le'] is True
    with pytest.raises(V.U3Error, match='trial'):
        V.build_v3(th, tr, dict(tau_hat_s=0.004, thu=True), 0.002, noi, ch)
    with pytest.raises(V.U3Error, match='scale'):
        V.build_v3(th, dict(tr, s_plus=0.01), dict(tau_hat_s=0.004), 0.002, noi, ch)
    p = tmp_path / 'v3.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    assert V.load_v3(str(p))['version'] == V.VERSION


def test_window_terms_when_w_moves(tmp_path):
    # W inside the turn: the W means of E_tre and E_noi enter (design 7b: Ebar_tre(W), Ebar_noi(W))
    S, _ = _S(tmp_path)
    t_W = np.array([A + 1.0, A + 1.1])
    u, parts, why = V.u_imu_delta(_P3(), S, A + 6.0, t_W, 0.0, 0.0, np.ones(2), np.zeros(2), 0.01, 0.01)
    assert why == [] and parts['tre_W'] == pytest.approx(0.012, abs=0.0015) and parts['tre'] < 1e-6
    assert u == pytest.approx(sum(parts[k] for k in ('thang', 'troi', 'q', 'tre', 'tre_W', 'noi', 'noi_W', 'chuyen')))
    assert u > parts['tre_W'] + parts['noi_W'] + parts['noi']
