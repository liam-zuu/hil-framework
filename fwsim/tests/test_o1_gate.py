"""Tests of fwsim/o1_gate.py (tune stop gate v3, G1, G2, G3, G5; G4 not computed yet).
After Codex round 1 (GV3-2, 3, 4, 7; Liam "Cho sửa 7 chỗ", CODE local_87282b70)."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import o1_gate as G                      # noqa: E402
import o1_feasibility as O               # noqa: E402


def test_same_D_as_the_evaluator():
    assert G.D == O.D == 0.10
    assert G.THETA_MAX_DEG == 7.5


@pytest.mark.parametrize('q,u,state', [
    (0.10, 0.0, 'dat'),                  # q <= threshold passes (O1: e_perp_max <= D)
    (0.08, 0.02, 'dat'),                 # GV3-4 (rejected): upper edge on the threshold, "<=": certain pass
    (0.1000001, 0.0, 'khong_dat'),
    (0.09, 0.02, 'chua_ket_luan'),       # interval [0.07, 0.11] holds 0.10
    (0.12, 0.01, 'khong_dat'),
    (0.11, 0.01, 'chua_ket_luan'),       # q - u = 0.10 is not > 0.10
    (float('nan'), 0.0, 'chua_ket_luan'),
    (None, 0.0, 'chua_ket_luan'),
    (True, 0.0, 'chua_ket_luan'),        # a bool is not a measured value
    (0.05, float('nan'), 'chua_ket_luan'),
    (0.05, None, 'chua_ket_luan'),
    (0.05, -0.01, 'chua_ket_luan'),      # a negative bound is an input error, never 'dat'
])
def test_tri_state(q, u, state):
    assert G.tri_state(q, 0.10, u) == state


def test_g1_and_outcome_order():
    assert [G.g1_state(c) for c in ('true', 'false', 'inconclusive', True, False, None)] == \
        ['dat', 'khong_dat', 'chua_ket_luan', 'dat', 'khong_dat', 'chua_ket_luan']
    assert G.run_outcome(['dat', 'chua_ket_luan', 'khong_dat']) == 'khong_dat'
    assert G.run_outcome(['dat', 'chua_ket_luan']) == 'chua_ket_luan'
    assert G.run_outcome(['dat', 'dat']) == 'dat'
    assert G.run_outcome([]) == 'chua_ket_luan'


def _dense_peak(v):
    """Reference: |dpsi/dtau| w0 and |p'| w0 on a 2e6-point grid (independent of _refined_max)."""
    s = O.Schedule(v)
    tau = np.linspace(0.0, 2.0 * math.pi, 2_000_001)
    dx, dy = np.cos(tau), np.cos(2.0 * tau)
    ddx, ddy = -np.sin(tau), -2.0 * np.sin(2.0 * tau)
    return float(np.hypot(dx, dy).max() * s.w0), float(np.abs((dx * ddy - dy * ddx) / (dx * dx + dy * dy)).max() * s.w0)


def test_command_maxima_match_the_dossier_and_bound_the_peak():
    # dossier v3 section 5.1: v_cmd,max 0.291 / 0.729 / 1.166 m/s, wz_cmd,max 0.65 / 1.63 / 2.61 rad/s
    got = [G.fig8_cmd_max(O.Schedule(v)) for v in (0.2, 0.5, 0.8)]
    assert [round(v, 3) for v, _ in got] == [0.291, 0.729, 1.166]
    assert [round(w, 2) for _, w in got] == [0.65, 1.63, 2.61]
    for v, (vm, wm) in zip((0.2, 0.5, 0.8), got):
        rv, rw = _dense_peak(v)
        assert rv <= vm <= rv * (1 + 3 * G.CMD_MARGIN)       # GV3-7: an upper bound, and a tight one
        assert rw <= wm <= rw * (1 + 3 * G.CMD_MARGIN)
    # Codex round 1 counterexample: the continuous peak at 0.8 m/s is about 2.61433749377 rad/s
    assert got[2][1] >= 2.61433749377


def _summary(**kw):
    s = dict(completed='true', status='ok', e_perp_max=0.02, e_trk_max=0.03, e_theta_cmd_max_deg=3.0)
    s.update(kw)
    return s


def test_gate_states_and_fields():
    g = G.gate_v3(_summary(e_trk_max=0.11), v_cmd_max=1.166, wz_cmd_max=2.614, u_t=0.0)
    assert (g['g1']['state'], g['g2']['state'], g['g3']['state'], g['g4']['state'], g['g5']['state']) == \
        ('dat', 'dat', 'khong_dat', 'chua_tinh', 'dat')
    assert g['outcome'] == g['outcome_decide'] == 'khong_dat' and g['any_certain_fail'] and not g['complete']
    assert g['decide'] == list(G.CONDS) and g['version'] == G.GATE_VERSION
    assert g['max_ratio'] == pytest.approx(1.1) and g['track_err'] is False


def test_u_t_is_required_and_none_means_unknown():
    with pytest.raises(TypeError):
        G.gate_v3(_summary(), v_cmd_max=1.0, wz_cmd_max=1.0)              # GV3-3: no default
    g = G.gate_v3(_summary(), v_cmd_max=1.0, wz_cmd_max=1.0, u_t=None)
    assert g['g3']['state'] == g['g5']['state'] == 'chua_ket_luan' and g['g2']['state'] == 'dat'
    assert math.isnan(g['g3']['u']) and g['outcome'] == 'chua_ket_luan'


def test_gate_bounds_from_u_t():
    # U(G3) = U_p + v_max U_t; U(G5) = U_psi + wz_max U_t in degrees (dossier 5.1)
    g = G.gate_v3(_summary(e_trk_max=0.095, e_theta_cmd_max_deg=7.0), v_cmd_max=1.166, wz_cmd_max=2.614,
                  u_p=0.002, u_psi_deg=0.5, u_t=0.01)
    assert g['g3']['u'] == pytest.approx(0.002 + 0.01166)
    assert g['g5']['u'] == pytest.approx(0.5 + math.degrees(0.02614))
    assert g['g2']['u'] == pytest.approx(0.002)
    assert g['g3']['state'] == 'chua_ket_luan' and g['g5']['state'] == 'chua_ket_luan'
    assert g['outcome'] == 'chua_ket_luan' and not g['any_certain_fail']


def test_decide_subset_hil():
    # GV3-2: on HIL G1, G2 decide, G3, G5 report only; a certain fail of G1 or G2 still decides
    g = G.gate_v3(_summary(completed='false', e_trk_max=0.5), v_cmd_max=0.7, wz_cmd_max=1.6, u_t=None,
                  decide=G.DECIDE_HIL)
    assert g['decide'] == ['g1', 'g2'] and g['outcome_decide'] == 'khong_dat' and g['any_certain_fail']
    g = G.gate_v3(_summary(e_trk_max=0.5), v_cmd_max=0.7, wz_cmd_max=1.6, u_t=None, decide=G.DECIDE_HIL)
    assert g['outcome'] == 'chua_ket_luan' and g['outcome_decide'] == 'dat' and not g['complete']
    assert not g['any_certain_fail']                     # G3 (U_t unknown) does not decide on HIL
    g = G.gate_v3(_summary(e_trk_max=0.5), v_cmd_max=0.7, wz_cmd_max=1.6, u_t=0.0, decide=G.DECIDE_HIL)
    assert g['g3']['state'] == 'khong_dat' and g['outcome'] == 'khong_dat' and g['outcome_decide'] == 'dat'
    with pytest.raises(ValueError):
        G.gate_v3(_summary(), v_cmd_max=0.7, wz_cmd_max=1.6, u_t=0.0, decide=())
    with pytest.raises(ValueError, match='g1 and g2'):                   # GV3-R2-2: G1, G2 always decide
        G.gate_v3(_summary(), v_cmd_max=0.7, wz_cmd_max=1.6, u_t=0.0, decide=('g1', 'g3'))


def test_decide_iterator_keeps_every_condition():
    # GV3-R2-5: an iterator is read once; G2 certainly failing must still decide
    s = _summary(e_perp_max=0.2)
    g = G.gate_v3(s, v_cmd_max=0.7, wz_cmd_max=1.6, u_t=None, decide=iter(G.DECIDE_HIL))
    assert g['decide'] == ['g1', 'g2'] and g['outcome_decide'] == 'khong_dat' and g['any_certain_fail']
    ev = G.evaluate_outcomes(g, iter(['g1', 'g2']))
    assert ev['outcome_decide'] == 'khong_dat' and ev['any_certain_fail']


def test_gate_k_task_heading_key_and_missing_values():
    g = G.gate_v3(dict(completed='inconclusive', status='track_err', e_perp_max=0.01, e_trk_max=0.01,
                       e_theta_start_max_deg=9.0), v_cmd_max=0.5, wz_cmd_max=0.0, u_t=0.0,
                  theta_key='e_theta_start_max_deg')
    assert g['g5']['state'] == 'khong_dat' and g['g1']['state'] == 'chua_ket_luan'
    assert g['outcome'] == 'khong_dat' and g['track_err']
    g = G.gate_v3(dict(completed='true', e_perp_max=0.01), v_cmd_max=0.5, wz_cmd_max=0.0, u_t=0.0)
    assert g['g3']['state'] == 'chua_ket_luan' and math.isnan(g['g3']['q'])   # key missing: never 'dat'
    assert g['outcome'] == 'chua_ket_luan'
