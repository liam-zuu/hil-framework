"""Tests of fwsim/o1_gate.py (tune stop gate v3, G1, G2, G3, G5; G4 not computed yet)."""
import math
import os
import sys

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
    (0.1000001, 0.0, 'khong_dat'),
    (0.09, 0.02, 'chua_ket_luan'),       # interval [0.07, 0.11] holds 0.10
    (0.12, 0.01, 'khong_dat'),
    (0.08, 0.01, 'dat'),
    (0.11, 0.01, 'chua_ket_luan'),       # q - u = 0.10 is not > 0.10
    (float('nan'), 0.0, 'chua_ket_luan'),
    (None, 0.0, 'chua_ket_luan'),
    (0.05, float('nan'), 'chua_ket_luan'),
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


def test_command_maxima_match_the_dossier():
    # dossier v3 section 5.1: v_cmd,max 0.291 / 0.729 / 1.166 m/s, wz_cmd,max 0.65 / 1.63 / 2.61 rad/s
    got = [G.fig8_cmd_max(O.Schedule(v)) for v in (0.2, 0.5, 0.8)]
    assert [round(v, 3) for v, _ in got] == [0.291, 0.729, 1.166]
    assert [round(w, 2) for _, w in got] == [0.65, 1.63, 2.61]
    for v, s in zip((0.2, 0.5, 0.8), got):
        assert s[0] == pytest.approx(O.Schedule(v).vmax, rel=1e-6)


def _summary(**kw):
    s = dict(completed='true', status='ok', e_perp_max=0.02, e_trk_max=0.03, e_theta_cmd_max_deg=3.0)
    s.update(kw)
    return s


def test_gate_states_and_fields():
    g = G.gate_v3(_summary(e_trk_max=0.11), v_cmd_max=1.166, wz_cmd_max=2.614)
    assert (g['g1']['state'], g['g2']['state'], g['g3']['state'], g['g4']['state'], g['g5']['state']) == \
        ('dat', 'dat', 'khong_dat', 'chua_tinh', 'dat')
    assert g['outcome'] == 'khong_dat' and g['any_certain_fail'] and not g['complete']
    assert g['conditions'] == 'G1,G2,G3,G5' and g['version'] == G.GATE_VERSION
    assert g['max_ratio'] == pytest.approx(1.1)
    assert g['usable_for_decision'] is True and g['track_err'] is False


def test_gate_bounds_from_u_t():
    # U(G3) = U_p + v_max U_t; U(G5) = U_psi + wz_max U_t in degrees (dossier 5.1)
    g = G.gate_v3(_summary(e_trk_max=0.095, e_theta_cmd_max_deg=7.0), v_cmd_max=1.166, wz_cmd_max=2.614,
                  u_p=0.002, u_psi_deg=0.5, u_t=0.01)
    assert g['g3']['u'] == pytest.approx(0.002 + 0.01166)
    assert g['g5']['u'] == pytest.approx(0.5 + math.degrees(0.02614))
    assert g['g2']['u'] == pytest.approx(0.002)
    assert g['g3']['state'] == 'chua_ket_luan' and g['g5']['state'] == 'chua_ket_luan'
    assert g['outcome'] == 'chua_ket_luan' and not g['any_certain_fail']


def test_gate_k_task_heading_key_and_missing_values():
    g = G.gate_v3(dict(completed='inconclusive', status='track_err', e_perp_max=0.01, e_trk_max=0.01,
                       e_theta_start_max_deg=9.0), v_cmd_max=0.5, wz_cmd_max=0.0,
                  theta_key='e_theta_start_max_deg', usable_for_decision=False)
    assert g['g5']['state'] == 'khong_dat' and g['g1']['state'] == 'chua_ket_luan'
    assert g['outcome'] == 'khong_dat' and g['track_err'] and g['usable_for_decision'] is False
    g = G.gate_v3(dict(completed='true', e_perp_max=0.01), v_cmd_max=0.5, wz_cmd_max=0.0)
    assert g['g3']['state'] == 'chua_ket_luan' and math.isnan(g['g3']['q'])   # key missing: never 'dat'
    assert g['outcome'] == 'chua_ket_luan'
