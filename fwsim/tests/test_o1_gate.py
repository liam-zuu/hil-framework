"""Tests of fwsim/o1_gate.py (tune stop gate v3.2: G1-G5, per-frame lo / hi / n_undef, O1 3.5 SC-1, SC3-1).
12/11 (gate v3 G1235, Codex rounds 1-4) and 13/11 (Liam CODE local_db99d287 "Làm rs4 + g4 đi",
local_7f7b0ef4 "Duyệt gộp luôn lo/hi")."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import o1_gate as G                      # noqa: E402
import o1_feasibility as O               # noqa: E402

INF, NAN = float('inf'), float('nan')


def test_same_D_as_the_evaluator():
    assert G.D == O.D == 0.10
    assert G.THETA_MAX_DEG == 7.5
    assert G.THRESHOLDS == dict(g2=0.10, g3=0.10, g4=0.10, g5=7.5)


# ── SC3-1: per-frame bounds, maxima over finite frames field by field, n_undef ──

@pytest.mark.parametrize('frames,state', [
    ([(0.12, 0.11, INF)], 'khong_dat'),                       # SC3-1 example (THESIS 13/11, Codex TX-4)
    ([(0.05, 0.05, 0.05), (0.12, 0.11, INF)], 'khong_dat'),   # one certain frame decides, undefined beside it
    ([(0.05, 0.05, 0.05), (0.09, 0.08, INF)], 'chua_ket_luan'),
    ([(0.05, 0.05, 0.05), (NAN, NAN, NAN)], 'chua_ket_luan'),
    ([(0.10, 0.10, 0.10)], 'dat'),                            # "<=": the edge on the threshold passes
    ([(0.08, 0.06, 0.10)], 'dat'),
    ([(0.09, 0.07, 0.11)], 'chua_ket_luan'),
    ([(0.11, 0.10, 0.12)], 'chua_ket_luan'),                  # lo = threshold is not > threshold
    ([(0.1000001, 0.1000001, 0.1000001)], 'khong_dat'),
    ([], 'chua_ket_luan'),                                    # no frame scored
])
def test_aggregate_states(frames, state):
    q, lo, hi = (np.array([f[i] for f in frames], dtype=float) for i in range(3))
    a = G.aggregate(q, lo, hi, 0.10)
    assert a['state'] == state
    assert a['n_frames'] == len(frames)
    assert a['n_undef'] == sum(not all(math.isfinite(x) for x in f) for f in frames)


def test_aggregate_fields_are_maxima_over_finite_frames_separately():
    q = np.array([0.05, 0.12, 0.02])
    lo = np.array([0.04, 0.11, NAN])
    hi = np.array([0.06, INF, 0.03])
    a = G.aggregate(q, lo, hi, 0.10)
    assert (a['q'], a['lo'], a['hi'], a['n_undef']) == (0.12, 0.11, 0.06, 2)
    assert a['state'] == 'khong_dat' and a['ratio'] == pytest.approx(1.2)
    # the same from the stored numbers alone (what rq3_log recomputes)
    assert G.cond_state(a['lo'], a['hi'], a['n_undef'], 0.10) == 'khong_dat'


@pytest.mark.parametrize('lo,hi,nu,state', [
    (0.11, None, 3, 'khong_dat'), (None, None, 0, 'chua_ket_luan'), (0.05, 0.08, 1, 'chua_ket_luan'),
    (0.05, 0.08, 0, 'dat'), (0.05, 0.12, 0, 'chua_ket_luan'), (0.05, 0.08, True, 'chua_ket_luan'),
    (0.05, 0.08, 0.0, 'chua_ket_luan'),                       # n_undef must be an int
])
def test_cond_state(lo, hi, nu, state):
    assert G.cond_state(lo, hi, nu, 0.10) == state


def test_aggregate_refuses_bounds_around_the_wrong_side():
    with pytest.raises(ValueError, match='lo > q'):
        G.aggregate([0.05], [0.06], [0.07], 0.10)
    with pytest.raises(ValueError, match='lengths'):
        G.aggregate([0.05], [0.04], [0.06, 0.07], 0.10)


@pytest.mark.parametrize('u', [None, NAN, -0.01, True])
def test_frame_bounds_unknown(u):
    lo, hi = G.frame_bounds([0.01, 0.02], u)
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_frame_bounds_symmetric():
    lo, hi = G.frame_bounds([0.01, 0.02], 0.005)
    assert lo == pytest.approx([0.005, 0.015]) and hi == pytest.approx([0.015, 0.025])


# ── G1, outcomes, command maxima ──

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
    assert got[2][1] >= 2.61433749377                       # Codex round 1 counterexample


# ── gate ──

def _frames(n=50, perp=0.02, trk=0.03, bao=0.04, th=3.0, **peak):
    f = dict(e_perp=np.full(n, perp), e_trk=np.full(n, trk), e_bao=np.full(n, bao), e_theta_cmd_deg=np.full(n, th))
    for k, v in peak.items():
        f[k] = f[k].copy()
        f[k][n // 2] = v
    return f


@pytest.mark.parametrize('completed,q', [(True, 'true'), (False, 'false'), ('true', 'true'), ('false', 'false'),
                                         ('inconclusive', 'inconclusive'), (None, 'inconclusive')])
def test_g1_q_is_always_a_string(completed, q):
    g = G.gate(_frames(), completed, v_cmd_max=0.7, wz_cmd_max=1.6, u_t=0.0)
    assert g['g1']['q'] == q and g['g1']['state'] == G.g1_state(completed)


def test_gate_states_and_fields():
    g = G.gate(_frames(e_trk=0.11), 'true', v_cmd_max=1.166, wz_cmd_max=2.614, u_t=0.0)
    assert [g[c]['state'] for c in G.CONDS] == ['dat', 'dat', 'khong_dat', 'dat', 'dat']
    assert g['g3']['q'] == g['g3']['lo'] == g['g3']['hi'] == 0.11 and g['g3']['n_undef'] == 0
    assert g['outcome'] == g['outcome_decide'] == 'khong_dat' and g['any_certain_fail'] and g['complete']
    assert g['decide'] == list(G.CONDS) and g['version'] == G.GATE_VERSION == 'gate_v3.2'
    assert g['max_ratio'] == pytest.approx(1.1) and g['track_err'] is False
    g = G.gate(_frames(), 'true', v_cmd_max=1.166, wz_cmd_max=2.614, u_t=0.0, status='track_err')
    assert g['outcome'] == 'dat' and g['complete'] and g['track_err']


def test_g4_decides():
    g = G.gate(_frames(e_bao=0.1001), 'true', v_cmd_max=0.729, wz_cmd_max=1.634, u_t=0.0)
    assert g['g4']['state'] == 'khong_dat' and g['outcome_decide'] == 'khong_dat'
    g = G.gate(_frames(e_bao=0.1001), 'true', v_cmd_max=0.729, wz_cmd_max=1.634, u_t=None, decide=G.DECIDE_HIL)
    assert g['g4']['state'] == 'khong_dat' and g['outcome_decide'] == 'khong_dat' and g['any_certain_fail']


def test_u_t_is_required_and_none_means_unknown():
    with pytest.raises(TypeError):
        G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0)              # GV3-3: no default
    g = G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=None)
    assert g['g3']['state'] == g['g5']['state'] == 'chua_ket_luan' and g['g2']['state'] == 'dat'
    assert g['g3']['n_undef'] == 50 and math.isnan(g['g3']['lo']) and g['g3']['q'] == 0.03
    assert g['outcome'] == 'chua_ket_luan'
    with pytest.raises(ValueError, match='u_t'):
        G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=-0.001)


def test_gate_bounds_from_u_t():
    # U(G3) = v_max U_t; U(G5) = wz_max U_t in degrees (dossier 5.1), U_p = U_psi = 0
    g = G.gate(_frames(trk=0.095, th=7.0), 'true', v_cmd_max=1.166, wz_cmd_max=2.614, u_t=0.01)
    assert g['g3']['u'] == pytest.approx(0.01166) and g['g3']['hi'] == pytest.approx(0.095 + 0.01166)
    assert g['g5']['u'] == pytest.approx(math.degrees(0.02614))
    assert g['g2']['u'] == 0.0 and g['g4']['u'] == 0.0
    assert g['g3']['state'] == 'chua_ket_luan' and g['g5']['state'] == 'chua_ket_luan'
    assert g['outcome'] == 'chua_ket_luan' and not g['any_certain_fail']


def test_camera_bounds_are_not_handled_here():
    with pytest.raises(ValueError, match='camera'):
        G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.0, u_p=0.002)


def test_decide_sets():
    g = G.gate(_frames(trk=0.5), 'true', v_cmd_max=0.7, wz_cmd_max=1.6, u_t=None, decide=G.DECIDE_HIL)
    assert g['decide'] == ['g1', 'g2', 'g4'] and g['outcome'] == 'chua_ket_luan'
    assert g['outcome_decide'] == 'dat' and not g['complete'] and not g['any_certain_fail']
    g = G.gate(_frames(perp=0.2), 'true', v_cmd_max=0.7, wz_cmd_max=1.6, u_t=None, decide=iter(G.DECIDE_HIL))
    assert g['outcome_decide'] == 'khong_dat' and g['any_certain_fail']         # GV3-R2-5: iterator read once
    for bad in ((), ('g1', 'g2'), ('g1', 'g3'), ('g1', 'g2', 'g4', 'g4')):
        with pytest.raises(ValueError, match='decide'):
            G.gate(_frames(), 'true', v_cmd_max=0.7, wz_cmd_max=1.6, u_t=0.0, decide=bad)


def test_missing_frame_key_is_an_error():
    f = _frames()
    f.pop('e_bao')
    with pytest.raises(ValueError, match='e_bao'):
        G.gate(f, 'true', v_cmd_max=0.7, wz_cmd_max=1.6, u_t=0.0)


def test_nan_frames_count_as_undefined_and_never_pass():
    f = _frames()
    f['e_trk'][3] = NAN
    g = G.gate(f, 'true', v_cmd_max=0.7, wz_cmd_max=1.6, u_t=0.0)
    assert g['g3']['n_undef'] == 1 and g['g3']['state'] == 'chua_ket_luan' and g['outcome'] == 'chua_ket_luan'
