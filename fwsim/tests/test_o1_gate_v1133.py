"""Tests of the O1 v11.33 part of fwsim/o1_gate.py (CODE_GATE 17/11/2026, design CLAUDE_THESIS
03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md sections 3, 4, 6; Liam 17/11 at CODE_GATE "Duyệt, gửi CODE luôn
(Recommended)"): per-frame bounds of G3, G5 (cmd_rate_frames, v_k / w_k), the report-only conditions of a level
(report_only, required, complete), and the gate_v3.2 dict kept when none of them is given. Every number is invented."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import o1_gate as G                      # noqa: E402
import o1_feasibility as O               # noqa: E402
import o1_tasks as OT                    # noqa: E402

SHA = 'ab' * 32


def _frames(n=50, perp=0.02, trk=0.03, bao=0.04, th=3.0, t=None):
    f = dict(e_perp=np.full(n, perp), e_trk=np.full(n, trk), e_bao=np.full(n, bao), e_theta_cmd_deg=np.full(n, th))
    if t is not None:
        f['t'] = np.asarray(t, dtype=float)
    return f


def _f08():
    sch = O.Schedule(OT.FIG8['O1-F08'])
    v, wz = G.fig8_cmd_max(sch)
    return sch, v, wz


# ── 6.1: U = 0 gives the same numbers and states on both paths; no new parameter = the gate_v3.2 dict ──

@pytest.mark.parametrize('kw', [dict(trk=0.11), dict(th=8.0), dict(), dict(perp=0.2, bao=0.3)])
@pytest.mark.parametrize('completed', ['true', 'false', 'inconclusive'])
def test_u_zero_same_on_both_paths(kw, completed):
    sch, v, wz = _f08()
    t = np.linspace(0.0, sch.Tcmd, 50)
    fr = _frames(t=t, **kw)
    old = G.gate(fr, completed, v_cmd_max=v, wz_cmd_max=wz, u_t=0.0)
    vk, wk = G.cmd_rate_frames(sch.setpoint, sch.Tcmd, t, 0.0, v_cap=v, wz_cap=wz)
    new = G.gate(fr, completed, v_cmd_max=v, wz_cmd_max=wz, u_t=0.0, v_k=vk, w_k=wk)
    assert old['version'] == G.GATE_VERSION and new['version'] == G.GATE_VERSION_V1133
    for c in G.MEASURED:
        for k in ('q', 'lo', 'hi', 'n_undef', 'state', 'u'):
            assert old[c][k] == new[c][k], (c, k)
    for k in ('outcome', 'outcome_decide', 'complete', 'any_certain_fail', 'decide', 'max_ratio'):
        assert old[k] == new[k], k


def test_no_new_parameter_keeps_the_v32_keys():
    g = G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.01)
    assert g['version'] == 'gate_v3.2'
    for k in ('u_mode', 'level', 'gate_file_sha256', 'report_only', 'required'):
        assert k not in g


# ── 6.2: per-frame bounds ──

def _true_sup(f, t_cmd, tk, u, turn, step=1e-6):
    a, b = max(0.0, tk - u), min(t_cmd, tk + u)
    if b < a:
        return 0.0, 0.0
    xs = np.arange(a, b + step / 2, step)
    sp = [math.hypot(*f(x)[:2]) for x in xs]
    w = [abs(f(x)[2]) for x in xs] if turn else [0.0]
    return max(sp), max(w)


def test_cmd_rate_frames_bound_and_ends():
    sch, v, wz = _f08()
    u = 0.014
    t = np.array([-1.0, -0.02, 0.0, 0.005, 0.3, 1.0, 2.5, sch.Tcmd * 0.5, sch.Tcmd - 0.01, sch.Tcmd + 0.01,
                  sch.Tcmd + 0.5])
    vk, wk = G.cmd_rate_frames(sch.setpoint, sch.Tcmd, t, u, v_cap=v, wz_cap=wz)
    assert vk[0] == 0.0 and wk[0] == 0.0 and vk[1] == 0.0                   # window wholly before t = 0
    assert vk[-1] == 0.0 and wk[-1] == 0.0                                   # wholly after T_cmd
    assert vk[2] < 0.05 * v                                                  # the ramp start: far below the peak
    for k, tk in enumerate(t):
        ts, tw = _true_sup(sch.setpoint, sch.Tcmd, tk, u, True)
        assert vk[k] >= ts and wk[k] >= tw                                   # an upper bound
        assert vk[k] <= ts + 2e-3 and wk[k] <= tw + 2e-2                     # and close to the sup
        assert vk[k] <= v and wk[k] <= wz                                    # capped by the peak
    assert max(vk) < v or math.isclose(max(vk), v)


def test_cmd_rate_frames_k_tasks_no_turn():
    t_cmd, f = OT.task('O1-K2', allow_final_set=True)
    t = np.linspace(-0.5, t_cmd + 0.5, 40)
    vk, wk = G.cmd_rate_frames(f, t_cmd, t, 0.01, turn=False)
    assert np.all(wk == 0.0) and np.all(vk >= 0.0) and vk.max() <= OT.K_V * (1 + 1e-3) + 1e-3


def test_cmd_rate_frames_heading_rate_is_continuous_through_pi():
    # the tangent heading of the figure 8 crosses +-pi; the rate comes from the analytic tangent (no wrapped difference)
    sch, v, wz = _f08()
    tt = np.linspace(0.0, sch.Tcmd, 4001)
    _, wk = G.cmd_rate_frames(sch.setpoint, sch.Tcmd, tt, 0.0, wz_cap=wz)
    assert np.max(np.abs(np.diff(wk))) < 0.05 * wz


def test_per_frame_bound_is_tighter_away_from_the_peak():
    sch, v, wz = _f08()
    u = 0.014
    t = np.array([0.01, sch.Tcmd * 0.5])
    fr = _frames(n=2, trk=0.095, th=7.0, t=t)
    vk, wk = G.cmd_rate_frames(sch.setpoint, sch.Tcmd, t, u, v_cap=v, wz_cap=wz)
    old = G.gate(fr, 'true', v_cmd_max=v, wz_cmd_max=wz, u_t=u)
    new = G.gate(fr, 'true', v_cmd_max=v, wz_cmd_max=wz, u_t=u, v_k=vk, w_k=wk)
    assert old['g3']['hi'] == pytest.approx(0.095 + v * u)
    assert new['g3']['hi'] == pytest.approx(0.095 + vk.max() * u) and new['g3']['hi'] <= old['g3']['hi']
    assert new['g3']['lo'] == pytest.approx(0.095 - vk.min() * u)            # lo, hi maxima taken apart (O1 line 301)
    assert new['g3']['u'] == pytest.approx(vk.max() * u) and new['u_mode'] == 'khung'
    assert new['g5']['lo'] == pytest.approx(7.0 - math.degrees(wk.min() * u))


def test_per_frame_inputs_checked():
    with pytest.raises(ValueError, match='go together'):
        G.gate(_frames(n=3), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.01, v_k=np.zeros(3))
    with pytest.raises(ValueError, match='one per frame'):
        G.gate(_frames(n=3), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.01, v_k=np.zeros(2), w_k=np.zeros(2))
    with pytest.raises(ValueError, match='one per frame'):
        G.gate(_frames(n=2), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.01, v_k=np.array([0.1, -0.1]),
               w_k=np.zeros(2))
    g = G.gate(_frames(n=2), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=None, v_k=np.zeros(2), w_k=np.zeros(2))
    assert g['g3']['state'] == 'chua_ket_luan' and g['u_mode'] == 'khung'    # U_t not known: as before


# ── 6.3: report-only conditions of the level (O1 3.9 rule 2b, line 321, OG2-3, OG2-6) ──

def test_report_only_g5_does_not_decide():
    fr = _frames(th=9.0)                                                     # G5 certainly above 7.5 deg
    base = G.gate(fr, 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.0)
    assert base['outcome_decide'] == 'khong_dat' and base['any_certain_fail']
    g = G.gate(fr, 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.0, report_only=['g5'], level='O1-F08',
               gate_file_sha256=SHA)
    assert g['g5']['state'] == 'khong_dat' and g['g5']['q'] == 9.0          # still computed, stored
    assert g['decide'] == ['g1', 'g2', 'g3', 'g4'] and g['required'] == ['g1', 'g2', 'g3', 'g4']
    assert g['outcome_decide'] == 'dat' and g['complete'] is True and g['any_certain_fail'] is False
    assert g['outcome'] == 'khong_dat'                                       # the report over all five
    assert (g['level'], g['gate_file_sha256'], g['report_only']) == ('O1-F08', SHA, ['g5'])


def test_report_only_on_hil_without_the_mapping():
    g = G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=None, decide=G.DECIDE_HIL,
               report_only=['g2'], level='O1-F08', gate_file_sha256=SHA)
    assert g['decide'] == ['g1'] and g['complete'] is False
    g = G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=None, decide=G.DECIDE_HIL,
               report_only=[], level='O1-F05', gate_file_sha256=SHA)
    assert g['decide'] == ['g1', 'g2'] and g['complete'] is False and g['version'] == 'gate_v3.3'


@pytest.mark.parametrize('kw, msg', [
    (dict(report_only=['g4'], level='O1-F05', gate_file_sha256=SHA), 'report_only'),
    (dict(report_only=['g1'], level='O1-F05', gate_file_sha256=SHA), 'report_only'),
    (dict(report_only=['g5', 'g5'], level='O1-F05', gate_file_sha256=SHA), 'report_only'),
    (dict(report_only=['g5'], level='O1-F05'), 'gate_file_sha256'),
    (dict(report_only=['g5'], level='O1-F05', gate_file_sha256='AB' * 32), 'gate_file_sha256'),
    (dict(report_only=['g5'], gate_file_sha256=SHA), 'level'),
    (dict(level='O1-F05'), 'without report_only'),
])
def test_report_only_inputs_checked(kw, msg):
    with pytest.raises(ValueError, match=msg):
        G.gate(_frames(), 'true', v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.0, **kw)


def test_check_decide_masked_and_evaluate_outcomes():
    req = G.required_of(['g3', 'g5'])
    assert req == ('g1', 'g2', 'g4')
    assert G.check_decide_masked(['g4', 'g1', 'g2'], req) == ('g1', 'g2', 'g4')
    assert G.check_decide_masked(['g1', 'g2'], req) == ('g1', 'g2')
    with pytest.raises(ValueError):
        G.check_decide_masked(list(G.CONDS), req)
    g = {c: dict(state='dat') for c in G.CONDS}
    g['g3']['state'] = 'khong_dat'
    ev = G.evaluate_outcomes(g, req, req)
    assert ev['complete'] is True and ev['outcome_decide'] == 'dat' and ev['outcome'] == 'khong_dat'
    assert G.evaluate_outcomes(g, req)['complete'] is False                  # required None = all five (v3.2)
