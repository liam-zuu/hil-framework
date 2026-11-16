"""Tests of fwsim/o1_gate_eval.py (gate v3.2 evaluator: G4 body envelope, tie set, outgoing side at a K2
corner, per-frame values; evaluate_v7 / evaluate_k unchanged). Liam 13/11 CODE local_db99d287 ("Làm rs4 +
g4 đi"). Every pose is invented; expected G4 values are worked by hand in the comments."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'tools'), os.path.join(HERE, '..', '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import o1_gate_eval as E                 # noqa: E402
import o1_gate as G                      # noqa: E402
import o1_feasibility as O               # noqa: E402
import o1_lanes as L                     # noqa: E402

DT = 0.01
D7 = math.radians(7.5)


def _log(t, pose):
    pose = np.asarray(pose, dtype=float)
    v = np.gradient(pose[:, :2], DT, axis=0) if len(t) > 1 else np.zeros((len(t), 2))
    wz = np.gradient(np.unwrap(pose[:, 2]), DT) if len(t) > 1 else np.zeros(len(t))
    return dict(t=np.asarray(t), pose=pose, v_true=np.column_stack([v, wz]), duty=np.zeros((len(t), 4)),
                pwm=np.zeros((len(t), 4)))


def k_log(task, offset=(0.0, 0.0), yaw=0.0, tail=1.5):
    """Car on the command point of K1 / K2 shifted by `offset` (start frame), constant yaw."""
    lane, sch = L.Lane(L.k_corners(task)), L.KSchedule(task)
    t = np.arange(0.0, sch.Tcmd + tail, DT)
    pts = np.array([lane.point_at(sch.s_cmd(tk))[:2] for tk in t]) + np.asarray(offset)
    return _log(t, np.column_stack([pts, np.full(len(t), yaw)]))


def fig8_log(v=0.5, lateral=0.0, yaw=0.0, tail=2.5):
    """Plant-frame pose of the figure 8 (heading on the tangent + yaw), shifted `lateral` along the normal."""
    sch = O.Schedule(v)
    c, s = math.cos(O.HEAD0), math.sin(O.HEAD0)
    t = np.arange(0.0, sch.Tcmd + tail, DT)
    out = []
    for tk in t:
        fx, fy, h = O.path_point(sch.tau(tk)[0])
        fx, fy = fx - lateral * math.sin(h), fy + lateral * math.cos(h)
        out.append((c * fx + s * fy, -s * fx + c * fy, h - O.HEAD0 + yaw))
    return _log(t, out), sch


def test_fig8_summary_is_evaluate_v7_plus_g4():
    lg, sch = fig8_log()
    ref, ref_rows = O.evaluate_v7(lg, O.Path8(), sch)
    out, rows, fr = E.evaluate_fig8(lg, O.Path8(), sch)
    for k, v in ref.items():                                     # every evaluate_v7 field unchanged
        assert (out[k] == v) or (isinstance(v, float) and math.isnan(v) and math.isnan(out[k])), k
    assert len(rows) == len(ref_rows) and set(fr) == set(G.FRAME_KEYS.values())
    assert all(len(a) == len(rows) for a in fr.values())
    # on the path with the heading on the tangent: G4 = e_perp (no turn), tiny
    assert out['e_bao_max'] == pytest.approx(out['e_perp_max'], abs=1e-6) and out['e_bao_max'] < 1e-3
    assert out['eval_version'] == E.EVAL_VERSION


def test_fig8_lateral_and_yaw():
    # 0.03 m to the side, body turned 5 deg from the tangent: G4 = 0.03 + 0.17 sin 5 + 0.145 (cos 5 - 1)
    lg, sch = fig8_log(lateral=0.03, yaw=math.radians(5.0))
    out, _, fr = E.evaluate_fig8(lg, O.Path8(), sch)
    expect = 0.03 + 0.17 * math.sin(math.radians(5)) + 0.145 * (math.cos(math.radians(5)) - 1)
    mid = slice(len(fr['e_bao']) // 4, len(fr['e_bao']) // 2)   # away from the start (window catching up)
    assert np.median(fr['e_bao'][mid]) == pytest.approx(expect, abs=2e-3)
    assert np.median(fr['e_theta_cmd_deg'][mid]) == pytest.approx(5.0, abs=0.1)


def test_k1_codex_counterexample():
    # dossier v3 CG2-1: K1, 0.095 m off the lane, yaw 7.5 deg: 0.095 + 0.145 sin 7.5 + 0.17 (cos 7.5 - 1)
    out, _, fr = E.evaluate_k(k_log('O1-K1', offset=(0.095, 0.0), yaw=D7), 'O1-K1')
    expect = 0.095 + 0.145 * math.sin(D7) + 0.17 * (math.cos(D7) - 1)
    assert expect == pytest.approx(0.11247, abs=1e-5)
    assert np.median(fr['e_bao']) == pytest.approx(expect, abs=1e-4)
    assert out['e_perp_max'] == pytest.approx(0.095, abs=1e-4)
    g = G.gate(fr, out['completed'], v_cmd_max=0.5, wz_cmd_max=0.0, u_t=0.0)
    assert g['g2']['state'] == 'dat' and g['g4']['state'] == 'khong_dat' and g['g5']['state'] == 'dat'


def test_k1_turned_90_deg_is_negative_and_physical_overhang_zero():
    # dossier v3 section 2: K1, centred, body turned 90 deg: 0.145 - 0.17 = -0.025
    out, _, fr = E.evaluate_k(k_log('O1-K1', yaw=math.pi / 2), 'O1-K1')
    assert np.median(fr['e_bao']) == pytest.approx(-0.025, abs=1e-4)
    assert out['e_bao_phys_max'] == pytest.approx(max(0.0, out['e_bao_max']))


def _k_g4(task, px, py, lo, hi, yaw):
    poly = E.Polyline(L.k_corners(task))
    sol = poly.solutions(px, py, lo, hi)
    return sol, sorted(E.g4_value(d, yaw, h, *E.k_nominal(h)) for _, d, h in sol)


def test_tie_at_k2_corner_takes_both_sides():
    # dossier v3 section 2 (Codex): corner (1, 0), centre (0.919, 0.081), yaw 7.5 deg: both sides 81 mm away;
    # incoming side (vx): 0.081 + 0.17 sin 7.5 + 0.145 (cos 7.5 - 1) = 0.10195; outgoing (vy): 0.09847
    sol, vals = _k_g4('O1-K2', 0.919, 0.081, 0.8, 1.2, D7)
    assert len(sol) == 2 and [d for _, d, _ in sol] == pytest.approx([0.081, 0.081], abs=1e-12)
    assert vals == pytest.approx([0.09847, 0.10195], abs=1e-5)
    sol, _ = _k_g4('O1-K2', 0.919, 0.070, 0.8, 1.2, D7)                     # off the bisector: one solution
    assert len(sol) == 1


def test_g32_7_no_false_tie_and_no_missed_tie():
    # Codex G32-7 (a): (0.92, 0.0800005), the incoming side is 0.5 um farther: NOT a solution; only the
    # outgoing (vy) side: 0.08 + 0.145 sin 7.5 + 0.17 (cos 7.5 - 1) = 0.0974719
    sol, vals = _k_g4('O1-K2', 0.92, 0.0800005, 0.8, 1.2, D7)
    assert len(sol) == 1 and vals == pytest.approx([0.0974719], abs=1e-6)
    # (b): (0.99999, 0.00001): the feet on both sides are 10 um away (a grid sees only the corner):
    # incoming 1e-5 + 0.17 sin 7.5 + 0.145 (cos 7.5 - 1) = 0.0209590; outgoing 1e-5 + 0.145 sin 7.5
    # + 0.17 (cos 7.5 - 1) = 0.0174819 (Claude by hand; Codex wrote 0.0174861; the maximum is the one that counts)
    sol, vals = _k_g4('O1-K2', 0.99999, 0.00001, 0.9, 1.1, D7)
    assert len(sol) == 2 and vals == pytest.approx([0.0174819, 0.0209590], abs=1e-6)


def test_vertex_uses_the_outgoing_side():
    # outside the corner (1, 0): the nearest lane point is the corner; yaw 10 deg tells the sides apart
    y = math.radians(10)
    sol, vals = _k_g4('O1-K2', 1.05, -0.05, 0.9, 1.1, y)
    d = math.hypot(0.05, 0.05)
    assert len(sol) == 1 and sol[0][1] == pytest.approx(d) and sol[0][0] == pytest.approx(1.0)
    assert vals[0] == pytest.approx(d + 0.145 * math.sin(y) + 0.17 * (math.cos(y) - 1))   # 'vy' (outgoing) side


def test_g32_8_closing_corner_takes_the_first_side():
    # Codex G32-8: K2 ends on its start corner; centre (0.081, 0), yaw 7.49 deg: the outgoing side of the
    # closing corner is the first side (vx): 0.081 + 0.17 sin 7.49 + 0.145 (cos 7.49 - 1) = 0.1019228
    sol, vals = _k_g4('O1-K2', 0.081, 0.0, 3.8, 4.0, math.radians(7.49))
    assert len(sol) == 1 and sol[0][0] == pytest.approx(4.0)
    assert vals == pytest.approx([0.1019228], abs=1e-6)
    # K1 is open: its end point keeps its (only) side, heading +90 deg
    sol, _ = _k_g4('O1-K1', 0.05, 2.05, 1.9, 2.0, 0.0)
    assert sol[0][0] == pytest.approx(2.0) and sol[0][2] == pytest.approx(math.pi / 2)


def test_k_nominal_dimensions():
    assert E.k_nominal(0.0)[1:] == (0.17, 0.145) and E.k_nominal(math.pi)[1:] == (0.17, 0.145)
    assert E.k_nominal(math.pi / 2)[1:] == (0.145, 0.17) and E.k_nominal(-math.pi / 2)[1:] == (0.145, 0.17)


def test_grid_projection_keeps_the_argmin_of_nearest():
    # the first grid solution is the point Path8.nearest picks (same window), so e_perp and s_perp agree
    p = O.Path8()
    rng = np.random.default_rng(1)
    for _ in range(50):
        s0 = rng.uniform(0.1, p.S - 0.1)
        i = int(np.searchsorted(p.s, s0))
        px, py = p.x[i] + rng.normal(0, 0.03), p.y[i] + rng.normal(0, 0.03)
        s_n, d_n, h_n, _ = p.nearest(px, py, s0 - 0.03, s0 + 0.03)
        idx, d, dmin = E.grid_projections(p, px, py, s0 - 0.03, s0 + 0.03)
        assert dmin == d_n and p.s[idx[int(np.argmin(d))]] == s_n


def test_rebuilt_window_checked_against_rows():
    lg, sch = fig8_log(lateral=0.01)
    out, rows = O.evaluate_v7(lg, O.Path8(), sch)
    rows[5]['e_perp'] += 1e-6                                   # a row that does not come from this window
    p = O.Path8()
    with pytest.raises(ValueError, match='row e_perp'):
        E.g4_frames(rows, p, sch.vmax, lambda *a: [0.0], E.fig8_nominal)


# -- B81-3 G1 fields of fwsim (15/11; design 03_CODE/docs/2026-11-15_Thiet_ke_bo_cham_fwsim_G1_v1.md) --

def _summ(completed, t_end=30.0, t1=float('nan'), t_prog=float('nan'), c25=False):
    return dict(completed=completed, t_end=t_end, t_deadline=40.0, t_first_track_err=t1, t_progress=t_prog,
                cond2_5_ever=c25)


@pytest.mark.parametrize('summ, want', [
    (_summ('true'), dict(t_done=[30.0], t1=None, prog_before_t1=None, cond25_before_deadline=None)),
    (_summ('false', t_end=39.99, c25=True), dict(t_done=[None], t1=None, prog_before_t1=None,
                                                 cond25_before_deadline=None)),
    (_summ('true', t1=35.0, t_prog=20.0, c25=True), dict(t_done=[30.0], t1=35.0, prog_before_t1=True,
                                                         cond25_before_deadline=True)),
    (_summ('inconclusive', t1=12.0, t_prog=20.0, c25=True), dict(t_done=[None], t1=12.0, prog_before_t1=False,
                                                                 cond25_before_deadline=True)),
    (_summ('false', t1=12.0, t_prog=12.0, c25=False), dict(t_done=[None], t1=12.0, prog_before_t1=False,
                                                           cond25_before_deadline=False)),      # same frame: not before
    (_summ('false', t1=12.0, c25=False), dict(t_done=[None], t1=12.0, prog_before_t1=False,
                                              cond25_before_deadline=False)),                   # no progress at all
])
def test_g1_fields_v7_from_the_summary(summ, want):
    f = E.g1_fields_v7(summ)
    assert f == dict(want, t_deadline=40.0)
    assert G.g1_state_from_fields(f) == G.g1_state(summ['completed'])


# G1F-3 (Codex round 1): the whole evaluate_fig8 summary of the base code (hil 09ff237, run from its own source tree
# with the same fig8_log) on two logs: every key and the exact value (IEEE-754 bytes of floats, repr of strings).
# tools/hil_o1_score.py writes this summary whole, so a key added here would change the HIL output.
BASE_SUMMARY_09FF237 = {
    (0.5, 0.03, 0.05): {
        'S': 'f2804a28c8e631840',
        'T': 'f5a04a28c8e632840',
        'T_cmd': 'fece3ebd6b74d2b40',
        'T_r': 'f90fc4e524a51f73f',
        'completed': "r'true'",
        'cond2_5_ever': 'bTrue',
        'ds_absmax': 'f0000708e323ae73e',
        'ds_max': 'f00002879a01be73e',
        'ds_min': 'f0000708e323ae7be',
        'e_bao_max': 'fe95f7647959ea33f',
        'e_bao_phys_max': 'fe95f7647959ea33f',
        'e_meas_note': "r'no firmware pose columns'",
        'e_meas_perp_max': 'f000000000000f87f',
        'e_perp_global_max': 'f9094630e52b89e3f',
        'e_perp_max': 'f60bba10e52b89e3f',
        'e_perp_max_cens': 'f60bba10e52b89e3f',
        'e_perp_max_le_D': 'bTrue',
        'e_perp_rms': 'f1f1855f151b89e3f',
        'e_s_absmax': 'f000000000080b53c',
        'e_theta_cmd_max_deg': 'fb601837b16eb0640',
        'e_theta_lane_max_deg': 'f8ecfa1acfded0640',
        'e_trk_max': 'f251f85eb51b89e3f',
        'e_trk_rms': 'fb91e85eb51b89e3f',
        'eval_version': "r'o1_gate_eval_v2'",
        'g4_tie_frames': 'i0',
        'n_back_frames': 'i0',
        'n_upper_frames': 'i0',
        's_car_end': 'f2804a28c8e631840',
        'sat_any_frac': 'f0000000000000000',
        'sat_frac': 'f0000000000000000',
        'status': "r'ok'",
        'stop_dist_origin': 'fba1e85eb51b89e3f',
        'stop_heading_err_deg': 'f7c01837b16eb0640',
        'stop_x': 'fee43fffbe9b895bf',
        'stop_y': 'f6043fffbe9b8953f',
        't_deadline': 'fece3ebd6b74d2f40',
        't_end': 'f48e17a14ae472d40',
        't_first_track_err': 'f000000000000f87f',
        't_progress': 'f0ad7a3703d0a2a40',
        'track_err_frames': 'i0',
        'tv_duty': 'f0000000000000000',
        'v': 'f000000000000e03f',
    },
    (0.8, 0.0, 0.0): {
        'S': 'f2804a28c8e631840',
        'T': 'f7085ca2f727c1e40',
        'T_cmd': 'fd50ef5c114e82340',
        'T_r': 'f74303fa86ea70240',
        'completed': "r'true'",
        'cond2_5_ever': 'bTrue',
        'ds_absmax': 'f6062bc2d4638e73e',
        'ds_max': 'f000078101a01e73e',
        'ds_min': 'f6062bc2d4638e7be',
        'e_bao_max': 'fb0beb70e1961e73e',
        'e_bao_phys_max': 'fb0beb70e1961e73e',
        'e_meas_note': "r'no firmware pose columns'",
        'e_meas_perp_max': 'f000000000000f87f',
        'e_perp_global_max': 'f88137c2d4638e73e',
        'e_perp_max': 'f88137c2d4638e73e',
        'e_perp_max_cens': 'f88137c2d4638e73e',
        'e_perp_max_le_D': 'bTrue',
        'e_perp_rms': 'ffaed8e9bc19ad33e',
        'e_s_absmax': 'f6b263d4b898fb33c',
        'e_theta_cmd_max_deg': 'f0000000000000000',
        'e_theta_lane_max_deg': 'f6a8f2444b958573f',
        'e_trk_max': 'f535bda3a584cb93c',
        'e_trk_rms': 'fd4f66955ebe9a03c',
        'eval_version': "r'o1_gate_eval_v2'",
        'g4_tie_frames': 'i0',
        'n_back_frames': 'i0',
        'n_upper_frames': 'i0',
        's_car_end': 'f2804a28c8e631840',
        'sat_any_frac': 'f0000000000000000',
        'sat_frac': 'f0000000000000000',
        'status': "r'ok'",
        'stop_dist_origin': 'fd78e03bea0f5b83c',
        'stop_heading_err_deg': 'f0000000000000000',
        'stop_x': 'f085c143326a6b1bc',
        'stop_y': 'f085c143326a6b1bc',
        't_deadline': 'fd50ef5c114e82740',
        't_end': 'fe17a14ae47e12540',
        't_first_track_err': 'f000000000000f87f',
        't_progress': 'f295c8fc2f5a82240',
        'track_err_frames': 'i0',
        'tv_duty': 'f0000000000000000',
        'v': 'f9a9999999999e93f',
    },
}


def _bits(x):
    if isinstance(x, (bool, np.bool_)):
        return 'b' + str(bool(x))
    if isinstance(x, (float, np.floating)):
        import struct
        return 'f' + struct.pack('<d', float(x)).hex()
    if isinstance(x, (int, np.integer)):
        return 'i' + str(int(x))
    return 'r' + repr(x)


@pytest.mark.parametrize('args', sorted(BASE_SUMMARY_09FF237))
def test_evaluate_fig8_summary_and_version_unchanged_for_hil(args):
    """G1F-3: the evaluate_fig8 summary has exactly the keys and values of the base code (no key added, none
    changed) and EVAL_VERSION is unchanged; the G1 fields are the separate function g1_fields_v7."""
    lg, sch = fig8_log(v=args[0], lateral=args[1], yaw=args[2])
    out, _, _ = E.evaluate_fig8(lg, O.Path8(), sch)
    assert {k: _bits(v) for k, v in out.items()} == BASE_SUMMARY_09FF237[args]
    assert E.EVAL_VERSION == 'o1_gate_eval_v2'

