"""s02_ref.py (CODE_S02 design v3 sections 18.2, 19, 20): reference r(t), command pose, test frame, steps."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import s02_ref as R  # noqa: E402
from trajectory_pub import GROUP_A, get_body_velocity, get_reference  # noqa: E402


def test_reference_is_the_held_schedule_and_zero_outside():
    u = R.samples('A-F2')
    t = np.array([-0.001, 0.0, 0.0199, 0.02, 0.5 + 1e-4, R.n_samples('A-F2') * 0.02, 99.0])
    r = R.reference('A-F2', t)
    assert np.all(r[0] == 0) and np.all(r[-1] == 0) and np.all(r[-2] == 0)
    assert np.allclose(r[1], u[0]) and np.allclose(r[2], u[0]) and np.allclose(r[3], u[1])
    assert np.allclose(r[4], get_body_velocity(0.5, GROUP_A['A-F2']))


def test_cell_edges_survive_float_rounding():
    """A tick exactly on a cell edge (e.g. 3.02 s from an integer-us clock) belongs to the new cell."""
    ts_us = np.array([3_020_000, 3_000_000, 2_999_999])
    k = R.cell_index((ts_us - 0.0) * 1e-6)
    assert list(k) == [151, 150, 149]


@pytest.mark.parametrize('tid', ['A-L1', 'A-L2', 'A-C1', 'A-C2', 'A-S1', 'A-S2', 'A-F1', 'A-F2'])
def test_command_pose_end_matches_trajectory_pub(tid):
    """p_ref integrates the 20 ms held schedule exactly; its end point matches get_reference (strafe: 2 m along y)
    within the held-cell discretisation (figure 8 path within 12 mm everywhere, end within 0.1 mm)."""
    T = R.t_cmd(tid)
    ts = np.linspace(0, T, 300)
    p = R.pose_ref(tid, ts)
    spec = GROUP_A[tid]
    g = (np.stack([0 * ts, spec.v * ts], 1) if spec.kind == 'strafe' else
         np.array([get_reference(t, spec)[:2] for t in ts]))
    dev = np.hypot(*(p[:, :2] - g).T)
    assert dev[-1] < 1e-4
    assert dev.max() < (0.012 if spec.kind == 'figure8' else 1e-6)


def test_start_heading():
    assert R.start_heading('A-L1') == 0.0 and R.start_heading('A-S1') == 0.0 and R.start_heading('A-K1') == 0.0
    assert abs(R.start_heading('A-C1')) < 1e-6
    assert R.start_heading('A-F1') == pytest.approx(math.pi / 4, abs=1e-6)


def test_arc_integration_exact():
    """A circle cell by cell lands on the circle (wz != 0 branch), a straight cell on the line (wz = 0)."""
    x, y, psi = 0.0, 0.0, 0.3
    for _ in range(50):
        x, y, psi = R._advance(x, y, psi, (0.5, 0.0, 0.5), 0.02)
    cx, cy = -math.sin(0.3) * 1.0, math.cos(0.3) * 1.0          # centre at radius 1 to the left
    assert math.hypot(x - cx, y - cy) == pytest.approx(1.0, abs=1e-12)
    assert R._advance(0, 0, 0.0, (0.0, 0.4, 0.0), 0.5)[:2] == pytest.approx((0.0, 0.2))


def test_test_frame_round_trip_and_wrap():
    px, py, th = R.to_test_frame(np.array([1.0 + math.cos(0.7)]), np.array([2.0 + math.sin(0.7)]),
                                 np.array([0.7]), 1.0, 2.0, 0.7, math.pi / 4)
    assert px[0] == pytest.approx(math.cos(math.pi / 4)) and py[0] == pytest.approx(math.sin(math.pi / 4))
    assert th[0] == pytest.approx(math.pi / 4)
    assert R.wrap(np.array([math.pi, -math.pi, 3 * math.pi + 0.1]))[0] == pytest.approx(math.pi)
    assert R.wrap(np.array([-math.pi]))[0] == pytest.approx(math.pi)
    assert R.wrap(np.array([3 * math.pi + 0.1]))[0] == pytest.approx(-math.pi + 0.1)


def test_fk_inverts_ik():
    v = np.array([[0.3, -0.1, 0.4], [0.0, 0.5, -1.0]])
    assert np.allclose(R.fk(R.ik(*v.T)), v)


def test_segments_of_lifted_steps():
    assert R.segments('A-K1') == [(0.0, 3.0, 0.0), (3.0, 6.0, 0.3), (6.0, 9.0, 0.0)]
    assert R.segments('A-K2') == [(0.0, 3.0, 0.3), (3.0, 6.0, -0.3), (6.0, 9.0, 0.3)]


def _first_order(tid, tau, t):
    """vx of a first-order follower of r_vx with time constant tau (exact between steps)."""
    out = np.zeros(len(t))
    for i, ti in enumerate(t):
        v = R.segments(tid)[0][2]
        for a, b, lev in R.segments(tid)[1:]:
            if ti >= a:
                v = lev + (v - lev) * math.exp(-(min(ti, b) - a) / tau) if ti < b else lev + (v - lev) * math.exp(-(b - a) / tau)
        out[i] = v
    return out


@pytest.mark.parametrize('tid', ['A-K1', 'A-K2'])
def test_rise_of_a_first_order_response(tid):
    """t_rise of a first-order lag is tau ln 9 (lines every 1 ms here); A-K2 steps are negative (A < 0)."""
    t = np.arange(0, 9.0, 0.001)
    q = R.step_quantities(t, _first_order(tid, 0.1, t), tid)
    for n in (2, 3):
        assert q[f'rise_s_seg{n}'] == pytest.approx(0.1 * math.log(9), abs=0.002)
        assert q[f'no90_seg{n}'] == 0 and q[f'overshoot_pct_seg{n}'] == 0.0
        assert q[f'e_xl_seg{n}'] < 1e-6 and q[f'rise_uninterp_seg{n}'] == 0
        assert 0.75 < q[f'frac90_seg{n}'] < 0.95
    assert q['step_seg2'] == ('+0.00->+0.30' if tid == 'A-K1' else '+0.30->-0.30')


def test_overshoot_no90_and_uninterpretable_rise():
    t = np.arange(0, 9.0, 0.01)
    v = np.where(t < 3, 0.0, np.where(t < 3.2, 0.36, 0.3))            # 20 % overshoot of 0 -> 0.3
    v = np.where(t >= 6, 0.15, v)                                     # never reaches 0.9 of 0.3 -> 0
    q = R.step_quantities(t, v, 'A-K1')
    assert q['overshoot_pct_seg2'] == pytest.approx(20.0)
    assert q['rise_s_seg2'] == pytest.approx(0.0) and q['no90_seg2'] == 0
    assert q['no90_seg3'] == 1 and math.isnan(q['rise_s_seg3'])
    assert q['e_xl_seg3'] == pytest.approx(0.15) and q['rise_uninterp_seg3'] == 1
    assert q['frac90_seg3'] == 0.0
    assert q['e_xl_seg1'] == 0.0


def test_e_xl_uses_only_the_last_second_and_no_stop_tail():
    t = np.arange(0, 10.0, 0.01)
    v = np.where(t < 8.0, 0.3, 0.0)                                   # A-K2 seg3 level 0.3: drop in its last s
    v = np.where((t >= 3) & (t < 6), -0.3, v)
    v = np.where(t >= 9.0, 5.0, v)                                    # stop tail: not in any segment
    q = R.step_quantities(t, v, 'A-K2')
    assert q['e_xl_seg3'] == pytest.approx(0.3) and q['e_xl_seg2'] == pytest.approx(0.0)


def test_empty_segment_is_nan():
    t = np.arange(0, 3.0, 0.01)
    q = R.step_quantities(t, np.zeros(len(t)), 'A-K1')
    assert math.isnan(q['rise_s_seg2']) and math.isnan(q['e_xl_seg3'])


def test_strafe_moves_left_of_the_heading():
    """A pure lateral command (vy > 0) moves the pose to the left of the heading: +y at heading 0, -x at pi/2
    (the vy terms of the exact pose integration)."""
    u = np.tile([0.0, 0.2, 0.0], (50, 1))                              # 1 s of vy = 0.2 m/s
    for psi0, want in ((0.0, (0.0, 0.2)), (math.pi / 2, (-0.2, 0.0))):
        p = R.pose_ref('A-S1', [1.0], u=u, psi0=psi0)[0]
        assert (p[0], p[1]) == pytest.approx(want, abs=1e-12) and p[2] == pytest.approx(psi0)
    u = np.tile([0.0, 0.2, 0.5], (50, 1))                              # strafe while turning: an arc
    p = R.pose_ref('A-S1', [1.0], u=u, psi0=0.0)[0]
    assert p[0] == pytest.approx(-0.2 * (1 - math.cos(0.5)) / 0.5, abs=1e-12)
    assert p[1] == pytest.approx(0.2 * math.sin(0.5) / 0.5, abs=1e-12)


def test_frac90_weights_lines_by_the_time_they_hold():
    """Lines 10 ms apart before the response, then 0.5 s apart: frac90 is the time share (2.5 s of 3 s), not the
    line share."""
    t = np.r_[2.99, np.arange(300, 350) / 100.0, [3.5, 4.0, 4.5, 5.0, 5.5]]
    v = np.r_[0.0, np.zeros(50), np.full(5, 0.3)]
    q = R.step_quantities(t, v, 'A-K1')
    assert q['frac90_seg2'] == pytest.approx(2.5 / 3.0) and q['no90_seg2'] == 0


def test_frac90_start_of_the_segment_held_by_the_line_before():
    """S2M1-10 (Codex round 1 of v3): the first lines of the segment lost: [3.0, 3.4) is held by the last line
    before the segment (y = 0), so it is in the denominator; no line before the segment: frac90 not defined."""
    t = np.r_[2.99, [3.4, 4.0, 5.0]]
    v = np.r_[0.0, [0.3, 0.3, 0.3]]
    q = R.step_quantities(t, v, 'A-K1')
    assert q['frac90_seg2'] == pytest.approx(2.6 / 3.0)
    q = R.step_quantities(t[1:], v[1:], 'A-K1')
    assert math.isnan(q['frac90_seg2']) and q['no90_seg2'] == 0
