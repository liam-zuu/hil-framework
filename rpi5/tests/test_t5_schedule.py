"""t5_schedule.py (S06 T5 wheel-speed schedule, W15) and run_hil.py --omega-steps @CSV."""
import os

import pytest

import t5_schedule as t5
from plant_interface import OmegaProfilePlant
from run_hil import _parse_omega_steps


POINTS_V34 = (0.32, 0.5, 1.0, 2.0, 10.0, 20.0, 39.2)      # scored points before spec S06 v3.5


def _mags(sched):
    return [abs(v.w_fl) for v in sched if v.kind != 'w15']


def test_points_both_directions_alternating_and_indicator():
    for p in (1, 2):
        s = t5.make_schedule(p)
        pts = [v for v in s if v.kind != 'w15']
        assert sorted(set(_mags(s))) == sorted((t5.INDICATOR,) + t5.POINTS)
        for a, b in zip(pts[0::2], pts[1::2]):                 # + then - at each omega
            assert a.omegas == [abs(a.w_fl)] * 4 and b.omegas == [-abs(a.w_fl)] * 4
        assert [v.kind for v in pts if abs(v.w_fl) == t5.INDICATOR] == ['indicator'] * 2
        assert all(v.kind == 'point' for v in pts if abs(v.w_fl) != t5.INDICATOR)


def test_pass1_sweeps_up_pass2_down():
    up, down = _mags(t5.make_schedule(1))[0::2], _mags(t5.make_schedule(2))[0::2]
    assert up == sorted(up) and down == sorted(down, reverse=True) and up == down[::-1]


def test_w15_wheels_differ_and_are_scored_points():
    for p in (1, 2):
        w15 = [v for v in t5.make_schedule(p) if v.kind == 'w15']
        assert len(w15) == 1
        w = w15[0].omegas
        assert len({abs(x) for x in w}) == 4 and len(set(w)) == 4
        assert all(abs(x) in t5.POINTS for x in w)


def test_visits_contiguous_70s_and_csv_roundtrip(tmp_path):
    s = t5.make_schedule(1)
    assert s[0].t_start_ms == 0
    assert all(v.t_end_ms - v.t_start_ms == 70_000 for v in s)
    assert all(a.t_end_ms == b.t_start_ms for a, b in zip(s, s[1:]))
    path = os.path.join(tmp_path, 'T5_pass1.csv')
    t5.write_csv(path, s)
    assert t5.read_csv(path) == s
    segs = _parse_omega_steps('@' + path)
    assert segs == t5.segments(s) and len(segs) == len(s)
    assert segs[3] == ([-0.32] * 4, 70.0, False)


def test_profile_plant_plays_the_schedule(tmp_path):
    path = os.path.join(tmp_path, 'T5_pass2.csv')
    s = t5.make_schedule(2)
    t5.write_csv(path, s)
    plant = OmegaProfilePlant(_parse_omega_steps('@' + path))
    assert plant.total_steps == s[-1].t_end_ms
    first = plant.step([0.0] * 4)[:4]
    assert first == s[0].omegas


def test_logger_command_covers_schedule():
    s = t5.make_schedule(1)
    # spec S06 v3.5 X11: 25 visits per pass (2 indicator + 22 point + W15)
    assert len(s) == 25 and s[-1].t_end_ms == 1_750_000
    assert t5.logger_duration_s(s) == 1810
    assert '--profile heavy --duration 1810 ' in t5.logger_command(s)


def test_v35_eleven_points_both_directions():
    # spec S06 v3.5 X11: the 7 old points plus the 4 off-grid points (cau 7), both directions, scored
    assert t5.OFF_GRID_POINTS == (9.995, 19.979, 31.68, 38.78)
    assert set(t5.POINTS) == set(POINTS_V34) | set(t5.OFF_GRID_POINTS)
    for p in (1, 2):
        s = t5.make_schedule(p)
        pts = [v for v in s if v.kind == 'point']
        assert len(pts) == 22
        assert {abs(v.w_fl) for v in pts} == set(t5.POINTS)
        for m in t5.POINTS:
            assert sum(v.omegas == [m] * 4 for v in pts) == 1
            assert sum(v.omegas == [-m] * 4 for v in pts) == 1
        assert sum(v.kind == 'indicator' for v in s) == 2
        assert sum(v.kind == 'w15' for v in s) == 1
        assert len(s) == 25


def test_old_points_and_order_unchanged():
    # the v3.4 sequence (indicator pair, old points swept, W15 last) must not move
    assert t5.INDICATOR == 0.29 and t5.W15_OMEGAS == (1.0, -2.0, 10.0, -20.0) and t5.VISIT_WINDOWS == 7
    mags = (t5.INDICATOR,) + POINTS_V34
    for p, order in ((1, mags), (2, mags[::-1])):
        keep = [v for v in t5.make_schedule(p) if abs(v.w_fl) in mags or v.kind == 'w15']
        want = []
        for m in order:
            kind = 'indicator' if m == t5.INDICATOR else 'point'
            want += [(kind, [m] * 4), (kind, [-m] * 4)]
        want.append(('w15', list(t5.W15_OMEGAS)))
        assert [(v.kind, v.omegas) for v in keep] == want
    # and the new points sit in sorted position inside each sweep
    pos = [abs(v.w_fl) for v in t5.make_schedule(1) if v.kind != 'w15'][0::2]
    assert pos == sorted((t5.INDICATOR,) + t5.POINTS)


def test_off_grid_points_exact_through_csv_and_omega_steps(tmp_path):
    s = t5.make_schedule(1)
    path = os.path.join(tmp_path, 'T5_pass1.csv')
    t5.write_csv(path, s)
    back = t5.read_csv(path)
    for m in t5.OFF_GRID_POINTS:
        assert [v.w_fl for v in back].count(m) == 1
        assert [v.w_fl for v in back].count(-m) == 1
    segs = _parse_omega_steps('@' + path)
    for m in t5.OFF_GRID_POINTS:
        assert ([m] * 4, 70.0, False) in segs
        assert ([-m] * 4, 70.0, False) in segs


def test_gap_in_schedule_rejected():
    s = t5.make_schedule(1)
    s[2].t_start_ms += 1
    with pytest.raises(ValueError):
        t5.segments(s)
