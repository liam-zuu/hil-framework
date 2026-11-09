"""t5_schedule.py (S06 T5 wheel-speed schedule, W15) and run_hil.py --omega-steps @CSV."""
import os

import pytest

import t5_schedule as t5
from plant_interface import OmegaProfilePlant
from run_hil import _parse_omega_steps


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
    assert len(s) == 17 and s[-1].t_end_ms == 1_190_000
    assert t5.logger_duration_s(s) == 1250
    assert '--profile heavy --duration 1250 ' in t5.logger_command(s)


def test_gap_in_schedule_rejected():
    s = t5.make_schedule(1)
    s[2].t_start_ms += 1
    with pytest.raises(ValueError):
        t5.segments(s)
