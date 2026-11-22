"""t5_schedule.py (S06 T5 wheel-speed schedule, W15) and run_hil.py --omega-steps @CSV."""
import math
import os
import struct
import sys

import numpy as np
import pytest

import protocol
import t5_schedule as t5
from plant_interface import OmegaProfilePlant
from run_hil import _parse_omega_steps


POINTS_V34 = (0.32, 0.5, 1.0, 2.0, 10.0, 20.0, 39.2)      # scored points before spec S06 v3.5
F = np.float32


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
    # spec S06 v3.5 X11: the 7 old points plus the 4 off-grid points (cau 7), both directions, scored;
    # C3-18: 9.995 of the spec moved to 9.9942 (int16 8187, "chinh omega dat neu can", CC1-10)
    assert t5.OFF_GRID_POINTS == (9.9942, 19.979, 31.68, 38.78)
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


# C3-18 (Codex vong 1): reference values computed here, independently of t5_schedule: int16 = the big-endian
# bytes 0-7 of protocol.encode_state; H7 decode (float)raw * (40.0f / 32767.0f) (spi1_handler.c:116);
# half period 1e6f * (float)M_PI / (|omega| * 330.0f), then (uint32_t)(hp + 0.5f) (encoder_gen.c:45-49)
def _frame_raw(w):
    return struct.unpack('>4h', protocol.encode_state([w] * 4 + [0.0] * 6)[:8])


def _h7(raw):
    wd = F(F(raw) * F(F(40.0) / F(32767.0)))
    hp = F(F(F(1e6) * F(1.0)) * F(math.pi)) / F(F(abs(wd)) * F(330.0))
    timer = 65535 if hp > F(65535.0) else 1 if hp < F(1.0) else int(F(hp + F(0.5)))
    return wd, hp, timer


def test_c3_18_point_9995_moved_to_code_8187():
    # 9.995 gives int16 8188 (frac of the H7 half period 0.434); 8187 gives 0.550, the nearest to 0.5
    m = t5.OFF_GRID_POINTS[0]
    assert m == 9.9942 and 9.995 not in t5.POINTS and m in t5.POINTS
    for s in (1, -1):
        assert _frame_raw(s * m) == (s * 8187,) * 4
        for d in (-1e-9, 1e-9):                                  # 1e-9 off the set value: same code
            assert _frame_raw(s * m + d) == (s * 8187,) * 4
    fr = {c: float(_h7(c)[1]) % 1.0 for c in (8186, 8187, 8188, 8189)}
    assert min(fr, key=lambda c: abs(fr[c] - 0.5)) == 8187
    assert fr[8187] == pytest.approx(0.5502, abs=1e-4) and fr[8188] == pytest.approx(0.4338, abs=1e-4)
    assert _h7(8187)[2] == 953 and _h7(8188)[2] == 952


def test_c3_18_lock_matches_frame_and_h7_both_signs():
    # int16, omega_d (H7 float32), half period before / after rounding are stored and match the frame
    assert tuple(t5.OFF_GRID_LOCK) == t5.OFF_GRID_POINTS
    assert t5.OFF_GRID_LOCK[9.9942] == (8187, 9.99420166015625, 952.5501708984375, 953)
    for m, (raw, wd, hp, timer) in t5.OFF_GRID_LOCK.items():
        for s in (1, -1):
            assert _frame_raw(s * m) == (s * raw,) * 4
            w, h, tm = _h7(s * raw)
            assert float(w) == s * wd and float(h) == hp and tm == timer
            assert t5.h7_off_grid(s * m) == (s * raw, s * wd, hp, timer)
            assert 0.4 <= hp % 1.0 <= 0.6
    t5.check_off_grid_lock()


def test_c3_18_h7_timer_rounding_edges():
    # (uint32_t)(hp + 0.5f) after the clamp [1, 65535]: half up, float32 neighbour below stays down
    assert t5.h7_timer(952.5) == 953
    assert t5.h7_timer(float(np.nextafter(F(952.5), F(0)))) == 952
    assert t5.h7_timer(952.5501708984375) == 953 and t5.h7_timer(952.433837890625) == 952
    assert t5.h7_timer(65535.0) == 65535 and t5.h7_timer(65535.6) == 65535 and t5.h7_timer(7e4) == 65535
    assert t5.h7_timer(1.0) == 1 and t5.h7_timer(0.7) == 1 and t5.h7_timer(1.5) == 2


@pytest.mark.parametrize('case', ['old_code_8188', 'timer_cut_no_half', 'point_not_locked'])
def test_c3_18_wrong_lock_stops_schedule(monkeypatch, tmp_path, case):
    out = tmp_path / 'T5_pass1.csv'
    monkeypatch.setattr(sys, 'argv', ['t5_schedule.py', '--pass', '1', '--out', str(out)])
    assert t5.main() == 0 and out.exists()
    out.unlink()
    bad = {'old_code_8188': {**t5.OFF_GRID_LOCK, 9.9942: (8188, 9.99542236328125, 952.433837890625, 952)},
           'timer_cut_no_half': {**t5.OFF_GRID_LOCK, 9.9942: (8187, 9.99420166015625, 952.5501708984375, 952)},
           'point_not_locked': {k: v for k, v in t5.OFF_GRID_LOCK.items() if k != 38.78}}[case]
    monkeypatch.setattr(t5, 'OFF_GRID_LOCK', bad)
    with pytest.raises(ValueError):
        t5.main()
    assert not out.exists()


MAGS_C3_18 = (0.29, 0.32, 0.5, 1.0, 2.0, 9.9942, 10.0, 19.979, 20.0, 31.68, 38.78, 39.2)


def test_c3_18_full_25_visits_each_pass():
    # every visit of both passes: up / down, + then -, indicator 0.29 first (pass 1) / last (pass 2), W15 last
    for p, mags in ((1, MAGS_C3_18), (2, MAGS_C3_18[::-1])):
        want = []
        for m in mags:
            kind = 'indicator' if m == 0.29 else 'point'
            want += [(kind, [m] * 4), (kind, [-m] * 4)]
        want.append(('w15', [1.0, -2.0, 10.0, -20.0]))
        s = t5.make_schedule(p)
        assert len(want) == 25 and [(v.kind, v.omegas) for v in s] == want
        assert [v.visit for v in s] == list(range(25)) and all(v.run_pass == p for v in s)
