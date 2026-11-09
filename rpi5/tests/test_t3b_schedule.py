"""S06 T3b schedule (t3b_schedule.py): structure of the schedule, playback, and
N_tol / u_t limit against the worked examples of spec S06 v3.4 T3b."""
import math

import pytest

from protocol import encode_state, decode_state
import t3b_schedule as t3


@pytest.fixture(scope='module')
def sched():
    return t3.make_schedule(seed=1)


def _one(kind, w_from, w_to, shape, tau=0.0, dur=0, delay=0):
    """A single transition on wheel 0 after a 1 s run-in, like the generator lays it."""
    t = 1000
    t0 = t + t3.PREP_MS + delay
    te = t0 + dur
    return t3.Transition(0, 0, kind, w_from, w_to, shape, tau, delay, t, t0, te,
                         t0 - 100, te + 300, te + t3.POST_MS)


def _tol(row, before=None):
    table = t3.omega_table([row])
    w = table[0]
    if before is not None:                       # level held before the slot
        for k in range(row.t_prep_ms):
            w[k] = before
    return t3.tolerance(row, [t3.quantize(v) for v in w])


def test_counts_per_wheel(sched):
    for wheel in range(4):
        rows = [r for r in sched if r.wheel == wheel]
        assert len(rows) == len(t3.KINDS) * t3.REPS + 8 * t3.EXP_REPS     # 40 + 80
        exp = [r for r in rows if r.shape == 'exp']
        combos = {(r.tau_s, r.w_to) for r in exp}
        assert len(combos) == 8


def test_random_delay_only_for_starts_from_rest(sched):
    for r in sched:
        if r.w_from == 0.0:
            assert 0 <= r.delay_ms <= t3.DELAY_MAX_MS
        else:
            assert r.delay_ms == 0
    assert len({r.delay_ms for r in sched if r.w_from == 0.0}) > 20     # not all the same


def test_wheels_have_different_orders(sched):
    orders = [[r.kind for r in sched if r.wheel == w] for w in range(4)]
    assert len({tuple(o) for o in orders}) == 4


def test_one_transition_per_window(sched):
    for w in range(4):
        rows = [r for r in sched if r.wheel == w]
        for r in rows:
            for o in rows:
                if o is r:
                    continue
                # no other slot's preparation step or transition inside this window
                assert not (r.win_start_ms <= o.t_prep_ms <= r.win_end_ms)
                assert not (r.win_start_ms <= o.t0_ms <= r.win_end_ms)
            assert r.t_prep_ms < r.win_start_ms


def test_seed_reproducible():
    a = t3.make_schedule(seed=7)
    b = t3.make_schedule(seed=7)
    c = t3.make_schedule(seed=8)
    assert a == b and a != c


def test_quantize_is_the_frame(sched):
    for w in (0.0, 0.3, 0.31983, 2.0, 5.0, 12.5, 39.2, -10.0, 0.4):
        frame = encode_state([w, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert decode_state(frame)[0] == pytest.approx(t3.quantize(w), abs=1e-12)


def test_csv_round_trip(tmp_path, sched):
    p = tmp_path / 'T3b_schedule.csv'
    t3.write_csv(str(p), sched)
    assert t3.read_csv(str(p)) == sched


def test_plant_plays_the_table(sched):
    plant = t3.ScheduleOmegaPlant(sched)
    table = t3.omega_table(sched)
    for k in range(3000):
        s = plant.step([0.0] * 4)
        assert s[0:4] == [table[w][k] for w in range(4)]
    assert plant.total_steps == max(r.slot_end_ms for r in sched)


# Worked examples of the spec (T3b, "Ví dụ (Claude tính, K ≈ 210,08)")
def test_example_ramp():
    t = _tol(_one('ramp', 0.4, 10.0, 'ramp', dur=300), before=0.4)
    assert t['n_tol'] == pytest.approx(8.82, abs=0.02)
    assert t['n_start'] == 0 and t['n_stop_intervals'] == 0


def test_example_stop_and_tightest_time_limit():
    t = _tol(_one('stop', 2.0, 0.0, 'step'), before=2.0)
    assert t['n_tol'] == pytest.approx(3.67, abs=0.01)
    assert t['n_stop_intervals'] == 1 and t['n_start'] == 0
    # spec: (read span + sync) <= 0.168 / (210.08 * 2) s = 0.40 ms, only the start end counts
    assert t['t_allow_ms_per_end'] == pytest.approx(0.40, abs=0.005)


def test_example_exp_start():
    t = _tol(_one('exp', 0.0, 10.0, 'exp', tau=0.05, dur=250), before=0.0)
    assert t['n_tol'] == pytest.approx(12.4, abs=0.05)
    assert t['n_start'] == 1 and t['n_stop_intervals'] == 1


def test_example_step_up():
    t = _tol(_one('step', 2.0, 39.2, 'step'), before=2.0)
    assert t['n_tol'] == pytest.approx(24.0, abs=0.1)


def test_reversal_is_a_stop_and_a_start():
    t = _tol(_one('rev', 10.0, -10.0, 'step'), before=10.0)
    assert t['n_start'] == 1 and t['n_stop_intervals'] == 1


def test_all_windows_have_finite_tolerances(sched):
    for t in t3.tolerances(sched[:40]):
        assert t['n_tol'] > 1.0 and math.isfinite(t['n_tol'])
        assert t['u_t_limit'] >= 0.0
