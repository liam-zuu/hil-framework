"""Tests for timing.py — PrecisionTimer and jitter measurement.

The timer algorithm is checked with a fake clock and sleep on every OS (B0 review round 3
N18): deadlines, sleep length, busy-wait, overruns, catch-up, reset. The two tests with a
real-clock bound are a benchmark of the machine, not a check of the code (B41): opt-in,
run on the Pi with HIL_TIMING_BENCH=1 and the machine state written down with the result
(kernel, load, privileges); they do not prove anything on another Linux."""
import os
import sys
import time
import types

import pytest

import timing
from timing import PrecisionTimer, jitter_ok, measure_jitter

bench = pytest.mark.skipif(
    os.environ.get('HIL_TIMING_BENCH') != '1' or not sys.platform.startswith('linux'),
    reason='real-clock benchmark: opt-in on the Pi (HIL_TIMING_BENCH=1), B41 / N18')


class FakeClock:
    """Monotonic ns clock that moves only when read (step ns per read) or slept."""

    def __init__(self, now=0, step=1_000):
        self.now = now
        self.step = step
        self.sleeps = []

    def __call__(self):
        self.now += self.step
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += int(round(seconds * 1e9))


def fake_timer(period_us=1_000, **kw):
    c = FakeClock(**kw)
    return PrecisionTimer(period_us=period_us, clock=c, sleep=c.sleep), c


class TestPrecisionTimerFakeClock:
    def test_sleeps_until_busy_wait_then_wakes_at_the_deadline(self):
        t, c = fake_timer()
        first = t._next_ns
        jitter = t.wait_next_tick()
        # one sleep up to BUSY_WAIT_NS before the deadline, the rest busy-waited
        assert len(c.sleeps) == 1
        assert c.sleeps[0] == pytest.approx((first - timing.BUSY_WAIT_NS - (first - 1_000_000 + 1_000)) * 1e-9)
        assert 0 <= jitter <= 2 * c.step
        assert t._next_ns == first + 1_000_000

    def test_deadlines_are_fixed_rate(self):
        t, c = fake_timer()
        start = t._next_ns
        for k in range(5):
            t.wait_next_tick()
            assert t._next_ns == start + (k + 1) * 1_000_000

    def test_overrun_counts_and_returns_late_without_sleeping(self):
        t, c = fake_timer()
        c.now += 2_500_000                      # the caller's work took 2.5 periods
        jitter = t.wait_next_tick()
        assert c.sleeps == [] and jitter > t.period_ns and t.overruns == 1

    def test_catches_up_after_an_overrun(self):
        """After a late tick the next deadlines stay on the original grid: the ticks already
        due return at once (no sleep), then the timer sleeps again."""
        t, c = fake_timer()
        start = t._next_ns
        c.now += 3_500_000
        for _ in range(3):
            t.wait_next_tick()                  # deadlines start, +1, +2 periods: all past
        assert c.sleeps == []
        t.wait_next_tick()                      # deadline start + 3 periods: in the future
        assert len(c.sleeps) == 1 and t._next_ns == start + 4 * 1_000_000

    def test_reset_rearms_from_now(self):
        t, c = fake_timer()
        t.wait_next_tick()
        c.now += 10_000_000
        t.reset()
        assert t.jitter_stats == {} and t.overruns == 0
        assert c.now < t._next_ns <= c.now + 1_000_000 + c.step

    def test_stats_from_the_jitter_history(self):
        t, c = fake_timer()
        for _ in range(5):
            t.wait_next_tick()
        s = t.jitter_stats
        assert s['n'] == 5 and s['overruns'] == 0 and 0 <= s['max_us'] <= 2 * c.step / 1000


class TestPrecisionTimer:
    def test_wait_returns_int(self):
        t = PrecisionTimer(period_us=10_000)
        assert isinstance(t.wait_next_tick(), int)

    @bench
    def test_period_accuracy_100hz(self):
        """100Hz = 10ms period. 20 ticks should take ~200ms ±10%."""
        t = PrecisionTimer(period_us=10_000)
        n = 20
        t0 = time.monotonic()
        for _ in range(n):
            t.wait_next_tick()
        elapsed = time.monotonic() - t0
        expected = n * 0.010
        assert abs(elapsed - expected) / expected < 0.10, (
            f"Elapsed {elapsed:.3f}s vs expected {expected:.3f}s"
        )

    @bench
    def test_jitter_reasonable_on_the_pi(self):
        t = PrecisionTimer(period_us=10_000)
        for _ in range(20):
            t.wait_next_tick()
        assert t.jitter_stats["max_us"] < 10_000

    def test_print_jitter_report_no_crash(self, capsys):
        t, _ = fake_timer()
        for _ in range(5):
            t.wait_next_tick()
        t.print_jitter_report()
        assert len(capsys.readouterr().out) > 0


def _fake_sched(monkeypatch, fail=()):
    """Fake os.sched_* on the timing module; fail: names of calls that raise OSError."""
    state = {'policy': 0, 'prio': 0, 'aff': {0, 1, 2, 3}}

    def maybe(name, fn):
        def call(*a):
            if name in fail:
                raise OSError(f'{name} failed')
            return fn(*a)
        return call
    fake_os = types.SimpleNamespace(
        sched_getscheduler=maybe('getscheduler', lambda pid: state['policy']),
        sched_getparam=maybe('getparam', lambda pid: types.SimpleNamespace(sched_priority=state['prio'])),
        sched_getaffinity=maybe('getaffinity', lambda pid: set(state['aff'])),
        sched_param=lambda prio: prio,
        sched_setscheduler=maybe('setscheduler', lambda pid, pol, prio: state.update(policy=pol, prio=prio)),
        sched_setaffinity=maybe('setaffinity', lambda pid, aff: state.update(aff=set(aff))),
    )
    monkeypatch.setattr(timing, 'os', fake_os)

    def set_rt(priority):
        state.update(policy=1, prio=priority)
        return True

    def pin(core):
        state.update(aff={core})
        return True
    monkeypatch.setattr(timing, 'set_realtime', set_rt)
    monkeypatch.setattr(timing, 'pin_cpu', pin)
    return state


ORIGINAL = {'policy': 0, 'prio': 0, 'aff': {0, 1, 2, 3}}


class TestMeasureJitter:
    def test_measure_jitter_returns_dict(self):
        result = measure_jitter(n_ticks=10, period_us=10_000, rt=False)
        assert {'max_us', 'mean_us', 'n'} <= set(result)
        assert result['rt'] == {'scheduler': False, 'affinity': False} and result['restore_errors'] == []

    def test_rt_measurement_restores_scheduler_and_affinity(self, monkeypatch):
        """B41: rt=True changes the calling thread for the measurement only."""
        state = _fake_sched(monkeypatch)
        seen = {}
        orig = timing.PrecisionTimer.wait_next_tick

        def tick(self):
            seen.update(policy=state['policy'], aff=set(state['aff']))
            return orig(self)
        monkeypatch.setattr(timing.PrecisionTimer, 'wait_next_tick', tick)
        r = measure_jitter(n_ticks=3, period_us=10_000, rt=True)
        assert seen == {'policy': 1, 'aff': {0}}                 # changed while measuring
        assert state == ORIGINAL                                 # put back afterwards
        assert r['rt'] == {'scheduler': True, 'affinity': True} and r['restore_errors'] == []

    def test_failed_scheduler_restore_still_restores_affinity_codex_n17(self, monkeypatch):
        state = _fake_sched(monkeypatch)
        monkeypatch.setattr(timing.os, 'sched_setscheduler',
                            lambda *a: (_ for _ in ()).throw(OSError('EPERM')))
        r = measure_jitter(n_ticks=3, period_us=10_000, rt=True)
        assert state['aff'] == ORIGINAL['aff']
        assert len(r['restore_errors']) == 1 and r['restore_errors'][0].startswith('scheduler')
        assert not jitter_ok(dict(r, max_us=1.0))                # --jitter-only exits 1

    def test_failed_affinity_restore_is_reported(self, monkeypatch):
        state = _fake_sched(monkeypatch, fail=('setaffinity',))
        monkeypatch.setattr(timing, 'pin_cpu', lambda core: state.update(aff={core}) or True)
        r = measure_jitter(n_ticks=3, period_us=10_000, rt=True)
        assert (state['policy'], state['prio']) == (0, 0)
        assert [e.split(':')[0] for e in r['restore_errors']] == ['affinity']

    def test_unreadable_scheduler_is_not_changed_codex_n17(self, monkeypatch):
        state = _fake_sched(monkeypatch, fail=('getscheduler',))
        seen = {}
        orig = timing.PrecisionTimer.wait_next_tick

        def tick(self):
            seen.update(policy=state['policy'], aff=set(state['aff']))
            return orig(self)
        monkeypatch.setattr(timing.PrecisionTimer, 'wait_next_tick', tick)
        r = measure_jitter(n_ticks=3, period_us=10_000, rt=True)
        assert seen == {'policy': 0, 'aff': {0}}                 # only the affinity changed
        assert r['rt'] == {'scheduler': False, 'affinity': True} and state['aff'] == ORIGINAL['aff']

    def test_exception_while_measuring_still_restores(self, monkeypatch):
        state = _fake_sched(monkeypatch)

        def boom(self):
            raise RuntimeError('tick failed')
        monkeypatch.setattr(timing.PrecisionTimer, 'wait_next_tick', boom)
        with pytest.raises(RuntimeError):
            measure_jitter(n_ticks=3, period_us=10_000, rt=True)
        assert state == ORIGINAL

    def test_jitter_ok_needs_budget_and_restore(self):
        assert jitter_ok({'max_us': 100.0, 'restore_errors': []})
        assert not jitter_ok({'max_us': 600.0, 'restore_errors': []})
        assert not jitter_ok({'max_us': 100.0, 'restore_errors': ['scheduler: x']})
        assert not jitter_ok({})
