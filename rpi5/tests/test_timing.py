"""Tests for timing.py — PrecisionTimer and jitter measurement.

The two tests with a real-clock bound are a benchmark of the machine, not a check of the
code (B0 review B41): they run on Linux (the Pi, where pytest must pass them) and are
skipped elsewhere, where the scheduler tick (15 ms on Windows) decides the result."""
import sys
import time
import types

import pytest

import timing
from timing import PrecisionTimer, measure_jitter

on_linux_only = pytest.mark.skipif(
    not sys.platform.startswith('linux'),
    reason='real-clock timer bound: run on the Pi; other OS scheduler tick decides it (B41)')


class TestPrecisionTimer:
    def test_instantiate(self):
        t = PrecisionTimer(period_us=10_000)
        assert t is not None

    def test_wait_returns_int(self):
        t = PrecisionTimer(period_us=10_000)
        jitter = t.wait_next_tick()
        assert isinstance(jitter, int)

    @on_linux_only
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

    def test_stats_populated_after_ticks(self):
        t = PrecisionTimer(period_us=10_000)
        for _ in range(5):
            t.wait_next_tick()
        s = t.jitter_stats
        assert s["n"] == 5
        assert s["mean_us"] >= 0
        assert s["max_us"] >= 0
        assert s["p99_us"] >= 0

    @on_linux_only
    def test_jitter_reasonable_on_dev_machine(self):
        """On any machine jitter should be < 10ms (very loose bound)."""
        t = PrecisionTimer(period_us=10_000)
        for _ in range(20):
            t.wait_next_tick()
        assert t.jitter_stats["max_us"] < 10_000

    def test_reset_clears_history(self):
        t = PrecisionTimer(period_us=10_000)
        for _ in range(5):
            t.wait_next_tick()
        assert t.jitter_stats["n"] == 5
        t.reset()
        # After reset, deque is empty → jitter_stats returns {}
        assert t.jitter_stats == {}
        assert t._overruns == 0

    def test_overrun_counter_exists(self):
        t = PrecisionTimer(period_us=10_000)
        # overruns field only present after at least one tick
        for _ in range(3):
            t.wait_next_tick()
        assert "overruns" in t.jitter_stats

    def test_print_jitter_report_no_crash(self, capsys):
        t = PrecisionTimer(period_us=10_000)
        for _ in range(5):
            t.wait_next_tick()
        t.print_jitter_report()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestMeasureJitter:
    def test_measure_jitter_returns_dict(self):
        result = measure_jitter(n_ticks=10, period_us=10_000, rt=False)
        assert isinstance(result, dict)
        assert "max_us" in result
        assert "mean_us" in result
        assert "n" in result

    def test_measure_jitter_max_us_key(self):
        """max_us always present and non-negative."""
        result = measure_jitter(n_ticks=10, period_us=10_000, rt=False)
        assert result["max_us"] >= 0

    def test_rt_measurement_restores_scheduler_and_affinity(self, monkeypatch):
        """B41: rt=True changes the calling process for the measurement only."""
        state = {'policy': 0, 'prio': 0, 'aff': {0, 1, 2, 3}}
        fake_os = types.SimpleNamespace(
            sched_getscheduler=lambda pid: state['policy'],
            sched_getparam=lambda pid: types.SimpleNamespace(sched_priority=state['prio']),
            sched_getaffinity=lambda pid: set(state['aff']),
            sched_param=lambda prio: prio,
            sched_setscheduler=lambda pid, pol, prio: state.update(policy=pol, prio=prio),
            sched_setaffinity=lambda pid, aff: state.update(aff=set(aff)),
        )
        monkeypatch.setattr(timing, 'os', fake_os)
        monkeypatch.setattr(timing, 'set_realtime', lambda priority: state.update(policy=1, prio=priority))
        monkeypatch.setattr(timing, 'pin_cpu', lambda core: state.update(aff={core}))
        seen = {}
        orig = timing.PrecisionTimer.wait_next_tick
        def tick(self):
            seen.update(policy=state['policy'], aff=set(state['aff']))
            return orig(self)
        monkeypatch.setattr(timing.PrecisionTimer, 'wait_next_tick', tick)
        measure_jitter(n_ticks=3, period_us=10_000, rt=True)
        assert seen == {'policy': 1, 'aff': {0}}                 # changed while measuring
        assert state == {'policy': 0, 'prio': 0, 'aff': {0, 1, 2, 3}}   # put back afterwards
