"""Tests for timing.py — PrecisionTimer and jitter measurement."""
import time
import pytest
from timing import PrecisionTimer, measure_jitter


class TestPrecisionTimer:
    def test_instantiate(self):
        t = PrecisionTimer(period_us=10_000)
        assert t is not None

    def test_wait_returns_int(self):
        t = PrecisionTimer(period_us=10_000)
        jitter = t.wait_next_tick()
        assert isinstance(jitter, int)

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
        result = measure_jitter(n_ticks=10, period_us=10_000)
        assert isinstance(result, dict)
        assert "max_us" in result
        assert "mean_us" in result
        assert "n" in result

    def test_measure_jitter_max_us_key(self):
        """max_us always present and non-negative."""
        result = measure_jitter(n_ticks=10, period_us=10_000, rt=False)
        assert result["max_us"] >= 0
