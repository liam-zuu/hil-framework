"""Tests for rt_monitor.py and the plant loop without in-loop logging (THESIS H2)."""
import logging
import multiprocessing as mp

from hil_node import HILConfig, HILNode
from plant_interface import MockPlant
from rt_monitor import RTMonitor, N_BINS, I_STEP


class TestWindowStats:
    def test_percentiles_and_max_from_histogram(self):
        m = RTMonitor(warn_us=500)
        for k in range(1, 1001):
            m.record(k, 10_000 if k <= 990 else 600_000, overruns=0)   # 990 x 10 us, 10 x 600 us
        s = RTMonitor.window_stats(m.snapshot())
        assert s['n'] == 1000 and s['step'] == 1000
        assert s['p99_us'] == 10           # 990th sample
        assert s['p999_us'] == 600
        assert s['max_us'] == 600 and s['max_all_us'] == 600.0
        assert s['warn'] == 10

    def test_window_is_difference_of_snapshots(self):
        m = RTMonitor()
        for k in range(1, 101):
            m.record(k, 900_000, 0)
        old = m.snapshot()
        for k in range(101, 201):
            m.record(k, 5_000, 2)
        s = RTMonitor.window_stats(m.snapshot(), old)
        assert s['n'] == 100 and s['max_us'] == 5 and s['p99_us'] == 5
        assert s['max_all_us'] == 900.0      # all-time max keeps the first window's peak
        assert s['overruns'] == 2

    def test_lateness_beyond_scale_goes_to_last_bin(self):
        m = RTMonitor()
        m.record(1, 5_000_000, 1)
        s = RTMonitor.window_stats(m.snapshot())
        assert s['max_us'] == N_BINS - 1 and s['max_over_scale']
        assert s['max_all_us'] == 5_000.0


class TestNodeWithMonitor:
    def test_no_log_inside_loop_and_counts_every_tick(self, caplog):
        """With a monitor the loop logs only start/stop, even past stats_interval,
        and records every tick."""
        cfg = HILConfig(simulate_hw=True, max_steps=300, period_us=1_000, stats_interval=100,
                        jitter_warn_us=0.0)          # every tick 'late': old code would warn 300 times
        mon = RTMonitor(warn_us=cfg.jitter_warn_us)
        node = HILNode(MockPlant(), config=cfg, shared_state=mp.Array('d', 10), monitor=mon)
        with caplog.at_level(logging.INFO, logger='hil.node'):
            node.run()
        msgs = [r.getMessage() for r in caplog.records if r.name == 'hil.node']
        assert not any('step=' in m or 'Jitter overrun' in m for m in msgs), msgs
        snap = mon.snapshot()
        assert snap[I_STEP] == 300
        assert RTMonitor.window_stats(snap)['n'] == 300

    def test_without_monitor_old_behaviour(self, caplog):
        cfg = HILConfig(simulate_hw=True, max_steps=200, period_us=1_000, stats_interval=100)
        node = HILNode(MockPlant(), config=cfg)
        with caplog.at_level(logging.INFO, logger='hil.node'):
            node.run()
        assert sum('rate=' in r.getMessage() for r in caplog.records) == 2   # stats every 100 steps

    def test_monitor_survives_spawn(self):
        """The plant runs in a spawned process (run_hil.py): the shared array must reach it."""
        ctx = mp.get_context('spawn')
        mon = RTMonitor()
        p = ctx.Process(target=_child_record, args=(mon,))
        p.start()
        p.join(timeout=30)
        assert p.exitcode == 0
        assert mon.snapshot()[I_STEP] == 7


def _child_record(mon):
    mon.record(7, 3_000, 0)
