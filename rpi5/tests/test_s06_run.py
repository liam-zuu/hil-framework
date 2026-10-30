"""S06 on the RPi5 side (29/10): frame log of raw MOSI/MISO with send times,
#run_start / #run_end lines, T3b schedule played through the real HIL loop
(simulate_hw=True, no hardware), trajectory publisher loop mode."""
import gc
import json
import logging
import time
import types

import numpy as np
import pytest

import gpio_sync
import hil_node as hn
from hil_node import HILConfig, HILNode
from protocol import decode_state, encode_state
from trajectory_pub import TrajectorySpec
import t3b_schedule as t3


def _run(tmp_path, sched, steps, caplog):
    plant = t3.ScheduleOmegaPlant(sched)
    log_path = str(tmp_path / 'frames.npz')
    cfg = HILConfig(simulate_hw=True, max_steps=steps, period_us=1_000, frame_log=log_path)
    node = HILNode(plant, config=cfg)
    # simulated SPI echoes MOSI; answer with different bytes so MISO is told apart
    node._spi.transfer = lambda mosi: bytes(b ^ 0x5A for b in mosi)
    with caplog.at_level(logging.INFO):
        node.run()
    return np.load(log_path), caplog.text


def test_frame_log_and_run_lines(tmp_path, caplog):
    sched = t3.make_schedule(seed=3)
    steps = 1500
    d, text = _run(tmp_path, sched, steps, caplog)
    assert d['mosi'].shape == (steps, 40) and d['miso'].shape == (steps, 40)
    t = d['t_send_ns']
    assert np.all(np.diff(t) >= 0)                    # one send time per frame (Windows clock: ~15 ms steps)
    # sync pulse bracketed just before the send (S06 spec line 111, P_send := t_pulse_rise)
    assert np.all(d['t_pulse0_ns'] <= d['t_pulse1_ns']) and np.all(d['t_pulse1_ns'] <= t)
    assert np.all(d['t_pulse0_ns'] > 0)
    # the frame sent at step k carries the plant state before step k:
    # omega of the schedule at ms k-1 (0 at k = 0)
    table = t3.omega_table(sched)
    for k in (0, 1, 650, 700, 1499):
        got = decode_state(bytes(d['mosi'][k]))[:4]
        want = [0.0] * 4 if k == 0 else [table[w][k - 1] for w in range(4)]
        assert got == pytest.approx(want, abs=40.0 / 32767)
        assert d['mosi'][k][21] == k & 0xFF            # seq byte
        assert bytes(d['miso'][k]) == bytes(x ^ 0x5A for x in bytes(d['mosi'][k]))
    assert '#run_start,timestamp_us=' in text
    assert f'#run_end,timestamp_us=' in text and f'steps={steps},reason=max_steps' in text
    t_start = int(text.split('#run_start,timestamp_us=')[1].split(',')[0])
    assert t_start <= t[0] // 1000                    # same clock as t_send_ns
    # run manifest saved with the frames (Codex Sol round 3 S57, S64): what the scoring tools check
    m = json.loads(str(d['manifest']))
    assert (m['step_first'], m['step_last'], m['steps']) == (0, steps - 1, steps)
    assert m['period_us'] == 1_000 and m['stop_reason'] == 'max_steps' and m['imu_profile'] is None
    assert m['imu_source'] == 'none'                  # S73: the source really used, not the one requested
    assert m['rt']['gc_off'] is True and set(m['rt']) == {'sched_fifo', 'cpu_pinned', 'mem_locked', 'gc_off'}


def _run_noting_gc(tmp_path, monkeypatch, **cfg):
    seen = {}
    plant = t3.ScheduleOmegaPlant(t3.make_schedule(seed=3))
    node = HILNode(plant, config=HILConfig(simulate_hw=True, max_steps=20, period_us=1_000,
                                           frame_log=str(tmp_path / 'f.npz'), **cfg))
    tick = node._tick

    def noted(jitter_ns):
        seen['gc_on'], seen['frozen'] = gc.isenabled(), gc.get_freeze_count()
        tick(jitter_ns)
    monkeypatch.setattr(node, '_tick', noted)
    calls = []
    monkeypatch.setattr(hn, 'lock_memory', lambda: calls.append(1) or True)
    freeze = gc.freeze
    monkeypatch.setattr(gc, 'freeze', lambda: (seen.update(froze=True), freeze())[1])
    node.run()
    return seen, calls, json.loads(str(np.load(tmp_path / 'f.npz')['manifest']))


def test_rt_tune_around_the_loop(tmp_path, monkeypatch):
    # nhóm A: memory locked and no GC pass inside the loop, the GC back after it
    assert gc.isenabled()
    seen, calls, m = _run_noting_gc(tmp_path, monkeypatch)
    assert calls == [1] and not seen['gc_on'] and seen.get('froze') and seen['frozen'] > 0
    assert m['rt']['mem_locked'] is True and m['rt']['gc_off'] is True
    assert gc.isenabled() and gc.get_freeze_count() == 0


def test_rt_tune_off_for_comparison(tmp_path, monkeypatch):
    seen, calls, m = _run_noting_gc(tmp_path, monkeypatch, rt_tune=False)
    assert calls == [] and seen['gc_on'] and not seen.get('froze')
    assert m['rt']['mem_locked'] is False and m['rt']['gc_off'] is False


def test_frame_log_pages_touched_before_the_loop(monkeypatch):
    # np.zeros maps pages lazily; fill(0) writes them
    filled = []
    zeros = np.zeros

    class Arr(np.ndarray):
        def fill(self, v):
            filled.append(self.shape)
            super().fill(v)
    monkeypatch.setattr(np, 'zeros', lambda *a, **k: zeros(*a, **k).view(Arr))
    plant = t3.ScheduleOmegaPlant(t3.make_schedule(seed=3))
    HILNode(plant, config=HILConfig(simulate_hw=True, max_steps=10, period_us=1_000, frame_log='x.npz'))
    assert sorted(filled) == [(10,), (10,), (10,), (10, 40), (10, 40)]


def test_sync_pulse_busy_waits(monkeypatch):
    # nhóm A: no time.sleep in the 1 kHz loop; the pin stays high for the whole pulse width
    writes = []
    fake = types.SimpleNamespace(gpio_write=lambda h, pin, v: writes.append((v, time.monotonic_ns())))
    monkeypatch.setattr(gpio_sync, '_lgpio', fake)
    monkeypatch.setattr(gpio_sync.time, 'sleep', lambda s: pytest.fail('time.sleep in the pulse'))
    g = gpio_sync.GPIOSync(simulate=True)
    g.simulate, g._handle = False, 1
    g.pulse()
    g._handle, g.simulate = None, True
    assert [v for v, _ in writes] == [1, 0] and writes[1][1] - writes[0][1] >= gpio_sync._PULSE_WIDTH_US * 1_000


def test_frame_log_cap_stops_the_run(tmp_path, caplog):
    plant = t3.ScheduleOmegaPlant(t3.make_schedule(seed=3))
    cfg = HILConfig(simulate_hw=True, max_steps=0, period_us=1_000,
                    frame_log=str(tmp_path / 'f.npz'), frame_log_max=200)
    with caplog.at_level(logging.INFO):
        HILNode(plant, config=cfg).run()
    d = np.load(str(tmp_path / 'f.npz'))
    assert d['mosi'].shape[0] == 200 and json.loads(str(d['manifest']))['stop_reason'] == 'frame_log_full'
    assert 'reason=frame_log_full' in caplog.text


def test_publisher_loop_flag():
    assert TrajectorySpec(kind='line', duration=1.0, loop=True).loop
    assert not TrajectorySpec().loop
