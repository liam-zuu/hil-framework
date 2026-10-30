"""S06 on the RPi5 side (29/10): frame log of raw MOSI/MISO with send times,
#run_start / #run_end lines, T3b schedule played through the real HIL loop
(simulate_hw=True, no hardware), trajectory publisher loop mode."""
import json
import logging

import numpy as np
import pytest

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
