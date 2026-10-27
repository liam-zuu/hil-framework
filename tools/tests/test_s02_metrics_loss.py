"""s02_metrics.log_loss: log loss from log_seq and '#run_end' lines (Codex review of
AGV ctrl-imu #5, Liam approved 28/10). Synthetic logs, no hardware.
Run: python -m pytest tools/tests"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import s02_metrics  # noqa: E402

HEAD = ['timestamp_us', 'tick', 'log_seq', 'motor_on']


def write_log(tmp_path, lines, run_end_last=None):
    """lines: (log_seq, motor_on); tick counts the motor-on lines."""
    p = tmp_path / 'run.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        tick = 0
        for seq, on in lines:
            tick += on
            w.writerow([seq * 10_000, tick, seq, on])
    if run_end_last is not None:
        with open(str(p) + '.events.txt', 'w') as f:
            for k in range(3):
                f.write(f'2026-10-28T20:00:00 #run_end,seq={run_end_last + 1 + k},'
                        f'last_data_seq={run_end_last},log_qdrop=0,log_send_err=0\n')
    return str(p)


def test_no_loss(tmp_path):
    lines = [(s, 0) for s in range(0, 5)] + [(s, 1) for s in range(5, 105)] + [(s, 0) for s in range(108, 112)]
    log = s02_metrics.load_log(write_log(tmp_path, lines, run_end_last=104))
    assert s02_metrics.log_loss(log) == (0.0, 1)


def test_lost_end_of_run_is_counted(tmp_path):
    """The last 10 data lines never arrive; the #run_end line says the run ended at 104.
    The tick method would see 0 % here."""
    lines = [(s, 1) for s in range(5, 95)]
    log = s02_metrics.load_log(write_log(tmp_path, lines, run_end_last=104))
    pct, tail = s02_metrics.log_loss(log)
    assert pct == pytest.approx(10 / 100 * 100)
    assert tail == 1


def test_gap_in_middle_and_no_events_file(tmp_path):
    lines = [(s, 1) for s in range(0, 100) if not 40 <= s < 45]
    log = s02_metrics.load_log(write_log(tmp_path, lines))
    pct, tail = s02_metrics.log_loss(log)
    assert pct == pytest.approx(5.0)
    assert tail == 0


def test_old_format_uses_ticks(tmp_path):
    p = tmp_path / 'old.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_us', 'tick'])
        for t in [1, 2, 3, 5]:
            w.writerow([t * 10_000, t])
    pct, tail = s02_metrics.log_loss(s02_metrics.load_log(str(p)))
    assert pct == pytest.approx(20.0) and tail == 0
