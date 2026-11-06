"""run_validity.pos_reset_mid and its use in s02_metrics (STATUS_CODE L37: a run whose pose
reset in the middle is broken, run again). Synthetic logs, no hardware.
Run: python -m pytest tools/tests"""
import csv
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import s02_metrics  # noqa: E402
from agv_logger import COLUMNS  # noqa: E402
from run_validity import pos_reset_mid, run_bad_fields  # noqa: E402

nan = float('nan')


def test_constant_counter_is_zero():
    assert pos_reset_mid([1] * 10, [3] * 10) == 0


def test_reset_at_start_is_not_counted():
    # pos_step increments before the first active line is logged: the line before shows the old value
    assert pos_reset_mid([0, 0, 1, 1, 1, 0], [2, 2, 3, 3, 3, 3]) == 0


def test_mid_run_reset_counted():
    # packets stale (run_active 0) then fresh again: pose reset, counter +1
    ra = [1, 1, 1, 0, 0, 1, 1, 1]
    rn = [1, 1, 1, 1, 1, 2, 2, 2]
    assert pos_reset_mid(ra, rn) == 1


def test_two_mid_run_resets():
    assert pos_reset_mid([1, 0, 1, 0, 1], [5, 5, 6, 6, 7]) == 2


def test_rise_after_the_last_logged_active_line_is_counted():
    # Codex S3C-7: the active tick that raised the counter was lost, the inactive lines after it
    # still carry 2: one log = one run, so this is a reset inside the run
    assert pos_reset_mid([1, 1, 0, 0], [1, 1, 1, 2]) == 1


def test_motor_off_then_a_reset_is_refused_as_two_runs():
    # S3C-8: run, motor off, new run (reset): the log may hold two runs; do not merge them
    with pytest.raises(ValueError, match='more than one run'):
        pos_reset_mid([1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 2, 2], motor_on=[1, 1, 0, 0, 1, 1])


def test_rise_already_at_the_first_motor_off_line_is_counted_not_refused():
    # Codex S3D-2: the active tick that raised the counter was lost just before the motor went
    # off; the off lines carry the new value from the first one on: one mid-run reset, no error
    assert pos_reset_mid([1, 0, 0], [1, 2, 2], motor_on=[1, 0, 0]) == 1


def test_motor_off_at_the_end_is_fine():
    assert pos_reset_mid([1, 1, 1, 0, 0], [1, 1, 1, 1, 1], motor_on=[1, 1, 1, 0, 0]) == 0
    # a reset before the motor goes off is still a mid-run reset
    assert pos_reset_mid([1, 0, 1, 0, 0], [1, 1, 2, 2, 2], motor_on=[1, 1, 1, 0, 0]) == 1


def test_lost_lines_do_not_hide_a_reset():
    # the stale stretch itself was never logged (empty fields): the counter still shows it
    assert pos_reset_mid([1, 1, nan, 1, 1], [1, 1, nan, 2, 2]) == 1


@pytest.mark.parametrize('ra, rn', [(None, [1, 1]), ([1, 1], None), ([nan, nan], [1, 1]),
                                    ([1, 1], [nan, nan]), ([0, 0, 0], [0, 0, 0])])
def test_not_checkable(ra, rn):
    assert pos_reset_mid(ra, rn) is None


def test_counter_going_back_is_refused():
    with pytest.raises(ValueError, match='goes back'):
        pos_reset_mid([1, 1, 1], [4, 4, 1])


def test_run_bad_fields():
    assert run_bad_fields(0) == dict(pos_reset_mid=0, run_bad=0, run_bad_reason='')
    f = run_bad_fields(2)
    assert f['run_bad'] == 1 and f['pos_reset_mid'] == 2 and 'run again' in f['run_bad_reason']
    f = run_bad_fields(None)
    assert math.isnan(f['run_bad']) and math.isnan(f['pos_reset_mid'])


def _write_dut_log(tmp_path, reset_at=None, n=300):
    """A DUT log (85 columns), motor on, sp_vx 0.3 m/s, run_active 1 except lines 140..149
    (stale packets); pos_reset_n 1, +1 from line reset_at on."""
    p = tmp_path / 'run.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for s in range(n):
            v = {c: 0 for c in COLUMNS}
            stale = 140 <= s < 150 and reset_at is not None
            v.update(timestamp_us=s * 10_000, tick=s + 1, log_seq=s, motor_on=1, sp_vx=0.3,
                     run_active=0 if stale else 1,
                     pos_reset_n=1 + (reset_at is not None and s >= reset_at))
            w.writerow([v[c] for c in COLUMNS])
    return str(p)


def test_s02_metrics_marks_mid_run_reset(tmp_path):
    m = s02_metrics.compute(s02_metrics.load_log(_write_dut_log(tmp_path, reset_at=150)), 'A-L1')
    assert m['pos_reset_mid'] == 1 and m['run_bad'] == 1
    assert np.isfinite(m['e_vx'])            # metrics still written, the row is marked


def test_s02_metrics_sees_a_second_run_behind_motor_off_lines(tmp_path):
    """S3C-8: two runs in one DUT log, separated by motor-off lines that load_log drops; the
    check runs on every line and refuses the log instead of calling one run broken."""
    p = tmp_path / 'two.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for s in range(300):
            v = {c: 0 for c in COLUMNS}
            off = 140 <= s < 160
            v.update(timestamp_us=s * 10_000, tick=s + 1, log_seq=s, motor_on=0 if off else 1, sp_vx=0.3,
                     run_active=0 if off else 1, pos_reset_n=1 + (s >= 160))
            w.writerow([v[c] for c in COLUMNS])
    with pytest.raises(ValueError, match='more than one run'):
        s02_metrics.compute(s02_metrics.load_log(str(p)), 'A-L1')


def test_s02_metrics_clean_run(tmp_path):
    m = s02_metrics.compute(s02_metrics.load_log(_write_dut_log(tmp_path)), 'A-L1')
    assert m['pos_reset_mid'] == 0 and m['run_bad'] == 0 and m['run_bad_reason'] == ''


def test_s02_metrics_legacy_log_not_checkable(tmp_path):
    """37-column log of older firmware (tick + PWM counters, a6644a5): no pos_reset_n, run_bad
    left empty (NaN)."""
    head = COLUMNS[:37]
    p = tmp_path / 'old.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(head)
        for s in range(300):
            v = {c: 0 for c in head}
            v.update(timestamp_us=s * 10_000, tick=s + 1, sp_vx=0.3)
            w.writerow([v[c] for c in head])
    assert 'dpwm_fl' in head and 'pos_reset_n' not in head
    m = s02_metrics.compute(s02_metrics.load_log(str(p)), 'A-L1')
    assert math.isnan(m['run_bad'])
