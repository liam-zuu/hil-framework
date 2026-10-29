"""tools/s06_t2_check.py (S06 T2) on synthetic step logs in the format of rpi5/step_log.py."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_t2_check as t2  # noqa: E402
from step_log import CSV_HEADER, format_row  # noqa: E402

N_OK = 620_000            # 620 s: 10 s discarded + 610 s scored


def write(tmp_path, n=N_OK, late=(), end_late=(), skip=(), overflow=0, lost=0, name='steps.csv', meta=True):
    p = tmp_path / name
    with open(p, 'w', newline='\n') as f:
        f.write(CSV_HEADER + '\n')
        t_base = 5_000_000_000
        for k in range(n):
            if k in skip:
                continue
            ts = t_base + k * 1_000_000
            td = ts + (1_200_000 if k in late else 400_000)
            te = ts + (1_100_000 if k in end_late else 600_000)
            if k in late:
                te = max(te, td + 1000)
            f.write(format_row((k, ts, ts + 20_000, td, te)) + '\n')
    if meta:
        json.dump(dict(rows=n - len(skip), overflow=overflow, lost_or_torn=lost, monotonic_resolution_s=1e-9),
                  open(str(p) + '.json', 'w'))
    return str(p)


def test_clean_run_passes(tmp_path):
    r = t2.check_run(write(tmp_path))
    assert r['status'] == 'pass' and r['misses'] == 0 and r['t_end_late'] == 0
    assert r['steps_discarded'] == 10_000 and r['steps_scored'] == N_OK - 10_000
    assert abs(r['done_max_us'] - 400) < 1e-6 and abs(r['start_max_us'] - 20) < 1e-6
    assert t2.main([write(tmp_path, name='b.csv')]) == 0


def test_miss_and_clusters(tmp_path):
    late = {20_000, 20_050, 300_000}         # two clusters (gap 50 <= 100), one apart
    r = t2.check_run(write(tmp_path, late=late))
    assert r['status'] == 'fail' and r['misses'] == 3 and r['miss_clusters'] == 2
    assert r['cluster_spacing_steps'] == [300_000 - 20_050]
    assert r['miss_col_mismatch'] == 0


def test_miss_in_first_10_s_not_scored(tmp_path):
    r = t2.check_run(write(tmp_path, late={5_000}))
    assert r['status'] == 'pass' and r['misses'] == 0


def test_t_end_after_next_deadline_fails(tmp_path):
    r = t2.check_run(write(tmp_path, end_late={100_000}))
    assert r['status'] == 'fail' and r['t_end_late'] == 1 and r['misses'] == 0


@pytest.mark.parametrize('kw', [dict(overflow=1), dict(lost=2), dict(skip={50_000}), dict(meta=False)])
def test_data_loss_is_failure(tmp_path, kw):
    r = t2.check_run(write(tmp_path, **kw))
    assert r['status'] == 'fail_data_loss'


def test_too_short_and_exit_codes(tmp_path):
    short = write(tmp_path, n=300_000, name='s.csv')
    assert t2.check_run(short)['status'] == 'too_short'
    assert t2.main([short]) == 3
    assert t2.main([short, write(tmp_path, late={400_000}, name='f.csv')]) == 1


def test_rule_of_three(tmp_path, capsys):
    out = tmp_path / 'sum.json'
    t2.main([write(tmp_path, name=f'r{i}.csv') for i in range(3)] + ['--out', str(out)])
    tot = json.load(open(out))['total']
    assert tot['steps_scored'] == 3 * (N_OK - 10_000)
    assert abs(tot['rule_of_three_upper_95'] - 3 / tot['steps_scored']) < 1e-15
