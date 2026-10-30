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


def write(tmp_path, n=N_OK, late=(), end_late=(), skip=(), overflow=0, lost=0, name='steps.csv', meta=True,
          argv=None, near=(), cut_last=0, res_s=1e-9):
    """CSV + .json as run_hil --step-log writes them (counters of the whole drained ring); cut_last drops
    that many rows from the CSV only (as a truncated copy)."""
    p = tmp_path / name
    with open(p, 'w', newline='\n') as f:
        f.write(CSV_HEADER + '\n')
        t_base = 5_000_000_000
        for k in range(n - cut_last):
            if k in skip:
                continue
            ts = t_base + k * 1_000_000
            td = ts + (1_200_000 if k in late else (1_000_500 if k in near else 400_000))
            te = ts + (1_100_000 if k in end_late else 600_000)
            if k in late:
                te = max(te, td + 1000)
            if k in near:
                te = td + 100                       # 1000.6 µs: also within 1 µs of the next deadline
            f.write(format_row((k, ts, ts + 20_000, td, te)) + '\n')
    if meta:
        m = dict(rows=n - len(skip), first_step=0, steps_written=n, overflow=overflow, lost_or_torn=lost,
                 monotonic_resolution_s=res_s)
        if argv is not None:
            m['argv'] = argv
        json.dump(m, open(str(p) + '.json', 'w'))
    return str(p)


@pytest.mark.parametrize('argv, want', [(None, None), (['run_hil.py', '--step-log', 's.csv'], False),
                                        (['run_hil.py', '--frame-log', 'f.npz', '--step-log', 's.csv'], True)])
def test_frame_log_flag_reported(tmp_path, argv, want):
    r = t2.check_run(write(tmp_path, argv=argv))
    assert r['frame_log'] is want and r['status'] == 'pass'
    assert (argv is None) == any('dòng lệnh' in n for n in r['notes'])


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


def test_too_short_and_exit_codes(tmp_path, capsys):
    short = write(tmp_path, n=300_000, name='s.csv')
    assert t2.check_run(short)['status'] == 'too_short'
    assert t2.main([short]) == 3
    assert '#ket_qua,ma=3,ly_do=khong_du_dieu_kien' in capsys.readouterr().out
    assert t2.main([short, write(tmp_path, late={400_000}, name='f.csv')]) == 1
    assert '#ket_qua,ma=1' in capsys.readouterr().out
    assert t2.main([write(tmp_path, near={300_000}, res_s=1e-6, name='n.csv')]) == 3
    assert 'ly_do=chua_ket_luan' in capsys.readouterr().out


@pytest.mark.parametrize('bad', ['nan', 'inf'])
def test_c1_10_non_finite_row_is_data_loss(tmp_path, bad):
    # a late step whose t_done is NaN: `lat > 1000` is False, it would count as met
    p = write(tmp_path)
    lines = open(p).read().splitlines()
    f = lines[300_000].split(',')
    f[3] = bad
    lines[300_000] = ','.join(f)
    open(p, 'w', newline='\n').write('\n'.join(lines) + '\n')
    r = t2.check_run(p)
    assert r['status'] == 'fail_data_loss' and r['non_finite_rows'] == 1


def test_c1_10_non_finite_resolution_is_data_loss(tmp_path):
    p = write(tmp_path)
    m = json.load(open(p + '.json'))
    m['monotonic_resolution_s'] = float('nan')
    json.dump(m, open(p + '.json', 'w'))
    assert t2.check_run(p)['status'] == 'fail_data_loss'


def test_x8_uname_reported(tmp_path):
    p = write(tmp_path)
    assert any('uname' in x for x in t2.check_run(p)['notes'])
    m = json.load(open(p + '.json'))
    m['uname'] = 'Linux thesis-hil 6.18'
    json.dump(m, open(p + '.json', 'w'))
    r = t2.check_run(p)
    assert r['uname'] == 'Linux thesis-hil 6.18' and not any('uname' in x for x in r['notes'])


def test_x7_step_log_cost(tmp_path):
    import step_log_cost as c
    out = tmp_path / 'cost.json'
    assert c.main(['--n', '3000', '--no-pace', '--out', str(out)]) == 0
    r = json.load(open(out))
    assert r['n'] == 3000 and r['overflow'] == 0 and 0 <= r['median_us'] <= r['p999_us'] <= r['max_us']
    assert set(r['rt']) == {'sched_fifo', 'cpu_pinned'} and r['uname']


def test_rule_of_three(tmp_path, capsys):
    out = tmp_path / 'sum.json'
    t2.main([write(tmp_path, name=f'r{i}.csv') for i in range(3)] + ['--out', str(out)])
    tot = json.load(open(out))['total']
    assert tot['steps_scored'] == 3 * (N_OK - 10_000)
    assert abs(tot['rule_of_three_upper_95'] - 3 / tot['steps_scored']) < 1e-15
    # S14: with misses the rule of three is no upper bound: not reported, the observed rate is
    t2.main([write(tmp_path, name='m.csv', late={100_000, 200_000}), '--out', str(out)])
    tot = json.load(open(out))['total']
    assert tot['rule_of_three_upper_95'] is None and tot['misses'] == 2
    assert abs(tot['miss_rate_observed'] - 2 / (N_OK - 10_000)) < 1e-15


def test_s4_csv_cut_against_json(tmp_path):
    # the last row cut from the CSV: still continuous and long enough, but the .json counted it
    r = t2.check_run(write(tmp_path, cut_last=1))
    assert r['status'] == 'fail_data_loss' and r['meta_mismatch']
    p = write(tmp_path, name='old.csv')
    m = json.load(open(p + '.json'))
    del m['steps_written']                               # .json of an older run_hil: not checkable
    json.dump(m, open(p + '.json', 'w'))
    assert t2.check_run(p)['status'] == 'fail_data_loss'


def test_s13_last_step_and_resolution(tmp_path):
    # t_end of the last step after its next deadline (= own deadline + 1 ms) fails
    assert t2.check_run(write(tmp_path, end_late={N_OK - 1}))['status'] == 'fail'
    # done 0.5 µs after the deadline with a 1 µs clock: cannot tell, not concluded
    r = t2.check_run(write(tmp_path, near={300_000}, res_s=1e-6, name='n.csv'))
    assert r['status'] == 'inconclusive' and r['misses'] == 0 and r['near_deadline'] == 2   # done + end
    assert t2.main([write(tmp_path, near={300_000}, res_s=1e-6, name='n2.csv')]) == 3
