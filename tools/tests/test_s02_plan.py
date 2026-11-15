"""s02_plan.py (CODE_S02 design 12): block order grouped by condition, controllers alternating with the
first one switching per repetition, tests shuffled by the order seed, A-S only on request; the check."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import s02_plan as sp  # noqa: E402


def test_blocks_grouped_by_condition():
    b = sp.blocks(['PID', 'LADRC'], 5)
    assert len(b) == 20
    assert [x[0] for x in b] == ['floor'] * 10 + ['lifted'] * 10
    assert b[:4] == [('floor', 1, 'PID'), ('floor', 1, 'LADRC'), ('floor', 2, 'LADRC'), ('floor', 2, 'PID')]
    assert b[10:12] == [('lifted', 1, 'PID'), ('lifted', 1, 'LADRC')]


def test_plan_counts_and_columns():
    rows, blist = sp.plan('real', ['PID', 'LADRC'], 5, 42)
    assert len(rows) == 5 * 2 * (6 + 3)                 # A-S held back: 6 floor + 3 lifted tests
    assert not any(r['test_id'].startswith('A-S') for r in rows)
    assert all(list(r) == sp.COLUMNS for r in rows)
    assert {r['env'] for r in rows if r['condition'] == 'lifted'} == {'wheels_lifted', 'ladrc_wheels_lifted'}
    assert all(r['seed'] == r['rep'] for r in rows)
    assert len({r['run_id'] for r in rows}) == len(rows)
    rows2, _ = sp.plan('real', ['PID', 'LADRC'], 5, 42, with_strafe=True)
    assert sum(r['test_id'].startswith('A-S') for r in rows2) == 20


def test_order_is_from_the_seed():
    a = [r['test_id'] for r in sp.plan('hil', ['PID', 'LADRC'], 5, 1)[0]]
    b = [r['test_id'] for r in sp.plan('hil', ['PID', 'LADRC'], 5, 1)[0]]
    c = [r['test_id'] for r in sp.plan('hil', ['PID', 'LADRC'], 5, 2)[0]]
    assert a == b and a != c
    first = sp.plan('hil', ['PID', 'LADRC'], 5, 1)[1]
    assert len({tuple(x['tests']) for x in first if x['condition'] == 'floor'}) > 1    # not one order for all


def test_commands():
    rows, _ = sp.plan('hil', ['PID', 'LADRC'], 1, 3)
    r = next(x for x in rows if x['test_id'] == 'A-K2' and x['controller'] == 'PID')
    assert r['command'] == '--run A-K2_PID_r1="--test A-K2 --s02-prep 3 --ground traction --mu 0.436 --imu-noise 20 --seed 1"'
    assert r['file'] == 'pi/A-K2_PID_r1.csv' and r['noise_seed'] == 1
    rows, _ = sp.plan('real', ['PID', 'LADRC'], 1, 3, with_strafe=True)
    r = next(x for x in rows if x['test_id'] == 'A-S1')
    assert r['command'].endswith('--test A-S1 --prep-s 3 --allow-strafe')


def test_refuses_sim_and_other_controllers():
    with pytest.raises(ValueError):
        sp.plan('sim', ['PID', 'LADRC'], 5, 1)
    with pytest.raises(ValueError):
        sp.plan('hil', ['PID'], 5, 1)


def _rows(*specs):
    base = sp.plan('real', ['PID', 'LADRC'], 1, 3)[0][0]
    out = []
    for tid, rep, res, cause, reason in specs:
        r = dict(base, test_id=tid, rep=str(rep), result=res, cause_group=cause, reason=reason)
        out.append({k: str(v) for k, v in r.items()})
    return out


def test_check():
    ok = _rows(('A-L1', 1, 'replaced', '1', 'camera lost'), ('A-L1', 1, 'completed', '', ''),
               ('A-L2', 1, 'broken', '', 'controller'))
    assert sp.check(ok) == []
    assert any('cause_group' in p for p in sp.check(_rows(('A-L1', 1, 'replaced', '', 'x'), ('A-L1', 1, 'completed', '', ''))))
    assert any('no later run' in p for p in sp.check(_rows(('A-L1', 1, 'replaced', '2', 'x'))))
    assert any('needs a reason' in p for p in sp.check(_rows(('A-L1', 1, 'broken', '', ''))))
    assert any('already has' in p for p in sp.check(_rows(('A-L1', 1, 'broken', '', 'c'), ('A-L1', 1, 'completed', '', ''))))
    assert any('not in' in p for p in sp.check(_rows(('A-L1', 1, '', '', ''))))


def test_main_never_overwrites(tmp_path):
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '5', '--out', str(tmp_path)]) == 0
    with open(tmp_path / 'runs.csv', newline='') as f:
        assert len(list(csv.DictReader(f))) == 90
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '6', '--out', str(tmp_path)]) == 2
