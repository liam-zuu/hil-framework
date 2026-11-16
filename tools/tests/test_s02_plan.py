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
    assert all(r['block_status'] == '' and r['missing_pct'] == '' for r in rows)
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
    rows, _ = sp.plan('hil', ['PID', 'LADRC'], 1, 3, trial=True)
    r = next(x for x in rows if x['test_id'] == 'A-K2' and x['controller'] == 'PID')
    assert r['command'] == '--run A-K2_PID_r1="--test A-K2 --s02-prep 3 --ground traction --mu 0.436 --imu-noise 20 --seed 1"'
    assert r['file'] == 'pi/A-K2_PID_r1.csv' and r['noise_seed'] == 1
    rows, _ = sp.plan('real', ['PID', 'LADRC'], 1, 3, with_strafe=True, trial=True)
    r = next(x for x in rows if x['test_id'] == 'A-S1')
    assert r['command'].endswith('--test A-S1 --prep-s 3 --allow-strafe')


def test_refuses_sim_and_other_controllers():
    with pytest.raises(ValueError):
        sp.plan('sim', ['PID', 'LADRC'], 5, 1)
    with pytest.raises(ValueError):
        sp.plan('hil', ['PID'], 5, 1)


def _rows(*specs, platform='real', block='1', v=('12.4', '12.3')):
    base = sp.plan(platform, ['PID', 'LADRC'], 1, 3, trial=True)[0][0]
    out = []
    for tid, rep, res, cause, reason in specs:
        r = dict(base, test_id=tid, rep=str(rep), result=res, cause_group=cause, reason=reason, block=block,
                 missing_pct='0.1', gen_pause_n='0')
        out.append({k: str(x) for k, x in r.items()})
    out[0]['v_batt_start'], out[-1]['v_batt_end'] = v
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


def test_check_rep_is_a_number():
    """S2P1-8: "1" and "01" are one cell."""
    rows = _rows(('A-L1', 1, 'broken', '', 'c'), ('A-L1', '01', 'completed', '', ''))
    assert any('already has' in p for p in sp.check(rows))
    assert any('not a repetition' in p for p in sp.check(_rows(('A-L1', 'x', 'completed', '', ''))))


def test_check_against_the_plan(tmp_path):
    """S2P1-8: with s02_plan.json, every planned cell exactly once and no other; reps 1..N."""
    rows, blocks = sp.plan('hil', ['PID', 'LADRC'], 2, 7, trial=True)
    full = [{k: str(x) for k, x in dict(r, result='completed', missing_pct='0', gen_pause_n='0').items()}
            for r in rows]
    assert sp.check([dict(r) for r in full], blocks) == []
    assert any('planned, no completed' in p for p in sp.check([dict(r) for r in full[1:]], blocks))
    extra = [dict(r) for r in full] + [dict(full[0], rep='3', run_id='x')]
    assert any('not in s02_plan.json' in p for p in sp.check(extra, blocks))
    rows6, blocks6 = sp.plan('hil', ['PID', 'LADRC'], 6, 7, trial=True)
    bad = [b for b in blocks6 if b['rep'] != 4]
    assert any('not 1..N' in p for p in sp.check([{k: str(x) for k, x in dict(r, result='completed').items()}
                                                  for r in rows6 if r['rep'] != 4], bad))


def _filled(rows):
    return [{k: str(x) for k, x in dict(r, result='completed', missing_pct='0', gen_pause_n='0').items()}
            for r in rows]


def test_main_numbers_need_five_reps_and_the_plan(tmp_path):
    """S2P1-8 (Codex round 2): a plan of another N than 5 only as a declared trial; check of the main numbers
    refuses a 6-repetition plan and a folder without s02_plan.json."""
    with pytest.raises(ValueError):
        sp.plan('hil', ['PID', 'LADRC'], 6, 7)
    rows6, blocks6 = sp.plan('hil', ['PID', 'LADRC'], 6, 7, trial=True)
    assert any('not 1..5' in p for p in sp.check(_filled(rows6), blocks6, sp.REPS_MAIN))
    assert sp.check(_filled(rows6), blocks6, None) == []
    rows5, blocks5 = sp.plan('hil', ['PID', 'LADRC'], 5, 7)
    assert sp.check(_filled(rows5), blocks5, sp.REPS_MAIN) == []
    d = tmp_path / 'nojson'
    d.mkdir()
    with open(d / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sp.COLUMNS)
        w.writeheader()
        w.writerows(_filled(rows5[:1]))
    assert sp.main(['check', str(d)]) == 1
    assert sp.main(['check', str(d), '--trial']) == 0
    t = tmp_path / 'trial'
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '5', '--reps', '2', '--out', str(t)]) == 2
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '5', '--reps', '2', '--trial', '--out', str(t)]) == 0
    with open(t / 'runs.csv', newline='') as f:
        rr = [dict(r, result='completed', missing_pct='0', gen_pause_n='0') for r in csv.DictReader(f)]
    with open(t / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sp.COLUMNS)
        w.writeheader()
        w.writerows(rr)
    assert sp.main(['check', str(t)]) == 1                   # a trial plan is not main numbers
    assert sp.main(['check', str(t), '--trial']) == 0
    t5 = tmp_path / 'trial5'                                 # a trial plan of 5 repetitions: the flag alone
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '5', '--trial', '--out', str(t5)]) == 0
    with open(t5 / 'runs.csv', newline='') as f:
        rr = [dict(r, result='completed', missing_pct='0', gen_pause_n='0') for r in csv.DictReader(f)]
    with open(t5 / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sp.COLUMNS)
        w.writeheader()
        w.writerows(rr)
    assert sp.main(['check', str(t5)]) == 1 and sp.main(['check', str(t5), '--trial']) == 0


def test_completed_needs_the_loss_fields():
    """S2P1-14 (Codex round 2): a completed run has missing_pct 0..100 and gen_pause_n >= 0; empty is not 0."""
    base = _rows(('A-L1', 1, 'completed', '', ''))
    assert sp.check([dict(r) for r in base]) == []
    for mp, gp in (('', '0'), ('nan', '0'), ('-1', '0'), ('0.2', ''), ('0.2', '1.5'), ('0.2', '-1')):
        rows = [dict(base[0], missing_pct=mp, gen_pause_n=gp)]
        assert sp.check(rows), (mp, gp)
    assert sp.check([dict(base[0], result='broken', reason='controller', missing_pct='')]) == []


def test_check_battery_block():
    """S2P1-9 (P2-5): a sag > 0.3 V needs the block marked battery_rerun and a later block run again; the old
    rows are history, not counted."""
    sag = _rows(('A-L1', 1, 'completed', '', ''), ('A-L2', 1, 'completed', '', ''), v=('12.4', '12.0'))
    assert any('battery sagged' in p for p in sp.check(sag))
    old = [dict(r, block_status='battery_rerun') for r in sag]
    assert any('no later block' in p for p in sp.check([dict(r) for r in old]))
    new = _rows(('A-L1', 1, 'completed', '', ''), ('A-L2', 1, 'completed', '', ''), block='2')
    assert sp.check([dict(r) for r in old] + new) == []
    assert any('rest voltage' in p for p in sp.check(_rows(('A-L1', 1, 'completed', '', ''), v=('', ''))))


def test_battery_rerun_needs_its_own_sag():
    """S2P2-3 (Codex round 2): a block without a sag > 0.3 V (here 12.4 -> 12.3 V, with a broken run) cannot be
    set aside as battery_rerun; nor a HIL block; the later block must run every test of it again."""
    keep = _rows(('A-L1', 1, 'broken', '', 'controller'), ('A-L2', 1, 'completed', '', ''), v=('12.4', '12.3'))
    old = [dict(r, block_status='battery_rerun') for r in keep]
    new = _rows(('A-L1', 1, 'completed', '', ''), ('A-L2', 1, 'completed', '', ''), block='2')
    assert any('without its own sag' in p for p in sp.check(old + new))
    hil = [dict(r, block_status='battery_rerun') for r in _rows(('A-L1', 1, 'completed', '', ''), platform='hil')]
    hil_new = _rows(('A-L1', 1, 'completed', '', ''), platform='hil', block='2')
    assert any('not a REAL block' in p for p in sp.check(hil + hil_new))
    sag = [dict(r, block_status='battery_rerun') for r in
           _rows(('A-L1', 1, 'completed', '', ''), ('A-L2', 1, 'completed', '', ''), v=('12.4', '12.0'))]
    part = _rows(('A-L1', 1, 'completed', '', ''), block='2')
    assert any('does not run again' in p for p in sp.check(sag + part))


def test_sag_of_exactly_0_3_v_is_not_over():
    """S2P2-4 (Codex round 2): 12.4 -> 12.1 V is a sag of exactly 0.3 V (not > 0.3 in float); 12.4 -> 12.09 is."""
    assert sp.check(_rows(('A-L1', 1, 'completed', '', ''), v=('12.4', '12.1'))) == []
    assert any('battery sagged' in p for p in sp.check(_rows(('A-L1', 1, 'completed', '', ''), v=('12.4', '12.09'))))


def test_columns_follow_p2_6():
    """S2P1-14: the P2-6 runs.csv fields (independent list, not COLUMNS itself)."""
    p2_6 = ['run_id', 'platform', 'controller', 'test_id', 'speed_mps', 'rep', 'seed', 'noise_seed', 'block',
            'v_batt_start', 'v_batt_end', 'result', 'reason', 'cause_group', 'missing_pct', 'gen_pause_n']
    assert all(c in sp.COLUMNS for c in p2_6)


def test_main_never_overwrites(tmp_path):
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '5', '--out', str(tmp_path)]) == 0
    with open(tmp_path / 'runs.csv', newline='') as f:
        assert len(list(csv.DictReader(f))) == 90
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '6', '--out', str(tmp_path)]) == 2


def test_metrics_skip_battery_rerun_and_replaced(tmp_path, monkeypatch):
    """S2P1-9: tools/s02_metrics.py does not score the rows of a battery_rerun block nor replaced runs (v3: they keep
    a status line)."""
    import s02_metrics
    (tmp_path / 'runs.csv').write_text(
        'run_id,platform,controller,test_id,speed_mps,rep,file,aborted,block_status,result\n'
        'a,real,PID,A-L2,0.5,1,a.csv,,battery_rerun,completed\n'
        'b,real,PID,A-L2,0.5,1,b.csv,,,replaced\n'
        'c,real,PID,A-L2,0.5,1,c.csv,,,completed\n', encoding='utf-8')
    monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {})
    monkeypatch.setattr(s02_metrics, 'score_run', lambda log, tid, *a, **k: {'e_vx': 0.0})
    out = s02_metrics.run_dir_metrics(str(tmp_path))
    # CODE_S02 v3 (S02 v1.13 P2-6): every row has a line, only c is scored
    assert [(o['run_id'], o['status']) for o in out] == [('a', 'battery_rerun'), ('b', 'replaced'), ('c', 'ok')]
    assert 'e_vx' not in out[0] and 'e_vx' not in out[1] and out[2]['e_vx'] == 0.0


def test_battery_rerun_needs_one_whole_block():
    """S2P3-4 (Codex round 3): an old block (12.4 -> 12.0 V) re-run as two half blocks (12.4 -> 12.2, 12.2 -> 12.0 V)
    passes each sag rule but not the whole-block rule."""
    old = [dict(r, block_status='battery_rerun') for r in
           _rows(('A-L1', 1, 'completed', '', ''), ('A-L2', 1, 'completed', '', ''), v=('12.4', '12.0'))]
    b = _rows(('A-L1', 1, 'completed', '', ''), block='2', v=('12.4', '12.2'))
    c = _rows(('A-L2', 1, 'completed', '', ''), block='3', v=('12.2', '12.0'))
    assert any('replaced by 2 blocks' in p for p in sp.check(old + b + c))
    whole = _rows(('A-L1', 1, 'completed', '', ''), ('A-L2', 1, 'completed', '', ''), block='2', v=('12.4', '12.2'))
    assert sp.check(old + whole) == []


@pytest.mark.parametrize('bad', ['inf', '-inf', '1e309'])
def test_loss_fields_infinite_are_problems_not_errors(bad):
    """S2P3-5 (Codex round 3): inf or an overflowing number is a reported problem, not an exception."""
    base = _rows(('A-L1', 1, 'completed', '', ''))
    assert sp.check([dict(base[0], gen_pause_n=bad)])
    assert sp.check([dict(base[0], missing_pct=bad)])


def test_every_cell_has_five_reps_and_the_manifest_is_its_plan(tmp_path):
    """S2P3-6 (Codex round 3): drop A-L1/PID/rep 5 from the manifest and from runs.csv: the reps of the whole
    manifest are still 1..5 but that cell has 4; and the CLI rebuilds the plan from the manifest's arguments."""
    rows, blocks = sp.plan('hil', ['PID', 'LADRC'], 5, 7)
    cut = [dict(bl, tests=[t for t in bl['tests'] if not (t == 'A-L1' and bl['controller'] == 'PID' and bl['rep'] == 5)])
           for bl in blocks]
    keep = [r for r in rows if not (r['test_id'] == 'A-L1' and r['controller'] == 'PID' and r['rep'] == 5)]
    probs = sp.check(_filled(keep), cut, sp.REPS_MAIN)
    assert any("('A-L1', 'PID'): planned repetitions [1, 2, 3, 4]" in p for p in probs)
    assert sp.main(['plan', '--platform', 'hil', '--order-seed', '7', '--out', str(tmp_path)]) == 0
    import json
    js = tmp_path / 's02_plan.json'
    pj = json.loads(js.read_text(encoding='utf-8'))
    with open(tmp_path / 'runs.csv', newline='') as f:
        rr = [dict(r, result='completed', missing_pct='0', gen_pause_n='0') for r in csv.DictReader(f)]
    def write(rows_):
        with open(tmp_path / 'runs.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sp.COLUMNS)
            w.writeheader()
            w.writerows(rows_)
    write(rr)
    assert sp.main(['check', str(tmp_path)]) == 0
    pj['blocks'][0]['tests'] = pj['blocks'][0]['tests'][1:]           # edited manifest, one cell gone
    gone = pj['blocks'][0]['tests'] and blocks[0]['tests'][0]
    js.write_text(json.dumps(pj), encoding='utf-8')
    write([r for r in rr if not (r['block'] == '1' and r['test_id'] == gone)])
    assert sp.main(['check', str(tmp_path)]) == 1
    pj = json.loads(json.dumps(blocks))                                # a test order not from the seed: same cells
    pj[0]['tests'] = pj[0]['tests'][::-1]
    js.write_text(json.dumps(dict(json.loads(js.read_text(encoding='utf-8')), blocks=pj)), encoding='utf-8')
    write(rr)
    assert sp.check(_filled(rows), pj, sp.REPS_MAIN) == []             # every cell still once
    assert sp.main(['check', str(tmp_path)]) == 1                       # but not the plan of its order_seed
