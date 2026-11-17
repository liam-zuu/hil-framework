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
    p = sp.check(_filled(rows), pj, sp.REPS_MAIN)                    # every cell still once; only the order
    assert len(p) == 1 and p[0].startswith('block 1: ') and 'runs before' in p[0], p   # (S2M2-3) differs
    assert sp.main(['check', str(tmp_path)]) == 1                       # but not the plan of its order_seed


def test_blocks_are_whole_together_and_follow_the_plan():
    """S2M1-6 (Codex round 1 of v3, S2P3-4): a test moved out of its planned block into a later block (the cell still
    has one live run and reps 1..5), a block mixing conditions, a test of another condition, a block split in two
    places of runs.csv: each is a problem; a replacement run in a later block of the same key is not."""
    rows, blocks = sp.plan('real', ['PID', 'LADRC'], 5, 7)
    full = _filled(rows)
    for b in {r['block'] for r in full}:
        rs = [r for r in full if r['block'] == b]
        rs[0]['v_batt_start'], rs[-1]['v_batt_end'] = '12.4', '12.3'

    def with_later(moved):
        out = [dict(r) for r in full]
        i = next(j for j, r in enumerate(out) if r['block'] == '1')
        out[i]['result'], out[i]['cause_group'], out[i]['reason'] = 'replaced', '3', 'operator'
        later = [dict(out[i], block='21', result='completed', cause_group='', reason='', run_id='again')]
        if moved is not None:
            k = next(j for j, r in enumerate(out) if r['test_id'] == 'A-K1' and r['controller'] == 'PID'
                     and r['rep'] == '1')
            later.append(dict(out.pop(k), block='21', **moved))
        later[0]['v_batt_start'], later[-1]['v_batt_end'] = '12.4', '12.3'
        return out + later
    assert sp.check(with_later(None), blocks, sp.REPS_MAIN) == []
    p = sp.check(with_later({}), blocks, sp.REPS_MAIN)                  # A-K1 moved, its row still 'lifted'
    assert any(x.startswith('block 21: rows of 2') for x in p), p
    assert any("planned tests ['A-K1'] have no row" in x for x in p), p
    p = sp.check(with_later(dict(condition='floor')), blocks, sp.REPS_MAIN)   # its condition edited to floor
    assert any("test 'A-K1' is not a 'floor' test" in x for x in p), p
    assert any("planned tests ['A-K1'] have no row" in x for x in p), p
    split = [dict(r) for r in full]
    k = max(j for j, r in enumerate(split) if r['block'] == '2')
    split.append(split.pop(k))
    assert any(x.startswith('block 2: its rows are not together') for x in sp.check(split, blocks, sp.REPS_MAIN))
    other = [dict(r) for r in full]
    other[0]['controller'] = 'LADRC'                                    # one row of block 1 with the other controller
    assert any(x.startswith('block 1: rows of 2') for x in sp.check(other, blocks, sp.REPS_MAIN))
    p = sp.check(with_later({}), blocks, sp.REPS_MAIN)
    assert any("block 21: tests ['A-K1'] not in the planned block 1" in x for x in p), p
    into = [dict(r) for r in full]                                       # A-K1 moved into planned block 1
    k = next(j for j, r in enumerate(into) if r['test_id'] == 'A-K1' and r['controller'] == 'PID' and r['rep'] == '1')
    row = dict(into.pop(k), block='1', condition='floor')
    into.insert(next(j for j, r in enumerate(into) if r['block'] == '1') + 1, row)
    assert any("block 1: tests ['A-K1'] not planned in this block" in x for x in sp.check(into, blocks, sp.REPS_MAIN))
    swap = [dict(r, controller='LADRC') if r['block'] == '1' else dict(r) for r in full]   # whole block relabelled
    assert any(x.startswith("block 1: ('LADRC', 'floor', 1) in runs.csv, ('PID', 'floor', 1) in s02_plan.json")
               for x in sp.check(swap, blocks, sp.REPS_MAIN))
    later6 = [dict(r) for r in full] + [dict(full[0], block='21', rep='6', run_id='x6')]   # a later block, no plan
    assert any(x.startswith("block 21: not in s02_plan.json and no planned block ('PID', 'floor', 6)")
               for x in sp.check(later6, blocks, sp.REPS_MAIN))
    lifted_first = ([dict(r) for r in full if r['block'] == '11'] + [dict(r) for r in full if r['block'] != '11'])
    assert any(x.startswith('planned blocks run in the order') for x in sp.check(lifted_first, blocks, sp.REPS_MAIN))
    swapped = [dict(r) for r in full]                                    # S2M2-3: two tests of block 1 swapped
    i = [k for k, r in enumerate(swapped) if r['block'] == '1'][:2]
    swapped[i[0]], swapped[i[1]] = swapped[i[1]], swapped[i[0]]
    assert any(x.startswith('block 1: ') and 'runs before' in x for x in sp.check(swapped, blocks, sp.REPS_MAIN))
    # S2M3-2: a wrong test run first by mistake, replaced (group 3), then the planned order: allowed; the first test
    # failed by an operator error and replaced at the end of the block: allowed
    b1 = [k for k, r in enumerate(full) if r['block'] == '1']
    t0, t1 = full[b1[0]]['test_id'], full[b1[1]]['test_id']
    wrong = dict(full[b1[1]], result='replaced', cause_group='3', reason='wrong test run', run_id='mistake')
    mis = [dict(r) for r in full]
    mis.insert(b1[0], wrong)
    assert sp.check(mis, blocks, sp.REPS_MAIN) == [], sp.check(mis, blocks, sp.REPS_MAIN)
    late = [dict(r) for r in full]
    late[b1[0]] = dict(late[b1[0]], result='replaced', cause_group='3', reason='operator error')
    late.insert(b1[-1] + 1, dict(full[b1[0]], run_id='again'))
    assert sp.check(late, blocks, sp.REPS_MAIN) == [], sp.check(late, blocks, sp.REPS_MAIN)
    early_done = [dict(r) for r in full]                                 # run early and kept: wrong order
    early_done.insert(b1[0], dict(full[b1[1]], run_id='early'))
    assert any('early (' + t1 + ') runs before ' + t0 in x for x in sp.check(early_done, blocks, sp.REPS_MAIN))
    gone = [dict(r) for r in full if r['block'] != '3']
    assert any(x == 'block 3: planned (s02_plan.json), no row in runs.csv' for x in sp.check(gone, blocks, sp.REPS_MAIN))


CSF = 'clock_sync_failed (corr 0.71 < 0.90)'


def test_three_clock_sync_failed_in_a_row_stop_the_test():
    """S02 v1.17 P2-5 (Liam 17/11, CODE_S02 'Nhanh nho sau khi gop', 'Them kiem nhom 1'): 3 runs in a row of one
    test and one controller replaced for clock_sync_failed -> stop the test, tell Liam."""
    two = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'completed', '', ''))
    assert sp.check(two) == []
    three = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', 'Clock_sync_failed (edges_not_covered)'),
                  ('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'completed', '', ''))
    p = [x for x in sp.check(three) if 'clock_sync_failed' in x]
    assert len(p) == 1 and 'stop' in p[0] and 'Liam' in p[0] and 'lines [2, 3, 4]' in p[0]
    # across repetitions of the same test and controller (rows in file order), the other tests in between do not reset
    across = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'completed', '', ''))[:1] + \
        _rows(('A-L2', 1, 'completed', '', ''))[:1] + \
        _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'completed', '', ''))
    for r in across:
        r['v_batt_start'] = r['v_batt_end'] = '12.4'
    assert any('stop' in x and 'A-L1' in x for x in sp.check(across))


def test_another_run_of_the_pair_resets_the_count():
    reset = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', CSF),
                  ('A-L1', 1, 'replaced', '1', 'camera_lost'), ('A-L1', 1, 'replaced', '1', CSF),
                  ('A-L1', 1, 'completed', '', ''))
    assert not any('stop' in x for x in sp.check(reset))
    other_ctrl = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', CSF),
                       ('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'completed', '', ''),
                       ('A-L1', 1, 'completed', '', ''))
    other_ctrl[1]['controller'] = other_ctrl[4]['controller'] = 'LADRC'   # PID, LADRC, PID in a row: 2 + 1, no stop
    assert not any('stop' in x for x in sp.check(other_ctrl))
    not_replaced = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', CSF),
                         ('A-L1', 1, 'broken', '', CSF))
    assert not any('stop' in x for x in sp.check(not_replaced))


def test_measuring_system_faults_need_cause_group_1():
    for why in (CSF, 'camera_lost', ' camera_lost (2 lines)'):
        bad = _rows(('A-L1', 1, 'replaced', '2', why), ('A-L1', 1, 'completed', '', ''))
        p = sp.check(bad)
        assert len(p) == 1 and 'cause_group 1' in p[0], (why, p)
    assert sp.check(_rows(('A-L1', 1, 'replaced', '1', 'camera_lost'), ('A-L1', 1, 'completed', '', ''))) == []
    assert sp.check(_rows(('A-L1', 1, 'replaced', '3', 'wrong test run'), ('A-L1', 1, 'completed', '', ''))) == []
    empty = sp.check(_rows(('A-L1', 1, 'replaced', '', CSF), ('A-L1', 1, 'completed', '', '')))
    assert len(empty) == 1 and 'cause_group 1, 2 or 3' in empty[0]


def test_streaks_across_repetitions_and_battery_history():
    """S2C1-2 (Codex round 1): the count runs across repetitions (blocks of rep 1, 2, 3, the replacements later) and
    through a battery_rerun block's history (two there, one in the re-run block)."""
    reps = []
    for b, rep in (('1', 1), ('2', 2), ('3', 3)):
        reps += _rows(('A-L1', rep, 'replaced', '1', CSF), block=b)
    for b, rep in (('4', 1), ('5', 2), ('6', 3)):
        reps += _rows(('A-L1', rep, 'completed', '', ''), block=b)
    p = sp.check(reps)
    assert len(p) == 1 and 'stop' in p[0] and 'lines [2, 3, 4]' in p[0], p
    old = [dict(r, block_status='battery_rerun') for r in
           _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', CSF), ('A-L2', 1, 'completed', '', ''),
                 v=('12.4', '12.0'))]
    new = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'completed', '', ''), ('A-L2', 1, 'completed', '', ''),
                block='2')
    p = sp.check(old + new)
    assert len(p) == 1 and 'stop' in p[0] and 'lines [2, 3, 5]' in p[0], p


@pytest.mark.parametrize('end', [('completed', '', ''), ('broken', '', 'controller')])
def test_a_completed_or_broken_run_resets_before_the_next_repetition(end):
    """S2C1-3 (Codex round 1): rep 1 CSF, CSF, then its completed (or broken) run; rep 2 CSF, completed: 2 + 1, no
    stop (skipping the non-replaced rows instead of resetting would count 3)."""
    rows = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1) + end) +         _rows(('A-L1', 2, 'replaced', '1', CSF), ('A-L1', 2, 'completed', '', ''), block='2')
    assert sp.check(rows) == []


@pytest.mark.parametrize('why', [CSF, 'camera_lost'])
@pytest.mark.parametrize('group', ['2', '3', ''])
def test_group_1_also_in_a_battery_rerun_history(why, group):
    """S2C1-1 (Codex round 1): the battery_rerun rows are history, not counted, but a measuring system fault there
    still needs cause_group 1."""
    old = [dict(r, block_status='battery_rerun') for r in
           _rows(('A-L1', 1, 'replaced', group, why), ('A-L1', 1, 'completed', '', ''), v=('12.4', '12.0'))]
    new = _rows(('A-L1', 1, 'completed', '', ''), block='2')
    p = sp.check(old + new)
    assert len(p) == 1 and 'cause_group 1' in p[0] and p[0].startswith('line 2 '), p
    old[0]['cause_group'] = '1'
    assert sp.check(old + new) == []


def test_a_short_csv_line_of_a_broken_run_still_checks():
    """S2C2-1 (Codex round 2): a broken row whose CSV line ends at reason (DictReader fills None) was accepted
    before the group-1 check and must still be: the check reads cause_group only on a replaced measuring fault."""
    rows = _rows(('A-L1', 1, 'broken', '', 'controller'))
    rows[0].update(cause_group=None, missing_pct=None, gen_pause_n=None)
    assert sp.check(rows) == []
    short = _rows(('A-L1', 1, 'replaced', '1', CSF), ('A-L1', 1, 'completed', '', ''))
    old = [dict(r, block_status='battery_rerun') for r in
           _rows(('A-L1', 1, 'replaced', '1', 'camera_lost'), ('A-L1', 1, 'completed', '', ''), v=('12.4', '12.0'))]
    old[0]['cause_group'] = None
    p = sp.check(old + _rows(('A-L1', 1, 'completed', '', ''), block='2'))
    assert len(p) == 1 and 'cause_group 1' in p[0], p
    assert sp.check(short) == []
