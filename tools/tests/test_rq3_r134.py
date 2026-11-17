"""rq3_log R134 (b): boot of gain_send --query before and after the run (S05 4.7 (iv), R3V-9).
Liam 14/11 CODE_R134 "duyet goi sua code fix/r134-answers". Kept apart from test_gate_v3_tools.py."""
import copy
import csv
import datetime as dt
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rq3_log as rl  # noqa: E402

# B81-1, B81-3: the camera scorer output of a judged car run (FAKE: two equal frame domains), G3 failed
CAR_KD = dict(gate=rl.floor_gate(rl.fake_gate(e_trk=0.5)), results_dir='r')
ANSWER = '\n'.join(['#gain_running,sha=16702ff0e688,env=pid,mac=aa:bb,boot={b},id=0,applied_n=0,ctrl=PID',
                    '#ket_qua,ma=0', ''])


@pytest.fixture
def ds(tmp_path, monkeypatch):
    clock = [dt.datetime(2026, 11, 20, 8, 0, 0)]
    monkeypatch.setattr(rl, 'now', lambda: clock[0])
    d = str(tmp_path / 'ds')
    os.makedirs(d)
    rl.migrate(d)
    return d, clock


def _open(d, clock, platform='HW', stage='screen'):
    rl.begin(d, 'S', 'A-PID', platform, 'PID', 'O1-F05', '0.5', stage,
             gain_set='A-PID/G1')
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)


def _rows(d):
    with open(os.path.join(d, rl.TUNING), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_boot_columns_recorded(ds):
    d, clock = ds
    _open(d, clock)
    qb, qa = rl.fake_queries('4753dc47')
    rl.end(d, 'khong_dat', queries=(qb, dict(qa, boot='4753DC47')), **CAR_KD)
    r = _rows(d)[-1]
    assert (r['boot_before'], r['boot_after']) == ('4753dc47', '4753DC47')
    assert {'boot_before', 'boot_after'} <= set(rl.TUNING_COLS)
    rl.summary(d)


@pytest.mark.parametrize('before, after', [('4753dc47', 'deadbeef'), ('', '4753dc47'), ('4753dc47', ''),
                                           ('zz', 'zz'), (None, '4753dc47')])
def test_missing_or_different_boot_is_a_run_failed_by_operation(ds, before, after):
    d, clock = ds
    _open(d, clock)
    for o in ('khong_dat', 'chua_ket_luan'):
        with pytest.raises(rl.LogError, match=rl.BOOT_FAIL):
            rl.end(d, o, boot_before=before, boot_after=after)
    with pytest.raises(rl.LogError, match=rl.BOOT_FAIL):
        rl.end(d, 'hong_he_do', fail_cause='mat_khung', boot_before=before, boot_after=after)
    rl.end(d, 'hong_he_do', fail_cause=rl.BOOT_FAIL, boot_before=before, boot_after=after)
    assert _rows(d)[-1]['fail_cause'] == rl.BOOT_FAIL


def test_boot_fail_cause_needs_the_evidence(ds):
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='needs the two boot answers'):
        rl.end(d, 'hong_he_do', fail_cause=rl.BOOT_FAIL, boot_before='ab', boot_after='ab')
    # R134C2-7: no answer file at all is "both missing": the run can be closed as failed by operation
    row = rl.end(d, 'hong_he_do', fail_cause=rl.BOOT_FAIL)
    assert row['fail_cause'] == rl.BOOT_FAIL and row.get('boot_before', '') == ''


def test_summary_refuses_a_judged_row_with_two_boots(ds):
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'khong_dat', queries=rl.fake_queries('ab'), **CAR_KD)
    rows = _rows(d)
    rows[-1]['boot_after'] = 'cd'                                      # edited by hand
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='restarted'):
        rl.summary(d)


def test_boot_needed_scope():
    assert rl.boot_needed('HW', 'screen') and rl.boot_needed('HW', 'final') and rl.boot_needed('HIL', 'final_hil')
    assert not rl.boot_needed('HIL', 'hil_check') and not rl.boot_needed('SIM', 'screen')


def test_cli_end_reads_the_query_files(ds, tmp_path):
    d, clock = ds
    _open(d, clock)
    qb, qa = tmp_path / 'qb.txt', tmp_path / 'qa.txt'
    qb.write_text(ANSWER.format(b='4753dc47'), encoding='utf-8')
    qa.write_text(ANSWER.format(b='0badf00d'), encoding='utf-8')
    assert rl.main(['--dataset', d, 'end', '--outcome', 'khong_dat', '--query-before', str(qb),
                    '--query-after', str(qa)]) != 0                 # different boots: refused
    assert rl.main(['--dataset', d, 'end', '--outcome', 'hong_he_do', '--fail-cause', rl.BOOT_FAIL,
                    '--query-before', str(qb), '--query-after', str(qa)]) == 0
    r = _rows(d)[-1]
    assert (r['boot_before'], r['boot_after'], r['run_outcome']) == ('4753dc47', '0badf00d', 'hong_he_do')


def test_next_iter_no_counts_the_open_run_of_the_same_session(ds):
    """S05 v3.15 Codex round 5 P3 (DEFERRED there; Liam 14/11 CODE_R134 "Lam cung goi (Recommended)"):
    begin(session='MACHINE') then machine gave two MACHINE:1."""
    d, clock = ds
    rl.begin(d, 'MACHINE', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G1')
    assert rl.next_iter_no(d, 'MACHINE') == 2 and rl.next_iter_no(d, 'OTHER') == 1
    m = rl.machine(d, 'A-PID', 'PID', 'X', 'r', 1, 10, '2026-11-20T09:00:00', '2026-11-20T09:01:00')
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)
    e = rl.end(d, 'hong_he_do', fail_cause='mat_khung')           # B81-1, -2: a judged car run needs more
    uids = [r['run_uid'] for r in _rows(d)]
    assert m['run_uid'] == 'MACHINE:2' and e['run_uid'] == 'MACHINE:1' and len(set(uids)) == len(uids)
    assert rl.next_iter_no(d, 'MACHINE') == 3
    rl.summary(d)


def _answer_file(path, boot='4753dc47', t=(1.0, 2.0), pc=True):
    path.write_text((f'#query_pc,t_start={t[0]},t_end={t[1]}\n' if pc else '') + ANSWER.format(b=boot), encoding='utf-8')
    return str(path)


@pytest.mark.parametrize('case', ['same_file', 'no_pc_line', 'wrong_order'])
def test_cli_end_checks_the_two_answers_like_the_scorers(ds, tmp_path, case):
    """R134C3-3 (Codex counterexample): one file given for both, a file without '#query_pc', or the two in
    the wrong order: not two questions around the run, so only hong_he_do with boot_thieu_hay_khac."""
    d, clock = ds
    _open(d, clock)
    qb = _answer_file(tmp_path / 'qb.txt', pc=case != 'no_pc_line')
    qa = {'same_file': qb, 'no_pc_line': _answer_file(tmp_path / 'qa.txt', t=(3, 4)),
          'wrong_order': _answer_file(tmp_path / 'qa.txt', t=(0.0, 0.5))}[case]
    assert rl.main(['--dataset', d, 'end', '--outcome', 'khong_dat', '--query-before', qb, '--query-after', qa]) == 2
    assert rl.main(['--dataset', d, 'end', '--outcome', 'hong_he_do', '--fail-cause', rl.BOOT_FAIL,
                    '--query-before', qb, '--query-after', qa]) == 0
    assert _rows(d)[-1]['fail_cause'] == rl.BOOT_FAIL


def test_cli_end_accepts_two_proper_answers(ds, tmp_path):
    d, clock = ds
    _open(d, clock)
    qb, qa = _answer_file(tmp_path / 'qb.txt'), _answer_file(tmp_path / 'qa.txt', t=(3.0, 4.0))
    gj = tmp_path / 'score.json'
    gj.write_text(json.dumps(dict(gate_v3=CAR_KD['gate'])), encoding='utf-8')            # B81-1: camera gate
    assert rl.main(['--dataset', d, 'end', '--outcome', 'khong_dat', '--query-before', qb, '--query-after', qa,
                    '--gate-json', str(gj), '--results-dir', 'r']) == 0
    assert (_rows(d)[-1]['boot_before'], _rows(d)[-1]['boot_after']) == ('4753dc47', '4753dc47')


# ── B8-1 (S05 v3.20 8.1 C9 (a)-(c); O1 v11.15 3.3, 3.5; CODE 14/11 to CODE_R134) ──

def _hw_gate(t_done=(10.0, 10.0), **kw):
    """FAKE camera gate: rl.fake_gate as a floor output (two equal domains unless edited)."""
    return rl.floor_gate(rl.fake_gate(**kw), t_done=t_done)


def _restate(g):
    """States and outcomes of an edited gate, recomputed as the scorer would (two-domain rule, G1 from fields)."""
    f1 = rl.gate_g1_fields(g['g1'])
    st = rl.gate_g1_state(f1)
    g['g1'].update(state=st, q={'dat': 'true', 'khong_dat': 'false'}.get(st, 'inconclusive'))
    for c, thr in rl.o1_gate.THRESHOLDS.items():
        com = g[c].get('chung', g[c])
        nan = float('nan')
        num = lambda x: nan if x is None else x                       # noqa: E731
        g[c]['state'] = rl.cond_state2(num(com['lo']), num(g[c]['lo']), num(g[c]['hi']), g[c]['n_undef'], thr)
    g.update(rl.o1_gate.evaluate_outcomes(g, g['decide']))
    return g


BOOTS = dict(queries=rl.fake_queries('ab'))                   # R134C4-1: two answers, not two strings


@pytest.mark.parametrize('outcome', ['dat', 'khong_dat', 'chua_ket_luan'])
def test_b81_1_judged_car_run_needs_the_camera_gate(ds, outcome):
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='camera scorer'):
        rl.end(d, outcome, **BOOTS)
    one = rl.fake_gate()                                              # o1_gate output: one domain, no G1 fields
    with pytest.raises(rl.LogError, match='t_done'):
        rl.end(d, outcome, gate=one, results_dir='r', **BOOTS)
    no_common = _hw_gate()
    no_common['g3'].pop('chung')
    with pytest.raises(rl.LogError, match='common frame domain'):
        rl.end(d, outcome, gate=no_common, results_dir='r', **BOOTS)
    g = {'dat': _hw_gate(), 'khong_dat': _hw_gate(e_trk=0.5), 'chua_ket_luan': _hw_gate(t_done=(10.0, None))}[outcome]
    rl.end(d, outcome, gate=g, results_dir='r', **BOOTS)
    r = _rows(d)[-1]
    assert r['g1_ways'] == '2' and r['g3_q_chung'] == r['g3_q'] and r['run_outcome'] == outcome
    rl.summary(d)


def test_b81_1_summary_refuses_a_judged_car_row_without_gate_columns(ds):
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'khong_dat', **BOOTS, **CAR_KD)
    rows = _rows(d)
    bare = dict(rows[0], **{c: '' for c in rl.GATE_COLS})
    path = os.path.join(d, rl.TUNING)
    rl._write_all(path, rl.TUNING_COLS, [bare])
    with pytest.raises(rl.LogError, match='camera scorer'):
        rl.summary(d)
    hw_one = dict(rows[0], **{c: '' for c in rl.G1_COLS})          # gate columns, but no G1 fields of the floor
    rl._write_all(path, rl.TUNING_COLS, [hw_one])
    with pytest.raises(rl.LogError, match='t_done'):
        rl.summary(d)
    rl._write_all(path, rl.TUNING_COLS, [dict(bare, campaign='', session_id='OLD', run_uid='OLD:1')])
    rl.summary(d)                                                    # historic row (no campaign): SC3-2 note


@pytest.mark.parametrize('platform, stage, needed', [('HW', 'screen', True), ('HW', 'final', True),
                                                    ('HIL', 'final_hil', True), ('HIL', 'hil_check', False)])
def test_b81_2_end_and_summary_need_both_boots(ds, platform, stage, needed):
    d, clock = ds
    campaign, gs = {('HW', 'screen'): ('A-PID', 'A-PID/G1'), ('HW', 'final'): ('A-PID', 'A-PID/G0'),
                    ('HIL', 'final_hil'): ('RQ2', 'RQ2-INIT-PID/G0'), ('HIL', 'hil_check'): ('B-PID', 'B-PID/GA1')}[
        (platform, stage)]
    rl.begin(d, 'S', campaign, platform, 'PID', 'O1-F05', '0.5', stage, gain_set=gs)
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)
    gate = _hw_gate(e_trk=0.5) if platform == 'HW' else rl.fake_gate(e_trk=0.5)
    if needed:
        with pytest.raises(rl.LogError, match=rl.BOOT_FAIL):        # the Python API too, not only the CLI
            rl.end(d, 'khong_dat', gate=gate, results_dir='r')
        rl.end(d, 'khong_dat', gate=gate, results_dir='r', **BOOTS)
        rows = _rows(d)
        path = os.path.join(d, rl.TUNING)
        for cut in (dict(boot_before=''), dict(boot_after=''), dict(boot_before='', boot_after='')):
            rl._write_all(path, rl.TUNING_COLS, [dict(rows[0], **cut)])
            with pytest.raises(rl.LogError, match='missing or unreadable'):
                rl.summary(d)
    else:
        rl.end(d, 'khong_dat', gate=gate, results_dir='r')
        rl.summary(d)


def test_b81_2_lost_run_without_answers_is_boot_fail(ds):
    d, clock = ds
    _open(d, clock)
    row = rl.end(d, 'hong_he_do', fail_cause=rl.BOOT_FAIL)
    assert row['fail_cause'] == rl.BOOT_FAIL
    rl.summary(d)


def _two_domains():
    """G2: the union certainly above D (a frame after the earlier t_done), the common domain below it."""
    g = _hw_gate(t_done=(10.0, 12.0))
    g['g2'].update(q=0.12, lo=0.12, hi=0.12)                         # union
    return _restate(g)


def test_b81_3_two_domains_violation_only_in_the_union_is_inconclusive(ds):
    g = _two_domains()
    assert g['g2']['state'] == 'chua_ket_luan' and g['outcome_decide'] == 'chua_ket_luan'
    lie = copy.deepcopy(g)
    lie['g2']['state'] = 'khong_dat'                                 # the one-domain rule on the union
    lie.update(outcome='khong_dat', outcome_decide='khong_dat', any_certain_fail=True)
    with pytest.raises(rl.LogError, match='common domain'):
        rl.validate_gate(lie, 'HW')
    common = copy.deepcopy(g)                                        # certain in the common domain: khong_dat
    common['g2']['chung'].update(q=0.11, lo=0.11, hi=0.11)
    assert _restate(common)['g2']['state'] == 'khong_dat'
    rl.validate_gate(common, 'HW')
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='disagrees'):
        rl.end(d, 'khong_dat', gate=g, results_dir='r', **BOOTS)
    rl.end(d, 'chua_ket_luan', gate=g, results_dir='r', **BOOTS)
    r = _rows(d)[-1]
    assert (r['g2_lo'], r['g2_lo_chung'], r['g1_t_done_1'], r['g1_t_done_2']) == ('0.12', '0.02', '10.0', '12.0')
    rl.summary(d)
    rows = _rows(d)                                                  # summary recomputes from the stored numbers
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, [dict(rows[0], g2_lo_chung='0.11', g2_q_chung='0.11',
                                                                    g2_hi_chung='0.11')])
    with pytest.raises(rl.LogError, match='recomputed from lo'):
        rl.summary(d)


@pytest.mark.parametrize('patch, match', [
    (lambda g: g['g2']['chung'].update(q=0.5, lo=0.5, hi=0.5), 'above that of the union'),
    (lambda g: g['g2']['chung'].update(n_undef=3), 'n_undef of the common domain'),
    (lambda g: g['g2']['chung'].pop('hi'), 'chung: not an object'),
    (lambda g: g['g1'].update(t_done=[10.0, 10.0]), 'numbers differ'),       # one domain: the sets must be equal
    (lambda g: g['g1'].update(t_done=[10.0, 25.0]), 'not before t_deadline'),
    (lambda g: g['g1'].update(t_done=[10.0]), 'way'),
    (lambda g: g['g1'].update(t_deadline=None), 't_deadline'),
    (lambda g: g['g1'].update(t_done=[10.0, float('nan')]), 'finite time'),
    (lambda g: g['g1'].update(prog_before_t1=True), 'fwsim / HIL fields'),
    (lambda g: g['g1'].update(cat_t1=True), 'needs t1'),
    (lambda g: g['g1'].update(t_done=[10.0, None]), 'recomputed from t_done'),   # state dat kept: a lie
])
def test_b81_3_floor_gate_checks(patch, match):
    g = _two_domains()
    patch(g)
    with pytest.raises(rl.LogError, match=match):
        rl.validate_gate(g, 'HW')


@pytest.mark.parametrize('t_done, state', [((10.0, 10.0), 'dat'), ((None, None), 'khong_dat'),
                                           ((10.0, None), 'chua_ket_luan'), ((None, 11.0), 'chua_ket_luan')])
def test_b81_3_floor_g1_from_the_two_ways(t_done, state):
    g = _hw_gate(t_done=t_done)
    assert g['g1']['state'] == state
    rl.validate_gate(g, 'HW')


@pytest.mark.parametrize('fields, state', [
    (dict(t_done=[10.0]), 'dat'), (dict(t_done=[None]), 'khong_dat'),
    (dict(t_done=[10.0], t1=5.0, prog_before_t1=True), 'dat'),
    (dict(t_done=[None], t1=5.0, prog_before_t1=True), 'khong_dat'),
    (dict(t_done=[None], t1=5.0, prog_before_t1=False, cond25_before_deadline=True), 'chua_ket_luan'),
    (dict(t_done=[None], t1=5.0, prog_before_t1=False, cond25_before_deadline=False), 'khong_dat'),
])
def test_b81_3_one_way_g1_from_t1_and_flags(fields, state):
    g = rl.fake_gate()
    g['g1'].update(t_deadline=20.0, **fields)
    g = _restate(g)
    assert g['g1']['state'] == state
    rl.validate_gate(g, 'SIM')
    rl.validate_gate(g, 'HIL')
    cols = rl.gate_columns(g)
    assert cols['g1_ways'] == '1' and cols['g1_t_done_2'] == ''
    back = rl.gate_from_row(dict(cols, run_uid='x'))
    assert rl.gate_g1_fields(back['g1'], 'SIM') == rl.gate_g1_fields(g['g1'], 'SIM')


@pytest.mark.parametrize('fields, match', [
    (dict(t_done=[10.0], t1=5.0), 'prog_before_t1'),
    (dict(t_done=[10.0], t1=5.0, prog_before_t1=False), 'cond25_before_deadline'),
    (dict(t_done=[10.0], prog_before_t1=True), 'without t1'),
    (dict(t_done=[10.0], t1=5.0, prog_before_t1=False, cond25_before_deadline=False), 'conditions 2-5'),
    (dict(t_done=[10.0], cat_t1=True), 'floor field'),
    (dict(t1=5.0), 'without t_done'),
])
def test_b81_3_one_way_g1_field_errors(fields, match):
    g = rl.fake_gate()
    g['g1'].update(t_deadline=20.0, **fields)
    with pytest.raises(rl.LogError, match=match):
        rl.validate_gate(g, 'SIM')


def test_b81_3_one_way_gate_with_two_ways_on_hil_is_refused():
    with pytest.raises(rl.LogError, match='way'):
        rl.validate_gate(_hw_gate(), 'HIL')
    g = rl.fake_gate()                                               # an o1_gate gate stays valid on SIM / HIL
    rl.validate_gate(g, 'SIM')
    cols = rl.gate_columns(g)
    assert cols['g2_q_chung'] == cols['g2_q'] and cols.get('g1_ways', '') == ''


def test_b81_3_cat_t1_and_the_violation_before_t1(ds):
    d, clock = ds
    g = rl.floor_gate(rl.fake_gate(e_trk=0.5), t_done=(None, None))
    g['g1'].update(t1=6.0, cat_t1=True)
    g = _restate(g)
    assert g['g1']['state'] == 'chua_ket_luan' and g['outcome_decide'] == 'khong_dat'
    _open(d, clock)
    with pytest.raises(rl.LogError, match='R2V-4'):                  # cat_t1 without the fail cause
        rl.end(d, 'khong_dat', gate=g, results_dir='r', **BOOTS)
    plain = rl.floor_gate(rl.fake_gate(e_trk=0.5))
    with pytest.raises(rl.LogError, match='R2V-4'):                  # the fail cause without cat_t1
        rl.end(d, 'khong_dat', gate=plain, results_dir='r', fail_cause=rl.VIOL_BEFORE, **BOOTS)
    with pytest.raises(rl.LogError, match='R2V-4'):                  # the fail cause without a gate
        rl.end(d, 'hong_he_do', fail_cause=rl.VIOL_BEFORE)
    rl.end(d, 'khong_dat', gate=g, results_dir='r', fail_cause=rl.VIOL_BEFORE, **BOOTS)
    r = _rows(d)[-1]
    assert (r['g1_cat_t1'], r['g1_t1'], r['g1_state']) == ('true', '6.0', 'chua_ket_luan')
    rl.summary(d)
    rows = _rows(d)
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, [dict(rows[0], fail_cause='')])
    with pytest.raises(rl.LogError, match='R2V-4'):
        rl.summary(d)


def test_b81_demo_writes_two_domain_rows(tmp_path):
    rl.demo(str(tmp_path / 'demo'))
    with open(os.path.join(str(tmp_path / 'demo'), rl.TUNING), newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    car = [r for r in rows if r['platform'] == 'HW' and r['run_outcome'] in rl.VALID]
    assert car and all(r['g1_ways'] == '2' and r['boot_before'] for r in car)
    inc = [r for r in car if r['run_outcome'] == 'chua_ket_luan']
    assert inc and inc[0]['g1_t_done_1'] and not inc[0]['g1_t_done_2']


# ── Codex Sol round 4 (Liam 14/11 CODE_R134 "Sua 6 muc, goi vong 5 (Recommended)") ──

def test_c4_1_end_needs_two_answers_not_two_strings(ds):
    """R134C4-1: a judged car / final_hil run needs the two read_query answers; end checks them itself."""
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='R134C4-1'):
        rl.end(d, 'khong_dat', boot_before='ab', boot_after='ab', **CAR_KD)
    qb, qa = rl.fake_queries('ab')
    for qs, why in (((qb, qb), 'cung mot file'), ((qa, qb), 'khong phai truoc va sau'),
                    ((qb, dict(qa, mac='cc')), 'hai board'), ((qb, None), 'sau')):
        with pytest.raises(rl.LogError, match=why):
            rl.end(d, 'khong_dat', queries=qs, **CAR_KD)
    row = rl.end(d, 'hong_he_do', fail_cause=rl.BOOT_FAIL, queries=(qb, qb))      # the failed run is logged
    assert row['fail_cause'] == rl.BOOT_FAIL
    rl.begin(d, 'S', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='RQ2-INIT-PID/G0')
    rl.step(d, 'run')
    with pytest.raises(rl.LogError, match='R134C4-1'):
        rl.end(d, 'khong_dat', gate=rl.fake_gate(e_trk=0.5), results_dir='r', boot_before='ab', boot_after='ab')
    rl.end(d, 'khong_dat', gate=rl.fake_gate(e_trk=0.5), results_dir='r', queries=(qb, qa))
    rl.begin(d, 'S', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1')
    rl.step(d, 'run')
    rl.end(d, 'khong_dat', gate=rl.fake_gate(e_trk=0.5), results_dir='r', boot_before='ab', boot_after='ab')
    rl.summary(d)                                                     # hil_check: the boots are not required


def _t1_gate(t_done=(None, None), t1=5.0, e_trk=0.03):
    g = rl.floor_gate(rl.fake_gate(e_trk=e_trk), t_done=t_done)
    g['g1'].update(t1=t1)
    return _restate(g)


@pytest.mark.parametrize('t_done, outcome', [((None, None), 'khong_dat'), ((10.0, None), 'chua_ket_luan'),
                                             ((10.0, 10.0), 'dat')])
def test_c4_3_floor_run_with_t1_is_lost(ds, t_done, outcome):
    """R134C4-3, O1 3.5: a floor run with a tracking error or a long gap (t1) and no violation before it is a run
    lost to the measurement; it is never judged."""
    d, clock = ds
    g = _t1_gate(t_done)
    assert g['outcome_decide'] == outcome
    _open(d, clock)
    with pytest.raises(rl.LogError, match='lost to the measurement'):
        rl.end(d, outcome, gate=g, results_dir='r', **BOOTS)
    rl.end(d, 'hong_he_do', fail_cause='loi_theo_doi', gate=g, results_dir='r')
    rl.summary(d)
    _open(d, clock)
    rl.end(d, outcome, gate=_hw_gate(t_done=t_done), results_dir='r', **BOOTS)   # the same run without t1
    rows = _rows(d)
    rows[-1]['g1_t1'] = '5.0'                                         # t1 written in by hand
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='lost to the measurement'):
        rl.summary(d)


def test_c4_4_completion_at_the_deadline_is_not_before_it(ds):
    """R134C4-4, O1 3.4 condition 4: completed strictly before t_deadline."""
    g = _hw_gate(t_done=(19.999, 19.999))
    rl.validate_gate(g, 'HW')
    for td in ([20.0, 20.0], [10.0, 20.0]):
        bad = copy.deepcopy(g)
        bad['g1']['t_done'] = td
        with pytest.raises(rl.LogError, match='not before t_deadline'):
            rl.validate_gate(bad, 'HW')
    one = rl.fake_gate()
    one['g1'].update(t_done=[20.0], t_deadline=20.0)
    with pytest.raises(rl.LogError, match='not before t_deadline'):
        rl.validate_gate(one, 'SIM')
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'dat', gate=g, results_dir='r', **BOOTS)
    rows = _rows(d)
    rows[-1]['g1_t_done_2'] = rows[-1]['g1_t_deadline']               # the CSV path too
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='not before t_deadline'):
        rl.summary(d)
