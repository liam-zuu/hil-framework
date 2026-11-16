"""Tests of C9 dry run (rq3_log demo builds every judged run folder) and R134 (1): R134C10-1 (log_seq wrap with a
'#gain' line of first_seq = 0) and the summary reading the two saved query answers again (R134C4-1 summary part).
CODE 16/11/2026 to CODE_C13; Liam 16/11 at CODE "Sửa demo + gộp R134(1) (Recommended)"; Liam 16/11 at CODE_C13
"Ghi đường dẫn + SHA (Recommended)"."""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rq3_log as rl  # noqa: E402
import s05_final_table as T  # noqa: E402
import s05_fake_runs as fr  # noqa: E402

GAIN = dict(ctrl='PID', id=1, gains=dict(kp_xy=1.0, ki_xy=0.5, kp_wz=1.0, ki_wz=0.2, pos_kp=6.0, pos_ki=3.0, k_psi=1.0))


def _rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def demo_ds(tmp_path):
    d = str(tmp_path / 'dry')
    rl.demo(d)
    return d


# ── C9 dry run: the whole chain ──────────────────────────────────────────────────────────────────────────────────
def test_demo_final_table_runs_the_whole_chain(demo_ds, tmp_path):
    out = str(tmp_path / 'tbl')
    os.makedirs(out)
    assert T.main(['table', '--dataset', demo_ds, '--out', out]) == 0
    runs = _rows(os.path.join(out, 'S05_final_runs.csv'))
    judged = [r for r in runs if r.get('outcome') in T.VALID]
    assert len(judged) == 4                                      # two final_hil, two judged car final runs
    for r in judged:
        assert r.get('tv_duty') not in ('', None), r              # tv_duty from the ESP32 counters
    hil = [r for r in judged if r.get('platform') == 'HIL']
    assert len(hil) == 2 and all(r.get('t_end_err_ms') for r in hil)


def test_demo_rows_name_their_query_files(demo_ds):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    need = [r for r in rows if r.get('campaign') in rl.CAMPAIGNS and r.get('run_outcome') in rl.VALID
            and rl.boot_needed(r.get('platform'), r.get('stage'))]
    assert len(need) == 6                                        # 2 tune car, 2 final_hil, 2 final car
    for r in need:
        for c in rl.QUERY_FILE_COLS:
            assert r.get(c), (r['run_uid'], c)
        assert os.path.isfile(r['query_before_file']) and r['query_before_file'].startswith(os.path.abspath(demo_ds))
        assert rl.sha256_file(r['query_after_file']) == r['query_after_sha256']


# ── R134 (1): the summary reads the two saved answers again ───────────────────────────────────────────────────────
def _summary(ds):
    gate, sha = rl.read_gate(os.path.join(ds, 'FAKE_gate.csv'))
    return rl.summary(ds, gate, sha)


def _judged_car_row(rows):
    return next(i for i, r in enumerate(rows) if r.get('platform') == 'HW' and r.get('run_outcome') in rl.VALID
                and r.get('campaign') in rl.CAMPAIGNS)


def test_summary_accepts_the_demo(demo_ds):
    assert _summary(demo_ds)


def test_summary_refuses_a_query_file_changed_after_end(demo_ds):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    p = rows[_judged_car_row(rows)]['query_after_file']
    with open(p, 'a', encoding='utf-8') as f:
        f.write('\n')                                             # still readable, other bytes
    with pytest.raises(rl.LogError, match='changed after the run was logged'):
        _summary(demo_ds)


def test_summary_refuses_a_missing_query_file(demo_ds):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    os.remove(rows[_judged_car_row(rows)]['query_before_file'])
    with pytest.raises(rl.LogError, match='not found'):
        _summary(demo_ds)


def test_summary_refuses_a_judged_car_row_without_its_query_files(demo_ds):
    path = os.path.join(demo_ds, rl.TUNING)
    rows = _rows(path)
    i = _judged_car_row(rows)
    for c in rl.QUERY_FILE_COLS:
        rows[i][c] = ''
    _write_rows(path, rows)
    with pytest.raises(rl.LogError, match='cannot be read again'):
        _summary(demo_ds)


def test_summary_refuses_boot_columns_edited_by_hand(demo_ds):
    path = os.path.join(demo_ds, rl.TUNING)
    rows = _rows(path)
    i = _judged_car_row(rows)
    rows[i]['boot_before'] = rows[i]['boot_after'] = 'abcdef01'  # still one boot, other than the saved answers
    _write_rows(path, rows)
    with pytest.raises(rl.LogError, match='differ from the saved answers'):
        _summary(demo_ds)


def test_summary_refuses_two_answers_that_are_not_two_questions(demo_ds):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    r = rows[_judged_car_row(rows)]
    with open(r['query_after_file'], encoding='utf-8') as f:
        text = f.read()
    with open(r['query_before_file'], 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)                                             # the after answer twice: one question
    path = os.path.join(demo_ds, rl.TUNING)
    rows[_judged_car_row(rows)]['query_before_sha256'] = rl.sha256_file(r['query_before_file'])
    _write_rows(path, rows)
    with pytest.raises(rl.LogError, match='two saved answers'):
        _summary(demo_ds)


def test_end_records_the_query_files(tmp_path):
    ds = str(tmp_path / 'ds')
    os.makedirs(ds)
    rl.migrate(ds)
    qdir = str(tmp_path / 'q')
    qs = rl.fake_queries('ab', folder=qdir)
    rl.begin(ds, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')
    rl.step(ds, 'run')
    row = rl.end(ds, 'khong_dat', completed=1, e_perp_max=0.2, results_dir=qdir,
                 gate=rl.floor_gate(rl.fake_gate(e_trk=0.5)), queries=qs)
    assert row['query_before_file'] == os.path.abspath(os.path.join(qdir, 'gain_query_before.txt'))
    assert row['query_after_sha256'] == rl.sha256_file(os.path.join(qdir, 'gain_query_after.txt'))


# ── R134C10-1: log_seq wrap and a '#gain' line with first_seq = 0 ────────────────────────────────────────────────
def _run_near_wrap(tmp_path, seq0, extra_end=()):
    d = str(tmp_path / f'run_{seq0}')
    log = fr.Log(seq0=seq0)
    log.off(30)
    log.on(1003)
    log.off(20, after_on=True)
    fr.write_final_run(d, 'HIL', GAIN, log=log, run_end_extra=extra_end)
    return os.path.join(d, 'esp32.csv')


def test_c10_1_set_line_far_from_the_wrap_is_still_a_setting(tmp_path):
    assert T.run_gain_problem(_run_near_wrap(tmp_path, 50), _want(tmp_path)) is None


def test_c10_1_data_near_the_wrap_refuses_an_unanchored_first_seq_0(tmp_path):
    p = _run_near_wrap(tmp_path, fr_seq := T.SEQ_MAX - 3000)
    assert fr_seq > T.SEQ_MAX - T.SEQ_WRAP_MARGIN
    why = T.run_gain_problem(p, _want(tmp_path))
    assert why and 'R134C10-1' in why


def test_c10_1_run_end_near_the_wrap_alone_is_enough(tmp_path):
    p = _run_near_wrap(tmp_path, 50, extra_end=(T.SEQ_MAX,))      # data lines after the wrap lost
    why = T.run_gain_problem(p, _want(tmp_path))
    assert why and 'R134C10-1' in why


def test_c10_1_seq_wrap_evidence():
    assert T.seq_wrap_evidence([10, 11, 12], [13]) is None
    assert 'giam' in T.seq_wrap_evidence([T.SEQ_MAX - 70000, 5], [])
    assert 'gan' in T.seq_wrap_evidence([1, 2], [T.SEQ_MAX - 10])
    assert T.seq_wrap_evidence([None, 3], [None]) is None


def _want(tmp_path):
    import gain_line_check as glc
    return glc.load_gain_file(fr.write_gain_file(str(tmp_path / 'g.json'), GAIN))


# ── S05 v3.37 4.3 (B8S-3), O1 4 clip H: outcome after the clip H comparison (so-h) ──────────────────────────────────
@pytest.mark.parametrize('o0,o1,vuot,thieu,want', [
    ('dat', None, False, False, ('dat', [])),
    ('khong_dat', None, True, True, ('chua_ket_luan', ['thieu_H1'])),
    ('hong_he_do', None, True, True, ('hong_he_do', ['thieu_H1'])),
    ('dat', 'dat', True, False, ('dat', ['calib_drift'])),
    ('dat', 'khong_dat', True, False, ('chua_ket_luan', ['calib_drift', 'calib_drift_split'])),
    ('dat', 'hong_he_do', True, False, ('hong_he_do', ['calib_drift'])),
    ('chua_ket_luan', 'dat', True, False, ('chua_ket_luan', ['calib_drift', 'calib_drift_split'])),
])
def test_so_h_rule(o0, o1, vuot, thieu, want):
    assert rl.so_h_outcome(o0, o1, vuot, thieu) == want


def _a1_3(ds):
    return next(r for r in rl.live_rows(rl.read_rows(os.path.join(ds, rl.TUNING)))
                if r['run_uid'] == 'FAKE-A1:3')


def test_demo_so_h_row_and_summary_use_the_outcome_after(demo_ds):
    s = _rows(os.path.join(demo_ds, rl.SO_H))
    assert [(r['run_uid'], r['outcome_tam'], r['outcome_sau'], r['flags'], r['vuot']) for r in s] == [
        ('FAKE-A1:2', 'khong_dat', 'khong_dat', '', 'false'),            # C9D-2: below eps_H, given too
        ('FAKE-A1:3', 'khong_dat', 'chua_ket_luan', 'calib_drift,calib_drift_split', 'true'),
        ('FAKE-F1:2', 'dat', 'dat', '', 'false'),
        ('FAKE-F1:3', 'chua_ket_luan', 'chua_ket_luan', '', 'false')]
    assert _a1_3(demo_ds)['run_outcome'] == 'khong_dat'                 # the tuning log row is not changed
    after = {r['run_uid']: r for r in rl.apply_so_h(rl.live_rows(rl.read_rows(os.path.join(demo_ds, rl.TUNING))),
                                                    demo_ds)}
    assert after['FAKE-A1:3']['run_outcome'] == 'chua_ket_luan'
    assert after['FAKE-A1:3']['so_h_flags'] == 'calib_drift,calib_drift_split'


def test_so_h_refuses_twice_and_a_changed_file(demo_ds):
    sf = os.path.join(demo_ds, 'luot', 'a1_2', 'score_final.json')
    with pytest.raises(rl.LogError, match='already in'):
        rl.so_h(demo_ds, 'FAKE-A1:3', sf)
    with open(sf, 'a', encoding='utf-8') as f:
        f.write(' ')
    with pytest.raises(rl.LogError, match='missing or changed since so-h'):
        _summary(demo_ds)


GATES = {'dat': lambda: rl.floor_gate(rl.fake_gate()),                    # camera gates that recompute to the outcome
         'khong_dat': lambda: rl.floor_gate(rl.fake_gate(e_trk=0.5)),
         'chua_ket_luan': lambda: rl.floor_gate(rl.fake_gate(), t_done=(10.0, None))}


def _row(ds, uid):
    return next(r for r in rl.live_rows(rl.read_rows(os.path.join(ds, rl.TUNING))) if r['run_uid'] == uid)


def _drop_so_h(ds, *uids):
    """The so-h rows of `uids` taken out again (the demo gives every judged car run its comparison)."""
    p = os.path.join(ds, rl.SO_H)
    rows = [r for r in _rows(p) if r['run_uid'] not in uids]
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rl.SO_H_COLS)
        w.writeheader()
        w.writerows(rows)


def _score_final(ds, uid, d_h, o1=None, h1_gate=None, name='sf_test.json', edit=None, folder=None):
    """score_final of run uid in its run folder (rl.fake_score_final: its H0 file score.json + calib_drift); above
    eps_H (2.0) the H1 score has outcome o1 and a gate that recomputes to it unless h1_gate is given; edit(d)
    changes the dict before it is written."""
    folder = folder or _row(ds, uid)['results_dir']
    h1 = None
    if o1 is not None:
        h1 = dict(FAKE=True, outcome=o1)
        if h1_gate is not None or o1 in GATES:
            h1['gate_v3'] = h1_gate if h1_gate is not None else GATES[o1]()
    p = rl.fake_score_final(folder, d_h, h1=h1, name=name)
    if edit:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        edit(d)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(d, f)
    return p


def test_so_h_checks_the_file_against_the_log_and_the_rule(demo_ds):
    uid = 'FAKE-A1:2'                                                   # khong_dat, camera gate e_trk 0.5
    _drop_so_h(demo_ds, uid)
    with pytest.raises(rl.LogError, match='is not the logged outcome'):
        rl.so_h(demo_ds, uid, _score_final(demo_ds, uid, 3.0, 'dat', name='a.json',
                                           edit=lambda d: d['calib_drift'].update(outcome_h0='dat')))
    with pytest.raises(rl.LogError, match='disagree with the rule'):
        rl.so_h(demo_ds, uid, _score_final(demo_ds, uid, 3.0, 'dat', name='b.json',
                                           edit=lambda d: d.update(outcome='khong_dat')))
    with pytest.raises(rl.LogError, match='disagree with the rule'):
        rl.so_h(demo_ds, uid, _score_final(demo_ds, uid, 3.0, 'khong_dat', name='c.json',
                                           edit=lambda d: d['calib_drift'].update(flags=[])))
    with pytest.raises(rl.LogError, match='car runs of a campaign'):
        rl.so_h(demo_ds, 'FAKE-FH:1', _score_final(demo_ds, uid, 3.0, 'dat', name='e.json'))
    r = rl.so_h(demo_ds, uid, _score_final(demo_ds, uid, 3.0, 'hong_he_do', name='f.json'))
    assert (r['outcome_sau'], r['flags']) == ('hong_he_do', 'calib_drift')
    after = {x['run_uid']: x for x in rl.apply_so_h(rl.live_rows(rl.read_rows(os.path.join(demo_ds, rl.TUNING))),
                                                    demo_ds)}
    assert after[uid]['run_outcome'] == 'hong_he_do' and after[uid]['fail_cause'] == rl.FAIL_SO_H
    assert _summary(demo_ds)


def test_so_h_final_car_run_lost_after_the_comparison_leaves_the_table(demo_ds, tmp_path):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    fin = next(r for r in rows if r['stage'] == 'final' and r['run_outcome'] == 'dat')
    _drop_so_h(demo_ds, fin['run_uid'])
    rl.so_h(demo_ds, fin['run_uid'], _score_final(demo_ds, fin['run_uid'], 3.0, 'hong_he_do'))
    out = str(tmp_path / 'tbl')
    os.makedirs(out)
    assert T.main(['table', '--dataset', demo_ds, '--out', out]) == 0
    runs = {r['run_uid']: r for r in _rows(os.path.join(out, 'S05_final_runs.csv'))}
    assert runs[fin['run_uid']]['outcome'] == 'hong_he_do' and runs[fin['run_uid']]['so_h_flags'] == 'calib_drift'
    assert _summary(demo_ds)


def test_so_h_lost_runs_count_in_the_n3_streak(demo_ds, tmp_path):
    # S05 v3.37 4.3: the N = 3 streak of 4.6 uses the outcome after the comparison
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    fin = {r['run_outcome']: r['run_uid'] for r in rows if r['stage'] == 'final' and r['platform'] == 'HW'}
    _drop_so_h(demo_ds, fin['dat'], fin['chua_ket_luan'])
    rl.so_h(demo_ds, fin['dat'], _score_final(demo_ds, fin['dat'], 3.0, 'hong_he_do'))
    rl.so_h(demo_ds, fin['chua_ket_luan'], _score_final(demo_ds, fin['chua_ket_luan'], 3.0, 'hong_he_do'))
    with pytest.raises(rl.LogError, match='lost runs in a row'):
        rl.begin(demo_ds, 'FAKE-F2', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G00')


# ── B8S2-1, B8S2-2 (Codex B8-SPEC round 2; S05 v3.38 4.3, Liam 17/11 at THESIS_B8 "Duyệt, áp luôn (Recommended)") ──
def test_b8s2_1_end_refuses_a_score_final_file(demo_ds, tmp_path):
    rl.begin(demo_ds, 'FAKE-X', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G05')
    rl.step(demo_ds, 'run')
    sf = _score_final(demo_ds, 'FAKE-A1:2', 3.0, 'dat', name='sf_end.json')
    qd = str(tmp_path / 'q')
    rl.fake_queries('ab', folder=qd)
    code = rl.main(['--dataset', demo_ds, 'end', '--outcome', 'chua_ket_luan', '--completed', '1',
                    '--results-dir', qd, '--gate-json', sf, '--query-before', os.path.join(qd, 'gain_query_before.txt'),
                    '--query-after', os.path.join(qd, 'gain_query_after.txt')])
    assert code != 0
    assert rl._load_open(demo_ds)                                       # the run is still open, nothing logged


def _judged_run(ds, d, campaign, ctrl, stage, outcome, gain_set, block_id='', platform='HW'):
    """A judged run with its folder and scorer file score.json (outcome, gate_v3) given to end with its sha256."""
    qs = tuple(T.glc.read_query(p) for p in fr.write_final_run(
        d, platform, GAIN, score_extra=dict(outcome=outcome, gate_v3=GATES[outcome]())))
    sp = os.path.join(d, 'score.json')
    rl.begin(ds, 'FAKE-' + stage, campaign, platform, ctrl, 'O1-F05', '0.5', stage, gain_set=gain_set,
             block_id=block_id)
    rl.step(ds, 'run')
    return rl.end(ds, outcome, completed=1, e_perp_max=0.1, results_dir=d, gate=rl.read_gate_json(sp),
                  gate_sha256=rl.sha256_file(sp), queries=qs)


def _fourth_final_run(ds, tmp_path):
    return _judged_run(ds, str(tmp_path / 'f1_4'), 'A-PID', 'PID', 'final', 'dat', 'A-PID/G00')


def test_b8s2_2_run_after_a_stop_seen_only_at_the_comparison(demo_ds, tmp_path):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    fin = {r['run_outcome']: r['run_uid'] for r in rows if r['stage'] == 'final' and r['platform'] == 'HW'}
    late = _fourth_final_run(demo_ds, tmp_path)                         # taken before the comparison: no stop yet
    _drop_so_h(demo_ds, fin['dat'], fin['chua_ket_luan'])
    rl.so_h(demo_ds, fin['dat'], _score_final(demo_ds, fin['dat'], 3.0, 'hong_he_do'))
    rl.so_h(demo_ds, fin['chua_ket_luan'], _score_final(demo_ds, fin['chua_ket_luan'], 3.0, 'hong_he_do'))
    rl.so_h(demo_ds, late['run_uid'], _score_final(demo_ds, late['run_uid'], 0.5))   # C9D-2: the late run too
    after = {r['run_uid']: r for r in rl.apply_so_h(rl.live_rows(rl.read_rows(os.path.join(demo_ds, rl.TUNING))),
                                                    demo_ds)}
    assert after[late['run_uid']].get(rl.RETRO_FLAG) == '1'
    s = {r['campaign']: r for r in _summary(demo_ds)}['A-PID']        # no error: the late run is kept, flagged
    assert s[f'{rl.RETRO_FLAG}_n'] == '1'
    assert s.get('final_O1-F05@0.5_n_valid', '0') == '0' and s['final_O1-F05@0.5_invalid'] == '3'
    assert s['N_HW'] == '7'                                             # 3 tune + 3 final + the late run
    out = str(tmp_path / 'tbl')
    os.makedirs(out)
    assert T.main(['table', '--dataset', demo_ds, '--out', out]) == 0
    assert late['run_uid'] not in {r['run_uid'] for r in _rows(os.path.join(out, 'S05_final_runs.csv'))}
    with pytest.raises(rl.LogError, match='lost runs in a row'):        # and no run after it any more
        rl.begin(demo_ds, 'FAKE-F2', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G00')


def test_b8s2_2_a_stop_seen_at_the_time_still_refuses_a_later_run(tmp_path):
    rows = [dict(run_uid=f'S:{k}', stage='final', campaign='A-PID', gain_set_id='A-PID/G00', platform='HW',
                 trajectory='O1-F05', speed_mps='0.5', run_outcome=o, fail_cause='x' if o == 'hong_he_do' else '')
            for k, o in enumerate(('hong_he_do', 'hong_he_do', 'hong_he_do', 'dat'), start=1)]
    assert not any(r.get(rl.RETRO_FLAG) for r in rl.mark_retro_stops(rows))
    with pytest.raises(rl.LogError, match='no run after that'):
        rl._final_stops(rows)


# ── C9D-1…7 (Codex C9 demo round 1; Liam 17/11 at CODE_C13 "Sửa cả 7 + vòng 2 (Recommended)") ─────────────────────────
def test_c9d1_score_final_of_another_run_with_the_same_outcome_and_gate(demo_ds):
    # FAKE-A1:2 and FAKE-A1:3 are both khong_dat with the same camera gate: a file of A1:3 is not one of A1:2
    _drop_so_h(demo_ds, 'FAKE-A1:2')
    other = _score_final(demo_ds, 'FAKE-A1:3', 1.0, name='sf_other.json')
    with pytest.raises(rl.LogError, match='not under the results_dir'):
        rl.so_h(demo_ds, 'FAKE-A1:2', other)


def test_c9d1_score_final_not_built_on_the_h0_file_of_the_run(demo_ds):
    _drop_so_h(demo_ds, 'FAKE-A1:2')
    p = _score_final(demo_ds, 'FAKE-A1:2', 1.0, edit=lambda d: d.update(e_perp_max=0.5))
    with pytest.raises(rl.LogError, match='not the H0 scorer file'):
        rl.so_h(demo_ds, 'FAKE-A1:2', p)


def test_c9d1_run_logged_without_its_scorer_sha(demo_ds):
    path = os.path.join(demo_ds, rl.TUNING)
    rows = _rows(path)
    i = next(k for k, r in enumerate(rows) if r['run_uid'] == 'FAKE-A1:2')
    rows[i]['gate_json_sha256'] = ''
    _write_rows(path, rows)
    _drop_so_h(demo_ds, 'FAKE-A1:2')
    with pytest.raises(rl.LogError, match='C9D-1'):
        rl.so_h(demo_ds, 'FAKE-A1:2', _score_final(demo_ds, 'FAKE-A1:2', 1.0))


def test_c9d1_summary_checks_the_binding_again(demo_ds):
    p = os.path.join(_row(demo_ds, 'FAKE-A1:2')['results_dir'], 'score.json')
    with open(p, 'a', encoding='utf-8') as f:
        f.write(' ')                                                    # the H0 file changed after so-h
    with pytest.raises(rl.LogError, match='C9D-1'):
        _summary(demo_ds)


def test_c9d2_summary_counts_judged_car_runs_without_so_h(demo_ds):
    assert {r['campaign']: r for r in _summary(demo_ds)}['A-PID']['so_h_chua_n'] == '0'
    _drop_so_h(demo_ds, 'FAKE-A1:2', 'FAKE-F1:3')
    assert {r['campaign']: r for r in _summary(demo_ds)}['A-PID']['so_h_chua_n'] == '2'


def test_c9d2_final_table_stops_before_so_h_of_a_final_car_run(demo_ds, tmp_path):
    _drop_so_h(demo_ds, 'FAKE-F1:3')
    out = str(tmp_path / 'tbl')
    os.makedirs(out)
    assert T.main(['table', '--dataset', demo_ds, '--out', out]) != 0
    assert not os.path.exists(os.path.join(out, 'S05_final_runs.csv'))


def test_c9d2_final_table_does_not_need_so_h_of_tune_runs(demo_ds, tmp_path):
    _drop_so_h(demo_ds, 'FAKE-A1:2')                                   # screen run: counted in the summary only
    out = str(tmp_path / 'tbl')
    os.makedirs(out)
    assert T.main(['table', '--dataset', demo_ds, '--out', out]) == 0


def _confirm_block(ds, tmp_path, block, outcomes):
    return [_judged_run(ds, str(tmp_path / f'c_{block}_{k}'), 'A-LADRC', 'LADRC', 'confirm', o, 'A-LADRC/G01',
                        block_id=block)['run_uid'] for k, o in enumerate(outcomes)]


def _freeze_ok(ds, tmp_path, at=''):
    return rl.freeze(ds, 'A-LADRC', 'khoi_xac_nhan_dat', 'A-LADRC/G01',
                     fr.write_gain_file(str(tmp_path / 'FAKE_gain_L.json'), GAIN), at=at)


def test_c9d4_freeze_needs_the_so_h_of_the_last_block(demo_ds, tmp_path):
    uids = _confirm_block(demo_ds, tmp_path, 'K1', ('dat', 'dat'))
    with pytest.raises(rl.LogError, match='without their clip H comparison'):
        _freeze_ok(demo_ds, tmp_path)
    for u in uids:
        rl.so_h(demo_ds, u, _score_final(demo_ds, u, 0.5))
    with pytest.raises(rl.LogError, match='earlier than the last so-h'):
        _freeze_ok(demo_ds, tmp_path, at='2026-01-01T00:00:00')
    _freeze_ok(demo_ds, tmp_path)
    assert {r['campaign']: r for r in _summary(demo_ds)}['A-LADRC']['stop_reason_check'] == 'ok'


def test_c9d4_freeze_needs_dat_after_the_comparison(demo_ds, tmp_path):
    _confirm_block(demo_ds, tmp_path, 'K1', ('khong_dat',))             # an earlier block does not count
    uids = _confirm_block(demo_ds, tmp_path, 'K2', ('dat', 'dat'))
    rl.so_h(demo_ds, uids[0], _score_final(demo_ds, uids[0], 0.5))
    rl.so_h(demo_ds, uids[1], _score_final(demo_ds, uids[1], 3.0, 'khong_dat'))   # split: chua_ket_luan
    with pytest.raises(rl.LogError, match='not all dat'):
        _freeze_ok(demo_ds, tmp_path)


def test_c9d4_summary_mismatch_for_a_freeze_row_the_log_does_not_back(demo_ds, tmp_path):
    _confirm_block(demo_ds, tmp_path, 'K1', ('dat',))
    gf = fr.write_gain_file(str(tmp_path / 'FAKE_gain_L.json'), GAIN)
    rl.append_row(os.path.join(demo_ds, rl.FREEZE), rl.FREEZE_COLS, dict(
        campaign='A-LADRC', frozen_at='2030-01-01T00:00:00', stop_reason='khoi_xac_nhan_dat', stop_cause='',
        gain_set_id='A-LADRC/G01', gain_file=gf, gain_sha256=rl.sha256_file(gf), notes='FAKE: by hand'))
    assert {r['campaign']: r for r in _summary(demo_ds)}['A-LADRC']['stop_reason_check'] == 'mismatch'


def _so_h_edit(demo_ds, d_h, o1, edit):
    _drop_so_h(demo_ds, 'FAKE-A1:2')
    return _score_final(demo_ds, 'FAKE-A1:2', d_h, o1, edit=edit)


def test_c9d5_vuot_is_computed_from_d_h(demo_ds):
    def below(d):                                                       # claims below eps_H with D_H 3.0 > 2.0
        d['outcome'] = 'khong_dat'
        d['calib_drift'].update(vuot=False, flags=[], outcome_h1=None)
        d['calib_drift'].pop('h1_score')
    with pytest.raises(rl.LogError, match='C9D-5'):
        rl.so_h(demo_ds, 'FAKE-A1:2', _so_h_edit(demo_ds, 3.0, 'dat', below))


def test_c9d5_outcome_h1_is_the_one_of_the_h1_score(demo_ds):
    def h1_says_dat(d):                                                 # block khong_dat, kept H1 score dat
        d['calib_drift']['h1_score']['outcome'] = 'dat'
    with pytest.raises(rl.LogError, match='C9D-5'):
        rl.so_h(demo_ds, 'FAKE-A1:2', _so_h_edit(demo_ds, 3.0, 'khong_dat', h1_says_dat))


def test_c9d5_h1_gate_recomputes_to_its_outcome(demo_ds):
    _drop_so_h(demo_ds, 'FAKE-A1:2')
    p = _score_final(demo_ds, 'FAKE-A1:2', 3.0, 'dat', h1_gate=GATES['khong_dat']())
    with pytest.raises(rl.LogError, match='disagrees with the gate of the H1 score'):
        rl.so_h(demo_ds, 'FAKE-A1:2', p)


def test_c9d5_thieu_h1_has_no_d_h(demo_ds):
    with pytest.raises(rl.LogError, match='C9D-5'):
        rl.so_h(demo_ds, 'FAKE-A1:2', _so_h_edit(demo_ds, None, None,
                                                 lambda d: d['calib_drift'].update(D_H_mm=5.0)))


def test_c9d6_every_judged_run_of_the_demo_has_its_folder(demo_ds):
    rows = [r for r in rl.live_rows(rl.read_rows(os.path.join(demo_ds, rl.TUNING))) if r.get('campaign')]
    judged = [r for r in rows if r['run_outcome'] in rl.VALID]
    assert {r['run_uid'] for r in judged} >= {'FAKE-A1:2', 'FAKE-A1:3', 'FAKE-B1:2', 'FAKE-F1:2', 'FAKE-F1:3'}
    for r in judged + [r for r in rows if any(r.get(c) for c in rl.GATE_COLS)]:
        d = r['results_dir']
        assert os.path.isdir(d), r['run_uid']
        for name in ('score.json', 'esp32.csv', 'esp32.csv.events.txt'):
            assert os.path.isfile(os.path.join(d, name)), (r['run_uid'], name)
        assert rl.sha256_file(rl.h0_file(r, demo_ds)) == r['gate_json_sha256']


def test_c9d7_first_seq_near_the_wrap_in_the_events_is_evidence(tmp_path):
    p = _run_near_wrap(tmp_path, 50)                                    # data far from the wrap
    assert T.run_gain_problem(p, _want(tmp_path)) is None
    s = T.SEQ_MAX - 5                                                   # a start whose data lines were lost
    lines = fr.stamp([f'#fw,sha={fr.SHA},dirty=0,env=pid'] * 3 + [fr.gain_line(GAIN, s)] * 3
                     + [f'#us,ev=start,boot={fr.BOOT},on=0,swb=1500,first_seq={s},line_drop=0'] * 3,
                     '2026-11-20T10:00:30')
    with open(p + '.events.txt', 'a', encoding='utf-8', newline='\n') as f:
        f.write(''.join(f'{x}\n' for x in lines))
    why = T.run_gain_problem(p, _want(tmp_path))
    assert why and 'R134C10-1' in why
