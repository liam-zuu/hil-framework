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
    assert [(r['run_uid'], r['outcome_tam'], r['outcome_sau'], r['flags']) for r in s] == \
        [('FAKE-A1:3', 'khong_dat', 'chua_ket_luan', 'calib_drift,calib_drift_split')]
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


def _score_final(path, o0, o1, vuot=True, thieu=False, outcome=None, flags=None, gate=None):
    final, want = rl.so_h_outcome(o0, o1, vuot, thieu)
    d = dict(outcome=outcome or final, gate_v3=gate or rl.floor_gate(rl.fake_gate(e_trk=0.5)),
             calib_drift=dict(vuot=vuot, thieu_H1=thieu, outcome_h0=o0, outcome_h1=o1, D_H_mm=3.0,
                              flags=want if flags is None else flags))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f)
    return str(path)


def test_so_h_checks_the_file_against_the_log_and_the_rule(demo_ds, tmp_path):
    uid = 'FAKE-A1:2'                                                   # khong_dat, camera gate e_trk 0.5
    with pytest.raises(rl.LogError, match='is not the logged outcome'):
        rl.so_h(demo_ds, uid, _score_final(tmp_path / 'a.json', 'dat', 'dat'))
    with pytest.raises(rl.LogError, match='disagree with the rule'):
        rl.so_h(demo_ds, uid, _score_final(tmp_path / 'b.json', 'khong_dat', 'dat', outcome='khong_dat'))
    with pytest.raises(rl.LogError, match='disagree with the rule'):
        rl.so_h(demo_ds, uid, _score_final(tmp_path / 'c.json', 'khong_dat', 'khong_dat', flags=[]))
    with pytest.raises(rl.LogError, match='gate of the file'):
        rl.so_h(demo_ds, uid, _score_final(tmp_path / 'd.json', 'khong_dat', 'khong_dat',
                                           gate=rl.floor_gate(rl.fake_gate(e_trk=0.6))))
    with pytest.raises(rl.LogError, match='car runs of a campaign'):
        rl.so_h(demo_ds, 'FAKE-FH:1', _score_final(tmp_path / 'e.json', 'dat', 'dat'))
    r = rl.so_h(demo_ds, uid, _score_final(tmp_path / 'f.json', 'khong_dat', 'hong_he_do'))
    assert (r['outcome_sau'], r['flags']) == ('hong_he_do', 'calib_drift')
    after = {x['run_uid']: x for x in rl.apply_so_h(rl.live_rows(rl.read_rows(os.path.join(demo_ds, rl.TUNING))),
                                                    demo_ds)}
    assert after[uid]['run_outcome'] == 'hong_he_do' and after[uid]['fail_cause'] == rl.FAIL_SO_H
    assert _summary(demo_ds)


def test_so_h_final_car_run_lost_after_the_comparison_leaves_the_table(demo_ds, tmp_path):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    fin = next(r for r in rows if r['stage'] == 'final' and r['run_outcome'] == 'dat')
    rl.so_h(demo_ds, fin['run_uid'], _score_final(tmp_path / 'g.json', 'dat', 'hong_he_do',
                                                  gate=rl.floor_gate(rl.fake_gate())))
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
    rl.so_h(demo_ds, fin['dat'], _score_final(tmp_path / 'h.json', 'dat', 'hong_he_do',
                                              gate=rl.floor_gate(rl.fake_gate())))
    rl.so_h(demo_ds, fin['chua_ket_luan'], _score_final(tmp_path / 'i.json', 'chua_ket_luan', 'hong_he_do',
                                                        gate=rl.floor_gate(rl.fake_gate(), t_done=(10.0, None))))
    with pytest.raises(rl.LogError, match='lost runs in a row'):
        rl.begin(demo_ds, 'FAKE-F2', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G00')


# ── B8S2-1, B8S2-2 (Codex B8-SPEC round 2; S05 v3.38 4.3, Liam 17/11 at THESIS_B8 "Duyệt, áp luôn (Recommended)") ──
def test_b8s2_1_end_refuses_a_score_final_file(demo_ds, tmp_path):
    rl.begin(demo_ds, 'FAKE-X', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G05')
    rl.step(demo_ds, 'run')
    sf = _score_final(tmp_path / 'sf.json', 'khong_dat', 'dat')
    qd = str(tmp_path / 'q')
    rl.fake_queries('ab', folder=qd)
    code = rl.main(['--dataset', demo_ds, 'end', '--outcome', 'chua_ket_luan', '--completed', '1',
                    '--results-dir', qd, '--gate-json', sf, '--query-before', os.path.join(qd, 'gain_query_before.txt'),
                    '--query-after', os.path.join(qd, 'gain_query_after.txt')])
    assert code != 0
    assert rl._load_open(demo_ds)                                       # the run is still open, nothing logged


def _fourth_final_run(ds, tmp_path):
    d = str(tmp_path / 'f1_4')
    qs = tuple(T.glc.read_query(p) for p in fr.write_final_run(d, 'HW', GAIN))
    rl.begin(ds, 'FAKE-F1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G00')
    rl.step(ds, 'run')
    return rl.end(ds, 'dat', completed=1, e_perp_max=0.1, results_dir=d, gate=rl.floor_gate(rl.fake_gate()),
                  queries=qs)


def test_b8s2_2_run_after_a_stop_seen_only_at_the_comparison(demo_ds, tmp_path):
    rows = _rows(os.path.join(demo_ds, rl.TUNING))
    fin = {r['run_outcome']: r['run_uid'] for r in rows if r['stage'] == 'final' and r['platform'] == 'HW'}
    late = _fourth_final_run(demo_ds, tmp_path)                         # taken before the comparison: no stop yet
    rl.so_h(demo_ds, fin['dat'], _score_final(tmp_path / 'h.json', 'dat', 'hong_he_do',
                                              gate=rl.floor_gate(rl.fake_gate())))
    rl.so_h(demo_ds, fin['chua_ket_luan'], _score_final(tmp_path / 'i.json', 'chua_ket_luan', 'hong_he_do',
                                                        gate=rl.floor_gate(rl.fake_gate(), t_done=(10.0, None))))
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
