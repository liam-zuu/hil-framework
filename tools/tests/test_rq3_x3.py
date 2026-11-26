"""S05 v3.48 regressions from the X3 review."""
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rq3_log as rl
import s05_final_table as table
from test_s05_final_table import GAIN7_FILE, LR, _table_ds, write_run


def test_x3_1_subsecond_split_and_source_total(tmp_path):
    ds = str(tmp_path)
    start, end = '2026-11-20T09:00:00', '2026-11-20T09:00:01'
    row = rl.machine(ds, 'B-chung', 'PID', 'X', 'r', 100, .5, start, end)
    assert float(row['t_machine_s']) == .5
    rl.machine(ds, 'B-chung', 'LADRC', 'X', 'r', 200, .5, start, end)
    with pytest.raises(rl.LogError, match='controller/account'):
        rl.machine(ds, 'B-chung', '', 'X', 'r', 300, 1, start, end)
    with pytest.raises(rl.LogError, match='duplicate'):
        rl.machine(ds, 'B-chung', 'PID', 'X', 'r', 100, .5, start, end)
    rl.append_row(str(tmp_path / rl.TUNING), rl.TUNING_COLS, dict(row, run_uid='source:1', controller=''))
    with pytest.raises(rl.LogError, match='controller/account'):
        rl.summary(ds)


def test_x3_1_c9h_phase_totals_and_s3(tmp_path):
    ds = str(tmp_path / 'demo')
    summary = {r['campaign']: r for r in rl.demo(ds)}
    rows = rl.read_rows(str(Path(ds) / rl.TUNING))
    grouped = {(r['script_run_id'], r['campaign'], r['controller']): r for r in rows if r['script_run_id']}
    assert len(grouped) == 6
    assert sum(float(r['t_machine_s']) for k, r in grouped.items() if k[0] == 'FAKE-S03') == pytest.approx(10801)
    assert sum(float(r['t_machine_s']) for k, r in grouped.items() if k[0] == 'FAKE-S3') == 1800
    assert all(k[1] == 'RQ2' for k in grouped if k[0] == 'FAKE-S3')
    assert summary['B-chung']['N_SIM_diagnostic'] == '7'
    with pytest.raises(rl.LogError, match='controller/account'):
        rl.machine(ds, 'B-chung', 'PID', 'bad-S3', 'r', 1, 1, '2026-11-20', '2026-11-21', script_kind='S3')


P1 = ('B-chung', '2026-11-20T09:00:00', '2026-11-20T09:10:00', '100', '200', '1', '2',
      'S03/run/console.txt:12-40', 'S03/run/sealed/notes.json:7')
P2 = ('RQ2', '2026-11-20T09:10:00', '2026-11-20T09:20:00', '50', '50', '0', '0',
      'S03/run/console.txt:41-60', 'S03/run/sealed/notes.json:9')


def test_x3_1_machine_phases_sources_in_every_row(tmp_path):
    # CODE 26/11: every typed number of machine-phases carries its source in the row
    rows = rl.machine_phases(str(tmp_path), 'S03', 'S03-R1', 'results/S03/R1', [P1, P2], 'quet')
    assert len(rows) == 4
    for r in rows:
        own = P1 if r['campaign'] == 'B-chung' else P2
        assert r['notes'].startswith(f'quet {rl.MANUAL} nguon pha {own[0]}: ')
        assert f'{own[1]}..{own[2]} PID {own[3]}/{own[5]} LADRC {own[4]}/{own[6]}' in r['notes']
        assert f'gio: {own[7]}; so: {own[8]}' in r['notes']
        other = P2 if own is P1 else P1
        assert other[7] not in r['notes'] and other[8] not in r['notes']
    on_disk = rl.read_rows(str(tmp_path / rl.TUNING))
    assert [r['notes'] for r in on_disk] == [r['notes'] for r in rows]


@pytest.mark.parametrize('i', [7, 8])
@pytest.mark.parametrize('bad', ['', '   ', 'a\nb', 'a\rb', None])
def test_x3_1_machine_phases_refuses_a_missing_source(tmp_path, i, bad):
    ph = list(P1)
    ph[i] = bad
    with pytest.raises(rl.LogError, match='source of the phase'):
        rl.machine_phases(str(tmp_path), 'S03', 'S03-R1', 'results/S03/R1', [tuple(ph)])
    assert not (tmp_path / rl.TUNING).exists()


def test_x3_1_machine_phases_refuses_a_second_import(tmp_path):
    ds = str(tmp_path)
    rl.machine_phases(ds, 'S03', 'S03-R1', 'results/S03/R1', [P1])
    before = (tmp_path / rl.TUNING).read_bytes()
    with pytest.raises(rl.LogError, match='duplicate script import'):
        rl.machine_phases(ds, 'S03', 'S03-R1', 'results/S03/R1', [P2])     # other phase, same script run
    assert (tmp_path / rl.TUNING).read_bytes() == before


def test_x3_1_machine_phases_needs_nine_fields(tmp_path):
    with pytest.raises(rl.LogError, match='9 fields'):
        rl.machine_phases(str(tmp_path), 'S03', 'S03-R1', 'results/S03/R1', [P1[:7]])


def test_x3_1_machine_phases_cli_takes_the_sources(tmp_path):
    ds = str(tmp_path)
    args = ['--dataset', ds, 'machine-phases', '--script-kind', 'S03', '--script-run-id', 'S03-R1',
            '--results-dir', 'results/S03/R1', '--phase', *P1]
    assert rl.main(args) == 0
    (row_pid, row_ladrc) = rl.read_rows(str(tmp_path / rl.TUNING))
    assert 'gio: S03/run/console.txt:12-40; so: S03/run/sealed/notes.json:7' in row_pid['notes']
    with pytest.raises(SystemExit):
        rl.main(['--dataset', str(tmp_path / 'x'), 'machine-phases', '--script-kind', 'S03', '--script-run-id', 'R',
                 '--results-dir', 'r', '--phase', *P1[:7]])


def test_x3_2_12b_is_budgeted_effort(tmp_path):
    ds = str(tmp_path)
    effort = ('2026-11-20', '09:00', '10:00')
    with pytest.raises(rl.LogError, match='12b.*hoc'):
        rl.effort(ds, *effort, campaign='A-PID', nhan='hoc', ghi_chu='12b lai')
    with pytest.raises(rl.LogError, match='12b'):
        rl.begin(ds, 'S', 'A-PID', 'HW', 'PID', '12b', '0.5', 'screen', gain_set='A-PID/G')
    rl.effort(ds, *effort, campaign='A-PID', ghi_chu='12b lai')
    a = rl.summary(ds)[0]
    assert (a['T_operator_h'], a['T_tune_no_hoc_h'], a['T_12b_h'], a['T_operator_no_12b_h'], a['N_HW_tune']) == (
        '1.000', '1.000', '1.000', '0.000', '0')


def test_x3_5_camera_score_schema_and_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, 'apply_moc', lambda rows, ds: rows)
    r = write_run(tmp_path, 'cam', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', .05)
    path = Path(r['results_dir']) / 'score.json'
    score = json.loads(path.read_text(encoding='utf-8'))
    score.pop('t_end', None)
    score.update(t_end_union=10.0, t0_log_us=score.get('t0_log_us', 300000))
    path.write_text(json.dumps(score), encoding='utf-8')
    r['gate_json_sha256'] = rl.sha256_file(str(path))
    ds = _table_ds(tmp_path, [r])
    runs, _ = LR(ds)
    assert runs[0]['values']['tv_duty'] == pytest.approx(4.0)
    with pytest.raises(ValueError, match='SHA-256'):
        table.read_run_score(dict(r, gate_json_sha256='0' * 64), ds, 'score.json')


def test_x3_5_real_c7_score_under_c8():
    root = Path('D:/CLAUDE_THESIS/05_DATA/results/C8')
    p = root / '2026-11-24_run06/c8b/T13_8728575/C8-L-F02-2_lai1/mot_nut/nut_012.json'
    if not p.is_file():
        pytest.skip('C8 camera score unavailable')
    row = dict(platform='HW', results_dir=str(p.parent), gate_json_sha256=rl.sha256_file(str(p)))
    score = table.read_run_score(row, str(p.parent), p.name)
    assert 't_end' not in score
    assert table.score_window(score, 'HW') == (score['t0_log_us'], score['t_end_union'], '')


def test_x3_6_envelope_and_gate_appendix(tmp_path):
    values, missing = table.run_values({'e_bao_max': -.08}, 'O1-K1', {})
    assert (values.get('e_bao_max'), values.get('e_bao_phys_max')) == (-.08, 0)
    r = dict(run_uid='r1', platform='HIL', controller='PID', condition='O1-K1', gain_set='A', outcome='dat',
             results_dir='', values=values, missing=missing, g4_q='.08', g4_lo='.07', g4_hi='.09', g4_state='dat')
    cells = table.write_table([r], str(tmp_path))
    e = next(c for c in cells if c['quantity'] == 'e_bao_max' and c['n_valid'])
    assert (e['n_planned'], e['n_g4_dat'], e['median']) == (5, 1, -.08)
    with (tmp_path / 'S05_final_runs.csv').open(encoding='utf-8', newline='') as f:
        appendix = next(csv.DictReader(f))
    assert [appendix[f'g4_{k}'] for k in ('q', 'lo', 'hi', 'state')] == ['.08', '.07', '.09', 'dat']
    assert 'r1: khong ap dung' in (tmp_path / 'S05_final_table.md').read_text(encoding='utf-8')


@pytest.mark.parametrize('bad', ['90', '100.5'])
def test_x3_7_duty_counts_interior_reset_or_fraction(bad):
    from test_s05_final_table import Log, zigzag

    log = Log()
    log.off(10)
    log.on(5, zigzag)
    active = [r for r in log.rows if r['run_active'] == '1']
    for r in active:
        r['dpwm_fl'] = '100'
    active[1]['dpwm_fl'] = bad
    tv = table.tv_duty_esp32(log.rows, .025, check_boot=False, check_gain_line=False)
    assert tv['tv_duty'] is None and 'dpwm_fl' in tv['reason']


def test_x3_8_table_reads_the_logged_queries(tmp_path):
    r = write_run(tmp_path, 'queries', 'HIL', 'final_hil', 'PID', 'O1-F05', 'A-PID/G07', 'dat', .03)
    d = Path(r['results_dir'])
    for cf, cs, when in ((*rl.QUERY_FILE_COLS[:2], 'before'), (*rl.QUERY_FILE_COLS[2:], 'after')):
        p = (d / f'gain_query_{when}.txt').rename(d / f'archived_{when}.txt')
        r[cf], r[cs] = str(p), rl.sha256_file(str(p))
    ds = _table_ds(tmp_path, [r])
    assert len(LR(ds)[0]) == 1
    p.write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='changed after'):
        LR(ds)


def test_x3_9_end_requires_trial_gain(tmp_path):
    ds = str(tmp_path)
    rl.begin(ds, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G')
    with pytest.raises(rl.LogError, match='judged tune run needs --gain-file'):
        rl.end(ds, 'dat', results_dir='missing', gate=rl.floor_gate(rl.fake_gate()), queries=rl.fake_queries())


def test_x3_9_summary_rechecks_trial_gain_and_legacy_firmware(tmp_path):
    ds = str(tmp_path / 'demo')
    rl.demo(ds)
    row = next(r for r in rl.read_rows(str(Path(ds) / rl.TUNING)) if r['run_uid'] == 'FAKE-A1:2')
    events = Path(row['results_dir']) / 'esp32.csv.events.txt'
    gf = rl.read_gate_file(str(Path(ds) / 'FAKE_gate_c8.json'), allow_fake=True)
    source = events.read_text(encoding='utf-8')
    events.write_text(source.replace('kp_xy=1.0', 'kp_xy=1.000001'), encoding='utf-8')
    with pytest.raises(rl.LogError, match='kp_xy'):
        rl.summary(ds, gate_file=gf)
    events.write_text('\n'.join(line.replace(',boot=4753dc47', '') if '#gain,' in line else line
                                for line in source.splitlines()), encoding='utf-8')
    with pytest.raises(rl.LogError, match='known legacy build'):
        rl.summary(ds, gate_file=gf)
    events.write_text(events.read_text(encoding='utf-8').replace('5bde4c55bf3a', '16702ff0e688'), encoding='utf-8')
    assert rl.summary(ds, gate_file=gf)


@pytest.mark.parametrize('replacement', [False, True])
def test_x3_10_table_checks_replacements_and_five_run_limit(tmp_path, replacement):
    rows = [write_run(tmp_path, f'r{i}', 'HIL', 'final_hil', 'PID', 'O1-K1', 'A-PID/G07', 'dat', .03)
            for i in range(2 if replacement else 6)]
    if replacement:
        rows[1]['replaces_run_uid'] = rows[0]['run_uid']
    with pytest.raises(ValueError, match='not a hong_he_do' if replacement else 'more than 5 valid'):
        LR(_table_ds(tmp_path, rows))


def test_x3_11_k_speed_and_n3_share_condition_key(tmp_path):
    with pytest.raises(rl.LogError, match='level 0.5'):
        rl.check_task('O1-K1', '0.2')
    rows = [dict(stage='final_hil', campaign='RQ2', gain_set_id='A-PID/G07', platform='HIL',
                 trajectory='O1-K1', speed_mps='0.5' if i < 3 else '0.50', run_uid=f'S:{i}',
                 run_outcome='hong_he_do') for i in range(4)]
    with pytest.raises(rl.LogError, match='stopped after 3'):
        rl._final_stops(rows)
    rl._check_replacements([rows[0], dict(rows[3], replaces_run_uid=rows[0]['run_uid'])])


def test_x3_13_over_the_rerun_limit_only_post_freeze_or_explore(tmp_path):
    # enough kiem_gio_c8b refusals to stop the level (> C8B_RERUNS_MAX), but each set only after frozen_at (screen)
    # or in explore: neither is history of the block; the same refusals before frozen_at in screen do stop it
    ds = str(tmp_path)
    n = rl.C8B_RERUNS_MAX + 1
    common = dict(campaign='A-PID', platform='HW', trajectory='O1-F05', speed_mps='0.5', gain_set_id='A-PID/G')
    block = [dict(common, stage='confirm', block_id='B', run_uid=f'B:{i}', run_outcome='dat',
                  wall_end='2026-11-20T09:00:00') for i in range(2)]
    for r in block:
        rl.append_row(str(tmp_path / rl.SO_H), rl.SO_H_COLS, dict(run_uid=r['run_uid'], ghi_luc='2026-11-20T09:10:00'))

    def refused(stage, hour, tag):
        return [dict(common, stage=stage, run_uid=f'{tag}:{i}', run_outcome='hong_he_do', fail_cause=rl.C8B_GIO_FAIL,
                     wall_end=f'2026-11-20T{hour:02d}:{i:02d}:00') for i in range(n)]

    frz = '2026-11-20T10:00:00'
    for extra in (refused('screen', 11, 'S'), refused('explore', 8, 'E')):
        assert rl.confirm_block_problem(block + extra, ds, 'A-PID', 'A-PID/G', frz, ['O1-F05']) is None
    why = rl.confirm_block_problem(block + refused('screen', 8, 'P'), ds, 'A-PID', 'A-PID/G', frz, ['O1-F05'])
    assert why and 'stopped after' in why


def test_x3_13_confirm_ignores_post_freeze_and_explore(tmp_path):
    ds = str(tmp_path)
    common = dict(campaign='A-PID', platform='HW', trajectory='O1-F05', speed_mps='0.5', gain_set_id='A-PID/G')
    runs = [dict(common, stage='confirm', block_id='B', run_uid=f'B:{i}', run_outcome='dat',
                 wall_end='2026-11-20T09:00:00') for i in range(2)]
    for r in runs:
        rl.append_row(str(tmp_path / rl.SO_H), rl.SO_H_COLS, dict(run_uid=r['run_uid'], ghi_luc='2026-11-20T09:10:00'))
    runs += [dict(common, stage='explore' if i < 2 else 'screen', run_uid=f'E:{i}', run_outcome='hong_he_do',
                  fail_cause=rl.C8B_GIO_FAIL, wall_end=f'2026-11-20T{9 if i < 2 else 11:02d}:00:00') for i in range(4)]
    assert rl.confirm_block_problem(runs, ds, 'A-PID', 'A-PID/G', '2026-11-20T10:00:00', ['O1-F05']) is None


def test_x3_14_final_requires_an_earlier_unique_freeze(tmp_path, monkeypatch):
    ds = str(tmp_path)
    import datetime as dt
    monkeypatch.setattr(rl, 'now', lambda: dt.datetime.fromisoformat('2026-11-20T09:00:00'))
    args = (ds, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final')
    with pytest.raises(rl.LogError, match='needs a row'):
        rl.begin(*args, gain_set='A-PID/G')
    f = tmp_path / 'gain_X.json'
    f.write_text(json.dumps(GAIN7_FILE), encoding='utf-8')
    fr = rl.freeze(ds, 'A-PID', 'cham_tran_gio',
                   'A-PID/G', str(f), at='2026-11-20T09:01:00')
    with pytest.raises(rl.LogError, match='after frozen_at'):
        rl.begin(*args, gain_set='A-PID/G')
    rl.append_row(str(tmp_path / rl.FREEZE), rl.FREEZE_COLS, fr)
    with pytest.raises(rl.LogError, match='frozen twice'):
        rl.summary(ds)


def test_x3_15_same_account_and_numeric_checks_on_write_and_replay(tmp_path):
    ds = str(tmp_path)
    args = ('B-chung', 'PDI', 'X', 'r', 100, float('inf'), '2026-11-20', '2026-11-21')
    with pytest.raises(rl.LogError, match='controller'):
        rl.machine(ds, *args)
    with pytest.raises(rl.LogError, match='finite number'):
        rl.machine(ds, args[0], 'PID', *args[2:])
    p = str(tmp_path / rl.EFFORT)
    rl.append_row(p, rl.EFFORT_COLS, dict(chien_dich='A-PID', khoan='T_H', gio='1'))
    with pytest.raises(rl.LogError, match='exactly one'):
        rl.summary(ds)
    rl._write_all(p, rl.EFFORT_COLS, [dict(chien_dich='A-PID', gio='abc')])
    with pytest.raises(rl.LogError, match='gio.*finite'):
        rl.summary(ds)


@pytest.mark.parametrize('heading,missing', [(2., False), (None, True)])
def test_x3_18_so_h_requires_h1_for_heading_or_missing_quantity(tmp_path, heading, missing):
    ds = str(tmp_path / 'demo')
    rl.demo(ds)
    uid = 'FAKE-A1:3'
    original = next(r for r in rl.read_rows(str(Path(ds) / rl.SO_H)) if r['run_uid'] == uid)
    path = Path(original['score_final_file'])
    d = json.loads(path.read_text(encoding='utf-8'))
    cd = d['calib_drift']
    cd.update(D_H_mm=.5, s0_session_sha256='0' * 64, s1_session_sha256='1' * 64)
    path.write_text(json.dumps(d), encoding='utf-8')
    drift = {k: cd[k] for k in ('s0_session_sha256', 's1_session_sha256', 'D_H_mm', 'eps_H_mm', 'thieu_H1', 'vuot')}
    drift.update(kind='c7_calib_drift', version='c7_calib_drift_v2', D_H_huong_deg=heading, eps_H_huong_deg=1.,
                 trang_thai_vi_tri='so', trang_thai_huong='thieu' if missing else 'so', vuot_do_thieu=missing)
    run = next(r for r in rl.read_rows(str(Path(ds) / rl.TUNING)) if r['run_uid'] == uid)
    path.write_text(json.dumps(dict(d, calib_drift={**cd, **drift})), encoding='utf-8')
    assert rl.check_score_final(run, str(path), ds)[0] == 'chua_ket_luan'
    path.write_text(json.dumps(d), encoding='utf-8')
    drift_path = tmp_path / 'drift.json'
    drift_path.write_text(json.dumps(drift), encoding='utf-8')
    so = str(Path(ds) / rl.SO_H)
    rl._write_all(so, rl.SO_H_COLS, [r for r in rl.read_rows(so) if r['run_uid'] != uid])
    assert rl.main(['--dataset', ds, 'so-h', '--run-uid', uid, '--score-final', str(path),
                    '--drift', str(drift_path)]) == 0
    row = next(r for r in rl.read_rows(so) if r['run_uid'] == uid)
    assert (row['vuot'], row['vuot_do_thieu'], row['outcome_sau']) == (
        'true', str(missing).lower(), 'chua_ket_luan')
    assert row['D_H_huong_deg'] == ('' if missing else '2.0')
    assert rl.apply_so_h(rl.read_rows(str(Path(ds) / rl.TUNING)), ds)
