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
        rl.machine_phases(ds, 'S3', 'bad-S3', 'r', [P1])


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
    del cd['version']  # historical gop omitted both version and heading evidence
    cd.update(D_H_mm=.5, s0_session_sha256='0' * 64, s1_session_sha256='1' * 64)
    path.write_text(json.dumps(d), encoding='utf-8')
    drift = {k: cd[k] for k in ('s0_session_sha256', 's1_session_sha256', 'D_H_mm', 'eps_H_mm', 'thieu_H1', 'vuot')}
    drift.update(kind='c7_calib_drift', version='c7_calib_drift_v2', tasks=['O1-F05'], tag_offset=[0, 0, 0],
                 k_start=None, D_H_vi_tri_mm=.5, D_H_huong_deg=heading, eps_H_huong_deg=1.,
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


@pytest.mark.parametrize('kind', ['S03', 'S3'])
def test_x3_1_round3_machine_requires_phases(tmp_path, kind):
    args = ('RQ2', 'PID', 'X', 'r', 1, 1, P1[1], P1[2])
    with pytest.raises(rl.LogError, match='machine-phases'):
        rl.machine(str(tmp_path), *args, script_kind=kind)
    flags = ('campaign', 'controller', 'script-run-id', 'results-dir', 'n-sim-evals', 't-machine-s',
             'wall-start', 'wall-end')
    cli = [x for k, v in zip(flags, args) for x in ('--' + k, str(v))]
    assert rl.main(['--dataset', str(tmp_path), 'machine', *cli, '--script-kind', kind]) != 0
    assert not (tmp_path / rl.TUNING).exists()


@pytest.mark.parametrize('ph,match', [(P1[:7] + ('x', P1[8]), 'source'), (P1, 'duplicate'),
                                    (P1[:8] + ('sealed/*.json:3',), 'source'),
                                    (P1[:8] + ('notes.json:9-4',), 'source')])
def test_x3_1_round3_phase_source_and_duplicate(tmp_path, ph, match):
    with pytest.raises(rl.LogError, match=match):
        rl.machine_phases(str(tmp_path), 'S03', 'X', 'r', [P1, ph])
    assert not (tmp_path / rl.TUNING).exists()


def test_x3_6_round3_hil_appendix_preserves_scorer_fields(tmp_path):
    r = write_run(tmp_path, 'r', 'HIL', 'final_hil', 'PID', 'O1-F05', 'A-PID/G07', 'dat', .03)
    path = Path(r['results_dir']) / 'score.json'
    score = json.loads(path.read_text(encoding='utf-8'))
    gate = score['gate_v3']
    gate['g1'].update(t_done=[9.5], t_deadline=12., t1=None,
                      prog_before_t1=None, cond25_before_deadline=None)
    metrics = dict(e_trk_rms=.03, stop_dist_origin=.04, ds_min=-.01, n_upper_frames=3,
                   tv_duty_h7_1khz=8., sat_any_frac=.1)
    blocks = dict(start_pose=[.1, .2, .3], pi_period_steps=[2, 42], edge_map={'esp_pi_ppm': -2.1},
                  imu_noise={'seed': 17}, gate_v3=gate)
    path.write_text(json.dumps(dict(score, **metrics, **blocks)), encoding='utf-8')
    r.update(rl.gate_columns(gate, rl.sha256_file(str(path))))
    ds = _table_ds(tmp_path, [r])
    table.write_table(LR(ds)[0], str(tmp_path))
    with (tmp_path / 'S05_final_runs.csv').open(encoding='utf-8', newline='') as f:
        saved = next(csv.DictReader(f))
    assert {k: float(saved[k]) for k in metrics} == metrics
    assert {k: json.loads(saved[k]) for k in blocks} == blocks
    assert (saved['g1_t_done_1'], saved['g1_t_deadline']) == (r['g1_t_done_1'], r['g1_t_deadline'])
    assert float(saved['tv_duty']) == pytest.approx(4.)
    ds = _table_ds(tmp_path, [dict(r, g1_t_done_1='9.6')])
    with pytest.raises(ValueError, match='g1_t_done_1'):
        LR(ds)


@pytest.mark.parametrize('task', ['O1-F05', 'O1-K1'])
def test_x3_6_round3_appendix_covers_actual_hil_schema(tmp_path, task):
    from test_gate_v3_tools import H, write_frame_log, write_k_frame_log
    # No ESP32 edge mapping: this is a scored, lost HIL run. Its numbers still belong in the appendix.
    row = write_run(tmp_path, 'run', 'HIL', 'final_hil', 'PID', task, 'A-PID/G07', 'hong_he_do', .03)
    frames = str(tmp_path / 'frames.npz')
    if task == 'O1-F05':
        write_frame_log(frames, task)
    else:
        write_k_frame_log(frames, task, .02)
    score = H._json_safe(H.score(frames, task, allow_final_set=True))
    path = Path(row['results_dir']) / 'score.json'
    path.write_text(json.dumps(score), encoding='utf-8')
    row.update(rl.gate_columns(score['gate_v3'], rl.sha256_file(str(path))), fail_cause=rl.MAP_REFUSED)
    ds = _table_ds(tmp_path, [row])
    table.write_table(LR(ds)[0], str(tmp_path))
    with (tmp_path / 'S05_final_runs.csv').open(encoding='utf-8', newline='') as f:
        saved = next(csv.DictReader(f))
    for name, value in score.items():
        if type(value) in (float, int):
            assert float(saved[name]) == value, name
    assert json.loads(saved['gate_v3']) == score['gate_v3']


def test_x3_8_round3_end_and_summary_check_final_hil_boot(tmp_path):
    import s05_fake_runs as fr
    ds = str(tmp_path)
    d = tmp_path / 'run'
    fr.write_trial_gain(str(d), boot='abcd')
    rl.begin(ds, 'S', 'RQ2', 'HIL', 'PID', 'O1-F05', '.5', 'final_hil', gain_set='RQ2-INIT-PID/G1')
    kw = dict(results_dir=str(d), gate=rl.fake_gate())
    with pytest.raises(rl.LogError, match='boot'):
        rl.end(ds, 'dat', queries=rl.fake_queries('dcba', folder=ds), **kw)
    ep = d / 'esp32.csv.events.txt'
    events = ep.read_text(encoding='utf-8')
    ep.write_text('\n'.join(line.replace('boot=abcd', 'boot=dcba') if '#gain,' in line else line
                            for line in events.splitlines()), encoding='utf-8')
    with pytest.raises(rl.LogError, match='boot'):
        rl.end(ds, 'dat', queries=rl.fake_queries('abcd', folder=ds), **kw)
    ep.write_text(events, encoding='utf-8')
    rl.end(ds, 'dat', queries=rl.fake_queries('abcd', folder=ds), **kw)
    ep.write_text(events.replace('boot=abcd', 'boot=dcba'), encoding='utf-8')
    with pytest.raises(rl.LogError, match='boot'):
        rl.summary(ds)


def test_x3_8_round3_gain_send_saves_query_tx(monkeypatch, capsys, tmp_path):
    import gain_line_check as glc
    import gain_send
    import s05_fake_runs as fr
    gain = fr.sample_gain()
    ack = dict(gain['gains'], ctrl='PID', id='1', applied_n='1', mac='aa:bb', boot='ab', sha='123abc',
               env='pid', ok='1', tx='472')
    monkeypatch.setattr(gain_send, 'exchange', lambda *a: (ack, 1))
    assert gain_send.main(['--query', '--ip', 'localhost']) == 0
    reply = capsys.readouterr().out
    queries = []
    for i in (1, 2):
        path = tmp_path / f'query{i}.txt'
        path.write_text(glc.query_pc_line(i, i + .1) + reply, encoding='utf-8')
        queries.append(glc.read_query(str(path)))
    assert queries[0]['tx'] == '472'
    assert 'cung tx' in glc.query_problem(*queries, check_run_boot=False)


def test_x3_9_round3_trial_gain_write_and_replay(tmp_path):
    import s05_fake_runs as fr
    ds = str(tmp_path)
    d = tmp_path / 'run'
    path = fr.write_trial_gain(str(d), 'LADRC')
    args = (ds, 'S', 'B-LADRC', 'HIL', 'LADRC', 'O1-F05', '.5', 'hil_check')
    rl.begin(*args, gain_set='B-LADRC/G1', gains='wc_xy=9999')
    kw = dict(results_dir=str(d), gate=rl.fake_gate(), gain_file=path)
    with pytest.raises(rl.LogError, match='g_wc_xy'):
        rl.end(ds, 'dat', **kw)
    rl.cancel(ds)
    rl.begin(*args, gain_set='B-LADRC/G1', gains='wc_xy=2.00000001')
    r = rl.end(ds, 'dat', **kw)
    assert float(r['g_wo_xy']) == 6 and not r.get('g_kp_xy')
    assert rl.summary(ds)
    rl._write_all(str(tmp_path / rl.TUNING), rl.TUNING_COLS, [dict(r, g_wc_xy='9999')])
    with pytest.raises(rl.LogError, match='g_wc_xy'):
        rl.summary(ds)


@pytest.mark.parametrize('bad,match', [(dict(gain_file='missing.json'), 'not found'),
                                      (dict(gain_sha256='0' * 64), 'gain_sha256'),
                                      (dict(frozen_at='bad'), 'frozen_at'),
                                      (dict(frozen_at='2026-11-19'), 'frozen_at')])
def test_x3_14_round3_replay_each_freeze(tmp_path, bad, match):
    from test_s05_final_table import write_gains
    ds = str(tmp_path)
    rl.migrate(ds)
    write_gains(ds)
    p = str(tmp_path / rl.FREEZE)
    rl._write_all(p, rl.FREEZE_COLS, [dict(rl.read_rows(p)[0], **bad)])
    with pytest.raises(rl.LogError, match=match):
        rl.summary(ds)
    with pytest.raises(ValueError, match=match):
        LR(ds)


def test_x3_14_round3_begin_final_hil_requires_its_freeze(tmp_path):
    from test_s05_final_table import write_gains
    ds = str(tmp_path)
    args = (ds, 'S', 'RQ2', 'HIL', 'PID', 'O1-F05', '.5', 'final_hil')
    with pytest.raises(rl.LogError, match='needs a row'):
        rl.begin(*args, gain_set='A-PID/G07')
    write_gains(ds)
    with pytest.raises(rl.LogError, match='frozen'):
        rl.begin(*args, gain_set='A-PID/G999')
    assert not (tmp_path / rl.OPEN_RUN).exists()


def test_x3v2_1_round3_legacy_hours_are_reported_apart(tmp_path, capsys):
    ds = str(tmp_path)
    rl.append_row(str(tmp_path / rl.EFFORT), rl.EFFORT_COLS, dict(chien_dich='chua_biet', gio='1.17'))
    rl.effort(ds, '2026-11-20', '10:00', '11:00', khoan='C_chung')
    totals = {r['campaign']: float(r['T_operator_h']) for r in rl.summary(ds)}
    assert sum(totals.values()) == totals['C_chung'] == 1
    assert '1.17' in capsys.readouterr().err
    with pytest.raises(rl.LogError):
        rl.effort(ds, '2026-11-20', '10:00', '11:00', campaign='chua_biet')


@pytest.mark.parametrize('platform,stage', [('HIL', 'final_hil'), ('HW', 'final')])
@pytest.mark.parametrize('outcome,n,match', [('dat', 6, 'more than 5 valid'), ('hong_he_do', 4, 'stopped after 3')])
def test_x3v2_3_round3_initial_gain_one_cell(tmp_path, outcome, n, match, platform, stage):
    rows = [write_run(tmp_path, f'r{i}', platform, stage, 'PID', 'O1-F05', f'RQ2-INIT-PID/G{i % 2 + 1}',
                      outcome, .03) for i in range(n)]
    ds = _table_ds(tmp_path, rows[:-1])
    summary = {r['campaign']: r for r in rl.summary(ds)}
    field = 'n_valid' if outcome == 'dat' else 'stopped_n3'
    assert summary['RQ2-INIT-PID'][stage + '_O1-F05@0.5_' + field] == ('5' if outcome == 'dat' else '1')
    _table_ds(tmp_path, rows)
    with pytest.raises(rl.LogError, match=match):
        rl.summary(ds)
    with pytest.raises(ValueError, match=match):
        LR(ds)


def test_x3v2_3_round3_initial_cell_requires_the_run_account(tmp_path):
    row = write_run(tmp_path, 'run', 'HIL', 'final_hil', 'PID', 'O1-F05', 'RQ2-INIT-PID/G1', 'hong_he_do', .03)
    ds = _table_ds(tmp_path, [dict(row, campaign='A-PID')])
    with pytest.raises(ValueError, match='account RQ2'):
        LR(ds)
    row.update(stage='final', platform='HW', campaign='RQ2-INIT-PID', gain_set_id='RQ2-INIT-LADRC/G1')
    ds = _table_ds(tmp_path, [row])
    with pytest.raises(rl.LogError, match='does not belong'):
        rl.summary(ds)


@pytest.mark.parametrize('second_edge', [False, True])
def test_x3v2_4_round3_t0_must_match_unique_edge(second_edge):
    from test_s05_final_table import Log, zigzag
    log = Log()
    log.off(10)
    log.on(15, zigzag)
    active = [r for r in log.rows if r['run_active'] == '1']
    if second_edge:
        active[10]['run_active'] = '0'
    t0 = int(active[0 if second_edge else 1]['timestamp_us'])
    tv = table.tv_duty_esp32(log.rows, .05, t0_us=t0, check_gain_line=False, check_boot=False)
    assert tv['tv_duty'] is None
    assert ('run_active' if second_edge else 't0_log_us') in tv['reason']


def test_x3v2_4_round3_t0_needs_a_recorded_zero():
    from test_s05_final_table import Log, zigzag
    log = Log()
    log.off(10)
    log.on(15, zigzag)
    i = next(i for i, r in enumerate(log.rows) if r['run_active'] == '1')
    log.rows[i - 1]['run_active'] = ''
    tv = table.tv_duty_esp32(log.rows, .05, t0_us=int(log.rows[i]['timestamp_us']),
                             check_gain_line=False, check_boot=False)
    assert tv['tv_duty'] is None and 'run_active' in tv['reason']
    log.rows[i - 1].update(run_active='0', _segment='earlier boot')
    tv = table.tv_duty_esp32(log.rows, .05, t0_us=int(log.rows[i]['timestamp_us']),
                             check_gain_line=False, check_boot=False)
    assert tv['tv_duty'] is None and 'doan logger' in tv['reason']
    del log.rows[i - 1]['_segment']
    t0 = int(log.rows[i]['timestamp_us'])
    log.rows[i]['timestamp_us'] = f'{t0}.5'
    tv = table.tv_duty_esp32(log.rows, .05, t0_us=t0, check_gain_line=False, check_boot=False)
    assert tv['tv_duty'] is None and 'timestamp_us' in tv['reason']


def test_x3v2_7_round3_query_names_check_logged_pair(tmp_path):
    r = write_run(tmp_path, 'r', 'HIL', 'final_hil', 'PID', 'O1-F05', 'A-PID/G07', 'dat', .03)
    ds = _table_ds(tmp_path, [r])
    args = ['table', '--dataset', ds, '--out', str(tmp_path / 'out'),
            '--query-before-name', 'gain_query_before.txt', '--query-after-name', 'gain_query_after.txt']
    assert table.main(args) == 0
    assert table.main(args[:-1] + ['gain_query_before.txt']) == 2
    other = Path(r['results_dir']) / 'copy'
    other.mkdir()
    (other / 'gain_query_before.txt').write_bytes(Path(r['query_before_file']).read_bytes())
    assert table.main(args[:5] + ['--query-before-name', 'copy/gain_query_before.txt']) == 2
    # A failed run can still name both answers. An explicit name must agree with any bound answer.
    r.update(run_outcome='hong_he_do', fail_cause='loi_ban')
    r.update({k: '' for k in rl.GATE_COLS})
    rl._write_all(str(Path(ds) / rl.TUNING), rl.TUNING_COLS, [r])
    assert table.main(args) == 0
    assert table.main(args[:-1] + ['gain_query_before.txt']) == 2


def test_x3v2_8_round3_bench_log_end_migrate_summary_table(tmp_path):
    import s05_fake_runs as fr
    ds = str(tmp_path)
    old_cols = [c for c in rl.TUNING_COLS if c != 'esp32_log']
    old = dict(session_id='EXAMPLE', notes='old results')
    rl._write_all(str(tmp_path / rl.TUNING), old_cols, [old])
    assert rl.migrate(ds)[rl.TUNING] == 'migrated'
    migrated, = rl.read_rows(str(tmp_path / rl.TUNING))
    assert (migrated['esp32_log'], migrated['notes']) == ('', 'old results')
    for stage, campaign, gs in [('hil_check', 'B-PID', 'B-PID/G1'), ('final_hil', 'RQ2', 'RQ2-INIT-PID/G1')]:
        d = tmp_path / stage
        d.mkdir()
        gain = fr.sample_gain()
        before, after = fr.write_final_run(str(d), 'HIL', gain, score_extra=dict(outcome='dat', gate_v3=rl.fake_gate()))
        gf = fr.write_gain_file(str(d / 'gain_trial.json'), gain)
        (d / 'pi').mkdir()
        for suffix in ['', '.events.txt']:
            (d / ('esp32.csv' + suffix)).rename(d / ('pi/F05-PID.csv' + suffix))
        rl.begin(ds, 'S', campaign, 'HIL', 'PID', 'O1-F05', '.5', stage, gain_set=gs)
        args = ['--dataset', ds, 'end', '--outcome', 'dat', '--gate-json', str(d / 'score.json'),
                '--results-dir', str(d), '--esp32-log', 'pi/F05-PID.csv', '--gain-file', gf]
        if stage == 'final_hil':
            args += ['--query-before', before, '--query-after', after]
        assert rl.main(args) == 0
    rows = rl.live_rows(rl.read_rows(str(tmp_path / rl.TUNING)))
    assert all(r['esp32_log'].replace('\\', '/').endswith('pi/F05-PID.csv') for r in rows)
    assert rl.summary(ds)
    assert LR(ds, init_gain={'PID': gf})[0][0]['values']['tv_duty'] is not None
    # A default-named copy cannot stand in for the logged events, on either reader.
    d = tmp_path / 'final_hil'
    for suffix in ('', '.events.txt'):
        (d / ('esp32.csv' + suffix)).write_bytes((d / ('pi/F05-PID.csv' + suffix)).read_bytes())
    (d / 'pi/F05-PID.csv.events.txt').unlink()
    with pytest.raises(rl.LogError, match='events'):
        rl.summary(ds)
    with pytest.raises(ValueError, match='events'):
        LR(ds, init_gain={'PID': gf})
    (tmp_path / 'hil_check/pi/F05-PID.csv.events.txt').unlink()
    with pytest.raises(rl.LogError, match='events'):
        rl.check_trial_gain(rows[0], ds)


def test_x3v2_9_round3_freeze_and_final_same_second(tmp_path, monkeypatch):
    import datetime as dt
    ds = str(tmp_path)
    t = dt.datetime(2026, 11, 20, 9, 0, 0, 123456)
    clock = iter([t, t + dt.timedelta(microseconds=1)])
    monkeypatch.setattr(rl, 'now', lambda: next(clock))
    p = tmp_path / 'gain_PID.json'
    p.write_text(json.dumps(GAIN7_FILE), encoding='utf-8')
    frozen = rl.freeze(ds, 'A-PID', 'het_ung_vien', 'A-PID/G1', str(p))
    run = rl.begin(ds, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '.5', 'final', gain_set='A-PID/G1')
    assert (frozen['frozen_at'], run['row']['wall_start']) == (t.isoformat(), t.replace(microsecond=123457).isoformat())
