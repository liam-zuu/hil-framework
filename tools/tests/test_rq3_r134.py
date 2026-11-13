"""rq3_log R134 (b): boot of gain_send --query before and after the run (S05 4.7 (iv), R3V-9).
Liam 14/11 CODE_R134 "duyet goi sua code fix/r134-answers". Kept apart from test_gate_v3_tools.py."""
import csv
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rq3_log as rl  # noqa: E402

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
             gain_set='A-PID/G1' if platform == 'HIL' or stage == 'final' else '')
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)


def _rows(d):
    with open(os.path.join(d, rl.TUNING), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_boot_columns_recorded(ds):
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'khong_dat', boot_before='4753dc47', boot_after='4753DC47')
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
    with pytest.raises(rl.LogError, match='needs the two boot answers'):
        rl.end(d, 'hong_he_do', fail_cause=rl.BOOT_FAIL)


def test_summary_refuses_a_judged_row_with_two_boots(ds):
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'khong_dat', boot_before='ab', boot_after='ab')
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
