"""rq3_log (C9, spec S05 §6): migration keeps every value, a live run sums its segments (pause not
counted, wait reported apart, not added), refusals, the B-chung split, caps, final counts, and the
C9 dry run: a made-up campaign written through the commands is recomputed into S05_summary.csv.
Every number here is invented. Run: python -m pytest tools/tests/test_rq3_log.py"""
import csv
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rq3_log as rl  # noqa: E402

OLD_TUNING = ('session_id,date,platform,controller,trajectory,iter_no,t_flash_s,t_setup_s,t_run_s,t_log_s,'
              't_analysis_s,gain_changed,old_value,new_value,rmse_vx,rmse_vy,rmse_wz,stop_met,bug_id,notes,dut,buoc\n'
              'EXAMPLE-S01,2026-06-15,HIL,PID,circle,1,45,30,60,15,180,Kp_vx,1.5,2.0,0.082,0.095,0.110,N,,'
              'vi du - xoa dong nay,,\n'
              'PREHIL-R01,2026-06-10,HW,PID,manual,,,,,,,Kp_vx,?,1.5,,,,Y,,reconstruct,,\n')
OLD_EFFORT = ('ngay,bat_dau,ket_thuc,gio,buoc,giai_doan,mot_lan_hay_thich_nghi,dut,ghi_chu,chien_dich,lap_lai\n'
              '2026-11-05,20:00,22:41,2.68,,,,,"Liam nguyên văn: ""sửa giờ luôn đi""",,chua_biet\n')
OLD_HW = 'ngay,session_id,chien_dich,nen_tang,thu_hong,nguyen_nhan,gio_sua,ghi_chu,dut,iter_no,buoc,lap_lai\n'


@pytest.fixture
def ds(tmp_path, monkeypatch):
    (tmp_path / rl.TUNING).write_text(OLD_TUNING, encoding='utf-8')
    (tmp_path / rl.EFFORT).write_text(OLD_EFFORT, encoding='utf-8')
    (tmp_path / rl.HWFAIL).write_text(OLD_HW, encoding='utf-8')
    clock = [dt.datetime(2026, 11, 20, 8, 0, 0)]
    monkeypatch.setattr(rl, 'now', lambda: clock[0])
    d = str(tmp_path)
    d_clock = clock
    return d, d_clock


def tick(clock, s):
    clock[0] += dt.timedelta(seconds=s)


def gain_file(d, name='g.json'):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        with open(p, 'w', encoding='utf-8') as f:
            f.write('{"kp_xy": 1.0}\n')
    return p


def car_run(d, clock, campaign, stage, outcome, task='O1-F05', speed='0.5', s=60, **kw):
    rl.begin(d, kw.pop('session', 'S'), campaign, kw.pop('platform', 'HW'), 'PID', task, speed, stage, **kw)
    rl.step(d, 'run')
    tick(clock, s)
    return rl.end(d, outcome, fail_cause='mat_khung' if outcome == 'hong_he_do' else '')


def rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_migrate_adds_columns_keeps_values_and_is_idempotent(ds):
    d, _ = ds
    before = {n: rows(os.path.join(d, n)) for n in (rl.TUNING, rl.EFFORT)}
    assert rl.migrate(d) == {rl.TUNING: 'migrated', rl.EFFORT: 'migrated', rl.HWFAIL: 'ok'}
    for n, cols in ((rl.TUNING, rl.TUNING_COLS), (rl.EFFORT, rl.EFFORT_COLS)):
        after = rows(os.path.join(d, n))
        assert list(after[0]) == cols and len(after) == len(before[n])
        for b, a in zip(before[n], after):
            assert all(a[k] == v for k, v in b.items())          # every old value kept
            assert all(a[k] == '' for k in cols if k not in b)
    assert rl.migrate(d) == {rl.TUNING: 'ok', rl.EFFORT: 'ok', rl.HWFAIL: 'ok'}


def test_migrate_refuses_unknown_columns(ds):
    d, _ = ds
    with open(os.path.join(d, rl.HWFAIL), 'w', encoding='utf-8') as f:
        f.write('ngay,la\n')
    with pytest.raises(rl.LogError):
        rl.migrate(d)


def test_append_needs_migration(ds):
    d, _ = ds
    with pytest.raises(rl.LogError, match='migrate'):
        rl.effort(d, '2026-11-20', '08:00', '09:00', khoan='C_chung', ghi_chu='x')


def test_live_run_segments_pause_and_wait(ds):
    d, clock = ds
    rl.migrate(d)
    rl.begin(d, 'A1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='G1', gains='kp_xy=1.2,k_psi=1')
    rl.step(d, 'flash'); tick(clock, 100)
    rl.step(d, 'run', wait=True); tick(clock, 60)
    rl.pause(d); tick(clock, 900)                 # other work
    rl.resume(d); tick(clock, 20)                 # still 'run', still waiting
    rl.step(d, 'analysis'); tick(clock, 200)
    row = rl.end(d, 'khong_dat', completed=1, e_perp_max=0.123)
    assert row['run_uid'] == 'A1:1' and row['t_flash_s'] == '100' and row['t_run_s'] == '80'
    assert row['t_analysis_s'] == '200' and row['t_operator_wait_s'] == '80' and row['t_setup_s'] == '0'
    assert row['g_kp_xy'] == '1.2' and row['g_k_psi'] == '1' and row['e_perp_max_m'] == '0.1230'
    assert row['wall_start'] == '2026-11-20T08:00:00' and row['wall_end'] == '2026-11-20T08:21:20'
    assert not os.path.exists(os.path.join(d, rl.OPEN_RUN))
    rl.begin(d, 'A1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.8', 'screen')
    assert rl._load_open(d)['row']['iter_no'] == '2'


def test_refusals(ds):
    d, _ = ds
    rl.migrate(d)
    with pytest.raises(rl.LogError):
        rl.begin(d, 'A1', 'A-XX', 'HW', 'PID', 'O1-F05', '0.5', 'screen')
    with pytest.raises(rl.LogError):
        rl.begin(d, 'A:1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen')
    with pytest.raises(rl.LogError):
        rl.begin(d, 'A1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gains='kd=1')
    with pytest.raises(rl.LogError, match='no open run'):
        rl.step(d, 'run')
    rl.begin(d, 'A1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen')
    with pytest.raises(rl.LogError, match='open'):
        rl.begin(d, 'A1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen')
    with pytest.raises(rl.LogError, match='fail-cause'):
        rl.end(d, 'hong_he_do')
    rl.pause(d)
    with pytest.raises(rl.LogError, match='resume'):
        rl.end(d, 'dat')
    rl.cancel(d)
    assert len(rows(os.path.join(d, rl.TUNING))) == 2           # only the June rows
    with pytest.raises(rl.LogError):
        rl.effort(d, '2026-11-20', '08:00', '09:00', campaign='A-PID', khoan='C_chung')
    with pytest.raises(rl.LogError, match='hoc'):
        rl.effort(d, '2026-11-20', '08:00', '09:00', campaign='A-PID', nhan='hoc')
    g = os.path.join(d, 'g.json')
    with open(g, 'wb') as f:
        f.write(b'{"kp_xy": 1.25}\n')
    with pytest.raises(rl.LogError, match='stop-reason'):
        rl.freeze(d, 'A-PID', 'cham_tran', 'G1', g)
    with pytest.raises(rl.LogError, match='no such file'):
        rl.freeze(d, 'A-PID', 'cham_tran_xe', 'G1', g + 'x')
    row = rl.freeze(d, 'A-PID', 'cham_tran_xe', 'G1', g)
    import hashlib
    assert row['gain_sha256'] == hashlib.sha256(b'{"kp_xy": 1.25}\n').hexdigest()
    assert list(rows(os.path.join(d, rl.FREEZE))[0]) == rl.FREEZE_COLS and 'gate_result' not in rl.FREEZE_COLS
    with pytest.raises(rl.LogError, match='already'):
        rl.freeze(d, 'A-PID', 'cham_tran_gio', 'G1', g)


def test_effort_past_midnight(ds):
    d, _ = ds
    rl.migrate(d)
    assert rl.effort(d, '2026-11-20', '23:30', '00:15', khoan='T_H', ghi_chu='x')['gio'] == '0.75'


def test_summary_accounts_split_caps_and_finals(ds):
    d, clock = ds
    rl.migrate(d)
    # A-PID: 2 tune car runs of 600 s active, one final run 100 s, 1 h hoc, 1 h other work
    for sp in ('0.2', '0.5'):
        rl.begin(d, 'A1', 'A-PID', 'HW', 'PID', 'O1-F05', sp, 'screen'); rl.step(d, 'setup'); tick(clock, 600)
        rl.end(d, 'khong_dat')
    rl.begin(d, 'F1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final'); rl.step(d, 'run'); tick(clock, 100)
    rl.end(d, 'dat')
    rl.begin(d, 'F1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final'); rl.step(d, 'run'); tick(clock, 100)
    rl.end(d, 'hong_he_do', fail_cause='mat_khung')
    rl.effort(d, '2026-11-20', '06:00', '07:00', campaign='A-PID', nhan='hoc', ghi_chu='doc')
    rl.effort(d, '2026-11-20', '07:00', '08:00', campaign='A-PID', ghi_chu='lap xe')
    # B-chung: S03 rows per controller (machine 3600 s each, 100 s setup on the PID row), 2 h common
    rl.machine(d, 'B-chung', 'PID', 'S03X', 'r', 2739, 3600, '2026-11-20T09:00:00', '2026-11-20T10:00:00',
               t_setup_s=100)
    rl.machine(d, 'B-chung', 'LADRC', 'S03X', 'r', 2739, 3600, '2026-11-20T09:00:00', '2026-11-20T10:00:00')
    rl.effort(d, '2026-11-20', '10:00', '12:00', campaign='B-chung', ghi_chu='doc S03')
    rl.hwfail(d, '2026-11-20', 'A1', 'A-PID', 'HW', 'tuot banh', 'phan_cung_ngau_nhien', 0.5, iter_no='2')
    rl.freeze(d, 'A-PID', 'cham_tran_xe', 'G0', gain_file(d), at='2026-11-20T12:00:00')
    s = {r['campaign']: r for r in rl.summary(d)}
    a = s['A-PID']
    assert a['T_operator_h'] == f'{(1400 + 7200) / 3600:.3f}' and a['T_operator_no_hoc_h'] == f'{(1400 + 3600) / 3600:.3f}'
    assert a['T_tune_no_hoc_h'] == f'{(1200 + 3600) / 3600:.3f}'          # final runs and hoc out
    assert a['N_HW'] == '4' and a['N_HW_tune'] == '2' and a['N_hong'] == '1' and a['T_sua_h'] == '0.500'
    assert a['stop_reason'] == 'cham_tran_xe' and a['stop_reason_check'] == 'mismatch'   # 2 car runs, not 120
    assert a['T_wall_end_src'] == 'freeze'
    assert a['T_wall_h'] == '6.000'                                     # 06:00 hoc row -> 12:00 freeze
    assert a['final_O1-F05@0.5_n_valid'] == '1' and a['final_O1-F05@0.5_pass'] == '1'
    assert a['final_O1-F05@0.5_invalid'] == '1'
    bp, bl, bc = s['B-PID'], s['B-LADRC'], s['B-chung']
    assert bp['T_machine_h'] == '1.000' and bl['T_machine_h'] == '1.000' and bc['T_machine_h'] == '2.000'
    assert bp['T_operator_h'] == f'{(100 + 3600) / 3600:.3f}' and bl['T_operator_h'] == '1.000'
    assert bc['T_operator_h'] == f'{(100 + 7200) / 3600:.3f}' and bc['B_chung_share'] == 'unsplit total'
    assert bp['N_SIM'] == '2739' and bl['N_SIM'] == '2739' and bc['N_SIM'] == '0'
    assert all(s[c]['T_operator_h'] == '0.000' for c in ('RQ2', 'C_chung', 'T_H'))   # old rows: no account


def test_caps(ds):
    d, clock = ds
    rl.migrate(d)
    for _ in range(rl.CAP_HW):
        rl.begin(d, 'A1', 'A-LADRC', 'HW', 'LADRC', 'O1-F05', '0.5', 'confirm'); rl.step(d, 'run'); tick(clock, 300)
        rl.end(d, 'khong_dat')
    a = {r['campaign']: r for r in rl.summary(d)}['A-LADRC']
    assert a['cap_hit'] == 'xe+gio' and a['N_HW_tune'] == '120' and a['T_tune_no_hoc_h'] == '10.000'


def test_cap_hours_stop_at_the_freeze_and_stop_reason_check(ds):
    d, clock = ds
    rl.migrate(d)
    car_run(d, clock, 'A-PID', 'screen', 'khong_dat', s=3600)
    rl.freeze(d, 'A-PID', 'cham_tran_gio', 'G0', gain_file(d))
    tick(clock, 10)
    car_run(d, clock, 'A-PID', 'explore', 'khong_dat', s=36000)       # after the freeze (§4.8)
    a = {r['campaign']: r for r in rl.summary(d)}['A-PID']
    assert a['T_tune_no_hoc_h'] == '1.000' and a['T_operator_h'] == '11.000'
    assert a['cap_hit'] == '' and a['stop_reason_check'] == 'mismatch'


@pytest.mark.parametrize('outs,gate,want', [
    ({'F05': ['dat'] * 5}, ['F05'], 'dat'),
    ({'F05': ['dat'] * 5, 'F08': ['dat'] * 4}, ['F05', 'F08'], 'chua_ket_luan'),       # runs missing
    ({'F05': ['dat'] * 4 + ['chua_ket_luan']}, ['F05'], 'chua_ket_luan'),
    ({'F05': ['dat'] * 4 + ['khong_dat'], 'F08': ['dat']}, ['F05', 'F08'], 'khong_dat'),
    ({'F05': ['dat'] * 5 + ['hong_he_do'], 'F02': ['khong_dat']}, ['F05'], 'dat'),     # F02 outside the gate
    ({}, [], 'chua_ket_luan'),
])
def test_gate_result_rule_of_4_7(outs, gate, want):
    assert rl.gate_result(outs, gate) == want


def test_finals_gate_replacement_and_final_hil_join(ds):
    d, clock = ds
    rl.migrate(d)
    rl.freeze(d, 'A-PID', 'khoi_xac_nhan_dat', 'G7', gain_file(d))
    lost = car_run(d, clock, 'A-PID', 'final', 'hong_he_do', session='F')['run_uid']
    car_run(d, clock, 'A-PID', 'final', 'dat', session='F', replaces_run_uid=lost)
    for _ in range(4):
        car_run(d, clock, 'A-PID', 'final', 'dat', session='F')
    for _ in range(3):
        car_run(d, clock, 'RQ2', 'final_hil', 'dat', session='H', platform='HIL', gain_set_id='G7')
    car_run(d, clock, 'RQ2', 'final_hil', 'khong_dat', session='H', platform='HIL', gain_set_id='G99')
    s = {r['campaign']: r for r in rl.summary(d, gate=['O1-F05'])}
    a, q = s['A-PID'], s['RQ2']
    assert a['gate_result'] == 'dat' and a['final_O1-F05@0.5_n_valid'] == '5'
    assert a['final_O1-F05@0.5_invalid'] == '1' and a['final_hil_O1-F05@0.5_pass'] == '3'
    assert a['N_HIL'] == '0' and q['N_HIL'] == '4'                        # HIL final = account RQ2
    assert q['final_hil_G99_O1-F05@0.5_fail'] == '1'
    assert {r['campaign']: r for r in rl.summary(d)}['A-PID']['gate_result'] == ''   # no gate given
    car_run(d, clock, 'A-PID', 'final', 'dat', session='F')               # a 6th valid run
    with pytest.raises(rl.LogError, match='more than 5'):
        rl.summary(d)


def test_bad_replacement_and_final_hil_account_are_errors(ds):
    d, clock = ds
    rl.migrate(d)
    ok = car_run(d, clock, 'A-PID', 'final', 'dat', session='F')['run_uid']
    car_run(d, clock, 'A-PID', 'final', 'dat', session='F', replaces_run_uid=ok)
    with pytest.raises(rl.LogError, match='not a hong_he_do'):
        rl.summary(d)
    d2 = os.path.join(d, 'b')
    os.makedirs(d2)
    rl.migrate(d2)
    car_run(d2, clock, 'A-PID', 'final_hil', 'dat', platform='HIL')
    with pytest.raises(rl.LogError, match='account RQ2'):
        rl.summary(d2)


def test_c9_dry_run_recomputes_summary(tmp_path):
    """S05 §8.1 C9: one made-up campaign written through the commands, then S05_summary.csv;
    recomputing from the written logs gives the same file."""
    out = tmp_path / 'dry'
    rows1 = rl.demo(str(out))
    with open(out / rl.SUMMARY, newline='', encoding='utf-8') as f:
        written = list(csv.DictReader(f))
    assert [r['campaign'] for r in written] == list(rl.SUMMARY_ROWS)
    again = rl.summary(str(out), gate=['O1-F05'])
    assert [{k: v for k, v in r.items()} for r in again] == rows1
    for w, r in zip(written, rows1):
        assert all(w[k] == r.get(k, '') for k in w)
    a = {r['campaign']: r for r in written}
    assert a['A-PID']['N_HW'] == '6' and a['A-PID']['stop_reason'] == 'khong_du_luot_xac_nhan'
    assert a['A-PID']['stop_reason_check'] == 'khong_kiem_duoc' and a['A-PID']['gate_result'] == 'chua_ket_luan'
    assert a['A-PID']['final_O1-F05@0.5_n_valid'] == '2' and a['A-PID']['final_hil_O1-F05@0.5_pass'] == '2'
    assert a['RQ2']['N_HIL'] == '2' and a['A-PID']['N_HIL'] == '0'
    assert a['B-PID']['N_SIM'] == '2739' and a['B-PID']['N_HIL'] == '1' and a['B-PID']['N_hong'] == '1'
    assert float(a['C_chung']['T_operator_h']) == 0.5
