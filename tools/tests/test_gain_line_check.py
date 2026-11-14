"""Tests of tools/gain_line_check.py (S05 6.3 RS-9, R134 package): synthetic logs and gain files.
Run: python -m pytest tools/tests/test_gain_line_check.py -q"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import gain_line_check as glc  # noqa: E402

PID = dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)


def gain_line(set_id=7, ctrl='PID', gains=None, first_seq=11, applied_n=1, line_drop=0):
    g = gains or PID
    body = ','.join(f'{n}={"%.9g" % float(__import__("numpy").float32(g[n]))}' for n in glc.NAMES[ctrl])
    return f'#gain,id={set_id},applied_n={applied_n},ctrl={ctrl},{body},first_seq={first_seq},line_drop={line_drop}'


def write_run(tmp_path, gain_lines, on=(0, 0, 1, 1, 1, 0), seq0=9, copies=3, name='r.csv', boot='4753dc47',
              us=True):
    """CSV with motor_on per line and log_seq from seq0; events file with each gain line `copies` times, then
    the run's '#us,ev=start' (boot, first_seq = first motor-on line) three times when `us` (AGV 16702ff)."""
    csv_p = tmp_path / name
    rows = ['timestamp_us,log_seq,motor_on'] + [f'{1000 * (i + 1)},{seq0 + i},{m}' for i, m in enumerate(on)]
    csv_p.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    ev = []
    for g in gain_lines:
        ev += [f'2026-11-14T10:00:00.{k:06d} {g}' for k in range(copies)]
    first = next((seq0 + i for i, m in enumerate(on) if m), None)
    if us and first is not None:
        ev += [f'2026-11-14T10:00:00.1{k:05d} #us,ev=start,boot={boot},on=0,swb=1500,first_seq={first},line_drop=0'
               for k in range(3)]                                     # 0.1 s after the #gain lines
    ev.append('2026-11-14T10:00:01.000000 #fw,sha=16702ff0e688,dirty=0,env=pid')
    (tmp_path / (name + '.events.txt')).write_text('\n'.join(ev) + '\n', encoding='utf-8')
    return str(csv_p)


def write_gain(tmp_path, set_id=7, ctrl='PID', gains=None, name='S05_gain_A-PID.json'):
    p = tmp_path / name
    p.write_text(json.dumps(dict(ctrl=ctrl, id=set_id, gains=gains or PID)), encoding='utf-8')
    return str(p)


def test_match_bit_for_bit(tmp_path):
    log = write_run(tmp_path, [gain_line(first_seq=11)])          # first motor-on line: index 2 -> seq 11
    s = glc.check(log, write_gain(tmp_path))
    assert s['ok'] and s['first_data_seq'] == 11 and s['line_id'] == 7 and s['n_gain_lines'] == 3   # the three copies of the run's line
    assert glc.main(['--log', log, '--gain-file', write_gain(tmp_path)]) == 0


def test_line_of_another_run_or_first_seq_zero_does_not_count(tmp_path):
    log = write_run(tmp_path, [gain_line(first_seq=0), gain_line(first_seq=3)])
    s = glc.check(log, write_gain(tmp_path))
    assert not s['ok'] and 'no #gain line with first_seq = 11' in s['why'][0]
    assert glc.main(['--log', log, '--gain-file', write_gain(tmp_path)]) == 1


@pytest.mark.parametrize('change, why', [
    (dict(set_id=8), 'id=8, gain file id 7'),
    (dict(set_id=0), 'id=0 (built-in set)'),
    (dict(gains=dict(PID, kp_wz=1.0000001)), 'kp_wz'),                # another float32
    (dict(ctrl='ADRC', gains=dict(omega_c_xy=8, omega_o_xy=30, omega_c_wz=8, omega_o_wz=30, pos_kp=6, pos_ki=3,
                                  k_psi=1)), 'ctrl=ADRC'),
])
def test_differences_fail(tmp_path, change, why):
    log = write_run(tmp_path, [gain_line(first_seq=11, **change)])
    s = glc.check(log, write_gain(tmp_path))
    assert not s['ok'] and any(why in w for w in s['why']), s['why']


def test_same_float32_from_another_decimal_matches(tmp_path):
    """The file may hold the double the GA chose; the firmware holds its float32 (gain_send.fmt)."""
    g = dict(PID, kp_xy=1.2345678901234)
    log = write_run(tmp_path, [gain_line(first_seq=11, gains=g)])
    assert glc.check(log, write_gain(tmp_path, gains=g))['ok']


def test_builtin(tmp_path):
    log = write_run(tmp_path, [gain_line(set_id=0, first_seq=11)])
    assert glc.check(log, builtin=True)['ok']
    assert glc.check(log, write_gain(tmp_path), builtin=True)['ok']             # values compared too
    log2 = write_run(tmp_path, [gain_line(set_id=5, first_seq=11)], name='r2.csv')
    assert 'built-in' in glc.check(log2, builtin=True)['why'][0]


@pytest.mark.parametrize('on, msg', [
    ((0, 0, 0), 'no line with motor_on = 1'),
    ((0, 1, 0, 1), '2 motor-on stretches'),
    ((1, 1, 0), 'first data line already has motor_on = 1'),
])
def test_run_shape_errors(tmp_path, on, msg):
    log = write_run(tmp_path, [gain_line(first_seq=10)], on=on)
    with pytest.raises(glc.InputError, match=msg):
        glc.check(log, write_gain(tmp_path))
    assert glc.main(['--log', log, '--gain-file', write_gain(tmp_path)]) == 2


def test_conflicting_lines_fail_and_bad_lines_are_input_errors(tmp_path):
    """R134C-5: copies disagreeing in the gain content are a conflict (a run failed by operation, ma=1)."""
    log = write_run(tmp_path, [gain_line(first_seq=11), gain_line(first_seq=11, set_id=9)], copies=1)
    s = glc.check(log, write_gain(tmp_path))
    assert not s['ok'] and any('conflict' in w for w in s['why'])
    assert glc.main(['--log', log, '--gain-file', write_gain(tmp_path)]) == 1
    with pytest.raises(glc.InputError, match='firmware order'):
        glc.parse_gain_line('#gain,id=1,applied_n=1,ctrl=PID,ki_xy=1,kp_xy=1,kp_wz=1,ki_wz=1,pos_kp=1,pos_ki=1,'
                            'k_psi=1,first_seq=2,line_drop=0')
    with pytest.raises(glc.InputError, match='without'):
        glc.parse_gain_line('#gain,id=1,ctrl=PID,first_seq=2')


def test_line_drop_is_not_a_conflict(tmp_path):
    """R134C-5 (Codex counterexample): two copies differing only in line_drop (read live, logger.c 181-182)."""
    log = write_run(tmp_path, [gain_line(first_seq=11, line_drop=0), gain_line(first_seq=11, line_drop=1)], copies=1)
    s = glc.check(log, write_gain(tmp_path))
    assert s['ok'] and s['line_drop_max'] == 1 and s['n_gain_lines'] == 2


def test_gain_line_needs_the_run_start_with_its_boot(tmp_path):
    """R134C-2: no '#us,ev=start' of the run (boot not shown), or a '#gain' line of an earlier boot with the
    same first_seq before that boot's end lines: no evidence."""
    log = write_run(tmp_path, [gain_line(first_seq=11)], us=False)
    s = glc.check(log, write_gain(tmp_path))
    assert not s['ok'] and 'boot of the run is not shown' in s['why'][0]
    log = write_run(tmp_path, [gain_line(first_seq=11)], name='r2.csv')
    ev = (tmp_path / 'r2.csv.events.txt').read_text(encoding='utf-8').splitlines()
    gl = [x for x in ev if '#gain,' in x]
    rest = [x for x in ev if '#gain,' not in x]
    gl = ['2026-11-14T09:58:59.00000%d %s' % (k, x.split(' ', 1)[1]) for k, x in enumerate(gl)]   # an earlier boot
    old = gl + ['2026-11-14T09:59:00.000000 #us_end,boot=0badf00d,on=0',
                '2026-11-14T09:59:00.000001 #run_end,seq=20,last_data_seq=19']
    (tmp_path / 'r2.csv.events.txt').write_text('\n'.join(old + rest) + '\n', encoding='utf-8')
    s = glc.check(log, write_gain(tmp_path))
    assert not s['ok'] and 'no #gain line with first_seq = 11' in s['why'][0] and s['run_boot'] == '4753dc47'


def _answer(tmp_path, name, boot='4753dc47', mac='aa:bb:cc:dd:ee:ff', tail='#ket_qua,ma=0', t=(1.0, 2.0)):
    p = tmp_path / name
    p.write_text(f'#query_pc,t_start={t[0]},t_end={t[1]}\n'
                 f'#gain_running,sha=16702ff0e688,env=pid,mac={mac},boot={boot},id=7,applied_n=1,ctrl=PID\n'
                 f'{tail}\n', encoding='utf-8')
    return str(p)


def test_query_answers_against_the_run_boot(tmp_path):
    """R134C-1: with --query-before / --query-after, both answers complete, same boot and MAC, = run boot."""
    log = write_run(tmp_path, [gain_line(first_seq=11)])
    g = write_gain(tmp_path)
    qb, qa = _answer(tmp_path, 'b.txt'), _answer(tmp_path, 'a.txt', t=(3.0, 4.0))
    assert glc.check(log, g, queries=(qb, qa))['ok']
    assert glc.main(['--log', log, '--gain-file', g, '--query-before', qb, '--query-after', qa]) == 0
    for after, why in ((_answer(tmp_path, 'o.txt', boot='0badf00d', t=(3, 4)), 'khac boot sau'),
                       (_answer(tmp_path, 'm.txt', mac='11:22:33:44:55:66', t=(3, 4)), 'hai board'),
                       (_answer(tmp_path, 'e.txt', tail='#ket_qua,ma=0\n#ket_qua,ma=3', t=(3, 4)), 'thieu hoi boot sau'),
                       (qb, 'cung mot file'),                                         # R134C2-1
                       (_answer(tmp_path, 'early.txt', t=(0.0, 0.5)), 'khong phai truoc va sau')):
        s = glc.check(log, g, queries=(qb, after))
        assert not s['ok'] and any(why in w for w in s['why']), (why, s['why'])
    old = (_answer(tmp_path, 'ob.txt', boot='0badf00d'), _answer(tmp_path, 'oa.txt', boot='0badf00d', t=(3, 4)))
    s = glc.check(log, g, queries=old)                                # an old pair of the same board
    assert not s['ok'] and any('khac boot cua luot' in w for w in s['why'])


def test_duplicate_data_lines_need_the_logger_marks(tmp_path):
    """R134C-3: a duplicate data line without <csv>.segments.csv: input error; marked duplicate: dropped."""
    log = write_run(tmp_path, [gain_line(first_seq=11)])
    lines = open(log, encoding='utf-8').read().splitlines()
    lines.insert(4, lines[1])                                         # copy of the first off line after an on line
    open(log, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    with pytest.raises(glc.InputError, match='two data lines with log_seq'):
        glc.check(log, write_gain(tmp_path))
    rows = [x.split(',') for x in lines[1:]]
    seg = ['row,log_seq,timestamp_us,segment,status'] + [
        f'{i},{r[1]},{r[0]},0,{"duplicate" if i == 4 else "ok"}' for i, r in enumerate(rows, start=1)]
    open(log + '.segments.csv', 'w', encoding='utf-8').write('\n'.join(seg) + '\n')
    s = glc.check(log, write_gain(tmp_path))
    assert s['ok'] and s['rows_dropped'] == 1


@pytest.mark.parametrize('content, msg', [
    (dict(ctrl='PID', id=0, gains=PID), 'id must be'),
    (dict(ctrl='PID', id=True, gains=PID), 'id must be'),
    (dict(ctrl='XX', id=1, gains=PID), 'ctrl'),
    (dict(ctrl='PID', id=1, gains=dict(PID, extra=1)), 'needs exactly'),
    (dict(ctrl='PID', id=1, gains=dict(PID, kp_xy=1e100)), 'float32'),
])
def test_bad_gain_files(tmp_path, content, msg):
    p = tmp_path / 'g.json'
    p.write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(glc.InputError, match=msg):
        glc.load_gain_file(str(p))


def test_aliases_of_gain_send(tmp_path):
    g = dict(wc_xy=8, wo_xy=30, wc_wz=8, wo_wz=30, pos_kp=6, pos_ki=3, k_psi=1)
    p = tmp_path / 'g.json'
    p.write_text(json.dumps(dict(ctrl='ADRC', id=3, gains=g)), encoding='utf-8')
    assert sorted(glc.load_gain_file(str(p))['gains']) == sorted(glc.NAMES['ADRC'])


def test_gain_line_far_from_the_run_start_does_not_count(tmp_path):
    """R134C2-2 (Codex counterexample): an old '#gain' line of the same first_seq, no boundary line left,
    then the run's '#us,ev=start' 40 s later: not the run's line. Real log: 6.5 ms apart (run14 k1a)."""
    log = write_run(tmp_path, [], name='r.csv')
    ev = (tmp_path / 'r.csv.events.txt').read_text(encoding='utf-8').splitlines()
    old = ['2026-11-14T09:59:20.000000 ' + gain_line(first_seq=11)]
    (tmp_path / 'r.csv.events.txt').write_text('\n'.join(old + ev) + '\n', encoding='utf-8')
    s = glc.check(log, write_gain(tmp_path))
    assert not s['ok'] and 'within 0.5 s' in s['why'][0] and s['n_gain_outside'] == 1


REAL14 = r'D:\CLAUDE_THESIS\05_DATA\results\CLOSED_LOOP\2026-11-13_run14\pi'


@pytest.mark.skipif(not os.path.isdir(REAL14), reason='real HIL run not on this machine')
@pytest.mark.parametrize('name', ['k1a', 'k1b', 'k2'])
def test_real_run_2026_11_13_builtin_set(name):
    """The three runs of CLOSED_LOOP/2026-11-13_run14 (AGV 16702ff, built-in gains): the line is found."""
    s = glc.check(os.path.join(REAL14, f'{name}.csv'), builtin=True)
    assert s['ok'] and s['line_id'] == 0 and s['run_boot'] == '4753dc47' and s['n_gain_lines'] == 3
