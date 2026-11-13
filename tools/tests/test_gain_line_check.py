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


def write_run(tmp_path, gain_lines, on=(0, 0, 1, 1, 1, 0), seq0=9, copies=3, name='r.csv'):
    """CSV with motor_on per line and log_seq from seq0; events file with each gain line `copies` times."""
    csv_p = tmp_path / name
    rows = ['timestamp_us,log_seq,motor_on'] + [f'{1000 * (i + 1)},{seq0 + i},{m}' for i, m in enumerate(on)]
    csv_p.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    ev = []
    for g in gain_lines:
        ev += [f'2026-11-14T10:00:00.{k:06d} {g}' for k in range(copies)]
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
    assert s['ok'] and s['first_data_seq'] == 11 and s['line_id'] == 7 and s['n_gain_lines'] == 1
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


def test_conflicting_lines_and_bad_lines_are_input_errors(tmp_path):
    log = write_run(tmp_path, [gain_line(first_seq=11), gain_line(first_seq=11, set_id=9)], copies=1)
    with pytest.raises(glc.InputError, match='conflict'):
        glc.check(log, write_gain(tmp_path))
    with pytest.raises(glc.InputError, match='firmware order'):
        glc.parse_gain_line('#gain,id=1,applied_n=1,ctrl=PID,ki_xy=1,kp_xy=1,kp_wz=1,ki_wz=1,pos_kp=1,pos_ki=1,'
                            'k_psi=1,first_seq=2,line_drop=0')
    with pytest.raises(glc.InputError, match='without'):
        glc.parse_gain_line('#gain,id=1,ctrl=PID,first_seq=2')


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
