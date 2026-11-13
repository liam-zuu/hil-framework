"""tools/s06_relay_check.py (spec S06 W8, C3-3): the three ends of the relay, on files written here. No hardware."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
from protocol import FRAME_SIZE as FRAME_LEN  # noqa: E402  RS-4 P1: 48-byte frames
import protocol  # noqa: E402
import s06_h7_serial as ser  # noqa: E402
import s06_relay_check as rc  # noqa: E402

N = 20_000                        # frames, 1 ms each
EVERY = 20                        # a setpoint every 20 ms, as the 50 Hz publisher
T_START, T_END = 2_000_000, 18_000_000      # #run_start, #run_end on the Pi clock, µs
SETPOINTS = N // EVERY


def frames(tmp_path, manifest='ok', skip_new=()):
    mosi = np.zeros((N, FRAME_LEN), dtype=np.uint8)
    miso = np.zeros((N, FRAME_LEN), dtype=np.uint8)
    for k in range(N):
        mosi[k, 21] = k & 0xFF
        if k % EVERY == 0 and k not in skip_new:
            mosi[k, rc.la.OFS_TRAJ_FLAGS] = protocol.TRAJ_VALID | protocol.TRAJ_NEW
        else:
            mosi[k, rc.la.OFS_TRAJ_FLAGS] = protocol.TRAJ_VALID
    z = dict(t_send_ns=np.arange(N, dtype=np.int64) * 1_000_000, mosi=mosi, miso=miso)
    if manifest == 'ok':
        z['manifest'] = np.array(json.dumps(dict(step_first=0, step_last=N - 1, steps=N)))
    elif manifest == 'late':
        z['manifest'] = np.array(json.dumps(dict(step_first=500, step_last=500 + N - 1, steps=N)))
    p = str(tmp_path / 'f.npz')
    np.savez(p, **z)
    return p


def events(tmp_path, ok_rate=1.0, bad=0, s_lines=True, marks=True, extra=()):
    lines = [f'{T_START}\t#run_start,run=1,first_data_seq=1,timestamp_us={T_START},qseq=0'] if marks else []
    if s_lines:
        for sec in range(2, 19):
            t = sec * 1_000_000
            ok = int((sec * 1000 // EVERY + 1) * ok_rate)
            b = bad if sec > 10 else 0
            lines.append(f'{t}\tS,{sec},{t},1,0,0,0,0,0,0,{ok},{b},0,0,0,25.0')
    if marks:
        lines.append(f'{T_END}\t#run_end,run=1,timestamp_us={T_END},enc_ovf_run=0,ev_qdrop_run=0,qseq=99')
    p = tmp_path / 'ev.txt'
    p.write_text('\n'.join(lines + list(extra)) + '\n')
    return str(p)


def serial(tmp_path, sent0=7, sent=SETPOINTS, dropped=0, head=True, tail=True, lost=True):
    lines = ['--- H753 HIL boot ---']
    if head:
        lines.append(f'Relay: sent={sent0} dropped=0  traj vx=0.00 vy=0.00 wz=0.00')
    lines.append('SPI link up at t=1000 ms')
    lines.append(f'Relay: sent={sent0 + sent // 2} dropped=0  traj vx=0.10 vy=0.00 wz=0.00')
    if lost:
        lines.append('SPI link lost (no frame for 50 ms)')
    if tail:
        lines.append(f'Relay: sent={sent0 + sent} dropped={dropped}  traj vx=0.00 vy=0.00 wz=0.00')
    p = tmp_path / 'h7.log'
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


def go(tmp_path, f=None, ev=None, s=None, *more):
    return rc.main([f or frames(tmp_path), '--esp32-events', ev or events(tmp_path), '--h7-serial', s or serial(tmp_path),
                    '--out', str(tmp_path / 'o'), *more])


def summ(tmp_path):
    return json.load(open(tmp_path / 'o' / 'S06_relay.json', encoding='utf-8'))


def last(capsys):
    return capsys.readouterr().out.splitlines()[-1]


def test_mask_matches_the_protocol():
    assert rc.TRAJ_SEND == protocol.TRAJ_VALID | protocol.TRAJ_NEW


def test_serial_parser_reads_relay_line():
    ev = ser.parse_lines(['Relay: sent=12 dropped=3  traj vx=0.10 vy=0.00 wz=0.00'])
    assert [(e['kind'], e['sent'], e['dropped']) for e in ev] == [('relay', 12, 3)]


def test_three_ends_agree(tmp_path, capsys):
    assert go(tmp_path) == 0
    assert last(capsys) == '#ket_qua,ma=0'
    s = summ(tmp_path)
    assert (s['pi_new'], s['h7_sent'], s['h7_dropped'], s['esp32_bad']) == (SETPOINTS, SETPOINTS, 0, 0)
    assert s['esp32_ok'] == 800 and s['pi_new_in_s_span'] == 801               # both edges of the S span included
    assert (tmp_path / 'o' / 'S06_relay.csv').read_text().splitlines()[0].startswith('idx,pi_new,')


def test_h7_drops_everything_is_caught(tmp_path, capsys):
    # the counterexample of C3-3: NEW at the Pi, nothing relayed, ESP32 with no bad frame
    assert go(tmp_path, ev=events(tmp_path, ok_rate=0.0), s=serial(tmp_path, sent=0)) == 1
    assert last(capsys) == '#ket_qua,ma=1'
    assert any('Pi gửi' in x for x in summ(tmp_path)['fails'])


@pytest.mark.parametrize('sent, dropped', [(SETPOINTS - 1, 0), (SETPOINTS + 1, 0), (SETPOINTS - 2, 2)])
def test_h7_count_must_equal_the_pi(tmp_path, sent, dropped):
    assert go(tmp_path, s=serial(tmp_path, sent=sent, dropped=dropped)) == 1


def test_dropped_alone_fails(tmp_path):
    # W8: (1) = sent + dropped holds here; only the drops are wrong
    go(tmp_path, s=serial(tmp_path, sent=SETPOINTS - 2, dropped=2))
    fails = summ(tmp_path)['fails']
    assert any('H7 bỏ 2' in x for x in fails) and not any('Pi gửi' in x for x in fails)


def test_counters_taken_next_to_the_link(tmp_path):
    # the Relay line just before link up and just after link lost, whatever else the log holds
    p = tmp_path / 'h7.log'
    p.write_text('Relay: sent=1 dropped=0\nRelay: sent=7 dropped=0\nSPI link up at t=1 ms\n'
                 'SPI link lost (no frame for 50 ms)\n'
                 f'Relay: sent={7 + SETPOINTS} dropped=0\nRelay: sent={9 + SETPOINTS} dropped=0\n')
    assert go(tmp_path, s=str(p)) == 0


def test_s_lines_outside_the_run_do_not_count(tmp_path):
    t = T_END + 1_000_000
    late = (f'{t}\tS,50,{t},1,0,0,0,0,0,0,9999,3,0,0,0,25.0',)
    assert go(tmp_path, ev=events(tmp_path, extra=late)) == 0


def test_bad_frames_at_the_esp32_fail(tmp_path):
    assert go(tmp_path, ev=events(tmp_path, bad=1)) == 1


@pytest.mark.parametrize('rate, code', [(0.96, 0), (1.04, 0), (0.94, 1), (1.06, 1)])
def test_ok_within_5_percent(tmp_path, rate, code):
    assert go(tmp_path, ev=events(tmp_path, ok_rate=rate)) == code


def test_counter_going_back_fails(tmp_path):
    p = tmp_path / 'h7.log'
    p.write_text('Relay: sent=500 dropped=0\nSPI link up at t=1 ms\nSPI link lost (no frame for 50 ms)\n'
                 'Relay: sent=3 dropped=0\n')
    assert go(tmp_path, s=str(p)) == 1
    assert any('giảm' in x for x in summ(tmp_path)['fails'])


@pytest.mark.parametrize('kw', [dict(head=False), dict(tail=False), dict(lost=False, tail=False)])
def test_relay_lines_missing_is_missing_evidence(tmp_path, capsys, kw):
    assert go(tmp_path, s=serial(tmp_path, **kw)) == 3
    assert last(capsys) == '#ket_qua,ma=3,ly_do=thieu_bang_chung'


def test_failure_wins_over_missing_evidence(tmp_path):
    assert go(tmp_path, ev=events(tmp_path, bad=1), s=serial(tmp_path, tail=False)) == 1


def test_no_s_lines_is_missing_evidence(tmp_path, capsys):
    assert go(tmp_path, ev=events(tmp_path, s_lines=False)) == 3
    assert last(capsys) == '#ket_qua,ma=3,ly_do=thieu_bang_chung'


def test_no_manifest_is_missing_evidence(tmp_path, capsys):
    assert go(tmp_path, f=frames(tmp_path, manifest=None)) == 3
    assert last(capsys) == '#ket_qua,ma=3,ly_do=thieu_bang_chung'


def test_frame_log_from_a_later_step_is_a_run_condition(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rc.dec, 'manifest_check', lambda *a: [])              # the later start itself is the point
    assert go(tmp_path, f=frames(tmp_path, manifest='late')) == 3
    assert last(capsys) == '#ket_qua,ma=3,ly_do=khong_du_dieu_kien'


def test_no_setpoint_is_a_run_condition(tmp_path, capsys):
    f = frames(tmp_path, skip_new=range(0, N, EVERY))
    assert go(tmp_path, f, events(tmp_path, ok_rate=0.0), serial(tmp_path, sent=0)) == 3
    assert last(capsys) == '#ket_qua,ma=3,ly_do=khong_du_dieu_kien'


def test_no_run_marks(tmp_path, capsys):
    assert go(tmp_path, ev=events(tmp_path, marks=False)) == 3
    assert last(capsys) == '#ket_qua,ma=3,ly_do=khong_du_dieu_kien'


def test_two_runs(tmp_path, capsys):
    other = ('30000000\t#run_start,run=2,first_data_seq=1,timestamp_us=30000000,qseq=100',)
    ev = events(tmp_path, extra=other)
    assert go(tmp_path, ev=ev) == 3
    assert go(tmp_path, ev=ev, *()) == 3
    assert rc.main([frames(tmp_path), '--esp32-events', ev, '--h7-serial', serial(tmp_path), '--out',
                    str(tmp_path / 'o'), '--run', '1']) == 0
    assert rc.main([frames(tmp_path), '--esp32-events', ev, '--h7-serial', serial(tmp_path), '--out',
                    str(tmp_path / 'o'), '--run', '7']) == 2


def test_loss_counter_without_a_run_is_code_1(tmp_path, capsys):
    other = ('30000000\t#run_start,run=2,first_data_seq=1,timestamp_us=30000000,qseq=100',
             '31000000\t#run_end,run=2,timestamp_us=31000000,enc_ovf_run=4,ev_qdrop_run=0,qseq=101')
    assert go(tmp_path, ev=events(tmp_path, extra=other)) == 1
    assert last(capsys) == '#ket_qua,ma=1'


def test_frames_lost_fail(tmp_path):
    z = dict(np.load(frames(tmp_path)))
    z['t_send_ns'], z['mosi'], z['miso'] = (np.delete(z[k], 5001, axis=0) for k in ('t_send_ns', 'mosi', 'miso'))
    p = str(tmp_path / 'lost.npz')
    np.savez(p, **z)
    assert go(tmp_path, f=p) == 1


def test_missing_file_is_input_fault(tmp_path):
    assert go(tmp_path, f=str(tmp_path / 'thieu.npz')) == 2
