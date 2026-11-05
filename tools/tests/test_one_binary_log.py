"""One binary, level 1 (Outline QĐ 6; 03_CODE/docs/2026-10-31_Danh_gia_Mot_binary.md sec. 3, 4):
agv_logger locks on the first source, '#run_end' carries udp_drop / udp_take, a HIL run with
udp_take > 0 is refused. The footer below is copied from AGV_MECANUM feat/one-binary
src/logger/logger.c. Synthetic packets, no hardware. Run: python -m pytest tools/tests"""
import json
import os
import socket
import sys
import time

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import agv_logger as al  # noqa: E402
import s02_metrics  # noqa: E402
from test_agv_logger import DUT_HEADER, _free_port, _run, dut_line, feed_all, run_end  # noqa: E402


def udp_end(last, k, drop=0, take=0):
    """logger.c: "#run_end,seq=%lu,last_data_seq=%lu,log_qdrop=%lu,log_send_err=%lu,udp_drop=%lu,udp_take=%lu"."""
    return f'{run_end(last, k)},udp_drop={drop},udp_take={take}'


# ── lock on the first source ──────────────────────────────────────────────

def test_first_locks_on_the_first_packet_and_counts_the_other_source():
    core = al.LoggerCore(strict=True, src_ip='first')
    assert core.feed(dut_line(1), '10.0.0.2')[0] == 'data'
    assert core.feed(dut_line(2), '10.0.0.9') == ('reject', 'source')
    assert core.feed(dut_line(3), '10.0.0.2')[0] == 'data'
    assert core.feed(dut_line(4), '10.0.0.9') == ('reject', 'source')
    s = core.summary()
    assert s['src_ip'] == '10.0.0.2' and s['src_lock'] == 'first'
    assert s['rejects'] == {'source': 2} and s['data_lines'] == 2
    assert core.events.count('#logger,src_ip_locked,ip=10.0.0.2') == 1


def test_first_stays_open_while_no_source_is_known():
    """Offline feeding gives no source: nothing to lock on, nothing refused."""
    core = al.LoggerCore(strict=True, src_ip='first')
    assert core.feed(dut_line(1))[0] == 'data'
    assert core.src_ip is None
    assert not [e for e in core.events if 'src_ip_locked' in e]
    assert core.feed(dut_line(2), '10.0.0.2')[0] == 'data'
    assert core.src_ip == '10.0.0.2'
    assert core.events.count('#logger,src_ip_locked,ip=10.0.0.2') == 1


def test_undecodable_packet_does_not_take_the_lock():
    core = al.LoggerCore(strict=True, src_ip='first')
    assert core.feed_datagram(b'\xff\xfe', '10.0.0.9') == ('reject', 'decode')
    assert core.feed_datagram((dut_line(1) + '\n').encode(), '10.0.0.2')[0] == 'data'
    assert core.src_ip == '10.0.0.2'


def test_fixed_and_no_filter_behave_as_before():
    fixed = al.LoggerCore(strict=True, src_ip='10.0.0.2')
    assert fixed.feed(dut_line(1), '10.0.0.9') == ('reject', 'source')
    assert fixed.summary()['src_lock'] == 'fixed' and fixed.src_ip == '10.0.0.2'
    assert not [e for e in fixed.events if 'src_ip_locked' in e]
    free = al.LoggerCore(strict=True)
    assert free.feed(dut_line(1), '10.0.0.2')[0] == 'data'
    assert free.feed(dut_line(2), '10.0.0.9')[0] == 'data'
    assert free.summary()['src_lock'] == 'none' and free.summary()['src_ip'] is None


def test_end_to_end_two_sources_on_localhost(tmp_path):
    out = tmp_path / 'run.csv'
    ready = tmp_path / 'ready.json'
    port = _free_port()
    proc = _run(out, port, '--strict', '--src-ip', 'first', '--ready-file', str(ready))
    t0 = time.monotonic()
    while not ready.exists():
        assert time.monotonic() - t0 < 10, proc.communicate()[0]
        time.sleep(0.05)
    a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    a.bind(('127.0.0.1', 0))
    b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    b.bind(('127.0.0.2', 0))
    a.sendto((DUT_HEADER + '\n').encode(), ('127.0.0.1', port))
    time.sleep(0.05)
    for s in range(0, 5):
        a.sendto((dut_line(s) + '\n').encode(), ('127.0.0.1', port))
        b.sendto((dut_line(100 + s) + '\n').encode(), ('127.0.0.1', port))
        time.sleep(0.002)
    a.close()
    b.close()
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode == 0, stdout
    summ = json.loads((tmp_path / 'run.csv.summary.json').read_text(encoding='utf-8'))
    assert summ['src_ip'] == '127.0.0.1' and summ['src_lock'] == 'first'
    assert summ['data_lines'] == 5 and summ['rejects'] == {'source': 5}


# ── '#run_end' counters ───────────────────────────────────────────────────

def test_parse_run_end_takes_the_pair_or_neither():
    kv = al.parse_run_end(udp_end(9, 0, drop=4, take=2))
    assert (kv['udp_drop'], kv['udp_take']) == (4, 2)
    assert 'udp_take' not in al.parse_run_end(run_end(9, 0))
    assert al.parse_run_end(run_end(9, 0) + ',udp_drop=4') is None
    assert al.parse_run_end(run_end(9, 0) + ',udp_take=2') is None
    assert al.parse_run_end(run_end(9, 0) + ',udp_drop=x,udp_take=2') is None


def test_summary_lists_each_run_once_and_flags_a_taken_packet():
    core = al.LoggerCore(strict=True)
    feed_all(core, [DUT_HEADER] + [dut_line(s) for s in range(1, 20)])
    feed_all(core, [udp_end(19, k, drop=3, take=2) for k in range(3)])
    s = core.summary()
    assert s['run_end_udp'] == [dict(last_data_seq=19, udp_drop=3, udp_take=2)]
    assert core.events.count('#logger,udp_take,last_data_seq=19,udp_drop=3,udp_take=2') == 1


def test_no_flag_when_nothing_was_taken_and_groups_stay_consistent():
    core = al.LoggerCore(strict=True)
    feed_all(core, [DUT_HEADER] + [dut_line(s) for s in range(1, 20)])
    feed_all(core, [udp_end(19, k, drop=7) for k in range(3)])
    s = core.summary()
    assert s['run_end_udp'] == [dict(last_data_seq=19, udp_drop=7, udp_take=0)]
    assert not [e for e in core.events if ',udp_take,' in e]
    assert s['segments'][0]['consistent'] is True


def test_old_firmware_footer_gives_an_empty_list():
    core = al.LoggerCore(strict=True)
    feed_all(core, [DUT_HEADER] + [dut_line(s) for s in range(1, 20)] +
             [run_end(19, k) for k in range(3)])
    assert core.summary()['run_end_udp'] == []


# ── s02_metrics ───────────────────────────────────────────────────────────

def test_read_run_end_refuses_two_values_of_the_udp_counters(tmp_path):
    p = tmp_path / 'ev.txt'
    p.write_text(f'{udp_end(9, 0, take=0)}\n{udp_end(9, 0, take=1)}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='two different counter values'):
        s02_metrics.read_run_end(str(p))
    p.write_text(f'{udp_end(9, 0, take=1)}\n{udp_end(9, 1, take=1)}\n', encoding='utf-8')
    assert s02_metrics.read_run_end(str(p)) == [9]


def test_check_no_udp_taken(tmp_path):
    p = tmp_path / 'ev.txt'
    s02_metrics.check_no_udp_taken(str(p))                  # no events file
    p.write_text(f'{run_end(9, 0)}\n#run_end,seq=x\n{udp_end(30, 0, drop=9)}\n', encoding='utf-8')
    s02_metrics.check_no_udp_taken(str(p))                  # old, malformed, nothing taken
    p.write_text(f'{udp_end(9, 0)}\n2026 {udp_end(30, 0, take=1)}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='udp_take=1'):
        s02_metrics.check_no_udp_taken(str(p))


@pytest.mark.parametrize('platform, refused', [('HIL', True), ('hil', True), (' Hil ', True),
                                               ('sim', False), ('REAL', False)])
def test_run_dir_metrics_refuses_only_hil_runs(tmp_path, monkeypatch, platform, refused):
    (tmp_path / 'runs.csv').write_text(
        'run_id,platform,controller,test_id,speed_mps,rep,file,aborted\n'
        f'r1,{platform},PID,A-L1,0.3,1,a.csv,0\n', encoding='utf-8')
    (tmp_path / 'a.csv.events.txt').write_text(udp_end(9, 0, take=1) + '\n', encoding='utf-8')
    monkeypatch.setattr(s02_metrics, 'load_log', lambda path: {})
    monkeypatch.setattr(s02_metrics, 'compute', lambda log, test_id: {'e_vx': 0.0})
    if refused:
        with pytest.raises(ValueError, match='udp_take=1'):
            s02_metrics.run_dir_metrics(str(tmp_path))
    else:
        assert len(s02_metrics.run_dir_metrics(str(tmp_path))) == 1


def test_aborted_hil_run_is_skipped_before_the_check(tmp_path, monkeypatch):
    (tmp_path / 'runs.csv').write_text(
        'run_id,platform,controller,test_id,speed_mps,rep,file,aborted\n'
        'r1,HIL,PID,A-L1,0.3,1,a.csv,1\n', encoding='utf-8')
    (tmp_path / 'a.csv.events.txt').write_text(udp_end(9, 0, take=1) + '\n', encoding='utf-8')
    assert s02_metrics.run_dir_metrics(str(tmp_path)) == []
