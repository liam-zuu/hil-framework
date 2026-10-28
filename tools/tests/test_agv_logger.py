"""agv_logger.py: strict schema, DUT restart vs UDP reorder, raw/events/summary files
(luồng làm việc HIL bản 3.1 mục 9.5, Codex vòng 2–3 #25, #43; Liam duyệt B0 29/10).
Synthetic packets, no hardware. Run: python -m pytest tools/tests"""
import csv
import json
import os
import socket
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import agv_logger as al  # noqa: E402
import s02_metrics  # noqa: E402

HEADER = ','.join(al.COLUMNS)


def line(seq=0, ts=None, **over):
    """A valid 66-column DUT line; ints where the firmware prints ints."""
    vals = {}
    for c in al.COLUMNS:
        vals[c] = '0' if c in al.INT_COLUMNS else '0.000'
    vals['timestamp_us'] = str(ts if ts is not None else 1_000_000 + seq * 10_000)
    vals['log_seq'] = str(seq)
    vals['motor_on'] = '1'
    for k, v in over.items():
        vals[k] = str(v)
    return ','.join(vals[c] for c in al.COLUMNS)


def feed_all(core, lines, src='10.0.0.2'):
    return [core.feed(x, src) for x in lines]


# ── schema ────────────────────────────────────────────────────────────────

def test_strict_keeps_valid_line_and_rejects_old_formats():
    core = al.LoggerCore(strict=True)
    full = line(1)
    old37 = ','.join(full.split(',')[:al.N_S02])
    old28 = ','.join(full.split(',')[:al.N_OLD])
    out = feed_all(core, [full, old37, old28])
    assert out[0][0] == 'data' and len(out[0][1]) == len(al.COLUMNS)
    assert out[1] == ('reject', 'n_cols_37')
    assert out[2] == ('reject', 'n_cols_28')


def test_loose_pads_old_formats_like_before():
    core = al.LoggerCore(strict=False)
    old37 = ','.join(line(1).split(',')[:al.N_S02])
    kind, fields = core.feed(old37)
    assert kind == 'data' and len(fields) == len(al.COLUMNS) and fields[-1] == ''


@pytest.mark.parametrize('col,val', [('pwm_fl', '1.5'), ('log_seq', 'x'), ('vx', 'abc'), ('timestamp_us', '')])
def test_strict_rejects_wrong_types(col, val):
    core = al.LoggerCore(strict=True)
    assert core.feed(line(1, **{col: val})) == ('reject', 'type')


def test_nan_in_float_column_is_kept_and_counted():
    core = al.LoggerCore(strict=True)
    assert core.feed(line(1, ax='nan'))[0] == 'data'
    assert core.n_nan == 1


def test_header_mismatch_blocks_data_in_strict_until_matching_header():
    core = al.LoggerCore(strict=True)
    names = list(al.COLUMNS)
    names[40], names[41] = names[41], names[40]          # same count, two names swapped
    assert core.feed(','.join(names)) == ('header', None)
    assert core.header == 'mismatch'
    assert any('first_diff_col=40' in e for e in core.events)
    assert core.feed(line(1)) == ('reject', 'schema_header')
    core.feed(HEADER)
    assert core.header == 'match'
    assert core.feed(line(2))[0] == 'data'


def test_header_mismatch_only_reported_in_loose():
    core = al.LoggerCore(strict=False)
    core.feed(','.join(al.COLUMNS[:al.N_S02]))
    assert core.header == 'old_prefix'
    assert core.feed(line(1))[0] == 'data'


def test_source_filter():
    core = al.LoggerCore(strict=True, src_ip='10.0.0.2')
    assert core.feed(line(1), '10.0.0.9') == ('reject', 'source')
    assert core.feed(line(2), '10.0.0.2')[0] == 'data'


def test_dut_event_lines_pass_through():
    core = al.LoggerCore(strict=True)
    assert core.feed('#run_end,seq=11,last_data_seq=10,log_qdrop=0,log_send_err=0') == \
        ('dut_event', '#run_end,seq=11,last_data_seq=10,log_qdrop=0,log_send_err=0')


# ── restart / reorder / duplicates / counters ───────────────────────────

def test_seq_drop_without_time_reset_is_reorder_not_restart():
    core = al.LoggerCore(strict=True)
    feed_all(core, [line(s, ts=50_000_000 + s * 10_000) for s in (100, 101, 103, 102, 104)])
    assert core.n_restart == 0 and core.n_reorder == 1
    assert len(core.boots) == 1 and core.boots[0]['last_seq'] == 104


def test_restart_needs_both_time_and_seq_back_near_zero():
    core = al.LoggerCore(strict=True)
    feed_all(core, [line(s, ts=200_000_000 + s * 10_000) for s in range(5000, 5010)])
    feed_all(core, [line(s, ts=8_000_000 + s * 10_000) for s in range(0, 5)])   # reboot
    assert core.n_restart == 1 and len(core.boots) == 2
    assert core.boots[1]['first_seq'] == 0
    assert any(e.startswith('#logger,dut_restart') for e in core.events)


def test_seq_back_to_small_but_time_not_reset_is_reorder():
    core = al.LoggerCore(strict=True)
    feed_all(core, [line(s, ts=1_000_000 + s * 10_000) for s in range(10, 20)])
    core.feed(line(3, ts=1_500_000))            # seq near zero, but time did not go back
    core.feed(line(4, ts=1_150_000))            # seq near zero, time back only 50 ms
    assert core.n_restart == 0 and core.n_reorder == 2


def test_duplicate_seq_counted():
    core = al.LoggerCore(strict=True)
    feed_all(core, [line(1), line(2), line(2), line(3)])
    assert core.n_dup == 1


def test_qdrop_and_send_err_increase_logged():
    core = al.LoggerCore(strict=True)
    feed_all(core, [line(1), line(2, log_qdrop=3), line(3, log_qdrop=3, log_send_err=1)])
    ev = ' '.join(core.events)
    assert 'log_qdrop,from=0,to=3' in ev and 'log_send_err,from=0,to=1' in ev
    assert core.summary()['log_qdrop_last'] == 3


def test_max_gap_per_boot():
    core = al.LoggerCore(strict=True)
    feed_all(core, [line(1, ts=1_000_000), line(2, ts=1_010_000), line(3, ts=1_250_000)])
    assert core.boots[0]['max_gap_us'] == 240_000


def test_progress_reads_columns_by_name():
    f = line(1, vx=0.25, vy=-0.5, wz=1.0, fl_rpm=12.5, rr_count=999, sp_vx=0.3).split(',')
    p = al._progress(f)
    assert 'vx=+0.250' in p and 'vy=-0.500' in p and 'wz=+1.000' in p and 'fl_rpm= +12.5' in p


# ── end to end over UDP on localhost ──────────────────────────────────────

def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run_logger(tmp_path, out, port, extra=()):
    ready = tmp_path / 'ready.txt'
    cmd = [sys.executable, os.path.join(HERE, '..', 'agv_logger.py'), '--bind', '127.0.0.1',
           '--port', str(port), '--out', str(out), '--ready-file', str(ready), '--duration', '3', *extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    return proc, ready


def test_udp_end_to_end(tmp_path):
    out = tmp_path / 'run.csv'
    port = _free_port()
    proc, ready = _run_logger(tmp_path, out, port, ['--strict', '--src-ip', '127.0.0.1'])
    t0 = time.monotonic()
    while not ready.exists():
        assert time.monotonic() - t0 < 10, proc.communicate()[0]
        time.sleep(0.05)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pkts = [HEADER] + [line(s) for s in range(0, 20)] + \
        [','.join(line(20).split(',')[:al.N_S02])] + \
        [f'#run_end,seq={21 + k},last_data_seq=19,log_qdrop=0,log_send_err=0' for k in range(3)]
    for p in pkts:
        tx.sendto((p + '\n').encode(), ('127.0.0.1', port))
        time.sleep(0.002)
    tx.close()
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode == 0, stdout
    assert 'LOGGER_OPEN' in stdout

    with open(out, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == al.COLUMNS and len(rows) == 21
    with open(str(out) + '.raw.txt', encoding='utf-8') as f:
        assert len(f.read().splitlines()) == len(pkts)          # rejected 37-col line kept raw
    with open(str(out) + '.summary.json', encoding='utf-8') as f:
        summ = json.load(f)
    assert summ['header'] == 'match' and summ['rejects'] == {'n_cols_37': 1}
    assert summ['data_lines'] == 20 and summ['dut_event_lines'] == 3

    # s02_metrics still reads the CSV and the '#run_end' lines of the events file
    log = s02_metrics.load_log(str(out))
    assert log['_run_end_last'] == [19]
    assert s02_metrics.log_loss(log) == (0.0, 1)

    # a second logger with the same --out must refuse to overwrite the run
    proc2, _ = _run_logger(tmp_path, out, _free_port())
    stdout2, _ = proc2.communicate(timeout=15)
    assert proc2.returncode == 2, stdout2
    with open(out, newline='') as f:
        assert len(list(csv.reader(f))) == 21
