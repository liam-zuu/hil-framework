"""agv_logger.py: strict schema, DUT restart vs UDP reordering, footers, files
(workflow v3.1 section 9.5; Codex rounds 2-3 #25, #43; B0 review round 1 B3-B10, B20).
Fixtures are written from the DUT source, not from agv_logger: the header and the printf
conversions below are copied from AGV_MECANUM ctrl-imu 030a836 src/logger/logger.c:92-161.
Synthetic packets, no hardware. Run: python -m pytest tools/tests"""
import base64
import csv
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import agv_logger as al  # noqa: E402
import s02_metrics  # noqa: E402

# logger.c:93-107 (the header string, without its trailing newline)
DUT_HEADER = (
    "timestamp_us,fl_count,fr_count,rl_count,rr_count,fl_rpm,fr_rpm,rl_rpm,rr_rpm,vx,vy,wz,"
    "sp_vx,sp_vy,sp_wz,pwm_fl,pwm_fr,pwm_rl,pwm_rr,ax,ay,az,gx,gy,gz,yaw,pitch,roll,"
    "tick,dpwm_fl,dpwm_fr,dpwm_rl,dpwm_rr,sat_fl,sat_fr,sat_rl,sat_rr,"
    "log_seq,log_qdrop,log_send_err,motor_on,imu_init,"
    "wz_meas,wz_cmd,gz_fused,yaw_rad,age_g_ms,age_r_ms,gyro_n,rv_n,reset_n,"
    "g_valid,r_valid,wz_src,src_switch_count,psi_ref,e_psi,head_on,psi_reset_count,"
    "slip_flag,slip_on_count,r,bias_gz,"
    "src_switch_dwz,bias_updates,slip_eval"
)
# logger.c:132-145, the conversion of each column in order
DUT_FMT = ("%lld," "%ld,%ld,%ld,%ld," "%.2f,%.2f,%.2f,%.2f," "%.3f,%.3f,%.3f," "%.3f,%.3f,%.3f,"
           "%d,%d,%d,%d," "%.3f,%.3f,%.3f," "%.3f,%.3f,%.3f," "%.2f,%.2f,%.2f,"
           "%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu," "%lu,%lu,%lu,%u,%d,"
           "%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%lu,%lu,%lu," "%u,%u,%u,%lu,%.4f,%.4f,%u,%lu,"
           "%u,%lu,%.4f,%.5f," "%.4f,%lu,%u").split(',')
NAMES = DUT_HEADER.split(',')


def dut_line(seq, ts=None, **over):
    """One data line as the DUT prints it (C conversions mapped to Python)."""
    vals = {}
    for name, conv in zip(NAMES, DUT_FMT):
        if 'f' in conv:
            vals[name] = conv % 0.0
        else:
            vals[name] = '0'
    vals['timestamp_us'] = str(ts if ts is not None else 5_000_000 + seq * 10_000)
    vals['log_seq'] = str(seq)
    vals['motor_on'] = '1'
    for k, v in over.items():
        vals[k] = str(v)
    return ','.join(vals[n] for n in NAMES)


def run_end(last, k, qdrop=0, send_err=0):
    """main.c:305-308 pushes 3 footers after the last data line L; each takes the next
    log_seq, so their seq are L+1, L+2, L+3 (logger.c:123)."""
    return f'#run_end,seq={last + 1 + k},last_data_seq={last},log_qdrop={qdrop},log_send_err={send_err}'


def feed_all(core, lines, src='10.0.0.2'):
    return [core.feed(x, src) for x in lines]


def test_fixture_matches_logger_schema():
    assert len(NAMES) == len(DUT_FMT) == 66
    assert NAMES == al.COLUMNS
    ints = {n for n, c in zip(NAMES, DUT_FMT) if 'f' not in c}
    assert ints == set(al.INT_COLUMNS)


# ── schema ────────────────────────────────────────────────────────────────

def test_strict_keeps_dut_line_and_rejects_old_formats():
    core = al.LoggerCore(strict=True)
    full = dut_line(1)
    old37 = ','.join(full.split(',')[:al.N_S02])
    out = feed_all(core, [full, old37, ','.join(full.split(',')[:al.N_OLD])])
    assert out[0][0] == 'data'
    assert out[1] == ('reject', 'n_cols_37') and out[2] == ('reject', 'n_cols_28')


def test_loose_pads_old_formats_like_before():
    core = al.LoggerCore(strict=False)
    kind, fields = core.feed(','.join(dut_line(1).split(',')[:al.N_S02]))
    assert kind == 'data' and len(fields) == 66 and fields[-1] == ''


@pytest.mark.parametrize('col,val,why', [
    ('pwm_fl', '1.5', 'type'), ('log_seq', 'x', 'type'), ('vx', 'abc', 'type'), ('timestamp_us', '', 'type'),
    ('log_seq', '-1', 'range'), ('tick', str(2**32), 'range'), ('motor_on', '256', 'range'),
    ('imu_init', '-129', 'range')])
def test_strict_rejects_wrong_type_or_c_range(col, val, why):
    assert al.LoggerCore(strict=True).feed(dut_line(1, **{col: val})) == ('reject', why)


@pytest.mark.parametrize('val', ['nan', '-nan', 'inf', '-inf'])
def test_nonfinite_floats_kept_and_counted(val):
    core = al.LoggerCore(strict=True)
    assert core.feed(dut_line(1, ax=val))[0] == 'data'
    assert core.n_nonfinite == 1


def test_header_mismatch_blocks_until_matching_header():
    core = al.LoggerCore(strict=True)
    names = list(NAMES)
    names[40], names[41] = names[41], names[40]
    assert core.feed(','.join(names)) == ('header', None)
    assert core.header == 'mismatch' and any('first_diff_col=40' in e for e in core.events)
    assert core.feed(dut_line(1)) == ('reject', 'schema_header')
    core.feed(DUT_HEADER)
    assert core.feed(dut_line(2))[0] == 'data'


def test_header_with_renamed_first_column_is_still_a_header():
    core = al.LoggerCore(strict=True)
    assert core.feed(DUT_HEADER.replace('timestamp_us', 'time_us', 1)) == ('header', None)
    assert core.feed(dut_line(1)) == ('reject', 'schema_header')


def test_header_with_bom_matches():
    core = al.LoggerCore(strict=True)
    core.feed('﻿' + DUT_HEADER)
    assert core.header == 'match'


def test_header_belongs_to_the_boot_it_precedes():
    core = al.LoggerCore(strict=True)
    feed_all(core, [DUT_HEADER] + [dut_line(s, ts=200_000_000 + s * 10_000) for s in range(100, 120)])
    feed_all(core, [DUT_HEADER] + [dut_line(s, ts=6_000_000 + s * 10_000) for s in range(0, 5)])
    b = core.summary()['boots']
    assert [x['header'] for x in b] == ['match', 'match']


def test_source_filter_and_datagram_checks():
    core = al.LoggerCore(strict=True, src_ip='10.0.0.2')
    assert core.feed(dut_line(1), '10.0.0.9') == ('reject', 'source')
    assert core.feed_datagram(b'\xff\xfe' + dut_line(2).encode(), '10.0.0.2') == ('reject', 'decode')
    assert core.feed_datagram((dut_line(3) + '\n' + dut_line(4) + '\n').encode(), '10.0.0.2') == \
        ('reject', 'multiline')
    assert core.feed_datagram((dut_line(5) + '\r\n').encode(), '10.0.0.2')[0] == 'data'
    assert core.feed_datagram(b'\n', '10.0.0.2') == ('reject', 'empty')


# ── boots, duplicates, reordering ─────────────────────────────────────────

def test_late_duplicate_is_not_a_restart():
    """Codex B3 counter-example: (1000, 20 s), (1200, 22 s), then (1000, 20 s) again."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(1000, ts=20_000_000), dut_line(1200, ts=22_000_000), dut_line(1000, ts=20_000_000)])
    s = core.summary()
    assert s['dut_restarts_suspected'] == 0 and s['boots'][0]['duplicates'] == 1


def test_reordered_line_stays_in_the_boot():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in (100, 101, 103, 102, 104)])
    b = core.summary()['boots']
    assert len(b) == 1 and b[0]['reordered'] == 1 and b[0]['seq_holes'] == 0


def test_restart_with_larger_first_timestamp_is_detected():
    """Codex B3: a short old boot (ends at 4 s) and a new boot whose first line is at 12 s."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=3_000_000 + s * 10_000) for s in range(0, 100)])
    feed_all(core, [dut_line(s, ts=12_000_000 + s * 10_000) for s in range(0, 20)])
    s = core.summary()
    assert s['dut_restarts_suspected'] == 1 and len(s['boots']) == 2


def test_restart_with_smaller_timestamp_is_detected():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=200_000_000 + s * 10_000) for s in range(5000, 5010)])
    feed_all(core, [dut_line(s, ts=8_000_000 + s * 10_000) for s in range(0, 5)])
    s = core.summary()
    assert s['dut_restarts_suspected'] == 1 and s['boots'][1]['first_seq'] == 0
    assert s['boots'][1]['few_lines'] is True


def test_late_line_of_old_boot_after_restart_goes_to_old_boot():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=200_000_000 + s * 10_000) for s in (5000, 5001, 5003)])
    feed_all(core, [dut_line(s, ts=8_000_000 + s * 10_000) for s in range(0, 5)])
    core.feed(dut_line(5002, ts=200_000_000 + 5002 * 10_000))       # arrives after the reboot
    s = core.summary()
    assert s['dut_restarts_suspected'] == 1
    assert s['boots'][0]['late_lines'] == 1 and s['boots'][0]['seq_holes'] == 0


def test_gap_uses_sample_time_not_arrival_order():
    """Codex B4: samples at 0, 20 and then 10 ms leave no 20 ms gap once in order."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(0, ts=1_000_000), dut_line(2, ts=1_020_000), dut_line(1, ts=1_010_000)])
    assert core.summary()['boots'][0]['max_gap_us'] == 10_000


# ── footers and counters ──────────────────────────────────────────────────

def test_footer_counters_and_seq_holes():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, log_qdrop=0) for s in range(0, 20)])
    out = feed_all(core, [run_end(19, k, qdrop=2, send_err=1) for k in range(3)])
    assert all(o[0] == 'dut_event' for o in out)
    b = core.summary()['boots'][0]
    assert b['counters']['log_qdrop'] == [0, 2] and b['counters']['log_send_err'] == [0, 1]
    assert [f['seq'] for f in b['footers']] == [20, 21, 22]
    assert any('log_qdrop,from=0,to=2' in e for e in core.events)


def test_malformed_footer_is_escaped_and_skipped_by_metrics(tmp_path):
    core = al.LoggerCore(strict=True)
    kind, _ = core.feed('#run_end,seq=21,last_data_seq=x,log_qdrop=0,log_send_err=0')
    assert kind == 'dut_event_bad'
    ev = core.events[-1]
    assert ev.startswith('#logger,bad_dut_event') and 'last_data_seq=' not in ev.split(',')
    p = tmp_path / 'ev.txt'
    p.write_text(f'2026 {ev}\n2026 #run_end,seq=1,last_data_seq=x\n2026 {run_end(9, 0)}\n', encoding='utf-8')
    assert s02_metrics.read_run_end(str(p)) == [9]


# ── s02_metrics on logs kept as received ──────────────────────────────────

def _write_csv(path, lines):
    with open(path, 'w', newline='') as f:
        f.write(DUT_HEADER + '\r\n')
        for ln in lines:
            f.write(ln + '\r\n')


def test_metrics_drop_duplicates_and_undo_reordering(tmp_path):
    p = tmp_path / 'r.csv'
    lines = [dut_line(s) for s in range(0, 10)]
    _write_csv(p, lines[:5] + [lines[7], lines[5], lines[6], lines[6]] + lines[8:])
    log = s02_metrics.load_log(str(p))
    assert list(log['log_seq']) == list(range(10))


def test_metrics_refuse_a_log_across_a_restart(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s, ts=200_000_000 + s * 10_000) for s in range(100, 110)] +
               [dut_line(s, ts=8_000_000 + s * 10_000) for s in range(0, 5)])
    with pytest.raises(ValueError, match='restart'):
        s02_metrics.load_log(str(p))


def test_progress_reads_columns_by_name():
    p = al._progress(dut_line(1, vx=0.25, vy=-0.5, wz=1.0, fl_rpm=12.5, rr_count=999, sp_vx=0.3).split(','))
    assert 'vx=+0.250' in p and 'vy=-0.500' in p and 'wz=+1.000' in p and 'fl_rpm= +12.5' in p


# ── end to end over UDP on localhost ──────────────────────────────────────

def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(out, port, *extra, duration='3'):
    cmd = [sys.executable, os.path.join(HERE, '..', 'agv_logger.py'), '--bind', '127.0.0.1',
           '--port', str(port), '--out', str(out), '--duration', duration, *extra]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')


def test_udp_end_to_end(tmp_path):
    out = tmp_path / 'run.csv'
    ready = tmp_path / 'ready.json'
    port = _free_port()
    proc = _run(out, port, '--strict', '--src-ip', '127.0.0.1', '--ready-file', str(ready))
    t0 = time.monotonic()
    while not ready.exists():
        assert time.monotonic() - t0 < 10, proc.communicate()[0]
        time.sleep(0.05)
    info = json.loads(ready.read_text(encoding='utf-8'))
    assert info['state'] == 'LOGGER_OPEN' and info['pid'] == proc.pid
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pkts = [DUT_HEADER] + [dut_line(s) for s in range(0, 20)] + \
        [','.join(dut_line(20).split(',')[:al.N_S02])] + [run_end(19, k) for k in range(3)]
    for p in pkts:
        tx.sendto((p + '\n').encode(), ('127.0.0.1', port))
        time.sleep(0.002)
    tx.sendto(b'\xff\x00bad\n', ('127.0.0.1', port))
    tx.close()
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode == 0, stdout
    assert 'LOGGER_OPEN' in stdout and not ready.exists()        # ready removed at a clean exit

    with open(out, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == NAMES and len(rows) == 21
    with open(str(out) + '.raw.jsonl', encoding='utf-8') as f:
        raw = [json.loads(l) for l in f]
    assert len(raw) == len(pkts) + 1
    assert base64.b64decode(raw[-1]['b64']) == b'\xff\x00bad\n'      # bytes kept as received
    assert raw[0]['len'] == len(DUT_HEADER) + 1 and raw[0]['ip'] == '127.0.0.1'
    with open(str(out) + '.summary.json', encoding='utf-8') as f:
        summ = json.load(f)
    assert summ['status'] == 'closed' and summ['header'] == 'match'
    assert summ['rejects'] == {'n_cols_37': 1, 'decode': 1} and summ['data_lines'] == 20

    log = s02_metrics.load_log(str(out))
    assert log['_run_end_last'] == [19]
    assert s02_metrics.log_loss(log) == (0.0, 1)


@pytest.mark.parametrize('existing', ['.summary.json', '.raw.jsonl', '.events.txt', ''])
def test_refuses_before_creating_anything_if_any_output_exists(tmp_path, existing):
    out = tmp_path / 'run.csv'
    (tmp_path / ('run.csv' + existing)).write_text('keep', encoding='utf-8')
    before = sorted(os.listdir(tmp_path))
    proc = _run(out, _free_port(), duration='1')
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode == 2, stdout
    assert sorted(os.listdir(tmp_path)) == before
    assert (tmp_path / ('run.csv' + existing)).read_text(encoding='utf-8') == 'keep'


def test_ready_file_equal_to_output_is_refused(tmp_path):
    out = tmp_path / 'run.csv'
    proc = _run(out, _free_port(), '--ready-file', str(out), duration='1')
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode == 2, stdout
    assert os.listdir(tmp_path) == []
