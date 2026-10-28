"""agv_logger.py: strict schema, segments (suspected DUT restarts) vs UDP reordering, footers,
files (workflow v3.1 section 9.5; Codex rounds 2-3 #25, #43; B0 review round 1 B3-B10, B20;
round 2 B30-B36). Fixtures are written from the DUT source, not from agv_logger: the header and
the printf conversions below are copied from AGV_MECANUM ctrl-imu 030a836
src/logger/logger.c:92-161, the footers from main.c:305-308. The round 2 counter-examples are
Codex's (codex_tra_loi_vong2.md, B30-B36); expected values are worked out by hand from the
rules in the agv_logger docstring. Synthetic packets, no hardware. Run: python -m pytest tools/tests"""
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
S = 1_000_000           # microseconds per second


def dut_line(seq, ts=None, **over):
    """One data line as the DUT prints it (C conversions mapped to Python)."""
    vals = {}
    for name, conv in zip(NAMES, DUT_FMT):
        if 'f' in conv:
            vals[name] = conv % 0.0
        else:
            vals[name] = '0'
    vals['timestamp_us'] = str(ts if ts is not None else 5 * S + seq * 10_000)
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
    """Offline feeding: no receive time (sample-time rules only)."""
    return [core.feed(x, src) for x in lines]


def feed_rx(core, items, src='10.0.0.2'):
    """(line, receive time in us) pairs, as the logger loop feeds them."""
    return [core.feed(x, src, t_rx_us=rx) for x, rx in items]


def seg(core, i):
    return core.summary()['segments'][i]


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
    assert core.placement == (None, 'no_seq')


@pytest.mark.parametrize('col,val,why', [
    ('pwm_fl', '1.5', 'type'), ('log_seq', 'x', 'type'), ('vx', 'abc', 'type'), ('timestamp_us', '', 'type'),
    ('log_seq', '-1', 'range'), ('tick', str(2**32), 'range'), ('motor_on', '256', 'range'),
    ('imu_init', '-129', 'range'),
    # B36: values the DUT never prints in flag / enum columns
    ('motor_on', '2', 'domain'), ('slip_eval', '3', 'domain'), ('wz_src', '2', 'domain'),
    ('imu_init', '4', 'domain'), ('imu_init', '-2', 'domain')])
def test_strict_rejects_wrong_type_range_or_domain(col, val, why):
    assert al.LoggerCore(strict=True).feed(dut_line(1, **{col: val})) == ('reject', why)


def test_imu_init_not_run_yet_is_accepted():
    assert al.LoggerCore(strict=True).feed(dut_line(1, imu_init=-1))[0] == 'data'      # imu.cpp:29


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


def test_header_with_a_name_that_parses_as_a_number_still_blocks():
    """Codex B36: column 'vx' renamed 'nan' (float('nan') parses) must still lock the schema."""
    core = al.LoggerCore(strict=True)
    names = list(NAMES)
    names[NAMES.index('vx')] = 'nan'
    assert core.feed(','.join(names)) == ('header', None)
    assert core.header == 'mismatch' and core.feed(dut_line(1)) == ('reject', 'schema_header')


def test_header_with_bom_matches():
    core = al.LoggerCore(strict=True)
    core.feed('﻿' + DUT_HEADER)
    assert core.header == 'match'


def test_header_belongs_to_the_segment_it_precedes():
    core = al.LoggerCore(strict=True)
    feed_all(core, [DUT_HEADER] + [dut_line(s, ts=200 * S + s * 10_000) for s in range(100, 120)])
    feed_all(core, [DUT_HEADER] + [dut_line(s, ts=6 * S + s * 10_000) for s in range(0, 5)])
    assert [x['header'] for x in core.summary()['segments']] == ['match', 'match']


def test_source_filter_and_datagram_checks():
    core = al.LoggerCore(strict=True, src_ip='10.0.0.2')
    assert core.feed(dut_line(1), '10.0.0.9') == ('reject', 'source')
    assert core.feed_datagram(b'\xff\xfe' + dut_line(2).encode(), '10.0.0.2') == ('reject', 'decode')
    assert core.feed_datagram((dut_line(3) + '\n' + dut_line(4) + '\n').encode(), '10.0.0.2') == \
        ('reject', 'multiline')
    assert core.feed_datagram((dut_line(5) + '\r\n').encode(), '10.0.0.2')[0] == 'data'
    assert core.feed_datagram(b'\n', '10.0.0.2') == ('reject', 'empty')


# ── segments, duplicates, reordering ──────────────────────────────────────

def test_late_duplicate_is_not_a_restart():
    """Codex B3 counter-example: (1000, 20 s), (1200, 22 s), then (1000, 20 s) again."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(1000, ts=20 * S), dut_line(1200, ts=22 * S), dut_line(1000, ts=20 * S)])
    s = core.summary()
    assert s['segment_breaks'] == 0 and s['segments'][0]['duplicates'] == 1
    assert core.placement == (0, 'duplicate')


def test_same_seq_and_time_with_other_content_is_a_conflict():
    """Codex B32: equal (seq, timestamp) but other PWM is not a duplicate."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(10, pwm_fl=100), dut_line(11), dut_line(10, pwm_fl=250)])
    assert core.placement == (0, 'conflict')
    s = seg(core, 0)
    assert s['conflicts'] == 1 and s['duplicates'] == 0 and s['usable'] is False


def test_reordered_line_stays_in_the_segment():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in (100, 101, 103, 102, 104)])
    b = core.summary()['segments']
    assert len(b) == 1 and b[0]['reordered'] == 1 and b[0]['seq_holes'] == 0 and b[0]['usable']


def test_restart_with_larger_first_timestamp_is_detected():
    """Codex B3: a short old boot (ends at 4 s) and a new boot whose first line is at 12 s."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=3 * S + s * 10_000) for s in range(0, 100)])
    feed_all(core, [dut_line(s, ts=12 * S + s * 10_000) for s in range(0, 20)])
    s = core.summary()
    assert s['segment_breaks'] == 1 and len(s['segments']) == 2


def test_restart_with_smaller_timestamp_is_detected():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=200 * S + s * 10_000) for s in range(5000, 5010)])
    feed_all(core, [dut_line(s, ts=8 * S + s * 10_000) for s in range(0, 5)])
    s = core.summary()
    assert s['segment_breaks'] == 1 and s['segments'][1]['first_seq'] == 0
    assert s['segments'][1]['few_lines'] is True


def test_restart_lines_that_fit_old_seqs_still_split_codex_b30a():
    """Codex B30 (a): boot A seq 50..60 at ~20 s, then reboot B seq 1..10 at ~12 s. Sorted by seq
    the time still grows, but the receive clock moves by ~8 s (> LATE_MAX_US): two segments."""
    core = al.LoggerCore(strict=True)
    a = [(dut_line(s, ts=20 * S + (s - 50) * 10_000), 1 * S + (s - 50) * 10_000) for s in range(50, 61)]
    b = [(dut_line(s, ts=12 * S + (s - 1) * 10_000), 1_200_000 + (s - 1) * 10_000) for s in range(1, 11)]
    feed_rx(core, a + b)
    s = core.summary()
    assert s['segment_breaks'] == 1
    assert [(g['first_seq'], g['last_seq'], g['data_lines']) for g in s['segments']] == [(50, 60, 11), (1, 10, 10)]


def test_late_line_of_old_boot_is_not_taken_by_the_new_one_codex_b30b():
    """Codex B30 (b): A (100, 20 s), B (1, 5 s), late A (101, 20.01 s), B (2, 5.01 s), all received
    within 3 ms. Old rule gave 101 to B and opened a third segment for 2."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(100, ts=20 * S), 1_000_000), (dut_line(1, ts=5 * S), 1_001_000),
                   (dut_line(101, ts=20_010_000), 1_002_000), (dut_line(2, ts=5_010_000), 1_003_000)])
    s = core.summary()
    assert s['segment_breaks'] == 1 and len(s['segments']) == 2
    assert s['segments'][0]['first_seq'] == 100 and s['segments'][0]['last_seq'] == 101
    assert s['segments'][0]['late_lines'] == 1
    assert (s['segments'][1]['first_seq'], s['segments'][1]['last_seq']) == (1, 2)
    assert all(g['usable'] for g in s['segments'])


def test_late_line_of_old_boot_after_restart_goes_to_old_segment():
    core = al.LoggerCore(strict=True)
    a = [(dut_line(s, ts=250 * S + (s - 5000) * 10_000), 1 * S + k * 1000) for k, s in enumerate((5000, 5001, 5003))]
    b = [(dut_line(s, ts=8 * S + s * 10_000), 1_003_000 + s * 1000) for s in range(0, 5)]
    late = [(dut_line(5002, ts=250 * S + 20_000), 1_010_000)]        # arrives after the reboot
    feed_rx(core, a + b + late)
    s = core.summary()
    assert s['segment_breaks'] == 1
    assert s['segments'][0]['late_lines'] == 1 and s['segments'][0]['seq_holes'] == 0
    assert s['segments'][0]['ambiguous_lines'] == 0


def test_line_that_fits_two_segments_is_marked_ambiguous():
    """B30: seq 10 at 20.1 s cannot follow seq 1000 at 20.0 s (time would go back), so it opens
    segment 1; seq 1001 at 20.15 s fits both: placed in the nearest by seq, both unusable."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(1000, ts=20 * S), dut_line(10, ts=20_100_000), dut_line(1001, ts=20_150_000)])
    assert core.placement == (0, 'ambiguous')
    s = core.summary()
    assert [g['ambiguous_lines'] for g in s['segments']] == [1, 1]
    assert [g['usable'] for g in s['segments']] == [False, False]


def test_slow_seq_rate_breaks_a_segment():
    """B30 rate bound: 10 seqs in 3 s is slower than any boot pushes (0.15 s/seq + 0.05 s)."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(1, ts=10 * S), dut_line(11, ts=13 * S)])
    assert core.summary()['segment_breaks'] == 1


def test_counters_first_and_last_by_seq():
    """Codex B32: seq 102 (qdrop 2) arrives before seq 101 (qdrop 0): [0, 2], not [2, 2]."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(102, log_qdrop=2), dut_line(101, log_qdrop=0)])
    assert seg(core, 0)['counters']['log_qdrop'] == [0, 2]


def test_gap_uses_sample_time_not_arrival_order():
    """Codex B4: samples at 0, 20 and then 10 ms leave no 20 ms gap once in order."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(0, ts=1 * S), dut_line(2, ts=1_020_000), dut_line(1, ts=1_010_000)])
    assert seg(core, 0)['max_gap_us'] == 10_000


# ── footers and counters ──────────────────────────────────────────────────

def test_footer_counters_and_seq_holes():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, log_qdrop=0) for s in range(0, 20)])
    out = feed_all(core, [run_end(19, k, qdrop=2, send_err=1) for k in range(3)])
    assert all(o[0] == 'dut_event' for o in out)
    b = seg(core, 0)
    assert b['counters']['log_qdrop'] == [0, 2] and b['counters']['log_send_err'] == [0, 1]
    assert b['run_end_groups'] == [dict(last_data_seq=19, footer_seqs=[20, 21, 22], footers_missing=0,
                                        tail_missing=0)]
    assert b['seq_holes'] == 0


def test_lost_footers_are_not_seq_holes_codex_b33():
    """Codex B33: data 1..10, only footer 11 (L = 10), then a motor-off line 14: 12 and 13 are
    the two lost footers, not lost data."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in range(1, 11)] + [run_end(10, 0)] +
             [dut_line(14, ts=5 * S + 10 * 10_000 + 100_000, motor_on=0)])
    b = seg(core, 0)
    assert b['seq_holes'] == 0
    assert b['run_end_groups'] == [dict(last_data_seq=10, footer_seqs=[11], footers_missing=2, tail_missing=0)]


def test_lost_tail_is_counted_codex_b33():
    """Codex B33: data 1..10 arrive, 11..20 are lost, footer 21 (L = 20) arrives: the span
    reaches 23, 10 data lines missing, tail_missing = 20 - 10."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in range(1, 11)] + [run_end(20, 0)])
    b = seg(core, 0)
    assert b['seq_holes'] == 10
    assert b['run_end_groups'][0]['tail_missing'] == 10 and b['run_end_groups'][0]['footers_missing'] == 2


@pytest.mark.parametrize('line', [
    '#run_end,seq=15,last_data_seq=10,log_qdrop=0,log_send_err=0',      # not one of L+1..L+3
    '#run_end,seq=10,last_data_seq=10,log_qdrop=0,log_send_err=0',
    '#run_end,seq=11,last_data_seq=10,log_qdrop=-1,log_send_err=0',     # negative
    f'#run_end,seq=11,last_data_seq=10,log_qdrop={2**32},log_send_err=0',   # beyond U32
    '#run_end,seq=11,last_data_seq=10,log_qdrop=0,log_qdrop=1,log_send_err=0',   # repeated key
    '#run_end,seq=11,last_data_seq=10,log_qdrop=0'])                     # missing key
def test_malformed_footers_are_refused(line):
    core = al.LoggerCore(strict=True)
    assert core.feed(line)[0] == 'dut_event_bad' and core.footers == {}


def test_footer_matching_two_segments_stays_unassigned():
    """B33: last_data_seq 20 is a data line of both segments: the footer is not guessed."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=200 * S + s * 10_000) for s in range(1, 21)])
    feed_all(core, [dut_line(s, ts=8 * S + s * 10_000) for s in range(1, 21)] + [run_end(20, 0)])
    s = core.summary()
    assert s['run_end_unassigned'] == [dict(seq=21, last_data_seq=20)]
    assert all(g['run_end_groups'] == [] for g in s['segments'])


def test_malformed_footer_is_escaped_and_skipped_by_metrics(tmp_path):
    core = al.LoggerCore(strict=True)
    kind, _ = core.feed('#run_end,seq=21,last_data_seq=x,log_qdrop=0,log_send_err=0')
    assert kind == 'dut_event_bad'
    ev = core.events[-1]
    assert ev.startswith('#logger,bad_dut_event') and 'last_data_seq=' not in ev.split(',')
    p = tmp_path / 'ev.txt'
    p.write_text(f'2026 {ev}\n2026 #run_end,seq=1,last_data_seq=x\n2026 {run_end(9, 0)}\n'
                 f'2026 #run_end,seq=15,last_data_seq=11,log_qdrop=0,log_send_err=0\n', encoding='utf-8')
    assert s02_metrics.read_run_end(str(p)) == [9]


# ── s02_metrics on logs kept as received ──────────────────────────────────

def _write_csv(path, lines, segments=None):
    with open(path, 'w', newline='') as f:
        f.write(DUT_HEADER + '\r\n')
        for ln in lines:
            f.write(ln + '\r\n')
    if segments is not None:
        with open(str(path) + '.segments.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(al.SEGMENT_COLUMNS)
            for i, (ln, (g, st)) in enumerate(zip(lines, segments), start=1):
                v = ln.split(',')
                w.writerow([i, v[al.COL['log_seq']], v[0], g, st])


def test_metrics_drop_duplicates_and_undo_reordering(tmp_path):
    p = tmp_path / 'r.csv'
    lines = [dut_line(s) for s in range(0, 10)]
    _write_csv(p, lines[:5] + [lines[7], lines[5], lines[6], lines[6]] + lines[8:])
    assert list(s02_metrics.load_log(str(p))['log_seq']) == list(range(10))


def test_metrics_drop_duplicates_with_nan_fields(tmp_path):
    """Codex B31: np.unique(axis=0) does not merge rows holding NaN; text comparison does."""
    p = tmp_path / 'r.csv'
    lines = [dut_line(s, ax='nan') for s in range(0, 5)]
    _write_csv(p, lines + [lines[2]])
    assert list(s02_metrics.load_log(str(p))['log_seq']) == list(range(5))


def test_metrics_keep_a_line_that_arrives_after_a_big_seq_jump(tmp_path):
    """Codex B31: seq 100, 200, 101 with time growing with seq is one boot (lost lines)."""
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s) for s in (100, 200, 101)])
    assert list(s02_metrics.load_log(str(p))['log_seq']) == [100, 101, 200]


def test_metrics_refuse_a_log_across_a_restart(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s, ts=200 * S + s * 10_000) for s in range(100, 110)] +
               [dut_line(s, ts=8 * S + s * 10_000) for s in range(0, 5)])
    with pytest.raises(ValueError, match='restart'):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_codex_b30a_without_segment_file(tmp_path):
    """B30 (a): sorted by seq the time grows, but 10 -> 50 takes 7.91 s > 40 * 0.15 + 0.05 s."""
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s, ts=20 * S + (s - 50) * 10_000) for s in range(50, 61)] +
               [dut_line(s, ts=12 * S + (s - 1) * 10_000) for s in range(1, 11)])
    with pytest.raises(ValueError, match='restart'):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_mixed_formats(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(1), dut_line(2, log_seq='')])
    with pytest.raises(ValueError, match='mixe'):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_motor_on_outside_0_1(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(1), dut_line(2, motor_on=2)])
    with pytest.raises(ValueError, match='motor_on'):
        s02_metrics.load_log(str(p))


def test_metrics_use_the_logger_segment_file(tmp_path):
    p = tmp_path / 'r.csv'
    lines = [dut_line(s) for s in range(0, 6)]
    _write_csv(p, lines + [lines[3]], segments=[(0, 'ok')] * 6 + [(0, 'duplicate')])
    assert list(s02_metrics.load_log(str(p))['log_seq']) == list(range(6))


@pytest.mark.parametrize('marks,why', [
    ([(0, 'ok'), (0, 'ok'), (1, 'ok')], 'restart'),
    ([(0, 'ok'), (0, 'ambiguous'), (0, 'ok')], 'ambiguous'),
    ([(0, 'ok'), (0, 'conflict'), (0, 'ok')], 'conflict')])
def test_metrics_refuse_what_the_segment_file_flags(tmp_path, marks, why):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s) for s in range(0, 3)], segments=marks)
    with pytest.raises(ValueError, match=why):
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
    info = json.loads(ready.read_text(encoding='utf-8'))          # published whole (os.replace)
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
    assert 'LOGGER_OPEN' in stdout and not ready.exists()        # ready removed at exit
    assert not any(n.endswith('.tmp') for n in os.listdir(tmp_path))

    with open(out, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == NAMES and len(rows) == 21
    with open(str(out) + '.segments.csv', newline='') as f:
        segrows = list(csv.reader(f))
    assert segrows[0] == al.SEGMENT_COLUMNS
    assert segrows[1:] == [[str(i + 1), str(i), str(5 * S + i * 10_000), '0', 'ok'] for i in range(20)]
    with open(str(out) + '.raw.jsonl', encoding='utf-8') as f:
        raw = [json.loads(line) for line in f]
    assert len(raw) == len(pkts) + 1
    assert base64.b64decode(raw[-1]['b64']) == b'\xff\x00bad\n'      # bytes kept as received
    assert raw[0]['len'] == len(DUT_HEADER) + 1 and raw[0]['ip'] == '127.0.0.1'
    with open(str(out) + '.summary.json', encoding='utf-8') as f:
        summ = json.load(f)
    assert summ['status'] == 'closed' and summ['stop_reason'] == 'duration' and summ['recv_error'] is None
    assert summ['header'] == 'match' and summ['segment_breaks'] == 0
    assert summ['rejects'] == {'n_cols_37': 1, 'decode': 1} and summ['data_lines'] == 20
    assert summ['segments'][0]['run_end_groups'][0]['footers_missing'] == 0

    log = s02_metrics.load_log(str(out))
    assert log['_run_end_last'] == [19]
    assert s02_metrics.log_loss(log) == (0.0, 1)


@pytest.mark.parametrize('existing', ['.summary.json', '.summary.json.tmp', '.raw.jsonl', '.events.txt',
                                      '.segments.csv', ''])
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


def test_failure_while_opening_is_cleaned_up_and_recorded(tmp_path):
    """Codex B34: the ready file cannot be created (no such folder) after the outputs are open:
    every file is closed, the summary says why, exit code 2."""
    out = tmp_path / 'run.csv'
    proc = _run(out, _free_port(), '--ready-file', str(tmp_path / 'no_dir' / 'r.json'), duration='1')
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode == 2, stdout
    summ = json.loads((tmp_path / 'run.csv.summary.json').read_text(encoding='utf-8'))
    assert summ['status'] == 'failed' and summ['stop_reason'] == 'init_failed'
    assert 'r.json' in summ['recv_error']
    assert not any(n.endswith('.tmp') for n in os.listdir(tmp_path))


class _FailingSocket:
    def __init__(self, *a):
        pass

    def bind(self, addr):
        pass

    def settimeout(self, t):
        pass

    def recvfrom(self, n):
        raise OSError('boom')

    def close(self):
        pass


def test_receive_error_is_not_reported_as_closed(tmp_path, monkeypatch, capsys):
    """Codex B35: an exception in the receive loop gives status 'failed', the error and exit 4."""
    monkeypatch.setattr(al.socket, 'socket', _FailingSocket)
    monkeypatch.setattr(al.signal, 'signal', lambda *a: None)
    out = tmp_path / 'run.csv'
    assert al.main(['--out', str(out), '--duration', '5']) == 4
    summ = json.loads((tmp_path / 'run.csv.summary.json').read_text(encoding='utf-8'))
    assert summ['status'] == 'failed' and summ['stop_reason'] == 'error' and 'boom' in summ['recv_error']
