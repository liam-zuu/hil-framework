"""agv_logger.py: strict schema, segments (suspected DUT restarts) vs UDP reordering, footers,
files (workflow v3.1 section 9.5; Codex rounds 2-3 #25, #43; B0 review round 1 B3-B10, B20;
round 2 B30-B36; round 3 B43-B51, N1-N11). Fixtures are written from the DUT source, not from
agv_logger: the header and the printf conversions below are copied from AGV_MECANUM ctrl-imu
030a836 src/logger/logger.c:92-161, the footers from main.c:305-308. The counter-examples are
Codex's (codex_tra_loi_vong2.md, codex_tra_loi_vong3.md, codex_tra_loi_vong3b.md); expected values
are worked out by hand from the rules in the agv_logger docstring. Synthetic packets, no hardware.
Run: python -m pytest tools/tests"""
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
    assert all(x['consistent'] for x in core.summary()['segments'])


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
    assert s['conflicts'] == 1 and s['duplicates'] == 0 and s['consistent'] is False


def test_same_line_long_after_is_not_taken_as_duplicate_codex_n2():
    """Codex N2: (5, 2 s) received at 10 s, (6, 2.01 s), then the same line 5 received at 500 s:
    the receive clock does not fit the first segment, so it may be a new boot printing the same
    line: ambiguous, not a sure duplicate."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(5, ts=2 * S), 10 * S), (dut_line(6, ts=2_010_000), 10_010_000),
                   (dut_line(5, ts=2 * S), 500 * S)])
    assert core.placement == (1, 'ambiguous')
    s = core.summary()['segments']
    assert [g['duplicates'] for g in s] == [0, 0] and [g['consistent'] for g in s] == [False, False]


def test_reordered_line_stays_in_the_segment():
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in (100, 101, 103, 102, 104)])
    b = core.summary()['segments']
    assert len(b) == 1 and b[0]['reordered'] == 1 and b[0]['seq_holes'] == 0 and b[0]['consistent']


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
    the time still grows, but the receive clock moves by ~8 s (> WINDOW_US): two segments."""
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
    assert all(g['consistent'] for g in s['segments'])


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
    segment 1; seq 1001 at 20.15 s fits both: placed in the nearest by seq, neither consistent."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(1000, ts=20 * S), dut_line(10, ts=20_100_000), dut_line(1001, ts=20_150_000)])
    assert core.placement == (0, 'ambiguous')
    s = core.summary()
    assert [g['ambiguous_lines'] for g in s['segments']] == [1, 1]
    assert [g['consistent'] for g in s['segments']] == [False, False]


def test_early_restart_merges_into_one_segment_codex_n1():
    """Codex N1 (B43 the same in other numbers): A seq 11..200 received at 100 s + ts, reset,
    B seq 1..10 received at 103.01 s + ts with the same timestamps per seq. The offset moves by
    3.01 s < WINDOW_US: one segment, no hole. This is the known limit of the policy; the summary
    must not claim a single boot."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(s, ts=1 * S + s * 10_000), 100 * S + 1 * S + s * 10_000) for s in range(11, 201)])
    feed_rx(core, [(dut_line(s, ts=1 * S + s * 10_000), 103_010_000 + 1 * S + s * 10_000) for s in range(1, 11)])
    s = core.summary()['segments']
    assert len(s) == 1 and s[0]['seq_holes'] == 0 and s[0]['consistent'] is True
    assert s[0]['boot_verified'] is False


def test_header_inside_a_segment_makes_it_not_consistent_codex_b43():
    """B43 with the new boot's header received: the header is sent once at boot, so a header
    followed by a line of an existing segment is evidence of a restart inside it."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(s, ts=1 * S + s * 10_000), 101 * S + s * 10_000) for s in range(11, 201)])
    core.feed(DUT_HEADER, '10.0.0.2', t_rx_us=104 * S)
    feed_rx(core, [(dut_line(s, ts=1 * S + s * 10_000), 104_010_000 + s * 10_000) for s in range(1, 11)])
    g = seg(core, 0)
    assert g['header_mid_segment'] == 1 and g['consistent'] is False


def test_header_followed_only_by_duplicates_codex_v7():
    """Codex V7: data 1..10, a DUT header, then the same lines 1..10 again inside the clock
    window. The duplicate branch returned before the header was noted: consistent stayed True."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(s), 6 * S + s * 10_000) for s in range(1, 11)])
    core.feed(DUT_HEADER, '10.0.0.2', t_rx_us=6_200_000)
    feed_rx(core, [(dut_line(s), 6_300_000 + s * 10_000) for s in range(1, 11)])
    g = seg(core, 0)
    assert g['duplicates'] == 10 and g['header_mid_segment'] == 1 and g['consistent'] is False


def test_queue_flush_then_live_line_stays_one_segment_codex_n3():
    """Codex N3 (B44): seq 1..32 (5.01..5.32 s) received together from 6.000 s, 1 ms apart, then
    live seq 101 (6.01 s) at 6.04 s. The offset drops by 0.96 s: inside the window, one boot."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(s, ts=5 * S + s * 10_000), 6 * S + (s - 1) * 1000) for s in range(1, 33)])
    core.feed(dut_line(101, ts=6_010_000), '10.0.0.2', t_rx_us=6_040_000)
    s = core.summary()
    assert s['segment_breaks'] == 0 and s['segments'][0]['consistent']


def test_delay_beyond_the_window_splits_one_boot():
    """The other side of the policy: a line 6 s later than its neighbours (> WINDOW_US) opens a
    segment even if it is the same boot (stated in the docstring)."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(s), 5 * S + s * 10_000 + 1 * S) for s in range(1, 11)])
    core.feed(dut_line(11), '10.0.0.2', t_rx_us=5 * S + 11 * 10_000 + 7 * S)
    assert core.summary()['segment_breaks'] == 1


def test_slow_seq_rate_is_a_timing_violation_not_a_break():
    """N3: 10 seqs in 3 s is slower than the rate policy (0.15 s/seq + 0.05 s), but the firmware
    does not guarantee that rate (a stalled control task): kept in the segment, counted."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(1, ts=10 * S), dut_line(11, ts=13 * S)])
    s = core.summary()
    assert s['segment_breaks'] == 0 and s['segments'][0]['timing_violations'] == 1
    assert any(e.startswith('#logger,rate_exceeded') for e in core.events)


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
                                        tail_seq_distance=0, tail_data_missing=0)]
    assert b['seq_holes'] == 0 and b['consistent']


def test_lost_footers_are_not_seq_holes_codex_b33():
    """Codex B33: data 1..10, only footer 11 (L = 10), then a motor-off line 14: 12 and 13 are
    the two lost footers, not lost data."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in range(1, 11)] + [run_end(10, 0)] +
             [dut_line(14, ts=5 * S + 10 * 10_000 + 100_000, motor_on=0)])
    b = seg(core, 0)
    assert b['seq_holes'] == 0
    assert b['run_end_groups'] == [dict(last_data_seq=10, footer_seqs=[11], footers_missing=2,
                                        tail_seq_distance=0, tail_data_missing=0)]


def test_lost_tail_is_counted_codex_b33():
    """Codex B33: data 1..10 arrive, 11..20 are lost, footer 21 (L = 20) arrives: the span
    reaches 23, 10 data lines missing."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in range(1, 11)] + [run_end(20, 0)])
    b = seg(core, 0)
    assert b['seq_holes'] == 10
    g = b['run_end_groups'][0]
    assert (g['tail_seq_distance'], g['tail_data_missing'], g['footers_missing']) == (10, 10, 2)


def test_tail_data_missing_leaves_out_earlier_footer_slots_codex_n6():
    """Codex N6 (B46): data 1..10, footers of L10, data 14..20 lost, footer of L20: the seq
    distance is 10 but only 7 data lines are missing (11..13 are footers)."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in range(1, 11)] + [run_end(10, k) for k in range(3)] + [run_end(20, 0)])
    b = seg(core, 0)
    assert b['seq_holes'] == 7
    g = b['run_end_groups'][1]
    assert (g['last_data_seq'], g['tail_seq_distance'], g['tail_data_missing']) == (20, 10, 7)


@pytest.mark.parametrize('line', [
    '#run_end,seq=15,last_data_seq=10,log_qdrop=0,log_send_err=0',      # not one of L+1..L+3
    '#run_end,seq=10,last_data_seq=10,log_qdrop=0,log_send_err=0',
    '#run_end,seq=11,last_data_seq=10,log_qdrop=-1,log_send_err=0',     # negative
    f'#run_end,seq=11,last_data_seq=10,log_qdrop={2**32},log_send_err=0',   # beyond U32
    '#run_end,seq=11,last_data_seq=10,log_qdrop=0,log_qdrop=1,log_send_err=0',   # repeated key
    '#run_end,seq=11,last_data_seq=10,log_qdrop=0'])                     # missing key
def test_malformed_footers_are_refused(line):
    core = al.LoggerCore(strict=True)
    assert core.feed(line)[0] == 'dut_event_bad' and core.footer_obs == []


def test_footer_matching_two_segments_stays_unassigned():
    """B33: last_data_seq 20 is a data line of both segments: the footer is not guessed."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=200 * S + s * 10_000) for s in range(1, 21)])
    feed_all(core, [dut_line(s, ts=8 * S + s * 10_000) for s in range(1, 21)] + [run_end(20, 0)])
    s = core.summary()
    assert s['run_end_unassigned'] == [dict(seq=21, last_data_seq=20, candidates=[0, 1])]
    assert all(g['run_end_groups'] == [] and not g['consistent'] for g in s['segments'])


def test_footer_is_not_given_to_the_segment_holding_L_codex_n4():
    """Codex N4 (B45): A has data 1..10, B has 1..9 and lost 10; footer of L10 (from B) arrives.
    Holding L does not exclude B: unassigned, both segments unresolved."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s, ts=200 * S + s * 10_000) for s in range(1, 11)])
    feed_all(core, [dut_line(s, ts=8 * S + s * 10_000) for s in range(1, 10)] + [run_end(10, 0)])
    s = core.summary()
    assert s['run_end_unassigned'] == [dict(seq=11, last_data_seq=10, candidates=[0, 1])]
    assert [g['run_end_unresolved'] for g in s['segments']] == [1, 1]
    assert [g['consistent'] for g in s['segments']] == [False, False]


def test_footer_assigned_by_receive_clock():
    """A: seq 1..10 and 30..40 at ~200 s, received at ~1 s; B: seq 1..20 at ~8 s, received at ~20 s.
    The footer of L20 received at 20.021 s: A has no data at 21..23, but its sample time would
    lie between A's lines 10 and 30 (200.1..200.3 s), far outside A's clock window: B only."""
    core = al.LoggerCore(strict=True)
    feed_rx(core, [(dut_line(s, ts=200 * S + s * 10_000), 1 * S + s * 1000) for s in list(range(1, 11)) + list(range(30, 41))])
    feed_rx(core, [(dut_line(s, ts=8 * S + s * 10_000), 20 * S + s * 1000) for s in range(1, 21)])
    core.feed(run_end(20, 0), '10.0.0.2', t_rx_us=20_021_000)
    s = core.summary()
    assert s['run_end_unassigned'] == []
    assert s['segments'][1]['run_end_groups'][0]['last_data_seq'] == 20
    assert s['segments'][0]['run_end_groups'] == [] and all(g['consistent'] for g in s['segments'])


def test_footer_with_two_counter_values_is_a_conflict_codex_n4():
    """Codex N4: two footers with the same (seq, L) but other counters are not merged quietly."""
    core = al.LoggerCore(strict=True)
    feed_all(core, [dut_line(s) for s in range(1, 11)] + [run_end(10, 0, qdrop=0), run_end(10, 0, qdrop=5)])
    b = seg(core, 0)
    assert b['run_end_conflicts'] == 1 and b['consistent'] is False


def test_malformed_footer_is_escaped_and_skipped_by_metrics(tmp_path):
    core = al.LoggerCore(strict=True)
    kind, _ = core.feed('#run_end,seq=21,last_data_seq=x,log_qdrop=0,log_send_err=0')
    assert kind == 'dut_event_bad'
    ev = core.events[-1]
    assert ev.startswith('#logger,bad_dut_event') and 'last_data_seq=' not in ev.split(',')
    p = tmp_path / 'ev.txt'
    p.write_text(f'2026 {ev}\n2026 #run_end,seq=1,last_data_seq=x\n2026 {run_end(9, 0)}\n'
                 f'2026 #run_end,seq=15,last_data_seq=11,log_qdrop=0,log_send_err=0\n'
                 f'2026 #run_end,seq=13,last_data_seq=12,log_qdrop=0\n', encoding='utf-8')
    assert s02_metrics.read_run_end(str(p)) == [9]          # same parser as the logger (N5)


# ── s02_metrics on logs kept as received ──────────────────────────────────

def _write_csv(path, lines, header=DUT_HEADER):
    with open(path, 'w', newline='') as f:
        f.write(header + '\r\n')
        for ln in lines:
            f.write(ln + '\r\n')


def _logger_run(path, items, status='closed', tamper=None, edit_summary=None):
    """Write a log the way agv_logger.main does: items are lines or (line, receive time);
    tamper(rows, marks) may change the CSV / segment rows before they are written (the
    summary manifest is then taken from the written files, as an edited run would not be)."""
    core = al.LoggerCore(strict=True)
    rows, marks = [], []
    for it in items:
        line, rx = it if isinstance(it, tuple) else (it, None)
        kind, payload = core.feed(line, '10.0.0.2', t_rx_us=rx)
        if kind == 'data':
            rows.append(list(payload))
            g, st = core.placement
            marks.append([str(len(rows)), payload[al.COL['log_seq']], payload[0], '' if g is None else str(g), st])
    if tamper is not None:
        tamper(rows, marks)
    seg_path = str(path) + '.segments.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(al.COLUMNS)
        w.writerows(rows)
    with open(seg_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(al.SEGMENT_COLUMNS)
        w.writerows(marks)
    summ = core.summary()
    summ['status'] = status
    summ['files'] = {'csv': al.file_digest(str(path)), 'segments': al.file_digest(seg_path)}
    if edit_summary is not None:
        edit_summary(summ)
    with open(str(path) + '.summary.json', 'w', encoding='utf-8') as f:
        json.dump(summ, f)
    return summ


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


def test_metrics_refuse_footer_slots_holding_data_codex_n5(tmp_path):
    """Codex N5 (B50): data 1, 2, 4 and a footer of L1: its slots 2..4 hold data, so the footer
    is not of this log (or bad); the old reader dropped 2..4 and reported 0 % loss."""
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s) for s in (1, 2, 4)])
    (tmp_path / 'r.csv.events.txt').write_text(f'2026 {run_end(1, 0)}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='footers must be'):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_two_footers_with_other_counters_codex_v10(tmp_path):
    """Codex V10: without the logger's files, two footers (seq 11, L 10) with qdrop 0 and 5
    were reduced to L = 10 and the log scored 0 % loss; the logger flags the same as a conflict."""
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s) for s in range(1, 11)])
    (tmp_path / 'r.csv.events.txt').write_text(f'x {run_end(10, 0, qdrop=0)}\nx {run_end(10, 0, qdrop=5)}\n',
                                               encoding='utf-8')
    with pytest.raises(ValueError, match='different counter'):
        s02_metrics.load_log(str(p))


@pytest.mark.parametrize('over,why', [({'log_seq': ''}, 'mixe'), ({'motor_on': ''}, 'mixe'),
                                      ({'motor_on': 2}, 'motor_on'), ({'motor_on': 'nan'}, 'motor_on'),
                                      ({'motor_on': 'inf'}, 'motor_on'), ({'log_seq': -1}, 'unsigned'),
                                      ({'log_seq': 2**32}, 'unsigned'), ({'timestamp_us': 'nan'}, 'integer')])
def test_metrics_refuse_bad_dut_rows_codex_n9(tmp_path, over, why):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(1), dut_line(2, **over)])
    with pytest.raises(ValueError, match=why):
        s02_metrics.load_log(str(p))


def test_metrics_check_time_order_of_37_column_logs_codex_n9(tmp_path):
    """Codex N9 (B51): a real 37-column log has no log_seq; time going back is still refused."""
    p = tmp_path / 'r.csv'
    head = ','.join(NAMES[:al.N_S02])
    _write_csv(p, [','.join(dut_line(1, ts=t).split(',')[:al.N_S02]) for t in (3 * S, 2 * S, 4 * S)], header=head)
    with pytest.raises(ValueError, match='goes back'):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_nonfinite_time_in_sim_logs(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(1, log_seq='', motor_on=''), dut_line(2, log_seq='', motor_on='', timestamp_us='nan')])
    with pytest.raises(ValueError, match='not finite'):
        s02_metrics.load_log(str(p))


def test_metrics_keep_sim_rows_in_file_order(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(s, log_seq='', motor_on='') for s in range(3)])
    log = s02_metrics.load_log(str(p))
    assert len(log['timestamp_us']) == 3 and '_seq_all' not in log


def test_metrics_refuse_an_unknown_header(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, [dut_line(1)], header=DUT_HEADER.replace('vx', 'v_x', 1))
    with pytest.raises(ValueError, match='header'):
        s02_metrics.load_log(str(p))


def _moving_log(n=400, at=None):
    """n lines at 10 ms with a constant setpoint and counters growing by one per line;
    at = {seq: {column: value}} overrides."""
    out = []
    for s in range(n):
        v = dict(sp_vx=0.2, tick=s, dpwm_fl=10 * s, dpwm_fr=10 * s, dpwm_rl=10 * s, dpwm_rr=10 * s)
        v.update((at or {}).get(s, {}))
        out.append(dut_line(s, **v))
    return out


def test_metrics_compute_on_a_clean_log(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, _moving_log())
    m = s02_metrics.compute(s02_metrics.load_log(str(p)), 'A-L2')
    assert m['pwm_tv'] > 0 and m['sat_pct'] == 0


@pytest.mark.parametrize('at', [{200: dict(dpwm_fl=5)}, {200: dict(tick=0)}])
def test_metrics_refuse_counters_going_back_codex_n9(tmp_path, at):
    """Codex N9: a counter going back inside the window gave negative PWM metrics."""
    p = tmp_path / 'r.csv'
    _write_csv(p, _moving_log(at=at))
    with pytest.raises(ValueError, match='going back'):
        s02_metrics.compute(s02_metrics.load_log(str(p)), 'A-L2')


def test_metrics_refuse_nonfinite_counter_codex_n9(tmp_path):
    p = tmp_path / 'r.csv'
    _write_csv(p, _moving_log(at={200: dict(dpwm_fl='nan')}))
    with pytest.raises(ValueError, match='not finite'):
        s02_metrics.compute(s02_metrics.load_log(str(p)), 'A-L2')


def test_metrics_refuse_tick_not_advancing_codex_n9(tmp_path):
    """Codex round 4 (N9 test gap): tick constant over the whole window."""
    p = tmp_path / 'r.csv'
    _write_csv(p, _moving_log(at={s: dict(tick=7) for s in range(400)}))
    with pytest.raises(ValueError, match='does not advance'):
        s02_metrics.compute(s02_metrics.load_log(str(p)), 'A-L2')


def test_metrics_refuse_more_saturated_ticks_than_ticks_codex_n9(tmp_path):
    """Codex round 4: the sat > dticks branch on its own (counters growing, never back)."""
    p = tmp_path / 'r.csv'
    _write_csv(p, _moving_log(at={s: dict(sat_rr=2 * s) for s in range(400)}))
    with pytest.raises(ValueError, match='saturated ticks exceed'):
        s02_metrics.compute(s02_metrics.load_log(str(p)), 'A-L2')


def test_metrics_use_the_logger_segment_file(tmp_path):
    p = tmp_path / 'r.csv'
    lines = [dut_line(s) for s in range(0, 6)]
    _logger_run(p, lines + [lines[3]] + [run_end(5, k) for k in range(3)])
    log = s02_metrics.load_log(str(p))
    assert list(log['log_seq']) == list(range(6)) and log['_run_end_last'] == [5]


@pytest.mark.parametrize('lines,why', [
    ([dut_line(s, ts=200 * S + s * 10_000) for s in range(100, 110)] +
     [dut_line(s, ts=8 * S + s * 10_000) for s in range(0, 5)], 'restart'),
    ([dut_line(1000, ts=20 * S), dut_line(10, ts=20_100_000), dut_line(1001, ts=20_150_000)], 'ambiguous'),
    ([dut_line(10, pwm_fl=100), dut_line(11), dut_line(10, pwm_fl=250)], 'conflict')])
def test_metrics_refuse_what_the_logger_flags(tmp_path, lines, why):
    p = tmp_path / 'r.csv'
    _logger_run(p, lines)
    with pytest.raises(ValueError, match=why):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_a_segment_flagged_in_the_summary_only_codex_n8(tmp_path):
    """Codex N8 (B48): rows all 'ok' but the summary says the segment is not consistent (an
    unresolved footer, a header inside it...): refused."""
    p = tmp_path / 'r.csv'
    _logger_run(p, [dut_line(s) for s in range(5)],
                edit_summary=lambda s: s['segments'][0].update(consistent=False))
    with pytest.raises(ValueError, match='not consistent'):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_a_split_segment_codex_n8(tmp_path):
    """Codex N8: split the 'fits two segments' log and keep only segment 1 (rows all 'ok'),
    with the run's summary: the manifest no longer matches."""
    full = tmp_path / 'full.csv'
    _logger_run(full, [dut_line(1000, ts=20 * S), dut_line(10, ts=20_100_000), dut_line(1001, ts=20_150_000)])
    with open(full, newline='') as f:
        rows = list(csv.reader(f))
    with open(str(full) + '.segments.csv', newline='') as f:
        marks = list(csv.reader(f))
    part = tmp_path / 'part.csv'
    keep = [i for i, m in enumerate(marks[1:], start=1) if m[3] == '1']
    with open(part, 'w', newline='') as f:
        csv.writer(f).writerows([rows[0]] + [rows[i] for i in keep])
    with open(str(part) + '.segments.csv', 'w', newline='') as f:
        csv.writer(f).writerows([marks[0]] + [[str(k + 1)] + marks[i][1:] for k, i in enumerate(keep)])
    with open(str(full) + '.summary.json', encoding='utf-8') as f:
        (tmp_path / 'part.csv.summary.json').write_text(f.read(), encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        s02_metrics.load_log(str(part))


def _set(rows_or_marks, i, j, v):
    rows_or_marks[i][j] = v


@pytest.mark.parametrize('tamper,why', [
    (lambda r, m: _set(m, 2, 2, str(int(m[2][2]) + 1)), 'does not match CSV'),         # timestamp
    (lambda r, m: _set(m, 2, 1, '99'), 'does not match CSV'),                            # log_seq
    (lambda r, m: _set(m, 2, 0, '7'), 'does not match CSV'),                             # row number
    (lambda r, m: _set(m, 2, 4, 'fine'), 'not valid'),                                   # unknown status
    (lambda r, m: _set(m, 0, 4, 'duplicate'), 'no earlier equal')])                      # drops a real line
def test_metrics_check_the_segment_file_row_by_row_codex_n7(tmp_path, tamper, why):
    """Codex N7 (B47): the segment file is checked row by row against the CSV, not only counted."""
    p = tmp_path / 'r.csv'
    _logger_run(p, [dut_line(s) for s in range(5)], tamper=tamper)
    with pytest.raises(ValueError, match=why):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_an_edited_csv(tmp_path):
    p = tmp_path / 'r.csv'
    _logger_run(p, [dut_line(s) for s in range(5)])
    text = p.read_text()
    p.write_text(text.replace(',0.000,', ',0.001,', 1))
    with pytest.raises(ValueError, match='does not match files.csv'):
        s02_metrics.load_log(str(p))


@pytest.mark.parametrize('status', ['failed', 'open'])
def test_metrics_refuse_a_run_not_closed_cleanly(tmp_path, status):
    p = tmp_path / 'r.csv'
    _logger_run(p, [dut_line(s) for s in range(5)], status=status)
    with pytest.raises(ValueError, match='status'):
        s02_metrics.load_log(str(p))


@pytest.mark.parametrize('status,consistent', [('closed', False), ('failed', True), ('open', True)])
def test_metrics_refuse_logger_output_without_its_segment_file_codex_v6(tmp_path, status, consistent):
    """Codex V6: the logger's summary is there (segment not consistent, or run not closed) but
    .segments.csv is missing: the plain-CSV branch used to compute metrics anyway."""
    p = tmp_path / 'r.csv'
    _logger_run(p, _moving_log(), status=status,
                edit_summary=lambda s: s['segments'][0].update(consistent=consistent))
    os.remove(str(p) + '.segments.csv')
    with pytest.raises(ValueError, match='segments.csv is missing'):
        s02_metrics.load_log(str(p))


def test_metrics_refuse_a_segment_file_without_summary(tmp_path):
    p = tmp_path / 'r.csv'
    _logger_run(p, [dut_line(s) for s in range(5)])
    os.remove(str(p) + '.summary.json')
    with pytest.raises(ValueError, match='without'):
        s02_metrics.load_log(str(p))


def test_metrics_footers_from_the_logger_assignment_only(tmp_path):
    """N5: with the logger's files, only footers the logger assigned count; the events file is
    not read (a footer of another boot written there changes nothing)."""
    p = tmp_path / 'r.csv'
    _logger_run(p, [dut_line(s) for s in range(0, 20)])
    (tmp_path / 'r.csv.events.txt').write_text(f'2026 {run_end(30, 0)}\n', encoding='utf-8')
    assert s02_metrics.load_log(str(p))['_run_end_last'] == []


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
    info = json.loads(ready.read_text(encoding='utf-8'))          # published whole (hard link)
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
    g = summ['segments'][0]
    assert g['run_end_groups'][0]['footers_missing'] == 0 and g['consistent'] and g['boot_verified'] is False
    assert summ['run_id'] == info['run_id']
    assert set(summ['files']) == {'csv', 'segments', 'events', 'raw'}
    assert summ['files']['csv']['sha256'] == al.file_digest(str(out))['sha256']

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


def test_ready_file_aliasing_the_output_through_a_link_is_refused_codex_n10(tmp_path):
    """Codex N10 (B49): --ready-file through a folder link names the CSV itself."""
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    try:
        os.symlink(real, alias, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('cannot create a folder symlink here (Windows without the privilege)')
    proc = _run(real / 'run.csv', _free_port(), '--ready-file', str(alias / 'run.csv'), duration='1')
    stdout, _ = proc.communicate(timeout=15)
    assert proc.returncode == 2, stdout
    assert os.listdir(real) == []


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


def test_ready_file_made_by_another_process_is_not_replaced_codex_n10(tmp_path, monkeypatch):
    """Codex N10 (B49): another process creates the ready file after the logger's checks. The
    logger must not replace it (hard link fails), must not remove it, and removes its tmp."""
    ready = tmp_path / 'ready.json'
    real_link = os.link

    def racing_link(src, dst):
        with open(dst, 'w', encoding='utf-8') as f:
            f.write('other')
        return real_link(src, dst)

    monkeypatch.setattr(al.os, 'link', racing_link)
    monkeypatch.setattr(al.signal, 'signal', lambda *a: None)
    code = al.main(['--bind', '127.0.0.1', '--port', str(_free_port()), '--out', str(tmp_path / 'run.csv'),
                    '--ready-file', str(ready), '--duration', '0.2'])
    assert code == 2
    assert ready.read_text(encoding='utf-8') == 'other'
    assert not (tmp_path / 'ready.json.tmp').exists()


class _BrokenStdout:
    def write(self, s):
        raise BrokenPipeError('stdout gone')

    def flush(self):
        raise BrokenPipeError('stdout gone')

    def isatty(self):
        return False


def test_broken_stdout_does_not_change_the_exit_code_codex_n11(tmp_path, monkeypatch):
    monkeypatch.setattr(al.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'stdout', _BrokenStdout())
    code = al.main(['--bind', '127.0.0.1', '--port', str(_free_port()), '--out', str(tmp_path / 'run.csv'),
                    '--duration', '0.2'])
    assert code == 0
    summ = json.loads((tmp_path / 'run.csv.summary.json').read_text(encoding='utf-8'))
    assert summ['status'] == 'closed'


def test_closed_stdout_does_not_change_the_exit_code_codex_v11(tmp_path, monkeypatch):
    """Codex V11: a closed stdout raised ValueError at isatty(), before any output was opened."""
    import io
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(al.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'stdout', closed)
    code = al.main(['--bind', '127.0.0.1', '--port', str(_free_port()), '--out', str(tmp_path / 'run.csv'),
                    '--duration', '0.2'])
    assert code == 0
    assert json.loads((tmp_path / 'run.csv.summary.json').read_text(encoding='utf-8'))['status'] == 'closed'


def test_failure_right_after_publishing_ready_removes_it_codex_v12(tmp_path, monkeypatch):
    """Codex V12: the hard link succeeded, then the next step failed: the ready file of a
    logger that is no longer running must not stay behind."""
    ready = tmp_path / 'ready.json'
    real_link = os.link

    def link_then_fail(src, dst):
        real_link(src, dst)
        raise OSError('failed right after link')

    monkeypatch.setattr(al.os, 'link', link_then_fail)
    monkeypatch.setattr(al.signal, 'signal', lambda *a: None)
    code = al.main(['--bind', '127.0.0.1', '--port', str(_free_port()), '--out', str(tmp_path / 'run.csv'),
                    '--ready-file', str(ready), '--duration', '0.2'])
    assert code == 2
    assert not ready.exists() and not (tmp_path / 'ready.json.tmp').exists()


def test_failed_summary_write_leaves_no_tmp_codex_n11(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{"status": "open"}\n', encoding='utf-8')
    with pytest.raises(TypeError):
        al._write_json_atomic(str(p), {'x': object()})
    assert not (tmp_path / 's.json.tmp').exists()
    assert p.read_text(encoding='utf-8') == '{"status": "open"}\n'


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
