#!/usr/bin/env python3
"""
agv_logger.py — nhận UDP từ ESP32 (hoặc fake_esp32.py), lưu CSV
Không thay đổi gì giữa test laptop và hardware thật.

Usage:
    python agv_logger.py                    # lắng nghe 0.0.0.0:5005
    python agv_logger.py --port 5006        # đổi port
    python agv_logger.py --out my_log.csv   # đặt tên file output
    python agv_logger.py --strict --src-ip 192.168.1.50 --out run.csv --ready-file run.ready
                                            # chế độ đo (luồng làm việc HIL bản 3.1, mục 9.5)

Files next to the CSV. Every path (temporary ones too) is checked before anything is created
and each output is created exclusively, so the logger never overwrites another run:
    <out>                CSV, accepted data lines in arrival order (header = COLUMNS); kept as
                         received: duplicates, reordering and segments are reported, not fixed
    <out>.segments.csv   one row per CSV data row: row, log_seq, timestamp_us, segment, status
                         (ok / duplicate / conflict / ambiguous / no_seq); s02_metrics reads it
    <out>.events.txt     '#' lines from the DUT ('#run_end', ...) and '#logger,...' events
    <out>.raw.jsonl      every datagram as received (base64), rejected ones too, with source
                         ip/port, length, UTC and monotonic receive time
    <out>.summary.json   "open" placeholder at start, replaced atomically (through
                         <out>.summary.json.tmp) at the end: stop reason, receive and close
                         errors, counters per segment
    --ready-file         JSON {state: LOGGER_OPEN, pid, out}, published atomically (through
                         <ready>.tmp) once every output is open; removed at exit
Exit code: 0 closed cleanly, 2 refused or could not open, 3 error while closing, 4 error while
receiving. The run owner checks the exit code AND the summary (B0 review B35): status "closed"
alone does not prove a complete run.

--strict (measurement mode, Codex rounds 2-3 #25, #43; B0 review B3-B10, B36): only 66-column
lines with the DUT's printf types, C type ranges and flag/enum values are kept; a header-like
line that is not COLUMNS name by name blocks data until a matching header arrives. Without
--strict, old 28/37-column lines are padded as before. LOGGER_OPEN is not DUT READY (#35).

Segments, not boots (B0 review B30): without a boot id from the DUT a restart cannot be
confirmed. Inside one boot log_seq and timestamp_us grow together at a bounded rate (the DUT
pushes a line at least every ~0.1 s and every push takes the next seq), and receive time minus
sample time stays near its lowest value (a restart moves it by the old boot's uptime; see the
constants below). A line that fits one segment joins it; a line that fits
none opens a new segment ('segment_break'); a line that fits several joins the nearest by seq
and marks all of them ambiguous. Analysis must refuse segments that are not 'usable'.
"""

import argparse
import base64
import bisect
import csv
import datetime
import json
import math
import os
import signal
import socket
import sys
import time

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEFAULT_PORT = 5005
BUFFER_SIZE  = 65535        # whole UDP datagram, never truncated

COLUMNS = [
    "timestamp_us",
    "fl_count", "fr_count", "rl_count", "rr_count",
    "fl_rpm",   "fr_rpm",   "rl_rpm",   "rr_rpm",
    "vx", "vy", "wz",
    "sp_vx", "sp_vy", "sp_wz",
    "pwm_fl", "pwm_fr", "pwm_rl", "pwm_rr",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "yaw", "pitch", "roll",
    # AGV_MECANUM vel_ctrl metric counters (spec S02): tick, cumulative |dpwm|, saturated ticks
    "tick", "dpwm_fl", "dpwm_fr", "dpwm_rl", "dpwm_rr", "sat_fl", "sat_fr", "sat_rl", "sat_rr",
    # AGV_MECANUM ctrl-imu (controller-IMU spec v2.2 section 8): log loss counters, motor_on
    # (0 = 10 Hz line while the motor is off), imu_init code, IMU fusion / heading loop.
    # Old columns keep their meaning: wz = encoder wz, gz = raw chip gyro, yaw in degrees.
    "log_seq", "log_qdrop", "log_send_err", "motor_on", "imu_init",
    "wz_meas", "wz_cmd", "gz_fused", "yaw_rad", "age_g_ms", "age_r_ms", "gyro_n", "rv_n", "reset_n",
    "g_valid", "r_valid", "wz_src", "src_switch_count", "psi_ref", "e_psi", "head_on", "psi_reset_count",
    "slip_flag", "slip_on_count", "r", "bias_gz",
    # Added 28/10 (Codex review of ctrl-imu #11, #8 (c)): slip_eval 0 = r could not be checked
    # this tick (gyro invalid or wheels below omega_min), so slip_flag 0 does not mean "no slip".
    "src_switch_dwz", "bias_updates", "slip_eval",
]
N_OLD = 28      # firmware before the counters: same columns without the last 9 + 29
N_S02 = 37      # firmware a6644a5 .. before ctrl-imu: without the last 29
KNOWN_NAMES = frozenset(COLUMNS)
SEGMENT_COLUMNS = ["row", "log_seq", "timestamp_us", "segment", "status"]

# C type of each integer column in the DUT's log_data_t (AGV ctrl-imu 030a836 logger.h:17-62,
# printed by logger.c:131-161); every other column is a float printed with %f.
_I64 = (-2**63, 2**63 - 1)
_I32 = (-2**31, 2**31 - 1)
_U32 = (0, 2**32 - 1)
_U8  = (0, 255)
_I8  = (-128, 127)
INT_RANGES = {"timestamp_us": _I64}
INT_RANGES.update({c: _I32 for c in ("fl_count", "fr_count", "rl_count", "rr_count",
                                     "pwm_fl", "pwm_fr", "pwm_rl", "pwm_rr")})
INT_RANGES.update({c: _U32 for c in ("tick", "dpwm_fl", "dpwm_fr", "dpwm_rl", "dpwm_rr",
                                     "sat_fl", "sat_fr", "sat_rl", "sat_rr",
                                     "log_seq", "log_qdrop", "log_send_err", "gyro_n", "rv_n", "reset_n",
                                     "src_switch_count", "psi_reset_count", "slip_on_count",
                                     "bias_updates")})
INT_RANGES.update({c: _U8 for c in ("motor_on", "g_valid", "r_valid", "wz_src", "head_on",
                                    "slip_flag", "slip_eval")})
INT_RANGES["imu_init"] = _I8
INT_COLUMNS = frozenset(INT_RANGES)
# Values the DUT can print in the flag / enum columns (B0 review B36): bools; wz_src
# WZ_SRC_ENC 0 / WZ_SRC_IMU 1 (imu_fusion.h:60-61); imu_init IMU_INIT_* 0..3 (imu.h:29-32)
# or -1 before imu_init() has run (imu.cpp:29)
ENUM_DOMAINS = {c: frozenset((0, 1)) for c in ("motor_on", "g_valid", "r_valid", "wz_src",
                                               "head_on", "slip_flag", "slip_eval")}
ENUM_DOMAINS["imu_init"] = frozenset((-1, 0, 1, 2, 3))
COL = {c: i for i, c in enumerate(COLUMNS)}

RUN_END_KEYS = ("seq", "last_data_seq", "log_qdrop", "log_send_err")   # logger.c:123
CONFIRM_LINES = 10          # a segment with fewer data lines is flagged 'few_lines' [ĐỀ XUẤT]
# Segment rules (B0 review B30).
#  * Rate: the DUT pushes a line every 10 ms with the motor on and every 100 ms with it off
#    (AGV ctrl-imu main.c:515-526), and each push takes the next log_seq even when the queue
#    drops it (logger.c:44-50): inside one boot, lines k seqs apart are at most
#    MAX_US_PER_SEQ * k + RATE_SLACK_US apart in sample time.
#  * Clock: inside one boot, receive time - sample time (logger monotonic clock - DUT
#    timestamp_us) stays near its lowest value; it only grows by the delay of a datagram
#    (network, or queued in the DUT while WiFi stalls) and slowly by clock drift. A line more
#    than LATE_MAX_US above, or CLOCK_TOL_US below, a segment's lowest offset is not of that
#    boot. A DUT restart moves the offset by about the old boot's uptime. A WiFi stall longer
#    than LATE_MAX_US splits a segment: conservative, the analysis then refuses the log.
#    Lines fed without a receive time (offline) use the sample time alone: not more than
#    LATE_MAX_US older than the newest sample of the segment.
MAX_US_PER_SEQ = 150_000    # [ĐỀ XUẤT] 10 Hz line + 50 % for late control ticks
RATE_SLACK_US = 50_000      # [ĐỀ XUẤT]
LATE_MAX_US = 5_000_000     # [ĐỀ XUẤT]
CLOCK_TOL_US = 500_000      # [ĐỀ XUẤT] drift (~100 ppm over an hour = 0.36 s) and jitter


def _int(s):
    try:
        return int(s)
    except ValueError:
        return None


def _float(s):
    try:
        return float(s)
    except ValueError:
        return None


def _u32_text(v):
    """Unsigned decimal text in U32 range (ASCII digits, no sign) -> int, else None."""
    if not (v.isascii() and v.isdigit()):
        return None
    n = int(v)
    return n if n <= _U32[1] else None


def parse_run_end(line):
    """'#run_end,seq=..,last_data_seq=..,log_qdrop=..,log_send_err=..' -> dict, or None if
    malformed: a missing or repeated key, a value that is not U32, or seq not one of the three
    footers L+1..L+3 the firmware pushes after the last data line L (main.c:305-308,
    logger.c:123; B0 review B33)."""
    parts = line.split(',')
    if parts[0] != '#run_end':
        return None
    kv = {}
    for p in parts[1:]:
        k, sep, v = p.partition('=')
        n = _u32_text(v)
        if not sep or n is None or k in kv:
            return None
        kv[k] = n
    if not all(k in kv for k in RUN_END_KEYS):
        return None
    if not 1 <= kv['seq'] - kv['last_data_seq'] <= 3:
        return None
    return kv


def _rate_ok(seq_a, ts_a, seq_b, ts_b):
    """Lines a (lower seq) and b of one boot: time grows with seq, at a bounded rate."""
    return ts_a <= ts_b <= ts_a + MAX_US_PER_SEQ * (seq_b - seq_a) + RATE_SLACK_US


class _Segment:
    """Data lines that can be one DUT boot: log_seq -> timestamp_us, kept sorted by seq."""

    def __init__(self, index):
        self.index = index
        self.ts = {}
        self.key = {}               # seq -> the whole line, to tell a duplicate from a conflict
        self.cnt = {}               # seq -> (log_qdrop, log_send_err) of the data line
        self.seqs = []
        self.max_ts = None
        self.min_off = None         # lowest receive time - sample time, us
        self.n = 0
        self.dup = 0
        self.conflict = 0
        self.reorder = 0
        self.late = 0               # lines placed here after a later segment had started
        self.ambiguous = 0          # lines that fitted this and another segment
        self.peak = {}              # name -> highest loss counter seen so far (events only)
        self.header = 'not_seen'

    def fits(self, seq, ts, t_rx=None):
        if seq in self.ts:
            return False
        if t_rx is not None and self.min_off is not None:
            off = t_rx - ts
            if not self.min_off - CLOCK_TOL_US <= off <= self.min_off + LATE_MAX_US:
                return False
        elif self.max_ts is not None and ts < self.max_ts - LATE_MAX_US:
            return False
        i = bisect.bisect_left(self.seqs, seq)
        if i > 0 and not _rate_ok(self.seqs[i - 1], self.ts[self.seqs[i - 1]], seq, ts):
            return False
        if i < len(self.seqs) and not _rate_ok(seq, ts, self.seqs[i], self.ts[self.seqs[i]]):
            return False
        return True

    def distance(self, seq):
        i = bisect.bisect_left(self.seqs, seq)
        return min(abs(seq - s) for s in self.seqs[max(i - 1, 0):i + 1])

    def add(self, seq, ts, key, cnt, t_rx=None):
        if self.seqs and seq < self.seqs[-1]:
            self.reorder += 1
        bisect.insort(self.seqs, seq)
        self.ts[seq] = ts
        self.key[seq] = key
        self.cnt[seq] = cnt
        self.max_ts = ts if self.max_ts is None else max(self.max_ts, ts)
        if t_rx is not None:
            self.min_off = t_rx - ts if self.min_off is None else min(self.min_off, t_rx - ts)
        self.n += 1

    def summary(self, footers):
        """footers: valid '#run_end' lines assigned to this segment, (seq, L, qdrop, send_err)."""
        tss = [self.ts[s] for s in self.seqs]
        gaps = [b - a for a, b in zip(tss, tss[1:])]
        groups = {}
        for f_seq, L, _, _ in footers:
            groups.setdefault(L, set()).add(f_seq)
        data = set(self.seqs)
        # the three footer seqs L+1..L+3 are known from the firmware even when not received (B33)
        slots = {L + k for L in groups for k in (1, 2, 3)} - data
        holes = 0
        if self.seqs:
            first = self.seqs[0]
            end = max([self.seqs[-1]] + [L + 3 for L in groups])    # span reaches a lost tail
            holes = (end - first + 1) - len(data) - sum(1 for s in slots if first <= s <= end)
        group_rows = []
        for L in sorted(groups):
            i = bisect.bisect_right(self.seqs, L)
            last_data = self.seqs[i - 1] if i else None
            group_rows.append(dict(last_data_seq=L, footer_seqs=sorted(groups[L]),
                                   footers_missing=3 - len(groups[L]),
                                   tail_missing=None if last_data is None else L - last_data))
        # loss counters first/last by seq (B32), data lines and footers together
        by_seq = sorted([(s, c) for s, c in self.cnt.items() if c is not None] +
                        [(f_seq, (q, e)) for f_seq, _, q, e in footers])
        counters = {}
        if by_seq:
            for k, name in enumerate(('log_qdrop', 'log_send_err')):
                counters[name] = [by_seq[0][1][k], by_seq[-1][1][k]]
        return dict(segment=self.index, data_lines=self.n,
                    first_seq=self.seqs[0] if self.seqs else None, last_seq=self.seqs[-1] if self.seqs else None,
                    first_ts=tss[0] if tss else None, last_ts=tss[-1] if tss else None,
                    max_gap_us=max(gaps) if gaps else 0,   # between timestamps sorted by seq
                    seq_holes=holes, duplicates=self.dup, conflicts=self.conflict,
                    reordered=self.reorder, late_lines=self.late, ambiguous_lines=self.ambiguous,
                    run_end_groups=group_rows, counters=counters, header=self.header,
                    few_lines=self.n < CONFIRM_LINES,
                    usable=self.ambiguous == 0 and self.conflict == 0)


class LoggerCore:
    """Datagram checks without sockets or files (tested in tools/tests/test_agv_logger.py).

    feed() returns (kind, payload):
      ('data', fields)          keep in the CSV (fields padded to COLUMNS in loose mode);
                                self.placement = (segment index or None, status) of that row
      ('dut_event', line)       '#' line from the DUT, goes to the events file as is
      ('dut_event_bad', line)   malformed '#run_end': goes to the events file escaped
      ('header', None)          header-like line (not written to the CSV)
      ('reject', reason)        not kept in the CSV (the raw file still has it)
    Logger events collect in self.events as '#logger,...' lines; the caller drains them.
    """

    def __init__(self, strict=False, src_ip=None):
        self.strict = strict
        self.src_ip = src_ip
        self.header = 'not_seen'            # last header-like line: not_seen / match / old_prefix / mismatch
        self.schema_blocked = False
        self.events = []
        self.n_packets = 0
        self.n_data = 0
        self.n_dut_events = 0
        self.rejects = {}
        self.n_nonfinite = 0
        self.n_breaks = 0
        self.segments = []
        self.footers = {}                   # (seq, last_data_seq) -> (log_qdrop, log_send_err)
        self.placement = None
        self._last_ts_noseq = None
        self._header_since_data = None      # header seen after the last data line (belongs to the next segment)

    # ── helpers ───────────────────────────────────────────────────────────
    def _reject(self, reason):
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return ('reject', reason)

    def _event(self, text):
        self.events.append('#logger,' + text)

    def _counter_event(self, seg, name, v, seq):
        """Event when a loss counter goes above anything seen so far in the segment (arrival
        order; the summary's first/last values are by seq)."""
        top = seg.peak.get(name)
        if top is not None and v > top:
            self._event(f'{name},from={top},to={v},log_seq={seq},segment={seg.index}')
        if top is None or v > top:
            seg.peak[name] = v

    # ── main entry ────────────────────────────────────────────────────────
    def feed_datagram(self, data, src_ip=None, t_rx_us=None):
        """Raw UDP payload -> feed(); undecodable or multi-line datagrams are rejected.
        t_rx_us: receive time on a monotonic clock, microseconds (segment rules)."""
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            self.n_packets += 1
            self.placement = None
            return self._reject('decode')
        text = text.rstrip('\r\n')
        if '\n' in text or '\r' in text:
            self.n_packets += 1
            self.placement = None
            return self._reject('multiline')
        return self.feed(text, src_ip, t_rx_us)

    def feed(self, line, src_ip=None, t_rx_us=None):
        self.n_packets += 1
        self.placement = None
        if self.src_ip is not None and src_ip != self.src_ip:
            return self._reject('source')
        line = line.lstrip('﻿')
        if not line.strip():
            return self._reject('empty')

        if line.startswith('#'):
            if line.startswith('#run_end'):
                kv = parse_run_end(line)
                if kv is None:
                    self._event('bad_dut_event,raw=' + line.replace(',', ';'))
                    return ('dut_event_bad', line)
                self._footer(kv)
            self.n_dut_events += 1
            return ('dut_event', line)

        fields = line.split(',')
        names = [f.strip() for f in fields]
        # header: no numeric field, or known column names in it (B36: a name such as 'nan'
        # parses as a float, so one odd name must not turn the header into a data line)
        if all(_float(f) is None for f in names) or sum(n in KNOWN_NAMES for n in names) >= 2:
            return self._header(names)

        if self.strict:
            if self.schema_blocked:
                return self._reject('schema_header')
            if len(fields) != len(COLUMNS):
                return self._reject(f'n_cols_{len(fields)}')
            for name, val in zip(COLUMNS, fields):
                if name in INT_RANGES:
                    v = _int(val)
                    if v is None:
                        return self._reject('type')
                    lo, hi = INT_RANGES[name]
                    if not lo <= v <= hi:
                        return self._reject('range')
                    if name in ENUM_DOMAINS and v not in ENUM_DOMAINS[name]:
                        return self._reject('domain')
                elif _float(val) is None:
                    return self._reject('type')
        else:
            if len(fields) in (N_OLD, N_S02):
                fields = fields + [""] * (len(COLUMNS) - len(fields))
            if len(fields) != len(COLUMNS):
                return self._reject(f'n_cols_{len(fields)}')

        if any((x := _float(v)) is not None and not math.isfinite(x) for v in fields):
            self.n_nonfinite += 1
        self.placement = self._track(fields, t_rx_us)
        self.n_data += 1
        return ('data', fields)

    def _header(self, names):
        if names == COLUMNS:
            self.header = 'match'
            self.schema_blocked = False
        elif names in (COLUMNS[:N_OLD], COLUMNS[:N_S02]):
            self.header = 'old_prefix'
            self.schema_blocked = self.strict
            self._event(f'header_old,n_cols={len(names)}')
        else:
            self.header = 'mismatch'
            self.schema_blocked = self.strict
            bad = next((i for i, (a, b) in enumerate(zip(names, COLUMNS)) if a != b),
                       min(len(names), len(COLUMNS)))
            self._event(f'header_mismatch,n_cols={len(names)},first_diff_col={bad}')
        self._header_since_data = self.header
        return ('header', None)

    def _footer(self, kv):
        """Keep each valid '#run_end' once; it is assigned to a segment in summary(), after the
        data around its last_data_seq has arrived (a footer carries no timestamp)."""
        key = (kv['seq'], kv['last_data_seq'])
        val = (kv['log_qdrop'], kv['log_send_err'])
        old = self.footers.get(key)
        if old is None:
            self.footers[key] = val
        elif old != val:
            self._event(f'run_end_conflict,seq={key[0]},last_data_seq={key[1]}')

    def _footer_segment(self, L):
        """The one segment a footer of last data line L belongs to, or None ('unassigned', B33):
        the segment holding data line L if exactly one does; otherwise the one segment whose
        seq range covers L, or failing that the one whose data ends before L (lost tail).
        A segment with a data line at L+1..L+3, where the footers must be, is refused."""
        have = [g for g in self.segments if L in g.ts]
        if not have:
            have = [g for g in self.segments if g.seqs and g.seqs[0] <= L <= g.seqs[-1]] or \
                   [g for g in self.segments if g.seqs and g.seqs[-1] < L]
        if len(have) != 1 or any(L + k in have[0].ts for k in (1, 2, 3)):
            return None
        return have[0]

    def _new_segment(self):
        g = _Segment(len(self.segments))
        self.segments.append(g)
        return g

    def _track(self, fields, t_rx=None):
        """Place a data line into a segment (B30-B32); returns (segment index or None, status)."""
        ts = _int(fields[COL['timestamp_us']])
        seq = _int(fields[COL['log_seq']])
        if ts is None or seq is None:            # old formats: time order only
            if ts is not None:
                if self._last_ts_noseq is not None and ts < self._last_ts_noseq:
                    self._event(f'ts_backwards,ts_us={ts},after_ts_us={self._last_ts_noseq}')
                self._last_ts_noseq = ts if self._last_ts_noseq is None else max(ts, self._last_ts_noseq)
            return (None, 'no_seq')

        key = tuple(fields)
        for g in self.segments:                  # same seq and time: the same line twice?
            if g.ts.get(seq) == ts:
                if g.key[seq] == key:
                    g.dup += 1
                    self._event(f'dup_seq,seq={seq},segment={g.index}')
                    return (g.index, 'duplicate')
                g.conflict += 1
                self._event(f'seq_conflict,seq={seq},ts_us={ts},segment={g.index}')
                return (g.index, 'conflict')

        cands = [g for g in self.segments if g.fits(seq, ts, t_rx)]
        status = 'ok'
        if not cands:
            if self.segments:
                prev = self.segments[-1]
                self.n_breaks += 1
                self._event(f'segment_break,seq={seq},ts_us={ts},prev_last_seq={prev.seqs[-1]},'
                            f'prev_last_ts_us={prev.ts[prev.seqs[-1]]}')
            g = self._new_segment()
        elif len(cands) == 1:
            g = cands[0]
        else:
            g = min(cands, key=lambda c: (c.distance(seq), -c.index))
            for c in cands:
                c.ambiguous += 1
            status = 'ambiguous'
            self._event(f'ambiguous_line,seq={seq},ts_us={ts},'
                        f'segments={"/".join(str(c.index) for c in cands)},placed={g.index}')
        if g is not self.segments[-1]:
            g.late += 1
            self._event(f'late_line_of_earlier_segment,seq={seq},ts_us={ts},segment={g.index}')
        elif g.seqs and seq < g.seqs[-1]:
            self._event(f'seq_reorder,seq={seq},after_seq={g.seqs[-1]},segment={g.index}')

        q, e = _int(fields[COL['log_qdrop']]), _int(fields[COL['log_send_err']])
        g.add(seq, ts, key, (q, e) if q is not None and e is not None else None, t_rx)
        if self._header_since_data is not None:   # the DUT sends its header once, at boot
            if g.n > 1:
                self._event(f'header_mid_segment,segment={g.index}')
            g.header = self._header_since_data
            self._header_since_data = None
        for name, v in (('log_qdrop', q), ('log_send_err', e)):
            if v is not None:
                self._counter_event(g, name, v, seq)
        return (g.index, status)

    def summary(self):
        per_seg = {g.index: [] for g in self.segments}
        unassigned = []
        for (f_seq, L), (q, e) in sorted(self.footers.items()):
            g = self._footer_segment(L)
            if g is None:
                unassigned.append(dict(seq=f_seq, last_data_seq=L))
            else:
                per_seg[g.index].append((f_seq, L, q, e))
        return dict(
            mode='strict' if self.strict else 'loose',
            src_ip=self.src_ip,
            header=self.header,
            packets=self.n_packets,
            data_lines=self.n_data,
            dut_event_lines=self.n_dut_events,
            rejects=self.rejects,
            lines_with_nonfinite=self.n_nonfinite,
            segment_breaks=self.n_breaks,
            segments=[g.summary(per_seg[g.index]) for g in self.segments],
            run_end_unassigned=unassigned,
            note=('segments split by seq/time consistency; no DUT boot id, so a break is a suspected '
                  'restart, not a confirmed one; refuse segments that are not usable'),
        )


def _progress(fields):
    """Progress line read by column name (old version printed vy/wz/sp_vx/rr_count as vx/vy/wz/fl_rpm)."""
    try:
        ts_ms = int(fields[COL['timestamp_us']]) // 1000
        vals = {c: float(fields[COL[c]]) for c in ('vx', 'vy', 'wz', 'fl_rpm')}
    except (ValueError, IndexError):
        return None
    return (f"[{ts_ms:8d}ms] vx={vals['vx']:+.3f} vy={vals['vy']:+.3f} wz={vals['wz']:+.3f} "
            f"| fl_rpm={vals['fl_rpm']:+6.1f}")


def _raw_record(data, addr, t_mono_ns):
    return json.dumps(dict(t_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           t_mono_ns=t_mono_ns, ip=addr[0], port=addr[1],
                           len=len(data), b64=base64.b64encode(data).decode('ascii')))


def _fsync_dir(path):
    if os.name == 'nt':                       # no directory fsync on Windows
        return
    fd = os.open(os.path.dirname(os.path.abspath(path)) or '.', os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path, obj):
    """Write obj to path.tmp, fsync, then rename over path: a reader sees the old file or the
    whole new one, never half a JSON (B34, B35)."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description="AGV UDP Logger")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bind", default="0.0.0.0", help="Địa chỉ bind (mặc định mọi card mạng)")
    parser.add_argument("--out",  default=None, help="Tên file CSV output")
    parser.add_argument("--strict", action="store_true",
                        help="Chế độ đo: chỉ giữ dòng 66 cột đúng kiểu, header DUT phải khớp")
    parser.add_argument("--src-ip", default=None, help="Chỉ nhận gói từ IP này (ESP32 của bàn HIL)")
    parser.add_argument("--ready-file", default=None,
                        help="Tạo file này (JSON, PID) khi đã bind cổng và mở file (LOGGER_OPEN)")
    parser.add_argument("--duration", type=float, default=None, help="Tự dừng sau N giây")
    args = parser.parse_args(argv)

    # Piped stdout on Windows is cp1252 and cannot print Vietnamese (hil_session reads it)
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Tên file tự động nếu không chỉ định
    if args.out is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agv_log_{ts}.csv"
    else:
        filename = args.out
    paths = {"csv": filename, "segments": filename + ".segments.csv",
             "events": filename + ".events.txt", "raw": filename + ".raw.jsonl",
             "summary": filename + ".summary.json", "summary_tmp": filename + ".summary.json.tmp"}
    if args.ready_file:
        paths["ready"] = args.ready_file
        paths["ready_tmp"] = args.ready_file + ".tmp"
    norm = [os.path.normcase(os.path.abspath(p)) for p in paths.values()]
    if len(set(norm)) != len(norm):
        print(f"[ERROR] Hai file output trùng đường dẫn: {paths}", file=sys.stderr)
        return 2
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing:
        print(f"[ERROR] File đã có, không ghi đè: {existing}", file=sys.stderr)
        return 2

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as exc:
        print(f"[ERROR] Không mở được socket: {exc}", file=sys.stderr)
        return 2
    try:
        sock.bind((args.bind, args.port))
        sock.settimeout(0.5)
    except OSError as exc:
        sock.close()
        print(f"[ERROR] Không bind được {args.bind}:{args.port}: {exc}", file=sys.stderr)
        return 2

    core = LoggerCore(strict=args.strict, src_ip=args.src_ip)
    files = {}
    close_errors = []
    recv_error = None
    stop_reason = None
    summary_made = ready_made = False
    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True

    # One cleanup covers everything from the first file on (B34): an error anywhere below
    # still closes each file on its own, removes the ready file and writes the summary.
    try:
        with open(paths["summary"], "x", encoding="utf-8") as f:     # exclusive placeholder
            f.write('{"status": "open"}\n')
        summary_made = True
        files["csv"] = open(paths["csv"], "x", newline="")
        files["segments"] = open(paths["segments"], "x", newline="")
        files["events"] = open(paths["events"], "x", encoding="utf-8")
        files["raw"] = open(paths["raw"], "x", encoding="utf-8")
        writer = csv.writer(files["csv"])
        writer.writerow(COLUMNS)
        segwriter = csv.writer(files["segments"])
        segwriter.writerow(SEGMENT_COLUMNS)
        for f in files.values():
            f.flush()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _on_signal)

        print(f"[INFO] Lắng nghe UDP {args.bind}:{args.port}  mode={'strict' if args.strict else 'loose'}"
              f"  src_ip={args.src_ip or 'any'}")
        print(f"[INFO] Lưu vào: {filename}")
        if args.ready_file:
            with open(paths["ready_tmp"], "x", encoding="utf-8") as f:
                json.dump(dict(state="LOGGER_OPEN", pid=os.getpid(), out=os.path.abspath(filename),
                               t_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(paths["ready_tmp"], paths["ready"])
            ready_made = True
        print("LOGGER_OPEN", flush=True)
        print("[INFO] Ctrl+C để dừng\n", flush=True)
        stop_reason = "running"

        t_end = None if args.duration is None else time.monotonic() + args.duration
        last_wait = time.monotonic()
        while True:
            if stop["flag"]:
                stop_reason = "signal"
                break
            if t_end is not None and time.monotonic() >= t_end:
                stop_reason = "duration"
                break
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                if time.monotonic() - last_wait > 5.0:
                    print(f"[WAIT] Chưa nhận được data... ({core.n_data} lines saved)")
                    last_wait = time.monotonic()
                continue
            except InterruptedError:
                continue
            t_rx_ns = time.monotonic_ns()
            last_wait = time.monotonic()
            now = datetime.datetime.now().isoformat()
            files["raw"].write(_raw_record(data, addr, t_rx_ns) + "\n")

            kind, payload = core.feed_datagram(data, addr[0], t_rx_ns // 1000)
            if kind == 'dut_event':
                print(f"[RUN] {payload}")
                files["events"].write(f"{now} {payload}\n")
            elif kind == 'header':
                print(f"[INFO] Header từ {addr[0]}: {core.header}")
            elif kind == 'data':
                writer.writerow(payload)
                seg, status = core.placement
                segwriter.writerow([core.n_data, payload[COL['log_seq']], payload[COL['timestamp_us']],
                                    '' if seg is None else seg, status])
                if core.n_data % 100 == 0:
                    p = _progress(payload)
                    if p:
                        print(f"{p} | saved={core.n_data} rejects={sum(core.rejects.values())}")
            for ev in core.events:             # malformed DUT events arrive here, escaped
                print(f"[LOG] {ev}")
                files["events"].write(f"{now} {ev}\n")
            core.events.clear()
            for f in files.values():
                f.flush()
    except BaseException as exc:               # recorded in the summary, exit code 2 or 4
        recv_error = f"{type(exc).__name__}: {exc}"
        stop_reason = "init_failed" if stop_reason is None else "error"
    finally:
        for name, f in files.items():          # each file on its own (B34)
            try:
                f.flush()
                os.fsync(f.fileno())
            except (OSError, ValueError) as exc:
                close_errors.append(f"{name} flush: {exc}")
            try:
                f.close()
            except OSError as exc:
                close_errors.append(f"{name} close: {exc}")
        try:
            sock.close()
        except OSError as exc:
            close_errors.append(f"socket: {exc}")
        if ready_made:
            try:
                os.remove(paths["ready"])
            except OSError as exc:
                close_errors.append(f"ready: {exc}")
        if summary_made:
            summ = core.summary()
            summ["stop_reason"] = stop_reason
            summ["recv_error"] = recv_error
            summ["close_errors"] = list(close_errors)
            summ["status"] = "closed" if recv_error is None and not close_errors else "failed"
            try:
                _write_json_atomic(paths["summary"], summ)
                _fsync_dir(paths["csv"])
            except OSError as exc:              # not in the summary: the exit code carries it
                close_errors.append(f"summary: {exc}")

    print(f"\n[DONE] Dừng ({stop_reason}) — đã lưu {core.n_data} dòng, từ chối "
          f"{sum(core.rejects.values())} {core.rejects or ''}")
    print(f"[DONE] File: {filename}")
    if recv_error is not None:
        print(f"[ERROR] {stop_reason}: {recv_error}", file=sys.stderr)
    if close_errors:
        print(f"[ERROR] Lỗi khi đóng: {close_errors}", file=sys.stderr)
    if stop_reason == "init_failed":
        return 2
    if recv_error is not None:
        return 4
    if close_errors:
        return 3
    if core.n_data > 0:
        print("\n[NEXT] Plot data:")
        print(f"       python plot_log.py {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
