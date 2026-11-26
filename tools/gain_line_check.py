#!/usr/bin/env python3
"""
gain_line_check.py: the '#gain' line of one run against the gain file of the set that should run
(S05 v3.13 section 6.3 "Kiem gain cua tung luot [L90]", RS-9 of the R130 review; L90 stage 2).
R134 package, Liam 14/11/2026 CODE_R134 local_eb9f3b5c: "duyet goi sua code fix/r134-answers"; after
Codex Sol round 1 (R134C-1, -2, -3, -5; Liam 14/11 CODE_R134 "Sua + vong 2 (Recommended)").

Rule (S05 6.3): every campaign run and final-check run has a '#gain' line in its log (id, ctrl, the 7
gains) equal bit for bit to the gain file of the running set. The gain file holds the integer id sent to
the firmware ('gain_file' of S05_freeze.csv, or the trial file of the tuning round); gain_set_id (with
its campaign prefix) is the key of S05_freeze.csv and is never sent. A missing line, a difference, or
id = 0 when the run should run a set other than the built-in one: a run failed by operation (handled as
a run failed by the measurement system, O1 3.5).

Firmware facts (AGV 16702ff): main.c 342-356 run_udp_mark, at the motor-on edge, pushes '#fw' three times,
'#gain' three times (logger.c 176-185: '#gain' + gain_set_format, gain_set.c 241-252: ',id=%lu,
applied_n=%lu,ctrl=%s' then ',<name>=%.9g' for the 7 gains, then ',first_seq=%lu,line_drop=%lu'), then one
'#us,ev=start,boot=%08lx,...,first_seq=%lu,line_drop=%lu' event that the logger sends three times
(us_state.c 44; boot = gain_rx_boot_id(), the number every '#gain_ack' / gain_send --query answer carries).
first_seq = logger_seq_next(), the log_seq of the first data line of the run (pushed in the same tick);
these lines take no log_seq (logger.c 55-85). line_drop of '#gain' is read live when the line is
formatted (logger.c 181-182), so the copies of one snapshot may differ in it. The run ends with
'#us_end' (three copies) and '#run_end' (three lines). agv_logger.py writes '#' lines to
<csv>.events.txt as '<iso time> <line>', in arrival order. A run is one motor-on stretch; one file = one
run (CG7).

The run's lines (R134C-2; Codex round 2 R134C2-2, -4): the '#us,ev=start' lines with first_seq = log_seq
of the first motor-on data line give the boot of the run (all of them the same boot, else no evidence).
The '#gain' lines that count are those with that first_seq whose receive time (the ISO time agv_logger
writes in front of each events line: one clock, one writer) is within GAIN_US_WINDOW_S of the first copy
of the run's '#us,ev=start': the firmware pushes the three '#gain' and the '#us,ev=start' in the same
control tick (main.c 342-356; CLOSED_LOOP/2026-11-13_run14 k1a: 6.5 ms apart), while a line of an earlier
boot is older by at least the boot time of the ESP32. Order of arrival is not used (UDP may reorder). A
line without a readable time does not count. Their gain content (id, applied_n, ctrl, the 7 gains,
first_seq) must agree; line_drop is reported (largest), not compared (R134C-5). Limit (R134C2-4,
DEFERRED): a run whose '#us,ev=start' was dropped by the firmware (queue full, logger.c 88) has no boot
evidence and fails, although its '#gain' lines are there. Limit (Codex round 3 R134C3-1, -2): a receive
time does not prove the source of a line (a line of an earlier boot read late by a stalled logger could
fall in the window; a valid line read more than GAIN_US_WINDOW_S late is refused).
'#gain' with boot (Liam 14/11 CODE_R134, hop chon, nguyen van "Them boot vao #gain"; firmware change
scheduled by CODE, not in AGV 16702ff): a '#gain' line may carry a field boot=<hex> anywhere after ctrl.
When the lines of the run's first_seq carry it, only those whose boot equals the boot of the run's
'#us,ev=start' count, whatever their receive time (the window is not used); lines without it (firmware
16702ff) keep the window rule.

ESP32 log (R134C-3, R134C2-3, S05 6.3, L92): read_esp32_log is the one reader of the data lines for this
tool and s05_final_table.py. With <csv>.segments.csv of agv_logger.py, lines marked 'duplicate' are dropped
(counted), 'conflict' / 'ambiguous' lines are refused; with or without it, two kept data lines with the
same log_seq are refused, in one logger segment or in two ("cung log_seq ma khac noi dung": a run failed
by the measurement system; a duplicate is dropped only when the logger marked it).

Query answers (R134C-1, R134C2-1): read_query reads one gain_send.py --query output (exactly one
'#gain_running' line, last line '#ket_qua,ma=0') and the '#query_pc,t_start=..,t_end=..' line (PC epoch
seconds around the question) that the two writers put first (hil_bench_session.py, pc_sender.py);
query_problem says why two answers do not show one boot of one board equal to the boot of the run, asked
in two questions one after the other (the 'before' answer ended before the 'after' question started; the
same file given twice fails that). Limit (R134C2-1, DEFERRED): the question times are not compared with
the time of the ESP32 log (on the bench the logger runs on the Pi, the questions on the PC: two clocks).

Gain file (format of tools/gain_send.py load_file, hil feat/wifi-gain 331cd76): JSON object
{"ctrl": "PID"|"ADRC", "id": 1..2147483647, "gains": {<7 names>: number}}; the names and aliases are
those of gain_send.NAMES / ALIASES (copied below; that branch is not merged into this one yet). The
firmware receives the float32 of each value (gain_send.fmt, %.9g of the float32); '#gain' prints %.9g of
the float32 it holds, so the comparison is float32 bit for bit. Limit (R134C-6, kept): a flat file
(gains only, ctrl and id given on the gain_send.py command line) is refused here; campaign gain files
(*gain_*.json, L81) hold ctrl, id and gains.

  python tools/gain_line_check.py --log <run.csv> --gain-file <..._gain_....json>
      [--query-before F --query-after F]
  python tools/gain_line_check.py --log <run.csv> --builtin [--gain-file <file of the built-in set>]
      --builtin: the run should run the firmware's built-in gains (id 0); with --gain-file the 7 values
      are compared too. --query-before / --query-after (gain_send.py --query outputs): their boot and MAC
      must agree and the boot must be the run's.
Output: one JSON summary on stdout, last line '#ket_qua,ma=N': 0 the line matches; 1 missing, different,
or id 0 against a set (a run failed by operation); 2 input error (no log, no data line, bad file).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys

NAMES = {'PID': ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi'],
         'ADRC': ['omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi']}
ALIASES = {'wc_xy': 'omega_c_xy', 'wo_xy': 'omega_o_xy', 'wc_wz': 'omega_c_wz', 'wo_wz': 'omega_o_wz'}
ID_MAX = 2147483647
GAIN_FIELDS = ('id', 'applied_n', 'ctrl')
TAIL_FIELDS = ('first_seq', 'line_drop')
OPT_FIELDS = ('boot',)           # Liam 14/11 "Them boot vao #gain" (firmware after 16702ff)
SEGMENT_STATUS_BAD = ('conflict', 'ambiguous')
HEX = set('0123456789abcdefABCDEF')
GAIN_US_WINDOW_S = 0.5           # [DE XUAT] '#gain' to '#us,ev=start' of one run (same tick, ms apart)
LEGACY_GAIN_FW = ('16702ff0e688', '54b7269ce21d')


class InputError(ValueError):
    pass


def f32_bits(x) -> bytes:
    """float32 bytes of x (raises InputError when it is not a finite number that fits)."""
    try:
        v = float(x)
        b = struct.pack('<f', v)
    except (TypeError, ValueError, OverflowError, struct.error):
        raise InputError(f'{x!r} is not a number that fits a float32')
    if not math.isfinite(struct.unpack('<f', b)[0]):
        raise InputError(f'{x!r} is not finite as a float32')
    return b


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- ESP32 data lines (R134C-3)
def read_esp32_log_info(path):
    """(data lines, number of lines dropped as duplicates) of an agv_logger CSV (header line first; '#'
    lines go to the events file). When <path>.segments.csv of agv_logger.py is there (one row per data
    line: row, log_seq, timestamp_us, segment, status), each line gets its segment in '_segment'; lines
    marked 'duplicate' (identical to an earlier line, L92) are dropped; a 'conflict' or 'ambiguous' line,
    rows that do not match the CSV, or two kept lines with the same (segment, log_seq) raise InputError.
    Without the sidecar, two data lines with the same log_seq raise InputError (S05 6.3: a duplicate is
    dropped only when the logger marked it)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if _int(r.get('timestamp_us')) is not None]
    seg_path = path + '.segments.csv'
    if not os.path.isfile(seg_path):
        seen = set()
        for i, r in enumerate(rows, start=1):
            s = r.get('log_seq')
            if s in seen:
                raise InputError(f'{path}: two data lines with log_seq {s} (line {i}) and no {seg_path}: '
                                 'not usable as is (S05 6.3)')
            seen.add(s)
        return rows, 0
    with open(seg_path, newline='', encoding='utf-8') as f:
        marks = list(csv.DictReader(f))
    if len(marks) != len(rows):
        raise InputError(f'{seg_path}: {len(marks)} rows for {len(rows)} data lines')
    out, seen, keys, dropped = [], set(), set(), 0
    for i, (r, m) in enumerate(zip(rows, marks), start=1):
        if (m.get('row'), m.get('log_seq'), m.get('timestamp_us')) != (str(i), r.get('log_seq'),
                                                                        r.get('timestamp_us')):
            raise InputError(f'{seg_path} row {i} does not match data line {i} of {path}')
        status = m.get('status')
        if status in SEGMENT_STATUS_BAD:
            raise InputError(f'{path}: {status} line {i} (see {seg_path}); not usable as is')
        key = tuple(r.values())
        if status == 'duplicate':
            if key not in seen:
                raise InputError(f'{seg_path} row {i}: marked duplicate but no earlier equal line')
            dropped += 1
            continue
        sk = r.get('log_seq')
        if sk in keys:                   # R134C2-3: in one segment or in two (S05 6.3)
            raise InputError(f'{path}: line {i}: log_seq {sk} on two kept lines (segment {m.get("segment")}), '
                             'not marked duplicate: not usable as is (S05 6.3)')
        keys.add(sk)
        seen.add(key)
        out.append(dict(r, _segment=m.get('segment')))
    return out, dropped


def read_esp32_log(path):
    """Data lines only (see read_esp32_log_info)."""
    return read_esp32_log_info(path)[0]


def first_run_seq(rows, name='log') -> int:
    """log_seq of the first data line with motor_on = 1 (the first line of the run, CG7: one file = one run).
    rows: data lines (read_esp32_log), or a CSV path. More than one motor-on stretch: InputError."""
    if isinstance(rows, str):
        name, rows = rows, read_esp32_log(rows)
    if not rows or 'motor_on' not in rows[0] or 'log_seq' not in rows[0]:
        raise InputError(f'{name}: no data line with motor_on and log_seq')
    on = [r['motor_on'] == '1' for r in rows]
    rises = [i for i in range(len(rows)) if on[i] and (i == 0 or not on[i - 1])]
    if not rises:
        raise InputError(f'{name}: no line with motor_on = 1 (no run)')
    if len(rises) > 1:
        raise InputError(f'{name}: {len(rises)} motor-on stretches (one file = one run)')
    if rises[0] == 0:
        raise InputError(f'{name}: the first data line already has motor_on = 1 (start of the run not logged)')
    return int(rows[rises[0]]['log_seq'])


# ---------------------------------------------------------------- events file (R134C-2, -5)
def parse_gain_line(text: str) -> dict:
    """Fields of one '#gain,...' line (after the logger's time prefix); InputError when malformed."""
    parts = text.strip().split(',')
    if not parts or parts[0] != '#gain':
        raise InputError(f'not a #gain line: {text[:80]!r}')
    out = {}
    for p in parts[1:]:
        k, sep, v = p.partition('=')
        if not sep or not k or k in out:
            raise InputError(f'bad field {p!r} in {text[:80]!r}')
        out[k] = v
    missing = [k for k in GAIN_FIELDS + TAIL_FIELDS if k not in out]
    if missing:
        raise InputError(f'#gain line without {missing}: {text[:80]!r}')
    ctrl = out['ctrl']
    if ctrl not in NAMES:
        raise InputError(f'#gain ctrl={ctrl!r}')
    names = [k for k in out if k not in GAIN_FIELDS + TAIL_FIELDS + OPT_FIELDS]
    if 'boot' in out and (not out['boot'] or not set(out['boot']) <= HEX):
        raise InputError(f'#gain boot={out["boot"]!r}')
    if names != NAMES[ctrl]:
        raise InputError(f'#gain names {names}, firmware order is {NAMES[ctrl]}')
    for k in ('id', 'applied_n', 'first_seq', 'line_drop'):
        if not out[k].isdigit():
            raise InputError(f'#gain {k}={out[k]!r}')
    return out


def _fields(text: str) -> dict:
    return dict(p.partition('=')[::2] for p in text.strip().split(',')[1:])


def read_events(path) -> list:
    """Lines of an agv_logger events file that hold a '#' line ('<iso time> <line>'), in file order, time
    kept; None when there is no file."""
    if not path or not os.path.isfile(path):
        return None
    out = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            if '#' in line:
                out.append(line)
    return out


def split_event(line: str):
    """(receive time in s, or None, '#...' text) of one events line."""
    import datetime as _dt
    i = line.find('#')
    head = line[:i].strip()
    try:
        t = _dt.datetime.fromisoformat(head).timestamp() if head else None
    except ValueError:
        t = None
    return t, line[i:]


def gain_lines(events_path: str) -> list:
    """Every '#gain' line of an events file, parsed (copies kept). InputError on a malformed line."""
    return [parse_gain_line(t) for _, t in map(split_event, read_events(events_path) or []) if t.startswith('#gain,')]


def _content(g: dict) -> tuple:
    return tuple((k, v) for k, v in g.items() if k != 'line_drop')


def run_start(events, seq0: int, window_s: float = GAIN_US_WINDOW_S) -> dict:
    """The run's start in the events (module docstring): dict(boot, gain (parsed '#gain' dict or None),
    n_gain_lines, n_gain_outside (same first_seq, outside the time window), line_drop_max, why (list: empty
    when boot and gain line are both shown)). events: read_events output (None = no file). InputError on a
    malformed '#gain' line."""
    out = dict(boot=None, gain=None, n_gain_lines=0, n_gain_outside=0, line_drop_max=None, why=[])
    if events is None:
        out['why'].append('no events file (#gain, #us lines)')
        return out
    starts, gains, firmware = [], [], set()
    for line in events:
        t, text = split_event(line)
        if text.startswith('#gain,'):
            g = parse_gain_line(text)
            if _int(g['first_seq']) == seq0:
                gains.append((t, g))
        elif text.startswith('#us,'):
            f = _fields(text)
            if f.get('ev') == 'start' and _int(f.get('first_seq')) == seq0:
                starts.append((t, f))
        elif text.startswith('#fw,'):
            f = _fields(text)
            firmware.add((f.get('sha', '').lower(), f.get('env', ''), f.get('dirty')))
    if not starts:
        out['why'].append(f'no #us,ev=start line with first_seq = {seq0}: the boot of the run is not shown')
        return out
    boots = {f.get('boot', '').lower() for _, f in starts}
    boot = next(iter(boots))
    if len(boots) != 1 or not boot or not set(boot) <= HEX:
        out['why'].append(f'#us,ev=start lines with first_seq = {seq0} carry boots {sorted(boots)}: '
                          'the run cannot be told apart')
        return out
    out['boot'] = boot
    times = [t for t, _ in starts if t is not None]
    if not times:
        out['why'].append(f'the #us,ev=start lines of the run (first_seq = {seq0}) have no logger time')
        return out
    t_us = min(times)
    with_boot = [g for _, g in gains if 'boot' in g]
    if with_boot:                        # firmware with boot on '#gain': the boot decides, not the time
        gs = [g for g in with_boot if g['boot'].lower() == boot]
        out.update(n_gain_lines=len(gs), n_gain_outside=len(gains) - len(gs), gain_rule='boot')
        if not gs:
            out['why'].append(f'no #gain line with first_seq = {seq0} and boot {boot} (the boot of the run)')
            return out
    else:
        if not firmware or any(not env or dirty != '0' or not set(sha) <= HEX
                               or not any(sha == old[:7] or sha.startswith(old) for old in LEGACY_GAIN_FW)
                               for sha, env, dirty in firmware):
            out['why'].append(f'no #gain line with boot {boot} for first_seq = {seq0}; '
                              'the 0.5 s rule needs #fw of a known legacy build')
            return out
        gs = [g for t, g in gains if t is not None and abs(t - t_us) <= window_s]
        out.update(n_gain_lines=len(gs), n_gain_outside=len(gains) - len(gs), gain_rule='time_window')
        if not gs:
            out['why'].append(f'no #gain line with first_seq = {seq0} (the first data line of the run) within '
                              f'{window_s:g} s of its #us,ev=start (boot {boot})')
            return out
    out['line_drop_max'] = max(int(g['line_drop']) for g in gs)
    if len({_content(g) for g in gs}) > 1:
        out['why'].append(f'#gain lines of the run (first_seq = {seq0}) disagree in id, applied_n, ctrl or a '
                          'gain: a conflict')
        return out
    out['gain'] = gs[0]
    return out


# ---------------------------------------------------------------- gain_send --query answers (R134C-1)
def read_query(path):
    """Fields (sha, env, mac, boot, id, ...) of the '#gain_running' line of one gain_send.py --query output
    file, or None (no file, not exactly one such line, the last line is not '#ket_qua,ma=0', no hex boot)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        lines = [x.strip() for x in f if x.strip()]
    run = [x for x in lines if x.startswith('#gain_running,')]
    if len(run) != 1 or not lines or lines[-1] != '#ket_qua,ma=0':
        return None
    fields = _fields(run[0])
    b = fields.get('boot', '')
    if not b or not set(b) <= HEX:
        return None
    pc = [x for x in lines if x.startswith('#query_pc,')]     # R134C2-1: written by the two writers
    t = _fields(pc[0]) if len(pc) == 1 else {}
    try:
        fields['_t_start'], fields['_t_end'] = float(t['t_start']), float(t['t_end'])
    except (KeyError, ValueError):
        fields['_t_start'] = fields['_t_end'] = None
    fields['_path'] = os.path.realpath(path)
    return fields


def query_pc_line(t_start: float, t_end: float) -> str:
    """First line of a saved query output (R134C2-1): PC epoch seconds just before and just after it."""
    return f'#query_pc,t_start={t_start:.6f},t_end={t_end:.6f}\n'


def query_problem(before, after, run_boot=None, check_run_boot=True):
    """None, or why the two query answers (read_query dicts, None = missing) do not show one boot of one
    board over the run, equal to the boot of the run's '#us,ev=start' (run_boot; None = not shown).
    check_run_boot False: only before against after (no ESP32 log at hand)."""
    if before is None or after is None:
        return f'thieu hoi boot {"truoc" if before is None else "sau"} luot (gain_send --query; S05 4.7 (iv))'
    if before.get('_path') and before.get('_path') == after.get('_path'):
        return 'cung mot file cho hoi truoc va hoi sau: khong phai hai lan hoi (R134C2-1)'
    times = [before.get('_t_start'), before.get('_t_end'), after.get('_t_start'), after.get('_t_end')]
    if any(t is None for t in times):
        return 'file hoi khong co dong #query_pc (gio PC cua lan hoi): khong biet hai lan hoi la hai lan (R134C2-1)'
    if not all(math.isfinite(t) for t in times) or times[0] > times[1] or times[2] > times[3]:
        return f'gio #query_pc khong hop le {times} (khong huu han hay khoang nguoc, R134C3-4)'
    tb, ta = times[1], times[2]
    if tb > ta:
        return f'hoi truoc xong luc {tb:.3f} sau khi hoi sau bat dau {ta:.3f}: khong phai truoc va sau luot'
    if before.get('tx') and before['tx'] == after.get('tx'):
        return 'hai file hoi co cung tx: khong phai hai lan hoi'
    bb, ba = before['boot'].lower(), after['boot'].lower()
    if bb != ba:
        return f'boot truoc {before["boot"]} khac boot sau {after["boot"]}: ESP32 khoi dong lai giua luot'
    if (before.get('mac') or '').lower() != (after.get('mac') or '').lower():
        return f'mac truoc {before.get("mac")} khac mac sau {after.get("mac")}: hai board khac nhau'
    if check_run_boot:
        if run_boot is None:
            return 'log cua luot khong co boot (#us,ev=start): khong gan duoc hai lan hoi voi luot'
        if run_boot.lower() != bb:
            return (f'boot cua hai lan hoi {before["boot"]} khac boot cua luot {run_boot} (#us,ev=start): '
                    'cau tra loi cu hay board khac')
    return None


def read_ack(path):
    """B8-1 B81-14 (S05 v3.20 4.7 (iv)): fields of the '#gain_ack' answer in one saved output of gain_send.py
    setting a gain set (hil feat/wifi-gain 331cd76 main: one '# ack <k=v,...>' line, last line '#ket_qua,ma=0'
    when applied and verified), or None (no file, not exactly one ack line, not ok=1, not ma=0, no hex boot)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        lines = [x.strip() for x in f if x.strip()]
    acks = [x for x in lines if x.startswith('# ack ')]
    if len(acks) != 1 or not lines or lines[-1] != '#ket_qua,ma=0':
        return None
    fields = dict(p.partition('=')[::2] for p in acks[0][len('# ack '):].split(','))
    b = fields.get('boot', '')
    if fields.get('ok') != '1' or not b or not set(b) <= HEX:
        return None
    return fields


def ack_problem(ack, before, run_gain=None):
    """None, or why the '#gain_ack' of the gain setting (read_ack dict, None = missing) is not of the boot and the
    board of the two query answers (`before`, already checked against `after` by query_problem) and of the set the
    run ran (`run_gain`: the run's '#gain' dict, or the running id of the question before, {'id': ..}; None = not
    shown). Every final run has an ack (R134C5-3; Liam 14/11 CODE_R134 hop chon "Bat buoc ack moi luot
    (Recommended)": every final run, the initial gain too, sets its gain from a file with --gain-file), so a
    missing ack is always a problem, also with id 0 on the run's '#gain' line (R134C4-5: a set applied, then an
    ESP32 restart, leaves id 0 and the ack of the older boot)."""
    if ack is None:
        return 'thieu cau tra loi #gain_ack luc dat gain (gain_send.py, ok=1, ma=0; S05 4.7 (iv))'
    if before is None:
        return 'thieu hoi boot truoc luot: khong so duoc #gain_ack'
    if ack['boot'].lower() != before['boot'].lower():
        return (f'boot #gain_ack luc dat gain {ack["boot"]} khac boot hai lan hoi {before["boot"]}: ESP32 khoi dong '
                'lai sau khi dat gain (gain da mat) hay file ack cu')
    if (ack.get('mac') or '').lower() != (before.get('mac') or '').lower():
        return f'mac #gain_ack {ack.get("mac")} khac mac hai lan hoi {before.get("mac")}: board khac'
    if run_gain is not None:
        # R134C6-2 (Codex round 6): both ids present and decimal before they are compared; two missing or two empty
        # ids are no match
        a_id, r_id = ack.get('id'), run_gain.get('id')
        if not _dec(a_id) or not _dec(r_id):
            return (f'id #gain_ack {a_id!r} hay id dang chay {r_id!r} thieu hay khong phai so nguyen: khong so duoc '
                    'bo gain')
        if int(a_id) != int(r_id):
            return f'id #gain_ack {a_id} khac id dong #gain cua luot {r_id}: ack cua bo gain khac'
    return None


def _dec(v) -> bool:
    """A non-empty text of decimal digits (an id of the firmware lines)."""
    return isinstance(v, str) and v.isascii() and v.isdigit()


def gain_mismatch(g: dict, want: dict) -> list:
    """Why the run's '#gain' line `g` (run_start()['gain']) is not the set of the gain file `want`
    (load_gain_file); [] when id, ctrl and every gain (float32 bits) agree. Same rules as check() without
    --builtin; used by the table (s05_final_table, R134C6-1)."""
    why = []
    if g['id'] == '0':
        why.append(f'id=0 (built-in set) while the run should run id {want["id"]}')
    elif not _dec(g['id']) or int(g['id']) != want['id']:
        why.append(f'id={g["id"]}, gain file id {want["id"]}')
    if g['ctrl'] != want['ctrl']:
        why.append(f'ctrl={g["ctrl"]}, gain file {want["ctrl"]}')
    else:
        for n in NAMES[want['ctrl']]:
            if f32_bits(g[n]) != want['gains'][n]:
                why.append(f'{n}: log {g[n]}, file {struct.unpack("<f", want["gains"][n])[0]!r}')
    return why


def run_boot_of(csv_path, events_path=None):
    """(boot of the run's '#us,ev=start', None) or (None, why) for one agv_logger CSV of one run."""
    try:
        seq0 = first_run_seq(read_esp32_log(csv_path), csv_path)
        st = run_start(read_events(events_path or csv_path + '.events.txt'), seq0)
    except (OSError, InputError) as e:
        return None, str(e)
    return st['boot'], ('; '.join(st['why']) if st['boot'] is None else None)


# ---------------------------------------------------------------- gain file
def load_gain_file(path: str) -> dict:
    """{'ctrl', 'id', 'gains' {name: float32 bytes}} of a gain file (gain_send.load_file format)."""
    def no_dup(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise InputError(f'{path}: JSON key {k!r} given twice')
            d[k] = v
        return d
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f, object_pairs_hook=no_dup)
    except (OSError, ValueError) as e:
        raise InputError(f'{path}: {e}')
    if not isinstance(d, dict) or not isinstance(d.get('gains'), dict):
        raise InputError(f'{path}: an object with "ctrl", "id" and "gains" expected')
    ctrl, set_id = d.get('ctrl'), d.get('id')
    if ctrl not in NAMES:
        raise InputError(f'{path}: ctrl {ctrl!r}')
    if isinstance(set_id, bool) or not isinstance(set_id, int) or not 1 <= set_id <= ID_MAX:
        raise InputError(f'{path}: id must be an integer 1..{ID_MAX}, got {set_id!r}')
    g = {}
    for k, v in d['gains'].items():
        n = ALIASES.get(k, k)
        if n in g:
            raise InputError(f'{path}: gain {n} given twice')
        if isinstance(v, bool):
            raise InputError(f'{path}: gain {k}={v!r}')
        g[n] = f32_bits(v)
    if sorted(g) != sorted(NAMES[ctrl]):
        raise InputError(f'{path}: {ctrl} needs exactly {NAMES[ctrl]}, got {sorted(g)}')
    return dict(ctrl=ctrl, id=set_id, gains=g)


def check(csv_path: str, gain_file=None, builtin=False, events_path=None, queries=None) -> dict:
    """Summary dict with 'ok' (bool) and 'why' (list of reasons); InputError on bad input.
    queries: (path before, path after) of gain_send.py --query outputs, or None (not checked)."""
    if gain_file is None and not builtin:
        raise InputError('give --gain-file, or --builtin for the firmware built-in set')
    events_path = events_path or csv_path + '.events.txt'
    if not os.path.isfile(events_path):
        raise InputError(f'{events_path}: no events file')
    rows, n_drop = read_esp32_log_info(csv_path)
    seq0 = first_run_seq(rows, csv_path)
    st = run_start(read_events(events_path), seq0)
    want = load_gain_file(gain_file) if gain_file else None
    out = dict(log=csv_path, events=events_path, first_data_seq=seq0, rows_dropped=n_drop,
               run_boot=st['boot'], n_gain_lines=st['n_gain_lines'], n_gain_outside=st['n_gain_outside'],
               line_drop_max=st['line_drop_max'],
               gain_file=gain_file, builtin=builtin, ok=False, why=list(st['why']))
    if queries is not None:
        qb, qa = (read_query(p) for p in queries)
        out.update(query_boot_before=qb and qb['boot'], query_boot_after=qa and qa['boot'],
                   query_mac_before=qb and qb.get('mac'), query_mac_after=qa and qa.get('mac'))
        bad = query_problem(qb, qa, st['boot'])
        if bad:
            out['why'].append(bad)
    g = st['gain']
    if g is None:
        return out
    out.update(line_id=int(g['id']), line_ctrl=g['ctrl'], applied_n=int(g['applied_n']))
    if builtin:
        if g['id'] != '0':
            out['why'].append(f'id={g["id"]}: the run should run the built-in set (id 0)')
    else:
        if g['id'] == '0':
            out['why'].append(f'id=0 (built-in set) while the run should run id {want["id"]}')
        elif int(g['id']) != want['id']:
            out['why'].append(f'id={g["id"]}, gain file id {want["id"]}')
    if want is not None:
        if g['ctrl'] != want['ctrl']:
            out['why'].append(f'ctrl={g["ctrl"]}, gain file {want["ctrl"]}')
        else:
            for n in NAMES[want['ctrl']]:
                if f32_bits(g[n]) != want['gains'][n]:
                    out['why'].append(f'{n}: log {g[n]}, file {struct.unpack("<f", want["gains"][n])[0]!r}')
    out['ok'] = not out['why']
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--log', required=True, help='agv_logger CSV of one run (its .events.txt next to it)')
    ap.add_argument('--events', default=None, help='events file (default <log>.events.txt)')
    ap.add_argument('--gain-file', default=None, help='gain file of the set that should run')
    ap.add_argument('--builtin', action='store_true', help='the run should run the built-in set (id 0)')
    ap.add_argument('--query-before', default=None, help='gain_send.py --query output just before the run')
    ap.add_argument('--query-after', default=None, help='gain_send.py --query output just after the run')
    a = ap.parse_args(argv)
    if (a.query_before is None) != (a.query_after is None):
        ap.error('--query-before and --query-after go together')
    try:
        s = check(a.log, a.gain_file, a.builtin, a.events,
                  queries=(a.query_before, a.query_after) if a.query_before is not None else None)
    except (OSError, InputError) as e:
        print(json.dumps(dict(error=str(e))))
        print('#ket_qua,ma=2')
        return 2
    print(json.dumps(s, ensure_ascii=False))
    code = 0 if s['ok'] else 1
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
