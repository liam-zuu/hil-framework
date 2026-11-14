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

The run's lines (R134C-2): the '#us,ev=start' lines with first_seq = log_seq of the first motor-on data
line give the boot of the run (all of them the same boot, else no evidence). The '#gain' lines that count
are those with that first_seq in the same block of the events file: after the last '#run_end',
'#us_end', or '#us,ev=start' of another (boot, first_seq) before the run's '#us,ev=start', and before the
first such line after it. Their gain content (id, applied_n, ctrl, the 7 gains, first_seq) must agree;
line_drop is reported (largest), not compared (R134C-5). Limit: a '#gain' line of an earlier boot with
the same first_seq whose '#us,ev=start', '#us_end' and '#run_end' were all lost cannot be told apart.

ESP32 log (R134C-3, S05 6.3, L92): read_esp32_log is the one reader of the data lines for this tool and
s05_final_table.py. With <csv>.segments.csv of agv_logger.py, lines marked 'duplicate' are dropped
(counted), 'conflict' / 'ambiguous' lines are refused; without it, two data lines with the same log_seq
are refused (a duplicate is dropped only when the logger marked it).

Query answers (R134C-1): read_query reads one gain_send.py --query output (exactly one '#gain_running'
line, last line '#ket_qua,ma=0'); query_problem says why two answers around a run do not show one boot of
one board equal to the boot of the run.

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
SEGMENT_STATUS_BAD = ('conflict', 'ambiguous')
HEX = set('0123456789abcdefABCDEF')


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
        sk = (m.get('segment'), r.get('log_seq'))
        if sk in keys:
            raise InputError(f'{path}: line {i}: log_seq {sk[1]} twice in segment {sk[0]}, not marked duplicate')
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
    names = [k for k in out if k not in GAIN_FIELDS + TAIL_FIELDS]
    if names != NAMES[ctrl]:
        raise InputError(f'#gain names {names}, firmware order is {NAMES[ctrl]}')
    for k in ('id', 'applied_n', 'first_seq', 'line_drop'):
        if not out[k].isdigit():
            raise InputError(f'#gain {k}={out[k]!r}')
    return out


def _fields(text: str) -> dict:
    return dict(p.partition('=')[::2] for p in text.strip().split(',')[1:])


def read_events(path) -> list:
    """'#' lines of an agv_logger events file ('<iso time> <line>'), in file order, without the time;
    None when there is no file."""
    if not path or not os.path.isfile(path):
        return None
    out = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            i = line.find('#')
            if i >= 0:
                out.append(line[i:])
    return out


def gain_lines(events_path: str) -> list:
    """Every '#gain' line of an events file, parsed (copies kept). InputError on a malformed line."""
    return [parse_gain_line(t) for t in (read_events(events_path) or []) if t.startswith('#gain,')]


def _content(g: dict) -> tuple:
    return tuple((k, v) for k, v in g.items() if k != 'line_drop')


def run_start(events, seq0: int) -> dict:
    """The run's start in the events (module docstring): dict(boot, gain (parsed '#gain' dict or None),
    n_gain_lines, line_drop_max, why (list: empty when boot and gain line are both shown)).
    events: read_events output (None = no file). InputError on a malformed '#gain' line."""
    out = dict(boot=None, gain=None, n_gain_lines=0, line_drop_max=None, why=[])
    if events is None:
        out['why'].append('no events file (#gain, #us lines)')
        return out
    parsed = []
    for t in events:
        if t.startswith('#gain,'):
            parsed.append(('gain', parse_gain_line(t)))
        elif t.startswith('#us,'):
            f = _fields(t)
            parsed.append(('us_start' if f.get('ev') == 'start' else 'us', f))
        elif t.startswith('#us_end,') or t.startswith('#run_end,'):
            parsed.append(('end', None))
        else:
            parsed.append(('other', None))
    mine = [i for i, (k, f) in enumerate(parsed) if k == 'us_start' and _int(f.get('first_seq')) == seq0]
    if not mine:
        out['why'].append(f'no #us,ev=start line with first_seq = {seq0}: the boot of the run is not shown')
        return out
    boots = {parsed[i][1].get('boot', '').lower() for i in mine}
    boot = next(iter(boots))
    if len(boots) != 1 or not boot or not set(boot) <= HEX:
        out['why'].append(f'#us,ev=start lines with first_seq = {seq0} carry boots {sorted(boots)}: '
                          'the run cannot be told apart')
        return out
    out['boot'] = boot

    def boundary(i):
        k, f = parsed[i]
        return k == 'end' or (k == 'us_start' and (_int(f.get('first_seq')) != seq0
                                                   or f.get('boot', '').lower() != boot))
    if any(boundary(i) for i in range(mine[0], mine[-1] + 1)):
        out['why'].append(f'the #us,ev=start lines of the run (first_seq = {seq0}) are split by lines of '
                          'another run: the run cannot be told apart')
        return out
    lo = max((i for i in range(mine[0]) if boundary(i)), default=-1)
    hi = min((i for i in range(mine[-1] + 1, len(parsed)) if boundary(i)), default=len(parsed))
    gs = [f for k, f in parsed[lo + 1:hi] if k == 'gain' and _int(f['first_seq']) == seq0]
    out['n_gain_lines'] = len(gs)
    if not gs:
        out['why'].append(f'no #gain line with first_seq = {seq0} (the first data line of the run) '
                          f'next to its #us,ev=start (boot {boot})')
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
    return fields if b and set(b) <= HEX else None


def query_problem(before, after, run_boot=None, check_run_boot=True):
    """None, or why the two query answers (read_query dicts, None = missing) do not show one boot of one
    board over the run, equal to the boot of the run's '#us,ev=start' (run_boot; None = not shown).
    check_run_boot False: only before against after (no ESP32 log at hand)."""
    if before is None or after is None:
        return f'thieu hoi boot {"truoc" if before is None else "sau"} luot (gain_send --query; S05 4.7 (iv))'
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
               run_boot=st['boot'], n_gain_lines=st['n_gain_lines'], line_drop_max=st['line_drop_max'],
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
