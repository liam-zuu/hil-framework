#!/usr/bin/env python3
"""
gain_line_check.py: the '#gain' line of one run against the gain file of the set that should run
(S05 v3.13 section 6.3 "Kiem gain cua tung luot [L90]", RS-9 of the R130 review; L90 stage 2).
R134 package, Liam 14/11/2026 CODE_R134 local_eb9f3b5c: "duyet goi sua code fix/r134-answers".

Rule (S05 6.3): every campaign run and final-check run has a '#gain' line in its log (id, ctrl, the 7
gains) equal bit for bit to the gain file of the running set. The gain file holds the integer id sent to
the firmware ('gain_file' of S05_freeze.csv, or the trial file of the tuning round); gain_set_id (with
its campaign prefix) is the key of S05_freeze.csv and is never sent. The '#gain' line of the run is the
one whose first_seq equals log_seq of the first data line of the run (lines with first_seq = 0, written
when gains change with the motors off, do not count). A missing line, a difference, or id = 0 when the
run should run a set other than the built-in one: a run failed by operation (handled as a run failed by
the measurement system, O1 3.5).

Firmware facts (AGV 16702ff): logger.c 176-185 '#gain' + gain_set_format (gain_set.c 241-252:
',id=%lu,applied_n=%lu,ctrl=%s' then ',<name>=%.9g' for the 7 gains of the controller) +
',first_seq=%lu,line_drop=%lu'; the line takes no log_seq (logger.c 63-75); main.c 342-356 run_udp_mark
pushes it three times at the motor-on edge with first_seq = logger_seq_next(), the log_seq of the first
data line of the run (pushed in the same tick). agv_logger.py writes '#' lines to <csv>.events.txt as
'<iso time> <line>'. A run is one motor-on stretch; one file = one run (CG7).

Gain file (format of tools/gain_send.py load_file, hil feat/wifi-gain 331cd76): JSON object
{"ctrl": "PID"|"ADRC", "id": 1..2147483647, "gains": {<7 names>: number}}; the names and aliases are
those of gain_send.NAMES / ALIASES (copied below; that branch is not merged into this one yet). The
firmware receives the float32 of each value (gain_send.fmt, %.9g of the float32); '#gain' prints %.9g of
the float32 it holds, so the comparison is float32 bit for bit.

  python tools/gain_line_check.py --log <run.csv> --gain-file <..._gain_....json>
  python tools/gain_line_check.py --log <run.csv> --builtin [--gain-file <file of the built-in set>]
      --builtin: the run should run the firmware's built-in gains (id 0); with --gain-file the 7 values
      are compared too.
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


class InputError(Exception):
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


def gain_lines(events_path: str) -> list:
    """Distinct '#gain' lines of an agv_logger events file ('<iso time> <line>'); the firmware sends each
    three times, identical copies count once; two different lines with the same first_seq: InputError."""
    seen, out = set(), []
    with open(events_path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            i = line.find('#gain,')
            if i < 0 or line[i:].startswith('#gain_'):
                continue
            text = line[i:]
            if text in seen:
                continue
            seen.add(text)
            out.append(parse_gain_line(text))
    by = {}
    for g in out:
        if g['first_seq'] != '0' and g['first_seq'] in by and by[g['first_seq']] != g:
            raise InputError(f'two different #gain lines with first_seq {g["first_seq"]}: a conflict')
        by[g['first_seq']] = g
    return out


def first_run_seq(csv_path: str) -> int:
    """log_seq of the first data line with motor_on = 1 (the first line of the run, CG7: one file = one run).
    More than one motor-on stretch in the file: InputError."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if (r.get('timestamp_us') or '').lstrip('-').isdigit()]
    if not rows or 'motor_on' not in rows[0] or 'log_seq' not in rows[0]:
        raise InputError(f'{csv_path}: no data line with motor_on and log_seq')
    on = [r['motor_on'] == '1' for r in rows]
    rises = [i for i in range(len(rows)) if on[i] and (i == 0 or not on[i - 1])]
    if not rises:
        raise InputError(f'{csv_path}: no line with motor_on = 1 (no run)')
    if len(rises) > 1:
        raise InputError(f'{csv_path}: {len(rises)} motor-on stretches (one file = one run)')
    if rises[0] == 0:
        raise InputError(f'{csv_path}: the first data line already has motor_on = 1 (start of the run not logged)')
    return int(rows[rises[0]]['log_seq'])


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


def check(csv_path: str, gain_file=None, builtin=False, events_path=None) -> dict:
    """Summary dict with 'ok' (bool) and 'why' (list of reasons); InputError on bad input."""
    if gain_file is None and not builtin:
        raise InputError('give --gain-file, or --builtin for the firmware built-in set')
    events_path = events_path or csv_path + '.events.txt'
    if not os.path.isfile(events_path):
        raise InputError(f'{events_path}: no events file')
    seq0 = first_run_seq(csv_path)
    lines = gain_lines(events_path)
    mine = [g for g in lines if int(g['first_seq']) == seq0]
    out = dict(log=csv_path, events=events_path, first_data_seq=seq0, n_gain_lines=len(lines),
               gain_file=gain_file, builtin=builtin, ok=False, why=[])
    if not mine:
        out['why'].append(f'no #gain line with first_seq = {seq0} (the first data line of the run)')
        return out
    g = mine[0]
    out.update(line_id=int(g['id']), line_ctrl=g['ctrl'], applied_n=int(g['applied_n']), line_drop=int(g['line_drop']))
    want = load_gain_file(gain_file) if gain_file else None
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
    a = ap.parse_args(argv)
    try:
        s = check(a.log, a.gain_file, a.builtin, a.events)
    except InputError as e:
        print(json.dumps(dict(error=str(e))))
        print('#ket_qua,ma=2')
        return 2
    print(json.dumps(s, ensure_ascii=False))
    code = 0 if s['ok'] else 1
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
