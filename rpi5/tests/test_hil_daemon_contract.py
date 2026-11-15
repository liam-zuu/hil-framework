"""Contract between run_hil.py / hil_node.py and hil_daemon.py (CODE_PANEL 14/11, plan
03_CODE/docs/2026-11-14_Ke_hoach_CODE_PANEL_phia_Pi_v1.md item 6; Codex round 1 PPI-11).

The daemon follows a run only through the lines run_hil.py prints. run_hil.py itself cannot run
on the PC (os.sched_setaffinity, SPI), and tests/fake_run_hil.py prints lines written by hand, so
a change in run_hil.py or hil_node.py would not be seen by test_hil_daemon.py. Here, from the
source (ast) of every log call the daemon depends on:
  * the format string, filled with values of the right type, must be read by the daemon's single
    parser (HilDaemon._parse, the one the reader thread uses) into the right snapshot fields;
  * the expressions passed as arguments are pinned: a change of WHAT is printed (e.g. the CRC
    counter replaced by a constant) fails here and must be looked at;
  * lines that only look alike (#phase,RUN_BOGUS) are not taken."""
import ast
import json
import os
import re

import pytest

import hil_daemon as hd
import panel_proto as pp

RPI5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_calls(fname):
    """(format string, [argument source]) of every log.<level>(...) call in rpi5/<fname>."""
    with open(os.path.join(RPI5, fname), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('info', 'warning', 'error') and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out.append((node.args[0].value, [ast.unparse(a) for a in node.args[1:]]))
    return out


def one(fname, marker):
    calls = [c for c in log_calls(fname) if marker in c[0]]
    assert len(calls) == 1, f'{fname}: {len(calls)} log calls contain {marker!r}'
    return calls[0]


_SPEC = re.compile(r'%[-+ 0#]*\d*(?:\.\d+)?([sdf])')


def fill(fmt, values):
    """fmt % values, after checking each value has the type its conversion takes (d: int,
    f: float, s: anything), in order."""
    kinds = _SPEC.findall(fmt)
    assert len(kinds) == len(values), (fmt, values)
    for k, v in zip(kinds, values):
        assert {'d': isinstance(v, int), 'f': isinstance(v, float), 's': True}[k], (k, v)
    return fmt % tuple(values)


def daemon(tmp_path):
    return hd.HilDaemon(str(tmp_path), tests=[('T', ['--test', 'A-L1'])], busy_check=lambda pid: False)


PINNED = {
    ('run_hil.py', '#phase,'): ['ph', 'time.monotonic_ns()'],
    ('run_hil.py', 'jitter (last'): ["s['step']", 'el', "s['step'] / el if el > 0 else 0", "s['n']",
                                     "s.get('p99_us')", "s.get('max_us')", "s['warn']", "s['overruns']"],
    ('run_hil.py', 'whole run'): ["s['n']", "s.get('p99_us')", "s.get('p999_us')", "s['max_all_us']",
                                  "s['warn']", "s['overruns']"],
    ('hil_node.py', '#run_summary,'): ['json.dumps(self.lifecycle_summary())'],
    ('hil_node.py', '#miso,'): ["m['checked']", "m['crc_bad']", "m['empty']", "m['first_crc_bad']",
                                "m['first_empty']"],
    ('hil_node.py', '#swa,'): ['self.cfg.swa_mode', 'json.dumps(self.swa_events())', 'json.dumps(self.swa_tail())'],
}


@pytest.mark.parametrize('key', list(PINNED))
def test_what_is_printed_is_pinned(key):
    _, args = one(*key)
    assert args == PINNED[key], f'{key}: the producer now prints {args}; check hil_daemon still reads it right'


@pytest.mark.parametrize('ph', hd._PHASES)
def test_phase_line_through_the_parser(tmp_path, ph):
    d = daemon(tmp_path)
    d._parse(fill(one('run_hil.py', '#phase,')[0], [ph, 123456789]), {})
    assert d._seen_phase and d.state == hd._PHASE_ST[ph]


@pytest.mark.parametrize('bogus', ['#phase,RUN_BOGUS,t_ns=1', '#phase,RUN', '#phase,run,t_ns=1',
                                   '#phases,RUN,t_ns=1x'])
def test_lookalike_phase_lines_are_not_taken(tmp_path, bogus):
    d = daemon(tmp_path)
    d._parse(bogus, {})
    assert not d._seen_phase and d.state == pp.ST_IDLE


@pytest.mark.parametrize('p99,mx', [('312.5', '480.0'), ('None', 'None')])
def test_window_stats_line(tmp_path, p99, mx):
    line = fill(one('run_hil.py', 'jitter (last')[0], [10000, 10.0, 1000.0, 10000, p99, mx, 2, 1])
    d, acc = daemon(tmp_path), {}
    d._parse(line, acc)
    assert d.loop_us == (None if p99 == 'None' else 312) and d.overruns == 1 and 'last' not in acc


@pytest.mark.parametrize('p99', ['312.5', 'None'])
def test_whole_run_line(tmp_path, p99):
    line = fill(one('run_hil.py', 'whole run')[0], [47000, p99, '400.0', 512.3, 4, 3])
    d, acc = daemon(tmp_path), {}
    d._parse(line, acc)
    assert acc['last'] == (None if p99 == 'None' else 312.5, 512.3, 3) and d.overruns == 3


def test_run_summary_line(tmp_path):
    summary = {'outcome': 'completed', 'phases': {'RUN': 1.0}, 'detail': None}
    d, acc = daemon(tmp_path), {}
    d._parse(fill(one('hil_node.py', '#run_summary,')[0], [json.dumps(summary)]), acc)
    assert acc['summary'] == summary


def test_miso_line(tmp_path):
    d, acc = daemon(tmp_path), {}
    d._parse(fill(one('hil_node.py', '#miso,')[0], [1498, 0, 2, 'None', 17]), acc)
    assert acc['miso'] == (1498, 0, 2)


def test_swa_line(tmp_path):
    d = daemon(tmp_path)
    d._parse(fill(one('hil_node.py', '#swa,')[0], ['phase', json.dumps([[0, 1]]), json.dumps({'ok': True})]), {})
    assert d._meta['swa_mode'] == 'phase'


def test_warning_and_traceback_lines_change_nothing(tmp_path):
    d, acc = daemon(tmp_path), {}
    for line in ('02:22:16 WARNING  hil.node: MISO: 3 frame CRC sai, 0 frame rỗng trong lượt (L50: không làm số chính)',
                 'Traceback (most recent call last):', '  File "run_hil.py", line 115, in _run_plant',
                 'AttributeError: module os has no attribute sched_setaffinity', 'x' * 5000):
        d._parse(line, acc)
    assert acc == {} and d.state == pp.ST_IDLE and d.loop_us is None


def test_the_lines_reach_the_daemon_as_logging_prints_them(tmp_path):
    """run_hil.py's log format puts a time and the logger name before the message: the daemon
    searches, it does not match from the start of the line."""
    with open(os.path.join(RPI5, 'run_hil.py'), encoding='utf-8') as f:
        src = f.read()
    m = re.search(r"format\s*=\s*['\"]([^'\"]+)['\"]", src)
    assert m, 'run_hil.py logging format not found'
    prefix = m.group(1).replace('%(asctime)s', '02:22:15').replace('%(levelname)-8s', 'INFO    ') \
        .replace('%(levelname)s', 'INFO').replace('%(name)s', 'hil.main')
    line = prefix.replace('%(message)s', fill(one('run_hil.py', '#phase,')[0], ['ARM', 5]))
    d = daemon(tmp_path)
    d._parse(line, {})
    assert d.state == pp.ST_ARM
