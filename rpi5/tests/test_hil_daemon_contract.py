"""Contract between run_hil.py / hil_node.py and hil_daemon.py (CODE_PANEL 14/11, plan
03_CODE/docs/2026-11-14_Ke_hoach_CODE_PANEL_phia_Pi_v1.md item 6).

The daemon follows a run only through the lines run_hil.py prints. run_hil.py itself cannot run
on the PC (os.sched_setaffinity, SPI), and tests/fake_run_hil.py prints lines written by hand, so
a change of a format string in run_hil.py or hil_node.py would not be seen by test_hil_daemon.py.
Here the real format strings are read from the source (ast: the literal first argument of every
log.info call), filled with sample values the way logging does ('fmt % args'), and the daemon's
regular expressions must still read them. A format change that the daemon cannot read fails here."""
import ast
import json
import os
import re

import pytest

import hil_daemon as hd

RPI5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_formats(fname):
    """Literal format strings of log.<level>(...) calls in rpi5/<fname>."""
    with open(os.path.join(RPI5, fname), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('info', 'warning', 'error') and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out.append(node.args[0].value)
    return out


def one(fname, marker):
    fmts = [f for f in log_formats(fname) if marker in f]
    assert len(fmts) == 1, f'{fname}: {len(fmts)} log formats contain {marker!r}: {fmts}'
    return fmts[0]


_SPEC = re.compile(r'%[-+ 0#]*\d*(?:\.\d+)?([sdf])')


def fill(fmt, values):
    """fmt % values, after checking each value fits its conversion (s, d, f) in order."""
    kinds = _SPEC.findall(fmt)
    assert len(kinds) == len(values), (fmt, values)
    return fmt % tuple(values)


def test_phase_line():
    line = fill(one('run_hil.py', '#phase,'), ['RUN', 123456789])
    m = hd._RE_PHASE.search(line)
    assert m and m.group(1) == 'RUN'
    for ph in hd._PHASE_ST:
        assert hd._RE_PHASE.search(fill(one('run_hil.py', '#phase,'), [ph, 1])).group(1) == ph


@pytest.mark.parametrize('p99,mx', [('312.5', '480.0'), ('None', 'None')])
def test_window_stats_line(p99, mx):
    line = fill(one('run_hil.py', 'jitter (last'), [10000, 10.0, 1000.0, 10000, p99, mx, 2, 1])
    assert not hd._RE_WHOLE.search(line)
    m = hd._RE_STATS.search(line)
    assert m and m.group(1) == p99 and m.group(3) == '1'


@pytest.mark.parametrize('p99', ['312.5', 'None'])
def test_whole_run_line(p99):
    line = fill(one('run_hil.py', 'whole run'), [47000, p99, '400.0', 512.3, 4, 3])
    m = hd._RE_WHOLE.search(line)
    assert m and m.group(1) == p99 and float(m.group(2)) == 512.3 and m.group(3) == '3'


def test_run_summary_line():
    summary = {'outcome': 'completed', 'phases': {'RUN': 1.0}, 'detail': None}
    line = fill(one('hil_node.py', '#run_summary,'), [json.dumps(summary)])
    m = hd._RE_SUMMARY.search(line)
    assert m and json.loads(m.group(1)) == summary


def test_miso_line():
    line = fill(one('hil_node.py', '#miso,'), [1498, 0, 2, 'None', 17])
    m = hd._RE_MISO.search(line)
    assert m and m.groups() == ('1498', '0', '2')


def test_swa_line():
    line = fill(one('hil_node.py', '#swa,'), ['phase', json.dumps([[0, 1]]), json.dumps({'ok': True})])
    m = hd._RE_SWA.search(line)
    assert m and m.group(1) == 'phase'


def test_the_lines_reach_the_daemon_as_logging_prints_them():
    """run_hil.py's log format puts a time and the logger name before the message: the daemon
    searches, it does not match from the start of the line."""
    with open(os.path.join(RPI5, 'run_hil.py'), encoding='utf-8') as f:
        src = f.read()
    m = re.search(r"format\s*=\s*['\"]([^'\"]+)['\"]", src)
    assert m, 'run_hil.py logging format not found'
    prefix = m.group(1).replace('%(asctime)s', '02:22:15').replace('%(levelname)-8s', 'INFO    ') \
        .replace('%(levelname)s', 'INFO').replace('%(name)s', 'hil.main')
    line = prefix.replace('%(message)s', fill(one('run_hil.py', '#phase,'), ['ARM', 5]))
    assert hd._RE_PHASE.search(line).group(1) == 'ARM'
