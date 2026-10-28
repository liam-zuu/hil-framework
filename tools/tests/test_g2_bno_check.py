"""Tests for tools/g2_bno_check.py PASS rule (no hardware).

Run: python -m pytest tools/tests/test_g2_bno_check.py -q
Rows are synthetic report lines as parse_line() returns them: [kind, t_us, n, v0, v1, v2, qdrop].
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import g2_bno_check as g2  # noqa: E402

PERIOD = 50000
EXPECT = (0.5, 0.2, -0.1, 9.81, 1.0)
EXP = g2.expected(EXPECT)
VALUES = {'A': (0.2, -0.1, 9.81), 'G': (0.0, 0.0, 0.5), 'R': (0.0, 0.0, 57.29578)}


def rows(n=20, qdrop=0, skip=None, start=1):
    """n reports per sensor at the exact period; skip = (kind, count) dropped by the host."""
    out = []
    for i in range(n):
        for j, kind in enumerate('AGR'):
            cnt = start + i
            if skip and skip == (kind, cnt):
                continue
            out.append([kind, i * PERIOD + 300 * j, cnt, *VALUES[kind], qdrop])
    return out


def boot(rr, bad=0, ready=0.8, exp=EXP):
    b = {'boot': 1, 'ready_s': ready, 'bad_lines': bad}
    b.update(g2.check_boot(rr, PERIOD, exp))
    return b


def test_clean_boot_passes():
    b = boot(rows())
    assert g2.fail_reasons(b, EXP) == []
    assert g2.boot_pass(b, EXP)


@pytest.mark.parametrize('case, rr, bad, reason', [
    ('qdrop', rows(qdrop=3), 0, 'qdrop=3'),
    ('bad line', rows(), 2, '2 bad lines'),
    ('count gap', rows(skip=('G', 7)), 0, 'G count gaps=1'),
    ('lost before window', rows(start=4), 0, 'A first n=4'),
])
def test_host_side_loss_fails(case, rr, bad, reason):
    b = boot(rr, bad=bad)
    why = g2.fail_reasons(b, EXP)
    assert reason in why, (case, why)
    assert not g2.boot_pass(b, EXP)


def test_qdrop_taken_as_max_not_last():
    rr = rows()
    rr[10][6] = 1                       # a counter glitch in the middle must not hide
    assert boot(rr)['qdrop'] == 1


def test_boot_only_check():
    assert g2.boot_pass(boot([], exp=None), None)
    assert not g2.boot_pass(boot([], bad=1, exp=None), None)
    assert g2.fail_reasons(boot([], ready='', exp=None), None) == ['no IMU READY']


def test_value_and_period_rules_unchanged():
    rr = rows()
    rr[3][3] += 0.1                     # rows go A, G, R, A...: second accel x far off
    assert 'A value' in g2.fail_reasons(boot(rr), EXP)
    slow = [[k, int(t * 1.05), n, *v, q] for k, t, n, *v, q in rows()]
    assert any('period' in w for w in g2.fail_reasons(boot(slow), EXP))


class FakeSerial:
    def __init__(self, lines):
        self.lines = [l.encode() + b'\n' for l in lines]
        self.rts = False

    def reset_input_buffer(self):
        pass

    def readline(self):
        return self.lines.pop(0) if self.lines else b''


def test_ready_line_with_init_ms(monkeypatch):
    monkeypatch.setattr(g2.time, 'sleep', lambda s: None)
    s = FakeSerial(['boot rom junk', 'IMU READY v2 period_us=50000 init_ms=812',
                    'A,1,1,0.2,-0.1,9.81,0', 'not a report'])
    ready, rr, bad = g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.05)
    assert ready is not None and len(rr) == 1 and bad == 1


def test_ready_period_mismatch_stops(monkeypatch):
    monkeypatch.setattr(g2.time, 'sleep', lambda s: None)
    s = FakeSerial(['IMU READY v2 period_us=10000'])
    with pytest.raises(SystemExit):
        g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.05)


def test_init_fail_line(monkeypatch, capsys):
    monkeypatch.setattr(g2.time, 'sleep', lambda s: None)
    s = FakeSerial(['IMU INIT FAIL reason=no_chip'])
    ready, rr, bad = g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.05)
    assert ready is None and rr == []
    assert 'IMU INIT FAIL' in capsys.readouterr().out
