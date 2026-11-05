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
    """Lines before STOP; after the script writes STOP, the after_stop lines."""
    def __init__(self, lines, after_stop=()):
        self.lines = [l.encode() + b'\n' for l in lines]
        self.after = [l.encode() + b'\n' for l in after_stop]
        self.rts = False
        self.written = b''

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.written += data
        if b'STOP' in data:
            self.lines += self.after
            self.after = []

    def readline(self):
        return self.lines.pop(0) if self.lines else b''


def fast(monkeypatch):
    monkeypatch.setattr(g2.time, 'sleep', lambda s: None)
    monkeypatch.setattr(g2, 'STOP_MARGIN_S', 0.0)


def test_ready_line_with_init_ms(monkeypatch):
    fast(monkeypatch)
    s = FakeSerial(['boot rom junk', 'IMU READY v3 period_us=50000 t_us=700000 init_ms=812',
                    'A,1,1,0.2,-0.1,9.81,0', 'not a report'], ['END,800000,1,0,0,0,H'])
    ready, t_ready, rr, bad, end = g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.05, end_timeout=0.5)
    assert ready is not None and t_ready == 700000 and len(rr) == 1 and bad == 1
    assert s.written == b'STOP\n'
    assert end == {'t_us': 800000, 'A': 1, 'G': 0, 'R': 0, 'qdrop': 0, 'via': 'H'}


def test_ready_period_mismatch_stops(monkeypatch):
    fast(monkeypatch)
    s = FakeSerial(['IMU READY v3 period_us=10000 t_us=5'])
    with pytest.raises(SystemExit):
        g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.05)


def test_ready_without_t_us_stops(monkeypatch):
    """Old test_imu_log (v2) has no t_us: the window cannot be cut on the ESP32 clock."""
    fast(monkeypatch)
    s = FakeSerial(['IMU READY v2 period_us=50000'])
    with pytest.raises(SystemExit, match='fix/g2-imu-log'):
        g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.05)


def test_init_fail_line(monkeypatch, capsys):
    fast(monkeypatch)
    s = FakeSerial(['IMU INIT FAIL reason=no_chip'])
    ready, t_ready, rr, bad, end = g2.one_boot(s, PERIOD, ready_timeout=0.2, seconds=0.05)
    assert ready is None and rr == [] and end is None
    assert 'IMU INIT FAIL' in capsys.readouterr().out


def test_no_end_line_after_stop(monkeypatch):
    fast(monkeypatch)
    s = FakeSerial(['IMU READY v3 period_us=50000 t_us=0', 'A,1,1,0.2,-0.1,9.81,0'])
    ready, t_ready, rr, bad, end = g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.0, end_timeout=0.1)
    assert end is None and len(rr) == 1


def test_bad_end_line_is_a_bad_line(monkeypatch):
    fast(monkeypatch)
    s = FakeSerial(['IMU READY v3 period_us=50000 t_us=0'], ['END,1,2,3', 'END,9,0,0,0,0,H'])
    ready, t_ready, rr, bad, end = g2.one_boot(s, PERIOD, ready_timeout=1.0, seconds=0.0, end_timeout=0.5)
    assert bad == 1 and end['t_us'] == 9


# ---- window [t_READY; t_READY + D] and END line (spec S06 v3.4 Z18, L7-20, V2-10) ----

N = 20
D = N * PERIOD / 1e6                       # rows(N) span exactly one window of D s from t = 0


def end_for(rr, qdrop=0, via='H'):
    last = {k: max([r[2] for r in rr if r[0] == k], default=0) for k in 'AGR'}
    return {'t_us': 2_000_000, 'A': last['A'], 'G': last['G'], 'R': last['R'], 'qdrop': qdrop, 'via': via}


def wboot(rr, end='auto', t_ready=0, seconds=D, exp=EXP, bad=0):
    if end == 'auto':
        end = end_for(rr)
    b = {'boot': 1, 'ready_s': 0.8, 't_ready_us': t_ready, 'bad_lines': bad}
    b.update(g2.check_boot(rr, PERIOD, exp, t_ready, seconds, end))
    return b


def test_window_clean_boot_passes():
    b = wboot(rows(N))
    assert b['A_n_min'] == 20 and b['A_tail_gap_us'] == 1_000_000 - 950_000
    assert g2.fail_reasons(b, EXP, PERIOD) == []


def test_window_missing_end_fails():
    why = g2.fail_reasons(wboot(rows(N), end=None), EXP, PERIOD)
    assert 'no END line' in why


def test_window_end_qdrop_fails():
    why = g2.fail_reasons(wboot(rows(N), end=end_for(rows(N), qdrop=2)), EXP, PERIOD)
    assert 'END qdrop=2' in why


def test_window_tail_loss_caught_by_end_counter():
    """The last reports never reach the PC: no count gap, first n = 1, only END shows it."""
    rr = rows(N)
    e = end_for(rr)
    e['R'] += 2                            # source counted 2 more game RV than arrived
    why = g2.fail_reasons(wboot(rr, end=e), EXP, PERIOD)
    assert 'R last n=20 END n=22' in why


def test_reports_before_ready_not_judged_but_counted():
    """Reports before t_READY: out of the window (period, values, counts) but n still from 1."""
    pre = rows(5)
    for r in pre:
        r[3] += 100.0                      # wrong values before READY must not fail the boot
    body = [[k, t + 5 * PERIOD, n + 5, *v, q] for k, t, n, *v, q in rows(N)]
    rr = pre + body
    b = wboot(rr, t_ready=5 * PERIOD)
    assert b['A_n'] == N and b['A_n_all'] == N + 5
    assert g2.fail_reasons(b, EXP, PERIOD) == []
    b2 = wboot(body, t_ready=5 * PERIOD)     # same window, first 5 lost on the way
    assert 'A first n=6' in g2.fail_reasons(b2, EXP, PERIOD)


def test_window_count_and_tail_gap():
    short = [r for r in rows(N) if r[1] < 15 * PERIOD]   # reports stop 0.25 s before the end
    b = wboot(short)
    why = g2.fail_reasons(b, EXP, PERIOD)
    assert 'A n=15<20' in why
    assert any(w.startswith('A tail gap') for w in why)


def test_tail_gap_limit_is_two_periods():
    rr = [r for r in rows(N) if r[2] <= 18]               # last at 17*P+300j: gap about 3 periods
    b = wboot(rr, seconds=(N - 1) * PERIOD / 1e6)          # window ends at 19 P: gap = 2 P - 300j
    assert b['A_tail_gap_us'] == 2 * PERIOD
    assert not any('tail gap' in w for w in g2.fail_reasons(b, EXP, PERIOD))
    b = wboot(rr, seconds=(N - 1) * PERIOD / 1e6 + 1e-6)
    assert any(w.startswith('A tail gap') for w in g2.fail_reasons(b, EXP, PERIOD))


def test_boot_only_with_end_and_no_reports():
    b = wboot([], exp=None)
    assert g2.fail_reasons(b, None, PERIOD) == []
    b = wboot([], end=None, exp=None)
    assert g2.fail_reasons(b, None, PERIOD) == ['no END line']
