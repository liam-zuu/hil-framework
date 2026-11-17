"""pc_sender.py --pause-swa (CODE_GAIN0 17/11; Liam 17/11 at the CODE session, hop chon "Them cho dung, GAIN0 sua
(Recommended)"; design 03_CODE/docs/2026-11-17_Thiet_ke_pc_sender_wait_swa_v1.md of the thesis repo; Codex Sol
round 1 WS-1..WS-7).

On the car the ESP32 takes a gain set only with the motors off, while the schedule must start with SWA already on.
--pause-swa stops after the gain set and the question before, asks the operator to turn SWA on and press Enter, asks
the running set again (the firmware answers #gain_get with the motors on) and sends the schedule only when that
answer is the answer before. Without --pause-swa the --o1 path is the one of 9476500 (order, lines, exit codes)."""
import os
import socket
import subprocess
import sys

import pytest

import pc_sender as ps

RPI5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PID_LINE = ('#gain_running,sha=5bde4c55bf3a,env=esp32,mac=aa:bb:cc:dd:ee:ff,boot=4753dc47,id=1,applied_n=1,ctrl=PID,'
            'kp_xy=1.5,ki_xy=0.0500000007,kp_wz=1,ki_wz=0.0500000007,pos_kp=6,pos_ki=3,k_psi=1')
ADRC_LINE = ('#gain_running,sha=5bde4c55bf3a,env=ladrc,mac=aa:bb:cc:dd:ee:ff,boot=4753dc47,id=1,applied_n=1,ctrl=ADRC,'
             'omega_c_xy=8,omega_o_xy=30,omega_c_wz=8,omega_o_wz=30,pos_kp=6,pos_ki=3,k_psi=1')


def answer(line=PID_LINE, t=(10.0, 10.5)):
    return f'#query_pc,t_start={t[0]:.6f},t_end={t[1]:.6f}\n{line}\n#ket_qua,ma=0\n'


BEFORE = answer()
SWA = answer(t=(20.0, 20.4))


class Rec:
    """Fakes of the steps of o1_main; every call is recorded in order (self.calls, shared with FakeSockets)."""
    def __init__(self, set_code=0, ack_bad=None, swa_bad=None, before_bad=None, lines=('',), files=None):
        self.calls, self.lines = [], []
        self.set_code, self.ack_bad, self.swa_bad, self.before_bad = set_code, ack_bad, swa_bad, before_bad
        self.inputs = list(lines)              # what input() returns, in order; 'EOF' / 'CTRL_C' raise
        self.files = files                     # {file name: text} written by query() (real-file flow)

    def out(self, s=''):
        self.lines.append(s)

    def set_gain(self, ip, f, path):
        self.calls.append(('set', os.path.basename(path)))
        return self.set_code

    def query(self, ip, path):
        self.calls.append(('query', os.path.basename(path)))
        if self.files is not None:
            with open(path, 'x', encoding='utf-8') as fh:
                fh.write(self.files[os.path.basename(path)])
        return 0

    def ack_check(self, d):
        self.calls.append(('ack',))
        return self.ack_bad

    def before_check(self, d):
        self.calls.append(('before_check',))
        return self.before_bad

    def swa_check(self, d):
        self.calls.append(('swa_check',))
        return self.swa_bad

    def inp(self, prompt=''):
        self.calls.append(('input',))
        x = self.inputs.pop(0)
        if x == 'EOF':
            raise EOFError
        if x == 'CTRL_C':
            raise KeyboardInterrupt
        return x

    def run_test(self, ip, o1, allow_final_set=False):
        self.calls.append(('test', o1, allow_final_set))
        return 1

    def kw(self, real=()):
        d = dict(out=self.out, inp=self.inp, set_gain=self.set_gain, query=self.query, ack_check=self.ack_check,
                 run_test=self.run_test, swa_check=self.swa_check, before_check=self.before_check)
        for k in real:                         # use the real step of pc_sender
            d.pop(k)
        return d


def no_input(prompt=''):
    raise AssertionError('input() called without --pause-swa')


# ── the path without --pause-swa is the path of 9476500 ─────────────────────────────────────────────────────────

def test_old_path_order_lines_and_exit(tmp_path):
    r = Rec()
    kw = r.kw()
    kw['inp'] = no_input
    code = ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, **kw)
    assert code == 0
    assert r.calls == [('set', 'gain_ack.txt'), ('query', 'gain_query_before.txt'), ('ack',),
                       ('test', 'O1-F05', False), ('query', 'gain_query_after.txt')]
    assert r.lines == ['#gain_set,exit=0', '#gain_query,when=before,exit=0', '#gain_query,when=after,exit=0']


def test_old_path_without_gain_file_and_without_query_dir(tmp_path):
    r = Rec()
    kw = r.kw()
    kw['inp'] = no_input
    assert ps.o1_main('1.2.3.4', 'O1-F02', str(tmp_path), None, False, **kw) == 0
    assert r.calls == [('query', 'gain_query_before.txt'), ('test', 'O1-F02', False),
                       ('query', 'gain_query_after.txt')]
    r = Rec()
    kw = r.kw()
    kw['inp'] = no_input
    assert ps.o1_main('1.2.3.4', 'O1-K1', None, None, True, **kw) == 0
    assert r.calls == [('test', 'O1-K1', True)] and r.lines == []


def test_old_path_gain_not_applied_or_ack_wrong_sends_nothing(tmp_path):
    r = Rec(set_code=2)
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, **r.kw()) == 2
    assert r.calls == [('set', 'gain_ack.txt')]
    assert r.lines == ['#gain_set,exit=2', 'gain g.json not applied (see gain_ack.txt): no task sent']
    r = Rec(ack_bad='boot khac')
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, **r.kw()) == 2
    assert r.calls == [('set', 'gain_ack.txt'), ('query', 'gain_query_before.txt'), ('ack',)]
    assert r.lines[-1] == 'boot khac: no task sent'


def test_old_path_a_failed_question_before_still_runs(tmp_path):
    """9476500 only prints the exit code of the question before (no --gain-file: no check); kept."""
    r = Rec()
    r.query = lambda ip, path: (r.calls.append(('query', os.path.basename(path))), 3)[1]
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), None, False, **r.kw()) == 0
    assert '#gain_query,when=before,exit=3' in r.lines and ('test', 'O1-F05', False) in r.calls


def test_only_a_schedule_error_becomes_schedule_error(tmp_path):
    """__main__ turns ScheduleError (and only it) into parser.error, as 9476500 did for test_loop's ValueError."""
    r = Rec()

    def bad_set(ip, f, path):
        raise ValueError('not a schedule error')
    kw = r.kw()
    kw['set_gain'] = bad_set
    with pytest.raises(ValueError) as e:
        ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, **kw)
    assert type(e.value) is ValueError
    kw = Rec().kw(real=('run_test',))
    with pytest.raises(ps.ScheduleError, match='only with --allow-final-set'):
        ps.o1_main('1.2.3.4', 'O1-K2', None, None, False, **kw)


# ── --pause-swa ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_pause_order(tmp_path):
    r = Rec()
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, pause_swa=True, **r.kw()) == 0
    assert r.calls == [('set', 'gain_ack.txt'), ('query', 'gain_query_before.txt'), ('ack',), ('before_check',),
                       ('input',), ('query', ps.SWA_QUERY_NAME), ('swa_check',), ('test', 'O1-F05', False),
                       ('query', 'gain_query_after.txt')]
    txt = '\n'.join(r.lines)
    assert '#pause_swa,cho,' in txt and 'Bat SWA' in txt and '#pause_swa,enter,' in txt and 'waited_s=' in txt
    assert '#gain_query,when=swa,exit=0' in txt
    assert r.lines.index('#gain_query,when=before,exit=0') < [i for i, s in enumerate(r.lines) if 'Bat SWA' in s][0]


@pytest.mark.parametrize('how', ['CTRL_C', 'EOF'])
def test_pause_abort_sends_no_schedule(tmp_path, how):
    r = Rec(lines=(how,))
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, pause_swa=True, **r.kw()) == 2
    assert r.calls[-1] == ('input',) and ('test', 'O1-F05', False) not in r.calls
    assert any(s.startswith('#pause_swa,huy,') for s in r.lines)


def test_pause_only_an_empty_line_goes_on(tmp_path):
    """WS-2: a typed line is not the Enter; the question is asked again."""
    r = Rec(lines=('abc', '  x ', '', ))
    assert ps.o1_main('1.2.3.4', 'O1-F05', None, None, False, pause_swa=True, **r.kw()) == 0
    assert r.calls == [('input',), ('input',), ('input',), ('test', 'O1-F05', False)]
    assert sum('Chi nhan Enter trong' in s for s in r.lines) == 2
    r = Rec(lines=('abc', 'EOF'))
    assert ps.o1_main('1.2.3.4', 'O1-F05', None, None, False, pause_swa=True, **r.kw()) == 2
    assert ('test', 'O1-F05', False) not in r.calls
    r = Rec(lines=('\r', ))                       # a bare CR (Windows console) is an empty line
    assert ps.o1_main('1.2.3.4', 'O1-F05', None, None, False, pause_swa=True, **r.kw()) == 0


def test_pause_answer_differs_sends_no_schedule(tmp_path):
    r = Rec(swa_bad='boot truoc 4753dc47 khac boot sau 0badf00d')
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, pause_swa=True, **r.kw()) == 2
    assert ('test', 'O1-F05', False) not in r.calls and r.calls[-1] == ('swa_check',)
    assert r.lines[-1].endswith(': no task sent') and 'boot truoc' in r.lines[-1]


def test_pause_nothing_asked_when_an_earlier_step_failed(tmp_path):
    for r in (Rec(set_code=3), Rec(ack_bad='x'), Rec(before_bad='no readable answer (gain_query_before.txt)')):
        assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, pause_swa=True, **r.kw()) == 2
        assert ('input',) not in r.calls and not any('Bat SWA' in s for s in r.lines)


def test_pause_without_query_dir_only_waits(tmp_path):
    r = Rec()
    assert ps.o1_main('1.2.3.4', 'O1-F02', None, None, False, pause_swa=True, **r.kw()) == 0
    assert r.calls == [('input',), ('test', 'O1-F02', False)]


@pytest.mark.parametrize('task', ['O1-K1', 'O1-K2'])
def test_pause_checks_the_schedule_before_anything(tmp_path, task):
    r = Rec()
    with pytest.raises(ps.ScheduleError):
        ps.o1_main('1.2.3.4', task, str(tmp_path), 'g.json', False, pause_swa=True, **r.kw())
    assert r.calls == [] and r.lines == []


# ── every datagram is seen (WS-7): none before the Enter, none after Ctrl-C / EOF ───────────────────────────────

class FakeSockets:
    """Replaces socket.socket: every sendto is an event in the shared call list."""
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, *a, **k):
        calls = self.calls

        class S:
            def sendto(self, pkt, addr):
                calls.append(('send', addr[1]))

            def close(self):
                pass

            def settimeout(self, t):
                pass
        return S()


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 0.0003
        return self.t

    def sleep(self, d):
        self.t += max(d, 0.0)


def _real_loop(ip, o1, allow_final_set=False):
    c = FakeClock()
    return ps.test_loop(ip, o1, allow_final_set=allow_final_set, clock=c, sleep=c.sleep, out=lambda s: None)


@pytest.mark.parametrize('lines,sent', [(('',), True), (('CTRL_C',), False), (('EOF',), False),
                                        (('abc', 'EOF'), False)])
def test_no_datagram_before_the_enter(tmp_path, monkeypatch, lines, sent):
    r = Rec(lines=lines)
    monkeypatch.setattr(ps.socket, 'socket', FakeSockets(r.calls))
    kw = r.kw()
    kw['run_test'] = _real_loop
    code = ps.o1_main('1.2.3.4', 'O1-F02', str(tmp_path), 'g.json', False, pause_swa=True, **kw)
    sends = [i for i, c in enumerate(r.calls) if c[0] == 'send']
    last_input = max(i for i, c in enumerate(r.calls) if c == ('input',))
    if sent:
        assert code == 0 and sends and min(sends) > last_input
        assert {r.calls[i][1] for i in sends} == {ps.TRAJ_PORT}
    else:
        assert code == 2 and sends == []


# ── the real checks on the files the flow writes ────────────────────────────────────────────────────────────────

def test_real_file_flow_same_answer_sends(tmp_path):
    r = Rec(files={'gain_query_before.txt': BEFORE, ps.SWA_QUERY_NAME: SWA, 'gain_query_after.txt': SWA})
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), None, False, pause_swa=True,
                      **r.kw(real=('swa_check', 'before_check'))) == 0
    assert ('test', 'O1-F05', False) in r.calls


@pytest.mark.parametrize('swa', [answer(PID_LINE.replace('kp_wz=1,', 'kp_wz=1.2,'), t=(20, 20.4)),
                                 answer(PID_LINE.replace('boot=4753dc47', 'boot=0badf00d'), t=(20, 20.4)),
                                 '#ket_qua,ma=3\n'])
def test_real_file_flow_other_answer_stops(tmp_path, swa):
    r = Rec(files={'gain_query_before.txt': BEFORE, ps.SWA_QUERY_NAME: swa})
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), None, False, pause_swa=True,
                      **r.kw(real=('swa_check', 'before_check'))) == 2
    assert ('test', 'O1-F05', False) not in r.calls


def test_real_file_flow_bad_before_asks_no_swa(tmp_path):
    r = Rec(files={'gain_query_before.txt': '#query_pc,t_start=1,t_end=2\n#ket_qua,ma=3\n'})
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), None, False, pause_swa=True,
                      **r.kw(real=('swa_check', 'before_check'))) == 2
    assert ('input',) not in r.calls


# ── swa_query_problem / answer_problem on real files ────────────────────────────────────────────────────────────

def _write(d, before=BEFORE, swa=SWA):
    for name, text in (('gain_query_before.txt', before), (ps.SWA_QUERY_NAME, swa)):
        p = d / name
        if p.exists():
            p.unlink()
        if text is not None:
            p.write_text(text, encoding='utf-8')


@pytest.mark.parametrize('line', [PID_LINE, ADRC_LINE])
def test_swa_query_problem_same_answer(tmp_path, line):
    _write(tmp_path, answer(line), answer(line, t=(20, 20.4)))
    assert ps.swa_query_problem(str(tmp_path)) is None


def _fields(line):
    return [p.split('=', 1) for p in line.split(',')[1:]]


@pytest.mark.parametrize('line', [PID_LINE, ADRC_LINE])
def test_swa_query_problem_any_field_differs(tmp_path, line):
    """Every field of '#gain_running' (each of the 7 gains, sha, env, mac, boot, id, applied_n) is compared."""
    for k, v in _fields(line):
        if k == 'ctrl':
            continue
        new = '0badf00d' if k == 'boot' else ('11:22:33:44:55:66' if k == 'mac' else
                                              ('9' if k in ('id', 'applied_n') else
                                               ('x' + v if k in ('sha', 'env') else repr(float(v) + 0.5))))
        other = line.replace(f',{k}={v}', f',{k}={new}', 1)
        assert other != line, k
        _write(tmp_path, answer(line), answer(other, t=(20, 20.4)))
        bad = ps.swa_query_problem(str(tmp_path))
        assert bad and (k in bad or (k == 'boot' and 'boot' in bad)), (k, bad)


def test_swa_query_problem_other_ctrl(tmp_path):
    _write(tmp_path, answer(PID_LINE), answer(ADRC_LINE, t=(20, 20.4)))
    assert 'ctrl' in ps.swa_query_problem(str(tmp_path))


@pytest.mark.parametrize('drop', ['mac', 'id', 'applied_n', 'k_psi', 'kp_xy'])
def test_answer_missing_the_same_field_on_both_sides(tmp_path, drop):
    """WS-4: two answers that both lack a field are not 'equal answers'."""
    line = ','.join(p for p in PID_LINE.split(',') if not p.startswith(drop + '='))
    _write(tmp_path, answer(line), answer(line, t=(20, 20.4)))
    bad = ps.swa_query_problem(str(tmp_path))
    assert bad and 'incomplete answer' in bad and drop in bad


@pytest.mark.parametrize('old,new', [('mac=aa:bb:cc:dd:ee:ff', 'mac='), ('k_psi=1', 'k_psi=nan'),
                                     ('k_psi=1', 'k_psi=abc'), (',id=1,', ',id=x,'), ('ctrl=PID', 'ctrl=LQR')])
def test_answer_with_an_empty_or_bad_field(tmp_path, old, new):
    line = PID_LINE.replace(old, new)
    _write(tmp_path, answer(line), answer(line, t=(20, 20.4)))
    assert 'incomplete answer' in ps.swa_query_problem(str(tmp_path))
    assert 'incomplete answer' in ps.before_problem(str(tmp_path))


def test_swa_query_problem_missing_or_unreadable(tmp_path):
    _write(tmp_path, swa=None)
    assert ps.SWA_QUERY_NAME in ps.swa_query_problem(str(tmp_path))
    _write(tmp_path, swa=SWA.replace('#ket_qua,ma=0', '#ket_qua,ma=3'))
    assert ps.SWA_QUERY_NAME in ps.swa_query_problem(str(tmp_path))
    _write(tmp_path, before=None)
    assert 'gain_query_before.txt' in ps.swa_query_problem(str(tmp_path))
    assert 'gain_query_before.txt' in ps.before_problem(str(tmp_path))


def test_swa_query_problem_order_of_questions(tmp_path):
    _write(tmp_path, swa=answer(t=(10.2, 10.6)))       # the question after SWA started before 'before' ended
    assert ps.swa_query_problem(str(tmp_path))


def test_query_dir_refuses_an_old_swa_answer_only_with_pause(tmp_path):
    (tmp_path / ps.SWA_QUERY_NAME).write_text(SWA, encoding='utf-8')
    assert ps.query_dir_problem(str(tmp_path)) is None
    assert ps.query_dir_problem(str(tmp_path), with_ack=True) is None
    assert ps.SWA_QUERY_NAME in ps.query_dir_problem(str(tmp_path), with_swa=True)


def test_query_dir_broken_link_with_pause(tmp_path):
    """WS-5: a broken link with the name of a run file counts with --pause-swa."""
    link = tmp_path / ps.SWA_QUERY_NAME
    try:
        os.symlink(str(tmp_path / 'gone' / 'x.txt'), str(link))
    except (OSError, NotImplementedError):
        pytest.skip('no symlink on this machine')
    assert not os.path.exists(link) and os.path.lexists(link)
    assert ps.SWA_QUERY_NAME in ps.query_dir_problem(str(tmp_path), with_swa=True)


def test_query_dir_uses_lexists_with_pause(tmp_path, monkeypatch):
    """WS-5 without a real symlink: with --pause-swa the name counts when only lexists sees it."""
    monkeypatch.setattr(ps.os.path, 'exists', lambda p: False)
    monkeypatch.setattr(ps.os.path, 'lexists', lambda p: os.path.basename(p) == ps.SWA_QUERY_NAME)
    assert ps.query_dir_problem(str(tmp_path)) is None                   # without the flag: exists, as 9476500
    assert ps.SWA_QUERY_NAME in ps.query_dir_problem(str(tmp_path), with_swa=True)


def test_main_lets_a_non_schedule_value_error_through(tmp_path, monkeypatch, capsys):
    """__main__ turns only ScheduleError into parser.error (9476500 wrapped only test_loop): a ValueError of the gain
    set step still ends the program with that ValueError, with and without --pause-swa."""
    import runpy
    gfile = tmp_path / 'gain_x.json'
    gfile.write_text('{}', encoding='utf-8')

    def boom(*a, **k):
        raise ValueError('not a schedule error')
    monkeypatch.setattr(subprocess, 'run', boom)
    for extra in ([], ['--pause-swa']):
        d = tmp_path / ('q' + str(len(extra)))
        monkeypatch.setattr(sys, 'argv', ['pc_sender.py', '--ip', '127.0.0.1', '--o1', 'O1-F02', '--gain-query-dir',
                                          str(d), '--gain-file', str(gfile)] + extra)
        with pytest.raises(ValueError, match='not a schedule error'):
            runpy.run_path(os.path.join(RPI5, 'pc_sender.py'), run_name='__main__')


# ── command line ────────────────────────────────────────────────────────────────────────────────────────────────

def _cli(*args, env=None):
    return subprocess.run([sys.executable, os.path.join(RPI5, 'pc_sender.py'), '--ip', '127.0.0.1', *args],
                          capture_output=True, text=True, encoding='utf-8', stdin=subprocess.DEVNULL, timeout=60,
                          env=env)


def test_cli_pause_needs_o1():
    p = _cli('--pause-swa')
    assert p.returncode == 2 and '--pause-swa goes with --o1' in p.stderr


@pytest.mark.parametrize('task', ['O1-K1', 'O1-K2'])
def test_cli_pause_refuses_a_final_task_before_the_prompt(task):
    p = _cli('--o1', task, '--pause-swa')
    assert p.returncode == 2 and 'Bat SWA' not in p.stdout and '#pause_swa' not in p.stdout
    assert 'only with --allow-final-set' in p.stderr                  # refused for the schedule, not the flag


def test_cli_old_abbreviation_still_parses():
    """WS-1: '--w' was '--wz' in 9476500; the new flag must not make it ambiguous."""
    p = _cli('--o1', 'O1-K1', '--w', '0')
    assert p.returncode == 2 and 'ambiguous' not in p.stderr and 'only with --allow-final-set' in p.stderr


def test_cli_pause_refuses_a_folder_with_an_old_swa_answer(tmp_path):
    (tmp_path / ps.SWA_QUERY_NAME).write_text(SWA, encoding='utf-8')
    p = _cli('--o1', 'O1-F02', '--gain-query-dir', str(tmp_path), '--pause-swa')
    assert p.returncode == 2 and ps.SWA_QUERY_NAME in p.stderr and 'already holds' in p.stderr
    assert '#gain_query' not in p.stdout and '#pause_swa' not in p.stdout


def test_cli_pause_prompt_survives_a_cp1252_pipe():
    env = dict(os.environ, PYTHONIOENCODING='cp1252')           # stdout a pipe in the Windows code page
    p = subprocess.run([sys.executable, os.path.join(RPI5, 'pc_sender.py'), '--ip', '127.0.0.1', '--o1', 'O1-F02',
                        '--pause-swa'], capture_output=True, stdin=subprocess.DEVNULL, env=env, timeout=60)
    assert b'UnicodeEncodeError' not in p.stderr and b'Bat SWA' in p.stdout and p.returncode == 2


def test_cli_pause_eof_sends_nothing():
    """stdin closed: EOF at the wait; a UDP receiver on the schedule port sees no datagram."""
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx.bind(('127.0.0.1', ps.TRAJ_PORT))
    except OSError:
        rx.close()
        pytest.skip(f'port {ps.TRAJ_PORT} busy')
    rx.settimeout(0.5)
    try:
        p = _cli('--o1', 'O1-F02', '--pause-swa')
        assert p.returncode == 2
        assert '#pause_swa,huy,' in p.stdout and '#test_start' not in p.stdout
        with pytest.raises(socket.timeout):
            rx.recvfrom(64)
    finally:
        rx.close()
