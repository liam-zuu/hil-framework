"""pc_sender.py --wait-swa (CODE_GAIN0 17/11; Liam 17/11 at the CODE session, hop chon "Them cho dung, GAIN0 sua
(Recommended)"; design 03_CODE/docs/2026-11-17_Thiet_ke_pc_sender_wait_swa_v1.md of the thesis repo).

On the car the ESP32 takes a gain set only with the motors off, while the schedule must start with SWA already on.
--wait-swa stops after the gain set and the question before, asks the operator to turn SWA on and press Enter, asks
the running set again (the firmware answers #gain_get with the motors on) and sends the schedule only when that
answer is the answer before. Without --wait-swa the --o1 path is the one of 9476500 (order, lines, exit codes)."""
import os
import subprocess
import sys

import pytest

import pc_sender as ps

RPI5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEFORE = ('#query_pc,t_start=10.000000,t_end=10.500000\n'
          '#gain_running,sha=5bde4c55bf3a,env=esp32,mac=aa:bb:cc:dd:ee:ff,boot=4753dc47,id=1,applied_n=1,ctrl=PID,'
          'kp_xy=1.5,ki_xy=0.0500000007,kp_wz=1,ki_wz=0.0500000007,pos_kp=6,pos_ki=3,k_psi=1\n#ket_qua,ma=0\n')
SWA = BEFORE.replace('t_start=10.000000,t_end=10.500000', 't_start=20.000000,t_end=20.400000')


class Rec:
    """Fakes of the steps of o1_main; every call is recorded in order."""
    def __init__(self, set_code=0, ack_bad=None, swa_bad=None, enter=True, eof=False):
        self.calls, self.lines = [], []
        self.set_code, self.ack_bad, self.swa_bad, self.enter, self.eof = set_code, ack_bad, swa_bad, enter, eof

    def out(self, s=''):
        self.lines.append(s)

    def set_gain(self, ip, f, path):
        self.calls.append(('set', os.path.basename(path)))
        return self.set_code

    def query(self, ip, path):
        self.calls.append(('query', os.path.basename(path)))
        return 0

    def ack_check(self, d):
        self.calls.append(('ack',))
        return self.ack_bad

    def swa_check(self, d):
        self.calls.append(('swa_check',))
        return self.swa_bad

    def inp(self, prompt=''):
        self.calls.append(('input',))
        if self.eof:
            raise EOFError
        if not self.enter:
            raise KeyboardInterrupt
        return ''

    def run_test(self, ip, o1, allow_final_set=False):
        self.calls.append(('test', o1, allow_final_set))
        return 1

    def kw(self):
        return dict(out=self.out, inp=self.inp, set_gain=self.set_gain, query=self.query,
                    ack_check=self.ack_check, run_test=self.run_test, swa_check=self.swa_check)


def no_input(prompt=''):
    raise AssertionError('input() called without --wait-swa')


# ── the path without --wait-swa is the path of 9476500 ──────────────────────────────────────────────────────────

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


# ── --wait-swa ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_wait_swa_order(tmp_path):
    r = Rec()
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, wait_swa=True, **r.kw()) == 0
    assert r.calls == [('set', 'gain_ack.txt'), ('query', 'gain_query_before.txt'), ('ack',), ('input',),
                       ('query', ps.SWA_QUERY_NAME), ('swa_check',), ('test', 'O1-F05', False),
                       ('query', 'gain_query_after.txt')]
    txt = '\n'.join(r.lines)
    assert '#wait_swa,cho,' in txt and 'Bat SWA' in txt and '#wait_swa,enter,' in txt
    assert '#gain_query,when=swa,exit=0' in txt
    # the prompt is printed after the ack check and before the wait
    assert r.lines.index('#gain_query,when=before,exit=0') < [i for i, s in enumerate(r.lines) if 'Bat SWA' in s][0]


@pytest.mark.parametrize('how', ['ctrl_c', 'eof'])
def test_wait_swa_abort_sends_no_schedule(tmp_path, how):
    r = Rec(enter=False) if how == 'ctrl_c' else Rec(eof=True)
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, wait_swa=True, **r.kw()) == 2
    assert r.calls == [('set', 'gain_ack.txt'), ('query', 'gain_query_before.txt'), ('ack',), ('input',)]
    assert any(s.startswith('#wait_swa,huy,') for s in r.lines)


def test_wait_swa_answer_differs_sends_no_schedule(tmp_path):
    r = Rec(swa_bad='boot truoc 4753dc47 khac boot sau 0badf00d')
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, wait_swa=True, **r.kw()) == 2
    assert ('test', 'O1-F05', False) not in r.calls and r.calls[-1] == ('swa_check',)
    assert r.lines[-1].endswith(': no task sent') and 'boot truoc' in r.lines[-1]


def test_wait_swa_before_anything_wrong_does_not_ask_swa(tmp_path):
    r = Rec(set_code=3)
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, wait_swa=True, **r.kw()) == 2
    assert ('input',) not in r.calls
    r = Rec(ack_bad='x')
    assert ps.o1_main('1.2.3.4', 'O1-F05', str(tmp_path), 'g.json', False, wait_swa=True, **r.kw()) == 2
    assert ('input',) not in r.calls


def test_wait_swa_without_query_dir_only_waits(tmp_path):
    r = Rec()
    assert ps.o1_main('1.2.3.4', 'O1-F02', None, None, False, wait_swa=True, **r.kw()) == 0
    assert r.calls == [('input',), ('test', 'O1-F02', False)]


def test_wait_swa_checks_the_schedule_before_asking():
    r = Rec()
    with pytest.raises(ValueError):
        ps.o1_main('1.2.3.4', 'O1-K1', None, None, False, wait_swa=True, **r.kw())
    assert r.calls == []


# ── swa_query_problem on real files ─────────────────────────────────────────────────────────────────────────────

def _write(d, before=BEFORE, swa=SWA):
    if before is not None:
        (d / 'gain_query_before.txt').write_text(before, encoding='utf-8')
    if swa is not None:
        (d / ps.SWA_QUERY_NAME).write_text(swa, encoding='utf-8')


def test_swa_query_problem_same_answer(tmp_path):
    _write(tmp_path)
    assert ps.swa_query_problem(str(tmp_path)) is None


@pytest.mark.parametrize('old,new,why', [
    ('boot=4753dc47', 'boot=0badf00d', 'boot'),                    # ESP32 restarted while SWA went on
    ('mac=aa:bb:cc:dd:ee:ff', 'mac=11:22:33:44:55:66', 'mac'),     # another board
    (',id=1,', ',id=2,', 'id'),                                    # another set runs
    ('applied_n=1', 'applied_n=2', 'applied_n'),                   # a set applied again in between
    ('kp_xy=1.5', 'kp_xy=1.4', 'kp_xy'),
    ('ctrl=PID', 'ctrl=ADRC', 'ctrl'),
])
def test_swa_query_problem_differs(tmp_path, old, new, why):
    _write(tmp_path, swa=SWA.replace(old, new))
    bad = ps.swa_query_problem(str(tmp_path))
    assert bad and why in bad


def test_swa_query_problem_missing_or_unreadable(tmp_path):
    _write(tmp_path, swa=None)
    assert 'gain_query_swa.txt' in ps.swa_query_problem(str(tmp_path))
    _write(tmp_path, swa=SWA.replace('#ket_qua,ma=0', '#ket_qua,ma=3'))
    assert 'gain_query_swa.txt' in ps.swa_query_problem(str(tmp_path))
    _write(tmp_path, before=None, swa=SWA)
    os.remove(tmp_path / 'gain_query_before.txt')
    assert 'gain_query_before.txt' in ps.swa_query_problem(str(tmp_path))


def test_swa_query_problem_order_of_questions(tmp_path):
    # the question after SWA must start after the question before ended (gain_line_check.query_problem)
    _write(tmp_path, swa=SWA.replace('t_start=20.000000', 't_start=10.200000'))
    assert ps.swa_query_problem(str(tmp_path))


def test_query_dir_refuses_an_old_swa_answer_only_with_wait_swa(tmp_path):
    (tmp_path / ps.SWA_QUERY_NAME).write_text(SWA, encoding='utf-8')
    assert ps.query_dir_problem(str(tmp_path)) is None
    assert ps.query_dir_problem(str(tmp_path), with_ack=True) is None
    assert ps.SWA_QUERY_NAME in ps.query_dir_problem(str(tmp_path), with_swa=True)


# ── command line ────────────────────────────────────────────────────────────────────────────────────────────────

def _cli(*args):
    return subprocess.run([sys.executable, os.path.join(RPI5, 'pc_sender.py'), '--ip', '127.0.0.1', *args],
                          capture_output=True, text=True, encoding='utf-8', stdin=subprocess.DEVNULL, timeout=60)


def test_cli_wait_swa_needs_o1():
    p = _cli('--wait-swa')
    assert p.returncode == 2 and '--wait-swa goes with --o1' in p.stderr


def test_cli_wait_swa_refuses_a_final_task_before_the_prompt():
    p = _cli('--o1', 'O1-K1', '--wait-swa')
    assert p.returncode == 2 and 'Bat SWA' not in p.stdout and '#wait_swa' not in p.stdout
    assert 'only with --allow-final-set' in p.stderr                  # refused for the schedule, not the flag


def test_cli_wait_swa_refuses_a_folder_with_an_old_swa_answer(tmp_path):
    (tmp_path / ps.SWA_QUERY_NAME).write_text(SWA, encoding='utf-8')
    p = _cli('--o1', 'O1-F02', '--gain-query-dir', str(tmp_path), '--wait-swa')
    assert p.returncode == 2 and ps.SWA_QUERY_NAME in p.stderr and 'already holds' in p.stderr
    assert '#gain_query' not in p.stdout and '#wait_swa' not in p.stdout


def test_cli_wait_swa_eof_sends_nothing():
    p = _cli('--o1', 'O1-F02', '--wait-swa')                   # stdin closed: EOF at the wait
    assert p.returncode == 2
    assert '#wait_swa,huy,' in p.stdout and '#test_start' not in p.stdout
