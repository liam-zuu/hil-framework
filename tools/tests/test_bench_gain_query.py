"""hil_bench_session --gain-query helpers (S05 4.7 (iv), R3V-9; R134 package, Liam 14/11 CODE_R134
"duyet goi sua code fix/r134-answers"; after Codex round 1 R134C-1, -9, -10, Liam 14/11 CODE_R134
"Sua + vong 2 (Recommended)"): gain_send --query output saved, answer read back, before = after = boot of
the run's '#us,ev=start'; question before the run's clock, question after before scp; analyze-only judges
the saved answers again."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import hil_bench_session as B  # noqa: E402
from test_hil_bench_session import _bench_env  # noqa: E402

ANSWER = ('#gain_running,sha=16702ff0e688,env=ladrc,mac=aa:bb:cc:dd:ee:ff,boot=4753dc47,id=0,applied_n=0,ctrl=ADRC,'
          'omega_c_xy=8,omega_o_xy=30,omega_c_wz=8,omega_o_wz=30,pos_kp=6,pos_ki=3,k_psi=1\n#ket_qua,ma=0\n')


def fake_run(out, calls):
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=out, stderr='', returncode=0)
    return run


def test_query_saved_and_answer_read(tmp_path):
    calls = []
    p = str(tmp_path / 'r1_gain_query_before.txt')
    q = B.gain_query('192.168.1.69', p, run=fake_run(ANSWER, calls))
    assert q['boot'] == '4753dc47' and q['mac'] == 'aa:bb:cc:dd:ee:ff'
    assert calls[0][1:] == [B.GAIN_SEND, '--ip', '192.168.1.69', '--query']
    first, rest = open(p, encoding='utf-8').read().split('\n', 1)
    assert first.startswith('#query_pc,t_start=') and rest == ANSWER                   # R134C2-1: PC times
    assert q['_t_start'] <= q['_t_end']
    with pytest.raises(FileExistsError):                       # never overwrites a record
        B.gain_query('192.168.1.69', p, run=fake_run(ANSWER, calls))


@pytest.mark.parametrize('out', ['# no answer\n#ket_qua,ma=3\n', ANSWER.replace('#ket_qua,ma=0', '#ket_qua,ma=4'),
                                 ANSWER + ANSWER.splitlines()[0] + '\n',
                                 ANSWER + '#ket_qua,ma=3\n'])                # R134C-1: the output must END with ma=0
def test_no_answer_without_a_clean_single_answer(tmp_path, out):
    assert B.gain_query('1.2.3.4', str(tmp_path / 'q.txt'), run=fake_run(out, [])) is None


def test_no_ip_and_timeout(tmp_path):
    assert B.gain_query(None, str(tmp_path / 'a.txt'), run=fake_run(ANSWER, [])) is None
    assert 'no ESP32 IP' in open(tmp_path / 'a.txt', encoding='utf-8').read()

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get('timeout'))
    assert B.gain_query('1.2.3.4', str(tmp_path / 'b.txt'), run=slow) is None


def _q(boot, mac='aa:bb', t=(1.0, 2.0)):
    return dict(boot=boot, mac=mac, _t_start=t[0], _t_end=t[1])


def write_run_log(out, name, boot='4753dc47'):
    """pi/<name>.csv of one run (off, on, on, off) and its events with the run's '#us,ev=start'."""
    os.makedirs(os.path.join(out, 'pi'), exist_ok=True)
    p = os.path.join(out, 'pi', f'{name}.csv')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('timestamp_us,log_seq,motor_on\n1000,9,0\n2000,10,1\n3000,11,1\n4000,12,0\n')
    with open(p + '.events.txt', 'w', encoding='utf-8') as f:
        f.write(f'2026-11-14T10:00:00.000001 #us,ev=start,boot={boot},on=0,swb=1500,first_seq=10,line_drop=0\n')
    return p


A = (3.0, 4.0)


@pytest.mark.parametrize('before, after, why', [(_q('ab'), _q('ab', t=A), None), (None, _q('ab', t=A), 'truoc'),
                                                (_q('ab'), None, 'sau'), (_q('ab'), _q('AB', t=A), None),
                                                (_q('ab'), _q('cd', t=A), 'khac boot'),
                                                (_q('ab', 'm1'), _q('ab', 'm2', t=A), 'hai board'),
                                                (_q('ab', t=A), _q('ab'), 'khong phai truoc va sau')])
def test_boot_problem_between_the_two_answers(before, after, why):
    w = B.boot_problem(before, after)
    assert (w is None) if why is None else (why in w)


def test_boot_problem_against_the_run_boot(tmp_path):
    """R134C-1: two equal answers of an old boot (or another board) are not the run's boot."""
    out = str(tmp_path)
    write_run_log(out, 'x', boot='4753dc47')
    assert B.boot_problem(_q('4753dc47'), _q('4753dc47', t=A), out, 'x') is None
    assert 'khac boot cua luot' in B.boot_problem(_q('0badf00d'), _q('0badf00d', t=A), out, 'x')
    assert 'khong co boot' in B.boot_problem(_q('4753dc47'), _q('4753dc47', t=A), out, 'missing')


def test_question_before_the_run_clock_and_after_before_scp(tmp_path, monkeypatch):
    """R134C-9: the --pi-stop timer starts after the question before the run (which may take seconds);
    the question after is sent as soon as the run's ssh returns, before any scp."""
    order = []

    def reply(c):
        if 'run_hil.py' in c and 'finite-run' in c:
            order.append('run')
            return (0, 'ok')
        if c.startswith('P=$(cat'):
            order.append('stop')
            return (0, 'pid=1')
        return (0, '## HEAD\nabc') if 'git' in c else (1, '')
    a, calls, timers = _bench_env(monkeypatch, tmp_path, reply, run=['r=--test A-L1'], pi_stop=['r=1'],
                                  gain_query=True, esp32_ip='1.2.3.4')

    def fake_query(ip, path, run=None):
        order.append('q_' + ('before' if path.endswith('before.txt') else 'after'))
        B.time.sleep(3.0)                                             # a slow answer (fake clock)
        with open(path, 'x', encoding='utf-8') as f:
            f.write(ANSWER)
        return dict(boot='4753dc47', mac='aa')
    monkeypatch.setattr(B, 'gain_query', fake_query)
    monkeypatch.setattr(B, 'scp', lambda *x: order.append('scp') or 0)
    fake_timer = B.threading.Timer                                    # the FakeTimer of _bench_env: fires at start

    class OrderTimer(fake_timer):
        def start(self):
            order.append('timer')
            super().start()
    monkeypatch.setattr(B.threading, 'Timer', OrderTimer)
    B.bench(a)
    assert order[:6] == ['q_before', 'timer', 'stop', 'run', 'q_after', 'scp'], order
    s = json.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))
    rec = s['runs'][0]
    qb, qa = rec['boot']['before'], rec['boot']['after']
    assert qb['t_end'] <= rec['t0'] and rec['t1'] <= qa['t_start'] and qb['mac'] == 'aa' and qa['boot'] == '4753dc47'


def test_analyze_only_judges_the_saved_answers_again(tmp_path, monkeypatch):
    """R134C-10: a session made with --gain-query is judged again from its saved answers; a session made
    without it keeps its old verdict."""
    out = str(tmp_path)
    os.makedirs(os.path.join(out, 'frames'))
    for n in ('x', 'y'):
        open(os.path.join(out, 'frames', f'{n}_frames.npz'), 'w').close()
        write_run_log(out, n)
    for w, boot, t in (('before', '4753dc47', '1,2'), ('after', '0badf00d', '3,4')):
        with open(os.path.join(out, 'pi', f'x_gain_query_{w}.txt'), 'w', encoding='utf-8') as f:
            a, b = t.split(',')
            f.write(f'#query_pc,t_start={a},t_end={b}\n' + ANSWER.replace('4753dc47', boot))
    with open(os.path.join(out, 'session.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(runs=[dict(name='x', boot=dict(before=dict(boot='4753dc47'), after=dict(boot='0badf00d'))),
                             dict(name='y')]), f)
    monkeypatch.setattr(B, 'analyze_run', lambda o, n, *x: dict(name=n, clean=True, why=[]))
    monkeypatch.setattr(B, 'summary_line', lambda r: r['name'])
    code, res = B.analyze_only(SimpleNamespace(out=out))
    by = {r['name']: r for r in res}
    assert not by['x']['clean'] and any('khac boot sau' in w for w in by['x']['why'])
    assert by['y']['clean'] and code == 1


def test_gain_query_needs_the_udp_log(monkeypatch, capsys):
    """R134C-1: the run's boot is read from its UDP log, so --gain-query refuses --no-logger."""
    monkeypatch.setattr(B, 'GAIN_SEND', __file__)                    # gain_send.py is on feat/wifi-gain only
    with pytest.raises(SystemExit):
        B.main(['--out', 'x', '--run', 'a=b', '--gain-query', '--no-logger'])
    assert 'no-logger' in capsys.readouterr().err


def test_stop_timer_cancelled_before_the_question_after(tmp_path, monkeypatch):
    """R134C2-5 (Codex example): --pi-stop r=11, the run's ssh returns at 10 s, the question after takes 3 s:
    the stop must not be sent (the run ended first). Timer driven by the fake clock."""
    order, timers = [], []

    def reply(c):
        if 'run_hil.py' in c and 'finite-run' in c:
            B.time.sleep(10.0)                                        # the run lasts 10 s
            order.append('run_end')
            return (0, 'ok')
        if c.startswith('P=$(cat'):
            order.append('stop')
            return (0, 'pid=1')
        return (0, '## HEAD\nabc') if 'git' in c else (1, '')
    a, calls, _ = _bench_env(monkeypatch, tmp_path, reply, run=['r=--test A-L1'], pi_stop=['r=11'],
                             gain_query=True, esp32_ip='1.2.3.4')
    clock = dict(t=B.time.time())

    class ClockTimer:
        def __init__(self, sec, fn):
            self.sec, self.fn, self.daemon, self.t0, self.alive = sec, fn, False, None, False
            timers.append(self)

        def start(self):
            self.t0, self.alive = clock['t'], True

        def cancel(self):
            self.alive = False

        def join(self, t=None):
            pass

    def sleep(s):
        clock['t'] += s
        for tm in timers:
            if tm.alive and clock['t'] >= tm.t0 + tm.sec:
                tm.alive = False
                tm.fn()
    monkeypatch.setattr(B, 'time', SimpleNamespace(time=lambda: clock['t'], sleep=sleep))
    monkeypatch.setattr(B.threading, 'Timer', ClockTimer)

    def fake_query(ip, path, run=None):
        order.append('q_' + ('before' if path.endswith('before.txt') else 'after'))
        B.time.sleep(3.0)
        with open(path, 'x', encoding='utf-8') as f:
            f.write(ANSWER)
        return dict(boot='4753dc47', mac='aa')
    monkeypatch.setattr(B, 'gain_query', fake_query)
    B.bench(a)
    assert 'stop' not in order and order[:3] == ['q_before', 'run_end', 'q_after'], order
    s = json.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))
    assert 'không gửi' in s['runs'][0]['pi_stop']['note']
