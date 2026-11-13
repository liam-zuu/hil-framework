"""hil_bench_session --gain-query helpers (S05 4.7 (iv), R3V-9; R134 package, Liam 14/11 CODE_R134
"duyet goi sua code fix/r134-answers"): gain_send --query output saved, boot read back, before = after."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import hil_bench_session as B  # noqa: E402

ANSWER = ('#gain_running,sha=16702ff0e688,env=ladrc,mac=aa:bb:cc:dd:ee:ff,boot=4753dc47,id=0,applied_n=0,ctrl=ADRC,'
          'omega_c_xy=8,omega_o_xy=30,omega_c_wz=8,omega_o_wz=30,pos_kp=6,pos_ki=3,k_psi=1\n#ket_qua,ma=0\n')


def fake_run(out, calls):
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=out, stderr='', returncode=0)
    return run


def test_query_saved_and_boot_read(tmp_path):
    calls = []
    p = str(tmp_path / 'r1_gain_query_before.txt')
    assert B.gain_query('192.168.1.69', p, run=fake_run(ANSWER, calls)) == '4753dc47'
    assert calls[0][1:] == [B.GAIN_SEND, '--ip', '192.168.1.69', '--query']
    assert open(p, encoding='utf-8').read() == ANSWER
    with pytest.raises(FileExistsError):                       # never overwrites a record
        B.gain_query('192.168.1.69', p, run=fake_run(ANSWER, calls))


@pytest.mark.parametrize('out', ['# no answer\n#ket_qua,ma=3\n', ANSWER.replace('#ket_qua,ma=0', '#ket_qua,ma=4'),
                                 ANSWER + ANSWER.splitlines()[0] + '\n'])
def test_no_boot_without_a_clean_single_answer(tmp_path, out):
    assert B.gain_query('1.2.3.4', str(tmp_path / 'q.txt'), run=fake_run(out, [])) is None


def test_no_ip_and_timeout(tmp_path):
    assert B.gain_query(None, str(tmp_path / 'a.txt'), run=fake_run(ANSWER, [])) is None
    assert 'no ESP32 IP' in open(tmp_path / 'a.txt', encoding='utf-8').read()

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get('timeout'))
    assert B.gain_query('1.2.3.4', str(tmp_path / 'b.txt'), run=slow) is None


@pytest.mark.parametrize('before, after, why', [('ab', 'ab', None), (None, 'ab', 'trước'), ('ab', None, 'sau'),
                                                ('ab', 'AB', None), ('ab', 'cd', 'khác boot')])
def test_boot_problem(before, after, why):
    w = B.boot_problem(before, after)
    assert (w is None) if why is None else (why in w)
