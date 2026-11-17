"""hil_bench_session --run-gain TEN=FILE (CODE_GAIN0 17/11; Liam 17/11 at the CODE session, hop chon "GAIN0 sua sau
(A) (Recommended)"; design 03_CODE/docs/2026-11-17_Thiet_ke_bench_run_gain_v1.md of the thesis repo): one gain file
per run, so one bench session can run the final check of S05 4.6 with the six gain sets in random order. A run
without --run-gain uses --gain-file (if any), as before; without --run-gain the session is the one of 9476500."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import hil_bench_session as B  # noqa: E402
from test_hil_bench_session import _bench_env  # noqa: E402

ANSWER = ('#gain_running,sha=5bde4c55bf3a,env=esp32,mac=aa:bb:cc:dd:ee:ff,boot=4753dc47,id=7,applied_n=1,ctrl=PID,'
          'kp_xy=1.5,ki_xy=0.0500000007,kp_wz=1,ki_wz=0.0500000007,pos_kp=6,pos_ki=3,k_psi=1\n#ket_qua,ma=0\n')
ACK = ('# send 1.2.3.4:5014 #gain_set,v=1,tx=<new each try>,id=7,ctrl=PID\n'
       '# ack tx=9,req_id=7,ok=1,err=,sha=5bde4c55bf3a,env=esp32,mac=aa:bb:cc:dd:ee:ff,boot=4753dc47,id=7,'
       'applied_n=1,ctrl=PID\n#ket_qua,ma=0\n')
RUNS = ['r=--test A-L1', 's=--test A-L1', 't=--test A-L1']


def gfile(tmp_path, name, text=None):
    p = tmp_path / name
    p.write_text(text or json.dumps(dict(ctrl='PID', id=7, gains={}, tag=name)), encoding='utf-8')
    return str(p)


def bench(monkeypatch, tmp_path, runs=RUNS, ack_for=None, **kw):
    """bench() with fake ssh, gain_set, gain_query; returns (args, events, exit code). events: ('set', run, file),
    ('q_before'|'q_after', run), ('run', run)."""
    ev = []

    def reply(c):
        if 'run_hil.py' in c and 'finite-run' in c:
            for n in ('r', 's', 't'):
                if f'/{n}_' in c or f' {n} ' in c or f'{n}.csv' in c or f'{n}_frames' in c:
                    ev.append(('run', n))
                    break
            else:
                ev.append(('run', '?'))
            return (0, 'ok')
        return (0, '## HEAD\nabc') if 'git' in c else (1, '')
    a, calls, timers = _bench_env(monkeypatch, tmp_path, reply, run=list(runs), gain_query=True,
                                  esp32_ip='1.2.3.4', **kw)

    def fake_set(ip, gain_file, path, run=None):
        import gain_line_check as glc
        name = os.path.basename(path)[:-len('_gain_ack.txt')]
        ev.append(('set', name, os.path.basename(gain_file)))
        with open(path, 'x', encoding='utf-8') as f:
            f.write((ack_for or {}).get(name, ACK))
        return glc.read_ack(path)

    def fake_query(ip, path, run=None):
        name = os.path.basename(path).split('_gain_query_')[0]
        ev.append(('q_before' if path.endswith('before.txt') else 'q_after', name))
        with open(path, 'x', encoding='utf-8') as f:
            f.write(ANSWER)
        return dict(boot='4753dc47', mac='aa:bb:cc:dd:ee:ff', id='7')
    monkeypatch.setattr(B, 'gain_set', fake_set)
    monkeypatch.setattr(B, 'gain_query', fake_query)
    code, _ = B.bench(a)
    return a, ev, code, calls


def runs_of(a):
    return json.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))['runs']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ── the path without --run-gain is the path of 9476500 ─────────────────────────────────────────────────────────

def test_without_run_gain_every_run_sets_the_gain_file(tmp_path, monkeypatch):
    g0 = gfile(tmp_path, 'gain_0.json')
    a, ev, code, _ = bench(monkeypatch, tmp_path, gain_file=g0)
    assert [e for e in ev if e[0] == 'set'] == [('set', n, 'gain_0.json') for n in 'rst']
    for rec in runs_of(a):                                           # the record keys of 9476500, nothing added
        assert rec['gain_ack'] == dict(file=f"{rec['name']}_gain_ack.txt", applied=True, boot='4753dc47',
                                       mac='aa:bb:cc:dd:ee:ff', id='7')


def test_without_any_gain_file_no_gain_is_set(tmp_path, monkeypatch):
    a, ev, code, _ = bench(monkeypatch, tmp_path)
    assert [e for e in ev if e[0] == 'set'] == [] and all('gain_ack' not in r for r in runs_of(a))


# ── --run-gain ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_each_run_sets_its_own_file_in_order(tmp_path, monkeypatch):
    g0, g1, g2 = (gfile(tmp_path, f'gain_{i}.json') for i in range(3))
    a, ev, code, _ = bench(monkeypatch, tmp_path, gain_file=g0, run_gain=[f't={g2}', f'r={g1}'])
    assert [e for e in ev if e[0] == 'set'] == [('set', 'r', 'gain_1.json'), ('set', 's', 'gain_0.json'),
                                                 ('set', 't', 'gain_2.json')]
    # each set comes just before the question before of its own run
    for n in 'rst':
        i = ev.index(next(e for e in ev if e[0] == 'set' and e[1] == n))
        assert ev[i + 1] == ('q_before', n)
    by = {r['name']: r for r in runs_of(a)}
    assert by['r']['gain_ack']['gain_file'] == g1 and by['r']['gain_ack']['gain_sha256'] == sha(g1)
    assert by['t']['gain_ack']['gain_file'] == g2 and by['t']['gain_ack']['gain_sha256'] == sha(g2)
    assert 'gain_file' not in by['s']['gain_ack'] and 'gain_sha256' not in by['s']['gain_ack']


def test_run_gain_without_gain_file(tmp_path, monkeypatch):
    g1 = gfile(tmp_path, 'gain_1.json')
    a, ev, code, _ = bench(monkeypatch, tmp_path, run_gain=[f's={g1}'])
    assert [e for e in ev if e[0] == 'set'] == [('set', 's', 'gain_1.json')]
    by = {r['name']: r for r in runs_of(a)}
    assert 'gain_ack' not in by['r'] and 'gain_ack' not in by['t'] and by['s']['gain_ack']['gain_file'] == g1


def test_sha_is_of_the_file_at_the_set(tmp_path, monkeypatch):
    """The sha256 recorded is the one of the bytes that were set, read just before gain_set."""
    (tmp_path / 'gain_1.json').write_bytes(b'{"ctrl": "PID", "id": 7, "gains": {}}\n')   # the bytes on disk
    g1 = str(tmp_path / 'gain_1.json')
    a, ev, code, _ = bench(monkeypatch, tmp_path, runs=RUNS[:1], run_gain=[f'r={g1}'])
    assert runs_of(a)[0]['gain_ack']['gain_sha256'] == hashlib.sha256(
        b'{"ctrl": "PID", "id": 7, "gains": {}}\n').hexdigest()


def test_gain_not_applied_at_the_second_run_stops_there(tmp_path, monkeypatch):
    g0, g1 = gfile(tmp_path, 'gain_0.json'), gfile(tmp_path, 'gain_1.json')
    a, ev, code, _ = bench(monkeypatch, tmp_path, gain_file=g0, run_gain=[f's={g1}'],
                           ack_for={'s': '# rejected: err=motor_on\n#ket_qua,ma=2\n'})
    assert code == 2
    assert [e for e in ev if e[0] == 'set'] == [('set', 'r', 'gain_0.json'), ('set', 's', 'gain_1.json')]
    assert ('q_before', 's') not in ev and ('set', 't', 'gain_0.json') not in ev
    rec = runs_of(a)[-1]
    assert rec['name'] == 's' and rec['gain_ack']['applied'] is False and rec['gain_ack']['gain_file'] == g1
    assert rec['gain_ack']['gain_sha256'] == sha(g1)


def test_ack_of_another_boot_at_a_run_gain_run_records_its_file(tmp_path, monkeypatch):
    g1 = gfile(tmp_path, 'gain_1.json')
    a, ev, code, _ = bench(monkeypatch, tmp_path, runs=RUNS[:1], run_gain=[f'r={g1}'],
                           ack_for={'r': ACK.replace('4753dc47', '0badf00d')})
    assert code == 2
    rec = runs_of(a)[0]
    assert 'khoi dong lai sau khi dat gain' in rec['gain_ack']['problem'] and rec['gain_ack']['gain_file'] == g1


@pytest.mark.parametrize('spec,why', [
    ('x={g}', 'không có lượt'),          # no run of that name
    ('r', 'TÊN=FILE'),                   # no '='
    ('r=', 'TÊN=FILE'),                  # no file
    ('r={missing}', 'không có file'),
])
def test_bad_run_gain_stops_before_anything(tmp_path, monkeypatch, capsys, spec, why):
    g = gfile(tmp_path, 'gain_1.json')
    s = spec.format(g=g, missing=str(tmp_path / 'none.json'))
    a, ev, code, calls = bench(monkeypatch, tmp_path, run_gain=[s])
    assert code == 2 and ev == [] and calls == [] and not os.path.exists(a.out)
    assert why in capsys.readouterr().out


def test_run_gain_twice_for_one_run_is_refused(tmp_path, monkeypatch, capsys):
    g1, g2 = gfile(tmp_path, 'gain_1.json'), gfile(tmp_path, 'gain_2.json')
    a, ev, code, calls = bench(monkeypatch, tmp_path, run_gain=[f'r={g1}', f'r={g2}'])
    assert code == 2 and ev == [] and calls == [] and 'hai --run-gain' in capsys.readouterr().out


def test_run_gain_needs_gain_query(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(B, 'GAIN_SEND', __file__)
    g = gfile(tmp_path, 'gain_1.json')
    with pytest.raises(SystemExit):
        B.main(['--out', str(tmp_path / 'x'), '--run', 'a=b', '--run-gain', f'a={g}'])
    assert '--run-gain đi cùng --gain-query' in capsys.readouterr().err
    assert not os.path.exists(tmp_path / 'x')
