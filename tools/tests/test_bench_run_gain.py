"""hil_bench_session --gain-per-run TEN=FILE (CODE_GAIN0 17/11; Liam 17/11 at the CODE session, hop chon "GAIN0 sua
sau (A) (Recommended)"; design 03_CODE/docs/2026-11-17_Thiet_ke_bench_run_gain_v1.md of the thesis repo; Codex Sol
round 1 RG-1..RG-5): one gain file per run, so one bench session can run the final check of S05 4.6 with the six
gain sets in random order. The run's file is copied once to pi/<run>_gain_src.json; that copy is sent and hashed.
A run without --gain-per-run uses --gain-file (if any), as before; without the flag the session is the one of
9476500."""
import hashlib
import json
import os
import re
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


def gfile(d, name, data=None):
    p = d / name
    p.write_bytes(data if data is not None else json.dumps(dict(ctrl='PID', id=7, gains={}, tag=name)).encode())
    return str(p)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def bench(monkeypatch, tmp_path, runs=RUNS, ack_for=None, on_first_ssh=None, **kw):
    """bench() with fake ssh, gain_set, gain_query. Returns (args, events, exit code, ssh calls, sent) where events
    are ('set', run), ('q_before'|'q_after', run), ('run', run) in order, and sent = {run: (path given to gain_set,
    bytes read from it at that moment)}."""
    ev, sent, first = [], {}, [True]

    def reply(c):
        if first[0]:
            first[0] = False
            if on_first_ssh:
                on_first_ssh()                       # after the planning, before any run (file changed or gone)
        if 'run_hil.py' in c and 'finite-run' in c:
            m = re.search(r'/([A-Za-z0-9_\-]+)_frames\.npz', c)
            ev.append(('run', m.group(1) if m else '?'))
            return (0, 'ok')
        return (0, '## HEAD\nabc') if 'git' in c else (1, '')
    a, calls, timers = _bench_env(monkeypatch, tmp_path, reply, run=list(runs), gain_query=True,
                                  esp32_ip='1.2.3.4', **kw)

    def fake_set(ip, gain_file, path, run=None):
        import gain_line_check as glc
        name = os.path.basename(path)[:-len('_gain_ack.txt')]
        ev.append(('set', name))
        with open(gain_file, 'rb') as fh:
            sent[name] = (gain_file, fh.read())
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
    return a, ev, code, calls, sent


def runs_of(a):
    return json.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))['runs']


def full_order(names, with_set=True):
    out = []
    for n in names:
        out += ([('set', n)] if with_set else []) + [('q_before', n), ('run', n), ('q_after', n)]
    return out


# ── the path without --gain-per-run is the path of 9476500 ─────────────────────────────────────────────────────

def test_without_the_flag_every_run_sets_the_gain_file(tmp_path, monkeypatch):
    g0 = gfile(tmp_path, 'gain_0.json')
    a, ev, code, _, sent = bench(monkeypatch, tmp_path, gain_file=g0)
    assert ev == full_order('rst') and code != 2
    assert {n: p for n, (p, _) in sent.items()} == {n: g0 for n in 'rst'}          # the file itself, as before
    for rec in runs_of(a):                                           # the record keys of 9476500, nothing added
        assert rec['gain_ack'] == dict(file=f"{rec['name']}_gain_ack.txt", applied=True, boot='4753dc47',
                                       mac='aa:bb:cc:dd:ee:ff', id='7')
    assert not [f for f in os.listdir(os.path.join(a.out, 'pi')) if f.endswith('_gain_src.json')]


def test_without_any_gain_file_no_gain_is_set(tmp_path, monkeypatch):
    a, ev, code, _, _ = bench(monkeypatch, tmp_path)
    assert ev == full_order('rst', with_set=False) and all('gain_ack' not in r for r in runs_of(a))


# ── --gain-per-run ──────────────────────────────────────────────────────────────────────────────────────────────

def test_each_run_sends_a_snapshot_of_its_own_file(tmp_path, monkeypatch):
    g0, g1, g2 = (gfile(tmp_path, f'gain_{i}.json') for i in range(3))
    a, ev, code, _, sent = bench(monkeypatch, tmp_path, gain_file=g0, gain_per_run=[f't={g2}', f'r={g1}'])
    assert ev == full_order('rst') and code != 2
    pi = os.path.join(a.out, 'pi')
    # r, t: the snapshot in pi/ is sent; its bytes are those of the source; s: --gain-file itself, as before
    assert sent['r'] == (os.path.join(pi, 'r_gain_src.json'), open(g1, 'rb').read())
    assert sent['t'] == (os.path.join(pi, 't_gain_src.json'), open(g2, 'rb').read())
    assert sent['s'] == (g0, open(g0, 'rb').read())
    by = {r['name']: r for r in runs_of(a)}
    for n, g in (('r', g1), ('t', g2)):
        ga = by[n]['gain_ack']
        assert ga['gain_file'] == os.path.abspath(g) and ga['gain_snapshot'] == f'{n}_gain_src.json'
        assert ga['gain_sha256'] == sha(open(os.path.join(pi, f'{n}_gain_src.json'), 'rb').read()) == sha(sent[n][1])
    assert not {'gain_file', 'gain_snapshot', 'gain_sha256'} & set(by['s']['gain_ack'])
    assert not os.path.exists(os.path.join(pi, 's_gain_src.json'))


def test_relative_path_is_recorded_absolute(tmp_path, monkeypatch):
    gfile(tmp_path, 'gain_1.json')
    (tmp_path / 'sub').mkdir()
    monkeypatch.chdir(tmp_path)
    a, ev, code, _, sent = bench(monkeypatch, tmp_path, runs=RUNS[:1], gain_per_run=['r=sub/../gain_1.json'])
    assert runs_of(a)[0]['gain_ack']['gain_file'] == os.path.realpath(str(tmp_path / 'gain_1.json'))


def test_path_through_a_link_and_dotdot_is_the_file_the_os_opens(tmp_path, monkeypatch):
    """RG2-1: 'link/../g.json' with link -> real/sub is real/g.json for the OS; a text-only normalisation would pick
    the g.json next to link."""
    (tmp_path / 'real' / 'sub').mkdir(parents=True)
    a_bytes, b_bytes = b'{"tag": "A"}', b'{"tag": "B"}'
    (tmp_path / 'real' / 'g.json').write_bytes(a_bytes)
    (tmp_path / 'w').mkdir()
    (tmp_path / 'w' / 'g.json').write_bytes(b_bytes)
    try:
        os.symlink(str(tmp_path / 'real' / 'sub'), str(tmp_path / 'w' / 'link'), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('no symlink on this machine')
    monkeypatch.chdir(tmp_path / 'w')
    a, ev, code, _, sent = bench(monkeypatch, tmp_path, runs=RUNS[:1], gain_per_run=['r=link/../g.json'])
    assert sent['r'][1] == a_bytes


def test_realpath_is_used_not_text_normalisation(tmp_path, monkeypatch):
    """RG2-1 without a real link (this machine may not allow one): the recorded path is os.path.realpath of the
    argument, and abspath is not used for it."""
    g = gfile(tmp_path, 'gain_1.json')
    seen = []
    real = os.path.realpath
    monkeypatch.setattr(B.os.path, 'realpath', lambda p: seen.append(p) or real(p))
    monkeypatch.setattr(B.os.path, 'abspath', lambda p: (_ for _ in ()).throw(AssertionError('abspath used')))
    assert B.plan_run_gains([f'r={g}'], [('r', '', None)]) == {'r': real(g)} and g in seen


def test_the_source_is_read_once(tmp_path, monkeypatch):
    """RG-3: a file that changes right after the first read: what is hashed, written and sent is the first read."""
    g = gfile(tmp_path, 'gain_1.json', b'{"v": 1}')
    first, n = b'{"v": 1}', [0]
    real_open = open

    def watch_open(path, mode='r', *x, **k):
        fh = real_open(path, mode, *x, **k)
        if os.path.realpath(str(path)) == os.path.realpath(g) and 'r' in mode:
            n[0] += 1
            data = fh.read()
            fh.close()
            with real_open(g, 'wb') as w:                      # the file changes just after this read
                w.write(b'{"v": 2}')
            import io
            return io.BytesIO(data) if 'b' in mode else io.StringIO(data.decode())
        return fh
    monkeypatch.setattr(B, 'open', watch_open, raising=False)
    a, ev, code, _, sent = bench(monkeypatch, tmp_path, runs=RUNS[:1], gain_per_run=[f'r={g}'])
    snap = os.path.join(a.out, 'pi', 'r_gain_src.json')
    assert n[0] == 1 and sent['r'] == (snap, first) and open(snap, 'rb').read() == first
    assert runs_of(a)[0]['gain_ack']['gain_sha256'] == sha(first)


def test_file_changed_after_the_planning_the_bytes_at_the_set_are_sent_and_hashed(tmp_path, monkeypatch):
    """RG-1: the sha256 is of the bytes sent, read once at the set (not at the planning)."""
    g1 = gfile(tmp_path, 'gain_1.json', b'{"ctrl": "PID", "id": 7, "gains": {}}\n')
    new = b'{"ctrl": "PID", "id": 7, "gains": {"kp_xy": 2}}\n'
    a, ev, code, _, sent = bench(monkeypatch, tmp_path, runs=RUNS[:1], gain_per_run=[f'r={g1}'],
                                 on_first_ssh=lambda: open(g1, 'wb').write(new))
    assert sent['r'][1] == new and runs_of(a)[0]['gain_ack']['gain_sha256'] == sha(new)


def test_one_file_for_several_runs_each_its_own_snapshot(tmp_path, monkeypatch):
    g1 = gfile(tmp_path, 'gain_1.json')
    a, ev, code, _, sent = bench(monkeypatch, tmp_path, gain_per_run=[f'r={g1}', f's={g1}', f't={g1}'])
    assert ev == full_order('rst')
    assert [os.path.basename(sent[n][0]) for n in 'rst'] == [f'{n}_gain_src.json' for n in 'rst']


def test_gain_per_run_without_gain_file(tmp_path, monkeypatch):
    g1 = gfile(tmp_path, 'gain_1.json')
    a, ev, code, _, _ = bench(monkeypatch, tmp_path, gain_per_run=[f's={g1}'])
    assert ev == [('q_before', 'r'), ('run', 'r'), ('q_after', 'r'), ('set', 's'), ('q_before', 's'), ('run', 's'),
                  ('q_after', 's'), ('q_before', 't'), ('run', 't'), ('q_after', 't')]
    by = {r['name']: r for r in runs_of(a)}
    assert 'gain_ack' not in by['r'] and 'gain_ack' not in by['t'] and by['s']['gain_ack']['gain_file'] == g1


def test_gain_not_applied_at_the_second_run_stops_there(tmp_path, monkeypatch):
    g0, g1 = gfile(tmp_path, 'gain_0.json'), gfile(tmp_path, 'gain_1.json')
    a, ev, code, _, _ = bench(monkeypatch, tmp_path, gain_file=g0, gain_per_run=[f's={g1}'],
                              ack_for={'s': '# rejected: err=motor_on\n#ket_qua,ma=2\n'})
    assert code == 2 and ev == full_order('r') + [('set', 's')]
    rec = runs_of(a)[-1]
    assert rec['name'] == 's' and rec['gain_ack']['applied'] is False and rec['gain_ack']['gain_file'] == g1
    assert rec['gain_ack']['gain_sha256'] == sha(open(g1, 'rb').read())


def test_ack_of_another_boot_records_file_and_sha(tmp_path, monkeypatch):
    g1 = gfile(tmp_path, 'gain_1.json')
    a, ev, code, _, _ = bench(monkeypatch, tmp_path, runs=RUNS[:1], gain_per_run=[f'r={g1}'],
                              ack_for={'r': ACK.replace('4753dc47', '0badf00d')})
    assert code == 2 and ev == [('set', 'r'), ('q_before', 'r')]
    ga = runs_of(a)[0]['gain_ack']
    assert 'khoi dong lai sau khi dat gain' in ga['problem'] and ga['gain_file'] == g1
    assert ga['gain_sha256'] == sha(open(g1, 'rb').read()) and ga['gain_snapshot'] == 'r_gain_src.json'


def test_file_gone_before_its_run_is_a_controlled_stop(tmp_path, monkeypatch, capsys):
    """RG-2: the file is there at the planning, gone at the set: exit 2, the run recorded, nothing set, no run."""
    g0, g1 = gfile(tmp_path, 'gain_0.json'), gfile(tmp_path, 'gain_1.json')
    a, ev, code, _, _ = bench(monkeypatch, tmp_path, gain_file=g0, gain_per_run=[f's={g1}'],
                              on_first_ssh=lambda: os.remove(g1))
    assert code == 2 and ev == full_order('r')
    rec = runs_of(a)[-1]
    assert rec['name'] == 's' and rec['gain_ack']['applied'] is False and 'doc file gain' in rec['gain_ack']['problem']
    assert 'gain_sha256' not in rec['gain_ack'] and 'không đọc hay chép được file gain' in capsys.readouterr().out


@pytest.mark.parametrize('spec,why', [
    ('x={g}', 'không có lượt'),          # no run of that name
    ('r', 'TÊN=FILE'),                   # no '='
    ('r=', 'TÊN=FILE'),                  # no file
    ('r={missing}', 'không có file'),
])
def test_bad_gain_per_run_stops_before_anything(tmp_path, monkeypatch, capsys, spec, why):
    g = gfile(tmp_path, 'gain_1.json')
    s = spec.format(g=g, missing=str(tmp_path / 'none.json'))
    a, ev, code, calls, _ = bench(monkeypatch, tmp_path, gain_per_run=[s])
    assert code == 2 and ev == [] and calls == [] and not os.path.exists(a.out)
    assert why in capsys.readouterr().out


def test_gain_per_run_twice_for_one_run_is_refused(tmp_path, monkeypatch, capsys):
    g1, g2 = gfile(tmp_path, 'gain_1.json'), gfile(tmp_path, 'gain_2.json')
    a, ev, code, calls, _ = bench(monkeypatch, tmp_path, gain_per_run=[f'r={g1}', f'r={g2}'])
    assert code == 2 and ev == [] and calls == [] and 'hai --gain-per-run' in capsys.readouterr().out


def test_bench_without_gain_query_refuses_gain_per_run(tmp_path, monkeypatch, capsys):
    """RG-5: checked in bench() too, before any folder or SSH."""
    g = gfile(tmp_path, 'gain_1.json')
    a, calls, _ = _bench_env(monkeypatch, tmp_path, run=['r=--test A-L1'], gain_per_run=[f'r={g}'], gain_query=False)
    code, _ = B.bench(a)
    assert code == 2 and calls == [] and not os.path.exists(a.out)
    assert '--gain-per-run đi cùng --gain-query' in capsys.readouterr().out


def test_main_gain_per_run_needs_gain_query(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(B, 'GAIN_SEND', __file__)
    g = gfile(tmp_path, 'gain_1.json')
    with pytest.raises(SystemExit):
        B.main(['--out', str(tmp_path / 'x'), '--run', 'a=b', '--gain-per-run', f'a={g}'])
    assert '--gain-per-run đi cùng --gain-query' in capsys.readouterr().err
    assert not os.path.exists(tmp_path / 'x')


# ── a run that goes all the way (frame log, clean analysis, run log with the boot) ───────────────────────────────

def _full(monkeypatch, tmp_path, after_boot='4753dc47', **kw):
    """bench() where each run leaves its frame log and its run log (boot 4753dc47); analyze_run says clean; the
    question after answers after_boot. Returns (args, exit code, results)."""
    from test_bench_gain_query import write_run_log
    holder = {}

    def reply(c):
        if 'run_hil.py' in c and 'finite-run' in c:
            name = re.search(r'/([A-Za-z0-9_\-]+)_frames\.npz', c).group(1)
            open(os.path.join(holder['a'].out, 'frames', f'{name}_frames.npz'), 'w').close()
            write_run_log(holder['a'].out, name)
            return (0, 'ok')
        return (0, '## HEAD\nabc') if 'git' in c else (1, '')
    a, calls, timers = _bench_env(monkeypatch, tmp_path, reply, run=['r=--test A-L1', 's=--test A-L1'],
                                  gain_query=True, esp32_ip='1.2.3.4', campaign_env=False, **kw)
    holder['a'] = a

    def fake_set(ip, gain_file, path, run=None):
        import gain_line_check as glc
        with open(path, 'x', encoding='utf-8') as f:
            f.write(ACK)
        return glc.read_ack(path)

    tick = [10.0]

    def fake_query(ip, path, run=None):                 # like gain_query: the file, read back by read_query
        import gain_line_check as glc
        after = path.endswith('after.txt')
        tick[0] += 10.0
        with open(path, 'x', encoding='utf-8') as f:
            f.write(glc.query_pc_line(tick[0], tick[0] + 0.5)
                    + (ANSWER.replace('4753dc47', after_boot) if after else ANSWER))
        return glc.read_query(path)
    monkeypatch.setattr(B, 'gain_set', fake_set)
    monkeypatch.setattr(B, 'gain_query', fake_query)
    monkeypatch.setattr(B, 'analyze_run', lambda o, n, *x, **k: dict(name=n, clean=True, why=[]))
    monkeypatch.setattr(B, 'summary_line', lambda r: r['name'])         # as test_bench_gain_query does
    code, res = B.bench(a)
    return a, code, res


def test_full_runs_exit_0(tmp_path, monkeypatch):
    g0, g1 = gfile(tmp_path, 'gain_0.json'), gfile(tmp_path, 'gain_1.json')
    a, code, res = _full(monkeypatch, tmp_path, gain_file=g0, gain_per_run=[f'r={g1}'])
    assert code == 0 and [r['name'] for r in res] == ['r', 's'] and all(r['clean'] for r in res), res
    by = {r['name']: r for r in runs_of(a)}
    assert by['r']['gain_ack']['gain_snapshot'] == 'r_gain_src.json' and 'gain_snapshot' not in by['s']['gain_ack']


def test_full_runs_boot_after_differs_not_clean(tmp_path, monkeypatch):
    """The checks after the run still apply to a --gain-per-run run (ESP32 restart during the run)."""
    g1 = gfile(tmp_path, 'gain_1.json')
    a, code, res = _full(monkeypatch, tmp_path, after_boot='0badf00d', gain_per_run=[f'r={g1}', f's={g1}'])
    assert code == 1 and not any(r['clean'] for r in res)
    assert all(any('boot' in w for w in r['why']) for r in res), res


def test_a_snapshot_left_in_the_folder_refuses_the_session(tmp_path, monkeypatch, capsys):
    g1 = gfile(tmp_path, 'gain_1.json')
    os.makedirs(tmp_path / 's' / 'pi')
    (tmp_path / 's' / 'pi' / 'r_gain_src.json').write_bytes(b'old')
    a, ev, code, calls, _ = bench(monkeypatch, tmp_path, gain_per_run=[f'r={g1}'])
    assert code == 2 and calls == [] and ev == [] and (tmp_path / 's' / 'pi' / 'r_gain_src.json').read_bytes() == b'old'


def test_analyze_only_leaves_the_snapshot_as_it_is(tmp_path, monkeypatch):
    from types import SimpleNamespace
    g1 = gfile(tmp_path, 'gain_1.json')
    a, code, res = _full(monkeypatch, tmp_path, gain_per_run=[f'r={g1}'])
    snap = os.path.join(a.out, 'pi', 'r_gain_src.json')
    before = (open(snap, 'rb').read(), os.path.getmtime(snap))
    B.analyze_only(SimpleNamespace(out=a.out))
    assert (open(snap, 'rb').read(), os.path.getmtime(snap)) == before


@pytest.mark.parametrize('abbrev', ['--run-', '--run-t'])
def test_old_abbreviation_of_run_timeout_still_parses(monkeypatch, tmp_path, abbrev):
    """RG-4: '--run-' was a unique prefix of --run-timeout in 9476500; the new flag must not make it ambiguous."""
    seen = {}

    def fake_bench(a):
        seen['t'] = a.run_timeout
        return 0, []
    monkeypatch.setattr(B, 'bench', fake_bench)
    B.main(['--out', str(tmp_path / 'x'), '--run', 'a=b', abbrev, '600'])
    assert seen['t'] == 600
