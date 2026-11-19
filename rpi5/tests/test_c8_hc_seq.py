"""c8_hc_seq.py (CODE_HIEUCHUAN 19/11, design 03_CODE/docs/2026-11-19_Thiet_ke_CODE_HIEUCHUAN_v4.md section 2):
schedule of 12b and E, the 50 Hz sender with abort, and one attempt owning the logger."""
import math
import os
import struct
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rpi5/, from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # c8_hc_seq_e2e
import c8_hc_seq as hs  # noqa: E402
import pc_sender as ps  # noqa: E402


def decode(pkt):
    assert len(pkt) == 14 and pkt[0] == ps.TRAJ_START
    chk = 0
    for b in pkt[:13]:
        chk ^= b
    assert chk == pkt[13]
    return struct.unpack_from('<fff', pkt, 1)


class FakeSock:
    def __init__(self, fail_at=None):
        self.sent, self.fail_at = [], fail_at

    def sendto(self, pkt, addr):
        if self.fail_at is not None and len(self.sent) == self.fail_at:
            raise OSError('net down')
        self.sent.append(decode(pkt))


class FakeClock:
    """Time moves only by sleep(); stall[k] adds an extra delay after the k-th sleep."""
    def __init__(self, stall=None, ctrl_c_at=None):
        self.t, self.n, self.stall, self.ctrl_c_at = 100.0, 0, stall or {}, ctrl_c_at

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.n += 1
        if self.ctrl_c_at is not None and self.n == self.ctrl_c_at:
            raise KeyboardInterrupt
        self.t += max(d, 0.0) + self.stall.get(self.n, 0.0)


def lines():
    out = []
    return out, out.append


# ------------------------------------------------------------------ schedule
@pytest.mark.parametrize('cho,dur', [('goc', 152.232), ('K1', 127.232), ('K2', 53.4)])
def test_duration_matches_design(cho, dur):
    assert hs.duration(hs.schedule(cho)) == pytest.approx(dur, abs=1e-3)


@pytest.mark.parametrize('cho', ['goc', 'K1', 'K2'])
def test_stops_order_12b_before_E_and_holds(cho):
    segs = hs.schedule(cho)
    stops, has_e = hs.PLACES[cho]
    moves = [s for s in segs if s['loai'] == 'tien']
    assert [m['toi'] for m in moves] == list(stops[1:])
    turns = [s for s in segs if s['loai'] == 'xoay']
    assert len(turns) == (2 if has_e else 0)
    if has_e:
        assert max(s['i'] for s in moves) < min(s['i'] for s in turns)            # 12b before E (HC1-1)
        assert [t['toi'] - t['tu'] for t in turns] == pytest.approx([5 * math.pi, -5 * math.pi])
    assert segs[0]['loai'] == 'dung' and segs[-1]['loai'] == 'dung'
    for s in segs:
        if s['loai'] == 'dung':
            assert s['t1'] - s['t0'] == pytest.approx(hs.HOLD_S)
    for a, b in zip(segs[:-1], segs[1:]):
        assert b['t0'] == pytest.approx(a['t1'])
        assert {a['loai'], b['loai']} & {'dung'}                                    # a stop between any two moves


@pytest.mark.parametrize('cho,phan', [('goc', 'all'), ('K1', '12b'), ('K2', 'all'), ('K1', 'E')])
def test_packets_integrate_to_plan_and_respect_limits(cho, phan):
    segs = hs.schedule(cho, phan)
    n = hs.n_samples(segs)
    v = [hs.sample(segs, k) for k in range(n)]
    assert max(abs(x[0]) for x in v) <= hs.V_MAX
    assert max(abs(x[2]) for x in v) <= hs.W_MAX
    assert all(x[1] == 0.0 for x in v)
    assert all(x[0] == 0.0 or x[2] == 0.0 for x in v)                                # never translate and turn together
    for s in segs:
        k0, k1 = int(math.floor(s['t0'] * hs.SEND_HZ)), int(math.ceil(s['t1'] * hs.SEND_HZ))
        if s['loai'] == 'dung':
            continue
        col = 0 if s['loai'] == 'tien' else 2
        tot = sum(v[k][col] for k in range(k0, min(k1, n))) * hs.DT
        assert tot == pytest.approx(s['toi'] - s['tu'], abs=1e-9)
    # every stop: zero commands across the whole stop
    for s in segs:
        if s['loai'] == 'dung':
            ks = range(int(math.ceil(s['t0'] * hs.SEND_HZ)), int(math.floor(s['t1'] * hs.SEND_HZ)))
            assert all(v[k] == (0.0, 0.0, 0.0) for k in ks if k < n)


def test_k2_back_leg_is_050_from_forward_point():
    moves = [s for s in hs.schedule('K2') if s['loai'] == 'tien']
    assert moves[1]['tu'] == pytest.approx(0.15) and moves[1]['toi'] == pytest.approx(-0.35)


def test_refusals():
    with pytest.raises(hs.SeqError):
        hs.schedule('K2', 'E')
    with pytest.raises(hs.SeqError):
        hs.schedule('X')


def test_lich_csv_stable_sha():
    a, b = hs.lich_csv(hs.schedule('goc')), hs.lich_csv(hs.schedule('goc'))
    assert a == b and a.startswith('i,loai,t0_s')


def test_in_lich_opens_no_socket(monkeypatch, capsys):
    monkeypatch.setattr(hs.socket, 'socket', lambda *a, **k: (_ for _ in ()).throw(AssertionError('socket')))
    assert hs.main(['--cho', 'K1', '--in-lich']) == 0
    assert '#ket_qua,ma=0' in capsys.readouterr().out


# ------------------------------------------------------------------ sending
def run_send(segs, clock, sock=None, alive=lambda: True):
    out, say = lines()
    sock = sock or FakeSock()
    try:
        r = hs.send_schedule('1.2.3.4', segs, sock, say, clock, clock.sleep, lambda: 1.0e9, alive=alive, cho='K2')
        err = None
    except hs.Abort as exc:
        r, err = None, exc
    return r, err, out, sock


def test_send_full_schedule_then_tail():
    segs = hs.schedule('K2')
    r, err, out, sock = run_send(segs, FakeClock())
    assert err is None and r['complete'] == 1 and r['n_bo'] == 0
    n = hs.n_samples(segs)
    assert len(sock.sent) == n + int(math.ceil(hs.TAIL_S * hs.SEND_HZ))
    for got, want in zip(sock.sent[:n], [hs.sample(segs, k) for k in range(n)]):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-7)                       # float32 packets
    assert all(p == (0.0, 0.0, 0.0) for p in sock.sent[n:])
    assert sum(1 for ln in out if ln.startswith('#seg,')) == len(segs)
    assert out[0].startswith('#seq_start,ver=c8_hc_seq_v3') and out[-1].startswith('#seq_end') and 'complete=1' in out[-1]


def test_gap_over_100ms_aborts_and_never_resumes():
    segs = hs.schedule('K2')
    r, err, out, sock = run_send(segs, FakeClock(stall={300: 0.15}))
    assert err is not None and err.why == 'tre_gui'
    assert any(ln.startswith('#seq_huy,ly_do=tre_gui') for ln in out)
    tail = sock.sent[-int(hs.ABORT_ZERO_S * hs.SEND_HZ):]
    assert all(p == (0.0, 0.0, 0.0) for p in tail)
    assert len(sock.sent) < 400                                                     # stopped, not continued
    assert 'complete=0' in [ln for ln in out if ln.startswith('#seq_end')][0]


def test_short_stall_skips_samples_without_burst():
    segs = hs.schedule('K2')
    r, err, out, sock = run_send(segs, FakeClock(stall={500: 0.05}))
    assert err is None and r['n_bo'] >= 2 and r['complete'] == 0
    assert len(sock.sent) == r['sent'] == r['planned'] - r['n_bo']


def test_socket_error_and_dead_logger_abort():
    segs = hs.schedule('K2')
    _, err, out, _ = run_send(segs, FakeClock(), sock=FakeSock(fail_at=50))
    assert err.why == 'socket'
    calls = {'n': 0}

    def alive():
        calls['n'] += 1
        return calls['n'] < 80
    _, err, out, _ = run_send(segs, FakeClock(), alive=alive)
    assert err.why == 'logger_chet'


def test_ctrl_c_sends_zeros_and_aborts():
    segs = hs.schedule('K2')
    _, err, out, sock = run_send(segs, FakeClock(ctrl_c_at=200))
    assert err.why == 'ctrl_c'
    assert sock.sent[-1] == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------ one attempt
class FakeProc:
    """Logger stand-in: writes CSV rows; SIGINT writes a closed summary whose CSV sha256 matches (as agv_logger)."""
    pid = 4242

    def __init__(self, csv_path, rows=3, die_after=None, summary='ok', rc=0):
        self.csv, self.returncode, self.rows, self.die_after, self.polls = csv_path, None, rows, die_after, 0
        self.summary, self.rc = summary, rc
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('timestamp_us,fl_count\n' + ''.join(f'{i},0\n' for i in range(rows)))

    def poll(self):
        self.polls += 1
        if self.die_after is not None and self.polls > self.die_after:
            self.returncode = 1
        return self.returncode

    def send_signal(self, sig):
        self.returncode = self.rc
        if self.summary is None:
            return
        import hashlib
        import json
        sha = hashlib.sha256(open(self.csv, 'rb').read()).hexdigest()
        if self.summary == 'sai_sha':
            sha = '0' * 64
        with open(self.csv + '.summary.json', 'w', encoding='utf-8') as f:
            json.dump(dict(status='closed', files=dict(csv=dict(sha256=sha))), f)

    def wait(self, timeout=None):
        return self.returncode


def fake_popen(rows=3, die_after=None, summary='ok', rc=0):
    made = {}

    def popen(cmd, **kw):
        out = cmd[cmd.index('--out') + 1]
        assert '--strict' in cmd and cmd[cmd.index('--src-ip') + 1] == '1.2.3.4'
        made['p'] = FakeProc(out, rows, die_after, summary, rc)
        return made['p']
    return popen, made


def args(d, cho='K2', phan='all', src_ip='1.2.3.4'):
    return types.SimpleNamespace(ip='1.2.3.4', src_ip=src_ip, cho=cho, phan=phan, thu_muc=str(d))


def attempt(tmp_path, **kw):
    clock = FakeClock(stall=kw.pop('stall', None))
    popen, made = fake_popen(kw.pop('rows', 3), kw.pop('die_after', None), kw.pop('summary', 'ok'), kw.pop('rc', 0))
    q = kw.pop('query', lambda ip, path: open(path, 'x').write('#ket_qua,ma=0\n') and 0)
    ready = kw.pop('ready', lambda: None)
    pgrep = kw.pop('pgrep', lambda: '')
    sock = FakeSock()
    out, say = lines()
    code = hs.run_attempt(args(tmp_path / 'HC_K2_1', **kw), out=say, clock=clock, sleep=clock.sleep,
                          wall=lambda: 1.0e9, sock=sock, popen=popen, query=q, pgrep=pgrep, ready=ready,
                          lock=lambda: None)
    return code, out, sock, tmp_path / 'HC_K2_1'


def test_attempt_from_empty_folder(tmp_path):
    code, out, sock, d = attempt(tmp_path)
    assert code == 0
    for f in ('sender.txt', 'lich.csv', 'log.csv', 'logger_stdout.txt', 'gain_query_truoc.txt', 'gain_query_sau.txt'):
        assert (d / f).exists(), f
    txt = (d / 'sender.txt').read_text(encoding='utf-8')
    assert '#log_dong_dau' in txt and '#seq_end' in txt and '#logger_dong,ma=0' in txt and txt.rstrip().endswith('#ket_qua,ma=0')
    assert txt.index('#log_dong_dau') < txt.index('#gain_query,when=truoc') < txt.index('#seq_start')


def test_attempt_refuses_existing_folder(tmp_path):
    (tmp_path / 'HC_K2_1').mkdir()
    with pytest.raises(hs.SeqError):
        attempt(tmp_path)


def test_attempt_no_schedule_when_question_fails(tmp_path):
    code, out, sock, d = attempt(tmp_path, query=lambda ip, path: 1)
    assert code == 2 and sock.sent == []


def test_attempt_no_data_row(tmp_path):
    code, out, sock, d = attempt(tmp_path, rows=0)
    assert code == 2 and sock.sent == []


def test_attempt_logger_dies_during_schedule(tmp_path):
    code, out, sock, d = attempt(tmp_path, die_after=200)                       # before the first motion packet
    assert code == 2 and all(p == (0.0, 0.0, 0.0) for p in sock.sent)
    code, out, sock, d = attempt(tmp_path / 'b', die_after=600)                 # while moving
    assert code == 3
    assert '#seq_huy,ly_do=logger_chet' in (d / 'sender.txt').read_text(encoding='utf-8')


def test_others_running_parses_pgrep():
    txt = '11 python3 rpi5/pc_sender.py --ip x\n12 python3 tools/agv_logger.py --strict\n13 python3 other.py\n'
    assert hs.others_running(lambda: txt, me=99) == ['11:pc_sender.py', '12:agv_logger.py']
    assert hs.others_running(lambda: '5 python3 rpi5/c8_hc_seq.py', me=5) == []


def test_imports_exist_in_pc_sender():
    for n in ('SEND_HZ', 'TRAJ_PORT', 'gain_query_file', 'pack_packet'):
        assert hasattr(ps, n)
    assert os.path.isfile(hs.LOGGER)


# ------------------------------------------------------------------ Codex code review round 1 (HCC1-1..3, 14)
class TimedSock(FakeSock):
    """Records (time, packet); delay[n] moves the clock forward inside the n-th sendto (a blocked socket)."""
    def __init__(self, clock, delay=None):
        super().__init__()
        self.clock, self.delay, self.times = clock, delay or {}, []

    def sendto(self, pkt, addr):
        n = len(self.sent)
        self.times.append(self.clock.t)
        super().sendto(pkt, addr)
        self.clock.t += self.delay.get(n, 0.0)


def max_gap(sock):
    return max(b - a for a, b in zip(sock.times[:-1], sock.times[1:]))


def test_slow_seg_write_does_not_delay_packets():
    """HCC1-1: '#seg' writes take 150 ms each (slow SD card): nothing is written between check and send, the lines
    come after the schedule, every packet keeps the 20 ms grid."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    sock = TimedSock(clock)
    out = []

    def slow(line):
        if line.startswith('#seg'):
            clock.t += 0.150
        out.append(line)
    r = hs.send_schedule('1.2.3.4', segs, sock, slow, clock, clock.sleep, lambda: 1.0e9)
    assert r['complete'] == 1 and max_gap(sock) <= 0.0201
    assert sum(1 for ln in out if ln.startswith('#seg')) == len(segs)


def test_blocked_sendto_aborts_before_next_motion_packet():
    """HCC1-1: a sendto that takes 150 ms aborts at once (no further motion packet)."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    k_move = hs.first_motion_k(segs)
    sock = TimedSock(clock, delay={k_move + 10: 0.150})
    _, err, out, _ = run_send(segs, clock, sock=sock)
    assert err is not None and err.why == 'sendto_cham'
    assert all(p == (0.0, 0.0, 0.0) for p in sock.sent[k_move + 11:])            # only abort zeros after it


def test_stall_jumping_past_the_end_is_still_a_gap():
    """HCC1-1: a stall on the last wait that jumps past the end of the schedule is checked before leaving."""
    segs = hs.schedule('K2')
    total = hs.n_samples(segs) + int(math.ceil(hs.TAIL_S * hs.SEND_HZ - 1e-9))
    r, err, out, sock = run_send(segs, FakeClock(stall={total - 1: 0.30}))
    assert err is not None and err.why == 'tre_gui'


def test_write_error_keeps_zero_tail():
    """HCC1-2: the log write fails (broken pipe) for the motion -> stop line: the packets already ended in zeros."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    sock = FakeSock()

    def broken(line):
        if line.startswith('#seg,i=2'):
            raise BrokenPipeError('pipe')
    with pytest.raises(BrokenPipeError):
        hs.send_schedule('1.2.3.4', segs, sock, broken, clock, clock.sleep, lambda: 1.0e9)
    assert sock.sent[-1] == (0.0, 0.0, 0.0)
    assert len(sock.sent) == hs.n_samples(segs) + int(math.ceil(hs.TAIL_S * hs.SEND_HZ - 1e-9))


def test_unexpected_error_in_loop_sends_zeros_first():
    """HCC1-2: any exception once sending started (here from ready()) sends the abort zeros before re-raising."""
    segs = hs.schedule('K2')

    def boom():
        raise OSError('disk')
    sock = FakeSock()
    clock = FakeClock()
    with pytest.raises(hs.Abort) as ei:
        hs.send_schedule('1.2.3.4', segs, sock, lambda s: None, clock, clock.sleep, lambda: 1.0e9, ready=boom)
    assert ei.value.why == 'loi_OSError'
    assert sock.sent and all(p == (0.0, 0.0, 0.0) for p in sock.sent)


def test_car_not_ready_aborts_before_any_motion():
    """HCC1-3: the check 0.5 s before the first motion packet fails -> no motion packet, reason in the log."""
    segs = hs.schedule('K2')
    sock = FakeSock()
    clock = FakeClock()
    got = []
    with pytest.raises(hs.Abort) as ei:
        hs.send_schedule('1.2.3.4', segs, sock, got.append, clock, clock.sleep, lambda: 1.0e9,
                         ready=lambda: 'run_active=0 (MANUAL)')
    assert ei.value.why == 'xe_chua_san_sang'
    assert all(p == (0.0, 0.0, 0.0) for p in sock.sent)
    assert any(ln.startswith('#kiem_truoc_chay') and 'ket_qua=hong' in ln for ln in got)


def write_log(path, rows, cols=('timestamp_us', 'motor_on', 'run_active', 'sp_vx', 'sp_vy', 'sp_wz', 'traj_age_ms')):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')


def test_car_ready_rows(tmp_path):
    p = str(tmp_path / 'log.csv')
    good = [(i, 1, 1, '0.000', '0.000', '0.000', 15) for i in range(40)]
    write_log(p, good)
    assert hs.car_ready(p) is None
    for bad, word in [((1, 0, 1, '0.000', '0.000', '0.000', 15), 'motor_on'),
                      ((1, 1, 0, '0.000', '0.000', '0.000', 15), 'run_active'),
                      ((1, 1, 1, '0.100', '0.000', '0.000', 15), 'sp_khac_0'),
                      ((1, 1, 1, '0.000', '0.000', '0.000', 190), 'traj_age')]:
        write_log(p, good + [bad] + good[:5])
        assert word in hs.car_ready(p)
    write_log(p, good[:5])
    assert 'dong log' in hs.car_ready(p)
    write_log(p, good, cols=('timestamp_us', 'motor_on'))
    assert 'thieu cot' in hs.car_ready(p)


def test_attempt_refuses_src_ip_other_than_ip(tmp_path):
    with pytest.raises(hs.SeqError):
        attempt(tmp_path, src_ip='9.9.9.9')


def test_attempt_not_ready_is_code_2_without_motion(tmp_path):
    code, out, sock, d = attempt(tmp_path, ready=lambda: 'motor_on=0')
    assert code == 2 and all(p == (0.0, 0.0, 0.0) for p in sock.sent)
    assert '#ket_qua,ma=2,ly_do=huy_xe_chua_san_sang' in (d / 'sender.txt').read_text(encoding='utf-8')


def _q_after_fails(ip, path):
    if 'truoc' in path:
        open(path, 'x').write('#ket_qua,ma=0\n')
        return 0
    return 1


@pytest.mark.parametrize('kw,why', [
    (dict(stall={500: 0.05}), 'complete=0'),
    (dict(query=_q_after_fails), 'gain_sau'),
    (dict(rc=3), 'logger'),
    (dict(summary=None), 'logger'),
    (dict(summary='sai_sha'), 'logger'),
])
def test_attempt_success_needs_every_part(tmp_path, kw, why):
    """HCC1-2: '#ket_qua,ma=0' only with complete=1, the question after, logger exit 0 and a matching summary."""
    code, out, sock, d = attempt(tmp_path, **kw)
    last = (d / 'sender.txt').read_text(encoding='utf-8').strip().splitlines()[-1]
    assert code == 3 and why in last, last


def test_attempt_other_process_after_schedule(tmp_path):
    calls = {'n': 0}

    def pg():
        calls['n'] += 1
        return '' if calls['n'] == 1 else '77 python3 rpi5/pc_sender.py --ip x'
    code, out, sock, d = attempt(tmp_path, pgrep=pg)
    assert code == 3 and 'tien_trinh_khac' in (d / 'sender.txt').read_text(encoding='utf-8')


def test_pgrep_failure_refuses(monkeypatch):
    def bad(*a, **k):
        raise FileNotFoundError('pgrep')
    monkeypatch.setattr(hs.subprocess, 'run', bad)
    with pytest.raises(hs.SeqError):
        hs.others_running()
    monkeypatch.setattr(hs.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=2, stdout=''))
    with pytest.raises(hs.SeqError):
        hs.others_running()
    monkeypatch.setattr(hs.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=''))
    assert hs.others_running() == []


def test_others_running_excludes_own_logger():
    txt = '12 python3 tools/agv_logger.py --strict\n13 python3 tools/agv_logger.py --strict'
    assert hs.others_running(lambda: txt, me=99, exclude=(12,)) == ['13:agv_logger.py']


@pytest.mark.skipif(os.name != 'posix', reason='real logger + SIGINT: Linux (Pi, WSL)')
@pytest.mark.parametrize('ctrl_c', [False, True])
def test_e2e_real_logger_posix(tmp_path, ctrl_c):
    """HCC1-14: the real agv_logger.py as a child (own session, SIGINT, summary), real clock and sockets."""
    import c8_hc_seq_e2e as e2e
    code, txt, summ, esp = e2e.chay(str(tmp_path), ctrl_c_after=12.0 if ctrl_c else None)
    assert e2e.kiem(code, txt, summ, esp, ctrl_c) == []


# ------------------------------------------------------------------ Codex code review round 2 (HCC2-1)
READY_COLS = ('timestamp_us', 'log_seq', 'motor_on', 'run_active', 'sp_vx', 'sp_vy', 'sp_wz', 'traj_age_ms')


def ready_rows(path, seqs, tail=''):
    """Logger CSV with READY_COLS rows (timestamp_us = 10000 * i, log_seq from seqs), then raw text `tail`."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(READY_COLS) + '\n')
        for i, s in enumerate(seqs):
            f.write(f'{10000 * i},{s},1,1,0.000,0.000,0.000,10\n')
        f.write(tail)


def test_ready_check_needs_a_moving_log(tmp_path):
    """HCC2-1, HCC5-4, HCC7-3: log_seq must advance between the two ready checks and never go back; an unfinished
    last CSV line is ignored (not 'dong log hong')."""
    p = str(tmp_path / 'log.csv')
    ready_rows(p, range(40))
    rc = hs.ReadyCheck(p)
    assert rc() is None
    assert 'chi tang' in rc()                                                   # same rows: the stream stopped
    ready_rows(p, range(80))
    assert rc() is None
    ready_rows(p, list(range(80)) + [50])                                       # max over the tail: still 79
    assert 'chi tang' in rc()
    ready_rows(p, range(30))                                                    # DUT reboot
    assert 'lui' in rc()
    ready_rows(p, range(120), tail='1200000,')                                  # the logger is writing a line
    assert rc.max_row() == (119, 1190000)


def test_late_sample_is_chosen_again_not_sent_twice():
    """HCC2-1 (Codex): a 60 ms delay after the sample was chosen sent two forward packets at the same time."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    sock = TimedSock(clock)
    calls = {'n': 0}

    def wall():
        calls['n'] += 1
        if calls['n'] == 3:                                                     # inside the loop ('#seg' of a move)
            clock.t += 0.060
        return 1.0e9
    try:
        hs.send_schedule('1.2.3.4', segs, sock, lambda s: None, clock, clock.sleep, wall)
    except hs.Abort:
        pass
    gaps = [b - a for a, b in zip(sock.times[:-1], sock.times[1:])]
    assert min(gaps) >= hs.DT / 2 - 1e-9


def test_reported_gap_bounds_the_true_gap():
    """HCC2-1 (Codex): a 101 ms gap between packets was reported as 99 ms (clock read before sendto)."""
    segs = hs.schedule('K2')
    clock = FakeClock(stall={300: 0.065})
    sock = TimedSock(clock, delay={299: 0.015, 300: 0.015})
    r, err, out, _ = run_send(segs, clock, sock=sock)
    end = [ln for ln in out if ln.startswith('#seq_end')][0]
    rep = float(end.split('gap_max_ms=')[1].split(',')[0]) / 1000
    assert rep >= max_gap(sock) - 1e-9


# ------------------------------------------------------------------ Codex code review round 3 (HCC3-1, HCC3-2)
class EmitSock(FakeSock):
    """The packet leaves at the END of a slow sendto (delay[n] s blocked before emission); records leave times."""
    def __init__(self, clock, delay=None):
        super().__init__()
        self.clock, self.delay, self.times = clock, delay or {}, []

    def sendto(self, pkt, addr):
        self.clock.t += self.delay.get(len(self.sent), 0.0)
        self.times.append(self.clock.t)
        super().sendto(pkt, addr)


class LateClock(FakeClock):
    """sleep() returns late by late[n] s on the n-th sleep."""
    def __init__(self, late):
        super().__init__()
        self.late = late

    def sleep(self, d):
        super().sleep(d)
        self.t += self.late.get(self.n, 0.0)


def test_slow_ready_checks_do_not_drop_samples():
    """HCC3-1 (Codex): two 12 ms ready checks dropped two zero packets and the attempt ended complete=0."""
    segs = hs.schedule('K2')
    clock = FakeClock()

    def slow_ready():
        clock.t += 0.012
        return None
    sock = FakeSock()
    r = hs.send_schedule('1.2.3.4', segs, sock, lambda s: None, clock, clock.sleep, lambda: 1.0e9, ready=slow_ready)
    assert r['complete'] == 1 and r['n_bo'] == 0


@pytest.mark.parametrize('late', [0.009, 0.011, 0.019])
def test_late_sleep_inside_the_cell_keeps_the_sample(late):
    """HCC3-1 (Codex): a sleep 11 ms late in the motion phase lost a packet although still inside its 20 ms cell."""
    segs = hs.schedule('K2')
    clock = LateClock({hs.first_motion_k(segs) + 30: late})
    sock = EmitSock(clock)
    r = hs.send_schedule('1.2.3.4', segs, sock, lambda s: None, clock, clock.sleep, lambda: 1.0e9)
    assert r['complete'] == 1 and r['n_bo'] == 0
    gaps = [b - a for a, b in zip(sock.times[:-1], sock.times[1:])]
    assert min(gaps) >= hs.MIN_SPACING_S - 1e-9


def test_blocked_sendto_cannot_send_two_packets_together():
    """HCC3-2 (Codex): a sendto blocked 19 ms before emission, the next packet 1 ms later, complete=1."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    k_move = hs.first_motion_k(segs)
    sock = EmitSock(clock, delay={k_move: 0.019})
    hs.send_schedule('1.2.3.4', segs, sock, lambda s: None, clock, clock.sleep, lambda: 1.0e9)
    gaps = [b - a for a, b in zip(sock.times[:-1], sock.times[1:])]
    assert min(gaps) >= hs.MIN_SPACING_S - 1e-9


def test_84ms_gap_then_send_is_not_aborted():
    """HCC3-2 (Codex): an 84 ms gap with an immediate next send was aborted by the fixed 20 ms deduction."""
    segs = hs.schedule('K2')
    clock = FakeClock(stall={400: 0.064})
    sock = EmitSock(clock)
    r, err, out, _ = run_send(segs, clock, sock=sock)
    assert err is None and r['gap_max_ms'] <= 100.0


# ------------------------------------------------------------------ Codex code review round 4 (HCC4-1..3)
class BudgetSock(EmitSock):
    """Honours settimeout like a real socket: a sendto blocked longer than the timeout raises, nothing is sent."""
    timeout = None

    def settimeout(self, v):
        self.timeout = v

    def sendto(self, pkt, addr):
        d = self.delay.get(len(self.sent), 0.0)
        if self.timeout is not None and d > self.timeout:
            self.clock.t += self.timeout
            self.delay.pop(len(self.sent))
            import socket as _s
            raise _s.timeout('timed out')
        super().sendto(pkt, addr)


def test_no_motion_packet_after_the_100ms_budget():
    """HCC4-1, HCC5-1: the socket is non-blocking (a full buffer raises instead of sending late); a check 99.5 ms
    after the previous one aborts before sending, so no motion packet leaves after the 100 ms budget."""
    segs = hs.schedule('K2')
    k_move = hs.first_motion_k(segs)
    clock = FakeClock(stall={k_move + 20: 0.0795})                              # next check 99.5 ms after the last
    sock = EmitSock(clock)
    r, err, out, _ = run_send(segs, clock, sock=sock)
    moving = [tm for tm, p in zip(sock.times, sock.sent) if p != (0.0, 0.0, 0.0)]
    gaps = [b - a for a, b in zip(moving[:-1], moving[1:])]
    assert err is not None and err.why == 'tre_gui' and max(gaps) <= hs.GAP_ABORT_S + 1e-9


def test_abort_zeros_get_their_own_budget():
    """HCC5-2 (Codex): the abort zeros kept the shortened motion timeout and 0/25 left; they get a fresh budget and
    the count of zeros sent is logged."""
    segs = hs.schedule('K2')

    class Sock(FakeSock):
        timeout, blocking = 0.005, False

        def setblocking(self, b):
            self.blocking = b

        def settimeout(self, v):                                             # as socket: a timeout -> blocking
            self.timeout = v
            self.blocking = v is None or v > 0

        def sendto(self, pkt, addr):
            if not self.blocking or self.timeout < 0.010:                    # this socket needs 10 ms to send
                raise OSError('would block')
            super().sendto(pkt, addr)
    sock = Sock()
    clock = FakeClock(ctrl_c_at=300)
    got = []
    with pytest.raises(hs.Abort):
        hs.send_schedule('1.2.3.4', segs, sock, got.append, clock, clock.sleep, lambda: 1.0e9)
    assert len(sock.sent) == int(round(hs.ABORT_ZERO_S * hs.SEND_HZ))
    assert any('goi0=25/25' in ln for ln in got)


def test_stall_jumping_into_motion_refuses_before_moving():
    """HCC4-2 (Codex): K2, the wake for k=199 45 ms late jumped to k=201; the car got motion and ran the schedule."""
    segs = hs.schedule('K2')
    k_move = hs.first_motion_k(segs)
    clock = LateClock({k_move - 1: 0.045})
    sock = FakeSock()
    _, err, out, _ = run_send(segs, clock, sock=sock)
    assert err is not None and err.why == 'tre_truoc_chay' and not err.moved
    assert all(p == (0.0, 0.0, 0.0) for p in sock.sent)


def test_ctrl_c_right_after_first_motion_packet_is_moved():
    """HCC4-3 (Codex): Ctrl-C at the clock read right after the first motion sendto reported 'refused' (2)."""
    segs = hs.schedule('K2')
    k_move = hs.first_motion_k(segs)

    class C(FakeClock):
        def __init__(self, sock):
            super().__init__()
            self.sock = sock

        def __call__(self):
            if any(p != (0.0, 0.0, 0.0) for p in self.sock.sent):
                raise KeyboardInterrupt
            return self.t
    sock = FakeSock()
    clock = C(sock)
    with pytest.raises(hs.Abort) as ei:
        hs.send_schedule('1.2.3.4', segs, sock, lambda s: None, clock, clock.sleep, lambda: 1.0e9)
    assert ei.value.moved and k_move > 0


# ------------------------------------------------------------------ Codex code review rounds 6-7 (HCC6, HCC7)
class Clk:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def test_poll_deadline_only_when_the_accepted_log_grows(tmp_path):
    """HCC7-1..3: freshness from the logger CSV (accepted rows) timed on the sender clock: a new largest log_seq
    after a poll gives deadline = that poll + READY_FRESH_S; nothing else does."""
    p = str(tmp_path / 'log.csv')
    rc, c = hs.ReadyCheck(p), Clk()
    ready_rows(p, range(40))
    assert rc.poll(c) is None                                                   # first poll: no earlier one
    c.t = 100.02
    ready_rows(p, range(42))
    assert rc.poll(c) == pytest.approx(100.0 + hs.READY_FRESH_S)
    c.t = 100.04
    ready_rows(p, list(range(42)) + [41])                                       # repeated datagram written again
    assert rc.poll(c) is None and 'khong tang' in rc.ly_do
    c.t = 100.06
    ready_rows(p, list(range(42)) + [41, 30])                                   # late copy of an old row
    assert rc.poll(c) is None
    c.t = 100.08
    ready_rows(p, list(range(42)) + [41, 30], tail='4200000,42,1')              # row being written: not counted
    assert rc.poll(c) is None
    c.t = 100.10
    ready_rows(p, list(range(42)) + [41, 30, 42])
    assert rc.poll(c) == pytest.approx(100.08 + hs.READY_FRESH_S)


def test_poll_bad_rows_and_reboot(tmp_path):
    """HCC7-2: a badly typed row (logger would refuse it; here it is in the CSV) is skipped, no exception.
    HCC7-3: a DUT reboot (log_seq back) is reported and never counts as fresh, even when the new boot grows."""
    p = str(tmp_path / 'log.csv')
    rc, c = hs.ReadyCheck(p), Clk()
    ready_rows(p, range(40))
    rc.poll(c)
    c.t = 100.02
    ready_rows(p, list(range(40)), tail='400000,inf,1,1,0.000,0.000,0.000,10\n')
    assert rc.poll(c) is None
    c.t = 100.04
    ready_rows(p, range(35))                                                    # after a reboot: 0..34
    assert rc.poll(c) is None and 'lui' in rc.ly_do
    c.t = 100.06
    ready_rows(p, range(37))
    assert rc.poll(c) is None and 'lui' in rc.ly_do


class PollReady:
    """ready() passes; poll() as ReadyCheck on the fake clock: grows (deadline previous poll + 0.2 s) or not, `lag`
    s per read, or a fixed absolute deadline."""
    def __init__(self, clock, grow=True, lag=0.0, fixed=None):
        self.clock, self.grow, self.lag, self.fixed = clock, grow, lag, fixed
        self.t_prev, self.calls, self.ly_do = None, 0, 'log khong tang'

    def __call__(self):
        return None

    def poll(self, clock):
        self.calls += 1
        t = clock()
        self.clock.t += self.lag
        dl = None
        if self.fixed is not None:
            dl = self.fixed
        elif self.grow and self.t_prev is not None:
            dl = self.t_prev + hs.READY_FRESH_S
        self.t_prev = t
        return dl


def test_stale_log_at_first_motion_packet_refuses():
    """HCC4-2, HCC7: the log does not grow in the polls before the first motion packet: refused, nothing moved."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    sock = FakeSock()
    out = []
    with pytest.raises(hs.Abort) as ei:
        hs.send_schedule('1.2.3.4', segs, sock, out.append, clock, clock.sleep, lambda: 1.0e9,
                         ready=PollReady(clock, grow=False))
    assert ei.value.why == 'log_cu' and not ei.value.moved and all(p == (0.0, 0.0, 0.0) for p in sock.sent)
    assert any(ln.startswith('#log_moi_truoc_goi_dong,ket_qua=hong') for ln in out)


def test_nominal_polls_then_motion():
    """The polls run after the last zeros; a growing log lets the schedule run to the end."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    rd = PollReady(clock)
    out = []
    r = hs.send_schedule('1.2.3.4', segs, FakeSock(), out.append, clock, clock.sleep, lambda: 1.0e9, ready=rd)
    assert r['complete'] == 1 and rd.calls == hs.FRESH_POLL_K + 1
    assert any(ln.startswith('#log_moi_truoc_goi_dong,ket_qua=dat') for ln in out)


def test_slow_poll_does_not_refuse_a_nominal_attempt():
    """HCC6-3, HCC7-4 (Codex): a 24 ms read in the cell of the first motion packet refused the attempt; the polls
    now run in the wait after a zero, and a slower read than the wait still only refuses (code 2, nothing moved)."""
    segs = hs.schedule('K2')
    clock = FakeClock()
    r = hs.send_schedule('1.2.3.4', segs, FakeSock(), lambda s: None, clock, clock.sleep, lambda: 1.0e9,
                         ready=PollReady(clock, lag=0.012))
    assert r['complete'] == 1
    clock = FakeClock()
    sock = FakeSock()
    with pytest.raises(hs.Abort) as ei:
        hs.send_schedule('1.2.3.4', segs, sock, lambda s: None, clock, clock.sleep, lambda: 1.0e9,
                         ready=PollReady(clock, lag=0.045))
    assert not ei.value.moved and all(p == (0.0, 0.0, 0.0) for p in sock.sent)


def test_deadline_passed_inside_the_first_motion_sendto_is_not_complete():
    """HCC7-4 (Codex): preempted 2 ms between the last clock read and the syscall, the first motion packet left
    after the freshness deadline and the attempt was reported complete=1; now it aborts (zeros, complete=0)."""
    segs = hs.schedule('K2')
    k_move = hs.first_motion_k(segs)
    clock = FakeClock()
    dl = clock.t + k_move * hs.DT + 0.001                                       # just after the first motion check
    sock = TimedSock(clock, delay={k_move: 0.002})
    out = []
    with pytest.raises(hs.Abort) as ei:
        hs.send_schedule('1.2.3.4', segs, sock, out.append, clock, clock.sleep, lambda: 1.0e9,
                         ready=PollReady(clock, fixed=dl))
    assert ei.value.why == 'log_cu' and ei.value.moved and ei.value.kv.get('sau_gui') == 1
    assert any('complete=0' in ln for ln in out)


def test_sigint_handler_restored_when_interrupted_while_set(monkeypatch):
    """HCC7-5 (Codex): Ctrl-C right after SIG_IGN was set, before signal.signal() returned: the old handler was
    never restored and SIGINT stayed ignored."""
    import signal as sg
    real = sg.signal
    before = sg.getsignal(sg.SIGINT)
    calls = {'n': 0}

    def flaky(sig, h):
        calls['n'] += 1
        r = real(sig, h)
        if calls['n'] == 1:
            raise KeyboardInterrupt
        return r
    monkeypatch.setattr(hs.signal, 'signal', flaky)
    try:
        sock = FakeSock()
        n = hs._abort_zeros('1.2.3.4', sock, lambda d: None)
        assert n == int(round(hs.ABORT_ZERO_S * hs.SEND_HZ))
        assert sg.getsignal(sg.SIGINT) == before
    finally:
        real(sg.SIGINT, before)


@pytest.mark.parametrize('where', ['settimeout', 'sendto'])
def test_second_ctrl_c_in_abort_zeros_keeps_the_zeros(where):
    """HCC6-4 (Codex): Ctrl-C again inside the socket setting of the abort path left 0 zeros and no goi0."""
    segs = hs.schedule('K2')
    k_move = hs.first_motion_k(segs)

    class Sock(FakeSock):
        hit = False

        def settimeout(self, v):
            if where == 'settimeout' and not self.hit:
                self.hit = True
                raise KeyboardInterrupt

        def sendto(self, pkt, addr):
            moved = any(p != (0.0, 0.0, 0.0) for p in self.sent)
            if where == 'sendto' and moved and decode(pkt) == (0.0, 0.0, 0.0) and not self.hit:
                self.hit = True
                raise KeyboardInterrupt
            super().sendto(pkt, addr)
    sock = Sock()
    clock = FakeClock(ctrl_c_at=k_move + 5)
    got = []
    with pytest.raises(hs.Abort):
        hs.send_schedule('1.2.3.4', segs, sock, got.append, clock, clock.sleep, lambda: 1.0e9)
    n0 = int(round(hs.ABORT_ZERO_S * hs.SEND_HZ))
    zeros_after = len(sock.sent) - [i for i, p in enumerate(sock.sent) if p != (0.0, 0.0, 0.0)][-1] - 1
    want = n0 if where == 'settimeout' else n0 - 1
    assert zeros_after == want and any(f'goi0={want}/{n0}' in ln for ln in got)
