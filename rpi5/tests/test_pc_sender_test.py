"""pc_sender.py --test A-K* (C5, Liam 10/11 "Duyet ca ba"): the schedule is trajectory_pub's,
sent on an absolute 50 Hz grid, then zeros; packets decode to the schedule values."""
import struct

import pytest

import pc_sender as ps
from trajectory_pub import GROUP_A, get_body_velocity


class FakeSock:
    def __init__(self):
        self.sent = []

    def sendto(self, pkt, addr):
        self.sent.append((pkt, addr))


class FakeClock:
    """Time advances only by sleep() plus a fixed cost per clock() call (send jitter)."""
    def __init__(self, t0=1000.0, cost=0.0003):
        self.t, self.cost = t0, cost

    def __call__(self):
        self.t += self.cost
        return self.t

    def sleep(self, d):
        self.t += max(d, 0.0)


def _decode(pkt):
    assert len(pkt) == 14 and pkt[0] == ps.TRAJ_START
    chk = 0
    for b in pkt[:13]:
        chk ^= b
    assert pkt[13] == chk
    return struct.unpack_from('<fff', pkt, 1)


@pytest.mark.parametrize('tid', ['A-K1', 'A-K2', 'A-K3'])
def test_schedule_is_trajectory_pub(tid):
    sock, clk, lines = FakeSock(), FakeClock(), []
    n = ps.test_loop('10.0.0.2', tid, sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    spec = GROUP_A[tid]
    n_plan = int(round(spec.duration * ps.SEND_HZ))
    assert n == n_plan
    body, tail = sock.sent[:n_plan], sock.sent[n_plan:]
    for k, (pkt, addr) in enumerate(body):
        assert addr == ('10.0.0.2', ps.TRAJ_PORT)
        assert _decode(pkt) == pytest.approx(get_body_velocity(k / ps.SEND_HZ, spec), abs=1e-6)
    assert len(tail) == 10 and all(_decode(p) == (0.0, 0.0, 0.0) for p, _ in tail)
    assert lines[0].startswith(f'#test_start,test={tid},') and 'duration_s=' in lines[0]
    assert lines[-1].startswith(f'#test_end,test={tid},')
    assert f'sent={n},planned={n_plan},complete=1,sched_first_s=' in lines[-1]


def _fields(line):
    return [p.split('=', 1)[0] for p in line.split(',')[1:]]


def test_test_end_keeps_the_seven_fields_then_the_two_send_times():
    """'#test_end': the 7 fields of 20f1f81 unchanged in name and order, then sched_first_s, sched_last_s
    (Liam 15/11 CODE_CAM "Duyet, lam khi xen duoc (Recommended)", format agreed with CODE 15/11)."""
    sock, clk, lines = FakeSock(), FakeClock(), []
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    assert _fields(lines[-1]) == ['test', 't_epoch', 'zeros', 'tail_planned', 'sent', 'planned', 'complete',
                                  'sched_first_s', 'sched_last_s']


def test_send_times_of_the_first_and_last_schedule_packets():
    """The two fields are clock() - t0 read just after sendto of packet 0 and of the last schedule packet,
    on the clock of the grid: a late sender shows in sched_last_s."""
    sock, clk, lines = FakeSock(), FakeClock(cost=0.0003), []
    times = []
    real_send = sock.sendto
    sock.sendto = lambda p, a: (times.append(clk.t), real_send(p, a))
    t_start = clk.t
    n = ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    kv = dict(p.split('=', 1) for p in lines[-1].split(',')[1:])
    first, last = float(kv['sched_first_s']), float(kv['sched_last_s'])
    t0 = t_start + clk.cost                             # first clock() call of test_loop
    assert first == pytest.approx(times[0] - t0, abs=2 * clk.cost)
    assert last == pytest.approx(times[n - 1] - t0, abs=2 * clk.cost)
    assert last - first == pytest.approx((n - 1) / ps.SEND_HZ, abs=0.01)
    assert len(kv['sched_first_s'].split('.')[1]) == 6


class StillClock:
    """Time moves only by sleep() and by what a test adds; reading it costs nothing (SS1-3)."""
    def __init__(self, t0=1000.0):
        self.t = t0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += max(d, 0.0)


def test_first_send_time_is_read_after_sendto():
    """sendto of packet 0 takes 50 ms: sched_first_s must be 0.050000 exactly (read after the send, SS1-3)."""
    sock, clk, lines = FakeSock(), StillClock(), []
    real_send = sock.sendto

    def send(p, a):
        if not sock.sent:
            clk.t += 0.05
        real_send(p, a)
    sock.sendto = send
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    kv = dict(p.split('=', 1) for p in lines[-1].split(',')[1:])
    assert kv['sched_first_s'] == '0.050000'


def test_one_packet_schedule_has_one_send_time(monkeypatch):
    """n_total = 1: both fields are the same reading (SS1-1); n_total = 0: both empty."""
    for n_total, want_equal in ((1, True), (0, None)):
        monkeypatch.setattr(ps, 'test_schedule', lambda tid, allow=False: (0.02, lambda t: (0.1, 0.0, 0.0)))
        monkeypatch.setattr(ps, 'grid_counts', lambda tid, d, n=n_total: (n, 2))
        sock, clk, lines = FakeSock(), FakeClock(cost=0.0003), []
        ps.test_loop('10.0.0.2', 'X', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
        kv = dict(p.split('=', 1) for p in lines[-1].split(',')[1:])
        if want_equal:
            assert kv['sched_first_s'] != '' and kv['sched_first_s'] == kv['sched_last_s']
        else:
            assert kv['sched_first_s'] == '' and kv['sched_last_s'] == ''


def test_interrupt_in_the_zero_tail_keeps_both_send_times():
    sock, clk, lines = FakeSock(), FakeClock(), []
    real_send = sock.sendto
    n_plan = int(round(GROUP_A['A-K1'].duration * ps.SEND_HZ))
    hit = []

    def send(p, a):
        if len(sock.sent) == n_plan + 3 and not hit:
            hit.append(1)
            raise KeyboardInterrupt
        real_send(p, a)
    sock.sendto = send
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    kv = dict(p.split('=', 1) for p in lines[-1].split(',')[1:])
    assert '#test_interrupted' in lines and kv['complete'] == '1'
    assert kv['sched_first_s'] != '' and kv['sched_last_s'] != ''


def test_late_sender_shows_in_the_last_send_time():
    """A sender stalled 0.3 s just before the last schedule packet: sched_last_s is the real (late) time,
    not the grid time; a stall mid schedule is caught up by the absolute grid (burst) and does not show."""
    sock, clk, lines = FakeSock(), FakeClock(cost=0.0003), []
    real_send = sock.sendto
    n_plan = int(round(GROUP_A['A-K1'].duration * ps.SEND_HZ))

    def send(p, a):
        if len(sock.sent) == n_plan - 1:
            clk.t += 0.3
        real_send(p, a)
    sock.sendto = send
    n = ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    kv = dict(p.split('=', 1) for p in lines[-1].split(',')[1:])
    assert n == n_plan
    assert float(kv['sched_last_s']) - float(kv['sched_first_s']) >= (n - 1) / ps.SEND_HZ + 0.29


def test_interrupted_run_leaves_the_last_send_time_empty():
    sock, clk, lines = FakeSock(), FakeClock(), []
    real_send = sock.sendto

    hit = []

    def send(p, a):
        if len(sock.sent) == 50 and not hit:
            hit.append(1)
            raise KeyboardInterrupt
        real_send(p, a)
    sock.sendto = send
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    kv = dict(p.split('=', 1) for p in lines[-1].split(',')[1:])
    assert '#test_interrupted' in lines and kv['complete'] == '0'
    assert kv['sched_first_s'] != '' and kv['sched_last_s'] == ''


def test_absolute_grid_does_not_drift():
    """With a send cost per packet the k-th packet still leaves at t0 + k/50 (+ one cost)."""
    sock, clk = FakeSock(), FakeClock(cost=0.004)
    times = []
    real_send = sock.sendto
    sock.sendto = lambda p, a: (times.append(clk.t), real_send(p, a))
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lambda s: None)
    t0 = times[0]
    for k in (50, 200, 449):
        assert times[k] - t0 == pytest.approx(k / ps.SEND_HZ, abs=0.01)


def test_k1_has_the_expected_levels():
    """A-K1: vx 0 -> 0.3 (3-6 s) -> 0, vy = wz = 0 (03_CODE/docs C5 paper run, section 2)."""
    sock, clk = FakeSock(), FakeClock()
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lambda s: None)
    v = [_decode(p) for p, _ in sock.sent]
    assert v[50] == (0.0, 0.0, 0.0) and v[200][0] == pytest.approx(0.3) and v[400] == (0.0, 0.0, 0.0)
    assert all(abs(x[1]) < 1e-9 and abs(x[2]) < 1e-9 for x in v)


def test_only_lifted_tests():
    with pytest.raises(ValueError):
        ps.test_schedule('A-L1')



# R134 package (S05 4.7 (iv), R3V-9): --gain-query-dir with --o1
ANSWER = '\n'.join(['#gain_running,boot=4753dc47,id=0', '#ket_qua,ma=0', ''])


def test_gain_query_file_saves_the_whole_answer(tmp_path):
    from types import SimpleNamespace
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=ANSWER, stderr='', returncode=0)
    p = tmp_path / 'gain_query_before.txt'
    assert ps.gain_query_file('192.168.1.69', str(p), run=run) == 0
    assert calls[0][1:] == [ps.GAIN_SEND, '--ip', '192.168.1.69', '--query']
    txt = p.read_text(encoding='utf-8')
    assert 'boot=4753dc47' in txt and txt.startswith('#query_pc,t_start=')         # R134C2-1: PC times first
    with pytest.raises(FileExistsError):
        ps.gain_query_file('192.168.1.69', str(p), run=run)


def test_gain_query_file_records_a_failure(tmp_path):
    import subprocess

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 20)
    p = tmp_path / 'q.txt'
    assert ps.gain_query_file('1.2.3.4', str(p), run=slow) == 1
    assert '#ket_qua,ma=1' in p.read_text(encoding='utf-8')


@pytest.mark.parametrize('present', [(), ('gain_query_before.txt',), ('gain_query_after.txt',),
                                     ('gain_query_before.txt', 'gain_query_after.txt')])
def test_query_dir_with_a_file_of_an_earlier_run_is_refused(tmp_path, present):
    """R134C-11: an 'after' (or 'before') already in the folder refuses the run before #test_start."""
    for n in present:
        (tmp_path / n).write_text(ANSWER, encoding='utf-8')
    bad = ps.query_dir_problem(str(tmp_path))
    assert (bad is None) == (not present) and all(n in (bad or '') for n in present)


# ── B8-1 B81-14: --gain-file (Liam 14/11 CODE_R134 "Cong cu tu dat va luu (Recommended)") ──

ACK = '\n'.join(['# send 1.2.3.4:5014 #gain_set,v=1,tx=<new each try>,id=7,ctrl=PID',
                 '# ack tx=9,req_id=7,ok=1,err=,sha=16702ff0e688,env=pid,mac=aa:bb,boot=4753dc47,id=7,applied_n=1,ctrl=PID',
                 '#ket_qua,ma=0', ''])


def test_gain_set_file_saves_the_whole_output(tmp_path):
    from types import SimpleNamespace
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=ACK, stderr='', returncode=0)
    p = tmp_path / ps.ACK_NAME
    assert ps.gain_set_file('1.2.3.4', 'g.json', str(p), run=run) == 0
    assert calls[0][1:] == [ps.GAIN_SEND, '--ip', '1.2.3.4', '--file', 'g.json']
    assert p.read_text(encoding='utf-8') == ACK
    with pytest.raises(FileExistsError):                          # never overwrites a record
        ps.gain_set_file('1.2.3.4', 'g.json', str(p), run=run)

    def refused(cmd, **kw):
        return SimpleNamespace(stdout='# rejected: err=motor_on\n#ket_qua,ma=2\n', stderr='', returncode=2)
    assert ps.gain_set_file('1.2.3.4', 'g.json', str(tmp_path / 'b.txt'), run=refused) == 2


def test_query_dir_refuses_an_old_ack_with_gain_file(tmp_path):
    (tmp_path / ps.ACK_NAME).write_text(ACK, encoding='utf-8')
    assert ps.query_dir_problem(str(tmp_path)) is None                 # without --gain-file: not looked at
    assert ps.ACK_NAME in ps.query_dir_problem(str(tmp_path), with_ack=True)


def test_c4_5_ack_before_problem(tmp_path):
    """R134C4-5: the car's ack is compared with the question just before the task."""
    before = ('#query_pc,t_start=1.0,t_end=2.0\n'
              + ANSWER.replace('boot=4753dc47', 'boot=4753dc47,mac=aa:bb').replace('id=0', 'id=7'))
    assert 'no readable answer' in ps.ack_before_problem(str(tmp_path))
    (tmp_path / 'gain_query_before.txt').write_text(before, encoding='utf-8')
    assert 'thieu cau tra loi #gain_ack' in ps.ack_before_problem(str(tmp_path))
    (tmp_path / ps.ACK_NAME).write_text(ACK, encoding='utf-8')
    assert ps.ack_before_problem(str(tmp_path)) is None
    (tmp_path / ps.ACK_NAME).write_text(ACK.replace('4753dc47', '0badf00d'), encoding='utf-8')
    assert 'khoi dong lai sau khi dat gain' in ps.ack_before_problem(str(tmp_path))
    (tmp_path / ps.ACK_NAME).write_text(ACK, encoding='utf-8')     # R134C5-4: another set runs (id 8) since the ack
    (tmp_path / 'gain_query_before.txt').write_text(before.replace('id=7', 'id=8'), encoding='utf-8')
    assert 'bo gain khac' in ps.ack_before_problem(str(tmp_path))
    # R134C6-2: the ack and the question before both without id, or both with an empty id: no match
    for gone, empty in (('', ''), (',id=', ',id=')):
        (tmp_path / ps.ACK_NAME).write_text(ACK.replace(',id=7,applied_n', gone + ',applied_n'), encoding='utf-8')
        (tmp_path / 'gain_query_before.txt').write_text(before.replace(',id=7', empty), encoding='utf-8')
        assert 'khong so duoc bo gain' in ps.ack_before_problem(str(tmp_path)), (gone, empty)


def test_test_start_keeps_its_fields_then_t_mono_ns_last():
    """'#test_start': the fields before unchanged in name and order, t_mono_ns appended last (Liam 23/11 CODE
    local_be52def9 "Chot + gioi han + t_mono (Recommended)"), so a reader by key or by position still works."""
    sock, clk, lines = FakeSock(), FakeClock(), []
    t_start = clk.t
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    assert _fields(lines[0]) == ['test', 't_epoch', 'iso', 'duration_s', 'send_hz', 'tail_zeros', 'dst',
                                 't_mono_ns']
    # the reading of the grid origin itself (first clock() call), not a later one
    assert int(lines[0].rsplit('t_mono_ns=', 1)[1]) == int((t_start + clk.cost) * 1e9)


def test_t_mono_ns_is_the_grid_origin_and_no_packet_is_sent_before_it():
    """t_mono_ns = floor(t0 * 1e9), t0 the first clock() reading of test_loop (the grid origin): an integer,
    on the grid clock, and every packet (schedule and zeros) is sent at or after it."""
    # cost 0 and t0 0.7 ns past an integer ns: packet 0 goes at t0 itself, so rounding up or to nearest is caught
    sock, clk, lines = FakeSock(), FakeClock(t0=1000.0000000007, cost=0.0), []
    times = []
    real_send = sock.sendto
    sock.sendto = lambda p, a: (times.append(clk.t), real_send(p, a))
    t_start = clk.t
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    kv = dict(p.split('=', 1) for p in lines[0].split(',')[1:])
    assert kv['t_mono_ns'].isdigit()
    t_mono_ns = int(kv['t_mono_ns'])
    t0 = t_start                                        # first clock() call of test_loop (cost 0)
    assert 0.6 < t0 * 1e9 % 1 < 0.8                     # the case really has 0.7 ns past an integer
    assert t_mono_ns == int(t0 * 1e9)
    assert t_mono_ns <= t0 * 1e9
    assert times and min(times) * 1e9 >= t_mono_ns


def test_t_mono_ns_is_not_the_wall_clock():
    """A wall clock far from the grid clock: t_mono_ns follows the grid clock only (t_epoch keeps the wall)."""
    sock, clk, lines = FakeSock(), FakeClock(t0=50.0, cost=0.0), []
    ps.test_loop('10.0.0.2', 'A-K1', sock=sock, clock=clk, sleep=clk.sleep, out=lines.append,
                 wall=lambda: 1700000000.0)
    kv = dict(p.split('=', 1) for p in lines[0].split(',')[1:])
    assert kv['t_mono_ns'] == str(50 * 10**9) and kv['t_epoch'] == '1700000000.000'
