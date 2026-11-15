"""s02_sender.py (CODE_S02 design sections 3.1, 3.2): preparation zeros, the schedule from k = 0, 1 s of
zeros, all on one absolute 50 Hz grid; the packet is pc_sender's; prep_spec for run_hil --s02-prep."""
import hashlib
import os
import struct
import subprocess

import pytest

import pc_sender as ps
import s02_sender as s2
from trajectory_pub import GROUP_A, get_body_velocity

HERE = os.path.dirname(os.path.abspath(__file__))


class FakeSock:
    def __init__(self, clk=None, stop_at=None):
        self.sent, self.clk, self.stop_at = [], clk, stop_at

    def sendto(self, pkt, addr):
        if self.stop_at is not None and len(self.sent) == self.stop_at:
            self.stop_at = None
            raise KeyboardInterrupt
        self.sent.append((pkt, addr, None if self.clk is None else self.clk.t))


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


def _run(tid, prep_s=3.0, stop_at=None, allow_strafe=True):
    clk = FakeClock()
    sock, lines = FakeSock(clk, stop_at), []
    t_start = clk.t
    r = s2.s02_loop('10.0.0.2', tid, prep_s, allow_strafe, sock=sock, clock=clk, sleep=clk.sleep,
                    out=lines.append, wall=lambda: 1.7e9)
    return r, sock.sent, lines, t_start


@pytest.mark.parametrize('tid', sorted(GROUP_A))
def test_prep_schedule_tail(tid):
    r, sent, lines, _ = _run(tid)
    spec = GROUP_A[tid]
    P, K = 150, s2.sched_count(spec)
    assert (r['prep'], r['sched'], r['tail'], r['complete']) == (P, K, 50, 1)
    assert len(sent) == P + K + 50
    assert all(_decode(p) == (0.0, 0.0, 0.0) for p, _, _ in sent[:P])
    for k in range(K):
        assert _decode(sent[P + k][0]) == pytest.approx(get_body_velocity(k / 50, spec), abs=1e-6)
    assert all(_decode(p) == (0.0, 0.0, 0.0) for p, _, _ in sent[P + K:])
    assert all(a == ('10.0.0.2', ps.TRAJ_PORT) for _, a, _ in sent)


def test_packet_bytes_are_pc_sender_packets():
    _, sent, _, _ = _run('A-F2')
    spec = GROUP_A['A-F2']
    for k in (0, 7, 300):
        assert sent[150 + k][0] == ps.pack_packet(*get_body_velocity(k / 50, spec))


def test_grid_is_absolute():
    """Packet j leaves at or just after t_start + j/50 (clock cost only), never drifting."""
    _, sent, _, t_start = _run('A-K1')
    for j, (_, _, t) in enumerate(sent):
        assert 0.0 <= t - (t_start + j / 50) < 0.002


def test_k0_line_and_end_fields():
    r, sent, lines, _ = _run('A-K2')
    assert lines[0].startswith('#s02_start,test=A-K2,') and 'prep_s=3,n_prep=150,n_sched=450,n_tail=50' in lines[0]
    assert lines[1] == '#s02_k0,t_epoch=1700000000.000'
    end = dict(p.split('=', 1) for p in lines[-1].split(',')[1:])
    assert (end['sent_prep'], end['sent_sched'], end['sent_tail'], end['complete']) == ('150', '450', '50', '1')
    assert float(end['sched_first_s']) == pytest.approx(150 / 50, abs=0.002)
    assert float(end['sched_last_s']) == pytest.approx((150 + 449) / 50, abs=0.002)


@pytest.mark.parametrize('stop_at,phase,k0', [(10, 'prep', False), (200, 'sched', True), (150 + 450 + 5, 'tail', True)])
def test_ctrl_c_sends_zeros(stop_at, phase, k0):
    r, sent, lines, _ = _run('A-K2', stop_at=stop_at)
    assert r['complete'] == 0
    assert f'#s02_interrupted,phase={phase}' in lines
    assert any(l.startswith('#s02_k0') for l in lines) == k0
    assert len(sent) == stop_at + s2.STOP_ZEROS
    assert all(_decode(p) == (0.0, 0.0, 0.0) for p, _, _ in sent[-s2.STOP_ZEROS:])
    assert 'complete=0' in lines[-1]


def test_strafe_needs_flag():
    with pytest.raises(ValueError, match='allow-strafe'):
        _run('A-S1', allow_strafe=False)
    with pytest.raises(ValueError, match='not a group A'):
        s2.check_test('A-X9')


@pytest.mark.parametrize('bad', [0.0, -1.0, 3.01, float('nan')])
def test_prep_must_be_on_the_grid(bad):
    with pytest.raises(ValueError, match='prep-s'):
        s2.prep_packets(bad)


@pytest.mark.parametrize('tid', sorted(GROUP_A))
def test_prep_spec_is_shifted_schedule(tid):
    spec = GROUP_A[tid]
    w = s2.prep_spec(spec, 3.0)
    assert w.duration == pytest.approx(3.0 + spec.duration + 1.0)          # S2P1-7: 1 s tail inside
    for t in (3.0 + spec.duration + 0.01, 3.0 + spec.duration + 0.99):
        assert get_body_velocity(t, w) == (0.0, 0.0, 0.0)
    for i in range(0, 150):
        assert get_body_velocity(i / 50 + 0.0037, w) == (0.0, 0.0, 0.0)
    for k in range(0, s2.sched_count(spec), 7):
        t = k / 50 + 0.0037
        if t < spec.duration:
            assert get_body_velocity(3.0 + t, w) == pytest.approx(get_body_velocity(t, spec), abs=1e-12)


def test_trajectory_pub_is_not_changed():
    """trajectory_pub.py is in the S04 blob lock (fwsim/s04_ga.py SOURCES): this branch must not change it."""
    out = subprocess.run(['git', 'diff', '--name-only', '68701b4', '--', 'trajectory_pub.py'],
                         cwd=os.path.join(HERE, '..'), capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip('no git history here')
    assert out.stdout.strip() == ''


def test_main_refuses_strafe_without_flag(capsys):
    with pytest.raises(SystemExit):
        s2.main(['--ip', '10.0.0.2', '--test', 'A-S2'])
    assert 'allow-strafe' in capsys.readouterr().err


def test_stall_in_the_preparation_stops_the_run():
    """S2P1-6: the clock jumps 4 s after packet 3 (a blocked process): the late packets are not sent in a
    burst; the run stops with zeros and complete=0."""
    clk = FakeClock()
    sock, lines = FakeSock(clk), []
    real = sock.sendto

    def sendto(p, a):
        real(p, a)
        if len(sock.sent) == 4:
            clk.t += 4.0
    sock.sendto = sendto
    r = s2.s02_loop('10.0.0.2', 'A-L2', 3.0, sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    assert r['complete'] == 0 and r['prep'] == 4 and r['sched'] == 0
    assert any(l.startswith('#s02_late,phase=prep,packet=4') for l in lines)    # S2P2-5: caught before sendto
    assert not any(l.startswith('#s02_k0') for l in lines)
    assert len(sock.sent) == 4 + s2.STOP_ZEROS


def test_start_line_delay_does_not_compress_the_grid():
    """S2P1-6: a slow start line (stdout blocked 4 s) comes before t0, so the preparation still lasts 3 s."""
    clk = FakeClock()
    sock, lines = FakeSock(clk), []

    def out(line):
        lines.append(line)
        if line.startswith('#s02_start'):
            clk.t += 4.0
    r = s2.s02_loop('10.0.0.2', 'A-L2', 3.0, sock=sock, clock=clk, sleep=clk.sleep, out=out)
    assert r['complete'] == 1
    ts = [t for _, _, t in sock.sent]
    assert ts[150] - ts[0] == pytest.approx(3.0, abs=0.002)


@pytest.mark.parametrize('when', ['#s02_start', '#s02_k0'])
def test_ctrl_c_in_a_print(when):
    """S2P1-12: Ctrl-C inside a print: counters match the socket, '#s02_end' is printed, zeros sent."""
    clk = FakeClock()
    sock, lines = FakeSock(clk), []

    def out(line):
        lines.append(line)
        if line.startswith(when):
            raise KeyboardInterrupt
    r = s2.s02_loop('10.0.0.2', 'A-K2', 3.0, sock=sock, clock=clk, sleep=clk.sleep, out=out)
    assert r['complete'] == 0 and lines[-1].startswith('#s02_end')
    assert r['prep'] + r['sched'] + r['tail'] == len(sock.sent) - s2.STOP_ZEROS
    if when == '#s02_k0':
        assert r['sched'] == 1


def _stall_after(n_sent, jump=4.0):
    """s02_loop on A-L2 where the clock jumps by jump s right after the n_sent-th sendto."""
    clk = FakeClock()
    sock, lines = FakeSock(clk), []
    real = sock.sendto

    def sendto(p, a):
        real(p, a)
        if len(sock.sent) == n_sent:
            clk.t += jump
    sock.sendto = sendto
    r = s2.s02_loop('10.0.0.2', 'A-L2', 3.0, sock=sock, clock=clk, sleep=clk.sleep, out=lines.append)
    return r, sock, lines


def test_stall_before_the_first_schedule_packet_sends_nothing_late():
    """S2P2-5 (Codex round 2): stalled right before k = 0: the late non-zero packet is not sent (checked before
    sendto), sent_sched 0, no '#s02_k0', sched_first_s empty, stop zeros."""
    r, sock, lines = _stall_after(150)
    assert r['complete'] == 0 and r['prep'] == 150 and r['sched'] == 0
    assert any(l.startswith('#s02_late,phase=sched,packet=150') for l in lines)
    assert not any(l.startswith('#s02_k0') for l in lines)
    assert 'sched_first_s=,' in lines[-1]
    assert all(_decode(p) == (0.0, 0.0, 0.0) for p, _, _ in sock.sent)


def test_stall_after_the_last_packet_keeps_a_complete_run():
    """S2P2-5: every packet sent on time, then the process stalls before the clock read of the last one: the
    run is complete, no extra zeros."""
    total = 150 + s2.sched_count(GROUP_A['A-L2']) + s2.TAIL_ZEROS
    r, sock, lines = _stall_after(total)
    assert r['complete'] == 1 and len(sock.sent) == total and r['late_after_n'] == 0
    assert not any(l.startswith('#s02_late') for l in lines)


def test_stall_after_a_schedule_sendto_is_reported_not_stopped():
    """S2P2-5: a stall between a schedule sendto and its clock read: counted in late_after_n, the run goes on;
    the next packet is checked before its sendto (here: the grid is 4 s behind, so it stops there)."""
    r, sock, lines = _stall_after(160, jump=0.060)          # read 60 ms late; the next slot is then 40 ms late
    assert r['late_after_n'] == 1 and 'late_after_n=1' in lines[-1] and r['complete'] == 1
    assert not any(l.startswith('#s02_late') for l in lines)
    r, sock, lines = _stall_after(160, jump=0.03 + s2.LATE_MAX_S)
    assert r['late_after_n'] == 1
    assert r['complete'] == 0 and any(l.startswith('#s02_late,phase=sched,packet=160') for l in lines)
    r, sock, lines = _stall_after(160, jump=0.0)
    assert r['late_after_n'] == 0 and r['complete'] == 1
