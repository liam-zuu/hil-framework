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
    assert lines[-1].startswith(f'#test_end,test={tid},') and lines[-1].endswith(f'sent={n},planned={n_plan},complete=1')


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
