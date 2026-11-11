"""rpi5/o1_tasks.py (C6, L76): the figure 8 is the S03/S04 schedule of fwsim/o1_feasibility value
for value; integrating each task's body-velocity command gives its path (figure 8 back on the
origin, K1 2 m to the left, K2 the 1 m square back on the start corner); K1, K2 are refused
without allow_final_set; install() leaves every other trajectory unchanged; pc_sender --o1 sends
the schedule on its 50 Hz grid."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'fwsim'))
import o1_tasks  # noqa: E402
import o1_feasibility  # noqa: E402
import trajectory_pub  # noqa: E402
import pc_sender  # noqa: E402


def integrate(f, T, head0, dt=0.001):
    """Pose (x, y, th) on the floor from body-velocity commands, held over each dt (midpoint)."""
    x = y = 0.0
    th = head0
    for k in range(int(round(T / dt)) + 1):
        vx, vy, wz = f((k + 0.5) * dt)
        thm = th + wz * dt / 2.0
        c, s = math.cos(thm), math.sin(thm)
        x += (c * vx - s * vy) * dt
        y += (s * vx + c * vy) * dt
        th += wz * dt
    return x, y, th


@pytest.mark.parametrize('tid, v', [('O1-F02', 0.2), ('O1-F05', 0.5), ('O1-F08', 0.8)])
def test_figure8_is_the_s03_schedule(tid, v):
    T, f = o1_tasks.task(tid)
    sch = o1_feasibility.Schedule(v)
    assert T == sch.Tcmd
    for t in np.linspace(-0.5, sch.Tcmd + 1.0, 4001):
        assert f(float(t)) == sch.setpoint(float(t))


@pytest.mark.parametrize('tid, tcmd', [('O1-F02', 31.07), ('O1-F05', 13.65), ('O1-F08', 9.95)])
def test_figure8_duration_matches_o1_table(tid, tcmd):
    assert o1_tasks.task(tid)[0] == pytest.approx(tcmd, abs=0.006)     # O1 v11 section 3.1 table


@pytest.mark.parametrize('tid', ['O1-F02', 'O1-F05', 'O1-F08'])
def test_figure8_command_closes_on_the_origin(tid):
    T, f = o1_tasks.task(tid)
    x, y, th = integrate(f, T, math.pi / 4)
    assert math.hypot(x, y) < 0.005
    assert abs(math.remainder(th - math.pi / 4, 2 * math.pi)) < 0.01


def test_k1_strafes_2m_left():
    T, f = o1_tasks.task('O1-K1', allow_final_set=True)
    assert T == pytest.approx(5.0)
    ts = np.arange(0.0, T + 0.5, 0.01)
    v = np.array([f(t) for t in ts])
    assert np.all(v[:, 0] == 0.0) and np.all(v[:, 2] == 0.0)
    assert v[:, 1].max() == pytest.approx(0.5) and v[:, 1].min() >= 0.0
    x, y, th = integrate(f, T, 0.0)
    assert x == pytest.approx(0.0, abs=1e-9) and y == pytest.approx(2.0, abs=1e-3) and th == 0.0


def test_k2_square_back_on_the_start_corner():
    T, f = o1_tasks.task('O1-K2', allow_final_set=True)
    assert T == pytest.approx(15.0)
    corners = []
    for i in range(4):
        t_end = (i + 1) * 3.0 + i * 1.0
        corners.append(integrate(f, t_end, 0.0)[:2])
        assert f(t_end + 0.5) == (0.0, 0.0, 0.0) or i == 3          # command 0 during t_g
    expect = [(1, 0), (1, 1), (0, 1), (0, 0)]
    for (x, y), (ex, ey) in zip(corners, expect):
        assert x == pytest.approx(ex, abs=2e-3) and y == pytest.approx(ey, abs=2e-3)
    assert all(f(t)[2] == 0.0 for t in np.arange(0, T, 0.01))         # no body rotation
    assert all(abs(f(t)[0]) <= 0.5 and abs(f(t)[1]) <= 0.5 for t in np.arange(0, T, 0.01))


def test_segment_profile():
    assert o1_tasks.segment_time(1.0) == pytest.approx(3.0)
    assert o1_tasks.segment_speed(0.5, 1.0) == pytest.approx(0.25)     # a_r t
    assert o1_tasks.segment_speed(1.5, 1.0) == pytest.approx(0.5)
    assert o1_tasks.segment_speed(2.5, 1.0) == pytest.approx(0.25)
    assert o1_tasks.segment_speed(3.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        o1_tasks.segment_time(0.4)


@pytest.mark.parametrize('tid', ['O1-K1', 'O1-K2'])
def test_final_set_needs_the_flag(tid):
    with pytest.raises(ValueError, match='final test set'):
        o1_tasks.task(tid)
    with pytest.raises(ValueError, match='final test set'):
        o1_tasks.make_spec(tid)
    assert o1_tasks.make_spec(tid, allow_final_set=True).duration > 0


def test_unknown_task():
    with pytest.raises(ValueError):
        o1_tasks.task('O1-F03')


def test_install_wraps_only_o1_specs():
    class M:                                   # a stand-in module, trajectory_pub itself untouched
        get_body_velocity = staticmethod(trajectory_pub.get_body_velocity)
    o1_tasks.install(M)
    o1_tasks.install(M)                        # idempotent
    for tid, spec in trajectory_pub.GROUP_A.items():
        for t in (0.0, 1.3, 4.0, 7.7):
            assert M.get_body_velocity(t, spec) == trajectory_pub.get_body_velocity(t, spec), tid
    spec = o1_tasks.make_spec('O1-F05')
    sch = o1_feasibility.Schedule(0.5)
    for t in (0.0, 0.7, 5.0, 13.0, 14.0):
        assert M.get_body_velocity(t, spec) == sch.setpoint(t)
    assert spec.duration == sch.Tcmd and spec.kind == 'o1'


def test_pc_sender_o1_sends_the_schedule():
    sent, lines = [], []

    class Sock:
        def sendto(self, pkt, addr):
            sent.append(pkt)

    clock = [1000.0]
    n = pc_sender.test_loop('10.0.0.1', 'O1-F05', sock=Sock(), clock=lambda: clock[0],
                            sleep=lambda s: clock.__setitem__(0, clock[0] + s), out=lines.append)
    T = o1_feasibility.Schedule(0.5).Tcmd
    # L85 Q1: every k with k/50 < T_cmd, then zeros on the grid up to T_cmd + O1_TAIL_S (was: 10 zeros)
    assert n == math.ceil(T * pc_sender.SEND_HZ - 1e-9)
    assert len(sent) == math.ceil((T + o1_tasks.O1_TAIL_S) * pc_sender.SEND_HZ - 1e-9)
    assert lines[0].startswith('#test_start,test=O1-F05') and 'complete=1' in lines[-1]
    k = 300
    import struct
    vx, vy, wz = struct.unpack('<fff', sent[k][1:13])
    ex = o1_feasibility.Schedule(0.5).setpoint(k / pc_sender.SEND_HZ)
    assert (vx, vy, wz) == pytest.approx(ex, rel=1e-6, abs=1e-7)


def test_pc_sender_refuses_final_set_without_flag():
    with pytest.raises(ValueError, match='final test set'):
        pc_sender.test_loop('10.0.0.1', 'O1-K2', sock=object(), clock=lambda: 0.0, sleep=lambda s: None,
                            out=lambda s: None)
