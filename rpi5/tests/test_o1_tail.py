"""L85 Q1 (Liam 11/11, CODE local_821ee4dc; Fable O1/C6 review F-1, F-5, F-7, F-10): what the
vehicle and the HIL bench receive after the O1 command ends.

pc_sender --o1: every packet k (k/50 < T_cmd) is the fwsim setpoint sample of the same k, bit for
bit in float32; the zeros follow on the SAME 50 Hz grid (the first zero does not overwrite the last
schedule packet) and last at least t_late past T_cmd (fwsim holds the zero command to the deadline
T_cmd + t_late); the grid runs on a monotonic clock. run_hil --o1 --finite-run: stop_s covers
O1_TAIL_S. Every number below is from the generators; no measured data."""
import inspect
import math
import os
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'fwsim'))
import o1_tasks  # noqa: E402
import o1_feasibility  # noqa: E402
import pc_sender  # noqa: E402


class Clock:
    """Time advances only by sleep(); records the clock time of each sendto."""
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += max(d, 0.0)


def _run(task):
    clk, sent = Clock(), []

    class Sock:
        def sendto(self, pkt, addr):
            sent.append((clk.t, struct.unpack('<fff', pkt[1:13])))

    lines = []
    n = pc_sender.test_loop('10.0.0.1', task, sock=Sock(), clock=clk, sleep=clk.sleep, out=lines.append,
                            allow_final_set=True, wall=lambda: 1.7e9)
    t0 = sent[0][0]
    return n, [(t - t0, v) for t, v in sent], lines


@pytest.mark.parametrize('task', sorted(o1_tasks.FIG8))
def test_packets_are_the_fwsim_samples_bit_for_bit(task):
    sch = o1_feasibility.Schedule(o1_tasks.FIG8[task])
    n, sent, _ = _run(task)
    ks = [k for k in range(int(sch.Tcmd * 50) + 3) if k / 50 < sch.Tcmd]
    assert n == len(ks)                                   # F-5: no schedule sample with k/50 < T_cmd dropped
    for k in ks:
        assert sent[k][1] == tuple(float(np.float32(x)) for x in sch.setpoint(k / 50)), k


@pytest.mark.parametrize('task', list(o1_tasks.FIG8) + list(o1_tasks.FINAL_SET))
def test_zero_command_lasts_past_the_deadline_on_the_grid(task):
    T, _ = o1_tasks.task(task, allow_final_set=True)
    n, sent, lines = _run(task)
    tail = sent[n:]
    assert all(v == (0.0, 0.0, 0.0) for _, v in tail)
    assert tail[0][0] >= T - 1e-9                          # F-5: the first zero waits for its grid slot
    assert tail[0][0] - sent[n - 1][0] == pytest.approx(1 / 50)
    for j, (t, _) in enumerate(sent):
        assert t == pytest.approx(j / 50, abs=1e-9)        # one absolute grid, schedule and zeros
    assert tail[-1][0] >= T + o1_feasibility.T_LATE        # F-7: zeros reach the deadline T_cmd + t_late
    assert tail[-1][0] < T + o1_tasks.O1_TAIL_S
    assert f'zeros={len(tail)},tail_planned={len(tail)},sent={n}' in lines[-1] and lines[-1].endswith('complete=1')


def test_tail_is_t_late_plus_margin():
    assert o1_tasks.O1_TAIL_S == pytest.approx(o1_feasibility.T_LATE + 0.5)


def test_lifted_tests_keep_ten_zeros():
    """C5 runs (A-K*) are unchanged: 10 zero packets (0.2 s) after the schedule."""
    assert pc_sender.grid_counts('A-K1', 15.0) == (750, 10)


def test_grid_clock_is_monotonic_and_epoch_is_wall():
    """F-10: a wall-clock step cannot bunch or stall the packets; '#test_start' keeps the epoch."""
    sig = inspect.signature(pc_sender.test_loop)
    assert sig.parameters['clock'].default is time.monotonic
    assert sig.parameters['wall'].default is time.time
    _, _, lines = _run('O1-F05')
    assert 't_epoch=1700000000.000' in lines[0] and 't_epoch=1700000000.000' in lines[-1]


def test_interrupt_sends_zeros_at_once():
    clk, sent = Clock(), []

    class Sock:
        def sendto(self, pkt, addr):
            v = struct.unpack('<fff', pkt[1:13])
            sent.append(v)
            if len(sent) == 100:
                raise KeyboardInterrupt

    lines = []
    n = pc_sender.test_loop('10.0.0.1', 'O1-F05', sock=Sock(), clock=clk, sleep=clk.sleep, out=lines.append)
    assert n == 99 and '#test_interrupted' in lines
    assert sent[100:] == [(0.0, 0.0, 0.0)] * pc_sender.LIFTED_TAIL_ZEROS
    assert lines[-1].endswith('complete=0')


def test_run_hil_o1_stop_covers_the_tail():
    from run_hil import _finite_run_config
    from trajectory_pub import TrajectorySpec
    args = dict(steps=0, no_traj=False, traj_loop=False, fixed_omega=None, omega_steps=None,
                omega_schedule=None, load_schedule=None)
    spec = o1_tasks.make_spec('O1-F05')
    lc, ceiling = _finite_run_config(SimpleNamespace(o1='O1-F05', **args), spec)
    assert lc.stop_s == pytest.approx(o1_tasks.O1_TAIL_S) and lc.run_s == spec.duration
    assert ceiling / 1000 > lc.arm_timeout_s + lc.run_s + lc.run_grace_s + lc.stop_s
    lc2, _ = _finite_run_config(SimpleNamespace(o1=None, **args), TrajectorySpec('line', duration=3.0))
    assert lc2.stop_s == 1.0                                # every other finite run unchanged
