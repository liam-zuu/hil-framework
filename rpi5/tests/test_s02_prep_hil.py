"""run_hil.py --s02-prep (CODE_S02 design 3.2): the publisher of a finite run, started at RUN, writes 0 for
the preparation then the test shifted by it; the flag needs --test and --finite-run."""
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time

import pytest

import s02_sender as s2
from run_hil import _finite_run_config
from run_lifecycle import RUN, ControlBox
from trajectory_pub import TrajectoryPublisher, TrajectorySpec

HERE = os.path.dirname(os.path.abspath(__file__))
RPI5 = os.path.join(HERE, '..')


def _no_signal_run(pub):
    import trajectory_pub as tp
    real, tp.signal.signal = tp.signal.signal, lambda *a: None
    try:
        pub.run()
    finally:
        tp.signal.signal = real


def test_publisher_sends_zero_through_the_preparation_then_the_test():
    box = ControlBox()
    box.set_phase(RUN)
    spec = s2.prep_spec(TrajectorySpec('line', v=0.3, duration=0.3), 0.3, tail_s=0.2)
    ref = mp.Array('d', 4)
    seen = []                                       # (time after start, vx) at every new setpoint
    pub = TrajectoryPublisher(spec=spec, simulate=True, shared_ref=ref, control=box)
    th = threading.Thread(target=_no_signal_run, args=(pub,), daemon=True)
    t0 = time.monotonic()
    th.start()
    last = 0.0
    while th.is_alive() and time.monotonic() - t0 < 3.0:
        with ref.get_lock():
            vx, cnt = ref[0], ref[3]
        if cnt != last:
            seen.append((time.monotonic() - t0, vx))
            last = cnt
        time.sleep(0.0005)
    th.join(1.0)
    assert not th.is_alive()
    zeros = [t for t, v in seen if v == 0.0]
    ones = [t for t, v in seen if v == pytest.approx(0.3)]
    assert zeros and ones and len(zeros) + len(ones) == len(seen)
    prep_z = [t for t in zeros if t < min(ones)]
    tail_z = [t for t in zeros if t > max(ones)]
    assert prep_z and tail_z and len(prep_z) + len(tail_z) == len(zeros)   # 0, then the test, then 0
    assert min(ones) >= 0.3 - 0.04                  # Windows sleep / monotonic granularity
    assert max(ones) <= 0.6 + 0.04 and min(tail_z) <= 0.6 + 0.06            # S2P1-7: the tail on time
    assert 0.8 - 0.05 <= seen[-1][0] <= 0.8 + 0.1


def test_finite_run_lasts_preparation_plus_test():
    from types import SimpleNamespace
    args = SimpleNamespace(steps=0, no_traj=False, traj_loop=False, fixed_omega=None, omega_steps=None,
                           omega_schedule=None, load_schedule=None, o1=None)
    spec = s2.prep_spec(TrajectorySpec('lift_reverse', v=0.3, duration=9.0), 3.0)
    lc, _ = _finite_run_config(args, spec)
    assert lc.run_s == pytest.approx(13.0) and lc.stop_s == 1.0            # 3 + 9 + 1 s tail (S2P1-7)


@pytest.mark.parametrize('extra,msg', [
    (['--test', 'A-K2', '--s02-prep', '3'], '--s02-prep needs --test and --finite-run'),
    (['--traj', 'line', '--duration', '3', '--finite-run', '--s02-prep', '3'], '--s02-prep needs --test'),
    (['--test', 'A-K2', '--finite-run', '--s02-prep', '3.01'], 'multiple of'),
])
def test_bad_combinations_are_refused_before_any_process(extra, msg):
    p = subprocess.run([sys.executable, 'run_hil.py', '--mock'] + extra, cwd=RPI5, capture_output=True,
                       text=True, timeout=60)
    assert p.returncode != 0 and msg in p.stderr
