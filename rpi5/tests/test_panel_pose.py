"""Exercise run_hil's actual mock plant, publisher and owner on the PC."""
import csv
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading

import pytest

RPI = Path(__file__).resolve().parents[1]


def test_pose_uses_raw_state_and_closes_on_owner_error(tmp_path):
    from run_hil import _pose_log

    sampled = threading.Event()

    class State:
        def get_obj(self):
            return (1.5, -2.0, .75)

    class Control:
        def phase(self):
            sampled.set()
            return 'RUN'

    path = tmp_path / 'pose.csv'
    with pytest.raises(RuntimeError, match='owner error'):
        with _pose_log(path, State(), Control()):
            assert sampled.wait(2)
            raise RuntimeError('owner error')
    with path.open(newline='') as f:
        row = next(csv.DictReader(f))
    assert (row['phase'], row['x'], row['y'], row['th']) == ('RUN', '1.5', '-2.0', '0.75')


@pytest.mark.parametrize('enabled', [False, True], ids=['off', 'on'])
def test_mock_pose_log(tmp_path, enabled):
    # run_hil pins each spawned child before importing the plant/publisher. Windows
    # lacks that syscall; some Linux CI cpusets also exclude Pi cores 0..3.
    # sitecustomize reaches spawned interpreters too. Older Windows Python uses
    # GetTickCount64 for monotonic (typically 15.6 ms), coarser than the 10 ms
    # H7 freshness gate. Request 1 ms wakeups and use QPC for real elapsed time.
    (tmp_path / 'sitecustomize.py').write_text(
        "import os, sys, threading, time\n"
        "if sys.platform == 'win32':\n"
        "    import ctypes, atexit\n"
        "    if ctypes.windll.winmm.timeBeginPeriod(1) == 0:\n"
        "        atexit.register(ctypes.windll.winmm.timeEndPeriod, 1)\n"
        "    time.monotonic, time.monotonic_ns = time.perf_counter, time.perf_counter_ns\n"
        "if not hasattr(os, 'sched_getaffinity') or not set(range(4)) <= os.sched_getaffinity(0):\n"
        "    os.sched_setaffinity = lambda pid, cores: None\n"
        "if os.environ['TEST_NO_PANEL_POSE'] == '1':\n"
        "    start = threading.Thread.start\n"
        "    def checked_start(thread):\n"
        "        assert thread.name != 'panel-pose', 'sampler started without --pose-log'\n"
        "        return start(thread)\n"
        "    threading.Thread.start = checked_start\n", encoding='ascii')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), os.environ.get('PYTHONPATH', ''))),
               PYTHONIOENCODING='utf-8', TEST_NO_PANEL_POSE=str(int(not enabled)))
    path = tmp_path / 'pose.csv'
    cmd = [sys.executable, str(RPI / 'run_hil.py'), '--mock', '--traj', 'line', '--duration', '1', '--finite-run']
    if enabled:
        cmd.extend(['--pose-log', str(path)])
    run = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, encoding='utf-8', timeout=40)
    assert run.returncode == 0, run.stdout + run.stderr
    print(next(line for line in run.stderr.splitlines() if 'jitter, whole run' in line))
    if not enabled:
        assert not list(tmp_path.glob('*.csv'))
        return
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['t_ns', 'phase', 'x', 'y', 'th']
        rows = list(reader)
    phases = [r['phase'] for r in rows]
    assert [p for i, p in enumerate(phases) if not i or p != phases[i-1]] == [
        'PREPARE', 'ARM', 'RUN', 'STOPPING', 'END']
    ts = [int(r['t_ns']) / 1e9 for r in rows]
    assert 17 <= (len(ts) - 1) / (ts[-1] - ts[0]) <= 23
    assert 0.04 <= statistics.median(b-a for a, b in zip(ts, ts[1:])) <= 0.06
