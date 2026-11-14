"""rpi5/fault_tasks.py (L82 type II, Liam 14/11 CODE_FAULT "Them A-W1 cho Pi"): the bench A-W1 setpoint is
the fwsim one (fwsim/imu_freq_exp.a_w1, and fwsim/fault_run.task_setpoint for the task part), install() wraps
only FaultTaskSpec and composes with o1_tasks.install(), run_hil._run_traj installs it, and run_hil --test A-W1
takes the task (run length 8 s for the fault schedule, RUN = 8 s and the 1 s zero tail with --finite-run)."""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fault_schedule as fs        # noqa: E402
import fault_tasks                 # noqa: E402
import trajectory_pub              # noqa: E402

GRID = sorted(set(np.round(np.arange(0.0, 9.5, 0.001), 6)) | {1.9999999, 2.0, 3.9999999, 4.0, 5.9999999, 6.0, 8.0})


def test_a_w1_is_the_fwsim_setpoint():
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'fwsim'))
    try:
        import imu_freq_exp
        import fault_run
    except Exception as exc:                   # the Pi may lack what fwsim imports
        pytest.skip(f'fwsim modules not importable here: {exc}')
    sp, dur = fault_run.task_setpoint('A-W1')
    assert dur == fault_tasks.TASKS['A-W1'][0] == 8.0
    spec = fault_tasks.make_spec('A-W1')
    for t in GRID:
        assert fault_tasks.a_w1(t) == imu_freq_exp.a_w1(t), t
        if t < dur:
            assert fault_tasks.body_velocity(t, spec) == sp(t), t
    assert fault_tasks.a_w1(3.0) == (0.0, 0.0, 1.0) and fault_tasks.a_w1(4.0) == (0.0, 0.0, -1.0)


def test_spec_and_unknown_task():
    spec = fault_tasks.make_spec('A-W1')
    assert (spec.kind, spec.duration, spec.fault_task, spec.loop) == ('fault_task', 8.0, 'A-W1', False)
    assert not getattr(spec, 'task', '')       # run_hil._run_traj does not take it for an O1 task
    with pytest.raises(ValueError):
        fault_tasks.make_spec('A-W2')


def test_install_wraps_only_fault_task_specs_and_composes_with_o1():
    import o1_tasks

    for order in ((fault_tasks, o1_tasks), (o1_tasks, fault_tasks)):
        class M:                               # a stand-in module, trajectory_pub itself untouched
            get_body_velocity = staticmethod(trajectory_pub.get_body_velocity)
        for mod in order + order:              # twice: idempotent
            mod.install(M)
        for tid, spec in trajectory_pub.GROUP_A.items():
            for t in (0.0, 1.3, 4.0, 7.7):
                assert M.get_body_velocity(t, spec) == trajectory_pub.get_body_velocity(t, spec), tid
        spec = fault_tasks.make_spec('A-W1')
        for t in (0.0, 2.5, 4.5, 7.0):
            assert M.get_body_velocity(t, spec) == fault_tasks.a_w1(t)
        o1 = o1_tasks.make_spec('O1-F05')
        assert M.get_body_velocity(3.0, o1) == o1_tasks.body_velocity(3.0, o1)


def test_run_traj_installs_it(monkeypatch):
    """run_hil._run_traj (the publishing process) wraps trajectory_pub.get_body_velocity for a FaultTaskSpec."""
    import run_hil
    got = []

    class FakePub:
        def __init__(self, spec, **kw):
            self.spec = spec

        def run(self):
            got.extend(trajectory_pub.get_body_velocity(t, self.spec) for t in (1.0, 3.0, 5.0, 7.0))

    monkeypatch.setattr(trajectory_pub, 'get_body_velocity', trajectory_pub.get_body_velocity)   # restored after
    monkeypatch.setattr(trajectory_pub, 'TrajectoryPublisher', FakePub)
    monkeypatch.setattr(os, 'sched_setaffinity', lambda *a: None, raising=False)
    run_hil._run_traj(fault_tasks.make_spec('A-W1'), '/dev/null', True, None, False)
    assert got == [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)]


def test_finite_run_config_for_a_w1():
    import run_hil
    from run_lifecycle import LifecycleConfig
    args = SimpleNamespace(steps=0, no_traj=False, traj_loop=False, fixed_omega=None, omega_steps=None,
                           omega_schedule=None, load_schedule=None, o1=None)
    lc, ceiling = run_hil._finite_run_config(args, fault_tasks.make_spec('A-W1'))
    assert lc.run_s == 8.0 and lc.stop_s == LifecycleConfig.stop_s == 1.0     # as fault_run: task + 1 s zeros
    assert ceiling > 9000


def _run_hil(tmp_path, body, *extra):
    p = tmp_path / 's.csv'
    p.write_text(fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n' + body)
    return subprocess.run([sys.executable, os.path.join(HERE, '..', 'run_hil.py'), '--mock', '--test', 'A-W1',
                           '--finite-run', '--faults', str(p)] + list(extra),
                          capture_output=True, text=True, timeout=60)


def test_run_hil_takes_a_w1(tmp_path):
    """--test A-W1 passes the task selection (the run stops later, at the fault checks, before any SPI):
    the schedule's run length is the task's 8 s (a window ending at 8000 ms is accepted up to the frame-log
    check, one ending at 8001 ms is after the run); an unknown task is refused with both lists."""
    r = _run_hil(tmp_path, 'E1,enc_loss,FL,3000,8000,\n')
    assert r.returncode != 0 and '--faults needs --frame-log' in r.stderr, r.stderr[-800:]
    r = _run_hil(tmp_path, 'E1,enc_loss,FL,3000,8001,\n', '--frame-log', str(tmp_path / 'f.npz'))
    assert r.returncode != 0 and 'after the run' in r.stderr, r.stderr[-800:]
    r = subprocess.run([sys.executable, os.path.join(HERE, '..', 'run_hil.py'), '--mock', '--test', 'A-W2'],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "--test must be one of" in r.stderr and "'A-W1'" in r.stderr, r.stderr[-800:]
