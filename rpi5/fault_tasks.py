"""
fault_tasks.py: setpoint of the task(s) the type II fault runs use that rpi5/trajectory_pub.py does not have,
for the HIL bench (run_hil.py --test A-W1). L82, spec S06 T8 "Loai II" (task A-W1, Q6); Liam 14/11 CODE_FAULT
"Them A-W1 cho Pi" (hop chon: A-W1 for the bench, so bench and fwsim run the same task).

  A-W1   turn on the spot, wz 0 / +1 / -1 / 0 rad/s, 2 s each (8 s), vx = vy = 0: the task of
         fwsim/imu_freq_exp.py a_w1, which fwsim/fault_run.py --task A-W1 runs. The function is written
         here again (the Pi publisher does not import the fwsim experiment script, which imports fwsim,
         ctypes and the build tools); test_fault_tasks.py pins it equal to imu_freq_exp.a_w1.

On the bench, with --finite-run, LifecycleConfig.run_s is the task (8 s) and STOPPING then sends the zero
setpoint for LifecycleConfig.stop_s (1 s), as fault_run.py runs the task then TAIL_S = 1 s of zeros. RUN is not
exactly 8 s (Codex C7T1-1; Liam 15/11 CODE_FAULT "Sua 10, giu RUN": the mechanism is kept): the lifecycle ends
RUN when the publisher reports the end of the task, 8.083-8.102 s in FAULT_C7 2026-11-14 run02 / run03, and the
publisher's clock starts when it sees RUN, so the setpoint changes 2-4 ms after t0 + 2 / 4 / 6 s. A fault window
counts from t0 = the first RUN step on the bench and from 0 in fwsim; tools/fault_c7_table.py records the RUN
length (run_s) and the first setpoint change (ref_first_s) of each bench run.

Why a new module (as rpi5/o1_tasks.py): rpi5/trajectory_pub.py is in the S04 blob lock (fwsim/s04_ga.py
SIM_SOURCES), so it is not edited. install() wraps its get_body_velocity in the publishing process
(run_hil._run_traj): a FaultTaskSpec gets this setpoint, every other spec the unchanged one.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import trajectory_pub                                   # noqa: E402
from trajectory_pub import TrajectorySpec               # noqa: E402


def a_w1(t):
    """A-W1: turn on the spot, wz 0 / +1 / -1 / 0 rad/s, 2 s each (= fwsim/imu_freq_exp.a_w1)."""
    wz = 0.0 if t < 2.0 else 1.0 if t < 4.0 else -1.0 if t < 6.0 else 0.0
    return 0.0, 0.0, wz


TASKS = {'A-W1': (8.0, a_w1)}                           # id -> (duration [s], f(t) -> (vx, vy, wz))


@dataclass
class FaultTaskSpec(TrajectorySpec):
    """TrajectorySpec of a fault-run task for the trajectory publisher (kind 'fault_task', see install()).
    The field is fault_task, not task: run_hil._run_traj installs o1_tasks for a spec with a 'task'."""
    fault_task: str = ''


def make_spec(task_id):
    if task_id not in TASKS:
        raise ValueError(f'fault task {task_id!r}: one of {sorted(TASKS)}')
    return FaultTaskSpec(kind='fault_task', v=0.0, duration=TASKS[task_id][0], fault_task=task_id)


def body_velocity(t, spec):
    return TASKS[spec.fault_task][1](t)


def install(module=trajectory_pub):
    """Wrap module.get_body_velocity: FaultTaskSpec -> body_velocity, anything else -> the original."""
    orig = module.get_body_velocity
    if getattr(orig, '_fault_task_wrapped', False):
        return

    def get_body_velocity(t, spec):
        if isinstance(spec, FaultTaskSpec):
            return body_velocity(t, spec)
        return orig(t, spec)

    get_body_velocity._fault_task_wrapped = True
    module.get_body_velocity = get_body_velocity
