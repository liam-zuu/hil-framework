"""
o1_tasks.py: setpoint generator of the O1 floor tasks (figure 8 at three levels, K1, K2) for the
HIL bench (run_hil.py --o1) and the vehicle (pc_sender.py --o1), the same generator as fwsim.

C6 of spec S05 v3.4 section 8.1 (O1 v11 section 5 items 1, 1b). Design and Liam's approval (L76,
10/11, CODE session local_796cfeea, "Duyet, lam luon"): CLAUDE_THESIS
03_CODE/docs/2026-11-10_Thiet_ke_C6_bo_phat_O1.md.

  O1-F02, O1-F05, O1-F08  figure 8 x = A sin tau, y = (A/2) sin 2 tau, A = 1 m, at level
                          0.2 / 0.5 / 0.8 m/s, schedule tau(t) of O1 section 3.1 (ramp T_r, hold,
                          ramp down, ends at tau = 2 pi on the origin); heading on the tangent
                          (vx = path speed, vy = 0, wz = curvature x speed). The setpoint IS
                          fwsim/o1_feasibility.Schedule(v).setpoint (imported, not copied), the
                          schedule S03 and S04 run on fwsim. Duration T_cmd = T + T_r.
  O1-K1                   final-test strafe (O1 section 3.7): 2 m with vy only, to the LEFT (+vy),
                          vx = wz = 0; ramp a_r 0.5 m/s^2 to 0.5 m/s, hold, ramp down to 0 at the
                          end of the segment: 5.0 s.
  O1-K2                   final-test square, side 1 m, no body rotation: forward (+vx), left (+vy),
                          back (-vx), right (-vy), each side as K1's profile (3.0 s), command 0 for
                          t_g = 1 s between two sides, ends on the start corner: 15.0 s.
K1, K2 are the final test set (O1 section 3.7, no run in any tuning, screening or GA step):
task() refuses them unless allow_final_set=True (--allow-final-set on the command lines).
The direction of K1 and the order of K2 are Claude's choice (the spec fixes neither), approved
by Liam (L76).

Why a new module (L76 item 1): rpi5/trajectory_pub.py and fwsim/o1_feasibility.py are in the
SIM_SOURCES of fwsim/s04_ga.py, whose main run stops when their blob differs from the S03 main
run. trajectory_pub is therefore not edited: install() wraps its get_body_velocity in the
process that publishes (run_hil._run_traj), so an O1Spec gets the O1 setpoint and every other
spec the unchanged one.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..', 'fwsim')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import trajectory_pub                                   # noqa: E402
from trajectory_pub import TrajectorySpec               # noqa: E402
from o1_feasibility import Schedule, A_R                # noqa: E402  (A_R = 0.5 m/s^2, O1 3.1)

FIG8 = {'O1-F02': 0.2, 'O1-F05': 0.5, 'O1-F08': 0.8}    # O1 section 1: three levels
FINAL_SET = ('O1-K1', 'O1-K2')                          # O1 section 3.7
TASKS = tuple(FIG8) + FINAL_SET
K_V = 0.5           # m/s, K1 and K2 run at 0.5 only (O1 3.7)
K_TG = 1.0          # s, t_g = t_d (O1 3.7, Liam 04/11)
K1_LEN = 2.0        # m
K2_SIDE = 1.0       # m
# K2 sides as body-frame unit directions (vx, vy): forward, left, back, right (L76 item 3)
K2_DIRS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def segment_time(length, v=K_V, a=A_R):
    """Duration of one straight segment: ramp v/a, hold, ramp v/a; length/v + v/a."""
    if length < v * v / a:
        raise ValueError(f'segment {length} m too short to reach {v} m/s at {a} m/s^2')
    return length / v + v / a


def segment_speed(t, length, v=K_V, a=A_R):
    """Path speed at time t of one straight segment (0 outside [0, segment_time))."""
    T, Tr = segment_time(length, v, a), v / a
    if t < 0.0 or t >= T:
        return 0.0
    if t < Tr:
        return a * t
    if t < T - Tr:
        return v
    return a * (T - t)


def _k1(t):
    return 0.0, segment_speed(t, K1_LEN), 0.0


K1_T = segment_time(K1_LEN)
K2_SIDE_T = segment_time(K2_SIDE)
K2_T = 4 * K2_SIDE_T + 3 * K_TG


def _k2(t):
    for i, (ux, uy) in enumerate(K2_DIRS):
        t0 = i * (K2_SIDE_T + K_TG)
        if t0 <= t < t0 + K2_SIDE_T:
            s = segment_speed(t - t0, K2_SIDE)
            return ux * s, uy * s, 0.0
    return 0.0, 0.0, 0.0


def task(task_id, allow_final_set=False):
    """(duration [s], f(t) -> (vx, vy, wz)) of an O1 task; t from the start of the command."""
    if task_id not in TASKS:
        raise ValueError(f'O1 task {task_id!r}: one of {TASKS}')
    if task_id in FINAL_SET and not allow_final_set:
        raise ValueError(f'{task_id} is the final test set (O1 3.7): only with --allow-final-set')
    if task_id in FIG8:
        sch = Schedule(FIG8[task_id])
        return sch.Tcmd, sch.setpoint
    if task_id == 'O1-K1':
        return K1_T, _k1
    return K2_T, _k2


@dataclass
class O1Spec(TrajectorySpec):
    """TrajectorySpec of an O1 task for the trajectory publisher (kind 'o1', see install())."""
    task: str = ''


def make_spec(task_id, allow_final_set=False):
    duration, _ = task(task_id, allow_final_set)
    return O1Spec(kind='o1', v=FIG8.get(task_id, K_V), radius=1.0, duration=duration, task=task_id)


_FNS = {}


def body_velocity(t, spec):
    """O1 setpoint of an O1Spec at time t (the final set was checked in make_spec)."""
    f = _FNS.get(spec.task)
    if f is None:
        f = _FNS[spec.task] = task(spec.task, allow_final_set=True)[1]
    return f(t)


def install(module=trajectory_pub):
    """Wrap module.get_body_velocity: O1Spec -> body_velocity, anything else -> the original."""
    orig = module.get_body_velocity
    if getattr(orig, '_o1_wrapped', False):
        return

    def get_body_velocity(t, spec):
        if isinstance(spec, O1Spec):
            return body_velocity(t, spec)
        return orig(t, spec)

    get_body_velocity._o1_wrapped = True
    module.get_body_velocity = get_body_velocity
