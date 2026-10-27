"""
plant_timing.py: how long one plant step takes on this machine (RPi5 check
before a heavier plant goes into the 1 kHz HIL loop, QĐ 25 / slip method A).

Times, under the same scheduling as the plant loop (SCHED_FIFO 90, core 0):
  current      plant_step() as used by the HIL today (motor 'dc', no slip)
  tick         current plant + the per-tick work of hil_node besides SPI/GPIO
               (IMU sample, state reorder, encode_state, decode_torques)
  slipA_nsubN  THROWAWAY timing prototype of a minimal slip model (method A):
               body with its own dynamics, traction per wheel capped by mu*N,
               N sub-steps per 1 ms. Only for timing: parameters are guesses,
               the model is not validated and is not used anywhere else.
Budget reference (estimates in hil_node.py, not measured): SPI 40 bytes at
1 MHz ~320 us + GPIO pulse ~10 us, period 1000 us.

Usage (on the Pi, nothing else running; does not touch SPI/GPIO, so the H7 can
stay connected):
  cd ~/hil-framework/rpi5 && sudo python3 plant_timing.py
  sudo python3 plant_timing.py --n 20000 --nsub 5,10,20
Copy the whole output to Claude (it ends with the machine and commit lines).
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np

from params import MecanumParams
from plant_step import plant_step, motor_torque
from protocol import encode_state, decode_torques
from hil_node import plant_to_frame_state
from imu_model import imu_from_plant
from timing import set_realtime, pin_cpu


def _slip_a_step(p, Minv, Jw, N, mu, k_s):
    """Prototype method-A step with nsub sub-steps (timing only)."""
    cap = mu * N * p.r
    HT = p.H_inv.T

    def step(v, w, u, nsub, dt=0.001):
        h = dt / nsub
        for _ in range(nsub):
            wc = p.H_inv @ v                                   # wheel-equivalent contact speed
            tc = np.clip(k_s * (w - wc), -cap, cap)            # traction torque, capped
            tm = motor_torque(u, w, p) - (p.motor_b * w + p.motor_Tc * np.sign(w))
            w = w + h * (tm - tc) / Jw
            v = v + h * (Minv @ (HT @ tc))
        return v, w
    return step


def _time(fn, n):
    """Per-call times [us] of n calls after 500 warm-up calls."""
    for _ in range(500):
        fn()
    t = np.empty(n)
    pc = time.perf_counter_ns
    for i in range(n):
        t0 = pc()
        fn()
        t[i] = pc() - t0
    return t / 1000.0


def _git(*args):
    try:
        return subprocess.run(['git'] + list(args), capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'UNKNOWN'


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return 'n/a'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=20000, help='timed calls per variant')
    ap.add_argument('--nsub', default='5,10,20', help='sub-steps of the method-A prototype')
    ap.add_argument('--no-rt', action='store_true', help='skip SCHED_FIFO / core pinning')
    a = ap.parse_args()

    rt = pin = False
    if not a.no_rt:
        pin = pin_cpu(0)
        rt = set_realtime(90)

    p = MecanumParams()
    rng = np.random.default_rng(0)
    x = np.zeros(10); x[6:10] = 5.0
    u = np.full(4, 0.3)
    state = {'x': x, 'prev': None, 'seq': 0}

    def current():
        plant_step(x, u, p, 0.001, rng)

    def tick():
        s = plant_step(state['x'], u, p, 0.001, rng)
        imu = imu_from_plant(s, state['prev'], 0.001, p.g)
        state['prev'] = s
        state['seq'] = state['seq'] % 255 + 1
        frame = encode_state(plant_to_frame_state(list(s)), traj_ref=(0.1, 0.0, 0.2), traj_flags=0,
                             seq=state['seq'], imu=imu, imu_seq=state['seq'])
        decode_torques(frame)

    variants = [('current', current), ('tick', tick)]
    Minv = np.diag([1.0 / p.M, 1.0 / p.M, 1.0 / p.Iz])
    stepA = _slip_a_step(p, Minv, np.full(4, p.J_w), p.M * p.g / 4.0, 0.8, 5.0)
    v0, w0 = np.zeros(3), np.full(4, 5.0)
    for ns in (int(s) for s in a.nsub.split(',') if s):
        variants.append((f'slipA_nsub{ns}', lambda ns=ns: stepA(v0, w0, u, ns)))

    print(f'{"variant":<16}{"mean_us":>9}{"p50_us":>9}{"p99_us":>9}{"max_us":>9}   n={a.n}', flush=True)
    for name, fn in variants:
        t = _time(fn, a.n)
        print(f'{name:<16}{t.mean():9.1f}{np.percentile(t, 50):9.1f}{np.percentile(t, 99):9.1f}{t.max():9.1f}',
              flush=True)

    print(f'machine: {platform.node()} | {platform.platform()} | python {platform.python_version()} '
          f'| numpy {np.__version__}')
    print(f'cpu governor: {_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")} '
          f'| cur freq kHz: {_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")}')
    print(f'commit: {_git("rev-parse", "--short", "HEAD")} ({_git("rev-parse", "--abbrev-ref", "HEAD")}) '
          f'| dirty: {bool(_git("status", "--porcelain", "--", "."))} | SCHED_FIFO: {rt} | pinned core 0: {pin}')
    print(f'date: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
