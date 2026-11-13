"""
plant_timing.py: how long one plant step takes on this machine (RPi5 check
before a heavier plant goes into the 1 kHz HIL loop, QĐ 25 / slip method A).

Times, under the same scheduling as the plant loop (SCHED_FIFO 90, core 0):
  current      plant_step() as used by the HIL today (motor 'dc', no slip)
  tick         current plant + the per-tick work of hil_node besides SPI/GPIO
               (IMU sample, state reorder, encode_state, decode_torques)
  traction_muM plant_step() with ground_model 'traction' (method A as implemented,
               mu_hd = M, sub-steps from the stability bound)
  frame_log    the per-frame writes of run_hil.py --frame-log (hil_node.py: t_send + 40 B MOSI
               + 40 B MISO into preallocated arrays, inside the step before t_done): extra
               load of S06 T1/T2 runs that log frames (S06, Liam 30/10)
  fault_idle   L82: fault_schedule.FramePatcher.patch outside every window (with a delay row,
               so the ring write of every step is counted)
  fault_all4   L82: the same with enc_loss (2 wheels), imu_loss, imu_delay and gyro_scale all on
  mosi_true_log L82: the extra 40-byte write of the true frame into the frame log (--faults)
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
    for mu in (0.5, 0.8):
        pt = MecanumParams()
        pt.ground_model = 'traction'
        pt.mu_hd = np.full(4, mu)
        variants.append((f'traction_mu{mu}', lambda pt=pt: plant_step(x, u, pt, 0.001, rng)))
    nlog = 1 << 16
    flog = {'t_send_ns': np.zeros(nlog, dtype=np.int64), 'mosi': np.zeros((nlog, 40), dtype=np.uint8),
            'miso': np.zeros((nlog, 40), dtype=np.uint8)}
    mosi_b, miso_b = bytearray(40), bytearray(40)
    flog_k = [0]

    def frame_log():
        # same three writes as HILNode._tick with --frame-log
        k = flog_k[0] = (flog_k[0] + 1) % nlog
        flog['t_send_ns'][k] = time.monotonic_ns()
        flog['mosi'][k] = memoryview(bytes(mosi_b))
        flog['miso'][k] = memoryview(bytes(miso_b))

    variants.append(('frame_log', frame_log))
    # L82 fault injection: FramePatcher.patch outside every window (one delay row: the ring write runs every
    # step) and with all four faults on; the extra mosi_true write of the frame log
    import fault_schedule as fsch
    frame0 = encode_state(plant_to_frame_state(list(x)), imu=(0.5, 0.1, 0.2, 9.8, 0.3), imu_seq=1, seq=1)
    head = fsch.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'

    def patcher(body):
        rows = fsch.parse(head + body)
        fsch.check(rows, 1 << 30, 'plant')
        pa = fsch.FramePatcher(fsch.Schedule(tuple(rows)))
        pa.start(0)
        return pa
    idle = patcher(f'D1,imu_delay,imu,{1 << 29},{(1 << 29) + 1},d_ms=100\n')
    busy = patcher(f'E1,enc_loss,FL+RR,0,{1 << 29},\nL1,imu_loss,imu,0,{1 << 29},\n'
                   f'D1,imu_delay,imu,100,{1 << 29},d_ms=100\nS1,gyro_scale,gz,0,{1 << 29},s=0.3\n')
    step_i, step_b = [0], [0]
    for k in range(200):                                   # past d: the delay reads the ring
        busy.patch(k, frame0)
    step_b[0] = 200

    def fault_idle():
        step_i[0] += 1
        idle.patch(step_i[0], frame0)

    def fault_all4():
        step_b[0] += 1
        busy.patch(step_b[0], frame0)

    def mosi_true_log():
        k = flog_k[0]
        flog['mosi'][k] = memoryview(frame0)

    variants += [('fault_idle', fault_idle), ('fault_all4', fault_all4), ('mosi_true_log', mosi_true_log)]
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
