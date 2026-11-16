"""
fault_run.py: fault injection batch 1 in fwsim (L82, Q1 = F2), without changing any file of the S04 lock
(fwsim.py, imu_model.py, trajectory_pub.py, plant_step.py, params.py: fwsim/s04_ga.py SIM_SOURCES).

The same schedule file and the same integer operations as the Pi (rpi5/fault_schedule.py), applied around
fwsim.simulate while it runs:
  1. the module global fwsim.plant_step is wrapped: it keeps the plant state of every 1 ms step (IMU delay)
     and integrates the wheel angle "sent" to the firmware, adding 0 for a wheel in an enc_loss window;
  2. imu['noise'] becomes an object with .apply(sample, t_us) (simulate only calls that): the IMU sample of
     step k - d with its noise row at (k - d) ms (delay), then gz * (1 - s) on the int16 of the frame
     (gyro scale); simulate then passes it through imu_model.firmware_view as on the HIL;
  3. imu['hook'] (fwsim extension point) drops the three reports in an imu_loss window;
  4. the fw object is proxied: fw.tick gets the counts of the "sent" wheel angle;
  5. batch 2 plant faults (rpi5/fault_plant.py, design 2026-11-15 section 6): the wrapped plant_step of step k
     gets the parameters of fault_plant.PlantVariant (the same function the Pi uses) instead of the nominal
     ones while a plant row is on; every such step is recorded (fault_plant, faults_plant.csv).
  6. batch 2 ibus_loss is refused (BENCH_ONLY): the RC / iBUS logic is in the firmware main.c, which fwsim
     does not run (Liam 15/11 Q2 "Chi ban HIL").
The wrapper is removed in a finally block. Not re-entrant: one run at a time per process.

Assumptions (stated by the author, to be checked): inside one fwsim step k (t = k ms) simulate calls the
IMU report (noise.apply, then hook), then fw.tick, then plant_step(x_k) -> x_{k+1}, and adds omega(x_{k+1})*dt
to the wheel angle (fwsim.py 2b2426b lines 355-392; pinned by tests/test_fault_run.py). t0 = 0.

Differences to the bench (to state with the results): imu['noise'] is wrapped only when the run has IMU
noise (Codex FI1-5); then the IMU values go through firmware_view and fault_imu has a row per report; else
the IMU path is exactly simulate's and fault_imu is empty. An imu_delay / gyro_scale row with the noise off
is refused (Codex FI2-4: it would change the IMU path of the whole run, so the run would differ from its
baseline before any window). Compare a fault run with a --baseline run (this wrapper, no row) of the same noise;
the IMU loss acts from the next report (the H7 may still deliver one queued report); after an encoder loss
the count goes on from where it stood (the H7 restarts its pulse train with a new phase).

  python fwsim/fault_run.py --task A-W1 --mode PID --faults <CSV> --out <05_DATA/results/FAULT_C7/...>
  python fwsim/fault_run.py --task A-W1 --mode PID --baseline --out <...>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
import fwsim                                                   # noqa: E402
import fault_plant                                             # noqa: E402
from fault_schedule import ActiveSet, Schedule, ScheduleError, check, has_plant_rows, scale_raw   # noqa: E402
from imu_model import ImuNoise, imu_from_plant                 # noqa: E402
from protocol import IMU_RANGES, _to_int16                     # noqa: E402

GZ_RANGE = IMU_RANGES[0]
TAIL_S = 1.0                     # setpoint 0 after the task, as s02_groupA_sim.py STOP_TAIL_S
# fault_tick / fault_imu mask bits: wheels 0..3, then loss, delay, scale
M_LOSS, M_DELAY, M_SCALE = 16, 32, 64
BENCH_ONLY = ('ibus_loss',)      # batch 2: fwsim has no RC model (design 2026-11-15 section 6, Liam 15/11 Q2)


def refuse_bench_only(rows) -> None:
    """ScheduleError when a row is a fault fwsim cannot run (the iBUS path is firmware main.c, not simulated)."""
    for r in rows:
        if r.fault in BENCH_ONLY:
            raise ScheduleError(f'{r.id!r}: {r.fault} runs on the HIL bench only (fwsim has no RC / iBUS model)')


class _FwProxy:
    def __init__(self, fw, owner):
        self._fw, self._o = fw, owner

    def __getattr__(self, name):
        return getattr(self._fw, name)

    def tick(self, t_us, counts, sp):
        o = self._o
        o.adv(int(t_us) // 1000)                     # the mask recorded is the one of this step
        sent = np.floor(o.theta_sent * fwsim.COUNTS_PER_REV / (2 * np.pi)) if o.enc_used else counts
        o.rec_tick.append((t_us * 1e-6, o.mask(), *[int(c) for c in counts], *[int(c) for c in sent]))
        return self._fw.tick(t_us, sent, sp)


class _Noise:
    def __init__(self, inner, owner):
        self._inner, self._o = inner, owner

    def _noisy(self, sample, t_us):
        return tuple(self._inner.apply(sample, t_us)) if self._inner is not None else tuple(sample)

    def apply(self, sample, t_us):
        o = self._o
        k = int(t_us) // 1000
        o.adv(k)
        a = o.act
        true = self._noisy(sample, t_us)
        out = true
        if a.delay:
            kd = k - a.delay
            s = imu_from_plant(o.X[kd], o.X[kd - 1] if kd >= 1 else None, 0.001, o.g)
            out = self._noisy(s, t_us - a.delay * 1000)
        if a.scale is not None:
            raw = _to_int16(out[0], GZ_RANGE)
            out = (scale_raw(raw, a.scale) / 32767.0 * GZ_RANGE,) + tuple(out[1:])
        o.rec_imu.append((t_us * 1e-6, o.mask(), *true, *out))
        return out


class FwsimFaults:
    """run(fw, setpoint, duration, params, imu=..., **simulate kwargs) -> the log of fwsim.simulate plus
    'fault_tick' (t, mask, count_true x4, count_sent x4), 'fault_imu' (t, mask, true x5, sent x5; one row
    per report, the reports of an imu_loss window are dropped after the row), 'fault_plant' (one row per
    plant step with a plant row on, columns fault_plant.RECORD_COLS) and 'faults' (summary)."""

    def __init__(self, schedule: Schedule):
        self.schedule = schedule
        self.enc_used = any(r.fault == 'enc_loss' for r in schedule.rows)

    def adv(self, k: int) -> None:
        a = self.act
        if a.next is not None and k >= a.next:
            a.advance(k)

    def mask(self) -> int:
        a = self.act
        return a.enc_mask | (M_LOSS if a.loss else 0) | (M_DELAY if a.delay else 0) | \
            (M_SCALE if a.scale is not None else 0)

    def run(self, fw, setpoint, duration, params, imu=None, **kw):
        if imu is None:
            raise ValueError('fault_run needs the IMU (QD 23: every firmware has the IMU)')
        imu = dict(imu)
        if imu.get('injection') is not None:
            raise ValueError("imu['injection'] (method C before the noise) is replaced by gyro_scale (L82 Q2)")
        if fwsim.plant_step is not _REAL_PLANT_STEP:
            raise RuntimeError('fwsim.plant_step is already wrapped: one fault run at a time per process')
        run_ms = int(round(duration * 1000))
        for r in self.schedule.rows:
            if r.end_ms > run_ms:
                raise ValueError(f'{r.id!r} ends at {r.end_ms} ms, after the run ({run_ms} ms)')
        if self.schedule.rows:         # Codex FP1-8: the full check of the schedule, also for API callers
            check(self.schedule.rows, run_ms, 'plant', None if fault_plant.nominal_problem(params) else 'traction')
        refuse_bench_only(self.schedule.rows)
        self.has_plant = has_plant_rows(self.schedule.rows)
        self.act = ActiveSet(self.schedule.rows)
        self.X, self.theta_sent = [], np.zeros(4)
        self.rec_tick, self.rec_imu, self.rec_plant = [], [], []
        self.g = params.g
        fault_plant.check_params(self.schedule.rows, params)       # batch 2: against the nominal values
        var = fault_plant.PlantVariant(self.schedule.rows, params)
        real = fwsim.plant_step
        j = [0]

        def step(x, u, p, dt, rng):
            if not self.X:
                self.X.append(np.array(x, copy=True))
            self.adv(j[0])
            on = self.act.plant_on
            if on:                                     # batch 2: the parameters of this step
                if p is not params:
                    raise RuntimeError('fwsim.simulate passed a parameter object other than the one of run()')
                p, _ = var.at(on, j[0], x)
                self.rec_plant.append((j[0], sum(1 << i for i in on), p.V_batt, *p.mu_hd, *p.motor_b,
                                       *p.motor_Tc, *p.motor_Ts, x[0], x[1], x[2]))
            x_new = real(x, u, p, dt, rng)
            self.X.append(x_new)
            w = x_new[6:10]
            m = self.act.enc_mask
            if m:
                w = w.copy()
                for i in range(4):
                    if m >> i & 1:
                        w[i] = 0.0
            self.theta_sent += w * dt
            j[0] += 1
            return x_new

        inner = imu.get('noise')
        if inner == 'replay':
            inner = ImuNoise(imu.get('rate_hz', 20.0), seed=int(imu['noise_seed']))
        elif isinstance(inner, str):
            raise ValueError(f"imu noise {inner!r}: 'replay', an ImuNoise or None")
        user_hook = imu.get('hook')

        def hook(t, kind, v):
            self.adv(int(round(t * 1000.0)))
            if self.act.loss:
                return None
            return user_hook(t, kind, v) if user_hook else v

        if inner is None and any(r.fault in ('imu_delay', 'gyro_scale') for r in self.schedule.rows):
            # Codex FI2-4: with the noise off the values would go through firmware_view for the whole run
            # (the baseline does not), so the run would differ from its baseline before any window. Type II
            # runs with the noise on (S06 T8); type I runs on the Pi, not in fwsim.
            raise ValueError('imu_delay / gyro_scale in fwsim need the IMU noise on (--imu-noise replay): '
                             'with the noise off the run would not share the IMU path of its baseline')
        if inner is not None:
            imu['noise'] = _Noise(inner, self)
        else:
            imu['noise'] = None                      # FI1-5: simulate's noise-free path, no firmware_view
        imu['hook'] = hook
        fwsim.plant_step = step
        try:
            log = fwsim.simulate(_FwProxy(fw, self), setpoint, duration, params, imu=imu, **kw)
        finally:
            fwsim.plant_step = real
        last = j[0]                                  # steps run; windows still open end here
        if self.act.next is not None or self.act.on:
            self.act.close(last)
        log['fault_tick'] = np.asarray(self.rec_tick, dtype=float).reshape(-1, 10)
        log['fault_imu'] = np.asarray(self.rec_imu, dtype=float).reshape(-1, 12)
        if self.has_plant:             # Codex FP1-10: the output of a run without plant rows stays as before
            log['fault_plant'] = np.asarray(self.rec_plant, dtype=float).reshape(-1, len(fault_plant.RECORD_COLS))
        cut = any(e[3] == 'cut' for e in self.act.events)
        nr = self.act.not_run()
        log['faults'] = dict(self.schedule.describe(), t0_step=0, events=[list(e) for e in self.act.events],
                             not_run=nr, cut=cut, complete=not cut and not nr)
        return log


_REAL_PLANT_STEP = fwsim.plant_step


def task_setpoint(task: str):
    """(setpoint(t), task duration [s]) of a group A test (trajectory_pub.GROUP_A) or A-W1
    (fwsim/imu_freq_exp.py); the setpoint is 0 after the task."""
    if task == 'A-W1':
        from imu_freq_exp import a_w1
        fn, dur = a_w1, 8.0
    else:
        from trajectory_pub import GROUP_A, LIFTED_TESTS, get_body_velocity
        if task not in GROUP_A or task in LIFTED_TESTS:
            raise SystemExit(f'--task {task!r}: A-W1 or a group A test on the ground')
        spec = GROUP_A[task]
        fn, dur = (lambda s: lambda t: get_body_velocity(t, s))(spec), spec.duration
    return (lambda t: fn(t) if t < dur else (0.0, 0.0, 0.0)), dur


def write_faults(out, log):
    """faults_tick.csv, faults_imu.csv, faults.json next to the fwsim.write_run files."""
    np.savetxt(os.path.join(out, 'faults_tick.csv'), log['fault_tick'], delimiter=',', comments='', fmt='%.9g',
               header='t,mask,count_true_fl,count_true_fr,count_true_rl,count_true_rr,'
                      'count_sent_fl,count_sent_fr,count_sent_rl,count_sent_rr')
    np.savetxt(os.path.join(out, 'faults_imu.csv'), log['fault_imu'], delimiter=',', comments='', fmt='%.9g',
               header='t,mask,gz_true,ax_true,ay_true,az_true,yaw_true,gz_sent,ax_sent,ay_sent,az_sent,yaw_sent')
    plant = 'fault_plant' in log                      # only with plant rows (Codex FP1-10)
    if plant:
        np.savetxt(os.path.join(out, 'faults_plant.csv'), log['fault_plant'], delimiter=',', comments='',
                   fmt='%.17g', header=','.join(fault_plant.RECORD_COLS))
    with open(os.path.join(out, 'faults.json'), 'w') as f:
        json.dump(log['faults'], f, indent=1)
    with open(os.path.join(out, 'meta.yaml'), 'a', encoding='utf-8') as f:
        f.write(f"faults: {json.dumps(log['faults'])}\n"
                "fault_files:\n"
                "  - faults_tick.csv: \"every firmware tick: mask (bits 0-3 enc_loss wheels FL FR RL RR, 16 imu_loss, "
                "32 imu_delay, 64 gyro_scale), counts of the plant wheel angle and counts given to the firmware\"\n"
                "  - faults_imu.csv: \"every IMU report: mask, true values (after the noise) and values given to "
                "firmware_view; reports of an imu_loss window are dropped after this row\"\n")
        if plant:
            f.write("  - faults_plant.csv: \"every plant step with a plant fault row on (batch 2): step, mask of "
                    "the rows on (bit = row index), the parameters used (V_batt, mu_hd, motor_b, motor_Tc, motor_Ts "
                    "per wheel FL FR RL RR) and x, y, theta at the start of the step\"\n")


def main():
    from fault_schedule import load
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, help='A-W1 or a group A test on the ground')
    ap.add_argument('--mode', choices=['PID', 'ADRC'], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--faults', metavar='CSV', help='fault schedule (rpi5/fault_schedule.py), t0 = 0')
    g.add_argument('--baseline', action='store_true', help='the same wrapper with no row (the reference run)')
    ap.add_argument('--seed', type=int, default=0, help='plant RNG seed')
    fwsim.add_plant_imu_args(ap)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.no_imu:
        raise SystemExit('--no-imu: QD 23, every firmware has the IMU')
    sp, dur = task_setpoint(a.task)
    duration = dur + TAIL_S
    imu = fwsim.imu_config(a)
    params = fwsim.plant_params(a)
    if a.faults:
        try:
            plant = None if fault_plant.nominal_problem(params) else 'traction'     # batch 2 plant faults
            sched = load(a.faults, int(round(duration * 1000)), 'plant', plant)
            fault_plant.check_params(sched.rows, params)
            refuse_bench_only(sched.rows)
        except Exception as exc:                     # ScheduleError, OSError
            raise SystemExit(f'--faults {a.faults}: {exc}')
    else:
        sched = Schedule((), '', '')
    fw = fwsim.FirmwareSim()
    log = FwsimFaults(sched).run(fw, sp, duration, params, imu=imu, seed=a.seed, mode=a.mode)
    met = fwsim.metrics(log)
    fwsim.write_run(a.out, log, met, fw, dict(vars(a), duration=duration), params)
    write_faults(a.out, log)
    print(f'{a.task} {a.mode} {"baseline" if a.baseline else a.faults}: complete={log["faults"]["complete"]} '
          f'-> {a.out}')


if __name__ == '__main__':
    main()
