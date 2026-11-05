"""
s01_openloop_runner.py: PC side of spec S01 (wheels lifted, open-loop PWM tests).

Drives the ESP32 test mode `-DTEST_MODE -DTEST_OPENLOOP` (AGV_MECANUM
src/tests/test_openloop.c) over its USB-UART port at 921600 baud, runs the
S01 plan wheel by wheel, and writes the files the spec asks for:

  <out>/raw/<run_id>.csv   t_us, wheel, pwm_cmd, en, count, omega_rad_s
  <out>/runs.csv           one row per run (spec columns + extra QC columns)
  <out>/plan.csv           the planned run list (seeded random order per wheel)
  <out>/meta.yaml          skeleton, filled with what the script knows

Usage
  python s01_openloop_runner.py --port COM5 --out <dir> --esp32-commit dc24443
  python s01_openloop_runner.py --dry-run --out <dir>      # simulated motor, no hardware
  python s01_openloop_runner.py --port COM5 --out <dir> --wheels 0 --reps 1 --quick

Before each wheel block the script asks for the battery voltage (multimeter),
and again after the block; a drop above 0.3 V is flagged so the block can be redone.
Ctrl+C sends STOP. The ESP32 also stops by itself if heartbeats stop for 0.5 s.

Run types (PWM on the ESP32 scale -1000..1000):
  step   pre 300 ms at 0 (EN=1), hold 3000 ms at the level, 2000 ms at 0 (EN=1, brake)
  coast  1500 ms at the level, then EN=0 (motor open, free coast) for 4000 ms
  brake  1500 ms at the level, then PWM 0 with EN=1 (short-circuit brake) for 3000 ms
  rev    +600 2000 ms, -600 2000 ms, +600 2000 ms, 0 (brake) 2000 ms
Omega is computed here from counts: central difference over +-5 ms, 1320 counts/rev.
steady_ok: the five 100 ms means of the last 0.5 s of hold agree within 2% (spec S01).
"""
import argparse
import csv
import math
import os
import random
import subprocess
import sys
import threading
import time

COUNTS_PER_REV = 1320                  # 330 pulses/channel x 4 (GB37-520, 11 x 30:1)
WHEELS = ['FL', 'FR', 'RL', 'RR']
STEP_LEVELS = [50, 75, 100, 125, 150, 200, 400, 600, 800, 1000]
COAST_LEVELS = [400, 700, 1000]
QUICK_STEP_LEVELS = [100, 400, 1000]
VBATT_MAX_DROP = 0.3                   # V per block (spec S01)
STEADY_TOL = 0.02                      # omega spread in last 0.5 s of hold (spec S01)


# ── Plan ──────────────────────────────────────────────────────────────────────

def segments(kind, level):
    if kind == 'step':
        return [(0, 1, 300), (level, 1, 3000), (0, 1, 2000)]
    if kind == 'coast':
        return [(level, 1, 1500), (0, 0, 4000)]
    if kind == 'brake':
        return [(level, 1, 1500), (0, 1, 3000)]
    if kind == 'rev':
        return [(600, 1, 2000), (-600, 1, 2000), (600, 1, 2000), (0, 1, 2000)]
    raise ValueError(kind)


def build_plan(wheels, reps, quick, seed):
    rng = random.Random(seed)
    plan = []
    steps = QUICK_STEP_LEVELS if quick else STEP_LEVELS
    coasts = COAST_LEVELS[-1:] if quick else COAST_LEVELS
    for w in wheels:
        block = []
        for rep in range(1, reps + 1):
            for lv in steps:
                for sgn in (1, -1):
                    block.append(('step', sgn * lv, rep))
            for lv in coasts:
                block.append(('coast', lv, rep))
                block.append(('brake', lv, rep))
            block.append(('rev', 600, rep))
        rng.shuffle(block)          # randomised order: battery drift is not confounded with level
        for i, (kind, lv, rep) in enumerate(block):
            plan.append(dict(wheel=w, scenario=kind, pwm_level=abs(lv),
                             direction='pos' if lv >= 0 else 'neg', rep=rep, order=i))
    return plan


# ── Device links ──────────────────────────────────────────────────────────────

class SerialDevice:
    def __init__(self, port):
        try:
            import serial
            serial.Serial
        except (ImportError, AttributeError):
            sys.exit('pyserial is missing (a different package named "serial" may be installed).\n'
                     'Fix:  python -m pip uninstall -y serial  &&  python -m pip install pyserial')
        # Open with DTR/RTS released: pyserial asserts both by default, which the
        # board's auto-reset circuit (CH343) can turn into "enter download mode",
        # and the ESP32 then never answers. Instead pulse RTS (= EN) to restart
        # the firmware cleanly, wait for it to boot, then PING until it answers.
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, 921600, 0.2
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.lock = threading.Lock()
        self.ser.rts = True
        time.sleep(0.15)
        self.ser.rts = False
        time.sleep(1.5)                        # boot log at 115200 arrives as garbage here
        self.ser.reset_input_buffer()
        for _ in range(10):
            self.write('PING')
            t_end = time.time() + 0.5
            while time.time() < t_end:
                if b'PONG' in self.ser.readline():
                    return
        sys.exit(f'No PONG from ESP32 on {port}. Check: S01 firmware flashed '
                 '(env s01_openloop), right COM port, no serial monitor open.')

    def write(self, line):
        with self.lock:
            self.ser.write((line + '\n').encode())

    def readline(self, timeout=5.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            b = self.ser.readline()
            if b:
                return b.decode(errors='replace').strip()
        raise TimeoutError('no reply from ESP32')

    def cmd(self, line, expect='OK'):
        self.write(line)
        while True:
            r = self.readline()
            if r.startswith(expect):
                return r
            if r.startswith('ERR'):
                raise RuntimeError(f'{line} -> {r}')

    def run(self, wheel, segs):
        self.cmd('CLR')
        for pwm, en, dur in segs:
            self.cmd(f'SEG {pwm} {en} {dur}')
        self.cmd(f'RUN {wheel}')
        stop_hb = threading.Event()

        def heartbeat():
            while not stop_hb.is_set():
                self.write('H')
                time.sleep(0.2)
        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        try:
            total_s = sum(d for _, _, d in segs) / 1000
            head = self.readline(timeout=total_s + 3)
            while not head.startswith('DONE'):
                head = self.readline(timeout=total_s + 3)
        finally:
            stop_hb.set()
            hb.join()
        info = dict(kv.split('=') for kv in head.split()[1:])
        rows, imu = [], []
        while True:
            r = self.readline(timeout=5)
            if r == 'END':
                break
            if r.startswith('I,'):          # IMU report (firmware OL v2, AGV ctrl-imu)
                _, t, kind, v0, v1, v2 = r.split(',')
                imu.append((int(t), kind, float(v0), float(v1), float(v2)))
                continue
            t, c, p, e = r.split(',')
            rows.append((int(t), int(c), int(p), int(e)))
        info['imu_rows'] = imu
        return rows, info

    def stop(self):
        self.write('STOP')


class FakeDevice:
    """Simulated DC motor + gearbox at the output shaft, for testing this script and the
    identification script without hardware. Values are invented, NOT measurements."""
    J, B, K, R, TC = 0.0025, 0.002, 0.33, 2.0, 0.01     # kg m^2, N m s/rad, V s/rad, ohm, N m
    def __init__(self, seed=0, vbatt=12.0):
        self.rng = random.Random(seed)
        self.v = vbatt

    def run(self, wheel, segs):
        dt, w, theta, t, rows = 1e-4, 0.0, 0.0, 0.0, []
        v = self.v + self.rng.gauss(0, 0.05)          # run-to-run supply variation
        gain = 1.0 + 0.05 * (wheel - 1.5)            # small wheel-to-wheel spread
        sched = []
        for pwm, en, dur in segs:
            sched += [(pwm, en)] * dur
        for k, (pwm, en) in enumerate(sched):
            for _ in range(10):
                if en:
                    tau = gain * self.K / self.R * (v * pwm / 1000 - self.K * w)
                else:
                    tau = 0.0
                fric = self.TC * math.copysign(1, w) if abs(w) > 1e-3 else 0.0
                if abs(w) <= 1e-3 and abs(tau) <= self.TC:
                    w = 0.0
                else:
                    w += dt * (tau - self.B * w - fric) / self.J
                theta += w * dt
            count = int(math.floor(theta / (2 * math.pi) * COUNTS_PER_REV + 0.5))
            rows.append((k * 1000 + self.rng.randint(0, 20), count, pwm if en else 0, en))
        return rows, {'n': str(len(rows)), 'aborted': '0', 'reason': 'ok', 'imu_rows': []}

    def stop(self):
        pass


# ── Processing ────────────────────────────────────────────────────────────────

def omega_series(rows, half=5):
    t = [r[0] for r in rows]
    c = [r[1] for r in rows]
    out = []
    for i in range(len(rows)):
        a, b = max(0, i - half), min(len(rows) - 1, i + half)
        dt = (t[b] - t[a]) * 1e-6
        out.append((c[b] - c[a]) / COUNTS_PER_REV * 2 * math.pi / dt if dt > 0 else 0.0)
    return out


def qc(rows, omega, scenario):
    gaps = sum(1 for a, b in zip(rows, rows[1:]) if b[0] - a[0] > 1500)
    missing_pct = 100.0 * gaps / max(1, len(rows))
    steady_ok, w_ss = '', ''
    if scenario == 'step':
        # Last 0.5 s of hold, split into five 100 ms means: raw 1 kHz omega carries encoder
        # quantisation (2*pi/1320/0.01 s = 0.48 rad/s per count over the +-5 ms window),
        # so the 2% criterion is applied to the 100 ms means, not to single samples.
        sub = [[w for r, w in zip(rows, omega) if lo <= r[0] < lo + 100_000]
               for lo in range(2800_000, 3300_000, 100_000)]
        sub = [sum(x) / len(x) for x in sub if x]
        if sub:
            m = sum(sub) / len(sub)
            spread = (max(sub) - min(sub)) / 2
            w_ss = f'{m:.4f}'
            steady_ok = int(abs(m) > 1e-6 and spread / abs(m) < STEADY_TOL)
    return missing_pct, steady_ok, w_ss


def esp32_git_state(repo):
    """Commit and working-tree state of AGV_FIRMWARE at run time.
    .vscode/ is ignored (editor settings, not part of the build)."""
    def git(*args):
        out = subprocess.run(['git', '-C', repo] + list(args), capture_output=True,
                             text=True, check=True).stdout
        return out.rstrip('\n') if args[0] == 'status' else out.strip()   # keep porcelain column 1
    try:
        commit = git('rev-parse', '--short', 'HEAD')
        branch = git('rev-parse', '--abbrev-ref', 'HEAD')
        changed = [ln[3:] for ln in git('status', '--porcelain').splitlines()
                   if ln[3:] and not ln[3:].startswith('.vscode/')]
    except (OSError, subprocess.CalledProcessError) as e:
        return {'commit': 'UNKNOWN', 'branch': 'UNKNOWN', 'dirty': True, 'modified': [f'git error: {e}']}
    return {'commit': commit, 'branch': branch, 'dirty': bool(changed), 'modified': changed}


def ask_voltage(prompt, dry):
    if dry:
        return 12.0
    while True:
        s = input(prompt).strip().replace(',', '.')
        try:
            return float(s)
        except ValueError:
            print('  enter a number in volts, e.g. 12.4')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--wheels', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--quick', action='store_true', help='3 step levels, 1 coast level (rehearsal)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--esp32-repo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          '..', '..', 'AGV_FIRMWARE'),
                    help='AGV_FIRMWARE checkout used to build the flashed firmware (git state is recorded)')
    ap.add_argument('--esp32-commit', default=None, help='override the commit read from --esp32-repo')
    ap.add_argument('--operator', default='Liam')
    a = ap.parse_args()
    if not a.dry_run and not a.port:
        ap.error('--port is required unless --dry-run')

    esp = esp32_git_state(a.esp32_repo)
    if a.esp32_commit:
        esp['commit'] = a.esp32_commit
    print(f"ESP32 firmware: {esp['commit']} ({esp['branch']}), dirty={esp['dirty']} {esp['modified'] or ''}")
    if esp['dirty'] and not a.dry_run:
        print('  WARNING: AGV_FIRMWARE has uncommitted changes: this run cannot be a main result (03_CODE/CLAUDE.md).')
    raw_dir = os.path.join(a.out, 'raw')
    if os.path.exists(os.path.join(a.out, 'runs.csv')) or (os.path.isdir(raw_dir) and os.listdir(raw_dir)):
        sys.exit(f'{a.out} already holds data. Raw data must never be overwritten: '
                 f'use a new folder, e.g. ..._run02 (redoing one wheel: new folder + --wheels N).')
    os.makedirs(raw_dir, exist_ok=True)
    plan = build_plan(a.wheels, a.reps, a.quick, a.seed)
    with open(os.path.join(a.out, 'plan.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(plan[0].keys()))
        wr.writeheader(); wr.writerows(plan)
    print(f'{len(plan)} runs planned. Wheels lifted? Hands and cables clear? Ctrl+C stops.')

    dev = FakeDevice(a.seed) if a.dry_run else SerialDevice(a.port)
    runs_path = os.path.join(a.out, 'runs.csv')
    fields = ['run_id', 'wheel', 'scenario', 'pwm_level', 'direction', 'rep', 'v_batt_start', 'v_batt_end',
              'file', 'order', 'n_samples', 'aborted', 'reason', 'missing_pct', 'steady_ok', 'omega_ss_rad_s', 'block_ok',
              'imu_n', 'imu_overflow']
    fruns = open(runs_path, 'w', newline='')
    wr = csv.DictWriter(fruns, fieldnames=fields)
    wr.writeheader()

    try:
        for w in a.wheels:
            block = [p for p in plan if p['wheel'] == w]
            v0 = ask_voltage(f'\n[{WHEELS[w]}] battery voltage BEFORE block (V): ', a.dry_run)
            done = []
            for i, p in enumerate(block, 1):
                lv = p['pwm_level'] * (1 if p['direction'] == 'pos' else -1)
                run_id = f"{WHEELS[w]}_{p['scenario']}_{p['direction']}{p['pwm_level']:04d}_r{p['rep']}"
                rows, info = dev.run(w, segments(p['scenario'], lv))
                omega = omega_series(rows)
                fname = os.path.join('raw', run_id + '.csv')
                with open(os.path.join(a.out, fname), 'w', newline='') as f:
                    fw = csv.writer(f)
                    fw.writerow(['t_us', 'wheel', 'pwm_cmd', 'en', 'count', 'omega_rad_s'])
                    for r, om in zip(rows, omega):
                        fw.writerow([r[0], WHEELS[w], r[2], r[3], r[1], f'{om:.4f}'])
                if info.get('imu_rows'):
                    with open(os.path.join(a.out, 'raw', run_id + '_imu.csv'), 'w', newline='') as f:
                        fw = csv.writer(f)
                        fw.writerow(['t_us', 'kind', 'v0', 'v1', 'v2'])
                        fw.writerows(info['imu_rows'])
                miss, ok, wss = qc(rows, omega, p['scenario'])
                done.append(dict(run_id=run_id, wheel=WHEELS[w], scenario=p['scenario'],
                                 pwm_level=p['pwm_level'], direction=p['direction'], rep=p['rep'],
                                 v_batt_start=v0, file=fname, order=p['order'], n_samples=len(rows),
                                 aborted=info.get('aborted'), reason=info.get('reason'),
                                 imu_n=info.get('imu_n'), imu_overflow=info.get('imu_overflow'),
                                 missing_pct=f'{miss:.2f}', steady_ok=ok, omega_ss_rad_s=wss))
                flag = '' if miss <= 1.0 and info.get('aborted') == '0' else '  <-- CHECK'
                print(f'  [{i:3d}/{len(block)}] {run_id:28s} n={len(rows):5d} miss={miss:4.1f}% '
                      f'steady={ok!s:2s} w_ss={wss or "-":>8s}{flag}')
            v1 = ask_voltage(f'[{WHEELS[w]}] battery voltage AFTER block (V): ', a.dry_run)
            block_ok = int(v0 - v1 <= VBATT_MAX_DROP)
            if not block_ok:
                print(f'  battery dropped {v0 - v1:.2f} V > {VBATT_MAX_DROP} V: recharge and redo this wheel')
            for d in done:
                d.update(v_batt_end=v1, block_ok=block_ok)
                wr.writerow(d)
            fruns.flush()
    except KeyboardInterrupt:
        dev.stop()
        print('\nStopped. Runs of the unfinished block were not written to runs.csv.')
    finally:
        fruns.close()

    meta = os.path.join(a.out, 'meta.yaml')
    if not os.path.exists(meta):
        with open(meta, 'w', encoding='utf-8') as f:
            f.write(f"""spec: S01
run: 1                      # edit: run number of this folder
date: {time.strftime('%Y-%m-%d')}
operator: {a.operator}
firmware:
  rpi5:  null               # not used (AGV only, wheels lifted)
  h7:    null
  esp32: {{repo: AGV_MECANUM, commit: {esp['commit']}, branch: {esp['branch']}, pio_env: s01_openloop, build_flags: "-DTEST_MODE -DTEST_OPENLOOP", dirty: {str(esp['dirty']).lower()}, modified: {esp['modified']}}}   # read by the runner from git; must match the build that was flashed
setup: real_agv_lifted
dry_run: {str(a.dry_run).lower()}       # true = simulated motor, NOT a measurement
config:
  wheels: {[WHEELS[w] for w in a.wheels]}
  reps: {a.reps}
  quick: {str(a.quick).lower()}
  plan_seed: {a.seed}
  sample_rate_hz: 1000
  counts_per_rev: {COUNTS_PER_REV}
  motor_driver: "BTS7960 (IBT-2), EN shared on GPIO45, PWM 20 kHz"
  pwm_sign: "pwm_cmd > 0 = physical RPWM high, LPWM pulsed (motor.c swaps pins); off-phase = brake"
  wheel_direction_at_positive_pwm: {{FL: null, FR: null, RL: null, RR: null}}   # edit after trial run: forward/backward, and count sign
  motor_resistance_ohm: null   # edit later: R at driver OUT terminals, all readings + probe offset (see S01 guide)
files:
  - runs.csv: "one row per run; v_batt in V; steady_ok = omega spread < 2% in last 0.5 s of hold"
  - raw/*.csv: "t_us from run start, pwm_cmd -1000..1000, en, count (1320/rev), omega_rad_s (+-5 ms central difference)"
  - raw/*_imu.csv: "firmware OL v2 only: IMU reports during the run, t_us = host receive time from run start; kind A accel (m/s^2 x,y,z), G calibrated gyro (rad/s x,y,z), R game RV euler (deg roll,pitch,yaw)"
notes: ""
""")
    print(f'\nDone. Output in {a.out}')


if __name__ == '__main__':
    main()
