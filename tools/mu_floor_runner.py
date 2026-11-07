"""
mu_floor_runner.py: PC side of the effective wheel-floor friction test (mu_hd) on the car.

Method (spec controller-IMU v2 section 1b.4, Liam 27/10): a rope ties the car to a fixed point
through a force gauge; the four wheels push the car away from that point (forward for XP, left for
YP, ...); PWM rises very slowly until the wheels spin; the largest force read just before the spin
(video of the gauge) is F_max; mu_hd = F_max / (m g) = F_max[kgf] / m[kg]. Five runs straight and
five sideways (spec 1b.4); the S03 mu levels are low = min, middle = mean, high = max of the ten
(S03 section 2.2). The same is repeated on the slippery sheet for X3 (--surface mica).
Liam 07/11, CODE session: "lam het viec A di".

Drives the ESP32 env mu_floor (AGV_MECANUM src/tests/test_mu_floor.c) over its USB-UART at
921600 baud and writes:
  <out>/raw/<trial>.csv    t_ms, pwm, c_fl, c_fr, c_rl, c_rr, ax, ay, az, gz, imu_n (100 Hz)
  <out>/trials.csv         one row per trial: direction, surface, end reason, PWM and time at the
                           automatic slip stop, F_max typed by Liam (kgf), mu_hd
  <out>/summary.csv        per surface: n, min, mean, max of mu_hd (both directions together) and
                           per direction
  <out>/meta.yaml          what the script knows (commits, mass, gauge, settings)

Usage
  python tools/mu_floor_runner.py --port COM5 --out 05_DATA/results/MU_FLOOR/<date>_run01 \\
      --esp32-commit <sha> --mass-kg <m> --gauge "<model, range, resolution>"
  python tools/mu_floor_runner.py --dry-run --out <dir> --mass-kg 4.0     # no hardware

After each ramp the script asks for F_max read on the video (kgf, as the gauge shows it); Enter
alone marks the trial bad (redo it). Ctrl+C sends STOP; the ESP32 also stops by itself without a
heartbeat for 0.5 s. The independent stop is the motor power switch.
Simulation of --dry-run is invented, NOT a measurement.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time

G = 9.80665
COUNTS_PER_REV = 1320
DIRS = ('XP', 'XN', 'YP', 'YN')
RAW_COLS = ['t_ms', 'pwm', 'c_fl', 'c_fr', 'c_rl', 'c_rr', 'ax', 'ay', 'az', 'gz', 'imu_n']


def mu_from_force(f_kgf, mass_kg):
    """mu_hd = F_max / (m g); with F in kgf this is F_kgf / m_kg."""
    if not (mass_kg > 0) or not (f_kgf >= 0):
        raise ValueError('mass > 0 and F >= 0 needed')
    return f_kgf * G / (mass_kg * G)


def summarize(trials):
    """Per surface (and per surface and direction): n, min, mean, max of mu over the good trials
    (mu not None). The S03 levels are low = min, mid = mean, high = max (S03 2.2)."""
    out = []
    groups = {}
    for t in trials:
        if t.get('mu') is None:
            continue
        groups.setdefault((t['surface'], 'all'), []).append(t['mu'])
        groups.setdefault((t['surface'], t['direction']), []).append(t['mu'])
    for (surf, d), v in sorted(groups.items()):
        out.append(dict(surface=surf, direction=d, n=len(v), mu_min=min(v), mu_mean=sum(v) / len(v), mu_max=max(v)))
    return out


def parse_data(line):
    """'D,<t_ms>,<pwm>,<c_fl>,<c_fr>,<c_rl>,<c_rr>,<ax>,<ay>,<az>,<gz>,<imu_n>' -> row dict, else None."""
    if not line.startswith('D,'):
        return None
    p = line.split(',')[1:]
    if len(p) != len(RAW_COLS):
        return None
    try:
        v = [int(x) for x in p[:6]] + [float(x) for x in p[6:10]] + [int(p[10])]
    except ValueError:
        return None
    return dict(zip(RAW_COLS, v))


def parse_kv(line):
    """'DONE reason=slip t_ms=123 pwm=400' -> {'reason': 'slip', 't_ms': '123', 'pwm': '400'}."""
    return dict(kv.split('=', 1) for kv in line.split()[2 if line.startswith('EV') else 1:] if '=' in kv)


class SerialDevice:
    """USB-UART link to env mu_floor (same opening sequence as s01_openloop_runner.py)."""

    def __init__(self, port):
        try:
            import serial
            serial.Serial
        except (ImportError, AttributeError):
            sys.exit('pyserial is missing. Fix:  python -m pip uninstall -y serial  &&  python -m pip install pyserial')
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, 921600, 0.2
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.lock = threading.Lock()
        self.ser.rts = True                       # pulse EN: restart the firmware cleanly
        time.sleep(0.15)
        self.ser.rts = False
        self.ready = ''
        t_end = time.time() + 6.0                 # the firmware waits up to 2 s for the IMU
        while time.time() < t_end:
            b = self.ser.readline().decode(errors='replace').strip()
            if b.startswith('MU READY'):
                self.ready = b
                break
        for _ in range(10):
            self.write('PING')
            t_end = time.time() + 0.5
            while time.time() < t_end:
                if b'PONG' in self.ser.readline():
                    return
        sys.exit(f'No PONG from ESP32 on {port}. Check: env mu_floor flashed on the CAR ESP32, right COM '
                 'port, no serial monitor open.')

    def write(self, line):
        with self.lock:
            self.ser.write((line + '\n').encode())

    def readline(self, timeout=5.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            b = self.ser.readline()
            if b:
                return b.decode(errors='replace').strip()
        raise TimeoutError('no line from ESP32')

    def ramp(self, direction, pwm0, rate, pwm_max, w_slip, on_line=None):
        self.write(f'RAMP {direction} {pwm0} {rate} {pwm_max} {w_slip:g}')
        r = self.readline()
        while not (r.startswith('OK RAMP') or r.startswith('ERR')):
            r = self.readline()
        if r.startswith('ERR'):
            raise RuntimeError(f'RAMP refused: {r}')
        stop_hb = threading.Event()

        def heartbeat():
            while not stop_hb.is_set():
                self.write('H')
                time.sleep(0.2)
        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        rows, ev = [], {}
        try:
            while True:
                r = self.readline(timeout=3.0)
                if on_line:
                    on_line(r)
                d = parse_data(r)
                if d:
                    rows.append(d)
                elif r.startswith('EV SLIP'):
                    ev['slip'] = parse_kv(r)
                elif r.startswith('DONE'):
                    ev['done'] = parse_kv(r)
                    break
        except KeyboardInterrupt:
            self.write('STOP')
            raise
        finally:
            stop_hb.set()
            hb.join()
        return rows, ev

    def stop(self):
        self.write('STOP')


class FakeDevice:
    """Invented car on a spring gauge, for testing this script without hardware (NOT a measurement):
    wheel force grows with PWM above a dead band; the gauge stretches; above mu*m*g the wheels spin."""

    def __init__(self, mass_kg=4.0, mu=0.6, k_n_per_m=300.0):
        self.ready = 'MU READY v1 imu=ok (dry run)'
        self.m, self.mu, self.k = mass_kg, mu, k_n_per_m

    def ramp(self, direction, pwm0, rate, pwm_max, w_slip, on_line=None):
        rows, ev, c, x = [], {}, 0, 0.0
        sign = [1, 1, 1, 1] if direction[0] == 'X' else [-1, 1, 1, -1]
        if direction[1] == 'N':
            sign = [-s for s in sign]
        for k in range(12000):
            t_ms = 10 * k
            pwm = min(pwm_max, pwm0 + rate * t_ms // 1000)
            f = max(0.0, (pwm - 150) * 0.05)                  # N, invented
            spin = f > self.mu * self.m * G
            x_new = min(f, self.mu * self.m * G) / self.k
            dc = int((x_new - x) / (2 * math.pi * 0.0485) * COUNTS_PER_REV) + (40 if spin else 0)
            x, c = x_new, c + dc
            rows.append(dict(t_ms=t_ms, pwm=pwm, c_fl=sign[0] * c, c_fr=sign[1] * c, c_rl=sign[2] * c,
                             c_rr=sign[3] * c, ax=0.0, ay=0.0, az=9.81, gz=0.0, imu_n=k // 5))
            w = 40 * 100 / COUNTS_PER_REV * 2 * math.pi if spin else 0.0
            if spin and w_slip > 0 and w > w_slip and k > 20:
                ev['slip'] = dict(t_ms=str(t_ms), pwm=str(pwm), w=f'{w:.2f}')
                ev['done'] = dict(reason='slip', t_ms=str(t_ms), pwm=str(pwm))
                break
            if pwm >= pwm_max and (pwm0 + rate * t_ms // 1000) > pwm_max + 2 * rate:
                ev['done'] = dict(reason='max', t_ms=str(t_ms), pwm=str(pwm))
                break
        return rows, ev

    def stop(self):
        pass


def git_state(path):
    try:
        def g(*a):
            return subprocess.run(['git', '-C', path] + list(a), capture_output=True, text=True, check=True).stdout.strip()
        changed = [l[3:] for l in g('status', '--porcelain').splitlines() if l[3:]]
        return dict(commit=g('rev-parse', '--short', 'HEAD'), branch=g('rev-parse', '--abbrev-ref', 'HEAD'),
                    dirty=bool(changed), modified=changed)
    except (OSError, subprocess.CalledProcessError) as e:
        return dict(commit='UNKNOWN', dirty=True, modified=[str(e)])


def write_csv(path, rows, cols):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def ask_force(prompt, inp=input):
    """F_max in kgf typed by Liam; '' = bad trial (None)."""
    while True:
        s = inp(prompt).strip().replace(',', '.')
        if s == '':
            return None
        try:
            v = float(s)
            if v >= 0:
                return v
        except ValueError:
            pass
        print('  so >= 0 (kgf), hoac Enter de danh dau luot hong')


TRIAL_COLS = ['trial', 'surface', 'direction', 'rep', 'start', 'reason', 't_end_ms', 'pwm_end', 'slip_t_ms',
              'slip_pwm', 'slip_w', 'n_samples', 'f_max_kgf', 'mu', 'note']


def run(a, dev, inp=input):
    os.makedirs(os.path.join(a.out, 'raw'), exist_ok=True)
    tpath = os.path.join(a.out, 'trials.csv')
    trials = []
    if os.path.isfile(tpath):                                   # resume: keep the trials already done
        with open(tpath, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                r['mu'] = float(r['mu']) if r['mu'] not in ('', 'None') else None
                trials.append(r)
    plan = [(d, rep) for d in a.dirs for rep in range(1, a.reps + 1)]
    for d, rep in plan:
        name = f'{a.surface}_{d}_{rep:02d}'
        if any(t['trial'] == name and t['mu'] is not None for t in trials):
            continue
        inp(f'\n[{name}] Dat day va luc ke ({d}), luc ke ve 0, bat quay video, Enter de chay (Ctrl+C dung): ')
        start = time.strftime('%Y-%m-%d %H:%M:%S')
        rows, ev = dev.ramp(d, a.pwm0, a.rate, a.pwm_max, a.w_slip)
        write_csv(os.path.join(a.out, 'raw', f'{name}.csv'), rows, RAW_COLS)
        done, slip = ev.get('done', {}), ev.get('slip', {})
        print(f'  ket thuc: {done.get("reason")} o PWM {done.get("pwm")}, {len(rows)} mau'
              + (f'; tu dung khi truot o PWM {slip.get("pwm")}' if slip else ''))
        f = ask_force('  F_max doc tren video (kgf), Enter = luot hong: ', inp)
        trials = [t for t in trials if t['trial'] != name]
        trials.append(dict(trial=name, surface=a.surface, direction=d, rep=rep, start=start,
                           reason=done.get('reason', ''), t_end_ms=done.get('t_ms', ''), pwm_end=done.get('pwm', ''),
                           slip_t_ms=slip.get('t_ms', ''), slip_pwm=slip.get('pwm', ''), slip_w=slip.get('w', ''),
                           n_samples=len(rows), f_max_kgf='' if f is None else f,
                           mu=None if f is None else mu_from_force(f, a.mass_kg),
                           note='' if f is not None else 'bad trial, redo'))
        write_csv(tpath, [dict(t, mu='' if t['mu'] is None else t['mu']) for t in trials], TRIAL_COLS)
        if f is not None:
            print(f'  mu_hd = {mu_from_force(f, a.mass_kg):.3f}')
    summ = summarize(trials)
    write_csv(os.path.join(a.out, 'summary.csv'), summ, ['surface', 'direction', 'n', 'mu_min', 'mu_mean', 'mu_max'])
    for s in summ:
        print(f"{s['surface']} {s['direction']}: n={s['n']} min={s['mu_min']:.3f} mean={s['mu_mean']:.3f} "
              f"max={s['mu_max']:.3f}")
    return trials, summ


def write_meta(a, dev, here):
    meta = dict(
        spec='spec controller-IMU v2 section 1b.4 (mu_hd); S03 section 2.2 (mu levels from 10 runs)',
        date=time.strftime('%Y-%m-%d'),
        setup='REAL car on the floor (dry run: invented simulation)' if not a.dry_run else 'dry run, NOT a measurement',
        operator='Liam (car, gauge, video, F_max reading); script tools/mu_floor_runner.py',
        firmware=dict(esp32=dict(commit=a.esp32_commit, env='mu_floor', board='ESP32 on the car',
                                 dirty=a.esp32_dirty), ready_line=dev.ready),
        script=git_state(here), mass_kg=a.mass_kg, gauge=a.gauge, surface=a.surface,
        ramp=dict(pwm0=a.pwm0, rate_pwm_per_s=a.rate, pwm_max=a.pwm_max, w_slip_rad_s=a.w_slip, dirs=a.dirs, reps=a.reps),
        method=('rope from the car to a fixed point through the gauge, the car pushes away from it; PWM rises '
                'slowly on four wheels with the sign of kinematics_inverse for the direction; F_max = largest gauge '
                'reading just before the wheels spin, read on the video by Liam; mu_hd = F_max[kgf] / m[kg]'),
        files=dict(raw='raw/<surface>_<dir>_<rep>.csv (100 Hz)', trials='trials.csv', summary='summary.csv'),
    )
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write('# MU_FLOOR run (written by tools/mu_floor_runner.py; every value is JSON, so the file is valid YAML)\n')
        for k, v in meta.items():
            f.write(f'{k}: {json.dumps(v, ensure_ascii=False)}\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--esp32-commit', default='')
    ap.add_argument('--esp32-dirty', action='store_true')
    ap.add_argument('--mass-kg', type=float, required=True, help='car mass on a scale, kg')
    ap.add_argument('--gauge', default='', help='force gauge: model, range, resolution')
    ap.add_argument('--surface', default='floor', help='floor, or mica for X3 (slippery sheet under the wheels)')
    ap.add_argument('--dirs', nargs='+', default=['XP', 'YP'], choices=DIRS)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--pwm0', type=int, default=100)
    ap.add_argument('--rate', type=int, default=10, help='PWM per second')
    ap.add_argument('--pwm-max', type=int, default=1000)
    ap.add_argument('--w-slip', type=float, default=3.0, help='rad/s, automatic stop (0 = off)')
    a = ap.parse_args(argv)
    if not a.dry_run and (not a.port or not a.esp32_commit):
        sys.exit('--port and --esp32-commit are needed (or --dry-run)')
    here = os.path.dirname(os.path.abspath(__file__))
    dev = FakeDevice(mass_kg=a.mass_kg) if a.dry_run else SerialDevice(a.port)
    if 'imu=fail' in dev.ready:
        sys.exit(f'{dev.ready}: the IMU did not start; the firmware refuses RAMP (check the BNO085, power cycle)')
    os.makedirs(a.out, exist_ok=True)
    write_meta(a, dev, here)
    try:
        run(a, dev)
    except KeyboardInterrupt:
        dev.stop()
        print('\nSTOP sent; trials.csv keeps the trials done, run again with the same --out to go on')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
