"""
mu_floor_runner.py: PC side of the effective wheel-floor friction test (mu_hd) on the car.

Method (spec controller-IMU v2 section 1b.4, Liam 27/10): a rope ties the car to a fixed point
through a force gauge; the four wheels push the car away from that point (forward for XP, left for
YP, ...); PWM rises very slowly until the wheels spin; the largest force read just before the spin
(video of the gauge) is F_max; mu_hd = F_max / (m g) = F_max[kgf] / m[kg]. Five runs straight and
five sideways (spec 1b.4); the S03 mu levels are low = min, middle = mean, high = max of the ten
(S03 section 2.2). The same is repeated on the slippery sheet for X3 (--surface mica).
Liam 07/11, CODE session: "lam het viec A di".

Link (protocol v2, Liam 09/11, CODE session: no USB cable while the car pushes, commands over WiFi
from the laptop; "Duyet, code di" for UDP port 5013 and the 2 % rule below):
  --udp auto       broadcast PING to port 5013 (255.255.255.255 and a.b.c.255 of every IPv4 address of
                   this PC, or --bcast), the ESP32 that answers PONG is used
  --udp <IP>       that address (the ESP32 prints "EV WIFI ip=..." on its UART at boot; PONG has ip=)
  --port COMx      USB-UART 921600 (bench only; restarts the ESP32)
The ESP32 env mu_floor is AGV_MECANUM src/tests/test_mu_floor.c (protocol in its header comment).
Every RAMP has an id (from PONG last_id + 1); without an answer (OK RAMP, EV START or a D line) in
1 s the same RAMP is sent again, at most 3 times; the ESP32 never starts a second run for the same
id. A heartbeat 'H' goes every 100 ms from the first RAMP to the end; the ESP32 stops by itself
without a line for 0.5 s. Ctrl+C, a lost link (no line for 3 s) or a missing DONE send STOP 3 times.
A trial is BAD (no F_max asked, redo it) when there is no EV START, no DONE, the ESP32 stopped on
the heartbeat, or more than 2 % of the D lines (seq 0 .. n-1 from DONE) did not arrive.

Writes:
  <out>/raw/<trial>.csv    t_ms, pwm, c_fl, c_fr, c_rl, c_rr, ax, ay, az, gz, imu_n, seq (100 Hz)
  <out>/trials.csv         one row per trial: direction, surface, end reason, PWM and time at the
                           automatic slip stop, link check, F_max typed by Liam (kgf), mu_hd
  <out>/summary.csv        per surface: n, min, mean, max of mu_hd (both directions together) and
                           per direction
  <out>/meta.yaml          what the script knows (firmware from PONG, link, mass, gauge, settings)

Usage
  python tools/mu_floor_runner.py --udp auto --out 05_DATA/results/MU_FLOOR/<date>_run01 \\
      --mass-kg 4.12 --gauge "<model, range, resolution>" --surface floor
  python tools/mu_floor_runner.py --dry-run --out <dir> --mass-kg 4.0     # no hardware

After each ramp the script asks for F_max read on the video (kgf, as the gauge shows it); Enter
alone marks the trial bad (redo it). The independent stop is the motor power switch.
Simulation of --dry-run is invented, NOT a measurement.

Other tests on the same firmware (--test, Liam 11/11 CODE session local_b905f2ba: "Sua ca hai";
guide 00_OVERVIEW/Setup_Guides/2026-11-11_Huong_dan_Buoi_xe_san_thang_momen_yaw.md; Fable item 6
M-02 and item 4 F-01, F-09; L87 Q2, L85 Q5, Q7). The firmware and the link rules are the same.
  --test mu    (default) the friction test above; nothing changes
  --test hold  loaded torque scale: rope and gauge as for mu, PWM rises slowly to --pwm-max and holds
               2 s; the script asks for the gauge reading just before the long beep (gauge_kgf)
  --test step  open-loop duty step, no rope: --pwm0 must equal --pwm-max (PWM jumps at t = 0, held
               2 s) and --w-slip must be 0 (the wheels turn by design); the script asks whether the
               trial was clean
hold and step name the trials <test><pwm_max>_<dir>_<rep> (hold250_XP_01), ask the battery rest
voltage before every trial (refused below --vbat-min, charge first; Liam 11/11: 12.2 V) and after
it (Enter = not measured), compute no mu and write no summary.csv. Every trial keeps its own ramp
settings in trials.csv (several commands may share one --out); commands.csv lists every command.
"""
import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time

G = 9.80665
COUNTS_PER_REV = 1320
DIRS = ('XP', 'XN', 'YP', 'YN')
RAW_COLS = ['t_ms', 'pwm', 'c_fl', 'c_fr', 'c_rl', 'c_rr', 'ax', 'ay', 'az', 'gz', 'imu_n', 'seq']
UDP_PORT = 5013            # test_mu_floor.c MU_UDP_PORT
HB_PERIOD_S = 0.1          # heartbeat; the ESP32 stops after 0.5 s without a line
ACK_S = 1.0                # wait for OK RAMP / EV START / D before sending the RAMP again
RAMP_SENDS = 3
LINE_TIMEOUT_S = 3.0       # no line for this long during a run = link lost
MISSING_MAX = 0.02         # more missing D lines than this fraction = bad trial [Liam 09/11]
FW_ENV = 'mu_floor'
TESTS = ('mu', 'hold', 'step')
VBAT_MIN = 12.2            # V, battery rest voltage for hold and step [Liam 11/11, guide 7a]
METHODS = dict(
    mu=('rope from the car to a fixed point through the gauge, the car pushes away from it; PWM rises '
        'slowly on four wheels with the sign of kinematics_inverse for the direction; F_max = largest gauge '
        'reading just before the wheels spin, read on the video by Liam; mu_hd = F_max[kgf] / m[kg]'),
    hold=('loaded torque scale (Fable item 6 M-02): rope from the car to a fixed point through the gauge as for '
          'mu; PWM rises at --rate from --pwm0 to --pwm-max on four wheels with the sign of kinematics_inverse, '
          'holds 2 s at --pwm-max, then the motors stop (long beep); gauge_kgf = gauge reading just before the '
          'long beep, read by Liam (video kept); a run that ends on slip is kept (the wheels spun below '
          '--pwm-max); per-trial ramp settings in trials.csv'),
    step=('open-loop duty step (Fable item 4 F-01): no rope; PWM jumps from 0 to --pwm-max at t = 0 on four '
          'wheels with the sign of kinematics_inverse, held 2 s, then the motors stop; the car runs straight on '
          'the floor and is carried back to the start line; the log (counts of the four wheels, BNO085 accel, '
          '100 Hz) is the data; per-trial ramp settings in trials.csv'),
)


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
    """'D,<t_ms>,<pwm>,<c_fl>,<c_fr>,<c_rl>,<c_rr>,<ax>,<ay>,<az>,<gz>,<imu_n>,<seq>' -> row dict, else None."""
    if not line.startswith('D,'):
        return None
    p = line.split(',')[1:]
    if len(p) != len(RAW_COLS):
        return None
    try:
        v = [int(x) for x in p[:6]] + [float(x) for x in p[6:10]] + [int(p[10]), int(p[11])]
    except ValueError:
        return None
    return dict(zip(RAW_COLS, v))


def parse_kv(line):
    """'DONE reason=slip t_ms=123 pwm=400' -> {'reason': 'slip', 't_ms': '123', 'pwm': '400'}.
    'EV X ...' and 'OK RAMP ...', 'DUP RAMP ...' skip two words."""
    skip = 2 if line.split(' ', 1)[0] in ('EV', 'OK', 'DUP') else 1
    return dict(kv.split('=', 1) for kv in line.split()[skip:] if '=' in kv)


def link_check(rows, ev, max_missing=MISSING_MAX):
    """Whether a trial's data arrived whole. n_expected = n of DONE (D lines the ESP32 sent), else
    the highest seq + 1; n_missing = seq in 0 .. n_expected-1 not received. Bad when there is no
    EV START, no DONE, the ESP32 stopped on the heartbeat, or n_missing > max_missing * n_expected."""
    done = ev.get('done')
    seqs = {r['seq'] for r in rows}
    try:
        expected = int(done['n']) if done and 'n' in done else (max(seqs) + 1 if seqs else 0)
    except ValueError:
        expected = max(seqs) + 1 if seqs else 0
    missing = expected - len({s for s in seqs if 0 <= s < expected})
    why = []
    if 'start' not in ev:
        why.append('no EV START')
    if not done:
        why.append('no DONE' + (f" ({ev['link']})" if ev.get('link') else ''))
    elif done.get('reason') == 'heartbeat':
        why.append('ESP32 stopped: no heartbeat for 0.5 s')
    if expected and missing > max_missing * expected:
        why.append(f'{missing}/{expected} D lines missing (> {max_missing:.0%})')
    if ev.get('link') == 'dup':
        why.append('RAMP answered DUP after a resend (the run went on without our acks)')
    return dict(n_expected=expected, n_missing=missing, link_ok=not why, link_note='; '.join(why))


class LineLink:
    """Common protocol v2 over a line link (subclasses: write(line), readline(timeout) -> str|None)."""

    kind = ''
    ready = ''

    def hello(self, tries=10):
        """PING until PONG; keeps the PONG fields (ip, imu, fw, dirty, env, last_id...) in self.pong."""
        for _ in range(tries):
            self.write('PING')
            t_end = time.time() + 0.5
            while time.time() < t_end:
                r = self.readline(t_end - time.time())
                if r and r.startswith('PONG'):
                    self.pong_line, self.pong = r, parse_kv(r)
                    self.next_id = int(self.pong.get('last_id', '0')) + 1
                    if not self.ready:
                        self.ready = r
                    return self.pong
        return None

    def stop(self, times=3):
        for _ in range(times):
            self.write('STOP')
            time.sleep(0.05)

    def ramp(self, direction, pwm0, rate, pwm_max, w_slip, on_line=None):
        rid = self.next_id
        self.next_id += 1
        cmd = f'RAMP {rid} {direction} {pwm0} {rate} {pwm_max} {w_slip:g}'
        stop_hb = threading.Event()

        def heartbeat():
            while not stop_hb.wait(HB_PERIOD_S):
                self.write('H')
        hb = threading.Thread(target=heartbeat, daemon=True)
        rows, ev = [], dict(id=rid, sends=0)

        def take(r):
            """Handles one line; returns 'ack', 'done' or None."""
            if on_line:
                on_line(r)
            d = parse_data(r)
            if d:
                rows.append(d)
                return 'ack'
            if r.startswith(f'OK RAMP id={rid}'):
                return 'ack'
            if r.startswith('EV START'):
                ev['start'] = parse_kv(r)
                return 'ack'
            if r.startswith('EV SLIP'):
                ev['slip'] = parse_kv(r)
            elif r.startswith('DONE'):
                ev['done'] = parse_kv(r)
                return 'done'
            elif r.startswith('DUP RAMP'):
                ev['dup'] = r
            elif r.startswith('ERR'):
                ev['err'] = r
            return None

        try:
            acked = False
            for attempt in range(RAMP_SENDS):
                self.write(cmd)
                ev['sends'] += 1
                if attempt == 0:
                    hb.start()
                t_end = time.time() + ACK_S
                while not acked and time.time() < t_end:
                    r = self.readline(t_end - time.time())
                    if r is None:
                        break
                    k = take(r)
                    if 'err' in ev:
                        raise RuntimeError(f'RAMP refused: {ev["err"]}')
                    if 'dup' in ev:
                        ev['link'] = 'dup' if attempt > 0 else 'stale id'
                        return rows, ev
                    acked = k in ('ack', 'done')
                    if k == 'done':
                        return rows, ev
                if acked:
                    break
            if not acked:
                ev['link'] = f'no answer to RAMP after {RAMP_SENDS} sends'
                return rows, ev
            while True:
                r = self.readline(LINE_TIMEOUT_S)
                if r is None:
                    ev['link'] = f'no line for {LINE_TIMEOUT_S:g} s'
                    break
                if take(r) == 'done':
                    break
        except KeyboardInterrupt:
            self.stop()
            raise
        finally:
            stop_hb.set()
            if hb.is_alive():
                hb.join()
            if 'done' not in ev:
                self.stop()
            if 'dup' in ev:
                self.hello(tries=3)                       # refresh next_id
        return rows, ev


class SerialLink(LineLink):
    """USB-UART link to env mu_floor (same opening sequence as s01_openloop_runner.py). Bench only."""

    kind = 'serial'

    def __init__(self, port):
        try:
            import serial
            serial.Serial
        except (ImportError, AttributeError):
            sys.exit('pyserial is missing. Fix:  python -m pip uninstall -y serial  &&  python -m pip install pyserial')
        self.port = port
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, 921600, 0.05
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.lock = threading.Lock()
        self.ser.rts = True                       # pulse EN: restart the firmware cleanly
        time.sleep(0.15)
        self.ser.rts = False
        t_end = time.time() + 6.0                 # the firmware waits up to 2 s for the IMU
        while time.time() < t_end:
            b = self.ser.readline().decode(errors='replace').strip()
            if b.startswith('MU READY'):
                self.ready = b
                break
        if not self.hello():
            sys.exit(f'No PONG from ESP32 on {port}. Check: env mu_floor flashed, right COM port, no serial '
                     'monitor open.')

    def describe(self):
        return dict(kind='serial', port=self.port, baud=921600)

    def write(self, line):
        with self.lock:
            self.ser.write((line + '\n').encode())

    def readline(self, timeout):
        t_end = time.time() + max(0.0, timeout)
        while True:
            b = self.ser.readline()
            if b:
                s = b.decode(errors='replace').strip()
                if s:
                    return s
            if time.time() >= t_end:
                return None


def broadcast_targets():
    """255.255.255.255 and a.b.c.255 for every IPv4 address of this PC (a /24 net assumed): on a PC
    with another adapter (bench PC 09/11: ZeroTier) the limited broadcast alone did not reach the
    ESP32, 192.168.1.255 did."""
    out = ['255.255.255.255']
    try:
        ips = {ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        ips = set()
    for ip in sorted(ips):
        if not ip.startswith('127.'):
            b = ip.rsplit('.', 1)[0] + '.255'
            if b not in out:
                out.append(b)
    return out


class UdpLink(LineLink):
    """WiFi UDP link to env mu_floor. host 'auto': PING to bcast:port, the first PONG sets the host.
    Only datagrams from that host are read afterwards."""

    kind = 'udp'

    def __init__(self, host, port=UDP_PORT, bcast='auto', tries=10):
        self.port, self.bcast = port, bcast
        self.targets = broadcast_targets() if bcast == 'auto' else [bcast]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(('', 0))
        self.lock = threading.Lock()
        self.lines = []
        self.host = None if host == 'auto' else host
        if self.host is None:
            self.host = self._discover(tries)
            if not self.host:
                sys.exit(f'No PONG to a broadcast PING on {self.targets} port {port}. Check: car ESP32 on (env mu_floor), '
                         'same WiFi as this PC, Windows firewall lets python receive UDP; or give --udp <IP> '
                         '(the ESP32 prints "EV WIFI ip=..." on its UART).')
        elif not self.hello(tries):
            sys.exit(f'No PONG from {self.host}:{port}. Check the IP, the WiFi, the firmware env mu_floor.')

    def _discover(self, tries):
        for _ in range(tries):
            for b in self.targets:
                try:
                    self.sock.sendto(b'PING\n', (b, self.port))
                except OSError:                       # an interface without a route: try the others
                    pass
            t_end = time.time() + 0.5
            while time.time() < t_end:
                got = self._recv(t_end - time.time(), any_host=True)
                if got and got[0].startswith('PONG'):
                    self.host = got[1]
                    self.pong_line, self.pong = got[0], parse_kv(got[0])
                    self.next_id = int(self.pong.get('last_id', '0')) + 1
                    self.ready = got[0]
                    return self.host
        return None

    def _recv(self, timeout, any_host=False):
        """One line and its source host, or None after timeout."""
        t_end = time.time() + max(0.0, timeout)
        while True:
            if self.lines:
                return self.lines.pop(0)
            left = t_end - time.time()
            if left <= 0:
                return None
            self.sock.settimeout(left)
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                return None
            except ConnectionResetError:          # Windows: ICMP port unreachable from an earlier send
                continue
            if not any_host and addr[0] != self.host:
                continue
            for s in data.decode(errors='replace').splitlines():
                if s.strip():
                    self.lines.append((s.strip(), addr[0]))

    def describe(self):
        return dict(kind='udp', host=self.host, port=self.port, bcast=self.bcast)

    def write(self, line):
        with self.lock:
            self.sock.sendto((line + '\n').encode(), (self.host, self.port))

    def readline(self, timeout):
        got = self._recv(timeout)
        return got[0] if got else None


class FakeDevice:
    """Invented car on a spring gauge, for testing this script without hardware (NOT a measurement):
    wheel force grows with PWM above a dead band; the gauge stretches; above mu*m*g the wheels spin."""

    kind = 'dry-run'

    def __init__(self, mass_kg=4.0, mu=0.6, k_n_per_m=300.0):
        self.ready = 'MU READY v2 imu=ok (dry run)'
        self.pong = dict(ip='none', imu='ok', fw='dryrun', dirty='0', env=FW_ENV, last_id='0')
        self.pong_line = 'PONG (dry run)'
        self.m, self.mu, self.k = mass_kg, mu, k_n_per_m
        self.next_id = 1

    def describe(self):
        return dict(kind='dry-run')

    def ramp(self, direction, pwm0, rate, pwm_max, w_slip, on_line=None):
        rid = self.next_id
        self.next_id += 1
        rows, ev, c, x = [], dict(id=rid, sends=1, start=dict(id=str(rid), t_ms='0')), 0, 0.0
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
                             c_rr=sign[3] * c, ax=0.0, ay=0.0, az=9.81, gz=0.0, imu_n=k // 5, seq=k))
            w = 40 * 100 / COUNTS_PER_REV * 2 * math.pi if spin else 0.0
            if spin and w_slip > 0 and w > w_slip and k > 20:
                ev['slip'] = dict(t_ms=str(t_ms), pwm=str(pwm), w=f'{w:.2f}')
                ev['done'] = dict(reason='slip', t_ms=str(t_ms), pwm=str(pwm), id=str(rid), n=str(k + 1), udp_err='0')
                break
            if pwm >= pwm_max and (pwm0 + rate * t_ms // 1000) > pwm_max + 2 * rate:
                ev['done'] = dict(reason='max', t_ms=str(t_ms), pwm=str(pwm), id=str(rid), n=str(k + 1), udp_err='0')
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


TRIAL_COLS = ['trial', 'surface', 'direction', 'rep', 'start', 'run_id', 'sends', 'reason', 't_end_ms', 'pwm_end',
              'slip_t_ms', 'slip_pwm', 'slip_w', 'n_samples', 'n_expected', 'n_missing', 'udp_err', 'link_ok',
              'f_max_kgf', 'mu', 'note',
              'test', 'pwm0', 'rate', 'pwm_max', 'w_slip', 'gauge_kgf', 'vbat_before_v', 'vbat_after_v', 'ok']


def ask_number(prompt, inp=input, lo=0.0, hi=float('inf'), default=None):
    """A number in [lo, hi] (comma or dot); '' returns default (None = not given)."""
    while True:
        s = inp(prompt).strip().replace(',', '.')
        if s == '':
            return default
        try:
            v = float(s)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f'  so trong [{lo:g}, {hi:g}], hoac Enter')


def ask_vbat_before(prev, vmin, inp=input):
    """Battery rest voltage before a trial; Enter keeps prev; below vmin the trial does not start."""
    while True:
        hint = f' [Enter = {prev:g}]' if prev is not None else ''
        v = ask_number(f'  Dien ap pin nghi truoc luot (V){hint}, Enter xong la xe CHAY: ', inp, 9.0, 13.0, prev)
        if v is None:
            print('  can so dien ap (do bang dong ho van nang)')
        elif v < vmin:
            print(f'  Pin {v:g} V < {vmin:g} V: sac pin roi go lai so moi')
            prev = None
        else:
            return v


def trial_done(t):
    """A trial that need not run again: mu given (test mu, also old files) or ok (hold, step)."""
    if t.get('test', 'mu') in ('', 'mu'):
        return t.get('mu') is not None
    return str(t.get('ok')) == 'True'


def run(a, dev, inp=input):
    test = getattr(a, 'test', 'mu')
    vmin = getattr(a, 'vbat_min', VBAT_MIN)
    os.makedirs(os.path.join(a.out, 'raw'), exist_ok=True)
    tpath = os.path.join(a.out, 'trials.csv')
    trials = []
    if os.path.isfile(tpath):                                   # resume: keep the trials already done
        with open(tpath, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                r['mu'] = float(r['mu']) if r['mu'] not in ('', 'None') else None
                trials.append(r)
    plan = [(d, rep) for d in a.dirs for rep in range(1, a.reps + 1)]
    vbat = None
    for d, rep in plan:
        name = f'{a.surface}_{d}_{rep:02d}' if test == 'mu' else f'{test}{a.pwm_max}_{d}_{rep:02d}'
        if any(t['trial'] == name and trial_done(t) for t in trials):
            continue
        if test == 'mu':
            inp(f'\n[{name}] Dat day va luc ke ({d}), luc ke ve 0, bat quay video, Enter de chay (Ctrl+C dung): ')
        elif test == 'hold':
            inp(f'\n[{name}] Dat day va can ({d}), can ve 0, bat quay video mat can, Enter de tiep (Ctrl+C dung): ')
        else:
            inp(f'\n[{name}] Dat xe sau vach xuat phat, duong chay trong >= 1.5 m, Enter de tiep (Ctrl+C dung): ')
        vb_before = None
        if test != 'mu':                                       # the run starts right after this answer
            vbat = vb_before = ask_vbat_before(vbat, vmin, inp)
        start = time.strftime('%Y-%m-%d %H:%M:%S')
        rows, ev = dev.ramp(d, a.pwm0, a.rate, a.pwm_max, a.w_slip)
        rows = sorted({r['seq']: r for r in rows}.values(), key=lambda r: r['seq'])   # one row per seq
        write_csv(os.path.join(a.out, 'raw', f'{name}.csv'), rows, RAW_COLS)
        done, slip = ev.get('done', {}), ev.get('slip', {})
        chk = link_check(rows, ev)
        print(f'  ket thuc: {done.get("reason")} o PWM {done.get("pwm")}, {len(rows)}/{chk["n_expected"]} mau'
              + (f'; tu dung khi truot o PWM {slip.get("pwm")}' if slip else ''))
        f = g = vb_after = None
        ok = False
        if not chk['link_ok']:
            print(f'  LUOT HONG (duong truyen): {chk["link_note"]}; chay lai luot nay')
            note = f'link: {chk["link_note"]}'
        elif test == 'mu':
            f = ask_force('  F_max doc tren video (kgf), Enter = luot hong: ', inp)
            note = '' if f is not None else 'bad trial, redo'
        elif test == 'hold':
            slipped = done.get('reason') == 'slip'
            if slipped:
                print('  BANH QUAY TRON truoc pwm_max: giu luot nay (la ket qua), khong chay lai')
            g = ask_force('  So can ngay truoc bip dai (kgf), Enter = ' + ('khong doc duoc: ' if slipped else
                                                                             'luot hong: '), inp)
            ok = g is not None or slipped
            note = ('slip before pwm_max' if slipped else '') if ok else 'bad trial, redo'
        else:
            full = done.get('reason') == 'max'
            if not full:
                print(f'  Luot khong chay du 2 s (ket thuc {done.get("reason")}): chay lai')
                note = f'step ended on {done.get("reason")}'
            else:
                s = inp('  Luot sach? Enter = sach, chu bat ky + Enter = hong (xe dam vat, truot, ...): ').strip()
                ok = s == ''
                note = '' if ok else f'bad trial, redo: {s}'
        if test != 'mu':
            vb_after = ask_number('  Dien ap pin nghi sau luot (V), Enter = khong do: ', inp, 9.0, 13.0)
            if vb_after is not None:
                vbat = vb_after
        trials = [t for t in trials if t['trial'] != name]
        trials.append(dict(trial=name, surface=a.surface, direction=d, rep=rep, start=start, run_id=ev.get('id', ''),
                           sends=ev.get('sends', ''), reason=done.get('reason', ''), t_end_ms=done.get('t_ms', ''),
                           pwm_end=done.get('pwm', ''), slip_t_ms=slip.get('t_ms', ''), slip_pwm=slip.get('pwm', ''),
                           slip_w=slip.get('w', ''), n_samples=len(rows), n_expected=chk['n_expected'],
                           n_missing=chk['n_missing'], udp_err=done.get('udp_err', ''), link_ok=chk['link_ok'],
                           f_max_kgf='' if f is None else f, mu=None if f is None else mu_from_force(f, a.mass_kg),
                           note=note, test=test, pwm0=a.pwm0, rate=a.rate, pwm_max=a.pwm_max, w_slip=a.w_slip,
                           gauge_kgf='' if g is None else g, vbat_before_v='' if vb_before is None else vb_before,
                           vbat_after_v='' if vb_after is None else vb_after,
                           ok=(f is not None) if test == 'mu' else ok))
        write_csv(tpath, [dict(t, mu='' if t['mu'] is None else t['mu']) for t in trials], TRIAL_COLS)
        if f is not None:
            print(f'  mu_hd = {mu_from_force(f, a.mass_kg):.3f}')
    if test != 'mu':
        return trials, []
    summ = summarize(trials)
    write_csv(os.path.join(a.out, 'summary.csv'), summ, ['surface', 'direction', 'n', 'mu_min', 'mu_mean', 'mu_max'])
    for s in summ:
        print(f"{s['surface']} {s['direction']}: n={s['n']} min={s['mu_min']:.3f} mean={s['mu_mean']:.3f} "
              f"max={s['mu_max']:.3f}")
    return trials, summ


def write_meta(a, dev, here):
    p = getattr(dev, 'pong', {}) or {}
    test = getattr(a, 'test', 'mu')
    spec = {'mu': 'spec controller-IMU v2 section 1b.4 (mu_hd); S03 section 2.2 (mu levels from 10 runs)',
            'hold': ('no spec: Fable item 6 M-02, L87 Q2 (03_CODE/docs/review/2026-11-11_Tham_dinh_Fable_mu_IMU.md); '
                     'battery L85 Q7'),
            'step': ('no spec: Fable item 4 F-01, L85 Q5, Q7 '
                     '(03_CODE/docs/review/2026-11-11_Tham_dinh_Fable_plant_va_O1.md)')}[test]
    meta = dict(
        spec=spec, test=test,
        date=time.strftime('%Y-%m-%d'),
        setup='REAL car on the floor (dry run: invented simulation)' if not a.dry_run else 'dry run, NOT a measurement',
        operator=('Liam (car, gauge, video, F_max reading); script tools/mu_floor_runner.py' if test == 'mu' else
                  'Liam (car, gauge and video for hold, battery voltmeter); script tools/mu_floor_runner.py'),
        firmware=dict(esp32=dict(commit=p.get('fw', ''), dirty=p.get('dirty', ''), env=p.get('env', ''),
                                 board='ESP32 on the car', source='PONG line of the board'),
                      ready_line=dev.ready, pong_line=getattr(dev, 'pong_line', '')),
        link=dev.describe(), link_rules=dict(heartbeat_s=HB_PERIOD_S, ack_s=ACK_S, ramp_sends=RAMP_SENDS,
                                             line_timeout_s=LINE_TIMEOUT_S, missing_max=MISSING_MAX),
        script=git_state(here), mass_kg=a.mass_kg, gauge=a.gauge, surface=a.surface,
        ramp=dict(pwm0=a.pwm0, rate_pwm_per_s=a.rate, pwm_max=a.pwm_max, w_slip_rad_s=a.w_slip, dirs=a.dirs, reps=a.reps),
        method=METHODS[test],
        files=(dict(raw='raw/<surface>_<dir>_<rep>.csv (100 Hz)', trials='trials.csv', summary='summary.csv')
               if test == 'mu' else
               dict(raw='raw/<test><pwm_max>_<dir>_<rep>.csv (100 Hz)',
                    trials='trials.csv (ramp settings, gauge_kgf, vbat_before_v, vbat_after_v, ok per trial)',
                    commands='commands.csv (every command run into this folder)')),
    )
    if test != 'mu':
        meta['ramp_note'] = 'settings of the LAST command; each trial keeps its own in trials.csv'
        meta['vbat_min_v'] = getattr(a, 'vbat_min', VBAT_MIN)
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write('# MU_FLOOR run (written by tools/mu_floor_runner.py; every value is JSON, so the file is valid YAML)\n')
        for k, v in meta.items():
            f.write(f'{k}: {json.dumps(v, ensure_ascii=False)}\n')
    if test == 'mu':
        return
    cpath = os.path.join(a.out, 'commands.csv')
    new = not os.path.isfile(cpath)
    with open(cpath, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['time', 'test', 'argv', 'script_commit', 'script_dirty'])
        st = meta['script']
        w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), test, ' '.join(getattr(a, 'argv', []) or []),
                    st.get('commit', ''), st.get('dirty', '')])


def check_firmware(dev, expect=''):
    """Refuses a board that is not env mu_floor protocol v2, or not the commit given by --expect-esp32."""
    p = getattr(dev, 'pong', None) or {}
    if 'fw' not in p or 'last_id' not in p:
        return f'PONG without fw/last_id ({getattr(dev, "pong_line", "")}): flash env mu_floor protocol v2'
    if p.get('env') != FW_ENV:
        return f'board runs env {p.get("env")!r}, not {FW_ENV!r}'
    if expect and not p['fw'].startswith(expect[:12]) and not expect.startswith(p['fw']):
        return f'board fw {p["fw"]} is not --expect-esp32 {expect}'
    if p.get('imu') != 'ok':
        return f'{dev.pong_line}: the IMU did not start; the firmware refuses RAMP (check the BNO085, power cycle)'
    return ''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--udp', help="'auto' (broadcast PING) or the ESP32 IP")
    ap.add_argument('--udp-port', type=int, default=UDP_PORT)
    ap.add_argument('--bcast', default='auto',
                    help="broadcast address for --udp auto; 'auto' = 255.255.255.255 and a.b.c.255 of every "
                         'IPv4 address of this PC')
    ap.add_argument('--port', help='serial port (bench only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--expect-esp32', default='', help='commit the board must report in PONG fw=')
    ap.add_argument('--mass-kg', type=float, required=True, help='car mass on a scale, kg')
    ap.add_argument('--gauge', default='', help='force gauge: model, range, resolution')
    ap.add_argument('--surface', default='floor', help='floor, or mica for X3 (slippery sheet under the wheels)')
    ap.add_argument('--dirs', nargs='+', default=None, choices=DIRS, help='default XP YP for mu, XP for hold and step')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--pwm0', type=int, default=100)
    ap.add_argument('--rate', type=int, default=10, help='PWM per second')
    ap.add_argument('--pwm-max', type=int, default=1000)
    ap.add_argument('--w-slip', type=float, default=3.0, help='rad/s, automatic stop (0 = off)')
    ap.add_argument('--test', choices=TESTS, default='mu', help='mu (default), hold or step (see the top of the file)')
    ap.add_argument('--vbat-min', type=float, default=VBAT_MIN, help='V, battery rest voltage for hold and step')
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    if a.dirs is None:
        a.dirs = ['XP', 'YP'] if a.test == 'mu' else ['XP']
    if not a.dry_run and bool(a.udp) == bool(a.port):
        sys.exit('give exactly one of --udp auto|IP and --port COMx (or --dry-run)')
    if a.test == 'step' and (a.pwm0 != a.pwm_max or a.w_slip != 0):
        sys.exit('--test step needs --pwm0 equal to --pwm-max (a step) and --w-slip 0 (the wheels turn by design)')
    if a.test != 'mu' and a.surface != 'floor':
        sys.exit('--test hold/step: the trial name comes from --test and --pwm-max; keep --surface floor')
    here = os.path.dirname(os.path.abspath(__file__))
    if a.dry_run:
        dev = FakeDevice(mass_kg=a.mass_kg)
    elif a.udp:
        dev = UdpLink(a.udp, a.udp_port, a.bcast)
    else:
        dev = SerialLink(a.port)
    bad = '' if a.dry_run else check_firmware(dev, a.expect_esp32)
    if bad:
        sys.exit(bad)
    p = dev.pong
    print(f'ESP32 {dev.describe()}: fw={p.get("fw")} dirty={p.get("dirty")} env={p.get("env")} ip={p.get("ip")}')
    if p.get('dirty') not in ('0', None):
        print('  CANH BAO: firmware dirty=1, khong dung lam so chinh')
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
