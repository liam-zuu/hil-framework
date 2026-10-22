"""
imu_capture.py: record raw BNO085 data from the ESP32 test mode TEST_IMU_LOG
(AGV_MECANUM envs imu_log_20 / imu_log_100) and compute the IMU noise figures for the plant model.

Usage
  python tools/imu_capture.py --port COM4 --rate 20  --out D:\\CLAUDE_THESIS\\05_DATA\\results\\IMU_NOISE\\<date>_run01_20hz
  python tools/imu_capture.py --port COM4 --rate 100 --out D:\\CLAUDE_THESIS\\05_DATA\\results\\IMU_NOISE\\<date>_run02_100hz
  python tools/imu_capture.py --dry-run --rate 20 --out <dir>      # simulated sensor, no hardware

--rate must match the flashed env (imu_log_20 = 50 ms, imu_log_100 = 10 ms); the script reads
"IMU READY v2 period_us=..." from the ESP32 and stops if they differ.

Phases (the script tells you what to do and waits for Enter):
  static   300 s  AGV on the floor, motors off, nobody touches it or the table/floor nearby
  yaw90     20 s  start still, turn the AGV by hand about 90 deg to the left, hold still

Output in --out (refuses a folder that already holds data):
  raw_<phase>.csv   kind, t_us, n, v0, v1, v2, qdrop  (one line per report, see below)
  summary.csv       per phase and channel: samples, rate, gaps, mean, std, drift, smallest step
  meta.yaml         firmware commit and git state (read from AGV_FIRMWARE), rate, phases, notes

Firmware line format v2 (test_imu_log.c): one line PER REPORT, no change filter.
  kind A = accelerometer x,y,z [m/s^2], G = calibrated gyro x,y,z [rad/s], R = game RV roll,pitch,yaw [deg]
  t_us = ESP32 receive time, n = host count of that report, qdrop = lines lost in the ESP32 print queue.
Statistics use every sample. (v1, 21/10, kept only lines where a value changed: that biased std,
because repeated quantised values were dropped. No v1 data was recorded.)
"""
import argparse
import csv
import math
import os
import random
import subprocess
import sys
import time

KINDS = {'A': ['ax', 'ay', 'az'], 'G': ['gx', 'gy', 'gz'], 'R': ['roll', 'pitch', 'yaw']}
PHASES = [
    ('static', 300, 'Đặt xe nằm yên trên sàn, tắt motor, không chạm vào xe hay bàn/sàn gần đó.'),
    ('yaw90', 20, 'Giữ yên 3 s, rồi xoay xe bằng tay khoảng 90 độ sang TRÁI, sau đó giữ yên.'),
]
RATES = {20: ('imu_log_20', 50000), 100: ('imu_log_100', 10000)}


def git_state(repo):
    def git(*a):
        out = subprocess.run(['git', '-C', repo] + list(a), capture_output=True, text=True, check=True).stdout
        return out.rstrip('\n') if a[0] == 'status' else out.strip()
    try:
        changed = [l[3:] for l in git('status', '--porcelain').splitlines() if l[3:] and not l[3:].startswith('.vscode/')]
        return dict(commit=git('rev-parse', '--short', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                    dirty=bool(changed), modified=changed)
    except (OSError, subprocess.CalledProcessError) as e:
        return dict(commit='UNKNOWN', branch='UNKNOWN', dirty=True, modified=[f'git error: {e}'])


def parse_line(line):
    """'G,123,4,0.1,0.2,0.3,0' -> ['G', 123, 4, 0.1, 0.2, 0.3, 0] or None."""
    parts = line.split(',')
    if len(parts) != 7 or parts[0] not in KINDS:
        return None
    try:
        return [parts[0], int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), int(parts[6])]
    except ValueError:
        return None


class SerialSource:
    def __init__(self, port, period_us):
        try:
            import serial
            serial.Serial
        except (ImportError, AttributeError):
            sys.exit('pyserial is missing: python -m pip install pyserial')
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = port, 921600, 0.2
        s.dtr = False; s.rts = False          # see s01_openloop_runner.py: avoid download mode
        s.open()
        s.rts = True; time.sleep(0.15); s.rts = False
        self.s = s
        t_end = time.time() + 8
        while time.time() < t_end:
            line = s.readline().decode(errors='replace').strip()
            if line.startswith('IMU READY'):
                if 'v2' not in line:
                    sys.exit(f'ESP32 prints "{line}": old firmware (v1). Flash env imu_log_20 or imu_log_100.')
                got = int(line.split('period_us=')[1])
                if got != period_us:
                    sys.exit(f'ESP32 reports period_us={got}, --rate expects {period_us}. Flash the matching env.')
                return
        sys.exit(f'No "IMU READY" on {port}. Check: env imu_log_20/100 flashed, right port, no serial monitor open.')

    def read(self, seconds):
        self.s.reset_input_buffer()
        rows, bad, t_end = [], 0, time.time() + seconds
        while time.time() < t_end:
            line = self.s.readline().decode(errors='replace').strip()
            if not line:
                continue
            r = parse_line(line)
            if r is None:
                bad += 1
            else:
                rows.append(r)
        return rows, bad


class FakeSource:
    """Invented sensor for testing the script; NOT a measurement."""
    def __init__(self, period_us, seed=0):
        self.r = random.Random(seed)
        self.t = 0
        self.p = period_us
        self.n = {k: 0 for k in KINDS}

    def q(self, x, step):
        return round(x / step) * step     # chip output is quantised (Q8 accel, Q9 gyro)

    def read(self, seconds):
        rows, yaw = [], 0.0
        dt = self.p * 1e-6
        steps = int(seconds / dt)
        for k in range(steps):
            self.t += self.p
            turning = seconds < 60 and 3 < k * dt < 6
            gz = (math.pi / 2 / 3 if turning else 0.0) + self.r.gauss(0, 1e-3)
            yaw += math.degrees(gz * dt)
            for kind, v in (('A', [self.q(self.r.gauss(0, 0.012), 1 / 256), self.q(self.r.gauss(0, 0.012), 1 / 256),
                                   self.q(9.81 + self.r.gauss(0, 0.0125), 1 / 256)]),
                            ('G', [self.q(self.r.gauss(0, 1e-3), 1 / 512), self.q(self.r.gauss(0, 1e-3), 1 / 512),
                                   self.q(gz, 1 / 512)]),
                            ('R', [0.1, -0.2, (yaw + 180) % 360 - 180])):
                self.n[kind] += 1
                rows.append([kind, self.t + {'A': 0, 'G': 300, 'R': 600}[kind], self.n[kind]] + v + [0])
        return rows, 0


def channel_stats(rows, kind, axis, period_us):
    samp = [(r[1], r[3 + axis]) for r in rows if r[0] == kind]
    if len(samp) < 3:
        return dict(n_samples=len(samp))
    t = [(s[0] - samp[0][0]) * 1e-6 for s in samp]
    v = [s[1] for s in samp]
    n = len(v); mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
    tm = sum(t) / n
    slope = sum((a - tm) * (b - mean) for a, b in zip(t, v)) / max(sum((a - tm) ** 2 for a in t), 1e-12)
    gaps = sum(1 for a, b in zip(samp, samp[1:]) if b[0] - a[0] > 1.5 * period_us)
    steps = [abs(b - a) for a, b in zip(v, v[1:]) if abs(b - a) > 0]
    return dict(n_samples=n, rate_hz=(n - 1) / max(t[-1], 1e-9), gaps=gaps, mean=mean, std=std,
                drift_per_s=slope, min_step=min(steps) if steps else 0.0,
                frac_repeat=1 - len(steps) / max(n - 1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port')
    ap.add_argument('--rate', type=int, choices=sorted(RATES), required=True, help='report rate of the flashed env [Hz]')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--esp32-repo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'AGV_FIRMWARE'))
    ap.add_argument('--operator', default='Liam')
    a = ap.parse_args()
    if not a.dry_run and not a.port:
        ap.error('--port is required unless --dry-run')
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data. Raw data must never be overwritten: use a new folder.')
    env, period_us = RATES[a.rate]

    esp = git_state(a.esp32_repo)
    print(f"ESP32 firmware: {esp['commit']} ({esp['branch']}), dirty={esp['dirty']} {esp['modified'] or ''}")
    src = FakeSource(period_us) if a.dry_run else SerialSource(a.port, period_us)
    os.makedirs(a.out, exist_ok=True)

    summary, checks = [], []
    for name, dur, what in PHASES:
        print(f'\n[{name}] {what}\n  Thời gian: {dur} s.')
        if not a.dry_run:
            input('  Sẵn sàng thì nhấn Enter...')
        rows, bad = src.read(dur)
        with open(os.path.join(a.out, f'raw_{name}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['kind', 't_us', 'n', 'v0', 'v1', 'v2', 'qdrop'])
            w.writerows(rows)
        qdrop = (rows[-1][6] - rows[0][6]) if rows else 0
        print(f'  {len(rows)} dòng đã ghi, {bad} dòng hỏng, {qdrop} dòng mất trong hàng đợi ESP32.')
        checks.append(dict(phase=name, lines=len(rows), bad_lines=bad, esp32_queue_drops=qdrop))
        for kind, names in KINDS.items():
            for axis, ch in enumerate(names):
                s = channel_stats(rows, kind, axis, period_us)
                summary.append(dict(phase=name, channel=ch, **{k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in s.items()}))
        if name == 'yaw90':
            g = [r for r in rows if r[0] == 'G']
            y = [r for r in rows if r[0] == 'R']
            if len(g) > 2 and len(y) > 2:
                gz_int = sum(r[5] * (r2[1] - r[1]) * 1e-6 for r, r2 in zip(g, g[1:]))
                dyaw = (y[-1][5] - y[0][5] + 180) % 360 - 180
                summary.append(dict(phase=name, channel='check', mean=f'integral gz = {math.degrees(gz_int):.2f} deg',
                                    std=f'delta yaw = {dyaw:.2f} deg'))
                print(f'  Tích phân gz = {math.degrees(gz_int):.1f}°, yaw đổi = {dyaw:.1f}° (hai số nên gần bằng nhau)')

    keys = ['phase', 'channel', 'n_samples', 'rate_hz', 'gaps', 'mean', 'std', 'drift_per_s', 'min_step', 'frac_repeat']
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for s in summary:
            w.writerow({k: s.get(k, '') for k in keys})
    for s in summary:
        if s['phase'] == 'static' and s['channel'] in ('ax', 'ay', 'az', 'gx', 'gy', 'gz'):
            print(f"  static {s['channel']}: std {s.get('std')}  mean {s.get('mean')}  rate {s.get('rate_hz')} Hz  gaps {s.get('gaps')}")

    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: null                 # đo nhiễu IMU theo yêu cầu trực tiếp của Liam (21/10/2026); 2 tần số theo K1 (Liam 22/10)
run: 1
date: {time.strftime('%Y-%m-%d')}
operator: {a.operator}
firmware:
  rpi5: null
  h7: null
  esp32: {{repo: AGV_MECANUM, commit: {esp['commit']}, branch: {esp['branch']}, pio_env: {env}, build_flags: "-DTEST_MODE -DTEST_IMU_LOG -DIMU_REPORT_PERIOD_US={period_us}UL", dirty: {str(esp['dirty']).lower()}, modified: {esp['modified']}}}
setup: real_agv_imu
dry_run: {str(a.dry_run).lower()}       # true = simulated sensor, NOT a measurement
config:
  sensor: BNO085 (SPI), reports accelerometer, calibrated gyro, game rotation vector, each every {period_us} us ({a.rate} Hz)
  phases: {[(n, d) for n, d, _ in PHASES]}
checks: {checks}
files:
  - raw_<phase>.csv: "kind (A accel m/s^2, G gyro rad/s, R roll pitch yaw deg), t_us (ESP32 receive time), n (host count of that report), v0 v1 v2, qdrop (cumulative ESP32 print-queue drops); one line per report"
  - summary.csv: "per phase and channel over ALL samples: n_samples, rate_hz, gaps (intervals > 1.5 periods), mean, std, drift_per_s (linear trend), min_step (smallest nonzero change = output resolution), frac_repeat (share of samples equal to the previous one)"
notes: ""
""")
    print(f'\nXong. Kết quả ở {a.out}')


if __name__ == '__main__':
    main()
