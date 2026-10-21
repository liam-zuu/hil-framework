"""
imu_capture.py: record raw BNO085 data from the ESP32 test mode TEST_IMU_LOG
(AGV_MECANUM env imu_log) and compute the IMU noise figures for the plant model.

Usage
  python tools/imu_capture.py --port COM4 --out D:\\CLAUDE_THESIS\\05_DATA\\results\\IMU_NOISE\\2026-10-21_run01
  python tools/imu_capture.py --dry-run --out <dir>          # simulated sensor, no hardware

Phases (the script tells you what to do and waits for Enter):
  static   300 s  AGV on the floor, motors off, nobody touches it or the table/floor nearby
  yaw90     20 s  start still, turn the AGV by hand about 90 deg to the left, hold still

Output in --out (refuses a folder that already holds data):
  raw_<phase>.csv   t_us, ax, ay, az [m/s^2], gx, gy, gz [rad/s], roll, pitch, yaw [deg]
  summary.csv       per phase and channel: update rate, mean, std of updates, bias drift
  meta.yaml         firmware commit and git state (read from AGV_FIRMWARE), phases, notes

The ESP32 prints a line whenever any field changes; reports arrive separately
at 20 Hz, so each channel's statistics use only the lines where it changed.
"""
import argparse
import csv
import math
import os
import random
import subprocess
import sys
import time

CHANNELS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
PHASES = [
    ('static', 300, 'Đặt xe nằm yên trên sàn, tắt motor, không chạm vào xe hay bàn/sàn gần đó.'),
    ('yaw90', 20, 'Giữ yên 3 s, rồi xoay xe bằng tay khoảng 90 độ sang TRÁI, sau đó giữ yên.'),
]


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


class SerialSource:
    def __init__(self, port):
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
            if b'IMU READY' in s.readline():
                return
        sys.exit(f'No "IMU READY" on {port}. Check: env imu_log flashed, right port, no serial monitor open.')

    def read(self, seconds):
        self.s.reset_input_buffer()
        rows, t_end = [], time.time() + seconds
        while time.time() < t_end:
            line = self.s.readline().decode(errors='replace').strip()
            parts = line.split(',')
            if len(parts) == 10:
                try:
                    rows.append([int(parts[0])] + [float(x) for x in parts[1:]])
                except ValueError:
                    pass
        return rows


class FakeSource:
    """Invented sensor for testing the script; NOT a measurement."""
    def __init__(self, seed=0):
        self.r = random.Random(seed)
        self.t = 0

    def read(self, seconds):
        rows, yaw = [], 0.0
        for k in range(int(seconds * 20)):
            self.t += 50000
            turning = seconds < 60 and 3 < k / 20 < 6
            gz = (math.pi / 2 / 3 if turning else 0.0) + self.r.gauss(0, 3e-4)
            yaw += math.degrees(gz * 0.05)
            rows.append([self.t, self.r.gauss(0, 0.012), self.r.gauss(0, 0.012), 9.81 + self.r.gauss(0, 0.0125),
                         self.r.gauss(0, 3e-4), self.r.gauss(0, 3e-4), gz, 0.1, -0.2, yaw])
        return rows


def channel_stats(rows, ch):
    i = 1 + CHANNELS.index(ch)
    upd = [(r[0], r[i]) for k, r in enumerate(rows) if k == 0 or r[i] != rows[k - 1][i]]
    if len(upd) < 3:
        return dict(n_updates=len(upd))
    t = [(u[0] - upd[0][0]) * 1e-6 for u in upd]
    v = [u[1] for u in upd]
    n = len(v); mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
    tm = sum(t) / n
    slope = sum((a - tm) * (b - mean) for a, b in zip(t, v)) / max(sum((a - tm) ** 2 for a in t), 1e-12)
    return dict(n_updates=n, rate_hz=(n - 1) / max(t[-1], 1e-9), mean=mean, std=std, drift_per_s=slope)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--esp32-repo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'AGV_FIRMWARE'))
    ap.add_argument('--operator', default='Liam')
    a = ap.parse_args()
    if not a.dry_run and not a.port:
        ap.error('--port is required unless --dry-run')
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data. Raw data must never be overwritten: use a new folder.')
    os.makedirs(a.out, exist_ok=True)

    esp = git_state(a.esp32_repo)
    print(f"ESP32 firmware: {esp['commit']} ({esp['branch']}), dirty={esp['dirty']} {esp['modified'] or ''}")
    src = FakeSource() if a.dry_run else SerialSource(a.port)

    summary = []
    for name, dur, what in PHASES:
        print(f'\n[{name}] {what}\n  Thời gian: {dur} s.')
        if not a.dry_run:
            input('  Sẵn sàng thì nhấn Enter...')
        rows = src.read(dur)
        with open(os.path.join(a.out, f'raw_{name}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_us'] + CHANNELS)
            w.writerows(rows)
        print(f'  {len(rows)} dòng đã ghi.')
        for ch in CHANNELS:
            s = channel_stats(rows, ch)
            summary.append(dict(phase=name, channel=ch, **{k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in s.items()}))
        if name == 'yaw90' and len(rows) > 2:
            gz_int = sum(r[6] * (r2[0] - r[0]) * 1e-6 for r, r2 in zip(rows, rows[1:]))
            dyaw = rows[-1][9] - rows[0][9]
            dyaw = (dyaw + 180) % 360 - 180
            summary.append(dict(phase=name, channel='check', n_updates='', rate_hz='',
                                mean=f'integral gz = {math.degrees(gz_int):.2f} deg', std=f'delta yaw = {dyaw:.2f} deg', drift_per_s=''))
            print(f'  Tích phân gz = {math.degrees(gz_int):.1f}°, yaw đổi = {dyaw:.1f}° (hai số nên gần bằng nhau)')

    keys = ['phase', 'channel', 'n_updates', 'rate_hz', 'mean', 'std', 'drift_per_s']
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for s in summary:
            w.writerow({k: s.get(k, '') for k in keys})
    for s in summary:
        if s['phase'] == 'static' and s['channel'] in ('ax', 'ay', 'az', 'gx', 'gy', 'gz'):
            print(f"  static {s['channel']}: std {s.get('std')}  mean {s.get('mean')}  rate {s.get('rate_hz')} Hz")

    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: null                 # đo nhiễu IMU theo yêu cầu trực tiếp của Liam (21/10/2026)
run: 1
date: {time.strftime('%Y-%m-%d')}
operator: {a.operator}
firmware:
  rpi5: null
  h7: null
  esp32: {{repo: AGV_MECANUM, commit: {esp['commit']}, branch: {esp['branch']}, pio_env: imu_log, build_flags: "-DTEST_MODE -DTEST_IMU_LOG", dirty: {str(esp['dirty']).lower()}, modified: {esp['modified']}}}
setup: real_agv_imu
dry_run: {str(a.dry_run).lower()}       # true = simulated sensor, NOT a measurement
config:
  sensor: BNO085 (SPI), reports accelerometer, calibrated gyro, game rotation vector at 50 ms (imu.cpp)
  phases: {[(n, d) for n, d, _ in PHASES]}
files:
  - raw_<phase>.csv: "t_us (ESP32 clock), ax ay az [m/s^2], gx gy gz [rad/s], roll pitch yaw [deg]; one line per change of any field"
  - summary.csv: "per phase and channel: updates, rate_hz, mean, std (of updates), drift_per_s (linear trend of the updates)"
notes: ""
""")
    print(f'\nXong. Kết quả ở {a.out}')


if __name__ == '__main__':
    main()
