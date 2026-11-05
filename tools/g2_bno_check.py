"""
g2_bno_check.py: G2 check of the H7 BNO085 emulator with the real ESP32
(spec docs/2026-10-22_Spec_G0_Gia_lap_BNO085.md S8; plan Liam approved 27/10).

The standalone ESP32 runs AGV_MECANUM env imu_log_20 or imu_log_100 (unchanged
IMU driver, one CSV line per report, see tools/imu_capture.py). The RPi5 sends
constant IMU values:  sudo python3 run_hil.py --imu-static GZ,AX,AY,AZ,YAW --no-traj
For every boot the script resets the ESP32 (RTS pulse, as imu_capture.py), waits
for "IMU READY", records the reports for --seconds and checks:
  boot     "IMU READY" within --ready-timeout and reports from all 3 sensors
  period   mean report period per sensor within 2 % of the env period
  values   every report within one chip quantisation step (+ one SPI frame LSB)
           of the static values: A = (ax, ay, az), G = (0, 0, gz), R = (0, 0, yaw deg)
  loss     (added 28/10, Codex #16 on spec S06 v3.1) no data lost on the host side:
           qdrop = 0 at the end (ESP32 print queue never full since boot), no line that
           fails to parse after IMU READY, host count n of each sensor starts at 1 and
           steps by exactly 1. Before 28/10 PASS ignored these, so a boot could pass
           with lost reports. Gaps in TIME (H7 sending late or skipping) are reported as
           *_max_gap_us but not judged: G0 v1 S8 sets no threshold for them.
  window   (added 29/10, spec S06 v3.4 Z18, L7-20, V2-10; needs ESP32 test_imu_log v3,
           AGV branch fix/g2-imu-log) READY carries t_us. Period, values and counts use
           only reports with t_us in [t_READY; t_READY + seconds] on the ESP32 clock.
           Reports before READY are not judged but still count for the n = 1, 2, ...
           check. After the window the script sends STOP; the ESP32 then writes
           END,t_us,accel_n,gyro_n,rv_n,qdrop,via through the same queue as the reports.
           PASS also needs: the END line within --end-timeout, END qdrop = 0, each kind
           received as 1 .. its END counter with no gap; with --expect values, each kind
           has at least seconds / period * (1 - 2 %) reports in the window and the last
           one is at most 2 periods before the window end.

Usage
  python tools/g2_bno_check.py --port COM4 --rate 20 --boots 10 --seconds 60 \\
      --expect 0.5,0.2,-0.1,9.81,1.0 --out D:\\CLAUDE_THESIS\\05_DATA\\results\\G2_BNO\\<date>_run01_20hz
  --expect none   boot check only (RPi5 not running: no reports expected)

Output in --out (refuses a folder that already holds data): raw_bootNN.csv,
boots.csv (one line per boot), meta.yaml (commits, settings, verdict).
The pass thresholds are G0 v1 S8 values (proposed, [DE XUAT]); the H7 "BNO:" lines are
copied by hand (H7 VCP) into notes if needed.
"""
import argparse
import csv
import math
import os
import sys
import time

from imu_capture import git_state, parse_line, RATES

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FW_REPO = os.path.join(os.path.dirname(REPO), 'AGV_FIRMWARE')

# Chip quantisation (SH-2 Q points): accel Q8, calibrated gyro Q9, game RV Q14.
ACC_STEP = 1.0 / 256
GYR_STEP = 1.0 / 512
# Game RV printed as euler degrees with 3 decimals: one Q14 step of the
# quaternion is about 2 * 2^-14 rad in yaw, plus the 0.0005 deg print rounding.
RV_STEP_DEG = math.degrees(2.0 / 16384) + 0.0005
# SPI frame int16 LSB (protocol.py IMU_RANGES)
FRAME_LSB = {'gz': 34.906585 / 32767, 'acc': 39.24 / 32767, 'yaw_deg': math.degrees(math.pi / 32767)}
PERIOD_TOL = 0.02
COUNT_TOL = PERIOD_TOL      # reports per window >= seconds / period * (1 - COUNT_TOL)
TAIL_PERIODS = 2            # last report in the window at most 2 periods before its end
STOP_MARGIN_S = 0.5         # host waits this long past the window before sending STOP
KINDS = ('A', 'G', 'R')


def expected(expect):
    gz, ax, ay, az, yaw = expect
    return {
        'A': ((ax, ay, az), ACC_STEP + FRAME_LSB['acc']),
        'G': ((0.0, 0.0, gz), GYR_STEP + FRAME_LSB['gz']),
        'R': ((0.0, 0.0, math.degrees(yaw)), RV_STEP_DEG + FRAME_LSB['yaw_deg']),
    }


def open_port(port):
    try:
        import serial
    except ImportError:
        sys.exit('pyserial is missing: python -m pip install pyserial')
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 921600, 0.2
    s.dtr = False; s.rts = False          # avoid download mode (see imu_capture.py)
    s.open()
    return s


def parse_ready(line):
    """'IMU READY v3 period_us=50000 t_us=812345' -> (period_us, t_us or None)."""
    def tok(name):
        if f'{name}=' not in line:
            return None
        try:
            return int(line.split(f'{name}=')[1].split()[0])
        except ValueError:
            return None
    return tok('period_us'), tok('t_us')


def parse_end(line):
    """'END,t_us,accel_n,gyro_n,rv_n,qdrop,via' -> dict or None."""
    p = line.split(',')
    if len(p) != 7 or p[0] != 'END' or p[6] not in ('H', 'T'):
        return None
    try:
        t, a, g, r, q = (int(x) for x in p[1:6])
    except ValueError:
        return None
    return {'t_us': t, 'A': a, 'G': g, 'R': r, 'qdrop': q, 'via': p[6]}


def one_boot(s, period_us, ready_timeout, seconds, end_timeout=3.0):
    """Reset the ESP32, wait for IMU READY, record the window, send STOP, wait for END.

    Returns (ready_s or None, t_ready_us, rows, bad, end or None)."""
    s.reset_input_buffer()
    t0 = time.time()
    s.rts = True; time.sleep(0.15); s.rts = False
    ready, t_ready_us = None, None
    while time.time() - t0 < ready_timeout:
        line = s.readline().decode(errors='replace').strip()
        if line.startswith('IMU INIT FAIL'):
            print(f'ESP32: "{line}"')
            break
        if line.startswith('IMU READY'):
            # ctrl-imu prints "... period_us=<n> init_ms=<m>" (Codex #13): take the first token
            got, t_ready_us = parse_ready(line)
            if got != period_us:
                sys.exit(f'ESP32 reports "{line}", --rate expects period_us={period_us}. Flash the matching env.')
            if t_ready_us is None:
                sys.exit(f'ESP32 reports "{line}" without t_us: old test_imu_log. Flash env imu_log_20/100 '
                         'from AGV_MECANUM branch fix/g2-imu-log (READY v3 + END line).')
            ready = time.time() - t0
            break
    if ready is None:
        return None, None, [], 0, None
    rows, bad, end = [], 0, None
    t_stop = time.time() + seconds + STOP_MARGIN_S
    stop_sent = False
    while True:
        now = time.time()
        if not stop_sent and now >= t_stop:
            s.write(b'STOP\n')
            stop_sent = True
        if stop_sent and now >= t_stop + end_timeout:
            break
        line = s.readline().decode(errors='replace').strip()
        if not line:
            continue
        if line.startswith('END'):
            end = parse_end(line)
            if end is None:
                bad += 1
                continue
            break
        r = parse_line(line)
        if r is None:
            bad += 1
        else:
            rows.append(r)
    return ready, t_ready_us, rows, bad, end


def check_boot(rows, period_us, exp, t_ready_us=None, seconds=None, end=None):
    """Per-boot figures; exp None = boot check only.

    Count continuity (first n, gaps, last n vs END) uses every row. Period, values,
    window count and tail gap use the window [t_ready_us; t_ready_us + seconds]
    when t_ready_us is given, else every row (old firmware, tests)."""
    out = {}
    if t_ready_us is not None and seconds is not None:
        w0, w1 = t_ready_us, t_ready_us + int(round(seconds * 1e6))
        win = [r for r in rows if w0 <= r[1] <= w1]
        out['window_us'] = f'{w0}..{w1}'
    else:
        w1, win = None, rows
        out['window_us'] = ''
    for kind in KINDS:
        rr = [r for r in rows if r[0] == kind]
        ww = [r for r in win if r[0] == kind]
        out[f'{kind}_n'] = len(ww)
        out[f'{kind}_n_all'] = len(rr)
        if len(ww) >= 2:
            span = ww[-1][1] - ww[0][1]
            mean_p = span / (len(ww) - 1)
            gaps = [b[1] - a[1] for a, b in zip(ww, ww[1:])]
            out[f'{kind}_period_dev_pct'] = round(100.0 * (mean_p - period_us) / period_us, 3)
            out[f'{kind}_max_gap_us'] = max(gaps)
        else:
            out[f'{kind}_period_dev_pct'] = ''
            out[f'{kind}_max_gap_us'] = ''
        out[f'{kind}_count_gaps'] = sum(1 for a, b in zip(rr, rr[1:]) if b[2] - a[2] != 1) if rr else ''
        out[f'{kind}_first_n'] = rr[0][2] if rr else ''
        out[f'{kind}_last_n'] = rr[-1][2] if rr else ''
        out[f'{kind}_end_n'] = end[kind] if end else ''
        if w1 is not None:
            out[f'{kind}_n_min'] = math.ceil(seconds * 1e6 / period_us * (1 - COUNT_TOL) - 1e-9)
            out[f'{kind}_tail_gap_us'] = (w1 - ww[-1][1]) if ww else ''
        else:
            out[f'{kind}_n_min'] = ''
            out[f'{kind}_tail_gap_us'] = ''
        if exp is not None and ww:
            want, tol = exp[kind]
            err = max(abs(r[3 + i] - want[i]) for r in ww for i in range(3))
            out[f'{kind}_max_err'] = round(err, 6)
            out[f'{kind}_tol'] = round(tol, 6)
        else:
            out[f'{kind}_max_err'] = ''
            out[f'{kind}_tol'] = ''
    out['qdrop'] = max(r[6] for r in rows) if rows else ''
    out['end_seen'] = end is not None
    out['end_qdrop'] = end['qdrop'] if end else ''
    out['end_via'] = end['via'] if end else ''
    return out


def fail_reasons(b, exp, period_us=None):
    """Why a boot fails; empty list = PASS.

    Boots checked without END support (end_seen missing, old tests) skip the END rules."""
    if b['ready_s'] == '':
        return ['no IMU READY']
    why = []
    if b['bad_lines']:
        why.append(f"{b['bad_lines']} bad lines")
    if b['qdrop'] not in ('', 0):
        why.append(f"qdrop={b['qdrop']}")
    end_rules = 'end_seen' in b and b.get('window_us', '') != ''
    if end_rules and not b['end_seen']:
        why.append('no END line')
    if end_rules and b['end_seen'] and b['end_qdrop'] != 0:
        why.append(f"END qdrop={b['end_qdrop']}")
    for k in KINDS:
        n_all = b.get(f'{k}_n_all', b[f'{k}_n'])
        if n_all and b[f'{k}_first_n'] != 1:
            why.append(f"{k} first n={b[f'{k}_first_n']}")
        if b[f'{k}_count_gaps'] not in ('', 0):
            why.append(f"{k} count gaps={b[f'{k}_count_gaps']}")
        if end_rules and b['end_seen']:
            last = b[f'{k}_last_n'] if n_all else 0
            if last != b[f'{k}_end_n']:
                why.append(f"{k} last n={last} END n={b[f'{k}_end_n']}")
    if exp is None:
        return why
    for k in KINDS:
        if b[f'{k}_n'] < 2:
            why.append(f'{k} n<2')
            continue
        if b.get(f'{k}_n_min', '') != '' and b[f'{k}_n'] < b[f'{k}_n_min']:
            why.append(f"{k} n={b[f'{k}_n']}<{b[f'{k}_n_min']}")
        if (period_us and b.get(f'{k}_tail_gap_us', '') != ''
                and b[f'{k}_tail_gap_us'] > TAIL_PERIODS * period_us):
            why.append(f"{k} tail gap {b[f'{k}_tail_gap_us']} us")
        if abs(b[f'{k}_period_dev_pct']) > 100 * PERIOD_TOL:
            why.append(f"{k} period {b[f'{k}_period_dev_pct']}%")
        if b[f'{k}_max_err'] == '' or b[f'{k}_max_err'] > b[f'{k}_tol']:
            why.append(f'{k} value')
    return why


def boot_pass(b, exp, period_us=None):
    return not fail_reasons(b, exp, period_us)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', required=True)
    ap.add_argument('--rate', type=int, choices=sorted(RATES), required=True)
    ap.add_argument('--boots', type=int, default=10)
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--ready-timeout', type=float, default=10.0)
    ap.add_argument('--end-timeout', type=float, default=3.0, help='wait for the END line after STOP [s]')
    ap.add_argument('--expect', required=True, help='gz,ax,ay,az,yaw as sent with --imu-static, or none')
    ap.add_argument('--out', required=True)
    ap.add_argument('--notes', default='')
    a = ap.parse_args()

    exp_vals = None if a.expect.lower() == 'none' else tuple(float(v) for v in a.expect.split(','))
    if exp_vals is not None and len(exp_vals) != 5:
        sys.exit('--expect needs gz,ax,ay,az,yaw or none')
    exp = expected(exp_vals) if exp_vals else None
    env, period_us = RATES[a.rate]
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data; use a new runNN folder')
    os.makedirs(a.out, exist_ok=True)

    s = open_port(a.port)
    boots = []
    for k in range(1, a.boots + 1):
        ready, t_ready_us, rows, bad, end = one_boot(s, period_us, a.ready_timeout, a.seconds, a.end_timeout)
        with open(os.path.join(a.out, f'raw_boot{k:02d}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['kind', 't_us', 'n', 'v0', 'v1', 'v2', 'qdrop'])
            w.writerows(rows)
        b = {'boot': k, 'ready_s': '' if ready is None else round(ready, 3),
             't_ready_us': '' if t_ready_us is None else t_ready_us, 'bad_lines': bad}
        b.update(check_boot(rows, period_us, exp, t_ready_us, a.seconds, end))
        why = fail_reasons(b, exp, period_us)
        b['pass'] = not why
        b['fail_reasons'] = '; '.join(why)
        boots.append(b)
        print(f"boot {k:2d}: ready={b['ready_s'] or 'NO'}  n A/G/R={b['A_n']}/{b['G_n']}/{b['R_n']}  "
              f"dev% {b['A_period_dev_pct']}/{b['G_period_dev_pct']}/{b['R_period_dev_pct']}  "
              f"err {b['A_max_err']}/{b['G_max_err']}/{b['R_max_err']}  "
              f"bad={bad} qdrop={b['qdrop']} END={'yes/' + b['end_via'] if b['end_seen'] else 'NO'}  -> {'PASS' if b['pass'] else 'FAIL: ' + b['fail_reasons']}",
              flush=True)
    s.close()

    with open(os.path.join(a.out, 'boots.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(boots[0].keys()))
        w.writeheader()
        w.writerows(boots)

    n_pass = sum(b['pass'] for b in boots)
    verdict = 'PASS' if n_pass == len(boots) else 'FAIL'
    hil, fw = git_state(REPO), git_state(FW_REPO)
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write('# G2 BNO085 emulator check (tools/g2_bno_check.py)\n')
        f.write(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"hil_framework_commit: {hil['commit']}  # branch {hil['branch']}, dirty={hil['dirty']}\n")
        f.write(f"esp32_firmware_commit: {fw['commit']}  # branch {fw['branch']}, dirty={fw['dirty']}\n")
        f.write(f'esp32_env: {env}\nperiod_us: {period_us}\nport: {a.port}\n')
        f.write(f'boots: {a.boots}\nseconds_per_boot: {a.seconds}\n')
        f.write(f"expect_gz_ax_ay_az_yaw: {'none' if exp_vals is None else list(exp_vals)}\n")
        f.write(f'period_tolerance_pct: {100 * PERIOD_TOL}  # G0 v1 S8 [DE XUAT]\n')
        f.write('value_tolerance: one chip quantisation step + one SPI frame LSB per channel\n')
        f.write('loss_rule: bad_lines = 0, qdrop = 0, first n = 1, no count gaps  # 28/10, Codex #16\n')
        f.write('window_rule: window [t_READY; t_READY + seconds] on the ESP32 clock; END line within '
                f'end_timeout ({a.end_timeout} s); END qdrop = 0; each kind received 1 .. END counter; '
                f'with values: window count >= seconds/period*(1-{COUNT_TOL}), tail gap <= {TAIL_PERIODS} periods'
                '  # 29/10, spec S06 v3.4 Z18, L7-20, V2-10\n')
        f.write('esp32_firmware_note: commit read from the AGV_FIRMWARE main folder; flash the board from there '
                '(test_imu_log v3 = branch fix/g2-imu-log or later)\n')
        f.write(f'boots_pass: {n_pass}\nverdict: {verdict}\n')
        f.write('h7_firmware: FILL IN (commit flashed on the H7)\n')
        f.write(f'notes: "{a.notes}"\n')
    print(f'{n_pass}/{len(boots)} boots pass -> {verdict}  ({a.out})')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
