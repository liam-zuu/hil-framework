"""
make_imu_noise_replay.py: build the fixed replay file of the measured BNO085 noise
(rpi5/data/imu_noise_replay_<rate>hz.csv) from one raw static log of tools/imu_capture.py.

Liam 10/11/2026 (CODE, STATUS_CODE CG12, L72): "Phát lại nhiễu đo" replaces the Gaussian
model (Codex round 1, IMN-3/-5). The replay file keeps, per gyro report of the static phase
(vehicle at rest, motors off, 300 s), the paired gyro z and game-RV yaw outputs as integer
numbers of output steps:

  k_gz  = gz / 2^-9 rad/s                     (SH-2 Q9 output of the chip; exact in the log)
  k_yaw = (yaw - yaw_mode) / 2^-13 rad         (one Q14 step of the quaternion k around yaw 0;
                                               yaw_mode = most common yaw level of the log)

Pairing: G and R rows with the same report count n (one gyro and one game-RV report per
period). Rows of other kinds (accelerometer) are not used (IMN-11, DEFERRED: the accelerometer
is only rounded, see imu_model.py).

Why steps and not the raw residual: the replay is added to the true value and then rounded
again (gyro 2^-9, yaw through the Q14 quaternion, imu_model.firmware_view / the H7). The log is
already on the chip's output grid, so adding the raw residual and rounding again would add a
second rounding error (about +24 % std of the yaw at 100 Hz). With whole steps, rounding
(truth + k*step) = rounding(truth) + k*step at rest, which gives back the measured levels.
The yaw log is printed with 3 decimals of a degree and its levels move by about 0.001 deg
with the roll/pitch quantisation (not modelled: roll = pitch = 0 in HIL and fwsim); the
largest |yaw - yaw_mode - k_yaw*step| is written into the header (yaw_snap_max_deg).

  python tools/make_imu_noise_replay.py --src <.../IMU_NOISE/2026-11-10_run01_20hz/raw_static.csv> \
      --rate 20 --src-label 05_DATA/results/IMU_NOISE/2026-11-10_run01_20hz/raw_static.csv
"""
import argparse
import csv
import hashlib
import math
import os
from collections import Counter

GYRO_STEP = 2.0 ** -9            # rad/s
YAW_STEP = 2.0 ** -13            # rad: d(yaw) for one Q14 step of qk at yaw 0 (yaw ~ 2 qk)
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_pairs(path):
    g, r, bad = {}, {}, 0
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                kind, n = row['kind'], int(row['n'])
                v2 = float(row['v2'])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if kind == 'G':
                g[n] = v2
            elif kind == 'R':
                r[n] = v2
    ns = sorted(set(g) & set(r))
    return ns, g, r, bad, len(g), len(r)


def build(src, rate_hz):
    ns, g, r, bad, n_g, n_r = load_pairs(src)
    if len(ns) < 100:
        raise SystemExit(f'too few paired reports ({len(ns)})')
    k_gz = []
    for n in ns:
        k = g[n] / GYRO_STEP
        if abs(k - round(k)) > 1e-3:
            raise SystemExit(f'gyro report n={n} not on the 2^-9 grid: {g[n]}')
        k_gz.append(int(round(k)))
    mode_deg = Counter(r[n] for n in ns).most_common(1)[0][0]
    step_deg = math.degrees(YAW_STEP)
    k_yaw, snap = [], 0.0
    for n in ns:
        x = (r[n] - mode_deg) / step_deg
        k_yaw.append(int(round(x)))
        snap = max(snap, abs(x - round(x)) * step_deg)
    gaps = sum(1 for a, b in zip(ns, ns[1:]) if b != a + 1)
    info = dict(rate_hz=rate_hz, n=len(ns), n_gyro_rows=n_g, n_rv_rows=n_r, unpaired=n_g + n_r - 2 * len(ns),
                bad_rows=bad, n_gaps=gaps, gyro_step=GYRO_STEP, yaw_step_rad=YAW_STEP,
                yaw_mode_deg=mode_deg, yaw_snap_max_deg=round(snap, 6))
    return info, list(zip(range(len(ns)), k_gz, k_yaw))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw_static.csv of tools/imu_capture.py')
    ap.add_argument('--rate', type=int, required=True, choices=[20, 100])
    ap.add_argument('--src-label', required=True, help='path of --src written into the header (repo-neutral)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, '..', 'rpi5', 'data', f'imu_noise_replay_{a.rate}hz.csv')
    info, rows = build(a.src, float(a.rate))
    info['source'] = a.src_label.replace('\\', '/')
    info['source_sha256'] = sha256(a.src)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as f:
        f.write('# Measured BNO085 noise for replay (imu_model.ImuNoise), made by tools/make_imu_noise_replay.py\n')
        for k, v in info.items():
            f.write(f'# {k}: {v}\n')
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['i', 'k_gz', 'k_yaw'])
        w.writerows(rows)
    print(f'#ket_qua,ma=0,out={out},n={info["n"]},snap_deg={info["yaw_snap_max_deg"]}')


if __name__ == '__main__':
    main()
