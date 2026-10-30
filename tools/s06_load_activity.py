#!/usr/bin/env python3
"""
s06_load_activity.py — tải của một lượt S06 theo lịch (spec mục 2.2, V7, Z8), đọc từ frames.npz.

  * Kiểm lượt chạy đúng lịch: manifest ghi đúng file lịch (sha256), plant là MecanumPlant khi có khối tải
    nặng (Z8: không dùng MockPlant), mọi frame trong khối mang đúng ω của lịch (sai quá 1 LSB của frame).
  * Báo % thời gian mỗi tác vụ hoạt động, riêng khối tải nặng và phần nhóm A (V7, "tính từ log"):
      encoder bánh i: |ω_d| > 0,3 rad/s trong MOSI (EncoderGen có phát cạnh, spec Z1);
      PWM bánh i: 0 < |duty| < 1 trong MISO (có băm);
      relay: cửa sổ 100 ms có ít nhất một frame cờ NEW;
      IMU: cửa sổ 1 s có report cả 3 loại trong luồng log H7 (cần bản ghi REPORT; không có thì để trống).

Đây chỉ là chỉ báo phía Pi. **Chưa làm** "điều kiện đủ tải" của Z8 theo cửa sổ 10 ms: số count ESP32 cần đồng
hồ ESP32 → Pi (chân đánh dấu, chờ G2), số cạnh PWM mỗi 10 ms H7 chưa xuất.

Usage:
  python tools/s06_load_activity.py <frames.npz> --schedule S06_load.csv --out <thư mục>
Mã thoát và dòng cuối `#ket_qua` theo bảng mã kết quả (spec 3.1): 0 đúng lịch; 1 lượt không chạy đúng lịch hoặc
frames.npz mất frame; 2 không đọc được đầu vào; 3 `thieu_bang_chung` không có manifest; 3 `khong_du_dieu_kien` lịch
không có khối tải nặng (CB3 (9), C3-4: lượt T2 chính phải có `heavy`; lượt cấu hình S02 chạy `--no-heavy` thì chấm với
`--allow-no-heavy`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s06_frames_decode as dec  # noqa: E402
import s06_stream as stm  # noqa: E402

OMEGA_FULL = 40.0                 # hil_protocol.h: ω = int16 · 40 / 32767
OMEGA_LSB = OMEGA_FULL / 32767.0
ENC_MIN = 0.3                     # spec Z1: |ω_d| ≤ 0,3 rad/s không phát cạnh
OFS_TRAJ_FLAGS = 30               # rpi5/protocol.py
TRAJ_NEW = 1 << 1
SWAP_MS, W_LOW, W_HIGH = 50, 30.0, 39.2      # bản sao hằng số của rpi5/s06_load_schedule.py (Z8)
PHASE_MS = (0, 13, 25, 38)
RELAY_WIN = 100                   # frame
IMU_WIN = 1000


def read_schedule(path: str) -> List[dict]:
    with open(path, newline='') as f:
        rows = [dict(kind=r['kind'], test=r['test'], start=int(r['start_ms']), end=int(r['end_ms']))
                for r in csv.DictReader(f)]
    if not rows:
        raise ValueError('file lịch rỗng')
    return rows


def heavy_omega_table(n: int) -> np.ndarray:
    """ω của 4 bánh, n bước từ đầu khối."""
    k = np.arange(n)[:, None]
    return np.where(((k + np.array(PHASE_MS)) // SWAP_MS) & 1, W_HIGH, W_LOW)


def frame_omega(mosi: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.asarray(mosi)[:, :8].tobytes(), dtype='>i2').reshape(-1, 4) * OMEGA_LSB


def frame_duty(miso: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.asarray(miso)[:, :8].tobytes(), dtype='>i2').reshape(-1, 4) / 32767.0


def windows_pct(flags: np.ndarray, size: int) -> Optional[float]:
    """% cửa sổ trọn `size` frame có ít nhất một frame đúng."""
    n = len(flags) // size
    if n == 0:
        return None
    return 100.0 * float(np.mean(flags[:n * size].reshape(n, size).any(axis=1)))


def imu_frames(t_send_ns, mosi, miso) -> Optional[Dict[str, np.ndarray]]:
    """Từng loại report: cờ theo frame mang bản ghi; None khi luồng log không có bản ghi REPORT."""
    rows, _ = dec.decode_frames(t_send_ns, mosi, miso)
    recs, _ = dec.log_records(rows)
    out = {k: np.zeros(len(t_send_ns), dtype=bool) for k in dec.REPORT_KIND.values()}
    found = False
    for r in recs:
        if dec.REC_REPORT <= r['type'] <= dec.REC_REPORT + 2:
            out[dec.REPORT_KIND[r['type'] - dec.REC_REPORT]][r['carrier_j']] = True
            found = True
    return out if found else None


def segment_row(name: str, mask: np.ndarray, omega, duty, new, imu) -> dict:
    n = int(mask.sum())
    row = dict(segment=name, frames=n)
    for w in range(4):
        row[f'enc_pct_{w + 1}'] = 100.0 * float(np.mean(np.abs(omega[mask, w]) > ENC_MIN)) if n else None
        d = np.abs(duty[mask, w])
        row[f'pwm_pct_{w + 1}'] = 100.0 * float(np.mean((d > 0.0) & (d < 1.0))) if n else None
    row['relay_pct'] = windows_pct(new[mask], RELAY_WIN)
    if imu is None:
        row['imu_pct'] = None
    else:
        per = [windows_pct(v[mask], IMU_WIN) for v in imu.values()]
        row['imu_pct'] = None if any(p is None for p in per) else min(per)
    return row


def check_run(manifest, sched_path: str, sched, omega) -> Tuple[List[str], List[str]]:
    """(lỗi, thiếu bằng chứng)."""
    fails, missing = [], []
    heavy = [r for r in sched if r['kind'] == 'heavy']
    if manifest is None:
        return fails, ['frames.npz không có manifest: không biết lượt chạy file lịch nào']
    with open(sched_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    run_sha = (manifest.get('load_schedule') or ' ').rsplit(' ', 1)[-1]
    if run_sha != sha:
        fails.append(f'manifest ghi lịch {run_sha or "(không có)"}, khác file {sha}')
    if heavy:
        h = heavy[0]
        if manifest.get('plant') != 'MecanumPlant':
            fails.append(f"plant của lượt là {manifest.get('plant')}, Z8 cần MecanumPlant")
        if list(manifest.get('heavy_block') or []) != [h['start'], h['end']]:
            fails.append(f"manifest ghi khối {manifest.get('heavy_block')}, lịch {h['start']}..{h['end']}")
        if h['end'] > len(omega):
            fails.append(f"khối tải nặng tới bước {h['end']}, lượt chỉ có {len(omega)} frame")
        else:
            bad = np.nonzero(np.any(np.abs(omega[h['start']:h['end']] - heavy_omega_table(h['end'] - h['start']))
                                    > 1.5 * OMEGA_LSB, axis=1))[0]
            if len(bad):
                fails.append(f"{len(bad)} frame trong khối tải nặng khác lịch, đầu tiên bước {h['start'] + int(bad[0])}")
    return fails, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Tải của một lượt S06 theo lịch (mục 2.2)')
    ap.add_argument('frames')
    ap.add_argument('--schedule', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-no-heavy', action='store_true', help='lượt cấu hình S02, lịch tạo bằng --no-heavy')
    a = ap.parse_args(argv)
    try:
        sched = read_schedule(a.schedule)
        z = np.load(a.frames, allow_pickle=False)
        t_send, mosi, miso = z['t_send_ns'], z['mosi'], z['miso']
        manifest = dec.load_manifest(z)
    except (OSError, KeyError, ValueError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    omega, duty = frame_omega(mosi), frame_duty(miso)
    new = (np.asarray(mosi)[:, OFS_TRAJ_FLAGS] & TRAJ_NEW) != 0
    fails, missing = check_run(manifest, a.schedule, sched, omega)
    fails += dec.manifest_check(manifest, t_send, mosi, miso)
    # C3-4: a schedule without a heavy block skips every check above, so it cannot close a main T2 run
    conds = [] if a.allow_no_heavy or any(r['kind'] == 'heavy' for r in sched) else ['lịch không có khối tải nặng (Z8)']
    imu = imu_frames(t_send, mosi, miso)
    in_heavy = np.zeros(len(omega), dtype=bool)
    for r in sched:
        if r['kind'] == 'heavy':
            in_heavy[r['start']:r['end']] = True
    rows = [segment_row('heavy', in_heavy, omega, duty, new, imu), segment_row('nhom_a', ~in_heavy, omega, duty, new, imu)]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'S06_load_activity.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = dict(frames=len(omega), fails=fails, missing=missing, conditions=conds, segments=rows,
                note='chỉ báo phía Pi; điều kiện đủ tải 10 ms của Z8 chưa kiểm ở đây')
    with open(os.path.join(a.out, 'S06_load_activity.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    return stm.report_result(1 if fails else (3 if missing or conds else 0),
                             (['khong_du_dieu_kien'] if conds else []) + (['thieu_bang_chung'] if missing else []))


if __name__ == '__main__':
    raise SystemExit(main())
