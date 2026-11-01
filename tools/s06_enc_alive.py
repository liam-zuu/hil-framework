#!/usr/bin/env python3
"""
s06_enc_alive.py — kiểm encoder còn tới ESP32 suốt lượt (spec S06 CB5: lượt nguồn buck, không cáp ST-Link; "lượt
đúng giờ ở Pi mà H7 ngừng trả lời hay encoder dừng là không đạt"). Liam 30/10: "Viết công cụ".

`s06_esp32_check.py` chỉ đếm số lần đọc, count đứng yên vẫn đạt. Công cụ này so count ESP32 đọc được với ω trong
frame Pi gửi, theo cửa sổ `--bin-s` (mặc định 1 s) trong #run_start..#run_end, trên đồng hồ Pi (giờ logger nhận dòng
E, cùng đồng hồ với t_send_ns). Cửa sổ dài hơn trễ UDP nhiều nên không cần chân đánh dấu.
  * Một bánh được xét ở một cửa sổ khi mọi frame của cửa sổ có |ω| ≥ `--omega-min` cùng dấu (bánh dừng hay đổi
    chiều hợp lệ thì không xét, không báo sai).
  * Bánh còn sống khi Δcount cùng dấu ω và |Δcount| ≥ `--ratio` × K·ω_trung bình·(khoảng giữa hai lần đọc đầu,
    cuối của cửa sổ), K = 1320/2π. Đây chỉ là phép kiểm còn chạy, không phải H5: độ chính xác tần số thuộc T5.
  * Cửa sổ thiếu dòng E (dưới hai dòng, hay hai đầu cách nhau chưa tới nửa cửa sổ) = encoder không tới được.

Ngưỡng `--omega-min` 0,5 rad/s, `--ratio` 0,5, `--min-bins` 10 là [ĐỀ XUẤT CODE], chưa có số đo trên board.

Usage:
  python tools/s06_enc_alive.py <frames.npz> --esp32-events esp32_events.txt --out <thư mục> [--run N]
Mã thoát và dòng cuối `#ket_qua` theo bảng mã kết quả (spec 3.1): 0 mọi bánh còn sống; 1 có bánh dừng khi có lệnh
quay, thiếu dòng E, mất frame, hoặc bộ đếm mất dữ liệu ở file không chọn được lượt; 3 `khong_du_dieu_kien` thiếu mốc
lượt, frame không phủ lượt, hay có bánh chưa đủ `--min-bins` cửa sổ để xét; 3 `thieu_bang_chung` không có manifest;
2 không đọc được đầu vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s06_frames_decode as dec  # noqa: E402
import s06_load_activity as la  # noqa: E402
import s06_stream as stm  # noqa: E402

K_COUNT = 1320.0 / (2.0 * math.pi)     # count mỗi rad, như rpi5/t3b_schedule.py
OMEGA_MIN = 0.5                        # [ĐỀ XUẤT CODE] rad/s, trên vùng chết 0,3 rad/s của Z1
RATIO = 0.5                            # [ĐỀ XUẤT CODE]
MIN_BINS = 10                          # [ĐỀ XUẤT CODE]
BIN_S = 1.0
WHEELS = ('FL', 'FR', 'RL', 'RR')
COLUMNS = ['idx', 'bin', 't_start_us', 'wheel', 'omega_mean', 'count_expected', 'count_meas', 'state']


def read_e(path: str, marks: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(t_rx µs, count int64 đã mở vòng [n, 4]) các dòng E của lượt, theo thứ tự đọc (eseq)."""
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line.startswith('E,'):
                continue
            e = stm.parse_e(line.split(','))
            if e is not None and stm.e_of_run(e, marks):
                rows.append((e[0], int(t_rx), e[3]))
    rows.sort(key=lambda r: r[0])
    if not rows:
        return np.empty(0), np.empty((0, 4), dtype=np.int64)
    return np.array([r[1] for r in rows], dtype=float), stm.unwrap_i32(np.array([r[2] for r in rows], dtype=np.int64))


def score_bin(omega, t_e, c_e, wheel: int, sign: int, bin_us: float, omega_min: float, ratio: float) -> dict:
    """Một bánh, một cửa sổ. state: `skip` không xét, `no_reads` thiếu dòng E, `dead`, `alive`."""
    w = omega[:, wheel]
    out = dict(omega_mean=float(w.mean()) if len(w) else 0.0, count_expected=None, count_meas=None)
    if not len(w) or not (np.all(w >= omega_min) or np.all(w <= -omega_min)):
        return dict(out, state='skip')
    if len(t_e) < 2 or t_e[-1] - t_e[0] < bin_us / 2.0:
        return dict(out, state='no_reads')
    want = K_COUNT * out['omega_mean'] * (t_e[-1] - t_e[0]) * 1e-6
    got = sign * float(c_e[-1, wheel] - c_e[0, wheel])
    ok = got * want > 0 and abs(got) >= ratio * abs(want)
    return dict(out, count_expected=want, count_meas=got, state='alive' if ok else 'dead')


def check(t_send_us, omega, t_e, c_e, window, signs, bin_s=BIN_S, omega_min=OMEGA_MIN, ratio=RATIO) -> List[dict]:
    """Mọi cửa sổ trọn `bin_s` trong `window` (giờ Pi), mọi bánh."""
    t0, t1 = window
    bin_us = bin_s * 1e6
    rows = []
    for b in range(int((t1 - t0) // bin_us)):
        lo, hi = t0 + b * bin_us, t0 + (b + 1) * bin_us
        fr = (t_send_us >= lo) & (t_send_us < hi)
        er = (t_e >= lo) & (t_e < hi)
        for w in range(4):
            row = score_bin(omega[fr], t_e[er], c_e[er], w, signs[w], bin_us, omega_min, ratio)
            rows.append(dict(row, bin=b, t_start_us=lo, wheel=WHEELS[w]))
    return rows


def summarize(rows: List[dict], min_bins: int = MIN_BINS) -> dict:
    per = {w: {s: sum(1 for r in rows if r['wheel'] == w and r['state'] == s)
               for s in ('alive', 'dead', 'no_reads', 'skip')} for w in WHEELS}
    return dict(per_wheel=per, dead=sum(p['dead'] for p in per.values()), no_reads=sum(p['no_reads'] for p in per.values()),
                too_few=[w for w, p in per.items() if p['alive'] + p['dead'] + p['no_reads'] < min_bins], min_bins=min_bins,
                first_bad=next((dict(bin=r['bin'], wheel=r['wheel'], state=r['state']) for r in rows
                                if r['state'] in ('dead', 'no_reads')), None))


def outcome(summ: dict, frame_faults: List[str], manifest, cover_note: str) -> Tuple[int, Optional[str]]:
    if summ['dead'] or summ['no_reads'] or frame_faults:
        return 1, None
    if cover_note or summ['too_few']:
        return 3, 'khong_du_dieu_kien'
    if manifest is None:
        return 3, 'thieu_bang_chung'
    return 0, None


def write_rows(path: str, rows: List[dict]) -> None:
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({k: (f'{v:.3f}' if k == 't_start_us' else f'{v:.6g}') if isinstance(v, float) else v
                         for k, v in dict(r, idx=i).items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Kiểm encoder còn tới ESP32 suốt lượt (S06 CB5)')
    ap.add_argument('frames')
    ap.add_argument('--esp32-events', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', type=int, default=None, help='số lượt khi file có nhiều lượt')
    ap.add_argument('--count-sign', default='1,1,1,1',
                    help='dấu count ESP32 theo bánh (ENC_PI 2026-10-27_run06: count cùng dấu ω cả 4 bánh)')
    ap.add_argument('--bin-s', type=float, default=BIN_S)
    ap.add_argument('--omega-min', type=float, default=OMEGA_MIN)
    ap.add_argument('--ratio', type=float, default=RATIO)
    ap.add_argument('--min-bins', type=int, default=MIN_BINS)
    a = ap.parse_args(argv)
    try:
        signs = [int(x) for x in a.count_sign.split(',')]
        if len(signs) != 4 or set(signs) - {1, -1}:
            raise ValueError(f'--count-sign: {a.count_sign}')
        z = np.load(a.frames, allow_pickle=False)
        t_send_ns, mosi, miso = z['t_send_ns'], z['mosi'], z['miso']
        manifest = dec.load_manifest(z)
        marks = stm.run_marks(a.esp32_events, a.run)
        stm.require_run(marks)
        if 'run_start' not in marks or 'run_end' not in marks:
            print('không chấm: thiếu #run_start hoặc #run_end', file=sys.stderr)
            return stm.report_result(3, ['khong_du_dieu_kien'])
        t_e, c_e = read_e(a.esp32_events, marks)
    except stm.RunProblem as exc:
        return stm.report_run_problem(exc)
    except (OSError, KeyError, ValueError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    window = (float(marks['run_start']['t_rx']), float(marks['run_end']['t_rx']))
    t_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    cover_note = '' if t_us[0] <= window[0] and t_us[-1] >= window[1] else 'frame Pi không phủ lượt'
    rows = check(t_us, la.frame_omega(mosi), t_e, c_e, window, signs, a.bin_s, a.omega_min, a.ratio)
    summ = summarize(rows, a.min_bins)
    faults = dec.manifest_check(manifest, t_send_ns, mosi, miso)
    code, why = outcome(summ, faults, manifest, cover_note)
    os.makedirs(a.out, exist_ok=True)
    write_rows(os.path.join(a.out, 'S06_enc_alive.csv'), rows)
    summ.update(window_pi_us=window, frame_log_faults=faults, cover_note=cover_note, manifest_present=manifest is not None,
                omega_min=a.omega_min, ratio=a.ratio, bin_s=a.bin_s, count_sign=a.count_sign, exit_code=code)
    with open(os.path.join(a.out, 'S06_enc_alive.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    return stm.report_result(code, [why] if why else ())


if __name__ == '__main__':
    raise SystemExit(main())
