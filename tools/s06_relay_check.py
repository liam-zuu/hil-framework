#!/usr/bin/env python3
"""
s06_relay_check.py — đối chiếu ba đầu relay UART của một lượt (spec S06 mục 2.2 W8, C3-3; Liam 30/10 "Làm cả gói,
0,2 %"). Thiếu công cụ này thì H7 bỏ mọi setpoint mà các công cụ bắt buộc khác vẫn có thể cùng ra mã 0.

Ba đầu:
  (1) Pi: số setpoint trong MOSI của frames.npz, tức số frame cờ VALID | NEW (`hil_node.py`: mỗi lần publisher
      đổi `shared_ref[3]` thì đúng một frame mang cờ). Đây là số Pi thực gửi, chặt hơn `sent=` lúc publisher dừng.
  (2) H7: `sent`, `dropped` ở dòng `Relay:` của log serial, hiệu giữa dòng ngay trước `SPI link up` đầu và dòng ngay
      sau `SPI link lost` cuối (`s06_h7_serial.run_span`).
  (3) ESP32: `relay_ok`, `relay_bad` ở dòng S, hiệu giữa dòng đầu và dòng cuối trong #run_start..#run_end.

Đạt khi (1) = sent + dropped, `dropped` = 0, `bad` = 0, và `ok` theo kịp ±5 %: H7 không có giờ Pi nên `ok` so với số
frame NEW giữa hai dòng S đó (giờ Pi), sau khi (1) đã bằng tổng (2).

**Chưa kiểm ở đây:** so giá trị đầu–cuối (Y10): dòng S của `s06_load` chỉ có bộ đếm, không có giá trị setpoint; phép
so đó ở env `relay_check` của hướng dẫn G2.

Usage:
  python tools/s06_relay_check.py <frames.npz> --esp32-events esp32_events.txt --h7-serial h7_serial.log --out <thư mục>
Mã thoát và dòng cuối `#ket_qua` theo bảng mã kết quả (spec 3.1): 0 đạt; 1 lệch một trong các điều kiện trên, mất
frame, hoặc bộ đếm mất dữ liệu ở file không chọn được lượt; 3 `khong_du_dieu_kien` thiếu mốc lượt, không có setpoint
nào, log frame không từ bước đầu; 3 `thieu_bang_chung` không có manifest, thiếu dòng `Relay:` hai đầu, dưới 2 dòng S;
2 không đọc được đầu vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s06_frames_decode as dec  # noqa: E402
import s06_h7_serial as ser  # noqa: E402
import s06_load_activity as la  # noqa: E402
import s06_stream as stm  # noqa: E402

TRAJ_SEND = 0b11                  # rpi5/protocol.py TRAJ_VALID | TRAJ_NEW; uart_relay.c _TRAJ_SEND_MASK
OK_TOL = 0.05                     # W8: `ok` theo kịp `sent` ±5 %
S_FIELDS = (15, 16)
COLUMNS = ['idx', 'pi_new', 'h7_sent', 'h7_dropped', 'esp32_ok', 'esp32_bad', 'pi_new_in_s_span']


def relay_span(ev: List[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """Dòng `Relay:` ngay trước `SPI link up` đầu và ngay sau `SPI link lost` cuối; None nếu thiếu."""
    span = ser.run_span(ev)
    if span is None:
        return None, None
    a, b = span
    before = [e for e in ev[:a] if e['kind'] == 'relay']
    after = [e for e in ev[b:] if e['kind'] == 'relay']
    return (before[-1] if before else None), (after[0] if after else None)


def read_s(path: str, lo: float, hi: float) -> List[Tuple[int, int, int]]:
    """(t_rx, relay_ok, relay_bad) các dòng S trong khoảng giờ Pi [lo; hi]. Dòng S không mang số lượt (s06_load.c)."""
    out = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line.startswith('S,'):
                continue
            p = line.split(',')
            if len(p) in S_FIELDS:
                try:
                    if lo <= int(t_rx) <= hi:
                        out.append((int(t_rx), int(p[10]), int(p[11])))
                except ValueError:
                    continue
    return out


def new_frames(mosi) -> np.ndarray:
    return (np.asarray(mosi)[:, la.OFS_TRAJ_FLAGS] & TRAJ_SEND) == TRAJ_SEND


def compare(new_total: int, new_in_span: Optional[int], before, after, s_rows) -> Tuple[dict, List[str], List[str]]:
    """(số đếm, lỗi, thiếu bằng chứng)."""
    fails, missing = [], []
    c = dict(pi_new=new_total, h7_sent=None, h7_dropped=None, esp32_ok=None, esp32_bad=None, pi_new_in_s_span=new_in_span)
    if before is None or after is None:
        missing.append('thiếu dòng Relay: ngay trước link up đầu hoặc ngay sau link lost cuối')
    else:
        c['h7_sent'], c['h7_dropped'] = after['sent'] - before['sent'], after['dropped'] - before['dropped']
        if c['h7_sent'] < 0 or c['h7_dropped'] < 0:
            fails.append(f"bộ đếm relay H7 giảm trong lượt ({before['sent']} → {after['sent']}): H7 khởi động lại?")
        elif c['h7_sent'] + c['h7_dropped'] != new_total:
            fails.append(f"Pi gửi {new_total} setpoint, H7 sent + dropped = {c['h7_sent'] + c['h7_dropped']}")
        if c['h7_dropped'] > 0:
            fails.append(f"H7 bỏ {c['h7_dropped']} setpoint")
    if len(s_rows) < 2:
        missing.append(f'chỉ có {len(s_rows)} dòng S trong lượt')
    else:
        c['esp32_ok'], c['esp32_bad'] = s_rows[-1][1] - s_rows[0][1], s_rows[-1][2] - s_rows[0][2]
        if c['esp32_bad']:
            fails.append(f"ESP32 có {c['esp32_bad']} gói relay sai")
        if abs(c['esp32_ok'] - new_in_span) > OK_TOL * new_in_span:
            fails.append(f"ESP32 nhận {c['esp32_ok']} gói, Pi gửi {new_in_span} trong cùng khoảng (lệch quá ±5 %)")
    return c, fails, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Đối chiếu ba đầu relay UART của một lượt (S06 W8)')
    ap.add_argument('frames')
    ap.add_argument('--esp32-events', required=True)
    ap.add_argument('--h7-serial', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', type=int, default=None, help='số lượt khi file có nhiều lượt')
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames, allow_pickle=False)
        t_send_ns, mosi, miso = z['t_send_ns'], z['mosi'], z['miso']
        manifest = dec.load_manifest(z)
        serial = ser.read_log(a.h7_serial)
        marks = stm.run_marks(a.esp32_events, a.run)
        stm.require_run(marks)
        if 'run_start' not in marks or 'run_end' not in marks:
            print('không chấm: thiếu #run_start hoặc #run_end', file=sys.stderr)
            return stm.report_result(3, ['khong_du_dieu_kien'])
        s_rows = read_s(a.esp32_events, marks['run_start']['t_rx'], marks['run_end']['t_rx'])
    except stm.RunProblem as exc:
        return stm.report_run_problem(exc)
    except (OSError, KeyError, ValueError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    t_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    new = new_frames(mosi)
    in_span = None
    if len(s_rows) >= 2:
        in_span = int(np.sum(new & (t_us >= s_rows[0][0]) & (t_us <= s_rows[-1][0])))
    counts, fails, missing = compare(int(new.sum()), in_span, *relay_span(serial), s_rows)
    fails += dec.manifest_check(manifest, t_send_ns, mosi, miso)
    conds = []
    if not new.any():
        conds.append('không có frame setpoint nào: lượt không chạy publisher')
    if manifest is None:
        missing.append('frames.npz không có manifest: không chứng minh được log frame đủ')
    elif manifest.get('step_first') != 0:
        conds.append(f"log frame bắt đầu ở bước {manifest.get('step_first')}: không so được với bộ đếm cả lượt của H7")
    code = 1 if fails else (3 if conds or missing else 0)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'S06_relay.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow(dict(counts, idx=0))
    summ = dict(counts, fails=fails, missing=missing, conditions=conds, exit_code=code,
                note='chỉ đối chiếu bộ đếm; so giá trị đầu–cuối (Y10) ở env relay_check, hướng dẫn G2')
    with open(os.path.join(a.out, 'S06_relay.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    return stm.report_result(code, (['khong_du_dieu_kien'] if conds else []) + (['thieu_bang_chung'] if missing else []))


if __name__ == '__main__':
    raise SystemExit(main())
