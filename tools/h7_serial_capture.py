#!/usr/bin/env python3
"""
h7_serial_capture.py — ghi log serial của H7 (ST-Link VCP, USART3, 115200) ra file cho S06 T1, T10 và
mọi lượt đo có H7, thay cho việc nhớ bật ghi log trong PuTTY (chỗ hở T1/T10, Liam 30/10).

Mỗi dòng ghi thành `<t_pc_us>\t<dòng>`; t_pc_us = time.monotonic_ns() // 1000 của máy chạy script (chỉ
để biết thứ tự và khoảng cách, không đồng bộ với Pi hay ESP32). tools/s06_h7_serial.py đọc được file này
(bỏ qua phần trước dấu tab). Dòng đầu file là `#capture,port=...,baud=...,start=<ngày giờ>`.
Dừng bằng Ctrl+C hoặc `--duration`; cuối file ghi `#capture_end,lines=<n>,bytes=<n>,reason=<...>`.
PuTTY và script này không mở cùng một cổng COM được: đóng PuTTY trước.

Usage:
  python tools/h7_serial_capture.py --port COM6 --out <thư mục lượt>/h7_serial.log [--duration 700]
Mã thoát: 0 ghi xong; 2 không mở được cổng.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time

BAUD = 115200


def now_us():
    return time.monotonic_ns() // 1000


def capture(ser, out, duration_s=None, clock=now_us, max_idle_reads=None):
    """Đọc từ `ser` (có read(n), in_waiting) tới khi hết thời lượng; trả (số dòng, số byte, lý do)."""
    buf, n_lines, n_bytes = b'', 0, 0
    t_end = None if duration_s is None else clock() + duration_s * 1e6
    idle = 0
    reason = 'duration'
    try:
        while t_end is None or clock() < t_end:
            chunk = ser.read(max(1, getattr(ser, 'in_waiting', 0) or 1))
            if not chunk:
                idle += 1
                if max_idle_reads is not None and idle >= max_idle_reads:
                    reason = 'idle'
                    break
                continue
            idle = 0
            n_bytes += len(chunk)
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                out.write(f"{clock()}\t{line.decode('utf-8', errors='replace').rstrip(chr(13))}\n")
                n_lines += 1
            out.flush()
    except KeyboardInterrupt:
        reason = 'operator'
    if buf:
        out.write(f"{clock()}\t{buf.decode('utf-8', errors='replace').rstrip(chr(13))}\n")
        n_lines += 1
    return n_lines, n_bytes, reason


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Ghi log serial H7 (ST-Link VCP)')
    ap.add_argument('--port', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--baud', type=int, default=BAUD)
    ap.add_argument('--duration', type=float, default=None, help='giây; bỏ trống = tới khi Ctrl+C')
    a = ap.parse_args(argv)
    try:
        import serial
        ser = serial.Serial(a.port, a.baud, timeout=0.2)
    except Exception as exc:                       # noqa: BLE001 (pyserial raises several types)
        print(f'không mở được cổng {a.port}: {exc}', file=sys.stderr)
        return 2
    start = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    with ser, open(a.out, 'w', encoding='utf-8', newline='\n') as out:
        out.write(f'{now_us()}\t#capture,port={a.port},baud={a.baud},start={start}\n')
        print(f'[H7] ghi {a.port} -> {a.out} (Ctrl+C để dừng)', flush=True)
        n, nb, reason = capture(ser, out, a.duration)
        out.write(f'{now_us()}\t#capture_end,lines={n},bytes={nb},reason={reason}\n')
    print(f'[H7] {n} dòng, {nb} byte, dừng: {reason}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
