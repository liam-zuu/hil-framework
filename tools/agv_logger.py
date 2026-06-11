#!/usr/bin/env python3
"""
agv_logger.py — nhận UDP từ ESP32 (hoặc fake_esp32.py), lưu CSV
Không thay đổi gì giữa test laptop và hardware thật.

Usage:
    python agv_logger.py                    # lắng nghe 0.0.0.0:5005
    python agv_logger.py --port 5006        # đổi port
    python agv_logger.py --out my_log.csv   # đặt tên file output
"""

import socket
import csv
import argparse
import datetime
import os

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEFAULT_PORT = 5005
BUFFER_SIZE  = 512

COLUMNS = [
    "timestamp_us",
    "fl_count", "fr_count", "rl_count", "rr_count",
    "fl_rpm",   "fr_rpm",   "rl_rpm",   "rr_rpm",
    "vx", "vy", "wz",
    "sp_vx", "sp_vy", "sp_wz",
    "pwm_fl", "pwm_fr", "pwm_rl", "pwm_rr",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "yaw", "pitch", "roll",
]

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AGV UDP Logger")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out",  default=None, help="Tên file CSV output")
    args = parser.parse_args()

    # Tên file tự động nếu không chỉ định
    if args.out is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agv_log_{ts}.csv"
    else:
        filename = args.out

    print(f"[INFO] Lắng nghe UDP 0.0.0.0:{args.port}")
    print(f"[INFO] Lưu vào: {filename}")
    print(f"[INFO] Ctrl+C để dừng\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(5.0)

    csvfile = open(filename, "w", newline="")
    writer  = csv.writer(csvfile)
    writer.writerow(COLUMNS)

    packet_count  = 0
    drop_count    = 0
    header_passed = False

    try:
        while True:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                print(f"[WAIT] Chưa nhận được data... ({packet_count} packets saved)")
                continue

            line = data.decode("utf-8", errors="ignore").strip()

            # Skip header packet
            if "timestamp_us" in line:
                if not header_passed:
                    print(f"[INFO] Kết nối từ {addr[0]} — bắt đầu nhận data")
                    header_passed = True
                continue

            # Parse và validate
            fields = line.split(",")
            if len(fields) != len(COLUMNS):
                drop_count += 1
                continue

            writer.writerow(fields)
            csvfile.flush()
            packet_count += 1

            # Progress mỗi 100 packets
            if packet_count % 100 == 0:
                try:
                    ts_ms  = int(fields[0]) // 1000
                    vx     = float(fields[10])
                    vy     = float(fields[11])
                    wz     = float(fields[12])
                    fl_rpm = float(fields[4])
                    print(f"[{ts_ms:8d}ms] vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.3f} "
                          f"| fl_rpm={fl_rpm:+6.1f} | saved={packet_count} drops={drop_count}")
                except (ValueError, IndexError):
                    pass

    except KeyboardInterrupt:
        print(f"\n[DONE] Dừng — đã lưu {packet_count} packets ({drop_count} drops)")
        print(f"[DONE] File: {filename}")
        if packet_count > 0:
            print(f"\n[NEXT] Plot data:")
            print(f"       python plot_log.py {filename}")

    finally:
        csvfile.close()
        sock.close()

if __name__ == "__main__":
    main()
