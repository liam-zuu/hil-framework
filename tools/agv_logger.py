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
BUFFER_SIZE  = 2048

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
    # AGV_MECANUM vel_ctrl metric counters (spec S02): tick, cumulative |dpwm|, saturated ticks
    "tick", "dpwm_fl", "dpwm_fr", "dpwm_rl", "dpwm_rr", "sat_fl", "sat_fr", "sat_rl", "sat_rr",
    # AGV_MECANUM ctrl-imu (controller-IMU spec v2.2 section 8): log loss counters, motor_on
    # (0 = 10 Hz line while the motor is off), imu_init code, IMU fusion / heading loop.
    # Old columns keep their meaning: wz = encoder wz, gz = raw chip gyro, yaw in degrees.
    "log_seq", "log_qdrop", "log_send_err", "motor_on", "imu_init",
    "wz_meas", "wz_cmd", "gz_fused", "yaw_rad", "age_g_ms", "age_r_ms", "gyro_n", "rv_n", "reset_n",
    "g_valid", "r_valid", "wz_src", "src_switch_count", "psi_ref", "e_psi", "head_on", "psi_reset_count",
    "slip_flag", "slip_on_count", "r", "bias_gz",
    # Added 28/10 (Codex review of ctrl-imu #11, #8 (c)): slip_eval 0 = r could not be checked
    # this tick (gyro invalid or wheels below omega_min), so slip_flag 0 does not mean "no slip".
    "src_switch_dwz", "bias_updates", "slip_eval",
]
N_OLD = 28      # firmware before the counters: same columns without the last 9 + 29
N_S02 = 37      # firmware a6644a5 .. before ctrl-imu: without the last 29

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

            # End-of-run summary (sent 3 times): keep it next to the CSV, not in it
            if line.startswith("#"):
                print(f"[RUN] {line}")
                with open(filename + ".events.txt", "a") as ev:
                    ev.write(f"{datetime.datetime.now().isoformat()} {line}\n")
                continue

            # Skip header packet
            if "timestamp_us" in line:
                if not header_passed:
                    print(f"[INFO] Kết nối từ {addr[0]} — bắt đầu nhận data")
                    header_passed = True
                continue

            # Parse và validate
            fields = line.split(",")
            if len(fields) in (N_OLD, N_S02):
                fields += [""] * (len(COLUMNS) - len(fields))
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
