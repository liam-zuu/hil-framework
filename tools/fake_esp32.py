#!/usr/bin/env python3
"""
fake_esp32.py — giả lập ESP32 gửi UDP packet đúng format
Dùng khi không có hardware để test agv_logger.py và plot_log.py

Usage:
    python fake_esp32.py                    # gửi lên localhost:5005
    python fake_esp32.py --host 192.168.1.x # gửi lên PC khác
    python fake_esp32.py --scenario circle  # chạy scenario hình tròn
    python fake_esp32.py --scenario drift   # simulate drift không có controller
    python fake_esp32.py --scenario still   # xe đứng yên, chỉ noise IMU
"""

import socket
import time
import math
import argparse
import random

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005
SEND_HZ      = 50       # 50Hz giống ESP32 control_task

# ─────────────────────────────────────────
# Noise helper
# ─────────────────────────────────────────
def noise(sigma):
    return random.gauss(0, sigma)

# ─────────────────────────────────────────
# Scenarios — trả về dict các giá trị tại thời điểm t (giây)
# ─────────────────────────────────────────
def scenario_still(t):
    """Xe đứng yên — chỉ có IMU noise, baseline test"""
    return dict(
        vx=0.0, vy=0.0, wz=0.0,
        sp_vx=0.0, sp_vy=0.0, sp_wz=0.0,
        pwm_fl=0, pwm_fr=0, pwm_rl=0, pwm_rr=0,
        fl_rpm=noise(0.5), fr_rpm=noise(0.5),
        rl_rpm=noise(0.5), rr_rpm=noise(0.5),
        ax=noise(0.02), ay=noise(0.02), az=9.81+noise(0.02),
        gx=noise(0.005), gy=noise(0.005), gz=noise(0.005),
        yaw=noise(0.1), pitch=noise(0.05), roll=noise(0.05),
    )

def scenario_drift(t):
    """Xe chạy thẳng không có controller — simulate drift"""
    # Setpoint: chạy thẳng vx=0.3 m/s
    sp_vx = 0.3
    # Actual bị drift — vy lệch dần, wz nhỏ
    drift_vy  = 0.02 * math.sin(0.3 * t)          # drift ngang
    drift_wz  = 0.05 * math.sin(0.1 * t)          # drift xoay
    actual_vx = sp_vx + noise(0.01)

    # RPM tương ứng (w = v/r, r=0.0485, PPR không quan trọng cho fake)
    base_rpm = sp_vx / 0.0485 * 60 / (2 * math.pi)
    return dict(
        vx=actual_vx, vy=drift_vy, wz=drift_wz,
        sp_vx=sp_vx, sp_vy=0.0, sp_wz=0.0,
        pwm_fl=300, pwm_fr=300, pwm_rl=300, pwm_rr=300,
        fl_rpm= base_rpm+noise(2), fr_rpm= base_rpm+noise(2),
        rl_rpm= base_rpm+noise(2), rr_rpm= base_rpm+noise(2),
        ax=noise(0.05), ay=drift_vy*0.5+noise(0.03),
        az=9.81+noise(0.02),
        gx=noise(0.01), gy=noise(0.01), gz=drift_wz+noise(0.005),
        yaw=drift_wz*t*57.3, pitch=noise(0.1), roll=noise(0.1),
    )

def scenario_circle(t):
    """Xe chạy hình tròn — trajectory test"""
    # Mecanum circle: vx=0, vy=0.3, wz=0.5
    sp_vy = 0.3
    sp_wz = 0.5
    actual_vy = sp_vy + noise(0.01)
    actual_wz = sp_wz + noise(0.005)

    base_rpm = sp_vy / 0.0485 * 60 / (2 * math.pi)
    yaw = sp_wz * t * 57.3  # degrees

    return dict(
        vx=noise(0.005), vy=actual_vy, wz=actual_wz,
        sp_vx=0.0, sp_vy=sp_vy, sp_wz=sp_wz,
        pwm_fl=-200, pwm_fr=500, pwm_rl=-200, pwm_rr=500,
        fl_rpm=-base_rpm+noise(2), fr_rpm= base_rpm+noise(2),
        rl_rpm=-base_rpm+noise(2), rr_rpm= base_rpm+noise(2),
        ax=noise(0.05), ay=actual_vy*actual_wz+noise(0.03),
        az=9.81+noise(0.02),
        gx=noise(0.01), gy=noise(0.01), gz=actual_wz+noise(0.005),
        yaw=yaw % 360, pitch=noise(0.1), roll=noise(0.1),
    )

SCENARIOS = {
    "still":  scenario_still,
    "drift":  scenario_drift,
    "circle": scenario_circle,
}

# ─────────────────────────────────────────
# CSV format — phải khớp với logger.c
# ─────────────────────────────────────────
HEADER = (
    "timestamp_us,"
    "fl_count,fr_count,rl_count,rr_count,"
    "fl_rpm,fr_rpm,rl_rpm,rr_rpm,"
    "vx,vy,wz,"
    "sp_vx,sp_vy,sp_wz,"
    "pwm_fl,pwm_fr,pwm_rl,pwm_rr,"
    "ax,ay,az,"
    "gx,gy,gz,"
    "yaw,pitch,roll\n"
)

def build_packet(t_us, enc_counts, d):
    return (
        f"{t_us},"
        f"{enc_counts[0]},{enc_counts[1]},{enc_counts[2]},{enc_counts[3]},"
        f"{d['fl_rpm']:.2f},{d['fr_rpm']:.2f},{d['rl_rpm']:.2f},{d['rr_rpm']:.2f},"
        f"{d['vx']:.3f},{d['vy']:.3f},{d['wz']:.3f},"
        f"{d['sp_vx']:.3f},{d['sp_vy']:.3f},{d['sp_wz']:.3f},"
        f"{d['pwm_fl']},{d['pwm_fr']},{d['pwm_rl']},{d['pwm_rr']},"
        f"{d['ax']:.3f},{d['ay']:.3f},{d['az']:.3f},"
        f"{d['gx']:.3f},{d['gy']:.3f},{d['gz']:.3f},"
        f"{d['yaw']:.2f},{d['pitch']:.2f},{d['roll']:.2f}\n"
    )

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fake ESP32 UDP sender")
    parser.add_argument("--host",     default=DEFAULT_HOST)
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT)
    parser.add_argument("--scenario", default="drift",
                        choices=list(SCENARIOS.keys()),
                        help="Scenario: still | drift | circle")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Thời gian chạy (giây), 0 = vô hạn")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.port)
    fn   = SCENARIOS[args.scenario]

    print(f"[INFO] Fake ESP32 → {args.host}:{args.port}")
    print(f"[INFO] Scenario: {args.scenario}")
    print(f"[INFO] Rate: {SEND_HZ}Hz | Ctrl+C để dừng\n")

    # Gửi header trước — giống ESP32 thật
    sock.sendto(HEADER.encode(), dest)
    time.sleep(0.05)

    period   = 1.0 / SEND_HZ
    t_start  = time.time()
    enc      = [0, 0, 0, 0]   # accumulate fake encoder counts
    pkt_count = 0

    try:
        while True:
            t_now = time.time()
            t_rel = t_now - t_start

            if args.duration > 0 and t_rel >= args.duration:
                break

            t_us = int(t_rel * 1e6)

            # Fake encoder count — tích lũy từ RPM
            d = fn(t_rel)
            for i, key in enumerate(["fl_rpm","fr_rpm","rl_rpm","rr_rpm"]):
                enc[i] += int(d[key] / 60.0 * 1320 * period)  # PPR=1320

            pkt = build_packet(t_us, enc, d)
            sock.sendto(pkt.encode(), dest)
            pkt_count += 1

            if pkt_count % (SEND_HZ * 5) == 0:   # print mỗi 5 giây
                print(f"[{t_rel:6.1f}s] vx={d['vx']:+.3f} vy={d['vy']:+.3f} "
                      f"wz={d['wz']:+.3f} | packets={pkt_count}")

            # Sleep chính xác
            next_t = t_start + pkt_count * period
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        pass

    print(f"\n[DONE] Gửi {pkt_count} packets ({t_rel:.1f}s)")
    sock.close()

if __name__ == "__main__":
    main()
