#!/usr/bin/env python3
"""
pc_sender.py — Gửi trajectory setpoint qua WiFi UDP đến ESP32.

Dùng để test traj_receiver WiFi source trước khi có H7 UART relay.

Usage:
    python3 pc_sender.py --ip 192.168.x.x --vx 0.2 --vy 0.0 --wz 0.0
    python3 pc_sender.py --ip 192.168.x.x --sweep
"""

import socket
import struct
import time
import argparse

TRAJ_START  = 0xBB
TRAJ_PORT   = 5010
SEND_HZ     = 50        # 50Hz

def pack_packet(vx: float, vy: float, wz: float) -> bytes:
    buf = bytearray(14)
    buf[0] = TRAJ_START
    struct.pack_into('<f', buf, 1, vx)
    struct.pack_into('<f', buf, 5, vy)
    struct.pack_into('<f', buf, 9, wz)
    chk = 0
    for b in buf[:13]:
        chk ^= b
    buf[13] = chk
    return bytes(buf)

def send_loop(ip: str, vx: float, vy: float, wz: float, duration: float = 0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / SEND_HZ
    t_start = time.time()
    n = 0
    print(f"Sending vx={vx:.2f} vy={vy:.2f} wz={wz:.2f} → {ip}:{TRAJ_PORT} @ {SEND_HZ}Hz")
    try:
        while True:
            pkt = pack_packet(vx, vy, wz)
            sock.sendto(pkt, (ip, TRAJ_PORT))
            n += 1
            if n % (SEND_HZ * 2) == 0:
                print(f"  sent {n} packets  t={time.time()-t_start:.1f}s")
            if duration > 0 and (time.time() - t_start) >= duration:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        # Gửi zero để dừng xe
        for _ in range(10):
            sock.sendto(pack_packet(0, 0, 0), (ip, TRAJ_PORT))
            time.sleep(0.02)
        sock.close()
        print(f"Done. Sent {n} packets total.")

def sweep_loop(ip: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / SEND_HZ
    steps = [
        (0.0,  0.0, 0.0,  2.0, "ZERO"),
        (0.2,  0.0, 0.0,  3.0, "FWD 0.2 m/s"),
        (0.0,  0.0, 0.0,  1.0, "ZERO"),
        (-0.2, 0.0, 0.0,  3.0, "BWD 0.2 m/s"),
        (0.0,  0.0, 0.0,  1.0, "ZERO"),
        (0.0,  0.2, 0.0,  3.0, "LEFT 0.2 m/s"),
        (0.0,  0.0, 0.0,  1.0, "ZERO"),
        (0.0,  0.0, 0.5,  3.0, "ROT 0.5 rad/s"),
        (0.0,  0.0, 0.0,  2.0, "ZERO"),
    ]
    print(f"Sweep mode → {ip}:{TRAJ_PORT}")
    try:
        for (vx, vy, wz, dur, label) in steps:
            print(f"  [{label}]  vx={vx:.2f} vy={vy:.2f} wz={wz:.2f}  ({dur}s)")
            t_end = time.time() + dur
            while time.time() < t_end:
                sock.sendto(pack_packet(vx, vy, wz), (ip, TRAJ_PORT))
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            sock.sendto(pack_packet(0, 0, 0), (ip, TRAJ_PORT))
            time.sleep(0.02)
        sock.close()
        print("Sweep done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory UDP sender")
    parser.add_argument("--ip",       required=True,       help="ESP32 IP address")
    parser.add_argument("--vx",       type=float, default=0.0)
    parser.add_argument("--vy",       type=float, default=0.0)
    parser.add_argument("--wz",       type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds (0=forever)")
    parser.add_argument("--sweep",    action="store_true",  help="Run sweep scenario")
    args = parser.parse_args()

    if args.sweep:
        sweep_loop(args.ip)
    else:
        send_loop(args.ip, args.vx, args.vy, args.wz, args.duration)
