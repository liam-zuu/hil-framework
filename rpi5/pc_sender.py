#!/usr/bin/env python3
"""
pc_sender.py — Gửi trajectory setpoint qua WiFi UDP đến ESP32.

Dùng để test traj_receiver WiFi source trước khi có H7 UART relay.

Usage:
    python3 pc_sender.py --ip 192.168.x.x --vx 0.2 --vy 0.0 --wz 0.0
    python3 pc_sender.py --ip 192.168.x.x --sweep
    python3 pc_sender.py --ip 192.168.x.x --test A-K1

--test A-K1|A-K2|A-K3 (C5, xe ke banh; Liam 10/11 "Duyet ca ba", 03_CODE/docs/
2026-11-10_Chay_thu_tren_giay_Kiem_xe_ke_banh.md muc 3.3): lich cua bai lay tu
trajectory_pub.get_body_velocity (cung lich voi HIL), goi k gui o t = k / 50 s tinh tu
luc bat dau (lich tuyet doi, khong troi theo thoi gian ngu), het bai gui 0 trong 0,2 s.
In dong '#test_start' va '#test_end' co gio may (epoch, ISO) de ghep voi log xe.

--o1 O1-F02|O1-F05|O1-F08|O1-K1|O1-K2 (C6, L76, Liam 10/11 "Duyet, lam luon"): nhiem vu san O1,
lich cua rpi5/o1_tasks.py (so 8 = fwsim/o1_feasibility.Schedule, cung bo sinh voi S03/S04 va
run_hil --o1); cung vong gui nhu --test. O1-K1, O1-K2 la tap kiem cuoi: can --allow-final-set.

L85 Q1 (Liam 11/11, CODE local_821ee4dc "duyet Q1-8 theo khuyen nghi"; Fable O1/C6 F-1, F-5, F-10):
gui moi goi k co k/50 < T_lenh, roi goi 0 tren CUNG luoi (khong de goi 0 dau de goi lich cuoi);
--o1 gui 0 trong o1_tasks.O1_TAIL_S = t_tre + 0,5 = 2,5 s (bang fwsim: lenh 0 toi han T_lenh + t_tre),
--test A-K* giu 10 goi 0 (0,2 s) nhu C5. Luoi tinh bang dong ho don dieu (time.monotonic); gio may
(time.time) chi de in '#test_start', '#test_end'. Ctrl-C: 10 goi 0 ngay.
"""
import math

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

def test_schedule(test_id: str, allow_final_set: bool = False):
    """(duration [s], f(t) -> (vx, vy, wz)) of a lifted-wheel test of group A (trajectory_pub)
    or of an O1 floor task (o1_tasks, C6)."""
    if test_id.startswith('O1-'):
        import o1_tasks
        return o1_tasks.task(test_id, allow_final_set)
    from trajectory_pub import GROUP_A, LIFTED_TESTS, get_body_velocity
    if test_id not in LIFTED_TESTS:
        raise ValueError(f'--test {test_id}: only the lifted-wheel tests {LIFTED_TESTS}')
    spec = GROUP_A[test_id]
    return spec.duration, (lambda t: get_body_velocity(t, spec))


LIFTED_TAIL_ZEROS = 10      # --test A-K*: 0.2 s of zeros after the schedule (C5, unchanged)


def grid_counts(test_id: str, duration: float):
    """(schedule packets, zero packets) on the 50 Hz grid: every k with k/50 < duration, then the
    zeros: O1 tasks up to duration + o1_tasks.O1_TAIL_S (k/50 < that), A-K tests 10 packets."""
    n_total = int(math.ceil(duration * SEND_HZ - 1e-9))
    if test_id.startswith('O1-'):
        import o1_tasks
        n_zero = int(math.ceil((duration + o1_tasks.O1_TAIL_S) * SEND_HZ - 1e-9)) - n_total
    else:
        n_zero = LIFTED_TAIL_ZEROS
    return n_total, n_zero


def test_loop(ip: str, test_id: str, sock=None, clock=time.monotonic, sleep=time.sleep, out=print,
              allow_final_set: bool = False, wall=time.time) -> int:
    """Send the schedule of test_id at SEND_HZ on an absolute time grid (clock: monotonic), then
    the zeros of grid_counts on the same grid; returns the number of schedule packets sent.
    wall gives the epoch printed in '#test_start' / '#test_end' only."""
    duration, f = test_schedule(test_id, allow_final_set)
    own = sock is None
    sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n_total, n_zero = grid_counts(test_id, duration)
    t0, e0 = clock(), wall()
    out(f"#test_start,test={test_id},t_epoch={e0:.3f},iso={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(e0))},"
        f"duration_s={duration:g},send_hz={SEND_HZ},tail_zeros={n_zero},dst={ip}:{TRAJ_PORT}")

    def at(k):
        wait = t0 + k / SEND_HZ - clock()
        if wait > 0:
            sleep(wait)

    n, nz, interrupted = 0, 0, False
    try:
        for k in range(n_total):
            at(k)
            vx, vy, wz = f(k / SEND_HZ)
            sock.sendto(pack_packet(vx, vy, wz), (ip, TRAJ_PORT))
            n += 1
        for k in range(n_total, n_total + n_zero):
            at(k)
            sock.sendto(pack_packet(0, 0, 0), (ip, TRAJ_PORT))
            nz += 1
    except KeyboardInterrupt:
        interrupted = True
        out('#test_interrupted')
    finally:
        if interrupted or nz < n_zero:
            for _ in range(LIFTED_TAIL_ZEROS):
                sock.sendto(pack_packet(0, 0, 0), (ip, TRAJ_PORT))
                sleep(0.02)
        out(f"#test_end,test={test_id},t_epoch={wall():.3f},zeros={nz},tail_planned={n_zero},"
            f"sent={n},planned={n_total},complete={int(n == n_total)}")
        if own:
            sock.close()
    return n


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
    parser.add_argument("--test",     default=None, choices=["A-K1", "A-K2", "A-K3"],
                        help="lifted-wheel test of group A, schedule of trajectory_pub (C5)")
    parser.add_argument("--o1",       default=None,
                        choices=["O1-F02", "O1-F05", "O1-F08", "O1-K1", "O1-K2"],
                        help="O1 floor task, schedule of o1_tasks (C6)")
    parser.add_argument("--allow-final-set", action="store_true",
                        help="allow O1-K1, O1-K2 (final test set, O1 3.7: camera trial or final check only)")
    args = parser.parse_args()

    if args.test and args.o1:
        parser.error("--test and --o1 exclude each other")
    if args.o1:
        try:
            test_loop(args.ip, args.o1, allow_final_set=args.allow_final_set)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.test:
        test_loop(args.ip, args.test)
    elif args.sweep:
        sweep_loop(args.ip)
    else:
        send_loop(args.ip, args.vx, args.vy, args.wz, args.duration)
