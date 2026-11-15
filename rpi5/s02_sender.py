#!/usr/bin/env python3
"""
s02_sender.py: spec S02 group A with the command path of the campaign floor tests (S02 v1.5 P2-2b,
"Viec CODE" 1), and a preparation phase before k = 0 (S02 P2-2 "Trinh tu khoi tao": every reset of the
firmware happens before t = 0).

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van), sections 3.1, 3.2; Liam 15/11,
CODE_S02 local_ff3bd6e3, "Them co --s02-prep (Recommended)", "3 s (Recommended)".

Car (this script, UDP to the ESP32 of the car, the 14-byte packet of pc_sender.pack_packet, nothing else):
    python rpi5/s02_sender.py --ip 192.168.x.x --test A-L1 [--prep-s 3.0] [--allow-strafe]
One absolute 50 Hz grid from time.monotonic(): packet j at t_start + j/50.
    j = 0 .. P-1          zeros (preparation, P = prep_s * 50): run_active rises on the first one when the
                          motors are on (SWA off -> on before the command, C7b L77), the firmware resets
                          PID/LADRC, the position loop and psi0, holds still after 0.5 s and can run a 2 s
                          gyro bias window (AGV 5bde4c5 main.c 611, pos_ctrl.c 67-73, vel_ctrl.c 103-118,
                          imu_fusion.h 55)
    j = P + k, k < K      trajectory_pub.get_body_velocity(k / 50) of GROUP_A[test] (the HIL schedule),
                          K = ceil(T_cmd * 50) as pc_sender.grid_counts
    j = P + K .. +49      zeros: the 1 s stop of P2-2b (k = K .. K + 49)
Lines printed (PC clock, for diagnosis only: the master clock is the ESP32, C7b section 1):
    #s02_start,test=,t_epoch=,iso=,prep_s=,n_prep=,n_sched=,n_tail=,send_hz=,dst=
    #s02_k0,t_epoch=                     right after packet k = 0 is sent
    #s02_end,test=,t_epoch=,sent_prep=,sent_sched=,sent_tail=,complete=,sched_first_s=,sched_last_s=
      sched_first_s / sched_last_s: send time of packets k = 0 and k = K - 1 after t_start, read just after
      sendto (as pc_sender '#test_end'), empty when not sent
Ctrl-C in any phase: '#s02_interrupted,phase=prep|sched|tail', 10 zero packets at once, complete=0.
A-S1, A-S2 need --allow-strafe (S02 P2-7: after the S05 final check).

HIL: prep_spec() wraps the test in the 'load' kind of trajectory_pub (unchanged; it is in the S04 blob
lock), used by run_hil.py --s02-prep.
"""
import argparse
import math
import socket
import time

from pc_sender import SEND_HZ, TRAJ_PORT, pack_packet
from trajectory_pub import GROUP_A, TrajectorySpec, get_body_velocity

PREP_S_DEFAULT = 3.0          # Liam 15/11 CODE_S02 "3 s (Recommended)"
TAIL_ZEROS = 50               # 1 s of zeros after the schedule (P2-2b k = K .. K + 49)
STOP_ZEROS = 10               # Ctrl-C: as pc_sender
STRAFE_TESTS = ('A-S1', 'A-S2')


def prep_packets(prep_s: float) -> int:
    """Number of preparation packets; prep_s must be > 0 and a multiple of the 20 ms grid."""
    p = float(prep_s) * SEND_HZ
    n = int(round(p)) if math.isfinite(p) else 0
    if n < 1 or abs(p - n) > 1e-6:
        raise ValueError(f'--prep-s {prep_s}: must be > 0 and a multiple of {1 / SEND_HZ:g} s')
    return n


def check_test(test_id: str, allow_strafe: bool = False):
    """The GROUP_A spec of test_id, or ValueError (unknown test; A-S without allow_strafe)."""
    if test_id not in GROUP_A:
        raise ValueError(f'--test {test_id}: not a group A test {sorted(GROUP_A)}')
    if test_id in STRAFE_TESTS and not allow_strafe:
        raise ValueError(f'--test {test_id}: strafe tests run after the S05 final check (S02 P2-7); '
                         'add --allow-strafe')
    return GROUP_A[test_id]


def sched_count(spec) -> int:
    """K: schedule packets k with k/50 < T_cmd (pc_sender.grid_counts)."""
    return int(math.ceil(spec.duration * SEND_HZ - 1e-9))


def prep_spec(spec: TrajectorySpec, prep_s: float) -> TrajectorySpec:
    """HIL (run_hil.py --s02-prep): 0 in [0, prep_s), get_body_velocity(t - prep_s, spec) in
    [prep_s, prep_s + T_cmd), duration prep_s + T_cmd (trajectory_pub 'load' kind, lines 163-167)."""
    if spec.duration <= 0:
        raise ValueError('prep_spec needs a finite test')
    prep_packets(prep_s)
    return TrajectorySpec(kind='load', duration=prep_s + spec.duration,
                          sequence=((prep_s, prep_s + spec.duration, spec),))


def s02_loop(ip: str, test_id: str, prep_s: float = PREP_S_DEFAULT, allow_strafe: bool = False, sock=None,
             clock=time.monotonic, sleep=time.sleep, out=print, wall=time.time) -> dict:
    """Send preparation zeros, the schedule, then the tail zeros on one 50 Hz grid; returns the counts."""
    spec = check_test(test_id, allow_strafe)
    n_prep, n_sched = prep_packets(prep_s), sched_count(spec)
    own = sock is None
    sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (ip, TRAJ_PORT)
    t0, e0 = clock(), wall()
    out(f"#s02_start,test={test_id},t_epoch={e0:.3f},iso={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(e0))},"
        f"prep_s={prep_s:g},n_prep={n_prep},n_sched={n_sched},n_tail={TAIL_ZEROS},send_hz={SEND_HZ},"
        f"dst={ip}:{TRAJ_PORT}")

    def at(j):
        wait = t0 + j / SEND_HZ - clock()
        if wait > 0:
            sleep(wait)

    sent = {'prep': 0, 'sched': 0, 'tail': 0}
    first = last = None
    phase = 'prep'
    try:
        for j in range(n_prep):
            at(j)
            sock.sendto(pack_packet(0.0, 0.0, 0.0), dst)
            sent['prep'] += 1
        phase = 'sched'
        for k in range(n_sched):
            at(n_prep + k)
            vx, vy, wz = get_body_velocity(k / SEND_HZ, spec)
            sock.sendto(pack_packet(vx, vy, wz), dst)
            if k == 0 or k == n_sched - 1:
                t_sent = clock() - t0                 # one reading per packet, as pc_sender
                if k == 0:
                    first = t_sent
                    out(f'#s02_k0,t_epoch={wall():.3f}')
                if k == n_sched - 1:
                    last = t_sent
            sent['sched'] += 1
        phase = 'tail'
        for j in range(n_prep + n_sched, n_prep + n_sched + TAIL_ZEROS):
            at(j)
            sock.sendto(pack_packet(0.0, 0.0, 0.0), dst)
            sent['tail'] += 1
        phase = 'done'
    except KeyboardInterrupt:
        out(f'#s02_interrupted,phase={phase}')
    finally:
        if phase != 'done':
            for _ in range(STOP_ZEROS):
                sock.sendto(pack_packet(0.0, 0.0, 0.0), dst)
                sleep(0.02)
        complete = int(phase == 'done')
        out(f"#s02_end,test={test_id},t_epoch={wall():.3f},sent_prep={sent['prep']},sent_sched={sent['sched']},"
            f"sent_tail={sent['tail']},complete={complete},"
            f"sched_first_s={'' if first is None else f'{first:.6f}'},"
            f"sched_last_s={'' if last is None else f'{last:.6f}'}")
        if own:
            sock.close()
    return dict(sent, n_prep=n_prep, n_sched=n_sched, complete=complete)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', required=True, help='ESP32 IP address of the car')
    ap.add_argument('--test', required=True, choices=sorted(GROUP_A), help='spec S02 group A test')
    ap.add_argument('--prep-s', type=float, default=PREP_S_DEFAULT,
                    help='preparation zeros before k = 0 [s], multiple of 0.02 (S02 P2-2: 3 s)')
    ap.add_argument('--allow-strafe', action='store_true',
                    help='allow A-S1, A-S2 (S02 P2-7: only after the S05 final check)')
    a = ap.parse_args(argv)
    try:
        check_test(a.test, a.allow_strafe)
        prep_packets(a.prep_s)
    except ValueError as exc:
        ap.error(str(exc))
    r = s02_loop(a.ip, a.test, a.prep_s, a.allow_strafe)
    return 0 if r['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
