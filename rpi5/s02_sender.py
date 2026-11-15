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
    #s02_end,test=,t_epoch=,sent_prep=,sent_sched=,sent_tail=,complete=,sched_first_s=,sched_last_s=,late_max_ms=,
            late_after_n=
      sched_first_s / sched_last_s: send time of packets k = 0 and k = K - 1 after t_start, read just after
      sendto (as pc_sender '#test_end'), empty when not sent
Ctrl-C in any phase: '#s02_interrupted,phase=start|prep|sched|tail', 10 zero packets at once, complete=0.
Deadline (S2P1-6, S2P2-5 Codex round 2): checked right BEFORE each sendto; a packet whose slot is more than 50 ms
past is not sent: '#s02_late,phase=,packet=,late_ms=', the same stop, complete=0. The clock read after sendto is
an upper bound of the send time (the process can be stopped between sendto and that read): it goes into
late_max_ms, and a schedule packet over 50 ms there counts in late_after_n (the packet may have left late, or
not); the run is not stopped for it: the analysis reads late_after_n.
HIL (run_hil.py --s02-prep): the 1 s of zeros after the schedule is inside RUN (prep_spec); the bench's own
STOPPING (1 s of zeros) comes after it. STOPPING is the bench lifecycle, not the S02 tail: the S02 tail is the
publisher's 1 s, the metrics stop at T_cmd (S2P2-6, Codex round 2; Liam 15/11 CODE_S02 "Chi ghi ro").
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
TAIL_S = TAIL_ZEROS / SEND_HZ  # 1 s of zeros (HIL: inside the publisher, prep_spec)
LATE_MAX_S = 0.050            # [ĐỀ XUẤT] a packet later than this after its slot stops the run (S2P1-6)
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


def prep_spec(spec: TrajectorySpec, prep_s: float, tail_s: float = TAIL_S) -> TrajectorySpec:
    """HIL (run_hil.py --s02-prep): 0 in [0, prep_s), get_body_velocity(t - prep_s, spec) in
    [prep_s, prep_s + T_cmd), 0 in [prep_s + T_cmd, prep_s + T_cmd + tail_s); duration all three
    (trajectory_pub 'load' kind, lines 163-167). The tail zeros are sent by the publisher on its own clock,
    so the command falls to 0 at the end of the schedule, not when the run owner sees the publisher done
    (S2P1-7, Codex round 1: STOPPING could start up to one 0.1 s owner beat late). SIM passes tail_s=0
    (fwsim_groupA_sim adds its own 1 s)."""
    if spec.duration <= 0:
        raise ValueError('prep_spec needs a finite test')
    if not (math.isfinite(tail_s) and tail_s >= 0):
        raise ValueError(f'tail_s {tail_s}: must be >= 0')
    prep_packets(prep_s)
    return TrajectorySpec(kind='load', duration=prep_s + spec.duration + tail_s,
                          sequence=((prep_s, prep_s + spec.duration, spec),))


class _Late(Exception):
    pass


def s02_loop(ip: str, test_id: str, prep_s: float = PREP_S_DEFAULT, allow_strafe: bool = False, sock=None,
             clock=time.monotonic, sleep=time.sleep, out=print, wall=time.time) -> dict:
    """Send preparation zeros, the schedule, then the tail zeros on one 50 Hz grid; returns the counts.
    S2P1-6 (Codex round 1): the grid starts right before the first packet (after the start line), and a
    packet sent more than LATE_MAX_S after its slot stops the run (a stall would send the late packets in a
    burst, e.g. the whole preparation at once): '#s02_late', 10 zeros, complete=0. S2P1-12: a counter
    steps right after its sendto, and the start line is inside the interrupt handling."""
    spec = check_test(test_id, allow_strafe)
    n_prep, n_sched = prep_packets(prep_s), sched_count(spec)
    own = sock is None
    sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (ip, TRAJ_PORT)
    sent = {'prep': 0, 'sched': 0, 'tail': 0}
    first = last = None
    phase, t0 = 'start', None
    late_max = 0.0
    late_after = 0

    def send(j, vx, vy, wz, key):
        nonlocal late_max, late_after
        slot = t0 + j / SEND_HZ
        wait = slot - clock()
        if wait > 0:
            sleep(wait)
        pre = clock() - slot
        if pre > LATE_MAX_S:                       # S2P2-5: a packet past its deadline is never sent
            raise _Late(j, pre)
        sock.sendto(pack_packet(vx, vy, wz), dst)
        sent[key] += 1
        late = clock() - slot                      # upper bound of the send time
        late_max = max(late_max, late)
        if late > LATE_MAX_S and key == 'sched':
            late_after += 1                        # S2P2-5: reported, not a stop (it may have left on time)
        return late

    try:
        e0 = wall()
        out(f"#s02_start,test={test_id},t_epoch={e0:.3f},iso={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(e0))},"
            f"prep_s={prep_s:g},n_prep={n_prep},n_sched={n_sched},n_tail={TAIL_ZEROS},send_hz={SEND_HZ},"
            f"dst={ip}:{TRAJ_PORT}")
        t0 = clock()
        phase = 'prep'
        for j in range(n_prep):
            send(j, 0.0, 0.0, 0.0, 'prep')
        phase = 'sched'
        for k in range(n_sched):
            vx, vy, wz = get_body_velocity(k / SEND_HZ, spec)
            late = send(n_prep + k, vx, vy, wz, 'sched')
            if k == 0:
                first = n_prep / SEND_HZ + late           # send time after t0, as pc_sender sched_first_s
                out(f'#s02_k0,t_epoch={wall():.3f}')
            if k == n_sched - 1:
                last = (n_prep + k) / SEND_HZ + late
        phase = 'tail'
        for j in range(n_prep + n_sched, n_prep + n_sched + TAIL_ZEROS):
            send(j, 0.0, 0.0, 0.0, 'tail')
        phase = 'done'
    except KeyboardInterrupt:
        out(f'#s02_interrupted,phase={phase}')
    except _Late as exc:
        j, late = exc.args
        out(f'#s02_late,phase={phase},packet={j},late_ms={late * 1e3:.1f}')
    finally:
        if phase != 'done':
            for _ in range(STOP_ZEROS):
                sock.sendto(pack_packet(0.0, 0.0, 0.0), dst)
                sleep(0.02)
        complete = int(phase == 'done')
        out(f"#s02_end,test={test_id},t_epoch={wall():.3f},sent_prep={sent['prep']},sent_sched={sent['sched']},"
            f"sent_tail={sent['tail']},complete={complete},"
            f"sched_first_s={'' if first is None else f'{first:.6f}'},"
            f"sched_last_s={'' if last is None else f'{last:.6f}'},late_max_ms={late_max * 1e3:.1f},"
            f"late_after_n={late_after}")
        if own:
            sock.close()
    return dict(sent, n_prep=n_prep, n_sched=n_sched, complete=complete, late_after_n=late_after)


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
