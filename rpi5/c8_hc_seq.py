#!/usr/bin/env python3
"""
c8_hc_seq.py: Pi command sequence for C8 step 12b (slow straight runs, axis calibration) and step E (turn in place),
one command per place, the logger run as a child process (CODE_HIEUCHUAN, 19/11/2026).

Design: CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_CODE_HIEUCHUAN_v4.md section 2 (Liam 19/11 at CODE_RASOAT:
the car is never driven by hand, the Pi sends the trajectory; at CODE_HIEUCHUAN: "Vong 3 hep roi code (Recommended)").

    python3 rpi5/c8_hc_seq.py --ip 192.168.1.69 --src-ip 192.168.1.69 --cho goc|K1|K2 --thu-muc DIR [--phan all|12b|E]
    python3 rpi5/c8_hc_seq.py --cho goc --in-lich          # print the schedule table only, nothing is sent

One process owns the attempt (design 2.3): refuse when DIR exists or another sender / logger runs; create DIR; start
tools/agv_logger.py --strict --src-ip .. --out DIR/log.csv in its own process group; wait for its first data row
(10 s) then 6 s still; gain question before (pc_sender.gain_query_file -> DIR/gain_query_truoc.txt; unreadable: no
schedule); the schedule; gain question after; 6 s; SIGINT to the logger only, wait for it to close (10 s).

Sending (design 2.3): sample k belongs to t0 + k/50 on a monotonic clock; each send uses the sample of the current
time (no burst of late samples; skipped samples are counted in n_bo). A gap between two sends above 100 ms, a socket
error, the logger dying, or Ctrl-C abort the schedule: zeros for 0.5 s, '#seq_huy', exit 3; an aborted schedule is
never resumed (a new attempt uses a new folder). The velocity of sample k is the mean velocity of the planned motion
over [k/50, (k+1)/50), so the packets integrate exactly to the planned distances and angles.

The schedule (design 2.1): 12b along the car axis (vx) 0.10 m/s, 0.25 m/s^2, stops 4 s:
  goc  0 -> +0.50 -> -0.50 -> +0.50 -> -0.50 -> +0.50 -> 0   (car nose at 45 deg on the floor)
  K1   0 -> +0.25 -> -0.25 -> +0.25 -> -0.25 -> +0.25 -> 0   (nose 90 deg; +-0.4 m could cover tags 29/30)
  K2   0 -> +0.15 -> -0.35 -> +0.15 -> -0.35 -> +0.15 -> 0   (nose 0 deg)
then E (goc, K1 only; always after 12b): +2.5 turns, stop, -2.5 turns, stop, wz 0.5 rad/s, 1.0 rad/s^2.
Output DIR/sender.txt ('#seq_start', '#seg' per segment at its first packet, '#seq_end'), DIR/lich.csv (its sha256 in
'#seq_start'), the logger files, logger_stdout.txt. complete=1 only says the SENDER sent every sample (n_bo = 0); what
the car received is checked by tools/c8_hc_a.py kiem.
Last line: #ket_qua,ma=0 sent whole + question after + logger summary closed / 2 refused before any motion
packet / 3 aborted or not usable (',ly_do=...').

Codex code review round 1 (HCC1-1..3, 19/11): no write between the clock check and sendto ('#seg' kept in memory),
sendto bounded (SENDTO_MAX_S), any error after sending started sends the zeros first, '#ket_qua' 0 only with
complete=1, question after 0 and a closed logger summary whose CSV sha256 matches; --src-ip must equal --ip; floor
lock (LOCK_PATH, fcntl) + pgrep before and after (pgrep failing refuses); 0.5 s before the first motion packet the
last 20 logger rows must show motor_on = 1, run_active = 1 (AUTO: SWC not in MANUAL), sp_* = 0, traj_age_ms <= 60.
Condition, not checked here: the car is not wired to the H7 (no UART trajectory source); this tool is the only UDP
sender (pgrep snapshot + lock; a sender started after the check that does not take the lock is not seen live).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from pc_sender import SEND_HZ, TRAJ_PORT, gain_query_file, pack_packet  # noqa: E402

VERSION = 'c8_hc_seq_v3'
DT = 1.0 / SEND_HZ
V_LIN, A_LIN = 0.10, 0.25          # m/s, m/s^2 [de xuat]
W_ROT, A_ROT = 0.5, 1.0            # rad/s, rad/s^2 [de xuat]
HOLD_S = 4.0                       # stop between segments (design 2.1, HC1-7)
E_TURN = 5.0 * math.pi             # 2.5 turns each way (HC1-6)
TAIL_S = 2.5                       # zeros after the schedule (as O1_TAIL_S)
ABORT_ZERO_S = 0.5                 # zeros sent on an abort
GAP_ABORT_S = 0.100                # a gap between two sends above this aborts (half the firmware 200 ms timeout)
SENDTO_MAX_S = 0.020               # one sendto longer than this aborts (HCC1-1); real socket timeout too
READY_LEAD_S = 0.1                 # the final car-ready check runs this long before the first motion packet (HCC3-1)
MIN_SPACING_S = 0.010              # DT/2 between the end of a sendto and the next one (HCC3-2)
BUDGET_MIN_S = 0.001               # less time left than this inside the 100 ms gap budget: abort, no packet (HCC4-1)
# residual (HCC5-1): the process can still be preempted between the last clock read and the sendto syscall; the
# gap is then checked right after the sendto (abort) and the firmware 200 ms timeout stays the last stop
READY_FRESH_S = 0.20               # newest log row at most this old at the first motion packet (HCC4-2)
READY_FRESH_NS = 200_000_000       # the same in integer ns (HCC6-6)
RAW_SCAN0 = 16384                  # first raw.jsonl window read backwards (HCC6-5), x4 up to RAW_SCAN_MAX
RAW_SCAN_MAX = 1 << 20
RAW_REORDER_SEQ = 50               # a row's first reception is after every record this many samples older (1 s)
READY_ROWS = 20                    # logger rows (~0.2 s at 10 ms) that must show a ready car
READY_PRIME_K = 25                 # first ready check 0.5 s before the final one; the log must advance between
READY_ADVANCE = 25                 # log_seq rows the stream must add between the two checks (~50 expected)
READY_AGE_MS = 60                  # traj_age_ms in those rows (zeros are being sent at 50 Hz)
LOCK_PATH = '/tmp/hil_tai_san.lock'  # shared floor lock (HCC1-3); tools that do not take it are seen by pgrep
WAIT_FIRST_ROW_S = 10.0            # logger must write a data row within this
STILL_BEFORE_S = 6.0               # still time after the first log row before the schedule (CODE 19/11)
STILL_AFTER_S = 6.0                # logger kept after the schedule
LOGGER_CLOSE_S = 10.0
V_MAX, W_MAX = 0.15, 0.6           # hard limits on the commanded values (tests)
LOGGER = os.path.join(HERE, '..', 'tools', 'agv_logger.py')

PLACES = {   # stops along the car axis [m], E after 12b
    'goc': ((0.0, 0.50, -0.50, 0.50, -0.50, 0.50, 0.0), True),
    'K1': ((0.0, 0.25, -0.25, 0.25, -0.25, 0.25, 0.0), True),
    'K2': ((0.0, 0.15, -0.35, 0.15, -0.35, 0.15, 0.0), False),
}
OTHERS = ('pc_sender.py', 'c8_hc_seq.py', 'agv_logger.py')


class SeqError(Exception):
    """Refused before anything was sent (exit 2)."""


# ---------------------------------------------------------------- schedule
def trap_time(dist: float, v: float, a: float) -> float:
    """Duration of a trapezoid (or triangle) move of |dist| with top speed v and acceleration a."""
    d = abs(dist)
    if d <= v * v / a:
        return 2.0 * math.sqrt(d / a)
    return d / v + v / a


def trap_pos(tau: float, dist: float, v: float, a: float) -> float:
    """Signed position at time tau of the move (0 before, dist after)."""
    T = trap_time(dist, v, a)
    if tau <= 0.0 or dist == 0.0:
        return 0.0
    if tau >= T:
        return dist
    d, sg = abs(dist), math.copysign(1.0, dist)
    if d <= v * v / a:                       # triangle
        vp = math.sqrt(d * a)
        tr = vp / a
    else:
        vp, tr = v, v / a
    if tau < tr:
        s = 0.5 * a * tau * tau
    elif tau > T - tr:
        r = T - tau
        s = d - 0.5 * a * r * r
    else:
        s = 0.5 * a * tr * tr + vp * (tau - tr)
    return sg * s


def schedule(cho: str, phan: str = 'all'):
    """List of segments: dict(i, loai 'dung'|'tien'|'xoay', t0, t1, truc '-'|'vx'|'wz', tu, toi, v, a)."""
    if cho not in PLACES:
        raise SeqError(f'--cho {cho}: one of {sorted(PLACES)}')
    stops, has_e = PLACES[cho]
    if phan not in ('all', '12b', 'E'):
        raise SeqError(f'--phan {phan}: all, 12b or E')
    if phan == 'E' and not has_e:
        raise SeqError(f'--phan E: no E at {cho}')
    segs, t = [], 0.0

    def add(loai, dur, truc='-', tu=0.0, toi=0.0, v=0.0, a=0.0):
        nonlocal t
        segs.append(dict(i=len(segs), loai=loai, t0=t, t1=t + dur, truc=truc, tu=tu, toi=toi, v=v, a=a))
        t += dur

    add('dung', HOLD_S)
    if phan in ('all', '12b'):
        for s0, s1 in zip(stops[:-1], stops[1:]):
            add('tien', trap_time(s1 - s0, V_LIN, A_LIN), 'vx', s0, s1, V_LIN, A_LIN)
            add('dung', HOLD_S, '-', s1, s1)
    if has_e and phan in ('all', 'E'):
        for g0, g1 in ((0.0, E_TURN), (E_TURN, 0.0)):
            add('xoay', trap_time(g1 - g0, W_ROT, A_ROT), 'wz', g0, g1, W_ROT, A_ROT)
            add('dung', HOLD_S, '-', g1, g1)
    return segs


def duration(segs) -> float:
    return segs[-1]['t1']


def plan_state(segs, t: float):
    """(x along the car axis [m], heading [rad]) planned at time t (segments are sequential, one axis each)."""
    x = th = 0.0
    for s in segs:
        if t <= s['t0']:
            break
        tau = min(t, s['t1']) - s['t0']
        if s['loai'] == 'tien':
            x = s['tu'] + trap_pos(tau, s['toi'] - s['tu'], s['v'], s['a'])
        elif s['loai'] == 'xoay':
            th = s['tu'] + trap_pos(tau, s['toi'] - s['tu'], s['v'], s['a'])
    return x, th


def sample(segs, k: int):
    """(vx, vy, wz) of packet k: the mean planned velocity over [k DT, (k+1) DT)."""
    x0, th0 = plan_state(segs, k * DT)
    x1, th1 = plan_state(segs, (k + 1) * DT)
    return (x1 - x0) / DT, 0.0, (th1 - th0) / DT


def n_samples(segs) -> int:
    return int(math.ceil(duration(segs) * SEND_HZ - 1e-9))


def lich_csv(segs) -> str:
    """The normalised schedule table (its sha256 goes into '#seq_start')."""
    b = io.StringIO()
    b.write('i,loai,t0_s,t1_s,truc,tu,toi,v,a\n')
    for s in segs:
        b.write(f"{s['i']},{s['loai']},{s['t0']:.6f},{s['t1']:.6f},{s['truc']},{s['tu']:.6f},{s['toi']:.6f},"
                f"{s['v']:.6f},{s['a']:.6f}\n")
    return b.getvalue()


def seg_at(segs, t: float) -> int:
    for s in segs:
        if s['t0'] <= t < s['t1']:
            return s['i']
    return len(segs) - 1


# ---------------------------------------------------------------- sending
class Abort(Exception):
    def __init__(self, why, **kv):
        super().__init__(why)
        self.why, self.kv = why, kv


def first_motion_k(segs) -> int:
    """Index of the first packet that is not (0, 0, 0)."""
    for k in range(n_samples(segs)):
        if sample(segs, k) != (0.0, 0.0, 0.0):
            return k
    return n_samples(segs)


def send_schedule(ip, segs, sock, out, clock=time.monotonic, sleep=time.sleep, wall=time.time,
                  alive=lambda: True, ready=None, cho='', phan='', thu_id='', lich_sha=''):
    """Send the schedule then TAIL_S of zeros on the 50 Hz grid; returns a dict of counters.

    Timing (HCC1-1): nothing that writes or prints runs between the clock check and sendto; the '#seg' lines are
    kept in memory and written by out() after the last packet (or after the abort zeros). The clock is read
    immediately before each sendto: a gap from the previous pre-send check above GAP_ABORT_S, or a sendto that
    takes longer than SENDTO_MAX_S, aborts. So two packets are never further apart than GAP_ABORT_S + SENDTO_MAX_S
    (120 ms, under the 200 ms firmware timeout), and a stall that jumps past the end of the schedule is checked too.
    ready() (HCC1-3, HCC2-1) is called twice, READY_LEAD_S + 0.5 s and READY_LEAD_S before the first motion packet
    (zeros are being sent): None or the reason the car is not ready -> abort before any motion. A sample lost
    before the first motion packet also aborts there (HCC3-1; exit 2: nothing moved). Sample k is sent anywhere in
    its 20 ms cell, but never closer than MIN_SPACING_S to the end of the previous sendto (HCC3-2); the reported
    gap is the upper bound (end of this sendto - check before the previous one) and aborts above GAP_ABORT_S.

    Any exception once sending has started (HCC1-2: Abort, a socket error, KeyboardInterrupt, an error writing
    the log) sends the abort zeros first, then tries to write '#seq_end' and '#seq_huy', then raises Abort."""
    n_plan = n_samples(segs)
    n_tail = int(math.ceil(TAIL_S * SEND_HZ - 1e-9))
    total = n_plan + n_tail
    k_move = first_motion_k(segs)
    k_ready = max(0, k_move - int(round(READY_LEAD_S * SEND_HZ)))
    e0 = wall()
    buf = [f"#seq_start,ver={VERSION},cho={cho},phan={phan},thu_id={thu_id},t_epoch={e0:.3f},"
           f"iso={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(e0))},duration_s={duration(segs):.3f},"
           f"send_hz={SEND_HZ},tail_s={TAIL_S},dst={ip}:{TRAJ_PORT},lich_sha256={lich_sha}"]
    st = dict(k=0, sent=0, skipped=0, gap_max=0.0, sendto_max=0.0, first_s=None, last_s=None, moved=False,
              primed=False, fresh_until=None)
    seg_done = set()
    ready_done = ready is None
    t0 = clock()
    last_chk = last_done = None

    def fresh():
        """HCC4-2, HCC6-3: ready.deadline() -> (deadline on the clock of `clock` (CLOCK_MONOTONIC on the Pi), age)
        or (None, reason); kept relative to t0 for the check before the first motion sendto."""
        fr = getattr(ready, 'deadline', None)
        if not callable(fr):
            st['fresh_until'] = math.inf
            return
        dl, info = fr()
        if dl is None:
            why = str(info).replace(',', ';')
            buf.append(f'#log_moi_truoc_goi_dong,ket_qua=hong,ly_do={why}')
            raise Abort('log_cu', k=st['k'], ly_do=why)
        st['fresh_until'] = dl - t0
        buf.append(f'#log_moi_truoc_goi_dong,cu_s={info:.3f},han_s={dl - t0:.6f}')
    try:
        while True:
            now = clock() - t0
            k_now = max(st['k'], int(math.floor(now * SEND_HZ + 1e-9)))
            if last_chk is not None and now - last_chk > GAP_ABORT_S:   # also a stall past the end (HCC1-1)
                st['gap_max'] = max(st['gap_max'], now - last_chk)
                raise Abort('tre_gui', k=k_now, gap_ms=f'{(now - last_chk) * 1000:.1f}')
            if k_now >= total:
                st['skipped'] += total - st['k']
                break
            if k_now > st['k'] and not st['moved']:
                # HCC3-1, HCC4-2: a sample lost before any motion packet (also a jump INTO the motion phase):
                # refuse before the first motion packet (code 2)
                raise Abort('tre_truoc_chay', k=k_now, bo=k_now - st['k'])
            st['skipped'] += k_now - st['k']
            st['k'] = k = k_now
            if not alive():
                raise Abort('logger_chet', k=k)
            if not st['primed'] and ready is not None and k_ready >= READY_PRIME_K and k >= k_ready - READY_PRIME_K:
                st['primed'] = True
                why = ready()
                buf.append(f"#kiem_truoc_chay,k={k},lan=1,ket_qua={'dat' if why is None else 'hong'}"
                           + ('' if why is None else ',ly_do=' + str(why).replace(',', ';')))
                if why is not None:
                    raise Abort('xe_chua_san_sang', k=k)
                continue                                  # the check took time: read the clock again
            if not ready_done and k >= k_ready:
                ready_done = True
                why = ready()
                buf.append(f"#kiem_truoc_chay,k={k},ket_qua={'dat' if why is None else 'hong'}"
                           + ('' if why is None else ',ly_do=' + str(why).replace(',', ';')))
                if why is not None:
                    raise Abort('xe_chua_san_sang', k=k)
                continue
            if k < n_plan:
                vx, vy, wz = sample(segs, k)
                i = seg_at(segs, k * DT)
                if i not in seg_done:
                    seg_done.add(i)
                    s = segs[i]
                    buf.append(f"#seg,i={i},loai={s['loai']},t_plan_s={s['t0']:.3f},t_sent_s={now:.6f},"
                               f"t_epoch={wall():.3f},truc={s['truc']},tu={s['tu']:.6f},toi={s['toi']:.6f},"
                               f"v={s['v']:g},a={s['a']:g}")
            else:
                vx = vy = wz = 0.0
            pkt = pack_packet(vx, vy, wz)
            if k >= k_move and not st['moved'] and st['fresh_until'] is None:
                # fallback only (k_move = 0): the freshness is normally read right after the sendto of k_move - 1
                fresh()
                continue                                  # read the clock again after the file read
            # HCC3-2: never closer than DT/2 to the END of the previous sendto (a packet leaves at the latest when
            # sendto returns), so two packets cannot go out together even when a sendto blocked
            if last_done is not None:
                w = last_done + MIN_SPACING_S - (clock() - t0)
                if w > 0:
                    sleep(w)
                    continue
            chk = clock() - t0                           # immediately before sendto
            if chk >= (k + 1) * DT:                      # the cell of sample k is over: choose again (HCC2-1)
                continue
            if last_chk is not None and GAP_ABORT_S - (chk - last_chk) < BUDGET_MIN_S:
                # HCC4-1, HCC5-1: not enough time left inside 100 ms of the previous packet: zeros, never motion.
                # The socket is non-blocking (run_attempt), so nothing that can block runs between this last clock
                # read and the sendto; the bound is checked again from the clock after the sendto
                st['gap_max'] = max(st['gap_max'], chk - last_chk)
                raise Abort('tre_gui', k=k, gap_ms=f'{(chk - last_chk) * 1000:.1f}')
            if k >= k_move and not st['moved']:
                if chk > st['fresh_until']:
                    # HCC6-3: the absolute freshness deadline, checked at the last clock read before the sendto
                    raise Abort('log_cu', k=k, qua_han_ms=f"{(chk - st['fresh_until']) * 1000:.1f}")
                st['moved'] = True                       # HCC4-3: set BEFORE the first motion sendto
            try:
                sock.sendto(pkt, (ip, TRAJ_PORT))
            except OSError as exc:
                raise Abort('socket', k=k, loi=str(exc).replace(',', ';'))
            done = clock() - t0
            st['sendto_max'] = max(st['sendto_max'], done - chk)
            if done - chk > SENDTO_MAX_S:
                raise Abort('sendto_cham', k=k, ms=f'{(done - chk) * 1000:.1f}')
            if last_chk is not None:
                ub = done - last_chk                     # upper bound of the gap between the two packets
                st['gap_max'] = max(st['gap_max'], ub)
                if ub > GAP_ABORT_S:
                    raise Abort('tre_gui', k=k, gap_ms=f'{ub * 1000:.1f}')
            last_chk, last_done = chk, done
            if k == 0:
                st['first_s'] = chk
            if k == n_plan - 1:
                st['last_s'] = chk
            st['sent'] += 1
            st['k'] = k + 1
            if st['k'] >= total:
                break
            if st['k'] == k_move and not st['moved'] and st['fresh_until'] is None:
                # HCC6-3: read the freshness in the wait after the last zero, not inside the cell of the first
                # motion packet; the deadline it gives is checked again just before that sendto
                fresh()
            wait = t0 + st['k'] * DT - clock()
            if wait > 0:
                sleep(wait)
    except BaseException as exc:                         # noqa: B902  (HCC1-2: zeros before any write)
        ab = exc if isinstance(exc, Abort) else Abort(
            'ctrl_c' if isinstance(exc, KeyboardInterrupt) else 'loi_' + type(exc).__name__, k=st['k'])
        ab.moved = st['moved']                           # no motion packet sent yet -> the attempt is refused (2)
        n0 = _abort_zeros(ip, sock, sleep)
        ab.kv['goi0'] = f'{n0}/{int(round(ABORT_ZERO_S * SEND_HZ))}'   # 0/25: the stop relies on the firmware timeout
        _flush(out, buf)
        _end(out, wall, st, total, complete=0)
        out('#seq_huy,ly_do=' + ab.why + ''.join(f',{a}={b}' for a, b in ab.kv.items()))
        raise ab from (None if exc is ab else exc)
    complete = int(st['sent'] == total and st['skipped'] == 0)
    _flush(out, buf)
    _end(out, wall, st, total, complete)
    return dict(sent=st['sent'], planned=total, n_bo=st['skipped'], gap_max_ms=st['gap_max'] * 1000,
                complete=complete)


def _flush(out, buf):
    """Write the lines kept in memory during sending; an error here is reported after the zeros were sent."""
    while buf:
        try:
            out(buf.pop(0))
        except Exception:                                 # noqa: BLE001  the zeros are already out
            buf.clear()
            raise


def _abort_zeros(ip, sock, sleep):
    """ABORT_ZERO_S of zero packets; nothing here may stop the zeros (Ctrl-C again, socket errors). The socket gets
    its own blocking budget SENDTO_MAX_S per packet (HCC5-2: not the motion settings); returns how many left.
    HCC6-4: SIGINT is ignored while the zeros go out (main thread; restored after), and a KeyboardInterrupt that
    still lands in a step (before the handler is set, or off the main thread) only skips that step. settimeout()
    alone sets blocking with a timeout in one call; a failed setting still tries every zero."""
    unset = object()
    old = unset
    try:
        old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError, KeyboardInterrupt):      # not the main thread / Ctrl-C before the handler was set
        pass
    n_ok = 0
    try:
        try:
            sock.settimeout(SENDTO_MAX_S)
        except (AttributeError, OSError, ValueError, KeyboardInterrupt):
            pass
        for _ in range(int(round(ABORT_ZERO_S * SEND_HZ))):
            try:
                sock.sendto(pack_packet(0.0, 0.0, 0.0), (ip, TRAJ_PORT))
                n_ok += 1
            except (OSError, ValueError, KeyboardInterrupt):
                pass
            try:
                sleep(DT)
            except KeyboardInterrupt:
                pass
    finally:
        if old is not unset:
            try:
                signal.signal(signal.SIGINT, signal.default_int_handler if old is None else old)
            except (ValueError, OSError):
                pass
    return n_ok


def _end(out, wall, st, total, complete):
    f = lambda v: '' if v is None else f'{v:.6f}'           # noqa: E731
    out(f"#seq_end,t_epoch={wall():.3f},sent={st['sent']},planned={total},n_bo={st['skipped']},"
        f"gap_max_ms={st['gap_max'] * 1000:.1f},sendto_max_ms={st['sendto_max'] * 1000:.1f},"
        f"complete={complete},sched_first_s={f(st['first_s'])},sched_last_s={f(st['last_s'])}")


# ---------------------------------------------------------------- one attempt
def _pgrep_default():
    """pgrep -af python: its text; exit 1 = no process (''), any other failure raises SeqError (HCC1-3)."""
    try:
        p = subprocess.run(['pgrep', '-af', 'python'], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SeqError(f'pgrep could not run ({exc}): cannot prove no other sender/logger runs')
    if p.returncode not in (0, 1):
        raise SeqError(f'pgrep exit {p.returncode}: cannot prove no other sender/logger runs')
    return p.stdout if p.returncode == 0 else ''


def others_running(pgrep=None, me=None, exclude=()):
    """Names of other sender / logger processes (pgrep -af). pgrep missing or failing refuses (SeqError), it is
    never read as 'nothing runs' (HCC1-3). A snapshot: tools started later are not seen (the floor lock covers
    other c8_hc_seq runs; the check is repeated after the schedule)."""
    me = os.getpid() if me is None else me
    txt = (pgrep or _pgrep_default)()
    found = []
    for line in (txt or '').splitlines():
        parts = line.split(None, 1)
        if not parts or not parts[0].isdigit() or int(parts[0]) == me or int(parts[0]) in exclude:
            continue
        cmd = parts[1] if len(parts) > 1 else ''
        found += [f'{parts[0]}:{n}' for n in OTHERS if n in cmd]
    return found


def data_rows(path) -> int:
    """Data rows of the logger CSV so far (header and '#' lines excluded)."""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            rows = [ln for ln in f if ln.strip() and not ln.startswith('#')]
    except FileNotFoundError:
        return 0
    return max(0, len(rows) - 1)


def _tail_rows(path, nbytes=16384):
    """(header, last data rows) of the logger CSV: header from the first line, rows from the last nbytes."""
    with open(path, 'rb') as f:
        head = f.readline().decode('utf-8', 'replace').strip()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        tail = f.read().decode('utf-8', 'replace').splitlines()
    if size > nbytes:
        tail = tail[1:]                                  # first line may be cut
    rows = [ln.strip() for ln in tail if ln.strip() and not ln.startswith('#') and ln.strip() != head]
    return head.split(','), rows


def car_ready(csv_path, n=READY_ROWS):
    """None when the last n logger rows show the car ready for motion (HCC1-3): motor_on = 1 (not the 10 Hz
    motor-off line), run_active = 1 (MODE_AUTO with fresh packets: SWC not in MANUAL), sp_vx = sp_vy = sp_wz = 0
    (only our zeros are applied), 0 <= traj_age_ms <= READY_AGE_MS; else the reason."""
    try:
        cols, rows = _tail_rows(csv_path)
    except OSError as exc:
        return f'log: {exc}'
    need = ('motor_on', 'run_active', 'sp_vx', 'sp_vy', 'sp_wz', 'traj_age_ms')
    if any(c not in cols for c in need):
        return 'log thieu cot ' + '/'.join(c for c in need if c not in cols)
    if len(rows) < n:
        return f'chi {len(rows)} dong log'
    ix = {c: cols.index(c) for c in need}
    for ln in rows[-n:]:
        f = ln.split(',')
        try:
            v = {c: float(f[ix[c]]) for c in need}
        except (ValueError, IndexError):
            return 'dong log hong'
        if v['motor_on'] != 1:
            return 'motor_on=0'
        if v['run_active'] != 1:
            return 'run_active=0 (MANUAL/SWC hay khong nhan goi)'
        if v['sp_vx'] != 0 or v['sp_vy'] != 0 or v['sp_wz'] != 0:
            return 'sp_khac_0 (co nguon lenh khac)'
        if not 0 <= v['traj_age_ms'] <= READY_AGE_MS:
            return f"traj_age_ms={v['traj_age_ms']:g}"
    return None


def _int(v):
    """A CSV / payload cell as an int ('12' or '12.0'); ValueError otherwise."""
    try:
        return int(v)
    except ValueError:
        return int(float(v))


class ReadyCheck:
    """car_ready() called twice before the first motion packet (HCC2-1), plus the freshness of the data stream on
    the logger's own receive time (HCC5-3). HCC6-1, HCC6-2: the sample is the newest row the logger ACCEPTED (its
    CSV: rows from another source, multi-line or badly typed datagrams never get there), and its time is the FIRST
    reception of that row (log_seq and timestamp_us) in <csv>.raw.jsonl (t_mono_ns = time.monotonic_ns() at
    reception, same CLOCK_MONOTONIC as this process on the Pi), so a datagram that arrives again later does not make
    a stopped stream fresh. HCC6-5: the raw file is scanned backwards in growing windows until a data record more
    than RAW_REORDER_SEQ samples older (the first reception cannot be before it), up to RAW_SCAN_MAX; out of
    window is its own reason. HCC6-6: ages in integer ns, 0 <= age <= READY_FRESH_NS (a receive time in the future
    is refused). No background thread (HCC5-5)."""
    def __init__(self, csv_path, clock=time.time, mono_ns=time.monotonic_ns):
        self.csv, self.raw, self.clock, self.mono_ns = csv_path, csv_path + '.raw.jsonl', clock, mono_ns
        self.seq = None

    def _first_rx_ns(self, seq, ts, ncols, ix_s, ix_t):
        """t_mono_ns of the first reception of the data row (seq, ts), or the reason (str)."""
        nb = RAW_SCAN0
        while True:
            try:
                with open(self.raw, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - nb))
                    buf = f.read()
            except OSError as exc:
                return f'raw.jsonl: {exc}'
            cut = size > nb
            lines = buf.split(b'\n')
            lines = lines[1:-1] if cut else lines[:-1]            # a cut first line, the unfinished tail
            best, stop = None, False
            for ln in reversed(lines):
                try:
                    r = json.loads(ln)
                    t = int(r['t_mono_ns'])
                    txt = base64.b64decode(r['b64']).decode('utf-8')
                except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                    continue
                txt = txt.rstrip('\r\n')
                if '\n' in txt or '\r' in txt or txt.startswith('#'):
                    continue
                g = txt.split(',')
                if len(g) != ncols:
                    continue
                try:
                    s, u = _int(g[ix_s]), _int(g[ix_t])
                except ValueError:
                    continue
                if s == seq and u == ts:
                    best = t if best is None else min(best, t)
                elif s < seq - RAW_REORDER_SEQ:
                    stop = True
                    break
            if stop or not cut:
                return best if best is not None else f'raw.jsonl: khong thay goi cua log_seq {seq}'
            if nb >= RAW_SCAN_MAX:
                return f'raw.jsonl: het gioi han quet {RAW_SCAN_MAX} B'
            nb = min(nb * 4, RAW_SCAN_MAX)

    def newest(self):
        """(t_rx_ns, log_seq) of the newest accepted data row at its first reception, or the reason (str). The raw
        record is written before the CSV row, but the files are flushed one after the other: when the last row has
        no raw record yet, the row before it is used (older, so never fresher than the truth)."""
        try:
            cols, rows = _tail_rows(self.csv)
            ix_s, ix_t = cols.index('log_seq'), cols.index('timestamp_us')
        except OSError as exc:
            return f'log: {exc}'
        except ValueError:
            return 'log thieu cot log_seq/timestamp_us'
        if not rows:
            return 'log: chua co dong du lieu'
        why = 'log: chua co dong du lieu'
        for ln in rows[-1:-3:-1]:
            f_ = ln.split(',')
            try:
                seq, ts = _int(f_[ix_s]), _int(f_[ix_t])
            except (ValueError, IndexError):
                return 'dong log hong'
            t = self._first_rx_ns(seq, ts, len(cols), ix_s, ix_t)
            if not isinstance(t, str):
                return t, seq
            why = t
        return why

    def deadline(self):
        """(deadline [s, CLOCK_MONOTONIC], age [s]) for the first motion packet: the newest accepted row must be at
        most READY_FRESH_S old when that packet leaves; or (None, reason). log_seq must not go back from the last
        ready check."""
        n = self.newest()
        if isinstance(n, str):
            return None, n
        t_rx, seq = n
        if self.seq is not None and seq < self.seq:
            return None, f'log_seq lui {self.seq}->{seq}'
        age = self.mono_ns() - t_rx
        if not 0 <= age <= READY_FRESH_NS:
            return None, f'tuoi log {age / 1e9:.3f} s (gio nhan cua logger)'
        return (t_rx + READY_FRESH_NS) * 1e-9, age * 1e-9

    def close(self):
        pass

    def __call__(self):
        why = car_ready(self.csv)
        if why is not None:
            return why
        n = self.newest()
        if isinstance(n, str):
            return n
        age = self.mono_ns() - n[0]
        if not 0 <= age <= READY_FRESH_NS:
            return f'log dung {age / 1e9:.3f} s (gio nhan cua logger)'
        prev, self.seq = self.seq, n[1]
        if prev is not None and n[1] < prev:
            return f'log_seq lui {prev}->{n[1]}'
        if prev is not None and n[1] - prev < READY_ADVANCE:
            return f'log_seq chi tang {n[1] - prev}'
        return None


def floor_lock(path=LOCK_PATH):
    """Exclusive non-blocking lock shared by the floor tools that take it (HCC1-3); None where fcntl is missing
    (Windows tests). Raises SeqError when another holder has it."""
    try:
        import fcntl
    except ImportError:
        return None
    f = open(path, 'a+')
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SeqError(f'floor lock {path} held by another run')
    return f


def summary_check(csv_path):
    """(ok, text) of the logger summary (HCC1-2): status closed and the CSV sha256 of its manifest equals the
    file on disk."""
    sp = csv_path + '.summary.json'
    try:
        import json
        with open(sp, encoding='utf-8') as f:
            s = json.load(f)
        h = hashlib.sha256()
        with open(csv_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    except (OSError, ValueError) as exc:
        return False, f'summary: {type(exc).__name__}'
    man = (s.get('files') or {}).get('csv') or {}
    ok = s.get('status') == 'closed' and man.get('sha256') == h.hexdigest()
    return ok, f"status={s.get('status')},csv_sha256={h.hexdigest()},khop={int(man.get('sha256') == h.hexdigest())}"


def start_logger(d, src_ip, popen=subprocess.Popen, logger=None, extra=()):
    out = open(os.path.join(d, 'logger_stdout.txt'), 'x', encoding='utf-8')
    kw = dict(stdout=out, stderr=subprocess.STDOUT)
    if os.name == 'posix':
        kw['start_new_session'] = True          # Ctrl-C at the terminal does not reach the logger (HC3-2)
    else:
        kw['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    try:
        p = popen([sys.executable, logger or LOGGER, '--strict', '--src-ip', src_ip, '--out', os.path.join(d, 'log.csv'),
                   *extra], **kw)
    except BaseException:
        out.close()
        raise
    return p, out


def stop_logger(p, timeout=LOGGER_CLOSE_S):
    if p.poll() is not None:
        return p.returncode
    try:
        p.send_signal(signal.SIGINT if os.name == 'posix' else signal.CTRL_BREAK_EVENT)
    except (OSError, ValueError, AttributeError):
        p.terminate()
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        return 'het_han'


def run_attempt(a, out=print, clock=time.monotonic, sleep=time.sleep, wall=time.time, sock=None, popen=subprocess.Popen,
                query=gain_query_file, pgrep=None, ready=None, lock=floor_lock, logger=None, logger_extra=()) -> int:
    """One attempt (design 2.3). Exit 0 only when the schedule was sent whole (complete=1), the gain question
    after the schedule answered (exit 0), no other sender/logger appeared, and the logger exited 0 with a
    closed summary whose CSV sha256 matches the file (HCC1-2); 2 refused before any motion packet; 3 aborted or
    the attempt is not usable (reasons in '#ket_qua')."""
    segs = schedule(a.cho, a.phan)
    if a.src_ip != a.ip:                                  # HCC1-3: logger, questions and packets: one ESP32
        raise SeqError(f'--src-ip {a.src_ip} != --ip {a.ip}: logger, gain questions and packets must be one ESP32')
    d = a.thu_muc
    if os.path.lexists(d):
        raise SeqError(f'--thu-muc {d} already exists: use a new folder for each attempt')
    held = lock()
    try:
        busy = others_running(pgrep)
        if busy:
            raise SeqError(f'other sender/logger processes running: {busy}')
        os.makedirs(d)
        return _attempt(a, segs, d, out, clock, sleep, wall, sock, popen, query, pgrep, ready, logger, logger_extra)
    finally:
        if held is not None:
            held.close()


def _attempt(a, segs, d, out, clock, sleep, wall, sock, popen, query, pgrep, ready, logger, logger_extra):
    thu_id = os.path.basename(os.path.normpath(d))
    lich = lich_csv(segs)
    with open(os.path.join(d, 'lich.csv'), 'x', encoding='utf-8', newline='\n') as f:
        f.write(lich)
    lich_sha = hashlib.sha256(lich.encode('utf-8')).hexdigest()
    txt = open(os.path.join(d, 'sender.txt'), 'x', encoding='utf-8')

    def say(line):
        txt.write(line + '\n')
        txt.flush()
        try:
            out(line)
        except (OSError, ValueError):                     # terminal gone: the file is the record
            pass

    csv_path = os.path.join(d, 'log.csv')
    proc = lout = None
    code, why, moved = 0, [], False
    try:
        proc, lout = start_logger(d, a.src_ip, popen, logger, logger_extra)
        t_start = clock()
        while data_rows(csv_path) < 1:
            if proc.poll() is not None:
                raise SeqError(f'logger exited ({proc.returncode}) before its first data row')
            if clock() - t_start > WAIT_FIRST_ROW_S:
                raise SeqError(f'no data row in {csv_path} after {WAIT_FIRST_ROW_S:g} s')
            sleep(0.2)
        say(f'#log_dong_dau,t_epoch={wall():.3f}')
        sleep(STILL_BEFORE_S)
        qcode = query(a.ip, os.path.join(d, 'gain_query_truoc.txt'))
        say(f'#gain_query,when=truoc,exit={qcode}')
        if qcode != 0:
            raise SeqError('gain question before the schedule failed (gain_query_truoc.txt): nothing sent')
        own = sock is None
        s = sock
        if own:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setblocking(False)                          # HCC5-1: a sendto never blocks (full buffer raises)
        rd = ready or ReadyCheck(csv_path)
        own_rd = ready is None
        try:
            moved = True
            r = send_schedule(a.ip, segs, s, say, clock, sleep, wall, alive=lambda: proc.poll() is None,
                              ready=rd, cho=a.cho, phan=a.phan, thu_id=thu_id, lich_sha=lich_sha)
            if not r['complete']:
                code, why = 3, why + ['complete=0']
        except Abort as exc:
            code, why = (3 if getattr(exc, 'moved', True) else 2), why + ['huy_' + exc.why]
        finally:
            if own:
                s.close()
            if own_rd:
                rd.close()
        qs = query(a.ip, os.path.join(d, 'gain_query_sau.txt'))
        say(f'#gain_query,when=sau,exit={qs}')
        if qs != 0:
            code, why = max(code, 3), why + ['gain_sau']
        busy = others_running(pgrep, exclude=(getattr(proc, 'pid', None),))
        say(f"#pgrep_sau,khac={'/'.join(busy) or '-'}")
        if busy:
            code, why = max(code, 3), why + ['tien_trinh_khac']
        sleep(STILL_AFTER_S)
    except SeqError as exc:
        say(f'#tu_choi,ly_do={str(exc).replace(",", ";")}')
        code, why = (3 if moved else 2), why + ['tu_choi']
    except BaseException as exc:                          # noqa: B902  Ctrl-C, a write error: still close the logger
        if moved:
            code, why = 3, why + ['loi_' + type(exc).__name__]
        else:
            code, why = max(code, 2), why + ['loi_' + type(exc).__name__]
        try:
            say(f'#seq_huy,ly_do=ngoai_lich_{type(exc).__name__}')
        except Exception:                                 # noqa: BLE001
            pass
    finally:
        rc = stop_logger(proc) if proc is not None else 'khong_chay'
        if lout is not None:
            lout.close()
        ok, stxt = summary_check(csv_path) if proc is not None else (False, 'khong_chay')
        say(f'#logger_dong,ma={rc},{stxt}')
        if code == 0 and (rc != 0 or not ok):
            code, why = 3, why + ['logger']
        say(f"#ket_qua,ma={code}" + (f",ly_do={'/'.join(why)}" if why else ''))
        txt.close()
    return code


def _hangup_as_interrupt():
    """SIGHUP (the SSH session drops) and SIGTERM take the same path as Ctrl-C: zeros, question after, logger
    closed (HCC1-2). The firmware 200 ms timeout stays the last stop if the process is killed outright."""
    def h(sig, frame):
        raise KeyboardInterrupt
    for name in ('SIGHUP', 'SIGTERM'):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), h)
            except (OSError, ValueError):
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default=None, help='ESP32 of the car')
    ap.add_argument('--src-ip', default=None, help='logger --src-ip (the car)')
    ap.add_argument('--cho', required=True, choices=sorted(PLACES))
    ap.add_argument('--phan', default='all', choices=['all', '12b', 'E'])
    ap.add_argument('--thu-muc', default=None, help='new folder of this attempt')
    ap.add_argument('--in-lich', action='store_true', help='print the schedule table only')
    a = ap.parse_args(argv)
    try:
        if a.in_lich:
            segs = schedule(a.cho, a.phan)
            sys.stdout.write(lich_csv(segs))
            print(f'#tong_s,{duration(segs):.3f},goi,{n_samples(segs)}')
            print('#ket_qua,ma=0')
            return 0
        if not (a.ip and a.src_ip and a.thu_muc):
            ap.error('--ip, --src-ip and --thu-muc are needed to send')
        _hangup_as_interrupt()
        return run_attempt(a)
    except SeqError as exc:
        print(f'#tu_choi,ly_do={exc}')
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    sys.exit(main())
