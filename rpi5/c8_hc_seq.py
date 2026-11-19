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
Last line: #ket_qua,ma=0 sent / 2 refused before sending / 3 aborted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
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


def send_schedule(ip, segs, sock, out, clock=time.monotonic, sleep=time.sleep, wall=time.time,
                  alive=lambda: True, cho='', phan='', thu_id='', lich_sha=''):
    """Send the schedule then TAIL_S of zeros on the 50 Hz grid; returns a dict of counters. Raises Abort (after the
    abort zeros) on a gap > GAP_ABORT_S, a socket error, or alive() false; KeyboardInterrupt is turned into Abort."""
    n_plan = n_samples(segs)
    n_tail = int(math.ceil(TAIL_S * SEND_HZ - 1e-9))
    e0 = wall()
    out(f"#seq_start,ver={VERSION},cho={cho},phan={phan},thu_id={thu_id},t_epoch={e0:.3f},"
        f"iso={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(e0))},duration_s={duration(segs):.3f},"
        f"send_hz={SEND_HZ},tail_s={TAIL_S},dst={ip}:{TRAJ_PORT},lich_sha256={lich_sha}")
    t0 = clock()
    last_send, k, sent, skipped, gap_max = None, 0, 0, 0, 0.0
    seg_done = set()
    first_s = last_s = None
    total = n_plan + n_tail
    try:
        while k < total:
            now = clock() - t0
            k_now = max(k, int(math.floor(now * SEND_HZ + 1e-9)))
            if k_now >= total:
                break
            if k_now > k:
                skipped += k_now - k
            k = k_now
            if last_send is not None:
                gap = now - last_send
                gap_max = max(gap_max, gap)
                if gap > GAP_ABORT_S:
                    raise Abort('tre_gui', k=k, gap_ms=f'{gap * 1000:.1f}')
            if not alive():
                raise Abort('logger_chet', k=k)
            if k < n_plan:
                vx, vy, wz = sample(segs, k)
                i = seg_at(segs, k * DT)
                if i not in seg_done:
                    seg_done.add(i)
                    s = segs[i]
                    out(f"#seg,i={i},loai={s['loai']},t_plan_s={s['t0']:.3f},t_sent_s={now:.6f},t_epoch={wall():.3f},"
                        f"truc={s['truc']},tu={s['tu']:.6f},toi={s['toi']:.6f},v={s['v']:g},a={s['a']:g}")
            else:
                vx = vy = wz = 0.0
            try:
                sock.sendto(pack_packet(vx, vy, wz), (ip, TRAJ_PORT))
            except OSError as exc:
                raise Abort('socket', k=k, loi=str(exc).replace(',', ';'))
            last_send = clock() - t0
            if k == 0:
                first_s = last_send
            if k == n_plan - 1:
                last_s = last_send
            sent += 1
            k += 1
            wait = t0 + k * DT - clock()
            if wait > 0:
                sleep(wait)
    except KeyboardInterrupt:
        exc = Abort('ctrl_c', k=k)
        _abort_zeros(ip, sock, sleep)
        _end(out, wall, sent, total, skipped, gap_max, first_s, last_s, complete=0)
        out(f"#seq_huy,ly_do={exc.why},k={k}")
        raise exc
    except Abort as exc:
        _abort_zeros(ip, sock, sleep)
        _end(out, wall, sent, total, skipped, gap_max, first_s, last_s, complete=0)
        out('#seq_huy,ly_do=' + exc.why + ''.join(f',{a}={b}' for a, b in exc.kv.items()))
        raise
    complete = int(sent == total and skipped == 0)
    _end(out, wall, sent, total, skipped, gap_max, first_s, last_s, complete)
    return dict(sent=sent, planned=total, n_bo=skipped, gap_max_ms=gap_max * 1000, complete=complete)


def _abort_zeros(ip, sock, sleep):
    for _ in range(int(round(ABORT_ZERO_S * SEND_HZ))):
        try:
            sock.sendto(pack_packet(0.0, 0.0, 0.0), (ip, TRAJ_PORT))
        except OSError:
            pass
        sleep(DT)


def _end(out, wall, sent, total, skipped, gap_max, first_s, last_s, complete):
    f = lambda v: '' if v is None else f'{v:.6f}'           # noqa: E731
    out(f"#seq_end,t_epoch={wall():.3f},sent={sent},planned={total},n_bo={skipped},gap_max_ms={gap_max * 1000:.1f},"
        f"complete={complete},sched_first_s={f(first_s)},sched_last_s={f(last_s)}")


# ---------------------------------------------------------------- one attempt
def others_running(pgrep=None, me=None):
    """Names of other sender / logger processes (pgrep -af); [] when pgrep is missing (Windows tests)."""
    me = os.getpid() if me is None else me
    run = pgrep or (lambda: subprocess.run(['pgrep', '-af', 'python'], capture_output=True, text=True).stdout)
    try:
        txt = run()
    except (OSError, FileNotFoundError):
        return []
    found = []
    for line in (txt or '').splitlines():
        parts = line.split(None, 1)
        if not parts or not parts[0].isdigit() or int(parts[0]) == me:
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


def start_logger(d, src_ip, popen=subprocess.Popen):
    out = open(os.path.join(d, 'logger_stdout.txt'), 'x', encoding='utf-8')
    kw = dict(stdout=out, stderr=subprocess.STDOUT)
    if os.name == 'posix':
        kw['start_new_session'] = True          # Ctrl-C at the terminal does not reach the logger (HC3-2)
    else:
        kw['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    p = popen([sys.executable, LOGGER, '--strict', '--src-ip', src_ip, '--out', os.path.join(d, 'log.csv')], **kw)
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
                query=gain_query_file, pgrep=None) -> int:
    segs = schedule(a.cho, a.phan)
    d = a.thu_muc
    if os.path.lexists(d):
        raise SeqError(f'--thu-muc {d} already exists: use a new folder for each attempt')
    busy = others_running(pgrep)
    if busy:
        raise SeqError(f'other sender/logger processes running: {busy}')
    os.makedirs(d)
    thu_id = os.path.basename(os.path.normpath(d))
    lich = lich_csv(segs)
    with open(os.path.join(d, 'lich.csv'), 'x', encoding='utf-8', newline='\n') as f:
        f.write(lich)
    lich_sha = hashlib.sha256(lich.encode('utf-8')).hexdigest()
    txt = open(os.path.join(d, 'sender.txt'), 'x', encoding='utf-8')

    def say(line):
        txt.write(line + '\n')
        txt.flush()
        out(line)

    proc, lout = start_logger(d, a.src_ip, popen)
    code = 0
    try:
        csv_path = os.path.join(d, 'log.csv')
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
        s = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            send_schedule(a.ip, segs, s, say, clock, sleep, wall, alive=lambda: proc.poll() is None,
                          cho=a.cho, phan=a.phan, thu_id=thu_id, lich_sha=lich_sha)
        except Abort:
            code = 3
        finally:
            if own:
                s.close()
        say(f"#gain_query,when=sau,exit={query(a.ip, os.path.join(d, 'gain_query_sau.txt'))}")
        sleep(STILL_AFTER_S)
    except SeqError as exc:
        say(f'#tu_choi,ly_do={str(exc).replace(",", ";")}')
        code = 2
    except KeyboardInterrupt:
        say('#seq_huy,ly_do=ctrl_c_ngoai_lich')
        code = 3
    finally:
        rc = stop_logger(proc)
        lout.close()
        say(f'#logger_dong,ma={rc}')
        say(f'#ket_qua,ma={code}')
        txt.close()
    return code


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
        return run_attempt(a)
    except SeqError as exc:
        print(f'#tu_choi,ly_do={exc}')
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    sys.exit(main())
