#!/usr/bin/env python3
"""
o1_cmd_path.py: command path of an O1 floor run from the ESP32 log (O1 v11.19 section 3.5, "Luot
hong do duong lenh" (1)-(4) and "Dieu (2), (3), (4) khi mat dong log"; S05 v3.24). Task given to
CODE_CAM by CODE 14/11 (Liam 14/11, THESIS_B8 local_25d28216, "Duyet, kiem o B8-1b (Recommended)"
for B8 v3 "Viec CODE" 1-2). Design: CLAUDE_THESIS 03_CODE/docs/2026-11-14_Thiet_ke_o1_cmd_path.md,
approved by Liam 14/11 in CODE_CAM local_377135de ("Duyet, code luon (Recommended)"; window
"Tu canh len (Recommended)"; branch "Nhanh moi (Recommended)").

Before C8 the thresholds of (2), (3), (4) are not locked: the script REPORTS the bounds and the
missing-evidence flags; it decides nothing about (2)-(4) values.

Log: agv_logger CSV of ONE run (one run_active rising edge), firmware columns of AGV 5bde4c5
(binary of the campaign): log_seq, timestamp_us, tick, motor_on, run_active, pos_reset_n,
pos_x_ref, pos_y_ref, sp_vx, sp_vy, sp_wz, traj_age_ms (older firmware without traj_age_ms: (3)
and the packet-age check are not computed, said so). '#' lines and <log>.events.txt: '#us_end'
(blk_ticks, cut_ticks) decides whether (4) applies (no ultrasonic detect or block tick), tied to the
run by '#us,ev=start' or '#gain' (us_for_edge).

Rising edge (O1 3.4): first data line with run_active = 1 whose line just before by log_seq was
received and has run_active = 0. That line lost: the edge is unknown (run lost to the measurement
system, edge_unknown). Exactly one rising edge in the log.

Window of the schedule on the ESP32 clock (Liam Q1): start t_w0 = timestamp_us of the edge line;
end t_w1 = t_w0 + (sched_last_s - sched_first_s) x (1 + E_MAX) + T_SEND, the REAL send times of the
first and last schedule packets on the Pi monotonic clock, from the '#test_end' line of the sender log
(fields sched_first_s, sched_last_s; O1 v11.23 3.5 B81c-1, the sender may be late then send in a
burst; Liam 15/11 CODE_CAM "Tinh tu canh len (Recommended)", format agreed with CODE 15/11). No such
time: the ENCLOSING window of O1 v11.24 3.5 (K-1) to '#test_end', t_w1 = t_w0 + (t_epoch #test_end -
t_epoch #test_start) x (1 + E_MAX) + T_SEND (Liam 15/11 "Tu canh len + gio #test_end"), refused when
shorter than the nominal schedule (wall clock step). Neither: (2), (3), (4) no evidence
(sched_times_missing).
E_MAX = 2002e-6 the Pi/ESP32 clock-rate bound of S05 4.7 (i),
T_SEND = one send period. The sender has no packet before packet 0 (pc_sender.test_loop) and the
firmware stamps a packet with its RECEIVE time (main.c 5bde4c5 88) and runs only while a packet is
fresh (< TRAJ_TIMEOUT 200 ms, main.c 45, 524-525, 611), so with the motor on the edge comes within
a tick of the first packet received; lost first packets make the edge late, a schedule phase error
that (4) sees.

Anchors (O1 3.5): received data lines sorted by log_seq.
  (2), (4): only from the edge line on, inside the same run_active = 1 stretch (pos_ctrl.c 5bde4c5
            67-81 zeroes the command pose at the edge tick and integrates the first step at once,
            so a pair across the edge is not bounded by |v| dt). End anchor: first line at or
            after t_w1. run_active falls and rises again before the end anchor: no evidence.
  (3)       start anchor: last line with t + L <= t_w0 in the same motor-on stretch; end anchor:
            first line at or after t_w1; a line b is used as a LATER anchor only when the line
            b+1 (next log_seq, motor on) exists: L = t_(b+1) - t_b. A start anchor age of a packet received
            before the log began (every line from the first to the start anchor received, each age >=
            TRAJ_TIMEOUT_MS: the run before) is read as -1 (Liam 23/11 CODE_CAM local_f3c5357d; Codex C3N1-1).
            The window is mapped from the edge (Liam 15/11): schedule packets lost at the start make the edge
            late, a phase error that (4) sees, not (3).
  (2), (4): the value of a run is the largest of the deviation ON the anchor lines and of the pair
  bound below over the pairs that HIDE ticks (a lost line, or a tick difference other than 1):
  every control tick pushes one data line (tick + 1 on every consecutive pair, 1030/1030 on
  CAR_FLOOR/2026-11-10_run01 c6_f08) and the command pose changes only at a tick, so between two
  consecutive lines no value is hidden. Liam 14/11 CODE_CAM, box "Chi cho mat dong (Recommended)";
  the bound over EVERY pair is still reported (value_all_m). THESIS words O1 3.5 accordingly.
    (2) lane deviation        (d(a) + d(b) + v dt)/2 + R
    (4) schedule deviation    (e(a) + e(b) + 2 v dt)/2 + R
  (3) for EVERY pair of consecutive anchors (O1 3.5): age(a) + 1 ms + dt + L_b
  R = 0.1 mm + 1e-6 v dt + 2e-4 mm x n_tick (x 2 when |x_ref| or |y_ref| >= 4 m), n_tick = tick
  difference of the pair. v = v_cmd,max of the level by o1_gate.fig8_cmd_max (K1, K2: k_cmd_max).
  Anchor coordinates must stay |x_ref|, |y_ref| < 8 m (else the run is lost to the command path).
Lane and ideal schedule in the FIRMWARE frame, by the NOMINAL start pose (O1 3.5 (4), SC-11):
figure 8 start at the origin with heading 45 deg (o1_feasibility.HEAD0), lane = the whole figure 8
(no window); K1: segment (0,0)-(0,2) m; K2: square (0,0)-(1,0)-(1,1)-(0,1)-(0,0) m (o1_tasks).
Ideal p_cmd(t), t = timestamp_us - t_w0: figure 8 = path_point(tau(t)); K1, K2 = integral of the
body command (no rotation).
(1): more than one pos_reset_n value on the run_active stretch from the edge to the end anchor.
Packet before the edge (task 2 of CODE, with C6): lines before the edge (motor on or off) with
0 <= traj_age_ms < 200 (a fresh packet that did not start the run); the lag from the earliest such
receive to the edge is reported, pre_edge_packet only from 200 ms (the 10 ms traj_task transfer puts the
first packet 1-2 lines before every normal edge; Liam 14/11 "Bao so, co tu 200 ms"). A non-zero
sp_* alone proves nothing (MANUAL mode, CP1-4). The edge line traj_age_ms is a diagnostic number
(note above EDGE_AGE_MS, no flag: no locked criterion, CP1-5).
Codex Sol round 1 (14/11, session 01a1da4f): CP1-1..5 accepted and fixed (reset check over the whole
log after the edge; required columns and empty cells; #us_end tied to the run of the edge; manual
commands; edge age as a note). Round 2 (14/11, CP2-1..4; Liam 14/11 CODE_CAM local_ed93bb1a "Sua hep +
vong 3 (Recommended)"): #us_end not past a later run start (#us or #gain first_seq) or the first
motor-off line after the edge; #gain first_seq as a run start when #us start is lost; run_active = 1
with traj_age_ms < 0 is a flag. Found on the real CAR_LIFTED 14/11 logs, same round: .events.txt lines
carry the agv_logger time prefix (read_events); pre_edge_packet from a 200 ms lag. Round 3 (15/11;
Liam 15/11 CODE_CAM "Sua hep + vong 4 (Recommended)"): '#run_end' bounds the '#us_end' of the run; the lag
is an interval (age read after timestamp_us, floored to ms), across 200 ms -> pre_edge_packet_undecided.
Round 4 (15/11; Liam 15/11 CODE_CAM "Sua + Codex vong 5"): the open upper bound (lag_hi == 200 ms is below
200 ms); tick/timestamp increase checked to the end of the motor-on run; L <= 0 is not an anchor.
K-1 (CODE 15/11, O1 v11.24 3.5; Liam 15/11 CODE_CAM "Tu canh len + gio #test_end (Recommended)", "Gop K-1
vao vong 5"): no sched times -> the enclosing window to #test_end (t_epoch span), checked against the
nominal schedule; both present -> they must agree.
Round 5 (15/11; Liam 15/11 CODE_CAM "Sua + Codex vong 6", "Ghi gioi han", "Them vao vong 6"): timestamp_us
increase on every line after the edge (CP5-1); enclosing span >= nominal exactly (CP5-2); note
window_from_wall_clock, a known limit (CP5-3); sender '#test_interrupted' or complete != 1 -> sender_incomplete.
Round 6 (15/11): epochs as exact decimals, span compared in whole ms (CP6-1); '#test_interrupted' checked
before the missing '#test_end' case (CP6-2).
Round 7 (15/11; Liam 15/11 CODE_CAM "Chay vong 7"): the span kept as an exact Decimal and compared
without rounding (CP7-1); a t_epoch that is not a finite number is a ValueError, exit 2 (CP7-2).
Round 8 (15/11; Liam 15/11 CODE_CAM "Chay vong 8"): t_epoch must be the sender format digits.ddd (at most
12 integer digits), anything else is exit 2, so the Decimal span is exact and finite (CP8-1, CP8-2).
Round 9 (15/11; Liam 15/11 CODE_CAM "Sua 3 muc + vong 10"): no leading zero in the integer part (CP9-1);
the sender log keeps the field text as written, only the line end is removed before the format check
(CP9-2); every t_epoch present is checked before a missing line gives None (CP9-3).
B8-1c (CODE 15/11, O1 v11.23 3.5; Liam 15/11 CODE_CAM "Lam B8-1c roi vong 4 gop (Recommended)"): window end
from the real send times (--sender, B81c-1); a pair whose tick or timestamp_us does not increase is no
evidence for (2), (4) (B81c-7).
On the HIL bench the H7 relays packets all the time, so pre_edge_packet is expected there.

  python tools/o1_cmd_path.py --log <run.csv> --task O1-F05 --sender <run>_sender.txt
                              [--events <run.csv.events.txt>]
                              [--allow-final-set] [--json out.json]
Last line: #ket_qua,ma=0 no flag / 1 run lost to the command path or to the measurement system
(flags in the JSON) / 2 error.
"""
from __future__ import annotations

import argparse
import csv
import re
from decimal import Decimal
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import o1_tasks                                              # noqa: E402  (read only, S04 blob)
import o1_feasibility as feas                                # noqa: E402
import o1_gate                                               # noqa: E402

SEND_HZ = 50
T_SEND = 1.0 / SEND_HZ
E_MAX = 2002e-6               # S05 4.7 (i): (1 + rho)/(1 - rho) - 1, rho = 1e-3
TRAJ_TIMEOUT_MS = 200         # main.c 5bde4c5 line 45
EDGE_AGE_MS = 30              # [DE XUAT] edge line packet age: about one tick + transfer
EPOCH_RE = re.compile(r'(?:0|[1-9][0-9]{0,11})\.[0-9]{3}')   # sender t_epoch (.3f), see read_sender_bao
COORD_MAX_M = 8.0             # O1 3.5: rounding term valid for |x_ref|, |y_ref| < 8 m
COORD_DOUBLE_M = 4.0          # from 4 m the float32 term doubles (O1 3.5, B81-18)
R_LOG_M = 1e-4                # 0.1 mm: log 4 decimals per coordinate (logger.c 223, 249-250)
R_DT_REL = 1e-6               # float dt
R_TICK_M = 2e-7               # 2e-4 mm per tick (float32 accumulation)


# ---------------------------------------------------------------- log
REQUIRED = ('log_seq', 'timestamp_us', 'tick', 'motor_on', 'run_active', 'pos_reset_n', 'pos_x_ref', 'pos_y_ref')


def read_log(path):
    """(rows sorted by log_seq, '#' lines, n_duplicates, has traj_age_ms column). Identical duplicate
    lines of one log_seq are dropped and counted (L92); two different lines with the same log_seq is
    an error. A missing REQUIRED column or an empty REQUIRED cell is an error (CP1-2)."""
    rows, hashes, dup = {}, {}, 0
    events = []
    with open(path, newline='', encoding='utf-8') as f:
        lines = [ln for ln in f]
    data = [ln for ln in lines if not ln.startswith('#')]
    events += [ln.strip() for ln in lines if ln.startswith('#')]
    for r in csv.DictReader(data):
        if r.get('log_seq', '') in ('', None):
            continue
        s = int(r['log_seq'])
        key = tuple(r.values())
        if s in rows:
            if hashes[s] == key:
                dup += 1
                continue
            raise ValueError(f'log_seq {s}: two different lines')
        rows[s] = r
        hashes[s] = key
    out = [rows[s] for s in sorted(rows)]
    if not out:
        raise ValueError('no data line')
    cols = set(out[0].keys())
    miss = [c for c in REQUIRED if c not in cols]
    if miss:
        raise ValueError(f'missing columns {miss} (CP1-2)')
    for r in out:
        empty = [c for c in REQUIRED if r.get(c, '') in ('', None)]
        if empty:
            raise ValueError(f'log_seq {r.get("log_seq")}: empty {empty} (CP1-2)')
        r['_seq'] = int(r['log_seq'])
        r['_t'] = int(r['timestamp_us'])
        r['_ra'] = r['run_active'] == '1'
        r['_mo'] = r['motor_on'] == '1'
        r['_x'] = float(r['pos_x_ref'])
        r['_y'] = float(r['pos_y_ref'])
        r['_tick'] = int(r['tick'])
        r['_rst'] = r['pos_reset_n']
        # traj_age_ms: absent column (firmware before RB-4) or empty cell -> None (MISSING, CP1-2);
        # -1 is the firmware value 'no packet since boot', kept as -1
        a = r.get('traj_age_ms', '') if 'traj_age_ms' in cols else ''
        r['_age'] = None if a in ('', None) else int(a)
    return out, events, dup, 'traj_age_ms' in cols


def read_events(path, keep_tail=False):
    """'#' lines of <log>.events.txt. agv_logger.py writes each as '<PC time ISO> <payload>' (lines 880,
    894); the time prefix is dropped so identical copies of one event compare equal. A bare '#' line
    is kept as it is. keep_tail: only the line end (CR, LF) is removed at the right, so a blank after
    the last field stays in that field (sender t_epoch format check, CP9-2)."""
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for ln in f:
            ln = ln.rstrip('\r\n').lstrip() if keep_tail else ln.strip()
            if not ln.startswith('#'):
                parts = ln.split(' ', 1)
                ln = (parts[1].lstrip() if keep_tail else parts[1].strip()) if len(parts) == 2 else ''
            if ln.startswith('#'):
                out.append(ln)
    return out


def _kv(e):
    return dict(p.split('=', 1) for p in e.split(',')[1:] if '=' in p)


def us_for_edge(events, edge_seq, off_seq=None):
    """(blk_ticks, cut_ticks) of the ultrasonic summary of the motor-on run that holds the edge, or
    None when it cannot be tied (CP1-3, CP2-1, CP2-2). Firmware AGV 5bde4c5 (main.c run_udp_mark
    349-355, push_run_end 358-361; us_state.c us_format_line; logger.c 184): at motor on, in the same
    tick, '#gain,...,first_seq=F,...,boot=B' and '#us,ev=start,boot=B,...,first_seq=F' (F = first data
    line of the run; 3 copies each); at motor off '#us_end,boot=B,...,blk_ticks,cut_ticks,ticks,
    last_data_seq=L'. '#gain' with first_seq=0 is a gain set (main.c 636), not a run start.
    Run starts = '#us' starts and '#gain' lines with F > 0 and a boot field (either one is enough,
    CP2-2; a '#gain' without boot only marks a boundary).
    Tied when the latest start with F <= edge_seq has boot B and the first '#us_end' of boot B with
    L >= edge_seq lies before every known boundary after the edge: any run start with F > F0 (any
    boot), off_seq, the first received motor_on = 0 data line after the edge (CP2-1), and the end of
    the run itself: '#run_end,seq=S,last_data_seq=R,...' (logger.c 202; same R as the '#us_end' of that
    run, push_run_end; no boot field, the 3 copies differ in seq) with R >= F0 means no data line of
    that run after R, so an '#us_end' with L > R belongs to a later run (CP2-1, Codex round 3)."""
    starts, ends, bounds = [], [], []
    for e in dict.fromkeys(events):                      # identical copies once (logger sends 3)
        if e.startswith('#run_end,'):
            kv = _kv(e)
            if 'last_data_seq' in kv:
                bounds.append(int(kv['last_data_seq']) + 1)
        elif e.startswith('#us,') and _kv(e).get('ev') == 'start':
            kv = _kv(e)
            starts.append((int(kv['first_seq']), kv.get('boot')))
        elif e.startswith('#gain,'):
            kv = _kv(e)
            f = int(kv.get('first_seq', '0'))
            if f > 0:
                (starts.append((f, kv['boot'])) if kv.get('boot') else bounds.append(f))
        elif e.startswith('#us_end'):
            kv = _kv(e)
            ends.append((int(kv['last_data_seq']), kv.get('boot'), int(kv['blk_ticks']), int(kv['cut_ticks'])))
    starts = list(dict.fromkeys(starts))
    before = [s for s in starts if s[0] <= edge_seq]
    if not before:
        return None
    f0 = max(s[0] for s in before)
    boots = {s[1] for s in before if s[0] == f0}
    if len(boots) != 1:                                  # two boots claim the same start line
        return None
    boot = boots.pop()
    cand = sorted(e for e in ends if e[1] == boot and e[0] >= edge_seq)
    if not cand:
        return None
    last = cand[0][0]
    bounds += [s[0] for s in starts if s[0] > f0]
    if off_seq is not None:
        bounds.append(off_seq)
    if any(b <= last for b in bounds if b > f0):         # the end belongs to a later motor-on run
        return None
    return cand[0][2], cand[0][3]


# ---------------------------------------------------------------- task geometry (firmware frame)
def _rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


class Task:
    """Lane distance and ideal command pose of an O1 task in the firmware frame (nominal start)."""

    def __init__(self, task_id, allow_final_set=False):
        self.id = task_id
        self.duration, self.f = o1_tasks.task(task_id, allow_final_set)
        n_total = int(math.ceil(self.duration * SEND_HZ - 1e-9))     # = pc_sender.grid_counts
        self.n_sched = n_total
        if task_id in o1_tasks.FIG8:
            self.sch = feas.Schedule(o1_tasks.FIG8[task_id])
            self.v = float(o1_gate.fig8_cmd_max(self.sch)[0])
            self.path = feas.Path8()
            self.kind = 'fig8'
        else:
            self.sch = None
            self.v = float(o1_tasks.K_V)
            self.kind = task_id
            self.poly = (np.array([[0.0, 0.0], [0.0, o1_tasks.K1_LEN]]) if task_id == 'O1-K1' else
                         np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float) * o1_tasks.K2_SIDE)
            tg = np.arange(0.0, self.duration + 0.5, 1e-3)
            vel = np.array([self.f(t)[:2] for t in tg])
            pos = np.vstack([[0.0, 0.0], np.cumsum((vel[1:] + vel[:-1]) / 2.0 * 1e-3, axis=0)])
            self._tg, self._pos = tg, pos

    def lane_dist(self, x, y):
        if self.kind == 'fig8':
            fx, fy = _rot(x, y, feas.HEAD0)              # firmware -> floor (start heading 45 deg)
            return float(self.path.distance_to_path(np.array([fx]), np.array([fy]))[0])
        best = math.inf
        p = np.array([x, y])
        for a, b in zip(self.poly[:-1], self.poly[1:]):
            ab = b - a
            u = min(1.0, max(0.0, float(np.dot(p - a, ab) / np.dot(ab, ab))))
            best = min(best, float(np.linalg.norm(p - (a + u * ab))))
        return best

    def ideal(self, t):
        """Ideal command pose (x, y) in the firmware frame at t [s] from the command start."""
        if self.kind == 'fig8':
            tau, _ = self.sch.tau(t)
            px, py, _ = feas.path_point(tau)
            return _rot(px, py, -feas.HEAD0)
        t = min(max(t, 0.0), self._tg[-1])
        return (float(np.interp(t, self._tg, self._pos[:, 0])), float(np.interp(t, self._tg, self._pos[:, 1])))


# ---------------------------------------------------------------- analysis
def round_term(v, dt_s, n_tick, big):
    return R_LOG_M + R_DT_REL * v * dt_s + R_TICK_M * n_tick * (2 if big else 1)


def find_edge(rows):
    """(index of the edge line or None, list of problems)."""
    by_seq = {r['_seq']: i for i, r in enumerate(rows)}
    rises, unknown = [], []
    for i, r in enumerate(rows):
        if not r['_ra']:
            continue
        j = by_seq.get(r['_seq'] - 1)
        if j is None:
            # first line of a run_active stretch whose predecessor is lost (or the log start)
            prev_ra = [q for q in rows[:i] if q['_seq'] < r['_seq']]
            if not prev_ra or not prev_ra[-1]['_ra']:
                unknown.append(r['_seq'])
            continue
        if not rows[j]['_ra']:
            rises.append(i)
    probs = []
    if unknown and (not rises or rows[by_seq[unknown[0]]]['_seq'] < rows[rises[0]]['_seq']):
        probs.append(f'edge_unknown: run_active = 1 at log_seq {unknown} after a lost line')
        return None, [], probs
    if not rises:
        probs.append('no run_active rising edge')
        return None, [], probs
    # later rises: run_active fell and rose again (judged against the window by the caller)
    later = [rows[i]['_seq'] for i in rises[1:]] + [q for q in unknown if q > rows[rises[0]]['_seq']]
    return rises[0], later, probs


def read_sender(path, task_id):
    """(sched_first_s, sched_last_s) from the '#test_end' line of the sender log of task_id, or None when
    the log or a time is missing (B81c-1). Two '#test_end' lines of the task, or last < first: error."""
    if not path or not os.path.exists(path):
        return None
    ends = [e for e in read_events(path) if e.startswith('#test_end,') and _kv(e).get('test') == task_id]
    if not ends:
        return None
    if len(set(ends)) > 1:
        raise ValueError(f'{path}: {len(set(ends))} different #test_end lines of {task_id}')
    kv = _kv(ends[0])
    a, b = kv.get('sched_first_s', ''), kv.get('sched_last_s', '')
    if a == '' or b == '':
        return None
    a, b = float(a), float(b)
    if b < a:
        raise ValueError(f'{path}: sched_last_s {b} < sched_first_s {a}')
    return a, b


def read_sender_bao(path, task_id):
    """t_epoch(#test_end) - t_epoch(#test_start) of task_id in the sender log [s], or None (K-1, O1 v11.24
    3.5: the enclosing window ends at '#test_end' when the log has no send time of the last schedule
    packet; pc_sender test_loop prints '#test_start' before packet 0 and '#test_end' after the zero tail).
    Two different lines of either kind: error. Every t_epoch present is checked first, also when the
    other line is missing (CP9-3); a line without a t_epoch field gives None."""
    if not path or not os.path.exists(path):
        return None
    ev = read_events(path, keep_tail=True)               # field text as written (CP9-2)
    got = {}
    for kind in ('#test_start,', '#test_end,'):
        ls = list(dict.fromkeys(e for e in ev if e.startswith(kind) and _kv(e).get('test') == task_id))
        if len(ls) > 1:
            raise ValueError(f'{path}: {len(ls)} different {kind[:-1]} lines of {task_id}')
        if not ls or 't_epoch' not in _kv(ls[0]):
            continue                                     # missing: no evidence, None below
        txt = _kv(ls[0])['t_epoch']
        # only the sender's own format, f'{epoch:.3f}' (pc_sender.py 20f1f81 117, 144): digits, a point,
        # exactly 3 decimals, at most 12 integer digits, no leading zero (CP9-1). Anything else (empty,
        # sign, exponent, more decimals, blank, NaN, text) is an error, exit 2 (CP7-2, CP8-1, CP8-2): so the
        # Decimal arithmetic below has at most 15 significant digits, is exact in the default context and
        # never leaves the float range
        if not EPOCH_RE.fullmatch(txt):
            raise ValueError(f'{path}: {kind[:-1]} t_epoch {txt!r} is not the sender format digits.ddd')
        got[kind] = Decimal(txt)                         # exact decimal: no binary rounding (CP6-1)
    if len(got) < 2:
        return None
    return got['#test_end,'] - got['#test_start,']      # Decimal, exact (CP7-1)


def read_sender_complete(path, task_id):
    """(ok, why) of the sender run: ok False when the sender log has '#test_interrupted' or its '#test_end'
    of task_id has complete other than 1 (an interrupt in the zero tail gives complete=1 with
    '#test_interrupted', review pc_sender round 2, SS1-2); None when there is no sender log or no
    '#test_end' (left to sched_times_missing). Liam 15/11 CODE_CAM "Them vao vong 6 (Recommended)"."""
    if not path or not os.path.exists(path):
        return None, 'no sender log'
    ev = read_events(path)
    ends = [e for e in ev if e.startswith('#test_end,') and _kv(e).get('test') == task_id]
    if any(e.startswith('#test_interrupted') for e in ev):     # before the no-end case (CP6-2)
        return False, '#test_interrupted'
    if not ends:
        return None, 'no #test_end'
    if any(_kv(e).get('complete') != '1' for e in ends):
        return False, 'complete != 1'
    return True, ''


def _packet_before_log(rows, start):
    """(3) start anchor (Liam 23/11; Codex Sol C3N1-1): True when its traj_age_ms is that of a packet received before
    the log began: every data line from the first one of the log to the start anchor received (consecutive
    log_seq) and each with traj_age_ms >= TRAJ_TIMEOUT_MS. A packet received inside that stretch would show on the
    line of the next tick with an age below one tick (MANUAL or AUTO alike); the first line at >= TRAJ_TIMEOUT_MS puts
    it 200 ms or more before the log. A lost line, an empty cell or an age below TRAJ_TIMEOUT_MS: False (the age
    stays a number)."""
    seg = [r for r in rows if r['_seq'] <= start['_seq']]
    if not seg or any(b['_seq'] != a['_seq'] + 1 for a, b in zip(seg, seg[1:])):
        return False
    return all(r['_age'] is not None and r['_age'] >= TRAJ_TIMEOUT_MS for r in seg)


def analyse(rows, task, events=(), has_age_col=True, sched=None, bao_s=None, sender_ok=None):
    """sched: (sched_first_s, sched_last_s) of the sender (B81c-1); bao_s: #test_start -> #test_end epoch
    span of the sender (K-1), used when sched is None; sender_ok: (ok, why) of read_sender_complete."""
    res = dict(task=task.id, v_cmd_max=task.v, n_sched=task.n_sched, flags=[], notes=[])
    edge, later, probs = find_edge(rows)
    res['flags'] += probs
    if edge is None:
        res['decided'] = False
        return res
    e = rows[edge]
    t0 = e['_t']
    # Window end (Liam 14/11 Q1 "Tu canh len"; 15/11 "Tinh tu canh len", "Tu canh len + gio #test_end"):
    #   sched present: edge + (sched_last_s - sched_first_s) x (1 + E_MAX) + T_SEND (last schedule packet);
    #   else the ENCLOSING window of O1 v11.24 3.5 (K-1): edge + (t_epoch end - t_epoch start) x (1 + E_MAX)
    #   + T_SEND. The epoch is the Pi wall clock (time.time): a span shorter than the nominal schedule
    #   (a clock step) is no evidence. Both present: the enclosing span must not be shorter than the
    #   schedule span (the two ways must agree), else no evidence.
    nominal = (task.n_sched - 1) * T_SEND
    # the enclosing span is compared EXACTLY (Decimal, no rounding: CP7-1); a float bao_s (tests) is taken
    # at its shortest decimal form
    bao_dec = None if bao_s is None else (bao_s if isinstance(bao_s, Decimal) else Decimal(repr(float(bao_s))))
    bao_s = None if bao_dec is None else float(bao_dec)
    win_src, win_ok, span_s = None, False, nominal                     # nominal: report only
    if sched is not None:
        win_src, win_ok, span_s = 'sched', True, sched[1] - sched[0]
        if bao_s is not None and bao_s < span_s:
            win_ok = False
            res['flags'].append(f'window_mismatch: #test_start -> #test_end {bao_s:.3f} s shorter than the '
                                f'schedule send span {span_s:.3f} s: (2), (3), (4) no evidence')
    elif bao_s is not None:
        win_src, span_s = 'test_end', bao_s
        # both on the Pi: no clock-rate term (CP5-2); exact decimal compare (CP6-1, CP7-1)
        win_ok = bao_dec >= Decimal(task.n_sched - 1) / Decimal(SEND_HZ)
        if not win_ok:
            res['flags'].append(f'test_end_span_short: #test_start -> #test_end {bao_s:.3f} s shorter than the '
                                f'nominal schedule {nominal:.3f} s (wall clock step?): (2), (3), (4) no evidence')
        # known limit (CP5-3; Liam 15/11 "Ghi gioi han (Recommended)"): the epochs are the Pi wall clock, a
        # step back while sending slowly is not seen; only sched_first_s / sched_last_s (monotonic) prove it
        res['notes'].append('window_from_wall_clock: enclosing window from the #test_start/#test_end epochs '
                            '(Pi wall clock); a clock step back during the run is not detected')
    if sender_ok is not None and sender_ok[0] is False:
        win_ok = False
        res['flags'].append(f'sender_incomplete: {sender_ok[1]} in the sender log: run not scored, '
                            f'(2), (3), (4) no evidence')
    t_w1 = t0 + int(round((span_s * (1 + E_MAX) + T_SEND) * 1e6))
    res.update(edge_seq=e['_seq'], t_w0_us=t0, t_w1_us=t_w1, edge_traj_age_ms=e['_age'],
               sched_send_span_s=(sched[1] - sched[0]) if sched is not None else None,
               test_end_span_s=bao_s, window_source=win_src)
    if win_src is None:
        res['flags'].append('sched_times_missing: no send time of the last schedule packet and no '
                            '#test_start/#test_end epoch in the sender log: (2), (3), (4) no evidence')
    has_age = has_age_col
    if not has_age_col:
        res['flags'].append('no_traj_age_column: (3) and the packet-before-edge check no evidence '
                            '(main runs need the campaign binary, O1 3.5)')
    off_seq = next((r['_seq'] for r in rows[edge:] if not r['_mo']), None)   # motor off after the run
    us = us_for_edge(events, e['_seq'], off_seq)

    # motor-on stretch around the edge
    lo = edge
    while lo > 0 and rows[lo - 1]['_mo']:
        lo -= 1
    pre = rows[lo:edge]
    # packet before the edge (task 2)
    # every line before the edge, motor on or off: a schedule started while the motor was off makes the
    # edge come mid-schedule with no motor-on line before it (one file = one run). Only the packet age
    # shows a received trajectory packet: a non-zero sp_* may be a MANUAL command (CP1-4)
    pe_rows = [r for r in rows[:edge] if r['_age'] is not None and 0 <= r['_age'] < TRAJ_TIMEOUT_MS]
    pe = [r['_seq'] for r in pe_rows]
    miss_pre = [r['_seq'] for r in rows[:edge] if has_age and r['_age'] is None]
    if miss_pre:
        res['flags'].append(f'pre_edge_age_missing: empty traj_age_ms before the edge at {miss_pre[:5]}')
    res['pre_edge_motor_on_lines'] = len(pre)
    res['pre_edge_packet_seqs'] = pe[:20]
    # traj_age_ms counts from the RECEIVE time (receiver snapshot, main.c 528-541) but run_active needs the
    # packet moved by traj_task every 10 ms (main.c 75-90, 520-525, 611): with the motor on in AUTO the
    # first packet comes 1-2 lines before the edge (CAR_LIFTED 2026-11-14_run01: 9/15 runs, 7-21 ms).
    # Lag = edge time - receive time of the earliest fresh packet before the edge, a number; flag only
    # from TRAJ_TIMEOUT_MS: a packet that old not running means motor off or MANUAL (Liam 14/11 CODE_CAM
    # local_ed93bb1a "Bao so, co tu 200 ms (Recommended)"; tighter limit locked at C8)
    # The age is read at t_age_ref, after timestamp_us of the line and before the next tick, and floored to
    # ms (main.c 532-540), so the receive time is known only within an interval (CP3-1, Codex round 3):
    #   D + a - S <= lag < D + a + 1 ms, D = t_edge - t_line, a = traj_age_ms, S = t_next_line - t_line.
    # The earliest packet seen has lag >= max(lower) and <= max(upper). Flag when the lower bound reaches
    # 200 ms; an interval across 200 ms is no evidence either way (pre_edge_packet_undecided).
    lag = lag_lo = lag_hi = None
    if pe_rows:
        idx = {r['_seq']: i for i, r in enumerate(rows)}
        pts, los, his = [], [], []
        for r in pe_rows:
            i = idx[r['_seq']]
            d_ms = (t0 - r['_t']) / 1e3
            s_ms = (rows[i + 1]['_t'] - r['_t']) / 1e3          # i < edge, so rows[i + 1] exists
            pts.append(d_ms + r['_age'])
            los.append(d_ms + r['_age'] - s_ms)
            his.append(d_ms + r['_age'] + 1.0)
        lag, lag_lo, lag_hi = max(pts), max(los), max(his)
    res.update(pre_edge_packet_lag_ms=lag, pre_edge_packet_lag_lo_ms=lag_lo, pre_edge_packet_lag_hi_ms=lag_hi)
    if lag is not None:
        span = f'{lag_lo:.1f}-{lag_hi:.1f} ms ({len(pe)} lines before the edge, 0 <= traj_age_ms < {TRAJ_TIMEOUT_MS})'
        if lag_lo >= TRAJ_TIMEOUT_MS:
            res['flags'].append(f'pre_edge_packet: a fresh packet received {span} before the edge')
        elif lag_hi > TRAJ_TIMEOUT_MS:                  # upper bound is open (CP4-1)
            res['flags'].append(f'pre_edge_packet_undecided: first packet {span} before the edge, '
                                f'across {TRAJ_TIMEOUT_MS} ms')
        else:
            res['notes'].append(f'pre_edge_packet_lag: first packet {span} before the edge '
                                f'(flag from {TRAJ_TIMEOUT_MS} ms)')
    if has_age and e['_age'] is not None and e['_age'] > EDGE_AGE_MS:
        # diagnostic number only: no locked criterion (CP1-5)
        res['notes'].append(f'edge_age: traj_age_ms {e["_age"]} at the edge (DE XUAT reference {EDGE_AGE_MS} ms)')
    # run_active = 1 needs a transferred packet (main.c 524, 611), and once a packet came the age is >= 0
    # (main.c 535-541): a run_active line with age < 0 is an inconsistent log, not a number (CP2-3)
    neg_ra = [r['_seq'] for r in rows[edge:] if r['_ra'] and r['_age'] is not None and r['_age'] < 0]
    if neg_ra:
        res['flags'].append(f'ra_age_negative: run_active = 1 with traj_age_ms < 0 at log_seq {neg_ra[:5]}')

    # run_active stretch from the edge to the end anchor
    end = None
    for i in range(edge, len(rows)):
        if rows[i]['_t'] >= t_w1:
            end = i
            break
    stretch = rows[edge:(end + 1 if end is not None else len(rows))]
    toggled = any(not r['_ra'] for r in stretch)
    res['run_active_toggle'] = toggled
    by_s = {r['_seq']: r for r in rows}
    after = [q for q in later if by_s[q]['_t'] > t_w1]
    if after:
        res['flags'].append(f'second_run: run_active rises again after the window at log_seq {after[:3]} '
                            f'(one file = one run)')
    if end is None:
        res['flags'].append('no_end_anchor: no line at or after the window end (2), (3), (4) no evidence')
    if toggled:
        res['flags'].append('run_active_toggle: run_active falls in the window: (2), (4) no evidence')
    # (1) and run integrity on EVERY line from the edge to the end of the log, not only the window:
    # a timeout then a new rise after the window with the run_active = 0 lines lost shows only as a
    # pos_reset_n change (CP1-1); one file = one run
    resets = list(dict.fromkeys(r['_rst'] for r in rows[edge:]))
    res['pos_reset_n_values'] = resets
    if len(resets) > 1:
        res['flags'].append(f'cond1_pos_reset: pos_reset_n {resets} after the edge')
    big = [r['_seq'] for r in stretch if max(abs(r['_x']), abs(r['_y'])) >= COORD_MAX_M]
    if big:
        res['flags'].append(f'coord_over_8m: |x_ref| or |y_ref| >= 8 m at log_seq {big[:5]}')
    gaps = sum(1 for a, b in zip(stretch[:-1], stretch[1:]) if b['_seq'] != a['_seq'] + 1)
    res['n_gap_in_window'] = gaps
    res['n_anchor'] = len(stretch)

    # a pair whose tick or timestamp_us does not increase (counter reset, vel_ctrl.c 5bde4c5 23) has no
    # bound: (2), (4) no evidence (O1 v11.23 3.5, B81c-7). Checked on every line from the edge to the end
    # of the motor-on run, not only up to the end anchor: the end anchor is chosen by time, so a line
    # coming back inside the window after it would be hidden (CP4-2, Codex round 4). timestamp_us is
    # esp_timer_get_time() (main.c 399-412) and never decreases in a sound log
    mo_hi = edge
    while mo_hi + 1 < len(rows) and rows[mo_hi + 1]['_mo']:
        mo_hi += 1
    run_rows = rows[edge:max(mo_hi, edge + len(stretch) - 1) + 1]
    # tick on the motor-on run; timestamp_us on EVERY line after the edge, motor off too (CP5-1: a
    # motor-off line coming back inside the window)
    # tick only between two motor-on lines: the counter is reset at motor on (vel_ctrl.c 5bde4c5 22-23, 41, 49)
    nonincr = [[a['_seq'], b['_seq']] for a, b in zip(run_rows[:-1], run_rows[1:])
               if a['_mo'] and b['_mo'] and b['_tick'] <= a['_tick']]
    nonincr += [[a['_seq'], b['_seq']] for a, b in zip(rows[edge:-1], rows[edge + 1:]) if b['_t'] <= a['_t']]
    if nonincr:
        res['flags'].append(f'pair_not_increasing: tick or timestamp_us does not increase at log_seq pairs '
                            f'{nonincr[:3]}: (2), (4) no evidence')

    # (2), (4)
    can24 = win_ok and end is not None and not toggled and not big and not nonincr
    c2 = c4 = None
    if can24:
        d = [task.lane_dist(r['_x'], r['_y']) for r in stretch]
        ev = []
        for r in stretch:
            ix, iy = task.ideal((r['_t'] - t0) / 1e6)
            ev.append(math.hypot(r['_x'] - ix, r['_y'] - iy))
        def pack(vals, pairs_all, pairs_hidden):
            i = int(np.argmax(vals))
            out = dict(anchor_max_m=float(vals[i]), anchor_seq=stretch[i]['_seq'],
                       pair_all_max_m=None, pair_all_seq=None, pair_hidden_max_m=None, pair_hidden_seq=None)
            for key, pairs in (('pair_all', pairs_all), ('pair_hidden', pairs_hidden)):
                if pairs:
                    v, sq = max(pairs, key=lambda q: q[0])
                    out[key + '_max_m'], out[key + '_seq'] = float(v), sq
            # bound over the run: lines themselves, then pairs (the two readings of O1 3.5, see docstring)
            out['value_all_m'] = max(out['anchor_max_m'], out['pair_all_max_m'] or 0.0)
            out['value_hidden_m'] = max(out['anchor_max_m'], out['pair_hidden_max_m'] or 0.0)
            out['value_m'] = out['value_hidden_m']           # Liam 14/11 CODE_CAM: pairs bound only at hidden ticks
            return out

        p2a, p2h, p4a, p4h = [], [], [], []
        for k in range(len(stretch) - 1):
            a, b = stretch[k], stretch[k + 1]
            dt = (b['_t'] - a['_t']) / 1e6
            nt = b['_tick'] - a['_tick']                       # >= 1 here (pair_not_increasing above)
            hidden = b['_seq'] != a['_seq'] + 1 or nt != 1     # a lost line or a tick not logged
            bg = max(abs(a['_x']), abs(a['_y']), abs(b['_x']), abs(b['_y'])) >= COORD_DOUBLE_M
            r = round_term(task.v, dt, nt, bg)
            b2 = ((d[k] + d[k + 1] + task.v * dt) / 2.0 + r, [a['_seq'], b['_seq']])
            b4 = ((ev[k] + ev[k + 1] + 2.0 * task.v * dt) / 2.0 + r, [a['_seq'], b['_seq']])
            p2a.append(b2)
            p4a.append(b4)
            if hidden:
                p2h.append(b2)
                p4h.append(b4)
        c2, c4 = pack(d, p2a, p2h), pack(ev, p4a, p4h)
        res['n_hidden_pairs'] = len(p2h)
    res['cond2'] = c2 if can24 else dict(anchor_max_m=None, no_evidence=True)
    if us is None:
        res['cond4'] = dict(c4 or dict(anchor_max_m=None), applies=None,
                            note='no #us_end line: ultrasonic block ticks unknown, (4) not decided')
        res['flags'].append('us_unknown: no #us_end, (4) cannot be applied')
    elif us[0] > 0 or us[1] > 0:
        res['cond4'] = dict(anchor_max_m=None, applies=False, note=f'ultrasonic blk_ticks {us[0]}, cut_ticks {us[1]}')
    else:
        res['cond4'] = dict(c4, applies=True) if can24 else dict(anchor_max_m=None, no_evidence=True, applies=True)

    # (3)
    if not has_age:
        res['cond3'] = dict(value_ms=None, note='log without traj_age_ms (firmware before RB-4)')
    elif end is None or not win_ok:
        res['cond3'] = dict(value_ms=None, no_evidence=True)
    else:
        by_seq = {r['_seq']: r for r in rows}

        def L_of(r):
            nx = by_seq.get(r['_seq'] + 1)
            ok = nx is not None and nx['_mo'] and nx['_t'] > r['_t']      # L <= 0 is no bound (CP4-2)
            return (nx['_t'] - r['_t']) if ok else None

        mo_end = end
        while mo_end + 1 < len(rows) and rows[mo_end + 1]['_mo']:
            mo_end += 1
        mo_rows = rows[lo:mo_end + 1]
        start = None
        for r in mo_rows:
            Lr = L_of(r)
            if r['_seq'] <= rows[end]['_seq'] and Lr is not None and r['_t'] + Lr <= t0:
                start = r
        # end anchor: first line at or after t_w1 usable as a LATER anchor (has L)
        fin = next((r for r in mo_rows if r['_t'] >= t_w1 and L_of(r) is not None), None)
        if start is None or fin is None:
            why = 'no start anchor before the window' if start is None else 'no end anchor with a next line'
            res['cond3'] = dict(value_ms=None, no_evidence=True, note=why)
            res['flags'].append(f'cond3_no_evidence: {why}')
        else:
            chain = [start] + [r for r in mo_rows
                               if start['_seq'] < r['_seq'] <= fin['_seq'] and L_of(r) is not None]
            worst, at, neg, stale = -math.inf, None, 0, None
            for pa, pb in zip(chain[:-1], chain[1:]):
                age = pa['_age']
                if age is None:                 # empty cell: no evidence for this pair (CP1-2)
                    res['flags'].append(f'cond3_age_missing: empty traj_age_ms at log_seq {pa["_seq"]}')
                    at = None
                    break
                if age < 0:                     # -1: no packet since boot; any later packet came after t_a
                    neg += 1
                    age = 0
                elif pa is start and _packet_before_log(rows, start):
                    # Liam 23/11 CODE_CAM local_f3c5357d, box "Dieu (3)": "Sua (Recommended)"; condition narrowed after
                    # Codex Sol C3N1-1 (round 1): the start anchor age is that of a packet received before the log
                    # began (the run before; the schedule starts after the logger, >= 6.15 s still), not of this
                    # schedule ("from the first packet", O1 3.5 (3)): as -1, any packet of the schedule came after
                    # t_a, so the pair bound counts from t_a (an upper bound over the window from the edge). A
                    # schedule packet lost from the start makes the edge late: a phase error, condition (4).
                    stale = age
                    age = 0
                b3 = age + 1.0 + (pb['_t'] - pa['_t']) / 1e3 + L_of(pb) / 1e3
                if b3 > worst:
                    worst, at = b3, [pa['_seq'], pb['_seq']]
            res['cond3'] = dict(value_ms=worst if at else None, at_seq=at, start_anchor_seq=start['_seq'],
                                end_anchor_seq=fin['_seq'], n_pairs=len(chain) - 1, age_minus1_as_0=neg,
                                start_age_run_before_ms=stale)
    res['decided'] = True
    return res


def lost(res):
    return bool(res['flags'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--events', default=None, help='default <log>.events.txt when it exists')
    ap.add_argument('--sender', default=None, help='sender log of the run (pc_sender stdout, #test_end)')
    ap.add_argument('--allow-final-set', action='store_true')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    try:
        task = Task(a.task, a.allow_final_set)
        rows, ev_in, dup, has_age = read_log(a.log)
        ev = ev_in + read_events(a.events or (a.log + '.events.txt'))
        res = analyse(rows, task, ev, has_age, read_sender(a.sender, a.task), read_sender_bao(a.sender, a.task),
                      read_sender_complete(a.sender, a.task))
        res.update(log=os.path.basename(a.log), n_lines=len(rows), n_duplicates=dup,
                   thresholds='not locked before C8 (O1 3.5): values reported only')
        if a.json:
            if os.path.exists(a.json):
                raise FileExistsError(a.json)
            with open(a.json, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=1)
        c2, c3, c4 = res.get('cond2', {}), res.get('cond3', {}), res.get('cond4', {})
        print(f'{a.task}: canh len log_seq {res.get("edge_seq")}, {res.get("n_anchor")} neo, '
              f'{res.get("n_gap_in_window")} khoang ho')
        for name, c in (('(2) lech lan ', c2), ('(4) lech lich', c4)):
            if c.get('anchor_max_m') is not None:
                print(f'  {name}  {c["value_m"] * 1000:.2f} mm (tren dong {c["anchor_max_m"] * 1000:.2f}; '
                      f'can moi cap, chi tham khao {c["value_all_m"] * 1000:.2f})'
                      + ('' if c is c2 or c.get('applies') else '  [(4) chua ap duoc]'))
        if c3.get('value_ms') is not None:
            print(f'  (3) tuoi goi       {c3["value_ms"]:.1f} ms (cap neo {c3["at_seq"]})')
        for fl in res['flags']:
            print(f'  !! {fl}')
        for nt in res['notes'] + [c.get('note') for c in (c2, c3, c4) if c.get('note')]:
            print(f'  -- {nt}')
        rc = 1 if lost(res) else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'LOI: {exc}')
        rc = 2
    print(f'#ket_qua,ma={rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
