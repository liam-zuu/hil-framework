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
(blk_ticks, cut_ticks) decides whether (4) applies (no ultrasonic detect or block tick).

Rising edge (O1 3.4): first data line with run_active = 1 whose line just before by log_seq was
received and has run_active = 0. That line lost: the edge is unknown (run lost to the measurement
system, edge_unknown). Exactly one rising edge in the log.

Window of the schedule on the ESP32 clock (Liam Q1): start t_w0 = timestamp_us of the edge line;
end t_w1 = t_w0 + (n_sched - 1)/50 s x (1 + E_MAX) + T_SEND, n_sched = schedule packets of the
task (rpi5/pc_sender.grid_counts), E_MAX = 2002e-6 the Pi/ESP32 clock-rate bound of S05 4.7 (i),
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
            b+1 (next log_seq, motor on) exists: L = t_(b+1) - t_b.
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
Packet before the edge (task 2 of CODE, with C6): any line before the edge (motor on or off) with
0 <= traj_age_ms < 200 (a fresh packet that did not start the run) or a non-zero command
(sp_vx, sp_vy, sp_wz) -> pre_edge_packet; edge line traj_age_ms above EDGE_AGE_MS -> edge_age_high.
On the HIL bench the H7 relays packets all the time, so pre_edge_packet is expected there.

  python tools/o1_cmd_path.py --log <run.csv> --task O1-F05 [--events <run.csv.events.txt>]
                              [--allow-final-set] [--json out.json]
Last line: #ket_qua,ma=0 no flag / 1 run lost to the command path or to the measurement system
(flags in the JSON) / 2 error.
"""
from __future__ import annotations

import argparse
import csv
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
COORD_MAX_M = 8.0             # O1 3.5: rounding term valid for |x_ref|, |y_ref| < 8 m
COORD_DOUBLE_M = 4.0          # from 4 m the float32 term doubles (O1 3.5, B81-18)
R_LOG_M = 1e-4                # 0.1 mm: log 4 decimals per coordinate (logger.c 223, 249-250)
R_DT_REL = 1e-6               # float dt
R_TICK_M = 2e-7               # 2e-4 mm per tick (float32 accumulation)


# ---------------------------------------------------------------- log
def read_log(path):
    """(rows sorted by log_seq, '#' lines, n_duplicates). Identical duplicate lines of one log_seq
    are dropped and counted (L92); two different lines with the same log_seq is an error."""
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
    for r in out:
        r['_seq'] = int(r['log_seq'])
        r['_t'] = int(r['timestamp_us'])
        r['_ra'] = r.get('run_active') == '1'
        r['_mo'] = r.get('motor_on') == '1'
        r['_x'] = float(r['pos_x_ref'])
        r['_y'] = float(r['pos_y_ref'])
        a = r.get('traj_age_ms', '')
        r['_age'] = None if a in ('', None) else int(a)
        tk = r.get('tick', '')
        r['_tick'] = None if tk in ('', None) else int(tk)
    return out, events, dup


def read_events(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.startswith('#')]


def us_ticks(events):
    """(blk_ticks, cut_ticks) summed over '#us_end' lines (identical copies counted once), or None."""
    seen, blk, cut = set(), 0, 0
    for e in events:
        if not e.startswith('#us_end'):
            continue
        if e in seen:
            continue
        seen.add(e)
        kv = dict(p.split('=', 1) for p in e.split(',')[1:] if '=' in p)
        blk += int(kv.get('blk_ticks', 0))
        cut += int(kv.get('cut_ticks', 0))
    return (blk, cut) if seen else None


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


def analyse(rows, task, us=None):
    res = dict(task=task.id, v_cmd_max=task.v, n_sched=task.n_sched, flags=[], notes=[])
    edge, later, probs = find_edge(rows)
    res['flags'] += probs
    if edge is None:
        res['decided'] = False
        return res
    e = rows[edge]
    t0 = e['_t']
    t_w1 = t0 + int(round(((task.n_sched - 1) * T_SEND * (1 + E_MAX) + T_SEND) * 1e6))
    res.update(edge_seq=e['_seq'], t_w0_us=t0, t_w1_us=t_w1, edge_traj_age_ms=e['_age'])
    has_age = e['_age'] is not None

    # motor-on stretch around the edge
    lo = edge
    while lo > 0 and rows[lo - 1]['_mo']:
        lo -= 1
    pre = rows[lo:edge]
    # packet before the edge (task 2)
    # every line before the edge, motor on or off: a schedule started while the motor was off makes the
    # edge come mid-schedule with no motor-on line before it (one file = one run)
    pe = [r['_seq'] for r in rows[:edge] if (has_age and r['_age'] is not None and 0 <= r['_age'] < TRAJ_TIMEOUT_MS)
          or abs(float(r['sp_vx'])) + abs(float(r['sp_vy'])) + abs(float(r['sp_wz'])) > 0]
    res['pre_edge_motor_on_lines'] = len(pre)
    res['pre_edge_packet_seqs'] = pe[:20]
    if pe:
        res['flags'].append(f'pre_edge_packet: {len(pe)} lines before the edge with a fresh packet '
                            f'or a non-zero command')
    if has_age and e['_age'] is not None and (e['_age'] < 0 or e['_age'] > EDGE_AGE_MS):
        res['flags'].append(f'edge_age_high: traj_age_ms {e["_age"]} at the edge (> {EDGE_AGE_MS}, DE XUAT)')

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
    resets = sorted({r.get('pos_reset_n') for r in stretch if r['_ra']})
    res['pos_reset_n_values'] = resets
    if len(resets) > 1:
        res['flags'].append(f'cond1_pos_reset: pos_reset_n {resets} in the run')
    big = [r['_seq'] for r in stretch if max(abs(r['_x']), abs(r['_y'])) >= COORD_MAX_M]
    if big:
        res['flags'].append(f'coord_over_8m: |x_ref| or |y_ref| >= 8 m at log_seq {big[:5]}')
    gaps = sum(1 for a, b in zip(stretch[:-1], stretch[1:]) if b['_seq'] != a['_seq'] + 1)
    res['n_gap_in_window'] = gaps
    res['n_anchor'] = len(stretch)

    # (2), (4)
    can24 = end is not None and not toggled and not big
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
            nt = (b['_tick'] - a['_tick']) if (a['_tick'] is not None and b['_tick'] is not None) else                 int(math.ceil(dt / 0.005))
            hidden = b['_seq'] != a['_seq'] + 1 or nt != 1     # a lost line or a tick not logged
            bg = max(abs(a['_x']), abs(a['_y']), abs(b['_x']), abs(b['_y'])) >= COORD_DOUBLE_M
            r = round_term(task.v, dt, max(nt, 0), bg)
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
    elif end is None:
        res['cond3'] = dict(value_ms=None, no_evidence=True)
    else:
        by_seq = {r['_seq']: r for r in rows}

        def L_of(r):
            nx = by_seq.get(r['_seq'] + 1)
            return (nx['_t'] - r['_t']) if (nx is not None and nx['_mo']) else None

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
            worst, at, neg = -math.inf, None, 0
            for pa, pb in zip(chain[:-1], chain[1:]):
                age = pa['_age']
                if age is None or age < 0:     # no packet yet at a: any later packet came after t_a
                    neg += 1
                    age = 0
                b3 = age + 1.0 + (pb['_t'] - pa['_t']) / 1e3 + L_of(pb) / 1e3
                if b3 > worst:
                    worst, at = b3, [pa['_seq'], pb['_seq']]
            res['cond3'] = dict(value_ms=worst if at else None, at_seq=at, start_anchor_seq=start['_seq'],
                                end_anchor_seq=fin['_seq'], n_pairs=len(chain) - 1, age_minus1_as_0=neg)
    res['decided'] = True
    return res


def lost(res):
    return bool(res['flags'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--events', default=None, help='default <log>.events.txt when it exists')
    ap.add_argument('--allow-final-set', action='store_true')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    try:
        task = Task(a.task, a.allow_final_set)
        rows, ev_in, dup = read_log(a.log)
        ev = ev_in + read_events(a.events or (a.log + '.events.txt'))
        res = analyse(rows, task, us_ticks(ev))
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
        rc = 1 if lost(res) else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'LOI: {exc}')
        rc = 2
    print(f'#ket_qua,ma={rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
