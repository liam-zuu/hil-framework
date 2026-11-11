#!/usr/bin/env python3
"""
c7_score.py: score one O1 floor run from the camera (frames.csv of c7_measure.py) and the ESP32 log,
with the camera rules of O1 v11.4 sections 3.2-3.5, 3.7. Writes score.json with the keys of
hil_o1_score.py / fwsim evaluate_v7 / o1_lanes.evaluate_k (so s05_final_table.py reads it) plus the
camera fields. L83 (Liam 11/11 CODE local_7787e22f, Fable C7 design recommendations "duyet");
Liam 11/11 CODE local_5e745270: "oke viet luon di".

Clock: t = 0 is the command start = the first ESP32 log line with run_active = 1 (S05 v3.8 4.7 (i));
the log must hold exactly one run_active rising edge (one file = one run, CG7). The video clock is
matched to the log clock by o1_time_sync.sync (C7b method A, t_log = t_cam + D), or --offset-s.
Command time of a frame: t = t_cam + D - t0_log.

Lane (O1 3.2): figure 8 = the mathematical curve in the floor frame (Path8 of fwsim/o1_feasibility);
K1, K2 = o1_lanes lanes built on the NOMINAL start pose --k-start X,Y,HEAD_DEG (locked at C8, written
before the first run, O1 3.2 EA-11); the camera pose is moved into that start-pose frame.

Frames used: VALID frames of c7_measure (gates O1 3.3 O-2). Gap = time between two consecutive valid
frames inside the run window (including the last valid frame before t = 0 and the window end).
Tracker (O1 3.3, camera version; differences to evaluate_v7 written in O1):
  - window [s_prev - delta, s_prev + D_k], D_k = 1.5 v_max dt + delta, dt capped at t_miss; a gap
    above t_miss makes the frame that ends it a TRACKING ERROR;
  - solution on the upper end -> search again in [s_prev - delta, s_prev + 1.23 m] (W11-4); still on
    the upper end -> TRACKING ERROR; the two-sided window of e_perp opens to the same upper bound;
  - two-sided nearest point below s_prev - delta = "backward on the lane": a REPORT flag only
    (controller behaviour, W10-1), progress holds;
  - s only grows; window cut at s = S (the end point is not a tracking error).
Completion (O1 3.4): at one frame t: s_car >= S - D_d; t >= T_cmd; t < T_cmd + t_late; and over an
unbroken trailing t_d of valid frames: speed < v_d, |wz| < v_d / (lx + ly), centre within D_d of the
end point (figure 8: origin; K1, K2: lane end, the o1_lanes reading). Speeds from the camera.
Run outcome (O1 3.5, S05 3.4):
  hong_he_do   a gap above t_gap (t_gap < t_miss), a tracking error, camera bumped (largest mark shift
               in the window above BUMP_PX), no valid frame in the window, or the clock not matched;
  else per condition: completion -> dat / khong_dat; e_perp_max with the uncertainty U:
               e_perp_max + U <= D dat, e_perp_max - U > D khong_dat, else chua_ket_luan;
               any khong_dat -> khong_dat; else any chua_ket_luan -> chua_ket_luan; else dat.
  [CACH HIEU] completion is taken without an uncertainty band (O1 3.5 gives one only to e_perp_max).
U is an input (--u-m): it comes from C8 (design 4.3); before C8 the value used is written with --u-src.
Reported, not deciding (thresholds locked at C8): placement error before t = 0 against the nominal start;
pos_reset_n changes in the run (O1 3.5 command-path rule (1)); drift of the mark shift between the first
and the last second (design 3.6: take clip H again before the next run).

  python tools/c7_score.py --frames <dir>/frames.csv --esp32 <agv_logger csv> --task O1-F05 \
      --u-m 0.005 --u-src "C8 2026-11-xx" --out score.json [--offset-s D] [--k-start X,Y,DEG]
  (K1, K2 need --allow-final-set, O1 3.7)
Last line: #ket_qua,ma=0 scored / 2 error.
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
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c7_measure as cm                                                  # noqa: E402
import o1_lanes                                                          # noqa: E402
import o1_time_sync                                                      # noqa: E402
from o1_feasibility import (D, DELTA, HEAD0, T_D, T_LATE, V_D, Path8, Schedule, _held,  # noqa: E402
                            path_point, wrap)
from o1_tasks import FIG8, FINAL_SET                                     # noqa: E402

LSUM = 0.24            # lx + ly [m], omega_d = v_d / LSUM (O1 3.4, params.py)
CAP = 1.23             # m, upper bound of the reopened window (O1 3.3 W11-1, W11-4)
T_MISS = 0.68          # s, O1 3.3 (v11.4 keeps 0.68 s) [chot o C8, <= 0.69 s]
T_GAP = 0.10           # s, O1 3.3 O-3 [DE XUAT, khoa o C8]
BUMP_PX = 12.0         # px, O1 3.5 O-4 "camera bumped" [DE XUAT, khoa o C8]
DRIFT_PX = 1.5         # px, design 3.6: mean shift of the last 1 s vs the first 1 s
PRE_START_S = 0.5      # s, placement error from the valid frames of the last 0.5 s before t = 0
OUTCOMES = ('dat', 'khong_dat', 'chua_ket_luan', 'hong_he_do')


# ---------------------------------------------------------------- ESP32 log
def read_log(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith('#'))]
    out = [r for r in rows if r.get('timestamp_us') not in (None, '')]
    if not out:
        raise ValueError(f'{path}: no data line')
    return out


def command_start_us(rows):
    """timestamp_us of the first line with run_active = 1 after one with 0; exactly one rising edge."""
    ra = [r.get('run_active') == '1' for r in rows]
    rises = [i for i in range(1, len(ra)) if ra[i] and not ra[i - 1]]
    if ra and ra[0]:
        raise ValueError('the first log line already has run_active = 1: no rising edge')
    if len(rises) != 1:
        raise ValueError(f'{len(rises)} run_active rising edges in the log (one file = one run)')
    return float(rows[rises[0]]['timestamp_us']), rises[0]


def log_speed(rows):
    t = np.array([float(r['timestamp_us']) for r in rows]) / 1e6
    v = np.hypot([float(r['vx']) for r in rows], [float(r['vy']) for r in rows])
    return t, v


# ---------------------------------------------------------------- lanes
class Fig8Task:
    def __init__(self, task):
        self.task, self.path, self.sch = task, Path8(), Schedule(FIG8[task])
        self.Tcmd, self.vmax, self.v, self.S = self.sch.Tcmd, self.sch.vmax, self.sch.v, self.path.S
        self.end = (0.0, 0.0)
        self.start = (0.0, 0.0, HEAD0)

    def to_lane(self, x, y, psi):
        return x, y, psi                                    # the floor frame is the lane frame (O1 3.2)

    def command(self, t):
        tau, _ = self.sch.tau(t)
        px, py, hc = path_point(tau)
        return float(self.path.s_of_tau(tau)), px, py, hc


class KTask:
    def __init__(self, task, start):
        self.task, self.path, self.sch = task, o1_lanes.Lane(o1_lanes.k_corners(task)), o1_lanes.KSchedule(task)
        self.Tcmd, self.vmax, self.v, self.S = self.sch.Tcmd, self.sch.vmax, self.sch.v, self.path.S
        self.end = self.path.end
        self.start = tuple(start)

    def to_lane(self, x, y, psi):
        x0, y0, h0 = self.start
        c, s = math.cos(h0), math.sin(h0)
        dx, dy = x - x0, y - y0
        return c * dx + s * dy, -s * dx + c * dy, wrap(psi - h0)

    def command(self, t):
        s_c = self.sch.s_cmd(t)
        px, py, hc = self.path.point_at(s_c)
        return s_c, px, py, hc


# ---------------------------------------------------------------- scoring
def evaluate_camera(t, x, y, psi, speed, wz, lane, shift=None):
    """Camera tracker and completion on the VALID frames (t: command time, increasing; pose in the lane
    frame). t may start before 0: frames before 0 only give the previous-frame time of the first window
    frame. Returns (summary, rows)."""
    t, x, y, psi = (np.asarray(a, float) for a in (t, x, y, psi))
    speed, wz = np.asarray(speed, float), np.asarray(wz, float)
    deadline = lane.Tcmd + T_LATE
    pre = np.nonzero(t < 0.0)[0]
    t_prev = float(t[pre[-1]]) if len(pre) else 0.0
    win = np.nonzero((t >= 0.0) & (t < deadline - 1e-9))[0]
    if not len(win):
        raise ValueError('no valid camera frame in the run window')
    tw = t[win]
    d_end = np.hypot(x[win] - lane.end[0], y[win] - lane.end[1])
    still = (np.nan_to_num(speed[win], nan=np.inf) < V_D) & (np.abs(np.nan_to_num(wz[win], nan=np.inf)) < V_D / LSUM)
    held = _held(tw, still & (d_end <= D))
    c25 = held & (tw >= lane.Tcmd - 1e-9)
    path = lane.path
    rows, s_prev = [], 0.0
    first_trk, t_prog, done = None, None, False
    n_upper = n_back = n_reopen = n_miss = 0
    gaps = []
    for j, k in enumerate(win):
        dt = float(t[k] - t_prev)
        gaps.append(dt)
        miss = dt > T_MISS
        dt = min(dt, T_MISS)
        dk = 1.5 * lane.vmax * dt + DELTA
        hi = s_prev + dk
        s_star, _, head, upper = path.nearest(x[k], y[k], s_prev - DELTA, hi)
        reopen = False
        if upper:                                            # W11-4: open to the cap, search again
            reopen = True
            hi = s_prev + CAP
            s_star, _, head, upper = path.nearest(x[k], y[k], s_prev - DELTA, hi)
        s_two, e_perp, _, _ = path.nearest(x[k], y[k], s_prev - dk, hi)
        back = bool(s_two < s_prev - DELTA - 1e-12)
        trk = bool(upper or miss)
        n_upper += int(upper)
        n_back += int(back)
        n_reopen += int(reopen)
        n_miss += int(miss)
        if trk and first_trk is None:
            first_trk = float(t[k])
        s_car = max(s_prev, s_star)
        s_c, px, py, hc = lane.command(float(t[k]))
        ex, ey = x[k] - px, y[k] - py
        rows.append(dict(t=float(t[k]), x=float(x[k]), y=float(y[k]), psi=float(psi[k]), s_cmd=s_c, s_car=s_car,
                         s_star=s_star, e_perp=float(e_perp), e_theta_lane=wrap(psi[k] - head),
                         e_theta_cmd=wrap(psi[k] - hc), e_theta_start=wrap(psi[k]),
                         e_s=-(ex * math.cos(hc) + ey * math.sin(hc)), e_trk=math.hypot(ex, ey),
                         dist_end=float(d_end[j]), speed=float(speed[k]), wz=float(wz[k]), track_err=int(trk),
                         back=int(back), reopen=int(reopen), gap=dt if not miss else float(t[k] - t_prev),
                         shift_px=float(shift[k]) if shift is not None else float('nan')))
        s_prev, t_prev = s_car, float(t[k])
        if t_prog is None and s_car >= path.S - D:
            t_prog = float(t[k])
        trusted = first_trk is None or (t_prog is not None and t_prog < first_trk)
        if t_prog is not None and c25[j] and trusted:
            done = True
            break
    t_end = rows[-1]['t'] if done else deadline
    if not done:                                             # window runs to the deadline: last gap to it
        gaps.append(deadline - rows[-1]['t'])
    if first_trk is None or (t_prog is not None and t_prog < first_trk):
        y1 = 'true' if done else 'false'
    else:
        y1 = 'inconclusive' if c25.any() else 'false'
    e = {kk: np.array([r[kk] for r in rows]) for kk in ('e_perp', 'e_theta_lane', 'e_theta_cmd', 'e_theta_start',
                                                        'e_s', 'e_trk')}
    ds = np.array([r['s_cmd'] - r['s_car'] for r in rows])
    small = [g for g in gaps if g > 0]
    last = rows[-1]
    out = dict(
        task=lane.task, v=lane.v, T_cmd=lane.Tcmd, t_deadline=deadline, t_end=t_end, completed=y1,
        status='track_err' if first_trk is not None else 'ok',
        track_err_frames=int(sum(r['track_err'] for r in rows)), n_upper_frames=n_upper, n_back_frames=n_back,
        n_reopen_frames=n_reopen, n_miss_frames=n_miss,
        t_first_track_err=first_trk if first_trk is not None else float('nan'),
        t_progress=t_prog if t_prog is not None else float('nan'), cond2_5_ever=bool(c25.any()),
        e_perp_max=float(e['e_perp'].max()), t_e_perp_max=rows[int(np.argmax(e['e_perp']))]['t'],
        e_perp_max_at=[rows[int(np.argmax(e['e_perp']))]['x'], rows[int(np.argmax(e['e_perp']))]['y']],
        e_perp_rms=float(np.sqrt(np.mean(e['e_perp'] ** 2))),
        e_s_absmax=float(np.abs(e['e_s']).max()), ds_absmax=float(np.abs(ds).max()),
        e_trk_max=float(e['e_trk'].max()), e_trk_rms=float(np.sqrt(np.mean(e['e_trk'] ** 2))),
        gap_max=float(max(small)) if small else 0.0, n_gaps_over_frame=None,
        gaps_total_s=None, frames_scored=len(rows), S=path.S, s_car_end=last['s_car'],
        stop_x=last['x'], stop_y=last['y'], stop_dist_end=last['dist_end'])
    if isinstance(lane, Fig8Task):
        out.update(e_theta_lane_max_deg=float(np.degrees(np.abs(e['e_theta_lane']).max())),
                   e_theta_cmd_max_deg=float(np.degrees(np.abs(e['e_theta_cmd']).max())),
                   stop_heading_err_deg=math.degrees(wrap(last['psi'] - HEAD0)))
    else:
        out.update(e_theta_start_max_deg=float(np.degrees(np.abs(e['e_theta_start']).max())),
                   stop_heading_err_deg=math.degrees(wrap(last['psi'])))
    return out, rows, gaps


def gap_summary(gaps, frame_dt):
    """Gaps longer than 1.5 frame periods: count and total time (O1 3.3: reported beside the result)."""
    big = [g for g in gaps if g > 1.5 * frame_dt]
    return len(big), float(sum(big)), float(max(gaps)) if gaps else 0.0


def outcome(summary, U, gap_max, bumped, has_trk):
    """(outcome, reasons, per-condition results) by O1 3.5."""
    reasons = []
    if gap_max > T_GAP + 1e-12:
        reasons.append(f'khoang trong {gap_max:.3f} s > t_trong {T_GAP:g} s')
    if has_trk:
        reasons.append('loi theo doi')
    if bumped:
        reasons.append(f'may quay bi va (dich moc > {BUMP_PX:g} px)')
    if reasons:
        return 'hong_he_do', reasons, {}
    cond = {}
    cond['hoan_thanh'] = 'dat' if summary['completed'] == 'true' else 'khong_dat'
    e = summary['e_perp_max']
    cond['e_perp_max'] = 'dat' if e + U <= D + 1e-12 else ('khong_dat' if e - U > D else 'chua_ket_luan')
    if 'khong_dat' in cond.values():
        return 'khong_dat', [k for k, v in cond.items() if v == 'khong_dat'], cond
    if 'chua_ket_luan' in cond.values():
        return 'chua_ket_luan', [k for k, v in cond.items() if v == 'chua_ket_luan'], cond
    return 'dat', [], cond


def placement(t, x, y, psi, lane):
    """Mean pose of the valid frames in [-PRE_START_S, 0) against the nominal start (O1 3.2)."""
    m = (t >= -PRE_START_S) & (t < 0.0)
    if not m.any():
        return dict(placement_n=0, placement_pos_mm=None, placement_heading_deg=None)
    x0, y0, h0 = lane.start
    mx, my = float(np.mean(x[m])), float(np.mean(y[m]))
    mpsi = float(np.angle(np.mean(np.exp(1j * psi[m]))))
    return dict(placement_n=int(m.sum()), placement_pos_mm=math.hypot(mx - x0, my - y0) * 1000,
                placement_heading_deg=math.degrees(wrap(mpsi - h0)), placement_xy=[mx, my])


def score(frames, log_rows, task, U, offset_s=None, k_start=None, allow_final_set=False):
    if task not in FIG8 and task not in FINAL_SET:
        raise ValueError(f'task {task!r}: one of {tuple(FIG8) + FINAL_SET}')
    if task in FINAL_SET and not allow_final_set:
        raise ValueError(f'{task} is the final test set (O1 3.7): only with --allow-final-set')
    if task in FINAL_SET and k_start is None:
        raise ValueError(f'{task}: give --k-start X,Y,HEAD_DEG (nominal start pose, locked at C8)')
    lane = Fig8Task(task) if task in FIG8 else KTask(task, k_start)
    t0_us, i0 = command_start_us(log_rows)
    valid = [r for r in frames if r['valid']]
    if not valid:
        raise ValueError('no valid camera frame in the clip')
    t_cam = np.array([r['t_s'] for r in frames])
    sync = None
    if offset_s is None:
        tl, vl = log_speed(log_rows)
        vc = np.array([r['speed'] if r['valid'] else np.nan for r in frames])
        sync = o1_time_sync.sync(t_cam, vc, tl, vl)
        offset_s = sync.offset_s
    t_all = t_cam + offset_s - t0_us / 1e6
    vi = [k for k, r in enumerate(frames) if r['valid']]
    tv = t_all[vi]
    pose = [lane.to_lane(frames[k]['x'], frames[k]['y'], frames[k]['psi']) for k in vi]
    x = np.array([p[0] for p in pose])
    y = np.array([p[1] for p in pose])
    psi = np.array([p[2] for p in pose])
    sp = np.array([frames[k]['speed'] for k in vi])
    wz = np.array([frames[k]['wz'] for k in vi])
    shift = np.array([frames[k]['shift_px'] for k in vi])
    out, rows, gaps = evaluate_camera(tv, x, y, psi, sp, wz, lane, shift)
    frame_dt = float(np.median(np.diff(t_cam))) if len(t_cam) > 1 else 1 / 60
    n_big, tot_big, gmax = gap_summary(gaps, frame_dt)
    out.update(gap_max=gmax, n_gaps_over_frame=n_big, gaps_total_s=tot_big)
    # camera bumped: largest mark shift of ANY frame (valid or not) inside the window [0, t_end]
    inwin = (t_all >= 0.0) & (t_all <= out['t_end'] + 1e-9)
    sh_all = np.array([r['shift_px'] for r in frames])[inwin]
    sh_all = sh_all[np.isfinite(sh_all)]
    shift_max = float(sh_all.max()) if len(sh_all) else float('nan')
    bumped = bool(shift_max > BUMP_PX) if len(sh_all) else False
    sw = np.array([r['shift_px'] for r in rows])
    tw = np.array([r['t'] for r in rows])
    drift = None
    if len(rows) and tw[-1] - tw[0] >= 2.0:
        a, b = sw[tw <= tw[0] + 1.0], sw[tw >= tw[-1] - 1.0]
        drift = float(abs(np.nanmean(b) - np.nanmean(a)))
    res, reasons, cond = outcome(out, U, gmax, bumped, out['track_err_frames'] > 0)
    end_us = t0_us + out['t_end'] * 1e6
    in_run = [r for r in log_rows[i0:] if float(r['timestamp_us']) <= end_us + 1e-3]
    resets = sorted({r.get('pos_reset_n') for r in in_run if r.get('pos_reset_n') not in (None, '')})
    out.update(placement(tv, x, y, psi, lane))
    out.update(outcome=res, outcome_reasons=reasons, conditions=cond, U_m=U, D=D,
               e_perp_max_le_D=bool(out['e_perp_max'] <= D),
               shift_max_px=shift_max, bumped=bumped, shift_drift_px=drift,
               shift_drift_note=('quay lai clip H truoc luot sau (design 3.6)' if drift is not None and drift > DRIFT_PX else ''),
               t0_log_us=t0_us, offset_s=offset_s, offset_src='--offset-s' if sync is None else 'o1_time_sync',
               sync=None if sync is None else dict(corr=sync.corr, offset_start_s=sync.offset_start_s,
                                                    offset_end_s=sync.offset_end_s, start_end_diff_s=sync.start_end_diff_s),
               pos_reset_n_values=resets, pos_reset_mid=len(resets) > 1,
               frames_total=len(frames), frames_valid=len(vi),
               thresholds=dict(delta=DELTA, t_miss=T_MISS, t_gap=T_GAP, bump_px=BUMP_PX, v_d=V_D, t_d=T_D,
                               t_late=T_LATE, cap=CAP, status='t_miss, t_gap, bump_px, v_d: khoa o C8'),
               k_start=list(lane.start), rule='O1 v11.4 3.2-3.5, 3.7 (camera); c7_score.py')
    return out, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--esp32', required=True, help='agv_logger CSV of the run (one run_active edge)')
    ap.add_argument('--task', required=True)
    ap.add_argument('--u-m', type=float, required=True, help='uncertainty U of e_perp_max [m] (C8, design 4.3)')
    ap.add_argument('--u-src', required=True, help='where U comes from (e.g. "C8 2026-11-13 run01")')
    ap.add_argument('--offset-s', type=float, default=None, help='t_log = t_cam + D, skips o1_time_sync')
    ap.add_argument('--k-start', default=None, help='X,Y,HEAD_DEG nominal start pose of K1/K2 (floor frame)')
    ap.add_argument('--allow-final-set', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--rows', default='', help='optional per-frame CSV of the scored frames')
    a = ap.parse_args(argv)
    try:
        if a.u_m < 0:
            raise ValueError('--u-m must be >= 0')
        ks = None
        if a.k_start:
            x0, y0, h = cm._pair(a.k_start, 3, '--k-start')
            ks = (x0, y0, math.radians(h))
        out, rows = score(cm.read_frames(a.frames), read_log(a.esp32), a.task, a.u_m, a.offset_s, ks,
                          a.allow_final_set)
        out.update(U_src=a.u_src, frames_file=os.path.basename(a.frames), frames_sha256=cm.sha256(a.frames),
                   esp32_file=os.path.basename(a.esp32), esp32_sha256=cm.sha256(a.esp32))
    except (ValueError, KeyError, OSError, argparse.ArgumentTypeError) as exc:
        print('c7_score:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    text = json.dumps(cm._json(out), indent=1, ensure_ascii=False)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    if a.rows and rows:
        with open(a.rows, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(text)
    print(f'#ket_qua,ma=0,outcome={out["outcome"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
