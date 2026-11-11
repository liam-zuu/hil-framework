"""
o1_lanes.py: lanes and scoring of the O1 final-test tasks K1, K2 (O1 v11 section 3.7) on a pose
log in the START-POSE frame (x forward, y left of the car as it was set down; heading 0 = the
heading it was set down with). CODE session local_7787e22f, 11/11/2026 (Liam: "Duyệt, làm theo
thứ tự"; EA-11 of the B8 early review: no K1, K2 scorer existed).

Lanes (from the command of rpi5/o1_tasks.py, body-frame velocities, no body rotation, so the lane
follows the start pose; O1 v11.2 A-R2-1: the lane is built on the NOMINAL start pose, fixed
before the first run; the real placement error is measured, the lane is not rebuilt):
  O1-K1  straight 2 m to the left: (0, 0) -> (0, K1_LEN)
  O1-K2  square of side 1 m: forward, left, back, right (o1_tasks.K2_DIRS), ends on the start corner

evaluate_k(log, task) follows fwsim/o1_feasibility.evaluate_v7 (the S03 section 4 rule, the one
S05 v3.7 section 4.6 asks for HIL runs, EA-13) with the two differences of O1 3.7:
  (1) the lane is a straight segment (K1) or a 4-side polyline (K2); progress is arc length;
      standing at a corner the progress does not grow;
  (2) e_theta is taken against the START heading (the command heading never changes).
Tracker, completion conditions 1-5, deadline T_cmd + t_late, early close only while condition 1
can be trusted, y1 true / false / inconclusive: as evaluate_v7, with the same delta, v_d, t_d,
D, t_late. [CÁCH HIỂU, not written in O1 3.7] condition 2 "centre within D of the origin" of the
figure 8 is read as "within D of the END point of the lane" (K1: (0, 2) m; K2: the start corner);
e_s, e_trk use the command point s_cmd(t) on the lane (s_cmd = integral of the commanded speed);
v_max of the window is K_V = 0.5 m/s.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import o1_tasks as ot                                                    # noqa: E402
from o1_feasibility import D, T_LATE, DELTA, V_D, wrap, _held            # noqa: E402

K_TASKS = ot.FINAL_SET


def k_corners(task):
    """Corner points of the lane of K1 / K2 in the start-pose frame."""
    if task == 'O1-K1':
        return [(0.0, 0.0), (0.0, ot.K1_LEN)]
    if task == 'O1-K2':
        pts, x, y = [(0.0, 0.0)], 0.0, 0.0
        for ux, uy in ot.K2_DIRS:
            x, y = x + ux * ot.K2_SIDE, y + uy * ot.K2_SIDE
            pts.append((round(x, 12), round(y, 12)))
        return pts
    raise ValueError(f'{task!r}: one of {K_TASKS}')


class Lane:
    """Polyline lane sampled finely by arc length (same interface as o1_feasibility.Path8)."""

    def __init__(self, corners, step=1e-4):
        pts = np.asarray(corners, dtype=float)
        xs, ys, hs = [], [], []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(math.ceil(math.hypot(*(b - a)) / step)) + 1)
            u = np.linspace(0.0, 1.0, n)
            seg = a + (b - a) * u[:, None]
            if xs:
                seg = seg[1:]
            xs.append(seg[:, 0])
            ys.append(seg[:, 1])
            hs.append(np.full(len(seg), math.atan2(b[1] - a[1], b[0] - a[0])))
        self.x, self.y, self.head = np.concatenate(xs), np.concatenate(ys), np.concatenate(hs)
        self.s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])
        self.S = float(self.s[-1])
        self.end = (float(self.x[-1]), float(self.y[-1]))

    def point_at(self, s):
        """(x, y, heading) of the lane at arc length s (clamped to [0, S])."""
        i = int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))
        return float(np.interp(s, self.s, self.x)), float(np.interp(s, self.s, self.y)), float(self.head[i])

    def nearest(self, px, py, s_lo, s_hi):
        """Nearest lane point with s in [s_lo, s_hi]: (s*, dist, heading, at_upper_end)."""
        i0 = int(np.searchsorted(self.s, max(s_lo, 0.0), side='left'))
        i1 = int(np.searchsorted(self.s, min(s_hi, self.S), side='right'))
        i1 = max(i1, i0 + 1)
        d = np.hypot(self.x[i0:i1] - px, self.y[i0:i1] - py)
        j = int(np.argmin(d))
        upper = (i0 + j == i1 - 1) and s_hi < self.S
        return float(self.s[i0 + j]), float(d[j]), float(self.head[i0 + j]), upper

    def distance_to_path(self, px, py):
        from scipy.spatial import cKDTree
        if not hasattr(self, '_tree'):
            self._tree = cKDTree(np.column_stack([self.x, self.y]))
        return self._tree.query(np.column_stack([px, py]))[0]


class KSchedule:
    """Command of K1 / K2 (rpi5/o1_tasks.task): T_cmd, v_max, s_cmd(t) = integral of the speed."""

    def __init__(self, task, dt=1e-4):
        self.task = task
        self.Tcmd, self.f = ot.task(task, allow_final_set=True)
        self.vmax = ot.K_V
        self.v = ot.K_V
        tt = np.arange(0.0, self.Tcmd + dt, dt)
        sp = np.array([math.hypot(*self.f(t)[:2]) for t in tt])
        self._t = tt
        self._s = np.concatenate([[0.0], np.cumsum((sp[1:] + sp[:-1]) / 2.0 * dt)])

    def s_cmd(self, t):
        return float(np.interp(t, self._t, self._s))


def evaluate_k(log, task, lane=None, sch=None, lsum=0.24):
    """K1 / K2 on a pose log (start-pose frame), the rules of the module docstring.
    log: t [s, 0 = command start], pose (n, 3) x, y, theta, v_true (n, 3) vx, vy, wz,
    duty (n, 4), pwm (n, 4). Returns (summary dict, per-frame rows)."""
    lane = lane or Lane(k_corners(task))
    sch = sch or KSchedule(task)
    t_all = np.asarray(log['t'], dtype=float)
    keep = t_all < sch.Tcmd + T_LATE - 1e-9
    t = t_all[keep]
    pose = np.asarray(log['pose'])[keep]
    fx, fy, psi = pose[:, 0], pose[:, 1], pose[:, 2]
    vt = np.asarray(log['v_true'])[keep]
    speed = np.hypot(vt[:, 0], vt[:, 1])
    omega_d = V_D / lsum
    d_end = np.hypot(fx - lane.end[0], fy - lane.end[1])
    held = _held(t, (speed < V_D) & (np.abs(vt[:, 2]) < omega_d) & (d_end <= D))
    c25 = held & (t >= sch.Tcmd - 1e-9)
    rows, s_prev, t_prev = [], 0.0, 0.0
    first_trk, t_prog, done = None, None, False
    n_upper = n_back = 0
    for k in range(len(t)):
        dk = 1.5 * sch.vmax * (t[k] - t_prev) + DELTA
        s_star, _, _, upper = lane.nearest(fx[k], fy[k], s_prev - DELTA, s_prev + dk)
        s_two, e_perp, _, _ = lane.nearest(fx[k], fy[k], s_prev - dk, s_prev + dk)
        back = bool(s_two < s_prev - DELTA - 1e-12)
        trk = bool(upper or back)
        n_upper += int(upper)
        n_back += int(back)
        if trk and first_trk is None:
            first_trk = t[k]
        s_car = max(s_prev, s_star)
        s_c = sch.s_cmd(t[k])
        px, py, hc = lane.point_at(s_c)
        ex, ey = fx[k] - px, fy[k] - py
        rows.append(dict(t=t[k], x=fx[k], y=fy[k], psi=psi[k], s_cmd=s_c, s_car=s_car, s_star=s_star,
                         e_perp=e_perp, e_theta_start=wrap(psi[k]), e_s=-(ex * math.cos(hc) + ey * math.sin(hc)),
                         e_trk=math.hypot(ex, ey), dist_end=d_end[k], speed=speed[k], wz=vt[k, 2],
                         track_err=int(trk)))
        s_prev, t_prev = s_car, t[k]
        if t_prog is None and s_car >= lane.S - D:
            t_prog = t[k]
        trusted = first_trk is None or (t_prog is not None and t_prog < first_trk)
        if t_prog is not None and c25[k] and trusted:
            done = True
            break
    t_end = rows[-1]['t']
    if first_trk is None or (t_prog is not None and t_prog < first_trk):
        y1 = 'true' if done else 'false'
    else:
        y1 = 'inconclusive' if c25.any() else 'false'
    e = {key: np.array([r[key] for r in rows]) for key in ('e_perp', 'e_theta_start', 'e_s', 'e_trk')}
    ds = np.array([r['s_cmd'] - r['s_car'] for r in rows])
    m = t_all <= t_end + 1e-9
    tm = t_all[m]
    duty, pwm = np.asarray(log['duty'])[m], np.abs(np.asarray(log['pwm'])[m]) >= 1000
    e_perp_max = float(e['e_perp'].max())
    last = rows[-1]
    out = dict(
        task=task, v=sch.v, T_cmd=sch.Tcmd, t_deadline=sch.Tcmd + T_LATE, t_end=t_end, completed=y1,
        status='track_err' if first_trk is not None else 'ok',
        track_err_frames=int(sum(r['track_err'] for r in rows)), n_upper_frames=n_upper, n_back_frames=n_back,
        t_first_track_err=first_trk if first_trk is not None else np.nan,
        t_progress=t_prog if t_prog is not None else np.nan, cond2_5_ever=bool(c25.any()),
        e_perp_max=e_perp_max, e_perp_max_le_D=bool(e_perp_max <= D),
        e_perp_global_max=float(lane.distance_to_path(fx[:len(rows)], fy[:len(rows)]).max()),
        e_theta_start_max_deg=float(np.degrees(np.abs(e['e_theta_start']).max())),
        e_s_absmax=float(np.abs(e['e_s']).max()), ds_absmax=float(np.abs(ds).max()),
        e_trk_max=float(e['e_trk'].max()), e_perp_rms=float(np.sqrt(np.mean(e['e_perp'] ** 2))),
        tv_duty=float(np.sum(np.abs(np.diff(duty, axis=0))) / max(tm[-1] - tm[0], 1e-9)) if len(tm) > 1 else 0.0,
        sat_frac=float(np.mean(pwm)) if len(pwm) else 0.0,
        stop_dist_end=last['dist_end'], stop_x=last['x'], stop_y=last['y'],
        stop_heading_err_deg=math.degrees(wrap(last['psi'])), s_car_end=last['s_car'], S=lane.S)
    return out, rows
