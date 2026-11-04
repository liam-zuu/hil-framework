"""
o1_feasibility.py: feasibility check of the O1 floor task (figure 8) on fwsim.

O1 v5 (CLAUDE_THESIS 06_SPECS/2026-11-04_De_xuat_O1_nhiem_vu_san_v5.md):
  path       x = A sin tau, y = (A/2) sin 2 tau, A = 1 m (section 3.1); the car keeps
             its heading on the tangent (vx = path speed, vy = 0, wz = curvature x speed,
             as trajectory_pub.get_body_velocity 'figure8'), start heading 45 deg;
  schedule   tau_dot rises linearly 0 -> w0 = 2 pi / T in T_r, holds, falls linearly to 0
             in T_r, ends at tau = 2 pi; T = figure8_period(1, v), T_r = sqrt(2) A w0 / a_r,
             a_r = 0.5 m/s^2; T_cmd = T + T_r;
  measures   (sections 3.3, 3.4) from the TRUE plant pose (not the firmware's estimate):
             progress s along the path by arc length, nearest point searched in the window
             [s_prev - delta, s_prev + 1.5 v_max dt + delta], s only grows; e_perp = distance
             to that point; e_theta = car heading - path tangent there; e_s = s_cmd - s_car
             (both signs); window from t = 0 (command start) to the end of the run (stop
             condition met, or t_deadline = T_cmd + t_late);
  stop       (section 3.4) s >= S - D_d, centre within D_d of the origin, speed < v_d for
             t_d, before t_deadline; D_d = D = 0.10 m (temporary), t_d = 1 s, t_late = 2 s.
K1, K2 are NOT run here (O1 v5 section 3.7: final test set only).

Assumptions of this script (not decided in O1 v5): delta = 0.02 m (the plant pose is
exact here; delta in O1 is the camera's position uncertainty, fixed after the camera
trial); v_d = 0.01 m/s on the true plant speed (O1: v_d fixed after the camera trial).
Simulation numbers for feasibility only: plant slip method A with the assumed mu_hd,
IMU model noise-free, default firmware gains, no tuning; not main results.

  python fwsim/o1_feasibility.py --out <05_DATA/results/O1_FEAS/<date>_runNN> [--dll <fwsim.dll>]
         [--pos-loop KP,KI]

--pos-loop (04/11, O1 PA3, Liam approved A1 + B1 in the CODE session): turns on the firmware
position loop pos_ctrl.c (AGV feat/pos-loop) with these gains; the command pose is integrated
on the ESP32 from the same body-velocity command (A1), heading stays with heading.c (B1).
The measures are unchanged (true plant pose); log.csv gets the pos_* columns.
Since 04/11 (Liam, CODE session: "làm cách 1 ban khuyến nghị đi") the firmware turns the loop,
(a) and (d) ON by default, so a run without --pos-loop already has them; turn them off with
--fw-param pos_on=0 --fw-param pos_e_win=0 --fw-param still_on=0. The summary column pos_loop
and meta.yaml are read back from the firmware, not taken from the options.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim                                            # noqa: E402
from trajectory_pub import figure8_period               # noqa: E402

A = 1.0
A_R = 0.5            # m/s^2, O1 v5 section 3.1 (assumption approved by Liam 04/11)
D = 0.10             # m, temporary (O1 v5 section 8 item 2); D_d = D
T_D = 1.0            # s
T_LATE = 2.0         # s
DELTA = 0.02         # m, assumption of this script (see docstring)
V_D = 0.01           # m/s, assumption of this script (see docstring)
SPEEDS = (0.2, 0.5, 0.8)
CONTROLLERS = ('PID', 'ADRC')
HEAD0 = math.pi / 4  # start heading on the floor (tangent at tau = 0)


class Path8:
    """Arc-length table of the figure 8 on a fine tau grid."""

    def __init__(self, n=400001):
        self.tau = np.linspace(0.0, 2.0 * math.pi, n)
        dx, dy = A * np.cos(self.tau), A * np.cos(2.0 * self.tau)
        sp = np.hypot(dx, dy)
        self.s = np.concatenate([[0.0], np.cumsum((sp[1:] + sp[:-1]) / 2.0 * np.diff(self.tau))])
        self.S = float(self.s[-1])
        self.x, self.y = A * np.sin(self.tau), A / 2.0 * np.sin(2.0 * self.tau)
        self.head = np.arctan2(dy, dx)

    def s_of_tau(self, tau):
        return np.interp(tau, self.tau, self.s)

    def nearest(self, px, py, s_lo, s_hi):
        """Nearest path point with s in [s_lo, s_hi]: (s*, dist, tangent heading, at_upper_end)."""
        i0 = int(np.searchsorted(self.s, max(s_lo, 0.0), side='left'))
        i1 = int(np.searchsorted(self.s, min(s_hi, self.S), side='right'))
        i1 = max(i1, i0 + 1)
        d = np.hypot(self.x[i0:i1] - px, self.y[i0:i1] - py)
        j = int(np.argmin(d))
        upper = (i0 + j == i1 - 1) and s_hi < self.S
        return float(self.s[i0 + j]), float(d[j]), float(self.head[i0 + j]), upper


class Schedule:
    """tau(t), tau_dot(t) of O1 v5 section 3.1 for level v."""

    def __init__(self, v):
        self.v = v
        self.T = figure8_period(A, v)
        self.w0 = 2.0 * math.pi / self.T
        self.vmax = math.sqrt(2.0) * A * self.w0
        self.Tr = self.vmax / A_R
        self.Tcmd = self.T + self.Tr

    def tau(self, t):
        w0, Tr, Tc = self.w0, self.Tr, self.Tcmd
        if t <= 0.0:
            return 0.0, 0.0
        if t < Tr:
            return w0 * t * t / (2.0 * Tr), w0 * t / Tr
        if t <= Tc - Tr:
            return w0 * Tr / 2.0 + w0 * (t - Tr), w0
        if t < Tc:
            return 2.0 * math.pi - w0 * (Tc - t) ** 2 / (2.0 * Tr), w0 * (Tc - t) / Tr
        return 2.0 * math.pi, 0.0

    def setpoint(self, t):
        """Body velocity (vx, vy, wz): heading on the tangent, as get_body_velocity 'figure8'."""
        tau, td = self.tau(t)
        if td <= 0.0:
            return 0.0, 0.0, 0.0
        dx, dy = A * math.cos(tau), A * math.cos(2.0 * tau)
        ddx, ddy = -A * math.sin(tau), -2.0 * A * math.sin(2.0 * tau)
        n2 = dx * dx + dy * dy
        return math.sqrt(n2) * td, 0.0, (dx * ddy - dy * ddx) / n2 * td


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def evaluate(log, path, sch):
    """O1 v5 sections 3.3, 3.4 on the true plant pose. Returns (summary dict, per-frame rows)."""
    t = log['t']
    pose = log['pose']                      # plant frame: start (0, 0, 0)
    c, s_ = math.cos(HEAD0), math.sin(HEAD0)
    fx = c * pose[:, 0] - s_ * pose[:, 1]   # floor frame: rotate by the start heading
    fy = s_ * pose[:, 0] + c * pose[:, 1]
    psi = pose[:, 2] + HEAD0
    speed = np.hypot(log['v_true'][:, 0], log['v_true'][:, 1])
    t_deadline = sch.Tcmd + T_LATE
    rows, s_prev, t_prev = [], 0.0, 0.0
    n_track_err, slow_since, t_end, done = 0, None, None, False
    for k in range(len(t)):
        if t[k] > t_deadline + 1e-9:
            break
        dt = t[k] - t_prev if k else 0.0
        s_star, e_perp, head, upper = path.nearest(fx[k], fy[k], s_prev - DELTA,
                                                   s_prev + 1.5 * sch.vmax * dt + DELTA)
        n_track_err += int(upper)
        s_car = max(s_prev, s_star)
        tau_c, _ = sch.tau(t[k])
        s_cmd = float(path.s_of_tau(tau_c))
        e_theta = wrap(psi[k] - head)
        d0 = math.hypot(fx[k], fy[k])
        rows.append(dict(t=t[k], x=fx[k], y=fy[k], psi=psi[k], s_cmd=s_cmd, s_car=s_car, s_star=s_star,
                         e_perp=e_perp, e_theta=e_theta, e_s=s_cmd - s_car, dist_origin=d0,
                         speed=speed[k], track_err=int(upper)))
        s_prev, t_prev = s_car, t[k]
        at_end = s_car >= path.S - D and d0 <= D
        if at_end and speed[k] < V_D:
            slow_since = t[k] if slow_since is None else slow_since
            if t[k] - slow_since >= T_D - 1e-9:
                t_end, done = t[k], True
                break
        else:
            slow_since = None
    if t_end is None:
        t_end = rows[-1]['t']
    e = {key: np.array([r[key] for r in rows]) for key in ('e_perp', 'e_theta', 'e_s')}
    last = rows[-1]
    out = dict(
        v=sch.v, T=sch.T, T_r=sch.Tr, T_cmd=sch.Tcmd, t_deadline=t_deadline, t_end=t_end, completed=done,
        e_perp_max=float(e['e_perp'].max()), e_theta_max_deg=float(np.degrees(np.abs(e['e_theta']).max())),
        e_s_max=float(e['e_s'].max()), e_s_min=float(e['e_s'].min()), e_s_absmax=float(np.abs(e['e_s']).max()),
        e_perp_rms=float(np.sqrt(np.mean(e['e_perp'] ** 2))),
        e_theta_rms_deg=float(np.degrees(np.sqrt(np.mean(e['e_theta'] ** 2)))),
        e_s_rms=float(np.sqrt(np.mean(e['e_s'] ** 2))),
        stop_dist_origin=last['dist_origin'], stop_x=last['x'], stop_y=last['y'],
        stop_heading_err_deg=math.degrees(wrap(last['psi'] - HEAD0)),
        s_car_end=last['s_car'], S=path.S, track_err_frames=n_track_err,
        sat_frac=float(np.mean(np.abs(log['pwm'][log['t'] <= t_end]) >= 1000)),
        e_perp_max_le_D=bool(e['e_perp'].max() <= D))
    return out, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    ap.add_argument('--pos-loop', default=None, metavar='KP,KI',
                    help='turn on the firmware position loop (pos_ctrl.c, O1 PA3) with these gains; '
                         'the DLL must be built from firmware that has it')
    ap.add_argument('--fw-param', action='append', default=[], metavar='NAME=VALUE',
                    help='extra firmware parameter by name (fwsim_set_param), e.g. pos_e_win=0.005, still_on=1')
    fwsim.add_plant_imu_args(ap)
    a = ap.parse_args()
    fw_params, pos_loop = None, None
    if a.pos_loop:
        kp, ki = (float(x) for x in a.pos_loop.split(','))
        fw_params = dict(pos_on=1, pos_kp=kp, pos_ki=ki)
        pos_loop = dict(kp=kp, ki=ki)
    for item in a.fw_param:
        name, val = item.split('=', 1)
        fw_params = dict(fw_params or {}, **{name.strip(): float(val)})
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    os.makedirs(a.out, exist_ok=True)
    imu = fwsim.imu_config(a)
    fw = fwsim.FirmwareSim(a.dll)
    if pos_loop and not fw.pos_cols:
        sys.exit('--pos-loop: this fwsim.dll has no position loop (build it from AGV feat/pos-loop)')
    if fw.build_info.get('esp32', {}).get('dirty', True):
        print('WARNING: fwsim.dll was built from a dirty firmware tree; not usable as a main result')
    path = Path8()
    summary, gains, t0 = [], {}, time.time()
    pos_params = None
    for ctrl in CONTROLLERS:
        for v in SPEEDS:
            sch = Schedule(v)
            params = fwsim.plant_params(a)
            log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + T_LATE, params, seed=1, mode=ctrl, imu=imu,
                                 fw_params=fw_params)
            pos_params = fwsim.pos_meta(fw)
            gains[ctrl] = ({ax: fw.get_adrc(i) for i, ax in enumerate(['vx', 'vy', 'wz'])} if ctrl == 'ADRC'
                           else {ax: fw.get_pid(i) for i, ax in enumerate(['vx', 'vy', 'wz'])})
            res, rows = evaluate(log, path, sch)
            run_id = f'fig8_v{v:.1f}_{ctrl}'
            rd = os.path.join(a.out, run_id)
            fwsim.write_run(rd, log, res, fw, dict(controller=ctrl, v=v, a_r=A_R, imu=imu, pos_loop=pos_params),
                            params)
            with open(os.path.join(rd, 'o1_frames.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            summary.append(dict(run_id=run_id, controller=ctrl,
                                pos_loop=int(bool(pos_params and pos_params['on'])), **res))
            print(f"{run_id}: e_perp_max={res['e_perp_max']:.3f} m, e_theta_max={res['e_theta_max_deg']:.1f} deg, "
                  f"|e_s|max={res['e_s_absmax']:.3f} m, stop {res['stop_dist_origin']:.3f} m, "
                  f"completed={res['completed']}, track_err={res['track_err_frames']}")
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    b = fw.build_info
    p0 = fwsim.plant_params(a)
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: O1 v5 feasibility (06_SPECS/2026-11-04_De_xuat_O1_nhiem_vu_san_v5.md, sections 3.1, 3.3, 3.4, 8 item 2)
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
purpose: "feasibility of D = 0.10 m with default gains, no tuning; simulation numbers, not main results (plant slip method A with assumed mu_hd, not checked against the car)"
controllers: {list(CONTROLLERS)}
speeds_mps: {list(SPEEDS)}
gains: {json.dumps(gains)}     # firmware defaults, read back after each run
pos_loop: {json.dumps(pos_params)}     # pos_ctrl.c + standstill hold (on, kp, ki, v_corr_max, e_win, e_exit, still_on, v_still, t_still_us) read back after the last run; null = DLL without pos_ctrl.c
fw_params: {json.dumps(fw_params)}     # parameters set by name before each run (--pos-loop, --fw-param)
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework_dll: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
  build_flags: "fwsim shim (no FreeRTOS); main.c glue repeated in fwsim/fw_sim.c"
  h7: null
  rpi5: null
plant: {fwsim.plant_meta(p0)}
imu: {json.dumps(imu)}
trajectory: "x = A sin tau, y = A/2 sin 2 tau, A = {A} m; heading on the tangent; tau_dot ramp a_r = {A_R} m/s^2; setpoint held at 50 Hz (fwsim.simulate); 0 after T_cmd"
frames: "plant frame rotated by +45 deg = floor frame (start heading 45 deg, start at the origin: placement error 0)"
evaluation:
  source: "true plant pose (log.csv x, y, theta), firmware tick 100 Hz"
  D: {D}
  D_d: {D}
  t_d: {T_D}
  t_late: {T_LATE}
  delta: {DELTA}            # assumption of this script (O1: camera position uncertainty, not fixed yet)
  v_d: {V_D}               # assumption of this script (O1: fixed after the camera trial)
  D_theta_L: "not fixed in O1 v5: values reported, no pass/fail"
seed: 1                       # plant and IMU noise-free: the seed has no effect
libraries: {json.dumps(fwsim.lib_versions())}
elapsed_s: {time.time() - t0:.0f}
files:
  - summary.csv: "one line per run: O1 measures (e_perp_max, e_theta_max_deg, e_s max/min/absmax, RMS), stop error, completed, tracking-error frames, PWM saturation fraction"
  - <run_id>/log.csv, metrics.json, meta.yaml: "fwsim.write_run (metrics.json = the summary line of that run)"
  - <run_id>/o1_frames.csv: "per frame: floor pose, s_cmd, s_car, s*, e_perp, e_theta [rad], e_s, distance to origin, speed, tracking error"
""")
    print(f'done in {time.time() - t0:.0f} s -> {a.out}')


if __name__ == '__main__':
    main()
