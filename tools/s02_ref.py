"""
s02_ref.py: spec S02 P2-2b / P2-3 reference quantities computed off the machine from the standard schedule:
the reference r(t), the command pose p_ref(t), psi_ref(t), the test frame, the step segments of A-K1, A-K2,
and the Mecanum kinematics used by tools/s02_metrics.py.

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van) sections 18.2, 19, 20 (v3, Liam 15/11
CODE_S02 local_0eb4085a: "Duyet, code sau gop (Recommended)"); spec S02 v1.13 P2-2b (standard schedule), P2-3.
rpi5/trajectory_pub.py is only imported (S04 blob lock).

Schedule: u_k = get_body_velocity(k / 50, GROUP_A[test]) for k = 0 .. K - 1, K = ceil(T_cmd 50), T_cmd =
GROUP_A[test].duration (stop tail not included). Reference r(t) = u_floor(t / 20 ms) for 0 <= t < K 20 ms, 0
outside (spec P2-2b: computed off the machine, not the logged setpoint).

Command pose: r held constant over each 20 ms cell is integrated exactly from p_ref(0) = 0, psi_ref(0) = the
start heading of the test (an arc when wz != 0, a segment when wz = 0; the partial cell exactly up to t). The
start heading is the direction of the path of get_reference at t = 0+ (line, strafe, A-K: 0; circle: 0;
figure 8: pi/4). get_reference's own theta is not used: for the circle it is pi/2 (atan2(cos, sin)), not the
path tangent 0 (a quirk of trajectory_pub, read only). e_p, e_psi, e_est do not depend on this choice: every
pose is turned into the test frame by the same psi_ref(0) (P2-3 "He cua bai").

Kinematics (wheel order FL, FR, RL, RR as the *_rpm columns and the plant omega1..4): ik() as tools/s02_metrics.ik
(kinematics.h, LX + LY = 0.24 m, r = 0.0485 m), fk() its inverse. The signs are checked against the plant in
fwsim/tests/test_s02_ref_sim.py (rigid ground: fk(omega) = plant body velocity).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from trajectory_pub import GROUP_A, get_body_velocity, get_reference  # noqa: E402

SEND_HZ = 50
CELL_S = 1.0 / SEND_HZ
CELL_US = 1e6 / SEND_HZ
R_WHEEL = 0.0485
L_SUM = 0.24
STEP_TESTS = ('A-K1', 'A-K2')
STEP_LO, STEP_HI = 0.1, 0.9            # P2-3: rise from y >= 0.1 to y >= 0.9
XL_LAST_S = 1.0                         # e_xl over the last 1 s of a held segment
RISE_UNINTERP = 0.1                     # e_xl of the segment after a step >= 10 % |A|: rise not interpretable


def n_samples(test_id):
    """K: packets of the standard schedule (as s02_sender and s02_cmd_path.schedule)."""
    return int(math.ceil(GROUP_A[test_id].duration * SEND_HZ - 1e-9))


def t_cmd(test_id):
    """T_cmd [s]: the schedule duration (stop tail not included); the metric window is [1 s, T_cmd]."""
    return float(GROUP_A[test_id].duration)


def samples(test_id):
    """u (K, 3): the standard schedule."""
    spec = GROUP_A[test_id]
    return np.array([get_body_velocity(k / SEND_HZ, spec) for k in range(n_samples(test_id))], dtype=float)


def cell_index(t_s):
    """k = floor(t / 20 ms) of times in seconds from t = 0 (a 1 ns guard against float rounding at cell edges)."""
    return np.floor(np.asarray(t_s, dtype=float) / CELL_S + 1e-9).astype(np.int64)


def reference(test_id, t_s, u=None):
    """r(t) (n, 3) at times t (s from t = 0): u_floor(t / 20 ms) inside the schedule, 0 outside."""
    u = samples(test_id) if u is None else u
    k = cell_index(t_s)
    out = np.zeros((len(k), 3))
    inside = (k >= 0) & (k < len(u))
    out[inside] = u[k[inside]]
    return out


def start_heading(test_id):
    """psi_ref(0): direction of the get_reference path at t = 0+ (module docstring); 0 for tests whose
    get_reference does not move (strafe, A-K: idle in trajectory_pub)."""
    spec = GROUP_A[test_id]
    x0, y0 = get_reference(0.0, spec)[:2]
    x1, y1 = get_reference(1e-6, spec)[:2]
    if math.hypot(x1 - x0, y1 - y0) < 1e-12:
        return 0.0
    return math.atan2(y1 - y0, x1 - x0)


def _advance(x, y, psi, u, dt):
    """Pose after holding body velocity u = (vx, vy, wz) for dt from (x, y, psi), exactly."""
    vx, vy, wz = u
    if abs(wz * dt) < 1e-12:
        S, C = dt * math.cos(psi), dt * math.sin(psi)
        psi1 = psi + wz * dt
    else:
        psi1 = psi + wz * dt
        S = (math.sin(psi1) - math.sin(psi)) / wz           # integral of cos
        C = (math.cos(psi) - math.cos(psi1)) / wz           # integral of sin
    return x + vx * S - vy * C, y + vx * C + vy * S, psi1


def cell_poses(test_id, u=None, psi0=None):
    """Poses (K + 1, 3) at the start of every cell k = 0 .. K (the last one: end of the schedule)."""
    u = samples(test_id) if u is None else u
    psi = start_heading(test_id) if psi0 is None else psi0
    out = np.zeros((len(u) + 1, 3))
    x = y = 0.0
    out[0] = (x, y, psi)
    for k in range(len(u)):
        x, y, psi = _advance(x, y, psi, u[k], CELL_S)
        out[k + 1] = (x, y, psi)
    return out


def pose_ref(test_id, t_s, u=None, psi0=None):
    """p_ref, psi_ref (n, 3) at times t (s from t = 0); psi unwrapped (continuous). Before t = 0: the start pose;
    after the schedule: the end pose (r = 0)."""
    u = samples(test_id) if u is None else u
    base = cell_poses(test_id, u, psi0)
    t = np.asarray(t_s, dtype=float)
    k = np.clip(cell_index(t), 0, len(u))
    out = np.zeros((len(t), 3))
    for i, (ti, ki) in enumerate(zip(t, k)):
        if ti <= 0.0:
            out[i] = base[0]
            continue
        x, y, psi = base[ki]
        if ki < len(u):
            x, y, psi = _advance(x, y, psi, u[ki], ti - ki * CELL_S)
        out[i] = (x, y, psi)
    return out


def to_test_frame(x, y, th, x0, y0, th0, psi0):
    """Pose (arrays) in the test frame: dp = R(-th0) (p - p0), dpsi = th - th0 (th unwrapped by the caller),
    then p = R(psi0) dp, psi = psi0 + dpsi (P2-3 "He cua bai")."""
    dx, dy = np.asarray(x) - x0, np.asarray(y) - y0
    c0, s0 = math.cos(th0), math.sin(th0)
    bx, by = c0 * dx + s0 * dy, -s0 * dx + c0 * dy
    c, s = math.cos(psi0), math.sin(psi0)
    return c * bx - s * by, s * bx + c * by, psi0 + (np.asarray(th) - th0)


def wrap(a):
    """Angle wrapped to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    w = np.mod(a + math.pi, 2 * math.pi) - math.pi
    return np.where(w == -math.pi, math.pi, w)


def ik(vx, vy, wz):
    """Wheel speeds (n, 4) FL, FR, RL, RR of body velocities (tools/s02_metrics.ik)."""
    k = 1.0 / R_WHEEL
    vx, vy, wz = (np.asarray(a, dtype=float) for a in (vx, vy, wz))
    return np.stack([k * (vx - vy - L_SUM * wz), k * (vx + vy + L_SUM * wz),
                     k * (vx + vy - L_SUM * wz), k * (vx - vy + L_SUM * wz)], axis=1)


def fk(w):
    """Body velocity (n, 3) of wheel speeds w (n, 4) FL, FR, RL, RR: the inverse of ik()."""
    w = np.asarray(w, dtype=float)
    q = R_WHEEL / 4.0
    vx = q * (w[:, 0] + w[:, 1] + w[:, 2] + w[:, 3])
    vy = q * (-w[:, 0] + w[:, 1] + w[:, 2] - w[:, 3])
    wz = q / L_SUM * (-w[:, 0] + w[:, 1] - w[:, 2] + w[:, 3])
    return np.stack([vx, vy, wz], axis=1)


def segments(test_id, u=None):
    """Held segments of r_vx over [0, T_cmd): [(t_start, t_end, level)] with t in s from t = 0 (A-K1: [0, 3),
    [3, 6), [6, 9) at 0, 0.3, 0; A-K2: 0.3, -0.3, 0.3). The stop tail belongs to no segment (P2-3)."""
    u = samples(test_id) if u is None else u
    vx = u[:, 0]
    ch = [0] + [k for k in range(1, len(vx)) if vx[k] != vx[k - 1]] + [len(vx)]
    return [(a * CELL_S, b * CELL_S, float(vx[a])) for a, b in zip(ch[:-1], ch[1:])]


def step_quantities(t_s, v, test_id, u=None):
    """P2-3 step quantities on one speed series v at times t (s from t = 0; rows of the metric lines, any order
    of gaps). Per segment N (1-based): e_xl_segN for every held segment; for a segment after a step also
    step_segN ('a->b'), rise_s_segN, overshoot_pct_segN, frac90_segN, no90_segN, rise_uninterp_segN. A segment
    without lines gives NaN (e_xl: also when no line falls in its last 1 s). Returns a dict."""
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(v, dtype=float)
    segs = segments(test_id, u)
    out = {}
    for n, (a, b, lev) in enumerate(segs, start=1):
        m = (t >= a - 1e-9) & (t < b - 1e-9)
        last = m & (t >= b - XL_LAST_S - 1e-9)
        out[f'e_xl_seg{n}'] = float(abs(np.mean(v[last]) - lev)) if last.any() else float('nan')
        if n == 1:
            continue
        prev = segs[n - 2][2]
        A = lev - prev
        out[f'step_seg{n}'] = f'{prev:+.2f}->{lev:+.2f}'
        ts, vs = t[m], v[m]
        if ts.size == 0 or A == 0:
            for q in ('rise_s', 'overshoot_pct', 'frac90'):
                out[f'{q}_seg{n}'] = float('nan')
            out[f'no90_seg{n}'] = float('nan')
            out[f'rise_uninterp_seg{n}'] = float('nan')
            continue
        o = np.argsort(ts, kind='stable')
        ts, vs = ts[o], vs[o]
        y = (vs - prev) / A
        i10, i90 = np.flatnonzero(y >= STEP_LO), np.flatnonzero(y >= STEP_HI)
        out[f'no90_seg{n}'] = int(i90.size == 0)
        out[f'rise_s_seg{n}'] = float(ts[i90[0]] - ts[i10[0]]) if i90.size and i10.size else float('nan')
        out[f'overshoot_pct_seg{n}'] = float(max(0.0, float(np.max(y)) - 1.0) * 100.0)
        # time with y >= 0.9: each line holds its value until the next line (or the segment end); the start of the
        # segment up to its first line is held by the last line before the segment (S2M1-10, Codex round 1 of v3:
        # a lost first line no longer drops the start of the segment from the denominator); no line before the
        # segment: frac90 not defined
        # S2M2-6 (Codex round 2 of v3): a line right at the segment start needs no line before it
        pre = np.flatnonzero(t < a - 1e-9)
        tf = yf = None
        if ts[0] <= a + 1e-9:
            tf, yf = ts, y
        elif pre.size:
            jp = pre[np.argmax(t[pre])]
            tf, yf = np.r_[a, ts], np.r_[(v[jp] - prev) / A, y]
        if tf is None:
            out[f'frac90_seg{n}'] = float('nan')
        else:
            dur = np.diff(np.append(tf, b))
            out[f'frac90_seg{n}'] = (float(np.sum(dur[yf >= STEP_HI]) / np.sum(dur)) if np.sum(dur) > 0
                                     else float('nan'))
        xl = out[f'e_xl_seg{n}']
        out[f'rise_uninterp_seg{n}'] = (int(xl >= RISE_UNINTERP * abs(A)) if math.isfinite(xl) else float('nan'))
    return out
