"""
fault_plant.py: fault injection batch 2, plant faults (design 03_CODE/docs/2026-11-15_Thiet_ke_bom_loi_dot2_v1.md
of the thesis repo, sections 2-4; Liam 15/11, CODE_FAULT2, "Duyet, code"; Q4 "Theo tung banh").

A plant fault changes the plant itself for the steps of its window: before plant_step of step k, a copy of the
parameters with a few fields changed is used instead of the nominal ones. plant_step.py and params.py are not
changed (S04 blob lock). The frame sent to the H7 stays the true frame of this (faulty) plant, so the evidence
of a plant fault is the record of the parameters used at every step (finish()), not a byte comparison.

  wheel_jam (s11)   motor_b[i] += b, motor_Tc[i] += tc, motor_Ts[i] += tc for the target wheels
  batt_sag  (s13)   V_batt = V0 + (v1 - V0) * min(1, (k - start) / ramp_ms)   (ramp_ms = 0: v1 at once)
  mu_time   (s15)   mu_hd[i] = mu for the target wheels
  mu_zone   (s15)   mu_hd[i] = mu for every wheel whose contact point (centre + R(theta) (+-lx, +-ly)) is inside
                    [x0, x1) x [y0, y1) at the start of the step (world frame, per wheel)
  body_drag (s14)   a world-frame force on the body at every traction sub-step: -F * ((vx, vy) / |v|), |v| =
                    math.hypot(vx, vy) of the sub-step, 0 while |v| < DRAG_V_MIN (Liam 17/11 "lam s14 ngay di")
  load_dist (s14)   a world-frame force F(k) * (cos dir, sin dir), constant over the step: F(k) = (F0 + min(Fmax,
                    (rate * d) / 1000)) + sigma * n[d], d = k - start, n = default_rng(seed).standard_normal(end -
                    start) drawn once for the row (not the plant's generator, and the same whatever steps run)
The two body loads change no parameter: the step is plant_step's traction branch with the force as f_ext
(body_step: plant_step._traction_step(x, u, p, dt, f_ext=f), the call plant_step itself makes with f_ext None).

One definition, two users: PlantVariant and BodyForce are used by FaultPlant (Pi, wraps MecanumPlant.step) and
by fwsim/fault_run.py (wraps fwsim.plant_step); tools/fault_check.py rebuilds the values on its own.

The rows on at relative step k come from the ActiveSet of the FramePatcher of the run (fault_schedule.py), which
the patcher has advanced to step k before the plant steps (hil_node._tick: patch, SPI, then plant.step), so the
plant faults follow the same phase rule as the frame faults: only in RUN, cut when the run leaves RUN.

Assumptions (stated by the author, to be checked): plant_step reads mu_hd, motor_b, motor_Tc, motor_Ts, V_batt
from the parameter object at every call and keeps nothing derived from them (plant_step.py 2acab94 lines
184-215); wheel order FL, FR, RL, RR with x forward, y left (params.py H_inv rows); the contact point of a wheel
is its hub position (+-lx, +-ly) in the body frame.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np

from fault_schedule import BODY_FAULTS, PLANT_FAULTS, Row, ScheduleError, has_plant_rows
from plant_interface import PlantInterface
from plant_step import _traction_step

# body-frame position signs of the wheel hubs FL, FR, RL, RR (x forward, y left; params.py H_inv rows)
HUB_SIGNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
# columns of the record: one row per plant step with a plant row on
# (s14: vx, vy the world velocity at the start of the step; f_drag the drag F of a body_drag row on when |v| >=
# DRAG_V_MIN at the start of the step, else 0; f_dist F(k) of a load_dist row on, else 0)
RECORD_COLS = ('step', 'mask', 'V_batt', 'mu_fl', 'mu_fr', 'mu_rl', 'mu_rr', 'b_fl', 'b_fr', 'b_rl', 'b_rr',
               'tc_fl', 'tc_fr', 'tc_rl', 'tc_rr', 'ts_fl', 'ts_fr', 'ts_rl', 'ts_rr', 'x', 'y', 'theta',
               'vx', 'vy', 'f_drag', 'f_dist')
MAX_ROWS = 52                      # mask of the record is a float64: row index bits must stay exact
DRAG_V_MIN = 1e-3                  # [m/s] body_drag acts only at |v| >= this (design 3)


def nominal_problem(params) -> Optional[str]:
    """None if the plant faults have a meaning with these parameters, else why not."""
    if getattr(params, 'motor_model', 'torque') != 'dc':
        return "plant faults need motor_model 'dc'"
    if getattr(params, 'ground_model', 'rigid') != 'traction':
        return "plant faults need ground_model 'traction'"
    if getattr(params, 'wheels_lifted', False):
        return 'plant faults need the wheels on the ground'
    return None


def _finite(v, what: str) -> float:
    """float(v) or ScheduleError when v has no finite float64 value (Codex FP1-6: a 400-digit number)."""
    try:
        f = float(v)
    except (OverflowError, ValueError, TypeError):
        raise ScheduleError(f'{what}={v!r} has no float value') from None
    if not math.isfinite(f):
        raise ScheduleError(f'{what}={v!r} is not finite')
    return f


def check_params(rows: Sequence[Row], params) -> None:
    """The plant rows against the nominal parameters of the run: every parameter of a plant row and the nominal
    mu_hd, V_batt have a finite float value, the nominal ones > 0 (Codex FP1-6); mu <= nominal mu_hd of every
    wheel it may touch (a fault only makes the floor more slippery, and the traction sub-steps, sized from
    max(mu_hd), do not grow), v1 <= nominal V_batt. ScheduleError otherwise."""
    if not has_plant_rows(rows):
        return
    prob = nominal_problem(params)
    if prob:
        raise ScheduleError(prob)
    if len(rows) > MAX_ROWS:
        raise ScheduleError(f'a schedule with plant rows has at most {MAX_ROWS} rows')
    mu0 = [_finite(v, 'nominal mu_hd') for v in np.broadcast_to(np.asarray(params.mu_hd, dtype=float), (4,))]
    if _finite(params.V_batt, 'nominal V_batt') <= 0 or min(mu0) <= 0:
        raise ScheduleError(f'nominal V_batt {params.V_batt} and mu_hd {mu0} must be > 0')
    for r in rows:
        if r.fault in PLANT_FAULTS:
            for k, v in r.params:
                if not isinstance(v, int):
                    _finite(v, f'{r.id!r} {k}')
    for r in rows:
        if r.fault in ('mu_time', 'mu_zone'):
            for w in r.wheels:
                if float(r.p('mu')) > mu0[w]:
                    raise ScheduleError(f'{r.id!r}: mu {float(r.p("mu"))} > nominal mu_hd {mu0[w]} of wheel {w}')
        elif r.fault == 'batt_sag' and float(r.p('v1')) > params.V_batt:
            raise ScheduleError(f'{r.id!r}: v1 {float(r.p("v1"))} > nominal V_batt {params.V_batt}')


def contact_points(x, params):
    """World (x, y) of the four wheel hubs FL, FR, RL, RR for plant state x = [x, y, theta, ...]. Definition
    (design 3, Codex FP1-1), in float64 and in this order: c = math.cos(theta), s = math.sin(theta),
    ax = sx * lx, ay = sy * ly; hub x = (x + c * ax) - s * ay, hub y = (y + s * ax) + c * ay."""
    px, py, th = float(x[0]), float(x[1]), float(x[2])
    c, s = math.cos(th), math.sin(th)
    out = []
    for sx, sy in HUB_SIGNS:
        ax, ay = sx * params.lx, sy * params.ly
        out.append(((px + c * ax) - s * ay, (py + s * ax) + c * ay))
    return out


def zone_mask(row: Row, x, params) -> int:
    """Bit w set when the hub of wheel w is inside [x0, x1) x [y0, y1), the bounds taken as float64 (the
    decimal of the file rounded once to the nearest float64, Codex FP1-1)."""
    x0, x1, y0, y1 = (float(row.p(k)) for k in ('x0', 'x1', 'y0', 'y1'))
    m = 0
    for w, (px, py) in enumerate(contact_points(x, params)):
        if x0 <= px < x1 and y0 <= py < y1:
            m |= 1 << w
    return m


def batt_value(row: Row, k_rel: int, v0: float) -> float:
    """V_batt of a batt_sag row at relative step k_rel inside its window. Definition (Codex FP2-1), float64,
    one operation at a time: v1 = float(v1 of the file); d = k_rel - start (int); with ramp_ms = 0 or d >=
    ramp_ms: v1; else v0 + (v1 - v0) * (d / ramp_ms), d / ramp_ms the float division of two ints."""
    v1, ramp = float(row.p('v1')), row.p('ramp_ms')
    if ramp == 0 or k_rel - row.start_ms >= ramp:
        return v1
    return v0 + (v1 - v0) * ((k_rel - row.start_ms) / ramp)


RUNTIME_FIELDS = ('motor_enable',)   # written on the nominal object while a run goes (fwsim.simulate: EN of every
                                     # firmware tick); copied into the variant at every step (Codex FP1-5)


class PlantVariant:
    """at(on, k_rel, x) -> (parameters for this step, zone mask). on = ActiveSet.plant_on (row indices of the
    plant rows on, in row order). No row on: the nominal object itself. A copy is built once per (rows on, zone
    mask) and kept; at every step the RUNTIME_FIELDS of the nominal object are copied into it and batt_sag
    writes its V_batt (the copy is private to this object)."""

    def __init__(self, rows: Sequence[Row], base):
        self.rows, self.base = list(rows), base
        self._cache = {}

    def at(self, on: Tuple[int, ...], k_rel: int, x):
        on = tuple(i for i in on if self.rows[i].fault not in BODY_FAULTS)     # s14 body rows change no parameter
        if not on:
            return self.base, 0
        zm = 0
        for i in on:
            if self.rows[i].fault == 'mu_zone':
                zm |= zone_mask(self.rows[i], x, self.base)
        p = self._cache.get((on, zm))
        if p is None:
            p = self._cache[(on, zm)] = self._build(on, zm)
        for f in RUNTIME_FIELDS:
            setattr(p, f, getattr(self.base, f))
        for i in on:
            if self.rows[i].fault == 'batt_sag':
                p.V_batt = batt_value(self.rows[i], k_rel, self.base.V_batt)
        return p, zm

    def _build(self, on, zm):
        base = self.base
        p = copy.copy(base)
        b = np.array(base.motor_b, dtype=float)
        tc = np.array(base.motor_Tc, dtype=float)
        ts = np.array(base.motor_Ts, dtype=float)
        mu = np.array(np.broadcast_to(np.asarray(base.mu_hd, dtype=float), (4,)))
        for i in on:                                   # row order
            r = self.rows[i]
            if r.fault == 'wheel_jam':
                for w in r.wheels:
                    b[w] += float(r.p('b'))
                    tc[w] += float(r.p('tc'))
                    ts[w] += float(r.p('tc'))
            elif r.fault == 'mu_time':
                for w in r.wheels:
                    mu[w] = float(r.p('mu'))
            elif r.fault == 'mu_zone':
                for w in range(4):
                    if zm >> w & 1:
                        mu[w] = float(r.p('mu'))
        p.motor_b, p.motor_Tc, p.motor_Ts, p.mu_hd = b, tc, ts, mu
        return p


def dist_noise(row: Row) -> np.ndarray:
    """n[d], d = 0 .. end - start - 1, of a load_dist row: its own generator, drawn once (design 3)."""
    return np.random.default_rng(int(row.p('seed'))).standard_normal(row.end_ms - row.start_ms)


def dist_value(row: Row, k_rel: int, n: np.ndarray) -> float:
    """F(k) of a load_dist row at relative step k_rel inside its window. Definition, float64, one operation at a
    time: d = k_rel - start (int); ramp = (float(rate) * d) / 1000.0; F = (float(F0) + min(float(Fmax), ramp)) +
    float(sigma) * float(n[d])."""
    d = k_rel - row.start_ms
    ramp = (float(row.p('rate')) * d) / 1000.0
    return (float(row.p('F0')) + min(float(row.p('Fmax')), ramp)) + float(row.p('sigma')) * float(n[d])


def dist_dir(row: Row) -> Tuple[float, float]:
    """(cos, sin) of dir_deg of a load_dist row: math.radians(float(dir_deg)), then math.cos, math.sin."""
    a = math.radians(float(row.p('dir_deg')))
    return math.cos(a), math.sin(a)


class BodyForce:
    """at(on, k_rel, x) -> (f, f_drag, f_dist): f the f_ext of _traction_step for this step (None when no body
    row is on), f_drag and f_dist the record values (RECORD_COLS). At most one body_drag and one load_dist row
    are on at a time (fault_schedule.check: same group, no overlap); the noise of every load_dist row is drawn
    here, once."""

    def __init__(self, rows: Sequence[Row]):
        self.rows = list(rows)
        self._noise = {i: dist_noise(r) for i, r in enumerate(self.rows) if r.fault == 'load_dist'}
        self._dir = {i: dist_dir(r) for i, r in enumerate(self.rows) if r.fault == 'load_dist'}
        # Codex FS1-5: every F(k) of a load_dist row must be a finite float, and so must the drag F plus the
        # largest |F(k)| over the steps where the two windows overlap (Codex FS2-4: only there do they add up);
        # each parameter alone is checked by check_params, a sum or a product may still overflow
        vals = {}
        for i, n in self._noise.items():
            r = self.rows[i]
            d = np.arange(len(n), dtype=float)
            with np.errstate(over='ignore', invalid='ignore'):     # an overflow is what is looked for here
                v = (float(r.p('F0')) + np.minimum(float(r.p('Fmax')), (float(r.p('rate')) * d) / 1000.0)) + \
                    float(r.p('sigma')) * n
            if not np.all(np.isfinite(v)):
                raise ScheduleError(f'{r.id!r}: load_dist force F(k) has no finite value')
            vals[i] = v
        for rd in (r for r in self.rows if r.fault == 'body_drag'):
            for i, v in vals.items():
                rl = self.rows[i]
                lo, hi = max(rd.start_ms, rl.start_ms), min(rd.end_ms, rl.end_ms)
                if lo < hi and not math.isfinite(float(rd.p('F')) + float(np.max(np.abs(
                        v[lo - rl.start_ms:hi - rl.start_ms])))):
                    raise ScheduleError(f'{rd.id!r} + {rl.id!r}: drag F plus the load |F(k)| where they overlap has '
                                        f'no finite value')

    def at(self, on: Tuple[int, ...], k_rel: int, x):
        fd, fdist, cd, sd = 0.0, 0.0, 0.0, 0.0
        drag = dist = False
        for i in on:
            r = self.rows[i]
            if r.fault == 'body_drag':
                drag, fd = True, float(r.p('F'))
            elif r.fault == 'load_dist':
                dist, fdist = True, dist_value(r, k_rel, self._noise[i])
                cd, sd = self._dir[i]
        if not (drag or dist):
            return None, 0.0, 0.0
        gx0, gy0 = fdist * cd, fdist * sd              # the load disturbance, constant over the step

        def f(xw, yw, vxw, vyw):
            gx, gy = gx0, gy0
            if drag:
                v = math.hypot(vxw, vyw)
                if v >= DRAG_V_MIN:
                    gx = gx - fd * (vxw / v)       # Codex FS1-5: unit vector first, no overflow of F * vx
                    gy = gy - fd * (vyw / v)
            return gx, gy

        rec_drag = fd if drag and math.hypot(float(x[3]), float(x[4])) >= DRAG_V_MIN else 0.0
        return f, rec_drag, fdist


def mu4(p):
    """The four mu_hd values of a parameter object (a scalar mu_hd broadcast, as check_params and plant_step read
    it; Codex FS1-3: a body row alone keeps the nominal object, whose mu_hd may be a scalar)."""
    return np.broadcast_to(np.asarray(p.mu_hd, dtype=float), (4,))


def body_step(x, u, p, dt, f):
    """One plant step with the body force f: plant_step's traction branch (plant_step.py: _traction_step(
    np.asarray(x, float), np.asarray(tau, float), params, dt)) with f_ext = f. plant_step.py is not changed (S04
    blob lock)."""
    return _traction_step(np.asarray(x, dtype=float), np.asarray(u, dtype=float), p, dt, f_ext=f)


def plant_window_steps(rows: Sequence[Row]) -> int:
    """Relative steps covered by the union of the plant windows (record size)."""
    cover = set()
    for r in rows:
        if r.fault in PLANT_FAULTS:
            cover.update(range(r.start_ms, r.end_ms))
    return len(cover)


def record_path(frame_log: str) -> str:
    base = frame_log[:-4] if frame_log.endswith('.npz') else frame_log
    return base + '_plant_faults.npz'


class FaultPlant(PlantInterface):
    """
    Pi side: wraps a MecanumPlant. step() with no plant row on is inner.step() itself; with plant rows on, the
    inner plant steps with the parameters of PlantVariant (swapped in for that one call), and the step is
    recorded. The patcher gives the rows on (patcher.act.plant_on), the step (patcher.step) and t0.
    """

    def __init__(self, inner, patcher, path: Optional[str]):
        self.inner, self._p, self._act = inner, patcher, patcher.act
        self._base = inner._params
        check_params(patcher.schedule.rows, self._base)
        self._var = PlantVariant(patcher.schedule.rows, self._base)
        self._body = BodyForce(patcher.schedule.rows)
        self._path = path
        self._rec = np.zeros((plant_window_steps(patcher.schedule.rows), len(RECORD_COLS)))
        self._n = 0
        self._finished: Optional[dict] = None

    # PlantInterface
    def step(self, torques: list) -> list:
        on = self._act.plant_on
        if not on:
            return self.inner.step(torques)
        inner, k = self.inner, self._p.step
        x = inner._state
        p, _ = self._var.at(on, k - self._p.t0, x)
        f, fd, fdist = self._body.at(on, k - self._p.t0, x)
        if self._n < len(self._rec):              # the union of windows bounds the steps with a row on
            self._rec[self._n] = (k, sum(1 << i for i in on), p.V_batt, *mu4(p), *p.motor_b, *p.motor_Tc,
                                  *p.motor_Ts, x[0], x[1], x[2], x[3], x[4], fd, fdist)
        self._n += 1
        if f is not None:                         # s14: MecanumPlant.step with the body force (no RNG on traction)
            inner._state = body_step(x, torques, p, inner._dt, f)
            inner._ticks += 1
            return inner._state.tolist()
        inner._params = p
        try:
            return inner.step(torques)
        finally:
            inner._params = self._base

    def get_state(self) -> list:
        return self.inner.get_state()

    def reset(self, state: list = None, seed: int = None) -> None:
        self.inner.reset(state, seed)

    def __getattr__(self, name):                    # step_count, ... of the inner plant
        return getattr(self.inner, name)

    def finish(self) -> dict:
        """Save the record once (np.savez next to the frame log) and describe it for the manifest."""
        if self._finished is not None:
            return self._finished
        n = min(self._n, len(self._rec))
        b = self._base
        nominal = dict(V_batt=float(b.V_batt), lx=float(b.lx), ly=float(b.ly),
                       **{k: [float(v) for v in np.broadcast_to(np.asarray(getattr(b, k), dtype=float), (4,))]
                          for k in ('mu_hd', 'motor_b', 'motor_Tc', 'motor_Ts')})
        d = dict(file=None, sha256=None, rows=self._n, overflow=self._n > len(self._rec),
                 cols=list(RECORD_COLS), nominal=nominal, inner=type(self.inner).__name__)   # FP1-11
        if self._path:
            np.savez(self._path, record=self._rec[:n], cols=np.array(json.dumps(list(RECORD_COLS))))
            with open(self._path, 'rb') as fh:
                d.update(file=os.path.basename(self._path), sha256=hashlib.sha256(fh.read()).hexdigest())
        self._finished = d
        return d


def attach(plant, patcher, frame_log: Optional[str]):
    """hil_node: the plant itself when there is no schedule or no plant row; else a FaultPlant around it
    (ScheduleError when the plant cannot take plant faults), hooked to the patcher so that its record goes in
    the fault summary of the manifest."""
    if patcher is None or not has_plant_rows(patcher.schedule.rows):
        return plant
    from mecanum_plant import MecanumPlant
    if type(plant) is not MecanumPlant:            # Codex FP1-11: the class itself, not a subclass
        raise ScheduleError(f'plant faults need MecanumPlant, the run has {type(plant).__name__}')
    fp = FaultPlant(plant, patcher, record_path(frame_log) if frame_log else None)
    patcher.plant_hook = fp
    return fp
