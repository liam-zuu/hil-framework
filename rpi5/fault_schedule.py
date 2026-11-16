"""
fault_schedule.py: fault injection, batch 1 (L82; design 03_CODE/docs/2026-11-14_Thiet_ke_bom_loi_dot1_v1.md
of the thesis repo, Liam 14/11 "Duyet, bat dau code"). Schedule file, its checks, and the Pi frame patcher.

A fault of batch 1 is an operation on the int16 fields of the MOSI frame Pi -> H7 (plus the H7 command bit
HIL_FAULT_IMU_LOSS for the IMU loss). The plant always runs its full dynamics; only what the H7, and so the
DUT, gets is changed. The same integer operations are applied in fwsim by fwsim/fault_run.py.

Schedule file (CSV, UTF-8):

    # hil-fault-schedule v1
    id,fault,target,start_ms,end_ms,params
    ENC-FL-1,enc_loss,FL,5000,7000,
    IMU-LOSS-1,imu_loss,imu,10000,12000,
    IMU-DLY-1,imu_delay,imu,15000,17000,d_ms=100
    GYRO-S-1,gyro_scale,gz,20000,22000,s=0.3

  * start_ms, end_ms: whole plant steps (1 ms) counted from t0; the window is half open [start, end).
    t0 = the first step of phase RUN with --finite-run; step 0 without phases; 0 in fwsim.
  * Faults of batch 1 (frame field, what the frame k of the window carries; T[k] = the true frame):
      enc_loss   target FL|FR|RL|RR, joined with '+': int16 omega of the wheel = 0
      imu_loss   target imu: byte 20 |= HIL_FAULT_IMU_LOSS (the H7 builds no report)
      imu_delay  target imu, d_ms (int 1..1000): the 10 IMU bytes (gz, ax, ay, az, yaw) of T[k - d]
      gyro_scale target gz, s (decimal number in [-1, 1], not 0): int16 gz = round half away((1 - s) *
                 int16 gz), after the IMU noise (L82 Q2); s is kept as the exact decimal written in the
                 file (fractions.Fraction) and the product is rounded exactly (Codex FI1-1); s has at
                 most S_PLACES = 9 decimal places (its exact value is a multiple of 1e-9; 0.30, 3e-1 and
                 0.300000000 are the same s) and an exponent of at most 2 digits (Codex FI2-1)
    Overlap on the IMU: delay (picks the source frame), then the gyro scale, then the loss bit.
  * Faults of batch 2, plant faults (design 03_CODE/docs/2026-11-15_Thiet_ke_bom_loi_dot2_v1.md, Liam 15/11
    CODE_FAULT2 "Duyet, code"): they change the plant itself, not the frame (rpi5/fault_plant.py applies
    them; the frame sent stays the true frame of the faulty plant):
      wheel_jam  target wheels as enc_loss; b (N.m.s/rad, >= 0), tc (N.m, >= 0), not both 0, default 0:
                 motor_b += b, motor_Tc += tc, motor_Ts += tc of the wheel (scenario s11)
      batt_sag   target batt; v1 (V, > 0), ramp_ms (int, 0 .. end - start): V_batt goes linearly from its
                 nominal value to v1 in ramp_ms from start (0 = a step), then stays v1 to end (s13)
      mu_time    target wheels or all; mu (> 0): mu_hd of the wheel = mu (s15 in time)
      mu_zone    target all; x0 < x1, y0 < y1 (m, world frame), mu (> 0): mu_hd = mu for every wheel whose
                 contact point is inside the rectangle (s15 by position, per wheel, Liam 15/11 Q4)
    A plant fault needs the dynamic plant with ground model 'traction' (plant='traction' in check()); the
    values against the nominal parameters (mu <= nominal, v1 <= nominal) are checked by fault_plant.
  * Lines starting with '#' after the first one are comments. ibus_loss (needs the H7 part) and the s14 faults
    body_drag, load_dist (after the main S04 run, Liam 11/11 and 15/11 Q3) are refused for now.

The patcher only injects in phase RUN (design 6.2): once the run leaves RUN, open windows are cut and the
frames are the true ones again; a window that never started is 'not_run'.

Assumptions (stated by the author, to be checked): the MOSI layout of rpi5/protocol.py; wheel order of the
frame omega1..omega4 = FL, FR, RL, RR (params.py line 194); frames reach the patcher with the step rising by
one each call.
"""
from __future__ import annotations

import csv
import hashlib
import io
from fractions import Fraction
import re
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from protocol import (FRAME_SIZE, OFS_FAULT, OFS_IMU_ACC, OFS_IMU_GZ, OFS_IMU_YAW)

MAGIC = '# hil-fault-schedule v1'
COLUMNS = ('id', 'fault', 'target', 'start_ms', 'end_ms', 'params')
WHEELS = ('FL', 'FR', 'RL', 'RR')            # frame omega1..omega4 (params.py: 1=FL, 2=FR, 3=RL, 4=RR)
FAULT_IMU_LOSS = 1 << 2                      # hil_protocol.h HIL_FAULT_IMU_LOSS
D_MS_MAX = 1000
# the 10 IMU bytes of the frame: gz, then ax, ay, az, yaw (imu_seq at byte 39 is not an IMU value)
IMU_SPANS = ((OFS_IMU_GZ, OFS_IMU_GZ + 2), (OFS_IMU_ACC, OFS_IMU_YAW + 2))

# fault -> (targets, {param: type}, field group). Two rows of the same group may not overlap in time
# (for a group of wheel rows: on the same wheel; mu_zone counts as all four wheels).
FAULTS = {
    'enc_loss':   ('wheels', {}, 'enc'),
    'imu_loss':   (('imu',), {}, 'imu_loss'),
    'imu_delay':  (('imu',), {'d_ms': int}, 'imu_delay'),
    'gyro_scale': (('gz',), {'s': Fraction}, 'gyro_scale'),
    # batch 2, plant faults (design 2026-11-15 section 3)
    'wheel_jam':  ('wheels', {'b': Fraction, 'tc': Fraction}, 'jam'),
    'batt_sag':   (('batt',), {'v1': Fraction, 'ramp_ms': int}, 'batt'),
    'mu_time':    ('wheels', {'mu': Fraction}, 'mu'),
    'mu_zone':    (('all',), {'x0': Fraction, 'x1': Fraction, 'y0': Fraction, 'y1': Fraction, 'mu': Fraction},
                   'mu'),
}
DEFAULTS = {'wheel_jam': {'b': Fraction(0), 'tc': Fraction(0)}}   # parameters that may be left out
IMU_FAULTS = ('imu_loss', 'imu_delay', 'gyro_scale')
FRAME_FAULTS = ('enc_loss',) + IMU_FAULTS                 # change bytes of the frame (FramePatcher)
PLANT_FAULTS = ('wheel_jam', 'batt_sag', 'mu_time', 'mu_zone')   # change the plant (fault_plant.py)
WHEEL_FAULTS = ('enc_loss', 'wheel_jam', 'mu_time')       # target = wheels joined with '+'
ALL_WHEELS_OK = ('mu_time',)                              # also target 'all' = FL+FR+RL+RR (design 3, Codex FP1-7)
# names kept for later parts of batch 2, refused for now: ibus_loss needs the H7 part (design section 5);
# body_drag, load_dist (s14) come after the main S04 run (Liam 11/11, 15/11 Q3 "Giu sau S04")
BATCH2 = ('ibus_loss', 'body_drag', 'load_dist')
IMU_SOURCES = ('plant', 'static', 'profile', None)
PLANT_SOURCES = ('traction', None)   # 'traction': MecanumPlant on the ground with ground_model 'traction'

_WHOLE = re.compile(r'^[0-9]+$')
_DECIMAL = re.compile(r'^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]{1,2})?$')
S_PLACES = 9                                 # decimal places of a decimal parameter (Codex FI2-1)


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True)
class Row:
    id: str
    fault: str
    target: str
    start_ms: int
    end_ms: int
    params: Tuple[Tuple[str, object], ...] = ()      # sorted ((key, value), ...)

    def p(self, key):
        return dict(self.params)[key]

    @property
    def wheels(self) -> Tuple[int, ...]:
        """Wheel indices 0..3 of a wheel row (mu_zone: all four, its zone decides per step), else ()."""
        if self.fault in WHEEL_FAULTS:
            if self.target == 'all' and self.fault in ALL_WHEELS_OK:
                return (0, 1, 2, 3)
            return tuple(WHEELS.index(w) for w in self.target.split('+'))
        return (0, 1, 2, 3) if self.fault == 'mu_zone' else ()


@dataclass(frozen=True)
class Schedule:
    rows: Tuple[Row, ...]
    path: str = ''
    sha256: str = ''

    def describe(self) -> dict:
        """For the run manifest and meta.yaml."""
        return dict(file=self.path, sha256=self.sha256, version=MAGIC, rows=rows_as_json(self.rows))


def _whole(text: str, where: str, name: str) -> int:
    t = text.strip()
    if not _WHOLE.match(t):                       # refuses 5000.0, 5e3, 5_000, -1, +5
        raise ScheduleError(f'{where}: {name} must be a whole number of ms, got {text!r}')
    return int(t)


def _parse_params(text: str, schema: dict, where: str, defaults: Optional[dict] = None) -> tuple:
    out = {}
    for part in (p for p in text.split(';') if p.strip()):
        k, sep, v = part.partition('=')
        k, v = k.strip(), v.strip()
        if not sep or k not in schema:
            raise ScheduleError(f'{where}: unknown parameter {part.strip()!r} (allowed {sorted(schema)})')
        if k in out:
            raise ScheduleError(f'{where}: parameter {k} given twice')
        if schema[k] is int:
            if not _WHOLE.match(v):
                raise ScheduleError(f'{where}: parameter {k}={v!r} is not a whole number')
            out[k] = int(v)
        else:
            if not _DECIMAL.match(v):                 # refuses nan, inf, 3/10, empty
                raise ScheduleError(f'{where}: parameter {k}={v!r} is not a decimal number')
            out[k] = Fraction(v)                      # exact value of the text (FI1-1)
            if (10 ** S_PLACES) % out[k].denominator:   # FI2-1: not a multiple of 1e-9
                raise ScheduleError(f'{where}: parameter {k}={v!r} has more than {S_PLACES} decimal places')
    missing = sorted(set(schema) - set(out) - set(defaults or {}))
    if missing:
        raise ScheduleError(f'{where}: missing parameter {missing}')
    for k, v in (defaults or {}).items():
        out.setdefault(k, v)
    return tuple(sorted(out.items()))


def parse(text: str) -> List[Row]:
    """Rows of a schedule text; format errors raise ScheduleError (meaning: check())."""
    lines = text.split('\n')
    if not lines or lines[0].rstrip('\r') != MAGIC:
        raise ScheduleError(f'first line must be exactly {MAGIC!r}')
    body = '\n'.join(l.rstrip('\r') for l in lines[1:] if l.strip() and not l.lstrip().startswith('#'))
    rd = csv.reader(io.StringIO(body))
    header = next(rd, None)
    if header is None or tuple(h.strip() for h in header) != COLUMNS:
        raise ScheduleError(f'columns must be {",".join(COLUMNS)}, got {header}')
    rows = []
    for n, f in enumerate(rd, start=1):
        where = f'row {n}'
        if len(f) != len(COLUMNS):
            raise ScheduleError(f'{where}: {len(f)} fields, need {len(COLUMNS)}')
        rid, fault, target = f[0].strip(), f[1].strip(), f[2].strip()
        where = f'row {n} ({rid!r})'
        if fault in BATCH2:
            raise ScheduleError(f'{where}: {fault} belongs to a later part of batch 2, not implemented yet')
        if fault not in FAULTS:
            raise ScheduleError(f'{where}: unknown fault {fault!r}')
        a = _whole(f[3], where, 'start_ms')
        b = _whole(f[4], where, 'end_ms')
        rows.append(Row(rid, fault, target, a, b,
                        _parse_params(f[5], FAULTS[fault][1], where, DEFAULTS.get(fault))))
    if not rows:
        raise ScheduleError('the schedule has no row')
    return rows


def check(rows: Sequence[Row], run_ms: Optional[int], imu_source: Optional[str],
          plant: Optional[str] = None) -> None:
    """Meaning of the rows for one run. run_ms: steps of the run from t0 (None refused: a window must be
    known to end inside the run). imu_source: the IMU of the frames ('plant', 'static', 'profile', None).
    plant: 'traction' when the run has the dynamic plant on the ground with ground_model 'traction' (plant
    faults need it), None otherwise (fixed or profile wheel speeds, --mock, wheels lifted, 'rigid')."""
    if imu_source not in IMU_SOURCES:
        raise ScheduleError(f'imu_source {imu_source!r}')
    if plant not in PLANT_SOURCES:
        raise ScheduleError(f'plant {plant!r}')
    if run_ms is None or run_ms <= 0:
        raise ScheduleError('the run length is not known: a fault schedule needs --finite-run or --steps')
    seen = set()
    for r in rows:
        where = repr(r.id)
        if not r.id or r.id in seen:
            raise ScheduleError(f'{where}: id empty or used twice')
        seen.add(r.id)
        targets = FAULTS[r.fault][0]
        if targets == 'wheels':
            parts = r.target.split('+')
            ok = (all(p in WHEELS for p in parts) and len(set(parts)) == len(parts)) or \
                (r.target == 'all' and r.fault in ALL_WHEELS_OK)
        else:
            ok = r.target in targets
        if not ok:
            raise ScheduleError(f'{where}: target {r.target!r} not valid for {r.fault}')
        if not 0 <= r.start_ms < r.end_ms:
            raise ScheduleError(f'{where}: need 0 <= start_ms < end_ms')
        if r.end_ms > run_ms:
            raise ScheduleError(f'{where}: ends at {r.end_ms} ms, after the run ({run_ms} ms)')
        if r.fault in IMU_FAULTS and imu_source is None:
            raise ScheduleError(f'{where}: {r.fault} needs an IMU in the frames (plant, --imu-static or '
                                f'--imu-profile)')
        if r.fault == 'imu_delay':
            if imu_source == 'static':
                raise ScheduleError(f'{where}: imu_delay on --imu-static values means nothing')
            d = r.p('d_ms')
            if not 1 <= d <= D_MS_MAX:
                raise ScheduleError(f'{where}: d_ms must be 1..{D_MS_MAX}')
            if r.start_ms < d:
                raise ScheduleError(f'{where}: start_ms must be >= d_ms (no source frame before t0)')
        if r.fault == 'gyro_scale':
            s = r.p('s')
            if not (-1.0 <= s <= 1.0) or s == 0.0:
                raise ScheduleError(f'{where}: s must be in [-1, 1], not 0')
        if r.fault in PLANT_FAULTS:
            if plant != 'traction':
                raise ScheduleError(f'{where}: {r.fault} needs the dynamic plant on the ground with ground model '
                                    f"'traction' (not --fixed-omega, --omega-steps, --mock, A-K*, --ground rigid)")
            _check_plant_row(r, where)
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if FAULTS[a.fault][2] != FAULTS[b.fault][2]:
                continue
            if a.start_ms < b.end_ms and b.start_ms < a.end_ms:
                if not a.wheels or set(a.wheels) & set(b.wheels):
                    raise ScheduleError(f'{a.id!r} and {b.id!r} drive the same field at the same time')


def _check_plant_row(r: Row, where: str) -> None:
    """Ranges of a plant row that do not depend on the nominal parameters (fault_plant checks those)."""
    if r.fault == 'wheel_jam':
        if r.p('b') < 0 or r.p('tc') < 0 or (r.p('b') == 0 and r.p('tc') == 0):
            raise ScheduleError(f'{where}: wheel_jam needs b >= 0, tc >= 0, not both 0')
    elif r.fault == 'batt_sag':
        if r.p('v1') <= 0:
            raise ScheduleError(f'{where}: batt_sag needs v1 > 0')
        if r.p('ramp_ms') > r.end_ms - r.start_ms:
            raise ScheduleError(f'{where}: ramp_ms longer than the window')
    elif r.fault in ('mu_time', 'mu_zone'):
        if r.p('mu') <= 0:
            raise ScheduleError(f'{where}: mu must be > 0')
        if r.fault == 'mu_zone' and not (r.p('x0') < r.p('x1') and r.p('y0') < r.p('y1')):
            raise ScheduleError(f'{where}: mu_zone needs x0 < x1 and y0 < y1')


def file_sha256(raw: bytes) -> str:
    """SHA-256 of the file with CRLF -> LF, so a checkout with core.autocrlf gives the same value."""
    return hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()


def load(path: str, run_ms: Optional[int], imu_source: Optional[str], plant: Optional[str] = None) -> Schedule:
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ScheduleError(f'{path}: not UTF-8 ({e})') from None
    rows = parse(text)
    check(rows, run_ms, imu_source, plant)
    return Schedule(tuple(rows), path, file_sha256(raw))


def has_plant_rows(rows: Sequence[Row]) -> bool:
    return any(r.fault in PLANT_FAULTS for r in rows)


def rows_as_json(rows: Sequence[Row]) -> list:
    """JSON form of the rows (manifest, meta.yaml): exact decimal parameters as float."""
    return [dict(id=r.id, fault=r.fault, target=r.target, start_ms=r.start_ms, end_ms=r.end_ms,
                 params={k: (float(v) if isinstance(v, Fraction) else v) for k, v in r.params})
            for r in rows]


def scale_pq(s) -> Tuple[int, int]:
    """(p, q) with (1 - s) = p / q exactly, q > 0. A float s is taken as its shortest decimal (repr)."""
    f = 1 - (s if isinstance(s, Fraction) else Fraction(repr(float(s))))
    return f.numerator, f.denominator


def scale_raw_pq(raw: int, p: int, q: int) -> int:
    """round half away from zero of raw * p / q in integers (no float rounding), clamped to the int16
    range of protocol._to_int16."""
    n = abs(raw * p)
    r = (2 * n + q) // (2 * q)
    v = r if raw * p >= 0 else -r
    return max(-32767, min(32767, v))


def scale_raw(raw: int, s) -> int:
    """gyro_scale on the int16 of the frame: round half away from zero of (1 - s) * raw, exactly."""
    return scale_raw_pq(raw, *scale_pq(s))


class ActiveSet:
    """Which rows are on at relative step k. advance(k) with k rising; one integer compare per step when
    nothing changes. Used by the Pi patcher and by fwsim/fault_run.py."""

    def __init__(self, rows: Sequence[Row]):
        self.rows = list(rows)
        ev = []
        for i, r in enumerate(self.rows):
            ev.append((r.start_ms, 1, i))
            ev.append((r.end_ms, 0, i))
        ev.sort(key=lambda e: (e[0], e[1]))          # same step: off before on
        self._ev, self._i = ev, 0
        self.next: Optional[int] = ev[0][0] if ev else None
        self.on: List[int] = []
        self.started = [False] * len(self.rows)
        self.events: List[list] = []                 # [k, id, 1/0, why] as executed
        self.enc_mask = 0
        self.loss = False
        self.delay = 0
        self.scale: Optional[Fraction] = None
        self.scale_pq: Optional[Tuple[int, int]] = None
        self.any = False                             # a frame (signal) row is on
        self.plant_on: Tuple[int, ...] = ()          # indices of the plant rows on, in row order (batch 2)

    def _pop(self, k: int) -> None:
        _, on, i = self._ev[self._i]
        if on:
            self.on.append(i)
            self.started[i] = True
            self.events.append([k, self.rows[i].id, 1, 'on'])
        else:
            self.on.remove(i)
            self.events.append([k, self.rows[i].id, 0, 'off'])
        self._i += 1
        self.next = self._ev[self._i][0] if self._i < len(self._ev) else None

    def advance(self, k: int) -> None:
        while self.next is not None and k >= self.next:
            self._pop(k)
        self._derive()

    def close(self, k: int) -> None:
        """The run left RUN at relative step k (the frame of k is no longer in RUN): windows that ended at
        or before k end normally, open ones are cut at k, the others never ran. Nothing is on afterwards."""
        while self.next is not None and (self.next < k or (self.next == k and self._ev[self._i][1] == 0)):
            self._pop(self.next)
        for i in list(self.on):
            self.events.append([k, self.rows[i].id, 0, 'cut'])
        self.on = []
        self.next = None
        self._derive()

    def _derive(self) -> None:
        self.enc_mask, self.loss, self.delay, self.scale, self.scale_pq = 0, False, 0, None, None
        for i in self.on:
            r = self.rows[i]
            if r.fault == 'enc_loss':
                for w in r.wheels:
                    self.enc_mask |= 1 << w
            elif r.fault == 'imu_loss':
                self.loss = True
            elif r.fault == 'imu_delay':
                self.delay = r.p('d_ms')
            elif r.fault == 'gyro_scale':
                self.scale = r.p('s')
                self.scale_pq = scale_pq(self.scale)
        # batch 2: plant rows do not change the frame; fault_plant reads plant_on
        self.plant_on = tuple(sorted(i for i in self.on if self.rows[i].fault in PLANT_FAULTS))
        self.any = len(self.on) > len(self.plant_on)

    def not_run(self) -> List[str]:
        return [r.id for r, s in zip(self.rows, self.started) if not s]


class FramePatcher:
    """
    Pi side. patch(step, mosi_true) -> the frame to send; mosi_true is the frame of today's code, also the
    truth record of that step. start(step): t0 (first RUN step, or 0 without phases); before it nothing is
    injected. stop(step): the run left RUN at this step (design 6.2). Called with step rising by one.
    """

    def __init__(self, schedule: Schedule):
        self.schedule = schedule
        self.act = ActiveSet(schedule.rows)
        dmax = max([r.p('d_ms') for r in schedule.rows if r.fault == 'imu_delay'], default=0)
        self._n = dmax + 1 if dmax else 0
        self._ring = [bytes(FRAME_SIZE)] * self._n if self._n else None
        self.t0: Optional[int] = None
        self.stop_step: Optional[int] = None
        self._live = False
        self.step: Optional[int] = None    # step of the last patch() call (fault_plant records it)
        self.plant_hook = None             # batch 2: fault_plant.FaultPlant, its record goes in summary()

    def start(self, step: int) -> None:
        """Once only, and never after stop()."""
        if self.t0 is None and self.stop_step is None:
            self.t0 = step
            self._live = True

    def stop(self, step: int) -> None:
        """No more injection from this step on; open windows are cut. Without a start, nothing ran."""
        if self.stop_step is not None:
            return
        self.stop_step = step
        self._live = False
        if self.t0 is not None:
            self.act.close(step - self.t0)

    def patch(self, step: int, mosi_true: bytes) -> bytes:
        self.step = step
        ring = self._ring
        if ring is not None:
            ring[step % self._n] = mosi_true
        if not self._live:
            return mosi_true
        a = self.act
        k = step - self.t0
        if a.next is not None and k >= a.next:
            a.advance(k)
        if not a.any:
            return mosi_true
        b = bytearray(mosi_true)
        m = a.enc_mask
        if m:
            for w in range(4):
                if m >> w & 1:
                    b[2 * w] = 0
                    b[2 * w + 1] = 0
        if a.delay:
            src = ring[(step - a.delay) % self._n]
            for lo, hi in IMU_SPANS:
                b[lo:hi] = src[lo:hi]
        if a.scale is not None:
            raw = struct.unpack_from('>h', b, OFS_IMU_GZ)[0]
            struct.pack_into('>h', b, OFS_IMU_GZ, scale_raw_pq(raw, *a.scale_pq))
        if a.loss:
            b[OFS_FAULT] |= FAULT_IMU_LOSS
        return bytes(b)

    def summary(self, last_step: Optional[int] = None) -> dict:
        """For the manifest. last_step: the last step of the loop; a run that ends while still live (no
        stop, e.g. a signal or --steps) has its open windows cut at last_step + 1."""
        if self._live and last_step is not None:
            self.act.close(last_step + 1 - self.t0)     # a window ending at last_step + 1 ends normally
            self._live = False
        cut = any(e[3] == 'cut' for e in self.act.events)
        not_run = self.act.not_run()
        d = self.schedule.describe()
        d.update(t0_step=self.t0, stop_step=self.stop_step, events=[list(e) for e in self.act.events],
                 not_run=not_run, cut=cut, complete=not cut and not not_run)
        if self.plant_hook is not None:
            d['plant_record'] = self.plant_hook.finish()     # batch 2: file, sha256, rows of the plant record
        return d
