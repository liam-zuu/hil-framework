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
      gyro_scale target gz, s (float in [-1, 1], not 0): int16 gz = round half away((1 - s) * int16 gz),
                 after the IMU noise (L82 Q2)
    Overlap on the IMU: delay (picks the source frame), then the gyro scale, then the loss bit.
  * Lines starting with '#' after the first one are comments. Faults of batch 2 are refused.

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
import math
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
# (for enc_loss: on the same wheel).
FAULTS = {
    'enc_loss':   ('wheels', {}, 'enc'),
    'imu_loss':   (('imu',), {}, 'imu_loss'),
    'imu_delay':  (('imu',), {'d_ms': int}, 'imu_delay'),
    'gyro_scale': (('gz',), {'s': float}, 'gyro_scale'),
}
IMU_FAULTS = ('imu_loss', 'imu_delay', 'gyro_scale')
BATCH2 = ('wheel_jam', 'batt_sag', 'mu_time', 'mu_zone', 'ibus_loss', 'body_drag')   # design section 4
IMU_SOURCES = ('plant', 'static', 'profile', None)

_WHOLE = re.compile(r'^[0-9]+$')


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
        return tuple(WHEELS.index(w) for w in self.target.split('+')) if self.fault == 'enc_loss' else ()


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


def _parse_params(text: str, schema: dict, where: str) -> tuple:
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
            try:
                f = float(v)
            except ValueError:
                raise ScheduleError(f'{where}: parameter {k}={v!r} is not a number') from None
            if not math.isfinite(f):
                raise ScheduleError(f'{where}: parameter {k}={v!r} is not finite')
            out[k] = f
    missing = sorted(set(schema) - set(out))
    if missing:
        raise ScheduleError(f'{where}: missing parameter {missing}')
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
            raise ScheduleError(f'{where}: {fault} belongs to batch 2, not implemented')
        if fault not in FAULTS:
            raise ScheduleError(f'{where}: unknown fault {fault!r}')
        a = _whole(f[3], where, 'start_ms')
        b = _whole(f[4], where, 'end_ms')
        rows.append(Row(rid, fault, target, a, b, _parse_params(f[5], FAULTS[fault][1], where)))
    if not rows:
        raise ScheduleError('the schedule has no row')
    return rows


def check(rows: Sequence[Row], run_ms: Optional[int], imu_source: Optional[str]) -> None:
    """Meaning of the rows for one run. run_ms: steps of the run from t0 (None refused: a window must be
    known to end inside the run). imu_source: the IMU of the frames ('plant', 'static', 'profile', None)."""
    if imu_source not in IMU_SOURCES:
        raise ScheduleError(f'imu_source {imu_source!r}')
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
            ok = all(p in WHEELS for p in parts) and len(set(parts)) == len(parts)
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
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if FAULTS[a.fault][2] != FAULTS[b.fault][2]:
                continue
            if a.start_ms < b.end_ms and b.start_ms < a.end_ms:
                if a.fault != 'enc_loss' or set(a.wheels) & set(b.wheels):
                    raise ScheduleError(f'{a.id!r} and {b.id!r} drive the same field at the same time')


def file_sha256(raw: bytes) -> str:
    """SHA-256 of the file with CRLF -> LF, so a checkout with core.autocrlf gives the same value."""
    return hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()


def load(path: str, run_ms: Optional[int], imu_source: Optional[str]) -> Schedule:
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ScheduleError(f'{path}: not UTF-8 ({e})') from None
    rows = parse(text)
    check(rows, run_ms, imu_source)
    return Schedule(tuple(rows), path, file_sha256(raw))


def rows_as_json(rows: Sequence[Row]) -> list:
    return [dict(id=r.id, fault=r.fault, target=r.target, start_ms=r.start_ms, end_ms=r.end_ms,
                 params=dict(r.params)) for r in rows]


def round_half_away(v: float) -> int:
    return int(v + 0.5) if v >= 0 else -int(-v + 0.5)


def scale_raw(raw: int, s: float) -> int:
    """gyro_scale on the int16 of the frame: round half away from zero of (1 - s) * raw, clamped to the
    int16 range of protocol._to_int16."""
    return max(-32767, min(32767, round_half_away((1.0 - s) * raw)))


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
        self.scale: Optional[float] = None
        self.any = False

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
        self.enc_mask, self.loss, self.delay, self.scale = 0, False, 0, None
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
        self.any = bool(self.on)

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
            struct.pack_into('>h', b, OFS_IMU_GZ, scale_raw(raw, a.scale))
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
        return d
