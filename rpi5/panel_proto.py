"""
Panel UART protocol — RPi5 daemon <-> panel ESP32 (WROOM + ST7796, QĐ 17, M2).

Contract only (workflow v3.1 section 9.6; Codex rounds 2-3 #15, #44, #45; Liam
approved B0 29/10; B0 review round 1 B1, B15-B18). The panel firmware (C) must
mirror these layouts byte for byte (build the frame in a byte buffer, never send
a C struct with padding). The daemon runs in its own process, never inside the
1 kHz loop. Version 2: 16-bit session id, CRC16 instead of XOR.

Frames (multi-byte fields little-endian; every frame ends with CRC-16/CCITT-FALSE,
poly 0x1021, init 0xFFFF, over all previous bytes of the frame, start byte included):

  CMD     panel -> daemon, 10 bytes
    0     0xCC
    1     version (PROTO_VERSION)
    2-3   sid        session id from the daemon (0 only in HELLO)
    4     seq        command counter of this session, panel side
    5     cmd        CMD_*
    6-7   arg        uint16 (HELLO: panel_boot; SELECT/START: profile index 0..PROFILE_MAX)
    8-9   crc

  STATUS  daemon -> panel, 26 bytes, ~10 Hz and after every command
    0     0xDD
    1     version
    2-3   sid           current session id (0 = none: the panel sends HELLO)
    4-5   daemon_boot   random per daemon start: when it changes the panel says HELLO again
    6-7   panel_boot    echo of the HELLO this session answers
    8     last_cmd_seq
    9     last_result   RES_* of that command
    10    run_state     ST_*
    11-12 phase_ds      time in the current phase [0.1 s], saturates
    13    profile       selected profile index (PROFILE_NONE = none)
    14    outcome       OUT_* of the last finished run
    15    quality       Q_* bits
    16-17 lost          records dropped by the 1 kHz loop in this run, saturates
    18-19 data_age_ms   now - timestamp of the newest sample shown (#45), saturates; the
                        panel adds its own time since this STATUS arrived and shows NO LINK
                        when no STATUS came for NO_LINK_MS
    20    links         L_* bits
    21    error_code    0 = none (table: decided with panel M2, B19)
    22-23 reserved      0 (M3)
    24-25 crc

  PROFILE daemon -> panel, 26 bytes, the list the panel scrolls with NAV
    0     0xDE
    1     version
    2-3   sid
    4     index
    5     count
    6-23  name, ASCII, NUL padded (18 bytes)
    24-25 crc

Session rules (#44, B1, B15, B16):
  * The panel sends HELLO(panel_boot = random 16 bit, sid 0) at start and whenever
    daemon_boot changes in STATUS. The daemon opens a session with a random nonzero
    16-bit sid, different from the previous one. A repeated HELLO with the current
    panel_boot gets the same session back (lost STATUS); a HELLO with a panel_boot
    already replaced by a newer one is ignored (old HELLO arriving late).
  * After a new session the panel drops any START it has not seen acknowledged; it
    never resends an old command under the new sid.
  * One command in flight: the panel sends the next command only after STATUS shows
    last_cmd_seq = its seq; until then it resends the same frame (same seq). The daemon
    runs seq = last + 1 (mod 256) once (the first command after HELLO may use any seq); the
    same frame again gets the stored result and is not run again; any other seq is refused
    with RES_BAD_SEQ, not run, and the panel continues from that seq + 1. STOP is the
    exception: it runs whatever its seq (stopping twice is harmless).
  * A command with another sid (e.g. an old START still in the UART after a daemon or
    panel restart) is dropped.
"""
from __future__ import annotations

import random
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

PROTO_VERSION = 2

START_CMD     = 0xCC
START_STATUS  = 0xDD
START_PROFILE = 0xDE
LEN_CMD     = 10
LEN_STATUS  = 26
LEN_PROFILE = 26
NAME_LEN    = 18
_LEN = {START_CMD: LEN_CMD, START_STATUS: LEN_STATUS, START_PROFILE: LEN_PROFILE}

PROFILE_MAX  = 254
PROFILE_NONE = 0xFF
NO_LINK_MS   = 500          # panel: no STATUS for this long -> NO LINK [ĐỀ XUẤT]

# Commands (NAV only moves the panel cursor, it is not sent)
CMD_HELLO  = 0x01
CMD_SELECT = 0x10
CMD_START  = 0x20
CMD_STOP   = 0x21
COMMANDS = (CMD_HELLO, CMD_SELECT, CMD_START, CMD_STOP)

# Results
RES_NONE        = 0
RES_OK          = 1
RES_BUSY        = 2     # START while a run is active, SELECT while running
RES_BAD_PROFILE = 3
RES_NOT_IDLE    = 4     # STOP with nothing to stop (harmless)
RES_UNKNOWN     = 5
RES_BAD_SEQ     = 6     # not the next seq of this session (command not run)

# Run states (workflow 9.1)
ST_IDLE, ST_PREPARE, ST_ARM, ST_RUN, ST_STOPPING, ST_END, ST_ERROR = range(7)

# Outcomes of the last run (summary.json 'outcome')
OUT_NONE, OUT_COMPLETED, OUT_USER_STOP, OUT_ERROR = range(4)

# Quality bits
Q_LOST         = 1 << 0     # 1 kHz loop dropped records
Q_SP_STALE     = 1 << 1     # setpoint older than allowed
Q_SPI_STALE    = 1 << 2     # no fresh SPI answer from the H7
Q_DUT_UNCONF   = 1 << 3     # DUT READY / stop not confirmed
Q_DATA_INVALID = 1 << 4

# Link bits (indicators on the panel)
L_RPI   = 1 << 0            # daemon alive (implicit, set by the daemon)
L_H7    = 1 << 1            # fresh SPI answers
L_ESP32 = 1 << 2            # known only after the H7 status bit exists: else '—'
L_ESP32_KNOWN = 1 << 3


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _seal(buf: bytearray) -> bytes:
    struct.pack_into('<H', buf, len(buf) - 2, crc16(bytes(buf[:-2])))
    return bytes(buf)


def _sat16(v) -> int:
    return max(0, min(0xFFFF, int(v)))


# ── encode / decode ───────────────────────────────────────────────────────

def encode_cmd(sid: int, seq: int, cmd: int, arg: int = 0) -> bytes:
    buf = bytearray(LEN_CMD)
    buf[0], buf[1] = START_CMD, PROTO_VERSION
    struct.pack_into('<HBBH', buf, 2, sid & 0xFFFF, seq & 0xFF, cmd & 0xFF, arg & 0xFFFF)
    return _seal(buf)


@dataclass
class Cmd:
    sid: int
    seq: int
    cmd: int
    arg: int


@dataclass
class Status:
    sid: int = 0
    daemon_boot: int = 0
    panel_boot: int = 0
    last_cmd_seq: int = 0
    last_result: int = RES_NONE
    run_state: int = ST_IDLE
    phase_ds: int = 0
    profile: int = PROFILE_NONE
    outcome: int = OUT_NONE
    quality: int = 0
    lost: int = 0
    data_age_ms: int = 0
    links: int = L_RPI
    error_code: int = 0


@dataclass
class Profile:
    sid: int
    index: int
    count: int
    name: str


_STATUS_FMT = '<HHHBBBHBBBHHBB'      # bytes 2..21


def encode_status(s: Status) -> bytes:
    buf = bytearray(LEN_STATUS)
    buf[0], buf[1] = START_STATUS, PROTO_VERSION
    struct.pack_into(_STATUS_FMT, buf, 2,
                     s.sid & 0xFFFF, s.daemon_boot & 0xFFFF, s.panel_boot & 0xFFFF,
                     s.last_cmd_seq & 0xFF, s.last_result & 0xFF, s.run_state & 0xFF,
                     _sat16(s.phase_ds), s.profile & 0xFF, s.outcome & 0xFF, s.quality & 0xFF,
                     _sat16(s.lost), _sat16(s.data_age_ms), s.links & 0xFF, s.error_code & 0xFF)
    return _seal(buf)


def encode_profile(sid: int, index: int, count: int, name: str) -> bytes:
    raw = name.encode('ascii', errors='replace')[:NAME_LEN]
    buf = bytearray(LEN_PROFILE)
    buf[0], buf[1] = START_PROFILE, PROTO_VERSION
    struct.pack_into('<HBB', buf, 2, sid & 0xFFFF, index & 0xFF, count & 0xFF)
    buf[6:6 + len(raw)] = raw
    return _seal(buf)


def decode(frame: bytes):
    """One complete, checked frame -> Cmd / Status / Profile."""
    start = frame[0]
    if start == START_CMD:
        return Cmd(*struct.unpack_from('<HBBH', frame, 2))
    if start == START_STATUS:
        return Status(*struct.unpack_from(_STATUS_FMT, frame, 2))
    if start == START_PROFILE:
        sid, index, count = struct.unpack_from('<HBB', frame, 2)
        name = frame[6:6 + NAME_LEN].split(b'\x00', 1)[0].decode('ascii', errors='replace')
        return Profile(sid, index, count, name)
    raise ValueError(f"unknown start byte 0x{start:02X}")


class StreamParser:
    """Byte stream -> frames. A wrong version (checked as soon as byte 1 arrives), a bad CRC,
    or a partial frame older than stale_s drops one byte and resyncs."""

    def __init__(self, accept=(START_CMD, START_STATUS, START_PROFILE), stale_s: float = 0.05,
                 clock: Callable[[], float] = time.monotonic):
        self.accept = set(accept)
        self.stale_s = stale_s
        self.clock = clock
        self._buf = bytearray()
        self._t_first: Optional[float] = None       # arrival time of the oldest buffered byte
        self.bad_frames = 0
        self.skipped_bytes = 0
        self.stale_drops = 0

    def _drop(self, n: int = 1) -> None:
        del self._buf[:n]
        self._t_first = self.clock() if self._buf else None

    def feed(self, data: bytes) -> list:
        now = self.clock()
        if data and not self._buf:
            self._t_first = now
        self._buf += data
        out = []
        while self._buf:
            if self._buf[0] not in self.accept:
                self._drop()
                self.skipped_bytes += 1
                continue
            if len(self._buf) >= 2 and self._buf[1] != PROTO_VERSION:
                self._drop()
                self.bad_frames += 1
                continue
            n = _LEN[self._buf[0]]
            if len(self._buf) < n:
                if self._t_first is not None and now - self._t_first > self.stale_s:
                    self._drop()
                    self.stale_drops += 1
                    continue
                break
            frame = bytes(self._buf[:n])
            if crc16(frame[:-2]) != struct.unpack_from('<H', frame, n - 2)[0]:
                self.bad_frames += 1
                self._drop()
                continue
            self._drop(n)
            out.append(decode(frame))
        return out


# ── daemon side: session and command rules ────────────────────────────────

class DaemonSession:
    """Which commands to run. on_cmd() calls run(cmd) -> RES_* for a command that must take
    effect, and only then: run() is the one place with side effects. The result goes into the
    next STATUS through fill()."""

    RAN, REPEATED, DROPPED_SID, BAD_SEQ, HELLO, HELLO_STALE = (
        'ran', 'repeated', 'dropped_sid', 'bad_seq', 'hello', 'hello_stale')

    def __init__(self, daemon_boot: Optional[int] = None, rng: Callable[[], int] = None):
        self._rng = rng or (lambda: random.randrange(1, 0x10000))
        self.daemon_boot = daemon_boot if daemon_boot is not None else self._rng()
        self.sid = 0
        self.panel_boot = 0
        self._old_panel_boots: deque = deque(maxlen=16)
        self.last_seq: Optional[int] = None
        self.last_result = RES_NONE
        self._last_key: Optional[tuple] = None     # (seq, cmd, arg) of the last command answered
        self._expect: Optional[int] = None         # next seq that runs (None: any, first after HELLO)
        self.counts = {}

    def _count(self, what):
        self.counts[what] = self.counts.get(what, 0) + 1
        return what

    def _new_sid(self) -> int:
        while True:
            s = self._rng() & 0xFFFF
            if s not in (0, self.sid):
                return s

    def on_cmd(self, c: Cmd, run: Callable[[Cmd], int]) -> str:
        if c.cmd == CMD_HELLO:
            if self.sid and c.arg == self.panel_boot:
                return self._count(self.HELLO)             # retry: same session back
            if c.arg in self._old_panel_boots:
                return self._count(self.HELLO_STALE)       # late HELLO of a replaced panel boot
            if self.sid:
                self._old_panel_boots.append(self.panel_boot)
            self.sid = self._new_sid()
            self.panel_boot = c.arg
            self.last_seq = None
            self._last_key = None
            self._expect = None
            self.last_result = RES_OK
            return self._count(self.HELLO)
        if self.sid == 0 or c.sid != self.sid:
            return self._count(self.DROPPED_SID)
        key = (c.seq, c.cmd, c.arg)
        if key == self._last_key:
            return self._count(self.REPEATED)          # resent after a lost STATUS
        self._last_key, self.last_seq = key, c.seq
        if c.cmd != CMD_STOP and self._expect is not None and c.seq != self._expect:
            self.last_result = RES_BAD_SEQ
            self._expect = (c.seq + 1) & 0xFF
            return self._count(self.BAD_SEQ)
        self._expect = (c.seq + 1) & 0xFF
        self.last_result = run(c) if c.cmd in COMMANDS else RES_UNKNOWN
        return self._count(self.RAN)

    def fill(self, s: Status) -> Status:
        s.sid, s.daemon_boot, s.panel_boot = self.sid, self.daemon_boot, self.panel_boot
        s.last_cmd_seq = self.last_seq if self.last_seq is not None else 0
        s.last_result = self.last_result
        return s


# ── daemon side: non-blocking UART writer (9.6) ───────────────────────────

class TxQueue:
    """write_fn(bytes) -> number of bytes the UART took now (0 when it cannot take any; it
    must never block: the adapter maps 'would block' to 0). A frame that has started going
    out is always finished first; never half a frame. Only the newest STATUS waits, and a
    STATUS not started yet is replaced by a newer one. STATUS goes before PROFILE frames;
    PROFILE frames queue up to max_profiles (oldest dropped) and belong to one session."""

    def __init__(self, write_fn: Callable[[bytes], int], max_profiles: int = 64):
        self.write_fn = write_fn
        self._current = b''          # frame being sent
        self._sent = 0               # bytes of _current already out
        self._status: Optional[bytes] = None
        self._profiles: deque = deque()
        self.max_profiles = max_profiles
        self.dropped_status = 0
        self.dropped_profiles = 0

    def put(self, frame: bytes) -> None:
        if frame[0] == START_STATUS:
            if self._current and self._current[0] == START_STATUS and self._sent == 0:
                self._current, self.dropped_status = b'', self.dropped_status + 1
            if self._status is not None:
                self.dropped_status += 1
            self._status = frame
        else:
            if len(self._profiles) >= self.max_profiles:
                self._profiles.popleft()
                self.dropped_profiles += 1
            self._profiles.append(frame)

    def clear_profiles(self) -> None:
        """New session: PROFILE frames of the old one are not sent (a started one finishes)."""
        self.dropped_profiles += len(self._profiles)
        self._profiles.clear()

    def pending(self) -> bool:
        return bool(self._current or self._status or self._profiles)

    def pump(self) -> None:
        while True:
            if not self._current:
                if self._status is not None:
                    self._current, self._status = self._status, None
                elif self._profiles:
                    self._current = self._profiles.popleft()
                else:
                    return
                self._sent = 0
            n = self.write_fn(self._current[self._sent:])
            if n <= 0:
                return
            self._sent += n
            if self._sent >= len(self._current):
                self._current, self._sent = b'', 0
