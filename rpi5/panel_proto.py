"""
Panel UART protocol — RPi5 daemon <-> panel ESP32 (WROOM + ST7796, QĐ 17, M2).

Contract only (workflow v3.1 section 9.6, Codex rounds 2-3 #15, #44, #45; Liam
approved B0 29/10). The panel firmware (C) must mirror these layouts byte for
byte; the daemon and the run manager use the classes below. No timing here:
the daemon runs in its own process, never inside the 1 kHz loop.

Frames (all multi-byte fields little-endian, every frame ends with XOR of all
previous bytes of the frame, start byte included):

  CMD     panel -> daemon, 8 bytes
    0  0xCC
    1  version (PROTO_VERSION)
    2  sid        session id from the daemon (0 only in HELLO)
    3  seq        command counter of this session, panel side
    4  cmd        CMD_*
    5-6 arg       uint16 (HELLO: panel_boot; SELECT/START: profile index)
    7  xor

  STATUS  daemon -> panel, 24 bytes, ~10 Hz and after every command
    0  0xDD
    1  version
    2  sid           current session id (0 = no session yet: panel sends HELLO)
    3-4 daemon_boot  random per daemon start: when it changes the panel says HELLO again
    5-6 panel_boot   echo of the HELLO this session answers
    7  last_cmd_seq
    8  last_result   RES_*
    9  run_state     ST_*
    10-11 phase_ds   time in the current phase [0.1 s], saturates
    12 profile       selected profile index (0xFF = none)
    13 outcome       OUT_* of the last finished run
    14 quality       Q_* bits
    15-16 lost       records dropped by the 1 kHz loop in this run, saturates
    17-18 data_age_ms  now - timestamp of the newest sample shown (#45), saturates
    19 links         L_* bits
    20 error_code    0 = none
    21-22 reserved   0 (M3)
    23 xor

  PROFILE daemon -> panel, 24 bytes, the list the panel scrolls with NAV
    0  0xDE
    1  version
    2  sid
    3  index
    4  count
    5-22 name, ASCII, NUL padded (18 bytes)
    23 xor

Sessions (#44): the panel sends HELLO(panel_boot = random) at start and
whenever daemon_boot changes; the daemon opens a new sid and echoes
panel_boot. Commands with another sid (e.g. an old START still in the UART
after a restart) are dropped. A seq that is not newer than the last accepted
one gets the stored result back and is not run again. STOP is always safe to
repeat (each new seq is a new STOP).
"""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from typing import Callable, Optional

PROTO_VERSION = 1

START_CMD     = 0xCC
START_STATUS  = 0xDD
START_PROFILE = 0xDE
LEN_CMD     = 8
LEN_STATUS  = 24
LEN_PROFILE = 24
NAME_LEN    = 18
_LEN = {START_CMD: LEN_CMD, START_STATUS: LEN_STATUS, START_PROFILE: LEN_PROFILE}

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


def xor8(data: bytes) -> int:
    x = 0
    for b in data:
        x ^= b
    return x


def _seal(buf: bytearray) -> bytes:
    buf[-1] = xor8(buf[:-1])
    return bytes(buf)


def _sat16(v) -> int:
    return max(0, min(0xFFFF, int(v)))


# ── encode / decode ───────────────────────────────────────────────────────

def encode_cmd(sid: int, seq: int, cmd: int, arg: int = 0) -> bytes:
    buf = bytearray(LEN_CMD)
    buf[0], buf[1], buf[2], buf[3], buf[4] = START_CMD, PROTO_VERSION, sid & 0xFF, seq & 0xFF, cmd & 0xFF
    struct.pack_into('<H', buf, 5, arg & 0xFFFF)
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
    profile: int = 0xFF
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


def encode_status(s: Status) -> bytes:
    buf = bytearray(LEN_STATUS)
    buf[0], buf[1], buf[2] = START_STATUS, PROTO_VERSION, s.sid & 0xFF
    struct.pack_into('<HHBBBHBBBHHBB', buf, 3,
                     s.daemon_boot & 0xFFFF, s.panel_boot & 0xFFFF,
                     s.last_cmd_seq & 0xFF, s.last_result & 0xFF, s.run_state & 0xFF,
                     _sat16(s.phase_ds), s.profile & 0xFF, s.outcome & 0xFF, s.quality & 0xFF,
                     _sat16(s.lost), _sat16(s.data_age_ms), s.links & 0xFF, s.error_code & 0xFF)
    return _seal(buf)


def encode_profile(sid: int, index: int, count: int, name: str) -> bytes:
    raw = name.encode('ascii', errors='replace')[:NAME_LEN]
    buf = bytearray(LEN_PROFILE)
    buf[0], buf[1], buf[2], buf[3], buf[4] = START_PROFILE, PROTO_VERSION, sid & 0xFF, index & 0xFF, count & 0xFF
    buf[5:5 + len(raw)] = raw
    return _seal(buf)


def decode(frame: bytes):
    """One complete, checked frame -> Cmd / Status / Profile."""
    start = frame[0]
    if start == START_CMD:
        return Cmd(frame[2], frame[3], frame[4], struct.unpack_from('<H', frame, 5)[0])
    if start == START_STATUS:
        f = struct.unpack_from('<HHBBBHBBBHHBB', frame, 3)
        return Status(frame[2], *f)
    if start == START_PROFILE:
        name = frame[5:5 + NAME_LEN].split(b'\x00', 1)[0].decode('ascii', errors='replace')
        return Profile(frame[2], frame[3], frame[4], name)
    raise ValueError(f"unknown start byte 0x{start:02X}")


class StreamParser:
    """Byte stream -> frames. Resyncs on a bad length/version/XOR by dropping one byte."""

    def __init__(self, accept=(START_CMD, START_STATUS, START_PROFILE)):
        self.accept = set(accept)
        self._buf = bytearray()
        self.bad_frames = 0
        self.skipped_bytes = 0

    def feed(self, data: bytes) -> list:
        self._buf += data
        out = []
        while self._buf:
            if self._buf[0] not in self.accept:
                del self._buf[0]
                self.skipped_bytes += 1
                continue
            n = _LEN[self._buf[0]]
            if len(self._buf) < n:
                break
            frame = bytes(self._buf[:n])
            if frame[1] != PROTO_VERSION or xor8(frame[:-1]) != frame[-1]:
                self.bad_frames += 1
                del self._buf[0]
                continue
            del self._buf[:n]
            out.append(decode(frame))
        return out


# ── daemon side: session and command rules (#44) ─────────────────────────

class DaemonSession:
    """Which commands to run. The caller executes the returned action and reports
    the result into the next STATUS (fields sid, panel_boot, last_cmd_seq, last_result)."""

    def __init__(self, daemon_boot: Optional[int] = None, rng: Callable[[], int] = None):
        rng = rng or (lambda: random.randrange(1, 0x10000))
        self.daemon_boot = daemon_boot if daemon_boot is not None else rng()
        self.sid = 0
        self.panel_boot = 0
        self.last_seq: Optional[int] = None
        self.last_result = RES_NONE
        self.dropped_sid = 0
        self.repeats = 0

    def on_cmd(self, c: Cmd, judge: Callable[[Cmd], int]) -> Optional[Cmd]:
        """judge(cmd) -> RES_* decides a new command (and starts it when RES_OK).
        Returns the command that was judged, or None when it was dropped or repeated."""
        if c.cmd == CMD_HELLO:
            self.sid = (self.sid % 255) + 1          # 1..255, never 0
            self.panel_boot = c.arg
            self.last_seq = None
            self.last_result = RES_OK
            return c
        if self.sid == 0 or c.sid != self.sid:
            self.dropped_sid += 1                    # old session: never run (old START after restart)
            return None
        if self.last_seq is not None and ((c.seq - self.last_seq) & 0xFF) not in range(1, 128):
            self.repeats += 1                        # not newer: answer again, do not run again
            return None
        self.last_seq = c.seq
        self.last_result = judge(c) if c.cmd in COMMANDS else RES_UNKNOWN
        return c

    def fill(self, s: Status) -> Status:
        s.sid, s.daemon_boot, s.panel_boot = self.sid, self.daemon_boot, self.panel_boot
        s.last_cmd_seq = self.last_seq if self.last_seq is not None else 0
        s.last_result = self.last_result
        return s


# ── daemon side: non-blocking UART writer (9.6) ───────────────────────────

class TxQueue:
    """write_fn(bytes) -> number of bytes the UART took (may be 0, never blocks).
    A frame that went out partly is finished first; never half a frame. Only the
    newest STATUS waits; an older waiting STATUS is dropped. PROFILE frames queue."""

    def __init__(self, write_fn: Callable[[bytes], int]):
        self.write_fn = write_fn
        self._current = b''          # rest of the frame being sent
        self._status: Optional[bytes] = None
        self._others: list = []
        self.dropped_status = 0

    def put(self, frame: bytes) -> None:
        if frame[0] == START_STATUS:
            if self._status is not None:
                self.dropped_status += 1
            self._status = frame
        else:
            self._others.append(frame)

    def pending(self) -> bool:
        return bool(self._current or self._status or self._others)

    def pump(self) -> None:
        while True:
            if not self._current:
                if self._others:
                    self._current = self._others.pop(0)
                elif self._status is not None:
                    self._current, self._status = self._status, None
                else:
                    return
            n = self.write_fn(self._current)
            if n <= 0:
                return
            self._current = self._current[n:]
