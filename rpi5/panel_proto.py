"""
Panel UART protocol — RPi5 daemon <-> panel ESP32 (WROOM + ST7796, QĐ 17, M2).

Contract only (workflow v3.1 section 9.6; Codex rounds 2-3 #15, #44, #45; Liam
approved B0 29/10; B0 review round 1 B1, B15-B18; round 2 B23-B29, Liam approved
29/10). The panel firmware (C) must mirror these layouts byte for byte (build the
frame in a byte buffer, never send a C struct with padding). The daemon runs in
its own process, never inside the 1 kHz loop. Version 3: 32-bit session id that
is never reused (daemon epoch kept on disk + session counter), next expected seq
in STATUS.

Frames (multi-byte fields little-endian; every frame ends with CRC-16/CCITT-FALSE,
poly 0x1021, init 0xFFFF, over all previous bytes of the frame, start byte included):

  CMD     panel -> daemon, 12 bytes
    0     0xCC
    1     version (PROTO_VERSION)
    2-5   sid        session id from the daemon; 0 in HELLO and only in HELLO
    6     seq        command counter of this session, panel side
    7     cmd        CMD_*
    8-9   arg        uint16 (HELLO: panel_boot; SELECT/START: profile index 0..PROFILE_MAX)
    10-11 crc

  STATUS  daemon -> panel, 28 bytes, ~10 Hz and after every command
    0     0xDD
    1     version
    2-5   sid           current session id (0 = none: the panel sends HELLO)
    6-7   daemon_boot   daemon epoch: when it changes the panel says HELLO again
    8-9   panel_boot    echo of the HELLO this session answers
    10    last_cmd_seq  seq of the last command handled (meaningless while last_result is RES_NONE)
    11    last_result   RES_* of that command; RES_NONE: no command handled in this session
    12    next_seq      the seq the daemon runs next (except STOP, see below)
    13    run_state     ST_*
    14-15 phase_ds      time in the current phase [0.1 s], saturates
    16    profile       selected profile index (PROFILE_NONE = none)
    17    outcome       OUT_* of the last finished run
    18    quality       Q_* bits
    19-20 lost          records dropped by the 1 kHz loop in this run, saturates
    21-22 data_age_ms   now - timestamp of the newest sample shown (#45), saturates; the
                        panel adds its own time since this STATUS arrived and shows NO LINK
                        when no STATUS came for NO_LINK_MS
    23    links         L_* bits
    24    error_code    0 = none (table: decided with panel M2, B19)
    25    reserved      0 (M3)
    26-27 crc

  PROFILE daemon -> panel, 28 bytes, the list the panel scrolls with NAV
    0     0xDE
    1     version
    2-5   sid
    6     index
    7     count
    8-25  name, ASCII, NUL padded (18 bytes)
    26-27 crc

Session rules (#44, B1, B15, B16, B23-B27):
  * The daemon gets a new epoch at every start (next_daemon_epoch: a counter in a file,
    written atomically) and numbers its sessions 1, 2, ...; sid = epoch << 16 | session.
    A sid is never used twice (until 65535 daemon starts), so a command carrying an old sid,
    from before a daemon restart or an earlier session, is always dropped (B23).
  * The panel sends HELLO (sid 0, arg = panel_boot, random 16 bit) at start and whenever
    daemon_boot changes in STATUS. A HELLO with sid != 0 is dropped. A HELLO with the current
    panel_boot gets the same session back (its STATUS was lost); any other panel_boot opens a
    new session. Assumption (B25): one panel on one UART, a FIFO, so an old HELLO cannot
    arrive after a newer one; no list of old panel_boot values is kept.
  * A new session starts with last_result RES_NONE and next_seq 0. The panel never resends
    an old command under a new sid.
  * One command in flight: the panel sends seq = next_seq and resends the same frame until a
    STATUS of its session shows last_cmd_seq = that seq with last_result != RES_NONE (the ACK).
    The daemon runs seq == next_seq once; the same frame again gets the stored result and is
    not run again; any other seq is not run and changes nothing (B24: a refused frame never
    moves next_seq, so a replayed START cannot run twice). If the panel sees next_seq move
    past its pending seq without an ACK, it drops that command and shows it as not done.
  * STOP pre-empts (B26): the panel may send STOP (seq = pending seq + 1) while another
    command waits for its ACK, and then stops resending that command. The daemon runs a STOP
    whatever its seq (stopping twice is harmless); a STOP ahead of next_seq moves next_seq
    past it, so the abandoned command, if it arrives later, is not run; a STOP behind
    next_seq does not move it back.
  * run() raising (B27): the command is reported RES_FAILED. A failed STOP is not stored as
    done, so its resend runs again; a failed SELECT/START is stored (its resend gets
    RES_FAILED back, it is not run again: whether it took effect is unknown).
"""
from __future__ import annotations

import os
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

PROTO_VERSION = 3

START_CMD     = 0xCC
START_STATUS  = 0xDD
START_PROFILE = 0xDE
LEN_CMD     = 12
LEN_STATUS  = 28
LEN_PROFILE = 28
NAME_LEN    = 18
_LEN = {START_CMD: LEN_CMD, START_STATUS: LEN_STATUS, START_PROFILE: LEN_PROFILE}

PROFILE_MAX  = 254
PROFILE_NONE = 0xFF
NO_LINK_MS   = 500          # panel: no STATUS for this long -> NO LINK [ĐỀ XUẤT]
EPOCH_MAX    = 0xFFFF
SESSION_MAX  = 0xFFFF

# Commands (NAV only moves the panel cursor, it is not sent)
CMD_HELLO  = 0x01
CMD_SELECT = 0x10
CMD_START  = 0x20
CMD_STOP   = 0x21
COMMANDS = (CMD_HELLO, CMD_SELECT, CMD_START, CMD_STOP)

# Results
RES_NONE        = 0     # no command handled in this session yet
RES_OK          = 1
RES_BUSY        = 2     # START while a run is active, SELECT while running
RES_BAD_PROFILE = 3
RES_NOT_IDLE    = 4     # STOP with nothing to stop (harmless)
RES_UNKNOWN     = 5
RES_FAILED      = 6     # run() raised: effect unknown (B27)

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
    struct.pack_into('<IBBH', buf, 2, sid & 0xFFFFFFFF, seq & 0xFF, cmd & 0xFF, arg & 0xFFFF)
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
    next_seq: int = 0
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


_STATUS_FMT = '<IHHBBBBHBBBHHBB'      # bytes 2..24


def encode_status(s: Status) -> bytes:
    buf = bytearray(LEN_STATUS)
    buf[0], buf[1] = START_STATUS, PROTO_VERSION
    struct.pack_into(_STATUS_FMT, buf, 2,
                     s.sid & 0xFFFFFFFF, s.daemon_boot & 0xFFFF, s.panel_boot & 0xFFFF,
                     s.last_cmd_seq & 0xFF, s.last_result & 0xFF, s.next_seq & 0xFF, s.run_state & 0xFF,
                     _sat16(s.phase_ds), s.profile & 0xFF, s.outcome & 0xFF, s.quality & 0xFF,
                     _sat16(s.lost), _sat16(s.data_age_ms), s.links & 0xFF, s.error_code & 0xFF)
    return _seal(buf)


def encode_profile(sid: int, index: int, count: int, name: str) -> bytes:
    raw = name.encode('ascii', errors='replace')[:NAME_LEN]
    buf = bytearray(LEN_PROFILE)
    buf[0], buf[1] = START_PROFILE, PROTO_VERSION
    struct.pack_into('<IBB', buf, 2, sid & 0xFFFFFFFF, index & 0xFF, count & 0xFF)
    buf[8:8 + len(raw)] = raw
    return _seal(buf)


def decode(frame: bytes):
    """One complete, checked frame -> Cmd / Status / Profile."""
    start = frame[0]
    if start == START_CMD:
        return Cmd(*struct.unpack_from('<IBBH', frame, 2))
    if start == START_STATUS:
        return Status(*struct.unpack_from(_STATUS_FMT, frame, 2))
    if start == START_PROFILE:
        sid, index, count = struct.unpack_from('<IBB', frame, 2)
        name = frame[8:8 + NAME_LEN].split(b'\x00', 1)[0].decode('ascii', errors='replace')
        return Profile(sid, index, count, name)
    raise ValueError(f"unknown start byte 0x{start:02X}")


class StreamParser:
    """Byte stream -> frames. A wrong version (checked as soon as byte 1 arrives) or a bad
    CRC drops one byte and resyncs. A partial frame whose last byte arrived more than stale_s
    ago is dropped before new bytes are added to it (B28: the age is that of the newest
    buffered byte, and dropping bytes does not make it younger)."""

    def __init__(self, accept=(START_CMD, START_STATUS, START_PROFILE), stale_s: float = 0.05,
                 clock: Callable[[], float] = time.monotonic):
        self.accept = set(accept)
        self.stale_s = stale_s
        self.clock = clock
        self._buf = bytearray()
        self._t_last: Optional[float] = None        # arrival time of the newest buffered byte
        self.bad_frames = 0
        self.skipped_bytes = 0
        self.stale_drops = 0

    def feed(self, data: bytes) -> list:
        now = self.clock()
        if self._buf and self._t_last is not None and now - self._t_last > self.stale_s:
            self.stale_drops += 1               # a cut frame (panel reset, lost bytes)
            self._buf.clear()
        if data:
            self._buf += data
            self._t_last = now
        out = []
        while self._buf:
            if self._buf[0] not in self.accept:
                del self._buf[:1]
                self.skipped_bytes += 1
                continue
            if len(self._buf) >= 2 and self._buf[1] != PROTO_VERSION:
                del self._buf[:1]
                self.bad_frames += 1
                continue
            n = _LEN[self._buf[0]]
            if len(self._buf) < n:
                break
            frame = bytes(self._buf[:n])
            if crc16(frame[:-2]) != struct.unpack_from('<H', frame, n - 2)[0]:
                self.bad_frames += 1
                del self._buf[:1]
                continue
            del self._buf[:n]
            out.append(decode(frame))
        return out


# ── daemon side: epoch, session and command rules ─────────────────────────

def next_daemon_epoch(path: str) -> int:
    """The daemon's epoch for this start: the number stored in path (missing file: 0) plus 1,
    written back before it is used (temporary file, fsync, rename), so two daemon starts never
    share an epoch even after a power cut. A file that does not hold a number in 0..65534 is
    refused: an operator fixes it, the daemon does not guess."""
    try:
        with open(path, encoding='ascii') as f:
            text = f.read().strip()
    except FileNotFoundError:
        text = '0'
    if not (text.isascii() and text.isdigit()) or not 0 <= int(text) < EPOCH_MAX:
        raise ValueError(f'{path}: {text!r} is not a daemon epoch in 0..{EPOCH_MAX - 1}')
    epoch = int(text) + 1
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='ascii') as f:
        f.write(f'{epoch}\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if os.name != 'nt':
        fd = os.open(os.path.dirname(os.path.abspath(path)) or '.', os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return epoch


def _ahead(seq: int, ref: int) -> bool:
    """seq is ref or after it in mod-256 order (less than half a turn ahead)."""
    return (seq - ref) & 0xFF < 128


class DaemonSession:
    """Which commands to run. on_cmd() calls run(cmd) -> RES_* for a command that must take
    effect, and only then: run() is the one place with side effects. The result goes into the
    next STATUS through fill()."""

    RAN, REPEATED, DROPPED_SID, BAD_SEQ, HELLO, BAD_HELLO, FAILED = (
        'ran', 'repeated', 'dropped_sid', 'bad_seq', 'hello', 'bad_hello', 'failed')

    def __init__(self, epoch: int):
        if not 1 <= epoch <= EPOCH_MAX:
            raise ValueError(f'epoch {epoch} not in 1..{EPOCH_MAX} (use next_daemon_epoch)')
        self.daemon_boot = epoch
        self._session = 0
        self.sid = 0
        self.panel_boot = 0
        self.last_seq = 0
        self.last_result = RES_NONE
        self._last_key: Optional[tuple] = None     # (seq, cmd, arg) of the last command stored as done
        self.next_seq = 0
        self.last_error: Optional[str] = None
        self.counts = {}

    def _count(self, what):
        self.counts[what] = self.counts.get(what, 0) + 1
        return what

    def _new_sid(self) -> int:
        if self._session >= SESSION_MAX:
            raise RuntimeError('session ids of this epoch used up: restart the daemon (new epoch)')
        self._session += 1
        return self.daemon_boot << 16 | self._session

    def on_cmd(self, c: Cmd, run: Callable[[Cmd], int]) -> str:
        if c.cmd == CMD_HELLO:
            if c.sid != 0:
                return self._count(self.BAD_HELLO)
            if self.sid and c.arg == self.panel_boot:
                return self._count(self.HELLO)             # retry: same session back
            self.sid = self._new_sid()
            self.panel_boot = c.arg
            self.last_seq, self.last_result = 0, RES_NONE
            self._last_key = None
            self.next_seq = 0
            return self._count(self.HELLO)
        if self.sid == 0 or c.sid != self.sid:
            return self._count(self.DROPPED_SID)
        key = (c.seq, c.cmd, c.arg)
        if key == self._last_key:
            return self._count(self.REPEATED)          # resent after a lost STATUS
        if c.cmd != CMD_STOP and c.seq != self.next_seq:
            return self._count(self.BAD_SEQ)           # not run, nothing changes (B24)
        try:
            result = run(c) if c.cmd in COMMANDS else RES_UNKNOWN
            failed = False
        except Exception as exc:                       # noqa: BLE001 (any failure of the action)
            result, failed = RES_FAILED, True
            self.last_error = f'{type(exc).__name__}: {exc}'
        self.last_seq, self.last_result = c.seq, result
        if not (failed and c.cmd == CMD_STOP):         # a failed STOP may run again (B27)
            self._last_key = key
        if c.cmd != CMD_STOP or _ahead(c.seq, self.next_seq):
            self.next_seq = (c.seq + 1) & 0xFF
        return self._count(self.FAILED if failed else self.RAN)

    def fill(self, s: Status) -> Status:
        s.sid, s.daemon_boot, s.panel_boot = self.sid, self.daemon_boot, self.panel_boot
        s.last_cmd_seq, s.last_result, s.next_seq = self.last_seq, self.last_result, self.next_seq
        return s


# ── daemon side: non-blocking UART writer (9.6) ───────────────────────────

class TxQueue:
    """write_fn(bytes) -> number of bytes the UART took now (0 when it cannot take any; it
    must never block: the adapter maps 'would block' to 0). A frame that has started going
    out is finished before any other frame, as long as the UART keeps taking bytes; never two
    frames interleaved. Only the newest STATUS waits, and a STATUS not started yet is replaced
    by a newer one. STATUS goes before PROFILE frames; PROFILE frames queue up to
    max_profiles (oldest dropped) and belong to one session."""

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
        """New session: PROFILE frames of the old one are not sent, including one taken for
        sending that has no byte out yet (B29); one already started finishes."""
        if self._current and self._current[0] == START_PROFILE and self._sent == 0:
            self._current = b''
            self.dropped_profiles += 1
        self.dropped_profiles += len(self._profiles)
        self._profiles.clear()

    def pending(self) -> bool:
        return bool(self._current or self._status or self._profiles)

    def pump(self, max_bytes: Optional[int] = None) -> int:
        """Write what the UART takes now, at most max_bytes (a bounded amount of work per call
        of the daemon loop, B29); returns the number of bytes written."""
        total = 0
        while max_bytes is None or total < max_bytes:
            if not self._current:
                if self._status is not None:
                    self._current, self._status = self._status, None
                elif self._profiles:
                    self._current = self._profiles.popleft()
                else:
                    break
                self._sent = 0
            chunk = self._current[self._sent:]
            if max_bytes is not None:
                chunk = chunk[:max_bytes - total]
            n = self.write_fn(chunk)
            if n <= 0:
                break
            self._sent += n
            total += n
            if self._sent >= len(self._current):
                self._current, self._sent = b'', 0
        return total
