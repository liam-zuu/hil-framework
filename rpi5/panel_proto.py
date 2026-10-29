"""
Panel UART protocol — RPi5 daemon <-> panel ESP32 (WROOM + ST7796, QĐ 17, M2).

Contract (workflow v3.1 section 9.6; Codex rounds 2-3 #15, #44, #45; Liam approved B0 29/10;
B0 review round 1 B1, B15-B18; round 2 B23-B29; round 3 B52-B58, N12-N16; round 4 V1-V5,
Liam approved 29/10). The panel firmware (C) must mirror these layouts byte for byte (build
the frame in a byte buffer, never send a C struct with padding) and the panel rules of
PanelClient below (the reference the C code is checked against). The daemon runs in its own
process, never inside the 1 kHz loop. Version 5: 16-bit command seq that never wraps inside
a session; STATUS carries the command code of the last frame handled, so an ACK matches a
command by seq AND code (V1, V3).

Frames (multi-byte fields little-endian; every frame ends with CRC-16/CCITT-FALSE,
poly 0x1021, init 0xFFFF, over all previous bytes of the frame, start byte included):

  CMD     panel -> daemon, 13 bytes
    0     0xCC
    1     version (PROTO_VERSION)
    2-5   sid        session id from the daemon; 0 in HELLO and only in HELLO
    6-7   seq        command number in this session (HELLO, PROFILES: 0, ignored)
    8     cmd        CMD_*
    9-10  arg        uint16 (HELLO: panel_boot; SELECT/START: profile index 0..PROFILE_MAX)
    11-12 crc

  STATUS  daemon -> panel, 30 bytes, ~10 Hz and after every command
    0     0xDD
    1     version
    2-5   sid           current session id (0 = none: the panel sends HELLO)
    6-7   daemon_boot   daemon epoch: when it changes the panel says HELLO again
    8-9   panel_boot    echo of the HELLO this session answers
    10-11 last_cmd_seq  seq of the last command handled (meaningless while last_result is RES_NONE)
    12    last_result   RES_* of that command; RES_NONE: no command handled in this session
    13-14 next_seq      the seq of the next command (>= SEQ_END - 1: no seq left for
                        SELECT/START, HELLO again)
    15    run_state     ST_*
    16-17 phase_ds      time in the current phase [0.1 s], saturates
    18    profile       selected profile index (PROFILE_NONE = none)
    19    outcome       OUT_* of the last finished run
    20    quality       Q_* bits
    21-22 lost          records dropped by the 1 kHz loop in this run, saturates
    23-24 data_age_ms   now - timestamp of the newest sample shown (#45), saturates; the
                        panel adds its own time since this STATUS arrived and shows NO LINK
                        when no STATUS came for NO_LINK_MS
    25    links         L_* bits
    26    error_code    0 = none (table: decided with panel M2, B19)
    27    last_cmd      CMD_* of the last command handled (0 while last_result is RES_NONE)
    28-29 crc

  PROFILE daemon -> panel, 28 bytes, the list the panel scrolls with NAV
    0     0xDE
    1     version
    2-5   sid
    6     index
    7     count
    8-25  name, ASCII, NUL padded (18 bytes)
    26-27 crc

Daemon rules (DaemonSession):
  * Epoch (B23, B52/N13): next_daemon_epoch() reads, increments and writes the epoch file
    under an exclusive lock, so two daemon starts never get the same epoch; a missing file is
    an error unless create=True (first start only). Preconditions it cannot check: the file
    is never restored from an older copy (that would reuse sids), and only one daemon owns
    the UART (the daemon must hold its own instance lock). sid = epoch << 16 | session.
  * HELLO (sid 0, arg = panel_boot) with the current panel_boot is a retry: same session,
    history kept; any other panel_boot opens a new session (RES_NONE, next_seq 0). A HELLO
    with sid != 0 is dropped. Assumption (B25): one panel on one UART, frames in order (FIFO),
    no copy of an old frame delivered later. Without it an old HELLO can open a new session;
    the panel then sees a sid it did not ask for and says HELLO again (PanelClient).
  * A command of another sid is dropped. Inside a session seqs are plain numbers, never
    reused (N12): SELECT/START run only at seq == next_seq and seq < SEQ_END - 1, then
    next_seq = seq + 1. STOP and SYNC are taken at next_seq or next_seq + 1 (the panel
    pre-empts a frame it sent that may still arrive) and up to seq SEQ_END - 1 (the last seq
    is kept for them, V2); next_seq then moves past them. SYNC does nothing else: it lets the
    panel skip the seq of a SELECT/START it gave up on. Any other seq is refused and changes
    nothing, so an old copy of a command (START, STOP, SELECT) never runs again.
  * The same frame as the last one handled gets the stored result back and does not run
    again; except a STOP whose result was not RES_OK / RES_NOT_IDLE: its resend runs again
    (B27, B54/N14, V4). Callbacks fail by raising or by returning RES_FAILED: both are
    RES_FAILED (effect unknown). last_cmd_seq / last_cmd / last_result are always those of
    that same last frame.
  * PROFILES (any seq) asks for the PROFILE list again: no other effect. The daemon sends the
    list after every new session and on PROFILES; an empty list is one PROFILE with count 0.
  * run() is called synchronously: it must only request actions and return at once (never
    wait for a run to end), or a STOP cannot be handled while it blocks.

Panel rules (PanelClient, B55, N14, N15): see that class.
"""
from __future__ import annotations

import os
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

PROTO_VERSION = 5

START_CMD     = 0xCC
START_STATUS  = 0xDD
START_PROFILE = 0xDE
LEN_CMD     = 13
LEN_STATUS  = 30
LEN_PROFILE = 28
NAME_LEN    = 18
_LEN = {START_CMD: LEN_CMD, START_STATUS: LEN_STATUS, START_PROFILE: LEN_PROFILE}

PROFILE_MAX  = 254
PROFILE_NONE = 0xFF
NO_LINK_MS   = 500          # panel: no STATUS for this long -> NO LINK [ĐỀ XUẤT]
EPOCH_MAX    = 0xFFFF
SESSION_MAX  = 0xFFFF
SEQ_END      = 0xFFFF       # seqs are < SEQ_END; SEQ_END - 1 is kept for STOP / SYNC (V2)

# Commands (NAV only moves the panel cursor, it is not sent)
CMD_HELLO    = 0x01
CMD_PROFILES = 0x02
CMD_SYNC     = 0x03       # no effect: skips the seq of a frame the panel gave up on
CMD_SELECT   = 0x10
CMD_START    = 0x20
CMD_STOP     = 0x21
RUN_COMMANDS = (CMD_SELECT, CMD_START, CMD_STOP)

# Results
RES_NONE        = 0     # no command handled in this session yet
RES_OK          = 1
RES_BUSY        = 2     # START while a run is active, SELECT while running
RES_BAD_PROFILE = 3
RES_NOT_IDLE    = 4     # STOP with nothing to stop (harmless)
RES_UNKNOWN     = 5
RES_FAILED      = 6     # run() raised or returned it: effect unknown (B27, B54)

# Run states (workflow 9.1)
ST_IDLE, ST_PREPARE, ST_ARM, ST_RUN, ST_STOPPING, ST_END, ST_ERROR = range(7)
ST_ACTIVE = (ST_PREPARE, ST_ARM, ST_RUN, ST_STOPPING)

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
    struct.pack_into('<IHBH', buf, 2, sid & 0xFFFFFFFF, seq & 0xFFFF, cmd & 0xFF, arg & 0xFFFF)
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
    last_cmd: int = 0


@dataclass
class Profile:
    sid: int
    index: int
    count: int
    name: str


_STATUS_FMT = '<IHHHBHBHBBBHHBBB'     # bytes 2..27


def encode_status(s: Status) -> bytes:
    buf = bytearray(LEN_STATUS)
    buf[0], buf[1] = START_STATUS, PROTO_VERSION
    struct.pack_into(_STATUS_FMT, buf, 2,
                     s.sid & 0xFFFFFFFF, s.daemon_boot & 0xFFFF, s.panel_boot & 0xFFFF,
                     s.last_cmd_seq & 0xFFFF, s.last_result & 0xFF, s.next_seq & 0xFFFF, s.run_state & 0xFF,
                     _sat16(s.phase_ds), s.profile & 0xFF, s.outcome & 0xFF, s.quality & 0xFF,
                     _sat16(s.lost), _sat16(s.data_age_ms), s.links & 0xFF, s.error_code & 0xFF,
                     s.last_cmd & 0xFF)
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
        return Cmd(*struct.unpack_from('<IHBH', frame, 2))
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
    buffered byte, and dropping bytes does not make it younger). This is a timeout on the
    silence between bytes, not a lifetime of a decoded command. Nothing bounds how long a
    complete frame waits in the UART path: a frame the panel gave up on can still arrive and
    run later (V1); the seq rules and PanelClient's STOP / SYNC handling account for it."""

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

class _FileLock:
    """Exclusive lock on a lock file, between processes and threads (flock on POSIX,
    msvcrt.locking on Windows); waits up to timeout_s."""

    def __init__(self, path: str, timeout_s: float = 10.0):
        self.path = path
        self.timeout_s = timeout_s
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        t_end = time.monotonic() + self.timeout_s
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    os.lseek(self.fd, 0, os.SEEK_SET)
                    msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= t_end:
                    os.close(self.fd)
                    raise TimeoutError(f'{self.path}: lock held by another process') from None
                time.sleep(0.01)

    def __exit__(self, *exc):
        try:
            if os.name == 'nt':
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


def next_daemon_epoch(path: str, create: bool = False, _after_read=None) -> int:
    """The daemon's epoch for this start: the number stored in path plus 1, written back
    (temporary file, fsync, rename) before it is used, all under an exclusive lock on
    path + '.lock' (B52/N13: two starts never read the same value). A missing file is an
    error unless create=True (the very first start: the file then counts from 0); a lost
    file must be set by hand above every epoch already used, never restarted from 0. A file
    that does not hold a number in 0..65534 is refused: an operator fixes it, the daemon
    does not guess. _after_read: test hook, called with the value read while the lock is held."""
    with _FileLock(path + '.lock'):
        try:
            with open(path, encoding='ascii') as f:
                text = f.read().strip()
        except FileNotFoundError:
            if not create:
                raise FileNotFoundError(f'{path}: daemon epoch file missing (first start: create=True; '
                                        'lost: set it by hand above every epoch used)') from None
            text = '0'
        if not (text.isascii() and text.isdigit()) or not 0 <= int(text) < EPOCH_MAX:
            raise ValueError(f'{path}: {text!r} is not a daemon epoch in 0..{EPOCH_MAX - 1}')
        if _after_read is not None:
            _after_read(int(text))
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


class DaemonSession:
    """Which commands to run. on_cmd() calls run(cmd) -> RES_* for a command that must take
    effect, and only then: run() is the one place with side effects. The result goes into the
    next STATUS through fill()."""

    RAN, REPEATED, DROPPED_SID, BAD_SEQ, HELLO, BAD_HELLO, FAILED, PROFILES = (
        'ran', 'repeated', 'dropped_sid', 'bad_seq', 'hello', 'bad_hello', 'failed', 'profiles')

    def __init__(self, epoch: int):
        if not 1 <= epoch <= EPOCH_MAX:
            raise ValueError(f'epoch {epoch} not in 1..{EPOCH_MAX} (use next_daemon_epoch)')
        self.daemon_boot = epoch
        self._session = 0
        self.sid = 0
        self.panel_boot = 0
        self.last_seq = 0
        self.last_cmd = 0
        self.last_result = RES_NONE
        self._last_key: Optional[tuple] = None     # (seq, cmd, arg) of the frame last_seq/last_result belong to
        self._last_retry = False                   # that frame was a STOP not done: its resend runs again
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
            self.last_seq, self.last_cmd, self.last_result = 0, 0, RES_NONE
            self._last_key, self._last_retry = None, False
            self.next_seq = 0
            return self._count(self.HELLO)
        if self.sid == 0 or c.sid != self.sid:
            return self._count(self.DROPPED_SID)
        if c.cmd == CMD_PROFILES:
            return self._count(self.PROFILES)
        key = (c.seq, c.cmd, c.arg)
        if key == self._last_key and not self._last_retry:
            return self._count(self.REPEATED)          # resent after a lost STATUS
        skip = c.cmd in (CMD_STOP, CMD_SYNC)           # may pre-empt a frame still on its way
        ok_seqs = (self.next_seq, self.next_seq + 1) if skip else (self.next_seq,)
        limit = SEQ_END if skip else SEQ_END - 1       # last seq kept for STOP / SYNC (V2)
        if (c.seq not in ok_seqs and key != self._last_key) or c.seq >= limit:
            return self._count(self.BAD_SEQ)           # not run, nothing changes (B24, N12)
        try:
            if c.cmd == CMD_SYNC:
                result = RES_OK
            else:
                result = run(c) if c.cmd in RUN_COMMANDS else RES_UNKNOWN
        except Exception as exc:                       # noqa: BLE001 (any failure of the action)
            result = RES_FAILED
            self.last_error = f'{type(exc).__name__}: {exc}'
        failed = result == RES_FAILED
        self.last_seq, self.last_cmd, self.last_result = c.seq, c.cmd, result
        self._last_key = key
        self._last_retry = c.cmd == CMD_STOP and result not in (RES_OK, RES_NOT_IDLE)   # V4
        self.next_seq = max(self.next_seq, c.seq + 1)
        return self._count(self.FAILED if failed else self.RAN)

    def fill(self, s: Status) -> Status:
        s.sid, s.daemon_boot, s.panel_boot = self.sid, self.daemon_boot, self.panel_boot
        s.last_cmd_seq, s.last_result, s.next_seq = self.last_seq, self.last_result, self.next_seq
        s.last_cmd = self.last_cmd
        return s


# ── panel side: reference state machine (B55, N14, N15, V1-V5) ────────────

RESEND_MS        = 200      # resend a command not yet acknowledged [ĐỀ XUẤT]
HELLO_RESEND_MS  = 200      # [ĐỀ XUẤT]
CMD_MAX_AGE_MS   = 3_000    # SELECT/START not acknowledged by then: given up, seq stays stale [ĐỀ XUẤT]
PROFILE_WAIT_MS  = 1_000    # PROFILE list incomplete this long after a session starts: ask again [ĐỀ XUẤT]


class PanelClient:
    """What the panel firmware does, as plain logic (no UART, no screen): the C code mirrors
    it and is checked against the same scenarios (tests/test_panel_proto.py, two ends).

      * Handshake: send HELLO(panel_boot = a random non-zero 16-bit nonce) every
        HELLO_RESEND_MS until a STATUS with sid != 0 and panel_boot == nonce arrives; other
        STATUS frames are ignored until then. Fence (W1): after every handshake the panel
        first sends SYNC at next_seq (resent until its ACK or until next_seq passes it) and
        refuses SELECT/START until then; frames are in order, so every frame sent before
        (e.g. before a panel reboot) has been handled by then and its ACK can no longer be
        taken for a new command. A nonce equal to that of the current session (a panel
        reboot, 1 in 65 535) joins that session: frames sent before the reboot and still on
        their way run as sent (as they would with any nonce); the fence keeps their results
        apart from the new commands (V3, W1).
      * Session lost: a STATUS whose daemon_boot changed, or whose sid / panel_boot is not the
        session's (a new session opened by another HELLO), or no seq left (next_seq >=
        SEQ_END - 1 with nothing of the panel still needing the last seq) -> new nonce, HELLO
        again. A pending SELECT/START is then shown 'unverified'; a STOP stays requested and
        is sent again in the new session; a stop already acknowledged stays 'acked' and is
        confirmed by the run_state of the new session's STATUS (W2); a confirmed stop is
        forgotten (V5).
      * NO LINK: no STATUS of the session for NO_LINK_MS. Commands are not sent while there
        is no link, except STOP, which keeps being resent.
      * ACK: a STATUS of the session with last_cmd_seq == seq, last_cmd == command code and
        last_result != RES_NONE.
      * One command in flight: SELECT/START get seq = next_seq of the last STATUS (refused if
        next_seq >= SEQ_END - 1) and are resent every RESEND_MS until the ACK: shown with its
        result. If next_seq moves past the seq without an ACK, it is shown 'unverified' (it
        may or may not have run: never 'not done'). If CMD_MAX_AGE_MS passes first, it is
        shown 'unverified' and its seq stays 'stale': the frame may still arrive and run, so
        that seq is never used for another command (V1). While a seq is stale, SELECT/START
        are refused and the panel sends SYNC at stale + 1 (resent until its ACK or until
        next_seq passes it); once next_seq is past the stale seq, the old frame can no longer
        run.
      * STOP pre-empts: a pending SELECT/START is given up ('unverified', its seq stale) and
        STOP goes out at pending seq + 1; a pending SYNC is replaced by STOP at the same seq;
        with a stale seq STOP goes out at stale + 1; else at next_seq. STOP is resent until
        acknowledged with RES_OK or RES_NOT_IDLE; another result counts a failure and the
        same frame is resent (the daemon runs it again, V4); if next_seq moves past it without
        an ACK, a new STOP goes out at the new next_seq. Acknowledged is not stopped:
        stop_state becomes 'confirmed' only when a later STATUS shows a run_state that is not
        active (ST_ACTIVE); a new START, an active run_state after that, or a lost session
        clears it (V5).
      * PROFILE list: kept per session; if not all `count` entries arrived PROFILE_WAIT_MS
        after the session started (or the last request), send PROFILES.
    tick(now_ms) and the on_* methods return the frames to send now.
    """

    def __init__(self, nonce: Callable[[], int]):
        self._nonce = nonce
        self.boot = self._new_boot()
        self.sid = 0
        self.daemon_boot: Optional[int] = None
        self.next_seq: Optional[int] = None
        self.last_status_ms: Optional[float] = None
        self.pending: Optional[dict] = None         # SELECT/START/SYNC in flight: seq, cmd, arg, t_first, t_sent
        self.stale: Optional[int] = None            # seq of a SELECT/START given up that may still arrive
        self.stop: Optional[dict] = None            # STOP in flight: seq (None until chosen), t_sent
        self.stop_state = 'none'                    # none / requested / acked / confirmed
        self.stop_failures = 0
        self.shown = []                             # what the screen reports: (what, cmd, result)
        self.profiles = {}
        self.profile_count: Optional[int] = None
        self._t_session: Optional[float] = None
        self._t_hello: Optional[float] = None
        self.fence = False                          # SYNC at next_seq still due after the handshake (W1)

    def _new_boot(self) -> int:
        while True:
            b = self._nonce() & 0xFFFF
            if b and b != getattr(self, 'boot', None):
                return b

    def link(self, now_ms: float) -> bool:
        return self.last_status_ms is not None and now_ms - self.last_status_ms < NO_LINK_MS

    def _give_up(self, p: dict) -> None:
        """A SELECT/START is no longer waited for; its frame may still arrive unless next_seq
        is already past it."""
        self.shown.append(('unverified', p['cmd'], None))
        if self.next_seq is None or self.next_seq <= p['seq']:
            self.stale = p['seq'] if self.stale is None else max(self.stale, p['seq'])

    # ── user actions ──
    def press(self, cmd: int, arg: int, now_ms: float) -> list:
        if cmd == CMD_STOP:
            if self.stop is not None:                         # one STOP in flight already
                return self.tick(now_ms)
            seq = None
            p = self.pending
            if p is not None and p['cmd'] == CMD_SYNC:
                seq = p['seq']                                # STOP takes the SYNC's place
            elif p is not None:
                self._give_up(p)
                seq = p['seq'] + 1
            self.pending = None
            self.stop = dict(seq=seq, t_sent=None)
            self.stop_state = 'requested'
            self.stop_failures = 0
            return self.tick(now_ms)
        if not self.sid or self.next_seq is None or not self.link(now_ms) or \
                self.pending is not None or self.stop is not None or self.stale is not None or \
                self.fence or \
                self.next_seq >= SEQ_END - 1:
            self.shown.append(('refused', cmd, None))         # busy / no link: the user retries
            return []
        if cmd == CMD_START:
            self.stop_state = 'none'                          # V5: a new run, no stop yet
        self.pending = dict(seq=self.next_seq, cmd=cmd, arg=arg, t_first=now_ms, t_sent=now_ms)
        return [encode_cmd(self.sid, self.next_seq, cmd, arg)]

    # ── time ──
    def tick(self, now_ms: float) -> list:
        out = []
        if not self.sid:
            if self._t_hello is None or now_ms - self._t_hello >= HELLO_RESEND_MS:
                self._t_hello = now_ms
                out.append(encode_cmd(0, 0, CMD_HELLO, self.boot))
            return out
        if self.stop is not None and self.next_seq is not None:
            if self.stop['seq'] is None:
                self.stop['seq'] = self.next_seq if self.stale is None else self.stale + 1
            if self.stop['t_sent'] is None or now_ms - self.stop['t_sent'] >= RESEND_MS:
                self.stop['t_sent'] = now_ms
                out.append(encode_cmd(self.sid, self.stop['seq'], CMD_STOP, 0))
        p = self.pending
        if p is not None and p['cmd'] != CMD_SYNC and now_ms - p['t_first'] >= CMD_MAX_AGE_MS:
            self._give_up(p)
            self.pending = p = None
        if p is None and self.stop is None and self.link(now_ms) and (self.stale is not None or self.fence):
            seq = self.stale + 1 if self.stale is not None else self.next_seq
            self.pending = p = dict(seq=seq, cmd=CMD_SYNC, arg=0, t_first=now_ms, t_sent=now_ms)
            out.append(encode_cmd(self.sid, seq, CMD_SYNC, 0))
        elif p is not None and self.link(now_ms) and now_ms - p['t_sent'] >= RESEND_MS:
            p['t_sent'] = now_ms
            out.append(encode_cmd(self.sid, p['seq'], p['cmd'], p['arg']))
        incomplete = self.profile_count is None or len(self.profiles) < self.profile_count
        if incomplete and self._t_session is not None:
            if now_ms - self._t_session >= PROFILE_WAIT_MS:
                self._t_session = now_ms
                out.append(encode_cmd(self.sid, 0, CMD_PROFILES, 0))
        return out

    # ── frames from the daemon ──
    def _lose_session(self, now_ms: float) -> list:
        if self.pending is not None and self.pending['cmd'] != CMD_SYNC:
            self.shown.append(('unverified', self.pending['cmd'], None))
        self.pending = None
        self.stale = None                                   # frames of the old sid are dropped
        if self.stop is not None:
            self.stop = dict(seq=None, t_sent=None)        # sent again in the new session
        elif self.stop_state == 'confirmed':
            self.stop_state = 'none'                       # 'acked' kept: confirmed in the new session (W2)
        self.fence = False
        self.boot = self._new_boot()
        self.sid, self.next_seq, self.daemon_boot = 0, None, None
        self.profiles, self.profile_count, self._t_session = {}, None, None
        self._t_hello = None
        return self.tick(now_ms)

    def on_status(self, st: Status, now_ms: float) -> list:
        if not self.sid:
            if st.sid and st.panel_boot == self.boot:
                self.sid, self.daemon_boot = st.sid, st.daemon_boot
                self._t_session = now_ms
                self.fence = True
            else:
                return []
        elif st.daemon_boot != self.daemon_boot or st.sid != self.sid or st.panel_boot != self.boot:
            return self._lose_session(now_ms)
        self.last_status_ms = now_ms
        self.next_seq = st.next_seq
        acked = st.last_result != RES_NONE

        def is_ack(seq, cmd):
            return acked and st.last_cmd_seq == seq and st.last_cmd == cmd

        p = self.pending
        if p is not None:
            if is_ack(p['seq'], p['cmd']):
                if p['cmd'] != CMD_SYNC:
                    self.shown.append(('done', p['cmd'], st.last_result))
                self.pending = None
            elif st.next_seq > p['seq']:
                if p['cmd'] != CMD_SYNC:
                    self.shown.append(('unverified', p['cmd'], None))
                self.pending = None
            if self.pending is None and p['cmd'] == CMD_SYNC:
                self.fence = False                          # every older frame is behind us
        if self.stale is not None and st.next_seq > self.stale:
            self.stale = None                               # the old frame can no longer run
        s = self.stop
        if s is not None and s['seq'] is not None:
            if is_ack(s['seq'], CMD_STOP):
                if st.last_result in (RES_OK, RES_NOT_IDLE):
                    self.stop = None
                    self.stop_state = 'acked'
                    self.fence = False                      # the STOP took the fence's place
                elif s['t_sent'] is not None and now_ms - s['t_sent'] < RESEND_MS:
                    pass                                    # same ACK seen again: one failure only
                else:
                    self.stop_failures += 1                 # resent by tick (daemon runs it again)
            elif st.next_seq > s['seq']:
                self.stop = dict(seq=st.next_seq, t_sent=None)    # passed without ACK: new STOP
        if self.stop_state == 'acked' and st.run_state not in ST_ACTIVE:
            self.stop_state = 'confirmed'
        elif self.stop_state == 'confirmed' and st.run_state in ST_ACTIVE:
            self.stop_state = 'none'                        # V5: a new run is active
        if st.next_seq >= SEQ_END or (st.next_seq >= SEQ_END - 1 and self.stop is None and
                                      self.pending is None and self.stale is None):
            return self._lose_session(now_ms)               # after the ACK of the last seq
        return self.tick(now_ms)

    def on_profile(self, pr: Profile) -> None:
        if self.sid and pr.sid == self.sid:
            self.profile_count = pr.count
            if pr.index < pr.count:                 # an empty list is one PROFILE with count 0
                self.profiles[pr.index] = pr.name


# ── daemon side: non-blocking UART writer (9.6) ───────────────────────────

class TxQueue:
    """write_fn(bytes) -> number of bytes the UART took now (0 when it cannot take any; it
    must never block: the adapter maps 'would block' to 0). A frame that has started going
    out is finished before any other frame, as long as the UART keeps taking bytes; never two
    frames interleaved. Only the newest STATUS waits, and a STATUS not started yet is replaced
    by a newer one. STATUS goes before PROFILE frames, also before one already taken for
    sending that has no byte out yet (B58/N16); PROFILE frames queue up to max_profiles
    (oldest dropped) and belong to one session."""

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
            if self._current and self._sent == 0:
                if self._current[0] == START_STATUS:
                    self.dropped_status += 1
                else:
                    self._profiles.appendleft(self._current)    # waits behind the STATUS
                self._current = b''
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
