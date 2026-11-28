"""
Panel link — the RPi5 side of the panel UART (protocol v7), in its own process, never inside
the 1 kHz loop (QĐ 17; workflow 9.6). Liam 04/11, PANEL session: "ổn, làm theo đề xuất về phía
pi": the PANEL session writes this link with a thin run-control interface; the CODE session
plugs the real run manager (hil_daemon / hil_ctl, stage E) into that interface.

    panel (ESP32-S3) <--UART ttyAMA0--> PanelLink <--RunControl--> run manager (stage E)

What PanelLink does: frames in/out (StreamParser, TxQueue), the session and seq rules
(DaemonSession), STATUS every STATUS_PERIOD_S and after every command, INFO and EXTRA every
INFO_PERIOD_S and when they change, the PROFILE list after a new session and on PROFILES.
What it does not do: decide anything about a run. Every command goes to RunControl.

RunControl contract (what stage E implements):
  * profiles() -> list of test names (ASCII, <= 18 chars shown); index = position. Called on
    every new panel session and on PROFILES; the list must not change while a run is active.
  * select(index) / start(index) / stop() -> RES_* (panel_proto). The link handles these in
    its UART loop. START may take one bounded DUT query or one reset pulse. A run and its
    STOP escalation continue in the background after the command returns.
    Raising or returning RES_FAILED both mean "effect unknown". stop() with nothing to stop
    returns RES_NOT_IDLE; stop() in ST_ERROR acknowledges the fault (the run manager then goes
    to ST_IDLE) [ĐỀ XUẤT: rule of this link and of the fake Pi, not decided for stage E].
  * snapshot() -> RunSnapshot, cheap and non-blocking (read a copy the run manager keeps).
    Numbers it does not know stay None and are shown as '-' / not valid on the panel.
  * run_no / result_run in the snapshot (Codex round 3 R1-R5, Liam 04/11: "(a) Sửa gốc + R3"):
    the run manager counts every run it starts, from the panel or from the PC
    (panel_proto.next_run_no: 1..65535); run numbers never wrap inside one daemon run: at
    65535 the run manager refuses START (RES_FAILED, state unchanged) until the daemon
    restarts with a new epoch (Codex round 5 T1, T2, Liam 05/11: "(a) Chặn quay vòng");
    run_no is the current run, or the last one
    when none is active, and goes into every STATUS (Codex round 4, Liam 04/11: "A: run_no vào
    STATUS"); result_run is the run the result lines belong to (0 = none), in INFO. Result
    lines with result_run 0 are not sent as valid: the panel could not tell whose they are.

Run it on the Pi (only when stage E provides a RunControl; --demo uses the FAKE run):
    python panel_link.py --port /dev/ttyAMA0 --epoch-file ~/.hil_panel_epoch [--create-epoch] --demo
"""
from __future__ import annotations

import argparse
from collections import deque
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import panel_proto as pp

STATUS_PERIOD_S = 0.1       # workflow 9.6: STATUS ~10 Hz
INFO_PERIOD_S = 1.0         # v6: INFO ~1 Hz and on change
INFO_MIN_GAP_S = 0.2        # a changing INFO (loop_us moves all the time) at most 5 per second
TRACK_PERIOD_S = 10.0
WIRE_BUDGET_MAX = 128       # bytes handed to the UART ahead of the wire at most (Codex Q9)
WIRE_RATE_FACTOR = 0.95     # hand bytes over at 95 % of the line rate (baud / 10 bytes per second)


@dataclass
class RunSnapshot:
    run_state: int = pp.ST_IDLE
    phase_s: float = 0.0                 # time in the current phase
    profile: Optional[int] = None        # selected / running test index
    outcome: int = pp.OUT_NONE           # of the last finished run
    quality: int = 0                     # Q_* bits
    lost: int = 0                        # records dropped by the 1 kHz loop in this run
    data_age_ms: Optional[int] = None    # now - timestamp of the newest sample (None: unknown)
    h7_ok: bool = False                  # fresh SPI answers from the H7
    esp32_ok: Optional[bool] = None      # None until the H7 status bit exists
    error_code: int = 0
    error_text: str = ''
    ctrl: str = ''                       # controller on the DUT
    ctrl_src: int = pp.CTRL_SRC_NONE
    loop_us: Optional[int] = None        # 1 kHz loop turnaround (statistics window), None: unknown
    overruns: Optional[int] = None
    result: Sequence[str] = field(default_factory=tuple)   # RESULT lines of the last finished run
    run_no: int = 0                      # current run, or the last one (0 = none yet)
    result_run: int = 0                  # run the result lines belong to (0 = none)
    sha_state: int = pp.SHA_NONE
    dut_sha: str = ''
    dut_env: str = ''
    fault_text: str = ''                 # populated while a scheduled window is open
    reset_index: int = pp.PROFILE_NONE
    note: str = ''                       # RESET DUT feedback, independent of the run result
    check_code: int = pp.CHK_NONE
    check_run: int = 0
    check_text: str = ''
    tracks: dict = field(default_factory=dict)  # series -> immutable chunks of (x_mm, y_mm)


class RunControl(Protocol):
    def profiles(self) -> list: ...
    def select(self, index: int) -> int: ...
    def start(self, index: int) -> int: ...
    def stop(self) -> int: ...
    def snapshot(self) -> RunSnapshot: ...


def status_of(snap: RunSnapshot, sess: pp.DaemonSession) -> pp.Status:
    links = pp.L_RPI | (pp.L_H7 if snap.h7_ok else 0)
    if snap.esp32_ok is not None:
        links |= pp.L_ESP32_KNOWN | (pp.L_ESP32 if snap.esp32_ok else 0)
    st = pp.Status(run_state=snap.run_state, phase_ds=int(max(0.0, snap.phase_s) * 10),
                   profile=pp.PROFILE_NONE if snap.profile is None else snap.profile,
                   outcome=snap.outcome, quality=snap.quality, lost=snap.lost,
                   data_age_ms=0xFFFF if snap.data_age_ms is None else snap.data_age_ms,
                   links=links, error_code=snap.error_code, run_no=snap.run_no)
    return sess.fill(st)


def info_of(snap: RunSnapshot, sid: int) -> pp.Info:
    # no result while a run is active, whatever the run manager says (contract v6, Codex Q1)
    # and no result lines without the run they belong to (R1-R5, S1-S6)
    result = () if snap.run_state in pp.ST_ACTIVE or not snap.result_run else tuple(snap.result)[:pp.RESULT_LINES]
    valid = any(result)
    flags = (pp.I_LOOP_VALID if snap.loop_us is not None else 0) | (pp.I_RESULT_VALID if valid else 0)
    return pp.Info(sid=sid, loop_us=snap.loop_us or 0, overruns=snap.overruns or 0, flags=flags,
                   ctrl_src=snap.ctrl_src, ctrl=snap.ctrl, result=result, error_text=snap.error_text,
                   result_run=snap.result_run if valid else 0)


def extra_of(snap: RunSnapshot, sid: int) -> pp.Extra:
    flags = pp.X_FAULT_ACTIVE if snap.fault_text else 0
    show_check = snap.check_run and (not snap.note or snap.check_code == pp.CHK_RUNNING)
    return pp.Extra(sid=sid, flags=flags | (pp.X_CHECK_VALID if show_check else 0), fault_text=snap.fault_text,
                    sha_state=snap.sha_state, dut_sha=snap.dut_sha, dut_env=snap.dut_env,
                    reset_index=snap.reset_index, check_run=snap.check_run if show_check else 0,
                    check_code=snap.check_code if show_check else pp.CHK_NONE,
                    check_text=snap.check_text if show_check else snap.note)


def _due(now, sent_at, frame, previous) -> bool:
    return (previous is None or now - sent_at >= INFO_PERIOD_S
            or (frame != previous and now - sent_at >= INFO_MIN_GAP_S))


def nonblocking_writer(port) -> Callable[[bytes], int]:
    """write(bytes) -> bytes the OS took now, never waiting (TxQueue contract; Codex Q6:
    pySerial 3.5 on POSIX loops on EAGAIN even with write_timeout=0). POSIX: one os.write on
    the port's descriptor (pySerial opens it O_NONBLOCK), EAGAIN -> 0. Elsewhere (the Windows
    bench): port.write with write_timeout=0."""
    if os.name != 'posix':
        return lambda data: port.write(data) or 0
    fd = port.fileno()

    def write(data: bytes) -> int:
        try:
            return os.write(fd, data)
        except BlockingIOError:
            return 0
    return write


class PanelLink:
    """write(bytes) -> bytes taken now (never blocks); clock() -> seconds. Call feed() with
    what the UART read and poll() every few ms."""

    def __init__(self, epoch: int, run: RunControl, write: Callable[[bytes], int],
                 clock: Callable[[], float] = time.monotonic, baud: int = 115200):
        self.sess = pp.DaemonSession(epoch)
        self.run = run
        self.clock = clock
        self.parser = pp.StreamParser(accept=(pp.START_CMD,), clock=clock)
        self.tx = pp.TxQueue(write, max_profiles=pp.PROFILE_MAX + 1)   # a whole list fits (P14)
        self.t_status = self.t_info = self.t_extra = None
        self.rate = baud / 10 * WIRE_RATE_FACTOR     # bytes per second handed to the UART
        self._budget, self._t_budget = float(WIRE_BUDGET_MAX), None
        self.last_info: Optional[bytes] = None
        self.last_extra: Optional[bytes] = None
        self._write = write
        self._pending_extra = self._tail = b''
        self._info_run = self._track_run = 0
        self._tracks = deque()
        self._t_tracks = None
        self._track_allowance = 0
        self.log = []                       # (t, cmd, seq, result of on_cmd) for the operator

    def _exec(self, c: pp.Cmd) -> int:
        if c.cmd == pp.CMD_SELECT:
            return self.run.select(c.arg)
        if c.cmd == pp.CMD_START:
            return self.run.start(c.arg)
        if c.cmd == pp.CMD_STOP:
            return self.run.stop()
        return pp.RES_UNKNOWN

    def _send_profiles(self) -> None:
        names = list(self.run.profiles())[:pp.PROFILE_MAX + 1]
        if not names:
            self.tx.put(pp.encode_profile(self.sess.sid, 0, 0, ''))     # empty list: count 0
        for i, name in enumerate(names):
            self.tx.put(pp.encode_profile(self.sess.sid, i, len(names), name))

    def feed(self, data: bytes) -> None:
        now = self.clock()
        for c in self.parser.feed(data):
            r = self.sess.on_cmd(c, self._exec)
            self.log.append((now, c.cmd, c.seq, r))
            if r == 'hello':
                self.tx.clear_session()
                self.last_info = self.last_extra = None
                self._pending_extra = b''            # any frame already on the wire must finish
                self._info_run = 0
                self._track_run = None  # discard unsent chunks even if the new session has no result
            self.tx.put(pp.encode_status(status_of(self.run.snapshot(), self.sess)))
            if r in ('hello', 'profiles'):
                self._send_profiles()

    def poll(self) -> None:
        now = self.clock()
        snap = None
        if self.t_status is None or now - self.t_status >= STATUS_PERIOD_S:
            self.t_status = now
            snap = self.run.snapshot()
            self.tx.put(pp.encode_status(status_of(snap, self.sess)))
            self._track_allowance = 2  # usable only once TxQueue has drained this STATUS
        if self.sess.sid:
            snap = snap or self.run.snapshot()
            result_info = info_of(snap, self.sess.sid)
            info = pp.encode_info(result_info)
            if _due(now, self.t_info, info, self.last_info):
                self.t_info, self.last_info = now, info
                self.tx.put(info)
                self._info_run = result_info.result_run
            extra = pp.encode_extra(extra_of(snap, self.sess.sid))
            self._pending_extra = extra if _due(now, self.t_extra, extra, self.last_extra) else b''
            run_no = result_info.result_run if result_info.result_run == self._info_run and snap.tracks else 0
            self._schedule_tracks(now, run_no, snap.tracks)
        self._pump(now)

    def _schedule_tracks(self, now, run_no, tracks):
        if run_no != self._track_run:
            self._track_run, self._t_tracks = run_no, None
            self._tracks.clear()
        if run_no and not self._tracks and (self._t_tracks is None or now - self._t_tracks >= TRACK_PERIOD_S):
            self._t_tracks = now
            for series, chunks in tracks.items():
                self._tracks.extend(pp.Track(self.sess.sid, run_no, series, i, len(chunks), points)
                                    for i, points in enumerate(chunks))

    def _write_frame(self, frame):
        """Keep the unwritten tail of a started frame across polls, including HELLO and STOP."""
        rest = frame
        while rest and self._budget >= 1:
            n = self._write(rest[:int(self._budget)])
            if not n:
                break
            rest = rest[n:]
            self._budget -= n
        if rest == frame:
            return False
        self._tail = rest
        return True

    def _pump(self, now: float) -> None:
        '''Hand bytes to the UART at the line rate, never more than WIRE_BUDGET_MAX ahead: a
        long PROFILE list cannot fill the UART buffer and hold a newer STATUS back on the
        wire (TxQueue only orders what it still holds; Codex Q9).'''
        if self._t_budget is not None:
            self._budget = min(float(WIRE_BUDGET_MAX), self._budget + (now - self._t_budget) * self.rate)
        self._t_budget = now
        # TxQueue's FIFO holds profiles. Keep one replaceable EXTRA outside it so a stalled
        # UART cannot accumulate stale countdowns or evict a profile. Finish each frame on the wire.
        if self._tail:
            self._write_frame(self._tail)
            if self._tail:
                return
        self._budget -= self.tx.pump(max_bytes=int(self._budget))
        if self.tx.pending():
            return
        if self._pending_extra:
            if self._write_frame(self._pending_extra):
                self.t_extra, self.last_extra = now, self._pending_extra
                self._pending_extra = b''
        elif self._tracks and self._track_allowance and self._budget >= pp.LEN_TRACK:
            # The unspent byte budget bounds how much space is free in the UART.
            # Include bytes already handed over when reserving time for the next STATUS.
            wire_room = (self.t_status + STATUS_PERIOD_S - now) * self.rate
            backlog = WIRE_BUDGET_MAX - self._budget
            if wire_room >= pp.LEN_TRACK + backlog and self._write_frame(pp.encode_track(self._tracks[0])):
                self._track_allowance -= 1
                self._tracks.popleft()


# ── FAKE run for the bench and --demo (nothing here is a measurement) ─────

PREPARE_S, ARM_S, STOPPING_S = 0.5, 0.5, 0.3


class FakeRun:
    """A made-up run manager: PREPARE 0.5 s, ARM 0.5 s, RUN run_s, END; optional fault in
    RUN. Every result line starts with 'FAKE'."""

    def __init__(self, names=('FAKE-1', 'FAKE-2', 'FAKE-3'), run_s=10.0, ctrl='PID', fault_at=None,
                 h7=True, clock: Callable[[], float] = time.monotonic, seed=0):
        self.names, self.run_s, self.ctrl, self.fault_at, self.h7 = list(names), run_s, ctrl, fault_at, h7
        self.clock = clock
        self.rnd = random.Random(seed)
        self.state, self.t_phase = pp.ST_IDLE, clock()
        self.profile, self.outcome = None, pp.OUT_NONE
        self.error_code, self.error_text, self.result = 0, '', ()
        self.run_no, self.result_run = 0, 0

    def _phase(self, st):
        self.state, self.t_phase = st, self.clock()

    def profiles(self):
        return self.names

    def select(self, i):
        if self.state in pp.ST_ACTIVE:
            return pp.RES_BUSY
        if i >= len(self.names):
            return pp.RES_BAD_PROFILE
        self.profile = i
        return pp.RES_OK

    def start(self, i):
        if self.state in pp.ST_ACTIVE:
            return pp.RES_BUSY
        if i >= len(self.names):
            return pp.RES_BAD_PROFILE
        try:
            run_no = pp.next_run_no(self.run_no)
        except OverflowError:                       # no number left: refuse, change nothing
            return pp.RES_FAILED
        self.profile = i
        self.error_code, self.error_text, self.result, self.result_run = 0, '', (), 0
        self.run_no = run_no
        self._phase(pp.ST_PREPARE)
        return pp.RES_OK

    def stop(self):
        if self.state in (pp.ST_PREPARE, pp.ST_ARM, pp.ST_RUN):
            self.outcome = pp.OUT_USER_STOP
            self._phase(pp.ST_STOPPING)
            return pp.RES_OK
        if self.state == pp.ST_STOPPING:
            return pp.RES_OK
        if self.state == pp.ST_ERROR:                 # STOP acknowledges the fault
            self._phase(pp.ST_IDLE)
            return pp.RES_OK
        return pp.RES_NOT_IDLE

    def _advance(self):
        dt = self.clock() - self.t_phase
        if self.state == pp.ST_PREPARE and dt >= PREPARE_S:
            self._phase(pp.ST_ARM)
        elif self.state == pp.ST_ARM and dt >= ARM_S:
            self._phase(pp.ST_RUN)
        elif self.state == pp.ST_RUN:
            if self.fault_at is not None and dt >= self.fault_at:
                self.outcome, self.error_code, self.error_text = pp.OUT_ERROR, 1, 'FAKE fault (bench)'
                self._phase(pp.ST_ERROR)
            elif dt >= self.run_s:
                self.outcome = pp.OUT_COMPLETED
                self.result = ('FAKE e_vx 0.0%02d m/s' % self.rnd.randrange(100),
                               'FAKE e_wz 0.%02d rad/s' % self.rnd.randrange(100), 'FAKE lost 0')
                self.result_run = self.run_no
                self._phase(pp.ST_END)
        elif self.state == pp.ST_STOPPING and dt >= STOPPING_S:
            self._phase(pp.ST_END)

    def snapshot(self):
        self._advance()
        return RunSnapshot(run_state=self.state, phase_s=self.clock() - self.t_phase, profile=self.profile,
                           outcome=self.outcome, data_age_ms=20, h7_ok=self.h7, error_code=self.error_code,
                           error_text=self.error_text, ctrl=self.ctrl, ctrl_src=pp.CTRL_SRC_PROFILE,
                           loop_us=25 + self.rnd.randrange(5), overruns=0, result=self.result,
                           run_no=self.run_no, result_run=self.result_run, dut_sha='facefeed0000', dut_env=self.ctrl)


def log_foreign(parser, seen: int, on_log=print) -> int:
    """One operator line when CRC-valid frames of another protocol version arrive (Sol PDS-13):
    the panel firmware and the Pi speak different versions, the panel needs a reflash."""
    n = parser.version_mismatch
    if n != seen:
        on_log(f'panel sends protocol v{parser.last_foreign_version}, Pi speaks v{pp.PROTO_VERSION}: '
               f'reflash the panel ({n} frames ignored)')
    return n


def serve(port, link: PanelLink, stop_after_s: Optional[float] = None, on_log=print) -> None:
    """Loop: read what the UART has, feed, poll, print new log lines. port: pyserial object
    opened with timeout=0 (read never waits) and write_timeout=0 (write never waits)."""
    t_end = None if stop_after_s is None else time.monotonic() + stop_after_s
    names = {v: k[4:] for k, v in vars(pp).items() if k.startswith('CMD_')}
    n_log = n_foreign = 0
    while t_end is None or time.monotonic() < t_end:
        data = port.read(256)
        if data:
            link.feed(data)
        link.poll()
        for t, cmd, seq, r in link.log[n_log:]:
            on_log(f'{t:10.3f}  {names.get(cmd, cmd):8s} seq {seq:5d} -> {r}')
        n_log = len(link.log)
        n_foreign = log_foreign(link.parser, n_foreign, on_log)
        time.sleep(0.005)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='RPi5 side of the panel UART (protocol v7)')
    ap.add_argument('--port', default='/dev/ttyAMA0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--epoch-file', default=os.path.expanduser('~/.hil_panel_epoch'))
    ap.add_argument('--create-epoch', action='store_true', help='first start only: create the epoch file')
    ap.add_argument('--demo', action='store_true', help='FAKE run (no run manager yet)')
    a = ap.parse_args(argv)
    if not a.demo:
        ap.error('no run manager yet (stage E): use --demo')
    import serial  # pyserial
    owner = pp._FileLock(a.epoch_file + '.owner', timeout_s=0.5)
    with owner:                                   # one link per UART / epoch file
        epoch = pp.next_daemon_epoch(a.epoch_file, create=a.create_epoch)
        port = serial.Serial(baudrate=a.baud, timeout=0, write_timeout=0)
        port.dtr = port.rts = False
        port.port = a.port
        port.open()
        with port:
            link = PanelLink(epoch, FakeRun(), nonblocking_writer(port), baud=a.baud)
            print(f'panel link on {a.port}, epoch {epoch}, FAKE run; Ctrl+C to quit')
            serve(port, link)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
