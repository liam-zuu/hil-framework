"""
Panel link — the RPi5 side of the panel UART (protocol v6), in its own process, never inside
the 1 kHz loop (QĐ 17; workflow 9.6). Liam 04/11, PANEL session: "ổn, làm theo đề xuất về phía
pi": the PANEL session writes this link with a thin run-control interface; the CODE session
plugs the real run manager (hil_daemon / hil_ctl, stage E) into that interface.

    panel (ESP32-S3) <--UART ttyAMA0--> PanelLink <--RunControl--> run manager (stage E)

What PanelLink does: frames in/out (StreamParser, TxQueue), the session and seq rules
(DaemonSession), STATUS every STATUS_PERIOD_S and after every command, INFO every
INFO_PERIOD_S and when it changes, the PROFILE list after a new session and on PROFILES.
What it does not do: decide anything about a run. Every command goes to RunControl.

RunControl contract (what stage E implements):
  * profiles() -> list of test names (ASCII, <= 18 chars shown); index = position. Called on
    every new panel session and on PROFILES; the list must not change while a run is active.
  * select(index) / start(index) / stop() -> RES_* (panel_proto). They must only REQUEST the
    action and return at once (never wait for a run to end): the link calls them from its own
    loop, and a STOP cannot be handled while one of them blocks (panel_proto DaemonSession).
    Raising or returning RES_FAILED both mean "effect unknown". stop() with nothing to stop
    returns RES_NOT_IDLE; stop() in ST_ERROR acknowledges the fault (the run manager then goes
    to ST_IDLE) [ĐỀ XUẤT: rule of this link and of the fake Pi, not decided for stage E].
  * snapshot() -> RunSnapshot, cheap and non-blocking (read a copy the run manager keeps).
    Numbers it does not know stay None and are shown as '-' / not valid on the panel.

Run it on the Pi (only when stage E provides a RunControl; --demo uses the FAKE run):
    python panel_link.py --port /dev/ttyAMA0 --epoch-file ~/.hil_panel_epoch [--create-epoch] --demo
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import panel_proto as pp

STATUS_PERIOD_S = 0.1       # workflow 9.6: STATUS ~10 Hz
INFO_PERIOD_S = 1.0         # v6: INFO ~1 Hz and on change
INFO_MIN_GAP_S = 0.2        # a changing INFO (loop_us moves all the time) at most 5 per second


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
                   links=links, error_code=snap.error_code)
    return sess.fill(st)


def info_of(snap: RunSnapshot, sid: int) -> pp.Info:
    flags = (pp.I_LOOP_VALID if snap.loop_us is not None else 0) | \
            (pp.I_RESULT_VALID if any(snap.result) else 0)
    return pp.Info(sid=sid, loop_us=snap.loop_us or 0, overruns=snap.overruns or 0, flags=flags,
                   ctrl_src=snap.ctrl_src, ctrl=snap.ctrl, result=tuple(snap.result)[:pp.RESULT_LINES],
                   error_text=snap.error_text)


class PanelLink:
    """write(bytes) -> bytes taken now (never blocks); clock() -> seconds. Call feed() with
    what the UART read and poll() every few ms."""

    def __init__(self, epoch: int, run: RunControl, write: Callable[[bytes], int],
                 clock: Callable[[], float] = time.monotonic):
        self.sess = pp.DaemonSession(epoch)
        self.run = run
        self.clock = clock
        self.parser = pp.StreamParser(accept=(pp.START_CMD,), clock=clock)
        self.tx = pp.TxQueue(write, max_profiles=pp.PROFILE_MAX + 1)   # a whole list fits (P14)
        self.t_status = self.t_info = None
        self.last_info: Optional[bytes] = None
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
                self.last_info = None                # INFO of the new session at once
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
        if self.sess.sid:
            snap = snap or self.run.snapshot()
            info = pp.encode_info(info_of(snap, self.sess.sid))
            age = None if self.t_info is None else now - self.t_info
            if self.last_info is None or age >= INFO_PERIOD_S or (info != self.last_info and age >= INFO_MIN_GAP_S):
                self.t_info, self.last_info = now, info
                self.tx.put(info)
        self.tx.pump()


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
        self.profile = i
        self.error_code, self.error_text, self.result = 0, '', ()
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
                self._phase(pp.ST_END)
        elif self.state == pp.ST_STOPPING and dt >= STOPPING_S:
            self._phase(pp.ST_END)

    def snapshot(self):
        self._advance()
        return RunSnapshot(run_state=self.state, phase_s=self.clock() - self.t_phase, profile=self.profile,
                           outcome=self.outcome, data_age_ms=20, h7_ok=self.h7, error_code=self.error_code,
                           error_text=self.error_text, ctrl=self.ctrl, ctrl_src=pp.CTRL_SRC_PROFILE,
                           loop_us=25 + self.rnd.randrange(5), overruns=0, result=self.result)


def serve(port, link: PanelLink, stop_after_s: Optional[float] = None, on_log=print) -> None:
    """Loop: read what the UART has, feed, poll, print new log lines. port: pyserial object
    opened with timeout=0 (read never waits) and write_timeout=0 (write never waits)."""
    t_end = None if stop_after_s is None else time.monotonic() + stop_after_s
    names = {v: k[4:] for k, v in vars(pp).items() if k.startswith('CMD_')}
    n_log = 0
    while t_end is None or time.monotonic() < t_end:
        data = port.read(256)
        if data:
            link.feed(data)
        link.poll()
        for t, cmd, seq, r in link.log[n_log:]:
            on_log(f'{t:10.3f}  {names.get(cmd, cmd):8s} seq {seq:5d} -> {r}')
        n_log = len(link.log)
        time.sleep(0.005)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='RPi5 side of the panel UART (protocol v6)')
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
            link = PanelLink(epoch, FakeRun(), port.write)
            print(f'panel link on {a.port}, epoch {epoch}, FAKE run; Ctrl+C to quit')
            serve(port, link)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
