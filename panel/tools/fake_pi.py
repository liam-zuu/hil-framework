"""Fake Pi for the panel bench: plays the RPi5 daemon side of the panel protocol (v6) over
a serial port, with a made-up run, so the panel screens and buttons can be tried without the
HIL box. Everything it shows is FAKE (result lines start with 'FAKE'); nothing here is a
measurement and nothing goes to 05_DATA.

  python panel/tools/fake_pi.py --port COM7                  # panel env panel_s3_bench
  python panel/tools/fake_pi.py --port COM7 --fault-at 4     # FAULT 4 s into each run
  python panel/tools/fake_pi.py --port COM7 --no-h7          # H7 link shown down

It uses rpi5/panel_proto.py (DaemonSession, StreamParser, TxQueue) as the real daemon will;
the run model (phases, times, texts) is only for the bench.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import panel_proto as pp   # noqa: E402

PREPARE_S, ARM_S, STOPPING_S = 0.5, 0.5, 0.3


class FakePi:
    """Daemon side with a fake run. feed(bytes, now) takes what the panel sent; poll(now)
    returns the bytes to write now (STATUS every 100 ms, INFO every 1 s and on change)."""

    def __init__(self, epoch: int, profiles=('FAKE-1', 'FAKE-2', 'FAKE-3'), run_s=10.0, ctrl='PID',
                 fault_at=None, h7=True, seed=0):
        self.sess = pp.DaemonSession(epoch)
        self.parser = pp.StreamParser(accept=(pp.START_CMD,), clock=lambda: self._now)
        self._now = 0.0
        self.out = bytearray()
        self.tx = pp.TxQueue(self._write)
        self.profiles = list(profiles)
        self.run_s, self.ctrl, self.fault_at, self.h7 = run_s, ctrl, fault_at, h7
        self.rnd = random.Random(seed)
        self.state, self.t_phase = pp.ST_IDLE, 0.0
        self.profile, self.outcome = pp.PROFILE_NONE, pp.OUT_NONE
        self.error_code, self.error_text = 0, ''
        self.result = ('', '', '')
        self.t_status = self.t_info = -1e9
        self.log = []

    def _write(self, data: bytes) -> int:
        self.out += data
        return len(data)

    # ── commands from the panel ──
    def run(self, c: pp.Cmd) -> int:
        if c.cmd == pp.CMD_SELECT:
            if self.state in pp.ST_ACTIVE:
                return pp.RES_BUSY
            if c.arg >= len(self.profiles):
                return pp.RES_BAD_PROFILE
            self.profile = c.arg
            return pp.RES_OK
        if c.cmd == pp.CMD_START:
            if self.state in pp.ST_ACTIVE:
                return pp.RES_BUSY
            if c.arg >= len(self.profiles):
                return pp.RES_BAD_PROFILE
            self.profile = c.arg
            self._phase(pp.ST_PREPARE)
            self.error_code, self.error_text, self.result = 0, '', ('', '', '')
            return pp.RES_OK
        if c.cmd == pp.CMD_STOP:
            if self.state in (pp.ST_PREPARE, pp.ST_ARM, pp.ST_RUN):
                self.outcome = pp.OUT_USER_STOP
                self._phase(pp.ST_STOPPING)
                return pp.RES_OK
            if self.state == pp.ST_STOPPING:
                return pp.RES_OK
            if self.state == pp.ST_ERROR:          # STOP acknowledges the fault
                self._phase(pp.ST_IDLE)
                return pp.RES_OK
            return pp.RES_NOT_IDLE
        return pp.RES_UNKNOWN

    def _phase(self, st: int) -> None:
        self.state, self.t_phase = st, self._now
        self.t_info = -1e9                          # send INFO now

    def feed(self, data: bytes, now: float) -> None:
        self._now = now
        for c in self.parser.feed(data):
            r = self.sess.on_cmd(c, self.run)
            self.log.append((round(now, 3), c.cmd, c.seq, r))
            self.tx.put(self._status())
            if r in ('hello', 'profiles'):
                if r == 'hello':
                    self.tx.clear_profiles()
                    self.t_info = -1e9
                for i, name in enumerate(self.profiles or [None]):
                    self.tx.put(pp.encode_profile(self.sess.sid, i, len(self.profiles), name or ''))

    # ── time ──
    def _advance(self) -> None:
        dt = self._now - self.t_phase
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
                               'FAKE e_wz 0.%02d rad/s' % self.rnd.randrange(100),
                               'FAKE lost 0')
                self._phase(pp.ST_END)
        elif self.state == pp.ST_STOPPING and dt >= STOPPING_S:
            self._phase(pp.ST_END)

    def _status(self) -> bytes:
        links = pp.L_RPI | (pp.L_H7 if self.h7 else 0)
        st = pp.Status(run_state=self.state, phase_ds=int((self._now - self.t_phase) * 10),
                       profile=self.profile, outcome=self.outcome, quality=0, lost=0,
                       data_age_ms=20, links=links, error_code=self.error_code)
        return pp.encode_status(self.sess.fill(st))

    def _info(self) -> bytes:
        flags = pp.I_LOOP_VALID | (pp.I_RESULT_VALID if any(self.result) else 0)
        return pp.encode_info(pp.Info(sid=self.sess.sid, loop_us=25 + self.rnd.randrange(5), overruns=0,
                                      flags=flags, ctrl_src=pp.CTRL_SRC_PROFILE, ctrl=self.ctrl,
                                      result=self.result, error_text=self.error_text))

    def poll(self, now: float) -> bytes:
        self._now = now
        self._advance()
        if now - self.t_status >= 0.1:
            self.t_status = now
            self.tx.put(self._status())
        if self.sess.sid and now - self.t_info >= 1.0:
            self.t_info = now
            self.tx.put(self._info())
        self.tx.pump()
        data, self.out = bytes(self.out), bytearray()
        return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--port', required=True, help='serial port of the panel (e.g. COM7)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--profiles', default='FAKE-1,FAKE-2,FAKE-3', help='comma separated names (<= 18 chars)')
    ap.add_argument('--run-s', type=float, default=10.0)
    ap.add_argument('--ctrl', default='PID')
    ap.add_argument('--fault-at', type=float, default=None, help='seconds into RUN')
    ap.add_argument('--no-h7', action='store_true')
    a = ap.parse_args(argv)
    import serial  # pyserial
    epoch = random.randrange(1, 0xFFFE)           # bench only: the real daemon keeps an epoch file
    pi = FakePi(epoch, [p for p in a.profiles.split(',') if p], a.run_s, a.ctrl, a.fault_at, not a.no_h7)
    names = {v: k[4:] for k, v in vars(pp).items() if k.startswith('CMD_')}
    port = serial.Serial(baudrate=a.baud, timeout=0)
    port.dtr = port.rts = False      # the "UART" port has DTR/RTS auto-reset: do not touch EN/IO0
    port.port = a.port
    port.open()
    with port:
        print(f'fake Pi on {a.port}, epoch {epoch}; Ctrl+C to quit')
        t0, n_log = time.monotonic(), 0
        while True:
            now = time.monotonic() - t0
            data = port.read(256)
            if data:
                pi.feed(data, now)
            out = pi.poll(now)
            if out:
                port.write(out)
            for entry in pi.log[n_log:]:
                print(f'{entry[0]:9.3f}s  {names.get(entry[1], entry[1]):8s} seq {entry[2]:5d} -> {entry[3]}')
            n_log = len(pi.log)
            time.sleep(0.005)


if __name__ == '__main__':
    raise SystemExit(main())
