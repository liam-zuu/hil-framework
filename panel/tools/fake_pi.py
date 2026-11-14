"""Fake Pi for the panel bench: the real RPi5 side of the panel protocol (rpi5/panel_link.py,
PanelLink) driving a FAKE run (panel_link.FakeRun), over a serial port of the PC, so the panel
screens and buttons can be tried without the HIL box. Everything it shows is FAKE (result
lines start with 'FAKE'); nothing here is a measurement and nothing goes to 05_DATA.

  python panel/tools/fake_pi.py --port COM7                  # panel env panel_s3_bench
  python panel/tools/fake_pi.py --port COM7 --fault-at 4     # FAULT 4 s into each run
  python panel/tools/fake_pi.py --port COM7 --no-h7          # H7 link shown down
  python panel/tools/fake_pi.py --port COM7 --demo           # walks through every screen alone

--demo (Liam 05/11, PANEL session: "tôi muốn hiện hết các màn ra"): the fake Pi starts and
stops runs itself, as runs started from the PC would, so every screen shows without buttons:
HOME -> RUN -> RESULT COMPLETED -> RUN -> STOPPING -> RESULT USER STOP -> RUN -> FAULT ->
HOME -> NO LINK (the Pi stops sending for 3 s) -> again. The buttons still work.

Only the run is fake: the link code (sessions, seqs, frames, STATUS/INFO/PROFILE timing) is
the one the Pi runs. The epoch is random per start (bench only; the Pi keeps an epoch file).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import panel_proto as pp                                                # noqa: E402
from panel_link import ARM_S, PREPARE_S, FakeRun, PanelLink, nonblocking_writer, serve   # noqa: E402

DEMO_HOLD_S = 5.0          # each screen stays this long before the next step
DEMO_STOP_AFTER_S = 3.0    # user-stop run: STOP this long into RUN
DEMO_FAULT_AT_S = 2.0      # fault run: FAULT this long into RUN
DEMO_NOLINK_S = 3.0        # NO LINK: the Pi sends nothing this long


class DemoRun(FakeRun):
    """FakeRun that drives itself through every panel screen (see --demo above)."""

    def __init__(self, *args, on_log=print, **kw):
        super().__init__(*args, **kw)
        self.on_log = on_log
        self.pause_until = 0.0
        self._go(0, 'HOME (%.0f s)' % DEMO_HOLD_S)

    def _go(self, step, what):
        self.step, self.t_step = step, self.clock()
        self.on_log('demo: ' + what)

    def _in_run(self, t: float) -> float:
        """a moment inside RUN, before the run would end by itself (--run-s may be short)"""
        return min(t, self.run_s / 2)

    def paused(self) -> bool:
        """True while the demo plays NO LINK: the caller must not poll the link then."""
        return self.clock() < self.pause_until

    def _script(self):
        dt, st = self.clock() - self.t_step, self.state
        n = len(self.names)
        if self.step == 0 and dt >= DEMO_HOLD_S:
            self.fault_at = None
            self.select(1 % n)
            self.start(1 % n)
            self._go(1, 'run %s -> PREPARE, ARM, RUN, then RESULT COMPLETED' % self.names[1 % n])
        elif self.step == 1 and st == pp.ST_END:
            self._go(2, 'RESULT COMPLETED (%.0f s)' % DEMO_HOLD_S)
        elif self.step == 2 and dt >= DEMO_HOLD_S:
            self.fault_at = None
            self.start(2 % n)
            self._go(3, 'run %s, STOP after %.1f s in RUN' % (self.names[2 % n], self._in_run(DEMO_STOP_AFTER_S)))
        elif self.step == 3 and st == pp.ST_RUN and dt >= PREPARE_S + ARM_S + self._in_run(DEMO_STOP_AFTER_S):
            self.stop()
            self._go(4, 'STOPPING, then RESULT USER STOP')
        elif self.step == 4 and st == pp.ST_END:
            self._go(5, 'RESULT USER STOP (%.0f s)' % DEMO_HOLD_S)
        elif self.step == 5 and dt >= DEMO_HOLD_S:
            self.fault_at = self._in_run(DEMO_FAULT_AT_S)
            self.start(0)
            self._go(6, 'run %s, FAULT after %.1f s in RUN' % (self.names[0], self.fault_at))
        elif self.step == 6 and st == pp.ST_ERROR:
            self._go(7, 'FAULT (%.0f s), then acknowledged' % DEMO_HOLD_S)
        elif self.step == 7 and dt >= DEMO_HOLD_S:
            self.stop()                                   # acknowledges the fault -> IDLE
            self.fault_at = None
            self._go(8, 'HOME (%.0f s)' % DEMO_HOLD_S)
        elif self.step == 8 and dt >= DEMO_HOLD_S:
            self.pause_until = self.clock() + DEMO_NOLINK_S
            self._go(9, 'NO LINK (the fake Pi sends nothing for %.0f s)' % DEMO_NOLINK_S)
        elif self.step == 9 and not self.paused() and dt >= DEMO_NOLINK_S + 2.0:
            self._go(0, 'HOME (%.0f s), again from the start' % DEMO_HOLD_S)

    def snapshot(self):
        self._script()
        return super().snapshot()


def serve_demo(port, link: PanelLink, run: DemoRun, on_log=print) -> None:
    """serve() with one change: while run.paused() the link is not polled, so no STATUS goes
    out and the panel shows NO LINK. Bytes from the panel are still read and fed."""
    names = {v: k[4:] for k, v in vars(pp).items() if k.startswith('CMD_')}
    n_log = 0
    while True:
        data = port.read(256)
        if data:
            link.feed(data)
        if run.paused():
            run.snapshot()                                # keeps the script clock moving
        else:
            link.poll()
        for t, cmd, seq, r in link.log[n_log:]:
            on_log(f'{t:10.3f}  {names.get(cmd, cmd):8s} seq {seq:5d} -> {r}')
        n_log = len(link.log)
        time.sleep(0.005)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--port', required=True, help='serial port of the panel (e.g. COM7)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--profiles', default='FAKE-1,FAKE-2,FAKE-3', help='comma separated names (<= 18 chars)')
    ap.add_argument('--run-s', type=float, default=10.0)
    ap.add_argument('--ctrl', default='PID')
    ap.add_argument('--fault-at', type=float, default=None, help='seconds into RUN')
    ap.add_argument('--no-h7', action='store_true')
    ap.add_argument('--demo', action='store_true', help='walk through every screen without buttons')
    a = ap.parse_args(argv)
    import serial  # pyserial
    port = serial.Serial(baudrate=a.baud, timeout=0, write_timeout=0)   # read and write never wait (P15)
    port.dtr = port.rts = False      # the "UART" port has DTR/RTS auto-reset: do not touch EN/IO0
    port.port = a.port
    port.open()
    epoch = random.randrange(1, 0xFFFE)
    names = [p for p in a.profiles.split(',') if p]
    with port:
        print(f'fake Pi on {a.port}, epoch {epoch}; Ctrl+C to quit')
        if a.demo:
            run = DemoRun(names, a.run_s, a.ctrl, None, not a.no_h7)
            serve_demo(port, PanelLink(epoch, run, nonblocking_writer(port), baud=a.baud), run)
        else:
            run = FakeRun(names, a.run_s, a.ctrl, a.fault_at, not a.no_h7)
            serve(port, PanelLink(epoch, run, nonblocking_writer(port), baud=a.baud))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
