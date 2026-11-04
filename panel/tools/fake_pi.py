"""Fake Pi for the panel bench: the real RPi5 side of the panel protocol (rpi5/panel_link.py,
PanelLink) driving a FAKE run (panel_link.FakeRun), over a serial port of the PC, so the panel
screens and buttons can be tried without the HIL box. Everything it shows is FAKE (result
lines start with 'FAKE'); nothing here is a measurement and nothing goes to 05_DATA.

  python panel/tools/fake_pi.py --port COM7                  # panel env panel_s3_bench
  python panel/tools/fake_pi.py --port COM7 --fault-at 4     # FAULT 4 s into each run
  python panel/tools/fake_pi.py --port COM7 --no-h7          # H7 link shown down

Only the run is fake: the link code (sessions, seqs, frames, STATUS/INFO/PROFILE timing) is
the one the Pi runs. The epoch is random per start (bench only; the Pi keeps an epoch file).
"""
from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
from panel_link import FakeRun, PanelLink, nonblocking_writer, serve   # noqa: E402


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
    port = serial.Serial(baudrate=a.baud, timeout=0, write_timeout=0)   # read and write never wait (P15)
    port.dtr = port.rts = False      # the "UART" port has DTR/RTS auto-reset: do not touch EN/IO0
    port.port = a.port
    port.open()
    epoch = random.randrange(1, 0xFFFE)
    run = FakeRun([p for p in a.profiles.split(',') if p], a.run_s, a.ctrl, a.fault_at, not a.no_h7)
    with port:
        print(f'fake Pi on {a.port}, epoch {epoch}; Ctrl+C to quit')
        serve(port, PanelLink(epoch, run, nonblocking_writer(port), baud=a.baud))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
