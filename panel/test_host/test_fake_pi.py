"""The Pi side of the panel link (rpi5/panel_link.py: PanelLink with the FAKE run that the bench
fake Pi uses) driven by the C panel client over a byte loop: handshake, profiles, INFO,
SELECT/START/STOP, a run ending, a fault acknowledged by STOP, a bad test index."""
import os
import sys

import pytest

from test_panel_c import CPanelClient        # noqa: E402  (also puts rpi5/ on sys.path)
import panel_proto as pp                     # noqa: E402
from panel_link import FakeRun, PanelLink    # noqa: E402


class Loop:
    def __init__(self, **kw):
        self.t = 0
        self.wire = bytearray()
        clock = lambda: self.t / 1000                                    # noqa: E731
        self.run = FakeRun(run_s=2.0, clock=clock, **kw)
        self.pi = PanelLink(3, self.run, self._write, clock=clock)
        self.panel = CPanelClient(iter(range(0x100, 0x200)).__next__)
        self.parser = pp.StreamParser(accept=(pp.START_STATUS, pp.START_PROFILE, pp.START_INFO),
                                      clock=lambda: self.t / 1000)
        self.up = []
        self.st = None

    def _write(self, data):
        self.wire += data
        return len(data)

    def step(self, n=1):
        for _ in range(n):
            self.t += 10
            self.pi.feed(b''.join(self.up))
            self.up = []
            self.pi.poll()
            data, self.wire = bytes(self.wire), bytearray()
            for x in self.parser.feed(data):
                if isinstance(x, pp.Status):
                    self.st = x
                    self.up += self.panel.on_status(x, self.t)
                elif isinstance(x, pp.Profile):
                    self.panel.on_profile(x)
                else:
                    self.panel.on_info(x)
            self.up += self.panel.tick(self.t)

    def press(self, cmd, arg=0):
        self.up += self.panel.press(cmd, arg, self.t)


def test_handshake_profiles_info_then_a_full_run():
    lp = Loop()
    lp.step(150)
    assert lp.panel.sid and lp.panel.profiles == {0: 'FAKE-1', 1: 'FAKE-2', 2: 'FAKE-3'}
    assert lp.panel.info.ctrl == 'PID' and lp.panel.info.flags & pp.I_LOOP_VALID
    lp.press(pp.CMD_SELECT, 1)
    lp.step(10)
    lp.press(pp.CMD_START, 1)
    lp.step(400)                                            # 0.5 + 0.5 + 2 s run
    assert lp.panel.shown == [('done', pp.CMD_SELECT, pp.RES_OK), ('done', pp.CMD_START, pp.RES_OK)]
    assert lp.st.run_state == pp.ST_END and lp.st.outcome == pp.OUT_COMPLETED and lp.st.profile == 1
    assert lp.panel.info.result[0].startswith('FAKE') and lp.panel.info.flags & pp.I_RESULT_VALID


def test_stop_during_run_and_fault_acknowledged_by_stop():
    lp = Loop(fault_at=1.0)
    lp.step(150)
    lp.press(pp.CMD_START, 0)
    lp.step(80)
    lp.press(pp.CMD_STOP)
    lp.step(100)
    assert lp.st.run_state == pp.ST_END and lp.st.outcome == pp.OUT_USER_STOP
    assert lp.panel.stop_state == 'confirmed'
    lp.press(pp.CMD_START, 2)
    lp.step(300)
    assert lp.st.run_state == pp.ST_ERROR and lp.st.error_code == 1
    assert lp.panel.info.error_text == 'FAKE fault (bench)'
    lp.press(pp.CMD_STOP)
    lp.step(50)
    assert lp.st.run_state == pp.ST_IDLE


@pytest.mark.parametrize('arg', [3, 200])
def test_bad_profile_refused(arg):
    lp = Loop()
    lp.step(150)
    lp.press(pp.CMD_START, arg)
    lp.step(20)
    assert lp.panel.shown == [('done', pp.CMD_START, pp.RES_BAD_PROFILE)]
