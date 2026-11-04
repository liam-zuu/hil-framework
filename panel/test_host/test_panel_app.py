"""panel_app.c (buttons -> commands, screen choice, result generation) on the PC, driven over a
raw byte wire: Pi side = rpi5/panel_link.py PanelLink with the FAKE run, panel side = C parser
-> C client -> C app. One scenario per fix of Codex round 1 (Liam approved 04/11): P1 STOP,
P2 drawn screen / cursor, P3 cursor and incomplete list, P7 result lines, P8 closed RESULT.
The 2-task timing of the firmware (P5) is not modelled here: it can only be checked on a board."""
import ctypes as ct

import pytest

from test_panel_c import FRAME_FN, CPanelClient, lib    # noqa: F401  (also puts rpi5/ on sys.path)
import panel_proto as pp                                # noqa: E402
from panel_link import FakeRun, PanelLink              # noqa: E402

lib.app_host_new.restype, lib.app_host_new.argtypes = ct.c_void_p, [ct.c_void_p]
lib.app_host_free.argtypes = [ct.c_void_p]
lib.app_host_cursor.restype, lib.app_host_cursor.argtypes = ct.c_int, [ct.c_void_p]
lib.app_host_msg.restype, lib.app_host_msg.argtypes = ct.c_char_p, [ct.c_void_p]
lib.app_host_gen.restype, lib.app_host_gen.argtypes = ct.c_uint32, [ct.c_void_p]
lib.app_update.restype, lib.app_update.argtypes = None, [ct.c_void_p, ct.c_uint32]
lib.app_screen.restype, lib.app_screen.argtypes = ct.c_int, [ct.c_void_p, ct.c_uint32]
lib.app_buttons.restype, lib.app_buttons.argtypes = None, [ct.c_void_p, ct.c_uint8, ct.c_uint32]
lib.app_drawn.restype, lib.app_drawn.argtypes = None, [ct.c_void_p, ct.c_int, ct.c_uint8]
lib.app_result_fresh.restype, lib.app_result_fresh.argtypes = ct.c_bool, [ct.c_void_p]

NAV, NAV_BACK, SELECT, START, STOP = 1, 2, 4, 8, 16
NOLINK, HOME, RUN, RESULT, FAULT = range(5)


class AppBench:
    def __init__(self, names=('T0', 'T1', 'T2'), run_s=1.0, epoch=4):
        self.t = 0
        self.clock = lambda: self.t / 1000
        self.wire = bytearray()
        self.run = FakeRun(names, run_s=run_s, clock=self.clock)
        self.link = PanelLink(epoch, self.run, self._w, clock=self.clock)
        self.panel = CPanelClient(iter(range(0x700, 0x800)).__next__)
        self.app = lib.app_host_new(self.panel._c)
        self.parser = lib.pp_host_parser_new(0b1110, 50)
        self.drop = lambda f: False
        self._cb = FRAME_FN(self._frame)

    def _w(self, data):
        self.wire += data
        return len(data)

    def _frame(self, _ctx, data, n):
        f = bytes(data[:n])
        if not self.drop(f):
            lib.pp_client_on_frame(self.panel._c, f, n, self.t)

    def _send_up(self):
        for f in self.panel._out:
            self.link.feed(f)
        self.panel._out = []

    def step(self, n=1, draw=True):
        for _ in range(n):
            self.t += 10
            self.panel._out = []
            data, self.wire = bytes(self.wire), bytearray()
            lib.pp_parser_feed(self.parser, data, len(data), self.t, self._cb, None)
            lib.pp_client_tick(self.panel._c, self.t)
            lib.app_update(self.app, self.t)
            if draw:
                self.draw()
            self._send_up()
            self.link.poll()

    def draw(self):
        lib.app_drawn(self.app, lib.app_screen(self.app, self.t), lib.app_host_cursor(self.app))

    def press(self, ev):
        self.panel._out = []
        lib.app_buttons(self.app, ev, self.t)
        lib.app_update(self.app, self.t)
        self._send_up()

    @property
    def screen(self):
        return lib.app_screen(self.app, self.t)

    @property
    def cursor(self):
        return lib.app_host_cursor(self.app)

    @property
    def msg(self):
        return lib.app_host_msg(self.app).decode()

    def cmds(self, cmd):
        return [e for e in self.link.log if e[1] == cmd]


def ready():
    b = AppBench()
    b.step(150)
    assert b.screen == HOME and b.panel.profile_count == 3
    return b


def test_stop_right_after_start_before_any_status_reaches_the_run_codex_p1():
    b = ready()
    b.press(START)
    b.press(STOP)                                       # still HOME: no STATUS since START
    b.step(300)
    assert b.cmds(pp.CMD_STOP) and b.run.outcome == pp.OUT_USER_STOP


def test_stop_wins_over_start_in_the_same_read_codex_p1():
    b = ready()
    b.press(START | STOP)
    b.step(100)
    assert b.cmds(pp.CMD_STOP) and not b.cmds(pp.CMD_START) and b.run.state == pp.ST_IDLE


def test_stop_on_result_is_sent_and_closes_the_screen_codex_p1():
    b = ready()
    b.press(START)
    b.step(250)
    assert b.screen == RESULT
    b.press(STOP)
    b.step(30)
    assert b.screen == HOME and b.cmds(pp.CMD_STOP)[-1][3] in ('ran', 'repeated')


def test_start_after_nav_not_yet_drawn_is_not_used_codex_p2():
    b = ready()
    b.press(NAV)                                        # cursor 1, glass still shows 0
    b.press(START)
    assert not b.cmds(pp.CMD_START) and 'press again' in b.msg
    b.step(1)                                           # drawn
    b.press(START)
    b.step(30)
    assert [e[1] for e in b.cmds(pp.CMD_START)] == [pp.CMD_START] and b.run.profile == 1


def test_nav_and_start_in_one_read_does_not_start_codex_p2():
    b = ready()
    b.press(NAV | START)
    b.step(30)
    assert not b.cmds(pp.CMD_START)


def test_start_against_a_screen_not_drawn_yet_is_ignored_codex_p2():
    b = ready()
    b.press(START)
    b.step(30)
    assert b.screen == RUN
    b.step(220, draw=False)                             # run ended: model RESULT, glass RUN
    assert b.screen == RESULT
    b.press(START)
    b.step(5, draw=False)
    assert len(b.cmds(pp.CMD_START)) == 1 and 'screen changing' in b.msg


def test_cursor_follows_a_shorter_list_codex_p3():
    b = ready()
    for _ in range(2):
        b.press(NAV)
        b.step(1)
    assert b.cursor == 2
    b.run = FakeRun(['X'], run_s=1.0, clock=b.clock)
    b.link = PanelLink(9, b.run, b._w, clock=b.clock)   # daemon restart with one test
    b.step(200)
    assert b.panel.profile_count == 1 and b.cursor == 0
    b.press(START)
    b.step(30)
    assert b.run.profile == 0


def test_no_start_for_a_test_whose_name_has_not_arrived_codex_p3():
    b = AppBench()
    b.drop = lambda f: f[0] == pp.START_PROFILE and f[6] == 1     # name of test 1 never arrives
    b.step(150)
    b.press(NAV)
    b.step(1)
    b.press(START)
    b.step(30)
    assert not b.cmds(pp.CMD_START) and 'not complete' in b.msg
    b.press(SELECT)
    assert not b.cmds(pp.CMD_SELECT)


def test_result_lines_of_the_previous_run_are_not_shown_codex_p7():
    b = ready()
    b.press(START)
    b.step(250)
    assert b.screen == RESULT and lib.app_result_fresh(b.app)
    b.press(NAV)                                        # close RESULT A
    b.step(1)
    b.drop = lambda f: f[0] == pp.START_INFO            # every INFO of run B lost
    b.press(START)
    b.step(250)
    assert b.screen == RESULT and not lib.app_result_fresh(b.app)
    b.drop = lambda f: False
    b.step(120)
    assert lib.app_result_fresh(b.app) and b.panel.info.result[0].startswith('FAKE')


def test_closed_result_does_not_hide_the_next_one_codex_p8():
    b = ready()
    b.press(START)
    b.step(250)
    b.press(NAV)
    b.step(1)
    assert b.screen == HOME
    b.drop = lambda f: f[0] == pp.START_STATUS and b.run.state in pp.ST_ACTIVE   # B never seen active
    b.press(START)
    b.step(300)
    b.drop = lambda f: False
    b.step(50)
    assert b.run.state == pp.ST_END and b.screen == RESULT


@pytest.mark.parametrize('ev', [NAV, NAV_BACK, SELECT, START])
def test_buttons_do_nothing_on_run_and_no_link_except_stop(ev):
    b = AppBench()
    b.step(1)
    b.press(ev)
    assert not b.panel._out
    b.step(150)
    b.press(START)
    b.step(30)
    n = len(b.link.log)
    b.press(ev)
    b.step(5)
    assert b.screen == RUN and all(e[1] in (pp.CMD_START, pp.CMD_SYNC, pp.CMD_PROFILES)
                                   for e in b.link.log[n:])


def test_cursor_clamped_when_the_list_shrinks_inside_a_session_codex_p3():
    b = ready()
    for _ in range(2):
        b.press(NAV)
        b.step(1)
    assert b.cursor == 2
    f = pp.encode_profile(b.panel.sid, 0, 1, 'X')       # same session, list of one
    lib.pp_client_on_frame(b.panel._c, f, len(f), b.t)
    lib.app_update(b.app, b.t)
    assert b.panel.profile_count == 1 and b.cursor == 0


def test_run_started_elsewhere_shows_its_result_after_a_closed_one_codex_p8():
    b = ready()
    b.press(START)
    b.step(250)
    b.press(NAV)                                        # close RESULT A
    b.step(1)
    b.run.start(2)                                      # run B started from the PC, not the panel
    b.step(250)
    assert b.run.state == pp.ST_END and b.screen == RESULT


def test_run_started_elsewhere_and_never_seen_active_still_shows_result_codex_p8():
    b = ready()
    b.press(START)
    b.step(250)
    b.press(NAV)
    b.step(1)
    b.drop = lambda f: f[0] == pp.START_STATUS and b.run.state in pp.ST_ACTIVE
    b.run.start(2)                                      # from the PC; every active STATUS lost
    b.step(250)
    b.drop = lambda f: False
    b.step(30)
    assert b.run.state == pp.ST_END and b.screen == RESULT
