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
lib.app_result_fresh.restype, lib.app_result_fresh.argtypes = ct.c_bool, [ct.c_void_p]
lib.app_result.restype, lib.app_result.argtypes = ct.c_int, [ct.c_void_p]


class AppView(ct.Structure):
    _fields_ = [('screen', ct.c_int), ('cursor', ct.c_uint8), ('sid', ct.c_uint32),
                ('profiles_gen', ct.c_uint32), ('result_profile', ct.c_uint8)]


lib.app_view.restype, lib.app_view.argtypes = AppView, [ct.c_void_p, ct.c_uint32]
lib.app_drawn.restype, lib.app_drawn.argtypes = None, [ct.c_void_p, ct.POINTER(AppView)]
lib.app_drawing.restype, lib.app_drawing.argtypes = None, [ct.c_void_p]
NOT_VERIFIED, WAITING, FRESH = range(3)

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

    def view(self):
        return lib.app_view(self.app, self.t)               # what a drawing started now shows

    def draw(self, view=None):
        v = self.view() if view is None else view
        lib.app_drawn(self.app, ct.byref(v))

    def press(self, ev):
        """returns the frames the press made, then sends them up"""
        self.panel._out = []
        lib.app_buttons(self.app, ev, self.t)
        lib.app_update(self.app, self.t)
        out = list(self.panel._out)
        self._send_up()
        return out

    def inject(self, frame):
        lib.pp_client_on_frame(self.panel._c, frame, len(frame), self.t)

    @property
    def result(self):
        return lib.app_result(self.app)

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
    assert b.screen == RESULT and b.result == NOT_VERIFIED
    b.drop = lambda f: False
    b.step(120)                                         # valid INFO of B, but no cleared one seen
    assert b.result == NOT_VERIFIED and b.panel.info.result[0].startswith('FAKE')


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
    assert b.screen == NOLINK and b.press(ev) == []         # frames of the press itself (Q7)
    b.step(150)
    b.press(START)
    b.step(30)
    assert b.screen == RUN
    n = len(b.link.log)
    assert b.press(ev) == []
    b.step(5)
    assert not [e for e in b.link.log[n:] if e[1] in (pp.CMD_START, pp.CMD_SELECT, pp.CMD_STOP)]


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


def test_a_normal_run_shows_its_result_codex_q1():
    b = ready()
    b.press(START)
    b.step(250)
    assert b.screen == RESULT and b.result == FRESH


def test_old_result_arriving_during_a_new_run_is_not_shown_codex_q1():
    b = ready()
    b.press(START)
    b.step(250)
    b.press(NAV)
    b.step(1)
    b.drop = lambda f: f[0] == pp.START_INFO            # the link's own INFO of run B lost
    b.press(START)
    b.step(20)
    old = pp.Info(sid=b.panel.sid, flags=pp.I_RESULT_VALID, result=('OLD A', '', ''))
    b.inject(pp.encode_info(old))                       # a valid INFO of run A arrives during B
    b.step(230)
    assert b.screen == RESULT and b.result == NOT_VERIFIED


def test_start_replaced_by_stop_and_a_short_run_still_shows_result_codex_q2():
    b = ready()
    b.press(START)
    b.step(250)
    b.press(NAV)                                        # close RESULT A
    b.step(1)
    b.drop = lambda f: f[0] == pp.START_STATUS and b.run.state in pp.ST_ACTIVE
    b.press(START)
    b.press(STOP)                                       # STOP before the START ACK
    b.step(40)                                          # run B: PREPARE -> STOPPING -> END in 0.3 s
    b.drop = lambda f: False
    b.step(20)
    assert b.run.state == pp.ST_END and b.screen == RESULT


def test_link_gap_does_not_reopen_a_closed_result_codex_q2():
    b = ready()
    b.press(START)
    b.step(250)
    b.press(NAV)
    b.step(1)
    assert b.screen == HOME
    b.drop = lambda f: f[0] == pp.START_STATUS          # 0.8 s without STATUS: NO LINK
    b.step(80)
    assert b.screen == NOLINK
    b.drop = lambda f: False
    b.step(20)
    assert b.screen == HOME


def test_start_answered_busy_gives_the_closed_state_back_codex_q2():
    b = ready()
    b.press(START)
    b.step(250)
    b.press(NAV)
    b.step(1)
    b.run.start = lambda i: pp.RES_BUSY
    b.press(START)
    b.step(30)
    assert b.screen == HOME and 'BUSY' in b.msg


def test_view_drawn_before_a_list_change_does_not_start_the_new_test_codex_q3():
    b = ready()
    old = b.view()                                      # snapshot taken for a drawing
    b.inject(pp.encode_profile(b.panel.sid, 0, 3, 'RENAMED'))
    lib.app_update(b.app, b.t)
    b.draw(old)                                         # the drawing finishes with the old list
    assert b.press(START) == [] and 'press again' in b.msg
    b.draw()
    assert [f[8] for f in b.press(START)] == [pp.CMD_START]


def test_only_stop_during_a_redraw_that_changes_the_view_codex_q3():
    b = ready()
    lib.app_drawing(b.app)
    assert b.press(START) == [] and b.press(SELECT) == []
    assert [f[8] for f in b.press(STOP)] == [pp.CMD_STOP]


def test_run_again_uses_the_test_shown_codex_q3():
    b = ready()
    b.press(NAV)
    b.step(1)
    b.press(START)                                      # test 1
    b.step(250)
    assert b.screen == RESULT
    out = b.press(START)
    assert [(f[8], pp.decode(f).arg) for f in out] == [(pp.CMD_START, 1)]


def test_list_shrinks_then_grows_old_names_not_reused_codex_q4():
    b = ready()
    sid = b.panel.sid
    b.inject(pp.encode_profile(sid, 0, 1, 'X'))
    b.inject(pp.encode_profile(sid, 0, 3, 'Y'))         # new list of 3, names 1 and 2 not here yet
    lib.app_update(b.app, b.t)
    b.draw()
    for _ in range(2):
        b.press(NAV)
        b.draw()
    assert b.cursor == 2 and b.press(START) == [] and 'not complete' in b.msg


def test_rejection_message_not_overwritten_by_an_ack_in_the_same_pass_codex_q8():
    b = ready()
    b.press(SELECT)                                     # SELECT in flight, reaches the link
    b.t += 10
    b.link.poll()                                       # the ACK (STATUS) is on the wire
    b.panel._out = []
    data, b.wire = bytes(b.wire), bytearray()
    lib.pp_parser_feed(b.parser, data, len(data), b.t, b._cb, None)
    lib.pp_client_tick(b.panel._c, b.t)
    lib.app_update(b.app, b.t)                          # firmware order: update, buttons, update
    lib.app_buttons(b.app, NAV | START, b.t)
    lib.app_update(b.app, b.t)
    assert 'cursor moved' in b.msg


def test_two_runs_in_a_row_both_show_their_result():
    b = ready()
    for k in (0, 1):
        b.press(START)
        b.step(250)
        assert b.screen == RESULT and b.result == FRESH, k
        b.press(NAV)
        b.step(1)


def test_run_started_elsewhere_and_seen_active_shows_a_verified_result():
    b = ready()
    b.run.start(2)                                      # from the PC
    b.step(250)
    assert b.screen == RESULT and b.result == FRESH


def test_start_on_a_result_not_drawn_yet_is_not_used_codex_q3():
    b = ready()
    b.press(START)                                      # test 0
    b.step(250)
    assert b.screen == RESULT                           # glass: RESULT of test 0
    b.run.start(2)                                      # PC runs test 2; panel does not redraw
    b.step(250, draw=False)
    assert b.press(START) == [] and 'press again' in b.msg
