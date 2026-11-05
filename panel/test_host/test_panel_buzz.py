"""lib/panel_buzz on the PC (Liam 05/11, PANEL session: "Làm còi trước"; buzzer passive,
low-level trigger): which STATUS changes beep, which do not, the tone timeline of each
pattern, every note as a fraction of f0, the start-up tune, and the buzzer tool of env
panel_s3_buzz_test (decision 21, Liam 05/11: "Duyệt, làm bước 1"). Builds panel_buzz.c and
buzz_tool.c with gcc (needs gcc on PATH)."""
import ctypes as ct
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'rpi5'))
import panel_proto as pp     # noqa: E402  (the same ST_* / OUT_* numbers as panel_proto.h)
BUZ = os.path.join(ROOT, 'panel', 'lib', 'panel_buzz')
PROTO = os.path.join(ROOT, 'panel', 'lib', 'panel_proto')
BUILD = os.path.join(HERE, 'build')

PB_NONE, PB_START, PB_DONE, PB_STOPPED, PB_FAULT, PB_LINK_LOST, PB_BOOT = range(7)
EV_NAV, EV_NAV_BACK, EV_SELECT, EV_START, EV_STOP = 1, 2, 4, 8, 16
F0 = 2700                    # PB_F0_DEFAULT


def hz(pm, f0=F0):
    """what pb_tone returns for a note of pm per mille of f0"""
    return (f0 * pm + 500) // 1000


def _build():
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, 'panel_buzz_host' + ('.dll' if os.name == 'nt' else '.so'))
    subprocess.run(['gcc', '-std=c99', '-O1', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I', BUZ, '-I', PROTO, os.path.join(BUZ, 'panel_buzz.c'),
                    os.path.join(BUZ, 'buzz_tool.c'), '-o', out], check=True)
    lib = ct.CDLL(out)
    lib.pb_watch.restype = ct.c_int
    lib.pb_watch.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_bool, ct.c_uint8, ct.c_uint16, ct.c_uint8, ct.c_uint32]
    lib.pb_tone.restype = ct.c_uint16
    lib.pb_tone.argtypes = [ct.c_void_p, ct.c_uint32]
    lib.pb_init.argtypes = [ct.c_void_p]
    lib.pb_set_f0.argtypes = [ct.c_void_p, ct.c_uint16]
    lib.pb_play.argtypes = [ct.c_void_p, ct.c_int, ct.c_uint32]
    lib.pb_busy.restype = ct.c_bool
    lib.pb_busy.argtypes = [ct.c_void_p]
    lib.pb_pattern.restype = ct.POINTER(ct.c_uint16)
    lib.pb_pattern.argtypes = [ct.c_int]
    lib.bt_init.argtypes = [ct.c_void_p, ct.c_uint32]
    lib.bt_buttons.argtypes = [ct.c_void_p, ct.c_uint8, ct.c_uint32]
    lib.bt_tone.restype = ct.c_uint16
    lib.bt_tone.argtypes = [ct.c_void_p, ct.c_uint32]
    for f in ('bt_mode_name', 'bt_pattern_name', 'bt_hint'):
        getattr(lib, f).restype = ct.c_char_p
        getattr(lib, f).argtypes = [ct.c_void_p]
    return lib


lib = _build()


def notes(ev):
    """[(pm, ms), ...] of a pattern, without the end mark"""
    p, out, k = lib.pb_pattern(ev), [], 0
    while p[2 * k + 1]:
        out.append((p[2 * k], p[2 * k + 1]))
        k += 1
    return out


class Buzz:
    def __init__(self):
        self.mem = ct.create_string_buffer(256)
        lib.pb_init(self.mem)
        self.t = 1000

    def watch(self, state, run_no, outcome=pp.OUT_NONE, link=True, sid=7, dt=10):
        self.t += dt
        return lib.pb_watch(self.mem, sid, link, state, run_no, outcome, self.t)

    def play(self, ev):
        lib.pb_play(self.mem, ev, self.t)

    def tone(self, dt=0):
        return lib.pb_tone(self.mem, self.t + dt)

    def timeline(self, step=10, n=60):
        return [self.tone(k * step) for k in range(n)]


def test_state_found_at_boot_is_silent():
    b = Buzz()
    assert b.watch(pp.ST_RUN, 5) == PB_NONE and b.tone() == 0
    assert b.watch(pp.ST_RUN, 5) == PB_NONE


def test_new_run_beeps_once_150ms_at_f0():
    b = Buzz()
    b.watch(pp.ST_IDLE, 0)
    assert b.watch(pp.ST_PREPARE, 1) == PB_START
    assert b.tone(0) == F0 and b.tone(149) == F0 and b.tone(150) == 0
    assert b.watch(pp.ST_ARM, 1) == PB_NONE and b.watch(pp.ST_RUN, 1) == PB_NONE


def test_completed_two_beeps_going_up_timeline():
    b = Buzz()
    b.watch(pp.ST_RUN, 1)
    assert b.watch(pp.ST_END, 1, pp.OUT_COMPLETED) == PB_DONE
    assert b.timeline(step=10, n=45) == [hz(889)] * 12 + [0] * 6 + [F0] * 25 + [0] * 2
    assert hz(889) < F0                                                  # going up
    assert b.watch(pp.ST_END, 1, pp.OUT_COMPLETED) == PB_NONE          # staying in END: silent


def test_user_stop_and_fault_and_end_with_error():
    b = Buzz()
    b.watch(pp.ST_RUN, 1)
    assert b.watch(pp.ST_STOPPING, 1, pp.OUT_USER_STOP) == PB_NONE
    assert b.watch(pp.ST_END, 1, pp.OUT_USER_STOP) == PB_STOPPED
    assert b.tone(0) == F0 and b.tone(399) == F0 and b.tone(400) == 0  # one long beep
    b.watch(pp.ST_RUN, 2)
    assert b.watch(pp.ST_ERROR, 2, pp.OUT_ERROR) == PB_FAULT and b.tone() == F0
    assert b.watch(pp.ST_ERROR, 2, pp.OUT_ERROR) == PB_NONE            # the fault beeps once
    assert b.watch(pp.ST_IDLE, 2) == PB_NONE                            # acknowledged: silent
    b.watch(pp.ST_RUN, 3)
    assert b.watch(pp.ST_END, 3, pp.OUT_ERROR) == PB_FAULT


def test_fault_pattern_is_three_long_beeps_at_f0():
    b = Buzz()
    b.watch(pp.ST_RUN, 1)
    b.watch(pp.ST_ERROR, 1, pp.OUT_ERROR)
    tl = b.timeline(step=50, n=30)
    assert tl == [F0] * 6 + [0] * 3 + [F0] * 6 + [0] * 3 + [F0] * 6 + [0] * 6


def test_link_lost_falls_link_back_is_silent():
    b = Buzz()
    b.watch(pp.ST_IDLE, 0)
    assert b.watch(pp.ST_IDLE, 0, link=False) == PB_LINK_LOST
    assert b.tone(0) == F0 and b.tone(150) == hz(667) and b.tone(400) == 0
    assert b.watch(pp.ST_IDLE, 0, link=False) == PB_NONE                # still down: once
    assert b.watch(pp.ST_RUN, 4, link=True, sid=9) == PB_NONE           # back, new session, mid-run
    assert b.watch(pp.ST_END, 4, pp.OUT_COMPLETED, sid=9) == PB_DONE    # then normal again


def test_new_session_without_a_gap_is_silent():
    b = Buzz()
    b.watch(pp.ST_END, 3, pp.OUT_COMPLETED, sid=7)
    assert b.watch(pp.ST_RUN, 1, sid=8) == PB_NONE
    assert b.watch(pp.ST_END, 1, pp.OUT_USER_STOP, sid=8) == PB_STOPPED


def test_run_seen_only_at_its_end_still_beeps_its_end():
    b = Buzz()
    b.watch(pp.ST_END, 3, pp.OUT_COMPLETED)
    assert b.watch(pp.ST_END, 4, pp.OUT_COMPLETED) == PB_DONE           # run 4 never seen active
    assert b.watch(pp.ST_ERROR, 5, pp.OUT_ERROR) == PB_FAULT


def test_new_event_replaces_the_pattern_playing():
    b = Buzz()
    b.watch(pp.ST_RUN, 1)
    b.watch(pp.ST_ERROR, 1, pp.OUT_ERROR)                               # long fault pattern
    assert b.tone(150) == F0
    assert b.watch(pp.ST_IDLE, 1, link=False, dt=100) == PB_LINK_LOST
    assert b.tone(0) == F0 and b.tone(150) == hz(667)                   # the fault would be F0


@pytest.mark.parametrize('ev', [PB_START, PB_DONE, PB_STOPPED, PB_FAULT, PB_LINK_LOST, PB_BOOT])
def test_patterns_end_stay_short_and_near_f0(ev):
    ns = notes(ev)
    assert ns and sum(ms for _, ms in ns) <= 2000
    assert all(pm == 0 or 600 <= pm <= 1000 for pm, _ in ns)           # 2/3 f0 .. f0: near resonance


def test_every_note_follows_f0_and_f0_is_clamped():
    b = Buzz()
    assert b.watch(pp.ST_IDLE, 0) == PB_NONE
    lib.pb_set_f0(b.mem, 2000)
    b.watch(pp.ST_IDLE, 0, link=False)
    assert b.tone(0) == 2000 and b.tone(150) == hz(667, 2000)
    for asked, kept in ((100, 500), (9000, 5000)):
        lib.pb_set_f0(b.mem, asked)
        b.play(PB_START)
        assert b.tone(0) == kept


def test_default_f0_is_2700():
    b = Buzz()
    b.play(PB_STOPPED)
    assert b.tone(0) == 2700


def test_boot_tune_rises_then_holds_f0_about_one_second():
    ns = notes(PB_BOOT)
    tones = [pm for pm, _ in ns if pm]
    assert tones[:4] == sorted(tones[:4]) and len(set(tones[:4])) == 4  # four notes going up
    assert ns[-1][0] == 1000 and ns[-1][1] == max(ms for _, ms in ns)  # ends on f0, held longest
    assert 900 <= sum(ms for _, ms in ns) <= 1500


def test_boot_tune_plays_through_the_first_status_and_link_up():
    b = Buzz()
    b.play(PB_BOOT)
    total = sum(ms for _, ms in notes(PB_BOOT))
    assert lib.pb_busy(b.mem) and b.tone(0) == hz(667)
    assert b.watch(pp.ST_IDLE, 0, dt=200) == PB_NONE                    # first look: silent
    assert b.watch(pp.ST_IDLE, 0, dt=10) == PB_NONE
    assert b.tone(0) != 0 and lib.pb_busy(b.mem)                        # still the tune (t = 210 ms)
    b.t = 1000                                                          # when the tune started
    assert b.tone(total - 1) == F0 and b.tone(total) == 0 and not lib.pb_busy(b.mem)


def test_real_event_replaces_the_boot_tune():
    b = Buzz()
    b.play(PB_BOOT)
    b.watch(pp.ST_IDLE, 0)
    assert b.watch(pp.ST_PREPARE, 1, dt=20) == PB_START
    assert b.tone(0) == F0 and b.tone(150) == 0                         # the tune is gone


# ---- buzzer tool (env panel_s3_buzz_test) ----

class Tool:
    def __init__(self):
        self.mem = ct.create_string_buffer(256)
        self.t = 5000
        lib.bt_init(self.mem, self.t)

    def press(self, ev, dt=10):
        self.t += dt
        lib.bt_buttons(self.mem, ev, self.t)

    def tone(self, dt=0):
        return lib.bt_tone(self.mem, self.t + dt)

    def mode(self):
        return lib.bt_mode_name(self.mem).decode()

    def pattern(self):
        return lib.bt_pattern_name(self.mem).decode()


def test_tool_starts_quiet_with_the_boot_tune():
    t = Tool()
    assert t.mode() == 'Quiet' and t.tone(0) == hz(667)
    assert t.tone(5000) == 0 and t.mode() == 'Quiet'
    assert 'START' in lib.bt_hint(t.mem).decode()


def test_tool_tone_steps_50hz_and_stop_is_quiet():
    t = Tool()
    t.press(EV_SELECT)
    assert t.mode() == 'Tone' and t.tone(0) == F0 + 50 and t.tone(10000) == F0 + 50
    t.press(EV_NAV)
    assert t.tone() == F0
    t.press(EV_NAV_BACK)
    assert t.tone() == F0 - 50
    t.press(EV_STOP)
    assert t.mode() == 'Quiet' and t.tone() == 0


def test_tool_tone_is_clamped():
    t = Tool()
    for _ in range(100):
        t.press(EV_NAV_BACK)
    assert t.tone() == 500
    for _ in range(200):
        t.press(EV_SELECT)
    assert t.tone() == 5000


def test_tool_sweep_1000_to_4000_every_500ms_then_quiet():
    t = Tool()
    t.press(EV_SELECT)
    t.press(EV_START)
    assert t.mode() == 'Sweep'
    assert [t.tone(k) for k in (0, 499, 500, 999, 1000)] == [1000, 1000, 1100, 1100, 1200]
    assert t.tone(15000) == 4000 and t.tone(15499) == 4000
    assert t.tone(15500) == 0 and t.mode() == 'Quiet'


def test_tool_stop_in_sweep_keeps_the_frequency_heard():
    t = Tool()
    t.press(EV_SELECT)
    t.press(EV_START)
    assert t.tone(2400) == 1400
    t.press(EV_STOP, dt=2550)                                           # 2550 ms into the sweep
    assert t.mode() == 'Tone' and t.tone(0) == 1500 and t.tone(3000) == 1500
    t.press(EV_SELECT)
    assert t.tone() == 1550


def test_tool_start_restarts_the_sweep():
    t = Tool()
    t.press(EV_SELECT)
    t.press(EV_START)
    t.press(EV_START, dt=3000)
    assert t.tone(0) == 1000


def test_tool_preview_plays_every_pattern_at_the_chosen_f0_then_quiet():
    t = Tool()
    t.press(EV_SELECT)
    for _ in range(14):
        t.press(EV_NAV)                                                 # 2750 - 700 = 2050
    t.press(EV_STOP)
    t.press(EV_START)
    assert t.mode() == 'Preview' and t.pattern() == 'Boot'
    f0 = 2050
    seen, heard, k, starts, ends = [], set(), 0, [], []
    while t.mode() == 'Preview' and k < 30000:
        f = t.tone(k)
        name = t.pattern()
        if f:
            heard.add(f)
            if not seen or seen[-1] != name:
                seen.append(name)
                starts.append(k)
            if len(ends) < len(starts):
                ends.append(k)
            ends[-1] = k
        k += 1
    assert seen == ['Boot', 'Start', 'Done', 'User stop', 'Fault', 'Link lost']
    assert all(s - e > 1200 for e, s in zip(ends, starts[1:]))          # BT_GAP_MS between patterns
    expected = {hz(pm, f0) for ev in range(1, 7) for pm, _ in notes(ev) if pm}
    assert heard == expected
    assert t.mode() == 'Quiet' and t.tone(k + 10) == 0


def test_tool_stop_in_preview_is_quiet():
    t = Tool()
    t.press(EV_START)
    assert t.tone(0) != 0
    t.press(EV_STOP)
    assert t.mode() == 'Quiet' and t.tone(0) == 0 and t.tone(3000) == 0
