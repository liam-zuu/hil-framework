"""lib/panel_buzz on the PC (Liam 05/11, PANEL session: "Làm còi trước"; buzzer passive,
low-level trigger): which STATUS changes beep, which do not, and the tone timeline of each
pattern. Builds panel_buzz.c with gcc (needs gcc on PATH)."""
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

PB_NONE, PB_START, PB_DONE, PB_STOPPED, PB_FAULT, PB_LINK_LOST = range(6)


def _build():
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, 'panel_buzz_host' + ('.dll' if os.name == 'nt' else '.so'))
    subprocess.run(['gcc', '-std=c99', '-O1', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I', BUZ, '-I', PROTO, os.path.join(BUZ, 'panel_buzz.c'), '-o', out], check=True)
    lib = ct.CDLL(out)
    lib.pb_watch.restype = ct.c_int
    lib.pb_watch.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_bool, ct.c_uint8, ct.c_uint16, ct.c_uint8, ct.c_uint32]
    lib.pb_tone.restype = ct.c_uint16
    lib.pb_tone.argtypes = [ct.c_void_p, ct.c_uint32]
    lib.pb_init.argtypes = [ct.c_void_p]
    return lib


lib = _build()


class Buzz:
    def __init__(self):
        self.mem = ct.create_string_buffer(256)
        lib.pb_init(self.mem)
        self.t = 1000

    def watch(self, state, run_no, outcome=pp.OUT_NONE, link=True, sid=7, dt=10):
        self.t += dt
        return lib.pb_watch(self.mem, sid, link, state, run_no, outcome, self.t)

    def tone(self, dt=0):
        return lib.pb_tone(self.mem, self.t + dt)

    def timeline(self, step=10, n=60):
        return [self.tone(k * step) for k in range(n)]


def test_state_found_at_boot_is_silent():
    b = Buzz()
    assert b.watch(pp.ST_RUN, 5) == PB_NONE and b.tone() == 0
    assert b.watch(pp.ST_RUN, 5) == PB_NONE


def test_new_run_beeps_once_short():
    b = Buzz()
    b.watch(pp.ST_IDLE, 0)
    assert b.watch(pp.ST_PREPARE, 1) == PB_START
    assert b.tone(0) == 2700 and b.tone(79) == 2700 and b.tone(80) == 0
    assert b.watch(pp.ST_ARM, 1) == PB_NONE and b.watch(pp.ST_RUN, 1) == PB_NONE


def test_completed_two_beeps_timeline():
    b = Buzz()
    b.watch(pp.ST_RUN, 1)
    assert b.watch(pp.ST_END, 1, pp.OUT_COMPLETED) == PB_DONE
    assert b.timeline(step=10, n=30) == [2700] * 9 + [0] * 9 + [2700] * 9 + [0] * 3
    assert b.watch(pp.ST_END, 1, pp.OUT_COMPLETED) == PB_NONE          # staying in END: silent


def test_user_stop_and_fault_and_end_with_error():
    b = Buzz()
    b.watch(pp.ST_RUN, 1)
    assert b.watch(pp.ST_STOPPING, 1, pp.OUT_USER_STOP) == PB_NONE
    assert b.watch(pp.ST_END, 1, pp.OUT_USER_STOP) == PB_STOPPED and b.tone() == 2000
    b.watch(pp.ST_RUN, 2)
    assert b.watch(pp.ST_ERROR, 2, pp.OUT_ERROR) == PB_FAULT and b.tone() == 1500
    assert b.watch(pp.ST_ERROR, 2, pp.OUT_ERROR) == PB_NONE            # the fault beeps once
    assert b.watch(pp.ST_IDLE, 2) == PB_NONE                            # acknowledged: silent
    b.watch(pp.ST_RUN, 3)
    assert b.watch(pp.ST_END, 3, pp.OUT_ERROR) == PB_FAULT


def test_fault_pattern_is_three_long_low_beeps():
    b = Buzz()
    b.watch(pp.ST_RUN, 1)
    b.watch(pp.ST_ERROR, 1, pp.OUT_ERROR)
    tl = b.timeline(step=50, n=30)
    assert tl == [1500] * 6 + [0] * 3 + [1500] * 6 + [0] * 3 + [1500] * 6 + [0] * 6


def test_link_lost_beeps_link_back_is_silent():
    b = Buzz()
    b.watch(pp.ST_IDLE, 0)
    assert b.watch(pp.ST_IDLE, 0, link=False) == PB_LINK_LOST
    assert b.tone(0) == 2000 and b.tone(150) == 1200 and b.tone(400) == 0
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
    assert b.tone(0) == 1500
    assert b.watch(pp.ST_IDLE, 1, link=False, dt=100) == PB_LINK_LOST
    assert b.tone(0) == 2000


@pytest.mark.parametrize('ev', [PB_START, PB_DONE, PB_STOPPED, PB_FAULT, PB_LINK_LOST])
def test_patterns_end_and_stay_short(ev):
    lib.pb_pattern.restype = ct.POINTER(ct.c_uint16 * 2)
    notes, k = [], 0
    p = ct.cast(lib.pb_pattern(ev), ct.POINTER(ct.c_uint16))
    while True:
        hz, ms = p[2 * k], p[2 * k + 1]
        if ms == 0:
            break
        notes.append((hz, ms))
        k += 1
    assert notes and sum(ms for _, ms in notes) <= 2000
    assert all(hz == 0 or 1000 <= hz <= 4000 for hz, _ in notes)
