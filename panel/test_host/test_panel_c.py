"""The C panel protocol (panel/lib/panel_proto) against the Python reference
(rpi5/panel_proto.py, v6), on the PC:

  * golden bytes: CRC, CMD encoding, STATUS / PROFILE / INFO decoding, stream parsing of
    mixed, cut and corrupted streams (same frames and counters as StreamParser);
  * every two-end scenario of rpi5/tests/test_panel_proto*.py that drives a PanelClient
    through a Bench is run again with the C client in its place (CPanelClient below).

Run:  python -m pytest panel/test_host -q      (needs gcc on PATH; builds the DLL itself)
"""
import ctypes as ct
import inspect
import os
import random
import struct
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'rpi5'))
sys.path.insert(0, os.path.join(ROOT, 'rpi5', 'tests'))

import panel_proto as pp            # noqa: E402
import test_panel_proto as tpp      # noqa: E402
import test_panel_proto_v6 as tv6   # noqa: E402

LIB_DIR = os.path.join(ROOT, 'panel', 'lib', 'panel_proto')
APP_DIR = os.path.join(ROOT, 'panel', 'lib', 'panel_app')
BUILD = os.path.join(HERE, 'build')


def _build():
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, 'panel_proto_host' + ('.dll' if os.name == 'nt' else '.so'))
    cmd = ['gcc', '-std=c99', '-O1', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
           '-I', LIB_DIR, '-I', APP_DIR, os.path.join(LIB_DIR, 'panel_proto.c'),
           os.path.join(APP_DIR, 'panel_app.c'), os.path.join(HERE, 'pp_host.c'), '-o', out]
    subprocess.run(cmd, check=True)
    return ct.CDLL(out)


lib = _build()

NONCE_FN = ct.CFUNCTYPE(ct.c_uint16, ct.c_void_p)
SEND_FN = ct.CFUNCTYPE(None, ct.c_void_p, ct.POINTER(ct.c_uint8), ct.c_size_t)
FRAME_FN = ct.CFUNCTYPE(None, ct.c_void_p, ct.POINTER(ct.c_uint8), ct.c_size_t)


class CStatus(ct.Structure):
    _fields_ = [('sid', ct.c_uint32), ('daemon_boot', ct.c_uint16), ('panel_boot', ct.c_uint16),
                ('last_cmd_seq', ct.c_uint16), ('last_result', ct.c_uint8), ('next_seq', ct.c_uint16),
                ('run_state', ct.c_uint8), ('phase_ds', ct.c_uint16), ('profile', ct.c_uint8),
                ('outcome', ct.c_uint8), ('quality', ct.c_uint8), ('lost', ct.c_uint16),
                ('data_age_ms', ct.c_uint16), ('links', ct.c_uint8), ('error_code', ct.c_uint8),
                ('last_cmd', ct.c_uint8)]


class CProfile(ct.Structure):
    _fields_ = [('sid', ct.c_uint32), ('index', ct.c_uint8), ('count', ct.c_uint8),
                ('name', ct.c_char * (pp.NAME_LEN + 1))]


class CInfo(ct.Structure):
    _fields_ = [('sid', ct.c_uint32), ('loop_us', ct.c_uint16), ('overruns', ct.c_uint16),
                ('flags', ct.c_uint8), ('ctrl_src', ct.c_uint8), ('ctrl', ct.c_char * (pp.CTRL_LEN + 1)),
                ('result', (ct.c_char * (pp.RESULT_LEN + 1)) * pp.RESULT_LINES),
                ('error_text', ct.c_char * (pp.ERROR_TEXT_LEN + 1)),
                ('run_no', ct.c_uint16), ('result_run', ct.c_uint16)]


def _sig(name, res, *args):
    f = getattr(lib, name)
    f.restype, f.argtypes = res, list(args)


_sig('pp_crc16', ct.c_uint16, ct.c_char_p, ct.c_size_t)
_sig('pp_encode_cmd', ct.c_size_t, ct.c_char_p, ct.c_uint32, ct.c_uint16, ct.c_uint8, ct.c_uint16)
_sig('pp_decode_status', None, ct.c_char_p, ct.POINTER(CStatus))
_sig('pp_decode_profile', None, ct.c_char_p, ct.POINTER(CProfile))
_sig('pp_decode_info', None, ct.c_char_p, ct.POINTER(CInfo))
_sig('pp_host_client_new', ct.c_void_p, NONCE_FN, SEND_FN)
_sig('pp_host_client_free', None, ct.c_void_p)
_sig('pp_client_press', None, ct.c_void_p, ct.c_uint8, ct.c_uint16, ct.c_uint32)
_sig('pp_client_tick', None, ct.c_void_p, ct.c_uint32)
_sig('pp_client_on_frame', None, ct.c_void_p, ct.c_char_p, ct.c_size_t, ct.c_uint32)
_sig('pp_client_link', ct.c_bool, ct.c_void_p, ct.c_uint32)
_sig('pp_client_pop_event', ct.c_bool, ct.c_void_p, ct.c_void_p)
_sig('pp_host_sid', ct.c_uint32, ct.c_void_p)
_sig('pp_host_boot', ct.c_uint16, ct.c_void_p)
_sig('pp_host_next_seq', ct.c_int, ct.c_void_p)
_sig('pp_host_stale', ct.c_int, ct.c_void_p)
_sig('pp_host_fence', ct.c_int, ct.c_void_p)
_sig('pp_host_stop_state', ct.c_int, ct.c_void_p)
_sig('pp_host_stop_failures', ct.c_uint32, ct.c_void_p)
_sig('pp_host_pending', ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_uint8),
     ct.POINTER(ct.c_uint16))
_sig('pp_host_stop', ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint16))
_sig('pp_host_profile', ct.c_int, ct.c_void_p, ct.c_int, ct.c_char_p)
_sig('pp_host_info', ct.POINTER(CInfo), ct.c_void_p)
_sig('pp_host_info_current', ct.c_int, ct.c_void_p)
_sig('pp_host_profile_count', ct.c_int, ct.c_void_p)
_sig('pp_host_ev_dropped', ct.c_uint32, ct.c_void_p)
_sig('pp_host_link_open', ct.c_int, ct.c_void_p)
_sig('pp_host_parser_new', ct.c_void_p, ct.c_uint8, ct.c_uint32)
_sig('pp_host_parser_free', None, ct.c_void_p)
_sig('pp_parser_feed', None, ct.c_void_p, ct.c_char_p, ct.c_size_t, ct.c_uint32, FRAME_FN, ct.c_void_p)
_sig('pp_host_parser_counts', None, ct.c_void_p, ct.POINTER(ct.c_uint32 * 3))


class CEvent(ct.Structure):
    _fields_ = [('what', ct.c_uint8), ('cmd', ct.c_uint8), ('has_result', ct.c_bool), ('result', ct.c_uint8)]


_WHAT = {1: 'done', 2: 'unverified', 3: 'refused'}
_STOP_STATE = {0: 'none', 1: 'requested', 2: 'acked', 3: 'confirmed'}


def _dec(b: bytes) -> str:
    return b.decode('ascii')


class CPanelClient:
    """Same interface as pp.PanelClient, backed by panel_proto.c."""

    def __init__(self, nonce):
        self._nonce = nonce
        self._err = None
        self._fallback = 0x8000

        def nonce_cb(_ctx):
            try:
                return self._nonce() & 0xFFFF
            except BaseException as e:          # the Python client would raise here
                self._err = self._err or e
                self._fallback += 1             # nonzero, never the current boot: C does not spin
                return self._fallback

        self._ncb = NONCE_FN(nonce_cb)
        self._scb = SEND_FN(lambda _ctx, data, n: self._out.append(bytes(data[:n])))
        self._out = []
        self.shown = []
        self._c = lib.pp_host_client_new(self._ncb, self._scb)
        self._check()

    def __del__(self):
        if getattr(self, '_c', None):
            lib.pp_host_client_free(self._c)

    def _check(self):
        if self._err is not None:
            e, self._err = self._err, None
            raise e

    def _drain(self):
        ev = CEvent()
        while lib.pp_client_pop_event(self._c, ct.byref(ev)):
            self.shown.append((_WHAT[ev.what], ev.cmd, ev.result if ev.has_result else None))

    def _call(self, fn, *args):
        self._out = []
        fn(self._c, *args)
        self._drain()
        self._check()
        return self._out

    def press(self, cmd, arg, now_ms):
        return self._call(lib.pp_client_press, cmd, arg, int(now_ms))

    def tick(self, now_ms):
        return self._call(lib.pp_client_tick, int(now_ms))

    def on_status(self, st, now_ms):
        f = pp.encode_status(st)
        return self._call(lib.pp_client_on_frame, f, len(f), int(now_ms))

    def on_profile(self, pr):
        f = pp.encode_profile(pr.sid, pr.index, pr.count, pr.name)
        self._call(lib.pp_client_on_frame, f, len(f), 0)

    def on_info(self, info):
        f = pp.encode_info(info)
        self._call(lib.pp_client_on_frame, f, len(f), 0)

    def link(self, now_ms):
        return bool(lib.pp_client_link(self._c, int(now_ms)))

    @property
    def sid(self):
        return lib.pp_host_sid(self._c)

    @property
    def boot(self):
        return lib.pp_host_boot(self._c)

    @property
    def next_seq(self):
        v = lib.pp_host_next_seq(self._c)
        return None if v < 0 else v

    @property
    def stale(self):
        v = lib.pp_host_stale(self._c)
        return None if v < 0 else v

    @property
    def fence(self):
        return bool(lib.pp_host_fence(self._c))

    @property
    def stop_state(self):
        return _STOP_STATE[lib.pp_host_stop_state(self._c)]

    @property
    def stop_failures(self):
        return lib.pp_host_stop_failures(self._c)

    @property
    def pending(self):
        seq, cmd, arg = ct.c_uint16(), ct.c_uint8(), ct.c_uint16()
        if not lib.pp_host_pending(self._c, ct.byref(seq), ct.byref(cmd), ct.byref(arg)):
            return None
        return dict(seq=seq.value, cmd=cmd.value, arg=arg.value)

    @property
    def stop(self):
        seq = ct.c_uint16()
        k = lib.pp_host_stop(self._c, ct.byref(seq))
        return None if k == 0 else dict(seq=seq.value if k == 2 else None)

    @property
    def profile_count(self):
        v = lib.pp_host_profile_count(self._c)
        return None if v < 0 else v

    @property
    def profiles(self):
        buf = ct.create_string_buffer(pp.NAME_LEN + 1)
        return {i: _dec(buf.value) for i in range(255) if lib.pp_host_profile(self._c, i, buf)}

    @property
    def info(self):
        p = lib.pp_host_info(self._c)
        if not p:
            return None
        i = p.contents
        return pp.Info(i.sid, i.loop_us, i.overruns, i.flags, i.ctrl_src, _dec(i.ctrl),
                       tuple(_dec(i.result[k].value) for k in range(pp.RESULT_LINES)),
                       _dec(i.error_text), i.run_no, i.result_run)

    @property
    def info_current(self):
        return bool(lib.pp_host_info_current(self._c))


# ── golden bytes ──────────────────────────────────────────────────────────

def test_crc_matches_python():
    rnd = random.Random(1)
    assert lib.pp_crc16(b'123456789', 9) == pp.crc16(b'123456789') == 0x29B1
    for n in range(0, 120):
        data = bytes(rnd.randrange(256) for _ in range(n))
        assert lib.pp_crc16(data, n) == pp.crc16(data)


def test_cmd_bytes_match_python():
    rnd = random.Random(2)
    buf = ct.create_string_buffer(pp.LEN_CMD)
    cases = [(0, 0, pp.CMD_HELLO, 0x1234), (0xFFFFFFFF, 0xFFFF, 0xFF, 0xFFFF)]
    cases += [(rnd.getrandbits(32), rnd.getrandbits(16), rnd.choice([1, 2, 3, 0x10, 0x20, 0x21]),
               rnd.getrandbits(16)) for _ in range(500)]
    for sid, seq, cmd, arg in cases:
        assert lib.pp_encode_cmd(buf, sid, seq, cmd, arg) == pp.LEN_CMD
        assert buf.raw == pp.encode_cmd(sid, seq, cmd, arg)


def test_status_decoding_matches_python():
    rnd = random.Random(3)
    names = [n for n, _ in CStatus._fields_]
    for _ in range(500):
        st = pp.Status(**{n: rnd.getrandbits(32 if n == 'sid' else (8 if CStatus._fields_[k][1] == ct.c_uint8 else 16))
                          for k, n in enumerate(names)})
        c = CStatus()
        lib.pp_decode_status(pp.encode_status(st), ct.byref(c))
        assert {n: getattr(c, n) for n in names} == {n: getattr(st, n) for n in names}


def test_profile_and_info_decoding_match_python():
    rnd = random.Random(4)
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz0123456789-_.:/%'

    def word(n):
        return ''.join(rnd.choice(abc) for _ in range(rnd.randrange(n + 5))).rstrip('\x00')

    for _ in range(300):
        pr = pp.decode(pp.encode_profile(rnd.getrandbits(32), rnd.randrange(256), rnd.randrange(256), word(18)))
        c = CProfile()
        lib.pp_decode_profile(pp.encode_profile(pr.sid, pr.index, pr.count, pr.name), ct.byref(c))
        assert (c.sid, c.index, c.count, _dec(c.name)) == (pr.sid, pr.index, pr.count, pr.name)
        info = pp.decode(pp.encode_info(pp.Info(rnd.getrandbits(32), rnd.getrandbits(16), rnd.getrandbits(16),
                                                rnd.getrandbits(8), rnd.randrange(3), word(8),
                                                (word(20), word(20), word(20)), word(20),
                                                rnd.getrandbits(16), rnd.getrandbits(16))))
        ci = CInfo()
        lib.pp_decode_info(pp.encode_info(info), ct.byref(ci))
        got = pp.Info(ci.sid, ci.loop_us, ci.overruns, ci.flags, ci.ctrl_src, _dec(ci.ctrl),
                      tuple(_dec(ci.result[k].value) for k in range(3)), _dec(ci.error_text),
                      ci.run_no, ci.result_run)
        assert got == info


def test_non_ascii_byte_drawn_as_question_mark():
    f = bytearray(pp.encode_profile(1, 0, 1, 'AB'))
    f[9] = 0xC3
    struct.pack_into('<H', f, 26, pp.crc16(bytes(f[:26])))
    c = CProfile()
    lib.pp_decode_profile(bytes(f), ct.byref(c))
    assert _dec(c.name) == 'A?'                       # Python gives 'A�': same position


def _c_parse(chunks, accept):
    p = lib.pp_host_parser_new(accept, 50)
    out = []
    cb = FRAME_FN(lambda _ctx, data, n: out.append(pp.decode(bytes(data[:n]))))
    for t_ms, data in chunks:
        lib.pp_parser_feed(p, data, len(data), t_ms, cb, None)
    counts = (ct.c_uint32 * 3)()
    lib.pp_host_parser_counts(p, ct.byref(counts))
    lib.pp_host_parser_free(p)
    return out, tuple(counts)


def _py_parse(chunks, accept):
    t = [0.0]
    sp = pp.StreamParser(accept=accept, stale_s=0.05, clock=lambda: t[0])
    out = []
    for t_ms, data in chunks:
        t[0] = t_ms / 1000.0
        out += sp.feed(data)
    return out, (sp.bad_frames, sp.skipped_bytes, sp.stale_drops)


@pytest.mark.parametrize('seed', range(40))
def test_stream_parser_matches_python_on_mixed_cut_and_corrupted_streams(seed):
    """Gaps never exactly 50 ms on purpose: the Python bench clock is a float in seconds, so at
    exactly 50.000 ms '0.2 - 0.15 > 0.05' may be true while the C integer test is false
    (Codex P10, kept as a known limit of this bench, not of the firmware)."""
    rnd = random.Random(100 + seed)
    stream = bytearray()
    for _ in range(60):
        k = rnd.randrange(5)
        if k == 0:
            f = pp.encode_status(pp.Status(sid=rnd.getrandbits(32), next_seq=rnd.getrandbits(16)))
        elif k == 1:
            f = pp.encode_profile(rnd.getrandbits(32), rnd.randrange(8), 8, 'A-L%d' % rnd.randrange(10))
        elif k == 2:
            f = pp.encode_info(pp.Info(sid=rnd.getrandbits(32), ctrl='PID', result=('x',)))
        elif k == 3:
            f = pp.encode_cmd(rnd.getrandbits(32), rnd.getrandbits(16), 0x20, 1)
        else:
            f = bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 12)))     # noise
        f = bytearray(f)
        if rnd.random() < 0.15 and len(f) > 3:
            f[rnd.randrange(len(f))] ^= 1 << rnd.randrange(8)                      # bit error
        if rnd.random() < 0.1 and len(f) > 3:
            f = f[:rnd.randrange(1, len(f))]                                      # cut frame
        stream += f
    chunks, i, t = [], 0, 0
    while i < len(stream):
        n = rnd.randrange(0, 40)
        t += rnd.choice([0, 1, 10, 30, 70, 200])                                  # never exactly 50
        chunks.append((t, bytes(stream[i:i + n])))
        i += n
    for accept, mask in [((pp.START_STATUS, pp.START_PROFILE, pp.START_INFO), 0b1110),
                         ((pp.START_CMD, pp.START_STATUS, pp.START_PROFILE, pp.START_INFO), 0b1111)]:
        assert _c_parse(chunks, mask) == _py_parse(chunks, accept)


# ── two-end scenarios with the C client ───────────────────────────────────

def _uses_bench(f):
    src = inspect.getsource(f)
    return 'Bench(' in src or 'PanelClient(' in src


_TWO_END = [(name, f) for mod in (tpp, tv6) for name, f in vars(mod).items()
            if name.startswith('test_') and callable(f) and _uses_bench(f)]
for _name, _f in _TWO_END:
    globals()[_name + '__c'] = _f


@pytest.fixture(autouse=True)
def _c_client(request, monkeypatch):
    if request.node.name.split('[')[0].endswith('__c'):
        monkeypatch.setattr(pp, 'PanelClient', CPanelClient)


def test_two_end_scenarios_were_found():
    # 23 Bench scenarios in test_panel_proto.py (handshake ... commands_refused) + 4 in v6 (2 of
    # them for info_current, R1-R5); a new Bench scenario there makes this fail until it is
    # counted here on purpose
    names = [n for n, _ in _TWO_END]
    assert len(names) == 27 and 'test_panel_keeps_info_of_its_session_and_forgets_it_when_lost' in names
    assert 'test_info_stops_being_current_after_a_status_gap_r1_r5' in names


def test_c_client_is_really_used_in_the_scenarios(monkeypatch):
    monkeypatch.setattr(pp, 'PanelClient', CPanelClient)
    b = tpp.Bench()
    b.step(10)
    assert isinstance(b.panel, CPanelClient) and b.panel.sid == tpp.SID1


# ── differential fuzz: Python PanelClient and C client side by side ───────

def _snap(panel):
    return (panel.sid, panel.boot, panel.next_seq, panel.stale, panel.fence, panel.stop_state,
            panel.stop_failures, panel.pending and (panel.pending['seq'], panel.pending['cmd'], panel.pending['arg']),
            panel.stop and panel.stop['seq'], tuple(panel.shown), tuple(sorted(panel.profiles.items())),
            panel.info, panel.profile_count, panel.info_current)


class _FuzzBench(tpp.HoldBench):
    """HoldBench whose drops, holds, presses, daemon restarts and late seqs come from one seed;
    two copies with the same seed differ only in the panel implementation."""

    def __init__(self, seed, client):
        self.rnd = random.Random(seed)
        nonces = [self.rnd.choice([0x1111, 0x2222, self.rnd.getrandbits(16) or 1]) for _ in range(400)]
        it = iter(nonces)
        super().__init__(stop_ms=self.rnd.choice([0, 50, 400]))
        self.panel = client(lambda: next(it))
        p_up, p_down = self.rnd.choice([0, 0.1, 0.4]), self.rnd.choice([0, 0.1, 0.4])
        self.drop_up = lambda f: self.rnd.random() < p_up
        self.blackout_until = -1
        self.drop_down = lambda f: self.t < self.blackout_until or self.rnd.random() < p_down
        self.fail_stop = self.rnd.choice([0, 0, 2])


@pytest.mark.parametrize('seed', range(150))
def test_c_client_matches_python_step_by_step(seed):
    benches = [_FuzzBench(seed, pp.PanelClient), _FuzzBench(seed, CPanelClient)]
    ctl = random.Random(10_000 + seed)
    for step in range(400):
        r = ctl.random()
        action = None
        if r < 0.04:
            action = ('press', ctl.choice([pp.CMD_SELECT, pp.CMD_START, pp.CMD_STOP]), ctl.randrange(3))
        elif r < 0.05:
            action = ('hold', ctl.random() < 0.5)
        elif r < 0.053:
            action = ('restart', 8 + step)
        elif r < 0.056:
            action = ('late_seq', pp.SEQ_END - ctl.randrange(1, 4))
        elif r < 0.06:
            action = ('info', ctl.choice(['PID', 'LADRC']), ctl.randrange(2000), ctl.randrange(4))
        elif r < 0.064:
            action = ('blackout', ctl.choice([600, 3500]))          # no STATUS: NO LINK, give-up
        for b in benches:
            if action is None:
                pass
            elif action[0] == 'press':
                b.press(action[1], action[2])
            elif action[0] == 'hold':
                b.hold = action[1]
            elif action[0] == 'restart':
                b.sess = pp.DaemonSession(action[1])
            elif action[0] == 'late_seq':
                if b.sess.sid:
                    b.sess.next_seq = max(b.sess.next_seq, action[1])
            elif action[0] == 'blackout':
                b.blackout_until = b.t + action[1]
            elif action[0] == 'info':
                b.panel.on_info(pp.Info(sid=b.sess.sid, ctrl=action[1], loop_us=action[2], run_no=action[3]))
            b.step(1)
        py, c = benches
        assert _snap(c.panel) == _snap(py.panel), f'step {step} action {action}'
        assert c.up == py.up and c.held == py.held and c.ran == py.ran and c.log == py.log, f'step {step}'


def test_event_ring_keeps_newest_and_counts_dropped_codex_p18():
    c = CPanelClient(iter(range(1, 100)).__next__)
    cmds = [pp.CMD_START if k % 3 else pp.CMD_SELECT for k in range(40)]   # distinct events (Q7)
    for cmd in cmds:                                     # no session: every press is refused
        lib.pp_client_press(c._c, cmd, 0, 0)
    c._drain()
    assert len(c.shown) == pp_EVENTS and lib.pp_host_ev_dropped(c._c) == 40 - pp_EVENTS
    assert [x[1] for x in c.shown] == cmds[40 - pp_EVENTS:]     # the newest, in order


pp_EVENTS = 32


def test_link_expires_and_a_millis_wrap_does_not_revive_it_codex_p9(monkeypatch):
    monkeypatch.setattr(pp, 'PanelClient', CPanelClient)
    b = tpp.Bench()
    b.step(10)
    assert b.panel.link(b.t)
    t_last = b.t
    b.panel.tick(t_last + 600)
    assert not lib.pp_host_link_open(b.panel._c)
    assert not b.panel.link((t_last + 2 ** 32) & 0xFFFFFFFF)     # same last_status_ms after a wrap
