"""panel_proto: frame layouts, stream resync, sessions (#44), data age field (#45),
non-blocking writer (workflow v3.1 section 9.6; Liam approved B0 29/10). No hardware."""
import pytest

import panel_proto as pp


def test_cmd_layout_and_roundtrip():
    f = pp.encode_cmd(sid=7, seq=200, cmd=pp.CMD_START, arg=0x0102)
    assert len(f) == pp.LEN_CMD
    assert f[:7] == bytes([0xCC, pp.PROTO_VERSION, 7, 200, pp.CMD_START, 0x02, 0x01])
    assert f[7] == pp.xor8(f[:7])
    assert pp.decode(f) == pp.Cmd(7, 200, pp.CMD_START, 0x0102)


def test_status_layout_and_roundtrip():
    s = pp.Status(sid=3, daemon_boot=0xBEEF, panel_boot=0x1234, last_cmd_seq=9, last_result=pp.RES_OK,
                  run_state=pp.ST_RUN, phase_ds=123, profile=4, outcome=pp.OUT_COMPLETED,
                  quality=pp.Q_LOST | pp.Q_SPI_STALE, lost=17, data_age_ms=250,
                  links=pp.L_RPI | pp.L_H7, error_code=0)
    f = pp.encode_status(s)
    assert len(f) == pp.LEN_STATUS and f[0] == 0xDD and f[21:23] == b'\x00\x00'
    assert f[3:5] == b'\xef\xbe' and f[17:19] == (250).to_bytes(2, 'little')
    assert pp.decode(f) == s


def test_status_saturates_instead_of_wrapping():
    f = pp.encode_status(pp.Status(phase_ds=10**6, lost=70_000, data_age_ms=-5))
    d = pp.decode(f)
    assert d.phase_ds == 0xFFFF and d.lost == 0xFFFF and d.data_age_ms == 0


def test_profile_roundtrip_and_truncation():
    f = pp.encode_profile(sid=2, index=3, count=11, name='A-F2_figure8_speed2_long')
    assert len(f) == pp.LEN_PROFILE
    assert pp.decode(f) == pp.Profile(2, 3, 11, 'A-F2_figure8_speed')


def test_stream_parser_splits_and_resyncs():
    a = pp.encode_cmd(1, 1, pp.CMD_SELECT, 2)
    b = pp.encode_cmd(1, 2, pp.CMD_START, 2)
    bad = bytearray(pp.encode_cmd(1, 3, pp.CMD_STOP))
    bad[4] ^= 0x40                                       # corrupted: XOR fails
    stream = b'\x00\x55' + a + bytes(bad) + b[:3]
    p = pp.StreamParser(accept=(pp.START_CMD,))
    got = p.feed(stream)
    got += p.feed(b[3:])                                 # rest of b arrives later
    assert got == [pp.decode(a), pp.decode(b)]
    assert p.bad_frames >= 1 and p.skipped_bytes >= 2


def test_stream_parser_rejects_other_version():
    f = bytearray(pp.encode_cmd(1, 1, pp.CMD_STOP))
    f[1] = pp.PROTO_VERSION + 1
    f[-1] = pp.xor8(f[:-1])
    assert pp.StreamParser().feed(bytes(f)) == []


# ── sessions ──────────────────────────────────────────────────────────────

class Judge:
    def __init__(self):
        self.ran = []

    def __call__(self, c):
        self.ran.append((c.cmd, c.arg))
        return pp.RES_OK


def hello(sess, judge, boot=0x4242):
    sess.on_cmd(pp.Cmd(0, 0, pp.CMD_HELLO, boot), judge)
    return sess.sid


def test_hello_opens_session_and_status_echoes_it():
    sess, judge = pp.DaemonSession(daemon_boot=0x1111), Judge()
    sid = hello(sess, judge)
    assert sid != 0
    st = sess.fill(pp.Status())
    assert (st.sid, st.daemon_boot, st.panel_boot) == (sid, 0x1111, 0x4242)


def test_command_before_hello_or_from_old_session_is_dropped():
    sess, judge = pp.DaemonSession(daemon_boot=1), Judge()
    assert sess.on_cmd(pp.Cmd(1, 1, pp.CMD_START, 0), judge) is None   # no session yet
    sid = hello(sess, judge)
    hello(sess, judge, boot=0x5555)                      # panel restarted: new session
    assert sess.on_cmd(pp.Cmd(sid, 2, pp.CMD_START, 0), judge) is None  # old START in the UART
    assert judge.ran == [] and sess.dropped_sid == 2


def test_repeated_seq_is_answered_not_run_again():
    sess, judge = pp.DaemonSession(daemon_boot=1), Judge()
    sid = hello(sess, judge)
    assert sess.on_cmd(pp.Cmd(sid, 5, pp.CMD_START, 1), judge) is not None
    assert sess.on_cmd(pp.Cmd(sid, 5, pp.CMD_START, 1), judge) is None  # resent after lost STATUS
    assert sess.on_cmd(pp.Cmd(sid, 4, pp.CMD_STOP, 0), judge) is None   # older
    assert sess.on_cmd(pp.Cmd(sid, 6, pp.CMD_STOP, 0), judge) is not None
    assert sess.on_cmd(pp.Cmd(sid, 7, pp.CMD_STOP, 0), judge) is not None  # STOP again: fine
    assert judge.ran == [(pp.CMD_START, 1), (pp.CMD_STOP, 0), (pp.CMD_STOP, 0)]
    assert sess.repeats == 2 and sess.fill(pp.Status()).last_cmd_seq == 7


def test_seq_wraps_around():
    sess, judge = pp.DaemonSession(daemon_boot=1), Judge()
    sid = hello(sess, judge)
    for seq in (254, 255, 0, 1):
        assert sess.on_cmd(pp.Cmd(sid, seq, pp.CMD_STOP, 0), judge) is not None
    assert len(judge.ran) == 4


def test_judge_result_reported_and_unknown_command():
    sess = pp.DaemonSession(daemon_boot=1)
    sid = hello(sess, Judge())
    sess.on_cmd(pp.Cmd(sid, 1, pp.CMD_START, 9), lambda c: pp.RES_BAD_PROFILE)
    assert sess.fill(pp.Status()).last_result == pp.RES_BAD_PROFILE
    sess.on_cmd(pp.Cmd(sid, 2, 0x77, 0), lambda c: pp.RES_OK)
    assert sess.fill(pp.Status()).last_result == pp.RES_UNKNOWN


def test_sid_never_zero_after_wrap():
    sess, judge = pp.DaemonSession(daemon_boot=1), Judge()
    sids = {hello(sess, judge) for _ in range(600)}
    assert 0 not in sids


# ── writer ────────────────────────────────────────────────────────────────

class SlowUart:
    def __init__(self, sizes):
        self.sizes = list(sizes)        # bytes accepted per call
        self.out = bytearray()

    def __call__(self, data):
        n = min(len(data), self.sizes.pop(0) if self.sizes else len(data))
        self.out += data[:n]
        return n


def test_writer_never_sends_half_frames_and_keeps_newest_status():
    uart = SlowUart([5, 0])             # first pump: 5 bytes of status #1, then the UART is full
    tx = pp.TxQueue(uart)
    s1 = pp.encode_status(pp.Status(phase_ds=1))
    s2 = pp.encode_status(pp.Status(phase_ds=2))
    s3 = pp.encode_status(pp.Status(phase_ds=3))
    prof = pp.encode_profile(1, 0, 1, 'A-L2')
    tx.put(s1)
    tx.pump()
    tx.put(s2)
    tx.put(s3)                          # s2 never sent: replaced by s3
    tx.put(prof)
    tx.pump()                           # unlimited from now on
    assert not tx.pending() and tx.dropped_status == 1
    frames = pp.StreamParser().feed(bytes(uart.out))
    assert [type(f).__name__ for f in frames] == ['Status', 'Profile', 'Status']
    assert frames[0].phase_ds == 1 and frames[2].phase_ds == 3
    assert len(uart.out) == 2 * pp.LEN_STATUS + pp.LEN_PROFILE      # every byte belongs to a whole frame
