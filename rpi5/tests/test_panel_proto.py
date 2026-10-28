"""panel_proto v2: frame layouts, CRC16, stream resync, sessions (#44, B1, B15, B16),
data age field (#45), non-blocking writer (B18) (workflow v3.1 section 9.6; Liam approved
B0 29/10; B0 review round 1). No hardware."""
import struct

import pytest

import panel_proto as pp


def test_crc16_ccitt_false_check_value():
    assert pp.crc16(b'123456789') == 0x29B1          # published check value of CRC-16/CCITT-FALSE


def test_cmd_layout_and_roundtrip():
    f = pp.encode_cmd(sid=0xBEEF, seq=200, cmd=pp.CMD_START, arg=0x0102)
    assert len(f) == pp.LEN_CMD
    assert f[:8] == bytes([0xCC, pp.PROTO_VERSION, 0xEF, 0xBE, 200, pp.CMD_START, 0x02, 0x01])
    assert struct.unpack_from('<H', f, 8)[0] == pp.crc16(f[:8])
    assert pp.decode(f) == pp.Cmd(0xBEEF, 200, pp.CMD_START, 0x0102)


def test_status_layout_and_roundtrip():
    s = pp.Status(sid=0x1234, daemon_boot=0xBEEF, panel_boot=0x4321, last_cmd_seq=9, last_result=pp.RES_OK,
                  run_state=pp.ST_RUN, phase_ds=123, profile=4, outcome=pp.OUT_COMPLETED,
                  quality=pp.Q_LOST | pp.Q_SPI_STALE, lost=17, data_age_ms=250,
                  links=pp.L_RPI | pp.L_H7, error_code=7)
    f = pp.encode_status(s)
    assert len(f) == pp.LEN_STATUS and f[0] == 0xDD and f[22:24] == b'\x00\x00'
    assert f[2:4] == b'\x34\x12' and f[4:6] == b'\xef\xbe' and f[18:20] == (250).to_bytes(2, 'little')
    assert f[20] == pp.L_RPI | pp.L_H7 and f[21] == 7
    assert pp.decode(f) == s


def test_status_saturates_instead_of_wrapping():
    d = pp.decode(pp.encode_status(pp.Status(phase_ds=10**6, lost=70_000, data_age_ms=-5)))
    assert d.phase_ds == 0xFFFF and d.lost == 0xFFFF and d.data_age_ms == 0


def test_profile_roundtrip_and_truncation():
    f = pp.encode_profile(sid=2, index=3, count=11, name='A-F2_figure8_speed2_long')
    assert len(f) == pp.LEN_PROFILE
    assert pp.decode(f) == pp.Profile(2, 3, 11, 'A-F2_figure8_speed')


# ── stream parser ─────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_stream_parser_splits_and_resyncs():
    a = pp.encode_cmd(1, 1, pp.CMD_SELECT, 2)
    b = pp.encode_cmd(1, 2, pp.CMD_START, 2)
    bad = bytearray(pp.encode_cmd(1, 3, pp.CMD_STOP))
    bad[5] ^= 0x40
    p = pp.StreamParser(accept=(pp.START_CMD,))
    got = p.feed(b'\x00\x55' + a + bytes(bad) + b[:3])
    got += p.feed(b[3:])
    assert got == [pp.decode(a), pp.decode(b)]
    assert p.bad_frames >= 1 and p.skipped_bytes >= 2


def test_start_byte_inside_payload_does_not_break_a_whole_frame():
    f = pp.encode_cmd(sid=0xCCCC, seq=0xCC, cmd=pp.CMD_STOP, arg=0xCCCC)
    assert pp.StreamParser().feed(b'\xcc' + f) == [pp.decode(f)]


def test_xor_blind_two_bit_error_is_caught():
    """B17: STOP 0x21 -> START 0x20 together with bit 0 of seq flipped kept the old XOR valid."""
    f = bytearray(pp.encode_cmd(5, 4, pp.CMD_STOP))
    f[5] ^= 0x01
    f[4] ^= 0x01
    assert pp.StreamParser().feed(bytes(f)) == []


def test_wrong_version_dropped_early():
    p = pp.StreamParser()
    assert p.feed(bytes([pp.START_CMD, pp.PROTO_VERSION + 1])) == []
    assert p.bad_frames == 1
    good = pp.encode_cmd(1, 1, pp.CMD_STOP)
    assert p.feed(good) == [pp.decode(good)]


def test_partial_frame_times_out_and_resyncs():
    clk = Clock()
    p = pp.StreamParser(stale_s=0.05, clock=clk)
    good = pp.encode_cmd(1, 1, pp.CMD_STOP)
    assert p.feed(good[:4]) == []                # a frame cut by a panel reset
    clk.t = 0.2
    assert p.feed(b'') == [] and p.stale_drops >= 1   # nothing else came: the cut frame expires
    assert p.feed(good) == [pp.decode(good)]
    p2 = pp.StreamParser(stale_s=0.05, clock=clk)
    p2.feed(good[:4])
    assert p2.feed(good) == [pp.decode(good)]          # next frame right behind: CRC resyncs


# ── sessions ──────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, result=pp.RES_OK):
        self.ran = []
        self.result = result

    def __call__(self, c):
        self.ran.append((c.seq, c.cmd, c.arg))
        return self.result


def rng_seq(*vals):
    it = iter(vals)
    return lambda: next(it)


def hello(sess, run, boot=0x4242):
    return sess.on_cmd(pp.Cmd(0, 0, pp.CMD_HELLO, boot), run)


def test_hello_opens_session_and_status_echoes_it():
    sess, run = pp.DaemonSession(daemon_boot=0x1111, rng=rng_seq(0x5A5A)), Runner()
    assert hello(sess, run) == 'hello'
    st = sess.fill(pp.Status())
    assert (st.sid, st.daemon_boot, st.panel_boot) == (0x5A5A, 0x1111, 0x4242)


def test_old_start_after_daemon_restart_is_dropped():
    """B1: the old daemon gave sid S; the new daemon must not run a START that still carries S."""
    old = pp.DaemonSession(daemon_boot=1)
    hello(old, Runner())
    stale_start = pp.Cmd(old.sid, 7, pp.CMD_START, 0)
    new, run = pp.DaemonSession(daemon_boot=2), Runner()
    assert new.on_cmd(stale_start, run) == 'dropped_sid'           # before HELLO
    hello(new, run, boot=0x4242)                                   # same panel says HELLO again
    if new.sid == old.sid:                                         # 1 in 65535: make the test exact
        pytest.skip('random sid collided')
    assert new.on_cmd(stale_start, run) == 'dropped_sid'
    assert run.ran == []


def test_hello_retry_keeps_the_session_and_late_old_hello_is_ignored():
    sess, run = pp.DaemonSession(daemon_boot=1, rng=rng_seq(0x1000, 0x2000, 0x3000)), Runner()
    hello(sess, run, boot=0xAAAA)
    assert sess.on_cmd(pp.Cmd(0x1000, 1, pp.CMD_SELECT, 3), run) == 'ran'
    assert hello(sess, run, boot=0xAAAA) == 'hello' and sess.sid == 0x1000    # retry: same session
    assert sess.on_cmd(pp.Cmd(0x1000, 2, pp.CMD_START, 3), run) == 'ran'     # seq history kept
    hello(sess, run, boot=0xBBBB)                                            # panel rebooted
    assert sess.sid == 0x2000
    assert hello(sess, run, boot=0xAAAA) == 'hello_stale' and sess.sid == 0x2000


def test_new_sid_differs_from_previous_and_is_never_zero():
    sess = pp.DaemonSession(daemon_boot=1, rng=rng_seq(0x0007, 0x0000, 0x0007, 0x0009))
    hello(sess, Runner(), boot=1)
    hello(sess, Runner(), boot=2)
    assert sess.sid == 0x0009


def test_one_command_in_flight_rules():
    sess, run = pp.DaemonSession(daemon_boot=1, rng=rng_seq(0x77)), Runner()
    hello(sess, run)
    sid = sess.sid
    assert sess.on_cmd(pp.Cmd(sid, 5, pp.CMD_START, 1), run) == 'ran'      # first: any seq
    assert sess.on_cmd(pp.Cmd(sid, 5, pp.CMD_START, 1), run) == 'repeated'  # lost STATUS, resent
    assert sess.on_cmd(pp.Cmd(sid, 9, pp.CMD_SELECT, 2), run) == 'bad_seq'  # skipped seqs: not run
    assert sess.fill(pp.Status()).last_result == pp.RES_BAD_SEQ
    assert sess.on_cmd(pp.Cmd(sid, 10, pp.CMD_SELECT, 2), run) == 'ran'     # continues from 9 + 1
    assert run.ran == [(5, pp.CMD_START, 1), (10, pp.CMD_SELECT, 2)]


@pytest.mark.parametrize('delta', [2, 61, 127, 128, 129, 255])
def test_any_jump_other_than_next_is_refused(delta):
    sess, run = pp.DaemonSession(daemon_boot=1, rng=rng_seq(0x77)), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(0x77, 200, pp.CMD_SELECT, 1), run)
    assert sess.on_cmd(pp.Cmd(0x77, (200 + delta) & 0xFF, pp.CMD_START, 1), run) == 'bad_seq'
    assert len(run.ran) == 1


def test_stop_always_runs_but_an_exact_resend_does_not_run_twice():
    sess, run = pp.DaemonSession(daemon_boot=1, rng=rng_seq(0x77)), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(0x77, 1, pp.CMD_START, 1), run)
    assert sess.on_cmd(pp.Cmd(0x77, 40, pp.CMD_STOP, 0), run) == 'ran'      # out of order: still runs
    assert sess.on_cmd(pp.Cmd(0x77, 40, pp.CMD_STOP, 0), run) == 'repeated'
    assert sess.on_cmd(pp.Cmd(0x77, 41, pp.CMD_STOP, 0), run) == 'ran'
    assert sess.on_cmd(pp.Cmd(0x77, 1, pp.CMD_STOP, 0), run) == 'ran'       # seq reused by a STOP
    assert [r[1] for r in run.ran] == [pp.CMD_START, pp.CMD_STOP, pp.CMD_STOP, pp.CMD_STOP]


def test_seq_wraps_around():
    sess, run = pp.DaemonSession(daemon_boot=1, rng=rng_seq(0x77)), Runner()
    hello(sess, run)
    for seq in (254, 255, 0, 1):
        assert sess.on_cmd(pp.Cmd(0x77, seq, pp.CMD_SELECT, seq), run) == 'ran'


def test_result_reported_for_the_acknowledged_command():
    sess = pp.DaemonSession(daemon_boot=1, rng=rng_seq(0x77))
    hello(sess, Runner())
    sess.on_cmd(pp.Cmd(0x77, 1, pp.CMD_START, 9), Runner(pp.RES_BAD_PROFILE))
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_result) == (1, pp.RES_BAD_PROFILE)
    sess.on_cmd(pp.Cmd(0x77, 2, 0x77, 0), Runner())
    assert sess.fill(pp.Status()).last_result == pp.RES_UNKNOWN


# ── writer ────────────────────────────────────────────────────────────────

class SlowUart:
    def __init__(self, sizes):
        self.sizes = list(sizes)        # bytes accepted per call; the list empty -> unlimited
        self.out = bytearray()

    def __call__(self, data):
        n = min(len(data), self.sizes.pop(0) if self.sizes else len(data))
        self.out += data[:n]
        return n


def frames_of(uart):
    return pp.StreamParser().feed(bytes(uart.out))


def test_writer_finishes_a_started_frame_and_keeps_newest_status():
    uart = SlowUart([5, 0])             # 5 bytes of status 1, then the UART is full
    tx = pp.TxQueue(uart)
    tx.put(pp.encode_status(pp.Status(phase_ds=1)))
    tx.pump()
    tx.put(pp.encode_status(pp.Status(phase_ds=2)))
    tx.put(pp.encode_status(pp.Status(phase_ds=3)))   # status 2 never sent
    tx.put(pp.encode_profile(1, 0, 1, 'A-L2'))
    tx.pump()
    got = frames_of(uart)
    assert [type(f).__name__ for f in got] == ['Status', 'Status', 'Profile']
    assert [got[0].phase_ds, got[1].phase_ds] == [1, 3]
    assert len(uart.out) == 2 * pp.LEN_STATUS + pp.LEN_PROFILE and not tx.pending()


def test_writer_replaces_a_status_not_started_yet():
    """B18: the first write takes 0 bytes; the old STATUS must not go out before the new one."""
    uart = SlowUart([0])
    tx = pp.TxQueue(uart)
    tx.put(pp.encode_status(pp.Status(phase_ds=1)))
    tx.pump()
    tx.put(pp.encode_status(pp.Status(phase_ds=2)))
    tx.pump()
    assert [f.phase_ds for f in frames_of(uart)] == [2] and tx.dropped_status == 1


def test_profiles_bounded_and_cleared_on_new_session():
    uart = SlowUart([0])
    tx = pp.TxQueue(uart, max_profiles=3)
    for i in range(5):
        tx.put(pp.encode_profile(1, i, 5, f'P{i}'))
    assert tx.dropped_profiles == 2
    tx.clear_profiles()
    tx.put(pp.encode_profile(2, 0, 1, 'NEW'))
    tx.pump()
    tx.pump()
    assert [f.name for f in frames_of(uart)] == ['NEW']
