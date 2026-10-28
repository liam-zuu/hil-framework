"""panel_proto v3: frame layouts, CRC16, stream resync, sessions (#44, B1, B15, B16, B23-B27),
data age field (#45), non-blocking writer (B18, B29) (workflow v3.1 section 9.6; Liam approved
B0 29/10; B0 review rounds 1-2). The round 2 counter-examples are Codex's
(codex_tra_loi_vong2.md, B23-B29); expected outcomes follow the session rules written in the
panel_proto docstring. No hardware."""
import struct

import pytest

import panel_proto as pp

EPOCH = 7
SID1 = EPOCH << 16 | 1          # first session of epoch 7


def test_crc16_ccitt_false_check_value():
    assert pp.crc16(b'123456789') == 0x29B1          # published check value of CRC-16/CCITT-FALSE


def test_cmd_layout_and_roundtrip():
    f = pp.encode_cmd(sid=0xDEADBEEF, seq=200, cmd=pp.CMD_START, arg=0x0102)
    assert len(f) == pp.LEN_CMD == 12
    assert f[:10] == bytes([0xCC, pp.PROTO_VERSION, 0xEF, 0xBE, 0xAD, 0xDE, 200, pp.CMD_START, 0x02, 0x01])
    assert struct.unpack_from('<H', f, 10)[0] == pp.crc16(f[:10])
    assert pp.decode(f) == pp.Cmd(0xDEADBEEF, 200, pp.CMD_START, 0x0102)


def test_status_layout_and_roundtrip():
    s = pp.Status(sid=0x00071234, daemon_boot=0xBEEF, panel_boot=0x4321, last_cmd_seq=9, last_result=pp.RES_OK,
                  next_seq=10, run_state=pp.ST_RUN, phase_ds=123, profile=4, outcome=pp.OUT_COMPLETED,
                  quality=pp.Q_LOST | pp.Q_SPI_STALE, lost=17, data_age_ms=250,
                  links=pp.L_RPI | pp.L_H7, error_code=7)
    f = pp.encode_status(s)
    assert len(f) == pp.LEN_STATUS == 28 and f[0] == 0xDD and f[25] == 0
    assert f[2:6] == b'\x34\x12\x07\x00' and f[6:8] == b'\xef\xbe' and f[8:10] == b'\x21\x43'
    assert (f[10], f[11], f[12], f[13]) == (9, pp.RES_OK, 10, pp.ST_RUN)
    assert f[21:23] == (250).to_bytes(2, 'little') and f[23] == pp.L_RPI | pp.L_H7 and f[24] == 7
    assert struct.unpack_from('<H', f, 26)[0] == pp.crc16(f[:26])
    assert pp.decode(f) == s


def test_status_saturates_instead_of_wrapping():
    d = pp.decode(pp.encode_status(pp.Status(phase_ds=10**6, lost=70_000, data_age_ms=-5)))
    assert d.phase_ds == 0xFFFF and d.lost == 0xFFFF and d.data_age_ms == 0


def test_profile_roundtrip_and_truncation():
    f = pp.encode_profile(sid=SID1, index=3, count=11, name='A-F2_figure8_speed2_long')
    assert len(f) == pp.LEN_PROFILE == 28 and f[8:26].rstrip(b'\x00') == b'A-F2_figure8_speed'
    assert pp.decode(f) == pp.Profile(SID1, 3, 11, 'A-F2_figure8_speed')


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
    bad[7] ^= 0x40
    p = pp.StreamParser(accept=(pp.START_CMD,))
    got = p.feed(b'\x00\x55' + a + bytes(bad) + b[:3])
    got += p.feed(b[3:])
    assert got == [pp.decode(a), pp.decode(b)]
    assert p.bad_frames >= 1 and p.skipped_bytes >= 2


def test_start_byte_inside_payload_does_not_break_a_whole_frame():
    f = pp.encode_cmd(sid=0xCCCCCCCC, seq=0xCC, cmd=pp.CMD_STOP, arg=0xCCCC)
    assert pp.StreamParser().feed(b'\xcc' + f) == [pp.decode(f)]


def test_xor_blind_two_bit_error_is_caught():
    """B17: STOP 0x21 -> START 0x20 together with bit 0 of seq flipped kept the old XOR valid."""
    f = bytearray(pp.encode_cmd(5, 4, pp.CMD_STOP))
    f[7] ^= 0x01
    f[6] ^= 0x01
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
    assert p.feed(b'') == [] and p.stale_drops == 1   # nothing else came: the cut frame expires
    assert p.feed(good) == [pp.decode(good)]
    p2 = pp.StreamParser(stale_s=0.05, clock=clk)
    p2.feed(good[:4])
    assert p2.feed(good) == [pp.decode(good)]          # next frame right behind: CRC resyncs


def test_tail_arriving_after_the_deadline_is_not_joined_to_the_old_head():
    """Codex B28: head at t = 0, tail at t = 0.2 with stale_s = 0.05 and no feed(b'') between."""
    clk = Clock()
    p = pp.StreamParser(stale_s=0.05, clock=clk)
    f = pp.encode_cmd(1, 1, pp.CMD_START, 2)
    assert p.feed(f[:4]) == []
    clk.t = 0.2
    assert p.feed(f[4:]) == [] and p.stale_drops == 1


def test_bytes_arriving_in_time_keep_a_frame_alive():
    """The age is that of the newest byte: a frame sent slowly, byte by byte, still decodes."""
    clk = Clock()
    p = pp.StreamParser(stale_s=0.05, clock=clk)
    f = pp.encode_cmd(1, 1, pp.CMD_START, 2)
    got = []
    for i, b in enumerate(f):
        clk.t = i * 0.04
        got += p.feed(bytes([b]))
    assert got == [pp.decode(f)]


# ── epoch ─────────────────────────────────────────────────────────────────

def test_epoch_counts_up_across_daemon_starts(tmp_path):
    path = str(tmp_path / 'panel_epoch')
    assert [pp.next_daemon_epoch(path) for _ in range(3)] == [1, 2, 3]
    assert (tmp_path / 'panel_epoch').read_text(encoding='ascii').strip() == '3'
    assert not (tmp_path / 'panel_epoch.tmp').exists()


@pytest.mark.parametrize('content', ['', 'x', '-1', '65535', '3.0'])
def test_unreadable_epoch_file_is_refused(tmp_path, content):
    p = tmp_path / 'panel_epoch'
    p.write_text(content, encoding='ascii')
    with pytest.raises(ValueError):
        pp.next_daemon_epoch(str(p))
    assert p.read_text(encoding='ascii') == content                    # left for the operator


# ── sessions ──────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, result=pp.RES_OK, fail_on=()):
        self.ran = []
        self.result = result
        self.fail_on = fail_on

    def __call__(self, c):
        self.ran.append((c.seq, c.cmd, c.arg))
        if c.cmd in self.fail_on:
            raise RuntimeError('action failed')
        return self.result


def hello(sess, run, boot=0x4242, sid=0):
    return sess.on_cmd(pp.Cmd(sid, 0, pp.CMD_HELLO, boot), run)


def test_hello_opens_session_and_status_echoes_it():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    assert hello(sess, run) == 'hello'
    st = sess.fill(pp.Status())
    assert (st.sid, st.daemon_boot, st.panel_boot) == (SID1, EPOCH, 0x4242)
    assert (st.last_result, st.next_seq) == (pp.RES_NONE, 0)       # B26: no ACK to mistake


def test_old_start_after_daemon_restart_is_dropped_codex_b23():
    """B23: the old daemon's sid can never be the new daemon's (epoch from the file), so its
    START is dropped before and after the panel's new HELLO. Old test skipped this case."""
    old = pp.DaemonSession(epoch=1)
    hello(old, Runner())
    stale_start = pp.Cmd(old.sid, 0, pp.CMD_START, 0)
    new, run = pp.DaemonSession(epoch=2), Runner()
    assert new.on_cmd(stale_start, run) == 'dropped_sid'            # before HELLO
    hello(new, run, boot=0x4242)                                    # same panel says HELLO again
    assert new.sid != old.sid
    assert new.on_cmd(stale_start, run) == 'dropped_sid'
    assert run.ran == []


def test_sid_of_an_earlier_session_is_never_reused():
    """B23 inside one daemon: sessions A -> B -> new HELLO never gets A's sid back."""
    sess = pp.DaemonSession(epoch=EPOCH)
    sids = []
    for boot in (0xA, 0xB, 0xA, 0xB):
        hello(sess, Runner(), boot=boot)
        sids.append(sess.sid)
    assert len(set(sids)) == 4 and sids[0] == SID1


def test_bad_epoch_refused():
    for e in (0, pp.EPOCH_MAX + 1):
        with pytest.raises(ValueError):
            pp.DaemonSession(epoch=e)


def test_hello_must_carry_sid_zero_codex_b25():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    assert hello(sess, run, sid=0x1234) == 'bad_hello' and sess.sid == 0


def test_hello_retry_keeps_the_session_any_other_boot_opens_a_new_one():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run, boot=0xAAAA)
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_SELECT, 3), run) == 'ran'
    assert hello(sess, run, boot=0xAAAA) == 'hello' and sess.sid == SID1     # retry: same session
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_START, 3), run) == 'ran'      # seq history kept
    hello(sess, run, boot=0xBBBB)                                           # panel rebooted
    assert sess.sid == EPOCH << 16 | 2
    # B25: a panel that picks 0xAAAA again later is not locked out (no list of old boots)
    assert hello(sess, run, boot=0xAAAA) == 'hello' and sess.sid == EPOCH << 16 | 3


def test_one_command_in_flight_rules():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    assert sess.on_cmd(pp.Cmd(SID1, 5, pp.CMD_START, 1), run) == 'bad_seq'   # first must be next_seq 0
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run) == 'ran'
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run) == 'repeated'  # lost STATUS, resent
    assert sess.on_cmd(pp.Cmd(SID1, 4, pp.CMD_SELECT, 2), run) == 'bad_seq'  # skipped seqs: not run
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_result, st.next_seq) == (0, pp.RES_OK, 1)   # refusal changes nothing
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_SELECT, 2), run) == 'ran'
    assert run.ran == [(0, pp.CMD_START, 1), (1, pp.CMD_SELECT, 2)]


def test_refused_replay_cannot_pull_the_seq_back_codex_b24():
    """B24: SELECT10, START11, STOP12, then copies of SELECT10 and START11: START runs once."""
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    for s in range(10):
        sess.on_cmd(pp.Cmd(SID1, s, pp.CMD_SELECT, 0), run)
    out = [sess.on_cmd(pp.Cmd(SID1, s, c, 1), run) for s, c in
           ((10, pp.CMD_SELECT), (11, pp.CMD_START), (12, pp.CMD_STOP), (10, pp.CMD_SELECT), (11, pp.CMD_START))]
    assert out == ['ran', 'ran', 'ran', 'bad_seq', 'bad_seq']
    assert [r[1] for r in run.ran].count(pp.CMD_START) == 1 and sess.next_seq == 13


@pytest.mark.parametrize('delta', [2, 61, 127, 128, 129, 255])
def test_any_jump_other_than_next_is_refused(delta):
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_SELECT, 1), run)                    # next_seq is now 1
    assert sess.on_cmd(pp.Cmd(SID1, delta & 0xFF, pp.CMD_START, 1), run) == 'bad_seq'
    assert len(run.ran) == 1


def test_stop_preempts_a_pending_start_codex_b26():
    """B26: START 0 was lost; the panel gives up on it and sends STOP 1. STOP runs, next_seq moves
    past both, and the abandoned START 0 arriving later is not run."""
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run) == 'ran'
    assert sess.next_seq == 2
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run) == 'bad_seq'
    assert [r[1] for r in run.ran] == [pp.CMD_STOP]


def test_stop_always_runs_but_does_not_pull_the_seq_back():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run)
    assert sess.on_cmd(pp.Cmd(SID1, 40, pp.CMD_STOP, 0), run) == 'ran'      # ahead: runs, next_seq 41
    assert sess.on_cmd(pp.Cmd(SID1, 40, pp.CMD_STOP, 0), run) == 'repeated'
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_STOP, 0), run) == 'ran'       # behind: runs ...
    assert sess.next_seq == 41                                                 # ... but no pull back
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_START, 1), run) == 'bad_seq'
    assert [r[1] for r in run.ran] == [pp.CMD_START, pp.CMD_STOP, pp.CMD_STOP]


def test_failed_stop_can_run_again_codex_b27():
    """B27: START 0 OK, STOP 1 raises before stopping: the resent STOP 1 must run, not get the
    old RES_OK back."""
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), Runner())
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), Runner(fail_on=(pp.CMD_STOP,))) == 'failed'
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_result) == (1, pp.RES_FAILED) and 'action failed' in sess.last_error
    run = Runner()
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run) == 'ran' and run.ran == [(1, pp.CMD_STOP, 0)]
    assert sess.fill(pp.Status()).last_result == pp.RES_OK


def test_failed_start_is_not_run_again():
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), Runner(fail_on=(pp.CMD_START,))) == 'failed'
    run = Runner()
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run) == 'repeated' and run.ran == []
    assert sess.fill(pp.Status()).last_result == pp.RES_FAILED


def test_seq_wraps_around():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    for s in range(254):
        sess.on_cmd(pp.Cmd(SID1, s, pp.CMD_SELECT, 0), run)
    for seq in (254, 255, 0, 1):
        assert sess.on_cmd(pp.Cmd(SID1, seq, pp.CMD_SELECT, seq), run) == 'ran'


def test_result_reported_for_the_acknowledged_command():
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 9), Runner(pp.RES_BAD_PROFILE))
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_result, st.next_seq) == (0, pp.RES_BAD_PROFILE, 1)
    sess.on_cmd(pp.Cmd(SID1, 1, 0x77, 0), Runner())
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


def test_clear_drops_a_profile_taken_but_not_started_codex_b29():
    """B29: PROFILE sid 1 is taken for sending, the UART takes 0 bytes, the session changes:
    it must not go out before the new session's frames."""
    uart = SlowUart([0])
    tx = pp.TxQueue(uart)
    tx.put(pp.encode_profile(SID1, 0, 1, 'OLD'))
    tx.pump()
    tx.clear_profiles()
    tx.put(pp.encode_status(pp.Status(sid=SID1 + 1)))
    tx.put(pp.encode_profile(SID1 + 1, 0, 1, 'NEW'))
    tx.pump()
    got = frames_of(uart)
    assert [type(f).__name__ for f in got] == ['Status', 'Profile'] and got[1].name == 'NEW'


def test_pump_writes_at_most_max_bytes_per_call():
    uart = SlowUart([])
    tx = pp.TxQueue(uart)
    tx.put(pp.encode_status(pp.Status(phase_ds=1)))
    tx.put(pp.encode_profile(1, 0, 1, 'A'))
    assert tx.pump(max_bytes=10) == 10 and len(uart.out) == 10
    assert tx.pump(max_bytes=30) == 30
    assert tx.pump() == pp.LEN_STATUS + pp.LEN_PROFILE - 40 and not tx.pending()
    assert [type(f).__name__ for f in frames_of(uart)] == ['Status', 'Profile']
