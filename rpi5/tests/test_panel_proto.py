"""panel_proto v4: frame layouts, CRC16, stream resync, epoch lock, sessions and 16-bit seqs
(#44, B1, B15, B16, B23-B27, B52-B54, N12-N14), data age field (#45), non-blocking writer
(B18, B29, B58/N16), and the panel state machine against the daemon over lossy links
(B55, N14, N15) (workflow v3.1 section 9.6; Liam approved B0 29/10; B0 review rounds 1-3).
The counter-examples are Codex's (codex_tra_loi_vong2.md B23-B29; codex_tra_loi_vong3.md
B52-B58; codex_tra_loi_vong3b.md N12-N16); expected outcomes follow the rules written in the
panel_proto docstrings. No hardware."""
import struct
import threading
import time

import pytest

import panel_proto as pp

EPOCH = 7
SID1 = EPOCH << 16 | 1          # first session of epoch 7


def test_crc16_ccitt_false_check_value():
    assert pp.crc16(b'123456789') == 0x29B1          # published check value of CRC-16/CCITT-FALSE


def test_cmd_layout_and_roundtrip():
    f = pp.encode_cmd(sid=0xDEADBEEF, seq=0x1234, cmd=pp.CMD_START, arg=0x0102)
    assert len(f) == pp.LEN_CMD == 13
    assert f[:11] == bytes([0xCC, pp.PROTO_VERSION, 0xEF, 0xBE, 0xAD, 0xDE, 0x34, 0x12, pp.CMD_START, 0x02, 0x01])
    assert struct.unpack_from('<H', f, 11)[0] == pp.crc16(f[:11])
    assert pp.decode(f) == pp.Cmd(0xDEADBEEF, 0x1234, pp.CMD_START, 0x0102)


def test_status_layout_and_roundtrip():
    s = pp.Status(sid=0x00071234, daemon_boot=0xBEEF, panel_boot=0x4321, last_cmd_seq=0x0309,
                  last_result=pp.RES_OK, next_seq=0x030A, run_state=pp.ST_RUN, phase_ds=123, profile=4,
                  outcome=pp.OUT_COMPLETED, quality=pp.Q_LOST | pp.Q_SPI_STALE, lost=17, data_age_ms=250,
                  links=pp.L_RPI | pp.L_H7, error_code=7, last_cmd=pp.CMD_START)
    f = pp.encode_status(s)
    assert len(f) == pp.LEN_STATUS == 30 and f[0] == 0xDD and f[27] == pp.CMD_START
    assert f[2:6] == b'\x34\x12\x07\x00' and f[6:8] == b'\xef\xbe' and f[8:10] == b'\x21\x43'
    assert f[10:12] == b'\x09\x03' and f[12] == pp.RES_OK and f[13:15] == b'\x0a\x03' and f[15] == pp.ST_RUN
    assert f[16:18] == (123).to_bytes(2, 'little') and (f[18], f[19], f[20]) == (4, pp.OUT_COMPLETED, 5)
    assert f[21:23] == (17).to_bytes(2, 'little') and f[23:25] == (250).to_bytes(2, 'little')
    assert f[25] == pp.L_RPI | pp.L_H7 and f[26] == 7
    assert struct.unpack_from('<H', f, 28)[0] == pp.crc16(f[:28])
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
    bad[8] ^= 0x40
    p = pp.StreamParser(accept=(pp.START_CMD,))
    got = p.feed(b'\x00\x55' + a + bytes(bad) + b[:3])
    got += p.feed(b[3:])
    assert got == [pp.decode(a), pp.decode(b)]
    assert p.bad_frames >= 1 and p.skipped_bytes >= 2


def test_start_byte_inside_payload_does_not_break_a_whole_frame():
    f = pp.encode_cmd(sid=0xCCCCCCCC, seq=0xCCCC, cmd=pp.CMD_STOP, arg=0xCCCC)
    assert pp.StreamParser().feed(b'\xcc' + f) == [pp.decode(f)]


def test_xor_blind_two_bit_error_is_caught():
    """B17: STOP 0x21 -> START 0x20 together with bit 0 of seq flipped kept the old XOR valid."""
    f = bytearray(pp.encode_cmd(5, 4, pp.CMD_STOP))
    f[8] ^= 0x01
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
    assert pp.next_daemon_epoch(path, create=True) == 1
    assert [pp.next_daemon_epoch(path) for _ in range(2)] == [2, 3]
    assert (tmp_path / 'panel_epoch').read_text(encoding='ascii').strip() == '3'
    assert not (tmp_path / 'panel_epoch.tmp').exists()


def test_missing_epoch_file_is_an_error_unless_first_start_codex_n13(tmp_path):
    """N13: a lost file must not silently restart at 1 (sids of before would come back)."""
    with pytest.raises(FileNotFoundError):
        pp.next_daemon_epoch(str(tmp_path / 'panel_epoch'))
    assert not (tmp_path / 'panel_epoch').exists()


@pytest.mark.parametrize('content', ['', 'x', '-1', '65535', '3.0'])
def test_unreadable_epoch_file_is_refused(tmp_path, content):
    p = tmp_path / 'panel_epoch'
    p.write_text(content, encoding='ascii')
    with pytest.raises(ValueError):
        pp.next_daemon_epoch(str(p))
    assert p.read_text(encoding='ascii') == content                    # left for the operator


def test_two_interleaved_starts_get_different_epochs_codex_n13(tmp_path):
    """Codex N13 (B52): A reads 7; B starts while A holds the value. B must wait for A's write,
    so the two starts get 8 and 9, not 8 twice."""
    path = str(tmp_path / 'panel_epoch')
    (tmp_path / 'panel_epoch').write_text('7\n', encoding='ascii')
    other = []
    state = {}

    def while_a_holds(value):
        t = threading.Thread(target=lambda: other.append(pp.next_daemon_epoch(path)))
        t.start()
        time.sleep(0.3)
        state['blocked'] = t.is_alive()                 # B is waiting on the lock
        state['thread'] = t

    a = pp.next_daemon_epoch(path, _after_read=while_a_holds)
    state['thread'].join(10)
    assert state['blocked'] and sorted([a] + other) == [8, 9]


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
    START is dropped before and after the panel's new HELLO."""
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


@pytest.mark.parametrize('seq', [0, 3, 62, 128, 129, 130, 256, 257, 0x8001, pp.SEQ_END])
def test_any_seq_other_than_next_is_refused(seq):
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_SELECT, 1), run)                    # next_seq is now 1
    assert sess.on_cmd(pp.Cmd(SID1, seq, pp.CMD_START, 1), run) == 'bad_seq'
    assert len(run.ran) == 1


def test_old_start_after_256_commands_does_not_run_again_codex_n12():
    """Codex N12 (B53) (a): START0 -> STOP1 -> SELECT2..255 -> copy of START0. With an 8-bit seq
    the copy ran again; with 16 bits nothing wraps and the copy is refused."""
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 0), run)
    sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run)
    for s in range(2, 256):
        sess.on_cmd(pp.Cmd(SID1, s, pp.CMD_SELECT, 0), run)
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 0), run) == 'bad_seq'
    assert sess.on_cmd(pp.Cmd(SID1, 256, pp.CMD_SELECT, 0), run) == 'ran'
    assert [r[1] for r in run.ran].count(pp.CMD_START) == 1


def test_old_stop_half_a_turn_back_neither_runs_nor_pulls_back_codex_n12():
    """Codex N12 (b): SELECT0, STOP1, START2, STOP3, SELECT4..129, then copies of STOP1 and
    START2. The old STOP pulled next_seq back to 2 and START2 ran again; now neither copy runs
    (a STOP is taken only at next_seq or next_seq + 1)."""
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    for s, c in ((0, pp.CMD_SELECT), (1, pp.CMD_STOP), (2, pp.CMD_START), (3, pp.CMD_STOP)):
        sess.on_cmd(pp.Cmd(SID1, s, c, 0), run)
    for s in range(4, 130):
        sess.on_cmd(pp.Cmd(SID1, s, pp.CMD_SELECT, 0), run)
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run) == 'bad_seq'
    assert sess.on_cmd(pp.Cmd(SID1, 2, pp.CMD_START, 0), run) == 'bad_seq'
    assert sess.next_seq == 130
    cmds = [r[1] for r in run.ran]
    assert cmds.count(pp.CMD_START) == 1 and cmds.count(pp.CMD_STOP) == 2


def test_stop_preempts_a_pending_start_codex_b26():
    """B26: START 0 was lost; the panel gives up on it and sends STOP 1. STOP runs, next_seq moves
    past both, and the abandoned START 0 arriving later is not run."""
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run) == 'ran'
    assert sess.next_seq == 2
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run) == 'bad_seq'
    assert [r[1] for r in run.ran] == [pp.CMD_STOP]


def test_stop_window_is_next_seq_or_one_ahead():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run)                       # next_seq 1
    assert sess.on_cmd(pp.Cmd(SID1, 3, pp.CMD_STOP, 0), run) == 'bad_seq'    # two ahead
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run) == 'ran'        # at next_seq
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run) == 'repeated'
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_STOP, 0), run) == 'bad_seq'    # behind
    assert sess.on_cmd(pp.Cmd(SID1, 3, pp.CMD_STOP, 0), run) == 'ran'        # one ahead of 2
    assert sess.next_seq == 4
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


def test_stop_returning_failed_can_run_again_codex_b54():
    """Codex B54: run() returning RES_FAILED (no exception) is the same failure: the resent STOP
    runs again instead of getting REPEATED."""
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), Runner())
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), Runner(result=pp.RES_FAILED)) == 'failed'
    run = Runner()
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), run) == 'ran' and len(run.ran) == 1


def test_resend_of_an_older_command_does_not_report_a_result_it_does_not_have_codex_n14():
    """Codex N14 / B54: START0 OK, STOP1 raises, then START0 is resent. The old code answered
    'repeated' while STATUS showed STOP1 FAILED. Now START0 is an old seq: refused, and STATUS
    keeps STOP1's own result."""
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 0), Runner())
    sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_STOP, 0), Runner(fail_on=(pp.CMD_STOP,)))
    run = Runner()
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 0), run) == 'bad_seq' and run.ran == []
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_result) == (1, pp.RES_FAILED)


def test_failed_start_is_not_run_again():
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), Runner(fail_on=(pp.CMD_START,))) == 'failed'
    run = Runner()
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run) == 'repeated' and run.ran == []
    assert sess.fill(pp.Status()).last_result == pp.RES_FAILED


def test_session_seqs_used_up_need_a_new_hello():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    sess.next_seq = pp.SEQ_END - 2
    assert sess.on_cmd(pp.Cmd(SID1, pp.SEQ_END - 2, pp.CMD_SELECT, 0), run) == 'ran'
    assert sess.on_cmd(pp.Cmd(SID1, pp.SEQ_END - 1, pp.CMD_START, 0), run) == 'bad_seq'   # kept for STOP (V2)
    assert sess.on_cmd(pp.Cmd(SID1, pp.SEQ_END - 1, pp.CMD_STOP, 0), run) == 'ran'
    assert sess.fill(pp.Status()).next_seq == pp.SEQ_END
    assert sess.on_cmd(pp.Cmd(SID1, pp.SEQ_END, pp.CMD_STOP, 0), run) == 'bad_seq'
    hello(sess, run, boot=0x5151)
    assert sess.next_seq == 0 and sess.sid == EPOCH << 16 | 2


def test_status_carries_the_code_of_the_last_command_codex_v1():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    assert sess.fill(pp.Status()).last_cmd == 0
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run)
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_cmd, st.last_result) == (0, pp.CMD_START, pp.RES_OK)


def test_sync_skips_a_seq_and_does_nothing_else():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    assert sess.on_cmd(pp.Cmd(SID1, 1, pp.CMD_SYNC, 0), run) == 'ran'       # one ahead: skips 0
    assert sess.next_seq == 2 and run.ran == []
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), run) == 'bad_seq'  # the skipped START
    assert sess.on_cmd(pp.Cmd(SID1, 4, pp.CMD_SYNC, 0), run) == 'bad_seq'   # two ahead
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_cmd, st.last_result) == (1, pp.CMD_SYNC, pp.RES_OK)


@pytest.mark.parametrize('result', [pp.RES_BUSY, pp.RES_UNKNOWN, pp.RES_BAD_PROFILE, pp.RES_FAILED])
def test_stop_not_done_runs_again_on_resend_codex_v4(result):
    """Codex V4: STOP answered RES_BUSY (any result but OK / NOT_IDLE) was stored as done; the
    panel kept resending and the callback never ran again."""
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_STOP, 0), Runner(result=result)) in ('ran', 'failed')
    run = Runner()
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_STOP, 0), run) == 'ran' and len(run.ran) == 1
    assert sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_STOP, 0), run) == 'repeated'   # OK now: stored


def test_profiles_request_changes_nothing():
    sess, run = pp.DaemonSession(epoch=EPOCH), Runner()
    hello(sess, run)
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_SELECT, 1), run)
    assert sess.on_cmd(pp.Cmd(SID1, 77, pp.CMD_PROFILES, 0), run) == 'profiles'
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_result, st.next_seq) == (0, pp.RES_OK, 1) and len(run.ran) == 1


def test_result_reported_for_the_acknowledged_command():
    sess = pp.DaemonSession(epoch=EPOCH)
    hello(sess, Runner())
    sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 9), Runner(pp.RES_BAD_PROFILE))
    st = sess.fill(pp.Status())
    assert (st.last_cmd_seq, st.last_result, st.next_seq) == (0, pp.RES_BAD_PROFILE, 1)
    sess.on_cmd(pp.Cmd(SID1, 1, 0x77, 0), Runner())
    assert sess.fill(pp.Status()).last_result == pp.RES_UNKNOWN


# ── panel state machine against the daemon (two ends) ─────────────────────

class Bench:
    """DaemonSession with a run model and PanelClient over two in-order links that drop what
    drop_up / drop_down say; 10 ms steps; the daemon sends STATUS every 100 ms, after every
    command, and the PROFILE list after a new session and on PROFILES."""

    def __init__(self, nonces=(0x1111, 0x2222, 0x3333, 0x4444, 0x5555), profiles=('A-L2', 'A-F2'),
                 stop_ms=50, epoch=EPOCH):
        self.sess = pp.DaemonSession(epoch)
        it = iter(nonces)
        self.panel = pp.PanelClient(lambda: next(it))
        self.t = 0
        self.run_state = pp.ST_IDLE
        self.t_stop = None
        self.stop_ms = stop_ms
        self.fail_stop = 0
        self.ran = []
        self.log = []
        self.profiles = profiles
        self.up, self.down = [], []
        self.drop_up = lambda f: False
        self.drop_down = lambda f: False

    def run(self, c):
        self.ran.append((c.seq, c.cmd, c.arg))
        if c.cmd == pp.CMD_START:
            if self.run_state in pp.ST_ACTIVE:
                return pp.RES_BUSY
            self.run_state = pp.ST_RUN
            return pp.RES_OK
        if c.cmd == pp.CMD_STOP:
            if self.fail_stop:
                self.fail_stop -= 1
                return pp.RES_FAILED
            if self.run_state in pp.ST_ACTIVE:
                self.run_state, self.t_stop = pp.ST_STOPPING, self.t
                return pp.RES_OK
            return pp.RES_NOT_IDLE
        return pp.RES_OK

    def status(self):
        return pp.encode_status(self.sess.fill(pp.Status(run_state=self.run_state)))

    def profile_frames(self):
        return [pp.encode_profile(self.sess.sid, i, len(self.profiles), n) for i, n in enumerate(self.profiles)]

    def step(self, n=1):
        for _ in range(n):
            self.t += 10
            if self.run_state == pp.ST_STOPPING and self.t - self.t_stop >= self.stop_ms:
                self.run_state = pp.ST_END
            up, self.up = self.up, []
            for f in up:
                if self.drop_up(f):
                    continue
                for c in pp.StreamParser(accept=(pp.START_CMD,)).feed(f):
                    r = self.sess.on_cmd(c, self.run)
                    self.log.append((c.seq, c.cmd, r))
                    self.down.append(self.status())
                    if r in ('hello', 'profiles'):
                        self.down += self.profile_frames()
            if self.t % 100 == 0:
                self.down.append(self.status())
            down, self.down = self.down, []
            for f in down:
                if self.drop_down(f):
                    continue
                for x in pp.StreamParser().feed(f):
                    if isinstance(x, pp.Status):
                        self.up += self.panel.on_status(x, self.t)
                    else:
                        self.panel.on_profile(x)
            self.up += self.panel.tick(self.t)

    def press(self, cmd, arg=0):
        self.up += self.panel.press(cmd, arg, self.t)


class HoldBench(Bench):
    """Bench whose panel -> daemon link can hold frames in order (a FIFO that stalls, e.g. a
    busy daemon) instead of dropping them; set hold = False to let them through."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.hold = False
        self.held = []

    def step(self, n=1):
        for _ in range(n):
            if self.hold:
                self.held += self.up
                self.up = []
            elif self.held:
                self.up = self.held + self.up
                self.held = []
            super().step(1)


def _is(start, cmd=None):
    return lambda f: f[0] == start and (cmd is None or f[8] == cmd)


def test_panel_handshake_select_start_and_profiles():
    b = Bench()
    b.step(10)
    assert b.panel.sid == SID1 and b.panel.link(b.t) and b.panel.profiles == {0: 'A-L2', 1: 'A-F2'}
    b.press(pp.CMD_SELECT, 1)
    b.step(5)
    b.press(pp.CMD_START, 1)
    b.step(5)
    assert b.ran == [(0, pp.CMD_SELECT, 1), (1, pp.CMD_START, 1)]
    assert b.panel.shown == [('done', pp.CMD_SELECT, pp.RES_OK), ('done', pp.CMD_START, pp.RES_OK)]


def test_panel_resends_until_ack_and_start_runs_once():
    b = Bench()
    b.step(10)
    b.press(pp.CMD_START, 0)
    t_lost = b.t
    b.drop_down = lambda f: f[0] == pp.START_STATUS and b.t < t_lost + 500     # ACKs lost for 0.5 s
    b.step(80)
    assert [r[1] for r in b.ran].count(pp.CMD_START) == 1
    assert any(r == 'repeated' for _, _, r in b.log)
    assert b.panel.shown == [('done', pp.CMD_START, pp.RES_OK)]


def test_lost_start_then_stop_shows_unverified_and_never_runs_the_start():
    b = Bench()
    b.step(10)
    b.drop_up = _is(pp.START_CMD, pp.CMD_START)                                 # every START lost
    b.press(pp.CMD_START, 1)
    b.step(3)
    b.press(pp.CMD_STOP)
    b.step(30)
    assert [r[1] for r in b.ran] == [pp.CMD_STOP]
    assert ('unverified', pp.CMD_START, None) in b.panel.shown
    assert b.panel.stop_state == 'confirmed'                                  # NOT_IDLE + idle
    assert b.sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 1), b.run) == 'bad_seq'   # late copy refused


def test_failed_stop_is_retried_and_confirmed_only_when_the_run_ends_codex_n14():
    b = Bench(stop_ms=300)
    b.step(10)
    b.press(pp.CMD_START, 0)
    b.step(10)
    assert b.run_state == pp.ST_RUN
    b.fail_stop = 2
    b.press(pp.CMD_STOP)
    seen = set()
    for _ in range(150):
        b.step()
        seen.add((b.panel.stop_state, b.run_state))
    stops = [r for r in b.ran if r[1] == pp.CMD_STOP]
    assert len(stops) == 3 and len({s[0] for s in stops}) == 1                 # same frame, run 3 times
    assert b.panel.stop_failures >= 1
    assert ('acked', pp.ST_STOPPING) in seen                                   # ACK OK is not stopped
    assert not any(st == 'confirmed' and rs in pp.ST_ACTIVE for st, rs in seen)
    assert b.run_state == pp.ST_END and b.panel.stop_state == 'confirmed'


def test_stop_keeps_going_without_link():
    """STOP is resent while no STATUS comes back (NO LINK); it runs once the link returns."""
    b = Bench()
    b.step(10)
    b.press(pp.CMD_START, 0)
    b.step(10)
    t0 = b.t
    b.drop_down = lambda f: b.t < t0 + 1_000
    b.press(pp.CMD_STOP)
    b.step(60)
    assert not b.panel.link(b.t)
    b.step(60)
    assert b.panel.stop_state == 'confirmed' and b.run_state == pp.ST_END


def test_daemon_restart_mid_command_codex_n15():
    """The daemon restarts (new epoch) while a SELECT waits for its ACK: the panel sees
    daemon_boot change, shows the SELECT unverified, says HELLO with a new nonce; the old
    SELECT frame is refused by the new daemon."""
    b = Bench()
    b.step(10)
    b.drop_up = _is(pp.START_CMD, pp.CMD_SELECT)
    b.press(pp.CMD_SELECT, 1)
    b.step(2)
    old = pp.Cmd(SID1, 0, pp.CMD_SELECT, 1)
    b.sess = pp.DaemonSession(EPOCH + 1)
    b.drop_up = lambda f: False
    b.step(30)
    assert ('unverified', pp.CMD_SELECT, None) in b.panel.shown
    assert b.panel.sid == (EPOCH + 1) << 16 | 1 and b.panel.boot == 0x2222
    assert b.sess.on_cmd(old, b.run) == 'dropped_sid'


def test_panel_reboot_with_the_same_nonce_continues_safely_codex_n15():
    """Nonce collision: the panel reboots and draws the nonce of its old session. The daemon
    takes it as a retry; the panel uses next_seq from STATUS, so its first command runs."""
    b = Bench()
    b.step(10)
    b.press(pp.CMD_SELECT, 1)
    b.step(10)
    b.panel = pp.PanelClient(lambda: 0x1111)
    b.step(10)
    assert b.panel.sid == SID1 and b.panel.next_seq == 1
    b.press(pp.CMD_START, 1)
    b.step(10)
    assert b.ran[-1] == (1, pp.CMD_START, 1) and b.panel.shown == [('done', pp.CMD_START, pp.RES_OK)]


def test_foreign_hello_moves_the_session_and_stop_still_runs_codex_n15():
    """An old HELLO (another nonce) reaches the daemon (outside the FIFO assumption): a new
    session opens. The panel sees a sid it did not ask for, says HELLO again, and the STOP it
    was sending goes out again in its new session."""
    b = Bench()
    b.step(10)
    b.press(pp.CMD_START, 0)
    b.step(10)
    b.sess.on_cmd(pp.Cmd(0, 0, pp.CMD_HELLO, 0x9999), b.run)
    b.press(pp.CMD_STOP)
    b.step(60)
    assert b.panel.sid == EPOCH << 16 | 3 and b.panel.boot == 0x2222
    assert any(r[1] == pp.CMD_STOP for r in b.ran)
    assert b.panel.stop_state == 'confirmed' and b.run_state == pp.ST_END


def test_used_up_session_reconnects_after_its_last_ack():
    b = Bench()
    b.step(10)
    b.sess.next_seq = pp.SEQ_END - 2
    b.step(10)
    b.press(pp.CMD_SELECT, 0)
    b.step(20)
    assert b.panel.shown == [('done', pp.CMD_SELECT, pp.RES_OK)]
    assert b.panel.sid == EPOCH << 16 | 2 and b.panel.next_seq == 0


def test_stop_after_a_start_held_past_the_timeout_codex_v1():
    """Codex V1: START0 is held in the FIFO for 3.2 s (the panel gives it up), then STOP. The
    old panel reused seq 0 for STOP, START0 ran, STOP0 was refused and START0's ACK was taken
    as STOP's. Now STOP goes at 1 (seq 0 stays stale), ACKs match seq and code: the STOP runs
    after the START and the run ends."""
    b = HoldBench()
    b.step(10)
    b.hold = True
    b.press(pp.CMD_START, 0)
    b.step(320)
    assert ('unverified', pp.CMD_START, None) in b.panel.shown and b.panel.stale == 0
    b.press(pp.CMD_STOP)
    b.step(5)
    b.hold = False
    b.step(80)
    assert [r[1] for r in b.ran] == [pp.CMD_START, pp.CMD_STOP]
    assert b.run_state == pp.ST_END and b.panel.stop_state == 'confirmed' and b.panel.stale is None


def test_stop_before_the_held_start_arrives_skips_it_codex_v1():
    """Same, but START0 is lost for good: once given up, the panel skips its seq with SYNC1,
    then STOP goes at 2, so a late copy of START0 can never run."""
    b = HoldBench()
    b.step(10)
    b.drop_up = _is(pp.START_CMD, pp.CMD_START)
    b.press(pp.CMD_START, 0)
    b.step(320)
    b.drop_up = lambda f: False
    b.press(pp.CMD_STOP)
    b.step(30)
    assert [r[1] for r in b.ran] == [pp.CMD_STOP] and b.sess.next_seq == 3
    assert [(s, c) for s, c, r in b.log if r == 'ran'] == [(1, pp.CMD_SYNC), (2, pp.CMD_STOP)]
    assert b.sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_START, 0), b.run) == 'bad_seq'
    assert b.panel.stop_state == 'confirmed'


def test_given_up_select_is_skipped_with_sync_before_the_next_command():
    """A SELECT given up after CMD_MAX_AGE_MS leaves its seq stale: SELECT/START are refused
    until SYNC (at stale + 1) is acknowledged; then the next START runs at a fresh seq."""
    b = Bench()
    b.step(10)
    b.drop_up = _is(pp.START_CMD, pp.CMD_SELECT)
    b.press(pp.CMD_SELECT, 1)
    b.step(310)
    b.drop_up = lambda f: False
    assert b.panel.press(pp.CMD_START, 1, b.t) == [] or b.panel.stale is None
    b.step(20)
    assert b.panel.stale is None and b.sess.next_seq == 2
    b.press(pp.CMD_START, 1)
    b.step(10)
    assert b.ran == [(2, pp.CMD_START, 1)] and b.panel.shown[-1] == ('done', pp.CMD_START, pp.RES_OK)
    assert b.sess.on_cmd(pp.Cmd(SID1, 0, pp.CMD_SELECT, 1), b.run) == 'bad_seq'


def test_stop_with_a_lost_start_at_the_last_seq_codex_v2():
    """Codex V2: at the end of the seqs a lost START left STOP refused forever. Now the last
    seq is kept for STOP: STOP runs, then the panel opens a new session."""
    b = Bench()
    b.step(10)
    b.sess.next_seq = pp.SEQ_END - 2
    b.step(10)
    b.drop_up = _is(pp.START_CMD, pp.CMD_START)
    b.press(pp.CMD_START, 0)
    b.step(3)
    b.drop_up = lambda f: False
    b.press(pp.CMD_STOP)
    b.step(60)
    assert [r for r in b.ran if r[1] == pp.CMD_STOP] == [(pp.SEQ_END - 1, pp.CMD_STOP, 0)]
    assert b.panel.sid == EPOCH << 16 | 2 and b.panel.stop is None


def test_panel_reboot_with_same_nonce_and_a_start_in_the_fifo_codex_v3():
    """Codex V3: START0 still in the FIFO, the panel reboots and draws the same nonce, takes a
    periodic STATUS as its handshake and sends STOP0. START0 runs first, STOP0 is refused; its
    ACK is no longer taken for STOP's, the panel re-sends STOP at the new next_seq: it runs."""
    b = HoldBench()
    b.step(10)
    b.hold = True
    b.press(pp.CMD_START, 0)
    b.step(1)
    b.panel = pp.PanelClient(lambda: 0x1111)
    b.step(15)
    b.press(pp.CMD_STOP)
    b.step(2)
    b.hold = False
    b.step(80)
    assert [r[1] for r in b.ran] == [pp.CMD_START, pp.CMD_STOP]
    assert b.run_state == pp.ST_END and b.panel.stop_state == 'confirmed'


def test_stop_answered_busy_then_ok_codex_v4():
    b = Bench()
    b.step(10)
    b.press(pp.CMD_START, 0)
    b.step(10)
    real = b.run
    answers = [pp.RES_BUSY]

    def run(c):
        if c.cmd == pp.CMD_STOP and answers:
            b.ran.append((c.seq, c.cmd, c.arg))
            return answers.pop()
        return real(c)
    b.run = run
    b.press(pp.CMD_STOP)
    b.step(60)
    assert len([r for r in b.ran if r[1] == pp.CMD_STOP]) == 2
    assert b.panel.stop_state == 'confirmed' and b.run_state == pp.ST_END


def test_confirmed_stop_does_not_carry_into_the_next_run_codex_v5():
    b = Bench()
    b.step(10)
    b.press(pp.CMD_START, 0)
    b.step(10)
    b.press(pp.CMD_STOP)
    b.step(30)
    assert b.panel.stop_state == 'confirmed'
    b.run_state = pp.ST_IDLE
    b.press(pp.CMD_START, 0)
    assert b.panel.stop_state == 'none'
    b.step(10)
    assert b.run_state == pp.ST_RUN and b.panel.stop_state == 'none'


def test_confirmed_stop_cleared_when_a_run_becomes_active():
    """V5: a run started by other means (another panel session, hil_ctl) clears 'confirmed'."""
    b = Bench()
    b.step(10)
    b.press(pp.CMD_STOP)
    b.step(30)
    assert b.panel.stop_state == 'confirmed'
    b.run_state = pp.ST_RUN
    b.step(15)
    assert b.panel.stop_state == 'none'


def test_missing_profile_is_asked_again():
    b = Bench()
    dropped = []

    def drop_one(f):
        if f[0] == pp.START_PROFILE and f[6] == 1 and not dropped:
            dropped.append(f)
            return True
        return False
    b.drop_down = drop_one
    b.step(10)
    assert b.panel.profiles == {0: 'A-L2'}
    b.step(150)
    assert b.panel.profiles == {0: 'A-L2', 1: 'A-F2'}
    assert any(c == pp.CMD_PROFILES for _, c, _ in b.log)


def test_commands_refused_while_busy_or_without_link():
    b = Bench()
    assert b.panel.press(pp.CMD_START, 0, b.t) == [] and b.panel.shown == [('refused', pp.CMD_START, None)]
    b.step(10)
    b.drop_up = lambda f: True
    b.press(pp.CMD_SELECT, 0)
    assert b.panel.press(pp.CMD_START, 0, b.t) == []                         # one in flight
    b.step(400)
    assert ('unverified', pp.CMD_SELECT, None) in b.panel.shown              # CMD_MAX_AGE_MS


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


def test_status_goes_before_a_profile_taken_but_not_started_codex_n16():
    """Codex N16 (B58): PROFILE taken, the UART takes 0 bytes, then a STATUS: STATUS first."""
    uart = SlowUart([0])
    tx = pp.TxQueue(uart)
    tx.put(pp.encode_profile(1, 0, 1, 'P'))
    tx.pump()
    tx.put(pp.encode_status(pp.Status(phase_ds=1)))
    tx.pump()
    assert [type(f).__name__ for f in frames_of(uart)] == ['Status', 'Profile']
    assert tx.dropped_profiles == 0


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
