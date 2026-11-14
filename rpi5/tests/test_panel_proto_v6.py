"""panel_proto v6: the INFO frame (0xDF) added for panel M2 (Liam 04/11, PANEL session:
"Thêm gói INFO 0xDF, bản 6"). Layout, padding and saturation, stream parsing next to the
older frames, the writer order (STATUS > INFO > PROFILE, newest INFO only), and the panel
keeping the INFO of its own session. No hardware."""
import struct

import pytest

import panel_proto as pp
from test_panel_proto import Bench, SID1, SlowUart, frames_of


def _info(**kw):
    base = dict(sid=SID1, loop_us=27, overruns=0, flags=pp.I_LOOP_VALID, ctrl_src=pp.CTRL_SRC_PROFILE,
                ctrl='LADRC', result=('e_vx 0.005 m/s', 'e_wz 0.20 rad/s', ''), error_text='')
    base.update(kw)
    return pp.Info(**base)


def test_info_layout_and_roundtrip():
    f = pp.encode_info(_info())
    assert len(f) == pp.LEN_INFO == 106
    assert f[0] == pp.START_INFO and f[1] == pp.PROTO_VERSION == 6
    assert struct.unpack_from('<IHHBB', f, 2) == (SID1, 27, 0, pp.I_LOOP_VALID, pp.CTRL_SRC_PROFILE)
    assert f[12:20] == b'LADRC\x00\x00\x00'
    assert f[20:40] == b'e_vx 0.005 m/s' + bytes(6)
    assert f[60:80] == bytes(20) and f[80:82] == bytes(2) and f[82:104] == bytes(22)
    assert struct.unpack_from('<H', f, 104)[0] == pp.crc16(f[:104])
    assert pp.decode(f) == _info()


def test_info_result_run_and_error_text_layout():
    """Codex round 3 R1-R5 and round 4 S1-S6 (Liam 04/11: "(a) Sửa gốc + R3", then "A: run_no
    vào STATUS"): INFO result_run at 80-81 (little-endian), error_text 82-103 (22 bytes); the
    run number itself is in STATUS."""
    f = pp.encode_info(_info(result_run=0xABCD, error_text='E' * 40))
    assert f[80:82] == bytes([0xCD, 0xAB]) and f[82:104] == b'E' * 22
    i = pp.decode(f)
    assert (i.result_run, i.error_text) == (0xABCD, 'E' * 22)


def test_status_run_no_layout_and_roundtrip():
    s = pp.Status(sid=SID1, run_state=pp.ST_END, run_no=0x1234)
    f = pp.encode_status(s)
    assert len(f) == pp.LEN_STATUS == 32 and f[28:30] == bytes([0x34, 0x12])
    assert struct.unpack_from('<H', f, 30)[0] == pp.crc16(f[:30])
    assert pp.decode(f) == s


def test_next_run_no_never_zero_and_never_wraps_codex_t1_t2():
    """Codex round 5 T1, T2 (Liam 05/11: "(a) Chặn quay vòng"): no run number is used twice
    inside one daemon run; after 65535 there is no next number, START must be refused."""
    assert [pp.next_run_no(k) for k in (0, 1, 41, pp.RUN_NO_MAX - 1)] == [1, 2, 42, pp.RUN_NO_MAX]
    for k in (pp.RUN_NO_MAX, pp.RUN_NO_MAX + 1):
        with pytest.raises(OverflowError):
            pp.next_run_no(k)
    with pytest.raises(ValueError):
        pp.next_run_no(-1)


def test_info_text_truncated_and_numbers_saturate():
    f = pp.encode_info(_info(ctrl='LADRC-LONGNAME', result=('x' * 30,), error_text='E' * 40,
                             loop_us=70000, overruns=-5))
    i = pp.decode(f)
    assert i.ctrl == 'LADRC-LO' and i.result == ('x' * 20, '', '') and i.error_text == 'E' * 22
    assert i.loop_us == 0xFFFF and i.overruns == 0


def test_older_frames_keep_their_layout_only_version_changes():
    # CMD and PROFILE keep their v5 layout; STATUS grew by run_no (Liam 04/11: "A: run_no vào STATUS")
    assert (pp.LEN_CMD, pp.LEN_STATUS, pp.LEN_PROFILE) == (13, 32, 28)
    f = pp.encode_status(pp.Status(sid=SID1))
    assert f[1] == 6
    old = bytearray(f)
    old[1] = 5
    struct.pack_into('<H', old, 30, pp.crc16(bytes(old[:30])))
    assert pp.StreamParser().feed(bytes(old)) == []                    # a v5 peer is refused


def test_parser_takes_info_between_other_frames():
    data = pp.encode_status(pp.Status(sid=SID1)) + pp.encode_info(_info()) + \
        pp.encode_profile(SID1, 0, 1, 'A-L2')
    out = pp.StreamParser().feed(data)
    assert [type(x) for x in out] == [pp.Status, pp.Info, pp.Profile]
    assert pp.StreamParser(accept=(pp.START_CMD,)).feed(pp.encode_info(_info())) == []


def test_writer_order_status_info_profile_and_newest_info_only():
    uart = SlowUart([1000] * 20)
    q = pp.TxQueue(uart)
    q.put(pp.encode_profile(SID1, 0, 1, 'A-L2'))
    q.put(pp.encode_info(_info(loop_us=1)))
    q.put(pp.encode_info(_info(loop_us=2)))
    q.put(pp.encode_status(pp.Status(sid=SID1)))
    while q.pending():
        q.pump()
    out = frames_of(uart)
    assert [type(x) for x in out] == [pp.Status, pp.Info, pp.Profile]
    assert out[1].loop_us == 2 and q.dropped_info == 1


def test_status_goes_before_an_info_taken_but_not_started():
    uart = SlowUart([0, 1000, 1000, 1000])
    q = pp.TxQueue(uart)
    q.put(pp.encode_info(_info(loop_us=5)))
    q.pump()                                                           # INFO taken, 0 bytes out
    q.put(pp.encode_status(pp.Status(sid=SID1)))
    while q.pending():
        q.pump()
    out = frames_of(uart)
    assert [type(x) for x in out] == [pp.Status, pp.Info] and out[1].loop_us == 5


def test_started_info_finishes_before_a_newer_one():
    uart = SlowUart([])
    q = pp.TxQueue(uart)
    q.put(pp.encode_info(_info(loop_us=1)))
    q.pump(max_bytes=10)                                               # 10 bytes of it out (P16)
    assert q._sent == 10 and q._current[0] == pp.START_INFO
    q.put(pp.encode_info(_info(loop_us=2)))
    q.put(pp.encode_status(pp.Status(sid=SID1)))
    while q.pending():
        q.pump()
    out = frames_of(uart)
    assert [type(x) for x in out] == [pp.Info, pp.Status, pp.Info] and [out[0].loop_us, out[2].loop_us] == [1, 2]


def test_info_goes_before_a_profile_taken_but_not_started_codex_p12():
    uart = SlowUart([0, 1000, 1000, 1000])
    q = pp.TxQueue(uart)
    q.put(pp.encode_profile(SID1, 0, 1, 'A-L2'))
    q.pump()                                                           # PROFILE taken, 0 bytes out
    q.put(pp.encode_info(_info(loop_us=7)))
    while q.pending():
        q.pump()
    assert [type(x) for x in frames_of(uart)] == [pp.Info, pp.Profile]


def test_started_profile_finishes_before_info():
    uart = SlowUart([])
    q = pp.TxQueue(uart)
    q.put(pp.encode_profile(SID1, 0, 1, 'A-L2'))
    q.pump(max_bytes=5)
    q.put(pp.encode_info(_info()))
    while q.pending():
        q.pump()
    assert [type(x) for x in frames_of(uart)] == [pp.Profile, pp.Info]


def test_clear_session_drops_old_info_and_profiles_not_started_codex_p13():
    uart = SlowUart([0] + [1000] * 10)
    q = pp.TxQueue(uart)
    q.put(pp.encode_info(_info(loop_us=1)))
    q.pump()                                                           # INFO taken, 0 bytes out
    q.put(pp.encode_profile(SID1, 0, 1, 'A-L2'))
    q.clear_session()
    q.put(pp.encode_status(pp.Status(sid=SID1 + 1)))
    while q.pending():
        q.pump()
    assert [type(x) for x in frames_of(uart)] == [pp.Status] and q.dropped_info == 1
    uart = SlowUart([])
    q = pp.TxQueue(uart)
    q.put(pp.encode_info(_info(loop_us=2)))
    q.pump(max_bytes=3)                                                # started: must finish
    q.put(pp.encode_info(_info(loop_us=3)))                            # waiting: dropped
    q.clear_session()
    while q.pending():
        q.pump()
    out = frames_of(uart)
    assert [x.loop_us for x in out] == [2]


def test_panel_keeps_info_of_its_session_and_forgets_it_when_lost():
    b = Bench()
    b.step(10)
    b.panel.on_info(pp.decode(pp.encode_info(_info(sid=SID1 + 1))))
    assert b.panel.info is None                                        # another sid
    b.panel.on_info(pp.decode(pp.encode_info(_info())))
    assert b.panel.info.ctrl == 'LADRC'
    b.sess = pp.DaemonSession(8)                                       # daemon restart: new epoch
    b.step(10)
    assert b.panel.info is None


def test_stop_without_link_after_a_given_up_start_goes_at_stale_plus_one():
    """Docstring rule: 'with a stale seq STOP goes out at stale + 1'. Reached when a START is
    given up (CMD_MAX_AGE_MS) while no STATUS arrives, and STOP is pressed before the link is
    back (no SYNC can go out without link)."""
    b = Bench()
    b.step(10)
    t0 = b.t
    b.drop_down = lambda f: b.t < t0 + 3500                           # no STATUS for 3.5 s
    b.drop_up = lambda f: f[8] == pp.CMD_START                         # every START copy lost
    b.press(pp.CMD_START, 1)
    seq = b.panel.pending['seq']
    b.step(310)                                                        # > CMD_MAX_AGE_MS
    assert b.panel.pending is None and b.panel.stale == seq and not b.panel.link(b.t)
    b.press(pp.CMD_STOP)
    stops = [pp.decode(f) for f in b.up if f[8] == pp.CMD_STOP]
    assert stops and stops[-1].seq == seq + 1
    b.step(100)
    assert b.panel.stop_state == 'confirmed' and b.panel.stale is None
    assert [r[1] for r in b.ran] == [pp.CMD_STOP]
