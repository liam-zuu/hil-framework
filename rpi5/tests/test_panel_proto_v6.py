"""panel_proto v6: the INFO frame (0xDF) added for panel M2 (Liam 04/11, PANEL session:
"Thêm gói INFO 0xDF, bản 6"). Layout, padding and saturation, stream parsing next to the
older frames, the writer order (STATUS > INFO > PROFILE, newest INFO only), and the panel
keeping the INFO of its own session. No hardware."""
import struct

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
    assert f[60:80] == bytes(20) and f[80:104] == bytes(24)
    assert struct.unpack_from('<H', f, 104)[0] == pp.crc16(f[:104])
    assert pp.decode(f) == _info()


def test_info_text_truncated_and_numbers_saturate():
    f = pp.encode_info(_info(ctrl='LADRC-LONGNAME', result=('x' * 30,), error_text='E' * 40,
                             loop_us=70000, overruns=-5))
    i = pp.decode(f)
    assert i.ctrl == 'LADRC-LO' and i.result == ('x' * 20, '', '') and i.error_text == 'E' * 24
    assert i.loop_us == 0xFFFF and i.overruns == 0


def test_older_frames_keep_their_layout_only_version_changes():
    assert (pp.LEN_CMD, pp.LEN_STATUS, pp.LEN_PROFILE) == (13, 30, 28)
    f = pp.encode_status(pp.Status(sid=SID1))
    assert f[1] == 6
    old = bytearray(f)
    old[1] = 5
    struct.pack_into('<H', old, 28, pp.crc16(bytes(old[:28])))
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
    uart = SlowUart([10, 1000, 1000])
    q = pp.TxQueue(uart)
    q.put(pp.encode_info(_info(loop_us=1)))
    q.pump()                                                           # 10 bytes of it out
    q.put(pp.encode_info(_info(loop_us=2)))
    while q.pending():
        q.pump()
    assert [x.loop_us for x in frames_of(uart)] == [1, 2]


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
