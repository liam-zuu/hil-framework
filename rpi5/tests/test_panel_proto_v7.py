"""Panel UART v7 wire layouts, parser diagnostics and session-scoped demo data."""
import binascii
import struct
from dataclasses import replace

import pytest

import panel_proto as pp
from test_panel_proto import Bench, SID1


def seal(payload):
    return bytes(payload) + struct.pack('<H', binascii.crc_hqx(payload, 0xffff))


def test_extra_wire_layout_and_text_fields():
    extra = pp.Extra(SID1, pp.X_FAULT_ACTIVE | pp.X_CHECK_VALID, pp.SHA_MISMATCH,
                     'abcdef123456789', 'ladrc', 'Lỗi', pp.CHK_CLEAN_NO_SPIFIX,
                     0x1234, 'x' * 25, 127)
    wire = pp.encode_extra(extra)
    payload = (struct.pack('<BBIBB', 0xe0, 7, SID1, 3, 3) + b'abcdef123456' +
               b'ladrc' + bytes(3) + b'L?i' + bytes(19) + b'\x02\x34\x12' +
               b'x' * 22 + b'\x7f\x00\x00')
    assert len(wire) == 80 and wire == seal(payload)
    assert pp.decode(wire) == replace(extra, dut_sha='abcdef123456', fault_text='L?i',
                                      check_text='x' * 22)
    empty = pp.encode_extra(pp.Extra())
    assert empty == seal(b'\xe0\x07' + bytes(73) + b'\xff' + bytes(2))
    assert pp.decode(empty) == pp.Extra()
    with pytest.raises(ValueError):
        pp.encode_extra(pp.Extra(flags=pp.X_CHECK_VALID))


@pytest.mark.parametrize('n', [0, 1, 22])
def test_track_wire_layout_and_saturation(n):
    points = ((-40000, 40000),) * n
    track = pp.Track(SID1, 65535, pp.TR_DUT, 15, 16, points)
    wire = pp.encode_track(track)
    payload = (struct.pack('<BBIHBBBB', 0xe1, 7, SID1, 65535, 2, 15, 16, n) +
               b'\x00\x80\xff\x7f' * n + bytes((22 - n) * 4 + 4))
    assert len(wire) == 106 and wire == seal(payload)
    assert pp.decode(wire) == replace(track, points=((-32768, 32767),) * n)


@pytest.mark.parametrize('fields', [
    dict(run_no=0), dict(run_no=65536), dict(series=-1), dict(series=3),
    dict(chunk=-1), dict(chunk=1), dict(count=0), dict(count=17), dict(points=((0, 0),) * 23),
])
def test_invalid_track_encode(fields):
    with pytest.raises(ValueError):
        pp.encode_track(replace(pp.Track(1, 1, pp.TR_REF, 0, 1), **fields))


def mixed_frames():
    return [pp.encode_status(pp.Status(sid=SID1)),
            pp.encode_extra(pp.Extra(sid=SID1, dut_env='ladrc')),
            pp.encode_info(pp.Info(sid=SID1, ctrl='PID')),
            pp.encode_track(pp.Track(SID1, 7, pp.TR_PLANT, 0, 1, ((-123, 456),))),
            pp.encode_profile(SID1, 0, 1, 'RESET DUT')]


def split_reads(data):
    yield [bytes([b]) for b in data]
    for split in range(len(data) + 1):
        yield [data[:split], data[split:]]


def test_mixed_v7_stream_at_every_boundary():
    frames = mixed_frames()
    expected = list(map(pp.decode, frames))
    for reads in split_reads(b''.join(frames)):
        parser = pp.StreamParser(clock=lambda: 0)
        assert [item for read in reads for item in parser.feed(read)] == expected
        assert (parser.bad_frames, parser.skipped_bytes, parser.version_mismatch) == (0, 0, 0)


def version_stream(version, corrupt=False):
    # The foreign STATUS straddles a 106-byte C buffer fill.
    before = pp.encode_extra(pp.Extra()) + pp.encode_cmd(1, 0, pp.CMD_SYNC)
    status = bytearray(pp.encode_status(pp.Status())[:-2])
    status[1] = version
    frame = bytearray(seal(status))
    if corrupt:
        frame[-1] ^= 1
    after = pp.encode_info(pp.Info())
    return before + frame + after, [pp.Extra(), pp.Cmd(1, 0, pp.CMD_SYNC, 0), pp.Info()]


def test_foreign_version_crc_and_recovery():
    parser = pp.StreamParser(clock=lambda: 0)
    for count, version in enumerate((6, 255), 1):
        stream, expected = version_stream(version)
        assert parser.feed(stream) == expected
        assert (parser.version_mismatch, parser.last_foreign_version) == (count, version)
        assert parser.bad_frames == parser.skipped_bytes == 0
    for version, corrupt in [(5, True), (0, False)]:
        stream, expected = version_stream(version, corrupt)
        previous = parser.bad_frames
        assert parser.feed(stream) == expected
        assert parser.bad_frames > previous     # resync may find start bytes inside bad frames
    assert (parser.version_mismatch, parser.last_foreign_version) == (2, 255)


def test_panel_keeps_extra_and_clears_demo_data_on_session_loss():
    bench = Bench()
    panel = bench.panel
    panel.on_extra(pp.Extra(sid=0))
    assert not panel.has_extra
    bench.step(10)
    first = pp.Extra(sid=SID1, dut_sha='123456abcdef')
    second = pp.Extra(sid=SID1, flags=pp.X_FAULT_ACTIVE, fault_text='LOI: ENC FL 2.1s')
    panel.on_extra(first)
    assert panel.has_extra and panel.extra == first
    panel.on_extra(second)
    panel.on_extra(replace(first, sid=SID1 + 1))
    assert panel.extra == second
    panel.on_track(pp.Track(SID1, 8, pp.TR_REF, 0, 1, ((1, 2),)))
    bench.sess = pp.DaemonSession(8)
    bench.step(10)
    assert panel.sid == 0
    assert not panel.has_extra and panel.extra is None
    assert panel.track_run_no == 0
    assert not panel.track_complete(pp.TR_REF) and panel.track_points(pp.TR_REF) == ()
    bench.step(10)
    assert panel.sid and panel.sid != SID1
    panel.on_track(pp.Track(panel.sid, 1, pp.TR_REF, 0, 1))
    assert panel.track_run_no == 1 and panel.track_complete(pp.TR_REF)


def test_panel_assembles_track_chunks():
    bench = Bench()
    bench.step(10)
    panel = bench.panel
    ref, plant, dut = pp.TR_REF, pp.TR_PLANT, pp.TR_DUT
    first = pp.Track(SID1, 7, ref, 0, 2, ((1, -1),))
    last = replace(first, chunk=1, points=((2, -2), (3, -3)))

    def check(series, complete, points):
        assert panel.track_complete(series) == complete
        assert panel.track_point_count(series) == len(points)
        assert panel.track_points(series) == points

    panel.on_track(last)
    check(ref, False, last.points)
    panel.on_track(first)
    check(ref, True, first.points + last.points)
    panel.on_track(replace(last, points=()))
    check(ref, True, first.points)
    panel.on_track(replace(first, series=plant))
    panel.on_track(replace(last, series=plant))
    check(plant, True, first.points + last.points)
    panel.on_track(replace(last, count=3))
    check(ref, False, last.points)
    check(plant, True, first.points + last.points)
    panel.on_track(replace(first, run_no=99, sid=SID1 + 1))
    assert panel.track_run_no == 7
    check(ref, False, last.points)
    panel.on_track(replace(first, run_no=8, series=dut, count=1))
    panel.on_track(first)
    assert panel.track_run_no == 8
    check(ref, False, ())
    check(plant, False, ())
    check(dut, True, first.points)


def test_panel_full_track_capacity():
    bench = Bench()
    bench.step(10)
    panel = bench.panel
    for series in (pp.TR_REF, pp.TR_PLANT, pp.TR_DUT):
        points = tuple((series, i) for i in range(352))
        for chunk in reversed(range(16)):
            start = 22 * chunk
            panel.on_track(pp.Track(SID1, 65535, series, chunk, 16, points[start:start + 22]))
        assert panel.track_complete(series)
        assert panel.track_point_count(series) == 352
        assert panel.track_points(series) == points
