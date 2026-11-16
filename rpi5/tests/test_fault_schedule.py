"""Fault injection batch 1 (L82): schedule format and checks, the Pi frame patcher (fault_schedule.py).
The patcher against the independent reference of tools/fault_check.py: tools/tests/test_fault_check.py."""
import math
import struct
from fractions import Fraction

import numpy as np
import pytest

import fault_schedule as fs
from protocol import decode_imu, decode_state, encode_state

HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
GOOD = HEAD + ('E1,enc_loss,FL,100,200,\nL1,imu_loss,imu,300,400,\n'
               'D1,imu_delay,imu,500,600,d_ms=30\nS1,gyro_scale,gz,700,800,s=0.3\n')


def frames(n, seed=1, fault_flags=8):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        st = list(rng.uniform(-30, 30, 4)) + list(rng.uniform(-2, 2, 6))
        imu = (math.sin(k / 50.0) * 3 + 0.001 * k, 0.01 * k - 3, 1.0, 9.81, ((k * 0.004 + 1) % 6.28) - 3.14)
        out.append(encode_state(st, traj_ref=(0.1, 0.0, 0.2), traj_flags=0, seq=k, imu=imu,
                                imu_seq=k % 255 + 1, fault_flags=fault_flags))
    return out


def sched(text, run_ms=1200, imu='plant'):
    rows = fs.parse(text)
    fs.check(rows, run_ms, imu)
    return fs.Schedule(tuple(rows), 'x.csv', 'abc')


def run_patcher(s, true, t0=0, stop=None):
    p = fs.FramePatcher(s)
    sent = []
    for k, f in enumerate(true):
        if k == t0:
            p.start(k)
        if stop is not None and k == stop:
            p.stop(k)
        sent.append(p.patch(k, f))
    return p, sent


# ── format ──────────────────────────────────────────────────────────────────
def test_parse_good():
    rows = fs.parse(GOOD)
    fs.check(rows, 1000, 'plant')
    assert [r.fault for r in rows] == ['enc_loss', 'imu_loss', 'imu_delay', 'gyro_scale']
    assert rows[2].p('d_ms') == 30 and rows[3].p('s') == Fraction(3, 10)      # exact decimal (FI1-1)
    assert rows[0].wheels == (0,) and fs.parse(HEAD + 'E,enc_loss,FR+RR,1,2,\n')[0].wheels == (1, 3)


def test_parse_comments_crlf_and_blank_lines():
    text = (HEAD + '# a comment\n\nE1,enc_loss,RL,10,20,\n').replace('\n', '\r\n')
    rows = fs.parse(text)
    assert [(r.id, r.target, r.start_ms, r.end_ms) for r in rows] == [('E1', 'RL', 10, 20)]


@pytest.mark.parametrize('bad', [
    'id,fault,target,start_ms,end_ms,params\nE1,enc_loss,FL,1,2,\n',     # no version line
    '﻿' + GOOD,                                                     # BOM before the version line
    HEAD.replace('target', 'wheel') + 'E1,enc_loss,FL,1,2,\n',          # columns
    HEAD,                                                                # no row
    HEAD + 'E1,enc_los,FL,1,2,\n',                                       # unknown fault
    HEAD + 'J1,wheel_jam,FL,1,2,b_mult=2\n',                             # unknown parameter of a batch 2 row
    HEAD + 'I1,ibus_loss,ibus,1,2,x=1\n',                                # ibus_loss has no parameter
    HEAD + 'B1,body_drag,body,1,2,F=3.3\n',                              # later parts of batch 2 are refused
    HEAD + 'N1,load_dist,body,1,2,\n',
    HEAD + 'E1,enc_loss,FL,1.5,2,\n',                                    # not whole ms
    HEAD + 'E1,enc_loss,FL,5e3,6000,\n',
    HEAD + 'E1,enc_loss,FL,5_000,6000,\n',
    HEAD + 'E1,enc_loss,FL,-1,2,\n',
    HEAD + 'E1,enc_loss,FL,1,2,x=1\n',                                   # unknown parameter
    HEAD + 'D1,imu_delay,imu,100,200,\n',                                # missing d_ms
    HEAD + 'D1,imu_delay,imu,100,200,d_ms=2.5\n',                        # d_ms not whole
    HEAD + 'D1,imu_delay,imu,100,200,d_ms=30;d_ms=31\n',                 # twice
    HEAD + 'S1,gyro_scale,gz,1,2,s=nan\n',                               # not finite
    HEAD + 'S1,gyro_scale,gz,1,2,s=abc\n',
    HEAD + 'S1,gyro_scale,gz,1,2,s=inf\n',
    HEAD + 'S1,gyro_scale,gz,1,2,s=3/10\n',                              # decimal only (FI1-1)
    HEAD + 'S1,gyro_scale,gz,1,2,s=\n',
    HEAD + 'E1,enc_loss,FL,1,2\n',                                      # a field short
    HEAD + 'E1,enc_loss,FL,1,2,,\n',                                     # a field too many
])
def test_parse_refuses(bad):
    with pytest.raises(fs.ScheduleError):
        fs.parse(bad)


@pytest.mark.parametrize('body,imu', [
    ('E1,enc_loss,XX,1,2,\n', 'plant'),                                  # bad wheel
    ('E1,enc_loss,FL+FL,1,2,\n', 'plant'),                               # wheel twice
    ('E1,enc_loss,imu,1,2,\n', 'plant'),
    ('E1,enc_loss,FL,5,5,\n', 'plant'),                                  # empty window
    ('E1,enc_loss,FL,1,1001,\n', 'plant'),                               # after the run
    ('E1,enc_loss,FL,1,9,\nE1,enc_loss,FR,1,9,\n', 'plant'),             # id twice
    (',enc_loss,FL,1,9,\n', 'plant'),                                    # empty id
    ('E1,enc_loss,FL,1,9,\nE2,enc_loss,FL+RR,8,20,\n', 'plant'),         # same wheel overlaps
    ('D1,imu_delay,imu,29,90,d_ms=30\n', 'plant'),                       # source frame before t0
    ('D1,imu_delay,imu,100,200,d_ms=0\n', 'plant'),
    ('D1,imu_delay,imu,1001,1002,d_ms=1001\n', 'plant'),
    ('D1,imu_delay,imu,100,200,d_ms=30\n', 'static'),                    # constant values: no meaning
    ('S1,gyro_scale,gz,1,2,s=0\n', 'plant'),
    ('S1,gyro_scale,gz,1,2,s=1.5\n', 'plant'),
    ('S1,gyro_scale,gz,1,9,s=0.2\nS2,gyro_scale,gz,5,12,s=0.4\n', 'plant'),
    ('D1,imu_delay,imu,40,90,d_ms=30\nD2,imu_delay,imu,80,99,d_ms=10\n', 'plant'),
    ('L1,imu_loss,imu,1,9,\nL2,imu_loss,imu,8,12,\n', 'plant'),
    ('L1,imu_loss,gz,1,9,\n', 'plant'),                                  # imu_loss is all three reports
    ('S1,gyro_scale,imu,1,9,s=0.2\n', 'plant'),
    ('L1,imu_loss,imu,1,9,\n', None),                                    # no IMU in the frames
    ('S1,gyro_scale,gz,1,9,s=0.2\n', None),
    ('I1,ibus_loss,imu,1,9,\n', 'plant'),                               # batch 2 iBUS loss: target ibus only
    ('I1,ibus_loss,FL,1,9,\n', 'plant'),
    ('I1,ibus_loss,ibus,1,9,\nI2,ibus_loss,ibus,8,12,\n', 'plant'),     # two rows on the same bit overlap
    ('I1,ibus_loss,ibus,1,1001,\n', None),                              # ends after the run
])
def test_check_refuses(body, imu):
    with pytest.raises(fs.ScheduleError):
        fs.check(fs.parse(HEAD + body), 1000, imu)


def test_check_needs_run_length():
    with pytest.raises(fs.ScheduleError):
        fs.check(fs.parse(GOOD), None, 'plant')


def test_check_allows_other_fields_and_edges():
    fs.check(fs.parse(HEAD + 'E1,enc_loss,FL,1,9,\nE2,enc_loss,FR,1,9,\nS1,gyro_scale,gz,1,9,s=0.2\n'
                             'L1,imu_loss,imu,1,9,\nD1,imu_delay,imu,1,9,d_ms=1\n'), 1000, 'plant')
    fs.check(fs.parse(HEAD + 'E1,enc_loss,FL,0,1000,\n'), 1000, None)     # enc_loss needs no IMU, end = run
    fs.check(fs.parse(HEAD + 'S1,gyro_scale,gz,1,9,s=0.2\nS2,gyro_scale,gz,9,12,s=0.4\n'), 1000, 'static')
    fs.check(fs.parse(HEAD + 'D1,imu_delay,imu,30,90,d_ms=30\n'), 1000, 'profile')   # source = t0 frame


def test_load_sha_ignores_crlf(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_bytes(GOOD.encode())
    b.write_bytes(GOOD.replace('\n', '\r\n').encode())
    sa, sb = fs.load(str(a), 1000, 'plant'), fs.load(str(b), 1000, 'plant')
    assert sa.sha256 == sb.sha256 and sa.rows == sb.rows and len(sa.sha256) == 64
    d = sa.describe()
    assert d['sha256'] == sa.sha256 and d['rows'][2] == dict(id='D1', fault='imu_delay', target='imu',
                                                              start_ms=500, end_ms=600, params={'d_ms': 30})


def test_scale_raw_rounding_and_clamp():
    assert fs.scale_raw(5, 0.5) == 3 and fs.scale_raw(-5, 0.5) == -3          # half away from zero
    assert fs.scale_raw(4, 0.5) == 2 and fs.scale_raw(0, 0.3) == 0
    assert fs.scale_raw(30000, -1.0) == 32767 and fs.scale_raw(-30000, -1.0) == -32767
    assert fs.scale_raw(1000, 1.0) == 0
    # FI1-1: exact half points that float arithmetic rounds toward zero
    assert fs.scale_raw(45, Fraction('0.3')) == 32 and fs.scale_raw(-45, Fraction('0.3')) == -32
    assert fs.scale_raw(5, Fraction('0.9')) == 1 and fs.scale_raw(-5, Fraction('0.9')) == -1
    assert fs.scale_raw(45, 0.3) == 32                     # a float s is taken as its shortest decimal
    assert fs.rows_as_json(fs.parse(HEAD + 'S1,gyro_scale,gz,1,2,s=0.3\n'))[0]['params'] == {'s': 0.3}


# ── Pi patcher ──────────────────────────────────────────────────────────────
def test_patcher_without_rows_in_window_returns_the_same_object():
    true = frames(50)
    p, sent = run_patcher(sched(HEAD + 'E1,enc_loss,FL,60,70,\n', run_ms=100), true)
    assert all(a is b for a, b in zip(true, sent))


def test_each_fault_and_half_open_window():
    true = frames(1000)
    s = sched(GOOD, run_ms=1000)
    p, sent = run_patcher(s, true)
    for k in range(1000):
        t, x = true[k], sent[k]
        on = [r.id for r in s.rows if r.start_ms <= k < r.end_ms]
        if not on:
            assert x is t
        elif on == ['E1']:
            assert x[0:2] == b'\x00\x00' and x[2:] == t[2:] and decode_state(x)[0] == 0.0
        elif on == ['L1']:
            assert x[20] == t[20] | 4 and x[20] & 8 and x[:20] == t[:20] and x[21:] == t[21:]   # SWA kept
        elif on == ['D1']:
            d = true[k - 30]
            assert x[22:24] == d[22:24] and x[31:39] == d[31:39]
            assert x[:22] == t[:22] and x[24:31] == t[24:31] and x[39] == t[39]   # omega, seq, imu_seq kept
        elif on == ['S1']:
            raw = struct.unpack_from('>h', t, 22)[0]
            assert struct.unpack_from('>h', x, 22)[0] == fs.scale_raw(raw, 0.3)
            assert x[:22] == t[:22] and x[24:] == t[24:]
        else:
            raise AssertionError(on)
    assert sent[99] is true[99] and sent[100] != true[100] and sent[199] != true[199] and sent[200] is true[200]
    sm = p.summary(last_step=999)
    assert sm['events'] == [[100, 'E1', 1, 'on'], [200, 'E1', 0, 'off'], [300, 'L1', 1, 'on'],
                            [400, 'L1', 0, 'off'], [500, 'D1', 1, 'on'], [600, 'D1', 0, 'off'],
                            [700, 'S1', 1, 'on'], [800, 'S1', 0, 'off']]
    assert sm['complete'] and not sm['cut'] and sm['not_run'] == [] and sm['t0_step'] == 0
    assert sm['sha256'] == 'abc' and len(sm['rows']) == 4


def test_imu_loss_sets_the_bit_with_swa_off_too():
    true = frames(20, fault_flags=0)
    _, sent = run_patcher(sched(HEAD + 'L1,imu_loss,imu,5,10,\n', run_ms=20), true)
    assert [x[20] for x in sent[4:11]] == [0, 4, 4, 4, 4, 4, 0]


def test_t0_shift_and_nothing_before_start():
    true = frames(600)
    p, sent = run_patcher(sched(HEAD + 'E1,enc_loss,RL,100,200,\n'), true, t0=250)
    assert all(sent[k] is true[k] for k in range(350))
    assert all(sent[k][4:6] == b'\x00\x00' for k in range(350, 450)) and sent[450] is true[450]
    sm = p.summary(last_step=599)
    assert sm['t0_step'] == 250 and sm['events'] == [[100, 'E1', 1, 'on'], [200, 'E1', 0, 'off']]


def test_truth_frame_is_not_changed():
    true = frames(300)
    copy = [bytes(f) for f in true]
    run_patcher(sched(HEAD + 'E1,enc_loss,FL+FR+RL+RR,10,290,\nL1,imu_loss,imu,10,290,\n'
                             'D1,imu_delay,imu,20,290,d_ms=20\nS1,gyro_scale,gz,30,290,s=0.5\n'), true)
    assert true == copy


def test_delay_window_start_uses_frames_from_before_the_window():
    true = frames(400)
    _, sent = run_patcher(sched(HEAD + 'D1,imu_delay,imu,200,300,d_ms=100\n'), true)
    assert decode_imu(sent[200])[:5] == decode_imu(true[100])[:5]
    assert decode_imu(sent[299])[:5] == decode_imu(true[199])[:5]
    assert sent[300] is true[300]


def test_delay_reads_true_frames_with_ring_over_t0():
    """The ring keeps every frame, before t0 too; d = start - 0 takes the t0 frame itself."""
    true = frames(200)
    _, sent = run_patcher(sched(HEAD + 'D1,imu_delay,imu,40,60,d_ms=40\n'), true, t0=100)
    for k in range(140, 160):
        assert sent[k][22:24] == true[k - 40][22:24] and sent[k][31:39] == true[k - 40][31:39]


def test_stop_cuts_open_windows_and_marks_not_run():
    """Design 6.2: once the run leaves RUN, frames are the true ones; an open window is 'cut', a later one
    'not_run'; a window ending exactly at the stop step ends normally."""
    true = frames(400)
    s = sched(HEAD + 'E1,enc_loss,FL,10,50,\nL1,imu_loss,imu,40,300,\nE2,enc_loss,RR,200,250,\n'
                     'S1,gyro_scale,gz,20,150,s=0.3\n')
    p, sent = run_patcher(s, true, t0=100, stop=250)     # relative stop 150
    assert sent[249][20] & 4 and all(sent[k] is true[k] for k in range(250, 400))
    sm = p.summary(last_step=399)
    assert sm['events'] == [[10, 'E1', 1, 'on'], [20, 'S1', 1, 'on'], [40, 'L1', 1, 'on'],
                            [50, 'E1', 0, 'off'], [150, 'S1', 0, 'off'], [150, 'L1', 0, 'cut']]
    assert sm['not_run'] == ['E2'] and sm['cut'] and not sm['complete'] and sm['stop_step'] == 250


def test_stop_without_start_and_double_calls():
    true = frames(50)
    p = fs.FramePatcher(sched(HEAD + 'E1,enc_loss,FL,1,5,\n'))
    p.stop(3)
    p.start(10)                                         # a start after the stop does not revive it
    assert all(p.patch(k, f) is f for k, f in enumerate(true))
    sm = p.summary(last_step=49)
    assert sm['t0_step'] is None and sm['stop_step'] == 3 and sm['events'] == [] and sm['not_run'] == ['E1'] and not sm['complete']


def test_summary_at_end_of_loop_cuts_open_window_ends_window_at_last_step():
    true = frames(100)
    s = sched(HEAD + 'E1,enc_loss,FL,10,100,\nE2,enc_loss,FR,90,120,\n', run_ms=120)
    p, _ = run_patcher(s, true)
    sm = p.summary(last_step=99)
    assert sm['events'][-2:] == [[100, 'E1', 0, 'off'], [100, 'E2', 0, 'cut']]
    assert p.summary(last_step=99)['events'] == sm['events']          # idempotent


# ── batch 2: ibus_loss (design 2026-11-15 sections 4.2, 5) ──────────────────
def test_ibus_loss_needs_no_imu_nor_plant_and_windows_may_touch():
    fs.check(fs.parse(HEAD + 'I1,ibus_loss,ibus,1,9,\nI2,ibus_loss,ibus,9,12,\n'), 1000, None, None)
    fs.check(fs.parse(HEAD + 'I1,ibus_loss,ibus,0,1000,\nL1,imu_loss,imu,0,1000,\nE1,enc_loss,FL,0,9,\n'),
             1000, 'plant')


def test_ibus_loss_sets_bit4_keeps_swa_and_imu_loss_bits():
    true = frames(16)                                                  # SWA bit 3 set in every true frame
    _, sent = run_patcher(sched(HEAD + 'I1,ibus_loss,ibus,5,10,\nL1,imu_loss,imu,8,12,\n', run_ms=16), true)
    assert [x[20] for x in sent[3:14]] == [8, 8, 24, 24, 24, 28, 28, 12, 12, 8, 8]
    for k in range(16):                                                # nothing else of the frame changes
        assert sent[k][:20] == true[k][:20] and sent[k][21:] == true[k][21:], k


def test_ibus_loss_with_swa_off_and_its_window_cut():
    true = frames(20, fault_flags=0)
    p, sent = run_patcher(sched(HEAD + 'I1,ibus_loss,ibus,3,15,\n', run_ms=20), true, t0=2, stop=10)
    assert [x[20] for x in sent] == [0] * 5 + [16] * 5 + [0] * 10      # t0 2 + 3 = 5 .. stop 10
    assert p.summary(last_step=19)['events'] == [[3, 'I1', 1, 'on'], [8, 'I1', 0, 'cut']]


def test_ibus_row_alone_is_a_frame_row():
    a = fs.ActiveSet(fs.parse(HEAD + 'I1,ibus_loss,ibus,1,3,\n'))
    a.advance(1)
    assert a.ibus and a.any and a.plant_on == () and a.enc_mask == 0 and not a.loss
    a.advance(3)
    assert not a.ibus and not a.any


def test_ibus_constants():
    from protocol import CMD_IBUS_OFF, CMD_SWA_ON
    assert CMD_IBUS_OFF == 16 and CMD_IBUS_OFF & (CMD_SWA_ON | fs.FAULT_IMU_LOSS) == 0
    assert 'ibus_loss' in fs.FRAME_FAULTS and 'ibus_loss' not in fs.BATCH2 + fs.PLANT_FAULTS
    assert fs.IBUS_LOSS_H7 is False                 # the H7 part is not merged: set True on its branch only
