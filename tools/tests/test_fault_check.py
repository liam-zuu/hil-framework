"""tools/fault_check.py (L82, design 8.1; Codex round 1 FI1-1, 2, 7, 8, 9, 10): the independent reference
against the Pi patcher, and the check on real frame logs of the 1 kHz loop (simulate_hw, MISO with a valid
CRC, a step log). A good run passes; each tampering fails: a frame injected one step late, the wrong wheel,
a window one frame too long, a touched seq byte, the patcher late with a matching t0 in its manifest, a
wrong stop step, events of the wrong type or order, flipped flags, MISO counts edited, a truth frame edited
in both arrays, no step log, a missed deadline, a missing step, float rounding of gyro_scale.
Run: python -m pytest tools/tests"""
import binascii
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import fault_check as fc                       # noqa: E402
import fault_schedule as fs                    # noqa: E402
from hil_node import HILConfig, HILNode        # noqa: E402
from mecanum_plant import MecanumPlant         # noqa: E402
from protocol import (FRAME_SIZE, ID_LEN, OFS_IMU_SEQ, OFS_MISO_CRC, OFS_MISO_ID, OFS_RUN_ID,   # noqa: E402
                      encode_state)
from run_lifecycle import RUN, ControlBox, LifecycleConfig   # noqa: E402
from step_log import CSV_HEADER                # noqa: E402

HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
BODY = ('E1,enc_loss,FR,50,120,\nL1,imu_loss,imu,100,200,\nD1,imu_delay,imu,220,300,d_ms=20\n'
        'S1,gyro_scale,gz,250,350,s=0.3\n')


def with_crc(b) -> bytes:
    """A MISO frame with the CRC the Pi checks (protocol.miso_check)."""
    b = bytearray(b)
    crc = binascii.crc_hqx(bytes(b[:OFS_MISO_CRC] + b[OFS_MISO_CRC + 2:]), 0xFFFF)
    b[OFS_MISO_CRC], b[OFS_MISO_CRC + 1] = crc & 0xFF, crc >> 8
    return bytes(b)


GOOD_MISO = with_crc(bytes([0x10, 0x00] * 4) + bytes(FRAME_SIZE - 8))     # no run_id: never confirms


def good_miso(mosi) -> bytes:
    """A MISO with a good CRC carrying back MOSI 40-47 (run_id, frame_no), as the H7 v6 and fake_h7 do (FI4-1)."""
    return with_crc(GOOD_MISO[:OFS_MISO_ID] + bytes(mosi[OFS_RUN_ID:OFS_RUN_ID + ID_LEN])
                    + GOOD_MISO[OFS_MISO_ID + ID_LEN:])


def moving_plant():
    """A plant that turns from its first step (wheels and body moving), so that every fault changes bytes."""
    p = MecanumPlant(dt=0.001, seed=0)
    p.reset([0, 0, 0, 0.2, 0, 0.8, 10, 12, 14, 16])
    return p


def write_step_log(path, n, miss_at=(), skip=(), lost=0):
    """A step log as run_hil.py --step-log writes it (CSV + .json), synthetic times."""
    with open(path, 'w', newline='\n') as fh:
        fh.write(CSV_HEADER + '\n')
        for k in range(n):
            if k in skip:
                continue
            fh.write(f'{k},{k * 1000.0:.3f},{k * 1000.0 + 1:.3f},{k * 1000.0 + 60:.3f},{k * 1000.0 + 61:.3f},'
                     f'{1 if k in miss_at else 0}\n')
    with open(path + '.json', 'w') as fh:
        json.dump(dict(rows=n - len(skip), lost_or_torn=lost, overflow=0), fh)
    return path


def frames(n, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        st = list(rng.uniform(-30, 30, 4)) + list(rng.uniform(-2, 2, 6))
        imu = (math.sin(k / 50.0) * 3 + 0.001 * k, 0.01 * k - 3, 1.0, 9.81, ((k * 0.004 + 1) % 6.28) - 3.14)
        out.append(encode_state(st, traj_ref=(0.1, 0.0, 0.2), seq=k, imu=imu, imu_seq=k % 255 + 1,
                                fault_flags=8))
    return out


def test_patcher_agrees_with_independent_reference_on_overlaps():
    true = frames(1200, seed=3)
    rows = fs.parse(HEAD + 'E1,enc_loss,FL+RR,100,900,\nD1,imu_delay,imu,300,700,d_ms=57\n'
                           'S1,gyro_scale,gz,400,800,s=-0.5\nL1,imu_loss,imu,650,1000,\nE2,enc_loss,FR,850,1100,\n'
                           'S2,gyro_scale,gz,900,1150,s=0.3\n')
    fs.check(rows, 1200, 'plant')
    for t0, stop in ((0, None), (37, None), (37, 900)):
        p = fs.FramePatcher(fs.Schedule(tuple(rows)))
        for k, f in enumerate(true):
            if k == t0:
                p.start(k)
            if stop is not None and k == stop:
                p.stop(k)
            assert p.patch(k, f) == fc.expected_sent(rows, k, t0, stop, true), (t0, stop, k)
        ev, nr = fc.expected_events(rows, t0, stop, len(true))
        sm = p.summary(last_step=len(true) - 1)
        assert fc.events_typed(sm['events'])
        assert sorted(sm['events'], key=lambda e: (e[0], fc.WHY_RANK[e[3]], e[1])) == ev
        assert sorted(sm['not_run']) == sorted(nr)


def test_reference_scale_is_exact_and_independent():
    """FI1-1: half points that float arithmetic rounds the wrong way; the two implementations agree on a sweep."""
    for raw, s, want in ((45, '0.3', 32), (-45, '0.3', -32), (5, '0.9', 1), (-5, '0.9', -1), (1, '0.5', 1),
                         (3, '0.5', 2), (32000, '-1', 32767), (100, '1', 0)):
        assert fc.ref_scale(raw, Fraction(s)) == want == fs.scale_raw(raw, Fraction(s)), (raw, s)
    for s in ('0.3', '0.05', '-0.5', '0.9', '0.123', '1e-1'):
        for raw in range(-2000, 2001, 7):
            assert fc.ref_scale(raw, Fraction(s)) == fs.scale_raw(raw, Fraction(s)), (raw, s)
    # a patcher rounding in float would send 31 here: the reference says 32
    true = [encode_state([0.0] * 10, imu=(45 * 34.906585 / 32767, 0, 0, 9.8, 0), imu_seq=1)]
    assert int.from_bytes(true[0][22:24], 'big', signed=True) == 45
    rows = fs.parse(HEAD + 'S1,gyro_scale,gz,0,1,s=0.3\n')
    exp = fc.expected_sent(rows, 0, 0, None, true)
    assert int.from_bytes(exp[22:24], 'big', signed=True) == 32


def open_loop(tmp_path, body=BODY, steps=400, run_ms=400, transfer=None, name='frames'):
    csv_ = tmp_path / f'{name}.csv'
    csv_.write_text(HEAD + body)
    sched = fs.load(str(csv_), run_ms, 'plant')
    p = str(tmp_path / f'{name}.npz')
    node = HILNode(moving_plant(), config=HILConfig(simulate_hw=True, max_steps=steps, frame_log=p, faults=sched))
    node._spi.transfer = transfer or good_miso
    node.run()
    return p, str(csv_), write_step_log(str(tmp_path / f'{name}_steps.csv'), steps)


@pytest.fixture
def run(tmp_path):
    return open_loop(tmp_path)


def resave(src, dst, edit):
    f = np.load(src)
    d = {k: f[k].copy() for k in f.files}
    man = json.loads(str(d['manifest']))
    edit(d, man)
    d['manifest'] = np.array(json.dumps(man))
    np.savez(dst, **d)
    return dst


def test_good_run_passes(run):
    r = fc.check(*run)
    assert r['ok'], r['fails']
    assert r['frames_bad'] == 0 and r['complete'] and r['t0_step'] == 0 and r['stop_step'] is None
    assert 150 + 80 <= r['frames_injected'] <= 150 + 130     # [50, 200), [220, 300); [300, 350) where gz != 0
    assert r['miso'] == dict(checked=398, crc_bad=0, empty=0)
    assert r['self_check']['seq_bad'] == 0


def _patch_with_t0(d, m, t0):
    """mosi rebuilt by the real FramePatcher started at t0 on the true frame of every step (Codex FI2-8):
    only the schedule is late, every other byte is still the one of the current frame."""
    rows = fs.parse(HEAD + BODY)
    p = fs.FramePatcher(fs.Schedule(tuple(rows)))
    for k in range(len(d['mosi_true'])):
        if k == t0:
            p.start(k)
        d['mosi'][k] = np.frombuffer(p.patch(k, bytes(d['mosi_true'][k])), dtype=np.uint8)
    m['faults']['t0_step'] = t0


def _shift_late(d, m):
    """The patcher one step late everywhere, with t0 = 1 in its own manifest (FI1-2, FI2-8)."""
    _patch_with_t0(d, m, 1)


def test_patcher_late_with_matching_t0_is_caught_only_by_the_phases(run, tmp_path):
    """Codex FI2-8: the bytes of a patcher started at t0 = 1 agree with the byte reference taken at t0 = 1
    (so a byte check alone, trusting the patcher's t0, would pass), and the frames do differ from the good
    run; the checker refuses the run because the phases give t0 = 0."""
    p = resave(run[0], str(tmp_path / 'late1.npz'), _shift_late)
    f = np.load(p)
    rows = fs.parse(HEAD + BODY)
    assert all(bytes(f['mosi'][k]) == fc.expected_sent(rows, k, 1, None, f['mosi_true'])
               for k in range(len(f['mosi'])))
    g = np.load(run[0])
    assert not np.array_equal(f['mosi'], g['mosi'])
    assert np.array_equal(f['mosi'][:, OFS_IMU_SEQ], g['mosi'][:, OFS_IMU_SEQ])   # imu_seq of the current frame
    r = fc.check(p, run[1], run[2])
    assert not r['ok'] and any('t0_step=1' in s for s in r['fails']), r['fails']


@pytest.mark.parametrize('name,edit', [
    ('late', lambda d, m: d['mosi'].__setitem__(50, d['mosi_true'][50])),
    ('wheel', lambda d, m: (d['mosi'].__setitem__((60, slice(2, 4)), d['mosi_true'][60, 2:4]),
                            d['mosi'].__setitem__((60, slice(0, 2)), 0))),
    ('long', lambda d, m: d['mosi'].__setitem__((120, slice(2, 4)), 0)),
    ('seq', lambda d, m: d['mosi'].__setitem__((110, 21), (int(d['mosi'][110, 21]) + 1) % 256)),
    ('patcher_late_t0', _shift_late),
    ('t0', lambda d, m: m['faults'].__setitem__('t0_step', 1)),
    ('stop', lambda d, m: m['faults'].__setitem__('stop_step', 399)),
    ('events_missing', lambda d, m: m['faults'].__setitem__('events', m['faults']['events'][:-1])),
    ('events_float', lambda d, m: m['faults']['events'][0].__setitem__(0, 50.0)),
    ('events_bool', lambda d, m: m['faults']['events'][0].__setitem__(2, True)),
    ('events_order', lambda d, m: m['faults']['events'].reverse()),
    ('cut_flag', lambda d, m: m['faults'].__setitem__('cut', True)),
    ('complete_flag', lambda d, m: m['faults'].__setitem__('complete', 1)),
    ('not_run', lambda d, m: m['faults'].__setitem__('not_run', ['E1'])),
    ('miso_count', lambda d, m: m['miso_check'].__setitem__('crc_bad', 3)),
    ('miso_checked', lambda d, m: m['miso_check'].__setitem__('checked', 397)),             # FI2-5
    ('manifest_steps', lambda d, m: m.__setitem__('steps', 399)),                          # FI2-5
    ('stop_reason', lambda d, m: m.__setitem__('stop_reason', 'signal_15')),               # FI1-4
    ('max_steps', lambda d, m: m.__setitem__('max_steps', 500)),                           # FI1-4
    ('miso_frame', lambda d, m: d['miso'].__setitem__((40, 0), 0x55)),
    ('true_seq', lambda d, m: (d['mosi_true'].__setitem__((5, 21), 99), d['mosi'].__setitem__((5, 21), 99))),
])
def test_tampered_run_fails(run, tmp_path, name, edit):
    p = resave(run[0], str(tmp_path / f'{name}.npz'), edit)
    r = fc.check(p, run[1], run[2])
    assert not r['ok'], name


def test_run_without_id_confirmed_fails(run, tmp_path):
    """Codex FI4-1: a run whose MISO never carried its run_id back (H7 not v6, wrong H7, link swapped):
    RS-4's run_failed takes it as failed on the real link; the checker refuses it for every run. A run with a
    good CRC but no run_id (40 frames, under the 50-transfer stop) has id_confirmed False and fails; the good
    run with the key set False or removed fails too; the good run itself has id_confirmed True."""
    assert json.loads(str(np.load(run[0])['manifest']))['id_confirmed'] is True
    p, csv_, steps = open_loop(tmp_path, body='E1,enc_loss,FL,10,20,\n', steps=40, run_ms=40, name='noid',
                               transfer=lambda mosi: GOOD_MISO)
    man = json.loads(str(np.load(p)['manifest']))
    assert man['id_confirmed'] is False and man['stop_reason'] == 'max_steps' and man['steps'] == 40
    r = fc.check(p, csv_, steps)
    assert not r['ok'] and any('id_confirmed=False' in s for s in r['fails']), r['fails']
    for name, edit in (('false', lambda d, m: m.__setitem__('id_confirmed', False)),
                       ('one', lambda d, m: m.__setitem__('id_confirmed', 1)),
                       ('gone', lambda d, m: m.pop('id_confirmed'))):
        r = fc.check(resave(run[0], str(tmp_path / f'id_{name}.npz'), edit), run[1], run[2])
        assert not r['ok'] and any('id_confirmed' in s for s in r['fails']), (name, r['fails'])


def test_bad_miso_fails_even_with_clean_manifest(tmp_path):
    p, csv_, steps = open_loop(tmp_path, transfer=lambda mosi: bytes([0x10, 0x00] * 4) + bytes(FRAME_SIZE - 8))
    p2 = resave(p, str(tmp_path / 'x.npz'), lambda d, m: m.__setitem__('miso_check', dict(
        m['miso_check'], crc_bad=0, empty=0)))
    r = fc.check(p2, csv_, steps)
    assert not r['ok'] and r['miso']['crc_bad'] == 398 and any('recomputed' in s for s in r['fails'])


def test_reference_scale_exact_past_60_digits_and_parse_limit():
    """Codex FI2-1: s = 1/2 +- 10^-61 with raw = +-1: the exact (1 - s) * raw is just under / over one half,
    so the result is 0 / +-1; a 60-digit Decimal took both as 1/2. The schedule refuses s with more than 9
    decimal places and an exponent of more than 2 digits."""
    eps = Fraction(1, 10 ** 61)
    for raw in (1, -1):
        assert fc.ref_scale(raw, Fraction(1, 2) + eps) == 0 == fs.scale_raw(raw, Fraction(1, 2) + eps)
        assert fc.ref_scale(raw, Fraction(1, 2) - eps) == raw == fs.scale_raw(raw, Fraction(1, 2) - eps)
    long_s = Fraction(3, 10) + Fraction(1, 10 ** 70)
    for raw in range(-2000, 2001, 13):
        assert fc.ref_scale(raw, long_s) == fs.scale_raw(raw, long_s), raw
    for bad in ('0.3000000001', '0.5000000000000000000000000000000000000000000000000000000000001', '1e-100',
                '3e100'):
        with pytest.raises(fs.ScheduleError):
            fs.parse(HEAD + f'S1,gyro_scale,gz,0,1,s={bad}\n')
    for good, want in (('0.300000000', Fraction(3, 10)), ('3e-1', Fraction(3, 10)), ('-0.123456789', None),
                       ('0.5e-8', Fraction(5, 10 ** 9))):
        s = fs.parse(HEAD + f'S1,gyro_scale,gz,0,1,s={good}\n')[0].p('s')
        assert s == (want if want is not None else Fraction(good))


def _lc_manifest(m, trs, outcome='completed'):
    m['lifecycle'] = dict(outcome=outcome, detail=None, steps=0, transitions=trs, first_zero_step=None,
                          zeros_sent=0, dut_ready=None)
    m['stop_reason'] = outcome
    m['swa_tail_complete'] = True


def test_lifecycle_checked_before_use(tmp_path):
    """Codex FI2-2: 100 frames, phases RUN at 10 and STOPPING / END at 1000, window [200, 300) and its events
    written as executed: every frame is the true one, yet the old checker took the window as run. Phase steps
    past the log, not whole numbers, or out of order are refused."""
    p, csv_, _ = open_loop(tmp_path, body='E1,enc_loss,FL,200,300,\n', steps=100, run_ms=1000, name='lc')
    steps = write_step_log(str(tmp_path / 'lc_st.csv'), 100)

    def past_log(d, m):
        _lc_manifest(m, [dict(step=0, t=0.0, phase='ARM'), dict(step=10, t=0.0, phase='RUN'),
                         dict(step=1000, t=0.0, phase='STOPPING'), dict(step=1000, t=0.0, phase='END')])
        m['faults'].update(t0_step=10, stop_step=1000, events=[[200, 'E1', 1, 'on'], [300, 'E1', 0, 'off']],
                           not_run=[], cut=False, complete=True)
        d['mosi'][:] = d['mosi_true']
    r = fc.check(resave(p, str(tmp_path / 'past.npz'), past_log), csv_, steps)
    assert not r['ok'] and any('outside the 100 frames' in s for s in r['fails']), r['fails']
    good = [dict(step=0, t=0.0, phase='ARM'), dict(step=10, t=0.0, phase='RUN'),
            dict(step=60, t=0.0, phase='STOPPING'), dict(step=90, t=0.0, phase='END')]
    for name, trs, msg in (
            ('no_stopping', good[:2] + good[3:], 'out of order'),                          # FI3-2
            ('prepare_first', [dict(step=0, t=0.0, phase='PREPARE')] + good, 'out of order'),  # FI3-2
            ('run_first', good[1:], 'out of order'),                                          # FI3-2
            ('end_then_run', good + [dict(step=95, t=0.0, phase='RUN')], 'out of order'),     # FI3-2
            ('float', [dict(good[0]), dict(good[1], step=10.0)] + good[2:], 'bad transition'),
            ('float9', [dict(good[0]), dict(good[1], step=10.9)] + good[2:], 'bad transition'),
            ('bool', [dict(good[0], step=False)] + good[1:], 'bad transition'),
            ('phase', good[:1] + [dict(good[1], phase='GO')] + good[2:], 'bad transition'),
            ('repeat', good[:2] + [dict(good[1], step=20)] + good[2:], 'out of order'),
            ('back', good[:2] + [dict(good[2], step=5)] + good[3:], 'out of order'),
            ('nolist', None, 'no transitions')):
        r = fc.check(resave(p, str(tmp_path / f'{name}.npz'), lambda d, m: _lc_manifest(m, trs)), csv_, steps)
        assert not r['ok'] and any(msg in s for s in r['fails']), (name, r['fails'])


def test_phase_paths_of_run_lifecycle(tmp_path):
    """Codex FI3-2: the paths RunLifecycle records pass the phase check, cut ones included (ARM; ARM, RUN;
    END alone after a failed PREPARE; ARM, STOPPING, END after an ARM timeout); and the counterexample: real
    phases ARM@0, RUN@10, STOPPING@30, END@80, window [50, 60) from t0 = frames 60..69 inside STOPPING; a
    log patched there with STOPPING dropped and stop 80 written (events as run in full) passed before, now
    fails on the phases."""
    def tr(*ps):
        return dict(lifecycle=dict(transitions=[dict(step=s, t=0.0, phase=p) for p, s in ps]))
    for ps in ((('ARM', 0),), (('ARM', 0), ('RUN', 10)), (('END', 0),), (('ARM', 0), ('STOPPING', 40), ('END', 50)),
               (('ARM', 0), ('RUN', 10), ('STOPPING', 30), ('END', 80)), (('ARM', 0), ('STOPPING', 0), ('END', 0))):
        assert fc.phases_t0_stop(tr(*ps), 100)[2] is None, ps
    p, csv_, _ = open_loop(tmp_path, body='E1,enc_loss,FL,50,60,\n', steps=100, run_ms=1000, name='nostop')
    steps = write_step_log(str(tmp_path / 'nostop_st.csv'), 100)

    def drop_stopping(d, m):
        rows = fs.parse(HEAD + 'E1,enc_loss,FL,50,60,\n')
        pt = fs.FramePatcher(fs.Schedule(tuple(rows)))
        for k in range(100):
            if k == 10:
                pt.start(k)
            if k == 80:
                pt.stop(k)
            d['mosi'][k] = np.frombuffer(pt.patch(k, bytes(d['mosi_true'][k])), dtype=np.uint8)
        _lc_manifest(m, [dict(step=0, t=0.0, phase='ARM'), dict(step=10, t=0.0, phase='RUN'),
                         dict(step=80, t=0.0, phase='END')])
        m['faults'].update(t0_step=10, stop_step=80, events=[[50, 'E1', 1, 'on'], [60, 'E1', 0, 'off']],
                           not_run=[], cut=False, complete=True)
    p2 = resave(p, str(tmp_path / 'nostop2.npz'), drop_stopping)
    f = np.load(p2)
    assert np.all(f['mosi'][60:70, 0:2] == 0) and np.any(f['mosi_true'][60:70, 0:2] != 0)   # patched in "STOPPING"
    r = fc.check(p2, csv_, steps)
    assert not r['ok'] and any('out of order' in s for s in r['fails']), r['fails']


@pytest.mark.parametrize('kind', ['cut2', 'empty', 'wide', 'true_short', 't_send', 'all_narrow', 'all_wide',
                                  'scalar', 'all_u16'])
def test_frame_arrays_must_cover_the_run(run, tmp_path, kind):
    """Codex FI2-5, FI3-3, FI3-4, FI4-2: MISO cut to 2 frames (0 checked) with the manifest left at 398, MISO
    empty, MISO of another width, mosi_true or t_send_ns shorter; the three frame arrays cut together to
    FRAME_SIZE - 1 bytes (MISO CRC recomputed) or widened together; mosi a scalar; the three frame arrays
    of dtype uint16 with shape (n, FRAME_SIZE) (2 bytes an element, MISO CRC recomputed on the 96 bytes):
    cannot be checked."""
    def edit(d, m):
        if kind == 'all_u16':
            for a in ('mosi', 'mosi_true', 'miso'):
                d[a] = d[a].astype('<u2')
            d['miso'] = np.array([np.frombuffer(with_crc(bytes(r)), '<u2') for r in d['miso']])
        elif kind in ('all_narrow', 'all_wide'):
            w = FRAME_SIZE - 1 if kind == 'all_narrow' else FRAME_SIZE + 8
            for a in ('mosi', 'mosi_true', 'miso'):
                x = d[a][:, :w] if w < FRAME_SIZE else np.hstack([d[a], np.zeros((len(d[a]), 8), np.uint8)])
                d[a] = np.ascontiguousarray(x)
            d['miso'] = np.array([np.frombuffer(with_crc(bytes(r)), np.uint8) for r in d['miso']])
        elif kind == 'scalar':
            d['mosi'] = np.array(0, dtype=np.uint8)
        elif kind == 'cut2':
            d['miso'] = d['miso'][:2]
        elif kind == 'empty':
            d['miso'] = d['miso'][:0]
        elif kind == 'wide':
            d['miso'] = np.hstack([d['miso'], np.zeros((len(d['miso']), 8), dtype=np.uint8)])
        elif kind == 'true_short':
            d['mosi_true'] = d['mosi_true'][:-1]
        else:
            d['t_send_ns'] = d['t_send_ns'][:-1]
    r = fc.check(resave(run[0], str(tmp_path / f'{kind}.npz'), edit), run[1], run[2])
    assert not r['ok'] and 'frame arrays do not match' in r.get('error', ''), r


@pytest.mark.parametrize('kind', ['json_rows_only', 'miss_pm1', 'miss_2', 'dup', 'order', 'extra', 'lost_bool',
                                  'overflow_str', 'json_rows'])
def test_step_log_strict(run, tmp_path, kind):
    """Codex FI2-6: a .json without its counters, miss = 1 and -1 (sum 0), miss 2, a repeated or out of order
    step, a step past the frame log, counters of the wrong type, rows not the CSV rows: all fail."""
    p, csv_, _ = run
    sl = write_step_log(str(tmp_path / 'sl.csv'), 400)
    with open(sl) as fh:
        lines = fh.read().splitlines()
    meta = dict(rows=400, lost_or_torn=0, overflow=0)
    row = lambda k, miss: f'{k},{k * 1000.0:.3f},{k * 1000.0 + 1:.3f},{k * 1000.0 + 60:.3f},{k * 1000.0 + 61:.3f},{miss}'
    if kind == 'json_rows_only':
        meta = dict(rows=400)
    elif kind == 'miss_pm1':
        lines[1 + 5], lines[1 + 6] = row(5, 1), row(6, -1)
    elif kind == 'miss_2':
        lines[1 + 5] = row(5, 2)
    elif kind == 'dup':
        lines.insert(1 + 6, row(5, 0))
        meta['rows'] = 401
    elif kind == 'order':
        lines[1 + 5], lines[1 + 6] = lines[1 + 6], lines[1 + 5]
    elif kind == 'extra':
        lines.append(row(400, 0))
        meta['rows'] = 401
    elif kind == 'lost_bool':
        meta['lost_or_torn'] = False
    elif kind == 'overflow_str':
        meta['overflow'] = '0'
    else:
        meta['rows'] = 399
    with open(sl, 'w', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(sl + '.json', 'w') as fh:
        json.dump(meta, fh)
    r = fc.check(p, csv_, sl)
    assert not r['ok'] and any(s.startswith('step log:') for s in r['fails']), r['fails']
    assert len(r['fails']) == 1, r['fails']                 # only the step log is wrong


def test_exception_after_the_schedule_fails(tmp_path):
    """Codex FI1-4: window [10, 20), plant.step raises at step 100 (after the transfer and the frame arrays,
    before the step count): frame 100 is not in the log, the schedule ran in full, the step log of 0..99 is
    good; the run broke, so the checker refuses it."""
    csv_ = tmp_path / 'e.csv'
    csv_.write_text(HEAD + 'E1,enc_loss,FL,10,20,\n')
    plant = moving_plant()
    real, c = plant.step, [0]

    def step(*a, **k):
        if c[0] == 100:
            raise RuntimeError('plant broke')
        c[0] += 1
        return real(*a, **k)
    plant.step = step
    p = str(tmp_path / 'e.npz')
    node = HILNode(plant, config=HILConfig(simulate_hw=True, max_steps=400, frame_log=p,
                                           faults=fs.load(str(csv_), 400, 'plant')))
    node._spi.transfer = good_miso
    with pytest.raises(RuntimeError):
        node.run()
    steps = write_step_log(str(tmp_path / 'e_st.csv'), 100)
    r = fc.check(p, str(csv_), steps)
    assert r['n_frames'] == 100 and r['complete'] and r['frames_bad'] == 0
    assert not r['ok'] and r['fails'] == [m for m in r['fails'] if 'did not end normally' in m] and r['fails']


def test_truth_self_check_plant_imu_without_noise(tmp_path):
    """FI1-9: plant IMU and no noise: yaw = theta and gz = wz hold; a yaw edited in both arrays fails."""
    csv_ = tmp_path / 's.csv'
    csv_.write_text(HEAD + 'E1,enc_loss,FL,10,20,\n')
    p = str(tmp_path / 'f.npz')
    node = HILNode(moving_plant(), config=HILConfig(simulate_hw=True, max_steps=200, frame_log=p,
                                                    faults=fs.load(str(csv_), 200, 'plant')))
    node._spi.transfer = good_miso
    node.run()
    steps = write_step_log(str(tmp_path / 'st.csv'), 200)
    r = fc.check(p, str(csv_), steps)
    assert r['ok'], r['fails']
    assert r['self_check'] == dict(seq_bad=0, first_seq_bad=None, imu_checked=True, yaw_theta_bad=0, gz_wz_bad=0)

    def yaw(d, m):
        for a in ('mosi', 'mosi_true'):
            d[a][30, 37] ^= 0x01
    r = fc.check(resave(p, str(tmp_path / 'y.npz'), yaw), str(csv_), steps)
    assert not r['ok'] and r['self_check']['yaw_theta_bad'] == 1


@pytest.mark.parametrize('kind', ['none', 'miss', 'skip', 'lost', 'no_json'])
def test_step_log_required_and_checked(run, tmp_path, kind):
    p, csv_, _ = run
    if kind == 'none':
        r = fc.check(p, csv_, None)
    else:
        sl = write_step_log(str(tmp_path / 'sl.csv'), 400, miss_at=(5,) if kind == 'miss' else (),
                            skip=(7,) if kind == 'skip' else (), lost=1 if kind == 'lost' else 0)
        if kind == 'no_json':
            os.remove(sl + '.json')
        r = fc.check(p, csv_, sl)
    assert not r['ok'] and any('step log' in s for s in r['fails'])


def test_real_step_log_format_is_read(tmp_path):
    """The checker reads the CSV that step_log.StepLogCSV writes (not only the synthetic one)."""
    from step_log import StepLog, StepLogCSV
    csv_ = tmp_path / 's.csv'
    csv_.write_text(HEAD + 'E1,enc_loss,FL,10,20,\n')
    p = str(tmp_path / 'f.npz')
    sl = StepLog(n_slots=1024)
    node = HILNode(moving_plant(), config=HILConfig(simulate_hw=True, max_steps=100, frame_log=p,
                                                    faults=fs.load(str(csv_), 100, 'plant')), step_log=sl)
    node._spi.transfer = good_miso
    node.run()
    w = StepLogCSV(sl, str(tmp_path / 'st.csv'))
    meta = w.close()
    with open(str(tmp_path / 'st.csv') + '.json', 'w') as fh:
        json.dump(meta, fh)
    t = fc.step_log_check(str(tmp_path / 'st.csv'), 100)
    assert t['rows'] == 100 and t['steps_missing'] == 0 and t['json'] and t['lost_or_torn'] == 0


class _Owner(threading.Thread):
    """Heartbeat of the run owner; ends RUN (publisher done) `after` s into it."""

    def __init__(self, box, after):
        super().__init__(daemon=True)
        self.box, self.after, self.quit, self.t_run = box, after, False, None

    def run(self):
        while not self.quit:
            self.box.beat(time.monotonic_ns())
            if self.box.phase() == RUN:
                self.t_run = self.t_run or time.monotonic()
                if time.monotonic() - self.t_run >= self.after:
                    self.box.publisher_done()
            time.sleep(0.002)


def finite_run(tmp_path, body, name='f', after=0.06):
    """A --finite-run of the 1 kHz loop (simulate_hw, fake H7 with CRC), its schedule and step log."""
    from fake_h7 import FakeH7
    csv_ = tmp_path / f'{name}.csv'
    csv_.write_text(HEAD + body)
    sched = fs.load(str(csv_), 150, 'static')
    p = str(tmp_path / f'{name}.npz')
    box = ControlBox()
    box.beat(time.monotonic_ns())
    fake = FakeH7()
    shared_ref = mp.Array('d', 4)
    shared_ref[:] = [0.2, 0.0, 0.0, 1.0]
    node = HILNode(MecanumPlant(dt=0.001, seed=0),
                   HILConfig(simulate_hw=True, max_steps=3000, frame_log=p, frame_log_max=3000, swa_mode='phase',
                             imu_static=(0.5, 0.1, 0.2, 9.81, 0.3), faults=sched,
                             lifecycle=LifecycleConfig(run_s=0.15, arm_h7_fresh_s=0.12, stop_s=0.06,
                                                       arm_timeout_s=0.5, heartbeat_max_s=0.1, h7_gap_max_s=0.05,
                                                       run_grace_s=0.1)),
                   shared_ref=shared_ref, control=box, responder=lambda m: with_crc(fake(m)))
    owner = _Owner(box, after)
    owner.start()
    node.run()
    owner.quit = True
    owner.join()
    n = len(np.load(p)['mosi'])
    return p, str(csv_), write_step_log(str(tmp_path / f'{name}_st.csv'), n), node


def test_finite_run_must_end_normally(tmp_path):
    """Codex FI1-4 with phases: a run that reaches END with its tail and runs the window in full passes;
    an error outcome, a cut END tail, a stop_reason rewritten after END (error, signal), or no END fail."""
    p, csv_, steps, node = finite_run(tmp_path, 'L1,imu_loss,imu,5,25,\n', 'n')
    assert not node.run_failed
    r = fc.check(p, csv_, steps)
    assert r['ok'], r['fails']
    for name, edit in (
            ('outcome', lambda d, m: (m['lifecycle'].__setitem__('outcome', 'error:heartbeat'),
                                      m.__setitem__('stop_reason', 'error:heartbeat'))),
            ('tail', lambda d, m: m.__setitem__('swa_tail_complete', False)),
            ('error', lambda d, m: m.__setitem__('stop_reason', 'error:RuntimeError')),
            ('signal', lambda d, m: m.__setitem__('stop_reason', 'signal_15')),
            ('no_end', lambda d, m: m['lifecycle']['transitions'].pop())):
        r = fc.check(resave(p, str(tmp_path / f'n_{name}.npz'), edit), csv_, steps)
        assert not r['ok'] and any('did not end normally' in s for s in r['fails']), (name, r['fails'])


def test_finite_run_t0_and_stop_from_phases(tmp_path):
    """FI1-2: with phases, t0 and stop come from the lifecycle transitions; a run whose RUN ends inside a
    window passes the byte check but is not complete; a manifest stop moved by one fails."""
    from fake_h7 import FakeH7
    csv_ = tmp_path / 's.csv'
    csv_.write_text(HEAD + 'L1,imu_loss,imu,20,140,\n')
    sched = fs.load(str(csv_), 150, 'static')
    p = str(tmp_path / 'f.npz')
    box = ControlBox()
    box.beat(time.monotonic_ns())
    fake = FakeH7()
    shared_ref = mp.Array('d', 4)
    shared_ref[:] = [0.2, 0.0, 0.0, 1.0]
    node = HILNode(MecanumPlant(dt=0.001, seed=0),
                   HILConfig(simulate_hw=True, max_steps=3000, frame_log=p, frame_log_max=3000, swa_mode='phase',
                             imu_static=(0.5, 0.1, 0.2, 9.81, 0.3), faults=sched,
                             lifecycle=LifecycleConfig(run_s=0.15, arm_h7_fresh_s=0.12, stop_s=0.06,
                                                       arm_timeout_s=0.5, heartbeat_max_s=0.1, h7_gap_max_s=0.05,
                                                       run_grace_s=0.1)),
                   shared_ref=shared_ref, control=box, responder=lambda m: with_crc(fake(m)))
    owner = _Owner(box, 0.06)
    owner.start()
    node.run()
    owner.quit = True
    owner.join()
    n = len(np.load(p)['mosi'])
    steps = write_step_log(str(tmp_path / 'st.csv'), n)
    r = fc.check(p, str(csv_), steps)
    man = json.loads(str(np.load(p)['manifest']))
    tr = man['lifecycle']['transitions']
    k_run = next(t['step'] for t in tr if t['phase'] == 'RUN')
    assert r['t0_step'] == k_run and r['stop_step'] is not None and r['frames_bad'] == 0
    assert not r['complete'] and [m for m in r['fails'] if 'did not run in full' in m]
    r2 = fc.check(resave(p, str(tmp_path / 'g.npz'),
                         lambda d, m: m['faults'].__setitem__('stop_step', m['faults']['stop_step'] + 1)),
                  str(csv_), steps)
    assert any('stop_step' in s for s in r2['fails'])


def test_run_without_faults_cannot_be_checked(tmp_path):
    p = str(tmp_path / 'f.npz')
    node = HILNode(MecanumPlant(dt=0.001, seed=0), config=HILConfig(simulate_hw=True, max_steps=20, frame_log=p))
    node.run()
    csv_ = tmp_path / 's.csv'
    csv_.write_text(HEAD + 'E1,enc_loss,FL,1,5,\n')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'fault_check.py'), p, '--faults', str(csv_)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and 'mosi_true' in r.stderr


def test_cli_exit_codes(run, tmp_path):
    p, csv_, steps = run
    tool = os.path.join(ROOT, 'tools', 'fault_check.py')
    r = subprocess.run([sys.executable, tool, p, '--faults', csv_, '--step-log', steps],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads((tmp_path / 'fault_check.json').read_text())
    assert res['ok'] and res['frames_injected'] > 0
    r = subprocess.run([sys.executable, tool, p, '--faults', csv_], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and 'no step log' in r.stdout
    # Codex FI3-4: a scalar mosi is "cannot be checked" (exit 2), not a traceback
    p2 = resave(p, str(tmp_path / 'scalar.npz'), lambda d, m: d.__setitem__('mosi', np.array(0, dtype=np.uint8)))
    r = subprocess.run([sys.executable, tool, p2, '--faults', csv_, '--step-log', steps],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and 'Traceback' not in r.stderr, r.stderr


# ── batch 2: ibus_loss, bit 4 of byte 20 (design 2026-11-15 section 7.1 item 1) ─
IBUS_BODY = 'I1,ibus_loss,ibus,60,180,\nL1,imu_loss,imu,150,220,\nE1,enc_loss,RL,170,190,\nI2,ibus_loss,ibus,300,330,\n'


def test_ibus_patcher_agrees_with_independent_reference():
    true = frames(400, seed=5)
    rows = fs.parse(HEAD + IBUS_BODY)
    fs.check(rows, 400, 'plant')
    for t0, stop in ((0, None), (13, None), (13, 170), (13, 320)):
        pa = fs.FramePatcher(fs.Schedule(tuple(rows)))
        for k, f in enumerate(true):
            if k == t0:
                pa.start(k)
            if stop is not None and k == stop:
                pa.stop(k)
            assert pa.patch(k, f) == fc.expected_sent(rows, k, t0, stop, true), (t0, stop, k)
    exp = fc.expected_sent(rows, 160, 0, None, true)
    assert exp[20] == true[160][20] | 16 | 4 and true[160][20] == 8          # SWA kept, IMU loss and iBUS off


def test_ibus_run_passes_and_bit4_tampering_fails(tmp_path):
    run_ = open_loop(tmp_path, body=IBUS_BODY, name='ibus')
    r = fc.check(*run_)
    assert r['ok'], r['fails']
    sent = np.load(run_[0])['mosi'][:, 20]
    assert np.flatnonzero(sent & 16).tolist() == list(range(60, 180)) + list(range(300, 330))
    for name, k, op in (('missing_in', 100, lambda v: v & ~16), ('late_end', 180, lambda v: v | 16),
                        ('early', 59, lambda v: v | 16), ('second', 329, lambda v: v & ~16)):
        def edit(d, m, k=k, op=op):
            d['mosi'][k, 20] = op(int(d['mosi'][k, 20]))
        q = resave(run_[0], str(tmp_path / f'{name}.npz'), edit)
        rr = fc.check(q, run_[1], run_[2])
        assert not rr['ok'] and rr['frames_bad'] == 1, name
