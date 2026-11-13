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
from protocol import FRAME_SIZE, OFS_MISO_CRC, encode_state   # noqa: E402
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


GOOD_MISO = with_crc(bytes([0x10, 0x00] * 4) + bytes(FRAME_SIZE - 8))


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
    node._spi.transfer = transfer or (lambda mosi: GOOD_MISO)
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


def _shift_late(d, m):
    """The patcher one step late everywhere, with t0 = 1 in its own manifest (FI1-2)."""
    d['mosi'][1:] = d['mosi'][:-1].copy()
    d['mosi'][:, 21] = d['mosi_true'][:, 21]
    m['faults']['t0_step'] = 1


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
    ('miso_frame', lambda d, m: d['miso'].__setitem__((40, 0), 0x55)),
    ('true_seq', lambda d, m: (d['mosi_true'].__setitem__((5, 21), 99), d['mosi'].__setitem__((5, 21), 99))),
])
def test_tampered_run_fails(run, tmp_path, name, edit):
    p = resave(run[0], str(tmp_path / f'{name}.npz'), edit)
    r = fc.check(p, run[1], run[2])
    assert not r['ok'], name


def test_bad_miso_fails_even_with_clean_manifest(tmp_path):
    p, csv_, steps = open_loop(tmp_path, transfer=lambda mosi: bytes([0x10, 0x00] * 4) + bytes(FRAME_SIZE - 8))
    p2 = resave(p, str(tmp_path / 'x.npz'), lambda d, m: m.__setitem__('miso_check', dict(
        m['miso_check'], crc_bad=0, empty=0)))
    r = fc.check(p2, csv_, steps)
    assert not r['ok'] and r['miso']['crc_bad'] == 398 and any('recomputed' in s for s in r['fails'])


def test_truth_self_check_plant_imu_without_noise(tmp_path):
    """FI1-9: plant IMU and no noise: yaw = theta and gz = wz hold; a yaw edited in both arrays fails."""
    csv_ = tmp_path / 's.csv'
    csv_.write_text(HEAD + 'E1,enc_loss,FL,10,20,\n')
    p = str(tmp_path / 'f.npz')
    node = HILNode(moving_plant(), config=HILConfig(simulate_hw=True, max_steps=200, frame_log=p,
                                                    faults=fs.load(str(csv_), 200, 'plant')))
    node._spi.transfer = lambda mosi: GOOD_MISO
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
    node._spi.transfer = lambda mosi: GOOD_MISO
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
