"""tools/fault_check.py (L82, design 8.1): the independent reference against the Pi patcher, and the check
on real frame logs of the 1 kHz loop (simulate_hw): a good run passes; a frame injected one step late, the
wrong wheel, a window one frame too long, a touched seq byte, a wrong file, missing events, a cut window and
a bad MISO all fail. Run: python -m pytest tools/tests"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import fault_check as fc                       # noqa: E402
import fault_schedule as fs                    # noqa: E402
from hil_node import HILConfig, HILNode        # noqa: E402
from mecanum_plant import MecanumPlant         # noqa: E402
from protocol import encode_state              # noqa: E402

HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
BODY = ('E1,enc_loss,FR,50,120,\nL1,imu_loss,imu,100,200,\nD1,imu_delay,imu,220,300,d_ms=20\n'
        'S1,gyro_scale,gz,250,350,s=0.3\n')


def moving_plant():
    """A plant that turns from its first step (wheels and body moving), so that every fault changes bytes."""
    p = MecanumPlant(dt=0.001, seed=0)
    p.reset([0, 0, 0, 0.2, 0, 0.8, 10, 12, 14, 16])
    return p


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
                           'S1,gyro_scale,gz,400,800,s=-0.5\nL1,imu_loss,imu,650,1000,\nE2,enc_loss,FR,850,1100,\n')
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
        assert fc._norm_events(sm['events']) == fc._norm_events(ev) and sorted(sm['not_run']) == sorted(nr)


@pytest.fixture
def run(tmp_path):
    """A real open-loop run of the node with the schedule file: (npz path, csv path)."""
    csv = tmp_path / 's.csv'
    csv.write_text(HEAD + BODY)
    sched = fs.load(str(csv), 400, 'plant')
    p = str(tmp_path / 'frames.npz')
    node = HILNode(moving_plant(),
                   config=HILConfig(simulate_hw=True, max_steps=400, frame_log=p, faults=sched))
    node._spi.transfer = lambda mosi: bytes([0x10, 0x00] * 4) + bytes(32)
    node.run()
    return p, str(csv)


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
    # the simulated SPI answers a constant MISO: the L50 MISO check fails, nothing else
    assert [m for m in r['fails'] if not m.startswith('MISO')] == []
    assert 150 + 80 <= r['frames_injected'] <= 150 + 130  # [50, 200), [220, 300); [300, 350) where gz != 0
    assert r['frames_bad'] == 0 and r['complete'] and r['t0_step'] == 0


def _miso_ok(d, man):
    man['miso_check'] = dict(man['miso_check'], crc_bad=0, empty=0)


@pytest.mark.parametrize('name,edit', [
    ('late', lambda d, m: d['mosi'].__setitem__(50, d['mosi_true'][50])),            # window starts 1 late
    ('wheel', lambda d, m: (d['mosi'].__setitem__((60, slice(2, 4)), d['mosi_true'][60, 2:4]),
                            d['mosi'].__setitem__((60, slice(0, 2)), 0))),             # FL instead of FR
    ('long', lambda d, m: d['mosi'].__setitem__((120, slice(2, 4)), 0)),               # one frame too long
    ('seq', lambda d, m: d['mosi'].__setitem__((110, 21), (int(d['mosi'][110, 21]) + 1) % 256)),
    ('events', lambda d, m: m['faults'].__setitem__('events', m['faults']['events'][:-1])),
    ('t0', lambda d, m: m['faults'].__setitem__('t0_step', 1)),
])
def test_tampered_run_fails(run, tmp_path, name, edit):
    p = resave(run[0], str(tmp_path / f'{name}.npz'), lambda d, m: (_miso_ok(d, m), edit(d, m)))
    ok = resave(run[0], str(tmp_path / 'ok.npz'), _miso_ok)
    assert fc.check(ok, run[1])['ok']
    r = fc.check(p, run[1])
    assert not r['ok'], name


def test_other_schedule_file_fails(run, tmp_path):
    other = tmp_path / 'o.csv'
    other.write_text(HEAD + BODY.replace('s=0.3', 's=0.9'))
    p = resave(run[0], str(tmp_path / 'ok.npz'), _miso_ok)
    r = fc.check(p, str(other))
    assert not r['ok'] and any('sha256' in m for m in r['fails']) and r['frames_bad'] > 0


def test_cut_window_fails_even_if_bytes_agree(tmp_path):
    csv = tmp_path / 's.csv'
    csv.write_text(HEAD + 'E1,enc_loss,FL,300,500,\n')
    p = str(tmp_path / 'f.npz')
    node = HILNode(MecanumPlant(dt=0.001, seed=0),
                   config=HILConfig(simulate_hw=True, max_steps=400, frame_log=p,
                                    faults=fs.load(str(csv), 500, 'plant')))
    node._spi.transfer = lambda mosi: bytes(40)
    node.run()
    p2 = resave(p, str(tmp_path / 'ok.npz'), _miso_ok)
    r = fc.check(p2, str(csv))
    assert r['frames_bad'] == 0 and not r['complete'] and not r['ok']
    assert r['events'][-1] == [400, 'E1', 0, 'cut']


def test_run_without_faults_cannot_be_checked(tmp_path):
    p = str(tmp_path / 'f.npz')
    node = HILNode(MecanumPlant(dt=0.001, seed=0), config=HILConfig(simulate_hw=True, max_steps=20, frame_log=p))
    node.run()
    csv = tmp_path / 's.csv'
    csv.write_text(HEAD + 'E1,enc_loss,FL,1,5,\n')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'fault_check.py'), p, '--faults', str(csv)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and 'mosi_true' in r.stderr


def test_cli_exit_codes(run, tmp_path):
    ok = resave(run[0], str(tmp_path / 'ok.npz'), _miso_ok)
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'fault_check.py'), ok, '--faults', run[1]],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads((tmp_path / 'fault_check.json').read_text())
    assert res['ok'] and res['frames_injected'] > 0
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'fault_check.py'), run[0], '--faults', run[1]],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and 'MISO' in r.stdout
