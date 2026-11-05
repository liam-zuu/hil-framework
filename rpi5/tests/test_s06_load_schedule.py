"""S06 V7, Z8: load schedule file, heavy block on the omega of the frame while the plant runs its full
dynamics, publisher sequence, run_hil.py --load-schedule. No hardware (simulate_hw=True)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from hil_node import HILConfig, HILNode
from mecanum_plant import MecanumPlant
from protocol import decode_state
import s06_load_schedule as ls
from trajectory_pub import GROUP_A, get_body_velocity

HERE = os.path.dirname(os.path.abspath(__file__))


def test_schedule_contiguous_and_heavy_row():
    rows = ls.make_schedule(610)
    ls.check(rows)
    tests = [r for r in rows if r.kind == 'test']
    assert tests[0].start_ms == 0 and tests[-1].end_ms == 610_000
    assert all(a.end_ms == b.start_ms for a, b in zip(tests, tests[1:]))
    assert [r.test for r in tests[:len(GROUP_A)]] == list(GROUP_A)        # locked order
    assert tests[len(GROUP_A)].test == list(GROUP_A)[0]                   # the sequence repeats
    assert ls.heavy_span(rows) == (10_000, 70_000)                        # Z8: start of the scored part
    assert ls.total_steps(rows) == 610_000


def test_schedule_without_heavy_and_skip():
    rows = ls.make_schedule(30, heavy_s=0.0, order=['A-L1', 'A-S1'])
    assert ls.heavy_span(rows) is None
    assert {r.test for r in rows} == {'A-L1', 'A-S1'}


@pytest.mark.parametrize('kw', [dict(total_s=30), dict(total_s=610, order=['A-X9']), dict(total_s=610, order=[])])
def test_schedule_refuses(kw):
    with pytest.raises(ValueError):
        ls.make_schedule(**kw)


def test_csv_round_trip(tmp_path):
    p = str(tmp_path / 's.csv')
    rows = ls.make_schedule(120)
    ls.write_csv(p, rows)
    assert ls.read_csv(p) == rows


@pytest.mark.parametrize('edit', ['gap', 'test', 'heavy_out', 'two_heavy', 'kind'])
def test_check_refuses_edited_file(edit):
    rows = ls.make_schedule(120)
    tests = [r for r in rows if r.kind == 'test']
    if edit == 'gap':
        tests[1].start_ms += 1
    elif edit == 'test':
        tests[0].test = 'A-Z'
    elif edit == 'heavy_out':
        rows[-1].end_ms = 999_999
    elif edit == 'two_heavy':
        rows.append(ls.Row(99, 'heavy', '', 0, 1))
    else:
        rows[-1].kind = 'other'
    with pytest.raises(ValueError):
        ls.check(rows)


def test_heavy_omegas_z8():
    t = np.array([ls.heavy_omegas(k) for k in range(2000)])
    assert set(np.unique(t)) == {30.0, 39.2}
    d = np.diff(t, axis=0) != 0
    for w in range(4):                                 # every wheel changes every 50 ms
        assert set(np.diff(np.nonzero(d[:, w])[0])) == {50}
    assert d.sum(axis=1).max() == 1                    # phase shift: never two wheels in one frame


def test_block_changes_frame_only_plant_keeps_dynamics(tmp_path):
    """Z8: same torque sequence with and without the block: plant trajectory identical, frames differ
    only in the wheel speeds and only inside the block."""
    steps = 400

    def run(block):
        plant = MecanumPlant(dt=0.001, seed=0)
        p = str(tmp_path / f'f{bool(block)}.npz')
        cfg = HILConfig(simulate_hw=True, max_steps=steps, period_us=1_000, frame_log=p, heavy_block=block,
                        load_schedule='s.csv abc')
        node = HILNode(plant, config=cfg)
        node._spi.transfer = lambda mosi: bytes([0x10, 0x00] * 4) + bytes(32)   # constant duty
        node.run()
        return plant.get_state(), np.load(p)

    s0, f0 = run(None)
    s1, f1 = run((130, 300))
    assert s0 == s1
    m0, m1 = f0['mosi'], f1['mosi']
    assert np.array_equal(m0[:130], m1[:130]) and np.array_equal(m0[300:], m1[300:])
    assert np.array_equal(m0[130:300, 8:], m1[130:300, 8:])       # everything but the 4 wheel speeds
    for k in (130, 179, 180, 299):                                # block time counts from its first step
        assert decode_state(bytes(m1[k]))[:4] == pytest.approx(ls.heavy_omegas(k - 130), abs=40.0 / 32767)
    man = json.loads(str(f1['manifest']))
    assert man['plant'] == 'MecanumPlant' and man['heavy_block'] == [130, 300] and man['load_schedule'] == 's.csv abc'


def test_publisher_sequence():
    rows = ls.make_schedule(60, heavy_s=0.0, order=['A-L1', 'A-S1'])
    spec = ls.trajectory_spec(rows)
    assert get_body_velocity(1.0, spec) == (0.2, 0.0, 0.0)                # A-L1, v = 0,2
    assert get_body_velocity(15.5, spec) == (0.0, 0.2, 0.0)               # A-S1 after 15 s
    assert get_body_velocity(25.5, spec) == (0.2, 0.0, 0.0)               # A-L1 again
    assert get_body_velocity(61.0, spec) == (0.0, 0.0, 0.0)               # after the schedule
    # each test runs on its own clock: A-K1 is at v only 3..6 s after its own start
    rows = ls.make_schedule(30, heavy_s=0.0, order=['A-L1', 'A-K1'])
    assert get_body_velocity(19.0, ls.trajectory_spec(rows)) == (0.3, 0.0, 0.0)


@pytest.mark.parametrize('extra', [['--traj', 'line'], ['--test', 'A-L1'], ['--no-traj'], ['--traj-loop'],
                                   ['--fixed-omega', '1,1,1,1'], ['--omega-steps', '1,1,1,1:1'], ['--steps', '5']])
def test_run_hil_refuses_conflict(tmp_path, extra):
    p = str(tmp_path / 's.csv')
    ls.write_csv(p, ls.make_schedule(120))
    r = subprocess.run([sys.executable, os.path.join(HERE, '..', 'run_hil.py'), '--mock', '--load-schedule', p] + extra,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and ('--load-schedule' in r.stderr or 'differs from the schedule' in r.stderr)
