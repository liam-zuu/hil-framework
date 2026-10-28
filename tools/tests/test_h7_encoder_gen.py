"""H7 EncoderGen (stm32h7/Core/Src/encoder_gen.c) on the PC: the unchanged C file against
fake 16-bit timers, 1 us ticks, random ISR latency, x4 decode like the ESP32 PCNT
(stm32h7/pc_tests/encgen/encgen_sim_main.c).

Bug H7-ENC-RETIME (28/10/2026, model 05_DATA/results/ENC_SCHED_MODEL/2026-10-28_run01):
before the fix a same-direction speed change only changed the half period, so (1) the
next edge after a speed-up came one OLD half period later and (2) the A/B offset stayed
half the old half period, so the ESP32 could decode the wheel backwards after a fast
speed-up or a start from rest. The same scenarios run on the old file (h7-relay-chk
cb91e7a) to show the test sees it.
Run: python -m pytest tools/tests  (needs gcc and git on PATH)"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
H7 = os.path.join(ROOT, 'stm32h7')
SIM = os.path.join(H7, 'pc_tests', 'encgen')
OLD_REV = 'cb91e7a'          # h7-relay-chk, encoder_gen.c before the fix
pytestmark = pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not on PATH')

SMOOTH = ['step_up', 'ratio2_5', 'startup_slow', 'startup_fast', 'ramp_down', 'sine']
RESTART = ['reverse', 'stop_start']        # direction change / stop use _stop + _start_wheel
SEEDS = [1, 2, 3]
COUNT_TOL = 6                              # counts, as the Python model of the fix
RESTART_TOL = 10                           # forcing both pins low at a restart costs a few counts


def build(tmp, src_c):
    """Copy next to the fake HAL header: a quoted #include looks in the including file's folder first."""
    for f in (os.path.join(H7, 'Core', 'Inc', 'encoder_gen.h'), os.path.join(SIM, 'stm32h7xx_hal.h'),
              os.path.join(SIM, 'encgen_sim_main.c')):
        shutil.copy(f, tmp)
    shutil.copy(src_c, os.path.join(tmp, 'encoder_gen.c'))
    exe = os.path.join(tmp, 'encgen_sim.exe')
    r = subprocess.run(['gcc', '-std=gnu11', '-O2', '-Wall', '-Werror', '-I', str(tmp),
                        os.path.join(tmp, 'encgen_sim_main.c'), os.path.join(tmp, 'encoder_gen.c'),
                        '-lm', '-o', exe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


@pytest.fixture(scope='module')
def sim_new(tmp_path_factory):
    return build(tmp_path_factory.mktemp('encgen_new'), os.path.join(H7, 'Core', 'Src', 'encoder_gen.c'))


@pytest.fixture(scope='module')
def sim_old(tmp_path_factory):
    d = tmp_path_factory.mktemp('encgen_old')
    src = subprocess.run(['git', 'show', f'{OLD_REV}:stm32h7/Core/Src/encoder_gen.c'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    p = os.path.join(d, 'encoder_gen_old.c')
    with open(p, 'w', newline='\n') as f:
        f.write(src)
    return build(d, p)


def run(exe, scenario, seed):
    out = subprocess.run([exe, scenario, str(seed)], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        _, w, count, ideal, rev, wait_ex, _sec = line.split()
        rows.append(dict(wheel=int(w), count=int(count), ideal=float(ideal), rev=int(rev), wait_ex=int(wait_ex)))
    assert len(rows) == 4
    return rows


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('scenario', SMOOTH)
def test_fixed_counts_and_direction(sim_new, scenario, seed):
    for r in run(sim_new, scenario, seed):
        assert r['rev'] == 0, r
        assert abs(r['count'] - r['ideal']) <= COUNT_TOL, r


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('scenario', SMOOTH)
def test_fixed_next_edge_within_new_half_period(sim_new, scenario, seed):
    """S06 T3 / review #2: after a speed-up the next edge comes within one new half period."""
    for r in run(sim_new, scenario, seed):
        assert r['wait_ex'] <= 0, r


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('scenario', RESTART)
def test_fixed_restarts_unchanged(sim_new, sim_old, scenario, seed):
    """Direction change and stop/start go through _stop + _start_wheel, not changed by the fix."""
    for new, old in zip(run(sim_new, scenario, seed), run(sim_old, scenario, seed)):
        assert abs(new['count'] - new['ideal']) <= RESTART_TOL, new
        assert new['count'] == old['count'] and new['rev'] == old['rev'], (new, old)


@pytest.mark.parametrize('scenario', ['step_up', 'startup_slow'])
def test_old_code_shows_the_bug(sim_old, scenario):
    rows = [r for s in SEEDS for r in run(sim_old, scenario, s)]
    assert any(r['rev'] > 0 and abs(r['count'] - r['ideal']) > 100 for r in rows), rows
    if scenario == 'step_up':
        assert max(r['wait_ex'] for r in rows) > 0
