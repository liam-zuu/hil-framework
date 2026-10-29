"""S06 T10 H7 load accounting (stm32h7/Core/Src/s06_load.c) on the PC: nested
interrupts played as nested calls with a hand-set cycle counter
(stm32h7/pc_tests/s06_load/s06_load_sim.c), so exclusive times are exact.
Not covered: the real interrupt entry/exit cost and the effect of the masked
bookkeeping on latency (board measurements, spec S06 T10).
Run: python -m pytest tools/tests  (needs gcc on PATH)"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
H7 = os.path.join(ROOT, 'stm32h7')
SIM = os.path.join(H7, 'pc_tests', 's06_load')
pytestmark = pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not on PATH')

CHECKS = ['outer_exclusive', 'middle_exclusive', 'inner_exclusive', 'sibling_exclusive',
          'primask_restored', 'wrap_inside_handler', 'count_sum_max', 'report_restarts_window',
          'idle_counted', 'report_restores_primask']


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp('s06_load')
    for f in (os.path.join(SIM, 'stm32h7xx_hal.h'), os.path.join(SIM, 's06_load_sim.c'),
              os.path.join(H7, 'Core', 'Inc', 's06_load.h'), os.path.join(H7, 'Core', 'Src', 's06_load.c')):
        shutil.copy(f, d)
    exe = os.path.join(d, 's06_load_sim.exe')
    r = subprocess.run(['gcc', '-std=gnu11', '-O2', '-Wall', '-Werror', '-DS06_PC_TEST', '-I', str(d),
                        os.path.join(d, 's06_load_sim.c'), os.path.join(d, 's06_load.c'), '-o', exe],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([exe], capture_output=True, text=True)
    res = {}
    for line in r.stdout.splitlines():
        status, _, rest = line.partition(' ')
        if status in ('PASS', 'FAIL'):
            res[rest.split(':')[0]] = status
    return r.returncode, res, r.stdout


@pytest.mark.parametrize('name', CHECKS)
def test_check(result, name):
    _, res, out = result
    assert res.get(name) == 'PASS', out


def test_exit_code(result):
    code, _, out = result
    assert code == 0, out
