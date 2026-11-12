"""L95 (Liam 12/11 "SWB + ghi log"; 13/11 "Dau luot + khi doi + cuoi"): host test of the firmware
module src/ultrasonic/us_state.c (SWB switches obstacle detection in every mode; per-run
'#us' / '#us_end' log lines). Compiles fwsim/tests/us_state_check.c with that file, runs it, and
requires every check to pass.

Firmware tree: $FWSIM_FW, else the default of build_fwsim.py (../AGV_FIRMWARE next to this repo).
Skipped when that tree has no us_state.c or gcc is missing.
Run: python -m pytest fwsim/tests/test_us_state.py"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FWSIM = os.path.dirname(HERE)
FW = os.path.abspath(os.environ.get('FWSIM_FW', os.path.join(FWSIM, '..', '..', 'AGV_FIRMWARE')))
US = os.path.join(FW, 'src', 'ultrasonic')

pytestmark = [
    pytest.mark.skipif(not os.path.isfile(os.path.join(US, 'us_state.c')),
                       reason=f'no src/ultrasonic/us_state.c in {FW}'),
    pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not found'),
]


def test_us_state_checks(tmp_path):
    exe = str(tmp_path / 'us_state_check.exe')
    cmd = ['gcc', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-I', US,
           os.path.join(US, 'us_state.c'), os.path.join(HERE, 'us_state_check.c'), '-o', exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f'gcc failed:\n{r.stderr}'
    assert 'us_state.c' not in r.stderr, f'warnings in us_state.c:\n{r.stderr}'
    run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    lines = run.stdout.splitlines()
    fails = [l for l in lines if l.startswith('FAIL')]
    assert run.returncode == 0 and not fails, '\n'.join(fails) or run.stdout + run.stderr
    assert sum(l.startswith('PASS') for l in lines) >= 25, run.stdout
