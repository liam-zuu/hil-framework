"""L95 (Liam 12/11 "SWB + ghi log"; 13/11 "Dau luot + khi doi + cuoi"; Codex Sol round 1 fixes, Liam 13/11
"Sua ca 8 diem + vong 2"): host test of the firmware modules src/ultrasonic/us_state.c (SWB switches
obstacle detection in every mode; per-run '#us' / '#us_end' lines with boot and raw SWB) and
src/ibus/ibus_age.h (age of the last valid iBUS frame). Compiles fwsim/tests/us_state_check.c with
them, runs it, and requires every check to pass. main.c (ESP-IDF) is not compiled here; its wiring is
checked on the board (#us lines against the data lines of a bench run).

Firmware tree: $FWSIM_FW, else the default of build_fwsim.py (../AGV_FIRMWARE next to this repo). The
firmware HEAD is printed. Skipped when that tree has no us_state.c or gcc is missing, unless
FWSIM_REQUIRE=1 (acceptance runs): then those are failures (Codex L95-7).
Run: python -m pytest fwsim/tests/test_us_state.py -s"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FWSIM = os.path.dirname(HERE)
FW = os.path.abspath(os.environ.get('FWSIM_FW', os.path.join(FWSIM, '..', '..', 'AGV_FIRMWARE')))
US = os.path.join(FW, 'src', 'ultrasonic')
IB = os.path.join(FW, 'src', 'ibus')
REQUIRE = os.environ.get('FWSIM_REQUIRE') == '1'


def _missing():
    if not os.path.isfile(os.path.join(US, 'us_state.c')):
        return f'no src/ultrasonic/us_state.c in {FW}'
    if not os.path.isfile(os.path.join(IB, 'ibus_age.h')):
        return f'no src/ibus/ibus_age.h in {FW}'
    if shutil.which('gcc') is None:
        return 'gcc not found'
    return None


def test_us_state_checks(tmp_path):
    why = _missing()
    if why:
        if REQUIRE:
            pytest.fail(why + ' (FWSIM_REQUIRE=1)')
        pytest.skip(why)
    head = subprocess.run(['git', '-C', FW, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', FW, 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True)
    print(f'firmware tree {FW} HEAD {head.stdout.strip()} dirty={int(bool(dirty.stdout.strip()))}')
    exe = str(tmp_path / 'us_state_check.exe')
    cmd = ['gcc', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-I', US, '-I', IB,
           os.path.join(US, 'us_state.c'), os.path.join(HERE, 'us_state_check.c'), '-o', exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f'gcc failed:\n{r.stderr}'
    assert 'us_state.c' not in r.stderr and 'ibus_age.h' not in r.stderr, f'warnings:\n{r.stderr}'
    run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    lines = run.stdout.splitlines()
    fails = [l for l in lines if l.startswith('FAIL')]
    assert run.returncode == 0 and not fails, '\n'.join(fails) or run.stdout + run.stderr
    assert sum(l.startswith('PASS') for l in lines) >= 32, run.stdout
