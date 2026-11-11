"""Gain over WiFi (L90, Liam 12/11): host test of the firmware module src/controller/gain_set.c
(packet parsing, safety ranges, apply = the same field writes as fwsim_set_pid / fwsim_set_adrc with
tune_ga.gains_for, read-back, %.9g round trip). Compiles fwsim/tests/gain_set_check.c with the fwsim
firmware sources and shim (build_fwsim.py lists), runs it, and requires every check to pass.

Firmware tree: $FWSIM_FW, else the default of build_fwsim.py (../AGV_FIRMWARE next to this repo).
Skipped when that tree has no gain_set.c or gcc is missing.
Run: python -m pytest fwsim/tests/test_gain_set.py"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FWSIM = os.path.dirname(HERE)
sys.path.insert(0, FWSIM)
import build_fwsim  # noqa: E402

FW = os.path.abspath(os.environ.get('FWSIM_FW', os.path.join(FWSIM, '..', '..', 'AGV_FIRMWARE')))
SRC = os.path.join(FW, 'src')

pytestmark = [
    pytest.mark.skipif(not os.path.isfile(os.path.join(SRC, 'controller', 'gain_set.c')),
                       reason=f'no src/controller/gain_set.c in {FW}'),
    pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not found'),
]


def test_gain_set_checks(tmp_path):
    sources = build_fwsim.FW_SOURCES + [f for f in build_fwsim.FW_OPTIONAL if os.path.isfile(os.path.join(SRC, f))]
    sources.append('controller/gain_set.c')
    exe = str(tmp_path / 'gain_set_check.exe')
    cmd = (['gcc', '-std=gnu11', '-O2', '-ffp-contract=off', '-Wall', '-Wno-unused-variable',
            '-I', os.path.join(FWSIM, 'shim')] +
           sum((['-I', os.path.join(SRC, d)] for d in build_fwsim.FW_INCLUDES), []) +
           [os.path.join(SRC, f) for f in sources] +
           [os.path.join(FWSIM, 'shim', 'fw_shim.c'), os.path.join(HERE, 'gain_set_check.c'), '-o', exe, '-lm'])
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f'gcc failed:\n{r.stderr}'
    assert 'gain_set.c' not in r.stderr, f'warnings in gain_set.c:\n{r.stderr}'
    run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    lines = run.stdout.splitlines()
    fails = [l for l in lines if l.startswith('FAIL')]
    assert run.returncode == 0 and not fails, '\n'.join(fails) or run.stdout + run.stderr
    assert sum(l.startswith('PASS') for l in lines) >= 40, run.stdout
