"""G1 tests of the BNO085 emulator (G0 v1 §8): the real SH-2 library of the ESP32
firmware against stm32h7/Core/Src/bno_emu.c, on the PC, in virtual time.
Run: python bnoemu_pc/build_bnoemu.py, then python -m pytest bnoemu_pc/tests"""
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, '..', 'build', 'bnoemu_harness.exe')

pytestmark = pytest.mark.skipif(not os.path.isfile(EXE), reason='bnoemu_harness.exe not built')

RATES = [50_000, 10_000]          # 20 Hz and 100 Hz report period (IMU_REPORT_PERIOD_US)


def run(*args):
    r = subprocess.run([EXE] + [str(a) for a in args], capture_output=True, text=True, timeout=600)
    kv = dict(l.split('=', 1) for l in r.stdout.splitlines() if '=' in l and not l.startswith('FAIL'))
    fails = [l for l in r.stdout.splitlines() if l.startswith('FAIL')]
    assert r.returncode == 0 and kv.get('RESULT') == 'PASS', '\n'.join(fails) or r.stdout[-2000:]
    return kv


@pytest.mark.parametrize('period', RATES)
def test_20_boots(period):
    """sh2_open gets reset complete, getProdIds 4 entries, 3 reports flow, 20 times in a row."""
    kv = run('boot', period, 20)
    assert kv['boots_ok'] == '20'


@pytest.mark.parametrize('period', RATES)
def test_10_min_run(period):
    """10 virtual minutes: report period error <= 1 %, values exact on the wire,
    quantization <= 0.5 LSB of Q8/Q9/Q14, no lost report, no host INT timeout."""
    kv = run('run', period, 600)
    for rid in ('01', '02', '08'):
        assert abs(float(kv[f'run.id{rid}.period_err_pct'])) <= 1.0
        assert int(kv[f'run.id{rid}.n']) >= 600 * 1e6 / period - 2


@pytest.mark.parametrize('period', RATES)
def test_host_write_while_packet_pending(period):
    kv = run('collision', period, 10)
    assert int(kv['collision.collisions']) > 0


@pytest.mark.parametrize('period', RATES)
def test_rst_pulse_while_running(period):
    run('rst_mid', period)


@pytest.mark.parametrize('period', RATES)
def test_host_silent_50ms(period):
    kv = run('silent', period)
    assert int(kv['silent.int_timeouts']) > 0


@pytest.mark.parametrize('period', RATES)
def test_frozen_plant_stops_reports(period):
    kv = run('stale', period)
    assert kv['stale.reports_while_frozen'] == '0'
