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
# 'core': emulator core called directly (G1). 'h7port': driven like the H7 layer
# bno_emu_port.c (100 us tick, MISO from a DMA buffer armed ahead), G2 28/10.
MODES = ['core', 'h7port']


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


def run(*args, mode='core'):
    env = dict(os.environ, BNOEMU_H7_PORT='1' if mode == 'h7port' else '0')
    r = subprocess.run([EXE] + [str(a) for a in args], capture_output=True, text=True, timeout=600, env=env)
    kv = dict(l.split('=', 1) for l in r.stdout.splitlines() if '=' in l and not l.startswith('FAIL'))
    fails = [l for l in r.stdout.splitlines() if l.startswith('FAIL')]
    assert r.returncode == 0 and kv.get('RESULT') == 'PASS', '\n'.join(fails) or r.stdout[-2000:]
    return kv


@pytest.mark.parametrize('period', RATES)
def test_20_boots(period, mode):
    """sh2_open gets reset complete, getProdIds 4 entries, 3 reports flow, 20 times in a row."""
    kv = run('boot', period, 20, mode=mode)
    assert kv['boots_ok'] == '20'
    # S06 V3-2: one reset-complete event per reset the host saw (epoch on both sides)
    assert kv['hook_reset_done'] == kv['host_reset_events']


@pytest.mark.parametrize('period', RATES)
def test_10_min_run(period, mode):
    """10 virtual minutes: report period error <= 1 %, values exact on the wire,
    quantization <= 0.5 LSB of Q8/Q9/Q14, no lost report, no host INT timeout."""
    kv = run('run', period, 600, mode=mode)
    for rid in ('01', '02', '08'):
        assert abs(float(kv[f'run.id{rid}.period_err_pct'])) <= 1.0
        assert int(kv[f'run.id{rid}.n']) >= 600 * 1e6 / period - 2


@pytest.mark.parametrize('period', RATES)
def test_host_write_while_packet_pending(period, mode):
    kv = run('collision', period, 10, mode=mode)
    assert int(kv['collision.collisions']) > 0


@pytest.mark.parametrize('period', RATES)
def test_rst_pulse_while_running(period, mode):
    run('rst_mid', period, mode=mode)


@pytest.mark.parametrize('period', RATES)
def test_host_silent_50ms(period, mode):
    kv = run('silent', period, mode=mode)
    assert int(kv['silent.int_timeouts']) > 0


@pytest.mark.parametrize('period', RATES)
def test_frozen_plant_stops_reports(period, mode):
    kv = run('stale', period, mode=mode)
    assert kv['stale.reports_while_frozen'] == '0'


@pytest.mark.parametrize('period', RATES)
def test_hold_imu_loss(period, mode):
    """S06 T8 (V3-1): during IMU loss injection no report is built (rpt_seq does not
    advance, no REPORT event), nothing new reaches the host; reports resume after.
    Every scenario also checks the S06 event hook (harness main): each report at the
    host callback was built and read as part of a packet, rpt_seq +1 per report."""
    kv = run('hold', period, mode=mode)
    assert kv['hold_rx_during'] == '0'
    assert int(kv['hold_skips']) > 0
    assert kv['hook_seq_gap'] == '0' and kv['hook_rx_not_read'] == '0'
