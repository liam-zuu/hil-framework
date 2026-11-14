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


@pytest.mark.parametrize('gap_s', [1000, 3221, 7430])
@pytest.mark.parametrize('period', RATES)
def test_frozen_plant_long_gap(period, gap_s, mode):
    """Plant frozen for gap_s with nobody reading, then host reset + init (14/11):
    no report on the old sample whatever the gap. The emulator clock is uint32 us
    (wraps every 2^32 us); 1000 s is below 2^31 us, 3221 s between 2^31 and 2^32,
    7430 s the bench gap of CLOSED_LOOP/2026-11-14_run02 (clock wraps inside it)."""
    kv = run('stale_gap', period, gap_s, mode=mode)
    assert kv['stale_gap.reports_after_reinit'] == '0'
    assert kv['stale_gap.built_while_frozen'] == '0'
    assert int(kv['stale_gap.skips_after_gap']) > 0
    if gap_s == 7430:       # the age passed 2^32 us inside the gap, chip running
        assert float(kv['stale_gap.age_end_s']) > 2**32 * 1e-6


@pytest.mark.parametrize('how', ['held', 'held_wrap'])
@pytest.mark.parametrize('period', RATES)
def test_frozen_plant_long_gap_rst_held(period, how, mode):
    """RST held low during the gap (host off), no report may be built on the old
    sample. 'held': 3221 s. 'held_wrap': the reinit releases RST 1 ms after the
    age passes 2^32 us. With the SH-2 driver of this harness the first report is
    due more than 150 ms after the release, so the unsigned age alone already
    refuses it: this test does not show that the latch runs while in reset
    (Codex BNOW-R1-2); test_core_rst_release_at_clock_wrap does."""
    kv = run('stale_gap', period, 3221, how, mode=mode)
    assert kv['stale_gap.held'] == '1'
    assert kv['stale_gap.reports_after_reinit'] == '0'
    assert kv['stale_gap.built_while_frozen'] == '0'
    if how == 'held_wrap':
        assert 0 < float(kv['stale_gap.release_age_s']) - 2**32 * 1e-6 < 0.01


def test_core_rst_release_at_clock_wrap(mode):
    """Core only, no host: RST released 1 ms after a frozen sample's age passes
    2^32 us, a sensor (50 ms) enabled 21 ms later straight in the core, due at once: no report built
    on the old sample (the latch must also run while the chip is in reset)."""
    kv = run('core_wrap', mode=mode)
    assert kv['core_wrap.built_frozen'] == '0'
    assert int(kv['core_wrap.built_after']) > 0
    # Codex BNOW-R1-3, R2-3: ages from the time the core took the sample, exact (the harness picks the times)
    assert int(kv['core_wrap.age_release_us']) == 2**32 + 1000
    assert int(kv['core_wrap.age_enable_us']) == 2**32 + 1000 + 20_000 + 1000     # boot delay 20 ms, then 1 ms


@pytest.mark.parametrize('sensor', [0, 1, 2])
@pytest.mark.parametrize('how', ['read', 'disable'])
def test_pending_report_not_built_once_stale(how, sensor, mode):
    """Codex BNOW-R1-1: reports that fell due on a fresh sample but wait behind an unread
    input packet are not built once the sample is stale: 'read' the host reads the
    queue after the threshold, with no report due then; 'disable' the host disables
    the only sensor first. With one sensor, the stale check drops the pending report
    anyway (pmin 0), so 'disable' does not show the drop in handle_control
    (Codex BNOW-R2-1): test_disable_one_of_two_pending_sensors does. Every sensor
    index (Codex BNOW-R3-2)."""
    kv = run('pend_stale', how, sensor, mode=mode)
    assert kv['pend_stale.sensor'] == str(sensor)
    assert kv['pend_stale.pend_before'] == '1'
    assert kv['pend_stale.built_after_read'] == '0'


@pytest.mark.parametrize('off,keep', [(0, 1), (1, 0), (1, 2), (2, 0)])
def test_disable_one_of_two_pending_sensors(off, keep, mode):
    """Codex BNOW-R2-1, R3-2: two sensors pending on a fresh sample; disabling sensor
    'off' drops its pending report only ('keep' keeps its own), and after the read a
    report of 'keep' alone is built."""
    kv = run('pend_two', off, keep, mode=mode)
    assert kv['pend_two.sensors'] == f'{off},{keep}'
    assert kv['pend_two.before'] == '1,1'
    assert kv['pend_two.after_disable'] == '0,1'
    assert kv['pend_two.built_after_read'] == '0,1'


@pytest.mark.parametrize('period,due_age,n,built', [
    (10_000, 29_900, 3, 1),          # just under 3 periods: fresh
    (10_000, 30_000, 3, 0),          # at 3 periods: stale
    (1_000_000, 2_900_000, 3, 1),    # 2.9 s old but under 3 periods: no early latch
    (65_536, 131_072, 65_537, 1),    # n x period = 2^32 + 2^16 us: needs the 64-bit product
    # Codex BNOW-R2-2: threshold above 2^31 us, so the latch alone decides near 2^31
    (1_000_000_000, 2**31 - 48, 3, 1),   # just under the latch: fresh
    (1_000_000_000, 2**31, 3, 0),        # at the latch: stale
])
def test_stale_threshold_edge(period, due_age, n, built, mode):
    """Codex BNOW-R1-2: a report due at sample age due_age is built exactly when
    due_age < n x period (64-bit product) and under the 2^31 us latch. Codex
    BNOW-R3-1: the core mode polls exactly at due_age (so 'at the latch' is the
    edge itself, a latch 1 us later fails); the port mode polls on its 100 us ticks.
    BNOW-R4-1: first_poll_age_us is the time the first poll at or after due_age saw;
    in the port mode the tick runs at the harness step (20 us) that reaches it, so it
    lies under one tick plus one step after due_age."""
    kv = run('stale_edge', period, due_age, n, mode=mode)
    first = int(kv['stale_edge.first_poll_age_us'])
    if mode == 'core':
        assert first == due_age
    else:
        assert due_age <= first < due_age + 100 + 20
    assert kv['stale_edge.built'] == str(built)


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
