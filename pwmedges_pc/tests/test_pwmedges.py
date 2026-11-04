"""PC tests of the H7 PWM edge decoder (stm32h7/Core/Src/pwm_edges.c).

The harness (pwmedges_pc/harness.c) runs the H7 file unchanged against a
model of one input: ESP32 MCPWM period grid, timer input filter, 10 MHz
16-bit capture, DMA ring, and the main-loop service of pwm_capture.c (wr1,
pin, 3 us, pin, wr2, lost flag). Model numbers are assumptions, not
measurements: filter 1067 ns (8 samples at 240 MHz / 32), DMA 300 ns.

  python -m pytest pwmedges_pc/tests -q      (needs gcc on PATH)

PWMEDGES_SRC=<file> builds the harness against another copy of pwm_edges.c
(mutation tests).
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PC = os.path.dirname(HERE)
REPO = os.path.dirname(PC)
TICK = 1.0 / 500           # one 10 MHz tick over a 20 kHz period
TOL = 0.003                # 1 tick of width plus filter rounding
T6_DUTIES = [0.03, 0.05, 0.10, 0.30, 0.50, 0.70, 0.95, 0.97]


@pytest.fixture(scope='session')
def harness(tmp_path_factory):
    gcc = shutil.which('gcc')
    if gcc is None:
        pytest.skip('gcc not on PATH')
    src = os.environ.get('PWMEDGES_SRC',
                         os.path.join(REPO, 'stm32h7', 'Core', 'Src', 'pwm_edges.c'))
    exe = str(tmp_path_factory.mktemp('pwmedges') / 'pwmedges_harness.exe')
    cmd = [gcc, '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror',
           '-I', os.path.join(REPO, 'stm32h7', 'Core', 'Inc'),
           src, os.path.join(PC, 'harness.c'), '-o', exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def run(exe, *args):
    r = subprocess.run([exe] + [str(a) for a in args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def window(res, t0, t1):
    return [d for t, d in res['samples'] if t0 <= t < t1]


@pytest.mark.parametrize('duty', T6_DUTIES)
def test_steady_duty_all_t6_levels(harness, duty):
    """Every T6 level, including 1.5 us high or low phases, reads right."""
    res = run(harness, f'seg={duty}:200', 'jitter_ns=40')
    vals = window(res, 30, 200)
    assert vals and max(abs(v - duty) for v in vals) <= TOL
    assert res['n_parity_fix'] == 0 and res['n_lost'] == 0 and res['n_lap'] == 0


@pytest.mark.parametrize('phase_ns', [0, 1234, 17000, 34999, 49900])
def test_start_at_any_phase(harness, phase_ns):
    """Edge type is unknown at start; the first anchor fixes it within a few ms."""
    res = run(harness, 'seg=0.3:60', f'phase_ns={phase_ns}')
    vals = window(res, 25, 60)
    assert max(abs(v - 0.3) for v in vals) <= TOL
    assert res['n_parity_fix'] == 0


def test_rl_step3_pattern(harness):
    """pwm_hold RL pin: held high, 30 % high, held high, 70 % high (step 3:
    35 us high, 15 us low). Held level reads 1 after the 50 ms timeout."""
    res = run(harness, 'seg=1:200', 'seg=0.3:200', 'seg=1:200', 'seg=0.7:200')
    assert max(abs(v - 1.0) for v in window(res, 0, 200)) == 0
    assert max(abs(v - 0.3) for v in window(res, 230, 400)) <= TOL
    assert max(abs(v - 1.0) for v in window(res, 451, 600)) == 0
    assert max(abs(v - 0.7) for v in window(res, 630, 800)) <= TOL
    assert res['n_parity_fix'] == 0


def test_timeout_gives_pin_level(harness):
    """PWM stops low: duty holds for timeout_ms, then reads 0."""
    res = run(harness, 'seg=0.5:100', 'seg=0:200')
    assert max(abs(v - 0.5) for v in window(res, 60, 145)) <= TOL
    assert window(res, 152, 300) and max(window(res, 152, 300)) == 0.0


def test_overcapture_flag_drops_and_recovers(harness):
    """A capture overwritten (flag set): no wrong value, back on duty after re-anchor."""
    res = run(harness, 'seg=0.97:120', 'lose=2001')
    vals = window(res, 30, 120)
    assert max(abs(v - 0.97) for v in vals) <= TOL
    assert res['n_lost'] == 1 and res['n_parity_fix'] == 0


def test_silent_loss_is_fixed_by_anchor(harness):
    """A missing capture without the flag (not expected from the timer): the
    anchor sees the wrong edge type; error lasts at most about 2 services."""
    res = run(harness, 'seg=0.3:150', 'silent=2001')
    assert res['n_parity_fix'] >= 1
    bad = [t for t, d in res['samples'] if t > 30 and abs(d - 0.3) > TOL]
    assert bad, 'loss had no effect: test does not reach the parity path'
    assert max(bad) - min(bad) <= 3.0
    assert max(abs(v - 0.3) for v in window(res, max(bad) + 25, 150)) <= TOL


def test_service_stall_longer_than_lap(harness):
    """Main loop blocked 40 ms (printf): ring may lap; drop, re-anchor, no wrong value."""
    res = run(harness, 'seg=0.3:200', 'stall=80:40', 'ring=256')
    assert res['n_lap'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 30, 200)) <= TOL


def test_short_stall_within_lap_keeps_counting(harness):
    """Blocked 3 ms with a 256-entry ring (120 edges): read through, no drop."""
    res = run(harness, 'seg=0.95:120', 'stall=60:3', 'ring=256')
    assert res['n_lap'] == 0 and res['n_parity_fix'] == 0
    assert max(abs(v - 0.95) for v in window(res, 30, 120)) <= TOL


def test_idle_gap_past_counter_wrap(harness):
    """Idle 20 ms (counter wraps every 6.5 ms) then PWM again: the open period
    is dropped, no value from a wrapped difference."""
    res = run(harness, 'seg=0.3:100', 'seg=0:20', 'seg=0.7:100')
    assert res['n_stale'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 30, 100)) <= TOL
    vals = window(res, 100, 220)
    assert min(vals) >= 0.3 - TOL and max(vals) <= 0.7 + TOL
    assert max(abs(v - 0.7) for v in window(res, 150, 220)) <= TOL


def test_idle_gap_short_of_stale(harness):
    """Idle 3 ms (below stale): the first period after it spans the gap and
    reads low, the filter pulls back within a few periods."""
    res = run(harness, 'seg=0.5:60', 'seg=0:3', 'seg=0.5:60')
    assert max(abs(v - 0.5) for v in window(res, 80, 123)) <= TOL


def test_glitch_shorter_than_filter(harness):
    """A 500 ns pin glitch is not captured; the anchor must not take it."""
    args = ['seg=0.97:100'] + [f'glitch={30000 + 50 * k + 10}:500' for k in range(40)]
    res = run(harness, *args)
    assert res['n_parity_fix'] == 0
    assert max(abs(v - 0.97) for v in window(res, 30, 100)) <= TOL


def test_glitch_on_held_level(harness):
    """Glitches on a held-low pin: output stays 0."""
    args = ['seg=0:100'] + [f'glitch={20000 + 997 * k}:400' for k in range(50)]
    res = run(harness, *args)
    assert res['captures'] == 0
    assert max(window(res, 0, 100)) == 0.0


def test_ring_index_wraps(harness):
    """Small ring (64 entries, 40 edges per service): index wraps every 2 services."""
    res = run(harness, 'seg=0.05:150', 'ring=64')
    assert res['n_lap'] == 0 and res['n_parity_fix'] == 0
    assert max(abs(v - 0.05) for v in window(res, 30, 150)) <= TOL


def test_duty_change_follows_within_a_ms(harness):
    """30 % to 70 %: the filter (0.25 per period) reaches the new value in about 1 ms."""
    res = run(harness, 'seg=0.3:50', 'seg=0.7:50')
    after = [(t, d) for t, d in res['samples'] if t >= 50]
    first_ok = next(t for t, d in after if abs(d - 0.7) <= TOL)
    assert first_ok - 50 <= 2.2


@pytest.mark.parametrize('duty', [0.03, 0.14, 0.5, 0.97])
def test_service_locked_to_pwm_phase(harness, duty):
    """Service every 1 ms = exactly 20 periods, so every anchor window sits at
    the same PWM phase. With 3 windows 7 us apart one always misses the edges:
    duty right within 10 ms at every phase (sweep 700 ns steps)."""
    for off in range(1000, 51000, 700):
        res = run(harness, f'seg={duty}:20', f'svc_off_ns={off}')
        vals = window(res, 10, 20)
        assert vals and max(abs(v - duty) for v in vals) <= TOL, (duty, off)


def test_single_window_can_lock_out(harness):
    """Same lock with one window only: some phase never anchors (why 3 windows)."""
    stuck = 0
    for off in range(1000, 51000, 700):
        res = run(harness, 'seg=0.5:20', f'svc_off_ns={off}', 'n_try=1')
        if max(abs(v - 0.5) for v in window(res, 10, 20)) > TOL:
            stuck += 1
    assert stuck > 0


def test_filter_weight_per_period(harness):
    """0.3 to 0.7 at 50 ms, service 160 us later: 3 periods done, so
    duty = 0.7 - 0.4 * 0.75**3 (filter 0.75 / 0.25 per period, Fable F-ND #8)."""
    res = run(harness, 'seg=0.3:50', 'seg=0.7:10', 'svc_off_ns=160000')
    t, d = next((t, d) for t, d in res['samples'] if t >= 50)
    assert abs(t - 50.16) < 1e-6
    assert abs(d - (0.7 - 0.4 * 0.75 ** 3)) <= TOL


def test_stall_while_held_keeps_pin_level(harness):
    """A long service stall while the pin is held: nothing arrived, so the
    output stays the pin level (no stale duty for another timeout)."""
    res = run(harness, 'seg=0.3:100', 'seg=1:300', 'stall=250:40')
    assert window(res, 155, 400) and min(window(res, 155, 400)) == 1.0


def test_overcapture_with_odd_edge_count(harness):
    """Service period not a multiple of the PWM period (40 or 41 edges per
    service): after a lost capture the edge type must be re-anchored, not
    carried over, or it comes out wrong half of the time."""
    res = run(harness, 'seg=0.3:200', 'svc_us=1013',
              'lose=2001', 'lose=3005', 'lose=4010', 'lose=5003', 'lose=6008', 'lose=7001')
    assert res['n_lost'] == 6 and res['n_parity_fix'] == 0
    assert max(abs(v - 0.3) for v in window(res, 30, 200)) <= TOL
