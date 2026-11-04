"""PC tests of the H7 PWM edge decoder (stm32h7/Core/Src/pwm_edges.c).

The harness (pwmedges_pc/harness.c) runs the H7 file unchanged against a
model of one input: ESP32 MCPWM period grid, timer input filter (8
consecutive samples at 7.5 MHz), 10 MHz 16-bit capture, DMA ring, and the
main-loop service of pwm_capture.c (windows at absolute 0, 7, 14 us: wr1,
pin, pin, pin, wr2; overcapture flag; process up to wr1; anchor; probe after
a lap). Model numbers are assumptions, not measurements: filter sampling,
DMA 300 ns. Limits (Codex M9): the overcapture flag is injected (lose=K), not
produced by a model of CCR and the DMA request; the DMA ring and its write
index change together; one channel only.

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
    assert res['n_cand_retry'] == 0 and res['n_bad_period'] == 0


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
    """Idle 3 ms (below stale): the period that spans the gap is longer than
    max_period, so it is skipped and counted, not filtered in (Codex M2:
    before, duty fell to 0.43 for about 1 ms)."""
    res = run(harness, 'seg=0.5:60', 'seg=0:3', 'seg=0.5:60')
    assert res['n_bad_period'] >= 1
    assert max(abs(v - 0.5) for v in window(res, 30, 123)) <= TOL


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
        res = run(harness, 'seg=0.5:20', f'svc_off_ns={off}', 'max_win=1')
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


def _read_points(t_ms, off_ns=123457, n=3):
    """Pin read times (ns) of the service at t_ms (harness schedule)."""
    t = int(t_ms * 1e6) + off_ns
    return [t + k * 7000 + d for k in range(n) for d in (50, 1750, 3550)]


def _glitches_at(points, width=400):
    return [f'glitch={(p - width // 2) / 1000:.3f}:{width}' for p in points]


def test_glitches_fool_one_service_only(harness):
    """Codex M1: 400 ns glitches (dropped by the input filter) on all 9 pin
    reads of the first service give a wrong first anchor; it needs a second
    service to agree, so no wrong duty comes out."""
    res = run(harness, 'seg=0.3:60', *_glitches_at(_read_points(0)))
    assert res['n_cand_retry'] >= 1
    vals = window(res, 0, 60)
    assert all(abs(v - 0.3) <= TOL or v in (0.0, 1.0) for v in vals)
    assert max(abs(v - 0.3) for v in window(res, 20, 60)) <= TOL


def test_glitches_fool_two_services_then_check_fixes(harness):
    """Glitches on every pin read of the first two services: the wrong edge
    type is accepted, then the 10 ms check (3 windows) disagrees, counts
    parity_fix and the reading recovers within about 25 ms."""
    args = _glitches_at(_read_points(0)) + _glitches_at(_read_points(1))
    res = run(harness, 'seg=0.3:80', *args)
    assert res['n_parity_fix'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 40, 80)) <= TOL


def test_stall_while_pwm_stops_gives_pin_level(harness):
    """Codex M3: main loop blocked 100 ms while the PWM stops (held high from
    100 ms). After the stall nothing new arrives in the 60 us probe, so the
    last edge keeps its age and the pin level comes out at once."""
    res = run(harness, 'seg=0.3:100', 'seg=1:300', 'stall=90:100')
    assert res['n_lap'] >= 1
    assert window(res, 189, 400) and min(window(res, 189, 400)) == 1.0


def test_stall_longer_than_timeout_while_pwm_runs(harness):
    """Blocked 80 ms (more than the 50 ms timeout) while the PWM keeps running:
    the probe sees new captures, so the output stays the duty, not the pin."""
    res = run(harness, 'seg=0.3:300', 'stall=100:80')
    assert res['n_lap'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 30, 300)) <= TOL


def test_lap_before_first_service_is_counted(harness):
    """Codex M12: first service 40 ms after the capture started (ring lapped)."""
    res = run(harness, 'seg=0.3:100', 'stall=0:40')
    assert res['n_lap'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 60, 100)) <= TOL


def test_many_overcaptures_no_wrong_value(harness):
    """Codex M4: overcaptures at many places in the service cycle (service
    1013 us): drop up to the index read after the flag, re-anchor over two
    services, never a wrong value."""
    loses = [f'lose={k}' for k in range(1500, 9000, 157)]
    res = run(harness, 'seg=0.3:250', 'svc_us=1013', *loses)
    assert res['n_lost'] >= 40 and res['n_parity_fix'] == 0
    vals = window(res, 30, 250)
    assert max(abs(v - 0.3) for v in vals) <= TOL


def test_glitches_two_services_phase_locked_needs_periodic_check(harness):
    """As above, but the service is locked so that its first window always
    holds the falling edge (single-window anchors never pass): only the 10 ms
    full check can find the wrong edge type."""
    off = 13000
    args = _glitches_at(_read_points(0, off)) + _glitches_at(_read_points(1, off))
    res = run(harness, 'seg=0.3:80', f'svc_off_ns={off}', *args)
    assert res['n_parity_fix'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 40, 80)) <= TOL


def test_one_glitched_check_does_not_flip_known(harness):
    """A glitched anchor while known only drops to unknown (duty held), it does
    not flip the edge type: no wrong value at any sample."""
    res = run(harness, 'seg=0.3:80', *_glitches_at(_read_points(30)))
    assert res['n_parity_fix'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 20, 80)) <= TOL
