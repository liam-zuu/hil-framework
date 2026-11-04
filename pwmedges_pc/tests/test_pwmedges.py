"""PC tests of the H7 PWM edge decoder (stm32h7/Core/Src/pwm_edges.c).

The harness (pwmedges_pc/harness.c) runs the H7 file unchanged against a
model of one input: ESP32 MCPWM period grid, timer input filter (8
consecutive samples at 7.5 MHz, sample by sample), 10 MHz 16-bit capture,
DMA ring, and the main-loop service of pwm_capture.c (windows from
shift + 0, 10, 20 us, shift 0 / 2.5 / 5 / 7.5 us by service while 3 windows:
wr1, pin a, pin b, pin c, wr2 with gaps counted from the end of the previous
read group; overcapture flag; process up to wr1; anchor; probe after every
lap: polls of the write index, optional preemption, one last read after
the loop). Model numbers are assumptions, not measurements:
filter sampling, DMA 300 ns, read group 500 ns. Limits (Codex M9, M20 part
two): the overcapture flag is injected (lose=K), not produced by a model of
CCR and the DMA request; the DMA ring and its write index change together;
one channel only; pwm_capture.c itself (HAL, EN bit, SR) is not run here.

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
    the same PWM phase. With 3 windows 10 us apart one always misses the edges:
    duty right within 10 ms at every phase (sweep 700 ns steps)."""
    for off in range(1000, 51000, 700):
        res = run(harness, f'seg={duty}:20', f'svc_off_ns={off}')
        vals = window(res, 10, 20)
        assert vals and max(abs(v - duty) for v in vals) <= TOL, (duty, off)


def test_single_window_can_lock_out(harness):
    """Same lock with one window only (the shift applies to 3 windows only):
    some phase never anchors (why 3 windows)."""
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


def _read_points(exe, base, services):
    """Pin read times (ns) of the given services (0-based, no stall), taken
    from the harness itself (dump_reads) for the run 'base' without glitches.
    Glitches placed later do not change the schedule of these services as
    long as the decoder state before them is the same."""
    res = run(exe, *base, f'dump_reads={max(services) + 1}')
    pts = []
    for s in services:
        pts += res['reads'][s][1:]
    return pts


def _glitches_at(points, width=400):
    return [f'glitch={(p - width // 2) / 1000:.3f}:{width}' for p in points]


def test_glitches_fool_one_service_only(harness):
    """Codex M1: 400 ns glitches (dropped by the input filter) on all 9 pin
    reads of the first service give a wrong first anchor; it needs a second
    service to agree, so no wrong duty comes out."""
    base = ['seg=0.3:60']
    res = run(harness, *base, *_glitches_at(_read_points(harness, base, [0])))
    assert res['n_cand_retry'] >= 1
    vals = window(res, 0, 60)
    assert all(abs(v - 0.3) <= TOL or v in (0.0, 1.0) for v in vals)
    assert max(abs(v - 0.3) for v in window(res, 20, 60)) <= TOL


def test_glitches_fool_two_services_then_check_fixes(harness):
    """Glitches on every pin read of the first two services: the wrong edge
    type is accepted, then the 10 ms check (3 windows) disagrees, counts
    parity_fix and the reading recovers within about 25 ms."""
    base = ['seg=0.3:80']
    res = run(harness, *base, *_glitches_at(_read_points(harness, base, [0, 1])))
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
    """As above, but the service is locked so that its single window (known
    state, no shift) always holds the falling edge (ring time about 16.4 us,
    window 13 to 17.5 us): single-window anchors never pass, only the 10 ms
    full check can find the wrong edge type."""
    base = ['seg=0.3:80', 'svc_off_ns=13000']
    res = run(harness, *base, *_glitches_at(_read_points(harness, base, [0, 1])))
    assert res['n_parity_fix'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 40, 80)) <= TOL


def test_glitches_on_reads_a_and_c_only_are_rejected(harness):
    """Glitches on pin reads a and c (not b) of every window of the first two
    services: a and c agree on the wrong level, b disagrees, so no anchor is
    taken there and no wrong value comes out (all three reads must agree)."""
    base = ['seg=0.3:60']
    pts = _read_points(harness, base, [0, 1])
    ac = [p for i, p in enumerate(pts) if i % 3 != 1]
    res = run(harness, *base, *_glitches_at(ac))
    vals = window(res, 0, 60)
    assert all(abs(v - 0.3) <= TOL or v in (0.0, 1.0) for v in vals)
    assert res['n_parity_fix'] == 0
    assert max(abs(v - 0.3) for v in window(res, 20, 60)) <= TOL


def test_one_glitched_check_does_not_flip_known(harness):
    """A glitched anchor while known only drops to unknown (duty held), it does
    not flip the edge type: no wrong value at any sample."""
    base = ['seg=0.3:80']
    res = run(harness, *base, *_glitches_at(_read_points(harness, base, [30])))
    assert res['n_parity_fix'] >= 1
    assert max(abs(v - 0.3) for v in window(res, 20, 80)) <= TOL


# ---- Codex Sol round 2 (04/11): M14-M20 -------------------------------------

def test_filter_counts_samples_not_raw_edges(harness):
    """Codex M19: rise at 10 ns, LOW 500-510 ns falls between two samples
    (400, 533.3 ns at 7.5 MHz). Samples 133.3 ... 1066.7 ns are all HIGH, so
    the 8th consecutive sample captures at 1066.7 ns, the same as without the
    short LOW (the old model restarted the count: 1466.7 ns)."""
    clean = run(harness, 'seg=0.3:1', 'phase_ns=10', 'dump_caps=1')
    short = run(harness, 'seg=0.3:1', 'phase_ns=10', 'glitch=0.5:10', 'dump_caps=1')
    assert clean['caps'][0] == 1066
    assert short['caps'][0] == 1066


def test_filter_sampled_short_low_restarts_count(harness):
    """A LOW that is sampled once (400-410 ns, sample at 400 ns) resets the
    count: capture at the 8th HIGH sample after it, 533.3 + 7 x 133.3 = 1466.7 ns."""
    res = run(harness, 'seg=0.3:1', 'phase_ns=10', 'glitch=0.4:10', 'dump_caps=1')
    assert res['caps'][0] == 1466


@pytest.mark.parametrize('duty', [0.03, 0.3, 0.97])
def test_filter_phase_sweep(harness, duty):
    """Codex M19: every phase of the filter sampling clock against the PWM grid."""
    for fph in range(0, 134, 19):
        res = run(harness, f'seg={duty}:60', f'fphase_ns={fph}', 'jitter_ns=40')
        vals = window(res, 30, 60)
        assert vals and max(abs(v - duty) for v in vals) <= TOL, (duty, fph)
        assert res['n_parity_fix'] == 0


# Codex M14: duty 0.2 on even, 0.8 on odd periods (compare updated at TEZ).
# Service at m ms - 10.12 us, every 1 ms = 20 periods: the falling edge of the
# odd period (m ms - 10 us), the rise (m ms) and the falling edge of the even
# period (m ms + 10 us) reach the ring about 1.4 us later, i.e. inside the
# three windows (0-4.5, 10-14.5, 20-24.5 us after the service start) at every
# service, so no window is quiet without the window shift.
ALT_LOCK = ['alt=0.2:0.8:40', 'svc_off_ns=989880']


def test_alternating_duty_locks_out_without_shift(harness):
    """Without the window shift the edge type never becomes known: the duty
    stays at its start value 0 with edges coming; n_unanchored counts it."""
    res = run(harness, *ALT_LOCK, 'win_shift=0')
    assert res['known'] == 0 and res['n_edges'] > 1000
    assert max(window(res, 0, 40)) == 0.0
    assert res['n_unanchored'] == 1


def test_alternating_duty_anchors_with_shift(harness):
    """With the shift (0, 2.5, 5, 7.5 us per service) some windows are quiet:
    known within a few ms, then the filtered duty of 0.2 / 0.8 periods
    (0.457 or 0.543 by the 0.75 / 0.25 filter)."""
    res = run(harness, *ALT_LOCK)
    assert res['known'] == 1 and res['n_unanchored'] == 0
    vals = window(res, 10, 40)
    assert vals and min(vals) >= 0.457 - TOL and max(vals) <= 0.543 + TOL


def test_unanchored_counts_once_per_run(harness):
    """A run ends when the edge type becomes known. Lock-out pattern 40 ms
    (run 1), plain 0.3 for 30 ms (anchors, known), lock-out again with an
    overcapture at about 82 ms (unknown again, run 2): 2, not one per service.
    A stall inside a run with the PWM still active keeps the same run."""
    res = run(harness, 'alt=0.2:0.8:40', 'seg=0.3:30', 'alt=0.2:0.8:60',
              'svc_off_ns=989880', 'win_shift=0', 'lose=3300')
    assert res['n_lost'] == 1 and res['n_unanchored'] == 2
    res = run(harness, 'alt=0.2:0.8:100', 'svc_off_ns=989880', 'win_shift=0',
              'stall=30:60')
    assert res['n_lap'] == 1 and res['n_unanchored'] == 1
    # every window holds a capture here, so the entries read in window 1 after
    # the lap in window 0 already show activity: no probe needed
    assert res['n_probe'] == 0


def test_idle_gap_wrapping_to_short_period(harness):
    """Idle 19.65 ms: the rise-to-rise time across the gap is 19.70 ms =
    197000 ticks, 392 ticks after the 16-bit wrap, shorter than max_period.
    Only the stale drop keeps it out: 3 periods of 0.7 after the restart
    give 0.7 - 0.4 x 0.75^3 at the service 160 us later (with the wrapped
    period filtered in: about 0.540)."""
    res = run(harness, 'seg=0.3:100', 'seg=0:19.65', 'seg=0.7:20', 'svc_off_ns=810000')
    t, d = next((t, d) for t, d in res['samples'] if t >= 119.65)
    assert abs(t - 119.81) < 1e-6
    assert res['n_stale'] >= 1 and res['n_bad_period'] == 0
    assert abs(d - (0.7 - 0.4 * 0.75 ** 3)) <= TOL


def _bad_after_stall(exe, final):
    """Codex M24: duty 0.03 runs on, main loop blocked 100-180 ms with an
    overcapture in the stall (lap + flag: entries up to the flag read are
    dropped), a 21 us preemption 40 us into the probe. Count service phases
    whose first sample after the stall is not the duty."""
    bad = 0
    for off in range(100000, 150000, 1000):
        res = run(exe, 'seg=0.03:300', 'stall=100:80', f'svc_off_ns={off}', 'lose=6000',
                  'probe_pre=40:21', f'probe_final={final}')
        s = window(res, 180, 181)
        assert len(s) == 1 and res['n_probe'] >= 1
        bad += abs(s[0] - 0.03) > TOL
    return bad


def test_probe_last_read_after_preemption(harness):
    """Codex M24: with the last index read after the probe loop, a preemption
    past the 60 us deadline cannot hide captures that came meanwhile; without
    it (firmware before round 3) some phases read the pin level (0)."""
    assert _bad_after_stall(harness, 0) > 0
    assert _bad_after_stall(harness, 1) == 0


def test_init_tick_not_zero(harness):
    """Codex M26: capture started at HAL tick 1000 ms. The lap test counts from
    the init tick: first service 0.12 ms (no stall) or 4.12 ms after the start
    is no lap, 6.12 ms is one."""
    res = run(harness, 'seg=0.3:100', 't0_ms=1000')
    assert res['n_lap'] == 0
    assert max(abs(v - 0.3) for v in window(res, 30, 100)) <= TOL
    assert run(harness, 'seg=0.3:100', 't0_ms=1000', 'stall=0:4')['n_lap'] == 0
    assert run(harness, 'seg=0.3:100', 't0_ms=1000', 'stall=0:6')['n_lap'] == 1


def test_overcapture_counts_as_activity(harness):
    """The last capture (rise at 100 ms, then held high) is overwritten: the
    flag at the next service (no lap) means a capture happened since the
    service before, so the 50 ms timeout runs from 100 ms: duty at the
    149.12 ms service, pin level at 150.12 ms."""
    res = run(harness, 'seg=0.3:100', 'seg=1:100', 'lose=3999')
    assert res['n_lost'] == 1 and res['n_lap'] == 0
    at149, at150 = window(res, 149, 150), window(res, 150, 151)
    assert len(at149) == 1 and abs(at149[0] - 0.3) <= TOL
    assert at150 == [1.0]


def test_held_level_between_runs_anchors(harness):
    """Lock-out pattern, input held low 100 ms, pattern again: while held the
    windows are quiet, so the held level fixes the edge type; the second run
    decodes at once (not counted)."""
    res = run(harness, 'alt=0.2:0.8:40', 'seg=0:100', 'alt=0.2:0.8:40',
              'svc_off_ns=989880', 'win_shift=0')
    assert res['n_unanchored'] == 1 and res['known'] == 1
    vals = window(res, 150, 180)
    assert vals and min(vals) >= 0.457 - TOL and max(vals) <= 0.543 + TOL


def test_lap_with_ring_index_back_at_rd_still_probes(harness):
    """Codex M15: service gap of exactly 2048 captures (51.2 ms): the write
    index is back at rd; the probe must still run and see the PWM is active,
    so the output stays the duty (before: 0.0 at about 151.12 ms)."""
    res = run(harness, 'seg=0.3:220', 'svc_us=100', 'stall=100:51.1')
    assert res['n_lap'] >= 1 and res['n_probe'] >= 1
    vals = window(res, 151, 220)
    assert vals and max(abs(v - 0.3) for v in vals) <= TOL


def test_probe_with_no_capture_runs_once(harness):
    """Codex M16: PWM stops during a 100 ms stall and the pin stays high: one
    probe after the stall, not one per service until the next edge."""
    res = run(harness, 'seg=0.3:100', 'seg=1:300', 'stall=90:100')
    assert res['n_lap'] == 1 and res['n_probe'] == 1
    assert min(window(res, 189, 400)) == 1.0


def test_old_overcapture_flag_after_lap_is_not_activity(harness):
    """Codex M17: an overcapture at about 99.75 ms during a 90-190 ms stall,
    PWM stops high at 100 ms. The flag seen at 190 ms is as old as the stall:
    no activity is taken from it, the probe sees nothing, output = pin level
    at once (before: duty 0.3 held to about 240 ms)."""
    res = run(harness, 'seg=0.3:100', 'seg=1:300', 'stall=90:100', 'lose=3990')
    assert res['n_lost'] == 1 and res['n_lap'] == 1
    assert min(window(res, 189, 400)) == 1.0


def test_stall_then_duty_change_recovers(harness):
    """Codex M20: after a stall longer than the timeout the measurement must
    recover, not only hold the old duty: 0.3 before, 0.7 from 150 ms, stall
    100-180 ms."""
    res = run(harness, 'seg=0.3:150', 'seg=0.7:150', 'stall=100:80')
    assert res['n_lap'] >= 1 and res['known'] == 1
    assert max(abs(v - 0.3) for v in window(res, 30, 100)) <= TOL
    vals = window(res, 185, 300)
    assert vals and max(abs(v - 0.7) for v in vals) <= TOL


def test_overcaptures_then_duty_change_recovers(harness):
    """Codex M20: overcaptures across a duty change 0.3 -> 0.7: the reading
    follows the change, it does not stay at the duty before the losses."""
    loses = [f'lose={k}' for k in range(1500, 7500, 157)]
    res = run(harness, 'seg=0.3:100', 'seg=0.7:100', 'svc_us=1013', *loses)
    assert res['n_lost'] >= 30 and res['n_parity_fix'] == 0
    assert max(abs(v - 0.3) for v in window(res, 30, 100)) <= TOL
    vals = window(res, 105, 200)
    assert vals and max(abs(v - 0.7) for v in vals) <= TOL


@pytest.mark.parametrize('duty', T6_DUTIES)
def test_preemption_before_read_groups(harness, duty):
    """Codex M18: random delays up to 20 us before read groups (one group in
    four) and slow groups (1.5 us): gaps are counted from the end of the
    previous group, so no window is squeezed; no wrong value."""
    res = run(harness, f'seg={duty}:120', 'pre_ns=20000', 'grp_ns=1500', 'jitter_ns=40')
    vals = window(res, 30, 120)
    assert vals and max(abs(v - duty) for v in vals) <= TOL
    assert res['n_parity_fix'] == 0 and res['n_cand_retry'] == 0


def test_preemption_with_glitches_on_first_service(harness):
    """Glitches on the first service's reads with preemption on: the one-service
    candidate still needs a second service, no wrong value."""
    base = ['seg=0.3:60', 'pre_ns=20000']
    res = run(harness, *base, *_glitches_at(_read_points(harness, base, [0])))
    vals = window(res, 0, 60)
    assert all(abs(v - 0.3) <= TOL or v in (0.0, 1.0) for v in vals)
    assert max(abs(v - 0.3) for v in window(res, 20, 60)) <= TOL
