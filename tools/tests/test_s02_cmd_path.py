"""s02_cmd_path.py (CODE_S02 design 4.1-4.3): t = 0 by two command anchors, the preparation check and the
command path numbers, on logs made by a model of the firmware (10 ms ticks with jitter, traj_task hand-over
every 10 ms, 200 ms timeout, traj_age_ms rounded down, sp printed %.3f), sender on its own 50 Hz grid."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import s02_cmd_path as cp  # noqa: E402
from trajectory_pub import GROUP_A, get_body_velocity  # noqa: E402

COLS = ('timestamp_us', 'log_seq', 'motor_on', 'run_active', 'sp_vx', 'sp_vy', 'sp_wz', 'traj_age_ms',
        'g_valid', 'bias_gz', 'bias_updates')


def gen(test_id='A-L2', prep_s=3.0, ppm=0.0, offset_s=5.0, link_s=0.002, swa_on_s=-0.5, drop_packets=(),
        seed=1, sim=False):
    """Log dict of one run. True time 0 = sender packet j = 0 (the first preparation zero); k = 0 is sent at
    prep_s. ESP32 clock = offset + true (1 + ppm). drop_packets: (from_s, to_s) true sender times lost."""
    rng = np.random.default_rng(seed)
    spec = GROUP_A[test_id]
    P, K = int(round(prep_s * 50)), cp.schedule(test_id)[0]
    pk = []                                                   # (arrival true time, (vx, vy, wz))
    for j in range(P + K + 50):
        ts = j / 50
        if any(a <= ts < b for a, b in drop_packets):
            continue
        u = get_body_velocity((j - P) / 50, spec) if P <= j < P + K else (0.0, 0.0, 0.0)
        pk.append((ts + link_s, u))
    arr = np.array([a for a, _ in pk])
    rows = []
    t_end = (P + K + 50) / 50 + 0.5
    n = int((t_end + 1.0) / 0.01)
    g_sp, g_last = (0.0, 0.0, 0.0), None
    bias_updates = 0
    for i in range(n):
        tick = -1.0 + i * 0.01 + 0.003 + rng.uniform(-0.0005, 0.0005)
        task = math.floor((tick - 0.007) / 0.01) * 0.01 + 0.007   # last traj_task run before the tick
        m = np.searchsorted(arr, task, 'right') - 1
        if m >= 0 and task - arr[m] < 0.2:
            g_sp, g_last = pk[m][1], arr[m]
        motor = tick >= swa_on_s
        fresh = g_last is not None and tick - g_last < 0.2
        sp = g_sp if (motor and fresh) else (0.0, 0.0, 0.0)
        r = np.searchsorted(arr, tick, 'right') - 1
        age = -1 if r < 0 else int(math.floor((tick - arr[r]) * (1 + ppm * 1e-6) * 1000 + 1e-9))
        if tick > 2.6:
            bias_updates = 1
        rows.append([round((offset_s + tick * (1 + ppm * 1e-6)) * 1e6), i, int(motor), int(motor and fresh)]
                    + [round(x, 3) for x in sp] + [age, 1, 0.0012, bias_updates])
    a = np.array(rows, dtype=float)
    log = {c: a[:, k] for k, c in enumerate(COLS)}
    if sim:                                                   # s02_groupA_sim.log_rows: log_seq, motor_on set
        log['traj_age_ms'] = np.full(len(a), np.nan)
    return log, offset_s * 1e6 + prep_s * (1 + ppm * 1e-6) * 1e6


def drop(log, mask_fn):
    keep = ~mask_fn(log)
    return {k: v[keep] for k, v in log.items()}


def test_clean_run():
    log, t0_true = gen()
    s = cp.stats(log, 'A-L2')
    assert s['t0_method'] == 'two_anchors'
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000
    assert abs(s['drift_ppm']) < 100
    assert s['edge_kind'] == 'edge' and s['prep_flags'] == ''
    assert 2.95 <= s['prep_s_dut'] <= 3.05
    assert s['run_active_drop'] == 0 and s['g_valid_t0'] == 1 and s['bias_updates_prep'] == 1
    assert s['cmd_gap_n'] == 0 and s['cmd_age_max_ms'] <= 30 and s['cmd_age_unknown_n'] == 0
    assert s['cmd_seq_holes'] == 0 and abs(s['cmd_rows'] - 300 * 2) <= 2


@pytest.mark.parametrize('ppm', [-2000.0, 2000.0])
def test_clock_drift_is_absorbed(ppm):
    log, t0_true = gen('A-L1', ppm=ppm)
    s = cp.stats(log, 'A-L1')
    assert s['t0_method'] == 'two_anchors' and abs(s['drift_ppm'] - ppm) < 100
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000


@pytest.mark.parametrize('gaps,n', [([(4.0, 4.25)], 1), ([(4.0, 4.25), (5.5, 5.8)], 2), ([], 0)])
def test_gaps_over_200_ms(gaps, n):
    log, _ = gen('A-L2', drop_packets=gaps)
    s = cp.stats(log, 'A-L2')
    assert s['cmd_gap_n'] == n
    if n:
        assert s['cmd_age_max_ms'] >= 230
        assert 'run_active_toggle' in s['prep_flags']


def test_lost_lines_never_split_a_run():
    """Two real gaps whose lines in between (a short recovery) are all lost count as one run (B81c-9)."""
    log, _ = gen('A-L2', drop_packets=[(4.0, 4.25), (4.29, 4.55)])
    t = (log['timestamp_us'] - 5e6) * 1e-6
    full = cp.stats(log, 'A-L2')
    assert full['cmd_gap_n'] == 2
    s = cp.stats(drop(log, lambda L: (t > 4.24) & (t < 4.49)), 'A-L2')
    assert s['cmd_gap_n'] == 1 and s['cmd_seq_holes'] > 0


def test_no_preparation_is_edge_after_t0():
    log, _ = gen('A-L2', prep_s=0.02)                         # one zero, then k = 0
    s = cp.stats(log, 'A-L2')
    assert 'prep_short' in s['prep_flags'] or 'edge_after_t0' in s['prep_flags']
    log, _ = gen('A-L2', prep_s=0.02, swa_on_s=0.1)           # motors on after k = 0
    assert 'edge_after_t0' in cp.stats(log, 'A-L2')['prep_flags']


def test_swa_on_late_is_prep_short():
    log, _ = gen('A-L2', swa_on_s=2.0)
    s = cp.stats(log, 'A-L2')
    assert 'prep_short' in s['prep_flags'] and 0.9 <= s['prep_s_dut'] <= 1.1


def test_edge_with_its_previous_line_lost_is_unknown():
    log, _ = gen('A-L2')
    ra = log['run_active']
    i = int(np.flatnonzero((ra[1:] == 1) & (ra[:-1] == 0))[0])     # the line before the edge
    s = cp.stats(drop(log, lambda L: np.arange(len(L['log_seq'])) == i), 'A-L2')
    assert 'edge_unknown' in s['prep_flags']


def test_anchor_with_its_previous_line_lost_falls_back_to_one_anchor():
    log, t0_true = gen('A-L2')
    nz = np.flatnonzero(np.abs(log['sp_vx']) > 0)
    s = cp.stats(drop(log, lambda L: np.arange(len(L['log_seq'])) == nz[0] - 1), 'A-L2')
    assert s['t0_method'] == 'one_anchor' and s['t0_reason'] == 'anchor_a_lost'
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000


def test_cut_at_the_start_anchor_drops_anchor_a():
    """An ultrasonic cut (command 0) over the first 0.2 s moves anchor a: the span is short, t = 0 comes
    from anchor b alone."""
    log, t0_true = gen('A-L2')
    nz = np.flatnonzero(np.abs(log['sp_vx']) > 0)
    log['sp_vx'][nz[:20]] = 0.0
    s = cp.stats(log, 'A-L2')
    assert s['t0_method'] == 'one_anchor' and s['t0_reason'] == 'anchor_a_late'
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000


def test_lost_first_tail_zeros_drop_anchor_b():
    log, t0_true = gen('A-L2', drop_packets=[(9.0, 9.1)])
    s = cp.stats(log, 'A-L2')
    assert s['t0_method'] == 'one_anchor' and s['t0_reason'] == 'anchor_b_late'
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000


def test_timeout_zero_is_not_anchor_b():
    """The last schedule packets and the tail lost: the command falls to 0 by the 200 ms timeout."""
    log, t0_true = gen('A-L2', drop_packets=[(8.7, 99.0)])
    s = cp.stats(log, 'A-L2')
    assert s['t0_method'] == 'one_anchor' and s['t0_reason'] == 'anchor_b_timeout'
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000 and s['cmd_gap_n'] == 1


def test_sim_cold_start():
    log, t0_true = gen('A-K3', sim=True, swa_on_s=-1.0, link_s=0.0)
    log['run_active'][:] = 1                                   # fwsim: run_active from the first tick
    s = cp.stats(log, 'A-K3', 'sim')
    assert s['edge_kind'] == 'cold_start' and s['prep_flags'] == ''
    assert 'edge_unknown' in cp.stats(log, 'A-K3', 'hil')['prep_flags']    # a DUT log starting mid-run
    assert s['cmd_reason'] == 'no_traj_age' and math.isnan(s['cmd_gap_n'])
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000


def test_ak1_anchors_are_the_step_edges():
    assert cp.schedule('A-K1') == (450, 150, 300)
    assert cp.schedule('A-K2') == (450, 0, 450)
    log, t0_true = gen('A-K1')
    s = cp.stats(log, 'A-K1')
    assert s['t0_method'] == 'two_anchors' and 0.0 <= s['t0_us'] - t0_true <= 30_000


def test_unknown_age_lines_are_counted():
    log, _ = gen('A-L2')
    t = (log['timestamp_us'] - 5e6) * 1e-6
    log['traj_age_ms'][(t > 4.0) & (t < 4.05)] = -1
    s = cp.stats(log, 'A-L2')
    assert s['cmd_age_unknown_n'] == 5 and s['cmd_gap_n'] == 0


def test_window_edges():
    """A gap that ends just before k = 0 is outside; one over the last packet is inside."""
    log, _ = gen('A-L2', drop_packets=[(2.5, 2.99)])
    assert cp.stats(log, 'A-L2')['cmd_gap_n'] == 0
    log, _ = gen('A-L2', drop_packets=[(3 + 299 / 50 - 0.3, 3 + 299 / 50 + 0.01)])
    assert cp.stats(log, 'A-L2')['cmd_gap_n'] == 1


def test_short_span_rate_is_clamped():
    """A-K1 anchors are 3 s apart: 11 ms of hand-over is -3700 ppm (CLOSED_LOOP/2026-11-14_run01 k1a)."""
    log, t0_true = gen('A-K1')
    nz = np.flatnonzero(np.abs(log['sp_vx']) > 0)
    log['sp_vx'][nz[-1]] = 0.0                       # the last non-zero line one tick early: b 10 ms early
    s = cp.stats(log, 'A-K1')
    assert s['t0_method'] == 'two_anchors' and s['drift_ppm'] < -2002 and s['t0_reason'] == 'drift_clamped'
    assert 0.0 <= s['t0_us'] - t0_true <= 30_000
