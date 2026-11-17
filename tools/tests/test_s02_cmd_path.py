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


def test_short_span_rate_is_flagged_not_clamped():
    """A-K1 anchors are 3 s apart: 10 ms of hand-over is -3300 ppm (CLOSED_LOOP/2026-11-14_run01 k1a -3741).
    S2P1-3 (Codex round 1): the map still goes through both anchors (spec P2-2b), only flagged."""
    log, t0_true = gen('A-K1')
    nz = np.flatnonzero(np.abs(log['sp_vx']) > 0)
    log['sp_vx'][nz[-1]] = 0.0                       # the last non-zero line one tick early: b 10 ms early
    s = cp.stats(log, 'A-K1')
    assert s['t0_method'] == 'two_anchors' and s['drift_ppm'] < -2002 and s['t0_reason'] == 'drift_over_bound'
    assert s['t0_us'] == pytest.approx(2 * s['anchor_a_us'] - s['anchor_b_us'], abs=1)   # k_a 150, k_b 300
    assert -30_000 <= s['t0_us'] - t0_true <= 60_000


def test_age_of_exactly_200_ms_is_not_over():
    """'over 200 ms' (P2-2b) is strict: the firmware timeout is age >= 200 ms, a line read at exactly 200 is not
    counted (the age is rounded down, so the true age there is in [200, 201) ms)."""
    log, _ = gen('A-L2')
    t = (log['timestamp_us'] - 5e6) * 1e-6
    i = int(np.flatnonzero(t > 5.0)[0])
    log['traj_age_ms'][i] = 200
    assert cp.stats(log, 'A-L2')['cmd_gap_n'] == 0
    log['traj_age_ms'][i] = 201
    assert cp.stats(log, 'A-L2')['cmd_gap_n'] == 1


def test_ultrasonic_cut_gives_no_t0(tmp_path):
    """S2P1-2 (Liam 15/11 CODE_S02 "Moc t = 0 khong xac dinh"): cut ticks in '#us_end' -> no t = 0; the
    counter-example of Codex (interlock cutting vx to the end of A-L2) is then not mapped at all."""
    log, _ = gen('A-L2')
    t = (log['timestamp_us'] - 5e6) * 1e-6
    log['sp_vx'][t > 8.01] = 0.0                     # interlock from 8.01 s (5 s into the test) to the end
    ev = tmp_path / 'r.csv.events.txt'
    ev.write_text('2026-11-15T10:00:00.000 #us_end,boot=ab,on=1,swb=1000,changes=0,blk_ticks=99,cut_ticks=99,'
                  'ticks=900,last_data_seq=900,line_drop=0' + chr(10), encoding='utf-8')
    cut = cp.us_cut_ticks(str(ev))
    assert cut == 99
    s = cp.stats(log, 'A-L2', 'real', cut)
    assert s['t0_method'] == 'none' and s['t0_reason'] == 'us_cut' and s['us_cut_ticks'] == 99
    assert cp.us_cut_ticks(str(tmp_path / 'missing.txt')) is None


def test_rejected_anchor_is_nan():
    """S2P1-4: an anchor dropped by the span check is NaN in the output (map_run starts from the other)."""
    log, _ = gen('A-L2')
    nz = np.flatnonzero(np.abs(log['sp_vx']) > 0)
    log['sp_vx'][nz[:20]] = 0.0
    s = cp.stats(log, 'A-L2', 'hil', 0)
    assert s['t0_reason'] == 'anchor_a_late' and math.isnan(s['anchor_a_us']) and math.isfinite(s['anchor_b_us'])


def test_sim_t0_is_checked_against_prep_s():
    """S2P1-5: SIM knows t = 0 = prep_s; a schedule shifted by 1 s with prep_s 3 in runs.csv is a tool error."""
    log, _ = gen('A-K3', sim=True, swa_on_s=-1.0, link_s=0.0, offset_s=0.0, prep_s=4.0)
    log['run_active'][:] = 1
    s = cp.stats(log, 'A-K3', 'sim', None, 3.0)
    assert 'sim_t0_mismatch' in s['prep_flags'] and abs(s['sim_t0_err_us'] - 1e6) < 20_000
    s = cp.stats(log, 'A-K3', 'sim', None, 4.0)
    log['timestamp_us'] = log['timestamp_us'] - s['sim_t0_err_us'] + 5_000   # fwsim: the tick at prep_s applies k = 0
    s = cp.stats(log, 'A-K3', 'sim', None, 4.0)
    assert 'sim_t0_mismatch' not in s['prep_flags'] and abs(s['sim_t0_err_us'] - 5_000) < 1


def test_metrics_csv_keeps_every_digit_of_a_timestamp(tmp_path, monkeypatch):
    """S2P1-13: 36 000 045 678 us (10 h uptime) is written as is, not as 3.6e10."""
    import csv as _csv
    import s02_metrics
    (tmp_path / 'runs.csv').write_text('run_id,platform,controller,test_id,speed_mps,rep,file,aborted\n'
                                       'r1,real,PID,A-L2,0.5,1,r1.csv,\n', encoding='utf-8')
    monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {})
    monkeypatch.setattr(s02_metrics, 'score_run', lambda log, tid, *a, **k: {'t0_us': 36000045678.0,
                                                                              'e_vx': 0.0123456789})
    s02_metrics.run_dir_metrics(str(tmp_path))
    with open(tmp_path / 'metrics.csv', newline='') as f:
        row = next(_csv.DictReader(f))
    assert row['t0_us'] == '36000045678' and row['e_vx'] == '0.0123457'
