"""tools/s06_t5_score.py (S06 T5, H5 encoder frequency error) on synthetic ESP32 count reads.

Counts are generated from the T5 schedule on the ESP32 clock (t_esp = offset + t_schedule), with
omega after frame quantization and the 0.3 rad/s deadband, read every 5 ms (1 ms on the board).
The scorer must find the offset by itself, pass an exact generator and a 0.4 % error, and catch a
1 % error, a reversed wheel, a swapped pair of wheels (only the W15 visit shows it), and data that
ends early. Run: python -m pytest tools/tests"""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_t5_score as sc  # noqa: E402
import t5_schedule as t5  # noqa: E402

READ_STEP_US, READ_SPAN_US = 5000.0, 30.0
K = 1320.0 / (2.0 * np.pi)


def omega_d(w):
    # independent of the scorer: int16 over +-40 rad/s in the frame, then the 0.3 rad/s deadband
    q = round(w / 40.0 * 32767) / 32767 * 40.0
    return 0.0 if abs(q) <= 0.3 else q


def synth(sched, offset_us, gain=(1, 1, 1, 1), lead_us=3e6, tail_us=3e6, swap=None, end_early_us=0.0):
    """Count reads (t_before, t_after, counts) on the ESP32 clock."""
    t_end = offset_us + sched[-1].t_end_ms * 1000.0 + tail_us - end_early_us
    tb = np.arange(offset_us - lead_us, t_end, READ_STEP_US) + 123.0
    ta = tb + READ_SPAN_US
    tm = 0.5 * (tb + ta) - offset_us                    # schedule time of each read
    edges = np.array([v.t_start_ms * 1000.0 for v in sched] + [sched[-1].t_end_ms * 1000.0])
    rates = np.array([[K * omega_d(w) for w in v.omegas] for v in sched])   # count/s
    cum = np.concatenate([np.zeros((1, 4)), np.cumsum(rates * np.diff(edges)[:, None] * 1e-6, axis=0)])
    c = np.zeros((len(tm), 4))
    for w in range(4):
        x = np.clip(tm, edges[0], edges[-1])
        c[:, w] = np.interp(x, edges, cum[:, w]) * gain[w]
    c = np.floor(c).astype(np.int64)
    if swap:
        i, j = swap
        c[:, [i, j]] = c[:, [j, i]]
    return sc.Reads(tb, ta, c, eseq=np.arange(1, len(tb) + 1))


@pytest.fixture(scope='module')
def sched1():
    return t5.make_schedule(1)


def run(reads, sched):
    off, corr = sc.align_offset(reads, sched)
    return off, corr, sc.score_run(reads, sched, off)


@pytest.mark.parametrize('offset_us, lead_us', [(7.3e6, 3e6), (5e6, -100e6)])
def test_exact_generator_passes_and_offset_found(sched1, offset_us, lead_us):
    # lead -100 s: the ESP32 log starts after the first visit (both indicator visits lost)
    off, corr, rows = run(synth(sched1, offset_us, lead_us=lead_us), sched1)
    assert abs(off - offset_us) <= sc.ALIGN_BIN_US and corr > 0.99
    scored = [r for r in rows if r['kind'] != 'indicator']
    assert len(scored) == 15 * 5 * 4 and all(r['verdict'] == 'pass' for r in scored)
    assert sc.exit_code(rows) == 0
    ind = [r for r in rows if r['kind'] == 'indicator']
    assert len(ind) == 2 * 5 * 4
    want = 'indicator_ok' if lead_us > 0 else 'missing'
    assert all(r['verdict'] == want for r in ind if r['visit'] == 0)
    r = next(r for r in scored if r['omega_cmd'] == 0.32 and r['wheel'] == 'FL')
    assert r['counts'] in (671, 672, 673)                                  # spec: about 672 / 10 s
    assert r['f_cmd_hz'] == pytest.approx(K * 0.31983, rel=2e-5)           # spec: omega_d = 0.31983
    assert all(r['f_cmd_hz'] == 0.0 for r in rows if r['kind'] == 'indicator')   # 0.29 -> 0.29054 <= 0.3
    assert abs(r['t_window_end_us'] - r['t_window_start_us'] - 1e7) < READ_STEP_US


def test_small_error_passes_large_error_fails(sched1):
    # at 0.32 rad/s one count is 0.15 % of a window, so a sure pass needs |e| + u <= 0.6 %
    _, _, rows = run(synth(sched1, 5e6, gain=(1.0025, 1, 1, 1)), sched1)
    assert sc.exit_code(rows) == 0
    _, _, rows = run(synth(sched1, 5e6, gain=(1, 1, 1.01, 1)), sched1)
    bad = {r['wheel'] for r in rows if r['verdict'] == 'fail'}
    assert bad == {'RL'} and sc.exit_code(rows) == 1
    assert all(r['verdict'] == 'fail' for r in rows if r['wheel'] == 'RL' and r['kind'] != 'indicator')


def test_reversed_wheel_fails(sched1):
    _, _, rows = run(synth(sched1, 5e6, gain=(1, -1, 1, 1)), sched1)
    fr = [r for r in rows if r['wheel'] == 'FR' and r['kind'] != 'indicator']
    assert all(r['verdict'] == 'fail' and abs(r['err_rel'] + 2) < 0.01 for r in fr)
    flipped = synth(sched1, 5e6, gain=(1, -1, 1, 1))
    reads = sc.Reads(flipped.tb, flipped.ta, flipped.c, count_sign=(1, -1, 1, 1))   # --count-sign
    assert sc.exit_code(sc.score_run(reads, sched1, 5e6)) == 0


def test_swapped_wheels_only_w15_catches(sched1):
    _, _, rows = run(synth(sched1, 5e6, swap=(0, 1)), sched1)
    fails = {(r['kind'], r['wheel']) for r in rows if r['verdict'] == 'fail'}
    assert fails == {('w15', 'FL'), ('w15', 'FR')} and sc.exit_code(rows) == 1


def test_inconclusive_near_limit():
    s = [v for v in t5.make_schedule(1) if v.kind != 'indicator'][:3]
    for i, v in enumerate(s):                       # contiguous from 0
        v.t_start_ms, v.t_end_ms = i * 70_000, (i + 1) * 70_000
    rows = sc.score_run(synth(s, 4e6, gain=(1.006, 1, 1, 1)), s, 4e6)
    fl = [r for r in rows if r['wheel'] == 'FL']
    assert {r['verdict'] for r in fl} == {'inconclusive'} and sc.exit_code(rows) == 3


def test_data_ends_early_is_missing(sched1):
    reads = synth(sched1, 5e6, tail_us=0, end_early_us=40e6)
    rows = sc.score_run(reads, sched1, 5e6)
    last = [r for r in rows if r['visit'] == sched1[-1].visit]
    assert {r['verdict'] for r in last if r['window'] >= 2} == {'missing'}
    assert sc.exit_code(rows) == 3


def test_hole_at_window_edge_is_missing_short_hole_is_not(sched1):
    full = synth(sched1, 5e6)
    v = sched1[4]                                           # +0.5 rad/s
    edge = 5e6 + v.t_start_ms * 1000.0 + 3e7               # start of window 3 = end of window 2
    for hole_us, want in ((150e3, 'missing'), (50e3, 'pass')):
        keep = ~((full.tb >= edge) & (full.tb < edge + hole_us))
        reads = sc.Reads(full.tb[keep], full.ta[keep], full.c[keep])
        rows = [r for r in sc.score_run(reads, sched1, 5e6) if r['visit'] == v.visit]
        assert {r['verdict'] for r in rows if r['window'] in (2, 3)} == {want}
        assert {r['verdict'] for r in rows if r['window'] not in (2, 3)} == {'pass'}


def test_indicator_counts_reported_not_scored(sched1):
    reads = synth(sched1, 5e6)
    v = sched1[0]
    assert v.kind == 'indicator'
    k = (reads.tb > 5e6 + 2.5e7)
    reads.c[k, 2] += 1                                       # one stray count on RL during the indicator
    rows = sc.score_run(reads, sched1, 5e6)
    ind = [r for r in rows if r['visit'] == 0 and r['wheel'] == 'RL']
    assert [r['counts'] for r in ind] == [0, 1, 0, 0, 0]
    assert sc.exit_code(rows) == 0 and sc.summarize(rows)['indicator_counts_total'] == 1


def test_window_mean_skips_nan_and_empty():
    t = np.array([0.0, 1e6, 2e6, 3e6])
    v = np.array([40.0, np.nan, 42.0, 50.0])                 # S line "nan": sensor read failed
    assert sc.window_mean((t, v), 0.0, 3e6) == 41.0
    assert sc.window_mean((t, np.full(4, np.nan)), 0.0, 3e6) == ''
    assert sc.window_mean((t, v), 5e6, 6e6) == '' and sc.window_mean(None, 0.0, 1.0) == ''


def test_uptime_condition():
    assert sc.uptime_condition(1, 119e6) and not sc.uptime_condition(1, 121e6)
    assert sc.uptime_condition(2, 1800e6) and not sc.uptime_condition(2, 1799e6)


def test_clock_fit_recovers_rate_from_packet_lower_envelope():
    rng = np.random.default_rng(3)
    t_esp = np.arange(0, 1200e6, 1000.0)
    lat = np.repeat(rng.uniform(2000, 8000, len(t_esp) // 10), 10)        # 10 lines per UDP packet
    t_rx = 5e9 + t_esp * (1 + 25e-6) + lat
    fit = sc.clock_fit(t_rx, t_esp)
    assert abs(fit['ppm'] - 25.0) < 0.5 and fit['n_bins'] == 120
    assert np.allclose(sc.pi_to_esp(fit, 5e9 + 1e8 * (1 + 25e-6) + 2000), 1e8, atol=500)
    assert sc.clock_fit(t_rx[:2], t_esp[:2]) is None


PI0, PPM = 9e9, 25e-6


def write_events(path, reads, lost=(), temp_c=lambda t: 40.0 + t / 1e8, enc_ovf=0, run_end=True):
    """esp32_events.txt like s06_logger.py: Pi receive time, 10 E lines per packet, an S line each 1 s,
    #run_end with the overflow counters of s06_load end_run, #run_stats with enc_reads_run (S41: the reads
    the producer made in the run, incl. the ones the ring dropped)."""
    rng = np.random.default_rng(7)
    lines = [f'1\t#run_start,timestamp_us={int(reads.tb[0]) - 500},qseq=0']
    t_s = reads.tb[0] - reads.tb[0] % 1e6 + 1e6
    for i in range(0, len(reads.tb), 10):
        t_rx = int(PI0 + reads.ta[min(i + 9, len(reads.tb) - 1)] * (1 + PPM) + rng.uniform(2000, 8000))
        for j in range(i, min(i + 10, len(reads.tb))):
            if j in lost:
                continue
            lines.append(f'{t_rx}\tE,{j + 1},{int(reads.tb[j])},{int(reads.ta[j])},'
                         + ','.join(str(x) for x in reads.c[j]))
        while t_s <= reads.ta[j]:
            lines.append(f'{t_rx}\tS,1,{int(t_s)},1,0,0,0,0,0,10000,0,0,0,0,0,{temp_c(t_s):.1f}')
            t_s += 1e6
    if run_end:
        lines.append(f'{t_rx + 1000}\t#run_end,last_data_seq=1,log_qdrop=0,log_send_err=0,timestamp_us='
                     f'{int(reads.ta[-1])},reason=duration,elapsed_us=0,log_qdrop_run=0,ev_qdrop_run=0,'
                     f'enc_ovf_run={enc_ovf},imu_drop_run=0/0/0,gyro_gap_max_us=0,qseq=9')
        lines.append(f'{t_rx + 1001}\t#run_stats,log_send_err_run=0,enc_reads_run={len(reads.tb) + enc_ovf},qseq=10')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def h7_log(path, up_s):
    with open(path, 'w') as f:
        f.write(f'Init done, entering main loop\nSPI link up at t={int(up_s * 1000)} ms\n')
    return path


def test_unwrap_int32_counts():
    # S17: the int32 count of encoder.c crossing +2^31 is not a step back of 4.29e9 counts
    c = np.array([[2147483640, 0, 0, 0], [2147483647, 0, 0, 0], [-2147483646, 0, 0, 0]], dtype=np.int64)
    r = sc.Reads([0.0, 1.0, 2.0], [0.5, 1.5, 2.5], c)
    assert list(r.c[:, 0]) == [2147483640, 2147483647, 2147483650]
    c = -c                                                   # and the negative bound
    assert list(sc.Reads([0.0, 1.0, 2.0], [0.5, 1.5, 2.5], c).c[:, 0]) == [-2147483640, -2147483647, -2147483650]


@pytest.fixture(scope='module')
def run2(tmp_path_factory):
    """A full locked pass-2 schedule on the ESP32 clock, its events, a Pi temperature file."""
    tmp = tmp_path_factory.mktemp('t5')
    s = t5.make_schedule(2)
    sched_csv = os.path.join(tmp, 'T5_pass2.csv')
    t5.write_csv(sched_csv, s)
    reads = synth(s, 6.2e6)
    pi_temp = os.path.join(tmp, 'pi_temp.txt')
    with open(pi_temp, 'w') as f:                            # Pi clock; 55.000 + t_esp/1e8 °C
        for t in np.arange(reads.tb[0], reads.tb[-1], 5e6):
            f.write(f'{int(PI0 + t * (1 + PPM) + 3000)}\t{int(round(55000 + t / 1e5))}\n')
    return tmp, s, sched_csv, reads, pi_temp


def _cli(run2, name, *extra, lost=(), enc_ovf=0, run_end=True, sched_csv=None, h7_up_s=1900.0):
    import json
    tmp, s, csv_path, reads, pi_temp = run2
    ev = os.path.join(tmp, f'{name}_events.txt')
    write_events(ev, reads, lost=lost, enc_ovf=enc_ovf, run_end=run_end)
    out = os.path.join(tmp, name)
    args = [sys.executable, os.path.join(ROOT, 'tools', 's06_t5_score.py'), '--schedule', sched_csv or csv_path,
            '--esp32-events', ev, '--out', out, '--pi-temp', pi_temp, *extra]
    if h7_up_s is not None:
        args += ['--h7-serial', h7_log(os.path.join(tmp, f'{name}_h7.log'), h7_up_s)]
    p = subprocess.run(args, capture_output=True, text=True)
    summ = json.load(open(os.path.join(out, 'T5_summary.json'), encoding='utf-8')) if p.returncode != 2 else None
    return p.returncode, summ, out, p


def test_cli_full_evidence_passes(run2):
    code, summ, out, p = _cli(run2, 'ok', '--xtal-ppm', '20')
    assert code == 0, p.stdout[-1500:] + p.stderr
    assert summ['run_pass'] == 2 and summ['fail'] == 0 and summ['inconclusive'] == 0 and summ['stream']['ok']
    assert summ['run_condition_ok'] is True and summ['h7_uptime_at_link_up_s'] == 1900.0
    assert abs(summ['esp32_uptime_at_start_s'] - 6.2) <= 0.1      # reported only (S20)
    assert abs(summ['offset_us'] - 6.2e6) <= sc.ALIGN_BIN_US
    assert abs(summ['clock_vs_pi']['ppm'] - 25.0) < 1.0 and summ['temp_esp32'] and summ['temp_pi']
    with open(os.path.join(out, 'T5.csv')) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0])[:1] == ['idx'] and {'wheel', 'omega_cmd', 'window', 't_window_start_us', 'counts',
                                             'f_meas_hz', 'f_cmd_hz', 'err_rel'} <= set(rows[0])
    for r in rows[:200]:                                     # window means of a linear temperature
        mid = float(r['t_window_start_us']) + 5e6
        assert abs(float(r['temp_esp32_c']) - (40.0 + mid / 1e8)) < 0.06
        assert abs(float(r['temp_pi_c']) - (55.0 + mid / 1e8)) < 0.1


def test_cli_lost_read_or_overflow_fails(run2):
    # S4: one lost E line inside the schedule is lost data (Z19), not a note
    code, summ, _, _ = _cli(run2, 'lost', '--xtal-ppm', '20', lost=(100_000,))
    assert code == 1 and summ['eseq_gaps'] == 1 and not summ['stream']['ok']
    code, summ, _, _ = _cli(run2, 'ovf', '--xtal-ppm', '20', enc_ovf=2)
    assert code == 1 and summ['stream']['enc_ovf_run'] == 2


def test_cli_lost_head_or_tail_fails(run2):
    # S41: every E line of the first 100 s (or the last 50) lost: no hole between the eseq received, but
    # fewer lines than the producer read (enc_reads_run) = lost data
    n = len(run2[3].tb)
    code, summ, _, _ = _cli(run2, 'head', '--xtal-ppm', '20', lost=set(range(100_000)))
    assert code == 1 and summ['stream']['run_reads_missing'] == 100_000 and summ['eseq_gaps'] == 0
    code, summ, _, _ = _cli(run2, 'tail', '--xtal-ppm', '20', lost=set(range(n - 50_000, n)))
    assert code == 1 and summ['stream']['run_reads_missing'] == 50_000


def test_cli_missing_evidence_is_not_a_pass(run2):
    assert _cli(run2, 'noxtal')[0] == 3                                        # S19: no --xtal-ppm
    assert _cli(run2, 'noh7', '--xtal-ppm', '20', h7_up_s=None)[0] == 3         # S20: H7 time unknown
    code, summ, _, _ = _cli(run2, 'early', '--xtal-ppm', '20', h7_up_s=6.0)     # pass 2 run 6 s after H7 boot
    assert code == 3 and summ['run_condition_ok'] is False
    assert _cli(run2, 'nouptime', '--xtal-ppm', '20', '--no-uptime-check')[0] == 3
    assert _cli(run2, 'noend', '--xtal-ppm', '20', run_end=False)[0] == 3       # overflow unknown


def test_cli_schedule_not_locked(run2, tmp_path):
    # S18: a continuous schedule that leaves points out is not the acceptance run
    tmp, s, _, reads, _ = run2
    cut = os.path.join(tmp_path, 'T5_pass2.csv')
    t5.write_csv(cut, s[:4])
    code, summ, _, _ = _cli(run2, 'cut', '--xtal-ppm', '20', sched_csv=cut)
    assert code == 3 and summ['schedule_problems']


def test_cli_input_error(run2, tmp_path):
    tmp, s, csv_path, _, _ = run2
    bad = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 's06_t5_score.py'), '--schedule', csv_path,
                          '--esp32-events', csv_path, '--out', str(tmp_path)], capture_output=True, text=True)
    assert bad.returncode == 2
