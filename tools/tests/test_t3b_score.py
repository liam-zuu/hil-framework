"""tools/t3b_score.py (S06 T3b scoring) and tools/t3b_sensitivity.py (Z23) on the PC.

Synthetic runs on one clock: 1 ms frames, counts read 1 ms apart. The scorer must give the
same N_tol as rpi5/t3b_schedule.py (which matches the 4 spec examples, test_t3b_schedule.py),
pass exact counts, and catch a backward step, a count error above N_tol, a wrong number of
_start_wheel/_stop calls; a count error inside the time uncertainty is inconclusive.
Run: python -m pytest tools/tests"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import t3b_schedule as ts  # noqa: E402
import t3b_score as sc  # noqa: E402

READ_PHASE, READ_SPAN = 310.0, 30.0


def frames(sched, tail=50):
    tab = ts.omega_table(sched)
    n = len(tab[0]) + tail
    w_d = np.zeros((n, 4))
    for w in range(4):
        for k, v in enumerate(tab[w]):
            w_d[k, w] = ts.quantize(v)
    return np.arange(n) * 1000.0, w_d


def exact_counts(t_send, w_d, tb, ta):
    """Count = rounded integral of omega_eff up to the middle of each read (no delay)."""
    weff = np.where(np.abs(w_d) <= ts.DEADBAND, 0.0, w_d)
    cum = np.concatenate([[np.zeros(4)], np.cumsum(weff * 1e-3, axis=0)])
    t = 0.5 * (tb + ta)
    k = np.clip((t // 1000).astype(int), 0, len(t_send) - 1)
    frac = (t - k * 1000.0) * 1e-6
    return np.rint(sc.K * (cum[k] + weff[k] * frac[:, None])).astype(np.int64)


@pytest.fixture(scope='module')
def small():
    sched = [r for r in ts.make_schedule(1) if r.slot_end_ms <= 20000]
    t_send, w_d = frames(sched)
    tb = np.arange(READ_PHASE, t_send[-1], 1000.0)
    ta = tb + READ_SPAN
    return sched, t_send, w_d, tb, ta


def test_tolerance_matches_schedule_module():
    sched = ts.make_schedule(1)
    t_send, w_d = frames(sched)
    run = sc.Run(t_send, w_d, t_send + READ_PHASE, t_send + READ_PHASE + READ_SPAN, np.zeros((len(t_send), 4)))
    for r, ref in zip(sched, ts.tolerances(sched)):
        y = run.tolerance(r.wheel, t_send[r.win_start_ms], t_send[r.win_end_ms])
        assert y['n_start'] == ref['n_start'] and y['n_stop_intervals'] == ref['n_stop_intervals'], r
        assert abs(y['n_tol'] - ref['n_tol']) < 0.1, (r, y['n_tol'], ref['n_tol'])   # one frame at the edge


def test_exact_counts_pass(small):
    sched, t_send, w_d, tb, ta = small
    rows = sc.score_run(sc.Run(t_send, w_d, tb, ta, exact_counts(t_send, w_d, tb, ta)), sched)
    assert {x['verdict'] for x in rows} == {'pass'}
    assert all(x['backward_steps'] == 0 and x['u_t_ok'] for x in rows)
    assert sc.exit_code(rows) == 0


def test_wrong_count_sign_fails(small):
    sched, t_send, w_d, tb, ta = small
    c = exact_counts(t_send, w_d, tb, ta)
    rows = sc.score_run(sc.Run(t_send, w_d, tb, ta, c, count_sign=(1, -1, 1, 1)), sched)
    assert all(x['verdict'] == 'fail' for x in rows if x['wheel'] == 1 and abs(x['count_cmd']) > 50)


def _pick(sched, kind):
    return next(r for r in sched if r.kind == kind)


def test_backward_step_caught(small):
    sched, t_send, w_d, tb, ta = small
    r = _pick(sched, 'ramp_0.4_10')
    c = exact_counts(t_send, w_d, tb, ta)
    i = int(np.searchsorted(tb, (r.t_end_ms + 100) * 1000.0))      # running at 10 rad/s, no event near
    c[i, r.wheel] -= 3          # one read 3 counts low: one interval backwards, the next one +5
    rows = {x['id']: x for x in sc.score_run(sc.Run(t_send, w_d, tb, ta, c), sched)}
    x = rows[r.id]
    assert x['backward_steps'] == 1 and x['verdict'] == 'fail', x


def test_backward_step_at_event_excluded(small):
    sched, t_send, w_d, tb, ta = small
    r = _pick(sched, 'reverse_10_-10')
    c = exact_counts(t_send, w_d, tb, ta)
    # first read after the reversal: the interval (i, i+1) starts inside [event; +1 ms] and the command
    # over it has one sign (the new one); the counts still move the old way there (H7 delay)
    i = int(np.searchsorted(tb, t_send[r.t0_ms]))
    assert tb[i] - t_send[r.t0_ms] < sc.EVENT_EXCL_US
    c[i + 1:, r.wheel] += 3 if r.w_from > 0 else -3
    x = {y['id']: y for y in sc.score_run(sc.Run(t_send, w_d, tb, ta, c), sched)}[r.id]
    assert x['backward_steps'] == 0 and x['verdict'] == 'pass', x
    # the same 3 counts 5 ms later are a backward step
    c = exact_counts(t_send, w_d, tb, ta)
    c[i + 6:, r.wheel] += 3 if r.w_from > 0 else -3
    x = {y['id']: y for y in sc.score_run(sc.Run(t_send, w_d, tb, ta, c), sched)}[r.id]
    assert x['backward_steps'] == 1, x


def test_count_error_over_ntol_fails_and_inside_u_is_inconclusive(small):
    sched, t_send, w_d, tb, ta = small
    r = _pick(sched, 'step_up_2_39.2')
    base = sc.Run(t_send, w_d, tb, ta, exact_counts(t_send, w_d, tb, ta))
    x0 = {y['id']: y for y in sc.score_run(base, sched)}[r.id]
    ib = int(np.searchsorted(tb, t_send[r.win_end_ms]))
    c = exact_counts(t_send, w_d, tb, ta)
    c[ib:, r.wheel] += int(np.ceil(x0['n_tol'])) + 1
    x = {y['id']: y for y in sc.score_run(sc.Run(t_send, w_d, tb, ta, c), sched)}[r.id]
    assert x['verdict_b'] == 'fail' and x['verdict'] == 'fail', x
    # error just at N_tol with a wide read span (u_t large): inconclusive
    c = exact_counts(t_send, w_d, tb, ta)
    c[ib:, r.wheel] += int(round(x0['n_tol']))
    wide = sc.Run(t_send, w_d, tb, tb + 400.0, c, delta_sync_us=200.0)
    x = {y['id']: y for y in sc.score_run(wide, sched)}[r.id]
    assert x['verdict_b'] == 'inconclusive' and x['verdict'] == 'inconclusive', x
    # tool requirement u_t <= 1/4 (N_tol - fixed): about 5.7 counts here, broken by a 1.5 ms read span
    wider = sc.Run(t_send, w_d, tb, tb + 1500.0, c)
    x = {y['id']: y for y in sc.score_run(wider, sched)}[r.id]
    assert not x['u_t_ok'] and sc.exit_code([x]) == 3, x


def test_calls_mismatch_fails(small):
    sched, t_send, w_d, tb, ta = small
    r = _pick(sched, 'reverse_10_-10')
    c = exact_counts(t_send, w_d, tb, ta)
    t_ev = t_send[r.t0_ms] + 400.0
    good = {w: [] for w in range(4)}
    good[r.wheel] = [('P', t_ev, 0), ('S', t_ev, 123)]
    x = {y['id']: y for y in sc.score_run(sc.Run(t_send, w_d, tb, ta, c, calls=good), sched)}[r.id]
    assert x['verdict'] == 'pass' and x['n_calls_actual'] == 2 and x['start_cnt'] == '123', x
    bad = {w: [] for w in range(4)}
    bad[r.wheel] = [('P', t_ev, 0), ('S', t_ev, 1), ('S', t_ev + 1, 2)]       # one start too many
    x = {y['id']: y for y in sc.score_run(sc.Run(t_send, w_d, tb, ta, c, calls=bad), sched)}[r.id]
    assert x['verdict'] == 'fail', x


def test_int32_counts_unwrapped(small):
    # S17: the int32 count crossing +2^31 inside a window is no backward step and no count error
    sched, t_send, w_d, tb, ta = small
    c = exact_counts(t_send, w_d, tb, ta) + (2**31 - 3000)
    c = (c + 2**31) % 2**32 - 2**31                                          # as the firmware prints it
    rows = sc.score_run(sc.Run(t_send, w_d, tb, ta, c), sched)
    assert {x['verdict'] for x in rows} == {'pass'}


def test_empty_and_lag_and_locked_schedule(small):
    sched, t_send, w_d, tb, ta = small
    assert sc.exit_code([]) == 3                                             # S18: nothing scored, no pass
    assert sc.schedule_lag(w_d, sched) == (0, 1.0)
    lagged = np.vstack([np.zeros((1, 4)), w_d[:-1]])                         # the HIL path: one frame later
    assert sc.schedule_lag(lagged, sched)[0] == 1
    assert sc.schedule_matches_locked(ts.make_schedule(1)) == []
    assert sc.schedule_matches_locked(sched) != []                           # a cut schedule is not the locked one


def _cli_run(tmp_path, small, lost=None, lag=1, extra=(), manifest=None):
    """A run as run_hil --omega-schedule writes it: frame k carries entry k - lag; ESP32 clock = Pi clock."""
    from protocol import encode_state
    sched, t_send, w_d, tb, ta = small
    sent = np.vstack([np.zeros((lag, 4)), w_d[:len(w_d) - lag]]) if lag else w_d
    mosi = np.array([np.frombuffer(encode_state(list(sent[k]) + [0.0] * 6, seq=k), dtype=np.uint8)
                     for k in range(len(t_send))])
    log = {} if manifest is None else dict(manifest=np.array(json.dumps(manifest)))
    np.savez(tmp_path / 'frames.npz', t_send_ns=(t_send * 1000).astype(np.int64), mosi=mosi,
             miso=np.zeros_like(mosi), **log)
    c = exact_counts(t_send, sent, tb, ta)
    lines = [f'{int(tb[0])}\t#run_start,timestamp_us={int(tb[0])},qseq=0']
    for j in range(len(tb)):
        if j != lost:
            lines.append(f'{int(ta[j])}\tE,{j + 1},{int(tb[j])},{int(ta[j])},' + ','.join(str(x) for x in c[j]))
    lines.append(f'{int(ta[-1]) + 10}\t#run_end,ev_qdrop_run=0,enc_ovf_run=0,qseq=1')
    lines.append(f'{int(ta[-1]) + 11}\t#run_stats,log_send_err_run=0,enc_reads_run={len(tb)},qseq=2')
    (tmp_path / 'ev.txt').write_text('\n'.join(lines) + '\n')
    ts.write_csv(str(tmp_path / 'sched.csv'), sched)
    return sc.main(['--schedule', str(tmp_path / 'sched.csv'), '--frames', str(tmp_path / 'frames.npz'),
                    '--esp32-events', str(tmp_path / 'ev.txt'), '--clock-affine', '0', '1', '--delta-sync-us', '0',
                    '--out', str(tmp_path / 'out'), *extra])


def test_cli_frame_log_checked(tmp_path, small, capsys):
    # S81: a manifest saying more steps than the file holds fails T3b, with or without --h7-calls
    n = len(small[1])
    assert _cli_run(tmp_path, small, manifest=dict(step_first=0, step_last=n, steps=n + 1)) == 1
    assert 'mất frame' in capsys.readouterr().out
    assert _cli_run(tmp_path, small, manifest=dict(step_first=0, step_last=n - 1, steps=n)) == 3   # no --h7-calls
    assert 'không có manifest' not in capsys.readouterr().out
    assert _cli_run(tmp_path, small) == 3
    assert 'không có manifest' in capsys.readouterr().out


def test_cli_run_that_cannot_be_chosen(tmp_path, small):
    # S71: --run of a run not in the file stops before scoring, it does not read every E line
    assert _cli_run(tmp_path, small, extra=('--run', '7')) == 2


def test_cli_lag_found_and_stream_checked(tmp_path, small, capsys):
    # S16: windows anchored on the lag seen in the MOSI; S15/S18: no H7 calls and a cut schedule -> 3
    assert _cli_run(tmp_path, small) == 3
    out = capsys.readouterr().out
    assert 'trễ lịch → frame: 1 frame' in out and 'KHÔNG ĐẠT' not in out, out[-2000:]
    # S4: one lost E line inside the windows is lost data
    assert _cli_run(tmp_path, small, lost=5000) == 1
    # S41: the first E line of the run lost: no hole between the eseq received, but one read missing
    assert _cli_run(tmp_path, small, lost=0) == 1


@pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not on PATH')
def test_sensitivity_quick(tmp_path):
    """Z23 on the first 20 s of the seed-1 schedule: (0) all pass, old code caught, gate PASS."""
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 't3b_sensitivity.py'), '--out', str(tmp_path),
                        '--max-ms', '20000'], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'Z23 gate' in p.stdout and 'PASS' in p.stdout.splitlines()[-1]
    old = open(tmp_path / 'T3b_old.csv').read()
    assert ',fail\n' in old


def test_truth_labels_from_the_ideal_generator(small):
    # S29 (c): a harness window whose count deviates from the ideal generator by more than N_tol + u_t is
    # labelled observable from the truth, before any verdict is read; the others are not
    import t3b_sensitivity as sens
    sched, t_send, w_d, tb, ta = small
    c = exact_counts(t_send, w_d, tb, ta)
    ideal = sc.Run(t_send, w_d, tb, ta, c)
    r = _pick(sched, 'reverse_10_-10')
    bad = c.copy()
    i_mid = int(np.searchsorted(tb, t_send[r.t0_ms] + 300.0))
    bad[i_mid:, r.wheel] += 500                                              # a jump of 500 counts on that wheel
    harness = sc.Run(t_send, w_d, tb, ta, bad)
    rows_i, rows_h = sc.score_run(ideal, sched), sc.score_run(harness, sched)
    sens.truth_big(rows_h, rows_i, ideal, harness)
    x = {y['id']: y for y in rows_h}[r.id]
    assert x['obs_big'] and x['e_truth'] == 500
    assert all(y['obs_big'] is not None for y in rows_h)
    assert not any(y['obs_big'] for y in rows_h if y['wheel'] != r.wheel)
