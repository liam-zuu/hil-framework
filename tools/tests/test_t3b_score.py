"""tools/t3b_score.py (S06 T3b scoring) and tools/t3b_sensitivity.py (Z23) on the PC.

Synthetic runs on one clock: 1 ms frames, counts read 1 ms apart. The scorer must give the
same N_tol as rpi5/t3b_schedule.py (which matches the 4 spec examples, test_t3b_schedule.py),
pass exact counts, and catch a backward step, a count error above N_tol, a wrong number of
_start_wheel/_stop calls; a count error inside the time uncertainty is inconclusive.
Run: python -m pytest tools/tests"""
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


@pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not on PATH')
def test_sensitivity_quick(tmp_path):
    """Z23 on the first 20 s of the seed-1 schedule: (0) all pass, old code caught, gate PASS."""
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 't3b_sensitivity.py'), '--out', str(tmp_path),
                        '--max-ms', '20000'], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'Z23 gate' in p.stdout and 'PASS' in p.stdout.splitlines()[-1]
    old = open(tmp_path / 'T3b_old.csv').read()
    assert ',fail\n' in old
