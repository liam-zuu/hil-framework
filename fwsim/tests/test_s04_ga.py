"""Tests of fwsim/s04_ga.py (spec S04 v5). Synthetic points and S03 folders with a known answer;
the last test needs fwsim/build/fwsim.dll and runs the three phases end to end with a tiny
budget (--tool-check). Run: python -m pytest fwsim/tests -q"""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim  # noqa: E402
import s03_doe as s3  # noqa: E402
import s04_ga as s4  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')


def P(f1, f2, x=(1.0,), seed=1, feasible=True):
    return dict(f1=f1, f2=f2, x=list(x), xga=list(x), ga_seed=seed, feasible=feasible, gen=1, cand_id=0, g1=0, g2=0)


# ── selection, Pareto, HV (sections 4.2, 5) ─────────────────────────────────────────

def test_nondominated_keeps_equal_points():
    pts = [P(0.02, 5.0), P(0.02, 5.0), P(0.03, 4.0), P(0.03, 6.0)]
    nd = s4.nondominated(pts)
    assert [(p['f1'], p['f2']) for p in nd] == [(0.02, 5.0), (0.02, 5.0), (0.03, 4.0)]


def test_select_tolerance_then_f2_then_f1_then_gains_then_seed():
    # f1* = 0.020; 0.0299 is within 0.01 and has the smallest f2 -> chosen over f1*
    assert s4.select([P(0.020, 9.0), P(0.0299, 4.0), P(0.031, 1.0)])['f1'] == 0.0299
    # 0.0301 is outside the tolerance even with a smaller f2
    assert s4.select([P(0.020, 9.0), P(0.0301, 4.0)])['f1'] == 0.020
    # equal f2: smaller f1; equal f1 and f2: lexicographic gains, then the smaller seed
    assert s4.select([P(0.025, 4.0), P(0.022, 4.0)])['f1'] == 0.022
    assert s4.select([P(0.02, 4.0, x=(2.0, 1.0)), P(0.02, 4.0, x=(1.0, 9.0))])['x'] == [1.0, 9.0]
    assert s4.select([P(0.02, 4.0, seed=3), P(0.02, 4.0, seed=2)])['ga_seed'] == 2
    assert s4.select([]) is None


def test_select_filters_dominated_first():
    # (0.025, 5.0) is dominated by (0.021, 4.0); the rule never returns a dominated point
    ch = s4.select([P(0.020, 9.0), P(0.025, 5.0), P(0.021, 4.0)])
    assert (ch['f1'], ch['f2']) == (0.021, 4.0)


def test_hypervolume_staircase_and_reference():
    ref = (0.10, 10.0)
    hv, out = s4.hypervolume([P(0.02, 6.0), P(0.06, 2.0)], ref)
    # (0.10 - 0.02)(10 - 6) + (0.10 - 0.06)(6 - 2)
    assert hv == pytest.approx(0.08 * 4 + 0.04 * 4) and out == 0
    hv, out = s4.hypervolume([P(0.10, 1.0), P(0.05, 12.0)], ref)      # on f1 = D, beyond f2 ref
    assert hv == 0.0 and out == 2
    assert s4.hypervolume([], ref) == (0.0, 0)
    # a dominated point adds nothing
    assert s4.hypervolume([P(0.02, 6.0), P(0.03, 7.0)], ref)[0] == pytest.approx(0.08 * 4)


# ── scoring (section 3) ──────────────────────────────────────────────────────────────

def R(completed='true', status='ok', e=0.02, tv=5.0):
    return dict(completed=completed, status=status, e_perp_max=e, tv_duty=tv)


def test_score_objectives_and_constraints():
    rows = [(0.4, 0.2, R(e=0.03, tv=4.0)), (0.4, 0.5, R(e=0.05, tv=6.0)),
            (0.6, 0.2, R(e=0.02, tv=10.0)), (0.6, 0.5, R(e=0.04, tv=2.0))]
    s = s4.score(rows, False)
    assert s['f1'] == 0.05                                # max over every run
    assert s['f2'] == pytest.approx(6.0)                  # max over mu of the mean over v: max(5, 6)
    assert s['g1'] == 0 and s['g2'] == pytest.approx((0.05 - s4.D) / s4.D)


def test_score_track_error_counts_in_g1_even_when_completed():
    # S04 v5 (B8b-R-1): a track_err run with y1 'true' violates g1; inconclusive and false too
    rows = [(0.5, 0.2, R(status='track_err')), (0.5, 0.5, R('inconclusive', 'track_err')), (0.5, 0.8, R('false'))]
    s = s4.score(rows, False)
    assert s['g1'] == 3 and s['n_track_err'] == 2 and s['n_inconclusive'] == 1 and s['n_false'] == 1


def test_score_run_error_penalty():
    rows = [(0.5, 0.2, R()), (0.5, 0.5, dict(completed='', status='run_error', error='x'))]
    s = s4.score(rows, True)
    assert (s['f1'], s['f2'], s['g1'], s['status']) == (s4.PENALTY, s4.PENALTY, 2.0, 'run_error')


# ── domain, mu rule (sections 2, 3) ──────────────────────────────────────────────────

def _s03_dir(tmp_path, runs=None, trial=False, mu=(0.4, 0.5, 0.6)):
    d = tmp_path / 's03'
    (d / 'sealed').mkdir(parents=True)
    with open(d / 'levels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note'])
        for c in s3.CONTROLLERS:
            for k, n in enumerate(s3.FACTORS[c]):
                L, H = (0.5, 8.0) if k % 2 == 0 else (1.0, 3.0)
                w.writerow([c, n, L, H, '', 'log' if H / L >= 4 else 'lin', '', ''])
    y3 = {f'{c} v={v:g}': 5.0 for c in s3.CONTROLLERS for v in s3.SPEEDS}
    (d / 'meta.yaml').write_text(
        f'trial: {str(trial).lower()}     # x\nstatus: "ran to the end"\nmu_levels: {json.dumps(list(mu))}\n'
        f'gate_speeds: {json.dumps(list(s3.SPEEDS))}\ny3_reference: {json.dumps(y3)}\n', encoding='utf-8')
    if runs is not None:
        s3.write_csv(str(d / 'sealed' / 'runs.csv'), runs)
    return d


def _runs(mod=None):
    out = []
    for pid, box in [(0, 'centre'), (1, 'outer'), (129, 'inner')]:
        for mu in (0.4, 0.5, 0.6):
            for v in s3.SPEEDS:
                r = dict(controller='PID', point_id=pid, box=box, mu=mu, v=v, status='ok', completed='true',
                         e_perp_max=0.03, tv_duty=5.0)
                if mod:
                    mod(r)
                out.append(r)
    return out


def test_domain_and_log_coding(tmp_path):
    d = _s03_dir(tmp_path)
    dom = s4.read_domain(str(d), 'PID')
    lo, hi = s4.ga_bounds(dom)
    assert dom[0]['scale'] == 'log' and lo[0] == pytest.approx(math.log(0.5)) and hi[1] == 3.0
    g = s4.decode(dom, hi)
    assert g[dom[0]['name']] == 8.0 and g[dom[1]['name']] == 3.0          # exp(log H) clipped to H
    mid = (lo + hi) / 2
    assert s4.decode(dom, mid)[dom[0]['name']] == pytest.approx(2.0)      # geometric centre of [0.5, 8]
    assert s4.edge_flags(dom, lo + 0.04 * (hi - lo)) == [x['name'] for x in dom]
    assert s4.edge_flags(dom, mid) == []


def test_mu_rule_mid_when_every_point_satisfies(tmp_path):
    d = _s03_dir(tmp_path, _runs())
    dec, rep = s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert dec == 'mid' and rep['points_checked'] == 9 and rep['points_failing'] == 0


@pytest.mark.parametrize('mod, why', [
    (lambda r: r.update(status='track_err') if r['point_id'] == 129 and r['mu'] == 0.6 and r['v'] == 0.8 else None,
     'track_err'),
    (lambda r: r.update(completed='inconclusive') if r['point_id'] == 1 and r['mu'] == 0.4 else None, 'inconclusive'),
    (lambda r: r.update(completed='false') if r['point_id'] == 0 and r['mu'] == 0.6 else None, 'differs'),
    (lambda r: r.update(e_perp_max=0.0411) if r['point_id'] == 1 and r['mu'] == 0.4 else None, 'dY2'),
    (lambda r: r.update(tv_duty=6.01) if r['point_id'] == 0 and r['mu'] == 0.6 else None, 'dY3'),
])
def test_mu_rule_all_when_one_point_fails(tmp_path, mod, why):
    """S04 3 (v4, B8b-4, B8b-5): one failing point of any box -> evaluate at the three mu."""
    d = _s03_dir(tmp_path, _runs(mod))
    dec, rep = s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert dec == 'all' and rep['points_failing'] >= 1 and any(why in w for w in rep['first'][0]['why'])


def test_mu_rule_tolerances_are_inclusive(tmp_path):
    def mod(r):
        if r['point_id'] == 1 and r['mu'] == 0.4:
            r.update(e_perp_max=0.03 + 0.01 - 1e-12, tv_duty=5.0 + 0.2 * 5.0 - 1e-9)
    dec, _ = s4.mu_rule(str(_s03_dir(tmp_path, _runs(mod))), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert dec == 'mid'


def test_mu_rule_only_gate_speeds(tmp_path):
    d = _s03_dir(tmp_path, _runs(lambda r: r.update(status='track_err') if r['v'] == 0.8 else None))
    assert s4.mu_rule(str(d), 'PID', [0.2, 0.5], [0.4, 0.5, 0.6])[0] == 'mid'
    assert s4.mu_rule(str(d), 'PID', [0.2, 0.5, 0.8], [0.4, 0.5, 0.6])[0] == 'all'


# ── seeds, summary (sections 3, 5) ───────────────────────────────────────────────────

def test_summary_union_and_infeasible_seeds():
    dom = [dict(name='a', L=1.0, H=2.0, scale='lin')]
    finals = {1: [P(0.020, 9.0, x=(1.1,), seed=1), P(0.05, 3.0, x=(1.2,), seed=1)],
              2: [P(0.022, 4.0, x=(1.3,), seed=2)],
              3: [P(0.01, 1.0, x=(1.4,), seed=3, feasible=False)]}
    rows, stats = s4.summarise('PID', dom, finals)
    per = [r for r in rows if r['kind'] == 'per_seed']
    assert [r['feasible'] for r in per] == [True, True, False]
    send = rows[-1]
    # per seed: seed 1 picks (0.020, 9.0) (0.05 is out of tolerance); union: (0.022, 4.0) of seed 2
    assert per[0]['f1'] == 0.020 and send['kind'] == 'send_to_HIL' and send['f1'] == 0.022 and send['ga_seed'] == 2
    assert send['a'] == repr(1.3) and float(send['a_f32']) == float(np.float32(1.3))
    assert stats['n_feasible_seeds'] == 2 and stats['per_seed_selected']['f1']['median'] == pytest.approx(0.021)


def test_summary_stops_when_more_than_half_infeasible():
    dom = [dict(name='a', L=1.0, H=2.0, scale='lin')]
    finals = {s: [P(0.02, 4.0, seed=s, feasible=s <= 2)] for s in range(1, 6)}
    rows, stats = s4.summarise('PID', dom, finals)
    assert rows[-1]['feasible'] is False and 'tell Liam' in rows[-1]['note'] and stats['n_feasible_seeds'] == 2
    finals = {s: [P(0.02, 4.0, seed=s, feasible=s <= 3)] for s in range(1, 6)}
    assert s4.summarise('PID', dom, finals)[0][-1]['feasible'] is True


def test_seed_error_rate_counts_first_attempts_only():
    ev = [dict(controller='PID', ga_seed=1, attempt=a, completed=c, status=st)
          for a, c, st in [(1, 'true', 'ok')] * 9 + [(1, '', 'run_error'), (2, 'true', 'ok'), (1, 'inconclusive', 'track_err')]]
    r = s4.seed_error_rates(ev, 'PID')[1]
    assert r['n'] == 11 and r['rate'] == pytest.approx(2 / 11) and r['over_10pct']


def test_meta_value():
    t = 'trial: true     # x\nmu_levels: [0.5]\nstatus: "ran to the end"\n'
    assert s4.meta_value(t, 'trial') is True and s4.meta_value(t, 'mu_levels') == [0.5]
    assert s4.meta_value(t, 'status') == 'ran to the end'


# ── guards (sections 4.2, 4.3, L56) ──────────────────────────────────────────────────

@pytest.mark.parametrize('extra, msg', [
    ([], 'main S03 run'),                                          # trial S03 without --tool-check
])
def test_guard_needs_main_s03(tmp_path, monkeypatch, extra, msg):
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,))
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--controller', 'PID', '--s03', str(d),
                                      '--out', str(tmp_path / 'o')] + extra)
    with pytest.raises(SystemExit, match=msg):
        s4.main()


def test_guard_l56_and_preconditions_and_budget(tmp_path, monkeypatch):
    d = _s03_dir(tmp_path, _runs())
    base = ['s04_ga.py', '--phase', 'trial', '--controller', 'PID', '--s03', str(d), '--out', str(tmp_path / 'o')]
    monkeypatch.setattr(sys, 'argv', base)
    with pytest.raises(SystemExit, match='L56'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', base + ['--pop', '6'])
    with pytest.raises(SystemExit, match='budget is locked'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', base + ['--seeds', '1', '2'])
    with pytest.raises(SystemExit, match='seeds 1-5'):
        s4.main()
    if os.path.isfile(_DLL):
        pre = tmp_path / 'pre.json'
        pre.write_text(json.dumps(dict(s03_run_and_locked='x', final_test_set_locked='', ctrl_imu_checked_on_board='y',
                                       x3_compared='z')))
        monkeypatch.setattr(sys, 'argv', base + ['--s03-unsealed', '--dll', _DLL, '--preconditions', str(pre)])
        with pytest.raises(SystemExit, match='final_test_set_locked'):
            s4.main()


# ── end to end, tiny budget (needs the DLL) ──────────────────────────────────────────

@needs_dll
def test_tool_check_three_phases(tmp_path, monkeypatch, capsys):
    """trial -> lock.yaml, main (2 seeds) -> selected.csv with a send_to_HIL row whose gains read back
    from the file reproduce every GA run bit for bit in a new process (post)."""
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,))
    out = tmp_path / 'run'
    common = ['--tool-check', '--controller', 'PID', '--s03', str(d), '--pop', '4', '--gen-base', '1',
              '--gen-max', '2', '--dll', _DLL, '--out', str(out)]
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial'] + common)
    assert s4.main() in (0, 2)
    lock = json.load(open(out / 'lock.yaml'))
    L = lock['controllers']['PID']
    assert L['eval_mu'] == [0.5] and L['n_gen'] in (1, 2) and L['hv_ref'][0] == s4.D and lock['tool_check']
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'main', '--seeds', '1', '2'] + common)
    s4.main()
    evals = list(csv.DictReader(open(out / 'evals.csv')))
    keys = [(e['controller'], e['ga_seed'], e['gen'], e['cand_id'], e['v'], e['mu'], e['attempt']) for e in evals]
    assert len(keys) == len(set(keys))                                   # S04 8: unique key
    sel = list(csv.DictReader(open(out / 'selected.csv')))
    send = [r for r in sel if r['kind'] == 'send_to_HIL'][0]
    hist = list(csv.DictReader(open(out / 'history.csv')))
    h = next(r for r in hist if (r['ga_seed'], r['gen'], r['cand_id']) == (send['ga_seed'], send['gen'], send['cand_id']))
    assert float(h['f1']) == float(send['f1'])                           # selected row is a GA candidate
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'post', '--dll', _DLL, '--out', str(out)])
    assert s4.post(s4_args(out)) == 0
    det = list(csv.DictReader(open(out / 'post_checks' / 'determinism.csv')))
    assert det and all(r['equal'] == 'true' for r in det)
    sens = list(csv.DictReader(open(out / 'post_checks' / 'sensitivity.csv')))
    assert {r['r_factor'] for r in sens} == {repr(k) for k in s4.R_FACTORS}
    changed = [json.loads(r['plant_changed']) for r in sens if r['r_factor'] != '1.0']
    assert changed and all(set(c) == {'motor_R'} for c in changed)       # only R changes (S04 6)


def s4_args(out):
    import argparse
    return argparse.Namespace(out=str(out), dll=_DLL)
