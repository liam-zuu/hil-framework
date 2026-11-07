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

def _head():
    import subprocess
    return subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def _s03_dir(tmp_path, runs=None, trial=False, mu=(0.4, 0.5, 0.6), levels=None, plant_src=None, o1c=None,
             script=None, imu=None):
    """levels: {controller: {factor: (L, H)}}; default a synthetic domain. plant_src, o1c, imu, script:
    the S03 source record (S4C-2); default the source of this checkout (HEAD, clean)."""
    d = tmp_path / 's03'
    (d / 'sealed').mkdir(parents=True)
    with open(d / 'levels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note'])
        for c in s3.CONTROLLERS:
            for k, n in enumerate(s3.FACTORS[c]):
                L, H = levels[c][n] if levels else ((0.5, 8.0) if k % 2 == 0 else (1.0, 3.0))
                w.writerow([c, n, L, H, '', 'log' if L > 0 and H / L >= 4 else 'lin', '', ''])
    with open(d / 'design.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'point_id', 'box'])
        for c in s3.CONTROLLERS:
            for pid, box in [(0, 'centre'), (1, 'outer'), (129, 'inner')]:
                w.writerow([c, pid, box])
    y3 = {f'{c} v={v:g}': 5.0 for c in s3.CONTROLLERS for v in s3.SPEEDS}
    dll_sha = s3.sha256_file(_DLL) if os.path.isfile(_DLL) else 'no dll'
    (d / 'meta.yaml').write_text(
        f'trial: {str(trial).lower()}     # x\nstatus: "ran to the end"\nmu_levels: {json.dumps(list(mu))}\n'
        f'script: {json.dumps(script if script is not None else dict(commit=_head(), dirty=False))}\n'
        f'firmware:\n  dll_sha256: {dll_sha}\n'
        f'plant: {json.dumps(dict(motor_model="dc", src=plant_src or fwsim.PLANT_SRC))}\n'
        f'o1: {json.dumps(o1c or s4.o1_constants())}\nimu: {json.dumps(imu or s3.IMU)}\n'
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


@pytest.mark.parametrize('e, tv, dec', [
    (0.04, 6.0, 'mid'),            # exactly on both bounds: 0.04 - 0.03 = 0.010000000000000002 in binary64 (S4C-14)
    (0.02, 4.0, 'mid'),            # the bounds below the middle mu
    (0.0401, 5.0, 'all'),          # just outside dY2
    (0.03, 6.001, 'all'),          # just outside dY3 (20 % of the reference 5.0)
])
def test_mu_rule_tolerances_are_inclusive_at_the_bound(tmp_path, e, tv, dec):
    def mod(r):
        if r['point_id'] == 1 and r['mu'] == 0.4:
            r.update(e_perp_max=e, tv_duty=tv)
    assert s4.mu_rule(str(_s03_dir(tmp_path, _runs(mod))), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])[0] == dec


def test_mu_rule_refuses_missing_or_duplicate_runs(tmp_path):
    """S4C-5: a design point missing at a gate speed, a duplicate run or a point not in design.csv
    stops the rule; it never concludes 'mid' on a partial S03 table."""
    runs = [r for r in _runs() if not (r['point_id'] == 129 and r['v'] == 0.8)]
    with pytest.raises(SystemExit, match='no run of PID point 129'):
        s4.mu_rule(str(_s03_dir(tmp_path / 'a', runs)), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    runs = _runs() + [dict(_runs()[0], status='ok')]
    with pytest.raises(SystemExit, match='duplicate'):
        s4.mu_rule(str(_s03_dir(tmp_path / 'b', runs)), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    runs = _runs() + [dict(r, point_id=7) for r in _runs() if r['point_id'] == 1]
    with pytest.raises(SystemExit, match='not in design.csv'):
        s4.mu_rule(str(_s03_dir(tmp_path / 'c', runs)), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])


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


@pytest.mark.parametrize('order', [0, 1, 2])
def test_summary_union_is_not_two_level(order):
    """S4C-15: seed 1 has A (0.020, 3) and B (0.029, 1) -> picks B; seed 2 has C (0.015, 4). On the
    union of the sets f1* = 0.015, keep f1 <= 0.025: A wins (f2 3). Choosing among the per-seed picks
    (B, C) would give C. Point order inside a seed must not matter."""
    dom = [dict(name='a', L=1.0, H=2.0, scale='lin')]
    s1 = [P(0.020, 3.0, x=(1.1,), seed=1), P(0.029, 1.0, x=(1.2,), seed=1)]
    if order:
        s1 = s1[::-1]
    finals = {1: s1, 2: [P(0.015, 4.0, x=(1.3,), seed=2)]}
    if order == 2:
        finals = {2: finals[2], 1: finals[1]}
    rows, _ = s4.summarise('PID', dom, finals)
    per = {r['ga_seed']: r for r in rows if r['kind'] == 'per_seed'}
    assert per[1]['f1'] == 0.029 and per[2]['f1'] == 0.015
    send = rows[-1]
    assert (send['f1'], send['f2'], send['ga_seed']) == (0.020, 3.0, 1)


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
    monkeypatch.setattr(sys, 'argv', base + ['--s03-unsealed', '--gate-speeds', '0.2'])       # S4C-3
    with pytest.raises(SystemExit, match='gate speeds are those of the S03 run'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', base + ['--tool-check', '--seeds', '1', '1'])            # S4C-13
    with pytest.raises(SystemExit, match='repeated seed'):
        s4.main()
    if os.path.isfile(_DLL):
        for bad in ('', None, False, 0, [], {}):                                             # S4C-4
            pre = tmp_path / 'pre.json'
            pre.write_text(json.dumps(dict(s03_run_and_locked='x', final_test_set_locked=bad,
                                           ctrl_imu_checked_on_board='y', x3_compared='z')))
            monkeypatch.setattr(sys, 'argv', base + ['--s03-unsealed', '--dll', _DLL, '--preconditions', str(pre)])
            with pytest.raises(SystemExit, match='final_test_set_locked'):
                s4.main()


def _lock(out, d, tool_check_common, tool_check_ctrl=None, fp=None, pre=None, budget=None):
    pop, base, mx = budget or (s4.POP, s4.GEN_BASE, s4.GEN_MAX)
    L = dict(tool_check=tool_check_common if tool_check_ctrl is None else tool_check_ctrl,
             domain=s4.read_domain(str(d), 'PID'), s03_run=os.path.abspath(str(d)), speeds=list(s3.SPEEDS),
             eval_mu=[0.5], pop=pop, n_gen=base, gen_rule=dict(base=base, max=mx), seeds=list(s4.SEEDS),
             preconditions=pre if pre is not None else {k: 'x' for k in s4.PRECONDITIONS}, hv_ref=[s4.D, 50.0],
             stop=None)
    out.mkdir(parents=True, exist_ok=True)
    unsealed = not tool_check_common
    lock = dict(controllers=dict(PID=L), common=dict(tool_check=tool_check_common, s03_run=os.path.abspath(str(d)),
                                                    fingerprint=fp or s4.fingerprint(_DLL, str(d), unsealed=unsealed),
                                                    s03_mu_levels=[0.4, 0.5, 0.6], unsealed=unsealed),
                s03_mu_levels=[0.4, 0.5, 0.6])
    (out / 'lock.yaml').write_text(json.dumps(lock))


@needs_dll
def test_guard_one_mode_and_one_source_per_folder(tmp_path, monkeypatch):
    """S4C-1: the trial and main guards refuse a lock.yaml (written by hand here) whose mode, budget,
    preconditions or fingerprint do not fit the run. The fingerprint against real changed files,
    through a real trial and the post phase: test_lock_from_a_real_trial_catches_changed_sources."""
    d = _s03_dir(tmp_path, _runs())
    out = tmp_path / 'o'
    pre = tmp_path / 'pre.json'
    pre.write_text(json.dumps({k: 'checked in x' for k in s4.PRECONDITIONS}))
    main_trial = ['s04_ga.py', '--phase', 'trial', '--controller', 'ADRC', '--s03', str(d), '--s03-unsealed',
                  '--preconditions', str(pre), '--dll', _DLL, '--out', str(out)]
    _lock(out, d, True)                                    # PID trial was a tool check
    monkeypatch.setattr(sys, 'argv', main_trial)
    with pytest.raises(SystemExit, match='never share a folder'):
        s4.main()
    main_pid = ['s04_ga.py', '--phase', 'main', '--controller', 'PID', '--s03', str(d), '--s03-unsealed',
                '--dll', _DLL, '--out', str(out)]
    _lock(out, d, False, tool_check_ctrl=True)             # common flag forged to false, PID locked as a tool check
    monkeypatch.setattr(sys, 'argv', main_pid)
    with pytest.raises(SystemExit, match='differ in --tool-check'):
        s4.main()
    _lock(out, d, False, budget=(6, 3, 5))                 # main flag but a small budget in the lock
    with pytest.raises(SystemExit, match='not a main-run lock'):
        s4.main()
    _lock(out, d, False, pre={})                           # main flag but no preconditions
    with pytest.raises(SystemExit, match='not a main-run lock'):
        s4.main()
    fp = s4.fingerprint(_DLL, str(d), unsealed=True)
    _lock(out, d, False, fp=dict(fp, dll_sha256='0' * 64))
    with pytest.raises(SystemExit, match='dll_sha256'):
        s4.main()
    _lock(out, d, False, fp=dict(fp, s03_files=dict(fp['s03_files'], **{'levels.csv': 'x'})))
    with pytest.raises(SystemExit, match='s03_files'):
        s4.main()


def _levels_near_defaults():
    fw = fwsim.FirmwareSim(_DLL)                  # a domain around the firmware defaults: both controllers feasible
    return {c: {n: ((v * 0.8, v * 1.25) if v > 0 else (0.0, 0.1)) for n, v in s3.defaults(fw, c).items()}
            for c in s3.CONTROLLERS}


@needs_dll
def test_lock_from_a_real_trial_catches_changed_sources(tmp_path, monkeypatch):
    """S4C-2, S4C-16: the lock is written by a real trial; one changed byte in a real S03 file
    (levels.csv) stops main, and with the file restored main runs; one changed byte in design.csv
    stops post, and with it restored post runs; a trial whose S03 meta.yaml records another DLL
    SHA-256 stops (the S03 record is changed here, not the DLL file). The other S03 source checks
    of the first trial: test_first_trial_refuses_a_source_s03_did_not_use."""
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,), levels=_levels_near_defaults())
    out = tmp_path / 'run'
    common = ['--tool-check', '--controller', 'PID', '--s03', str(d), '--pop', '4', '--gen-base', '1',
              '--gen-max', '1', '--dll', _DLL, '--out', str(out)]
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial'] + common)
    assert s4.main() in (0, 2)
    lv = d / 'levels.csv'
    orig = lv.read_bytes()
    lv.write_bytes(orig + b'\n')                                       # same content, other bytes
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'main', '--seeds', '1'] + common)
    with pytest.raises(SystemExit, match='s03_files'):
        s4.main()
    lv.write_bytes(orig)
    s4.main()                                                          # unchanged sources: main runs
    dz = d / 'design.csv'
    dorig = dz.read_bytes()
    dz.write_bytes(dorig + b'\n')
    with pytest.raises(SystemExit, match='s03_files'):
        s4.post(s4_args(out))
    dz.write_bytes(dorig)
    assert s4.post(s4_args(out)) == 0                                  # unchanged sources: post runs
    meta = d / 'meta.yaml'
    meta.write_text(meta.read_text(encoding='utf-8').replace(s3.sha256_file(_DLL), '0' * 64), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller', 'ADRC',
                                      '--s03', str(d), '--pop', '4', '--gen-base', '1', '--gen-max', '1',
                                      '--dll', _DLL, '--out', str(tmp_path / 'run2')])
    with pytest.raises(SystemExit, match='not the DLL of the S03 run'):
        s4.main()


def _other_blob(b):
    return ('0' if b[0] != '0' else '1') + b[1:]


@needs_dll
@pytest.mark.parametrize('kw, msg', [
    (dict(plant_src='plant'), 'not the S03 plant.src'),                       # params.py changed after S03
    (dict(o1c='D'), 'not the S03 o1'),                                         # an O1 constant changed after S03
    (dict(imu='rate'), 'not the S03 imu'),                                     # s03_doe.IMU changed after S03
    (dict(script='dirty'), 'no clean script commit'),
    (dict(script='unknown'), 'cannot be read in this repository'),             # git cannot read the commit
])
def test_first_trial_refuses_a_source_s03_did_not_use(tmp_path, monkeypatch, kw, msg):
    """S4C-2 (Codex rounds 3, 4): the first trial checks the S03 meta.yaml, so it cannot lock a
    plant, O1 constants or IMU configuration that the S03 run did not use, nor run when the S03 commit
    is not clean or not readable; these stop a --tool-check too, before any lock is written."""
    src = dict(fwsim.PLANT_SRC)
    arg = dict(plant_src=dict(src, params_py=_other_blob(src['params_py'])) if 'plant_src' in kw else None,
               o1c=dict(s4.o1_constants(), D=0.2) if 'o1c' in kw else None,
               imu=dict(s3.IMU, rate_hz=10.0) if 'imu' in kw else None,
               script={'dirty': dict(commit=_head(), dirty=True), 'unknown': dict(commit='0' * 40, dirty=False)}
               .get(kw.get('script')))
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,), **arg)
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller', 'PID',
                                      '--s03', str(d), '--pop', '4', '--gen-base', '1', '--gen-max', '1',
                                      '--dll', _DLL, '--out', str(tmp_path / 'run')])
    with pytest.raises(SystemExit, match=msg):
        s4.main()
    assert not (tmp_path / 'run' / 'lock.yaml').exists()


@needs_dll
@pytest.mark.parametrize('path', s4.SIM_SOURCES)
def test_each_sim_source_is_compared(tmp_path, monkeypatch, capsys, path):
    """S4C-R5-1: a blob at the S03 commit that differs for this one module (and no other) is
    reported as exactly that module; a main start stops, a --tool-check warns and returns it."""
    meta = (_s03_dir(tmp_path, trial=True, mu=(0.5,)) / 'meta.yaml').read_text(encoding='utf-8')
    fp = s4.fingerprint(_DLL, str(tmp_path / 's03'))
    real = s4.blob_at
    monkeypatch.setattr(s4, 'blob_at', lambda c, p: _other_blob(real(c, p)) if p == path else real(c, p))
    assert [n for n, _, _ in s4.source_diff(_head())] == [path]
    with pytest.raises(SystemExit, match=f'not the sources of the S03 script commit: {path} '):
        s4.check_s03_source(meta, fp, _DLL)
    assert [g['path'] for g in s4.check_s03_source(meta, fp, _DLL, tool_check=True)] == [path]
    assert f'WARNING (--tool-check): not the sources of the S03 script commit: {path} ' in capsys.readouterr().out


S03_COMMIT_OLD_FWSIM = '49bbb50'   # feat/s03-doe, pushed: fwsim/fwsim.py there is older than on feat/s04-ga


@needs_dll
def test_sources_compared_by_class_with_the_s03_commit(tmp_path, monkeypatch, capsys):
    """S4C-2 (Codex round 4): every module of SIM_SOURCES (all S04 imports for simulation and
    scoring except s04_ga.py) is compared, through git, with its blob at the S03 script commit. At
    HEAD nothing differs; at the fixed older commit S03_COMMIT_OLD_FWSIM fwsim/fwsim.py differs
    (real git history, no mock): a main trial stops before any lock, a --tool-check only warns and
    returns the difference. A source that differs only through a mocked blob is caught too, and an
    S03 meta.yaml without the plant line is refused."""
    assert set(s4.SIM_SOURCES) == set(s4.SOURCES) - {'fwsim/s04_ga.py'}
    assert s4.source_diff(_head()) == []
    old = s4.source_diff(S03_COMMIT_OLD_FWSIM)
    assert 'fwsim/fwsim.py' in [n for n, _, _ in old] and all(now != then for _, now, then in old)
    d = _s03_dir(tmp_path, _runs(), script=dict(commit=S03_COMMIT_OLD_FWSIM, dirty=False))
    pre = tmp_path / 'pre.json'
    pre.write_text(json.dumps({k: 'checked in x' for k in s4.PRECONDITIONS}))
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--controller', 'PID', '--s03', str(d),
                                      '--s03-unsealed', '--preconditions', str(pre), '--dll', _DLL,
                                      '--out', str(tmp_path / 'run')])
    with pytest.raises(SystemExit, match='not the sources of the S03 script commit: .*fwsim/fwsim.py'):
        s4.main()
    assert not (tmp_path / 'run' / 'lock.yaml').exists()
    meta = (d / 'meta.yaml').read_text(encoding='utf-8')
    fp = s4.fingerprint(_DLL, str(d))
    got = s4.check_s03_source(meta, fp, _DLL, tool_check=True)
    assert 'fwsim/fwsim.py' in [g['path'] for g in got] and 'WARNING (--tool-check)' in capsys.readouterr().out
    head_meta = (_s03_dir(tmp_path / 'h', trial=True, mu=(0.5,)) / 'meta.yaml').read_text(encoding='utf-8')
    assert s4.check_s03_source(head_meta, fp, _DLL) == []
    real = s4.blob_at
    monkeypatch.setattr(s4, 'blob_at', lambda c, p: _other_blob(real(c, p)) if p == 'rpi5/imu_model.py' else real(c, p))
    with pytest.raises(SystemExit, match='rpi5/imu_model.py'):
        s4.check_s03_source(head_meta, fp, _DLL)
    monkeypatch.setattr(s4, 'blob_at', real)
    no_plant = '\n'.join(l for l in head_meta.splitlines() if not l.startswith('plant:'))
    with pytest.raises(SystemExit, match="no 'plant' line"):
        s4.check_s03_source(no_plant, fp, _DLL)


# ── GA loop: generation rule, re-evaluation, run_error (sections 3, 4.3) ─────────────

class FakeEv:
    """Deterministic candidate scores from the gains, no fwsim."""
    ctrl = 'PID'

    def __init__(self):
        self.calls = []

    def candidate(self, g, ga_seed, gen, cand):
        v = list(g.values())
        self.calls.append((gen, cand))
        f1 = 0.01 + 0.05 * abs(v[0] - 1.5)
        f2 = 1.0 + abs(v[1] - 2.0) + 0.1 * v[0]
        return dict(f1=f1, f2=f2, g1=0.0, g2=(f1 - s4.D) / s4.D, status='ok', n_runs=1, n_false=0, n_inconclusive=0,
                    n_track_err=0, n_run_error=0)


DOM2 = [dict(name='a', L=1.0, H=2.0, scale='lin'), dict(name='b', L=1.0, H=3.0, scale='lin')]


@pytest.mark.parametrize('hv_seq, extended, n_gen', [
    ([1.0, 1.0, 1.0], False, 3),           # HV_3 vs HV_1 (gen_compare(3) = 1): no gain -> stop at gen_base
    ([1.0, 1.0, 1.5], True, 5),            # (1.5 - 1.0) / 1.5 > 1 % -> go on to gen_max
    ([1.0, 1.0, 1.005], False, 3),         # 0.5 % -> stop
    ([0.0, 0.0, 0.0], True, 5),            # HV_base = 0 -> go on
])
def test_generation_rule_branches(monkeypatch, hv_seq, extended, n_gen):
    """S4C-16: each branch of the S04 4.3 rule forced through a fake HV sequence."""
    seq = iter(hv_seq + [9.0] * 10)
    monkeypatch.setattr(s4, 'hypervolume', lambda feas, ref: (next(seq), 0))
    hist, hv_rows = [], []
    _, g, ext = s4.run_seed(FakeEv(), DOM2, 1, 6, None, (s4.D, 50.0), hist, hv_rows, ext=(3, 5), log=lambda *_: None)
    assert (ext, g, len(hv_rows)) == (extended, n_gen, n_gen)


def test_gen_compare_is_gen_base_minus_5():
    assert s4.gen_compare(20) == 15 and s4.gen_compare(3) == 1 and s4.gen_compare(1) == 1


def test_extended_trial_equals_a_straight_run(monkeypatch):
    """S4C-16: the trial that goes on to gen_max is the same NSGA-II run as one of gen_max generations."""
    n = iter(range(1, 100))
    monkeypatch.setattr(s4, 'hypervolume', lambda feas, ref: (float(next(n)), 0))     # rising HV: extends
    ev1, ev2 = FakeEv(), FakeEv()
    h1, h2 = [], []
    f1, g1, e1 = s4.run_seed(ev1, DOM2, 3, 6, None, (s4.D, 1e9), h1, [], ext=(2, 5), log=lambda *_: None)
    f2, g2, _ = s4.run_seed(ev2, DOM2, 3, 6, 5, (s4.D, 1e9), h2, [], log=lambda *_: None)
    assert e1 and g1 == g2 == 5
    assert [r['x_a'] for r in h1] == [r['x_a'] for r in h2] and ev1.calls == ev2.calls
    assert [p['xga'] for p in f1] == [p['xga'] for p in f2]
    # every final point carries its own evaluation id, and that evaluation holds the point's F, G
    hist = {(r['gen'], r['cand_id']): r for r in h1}
    for p in f1:
        assert p['gen'] != '' and 'does not match' not in p['id_note'] and 'no eval_id' not in p['id_note']
        h = hist[(p['gen'], p['cand_id'])]
        assert (h['f1'], h['f2']) == (p['f1'], p['f2'])


def test_points_of_uses_the_eval_id_on_the_individual():
    """S4C-6: a vector evaluated twice with the SAME F, G (for example two run_error penalties) maps
    to the evaluation id the Individual carries, not to the first evaluation of the vector."""
    from pymoo.core.individual import Individual
    X, F, G = np.array([1.2, 2.0]), np.array([1e6, 1e6]), np.array([3.0, 1e6])
    ind = Individual(X=X, F=F, G=G)
    ind.set('eval_id', (3, 2))
    seen = {X.tobytes(): [(1, 4, (1e6, 1e6, 3.0, 1e6)), (3, 2, (1e6, 1e6, 3.0, 1e6))]}
    p = s4.points_of([ind], DOM2, 1, seen)[0]
    assert (p['gen'], p['cand_id']) == (3, 2) and 'more than once' in p['id_note']
    bad = Individual(X=X, F=np.array([0.5, 0.5]), G=G)
    bad.set('eval_id', (3, 2))
    assert 'does not match' in s4.points_of([bad], DOM2, 1, seen)[0]['id_note']
    assert 'no eval_id' in s4.points_of([Individual(X=X, F=F, G=G)], DOM2, 1, seen)[0]['id_note']


def test_pareto_rows_feasible_and_infeasible_and_empty(tmp_path):
    """S4C-R-2: pareto.csv holds the non-dominated feasible points and, separately, the non-dominated
    infeasible ones; an empty set still gives a file with the header, and the merged file exists."""
    pts = [P(0.02, 5.0, x=(1.1,)), P(0.03, 6.0, x=(1.2,)), P(0.01, 1.0, x=(1.3,), feasible=False),
           P(0.02, 2.0, x=(1.4,), feasible=False)]
    rows = s4.pareto_rows('PID', {1: pts})
    assert [(r['feasible'], r['f1']) for r in rows] == [(True, 0.02), (False, 0.01)]
    flags = [(r['feasible'], r['nondominated_feasible'], r['nondominated_in_group']) for r in rows]
    assert flags == [(True, True, True), (False, False, True)]                      # S4C-R3-1: both flags
    full = s4.population_rows('PID', {1: pts})                                      # dominated points too
    assert [(r['f1'], r['nondominated_feasible'], r['nondominated_in_group']) for r in full] == [
        (0.02, True, True), (0.03, False, False), (0.01, False, True), (0.02, False, False)]
    s4.write_rows(str(tmp_path / 'pareto_PID.csv'), s4.pareto_rows('PID', {1: []}), s4.POP_COLS)
    s4.merge_controllers(str(tmp_path))
    with open(tmp_path / 'pareto.csv', newline='') as f:
        r = list(csv.reader(f))
    assert r == [s4.POP_COLS]


def test_root_meta_keeps_code_notes(tmp_path):
    """S4C-R-1: notes of CODE in code_notes.json survive every rewrite of meta.yaml."""
    yaml = pytest.importorskip('yaml')
    (tmp_path / 'code_notes.json').write_text(json.dumps({'PID seed 3': 'rate 12 %: lane reversal, no tool error'}))
    for _ in range(2):
        s4.write_root_meta(str(tmp_path), _DLL, dict(controllers={}, common=dict(tool_check=True)))
    m = yaml.safe_load(open(tmp_path / 'meta.yaml', encoding='utf-8'))
    assert m['code_notes'] == {'PID seed 3': 'rate 12 %: lane reversal, no tool error'}
    assert set(m['pymoo_files']) == set(s4.PYMOO_FILES)                  # S4C-18


def test_default_gains_must_give_a_valid_reference(monkeypatch):
    """S4C-7: no HV reference from a run_error penalty or a non-positive f2."""
    rows = iter([(dict(status='run_error', completed='', error='x'), None)])
    monkeypatch.setattr(s3, 'safe_run', lambda *a, **k: next(rows))
    monkeypatch.setattr(s3, 'run_in_new_process', lambda dll, jobs, work: [dict(row=dict(status='ok'), digest=None)])
    monkeypatch.setattr(s3, 'defaults', lambda fw, c: {})
    assert s4.default_f2(None, 'x.dll', 'PID', [0.2], 0.5, 'w')[3] is False
    for tv in (0.0, -3.0, float('nan'), float('inf')):                 # f2 not finite or not > 0
        rows = iter([(dict(status='ok', completed='true', e_perp_max=0.02, tv_duty=tv), None)])
        assert s4.default_f2(None, 'x.dll', 'PID', [0.2], 0.5, 'w')[3] is False, tv
    rows = iter([(dict(status='ok', completed='true', e_perp_max=0.02, tv_duty=3.0), None)])
    assert s4.default_f2(None, 'x.dll', 'PID', [0.2], 0.5, 'w')[3] is True


def test_check_eval_keys_refuses_duplicates():
    """S4C-13."""
    e = dict(controller='PID', ga_seed=1, gen=1, cand_id=0, v=0.2, mu=0.5, attempt=1)
    s4.check_eval_keys([e, dict(e, attempt=2)])
    with pytest.raises(SystemExit, match='not unique'):
        s4.check_eval_keys([e, dict(e)])


def test_run_error_retry_ok_keeps_the_first_attempt_penalty(monkeypatch):
    """S04 3 (B8b-20): a run_error is run once more in a new process for diagnosis; even when that
    attempt succeeds, the candidate keeps the penalty of the first attempt."""
    ok = dict(status='ok', completed='true', e_perp_max=0.02, tv_duty=4.0)
    calls = iter([(ok, None), (dict(status='run_error', completed='', error='x'), None)])
    monkeypatch.setattr(s3, 'safe_run', lambda *a, **k: next(calls))
    monkeypatch.setattr(s3, 'run_in_new_process', lambda dll, jobs, work: [dict(row=ok, digest=None)])
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.5], 'w')
    s = ev.candidate({}, 1, 1, 0)
    assert (s['status'], s['f1'], s['g1']) == ('run_error', s4.PENALTY, 2.0)
    assert [(e['v'], e['attempt'], e['status']) for e in ev.evals] == [(0.2, 1, 'ok'), (0.5, 1, 'run_error'),
                                                                       (0.5, 2, 'ok')]


def test_three_mu_findings_flag_track_err_even_when_completed():
    """S4C-9."""
    rows = [dict(controller='PID', v=0.2, mu=0.4, status='track_err', completed='true', e_perp_max=0.03),
            dict(controller='PID', v=0.5, mu=0.4, status='ok', completed='true', e_perp_max=0.03),
            dict(controller='PID', v=0.8, mu=0.6, status='ok', completed='true', e_perp_max=0.2)]
    f = s4.three_mu_findings(rows, 'PID')
    assert len(f) == 2 and 'track_err' in f[0] and 'v=0.8' in f[1]


def test_plant_dump_keeps_nested_values():
    """S4C-8: every attribute, nested objects included."""
    p = fwsim.plant_params(__import__('argparse').Namespace(ground='traction', mu=0.5))
    d = s4.plant_dump(p)
    assert set(d) == set(vars(p))
    assert isinstance(d['slip'], dict) and d['slip']                      # a nested object, not dropped
    json.dumps(d)


# ── end to end, tiny budget (needs the DLL) ──────────────────────────────────────────

@needs_dll
def test_tool_check_three_phases(tmp_path, monkeypatch, capsys):
    """trial -> lock.yaml, main (2 seeds) -> selected.csv with a send_to_HIL row whose gains read back
    from the file reproduce every GA run bit for bit in a new process (post)."""
    yaml = pytest.importorskip('yaml')
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,), levels=_levels_near_defaults())
    out = tmp_path / 'run'
    for ctrl in s3.CONTROLLERS:
        common = ['--tool-check', '--controller', ctrl, '--s03', str(d), '--pop', '4', '--gen-base', '1',
                  '--gen-max', '2', '--dll', _DLL, '--out', str(out)]
        monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial'] + common)
        assert s4.main() in (0, 2)
        assert yaml.safe_load(open(out / 'meta.yaml', encoding='utf-8'))['tool_check'] is True    # S4C-12
    lock = json.load(open(out / 'lock.yaml'))
    assert lock['common']['s03_source_diff'] == []                         # S4C-2: S03 record of this checkout
    for ctrl in s3.CONTROLLERS:
        L = lock['controllers'][ctrl]
        assert L['eval_mu'] == [0.5] and L['n_gen'] in (1, 2) and L['hv_ref'][0] == s4.D and L['tool_check']
        assert L['gen_rule']['compared_with'] == 1
    assert lock['common']['tool_check'] is True
    for ctrl in s3.CONTROLLERS:
        monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'main', '--tool-check', '--controller', ctrl,
                                          '--s03', str(d), '--seeds', '1', '2', '--dll', _DLL, '--out', str(out)])
        s4.main()
    evals = list(csv.DictReader(open(out / 'evals.csv')))
    keys = [(e['controller'], e['ga_seed'], e['gen'], e['cand_id'], e['v'], e['mu'], e['attempt']) for e in evals]
    assert len(keys) == len(set(keys)) and {e['controller'] for e in evals} == set(s3.CONTROLLERS)   # S04 8
    sel = list(csv.DictReader(open(out / 'selected.csv')))
    hist = list(csv.DictReader(open(out / 'history.csv')))
    for send in (r for r in sel if r['kind'] == 'send_to_HIL'):
        h = next(r for r in hist if (r['controller'], r['ga_seed'], r['gen'], r['cand_id']) ==
                 (send['controller'], send['ga_seed'], send['gen'], send['cand_id']))
        assert float(h['f1']) == float(send['f1'])                       # selected row is a GA candidate
    par = list(csv.DictReader(open(out / 'pareto.csv')))
    pop = list(csv.DictReader(open(out / 'final_population.csv')))
    assert par and len(pop) >= len(par)                                  # S4C-17
    assert all(r['nondominated_in_group'] == 'true' for r in par)        # S4C-R3-1: both groups, flag per group
    assert all(r['nondominated_feasible'] == r['feasible'] for r in par)
    assert sum(r['nondominated_in_group'] == 'true' for r in pop) == len(par)
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'post', '--dll', _DLL, '--out', str(out)])
    assert s4.post(s4_args(out)) == 0
    pc = out / 'post_checks'
    det = list(csv.DictReader(open(pc / 'determinism.csv')))
    assert det and all(r['equal'] == 'true' and r['tool_check'] == 'true' for r in det)
    sens = list(csv.DictReader(open(pc / 'sensitivity.csv')))
    assert {r['r_factor'] for r in sens if r['check'] == 'r_factor'} == {repr(k) for k in s4.R_FACTORS}
    assert {r['b0_factor'] for r in sens if r['check'] == 'b0_factor'} == {repr(k) for k in s4.B0_FACTORS}
    changed = [json.loads(r['plant_changed']) for r in sens if r['check'] == 'r_factor' and r['r_factor'] != '1.0']
    assert changed and all(set(c) == {'motor_R'} for c in changed)       # only R changes (S04 6)
    full = json.load(open(pc / 'plant_params.json'))['entries']          # S4C-8
    assert any('r_factor=1.2' in k for k in full) and all(set(v) == {'before', 'after'} for v in full.values())
    notes = json.load(open(pc / 'notes.json'))
    assert notes['tool_check'] is True                                   # S4C-10
    assert notes['notes']['PID three_mu']['done'] is False and 'NOT done' in notes['notes']['PID three_mu']['note']
    for p in [out / 'meta.yaml', out / 'meta_PID.yaml', out / 'meta_ADRC.yaml', out / 'trial' / 'PID' / 'meta.yaml',
              out / 'trial' / 'ADRC' / 'meta.yaml', out / 'lock.yaml']:
        yaml.safe_load(open(p, encoding='utf-8'))                        # S4C-11: valid YAML on Windows paths


def s4_args(out):
    import argparse
    return argparse.Namespace(out=str(out), dll=_DLL)
