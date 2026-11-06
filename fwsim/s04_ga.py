"""
s04_ga.py: spec S04 v5 (CLAUDE_THESIS 06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md), NSGA-II
(pymoo) on fwsim choosing the 7 gains of one controller, objective B of L54 on the figure 8 of
O1 v7 (o1_feasibility.evaluate_v7 on the true plant pose, the same runs as S03: s03_doe.run_one).

Section numbers below are those of S04 v5.
  2   variables: the 7 S03 factors of the controller (s03_doe.FACTORS); fixed parameters as
      S03 2.1 (s03_doe.sim_kwargs: kd = 0, limits, b0, out_limit from the firmware). Domain =
      the S03 levels [L, H] (levels.csv), not widened; a factor coded log in S03 is searched on
      log(gain). A chosen gain within 5 % of the coded width from a bound is flagged.
  3   one candidate = every gate speed (v) x mu level of the evaluation: middle mu only when
      the mu rule holds on every S03 design point (centre, outer, inner box) and gate speed,
      else the three mu levels, worst of. f1 = max e_perp_max (m); f2 = max over mu of the
      mean over v of tv_duty (1/s). g1 = number of runs not completed, inconclusive or with a
      track error; g2 = (f1 - D) / D; pymoo G <= 0. A run_error is run once more in a new
      process for diagnosis (attempt 2, column 'cause' left for CODE); the candidate then
      gets g1 = number of runs, f1 = f2 = 1e6, status run_error; the GA goes on.
  4.2 GA seeds 1-5, plant seed 1 (s03_doe.SEED). Hypervolume per generation on the feasible
      non-dominated set of that generation's population, reference (D, 10 x f2 of the default
      gains at the middle mu), points on or beyond the reference counted, HV 0 when none.
  4.3 --phase trial: one seed, full budget (pop 24, 20 generations); when
      (HV20 - HV15) / HV20 > 1 % or HV20 = 0 the same run goes on to 40 generations, and the
      main runs use 40; lock.yaml holds the domain, the mu decision, the budget, the seeds,
      the selection rule, the HV reference and the preconditions of 4.3 with where checked.
      --phase main: the locked seeds and budget, every run of every seed.
  5   selection, on a feasible non-dominated set: f1* = min f1; keep f1 <= f1* + 0.01 m;
      smallest f2, then smaller f1, then the gain vector in lexicographic order (then the
      smaller seed). per_seed rows on each seed's final feasible non-dominated set;
      send_to_HIL on the union of those sets over the feasible seeds, filtered again. A seed
      with no feasible point in its final population is 'infeasible': not in the median, IQR
      or union; more than half infeasible: no send_to_HIL, report to Liam.
  6   --phase post: in a NEW process with the gains read back from selected.csv, every
      evaluation run compared bit for bit with the GA run (log SHA-256); b0 x 0.7..1.3 (LADRC)
      and motor R x 0.8..1.2 (both, R only, plant parameters dumped before and after) with
      the 7 chosen gains, same speeds and mu as the GA, criterion of B0_SENS/2026-11-05_run01
      plus the tv_duty change in %; the three mu levels when the GA used the middle one only;
      the report-only measures.
  7   K1, K2 are never run; nothing here runs on the car.
  8   outputs in --out (columns controller, ga_seed): history.csv, evals.csv (key controller,
      ga_seed, gen, cand_id, v, mu, attempt), hv.csv, pareto.csv (feasible non-dominated),
      final_population.csv, selected.csv, meta.yaml (whole folder, written before the GA and after
      every phase), meta_<controller>.yaml, phases.json, lock.yaml (one mode and one source
      fingerprint for the folder), trial/ (the trial phase), post_checks/.

L56: the mu rule reads sealed/runs.csv of the S03 run, so --phase trial / main need
--s03-unsealed (campaign A finished, the S03 results may be looked at). --tool-check (tool test,
never main numbers): takes an S03 --trial run (one mu, the mu rule is not applied, sealed/ is
never read), any budget and seeds, no preconditions.

  python fwsim/s04_ga.py --phase trial --controller PID --s03 <S03 run> --s03-unsealed \\
      --preconditions <json> --out 05_DATA/results/S04/<date>_runNN
  python fwsim/s04_ga.py --phase main --controller PID --s03 <S03 run> --s03-unsealed --out <same dir>
  python fwsim/s04_ga.py --phase post --out <same dir>
  python fwsim/s04_ga.py --tool-check --phase trial --controller PID --s03 <S03 trial run> \\
      --pop 6 --gen-base 4 --gen-max 6 --out <dir>
Interpretations of this script, not written in S04 v5, are in meta.yaml 'interpretations'.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim                 # noqa: E402
import o1_feasibility as o1  # noqa: E402
import s03_doe as s3         # noqa: E402

D = o1.D                      # m, O1 v7 (temporary)
F1_TOL = 0.01                 # m, section 3 / 5 (convention, same as S03 5.2)
MU_Y2_TOL = 0.01              # m, mu rule (section 3)
MU_Y3_REL = 0.20              # of the S03 Y3 reference (section 3)
HV_REF_MULT = 10.0            # section 4.2
HV_EXT_REL = 0.01             # section 4.3
GEN_BASE, GEN_MAX, POP = 20, 40, 24
SEEDS = (1, 2, 3, 4, 5)
PENALTY = 1e6                 # section 3: f1 = f2 = 1e6 for a run_error candidate
SEED_ERR_LIMIT = 0.10         # section 3
EDGE_FRAC = 0.05              # section 2
B0_FACTORS = (0.7, 0.85, 1.0, 1.15, 1.3)
R_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
REL_LIMIT = 0.5               # B0_SENS criterion: e_perp_max change > 50 %
PRECONDITIONS = ('s03_run_and_locked', 'final_test_set_locked', 'ctrl_imu_checked_on_board', 'x3_compared')
CMP_EPS = 1e-12               # S4C-14: rounding tolerance of the inclusive mu-rule bounds (m; relative for Y3)
# pymoo 0.6.2 constraint handling, read in the installed library (S04 section 3; Liam asked CODE to
# record file:line before the trial). Re-read when the version differs.
PYMOO_CHECKED = '0.6.2'
PYMOO_CV = ("core/individual.py:29 cv_ieq default func=np.sum, eps 0, no scale, no pow; "
            "core/individual.py:600-633 constr_to_cv: c = max(0, g_i - eps), CV = func(c) = sum (line 633) over "
            "the inequality constraints; core/survival.py:59-80 Survival.do: feasible (CV = 0) individuals ranked "
            "by RankAndCrowding first, the rest filled by increasing CV; algorithms/moo/nsga2.py:23-77 binary "
            "tournament: if either is infeasible the smaller CV wins (random on a tie, 39-48), else dominance "
            "then crowding (50-75)")
EVAL_COLS = ['controller', 'ga_seed', 'gen', 'cand_id', 'v', 'mu', 'attempt', 'status', 'completed', 'e_perp_max',
             'tv_duty', 'sat_frac', 't_end', 'e_theta_lane_max_deg', 'e_theta_cmd_max_deg', 'ds_absmax', 'e_trk_max',
             'log_sha256', 'cause', 'error']


# ── small helpers ────────────────────────────────────────────────────────────────────

def meta_value(text, key):
    """Value of a 'key: <json>' line of a meta.yaml written by these scripts (comment dropped)."""
    for line in text.splitlines():
        if line.startswith(key + ':'):
            v = line[len(key) + 1:].split('  #')[0].strip()
            try:
                return json.loads(v)
            except ValueError:
                return v.strip('"')
    raise KeyError(key)


def log_sha(log):
    """One SHA-256 of a log: of the s03_doe.log_digest of every array (bit for bit)."""
    if log is None:
        return ''
    return hashlib.sha256(json.dumps(s3.log_digest(log), sort_keys=True).encode()).hexdigest()


def write_rows(path, rows, cols=None):
    s3.write_csv(path, rows, cols)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def nondominated(points):
    """Points (dicts with f1, f2) not dominated by another in the list (equal points are kept)."""
    out = []
    for p in points:
        if not any((q['f1'] <= p['f1'] and q['f2'] <= p['f2']) and (q['f1'] < p['f1'] or q['f2'] < p['f2'])
                   for q in points):
            out.append(p)
    return out


def select(points):
    """Section 5 on a list of feasible points (dicts f1, f2, x (gain list), ga_seed): the chosen one,
    after a non-dominated filter, or None."""
    nd = nondominated(points)
    if not nd:
        return None
    f1s = min(p['f1'] for p in nd)
    keep = [p for p in nd if p['f1'] <= f1s + F1_TOL]
    return min(keep, key=lambda p: (p['f2'], p['f1'], tuple(p['x']), p['ga_seed']))


def hypervolume(points, ref):
    """HV of the feasible non-dominated points strictly inside the reference (section 4.2); also the
    number of feasible non-dominated points on or beyond it."""
    nd = nondominated(points)
    inside = sorted((p['f1'], p['f2']) for p in nd if p['f1'] < ref[0] and p['f2'] < ref[1])
    out = len(nd) - len(inside)
    hv, f2_prev = 0.0, ref[1]
    for f1, f2 in inside:                       # sweep in f1, staircase of the 2-D front
        if f2 < f2_prev:
            hv += (ref[0] - f1) * (f2_prev - f2)
            f2_prev = f2
    return hv, out


# ── domain and mu rule ───────────────────────────────────────────────────────────────

def read_domain(s03_dir, ctrl):
    """[(name, L, H, scale)] of the controller from the S03 levels.csv (S03 order of FACTORS)."""
    rows = [r for r in read_csv(os.path.join(s03_dir, 'levels.csv')) if r['controller'] == ctrl]
    by = {r['factor']: r for r in rows}
    out = []
    for n in s3.FACTORS[ctrl]:
        if n not in by:
            sys.exit(f'{s03_dir}/levels.csv has no level for {ctrl} {n}')
        L, H = float(by[n]['L']), float(by[n]['H'])
        if not L < H:
            sys.exit(f'{ctrl} {n}: L = {L} is not below H = {H} (S03 3.1 (ii))')
        out.append(dict(name=n, L=L, H=H, scale=by[n]['scale']))
    return out


def ga_bounds(dom):
    lo = [math.log(d['L']) if d['scale'] == 'log' else d['L'] for d in dom]
    hi = [math.log(d['H']) if d['scale'] == 'log' else d['H'] for d in dom]
    return np.array(lo), np.array(hi)


def decode(dom, x):
    """GA vector -> gains {name: value} (exp on log factors; a coordinate at a bound gives exactly L
    or H, exp(log H) can be one ULP off; clipped to [L, H] against rounding)."""
    lo, hi = ga_bounds(dom)
    g = {}
    for d, xi, a_, b_ in zip(dom, x, lo, hi):
        xi = float(xi)
        if xi <= a_:
            v = d['L']
        elif xi >= b_:
            v = d['H']
        else:
            v = math.exp(xi) if d['scale'] == 'log' else xi
        g[d['name']] = min(max(v, d['L']), d['H'])
    return g


def edge_flags(dom, x):
    """Factors within EDGE_FRAC of the coded width from a bound (section 2: reported as a limit)."""
    lo, hi = ga_bounds(dom)
    return [d['name'] for d, xi, a, b in zip(dom, x, lo, hi) if min(xi - a, b - xi) <= EDGE_FRAC * (b - a)]


def mu_rule(s03_dir, ctrl, gate, mu_levels):
    """Section 3: ('mid', report) when every S03 design point (all boxes that ran) at every gate
    speed has the same Y1 at the three mu, |dY2| <= 0.01 m and |dY3| <= 20 % of the S03 Y3
    reference versus the middle mu, no track error, no inconclusive; else ('all', report)."""
    meta = open(os.path.join(s03_dir, 'meta.yaml'), encoding='utf-8').read()
    y3_ref = meta_value(meta, 'y3_reference')
    runs = [r for r in read_csv(os.path.join(s03_dir, 'sealed', 'runs.csv')) if r['controller'] == ctrl]
    design = [(r['point_id'], r['box']) for r in read_csv(os.path.join(s03_dir, 'design.csv'))
              if r['controller'] == ctrl]
    if not design or len(set(design)) != len(design):
        sys.exit(f'{s03_dir}/design.csv: no or duplicate design points for {ctrl} (S04 3, S4C-5)')
    lo, mid, hi = mu_levels
    by, dup = {}, []
    for r in runs:
        rs = by.setdefault((r['point_id'], r['box'], float(r['v'])), {})
        if float(r['mu']) in rs:
            dup.append((r['point_id'], r['box'], r['v'], r['mu']))
        rs[float(r['mu'])] = r
    if dup:
        sys.exit(f'{s03_dir}/sealed/runs.csv: duplicate runs for {ctrl} (S4C-5), first {dup[0]}')
    bad = []
    n = 0
    for (pid, box) in design:                   # every design point x every gate speed (S4C-5)
        for v in gate:
            key = next((k for k in by if k[:2] == (pid, box) and math.isclose(k[2], v)), None)
            if key is None:
                sys.exit(f'{s03_dir}/sealed/runs.csv: no run of {ctrl} point {pid} ({box}) at v = {v} (S4C-5)')
    for (pid, box, v), rs in sorted(by.items()):
        if not any(math.isclose(v, g) for g in gate):
            continue
        if (pid, box) not in design:
            sys.exit(f'{s03_dir}/sealed/runs.csv: {ctrl} point {pid} ({box}) not in design.csv (S4C-5)')
        n += 1
        why = []
        if set(rs) != {lo, mid, hi}:
            why.append('missing mu')
        else:
            if any(r['status'] != 'ok' for r in rs.values()):
                why.append('track_err or run_error')
            if any(r['completed'] not in ('true', 'false') for r in rs.values()):
                why.append('y1 inconclusive')
            if len({r['completed'] for r in rs.values()}) > 1:
                why.append('y1 differs over mu')
            y2 = {m: float(r['e_perp_max']) for m, r in rs.items()}
            if max(abs(y2[lo] - y2[mid]), abs(y2[hi] - y2[mid])) > MU_Y2_TOL + CMP_EPS:     # S4C-14
                why.append('dY2 > 0.01 m')
            y3 = {m: float(r['tv_duty']) for m, r in rs.items()}
            ref = y3_ref[f'{ctrl} v={v:g}']
            if max(abs(y3[lo] - y3[mid]), abs(y3[hi] - y3[mid])) > MU_Y3_REL * ref * (1 + CMP_EPS):
                why.append('dY3 > 20 % of the reference')
        if why:
            bad.append(dict(point_id=pid, box=box, v=v, why=why))
    return ('mid' if not bad and n else 'all'), dict(points_checked=n, points_failing=len(bad), first=bad[:5])


# ── one candidate ────────────────────────────────────────────────────────────────────

class Evaluator:
    """Runs one candidate on every (mu, v) of the evaluation; keeps every run (evals.csv)."""

    def __init__(self, fw, dll, ctrl, speeds, mus, work):
        self.fw, self.dll, self.ctrl, self.speeds, self.mus, self.work = fw, dll, ctrl, list(speeds), list(mus), work
        self.path = o1.Path8()
        self.evals = []

    def _rec(self, ga_seed, gen, cand, v, mu, attempt, row, log, **extra):
        r = dict(controller=self.ctrl, ga_seed=ga_seed, gen=gen, cand_id=cand, v=v, mu=mu, attempt=attempt,
                 log_sha256=log_sha(log), cause='', **{k: row.get(k, '') for k in EVAL_COLS if k in row})
        r.update(extra)
        self.evals.append(r)
        return r

    def candidate(self, g, ga_seed, gen, cand):
        rows, error = [], False
        for mu in self.mus:
            for v in self.speeds:
                row, log = s3.safe_run(self.fw, self.path, self.ctrl, g, v, mu)
                self._rec(ga_seed, gen, cand, v, mu, 1, row, log)
                rows.append((mu, v, row))
                if row['status'] == 'run_error':
                    error = True
                    try:                     # section 3: once more in a new process, for diagnosis only
                        again = s3.run_in_new_process(self.dll, [(self.ctrl, g, v, mu)], self.work)[0]
                        self._rec(ga_seed, gen, cand, v, mu, 2, again['row'], None,
                                  log_sha256=(hashlib.sha256(json.dumps(again['digest'], sort_keys=True).encode())
                                              .hexdigest() if again.get('digest') else ''))
                    except Exception as exc:  # noqa: BLE001
                        self._rec(ga_seed, gen, cand, v, mu, 2, dict(status='diagnosis_failed'), None,
                                  error=f'{type(exc).__name__}: {exc}')
        return score(rows, error)


def score(rows, error):
    """f1, f2, g1, g2 and counts of one candidate from its [(mu, v, row)] (section 3)."""
    n = len(rows)
    counts = dict(n_runs=n, n_false=sum(r['completed'] == 'false' for _, _, r in rows),
                  n_inconclusive=sum(r['completed'] == 'inconclusive' for _, _, r in rows),
                  n_track_err=sum(r['status'] == 'track_err' for _, _, r in rows),
                  n_run_error=sum(r['status'] == 'run_error' for _, _, r in rows))
    if error:
        return dict(f1=PENALTY, f2=PENALTY, g1=float(n), g2=(PENALTY - D) / D, status='run_error', **counts)
    f1 = max(float(r['e_perp_max']) for _, _, r in rows)
    mus = sorted({mu for mu, _, _ in rows})
    f2 = max(float(np.mean([float(r['tv_duty']) for m, _, r in rows if m == mu])) for mu in mus)
    g1 = sum(r['completed'] != 'true' or r['status'] == 'track_err' for _, _, r in rows)
    return dict(f1=f1, f2=f2, g1=float(g1), g2=(f1 - D) / D, status='ok', **counts)


# ── one GA seed ──────────────────────────────────────────────────────────────────────

def run_seed(ev, dom, ga_seed, pop, n_gen, ref, history, hv_rows, ext=None, log=print):
    """NSGA-II for one GA seed. ext = (gen_base, gen_max) for the trial phase (section 4.3), else
    n_gen generations. Returns (final points [dict], generations run, extended)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem

    lo, hi = ga_bounds(dom)
    state = dict(gen=0)
    seen = {}

    class Gains(Problem):
        def __init__(self):
            super().__init__(n_var=len(dom), n_obj=2, n_ieq_constr=2, xl=lo, xu=hi)

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = [], []
            state['ids'] = [(state['gen'], k) for k in range(len(X))]
            for k, x in enumerate(X):
                g = decode(dom, x)
                s = ev.candidate(g, ga_seed, state['gen'], k)
                F.append([s['f1'], s['f2']])
                G.append([s['g1'], s['g2']])
                # S4C-6: every evaluation of this vector (pymoo may evaluate a vector again)
                seen.setdefault(np.asarray(x, dtype=float).tobytes(), []).append(
                    (state['gen'], k, (float(s['f1']), float(s['f2']), float(s['g1']), float(s['g2']))))
                history.append(dict(controller=ev.ctrl, ga_seed=ga_seed, gen=state['gen'], cand_id=k,
                                    **{d['name']: g[d['name']] for d in dom},
                                    **{f'x_{d["name"]}': float(xi) for d, xi in zip(dom, x)}, **s))
            out['F'], out['G'] = np.array(F), np.array(G)

    problem = Gains()
    alg = NSGA2(pop_size=pop)
    total = ext[1] if ext else n_gen
    alg.setup(problem, termination=('n_gen', total), seed=ga_seed, verbose=False)
    hv_by_gen, extended = {}, False
    while alg.has_next():
        state['gen'] += 1
        infills = alg.ask()
        todo = [ind for ind in infills if not (ind.evaluated and 'F' in ind.evaluated)]
        state['ids'] = []
        alg.evaluator.eval(problem, infills)
        if len(state['ids']) != len(todo):
            raise RuntimeError(f'pymoo evaluated {len(state["ids"])} of {len(todo)} new individuals')
        for ind, eid in zip(todo, state['ids']):  # S4C-6: the evaluation id rides on the Individual
            ind.set('eval_id', eid)
        alg.tell(infills=infills)
        pts = points_of(alg.pop, dom, ga_seed, seen)
        feas = [p for p in pts if p['feasible']]
        hv, n_out = hypervolume(feas, ref)
        hv_by_gen[state['gen']] = hv
        hv_rows.append(dict(controller=ev.ctrl, ga_seed=ga_seed, gen=state['gen'], hv=hv, n_feasible=len(feas),
                            n_nd_feasible=len(nondominated(feas)), n_nd_on_or_beyond_ref=n_out))
        log(f'[{ev.ctrl} seed {ga_seed}] gen {state["gen"]}: feasible {len(feas)}/{len(pts)}, HV {hv:.6g}')
        if ext and state['gen'] == ext[0]:
            h_b, h_a = hv_by_gen[ext[0]], hv_by_gen[gen_compare(ext[0])]
            extended = h_b == 0.0 or (h_b - h_a) / h_b > HV_EXT_REL
            if not extended:
                break
    return points_of(alg.pop, dom, ga_seed, seen), state['gen'], extended


def points_of(pop, dom, ga_seed, seen):
    pts = []
    for ind in pop:
        x = np.asarray(ind.X, dtype=float)
        G = np.asarray(ind.G, dtype=float)
        fg = (float(ind.F[0]), float(ind.F[1]), float(G[0]), float(G[1]))
        hits = seen.get(x.tobytes(), [])
        eid = ind.get('eval_id')                        # S4C-6: set on the Individual right after its evaluation
        if eid is not None:
            gen, cand = eid
            mine = [e for e in hits if tuple(e[:2]) == tuple(eid)]
            note = '' if len(hits) <= 1 else 'vector evaluated more than once; this point is its own evaluation'
            if not mine or mine[0][2] != fg:
                note = 'eval_id does not match the recorded evaluation'
        else:
            gen, cand, note = '', '', 'no eval_id on the individual'
        pts.append(dict(ga_seed=ga_seed, gen=gen, cand_id=cand, f1=fg[0], f2=fg[1], g1=fg[2], g2=fg[3],
                        feasible=bool(np.all(G <= 0)), id_note=note,
                        x=[decode(dom, x)[d['name']] for d in dom], xga=x.tolist()))
    return pts


# ── outputs ──────────────────────────────────────────────────────────────────────────

def gain_cols(dom, p):
    """Section 6: every gain as repr (17 significant digits, bit exact) and its float32 value."""
    out = {}
    for d, v in zip(dom, p['x']):
        out[d['name']] = repr(float(v))
        out[d['name'] + '_f32'] = repr(float(np.float32(v)))
    return out


def summarise(ctrl, dom, finals):
    """selected.csv rows of one controller from {ga_seed: final points} (section 5)."""
    rows, feas_sets, per = [], [], []
    for seed in sorted(finals):
        feas = [p for p in finals[seed] if p['feasible']]
        nd = nondominated(feas)
        ch = select(nd)
        base = dict(kind='per_seed', controller=ctrl, ga_seed=seed, feasible=ch is not None)
        if ch is None:
            rows.append(dict(base, note='no feasible point in the final population'))
            continue
        feas_sets += nd
        per.append(ch)
        rows.append(dict(base, gen=ch['gen'], cand_id=ch['cand_id'], f1=ch['f1'], f2=ch['f2'],
                         edge=' '.join(edge_flags(dom, ch['xga'])), **gain_cols(dom, ch)))
    n_seeds, n_feas = len(finals), len(per)
    stats = {}
    if per:
        for k in ('f1', 'f2'):
            v = np.array([p[k] for p in per])
            stats[k] = dict(median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)))
    if n_feas * 2 < n_seeds or not per:
        rows.append(dict(kind='send_to_HIL', controller=ctrl, feasible=False,
                         note=f'stop: {n_seeds - n_feas} of {n_seeds} seeds infeasible: tell Liam (S04 5)'))
    else:
        ch = select(feas_sets)
        rows.append(dict(kind='send_to_HIL', controller=ctrl, ga_seed=ch['ga_seed'], feasible=True, gen=ch['gen'],
                         cand_id=ch['cand_id'], f1=ch['f1'], f2=ch['f2'], edge=' '.join(edge_flags(dom, ch['xga'])),
                         **gain_cols(dom, ch)))
    return rows, dict(n_seeds=n_seeds, n_feasible_seeds=n_feas, per_seed_selected=stats)


def seed_error_rates(evals, ctrl):
    """Section 3 (v4, B8b-7): per seed, first-attempt runs inconclusive or run_error / first-attempt runs."""
    out = {}
    for r in evals:
        if r['controller'] != ctrl or int(r['attempt']) != 1:
            continue
        c = out.setdefault(int(r['ga_seed']), [0, 0])
        c[1] += 1
        c[0] += r['completed'] == 'inconclusive' or r['status'] == 'run_error'
    return {s: dict(rate=a / n if n else 0.0, n=n, over_10pct=n > 0 and a / n > SEED_ERR_LIMIT)
            for s, (a, n) in sorted(out.items())}


POP_COLS = ['controller', 'ga_seed', 'gen', 'cand_id', 'feasible', 'nondominated_feasible', 'nondominated_in_group',
            'f1', 'f2', 'g1', 'g2', 'id_note', 'gains']


def population_rows(ctrl, finals, nd_only=False):
    """final_population.csv: every point of each seed's final population with non-dominated flags;
    nd_only: pareto.csv (S04 8, 'feasible and infeasible in separate columns'; S4C-17, S4C-R-2): the
    non-dominated points of the feasible points and, separately, of the infeasible points (column
    'feasible'). Written with POP_COLS, so an empty set still gives a file with its header."""
    rows = []
    for seed in sorted(finals):
        pts = finals[seed]
        feas = [p for p in pts if p['feasible']]
        infeas = [p for p in pts if not p['feasible']]
        nd_f = {id(p) for p in nondominated(feas)}
        nd_g = nd_f | {id(p) for p in nondominated(infeas)}
        for p in pts:
            if nd_only and id(p) not in nd_g:
                continue
            rows.append(dict(controller=ctrl, ga_seed=seed, gen=p['gen'], cand_id=p['cand_id'], feasible=p['feasible'],
                             nondominated_feasible=id(p) in nd_f, nondominated_in_group=id(p) in nd_g, f1=p['f1'],
                             f2=p['f2'], g1=p['g1'], g2=p['g2'], id_note=p.get('id_note', ''),
                             gains=json.dumps([repr(float(v)) for v in p['x']])))
    return rows


def pareto_rows(ctrl, finals):
    return population_rows(ctrl, finals, nd_only=True)


def default_f2(fw, dll, ctrl, speeds, mu_mid, work):
    """f2 of the firmware default gains at the middle mu (HV reference, section 4.2). S4C-7: the
    reference is only locked from a valid run (every run 'ok' or 'track_err' scored, no run_error,
    f2 finite and > 0); otherwise the trial stops."""
    ev = Evaluator(fw, dll, ctrl, speeds, [mu_mid], work)
    s = ev.candidate(s3.defaults(fw, ctrl), 0, 0, 0)
    ok = s['status'] == 'ok' and s['n_run_error'] == 0 and math.isfinite(s['f2']) and s['f2'] > 0
    return s['f2'], s, ev.evals, ok


# ── post checks (section 6) ──────────────────────────────────────────────────────────

def run_custom(fw, path, ctrl, g, v, mu, b0_factor=1.0, r_factor=1.0):
    """s03_doe.run_one with b0 (LADRC) and / or motor R scaled; returns (row, log, read-back, plant before/after)."""
    sch = o1.Schedule(v)
    p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=mu))
    before = plant_dump(p)
    if r_factor != 1.0:
        p.update_physical(motor_R=p.motor_R * r_factor)
    after = plant_dump(p)
    kw = s3.sim_kwargs(fw, ctrl, g)
    if ctrl == 'ADRC' and b0_factor != 1.0:
        kw['adrc'] = {i: (a, b, b0 * b0_factor, lim) for i, (a, b, b0, lim) in kw['adrc'].items()}
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, p, seed=s3.SEED, mode=ctrl, imu=dict(s3.IMU), **kw)
    back = [fw.get_adrc(i) for i in range(3)] if ctrl == 'ADRC' else [fw.get_pid(i) for i in range(3)]
    back.append(dict(pos=fwsim.pos_meta(fw, raw=True)))  # exact float32 pos_kp, pos_ki (S4C-8); k_psi has no getter (S3C-21)
    res, _ = o1.evaluate_v7(log, path, sch, lsum=p.lx + p.ly)
    return res, log, back, before, after


def plant_dump(p):
    """Every attribute of the plant parameter object actually used, nested objects included
    (S04 6; S4C-8): arrays as lists, numpy scalars as Python numbers, other values as repr."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if hasattr(v, '__dict__'):
            return {k: conv(x) for k, x in sorted(vars(v).items())}
        return repr(v)
    return {k: conv(v) for k, v in sorted(vars(p).items())}


def judge(rows, key):
    """B0_SENS criterion (written before, B0_SENS/2026-11-05_run01) per (v, mu) against factor 1.0."""
    reasons = []
    for (v, mu) in sorted({(r['v'], r['mu']) for r in rows}):
        nom = next(r for r in rows if r['v'] == v and r['mu'] == mu and r[key] == 1.0)
        for r in rows:
            if (r['v'], r['mu']) != (v, mu) or r[key] == 1.0:
                continue
            tag = f'v={v} mu={mu} {key}={r[key]}'
            if r['completed'] != nom['completed']:
                reasons.append(f'{tag}: completed {nom["completed"]} -> {r["completed"]}')
            if (r['e_perp_max'] <= D) != (nom['e_perp_max'] <= D):
                reasons.append(f'{tag}: e_perp_max <= D changes')
            if abs(r['e_perp_max'] - nom['e_perp_max']) > REL_LIMIT * nom['e_perp_max']:
                reasons.append(f'{tag}: e_perp_max {nom["e_perp_max"]:.4f} -> {r["e_perp_max"]:.4f}')
    return bool(reasons), reasons


def three_mu_findings(rows, ctrl):
    """Runs of the three-mu check not passing the gate. S4C-9: a track_err / run_error run is not a
    passed gate even with completed 'true'."""
    return [f"v={r['v']} mu={r['mu']}: status {r.get('status')}, completed {r['completed']}, e_perp_max "
            f"{r.get('e_perp_max')}" for r in rows if r['controller'] == ctrl and
            (r.get('status') != 'ok' or r['completed'] != 'true' or float(r.get('e_perp_max') or 0) > D)]


def post(a):
    out = a.out
    lock = json.load(open(os.path.join(out, 'lock.yaml'), encoding='utf-8'))
    sel = [r for r in read_csv(os.path.join(out, 'selected.csv')) if r['kind'] == 'send_to_HIL']
    evals = read_csv(os.path.join(out, 'evals.csv'))
    pc = os.path.join(out, 'post_checks')
    if os.path.isdir(pc) and os.listdir(pc):
        sys.exit(f'{pc} already holds data')
    common = lock.get('common')
    if not common:
        sys.exit(f'{out}/lock.yaml has no common block (written by this script version): run a new folder')
    check_fingerprint(common, fingerprint(a.dll, common['s03_run'], unsealed=common.get('unsealed', False)),
                      'post')                                                         # S4C-2
    tc = common['tool_check']
    os.makedirs(pc, exist_ok=True)
    t0 = time.time()
    fw, path = fwsim.FirmwareSim(a.dll), o1.Path8()
    det, sens, mu3, notes, plant_full = [], [], [], {}, {}
    for s in sel:
        ctrl = s['controller']
        if s.get('feasible') != 'true':
            notes[ctrl] = 'no send_to_HIL gains: ' + s.get('note', '')
            continue
        L = lock['controllers'][ctrl]
        g = {n: float(s[n]) for n in s3.FACTORS[ctrl]}          # read back from the file (section 6)
        mus, speeds = L['eval_mu'], L['speeds']
        key = (s['ga_seed'], s['gen'], s['cand_id'])
        ref = {(float(e['v']), float(e['mu'])): e['log_sha256'] for e in evals
               if e['controller'] == ctrl and (e['ga_seed'], e['gen'], e['cand_id']) == key and e['attempt'] == '1'}
        jobs = [(ctrl, g, v, mu) for mu in mus for v in speeds]
        res = s3.run_in_new_process(a.dll, jobs, os.path.join(pc, '_work'))
        for (c, _, v, mu), r in zip(jobs, res):
            got = hashlib.sha256(json.dumps(r['digest'], sort_keys=True).encode()).hexdigest() if r['digest'] else ''
            det.append(dict(tool_check=tc, controller=ctrl, v=v, mu=mu, ga_log_sha256=ref.get((v, mu), ''),
                            new_process_log_sha256=got, equal=got != '' and got == ref.get((v, mu))))
        for kind, factors in (('b0_factor', B0_FACTORS if ctrl == 'ADRC' else ()), ('r_factor', R_FACTORS)):
            rows = []
            for mu in mus:
                for v in speeds:
                    for k in factors:
                        res_, _, back, before, after = run_custom(fw, path, ctrl, g, v, mu, **{kind: k})
                        plant_full.setdefault(f'{ctrl} {kind}={k!r} mu={mu!r}', dict(before=before, after=after))
                        rows.append(dict(tool_check=tc, controller=ctrl, check=kind, v=v, mu=mu, **{kind: k},
                                         completed=res_['completed'], status=res_['status'],
                                         e_perp_max=res_['e_perp_max'], tv_duty=res_['tv_duty'],
                                         read_back=json.dumps(back), plant_changed=json.dumps(
                                             {x: (before[x], after[x]) for x in before if before[x] != after[x]})))
            for r in rows:
                nom = next(x for x in rows if x['v'] == r['v'] and x['mu'] == r['mu'] and x[kind] == 1.0)
                r['tv_duty_change_pct'] = 100.0 * (r['tv_duty'] - nom['tv_duty']) / nom['tv_duty'] if nom['tv_duty'] else ''
            if rows:
                sensitive, reasons = judge(rows, kind)
                notes[f'{ctrl} {kind}'] = dict(sensitive=sensitive, reasons=reasons)
                sens += rows
        if len(mus) == 1:                       # section 6: the three mu when the GA used the middle one
            levels = lock['s03_mu_levels']
            for mu in levels:
                for v in speeds:
                    row, _ = s3.safe_run(fw, path, ctrl, g, v, mu)
                    mu3.append(dict(tool_check=tc, controller=ctrl, v=v, mu=mu,
                                    **{k: row.get(k, '') for k in EVAL_COLS if k in row}))
            warn = three_mu_findings(mu3, ctrl)
            if len(set(levels)) < 3:            # S4C-10: never an empty list for a check that was not done
                notes[f'{ctrl} three_mu'] = dict(
                    done=False, note=f'not applicable: the S03 run has the mu levels {levels} only; the three-mu '
                                     'check of S04 6 was NOT done', findings=warn)
            else:
                notes[f'{ctrl} three_mu'] = dict(done=True, findings=warn,
                                                 note='findings: runs not passing the gate; empty = every run ok, '
                                                      'completed and e_perp_max <= D')
    write_rows(os.path.join(pc, 'determinism.csv'), det)
    write_rows(os.path.join(pc, 'sensitivity.csv'), sens)
    write_rows(os.path.join(pc, 'three_mu.csv'), mu3)
    with open(os.path.join(pc, 'plant_params.json'), 'w', encoding='utf-8') as f:     # S4C-8
        json.dump(dict(note='every attribute of the plant parameters used, before and after the change '
                            '(one entry per controller, check, factor and mu; the same for every speed)',
                       entries=plant_full), f, indent=1, default=str)
    work = os.path.join(pc, '_work')
    if os.path.isdir(work) and not os.listdir(work):
        os.rmdir(work)
    with open(os.path.join(pc, 'notes.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(tool_check=tc, determinism_all_equal=all(d['equal'] for d in det) if det else None, notes=notes,
                       script=s3.git_state(os.path.join(HERE, '..')), dll_sha256=s3.sha256_file(a.dll),
                       criterion='B0_SENS/2026-11-05_run01: sensitive if completed changes, e_perp_max <= D changes, '
                                 'or e_perp_max changes by more than 50 % of the factor-1.0 value; tv_duty change '
                                 'reported, not a criterion (S04 6, B8b-21); rank after the change not reported',
                       r_factors=R_FACTORS, b0_factors=B0_FACTORS), f, indent=1, default=str)
    log_phase(out, dict(phase='post', elapsed_s=round(time.time() - t0)))
    write_root_meta(out, a.dll, lock)
    print(f'post checks -> {pc}; tool_check {tc}; determinism equal: {all(d["equal"] for d in det) if det else None}')
    for k, v in notes.items():
        print(f'  {k}: {v}')
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phase', required=True, choices=('trial', 'main', 'post'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--controller', choices=s3.CONTROLLERS)
    ap.add_argument('--s03', help='S03 run folder (levels.csv, meta.yaml, sealed/runs.csv)')
    ap.add_argument('--s03-unsealed', action='store_true',
                    help='campaign A has ended (L56): the S03 results may be read for the mu rule')
    ap.add_argument('--preconditions', help='JSON {condition: where checked} for the trial (S04 4.3)')
    ap.add_argument('--gate-speeds', type=float, nargs='+', default=None,
                    help='speeds still in the O1 gate (default: gate_speeds of the S03 meta.yaml)')
    ap.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    ap.add_argument('--tool-check', action='store_true', help='tool test only, never main numbers')
    ap.add_argument('--pop', type=int, default=POP)
    ap.add_argument('--gen-base', type=int, default=GEN_BASE)
    ap.add_argument('--gen-max', type=int, default=GEN_MAX)
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='main phase GA seeds (default 1-5)')
    a = ap.parse_args()
    if a.phase == 'post':
        return post(a)
    if not a.controller or not a.s03:
        sys.exit('--controller and --s03 are needed for the trial and main phases')
    if not a.tool_check and (a.pop, a.gen_base, a.gen_max) != (POP, GEN_BASE, GEN_MAX):
        sys.exit(f'budget is locked by S04 4.3 (pop {POP}, {GEN_BASE} / {GEN_MAX} generations); --tool-check to change it')
    if not a.tool_check and a.seeds not in (None, list(SEEDS)):
        sys.exit('S04 4.2: GA seeds 1-5; --tool-check to change them')
    if a.seeds and len(set(a.seeds)) != len(a.seeds):                         # S4C-13
        sys.exit(f'--seeds has a repeated seed: {a.seeds}')
    if not a.tool_check and a.gate_speeds:                                     # S4C-3
        sys.exit('S04 3 / S03 8: the gate speeds are those of the S03 run; a new gate needs a new S03 run '
                 '(--gate-speeds only with --tool-check)')
    if a.gen_base > a.gen_max or a.gen_base < 1:
        sys.exit('--gen-base must be in 1 .. --gen-max')
    ctrl = a.controller
    s03_meta = open(os.path.join(a.s03, 'meta.yaml'), encoding='utf-8').read()
    s03_trial = meta_value(s03_meta, 'trial') in (True, 'true')
    s03_mu = [float(m) for m in meta_value(s03_meta, 'mu_levels')]
    if meta_value(s03_meta, 'status') != 'ran to the end':
        sys.exit(f'S03 run {a.s03} did not run to the end')
    if not a.tool_check:
        if s03_trial or len(s03_mu) != 3:
            sys.exit('S04 needs the main S03 run (three mu levels, not --trial); --tool-check for a tool test')
        if not a.s03_unsealed:
            sys.exit('L56: the mu rule reads sealed/runs.csv of S03; give --s03-unsealed only after campaign A ended')
    gate_all = [float(v) for v in meta_value(s03_meta, 'gate_speeds')]
    gate = [v for v in s3.SPEEDS if any(math.isclose(v, g) for g in (a.gate_speeds or gate_all))]
    if not gate or (a.gate_speeds and len(gate) != len(a.gate_speeds)):
        sys.exit(f'--gate-speeds must be a non-empty subset of {list(s3.SPEEDS)}')
    if any(not any(math.isclose(v, g) for g in gate_all) for v in gate):
        sys.exit(f'a gate speed is not in the S03 gate {gate_all}: the S03 levels were not set for it')
    dom = read_domain(a.s03, ctrl)
    mu_mid = s03_mu[len(s03_mu) // 2]
    lock_path = os.path.join(a.out, 'lock.yaml')
    fw = fwsim.FirmwareSim(a.dll)
    if fw.build_info.get('esp32', {}).get('dirty', True):
        print('WARNING: fwsim.dll was built from a dirty firmware tree; not usable as a main result')
    import pymoo
    t0 = time.time()

    fp = fingerprint(a.dll, a.s03, unsealed=a.s03_unsealed and not a.tool_check)
    if s03_dll_sha(s03_meta) != fp['dll_sha256']:                 # S4C-2: the DLL the S03 run used
        sys.exit(f'{a.dll} (sha256 {fp["dll_sha256"][:12]}…) is not the DLL of the S03 run '
                 f'({str(s03_dll_sha(s03_meta))[:12]}…): S04 runs on the S03 firmware (S04 7)')

    if a.phase == 'trial':
        tdir = os.path.join(a.out, 'trial', ctrl)
        if os.path.isdir(tdir) and os.listdir(tdir):
            sys.exit(f'{tdir} already holds data: use a new folder')
        lock = json.load(open(lock_path, encoding='utf-8')) if os.path.isfile(lock_path) else dict(controllers={})
        if ctrl in lock['controllers']:
            sys.exit(f'{lock_path} already locks {ctrl}')
        common = lock.get('common')
        if lock['controllers'] and not common:
            sys.exit(f'{lock_path} has no common block (older script version): use a new folder')
        if common:                              # S4C-1, S4C-2: one mode and one source for the whole folder
            if common['tool_check'] != a.tool_check:
                sys.exit(f'{lock_path} was started with tool_check = {common["tool_check"]}; this trial has '
                         f'{a.tool_check}: a tool check and a main run never share a folder (S4C-1)')
            check_fingerprint(common, fp, 'trial')
        else:
            common = lock['common'] = dict(tool_check=a.tool_check, s03_run=os.path.abspath(a.s03), fingerprint=fp,
                                           s03_mu_levels=s03_mu, unsealed=bool(a.s03_unsealed and not a.tool_check))
        pre = {}
        if not a.tool_check:
            if not a.preconditions:
                sys.exit('S04 4.3: --preconditions <json> with ' + ', '.join(PRECONDITIONS))
            pre = json.load(open(a.preconditions, encoding='utf-8'))
            miss = [k for k in PRECONDITIONS if not (isinstance(pre.get(k), str) and pre[k].strip())]   # S4C-4
            if miss:
                sys.exit(f'S04 4.3: preconditions without "where checked" (a non-empty text): {miss}')
        if a.tool_check:
            decision, report = 'mid', dict(note='tool check: mu rule not applied, the S03 run has one mu level '
                                                 'or sealed/ is not read (L56)')
        else:
            decision, report = mu_rule(a.s03, ctrl, gate, s03_mu)
        eval_mu = [mu_mid] if decision == 'mid' else list(s03_mu)
        os.makedirs(tdir, exist_ok=True)
        L_new = dict(tool_check=a.tool_check, domain=dom, s03_run=os.path.abspath(a.s03), speeds=gate, eval_mu=eval_mu,
                     mu_decision=decision, mu_rule_report=report, pop=a.pop, gen_rule=dict(base=a.gen_base, max=a.gen_max),
                     seeds=list(a.seeds or SEEDS), preconditions=pre, n_gen='set by this trial (S04 4.3)')
        # S4C-12: the root meta.yaml holds the locked items and the pymoo reading before the GA runs
        write_root_meta(a.out, a.dll, dict(lock, controllers=dict(lock['controllers'], **{ctrl: L_new})),
                        running=f'trial {ctrl}')
        work = os.path.join(tdir, '_work')
        f2_def, def_score, def_evals, def_ok = default_f2(fw, a.dll, ctrl, gate, mu_mid, work)
        if not def_ok:                          # S4C-7: no HV reference from a penalty or a failed run
            write_rows(os.path.join(tdir, 'evals.csv'), def_evals, EVAL_COLS)
            sys.exit(f'STOP: the default gains of {ctrl} did not give a valid run at mu {mu_mid} '
                     f'(status {def_score["status"]}, f2 {def_score["f2"]}): no HV reference (S04 4.2); tell Liam. '
                     f'Runs in {tdir}/evals.csv')
        ref = (D, HV_REF_MULT * f2_def)
        ev = Evaluator(fw, a.dll, ctrl, gate, eval_mu, work)
        history, hv_rows = [], []
        final, n_gen, extended = run_seed(ev, dom, SEEDS[0], a.pop, None, ref, history, hv_rows,
                                          ext=(a.gen_base, a.gen_max))
        feasible_final = any(p['feasible'] for p in final)
        locked_gen = a.gen_max if extended else a.gen_base
        stop = (extended and not feasible_final)
        check_eval_keys(def_evals + ev.evals)                                   # S4C-13
        write_rows(os.path.join(tdir, 'history.csv'), history)
        write_rows(os.path.join(tdir, 'evals.csv'), def_evals + ev.evals, EVAL_COLS)
        write_rows(os.path.join(tdir, 'hv.csv'), hv_rows)
        write_rows(os.path.join(tdir, 'pareto.csv'), pareto_rows(ctrl, {SEEDS[0]: final}), POP_COLS)
        write_rows(os.path.join(tdir, 'final_population.csv'), population_rows(ctrl, {SEEDS[0]: final}), POP_COLS)
        elapsed = time.time() - t0
        L_new.update(n_gen=locked_gen, gen_rule=dict(
            base=a.gen_base, max=a.gen_max, compared_with=gen_compare(a.gen_base),
            hv_base=hv_rows[a.gen_base - 1]['hv'] if len(hv_rows) >= a.gen_base else None,
            hv_compared=hv_rows[gen_compare(a.gen_base) - 1]['hv'] if len(hv_rows) >= a.gen_base else None,
            extended=extended), hv_ref=list(ref), default_f2=f2_def, default_score=def_score,
            trial_elapsed_s=round(elapsed), trial_generations=n_gen, trial_feasible_final=feasible_final,
            stop=('no feasible point after the extended trial: tell Liam (S04 4.3)' if stop else None))
        lock['controllers'][ctrl] = L_new
        lock.update(s03_mu_levels=s03_mu, rule=SELECTION_RULE, pymoo=pymoo.__version__)
        with open(lock_path, 'w', encoding='utf-8') as f:
            json.dump(lock, f, indent=1, default=str)
        write_meta(os.path.join(tdir, 'meta.yaml'), a, fw, ctrl, dict(
            phase='trial', elapsed_s=round(elapsed), generations=n_gen, extended=extended, locked_n_gen=locked_gen,
            mu_decision=decision, eval_mu=eval_mu, speeds=gate, pop=a.pop, seeds_locked=L_new['seeds'],
            gen_rule=L_new['gen_rule'], hv_ref=list(ref), preconditions=pre), pymoo, eval_mu)
        log_phase(a.out, dict(phase='trial', controller=ctrl, elapsed_s=round(elapsed)))
        write_root_meta(a.out, a.dll, lock)
        print(f'[{ctrl}] trial: {n_gen} generations, extended {extended}, feasible at the end {feasible_final}; '
              f'locked n_gen {locked_gen}; {round(elapsed)} s -> {tdir}')
        if stop:
            print('STOP: no feasible point after the extended trial: tell Liam (S04 4.3)')
            return 2
        return 0

    # main phase
    if not os.path.isfile(lock_path):
        sys.exit(f'{lock_path} missing: run --phase trial first (S04 4.3)')
    lock = json.load(open(lock_path, encoding='utf-8'))
    L = lock['controllers'].get(ctrl)
    if L is None:
        sys.exit(f'{lock_path} does not lock {ctrl}: run its trial first')
    if L.get('stop'):
        sys.exit(f'{ctrl}: the trial stopped ({L["stop"]})')
    common = lock.get('common')
    if not common:
        sys.exit(f'{lock_path} has no common block (older script version): use a new folder')
    if common['tool_check'] != a.tool_check or L.get('tool_check') != a.tool_check:          # S4C-1
        sys.exit('the lock and this run differ in --tool-check')
    check_fingerprint(common, fp, 'main')                                                      # S4C-2
    if not a.tool_check:                        # S4C-1: the controller's own trial had the full lock
        miss = [k for k in PRECONDITIONS if not (isinstance(L['preconditions'].get(k), str)
                                                 and L['preconditions'][k].strip())]
        if miss or (L['pop'], L['gen_rule']['base'], L['gen_rule']['max']) != (POP, GEN_BASE, GEN_MAX) \
                or L['seeds'] != list(SEEDS):
            sys.exit(f'{ctrl}: lock.yaml does not hold the S04 4.2-4.3 budget, seeds and preconditions '
                     f'(missing {miss}): not a main-run lock')
    if dom != L['domain'] or gate != L['speeds'] or os.path.abspath(a.s03) != L['s03_run']:
        sys.exit('the domain, gate speeds or S03 run differ from lock.yaml (S04 7: locked before the main runs)')
    seeds = list(a.seeds or L['seeds'])
    if not a.tool_check and seeds != L['seeds']:
        sys.exit('seeds differ from lock.yaml')
    if os.path.isfile(os.path.join(a.out, f'history_{ctrl}.csv')):
        sys.exit(f'{a.out} already holds the main runs of {ctrl}')
    ev = Evaluator(fw, a.dll, ctrl, L['speeds'], L['eval_mu'], os.path.join(a.out, '_work'))
    history, hv_rows, finals = [], [], {}
    ref = tuple(L['hv_ref'])
    for seed in seeds:
        finals[seed], _, _ = run_seed(ev, dom, seed, L['pop'], L['n_gen'], ref, history, hv_rows)
    sel, stats = summarise(ctrl, dom, finals)
    errs = seed_error_rates(ev.evals, ctrl)
    check_eval_keys(ev.evals)                                                   # S4C-13
    for name, rows, cols in ((f'history_{ctrl}.csv', history, None), (f'evals_{ctrl}.csv', ev.evals, EVAL_COLS),
                             (f'hv_{ctrl}.csv', hv_rows, None), (f'pareto_{ctrl}.csv', pareto_rows(ctrl, finals), POP_COLS),
                             (f'final_population_{ctrl}.csv', population_rows(ctrl, finals), POP_COLS),
                             (f'selected_{ctrl}.csv', sel, None)):
        write_rows(os.path.join(a.out, name), rows, cols)
    merge_controllers(a.out)
    elapsed = time.time() - t0
    write_meta(os.path.join(a.out, f'meta_{ctrl}.yaml'), a, fw, ctrl, dict(
        phase='main', elapsed_s=round(elapsed), seeds=seeds, n_gen=L['n_gen'], pop=L['pop'], eval_mu=L['eval_mu'],
        speeds=L['speeds'], selection=stats, seed_error_rates=errs,
        seeds_over_10pct=[s for s, e in errs.items() if e['over_10pct']]), pymoo, L['eval_mu'])
    log_phase(a.out, dict(phase='main', controller=ctrl, elapsed_s=round(elapsed), seeds=seeds))
    write_root_meta(a.out, a.dll, lock)
    w = os.path.join(a.out, '_work')
    if os.path.isdir(w) and not os.listdir(w):
        os.rmdir(w)
    send = sel[-1]
    print(f'[{ctrl}] main: {len(seeds)} seeds, {stats["n_feasible_seeds"]} feasible, {round(elapsed)} s; '
          f'send_to_HIL: {"f1 %.4f f2 %.3f" % (send["f1"], send["f2"]) if send["feasible"] else send["note"]}')
    if any(e['over_10pct'] for e in errs.values()):
        print('NOTE: a seed has more than 10 % inconclusive / run_error runs: CODE looks for a tool error (S04 3)')
    return 0 if send['feasible'] else 2


SELECTION_RULE = ('section 5: f1* = min f1; keep f1 <= f1* + 0.01 m; min f2, then f1, then gains lexicographic, '
                  'then seed; send_to_HIL on the union of the feasible non-dominated sets of the feasible seeds')


def gen_compare(gen_base):
    """Section 4.3: the generation compared with gen_base (HV_20 against HV_15). With --tool-check and
    gen_base < 6 there is no gen_base - 5: generation 1 is used (with gen_base = 1 the rule can only
    extend on HV = 0). S4C-16: written down here and in lock.yaml (gen_rule.compared_with)."""
    return max(1, gen_base - 5)


SOURCES = ('fwsim/s04_ga.py', 'fwsim/s03_doe.py', 'fwsim/o1_feasibility.py', 'fwsim/fwsim.py', 'fwsim/tune_ga.py',
           'rpi5/trajectory_pub.py', 'rpi5/imu_model.py', 'rpi5/plant_blob.py', 'tools/run_validity.py')
PYMOO_FILES = ('core/individual.py', 'core/survival.py', 'algorithms/moo/nsga2.py')     # PYMOO_CV refers to these


def pymoo_file_hashes():
    """S4C-18: SHA-256 of the installed pymoo files that PYMOO_CV cites, so the file:line can be re-checked."""
    import pymoo
    root = os.path.dirname(pymoo.__file__)
    return {n: s3.sha256_file(os.path.join(root, n)) for n in PYMOO_FILES}


def fingerprint(dll, s03_dir, unsealed=False):
    """S4C-2: what must not change between the trial, the main runs and the post checks of one folder
    (S04 7): the DLL, the S03 files the domain and the mu rule come from (sealed/runs.csv only with
    --s03-unsealed, L56), the plant source blobs (L34), the Python sources the simulation and the
    scoring import (SOURCES), D, the library versions and the pymoo files of PYMOO_CV."""
    names = ('meta.yaml', 'levels.csv', 'design.csv') + (('sealed/runs.csv',) if unsealed else ())
    s03_files = {}
    for n in names:
        p = os.path.join(s03_dir, n)
        s03_files[n] = s3.sha256_file(p) if os.path.isfile(p) else None
    root = os.path.join(HERE, '..')
    scripts = {n: s3.sha256_file(os.path.join(root, n)) for n in SOURCES}
    return dict(dll_sha256=s3.sha256_file(dll), s03_files=s03_files, plant_src=fwsim.PLANT_SRC, scripts=scripts, D=D,
                libraries=fwsim.lib_versions(), pymoo_files=pymoo_file_hashes())


def s03_dll_sha(s03_meta):
    """dll_sha256 of the S03 meta.yaml (indented under firmware:), or None."""
    for line in s03_meta.splitlines():
        if line.strip().startswith('dll_sha256:'):
            return line.split(':', 1)[1].split('#')[0].strip().strip('"')
    return None


def check_fingerprint(common, fp, phase):
    old = common['fingerprint']
    diff = sorted(k for k in set(old) | set(fp) if json.dumps(old.get(k), sort_keys=True, default=str)
                  != json.dumps(fp.get(k), sort_keys=True, default=str))
    if diff:
        sys.exit(f'{phase}: {diff} differ from lock.yaml (S04 7: a change of DLL, S03, plant, O1 or scripts needs '
                 f'a new S04 folder; S4C-2)')


def check_eval_keys(evals):
    keys = [(e['controller'], e['ga_seed'], e['gen'], e['cand_id'], e['v'], e['mu'], e['attempt']) for e in evals]
    if len(keys) != len(set(keys)):
        sys.exit('evals: the key (controller, ga_seed, gen, cand_id, v, mu, attempt) is not unique (S04 8; S4C-13)')


def log_phase(out, entry):
    p = os.path.join(out, 'phases.json')
    log = json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else []
    log.append(dict(entry, ended=time.strftime('%Y-%m-%d %H:%M:%S')))
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=1)


def write_root_meta(out, dll, lock, running=None):
    """S4C-12: <out>/meta.yaml of S04 8, rewritten at the start of a trial (before any GA run) and after
    every phase. Only JSON values (valid YAML). Do not edit meta.yaml by hand: notes of CODE (for
    example the cause found for a seed error rate over 10 %, S04 3) go to <out>/code_notes.json,
    which is copied verbatim into meta.yaml (code_notes) every time it is written (S4C-R-1)."""
    p = os.path.join(out, 'phases.json')
    phases = json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else []
    pn = os.path.join(out, 'code_notes.json')
    code_notes = json.load(open(pn, encoding='utf-8')) if os.path.isfile(pn) else None
    fw_info = json.load(open(os.path.join(os.path.dirname(dll), 'fwsim_build.json'), encoding='utf-8')) \
        if os.path.isfile(os.path.join(os.path.dirname(dll), 'fwsim_build.json')) else None
    import pymoo
    common = lock.get('common', {})
    ctrls = {}
    for c, L in lock.get('controllers', {}).items():
        ctrls[c] = {k: L.get(k) for k in ('tool_check', 'domain', 'speeds', 'eval_mu', 'mu_decision', 'mu_rule_report',
                                          'pop', 'n_gen', 'gen_rule', 'seeds', 'hv_ref', 'default_f2', 'preconditions',
                                          'stop')}
        ctrls[c]['plant_by_mu'] = {repr(m): json.loads(fwsim.plant_meta(fwsim.plant_params(
            argparse.Namespace(ground='traction', mu=m)))) for m in (L.get('eval_mu') or [])}
    lines = dict(spec='S04 v5 (06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md)', date=time.strftime('%Y-%m-%d'),
                 setup='sim: fwsim, ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware',
                 operator='Claude (phiên CODE)', tool_check=common.get('tool_check'),
                 not_main_numbers=bool(common.get('tool_check')), running=running,
                 script=s3.git_state(os.path.join(HERE, '..')), fwsim_build=fw_info,
                 s03_run=common.get('s03_run'), s03_mu_levels=common.get('s03_mu_levels'),
                 fingerprint=common.get('fingerprint'),
                 pymoo=dict(version=pymoo.__version__, checked_version=PYMOO_CHECKED, constraint_handling=PYMOO_CV,
                            recheck=pymoo.__version__ != PYMOO_CHECKED),
                 selection_rule=SELECTION_RULE, D=D, imu=s3.IMU, plant_seed=s3.SEED, controllers=ctrls,
                 phases=phases, elapsed_s=sum(int(x.get('elapsed_s', 0)) for x in phases),
                 code_notes=code_notes, pymoo_files=pymoo_file_hashes(),
                 files=('lock.yaml, phases.json, code_notes.json (CODE notes, optional), trial/<controller>/, '
                        'history/evals/hv/pareto/final_population/selected[_<controller>].csv, meta_<controller>.yaml, '
                        'post_checks/'))
    with open(os.path.join(out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write('# S04 run folder (written by fwsim/s04_ga.py; every value is JSON, so the file is valid YAML)\n')
        for k, v in lines.items():
            f.write(f'{k}: {json.dumps(v, ensure_ascii=False, default=str)}\n')


def merge_controllers(out):
    """history.csv, evals.csv, hv.csv, pareto.csv, final_population.csv, selected.csv over the controllers run so far."""
    for name in ('history', 'evals', 'hv', 'pareto', 'final_population', 'selected'):
        rows, found = [], False
        for ctrl in s3.CONTROLLERS:
            p = os.path.join(out, f'{name}_{ctrl}.csv')
            if os.path.isfile(p):
                found = True
                rows += read_csv(p)
        if found:                               # S4C-R-2: a per-controller file with a header only is kept
            fixed = {'evals': EVAL_COLS, 'pareto': POP_COLS, 'final_population': POP_COLS}
            cols = fixed.get(name) or list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(out, name + '.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)


def write_meta(path, a, fw, ctrl, extra, pymoo, eval_mu):
    """Per-controller meta (trial/<ctrl>/meta.yaml, meta_<ctrl>.yaml). S4C-11: every string written with
    json.dumps (a Windows path in a YAML double-quoted scalar is not valid YAML); S4C-12: the plant of
    every mu of the evaluation, not a fixed mu."""
    b = fw.build_info
    plant = {repr(m): json.loads(fwsim.plant_meta(fwsim.plant_params(argparse.Namespace(ground='traction', mu=m))))
             for m in eval_mu}
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"""spec: S04 v5 (06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md)
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
tool_check: {str(a.tool_check).lower()}     # true: tool test, never main numbers
controller: {ctrl}
run: {json.dumps(extra, default=str)}
script: {json.dumps(s3.git_state(os.path.join(HERE, '..')))}
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework_at_build: {json.dumps(b.get('hil_framework'))}
  compiler: {json.dumps(b.get('compiler'))}
  dll_sha256: {s3.sha256_file(a.dll)}
s03_run: {json.dumps(os.path.abspath(a.s03))}
plant_by_mu: {json.dumps(plant)}
imu: {json.dumps(s3.IMU)}
plant_seed: {s3.SEED}
objectives: "f1 = max e_perp_max over the evaluation runs (m); f2 = max over mu of the mean over the gate speeds of tv_duty (1/s); minimised"
constraints: "g1 = runs not completed, inconclusive or with a track error (all runs when a run_error); g2 = (f1 - D) / D, D = {D} m; pymoo G <= 0"
pymoo: {json.dumps(dict(version=pymoo.__version__, checked_version=PYMOO_CHECKED, constraint_handling=PYMOO_CV,
                       recheck=pymoo.__version__ != PYMOO_CHECKED))}
libraries: {json.dumps(fwsim.lib_versions())}
interpretations:   # of this script, not written in S04 v5; Liam or THESIS correct if wrong
  - "a new script fwsim/s04_ga.py (S04 9.1 names tune_ga.py): tune_ga.py keeps its velocity objective for the old runs, unchanged"
  - "mu rule: every S03 design point that ran (centre, outer, inner box: the inner box always runs since L41), every gate speed; status other than ok (track_err, run_error) or y1 inconclusive at any mu is 'not satisfied'"
  - "HV: exact 2-D staircase on the feasible non-dominated points of the population after each generation's survival, strictly inside the reference; the trial rule compares generation gen_base with gen_base - 5 (15 and 20); with --tool-check and gen_base < 6 with generation 1"
  - "pareto.csv: the feasible non-dominated points of each seed's final population; final_population.csv: every point of it with a non-dominated flag (S04 8)"
  - "the HV reference is locked only from a valid run of the default gains (no run_error, f2 finite and > 0); otherwise the trial stops"
  - "per seed set (section 5): the feasible non-dominated points of the final population (pymoo algorithm.pop after the last generation)"
  - "run_error: attempt 2 in a new process is recorded with 'cause' empty; the candidate is scored as the first attempt (S04 3); CODE fills 'cause' when it finds a tool error"
  - "R sensitivity: motor_R of the four wheels x 0.8, 0.9, 1.0, 1.1, 1.2 (S04 6 says +-20 %), every other plant parameter as params.py (J, b not rescaled)"
  - "gains stored as Python repr (17 significant digits) and as the float32 value the firmware receives (fwsim set_pid / set_adrc take c_float)"
  - "evals.csv rows with ga_seed 0, gen 0 (trial folder): the firmware default gains at the middle mu, run once for the HV reference (section 4.2)"
""")


if __name__ == '__main__':
    sys.exit(main())
