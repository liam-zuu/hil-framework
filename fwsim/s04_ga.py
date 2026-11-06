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
      ga_seed, gen, cand_id, v, mu, attempt), hv.csv, pareto.csv, selected.csv, meta.yaml,
      lock.yaml, trial/ (the trial phase), post_checks/.

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
# pymoo 0.6.2 constraint handling, read in the installed library (S04 section 3; Liam asked CODE to
# record file:line before the trial). Re-read when the version differs.
PYMOO_CHECKED = '0.6.2'
PYMOO_CV = ("core/individual.py:29 cv_ieq default func=np.sum, eps 0, no scale, no pow; "
            "core/individual.py:600-623 constr_to_cv: CV = sum(max(0, g_i)) over the inequality constraints; "
            "core/survival.py:59-80 Survival.do: feasible (CV = 0) individuals ranked by RankAndCrowding first, "
            "the rest filled by increasing CV; algorithms/moo/nsga2.py:33-47 binary tournament: if either is "
            "infeasible the smaller CV wins (random on a tie), else dominance then crowding")
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
    lo, mid, hi = mu_levels
    by = {}
    for r in runs:
        by.setdefault((r['point_id'], r['box'], float(r['v'])), {})[float(r['mu'])] = r
    bad = []
    n = 0
    for (pid, box, v), rs in sorted(by.items()):
        if not any(math.isclose(v, g) for g in gate):
            continue
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
            if max(abs(y2[lo] - y2[mid]), abs(y2[hi] - y2[mid])) > MU_Y2_TOL:
                why.append('dY2 > 0.01 m')
            y3 = {m: float(r['tv_duty']) for m, r in rs.items()}
            ref = y3_ref[f'{ctrl} v={v:g}']
            if max(abs(y3[lo] - y3[mid]), abs(y3[hi] - y3[mid])) > MU_Y3_REL * ref:
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
            for k, x in enumerate(X):
                g = decode(dom, x)
                s = ev.candidate(g, ga_seed, state['gen'], k)
                F.append([s['f1'], s['f2']])
                G.append([s['g1'], s['g2']])
                seen.setdefault(np.asarray(x, dtype=float).tobytes(), (state['gen'], k))
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
        alg.evaluator.eval(problem, infills)
        alg.tell(infills=infills)
        pts = points_of(alg.pop, dom, ga_seed, seen)
        feas = [p for p in pts if p['feasible']]
        hv, n_out = hypervolume(feas, ref)
        hv_by_gen[state['gen']] = hv
        hv_rows.append(dict(controller=ev.ctrl, ga_seed=ga_seed, gen=state['gen'], hv=hv, n_feasible=len(feas),
                            n_nd_feasible=len(nondominated(feas)), n_nd_on_or_beyond_ref=n_out))
        log(f'[{ev.ctrl} seed {ga_seed}] gen {state["gen"]}: feasible {len(feas)}/{len(pts)}, HV {hv:.6g}')
        if ext and state['gen'] == ext[0]:
            h_b, h_a = hv_by_gen[ext[0]], hv_by_gen[max(1, ext[0] - 5)]
            extended = h_b == 0.0 or (h_b - h_a) / h_b > HV_EXT_REL
            if not extended:
                break
    return points_of(alg.pop, dom, ga_seed, seen), state['gen'], extended


def points_of(pop, dom, ga_seed, seen):
    pts = []
    for ind in pop:
        x = np.asarray(ind.X, dtype=float)
        gen, cand = seen.get(x.tobytes(), ('', ''))
        G = np.asarray(ind.G, dtype=float)
        pts.append(dict(ga_seed=ga_seed, gen=gen, cand_id=cand, f1=float(ind.F[0]), f2=float(ind.F[1]),
                        g1=float(G[0]), g2=float(G[1]), feasible=bool(np.all(G <= 0)),
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


def pareto_rows(ctrl, finals):
    rows = []
    for seed in sorted(finals):
        pts = finals[seed]
        feas = [p for p in pts if p['feasible']]
        nd_ids = {id(p) for p in nondominated(feas)}
        for p in pts:
            rows.append(dict(controller=ctrl, ga_seed=seed, gen=p['gen'], cand_id=p['cand_id'], feasible=p['feasible'],
                             nondominated_feasible=id(p) in nd_ids, f1=p['f1'], f2=p['f2'], g1=p['g1'], g2=p['g2'],
                             gains=json.dumps([repr(float(v)) for v in p['x']])))
    return rows


def default_f2(fw, ctrl, speeds, mu_mid):
    """f2 of the firmware default gains at the middle mu (HV reference, section 4.2)."""
    ev = Evaluator(fw, None, ctrl, speeds, [mu_mid], None)
    s = ev.candidate(s3.defaults(fw, ctrl), 0, 0, 0)
    return s['f2'], s, ev.evals


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
    back.append(dict(pos=fwsim.pos_meta(fw)))           # pos_kp, pos_ki; k_psi has no getter (S03 S3C-21)
    res, _ = o1.evaluate_v7(log, path, sch, lsum=p.lx + p.ly)
    return res, log, back, before, after


def plant_dump(p):
    out = {}
    for k, v in sorted(vars(p).items()):
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
    return out


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


def post(a):
    out = a.out
    lock = json.load(open(os.path.join(out, 'lock.yaml'), encoding='utf-8'))
    sel = [r for r in read_csv(os.path.join(out, 'selected.csv')) if r['kind'] == 'send_to_HIL']
    evals = read_csv(os.path.join(out, 'evals.csv'))
    pc = os.path.join(out, 'post_checks')
    if os.path.isdir(pc) and os.listdir(pc):
        sys.exit(f'{pc} already holds data')
    os.makedirs(pc, exist_ok=True)
    fw, path = fwsim.FirmwareSim(a.dll), o1.Path8()
    det, sens, mu3, notes = [], [], [], {}
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
            det.append(dict(controller=ctrl, v=v, mu=mu, ga_log_sha256=ref.get((v, mu), ''), new_process_log_sha256=got,
                            equal=got != '' and got == ref.get((v, mu))))
        for kind, factors in (('b0_factor', B0_FACTORS if ctrl == 'ADRC' else ()), ('r_factor', R_FACTORS)):
            rows = []
            for mu in mus:
                for v in speeds:
                    for k in factors:
                        res_, _, back, before, after = run_custom(fw, path, ctrl, g, v, mu, **{kind: k})
                        rows.append(dict(controller=ctrl, check=kind, v=v, mu=mu, **{kind: k},
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
            for mu in lock['s03_mu_levels']:
                for v in speeds:
                    row, _ = s3.safe_run(fw, path, ctrl, g, v, mu)
                    mu3.append(dict(controller=ctrl, v=v, mu=mu, **{k: row.get(k, '') for k in EVAL_COLS if k in row}))
            notes[f'{ctrl} three_mu'] = [f"v={r['v']} mu={r['mu']}: completed {r['completed']}, e_perp_max "
                                         f"{r.get('e_perp_max')}" for r in mu3 if r['controller'] == ctrl and
                                         (r['completed'] != 'true' or float(r.get('e_perp_max') or 0) > D)]
    write_rows(os.path.join(pc, 'determinism.csv'), det)
    write_rows(os.path.join(pc, 'sensitivity.csv'), sens)
    write_rows(os.path.join(pc, 'three_mu.csv'), mu3)
    work = os.path.join(pc, '_work')
    if os.path.isdir(work) and not os.listdir(work):
        os.rmdir(work)
    with open(os.path.join(pc, 'notes.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(determinism_all_equal=all(d['equal'] for d in det) if det else None, notes=notes,
                       script=s3.git_state(os.path.join(HERE, '..')), dll_sha256=s3.sha256_file(a.dll),
                       criterion='B0_SENS/2026-11-05_run01: sensitive if completed changes, e_perp_max <= D changes, '
                                 'or e_perp_max changes by more than 50 % of the factor-1.0 value; tv_duty change '
                                 'reported, not a criterion (S04 6, B8b-21); rank after the change not reported',
                       r_factors=R_FACTORS, b0_factors=B0_FACTORS), f, indent=1, default=str)
    print(f'post checks -> {pc}; determinism equal: {all(d["equal"] for d in det) if det else None}')
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

    if a.phase == 'trial':
        tdir = os.path.join(a.out, 'trial', ctrl)
        if os.path.isdir(tdir) and os.listdir(tdir):
            sys.exit(f'{tdir} already holds data: use a new folder')
        lock = json.load(open(lock_path, encoding='utf-8')) if os.path.isfile(lock_path) else dict(controllers={})
        if ctrl in lock['controllers']:
            sys.exit(f'{lock_path} already locks {ctrl}')
        pre = {}
        if not a.tool_check:
            if not a.preconditions:
                sys.exit('S04 4.3: --preconditions <json> with ' + ', '.join(PRECONDITIONS))
            pre = json.load(open(a.preconditions, encoding='utf-8'))
            miss = [k for k in PRECONDITIONS if not str(pre.get(k, '')).strip()]
            if miss:
                sys.exit(f'S04 4.3: preconditions without "where checked": {miss}')
        if a.tool_check:
            decision, report = 'mid', dict(note='tool check: mu rule not applied, the S03 run has one mu level '
                                                 'or sealed/ is not read (L56)')
        else:
            decision, report = mu_rule(a.s03, ctrl, gate, s03_mu)
        eval_mu = [mu_mid] if decision == 'mid' else list(s03_mu)
        os.makedirs(tdir, exist_ok=True)
        f2_def, def_score, def_evals = default_f2(fw, ctrl, gate, mu_mid)
        ref = (D, HV_REF_MULT * f2_def)
        ev = Evaluator(fw, a.dll, ctrl, gate, eval_mu, os.path.join(tdir, '_work'))
        history, hv_rows = [], []
        final, n_gen, extended = run_seed(ev, dom, SEEDS[0], a.pop, None, ref, history, hv_rows,
                                          ext=(a.gen_base, a.gen_max))
        feasible_final = any(p['feasible'] for p in final)
        locked_gen = a.gen_max if extended else a.gen_base
        stop = (extended and not feasible_final)
        write_rows(os.path.join(tdir, 'history.csv'), history)
        write_rows(os.path.join(tdir, 'evals.csv'), def_evals + ev.evals, EVAL_COLS)
        write_rows(os.path.join(tdir, 'hv.csv'), hv_rows)
        write_rows(os.path.join(tdir, 'pareto.csv'), pareto_rows(ctrl, {SEEDS[0]: final}))
        elapsed = time.time() - t0
        lock['controllers'][ctrl] = dict(
            domain=dom, s03_run=os.path.abspath(a.s03), speeds=gate, eval_mu=eval_mu, mu_decision=decision,
            mu_rule_report=report, pop=a.pop, n_gen=locked_gen, gen_rule=dict(
                base=a.gen_base, max=a.gen_max, hv_base=hv_rows[a.gen_base - 1]['hv'] if len(hv_rows) >= a.gen_base else None,
                hv_base_minus_5=hv_rows[max(1, a.gen_base - 5) - 1]['hv'] if len(hv_rows) >= a.gen_base else None,
                extended=extended), seeds=list(a.seeds or SEEDS), hv_ref=list(ref), default_f2=f2_def,
            default_score=def_score, preconditions=pre, trial_elapsed_s=round(elapsed), trial_generations=n_gen,
            trial_feasible_final=feasible_final, stop=('no feasible point after the extended trial: tell Liam (S04 4.3)'
                                                       if stop else None))
        lock.update(s03_mu_levels=s03_mu, rule=('section 5: f1* = min f1; keep f1 <= f1* + 0.01 m; min f2, then f1, '
                                                'then gains lexicographic, then seed'), tool_check=a.tool_check,
                    pymoo=pymoo.__version__)
        with open(lock_path, 'w', encoding='utf-8') as f:
            json.dump(lock, f, indent=1, default=str)
        write_meta(os.path.join(tdir, 'meta.yaml'), a, fw, ctrl, dict(phase='trial', elapsed_s=round(elapsed),
                   generations=n_gen, extended=extended, locked_n_gen=locked_gen, mu_decision=decision), pymoo)
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
    if lock.get('tool_check') != a.tool_check:
        sys.exit('the lock and this run differ in --tool-check')
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
    for name, rows, cols in ((f'history_{ctrl}.csv', history, None), (f'evals_{ctrl}.csv', ev.evals, EVAL_COLS),
                             (f'hv_{ctrl}.csv', hv_rows, None), (f'pareto_{ctrl}.csv', pareto_rows(ctrl, finals), None),
                             (f'selected_{ctrl}.csv', sel, None)):
        write_rows(os.path.join(a.out, name), rows, cols)
    merge_controllers(a.out)
    elapsed = time.time() - t0
    write_meta(os.path.join(a.out, f'meta_{ctrl}.yaml'), a, fw, ctrl, dict(
        phase='main', elapsed_s=round(elapsed), seeds=seeds, n_gen=L['n_gen'], pop=L['pop'], eval_mu=L['eval_mu'],
        speeds=L['speeds'], selection=stats, seed_error_rates=errs,
        seeds_over_10pct=[s for s, e in errs.items() if e['over_10pct']]), pymoo)
    w = os.path.join(a.out, '_work')
    if os.path.isdir(w) and not os.listdir(w):
        os.rmdir(w)
    send = sel[-1]
    print(f'[{ctrl}] main: {len(seeds)} seeds, {stats["n_feasible_seeds"]} feasible, {round(elapsed)} s; '
          f'send_to_HIL: {"f1 %.4f f2 %.3f" % (send["f1"], send["f2"]) if send["feasible"] else send["note"]}')
    if any(e['over_10pct'] for e in errs.values()):
        print('NOTE: a seed has more than 10 % inconclusive / run_error runs: CODE looks for a tool error (S04 3)')
    return 0 if send['feasible'] else 2


def merge_controllers(out):
    """history.csv, evals.csv, hv.csv, pareto.csv, selected.csv over the controllers run so far."""
    for name in ('history', 'evals', 'hv', 'pareto', 'selected'):
        rows = []
        for ctrl in s3.CONTROLLERS:
            p = os.path.join(out, f'{name}_{ctrl}.csv')
            if os.path.isfile(p):
                rows += read_csv(p)
        if rows:
            cols = EVAL_COLS if name == 'evals' else list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(out, name + '.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)


def write_meta(path, a, fw, ctrl, extra, pymoo):
    b = fw.build_info
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
  compiler: "{b.get('compiler')}"
  dll_sha256: {s3.sha256_file(a.dll)}
s03_run: "{os.path.abspath(a.s03)}"
plant: {fwsim.plant_meta(fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.5)))}
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
  - "HV: exact 2-D staircase on the feasible non-dominated points of the population after each generation's survival, strictly inside the reference; the trial rule compares generation gen_base with gen_base - 5 (15 and 20)"
  - "per seed set (section 5): the feasible non-dominated points of the final population (pymoo algorithm.pop after the last generation)"
  - "run_error: attempt 2 in a new process is recorded with 'cause' empty; the candidate is scored as the first attempt (S04 3); CODE fills 'cause' when it finds a tool error"
  - "R sensitivity: motor_R of the four wheels x 0.8, 0.9, 1.0, 1.1, 1.2 (S04 6 says +-20 %), every other plant parameter as params.py (J, b not rescaled)"
  - "gains stored as Python repr (17 significant digits) and as the float32 value the firmware receives (fwsim set_pid / set_adrc take c_float)"
  - "evals.csv rows with ga_seed 0, gen 0 (trial folder): the firmware default gains at the middle mu, run once for the HV reference (section 4.2)"
""")


if __name__ == '__main__':
    sys.exit(main())
