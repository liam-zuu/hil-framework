"""
o1_feas_search.py: feasibility search of S05 C12 / O1 v11.12 section 3.9 item 5 on fwsim.

Question (O1 3.9 (5); gate dossier v3.2 section 4, CLAUDE_THESIS
00_OVERVIEW/2026-11-12_De_xuat_cong_dung_tune_bam_huong_van_toc_v3.md): for each controller, is there ONE
set of the 7 gains that passes G1-G5 at EVERY speed level of a level set at once, on fwsim?
Design: CLAUDE_THESIS 03_CODE/docs/2026-11-14_Thiet_ke_tim_kha_thi_C12_v1.md (Liam 14/11/2026, CODE_FEAS
local_bda8d6df, "Duyệt, code luôn (Recommended)"; Q1 "Trạng thái + số đo", Q2 "Sau Codex 2 vòng").

Configuration (as S03 3.3): campaign firmware DLL, plant 'traction' (method A), mu 0.436 (middle), IMU 20 Hz
with the measured-noise replay, seed 1 (s03_doe.IMU, s03_doe.SEED); figure 8 with o1_feasibility.Schedule
(a_r = o1_feasibility.A_R = 0.5 m/s^2 unless --a-r); gains set as s03_doe.sim_kwargs; scored with
o1_gate_eval.evaluate_fig8 + o1_gate.gate(u_t = 0, DECIDE_ALL). K1, K2 are not run.
Every module of the S04 SIM_SOURCES lock and the plant blobs are imported, never changed. --a-r sets the
module attribute o1_feasibility.A_R at run time (the file is not touched; recorded in meta.yaml).

Search (design 4):
  domain   per gain the S03 3.1 sweep box [a, b]: the ends of s03_doe.sweep_grid without the 0 point; log
           coding when b/a >= 4 (s03_doe.scale_of), else linear; GA variable ln g or g on [xl, xu].
  GA       pymoo GA (single objective), pop 24, 20 generations, seed 20261113, default operators; first
           population: the firmware default + 23 points drawn uniform in the coded box with
           numpy default_rng(20261113) (the same draw for every controller and level set).
  V        sum over levels of c + sum_{G2..G5} h; c = 0 only when G1 'dat', status 'ok', run_bad 0 and
           n_undef 0 for G2-G5, else 1; h = max(0, q / threshold - 1), 1 when q is not finite;
           a run_error level counts c = 1 and h = 1 each (the GA goes on).
  witness  the first evaluated candidate with V = 0 (generation, index order).
  no early stop: all 6 searches (2 controllers x 3 nested level sets) run their whole budget.
After all searches: every witness is run again in a NEW process (DLL loaded again): the log digests must be
equal bit for bit and G1-G5 must pass when scored independently of V; otherwise the script stops.
Nested sets T3 = {0.2, 0.5, 0.8} -> T2 = {0.2, 0.5} -> T1 = {0.2}: a witness of a larger set is a witness of
its subsets (checked on the stored per-level results); a set is 'tim_thay' when its own search or a larger
set's search has a witness. Feasibility level set = the first of T3, T2, T1 where both controllers have
'tim_thay'; none -> 'rong' (O1 3.9 (3b)). The script never gives 'bat_kha_thi_co_chung_minh'.

Information fence (O1 3.9 (5), dossier 4 (6)): the witnesses, the search history, the determinism check,
the subset check and the low / high mu check go to --out/sealed/ (SHA-256 in meta.yaml); nobody opens it
before campaign A ends. The console prints one start line, 'job k/6 started' lines and, at the very end,
the labels, the level set and the default-gain table (states and q of G1-G5 at the middle mu: Liam 14/11 Q1).
Child processes write their output under sealed/; a failing child is reported by exception class only.

Sub-commands:
  python fwsim/o1_feas_search.py search  --out 05_DATA/results/O1_FEAS/<date>_runNN [--dll D] [--jobs 6]
  python fwsim/o1_feas_search.py recheck --from <search run> --out <new> --levels 0.2,0.5 --a-r A [--dll D]
  python fwsim/o1_feas_search.py inbox   --from <search run> --out <new> --s03 <S03 main run>
  --tool-check (search): mu 0.5, pop 4, 2 generations, folder marked tool_check: true; labels and the
  default table are NOT printed, only the mean time of one run. Never main numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..', 'tools'), os.path.join(HERE, '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import fwsim                                            # noqa: E402
import o1_feasibility as o1                             # noqa: E402
import o1_gate as G                                     # noqa: E402
import o1_gate_eval as E                                # noqa: E402
import s03_doe as s3                                    # noqa: E402
import tune_ga                                          # noqa: E402

SCRIPT_VERSION = 'o1_feas_search_v1'
CONTROLLERS = ('PID', 'ADRC')
LEVEL_SETS = ((0.2, 0.5, 0.8), (0.2, 0.5), (0.2,))     # nested, largest first (O1 3.9 (5))
SET_NAMES = ('T3', 'T2', 'T1')
GA_SEED = 20261113
POP, N_GEN = 24, 20
TOOL_POP, TOOL_GEN = 4, 2
MU_MID, MU_LO, MU_HI = 0.436, 0.413, 0.473              # MU_FLOOR/2026-11-09_run02 min, mean, max
MU_TOOL = 0.5                                           # fwsim assumption (S03 --trial)
FOUND, NOT_FOUND = 'tim_thay', 'chua_tim_thay'
EMPTY_SET = 'rong'
LEVEL_KEYS = ('v', 'mu', 'V_level', 'c', 'status', 'run_bad', 'error', 'log_sha',
              'g1_state', 'g1_q') + tuple(f'{k}_{f}' for k in G.MEASURED for f in ('state', 'q', 'n_undef', 'h'))


class FeasStop(Exception):
    """A stop of the design (determinism check, configuration): print the reason only."""


# ── domain ────────────────────────────────────────────────────────────────────────────

def box_of(ctrl, default):
    """[{name, default, a, b, scale, xl, xu}] for the 7 gains (design 4.1). default: {name: value}."""
    lo_b, hi_b = tune_ga.BOUNDS[ctrl]
    out = []
    for name, lo, hi in zip(s3.FACTORS[ctrl], lo_b, hi_b):
        d = float(default[name])
        if not (d > 0.0 and math.isfinite(d)):
            raise FeasStop(f'{ctrl} {name}: default {d!r} is not > 0, the box needs a > 0 (dossier 4 (2))')
        grid = s3.sweep_grid(name, d, lo, hi)
        pos = [x for x in grid if x > 0.0]
        a, b = float(min(pos)), float(max(pos))
        if not (0.0 < a <= d <= b):
            raise FeasStop(f'{ctrl} {name}: default {d} outside its box [{a}, {b}]')
        scale = s3.scale_of(a, b)[0]
        xl, xu = (math.log(a), math.log(b)) if scale == 'log' else (a, b)
        out.append(dict(controller=ctrl, name=name, default=d, a=a, b=b, scale=scale, xl=xl, xu=xu))
    return out


def encode(box, g):
    return np.array([math.log(g[d['name']]) if d['scale'] == 'log' else float(g[d['name']]) for d in box])


def decode(box, x):
    """GA vector -> gains {name: value}, clipped to [a, b]; the coded default decodes to the default exactly."""
    g = {}
    for d, xi in zip(box, x):
        xi = float(xi)
        e = math.log(d['default']) if d['scale'] == 'log' else d['default']
        if xi == e:
            g[d['name']] = d['default']
            continue
        v = math.exp(xi) if d['scale'] == 'log' else xi
        g[d['name']] = min(max(v, d['a']), d['b'])
    return g


def first_population(box, seed=GA_SEED, pop=POP):
    """Row 0 the default, then pop - 1 rows uniform in the coded box (same draw for every box)."""
    u = np.random.default_rng(seed).random((pop - 1, len(box)))
    xl = np.array([d['xl'] for d in box])
    xu = np.array([d['xu'] for d in box])
    return np.vstack([encode(box, {d['name']: d['default'] for d in box}), xl + u * (xu - xl)])


# ── one level, one candidate ─────────────────────────────────────────────────────────

def _finite(x):
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def log_sha(log):
    """One SHA-256 over s03_doe.log_digest (equal <=> every log array equal bit for bit)."""
    return hashlib.sha256(json.dumps(s3.log_digest(log), sort_keys=True).encode()).hexdigest()


def level_penalty(r):
    """(c, {cond: h}, V_level) of one level result (design 4.3)."""
    if r.get('status') == 'run_error':
        return 1, {k: 1.0 for k in G.MEASURED}, 1.0 + len(G.MEASURED)
    ok = (r.get('g1_state') == 'dat' and r.get('status') == 'ok' and r.get('run_bad') == 0
          and all(r.get(f'{k}_n_undef') == 0 for k in G.MEASURED))
    c = 0 if ok else 1
    h = {}
    for k in G.MEASURED:
        q = r.get(f'{k}_q')
        h[k] = max(0.0, float(q) / G.THRESHOLDS[k] - 1.0) if _finite(q) else 1.0
    return c, h, float(c) + float(sum(h.values()))


def level_from_gate(v, mu, gt, status, run_bad):
    """Level result row from an o1_gate.gate() dict."""
    r = dict(v=v, mu=mu, status=status, run_bad=run_bad, error='', g1_state=gt['g1']['state'], g1_q=gt['g1']['q'])
    for k in G.MEASURED:
        r.update({f'{k}_state': gt[k]['state'], f'{k}_q': gt[k]['q'], f'{k}_n_undef': gt[k]['n_undef']})
    return finish_level(r)


def error_level(v, mu, err):
    return finish_level(dict(v=v, mu=mu, status='run_error', run_bad=float('nan'), error=err, g1_state='',
                             g1_q='', **{f'{k}_{f}': '' for k in G.MEASURED for f in ('state', 'q', 'n_undef')}))


def finish_level(r):
    c, h, vl = level_penalty(r)
    r.update(c=c, V_level=vl, **{f'{k}_h': h[k] for k in G.MEASURED})
    r.setdefault('log_sha', '')
    return r


def gate_pass(r):
    """Independent of V (SC-13): the gate outcome over the deciding conditions, status, run validity."""
    return r.get('outcome_decide') == 'dat' and r.get('status') == 'ok' and r.get('run_bad') == 0


class Runner:
    """Runs one level on fwsim (one DLL per process: the firmware keeps static state)."""

    def __init__(self, dll):
        self.fw = fwsim.FirmwareSim(dll)
        self.path = o1.Path8()
        self._cmd = {}

    def defaults(self, ctrl):
        try:
            return s3.defaults(self.fw, ctrl)
        except s3.S03Stop as e:
            raise FeasStop(str(e)) from None

    def cmd_max(self, sch):
        key = (sch.v, sch.w0)
        if key not in self._cmd:
            self._cmd[key] = G.fig8_cmd_max(sch)
        return self._cmd[key]

    def level(self, ctrl, g, v, mu, keep_log=False):
        """(level result, log or None). A DLL / numeric failure is a run_error level."""
        try:
            kw = s3.sim_kwargs(self.fw, ctrl, g)
        except s3.S03Stop as e:
            raise FeasStop(str(e)) from None
        try:
            sch = o1.Schedule(v)
            p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=mu))
            log = fwsim.simulate(self.fw, sch.setpoint, sch.Tcmd + o1.T_LATE, p, seed=s3.SEED, mode=ctrl,
                                 imu=dict(s3.IMU), **kw)
            for k in ('pose', 'v_true', 'duty', 'pwm'):
                if not np.all(np.isfinite(log[k])):
                    raise FloatingPointError(f'non-finite {k} in the log')
            out, _, frames = E.evaluate_fig8(log, self.path, sch, lsum=p.lx + p.ly)
            val = fwsim.run_validity(log)
            vmax, wzmax = self.cmd_max(sch)
            gt = G.gate(frames, out['completed'], v_cmd_max=vmax, wz_cmd_max=wzmax, u_t=0.0,
                        decide=G.DECIDE_ALL, status=out['status'])
        except FeasStop:
            raise
        except Exception as exc:                 # noqa: BLE001 (unstable gains end as a run_error level)
            return error_level(v, mu, type(exc).__name__ + ': ' + str(exc)[:300]), None
        rb = val.get('run_bad')
        r = level_from_gate(v, mu, gt, out['status'], int(rb) if _finite(rb) else float('nan'))
        r.update(log_sha=log_sha(log), outcome_decide=gt['outcome_decide'])
        return r, (log if keep_log else None)

    def candidate(self, ctrl, g, levels, mu):
        """Level results in level order and V (design 4.3)."""
        rows = [self.level(ctrl, g, v, mu)[0] for v in levels]
        return rows, float(sum(r['V_level'] for r in rows))


# ── one GA search (runs in its own process) ──────────────────────────────────────────

def ga_search(runner, ctrl, levels, mu, box, pop=POP, n_gen=N_GEN, seed=GA_SEED):
    """The whole budget, no early stop. Returns (history [rows], witness dict or None, n_evaluated)."""
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem

    xl = np.array([d['xl'] for d in box])
    xu = np.array([d['xu'] for d in box])
    state = dict(gen=0, n=0)
    history, witness = [], [None]

    class Gains(Problem):
        def __init__(self):
            super().__init__(n_var=len(box), n_obj=1, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F = []
            for k, x in enumerate(X):
                g = decode(box, x)
                rows, V = runner.candidate(ctrl, g, levels, mu)
                eid = state['n']
                state['n'] += 1
                for r in rows:
                    history.append(dict(controller=ctrl, eval_id=eid, gen=state['gen'], idx=k, V=V,
                                        **{f'gain_{n}': g[n] for n in g}, **r))
                if V == 0.0 and witness[0] is None:
                    witness[0] = dict(eval_id=eid, gen=state['gen'], idx=k, gains=g, levels=rows)
                F.append([V])
            out['F'] = np.array(F)

    problem = Gains()
    alg = GA(pop_size=pop, sampling=first_population(box, seed, pop))
    alg.setup(problem, termination=('n_gen', n_gen), seed=seed, verbose=False)
    while alg.has_next():
        state['gen'] += 1
        infills = alg.ask()
        alg.evaluator.eval(problem, infills)
        alg.tell(infills=infills)
    return history, witness[0], state['n']


def job_main(job_path):
    """--job: one search in this process; result JSON next to the job file (under sealed/)."""
    with open(job_path, encoding='utf-8') as f:
        job = json.load(f)
    if job.get('a_r') is not None:
        o1.A_R = float(job['a_r'])
    t0 = time.time()
    runner = Runner(job['dll'])
    kind = job['kind']
    if kind == 'search':
        hist, wit, n = ga_search(runner, job['controller'], tuple(job['levels']), job['mu'], job['box'],
                                 job['pop'], job['n_gen'], job['seed'])
        res = dict(history=hist, witness=wit, n_evaluated=n)
    elif kind == 'levels':                 # re-run given gains at given levels, mu (recheck, mu low / high)
        res = dict(levels=[])
        for g, v, mu in job['runs']:
            r, log = runner.level(job['controller'], g, v, mu, keep_log=bool(job.get('log_dir')))
            if log is not None:
                s3.save_log(job['log_dir'], f"{job['controller']}_{job['tag']}_v{v}_mu{mu}", log)
            res['levels'].append(r)
    else:
        raise ValueError(kind)
    res['elapsed_s'] = time.time() - t0
    with open(job_path + '.out.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, default=_json_default)


def _json_default(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x).__name__)


def run_jobs(jobs, work, n_parallel, say=None):
    """Run job dicts as child processes (at most n_parallel at once). stdout / stderr of each child go to
    a file under work (sealed/). Returns the result dicts in job order. A failing child: FeasStop with the
    exception class only (no sealed text on the console)."""
    os.makedirs(work, exist_ok=True)
    paths = []
    for i, job in enumerate(jobs):
        p = os.path.join(work, f"job_{i:02d}_{job['name']}.json")
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(job, f, default=_json_default)
        paths.append(p)
    pending, running, k = list(range(len(jobs))), {}, 0
    while pending or running:
        while pending and len(running) < max(1, n_parallel):
            i = pending.pop(0)
            k += 1
            if say:
                say(f'job {k}/{len(jobs)} started')
            logf = open(paths[i] + '.log', 'w', encoding='utf-8')
            running[i] = (subprocess.Popen([sys.executable, os.path.abspath(__file__), '--job', paths[i]],
                                           stdout=logf, stderr=subprocess.STDOUT), logf)
        time.sleep(0.2)
        for i in list(running):
            proc, logf = running[i]
            if proc.poll() is not None:
                logf.close()
                del running[i]
                if proc.returncode != 0:
                    for j in running:
                        running[j][0].kill()
                    raise FeasStop(f'child job {i} failed (exit {proc.returncode}, {_exc_class(paths[i] + ".log")})')
    out = []
    for p in paths:
        with open(p + '.out.json', encoding='utf-8') as f:
            out.append(json.load(f))
    return out


def _exc_class(log_path):
    """The exception class of a child's last traceback line, nothing else."""
    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            last = [l for l in f.read().splitlines() if l.strip()][-1]
        name = last.split(':', 1)[0].strip()
        return name if name.replace('.', '').replace('_', '').isalnum() else 'unknown'
    except (OSError, IndexError):
        return 'unknown'


# ── labels ───────────────────────────────────────────────────────────────────────────

def restrict_V(levels_rows, subset):
    """V of a candidate on the levels of a subset, from its stored per-level results."""
    rows = [r for r in levels_rows if float(r['v']) in set(subset)]
    if len(rows) != len(subset):
        raise ValueError(f'levels {subset} not all in the stored results')
    return float(sum(float(r['V_level']) for r in rows))


def labels(witnesses):
    """witnesses: {(ctrl, set_name): witness dict or None} (after the determinism check).
    Returns (label rows, inherit rows, level set name or EMPTY_SET)."""
    rows, inh = [], []
    for ctrl in CONTROLLERS:
        for j, (sn, lv) in enumerate(zip(SET_NAMES, LEVEL_SETS)):
            own = witnesses.get((ctrl, sn)) is not None
            via = None
            if not own:
                for bigger in SET_NAMES[:j]:
                    w = witnesses.get((ctrl, bigger))
                    if w is not None:
                        vs = restrict_V(w['levels'], lv)
                        ok = vs == 0.0 and all(gate_pass(r) for r in w['levels'] if float(r['v']) in set(lv))
                        inh.append(dict(controller=ctrl, set=sn, from_set=bigger, V_subset=vs, gate_pass=ok))
                        if ok and via is None:
                            via = bigger
            rows.append(dict(controller=ctrl, set=sn, levels=' '.join(str(v) for v in lv),
                             own_search=FOUND if own else NOT_FOUND,
                             label=FOUND if (own or via) else NOT_FOUND))
    chosen = EMPTY_SET
    for sn in SET_NAMES:
        if all(r['label'] == FOUND for r in rows if r['set'] == sn):
            chosen = sn
            break
    return rows, inh, chosen


# ── outputs ──────────────────────────────────────────────────────────────────────────

class Console:
    """Every console line also goes to console.txt (public): the only things printed (design 5)."""

    def __init__(self, out):
        self.f = open(os.path.join(out, 'console.txt'), 'a', encoding='utf-8')

    def __call__(self, text):
        print(text, flush=True)
        self.f.write(text + '\n')
        self.f.flush()


def write_csv(path, rows, cols=None):
    s3.write_csv(path, rows, cols)


def sealed_manifest(sealed):
    files = {}
    for root, _, names in os.walk(sealed):
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, sealed).replace(os.sep, '/')
            if rel != 'manifest.json':
                files[rel] = s3.sha256_file(p)
    with open(os.path.join(sealed, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(files, f, indent=1, sort_keys=True)
    return files


def new_out(out):
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f'{out} already holds data: use a new folder.')
    os.makedirs(os.path.join(out, 'sealed'), exist_ok=True)


def source_state(dll):
    with open(os.path.join(os.path.dirname(os.path.abspath(dll)), 'fwsim_build.json'), encoding='utf-8') as f:
        build = json.load(f)
    return dict(hil_framework=s3.git_state(os.path.join(HERE, '..')), dll_sha256=s3.sha256_file(dll),
                build=build, script=s3.file_git_state(os.path.abspath(__file__)))


def write_meta(out, kind, a, src, extra, sealed_files):
    lines = [f'spec: "S05 C12, O1 v11.12 section 3.9 item 5; gate dossier v3.2 section 4; design '
             f'03_CODE/docs/2026-11-14_Thiet_ke_tim_kha_thi_C12_v1.md"',
             f'kind: {kind}', f'script_version: {SCRIPT_VERSION}', f'date: {time.strftime("%Y-%m-%d")}',
             'setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware',
             'operator: Claude (phien CODE_FEAS)',
             f'tool_check: {"true" if getattr(a, "tool_check", False) else "false"}',
             f'firmware: {json.dumps(src["build"])}', f'dll_sha256: {src["dll_sha256"]}',
             f'hil_framework: {json.dumps(src["hil_framework"])}', f'script: {json.dumps(src["script"])}',
             f'plant: {fwsim.plant_meta(fwsim.plant_params(argparse.Namespace(ground="traction", mu=MU_MID)))}',
             f'imu: {json.dumps(s3.IMU)}', f'seed_plant_imu: {s3.SEED}',
             f'a_r: {o1.A_R}', f'o1_constants: {json.dumps(dict(D=o1.D, delta=o1.DELTA, v_d=o1.V_D, t_d=o1.T_D, t_late=o1.T_LATE))}',
             f'gate: {json.dumps(dict(version=G.GATE_VERSION, thresholds=G.THRESHOLDS, u_t=0.0, decide=list(G.DECIDE_ALL), eval=E.EVAL_VERSION))}',
             f'libraries: {json.dumps(fwsim.lib_versions())}']
    for k, v in extra.items():
        lines.append(f'{k}: {json.dumps(v, default=_json_default)}')
    lines.append(f'sealed_sha256: {json.dumps(sealed_files, sort_keys=True)}   # O1 3.9 (5); nobody opens sealed/ before A ends')
    with open(os.path.join(out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


INTERPRETATIONS = [
    'c = 0 needs n_undef = 0 for G2-G5 ("moi q huu han" read as every frame finite, as gate() needs for dat)',
    'a run_error level counts c = 1 and h = 1 each; the GA goes on (unstable gains are a search result)',
    'the same seed 20261113 and the same 23 coded points for every controller and level set',
    'witness = first V = 0 in evaluation order (generation, index); the 6 searches run in parallel processes',
    'recheck: a controller keeps its label when a witness of the C8 set or of a larger set passes on C8',
]


def cmd_search(a):
    new_out(a.out)
    say = Console(a.out)
    sealed = os.path.join(a.out, 'sealed')
    tool = a.tool_check
    if a.a_r is not None:
        o1.A_R = float(a.a_r)
    src = source_state(a.dll)
    if src['build'].get('esp32', {}).get('dirty', True) and not tool:
        raise FeasStop('fwsim.dll built from a dirty firmware tree: not a main run')
    mu = MU_TOOL if tool else MU_MID
    pop, n_gen = (TOOL_POP, TOOL_GEN) if tool else (POP, N_GEN)
    say(f'o1_feas_search search: {"TOOL-CHECK, " if tool else ""}mu {mu}, a_r {o1.A_R}, '
        f'firmware {src["build"].get("esp32", {}).get("commit")}, out {a.out}')
    t0 = time.time()
    runner = Runner(a.dll)
    boxes = {c: box_of(c, runner.defaults(c)) for c in CONTROLLERS}
    write_csv(os.path.join(a.out, 'box.csv'), [d for c in CONTROLLERS for d in boxes[c]])
    # public: default gains at the middle mu, every level (Liam 14/11 Q1: states and q)
    dflt, t_runs = [], []
    for c in CONTROLLERS:
        g = {d['name']: d['default'] for d in boxes[c]}
        for v in LEVEL_SETS[0]:
            t1 = time.time()
            r, log = runner.level(c, g, v, mu, keep_log=True)
            t_runs.append(time.time() - t1)
            if log is not None:
                s3.save_log(os.path.join(a.out, 'logs_default'), f'{c}_v{v}', log)
            dflt.append(dict(controller=c, **r))
    write_csv(os.path.join(a.out, 'default_gains.csv'), dflt)
    jobs = [dict(name=f'{c}_{sn}', kind='search', dll=os.path.abspath(a.dll), controller=c, levels=list(lv),
                 mu=mu, box=boxes[c], pop=pop, n_gen=n_gen, seed=GA_SEED, a_r=o1.A_R)
            for c in CONTROLLERS for sn, lv in zip(SET_NAMES, LEVEL_SETS)]
    res = run_jobs(jobs, os.path.join(sealed, 'jobs'), a.jobs, say)
    # determinism + independent gate check of every witness, in new processes (after all searches)
    wit, history, notes = {}, [], dict(jobs={})
    rechecks = []
    for job, r in zip(jobs, res):
        key = (job['controller'], SET_NAMES[LEVEL_SETS.index(tuple(job['levels']))])
        history += [dict(set=key[1], **h) for h in r['history']]
        notes['jobs'][job['name']] = dict(elapsed_s=r['elapsed_s'], n_evaluated=r['n_evaluated'])
        wit[key] = r['witness']
        if r['witness'] is not None:
            rechecks.append((key, r['witness']))
    rjobs = [dict(name=f'recheck_{k[0]}_{k[1]}', kind='levels', dll=os.path.abspath(a.dll), controller=k[0],
                  tag=k[1], runs=[(w['gains'], float(lv['v']), mu) for lv in w['levels']],
                  log_dir=os.path.join(sealed, 'logs'), a_r=o1.A_R) for k, w in rechecks]
    rres = run_jobs(rjobs, os.path.join(sealed, 'jobs'), a.jobs) if rjobs else []
    rc_rows, bad = [], []
    for (k, w), rr in zip(rechecks, rres):
        for first, again in zip(w['levels'], rr['levels']):
            same = first['log_sha'] != '' and first['log_sha'] == again['log_sha']
            ok = same and gate_pass(again)
            rc_rows.append(dict(controller=k[0], set=k[1], v=first['v'], same_log=same, gate_pass=gate_pass(again),
                                ok=ok))
            if not ok:
                bad.append(k)
    write_csv(os.path.join(sealed, 'recheck.csv'), rc_rows or [dict(controller='', set='', v='', same_log='',
                                                                    gate_pass='', ok='')])
    if bad:
        notes['stop'] = 'determinism / independent gate check failed'
        notes['stop_sets'] = sorted(set(bad))
        for k in set(bad):                               # SC-13: no witness written when the re-run differs
            wit[k] = None
        _close_sealed(a, sealed, history, wit, notes, t0)
        raise FeasStop('determinism check failed: no witness written (S03 3.4); report to CODE and Liam')
    lab, inh, chosen = labels(wit)
    write_csv(os.path.join(sealed, 'inherit.csv'), inh or [dict(controller='', set='', from_set='', V_subset='',
                                                                gate_pass='')])
    # sealed: witnesses at low / high mu (dossier 4 (7)), report only
    mus = (MU_TOOL,) if tool else (MU_LO, MU_HI)           # tool-check: the fwsim assumption only
    mjobs = [dict(name=f'mu_{k[0]}_{k[1]}', kind='levels', dll=os.path.abspath(a.dll), controller=k[0], tag=k[1],
                  runs=[(w['gains'], float(lv['v']), m) for m in mus for lv in w['levels']],
                  log_dir=None, a_r=o1.A_R) for k, w in rechecks]
    mres = run_jobs(mjobs, os.path.join(sealed, 'jobs'), a.jobs) if mjobs else []
    mu_rows = []
    for (k, _), mr in zip(rechecks, mres):
        for m in mus:
            lv = [r for r in mr['levels'] if r['mu'] == m]
            mu_rows.append(dict(controller=k[0], set=k[1], mu=m, gate_pass_all=all(gate_pass(r) for r in lv),
                                V=sum(r['V_level'] for r in lv)))
    write_csv(os.path.join(sealed, 'mu_lohi.csv'), mu_rows or [dict(controller='', set='', mu='', gate_pass_all='',
                                                                    V='')])
    write_csv(os.path.join(a.out, 'labels.csv'),
              lab + [dict(controller='both', set='feasibility_level_set', levels=chosen, own_search='', label='')])
    files = _close_sealed(a, sealed, history, wit, notes, t0)
    write_meta(a.out, 'search', a, src, dict(
        mu=mu, level_sets=dict(zip(SET_NAMES, LEVEL_SETS)), ga=dict(algorithm='pymoo GA (single objective)',
        pop=pop, n_gen=n_gen, seed=GA_SEED, first_population='default + 23 uniform in the coded box'),
        boxes=boxes, interpretations=INTERPRETATIONS,
        files=['box.csv', 'default_gains.csv', 'labels.csv', 'logs_default/', 'console.txt', 'sealed/']), files)
    if tool:
        say(f'tool-check done: mean time of one default-gain run {np.mean(t_runs):.2f} s '
            f'({len(t_runs)} runs, G4 scoring included)')
        return
    say('labels (O1 3.9 (5); tim_thay = feasible on this fwsim configuration, not on the car):')
    for r in lab:
        say(f"  {r['controller']:4s} {r['set']} {{{r['levels']}}}: {r['label']}")
    say(f'feasibility level set: {chosen}' + ('' if chosen == EMPTY_SET else f' {{{" ".join(map(str, LEVEL_SETS[SET_NAMES.index(chosen)]))}}}'))
    say('default gains at mu %.3f (G1 state; G2-G5 state and q; G5 in deg):' % mu)
    for r in dflt:
        say(f"  {r['controller']:4s} v {r['v']}: G1 {r['g1_state']}, " + ', '.join(
            f"{k.upper()} {r[f'{k}_state']} {_q(r[f'{k}_q'])}" for k in G.MEASURED) + f", status {r['status']}")


def _q(x):
    return f'{float(x):.4f}' if _finite(x) else 'nan'


def _close_sealed(a, sealed, history, wit, notes, t0):
    write_csv(os.path.join(sealed, 'history.csv'), history or [dict(controller='')])
    wrows = []
    for c in CONTROLLERS:
        for sn in SET_NAMES:
            w = wit.get((c, sn))
            wrows.append(dict(controller=c, set=sn, found=w is not None,
                              eval_id=w['eval_id'] if w else '', gen=w['gen'] if w else '',
                              **{f'gain_{n}': (w['gains'][n] if w else '') for n in s3.FACTORS[c]}))
    cols = ['controller', 'set', 'found', 'eval_id', 'gen'] + sorted({k for r in wrows for k in r if k.startswith('gain_')})
    write_csv(os.path.join(sealed, 'witness.csv'), wrows, cols)
    with open(os.path.join(sealed, 'witness.json'), 'w', encoding='utf-8') as f:
        json.dump({f'{c}|{sn}': wit.get((c, sn)) for c in CONTROLLERS for sn in SET_NAMES}, f, default=_json_default)
    notes['elapsed_s'] = time.time() - t0
    with open(os.path.join(sealed, 'notes.json'), 'w', encoding='utf-8') as f:
        json.dump(notes, f, default=_json_default, indent=1)
    for n in os.listdir(os.path.join(sealed, 'jobs')) if os.path.isdir(os.path.join(sealed, 'jobs')) else []:
        if n.endswith('.out.json'):                      # job results are in history / witness now
            os.remove(os.path.join(sealed, 'jobs', n))
    return sealed_manifest(sealed)


def read_witnesses(run):
    with open(os.path.join(run, 'sealed', 'witness.json'), encoding='utf-8') as f:
        raw = json.load(f)
    return {tuple(k.split('|')): v for k, v in raw.items()}


def cmd_recheck(a):
    """After C8 (O1 3.9 (5), R2V-11): each controller's witnesses of the C8 set or a larger set, on the C8
    configuration (a_r, binary); console: 'con_dat' / 'khong_con_dat' per controller."""
    levels = tuple(float(x) for x in a.levels.split(','))
    if levels not in LEVEL_SETS:
        sys.exit(f'--levels {a.levels}: one of {LEVEL_SETS}')
    new_out(a.out)
    say = Console(a.out)
    sealed = os.path.join(a.out, 'sealed')
    o1.A_R = float(a.a_r)
    src = source_state(a.dll)
    wit = read_witnesses(getattr(a, 'from'))
    j = LEVEL_SETS.index(levels)
    say(f'o1_feas_search recheck: levels {levels}, a_r {o1.A_R}, firmware {src["build"].get("esp32", {}).get("commit")}')
    t0 = time.time()
    jobs, keys = [], []
    for c in CONTROLLERS:
        for sn in SET_NAMES[:j + 1][::-1]:               # own set first, then larger
            w = wit.get((c, sn))
            if w is not None:
                keys.append((c, sn))
                jobs.append(dict(name=f'recheck_{c}_{sn}', kind='levels', dll=os.path.abspath(a.dll), controller=c,
                                 tag=sn, runs=[(w['gains'], v, MU_MID) for v in levels],
                                 log_dir=os.path.join(sealed, 'logs'), a_r=o1.A_R))
    res = run_jobs(jobs, os.path.join(sealed, 'jobs'), a.jobs) if jobs else []
    rows = []
    for (c, sn), r in zip(keys, res):
        for lv in r['levels']:
            rows.append(dict(controller=c, witness_of=sn, **lv, gate_pass=gate_pass(lv)))
    write_csv(os.path.join(sealed, 'recheck_c8.csv'), rows or [dict(controller='')])
    out = []
    for c in CONTROLLERS:
        ok = any(all(r['gate_pass'] for r in rows if r['controller'] == c and r['witness_of'] == sn)
                 and any(r['controller'] == c and r['witness_of'] == sn for r in rows) for sn in SET_NAMES)
        out.append(dict(controller=c, levels=' '.join(map(str, levels)), a_r=o1.A_R,
                        result='con_dat' if ok else 'khong_con_dat'))
    write_csv(os.path.join(a.out, 'labels_recheck.csv'), out)
    with open(os.path.join(sealed, 'notes.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(elapsed_s=time.time() - t0, source=getattr(a, 'from')), f)
    os.makedirs(os.path.join(sealed, 'jobs'), exist_ok=True)
    for n in os.listdir(os.path.join(sealed, 'jobs')):
        if n.endswith('.out.json'):
            os.remove(os.path.join(sealed, 'jobs', n))
    files = sealed_manifest(sealed)
    write_meta(a.out, 'recheck', a, src, dict(source_run=getattr(a, 'from'), levels=list(levels),
                                              interpretations=INTERPRETATIONS), files)
    for r in out:
        say(f"  {r['controller']:4s} {{{r['levels']}}} a_r {r['a_r']}: {r['result']}")


def cmd_inbox(a):
    """After the S03 main run: is each witness inside the [L, H] of S03 levels.csv (B's domain)? sealed only."""
    new_out(a.out)
    say = Console(a.out)
    sealed = os.path.join(a.out, 'sealed')
    wit = read_witnesses(getattr(a, 'from'))
    with open(os.path.join(a.s03, 'levels.csv'), encoding='utf-8') as f:
        lv = {(r['controller'], r['factor']): (float(r['L']), float(r['H'])) for r in csv.DictReader(f)}
    rows = []
    for (c, sn), w in sorted(wit.items()):
        if w is None:
            continue
        for n, g in w['gains'].items():
            L, H = lv[(c, n)]
            tol = s3.BOUND_RTOL * max(abs(L), abs(H), 1.0)
            rows.append(dict(controller=c, set=sn, factor=n, inside=bool(L - tol <= g <= H + tol)))
    write_csv(os.path.join(sealed, 'inbox.csv'), rows or [dict(controller='', set='', factor='', inside='')])
    files = sealed_manifest(sealed)
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f'kind: inbox\nscript_version: {SCRIPT_VERSION}\ndate: {time.strftime("%Y-%m-%d")}\n'
                f'source_run: "{getattr(a, "from")}"\ns03_run: "{a.s03}"\n'
                f's03_levels_sha256: {s3.sha256_file(os.path.join(a.s03, "levels.csv"))}\n'
                f'sealed_sha256: {json.dumps(files, sort_keys=True)}\n')
    say('inbox written (sealed)')


def main(argv=None):
    if argv and argv[0] == '--job' or (argv is None and len(sys.argv) > 2 and sys.argv[1] == '--job'):
        job_main((argv or sys.argv[1:])[1])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search')
    s.add_argument('--out', required=True)
    s.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    s.add_argument('--jobs', type=int, default=6)
    s.add_argument('--a-r', type=float, default=None, help='a_r [m/s^2] (default o1_feasibility.A_R; C8 re-search)')
    s.add_argument('--tool-check', action='store_true')
    r = sub.add_parser('recheck')
    r.add_argument('--from', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--levels', required=True)
    r.add_argument('--a-r', type=float, required=True)
    r.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    r.add_argument('--jobs', type=int, default=6)
    i = sub.add_parser('inbox')
    i.add_argument('--from', required=True)
    i.add_argument('--out', required=True)
    i.add_argument('--s03', required=True)
    a = ap.parse_args(argv)
    try:
        dict(search=cmd_search, recheck=cmd_recheck, inbox=cmd_inbox)[a.cmd](a)
    except FeasStop as e:
        print(f'stopped: {e}', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
