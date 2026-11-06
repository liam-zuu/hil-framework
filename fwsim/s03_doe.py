"""
s03_doe.py: spec S03 v6 (CLAUDE_THESIS 06_SPECS/2026-11-05_Spec_S03_DOE_chon_gain.md),
DOE of the 7 gains per controller on fwsim, figure 8 of O1 v7 (o1_feasibility.Schedule,
evaluate_v7 on the true plant pose).

Steps, as the spec orders them (section numbers of S03 v6):
  3.1  one-factor sweep per gain (others at firmware default), middle mu, every speed:
       grid = 11 log-spaced points in [a, b] + the default (if not already there) + 0 when
       the temporary lower bound (tune_ga.BOUNDS) is 0; a = default/10 if that bound is 0,
       else min(default/10, bound); b = max(10 default, upper bound); omega_o <= 48.
       Levels L, H: ends of the longest run of adjacent grid points holding the default in
       which every point completes (y1 'true') at every speed in the gate. Stops (3.1 (i))
       when the default does not complete at a gate speed, and before the main design (ii)
       when L = H. Scale: log when L > 0 and H/L >= 4 (centre sqrt(L H)), else linear.
  3.2  full factorial 2^7 at codes +-1 + one centre point, x mu levels x speeds; when more
       than 64/128 corners do not complete (y1 'false') at the middle mu at one speed, the
       inner box +-0.5 is run once (128 more points, all conditions).
  3.4  determinism: the centre and 3 corners drawn with a recorded seed are run again in a
       NEW process (DLL loaded again); every log array compared bit for bit in memory, and
       every runs.csv field.
  4    responses per run: evaluate_v7 (y1 true|false|inconclusive, status ok|track_err|
       run_error, e_perp_max, censored at 0.5 m, e_perp_global_max, tv_duty, sat_frac,
       sat_any_frac, report-only measures). A run_error is run again once in a new process
       for diagnosis; it never replaces the first run by itself (B8b-20): the script stops.
  5    effects (mean at + minus mean at -) of every term (7 main, 21 two-factor, higher
       orders flagged by 'order') on Y2 (as measured and censored), Y3, Y4 per speed and
       per mu, 'worst' (max over mu) and 'range' (max - min over mu); significant against
       the thresholds of 5.2 (Y3: 20 % of Y3 of the default-gain run at the middle mu, same
       speed, from the sweep); y2_versions.csv: whether the two Y2 versions give the same
       significant set and signs (5.1); curvature.csv (5.4); fail_counts.csv (5.3).
Outputs (section 7) in --out: sweep.csv, levels.csv, design.csv, determinism_check.csv, meta.yaml;
in --out/sealed/ (L56): runs.csv, effects.csv, fail_counts.csv, curvature.csv, y2_versions.csv,
design_inner.csv, notes.json, logs/.

L56 / S03 v6 section 1: the main design results (runs.csv, effects.csv and the tables of
section 5) are NOT looked at before campaign A ends. This script prints only progress
counts for the main design (no responses, no effects); the files are written and the
meta.yaml records their SHA-256. The sweep (section 3.1) is a source of campaign A's domain
and is printed.

  python fwsim/s03_doe.py --out 05_DATA/results/S03/<date>_runNN --mu 0.5 --trial
  python fwsim/s03_doe.py --out ... --mu LOW MID HIGH --mu-source "MU_FLOOR/<run>: min, mean, max"

--trial (S03 v6 section 2.2, 11.3): mu 0.5, the fwsim assumption, to check the tool and its
running time; never main numbers. Interpretations of this script, not written in the spec,
are in meta.yaml 'interpretations'.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim                                            # noqa: E402
import o1_feasibility as o1                             # noqa: E402
import tune_ga                                          # noqa: E402

CONTROLLERS = ('PID', 'ADRC')
SPEEDS = (0.2, 0.5, 0.8)
FACTORS = {m: list(tune_ga.VARS_BY_MODE[m]) for m in CONTROLLERS}       # X1..X7 (S03 section 2.1)
K_PSI_DEFAULT = 1.0           # heading.h HEADING_PARAMS_DEFAULT (no getter in fw_sim.c); AGV b0047ac line 25
OMEGA_O_MAX = 48.0            # S03 section 2.1: convention from the adrc.c comment, not a stability bound
N_GRID = 11
IMU = dict(rate_hz=20.0, boot_reset=-1.0)     # BNO085 20 Hz, noise-free (S03 section 3.3)
SEED = 1                                      # plant seed (no random input; recorded)
DET_SEED = 20261106                           # S03 section 3.4: seed that draws the 3 corners, fixed before running
THRESH = {'e_perp_max': 0.01, 'e_perp_max_cens': 0.01, 'sat_frac': 0.01}   # S03 section 5.2; tv_duty: 20 %
Y3_REL = 0.20
RESPONSES = ('e_perp_max', 'e_perp_max_cens', 'tv_duty', 'sat_frac')
RUN_COLS = ['controller', 'point_id', 'box', 'mu', 'v', 'status', 'completed', 'e_perp_max', 'e_perp_max_cens',
            'e_perp_global_max', 'tv_duty', 'sat_frac', 'sat_any_frac', 'e_theta_lane_max_deg',
            'e_theta_cmd_max_deg', 'ds_absmax', 'e_trk_max', 't_end', 'e_s_absmax', 'track_err_frames',
            'cond2_5_ever', 'pos_reset_mid']


class S03Stop(Exception):
    """A stop the spec orders (3.1 (i), (ii); 3.4; B8b-20): report, do not go on."""


# ── gains and one run ────────────────────────────────────────────────────────────────

def defaults(fw, ctrl):
    """Firmware default of the 7 factors, read back from the DLL after reset (k_psi: heading.h)."""
    fw.reset()
    fw.set_mode(ctrl)
    pos = fwsim.pos_meta(fw)
    if pos is None:
        raise S03Stop('fwsim.dll has no pos_ctrl.c: S03 needs the campaign firmware (S03 section 3.3)')
    if ctrl == 'PID':
        p = [fw.get_pid(i) for i in range(3)]
        inner = [p[0]['kp'], p[0]['ki'], p[2]['kp'], p[2]['ki']]
    else:
        a = [fw.get_adrc(i) for i in range(3)]
        inner = [a[0]['omega_c'], a[0]['omega_o'], a[2]['omega_c'], a[2]['omega_o']]
    return dict(zip(FACTORS[ctrl], inner + [pos['kp'], pos['ki'], K_PSI_DEFAULT]))


def sim_kwargs(fw, ctrl, g):
    """simulate() keywords for the 7 gains g (dict by factor name); every fixed parameter stays
    as the firmware holds it after reset (kd = 0, limits, b0, out_limit; S03 section 2.1)."""
    fw.reset()
    fw.set_mode(ctrl)
    f = FACTORS[ctrl]
    if ctrl == 'PID':
        base = [fw.get_pid(i) for i in range(3)]
        if any(b['kd'] != 0.0 for b in base):
            raise S03Stop('PID kd is not 0 in this firmware (S03 section 2.1 fixes kd = 0)')
        xy, wz = (g[f[0]], g[f[1]]), (g[f[2]], g[f[3]])
        kw = dict(pid={i: (*(wz if i == 2 else xy), 0.0, base[i]['int_limit'], base[i]['out_limit'])
                       for i in range(3)})
    else:
        base = [fw.get_adrc(i) for i in range(3)]
        xy, wz = (g[f[0]], g[f[1]]), (g[f[2]], g[f[3]])
        kw = dict(adrc={i: (*(wz if i == 2 else xy), base[i]['b0'], base[i]['out_limit']) for i in range(3)})
    kw['fw_params'] = {'pos_kp': g['pos_kp'], 'pos_ki': g['pos_ki'], 'k_psi': g['k_psi']}
    return kw


def run_one(fw, path, ctrl, g, v, mu):
    """One run: (row of section 4 / RUN_COLS, log dict of arrays). Raises on a DLL / NaN error."""
    sch = o1.Schedule(v)
    p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=mu))
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, p, seed=SEED, mode=ctrl, imu=dict(IMU),
                         **sim_kwargs(fw, ctrl, g))
    for k in ('pose', 'v_true', 'duty', 'pwm'):
        if not np.all(np.isfinite(log[k])):
            raise FloatingPointError(f'non-finite {k} in the log')
    res, _ = o1.evaluate_v7(log, path, sch, lsum=p.lx + p.ly)
    val = fwsim.run_validity(log)
    if val['run_bad'] == 1:
        raise RuntimeError(f"pose reset mid-run in fwsim ({val['run_bad_reason']})")
    row = {k: res[k] for k in RUN_COLS[5:] if k in res}          # responses only (no controller..v keys)
    row.update(pos_reset_mid=val['pos_reset_mid'])
    return row, log


def safe_run(fw, path, ctrl, g, v, mu):
    """run_one, or a 'run_error' row (S03 section 4) with the reason."""
    try:
        return run_one(fw, path, ctrl, g, v, mu)
    except S03Stop:
        raise
    except Exception as exc:                 # noqa: BLE001 (any DLL / numeric failure is a run_error)
        return dict(status='run_error', completed='', error=f'{type(exc).__name__}: {exc}'), None


# ── 3.1 sweep and levels ────────────────────────────────────────────────────────────

def sweep_grid(name, default, lo, hi):
    """Grid of S03 section 3.1 for one factor (sorted, unique, the default always in it)."""
    a = default / 10.0 if lo == 0.0 else min(default / 10.0, lo)
    b = max(default * 10.0, hi)
    if name.startswith('omega_o'):
        b = min(b, OMEGA_O_MAX)
    pts = list(np.geomspace(a, b, N_GRID))
    if not any(math.isclose(x, default, rel_tol=1e-9) for x in pts):
        pts.append(default)
    if lo == 0.0:
        pts.append(0.0)
    pts = sorted(set(float(x) for x in pts))
    return [default if math.isclose(x, default, rel_tol=1e-9) else x for x in pts]


def levels_from_sweep(grid, default, ok):
    """L, H of section 3.1: ends of the longest run of adjacent grid points holding the default in
    which ok[i] (completes at every gate speed). Raises S03Stop for case (i)."""
    i0 = grid.index(default)
    if not ok[i0]:
        raise S03Stop('default gains do not complete at a gate speed (S03 3.1 (i))')
    lo = hi = i0
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    while hi + 1 < len(grid) and ok[hi + 1]:
        hi += 1
    return grid[lo], grid[hi]


def scale_of(L, H):
    """('log', centre) when L > 0 and H/L >= 4, else ('lin', centre) (section 3.1)."""
    if L > 0 and H / L >= 4.0:
        return 'log', math.sqrt(L * H)
    return 'lin', (L + H) / 2.0


def decode(code, L, H, scale):
    """Real value of a coded level in [-1, 1] (section 3.1 coding: log or linear)."""
    if scale == 'log':
        return math.sqrt(L * H) * (H / L) ** (code / 2.0)
    return (L + H) / 2.0 + code * (H - L) / 2.0


# ── 3.2 design ───────────────────────────────────────────────────────────────────────

def design_points(half=1.0, start_id=1):
    """2^7 full factorial in standard order at codes +-half: list of (point_id, codes)."""
    pts = []
    for i, bits in enumerate(itertools.product((-1, 1), repeat=7)):
        pts.append((start_id + i, tuple(half * b for b in reversed(bits))))   # X1 changes fastest
    return pts


def term_names(k=7):
    """Every non-empty subset of X1..Xk as (name, index tuple), by order then index."""
    out = []
    for r in range(1, k + 1):
        for comb in itertools.combinations(range(k), r):
            out.append(('X' + ':X'.join(str(i + 1) for i in comb), comb))
    return out


def effects(codes, y):
    """Effect of every term: mean(y | sign of the code product > 0) - mean(y | < 0) (section 5.1;
    also for the inner box at +-0.5, S3-R-3). codes: (n, 7), y: (n,)."""
    codes, y = np.asarray(codes, dtype=float), np.asarray(y, dtype=float)
    sg = np.sign(codes)
    out = {}
    for name, comb in term_names(codes.shape[1]):
        s = np.prod(sg[:, comb], axis=1)
        out[name] = float(np.mean(y[s > 0]) - np.mean(y[s < 0]))
    return out


# ── determinism (3.4) ────────────────────────────────────────────────────────────────

def log_digest(log):
    """{key: (dtype, shape, sha256 of the raw bytes)}: equal digests <=> equal arrays bit for bit."""
    return {k: (str(np.asarray(v).dtype), list(np.asarray(v).shape),
                hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k, v in sorted(log.items())}


def row_text(row):
    """runs.csv fields as exact text (repr keeps every bit of a float)."""
    return {k: repr(row.get(k)) for k in RUN_COLS if k in row}


def child_main(job_path):
    """--child: run the jobs of a pickle in this new process, write digests + rows back."""
    with open(job_path, 'rb') as f:
        job = pickle.load(f)
    fw = fwsim.FirmwareSim(job['dll'])
    path = o1.Path8()
    out = []
    for ctrl, g, v, mu in job['runs']:
        row, log = safe_run(fw, path, ctrl, g, v, mu)
        out.append(dict(row=row, digest=log_digest(log) if log is not None else None))
    with open(job_path + '.out', 'wb') as f:
        pickle.dump(out, f)


def run_in_new_process(dll, runs, work):
    """Run (ctrl, gains, v, mu) jobs in a fresh Python process (DLL loaded again)."""
    job = os.path.join(work, f'job_{time.time_ns()}.pkl')
    with open(job, 'wb') as f:
        pickle.dump(dict(dll=dll, runs=runs), f)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', job], capture_output=True, text=True)
    if r.returncode != 0:
        raise S03Stop(f'child process failed ({r.returncode}): {r.stderr[-2000:]}')
    with open(job + '.out', 'rb') as f:
        res = pickle.load(f)
    os.remove(job)
    os.remove(job + '.out')
    return res


# ── outputs ──────────────────────────────────────────────────────────────────────────

def fmt(x):
    if isinstance(x, bool):
        return 'true' if x else 'false'
    if isinstance(x, float):
        return repr(x)
    return x


def write_csv(path, rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows({k: fmt(r.get(k, '')) for k in cols} for r in rows)


def save_log(d, name, log):
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, name + '.npz'), **log)


def sha256_file(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── analysis (section 5), pure functions on the runs ─────────────────────────────────

def analyse(ctrl, box, pts, runs, mu_levels, speeds, y3_ref, notes=None):
    """effects.csv, curvature.csv, y2_versions.csv and fail_counts.csv rows of one controller and
    box. pts: [(point_id, codes)] of the box (128); runs: {(point_id, mu, v): row} incl. centre 0.
    notes: {(box, v): text} written in the 'note' column of every effect of that box and speed
    (S03 3.2: outer box mostly not completed; inner box also: no conclusion on Y2-Y4)."""
    notes = notes or {}
    eff, curv, ver, fails = [], [], [], []
    mu_mid = mu_levels[len(mu_levels) // 2]
    ids = [p for p, _ in pts]
    codes = np.array([c for _, c in pts])
    for v in speeds:
        def agg(fun, resp):
            return [fun([runs[(i, mu, v)][resp] for mu in mu_levels]) for i in ids]
        for cond in [str(mu) for mu in mu_levels] + ['worst', 'range']:
            cell = [runs[(i, mu, v)] for i in ids for mu in (mu_levels if cond in ('worst', 'range') else [float(cond)])]
            n_trk = sum(r['status'] == 'track_err' for r in cell)
            n_inc = sum(r['completed'] == 'inconclusive' for r in cell)
            sig_sets = {}
            for resp in RESPONSES:
                if cond == 'worst':
                    y = agg(max, resp)
                elif cond == 'range':
                    y = agg(lambda a: max(a) - min(a), resp)
                else:
                    y = [runs[(i, float(cond), v)][resp] for i in ids]
                thr = THRESH.get(resp, Y3_REL * y3_ref[(ctrl, v)])
                e = effects(codes, y)
                sig_sets[resp] = {t: math.copysign(1.0, x) for t, x in e.items() if abs(x) > thr}
                for t, x in e.items():
                    eff.append(dict(controller=ctrl, box=box, response=resp, v=v, mu=cond, term=t,
                                    order=t.count(':') + 1, effect=x, threshold=thr, significant=abs(x) > thr,
                                    censored='yes' if resp == 'e_perp_max_cens' else 'no',
                                    n_track_err=n_trk, n_inconclusive=n_inc, note=notes.get((box, v), '')))
                if cond not in ('worst', 'range') and box == 'outer' and (0, float(cond), v) in runs:
                    c0 = runs[(0, float(cond), v)][resp]
                    diff = c0 - float(np.mean(y))
                    curv.append(dict(controller=ctrl, v=v, mu=cond, response=resp, centre=c0,
                                     factorial_mean=float(np.mean(y)), diff=diff, threshold=thr,
                                     curved=abs(diff) > thr))
            a, b = sig_sets['e_perp_max'], sig_sets['e_perp_max_cens']
            same_set, same_sign = set(a) == set(b), all(a[t] == b[t] for t in set(a) & set(b))
            ver.append(dict(controller=ctrl, box=box, v=v, mu=cond, same_set=same_set, same_sign=same_sign,
                            main='as_measured' if same_set and same_sign else 'both'))
        for mu in mu_levels:
            rows = [runs[(i, mu, v)] for i in ids]
            kinds = {'not_completed': [r['completed'] == 'false' for r in rows],
                     'inconclusive': [r['completed'] == 'inconclusive' for r in rows],
                     'saturated': [r['sat_frac'] > 0 for r in rows]}
            for kind, flag in kinds.items():
                flag = np.array(flag)
                for r_ in (1, 2):
                    for comb in itertools.combinations(range(7), r_):
                        for lv in itertools.product((-1, 1), repeat=r_):
                            sel = np.all(np.sign(codes[:, comb]) == lv, axis=1)
                            fails.append(dict(controller=ctrl, box=box, v=v, mu=mu, kind=kind,
                                              factors=':'.join(f'X{i + 1}' for i in comb),
                                              levels=','.join('+' if x > 0 else '-' for x in lv),
                                              count=int(flag[sel].sum()), n=int(sel.sum()), mu_mid=mu == mu_mid))
    return eff, curv, ver, fails


# ── main ─────────────────────────────────────────────────────────────────────────────

def git_state(repo):
    def git(*a):
        return subprocess.run(['git', '-C', repo] + list(a), capture_output=True, text=True, check=True).stdout.strip()
    try:
        changed = [l[3:] for l in git('status', '--porcelain').splitlines()
                   if l[3:] and not l[3:].startswith(('fwsim/build/', '.vscode/'))]
        return dict(commit=git('rev-parse', '--short', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                    dirty=bool(changed), modified=changed)
    except (OSError, subprocess.CalledProcessError) as e:
        return dict(commit='UNKNOWN', dirty=True, modified=[str(e)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    ap.add_argument('--mu', type=float, nargs='+', required=True, help='mu levels low [mid high]; one level = --trial only')
    ap.add_argument('--mu-source', default='', help='where the mu levels come from (meta.yaml)')
    ap.add_argument('--trial', action='store_true', help='tool check (S03 11.3): never main numbers')
    ap.add_argument('--gate-speeds', type=float, nargs='+', default=list(SPEEDS),
                    help='speeds still in the O1 gate (O1 v7 3.9); the level rule uses only these')
    ap.add_argument('--controllers', nargs='+', default=list(CONTROLLERS), choices=CONTROLLERS)
    ap.add_argument('--child', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if len(a.mu) not in (1, 3) or (len(a.mu) == 1 and not a.trial):
        sys.exit('--mu needs three levels (S03 2.2); a single level only with --trial')
    if a.trial and a.mu != [0.5]:
        sys.exit('--trial runs with mu 0.5 only (S03 2.2: the fwsim assumption)')
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    os.makedirs(a.out, exist_ok=True)
    mu_levels = sorted(a.mu)
    mu_mid = mu_levels[len(mu_levels) // 2]
    fw = fwsim.FirmwareSim(a.dll)
    if fw.build_info.get('esp32', {}).get('dirty', True):
        print('WARNING: fwsim.dll was built from a dirty firmware tree; not usable as a main result')
    path = o1.Path8()
    t0 = time.time()
    sweep_rows, level_rows, design_rows, all_runs = [], [], [], []
    sealed_logs = os.path.join(a.out, 'sealed', 'logs')     # determinism points, centre, runs not completed (L56)
    inner_rows = []
    eff_rows, curv_rows, ver_rows, fail_rows, det_rows, stops, inner_run = [], [], [], [], [], {}, {}
    y3_ref, rerun_notes = {}, []

    def checked(ctrl, g, v, mu):
        row, log = safe_run(fw, path, ctrl, g, v, mu)
        if row['status'] == 'run_error':
            again = run_in_new_process(a.dll, [(ctrl, g, v, mu)], a.out)[0]['row']
            rerun_notes.append(dict(controller=ctrl, gains=g, v=v, mu=mu, first=row.get('error'),
                                    second=again.get('error', 'ran')))
            raise S03Stop(f'run_error at {ctrl} {g} v={v} mu={mu}: {row.get("error")}; rerun in a new process: '
                          f'{again.get("error", "ran (fwsim not deterministic there)")} (B8b-20: S03 stops, '
                          'CODE finds the cause)')
        return row, log

    for ctrl in a.controllers:
        try:
            dflt = defaults(fw, ctrl)
            lo_b, hi_b = tune_ga.BOUNDS[ctrl]
            # 3.1 sweep
            levels = {}
            for j, name in enumerate(FACTORS[ctrl]):
                grid = sweep_grid(name, dflt[name], lo_b[j], hi_b[j])
                ok = []
                for x in grid:
                    g = dict(dflt, **{name: x})
                    done = True
                    for v in SPEEDS:
                        row, _ = checked(ctrl, g, v, mu_mid)
                        sweep_rows.append(dict(controller=ctrl, factor=name, value=x, is_default=x == dflt[name],
                                               v=v, mu=mu_mid, **row))
                        if x == dflt[name]:
                            y3_ref[(ctrl, v)] = row['tv_duty']
                        if v in a.gate_speeds:
                            done &= row['completed'] == 'true'
                    ok.append(done)
                    print(f'[sweep {ctrl}] {name} = {x:.4g}: completes at gate speeds = {done}', flush=True)
                try:
                    L, H = levels_from_sweep(grid, dflt[name], ok)
                except S03Stop as exc:
                    raise S03Stop(f'{name}: {exc}') from None
                sc, centre = scale_of(L, H)
                levels[name] = (L, H, sc)
                level_rows.append(dict(controller=ctrl, factor=name, L=L, H=H, centre=centre, scale=sc,
                                       default=dflt[name], rule_note='longest completing run of the grid with the '
                                       'default, gate speeds ' + ' '.join(map(str, a.gate_speeds))))
                print(f'[levels {ctrl}] {name}: L={L:.4g} H={H:.4g} scale={sc}', flush=True)
            same = [n for n, (L, H, _) in levels.items() if L == H]
            if same:
                raise S03Stop(f'L = H for {same} (S03 3.1 (ii)): Liam decides, no main design')
            # 3.1 extra: default gains at low / high mu too (section 5.2: reported, not the threshold base)
            for mu in mu_levels:
                if mu == mu_mid:
                    continue
                for v in SPEEDS:
                    row, _ = checked(ctrl, dflt, v, mu)
                    sweep_rows.append(dict(controller=ctrl, factor='(default)', value='', is_default=True,
                                           v=v, mu=mu, **row))

            # 3.2 design
            def gains_at(codes):
                return {n: decode(c, *levels[n]) for n, c in zip(FACTORS[ctrl], codes)}
            boxes = [('outer', design_points(1.0, 1))]
            runs, keep_logs = {}, {}
            centre_codes = (0.0,) * 7
            rng = np.random.default_rng(DET_SEED)
            det_ids = [0] + sorted(int(x) for x in rng.choice(np.arange(1, 129), size=3, replace=False))
            design_rows.append(dict(controller=ctrl, point_id=0, box='centre',
                                    **{f'X{k + 1}': 0.0 for k in range(7)},
                                    **{f'{n}_value': v_ for n, v_ in gains_at(centre_codes).items()}))
            todo = [(0, 'centre', centre_codes)]
            for pid, codes in boxes[0][1]:
                design_rows.append(dict(controller=ctrl, point_id=pid, box='outer',
                                        **{f'X{k + 1}': c for k, c in enumerate(codes)},
                                        **{f'{n}_value': v_ for n, v_ in gains_at(codes).items()}))
                todo.append((pid, 'outer', codes))

            def run_points(points):
                # L56: nothing about the main design goes to the console (not even run counts, which
                # would tell whether the inner box ran)
                for pid, box, codes in points:
                    g = gains_at(codes)
                    for mu in mu_levels:
                        for v in SPEEDS:
                            row, log = checked(ctrl, g, v, mu)
                            runs[(pid, mu, v)] = row
                            all_runs.append(dict(controller=ctrl, point_id=pid, box=box, mu=mu, v=v, **row))
                            if pid in det_ids and box != 'inner':
                                keep_logs[(pid, mu, v)] = (g, row, log_digest(log))
                            if pid in det_ids and box != 'inner' or box == 'centre' or row['completed'] != 'true':
                                save_log(sealed_logs, f'{ctrl}_p{pid}_mu{mu}_v{v}', log)
            print(f'[design {ctrl}] running the main design (results sealed, L56)', flush=True)
            run_points(todo)
            corner_ids = [p for p, _ in boxes[0][1]]
            need_inner = [v for v in SPEEDS
                          if sum(runs[(i, mu_mid, v)]['completed'] == 'false' for i in corner_ids) > 64]
            inner_run[ctrl] = bool(need_inner)
            if need_inner:
                inner = design_points(0.5, 129)
                boxes.append(('inner', inner))
                for pid, codes in inner:
                    inner_rows.append(dict(controller=ctrl, point_id=pid, box='inner',
                                            **{f'X{k + 1}': c for k, c in enumerate(codes)},
                                            **{f'{n}_value': v_ for n, v_ in gains_at(codes).items()}))
                run_points([(p, 'inner', c) for p, c in inner])
            notes = {('outer', v): 'most corners not completed at the middle mu (S03 3.2): reported with this '
                     'warning; see the inner box' for v in need_inner}
            for v in need_inner:
                if sum(runs[(p, mu_mid, v)]['completed'] == 'false' for p, _ in boxes[1][1]) > 64:
                    notes[('inner', v)] = ('inner box also mostly not completed at the middle mu (S03 3.2): '
                                           'NO conclusion on Y2-Y4 effects at this speed')

            # 3.4 determinism in a new process
            jobs = sorted(keep_logs)
            res = run_in_new_process(a.dll, [(ctrl, keep_logs[j][0], j[2], j[1]) for j in jobs], a.out)
            bad = []
            for j, r in zip(jobs, res):
                g, row0, dig0 = keep_logs[j]
                same_log = r['digest'] == dig0
                same_row = row_text(r['row']) == row_text(row0)
                det_rows.append(dict(controller=ctrl, point_id=j[0], mu=j[1], v=j[2], log_arrays_equal=same_log,
                                     runs_fields_equal=same_row, n_arrays=len(dig0)))
                if not (same_log and same_row):
                    bad.append(j)
            if bad:
                raise S03Stop(f'determinism check failed at {bad} (S03 3.4): stop, tell CODE and THESIS')

            # 5 analysis
            for box, pts in boxes:
                e, c, vr, f = analyse(ctrl, box, pts, runs, mu_levels, SPEEDS, y3_ref, notes)
                eff_rows += e
                curv_rows += c
                ver_rows += vr
                fail_rows += f
        except S03Stop as exc:
            stops[ctrl] = str(exc)
            print(f'[STOP {ctrl}] {exc}', flush=True)

    elapsed = time.time() - t0
    files = {}
    sealed = os.path.join(a.out, 'sealed')
    os.makedirs(sealed, exist_ok=True)
    with open(os.path.join(sealed, 'notes.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(inner_box_run=inner_run), f, indent=1)
    for d, name, rows in ((a.out, 'sweep.csv', sweep_rows), (a.out, 'levels.csv', level_rows),
                          (a.out, 'design.csv', design_rows), (a.out, 'determinism_check.csv', det_rows),
                          (sealed, 'design_inner.csv', inner_rows), (sealed, 'runs.csv', all_runs),
                          (sealed, 'effects.csv', eff_rows), (sealed, 'curvature.csv', curv_rows),
                          (sealed, 'y2_versions.csv', ver_rows), (sealed, 'fail_counts.csv', fail_rows)):
        p = os.path.join(d, name)
        write_csv(p, rows, RUN_COLS if name == 'runs.csv' else None)
        files[os.path.relpath(p, a.out).replace(os.sep, '/')] = sha256_file(p)
    files['sealed/notes.json'] = sha256_file(os.path.join(sealed, 'notes.json'))
    b = fw.build_info
    dll_sha = sha256_file(a.dll)
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: S03 v6 (06_SPECS/2026-11-05_Spec_S03_DOE_chon_gain.md)
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
trial: {str(a.trial).lower()}     # true: tool check with mu 0.5 (S03 2.2, 11.3), never main numbers
sealed: "L56 / S03 section 1: runs.csv, effects.csv and the section 5 tables are not looked at before campaign A ends; their SHA-256 below"
script: {json.dumps(git_state(os.path.join(HERE, '..')))}
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework_at_build: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
  dll_sha256: {dll_sha}
  h7: null
  rpi5: null
plant: {fwsim.plant_meta(fwsim.plant_params(argparse.Namespace(ground='traction', mu=mu_mid)))}
plant_check: chua_X3
mu_levels: {json.dumps(mu_levels)}
mu_source: "{a.mu_source or ('trial: fwsim assumption 0.5 (params.py), no MU_FLOOR measurement yet' if a.trial else '')}"
imu: {json.dumps(IMU)}
seed: {SEED}                       # plant has no random input
determinism_seed: {DET_SEED}       # draws the 3 corners of the determinism check (S03 3.4)
speeds: {json.dumps(list(SPEEDS))}
gate_speeds: {json.dumps(a.gate_speeds)}
controllers: {json.dumps(a.controllers)}
factors: {json.dumps(FACTORS)}
temporary_bounds_for_the_grid: {json.dumps({m: tune_ga.BOUNDS[m] for m in a.controllers})}   # tune_ga.BOUNDS (S03 3.1)
thresholds: {json.dumps(dict(THRESH, tv_duty=f'{Y3_REL:.0%} of the default-gain run, middle mu, same speed'))}
y3_reference: {json.dumps({f'{c} v={v}': y for (c, v), y in y3_ref.items()})}
o1: {json.dumps(dict(A=o1.A, a_r=o1.A_R, D=o1.D, D_d=o1.D, t_d=o1.T_D, t_late=o1.T_LATE, delta=o1.DELTA, v_d=o1.V_D, c_cens=o1.C_CENS))}
stops: {json.dumps(stops)}
run_error_reruns: {json.dumps(rerun_notes)}
interpretations:   # of this script, not written in S03 v6 / O1 v7; Liam or THESIS correct if wrong
  - "completion: conditions 1, 2, 5 at the instant, condition 3 as the trailing unbroken t_d window (o1_feasibility.evaluate_v7)"
  - "sweep level rule: a point is 'completing' only with y1 'true' at every gate speed (inconclusive breaks the run)"
  - "inner box trigger: corners with y1 'false' only (inconclusive not counted)"
  - "one run per (point, mu, speed) for the centre too; logs saved (npz of every fwsim.simulate array, exact bits; spec section 7 says log.csv) for the determinism points, the centre and every run with y1 not 'true', all in sealed/logs/ (they are main-design results)"
  - "sealed/ holds everything derived from the main-design responses (L56), including whether the inner box ran (sealed/notes.json, sealed/design_inner.csv)"
  - "a controller that hits a stop of the spec is stopped; the other controller still runs (meta 'stops')"
elapsed_s: {elapsed:.0f}
file_sha256: {json.dumps(files)}
files:
  - sweep.csv: "one line per sweep run (section 3.1) and default gains at the other mu levels"
  - levels.csv: "L, H, centre, scale per factor"
  - design.csv: "points: codes X1..X7 and real values"
  - design.csv: "centre + outer box (the inner box, if run, is in sealed/design_inner.csv)"
  - determinism_check.csv: "section 3.4: new-process rerun, log arrays and runs fields equal"
  - sealed/: "SEALED (L56): runs.csv (section 4 responses), effects.csv (every term, per box, response, speed, mu / worst / range), curvature.csv (5.4), y2_versions.csv (5.1), fail_counts.csv (5.3), design_inner.csv, notes.json, logs/"
""")
    print(f'S03 done in {elapsed:.0f} s -> {a.out}; stops: {stops or "none"}')
    return 0 if not stops else 2


if __name__ == '__main__':
    if '--child' in sys.argv:
        child_main(sys.argv[sys.argv.index('--child') + 1])
        sys.exit(0)
    sys.exit(main())
