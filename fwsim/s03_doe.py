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
       inner box +-0.5 is reported too (128 more points, all conditions). The inner box always
       runs (L41, Liam 06/11): the run time must not tell whether that rule is on; its runs
       enter the analysis only when the rule is on.
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
Outputs (section 7) in --out: sweep.csv, sweep_counts.csv, levels.csv, design.csv (centre,
outer and inner box), logs_sweep/, meta.yaml; in --out/sealed/ (L56): runs.csv, effects.csv,
fail_counts.csv, curvature.csv, y2_versions.csv, determinism_check.csv, notes.json,
stop.json, manifest.json (SHA-256 of the sealed files), logs/.

L56 / S03 v6 section 1: the main design results (runs.csv, effects.csv and the tables of
section 5) are NOT looked at before campaign A ends. Nothing derived from them leaves
sealed/ (Codex S3C-1..S3C-3): no hash of a sealed file outside it, no child-process job file
outside it, a stop in the main design prints only that S03 stopped, the determinism check
is only 'passed' there, design.csv holds every point of every controller before the main
design starts, and the inner box always runs (L41). Every spec stop ends S03
for both controllers (S3C-9). The sweep (section 3.1) is a source of campaign A's domain and
is printed.

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
import struct
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
            'cond2_5_ever', 'pos_reset_mid', 'e_perp_rms', 'e_trk_rms', 'error']     # RMS: S3C-13; error: S3C-12
SCALE_RTOL = 1e-12            # H/L >= 4 within float rounding of the log grid (S3C-6; 1e-9 too wide, S3D-7)


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
    """('log', centre) when L > 0 and H/L >= 4, else ('lin', centre) (section 3.1). The ratio is
    compared with a relative tolerance: grid points from np.geomspace carry rounding, e.g.
    L = 12.000000000000002 for the mathematical 12 (S3C-6); the tolerance is a few hundred ULP,
    so a ratio really below 4 (3.999999998) stays linear (S3D-7)."""
    if L > 0 and H / L >= 4.0 * (1.0 - SCALE_RTOL):
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


def _bits(x):
    """Exact representation: a float by its IEEE-754 bytes (NaN payloads too), others by repr."""
    if isinstance(x, (float, np.floating)):
        return 'f' + struct.pack('<d', float(x)).hex()
    if isinstance(x, (bool, np.bool_)):
        return 'b' + str(bool(x))
    if isinstance(x, (int, np.integer)):
        return 'i' + str(int(x))
    return 'r' + repr(x)


def row_text(row, ids=()):
    """Every runs.csv field of one run, identifiers included, bit-exact (S3C-15). ids: the
    (controller, point_id, box, mu, v) of the run."""
    full = dict(zip(RUN_COLS[:5], ids), **row)
    return {k: _bits(full.get(k)) for k in RUN_COLS}


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
    """Run (ctrl, gains, v, mu) jobs in a fresh Python process (DLL loaded again). work must be
    a folder under sealed/ when the runs belong to the main design (S3C-2); the job files are
    removed in every case."""
    os.makedirs(work, exist_ok=True)
    job = os.path.join(work, f'job_{time.time_ns()}.pkl')
    try:
        with open(job, 'wb') as f:
            pickle.dump(dict(dll=dll, runs=runs), f)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', job], capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise S03Stop(f'child process failed ({r.returncode}): {r.stderr[-2000:]}')
        with open(job + '.out', 'rb') as f:
            return pickle.load(f)
    finally:
        for p in (job, job + '.out'):
            if os.path.exists(p):
                os.remove(p)


# ── outputs ──────────────────────────────────────────────────────────────────────────

def fmt(x):
    """CSV text: Python and NumPy scalars as plain numbers (repr of np.float64 is
    'np.float64(...)' in NumPy 2, S3C-14), booleans as true/false."""
    if isinstance(x, (bool, np.bool_)):
        return 'true' if x else 'false'
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    if isinstance(x, np.integer):
        return int(x)
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


def save_log_csv(d, name, log):
    """log.csv of one run (S03 3.4: the four determinism points): one column per array column,
    %.17g so every float64 reads back to the same bits."""
    os.makedirs(d, exist_ok=True)
    n = len(log['t'])
    cols, data = [], []
    for k in sorted(log):
        a = np.asarray(log[k], dtype=float).reshape(n, -1)
        cols += [k] if a.shape[1] == 1 else [f'{k}_{i}' for i in range(a.shape[1])]
        data.append(a)
    np.savetxt(os.path.join(d, name + '.csv'), np.hstack(data), delimiter=',', header=','.join(cols),
               comments='', fmt='%.17g')


def sha256_file(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── analysis (section 5), pure functions on the runs ─────────────────────────────────

def analyse(ctrl, box, pts, runs, mu_levels, speeds, y3_ref, notes=None, no_conclusion=()):
    """effects.csv, curvature.csv, y2_versions.csv and fail_counts.csv rows of one controller and
    box. pts: [(point_id, codes)] of the box (128); runs: {(point_id, mu, v): row} incl. centre 0.
    notes: {(box, v): text} written in the 'note' column of every effect of that box and speed
    (S03 3.2: outer box mostly not completed). no_conclusion: speeds where the inner box is
    also mostly not completed: no effects, curvature or Y2 versions there, only the fail counts
    (S03 3.2, S3C-11)."""
    notes = notes or {}
    eff, curv, ver, fails = [], [], [], []
    mu_mid = mu_levels[len(mu_levels) // 2]
    ids = [p for p, _ in pts]
    codes = np.array([c for _, c in pts])
    for v in speeds:
        def agg(fun, resp):
            return [fun([runs[(i, mu, v)][resp] for mu in mu_levels]) for i in ids]
        for cond in ([] if v in no_conclusion else [str(mu) for mu in mu_levels] + ['worst', 'range']):
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
                note = notes.get((box, v), '')
                if resp == 'tv_duty' and thr == 0.0:            # S03 5.2: reference Y3 = 0 (S3C-13)
                    note = '; '.join(x for x in (note, 'Y3 reference run is 0: every non-zero effect '
                                                 'is reported') if x)
                e = effects(codes, y)
                sig_sets[resp] = {t: math.copysign(1.0, x) for t, x in e.items() if abs(x) > thr}
                for t, x in e.items():
                    eff.append(dict(controller=ctrl, box=box, response=resp, v=v, mu=cond, term=t,
                                    order=t.count(':') + 1, effect=x, threshold=thr, significant=abs(x) > thr,
                                    censored='yes' if resp == 'e_perp_max_cens' else 'no',
                                    n_track_err=n_trk, n_inconclusive=n_inc, note=note))
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
        for mu in list(mu_levels) + ['worst']:
            if mu == 'worst':    # S03 5.1: Y1 robust = completed at every mu (S3C-13)
                # three-valued worst Y1: 'false' if any mu is false, else 'inconclusive' if any mu is
                # inconclusive, else 'true'; the two counts never hold the same point
                per = [[runs[(i, m, v)] for m in mu_levels] for i in ids]
                nc = [any(r['completed'] == 'false' for r in rs) for rs in per]
                kinds = {'not_completed': nc,
                         'inconclusive': [not f_ and any(r['completed'] == 'inconclusive' for r in rs)
                                          for f_, rs in zip(nc, per)],
                         'saturated': [any(r['sat_frac'] > 0 for r in rs) for rs in per]}
            else:
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


def sweep_counts(rows):
    """S03 5.3 for the sweep (S3C-13): per controller, factor, speed, mu: points, not completed,
    inconclusive, saturated."""
    out = {}
    for r in rows:
        k = (r['controller'], r['factor'], r['v'], r['mu'])
        c = out.setdefault(k, dict(controller=k[0], factor=k[1], v=k[2], mu=k[3], n=0, not_completed=0,
                                   inconclusive=0, saturated=0))
        c['n'] += 1
        c['not_completed'] += r.get('completed') == 'false'
        c['inconclusive'] += r.get('completed') == 'inconclusive'
        c['saturated'] += bool(r.get('sat_frac', 0) and r['sat_frac'] > 0)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    ap.add_argument('--mu', type=float, nargs='+', required=True, help='mu levels low [mid high]; one level = --trial only')
    ap.add_argument('--mu-source', default='', help='where the mu levels come from (meta.yaml)')
    ap.add_argument('--trial', action='store_true', help='tool check (S03 11.3): never main numbers')
    ap.add_argument('--gate-speeds', type=float, nargs='+', default=list(SPEEDS),
                    help='speeds still in the O1 gate (O1 v7 3.9): a non-empty subset of 0.2 0.5 0.8')
    ap.add_argument('--controllers', nargs='+', default=list(CONTROLLERS), choices=CONTROLLERS)
    ap.add_argument('--child', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if len(a.mu) not in (1, 3) or (len(a.mu) == 1 and not a.trial):
        sys.exit('--mu needs three levels (S03 2.2); a single level only with --trial')
    if a.trial and a.mu != [0.5]:
        sys.exit('--trial runs with mu 0.5 only (S03 2.2: the fwsim assumption)')
    if not a.gate_speeds or any(not any(math.isclose(g, s) for s in SPEEDS) for g in a.gate_speeds):
        sys.exit(f'--gate-speeds must be a non-empty subset of {list(SPEEDS)} (S3C-10)')
    gate = [s for s in SPEEDS if any(math.isclose(g, s) for g in a.gate_speeds)]
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    sealed = os.path.join(a.out, 'sealed')
    os.makedirs(sealed, exist_ok=True)
    work_pub, work_sealed = os.path.join(a.out, '_work'), os.path.join(sealed, '_work')
    sealed_logs = os.path.join(sealed, 'logs')        # determinism points, centre, runs not completed (L56)
    mu_levels = sorted(a.mu)
    mu_mid = mu_levels[len(mu_levels) // 2]
    fw = fwsim.FirmwareSim(a.dll)
    if fw.build_info.get('esp32', {}).get('dirty', True):
        print('WARNING: fwsim.dll was built from a dirty firmware tree; not usable as a main result')
    path = o1.Path8()
    t0 = time.time()
    sweep_rows, level_rows, design_rows, all_runs = [], [], [], []
    eff_rows, curv_rows, ver_rows, fail_rows, det_rows = [], [], [], [], []
    y3_ref, rerun_notes, inner_rule, no_concl = {}, [], {}, {}
    stop_public, stop_sealed = None, None
    t_sweep_end = None

    def checked(ctrl, g, v, mu, ids=None):
        """One run; a run_error is recorded (S3C-12), rerun once in a new process for diagnosis
        (B8b-20), then S03 stops. ids = (point_id, box) for a main-design run (sealed)."""
        row, log = safe_run(fw, path, ctrl, g, v, mu)
        if row['status'] != 'run_error':
            return row, log
        if ids is not None:
            all_runs.append(dict(controller=ctrl, point_id=ids[0], box=ids[1], mu=mu, v=v, **row))
        note = dict(controller=ctrl, gains=g, v=v, mu=mu, point=ids, first=row.get('error'))
        rerun_notes.append(note)
        try:
            again = run_in_new_process(a.dll, [(ctrl, g, v, mu)], work_pub if ids is None else work_sealed)[0]['row']
            note['second'] = again.get('error', 'ran (fwsim not deterministic there)')
        except Exception as exc:          # noqa: BLE001 (S3C-12: any failure of the diagnosis is recorded)
            note['second'] = f'diagnosis process failed: {type(exc).__name__}: {exc}'
        raise S03Stop(f'run_error at {ctrl} {g} v={v} mu={mu}: {note["first"]}; rerun in a new process: '
                      f'{note["second"]} (B8b-20: S03 stops, CODE finds the cause)')

    try:
        # 3.1 (i) first, for every controller, so S03 stops before any sweep (S3C-9)
        dflts = {}
        for ctrl in a.controllers:
            dflts[ctrl] = defaults(fw, ctrl)
            for v in SPEEDS:
                row, log = checked(ctrl, dflts[ctrl], v, mu_mid)
                # recorded and logged before the stop test (S3D-4); public like the sweep
                sweep_rows.append(dict(controller=ctrl, factor='(default check)', value='', is_default=True,
                                       v=v, mu=mu_mid, **row))
                if row['completed'] != 'true':
                    save_log(os.path.join(a.out, 'logs_sweep'), f'{ctrl}_default_check_v{v}', log)
                if v in gate and row['completed'] != 'true':
                    raise S03Stop(f'{ctrl}: default gains do not complete at gate speed {v} (y1 '
                                  f'{row["completed"]}, S03 3.1 (i))')
        # 3.1 sweep and levels, every controller (S3C-9: a stop ends S03)
        levels = {}
        for ctrl in a.controllers:
            dflt = dflts[ctrl]
            lo_b, hi_b = tune_ga.BOUNDS[ctrl]
            levels[ctrl] = {}
            for j, name in enumerate(FACTORS[ctrl]):
                grid = sweep_grid(name, dflt[name], lo_b[j], hi_b[j])
                ok = []
                for x in grid:
                    g = dict(dflt, **{name: x})
                    done = True
                    for v in SPEEDS:
                        row, log = checked(ctrl, g, v, mu_mid)
                        sweep_rows.append(dict(controller=ctrl, factor=name, value=x, is_default=x == dflt[name],
                                               v=v, mu=mu_mid, **row))
                        if x == dflt[name]:
                            y3_ref[(ctrl, v)] = row['tv_duty']
                        if row['completed'] != 'true':          # sweep is public: its failed logs too (S3C-20)
                            save_log(os.path.join(a.out, 'logs_sweep'), f'{ctrl}_{name}_{x:.6g}_v{v}', log)
                        if v in gate:
                            done &= row['completed'] == 'true'
                    ok.append(done)
                    print(f'[sweep {ctrl}] {name} = {x:.4g}: completes at gate speeds = {done}', flush=True)
                try:
                    L, H = levels_from_sweep(grid, dflt[name], ok)
                except S03Stop as exc:
                    raise S03Stop(f'{ctrl} {name}: {exc}') from None
                sc, centre = scale_of(L, H)
                levels[ctrl][name] = (L, H, sc)
                level_rows.append(dict(controller=ctrl, factor=name, L=L, H=H, centre=centre, scale=sc,
                                       default=dflt[name], rule_note='longest completing run of the grid with the '
                                       'default, gate speeds ' + ' '.join(map(str, gate))))
                print(f'[levels {ctrl}] {name}: L={L:.4g} H={H:.4g} scale={sc}', flush=True)
            same = [n for n, (L, H, _) in levels[ctrl].items() if L == H]
            if same:
                raise S03Stop(f'{ctrl}: L = H for {same} (S03 3.1 (ii)): Liam decides, no main design')
            for mu in mu_levels:        # default gains at low / high mu too (5.2: reported, not the base)
                if mu == mu_mid:
                    continue
                for v in SPEEDS:
                    row, log = checked(ctrl, dflt, v, mu)
                    sweep_rows.append(dict(controller=ctrl, factor='(default)', value='', is_default=True,
                                           v=v, mu=mu, **row))
                    if row['completed'] != 'true':                      # S3C-20
                        save_log(os.path.join(a.out, 'logs_sweep'), f'{ctrl}_default_mu{mu}_v{v}', log)
        t_sweep_end = time.time()
    except S03Stop as exc:
        stop_public = str(exc)                          # sweep phase: public (S03 3.1 is not sealed)
        print(f'[STOP] {exc}', flush=True)

    if stop_public is None:
        # every design point of every controller is public before any main-design run: the points
        # follow from the public levels only, so design.csv does not depend on a stop (S3C-3 (c))
        plan = {}
        for ctrl in a.controllers:
            lv = levels[ctrl]

            def gains_at(codes, lv=lv, ctrl=ctrl):
                return {n: decode(c, *lv[n]) for n, c in zip(FACTORS[ctrl], codes)}
            centre_codes = (0.0,) * 7
            outer, inner = design_points(1.0, 1), design_points(0.5, 129)
            for pid, box, codes in ([(0, 'centre', centre_codes)] + [(p, 'outer', c) for p, c in outer] +
                                    [(p, 'inner', c) for p, c in inner]):
                design_rows.append(dict(controller=ctrl, point_id=pid, box=box,
                                        **{f'X{k + 1}': c for k, c in enumerate(codes)},
                                        **{f'{n}_value': v_ for n, v_ in gains_at(codes).items()}))
            plan[ctrl] = (gains_at, centre_codes, outer, inner)
        try:
            for ctrl in a.controllers:
                gains_at, centre_codes, outer, inner = plan[ctrl]
                boxes = [('outer', outer)]
                runs, keep_logs = {}, {}
                rng = np.random.default_rng(DET_SEED)
                det_ids = [0] + sorted(int(x) for x in rng.choice(np.arange(1, 129), size=3, replace=False))
                todo = [(0, 'centre', centre_codes)] + [(pid, 'outer', codes) for pid, codes in outer]

                def run_points(points):
                    # L56: nothing about the main design goes to the console or outside sealed/
                    for pid, box, codes in points:
                        g = gains_at(codes)
                        for mu in mu_levels:
                            for v in SPEEDS:
                                row, log = checked(ctrl, g, v, mu, ids=(pid, box))
                                runs[(pid, mu, v)] = row
                                all_runs.append(dict(controller=ctrl, point_id=pid, box=box, mu=mu, v=v, **row))
                                name = f'{ctrl}_p{pid}_mu{mu}_v{v}'
                                if pid in det_ids and box != 'inner':
                                    keep_logs[(pid, mu, v)] = (g, box, row, log_digest(log))
                                    save_log_csv(sealed_logs, name, log)          # S03 3.4: log.csv
                                if box == 'centre' or row['completed'] != 'true':
                                    save_log(sealed_logs, name, log)
                print(f'[design {ctrl}] running the main design (results sealed, L56)', flush=True)
                run_points(todo)
                # L41 (Liam 06/11, 'Chot B'): the inner box always runs, so the run time does not tell
                # whether the rule of S03 3.2 is on; it enters the analysis only when the rule is on
                run_points([(p, 'inner', c) for p, c in inner])
                corner_ids = [p for p, _ in outer]
                need_inner = [v for v in SPEEDS
                              if sum(runs[(i, mu_mid, v)]['completed'] == 'false' for i in corner_ids) > 64]
                inner_rule[ctrl] = need_inner
                if need_inner:
                    boxes.append(('inner', inner))
                notes = {('outer', v): 'most corners not completed at the middle mu (S03 3.2): reported with this '
                         'warning; see the inner box' for v in need_inner}
                no_concl[ctrl] = [v for v in need_inner
                                  if sum(runs[(p, mu_mid, v)]['completed'] == 'false' for p, _ in inner) > 64]

                # 3.4 determinism in a new process (job files under sealed/, S3C-2)
                jobs = sorted(keep_logs)
                res = run_in_new_process(a.dll, [(ctrl, keep_logs[j][0], j[2], j[1]) for j in jobs], work_sealed)
                for j, r in zip(jobs, res):
                    g, box, row0, dig0 = keep_logs[j]
                    ids = (ctrl, j[0], box, j[1], j[2])
                    det_rows.append(dict(controller=ctrl, point_id=j[0], mu=j[1], v=j[2],
                                         log_arrays_equal=r['digest'] == dig0,
                                         runs_fields_equal=row_text(r['row'], ids) == row_text(row0, ids),
                                         n_arrays=len(dig0)))
                if not all(d['log_arrays_equal'] and d['runs_fields_equal'] for d in det_rows):
                    raise S03Stop('determinism check failed (S03 3.4): stop, tell CODE and THESIS')

                # 5 analysis
                for box, pts in boxes:
                    e, c, vr, f = analyse(ctrl, box, pts, runs, mu_levels, SPEEDS, y3_ref, notes,
                                          no_conclusion=no_concl[ctrl])
                    eff_rows += e
                    curv_rows += c
                    ver_rows += vr
                    fail_rows += f
        except S03Stop as exc:
            stop_sealed = str(exc)
            print('[STOP] S03 stopped in the main design; the reason is in sealed/stop.json (L56)', flush=True)

    t_end = time.time()
    # sealed outputs, with their own manifest (no hash of a sealed file outside sealed/, S3C-1)
    sealed_files = {}
    for name, rows, cols in (('runs.csv', all_runs, RUN_COLS), ('effects.csv', eff_rows, None),
                             ('curvature.csv', curv_rows, None), ('y2_versions.csv', ver_rows, None),
                             ('fail_counts.csv', fail_rows, None), ('determinism_check.csv', det_rows, None)):
        p = os.path.join(sealed, name)
        write_csv(p, rows, cols)
        sealed_files[name] = sha256_file(p)
    with open(os.path.join(sealed, 'notes.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(inner_rule_speeds=inner_rule, no_conclusion_speeds=no_concl, stop=stop_sealed,
                       run_error_reruns=[n for n in rerun_notes if n['point'] is not None]),
                  f, indent=1, default=str)
    sealed_files['notes.json'] = sha256_file(os.path.join(sealed, 'notes.json'))
    if stop_sealed:
        with open(os.path.join(sealed, 'stop.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(stop=stop_sealed), f, indent=1)
    with open(os.path.join(sealed, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(files=sealed_files), f, indent=1)
    # public outputs: sweep only (S03 3.1 is a source of campaign A's domain) and the design points
    files = {}
    for name, rows in (('sweep.csv', sweep_rows), ('sweep_counts.csv', sweep_counts(sweep_rows)),
                       ('levels.csv', level_rows), ('design.csv', design_rows)):
        p = os.path.join(a.out, name)
        write_csv(p, rows)
        files[name] = sha256_file(p)
    # S3C-3 (b): no count in public; a failed check stops S03 (3.4), so 'passed' iff it ran to the end
    det_public = ('not run' if stop_public else 'see sealed/ (S03 stopped in the main design)' if stop_sealed
                  else 'passed (detail in sealed/determinism_check.csv)')
    b = fw.build_info
    dll_sha = sha256_file(a.dll)
    sweep_s = round((t_sweep_end or t_end) - t0)
    main_s = round(t_end - t_sweep_end) if t_sweep_end else 0
    status = ('stopped in the sweep: ' + stop_public if stop_public else
              'stopped in the main design (reason sealed)' if stop_sealed else 'ran to the end')
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: S03 v6 (06_SPECS/2026-11-05_Spec_S03_DOE_chon_gain.md)
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
trial: {str(a.trial).lower()}     # true: tool check with mu 0.5 (S03 2.2, 11.3), never main numbers
sealed: "L56 / S03 section 1: everything derived from the main-design responses is in sealed/ (with its own manifest.json) and is not looked at before campaign A ends"
status: {json.dumps(status)}
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
determinism: "{det_public}"
speeds: {json.dumps(list(SPEEDS))}
gate_speeds: {json.dumps(gate)}
controllers: {json.dumps(a.controllers)}
factors: {json.dumps(FACTORS)}
temporary_bounds_for_the_grid: {json.dumps({m: tune_ga.BOUNDS[m] for m in a.controllers})}   # tune_ga.BOUNDS (S03 3.1)
thresholds: {json.dumps(dict(THRESH, tv_duty=f'{Y3_REL:.0%} of the default-gain run, middle mu, same speed'))}
y3_reference: {json.dumps({f'{c} v={v}': y for (c, v), y in y3_ref.items()})}
o1: {json.dumps(dict(A=o1.A, a_r=o1.A_R, D=o1.D, D_d=o1.D, t_d=o1.T_D, t_late=o1.T_LATE, delta=o1.DELTA, v_d=o1.V_D, c_cens=o1.C_CENS))}
sweep_run_error_reruns: {json.dumps([n for n in rerun_notes if n['point'] is None], default=str)}
interpretations:   # of this script, not written in S03 v6 / O1 v7; Liam or THESIS correct if wrong
  - "completion: conditions 1, 2, 5 at the instant, condition 3 as the trailing unbroken t_d window, condition 4 strictly before the deadline; after a tracking error that came before the progress, the window runs to the deadline (o1_feasibility.evaluate_v7)"
  - "sweep level rule: a point is 'completing' only with y1 'true' at every gate speed (inconclusive breaks the run)"
  - "inner box trigger: corners with y1 'false' only (inconclusive not counted)"
  - "a stop of the spec (3.1 (i), (ii), 3.4, run_error) ends S03 for both controllers; the default gains of every controller are checked first"
  - "logs: log.csv (%.17g) of the four determinism points and npz (exact bits) of the centre and every main-design run not completed, all in sealed/logs/; npz of every sweep run not completed in logs_sweep/ (public, as the sweep)"
  - "the inner box (codes +-0.5) always runs for every controller (L41, Liam 06/11 'Chot B'), so the run time does not depend on the results; it enters the analysis (effects, curvature, Y2 versions, fail counts; at every speed, as before L41) only when the rule of S03 3.2 is on at one speed or more (> 64/128 corners 'false' at the middle mu); otherwise its runs are only rows of sealed/runs.csv"
  - "public channels left (L56, S3C-3): status, exit code and console say that S03 stopped in the main design (S03 3.4 asks to report the stop; the reason is in sealed/stop.json); scoring a run that does not complete takes a little longer (evaluate_v7 closes the window early on completion): not hidden, not measured"
  - "the upper-end tracking flag uses the 400 001-point path grid (about 15 um); at s = S the window is cut by the end of the path, not flagged (Codex S3C-5, deferred)"
elapsed_s: {sweep_s + main_s}
elapsed_s_sweep: {sweep_s}
elapsed_s_main_design: {main_s}
file_sha256: {json.dumps(files)}
files:
  - sweep.csv: "one line per sweep run (section 3.1) and default gains at the other mu levels"
  - sweep_counts.csv: "section 5.3 for the sweep: runs not completed, inconclusive, saturated per factor, speed, mu"
  - levels.csv: "L, H, centre, scale per factor"
  - design.csv: "centre, outer box and inner box of every controller (all run, L41)"
  - logs_sweep/: "npz of the sweep runs not completed"
  - sealed/: "SEALED (L56): runs.csv (section 4 responses), effects.csv (every term, per box, response, speed, mu / worst / range), curvature.csv (5.4), y2_versions.csv (5.1), fail_counts.csv (5.3), determinism_check.csv (3.4), notes.json, stop.json, manifest.json, logs/"
""")
    for d in (work_pub, work_sealed):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    print(f'S03 finished -> {a.out}; status: {status}')
    return 0 if not (stop_public or stop_sealed) else 2


if __name__ == '__main__':
    if '--child' in sys.argv:
        child_main(sys.argv[sys.argv.index('--child') + 1])
        sys.exit(0)
    sys.exit(main())
