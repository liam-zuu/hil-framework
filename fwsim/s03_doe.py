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

S03 v8 (L67, Liam 07/11; spec section 11 item 4): 14 face-centred axial points (point_id
257-270, box 'axial', 3 mu x 3 speeds) run with the main design and go to sealed/runs.csv;
100 Latin-hypercube test points (point_id 271-370, box 'test', middle mu x 3 speeds) read
from the committed file fwsim/s03_test_points.csv (made once with --make-test-points, never
generated again) run AFTER sealed/ is closed, for both controllers, and everything about
them is in --out/sealed_test/ (runs_test.csv, logs/, notes.json, stop.json, manifest.json).
S03 does not analyse either: section 5 and the rule of 3.2 use centre, outer and inner only.
design.csv: 371 rows per controller, checked point_id - box - codes and bounds (11.4 (i),
(ii)) before the main design. Two diagnostic columns n_upper_frames, n_back_frames (section 4).

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
import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import struct
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from fractions import Fraction

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
            'cond2_5_ever', 'pos_reset_mid', 'e_perp_rms', 'e_trk_rms',
            'n_upper_frames', 'n_back_frames',      # S03 v8 section 4 (W8-Q (b)): diagnostics, not responses
            'error']                                # RMS: S3C-13; error: S3C-12
SCALE_RTOL = 1e-14            # H/L >= 4 within float rounding of the log grid (S3C-6; S3D-7: about 45 ULP)
# S03 v8 (L67): axial points and the test set for the RSM comparison of S04; not analysed here
ANALYSED_BOXES = ('centre', 'outer', 'inner')            # section 5 and the rule of 3.2 use only these
AXIAL_FIRST, TEST_FIRST, N_TEST = 257, 271, 100
TEST_POINTS = os.path.join(HERE, 's03_test_points.csv')  # section 7: the committed file is the reference
TEST_POINTS_GEN = os.path.join(HERE, 's03_test_points.gen.json')
TEST_SEED = 20261107                                      # section 3.2
TEST_COLS = ['test_id', 'point_id'] + [f'X{k}' for k in range(1, 8)]
BOUND_RTOL = 1e-12                                        # section 11 item 4 (i)
PLANNED_RUNS = dict(centre=9, outer=1152, inner=1152, axial=126, test=300)   # per controller (sections 3.2, 7)


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
    L = 12.000000000000002 for the mathematical 12 (S3C-6); the tolerance is about 45 ULP of 4,
    so a ratio really below 4 (3.999999999998) stays linear (S3D-7)."""
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


def axial_points():
    """S03 v8 section 3.2: 14 face-centred axial points (alpha = 1): factor X_k at code -1 has
    point_id 255 + 2k, at +1 256 + 2k (k = 1..7); the six others at code 0 (centre of the box)."""
    pts = []
    for k in range(1, 8):
        for sign, pid in ((-1.0, 255 + 2 * k), (1.0, 256 + 2 * k)):
            pts.append((pid, tuple(sign if j == k - 1 else 0.0 for j in range(7))))
    return pts


def _float_bits(x):
    return struct.pack('<d', float(x))


def make_test_points(path=TEST_POINTS, gen_path=TEST_POINTS_GEN):
    """Generate the test-set file once (S03 v8 sections 3.2, 7): Latin hypercube, 100 points,
    codes 2u - 1, written to a temporary file, read back and compared bit for bit (64-bit
    patterns, so +0.0 and -0.0 differ) with the binary64 array just drawn; only then renamed to
    the main file. Records the generation (command, seed, BitGenerator, versions) beside it."""
    import platform
    import scipy
    from scipy.stats import qmc
    if os.path.exists(path):
        raise S03Stop(f'{path} exists: the committed file is the reference, it is never generated again')
    u = qmc.LatinHypercube(d=7, scramble=True, optimization=None,
                           rng=np.random.default_rng(TEST_SEED)).random(N_TEST)
    codes = 2.0 * u - 1.0
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(TEST_COLS)
        for j in range(N_TEST):
            w.writerow([j + 1, TEST_FIRST - 1 + j + 1] + [fmt(codes[j, k]) for k in range(7)])
    back = read_test_points_raw(tmp)
    same = len(back) == N_TEST and all(_float_bits(back[j][2 + k]) == _float_bits(codes[j, k])
                                       for j in range(N_TEST) for k in range(7))
    if not same:
        os.remove(tmp)
        raise S03Stop('test points: the file read back differs from the drawn array (S03 v8 section 7)')
    os.replace(tmp, path)
    gen = dict(command='scipy.stats.qmc.LatinHypercube(d=7, scramble=True, optimization=None, '
                       'rng=numpy.random.default_rng(20261107)).random(100); code = 2u - 1',
               seed=TEST_SEED, bit_generator=type(np.random.default_rng(TEST_SEED).bit_generator).__name__,
               python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               created=time.strftime('%Y-%m-%d %H:%M:%S'), sha256=sha256_file(path))
    with open(gen_path, 'w', encoding='utf-8') as f:
        json.dump(gen, f, indent=1)
    return gen


def _parse_test_points(text):
    """Rows of the test-set text as [test_id(str), point_id(str), X1..X7 (float)]; header checked.
    CRLF or LF: the csv module reads both the same."""
    rd = csv.reader(io.StringIO(text, newline=''))
    head = next(rd, None)
    if head != TEST_COLS:
        raise S03Stop(f'test points: columns {head} are not {TEST_COLS} (S03 v8 section 8.7)')
    return [r[:2] + [float(x) for x in r[2:]] for r in rd]


def read_test_points_raw(path):
    """Rows of the test-set file (see _parse_test_points)."""
    with open(path, 'rb') as f:
        return _parse_test_points(f.read().decode('utf-8'))


LATIN_STEP = Fraction(1, 50)      # bin width 0.02, exact (S3V8-5)


def _latin_bin(x):
    """Index i of the bin [-1 + 0.02 (i - 1), -1 + 0.02 i) holding the binary64 value x, compared
    with the exact decimal bounds (Fraction of the binary64, no rounding, no epsilon; S3V8-5)."""
    return math.floor((Fraction(float(x)) + 1) / LATIN_STEP) + 1


def check_test_points_text(text):
    """S03 v8 section 8 condition 7 on the text of the test-set file: 100 rows, test_id 1..100 and
    point_id 271..370 in order and unique, every code finite in [-1, 1], one point of every column
    in each bin [-1 + 0.02 (i - 1), -1 + 0.02 i), i = 1..100 (exact bounds). Returns
    [(point_id, codes)]; raises S03Stop."""
    rows = _parse_test_points(text)
    if len(rows) != N_TEST:
        raise S03Stop(f'test points: {len(rows)} rows, not {N_TEST} (S03 v8 section 8.7)')
    for j, r in enumerate(rows, start=1):
        if any(len(x) != len(x.strip()) for x in r[:2]) or r[0] != str(j) or r[1] != str(TEST_FIRST - 1 + j):
            raise S03Stop(f'test points: row {j} has test_id {r[0]!r}, point_id {r[1]!r} (expected {j}, '
                          f'{TEST_FIRST - 1 + j}; S03 v8 section 8.7)')
    codes = np.array([r[2:] for r in rows], dtype=float)
    if codes.shape != (N_TEST, 7) or not np.all(np.isfinite(codes)) or np.any(np.abs(codes) > 1.0):
        raise S03Stop('test points: a code is not finite or not in [-1, 1] (S03 v8 section 8.7)')
    for k in range(7):
        n_in = Counter(_latin_bin(x) for x in codes[:, k])
        for i in range(1, N_TEST + 1):
            if n_in.get(i, 0) != 1:
                lo, hi = -1 + LATIN_STEP * (i - 1), -1 + LATIN_STEP * i
                raise S03Stop(f'test points: X{k + 1} has {n_in.get(i, 0)} points in bin {i} '
                              f'[{float(lo)}, {float(hi)}) (S03 v8 section 8.7)')
    return [(TEST_FIRST - 1 + j, tuple(float(x) for x in codes[j - 1])) for j in range(1, N_TEST + 1)]


def check_test_points(path=TEST_POINTS):
    """check_test_points_text on the file at path."""
    with open(path, 'rb') as f:
        return check_test_points_text(f.read().decode('utf-8'))


def test_points_gen_path(path):
    """The generation record beside a test-set file: <name>.csv -> <name>.gen.json."""
    return os.path.splitext(path)[0] + '.gen.json'


def _head_blob(p):
    """Bytes of the file p in the commit HEAD of its repository (git show HEAD:./<name>, run in
    the folder of p), or None when there is no such blob (not in git, not committed)."""
    try:
        r = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(p)), 'show',
                            'HEAD:./' + os.path.basename(p)], capture_output=True)
    except OSError:
        return None
    return r.stdout if r.returncode == 0 else None


def _rel_here(p):
    return _rel_or_abs(p, os.path.join(HERE, '..'))


def load_test_points(path=TEST_POINTS, trial=False):
    """S03 v8 sections 7, 8 condition 7 (S3V8-3): the bytes of the test-set file are read ONCE,
    before the main design. From those bytes: the SHA-256 of the bytes as read and of the same
    bytes with CRLF -> LF (git keeps LF, the work tree may hold CRLF), the CSV and its checks
    (check_test_points_text). The content must equal the blob of the file in HEAD, both after
    CRLF -> LF (git status alone is not evidence: assume-unchanged hides a change). The
    generation record is <name>.gen.json beside this very file; its sha256 must be one of the
    two SHA-256 above. Without --trial a failed git / generation check stops S03; with --trial it
    is recorded in 'problems'. Returns (points, info for meta.yaml)."""
    with open(path, 'rb') as f:
        raw = f.read()
    lf = raw.replace(b'\r\n', b'\n')
    info = dict(file=_rel_here(path), sha256=hashlib.sha256(raw).hexdigest(),
                sha256_lf=hashlib.sha256(lf).hexdigest(), git=file_git_state(path))
    pts = check_test_points_text(raw.decode('utf-8'))
    blob = _head_blob(path)
    info['head_blob_sha256'] = hashlib.sha256(blob).hexdigest() if blob is not None else None
    info['matches_head_blob_lf'] = blob is not None and blob.replace(b'\r\n', b'\n') == lf
    gen_path, gen = test_points_gen_path(path), None
    if os.path.isfile(gen_path):
        try:
            with open(gen_path, 'rb') as f:
                gen = json.loads(f.read().decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as exc:
            gen = dict(unreadable=f'{type(exc).__name__}: {exc}')
    info['generation_file'] = _rel_here(gen_path)
    info['generation'] = gen
    info['generation_sha256_matches'] = bool(isinstance(gen, dict) and
                                             gen.get('sha256') in (info['sha256'], info['sha256_lf']))
    problems = []
    if not (info['git']['tracked'] and info['git']['clean']):
        problems.append('not committed / modified (git status)')
    if not info['matches_head_blob_lf']:
        problems.append('content differs from the blob in HEAD (compared after CRLF -> LF)')
    if not info['generation_sha256_matches']:
        problems.append(f'generation record {info["generation_file"]} missing, unreadable or its sha256 is '
                        f'not the SHA-256 of the file read')
    info['problems'] = problems
    if problems and not trial:
        raise S03Stop(f'{path}: ' + '; '.join(problems) + ' (S03 v8 sections 7, 8.7)')
    return pts, info


def check_design_map(rows, test_pts):
    """S03 v8 section 11 item 4: point_id - box - codes of design.csv, per controller, not only the
    row count: 371 rows, ids 0..370 unique, the box of each id, the codes of centre / outer /
    inner / axial points as the design defines them, the codes of the 100 test points as in the
    test file read (test_pts: [(point_id, codes)]); codes compared bit for bit (S3V8-4)."""
    expect = {0: ('centre', (0.0,) * 7)}
    expect.update({p: ('outer', c) for p, c in design_points(1.0, 1)})
    expect.update({p: ('inner', c) for p, c in design_points(0.5, 129)})
    expect.update({p: ('axial', c) for p, c in axial_points()})
    test_ids = [p for p, _ in test_pts]
    if test_ids != list(range(TEST_FIRST, TEST_FIRST + N_TEST)):
        raise S03Stop(f'test points given to the design check are not ids {TEST_FIRST}..{TEST_FIRST + N_TEST - 1}')
    expect.update({p: ('test', c) for p, c in test_pts})
    by_ctrl = {}
    for r in rows:
        by_ctrl.setdefault(r['controller'], []).append(r)
    for ctrl, rs in by_ctrl.items():
        ids = [int(r['point_id']) for r in rs]
        if len(rs) != 371 or sorted(ids) != list(range(371)):
            raise S03Stop(f'{ctrl}: design.csv has {len(rs)} rows / ids not 0..370 unique (S03 v8 section 8.7)')
        for r in rs:
            pid, codes = int(r['point_id']), tuple(float(r[f'X{k}']) for k in range(1, 8))
            box, want = expect[pid]
            if r['box'] != box or [_float_bits(x) for x in codes] != [_float_bits(x) for x in want]:
                raise S03Stop(f'{ctrl}: point {pid} is {r["box"]} {codes}, expected {box} {want} (S03 v8 11.4)')


def check_bounds(ctrl, lv, points, gains_at):
    """S03 v8 section 11 item 4: (i) every binary64 value after decode is finite and within
    [L - 1e-12 (H - L), H + 1e-12 (H - L)]; (ii) the float32 value the firmware uses is compared
    only with the convention of section 2.1, omega_o <= 48 (not with [L, H]). Raises S03Stop."""
    for pid, _box, codes in points:
        for name, x in gains_at(codes).items():
            L, H, _ = lv[name]
            tol = BOUND_RTOL * (H - L)
            if not (math.isfinite(x) and L - tol <= x <= H + tol):
                raise S03Stop(f'{ctrl} point {pid}: {name} = {x!r} outside [{L!r}, {H!r}] (S03 v8 11.4 (i))')
            if name.startswith('omega_o') and float(np.float32(x)) > OMEGA_O_MAX:
                raise S03Stop(f'{ctrl} point {pid}: {name} float32 {float(np.float32(x))!r} > {OMEGA_O_MAX} '
                              f'(S03 v8 11.4 (ii))')


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


def npz_bytes(log):
    """The compressed npz of a log, in memory."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **log)
    return buf.getvalue()


def save_log(d, name, log, data=None):
    """Write the npz of a log (data: npz_bytes(log) already made)."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + '.npz'), 'wb') as f:
        f.write(npz_bytes(log) if data is None else data)


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
            mus = mu_levels if cond in ('worst', 'range') else [float(cond)]
            per = [[runs[(i, m, v)] for m in mus] for i in ids]
            # counts of points (S3C-13 round 3): a track error at some mu; worst Y1 inconclusive
            # (no mu false, some mu inconclusive); at one mu these are counts of runs, the same
            n_trk = sum(any(r['status'] == 'track_err' for r in rs) for rs in per)
            n_inc = sum(not any(r['completed'] == 'false' for r in rs) and
                        any(r['completed'] == 'inconclusive' for r in rs) for rs in per)
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


def _rel_or_abs(p, start):
    """Path of p relative to start with '/', or the absolute path when they are on different
    drives (Windows: relpath raises ValueError)."""
    try:
        return os.path.relpath(p, start).replace(os.sep, '/')
    except ValueError:
        return os.path.abspath(p).replace(os.sep, '/')


def file_git_state(p):
    """Git state of one file (S03 v8 section 8.7): tracked, unmodified, last commit touching it."""
    d = os.path.dirname(os.path.abspath(p))

    def git(*a):
        return subprocess.run(['git', '-C', d] + list(a), capture_output=True, text=True)
    try:
        tracked = git('ls-files', '--error-unmatch', os.path.abspath(p)).returncode == 0
        clean = tracked and git('status', '--porcelain', '--', os.path.abspath(p)).stdout.strip() == ''
        commit = git('log', '-1', '--format=%h', '--', os.path.abspath(p)).stdout.strip() or None
        return dict(tracked=tracked, clean=clean, commit=commit)
    except OSError as e:
        return dict(tracked=False, clean=False, commit=None, error=str(e))


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
    ap.add_argument('--test-points', default=TEST_POINTS,
                    help='committed test-set file (S03 v8 section 7); default fwsim/s03_test_points.csv')
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
    tp_info = None                                    # S3V8-3: set when the test file is read (once)

    def checked(ctrl, g, v, mu, ids=None, sink=None, work=None, notes_to=None):
        """One run; a run_error is recorded (S3C-12), rerun once in a new process for diagnosis
        (B8b-20), then S03 stops. ids = (point_id, box) for a main-design run (sealed). The test
        set (v8) passes its own sink (rows), work folder and notes list, all under sealed_test/."""
        row, log = safe_run(fw, path, ctrl, g, v, mu)
        if row['status'] != 'run_error':
            return row, log
        if ids is not None:
            (all_runs if sink is None else sink).append(dict(controller=ctrl, point_id=ids[0], box=ids[1],
                                                             mu=mu, v=v, **row))
        note = dict(controller=ctrl, gains=g, v=v, mu=mu, point=ids, first=row.get('error'))
        (rerun_notes if notes_to is None else notes_to).append(note)
        try:
            wdir = work if work is not None else (work_pub if ids is None else work_sealed)
            again = run_in_new_process(a.dll, [(ctrl, g, v, mu)], wdir)[0]['row']
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

    plan = {}
    if stop_public is None:
        # every design point of every controller is public before any main-design run: the points
        # follow from the public levels and the committed test file only, so design.csv does not
        # depend on a stop (S3C-3 (c)); v8: 371 rows per controller incl. axial and test points
        try:
            # S3V8-3: the file is read once here; its SHA-256 is fixed from those bytes
            test_pts, tp_info = load_test_points(a.test_points, trial=a.trial)   # section 8 condition 7
            for ctrl in a.controllers:
                lv = levels[ctrl]

                def gains_at(codes, lv=lv, ctrl=ctrl):
                    return {n: decode(c, *lv[n]) for n, c in zip(FACTORS[ctrl], codes)}
                centre_codes = (0.0,) * 7
                outer, inner, axial = design_points(1.0, 1), design_points(0.5, 129), axial_points()
                points = ([(0, 'centre', centre_codes)] + [(p, 'outer', c) for p, c in outer] +
                          [(p, 'inner', c) for p, c in inner] + [(p, 'axial', c) for p, c in axial] +
                          [(p, 'test', c) for p, c in test_pts])
                check_bounds(ctrl, lv, points, gains_at)                 # section 11 item 4 (i), (ii)
                for pid, box, codes in points:
                    design_rows.append(dict(controller=ctrl, point_id=pid, box=box,
                                            **{f'X{k + 1}': c for k, c in enumerate(codes)},
                                            **{f'{n}_value': v_ for n, v_ in gains_at(codes).items()}))
                plan[ctrl] = (gains_at, centre_codes, outer, inner, axial, test_pts)
            check_design_map(design_rows, test_pts)                      # section 11 item 4: id - box - codes
        except S03Stop as exc:
            stop_public = f'before the main design: {exc}'
            print(f'[STOP] {stop_public}', flush=True)
    if stop_public is None:
        write_csv(os.path.join(a.out, 'design.csv'), design_rows)      # S3E-2: on disk before any run
        try:
            for ctrl in a.controllers:
                gains_at, centre_codes, outer, inner, axial, _test = plan[ctrl]
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
                                # v8: the determinism points are the centre and 3 OUTER corners,
                                # chosen by box, not only by point_id (section 11 item 4)
                                if pid in det_ids and box in ('centre', 'outer'):
                                    keep_logs[(pid, mu, v)] = (g, box, row, log_digest(log))
                                    save_log_csv(sealed_logs, name, log)          # S03 3.4: log.csv
                                data = npz_bytes(log)      # S3E-1: compressed for every run, same work
                                if box == 'centre' or row['completed'] != 'true':
                                    save_log(sealed_logs, name, log, data)
                print(f'[design {ctrl}] running the main design (results sealed, L56)', flush=True)
                run_points(todo)
                # L41 (Liam 06/11, 'Chot B'): the inner box always runs, so the run time does not tell
                # whether the rule of S03 3.2 is on; it enters the analysis only when the rule is on
                run_points([(p, 'inner', c) for p, c in inner])
                # v8 (L67): axial points always run too, 3 mu x 3 speeds, sealed; S03 does not
                # analyse them (section 5 and the rule of 3.2 use the outer / inner / centre runs)
                run_points([(p, 'axial', c) for p, c in axial])
                corner_ids = [p for p, _ in outer]
                need_inner = [v for v in SPEEDS
                              if sum(runs[(i, mu_mid, v)]['completed'] == 'false' for i in corner_ids) > 64]
                inner_rule[ctrl] = need_inner
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

                # 5 analysis: both boxes at every speed always (S3E-1: the same work whether the rule
                # of 3.2 is on or not); kept: the inner box only when the rule is on, no effects /
                # curvature / Y2 versions at a speed where the inner box is also mostly not completed
                nc = no_concl[ctrl]
                for box, pts in (('outer', outer), ('inner', inner)):
                    e, c, vr, f = analyse(ctrl, box, pts, runs, mu_levels, SPEEDS, y3_ref, notes)
                    if box == 'inner' and not need_inner:
                        continue
                    eff_rows += [r for r in e if r['v'] not in nc]
                    curv_rows += [r for r in c if r['v'] not in nc]
                    ver_rows += [r for r in vr if r['v'] not in nc]
                    fail_rows += f
        except S03Stop as exc:
            stop_sealed = str(exc)
            print('[STOP] S03 stopped in the main design; the reason is in sealed/stop.json (L56)', flush=True)

    t_end = time.time()
    # S3V8-1: the empty work folders of the sweep and of the main design are removed BEFORE sealed/
    # is written and closed; after the manifest only sealed_test/ changes
    for d in (work_pub, work_sealed):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
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
                       run_error_reruns=[n for n in rerun_notes if n['point'] is not None],
                       elapsed_s_total=round(t_end - t0),
                       elapsed_s_main_design=round(t_end - t_sweep_end) if t_sweep_end else None),
                  f, indent=1, default=str)
    sealed_files['notes.json'] = sha256_file(os.path.join(sealed, 'notes.json'))
    if stop_sealed:
        with open(os.path.join(sealed, 'stop.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(stop=stop_sealed), f, indent=1)
    with open(os.path.join(sealed, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(files=sealed_files), f, indent=1)
    # sealed/ is closed here: nothing below writes in it (S03 v8 section 7)

    # v8 (L67, S38-8): the test set of both controllers runs only now, after sealed/ is written,
    # so sealed/ does not depend on it; everything about it stays in sealed_test/
    stop_test, test_rows, test_notes, test_s = None, [], [], None
    sealed_test = os.path.join(a.out, 'sealed_test')
    work_test = os.path.join(sealed_test, '_work')
    if stop_public is None and stop_sealed is None:
        os.makedirs(sealed_test, exist_ok=True)
        test_logs = os.path.join(sealed_test, 'logs')
        t_test0 = time.time()
        print('[test set] running the test points (results sealed in sealed_test/, S03 v8 section 7)', flush=True)
        # S3V8-2: every warning, print and traceback of the test phase goes to sealed_test/diag.txt,
        # any exception of the phase is a stop recorded in sealed_test/stop.json; the console only
        # says that S03 stopped in the test set (the DLL does not print: fwsim/shim/esp_log.h)
        diag = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(diag), \
                contextlib.redirect_stderr(diag):
            warnings.simplefilter('always')
            try:
                for ctrl in a.controllers:
                    gains_at, test_pts = plan[ctrl][0], plan[ctrl][5]
                    for pid, codes in test_pts:
                        g = gains_at(codes)
                        for v in SPEEDS:
                            row, log = checked(ctrl, g, v, mu_mid, ids=(pid, 'test'), sink=test_rows,
                                               work=work_test, notes_to=test_notes)
                            test_rows.append(dict(controller=ctrl, point_id=pid, box='test', mu=mu_mid, v=v,
                                                  **row))
                            data = npz_bytes(log)             # same work for every run (S3E-1)
                            if row['completed'] != 'true':    # logs kept by the rule of section 7
                                save_log(test_logs, f'{ctrl}_p{pid}_mu{mu_mid}_v{v}', log, data)
            except S03Stop as exc:
                stop_test = str(exc)
            except Exception as exc:          # noqa: BLE001 (S3V8-2: any failure of the phase stays sealed)
                stop_test = f'{type(exc).__name__}: {exc}'
                traceback.print_exc(file=diag)
        if stop_test:
            print('[STOP] S03 stopped in the test set; the reason is in sealed_test/stop.json', flush=True)
        test_s = round(time.time() - t_test0)
        if os.path.isdir(work_test) and not os.listdir(work_test):    # before sealed_test/manifest.json
            os.rmdir(work_test)
        test_files = {}
        with open(os.path.join(sealed_test, 'diag.txt'), 'w', encoding='utf-8') as f:
            f.write(diag.getvalue())
            for w_ in caught:
                f.write(f'warning: {w_.category.__name__}: {w_.message} ({w_.filename}:{w_.lineno})\n')
        test_files['diag.txt'] = sha256_file(os.path.join(sealed_test, 'diag.txt'))
        p = os.path.join(sealed_test, 'runs_test.csv')
        write_csv(p, test_rows, RUN_COLS)
        test_files['runs_test.csv'] = sha256_file(p)
        with open(os.path.join(sealed_test, 'notes.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(stop=stop_test, rows_run=len(test_rows), run_error_reruns=test_notes,
                           elapsed_s_test=test_s), f, indent=1, default=str)
        test_files['notes.json'] = sha256_file(os.path.join(sealed_test, 'notes.json'))
        if stop_test:
            with open(os.path.join(sealed_test, 'stop.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(stop=stop_test), f, indent=1)
            test_files['stop.json'] = sha256_file(os.path.join(sealed_test, 'stop.json'))
        with open(os.path.join(sealed_test, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(files=test_files), f, indent=1)
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
    status = ('stopped in the sweep: ' + stop_public if stop_public else
              'stopped in the main design (reason sealed)' if stop_sealed else
              'stopped in the test set (reason in sealed_test/)' if stop_test else 'ran to the end')
    # S3V8-3: what was read before the main design, not a second read of the file now
    tp_meta = tp_info if tp_info is not None else dict(file=_rel_here(a.test_points),
                                                         read='no (S03 stopped before the test file was read)')
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: S03 v8 (06_SPECS/2026-11-05_Spec_S03_DOE_chon_gain.md)
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
planned_runs_per_controller: {json.dumps(PLANNED_RUNS)}   # S03 v8 3.2, 7 (rows actually run: sealed/, sealed_test/)
test_points: {json.dumps(tp_meta)}   # S03 v8 3.2, 7, 8.7 (S3V8-3): read once before the main design; sha256 of the bytes read, sha256_lf after CRLF -> LF, matches_head_blob_lf, generation from <name>.gen.json beside the file
test_set: {json.dumps('not run (S03 stopped before it)' if (stop_public or stop_sealed) else 'sealed in sealed_test/ (not looked at before the RSM coefficients are saved, S03 v8 section 7)')}
interpretations:   # of this script, not written in S03 v8 / O1 v7; Liam or THESIS correct if wrong
  - "completion: conditions 1, 5 at the instant, conditions 2 and 3 together as the trailing unbroken t_d window (condition 2 over the window: L63 (2), Liam 07/11), condition 4 strictly before the deadline; after a tracking error that came before the progress, the window runs to the deadline (o1_feasibility.evaluate_v7)"
  - "sweep level rule: a point is 'completing' only with y1 'true' at every gate speed (inconclusive breaks the run)"
  - "inner box trigger: corners with y1 'false' only (inconclusive not counted)"
  - "a stop of the spec (3.1 (i), (ii), 3.4, run_error) ends S03 for both controllers; the default gains of every controller are checked first"
  - "logs: log.csv (%.17g) of the four determinism points and npz (exact bits) of the centre and every main-design run not completed, all in sealed/logs/; npz of every sweep run not completed in logs_sweep/ (public, as the sweep)"
  - "the inner box (codes +-0.5) always runs for every controller (L41, Liam 06/11 'Chot B'), so the number of runs does not depend on the results; both boxes are always analysed (the same work), and the inner box is kept (effects, Y2 versions, fail counts; curvature is for the outer box only; at every speed, as before L41) only when the rule of S03 3.2 is on at one speed or more (> 64/128 corners 'false' at the middle mu); otherwise its runs are only rows of sealed/runs.csv"
  - "elapsed_s here is the sweep only; the total and the main-design time are in sealed/notes.json (L56, Codex S3E-1; S03 section 7 asks elapsed_s in meta.yaml)"
  - "public channels left (L56, S3C-3, S3E-1; Liam 06/11 'Chot B', 'Duyet khuyen nghi'): status, exit code and console say that S03 stopped in the main design (S03 3.4 asks to report the stop; the reason is in sealed/stop.json); the wall-clock time still varies a little with the results (scoring a run that does not complete is a little longer, evaluate_v7 closes the window early on completion; writing the npz of the runs not completed; machine noise): accepted, not measured"
  - "the upper-end tracking flag uses the 400 001-point path grid (about 15 um); at s = S the window is cut by the end of the path, not flagged (Codex S3C-5, deferred)"
  - "v8: run order per controller centre, outer, inner, axial (3 mu x 3 speeds each), then determinism and analysis; sealed/ is written and closed after every controller; then the test set of every controller (middle mu x 3 speeds); a stop before the test set means the test set does not run"
  - "v8: test-set logs follow the rule of section 7 (npz of every run not completed; there is no centre point in the test set); the compressed npz is made for every run (same work)"
  - "v8: the test-points file is read once before the main design (S3V8-3); 'committed' (section 8.7) = tracked, unmodified for git status AND the bytes read match the blob in HEAD after CRLF -> LF of both; its generation record is <name>.gen.json beside the file read, with the sha256 of the bytes read or of their LF form; with --trial a failed check is allowed and recorded in test_points.problems"
  - "v8 (S3V8-2): warnings, prints and tracebacks of the test phase go to sealed_test/diag.txt and any exception of the phase is a stop recorded in sealed_test/stop.json; the same channel in the main design (warnings to the console) is left as before (deferred, Codex S3V8-2)"
  - "v8: n_upper_frames, n_back_frames count the frames evaluate_v7 kept for scoring (t_k <= t_end), as section 4 defines them; they are diagnostics, never responses"
elapsed_s: {sweep_s}
file_sha256: {json.dumps(files)}
files:
  - sweep.csv: "one line per sweep run (section 3.1) and default gains at the other mu levels"
  - sweep_counts.csv: "section 5.3 for the sweep: runs not completed, inconclusive, saturated per factor, speed, mu"
  - levels.csv: "L, H, centre, scale per factor"
  - design.csv: "371 rows per controller: centre, outer box, inner box, axial points, test points (all run, L41, L67)"
  - logs_sweep/: "npz of the sweep runs not completed"
  - sealed/: "SEALED (L56): runs.csv (section 4 responses; centre, outer, inner, axial), effects.csv (every term, per box, response, speed, mu / worst / range; outer and inner only), curvature.csv (5.4), y2_versions.csv (5.1), fail_counts.csv (5.3), determinism_check.csv (3.4), notes.json, stop.json, manifest.json, logs/"
  - sealed_test/: "SEALED separately (S03 v8 section 7): runs_test.csv, logs/, notes.json, stop.json, diag.txt (warnings, prints, traceback of the test phase), manifest.json; opened only after the RSM coefficients at the middle mu are saved and committed"
""")
    print(f'S03 finished -> {a.out}; status: {status}')
    return 0 if not (stop_public or stop_sealed or stop_test) else 2


if __name__ == '__main__':
    if '--child' in sys.argv:
        child_main(sys.argv[sys.argv.index('--child') + 1])
        sys.exit(0)
    if '--make-test-points' in sys.argv:          # S03 v8 section 7: once, then commit the file
        print(json.dumps(make_test_points(), indent=1))
        sys.exit(0)
    sys.exit(main())
