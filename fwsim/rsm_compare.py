"""
rsm_compare.py: the one-step RSM comparison of spec S04 v6.4 section 12 (CLAUDE_THESIS
06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md; design 03_CODE/docs/2026-11-15_Thiet_ke_CODE_RSM_v1.md,
Liam 15/11/2026 CODE_RSM "Duyet, code luon", "Script tu commit"). Reads a finished S04 folder (main
runs and post checks) and the S03 run it used; writes <S04>/rsm/ (S04 section 8) in three phases
(section 12.7):

  fit   12.2  input checks; full quadratic in the 7 coded gains (36 terms, least squares) on the 143
              face-centred CCD points (centre 0, outer corners 1-128, axial 257-270) of
              S03 sealed/runs.csv; f1, f2 as S04 section 3 (gate speeds x evaluation mu of the S04
              lock.yaml); a middle-mu pair; variants no_flag, cens05 (f1 cut at 0.5 m),
              drop_axial_<point_id>; rsm/fit/ and rsm/fit/lock.yaml.
  opt   12.3  two SLSQP steps on the model from the 143 CCD points (min f1_hat; min f2_hat with
              f1_hat <= f1_hat* + 0.01 m and f1_hat <= D), candidate list (eps 1e-6, duplicates
              closer than 1e-4, order f2_hat, f1_hat, codes); 12.4 confirmation on fwsim of at most
              min(5, M) candidates with the GA evaluator (s04_ga.Evaluator: same runs, scoring,
              run_error rule), labels of every tried candidate; mu3 (x_RSM at the low and high mu
              when the GA used the middle one); 12.5 classes R0-R3; 12.6 GA visited points, axial
              sensitivity, cost; rsm/opt_lock.yaml.
  test  12.6  opens S03 sealed_test/ (only now): RMSE, largest error, misclassification of the
              middle-mu model on the 100 test points; rsm/test/.

Every phase first appends a start row to rsm/attempts.csv and commits it in the git repository
that holds the S04 folder (Liam 15/11 "Script tu commit": the script commits only its own paths,
git commit --only, never pushes), and stops unless that row is in HEAD before any data is read;
it ends with an end row (done | stopped) committed with the phase outputs. A phase folder that
already holds files stops the run (no overwrite); --abort-previous "<reason>" moves it to
rsm/aborted/<time>/.

The mode follows the S04 folder: lock.yaml common.tool_check true is a tool check (never main
numbers, L82); a tool check only reads an S03 folder whose meta.yaml has 'rsm_fake: true'
(S04 v6 C Q2: no sealed/ of a real S03 run). Nothing here runs on the car or the bench.

  python fwsim/rsm_compare.py --phase fit  --s04 05_DATA/results/S04/<run>
  python fwsim/rsm_compare.py --phase opt  --s04 05_DATA/results/S04/<run>
  python fwsim/rsm_compare.py --phase test --s04 05_DATA/results/S04/<run>
Interpretations of this script, not written in S04 v6.4, are in INTERPRETATIONS (rsm/meta.yaml).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim                 # noqa: E402
import s03_doe as s3         # noqa: E402
import s04_ga as s4          # noqa: E402

SPEC = 'S04 v6.4 section 12 (06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md)'
# settings of phases 1-2, constants of the script, locked by its commit before any data is read (12.7)
SOLVER = dict(method='SLSQP', ftol=1e-9, maxiter=500)      # 12.3
EPS = 1e-6                    # m, one tolerance everywhere in 12.3-12.4
DEDUP = 1e-4                  # Euclidean distance in code space: closer candidates are one (12.3)
K = 5                         # at most min(K, M) candidates confirmed (12.4)
CENS = 0.5                    # m, f1 cut of the cens05 sensitivity fit (12.2)
F2_REL = 0.2                  # R2 tolerance on f2 (12.5)
D = s4.D
F1_TOL = s4.F1_TOL
THETA_MAX_DEG = s4.THETA_MAX_DEG
N_TERMS = 36
CENTRE_ID = 0
OUTER_IDS = tuple(range(1, 129))
AXIAL_IDS = tuple(range(257, 271))
FIT_IDS = (CENTRE_ID,) + OUTER_IDS + AXIAL_IDS           # 143 points (12.2)
FIT_BOXES = ('centre', 'outer', 'axial')
TEST_IDS = tuple(range(s3.TEST_FIRST, s3.TEST_FIRST + s3.N_TEST))
USABLE_STATUS = ('ok', 'track_err')
USABLE_COMPLETED = ('true', 'false', 'inconclusive')
S03_PUBLIC = ['meta.yaml', 'design.csv', 'levels.csv']
S03_SEALED = ['sealed/runs.csv', 'sealed/notes.json']
S03_SEALED_TEST = ['sealed_test/runs_test.csv', 'sealed_test/notes.json']
CLIP_TOL = 1e-9               # RSM-15: largest move of an SLSQP solution back into the box (rounding only)
PHASES = ('fit', 'opt', 'test')
RUN_LOCK_PREFIX = 'rsm_compare_'   # RSM4-2: name of the run lock (Windows mutex) of an rsm/ folder
WRITTEN = set()               # RSM-14: files written by the running phase (absolute paths)
PHASE_DIRS = dict(fit=('fit',), opt=('opt', 'confirm', 'check'), test=('test',))
PHASE_LOCK = dict(fit='fit/lock.yaml', opt='opt_lock.yaml', test=None)
TOOL_CAUSE = 'tool:'          # evals.csv 'cause' prefix with which CODE marks a first-attempt tool error (S04 3)
ATT_COLS = ['attempt_id', 'chain_id', 'kind', 'phase', 'time', 'script_commit', 'script_dirty', 'inputs_sha256',
            'status', 'reason']
COST_COLS = ['controller', 'part', 'item', 'n_runs_planned', 'n_runs_actual', 'n_attempt2', 'machine_s',
             'machine_s_basis', 'covers', 'row_type', 'source_file']
EVAL_RSM_COLS = ['controller', 'phase', 'cand_id', 'ga_seed', 'gen', 'v', 'mu', 'attempt'] + \
    [c for c in s4.EVAL_COLS if c not in ('controller', 'ga_seed', 'gen', 'cand_id', 'v', 'mu', 'attempt')] + \
    ['gain_source']
INTERPRETATIONS = [
    'script fwsim/rsm_compare.py (S04 v6 C Q1); it imports s04_ga and s03_doe, edits neither, and is not in '
    's04_ga.SOURCES, so the S04 fingerprint does not change',
    'mode follows the S04 lock.yaml common.tool_check; a tool check reads only an S03 folder whose meta.yaml has '
    'rsm_fake: true (C Q2)',
    'f1, f2 of a CCD point: the same aggregation as s04_ga.score over the S04 lock.yaml speeds x eval_mu; the '
    'middle-mu pair over the same speeds at the middle S03 mu level',
    'n_upper_frames, n_back_frames of a point: summed over the runs of the main model',
    'fits: main (base, no_flag, cens05, drop_axial_<point_id>) and mid (base only); when eval_mu is the middle '
    'mu the two base fits are the same model and are checked equal',
    'rank: numpy.linalg.matrix_rank with its default tolerance; a fit of rank < 36 has no coefficients',
    'SLSQP with the analytic gradient; converged = scipy OptimizeResult.success with a finite solution whose move '
    'back into [-1, 1]^7 is at most 1e-9 (rounding; RSM-15); f1_hat, f2_hat on the clipped point',
    'duplicates: greedy in the sorted order, a candidate within 1e-4 (Euclidean, codes) of a kept one is dropped '
    '(Codex RSM-15 on chains A-B-C: kept, deterministic)',
    'code to gain: s03_doe.decode on levels.csv; bounds: s03_doe.check_bounds; edge flag and level source: '
    's04_ga.edge_report on the GA coordinate lo + (c + 1) / 2 (hi - lo)',
    'confirmation: s04_ga.Evaluator.candidate (run_gate, run_error once more in a new process, cause empty, '
    'score); a candidate is feasible when score gives status ok, g1 = 0 and g2..g5 <= 0',
    'R2 is tested first; else s04_ga.select on {x_GA, x_RSM} gives R1 (x_GA) or R3 (x_RSM)',
    'not robust over mu (RSM-10): the 12.4 (iii) labels over every attempt-1 row of the low and high mu: an '
    'observed violation -> not robust; else run_error / penalty -> undetermined; else inconclusive / track_err -> '
    '"inconclusive over mu" (g1 convention, not observed; RSM2-5); x_GA from the low / high rows of '
    'post_checks/three_mu.csv (stored pen flags kept, missing ones from s04_ga.run_values); per-mu labels reported',
    'GA visited points: evals_<controller>.csv of the main runs only (S04 4.3: the trial is not a result); '
    'attempt 2 replaces a run_error only when the cause on the attempt-2 row starts with "tool:" (CODE writes it '
    'when it found a tool error; S04 8 puts cause on the rerun) and it is ok / track_err; a penalised quantity '
    '(computed or stored pen_* = 1) leaves the candidate without a label; duplicates by gain vector, first kept',
    'cost: S03 sweep not in part (a) (12.6 counts the 2 313 design runs); n_runs_actual counts the S03 diagnosis '
    'reruns too; confirm planned = min(K, M) x runs per candidate; S04 post and S03 times are one shared row '
    '(controller all)',
    'test set: 100 points x the S04 lock speeds at the middle mu; feasible at the middle mu = s04_ga.score '
    'feasible on those runs; predicted feasible = f1_hat_mid <= D + eps',
    'main comparison (RSM-4): the chain whose test start is the first test start of attempts.csv and whose fit ran '
    'on the script commit of the first fit start; after a test start every run of any phase is exploratory',
    'start row (RSM-1): SHA-256 of the public S03 files and of S04, and of the sealed files as the S04 lock.yaml '
    'fingerprint recorded them; the sealed files (and sealed_test/ in phase 3) are hashed after the start commit, '
    'checked against that record, and written in the end row and the locks',
    'locks (RSM-3, RSM2-3): every locked file must be the blob of the output commit of the attempt that wrote it: '
    'the one commit that brings its end/done row into the history of HEAD (its attempts.csv holds the row, no '
    'parent\'s does; two such commits stop, RSM4-1; message checked on top); '
    'opt_lock.yaml records the SHA-256 of fit/lock.yaml; every committed version of attempts.csv in the history of '
    'HEAD (git log --full-history: the side branches of a merge too) must be a prefix of the HEAD version (RSM2-2, '
    'RSM3-1)',
    'main run (RSM-2, RSM-12): the repository is ROOT_PATH (the thesis root folder) with origin '
    'liam-zuu/hil-thesis, the S04 files read are committed '
    'unchanged, rsm_compare.py is its HEAD blob, the DLL SHA-256 is the S04 one before loading and at the end',
    'one run at a time (RSM-14, RSM3-3, RSM4-2): an operating-system lock that opens no file: on Windows a named '
    'mutex Global\\rsm_compare_<SHA-256 of the real rsm/ path>, on POSIX flock on the rsm/ folder itself (O_DIRECTORY, '
    'O_NOFOLLOW); held for the whole phase, freed by the system when the process ends however it ends; a phase, a '
    'stop and an abort commit '
    'only the files the run wrote or that were committed (RSM2-4); paths are checked before they are opened '
    '(RSM2-1): no symlink, no junction or other name-surrogate reparse point (RSM4-4, RSM5-2), no file with a '
    'second hard link, no '
    'directory entry that is a link or a junction under a folder walked (RSM3-2)',
    'file SHA-256 in the locks and attempts.csv: of the bytes as written and of the bytes with CRLF -> LF '
    '(compared: the LF value, so a checkout with core.autocrlf keeps the lock valid)',
    'no x_GA (send_to_HIL infeasible in S04) stops the run: 12.5 does not define the classes without it',
]

EVALUATOR = None              # tests replace it; None = s04_ga.Evaluator on fwsim.FirmwareSim(dll)


class Stop(Exception):
    """A stop the spec orders (input check, lock, tool): report, do not go on."""


# ── small helpers ────────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols=None):
    """CSV with LF line ends (lock SHA-256 independent of core.autocrlf after the LF normalisation)."""
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    WRITTEN.add(os.path.abspath(path))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows({k: s3.fmt(r.get(k, '')) for k in cols} for r in rows)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    WRITTEN.add(os.path.abspath(path))
    with open(path, 'w', newline='\n', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str, ensure_ascii=False)
        f.write('\n')


def write_yaml_lines(path, d, head):
    """key: <json> lines (valid YAML), as s04_ga.write_root_meta."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    WRITTEN.add(os.path.abspath(path))
    with open(path, 'w', newline='\n', encoding='utf-8') as f:
        f.write(f'# {head}\n')
        for k, v in d.items():
            f.write(f'{k}: {json.dumps(v, ensure_ascii=False, default=str)}\n')


def file_sha(path):
    """(sha256 of the bytes, sha256 of the bytes with CRLF -> LF)."""
    with open(path, 'rb') as f:
        b = f.read()
    return hashlib.sha256(b).hexdigest(), hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest()


def fnum(x):
    """float(x) when finite, else None."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def is_on(x):
    return str(x).strip().lower() in ('1', '1.0', 'true')


NAME_SURROGATE = 0x20000000   # RSM5-2: reparse tag bit of a point that names another file (symlink, junction)


def is_dir_link(p):
    """RSM3-2, RSM4-4, RSM5-2: p is a symlink, a junction (os.path.isjunction, Python 3.12) or, on any Python
    version, a Windows reparse point whose tag is a name surrogate (os.lstat st_reparse_tag: symlinks, junctions);
    a storage reparse point (deduplication, cloud placeholder) names no other file and is not refused."""
    if os.path.islink(p) or getattr(os.path, 'isjunction', lambda q: False)(p):
        return True
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return bool(getattr(st, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                and getattr(st, 'st_reparse_tag', 0) & NAME_SURROGATE)


def files_under(root, rel_dirs):
    """Relative (posix) paths of every file under root/<rel_dir> for each dir, sorted. A directory entry that is a
    link or a junction stops (RSM3-2: os.walk neither lists it as a file nor goes into a directory symlink)."""
    out = []
    for d in rel_dirs:
        base = os.path.join(root, d)
        if os.path.isfile(base):
            out.append(d.replace(os.sep, '/'))
            continue
        for dp, dns, fs in os.walk(base):
            for n in dns:
                if is_dir_link(os.path.join(dp, n)):
                    raise Stop(f'{os.path.join(dp, n)} is a directory link or junction: refused (RSM3-2)')
            for n in fs:
                out.append(os.path.relpath(os.path.join(dp, n), root).replace(os.sep, '/'))
    return sorted(out)


def hash_tree(root, rel_dirs, skip=()):
    return {p: dict(zip(('sha256', 'sha256_lf'), file_sha(os.path.join(root, p))))
            for p in files_under(root, rel_dirs) if p not in skip}


# ── quadratic model (12.2) ───────────────────────────────────────────────────────────

def term_names(k=7):
    t = ['1'] + [f'X{i}' for i in range(1, k + 1)] + [f'X{i}^2' for i in range(1, k + 1)]
    return t + [f'X{i}*X{j}' for i in range(1, k + 1) for j in range(i + 1, k + 1)]


TERMS = term_names()


def design_matrix(C):
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n, k = C.shape
    cols = [np.ones(n)] + [C[:, i] for i in range(k)] + [C[:, i] ** 2 for i in range(k)]
    cols += [C[:, i] * C[:, j] for i in range(k) for j in range(i + 1, k)]
    return np.column_stack(cols)


def quad_parts(beta, k=7):
    """(b0, b, Q) with f(c) = b0 + b.c + c' Q c (Q symmetric)."""
    beta = np.asarray(beta, dtype=float)
    b0, b = beta[0], beta[1:1 + k]
    Q = np.diag(beta[1 + k:1 + 2 * k]).astype(float)
    idx = 1 + 2 * k
    for i in range(k):
        for j in range(i + 1, k):
            Q[i, j] = Q[j, i] = beta[idx] / 2.0
            idx += 1
    return b0, b, Q


def predict(beta, c):
    return float((design_matrix(c) @ np.asarray(beta, dtype=float))[0])


def gradient(beta, c):
    _, b, Q = quad_parts(beta)
    return b + 2.0 * Q @ np.asarray(c, dtype=float)


def fit_ls(C, y):
    """Least squares on the full quadratic: (beta or None, diag)."""
    C = np.asarray(C, dtype=float).reshape(-1, 7)
    if C.shape[0] == 0:                                 # RSM-6: every point dropped (no_flag)
        return None, dict(n=0, rank=0, sv_max=None, sv_min=None)
    X = design_matrix(C)
    rank = int(np.linalg.matrix_rank(X))
    sv = np.linalg.svd(X, compute_uv=False)
    diag = dict(n=int(X.shape[0]), rank=rank, sv_max=float(sv.max()), sv_min=float(sv.min()))
    if rank < N_TERMS:
        return None, diag
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    diag['rms_resid'] = float(np.sqrt(np.mean((X @ beta - np.asarray(y, dtype=float)) ** 2)))
    return beta, diag


# ── git ──────────────────────────────────────────────────────────────────────────────

def git(repo, *a, check=True):
    # RSM-5: every path argument is a literal path, never a pattern
    r = subprocess.run(['git', '--literal-pathspecs', '-C', repo] + list(a), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise Stop(f'git {" ".join(a)} failed in {repo}: {r.stderr.strip()[-500:]}')
    return r.stdout


def repo_root(path):
    return os.path.normpath(git(path, 'rev-parse', '--show-toplevel').strip())


def blob_at(repo, rev, rel):
    r = subprocess.run(['git', '-C', repo, 'rev-parse', '--verify', '--quiet', f'{rev}:{rel}'],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def head_text(repo, rel):
    return head_text_at(repo, 'HEAD', rel)


def head_text_at(repo, rev, rel):
    """Text of <rev>:<rel> (LF), None when the file is not in that commit."""
    r = subprocess.run(['git', '-C', repo, 'show', f'{rev}:{rel}'], capture_output=True)
    return r.stdout.decode('utf-8').replace('\r\n', '\n') if r.returncode == 0 else None


def same_as(repo, rev, rel):
    """The working file (after git's filters) is the blob of <rev>:<rel>."""
    b = blob_at(repo, rev, rel)
    return b is not None and git(repo, 'hash-object', '--', rel).strip() == b


def same_as_head(repo, rel):
    return same_as(repo, 'HEAD', rel)


def commit_paths(repo, rels, msg):
    """git add + git commit --only of exactly these paths (literal); the staged change set of these paths and
    the commit must hold no other path (RSM-5)."""
    if not rels:
        return None
    git(repo, 'add', '--', *rels)
    staged = {p for p in git(repo, 'diff', '--cached', '--name-only', '--', *rels).splitlines() if p.strip()}
    if not staged <= set(rels):
        raise Stop(f'staged paths {sorted(staged - set(rels))} are not this script\'s (nothing committed)')
    git(repo, 'commit', '--only', '-q', '-m', msg, '--', *rels)
    got = {p for p in git(repo, 'show', '--name-only', '--format=', 'HEAD').splitlines() if p.strip()}
    if not got <= set(rels):
        raise Stop(f'commit {msg!r} holds paths that are not this script\'s: {sorted(got - set(rels))}')
    return git(repo, 'rev-parse', 'HEAD').strip()


def attempts_versions(inp):
    """[(commit, text)] of every committed version of rsm/attempts.csv in the history of HEAD, oldest first (read
    through git, LF). RSM3-1: --full-history keeps the side branches of a merge that git's history simplification
    would drop (a merge that takes one parent's attempts.csv); --topo-order puts every parent before its children."""
    path = rel(inp, 'attempts.csv')
    out = []
    for c in git(inp.repo, 'log', '--full-history', '--topo-order', '--reverse', '--format=%H', '--', path).split():
        r = subprocess.run(['git', '-C', inp.repo, 'show', f'{c}:{path}'], capture_output=True)
        out.append((c, r.stdout.decode('utf-8').replace('\r\n', '\n') if r.returncode == 0 else None))
    return out


def has_done(text, phase, aid):
    return any(r['attempt_id'] == aid and r['kind'] == 'end' and r['phase'] == phase and r['status'] == 'done'
               for r in csv.DictReader(io.StringIO(text or '')))


def output_commit(inp, phase, aid):
    """RSM2-3, RSM4-1: the output commit of attempt <aid> of <phase> (12.7) is the commit that brings its end/done
    row into the history of HEAD: its attempts.csv holds the row and no parent's does. Two such commits (two
    branches that each added the row, then merged) stop: the order of git log does not choose between them. Its
    message must name that attempt (a check added on top)."""
    path = rel(inp, 'attempts.csv')
    intro = []
    for c, text in attempts_versions(inp):
        if not has_done(text, phase, aid):
            continue
        parents = git(inp.repo, 'rev-list', '--parents', '-n', '1', c).split()[1:]
        if not any(has_done(head_text_at(inp.repo, p, path), phase, aid) for p in parents):
            intro.append(c)
    if not intro:
        raise Stop(f'no commit holds the done row of {phase} attempt {aid}')
    if len(intro) > 1:
        raise Stop(f'more than one commit brings the done row of {phase} attempt {aid} into the history '
                   f'({", ".join(c[:10] for c in intro)}): no single output commit (RSM4-1)')
    c = intro[0]
    msg = git(inp.repo, 'log', '-1', '--format=%s', c)
    if f'rsm_compare {phase} done {inp.rel} ({aid})' not in msg:
        raise Stop(f'commit {c[:10]} adds the done row of {aid} but is not its output commit ({msg!r})')
    return c


ROOT_ORIGIN = ('liam-zuu/hil-thesis', 'liam-zuu/hil-thesis.git')   # RSM-2: the thesis root repository
ROOT_PATH = r'D:\CLAUDE_THESIS'                                      # RSM-2: the root repository folder (12.7)


def check_root_repo(inp):
    """Main run (RSM-2): the repository is the thesis root (origin liam-zuu/hil-thesis) and the S04 files read
    are committed as they are."""
    if os.path.normcase(os.path.realpath(inp.repo)) != os.path.normcase(os.path.realpath(ROOT_PATH)):
        raise Stop(f'main RSM runs only in {ROOT_PATH} (this repository: {inp.repo})')
    url = git(inp.repo, 'remote', 'get-url', 'origin', check=False).strip().replace('\\', '/')
    if not url.endswith(ROOT_ORIGIN) or url.endswith('hil-thesis-cad') or url.endswith('hil-thesis-cad.git'):
        raise Stop(f'main RSM runs only in the thesis root repository (origin {url!r} is not liam-zuu/hil-thesis)')
    for p in inp.s04_paths():
        if not same_as_head(inp.repo, f'{inp.rel}/{p}'):
            raise Stop(f'S04 file {p} is not committed as it is: commit the S04 run before RSM (RSM-2)')


def check_script_blob(hil):
    """Main run (RSM-12): this script's content is its HEAD blob (assume-unchanged cannot hide an edit)."""
    if not same_as_head(hil, 'fwsim/rsm_compare.py'):
        raise Stop('fwsim/rsm_compare.py differs from its HEAD blob: 12.7 locks a clean commit')


# ── inputs and checks (section 3 of the design; 12.2 S46-5) ─────────────────────────

def refuse_link(p, s03=None):
    """RSM-13, RSM2-1, RSM3-2: p (if it exists) is no link or junction, resolves to itself, is no file with a second
    hard link (a hard link has islink false and realpath unchanged), and not into <s03>/sealed_test/; checked from the
    path entries, before p is opened."""
    if not os.path.exists(p) and not os.path.islink(p):
        return
    real = os.path.normcase(os.path.realpath(p))
    if is_dir_link(p) or real != os.path.normcase(os.path.abspath(p)):
        raise Stop(f'{p} is a link or resolves to {real}: refused (RSM-13)')
    st = os.lstat(p)
    if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
        raise Stop(f'{p} has {st.st_nlink} hard links: refused (RSM3-2)')
    if s03 is not None:
        st = os.path.normcase(os.path.realpath(os.path.join(s03, 'sealed_test')))
        if real == st or real.startswith(st + os.sep):
            raise Stop(f'{p} resolves into sealed_test/: refused before phase 3 (RSM-13)')


RSM_PATHS = ['attempts.csv', 'meta.yaml', 'opt_lock.yaml', 'fit', 'opt', 'confirm', 'check', 'test', 'aborted']


class Inputs:
    """S04 folder, its lock, the S03 run, the repository and the mode."""

    def __init__(self, s04_dir, dll):
        self.s04 = os.path.normpath(os.path.abspath(s04_dir))
        self.dll = os.path.abspath(dll)
        if not os.path.isfile(os.path.join(self.s04, 'lock.yaml')):
            raise Stop(f'{self.s04}/lock.yaml missing: not a finished S04 folder')
        self.repo = repo_root(self.s04)
        rel = os.path.relpath(self.s04, self.repo).replace(os.sep, '/')
        parts = rel.split('/')
        if len(parts) != 4 or parts[:3] != ['05_DATA', 'results', 'S04']:
            raise Stop(f'{self.s04} is not <repo>/05_DATA/results/S04/<run> (12.7: one canonical folder)')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', parts[3]):
            raise Stop(f'S04 run name {parts[3]!r}: letters, digits, _ . - only (RSM-5)')
        self.rel = rel
        self.out = os.path.join(self.s04, 'rsm')
        self.rel_out = rel + '/rsm'
        refuse_link(os.path.join(self.s04, 'lock.yaml'))              # RSM2-1: before the first open
        self.lock = json.load(open(os.path.join(self.s04, 'lock.yaml'), encoding='utf-8'))
        self.common = self.lock.get('common')
        if not self.common:
            raise Stop(f'{self.s04}/lock.yaml has no common block')
        self.tool_check = bool(self.common['tool_check'])
        self.s03 = self.common['s03_run']
        self.mu_levels = [float(m) for m in self.lock.get('s03_mu_levels') or self.common.get('s03_mu_levels')]
        self.mu_mid = self.mu_levels[len(self.mu_levels) // 2]
        self.ctrls = [c for c in s3.CONTROLLERS if c in self.lock['controllers']]
        self.hil = os.path.normpath(os.path.join(HERE, '..'))

    def ctrl_lock(self, ctrl):
        return self.lock['controllers'][ctrl]

    def s04_paths(self):
        """S04 files read by this script (S04 8: lock, meta, phases, selected, history, evals, trial/, post_checks/)."""
        names = ['lock.yaml', 'meta.yaml', 'phases.json', 'selected.csv']
        for c in self.ctrls:
            names += [f'history_{c}.csv', f'evals_{c}.csv']
        return [n for n in names if os.path.isfile(os.path.join(self.s04, n))] + files_under(self.s04, ['trial', 'post_checks'])

    def s04_files(self):
        return {p: file_sha(os.path.join(self.s04, p))[1] for p in self.s04_paths()}

    def s03_public(self):
        """SHA-256 (LF) of the public S03 files; no sealed file is opened (RSM-1)."""
        out = {}
        for n in S03_PUBLIC:
            p = os.path.join(self.s03, n)
            out[n] = file_sha(p)[1] if os.path.isfile(p) else None
        return out

    def s03_sealed(self, with_test=False):
        """SHA-256 of the sealed S03 files read by a phase: only after the start row is committed (RSM-1)."""
        names = list(S03_SEALED) + (list(S03_SEALED_TEST) if with_test else [])
        out = {}
        for n in names:
            p = os.path.join(self.s03, n)
            out[n] = dict(zip(('sha256', 'sha256_lf'), file_sha(p))) if os.path.isfile(p) else None
        return out

    def check_links(self, out_dirs=RSM_PATHS):
        """RSM-13, RSM2-1: no source or output path of this script is a link or resolves elsewhere; none resolves
        into sealed_test/ (checked from the path entries, no file content is read)."""
        paths = [self.s04, os.path.join(self.s04, 'rsm')] + [os.path.join(self.s04, p) for p in self.s04_paths()]
        paths += [os.path.join(self.s03, n) for n in S03_PUBLIC + S03_SEALED]
        paths += [os.path.join(self.out, d) for d in out_dirs]
        paths += [os.path.join(self.out, p) for p in files_under(self.out, [d for d in out_dirs
                                                                             if os.path.isdir(os.path.join(self.out, d))])]
        for p in paths:
            refuse_link(p, self.s03)


def s03_meta(inp):
    refuse_link(os.path.join(inp.s03, 'meta.yaml'), inp.s03)        # RSM2-1
    return open(os.path.join(inp.s03, 'meta.yaml'), encoding='utf-8').read()


def check_mode(inp):
    """Before the start row (RSM-1): the mode, from S03 meta.yaml and the S04 lock only; no sealed file is opened."""
    meta = s03_meta(inp)
    if s4.meta_value(meta, 'status') != 'ran to the end':
        raise Stop(f'S03 run {inp.s03} did not run to the end (12.2)')
    fake = False
    try:
        fake = s4.meta_value(meta, 'rsm_fake') in (True, 'true')
    except KeyError:
        pass
    if inp.tool_check:
        if not fake:
            raise Stop('tool check: the S03 folder is not an rsm_fake folder; C Q2 allows no sealed/ of a real S03 run')
    else:
        if fake:
            raise Stop('main run on an rsm_fake S03 folder')
        if s4.meta_value(meta, 'trial') in (True, 'true') or len(set(inp.mu_levels)) != 3:
            raise Stop('main RSM needs the main S03 run (three mu levels, not --trial)')
        if not inp.common.get('unsealed'):
            raise Stop('main RSM needs an S04 run made with --s03-unsealed (campaign A ended, L56)')
        check_root_repo(inp)                                # RSM-2
        check_script_blob(inp.hil)                          # RSM-12
    inp.check_links()                                       # RSM-13


def check_inputs(inp):
    """Checks after the start row is committed: the fingerprint (it reads sealed/runs.csv in a main run), the S04
    folder, the domain."""
    # 12.4 [v6]: the same DLL and evaluator blob as the S04 run (and the same S03 files, plant, sources)
    try:
        s4.check_fingerprint(inp.common, s4.fingerprint(inp.dll, inp.s03, unsealed=inp.common.get('unsealed', False)),
                             'rsm')
    except SystemExit as e:
        raise Stop(str(e)) from None
    sel = read_csv(os.path.join(inp.s04, 'selected.csv'))
    phases = json.load(open(os.path.join(inp.s04, 'phases.json'), encoding='utf-8'))
    if not any(p.get('phase') == 'post' for p in phases):
        raise Stop('S04 post checks (section 6) not done: RSM runs after them (12.7)')
    for c in inp.ctrls:
        if not any(p.get('phase') == 'main' and p.get('controller') == c for p in phases):
            raise Stop(f'S04 main runs of {c} not done')
        send = [r for r in sel if r['kind'] == 'send_to_HIL' and r['controller'] == c]
        if len(send) != 1 or send[0].get('feasible') != 'true':
            raise Stop(f'S04 {c}: no feasible send_to_HIL row (x_GA); 12.5 is not defined without it')
        for n in (f'history_{c}.csv', f'evals_{c}.csv'):
            if not os.path.isfile(os.path.join(inp.s04, n)):
                raise Stop(f'{inp.s04}/{n} missing')
        L = inp.ctrl_lock(c)
        if s4.read_domain(inp.s03, c) != L['domain']:
            raise Stop(f'{c}: the S03 levels.csv domain is not the S04 lock.yaml domain')
    for n in ('determinism.csv', 'sensitivity.csv', 'three_mu.csv', 'notes.json'):
        if not os.path.isfile(os.path.join(inp.s04, 'post_checks', n)):
            raise Stop(f'{inp.s04}/post_checks/{n} missing')


def expected_codes():
    exp = {CENTRE_ID: ('centre', (0.0,) * 7)}
    exp.update({p: ('outer', tuple(float(x) for x in c)) for p, c in s3.design_points(1.0, 1)})
    exp.update({p: ('axial', tuple(float(x) for x in c)) for p, c in s3.axial_points()})
    return exp


def read_design(s03_dir, ctrl):
    """{point_id: (box, codes)} of design.csv for one controller (duplicates stop)."""
    out = {}
    for r in read_csv(os.path.join(s03_dir, 'design.csv')):
        if r['controller'] != ctrl:
            continue
        pid = int(r['point_id'])
        if pid in out:
            raise Stop(f'design.csv: point {pid} of {ctrl} twice')
        try:
            codes = tuple(float(r[f'X{k}']) for k in range(1, 8))
        except (KeyError, ValueError):
            raise Stop(f'design.csv: point {pid} of {ctrl} has no X1..X7 codes') from None
        out[pid] = (r['box'], codes)
    return out


def check_fit_design(design, ctrl):
    exp = expected_codes()
    fit = {p: v for p, v in design.items() if v[0] in FIT_BOXES}
    if set(fit) != set(FIT_IDS):
        raise Stop(f'{ctrl}: the centre / outer / axial points of design.csv are not point_id 0, 1-128, 257-270 '
                   f'(12.2): extra {sorted(set(fit) - set(FIT_IDS))[:5]}, missing {sorted(set(FIT_IDS) - set(fit))[:5]}')
    for p in FIT_IDS:
        if fit[p] != exp[p]:
            raise Stop(f'{ctrl} point {p}: box / codes {fit[p]} are not {exp[p]} of the S03 v8 3.2 pattern')
    return {p: fit[p][1] for p in FIT_IDS}


def index_runs(rows, ctrl, pids, speeds, mus):
    """RSM-7: every row of the controller whose point_id is in pids (whatever its box) under the speed and mu
    level it matches (math.isclose); a row matching none of the levels is not needed and left aside. A
    condition with more than one row stops in need_run."""
    by = {}
    for r in rows:
        if r['controller'] != ctrl or int(r['point_id']) not in pids:
            continue
        vs = [v for v in speeds if math.isclose(float(r['v']), v)]
        ms = [m for m in mus if math.isclose(float(r['mu']), m)]
        if len(vs) == 1 and len(ms) == 1:
            by.setdefault((int(r['point_id']), vs[0], ms[0]), []).append(r)
    return by


def need_run(by, pid, v, mu, ctrl, src, box):
    rs = by.get((pid, v, mu), [])
    if len(rs) != 1:
        raise Stop(f'{src}: {len(rs)} runs of {ctrl} point {pid} at v = {v}, mu = {mu} (exactly one needed)')
    r = rs[0]
    if r['box'] != box:
        raise Stop(f'{src}: {ctrl} point {pid} v = {v} mu = {mu} has box {r["box"]!r}, design.csv {box!r}')
    if r['status'] == 'run_error':
        raise Stop(f'{src}: run_error at {ctrl} point {pid} v = {v} mu = {mu} (S03 has no replacement row)')
    if r['status'] not in USABLE_STATUS or r['completed'] not in USABLE_COMPLETED:
        raise Stop(f'{src}: {ctrl} point {pid} v = {v} mu = {mu}: status {r["status"]!r}, completed '
                   f'{r["completed"]!r} not usable')
    for k in ('e_perp_max', 'tv_duty'):
        if fnum(r.get(k)) is None:
            raise Stop(f'{src}: {ctrl} point {pid} v = {v} mu = {mu}: {k} = {r.get(k)!r} not finite')
    return r


def aggregate(runs_by_mu):
    """f1 = max e_perp_max over every run; f2 = max over mu of the mean over v of tv_duty (S04 3)."""
    f1 = max(float(r['e_perp_max']) for rs in runs_by_mu.values() for r in rs)
    f2 = max(float(np.mean([float(r['tv_duty']) for r in rs])) for rs in runs_by_mu.values())
    return f1, f2


def fit_points(inp, ctrl):
    """Rows of fit_points.csv for one controller (reads S03 sealed/runs.csv: phase 1 only)."""
    L = inp.ctrl_lock(ctrl)
    speeds, eval_mu = [float(v) for v in L['speeds']], [float(m) for m in L['eval_mu']]
    codes = check_fit_design(read_design(inp.s03, ctrl), ctrl)
    runs = read_csv(os.path.join(inp.s03, 'sealed', 'runs.csv'))
    by = index_runs(runs, ctrl, set(FIT_IDS), speeds, inp.mu_levels)
    extra = sorted({int(r['point_id']) for r in runs if r['controller'] == ctrl and r['box'] in FIT_BOXES}
                   - set(FIT_IDS))
    if extra:
        raise Stop(f'sealed/runs.csv: {ctrl} points {extra[:5]} of the fit boxes are not in the 143 points')
    rows = []
    exp = expected_codes()
    for pid in FIT_IDS:
        bx = exp[pid][0]
        main = {mu: [need_run(by, pid, v, mu, ctrl, 'sealed/runs.csv', bx) for v in speeds] for mu in eval_mu}
        mid = {inp.mu_mid: [need_run(by, pid, v, inp.mu_mid, ctrl, 'sealed/runs.csv', bx) for v in speeds]}
        f1, f2 = aggregate(main)
        f1m, f2m = aggregate(mid)
        used = [r for rs in main.values() for r in rs]
        frames = {}
        for k in ('n_upper_frames', 'n_back_frames'):
            vals = [fnum(r.get(k)) for r in used]
            if any(v is None for v in vals):
                raise Stop(f'sealed/runs.csv: {ctrl} point {pid}: {k} missing (S03 v8 4)')
            frames[k] = int(sum(vals))
        fl = dict(flag_inconclusive=any(r['completed'] == 'inconclusive' for r in used),
                  flag_track_err=any(r['status'] == 'track_err' for r in used),
                  flag_not_completed=any(r['completed'] == 'false' for r in used))
        box = 'centre' if pid == CENTRE_ID else ('outer' if pid in OUTER_IDS else 'axial')
        rows.append(dict(controller=ctrl, point_id=pid, box=box, **{f'X{k + 1}': c for k, c in enumerate(codes[pid])},
                         f1=f1, f2=f2, f1_mid=f1m, f2_mid=f2m, f1_cens05=min(f1, CENS), **fl,
                         flagged=any(fl.values()), **frames))
    return rows


# ── attempts.csv and the locks (12.7) ────────────────────────────────────────────────

def attempts(inp):
    p = os.path.join(inp.out, 'attempts.csv')
    return read_csv(p) if os.path.isfile(p) else []


def append_attempt(inp, row):
    p = os.path.join(inp.out, 'attempts.csv')
    os.makedirs(inp.out, exist_ok=True)
    new = not os.path.isfile(p)
    with open(p, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ATT_COLS, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in ATT_COLS})


def rel(inp, *parts):
    return '/'.join((inp.rel_out,) + parts)


def dangling(inp):
    """Start rows without an end row: {attempt_id: start row}."""
    rows = attempts(inp)
    ended = {r['attempt_id'] for r in rows if r['kind'] in ('end', 'moved')}
    return {r['attempt_id']: r for r in rows if r['kind'] == 'start' and r['attempt_id'] not in ended}


def phase_files(inp, phase):
    """Files of the phase folders (and its lock) that exist."""
    dirs = list(PHASE_DIRS[phase]) + ([PHASE_LOCK[phase]] if PHASE_LOCK[phase] and phase != 'fit' else [])
    return files_under(inp.out, [d for d in dirs if os.path.exists(os.path.join(inp.out, d))])


def stamp():
    return time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1_000_000_000:09d}'


def move_aside(inp, phases, reason, stamp):
    """Move the folders of these phases to rsm/aborted/<stamp>/ (kept, never deleted)."""
    moved = []
    for ph in phases:
        for d in list(PHASE_DIRS[ph]) + ([PHASE_LOCK[ph]] if PHASE_LOCK[ph] and ph != 'fit' else []):
            src = os.path.join(inp.out, d)
            if os.path.exists(src):
                dst = os.path.join(inp.out, 'aborted', stamp, d)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                moved.append(d)
    if moved:
        with open(os.path.join(inp.out, 'aborted', stamp, 'reason.txt'), 'w', newline='\n', encoding='utf-8') as f:
            f.write(reason + '\n')
    return moved


def split_aborted(inp, st, keep):
    """RSM2-4: files under rsm/aborted/<st>/ to commit (reason.txt and those whose original path is in <keep>) and the
    others (left uncommitted, named in attempts.csv)."""
    ours, stray = [], []
    base = f'aborted/{st}/'
    for p in files_under(inp.out, [f'aborted/{st}']) if os.path.isdir(os.path.join(inp.out, 'aborted', st)) else []:
        orig = p[len(base):]
        (ours if orig == 'reason.txt' or orig in keep else stray).append(p)
    return ours, stray


def tracked_rel(inp, rels):
    """Repository paths of rsm-relative paths."""
    return [rel(inp, r) for r in rels]


def removed_paths(inp, moved):
    """Repository paths under the moved folders that HEAD tracks (git commit --only records their removal)."""
    out = []
    for d in moved:
        lst = git(inp.repo, 'ls-files', '--', rel(inp, d)).splitlines()
        out += [p for p in lst if p.strip()]
    return out


def abort_previous(inp, phase, reason, script):
    """--abort-previous: move this phase's and later phases' outputs aside, close dangling attempts."""
    st = stamp()
    later = PHASES[PHASES.index(phase):]
    moved = move_aside(inp, later, reason, st)
    for aid, r in dangling(inp).items():
        append_attempt(inp, dict(attempt_id=aid, chain_id=r['chain_id'], kind='end', phase=r['phase'],
                                 time=time.strftime('%Y-%m-%d %H:%M:%S'), script_commit=script['commit'],
                                 script_dirty=script['dirty'], status='stopped',
                                 reason=f'aborted by --abort-previous: {reason}'))
    if moved:
        append_attempt(inp, dict(attempt_id=f'M{st}', chain_id='', kind='moved', phase=','.join(later),
                                 time=time.strftime('%Y-%m-%d %H:%M:%S'), script_commit=script['commit'],
                                 script_dirty=script['dirty'], status='moved',
                                 reason=f'{moved} -> aborted/{st}/: {reason}'))
    removed = removed_paths(inp, moved)
    # RSM2-4: commit the moved copies of committed files (and reason.txt); a partial file of a run that died is kept
    # under aborted/ uncommitted and named in attempts.csv
    keep = {p[len(inp.rel_out) + 1:] for p in removed}
    ours, stray = split_aborted(inp, st, keep)
    if stray:
        append_attempt(inp, dict(attempt_id=f'M{st}', chain_id='', kind='note', phase=','.join(later),
                                 time=time.strftime('%Y-%m-%d %H:%M:%S'), script_commit=script['commit'],
                                 script_dirty=script['dirty'], status='', reason=f'uncommitted under aborted/: {stray}'))
    if attempts_state(inp) in ('same', 'none') and not ours and not removed:
        return                                              # nothing moved, nothing closed: nothing to commit
    commit_paths(inp.repo, [rel(inp, 'attempts.csv')] + tracked_rel(inp, ours) + removed,
                 f'fw: rsm_compare abort {phase} {inp.rel}: {reason}'[:200])


def attempts_state(inp):
    """RSM-4: 'none' (no attempts.csv anywhere), 'same' (as committed), 'appended' (HEAD is a prefix of the working
    file: rows added and not committed), 'changed' (a committed row was altered or removed)."""
    p = os.path.join(inp.out, 'attempts.csv')
    head = head_text(inp.repo, rel(inp, 'attempts.csv'))
    if not os.path.isfile(p):
        return 'none' if head is None else 'changed'
    now = open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')
    head = (head or '').replace('\r\n', '\n')
    if now == head:
        return 'same'
    return 'appended' if now.startswith(head) else 'changed'


def require_attempts(inp, allow_appended=False):
    head = (head_text(inp.repo, rel(inp, 'attempts.csv')) or '').replace('\r\n', '\n')
    for c, text in attempts_versions(inp):     # RSM2-2, RSM3-1: every committed version is a prefix of HEAD's
        if text is None or not head.startswith(text):
            raise Stop(f'rsm/attempts.csv history: the version of commit {c[:10]} is not a prefix of the HEAD version '
                       f'(rows altered or removed, or dropped by a merge; 12.7: append only)')
    st = attempts_state(inp)
    if st == 'changed':
        raise Stop('rsm/attempts.csv: a committed row was altered or removed (12.7: append only)')
    if st == 'appended' and not allow_appended:
        raise Stop('rsm/attempts.csv holds rows not committed (an interrupted phase): run again with '
                   '--abort-previous "<reason>" (12.7)')


def lock_check(inp, lock_rel, dirs, phase):
    """RSM-3: a lock file and every file it lists are the blobs of the output commit of the attempt that wrote
    them (not merely of HEAD), and their LF SHA-256 match the lock (12.7)."""
    lp = os.path.join(inp.out, lock_rel)
    if not os.path.isfile(lp):
        raise Stop(f'rsm/{lock_rel} missing')
    lk = json.load(open(lp, encoding='utf-8'))
    c = output_commit(inp, phase, lk['attempt_id'])
    if not same_as(inp.repo, c, rel(inp, lock_rel)) or not same_as_head(inp.repo, rel(inp, lock_rel)):
        raise Stop(f'rsm/{lock_rel} is not its output commit {c[:10]} as it is (12.7)')
    files = lk['files']
    now = set(files_under(inp.out, dirs)) - {lock_rel}
    if now != set(files):
        raise Stop(f'rsm/{lock_rel}: the files {sorted(now ^ set(files))[:5]} differ from the lock')
    for p, h in files.items():
        if not same_as(inp.repo, c, rel(inp, p)) or not same_as_head(inp.repo, rel(inp, p)):
            raise Stop(f'rsm/{p} is not the file of its output commit {c[:10]} (12.7)')
        if file_sha(os.path.join(inp.out, p))[1] != h['sha256_lf']:
            raise Stop(f'rsm/{p}: SHA-256 differs from rsm/{lock_rel} (12.7)')
    return lk


def comparison_kind(inp, phase, aid, chain_id):
    """RSM-4 (12.7): the main comparison is the chain whose test phase is the FIRST test start of attempts.csv and
    whose fit ran on the script commit of the first fit start; once a test start exists, every later run of any
    phase is exploratory. Returns 'main' (test phase only), 'pending' or 'exploratory'."""
    rows = attempts(inp)
    fits = [r for r in rows if r['kind'] == 'start' and r['phase'] == 'fit']
    me = next(i for i, r in enumerate(rows) if r['attempt_id'] == aid and r['kind'] == 'start')
    chain_fit = next((r for r in fits if r['attempt_id'] == chain_id), None)
    if not fits or chain_fit is None or chain_fit['script_commit'] != fits[0]['script_commit']:
        return 'exploratory'
    tests = [i for i, r in enumerate(rows) if r['kind'] == 'start' and r['phase'] == 'test']
    if phase == 'test':
        return 'main' if tests and tests[0] == me else 'exploratory'
    return 'exploratory' if any(i < me for i in tests) else 'pending'


# ── phase 1: fit ─────────────────────────────────────────────────────────────────────

def run_fit(inp, ctx):
    pts_all, coefs, diag = [], [], dict(controllers={})
    for ctrl in inp.ctrls:
        pts = fit_points(inp, ctrl)
        pts_all += pts
        C = np.array([[p[f'X{k}'] for k in range(1, 8)] for p in pts])
        dg = dict(fits={}, not_fitted=[], n_flagged=dict(
            inconclusive=sum(p['flag_inconclusive'] for p in pts), track_err=sum(p['flag_track_err'] for p in pts),
            not_completed=sum(p['flag_not_completed'] for p in pts), any=sum(p['flagged'] for p in pts)),
            n_axial_centre_not_completed=sum(p['flag_not_completed'] for p in pts if p['box'] in ('centre', 'axial')),
            n_upper_frames=sum(p['n_upper_frames'] for p in pts), n_back_frames=sum(p['n_back_frames'] for p in pts),
            eval_mu=inp.ctrl_lock(ctrl)['eval_mu'], speeds=inp.ctrl_lock(ctrl)['speeds'], mu_mid=inp.mu_mid)
        variants = [('main', 'base', None, 'f1'), ('main', 'no_flag', lambda p: not p['flagged'], 'f1'),
                    ('main', 'cens05', None, 'f1_cens05'), ('mid', 'base', None, 'f1_mid')]
        variants += [('main', f'drop_axial_{a}', (lambda p, a=a: p['point_id'] != a), 'f1') for a in AXIAL_IDS]
        for model, variant, keep, f1key in variants:
            sub = [i for i, p in enumerate(pts) if keep is None or keep(p)]
            f2key = 'f2_mid' if model == 'mid' else 'f2'
            for resp, key in (('f1', f1key), ('f2', f2key)):
                beta, d = fit_ls(C[sub], [pts[i][key] for i in sub])
                dg['fits'][f'{model}/{variant}/{resp}'] = d
                if beta is None:
                    dg['not_fitted'].append(f'{model}/{variant}/{resp}: rank {d["rank"]} < {N_TERMS}')
                    continue
                coefs += [dict(controller=ctrl, model=model, variant=variant, response=resp, term=t, coef=float(b))
                          for t, b in zip(TERMS, beta)]
        if [float(m) for m in inp.ctrl_lock(ctrl)['eval_mu']] == [inp.mu_mid]:
            a = {(r['response'], r['term']): r['coef'] for r in coefs
                 if r['controller'] == ctrl and r['model'] == 'main' and r['variant'] == 'base'}
            b = {(r['response'], r['term']): r['coef'] for r in coefs
                 if r['controller'] == ctrl and r['model'] == 'mid' and r['variant'] == 'base'}
            if a != b:
                raise Stop(f'{ctrl}: eval_mu is the middle mu but the main and mid base fits differ')
            dg['main_is_mid'] = True
        for k in ('main/base/f1', 'main/base/f2', 'mid/base/f1', 'mid/base/f2'):
            if dg['fits'][k]['rank'] < N_TERMS:
                raise Stop(f'{ctrl}: fit {k} has rank {dg["fits"][k]["rank"]} < 36 (the 143-point CCD has rank 36)')
        diag['controllers'][ctrl] = dg
    write_csv(os.path.join(inp.out, 'fit', 'fit_points.csv'), pts_all)
    write_csv(os.path.join(inp.out, 'fit', 'coefs.csv'), coefs,
              ['controller', 'model', 'variant', 'response', 'term', 'coef'])
    write_yaml_lines(os.path.join(inp.out, 'fit', 'fit_diag.yaml'), diag['controllers'],
                     'rsm/fit/fit_diag.yaml (rsm_compare.py): rank, flagged points, fits not made, per controller')
    lock = dict(spec=SPEC, attempt_id=ctx['attempt_id'], script=ctx['script'], solver=SOLVER, eps=EPS, dedup=DEDUP,
                K=K, cens=CENS, f2_rel=F2_REL, D=D, f1_tol=F1_TOL, theta_max_deg=THETA_MAX_DEG,
                interpretations=INTERPRETATIONS, inputs=ctx['inputs'],
                files=hash_tree(inp.out, ['fit'], skip={'fit/lock.yaml'}))
    write_json(os.path.join(inp.out, 'fit', 'lock.yaml'), lock)
    return dict(n_points={c: len(FIT_IDS) for c in inp.ctrls}, not_fitted={c: d['not_fitted']
                                                                           for c, d in diag['controllers'].items()})


# ── phase 2: optimise, confirm, classify, check ──────────────────────────────────────

def load_coefs(inp):
    out = {}
    for r in read_csv(os.path.join(inp.out, 'fit', 'coefs.csv')):
        out.setdefault((r['controller'], r['model'], r['variant'], r['response']), {})[r['term']] = float(r['coef'])
    return {k: np.array([v[t] for t in TERMS]) for k, v in out.items()}


def optimise(b1, b2, starts):
    """12.3: (state, f1_star, start rows, candidates [dict code, f1h, f2h]). state None or 'i' / 'ii'."""
    from scipy.optimize import minimize
    bounds = [(-1.0, 1.0)] * 7
    opts = dict(ftol=SOLVER['ftol'], maxiter=SOLVER['maxiter'])
    rows = []

    def one(step, k, x0, fun, jac, cons=()):
        res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=list(cons), method=SOLVER['method'],
                       options=opts)
        raw = np.asarray(res.x, dtype=float)
        finite = bool(np.all(np.isfinite(raw)))
        x = np.clip(raw, -1.0, 1.0) if finite else raw
        move = float(np.max(np.abs(raw - x))) if finite else float('inf')
        # RSM-15: only a rounding move back into the box keeps the solver's success
        ok = bool(res.success) and finite and move <= CLIP_TOL
        rows.append(dict(step=step, start=k, start_code=json.dumps([float(v) for v in x0]), success=bool(res.success),
                         converged=ok, status=int(res.status), nit=int(getattr(res, 'nit', -1)),
                         message=str(res.message), raw_code=json.dumps([repr(float(v)) for v in raw]), clip_move=move,
                         code=json.dumps([repr(float(v)) for v in x]),
                         f1_hat=predict(b1, x) if finite else '', f2_hat=predict(b2, x) if finite else ''))
        return ok, x

    f1 = (lambda c: predict(b1, c)), (lambda c: gradient(b1, c))
    f2 = (lambda c: predict(b2, c)), (lambda c: gradient(b2, c))
    s1 = [one(1, k, x0, *f1) for k, x0 in enumerate(starts)]
    conv1 = [x for ok, x in s1 if ok]
    if not conv1:
        return 'i', None, rows, []
    f1s = min(predict(b1, x) for x in conv1)
    cons = (dict(type='ineq', fun=lambda c: f1s + F1_TOL - f1[0](c), jac=lambda c: -f1[1](c)),
            dict(type='ineq', fun=lambda c: D - f1[0](c), jac=lambda c: -f1[1](c)))
    s2 = [one(2, k, x0, *f2, cons=cons) for k, x0 in enumerate(starts)]
    conv2 = [x for ok, x in s2 if ok]
    if not conv2:
        return 'i', f1s, rows, []
    kept = candidates([dict(code=[float(v) for v in x], f1_hat=predict(b1, x), f2_hat=predict(b2, x)) for x in conv2],
                      f1s)
    return (None if kept else 'ii'), f1s, rows, kept


def candidates(points, f1s):
    """12.3: the converged step-2 points within f1* + 0.01 + eps and D + eps, ordered by f2_hat, f1_hat, codes;
    a point within DEDUP (Euclidean, codes) of a kept one is the same (greedy in that order)."""
    cand = [c for c in points if c['f1_hat'] <= f1s + F1_TOL + EPS and c['f1_hat'] <= D + EPS]
    cand.sort(key=lambda c: (c['f2_hat'], c['f1_hat'], tuple(c['code'])))
    kept = []
    for c in cand:
        if all(np.linalg.norm(np.array(c['code']) - np.array(k['code'])) >= DEDUP for k in kept):
            kept.append(c)
    return kept


def to_gains(inp, ctrl, code):
    """Code -> gains by s03_doe.decode on levels.csv, bounds by s03_doe.check_bounds (12.3 [v6])."""
    dom = inp.ctrl_lock(ctrl)['domain']
    lv = {d['name']: (d['L'], d['H'], d['scale']) for d in dom}
    names = [d['name'] for d in dom]

    def gains_at(c):
        return {n: s3.decode(ci, *lv[n]) for n, ci in zip(names, c)}
    try:
        s3.check_bounds(ctrl, lv, [('rsm', 'rsm', tuple(code))], gains_at)
    except s3.S03Stop as e:
        raise Stop(f'{e}: tell THESIS (12.3)') from None
    lo, hi = s4.ga_bounds(dom)
    xga = [a + (c + 1.0) / 2.0 * (b - a) for c, a, b in zip(code, lo, hi)]
    return gains_at(code), s4.edge_report(dom, xga), dom


def make_evaluator(inp, ctrl, mus, work, cache):
    L = inp.ctrl_lock(ctrl)
    if s3.sha256_file(inp.dll) != inp.common['fingerprint']['dll_sha256']:              # RSM-12: before loading
        raise Stop('the DLL is not the DLL of the S04 run')
    factory = EVALUATOR
    if factory is None:
        if 'fw' not in cache:
            cache['fw'] = fwsim.FirmwareSim(inp.dll)
        return s4.Evaluator(cache['fw'], inp.dll, ctrl, [float(v) for v in L['speeds']], mus, work)
    return factory(inp, ctrl, [float(v) for v in L['speeds']], mus, work)


def feasible(s):
    return s['status'] == 'ok' and s['g1'] == 0 and all(s[f'g{i}'] <= 0 for i in range(2, 6))


def label(rows):
    """12.4 (iii): (label, evidence rows) of one tried candidate from its confirm attempt-1 rows."""
    def pen(r, k):
        return is_on(r.get('pen_' + k, 0))

    def over(r, k, lim):
        x = fnum(r.get(k))
        return not pen(r, k) and x is not None and x > lim
    ev = []
    for r in rows:
        st, comp = r.get('status'), r.get('completed')
        if (comp == 'false' and st != 'run_error') or (st == 'ok' and over(r, 'e_perp_max', D)) or \
                (st in USABLE_STATUS and (over(r, 'e_trk_max', D) or over(r, 'e_theta_cmd_max_deg', THETA_MAX_DEG))) \
                or (st == 'ok' and over(r, 'e_bao_max', D)):
            ev.append(r)
    if ev:
        return 'violation_observed', ev
    ev = [r for r in rows if r.get('status') == 'run_error' or any(pen(r, k) for k in s4.PEN_KEYS)]
    if ev:
        return 'run_error_excluded', ev
    ev = [r for r in rows if r.get('completed') == 'inconclusive' or r.get('status') == 'track_err']
    if ev:
        return 'inconclusive', ev
    return 'pass', []


def row_key(r):
    return f"{r['controller']}|{r['phase']}|{r['cand_id']}|{r['v']}|{r['mu']}|{r['attempt']}"


def score_rows(rows):
    """s04_ga.score on attempt-1 evals rows (dicts with the run quantities)."""
    err = any(r.get('status') == 'run_error' for r in rows)
    return s4.score([(float(r['mu']), float(r['v']), r) for r in rows], err)


def confirm(inp, ctrl, cand_path, work, cache, log):
    """12.4: (x_RSM or None, tried [{cand_id, label, evidence}], evals rows, machine seconds)."""
    t0 = time.time()
    L = inp.ctrl_lock(ctrl)
    mine = [(i + 2, r) for i, r in enumerate(read_csv(cand_path)) if r['controller'] == ctrl]   # file line, row
    ev = make_evaluator(inp, ctrl, [float(m) for m in L['eval_mu']], work, cache)
    tried, evals, x = [], [], None
    for line, r in mine[:K]:
        g = {n: float(r[n]) for n in s3.FACTORS[ctrl]}                # read back from the file (12.4)
        n0 = len(ev.evals)
        s = ev.candidate(g, '', '', int(r['cand_id']))
        new = [dict(e, phase='confirm', ga_seed='', gen='', gain_source=f'rsm/opt/candidates.csv:{line}')
               for e in ev.evals[n0:]]
        evals += new
        lab, evid = label([e for e in new if int(e['attempt']) == 1])
        if (lab == 'pass') != feasible(s):
            raise Stop(f'{ctrl} candidate {r["cand_id"]}: label {lab} but score {s} (internal check)')
        tried.append(dict(controller=ctrl, cand_id=int(r['cand_id']), label=lab,
                          evidence_rows=json.dumps([row_key(e) for e in evid]), f1=s['f1'], f2=s['f2'],
                          **{f'g{i}': s[f'g{i}'] for i in range(1, 6)}, n_run_error=s['n_run_error']))
        log(f'[{ctrl}] candidate {r["cand_id"]}: {lab} (f1 {s["f1"]:.4g}, f2 {s["f2"]:.4g})')
        if lab == 'pass':
            x = dict(cand_id=int(r['cand_id']), gains=g, f1=s['f1'], f2=s['f2'], code=json.loads(r['code']),
                     line=line)
            break
    return x, tried, evals, time.time() - t0


MU_LABELS = dict(pass_='robust', violation_observed='not robust over mu',
                 run_error_excluded='undetermined (run_error or penalised quantity at the low / high mu)',
                 inconclusive='inconclusive over mu (inconclusive or track_err only: g1 convention, not observed)')


def with_flags(r):
    """A run row with its pen_* flags: as stored, or from s04_ga.run_values when the file has none (three_mu.csv)."""
    computed = s4.run_values(r)[1]
    return dict(r, **{k: v for k, v in computed.items() if k not in r})     # RSM2-5: a stored flag is kept


def mu_label_from_rows(rows_by_mu):
    """RSM-10: the label of 12.5 from the attempt-1 rows of the low and high mu, by the order of 12.4 (iii) over
    every row: an observed violation anywhere first, then run_error / penalty, then inconclusive / track_err.
    Returns (label, per-mu labels)."""
    rows = [with_flags(r) for rs in rows_by_mu.values() for r in rs]
    lab = label(rows)[0]
    per = {repr(mu): label([with_flags(r) for r in rs])[0] for mu, rs in sorted(rows_by_mu.items())}
    return MU_LABELS['pass_' if lab == 'pass' else lab], per


def ga_mu_label(inp, ctrl, low_high):
    """x_GA label from the low / high mu rows of post_checks/three_mu.csv (12.5)."""
    p = os.path.join(inp.s04, 'post_checks', 'three_mu.csv')
    rows = read_csv(p)
    need = {'status', 'completed', 'e_perp_max', 'e_trk_max', 'e_bao_max', 'e_theta_cmd_max_deg', 'v', 'mu'}
    if rows and not need <= set(rows[0]):
        raise Stop(f'post_checks/three_mu.csv lacks {sorted(need - set(rows[0]))}: no G3-G5 label for x_GA; tell CODE')
    speeds = [float(v) for v in inp.ctrl_lock(ctrl)['speeds']]
    by, lines = {}, []
    for i, r in enumerate(rows):
        if r.get('controller') != ctrl:
            continue
        mu = next((m for m in low_high if math.isclose(float(r['mu']), m)), None)
        if mu is None:
            continue
        r = dict(r, attempt=1)
        by.setdefault(mu, []).append(r)
        lines.append(i + 2)
    for mu in low_high:
        got = sorted(float(r['v']) for r in by.get(mu, []))
        if len(got) != len(speeds) or any(not math.isclose(a, b) for a, b in zip(got, sorted(speeds))):
            raise Stop(f'post_checks/three_mu.csv: {ctrl} mu {mu}: runs at {got}, expected {speeds}')
    lab, per = mu_label_from_rows(by)
    return lab, per, f'post_checks/three_mu.csv:{",".join(map(str, lines))}'


def classify(x_ga, x_rsm):
    """12.5: 'R1' | 'R2' | 'R3' for two feasible points (dicts f1, f2, x, ga_seed)."""
    a, b = x_ga, x_rsm
    tol_f2 = abs(b['f2'] - a['f2']) <= F2_REL * a['f2'] if a['f2'] != 0 else b['f2'] == a['f2']
    if abs(b['f1'] - a['f1']) <= F1_TOL and tol_f2:
        return 'R2'
    ch = s4.select([a, b])
    return 'R1' if ch is a else 'R3'


def ga_points(inp, ctrl, betas):
    """12.6: labelled GA visited points (main runs), with the main-model errors."""
    L = inp.ctrl_lock(ctrl)
    speeds, mus = [float(v) for v in L['speeds']], [float(m) for m in L['eval_mu']]
    dom = L['domain']
    lo, hi = s4.ga_bounds(dom)
    hist = {(r['ga_seed'], r['gen'], r['cand_id']): r for r in read_csv(os.path.join(inp.s04, f'history_{ctrl}.csv'))}
    groups = {}
    for e in read_csv(os.path.join(inp.s04, f'evals_{ctrl}.csv')):
        groups.setdefault((e['ga_seed'], e['gen'], e['cand_id']), []).append(e)
    pts, why_none, seen, dups = [], {}, {}, []
    for key in sorted(groups, key=lambda k: tuple(int(x) for x in k)):
        es = groups[key]
        use, reason = [], None
        for mu in mus:
            for v in speeds:
                at = {int(e['attempt']): e for e in es if math.isclose(float(e['v']), v) and math.isclose(float(e['mu']), mu)}
                r1 = at.get(1)
                if r1 is None:
                    reason = 'missing run'
                elif r1['status'] == 'run_error':
                    r2 = at.get(2)
                    # RSM-8: 'cause' is on the rerun row (S04 8); 'tool:' = CODE found a tool error
                    if r2 is not None and str(r2.get('cause', '')).startswith(TOOL_CAUSE) and \
                            r2['status'] in USABLE_STATUS:
                        use.append(r2)
                    else:
                        reason = 'run_error not replaced'
                else:
                    use.append(r1)
        if reason is None:
            s = score_rows(use)
            # RSM-9: a stored flag counts too (a finite 1e6 with pen_* = 1 is a penalty)
            if s['pen_f1'] or s['n_pen_runs'] or any(is_on(r.get('pen_' + k, 0)) for r in use for k in s4.PEN_KEYS):
                reason = 'penalised quantity'
        if reason:
            why_none[reason] = why_none.get(reason, 0) + 1
            continue
        h = hist.get(key)
        if h is None:
            raise Stop(f'history_{ctrl}.csv has no candidate {key}')
        gains = tuple(h[d['name']] for d in dom)
        if gains in seen:
            first = seen[gains]
            if (first['f1'], first['f2']) != (s['f1'], s['f2']):
                dups.append(dict(first=first['key'], again=list(key), f1=(first['f1'], s['f1']), f2=(first['f2'], s['f2'])))
            continue
        code = [2.0 * (float(h['x_' + d['name']]) - a) / (b - a) - 1.0 for d, a, b in zip(dom, lo, hi)]
        p = dict(controller=ctrl, ga_seed=key[0], gen=key[1], cand_id=key[2],
                 **{f'X{k + 1}': c for k, c in enumerate(code)}, f1=s['f1'], f2=s['f2'], feasible=feasible(s),
                 f1_hat=predict(betas['f1'], code), f2_hat=predict(betas['f2'], code), key=list(key))
        p['err_f1'], p['err_f2'] = p['f1_hat'] - p['f1'], p['f2_hat'] - p['f2']
        seen[gains] = p
        pts.append(p)
    summ = dict(controller=ctrl, n_labelled=len(pts), n_unlabelled=sum(why_none.values()),
                unlabelled_by_reason=json.dumps(why_none), n_dup_differing=len(dups), dup_differing=json.dumps(dups))
    for k in ('f1', 'f2'):
        e = np.array([p['err_' + k] for p in pts]) if pts else np.array([])
        summ[f'rmse_{k}'] = float(np.sqrt(np.mean(e ** 2))) if pts else ''
        summ[f'maxabs_{k}'] = float(np.max(np.abs(e))) if pts else ''
    return pts, summ


def count_rows(path, ctrl, pred=None):
    if not os.path.isfile(path):
        return 0, 0
    rows = [r for r in read_csv(path) if r.get('controller', ctrl) == ctrl and (pred is None or pred(r))]
    return len(rows), sum(str(r.get('attempt', '1')) == '2' for r in rows)


def cost_rows(inp, ctrl, evals_rsm, times, n_conf_planned):
    """12.6 cost, S46-7 columns."""
    rows = []
    s03_runs = read_csv(os.path.join(inp.s03, 'sealed', 'runs.csv'))
    notes = json.load(open(os.path.join(inp.s03, 'sealed', 'notes.json'), encoding='utf-8'))
    reruns = notes.get('run_error_reruns') or []

    def n_box(boxes):
        return sum(r['controller'] == ctrl and r['box'] in boxes for r in s03_runs)

    def n_rerun(boxes):
        return sum(n.get('controller') == ctrl and n.get('point') and n['point'][1] in boxes for n in reruns)
    pa = sum(s3.PLANNED_RUNS[b] for b in ('centre', 'outer', 'inner'))
    rows.append(dict(controller=ctrl, part='a', item='S03 design (centre, outer, inner)', n_runs_planned=pa,
                     n_runs_actual=n_box(('centre', 'outer', 'inner')) + n_rerun(('centre', 'outer', 'inner')),
                     n_attempt2=n_rerun(('centre', 'outer', 'inner')),
                     row_type='measured_total', source_file='S03 sealed/runs.csv, sealed/notes.json'))
    rows.append(dict(controller=ctrl, part='b', item='S03 axial points', n_runs_planned=s3.PLANNED_RUNS['axial'],
                     n_runs_actual=n_box(('axial',)) + n_rerun(('axial',)), n_attempt2=n_rerun(('axial',)),
                     row_type='measured_total',
                     source_file='S03 sealed/runs.csv, sealed/notes.json'))
    for ph in ('confirm', 'mu3'):
        rs = [r for r in evals_rsm if r['controller'] == ctrl and r['phase'] == ph]
        rows.append(dict(controller=ctrl, part='b', item=f'RSM {ph}',
                         n_runs_planned=n_conf_planned.get(ph, ''), n_runs_actual=len(rs),
                         n_attempt2=sum(int(r['attempt']) == 2 for r in rs), machine_s=round(times.get(ph, 0.0), 1),
                         machine_s_basis='measured', row_type='measured_total',
                         source_file='rsm/confirm/evals_rsm.csv'))
    L = inp.ctrl_lock(ctrl)
    phases = json.load(open(os.path.join(inp.s04, 'phases.json'), encoding='utf-8'))
    tr = os.path.join(inp.s04, 'trial', ctrl, 'evals.csv')
    n, n2 = count_rows(tr, ctrl)
    rows.append(dict(controller=ctrl, part='c', item='S04 trial', n_runs_actual=n, n_attempt2=n2,
                     machine_s=sum(int(p.get('elapsed_s', 0)) for p in phases
                                   if p.get('phase') == 'trial' and p.get('controller') == ctrl),
                     machine_s_basis='measured', row_type='measured_total',
                     source_file=f'trial/{ctrl}/evals.csv, phases.json'))
    n, n2 = count_rows(os.path.join(inp.s04, f'evals_{ctrl}.csv'), ctrl)
    # seeds actually run (phases.json main entry; a tool check may run fewer than the locked ones)
    seeds = next((p.get('seeds') for p in phases if p.get('phase') == 'main' and p.get('controller') == ctrl), None)
    planned = len(seeds or L['seeds']) * int(L['pop']) * int(L['n_gen']) * len(L['eval_mu']) * len(L['speeds'])
    rows.append(dict(controller=ctrl, part='c', item='S04 main', n_runs_planned=planned, n_runs_actual=n,
                     n_attempt2=n2, machine_s=sum(int(p.get('elapsed_s', 0)) for p in phases
                                                   if p.get('phase') == 'main' and p.get('controller') == ctrl),
                     machine_s_basis='measured', row_type='measured_total',
                     source_file=f'evals_{ctrl}.csv, phases.json'))
    npc = sum(count_rows(os.path.join(inp.s04, 'post_checks', f), ctrl)[0]
              for f in ('determinism.csv', 'sensitivity.csv', 'three_mu.csv'))
    rows.append(dict(controller=ctrl, part='c', item='S04 post checks (runs)', n_runs_actual=npc, n_attempt2=0,
                     row_type='measured_total', source_file='post_checks/determinism.csv, sensitivity.csv, three_mu.csv'))
    return rows


def shared_cost_rows(inp):
    notes = json.load(open(os.path.join(inp.s03, 'sealed', 'notes.json'), encoding='utf-8'))
    phases = json.load(open(os.path.join(inp.s04, 'phases.json'), encoding='utf-8'))
    return [dict(controller='all', part='mixed', covers='a+b', item='S03 main design time (every box, every controller)',
                 machine_s=notes.get('elapsed_s_main_design', ''), machine_s_basis='measured',
                 row_type='measured_total', source_file='S03 sealed/notes.json elapsed_s_main_design'),
            dict(controller='all', part='c', item='S04 post checks time',
                 machine_s=sum(int(p.get('elapsed_s', 0)) for p in phases if p.get('phase') == 'post'),
                 machine_s_basis='measured', row_type='measured_total', source_file='phases.json (phase post)')]


def run_opt(inp, ctx, log=print):
    coefs = load_coefs(inp)
    fitp = {}
    for r in read_csv(os.path.join(inp.out, 'fit', 'fit_points.csv')):
        fitp.setdefault(r['controller'], []).append(r)
    starts_rows, cand_rows, evals_rsm, tried_rows, results, ga_rows, ga_err, ax_rows, cost = \
        [], [], [], [], [], [], [], [], []
    cache, summary = {}, {}
    cand_path = os.path.join(inp.out, 'opt', 'candidates.csv')
    per_ctrl = {}
    for ctrl in inp.ctrls:
        b1, b2 = coefs[(ctrl, 'main', 'base', 'f1')], coefs[(ctrl, 'main', 'base', 'f2')]
        starts = [np.array([float(p[f'X{k}']) for k in range(1, 8)]) for p in sorted(fitp[ctrl], key=lambda p: int(p['point_id']))]
        state, f1s, srows, cands = optimise(b1, b2, starts)
        starts_rows += [dict(controller=ctrl, **r) for r in srows]
        for rank, c in enumerate(cands, 1):
            g, edge, dom = to_gains(inp, ctrl, c['code'])
            cand_rows.append(dict(controller=ctrl, cand_id=rank, code=json.dumps([repr(v) for v in c['code']]),
                                  **{f'X{k + 1}': v for k, v in enumerate(c['code'])}, f1_hat=c['f1_hat'],
                                  f2_hat=c['f2_hat'], neg_f1_hat=c['f1_hat'] < 0, neg_f2_hat=c['f2_hat'] < 0,
                                  **edge, **s4.gain_cols(dom, dict(x=[g[d['name']] for d in dom]))))
        per_ctrl[ctrl] = dict(state=state, f1_star=f1s, M=len(cands), b1=b1, b2=b2)
    write_csv(os.path.join(inp.out, 'opt', 'starts.csv'), starts_rows)
    fixed = ['controller', 'cand_id', 'code'] + [f'X{k}' for k in range(1, 8)] + \
        ['f1_hat', 'f2_hat', 'neg_f1_hat', 'neg_f2_hat', 'edge', 'edge_side', 'edge_src', 'edge_limit']
    write_csv(cand_path, cand_rows, fixed + [c for c in dict.fromkeys(k for r in cand_rows for k in r) if c not in fixed])
    sel = read_csv(os.path.join(inp.s04, 'selected.csv'))
    for ctrl in inp.ctrls:
        pc = per_ctrl[ctrl]
        L = inp.ctrl_lock(ctrl)
        work = os.path.join(inp.out, 'confirm', '_work')
        x, tried, ev, t_conf = (None, [], [], 0.0)
        if pc['state'] is None:
            x, tried, ev, t_conf = confirm(inp, ctrl, cand_path, work, cache, log)
            if x is None:
                pc['state'] = 'iii'
        evals_rsm += ev
        tried_rows += tried
        times = dict(confirm=t_conf)
        send = next(r for r in sel if r['kind'] == 'send_to_HIL' and r['controller'] == ctrl)
        x_ga = dict(f1=float(send['f1']), f2=float(send['f2']), x=[float(send[n]) for n in s3.FACTORS[ctrl]],
                    ga_seed=int(send['ga_seed']), who='GA')
        cls, mu_rsm, mu_ga, mu_src, per_rsm, per_ga = 'R0', '', '', '', '', ''
        lo_hi = [m for m in inp.mu_levels if not math.isclose(m, inp.mu_mid)]
        if x is not None:
            x_rsm = dict(f1=x['f1'], f2=x['f2'], x=[x['gains'][n] for n in s3.FACTORS[ctrl]], ga_seed=0, who='RSM')
            cls = classify(x_ga, x_rsm)
            if [float(m) for m in L['eval_mu']] != [inp.mu_mid]:
                mu_rsm = mu_ga = 'not applicable: the GA evaluated the three mu levels (worst of)'
            elif len(lo_hi) != 2:
                mu_rsm = mu_ga = f'not applicable: the S03 run has the mu levels {inp.mu_levels} only (tool check)'
            else:
                t0 = time.time()
                ev3 = make_evaluator(inp, ctrl, lo_hi, work, cache)
                ev3.candidate(x['gains'], '', '', x['cand_id'])
                new = [dict(e, phase='mu3', ga_seed='', gen='', gain_source=f'rsm/opt/candidates.csv:{x["line"]}')
                       for e in ev3.evals]
                evals_rsm += new
                times['mu3'] = time.time() - t0
                by = {}
                for e in new:
                    if int(e['attempt']) == 1:
                        by.setdefault(next(m for m in lo_hi if math.isclose(float(e['mu']), m)), []).append(e)
                mu_rsm, per_rsm = mu_label_from_rows(by)
                mu_ga, per_ga, mu_src = ga_mu_label(inp, ctrl, lo_hi)
                per_rsm, per_ga = json.dumps(per_rsm), json.dumps(per_ga)
        counts = {k: sum(t['label'] == k for t in tried) for k in ('violation_observed', 'run_error_excluded',
                                                                    'inconclusive')}
        n_tried = len(tried)
        if sum(counts.values()) != (n_tried - (1 if x is not None else 0)):
            raise Stop(f'{ctrl}: the three label counts do not add up (12.4 (iii), internal check)')
        results.append(dict(controller=ctrl, state={'i': 'R0 (i) no convergence', 'ii': 'R0 (ii) no candidate',
                                                    'iii': 'R0 (iii) no candidate passed'}.get(pc['state'], 'x_RSM'),
                            rsm_class=cls, M=pc['M'], f1_hat_star=pc['f1_star'] if pc['f1_star'] is not None else '',
                            n_tried=n_tried, n_violation_observed=counts['violation_observed'],
                            n_run_error_excluded=counts['run_error_excluded'], n_inconclusive=counts['inconclusive'],
                            x_rsm_cand_id=x['cand_id'] if x else '', f1_rsm=x['f1'] if x else '',
                            f2_rsm=x['f2'] if x else '', f1_ga=x_ga['f1'], f2_ga=x_ga['f2'],
                            mu_label_rsm=mu_rsm, mu_label_ga=mu_ga, mu_label_rsm_per_mu=per_rsm,
                            mu_label_ga_per_mu=per_ga, mu_label_ga_source=mu_src,
                            n_run_error_runs=sum(r['status'] == 'run_error' and int(r['attempt']) == 1 for r in ev)))
        # 12.6 GA points, axial sensitivity, cost
        pts, summ = ga_points(inp, ctrl, dict(f1=pc['b1'], f2=pc['b2']))
        ga_rows += [{k: v for k, v in p.items() if k != 'key'} for p in pts]
        ga_err += [dict(kind='point', controller=ctrl, ga_seed=p['ga_seed'], gen=p['gen'], cand_id=p['cand_id'],
                        err_f1=p['err_f1'], err_f2=p['err_f2']) for p in pts] + [dict(kind='summary', **summ)]
        where = x['code'] if x else (json.loads(next(r['code'] for r in cand_rows if r['controller'] == ctrl
                                                     and r['cand_id'] == 1)) if pc['M'] >= 1 else None)
        if where is not None:
            where = [float(v) for v in where]
            base = (predict(pc['b1'], where), predict(pc['b2'], where))
            for a in AXIAL_IDS:
                k1, k2 = (ctrl, 'main', f'drop_axial_{a}', 'f1'), (ctrl, 'main', f'drop_axial_{a}', 'f2')
                if k1 not in coefs or k2 not in coefs:
                    ax_rows.append(dict(controller=ctrl, dropped=a, at='x_RSM' if x else 'top candidate',
                                        note='variant not fitted'))
                    continue
                h1, h2 = predict(coefs[k1], where), predict(coefs[k2], where)
                ax_rows.append(dict(controller=ctrl, dropped=a, at='x_RSM' if x else 'top candidate',
                                    f1_hat=h1, f2_hat=h2, d_f1_hat=h1 - base[0], d_f2_hat=h2 - base[1]))
        else:
            ax_rows.append(dict(controller=ctrl, at='none', note='M = 0: no point for the axial sensitivity (12.6)'))
        runs_per = len(L['eval_mu']) * len(L['speeds'])
        cost += cost_rows(inp, ctrl, evals_rsm, times, dict(
            confirm=min(K, pc['M']) * runs_per, mu3=(len(lo_hi) * len(L['speeds']) if x is not None
                                                     and len(L['eval_mu']) == 1 and len(lo_hi) == 2 else 0)))
        summary[ctrl] = dict(state=results[-1]['state'], rsm_class=cls, M=pc['M'], n_tried=n_tried)
    cost += shared_cost_rows(inp)
    keys = [(r['controller'], r['phase'], r['cand_id'], r['v'], r['mu'], r['attempt']) for r in evals_rsm]
    if len(keys) != len(set(keys)):
        raise Stop('evals_rsm.csv: the key (controller, phase, cand_id, v, mu, attempt) is not unique (S46-6)')
    w = os.path.join(inp.out, 'confirm', '_work')
    if os.path.isdir(w) and not os.listdir(w):
        os.rmdir(w)
    write_csv(os.path.join(inp.out, 'confirm', 'evals_rsm.csv'), evals_rsm, EVAL_RSM_COLS)
    write_csv(os.path.join(inp.out, 'confirm', 'result.csv'), results)
    write_csv(os.path.join(inp.out, 'check', 'candidates_tried.csv'), tried_rows,
              ['controller', 'cand_id', 'label', 'evidence_rows', 'f1', 'f2', 'g1', 'g2', 'g3', 'g4', 'g5',
               'n_run_error'])
    write_csv(os.path.join(inp.out, 'check', 'ga_points.csv'), ga_rows)
    write_csv(os.path.join(inp.out, 'check', 'ga_errors.csv'), ga_err)
    write_csv(os.path.join(inp.out, 'check', 'axial_sens.csv'), ax_rows)
    write_csv(os.path.join(inp.out, 'check', 'cost.csv'), cost, COST_COLS)
    write_json(os.path.join(inp.out, 'opt_lock.yaml'), dict(
        spec=SPEC, attempt_id=ctx['attempt_id'], chain_id=ctx['chain_id'], script=ctx['script'],
        fit_lock_sha256_lf=file_sha(os.path.join(inp.out, 'fit', 'lock.yaml'))[1],          # RSM-3
        files=hash_tree(inp.out, ['opt', 'confirm', 'check'])))
    return summary


# ── phase 3: test set ────────────────────────────────────────────────────────────────

def run_test(inp, ctx):
    coefs = load_coefs(inp)
    rows_out, pts_out, cost = [], [], []
    st = os.path.join(inp.s03, 'sealed_test')
    runs = read_csv(os.path.join(st, 'runs_test.csv'))
    notes = json.load(open(os.path.join(st, 'notes.json'), encoding='utf-8'))
    for ctrl in inp.ctrls:
        L = inp.ctrl_lock(ctrl)
        speeds = [float(v) for v in L['speeds']]
        design = read_design(inp.s03, ctrl)
        tp = {p: c for p, (box, c) in design.items() if box == 'test'}
        if set(tp) != set(TEST_IDS):
            raise Stop(f'design.csv: the test points of {ctrl} are not point_id {TEST_IDS[0]}..{TEST_IDS[-1]}')
        by = index_runs(runs, ctrl, set(TEST_IDS), speeds, [inp.mu_mid])
        b1, b2 = coefs[(ctrl, 'mid', 'base', 'f1')], coefs[(ctrl, 'mid', 'base', 'f2')]
        e1, e2, mis, n_pen = [], [], 0, 0
        for pid in TEST_IDS:
            rs = [need_run(by, pid, v, inp.mu_mid, ctrl, 'sealed_test/runs_test.csv', 'test') for v in speeds]
            f1, f2 = aggregate({inp.mu_mid: rs})
            s = score_rows([dict(r, attempt=1) for r in rs])
            n_pen += bool(s['n_pen_runs'])
            h1, h2 = predict(b1, tp[pid]), predict(b2, tp[pid])
            pred_ok, true_ok = h1 <= D + EPS, feasible(s)
            mis += pred_ok != true_ok
            e1.append(h1 - f1)
            e2.append(h2 - f2)
            pts_out.append(dict(controller=ctrl, point_id=pid, **{f'X{k + 1}': c for k, c in enumerate(tp[pid])},
                                f1_mid=f1, f2_mid=f2, f1_hat_mid=h1, f2_hat_mid=h2, pred_feasible=pred_ok,
                                true_feasible_mid=true_ok))
        e1, e2 = np.array(e1), np.array(e2)
        rows_out.append(dict(controller=ctrl, n_points=len(TEST_IDS), n_usable=len(TEST_IDS), n_points_with_penalty=n_pen,
                             rmse_f1_mid=float(np.sqrt(np.mean(e1 ** 2))), maxabs_f1_mid=float(np.max(np.abs(e1))),
                             rmse_f2_mid=float(np.sqrt(np.mean(e2 ** 2))), maxabs_f2_mid=float(np.max(np.abs(e2))),
                             misclassified=mis, misclass_rate=mis / len(TEST_IDS)))
        n2 = sum(x.get('controller') == ctrl for x in (notes.get('run_error_reruns') or []))
        n = sum(r['controller'] == ctrl for r in runs) + n2                     # RSM-11: reruns counted
        cost.append(dict(controller=ctrl, part='b', item='S03 test set (reported apart, not cost to a solution)',
                         n_runs_planned=s3.PLANNED_RUNS['test'], n_runs_actual=n, n_attempt2=n2,
                         row_type='measured_total', source_file='S03 sealed_test/runs_test.csv, notes.json'))
    cost.append(dict(controller='all', part='b', item='S03 test set time', machine_s=notes.get('elapsed_s_test', ''),
                     machine_s_basis='measured', row_type='measured_total',
                     source_file='S03 sealed_test/notes.json elapsed_s_test'))
    write_csv(os.path.join(inp.out, 'test', 'test_errors.csv'), rows_out)
    write_csv(os.path.join(inp.out, 'test', 'test_points.csv'), pts_out)
    write_csv(os.path.join(inp.out, 'test', 'cost_test.csv'), cost, COST_COLS)
    return {r['controller']: dict(rmse_f1_mid=r['rmse_f1_mid'], misclass_rate=r['misclass_rate']) for r in rows_out}


# ── one phase, with its attempt rows and commits (12.7) ──────────────────────────────

def write_rsm_meta(inp, phase, info):
    p = os.path.join(inp.out, 'meta.yaml')
    old = {}
    if os.path.isfile(p):
        for line in open(p, encoding='utf-8').read().splitlines():
            if ':' in line and not line.startswith('#'):
                k, v = line.split(':', 1)
                try:
                    old[k] = json.loads(v)
                except ValueError:
                    pass
    import scipy
    coefs_commit = git(inp.repo, 'log', '-1', '--format=%H', '--', rel(inp, 'fit', 'coefs.csv'), check=False).strip()
    elapsed = dict(old.get('elapsed_s') or {})
    elapsed[phase] = info.get('elapsed_s')
    d = dict(spec=SPEC, tool_check=inp.tool_check, not_main_numbers=inp.tool_check,
             setup='sim: fwsim, ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware',
             operator='Claude (phiên CODE_RSM)', s04_run=inp.rel, s03_run=inp.s03, script=info['script'],
             dll_sha256=s3.sha256_file(inp.dll), evaluator_sha256=s3.evaluator_sha256(),
             versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
             solver=SOLVER, eps=EPS, dedup=DEDUP, K=K, inputs=dict(old.get('inputs') or {}, **{phase: info['inputs']}),
             locks=['rsm/fit/lock.yaml', 'rsm/opt_lock.yaml'], attempts='rsm/attempts.csv',
             coefs_commit_before_test=coefs_commit or None, comparison_kind=info['kind'],
             phases=dict(old.get('phases') or {}, **{phase: info.get('summary')}), elapsed_s=elapsed,
             interpretations=INTERPRETATIONS)
    write_yaml_lines(p, d, 'rsm/meta.yaml (written by fwsim/rsm_compare.py; every value is JSON, valid YAML)')


def run_phase(phase, s04_dir, dll, abort_reason=None, log=print):
    inp = Inputs(s04_dir, dll)
    os.makedirs(inp.out, exist_ok=True)
    refuse_link(inp.out)
    lk = acquire_run_lock(inp.out)                          # RSM-14, RSM3-3, RSM4-2: one run at a time
    try:
        return _run_phase(inp, phase, abort_reason, log)
    finally:
        release_run_lock(lk)


_HELD = set()                 # mutex names this process holds (a Windows mutex is re-entrant for its thread)


def run_lock_name(out):
    real = os.path.normcase(os.path.realpath(os.path.abspath(out)))
    # RSM5-1: Global namespace, one mutex per folder for every Windows logon session
    return 'Global\\' + RUN_LOCK_PREFIX + hashlib.sha256(real.encode('utf-8')).hexdigest()


def acquire_run_lock(out):
    """RSM3-3..5, RSM4-2: an operating-system lock of the rsm/ folder <out> that opens no file (a path checked then
    opened can be swapped for a link to a sealed file in between). Windows: the named mutex run_lock_name(out),
    taken without waiting; the system frees it when the process ends, however it ends (WAIT_ABANDONED: the owner
    died, the lock is ours). POSIX: flock on the folder itself, opened with O_DIRECTORY | O_NOFOLLOW. Non-blocking:
    a held lock stops the run."""
    busy = Stop('the run lock of rsm/ is locked: another rsm_compare run is on this folder (12.7: one run at a time)')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateMutexW.restype = wintypes.HANDLE
        k.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        name = run_lock_name(out)
        h = k.CreateMutexW(None, False, name)
        if not h:
            raise Stop(f'cannot create the run lock {name}: WinError {ctypes.get_last_error()}')
        r = k.WaitForSingleObject(h, 0)
        if r in (0, 0x80) and name not in _HELD:                # WAIT_OBJECT_0, WAIT_ABANDONED
            _HELD.add(name)
            return ('nt', k, h, name)
        if r in (0, 0x80):
            k.ReleaseMutex(h)                                   # this thread holds it already: a second run
        k.CloseHandle(h)
        if r in (0, 0x80, 0x102):                               # 0x102 WAIT_TIMEOUT
            raise busy
        raise Stop(f'the run lock {name} cannot be taken (WaitForSingleObject {r:#x}, WinError '
                   f'{ctypes.get_last_error()})')
    import fcntl
    fd = os.open(out, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise busy from None
    return ('posix', fd)


def release_run_lock(lk):
    if lk[0] == 'nt':
        _, k, h, name = lk
        try:
            k.ReleaseMutex(h)
        finally:
            k.CloseHandle(h)
            _HELD.discard(name)
    else:
        os.close(lk[1])                                     # closing the descriptor frees the flock


def _run_phase(inp, phase, abort_reason, log):
    inp.check_links()                                       # RSM2-1: before attempts.csv, locks, meta are opened
    script = s3.git_state(inp.hil)
    if not inp.tool_check and script['dirty']:
        raise Stop(f'the script tree is not clean ({script["modified"][:5]}): 12.7 locks a clean commit')
    if abort_reason:
        require_attempts(inp, allow_appended=True)
        abort_previous(inp, phase, abort_reason, script)
    require_attempts(inp)                                   # RSM-4: append only, nothing uncommitted
    if dangling(inp):
        raise Stop(f'attempts.csv has a start row without an end row {sorted(dangling(inp))}: run again with '
                   f'--abort-previous "<reason>" (12.7)')
    if phase_files(inp, phase):
        raise Stop(f'rsm/ already holds outputs of phase {phase}: no overwrite (12.7); --abort-previous to move them')
    chain = None
    if phase in ('opt', 'test'):
        chain = lock_check(inp, 'fit/lock.yaml', ['fit'], 'fit')
    if phase == 'test':
        ol = lock_check(inp, 'opt_lock.yaml', ['opt', 'confirm', 'check'], 'opt')
        if ol['chain_id'] != chain['attempt_id'] or \
                ol.get('fit_lock_sha256_lf') != file_sha(os.path.join(inp.out, 'fit', 'lock.yaml'))[1]:
            raise Stop('rsm/opt_lock.yaml was not made from this rsm/fit/lock.yaml (RSM-3)')
    if chain is not None and chain['script']['commit'] != script['commit']:
        raise Stop(f'script commit {script["commit"]} is not the commit of rsm/fit/lock.yaml '
                   f'{chain["script"]["commit"]} (12.7: phases 2, 3 use the same commit)')
    check_mode(inp)                                         # S03 meta.yaml and S04 only; no sealed file (RSM-1)
    aid = f'{phase}-{stamp()}'
    chain_id = aid if phase == 'fit' else chain['attempt_id']
    # RSM-1: the start row holds the SHA-256 of the public S03 files and of S04 (read now), and of the sealed
    # files as the S04 lock recorded them (not read now); the sealed files are hashed after the start commit
    fp_s03 = (inp.common.get('fingerprint') or {}).get('s03_files') or {}
    pub = dict(s03=inp.s03_public(), s04=inp.s04_files(),
               sealed_from_s04_lock={n: fp_s03.get(n) for n in S03_SEALED})
    if chain is not None and pub['s04'] != chain['inputs']['s04']:
        raise Stop('the S04 folder changed since rsm/fit/lock.yaml (12.7)')
    start = dict(attempt_id=aid, chain_id=chain_id, kind='start', phase=phase, time=time.strftime('%Y-%m-%d %H:%M:%S'),
                 script_commit=script['commit'], script_dirty=script['dirty'], inputs_sha256=json.dumps(pub),
                 status='', reason='')
    append_attempt(inp, start)
    commit_paths(inp.repo, [rel(inp, 'attempts.csv')], f'fw: rsm_compare start {phase} {inp.rel} ({aid})')
    head = head_text(inp.repo, rel(inp, 'attempts.csv')) or ''
    if not any(r['attempt_id'] == aid and r['kind'] == 'start' for r in csv.DictReader(io.StringIO(head))):
        raise Stop(f'the start row {aid} is not in HEAD (12.7 [v4])')
    t0 = time.time()
    ctx = dict(attempt_id=aid, chain_id=chain_id, script=script, inputs=pub)
    WRITTEN.clear()
    dirs = list(PHASE_DIRS[phase]) + (['opt_lock.yaml'] if phase == 'opt' else [])
    try:
        check_inputs(inp)                                   # the fingerprint reads sealed/runs.csv (main run)
        sealed = inp.s03_sealed(with_test=phase == 'test')
        for n, want in pub['sealed_from_s04_lock'].items():
            got = (sealed.get(n) or {}).get('sha256')
            if want is not None and got != want:
                raise Stop(f'S03 {n}: SHA-256 {got} is not the one of the S04 lock {want}')
        ctx['inputs'] = dict(pub, sealed=sealed)
        if phase == 'fit':
            summary = run_fit(inp, ctx)
        elif phase == 'opt':
            summary = run_opt(inp, ctx, log)
        else:
            summary = run_test(inp, ctx)
        mine = {os.path.relpath(p, inp.out).replace(os.sep, '/') for p in WRITTEN}
        on_disk = set(files_under(inp.out, dirs))
        if on_disk - mine:                                  # RSM-14: commit only what this run wrote
            raise Stop(f'files not written by this run in the phase folders: {sorted(on_disk - mine)[:5]}')
        if s3.sha256_file(inp.dll) != inp.common['fingerprint']['dll_sha256']:          # RSM-12
            raise Stop('the DLL changed during the phase: not the DLL of the S04 run')
    except (Stop, SystemExit, Exception) as exc:            # noqa: BLE001: every failure closes the attempt
        reason = f'{type(exc).__name__}: {exc}'
        st = stamp()
        mine = {os.path.relpath(p, inp.out).replace(os.sep, '/') for p in WRITTEN}
        moved = move_aside(inp, [phase], reason, st)
        ours, stray = split_aborted(inp, st, mine)
        append_attempt(inp, dict(start, kind='end', time=time.strftime('%Y-%m-%d %H:%M:%S'), status='stopped',
                                 reason=(reason + (f'; not committed (not written by this run): {stray}' if stray
                                                   else ''))[:2000], inputs_sha256=''))
        commit_paths(inp.repo, [rel(inp, 'attempts.csv')] + tracked_rel(inp, ours),
                     f'fw: rsm_compare stopped {phase} {inp.rel} ({aid})')
        raise
    append_attempt(inp, dict(start, kind='end', time=time.strftime('%Y-%m-%d %H:%M:%S'), status='done',
                             reason='', inputs_sha256=json.dumps(ctx['inputs'])))
    info = dict(script=script, inputs=ctx['inputs'], summary=summary, elapsed_s=round(time.time() - t0),
                chain_id=chain_id, kind=comparison_kind(inp, phase, aid, chain_id))
    write_rsm_meta(inp, phase, info)
    paths = [rel(inp, 'attempts.csv'), rel(inp, 'meta.yaml')] + tracked_rel(inp, sorted(on_disk))
    commit_paths(inp.repo, paths,
                 f'fw: rsm_compare {phase} done {inp.rel} ({aid}){" tool check" if inp.tool_check else ""}')
    log(f'rsm {phase}: done ({aid}), {info["kind"]}; {json.dumps(summary, default=str)[:600]}')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phase', required=True, choices=PHASES)
    ap.add_argument('--s04', required=True, help='S04 run folder: <repo>/05_DATA/results/S04/<run>')
    ap.add_argument('--dll', default=os.path.join(HERE, 'build', 'fwsim.dll'))
    ap.add_argument('--abort-previous', metavar='REASON', help='move this and later phases\' outputs to rsm/aborted/')
    a = ap.parse_args(argv)
    try:
        run_phase(a.phase, a.s04, a.dll, a.abort_previous)
    except Stop as e:
        print(f'STOP: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
