"""
s03_sangloc.py: spec S03 v8.6 section 3.1b (06_SPECS/2026-11-05_Spec_S03_DOE_chon_gain.md),
the screening step that sets the domain [L, H] of the DOE/GA on the reference box W: many gain
sets drawn log-uniformly on W are run on fwsim and scored with the S04 evaluator (s04_ga.py
safe_run_gate, Evaluator, score); a threshold per mode locks a competing set C_j, boxes on the
nodes of W hold it, and two independent checks (S2, S2b) accept a mode when every competing set
stays inside. Nothing of this module prints to the console in the parent process and every file
it writes is under the sealed folder it is given (section 3.1b item 13).

  1-3 the domain and the root numbers: the committed file s03_screen_u.csv (sets S0, S1, S2,
      S2b, S3; set t drawn once with default_rng(2026112100 + 10 r + t), PCG64) is the reference;
      the script reads and checks it (section 8 condition 9), never generates it.
  4   gains of a row: x_i = a_i (w_i / a_i)^u_i, binary64; s03_doe.check_candidate before any
      run (a bounds failure stops S03; a broken LADRC triangle is the flag 'ngoai_bang': a
      classifiable, never-feasible set that is not run).
  5   scoring: every set first at the middle mu (g gate speeds); when feasible there
      (g1 = 0, g2..g5 <= 0 of s04_ga.score) 2g more runs at the low and high mu. Modes 'giua'
      and 'ba_mu'; a run_error run makes the set unclassifiable in the mode; a run with tv_duty
      missing or not finite arrives as run_error through s04_ga.tv_duty_run_error. Scores always
      on the first-attempt rows; the diagnosis rerun (attempt 2) never replaces a run.
  6-7 thresholds f1_min + F1_TOL per mode on S1; B_j the smallest box on the nodes of W holding
      the C_j sets of S1; B1 = hull(B_giua, B_ba_mu); a mode without a threshold (S1 had no
      feasible set in it) builds no box and the controller takes the fallback domain.
  8   independent checks at 98.75 % each: check 1 on S2 against B1 (accepted when k = 0 and
      m >= m_min = 86, bound 1 - 0.0125^(1/m)); B2 = B1 extended over the S2 competing sets
      outside B1 when a mode was not accepted; check 2 on S2b against B2 decides the modes not
      accepted at check 1. A controller is accepted ('nhan') when both modes are accepted; its
      domain is B2. Fallback branches (item 9): du_phong_khong_nguong (a mode without
      threshold), du_phong_loi_chay (a set over 10 % unclassifiable in a mode), du_phong_kiem
      (boxes exist, a mode accepted by neither check), khong_chay_sang_loc (item 10).
  10  the sample size N (rows of each of S1, S2, S2b) is set by the sealed S0 try-out
      (sealed/sangloc_thu/): q_min the smallest over the four (controller, mode) shares of S0
      sets feasible and within F1_TOL of the best feasible f1; N_tk the smallest multiple of
      1000 in [3000, 10000] with N q_min >= 2 m_min (exact integer comparison); N_gio the
      largest multiple of 1000, at most 10000, with 3 (N/1000) t_S0 <= T_max = 24 h [ĐỀ XUẤT];
      N = min(N_tk, N_gio). No screening when N < 3000 or N q_min < m_min or S0 had over 10 %
      unclassifiable sets (item 10). Tool-check mode (section 8 condition 11): n0 and N are
      given (tool_check dict); the formula is still computed and reported but does not apply.
  11  sealed/sangloc/ket_qua.json is always written when the step ends: branch, reasons,
      thresholds, boxes, the per-mode per-set counts, the outermost-layer count, the feasible
      share inside the final domain and the float32 duplicate count.
  12  determinism: the first DET_ROWS rows of S1 without a run_error (and not ngoai_bang) run
      again in new processes through the same scoring path; log SHA-256 and every field from
      'status' on must be equal bit for bit.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
import os
import pickle
import struct
import subprocess
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim                                            # noqa: E402
import s03_doe as s3                                    # noqa: E402
import s04_ga as s4                                     # noqa: E402

SCREEN_U = os.path.join(HERE, 's03_screen_u.csv')       # committed reference file, r = 0
SCREEN_COLS = ['set', 'row'] + [f'u{k}' for k in range(1, 8)]
SETS = ('S0', 'S1', 'S2', 'S2b', 'S3')                  # set index t = 0..4
SET_ROWS = dict(S0=1000, S1=10000, S2=10000, S2b=10000, S3=1000)
SEED_BASE = 2026112100                                  # seed of set t in rerun r: SEED_BASE + 10*r + t
TRIAL_SEED_BASE = SEED_BASE + 5                         # tool-check file (8.11): seeds +5..+9, never 10*r + t
SCREEN_SETS = ('S1', 'S2', 'S2b')
MODES = ('giua', 'ba_mu')
M_MIN = 86                      # item 10: smallest m with 1 - 0.0125^(1/m) <= 5 % at 98.75 %
T_MAX_H = 24.0                  # item 10: T_max = 24 h of machine time (budget fixed beforehand)
T_MAX_LABEL = '[ĐỀ XUẤT]'       # the spec marks T_max as a proposal
CONF_ALPHA = 0.0125             # each check at 98.75 %; a controller has at most four checks
N_STEP, N_LO, N_HI = 1000, 3000, 10000
RESERVE = 2                     # N_tk asks N q_min >= RESERVE * m_min (the factor-2 margin)
CHUNK = 20                      # gain sets per worker job
DET_ROWS = 5                    # item 12: rows of S1 run again
BRANCH_PRIORITY = ('khong_chay_sang_loc', 'du_phong_khong_nguong', 'du_phong_loi_chay', 'du_phong_kiem')
LEVEL_SL_COLS = s3.LEVEL_COLS + ['nguon_mien', 'ly_do']        # sealed/levels_sangloc.csv (section 7)
EVAL_SL_COLS = ['controller', 'set', 'row'] + s4.EVAL_COLS[1:]  # evals_<set>.csv; ga_seed, gen, cand_id stay empty
POINTS_COLS = (['controller', 'set', 'row'] + [f'u{k}' for k in range(1, 8)]
               + [f'x{k}' for k in range(1, 8)] + [f'x{k}_f32' for k in range(1, 8)] + ['ngoai_bang'])
SCORE_KEYS = ['f1', 'f2', 'g1', 'g2', 'g3', 'g4', 'g5', 'status', 'n_runs', 'n_false', 'n_inconclusive',
              'n_track_err', 'n_run_error', 'pen_f1', 'n_pen_runs']       # the keys of s4.score / score_ngoai_bang
SCORE_COLS = (['controller', 'set', 'row', 'che_do', 'ngoai_bang', 'phan_loai_duoc', 'kha_thi', 'buoc_b']
              + SCORE_KEYS + ['thuoc_C', 'trong_hop', 'hop'])
DET_FIELDS = EVAL_SL_COLS[EVAL_SL_COLS.index('status'):]       # item 12: compared bit for bit
DET_COLS = ['controller', 'set', 'row', 'v', 'mu', 'log_arrays_equal', 'fields_equal']

_clock = time.time        # the wall clock, replaceable by the tests


# ── items 3: the root numbers file ───────────────────────────────────────────────────

def parse_screen_u(text):
    """{set: np.ndarray (n, 7) float64} of the screening-numbers text (S03 v8.5 section 8
    condition 9): header exactly SCREEN_COLS; the five sets in the order of SETS, each with
    exactly SET_ROWS[set] rows; the row text exactly str(j), j = 1..n; every u a finite float
    with 0 <= u < 1. CRLF and LF both read the same. Anything else raises S03Stop."""
    rd = csv.reader(io.StringIO(text, newline=''))
    head = next(rd, None)
    if head != SCREEN_COLS:
        raise s3.S03Stop(f'screen numbers: columns {head} are not {SCREEN_COLS} '
                         '(S03 v8.5 section 8 condition 9)')
    out = {}
    for s in SETS:
        arr = np.empty((SET_ROWS[s], 7), dtype=float)
        for j in range(1, SET_ROWS[s] + 1):
            r = next(rd, None)
            if r is None or len(r) != len(SCREEN_COLS) or r[0] != s or r[1] != str(j):
                raise s3.S03Stop(f'screen numbers: expected a row "{s},{j},u1..u7", got {r!r} '
                                 '(S03 v8.5 section 8 condition 9)')
            try:
                u = [float(x) for x in r[2:]]
            except ValueError:
                u = None
            if u is None or not all(math.isfinite(x) and 0.0 <= x < 1.0 for x in u):
                raise s3.S03Stop(f'screen numbers: {s} row {j}: u = {r[2:]!r} is not a finite float '
                                 'in [0, 1) (S03 v8.5 section 8 condition 9)')
            arr[j - 1] = u
        out[s] = arr
    if next(rd, None) is not None:
        raise s3.S03Stop('screen numbers: rows after the five sets (S03 v8.5 section 8 condition 9)')
    return out


def load_screen_u(path=SCREEN_U, trial=False):
    """(sets, info) of the committed screening-numbers file, modelled on s3.load_test_points (S03
    v8.5 section 8 condition 9, SL1-9): the bytes are read ONCE; info = file, sha256, sha256_lf,
    sha256_crlf, git (s3.file_git_state), head_blob_sha256, matches_head_blob_lf, problems.
    Problems: not tracked/clean in git; content differs from the HEAD blob after CRLF -> LF
    (s3._head_blob). A parse failure always raises S03Stop; the problems raise S03Stop unless
    trial (then only recorded). A raised S03Stop carries .screen_info = info (what was read so
    far). The script never generates the numbers."""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except OSError as exc:
        e = s3.S03Stop(f'screen numbers: {path} cannot be read: {type(exc).__name__}: {exc} '
                       '(S03 v8.5 section 8 condition 9)')
        e.screen_info = dict(file=s3._rel_here(path), read=f'no: {type(exc).__name__}: {exc}')
        raise e from None
    lf = raw.replace(b'\r\n', b'\n')
    info = dict(file=s3._rel_here(path), sha256=hashlib.sha256(raw).hexdigest(),
                sha256_lf=hashlib.sha256(lf).hexdigest(),
                sha256_crlf=hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest(),
                git=s3.file_git_state(path))

    def stop(msg):
        exc = s3.S03Stop(msg)
        exc.screen_info = info
        return exc
    try:
        sets = parse_screen_u(raw.decode('utf-8'))
    except s3.S03Stop as exc:
        info['problems'] = [f'condition 9: {exc}']
        raise stop(str(exc)) from None
    except (ValueError, UnicodeDecodeError) as exc:
        info['problems'] = [f'condition 9: {type(exc).__name__}: {exc}']
        raise stop(f'screen numbers: {type(exc).__name__}: {exc} (S03 v8.5 section 8 condition 9)') \
            from None
    blob = s3._head_blob(path)
    info['head_blob_sha256'] = hashlib.sha256(blob).hexdigest() if blob is not None else None
    info['matches_head_blob_lf'] = blob is not None and blob.replace(b'\r\n', b'\n') == lf
    problems = []
    if not (info['git']['tracked'] and info['git']['clean']):
        problems.append('not committed / modified (git status)')
    if not info['matches_head_blob_lf']:
        problems.append('content differs from the blob in HEAD (compared after CRLF -> LF)')
    info['problems'] = problems
    if problems and not trial:
        raise stop(f'{path}: ' + '; '.join(problems) + ' (S03 v8.5 sections 7, 8.9)')
    return sets, info


# ── items 4: gains of a row, bounds check ────────────────────────────────────────────

def w_box(nodes):
    """{factor: (a_i, w_i)} of the reference box W from {factor: [nodes ascending]} (item 2:
    a_i the first node, w_i the last = b_n)."""
    return {name: (ns[0], ns[-1]) for name, ns in nodes.items()}


def gains_of_row(ctrl, u, W):
    """{factor: binary64 x_i = a_i (w_i / a_i) ** u_i} of one row (item 4), plain Python floats;
    factors in s3.FACTORS[ctrl] order; W = {factor: [nodes]}."""
    a = w_box(W)
    out = {}
    for i, name in enumerate(s3.FACTORS[ctrl]):
        lo, hi = a[name]
        out[name] = lo * (hi / lo) ** u[i]
    return out


def classify_rows(ctrl, U, W, set_name=''):
    """[dict(row=j (1-based), u=[7 floats], g={factor: x}, flag='ok'|'ngoai_bang')] for the rows
    of U (array (n, 7)), calling s3.check_candidate(ctrl, g, w_box(W)) on EVERY row before any
    run of the set (item 4). An S03Stop goes up, prefixed with controller, set and row."""
    w = w_box(W)
    out = []
    for j in range(len(U)):
        u = [float(x) for x in U[j]]
        g = gains_of_row(ctrl, u, W)
        try:
            flag = s3.check_candidate(ctrl, g, w)
        except s3.S03Stop as exc:
            raise s3.S03Stop(f'{ctrl} {set_name} row {j + 1}: {exc}') from None
        out.append(dict(row=j + 1, u=u, g=g, flag=flag))
    return out


# ── item 5: scoring one gain set on the first-attempt eval rows ──────────────────────

def feasible(s):
    """Item 5 step b: feasible = g1 == 0 and every g2..g5 <= 0 (the spec expression)."""
    return s['g1'] == 0 and all(s[k] <= 0 for k in ('g2', 'g3', 'g4', 'g5'))


def mid_feasible(rows, mu_mid):
    """feasible() of the score over the first-attempt rows at mu_mid (Evaluator records; the
    diagnosis rows with attempt 2 never enter the score, item 5)."""
    first = [(r['mu'], r['v'], r) for r in rows if r.get('attempt') == 1 and r['mu'] == mu_mid]
    return bool(first) and feasible(s4.score(first, False))


def score_modes(rows, flag, speeds, mu_levels):
    """{mode: record} of one gain set (item 5): giua = the g first-attempt runs at the middle mu,
    ba_mu = all 3g runs (only when the set is feasible at the middle mu; else it is not feasible
    at three mu and step b was not run). rows = the eval rows of the set (empty for flag
    'ngoai_bang'); record keys phan_loai_duoc, kha_thi, buoc_b, ngoai_bang and the keys of
    s4.score (None when not computed). An eval row with status run_error makes the set
    unclassifiable in the mode. Anything but exactly the planned runs is a tool error (S03Stop)."""
    g = len(speeds)
    mu_mid = mu_levels[len(mu_levels) // 2]
    if flag == 'ngoai_bang':
        if rows:
            raise s3.S03Stop('a ngoai_bang gain set has eval rows (S03 3.1b item 4: tool error)')
        return {mode: dict(s4.score_ngoai_bang(n), phan_loai_duoc=True, kha_thi=False, buoc_b=False,
                           ngoai_bang=True)
                for mode, n in (('giua', g), ('ba_mu', 3 * g))}
    first = [r for r in rows if r.get('attempt') == 1]
    mid = [r for r in first if r['mu'] == mu_mid]
    rest = [r for r in first if r['mu'] != mu_mid]
    if len(mid) != g or {(r['v'], r['mu']) for r in mid} != {(v, mu_mid) for v in speeds}:
        raise s3.S03Stop(f'the middle-mu runs of a gain set are {len(mid)}, not {g} '
                         '(S03 3.1b item 5: tool error)')
    s_mid = s4.score([(r['mu'], r['v'], r) for r in mid], False)
    ok_mid = not any(r['status'] == 'run_error' for r in mid)
    feas_mid = feasible(s_mid)
    rec = {'giua': dict(s_mid, phan_loai_duoc=ok_mid, kha_thi=feas_mid, buoc_b=feas_mid,
                        ngoai_bang=False)}
    if feas_mid:
        if len(rest) != 2 * g or {(r['v'], r['mu']) for r in rest} != \
                {(v, m) for m in mu_levels if m != mu_mid for v in speeds}:
            raise s3.S03Stop('a gain set feasible at the middle mu lacks its 2g step-b runs '
                             '(S03 3.1b item 5: tool error)')
        s_all = s4.score([(r['mu'], r['v'], r) for r in first], False)
        rec['ba_mu'] = dict(s_all, phan_loai_duoc=not any(r['status'] == 'run_error' for r in first),
                            kha_thi=feasible(s_all), buoc_b=True, ngoai_bang=False)
    else:
        if rest:
            raise s3.S03Stop('a gain set not feasible at the middle mu has low/high-mu runs '
                             '(S03 3.1b item 5: tool error)')
        rec['ba_mu'] = dict.fromkeys(SCORE_KEYS, None)
        rec['ba_mu'].update(phan_loai_duoc=ok_mid, kha_thi=False, buoc_b=False, ngoai_bang=False)
    return rec


# ── items 6-8: thresholds, the competing set, boxes on the nodes, checks ─────────────

def threshold(records):
    """None | dict(f1_min=..., tau=f1_min + s4.F1_TOL): the threshold of one mode on the feasible
    records of S1 (item 6; S1 with no feasible set in the mode: no threshold)."""
    f1s = [r['f1'] for r in records if r['kha_thi']]
    if not f1s:
        return None
    f1_min = min(f1s)
    return dict(f1_min=f1_min, tau=f1_min + s4.F1_TOL)


def in_C(record, thr):
    """The competing set C_j (item 6): feasible and f1 <= f1_min + F1_TOL, the very expression of
    s4.select. thr = the threshold dict of threshold(), or None."""
    return thr is not None and record['kha_thi'] and record['f1'] <= thr['f1_min'] + s4.F1_TOL


def node_floor(nodes, x):
    """Largest node <= x (item 7); S03Stop when x is outside [nodes[0], nodes[-1]]."""
    if not (nodes[0] <= x <= nodes[-1]):
        raise s3.S03Stop(f'{x!r} is outside the nodes [{nodes[0]!r}, {nodes[-1]!r}] '
                         '(S03 3.1b item 7)')
    return nodes[max(0, bisect.bisect_right(nodes, x) - 1)]


def node_ceil(nodes, x):
    """Smallest node >= x (item 7); S03Stop when x is outside [nodes[0], nodes[-1]]."""
    if not (nodes[0] <= x <= nodes[-1]):
        raise s3.S03Stop(f'{x!r} is outside the nodes [{nodes[0]!r}, {nodes[-1]!r}] '
                         '(S03 3.1b item 7)')
    return nodes[bisect.bisect_left(nodes, x)]


def box_from_points(points, W):
    """{factor: (lo, hi)} = B_j (item 7): the smallest box on the nodes of W holding every gain
    set of points (list of g dicts): per gain lo = node_floor of the smallest value, hi =
    node_ceil of the largest; equal ends widen by one node on each side that has a node (at the
    edge of W only one side moves). Empty points: S03Stop (tool error)."""
    if not points:
        raise s3.S03Stop('box_from_points called with no gain set (S03 3.1b item 7: tool error)')
    out = {}
    for name, nodes in W.items():
        xs = [p[name] for p in points]
        lo, hi = node_floor(nodes, min(xs)), node_ceil(nodes, max(xs))
        if lo == hi:
            i = nodes.index(lo)
            if i > 0:
                lo = nodes[i - 1]
            if i + 1 < len(nodes):
                hi = nodes[i + 1]
        out[name] = (lo, hi)
    return out


def hull(b1, b2):
    """B1 = the hull of two boxes (item 7): per gain the smaller lo, the larger hi."""
    return {n: (min(b1[n][0], b2[n][0]), max(b1[n][1], b2[n][1])) for n in b1}


def extend_box(box, points, W):
    """B2 (item 8): per gain (min(box lo, node_floor of the smallest point), max(box hi,
    node_ceil of the largest point)); NO widening rule; no points -> the same box."""
    if not points:
        return dict(box)
    out = {}
    for name, nodes in W.items():
        xs = [p[name] for p in points]
        out[name] = (min(box[name][0], node_floor(nodes, min(xs))),
                     max(box[name][1], node_ceil(nodes, max(xs))))
    return out


def in_box(g, box):
    """Every gain of g inside [lo, hi] of box, both ends inclusive (binary64, exact)."""
    return all(box[n][0] <= g[n] <= box[n][1] for n in box)


def check_counts(records, gains, thr, box):
    """(m, k) of one mode on a check sample (item 8): m sets of C_j (feasible, hence
    classifiable), k of them outside box."""
    c_idx = [i for i, r in enumerate(records) if in_C(r, thr)]
    return len(c_idx), sum(not in_box(gains[i], box) for i in c_idx)


def accepted(m, k):
    """Item 8: a mode is accepted when k == 0 and m >= m_min."""
    return k == 0 and m >= M_MIN


def bound(m, k):
    """Item 8 statement: 1 - 0.0125^(1/m), only when m >= 1 and k == 0 (else None)."""
    return 1.0 - CONF_ALPHA ** (1.0 / m) if m >= 1 and k == 0 else None


def over_error_limit(n_unclassifiable, n_total):
    """Item 5: strictly more than 10 % of the sets unclassifiable; the denominator is EVERY gain
    set of the set (ngoai_bang is classifiable and counts in it)."""
    return 10 * n_unclassifiable > n_total


# ── item 10: the sample size from the S0 try-out ─────────────────────────────────────

def sample_size(q_num, n0, t_s0, s0_over, tool_n=None):
    """The N of section 3.1b item 10. q_num = {(ctrl, mode): count of S0 gain sets feasible in
    the mode with f1 <= smallest feasible f1 + F1_TOL (0 when none feasible)}; n0 = the number
    of S0 rows used (the denominator incl. ngoai_bang and unclassifiable sets); t_s0 =
    wall-clock seconds of the S0 phase (both controllers), finite and positive else S03Stop;
    s0_over = S0 had over 10 % unclassifiable sets in a controller/mode; tool_n = the given N of
    the tool check (8.11): the formula is computed and reported (the two N-based reasons under
    ly_do_cong_thuc) but does not apply; the s0_over rule still does.
    Returns dict(q (per ctrl, mode: tu, mau, q), q_min, t_S0, N_gio, N_tk, N, chay, ly_do,
    ly_do_cong_thuc, N_da_dung (0 when not run), m_min, T_max_gio, T_max_label)."""
    if not (math.isfinite(t_s0) and t_s0 > 0.0):
        raise s3.S03Stop(f't_S0 = {t_s0!r} is not finite and positive '
                         '(S03 3.1b item 10: tool error)')
    count_min = min(q_num.values()) if q_num else 0
    q = {f'{c} {m}': dict(tu=int(cnt), mau=n0, q=cnt / n0) for (c, m), cnt in q_num.items()}
    n_gio = 0                                  # largest multiple of 1000 within the 24 h budget
    for k in range(N_HI // N_STEP, 0, -1):
        if 3 * k * t_s0 <= T_MAX_H * 3600.0:   # equality allowed (item 10)
            n_gio = k * N_STEP
            break
    n_tk = N_HI                                # smallest multiple in [3000, 10000] with N q_min >= 2 m_min
    for n_ in range(N_LO, N_HI + N_STEP, N_STEP):
        if n_ * count_min >= RESERVE * M_MIN * n0:   # exact integer comparison (item 10)
            n_tk = n_
            break
    N = min(n_tk, n_gio)
    ly_cong_thuc = []
    if N < N_LO:
        ly_cong_thuc.append(f'khong_chay_sang_loc: N = {N} < {N_LO} (N_gio {n_gio}, N_tk {n_tk}; '
                            'S03 3.1b item 10)')
    if N * count_min < M_MIN * n0:
        ly_cong_thuc.append(f'khong_chay_sang_loc: N * q_min = {N} * {count_min}/{n0} < {M_MIN} '
                            '(S03 3.1b item 10)')
    ly_over = (['khong_chay_sang_loc: S0 has more than 10 % unclassifiable gain sets in a '
                'controller/mode (S03 3.1b items 5, 10)'] if s0_over else [])
    if tool_n is None:
        ly_do, ly_ct = ly_cong_thuc + ly_over, []
        chay, n_dung = not ly_do, N
    else:
        ly_do, ly_ct = ly_over, ly_cong_thuc           # 8.11: the formula does not set N
        chay, n_dung = not ly_do, tool_n
    return dict(q=q, q_min=count_min / n0, t_S0=t_s0, N_gio=n_gio, N_tk=n_tk, N=N, chay=chay,
                ly_do=ly_do, ly_do_cong_thuc=ly_ct, N_da_dung=n_dung if chay else 0, m_min=M_MIN,
                T_max_gio=T_MAX_H, T_max_label=T_MAX_LABEL)


# ── item 5: running the gain sets in worker processes ────────────────────────────────

def _firmware(dll):
    """The FirmwareSim of one worker process (a seam the tests replace)."""
    return fwsim.FirmwareSim(dll)


def do_job(job):
    """One worker job (dict(name, dll, controller, speeds, mu_levels, rows=[(row_no, g)], work)):
    for every (row_no, gain set) the set runs at the middle mu (g runs, Evaluator ev_mid), and
    only when feasible there at the two other mu levels (2g runs, Evaluator ev_rest) — the
    scoring path of item 5 (safe_run_gate + the tv_duty conversion inside Evaluator.candidate).
    Returns [(row_no, [eval rows appended])]. A set the evaluator marks ngoai_bang, or that
    appends no row, is a tool error (the parent filters those sets): S03Stop."""
    dll, ctrl, speeds, mu_levels = job['dll'], job['controller'], job['speeds'], job['mu_levels']
    mu_mid = mu_levels[len(mu_levels) // 2]
    fw = _firmware(dll)
    ev_mid = s4.Evaluator(fw, dll, ctrl, speeds, [mu_mid], job['work'])
    ev_rest = s4.Evaluator(fw, dll, ctrl, speeds, [m for m in mu_levels if m != mu_mid], job['work'])
    out = []
    for row_no, g in job['rows']:
        n0_ = len(ev_mid.evals)
        s = ev_mid.candidate(g, '', '', row_no)
        if s.get('status') == 'ngoai_bang':
            raise s3.S03Stop(f'{ctrl} row {row_no}: a ngoai_bang set reached a worker '
                             '(S03 3.1b item 5: tool error — the parent filters those sets)')
        rows = list(ev_mid.evals[n0_:])
        if not rows:
            raise s3.S03Stop(f'{ctrl} row {row_no}: the evaluator appended no run '
                             '(S03 3.1b item 5: tool error)')
        if mid_feasible(rows, mu_mid):
            n0_ = len(ev_rest.evals)
            s = ev_rest.candidate(g, '', '', row_no)
            if s.get('status') == 'ngoai_bang':
                raise s3.S03Stop(f'{ctrl} row {row_no}: a ngoai_bang set reached a worker '
                                 '(S03 3.1b item 5: tool error)')
            rows += list(ev_rest.evals[n0_:])
        out.append((row_no, rows))
    return out


def job_main(path):
    """--job <pickle> of a worker process: run the job, write path + '.out' (pickle dict(ok=True,
    rows) or dict(ok=False, stop)); a S03Stop is reported with exit code 0, any other exception
    goes to stderr and a non-zero exit."""
    try:
        with open(path, 'rb') as f:
            job = pickle.load(f)
        res = dict(ok=True, rows=do_job(job))
    except s3.S03Stop as exc:
        res = dict(ok=False, stop=str(exc))
    with open(path + '.out', 'wb') as f:
        pickle.dump(res, f)


def _collect_logs(jobs, paths, work, log_path):
    """Append every non-empty child log to log_path (each part headed by the job name — sealed
    channel, item 13), then remove the per-job files (.pkl, .out, .log) and the empty folders
    under work (and work itself when empty)."""
    parts = []
    for i, p in enumerate(paths):
        lp = p + '.log'
        if os.path.isfile(lp):
            with open(lp, 'rb') as f:
                data = f.read()
            if data:
                parts.append((jobs[i]['name'], data))
    if parts:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'ab') as f:
            for name, data in parts:
                f.write(('===== job %s =====\n' % name).encode('utf-8'))
                f.write(data)
                if not data.endswith(b'\n'):
                    f.write(b'\n')
    for p in paths:
        for q in (p, p + '.out', p + '.log'):
            if os.path.exists(q):
                os.remove(q)
    if os.path.isdir(work):
        for d in os.listdir(work):
            dp = os.path.join(work, d)
            if os.path.isdir(dp) and not os.listdir(dp):
                os.rmdir(dp)
        if not os.listdir(work):
            os.rmdir(work)


def run_jobs(jobs, work, n_parallel, log_path):
    """Run job dicts as child processes `python s03_sangloc.py --job <pickle>` (at most
    n_parallel at once), like o1_feas_search.run_jobs but with pickle job files job_<name>.pkl
    under work and each child's stdout+stderr to <job>.log under work. Returns the do_job
    results in job order. A child with a non-zero exit code: the others are killed and a
    S03Stop says a worker failed and names log_path (no gain values). A result ok=False: the
    S03Stop carries its text."""
    os.makedirs(work, exist_ok=True)
    paths = []
    for job in jobs:
        p = os.path.join(work, f"job_{job['name']}.pkl")
        with open(p, 'wb') as f:
            pickle.dump(job, f)
        paths.append(p)
    pending, running, failed = list(range(len(jobs))), {}, False
    while pending or running:
        while pending and len(running) < max(1, n_parallel):
            i = pending.pop(0)
            logf = open(paths[i] + '.log', 'w')
            running[i] = (subprocess.Popen([sys.executable, os.path.abspath(__file__), '--job',
                                            paths[i]], stdout=logf, stderr=subprocess.STDOUT), logf)
        time.sleep(0.1)
        for i in list(running):
            proc, logf = running[i]
            if proc.poll() is not None:
                logf.close()
                del running[i]
                if proc.returncode != 0:
                    failed = True
                    for j in list(running):
                        running[j][0].kill()
                        running[j][0].wait()
                        running[j][1].close()
                    running.clear()
                    pending.clear()
                    break                               # the stale indices of the loop are gone
    results = []
    for p in paths:
        try:
            with open(p + '.out', 'rb') as f:
                results.append(pickle.load(f))
        except OSError:
            results.append(None)
    _collect_logs(jobs, paths, work, log_path)
    if failed or any(r is None for r in results):
        raise s3.S03Stop(f'a worker of the screening step failed; its output is in {log_path} '
                         '(S03 3.1b item 5: tool error)')
    out = []
    for r in results:
        if not r.get('ok'):
            raise s3.S03Stop(r.get('stop', 'a worker stopped without a reason'))
        out.append(r['rows'])
    return out


def run_set(set_name, rows_by_ctrl, dll, speeds, mu_levels, work, n_proc, log_path):
    """Run the runnable gain sets (flag 'ok'; a ngoai_bang row is never dispatched and maps to
    []) of one set through run_jobs: jobs of CHUNK rows per controller named
    <set>_<ctrl>_<i:05d>, the controllers' jobs interleaved in one queue. Returns
    ({ctrl: {row_no: [eval rows]}}, wall-clock seconds)."""
    t0 = _clock()
    by_ctrl, order = {}, []
    for ctrl, records in rows_by_ctrl.items():
        todo = [(r['row'], r['g']) for r in records if r['flag'] == 'ok']
        js = [dict(name=f'{set_name}_{ctrl}_{i // CHUNK:05d}', dll=dll, controller=ctrl,
                   speeds=list(speeds), mu_levels=list(mu_levels), rows=todo[i:i + CHUNK],
                   work=os.path.join(work, f'{set_name}_{ctrl}_{i // CHUNK:05d}'))
              for i in range(0, len(todo), CHUNK)]
        by_ctrl[ctrl] = js
        order.append(ctrl)
    jobs = [by_ctrl[ctrl][i] for i in range(max((len(v) for v in by_ctrl.values()), default=0))
            for ctrl in order if i < len(by_ctrl[ctrl])]
    results = run_jobs(jobs, work, n_proc, log_path) if jobs else []
    out = {}
    for job, res in zip(jobs, results):
        out.setdefault(job['controller'], {}).update(res)
    for ctrl, records in rows_by_ctrl.items():
        for r in records:
            out.setdefault(ctrl, {}).setdefault(r['row'], [])
    return out, _clock() - t0


def eval_csv_rows(set_name, records, evals):
    """The evals_<set>.csv rows of one controller's records: each Evaluator record r becomes
    dict(r, set=set_name, row=r['cand_id'], ga_seed='', gen='', cand_id='')."""
    out = []
    for r in records:
        for e in evals.get(r['row'], []):
            out.append(dict(e, set=set_name, row=e['cand_id'], ga_seed='', gen='', cand_id=''))
    return out


# ── item 12: the determinism check of the step ───────────────────────────────────────

def determinism_check(set_name, rows_by_ctrl, evals_by_ctrl, dll, speeds, mu_levels, work, n_proc,
                      log_path):
    """Item 12: per controller the first DET_ROWS rows of the set (row order) that are not
    ngoai_bang and have no first-attempt run with status 'run_error' (fewer: all of them; none:
    skipped and recorded) run again through run_jobs — new processes, the same job format and
    the same scoring path (safe_run_gate + the tv_duty conversion, NOT s3.run_in_new_process).
    For every first-attempt run (v, mu) of the first pass the rerun must have the same set of
    (v, mu) runs, the same log_sha256 and every column of EVAL_SL_COLS from 'status' on equal
    bit for bit; a rerun row with status run_error counts as different. Returns (CSV rows:
    controller, set, row, v, mu, log_arrays_equal, fields_equal; {ctrl: [row numbers chosen]})."""
    t0 = _clock()
    chosen, jobs = {}, []
    for ctrl, records in rows_by_ctrl.items():
        todo = []
        for r in records:
            if r['flag'] != 'ok' or len(todo) >= DET_ROWS:
                continue
            if any(e.get('attempt') == 1 and e.get('status') == 'run_error'
                   for e in evals_by_ctrl.get(ctrl, {}).get(r['row'], [])):
                continue
            todo.append(r)
        chosen[ctrl] = [r['row'] for r in todo]
        jobs += [dict(name=f'kiem_tat_dinh_{ctrl}_{i // CHUNK:05d}', dll=dll, controller=ctrl,
                      speeds=list(speeds), mu_levels=list(mu_levels),
                      rows=[(r['row'], r['g']) for r in todo[i:i + CHUNK]],
                      work=os.path.join(work, f'kiem_tat_dinh_{ctrl}_{i // CHUNK:05d}'))
                 for i in range(0, len(todo), CHUNK)]
    results = run_jobs(jobs, work, n_proc, log_path) if jobs else []
    again = {}
    for job, res in zip(jobs, results):
        again.setdefault(job['controller'], {}).update(res)
    rows, n_rerun = [], 0
    for ctrl, records in rows_by_ctrl.items():
        for row_no in chosen[ctrl]:
            first = [e for e in evals_by_ctrl[ctrl][row_no] if e.get('attempt') == 1]
            rere = again.get(ctrl, {}).get(row_no, [])
            n_rerun += sum(e.get('attempt') == 2 for e in rere)
            back = {(e['v'], e['mu']): e for e in rere if e.get('attempt') == 1}
            same_set = {(e['v'], e['mu']) for e in first} == set(back)
            for e1 in first:
                e2 = back.get((e1['v'], e1['mu']))
                eq_log = e2 is not None and e2.get('log_sha256') == e1.get('log_sha256')
                eq_fields = (e2 is not None and e2.get('status') != 'run_error' and same_set
                             and all(s3._bits(e2.get(k)) == s3._bits(e1.get(k)) for k in DET_FIELDS))
                rows.append(dict(controller=ctrl, set=set_name, row=row_no, v=e1['v'], mu=e1['mu'],
                                 log_arrays_equal=eq_log, fields_equal=eq_fields))
    stats = dict(seconds=_clock() - t0, n_luot_thuc=sum(1 for r in rows), n_chay_lai=n_rerun)
    return rows, chosen, stats


# ── items 4-12: the step itself ──────────────────────────────────────────────────────

def _json_clean(x):
    """Strict-JSON form: non-finite floats -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def write_json(path, obj):
    """One JSON file of the step: strict (allow_nan=False), utf-8, ensure_ascii=False, indent 1."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_clean(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def _box_json(box):
    return None if box is None else {n: [box[n][0], box[n][1]] for n in box}


def _set_phase(set_name, n_use, out_dir, work_dir, log_path, controllers, gate, mu_levels, W,
               screen, n_proc, dll):
    """One set: classify every one of its first n_use rows BEFORE any run, write points_<set>.csv,
    run the runnable sets through run_set, write evals_<set>.csv and score_<set>.csv. Returns
    (records {ctrl: [...]}, evals {ctrl: {row: [...]}}, score rows for the CSV, scores
    {ctrl: {row: {mode: record}}}, seconds, n first-attempt runs, n attempt-2 rows)."""
    records = {ctrl: classify_rows(ctrl, screen[set_name][:n_use], W[ctrl], set_name)
               for ctrl in controllers}
    point_rows = []
    for ctrl in controllers:
        names = s3.FACTORS[ctrl]
        for r in records[ctrl]:
            xs = [r['g'][n] for n in names]
            point_rows.append(dict(controller=ctrl, set=set_name, row=r['row'],
                                   ngoai_bang=r['flag'] == 'ngoai_bang',
                                   **{f'u{k + 1}': r['u'][k] for k in range(7)},
                                   **{f'x{k + 1}': xs[k] for k in range(7)},
                                   **{f'x{k + 1}_f32': float(np.float32(xs[k])) for k in range(7)}))
    s3.write_csv(os.path.join(out_dir, f'points_{set_name}.csv'), point_rows, POINTS_COLS)
    evals, seconds = run_set(set_name, records, dll, gate, mu_levels, work_dir, n_proc, log_path)
    eval_rows = [e for ctrl in controllers for e in eval_csv_rows(set_name, records[ctrl], evals[ctrl])]
    s3.write_csv(os.path.join(out_dir, f'evals_{set_name}.csv'), eval_rows, EVAL_SL_COLS)
    scores = {ctrl: {r['row']: score_modes(evals[ctrl].get(r['row'], []), r['flag'], gate, mu_levels)
                     for r in records[ctrl]} for ctrl in controllers}
    score_rows = [dict(controller=ctrl, set=set_name, row=r['row'], che_do=mode,
                       ngoai_bang=r['flag'] == 'ngoai_bang', **{k: scores[ctrl][r['row']][mode].get(k)
                                                              for k in SCORE_KEYS},
                       **{k: scores[ctrl][r['row']][mode][k]
                          for k in ('phan_loai_duoc', 'kha_thi', 'buoc_b')})
                  for ctrl in controllers for r in records[ctrl] for mode in MODES]
    n_runs = sum(e.get('attempt') == 1 for e in eval_rows)
    n_rerun = sum(e.get('attempt') == 2 for e in eval_rows)
    return records, evals, score_rows, scores, seconds, n_runs, n_rerun


def _mode_stats(records, scores):
    """Per mode counts of one set: (classifiable, feasible, unclassifiable) over the records."""
    out = {}
    for mode in MODES:
        recs = [scores[r['row']][mode] for r in records]
        out[mode] = dict(n_bo_gain=len(recs), n_phan_loai_duoc=sum(r['phan_loai_duoc'] for r in recs),
                         n_kha_thi=sum(r['kha_thi'] for r in recs),
                         n_khong_phan_loai=sum(not r['phan_loai_duoc'] for r in recs))
    return out


def run_step(sealed, dll, controllers, gate, mu_levels, W, quet, screen, n_proc, tool_check=None):
    """Items 4-13 of section 3.1b, in the write order of section 7. sealed: the sealed/ path;
    gate: the speeds in the gate (g = len(gate)); mu_levels: the three sorted levels else
    S03Stop; W = {ctrl: {factor: [nodes]}} the reference box; quet = {ctrl: {factor: (L_q, H_q)}}
    the sweep levels of the fallback domain; screen = the sets of load_screen_u; n_proc workers;
    tool_check = None | dict(n0, n) (8.11: S0 the first n0 rows, S1/S2/S2b the first n; the N
    formula is reported but does not apply).
    Returns dict(mien={ctrl: None | {factor: (L, H)}}, nhanh={ctrl: str}, ly_do={ctrl: [str]},
    N_da_dung=int, pha=[notes entries, one per phase, both controllers sharing the pool])."""
    if list(mu_levels) != sorted(mu_levels) or len(mu_levels) != 3:
        raise s3.S03Stop(f'the screening step needs three sorted mu levels, got {mu_levels!r} '
                         '(S03 3.1b item 5)')
    if not controllers:
        raise s3.S03Stop('the screening step needs at least one controller (S03 3.1b)')
    sl_thu = os.path.join(sealed, 'sangloc_thu')
    sl_dir = os.path.join(sealed, 'sangloc')
    os.makedirs(sl_thu, exist_ok=True)
    os.makedirs(sl_dir, exist_ok=True)
    work_thu, work_sl = os.path.join(sl_thu, '_work'), os.path.join(sl_dir, '_work')
    n0 = tool_check['n0'] if tool_check else SET_ROWS['S0']
    if n0 > len(screen['S0']):
        raise s3.S03Stop(f'n0 = {n0} is more than the {len(screen["S0"])} rows of S0 '
                         '(S03 3.1b item 3)')
    pha = []

    def pha_entry(name, n_runs, n_rerun, seconds):
        return dict(pha=name, controller='all', n_bo_gain={c: None for c in controllers},
                    n_luot_ke_hoach=None, n_luot_thuc=n_runs, n_luot_thuc_theo_controller=None,
                    n_chay_lai_chan_doan=n_rerun, giay_may=seconds, so_tien_trinh=n_proc)

    # ── item 10: the S0 try-out (sealed/sangloc_thu/) ──
    rec0, ev0, score0_rows, sc0, t_s0, n_run0, n_rerun0 = _set_phase(
        'S0', n0, sl_thu, work_thu, os.path.join(sl_thu, 'tien_trinh_S0.log'), controllers, gate,
        mu_levels, W, screen, n_proc, dll)
    n_used0 = len(rec0[controllers[0]]) if controllers else 0
    thr0 = {ctrl: {m: threshold([sc0[ctrl][r['row']][m] for r in rec0[ctrl]]) for m in MODES}
            for ctrl in controllers}          # S0's own thresholds: for q and the CSV only
    for r in score0_rows:
        t_ = thr0[r['controller']][r['che_do']]
        r['thuoc_C'] = in_C(r, t_) if t_ is not None else ''
        r['trong_hop'] = r['hop'] = ''
    s3.write_csv(os.path.join(sl_thu, 'score_S0.csv'), score0_rows, SCORE_COLS)
    q_num, s0_over, dem0 = {}, False, {}
    for ctrl in controllers:
        n_ngoai0 = sum(r['flag'] == 'ngoai_bang' for r in rec0[ctrl])
        dem0[ctrl] = dict(n_ngoai_bang=n_ngoai0, che_do=_mode_stats(rec0[ctrl], sc0[ctrl]))
        for mode in MODES:
            st = dem0[ctrl]['che_do'][mode]
            s0_over = s0_over or over_error_limit(st['n_khong_phan_loai'], st['n_bo_gain'])
            q_num[(ctrl, mode)] = sum(in_C(sc0[ctrl][r['row']][mode], thr0[ctrl][mode])
                                      for r in rec0[ctrl])
            dem0[ctrl]['che_do'][mode]['n_trong_C'] = q_num[(ctrl, mode)]
    sd = sample_size(q_num, n_used0, t_s0, s0_over, tool_n=tool_check['n'] if tool_check else None)
    write_json(os.path.join(sl_thu, 'so_dem.json'),
               dict(sd, dem=dem0, kiem_cong_cu=tool_check,
                    nhanh_dieu_10='chay' if sd['chay'] else 'khong_chay_sang_loc'))
    pha.append(dict(pha_entry('S0', n_run0, n_rerun0, t_s0),
                    n_bo_gain={c: len(rec0[c]) for c in controllers},
                    n_luot_thuc_theo_controller={c: sum(e.get('attempt') == 1
                                                        for rows in ev0[c].values() for e in rows)
                                                 for c in controllers}))

    def result_doc(reasons_by):
        """ket_qua.json skeleton (item 11); reasons_by: {ctrl: [reason strings]}."""
        doc = dict(N=sd['N'], N_da_dung=0, m_min=M_MIN)
        for ctrl in controllers:
            reasons = reasons_by.get(ctrl, [])
            nhanh = 'nhan' if not reasons else next(
                b for b in BRANCH_PRIORITY if any(r.startswith(b) for r in reasons))
            doc[ctrl] = dict(nhanh=nhanh, trang_thai='hoan_tat', ly_do=reasons,
                             nguon_mien='sang_loc' if nhanh == 'nhan' else 'du_phong',
                             nguong={m: None for m in MODES},
                             hop={b: None for b in ('B_giua', 'B_ba_mu', 'B1', 'B2')}, mien=None,
                             che_do={m: dict(kiem_duoc_nhan_o_lan=None, can=None,
                                             tap={s2: dict(n_bo_gain=0, n_phan_loai_duoc=0, n_kha_thi=0,
                                                           n_thuoc_C=None, m=None, k=None,
                                                           n_C_lop_ngoai=None)
                                                  for s2 in SCREEN_SETS})
                                     for m in MODES},
                             n_ngoai_bang={s2: 0 for s2 in SCREEN_SETS},
                             ti_le_kha_thi_trong_mien={m: None for m in MODES},
                             n_cap_trung_float32=0, kiem_tat_dinh=None)
        return doc

    if not sd['chay']:
        write_json(os.path.join(sl_dir, 'ket_qua.json'),
                   result_doc({c: list(sd['ly_do']) for c in controllers}))
        return dict(mien={c: None for c in controllers},
                    nhanh={c: 'khong_chay_sang_loc' for c in controllers},
                    ly_do={c: list(sd['ly_do']) for c in controllers}, N_da_dung=0, pha=pha)

    N = sd['N_da_dung']
    for s_name in SCREEN_SETS:
        if N > len(screen[s_name]):
            raise s3.S03Stop(f'N = {N} is more than the {len(screen[s_name])} rows of {s_name} '
                             '(S03 3.1b item 3)')

    # ── S1: thresholds and the boxes (item 6-7) ──
    rec1, ev1, score1_rows, sc1, sec1, n_run1, n_rerun1 = _set_phase(
        'S1', N, sl_dir, work_sl, os.path.join(sl_dir, 'tien_trinh_S1.log'), controllers, gate,
        mu_levels, W, screen, n_proc, dll)
    pha.append(dict(pha_entry('S1', n_run1, n_rerun1, sec1),
                    n_bo_gain={c: len(rec1[c]) for c in controllers},
                    n_luot_thuc_theo_controller={c: sum(e.get('attempt') == 1
                                                        for rows in ev1[c].values() for e in rows)
                                                 for c in controllers}))
    thr = {ctrl: {m: threshold([sc1[ctrl][r['row']][m] for r in rec1[ctrl]]) for m in MODES}
           for ctrl in controllers}
    gains1 = {ctrl: {r['row']: r['g'] for r in rec1[ctrl]} for ctrl in controllers}
    boxes = {}
    for ctrl in controllers:
        if all(thr[ctrl].values()):
            b_giua = box_from_points([r['g'] for r in rec1[ctrl]
                                      if in_C(sc1[ctrl][r['row']]['giua'], thr[ctrl]['giua'])],
                                     W[ctrl])
            b_ba = box_from_points([r['g'] for r in rec1[ctrl]
                                    if in_C(sc1[ctrl][r['row']]['ba_mu'], thr[ctrl]['ba_mu'])],
                                   W[ctrl])
            boxes[ctrl] = dict(B_giua=b_giua, B_ba_mu=b_ba, B1=hull(b_giua, b_ba))
        else:                                       # a mode without threshold: no box (item 7)
            boxes[ctrl] = dict(B_giua=None, B_ba_mu=None, B1=None)
    for r in score1_rows:
        t_ = thr[r['controller']][r['che_do']]
        r['thuoc_C'] = in_C(r, t_) if t_ is not None else ''
        b = boxes[r['controller']]['B1']
        r['trong_hop'] = in_box(gains1[r['controller']][r['row']], b) if b is not None else ''
        r['hop'] = 'B1' if b is not None else ''
    s3.write_csv(os.path.join(sl_dir, 'score_S1.csv'), score1_rows, SCORE_COLS)
    khoa = dict(N=N, sha256={n: s3.sha256_file(os.path.join(sl_dir, n))
                             for n in ('points_S1.csv', 'evals_S1.csv', 'score_S1.csv')})
    for ctrl in controllers:
        khoa[ctrl] = dict(nguong={m: thr[ctrl][m] for m in MODES},
                          n_thuoc_C={m: (sum(in_C(sc1[ctrl][r['row']][m], thr[ctrl][m])
                                             for r in rec1[ctrl])
                                         if thr[ctrl][m] is not None else None)
                                     for m in MODES},
                          n_ngoai_bang=sum(r['flag'] == 'ngoai_bang' for r in rec1[ctrl]),
                          B_giua=_box_json(boxes[ctrl]['B_giua']),
                          B_ba_mu=_box_json(boxes[ctrl]['B_ba_mu']), B1=_box_json(boxes[ctrl]['B1']),
                          dem=_mode_stats(rec1[ctrl], sc1[ctrl]),
                          hon_10_phan_tram={m: over_error_limit(
                              _mode_stats(rec1[ctrl], sc1[ctrl])[m]['n_khong_phan_loai'],
                              len(rec1[ctrl])) for m in MODES})
    write_json(os.path.join(sl_dir, 'khoa_S1.json'), khoa)     # on disk before the first S2 job

    # ── S2: check 1 against B1, then B2 (item 8) ──
    rec2, ev2, score2_rows, sc2, sec2, n_run2, n_rerun2 = _set_phase(
        'S2', N, sl_dir, work_sl, os.path.join(sl_dir, 'tien_trinh_S2.log'), controllers, gate,
        mu_levels, W, screen, n_proc, dll)
    pha.append(dict(pha_entry('S2', n_run2, n_rerun2, sec2),
                    n_bo_gain={c: len(rec2[c]) for c in controllers},
                    n_luot_thuc_theo_controller={c: sum(e.get('attempt') == 1
                                                        for rows in ev2[c].values() for e in rows)
                                                 for c in controllers}))
    gains2 = {ctrl: {r['row']: r['g'] for r in rec2[ctrl]} for ctrl in controllers}
    kiem1, B2 = {}, {}
    for ctrl in controllers:
        b1 = boxes[ctrl]['B1']
        per = {}
        if b1 is not None:
            for mode in MODES:
                recs = [sc2[ctrl][r['row']][mode] for r in rec2[ctrl]]
                m, k = check_counts(recs, [r['g'] for r in rec2[ctrl]], thr[ctrl][mode], b1)
                per[mode] = dict(m=m, k=k, duoc_nhan=accepted(m, k), can=bound(m, k))
        else:
            # item 11: no box -> k null; a mode with a threshold still reports its count in C_j
            per = {m: dict(m=(sum(in_C(sc2[ctrl][r['row']][m], thr[ctrl][m]) for r in rec2[ctrl])
                              if thr[ctrl][m] is not None else None),
                           k=None, duoc_nhan=None, can=None) for m in MODES}
        kiem1[ctrl] = per
        if b1 is not None and not all(per[m]['duoc_nhan'] for m in MODES):
            out_pts = [r['g'] for r in rec2[ctrl]
                       if any(in_C(sc2[ctrl][r['row']][m], thr[ctrl][m]) and
                              not in_box(r['g'], b1) for m in MODES)]
            B2[ctrl] = extend_box(b1, out_pts, W[ctrl])
        else:
            B2[ctrl] = b1
        boxes[ctrl]['B2'] = B2[ctrl]
    for r in score2_rows:
        t_ = thr[r['controller']][r['che_do']]
        r['thuoc_C'] = in_C(r, t_) if t_ is not None else ''
        b = boxes[r['controller']]['B1']
        r['trong_hop'] = in_box(gains2[r['controller']][r['row']], b) if b is not None else ''
        r['hop'] = 'B1' if b is not None else ''
    s3.write_csv(os.path.join(sl_dir, 'score_S2.csv'), score2_rows, SCORE_COLS)
    kiem = dict(N=N, sha256={n: s3.sha256_file(os.path.join(sl_dir, n))
                             for n in ('points_S2.csv', 'evals_S2.csv', 'score_S2.csv')})
    for ctrl in controllers:
        kiem[ctrl] = dict(lan_1=kiem1[ctrl], B2=_box_json(B2[ctrl]))
    write_json(os.path.join(sl_dir, 'kiem_S2.json'), kiem)     # always, before S2b starts

    # ── S2b: check 2 against B2 (item 8) ──
    rec3, ev3, score3_rows, sc3, sec3, n_run3, n_rerun3 = _set_phase(
        'S2b', N, sl_dir, work_sl, os.path.join(sl_dir, 'tien_trinh_S2b.log'), controllers, gate,
        mu_levels, W, screen, n_proc, dll)
    pha.append(dict(pha_entry('S2b', n_run3, n_rerun3, sec3),
                    n_bo_gain={c: len(rec3[c]) for c in controllers},
                    n_luot_thuc_theo_controller={c: sum(e.get('attempt') == 1
                                                        for rows in ev3[c].values() for e in rows)
                                                 for c in controllers}))
    gains3 = {ctrl: {r['row']: r['g'] for r in rec3[ctrl]} for ctrl in controllers}
    kiem2, nhan_mode = {}, {}
    for ctrl in controllers:
        per = {}
        for mode in MODES:
            acc1 = kiem1[ctrl][mode]['duoc_nhan']
            if B2[ctrl] is None:
                per[mode] = dict(m=(sum(in_C(sc3[ctrl][r['row']][mode], thr[ctrl][mode])
                                        for r in rec3[ctrl])
                                    if thr[ctrl][mode] is not None else None),
                                 k=None, vai_tro=None, duoc_nhan=None, can=None)
                nhan_mode.setdefault(ctrl, {})[mode] = None
                continue
            recs = [sc3[ctrl][r['row']][mode] for r in rec3[ctrl]]
            m, k = check_counts(recs, [r['g'] for r in rec3[ctrl]], thr[ctrl][mode], B2[ctrl])
            if acc1:
                per[mode] = dict(m=m, k=k, vai_tro='chi_bao_cao', duoc_nhan=True, can=bound(m, k))
                nhan_mode.setdefault(ctrl, {})[mode] = 1
            else:
                acc2 = accepted(m, k)
                per[mode] = dict(m=m, k=k, vai_tro='quyet_dinh', duoc_nhan=acc2, can=bound(m, k))
                nhan_mode.setdefault(ctrl, {})[mode] = 2 if acc2 else None
        kiem2[ctrl] = per
    for r in score3_rows:
        t_ = thr[r['controller']][r['che_do']]
        r['thuoc_C'] = in_C(r, t_) if t_ is not None else ''
        b = B2[r['controller']]
        r['trong_hop'] = in_box(gains3[r['controller']][r['row']], b) if b is not None else ''
        r['hop'] = 'B2' if b is not None else ''
    s3.write_csv(os.path.join(sl_dir, 'score_S2b.csv'), score3_rows, SCORE_COLS)
    kiem_b = dict(N=N, sha256={n: s3.sha256_file(os.path.join(sl_dir, n))
                               for n in ('points_S2b.csv', 'evals_S2b.csv', 'score_S2b.csv')})
    for ctrl in controllers:
        kiem_b[ctrl] = dict(lan_2=kiem2[ctrl])
    write_json(os.path.join(sl_dir, 'kiem_S2b.json'), kiem_b)

    # ── item 12: the determinism check of the step ──
    scs = {'S1': sc1, 'S2': sc2, 'S2b': sc3}
    rec_sets = {'S1': (rec1, ev1), 'S2': (rec2, ev2), 'S2b': (rec3, ev3)}
    det_rows, det_chosen, det_st = determinism_check(
        'S1', rec1, ev1, dll, gate, mu_levels, work_sl, n_proc,
        os.path.join(sl_dir, 'tien_trinh_kiem_tat_dinh.log'))
    s3.write_csv(os.path.join(sl_dir, 'determinism_sangloc.csv'), det_rows, DET_COLS)
    if not all(r['log_arrays_equal'] and r['fields_equal'] for r in det_rows):
        raise s3.S03Stop('determinism check of the screening step failed (S03 3.1b item 12)')
    pha.append(dict(pha_entry('kiem_tat_dinh_sang_loc', det_st['n_luot_thuc'],
                              det_st['n_chay_lai'], det_st['seconds']),
                    n_bo_gain={c: len(det_chosen.get(c, [])) for c in controllers},
                    n_luot_thuc_theo_controller=None))

    # ── item 11: ket_qua.json, always when the step ends ──
    reasons_by, mien = {}, {}
    for ctrl in controllers:
        reasons = []
        for mode in MODES:
            if thr[ctrl][mode] is None:
                reasons.append(f'du_phong_khong_nguong: che do {mode} has no threshold '
                               '(S1 has no feasible gain set in it, S03 3.1b item 6)')
        for s_name, (recs_s, _) in rec_sets.items():
            for mode in MODES:
                st = _mode_stats(recs_s[ctrl], scs[s_name][ctrl])[mode]
                if over_error_limit(st['n_khong_phan_loai'], st['n_bo_gain']):
                    reasons.append(f'du_phong_loi_chay: {s_name} {mode}: '
                                   f'{st["n_khong_phan_loai"]}/{st["n_bo_gain"]} gain sets '
                                   'unclassifiable (> 10 %, S03 3.1b item 5)')
        if boxes[ctrl]['B1'] is not None:
            for mode in MODES:
                if nhan_mode[ctrl][mode] is None:
                    reasons.append(f'du_phong_kiem: che do {mode} was accepted by neither check '
                                   f'(k > 0 or m < {M_MIN}, S03 3.1b item 8)')
        reasons_by[ctrl] = reasons
        mien[ctrl] = B2[ctrl] if not reasons else None
    doc = result_doc(reasons_by)
    doc['N_da_dung'] = N
    for ctrl in controllers:
        c = doc[ctrl]
        c['nguong'] = thr[ctrl]
        for b in ('B_giua', 'B_ba_mu', 'B1', 'B2'):
            c['hop'][b] = _box_json(boxes[ctrl][b])
        c['mien'] = _box_json(mien[ctrl])
        c['n_ngoai_bang'] = {s_name: sum(r['flag'] == 'ngoai_bang' for r in rec_sets[s_name][0][ctrl])
                             for s_name in SCREEN_SETS}
        all_recs = [(r, sc_, s_name) for s_name, (recs_s, _) in rec_sets.items()
                    for r in recs_s[ctrl]
                    for sc_ in [(sc1 if s_name == 'S1' else sc2 if s_name == 'S2' else sc3)[ctrl]]]
        c['n_cap_trung_float32'] = sum(
            n_ * (n_ - 1) // 2 for n_ in Counter(
                tuple(float(np.float32(r['g'][f_])) for f_ in s3.FACTORS[ctrl])
                for r, _, _ in all_recs).values() if n_ > 1)
        dom = mien[ctrl] if mien[ctrl] is not None else quet[ctrl]
        for mode in MODES:
            cm = c['che_do'][mode]
            cm['kiem_duoc_nhan_o_lan'] = nhan_mode[ctrl][mode]
            check_m = kiem1[ctrl][mode] if nhan_mode[ctrl][mode] == 1 else kiem2[ctrl][mode]
            cm['can'] = check_m['can'] if nhan_mode[ctrl][mode] is not None else None
            for s_name, (recs_s, _) in rec_sets.items():
                sc_s = (sc1 if s_name == 'S1' else sc2 if s_name == 'S2' else sc3)[ctrl]
                st = _mode_stats(recs_s[ctrl], sc_s)
                t_ = thr[ctrl][mode]
                box = boxes[ctrl]['B1'] if s_name != 'S2b' else B2[ctrl]
                n_c = sum(in_C(sc_s[r['row']][mode], t_) for r in recs_s[ctrl]) \
                    if t_ is not None else None
                m_ = n_c                                            # item 11: m = count in C_j
                k_ = (sum(not in_box(r['g'], box) for r in recs_s[ctrl]
                          if in_C(sc_s[r['row']][mode], t_))
                      if box is not None else None)                 # no box -> k null
                layer = sum(1 for r in recs_s[ctrl]
                            if in_C(sc_s[r['row']][mode], t_) and any(
                                r['g'][f_] <= W[ctrl][f_][1] or r['g'][f_] >= W[ctrl][f_][-2]
                                for f_ in s3.FACTORS[ctrl])) if t_ is not None else None
                cm['tap'][s_name] = dict(n_bo_gain=st[mode]['n_bo_gain'],
                                         n_phan_loai_duoc=st[mode]['n_phan_loai_duoc'],
                                         n_kha_thi=st[mode]['n_kha_thi'],
                                         n_thuoc_C=n_c, m=m_, k=k_, n_C_lop_ngoai=layer,
                                         can=bound(m_, k_) if m_ is not None and k_ is not None
                                         else None)
            inside = [r for r, _, _ in all_recs if in_box(r['g'], dom)]
            feas = [r for r, sc_, _ in all_recs
                    if in_box(r['g'], dom) and sc_[r['row']][mode]['kha_thi']]
            c['ti_le_kha_thi_trong_mien'][mode] = dict(
                n_trong=len(inside), n_kha_thi=len(feas),
                ti_le=(len(feas) / len(inside)) if inside else None)
        c['kiem_tat_dinh'] = dict(n_dong=len(det_chosen.get(ctrl, [])),
                                  dong=list(det_chosen.get(ctrl, [])),
                                  ket_qua='dat' if det_chosen.get(ctrl)
                                  else 'bo_qua_khong_co_dong')
    write_json(os.path.join(sl_dir, 'ket_qua.json'), doc)
    return dict(mien=mien, nhanh={c: doc[c]['nhanh'] for c in controllers},
                ly_do={c: doc[c]['ly_do'] for c in controllers}, N_da_dung=N, pha=pha)


def skip_step(sealed, controllers, ly_do):
    """No screening at all (e.g. --trial with one mu level, part 2): only
    sealed/sangloc/ket_qua.json with every controller khong_chay_sang_loc and the one reason;
    returns the same dict shape as run_step."""
    sl_dir = os.path.join(sealed, 'sangloc')
    os.makedirs(sl_dir, exist_ok=True)
    doc = dict(N=None, N_da_dung=0, m_min=M_MIN)
    for ctrl in controllers:
        doc[ctrl] = dict(nhanh='khong_chay_sang_loc', trang_thai='hoan_tat', ly_do=[ly_do],
                         nguon_mien='du_phong', nguong={m: None for m in MODES},
                         hop={b: None for b in ('B_giua', 'B_ba_mu', 'B1', 'B2')}, mien=None,
                         che_do={m: dict(kiem_duoc_nhan_o_lan=None, can=None, tap={}) for m in MODES},
                         n_ngoai_bang={s: 0 for s in SCREEN_SETS},
                         ti_le_kha_thi_trong_mien={m: None for m in MODES},
                         n_cap_trung_float32=0, kiem_tat_dinh=None)
    write_json(os.path.join(sl_dir, 'ket_qua.json'), doc)
    return dict(mien={c: None for c in controllers},
                nhanh={c: 'khong_chay_sang_loc' for c in controllers},
                ly_do={c: [ly_do] for c in controllers}, N_da_dung=0, pha=[])


# ── section 7: the sealed levels file ────────────────────────────────────────────────

def levels_rows(ctrl, mien, W, grid_info, defaults, public_rows, nhanh, ly_do, gate):
    """Rows of sealed/levels_sangloc.csv for one controller (section 7). mien = the accepted
    box {factor: (L, H)} or None (fallback, item 9): the controller's rows of the public
    levels.csv unchanged plus nguon_mien 'du_phong' and the reasons. Else per factor: L, H from
    the box; scale/centre = s3.scale_of; a, b, b_n from grid_info[(ctrl, factor)] (the info dict
    of s3.sweep_grid_cut); L_nguon 'biên W' at the first node else 'sàng lọc'; H_nguon at the
    last node: 'trần bảng chặn' over 'biên W' when H = b_n = T (the cut), 'trần ω_o' for omega_o*
    at 48, else 'biên W'; below the last node 'sàng lọc'."""
    if mien is None:
        return [dict(r, nguon_mien='du_phong', ly_do='; '.join(ly_do)) for r in public_rows]
    rows = []
    for name in s3.FACTORS[ctrl]:
        L, H = mien[name]
        sc, centre = s3.scale_of(L, H)
        info = grid_info[(ctrl, name)]
        nodes = W[ctrl][name]
        l_src = 'biên W' if L == nodes[0] else 'sàng lọc'
        if H == nodes[-1]:
            if info.get('n_bo') and H == info.get('T'):
                h_src = 'trần bảng chặn'
            elif name.startswith('omega_o') and H == s3.OMEGA_O_MAX:
                h_src = 'trần ω_o'
            else:
                h_src = 'biên W'
        else:
            h_src = 'sàng lọc'
        rows.append(dict(controller=ctrl, factor=name, L=L, H=H, centre=centre, scale=sc,
                         default=defaults[name],
                         rule_note='screening box on the nodes of W (S03 3.1b), gate speeds '
                                   + ' '.join(map(str, gate)),
                         a=info['a'], b=info['b'], b_n=info['b_n'], L_nguon=l_src, H_nguon=h_src,
                         nguon_mien='sang_loc', ly_do=''))
    return rows


def read_levels(path, ctrl):
    """{factor: (L, H, scale)} of one controller read back from a levels_sangloc.csv (floats from
    the repr text, so bit-exact)."""
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['controller'] == ctrl:
                out[r['factor']] = (float(r['L']), float(r['H']), r['scale'])
    return out


if __name__ == '__main__':
    if '--job' in sys.argv:
        job_main(sys.argv[sys.argv.index('--job') + 1])
        sys.exit(0)
