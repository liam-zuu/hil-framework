"""
s04_ga.py: spec S04 v5 (CLAUDE_THESIS 06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md), NSGA-II
(pymoo) on fwsim choosing the 7 gains of one controller, objective B of L54 on the figure 8 of
O1 v7 (o1_feasibility.evaluate_v7 on the true plant pose, the same runs as S03: s03_doe.run_one).

Section numbers below are those of S04 v5.
  2   variables: the 7 S03 factors of the controller (s03_doe.FACTORS); fixed parameters as
      S03 2.1 (s03_doe.sim_kwargs: kd = 0, limits, b0, out_limit from the firmware). Domain =
      the S03 levels [L, H] (S04 v6.7 section 9 item 6: sealed/levels_sangloc.csv of the S03
      v8.5 screening, never the public levels.csv), not widened; a factor coded log in S03 is
      searched on log(gain). A chosen gain within 5 % of the coded width from a bound is flagged.
      S04 v6.8: every candidate is checked against the range table of the campaign binary
      (tools/gain_table.py, float32) before it runs; an ADRC candidate outside the triangle
      omega_c + 2 omega_o < 200 is not run (penalty, status ngoai_bang).
  3   one candidate = every gate speed (v) x mu level of the evaluation: middle mu only when
      the mu rule holds on every S03 design point (centre, outer, inner box) and gate speed,
      else the three mu levels, worst of. f1 = max e_perp_max (m); f2 = max over mu of the
      mean over v of tv_duty (1/s). g1 = number of runs not completed, inconclusive or with a
      track error; g2 = (f1 - D) / D; pymoo G <= 0. A run_error is run once more in a new
      process for diagnosis (attempt 2, column 'cause' left for CODE); the candidate then
      gets g1 = number of runs, f1 = f2 = 1e6, status run_error; the GA goes on.
      S04 v6.2-v6.3 (gate G1-G5, CS-23, SC-8, RS2-8; SC3-3; Liam 14/11 CODE_R134 local_eb9f3b5c
      "duyet goi sua code fix/r134-answers"): three more constraints g3 = (max e_trk_max - D) / D,
      g4 = (max e_bao_max - D) / D, g5 = (max e_theta_cmd_max_deg - 7.5) / 7.5, max over every run of
      the evaluation; e_bao_max from fwsim/o1_gate_eval.evaluate_fig8 (it calls evaluate_v7
      unchanged, so every older field keeps its value). A run_error run, or a run whose e_perp_max,
      e_trk_max, e_bao_max or e_theta_cmd_max_deg is missing or not finite, takes 1e6 for that
      quantity, flagged per run in evals.csv (pen_<quantity> = 1); history.csv has pen_f1 and
      n_pen_runs. G must be finite before pymoo sees it (else the run stops).
  4.2 GA seeds 1-5, plant seed 1 (s03_doe.SEED). Hypervolume per generation on the feasible
      non-dominated set of that generation's population, reference (D; 800 1/s) (v6.8),
      points on or beyond the reference counted, HV 0 when none.
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
      the report-only measures. S04 v6.8: the selected gains against the range table /
      triangle before HIL, plus the spectral radius of the linear sub-model of one
      translation axis (tools/mo_hinh_con.py; report only, not a GA constraint); G1-G5 read
      from the gate state columns; --phase sens_after: loop-delay variant (io_delay_ms) and
      the four plant variants of S04 v6.1 section 6, report only.
  7   K1, K2 are never run; nothing here runs on the car.
  8   outputs in --out (columns controller, ga_seed): history.csv, evals.csv (key controller,
      ga_seed, gen, cand_id, v, mu, attempt), hv.csv, pareto.csv (the non-dominated points of
      the feasible and, separately, of the infeasible points; column feasible), final_population.csv, selected.csv, meta.yaml (whole folder, written before the GA and after
      every phase), meta_<controller>.yaml, phases.json, lock.yaml (one mode and one source
      fingerprint for the folder), trial/ (the trial phase), post_checks/.

L56: the mu rule reads sealed/runs.csv of the S03 run, so --phase trial / main need
--s03-unsealed (campaign A finished, the S03 results may be looked at). S04 v6.7 section 9
item 6: the domain and the S03 design come from the sealed files of the S03 v8.5 screening
(sealed/levels_sangloc.csv, sealed/design.csv); they are read in every mode, so --tool-check
(tool test, never main numbers) takes an S03 --trial run (one mu, the mu rule is not applied),
any budget and seeds, no preconditions.

  python fwsim/s04_ga.py --phase trial --controller PID --s03 <S03 run> --s03-unsealed \\
      --preconditions <json> --out 05_DATA/results/S04/<date>_runNN
  python fwsim/s04_ga.py --phase main --controller PID --s03 <S03 run> --s03-unsealed --out <same dir>
  python fwsim/s04_ga.py --phase post --out <same dir>
  python fwsim/s04_ga.py --phase sens_after --out <same dir> [--io-delay-ms <measured ms>]
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
import stat
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim                 # noqa: E402
import o1_feasibility as o1  # noqa: E402
import s03_doe as s3         # noqa: E402
import o1_gate as og         # noqa: E402  (THETA_MAX_DEG of G5)
import o1_gate_eval as oge   # noqa: E402  (G4 e_bao_max; evaluate_v7 inside, unchanged)
from plant_blob import git_blob, same_plant   # noqa: E402  (rpi5/ is on sys.path through fwsim)

_TOOLS = os.path.join(HERE, '..', 'tools')
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import gain_table            # noqa: E402  (tools/: range table of the campaign binary, LADRC triangle)
import mo_hinh_con           # noqa: E402  (tools/: S04 v6.8 section 6 sub-model spectral radius, report only)

D = o1.D                      # m, O1 v7 (temporary)
F1_TOL = 0.01                 # m, section 3 / 5 (convention, same as S03 5.2)
MU_Y2_TOL = 0.01              # m, mu rule (section 3)
MU_Y3_TOL = 0.4               # 1/s, mu rule (section 3, v6.8): the absolute Y3 threshold of S03 5.2
HV_REF_F2 = 800.0             # 1/s, section 4.2 (v6.8): upper bound of f2 on a log with one sample per 10 ms tick
HV_EXT_REL = 0.01             # section 4.3
GEN_BASE, GEN_MAX, POP = 20, 40, 24
SEEDS = (1, 2, 3, 4, 5)
PENALTY = 1e6                 # section 3: f1 = f2 = 1e6 for a run_error candidate
SEED_ERR_LIMIT = 0.10         # section 3
EDGE_FRAC = 0.05              # section 2
SPEC = 'S04 v6.6 with v6.7 section 9 items 6, 7 and v6.8 section 9 item 9 (06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md)'   # S83-2: one version line for every meta.yaml
B0_FACTORS = (0.7, 0.85, 1.0, 1.15, 1.3)
R_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
REL_LIMIT = 0.5               # B0_SENS criterion: e_perp_max change > 50 %
PRECONDITIONS = ('s03_run_and_locked', 'final_test_set_locked', 'ctrl_imu_checked_on_board', 'x3_compared')
MU_RULE_BOXES = ('centre', 'outer', 'inner')        # S04 v6 section 9 item 4 (S03 v8 axial / test excluded)
MU_RULE_POINTS = ({('0', 'centre')} | {(str(i), 'outer') for i in range(1, 129)} |
                  {(str(i), 'inner') for i in range(129, 257)})   # the 257 point_id locked for the mu rule
# pymoo 0.6.2 constraint handling, read in the installed library (S04 section 3; Liam asked CODE to
# record file:line before the trial). Re-read when the version differs.
PYMOO_CHECKED = '0.6.2'
PYMOO_CV = ("core/individual.py:29 cv_ieq default func=np.sum, eps 0, no scale, no pow; "
            "core/individual.py:600-633 constr_to_cv: c = max(0, g_i - eps), CV = func(c) = sum (line 633) over "
            "the inequality constraints; core/survival.py:59-80 Survival.do: feasible (CV = 0) individuals ranked "
            "by RankAndCrowding first, the rest filled by increasing CV; algorithms/moo/nsga2.py:23-77 binary "
            "tournament: if either is infeasible the smaller CV wins (random on a tie, 39-48), else dominance "
            "then crowding (50-75)")
THETA_MAX_DEG = og.THETA_MAX_DEG    # deg, G5 (O1 v11.8 3.5; S04 v6.2 g5)
# S04 v6.2 SC-8 / v6.3 RS2-8: the run quantities replaced by PENALTY when missing, not finite or run_error,
# each with its own flag column pen_<quantity> in evals.csv (names fixed by CODE, S04 v6.3 section 3)
PEN_KEYS = ('e_perp_max', 'e_trk_max', 'e_bao_max', 'e_theta_cmd_max_deg')
PEN_COLS = ['pen_' + k for k in PEN_KEYS]
N_CONSTR = 5                  # g1 .. g5 (S04 v6.2)
KEP_FRAC_COLS = ['kep_frac_vx', 'kep_frac_vy', 'kep_frac_wz']   # S03 v8.6 section 11 item 6: PID clamp, report only
EVAL_COLS = ['controller', 'ga_seed', 'gen', 'cand_id', 'v', 'mu', 'attempt', 'status', 'completed', 'e_perp_max',
             'tv_duty', 'sat_frac', 't_end', 'e_theta_lane_max_deg', 'e_theta_cmd_max_deg', 'ds_absmax', 'e_trk_max',
             'e_bao_max'] + PEN_COLS + KEP_FRAC_COLS + ['kep_note'] + ['log_sha256', 'cause', 'error']
GATE_STATE_COLS = [f'{c}_state' for c in og.CONDS]   # g1_state .. g5_state of the gate scorer (o1_gate.gate)
MU_RULE_GATE = ('g3', 'g4', 'g5')                    # S04 v6.2 section 3: states that must not change with mu
GA_IO_DELAY_MS = 0            # S04 v6.8 section 6 / S03 v8.6 section 8 condition 12: the GA condition
GA_SLIP_OMEGA_S = 0.1
SENS_AFTER_DIR = 'sens_after'  # S04 v6.8 section 6: sensitivity check after selection, its own folder
DELAY_DEFAULT_MS = (1, 2)     # S04 v6.8 section 6: no measured bench loop delay yet
DELAY_INT_TOL = 1e-9          # ms: a measured delay this close to a whole number of 1 ms steps is that number
SENS_AFTER_COLS = ['tool_check', 'controller', 'variant', 'io_delay_ms', 'baseline', 'v', 'mu', 'status',
                   'completed', 'e_perp_max', 'e_trk_max', 'e_bao_max', 'e_theta_cmd_max_deg'] + GATE_STATE_COLS + \
    ['tv_duty']
# S04 v6.1 section 6 (RB-1): the plant variants of --phase sens_after (report only; K = motor_K and
# motor_u0 are never changed). S01 basis: matlab/identification/s01_identify.m lines 23-24, 97
# (J = K tau V / (R s), b = J / tau_coast), s01_derive_plant_params.py lines 13-16, 28
# (Tc = (Tc/J) J; u0 = u_int - Tc R / (K V) so u0 does not change; Ts = K/R V (u_start - u0)),
# rpi5/params.py lines 94-95.
PLANT_COUPLED = ('J_wheels', 'motor_b', 'motor_Tc', 'motor_Ts')   # arrays scaled 1/k with motor_R x k
PLANT_DERIVED = ('M_eff', 'M_eff_inv', 'traction_lam_max', 'adrc')   # recomputed by update_physical
PLANT_VARIANTS = ('R0.8_JbTcTs', 'R1.25_JbTcTs', 'R0.82_only', 'Vbatt11.1')
PLANT_VARIANTS_DEFS = {'R0.8_JbTcTs': 'motor_R x 0.8 and J_wheels, motor_b, motor_Tc, motor_Ts / 0.8',
                       'R1.25_JbTcTs': 'motor_R x 1.25 and J_wheels, motor_b, motor_Tc, motor_Ts / 1.25',
                       'R0.82_only': 'motor_R x 0.82, nothing else',
                       'Vbatt11.1': 'V_batt = 11.1 V, nothing else'}
PLANT_BASIS = ('S01 basis: matlab/identification/s01_identify.m lines 23-24, 97 (J = K tau V / (R s), '
               'b = J / tau_coast); s01_derive_plant_params.py lines 13-16, 28 (Tc = (Tc/J) J; '
               'u0 = u_int - Tc R / (K V) so u0 does not change; Ts = K/R V (u_start - u0)); '
               'rpi5/params.py lines 94-95')
PLANT_SENS_COLS = ['tool_check', 'controller', 'variant', 'baseline', 'v', 'mu', 'status', 'completed',
                   'e_perp_max', 'e_trk_max', 'e_bao_max', 'e_theta_cmd_max_deg'] + GATE_STATE_COLS + \
    ['tv_duty', 'plant_changed']
# S04 v6.7 section 9 item 6 / section 8: the sealed files of the S03 v8.5 screening the fingerprint locks
S03_SEALED_LOCK = ('sealed/levels_sangloc.csv', 'sealed/design.csv', 'sealed/sangloc/ket_qua.json',
                   'sealed/sangloc_thu/so_dem.json')
SO_GOC_META_KEY = 'screen_u'   # S03 v8.5 section 8 condition 9: meta.yaml key of the root number file record
S03_SCREEN_SCRIPT = 'fwsim/s03_sangloc.py'   # the S03 screening script (written in parallel), compared too


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


def digest_sha(digest):
    """SHA-256 of json.dumps({k: digest[k] for k in digest if k in fwsim.BIT_EXACT_KEYS}, sort_keys=True)
    (S03 v8.6 section 11 item 6: the kep channel is not in the compared bits); a key that is neither in
    BIT_EXACT_KEYS nor 'kep' -> ValueError."""
    bad = [k for k in digest if k not in fwsim.BIT_EXACT_KEYS and k != 'kep']
    if bad:
        raise ValueError(f'log digest keys {bad} are not fwsim.BIT_EXACT_KEYS (or kep)')
    return hashlib.sha256(json.dumps({k: digest[k] for k in digest if k in fwsim.BIT_EXACT_KEYS},
                                     sort_keys=True).encode()).hexdigest()


def log_sha(log):
    """'' for None else digest_sha(s3.log_digest(log)): hashed on fwsim.BIT_EXACT_KEYS only, so a log of a DLL
    with the kep channel and of an older DLL without it give the same value (for a log without 'kep' it is
    the value the old code gave); comparisons with runs of the old DLL are on those keys."""
    if log is None:
        return ''
    return digest_sha(s3.log_digest(log))


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

def _domain_rows(s03_dir, ctrl):
    """Levels rows of one controller from <s03>/sealed/levels_sangloc.csv (S04 v6.7 section 9 item 6: the
    sealed levels file of the S03 v8.5 screening, the columns of levels.csv plus nguon_mien and ly_do; the
    public levels.csv is never read). sys.exit when the file is missing (an S03 run before v8.5 cannot be
    used) or when nguon_mien of the controller's rows is not one identical value in ('sang_loc', 'du_phong')."""
    p = os.path.join(s03_dir, 'sealed', 'levels_sangloc.csv')
    if not os.path.isfile(p):
        sys.exit(f'{p} missing: no sealed/levels_sangloc.csv (the S03 v8.5 screening levels): an S03 run '
                 f'before v8.5 cannot be used (S04 v6.7 section 9 item 6)')
    rows = [r for r in read_csv(p) if r['controller'] == ctrl]
    nguon = {(r.get('nguon_mien') or '') for r in rows}
    if len(nguon) != 1 or next(iter(nguon)) not in ('sang_loc', 'du_phong'):
        sys.exit(f'{p}: nguon_mien of every {ctrl} row must be the same value, one of (sang_loc, du_phong); '
                 f'got {sorted(nguon)} (S04 v6.7 section 9 item 6)')
    return rows


def domain_source(s03_dir, ctrl):
    """dict(nguon_mien=..., ly_do=...) of the controller domain (S04 v6.7 section 9 item 6): the identical
    nguon_mien of its sealed levels rows, and the unique non-empty ly_do values joined by ' | ' ('' when
    none). Same rows and refusals as read_domain (one shared helper)."""
    rows = _domain_rows(s03_dir, ctrl)
    ly = sorted({(r.get('ly_do') or '') for r in rows} - {''})
    return dict(nguon_mien=rows[0]['nguon_mien'], ly_do=' | '.join(ly))


def _du_phong_tran_exception(s03_dir, ctrl, name, row, H):
    """S03 v8.6 section 3.1 rule (2), Codex C3-7 / C3-26: inside the du_phong refusal of read_domain, H above
    the original grid end b is accepted exactly when ALL hold: H == b_n == T as binary64 (T = s3.ceiling; a
    factor without a ceiling is not accepted), the H level source is TABLE_CEILING_SOURCE, b <= T and
    b within s3.TRAN_RTOL of T, and the PUBLIC <s03>/sweep.csv holds at least one row of this controller and
    factor with float(value) == H and every such row has tran == 'true' (rows with an empty value are
    skipped; no sweep.csv = not accepted)."""
    if (row.get('H_nguon') or '') != TABLE_CEILING_SOURCE:
        return False
    try:
        T = s3.ceiling(ctrl, name)
    except s3.S03Stop:
        return False
    try:
        b, b_n = float(row.get('b')), float(row.get('b_n'))
    except (TypeError, ValueError):
        return False
    if not (H == b_n == T and b <= T and math.isclose(b, T, rel_tol=s3.TRAN_RTOL)):
        return False
    p = os.path.join(s03_dir, 'sweep.csv')
    if not os.path.isfile(p):
        return False
    hit = [r for r in read_csv(p)
           if r['controller'] == ctrl and r['factor'] == name and (r.get('value') or '') != ''
           and float(r['value']) == H]
    return bool(hit) and all((r.get('tran') or '') == 'true' for r in hit)


def read_domain(s03_dir, ctrl):
    """[(name, L, H, scale, L_src, H_src, L_kiem, H_kiem)] of the controller from the S03
    sealed/levels_sangloc.csv (S04 v6.7 section 9 item 6; S03 order of FACTORS)."""
    rows = _domain_rows(s03_dir, ctrl)
    by = {r['factor']: r for r in rows}
    nguon = rows[0]['nguon_mien']
    kiem = read_out_of_range(s03_dir, ctrl)
    out = []
    for n in s3.FACTORS[ctrl]:
        if n not in by:
            sys.exit(f'{s03_dir}/sealed/levels_sangloc.csv has no level for {ctrl} {n}')
        L, H = float(by[n]['L']), float(by[n]['H'])
        if not L < H:
            sys.exit(f'{ctrl} {n}: L = {L} is not below H = {H} (S03 3.1 (ii))')
        # S04 v6.6: the GA domain is [L, H] of the ORIGINAL grid (S03 v8.4 3.1); levels of S03 v8.3 (set on
        # the extension) are refused. S04 v6.7 section 9 item 6: the refusal applies only to the 'du_phong'
        # source (for 'sang_loc' H may exceed b, up to b_n); the C3-7 / C3-26 exception accepts H = T.
        if nguon == 'du_phong' and (
                (by[n].get('H_nguon') or '') == 'hết lưới nối'
                or ((by[n].get('b') or '') != '' and H > float(by[n]['b'])
                    and not _du_phong_tran_exception(s03_dir, ctrl, n, by[n], H))):
            sys.exit(f'{ctrl} {n}: H = {H} is set on the extension of the S03 grid (S03 v8.3 levels): S04 v6.6 needs '
                     f'the levels of S03 v8.4 (original grid only)')
        # S04 v6.4 (C14-2): the source of each level, from the S03 levels file ('' for an S03 run before v8.3);
        # v6.6: with the result of the out-of-range check on each side (S03 v8.4 kiem_ngoai_mien.csv, '' without it)
        out.append(dict(name=n, L=L, H=H, scale=by[n]['scale'], L_src=by[n].get('L_nguon') or '',
                        H_src=by[n].get('H_nguon') or '', L_kiem=kiem.get((n, 'L'), ''), H_kiem=kiem.get((n, 'H'), '')))
    return out


KIEM_FILE = 'kiem_ngoai_mien.csv'      # S03 v8.4 section 3.1


def kiem_file(s03_dir):
    """S84-3: the path of the S03 v8.4 check file, None when the S03 run has none. Refused before it is opened or
    hashed when it is a link or junction, resolves anywhere else (a link on its path), is not a plain file or has
    a second hard link: it must never alias a sealed file (L56; every caller of read_domain and fingerprint, the
    RSM script too, goes through here)."""
    p = os.path.join(s03_dir, KIEM_FILE)
    if not os.path.lexists(p):
        return None
    real = os.path.normcase(os.path.realpath(p))
    if (os.path.islink(p) or getattr(os.path, 'isjunction', lambda q: False)(p)
            or real != os.path.normcase(os.path.abspath(p))):
        sys.exit(f'{p} is a link or resolves to {real}: refused (S04 v6.6, L56)')
    st = os.lstat(p)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
        sys.exit(f'{p}: not a plain file with one name ({st.st_nlink} links): refused (S04 v6.6, L56)')
    return p


def read_out_of_range(s03_dir, ctrl):
    """S04 v6.6 section 2: {(factor, 'L' | 'H'): summary} of the S03 v8.4 kiem_ngoai_mien.csv for one controller,
    e.g. '4 điểm: không cờ 4, chưa kết luận 0, cờ 0' (a side without points: '0 điểm'), with '(lượt kiểm: <total>)'
    when the total of the controller is not 'đã kiểm đủ'. {} when the S03 run has no such file (before v8.4)."""
    p = kiem_file(s03_dir)
    if p is None:
        return {}
    rows = [r for r in read_csv(p) if r['controller'] == ctrl]
    tot = [r['ket_qua'] for r in rows if r['dong'] == 'tong']
    tail = '' if tot == ['đã kiểm đủ'] else f' (lượt kiểm: {" ".join(tot) or "không có dòng tổng"})'
    out = {}
    for n in s3.FACTORS[ctrl]:
        for side in ('L', 'H'):
            res = [r['ket_qua'] for r in rows if r['dong'] == 'diem' and r['factor'] == n and r['can'] == side]
            out[(n, side)] = (f'{len(res)} điểm' + (': ' + ', '.join(f'{k} {res.count(k)}' for k in s3.KIEM_POINT)
                                                    if res else '') + tail)
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


GRID_END_SOURCES = ('hết lưới', 'hết lưới dưới')   # S03 v8.4 level sources: the original grid ended there (v6.6)
TABLE_CEILING_SOURCE = 'trần bảng chặn'   # S03 v8.6 3.1 level source: H = T, the range-table ceiling of the gain
EDGE_LIMIT_TABLE = 'vùng ngoài bảng chặn không thuộc thí nghiệm (DUT từ chối gain đó)'   # S04 v6.8 section 2


def edge_sides(dom, x):
    """[(name, 'L' | 'H')] of the factors flagged by edge_flags (same formula), with the side of the bound
    they are near (S04 v6.4 section 2)."""
    lo, hi = ga_bounds(dom)
    out = []
    for d, xi, a, b in zip(dom, x, lo, hi):
        w = EDGE_FRAC * (b - a)
        if xi - a <= w:
            out.append((d['name'], 'L'))
        elif b - xi <= w:
            out.append((d['name'], 'H'))
    return out


def edge_report(dom, x):
    """selected.csv edge columns (S04 v6.4 section 2, C14-2; v6.6): edge = the flagged factors (unchanged);
    edge_side = factor:side; edge_src = factor:side:source of that bound (S03 levels.csv, '?' when the S03 run
    has none); edge_limit = the flagged factors whose bound on that side is a grid end ('hết lưới',
    'hết lưới dưới'): the limit 'vùng ngoài biên chưa khảo sát bằng GA', each with the result of the S03 v8.4
    out-of-range check on that side ('?' when the S03 run has no check file). A flagged bound whose source is
    'trần bảng chặn' adds EDGE_LIMIT_TABLE (no out-of-range result: the grid is cut at T); the two kinds are
    joined by ' | '. No flag, no limit, whatever the sources. The four keys are fixed (rsm_compare.py writes
    them for the RSM candidates, section 12)."""
    by = {d['name']: d for d in dom}
    sides = edge_sides(dom, x)
    src = [(n, s, by[n].get('L_src' if s == 'L' else 'H_src') or '') for n, s in sides]
    lim = [f'{n}:{s} [kiểm ngoài miền: {by[n].get(s + "_kiem") or "?"}]' for n, s, c in src if c in GRID_END_SOURCES]
    tab = [f'{n}:{s}' for n, s, c in src if c == TABLE_CEILING_SOURCE]
    limits = ((['vùng ngoài biên chưa khảo sát bằng GA: ' + ' '.join(lim)] if lim else [])
              + ([EDGE_LIMIT_TABLE + ': ' + ' '.join(tab)] if tab else []))
    return dict(edge=' '.join(edge_flags(dom, x)), edge_side=' '.join(f'{n}:{s}' for n, s in sides),
                edge_src=' | '.join(f'{n}:{s}:{c or "?"}' for n, s, c in src),
                edge_limit=' | '.join(limits))


def gate_state_of(r, c, where):
    """State of gate condition c ('g3' ...) of one S03 sealed/runs.csv row, read from the gate's state column
    <c>_state (o1_gate.gate: undefined frames counted; S03 v8.2 section 7), never derived from a maximum against
    its threshold (S04 v6.7 section 9 item 7). A row that is not a run_error must hold one of o1_gate.STATES:
    else the mu rule cannot be applied and the script stops."""
    x = r.get(c + '_state')
    if x in og.STATES:
        return x
    if r.get('status') == 'run_error':
        return x or ''
    sys.exit(f'{where}: {c}_state is {x!r}, not one of {og.STATES}: no gate state column of the S03 run '
             f'(S03 v8.2 section 7; S04 v6.7 section 9 item 7)')


def mu_within(x, ref, thr):
    """S04 section 3 (C3-6): |x - ref| <= thr with the comparison convention of S03 (s03_doe.exact, not_above_by:
    inclusive, exact on the decimal of each number as written in runs.csv, no tolerance). A value that is not
    finite is not within (not comparable, as s03_doe.comparable)."""
    return (math.isfinite(x) and math.isfinite(ref)
            and s3.not_above_by(x, ref, thr) and s3.not_above_by(ref, x, thr))


def mu_rule(s03_dir, ctrl, gate, mu_levels):
    """Section 3: ('mid', report) when every S03 design point (the 257 points of the centre,
    outer and inner boxes; S03 v8 axial and test points are not used) at every gate
    speed has the same Y1 at the three mu, the same G3, G4 and G5 state at the three mu
    (gate state columns of sealed/runs.csv), |dY2| <= 0.01 m and |dY3| <= 0.4 1/s (S04 v6.8:
    the absolute threshold of S03 5.2) versus the middle mu, no track error, no inconclusive;
    else ('all', report). The dY2 / dY3 bounds are compared as S03 compares (mu_within,
    s03_doe.not_above_by); a Y2 or Y3 that is not finite does not satisfy the rule (C3-6)."""
    # S04 v6 section 9 item 4 (S03 v8 adds axial and test points): the mu rule uses only the
    # 257 points of the centre, outer and inner boxes, filtered by box in BOTH files before the
    # completeness and membership checks; the point_id list used is locked to 0..256
    runs = [r for r in read_csv(os.path.join(s03_dir, 'sealed', 'runs.csv'))
            if r['controller'] == ctrl and r['box'] in MU_RULE_BOXES]
    # S04 v6.7 section 9 item 6: the design of the screening, sealed/design.csv (never the public one)
    design = [(r['point_id'], r['box']) for r in read_csv(os.path.join(s03_dir, 'sealed', 'design.csv'))
              if r['controller'] == ctrl and r['box'] in MU_RULE_BOXES]
    if not design or len(set(design)) != len(design):
        sys.exit(f'{s03_dir}/sealed/design.csv: no or duplicate design points for {ctrl} (S04 3, S4C-5)')
    if set(design) != MU_RULE_POINTS:
        sys.exit(f'{s03_dir}/sealed/design.csv: the {len(design)} centre / outer / inner points of {ctrl} are not '
                 f'point_id 0..256 with their boxes (S04 v6 section 9 item 4)')
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
            sys.exit(f'{s03_dir}/sealed/runs.csv: {ctrl} point {pid} ({box}) not in sealed/design.csv (S4C-5)')
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
            for key, tol, text in (('e_perp_max', MU_Y2_TOL, 'dY2 > 0.01 m'),
                                   ('tv_duty', MU_Y3_TOL, 'dY3 > 0.4 1/s')):
                y = {m: float(r[key]) for m, r in rs.items()}
                if not all(math.isfinite(x) for x in y.values()):
                    why.append(f'{key} not finite')
                elif not (mu_within(y[lo], y[mid], tol) and mu_within(y[hi], y[mid], tol)):
                    why.append(text)                # C3-6: the S03 convention, no tolerance of its own
            for c in MU_RULE_GATE:
                st = {gate_state_of(r, c, f'{s03_dir}/sealed/runs.csv: {ctrl} point {pid} ({box}) v = {v} mu = {m}')
                      for m, r in rs.items()}
                if len(st) > 1:
                    why.append(f'{c} state differs over mu')
        if why:
            bad.append(dict(point_id=pid, box=box, v=v, why=why))
    # S04 v6 section 9 item 4: the point_id list actually used goes to lock.yaml (via mu_rule_report)
    used = sorted(int(p) for p, _ in design)
    return ('mid' if not bad and n else 'all'), dict(points_checked=n, points_failing=len(bad), first=bad[:5],
                                                     n_point_ids=len(used), point_ids=used)


# ── one candidate ────────────────────────────────────────────────────────────────────

def require_classify():
    """C3-25: the candidate check needs the classify() Verdict of tools/gain_table.py (the check() text
    fallback is removed: the wording of a message is never a violation class). S03Stop when the module has
    no callable classify; the callers that simulate turn it into sys.exit: nothing is simulated without
    classify()."""
    if not callable(getattr(gain_table, 'classify', None)):
        raise s3.S03Stop('tools/gain_table.py has no callable classify(): the range-table / triangle check '
                         'needs its Verdict (C3-25); nothing may be simulated without it')


def gain_check(ctrl, g):
    """S04 v6.8 section 2 (S03 v8.6 2.1, 3.1b item 4): one gain set against the range table of the campaign
    binary and the LADRC triangle, on the float32 values the firmware gets. The check itself is in
    tools/gain_table.py (never recomputed here; its warnings are not violations): the classify() Verdict,
    required (C3-25, no text fallback). Returns (table, triangle): the violations of one gain against its
    own bounds and the other violations (ADRC: omega_c + 2 omega_o of an axis not below 200). S03Stop for a
    gain set the table cannot read and for a PID violation that is not a bound (PID has no triangle)."""
    require_classify()
    try:
        v = gain_table.classify(ctrl, dict(g))
        table = list(v.vi_pham_bang) or ([] if v.trong_bang else ['outside the range table'])
        other = list(v.vi_pham_tam_giac) or ([] if v.trong_tam_giac or not v.trong_bang
                                             else ['outside the LADRC triangle'])
    except ValueError as e:
        raise s3.S03Stop(f'{ctrl}: gain set {g!r} refused by tools/gain_table ({e})') from None
    if other and ctrl != 'ADRC':
        raise s3.S03Stop(f'{ctrl}: tools/gain_table reports {other}: not a bound of one gain (tool error)')
    return table, other


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
        table, tri = gain_check(self.ctrl, g)          # S04 v6.8 section 2: before any run of the candidate
        if table:
            raise s3.S03Stop(f'{self.ctrl} candidate (ga_seed {ga_seed}, gen {gen}, cand {cand}): gain outside '
                             f'the range table of the campaign binary ({"; ".join(table)}): tool error, the GA '
                             f'domain lies inside the table (S04 v6.8 section 2)')
        if tri:
            return score_ngoai_bang(len(self.mus) * len(self.speeds))
        rows, error = [], False
        for mu in self.mus:
            for v in self.speeds:
                row, log = safe_run_gate(self.fw, self.path, self.ctrl, g, v, mu)
                row = tv_duty_run_error(row)         # S04 v6.7 section 3: at the run level, before the row is kept
                self._rec(ga_seed, gen, cand, v, mu, 1, row, log, **run_values(row)[1])
                rows.append((mu, v, row))
                if row['status'] == 'run_error':
                    error = True
                    try:                     # section 3: once more in a new process, for diagnosis only
                        again = s3.run_in_new_process(self.dll, [(self.ctrl, g, v, mu)], self.work)[0]
                        # diagnosis only: s03_doe's evaluator; an s03_doe without e_bao_max gives pen_e_bao_max = 1
                        # here (an s03_doe that writes G1-G5, hil feat/s03-gate, has it; S3G-6)
                        self._rec(ga_seed, gen, cand, v, mu, 2, again['row'], None,
                                  log_sha256=digest_sha(again['digest']) if again.get('digest') else '',
                                  **run_values(again['row'])[1])
                    except Exception as exc:  # noqa: BLE001
                        self._rec(ga_seed, gen, cand, v, mu, 2, dict(status='diagnosis_failed'), None,
                                  error=f'{type(exc).__name__}: {exc}', **run_values(dict(status='run_error'))[1])
        return score(rows, error)


def kep_fields(ctrl, log, t_end):
    """S03 v8.6 section 11 item 6: the PID integral clamp indicator ("tick cham kep") of one run log, the
    three kep_frac_* of fwsim.kep_indicator and kep_note ''. Report only: it never changes status or score.
    ADRC: three '' and kep_note ''. On a ValueError of kep_indicator: three '' and kep_note = the message."""
    if ctrl == 'ADRC':
        return dict(kep_frac_vx='', kep_frac_vy='', kep_frac_wz='', kep_note='')
    try:
        k = fwsim.kep_indicator(log, t_end)
        return dict(kep_frac_vx=k['kep_frac_vx'], kep_frac_vy=k['kep_frac_vy'], kep_frac_wz=k['kep_frac_wz'],
                    kep_note='')
    except ValueError as e:
        return dict(kep_frac_vx='', kep_frac_vy='', kep_frac_wz='', kep_note=str(e))


def run_gate(fw, path, ctrl, g, v, mu):
    """s03_doe.run_one with the gate evaluator (S04 v6.2: G4 needs e_bao_max): the same simulation,
    checks and fields, the summary of o1_gate_eval.evaluate_fig8, which runs evaluate_v7 unchanged and
    adds e_bao_max. Returns (row, log); raises on a DLL / NaN error, as run_one."""
    sch = o1.Schedule(v)
    p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=mu))
    if p.slip_omega_s != GA_SLIP_OMEGA_S:        # S04 v6.8 section 6 / S03 v8.6 section 8 condition 12
        raise s3.S03Stop(f'plant slip_omega_s {p.slip_omega_s!r} is not {GA_SLIP_OMEGA_S}: the GA runs with '
                         f'slip_omega_s = {GA_SLIP_OMEGA_S} and io_delay_ms = {GA_IO_DELAY_MS}')
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, p, seed=s3.SEED, mode=ctrl, imu=dict(s3.IMU),
                         io_delay_ms=GA_IO_DELAY_MS, **s3.sim_kwargs(fw, ctrl, g))
    for k in ('pose', 'v_true', 'duty', 'pwm'):
        if not np.all(np.isfinite(log[k])):
            raise FloatingPointError(f'non-finite {k} in the log')
    res, _, _ = oge.evaluate_fig8(log, path, sch, lsum=p.lx + p.ly)
    val = fwsim.run_validity(log)
    if val['run_bad'] == 1:
        raise RuntimeError(f"pose reset mid-run in fwsim ({val['run_bad_reason']})")
    row = {k: res[k] for k in s3.RUN_COLS[5:] if k in res}
    row.update(pos_reset_mid=val['pos_reset_mid'], e_bao_max=res['e_bao_max'],
               **kep_fields(ctrl, log, res['t_end']))
    return row, log


def safe_run_gate(fw, path, ctrl, g, v, mu):
    """run_gate, or a 'run_error' row (S03 section 4, S04 section 3) with the reason. S03Stop (a tool or
    configuration error, s03_doe.sim_kwargs) goes up as in s03_doe.safe_run (Codex R134C-7)."""
    try:
        return run_gate(fw, path, ctrl, g, v, mu)
    except s3.S03Stop:
        raise
    except Exception as exc:                 # noqa: BLE001 (any DLL / numeric / evaluator failure)
        return dict(status='run_error', completed='', error=f'{type(exc).__name__}: {exc}'), None


def _finite(x):
    """float(x) when it is a finite number, else None ('' and None are missing)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def run_values(row):
    """({quantity: value}, {pen_<quantity>: 0 / 1}) of one run for PEN_KEYS (S04 v6.2 SC-8, v6.3 RS2-8):
    a run_error run, or a value missing or not finite, gives PENALTY with its flag at 1. A flagged value
    is never evidence of an observed violation (S04 12.3)."""
    err = row.get('status') == 'run_error'
    vals, flags = {}, {}
    for k in PEN_KEYS:
        x = None if err else _finite(row.get(k))
        vals[k] = PENALTY if x is None else x
        flags['pen_' + k] = int(x is None)
    return vals, flags


def tv_duty_run_error(row):
    """S04 v6.7 section 3 (section 9 item 7): a run that is not a run_error but whose tv_duty is missing or not
    finite counts as a run_error, at the run level: the same row with status 'run_error' and the reason in
    'error' (the other fields are kept as scored)."""
    if row.get('status') == 'run_error' or _finite(row.get('tv_duty')) is not None:
        return row
    return dict(row, status='run_error', error=f'tv_duty {row.get("tv_duty")!r} missing or not finite: counted as '
                                               f'a run_error (S04 v6.7 section 3)')


def score_ngoai_bang(n):
    """S04 v6.8 section 2 (G-Q2): a candidate outside the LADRC triangle is not run; it gets the penalty of a
    run_error candidate of section 3 (g1 = n, the number of runs of the evaluation mode; f1, f2 and the gate
    quantities 1e6) with status 'ngoai_bang'. Same keys as score(); no run was made (n_runs 0, no evals row), so
    it is in neither part of the seed error rate (CC1-3)."""
    return dict(f1=PENALTY, f2=PENALTY, g1=float(n), g2=(PENALTY - D) / D, g3=(PENALTY - D) / D,
                g4=(PENALTY - D) / D, g5=(PENALTY - THETA_MAX_DEG) / THETA_MAX_DEG, status='ngoai_bang', n_runs=0,
                n_false=0, n_inconclusive=0, n_track_err=0, n_run_error=0, pen_f1=1, n_pen_runs=0)


def score(rows, error):
    """f1, f2, g1 .. g5 and counts of one candidate from its [(mu, v, row)] (section 3, S04 v6.2)."""
    # S04 v6.7 section 3 (C3-9): a run with tv_duty missing or not finite counts as a run_error also when the
    # caller did not convert the row (Evaluator.candidate does): the SAME converted row gives the score, the
    # penalty flags and the counts
    rows = [(mu, v, tv_duty_run_error(r)) for mu, v, r in rows]
    error = bool(error) or any(r.get('status') == 'run_error' for _, _, r in rows)
    n = len(rows)
    counts = dict(n_runs=n, n_false=sum(r['completed'] == 'false' for _, _, r in rows),
                  n_inconclusive=sum(r['completed'] == 'inconclusive' for _, _, r in rows),
                  n_track_err=sum(r['status'] == 'track_err' for _, _, r in rows),
                  n_run_error=sum(r['status'] == 'run_error' for _, _, r in rows))
    vals, flags = zip(*(run_values(r) for _, _, r in rows)) if rows else ((), ())
    counts.update(pen_f1=int(error or any(f['pen_e_perp_max'] for f in flags)),
                  n_pen_runs=sum(any(f.values()) for f in flags))
    if error:
        f1 = f2 = PENALTY
        g1 = float(n)
        q = dict(e_trk_max=PENALTY, e_bao_max=PENALTY, e_theta_cmd_max_deg=PENALTY)
        status = 'run_error'
    else:
        f1 = max(v['e_perp_max'] for v in vals)
        mus = sorted({mu for mu, _, _ in rows})
        f2 = max(float(np.mean([float(r['tv_duty']) for m, _, r in rows if m == mu])) for mu in mus)
        g1 = float(sum(r['completed'] != 'true' or r['status'] == 'track_err' for _, _, r in rows))
        q = {k: max(v[k] for v in vals) for k in ('e_trk_max', 'e_bao_max', 'e_theta_cmd_max_deg')}
        status = 'ok'
    s = dict(f1=f1, f2=f2, g1=g1, g2=(f1 - D) / D, g3=(q['e_trk_max'] - D) / D, g4=(q['e_bao_max'] - D) / D,
             g5=(q['e_theta_cmd_max_deg'] - THETA_MAX_DEG) / THETA_MAX_DEG, status=status, **counts)
    bad = [k for k in ('g1', 'g2', 'g3', 'g4', 'g5') if not math.isfinite(s[k])]
    if bad:                                  # S04 v6.2 section 3: G is checked finite before the optimiser
        raise RuntimeError(f'constraint(s) {bad} not finite: {s}')
    return s


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
            super().__init__(n_var=len(dom), n_obj=2, n_ieq_constr=N_CONSTR, xl=lo, xu=hi)

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = [], []
            state['ids'] = [(state['gen'], k) for k in range(len(X))]
            for k, x in enumerate(X):
                g = decode(dom, x)
                s = ev.candidate(g, ga_seed, state['gen'], k)
                F.append([s['f1'], s['f2']])
                G.append([s[f'g{i}'] for i in range(1, N_CONSTR + 1)])
                # S4C-6: every evaluation of this vector (pymoo may evaluate a vector again)
                seen.setdefault(np.asarray(x, dtype=float).tobytes(), []).append(
                    (state['gen'], k, (float(s['f1']), float(s['f2']),
                                       *(float(s[f'g{i}']) for i in range(1, N_CONSTR + 1)))))
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
        fg = (float(ind.F[0]), float(ind.F[1]), *(float(x) for x in G))
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
        pts.append(dict(ga_seed=ga_seed, gen=gen, cand_id=cand, f1=fg[0], f2=fg[1],
                        **{f'g{i}': fg[1 + i] for i in range(1, len(G) + 1)},
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
                         **edge_report(dom, ch['xga']), **gain_cols(dom, ch)))
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
                         cand_id=ch['cand_id'], f1=ch['f1'], f2=ch['f2'], **edge_report(dom, ch['xga']),
                         **gain_cols(dom, ch)))
    return rows, dict(n_seeds=n_seeds, n_feasible_seeds=n_feas, per_seed_selected=stats)


def seed_error_rates(evals, ctrl, seeds=None):
    """Section 3 (v4, B8b-7): per seed, first-attempt runs inconclusive or run_error / first-attempt runs.
    S04 v6.8 section 2 (CC1-3): a row with status 'ngoai_bang' (should a tool write one) is in neither the
    numerator nor the denominator. seeds = the seeds run (the locked list in a main run); a seed of that
    list with no first-attempt run (every candidate ngoai_bang, not run) is reported with n = 0 and rate
    None (JSON null), not left out (C3-10)."""
    out = {int(s): [0, 0] for s in seeds or ()}
    for r in evals:
        if r['controller'] != ctrl or int(r['attempt']) != 1 or r.get('status') == 'ngoai_bang':
            continue
        c = out.setdefault(int(r['ga_seed']), [0, 0])
        c[1] += 1
        c[0] += r['completed'] == 'inconclusive' or r['status'] == 'run_error'
    return {s: dict(rate=a / n if n else None, n=n, over_10pct=n > 0 and a / n > SEED_ERR_LIMIT)
            for s, (a, n) in sorted(out.items())}


def ngoai_bang_counts(history, ctrl):
    """S04 v6.8 section 2: {ga_seed: number of candidates with status 'ngoai_bang'} of one controller, for every
    seed in history (0 when it has none)."""
    out = {}
    for h in history:
        if h['controller'] == ctrl:
            out[int(h['ga_seed'])] = out.get(int(h['ga_seed']), 0) + int(h.get('status') == 'ngoai_bang')
    return dict(sorted(out.items()))


POP_COLS = ['controller', 'ga_seed', 'gen', 'cand_id', 'feasible', 'nondominated_feasible', 'nondominated_in_group',
            'f1', 'f2', 'g1', 'g2', 'g3', 'g4', 'g5', 'id_note', 'gains']


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
                             f2=p['f2'], **{f'g{i}': p[f'g{i}'] for i in range(1, N_CONSTR + 1)},
                             id_note=p.get('id_note', ''),
                             gains=json.dumps([repr(float(v)) for v in p['x']])))
    return rows


def pareto_rows(ctrl, finals):
    return population_rows(ctrl, finals, nd_only=True)


def default_f2(fw, dll, ctrl, speeds, mu_mid, work):
    """f2 of the firmware default gains at the middle mu. Until S04 v6.7 this gave the HV reference; since v6.8
    the reference is (D; 800 1/s) (section 4.2) and the trial no longer calls this function (kept for the
    record)."""
    ev = Evaluator(fw, dll, ctrl, speeds, [mu_mid], work)
    s = ev.candidate(s3.defaults(fw, ctrl), 0, 0, 0)
    ok = s['status'] == 'ok' and s['n_run_error'] == 0 and math.isfinite(s['f2']) and s['f2'] > 0
    return s['f2'], s, ev.evals, ok


# ── post checks (section 6) ──────────────────────────────────────────────────────────

def gate_states(res, frames, sch):
    """{g1_state .. g5_state} of one run from the gate scorer o1_gate.gate, called as s03_doe.run_one calls it
    (U = 0 on fwsim, every condition deciding); the states count the undefined frames (S04 v6.7 9.7)."""
    v_cmd_max, wz_cmd_max = s3.cmd_max(sch)
    gt = og.gate(frames, res['completed'], v_cmd_max=v_cmd_max, wz_cmd_max=wz_cmd_max, u_t=s3.GATE_U_T,
                 decide=og.DECIDE_ALL, status=res['status'], g1_fields=oge.g1_fields_v7(res))
    return {f'{c}_state': gt[c]['state'] for c in og.CONDS}


def apply_plant_variant(p, name):
    """S04 v6.1 section 6 (RB-1): one plant variant applied to the plant parameter object, ONE
    p.update_physical(...) call so the derived matrices are recomputed. motor_K and motor_u0 are never
    changed. S01 basis: matlab/identification/s01_identify.m lines 23-24, 97 (J = K tau V / (R s),
    b = J / tau_coast), s01_derive_plant_params.py lines 13-16, 28 (Tc = (Tc/J) J; u0 = u_int - Tc R / (K V)
    so u0 does not change; Ts = K/R V (u_start - u0)), rpi5/params.py lines 94-95. Returns
    dict(changed=[sorted keys of plant_dump that differ], expected=[primary keys], derived=[changed keys in
    PLANT_DERIVED]); raises s3.S03Stop unless the changed keys are exactly the expected primary keys plus a
    subset of PLANT_DERIVED (coupled: all of {motor_R} + PLANT_COUPLED; 'R0.82_only': only motor_R;
    'Vbatt11.1': only V_batt)."""
    if name not in PLANT_VARIANTS:
        raise s3.S03Stop(f'unknown plant variant {name!r} (S04 v6.1 section 6: {PLANT_VARIANTS})')
    before = plant_dump(p)
    if name == 'R0.8_JbTcTs':
        k = 0.8
        p.update_physical(motor_R=p.motor_R * k, **{n: getattr(p, n) / k for n in PLANT_COUPLED})
    elif name == 'R1.25_JbTcTs':
        k = 1.25
        p.update_physical(motor_R=p.motor_R * k, **{n: getattr(p, n) / k for n in PLANT_COUPLED})
    elif name == 'R0.82_only':
        p.update_physical(motor_R=p.motor_R * 0.82)
    else:                                        # Vbatt11.1
        p.update_physical(V_batt=11.1)
    after = plant_dump(p)
    changed = sorted(n for n in before if before[n] != after[n])
    if name in ('R0.8_JbTcTs', 'R1.25_JbTcTs'):
        expected = ['motor_R'] + list(PLANT_COUPLED)
        ok = all(n in changed for n in expected) and set(changed) - set(expected) <= set(PLANT_DERIVED)
    elif name == 'R0.82_only':
        expected, ok = ['motor_R'], changed == ['motor_R']
    else:
        expected, ok = ['V_batt'], changed == ['V_batt']
    if not ok:
        raise s3.S03Stop(f'plant variant {name}: the changed plant parameters {changed} are not the expected '
                         f'{expected} plus a subset of PLANT_DERIVED {PLANT_DERIVED} (S04 v6.1 section 6, RB-1)')
    return dict(changed=changed, expected=expected, derived=[n for n in changed if n in PLANT_DERIVED])


def run_custom(fw, path, ctrl, g, v, mu, b0_factor=1.0, r_factor=1.0, io_delay_ms=0, plant_variant=None):
    """s03_doe.run_one with b0 (LADRC) and / or motor R scaled; io_delay_ms = pure delay on the duty in 1 ms
    steps (fwsim.simulate); plant_variant = one of PLANT_VARIANTS (apply_plant_variant, applied after
    r_factor, before the `after` dump). The row is the gate evaluator summary (o1_gate_eval.evaluate_fig8,
    evaluate_v7 unchanged inside) plus the g1_state .. g5_state of the gate scorer (gate_states, S04 v6.7
    9.7). Returns (row, log, read-back, plant before/after)."""
    sch = o1.Schedule(v)
    p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=mu))
    before = plant_dump(p)
    if r_factor != 1.0:
        p.update_physical(motor_R=p.motor_R * r_factor)
    if plant_variant is not None:
        apply_plant_variant(p, plant_variant)
    after = plant_dump(p)
    kw = s3.sim_kwargs(fw, ctrl, g)
    if ctrl == 'ADRC' and b0_factor != 1.0:
        kw['adrc'] = {i: (a, b, b0 * b0_factor, lim) for i, (a, b, b0, lim) in kw['adrc'].items()}
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, p, seed=s3.SEED, mode=ctrl, imu=dict(s3.IMU),
                         io_delay_ms=io_delay_ms, **kw)
    back = [fw.get_adrc(i) for i in range(3)] if ctrl == 'ADRC' else [fw.get_pid(i) for i in range(3)]
    back.append(dict(pos=fwsim.pos_meta(fw, raw=True)))  # exact float32 pos_kp, pos_ki (S4C-8); k_psi has no getter (S3C-21)
    res, _, frames = oge.evaluate_fig8(log, path, sch, lsum=p.lx + p.ly)   # evaluate_v7 unchanged inside
    res.update(gate_states(res, frames, sch))
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
    """B0_SENS criterion (written before, B0_SENS/2026-11-05_run01) per (v, mu) against factor 1.0; the
    G3, G4, G5 states are read from the gate state columns (S04 v6.2 section 6, v6.7 9.7)."""
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
            for c in MU_RULE_GATE:               # S04 v6.2 section 6: a G3, G4 or G5 state change is 'sensitive'
                if r[f'{c}_state'] != nom[f'{c}_state']:
                    reasons.append(f'{tag}: {c.upper()} state {nom[f"{c}_state"]} -> {r[f"{c}_state"]}')
    return bool(reasons), reasons


def three_mu_findings(rows, ctrl):
    """Runs of the three-mu check not passing the gate G1-G5 (S04 v6.2 section 6; v6.7 section 9 item 7). The
    states are read from the gate's state columns g1_state .. g5_state (o1_gate.gate: undefined frames counted),
    never derived from a maximum against its threshold. A run passes only with status 'ok' (S4C-9: a track_err /
    run_error run is not a passed gate) and the five states 'dat'; a missing state is not a pass."""
    out = []
    for r in rows:
        if r['controller'] != ctrl:
            continue
        st = {c: r.get(f'{c}_state') or 'missing' for c in og.CONDS}
        if r.get('status') != 'ok' or any(s != 'dat' for s in st.values()):
            out.append(f"v={r['v']} mu={r['mu']}: status {r.get('status')}, "
                       + ', '.join(f'{c.upper()} {s}' for c, s in st.items())
                       + f", completed {r.get('completed')}, e_perp_max {r.get('e_perp_max')}, e_trk_max "
                         f"{r.get('e_trk_max')}, e_bao_max {r.get('e_bao_max')}, e_theta_cmd_max_deg "
                         f"{r.get('e_theta_cmd_max_deg')}")
    return out


def submodel_spectral_radius(ctrl, g):
    """S04 v6.8 section 6: spectral radius of the linear sub-model for the whole gain set (tools/mo_hinh_con.py:
    one translation axis without saturation and clamps; the retained states are those of ma_tran). The float,
    or None when mo_hinh_con raises ValueError (a gain set it cannot read)."""
    try:
        return mo_hinh_con.ban_kinh_pho(ctrl, g)
    except ValueError:
        return None


def pre_hil_check(ctrl, g):
    """S04 v6.8 section 6, before the selected gains go to HIL: inside the range table of the campaign binary
    (float32) and, for LADRC, inside the triangle of each axis (gain_check, i.e. tools/gain_table.py), plus the
    spectral radius of the linear sub-model (mo_hinh_con.ban_kinh_pho_va_trang_thai; report only, NOT a GA
    constraint). A failure is reported (tell Liam); the gains are never chosen again here. require_classify()
    is called OUTSIDE the try: a missing classify() is a tool error, never a table violation."""
    require_classify()                          # C3-25
    try:
        table, tri = gain_check(ctrl, g)
    except s3.S03Stop as e:
        table, tri = [str(e)], []
    try:
        m = mo_hinh_con.ban_kinh_pho_va_trang_thai(ctrl, g)
        rho, state, kept, dropped = (m[k] for k in
                                     ('ban_kinh_pho', 'trang_thai', 'bien_trang_thai', 'bien_bo'))
        model = dict(G=m['G'], tau=m['tau'], T=m['T'], b0=m['b0'], module='tools/mo_hinh_con.py',
                     sha256=s3.sha256_file(os.path.join(_TOOLS, 'mo_hinh_con.py')))
        note = ('report only, not a GA constraint: linear sub-model of one translation axis without '
                'saturation and clamps (tools/mo_hinh_con.py, Codex C3 round-1 note (a))')
    except ValueError as e:
        rho, state, kept, dropped, model = None, None, None, None, None
        note = f'not computed: {e} (tools/mo_hinh_con.py; S04 v6.8 section 6)'
    return dict(ok=not table and not tri, in_table=not table, in_triangle=(not tri) if ctrl == 'ADRC' else None,
                table_violations=table, triangle_violations=tri,
                spectral_radius=rho, spectral_radius_state=state, spectral_radius_states=kept,
                spectral_radius_dropped=dropped, spectral_radius_model=model, spectral_radius_note=note)


CHO_LIAM_NOTE = ('S04 section 6: cho_liam not empty = tell Liam before HIL (post returns 2); the selected gains are '
                 'kept, never chosen again here. It lists the pre-HIL check (range table, LADRC triangle: check '
                 'pre_hil), the spectral radius of the linear sub-model (check spectral_radius: rho >= 1, or rho '
                 'not computable although the gains are inside the table), the three-mu check (a run of the '
                 'selected gains not passing G1-G5 at a mu level, gate state columns: check three_mu) and the '
                 'b0 / R sensitivity result (check b0_factor / r_factor: judge() says sensitive)')


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
    check_s03_trial(common, 'post')             # S04 v6.7 9.6 (SL4-4): before anything under sealed/
    try:
        require_classify()                    # C3-25: nothing is simulated without classify()
    except s3.S03Stop as e:
        sys.exit(str(e))
    check_fingerprint(common, fingerprint(a.dll, common['s03_run'], unsealed=common.get('unsealed', False)),
                      'post')                                                         # S4C-2
    tc = common['tool_check']
    os.makedirs(pc, exist_ok=True)
    t0 = time.time()
    fw, path = fwsim.FirmwareSim(a.dll), o1.Path8()
    det, sens, mu3, notes, plant_full = [], [], [], {}, {}
    cho_liam = []                       # S04 section 6: what Liam must see before HIL; the selected gains are kept
    for s in sel:
        ctrl = s['controller']
        if s.get('feasible') != 'true':
            notes[ctrl] = 'no send_to_HIL gains: ' + s.get('note', '')
            continue
        L = lock['controllers'][ctrl]
        g = {n: float(s[n]) for n in s3.FACTORS[ctrl]}          # read back from the file (section 6)
        ph = notes[f'{ctrl} pre_hil'] = pre_hil_check(ctrl, g)  # S04 v6.8 section 6
        if not ph['ok']:
            cho_liam.append(dict(controller=ctrl, check='pre_hil',
                                 reasons=ph['table_violations'] + ph['triangle_violations']))
        # S04 v6.8 section 6 (report only, never a reselection): rho >= 1, or rho not computable although
        # the set is inside the table, waits for Liam
        if ph['spectral_radius_state'] == mo_hinh_con.KHONG_ON_DINH:
            cho_liam.append(dict(controller=ctrl, check='spectral_radius',
                                 reasons=[f'spectral radius {ph["spectral_radius"]!r} >= 1: the linear sub-model '
                                          f'is not stable (tools/mo_hinh_con.py; S04 v6.8 section 6)']))
        elif ph['spectral_radius'] is None and ph['in_table']:
            cho_liam.append(dict(controller=ctrl, check='spectral_radius',
                                 reasons=[f'spectral radius not computable although the gains are inside the '
                                          f'range table: {ph["spectral_radius_note"]}']))
        mus, speeds = L['eval_mu'], L['speeds']
        key = (s['ga_seed'], s['gen'], s['cand_id'])
        ref = {(float(e['v']), float(e['mu'])): e['log_sha256'] for e in evals
               if e['controller'] == ctrl and (e['ga_seed'], e['gen'], e['cand_id']) == key and e['attempt'] == '1'}
        jobs = [(ctrl, g, v, mu) for mu in mus for v in speeds]
        res = s3.run_in_new_process(a.dll, jobs, os.path.join(pc, '_work'))
        for (c, _, v, mu), r in zip(jobs, res):
            got = digest_sha(r['digest']) if r['digest'] else ''
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
                                         e_trk_max=res_['e_trk_max'], e_bao_max=res_['e_bao_max'],
                                         e_theta_cmd_max_deg=res_['e_theta_cmd_max_deg'],
                                         **{c: res_[c] for c in GATE_STATE_COLS},
                                         read_back=json.dumps(back), plant_changed=json.dumps(
                                             {x: (before[x], after[x]) for x in before if before[x] != after[x]})))
            for r in rows:
                nom = next(x for x in rows if x['v'] == r['v'] and x['mu'] == r['mu'] and x[kind] == 1.0)
                r['tv_duty_change_pct'] = 100.0 * (r['tv_duty'] - nom['tv_duty']) / nom['tv_duty'] if nom['tv_duty'] else ''
            if rows:
                sensitive, reasons = judge(rows, kind)
                notes[f'{ctrl} {kind}'] = dict(sensitive=sensitive, reasons=reasons)
                if sensitive:                 # S04 v6.8 section 6: a sensitive factor waits for Liam
                    cho_liam.append(dict(controller=ctrl, check=kind, reasons=reasons))
                sens += rows
        if len(mus) == 1:                       # section 6: the three mu when the GA used the middle one
            levels = lock['s03_mu_levels']
            for mu in levels:
                for v in speeds:
                    row, _ = s3.safe_run(fw, path, ctrl, g, v, mu)
                    mu3.append(dict(tool_check=tc, controller=ctrl, v=v, mu=mu,
                                    **{k: row.get(k, '') for k in EVAL_COLS + GATE_STATE_COLS if k in row}))
            warn = three_mu_findings(mu3, ctrl)
            if len(set(levels)) < 3:            # S4C-10: never an empty list for a check that was not done
                notes[f'{ctrl} three_mu'] = dict(
                    done=False, note=f'not applicable: the S03 run has the mu levels {levels} only; the three-mu '
                                     'check of S04 6 was NOT done', findings=warn)
            else:
                notes[f'{ctrl} three_mu'] = dict(done=True, findings=warn,
                                                 note='findings: runs not passing the gate; empty = every run has '
                                                      'status ok and G1-G5 dat (gate state columns)')
                if warn:                        # C3-8: a run not passing G1-G5 waits for Liam
                    cho_liam.append(dict(controller=ctrl, check='three_mu', reasons=warn))
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
                       cho_liam=cho_liam, cho_liam_note=CHO_LIAM_NOTE,
                       script=s3.git_state(os.path.join(HERE, '..')), dll_sha256=s3.sha256_file(a.dll),
                       criterion='B0_SENS/2026-11-05_run01: sensitive if completed changes, e_perp_max <= D changes, '
                                 'a G3, G4 or G5 state changes (gate state columns, S04 v6.2), or e_perp_max changes '
                                 'by more than 50 % of the factor-1.0 value; tv_duty change reported, not a criterion '
                                 '(S04 6, B8b-21); rank after the change not reported',
                       r_factors=R_FACTORS, b0_factors=B0_FACTORS), f, indent=1, default=str)
    log_phase(out, dict(phase='post', elapsed_s=round(time.time() - t0)))
    write_root_meta(out, a.dll, lock)
    print(f'post checks -> {pc}; tool_check {tc}; determinism equal: {all(d["equal"] for d in det) if det else None}')
    for k, v in notes.items():
        print(f'  {k}: {v}')
    if any(c['check'] == 'pre_hil' for c in cho_liam):
        print('TELL LIAM before HIL: selected gains outside the range table or the LADRC triangle (S04 v6.8 '
              'section 6); no reselection')
    if any(c['check'] == 'spectral_radius' for c in cho_liam):
        print('TELL LIAM before HIL: the spectral radius of the linear sub-model is >= 1 or not computable '
              '(tools/mo_hinh_con.py; S04 v6.8 section 6; post_checks/notes.json cho_liam); no reselection')
    if any(c['check'] == 'three_mu' for c in cho_liam):
        print('TELL LIAM before HIL: the selected gains do not pass G1-G5 at a mu level of the three-mu check '
              '(S04 section 6; post_checks/notes.json cho_liam); no reselection')
    if any(c['check'] in ('b0_factor', 'r_factor') for c in cho_liam):
        print('TELL LIAM before HIL: the b0 / R sensitivity check reports a sensitive factor '
              '(S04 v6.8 section 6; post_checks/notes.json cho_liam); no reselection')
    return 2 if cho_liam else 0


def delay_steps(measured_ms):
    """S04 v6.8 section 6 (S03 v8.6 section 8 condition 12): (io_delay_ms values, rule text) of the loop-delay
    variant, in whole 1 ms steps (fwsim.simulate takes whole steps). measured_ms None (no measured bench loop
    delay yet): 1 and 2 ms. A measured delay within DELAY_INT_TOL of a whole number of ms: that number; else both
    the rounding down and the rounding up. Not finite or negative: refused."""
    if measured_ms is None:
        return list(DELAY_DEFAULT_MS), 'no measured bench loop delay: 1 and 2 ms (S04 v6.8 section 6)'
    x = float(measured_ms)
    if not math.isfinite(x) or x < 0:
        sys.exit(f'--io-delay-ms {measured_ms!r}: a finite delay >= 0 in ms')
    n = round(x)
    if abs(x - n) <= DELAY_INT_TOL:
        return [int(n)], f'measured {x!r} ms is a whole number of 1 ms steps (within {DELAY_INT_TOL} ms): {int(n)} ms'
    return [math.floor(x), math.ceil(x)], (f'measured {x!r} ms is not a whole number of 1 ms steps: both the rounding '
                                           f'down ({math.floor(x)} ms) and up ({math.ceil(x)} ms)')


def sens_after(a):
    """S04 v6.1 / v6.8 section 6, sensitivity check after selection, REPORT ONLY (never used to change or choose
    gains): the loop-delay variant and the four plant variants of PLANT_VARIANTS (S04 v6.1 section 6, RB-1).
    The send_to_HIL gains read back from selected.csv run at the GA speeds and mu with io_delay_ms = 0 and no
    plant variant (the GA condition: one baseline run per (controller, mu, v), used for BOTH tables), the values
    of delay_steps(--io-delay-ms) and each plant variant. Reported per run: completed, e_perp_max, the G3-G5
    values and the G1-G5 states of the gate (gate state columns), tv_duty; plant.csv adds plant_changed (JSON
    list of the changed plant keys). Output only in <out>/sens_after/ (delay.csv, plant.csv, plant_params.json,
    notes.json): phases.json, meta.yaml, lock.yaml and post_checks/ are not written, so the inputs the RSM
    comparison hashes (section 12) do not change."""
    out = a.out
    lock = json.load(open(os.path.join(out, 'lock.yaml'), encoding='utf-8'))
    sel = [r for r in read_csv(os.path.join(out, 'selected.csv')) if r['kind'] == 'send_to_HIL']
    d = os.path.join(out, SENS_AFTER_DIR)
    if os.path.isdir(d) and os.listdir(d):
        sys.exit(f'{d} already holds data')
    common = lock.get('common')
    if not common:
        sys.exit(f'{out}/lock.yaml has no common block (written by this script version): run a new folder')
    check_s03_trial(common, 'sens_after')       # S04 v6.7 9.6 (SL4-4): before anything under sealed/
    try:
        require_classify()                    # C3-25: nothing is simulated without classify()
    except s3.S03Stop as e:
        sys.exit(str(e))
    check_fingerprint(common, fingerprint(a.dll, common['s03_run'], unsealed=common.get('unsealed', False)),
                      'sens_after')
    measured = getattr(a, 'io_delay_ms', None)
    steps, rule = delay_steps(measured)
    tc = common['tool_check']
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    fw, path = fwsim.FirmwareSim(a.dll), o1.Path8()
    rows, prows, notes, entries = [], [], {}, {}
    for s in sel:                               # RB-1: every (eval mu, variant) of EVERY controller is
        ctrl = s['controller']                  # declared valid BEFORE the first simulation of any of them
        if s.get('feasible') != 'true':
            continue
        for mu in lock['controllers'][ctrl]['eval_mu']:
            for name in PLANT_VARIANTS:
                p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=mu))
                before = plant_dump(p)
                nsub0 = fwsim.traction_nsub(p)
                rep = apply_plant_variant(p, name)            # a changed-key mismatch stops (S03Stop)
                entries[f'{ctrl} {name} mu={mu!r}'] = dict(before=before, after=plant_dump(p),
                                                           slip_nsub=[nsub0, fwsim.traction_nsub(p)], **rep)
    for s in sel:
        ctrl = s['controller']
        if s.get('feasible') != 'true':
            notes[ctrl] = 'no send_to_HIL gains: ' + s.get('note', '')
            continue
        L = lock['controllers'][ctrl]
        g = {n: float(s[n]) for n in s3.FACTORS[ctrl]}          # read back from the file (section 6)
        for mu in L['eval_mu']:
            for v in L['speeds']:
                base, _, _, _, _ = run_custom(fw, path, ctrl, g, v, mu, io_delay_ms=GA_IO_DELAY_MS)
                rows.append(dict(tool_check=tc, controller=ctrl, variant='io_delay_ms',
                                 io_delay_ms=GA_IO_DELAY_MS, baseline=True, v=v, mu=mu,
                                 **{c: base[c] for c in SENS_AFTER_COLS[7:]}))
                prows.append(dict(tool_check=tc, controller=ctrl, variant='baseline', baseline=True, v=v, mu=mu,
                                  **{c: base[c] for c in PLANT_SENS_COLS[6:-1]}, plant_changed='[]'))
                for k in [x for x in steps if x != GA_IO_DELAY_MS]:
                    res, _, _, _, _ = run_custom(fw, path, ctrl, g, v, mu, io_delay_ms=k)
                    rows.append(dict(tool_check=tc, controller=ctrl, variant='io_delay_ms', io_delay_ms=k,
                                     baseline=False, v=v, mu=mu,
                                     **{c: res[c] for c in SENS_AFTER_COLS[7:]}))
                for name in PLANT_VARIANTS:
                    res, _, _, before, after = run_custom(fw, path, ctrl, g, v, mu, plant_variant=name)
                    prows.append(dict(tool_check=tc, controller=ctrl, variant=name, baseline=False, v=v, mu=mu,
                                      **{c: res[c] for c in PLANT_SENS_COLS[6:-1]},
                                      plant_changed=json.dumps(sorted(k for k in before
                                                                      if before[k] != after[k]))))
    write_rows(os.path.join(d, 'delay.csv'), rows, SENS_AFTER_COLS)
    write_rows(os.path.join(d, 'plant.csv'), prows, PLANT_SENS_COLS)
    with open(os.path.join(d, 'plant_params.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(note='every attribute of the plant parameters used, before and after the variant '
                            '(one entry per controller, variant and eval mu; the same for every speed); '
                            'slip_nsub = [before, after] of fwsim.traction_nsub',
                       definitions=PLANT_VARIANTS_DEFS, entries=entries), f, indent=1, default=str)
    with open(os.path.join(d, 'notes.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(tool_check=tc, report_only=True, variant='io_delay_ms', measured_io_delay_ms=measured,
                       io_delay_ms_run=steps, rule=rule, baseline_io_delay_ms=GA_IO_DELAY_MS,
                       variants=['io_delay_ms'] + list(PLANT_VARIANTS), plant_variants=PLANT_VARIANTS_DEFS,
                       plant_basis=PLANT_BASIS,
                       when='after the seal is opened (S04 section 6); report only; the script cannot check '
                            'the seal',
                       notes=notes, elapsed_s=round(time.time() - t0), script=s3.git_state(os.path.join(HERE, '..')),
                       dll_sha256=s3.sha256_file(a.dll)), f, indent=1, default=str)
    print(f'sensitivity after selection (report only) -> {d}; io_delay_ms {steps} ({rule}), plant variants '
          f'{PLANT_VARIANTS}; tool_check {tc}')
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phase', required=True, choices=('trial', 'main', 'post', 'sens_after'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--controller', choices=s3.CONTROLLERS)
    ap.add_argument('--s03', help='S03 run folder (meta.yaml, levels.csv, sealed/levels_sangloc.csv, '
                                  'sealed/design.csv, sealed/runs.csv)')
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
    ap.add_argument('--io-delay-ms', type=float, default=None,
                    help='--phase sens_after only: measured loop delay of the HIL bench [ms] (S06); not a whole '
                         'number: both the rounding down and up are run; not given: 1 and 2 ms (S04 v6.8 section 6)')
    a = ap.parse_args()
    if a.io_delay_ms is not None and a.phase != 'sens_after':
        sys.exit('--io-delay-ms only with --phase sens_after: the GA, its trial and the post checks run with '
                 'io_delay_ms = 0 and slip_omega_s = 0.1 (S04 v6.8 section 6; S03 v8.6 section 8 condition 12)')
    if a.phase == 'post':
        return post(a)
    if a.phase == 'sens_after':
        return sens_after(a)
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
    try:
        s03_trial = meta_value(s03_meta, 'trial') in (True, 'true')
    except KeyError:
        s03_trial = False                            # a missing trial line counts as not trial
    if a.tool_check and not s03_trial:               # S04 v6.7 9.6 (SL4-4): BEFORE anything under sealed/
        sys.exit(f'--tool-check takes an S03 --trial run; {a.s03} is not one (trial: true): nothing under '
                 f'sealed/ is read for a tool check')
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
    try:
        require_classify()                    # C3-25: nothing is simulated without classify()
    except s3.S03Stop as e:
        sys.exit(str(e))
    fw = fwsim.FirmwareSim(a.dll)
    if ctrl == 'PID' and not getattr(fw, 'kep_cols', 0):
        sys.exit('the DLL has no kep channel (fwsim_kep_cols = 0): a DLL without the clamp export of S03 v8.6 '
                 'section 11 item 6 (c) cannot run S04; rebuild it')
    if fw.build_info.get('esp32', {}).get('dirty', True):
        print('WARNING: fwsim.dll was built from a dirty firmware tree; not usable as a main result')
    import pymoo
    t0 = time.time()

    fp = fingerprint(a.dll, a.s03, unsealed=a.s03_unsealed and not a.tool_check)
    check_s03_sealed(fp, a.tool_check)            # S04 v6.7 section 8
    src_diff = check_s03_source(s03_meta, fp, a.dll, a.tool_check)   # S4C-2: the source the S03 run used

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
                                           s03_mu_levels=s03_mu, unsealed=bool(a.s03_unsealed and not a.tool_check),
                                           s03_source_diff=src_diff)
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
                                                 'or sealed/runs.csv is not read (L56)')
        else:
            decision, report = mu_rule(a.s03, ctrl, gate, s03_mu)
        eval_mu = [mu_mid] if decision == 'mid' else list(s03_mu)
        os.makedirs(tdir, exist_ok=True)
        L_new = dict(tool_check=a.tool_check, domain=dom, s03_run=os.path.abspath(a.s03), speeds=gate, eval_mu=eval_mu,
                     mu_decision=decision, mu_rule_report=report, pop=a.pop, gen_rule=dict(base=a.gen_base, max=a.gen_max),
                     seeds=list(a.seeds or SEEDS), preconditions=pre, n_gen='set by this trial (S04 4.3)',
                     **domain_source(a.s03, ctrl))                # nguon_mien, ly_do (S04 v6.7 9.6)
        # S4C-12: the root meta.yaml holds the locked items and the pymoo reading before the GA runs
        write_root_meta(a.out, a.dll, dict(lock, controllers=dict(lock['controllers'], **{ctrl: L_new})),
                        running=f'trial {ctrl}')
        work = os.path.join(tdir, '_work')
        ref = (D, HV_REF_F2)                    # S04 v6.8 section 4.2: fixed reference, the default gains do not run
        ev = Evaluator(fw, a.dll, ctrl, gate, eval_mu, work)
        history, hv_rows = [], []
        final, n_gen, extended = run_seed(ev, dom, SEEDS[0], a.pop, None, ref, history, hv_rows,
                                          ext=(a.gen_base, a.gen_max))
        feasible_final = any(p['feasible'] for p in final)
        locked_gen = a.gen_max if extended else a.gen_base
        stop = (extended and not feasible_final)
        check_eval_keys(ev.evals)                                             # S4C-13
        write_rows(os.path.join(tdir, 'history.csv'), history)
        write_rows(os.path.join(tdir, 'evals.csv'), ev.evals, EVAL_COLS)
        write_rows(os.path.join(tdir, 'hv.csv'), hv_rows)
        write_rows(os.path.join(tdir, 'pareto.csv'), pareto_rows(ctrl, {SEEDS[0]: final}), POP_COLS)
        write_rows(os.path.join(tdir, 'final_population.csv'), population_rows(ctrl, {SEEDS[0]: final}), POP_COLS)
        elapsed = time.time() - t0
        L_new.update(n_gen=locked_gen, gen_rule=dict(
            base=a.gen_base, max=a.gen_max, compared_with=gen_compare(a.gen_base),
            hv_base=hv_rows[a.gen_base - 1]['hv'] if len(hv_rows) >= a.gen_base else None,
            hv_compared=hv_rows[gen_compare(a.gen_base) - 1]['hv'] if len(hv_rows) >= a.gen_base else None,
            extended=extended), hv_ref=list(ref),
            trial_elapsed_s=round(elapsed), trial_generations=n_gen, trial_feasible_final=feasible_final,
            stop=('no feasible point after the extended trial: tell Liam (S04 4.3)' if stop else None))
        lock['controllers'][ctrl] = L_new
        lock.update(s03_mu_levels=s03_mu, rule=SELECTION_RULE, pymoo=pymoo.__version__)
        with open(lock_path, 'w', encoding='utf-8') as f:
            json.dump(lock, f, indent=1, default=str)
        write_meta(os.path.join(tdir, 'meta.yaml'), a, fw, ctrl, dict(
            phase='trial', elapsed_s=round(elapsed), generations=n_gen, extended=extended, locked_n_gen=locked_gen,
            mu_decision=decision, eval_mu=eval_mu, speeds=gate, pop=a.pop, seeds_locked=L_new['seeds'],
            gen_rule=L_new['gen_rule'], hv_ref=list(ref), preconditions=pre,
            n_ngoai_bang=ngoai_bang_counts(history, ctrl).get(SEEDS[0], 0)), pymoo, eval_mu)
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
    if dom != L['domain'] or gate != L['speeds'] or os.path.abspath(a.s03) != L['s03_run'] \
            or domain_source(a.s03, ctrl) != {k: L.get(k) for k in ('nguon_mien', 'ly_do')}:
        sys.exit('the domain, its source, gate speeds or S03 run differ from lock.yaml (S04 7: locked before '
                 'the main runs)')
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
    errs = seed_error_rates(ev.evals, ctrl, seeds)
    n_ngoai = ngoai_bang_counts(history, ctrl)
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
        speeds=L['speeds'], selection=stats, seed_error_rates=errs, n_ngoai_bang_per_seed=n_ngoai,
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
    print(f'[{ctrl}] ngoai_bang candidates (outside the LADRC triangle, not run) per seed: {n_ngoai}')
    return 0 if send['feasible'] else 2


SELECTION_RULE = ('section 5: f1* = min f1; keep f1 <= f1* + 0.01 m; min f2, then f1, then gains lexicographic, '
                  'then seed; send_to_HIL on the union of the feasible non-dominated sets of the feasible seeds')


def gen_compare(gen_base):
    """Section 4.3: the generation compared with gen_base (HV_20 against HV_15). With --tool-check and
    gen_base < 6 there is no gen_base - 5: generation 1 is used (with gen_base = 1 the rule can only
    extend on HV = 0). S4C-16: written down here and in lock.yaml (gen_rule.compared_with)."""
    return max(1, gen_base - 5)


SOURCES = ('fwsim/s04_ga.py', 'fwsim/s03_doe.py', 'fwsim/o1_feasibility.py', 'fwsim/fwsim.py', 'fwsim/tune_ga.py',
           'rpi5/trajectory_pub.py', 'rpi5/imu_model.py', 'rpi5/plant_blob.py', 'tools/run_validity.py',
           'rpi5/data/imu_noise_replay_20hz.csv', 'rpi5/data/imu_noise_replay_100hz.csv',   # IMN-7: replay data
           'fwsim/o1_gate_eval.py', 'fwsim/o1_gate.py', 'tools/o1_lanes.py', 'rpi5/o1_tasks.py',   # g3-g5 (S04 v6.2)
           'tools/gain_table.py')            # S04 v6.8 section 2: the candidate check
PYMOO_FILES = ('core/individual.py', 'core/survival.py', 'algorithms/moo/nsga2.py')     # PYMOO_CV refers to these


def pymoo_file_hashes():
    """S4C-18: SHA-256 of the installed pymoo files that PYMOO_CV cites, so the file:line can be re-checked."""
    import pymoo
    root = os.path.dirname(pymoo.__file__)
    return {n: s3.sha256_file(os.path.join(root, n)) for n in PYMOO_FILES}


def fingerprint(dll, s03_dir, unsealed=False):
    """S4C-2: what must not change between the trial, the main runs and the post checks of one folder
    (S04 7): the DLL, the S03 files the domain and the mu rule come from (v6.6: kiem_ngoai_mien.csv too;
    v6.7 9.6: the four sealed files of S03_SEALED_LOCK and the root number file so_goc; sealed/runs.csv
    only with --s03-unsealed, L56; the root design.csv is not hashed, the sealed one is), the plant source
    blobs (L34), the Python sources the simulation and the scoring import (SOURCES), D, the library
    versions and the pymoo files of PYMOO_CV."""
    names = ('meta.yaml', 'levels.csv', 'sweep.csv') + S03_SEALED_LOCK + (('sealed/runs.csv',) if unsealed else ())
    s03_files = {}
    for n in names:
        p = os.path.join(s03_dir, n)
        s03_files[n] = s3.sha256_file(p) if os.path.isfile(p) else None
    kp = kiem_file(s03_dir)                     # S84-2: read_domain reads it (S04 v6.6), so it is locked too;
    s03_files[KIEM_FILE] = s3.sha256_file(kp) if kp else None     # S84-3: checked before it is hashed
    root = os.path.join(HERE, '..')
    scripts = {n: s3.sha256_file(os.path.join(root, n)) for n in SOURCES}
    return dict(dll_sha256=s3.sha256_file(dll), s03_files=s03_files, plant_src=fwsim.PLANT_SRC, scripts=scripts, D=D,
                libraries=fwsim.lib_versions(), pymoo_files=pymoo_file_hashes(), so_goc=so_goc(s03_dir))


def so_goc(s03_dir):
    """S03 v8.5 section 8 condition 9: the root number file the S03 screening recorded in its public
    meta.yaml under the SO_GOC_META_KEY line 'screen_u: {"file": "<path>", "sha256": "<hex>", ...}'.
    The meta key name and the record shape are this script's choice (the spec does not fix them; the
    screening script is written in parallel): one constant holds them. No such line -> None; a record
    that is not a dict with str 'file' and 'sha256' -> sys.exit. A relative file is relative to the repo
    root; a missing file -> sys.exit. The bytes and their CRLF->LF form are both hashed: accepted when
    rec['sha256'] equals either, or rec.get('sha256_lf') equals the LF hash; else sys.exit ("not the
    number file the S03 run recorded"). Returns dict(file=..., s03_sha256=..., sha256_lf=<LF hash>)."""
    meta = open(os.path.join(s03_dir, 'meta.yaml'), encoding='utf-8').read()
    try:
        rec = meta_value(meta, SO_GOC_META_KEY)
    except KeyError:
        return None
    if not isinstance(rec, dict) or not isinstance(rec.get('file'), str) \
            or not isinstance(rec.get('sha256'), str):
        sys.exit(f'{s03_dir}/meta.yaml: {SO_GOC_META_KEY} is {rec!r}, not a dict with str "file" and '
                 f'"sha256" (S03 v8.5 section 8 condition 9)')
    p = rec['file'] if os.path.isabs(rec['file']) else os.path.join(HERE, '..', rec['file'])
    if not os.path.isfile(p):
        sys.exit(f'{SO_GOC_META_KEY} file {p} (meta.yaml of {s03_dir}) does not exist: not the number file '
                 f'the S03 run recorded (S03 v8.5 section 8 condition 9)')
    raw = open(p, 'rb').read()
    h = hashlib.sha256(raw).hexdigest()
    h_lf = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    if rec['sha256'] not in (h, h_lf) and rec.get('sha256_lf') != h_lf:
        sys.exit(f'{p}: not the number file the S03 run recorded (meta.yaml {SO_GOC_META_KEY} sha256 '
                 f'{rec["sha256"][:12]}…, file {h[:12]}…; S03 v8.5 section 8 condition 9)')
    return dict(file=rec['file'], s03_sha256=rec['sha256'], sha256_lf=h_lf)


def check_s03_sealed(fp, tool_check):
    """S04 v6.7 section 8: the S03 files the sealed lock needs. tool_check: only sealed/levels_sangloc.csv
    and sealed/design.csv (the domain and the design are read in every mode). Else: every entry of
    S03_SEALED_LOCK and the root number file so_goc. sys.exit on a missing one (fingerprint holds None)."""
    need = ('sealed/levels_sangloc.csv', 'sealed/design.csv') if tool_check else S03_SEALED_LOCK
    miss = [n for n in need if fp['s03_files'].get(n) is None]
    if not tool_check and fp.get('so_goc') is None:
        miss.append(f'so_goc (meta.yaml {SO_GOC_META_KEY})')
    if miss:
        sys.exit(f'missing S03 sealed inputs {miss}: the S03 v8.5 screening files are needed '
                 f'(S04 v6.7 section 8)')


def check_s03_trial(common, phase):
    """S04 v6.7 section 9 item 6 (SL4-4): post() and sens_after() check the S03 run of the lock before
    fingerprint() opens anything under sealed/. Reads the PUBLIC <common.s03_run>/meta.yaml only. A tool_check
    lock: the S03 run must be --trial (trial: true; a missing trial line counts as not trial). A main lock:
    the S03 run must not be trial and common.unsealed must be true. sys.exit else."""
    meta = open(os.path.join(common['s03_run'], 'meta.yaml'), encoding='utf-8').read()
    try:
        trial = meta_value(meta, 'trial') in (True, 'true')
    except KeyError:
        trial = False                            # a missing trial line counts as not trial
    if common['tool_check']:
        if not trial:
            sys.exit(f'{phase}: the lock is a --tool-check run but {common["s03_run"]} is not an S03 --trial '
                     f'run (trial: true): nothing under sealed/ may be read for a tool check '
                     f'(S04 v6.7 section 9 item 6)')
    elif trial or not common.get('unsealed'):
        sys.exit(f'{phase}: the lock is a main run; it needs the main S03 run (not --trial) unsealed by '
                 f'--s03-unsealed (L56; S04 v6.7 section 9 item 6)')


def s03_dll_sha(s03_meta):
    """dll_sha256 of the S03 meta.yaml (indented under firmware:), or None."""
    for line in s03_meta.splitlines():
        if line.strip().startswith('dll_sha256:'):
            return line.split(':', 1)[1].split('#')[0].strip().strip('"')
    return None


def o1_constants():
    """The O1 constants as s03_doe.py writes them into the S03 meta.yaml 'o1' line."""
    return dict(A=o1.A, a_r=o1.A_R, D=o1.D, D_d=o1.D, t_d=o1.T_D, t_late=o1.T_LATE, delta=o1.DELTA, v_d=o1.V_D,
                c_cens=o1.C_CENS)


O1_FILE = 'fwsim/o1_feasibility.py'


def blob_at(commit, path):
    """`git rev-parse <commit>:<path>` in this repository, or None when git cannot read it."""
    import subprocess
    try:
        r = subprocess.run(['git', '-C', HERE, 'rev-parse', '--verify', '--quiet', f'{commit}:{path}'],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip()
    return out if r.returncode == 0 and len(out) == 40 else None


SIM_SOURCES = SOURCES + (S03_SCREEN_SCRIPT,)   # S4C-2 round 4 + S04 v6.7 9.6: this script and the screening
                                               # script are compared too; SOURCES itself stays unchanged
                                               # (fingerprint 'scripts' is unchanged)


def work_blob(rel):
    """git_blob of the WORKING-TREE file <rel> (repo-root relative), None when the file does not exist
    (fwsim/s03_sangloc.py is written in parallel)."""
    p = os.path.join(HERE, '..', rel)
    return git_blob(p) if os.path.isfile(p) else None


def source_diff(commit):
    """The SIM_SOURCES whose working-tree blob is not the blob at <commit>: [(path, now, then)]; a path is
    reported when then is None or now != then (a script missing at the S03 commit is reported)."""
    out = []
    for n in SIM_SOURCES:
        now, then = work_blob(n), blob_at(commit, n)
        if then is None or now != then:
            out.append((n, now, then))
    return out


def check_s03_source(s03_meta, fp, dll, tool_check=False):
    """S4C-2 (Codex rounds 3, 4): S04 runs on the source the S03 run recorded, checked against the
    S03 meta.yaml at every trial and main start, so the first trial cannot lock a source S03 did not
    use (later phases compare with lock.yaml). By class, not file by file: the DLL
    (firmware.dll_sha256), the plant blobs (plant.src, L34), the run configuration S03 wrote (o1, imu)
    and every module S04 imports for the simulation and the scoring, this script and the screening
    script fwsim/s03_sangloc.py included (S04 v6.7 section 9 item 6;
    SIM_SOURCES: the blob in this checkout against the blob at the S03 script commit, which must be
    clean). A main run stops on any difference, so the S03 main run is run from the commit S04 runs
    on (L34: one merged commit). --tool-check stops on the DLL, plant, configuration and commit, and
    only warns about SIM_SOURCES; the differences are returned and written to lock.yaml and meta.yaml
    (s03_source_diff). Not compared: the Python and library versions (S03 does not write them; S04
    fixes them for its own folder in the fingerprint)."""
    if s03_dll_sha(s03_meta) != fp['dll_sha256']:
        sys.exit(f'{dll} (sha256 {fp["dll_sha256"][:12]}…) is not the DLL of the S03 run '
                 f'({str(s03_dll_sha(s03_meta))[:12]}…): S04 runs on the S03 firmware (S04 7)')
    try:
        plant, o1m, imu, script = (meta_value(s03_meta, k) for k in ('plant', 'o1', 'imu', 'script'))
    except KeyError as e:
        sys.exit(f'the S03 meta.yaml has no {e} line: cannot check that S04 runs on the S03 source (S4C-2)')
    src = plant.get('src') if isinstance(plant, dict) else None
    if not isinstance(src, dict) or not same_plant(src, fp['plant_src']):
        sys.exit(f'plant source {fp["plant_src"]} is not the S03 plant.src {src} (L34; S04 7; S4C-2)')
    if o1m != o1_constants():
        sys.exit(f'O1 constants {o1_constants()} are not the S03 o1 {o1m} (S04 7; S4C-2)')
    if imu != s3.IMU:
        sys.exit(f'IMU configuration {s3.IMU} is not the S03 imu {imu} (S04 7; S4C-2)')
    if not isinstance(script, dict) or script.get('dirty', True) or not script.get('commit'):
        sys.exit(f'the S03 run has no clean script commit ({script}): its sources cannot be checked (S4C-2)')
    if blob_at(script['commit'], O1_FILE) is None:
        sys.exit(f'the S03 script commit {script["commit"]} cannot be read in this repository (no '
                 f'{O1_FILE} there): its sources cannot be checked (S4C-2)')
    diff = source_diff(script['commit'])
    if diff:
        txt = '; '.join(f'{n} (blob {now}, at {script["commit"]}: {then})' for n, now, then in diff)
        if not tool_check:
            sys.exit(f'not the sources of the S03 script commit: {txt}. Run S03 from the commit S04 runs on '
                     f'(S04 7; L34; S4C-2)')
        print(f'WARNING (--tool-check): not the sources of the S03 script commit: {txt}; written to '
              f'lock.yaml / meta.yaml s03_source_diff')
    return [dict(path=n, blob=now, s03_blob=then) for n, now, then in diff]


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
    pp = os.path.join(out, 'post_checks', 'notes.json')
    pre_hil = ({k: v for k, v in (json.load(open(pp, encoding='utf-8')).get('notes') or {}).items()
                if k.endswith(' pre_hil')} if os.path.isfile(pp) else None)
    fw_info = json.load(open(os.path.join(os.path.dirname(dll), 'fwsim_build.json'), encoding='utf-8')) \
        if os.path.isfile(os.path.join(os.path.dirname(dll), 'fwsim_build.json')) else None
    import pymoo
    common = lock.get('common', {})
    ctrls = {}
    for c, L in lock.get('controllers', {}).items():
        ctrls[c] = {k: L.get(k) for k in ('tool_check', 'domain', 'nguon_mien', 'ly_do', 'speeds', 'eval_mu',
                                          'mu_decision', 'mu_rule_report', 'pop', 'n_gen', 'gen_rule', 'seeds',
                                          'hv_ref', 'preconditions', 'stop')}
        ctrls[c]['plant_by_mu'] = {repr(m): json.loads(fwsim.plant_meta(fwsim.plant_params(
            argparse.Namespace(ground='traction', mu=m)))) for m in (L.get('eval_mu') or [])}
    lines = dict(spec=SPEC, date=time.strftime('%Y-%m-%d'),
                 setup='sim: fwsim, ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware',
                 operator='Claude (phiên CODE)', tool_check=common.get('tool_check'),
                 not_main_numbers=bool(common.get('tool_check')), running=running,
                 script=s3.git_state(os.path.join(HERE, '..')), fwsim_build=fw_info,
                 s03_run=common.get('s03_run'), s03_mu_levels=common.get('s03_mu_levels'),
                 fingerprint=common.get('fingerprint'), s03_source_diff=common.get('s03_source_diff'),
                 pymoo=dict(version=pymoo.__version__, checked_version=PYMOO_CHECKED, constraint_handling=PYMOO_CV,
                            recheck=pymoo.__version__ != PYMOO_CHECKED),
                 selection_rule=SELECTION_RULE, D=D, imu=s3.IMU, plant_seed=s3.SEED, controllers=ctrls,
                 phases=phases, elapsed_s=sum(int(x.get('elapsed_s', 0)) for x in phases),
                 code_notes=code_notes, pre_hil=pre_hil, pymoo_files=pymoo_file_hashes(),
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
        f.write(f"""spec: {SPEC}
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
constraints: "g1 = runs not completed, inconclusive or with a track error (all runs when a run_error); g2 = (f1 - D) / D, D = {D} m; g3 = (max e_trk_max - D) / D; g4 = (max e_bao_max - D) / D; g5 = (max e_theta_cmd_max_deg - {THETA_MAX_DEG}) / {THETA_MAX_DEG} (S04 v6.2); a run_error run, or a quantity missing or not finite, takes {PENALTY:g} with pen_<quantity> = 1 in evals.csv (S04 v6.3 RS2-8); a candidate outside the LADRC triangle (tools/gain_table.py) is not run: the same penalty as a run_error candidate, status ngoai_bang (S04 v6.8 section 2); pymoo G <= 0"
pymoo: {json.dumps(dict(version=pymoo.__version__, checked_version=PYMOO_CHECKED, constraint_handling=PYMOO_CV,
                       recheck=pymoo.__version__ != PYMOO_CHECKED))}
libraries: {json.dumps(fwsim.lib_versions())}
interpretations:   # of this script, not written in S04 v5; Liam or THESIS correct if wrong
  - "a new script fwsim/s04_ga.py (S04 9.1 names tune_ga.py): tune_ga.py keeps its velocity objective for the old runs, unchanged"
  - "S04 v6.4 (C14-2; Liam 15/11 CODE_FEAS 'Code ngay + Codex'): selected.csv keeps edge (the 5 % flag, formula unchanged) and adds edge_side (factor:L or :H), edge_src (factor:side:source of that bound from the S03 levels.csv L_nguon / H_nguon, ? when the S03 run has none) and edge_limit (the flagged factors whose bound is a grid end, 'het luoi noi' or 'het luoi duoi': the limit 'vung ngoai bien chua khao sat'); no flag, no limit; RSM candidates (section 12) are not in this script yet"
  - "S04 v6.6 (Liam 17/11 THESIS_SLIDE; script: Liam 17/11 CODE_FEAS 'Code + Codex 2 vong'): the GA domain is [L, H] of the original S03 grid (S03 v8.4: the extension only for the out-of-range check); levels of S03 v8.3 (H_nguon 'het luoi noi' or H above b) are refused; edge_limit names the grid ends 'het luoi', 'het luoi duoi' (limit 'vung ngoai bien chua khao sat bang GA') with the S03 kiem_ngoai_mien.csv result on that side (? without the file)"
  - "S04 v6.8 section 2 (C3-7): a flagged bound whose source is 'tran bang chan' (H = T, the range-table ceiling) adds to edge_limit the limit 'vung ngoai bang chan khong thuoc thi nghiem (DUT tu choi gain do)' with factor:side (written with its accents in the file); the two kinds of limit are joined by ' | '; no flag, no limit"
  - "mu rule: every S03 design point that ran (centre, outer, inner box: the inner box always runs since L41), every gate speed; status other than ok (track_err, run_error) or y1 inconclusive at any mu is 'not satisfied'; the G3, G4, G5 states must be the same at the three mu, read from the gate state columns g3_state, g4_state, g5_state of the S03 sealed/runs.csv (a row without them stops the rule; S04 v6.2, v6.7 9.7); a state that is the same at the three mu satisfies the rule whatever it is (S04 3 asks for the same state only); S04 v6.8: |dY3| <= 0.4 1/s (absolute, S03 5.2), the y3_reference of the S03 meta.yaml is not read; C3-6: both bounds are compared as S03 compares (s03_doe.not_above_by: inclusive, exact on the decimal of each number as written in runs.csv, no tolerance of its own); a Y2 or Y3 that is not finite does not satisfy the rule"
  - "HV: exact 2-D staircase on the feasible non-dominated points of the population after each generation's survival, strictly inside the reference (S04 v6.8 4.2: (D; 800 1/s), fixed; a point with f1 >= D or f2 >= 800 adds 0 and is counted in hv.csv n_nd_on_or_beyond_ref); the trial rule compares generation gen_base with gen_base - 5 (15 and 20); with --tool-check and gen_base < 6 with generation 1"
  - "pareto.csv: per seed's final population, the non-dominated points among the feasible points and, separately, among the infeasible points (column feasible; nondominated_feasible true only for the first group, nondominated_in_group true for both); always written with its header, also when empty; final_population.csv: every point of it with both flags (S04 8)"
  - "S04 v6.8 4.2: the HV reference no longer comes from the default gains; the trial does not run them (no evals.csv row with ga_seed 0, no default_f2 in lock.yaml)"
  - "per seed set (section 5): the feasible non-dominated points of the final population (pymoo algorithm.pop after the last generation)"
  - "run_error: attempt 2 in a new process is recorded with 'cause' empty; the candidate is scored as the first attempt (S04 3); CODE fills 'cause' when it finds a tool error"
  - "R sensitivity: motor_R of the four wheels x 0.8, 0.9, 1.0, 1.1, 1.2 (S04 6 says +-20 %), every other plant parameter as params.py (J, b not rescaled)"
  - "gains stored as Python repr (17 significant digits) and as the float32 value the firmware receives (fwsim set_pid / set_adrc take c_float)"
  - "g3-g5 (S04 v6.2, R134 package): e_bao_max of o1_gate_eval.evaluate_fig8 (evaluate_v7 unchanged inside); the diagnosis attempt 2 of a run_error runs s03_doe's evaluator in a new process (diagnosis only; pen_e_bao_max = 1 when that s03_doe writes no e_bao_max); a candidate with a run_error takes 1e6 for every quantity as for f1"
  - "S04 v6.7 9.7: the post checks read G1-G5 from the gate state columns (o1_gate.gate called as s03_doe.run_one calls it: U = 0, every condition deciding): run_custom scores with o1_gate_eval.evaluate_fig8 and the gate; sensitive when a G3, G4 or G5 state changes; three_mu.csv and sensitivity.csv carry g1_state .. g5_state; a three-mu run passes only with status ok and the five states dat"
  - "S04 v6.8 2: every candidate is checked by tools/gain_table.py (gain_check) before it runs (float32 values): a gain outside the range table stops the run (tool error); an ADRC candidate outside the triangle is not run: penalty of a run_error candidate, status ngoai_bang in history.csv, no evals.csv row (so in neither part of the seed error rate), n_runs 0, pen_f1 1; the number per seed is n_ngoai_bang_per_seed of meta_<controller>.yaml; what counts as a table or a triangle violation is decided by tools/gain_table.py alone: its classify() Verdict (C3-11; C3-25: required, the check() text fallback is removed - a module without callable classify() is a tool error and nothing is simulated); C3-10: a seed whose candidates were all ngoai_bang has no evals.csv row and is reported in seed_error_rates with n 0 and rate null"
  - "S04 v6.7 3: a run that is not a run_error but has tv_duty missing or not finite is recorded with status run_error and the reason in error (its other fields kept), then treated as any run_error (diagnosis rerun, candidate penalty, seed error rate)"
  - "S04 v6.8 6: before HIL (post) the selected gains are checked against the range table and the triangle (post_checks/notes.json '<controller> pre_hil', copied to meta.yaml pre_hil; a failure is reported with exit code 2, no reselection); the same entry holds the spectral radius of the linear sub-model of one translation axis (tools/mo_hinh_con.ban_kinh_pho_va_trang_thai, with the kept and dropped states and the model parameters and file hash): report only, NOT a GA constraint, but rho >= 1 (or not computable although the set is inside the table) is a cho_liam item 'spectral_radius'; C3-8: a run of the three-mu check that does not pass G1-G5 (three mu levels, check done) is a cho_liam item 'three_mu'; a sensitive b0_factor / r_factor result of judge() is a cho_liam item under its own check name; cho_liam not empty -> post returns 2 (no reselection)"
  - "S04 v6.8 6 / v6.1 6 (RB-1): --phase sens_after (sensitivity check after selection, report only) runs the loop-delay variant (io_delay_ms; --io-delay-ms is the measured bench delay: not whole -> rounding down and up, within 1e-9 ms of a whole number -> that number; not given -> 1 and 2 ms) and the four plant variants of PLANT_VARIANTS (apply_plant_variant, ONE update_physical call per variant, the changed keys checked against the declared ones); one baseline run (io_delay_ms 0, no variant) per (controller, mu, v) serves both delay.csv and plant.csv; plant_params.json holds the before/after dumps and slip_nsub; output in sens_after/ only (phases.json, meta.yaml, post_checks/ not written); the GA, the trial and the post checks run with io_delay_ms = 0 and slip_omega_s = 0.1 (run_gate stops on another slip_omega_s)"
  - "S04 v6.7 9.6: the GA domain and the S03 design come from the SEALED files of the S03 v8.5 screening (sealed/levels_sangloc.csv with nguon_mien, ly_do; sealed/design.csv), never the public levels.csv / design.csv; the v6.6 original-grid refusal applies only to nguon_mien 'du_phong' ('sang_loc' may reach b_n); inside it, the C3-7 / C3-26 exception accepts H = T = b_n when the sweep.csv rows at H are all tran 'true' (S03 v8.6 3.1 rule (2)); lock.yaml holds nguon_mien / ly_do and the main phase stops when they differ; --tool-check still takes an S03 --trial run but now reads the two sealed files it needs (checked before anything under sealed/ is opened)"
  - "S04 v6.7 8 / S03 v8.5 8 (9): the fingerprint locks meta.yaml, levels.csv, sweep.csv, the four files of S03_SEALED_LOCK (None when missing), sealed/runs.csv when unsealed and kiem_ngoai_mien.csv, plus so_goc: the root number file the S03 meta.yaml records under 'screen_u' (a dict file / sha256 / optional sha256_lf; key name and record shape are this script's choice, the screening script is written in parallel); its bytes or their CRLF->LF form must hash to the recorded sha256; check_s03_sealed needs all five sealed items for a main run, two for --tool-check; check_s03_source also compares the blobs of this script and of fwsim/s03_sangloc.py (a script missing at the S03 commit is reported)"
  - "S03 v8.6 11.6: the PID integral clamp indicator (fwsim.kep_indicator) is report only: evals.csv adds kep_frac_vx, kep_frac_vy, kep_frac_wz, kep_note (ADRC rows '' / ''); a kep_indicator ValueError only fills kep_note (status and score unchanged); a PID run on a DLL without the kep channel (fwsim_kep_cols = 0) stops in main() before the GA; log_sha hashes only fwsim.BIT_EXACT_KEYS (digest_sha), so a log with and without the 'kep' array gives the same value"
""")


if __name__ == '__main__':
    sys.exit(main())
