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
       the thresholds of 5.2 (v8.6: Y3 is the absolute 0.4 s^-1 at every speed and mu, no
       longer a share of the default-gain run); y2_versions.csv: whether the two Y2
       versions give the same significant set and signs (5.1); curvature.csv (5.4);
       fail_counts.csv (5.3).
Outputs (section 7) in --out: sweep.csv, sweep_counts.csv, levels.csv, kiem_ngoai_mien.csv,
logs_sweep/, meta.yaml; in --out/sealed/ (L56): levels_sangloc.csv, design.csv (centre, outer,
inner box, axial, test points), sangloc/, sangloc_thu/ (the screening step of section 3.1b),
runs.csv, effects.csv, fail_counts.csv, curvature.csv, y2_versions.csv,
determinism_check.csv, notes.json, stop.json, diag.txt, manifest.json (SHA-256 of the sealed
files), logs/.

S03 v8 (L67, Liam 07/11; spec section 11 item 4): 14 face-centred axial points (point_id
257-270, box 'axial', 3 mu x 3 speeds) run with the main design and go to sealed/runs.csv;
100 Latin-hypercube test points (point_id 271-370, box 'test', middle mu x 3 speeds) read
from the committed file fwsim/s03_test_points.csv (made once with --make-test-points, never
generated again) run AFTER sealed/ is closed, for both controllers, and everything about
them is in --out/sealed_test/ (runs_test.csv, logs/, notes.json, stop.json, manifest.json).
S03 does not analyse either: section 5 and the rule of 3.2 use centre, outer and inner only.
design.csv: 371 rows per controller, checked point_id - box - codes and bounds (11.4 (i),
(ii)) before the main design. Two diagnostic columns n_upper_frames, n_back_frames (section 4).

S03 v8.2 (Liam 13/11, THESIS local_7999bf79; section 4 row G, 7, 8 condition 8; script: Liam 14/11
CODE_FEAS local_4e9b745d "Code ngay", design 03_CODE/docs/2026-11-14_Thiet_ke_S03_G1G5_v1.md): every
run is scored with the gate evaluator of S04 and C12, o1_gate_eval.evaluate_fig8 (it calls evaluate_v7
UNCHANGED, so every older column keeps its value bit for bit) and o1_gate.gate(u_t = 0, DECIDE_ALL;
U = 0 on fwsim: the true plant pose). GATE_RUN_COLS are appended after 'error' in runs.csv,
runs_test.csv and the sweep (Liam "Co, mot cach cham"): e_bao_max (G4), gate_version, gate_decide,
gate_u_t, g1_q, g1_state, g2..g5 q, lo, hi, n_undef, state (names of tools/rq3_log.py GATE_COLS, B81-3)
and gate_outcome (O1 3.5 over G1-G5). G3-G5 belong to the gate and S04, not to the section 5 analysis.

S03 v8.3 (Liam 15/11, THESIS_SLIDE local_44e97934, C14-2; script: Liam 15/11 CODE_FEAS "Code ngay + Codex",
design 03_CODE/docs/2026-11-15_Thiet_ke_S03_v83_luoi_noi_v1.md): the sweep grid is extended upward,
sweep_grid_n = sweep_grid (unchanged; o1_feas_search keeps its box [a, b] with it) + b r^k, k = 1..K,
r = (b/a)^(1/10), K the smallest with r^K >= 10 (1 - 1e-9); omega_o is not extended. levels.csv adds a, b,
b_n, L_nguon, H_nguon (source of each level); sweep.csv adds luoi (log | mac_dinh | 0 | noi); meta.yaml
records the grid per gain.

S03 v8.4 (Liam 17/11, THESIS_SLIDE local_44e97934 "sua tat ca cho nao ban thay di"; script: Liam 17/11
CODE_FEAS local_1557d33f "Code + Codex 2 vong", design 03_CODE/docs/2026-11-17_Thiet_ke_S03_v84_kiem_ngoai_mien_v1.md):
the extension still runs with the sweep but only for a check. The level rule runs on the ORIGINAL grid (11 log
points, the default, 0); H_nguon 'hết lưới' (H = b), 'trần ω_o' first for omega_o at 48. The out-of-range check
(out_of_range_check) puts every original or extension point outside [L, H] against the run at the nearer level,
middle mu, gate speeds, in the order of section 3.1 (không cờ / chưa kết luận / cờ / không cờ), writes
kiem_ngoai_mien.csv (one row per point, one total row per controller: đã kiểm đủ / kiểm dở / không kiểm) and
stops S03 before the main design when a point is 'cờ' (Liam decides).

S03 v8.5 K6 and v8.6 (Liam 21-22/11):
  * K6: a flagged or inconclusive out-of-range point no longer stops S03; it is written in
    kiem_ngoai_mien.csv, printed and kept in meta.yaml kiem_ngoai_mien.
  * v8.6 section 3.1: the sweep grid is sweep_grid_n cut at the ceiling T of the gain in
    tools/gain_table.py (sweep_grid_cut): points above T are dropped; when a point was dropped,
    a remaining point within the relative error 1e-9 of T becomes T, else T is added. b_n is the
    largest point after the cut (T when cut; else the extension end; for omega_o, which has no
    extension, the end b of its grid: Codex C3-5); sweep.csv adds tran (true at value == T);
    H_nguon 'trần bảng chặn' when H = T. A default the cut made T (within 1e-9 below T, the same
    float32) is run at the grid point T, which stands for the default run (default_node, C3-4).
  * v8.5/v8.6 section 3.1b: the random multi-gain screening step (fwsim/s03_sangloc.py
    run_step) runs after the sweep of every controller, inside the sealed window: S0 (item
    10, sealed/sangloc_thu/), S1, S2, S2b with their lock files (sealed/sangloc/), its
    determinism check (item 12), then sealed/levels_sangloc.csv (the accepted box or the
    sweep levels as fallback) and sealed/design.csv, then the main design of every
    controller. The numbers are the committed file fwsim/s03_screen_u.csv
    (make_s03_screen_u.py, run once); every check of it runs before the sweep.
  * v8.6 section 3.2: a design point outside the LADRC triangle omega_c + 2 omega_o < 200 on an
    axis runs like any point and carries the column ngoai_tam_giac in sealed/design.csv and in
    sealed/runs.csv, sealed_test/runs_test.csv; before the main design the float32 of every
    gain of every point is also compared with the gain table (a failure stops S03).
  * v8.6 section 5.2: the Y3 significance threshold is the absolute 0.4 s^-1 (four wheels, one
    PWM step per 10 ms tick), exact as a decimal; y3_reference is only reported.

L56 / S03 v6 section 1: the main design results (runs.csv, effects.csv and the tables of
section 5) are NOT looked at before campaign A ends. Nothing derived from them leaves
sealed/ (Codex S3C-1..S3C-3): no hash of a sealed file outside it, no child-process job file
outside it, a stop after the sweep prints only that S03 stopped (the reason and its phase
are in sealed/stop.json, prints and warnings in sealed/diag.txt), the determinism check
is only 'passed' there, sealed/design.csv holds every point of every controller before the
main design starts, and the inner box always runs (L41). Every spec stop ends S03
for both controllers (S3C-9). The sweep (section 3.1) is a source of campaign A's domain and
is printed.

Codex round 3: the screening numbers file is regenerated in memory and checked bit for bit
against its generation source at the start (C3-36); a run with the screening step needs both
controllers once and three finite, distinct mu levels (C3-37); the closing of sealed/ is
guarded so an output error becomes a sealed stop, never a 'running' meta.yaml (C3-38); the
sealed phase notes exist from each phase start and the elapsed fields are split
(elapsed_s_sang_loc, elapsed_s_main_design; C3-39, D2A-2); a PID run on a DLL without the kep
channel, or a failed clamp-indicator self-check, stops S03 (C3-48).

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
if 'FirmwareSim' not in vars(fwsim):                    # 'python -m fwsim.s03_doe': fwsim is the
    import importlib                                  #  namespace package; the module is fwsim.fwsim
    fwsim = importlib.import_module('fwsim.fwsim')    # noqa: E402
    sys.modules['fwsim'] = fwsim                       # every later `import fwsim` gets the module
import o1_feasibility as o1                             # noqa: E402
import o1_gate as G                                     # noqa: E402  (S03 v8.2: G1-G5)
import o1_gate_eval as E                                # noqa: E402  (G4 e_bao; evaluate_v7 inside, unchanged)
import tune_ga                                          # noqa: E402
_TOOLS = os.path.join(HERE, '..', 'tools')              # as o1_gate_eval puts it on sys.path
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import gain_table                                       # noqa: E402  (tools/: the PC gain table, S03 v8.6 2.1)

CONTROLLERS = ('PID', 'ADRC')
SPEEDS = (0.2, 0.5, 0.8)
FACTORS = {m: list(tune_ga.VARS_BY_MODE[m]) for m in CONTROLLERS}       # X1..X7 (S03 section 2.1)
K_PSI_DEFAULT = 1.0           # heading.h HEADING_PARAMS_DEFAULT (no getter in fw_sim.c); AGV b0047ac line 25
OMEGA_O_MAX = 48.0            # S03 section 2.1: convention from the adrc.c comment, not a stability bound
N_GRID = 11
SEED = 1                                      # plant seed (no random plant input) and IMU noise seed; recorded
# BNO085 20 Hz (C4, L73) with the replay of the measured noise, seed fixed = SEED (S03 v8
# section 3.3: "when the IMU noise model is turned on, S03 is run again with a fixed seed
# recorded in meta.yaml"; IMN-7, Liam 10/11 "Lam IMN-7 + S03"). The noise-free configuration of
# S03/2026-11-09_run01 is dict(rate_hz=20.0, boot_reset=-1.0): re-run it from script 6f93607.
IMU = dict(rate_hz=20.0, boot_reset=-1.0, noise='replay', noise_seed=SEED)
DET_SEED = 20261106                           # S03 section 3.4: seed that draws the 3 corners, fixed before running
_mono = time.monotonic   # monotonic clock of the phase seconds of sealed/notes.json, replaceable by the tests
Y3_ABS = 0.4            # s^-1, S03 v8.6 section 5.2: four wheels one PWM step (1/1000) per 10 ms tick,
                        # 4 x 0.001 x 100; every speed, every mu, worst, range
THRESH = {'e_perp_max': 0.01, 'e_perp_max_cens': 0.01, 'sat_frac': 0.01, 'tv_duty': Y3_ABS}   # S03 5.2
RESPONSES = ('e_perp_max', 'e_perp_max_cens', 'tv_duty', 'sat_frac')
OLD_COLS = ['controller', 'point_id', 'box', 'mu', 'v', 'status', 'completed', 'e_perp_max', 'e_perp_max_cens',
            'e_perp_global_max', 'tv_duty', 'sat_frac', 'sat_any_frac', 'e_theta_lane_max_deg',
            'e_theta_cmd_max_deg', 'ds_absmax', 'e_trk_max', 't_end', 'e_s_absmax', 'track_err_frames',
            'cond2_5_ever', 'pos_reset_mid', 'e_perp_rms', 'e_trk_rms',
            'n_upper_frames', 'n_back_frames',      # S03 v8 section 4 (W8-Q (b)): diagnostics, not responses
            'error']                                # RMS: S3C-13; error: S3C-12
# S03 v8.2 section 7: e_bao_max and the G1-G5 states (o1_gate.py, U = 0), appended after the S03 v8 columns.
# Names as tools/rq3_log.py GATE_COLS (B81-3); not written: gate_json_sha256 (no gate file per run). The _chung
# columns and the G1 fields are G1_RUN_COLS below (15/11; deferred before, Liam 14/11 CODE_R134 "De goi rieng
# sau"). gate_outcome: O1 3.5 over G1-G5.
GATE_RUN_COLS = (['e_bao_max', 'gate_version', 'gate_decide', 'gate_u_t', 'g1_q', 'g1_state']
                 + [f'{c}_{x}' for c in G.MEASURED for x in ('q', 'lo', 'hi', 'n_undef', 'state')]
                 + ['gate_outcome'])
# B81-3 G1 fields and the common domain (design 03_CODE/docs/2026-11-15_Thiet_ke_bo_cham_fwsim_G1_v1.md; Liam 15/11
# CODE_FEAS "code di"): names of tools/rq3_log.py GATE_COLS; one way of using the frames on fwsim, so g1_ways = 1,
# g1_t_done_2 and g1_cat_t1 (floor fields) empty and the _chung columns equal the union ones
G1_RUN_COLS = ([f'{c}_{x}_chung' for c in G.MEASURED for x in ('q', 'lo', 'hi', 'n_undef')]
               + ['g1_ways', 'g1_t_deadline', 'g1_t_done_1', 'g1_t_done_2', 'g1_t1', 'g1_prog_before_t1',
                  'g1_cond25_before_deadline', 'g1_cat_t1'])
RUN_COLS = OLD_COLS + GATE_RUN_COLS + G1_RUN_COLS
TRIANGLE_COL = 'ngoai_tam_giac'          # S03 v8.6 section 3.2: design point outside the LADRC triangle
KEP_FRAC_COLS = ['kep_frac_vx', 'kep_frac_vy', 'kep_frac_wz']   # S03 v8.6 2.1, 7, 11 item 6 (c)
RUNS_FILE_COLS = RUN_COLS + [TRIANGLE_COL] + KEP_FRAC_COLS   # sealed/runs.csv, sealed_test/runs_test.csv only
LEVEL_COLS = ['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note',     # S03 v8
              'a', 'b', 'b_n', 'L_nguon', 'H_nguon']                                          # S03 v8.3 (S83-1: fixed)
SWEEP_ID_COLS = ('controller', 'factor', 'value', 'is_default', 'luoi', 'tran', 'v', 'mu')   # sweep.csv identifiers (section 7; luoi: v8.3, tran: v8.6)
GATE_U_T = 0.0                # s: fwsim knows the command start exactly (o1_gate docstring; as C12, S04)
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
    """A stop the spec orders (3.1 (i), (ii); 3.4; B8b-20): report, do not go on. case = '(i)' or '(ii)' for the
    two cases of section 3.1 (S03 v8.4: the out-of-range check is then 'không kiểm'), None for the others."""

    def __init__(self, msg, case=None):
        super().__init__(msg)
        self.case = case


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
    # S03 v8.2: the gate evaluator; evaluate_v7 runs unchanged inside, so res holds its keys as before
    res, _, frames = E.evaluate_fig8(log, path, sch, lsum=p.lx + p.ly)
    val = fwsim.run_validity(log)
    if val['run_bad'] == 1:
        raise RuntimeError(f"pose reset mid-run in fwsim ({val['run_bad_reason']})")
    v_cmd_max, wz_cmd_max = cmd_max(sch)
    gt = G.gate(frames, res['completed'], v_cmd_max=v_cmd_max, wz_cmd_max=wz_cmd_max, u_t=GATE_U_T,
                decide=G.DECIDE_ALL, status=res['status'], g1_fields=E.g1_fields_v7(res))
    row = {k: res[k] for k in OLD_COLS[5:] if k in res}          # responses only (no controller..v keys)
    row.update(pos_reset_mid=val['pos_reset_mid'])
    row.update(gate_fields(res, gt))
    return row, log


_CMD_MAX = {}


def cmd_max(sch):
    """o1_gate.fig8_cmd_max of the schedule, kept per (v, w0) as C12 does (the same numbers)."""
    key = (sch.v, sch.w0)
    if key not in _CMD_MAX:
        _CMD_MAX[key] = G.fig8_cmd_max(sch)
    return _CMD_MAX[key]


def gate_fields(res, gt):
    """GATE_RUN_COLS of one run from the evaluate_fig8 summary and the o1_gate.gate dict (states are the
    gate's, never recomputed here)."""
    out = dict(e_bao_max=res['e_bao_max'], gate_version=gt['version'], gate_decide=','.join(gt['decide']),
               gate_u_t=gt['u_t'], g1_q=gt['g1']['q'], g1_state=gt['g1']['state'], gate_outcome=gt['outcome'])
    for c in G.MEASURED:
        for x in ('q', 'lo', 'hi', 'n_undef', 'state'):
            out[f'{c}_{x}'] = gt[c][x]
        for x in ('q', 'lo', 'hi', 'n_undef'):
            out[f'{c}_{x}_chung'] = gt[c]['chung'][x]           # B81-3 common domain (one domain)
    g1 = gt['g1']
    out.update(g1_ways=len(g1['t_done']), g1_t_deadline=g1['t_deadline'], g1_t_done_1=g1['t_done'][0],
               g1_t_done_2=None, g1_t1=g1['t1'], g1_prog_before_t1=g1['prog_before_t1'],
               g1_cond25_before_deadline=g1['cond25_before_deadline'], g1_cat_t1=None)
    if set(out) != set(GATE_RUN_COLS) | set(G1_RUN_COLS):
        raise RuntimeError(f'gate fields {sorted(set(out) ^ (set(GATE_RUN_COLS) | set(G1_RUN_COLS)))} do not match')
    return out


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


GRID_N_TOL = 1e-9             # S03 v8.3 section 3.1: K smallest with r^K >= 10 (1 - 1e-9)
TRAN_RTOL = 1e-9              # S03 v8.6 section 3.1: a remaining point this near the ceiling T becomes T


def sweep_grid_n(name, default, lo, hi):
    """S03 v8.3 section 3.1 (C14-2): the grid of sweep_grid (unchanged, bit for bit; C12 keeps using it for
    its box [a, b]) plus the points b r^k, k = 1..K, r = (b/a)^(1/10) the step of the 11 log points, K the
    smallest integer with r^K >= 10 (1 - 1e-9); omega_o keeps its cap 48 and is not extended. Returns
    (grid, info) with info = a, b, r, K, b_n (None without extension), n_log, n_noi, n_points, noi (the
    extension points)."""
    base = sweep_grid(name, default, lo, hi)
    a = default / 10.0 if lo == 0.0 else min(default / 10.0, lo)        # the same expressions as sweep_grid
    b = max(default * 10.0, hi)
    if name.startswith('omega_o'):
        b = min(b, OMEGA_O_MAX)
    r = (b / a) ** (1.0 / (N_GRID - 1))
    noi = []
    if not name.startswith('omega_o'):
        k = 1
        while r ** k < 10.0 * (1.0 - GRID_N_TOL):
            k += 1
        noi = [float(b * r ** j) for j in range(1, k + 1)]
    if noi and not noi[0] > base[-1]:
        raise S03Stop(f'{name}: the extension starts at {noi[0]!r}, not above the grid end {base[-1]!r}')
    info = dict(a=float(a), b=float(b), r=float(r), K=len(noi), b_n=noi[-1] if noi else None, n_log=N_GRID,
                n_noi=len(noi), n_points=len(base) + len(noi), noi=noi)
    return base + noi, info


def ceiling(ctrl, name):
    """T of S03 v8.6 section 3.1: the gain-table upper bound of the gain (tools/gain_table.py);
    S03Stop when the factor is not a gain of the table."""
    try:
        return float(gain_table.upper(ctrl, name))
    except ValueError as exc:
        raise S03Stop(f'{ctrl} {name}: no ceiling in the gain table ({exc})') from None


def cut_at_ceiling(points, T):
    """Rules (1), (2) of S03 v8.6 section 3.1 on an ascending list of points: drop every point
    above T; when a point was dropped, a remaining point within the relative error TRAN_RTOL of T
    becomes T (two such points is an error), else T is appended. Returns (points, n_bo, them):
    n_bo points dropped; them True when T was appended as a new point. Nothing dropped: the
    points as given (same floats), 0, False."""
    keep = [x for x in points if not x > T]
    n_bo = len(points) - len(keep)
    if n_bo == 0:
        return keep, 0, False
    near = [i for i, x in enumerate(keep) if math.isclose(x, T, rel_tol=TRAN_RTOL)]
    if len(near) > 1:
        raise S03Stop(f'{len(near)} grid points within {TRAN_RTOL} of the ceiling {T!r} (S03 v8.6 3.1)')
    if near:
        keep[near[0]] = T
        return keep, n_bo, False
    keep.append(T)
    return keep, n_bo, True


def cut_grid(grid, info, T):
    """The (grid, info) of sweep_grid_n cut at the ceiling T (S03 v8.6 section 3.1). a, b, r, K,
    n_log keep their definitions (b may no longer be a grid point when the cut is inside the
    original grid); noi, n_noi, n_points describe the grid after the cut; n_bo is the number of
    points dropped (0 = not cut: the grid and every old info key exactly as sweep_grid_n gave
    them, but b_n of omega_o); b_n is the largest point after the cut (= T when cut; else the
    extension end; omega_o has no extension, so its b_n is the end of its grid, b: Codex C3-5)."""
    base = base_grid(grid, info)
    pts, n_bo, them = cut_at_ceiling(grid, T)
    if n_bo == 0:
        return grid, dict(info, T=float(T), n_bo=0, b_n=grid[-1] if info['b_n'] is None else info['b_n'])
    n_base = sum(1 for x in base if not x > T)          # a point made equal to T keeps its part of the grid
    if them and T < info['b']:
        n_base += 1                                     # the added T belongs to the original grid
    elif them and not T > info['b']:
        raise S03Stop(f'ceiling {T!r} equals b but b is not a grid point (S03 v8.6 3.1)')
    noi = pts[n_base:]
    return pts, dict(info, T=float(T), n_bo=n_bo, noi=noi, n_noi=len(noi), n_points=len(pts), b_n=pts[-1])


def default_node(grid, info, default):
    """The point of the cut grid that stands for the firmware default in the level rule, is_default
    and luoi (S03 v8.6 section 3.1; Codex C3-4): the default itself when it is a grid point. When
    rule (2) of the cut made the default, within TRAN_RTOL below the ceiling T, equal to T, the
    point T stands for it if both are the same float32 (the firmware receives one value: the run
    at T is the default-gain run). S03Stop otherwise: the default is above T (dropped), or T is
    another float32."""
    if default in grid:
        return default
    T = info.get('T')
    if info.get('n_bo') and not default > T and math.isclose(default, T, rel_tol=TRAN_RTOL):
        if float(np.float32(default)) == float(np.float32(T)):
            return T
        raise S03Stop(f'the default {default!r} became the ceiling {T!r} in the cut (within {TRAN_RTOL}) but is '
                      f'another float32, {float(np.float32(default))!r} not {float(np.float32(T))!r} (S03 v8.6 3.1)')
    raise S03Stop(f'the default {default!r} is not a point of the grid cut at the ceiling {T!r} (S03 v8.6 3.1)')


def sweep_grid_cut(ctrl, name, default, lo, hi):
    """The sweep grid of S03 v8.6 section 3.1: sweep_grid_n cut at ceiling(ctrl, name); the
    firmware default must still have its grid point (default_node: itself, or T when the cut made
    it T and both are one float32; a default above its ceiling stops S03)."""
    grid, info = cut_grid(*sweep_grid_n(name, default, lo, hi), ceiling(ctrl, name))
    try:
        default_node(grid, info, default)
    except S03Stop as exc:
        raise S03Stop(f'{ctrl} {name}: {exc}') from None
    return grid, info


def w_nodes(ctrl, name, default, lo, hi):
    """Nodes of the reference box W for one gain (S03 v8.6 section 3.1b item 2), ascending: the 11
    values of numpy.geomspace(a, b, 11) (the one within 1e-9 of the default is the default itself,
    as sweep_grid does) plus the extension points b r^k, then rules (1), (2) of section 3.1
    (cut_at_ceiling). Not nodes: the 0 point and the default when it was added as a point of its
    own. The nodes are taken from the grid lists, never from the luoi label of sweep.csv; first
    node a, last node w (b_n; b for omega_o, which has no extension)."""
    grid, info = sweep_grid_n(name, default, lo, hi)
    logs = [x for x in base_grid(grid, info) if x != 0.0]
    if len(logs) == N_GRID + 1:                         # the default was added beside the 11 log points
        logs.remove(default)
    if len(logs) != N_GRID:
        raise S03Stop(f'{ctrl} {name}: {len(logs)} log points in the grid, not {N_GRID}')
    return cut_at_ceiling(logs + info['noi'], ceiling(ctrl, name))[0]


def grid_label(x, default, noi):
    """sweep.csv 'luoi' of a grid value (S03 v8.3; agreed with CODE_C13 15/11): the default first (even on a
    log point), then 0, then the extension, else log."""
    if x == default:
        return 'mac_dinh'
    if x == 0.0:
        return '0'
    return 'noi' if x in noi else 'log'


L_SOURCES = ('0', 'hết lưới dưới', 'điểm kế tiếp không hoàn thành')
H_SOURCES = ('hết lưới', 'trần ω_o', 'trần bảng chặn', 'điểm kế tiếp không hoàn thành')   # 'trần bảng chặn': v8.6


def base_grid(grid, info):
    """The original grid of S03 v8.4 section 3.1 (11 log points, the default, 0): the grid of sweep_grid_n
    without its extension points, which come last."""
    base = grid[:len(grid) - info['n_noi']]
    if info['n_noi'] and not base[-1] < grid[len(base)]:
        raise S03Stop('the extension points are not after the original grid')
    return base


def level_sources(base, L, H, info, name):
    """(L_nguon, H_nguon) of S03 v8.4 section 3.1 on the ORIGINAL grid base: H = b is 'trần ω_o' for omega_o when
    b = 48 (the label comes first), else 'hết lưới'."""
    if L == 0.0:
        ls = '0'
    elif L == base[0]:
        ls = 'hết lưới dưới'
    else:
        ls = 'điểm kế tiếp không hoàn thành'
    if H == base[-1]:
        # v8.6: H = T, the end of the cut original grid, first over 'hết lưới'; the info of
        # sweep_grid_n (no T, n_bo keys) keeps working
        if info.get('n_bo') and H == info.get('T'):
            hs = 'trần bảng chặn'
        else:
            hs = 'trần ω_o' if name.startswith('omega_o') and info['b'] == OMEGA_O_MAX else 'hết lưới'
    else:
        hs = 'điểm kế tiếp không hoàn thành'
    return ls, hs


def levels_from_sweep(grid, default, ok):
    """L, H of section 3.1: ends of the longest run of adjacent grid points holding the default in
    which ok[i] (completes at every gate speed). Raises S03Stop for case (i)."""
    i0 = grid.index(default)
    if not ok[i0]:
        raise S03Stop('default gains do not complete at a gate speed (S03 3.1 (i))', case='(i)')
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


# ── 3.1 out-of-range check (S03 v8.4, C14-2) ─────────────────────────────────────────

KIEM_POINT = ('không cờ', 'chưa kết luận', 'cờ')
KIEM_TOTAL = ('đã kiểm đủ', 'kiểm dở', 'không kiểm')
KIEM_SPEED = ('trong_cong', 'y1_diem', 'status_diem', 'e_diem', 'tv_diem', 'y1_can', 'status_can', 'e_can', 'tv_can',
              'nguong_tv')
KIEM_COUNTS = ('n_diem_ngoai', 'n_da_kiem', 'n_chua_kiem', 'n_co', 'n_chua_ket_luan', 'n_khong_co', 'n_bao_rieng',
               'n_gain_chua_kiem')
KIEM_COLS = (['dong', 'controller', 'factor', 'value', 'luoi', 'can', 'can_value', 'ket_qua', 'buoc', 'bao_rieng']
             + [f'{k}_v{v}' for v in SPEEDS for k in KIEM_SPEED] + list(KIEM_COUNTS) + ['ghi_chu'])


def _num(r, k):
    try:
        x = float(r.get(k))
    except (TypeError, ValueError):
        return float('nan')
    return x


def comparable(r):
    """A run that can be compared (S03 v8.4 section 3.1): it exists, completes ('true'), status 'ok' (no
    track_err, no run_error, so Y2 carries no inconclusive flag), and its e_perp_max, tv_duty are finite
    (a non-finite value is not comparable: interpretation)."""
    return (r is not None and r.get('completed') == 'true' and r.get('status') == 'ok'
            and math.isfinite(_num(r, 'e_perp_max')) and math.isfinite(_num(r, 'tv_duty')))


def exact(x):
    """S84-1, S84V2-1, S84V3-1: the bounds of S03 v8.4 3.1 are inclusive and compared exactly, with no tolerance, on
    the decimal of each number as written in the CSV files (S03 writes a float as repr, the shortest decimal that
    reads back to the same binary64): 0.03 - 0.02 is 0.01 (binary64 subtraction gives 0.009999999999999998), two
    identical responses are never lower by a positive threshold, a point a few ULP past a bound is past it. A
    Fraction is already exact."""
    return x if isinstance(x, Fraction) else Fraction(repr(float(x)))


def not_above_by(x, ref, thr):
    """x - ref <= thr ('x not higher than ref by more than thr'), exactly (see exact)."""
    return exact(x) - exact(ref) <= exact(thr)


def lower_by_at_least(x, ref, thr):
    """ref - x >= thr ('x lower than ref by at least thr'), exactly (see exact)."""
    return exact(ref) - exact(x) >= exact(thr)


def y3_threshold():
    """S03 v8.6 section 5.2: the Y3 threshold of the check and of the analysis, the absolute
    0.4 s^-1 at every speed and every mu, worst and range (the exact decimal, Fraction(2, 5));
    no longer a share of the default-gain run (v8.4 and before)."""
    return exact(Y3_ABS)


def classify_point(pt, bd, gate, thr):
    """S03 v8.4 section 3.1: one point outside [L, H] against the run at the nearer level. pt, bd: {v: sweep row
    or None} of the point and of the level, middle mu; gate: speeds still in the gate; thr: {v: Y3 threshold}.
    Steps in order, the first that matches: 1 'không cờ' (at one gate speed the point has y1 'false' with
    status 'ok'), 2 'chưa kết luận' (at one gate speed the point or the level is not comparable; a non-finite Y3
    threshold too), 3 'cờ' (at every gate speed e_pt - e_lvl <= 0.01 m and tv_pt - tv_lvl <= thr; at one,
    e_lvl - e_pt >= 0.01 m or, thr > 0, tv_lvl - tv_pt >= thr), 4 'không cờ'. Returns (result, step, bao_rieng):
    bao_rieng = not 'cờ' and, at one gate speed where both are comparable, e_lvl - e_pt >= 0.01 m."""
    e_thr = THRESH['e_perp_max']
    if any(pt.get(v) is not None and pt[v].get('completed') == 'false' and pt[v].get('status') == 'ok' for v in gate):
        res, step = 'không cờ', 1
    elif any(not comparable(pt.get(v)) or not comparable(bd.get(v)) or not math.isfinite(thr[v]) for v in gate):
        res, step = 'chưa kết luận', 2
    else:
        e = {v: (_num(pt[v], 'e_perp_max'), _num(bd[v], 'e_perp_max')) for v in gate}
        tv = {v: (_num(pt[v], 'tv_duty'), _num(bd[v], 'tv_duty')) for v in gate}
        not_worse = all(not_above_by(*e[v], e_thr) and not_above_by(*tv[v], thr[v]) for v in gate)
        better = any(lower_by_at_least(*e[v], e_thr) or (thr[v] > 0 and lower_by_at_least(*tv[v], thr[v]))
                     for v in gate)
        res, step = ('cờ', 3) if (not_worse and better) else ('không cờ', 4)
    bao = res != 'cờ' and any(comparable(pt.get(v)) and comparable(bd.get(v))
                              and lower_by_at_least(_num(pt[v], 'e_perp_max'), _num(bd[v], 'e_perp_max'), e_thr)
                              for v in gate)
    return res, step, bao


def out_of_range_check(sweep_rows, levels, grids, defaults_by, controllers, gate, mu_mid, y3_ref, stop_case):
    """S03 v8.4 section 3.1 'Kiểm ngoài miền'. levels: {ctrl: {factor: (L, H, scale)}} (factors whose level rule
    ran); grids: {(ctrl, factor): (grid, info)} of sweep_grid_cut (original + extension after the cut, S03
    v8.6); defaults_by: {ctrl: {factor: default}}; stop_case: '(i)' / '(ii)' when S03 stopped there (then
    'không kiểm' for every controller). y3_ref is not used since v8.6 (the Y3 threshold is the absolute
    0.4 s^-1 of section 5.2); the signature is kept so callers do not change.
    A controller is 'đã kiểm đủ' when every factor has levels (every point outside them gets a result), else
    'kiểm dở' (the sweep stopped part way): the factors with levels are checked, every grid point of a factor
    without levels counts as not checked (which of them would be outside is not known). Returns (rows for
    kiem_ngoai_mien.csv: the points, then one total row per controller; summary {ctrl: ...})."""
    idx = {}
    for r in sweep_rows:
        if r.get('factor') in FACTORS.get(r.get('controller'), ()) and r.get('mu') == mu_mid:
            idx[(r['controller'], r['factor'], r['value'], r['v'])] = r
    rows, totals, summary = [], [], {}
    for ctrl in controllers:
        n = dict.fromkeys(KIEM_COUNTS, 0)
        lists = dict(co=[], chua_ket_luan=[], bao_rieng=[])
        notes = []
        if stop_case in ('(i)', '(ii)'):
            total = 'không kiểm'
            notes.append(f'S03 stopped at 3.1 {stop_case}')
        else:
            for name in FACTORS[ctrl]:
                lv = levels.get(ctrl, {}).get(name)
                if lv is None:
                    n['n_gain_chua_kiem'] += 1
                    if (ctrl, name) in grids:
                        n['n_chua_kiem'] += len(grids[(ctrl, name)][0])
                    else:
                        notes.append(f'{name}: grid not known')
                    continue
                L, H = lv[0], lv[1]
                grid, info = grids[(ctrl, name)]
                dflt = defaults_by[ctrl][name]
                thr = {v: y3_threshold() for v in SPEEDS}
                for x in grid:
                    if L <= x <= H:
                        continue
                    side, lvl = ('L', L) if x < L else ('H', H)
                    pt = {v: idx.get((ctrl, name, x, v)) for v in SPEEDS}
                    bd = {v: idx.get((ctrl, name, lvl, v)) for v in SPEEDS}
                    res, step, bao = classify_point(pt, bd, gate, thr)
                    row = dict(dong='diem', controller=ctrl, factor=name, value=x,
                               luoi=grid_label(x, dflt, info['noi']), can=side, can_value=lvl, ket_qua=res,
                               buoc=step, bao_rieng=bao)
                    for v in SPEEDS:
                        p, b = pt[v] or {}, bd[v] or {}
                        row.update({f'trong_cong_v{v}': v in gate,
                                    f'y1_diem_v{v}': p.get('completed', ''), f'status_diem_v{v}': p.get('status', ''),
                                    f'e_diem_v{v}': p.get('e_perp_max', ''), f'tv_diem_v{v}': p.get('tv_duty', ''),
                                    f'y1_can_v{v}': b.get('completed', ''), f'status_can_v{v}': b.get('status', ''),
                                    f'e_can_v{v}': b.get('e_perp_max', ''), f'tv_can_v{v}': b.get('tv_duty', ''),
                                    f'nguong_tv_v{v}': float(thr[v])})      # CSV: the threshold as a float
                    rows.append(row)
                    n['n_diem_ngoai'] += 1
                    n['n_da_kiem'] += 1
                    n[{'cờ': 'n_co', 'chưa kết luận': 'n_chua_ket_luan', 'không cờ': 'n_khong_co'}[res]] += 1
                    tag = f'{name}={x!r}'
                    if res == 'cờ':
                        lists['co'].append(tag)
                    elif res == 'chưa kết luận':
                        lists['chua_ket_luan'].append(tag)
                    if bao:
                        n['n_bao_rieng'] += 1
                        lists['bao_rieng'].append(tag)
            total = 'đã kiểm đủ' if n['n_gain_chua_kiem'] == 0 else 'kiểm dở'
        totals.append(dict(dong='tong', controller=ctrl, ket_qua=total, **n, ghi_chu='; '.join(notes)))
        summary[ctrl] = dict(ket_qua=total, **n, **lists)
    return rows + totals, summary


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


def gate_meta():
    """meta.yaml 'gate' (S03 v8.2): versions and settings of the G1-G5 scoring."""
    return dict(version=G.GATE_VERSION, eval_version=E.EVAL_VERSION, u_t=GATE_U_T, u_p=0.0, u_psi_deg=0.0,
                decide=list(G.DECIDE_ALL), thresholds=G.THRESHOLDS, columns=GATE_RUN_COLS,
                g1_columns=G1_RUN_COLS, g1_fields='o1_gate_eval.g1_fields_v7 (B81-3, one way of using the frames)')


def evaluator_sha256():
    """SHA-256 of the evaluator files as imported in this process (S03 v8.2 section 8 condition 8), of the
    bytes on disk and after CRLF -> LF (a Windows checkout holds CRLF; the git blob is LF)."""
    out = {}
    for m in (o1, E, G, E.L, E.L.ot):
        with open(m.__file__, 'rb') as f:
            b = f.read()
        out[_rel_here(m.__file__)] = dict(sha256=hashlib.sha256(b).hexdigest(),
                                          sha256_lf=hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest())
    return out


def load_test_points(path=TEST_POINTS, trial=False):
    """S03 v8 sections 7, 8 condition 7 (S3V8-3): the bytes of the test-set file are read ONCE,
    before the main design. From those bytes: the SHA-256 of the bytes as read and of the same
    bytes with CRLF -> LF (git keeps LF, the work tree may hold CRLF) and with LF -> CRLF, the
    CSV and its checks (check_test_points_text). The content must equal the blob of the file in
    HEAD, both after CRLF -> LF (git status alone is not evidence: assume-unchanged hides a
    change). The generation record is <name>.gen.json beside this very file; its sha256 must be
    the SHA-256 of the bytes read or of their LF or CRLF form (R2-2: the committed record holds
    the SHA-256 of the CRLF file written on Windows; a clean LF checkout is the same content).
    Without --trial a failed git / generation check stops S03; with --trial it is recorded in
    'problems'. Returns (points, info for meta.yaml). A stop carries the info read so far in
    S03Stop.tp_info (R2-4: meta.yaml records what was read, the file is not read again)."""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except OSError as exc:
        e = S03Stop(f'test points: {path} cannot be read: {type(exc).__name__}: {exc} (S03 v8 section 8.7)')
        e.tp_info = dict(file=_rel_here(path), read=f'no: {type(exc).__name__}: {exc}')
        raise e from None
    lf = raw.replace(b'\r\n', b'\n')
    info = dict(file=_rel_here(path), sha256=hashlib.sha256(raw).hexdigest(),
                sha256_lf=hashlib.sha256(lf).hexdigest(),
                sha256_crlf=hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest(), git=file_git_state(path))

    def stop(msg):
        exc = S03Stop(msg)
        exc.tp_info = info
        return exc
    try:
        pts = check_test_points_text(raw.decode('utf-8'))
    except S03Stop as exc:
        info['problems'] = [f'condition 7: {exc}']
        raise stop(str(exc)) from None
    except (ValueError, UnicodeDecodeError) as exc:          # e.g. a code that is not a number
        info['problems'] = [f'condition 7: {type(exc).__name__}: {exc}']
        raise stop(f'test points: {type(exc).__name__}: {exc} (S03 v8 section 8.7)') from None
    blob = _head_blob(path)
    info['head_blob_sha256'] = hashlib.sha256(blob).hexdigest() if blob is not None else None
    info['matches_head_blob_lf'] = blob is not None and blob.replace(b'\r\n', b'\n') == lf
    gen_path, gen = test_points_gen_path(path), None
    try:
        if os.path.isfile(gen_path):
            with open(gen_path, 'rb') as f:
                gen = json.loads(f.read().decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as exc:  # S3V8-3: I/O errors too, recorded
        gen = dict(unreadable=f'{type(exc).__name__}: {exc}')
    info['generation_file'] = _rel_here(gen_path)
    info['generation'] = gen
    info['generation_sha256_matches'] = bool(isinstance(gen, dict) and gen.get('sha256') in
                                             (info['sha256'], info['sha256_lf'], info['sha256_crlf']))
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
        raise stop(f'{path}: ' + '; '.join(problems) + ' (S03 v8 sections 7, 8.7)')
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


def check_candidate(ctrl, g, w):
    """S03 v8.6 section 3.1b item 4, Codex C3-2: the bounds check of one gain set of the screening
    step, in this order. g = {factor: binary64 value} with exactly the keys FACTORS[ctrl];
    w = {factor: (a_i, w_i)}, the reference box W; a wrong key set of g or w is malformed input
    (ValueError).
      1. every binary64 value finite and a_i <= x <= w_i (exact comparison, no tolerance);
      2. the float32 of every omega_o* gain <= OMEGA_O_MAX (48);
      3. the float32 of every gain inside the gain table (gain_table.classify, trong_bang; lower
         and upper bound, 0 accepted where the lower bound is 0);
      4. the LADRC triangle (classify, trong_tam_giac).
    A failure of 1-3 is a bounds failure: S03Stop (S03 stops; the caller keeps it sealed). Only a
    broken triangle is classifiable: returns 'ngoai_bang'; 'ok' otherwise."""
    want = set(FACTORS[ctrl])
    if set(g) != want or set(w) != want:
        raise ValueError(f'{ctrl}: gains {sorted(g)} and box {sorted(w)} must be exactly {sorted(want)}')
    for name in FACTORS[ctrl]:
        x, (a_i, w_i) = g[name], w[name]
        if not (math.isfinite(x) and a_i <= x <= w_i):
            raise S03Stop(f'{ctrl} {name}: {x!r} is not finite or outside the box [{a_i!r}, {w_i!r}] '
                          f'(S03 v8.6 3.1b item 4 step 1: a bounds failure, S03 stops)')
    for name in FACTORS[ctrl]:
        if name.startswith('omega_o') and float(np.float32(g[name])) > OMEGA_O_MAX:
            raise S03Stop(f'{ctrl} {name}: float32 {float(np.float32(g[name]))!r} > {OMEGA_O_MAX} '
                          f'(S03 v8.6 3.1b item 4 step 2: a bounds failure, S03 stops)')
    v = gain_table.classify(ctrl, g)
    if not v.trong_bang:
        raise S03Stop(f'{ctrl}: {"; ".join(v.vi_pham_bang)}: outside the gain table '
                      f'{gain_table.TABLE_VERSION} (S03 v8.6 3.1b item 4 step 3: a bounds failure, '
                      f'S03 stops)')
    return 'ok' if v.trong_tam_giac else 'ngoai_bang'


def outside_triangle(ctrl, g):
    """Column ngoai_tam_giac of a design point (S03 v8.6 section 3.2 last item): True when the set
    {factor: binary64 value} breaks the LADRC triangle omega_c + 2 omega_o < 200 on an axis
    (binary64 sum of the float32 values). PID: always False."""
    return not gain_table.classify(ctrl, g).trong_tam_giac


def kep_frac_fields(ctrl, row, log):
    """{kep_frac_vx, kep_frac_vy, kep_frac_wz} of one run (S03 v8.6 section 2.1 last item, 7,
    11 item 6 (c)): for PID with a log holding the 'kep' channel, the fractions of
    fwsim.kep_indicator on the O1 window up to row['t_end'] (ticks at the PID integral clamp over
    ticks with the motor on; NaN when no such tick). ADRC and a run_error row (log None): the
    three columns stay empty. A PID log without the 'kep' channel (C3-48: a DLL without the
    clamp export cannot run S03) or a ValueError of the indicator (kep_lech set, a bad flag) is
    a tool error: S03Stop."""
    out = dict.fromkeys(KEP_FRAC_COLS, '')
    if ctrl == 'ADRC' or log is None:
        return out
    if 'kep' not in log:
        raise S03Stop(f'{ctrl}: the log has no kep channel: a DLL without the clamp export '
                      'cannot run S03 (S03 v8.6 2.1, 11 item 6 (c); Codex C3-48)')
    try:
        ind = fwsim.kep_indicator(log, row['t_end'])
    except ValueError as exc:
        raise S03Stop(f'{ctrl}: the kep_indicator self-check failed: {exc} (S03 v8.6 2.1)') from None
    out.update({k: ind[k] for k in KEP_FRAC_COLS})
    return out


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


def check_table(ctrl, points, gains_at):
    """S03 v8.6 section 3.1b item 4 (section 11 item 6 (a)), the bounds check before the main design: the
    float32 of every gain of every point is within the gain table, lower and upper bound (0 accepted where
    the lower bound is 0; gain_table.classify, the same check as before sending a gain). The LADRC triangle
    is NOT checked here: a design point outside it runs, flagged ngoai_tam_giac (section 3.2). Kept apart
    from check_bounds, which other scripts call (rsm_compare) and which keeps its v8 behaviour. Raises
    S03Stop."""
    for pid, _box, codes in points:
        v = gain_table.classify(ctrl, gains_at(codes))
        if not v.trong_bang:
            raise S03Stop(f'{ctrl} point {pid}: {"; ".join(v.vi_pham_bang)}: outside the gain table '
                          f'{gain_table.TABLE_VERSION} (S03 v8.6 3.1b item 4)')


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


def run_in_new_process(dll, runs, work, timeout=None):
    """Run (ctrl, gains, v, mu) jobs in a fresh Python process (DLL loaded again). work must be
    a folder under sealed/ when the runs belong to the main design (S3C-2); the job files are
    removed in every case. timeout: seconds for subprocess.run (None: no limit, as before); a
    subprocess.TimeoutExpired goes up to the caller."""
    os.makedirs(work, exist_ok=True)
    job = os.path.join(work, f'job_{time.time_ns()}.pkl')
    try:
        with open(job, 'wb') as f:
            pickle.dump(dict(dll=dll, runs=runs), f)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', job], capture_output=True,
                           text=True, timeout=timeout)
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


def _write_csv_to(f, rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    w.writerows({k: fmt(r.get(k, '')) for k in cols} for r in rows)


def write_csv(path, rows, cols=None):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        _write_csv_to(f, rows, cols)


def design_cols(rows):
    """design.csv header (S03 v8.6): the old columns in row order, ngoai_tam_giac last; None for
    no rows (write_csv then writes nothing, the file is empty)."""
    if not rows:
        return None
    return [k for k in dict.fromkeys(k for r in rows for k in r) if k != TRIANGLE_COL] + [TRIANGLE_COL]


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
    (S03 3.2, S3C-11). y3_ref is not used since v8.6 (the Y3 threshold is the absolute 0.4 s^-1
    of section 5.2); the signature is kept so callers do not change."""
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
                thr = THRESH[resp]
                note = notes.get((box, v), '')
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


def tracked_changes(repo):
    """Tracked files that differ from HEAD (staged or not); untracked files are not sources (S3G2-2: an
    output folder inside the repo is not a change of the code)."""
    try:
        out = subprocess.run(['git', '-C', repo, 'diff', '--name-only', 'HEAD'], capture_output=True, text=True,
                             check=True).stdout
        return sorted(p for p in out.splitlines() if p and not p.startswith(('fwsim/build/', '.vscode/')))
    except (OSError, subprocess.CalledProcessError) as e:
        return [f'UNKNOWN: {e}']


def source_snapshot(repo):
    """S3G-1, S3G2-2: git state for meta.yaml, and the key compared at the start and the end of a run: the
    commit, the tracked files changed, the evaluator files."""
    st = git_state(repo)
    return dict(script=st, key=dict(commit=st['commit'], tracked=tracked_changes(repo),
                                    evaluator=evaluator_sha256()))


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
    import s03_sangloc as sl              # lazy: s03_sangloc imports this module (circular at top level)
    # the docstring holds Vietnamese; --help must print on every console encoding
    ap = argparse.ArgumentParser(description=__doc__.encode('ascii', 'replace').decode(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument('--screen-u', default=None,
                    help='screening numbers file (S03 v8.5 8.9); default the committed fwsim/s03_screen_u.csv')
    ap.add_argument('--screen-r', type=int, default=0,
                    help='rerun index r of the numbers file (S03 3.1b item 3): checked, the file '
                    'must be the numbers of seed 2026112100 + 10 r + t')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help='worker processes of the screening step (S03 3.1b item 5)')
    ap.add_argument('--sangloc-thu', action='store_true',
                    help='tool check of section 3.1b (S03 8.11): its own numbers file; trial: true')
    ap.add_argument('--sangloc-thu-n0', type=int, default=None,
                    help='S0 rows of --sangloc-thu (S03 8.11)')
    ap.add_argument('--sangloc-thu-n', type=int, default=None,
                    help='N of --sangloc-thu (S03 8.11); the formula is reported, not applied')
    ap.add_argument('--child', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if len(a.mu) not in (1, 3) or (len(a.mu) == 1 and not a.trial):
        sys.exit('--mu needs three levels (S03 2.2); a single level only with --trial')
    if a.trial and a.mu != [0.5]:
        sys.exit('--trial runs with mu 0.5 only (S03 2.2: the fwsim assumption)')
    if not a.gate_speeds or any(not any(math.isclose(g, s) for s in SPEEDS) for g in a.gate_speeds):
        sys.exit(f'--gate-speeds must be a non-empty subset of {list(SPEEDS)} (S3C-10)')
    gate = [s for s in SPEEDS if any(math.isclose(g, s) for g in a.gate_speeds)]
    is_trial = a.trial or a.sangloc_thu
    # S03 3.1b: --trial (one mu level) cannot run the step; --sangloc-thu is its tool check (8.11);
    # every other run is 'chinh' (three mu levels)
    screening = None if a.trial else ('thu' if a.sangloc_thu else 'chinh')
    if a.sangloc_thu:
        if a.trial:
            sys.exit('--sangloc-thu is not run with --trial (S03 8.11: the three measured mu levels)')
        if not a.screen_u:
            sys.exit('--sangloc-thu needs --screen-u, its own screening numbers file (S03 8.11)')
        if os.path.basename(a.screen_u) == os.path.basename(sl.SCREEN_U) or \
                os.path.abspath(a.screen_u) == os.path.abspath(sl.SCREEN_U):
            sys.exit('--sangloc-thu --screen-u must be another file than fwsim/s03_screen_u.csv '
                     '(S03 8.11)')
        if a.sangloc_thu_n0 is None or not 1 <= a.sangloc_thu_n0 <= sl.SET_ROWS['S0']:
            sys.exit(f'--sangloc-thu-n0 must be in 1..{sl.SET_ROWS["S0"]} (S03 8.11)')
        if a.sangloc_thu_n is None or not 1 <= a.sangloc_thu_n <= sl.SET_ROWS['S1']:
            sys.exit(f'--sangloc-thu-n must be in 1..{sl.SET_ROWS["S1"]} (S03 8.11)')
        if a.screen_r != 0:
            sys.exit('--screen-r is 0 with --sangloc-thu (S03 8.11)')
    elif a.sangloc_thu_n0 is not None or a.sangloc_thu_n is not None:
        sys.exit('--sangloc-thu-n0 and --sangloc-thu-n only with --sangloc-thu')
    if a.screen_r < 0:
        sys.exit('--screen-r must be >= 0 (S03 3.1b item 3)')
    if a.jobs < 1:
        sys.exit('--jobs must be >= 1')
    if a.trial and (a.screen_u is not None or a.screen_r != 0):
        sys.exit('--trial runs no screening step: --screen-u and a --screen-r other than 0 are refused')
    if a.screen_r >= 1:
        # S03 3.1b item 3: a rerun index r >= 1 runs on a NEW numbers file (never the committed r = 0)
        if not a.screen_u:
            sys.exit('--screen-r >= 1 needs --screen-u, the numbers file of that rerun '
                     '(S03 3.1b item 3)')
        if os.path.basename(a.screen_u) == os.path.basename(sl.SCREEN_U) or \
                os.path.abspath(a.screen_u) == os.path.abspath(sl.SCREEN_U):
            sys.exit('--screen-r >= 1 --screen-u must be another file than '
                     'fwsim/s03_screen_u.csv (S03 3.1b item 3)')
    # C3-37: a run WITH the screening step ('chinh' and the tool check 'thu'; --trial keeps its
    # own rule) needs both controllers once and three finite, distinct mu levels
    if screening is not None:
        if sorted(a.controllers) != sorted(CONTROLLERS):
            sys.exit('a run with the screening step needs both controllers PID and ADRC, each once '
                     '(S03 3.1b items 1, 10; Codex C3-37); --trial keeps its own rule')
        if not all(math.isfinite(m) for m in a.mu) or len(set(a.mu)) != 3:
            sys.exit('a run with the screening step needs three finite, distinct mu levels '
                     '(S03 2.2, 3.1b item 5; Codex C3-37)')
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
    # S3G-1 (Codex round 1): the script commit and the evaluator files are taken BEFORE the first run and
    # taken again at the end; S3G2-1: the evaluator files are also compared with their hash read at the end
    # of this module's import. A difference between these snapshots is recorded and S03 ends with exit
    # code 2. Limits (S3G3-1, S3G3-2, deferred): only the snapshots of THIS process are compared, so a
    # source changed and changed back between two snapshots is not seen; the hash at import is a read of
    # the files on disk a few milliseconds after the evaluator modules were loaded, not of the loaded
    # code itself; child processes (determinism check, run_error diagnosis) import again from disk and
    # their sources are not locked: the determinism check of section 3.4 can see such a change only
    # partly (its 4 points).
    src_start = source_snapshot(os.path.join(HERE, '..'))
    t0 = time.time()
    sweep_rows, level_rows, design_rows, all_runs = [], [], [], []
    eff_rows, curv_rows, ver_rows, fail_rows, det_rows = [], [], [], [], []
    y3_ref, rerun_notes, inner_rule, no_concl = {}, [], {}, {}
    grid_info = {}                                    # S03 v8.3: (controller, factor) -> sweep_grid_n info
    grids = {}                                        # S03 v8.4: (controller, factor) -> (grid, info) for the check
    levels, dflts = {}, {}
    stop_case = None                                  # S03 v8.4: '(i)' / '(ii)' when the sweep stopped there
    kiem_rows, kiem_sum = [], {}
    stop_before, stop_public, stop_sealed = None, None, None   # before the sweep / sweep / sealed window
    worker_tree_stopped = True
    stop_test, test_rows, test_notes, test_s = None, [], [], None
    t_sweep_end = None
    t_sl0 = t_sl1 = t_md0 = t_md1 = None        # _mono() bounds of the screening step and main design
    pha_live = None                           # the live per-controller phase entry of step 4e
    tp_info, test_pts = None, []                      # S3V8-3: the test file is read once, at the start
    screen, screen_info = None, None                  # S03 3.1b: sets and info of the numbers file
    pha_notes, sweep_planned = [], None               # sealed/notes.json 'pha' (section 7 [v8.5])
    plan = {}
    files = {}                                        # sha256 of the public files (empty before the sweep)
    meta_state = {}                                   # what write_meta(final=True) computed (status, src)

    def checked(ctrl, g, v, mu, ids=None, sink=None, work=None, notes_to=None, sweep=None):
        """One run; a run_error is recorded (S3C-12), rerun once in a new process for diagnosis
        (B8b-20), then S03 stops. ids = (point_id, box) for a main-design run (sealed). The test
        set (v8) passes its own sink (rows), work folder and notes list, all under sealed_test/.
        sweep = (factor, value, is_default, luoi, tran) of a sweep run: its run_error row goes to
        sweep.csv (S3G-2)."""
        row, log = safe_run(fw, path, ctrl, g, v, mu)
        if row['status'] != 'run_error':
            return row, log
        if ids is not None:
            (all_runs if sink is None else sink).append(dict(controller=ctrl, point_id=ids[0], box=ids[1],
                                                             mu=mu, v=v,
                                                             ngoai_tam_giac=outside_triangle(ctrl, g),
                                                             **row, **kep_frac_fields(ctrl, row, log)))
        elif sweep is not None:
            sweep_rows.append(dict(controller=ctrl, factor=sweep[0], value=sweep[1], is_default=sweep[2],
                                   luoi=sweep[3], tran=sweep[4], v=v, mu=mu, **row))
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

    def final_status():
        return ('stopped before the sweep: ' + stop_before if stop_before else
                'stopped in the sweep: ' + stop_public if stop_public else
                'stopped after the sweep (the reason is in sealed/stop.json)' if stop_sealed else
                'stopped in the test set (the reason is in sealed_test/stop.json)'
                if stop_test else 'ran to the end')

    def write_meta(final, running=None):
        """meta.yaml (S03 section 7, 8.9, 8.10): written three times — at the start, before the
        first run (final=False, running 'started': the SHA-256 of the numbers file is public from
        the start), with the PLAN of section 3.1b before S0 runs (final=False, running 'the sweep
        is done'), and again when S03 ends (final=True: status, source check, every count)."""
        b = fw.build_info
        dll_sha = sha256_file(a.dll)
        sweep_s = round((t_sweep_end or time.time()) - t0)
        if final:
            # S3G-1: the sources the runs were made with are those taken at the start; the end must
            # give the same (S3G2-1: and the evaluator files are those hashed at import; S3G2-2:
            # untracked files do not count)
            src_end = source_snapshot(os.path.join(HERE, '..'))
            changed = (src_end['key'] != src_start['key']
                       or src_start['key']['evaluator'] != EVAL_SHA_AT_IMPORT)
            check = (dict(result='CHANGED during the run: not usable as a main result',
                          start=src_start['key'], end=src_end['key'],
                          evaluator_at_import=EVAL_SHA_AT_IMPORT) if changed
                     else 'same at import, start and end of the run')
            meta_state['src_changed'] = changed
            det = ('not run' if (stop_before or stop_public)
                   else 'see sealed/ (S03 stopped after the sweep)' if stop_sealed
                   else 'passed (detail in sealed/determinism_check.csv)')
            tset = ('not run (S03 stopped before it)' if (stop_before or stop_public or stop_sealed)
                    else 'sealed in sealed_test/ (not looked at before the RSM coefficients are '
                    'saved, S03 v8 section 7)')
            stat = final_status()
            if changed:
                stat += '; the script or evaluator sources changed during the run (S3G-1); ' \
                        'see meta.yaml source_check'
            meta_state['status'] = stat
        else:
            check = 'pending (checked at the end of the run)'
            det, tset = 'pending', 'pending'
            stat = running
        # S3V8-3: what was read at the start, not a second read of the file now
        tp_meta = tp_info if tp_info is not None else dict(
            file=_rel_here(a.test_points), read='no (the test file was not read)')
        # S03 3.1b item 10, 8.10: the PLAN only — no threshold, box, nguon_mien, branch, N used or
        # count of the step ever leaves sealed/
        sang_plan = dict(che_do=('khong_chay: --trial mot muc mu' if screening is None
                                 else 'kiem_cong_cu' if screening == 'thu' else 'chinh'),
                         m_min=sl.M_MIN, T_max_gio=sl.T_MAX_H, T_max_nhan=sl.T_MAX_LABEL,
                         cong_thuc_N=('N = min(N_tk, N_gio); N_tk = the smallest multiple of 1000 '
                                      'in [3000, 10000] with N q_min >= 2 m_min (10000 when none); '
                                      'N_gio = the largest multiple of 1000, at most 10000, with '
                                      '3 (N / 1000) t_S0 <= T_max (0 when none), t_S0 = wall-clock '
                                      'seconds of the S0 phase; q_min = the smallest over '
                                      'controller x mode of (S0 gain sets feasible with f1 <= '
                                      'smallest feasible f1 + 0.01 m) / 1000; no screening when '
                                      'N < 3000, when N q_min < m_min, or when S0 has more than '
                                      '10 % unclassifiable gain sets in a controller and mode '
                                      '(S03 3.1b item 10)'),
                         ti_le_khong_phan_loai_toi_da=0.1, so_tien_trinh=a.jobs,
                         seed='2026112100 + 10 r + t (tool check: 2026112105 + t); checked bit '
                              'for bit, see file_so.nguon_sinh',
                         r=a.screen_r, file_so=screen_info,
                         kiem_cong_cu=(dict(n0=a.sangloc_thu_n0, n=a.sangloc_thu_n)
                                       if screening == 'thu' else None))
        kep_channel = bool(getattr(fw, 'kep_cols', 0))
        text = f"""spec: S03 v8.6 (06_SPECS/2026-11-05_Spec_S03_DOE_chon_gain.md)
date: {time.strftime('%Y-%m-%d')}
setup: sim                     # fwsim: ESP32 firmware C code (DLL) + rpi5/plant_step.py, NOT hardware
operator: Claude (phiên CODE)
trial: {str(is_trial).lower()}     # true: tool check (--trial mu 0.5, or --sangloc-thu, the tool check of section 3.1b with three mu levels), never main numbers
sealed: "L56 / S03 section 1: everything derived from the main-design responses is in sealed/ (with its own manifest.json) and is not looked at before campaign A ends"
status: {json.dumps(stat)}
script: {json.dumps(src_start['script'])}   # taken before the first run (S3G-1)
source_check: {json.dumps(check)}   # script and evaluator again at the end (S3G-1)
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
seed: {SEED}                       # plant has no random input; IMU noise replay seed (imu.noise_seed)
determinism_seed: {DET_SEED}       # draws the 3 corners of the determinism check (S03 3.4)
determinism: "{det}"
speeds: {json.dumps(list(SPEEDS))}
gate_speeds: {json.dumps(gate)}
controllers: {json.dumps(a.controllers)}
factors: {json.dumps(FACTORS)}
temporary_bounds_for_the_grid: {json.dumps({m: tune_ga.BOUNDS[m] for m in a.controllers})}   # tune_ga.BOUNDS (S03 3.1)
thresholds: {json.dumps(THRESH)}   # S03 v8.6 section 5.2: tv_duty 0.4 s^-1 absolute, every speed and mu, worst and range (y3_reference below is reported beside it, not a base)
y3_reference: {json.dumps({f'{c} v={v}': y for (c, v), y in y3_ref.items()})}
o1: {json.dumps(dict(A=o1.A, a_r=o1.A_R, D=o1.D, D_d=o1.D, t_d=o1.T_D, t_late=o1.T_LATE, delta=o1.DELTA, v_d=o1.V_D, c_cens=o1.C_CENS))}
grid: {json.dumps({f'{c} {n}': {k: v_ for k, v_ in i.items() if k != 'noi'} for (c, n), i in grid_info.items()})}   # S03 v8.3 section 3.1: a, b, r, K, b_n, points per gain; from v8.6 the grid is cut at the gain-table ceiling T: T, n_bo (points dropped), noi / n_noi / n_points after the cut, b_n = the largest point after the cut (T when cut, else the extension end; for omega_o, which has no extension, the end b of its grid)
kiem_ngoai_mien: {json.dumps(kiem_sum, ensure_ascii=False)}   # S03 v8.4 section 3.1: total per controller (đã kiểm đủ / kiểm dở / không kiểm), counts, flagged (co), inconclusive (chua_ket_luan) and bao_rieng points; detail in kiem_ngoai_mien.csv
gate: {json.dumps(gate_meta())}   # S03 v8.2 section 7: G1-G5 per run (o1_gate.gate), U = 0 on fwsim
evaluator_sha256: {json.dumps(EVAL_SHA_AT_IMPORT)}   # S03 v8.2 section 8 condition 8: S03 and S04 on the same evaluator; the files as hashed when the script was imported (S3G-1, S3G2-1)
sweep_run_error_reruns: {json.dumps([n for n in rerun_notes if n['point'] is None], default=str)}
planned_runs_per_controller: {json.dumps(PLANNED_RUNS)}   # S03 v8 3.2, 7 (rows actually run: sealed/, sealed_test/)
test_points: {json.dumps(tp_meta)}   # S03 v8 3.2, 7, 8.7 (S3V8-3): read once at the start; sha256 of the bytes read, sha256_lf after CRLF -> LF, matches_head_blob_lf, generation from <name>.gen.json beside the file
test_set: {json.dumps(tset)}
sang_loc: {json.dumps(sang_plan, ensure_ascii=False)}   # PLAN of S03 3.1b (item 10, 8.10): mode, m_min, T_max, the formula of N, the numbers-file info; NOTHING of the results of the step
kep_channel: {str(kep_channel).lower()}                 # fwsim.kep_cols of the DLL: the kep_frac columns are filled only when true
interpretations:   # of this script, not written in S03 v8 / O1 v7; Liam or THESIS correct if wrong
  - "completion: conditions 1, 5 at the instant, conditions 2 and 3 together as the trailing unbroken t_d window (condition 2 over the window: L63 (2), Liam 07/11), condition 4 strictly before the deadline; after a tracking error that came before the progress, the window runs to the deadline (o1_feasibility.evaluate_v7)"
  - "sweep level rule: a point is 'completing' only with y1 'true' at every gate speed (inconclusive breaks the run)"
  - "inner box trigger: corners with y1 'false' only (inconclusive not counted)"
  - "a stop of the spec (3.1 (i), (ii), 3.4, run_error) ends S03 for both controllers; the default gains of every controller are checked first"
  - "logs: log.csv (%.17g) of the four determinism points and npz (exact bits) of the centre and every main-design run not completed, all in sealed/logs/; npz of every sweep run not completed in logs_sweep/ (public, as the sweep)"
  - "the inner box (codes +-0.5) always runs for every controller (L41, Liam 06/11 'Chot B'), so the number of runs does not depend on the results; both boxes are always analysed (the same work), and the inner box is kept (effects, Y2 versions, fail counts; curvature is for the outer box only; at every speed, as before L41) only when the rule of S03 3.2 is on at one speed or more (> 64/128 corners 'false' at the middle mu); otherwise its runs are only rows of sealed/runs.csv"
  - "elapsed_s here is the sweep only; the total and the main-design time are in sealed/notes.json (L56, Codex S3E-1; S03 section 7 asks elapsed_s in meta.yaml)"
  - "public channels left (L56, S3C-3, S3E-1; Liam 06/11 'Chot B', 'Duyet khuyen nghi'): status, exit code and console say that S03 stopped after the sweep (the reason and its phase are in sealed/stop.json); the wall-clock time still varies a little with the results (scoring a run that does not complete is a little longer, evaluate_v7 closes the window early on completion; v8.2: the G4 scoring of o1_gate_eval works on the scored frames and their projection solutions, so it may add to this, S3G-7; writing the npz of the runs not completed; machine noise): accepted, not measured"
  - "the upper-end tracking flag uses the 400 001-point path grid (about 15 um); at s = S the window is cut by the end of the path, not flagged (Codex S3C-5, deferred)"
  - "v8: run order per controller centre, outer, inner, axial (3 mu x 3 speeds each), then determinism and analysis; sealed/ is written and closed after every controller; then the test set of every controller (middle mu x 3 speeds); a stop before the test set means the test set does not run"
  - "v8: test-set logs follow the rule of section 7 (npz of every run not completed; there is no centre point in the test set); the compressed npz is made for every run (same work)"
  - "v8: the test-points file is read once before the main design (S3V8-3); 'committed' (section 8.7) = tracked, unmodified for git status AND the bytes read match the blob in HEAD after CRLF -> LF of both; its generation record is <name>.gen.json beside the file read, with the sha256 of the bytes read or of their LF or CRLF form (R2-2: the committed record holds the SHA-256 of the CRLF file; a clean LF checkout is the same content); a stop at this check keeps what was read in test_points (R2-4); with --trial a failed check is allowed and recorded in test_points.problems"
  - "v8 (S3V8-2): warnings, prints and tracebacks of the test phase go to sealed_test/diag.txt and any exception of the phase, writing its outputs included, is a stop recorded in sealed_test/stop.json (written first; R2-1); the warning policy in force is kept, a warning made an error stays an error (R2-3); the same channel now holds for the sealed window from the end of the sweep until sealed/ is closed (sealed/diag.txt)"
  - "v8: n_upper_frames, n_back_frames count the frames evaluate_v7 kept for scoring (t_k <= t_end), as section 4 defines them; they are diagnostics, never responses"
  - "v8.2: every run (sweep, main design, axial, test set, determinism reruns) is scored by o1_gate_eval.evaluate_fig8, which calls evaluate_v7 unchanged (the S03 v8 columns keep their values), then o1_gate.gate with u_t = 0 (fwsim knows the command start; the 50 Hz hold is a property of the command path, not added to U, o1_gate docstring), U_p = U_psi = 0 (true plant pose) and every condition deciding (DECIDE_ALL)"
  - "v8.2: G1-G5 columns use the names of tools/rq3_log.py GATE_COLS (B81-3); gate_json_sha256 is not written; gate_outcome is the O1 3.5 outcome over G1-G5; with U = 0, lo = hi = q"
  - "v8.2: a failure of the gate scoring (e.g. the rebuilt nearest distance of G4 not the row e_perp) is a run_error, so S03 stops (B8b-20) as for a DLL error"
  - "B81-3 (15/11, Liam CODE_FEAS 'code di'): the G1 fields of O1 3.5 last block and the common domain are written (G1_RUN_COLS after gate_outcome): t_done = t_end when completed 'true', t1 = first tracking-error frame, prog_before_t1 = progress strictly before t1, cond25_before_deadline = cond2_5_ever (o1_gate_eval.g1_fields_v7, from the evaluate_v7 summary, which is unchanged); G1 recomputed from them must agree with the state from completed (o1_gate.gate raises, so a run_error); one way of using the frames: g1_ways 1, g1_t_done_2 and g1_cat_t1 empty, _chung the same as the union; gate_version stays gate_v3.2 (tools/rq3_log.py accepts only that version)"
  - "v8.3: the sweep grid is the v8 grid (sweep_grid, unchanged; C12 keeps it for its box [a, b]) plus b r^k, k = 1..K, r = (b/a)^(1/10), K the smallest integer with r^K >= 10 (1 - 1e-9), found by a loop on r**k; omega_o is not extended (cap 48); v8.4: the extension still runs with the sweep, only for the out-of-range check; the level rule runs on the original grid (11 log points, the default, 0)"
  - "v8.3 (agreed with CODE_C13 15/11): luoi of the default value is mac_dinh even when it is on a log point, the (default check) and (default) rows are mac_dinh; L_nguon 'hết lưới dưới' when L is the lowest grid point and L > 0; v8.4: H_nguon 'hết lưới' when H = b, 'trần ω_o' instead for omega_o when b = 48; levels.csv b_n: see v8.6"
  - "v8.4 out-of-range check: a run is comparable when it exists, completed 'true', status 'ok' and its e_perp_max and tv_duty are finite; a point below L is put against the run at L, above H against H; the Y2 threshold is 0.01 m and the Y3 threshold is the absolute 0.4 s^-1 of section 5.2 (v8.6; before: 20 % of the default-gain run), compared exactly as a decimal; the bounds are inclusive and compared exactly, with no tolerance, on the decimal of each number as written in the CSV (repr of the float) (S84-1, S84V2-1, S84V3-1: 0.03 - 0.02 is 0.01, not the binary64 0.009999999999999998; two identical responses never count as lower by a positive threshold; a point a few ULP past a bound is past it; the column nguong_tv holds the threshold as a float); bao_rieng = not flagged and, at one gate speed where both are comparable, e_perp_max lower by >= 0.01 m; buoc is the step of section 3.1 that matched"
  - "v8.4 out-of-range check totals: 'không kiểm' for every controller when S03 stopped at 3.1 (i) or (ii); 'đã kiểm đủ' when every factor of the controller has levels; else 'kiểm dở' (the sweep stopped part way): factors with levels are checked, every grid point (original and extension) of a factor without levels counts in n_chua_kiem, n_gain_chua_kiem counts those factors; every grid is computed before the first sweep run. v8.5 (K6): a flagged or inconclusive point does not stop S03, it is written in kiem_ngoai_mien.csv, printed and kept in meta.yaml kiem_ngoai_mien"
  - "v8.6 section 3.1: the sweep grid is sweep_grid_n cut at the ceiling T of the gain in tools/gain_table.py (table {gain_table.TABLE_VERSION}; the factor names are the table's names): points above T are dropped; when a point was dropped, a remaining point within the relative error 1e-9 of T becomes T, else T is added, to the original grid when T < b and to the extension when T > b; sweep.csv tran is true at value == T; an added T has luoi log (original grid) or noi (extension); a, b, r, K keep their definitions; b_n is the largest point after the cut (omega_o has no extension: its b_n is the end b of its grid); H_nguon 'trần bảng chặn' when H = T is the end of the cut original grid (first over 'hết lưới'); a firmware default above its ceiling stops S03; a default the cut made T (within 1e-9 below T) is run at the grid point T when both are the same float32: that point stands for the default run (is_default true, luoi mac_dinh, tran true, the level rule starts from it; levels.csv default keeps the firmware value); another float32 stops S03"
  - "v8.6 section 5.2: tv_duty threshold 0.4 s^-1 absolute for every speed, mu, worst and range; y3_reference is only reported"
  - "v8.6 sections 3.1b item 4, 3.2: before the main design the float32 of every gain of every design point is also compared with the gain table (0 accepted where the lower bound is 0), a failure stops S03; a design point with omega_c + 2 omega_o >= 200 on an axis (binary64 sum of the float32 values) runs like any point and has ngoai_tam_giac true in sealed/design.csv, runs.csv, runs_test.csv (PID always false); the number of such points per controller is in sealed/notes.json (centre, outer, inner, axial) and sealed_test/notes.json (test points)"
  - "v8.5/v8.6 section 3.1b: the screening step is fwsim/s03_sangloc.py (run_step); it runs after the sweep of every controller: S0, then S1, S2, S2b with their lock files, its determinism check, then sealed/levels_sangloc.csv and sealed/design.csv, then the main design of every controller, sealed/ is closed, then the test set"
  - "v8.5 3.1b item 13: from the end of the sweep until sealed/ is closed every stop reason goes to sealed/stop.json (with the phase), every print, warning and traceback of those phases to sealed/diag.txt; the console prints one line when S03 ends; status says only 'stopped after the sweep'; exit code 0 or 2 as before"
  - "the test-points file and the screening numbers file are read once at the start, before any run; a failed check of these committed input files is a public stop before the sweep (no result exists yet); S03 v8 read the test file after the sweep"
  - "meta.yaml is written three times: at the start, before the first run (the plan of section 3.1b - m_min, T_max, the formula of N, the SHA-256 of the numbers file, 8.9 - is public from the start), again before S0 runs, and when S03 ends; the public sweep files are written when the sweep ends"
  - "--trial (mu 0.5, one level): section 3.1b needs three mu levels and does not run; sealed/levels_sangloc.csv holds the sweep levels with nguon_mien du_phong and sealed/sangloc/ket_qua.json says khong_chay_sang_loc with that reason"
  - "--sangloc-thu (S03 section 8 condition 11): tool check of section 3.1b with the three measured mu levels, its own numbers file (another name and content than fwsim/s03_screen_u.csv, seeds 2026112105..09 when made by make_s03_screen_u.py --thu), S0 on the first n0 rows and N given by --sangloc-thu-n; the formula of N is computed and kept in sealed/sangloc_thu/so_dem.json but does not set N; trial: true"
  - "the main design decodes its points from sealed/levels_sangloc.csv read back from disk (S03 3.2 'Nguon muc'); levels.csv keeps the sweep levels"
  - "kep_frac_vx, kep_frac_vy, kep_frac_wz (runs.csv, runs_test.csv; S03 2.1, 7, 11 item 6 (c)): fwsim.kep_indicator on the O1 window up to the t_end of the run, ticks at the PID integral clamp over ticks with the motor on, one column per axis as section 2.1 asks; empty for LADRC and for a run_error row only; a PID run on a DLL without the kep channel, or a failed self-check of the indicator, stops S03 (Codex C3-48)"
  - "screening files: evals_<set>.csv has the columns of the S04 evals.csv after controller, set, row (ga_seed, gen, cand_id empty); the ba_mu record of a gain set not feasible at the middle mu has no score numbers (step b not run); S0, S1, S2, S2b run both controllers in one worker pool, so sealed/notes.json has one common total per phase (controller all)"
  - "sealed/notes.json pha: no entry with the label 'bo' (this script cannot resume a run; a new run needs a new folder)"
  - "the --sangloc-thu numbers file must be committed and is checked bit for bit against the seeds 2026112105..09; every numbers file is regenerated in memory by (mode, r, t) and compared bit for bit and as the canonical text before the sweep (S03 3.1b item 3, 8.11; Codex C3-36)"
  - "a run with the screening step needs both controllers PID and ADRC, each once, and three finite, distinct mu levels (exact binary64 comparison); --trial keeps its own rule (S03 2.2, 3.1b items 1, 5, 10; Codex C3-37)"
  - "when a stop is pending, stop.json is written first. A failure while closing sealed/ becomes a 'dong_niem_phong' stop. An atomic update of meta.yaml keeps the previous copy if the update fails (Codex C3-38)"
  - "sealed/notes.json: every phase entry exists from the phase start with trang_thai, moc_bat_dau, moc_ket_thuc and n_luot_tong (Codex C3-39, D2A-2); t_S0 of sangloc_thu/so_dem.json is the monotonic time of the whole S0 phase; elapsed_s_main_design is the main design only and elapsed_s_sang_loc the screening step (the field rsm_compare.shared_cost_rows reads keeps its name)"
  - "screening workers checkpoint each evaluation attempt. A crash preserves completed evaluations, triggers a fresh-process diagnosis for an interrupted first attempt and marks skipped evaluations not_run. The wait limits are 7200 s per job, 600 s per crash diagnosis and 30 s to end a worker tree. sealed/ closes after the trees have stopped (S03 3.1b item 5, Codex C3-50 and C3-51)"
  - "CODE may read sealed/sangloc_thu/loi_S0.json before campaign A ends. It holds unclassifiable S0 gain-set counts per controller and mode and fixed error codes (S03 3.1b items 5 and 13, Codex C3-52)"
  - "v8.2: the public sweep carries the G columns too (Liam 14/11 CODE_FEAS 'Co, mot cach cham'); G3-G5 are not responses of section 5"
elapsed_s: {sweep_s}
file_sha256: {json.dumps(files)}
files:
  - sweep.csv: "one line per sweep run (section 3.1) and default gains at the other mu levels; v8.3: luoi = log | mac_dinh | 0 | noi (the extension); v8.6: tran = true at the point of value T, the gain-table ceiling"
  - sweep_counts.csv: "section 5.3 for the sweep: runs not completed, inconclusive, saturated per factor, speed, mu"
  - levels.csv: "L, H, centre, scale per factor (v8.4: on the original grid); v8.3: a, b (ends of the 11 log points), b_n (v8.6: the largest point of the grid after the cut at the gain-table ceiling, T when the grid is cut, else the end of the extension; for omega_o, which has no extension, the end b of its grid; from v8.5 the upper end of campaign A's domain and of the reference box W), L_nguon, H_nguon (source of each level, S03 v8.4 section 3.1; v8.6: 'trần bảng chặn' when H = T ends the cut original grid)"
  - kiem_ngoai_mien.csv: "S03 v8.4 section 3.1: one row (dong diem) per original or extension point outside [L, H]: luoi, level compared (can L or H, can_value), ket_qua (không cờ / chưa kết luận / cờ), buoc, bao_rieng, per speed: in the gate, y1, status, e_perp_max, tv_duty of the point and of the level, Y3 threshold; one total row (dong tong) per controller: đã kiểm đủ / kiểm dở / không kiểm and counts"
  - logs_sweep/: "npz of the sweep runs not completed"
  - sealed/: "SEALED (L56): levels_sangloc.csv (the screening or fallback levels, section 3.2 'Nguồn mức'), design.csv (371 rows per controller: centre, outer box, inner box, axial points, test points; v8.6: ngoai_tam_giac), sangloc/, sangloc_thu/ (the screening step of section 3.1b), runs.csv (section 4 responses; centre, outer, inner, axial; kep_frac_vx/vy/wz columns at the end), effects.csv (every term, per box, response, speed, mu / worst / range; outer and inner only), curvature.csv (5.4), y2_versions.csv (5.1), fail_counts.csv (5.3), determinism_check.csv (3.4), notes.json (v8.6: ngoai_tam_giac_points per controller, pha), stop.json, diag.txt (prints, warnings, tracebacks of the sealed phases), manifest.json, logs/"
  - sealed_test/: "SEALED separately (S03 v8 section 7): runs_test.csv, logs/, notes.json (v8.6: ngoai_tam_giac_points per controller, test points), stop.json, diag.txt (warnings, prints, traceback of the test phase), manifest.json; opened only after the RSM coefficients at the middle mu are saved and committed"
"""
        tmp = os.path.join(a.out, '.meta.yaml.tmp')
        f = open(tmp, 'x', encoding='utf-8')
        try:
            with f:
                f.write(text)
            os.replace(tmp, os.path.join(a.out, 'meta.yaml'))
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise

    # ── step 1 (S03 v8.5 section 7 'Thứ tự chạy khóa', 3.1b item 13): the committed input files
    # are read ONCE, before any run; a failed check is a public stop BEFORE the sweep (no result
    # exists yet): no firmware call is made for it.
    try:
        if 'PID' in a.controllers and not getattr(fw, 'kep_cols', 0):
            raise S03Stop('the DLL has no kep channel (fwsim_kep_cols = 0): S03 v8.6 2.1 and '
                          '11 item 6 (c) ask the PID clamp indicator of every run; a DLL without '
                          'the clamp export cannot run S03, rebuild it (Codex C3-48)')
        test_pts, tp_info = load_test_points(a.test_points, trial=is_trial)   # section 8 condition 7
        if screening is not None:
            screen, screen_info = sl.load_screen_u(a.screen_u or sl.SCREEN_U, trial=False,
                                                   r=a.screen_r, che_do=screening)   # 8.9, C3-36
    except S03Stop as exc:
        stop_before = f'before the sweep: {exc}'
        tp_info = getattr(exc, 'tp_info', tp_info)
        screen_info = getattr(exc, 'screen_info', screen_info)
        print(f'[STOP] {stop_before}', flush=True)

    if stop_before is None:
        # S03 section 8 condition 9: the SHA-256 of the numbers file is in the public meta.yaml
        # from the start of the run, before the first firmware run
        try:
            write_meta(final=False, running='running: started; the sweep is not finished')
        except OSError:
            stop_before = 'writing meta.yaml failed'
            print(f'S03 finished -> {a.out}; status: {final_status()}')
            return 2
    if stop_before is None:
        try:
            # 3.1 (i) first, for every controller, so S03 stops before any sweep (S3C-9)
            for ctrl in a.controllers:
                dflts[ctrl] = defaults(fw, ctrl)
                for v in SPEEDS:
                    row, log = checked(ctrl, dflts[ctrl], v, mu_mid,
                                       sweep=('(default check)', '', True, 'mac_dinh', False))
                    # recorded and logged before the stop test (S3D-4); public like the sweep
                    sweep_rows.append(dict(controller=ctrl, factor='(default check)', value='',
                                           is_default=True, luoi='mac_dinh', tran=False, v=v,
                                           mu=mu_mid, **row))
                    if row['completed'] != 'true':
                        save_log(os.path.join(a.out, 'logs_sweep'), f'{ctrl}_default_check_v{v}', log)
                    if v in gate and row['completed'] != 'true':
                        raise S03Stop(f'{ctrl}: default gains do not complete at gate speed {v} (y1 '
                                      f'{row["completed"]}, S03 3.1 (i))', case='(i)')
            # S03 v8.4: every grid is known before the first sweep run, so a sweep that stops part
            # way can count the points it did not check (kiem_ngoai_mien.csv 'kiểm dở')
            for ctrl in a.controllers:
                lo_b, hi_b = tune_ga.BOUNDS[ctrl]
                for j, name in enumerate(FACTORS[ctrl]):
                    grids[(ctrl, name)] = sweep_grid_cut(ctrl, name, dflts[ctrl][name], lo_b[j], hi_b[j])  # v8.6: the cut
                    grid_info[(ctrl, name)] = grids[(ctrl, name)][1]
            sweep_planned = sum(3 + 3 * sum(len(grids[(ctrl, n)][0]) for n in FACTORS[ctrl])
                                + 3 * (len(mu_levels) - 1) for ctrl in a.controllers)
            # 3.1 sweep and levels, every controller (S3C-9: a stop ends S03)
            for ctrl in a.controllers:
                dflt = dflts[ctrl]
                levels[ctrl] = {}
                for j, name in enumerate(FACTORS[ctrl]):
                    grid, ginfo = grids[(ctrl, name)]
                    base = base_grid(grid, ginfo)              # S03 v8.4: the level rule runs on it only
                    d_node = default_node(grid, ginfo, dflt[name])   # C3-4: the default, or T standing for it
                    ok = []
                    for x in grid:
                        g = dict(dflt, **{name: x})
                        done = True
                        for v in SPEEDS:
                            row, log = checked(ctrl, g, v, mu_mid, sweep=(name, x, x == d_node,
                                                                          grid_label(x, d_node, ginfo['noi']),
                                                                          x == ginfo['T']))
                            sweep_rows.append(dict(controller=ctrl, factor=name, value=x,
                                                   is_default=x == d_node,
                                                   luoi=grid_label(x, d_node, ginfo['noi']),
                                                   tran=(x == ginfo['T']), v=v, mu=mu_mid, **row))
                            if x == d_node:
                                y3_ref[(ctrl, v)] = row['tv_duty']
                            if row['completed'] != 'true':      # sweep is public: failed logs too (S3C-20)
                                save_log(os.path.join(a.out, 'logs_sweep'), f'{ctrl}_{name}_{x:.6g}_v{v}', log)
                            if v in gate:
                                done &= row['completed'] == 'true'
                        ok.append(done)
                        print(f'[sweep {ctrl}] {name} = {x:.4g}: completes at gate speeds = {done}', flush=True)
                    try:
                        L, H = levels_from_sweep(base, d_node, ok[:len(base)])    # S03 v8.4: original grid only
                    except S03Stop as exc:
                        raise S03Stop(f'{ctrl} {name}: {exc}', case=exc.case) from None
                    sc, centre = scale_of(L, H)
                    levels[ctrl][name] = (L, H, sc)
                    l_src, h_src = level_sources(base, L, H, ginfo, name)          # S03 v8.4
                    level_rows.append(dict(controller=ctrl, factor=name, L=L, H=H, centre=centre, scale=sc,
                                           default=dflt[name], rule_note='longest completing run of the '
                                           'original grid with the default, gate speeds '
                                           + ' '.join(map(str, gate)),
                                           a=ginfo['a'], b=ginfo['b'],
                                           b_n='' if ginfo['b_n'] is None else ginfo['b_n'],
                                           L_nguon=l_src, H_nguon=h_src))
                    print(f'[levels {ctrl}] {name}: L={L:.4g} H={H:.4g} scale={sc}', flush=True)
                same = [n for n, (L, H, _) in levels[ctrl].items() if L == H]
                if same:
                    raise S03Stop(f'{ctrl}: L = H for {same} (S03 3.1 (ii)): Liam decides, no main design',
                                  case='(ii)')
                for mu in mu_levels:    # default gains at low / high mu too (5.2: reported, not the base)
                    if mu == mu_mid:
                        continue
                    for v in SPEEDS:
                        row, log = checked(ctrl, dflt, v, mu, sweep=('(default)', '', True, 'mac_dinh', False))
                        sweep_rows.append(dict(controller=ctrl, factor='(default)', value='',
                                               is_default=True, luoi='mac_dinh', tran=False, v=v, mu=mu, **row))
                        if row['completed'] != 'true':                      # S3C-20
                            save_log(os.path.join(a.out, 'logs_sweep'), f'{ctrl}_default_mu{mu}_v{v}', log)
            t_sweep_end = time.time()
        except S03Stop as exc:
            stop_public = str(exc)                          # sweep phase: public (S03 3.1 is not sealed)
            stop_case = exc.case
            print(f'[STOP] {exc}', flush=True)

    # S03 v8.4 section 3.1: the out-of-range check, from the public sweep only, before the main design
    if stop_before is not None:
        kiem_rows, kiem_sum = [], {}                 # stopped before the sweep: nothing to check
    else:
        kiem_rows, kiem_sum = out_of_range_check(sweep_rows, levels, grids, dflts, a.controllers,
                                                 gate, mu_mid, y3_ref, stop_case)
    for ctrl, s in kiem_sum.items():
        print(f'[out-of-range check {ctrl}] {s["ket_qua"]}: {s["n_diem_ngoai"]} points outside, cờ {s["n_co"]}, '
              f'chưa kết luận {s["n_chua_ket_luan"]}, không cờ {s["n_khong_co"]}, not checked {s["n_chua_kiem"]}',
              flush=True)
        if s['bao_rieng']:
            print(f'[out-of-range check {ctrl}] e_perp_max lower by >= 0.01 m at one speed, not flagged: '
                  f'{" ".join(s["bao_rieng"])}', flush=True)
    flagged = {c: s['co'] for c, s in kiem_sum.items() if s['co']}
    if flagged:                                     # S03 v8.5 3.1 (K6): reported and kept, S03 goes on
        print('[out-of-range check] flagged points (S03 v8.5 3.1, K6: written in kiem_ngoai_mien.csv and '
              'meta.yaml, S03 goes on): ' +
              '; '.join(f'{c}: {" ".join(p)}' for c, p in flagged.items()), flush=True)
    if any(s['chua_ket_luan'] for s in kiem_sum.values()):
        print('WARNING: out-of-range check has inconclusive points (S03 v8.5 3.1: reported, S03 goes on): ' +
              '; '.join(f'{c}: {" ".join(s["chua_ket_luan"])}' for c, s in kiem_sum.items() if s['chua_ket_luan']),
              flush=True)

    # step 3 (v8.5 section 7): the public files of the sweep are written when the sweep ends;
    # design.csv is not public any more (the design points are sealed with the screening)
    for name, rows in (('sweep.csv', sweep_rows), ('sweep_counts.csv', sweep_counts(sweep_rows)),
                       ('levels.csv', level_rows), ('kiem_ngoai_mien.csv', kiem_rows)):
        p = os.path.join(a.out, name)
        cols = (LEVEL_COLS if name == 'levels.csv' else     # S83-1: the header even when S03 stops early
                KIEM_COLS if name == 'kiem_ngoai_mien.csv' else None)   # S03 v8.4: fixed header too
        if name == 'sweep.csv':     # S3G-2, S3G2-3: one fixed header (identifiers, run columns), other keys after
            cols = list(dict.fromkeys(list(SWEEP_ID_COLS) + RUN_COLS[5:] + [k for r in rows for k in r]))
        write_csv(p, rows, cols)
        files[name] = sha256_file(p)

    def pha_open(name, ctrl, planned):
        """Append the live notes entry of a main-design phase at its start (C3-39, D2A-2):
        trang_thai 'dung_giua_pha', moc_bat_dau; returns the live record pha_close fills."""
        e = dict(pha=name, controller=ctrl, n_luot_ke_hoach=planned, n_luot_thuc=0,
                 n_chay_lai_chan_doan=0, n_luot_tong=0, giay_may=None, so_tien_trinh=1,
                 trang_thai='dung_giua_pha', moc_bat_dau=sl._stamp(), moc_ket_thuc=None)
        pha_notes.append(e)
        return dict(e=e, n=len(all_runs), rr=len(rerun_notes), t=_mono(), n_thuc=None)

    def pha_close(live, state='xong'):
        """Close a phase entry of pha_open: run/diagnosis counts from the all_runs /
        rerun_notes deltas (or live['n_thuc'] when set), giay_may, moc_ket_thuc."""
        e = live['e']
        n1 = len(all_runs) - live['n'] if live['n_thuc'] is None else live['n_thuc']
        n2 = len(rerun_notes) - live['rr']
        e.update(n_luot_thuc=n1, n_chay_lai_chan_doan=n2, n_luot_tong=n1 + n2,
                 giay_may=_mono() - live['t'], trang_thai=state, moc_ket_thuc=sl._stamp())

    # step 4: the SEALED WINDOW (v8.5 section 7 'Thứ tự chạy khóa', 3.1b item 13): from the end of
    # the sweep until sealed/ is closed nothing of these phases prints to the console; every stop
    # reason goes to sealed/stop.json with its phase, prints, warnings and tracebacks to
    # sealed/diag.txt.
    diag = io.StringIO()
    caught = ()
    if stop_before is None and stop_public is None:
        pha_name = 'meta_yaml'
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(diag), \
                contextlib.redirect_stderr(diag):
            try:
                # 4a: the public meta.yaml with the PLAN of section 3.1b on disk before S0
                # (item 10 last item, 8.10)
                write_meta(final=False, running='running: the sweep is done; the sealed phases '
                                                '(screening, main design) and the test set are '
                                                'not finished')
                # 4b 'sang_loc': the screening step of section 3.1b (fwsim/s03_sangloc.py)
                pha_name = 'sang_loc'
                W = {ctrl: {} for ctrl in a.controllers}
                for ctrl in a.controllers:
                    lo_b, hi_b = tune_ga.BOUNDS[ctrl]
                    for j, name in enumerate(FACTORS[ctrl]):
                        W[ctrl][name] = w_nodes(ctrl, name, dflts[ctrl][name], lo_b[j], hi_b[j])
                quet = {ctrl: {n: lv[:2] for n, lv in levels[ctrl].items()}
                        for ctrl in a.controllers}
                t_sl0 = _mono()
                try:
                    if screening is not None:
                        tool = dict(n0=a.sangloc_thu_n0, n=a.sangloc_thu_n) \
                            if screening == 'thu' else None
                        res = sl.run_step(sealed, a.dll, a.controllers, gate, mu_levels, W, quet,
                                          screen, a.jobs, tool_check=tool)
                    else:
                        res = sl.skip_step(sealed, a.controllers,
                                           'khong_chay_sang_loc: --trial with one mu level (S03 2.2); '
                                           'section 3.1b needs three mu levels')
                finally:
                    t_sl1 = _mono()
                pha_notes += res['pha']
                # 4c 'mien': sealed/levels_sangloc.csv — the accepted box or the sweep levels
                pha_name = 'mien'
                lv_sl_path = os.path.join(sealed, 'levels_sangloc.csv')
                sl_rows = []
                for ctrl in a.controllers:
                    sl_rows += sl.levels_rows(ctrl, res['mien'][ctrl], W, grid_info, dflts[ctrl],
                                              [r for r in level_rows if r['controller'] == ctrl],
                                              res['nhanh'][ctrl], res['ly_do'][ctrl], gate)
                write_csv(lv_sl_path, sl_rows, sl.LEVEL_SL_COLS)
                # 4d 'thiet_ke': the points decode from the FILE (S03 3.2 'Nguồn mức'); every point
                # of every controller is sealed but on disk before any main-design run; v8: 371
                # rows per controller incl. axial and test points
                pha_name = 'thiet_ke'
                for ctrl in a.controllers:
                    lv = sl.read_levels(lv_sl_path, ctrl)
                    for name in FACTORS[ctrl]:
                        if name not in lv or not lv[name][0] < lv[name][1]:
                            raise S03Stop(f'{ctrl} {name}: sealed/levels_sangloc.csv holds no level '
                                          f'with L < H (S03 3.2 Nguồn mức)')

                    def gains_at(codes, lv=lv, ctrl=ctrl):
                        return {n: decode(c, *lv[n]) for n, c in zip(FACTORS[ctrl], codes)}
                    centre_codes = (0.0,) * 7
                    outer, inner, axial = design_points(1.0, 1), design_points(0.5, 129), axial_points()
                    points = ([(0, 'centre', centre_codes)] + [(p, 'outer', c) for p, c in outer] +
                              [(p, 'inner', c) for p, c in inner] + [(p, 'axial', c) for p, c in axial] +
                              [(p, 'test', c) for p, c in test_pts])
                    check_bounds(ctrl, lv, points, gains_at)             # section 11 item 4 (i), (ii)
                    check_table(ctrl, points, gains_at)                  # v8.6 section 3.1b item 4: the gain table
                    for pid, box, codes in points:
                        design_rows.append(dict(controller=ctrl, point_id=pid, box=box,
                                                **{f'X{k + 1}': c for k, c in enumerate(codes)},
                                                **{f'{n}_value': v_ for n, v_ in gains_at(codes).items()},
                                                ngoai_tam_giac=outside_triangle(ctrl, gains_at(codes))))
                    plan[ctrl] = (gains_at, centre_codes, outer, inner, axial, test_pts)
                check_design_map(design_rows, test_pts)                  # section 11 item 4: id - box - codes
                write_csv(os.path.join(sealed, 'design.csv'), design_rows,
                          cols=design_cols(design_rows))                 # before the first main run
                # 4e 'thiet_ke_chinh': the main design of every controller
                pha_name = 'thiet_ke_chinh'
                t_md0 = _mono()
                for ctrl in a.controllers:
                    gains_at, centre_codes, outer, inner, axial, _test = plan[ctrl]
                    runs, keep_logs = {}, {}
                    rng = np.random.default_rng(DET_SEED)
                    det_ids = [0] + sorted(int(x) for x in rng.choice(np.arange(1, 129), size=3,
                                                                            replace=False))
                    todo = [(0, 'centre', centre_codes)] + [(pid, 'outer', codes)
                                                          for pid, codes in outer]

                    def run_points(points):
                        # L56: nothing about the main design goes to the console or outside sealed/
                        for pid, box, codes in points:
                            g = gains_at(codes)
                            for mu in mu_levels:
                                for v in SPEEDS:
                                    row, log = checked(ctrl, g, v, mu, ids=(pid, box))
                                    runs[(pid, mu, v)] = row
                                    all_runs.append(dict(controller=ctrl, point_id=pid, box=box,
                                                         mu=mu, v=v,
                                                         ngoai_tam_giac=outside_triangle(ctrl, g),
                                                         **row, **kep_frac_fields(ctrl, row, log)))
                                    name = f'{ctrl}_p{pid}_mu{mu}_v{v}'
                                    # v8: the determinism points are the centre and 3 OUTER corners,
                                    # chosen by box, not only by point_id (section 11 item 4)
                                    if pid in det_ids and box in ('centre', 'outer'):
                                        keep_logs[(pid, mu, v)] = (g, box, row, log_digest(log))
                                        save_log_csv(sealed_logs, name, log)      # S03 3.4: log.csv
                                    data = npz_bytes(log)    # S3E-1: compressed for every run, same work
                                    if box == 'centre' or row['completed'] != 'true':
                                        save_log(sealed_logs, name, log, data)
                    pha_live = pha_open('tam_va_hai_hop', ctrl, 257 * len(mu_levels) * len(SPEEDS))
                    run_points(todo)
                    # L41 (Liam 06/11, 'Chot B'): the inner box always runs, so the run time does not
                    # tell whether the rule of S03 3.2 is on; it enters the analysis only when the
                    # rule is on
                    run_points([(p, 'inner', c) for p, c in inner])
                    pha_close(pha_live); pha_live = None
                    # v8 (L67): axial points always run too, 3 mu x 3 speeds, sealed; S03 does not
                    # analyse them (section 5 and the rule of 3.2 use the outer / inner / centre runs)
                    pha_live = pha_open('truc', ctrl, 14 * len(mu_levels) * len(SPEEDS))
                    run_points([(p, 'axial', c) for p, c in axial])
                    pha_close(pha_live); pha_live = None
                    corner_ids = [p for p, _ in outer]
                    need_inner = [v for v in SPEEDS
                                  if sum(runs[(i, mu_mid, v)]['completed'] == 'false'
                                         for i in corner_ids) > 64]
                    inner_rule[ctrl] = need_inner
                    notes = {('outer', v): 'most corners not completed at the middle mu (S03 3.2): '
                             'reported with this warning; see the inner box' for v in need_inner}
                    no_concl[ctrl] = [v for v in need_inner
                                      if sum(runs[(p, mu_mid, v)]['completed'] == 'false'
                                             for p, _ in inner) > 64]

                    # 3.4 determinism in a new process (job files under sealed/, S3C-2)
                    pha_live = pha_open('kiem_tat_dinh_3_4', ctrl,
                                        4 * len(mu_levels) * len(SPEEDS))
                    jobs = sorted(keep_logs)
                    det_res = run_in_new_process(
                        a.dll, [(ctrl, keep_logs[j][0], j[2], j[1]) for j in jobs], work_sealed)
                    pha_live['n_thuc'] = len(det_res)
                    for j, r in zip(jobs, det_res):
                        g, box, row0, dig0 = keep_logs[j]
                        ids = (ctrl, j[0], box, j[1], j[2])
                        det_rows.append(dict(controller=ctrl, point_id=j[0], mu=j[1], v=j[2],
                                             log_arrays_equal=r['digest'] == dig0,
                                             runs_fields_equal=row_text(r['row'], ids)
                                             == row_text(row0, ids),
                                             n_arrays=len(dig0)))
                    pha_close(pha_live); pha_live = None
                    if not all(d['log_arrays_equal'] and d['runs_fields_equal'] for d in det_rows):
                        raise S03Stop('determinism check failed (S03 3.4): stop, tell CODE and THESIS')

                    # 5 analysis: both boxes at every speed always (S3E-1: the same work whether
                    # the rule of 3.2 is on or not); kept: the inner box only when the rule is on,
                    # no effects / curvature / Y2 versions at a speed where the inner box is also
                    # mostly not completed
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
                pha_notes += getattr(exc, 'pha_notes', [])        # run_step's phase notes (C3-39)
                stop_sealed = dict(pha=pha_name, ly_do=str(exc))
                worker_tree_stopped = not isinstance(exc, sl.WorkerTreeStop)
            except Exception as exc:            # noqa: BLE001 (item 13: any failure stays sealed)
                pha_notes += getattr(exc, 'pha_notes', [])
                stop_sealed = dict(pha=pha_name, ly_do=f'{type(exc).__name__}: {exc}')
                traceback.print_exc(file=diag)
            if pha_live is not None:
                pha_close(pha_live, 'dung_giua_pha')
            if t_md0 is not None:
                t_md1 = _mono()

    t_end = time.time()
    # C3-38 (v8.5 3.1b item 13): the closing of sealed/ — stop.json first when a stop is pending,
    # the work folders, the six CSVs, notes.json, diag.txt, the manifest — runs inside ONE guard
    # with prints, warnings and tracebacks in the same diag StringIO; an output error becomes a
    # 'dong_niem_phong' stop and stop.json / diag.txt / manifest.json are rewritten best effort
    with warnings.catch_warnings(record=True) as caught_close, contextlib.redirect_stdout(diag), \
            contextlib.redirect_stderr(diag):

        def _manifest():
            """sealed/manifest.json from the files ON DISK: the six CSVs, levels_sangloc.csv,
            design.csv, notes.json, diag.txt, stop.json (each when it exists) and every file
            under sangloc/ and sangloc_thu/ (relative name with '/')."""
            if not worker_tree_stopped:
                return
            sfiles = {}
            for name in ('runs.csv', 'effects.csv', 'curvature.csv', 'y2_versions.csv',
                         'fail_counts.csv', 'determinism_check.csv', 'levels_sangloc.csv',
                         'design.csv', 'notes.json', 'diag.txt', 'stop.json'):
                p = os.path.join(sealed, name)
                if os.path.isfile(p):
                    sfiles[name] = sha256_file(p)
            for sub in ('sangloc', 'sangloc_thu'):      # every file the screening step wrote
                root_dir = os.path.join(sealed, sub)
                if os.path.isdir(root_dir):
                    for root, _dirs, fnames in os.walk(root_dir):
                        for fn in sorted(fnames):
                            p = os.path.join(root, fn)
                            sfiles[os.path.relpath(p, sealed).replace(os.sep, '/')] = sha256_file(p)
            with open(os.path.join(sealed, 'manifest.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(files=sfiles), f, indent=1)

        def _diag_text():
            """sealed/diag.txt: every print, warning and traceback of the sealed phases (item 13);
            empty when nothing was captured."""
            return diag.getvalue() + ''.join(f'warning: {w_.category.__name__}: {w_.message} '
                                             f'({w_.filename}:{w_.lineno})\n'
                                             for w_ in itertools.chain(caught, caught_close))
        try:
            if stop_sealed:
                with open(os.path.join(sealed, 'stop.json'), 'w', encoding='utf-8') as f:
                    json.dump(dict(stop=stop_sealed['ly_do'], pha=stop_sealed['pha']), f, indent=1)
            # S3V8-1: the empty work folders of the sweep, of the main design and of the screening
            # worker pools are removed BEFORE sealed/ is written and closed; after the manifest
            # only sealed_test/ changes
            work_dirs = [work_pub, work_sealed]
            if worker_tree_stopped:
                work_dirs += [os.path.join(sealed, sub, '_work') for sub in ('sangloc', 'sangloc_thu')]
            for d in work_dirs:
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
            # sealed outputs, with their own manifest (no hash of a sealed file outside sealed/, S3C-1)
            for name, rows, cols in (('runs.csv', all_runs, RUNS_FILE_COLS),
                                     ('effects.csv', eff_rows, None),
                                     ('curvature.csv', curv_rows, None),
                                     ('y2_versions.csv', ver_rows, None),
                                     ('fail_counts.csv', fail_rows, None),
                                     ('determinism_check.csv', det_rows, None)):
                write_csv(os.path.join(sealed, name), rows, cols)
            quet_entry = dict(pha='quet', controller='all', n_luot_ke_hoach=sweep_planned,
                              n_luot_thuc=len(sweep_rows),
                              n_chay_lai_chan_doan=sum(1 for n in rerun_notes if n['point'] is None),
                              giay_may=(t_sweep_end or t_end) - t0, so_tien_trinh=1,
                              n_luot_tong=(len(sweep_rows)
                                           + sum(1 for n in rerun_notes if n['point'] is None)),
                              trang_thai='xong' if t_sweep_end else 'dung_giua_pha',
                              moc_bat_dau=sl._stamp(t0), moc_ket_thuc=sl._stamp(t_sweep_end or t_end))
            kep_note = ('PID: fwsim.kep_indicator on the O1 window up to the t_end of the run '
                        '(ticks at the PID integral clamp over ticks with the motor on); empty '
                        'only for ADRC and run_error rows: a PID run without the channel or with '
                        'a failed self-check stops S03 (Codex C3-48)'
                        if bool(getattr(fw, 'kep_cols', 0))
                        else 'a DLL without the kep channel cannot run S03 with PID (Codex C3-48)')
            with open(os.path.join(sealed, 'notes.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(inner_rule_speeds=inner_rule, no_conclusion_speeds=no_concl,
                               stop=stop_sealed['ly_do'] if stop_sealed else None,
                               run_error_reruns=[n for n in rerun_notes if n['point'] is not None],
                               ngoai_tam_giac_points=({c: sum(1 for r in design_rows
                                                             if r['controller'] == c
                                                             and r['box'] in ('centre', 'outer', 'inner',
                                                                            'axial')
                                                             and r[TRIANGLE_COL])
                                                      for c in a.controllers} if design_rows else {}),
                               elapsed_s_total=round(t_end - t0),
                               elapsed_s_main_design=(round(t_md1 - t_md0)
                                                      if t_md0 is not None else None),
                               elapsed_s_sang_loc=(round(t_sl1 - t_sl0)
                                                   if t_sl0 is not None else None),
                               pha=[quet_entry] + pha_notes,   # S03 v8.5 section 7: one entry per phase
                               kep_frac=kep_note),
                          f, indent=1, default=str)
            with open(os.path.join(sealed, 'diag.txt'), 'w', encoding='utf-8') as f:
                f.write(_diag_text())
            _manifest()
        except Exception as exc:        # noqa: BLE001 (an output error is a stop, kept sealed)
            out_error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc(file=diag)
            if stop_sealed is None:
                stop_sealed = dict(pha='dong_niem_phong',
                                   ly_do=f'writing the sealed outputs failed: {out_error}')
            for name, text in (('stop.json', lambda: json.dumps(
                    dict(stop=stop_sealed['ly_do'], pha=stop_sealed['pha'],
                         output_error=out_error), indent=1)),
                               ('diag.txt', _diag_text)):
                try:                    # best effort; a failure here leaves only the public line
                    with open(os.path.join(sealed, name), 'w', encoding='utf-8') as f_:
                        f_.write(text())
                except Exception:       # noqa: BLE001,S110
                    pass
            try:
                _manifest()
            except Exception:           # noqa: BLE001,S110
                pass
    # The parent has finished writing sealed/ (S03 v8 section 7).

    # v8 (L67, S38-8): the test set of both controllers runs only now, after sealed/ is written,
    # so sealed/ does not depend on it; everything about it stays in sealed_test/
    sealed_test = os.path.join(a.out, 'sealed_test')
    work_test = os.path.join(sealed_test, '_work')
    if stop_before is None and stop_public is None and stop_sealed is None:
        test_logs = os.path.join(sealed_test, 'logs')
        t_test0 = time.time()
        # S3V8-2, R2-1: the whole phase (folder, runs, cleaning, every output of sealed_test/) is in
        # one guard; warnings, prints and tracebacks go to sealed_test/diag.txt, any exception is a
        # stop recorded in sealed_test/stop.json (written first); if even that cannot be written,
        # the console still only says that S03 stopped in the test set (the DLL does not print:
        # fwsim/shim/esp_log.h). R2-3: the warning policy in force is kept (no simplefilter): a
        # warning made an error stays an error (run_error in safe_run), the others are recorded.
        diag = io.StringIO()
        out_error = None
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(diag), \
                contextlib.redirect_stderr(diag):
            try:
                os.makedirs(sealed_test, exist_ok=True)
                for ctrl in a.controllers:
                    gains_at, test_pts = plan[ctrl][0], plan[ctrl][5]
                    for pid, codes in test_pts:
                        g = gains_at(codes)
                        for v in SPEEDS:
                            row, log = checked(ctrl, g, v, mu_mid, ids=(pid, 'test'), sink=test_rows,
                                               work=work_test, notes_to=test_notes)
                            test_rows.append(dict(controller=ctrl, point_id=pid, box='test', mu=mu_mid, v=v,
                                                  ngoai_tam_giac=outside_triangle(ctrl, g), **row,
                                                  **kep_frac_fields(ctrl, row, log)))
                            data = npz_bytes(log)             # same work for every run (S3E-1)
                            if row['completed'] != 'true':    # logs kept by the rule of section 7
                                save_log(test_logs, f'{ctrl}_p{pid}_mu{mu_mid}_v{v}', log, data)
            except S03Stop as exc:
                stop_test = str(exc)
            except Exception as exc:          # noqa: BLE001 (S3V8-2: any failure of the phase stays sealed)
                stop_test = f'{type(exc).__name__}: {exc}'
                traceback.print_exc(file=diag)
            test_s = round(time.time() - t_test0)

            def diag_text():
                return diag.getvalue() + ''.join(f'warning: {w_.category.__name__}: {w_.message} '
                                                 f'({w_.filename}:{w_.lineno})\n' for w_ in caught)

            def put(name, write):
                p_ = os.path.join(sealed_test, name)
                with open(p_, 'w', newline='' if name.endswith('.csv') else None, encoding='utf-8') as f_:
                    write(f_)
                test_files[name] = sha256_file(p_)

            def write_test_manifest():
                with open(os.path.join(sealed_test, 'manifest.json'), 'w', encoding='utf-8') as f_:
                    json.dump(dict(files=test_files), f_, indent=1)

            test_files = {}
            try:
                if os.path.isdir(work_test) and not os.listdir(work_test):    # before the manifest
                    os.rmdir(work_test)
                if stop_test:
                    put('stop.json', lambda f_: json.dump(dict(stop=stop_test), f_, indent=1))
                put('runs_test.csv', lambda f_: _write_csv_to(f_, test_rows, RUNS_FILE_COLS))
                put('notes.json', lambda f_: json.dump(dict(stop=stop_test, rows_run=len(test_rows),
                                                            run_error_reruns=test_notes, elapsed_s_test=test_s,
                                                            ngoai_tam_giac_points={
                                                                c: sum(1 for r in design_rows
                                                                       if r['controller'] == c
                                                                       and r['box'] == 'test'
                                                                       and r[TRIANGLE_COL])
                                                                for c in a.controllers}),
                                                       f_, indent=1, default=str))
                put('diag.txt', lambda f_: f_.write(diag_text()))
                write_test_manifest()
            except Exception as exc:          # noqa: BLE001 (R2-1: an output error is a stop, kept sealed)
                out_error = f'{type(exc).__name__}: {exc}'
                traceback.print_exc(file=diag)
                stop_test = stop_test or f'writing the test-set outputs failed: {out_error}'
                for name, text in (('stop.json', lambda: json.dumps(dict(stop=stop_test, output_error=out_error),
                                                                    indent=1)),
                                   ('diag.txt', diag_text)):
                    try:                      # best effort; a failure here leaves only the public stop line
                        with open(os.path.join(sealed_test, name), 'w', encoding='utf-8') as f_:
                            f_.write(text())
                    except Exception:         # noqa: BLE001,S110
                        pass
        # the test-set stop is not printed: after the sweep the console prints one line when
        # S03 ends (v8.5 3.1b item 13); the status string carries the phase
    try:
        write_meta(final=True)
    except Exception as exc:                    # keep the last complete meta and the fixed end line
        if not (stop_before or stop_public or stop_sealed or stop_test):
            stop_test = f'writing final meta.yaml failed: {type(exc).__name__}: {exc}'
            traceback.print_exc(file=diag)
            with contextlib.suppress(Exception):
                put('stop.json', lambda f_: json.dump(dict(stop=stop_test), f_, indent=1))
                put('diag.txt', lambda f_: f_.write(diag_text()))
                write_test_manifest()
        meta_state['status'] = final_status()
    print(f'S03 finished -> {a.out}; status: {meta_state["status"]}')
    return 0 if not (stop_before or stop_public or stop_sealed or stop_test
                     or meta_state['src_changed']) else 2


# S3G2-1: the evaluator files on disk, read at the end of this module's import, a few milliseconds after
# the evaluator modules were loaded above (the start and end snapshots of main() are compared with it).
# Not a hash of the loaded code: a checkout changed inside that window is not seen (S3G3-1, deferred).
# Child processes import again from disk and their sources are not locked; the determinism check of
# section 3.4 can see such a change only partly (S3G3-2).
EVAL_SHA_AT_IMPORT = evaluator_sha256()


if __name__ == '__main__':
    # run through the canonical module: the S03Stop raised by s04_ga / s03_sangloc (which
    # 'import s03_doe') is then the class main() catches, in both 'python fwsim/s03_doe.py' and
    # 'python -m fwsim.s03_doe'
    import s03_doe as _m
    if '--child' in sys.argv:
        _m.child_main(sys.argv[sys.argv.index('--child') + 1])
        sys.exit(0)
    if '--make-test-points' in sys.argv:          # S03 v8 section 7: once, then commit the file
        print(json.dumps(_m.make_test_points(), indent=1))
        sys.exit(0)
    sys.exit(_m.main())
