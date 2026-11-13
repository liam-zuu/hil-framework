"""
rq3_log — C9 of spec S05 §8.1: the RQ3 logs (S05 §6) and S05_summary.csv recomputed from them.
CODE session local_b2daa0f4, 11/11 (L79 order: panel stage E, then C9). Liam asked for C9
("Làm C9 ... làm luôn được chứ?", 11/11).

Three logs in 05_DATA/M7_dataset/ (S05 §6.1, §6.2), one row each:
  m7_tuning_log.csv      one run on one platform (SIM summary rows: one per controller and
                         account of one script run, S5-23 v2)
  m7_dev_effort_log.csv  work not tied to a run (hours reported by Liam, never filled in by Claude)
  m7_hw_failure_log.csv  one hardware failure
plus S05_freeze.csv (S05 v3.6 §4.7, Liam 11/11 at THESIS local_5e72913b): one row per campaign
at its freeze: frozen_at, stop_reason (fixed codes), gain_set_id, gain_file + its sha256.

Commands (the operator types them while working; every time stamp is the PC clock):
  migrate                      add the S05 v3.4 §6 columns to the three logs (values kept)
  begin  --session S --campaign C --platform HW --controller PID --task O1-F05 --speed 0.5
         --stage screen [--gain-set G03 --gains kp_xy=1.2,...]   open one run (iter_no = next of S)
  step   flash|setup|run|log|analysis [--wait]   end the current time segment, start this one
                                         (--wait: waiting for the machine, also in t_operator_wait_s)
  pause / resume                         other work in between: not counted (S05 §3.1)
  end    --outcome dat|khong_dat|chua_ket_luan|hong_he_do [--e-perp-max M] [--completed 0|1]
         [--fail-cause X] [--results-dir D] [--gate-json F]   close the run, append its row;
                                         F = scorer output with a gate_v3 block (fwsim/o1_gate.py,
                                         version gate_v3.2): checked and recomputed from q, lo, hi,
                                         n_undef (O1 3.5, SC3-1); those, the state per condition, U_t
                                         and the file's sha256 go to the gate columns (--results-dir
                                         required); the outcome is checked against the deciding
                                         conditions; no 'dat' while the gate is not complete (HIL run
                                         without the run_active edge mapping, RS-4)
                                         (12/11, tune stop gate v3; 13/11 v3.2; the logs need "migrate")
                                         S05 v3.15 (14/11): a HIL run whose edge mapping was refused (or
                                         scored without --esp-log: gate decide g1,g2) is a lost run,
                                         --outcome hong_he_do --fail-cause anh_xa_tu_choi; at most 3 reruns
                                         of one gain at one level, then that level is chua_ket_luan
  cancel / status                        drop the open run / show it. cancel only before the "run"
                                         step (S05 §3.3: every started run counts, EA-6); time of the
                                         steps already logged goes to m7_dev_effort_log.csv
  machine --campaign C --controller PID --script-run-id ID --results-dir D --n-sim-evals N
         --t-machine-s S --wall-start T --wall-end T [--stage script]   summary row of a fwsim
                                         script run (stage script by default, S05 v3.7 A-R2-4)
  hoc-start --campaign C --ghi-chu X / hoc-end [--ghi-chu X]   a hoc segment, time stamps written
                                         by the script (S05 §3.1 R-2, EA-7)
  effort --date D --start HH:MM --end HH:MM (--campaign C | --khoan K) [--nhan hoc] --ghi-chu X
                                         (a hoc row typed here is flagged [nhap_tay], EA-7)
  hwfail --date D --session S --campaign C --platform HW --what X --cause gain|... --fix-h H
                                         (refused while a run is open: close it first, EA-1)
  freeze --campaign C --stop-reason R [--stop-cause K] --gain-set C/G --gain-file F [--at T]
                                         (sha256 of F computed; gain set id prefixed by the campaign)
  summary [--gate-file F] [--out S05_summary.csv]   recompute the summary from the logs;
                                         F = the locked gate (figure-8 levels, locked at C8), one
                                         level per row under the header "level"; its sha256 is reported
  demo   --out DIR                       C9 dry run: made-up campaign in DIR, then summary

Rules taken from S05 v3.7 (approved 11/11, not locked): 3.1 (time types, wait, hoc), 3.2
(accounts, B-chung split), 3.3 (counts by platform, stages, T_sua in T), 4.1 (caps 10 h, 120 car
runs, counted before frozen_at; stop_cause), 4.6 (final_hil = account RQ2, N = 3), 4.7 (final
counts, gate_result), 4.8 (explore out of T, N, caps), 6.1-6.3 (gain_set_id per campaign).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.normpath(os.path.join(HERE, '..', '..', '..', '05_DATA', 'M7_dataset'))

TUNING, EFFORT, HWFAIL = 'm7_tuning_log.csv', 'm7_dev_effort_log.csv', 'm7_hw_failure_log.csv'
FREEZE, OPEN_RUN, SUMMARY = 'S05_freeze.csv', 'rq3_open_run.json', 'S05_summary.csv'
OPEN_HOC = 'rq3_open_hoc.json'

# S05 §6.1: columns kept (checked 08/11) + added
TUNING_OLD = ['session_id', 'date', 'platform', 'controller', 'trajectory', 'iter_no', 't_flash_s',
              't_setup_s', 't_run_s', 't_log_s', 't_analysis_s', 'gain_changed', 'old_value',
              'new_value', 'rmse_vx', 'rmse_vy', 'rmse_wz', 'stop_met', 'bug_id', 'notes', 'dut', 'buoc']
GAINS = ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'wc_xy', 'wo_xy', 'wc_wz', 'wo_wz', 'pos_kp', 'pos_ki', 'k_psi']
TUNING_NEW = (['run_uid', 'campaign', 'script_run_id', 'gain_set_id', 'speed_mps', 'stage', 'block_id',
               'replaces_run_uid', 'candidate_no', 'results_dir', 'completed', 'e_perp_max_m',
               'run_outcome', 'fail_cause', 'rule_version', 'fw_hash']
              + ['g_' + g for g in GAINS]
              + ['t_operator_wait_s', 't_machine_s', 'wall_start', 'wall_end', 'n_sim_evals'])
# tune stop gate v3 (dossier 00_OVERVIEW/2026-11-12_De_xuat_cong_dung_tune_bam_huong_van_toc_v3.md section 6,
# CG2-12; Liam 12/11 CODE local_87282b70 "làm ngay đi"): per run the measured maximum q, its bound U and
# the state of each condition, so stopping, the S05 4.4 ranking and gate_result can be recomputed.
# After Codex round 1 (Liam "Cho sửa 7 chỗ"): gate_decide = conditions that may decide (GV3-2),
# gate_json_sha256 = the scorer file the row came from, with results_dir required (GV3-6)
_FWSIM = os.path.normpath(os.path.join(HERE, '..', 'fwsim'))
if _FWSIM not in sys.path:
    sys.path.insert(0, _FWSIM)
import o1_gate                                                          # noqa: E402
GATE_CONDS = o1_gate.CONDS
# 13/11 gate v3.2 (SC-1, SC3-1; Liam CODE local_7f7b0ef4 "Duyệt gộp luôn lo/hi"): per condition G2-G5 the
# maxima over the frames of q, lo, hi (each field over its finite frames) and n_undef; G1 q is 'completed';
# gate_u_t = the U_t used (empty = not known: HIL run without the run_active edge mapping, RS-4)
GATE_FIELDS = ('q', 'lo', 'hi', 'n_undef', 'state')
OLD_GATE_U = tuple(f'g{i}_u' for i in range(1, 6))       # gate_v3_g1235 columns, dropped by migrate when empty
GATE_COLS = (['gate_version', 'gate_decide', 'gate_json_sha256', 'gate_u_t', 'g1_q', 'g1_state']
             + [f'{g}_{x}' for g in o1_gate.MEASURED for x in GATE_FIELDS])
TUNING_COLS = TUNING_OLD + TUNING_NEW + GATE_COLS
EFFORT_OLD = ['ngay', 'bat_dau', 'ket_thuc', 'gio', 'buoc', 'giai_doan', 'mot_lan_hay_thich_nghi', 'dut',
              'ghi_chu', 'chien_dich', 'lap_lai']
EFFORT_COLS = EFFORT_OLD + ['khoan', 'nhan']
HWFAIL_COLS = ['ngay', 'session_id', 'chien_dich', 'nen_tang', 'thu_hong', 'nguyen_nhan', 'gio_sua',
               'ghi_chu', 'dut', 'iter_no', 'buoc', 'lap_lai']
FREEZE_COLS = ['campaign', 'frozen_at', 'stop_reason', 'stop_cause', 'gain_set_id', 'gain_file', 'gain_sha256',
               'notes']                                                  # stop_cause: S05 v3.7 §6.3
STOP_REASONS = ('khoi_xac_nhan_dat', 'cham_tran_gio', 'cham_tran_xe', 'het_ung_vien', 's04_dung',
                'khong_du_luot_xac_nhan', 'khong_co_bo_danh_gia_du')          # S05 v3.6 §4.7
STOP_CAUSES = ('cham_tran_gio', 'cham_tran_xe', 'het_ung_vien', 's04_dung', 'khong_du_luot_xac_nhan')  # v3.7 §4.1
FINAL_N = 5                              # S05 §4.6: 5 valid runs per condition
FINAL_LOST_N = 3                         # S05 §4.6: stop a condition after 3 lost runs in a row
# S05 v3.15 §4.5 step 2, O1 v11.12 §3.4 (Q1 (a), Liam 14/11 at THESIS local_701fb2e3; CODE may change the tools:
# Liam 14/11 CODE local_9951f4b7 "duyệt luật S05 v3.15 nhé"): a HIL run whose run_active edge could not be mapped
# to a plant step (hil_edge_map refused, or no --esp-log; the scorer then lets only G1, G2 decide) is lost to the
# measurement: run the same gain again, not counted in the candidate limit, at most MAP_RERUNS_MAX reruns for one
# gain at one level; past that the level is chua_ket_luan (counted as not passed), the reason logged.
# [CÁCH HIỂU] "at most 3 reruns" = the level stops at the 4th refused run (1 + 3 reruns), all refused runs of the
# level counted, in a row or not; final_hil keeps its own rule (§4.6, FINAL_LOST_N lost runs in a row).
MAP_REFUSED = 'anh_xa_tu_choi'           # --fail-cause of such a run
MAP_RERUNS_MAX = 3

CAMPAIGNS = ('A-PID', 'A-LADRC', 'B-PID', 'B-LADRC', 'B-chung', 'RQ2-INIT-PID', 'RQ2-INIT-LADRC', 'RQ2')
TUNED = ('A-PID', 'A-LADRC', 'B-PID', 'B-LADRC')
GAIN_OWNERS = TUNED + ('RQ2-INIT-PID', 'RQ2-INIT-LADRC')   # the six gain sets of the final session (§4.6)
KHOAN = ('T_H', 'C_chung', 'RQ2')
STAGES = ('screen', 'confirm', 'final', 'explore', 'hil_check', 'final_hil', 'script')   # script: v3.7 §3.3
FIG8 = {'O1-F02': 0.2, 'O1-F05': 0.5, 'O1-F08': 0.8}     # same as rpi5/o1_tasks.py FIG8 (a test checks)
MANUAL = '[nhap_tay]'                    # flag of a hoc row typed by hand (S05 §3.1 R-2, EA-7)
OUTCOMES = ('dat', 'khong_dat', 'chua_ket_luan', 'hong_he_do')
PLATFORMS = ('SIM', 'HIL', 'HW')
STEPS = ('flash', 'setup', 'run', 'log', 'analysis')
CAUSES = ('gain', 'cong_cu_HIL', 'phan_cung_ngau_nhien', 'chua_ro')
CAP_H, CAP_HW = 10.0, 120               # S05 §4.1: per campaign, tune part only
TUNE_STAGES = ('screen', 'confirm')     # [CÁCH HIỂU] 'explore' is not a tune run of §4.1 either way
SKIP_PREFIX = ('EXAMPLE', 'PREHIL')      # rows of the June files: examples, reconstruction


class LogError(Exception):
    pass


def now() -> dt.datetime:
    """PC clock, local, whole seconds (tests replace this)."""
    return dt.datetime.now().replace(microsecond=0)


def _iso(t: dt.datetime) -> str:
    return t.isoformat(timespec='seconds')


def _parse_t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _fmt(x: float, nd=1) -> str:
    return f'{x:.{nd}f}'


# ── CSV files ──────────────────────────────────────────────────────────────
def read_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _header(path: str) -> List[str]:
    with open(path, newline='', encoding='utf-8') as f:
        return next(csv.reader(f), [])


def _write_all(path: str, cols: List[str], rows: Iterable[dict]) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    os.replace(tmp, path)


def append_row(path: str, cols: List[str], row: dict) -> None:
    """Append one row; the file must already have exactly these columns (run `migrate`)."""
    unknown = set(row) - set(cols)
    if unknown:
        raise LogError(f'{os.path.basename(path)}: unknown columns {sorted(unknown)}')
    if not os.path.exists(path):
        _write_all(path, cols, [])
    if _header(path) != cols:
        raise LogError(f'{os.path.basename(path)}: header differs from S05 §6; run "rq3_log.py migrate" first')
    with open(path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=cols, lineterminator='\n').writerow({c: row.get(c, '') for c in cols})


def migrate_file(path: str, cols: List[str]) -> str:
    """Add the missing columns of `cols` at their place, keep every value. 'ok' / 'migrated' / 'created'.
    Refuses a file with a column S05 does not know (nothing is dropped). The single exception (G32-11): the
    g*_u columns of gate_v3_g1235 (hil 7768134) are dropped when every cell of them is empty; a value in any
    of them refuses the file (no conversion of gate_v3_g1235 rows)."""
    if not os.path.exists(path):
        _write_all(path, cols, [])
        return 'created'
    old = _header(path)
    if old == cols:
        return 'ok'
    extra = [c for c in old if c not in cols]
    rows = read_rows(path)
    old_u = [c for c in extra if c in OLD_GATE_U]
    if old_u and any(r.get(c) for r in rows for c in old_u):
        raise LogError(f'{os.path.basename(path)}: columns {old_u} of gate_v3_g1235 hold values; not migrated')
    extra = [c for c in extra if c not in OLD_GATE_U]
    if extra:
        raise LogError(f'{os.path.basename(path)}: columns not in S05 §6 {extra}; not migrated')
    _write_all(path, cols, rows)
    return 'migrated'


def migrate(ds: str) -> Dict[str, str]:
    out = {name: migrate_file(os.path.join(ds, name), cols)
           for name, cols in ((TUNING, TUNING_COLS), (EFFORT, EFFORT_COLS), (HWFAIL, HWFAIL_COLS))}
    if os.path.exists(os.path.join(ds, FREEZE)):              # v3.6 file without stop_cause (v3.7)
        out[FREEZE] = migrate_file(os.path.join(ds, FREEZE), FREEZE_COLS)
    return out


# ── checks shared by the commands and the summary ─────────────────────────
def check_task(task: str, speed) -> None:
    """EA-9: a figure-8 task name carries its level; the level and speed_mps must agree, so the
    gate (a list of task names) cannot merge two speeds."""
    try:
        bad = not math.isfinite(float(speed))
    except (TypeError, ValueError):
        bad = False                                     # not a number: the figure-8 check below refuses it
    if bad:                                             # Codex S315-R2-1: nan passes abs(nan - x) > tol
        raise LogError(f'speed {speed!r}: not a finite number')
    if task.startswith('O1-F'):
        if task not in FIG8:
            raise LogError(f'task {task!r}: figure-8 names are {sorted(FIG8)} (the name carries the level)')
        if abs(_f(speed) - FIG8[task]) > 1e-9:
            raise LogError(f'task {task} is level {FIG8[task]} m/s, speed {speed!r} given')


def gain_owner(gain_set: str) -> str:
    """EA-5 (S05 v3.7 §6.3): gain_set_id = '<campaign>/<id>', unique per campaign."""
    owner, sep, rest = (gain_set or '').partition('/')
    if not sep or not rest or owner not in GAIN_OWNERS:
        raise LogError(f'gain set {gain_set!r}: write it "<campaign>/<id>", campaign one of {GAIN_OWNERS}')
    return owner


def check_gain_set(campaign: str, stage: str, gain_set: str) -> None:
    """A run of campaign C carries a gain set of C; a final_hil run (account RQ2) carries the gain
    set of the campaign (or RQ2-INIT-*) it tests."""
    if not gain_set:
        return
    owner = gain_owner(gain_set)
    if stage != 'final_hil' and owner != campaign:
        raise LogError(f'gain set {gain_set!r} belongs to {owner}, the run is {campaign}')


# ── one run, live ──────────────────────────────────────────────────────────
def _open_path(ds):
    return os.path.join(ds, OPEN_RUN)


def _load_open(ds) -> Optional[dict]:
    p = _open_path(ds)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _save_open(ds, run: dict) -> None:
    p = _open_path(ds)
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=1, ensure_ascii=False)
    os.replace(p + '.tmp', p)


def _parse_gains(text: str) -> dict:
    out = {}
    for part in filter(None, (p.strip() for p in (text or '').split(','))):
        k, sep, v = part.partition('=')
        if not sep or k not in GAINS:
            raise LogError(f'--gains: "{part}" (names: {", ".join(GAINS)})')
        float(v)
        out['g_' + k] = v
    return out


def next_iter_no(ds: str, session: str) -> int:
    nums = [int(_f(r.get('iter_no'))) for r in read_rows(os.path.join(ds, TUNING)) if r.get('session_id') == session]
    return max(nums, default=0) + 1


def begin(ds: str, session: str, campaign: str, platform: str, controller: str, task: str, speed: str,
          stage: str, gain_set: str = '', gains: str = '', **extra) -> dict:
    if _load_open(ds):
        raise LogError('a run is open: "end" or "cancel" it first')
    if _load_hoc(ds):
        raise LogError('a hoc segment is open: "hoc-end" first (hoc touches no run, S05 §3.1)')
    if campaign not in CAMPAIGNS:
        raise LogError(f'--campaign {campaign!r}: one of {CAMPAIGNS}')
    if platform not in PLATFORMS or stage not in STAGES or stage == 'script':
        raise LogError(f'--platform one of {PLATFORMS}, --stage one of {STAGES[:-1]} (script: "machine")')
    if not session or session.startswith(SKIP_PREFIX) or ':' in session:
        raise LogError('--session: a name of your own, no ":" (run_uid = session_id:iter_no)')
    check_task(task, speed)
    check_gain_set(campaign, stage, gain_set)
    it = next_iter_no(ds, session)
    t = now()
    row = {'session_id': session, 'date': t.date().isoformat(), 'platform': platform, 'controller': controller,
           'trajectory': task, 'iter_no': str(it), 'run_uid': f'{session}:{it}', 'campaign': campaign,
           'gain_set_id': gain_set, 'speed_mps': speed, 'stage': stage, 'wall_start': _iso(t)}
    row.update(_parse_gains(gains))
    for k, v in extra.items():
        if v not in (None, ''):
            if k not in TUNING_COLS:
                raise LogError(f'unknown field {k}')
            row[k] = str(v)
    tun = live_rows(read_rows(os.path.join(ds, TUNING)))      # S315-R2-2
    if _map_counted(row):
        if not gain_set:
            raise LogError('a HIL run needs --gain-set (S05 v3.15 4.5 step 2 counts reruns per candidate)')
        if map_stopped(map_refusals(tun + [row]).get(map_key(row), 0)):
            raise LogError(f'{task}@{speed} of {gain_set}: stopped after {MAP_RERUNS_MAX} reruns for a refused '
                           'mapping (S05 v3.15 4.5 step 2): the level is chua_ket_luan, no further run')
    if stage in ('final', 'final_hil') and final_key(row) in _final_stops(tun):
        raise LogError(f'{task}@{speed} of {gain_set or campaign}: stopped after {FINAL_LOST_N} lost runs in a row '
                       '(S05 4.6), no further run')
    run = {'row': row, 'segments': [], 'current': None, 'paused_from': None}
    _save_open(ds, run)
    return run


def _close_segment(run: dict, t: dt.datetime) -> None:
    cur = run.get('current')
    if cur:
        cur['end'] = _iso(t)
        run['segments'].append(cur)
        run['current'] = None


def step(ds: str, kind: str, wait: bool = False) -> dict:
    run = _require_open(ds)
    if kind not in STEPS:
        raise LogError(f'step one of {STEPS}')
    if run.get('paused_from') is not None:
        raise LogError('paused: "resume" first')
    t = now()
    _close_segment(run, t)
    run['current'] = {'kind': kind, 'wait': bool(wait), 'start': _iso(t)}
    _save_open(ds, run)
    return run


def pause(ds: str) -> dict:
    run = _require_open(ds)
    if run.get('paused_from') is not None:
        raise LogError('already paused')
    cur = run.get('current')
    _close_segment(run, now())
    run['paused_from'] = cur or {}
    _save_open(ds, run)
    return run


def resume(ds: str) -> dict:
    run = _require_open(ds)
    prev = run.get('paused_from')
    if prev is None:
        raise LogError('not paused')
    run['paused_from'] = None
    if prev:
        run['current'] = {'kind': prev['kind'], 'wait': prev['wait'], 'start': _iso(now())}
    _save_open(ds, run)
    return run


def _require_open(ds) -> dict:
    run = _load_open(ds)
    if not run:
        raise LogError('no open run: "begin" first')
    return run


def segment_totals(segments: List[dict]) -> dict:
    tot = {k: 0.0 for k in STEPS}
    wait = 0.0
    for s in segments:
        d = (_parse_t(s['end']) - _parse_t(s['start'])).total_seconds()
        if d < 0:
            raise LogError(f'segment {s} ends before it starts')
        tot[s['kind']] += d
        if s.get('wait'):
            wait += d
    return {**{f't_{k}_s': v for k, v in tot.items()}, 't_operator_wait_s': wait}


def read_gate_json(path: str) -> dict:
    """The gate block of a scorer output (key 'gate_v3', hil_o1_score.py) or a bare gate dict."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError) as exc:
        raise LogError(f'--gate-json {path!r}: {exc}')
    g = d.get('gate_v3', d) if isinstance(d, dict) else None
    if not isinstance(g, dict) or not str(g.get('version', '')).startswith('gate_v3'):
        raise LogError(f'--gate-json {path!r}: no gate_v3 block (fwsim/o1_gate.py)')
    return g


def _num(x) -> str:
    return '' if x is None or isinstance(x, bool) else (repr(float(x)) if isinstance(x, (int, float)) else str(x))


def _qnum(x, what: str):
    """A q or U of a gate: null / NaN (unknown, as the scorer writes NaN) or a finite number; nothing else."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float('nan')
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise LogError(f'gate {what}: {x!r} is not null or a finite number')
    return float(x)


# conditions that decide, by platform (GV3-R2-2; 13/11 RS-4): HIL decides all five when t = 0 is the mapped
# ESP32 run_active edge (U_t known), else G1, G2 (U_t not known, O1 3.5); SIM and the floor decide all five
DECIDE_BY_PLATFORM = {'HIL': (o1_gate.DECIDE_ALL, o1_gate.DECIDE_HIL), 'SIM': (o1_gate.DECIDE_ALL,),
                      'HW': (o1_gate.DECIDE_ALL,)}
_G1_Q = ('true', 'false', 'inconclusive')
_TOP_KEYS = ('version', 'decide', 'u_t', 'outcome', 'outcome_decide', 'complete', 'any_certain_fail')
_BOOL_KEYS = ('complete', 'any_certain_fail')
_COND_KEYS = ('q', 'lo', 'hi', 'n_undef', 'threshold', 'state')


def _check_decide(decide, platform: Optional[str] = None) -> List[str]:
    if (not isinstance(decide, list) or not decide or len(set(decide)) != len(decide)
            or any(not isinstance(c, str) for c in decide) or set(decide) - set(GATE_CONDS)):
        raise LogError(f'gate decide {decide!r}: a non-empty list of distinct {GATE_CONDS}')
    if set(decide) not in [set(x) for x in o1_gate.DECIDE_SETS]:
        raise LogError(f'gate decide {decide!r}: {list(o1_gate.DECIDE_ALL)} or {list(o1_gate.DECIDE_HIL)}')
    if platform is not None:
        want = DECIDE_BY_PLATFORM.get(platform)
        if want is None or set(decide) not in [set(x) for x in want]:
            raise LogError(f'gate decide {decide!r} on platform {platform!r}: must be one of '
                           f'{[list(x) for x in (want or ())]} (GV3-R2-2)')
    return decide


def _n_undef(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not float(x).is_integer() or x < 0:
        raise LogError(f'gate {what}: {x!r} is not a whole number >= 0')
    return int(x)


def _recompute(gate: dict, platform: Optional[str] = None) -> dict:
    """States and outcomes recomputed from q, lo, hi, n_undef with fwsim/o1_gate (thresholds, O1 3.5
    three states SC3-1, O1 3.5 order); shared by the JSON path (end) and the logged rows (summary).
    Returns the outcomes."""
    if gate.get('version') != o1_gate.GATE_VERSION:
        raise LogError(f'gate version {gate.get("version")!r}: only {o1_gate.GATE_VERSION} is supported')
    decide = _check_decide(gate.get('decide'), platform)
    u_t = _qnum(gate.get('u_t'), 'u_t')
    if u_t < 0.0:
        raise LogError(f'gate u_t: {u_t!r} is negative (a bound is >= 0)')
    # U_t known <=> G3, G5 may decide (RS-4): a HIL gate without the mapping cannot claim all five
    if (set(decide) == set(o1_gate.DECIDE_ALL)) == math.isnan(u_t):
        raise LogError(f'gate decide {decide!r} with u_t {gate.get("u_t")!r}: all five decide exactly when U_t '
                       'is known (RS-4)')
    q1 = gate['g1'].get('q')
    if q1 not in _G1_Q or gate['g1'].get('state') != o1_gate.g1_state(q1):
        raise LogError(f'gate g1: q {q1!r} and state {gate["g1"].get("state")!r} do not agree (O1 3.4 completed)')
    for c, thr in o1_gate.THRESHOLDS.items():
        q, lo, hi = (_qnum(gate[c].get(k), f'{c} {k}') for k in ('q', 'lo', 'hi'))
        nu = _n_undef(gate[c].get('n_undef'), f'{c} n_undef')
        # n_undef 0: every frame finite, so q, lo, hi are all finite with lo <= q <= hi (some frame), or all
        # unknown (no frame scored: inconclusive). With undefined frames each field is the maximum over its own
        # finite frames (SC3-1), so hi may be below q (the frame of q may have hi = inf). G32-1
        fin = [not math.isnan(x) for x in (q, lo, hi)]
        if nu == 0 and any(fin) and not (all(fin) and lo <= q <= hi):
            raise LogError(f'gate {c}: with n_undef 0, q, lo, hi must all be numbers with lo <= q <= hi '
                           f'(or all unknown); got {lo!r}, {q!r}, {hi!r}')
        st = o1_gate.cond_state(lo, hi, nu, thr)
        if st != gate[c].get('state'):
            raise LogError(f'gate {c}: state {gate[c].get("state")!r}, recomputed from lo, hi, n_undef: {st!r}')
    return o1_gate.evaluate_outcomes(gate, decide)


def validate_gate(gate: dict, platform: Optional[str] = None) -> dict:
    """GV3-5, GV3-R2-3: a gate JSON checked key by key (a missing key is an error, null is 'unknown';
    booleans by type; thresholds), then every state and outcome recomputed (_recompute); any difference
    is an error. The outcome fields of the JSON are never trusted. platform: the run's platform, whose
    deciding sets the JSON must match (GV3-R2-2). Returns the recomputed outcomes."""
    if not isinstance(gate, dict):
        raise LogError(f'gate: {gate!r} is not an object')
    for k in _TOP_KEYS + GATE_CONDS:
        if k not in gate:
            raise LogError(f'gate {k}: missing')
    for c in GATE_CONDS:
        need = ('q', 'state') if c == 'g1' else _COND_KEYS
        if not isinstance(gate[c], dict) or any(k not in gate[c] for k in need):
            raise LogError(f'gate {c}: missing, or without {need}')
    for c, thr in o1_gate.THRESHOLDS.items():
        if isinstance(gate[c]['threshold'], bool) or gate[c]['threshold'] != thr:
            raise LogError(f'gate {c}: threshold {gate[c]["threshold"]!r}, expected {thr}')
    for k in _BOOL_KEYS:
        if not isinstance(gate[k], bool):
            raise LogError(f'gate {k}: {gate[k]!r} is not a boolean')
    ev = _recompute(gate, platform)
    for k, v in ev.items():
        if type(gate[k]) is not type(v) or gate[k] != v:
            raise LogError(f'gate {k}: {gate[k]!r}, recomputed: {v!r}')
    return ev


def gate_from_row(row: dict) -> dict:
    """The gate of a logged run (GATE_COLS) as a dict for _recompute ('' = unknown)."""
    def num(s, whole=False):
        try:
            if s in ('', None):
                return None
            v = float(s)
            return int(v) if whole and v.is_integer() else v
        except ValueError:
            raise LogError(f"{row.get('run_uid')}: gate value {s!r} is not a number")
    g = {'version': row.get('gate_version'), 'decide': [c for c in row.get('gate_decide', '').split(',') if c],
         'u_t': num(row.get('gate_u_t')), 'g1': {'q': row.get('g1_q'), 'state': row.get('g1_state')}}
    for c in o1_gate.MEASURED:
        g[c] = {k: num(row.get(f'{c}_{k}')) for k in ('q', 'lo', 'hi')}
        g[c].update(n_undef=num(row.get(f'{c}_n_undef'), whole=True), state=row.get(f'{c}_state'))
    return g


def map_refused(decide) -> bool:
    """The gate of a HIL run whose run_active edge was not mapped (only G1, G2 decide, O1 v11.12 §3.4)."""
    return set(decide) == set(o1_gate.DECIDE_HIL)


def outcome_vs_gate(outcome: str, ev: dict, decide, fail_cause: str = '') -> Optional[str]:
    """None when `outcome` (and its fail cause) agrees with the recomputed gate outcomes `ev`, else why not.
    Gate with all five deciding: hong_he_do is decided outside the gate (O1 3.5) but not for a refused
    mapping (the mapping was made); any other outcome equals outcome_decide.
    Gate of a HIL run without the run_active edge mapping (G1, G2 decide, G3, G5 report only): S05 v3.15
    §4.5 step 2 (replaces GV3-1, GV3-2 of 12/11, under which such a run could be khong_dat or chua_ket_luan
    by G1, G2): the run is lost to the measurement, hong_he_do with fail cause MAP_REFUSED; no run is dat."""
    od = ev['outcome_decide']
    if map_refused(decide):
        if outcome == 'hong_he_do' and fail_cause == MAP_REFUSED:
            return None
        return (f'run_active edge not mapped (gate decides {",".join(decide)} only): a lost run, outcome '
                f'hong_he_do with fail cause {MAP_REFUSED} (S05 v3.15 4.5 step 2); no run is dat')
    if fail_cause == MAP_REFUSED:
        return f'fail cause {MAP_REFUSED} but the gate has the mapping (all five decide)'
    if outcome == 'hong_he_do':
        return None
    if ev['complete']:
        return None if outcome == od else f'gate outcome {od}'
    # not reached: decide is DECIDE_ALL (complete) or DECIDE_HIL (o1_gate.DECIDE_SETS), kept as a guard
    return f'the gate is not complete (decides {",".join(decide)}): no run is dat'


def check_gate_rows(tun: List[dict]) -> None:
    """GV3-1, GV3-R2-1: every logged row that carries a gate is recomputed from its q, lo, hi, n_undef
    (version, deciding set of its platform, thresholds, states); its run_outcome must agree as at `end`.
    Rows whose gate columns are ALL empty (before gate v3, or written without --gate-json) are not
    checked; any non-empty gate cell makes the row a gate row, with a supported version (GV3-R3-1)."""
    for r in tun:
        if not any(r.get(c) for c in GATE_COLS):
            continue
        g = gate_from_row(r)
        try:
            ev = _recompute(g, r.get('platform'))
        except LogError as exc:
            raise LogError(f"{r.get('run_uid')}: {exc}")
        why = outcome_vs_gate(r.get('run_outcome', ''), ev, g['decide'], r.get('fail_cause', ''))
        if why:
            raise LogError(f"{r.get('run_uid')}: run_outcome {r.get('run_outcome')!r} disagrees with its gate "
                           f"({r.get('gate_version')}: {why})")


def gate_columns(gate: dict, gate_sha256: str = '') -> dict:
    """GATE_COLS of one run from a gate dict (fwsim/o1_gate.gate). q of g1 is the evaluator's
    'completed' (true / false / inconclusive)."""
    out = {'gate_version': gate['version'], 'gate_decide': ','.join(gate['decide']),
           'gate_json_sha256': gate_sha256, 'gate_u_t': _num(gate.get('u_t')),
           'g1_q': str(gate['g1'].get('q')), 'g1_state': gate['g1'].get('state', '')}
    for g in o1_gate.MEASURED:
        c = gate.get(g) or {}
        for k in ('q', 'lo', 'hi'):
            v = c.get(k)
            out[f'{g}_{k}'] = '' if v is None or (isinstance(v, float) and math.isnan(v)) else _num(v)
        out[f'{g}_n_undef'] = str(int(c['n_undef'])) if c.get('n_undef') is not None else ''
        out[f'{g}_state'] = c.get('state', '')
    return out


def check_outcome_vs_gate(outcome: str, gate: dict, platform: Optional[str] = None, fail_cause: str = '') -> None:
    """The run outcome typed by the operator against the gate, validated and recomputed first (GV3-5)."""
    ev = validate_gate(gate, platform)
    why = outcome_vs_gate(outcome, ev, gate['decide'], fail_cause)
    if why:
        raise LogError(f'--outcome {outcome} disagrees with the gate ({why}); check the scoring')


def _num_text(s) -> str:
    """A number as one text whatever way it was typed ('1', '1.0', '1e0' -> '1.0'); other text unchanged."""
    try:
        return '' if s in ('', None) else repr(float(s))
    except (TypeError, ValueError):
        return str(s)


def map_key(r: dict) -> tuple:
    """One candidate at one level (S05 v3.15 4.5 step 2): (campaign, gain set id, task, speed as a number).
    The stage is not in it (Codex S315-3: a run of the same gain at the same level in another HIL stage counts
    against the same reruns); the gain values are checked for agreement in map_refusals (S315-2). A figure-8
    task gives its level FIG8[task], not the speed as typed (S315-R2-1: check_task allows 1e-9 around it)."""
    task = r.get('trajectory', '')
    speed = repr(float(FIG8[task])) if task in FIG8 else _num_text(r.get('speed_mps'))
    return (r.get('campaign', ''), r.get('gain_set_id', ''), task, speed)


def live_rows(tun: List[dict]) -> List[dict]:
    """The tuning rows every rule looks at: the June example and reconstruction rows (SKIP_PREFIX) left out,
    as in summary (Codex S315-R2-2: begin, end and the CLI note use the same rows)."""
    return [r for r in tun if not r.get('session_id', '').startswith(SKIP_PREFIX)]


def _map_counted(r: dict) -> bool:
    return r.get('platform') == 'HIL' and r.get('stage') != 'final_hil'      # final_hil: S05 4.6 (FINAL_LOST_N)


def map_refusals(tun: List[dict]) -> Dict[tuple, int]:
    """S05 v3.15 4.5 step 2: refused-mapping runs per candidate and level (map_key), HIL stages but final_hil,
    in the order of the rows in the log (the order they were written; Codex S315-4: not re-sorted by the PC
    clock). The level stops at 1 + MAP_RERUNS_MAX refused runs; a run of it after that is an error, as is a
    counted run without gain set id or one whose gain values (as numbers) differ from an earlier run of the same
    key (S315-2) (LogError)."""
    n = defaultdict(int)
    gains = {}
    for r in tun:
        if not _map_counted(r):
            continue
        uid = r.get('run_uid')
        if not r.get('gain_set_id'):
            raise LogError(f'{uid}: a HIL run carries its gain set id (--gain-set; S05 v3.15 4.5 step 2 counts '
                           'reruns per candidate)')
        k = map_key(r)
        g = {x: _num_text(r.get('g_' + x)) for x in GAINS if r.get('g_' + x, '') != ''}
        old = gains.setdefault(k, g)
        diff = sorted(x for x in set(g) & set(old) if g[x] != old[x])
        if diff:
            raise LogError(f"{uid}: gain set {r.get('gain_set_id')} with other values of {', '.join(diff)} than an "
                           f"earlier HIL run at {r.get('trajectory')}@{r.get('speed_mps')} (one candidate, one gain)")
        old.update(g)
        if n[k] > MAP_RERUNS_MAX:
            raise LogError(f"{uid}: {r.get('trajectory')}@{r.get('speed_mps')} of {r.get('gain_set_id')} stopped "
                           f"after {MAP_RERUNS_MAX} reruns for a refused mapping (S05 v3.15 4.5 step 2: the level is "
                           f"chua_ket_luan), no run after that")
        if r.get('fail_cause') == MAP_REFUSED:
            n[k] += 1
    return dict(n)


def check_map_rows(tun: List[dict]) -> None:
    """S05 v3.15 4.5 step 2 on logged rows (Codex S315-1): a refused-mapping row is a HIL hong_he_do row whose
    gate lets only G1, G2 decide; a HIL row judged dat / khong_dat / chua_ket_luan carries its gate (a run
    without the edge mapping could otherwise pass). Rows of the gate are recomputed by check_gate_rows."""
    for r in tun:
        uid = r.get('run_uid')
        if r.get('fail_cause') == MAP_REFUSED and not (
                r.get('platform') == 'HIL' and r.get('run_outcome') == 'hong_he_do'
                and map_refused([c for c in r.get('gate_decide', '').split(',') if c])):
            raise LogError(f'{uid}: fail cause {MAP_REFUSED} on a row that is not a HIL hong_he_do run with a gate '
                           'deciding g1,g2 only')
        if r.get('platform') == 'HIL' and r.get('run_outcome') in VALID and not any(r.get(c) for c in GATE_COLS):
            raise LogError(f"{uid}: HIL run {r.get('run_outcome')} without its gate (--gate-json): S05 v3.15 4.5 "
                           'step 2 needs it to tell a mapped run from a refused one')


def final_key(r: dict) -> tuple:
    """A condition of the final session (S05 4.6), as in _final_stops."""
    return (r.get('stage', ''), r.get('campaign', ''), r.get('gain_set_id', ''), r.get('platform', ''),
            r.get('trajectory', ''), r.get('speed_mps', ''))


def final_lost_streak(tun: List[dict], key: tuple) -> int:
    """Lost runs in a row at the end of one final condition (S05 4.6), in the order of the log."""
    s = 0
    for r in tun:
        if r.get('stage') in ('final', 'final_hil') and final_key(r) == key:
            s = s + 1 if r.get('run_outcome') == 'hong_he_do' else 0
    return s


def map_stopped(n_refused: int) -> bool:
    return n_refused > MAP_RERUNS_MAX


def end(ds: str, outcome: str, completed: Optional[int] = None, e_perp_max: Optional[float] = None,
        fail_cause: str = '', results_dir: str = '', notes: str = '', gate: Optional[dict] = None,
        gate_sha256: str = '') -> dict:
    run = _require_open(ds)
    if outcome not in OUTCOMES:
        raise LogError(f'--outcome one of {OUTCOMES}')
    if fail_cause == MAP_REFUSED and (gate is None or run['row'].get('platform') != 'HIL'):
        raise LogError(f'--fail-cause {MAP_REFUSED}: a HIL run with its scorer output (--gate-json) showing the '
                       'refused mapping (S05 v3.15 4.5 step 2)')
    if gate is not None:
        if not results_dir:
            raise LogError('--gate-json needs --results-dir (the run must be scorable again, GV3-6)')
        check_outcome_vs_gate(outcome, gate, run['row'].get('platform'), fail_cause)
    if outcome == 'hong_he_do' and not fail_cause:
        raise LogError('hong_he_do needs --fail-cause (S05 §6.1)')
    if run['row'].get('platform') == 'HIL' and outcome in VALID and gate is None:
        raise LogError(f'a HIL run judged {outcome} needs its scorer output (--gate-json): S05 v3.15 4.5 step 2 '
                       '(a run without the run_active edge mapping is a lost run)')        # Codex S315-1
    if run.get('paused_from') is not None:
        raise LogError('paused: "resume" first (or the pause is lost time of the next segment)')
    t = now()
    _close_segment(run, t)
    row = dict(run['row'])
    row.update({k: _fmt(v, 0) for k, v in segment_totals(run['segments']).items()})
    row.update(run_outcome=outcome, wall_end=_iso(t), fail_cause=fail_cause, results_dir=results_dir)
    if completed is not None:
        row['completed'] = str(int(completed))
    if e_perp_max is not None:
        row['e_perp_max_m'] = f'{e_perp_max:.4f}'
    if gate is not None:
        row.update(gate_columns(gate, gate_sha256))
    if notes:
        row['notes'] = (row.get('notes', '') + ' ' + notes).strip()
    # Codex S315-5: the log as it will be, this row included, checked before the row is written
    after = live_rows(read_rows(os.path.join(ds, TUNING))) + [row]
    check_map_rows([row])
    if _map_counted(row):
        try:
            map_refusals(after)
        except LogError as exc:
            raise LogError(f'the run is not logged (the log changed since begin?): {exc}')
    append_row(os.path.join(ds, TUNING), TUNING_COLS, row)
    os.remove(_open_path(ds))
    return row


def refusal_note(ds: str, row: dict) -> str:
    """What the operator does next after a refused-mapping run (printed by the CLI after the row is written;
    the row was checked before it was written, so a failed recount here is only reported, Codex S315-5).
    final_hil follows S05 4.6 (lost runs in a row), the other HIL stages 4.5 step 2 (Codex S315-6)."""
    if row.get('fail_cause') != MAP_REFUSED:
        return ''
    try:
        tun = live_rows(read_rows(os.path.join(ds, TUNING)))
        if row.get('stage') == 'final_hil':
            s = final_lost_streak(tun, final_key(row))
            return (f'final_hil: anh xa bi tu choi, {s} luot hong lien (S05 4.6): ' +
                    ('dieu kien nay dung' if s >= FINAL_LOST_N else f'chay lai (dung khi du {FINAL_LOST_N} luot hong lien)'))
        n = map_refusals(tun).get(map_key(row), 0)
    except LogError as exc:
        return f'canh bao: dong da ghi, nhung khong dem lai duoc ({exc})'
    return f'anh xa bi tu choi lan {n}: ' + (
        f'qua {MAP_RERUNS_MAX} lan chay lai, muc nay chua ket luan (tinh nhu khong dat), ly do ghi o hil_map_stopped '
        'cua summary' if map_stopped(n) else
        f'chay lai cung gain (con {MAP_RERUNS_MAX + 1 - n} lan), khong dem vao gioi han ung vien')


def cancel(ds: str) -> dict:
    """EA-6: a run whose "run" step started counts (S05 §3.3): close it with "end" (an early end
    is an outcome plus --fail-cause), never cancel it. Before that step a cancel drops the run;
    the time of the steps already logged (flash, setup, ...) is active work of the campaign and
    goes to m7_dev_effort_log.csv as one row (gio = sum of the segments, pauses out)."""
    run = _require_open(ds)
    kinds = [s['kind'] for s in run['segments']] + [s['kind'] for s in (run.get('current'), run.get('paused_from'))
                                                     if s]
    if 'run' in kinds:
        raise LogError('the "run" step started: the run counts (S05 §3.3); close it with '
                       '"end --outcome ... --fail-cause ..."')
    t = now()
    if run.get('paused_from') is None:
        _close_segment(run, t)
    active = sum(v for k, v in segment_totals(run['segments']).items() if k != 't_operator_wait_s')
    row = run['row']
    if active > 0:
        t0 = _parse_t(row['wall_start'])
        eff = {'ngay': t0.date().isoformat(), 'bat_dau': t0.strftime('%H:%M:%S'), 'ket_thuc': t.strftime('%H:%M:%S'),
               'gio': f'{active / 3600:.4f}', 'chien_dich': row['campaign'],
               'ghi_chu': f"rq3_log cancel truoc buoc run: {row['run_uid']} ({row['trajectory']} {row['speed_mps']}); "
                          f"gio = tong cac buoc da ghi"}
        append_row(os.path.join(ds, EFFORT), EFFORT_COLS, eff)
        run['effort_row'] = eff
    os.remove(_open_path(ds))
    return run


# ── rows not tied to one live run ─────────────────────────────────────────
def machine(ds: str, campaign: str, controller: str, script_run_id: str, results_dir: str, n_sim_evals: int,
            t_machine_s: float, wall_start: str, wall_end: str, platform: str = 'SIM', stage: str = 'script',
            session: str = 'MACHINE', notes: str = '', **active) -> dict:
    """Summary row of a script run (S5-23 v2): one per (controller, account); n_sim_evals and
    t_machine_s are this row's part, the rows of one script run add up to the run. Stage `script`
    by default (S05 v3.7 §3.3, A-R2-4: an `explore` row would be left out of T and N, §4.8)."""
    if campaign not in CAMPAIGNS:
        raise LogError(f'--campaign {campaign!r}: one of {CAMPAIGNS}')
    if stage not in STAGES:
        raise LogError(f'--stage one of {STAGES}')
    it = next_iter_no(ds, session)
    row = {'session_id': session, 'iter_no': str(it), 'run_uid': f'{session}:{it}', 'date': wall_start[:10],
           'platform': platform, 'controller': controller, 'campaign': campaign, 'stage': stage,
           'script_run_id': script_run_id, 'results_dir': results_dir, 'n_sim_evals': str(int(n_sim_evals)),
           't_machine_s': _fmt(t_machine_s, 0), 'wall_start': wall_start, 'wall_end': wall_end, 'notes': notes}
    for k in STEPS:
        v = active.get(f't_{k}_s')
        if v is not None:
            row[f't_{k}_s'] = _fmt(float(v), 0)
    append_row(os.path.join(ds, TUNING), TUNING_COLS, row)
    return row


def effort(ds: str, date: str, start: str, end_: str, campaign: str = '', khoan: str = '', nhan: str = '',
           ghi_chu: str = '', **extra) -> dict:
    """One row of m7_dev_effort_log.csv from the times Liam reports (never made up)."""
    if bool(campaign) == bool(khoan):
        raise LogError('exactly one of --campaign, --khoan (S05 §6.2)')
    if campaign and campaign not in CAMPAIGNS or khoan and khoan not in KHOAN:
        raise LogError(f'--campaign one of {CAMPAIGNS}, --khoan one of {KHOAN}')
    if nhan not in ('', 'hoc'):
        raise LogError('--nhan hoc or nothing')
    if nhan == 'hoc' and not ghi_chu:
        raise LogError('a hoc row names what was read (S05 §3.1)')
    if nhan == 'hoc' and not ghi_chu.startswith(MANUAL):
        ghi_chu = f'{MANUAL} {ghi_chu}'                 # EA-7: times typed, not stamped by the script
    t0 = dt.datetime.fromisoformat(f'{date}T{start}')
    t1 = dt.datetime.fromisoformat(f'{date}T{end_}')
    if t1 <= t0:
        t1 += dt.timedelta(days=1)                      # past midnight
    row = {'ngay': date, 'bat_dau': start, 'ket_thuc': end_, 'gio': f'{(t1 - t0).total_seconds() / 3600:.2f}',
           'chien_dich': campaign, 'khoan': khoan, 'nhan': nhan, 'ghi_chu': ghi_chu}
    row.update({k: v for k, v in extra.items() if v not in (None, '')})
    append_row(os.path.join(ds, EFFORT), EFFORT_COLS, row)
    return row


def _hoc_path(ds):
    return os.path.join(ds, OPEN_HOC)


def _load_hoc(ds) -> Optional[dict]:
    p = _hoc_path(ds)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def hoc_start(ds: str, campaign: str, ghi_chu: str) -> dict:
    """EA-7 (S05 §3.1 R-2): open a hoc segment; the script stamps its start (PC clock)."""
    if campaign not in TUNED:
        raise LogError(f'--campaign one of {TUNED} (hoc of a campaign, S05 §3.1)')
    if not ghi_chu:
        raise LogError('a hoc row names what was read (S05 §3.1)')
    if _load_hoc(ds):
        raise LogError('a hoc segment is open: "hoc-end" first (one segment at a time)')
    if _load_open(ds):
        raise LogError('a run is open: hoc touches no run (S05 §3.1)')
    seg = {'campaign': campaign, 'ghi_chu': ghi_chu, 'start': _iso(now())}
    with open(_hoc_path(ds) + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(seg, f, ensure_ascii=False)
    os.replace(_hoc_path(ds) + '.tmp', _hoc_path(ds))
    return seg


def hoc_end(ds: str, ghi_chu: str = '') -> dict:
    """Close the hoc segment: one m7_dev_effort_log.csv row, both stamps from the script."""
    seg = _load_hoc(ds)
    if not seg:
        raise LogError('no open hoc segment: "hoc-start" first')
    t0, t1 = _parse_t(seg['start']), now()
    row = {'ngay': t0.date().isoformat(), 'bat_dau': t0.strftime('%H:%M:%S'), 'ket_thuc': t1.strftime('%H:%M:%S'),
           'gio': f'{(t1 - t0).total_seconds() / 3600:.4f}', 'chien_dich': seg['campaign'], 'nhan': 'hoc',
           'ghi_chu': (seg['ghi_chu'] + (' ' + ghi_chu if ghi_chu else '')).strip()}
    append_row(os.path.join(ds, EFFORT), EFFORT_COLS, row)
    os.remove(_hoc_path(ds))
    return row


def hwfail(ds: str, date: str, session: str, campaign: str, platform: str, what: str, cause: str,
           fix_h: float, iter_no: str = '', notes: str = '', **extra) -> dict:
    if _load_open(ds):
        raise LogError('a run is open: close it first ("end" with its outcome and --fail-cause); the repair '
                       'time is in no run (S05 v3.7 §3.3, EA-1)')
    if campaign not in CAMPAIGNS or cause not in CAUSES:
        raise LogError(f'--campaign one of {CAMPAIGNS}, --cause one of {CAUSES}')
    row = {'ngay': date, 'session_id': session, 'chien_dich': campaign, 'nen_tang': platform, 'thu_hong': what,
           'nguyen_nhan': cause, 'gio_sua': f'{fix_h:.2f}', 'iter_no': iter_no, 'ghi_chu': notes}
    row.update({k: v for k, v in extra.items() if v not in (None, '')})
    append_row(os.path.join(ds, HWFAIL), HWFAIL_COLS, row)
    return row


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze(ds: str, campaign: str, stop_reason: str, gain_set: str, gain_file: str, at: str = '',
           notes: str = '', stop_cause: str = '') -> dict:
    """S05_freeze.csv (S05 v3.6 §4.7): one row per campaign when its gains freeze. The gain file is
    the bit-exact copy of the frozen set; its sha256 is computed here, never typed. gate_result is
    not written by hand: summary() computes it from the final runs. S05 v3.7: stop_cause (the root
    cause) only with khong_co_bo_danh_gia_du, and then required (§4.1, EA-2); gain_set_id carries
    the campaign as prefix, '<campaign>/<id>' (§6.3, EA-5)."""
    if campaign not in TUNED:
        raise LogError(f'--campaign one of {TUNED}')
    if stop_reason not in STOP_REASONS:
        raise LogError(f'--stop-reason one of {STOP_REASONS}')
    if stop_reason == 'khong_co_bo_danh_gia_du':
        if stop_cause not in STOP_CAUSES:
            raise LogError(f'khong_co_bo_danh_gia_du needs --stop-cause, one of {STOP_CAUSES} (S05 v3.7 §4.1)')
    elif stop_cause:
        raise LogError('--stop-cause only with khong_co_bo_danh_gia_du (S05 v3.7 §4.1)')
    if not gain_set:
        raise LogError('--gain-set is needed')
    if gain_owner(gain_set) != campaign:
        raise LogError(f'--gain-set {gain_set!r}: prefix must be {campaign}/ (S05 v3.7 §6.3)')
    if not os.path.isfile(gain_file):
        raise LogError(f'--gain-file {gain_file!r}: no such file (the frozen set, saved bit-exact)')
    base = os.path.basename(gain_file)
    if 'gain_' not in base or not base.endswith('.json'):
        # L81 (Liam 11/11): .gitattributes keeps 05_DATA/**/*gain_*.json byte-exact (-text), so the
        # sha256 still matches after a checkout; another name would get its line ends converted
        raise LogError(f'--gain-file {base!r}: name must contain "gain_" and end in .json (.gitattributes L81)')
    if any(r['campaign'] == campaign for r in read_rows(os.path.join(ds, FREEZE))):
        raise LogError(f'{campaign} already frozen')
    row = {'campaign': campaign, 'frozen_at': at or _iso(now()), 'stop_reason': stop_reason,
           'stop_cause': stop_cause, 'gain_set_id': gain_set, 'gain_file': gain_file.replace('\\', '/'),       # same text on any OS
           'gain_sha256': sha256_file(gain_file), 'notes': notes}
    append_row(os.path.join(ds, FREEZE), FREEZE_COLS, row)
    return row


# ── summary (S05 §6.3) ─────────────────────────────────────────────────────
SUMMARY_ROWS = ('A-PID', 'A-LADRC', 'B-PID', 'B-LADRC', 'B-chung', 'RQ2-INIT-PID', 'RQ2-INIT-LADRC', 'RQ2',
                'C_chung', 'T_H')
SUMMARY_HEAD = ['campaign', 'T_operator_h', 'T_operator_no_hoc_h', 'T_machine_h', 'T_wall_h', 'N_SIM', 'N_HIL',
                'N_HW', 'N_hong', 'T_sua_h', 'stop_reason', 'stop_cause', 'stop_reason_check', 'cap_hit',
                'T_tune_no_hoc_h', 'N_HW_tune', 'gate_result', 'gate_sha256', 'T_wall_end_src', 'B_chung_share',
                'B_chung_after_freeze_h', 'hoc_nhap_tay_n', 'explore_T_operator_h', 'explore_N_SIM',
                'explore_N_HIL', 'explore_N_HW']
VALID = ('dat', 'khong_dat', 'chua_ket_luan')
_COUNT = {'dat': 'pass', 'khong_dat': 'fail', 'chua_ket_luan': 'inconclusive', 'hong_he_do': 'invalid'}
_SAME_CONDITION = ('campaign', 'platform', 'stage', 'trajectory', 'speed_mps', 'gain_set_id')


def read_gate(path: str):
    """EA-9: the gate is read from its locked file (one figure-8 level per row, header `level`),
    never typed again; returns (levels, sha256 of the file)."""
    if not os.path.isfile(path):
        raise LogError(f'gate file {path!r}: no such file')
    levels = [r.get('level', '').strip() for r in read_rows(path)]
    if not levels or any(lv not in FIG8 for lv in levels) or len(set(levels)) != len(levels):
        raise LogError(f'gate file {path!r}: column "level", figure-8 names {sorted(FIG8)}, each once')
    return levels, sha256_file(path)


def gate_result(runs_by_task: Dict[str, List[str]], gate: Iterable[str]) -> str:
    """S05 §4.7: 'dat' = every gate level has FINAL_N valid runs, all 'dat'; 'khong_dat' = a valid
    run of a gate level did not pass; else 'chua_ket_luan' (runs missing, or inconclusive ones)."""
    gate = list(gate)
    outs = {t: [o for o in runs_by_task.get(t, []) if o in VALID] for t in gate}
    if any('khong_dat' in o for o in outs.values()):
        return 'khong_dat'
    if gate and all(len(o) == FINAL_N and all(x == 'dat' for x in o) for o in outs.values()):
        return 'dat'
    return 'chua_ket_luan'


def _active_s(r: dict) -> float:
    return sum(_f(r.get(f't_{k}_s')) for k in STEPS)


def _b_target(controller: str) -> Optional[str]:
    c = (controller or '').upper()
    return {'PID': 'B-PID', 'LADRC': 'B-LADRC'}.get(c)


def _effort_span(r: dict):
    """(start, end) of an effort row; end past midnight is the next day."""
    if not (r.get('ngay') and r.get('bat_dau')):
        return None, None
    t0 = dt.datetime.fromisoformat(f"{r['ngay']}T{r['bat_dau']}")
    if not r.get('ket_thuc'):
        return t0, None
    t1 = dt.datetime.fromisoformat(f"{r['ngay']}T{r['ket_thuc']}")
    if t1 <= t0:
        t1 += dt.timedelta(days=1)
    return t0, t1


def _check_replacements(tun: List[dict]) -> None:
    """S05 §4.3, §4.6, §4.7 (EA-8): a replacement points to a lost run (hong_he_do) of the same
    campaign, platform, stage, task, speed and gain set, earlier than itself; a run is replaced once."""
    by_uid = {r['run_uid']: r for r in tun if r.get('run_uid')}
    seen = {}
    for r in tun:
        ref = r.get('replaces_run_uid')
        if not ref:
            continue
        old = by_uid.get(ref)
        if not old or old.get('run_outcome') != 'hong_he_do':
            raise LogError(f"{r.get('run_uid')}: replaces_run_uid {ref} is not a hong_he_do run (§4.7)")
        diff = [k for k in _SAME_CONDITION if old.get(k, '') != r.get(k, '')]
        if diff:
            raise LogError(f"{r.get('run_uid')}: replaces {ref} with another {', '.join(diff)} (a replacement "
                           f"runs the same condition, §4.3, §4.6)")
        if ref == r.get('run_uid') or (old.get('wall_start', '') > r.get('wall_start', '')):
            raise LogError(f"{r.get('run_uid')}: replaces {ref}, which is not an earlier run")
        if ref in seen:
            raise LogError(f"{ref} is replaced twice ({seen[ref]}, {r.get('run_uid')})")
        seen[ref] = r.get('run_uid')


def _final_stops(tun: List[dict]) -> set:
    """S05 §4.6 (EA-8): after FINAL_LOST_N lost runs in a row in one condition of one gain set
    (per platform), the condition stops; a later run of it is an error. Returns the stopped
    conditions as (stage, campaign, gain_set_id, platform, task, speed)."""
    groups = defaultdict(list)
    for r in tun:
        if r.get('stage') in ('final', 'final_hil'):
            groups[(r['stage'], r.get('campaign', ''), r.get('gain_set_id', ''), r.get('platform', ''),
                    r.get('trajectory', ''), r.get('speed_mps', ''))].append(r)
    stopped = set()
    for key, runs in groups.items():
        # S315-R2-3: the order of the log rows (as map_refusals, S315-4, and final_lost_streak), not wall_start
        lost = 0
        for i, r in enumerate(runs):
            if key in stopped:
                raise LogError(f"{r.get('run_uid')}: {key[4]}@{key[5]} of {key[2] or key[1]} stopped after "
                               f"{FINAL_LOST_N} lost runs in a row (§4.6), no run after that")
            lost = lost + 1 if r.get('run_outcome') == 'hong_he_do' else 0
            if lost >= FINAL_LOST_N:
                stopped.add(key)
    return stopped


def summary(ds: str, gate: Optional[Iterable[str]] = None, gate_sha256: str = '') -> List[dict]:
    """S05_summary.csv rows (S05 v3.7). Accounts (§3.2):
      * a tuning row counts for its `campaign`; an effort row for `chien_dich`, or `khoan` when it is empty;
        a hardware-failure row for `chien_dich` (B-chung split like the effort rows);
      * B-chung (§3.2 [v3.6]): a row whose controller is PID / LADRC (rows of one script run split by
        controller, S5-23 v2) goes whole to that B; a row without controller is split in two. The
        B-chung line reports the unsplit total, N_SIM included (EA-4); B_chung_after_freeze_h = the
        share that came after that B froze (§3.2 [S5-16 v2]);
      * T_operator = active time of the tuning rows + hours of the effort rows + repair hours T_sua
        (S05 v3.7 §3.3, EA-1); no_hoc without nhan=hoc; T_sua_h reports the repair part;
      * N_SIM of a SIM row = its n_sim_evals, 1 when the field is empty (one live fwsim run), 0 when 0;
      * explore (S05 v3.7 §4.8, EA-17): out of T, N, caps and T_wall; reported in explore_* only;
      * T_wall = first time stamp of the campaign to its frozen_at (S05_freeze.csv), else to its last one;
      * caps (§4.1 [v3.6]): N_HW_tune = car runs of stage screen/confirm (lost-to-measurement runs of
        the tune included); T_tune_no_hoc = active hours (repairs included) before the campaign's
        frozen_at, without hoc and stage final/final_hil; B-chung shares likewise before the freeze of
        that B. A row (tuning or effort) that starts before a freeze it counts for and ends after it
        is an error: split it at frozen_at (EA-3). [CÁCH HIỂU] a repair row has a date, no clock
        time: it counts before a freeze unless its date is after the freeze date;
      * stop_reason_check: cham_tran_xe / cham_tran_gio (or khong_co_bo_danh_gia_du with that
        stop_cause) checked against the counts above ('ok' / 'mismatch'), the other codes 'khong_kiem_duoc';
      * final (§4.7 [v3.6]): final_<task>@<speed>_* by run_outcome (hong_he_do out of n_valid);
        final_hil rows are account RQ2 (§4.6) and their final_hil_<task>@<speed>_* columns go on the row
        of the campaign whose frozen gain_set_id they carry, or of RQ2-INIT-* by the prefix of the id;
        any other id is an error (EA-5). <cond>_stopped_n3 = 1 when the condition stopped after 3 lost
        runs in a row (§4.6). Errors (LogError): a bad replacement (EA-8), more than FINAL_N valid
        runs, a run after the stop, a final_hil row not account RQ2, a final car run with another gain
        set than the frozen one, a figure-8 task whose name and speed disagree (EA-9);
      * refused mappings (S05 v3.15 4.5 step 2): hil_map_refused_n, hil_map_stopped (gain set task@speed
        of each level stopped after MAP_RERUNS_MAX reruns, chua_ket_luan) on the row of a campaign that has
        any; errors (check_map_rows, map_refusals): a run of a stopped level, a refused-mapping row that is not
        a HIL hong_he_do row with a g1,g2 gate, a HIL row judged without its gate, a counted HIL row without
        gain set id or with other gain values than the earlier runs of its key;
      * gate_result: gate_result() over the final runs of the gate levels (`gate`, read from the
        locked gate file by read_gate); empty without `gate` or without any final run."""
    tun = live_rows(read_rows(os.path.join(ds, TUNING)))
    eff = read_rows(os.path.join(ds, EFFORT))
    hw = read_rows(os.path.join(ds, HWFAIL))
    frz = {r['campaign']: r for r in read_rows(os.path.join(ds, FREEZE))}
    by_gain = {}
    for r in frz.values():
        gs = r.get('gain_set_id', '')
        if gs:
            if gain_owner(gs) != r['campaign'] or gs in by_gain:
                raise LogError(f"S05_freeze.csv: gain_set_id {gs!r} of {r['campaign']} (prefix = campaign, unique)")
            by_gain[gs] = r['campaign']

    acc = defaultdict(lambda: defaultdict(float))
    times = defaultdict(list)
    finals = defaultdict(lambda: defaultdict(int))
    final_tasks = defaultdict(lambda: defaultdict(list))      # campaign -> task -> outcomes (stage final)
    unassigned = 0

    for r in tun:
        if r.get('stage') == 'final_hil' and r.get('campaign') != 'RQ2':
            raise LogError(f"{r.get('run_uid')}: a final_hil run is account RQ2 (§4.6)")
        if r.get('platform') in ('HIL', 'HW'):
            try:
                check_task(r.get('trajectory', ''), r.get('speed_mps', ''))
            except LogError as exc:
                raise LogError(f"{r.get('run_uid')}: {exc}")
    _check_replacements(tun)
    check_gate_rows(tun)                            # GV3-1, GV3-R2-1: gate rows recomputed from q, lo, hi
    stopped = _final_stops(tun)
    check_map_rows(tun)                             # S05 v3.15 4.5 step 2 (Codex S315-1)
    map_n = map_refusals(tun)                       # refused mappings, levels stopped
    map_rep = defaultdict(lambda: {'n': 0, 'stopped': []})
    for k, n in map_n.items():
        if not n:
            continue
        rep = map_rep[k[0]]                         # map_key: campaign, gain set, task, speed
        rep['n'] += n
        if map_stopped(n):
            rep['stopped'].append(f'{k[1]} {k[2]}@{k[3]}')

    def add(target, key, val, share=1.0):
        acc[target][key] += val * share

    def frozen_before(target, t: Optional[dt.datetime]) -> bool:
        """True when `target` froze at or before t (work from frozen_at on: not in its caps)."""
        f = frz.get(target)
        return bool(f and t and t >= _parse_t(f['frozen_at']))

    def spans_freeze(target, t0, t1, what) -> None:
        f = frz.get(target)
        if f and t0 and t1 and t0 < _parse_t(f['frozen_at']) < t1:
            raise LogError(f"{what}: runs from {_iso(t0)} to {_iso(t1)}, across the freeze of {target} at "
                           f"{f['frozen_at']}; split it at frozen_at (S05 §4.1 caps, EA-3)")

    def b_split(c, controller=''):
        if c != 'B-chung':
            return [(c, 1.0)]
        tgt = _b_target(controller)
        return [(tgt, 1.0)] if tgt else [('B-PID', 0.5), ('B-LADRC', 0.5)]

    for r in tun:
        c = r.get('campaign', '')
        if not c:
            unassigned += 1
            continue
        t0 = _parse_t(r['wall_start']) if r.get('wall_start') else None
        t1 = _parse_t(r['wall_end']) if r.get('wall_end') else None
        stage = r.get('stage', '')
        act, mach = _active_s(r), _f(r.get('t_machine_s'))
        plat = r.get('platform')
        n_sim = (_f(r.get('n_sim_evals')) if r.get('n_sim_evals', '') != '' else 1.0) if plat == 'SIM' else 0.0
        targets = b_split(c, r.get('controller'))
        if stage == 'explore':                          # S05 v3.7 §4.8: reported apart only
            for tgt, share in targets:
                add(tgt, 'X_T_s', act, share)
                add(tgt, 'X_N_SIM', n_sim, share if c != 'B-chung' else 1.0)
                add(tgt, {'HIL': 'X_N_HIL', 'HW': 'X_N_HW'}.get(plat, 'X_none'), 1.0 if plat != 'SIM' else 0.0)
            continue
        if c == 'B-chung':
            add('B-chung', 'T_op_s', act)
            add('B-chung', 'T_noh_s', act)
            add('B-chung', 'T_mach_s', mach)
            add('B-chung', 'N_SIM', n_sim)
            for t in (t0, t1):
                if t:
                    times['B-chung'].append(t)
        for tgt, share in targets:
            add(tgt, 'T_op_s', act, share)
            add(tgt, 'T_noh_s', act, share)
            add(tgt, 'T_mach_s', mach, share)
            if stage not in ('final', 'final_hil'):
                spans_freeze(tgt, t0, t1, r.get('run_uid', 'a tuning row'))
                if not frozen_before(tgt, t0):
                    add(tgt, 'T_tune_s', act, share)
                elif c == 'B-chung':
                    add(tgt, 'Bc_after_s', act, share)
            if plat == 'SIM':
                add(tgt, 'N_SIM', n_sim, share if c != 'B-chung' else 1.0)
            elif plat == 'HIL':
                add(tgt, 'N_HIL', 1.0)
            elif plat == 'HW':
                add(tgt, 'N_HW', 1.0)
                if stage in TUNE_STAGES:
                    add(tgt, 'N_HW_tune', 1.0)
            for t in (t0, t1):
                if t:
                    times[tgt].append(t)
        if stage in ('final', 'final_hil'):
            task, o, sp = r.get('trajectory', ''), r.get('run_outcome', ''), r.get('speed_mps', '')
            gs = r.get('gain_set_id', '')
            if stage == 'final':
                home, cond = c, f"final_{task}@{sp}"
                frozen_gs = (frz.get(c) or {}).get('gain_set_id', '')
                if gs and frozen_gs and gs != frozen_gs:
                    raise LogError(f"{r.get('run_uid')}: final run of {c} with gain set {gs}, frozen is {frozen_gs}")
                final_tasks[c][task].append(o)
            else:
                owner = gain_owner(gs) if gs else ''
                if gs in by_gain:
                    home = by_gain[gs]
                elif owner in ('RQ2-INIT-PID', 'RQ2-INIT-LADRC'):
                    home = owner
                else:
                    raise LogError(f"{r.get('run_uid')}: final_hil gain set {gs!r} is no frozen set and no "
                                   f"RQ2-INIT-* set (S05 v3.7 §6.3, EA-5)")
                cond = f"final_hil_{task}@{sp}"
            f = finals[home]
            if o in VALID:
                f[f'{cond}_n_valid'] += 1
                if f[f'{cond}_n_valid'] > FINAL_N:
                    raise LogError(f'{home} {cond}: more than {FINAL_N} valid runs (§4.7)')
            f[f'{cond}_' + _COUNT.get(o, 'invalid')] += 1
            key = (stage, c, gs, r.get('platform', ''), task, sp)
            if key in stopped:
                f[f'{cond}_stopped_n3'] = 1

    for r in eff:
        c, k = r.get('chien_dich', ''), r.get('khoan', '')
        if not c and not k:
            continue                       # rows before S05 §6.2 (no account): not in any line
        h = _f(r.get('gio')) * 3600
        hoc = r.get('nhan') == 'hoc'
        t, t_end = _effort_span(r)
        if c == 'B-chung':
            add('B-chung', 'T_op_s', h)
            if not hoc:
                add('B-chung', 'T_noh_s', h)
        targets = b_split(c) if c else [(k, 1.0)]
        for tgt, share in targets:
            add(tgt, 'T_op_s', h, share)
            if hoc:
                if r.get('ghi_chu', '').startswith(MANUAL):
                    add(tgt, 'hoc_manual', 1.0)
            else:
                add(tgt, 'T_noh_s', h, share)
                spans_freeze(tgt, t, t_end, f"effort row {r.get('ngay')} {r.get('bat_dau')}")
                if not frozen_before(tgt, t):
                    add(tgt, 'T_tune_s', h, share)
                elif c == 'B-chung':
                    add(tgt, 'Bc_after_s', h, share)
            if t:
                times[tgt].append(t)

    for r in hw:
        c = r.get('chien_dich', '')
        if not c:
            continue
        h = _f(r.get('gio_sua')) * 3600
        day = dt.datetime.fromisoformat(r['ngay']) if r.get('ngay') else None
        add(c, 'N_hong', 1.0)
        add(c, 'T_sua_s', h)
        if c == 'B-chung':
            add('B-chung', 'T_op_s', h)
            add('B-chung', 'T_noh_s', h)
        for tgt, share in b_split(c):
            if c == 'B-chung':
                add(tgt, 'T_sua_s', h, share)
            add(tgt, 'T_op_s', h, share)
            add(tgt, 'T_noh_s', h, share)
            f = frz.get(tgt)
            after = bool(f and day and day.date() > _parse_t(f['frozen_at']).date())   # [CÁCH HIỂU] date only
            if not after:
                add(tgt, 'T_tune_s', h, share)
            elif c == 'B-chung':
                add(tgt, 'Bc_after_s', h, share)

    out = []
    for c in SUMMARY_ROWS:
        a = acc.get(c, {})
        ts = sorted(times.get(c, []))
        f = frz.get(c)
        t_end, src = (None, '')
        if f:
            t_end, src = _parse_t(f['frozen_at']), 'freeze'
        elif ts:
            t_end, src = ts[-1], 'last_row'
        wall = (t_end - ts[0]).total_seconds() / 3600 if ts and t_end else 0.0
        cap = []
        if c in TUNED:
            if a.get('N_HW_tune', 0) >= CAP_HW:
                cap.append('xe')
            if a.get('T_tune_s', 0) / 3600 >= CAP_H:
                cap.append('gio')
        reason = (f or {}).get('stop_reason', '')
        cause = (f or {}).get('stop_cause', '')
        basis = cause if reason == 'khong_co_bo_danh_gia_du' else reason
        check = ''
        if basis == 'cham_tran_xe':
            check = 'ok' if 'xe' in cap else 'mismatch'
        elif basis == 'cham_tran_gio':
            check = 'ok' if 'gio' in cap else 'mismatch'
        elif reason:
            check = 'khong_kiem_duoc'
        if check == 'mismatch':
            print(f'warning: {c} stop_reason {reason} {cause} does not match the logs (cap_hit {"+".join(cap) or "none"})',
                  file=sys.stderr)
        gres = gate_result(final_tasks[c], gate) if gate is not None and final_tasks.get(c) else ''
        row = {'campaign': c, 'T_operator_h': _fmt(a.get('T_op_s', 0) / 3600, 3),
               'T_operator_no_hoc_h': _fmt(a.get('T_noh_s', 0) / 3600, 3),
               'T_machine_h': _fmt(a.get('T_mach_s', 0) / 3600, 3), 'T_wall_h': _fmt(wall, 3),
               'N_SIM': str(int(round(a.get('N_SIM', 0)))), 'N_HIL': str(int(a.get('N_HIL', 0))),
               'N_HW': str(int(a.get('N_HW', 0))), 'N_hong': str(int(a.get('N_hong', 0))),
               'T_sua_h': _fmt(a.get('T_sua_s', 0) / 3600, 3), 'stop_reason': reason, 'stop_cause': cause,
               'stop_reason_check': check, 'cap_hit': '+'.join(cap),
               'T_tune_no_hoc_h': _fmt(a.get('T_tune_s', 0) / 3600, 3),
               'N_HW_tune': str(int(a.get('N_HW_tune', 0))), 'gate_result': gres,
               'gate_sha256': gate_sha256 if gres else '', 'T_wall_end_src': src,
               'B_chung_share': 'unsplit total' if c == 'B-chung' else '',
               'B_chung_after_freeze_h': _fmt(a.get('Bc_after_s', 0) / 3600, 3) if c in ('B-PID', 'B-LADRC') else '',
               'hoc_nhap_tay_n': str(int(a.get('hoc_manual', 0))),
               'explore_T_operator_h': _fmt(a.get('X_T_s', 0) / 3600, 3),
               'explore_N_SIM': str(int(round(a.get('X_N_SIM', 0)))), 'explore_N_HIL': str(int(a.get('X_N_HIL', 0))),
               'explore_N_HW': str(int(a.get('X_N_HW', 0)))}
        row.update({k: str(v) for k, v in sorted(finals.get(c, {}).items())})
        if c in map_rep:                            # extra columns, only for a campaign with refused mappings
            row['hil_map_refused_n'] = str(map_rep[c]['n'])
            row['hil_map_stopped'] = '; '.join(sorted(map_rep[c]['stopped']))
        out.append(row)
    if unassigned:
        print(f'note: {unassigned} tuning row(s) without campaign left out', file=sys.stderr)
    return out


def write_summary(ds: str, out_path: str, gate: Optional[Iterable[str]] = None, gate_sha256: str = '') -> List[dict]:
    rows = summary(ds, gate, gate_sha256)
    extra = sorted({k for r in rows for k in r} - set(SUMMARY_HEAD))
    _write_all(out_path, SUMMARY_HEAD + extra, rows)
    return rows


# ── C9 dry run ─────────────────────────────────────────────────────────────
def _plain(o):
    """A gate of o1_gate as the scorer writes it to JSON (numpy -> float / int / bool, NaN -> None)."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, 'item') and not isinstance(o, (str, bytes)):
        o = o.item()
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def fake_gate(e_trk: float = 0.03, mapped: bool = True) -> dict:
    """A made-up gate (FAKE: demo and tests only, no measurement): 20 frames of constant errors, e_trk as given;
    mapped: U_t known and all five decide, else the HIL set without the run_active edge mapping (G1, G2)."""
    n = 20
    frames = {'e_perp': [0.02] * n, 'e_trk': [e_trk] * n, 'e_bao': [0.04] * n, 'e_theta_cmd_deg': [3.0] * n}
    return _plain(o1_gate.gate(frames, 'true', v_cmd_max=0.729, wz_cmd_max=1.634, u_t=0.0015 if mapped else None,
                               decide=o1_gate.DECIDE_ALL if mapped else o1_gate.DECIDE_HIL, note='FAKE'))


def demo(out_dir: str, seed: int = 1) -> List[dict]:
    """A made-up A-PID, B-chung, B-PID, final session, written through the same commands, then
    the summary. Every number is invented (labelled FAKE); nothing here is a measurement."""
    global now
    os.makedirs(out_dir, exist_ok=False)
    rnd = random.Random(seed)
    clock = [dt.datetime(2026, 11, 20, 8, 0, 0)]
    real_now = now
    now = lambda: clock[0]                            # noqa: E731
    try:
        migrate(out_dir)

        def tick(s):
            clock[0] += dt.timedelta(seconds=s)

        clock[0] -= dt.timedelta(minutes=30)
        hoc_start(out_dir, 'A-PID', 'FAKE: doc huong dan tune')     # stamps by the script (EA-7)
        tick(1800)
        hoc_end(out_dir)
        for i, sp in enumerate(('0.2', '0.5', '0.8')):
            begin(out_dir, 'FAKE-A1', 'A-PID', 'HW', 'PID', 'O1-F0' + sp[-1], sp, 'screen', gain_set='A-PID/G00',
                  gains='kp_xy=1.0,ki_xy=0.5,kp_wz=1.0,ki_wz=0.2,pos_kp=6,pos_ki=3,k_psi=1')
            for k, s in (('flash', 120), ('setup', 300), ('run', 60), ('log', 30), ('analysis', 240)):
                step(out_dir, k, wait=(k == 'run'))
                tick(s + rnd.randrange(10))
            pause(out_dir)
            tick(600)                                  # other work: not counted
            resume(out_dir)
            tick(60)
            end(out_dir, 'khong_dat' if i else 'hong_he_do', completed=1, e_perp_max=0.12 + i / 100,
                fail_cause='' if i else 'mat_khung_FAKE')
        gfile = os.path.join(out_dir, 'FAKE_gain_G00.json')
        with open(gfile, 'w', encoding='utf-8', newline='\n') as f:
            f.write('{"FAKE": true, "kp_xy": 1.0, "ki_xy": 0.5, "kp_wz": 1.0, "ki_wz": 0.2, '
                    '"pos_kp": 6, "pos_ki": 3, "k_psi": 1}\n')
        freeze(out_dir, 'A-PID', 'khong_du_luot_xac_nhan', 'A-PID/G00', gfile, at=_iso(clock[0]),
               notes='FAKE')
        machine(out_dir, 'B-chung', 'PID', 'FAKE-S03', 'results/S03/FAKE', 2739, 5400, _iso(clock[0]),
                _iso(clock[0] + dt.timedelta(seconds=5400)), t_setup_s=600)
        machine(out_dir, 'B-chung', 'LADRC', 'FAKE-S03', 'results/S03/FAKE', 2739, 5400, _iso(clock[0]),
                _iso(clock[0] + dt.timedelta(seconds=5400)))
        effort(out_dir, '2026-11-20', '13:00', '14:00', campaign='B-chung', ghi_chu='FAKE: doc S03 chung')
        effort(out_dir, '2026-11-20', '14:00', '14:30', khoan='C_chung', ghi_chu='FAKE: script video')
        tick(7200)
        begin(out_dir, 'FAKE-B1', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1')
        step(out_dir, 'run')
        tick(90)                                        # S05 v3.15 4.5 step 2: mapping refused, run again
        end(out_dir, 'hong_he_do', completed=1, fail_cause=MAP_REFUSED, results_dir='results/FAKE/b1_1',
            gate=fake_gate(mapped=False))
        begin(out_dir, 'FAKE-B1', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1')
        step(out_dir, 'run')
        tick(90)
        end(out_dir, 'dat', completed=1, e_perp_max=0.04, results_dir='results/FAKE/b1_2', gate=fake_gate())
        hwfail(out_dir, '2026-11-20', 'FAKE-B1', 'B-PID', 'HW', 'FAKE: tuot banh', 'phan_cung_ngau_nhien', 0.5)
        for k in range(2):                              # final on HIL: account RQ2, joined by gain set
            begin(out_dir, 'FAKE-FH', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='A-PID/G00')
            step(out_dir, 'run')
            tick(75)
            end(out_dir, 'dat', completed=1, e_perp_max=0.05, results_dir=f'results/FAKE/fh_{k}', gate=fake_gate())
        lost = None
        for o in ('hong_he_do', 'dat', 'chua_ket_luan'):  # final on the car; the lost run is replaced
            begin(out_dir, 'FAKE-F1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G00',
                  replaces_run_uid=lost if o == 'dat' else '')
            step(out_dir, 'run')
            tick(60)
            row = end(out_dir, o, completed=1, e_perp_max=0.10, fail_cause='mat_khung_FAKE' if o == 'hong_he_do' else '')
            if o == 'hong_he_do':
                lost = row['run_uid']
        gate_path = os.path.join(out_dir, 'FAKE_gate.csv')            # stands for the gate locked at C8
        with open(gate_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('level\nO1-F05\n')
        gate, sha = read_gate(gate_path)
        return write_summary(out_dir, os.path.join(out_dir, SUMMARY), gate=gate, gate_sha256=sha)
    finally:
        now = real_now


# ── CLI ────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='RQ3 logs (spec S05 §6, C9)')
    ap.add_argument('--dataset', default=os.environ.get('RQ3_DATASET', DEFAULT_DATASET))
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('migrate')
    b = sub.add_parser('begin')
    for name in ('session', 'campaign', 'platform', 'controller', 'task', 'speed', 'stage'):
        b.add_argument('--' + name, required=True)
    for name in ('gain-set', 'gains', 'block-id', 'replaces', 'candidate', 'fw-hash', 'rule-version', 'dut', 'buoc',
                 'notes'):
        b.add_argument('--' + name, default='')
    s = sub.add_parser('step')
    s.add_argument('kind', choices=STEPS)
    s.add_argument('--wait', action='store_true')
    sub.add_parser('pause')
    sub.add_parser('resume')
    sub.add_parser('cancel')
    sub.add_parser('status')
    e = sub.add_parser('end')
    e.add_argument('--outcome', required=True, choices=OUTCOMES)
    e.add_argument('--completed', type=int, choices=(0, 1))
    e.add_argument('--e-perp-max', type=float)
    e.add_argument('--gate-json', default='', help='scorer output with a gate_v3 block (hil_o1_score.py --out)')
    for name in ('fail-cause', 'results-dir', 'notes'):
        e.add_argument('--' + name, default='')
    m = sub.add_parser('machine')
    for name in ('campaign', 'controller', 'script-run-id', 'results-dir', 'wall-start', 'wall-end'):
        m.add_argument('--' + name, required=True)
    m.add_argument('--n-sim-evals', type=int, required=True)
    m.add_argument('--t-machine-s', type=float, required=True)
    m.add_argument('--platform', default='SIM', choices=PLATFORMS)
    m.add_argument('--stage', default='script', choices=STAGES)
    m.add_argument('--notes', default='')
    for k in STEPS:
        m.add_argument(f'--t-{k}-s', type=float)
    hs = sub.add_parser('hoc-start')
    hs.add_argument('--campaign', required=True, choices=TUNED)
    hs.add_argument('--ghi-chu', required=True)
    he = sub.add_parser('hoc-end')
    he.add_argument('--ghi-chu', default='')
    f = sub.add_parser('effort')
    for name in ('date', 'start', 'end'):
        f.add_argument('--' + name, required=True)
    for name in ('campaign', 'khoan', 'nhan', 'ghi-chu', 'buoc', 'giai-doan', 'dut', 'lap-lai'):
        f.add_argument('--' + name, default='')
    h = sub.add_parser('hwfail')
    for name in ('date', 'session', 'campaign', 'platform', 'what', 'cause'):
        h.add_argument('--' + name, required=True)
    h.add_argument('--fix-h', type=float, required=True)
    for name in ('iter-no', 'notes', 'dut', 'buoc', 'lap-lai'):
        h.add_argument('--' + name, default='')
    z = sub.add_parser('freeze')
    z.add_argument('--campaign', required=True, choices=TUNED)
    z.add_argument('--stop-reason', required=True, choices=STOP_REASONS)
    for name in ('gain-set', 'gain-file'):
        z.add_argument('--' + name, required=True)
    for name in ('at', 'notes'):
        z.add_argument('--' + name, default='')
    z.add_argument('--stop-cause', default='', choices=('',) + STOP_CAUSES)
    o = sub.add_parser('summary')
    o.add_argument('--out', default='')
    o.add_argument('--gate-file', default=None, help='the locked gate (column "level": O1-F05, ...; locked at C8)')
    d = sub.add_parser('demo')
    d.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    ds = a.dataset
    try:
        if a.cmd == 'migrate':
            print(json.dumps(migrate(ds), indent=1))
        elif a.cmd == 'begin':
            run = begin(ds, a.session, a.campaign, a.platform, a.controller, a.task, a.speed, a.stage,
                        gain_set=a.gain_set, gains=a.gains, block_id=a.block_id, replaces_run_uid=a.replaces,
                        candidate_no=a.candidate, fw_hash=a.fw_hash, rule_version=a.rule_version, dut=a.dut,
                        buoc=a.buoc, notes=a.notes)
            print('open', run['row']['run_uid'], run['row']['wall_start'])
        elif a.cmd == 'step':
            run = step(ds, a.kind, a.wait)
            print('step', a.kind, run['current']['start'])
        elif a.cmd in ('pause', 'resume'):
            (pause if a.cmd == 'pause' else resume)(ds)
            print(a.cmd, _iso(now()))
        elif a.cmd == 'cancel':
            run = cancel(ds)
            eff = run.get('effort_row')
            print('cancelled', run['row']['run_uid'], '(no run row)',
                  f"effort row {eff['gio']} h" if eff else 'no time logged')
        elif a.cmd == 'status':
            run = _load_open(ds)
            if not run:
                print('no open run')
            else:
                segs = run['segments'] + ([dict(run['current'], end=_iso(now()))] if run.get('current') else [])
                print(run['row']['run_uid'], 'paused' if run.get('paused_from') is not None else 'running',
                      json.dumps(segment_totals(segs)))
        elif a.cmd == 'end':
            row = end(ds, a.outcome, a.completed, a.e_perp_max, a.fail_cause, a.results_dir, a.notes,
                      gate=read_gate_json(a.gate_json) if a.gate_json else None,
                      gate_sha256=sha256_file(a.gate_json) if a.gate_json else '')
            print('row', row['run_uid'], {k: row[f't_{k}_s'] for k in STEPS}, 'wait', row['t_operator_wait_s'])
            note = refusal_note(ds, row)
            if note:
                print(note)
        elif a.cmd == 'machine':
            row = machine(ds, a.campaign, a.controller, a.script_run_id, a.results_dir, a.n_sim_evals, a.t_machine_s,
                          a.wall_start, a.wall_end, a.platform, a.stage, notes=a.notes,
                          **{f't_{k}_s': getattr(a, f't_{k}_s') for k in STEPS})
            print('row', row['run_uid'])
        elif a.cmd == 'hoc-start':
            print('hoc open', hoc_start(ds, a.campaign, a.ghi_chu)['start'])
        elif a.cmd == 'hoc-end':
            row = hoc_end(ds, a.ghi_chu)
            print('row', row['ngay'], row['bat_dau'], row['ket_thuc'], row['gio'], 'h')
        elif a.cmd == 'effort':
            row = effort(ds, a.date, a.start, a.end, a.campaign, a.khoan, a.nhan, a.ghi_chu, buoc=a.buoc,
                         giai_doan=a.giai_doan, dut=a.dut, lap_lai=a.lap_lai)
            print('row', row['ngay'], row['gio'], 'h')
        elif a.cmd == 'hwfail':
            hwfail(ds, a.date, a.session, a.campaign, a.platform, a.what, a.cause, a.fix_h, a.iter_no, a.notes,
                   dut=a.dut, buoc=a.buoc, lap_lai=a.lap_lai)
            print('row written')
        elif a.cmd == 'freeze':
            row = freeze(ds, a.campaign, a.stop_reason, a.gain_set, a.gain_file, a.at, a.notes, a.stop_cause)
            print('frozen', a.campaign, row['gain_sha256'])
        elif a.cmd == 'summary':
            gate, sha = read_gate(a.gate_file) if a.gate_file else (None, '')
            rows = write_summary(ds, a.out or os.path.join(ds, SUMMARY), gate, sha)
            for r in rows:
                print(r['campaign'], r['T_operator_h'], 'h', 'N_HW', r['N_HW'], 'gate', r['gate_result'] or '-')
        elif a.cmd == 'demo':
            for r in demo(a.out):
                print(json.dumps(r, ensure_ascii=False))
    except LogError as exc:
        print('rq3_log:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
