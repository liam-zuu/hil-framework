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
         [--fail-cause X] [--results-dir D]   close the run, append its row
  cancel / status                        drop the open run without a row / show it
  machine --campaign C --controller PID --script-run-id ID --results-dir D --n-sim-evals N
         --t-machine-s S --wall-start T --wall-end T   summary row of a fwsim script run
  effort --date D --start HH:MM --end HH:MM (--campaign C | --khoan K) [--nhan hoc] --ghi-chu X
  hwfail --date D --session S --campaign C --platform HW --what X --cause gain|... --fix-h H
  freeze --campaign C --stop-reason R --gain-set G --gain-file F [--at T]   (sha256 of F computed)
  summary [--gate O1-F05,O1-F08] [--out S05_summary.csv]   recompute the summary from the logs;
                                         --gate = figure-8 levels inside the gate (locked at C8)
  demo   --out DIR                       C9 dry run: made-up campaign in DIR, then summary

Rules taken from S05 v3.6 (approved 11/11, not locked): 3.1 (time types, wait, hoc), 3.2
(accounts, B-chung split), 3.3 (counts by platform, stages), 4.1 (caps 10 h, 120 car runs, counted
before frozen_at), 4.6 (final_hil = account RQ2), 4.7 (final counts, gate_result), 6.1-6.3.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.normpath(os.path.join(HERE, '..', '..', '..', '05_DATA', 'M7_dataset'))

TUNING, EFFORT, HWFAIL = 'm7_tuning_log.csv', 'm7_dev_effort_log.csv', 'm7_hw_failure_log.csv'
FREEZE, OPEN_RUN, SUMMARY = 'S05_freeze.csv', 'rq3_open_run.json', 'S05_summary.csv'

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
TUNING_COLS = TUNING_OLD + TUNING_NEW
EFFORT_OLD = ['ngay', 'bat_dau', 'ket_thuc', 'gio', 'buoc', 'giai_doan', 'mot_lan_hay_thich_nghi', 'dut',
              'ghi_chu', 'chien_dich', 'lap_lai']
EFFORT_COLS = EFFORT_OLD + ['khoan', 'nhan']
HWFAIL_COLS = ['ngay', 'session_id', 'chien_dich', 'nen_tang', 'thu_hong', 'nguyen_nhan', 'gio_sua',
               'ghi_chu', 'dut', 'iter_no', 'buoc', 'lap_lai']
FREEZE_COLS = ['campaign', 'frozen_at', 'stop_reason', 'gain_set_id', 'gain_file', 'gain_sha256', 'notes']
STOP_REASONS = ('khoi_xac_nhan_dat', 'cham_tran_gio', 'cham_tran_xe', 'het_ung_vien', 's04_dung',
                'khong_du_luot_xac_nhan', 'khong_co_bo_danh_gia_du')          # S05 v3.6 §4.7
FINAL_N = 5                              # S05 §4.6: 5 valid runs per condition

CAMPAIGNS = ('A-PID', 'A-LADRC', 'B-PID', 'B-LADRC', 'B-chung', 'RQ2-INIT-PID', 'RQ2-INIT-LADRC', 'RQ2')
KHOAN = ('T_H', 'C_chung', 'RQ2')
STAGES = ('screen', 'confirm', 'final', 'explore', 'hil_check', 'final_hil')
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
    Refuses a file with a column S05 does not know (nothing is dropped)."""
    if not os.path.exists(path):
        _write_all(path, cols, [])
        return 'created'
    old = _header(path)
    if old == cols:
        return 'ok'
    extra = [c for c in old if c not in cols]
    if extra:
        raise LogError(f'{os.path.basename(path)}: columns not in S05 §6 {extra}; not migrated')
    rows = read_rows(path)
    _write_all(path, cols, rows)
    return 'migrated'


def migrate(ds: str) -> Dict[str, str]:
    return {name: migrate_file(os.path.join(ds, name), cols)
            for name, cols in ((TUNING, TUNING_COLS), (EFFORT, EFFORT_COLS), (HWFAIL, HWFAIL_COLS))}


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
    if campaign not in CAMPAIGNS:
        raise LogError(f'--campaign {campaign!r}: one of {CAMPAIGNS}')
    if platform not in PLATFORMS or stage not in STAGES:
        raise LogError(f'--platform one of {PLATFORMS}, --stage one of {STAGES}')
    if not session or session.startswith(SKIP_PREFIX) or ':' in session:
        raise LogError('--session: a name of your own, no ":" (run_uid = session_id:iter_no)')
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


def end(ds: str, outcome: str, completed: Optional[int] = None, e_perp_max: Optional[float] = None,
        fail_cause: str = '', results_dir: str = '', notes: str = '') -> dict:
    run = _require_open(ds)
    if outcome not in OUTCOMES:
        raise LogError(f'--outcome one of {OUTCOMES}')
    if outcome == 'hong_he_do' and not fail_cause:
        raise LogError('hong_he_do needs --fail-cause (S05 §6.1)')
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
    if notes:
        row['notes'] = (row.get('notes', '') + ' ' + notes).strip()
    append_row(os.path.join(ds, TUNING), TUNING_COLS, row)
    os.remove(_open_path(ds))
    return row


def cancel(ds: str) -> dict:
    run = _require_open(ds)
    os.remove(_open_path(ds))
    return run


# ── rows not tied to one live run ─────────────────────────────────────────
def machine(ds: str, campaign: str, controller: str, script_run_id: str, results_dir: str, n_sim_evals: int,
            t_machine_s: float, wall_start: str, wall_end: str, platform: str = 'SIM', stage: str = 'explore',
            session: str = 'MACHINE', notes: str = '', **active) -> dict:
    """Summary row of a script run (S5-23 v2): one per (controller, account); n_sim_evals and
    t_machine_s are this row's part, the rows of one script run add up to the run."""
    if campaign not in CAMPAIGNS:
        raise LogError(f'--campaign {campaign!r}: one of {CAMPAIGNS}')
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
    t0 = dt.datetime.fromisoformat(f'{date}T{start}')
    t1 = dt.datetime.fromisoformat(f'{date}T{end_}')
    if t1 <= t0:
        t1 += dt.timedelta(days=1)                      # past midnight
    row = {'ngay': date, 'bat_dau': start, 'ket_thuc': end_, 'gio': f'{(t1 - t0).total_seconds() / 3600:.2f}',
           'chien_dich': campaign, 'khoan': khoan, 'nhan': nhan, 'ghi_chu': ghi_chu}
    row.update({k: v for k, v in extra.items() if v not in (None, '')})
    append_row(os.path.join(ds, EFFORT), EFFORT_COLS, row)
    return row


def hwfail(ds: str, date: str, session: str, campaign: str, platform: str, what: str, cause: str,
           fix_h: float, iter_no: str = '', notes: str = '', **extra) -> dict:
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
           notes: str = '') -> dict:
    """S05_freeze.csv (S05 v3.6 §4.7): one row per campaign when its gains freeze. The gain file is
    the bit-exact copy of the frozen set; its sha256 is computed here, never typed. gate_result is
    not written by hand: summary() computes it from the final runs."""
    if campaign not in CAMPAIGNS:
        raise LogError(f'--campaign one of {CAMPAIGNS}')
    if stop_reason not in STOP_REASONS:
        raise LogError(f'--stop-reason one of {STOP_REASONS}')
    if not gain_set:
        raise LogError('--gain-set is needed')
    if not os.path.isfile(gain_file):
        raise LogError(f'--gain-file {gain_file!r}: no such file (the frozen set, saved bit-exact)')
    if any(r['campaign'] == campaign for r in read_rows(os.path.join(ds, FREEZE))):
        raise LogError(f'{campaign} already frozen')
    row = {'campaign': campaign, 'frozen_at': at or _iso(now()), 'stop_reason': stop_reason,
           'gain_set_id': gain_set, 'gain_file': gain_file.replace('\\', '/'),       # same text on any OS
           'gain_sha256': sha256_file(gain_file), 'notes': notes}
    append_row(os.path.join(ds, FREEZE), FREEZE_COLS, row)
    return row


# ── summary (S05 §6.3) ─────────────────────────────────────────────────────
SUMMARY_ROWS = ('A-PID', 'A-LADRC', 'B-PID', 'B-LADRC', 'B-chung', 'RQ2-INIT-PID', 'RQ2-INIT-LADRC', 'RQ2',
                'C_chung', 'T_H')
SUMMARY_HEAD = ['campaign', 'T_operator_h', 'T_operator_no_hoc_h', 'T_machine_h', 'T_wall_h', 'N_SIM', 'N_HIL',
                'N_HW', 'N_hong', 'T_sua_h', 'stop_reason', 'stop_reason_check', 'cap_hit', 'T_tune_no_hoc_h',
                'N_HW_tune', 'gate_result', 'T_wall_end_src', 'B_chung_share']
VALID = ('dat', 'khong_dat', 'chua_ket_luan')
_COUNT = {'dat': 'pass', 'khong_dat': 'fail', 'chua_ket_luan': 'inconclusive', 'hong_he_do': 'invalid'}


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


def summary(ds: str, gate: Optional[Iterable[str]] = None) -> List[dict]:
    """S05_summary.csv rows (S05 v3.6). Accounts (§3.2):
      * a tuning row counts for its `campaign`; an effort row for `chien_dich`, or `khoan` when it is empty;
      * B-chung (§3.2 [v3.6]): a row whose controller is PID / LADRC (rows of one script run split by
        controller, S5-23 v2) goes whole to that B; a row without controller is split in two. The
        B-chung line reports the unsplit total;
      * T_operator = active time of the tuning rows + hours of the effort rows; no_hoc without nhan=hoc;
      * T_wall = first time stamp of the campaign to its frozen_at (S05_freeze.csv), else to its last one;
      * caps (§4.1 [v3.6]): N_HW_tune = car runs of stage screen/confirm (lost-to-measurement runs of
        the tune included); T_tune_no_hoc = active hours before the campaign's frozen_at, without hoc
        and stage final/final_hil; B-chung shares likewise before the freeze of that B;
      * stop_reason_check: cham_tran_xe / cham_tran_gio checked against the counts above
        ('ok' / 'mismatch'), the other codes 'khong_kiem_duoc';
      * final (§4.7 [v3.6]): final_<task>@<speed>_* by run_outcome (hong_he_do out of n_valid);
        final_hil rows are account RQ2 (§4.6: not in T, caps or N_HIL of a campaign) and their
        final_hil_<task>@<speed>_* columns go on the row of the campaign whose frozen gain_set_id
        they carry (else on the RQ2 row, keyed by gain set). Error (LogError) when replaces_run_uid
        names a run that is not hong_he_do, when a condition has more than FINAL_N valid runs, or
        when a final_hil row is not account RQ2;
      * gate_result: gate_result() over the final runs of the gate levels (`gate` = figure-8 tasks in
        the gate, locked at C8); empty without `gate` or without any final run."""
    tun = [r for r in read_rows(os.path.join(ds, TUNING)) if not r.get('session_id', '').startswith(SKIP_PREFIX)]
    eff = read_rows(os.path.join(ds, EFFORT))
    hw = read_rows(os.path.join(ds, HWFAIL))
    frz = {r['campaign']: r for r in read_rows(os.path.join(ds, FREEZE))}
    by_gain = {r['gain_set_id']: r['campaign'] for r in frz.values() if r.get('gain_set_id')}

    acc = defaultdict(lambda: defaultdict(float))
    times = defaultdict(list)
    finals = defaultdict(lambda: defaultdict(int))
    final_tasks = defaultdict(lambda: defaultdict(list))      # campaign -> task -> outcomes (stage final)
    unassigned = 0

    by_uid = {r['run_uid']: r for r in tun if r.get('run_uid')}
    for r in tun:
        ref = r.get('replaces_run_uid')
        if ref and (ref not in by_uid or by_uid[ref].get('run_outcome') != 'hong_he_do'):
            raise LogError(f"{r.get('run_uid')}: replaces_run_uid {ref} is not a hong_he_do run (§4.7)")
        if r.get('stage') == 'final_hil' and r.get('campaign') != 'RQ2':
            raise LogError(f"{r.get('run_uid')}: a final_hil run is account RQ2 (§4.6)")

    def add(target, key, val, share=1.0):
        acc[target][key] += val * share

    def frozen_before(target, t: Optional[dt.datetime]) -> bool:
        """True when `target` froze before t (time after the freeze: not counted in its caps)."""
        f = frz.get(target)
        return bool(f and t and t > _parse_t(f['frozen_at']))

    for r in tun:
        c = r.get('campaign', '')
        if not c:
            unassigned += 1
            continue
        t0 = _parse_t(r['wall_start']) if r.get('wall_start') else None
        t1 = _parse_t(r['wall_end']) if r.get('wall_end') else None
        stage = r.get('stage', '')
        act, mach = _active_s(r), _f(r.get('t_machine_s'))
        if c == 'B-chung':
            tgt = _b_target(r.get('controller'))
            targets = [(tgt, 1.0)] if tgt else [('B-PID', 0.5), ('B-LADRC', 0.5)]
            add('B-chung', 'T_op_s', act)
            add('B-chung', 'T_noh_s', act)
            add('B-chung', 'T_mach_s', mach)
        else:
            targets = [(c, 1.0)]
        for tgt, share in targets:
            add(tgt, 'T_op_s', act, share)
            add(tgt, 'T_noh_s', act, share)
            add(tgt, 'T_mach_s', mach, share)
            if stage not in ('final', 'final_hil') and not frozen_before(tgt, t0):
                add(tgt, 'T_tune_s', act, share)
            plat = r.get('platform')
            if plat == 'SIM':
                add(tgt, 'N_SIM', _f(r.get('n_sim_evals')) or 1.0, share if c != 'B-chung' else 1.0)
            elif plat == 'HIL':
                add(tgt, 'N_HIL', 1.0)
            elif plat == 'HW':
                add(tgt, 'N_HW', 1.0)
                if stage in TUNE_STAGES:
                    add(tgt, 'N_HW_tune', 1.0)
            for t in (t0, t1):
                if t:
                    times[tgt].append(t)
        if c == 'B-chung':
            for t in (t0, t1):
                if t:
                    times['B-chung'].append(t)
        if stage in ('final', 'final_hil'):
            task, o = r.get('trajectory', ''), r.get('run_outcome', '')
            if stage == 'final':
                home, cond = c, f"final_{task}@{r.get('speed_mps', '')}"
                final_tasks[c][task].append(o)
            else:
                gs = r.get('gain_set_id', '')
                home = by_gain.get(gs, 'RQ2')
                cond = (f"final_hil_{task}@{r.get('speed_mps', '')}" if home != 'RQ2'
                        else f"final_hil_{gs}_{task}@{r.get('speed_mps', '')}")
            f = finals[home]
            if o in VALID:
                f[f'{cond}_n_valid'] += 1
                if f[f'{cond}_n_valid'] > FINAL_N:
                    raise LogError(f'{home} {cond}: more than {FINAL_N} valid runs (§4.7)')
            f[f'{cond}_' + _COUNT.get(o, 'invalid')] += 1

    for r in eff:
        c, k = r.get('chien_dich', ''), r.get('khoan', '')
        if not c and not k:
            continue                       # rows before S05 §6.2 (no account): not in any line
        h = _f(r.get('gio')) * 3600
        hoc = r.get('nhan') == 'hoc'
        t = None
        if r.get('ngay') and r.get('bat_dau'):
            t = dt.datetime.fromisoformat(f"{r['ngay']}T{r['bat_dau']}")
        if c == 'B-chung':
            targets = [('B-PID', 0.5), ('B-LADRC', 0.5)]
            add('B-chung', 'T_op_s', h)
            if not hoc:
                add('B-chung', 'T_noh_s', h)
        else:
            targets = [(c or k, 1.0)]
        for tgt, share in targets:
            add(tgt, 'T_op_s', h, share)
            if not hoc:
                add(tgt, 'T_noh_s', h, share)
                if not frozen_before(tgt, t):
                    add(tgt, 'T_tune_s', h, share)
            if t:
                times[tgt].append(t)

    for r in hw:
        c = r.get('chien_dich', '')
        if c:
            add(c, 'N_hong', 1.0)
            add(c, 'T_sua_s', _f(r.get('gio_sua')) * 3600)

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
        if c in ('A-PID', 'A-LADRC', 'B-PID', 'B-LADRC'):
            if a.get('N_HW_tune', 0) >= CAP_HW:
                cap.append('xe')
            if a.get('T_tune_s', 0) / 3600 >= CAP_H:
                cap.append('gio')
        reason = (f or {}).get('stop_reason', '')
        check = ''
        if reason == 'cham_tran_xe':
            check = 'ok' if 'xe' in cap else 'mismatch'
        elif reason == 'cham_tran_gio':
            check = 'ok' if 'gio' in cap else 'mismatch'
        elif reason:
            check = 'khong_kiem_duoc'
        if check == 'mismatch':
            print(f'warning: {c} stop_reason {reason} does not match the logs (cap_hit {"+".join(cap) or "none"})',
                  file=sys.stderr)
        gres = gate_result(final_tasks[c], gate) if gate is not None and final_tasks.get(c) else ''
        row = {'campaign': c, 'T_operator_h': _fmt(a.get('T_op_s', 0) / 3600, 3),
               'T_operator_no_hoc_h': _fmt(a.get('T_noh_s', 0) / 3600, 3),
               'T_machine_h': _fmt(a.get('T_mach_s', 0) / 3600, 3), 'T_wall_h': _fmt(wall, 3),
               'N_SIM': str(int(round(a.get('N_SIM', 0)))), 'N_HIL': str(int(a.get('N_HIL', 0))),
               'N_HW': str(int(a.get('N_HW', 0))), 'N_hong': str(int(a.get('N_hong', 0))),
               'T_sua_h': _fmt(a.get('T_sua_s', 0) / 3600, 3), 'stop_reason': reason,
               'stop_reason_check': check, 'cap_hit': '+'.join(cap),
               'T_tune_no_hoc_h': _fmt(a.get('T_tune_s', 0) / 3600, 3),
               'N_HW_tune': str(int(a.get('N_HW_tune', 0))), 'gate_result': gres,
               'T_wall_end_src': src, 'B_chung_share': 'unsplit total' if c == 'B-chung' else ''}
        row.update({k: str(v) for k, v in sorted(finals.get(c, {}).items())})
        out.append(row)
    if unassigned:
        print(f'note: {unassigned} tuning row(s) without campaign left out', file=sys.stderr)
    return out


def write_summary(ds: str, out_path: str, gate: Optional[Iterable[str]] = None) -> List[dict]:
    rows = summary(ds, gate)
    extra = sorted({k for r in rows for k in r} - set(SUMMARY_HEAD))
    _write_all(out_path, SUMMARY_HEAD + extra, rows)
    return rows


# ── C9 dry run ─────────────────────────────────────────────────────────────
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

        effort(out_dir, '2026-11-20', '07:30', '08:00', campaign='A-PID', nhan='hoc', ghi_chu='FAKE: doc huong dan tune')
        for i, sp in enumerate(('0.2', '0.5', '0.8')):
            begin(out_dir, 'FAKE-A1', 'A-PID', 'HW', 'PID', 'O1-F05', sp, 'screen', gain_set='G00',
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
        freeze(out_dir, 'A-PID', 'khong_du_luot_xac_nhan', 'G00', gfile, at=_iso(clock[0]),
               notes='FAKE')
        machine(out_dir, 'B-chung', 'PID', 'FAKE-S03', 'results/S03/FAKE', 2739, 5400, _iso(clock[0]),
                _iso(clock[0] + dt.timedelta(seconds=5400)), t_setup_s=600)
        machine(out_dir, 'B-chung', 'LADRC', 'FAKE-S03', 'results/S03/FAKE', 2739, 5400, _iso(clock[0]),
                _iso(clock[0] + dt.timedelta(seconds=5400)))
        effort(out_dir, '2026-11-20', '13:00', '14:00', campaign='B-chung', ghi_chu='FAKE: doc S03 chung')
        effort(out_dir, '2026-11-20', '14:00', '14:30', khoan='C_chung', ghi_chu='FAKE: script video')
        tick(7200)
        begin(out_dir, 'FAKE-B1', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='GA1')
        step(out_dir, 'run')
        tick(90)
        end(out_dir, 'dat', completed=1, e_perp_max=0.04)
        hwfail(out_dir, '2026-11-20', 'FAKE-B1', 'B-PID', 'HW', 'FAKE: tuot banh', 'phan_cung_ngau_nhien', 0.5)
        for k in range(2):                              # final on HIL: account RQ2, joined by gain set
            begin(out_dir, 'FAKE-FH', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='G00')
            step(out_dir, 'run')
            tick(75)
            end(out_dir, 'dat', completed=1, e_perp_max=0.05)
        lost = None
        for o in ('hong_he_do', 'dat', 'chua_ket_luan'):  # final on the car; the lost run is replaced
            begin(out_dir, 'FAKE-F1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='G00',
                  replaces_run_uid=lost if o == 'dat' else '')
            step(out_dir, 'run')
            tick(60)
            row = end(out_dir, o, completed=1, e_perp_max=0.10, fail_cause='mat_khung_FAKE' if o == 'hong_he_do' else '')
            if o == 'hong_he_do':
                lost = row['run_uid']
        return write_summary(out_dir, os.path.join(out_dir, SUMMARY), gate=['O1-F05'])
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
    for name in ('fail-cause', 'results-dir', 'notes'):
        e.add_argument('--' + name, default='')
    m = sub.add_parser('machine')
    for name in ('campaign', 'controller', 'script-run-id', 'results-dir', 'wall-start', 'wall-end'):
        m.add_argument('--' + name, required=True)
    m.add_argument('--n-sim-evals', type=int, required=True)
    m.add_argument('--t-machine-s', type=float, required=True)
    m.add_argument('--platform', default='SIM', choices=PLATFORMS)
    m.add_argument('--stage', default='explore', choices=STAGES)
    m.add_argument('--notes', default='')
    for k in STEPS:
        m.add_argument(f'--t-{k}-s', type=float)
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
    z.add_argument('--campaign', required=True, choices=CAMPAIGNS)
    z.add_argument('--stop-reason', required=True, choices=STOP_REASONS)
    for name in ('gain-set', 'gain-file'):
        z.add_argument('--' + name, required=True)
    for name in ('at', 'notes'):
        z.add_argument('--' + name, default='')
    o = sub.add_parser('summary')
    o.add_argument('--out', default='')
    o.add_argument('--gate', default=None, help='figure-8 tasks in the gate, e.g. O1-F05,O1-F08 (locked at C8)')
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
            print('cancelled', cancel(ds)['row']['run_uid'], '(no row written)')
        elif a.cmd == 'status':
            run = _load_open(ds)
            if not run:
                print('no open run')
            else:
                segs = run['segments'] + ([dict(run['current'], end=_iso(now()))] if run.get('current') else [])
                print(run['row']['run_uid'], 'paused' if run.get('paused_from') is not None else 'running',
                      json.dumps(segment_totals(segs)))
        elif a.cmd == 'end':
            row = end(ds, a.outcome, a.completed, a.e_perp_max, a.fail_cause, a.results_dir, a.notes)
            print('row', row['run_uid'], {k: row[f't_{k}_s'] for k in STEPS}, 'wait', row['t_operator_wait_s'])
        elif a.cmd == 'machine':
            row = machine(ds, a.campaign, a.controller, a.script_run_id, a.results_dir, a.n_sim_evals, a.t_machine_s,
                          a.wall_start, a.wall_end, a.platform, a.stage, notes=a.notes,
                          **{f't_{k}_s': getattr(a, f't_{k}_s') for k in STEPS})
            print('row', row['run_uid'])
        elif a.cmd == 'effort':
            row = effort(ds, a.date, a.start, a.end, a.campaign, a.khoan, a.nhan, a.ghi_chu, buoc=a.buoc,
                         giai_doan=a.giai_doan, dut=a.dut, lap_lai=a.lap_lai)
            print('row', row['ngay'], row['gio'], 'h')
        elif a.cmd == 'hwfail':
            hwfail(ds, a.date, a.session, a.campaign, a.platform, a.what, a.cause, a.fix_h, a.iter_no, a.notes,
                   dut=a.dut, buoc=a.buoc, lap_lai=a.lap_lai)
            print('row written')
        elif a.cmd == 'freeze':
            row = freeze(ds, a.campaign, a.stop_reason, a.gain_set, a.gain_file, a.at, a.notes)
            print('frozen', a.campaign, row['gain_sha256'])
        elif a.cmd == 'summary':
            gate = [g.strip() for g in a.gate.split(',') if g.strip()] if a.gate else None
            rows = write_summary(ds, a.out or os.path.join(ds, SUMMARY), gate)
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
