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
         [--fail-cause X] [--results-dir D] [--gate-json F] [--gate-file C]   close the run, append its row;
                                         C = the C8 gate file (O1 v11.33): the conditions that must
                                         decide are recomputed from it (level = task, platform, stage,
                                         campaign); needed when F was scored with it (gate_v3.3)
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
                                         B8-1 (S05 v3.20 C9 (a)-(c), 14/11): a judged car run needs the
                                         camera scorer output (two frame domains, G1 fields: see GATE_COLS);
                                         a judged car / final_hil run needs both boot answers (API and summary)
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
                                         F = the locked C8 gate file (gate_c8_v1 JSON, tools/gate_c8.py:
                                         gate levels, report-only conditions per level); its sha256 is
                                         reported; every gate row is checked against it (O1 v11.33)
  so-h   --run-uid UID --score-final F   S05 v3.37 4.3: outcome of a car run after the clip H comparison at the end
                                         of the session (F = c7_calib_drift.py gop output); appended to
                                         m7_so_clip_h.csv; summary, begin, s05_final_table use the outcome after it
  demo   --out DIR                       C9 dry run: made-up campaign in DIR (every judged run with its folder:
                                         s05_fake_runs.py), then summary

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
# S05 v3.37 §4.3 (B8S-3), O1 v11.27–v11.31 §4 clip H (CODE 17/11 to CODE_C13; Liam 17/11 at CODE_C13, hop chon, nguyen
# van: "Thêm lệnh ghi sau so H (Recommended)"): the result of a floor run is provisional until the clip H comparison
# at the end of the session (tools/c7_calib_drift.py so / gop). Command so-h appends one row per car run to SO_H
# (append only; the tuning log row is never changed); summary, begin and s05_final_table use the outcome after it.
SO_H = 'm7_so_clip_h.csv'
SO_H_COLS = ['run_uid', 'outcome_tam', 'outcome_sau', 'flags', 'vuot', 'D_H_mm', 'score_final_file',
             'score_final_sha256', 'ghi_luc']
SO_H_FLAGS = ('calib_drift', 'calib_drift_split', 'thieu_H1')
FAIL_SO_H = 'so_clip_h'                  # fail cause of a run that became hong_he_do at the comparison
SO_H_LOST = '_so_h_lost'                 # C9D2-4: in-memory mark set by apply_so_h only (never a log column)
RETRO_FLAG = 'sau_dung_hoi_to'           # S05 v3.38 4.3 (B8S2-2): run after a stop the comparison showed

# R134 (1) (deferred at R134 round 4, R134C4-1; CODE 16/11 to CODE_C13, Liam 16/11 at CODE "Sửa demo + gộp R134(1)
# (Recommended)"; Liam 16/11 at CODE_C13, hop chon, nguyen van: "Ghi đường dẫn + SHA (Recommended)"): 'end' writes
# the path and sha256 of the two gain_send --query outputs it read; the summary reads them again (check_query_rows).
QUERY_FILE_COLS = ('query_before_file', 'query_before_sha256', 'query_after_file', 'query_after_sha256')

# S05 §6.1: columns kept (checked 08/11) + added
TUNING_OLD = ['session_id', 'date', 'platform', 'controller', 'trajectory', 'iter_no', 't_flash_s',
              't_setup_s', 't_run_s', 't_log_s', 't_analysis_s', 'gain_changed', 'old_value',
              'new_value', 'rmse_vx', 'rmse_vy', 'rmse_wz', 'stop_met', 'bug_id', 'notes', 'dut', 'buoc']
GAINS = ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'wc_xy', 'wo_xy', 'wc_wz', 'wo_wz', 'pos_kp', 'pos_ki', 'k_psi']
TUNING_NEW = (['run_uid', 'campaign', 'script_run_id', 'gain_set_id', 'speed_mps', 'stage', 'block_id',
               'replaces_run_uid', 'candidate_no', 'results_dir', 'completed', 'e_perp_max_m',
               'run_outcome', 'fail_cause', 'rule_version', 'fw_hash']
              + ['g_' + g for g in GAINS]
              + ['t_operator_wait_s', 't_machine_s', 'wall_start', 'wall_end', 'n_sim_evals']
              + ['boot_before', 'boot_after']           # R134 (b): S05 4.7 (iv), R3V-9
              + list(QUERY_FILE_COLS))                  # R134 (1), summary reads the two answers again
# tune stop gate v3 (dossier 00_OVERVIEW/2026-11-12_De_xuat_cong_dung_tune_bam_huong_van_toc_v3.md section 6,
# CG2-12; Liam 12/11 CODE local_87282b70 "làm ngay đi"): per run the measured maximum q, its bound U and
# the state of each condition, so stopping, the S05 4.4 ranking and gate_result can be recomputed.
# After Codex round 1 (Liam "Cho sửa 7 chỗ"): gate_decide = conditions that may decide (GV3-2),
# gate_json_sha256 = the scorer file the row came from, with results_dir required (GV3-6)
_FWSIM = os.path.normpath(os.path.join(HERE, '..', 'fwsim'))
if _FWSIM not in sys.path:
    sys.path.insert(0, _FWSIM)
import o1_gate                                                          # noqa: E402
import gate_c8                                                          # noqa: E402  (C8 gate file, O1 v11.33)
GATE_CONDS = o1_gate.CONDS
GATE_VERSIONS = (o1_gate.GATE_VERSION, o1_gate.GATE_VERSION_V1133)
# 13/11 gate v3.2 (SC-1, SC3-1; Liam CODE local_7f7b0ef4 "Duyệt gộp luôn lo/hi"): per condition G2-G5 the
# maxima over the frames of q, lo, hi (each field over its finite frames) and n_undef; G1 q is 'completed';
# gate_u_t = the U_t used (empty = not known: HIL run without the run_active edge mapping, RS-4)
GATE_FIELDS = ('q', 'lo', 'hi', 'n_undef', 'state')
OLD_GATE_U = tuple(f'g{i}_u' for i in range(1, 6))       # gate_v3_g1235 columns, dropped by migrate when empty
# B8-1 B81-3 (S05 v3.20 6.3, C9 (c); O1 v11.15 3.3 "Khung nghi ngo vi tri", 3.5 R4V-3, R4V-4, RS2-10; CODE 14/11 gave
# it to CODE_R134, Liam 14/11 CODE_R134 "Gop vao vong 4 (Recommended)"): G2-G5 stored on the two frame domains of
# O1 3.3. g*_q, g*_lo, g*_hi, g*_n_undef (and g*_state) are the UNION domain (t = 0 to the later t_done, or to the
# deadline when one way does not complete: every frame the run reports, the S05 4.4 ranking); g*_<field>_chung the
# COMMON domain (t = 0 to the earlier t_done, or to the deadline when neither completes). One way of using the
# frames (fwsim, HIL: no camera, no position-suspect frame) or two equal t_done: the two domains are one and the
# two sets equal. G1 has no q, lo, hi (RS2-10): g1_ways (1 fwsim / HIL, 2 floor), g1_t_deadline, g1_t_done_1
# (floor: suspect frames used like every frame), g1_t_done_2 (floor: suspect frames as gaps), empty = not completed
# before the deadline; fwsim / HIL: g1_t1 (first tracking-error frame, empty = none), g1_prog_before_t1 (progress
# reached S - D_d before t1), g1_cond25_before_deadline (a time before the deadline with conditions 2-5 met);
# floor: g1_t1, g1_cat_t1 (R2V-4: a khong_dat run by the exception before t1, fail cause VIOL_BEFORE; G1 is then
# chua_ket_luan). States are recomputed from these (gate_g1_state, cond_state2), never trusted.
DOMAIN_FIELDS = ('q', 'lo', 'hi', 'n_undef')
G1_COLS = ['g1_ways', 'g1_t_deadline', 'g1_t_done_1', 'g1_t_done_2', 'g1_t1', 'g1_prog_before_t1',
           'g1_cond25_before_deadline', 'g1_cat_t1']
# O1 v11.33 (CODE_GATE 17/11, design 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md section 4; Liam 17/11 at
# CODE_GATE "Duyệt, gửi CODE luôn (Recommended)"): gate_v3.3 rows carry the C8 gate file SHA and the report-only
# conditions of their level (O1 3.9 rule 2b, OG2-7). The required set is recomputed from the gate file
# (tools/gate_c8.py, --gate-file of end / summary), never taken from the row: a row whose report-only list differs
# from the file is an error. Empty in gate_v3.2 rows (all five required).
GATE_V1133_COLS = ['gate_file_sha256', 'gate_report_only']
# C8b time grid (CODE_GATE design v2 section 2; Liam 17/11 at CODE_GATE "Duyệt, code luôn (Recommended)", the exception
# before t1 "Khi mọi nút đều có (Recommended)"): a floor gate scored on the nodes of the C8b time grid carries a block
# 'luoi' (format luoi_v1): G1 per node, lost / exception-before-t1 per node; G1 of the run is recomputed from it
# (luoi_g1), the G1 fields of the gate are those of node nut_tam. Columns luoi_* below, empty without a grid.
LUOI_FORMAT = 'luoi_v1'
LUOI_STEP_MAX_S = 1e-3                     # C8b v4 section 3: nodes at most 1 ms apart
LUOI_COLS = ['luoi_gio_sha256', 'luoi_n_nut', 'luoi_d_dau_s', 'luoi_d_cuoi_s', 'luoi_buoc_max_s', 'luoi_nut_tam',
             'luoi_g1_nut', 'luoi_hong_nut', 'luoi_cat_nut', 'luoi_g1_state']
_G1_LETTER = {'dat': 'd', 'khong_dat': 'k', 'chua_ket_luan': 'c'}
GATE_COLS = (['gate_version', 'gate_decide', 'gate_json_sha256', 'gate_u_t', 'g1_q', 'g1_state']
             + [f'{g}_{x}' for g in o1_gate.MEASURED for x in GATE_FIELDS]
             + [f'{g}_{x}_chung' for g in o1_gate.MEASURED for x in DOMAIN_FIELDS] + G1_COLS + GATE_V1133_COLS
             + LUOI_COLS)
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
# R134 package (Liam 14/11 CODE_R134 local_eb9f3b5c "duyet goi sua code fix/r134-answers"):
# (b) S05 v3.13-v3.17 4.7 (iv), R3V-9, O1 v11.10 R4: the boot of gain_send.py --query just before and just after
#     the run go to the run history (boot_before, boot_after). Both given and equal: one ESP32 boot over the run.
#     Missing, unreadable or different: a run failed by operation, so only hong_he_do with fail cause BOOT_FAIL.
#     The CLI 'end' requires the two query files (--query-before, --query-after) for a judged run
#     (dat / khong_dat / chua_ket_luan) on the car (every stage: the obstacle-stop rule of O1 R4 ties #us to
#     the run through them) and on the HIL at final_hil (tv_duty, S05 4.7 (iv)); query_files_needed().
#     [CACH HIEU] other HIL stages and SIM do not need them (the HIL rules of O1 3.5 do not use #us). The
#     Python API keeps them optional (older callers, demo); summary refuses a judged row whose two boots
#     disagree.
# (d) SC3-2 (S05 §6.3 "Kiem cuoi va tong hop", rule [SC3-2]): a run of a campaign of CAMPAIGNS (A-PID on) is
#     never dat without gate evidence: end refuses dat without --gate-json on every platform (HIL already
#     needs it for any judged outcome, Codex S315-1), and the summary refuses a dat row of a campaign with
#     empty gate columns (NO_GATE_EVIDENCE). [CACH HIEU] "historic rows before the gate columns" are rows
#     without a campaign of CAMPAIGNS (the example rows of m7_tuning_log.csv); they are not in any campaign
#     summary, so they never reach gate_result or a stop rule; count_no_gate() reports them apart.
# B8-1 (S05 v3.20 8.1 C9 (a), (b); B81-1, B81-2; CODE 14/11 to CODE_R134):
# B81-1 a car run (platform HW) judged dat / khong_dat / chua_ket_luan carries the camera scorer output
#     (--gate-json with the two frame domains and the G1 fields of the floor, B81-3 above), at end and in the
#     summary (a judged car row without gate columns is an error, as for HIL).
# B81-2 boot_needed (car runs, HIL final_hil) is applied by end and by the summary, not only by the CLI: a judged
#     run there without both boot answers is a run failed by operation (only hong_he_do with BOOT_FAIL).
BOOT_FAIL = 'boot_thieu_hay_khac'        # --fail-cause of a run without one ESP32 boot shown
NO_GATE_EVIDENCE = 'chua_du_bang_chung_cong'   # SC3-2: a dat row without gate columns
VIOL_BEFORE = 'vi_pham_truoc_loi_he_do'  # S05 6.1, O1 3.5 R2V-4: khong_dat by a violation before t1 (floor, cat_t1)

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
# fields begin / end set themselves; begin refuses them as extra fields (Codex S315-R3-1)
OWN_FIELDS = ('session_id', 'iter_no', 'run_uid', 'date', 'wall_start', 'wall_end', 'run_outcome', 'fail_cause',
              'results_dir', 'completed', 'e_perp_max_m') + tuple(f't_{k}_s' for k in ('flash', 'setup', 'run', 'log',
                                                                                        'analysis', 'operator_wait'))


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
    """Next iter_no of `session`: past the logged rows AND the open run of the same session (S05 v3.15
    Codex round 5 P3: begin(session='MACHINE') then machine gave two MACHINE:1; Liam 14/11 CODE_R134
    "Lam cung goi (Recommended)"). Limit (Codex R134C2-9, DEFERRED): one writer at a time (one operator, one
    command), as the S05 workflow runs it; two processes writing the same dataset at once could both take the
    same number, and summary then refuses the log (a run_uid on two rows, S315-R4-1)."""
    nums = [int(_f(r.get('iter_no'))) for r in read_rows(os.path.join(ds, TUNING)) if r.get('session_id') == session]
    run = _load_open(ds)
    if run and run.get('row', {}).get('session_id') == session:
        nums.append(int(_f(run['row'].get('iter_no'))))
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
            if k in row or k.startswith('g_') or k in OWN_FIELDS:   # Codex S315-R3-1: the identity fields
                raise LogError(f'field {k}: set by begin or end, not by an extra field')   # are checked, not given
            row[k] = str(v)
    tun = apply_so_h(live_rows(read_rows(os.path.join(ds, TUNING))), ds)   # S315-R2-2; v3.37 outcome after so-h
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


def _check_decide(decide, platform: Optional[str] = None, required=None) -> List[str]:
    """required (O1 v11.33): the conditions that must decide at the run's level, from the gate file (None = all five);
    decide must be required intersected with a deciding set of the platform."""
    req = set(GATE_CONDS if required is None else required)
    if (not isinstance(decide, list) or not decide or len(set(decide)) != len(decide)
            or any(not isinstance(c, str) for c in decide) or set(decide) - set(GATE_CONDS)):
        raise LogError(f'gate decide {decide!r}: a non-empty list of distinct {GATE_CONDS}')
    if set(decide) not in [set(x) & req for x in o1_gate.DECIDE_SETS]:
        raise LogError(f'gate decide {decide!r}: {list(o1_gate.DECIDE_ALL)} or {list(o1_gate.DECIDE_HIL)}, less the '
                       f'report-only conditions {sorted(set(GATE_CONDS) - req)} of the gate file')
    if platform is not None:
        want = DECIDE_BY_PLATFORM.get(platform)
        if want is None or set(decide) not in [set(x) & req for x in want]:
            raise LogError(f'gate decide {decide!r} on platform {platform!r}: must be one of '
                           f'{[list(x) for x in (want or ())]} (GV3-R2-2)')
    return decide


def _n_undef(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not float(x).is_integer() or x < 0:
        raise LogError(f'gate {what}: {x!r} is not a whole number >= 0')
    return int(x)


G1_KEYS = ('t_done', 't_deadline', 't1', 'prog_before_t1', 'cond25_before_deadline', 'cat_t1')


def _opt_bool(x, what: str):
    if x is not None and not isinstance(x, bool):
        raise LogError(f'gate {what}: {x!r} is not null or a boolean')
    return x


def _time(x, what: str, null_ok: bool = True):
    if x is None and null_ok:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or x < 0:
        raise LogError(f'gate {what}: {x!r} is not {"null or " if null_ok else ""}a finite time >= 0')
    return float(x)


def gate_g1_fields(g1: dict, platform: Optional[str] = None) -> Optional[dict]:
    """B81-3: the G1 fields of O1 3.5 (RS2-10) in a gate's g1 block, checked; None when the block has none
    (a fwsim / HIL scorer that does not write them yet: G1 then rests on q alone, refused for a car run).
    t_done: list, one time per way of using the frames (1 fwsim / HIL, 2 floor: suspect frames kept, as gaps),
    null = not completed before t_deadline."""
    if g1.get('t_done') is None:
        if platform == 'HW':
            raise LogError('gate g1: a car run needs t_done of the two ways of using the frames and t_deadline '
                           '(camera scorer output; O1 3.3, 3.5 RS2-10; B81-1, B81-3)')
        extra = [k for k in G1_KEYS if g1.get(k) is not None]
        if extra:
            raise LogError(f'gate g1: {", ".join(extra)} without t_done')
        return None
    td = g1['t_done']
    if not isinstance(td, list) or len(td) not in (1, 2):
        raise LogError(f'gate g1 t_done: {td!r} is not a list of 1 (fwsim, HIL) or 2 (floor) times')
    ways = len(td)
    if platform is not None and ways != (2 if platform == 'HW' else 1):
        raise LogError(f'gate g1 t_done: {ways} way(s) on platform {platform!r} (floor 2, fwsim / HIL 1; O1 3.5)')
    t_dl = _time(g1.get('t_deadline'), 'g1 t_deadline', null_ok=False)
    t_done = [_time(t, 'g1 t_done') for t in td]
    if any(t is not None and t >= t_dl for t in t_done):              # R134C4-4: O1 3.4 (4) strictly before
        raise LogError(f'gate g1 t_done {t_done!r} not before t_deadline {t_dl!r} (a time is given only when '
                       'completed strictly before the deadline, O1 3.4 condition 4)')
    t1 = _time(g1.get('t1'), 'g1 t1')
    prog = _opt_bool(g1.get('prog_before_t1'), 'g1 prog_before_t1')
    c25 = _opt_bool(g1.get('cond25_before_deadline'), 'g1 cond25_before_deadline')
    cat = _opt_bool(g1.get('cat_t1'), 'g1 cat_t1')
    if ways == 1:
        if cat is not None:
            raise LogError('gate g1 cat_t1: a floor field (O1 3.5 R2V-4), not for one way of using the frames')
        if t1 is None and (prog is not None or c25 is not None):
            raise LogError('gate g1: prog_before_t1 / cond25_before_deadline without t1')
        if t1 is not None and (prog is None or (not prog and c25 is None)):
            raise LogError('gate g1: with t1, prog_before_t1, and cond25_before_deadline when progress was not '
                           'reached before t1, are needed (O1 3.5, last block)')
        if t1 is not None and not prog and t_done[0] is not None and not c25:
            raise LogError('gate g1: completed (t_done) although no time before the deadline had conditions 2-5 met')
    else:
        if prog is not None or c25 is not None:
            raise LogError('gate g1: prog_before_t1 / cond25_before_deadline are fwsim / HIL fields (one way)')
        if cat and t1 is None:
            raise LogError('gate g1 cat_t1: needs t1 (O1 3.5 R2V-4)')
    return dict(ways=ways, t_deadline=t_dl, t_done=t_done, t1=t1, prog_before_t1=prog, cond25_before_deadline=c25,
                cat_t1=bool(cat) if ways == 2 else None)


def gate_g1_state(f: dict) -> str:
    """G1 recomputed from its fields (O1 v11.15 3.3 "Khung nghi ngo vi tri", 3.5 last block, R2V-4)."""
    done = [t is not None for t in f['t_done']]
    if f['ways'] == 2:                       # floor: both ways done -> dat, neither -> khong_dat, else inconclusive
        if f['cat_t1']:
            return 'chua_ket_luan'           # R2V-4: progress after t1 not known
        return 'dat' if all(done) else ('khong_dat' if not any(done) else 'chua_ket_luan')
    if f['t1'] is None or f['prog_before_t1']:  # no tracking error, or progress reached before t1
        return 'dat' if done[0] else 'khong_dat'
    return 'chua_ket_luan' if f['cond25_before_deadline'] else 'khong_dat'


def domains_one(f: Optional[dict]) -> bool:
    """The common and the union domain are the same frames: one way, no G1 fields, or two equal t_done."""
    return f is None or f['ways'] == 1 or f['t_done'][0] == f['t_done'][1]


def cond_state2(lo_common, lo_union, hi_union, n_undef_union, threshold: float) -> str:
    """O1 3.5 "Theo so luu" on two domains: khong_dat when lo of the common domain > threshold; else the three
    states on the union (a certain violation only in the union is chua_ket_luan). One domain: o1_gate.cond_state."""
    if o1_gate._finite(lo_common) and float(lo_common) > threshold:
        return 'khong_dat'
    st = o1_gate.cond_state(lo_union, hi_union, n_undef_union, threshold)
    return 'chua_ket_luan' if st == 'khong_dat' else st


def _domain_nums(block: dict, what: str) -> tuple:
    q, lo, hi = (_qnum(block.get(k), f'{what} {k}') for k in ('q', 'lo', 'hi'))
    nu = _n_undef(block.get('n_undef'), f'{what} n_undef')
    # n_undef 0: every frame finite, so q, lo, hi are all finite with lo <= q <= hi (some frame), or all
    # unknown (no frame scored: inconclusive). With undefined frames each field is the maximum over its own
    # finite frames (SC3-1), so hi may be below q (the frame of q may have hi = inf). G32-1
    fin = [not math.isnan(x) for x in (q, lo, hi)]
    if nu == 0 and any(fin) and not (all(fin) and lo <= q <= hi):
        raise LogError(f'gate {what}: with n_undef 0, q, lo, hi must all be numbers with lo <= q <= hi '
                       f'(or all unknown); got {lo!r}, {q!r}, {hi!r}')
    return q, lo, hi, nu


def _common_block(gate: dict, c: str, union: tuple, one: bool, platform: Optional[str]) -> tuple:
    """The common domain of condition c (B81-3), checked against the union: part of it (a finite maximum of the
    common domain has a finite union maximum not below it; n_undef not above), equal when the domains are one.
    No "chung" block (the gate of o1_gate, one domain): the union numbers, refused on the car."""
    blk = gate[c].get('chung')
    if blk is None:
        if platform == 'HW':
            raise LogError(f'gate {c}: a car run needs the common frame domain ("chung": q, lo, hi, n_undef; '
                           'O1 3.3, 3.5 R4V-4; B81-3)')
        if not one:
            raise LogError(f'gate {c}: two different t_done but no common domain ("chung")')
        return union
    if not isinstance(blk, dict) or any(k not in blk for k in DOMAIN_FIELDS):
        raise LogError(f'gate {c} chung: not an object with {DOMAIN_FIELDS}')
    com = _domain_nums(blk, f'{c} chung')
    for k, a, b in zip(('q', 'lo', 'hi'), com[:3], union[:3]):
        if not math.isnan(a) and (math.isnan(b) or a > b):
            raise LogError(f'gate {c}: {k} of the common domain {a!r} above that of the union {b!r} (the common '
                           'domain is part of the union, O1 3.3)')
    if com[3] > union[3]:
        raise LogError(f'gate {c}: n_undef of the common domain {com[3]} above that of the union {union[3]}')
    if one and any(not (a == b or (math.isnan(a) and math.isnan(b))) for a, b in zip(com, union)):
        raise LogError(f'gate {c}: one frame domain (one way, or the two t_done equal) but the common and union '
                       f'numbers differ: {com!r} / {union!r}')
    return com


def _recompute(gate: dict, platform: Optional[str] = None, required=None) -> dict:
    """States and outcomes recomputed from q, lo, hi, n_undef with fwsim/o1_gate (thresholds, O1 3.5
    three states SC3-1, O1 3.5 order); shared by the JSON path (end) and the logged rows (summary).
    B81-3: G2-G5 on the common and the union frame domain (cond_state2), G1 from its fields (gate_g1_state)
    when the gate has them (always on the car). Returns the outcomes."""
    if gate.get('version') not in GATE_VERSIONS:
        raise LogError(f'gate version {gate.get("version")!r}: only {GATE_VERSIONS} are supported')
    req = gate_required(gate, required)
    decide = _check_decide(gate.get('decide'), platform, req)
    u_t = _qnum(gate.get('u_t'), 'u_t')
    if u_t < 0.0:
        raise LogError(f'gate u_t: {u_t!r} is negative (a bound is >= 0)')
    # U_t known <=> G3, G5 may decide (RS-4): a HIL gate without the mapping cannot claim them. O1 v11.33: G4 is never
    # report only, so 'G4 decides' tells a mapped run from a refused one also when the level has report-only ones
    if ('g4' in decide) == math.isnan(u_t):
        raise LogError(f'gate decide {decide!r} with u_t {gate.get("u_t")!r}: G3, G4, G5 may decide exactly when '
                       'U_t is known (RS-4)')
    q1 = gate['g1'].get('q')
    if q1 not in _G1_Q or gate['g1'].get('state') != o1_gate.g1_state(q1):
        raise LogError(f'gate g1: q {q1!r} and state {gate["g1"].get("state")!r} do not agree (O1 3.4 completed)')
    f1 = gate_g1_fields(gate['g1'], platform)                                   # B81-3
    lb = None
    if gate.get('luoi') is not None:                                            # C8b time grid (design v2)
        if gate.get('version') != o1_gate.GATE_VERSION_V1133 or platform not in (None, 'HW') or f1 is None:
            raise LogError('gate luoi: only in a gate_v3.3 floor gate with its G1 fields')
        lb = check_luoi(gate['luoi'])
        if gate['g1']['state'] != lb['g1_state']:
            raise LogError(f'gate g1: state {gate["g1"]["state"]!r}, the grid gives {lb["g1_state"]!r} (luoi)')
        if 'cat_t1' in gate['g1'] and bool(gate['g1'].get('cat_t1')) != lb['cat_nut'][lb['nut_tam']]:
            raise LogError('gate g1 cat_t1: not that of the grid node nut_tam (luoi)')
    want1 = gate['g1']['state'] if lb is None else lb['g1_nut'][lb['nut_tam']]
    if f1 is not None and gate_g1_state(f1) != want1:
        raise LogError(f'gate g1: state {want1!r}{" (grid node nut_tam)" if lb else ""}, recomputed from t_done, t1 '
                       f'and the flags: {gate_g1_state(f1)!r} (O1 3.3, 3.5)')
    one = domains_one(f1) and lb is None             # grid numbers are not those of one node (design v2 section 2)
    for c, thr in o1_gate.THRESHOLDS.items():
        union = _domain_nums(gate[c], c)
        com = _common_block(gate, c, union, one, platform)
        st = cond_state2(com[1], union[1], union[2], union[3], thr)
        if st != gate[c].get('state'):
            raise LogError(f'gate {c}: state {gate[c].get("state")!r}, recomputed from lo of the common domain, '
                           f'hi and n_undef of the union: {st!r}')
    return o1_gate.evaluate_outcomes(gate, decide, req)


def luoi_g1(states) -> str:
    """G1 of a run on the C8b time grid (C8b v4 section 3, C8B4R5-1, -2): one node, its state; every node dat,
    dat; every node khong_dat, khong_dat (then the state does not change between the two nodes of any cell); else
    chua_ket_luan."""
    states = list(states)
    if len(states) == 1:
        return states[0]
    for st in ('dat', 'khong_dat'):
        if all(x == st for x in states):
            return st
    return 'chua_ket_luan'


def check_luoi(lb) -> dict:
    """The grid block of a floor gate checked (design v2 section 2); returns it. LogError on any problem."""
    keys = ('format', 'gio_sha256', 'n_nut', 'd_dau_s', 'd_cuoi_s', 'buoc_max_s', 'nut_tam', 'g1_nut', 'hong_nut',
            'cat_nut', 'g1_state')
    if not isinstance(lb, dict) or set(lb) != set(keys):
        raise LogError(f'gate luoi: an object with exactly {keys}')
    if lb['format'] != LUOI_FORMAT:
        raise LogError(f'gate luoi format {lb["format"]!r}: {LUOI_FORMAT!r}')
    sha = lb['gio_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(ch not in '0123456789abcdef' for ch in sha):
        raise LogError(f'gate luoi gio_sha256 {sha!r}: 64 lowercase hex digits')
    n = lb['n_nut']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise LogError(f'gate luoi n_nut {n!r}: a whole number >= 1')
    d0, d1 = _qnum(lb['d_dau_s'], 'luoi d_dau_s'), _qnum(lb['d_cuoi_s'], 'luoi d_cuoi_s')
    st = _qnum(lb['buoc_max_s'], 'luoi buoc_max_s')
    if math.isnan(d0) or math.isnan(d1) or math.isnan(st) or d0 > d1 or (n == 1) != (d0 == d1):
        raise LogError(f'gate luoi: d_dau_s {d0!r} <= d_cuoi_s {d1!r}, equal exactly when n_nut = 1')
    if st < 0.0 or st > LUOI_STEP_MAX_S or (n > 1 and (st == 0.0 or st * (n - 1) < d1 - d0 - 1e-12)):
        raise LogError(f'gate luoi buoc_max_s {st!r}: the nodes are at most {LUOI_STEP_MAX_S} s apart and cover '
                       f'[{d0!r}, {d1!r}] with {n} nodes (C8b v4 section 3)')
    k = lb['nut_tam']
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n:
        raise LogError(f'gate luoi nut_tam {k!r}: an index of the {n} nodes')
    if not isinstance(lb['g1_nut'], list) or len(lb['g1_nut']) != n or any(x not in o1_gate.STATES
                                                                         for x in lb['g1_nut']):
        raise LogError(f'gate luoi g1_nut: {n} states of {o1_gate.STATES}')
    for key in ('hong_nut', 'cat_nut'):
        if not isinstance(lb[key], list) or len(lb[key]) != n or any(not isinstance(x, bool) for x in lb[key]):
            raise LogError(f'gate luoi {key}: {n} booleans')
    if any(h and c for h, c in zip(lb['hong_nut'], lb['cat_nut'])):
        raise LogError('gate luoi: a node both lost to the measurement and with the exception before t1 (a node with '
                       'the exception is not a lost node: the exception replaces the loss, O1 3.5)')
    # O1 v11.35 (CODE asked CODE_GATE to confirm 17/11): a lost node and a node with the exception before t1 carry G1
    # chua_ket_luan (they cannot show the run completed, nor that it did not: O1 3.5 R2V-4 for the exception)
    bad = [i for i, (st_, h, c) in enumerate(zip(lb['g1_nut'], lb['hong_nut'], lb['cat_nut']))
           if (h or c) and st_ != 'chua_ket_luan']
    if bad:
        raise LogError(f'gate luoi g1_nut: node(s) {bad} lost or with the exception before t1 but G1 not chua_ket_luan')
    if lb['g1_state'] != luoi_g1(lb['g1_nut']):
        raise LogError(f'gate luoi g1_state {lb["g1_state"]!r}: from g1_nut it is {luoi_g1(lb["g1_nut"])!r}')
    return lb


def gate_required(gate: dict, required=None) -> tuple:
    """O1 v11.33 (OG2-7): the required set of a gate, checked against the one of the gate file (`required`, from
    gate_c8.GateC8.required for the run; None = no gate file: all five). A gate_v3.2 gate has no report-only
    conditions. A gate_v3.3 gate states its report_only, required and gate_file_sha256; they must agree with each
    other and with `required` (a condition missing from the run is never taken as exempt)."""
    want = tuple(GATE_CONDS) if required is None else tuple(c for c in GATE_CONDS if c in set(required))
    if gate.get('version') == o1_gate.GATE_VERSION:
        for k in ('report_only', 'required', 'gate_file_sha256'):
            if gate.get(k) not in (None, [], ''):
                raise LogError(f'gate {k} {gate.get(k)!r}: not a {o1_gate.GATE_VERSION} field')
        ro = []
    else:
        ro, rq, sha = gate.get('report_only'), gate.get('required'), gate.get('gate_file_sha256')
        if not isinstance(ro, list) or not isinstance(rq, list):
            raise LogError(f'gate report_only {ro!r} / required {rq!r}: lists ({o1_gate.GATE_VERSION_V1133})')
        try:
            if tuple(rq) != o1_gate.required_of(ro):
                raise LogError(f'gate required {rq!r}: not all five less report_only {ro!r}')
        except ValueError as exc:
            raise LogError(f'gate {exc}')
        if sha is None and ro:
            raise LogError(f'gate report_only {ro!r} without gate_file_sha256: no condition is exempt without the '
                           'C8 gate file (O1 v11.33, OG2-7)')
    got = tuple(c for c in GATE_CONDS if c not in set(ro))
    if got != want:
        raise LogError(f'gate report-only conditions {list(ro)!r}, the C8 gate file gives '
                       f'{[c for c in GATE_CONDS if c not in want]!r} for this run (O1 v11.33 3.9 rule 2b, OG2-7)')
    return want


def validate_gate(gate: dict, platform: Optional[str] = None, required=None) -> dict:
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
    ev = _recompute(gate, platform, required)
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
    def flag(k):
        s = row.get(k, '')
        if s not in ('', 'true', 'false'):
            raise LogError(f"{row.get('run_uid')}: {k} {s!r} is not true, false or empty")
        return None if s == '' else s == 'true'

    g = {'version': row.get('gate_version'), 'decide': [c for c in row.get('gate_decide', '').split(',') if c],
         'u_t': num(row.get('gate_u_t')), 'g1': {'q': row.get('g1_q'), 'state': row.get('g1_state')}}
    if row.get('gate_version') == o1_gate.GATE_VERSION_V1133:                   # O1 v11.33
        ro = [c for c in row.get('gate_report_only', '').split(',') if c]
        try:
            rq = list(o1_gate.required_of(ro))
        except ValueError as exc:
            raise LogError(f"{row.get('run_uid')}: gate_report_only: {exc}")
        g.update(report_only=ro, required=rq, gate_file_sha256=row.get('gate_file_sha256') or None,
                 level=row.get('trajectory') or None)
        if row.get('luoi_n_nut', ''):                                           # C8b grid (design v2)
            letters = {v: k for k, v in _G1_LETTER.items()}

            def bits(k):
                s_ = row.get(k, '')
                if any(ch not in '01' for ch in s_):
                    raise LogError(f"{row.get('run_uid')}: {k} {s_!r}: 0 / 1 per node")
                return [ch == '1' for ch in s_]
            try:
                g['luoi'] = dict(format=LUOI_FORMAT, gio_sha256=row.get('luoi_gio_sha256', ''),
                                 n_nut=int(row['luoi_n_nut']), d_dau_s=num(row.get('luoi_d_dau_s')),
                                 d_cuoi_s=num(row.get('luoi_d_cuoi_s')), buoc_max_s=num(row.get('luoi_buoc_max_s')),
                                 nut_tam=int(row.get('luoi_nut_tam', '')),
                                 g1_nut=[letters[ch] for ch in row.get('luoi_g1_nut', '')],
                                 hong_nut=bits('luoi_hong_nut'), cat_nut=bits('luoi_cat_nut'),
                                 g1_state=row.get('luoi_g1_state', ''))
            except (KeyError, ValueError) as exc:
                raise LogError(f"{row.get('run_uid')}: luoi columns: {exc}")
        elif any(row.get(c) for c in LUOI_COLS):
            raise LogError(f"{row.get('run_uid')}: luoi_* columns without luoi_n_nut")
    elif any(row.get(c) for c in LUOI_COLS):
        raise LogError(f"{row.get('run_uid')}: luoi_* columns on a {row.get('gate_version')!r} row")
    elif row.get('gate_file_sha256') or row.get('gate_report_only'):
        raise LogError(f"{row.get('run_uid')}: gate_file_sha256 / gate_report_only on a "
                       f"{row.get('gate_version')!r} row")
    ways = row.get('g1_ways', '')                                               # B81-3
    if ways:
        if ways not in ('1', '2') or (ways == '1' and row.get('g1_t_done_2', '') != ''):
            raise LogError(f"{row.get('run_uid')}: g1_ways {ways!r} with g1_t_done_2 {row.get('g1_t_done_2')!r}")
        g['g1'].update(t_done=[num(row.get(f'g1_t_done_{i}')) for i in range(1, int(ways) + 1)],
                       t_deadline=num(row.get('g1_t_deadline')), t1=num(row.get('g1_t1')),
                       prog_before_t1=flag('g1_prog_before_t1'), cond25_before_deadline=flag('g1_cond25_before_deadline'),
                       cat_t1=flag('g1_cat_t1'))
    elif any(row.get(k) for k in G1_COLS):
        raise LogError(f"{row.get('run_uid')}: G1 fields without g1_ways")
    for c in o1_gate.MEASURED:
        g[c] = {k: num(row.get(f'{c}_{k}')) for k in ('q', 'lo', 'hi')}
        g[c].update(n_undef=num(row.get(f'{c}_n_undef'), whole=True), state=row.get(f'{c}_state'))
        if any(row.get(f'{c}_{k}_chung') for k in DOMAIN_FIELDS):
            g[c]['chung'] = {k: num(row.get(f'{c}_{k}_chung'), whole=k == 'n_undef') for k in DOMAIN_FIELDS}
    return g


def map_refused(decide) -> bool:
    """The gate of a HIL run whose run_active edge was not mapped (only G1, G2 decide, O1 v11.12 §3.4; O1 v11.33: G1,
    G2 less a report-only G2): G4, never report only, does not decide."""
    d = set(decide)
    return bool(d) and d <= set(o1_gate.DECIDE_HIL)


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


def count_no_gate(tun: List[dict]) -> int:
    """SC3-2: dat rows outside the campaigns (historic rows) with empty gate columns, reported apart."""
    return sum(r.get('run_outcome') == 'dat' and r.get('campaign') not in CAMPAIGNS
               and not any(r.get(c) for c in GATE_COLS) for r in tun)


def query_files_needed(row: dict, outcome: str) -> bool:
    """R134 (b): the CLI 'end' of this open run needs --query-before and --query-after (comment above BOOT_FAIL)."""
    return outcome in VALID and boot_needed(row.get('platform'), row.get('stage'))


def check_query_rows(tun: List[dict], ds: str) -> None:
    """R134 (1) (R134C4-1 summary part, deferred at round 4): the summary reads the two gain_send --query outputs of a
    row again. A judged car / final_hil campaign row (query_files_needed) must name both files with their sha256;
    every named file must be there with that sha256 (a file changed after 'end' is refused); the two answers must be
    two questions one after the other (gain_line_check.query_problem) with the boots of the row's boot columns. A
    relative name is looked for in the dataset folder, then in the working folder. Rows outside the campaigns that
    name no file (historic rows) are not checked."""
    import gain_line_check as glc
    for r in tun:
        uid = r.get('run_uid')
        need = (r.get('campaign') in CAMPAIGNS and r.get('run_outcome') in VALID
                and boot_needed(r.get('platform'), r.get('stage')))
        if not need and not any(r.get(c) for c in QUERY_FILE_COLS):
            continue
        qs = []
        for when, cf, cs in (('before', *QUERY_FILE_COLS[:2]), ('after', *QUERY_FILE_COLS[2:])):
            p, sha = r.get(cf, ''), r.get(cs, '')
            if not p and not sha:
                if need:
                    raise LogError(f'{uid}: no {cf} / {cs}: the answer to the question {when} the run cannot be '
                                   'read again (R134 (1); a run without both answers is hong_he_do, '
                                   f'{BOOT_FAIL})')
                qs.append(None)
                continue
            if not p or not sha:
                raise LogError(f'{uid}: {cf} and {cs} go together, one is empty (R134 (1))')
            cand = [p] if os.path.isabs(p) else [os.path.join(ds, p), p]
            f = next((c for c in cand if os.path.isfile(c)), None)
            if f is None:
                raise LogError(f'{uid}: {cf} {p!r} not found (tried {cand}; R134 (1))')
            if sha256_file(f) != sha:
                raise LogError(f'{uid}: {f} changed after the run was logged (sha256 differs from {cs}; R134 (1))')
            qs.append(glc.read_query(f))
        if not need:
            continue
        why = glc.query_problem(qs[0], qs[1], check_run_boot=False)
        if why:
            raise LogError(f'{uid}: the two saved answers: {why} (R134 (1), R134C4-1)')
        cols = ((r.get('boot_before') or '').lower(), (r.get('boot_after') or '').lower())
        if cols != (qs[0]['boot'].lower(), qs[1]['boot'].lower()):
            raise LogError(f"{uid}: boot columns {cols} differ from the saved answers ({qs[0]['boot']}, "
                           f"{qs[1]['boot']}; R134 (1))")


# ── S05 v3.37 §4.3 (B8S-3): the outcome after the clip H comparison ────────────────────────────────────────────────
def so_h_outcome(o0: str, o1: Optional[str], vuot: bool, thieu: bool) -> tuple:
    """(outcome after the comparison, flags) by O1 §4 clip H (v11.27, v11.28; c7_calib_drift.combine): below eps_H
    the H0 outcome, no flag; clip H at the end missing: chua_ket_luan, flag thieu_H1, a lost run stays lost; above
    eps_H: lost in either -> hong_he_do, the same -> kept, different -> chua_ket_luan; flag calib_drift, plus
    calib_drift_split when the two differ (a lost run in one of the two is no split)."""
    if thieu:
        return ('hong_he_do' if o0 == 'hong_he_do' else 'chua_ket_luan'), ['thieu_H1']
    if not vuot:
        return o0, []
    if o1 not in OUTCOMES:
        raise LogError(f'above eps_H the H1 outcome must be one of {OUTCOMES}, got {o1!r}')
    if 'hong_he_do' in (o0, o1):
        return 'hong_he_do', ['calib_drift']
    if o0 == o1:
        return o0, ['calib_drift']
    return 'chua_ket_luan', ['calib_drift', 'calib_drift_split']


def _results_dir(r: dict, ds: str) -> Optional[str]:
    """The results_dir of a logged row as a folder that exists (a relative name: in the dataset folder, then in the
    working folder, as check_query_rows), or None."""
    p = r.get('results_dir', '')
    if not p:
        return None
    cand = [p] if os.path.isabs(p) else [os.path.join(ds, p), p]
    return next((c for c in cand if os.path.isdir(c)), None)


def _inside(path: str, folder: str) -> bool:
    """True when `path` is a file under `folder` (links resolved; another drive is outside)."""
    p, b = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(folder))
    try:
        return p != b and os.path.commonpath([p, b]) == b
    except ValueError:
        return False


H0_MAX_BYTES = 16 * 2 ** 20            # [DE XUAT] C9D2-6: files hashed when looking for the H0 scorer file


def h0_file(r: dict, ds: str) -> str:
    """C9D-1 (Codex C9 demo round 1, P1): the H0 scorer file of a logged car run, the file under its results_dir
    whose sha256 is the gate_json_sha256 of its row (the file given to 'end' with --gate-json, any name: C9D2-6).
    Files above H0_MAX_BYTES are not hashed (clips; a scorer file is a few kB). The file must resolve inside the
    folder, links resolved (C9D2-5)."""
    uid, sha = r.get('run_uid'), r.get('gate_json_sha256', '')
    d = _results_dir(r, ds)
    if d is None or not sha:
        raise LogError(f'run {uid}: no results_dir folder ({r.get("results_dir")!r}) or no gate_json_sha256 in the '
                       'log: the clip H comparison cannot be tied to the H0 scoring of this run (C9D-1)')
    for root, _, names in os.walk(d):
        for n in sorted(names):
            p = os.path.join(root, n)
            try:                                       # C9D3-1: a file that cannot be read is not the H0 file
                hit = os.path.isfile(p) and os.path.getsize(p) <= H0_MAX_BYTES and sha256_file(p) == sha
            except OSError:
                continue
            if hit:
                if not _inside(p, d):
                    raise LogError(f'run {uid}: the H0 scorer file {p} resolves outside the results_dir {d} '
                                   '(a link; C9D-1, C9D2-5)')
                return p
    raise LogError(f'run {uid}: no file under {d} with the gate_json_sha256 of the log (the H0 scorer file, C9D-1)')


def _finite_number(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise LogError(f'{what}: {x!r} is not a finite number >= 0')
    return float(x)


def check_score_final(r: dict, path: str, ds: str) -> tuple:
    """The output of c7_calib_drift.py gop for logged car run r, checked (command so-h and apply_so_h); returns
    (outcome after the comparison, flags, the calib_drift block). Checks:
      * the run is a car run of a campaign; outcome_h0 = its logged (provisional) outcome;
      * C9D-1: the file is under the run's results_dir and is the H0 scorer file of the run (h0_file) with only
        'outcome' and 'calib_drift' changed: a file of another run with the same outcome and gate is refused;
      * C9D-5: vuot is computed again, D_H_mm > eps_H_mm (clip H at the end missing: no D_H_mm, vuot true); above
        eps_H, outcome_h1 = the outcome of the kept H1 score (h1_score) and its gate, recomputed, gives it;
      * the outcome and flags of the file = so_h_outcome; the gate of the file (the H0 scoring) = the logged gate."""
    uid = r.get('run_uid')
    if r.get('platform') != 'HW' or r.get('campaign') not in CAMPAIGNS:
        raise LogError(f'run {uid}: the clip H comparison is for car runs of a campaign (O1 §4)')
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        cd = d['calib_drift']
        o0, o1, flags = cd['outcome_h0'], cd.get('outcome_h1'), list(cd['flags'])
        vuot, thieu = cd['vuot'], cd['thieu_H1']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LogError(f'score_final {path!r}: not an output of c7_calib_drift.py gop ({exc})')
    if not isinstance(vuot, bool) or not isinstance(thieu, bool) or not set(flags) <= set(SO_H_FLAGS):
        raise LogError(f'score_final {path!r}: vuot, thieu_H1 must be true/false, flags in {SO_H_FLAGS}')
    if o0 != r.get('run_outcome'):
        raise LogError(f"run {uid}: outcome_h0 {o0!r} of the file is not the logged outcome "
                       f"{r.get('run_outcome')!r} (another run, or the log was written with another scoring)")
    rd = _results_dir(r, ds)                                                       # C9D-1
    if rd is None or not _inside(path, rd):
        raise LogError(f'run {uid}: score_final {path!r} is not under the results_dir of the run '
                       f'({r.get("results_dir")!r}; C9D-1)')
    h0p = h0_file(r, ds)
    with open(h0p, encoding='utf-8') as f:
        h0 = json.load(f)
    if not isinstance(h0, dict) or h0.get('outcome') != o0 or \
            {k: v for k, v in d.items() if k not in ('outcome', 'calib_drift')} != \
            {k: v for k, v in h0.items() if k != 'outcome'}:
        raise LogError(f'run {uid}: score_final {path!r} is not the H0 scorer file of the run ({h0p}) with only '
                       f'outcome and calib_drift added (C9D-1)')
    if thieu:                                                                      # C9D-5
        if cd.get('D_H_mm') is not None or not vuot:
            raise LogError(f'run {uid}: thieu_H1 comes with vuot true and no D_H_mm (C9D-5)')
    else:
        dh = _finite_number(cd.get('D_H_mm'), f'run {uid}: D_H_mm')
        eps = _finite_number(cd.get('eps_H_mm'), f'run {uid}: eps_H_mm')
        if vuot != (dh > eps):
            raise LogError(f'run {uid}: vuot {vuot} but D_H_mm {dh} vs eps_H_mm {eps} (C9D-5)')
    if vuot and not thieu:
        h1 = cd.get('h1_score')
        if not isinstance(h1, dict) or h1.get('outcome') != o1:
            raise LogError(f'run {uid}: above eps_H, outcome_h1 {o1!r} must be the outcome of the kept H1 score '
                           f'(h1_score; C9D-5)')
        g1 = h1.get('gate_v3')                         # C9D2-1: a lost H1 may carry no gate, or null
        if o1 in VALID and g1 is None:
            raise LogError(f'run {uid}: the H1 score judged {o1} carries no gate_v3 block (C9D-5)')
        if g1 is not None:
            # C9D2-2: the same check as 'end' (check_outcome_vs_gate: gate recomputed, then outcome_vs_gate and the
            # floor rule of t1, viol_before_problem); the score carries no fail cause, so the exception before t1
            # (khong_dat, VIOL_BEFORE) is read from its G1 block (cat_t1)
            fc = VIOL_BEFORE if isinstance(g1, dict) and isinstance(g1.get('g1'), dict) and g1['g1'].get('cat_t1') \
                else ''
            try:
                # O1 v11.33: the H1 score uses the gate file of the logged row (checked against the file at end and in
                # the summary): same SHA, the required set of the row
                if isinstance(g1, dict) and (g1.get('gate_file_sha256') or '') != r.get('gate_file_sha256', ''):
                    raise LogError('gate file SHA of the H1 score differs from the logged run')
                req = (list(o1_gate.required_of([c for c in r.get('gate_report_only', '').split(',') if c]))
                       if r.get('gate_version') == o1_gate.GATE_VERSION_V1133 else None)
                check_outcome_vs_gate(o1, g1, 'HW', fc, req)
            except LogError as exc:
                raise LogError(f'run {uid}: outcome_h1 {o1} and the gate of the H1 score: {exc} (C9D-5, C9D2-2)')
    final, want = so_h_outcome(o0, o1, vuot, thieu)
    if d.get('outcome') != final or sorted(flags) != sorted(want):
        raise LogError(f"score_final {path!r}: outcome {d.get('outcome')!r}, flags {flags} disagree with "
                       f'the rule of O1 §4 ({final!r}, {want})')
    if o0 != 'hong_he_do' or any(r.get(c) for c in GATE_COLS):
        g = gate_columns(read_gate_json(path))
        diff = sorted(c for c, v in g.items() if c != 'gate_json_sha256' and str(r.get(c, '')) != v)
        if diff:
            raise LogError(f'run {uid}: the gate of the file (H0 scoring) differs from the logged gate in {diff}')
    return final, flags, cd


def so_h(ds: str, run_uid: str, score_final: str) -> dict:
    """Command so-h: one row of SO_H for car run `run_uid` from the output of c7_calib_drift.py gop (score_final.json:
    'outcome' and the 'calib_drift' block), checked by check_score_final; any run below eps_H too (outcome kept,
    C9D-2: the row shows the comparison was made). The run must be in the log and not given before."""
    tun = live_rows(read_rows(os.path.join(ds, TUNING)))
    r = next((x for x in tun if x.get('run_uid') == run_uid), None)
    if r is None:
        raise LogError(f'run {run_uid} not in the tuning log')
    if r.get('platform') != 'HW' or r.get('campaign') not in CAMPAIGNS:
        raise LogError(f'run {run_uid}: the clip H comparison is for car runs of a campaign (O1 §4)')
    if any(x.get('run_uid') == run_uid for x in read_rows(os.path.join(ds, SO_H))):
        raise LogError(f'run {run_uid}: already in {SO_H} (one row per run, append only)')
    final, flags, cd = check_score_final(r, score_final, ds)
    o0, vuot = cd['outcome_h0'], cd['vuot']
    row = dict(run_uid=run_uid, outcome_tam=o0, outcome_sau=final, flags=','.join(flags),
               vuot='true' if vuot else 'false', D_H_mm=_num(cd.get('D_H_mm')),
               score_final_file=os.path.abspath(score_final), score_final_sha256=sha256_file(score_final),
               ghi_luc=_iso(now()))
    append_row(os.path.join(ds, SO_H), SO_H_COLS, row)
    return row


def is_score_final(path: str) -> bool:
    """B8S2-1: True when the file is an output of c7_calib_drift.py gop (it carries the 'calib_drift' block); its gate
    is that of the H0 scoring while its outcome may be the one after the comparison, so it goes to so-h, not end."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and 'calib_drift' in d


def apply_so_h(tun: List[dict], ds: str) -> List[dict]:
    """The live tuning rows with the outcome after the clip H comparison (SO_H rows, checked again: run once, in the
    log, car run of a campaign, provisional outcome as logged, rule of O1 §4, score_final file unchanged). A run that
    became hong_he_do gets fail cause FAIL_SO_H; every changed or flagged row gets 'so_h_flags'. Rows are copies."""
    rows = read_rows(os.path.join(ds, SO_H))
    by_uid = {}
    for s in rows:
        uid = s.get('run_uid')
        if uid in by_uid:
            raise LogError(f'{SO_H}: run {uid} on two rows')
        by_uid[uid] = s
    known = {r.get('run_uid') for r in tun}
    out = []
    for r in tun:
        if r.get('fail_cause') == FAIL_SO_H or r.get(SO_H_LOST):                  # C9D2-4
            raise LogError(f"run {r.get('run_uid')}: fail cause {FAIL_SO_H} in the tuning log: only the clip H "
                           f'comparison gives it ({SO_H}), never a logged row (C9D2-4)')
        s = by_uid.get(r.get('run_uid'))
        if s is None:
            out.append(r)
            continue
        if r.get('platform') != 'HW' or r.get('campaign') not in CAMPAIGNS or s.get('outcome_tam') != r.get('run_outcome'):
            raise LogError(f"{SO_H}: run {r.get('run_uid')}: not a car run of a campaign with provisional outcome "
                           f"{s.get('outcome_tam')!r}")
        flags = [x for x in (s.get('flags') or '').split(',') if x]
        p = s.get('score_final_file', '')
        if not os.path.isfile(p) or sha256_file(p) != s.get('score_final_sha256'):
            raise LogError(f"{SO_H}: run {r.get('run_uid')}: {p!r} missing or changed since so-h")
        try:                                           # C9D-1, C9D-5: the same checks as at so-h
            final, want, _ = check_score_final(r, p, ds)
        except LogError as exc:
            raise LogError(f'{SO_H}: {exc}')
        if (final, sorted(want)) != (s.get('outcome_sau'), sorted(flags)):
            raise LogError(f"{SO_H}: run {r.get('run_uid')}: outcome_sau / flags disagree with the rule of O1 §4")
        r = dict(r, run_outcome=final, so_h_flags=','.join(flags))
        if final == 'hong_he_do' and s['outcome_tam'] != 'hong_he_do':
            r['fail_cause'] = FAIL_SO_H
            r[SO_H_LOST] = '1'                         # C9D2-4: from the checked SO_H row, not from a typed cause
        out.append(r)
    missing = sorted(set(by_uid) - known)
    if missing:
        raise LogError(f'{SO_H}: runs {missing} not in the tuning log')
    return mark_retro_stops(out)


def mark_retro_stops(tun: List[dict]) -> List[dict]:
    """S05 v3.38 4.3 (B8S2-2; Liam 17/11 at THESIS_B8 "Duyệt, áp luôn (Recommended)"): a run taken after a stop that
    only the comparison at the end of the session showed (the N = 3 lost-run streak of 4.6 complete only with a run
    that became hong_he_do at so-h, mark SO_H_LOST of apply_so_h, C9D2-4) stays in the log with flag RETRO_FLAG: out of
    the denominator, blocks and the final table; its time and N_HW still count. A stop seen without so-h keeps 4.6 (a run
    after it is an error). Rows are copies where flagged."""
    groups = defaultdict(list)
    for i, r in enumerate(tun):
        if r.get('stage') in ('final', 'final_hil'):
            groups[final_key(r)].append(i)
    out = list(tun)
    for idx in groups.values():
        streak, retro, stopped = 0, False, False
        for i in idx:
            r = out[i]
            if stopped:
                if retro:
                    out[i] = dict(r, **{RETRO_FLAG: '1'})
                continue
            if r.get('run_outcome') == 'hong_he_do':
                streak += 1
                retro = retro or r.get(SO_H_LOST) == '1'          # C9D2-4: set by apply_so_h only
            else:
                streak, retro = 0, False
            stopped = streak >= FINAL_LOST_N
    return out


def so_h_missing(tun: List[dict], ds: str, stages: Optional[Iterable[str]] = None) -> List[dict]:
    """C9D-2 (Codex C9 demo round 1; Liam 17/11 at CODE_C13 "Sửa cả 7 + vòng 2 (Recommended)"): the judged car runs
    of the campaigns (logged outcome in VALID; stage in `stages` when given) without a row in SO_H, i.e. whose clip H
    comparison was not given (so-h takes a run below eps_H too, outcome kept)."""
    done = {s.get('run_uid') for s in read_rows(os.path.join(ds, SO_H))}
    st = None if stages is None else set(stages)
    return [r for r in tun if r.get('platform') == 'HW' and r.get('campaign') in CAMPAIGNS
            and r.get('run_outcome') in VALID and (st is None or r.get('stage') in st)
            and r.get('run_uid') not in done]


CONFIRM_PER_LEVEL = 2                  # S05 4.3: 2 new valid runs at every level of the gate (2 x |G|)


def confirm_block_problem(tun_after: List[dict], ds: str, campaign: str, gain_set: str,
                          frozen_at: str, levels: Optional[Iterable[str]] = None) -> Optional[str]:
    """C9D-4 (Codex C9 demo round 1; Liam 17/11 at CODE_C13 "Sửa cả 7 + vòng 2 (Recommended)"): None, or why the stop
    reason khoi_xac_nhan_dat does not hold in the log. The last block of confirm car runs of the frozen gain set (rows
    of the campaign, platform HW, stage confirm, gain_set_id = the frozen set; the block = the block_id of the last
    such row in log order, not empty): every run judged before the comparison has its so-h row; the valid runs after
    it (S05 4.3: a run lost to the measurement, before or at so-h, is replaced and out of the denominator) are at
    least one and all dat; frozen_at is not earlier than the latest so-h of the block. levels (the gate file, given
    to summary, not to freeze; S05 v3.40 4.3 Q1, Liam 17/11 at THESIS "Nhờ C13 thêm kiểm (Recommended)"): at least
    CONFIRM_PER_LEVEL valid runs at every level (trajectory) of the gate. tun_after: the live rows after apply_so_h
    (a row without so-h keeps its logged outcome)."""
    conf = [r for r in tun_after if r.get('campaign') == campaign and r.get('platform') == 'HW'
            and r.get('stage') == 'confirm' and r.get('gain_set_id') == gain_set]
    if not conf:
        return f'no confirm car run of {gain_set} in the log'
    bid = conf[-1].get('block_id', '')
    if not bid:
        return f"the last confirm car run of {gain_set} ({conf[-1].get('run_uid')}) has no block_id"
    block = [r for r in conf if r.get('block_id') == bid]
    so = {s.get('run_uid'): s for s in read_rows(os.path.join(ds, SO_H))}
    miss = [r.get('run_uid') for r in block if r.get('run_uid') not in so and r.get('run_outcome') in VALID]
    if miss:
        return f'block {bid}: runs {miss} without their clip H comparison (so-h)'
    valid = [r for r in block if r.get('run_outcome') in VALID]
    if not valid:
        return f'block {bid}: no valid run after the clip H comparison'
    bad = [f"{r.get('run_uid')}={r.get('run_outcome')}" for r in valid if r.get('run_outcome') != 'dat']
    if bad:
        return f"block {bid}: not all dat after the clip H comparison ({', '.join(bad)})"
    if levels is not None:                                                         # Q1: 2 x |G|
        short = [f"{lv} {sum(r.get('trajectory') == lv for r in valid)}" for lv in levels
                 if sum(r.get('trajectory') == lv for r in valid) < CONFIRM_PER_LEVEL]
        if short:
            return (f"block {bid}: fewer than {CONFIRM_PER_LEVEL} valid runs at gate level(s) {', '.join(short)} "
                    '(S05 4.3, 2 x |G|)')
    try:
        last = max(_parse_t(so[r['run_uid']]['ghi_luc']) for r in block if r.get('run_uid') in so)
        t_frz = _parse_t(frozen_at)
    except (ValueError, TypeError, KeyError) as exc:
        return f'block {bid}: time stamps not readable ({exc})'
    if t_frz < last:
        return f'frozen_at {frozen_at} is earlier than the last so-h of block {bid} ({_iso(last)})'
    return None


def row_required(r: dict, gate_file=None):
    """O1 v11.33: the conditions that must decide for a run (row fields trajectory, platform, stage, campaign) from
    the C8 gate file (gate_c8.GateC8); None without a gate file (all five)."""
    if gate_file is None:
        return None
    try:
        return gate_file.required(r.get('trajectory', ''), r.get('platform', ''), r.get('stage', ''),
                                  r.get('campaign', ''))
    except gate_c8.GateFileError as exc:
        raise LogError(f"{r.get('run_uid')}: {exc}")


def gate_file_problem(r: dict, gate_sha: str, gate_file=None) -> Optional[str]:
    """O1 v11.33 (OG2-7): None when the gate file SHA of a run's gate (gate_sha, '' = none) fits the gate file in use,
    else why not. A gate scored with a gate file needs that file here; a campaign run whose level may carry
    report-only conditions (gate_c8.mask_applies: the floor, the HIL candidate check of B) must be scored with the
    gate file in use (the run record carries its level and the file SHA)."""
    if gate_sha and gate_file is None:
        return (f'its gate was scored with the C8 gate file {gate_sha[:12]}..., not given here (--gate-file): the '
                'report-only conditions are recomputed from the file (OG2-7)')
    if gate_file is None:
        return None
    if gate_sha and gate_sha != gate_file.sha256:
        return f'its gate was scored with the gate file {gate_sha[:12]}..., not {gate_file.sha256[:12]}... (given)'
    try:
        mask = gate_c8.mask_applies(r.get('platform', ''), r.get('stage', ''), r.get('campaign', ''))
    except gate_c8.GateFileError as exc:
        return str(exc)
    if not gate_sha and mask and r.get('campaign') in CAMPAIGNS:
        return ('a campaign run on the floor / HIL candidate check of B scored without the C8 gate file (its record '
                'carries the level and the gate file SHA, O1 v11.33 OG2-7)')
    return None


def check_gate_rows(tun: List[dict], gate_file=None) -> None:
    """GV3-1, GV3-R2-1: every logged row that carries a gate is recomputed from its q, lo, hi, n_undef
    (version, deciding set of its platform, thresholds, states); its run_outcome must agree as at `end`.
    Rows whose gate columns are ALL empty (before gate v3, or written without --gate-json) are not
    checked; any non-empty gate cell makes the row a gate row, with a supported version (GV3-R3-1).
    O1 v11.33: gate_file (gate_c8.GateC8, the summary's --gate-file) gives the required set of every gate row
    (row_required) and the SHA its gate must carry (gate_file_problem)."""
    for r in tun:
        judged = r.get('run_outcome') in VALID
        camp = r.get('campaign') in CAMPAIGNS           # [CACH HIEU] historic rows (no campaign): SC3-2 note above
        # R134 (b); B81-2: a judged car / final_hil campaign row needs both boots, given or not
        if judged and (r.get('boot_before') or r.get('boot_after')
                       or (camp and boot_needed(r.get('platform'), r.get('stage')))):
            bbad = boot_problem(r.get('boot_before'), r.get('boot_after'))
            if bbad:
                raise LogError(f"{r.get('run_uid')}: run_outcome {r.get('run_outcome')!r} but {bbad} (S05 4.7 (iv); "
                               f'a run failed by operation is hong_he_do, {BOOT_FAIL})')
        if not any(r.get(c) for c in GATE_COLS):
            # R134 (d) SC3-2 (HIL rows have their own rule for every judged outcome, Codex S315-1)
            if r.get('run_outcome') == 'dat' and r.get('campaign') in CAMPAIGNS and r.get('platform') != 'HIL':
                raise LogError(f"{r.get('run_uid')}: run_outcome 'dat' without gate columns ({NO_GATE_EVIDENCE}, "
                               'SC3-2): a campaign run is never dat without the gate')
            if judged and camp and r.get('platform') == 'HW':                           # B8-1 B81-1
                raise LogError(f"{r.get('run_uid')}: car run {r.get('run_outcome')} without gate columns (the camera "
                               'scorer output, S05 v3.20 8.1 C9 (a)): not in the summary')
            if r.get('fail_cause') == VIOL_BEFORE:
                raise LogError(f"{r.get('run_uid')}: {viol_before_problem(r.get('run_outcome', ''), VIOL_BEFORE, None)}")
            continue
        g = gate_from_row(r)
        gbad = gate_file_problem(r, r.get('gate_file_sha256', ''), gate_file)
        if gbad:
            raise LogError(f"{r.get('run_uid')}: {gbad}")
        try:
            ev = _recompute(g, r.get('platform'), row_required(r, gate_file))
        except LogError as exc:
            raise LogError(f"{r.get('run_uid')}: {exc}")
        why = (outcome_vs_gate(r.get('run_outcome', ''), ev, g['decide'], r.get('fail_cause', ''))
               or viol_before_problem(r.get('run_outcome', ''), r.get('fail_cause', ''), g))
        if why:
            raise LogError(f"{r.get('run_uid')}: run_outcome {r.get('run_outcome')!r} disagrees with its gate "
                           f"({r.get('gate_version')}: {why})")


def gate_columns(gate: dict, gate_sha256: str = '') -> dict:
    """GATE_COLS of one run from a gate dict (fwsim/o1_gate.gate). q of g1 is the evaluator's
    'completed' (true / false / inconclusive)."""
    out = {'gate_version': gate['version'], 'gate_decide': ','.join(gate['decide']),
           'gate_json_sha256': gate_sha256, 'gate_u_t': _num(gate.get('u_t')),
           'g1_q': str(gate['g1'].get('q')), 'g1_state': gate['g1'].get('state', ''),
           'gate_file_sha256': gate.get('gate_file_sha256') or '',                       # O1 v11.33
           'gate_report_only': ','.join(gate.get('report_only') or [])}
    lb = gate.get('luoi')                                                          # C8b grid (design v2)
    out.update({c: '' for c in LUOI_COLS})
    if lb is not None:
        out.update(luoi_gio_sha256=lb['gio_sha256'], luoi_n_nut=str(int(lb['n_nut'])), luoi_d_dau_s=_num(lb['d_dau_s']),
                   luoi_d_cuoi_s=_num(lb['d_cuoi_s']), luoi_buoc_max_s=_num(lb['buoc_max_s']),
                   luoi_nut_tam=str(int(lb['nut_tam'])), luoi_g1_nut=''.join(_G1_LETTER[x] for x in lb['g1_nut']),
                   luoi_hong_nut=''.join('1' if x else '0' for x in lb['hong_nut']),
                   luoi_cat_nut=''.join('1' if x else '0' for x in lb['cat_nut']), luoi_g1_state=lb['g1_state'])
    def cell(v):
        return '' if v is None or (isinstance(v, float) and math.isnan(v)) else _num(v)

    for g in o1_gate.MEASURED:
        c = gate.get(g) or {}
        for k in ('q', 'lo', 'hi'):
            out[f'{g}_{k}'] = cell(c.get(k))
        out[f'{g}_n_undef'] = str(int(c['n_undef'])) if c.get('n_undef') is not None else ''
        out[f'{g}_state'] = c.get('state', '')
        com = c.get('chung') or c                     # B81-3: one domain -> the common set equals the union
        for k in ('q', 'lo', 'hi'):
            out[f'{g}_{k}_chung'] = cell(com.get(k))
        out[f'{g}_n_undef_chung'] = str(int(com['n_undef'])) if com.get('n_undef') is not None else ''
    g1 = gate.get('g1') or {}
    if g1.get('t_done') is not None:                   # B81-3: the G1 fields (checked by validate_gate first)
        td = g1['t_done']
        out.update(g1_ways=str(len(td)), g1_t_deadline=cell(g1.get('t_deadline')), g1_t_done_1=cell(td[0]),
                   g1_t_done_2=cell(td[1]) if len(td) > 1 else '', g1_t1=cell(g1.get('t1')))
        for k in ('prog_before_t1', 'cond25_before_deadline', 'cat_t1'):
            v = g1.get(k)
            out[f'g1_{k}'] = '' if v is None else ('true' if v else 'false')
    return out


def viol_before_problem(outcome: str, fail_cause: str, gate: Optional[dict]) -> Optional[str]:
    """O1 3.5 R2V-4 (B81-3): the floor exception before t1 is khong_dat with fail cause VIOL_BEFORE and a gate whose
    G1 has cat_t1 true; one without the other is an error."""
    g1 = (gate or {}).get('g1') or {}
    lb = (gate or {}).get('luoi')
    if lb is not None:                         # C8b grid (design v2 section 2 points 4, 5; Liam 17/11)
        cat = all(lb['cat_nut'])
        if cat != (fail_cause == VIOL_BEFORE) or (cat and outcome != 'khong_dat'):
            return (f'grid with the exception before t1 at {sum(lb["cat_nut"])} of {lb["n_nut"]} nodes, outcome '
                    f'{outcome!r}, fail cause {fail_cause!r}: khong_dat with {VIOL_BEFORE} exactly when every node '
                    'has it (Liam 17/11 at CODE_GATE)')
        if all(lb['hong_nut']) and outcome != 'hong_he_do':
            return f'every grid node lost to the measurement: the run is hong_he_do, not {outcome} (C8b v4 section 3)'
        return None
    cat = bool(g1.get('cat_t1'))
    if cat != (fail_cause == VIOL_BEFORE) or (cat and outcome != 'khong_dat'):
        return (f'cat_t1 {cat} with outcome {outcome!r}, fail cause {fail_cause!r}: the floor exception before t1 is '
                f'khong_dat with fail cause {VIOL_BEFORE} and cat_t1 true (O1 3.5 R2V-4), and only that')
    floor = isinstance(g1.get('t_done'), list) and len(g1['t_done']) == 2
    if floor and g1.get('t1') is not None and not cat and outcome in VALID:          # R134C4-3
        return (f'floor run with t1 {g1.get("t1")!r} (tracking error or a gap above t_trong) and no violation '
                f'before it (cat_t1 false): a run lost to the measurement, hong_he_do, not {outcome} (O1 3.5)')
    return None


def check_outcome_vs_gate(outcome: str, gate: dict, platform: Optional[str] = None, fail_cause: str = '',
                          required=None) -> None:
    """The run outcome typed by the operator against the gate, validated and recomputed first (GV3-5);
    required: the required set of the run from the gate file (row_required; None = all five)."""
    ev = validate_gate(gate, platform, required)
    why = outcome_vs_gate(outcome, ev, gate['decide'], fail_cause) or viol_before_problem(outcome, fail_cause, gate)
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


def boot_needed(platform: str, stage: str) -> bool:
    """R134 (b): the CLI 'end' of a judged run needs both gain_send --query files (see BOOT_FAIL)."""
    return platform == 'HW' or (platform == 'HIL' and stage == 'final_hil')


def boot_problem(before, after) -> Optional[str]:
    """None when the two boot answers show one ESP32 boot over the run, else why not."""
    ok = [b for b in (before, after) if b and all(c in '0123456789abcdefABCDEF' for c in str(b))]
    if len(ok) < 2:
        return f'boot {"before" if not ok or ok[0] != before else "after"} the run missing or unreadable'
    if str(before).lower() != str(after).lower():
        return f'boot {before} before the run, {after} after it: the ESP32 restarted'
    return None


def end(ds: str, outcome: str, completed: Optional[int] = None, e_perp_max: Optional[float] = None,
        fail_cause: str = '', results_dir: str = '', notes: str = '', gate: Optional[dict] = None,
        gate_sha256: str = '', boot_before: Optional[str] = None, boot_after: Optional[str] = None,
        query_problem: Optional[str] = None, queries: Optional[tuple] = None, gate_file=None) -> dict:
    """query_problem (R134C3-3): why the two query answer files given on the command line are not two
    questions before and after the run (gain_line_check.query_problem); handled as missing / different boot.
    queries (R134C4-1): the two answers as read by gain_line_check.read_query (before, after; None = missing);
    end checks them itself (query_problem, without the run's boot) and takes boot_before / boot_after from them.
    A judged car / final_hil run needs them: two boot strings alone do not show two questions.
    gate_file (O1 v11.33): the C8 gate file (gate_c8.GateC8, CLI --gate-file); the run's required set is taken from it
    (row_required), and a gate scored with a gate file needs it (gate_file_problem)."""
    run = _require_open(ds)
    if outcome not in OUTCOMES:
        raise LogError(f'--outcome one of {OUTCOMES}')
    if fail_cause == FAIL_SO_H:                                                    # C9D2-4
        raise LogError(f'--fail-cause {FAIL_SO_H} is given by the clip H comparison (so-h) only')
    plat, stage = run['row'].get('platform'), run['row'].get('stage')
    if queries is not None:                            # R134C4-1: the two answers checked here, not by the caller
        import gain_line_check as glc
        qb, qa = queries
        query_problem = query_problem or glc.query_problem(qb, qa, check_run_boot=False)
        boot_before, boot_after = (qb or {}).get('boot', ''), (qa or {}).get('boot', '')
    asked = boot_before is not None or boot_after is not None or query_problem is not None
    bbad = query_problem or (boot_problem(boot_before, boot_after) if asked else None)
    if not asked and (fail_cause == BOOT_FAIL or (outcome in VALID and boot_needed(plat, stage))):
        bbad = boot_problem('', '')                    # R134C2-7, B81-2: no answer at all = both missing
    if not bbad and queries is None and outcome in VALID and boot_needed(plat, stage):
        bbad = ('the two query answers (gain_send --query outputs read by gain_line_check.read_query) not given: '
                'two boot strings do not show two questions (R134C4-1)')
    if bbad and not (outcome == 'hong_he_do' and fail_cause == BOOT_FAIL):   # R134 (b)
        raise LogError(f'{bbad}: a run failed by operation, outcome hong_he_do with fail cause {BOOT_FAIL} '
                       '(S05 4.7 (iv), R3V-9)')
    if fail_cause == BOOT_FAIL and not bbad:
        raise LogError(f'fail cause {BOOT_FAIL} needs the two boot answers showing it (missing or different)')
    if fail_cause == MAP_REFUSED and (gate is None or run['row'].get('platform') != 'HIL'):
        raise LogError(f'--fail-cause {MAP_REFUSED}: a HIL run with its scorer output (--gate-json) showing the '
                       'refused mapping (S05 v3.15 4.5 step 2)')
    if gate is not None:
        if not results_dir:
            raise LogError('--gate-json needs --results-dir (the run must be scorable again, GV3-6)')
        gbad = gate_file_problem(run['row'], gate.get('gate_file_sha256') or '', gate_file)
        if gbad:
            raise LogError(f'--gate-json: {gbad}')
        check_outcome_vs_gate(outcome, gate, run['row'].get('platform'), fail_cause,
                              row_required(run['row'], gate_file))
    if outcome == 'hong_he_do' and not fail_cause:
        raise LogError('hong_he_do needs --fail-cause (S05 §6.1)')
    if run['row'].get('platform') == 'HIL' and outcome in VALID and gate is None:
        raise LogError(f'a HIL run judged {outcome} needs its scorer output (--gate-json): S05 v3.15 4.5 step 2 '
                       '(a run without the run_active edge mapping is a lost run)')        # Codex S315-1
    if plat == 'HW' and outcome in VALID and gate is None:                                   # B8-1 B81-1
        raise LogError(f'a car run judged {outcome} needs the camera scorer output (--gate-json, with the two frame '
                       'domains and the G1 fields of the floor; S05 v3.20 8.1 C9 (a), O1 3.3, 3.5)')
    if outcome == 'dat' and gate is None:                                                    # R134 (d) SC3-2
        raise LogError(f"a run judged dat needs its gate evidence (--gate-json), on {run['row'].get('platform')} "
                       'too (SC3-2: no dat without the gate)')
    if gate is None and fail_cause == VIOL_BEFORE:                                           # R2V-4 (B81-3)
        raise LogError(viol_before_problem(outcome, fail_cause, None))
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
    if asked:
        row.update(boot_before=boot_before or '', boot_after=boot_after or '')
    if queries is not None:                            # R134 (1): the files the two answers came from
        for (cf, cs), q in zip((QUERY_FILE_COLS[:2], QUERY_FILE_COLS[2:]), queries):
            p = (q or {}).get('_path')
            if p and os.path.isfile(p):
                row.update({cf: os.path.abspath(p), cs: sha256_file(p)})
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
    at = at or _iso(now())
    if stop_reason == 'khoi_xac_nhan_dat':                                         # C9D-4
        why = confirm_block_problem(apply_so_h(live_rows(read_rows(os.path.join(ds, TUNING))), ds), ds, campaign,
                                    gain_set, at)
        if why:
            raise LogError(f'--stop-reason khoi_xac_nhan_dat: {why} (C9D-4)')
    row = {'campaign': campaign, 'frozen_at': at, 'stop_reason': stop_reason,
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
                'explore_N_HIL', 'explore_N_HW', 'so_h_chua_n']
VALID = ('dat', 'khong_dat', 'chua_ket_luan')
_COUNT = {'dat': 'pass', 'khong_dat': 'fail', 'chua_ket_luan': 'inconclusive', 'hong_he_do': 'invalid'}
_SAME_CONDITION = ('campaign', 'platform', 'stage', 'trajectory', 'speed_mps', 'gain_set_id')


def read_gate_file(path: str):
    """EA-9, O1 v11.33: the locked C8 gate file (format gate_c8_v1, tools/gate_c8.py: levels, report-only conditions
    per level), never typed again; a gate_c8.GateC8 (its sha256 is that of the file bytes)."""
    try:
        return gate_c8.read_gate_c8(path)
    except gate_c8.GateFileError as exc:
        raise LogError(str(exc))


def read_gate(path: str):
    """(levels, sha256) of the locked gate file (read_gate_file). The CSV with one column `level` (before O1 v11.33)
    is no longer read."""
    gf = read_gate_file(path)
    return list(gf.levels), gf.sha256


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
    campaign, platform, stage, task, speed and gain set, earlier than itself in the log (row order, as
    _final_stops; Codex S315-R3-2: not by wall_start, which can go back); a run is replaced once."""
    by_uid, pos = {}, {}
    for i, r in enumerate(tun):                    # Codex S315-R4-1: a run_uid names one row
        uid = r.get('run_uid')
        if uid in by_uid:
            raise LogError(f'run_uid {uid} on two rows (rows {pos[uid] + 1} and {i + 1} of the live log)')
        if uid:
            by_uid[uid], pos[uid] = r, i
    seen = {}
    for i, r in enumerate(tun):
        ref = r.get('replaces_run_uid')
        if not ref:
            continue
        if not r.get('run_uid'):
            raise LogError(f'row {i + 1} of the live log replaces {ref} but has no run_uid')
        old = by_uid.get(ref)
        if not old or old.get('run_outcome') != 'hong_he_do':
            raise LogError(f"{r.get('run_uid')}: replaces_run_uid {ref} is not a hong_he_do run (§4.7)")
        diff = [k for k in _SAME_CONDITION if old.get(k, '') != r.get(k, '')]
        if diff:
            raise LogError(f"{r.get('run_uid')}: replaces {ref} with another {', '.join(diff)} (a replacement "
                           f"runs the same condition, §4.3, §4.6)")
        if ref == r.get('run_uid') or pos[ref] > i:
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
            if r.get(RETRO_FLAG):                      # S05 v3.38 4.3: kept, out of the streak and the counts
                continue
            if key in stopped:
                raise LogError(f"{r.get('run_uid')}: {key[4]}@{key[5]} of {key[2] or key[1]} stopped after "
                               f"{FINAL_LOST_N} lost runs in a row (§4.6), no run after that")
            lost = lost + 1 if r.get('run_outcome') == 'hong_he_do' else 0
            if lost >= FINAL_LOST_N:
                stopped.add(key)
    return stopped


def summary(ds: str, gate: Optional[Iterable[str]] = None, gate_sha256: str = '', gate_file=None) -> List[dict]:
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
        stop_cause) checked against the counts above ('ok' / 'mismatch'); khoi_xac_nhan_dat checked by
        confirm_block_problem (C9D-4; with `gate`: also 2 valid runs per gate level, S05 v3.40 Q1); the other
        codes 'khong_kiem_duoc';
      * so_h_chua_n (C9D-2): judged car runs of the campaign without their clip H comparison (so-h);
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
        locked gate file by read_gate); empty without `gate` or without any final run.
      * O1 v11.33: gate_file (gate_c8.GateC8): the levels and the SHA come from it (gate, gate_sha256 must then be
        None / '' or the same), and every gate row is checked against it (check_gate_rows)."""
    if gate_file is not None:
        if (gate is not None and list(gate) != list(gate_file.levels)) or (gate_sha256 and gate_sha256 != gate_file.sha256):
            raise LogError('summary: gate / gate_sha256 given with another gate file')
        gate, gate_sha256 = list(gate_file.levels), gate_file.sha256
    gate = None if gate is None else list(gate)     # used per campaign (gate_result, Q1)
    raw = live_rows(read_rows(os.path.join(ds, TUNING)))
    tun = apply_so_h(raw, ds)                       # S05 v3.37 4.3: outcomes after the clip H comparison
    so_h_open = defaultdict(int)                    # C9D-2: judged car runs whose clip H comparison is not given
    for r in so_h_missing(raw, ds):
        so_h_open[r.get('campaign', '')] += 1
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
    check_gate_rows(raw, gate_file)                 # GV3-1, GV3-R2-1: gate rows recomputed (logged outcome)
    check_query_rows(raw, ds)                       # R134 (1): the two saved answers read again
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
        if stage in ('final', 'final_hil') and r.get(RETRO_FLAG):     # S05 v3.38 4.3: time, N_HW above only
            finals[c][f'{RETRO_FLAG}_n'] += 1
        elif stage in ('final', 'final_hil'):
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
        elif reason == 'khoi_xac_nhan_dat':          # C9D-4: the last confirm block, after the clip H comparison
            why = confirm_block_problem(tun, ds, c, f.get('gain_set_id', ''), f['frozen_at'],
                                        levels=None if gate is None else list(gate))      # Q1 with the gate file
            check = 'mismatch' if why else 'ok'
            if why:
                print(f'warning: {c} khoi_xac_nhan_dat: {why}', file=sys.stderr)
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
               'explore_N_HW': str(int(a.get('X_N_HW', 0))), 'so_h_chua_n': str(so_h_open.get(c, 0))}
        row.update({k: str(v) for k, v in sorted(finals.get(c, {}).items())})
        if c in map_rep:                            # extra columns, only for a campaign with refused mappings
            row['hil_map_refused_n'] = str(map_rep[c]['n'])
            row['hil_map_stopped'] = '; '.join(sorted(map_rep[c]['stopped']))
        out.append(row)
    if unassigned:
        print(f'note: {unassigned} tuning row(s) without campaign left out', file=sys.stderr)
    return out


def write_summary(ds: str, out_path: str, gate: Optional[Iterable[str]] = None, gate_sha256: str = '',
                  gate_file=None) -> List[dict]:
    rows = summary(ds, gate, gate_sha256, gate_file)
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


def fake_gate(e_trk: float = 0.03, mapped: bool = True, completed: str = 'true', gate_file=None,
              level: str = 'O1-F05', platform: str = 'HW', stage: str = 'final', campaign: str = 'A-PID',
              e_theta_deg: float = 3.0) -> dict:
    """A made-up gate (FAKE: demo and tests only, no measurement): 20 frames of constant errors, e_trk as given;
    mapped: U_t known and all five decide, else the HIL set without the run_active edge mapping (G1, G2).
    gate_file (gate_c8.GateC8, O1 v11.33): a gate_v3.3 gate scored with it, the report-only conditions of `level`
    for a run of platform / stage / campaign."""
    n = 20
    frames = {'e_perp': [0.02] * n, 'e_trk': [e_trk] * n, 'e_bao': [0.04] * n, 'e_theta_cmd_deg': [e_theta_deg] * n}
    kw = {}
    if gate_file is not None:
        kw = dict(report_only=gate_file.run_report_only(level, platform, stage, campaign), level=level,
                  gate_file_sha256=gate_file.sha256)
    return _plain(o1_gate.gate(frames, completed, v_cmd_max=0.729, wz_cmd_max=1.634, u_t=0.0015 if mapped else None,
                               decide=o1_gate.DECIDE_ALL if mapped else o1_gate.DECIDE_HIL, note='FAKE', **kw))


_FAKE_Q = {'dir': None, 'n': 0}


def fake_queries(boot: str = '4753dc47', mac: str = 'aa:bb', t0: float = 0.0, folder: Optional[str] = None,
                 gid: int = 1) -> tuple:
    """FAKE (demo and tests only): two read_query answers of one boot, asked one after the other (R134C4-1), read
    from two gain_send --query output files (R134 (1): 'end' records them, the summary reads them again). folder:
    where the two files go (gain_query_before.txt, gain_query_after.txt); None: new names in one temporary folder
    of the process, removed at exit."""
    import gain_line_check as glc
    if folder is None:
        if _FAKE_Q['dir'] is None:
            import atexit
            import shutil
            import tempfile
            _FAKE_Q['dir'] = tempfile.mkdtemp(prefix='rq3_fake_q_')
            atexit.register(shutil.rmtree, _FAKE_Q['dir'], True)
        _FAKE_Q['n'] += 1
        folder, names = _FAKE_Q['dir'], tuple(f'q{_FAKE_Q["n"]}_{w}.txt' for w in ('before', 'after'))
    else:
        names = ('gain_query_before.txt', 'gain_query_after.txt')
    os.makedirs(folder, exist_ok=True)
    out = []
    for name, (a, b) in zip(names, ((1.0, 2.0), (3.0, 4.0))):
        p = os.path.join(folder, name)
        with open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write(glc.query_pc_line(t0 + a, t0 + b)
                    + f'#gain_running,sha=16702ff0e688,env=pid,mac={mac},boot={boot},id={gid},applied_n=1,ctrl=PID\n'
                    '#ket_qua,ma=0\n')
        out.append(glc.read_query(p))
    return tuple(out)


def floor_gate(g: dict, t_done=None, t_deadline: float = 20.0) -> dict:
    """FAKE (demo and tests only): a one-domain gate of o1_gate dressed as a camera scorer output (B81-3): the
    common domain set equal to the union, the two t_done given (default from the G1 state: dat both 10 s, khong_dat
    neither, chua_ket_luan only the first); G1 and the outcomes recomputed from them."""
    g = json.loads(json.dumps(g))
    for c in o1_gate.MEASURED:
        g[c]['chung'] = {k: g[c][k] for k in DOMAIN_FIELDS}
    if t_done is None:
        t_done = {'dat': (10.0, 10.0), 'khong_dat': (None, None)}.get(g['g1']['state'], (10.0, None))
    g['g1'].update(t_done=list(t_done), t_deadline=t_deadline)
    st = gate_g1_state(gate_g1_fields(g['g1'], 'HW'))
    g['g1'].update(state=st, q={'dat': 'true', 'khong_dat': 'false'}.get(st, 'inconclusive'))
    g.update(o1_gate.evaluate_outcomes(g, g['decide'], g.get('required')))
    return g


def fake_score_final(folder: str, d_h_mm: Optional[float], h1: Optional[dict] = None, eps_h_mm: float = 2.0,
                     name: str = 'score_final.json', h0_name: str = 'score.json') -> str:
    """FAKE (demo and tests only): the output of c7_calib_drift.py gop (combine) for the run whose H0 scorer file is
    folder/<h0_name>: that file with 'outcome' after the comparison and the 'calib_drift' block. d_h_mm None: clip
    H at the end missing (thieu_H1); above eps_h_mm the H1 score `h1` (outcome, gate_v3) is kept whole. Returns the
    path of the new file (in the run folder, C9D-1)."""
    with open(os.path.join(folder, h0_name), encoding='utf-8') as f:
        h0 = json.load(f)
    o0, thieu = h0['outcome'], d_h_mm is None
    vuot = thieu or d_h_mm > eps_h_mm
    o1 = h1['outcome'] if vuot and not thieu else None
    final, flags = so_h_outcome(o0, o1, vuot, thieu)
    cd = dict(eps_H_mm=eps_h_mm, D_H_mm=d_h_mm, vuot=vuot, thieu_H1=thieu, outcome_h0=o0, outcome_h1=o1, flags=flags)
    if o1 is not None:
        cd['h1_score'] = h1
    p = os.path.join(folder, name)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(dict(h0, outcome=final, calib_drift=cd), f)
    return p


def demo(out_dir: str, seed: int = 1, levels=('O1-F05',)) -> List[dict]:
    """A made-up A-PID, B-chung, B-PID, final session, written through the same commands, then
    the summary. Every number is invented (labelled FAKE); nothing here is a measurement.
    levels: the gate levels of the FAKE C8 gate file (FAKE_gate_c8.json in out_dir, O1 v11.33)."""
    global now
    os.makedirs(out_dir, exist_ok=False)
    rnd = random.Random(seed)
    clock = [dt.datetime(2026, 11, 20, 8, 0, 0)]
    real_now = now
    now = lambda: clock[0]                            # noqa: E731
    try:
        import gain_line_check as glc
        import s05_fake_runs as fr       # R134 (1), C9 dry05: every judged run gets its folder (s05_final_table)
        migrate(out_dir)
        # O1 v11.33: the C8 gate file (FAKE numbers) the scorers and end use; the summary reads it at the end
        gate_path = os.path.join(out_dir, 'FAKE_gate_c8.json')        # stands for the gate locked at C8
        gate_c8.write_gate(gate_path, gate_c8.fake_gate(tuple(levels)))
        gf = read_gate_file(gate_path)
        gain = dict(ctrl='PID', id=1, gains=dict(kp_xy=1.0, ki_xy=0.5, kp_wz=1.0, ki_wz=0.2, pos_kp=6.0, pos_ki=3.0,
                                                 k_psi=1.0))

        def run_dir(name):
            return os.path.abspath(os.path.join(out_dir, 'luot', name))

        def judged(name, platform, outcome, g, ask=True):
            """C9D-1, C9D-6: the run's folder with its scorer file score.json (outcome, gate_v3); kwargs of end."""
            d = run_dir(name)
            qp = fr.write_final_run(d, platform, gain, score_extra=dict(outcome=outcome, gate_v3=g))
            sp = os.path.join(d, 'score.json')
            kw = dict(results_dir=d, gate=read_gate_json(sp), gate_sha256=sha256_file(sp), gate_file=gf)
            if ask:
                kw['queries'] = tuple(glc.read_query(p) for p in qp)
            return kw

        def tick(s):
            clock[0] += dt.timedelta(seconds=s)

        clock[0] -= dt.timedelta(minutes=30)
        hoc_start(out_dir, 'A-PID', 'FAKE: doc huong dan tune')     # stamps by the script (EA-7)
        tick(1800)
        hoc_end(out_dir)
        a1 = []
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
            if i:                                      # B81-1: a judged car run, camera gate; C9D-6: its folder
                a1.append(end(out_dir, 'khong_dat', completed=1, e_perp_max=0.12 + i / 100,
                              **judged(f'a1_{i}', 'HW', 'khong_dat', floor_gate(fake_gate(
                                  e_trk=0.5, gate_file=gf, level='O1-F0' + sp[-1], stage='screen')))))
            else:
                end(out_dir, 'hong_he_do', completed=1, e_perp_max=0.12, fail_cause='mat_khung_FAKE')
        # S05 v3.37 4.3, C9D-2: the clip H comparison at the end of the session, for every judged car run. FAKE-A1:2
        # below eps_H (outcome kept); FAKE-A1:3 above eps_H and the run scored with the clip H at the end is dat:
        # the two outcomes differ, chua_ket_luan with calib_drift_split (O1 4 clip H)
        so_h(out_dir, a1[0]['run_uid'], fake_score_final(run_dir('a1_1'), 1.2))
        so_h(out_dir, a1[1]['run_uid'], fake_score_final(run_dir('a1_2'), 3.1, h1=dict(
            FAKE=True, outcome='dat', gate_v3=floor_gate(fake_gate(gate_file=gf, level='O1-F08', stage='screen')))))
        gfile = fr.write_gain_file(os.path.join(out_dir, 'FAKE_gain_G00.json'), gain)   # gain_send file format
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
        end(out_dir, 'hong_he_do', completed=1, fail_cause=MAP_REFUSED,
            **judged('b1_1', 'HIL', 'hong_he_do', fake_gate(mapped=False, gate_file=gf, platform='HIL',
                                                            stage='hil_check', campaign='B-PID'), ask=False))
        begin(out_dir, 'FAKE-B1', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1')
        step(out_dir, 'run')
        tick(90)
        end(out_dir, 'dat', completed=1, e_perp_max=0.04, **judged('b1_2', 'HIL', 'dat', fake_gate(
            gate_file=gf, platform='HIL', stage='hil_check', campaign='B-PID'), ask=False))
        hwfail(out_dir, '2026-11-20', 'FAKE-B1', 'B-PID', 'HW', 'FAKE: tuot banh', 'phan_cung_ngau_nhien', 0.5)
        for k in range(2):                              # final on HIL: account RQ2, joined by gain set
            begin(out_dir, 'FAKE-FH', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='A-PID/G00')
            step(out_dir, 'run')
            tick(75)
            end(out_dir, 'dat', completed=1, e_perp_max=0.05,
                **judged(f'fh_{k}', 'HIL', 'dat', fake_gate()))            # B81-2: final_hil asks the boot too
        lost, fin = None, []
        # final on the car; the lost run is replaced. The chua_ket_luan run: one way of using the frames did not
        # complete (suspect frames as gaps), so G1 is inconclusive and the two frame domains differ (B81-3)
        car_gate = {'dat': floor_gate(fake_gate(gate_file=gf)),
                    'chua_ket_luan': floor_gate(fake_gate(gate_file=gf), t_done=(10.0, None))}
        for o in ('hong_he_do', 'dat', 'chua_ket_luan'):
            begin(out_dir, 'FAKE-F1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G00',
                  replaces_run_uid=lost if o == 'dat' else '')
            step(out_dir, 'run')
            tick(60)
            if o == 'hong_he_do':
                lost = end(out_dir, o, completed=1, e_perp_max=0.10, fail_cause='mat_khung_FAKE',
                           queries=fake_queries(folder=run_dir('f1_hong_he_do'), gid=gain['id']))['run_uid']
            else:                                       # R134: judged with its gate, boot asked
                fin.append(end(out_dir, o, completed=1, e_perp_max=0.10, **judged(f'f1_{o}', 'HW', o, car_gate[o])))
        for r in fin:                                   # C9D-2: clip H of the final session below eps_H
            so_h(out_dir, r['run_uid'], fake_score_final(r['results_dir'], 0.8))
        return write_summary(out_dir, os.path.join(out_dir, SUMMARY), gate_file=gf)
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
    e.add_argument('--gate-file', default='', help='the locked C8 gate file (gate_c8_v1): report-only conditions of '
                   'the level (O1 v11.33); needed when the scorer used it')
    e.add_argument('--query-before', default='', help='gain_send.py --query output just before the run (R134 (b))')
    e.add_argument('--query-after', default='', help='gain_send.py --query output just after the run (R134 (b))')
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
    o.add_argument('--gate-file', default=None, help='the locked C8 gate file (gate_c8_v1 JSON, tools/gate_c8.py)')
    d = sub.add_parser('demo')
    d.add_argument('--out', required=True)
    sh = sub.add_parser('so-h', help='outcome of a car run after the clip H comparison (S05 v3.37 4.3)')
    sh.add_argument('--run-uid', required=True)
    sh.add_argument('--score-final', required=True, help='output of c7_calib_drift.py gop for that run')
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
            import s05_final_table as s5
            boots = {}
            open_run = _load_open(ds)
            if open_run and query_files_needed(open_run['row'], a.outcome) and not (a.query_before and a.query_after):
                raise LogError(f"a {open_run['row'].get('platform')} run at stage {open_run['row'].get('stage')} "
                               f'judged {a.outcome} needs --query-before and --query-after (gain_send.py --query '
                               'outputs around the run; S05 4.7 (iv), R3V-9, O1 R4)')
            if a.query_before or a.query_after:
                import gain_line_check as glc            # R134C3-3: the checks of the two answers, as the scorers do
                boots = dict(queries=(glc.read_query(a.query_before), glc.read_query(a.query_after)))   # R134C4-1
            if a.gate_json and is_score_final(a.gate_json):                       # B8S2-1 (S05 v3.37 4.3)
                raise LogError(f'--gate-json {a.gate_json!r} is an output of c7_calib_drift.py gop (block calib_drift): '
                               'end logs the provisional outcome with the H0 scoring (score.json of the run), then '
                               'so-h --run-uid <run> --score-final <this file> after the clip H comparison')
            row = end(ds, a.outcome, a.completed, a.e_perp_max, a.fail_cause, a.results_dir, a.notes,
                      gate=read_gate_json(a.gate_json) if a.gate_json else None,
                      gate_sha256=sha256_file(a.gate_json) if a.gate_json else '',
                      gate_file=read_gate_file(a.gate_file) if a.gate_file else None, **boots)
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
            gf = read_gate_file(a.gate_file) if a.gate_file else None
            rows = write_summary(ds, a.out or os.path.join(ds, SUMMARY), gate_file=gf)
            for r in rows:
                print(r['campaign'], r['T_operator_h'], 'h', 'N_HW', r['N_HW'], 'gate', r['gate_result'] or '-')
            n_old = count_no_gate(read_rows(os.path.join(ds, TUNING)))         # R134C2-11 (SC3-2)
            if n_old:
                print(f'{NO_GATE_EVIDENCE}: {n_old} dat row(s) outside the campaigns without gate columns '
                      '(historic; not in gate_result nor in a stop rule)')
        elif a.cmd == 'so-h':
            r = so_h(ds, a.run_uid, a.score_final)
            print('so-h', r['run_uid'], r['outcome_tam'], '->', r['outcome_sau'], r['flags'] or '(khong co)')
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
