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
         [--fail-cause X] [--results-dir D] [--esp32-log CSV] [--gate-json F] [--gate-file C]
                                         close the run and append its row. CSV defaults to D/esp32.csv.
                                         For final car runs a single named CSV/events pair in D is also accepted.
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
                                         a judged car / final_hil run needs both boot answers (API and summary);
                                         a judged floor or HIL tune run needs --gain-file for its ESP32 #gain proof
  cancel / status                        drop the open run / show it. cancel only before the "run"
                                         step (S05 §3.3: every started run counts, EA-6); time of the
                                         steps already logged goes to m7_dev_effort_log.csv
  machine --campaign C --controller PID --script-run-id ID --results-dir D --n-sim-evals N
         --t-machine-s S --wall-start T --wall-end T [--n-sim-diagnostic N]   allocated fwsim script row
  machine-phases --script-kind S03|S3 --script-run-id ID --results-dir D
         --phase ACCOUNT START END N_PID N_LADRC DIAG_PID DIAG_LADRC SOURCE_TIMES SOURCE_COUNTS
                                         repeat for each elapsed phase, including bo. Cite both sources
                                         as FILE:LINE, FILE:FIRST-LAST or FILE@HH:MM:SS (X3-1).
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
                                         gate levels, so_do conditions per level); its sha256 is
                                         reported; every gate row is checked against it (O1 v11.33)
  so-h   --run-uid UID --score-final F [--drift D]   outcome after the clip H comparison at the end
                                         of the session (F = c7_calib_drift.py gop output); appended to
                                         m7_so_clip_h.csv; D = original v2 comparison from c7_calib_drift.py so
  demo   --out DIR                       C9 dry run: made-up campaign in DIR (every judged run with its folder:
                                         s05_fake_runs.py), then summary
  u-moc  --run-uid UID [--lan H0|H1] --file U --gate-file C   C8b v3 §5 (CODE_GATE 18/11, design
                                         03_CODE/docs/2026-11-18_Thiet_ke_rq3_u_moc_v4.md): the U file of one scoring
                                         of a car run (u_moc_luot_v1, CODE_CAM; H1 after so-h above eps_H), checked
                                         and appended to m7_u_moc.csv with the C8 lock read from C (camera.c8b)
  tien-de --session S --file F           the premise file of a session (tien_de_buoi_v1, CODE_KIEMCHEO), appended
                                         to m7_tien_de_buoi.csv. summary, freeze and s05_final_table use the outcome
                                         after so-h, then apply_moc: a dat / khong_dat car run of a campaign without
                                         both files, or failing any check, is chua_ket_luan (flags in moc_flags)

Rules taken from S05 v3.7 (approved 11/11, not locked): 3.1 (time types, wait, hoc), 3.2
(accounts, B-chung split), 3.3 (counts by platform, stages, T_sua in T), 4.1 (caps 10 h, 120 car
runs, counted before frozen_at; stop_cause), 4.6 (final_hil = account RQ2, N = 3), 4.7 (final
counts, gate_result), 4.8 (explore out of T, N, caps), 6.1-6.3 (gain_set_id per campaign).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import c7_drift_evidence as drift_evidence

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
DRIFT_FIELDS = tuple(k for k in drift_evidence.V2_FIELDS if k != 'D_H_vi_tri_mm')
SO_H_COLS += ['drift_file', 'drift_sha256', *DRIFT_FIELDS]
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
              + list(QUERY_FILE_COLS)                   # R134 (1), summary reads the two answers again
              + ['script_kind', 'n_sim_diagnostic', 'trial_gain_file', 'trial_gain_sha256', 'esp32_log']
              + ['c8b_gio_sha256'])                     # O1 v11.37 D1, S05 v3.43 D3: kiem_gio_c8b (CODE_GATE)
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
GATE_VERSIONS = (o1_gate.GATE_VERSION, o1_gate.GATE_VERSION_V1133, gate_c8.GATE_VERSION_SO_DO)
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
           'g1_cond25_before_deadline', 'g1_cat_t1', 'g1_ca_t_som', 'g1_t_som_s']
# J v6 section 5 "Thieu can cu vi tri" (the t_som case: phi missing, beta invalid, q / delta c_B missing, plate
# doi / khong_xac_dinh; CODE 19/11 from CODE_CAM, GATE fixed the names with CAM): floor gate g1.ca_t_som true with
# g1.t_som_s in [0, t_deadline] and t_done [null, null]; G1 is chua_ket_luan (even with cat_t1), G2-G4 must come out
# chua_ket_luan, G5 is scored on the common domain [0, t_som_s] ("chung") and the union [0, t_deadline].
# O1 v11.33 (CODE_GATE 17/11, design 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md section 4; Liam 17/11 at
# CODE_GATE "Duyệt, gửi CODE luôn (Recommended)"): gate_v3.3 rows carry the C8 gate file SHA and the report-only
# conditions of their level (O1 3.9 rule 2b, OG2-7). The required set is recomputed from the gate file
# (tools/gate_c8.py, --gate-file of end / summary), never taken from the row: a row whose report-only list differs
# from the file is an error. Empty in gate_v3.2 rows (all five required).
# 23/11 (CODE, 03_CODE/docs/2026-11-23_Chot_luat_mat_na_quyet_bang_so_do_v1.md; Liam at CODE: "duyệt bảng luật mới"):
# the masked conditions decide by their measured value q (gate_c8.so_do_state), no longer report only: a gate_v3.4
# gate / row carries gate_so_do (the so_do list of the run, recomputed from the gate file) and every condition
# decides; a gate_v3.3 gate / row carries no mask (its report_only, gate_report_only refused: the rule of before 23/11).
GATE_V1133_COLS = ['gate_file_sha256', 'gate_so_do']
# CODE 23/11 (2): on a C8b grid a so_do condition is decided at the middle node (gate block q_so_do {nut, q, q_chung})
SO_DO_NUT_COLS = [f'{c}_so_do_{k}' for c in gate_c8.MASKABLE for k in ('nut', 'q', 'q_chung')]
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
             + LUOI_COLS + SO_DO_NUT_COLS)
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
# O1 v11.37 §3.5 D1, S05 v3.43 §4.6 D2, §6.1 D3 (Liam 17/11 THESIS local_11f2c5a8: "Chỉ lượt trượt kiểm giờ
# (Recommended)", "Tổng (Recommended)", "Bắt buộc file (Recommended)", "Luật §4.6: 3 hỏng liên tiếp (Recommended)";
# the code: CODE_GATE design v2 section 6, Liam 17/11 at CODE_GATE "Duyệt, code luôn (Recommended)"): a floor run
# that fails the per-run check of the C8b time file is lost to the measurement, fail cause C8B_GIO_FAIL, with that
# file (format c8b_gio_v1, ok false, ket_luan hong_he_do) whose sha256 goes to c8b_gio_sha256. Only these runs are counted, in total (not
# in a row), per campaign, gain set and level (map_key), in the order of the log, stages other than final (final:
# S05 4.6, FINAL_LOST_N lost runs in a row, every cause); the fourth one (1 + C8B_RERUNS_MAX) makes the level of that
# gain set chua_ket_luan, a run of it after that is refused.
C8B_GIO_FAIL = 'kiem_gio_c8b'
# F11 (data contract C8 relative v1, GATE accepted 19/11; CODE_C8B design v6 7a): c8b_gio_v1 carries kiem_gio_d1
# {trang_thai dat | truot | khong_kiem, ...}; a kiem_gio_c8b run is a time file with trang_thai truot AND ket_luan
# hong_he_do (a run lost by cap_dem / ho_log alone is an ordinary lost run, other fail cause, not counted).
# c8b_luoi writes trang_thai_luot chua_ket_luan_khong_nut, n_nut_cham 0, gate_v3 null when no node could be scored:
# end logs such a run chua_ket_luan with fail cause KHONG_NUT (never hong_he_do).
D1_STATES = ('dat', 'truot', 'khong_kiem')
KHONG_NUT = 'chua_ket_luan_khong_nut'
C8B_RERUNS_MAX = 3
C8B_GIO_FORMAT = 'c8b_gio_v1'
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
FINAL_SPEEDS = {**FIG8, 'O1-K1': 0.5, 'O1-K2': 0.5}
MANUAL = '[nhap_tay]'                    # flag of a hoc row typed by hand (S05 §3.1 R-2, EA-7)
CLIP_H_NOTE = 'clip_h'                   # S05 v3.44 §3.2 H6: ghi_chu of an effort row of clip H filming


def floor_effort(ghi_chu: str) -> str:
    """Campaign work marked clip_h or 12b, including on manually entered rows."""
    g = ghi_chu or ''
    if g.startswith(MANUAL):
        g = g[len(MANUAL):].lstrip()
    return next((label for label in (CLIP_H_NOTE, '12b') if g.startswith(label)), '')
OUTCOMES = ('dat', 'khong_dat', 'chua_ket_luan', 'hong_he_do')
PLATFORMS = ('SIM', 'HIL', 'HW')
FINAL_PLATFORMS = {'final': 'HW', 'final_hil': 'HIL'}
STEPS = ('flash', 'setup', 'run', 'log', 'analysis')
CAUSES = ('gain', 'cong_cu_HIL', 'phan_cung_ngau_nhien', 'chua_ro')
CAP_H, CAP_HW = 10.0, 120               # S05 §4.1: per campaign, tune part only
TUNE_STAGES = ('screen', 'confirm')     # [CÁCH HIỂU] 'explore' is not a tune run of §4.1 either way
SKIP_PREFIX = ('EXAMPLE', 'PREHIL')      # rows of the June files: examples, reconstruction
# fields begin / end set themselves; begin refuses them as extra fields (Codex S315-R3-1)
OWN_FIELDS = ('session_id', 'iter_no', 'run_uid', 'date', 'wall_start', 'wall_end', 'run_outcome', 'fail_cause',
              'results_dir', 'esp32_log', 'script_kind', 'completed', 'e_perp_max_m', 'c8b_gio_sha256') + tuple(f't_{k}_s' for k in ('flash', 'setup', 'run', 'log',
                                                                                        'analysis', 'operator_wait'))


class LogError(Exception):
    pass


def now() -> dt.datetime:
    """PC clock, local (tests replace this)."""
    return dt.datetime.now()


def _iso(t: dt.datetime) -> str:
    return t.isoformat()


def _parse_t(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s)
    # Stored PC clocks are local. Compare an explicit UTC offset on the same clock.
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


def _freeze_time(s: str) -> dt.datetime:
    if not isinstance(s, str) or not ('T' in s or ' ' in s):
        raise ValueError('frozen_at needs both date and time')
    return _parse_t(s)


def _f(x) -> float:
    if x is None or x == '':
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        raise LogError(f'{x!r}: expected a number')


def _nonnegative(value, field, count=False) -> None:
    if value in (None, ''):
        return
    try:
        n = int(str(value)) if count else float(value)
    except (ValueError, TypeError, OverflowError):
        n = -1
    if isinstance(value, bool) or n < 0 or (not count and not math.isfinite(n)):
        kind = 'integer' if count else 'finite number'
        raise LogError(f'{field}: expected a {kind} >= 0, got {value!r}')


def check_final_platform(row):
    stage = row.get('stage')
    if stage in FINAL_PLATFORMS and row.get('platform') != FINAL_PLATFORMS[stage]:
        raise LogError(f"{row.get('run_uid')}: {stage} requires platform {FINAL_PLATFORMS[stage]}, "
                       f"got {row.get('platform')!r}")


def check_account_row(row, kind, legacy=False) -> None:
    campaign = row.get('campaign' if kind == TUNING else 'chien_dich') or ''
    account = row.get('khoan') or ''
    pending = legacy and kind == EFFORT and campaign == 'chua_biet'
    if (campaign and campaign not in CAMPAIGNS and not pending) or (account and account not in KHOAN):
        raise LogError(f'{kind}: invalid campaign / khoan ({campaign!r}, {account!r})')
    if kind == EFFORT and (campaign and account or not legacy and not (campaign or account)):
        raise LogError('exactly one of --campaign, --khoan (S05 §6.2)')
    ctrl = row.get('controller') or ''
    if ctrl not in ('', 'PID', 'LADRC'):
        raise LogError(f'{kind}: controller {ctrl!r}, expected PID, LADRC or empty')
    if kind == TUNING:
        check_final_platform(row)
        owner = campaign.split('-')[-1]
        if owner in ('PID', 'LADRC') and ctrl != owner:
            raise LogError(f'{kind}: controller {ctrl!r} does not match {campaign}')
        for field in ('iter_no', 'n_sim_evals', 'n_sim_diagnostic', 'completed', 'candidate_no'):
            _nonnegative(row.get(field), field, count=True)
        for field in (*[f't_{k}_s' for k in STEPS], 't_operator_wait_s', 't_machine_s'):
            _nonnegative(row.get(field), field)
    elif kind == EFFORT:
        _nonnegative(row.get('gio'), 'gio')
        nhan, note = row.get('nhan') or '', row.get('ghi_chu') or ''
        if nhan not in ('', 'hoc') or (nhan == 'hoc' and not note):
            raise LogError('nhan must be empty or hoc with a ghi_chu')
        work = floor_effort(note)
        if nhan == 'hoc' and work:
            raise LogError(f'a {work} row cannot be hoc (S05 §3.2)')
        if work == '12b' and not campaign:
            raise LogError('12b effort needs the campaign of the floor session (S05 §3.2)')
    else:
        _nonnegative(row.get('gio_sua'), 'gio_sua')
        _nonnegative(row.get('iter_no'), 'iter_no', count=True)


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
    if os.path.exists(os.path.join(ds, SO_H)):
        out[SO_H] = migrate_file(os.path.join(ds, SO_H), SO_H_COLS)
    for name, cols in ((U_MOC, U_MOC_COLS), (TIEN_DE, TIEN_DE_COLS)):   # C8b v3 §5, design section 4 (RC1-5)
        out[name] = migrate_file(os.path.join(ds, name), cols)
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
    if task.strip().lower() == '12b':
        raise LogError('task 12b is campaign effort: use effort with ghi_chu starting with 12b')
    if task.startswith('O1-F') and task not in FIG8:
        raise LogError(f'task {task!r}: figure-8 names are {sorted(FIG8)} (the name carries the level)')
    level = FINAL_SPEEDS.get(task)
    if level is not None and abs(_f(speed) - level) > 1e-9:
        raise LogError(f'task {task} is level {level} m/s, speed {speed!r} given')


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
           'gain_set_id': gain_set, 'speed_mps': condition_key({'trajectory': task, 'speed_mps': speed})[1],
           'stage': stage, 'wall_start': t.isoformat(timespec='microseconds')}
    row.update(_parse_gains(gains))
    for k, v in extra.items():
        if v not in (None, ''):
            if k not in TUNING_COLS:
                raise LogError(f'unknown field {k}')
            if k in row or k.startswith('g_') or k in OWN_FIELDS:   # Codex S315-R3-1: the identity fields
                raise LogError(f'field {k}: set by begin or end, not by an extra field')   # are checked, not given
            row[k] = str(v)
    check_account_row(row, TUNING)
    tun = apply_so_h(live_rows(read_rows(os.path.join(ds, TUNING))), ds)   # S315-R2-2; v3.37 outcome after so-h
    if _map_counted(row):
        if not gain_set:
            raise LogError('a HIL run needs --gain-set (S05 v3.15 4.5 step 2 counts reruns per candidate)')
        if map_stopped(map_refusals(tun + [row]).get(map_key(row), 0)):
            raise LogError(f'{task}@{speed} of {gain_set}: stopped after {MAP_RERUNS_MAX} reruns for a refused '
                           'mapping (S05 v3.15 4.5 step 2): the level is chua_ket_luan, no further run')
    if _c8b_counted(row) and campaign in GAIN_OWNERS and not gain_set:                     # GT1-7
        raise LogError(f'a floor run of {campaign} outside the final session needs --gain-set (the reruns after the C8b '
                       'time check are counted per gain set, O1 3.5 v11.37; GT1-7)')
    if _c8b_counted(row) and c8b_refusals(tun + [row]).get(map_key(row), 0) > C8B_RERUNS_MAX:
        raise LogError(f'{task}@{speed} of {gain_set}: stopped after {C8B_RERUNS_MAX} reruns for the C8b time check '
                       '(O1 3.5 v11.37): the level is chua_ket_luan, no further run')
    if stage in ('final', 'final_hil') and final_key(row) in _final_stops(tun):
        raise LogError(f'{task}@{speed} of {gain_set or campaign}: stopped after {FINAL_LOST_N} lost runs in a row '
                       '(S05 4.6), no further run')
    if stage in ('final', 'final_hil'):
        check_final_freeze(row, ds=ds)
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


GATE_FILE_COLS = ('gate_file_sha256', 'gate_so_do')   # O1 v11.33: the C8 gate file the run was scored with
NOT_A_GATE = ('gate_json_sha256',) + GATE_FILE_COLS


def has_gate(r: dict) -> bool:
    """A logged row carries a gate: a gate column other than NOT_A_GATE is filled (CC1-2: a lost run may carry the
    sha256 of its scorer file alone, without a gate; GT1-6: and the gate file it was scored with)."""
    return any(r.get(c) for c in GATE_COLS if c not in NOT_A_GATE)


def read_gate_json(path: str, allow_null: bool = False) -> Optional[dict]:
    """The gate block of a scorer output (key 'gate_v3', hil_o1_score.py) or a bare gate dict. allow_null (CC1-2,
    Codex round 1 of CODE_CAM c7_calib_drift; Liam 17/11 at CODE_GATE "Sửa, gộp nhánh này (Recommended)"): a scorer
    output of a run lost to the measurement with 'gate_v3': null gives None (the run is logged with the file's sha256
    only, so so-h finds its H0 file)."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError) as exc:
        raise LogError(f'--gate-json {path!r}: {exc}')
    return gate_of(d, path, allow_null)


def gate_of(d, path: str, allow_null: bool = False) -> Optional[dict]:
    """The gate block of a scorer output already read (read_gate_json; GC9-11: the CLI end parses it from the bytes
    it hashed)."""
    if allow_null and isinstance(d, dict) and 'gate_v3' in d and d['gate_v3'] is None:
        return None
    g = d.get('gate_v3', d) if isinstance(d, dict) else None
    if not isinstance(g, dict) or not str(g.get('version', '')).startswith('gate_v3'):
        raise LogError(f'--gate-json {path!r}: no gate_v3 block (fwsim/o1_gate.py)')
    return g


LUOI_OUTCOME = {'cham': ('dat', 'khong_dat', 'chua_ket_luan'), 'hong_he_do': ('hong_he_do',),
                KHONG_NUT: ('chua_ket_luan',)}


def check_luoi_dict(d, what: str) -> str:
    """GC6-12 (F11): a c8b_luoi file checked on its content; returns trang_thai_luot. Every key present (an absent
    gate_v3 is not a null one): trang_thai_luot, n_nut_cham (whole), gio_sha256 (of the run's time file), gate_v3,
    outcome: cham -> n >= 1, a gate_v3 block, a judged outcome; hong_he_do (every node lost) -> n 0, gate_v3 null,
    outcome hong_he_do; KHONG_NUT -> n 0, gate_v3 null, outcome chua_ket_luan."""
    keys = ('trang_thai_luot', 'n_nut_cham', 'gio_sha256', 'gate_v3', 'outcome')
    if not isinstance(d, dict) or any(k not in d for k in keys):
        raise LogError(f'{what}: a c8b_luoi file has {keys} (F11)')
    st, n = d['trang_thai_luot'], d['n_nut_cham']
    if st not in LUOI_OUTCOME or isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise LogError(f'{what}: trang_thai_luot {st!r} (cham, {KHONG_NUT} or hong_he_do) with n_nut_cham a whole '
                       f'number, got {n!r} (F11)')
    if not _hex64(d['gio_sha256']):
        raise LogError(f'{what}: a c8b_luoi file carries the gio_sha256 of the run time file (F11)')
    if d['outcome'] not in LUOI_OUTCOME[st]:
        raise LogError(f'{what}: {st} with outcome {d["outcome"]!r}, expected one of {LUOI_OUTCOME[st]} (F11)')
    if st == 'cham':
        if n < 1 or not isinstance(d['gate_v3'], dict):
            raise LogError(f'{what}: trang_thai_luot cham needs n_nut_cham >= 1 and its gate_v3 (F11)')
    elif n != 0 or d['gate_v3'] is not None:
        raise LogError(f'{what}: {st} needs n_nut_cham 0 and gate_v3 null (F11)')
    return st


def read_json_sha(path: str, what: str) -> Tuple[object, str]:
    """GC6-12 (round 3): a JSON file read once; its content and the sha256 of the very bytes parsed."""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        return json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as exc:
        raise LogError(f'{what} {path!r}: {exc}')


def read_luoi_sha(path: str) -> Tuple[Optional[dict], str]:
    """(the content of a scorer file when it is a c8b_luoi output, checked (check_luoi_dict), else None; the sha256 of
    the bytes parsed): hashed and parsed from one read (GC6-12)."""
    d, sha = read_json_sha(path, '--gate-json')
    if not isinstance(d, dict) or 'trang_thai_luot' not in d:
        return None, sha
    check_luoi_dict(d, f'--gate-json {path!r}')
    return d, sha


def read_luoi(path: str) -> Optional[dict]:
    """The content of a scorer file when it is a c8b_luoi output (it has trang_thai_luot), checked (check_luoi_dict);
    None for another scorer file."""
    return read_luoi_sha(path)[0]


def khong_nut_sources(lu: dict, lu_sha: str, tg, tg_sha: str, task: str, what: str) -> None:
    """GC6-12 (round 3): the one rule 'end' and the summary apply to a run without a scored node: its c8b_luoi file
    (KHONG_NUT, of the run's task) was made from the time file given (gio_sha256 = the sha256 of the bytes read), a
    c8b_gio_v1 file of that task naming its ESP32 log."""
    if check_luoi_dict(lu, f'{what}: H0 file') != KHONG_NUT or lu.get('task', task) != task or not _hex64(lu_sha):
        raise LogError(f'{what}: the H0 file is not a c8b_luoi file of task {task!r} without a scored node (F11)')
    if tg_sha != lu['gio_sha256']:
        raise LogError(f'{what}: the time file is not the one its c8b_luoi file was made from (gio_sha256; GC6-12)')
    if not isinstance(tg, dict) or tg.get('format') != 'c8b_gio_v1' or tg.get('task') != task \
            or not isinstance(tg.get('sha256'), dict) or not _hex64(tg['sha256'].get('esp32')):
        raise LogError(f'{what}: the time file is not a c8b_gio_v1 file of task {task!r} naming its ESP32 log (GC6-12)')


def read_luoi_khong_nut(path: str) -> bool:
    """F11 (KG-2): whether a scorer file is a c8b_luoi output of a run where no node could be scored
    (check_luoi_dict: trang_thai_luot chua_ket_luan_khong_nut). Other files: False."""
    d = read_luoi(path)
    return d is not None and d['trang_thai_luot'] == KHONG_NUT


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


G1_KEYS = ('t_done', 't_deadline', 't1', 'prog_before_t1', 'cond25_before_deadline', 'cat_t1', 'ca_t_som', 't_som_s')
T_SOM_ZERO = ('g2', 'g3', 'g4')          # J v6 section 5: G1-G4 chua_ket_luan in the t_som case (G1 by its fields)


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
    if 'ca_t_som' in g1 and not isinstance(g1['ca_t_som'], bool):                  # GC6-15 (F13), any block
        raise LogError(f'gate g1 ca_t_som {g1["ca_t_som"]!r}: true or false when present (absent = false; F13)')
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
    if 'ca_t_som' in g1 and not isinstance(g1['ca_t_som'], bool):                  # GC6-15 (F13)
        raise LogError(f'gate g1 ca_t_som {g1["ca_t_som"]!r}: true or false when present (absent = false; F13)')
    ts_case = _opt_bool(g1.get('ca_t_som'), 'g1 ca_t_som')
    t_som = _time(g1.get('t_som_s'), 'g1 t_som_s')
    if ts_case:
        if ways != 2:
            raise LogError('gate g1 ca_t_som: a floor field (two ways of using the frames; J v6 section 5)')
        if t_som is None or t_som > t_dl:
            raise LogError(f'gate g1 t_som_s {g1.get("t_som_s")!r}: with ca_t_som, a time in [0, t_deadline '
                           f'{t_dl!r}] (J v6 section 5)')
        if any(t is not None for t in t_done):
            raise LogError('gate g1: with ca_t_som (no position basis) t_done is [null, null] (J v6 section 5)')
    elif t_som is not None:
        raise LogError('gate g1 t_som_s: only with ca_t_som true')
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
                cat_t1=bool(cat) if ways == 2 else None, ca_t_som=bool(ts_case), t_som_s=t_som)


def gate_g1_state(f: dict) -> str:
    """G1 recomputed from its fields (O1 v11.15 3.3 "Khung nghi ngo vi tri", 3.5 last block, R2V-4)."""
    done = [t is not None for t in f['t_done']]
    if f['ways'] == 2:                       # floor: both ways done -> dat, neither -> khong_dat, else inconclusive
        if f.get('ca_t_som') or f['cat_t1']:  # J v6 section 5: no position basis; R2V-4: progress after t1 not known
            return 'chua_ket_luan'           # R2V-4: progress after t1 not known
        return 'dat' if all(done) else ('khong_dat' if not any(done) else 'chua_ket_luan')
    if f['t1'] is None or f['prog_before_t1']:  # no tracking error, or progress reached before t1
        return 'dat' if done[0] else 'khong_dat'
    return 'chua_ket_luan' if f['cond25_before_deadline'] else 'khong_dat'


def domains_one(f: Optional[dict]) -> bool:
    """The common and the union domain are the same frames: one way, no G1 fields, or two equal t_done."""
    if f is not None and f.get('ca_t_som'):     # J v6 section 5: G5 common [0, t_som], union [0, t_deadline]
        return f['t_som_s'] == f['t_deadline']  # GC6-8: the same domain when t_som = t_deadline (both blocks equal)
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


def _recompute(gate: dict, platform: Optional[str] = None, so_do=None) -> dict:
    """States and outcomes recomputed from q, lo, hi, n_undef with fwsim/o1_gate (thresholds, O1 3.5
    three states SC3-1, O1 3.5 order); shared by the JSON path (end) and the logged rows (summary).
    B81-3: G2-G5 on the common and the union frame domain (cond_state2), G1 from its fields (gate_g1_state)
    when the gate has them (always on the car). Returns the outcomes."""
    if gate.get('version') not in GATE_VERSIONS:
        raise LogError(f'gate version {gate.get("version")!r}: only {GATE_VERSIONS} are supported')
    sd = gate_so_do(gate, so_do)
    decide = _check_decide(gate.get('decide'), platform)
    u_t = _qnum(gate.get('u_t'), 'u_t')
    if u_t < 0.0:
        raise LogError(f'gate u_t: {u_t!r} is negative (a bound is >= 0)')
    # U_t known <=> G3, G5 may decide (RS-4): a HIL gate without the mapping cannot claim them (every condition
    # decides since 23/11, so_do ones by q)
    if ('g4' in decide) == math.isnan(u_t):
        raise LogError(f'gate decide {decide!r} with u_t {gate.get("u_t")!r}: G3, G4, G5 may decide exactly when '
                       'U_t is known (RS-4)')
    q1 = gate['g1'].get('q')
    if q1 not in _G1_Q or gate['g1'].get('state') != o1_gate.g1_state(q1):
        raise LogError(f'gate g1: q {q1!r} and state {gate["g1"].get("state")!r} do not agree (O1 3.4 completed)')
    f1 = gate_g1_fields(gate['g1'], platform)                                   # B81-3
    lb = None
    if gate.get('luoi') is not None:                                            # C8b time grid (design v2)
        if (gate.get('version') not in (o1_gate.GATE_VERSION_V1133, gate_c8.GATE_VERSION_SO_DO)
                or platform not in (None, 'HW') or f1 is None
                or f1['ways'] != 2):
            raise LogError('gate luoi: only in a gate_v3.3 / v3.4 floor gate with its G1 fields (2 ways)')
        lb = check_luoi(gate['luoi'])
        if gate['g1']['state'] != lb['g1_state']:
            raise LogError(f'gate g1: state {gate["g1"]["state"]!r}, the grid gives {lb["g1_state"]!r} (luoi)')
        if f1['ca_t_som'] and any(x != 'chua_ket_luan' for x in lb['g1_nut']):
            raise LogError('gate luoi: g1.ca_t_som but a node with G1 other than chua_ket_luan (J v6 section 5: by node)')
        # GT1-1 (Codex round 1 of CODE_GATE; Liam 17/11 at CODE_GATE "Sửa theo bảng (Recommended)"): the node nut_tam
        # follows the rule of one run on its own G1 fields: lost exactly when t1 is given without the exception before
        # it (R134C4-3), the exception exactly when cat_t1 (R2V-4); so a grid of one node keeps the rule of a run
        # without a grid
        k = lb['nut_tam']
        lost = f1['t1'] is not None and not f1['cat_t1']
        if lb['hong_nut'][k] != lost or lb['cat_nut'][k] != f1['cat_t1']:
            raise LogError(f'gate luoi: node nut_tam {k} lost {lb["hong_nut"][k]}, exception before t1 '
                           f'{lb["cat_nut"][k]}; its G1 fields (t1 {f1["t1"]!r}, cat_t1 {f1["cat_t1"]!r}) give lost '
                           f'{lost}, exception {f1["cat_t1"]} (a node is lost exactly when t1 is given without the '
                           'exception before it, R134C4-3, O1 3.5 R2V-4; GT1-1)')
        if any(lb['hong_nut']):                                                 # GT1-4: O1 L1
            short = [c for c in o1_gate.MEASURED if _n_undef(gate[c].get('n_undef'), f'{c} n_undef') < 1]
            if short:
                raise LogError(f'gate luoi: a node lost to the measurement, but n_undef 0 in {short} (a lost node gives '
                               'every measured condition at least one undefined frame of the union, O1 L1; GT1-4)')
    want1 = gate['g1']['state'] if lb is None else lb['g1_nut'][lb['nut_tam']]
    centre_lost = lb is not None and lb['hong_nut'][lb['nut_tam']]       # GT1-1: its G1 is chua_ket_luan (check_luoi)
    if f1 is not None and not centre_lost and gate_g1_state(f1) != want1:
        raise LogError(f'gate g1: state {want1!r}{" (grid node nut_tam)" if lb else ""}, recomputed from t_done, t1 '
                       f'and the flags: {gate_g1_state(f1)!r} (O1 3.3, 3.5)')
    # grid numbers are not those of one node (design v2 section 2); GC6-8: a grid of one node in the t_som case is
    # that node, its t_som_s is the one of the gate
    one = domains_one(f1) and (lb is None or (f1 is not None and f1['ca_t_som'] and lb['n_nut'] == 1))
    if f1 is not None and f1['ca_t_som'] and (f1['cat_t1'] or (lb is not None and (
            any(lb['cat_nut']) or gate['g5'].get('state') == 'khong_dat'))):
        # GC6-8, F18 (5): the gate holds t_som_s of nut_tam only; that a violating frame of EVERY node lies in that
        # node's common domain [0, t_som] and before its t1 needs the node evidence of the scorer (CAM), not yet given
        raise LogError('gate: the t_som case with an exception before t1 (cat_t1), or on a grid (any number of '
                       'nodes) with G5 khong_dat: the evidence of a violating frame in [0, min(t1, t_som)] of every node '
                       '(node scores of CAM: cat_t1, g1.t1, VIOL_BEFORE; F18 (5)) is not in the gate; the run cannot be '
                       'judged on it')
    for c, thr in o1_gate.THRESHOLDS.items():
        union = _domain_nums(gate[c], c)
        com = _common_block(gate, c, union, one, platform)
        st = cond_state2(com[1], union[1], union[2], union[3], thr)
        if c in sd:
            # so_do (CODE 23/11, Liam): the three-branch state is kept (state_ba_nhanh, checked when the gate holds it:
            # a logged row has no column for it), the state is decided by q of the common domain and of the union
            if 'state_ba_nhanh' in gate[c] and gate[c]['state_ba_nhanh'] != st:
                raise LogError(f'gate {c}: state_ba_nhanh {gate[c]["state_ba_nhanh"]!r}, recomputed from lo of the '
                               f'common domain, hi and n_undef of the union: {st!r}')
            # SD1-1, SD1-3 (Codex 24/11; Liam "Sửa theo bảng"): q of the DEFINED frames (q_so_do), required; on a C8b
            # grid that of the middle node (scored, not lost), with n_undef of the grid. That the node is the one
            # nearest the middle needs the node offsets: checked by gate_c8 at C8 and by c8b_luoi (limit here)
            try:
                qs = gate_c8.check_q_so_do(gate[c].get('q_so_do'), gate[c], f'gate {c}', lb is not None)
            except gate_c8.GateFileError as exc:
                raise LogError(str(exc))
            if lb is not None and (qs['nut'] >= lb['n_nut'] or qs['nut'] < 0 or lb['hong_nut'][qs['nut']]):
                raise LogError(f'gate {c} q_so_do.nut {qs["nut"]!r}: a scored node of the grid (not lost)')
            st = gate_c8.so_do_state(qs['q_chung'], qs['q'], union[3], thr)
            if st != gate[c].get('state'):
                raise LogError(f'gate {c}: state {gate[c].get("state")!r} (so_do), recomputed from q of the common '
                               f'domain, q and n_undef of the union: {st!r} (CODE 23/11)')
        elif st != gate[c].get('state'):
            raise LogError(f'gate {c}: state {gate[c].get("state")!r}, recomputed from lo of the common domain, '
                           f'hi and n_undef of the union: {st!r}')
        if f1 is not None and f1['ca_t_som'] and c in T_SOM_ZERO and st != 'chua_ket_luan':
            raise LogError(f'gate {c}: state {st!r} in the t_som case (g1.ca_t_som): G1-G4 have no position basis '
                           'and are chua_ket_luan (undefined frames, no finite lo above the threshold; J v6 section 5)')
    return o1_gate.evaluate_outcomes(gate, decide, GATE_CONDS)


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


def gate_so_do(gate: dict, so_do=None) -> tuple:
    """The so_do conditions of a gate (CODE 23/11, Liam "duyệt bảng luật mới"), checked against those the C8 gate
    file gives for the run (`so_do`, gate_c8.GateC8.run_so_do; None = no gate file: none). A gate_v3.2 gate and a
    gate_v3.3 gate have none (a gate_v3.3 report_only list, the rule of before 23/11, is refused). A gate_v3.4 gate
    states its so_do, level and gate_file_sha256 (OG2-7: never taken from the run record)."""
    want = () if so_do is None else tuple(c for c in gate_c8.MASKABLE if c in set(so_do))
    if gate.get('version') in (o1_gate.GATE_VERSION, o1_gate.GATE_VERSION_V1133):
        for k in ('report_only', 'required', 'so_do'):
            if gate.get(k) not in (None, []) and not (k == 'required' and gate.get(k) == list(GATE_CONDS)):
                raise LogError(f'gate {k} {gate.get(k)!r}: not a {gate.get("version")} field since 23/11 (the masked '
                               f'conditions decide by q: {gate_c8.GATE_VERSION_SO_DO}, CODE, Liam)')
        if gate.get('version') == o1_gate.GATE_VERSION and gate.get('gate_file_sha256') not in (None, ''):
            raise LogError(f'gate gate_file_sha256: not a {o1_gate.GATE_VERSION} field')
        got = ()
    else:
        sd, sha = gate.get('so_do'), gate.get('gate_file_sha256')
        if 'report_only' in gate or 'required' in gate:
            raise LogError(f'gate report_only / required: not {gate_c8.GATE_VERSION_SO_DO} fields')
        if not isinstance(sd, list):
            raise LogError(f'gate so_do {sd!r}: a list ({gate_c8.GATE_VERSION_SO_DO})')
        try:
            got = gate_c8.check_so_do(sd)
        except gate_c8.GateFileError as exc:
            raise LogError(f'gate {exc}')
        if list(got) != sd:
            raise LogError(f'gate so_do {sd!r}: in the order {gate_c8.MASKABLE}')
        if not isinstance(sha, str) or len(sha) != 64 or any(ch not in '0123456789abcdef' for ch in sha):
            raise LogError(f'gate gate_file_sha256 {sha!r}: the source of so_do (OG2-7), 64 lowercase hex digits')
    if got != want:
        raise LogError(f'gate so_do conditions {list(got)!r}, the C8 gate file gives {list(want)!r} for this run '
                       '(O1 3.9 rule 2b decided by the measured value, CODE 23/11; OG2-7)')
    for c in GATE_CONDS:
        b = gate.get(c)
        has = isinstance(b, dict) and ('nhanh' in b or 'state_ba_nhanh' in b)
        if c in got and isinstance(b, dict) and b.get('nhanh', gate_c8.NHANH_SO_DO) != gate_c8.NHANH_SO_DO:
            raise LogError(f'gate {c} nhanh {b.get("nhanh")!r}: {gate_c8.NHANH_SO_DO!r} for a so_do condition')
        if c not in got and has:
            raise LogError(f'gate {c}: nhanh / state_ba_nhanh on a condition that is not so_do')
    return got


def validate_gate(gate: dict, platform: Optional[str] = None, so_do=None) -> dict:
    """GV3-5, GV3-R2-3: a gate JSON checked key by key (a missing key is an error, null is 'unknown';
    booleans by type; thresholds), then every state and outcome recomputed (_recompute); any difference
    is an error. The outcome fields of the JSON are never trusted. platform: the run's platform, whose
    deciding sets the JSON must match (GV3-R2-2); so_do: the so_do conditions of the run from the gate file
    (row_so_do; None = no gate file), each with nhanh 'so_do' and state_ba_nhanh in the JSON. Returns the recomputed
    outcomes."""
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
    if gate.get('version') == gate_c8.GATE_VERSION_SO_DO:
        for c in gate.get('so_do') or []:
            if not isinstance(gate.get(c), dict) or gate[c].get('nhanh') != gate_c8.NHANH_SO_DO \
                    or gate[c].get('state_ba_nhanh') not in o1_gate_states():
                raise LogError(f'gate {c}: a so_do condition without nhanh {gate_c8.NHANH_SO_DO!r} and its '
                               'state_ba_nhanh (CODE 23/11)')
    ev = _recompute(gate, platform, so_do)
    for k, v in ev.items():
        if type(gate[k]) is not type(v) or gate[k] != v:
            raise LogError(f'gate {k}: {gate[k]!r}, recomputed: {v!r}')
    return ev


def o1_gate_states() -> tuple:
    return ('dat', 'khong_dat', 'chua_ket_luan')


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

    if row.get('gate_report_only'):                    # SD1-7 (Codex 24/11): the column of before 23/11
        raise LogError(f"{row.get('run_uid')}: gate_report_only {row.get('gate_report_only')!r}: report only is no "
                       'rule since 23/11 (CODE, Liam "duyệt bảng luật mới"); the run is scored again with so_do')
    g = {'version': row.get('gate_version'), 'decide': [c for c in row.get('gate_decide', '').split(',') if c],
         'u_t': num(row.get('gate_u_t')), 'g1': {'q': row.get('g1_q'), 'state': row.get('g1_state')}}
    if row.get('gate_version') in (o1_gate.GATE_VERSION_V1133, gate_c8.GATE_VERSION_SO_DO):   # O1 v11.33
        sd = [c for c in row.get('gate_so_do', '').split(',') if c]
        if row.get('gate_version') == o1_gate.GATE_VERSION_V1133:
            if sd:
                raise LogError(f"{row.get('run_uid')}: gate_so_do {sd} on a {o1_gate.GATE_VERSION_V1133} row (so_do "
                               f"is {gate_c8.GATE_VERSION_SO_DO})")
            g.update(gate_file_sha256=row.get('gate_file_sha256') or None, level=row.get('trajectory') or None)
        else:
            g.update(so_do=sd, gate_file_sha256=row.get('gate_file_sha256') or None,
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
    elif row.get('gate_file_sha256') or row.get('gate_so_do'):
        raise LogError(f"{row.get('run_uid')}: gate_file_sha256 / gate_so_do on a "
                       f"{row.get('gate_version')!r} row")
    ways = row.get('g1_ways', '')                                               # B81-3
    if ways:
        if ways not in ('1', '2') or (ways == '1' and row.get('g1_t_done_2', '') != ''):
            raise LogError(f"{row.get('run_uid')}: g1_ways {ways!r} with g1_t_done_2 {row.get('g1_t_done_2')!r}")
        g['g1'].update(t_done=[num(row.get(f'g1_t_done_{i}')) for i in range(1, int(ways) + 1)],
                       t_deadline=num(row.get('g1_t_deadline')), t1=num(row.get('g1_t1')),
                       prog_before_t1=flag('g1_prog_before_t1'), cond25_before_deadline=flag('g1_cond25_before_deadline'),
                       cat_t1=flag('g1_cat_t1'), t_som_s=num(row.get('g1_t_som_s')))
        if row.get('g1_ca_t_som', '') != '':          # GC6-15: an empty cell is the absent field (false), not null
            g['g1']['ca_t_som'] = flag('g1_ca_t_som')
    elif any(row.get(k) for k in G1_COLS):
        raise LogError(f"{row.get('run_uid')}: G1 fields without g1_ways")
    for c in o1_gate.MEASURED:
        g[c] = {k: num(row.get(f'{c}_{k}')) for k in ('q', 'lo', 'hi')}
        g[c].update(n_undef=num(row.get(f'{c}_n_undef'), whole=True), state=row.get(f'{c}_state'))
        if any(row.get(f'{c}_{k}_chung') for k in DOMAIN_FIELDS):
            g[c]['chung'] = {k: num(row.get(f'{c}_{k}_chung'), whole=k == 'n_undef') for k in DOMAIN_FIELDS}
    for c in gate_c8.MASKABLE:                         # CODE 23/11 (2); SD1-1, SD1-3: every so_do condition
        cols = [row.get(f'{c}_so_do_{k}', '') for k in gate_c8.SO_DO_KEYS]
        if c in (g.get('so_do') or []):
            g[c]['q_so_do'] = dict(nut=num(cols[0], whole=True), q=num(cols[1]), q_chung=num(cols[2]))
        elif any(cols):
            raise LogError(f"{row.get('run_uid')}: {c}_so_do_* on a condition that is not so_do")
    return g


def map_refused(decide) -> bool:
    """The gate of a HIL run whose run_active edge was not mapped (only G1, G2 decide, O1 v11.12 §3.4): G4 does not
    decide."""
    d = set(decide)
    return bool(d) and d <= set(o1_gate.DECIDE_HIL)


def outcome_vs_gate(outcome: str, ev: dict, decide, fail_cause: str = '', gate: Optional[dict] = None) -> Optional[str]:
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
    if (gate is not None and gate.get('luoi') is not None and fail_cause == VIOL_BEFORE and outcome == 'khong_dat'
            and viol_before_exception(gate)):
        # SD1-4 (Codex 24/11): every node of the grid has the exception before t1, maybe on different conditions, so
        # the per-condition grid states may stay chua_ket_luan; the run is khong_dat (c8b_luoi combine;
        # viol_before_problem checks the pair)
        return None
    if ev['complete']:
        return None if outcome == od else f'gate outcome {od}'
    # not reached: decide is DECIDE_ALL (complete) or DECIDE_HIL (o1_gate.DECIDE_SETS), kept as a guard
    return f'the gate is not complete (decides {",".join(decide)}): no run is dat'


def count_no_gate(tun: List[dict]) -> int:
    """SC3-2: dat rows outside the campaigns (historic rows) with empty gate columns, reported apart."""
    return sum(r.get('run_outcome') == 'dat' and r.get('campaign') not in CAMPAIGNS
               and not has_gate(r) for r in tun)


def query_files_needed(row: dict, outcome: str) -> bool:
    """R134 (b): the CLI 'end' of this open run needs --query-before and --query-after (comment above BOOT_FAIL)."""
    return outcome in VALID and boot_needed(row.get('platform'), row.get('stage'))


def check_query_rows(tun: List[dict], ds: str, query_before_name=None, query_after_name=None) -> dict:
    """R134 (1) (R134C4-1 summary part, deferred at round 4): the summary reads the two gain_send --query outputs of a
    row again. A judged car / final_hil campaign row (query_files_needed) must name both files with their sha256;
    every named file must be there with that sha256 (a file changed after 'end' is refused); the two answers must be
    two questions one after the other (gain_line_check.query_problem) with the boots of the row's boot columns. A
    relative name is looked for in the dataset folder, then in the working folder. Rows outside the campaigns that
    name no file (historic rows) are not checked."""
    import gain_line_check as glc
    checked = {}
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
            try:
                with open(f, 'rb') as stream:
                    raw = stream.read()
            except OSError as exc:
                raise LogError(f'{uid}: cannot read {cf} {f!r}: {exc}') from exc
            if hashlib.sha256(raw).hexdigest() != sha:
                raise LogError(f'{uid}: {f} changed after the run was logged (sha256 differs from {cs}; R134 (1))')
            name = query_before_name if when == 'before' else query_after_name
            if name is not None and r.get('stage') in ('final', 'final_hil'):
                folder = _results_dir(r, ds)
                expected = os.path.join(folder, name) if folder else ''
                if not expected or os.path.normcase(os.path.realpath(expected)) != os.path.normcase(os.path.realpath(f)):
                    raise LogError(f'{uid}: --query-{when}-name {name!r} differs from the logged query file {p!r}')
            qs.append(glc.parse_query(raw, f))
        if not need:
            continue
        why = glc.query_problem(qs[0], qs[1], check_run_boot=False)
        if why:
            raise LogError(f'{uid}: the two saved answers: {why} (R134 (1), R134C4-1)')
        cols = ((r.get('boot_before') or '').lower(), (r.get('boot_after') or '').lower())
        if cols != (qs[0]['boot'].lower(), qs[1]['boot'].lower()):
            raise LogError(f"{uid}: boot cua dong log / boot columns {cols} differ from the saved answers ({qs[0]['boot']}, "
                           f"{qs[1]['boot']}; R134 (1))")
        checked[uid] = tuple(qs)
    return checked


# ── S05 v3.37 §4.3 (B8S-3): the outcome after the clip H comparison ────────────────────────────────────────────────
def so_h_outcome(o0: str, o1: Optional[str], vuot: bool, thieu: bool) -> tuple:
    """(outcome after the comparison, flags) by O1 §4 clip H (v11.27, v11.28; c7_calib_drift.combine): below eps_H
    the H0 outcome, no flag; clip H at the end missing: chua_ket_luan, flag thieu_H1, a lost run stays lost; above
    eps_H: lost in either -> hong_he_do, the same -> kept, different -> chua_ket_luan; flag calib_drift, plus
    calib_drift_split when the two differ (a lost run in one of the two is no split)."""
    try:
        return drift_evidence.outcome(o0, o1, vuot, thieu)
    except ValueError as exc:
        raise LogError(str(exc)) from exc


def _results_dir(r: dict, ds: str, *, existing=True) -> Optional[str]:
    """The results_dir of a logged row as a folder that exists (a relative name: in the dataset folder, then in the
    working folder, as check_query_rows), or None. For path containment alone, existing=False also resolves
    a recorded folder of a lost run whose capture is absent."""
    p = r.get('results_dir', '')
    if not p:
        return None
    cand = [p] if os.path.isabs(p) else [os.path.join(ds, p), p]
    return next((c for c in cand if os.path.isdir(c)), None if existing else cand[0])


def esp32_path(row, results_dir, default='esp32.csv'):
    # An explicit run path is relative to its results_dir, including bench names under pi/.
    name = row.get('esp32_log') or default
    path = os.path.join(results_dir or '', name)
    if results_dir or row.get('esp32_log'):
        if not results_dir or not all(_inside(p, results_dir) for p in (path, path + '.events.txt')):
            raise LogError(f'{name!r}: ESP32 CSV and events must resolve under the results_dir of the run')
    return path


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


def check_drift_thresholds(cd, uid, original=False):
    try:
        return drift_evidence.check_file(cd, tuple(FINAL_SPEEDS)) if original else drift_evidence.check(cd)
    except ValueError as exc:
        raise LogError(f'run {uid}: {exc} (C9D-5)') from exc


def check_score_final(r: dict, path: str, ds: str, drift=None, new=False) -> tuple:
    """The output of c7_calib_drift.py gop for logged car run r, checked (command so-h and apply_so_h); returns
    (outcome after the comparison, flags, the calib_drift block). Checks:
      * the run is a car run of a campaign; outcome_h0 = its logged (provisional) outcome;
      * C9D-1: the file is under the run's results_dir and is the H0 scorer file of the run (h0_file) with only
        'outcome' and 'calib_drift' changed: a file of another run with the same outcome and gate is refused;
      * J v6 §5: vuot is computed from position, heading and missing quantities; above a threshold or when a
        quantity is missing with H1 present, recompute the outcome of the kept H1 score (h1_score) and its gate;
      * the outcome and flags of the file = so_h_outcome; the gate of the file (the H0 scoring) = the logged gate."""
    uid = r.get('run_uid')
    if r.get('platform') != 'HW' or r.get('campaign') not in CAMPAIGNS:
        raise LogError(f'run {uid}: the clip H comparison is for car runs of a campaign (O1 §4)')
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        cd = d['calib_drift']
        o0, o1, flags = cd['outcome_h0'], cd.get('outcome_h1'), cd['flags']
        vuot, thieu = cd['vuot'], cd['thieu_H1']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LogError(f'score_final {path!r}: not an output of c7_calib_drift.py gop ({exc})')
    if drift is not None:
        version = check_drift_thresholds(drift, uid, original=True)
        historical = 'version' not in cd and not any(k in cd for k in drift_evidence.V2_FIELDS)
        if historical:
            cd['version'] = version
            cd.update({k: drift[k] for k in drift_evidence.V2_FIELDS if k in drift})
        for k in (*drift_evidence.SESSIONS, *drift_evidence.BASE_FIELDS, *drift_evidence.V2_FIELDS):
            if k in drift and (k not in cd or cd[k] != drift[k]):
                raise LogError(f'run {uid}: score_final and --drift disagree in {k}')
    elif new and 'version' not in cd and not any(k in cd for k in drift_evidence.V2_FIELDS):
        raise LogError(f'run {uid}: unversioned position-only gop: give --drift with the original v1 or v2 comparison')
    combined_version = check_drift_thresholds(cd, uid)
    if drift is not None and combined_version != version:
        raise LogError(f'run {uid}: score_final and --drift disagree in version')
    if not isinstance(flags, list) or any(flag not in SO_H_FLAGS for flag in flags):
        raise LogError(f'score_final {path!r}: expected a list of flags in {SO_H_FLAGS}')
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
    if drift is not None:
        try:
            drift_evidence.check_score_scope(h0, drift, 'H0')
        except ValueError as exc:
            raise LogError(f'run {uid}: {exc}') from exc
    if vuot and not thieu:
        h1 = cd.get('h1_score')
        if not isinstance(h1, dict) or h1.get('outcome') != o1:
            raise LogError(f'run {uid}: vuot requires an H1 score, and outcome_h1 {o1!r} must match that score '
                           f'(h1_score; C9D-5)')
        g1 = h1.get('gate_v3')                         # C9D2-1: a lost H1 may carry no gate, or null
        h1_kn = 'trang_thai_luot' in h1 and check_luoi_dict(h1, f'run {uid}: h1_score') == KHONG_NUT    # GC6-13
        if h1_kn and o1 != 'chua_ket_luan':
            raise LogError(f'run {uid}: an H1 c8b_luoi file without a scored node is chua_ket_luan, not {o1!r}')
        if h1_kn:                                                                          # GC7-3
            # round 3: the H1 scoring is of the run's task and has its own time file (not that of H0)
            if h1.get('task') != r.get('trajectory') or h1['gio_sha256'] == h0.get('gio_sha256'):
                raise LogError(f'run {uid}: the H1 c8b_luoi file is of task {h1.get("task")!r} (the run is '
                               f'{r.get("trajectory")!r}) or names the time file of the H0 scoring: H1 has its own '
                               '(GC7-3)')
            g1f = _time_file_of(rd, h1['gio_sha256'], r.get('trajectory'), f'run {uid}: H1 time file')
            if 'trang_thai_luot' in h0:
                esp0 = _time_file_of(rd, h0.get('gio_sha256'), r.get('trajectory'),
                                     f'run {uid}: H0 time file')['sha256']['esp32']
            else:
                esp0 = h0.get('esp32_sha256')
            if not _hex64(esp0) or g1f['sha256']['esp32'] != esp0:
                raise LogError(f'run {uid}: the H1 time file reads another ESP32 log than the run (GC7-3)')
        if o1 in VALID and g1 is None and not h1_kn:
            raise LogError(f'run {uid}: the H1 score judged {o1} carries no gate_v3 block (C9D-5)')
        if g1 is not None:
            # C9D2-2: the same check as 'end' (check_outcome_vs_gate: gate recomputed, then outcome_vs_gate and the
            # floor rule of t1, viol_before_problem); the score carries no fail cause, so the exception before t1
            # (khong_dat, VIOL_BEFORE) is read from its gate by the rule of 'end' (GT1-5: every grid node, else cat_t1)
            fc = VIOL_BEFORE if viol_before_exception(g1) else ''
            try:
                # O1 v11.33: the H1 score uses the gate file of the logged row (checked against the file at end and in
                # the summary): same SHA, the so_do set of the row; GT1-6: a lost H0 without a gate carries them too
                if isinstance(g1, dict) and (g1.get('gate_file_sha256') or '') != r.get('gate_file_sha256', ''):
                    raise LogError('gate file SHA of the H1 score differs from the logged run')
                sd = ([c for c in r.get('gate_so_do', '').split(',') if c]
                      if r.get('gate_version') == gate_c8.GATE_VERSION_SO_DO or r.get('gate_file_sha256') else None)
                check_outcome_vs_gate(o1, g1, 'HW', fc, sd)
            except LogError as exc:
                raise LogError(f'run {uid}: outcome_h1 {o1} and the gate of the H1 score: {exc} (C9D-5, C9D2-2)')
    elif cd.get('h1_score') is not None:
        raise LogError(f'run {uid}: unexpected h1_score without a second scoring')
    final, want = so_h_outcome(o0, o1, vuot, thieu)
    if d.get('outcome') != final or sorted(flags) != sorted(want):
        raise LogError(f"score_final {path!r}: outcome {d.get('outcome')!r}, flags {flags} disagree with "
                       f'the rule of O1 §4 ({final!r}, {want})')
    if r.get('fail_cause') == KHONG_NUT:                                         # GC6-13: F11 run, no gate
        if has_gate(r) or check_luoi_dict(h0, f'run {uid}: H0 file {h0p}') != KHONG_NUT:
            raise LogError(f'run {uid}: a {KHONG_NUT} row has its c8b_luoi file without a scored node as H0 file')
    elif o0 != 'hong_he_do' or has_gate(r):
        g = gate_columns(read_gate_json(path))
        diff = sorted(c for c, v in g.items() if c != 'gate_json_sha256' and str(r.get(c, '')) != v)
        if diff:
            raise LogError(f'run {uid}: the gate of the file (H0 scoring) differs from the logged gate in {diff}')
    return final, flags, cd


def drift_columns(cd):
    return {k: _num(cd.get(k)) if k.startswith(('D_', 'eps_')) else
            ('true' if cd[k] else 'false') if k == 'vuot_do_thieu' and k in cd else cd.get(k) or ''
            for k in DRIFT_FIELDS}


def so_h(ds: str, run_uid: str, score_final: str, drift_file='') -> dict:
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
    drift, drift_sha = read_json_sha(drift_file, '--drift') if drift_file else (None, '')
    if drift_file and not isinstance(drift, dict):
        raise LogError('--drift: expected a JSON object')
    final, flags, cd = check_score_final(r, score_final, ds, drift, new=True)
    o0, vuot = cd['outcome_h0'], cd['vuot']
    row = dict(run_uid=run_uid, outcome_tam=o0, outcome_sau=final, flags=','.join(flags),
               vuot='true' if vuot else 'false', D_H_mm=_num(cd.get('D_H_mm')),
               score_final_file=os.path.abspath(score_final), score_final_sha256=sha256_file(score_final),
               ghi_luc=_iso(now()), drift_file=os.path.abspath(drift_file) if drift_file else '',
               drift_sha256=drift_sha, **drift_columns(cd))
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
            drift_path, drift_sha = s.get('drift_file'), s.get('drift_sha256')
            drift = None
            if drift_path or drift_sha:
                drift, sha = read_json_sha(drift_path or '', 'drift_file')
                if sha != drift_sha or not isinstance(drift, dict):
                    raise LogError(f'{drift_path!r}: drift_file changed since so-h')
            final, want, cd = check_score_final(r, p, ds, drift)
        except LogError as exc:
            raise LogError(f'{SO_H}: {exc}')
        if any((s.get(k) or '') != v for k, v in drift_columns(cd).items()):
            raise LogError(f"{SO_H}: run {r.get('run_uid')}: drift fields differ from the comparison file")
        if s.get('vuot') != ('true' if cd['vuot'] else 'false'):      # RC1-1: vuot decides the H1 scoring (apply_moc)
            raise LogError(f"{SO_H}: run {r.get('run_uid')}: vuot {s.get('vuot')!r} is not that of the score_final file")
        if (s.get('D_H_mm') or '') != _num(cd['D_H_mm']):
            raise LogError(f"{SO_H}: run {r.get('run_uid')}: D_H_mm differs from the comparison file")
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
    try:
        t_frz = _parse_t(frozen_at)
        history = [r for r in tun_after if r.get('campaign') == campaign and r.get('stage') in TUNE_STAGES
                   and _parse_t(r.get('wall_end') or r['wall_start']) <= t_frz]
    except (ValueError, TypeError, KeyError) as exc:
        return f'time stamps not readable ({exc})'
    conf = [r for r in history if r.get('platform') == 'HW'
            and r.get('stage') == 'confirm' and r.get('gain_set_id') == gain_set]
    if not conf:
        return f'no confirm car run of {gain_set} in the log'
    bid = conf[-1].get('block_id', '')
    if not bid:
        return f"the last confirm car run of {gain_set} ({conf[-1].get('run_uid')}) has no block_id"
    block = [r for r in conf if r.get('block_id') == bid]
    # GT1-2 (Codex round 1 of CODE_GATE; Liam 17/11 at CODE_GATE "Sửa theo bảng (Recommended)"): a level of the gain
    # set stopped after the reruns of the C8b time check is chua_ket_luan (O1 3.5 v11.37), so the set is not confirmed.
    # GT2-2 (round 2): only a level of the gate counts (`levels`, the gate file of the summary; freeze has none: the
    # levels of the block)
    levels = None if levels is None else tuple(levels)          # GT3-2: an iterator is read once, used twice
    lv = set(levels) if levels is not None else {r.get('trajectory') for r in block}
    stop = sorted(f'{k[2]}@{k[3]}' for k, n in c8b_refusals(history).items()
                  if k[0] == campaign and k[1] == gain_set and k[2] in lv and n > C8B_RERUNS_MAX)
    if stop:
        return (f"{gain_set}: level(s) {', '.join(stop)} stopped after {C8B_RERUNS_MAX} reruns for the C8b time check "
                '(chua_ket_luan, O1 3.5 v11.37; GT1-2)')
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
    except (ValueError, TypeError, KeyError) as exc:
        return f'block {bid}: time stamps not readable ({exc})'
    if t_frz < last:
        return f'frozen_at {frozen_at} is earlier than the last so-h of block {bid} ({_iso(last)})'
    try:                                               # design 6: and than the C8b v3 §5 rows of the block
        mt = moc_times(ds, [r.get('run_uid') for r in block], [r.get('session_id') for r in block])
    except (ValueError, TypeError, KeyError) as exc:
        return f'block {bid}: time stamps of {U_MOC} / {TIEN_DE} not readable ({exc})'
    if mt and t_frz < max(mt):
        return f'frozen_at {frozen_at} is earlier than the last u-moc / tien-de row of block {bid} ({_iso(max(mt))})'
    return None


def row_so_do(r: dict, gate_file=None):
    """The conditions of a run decided by q (row fields trajectory, platform, stage, campaign) from the C8 gate file
    (gate_c8.GateC8.run_so_do; CODE 23/11); None without a gate file (none)."""
    if gate_file is None:
        return None
    try:
        return gate_file.run_so_do(r.get('trajectory', ''), r.get('platform', ''), r.get('stage', ''),
                                   r.get('campaign', ''))
    except gate_c8.GateFileError as exc:
        raise LogError(f"{r.get('run_uid')}: {exc}")


def gate_file_problem(r: dict, gate_sha: str, gate_file=None) -> Optional[str]:
    """O1 v11.33 (OG2-7): None when the gate file SHA of a run's gate (gate_sha, '' = none) fits the gate file in use,
    else why not. A gate scored with a gate file needs that file here; a campaign run whose level may carry
    so_do conditions (gate_c8.mask_applies: the floor, the HIL candidate check of B) must be scored with the
    gate file in use (the run record carries its level and the file SHA)."""
    if gate_sha and gate_file is None:
        return (f'its gate was scored with the C8 gate file {gate_sha[:12]}..., not given here (--gate-file): the '
                'so_do conditions are recomputed from the file (OG2-7)')
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


def lost_gate_file_cols(r: dict, gate_file=None) -> dict:
    """GT1-6 (Codex round 1 of CODE_GATE): the gate file columns of a run lost to the measurement whose scorer file has
    no gate (CC1-2): the SHA of the C8 gate file and the so_do conditions of the run's level from it, so the H1
    score of so-h is checked against the same file and so_do set. {} without a gate file."""
    if gate_file is None:
        return {}
    return dict(gate_file_sha256=gate_file.sha256, gate_so_do=','.join(row_so_do(r, gate_file)))


def gate_file_cols_problem(r: dict, gate_file=None) -> Optional[str]:
    """GT1-6 on a logged row without a gate: gate_file_sha256 / gate_so_do only on a lost run with its scorer
    file SHA (CC1-2), equal to what the gate file in use gives (lost_gate_file_cols). GT2-1 (Codex round 2): a lost
    run logged without a gate file keeps the CC1-2 path (no columns, all five required)."""
    sha, ro = r.get('gate_file_sha256', ''), r.get('gate_so_do', '')
    if not sha and not ro:
        return None
    if r.get('run_outcome') != 'hong_he_do' or not r.get('gate_json_sha256'):
        return 'gate_file_sha256 / gate_so_do without a gate: only for a lost run with its scorer file (GT1-6)'
    if gate_file is None:
        return (f'scored with the C8 gate file {sha[:12]}..., not given here (--gate-file): the so_do conditions '
                'are recomputed from the file (OG2-7, GT1-6)')
    want = lost_gate_file_cols(r, gate_file)
    if (sha, ro) != (want['gate_file_sha256'], want['gate_so_do']):
        return (f'gate_file_sha256 {sha[:12]}... / gate_so_do {ro!r}: the gate file in use gives '
                f'{want["gate_file_sha256"][:12]}... / {want["gate_so_do"]!r} (GT1-6)')
    return None


def check_gate_rows(tun: List[dict], gate_file=None) -> None:
    """GV3-1, GV3-R2-1: every logged row that carries a gate is recomputed from its q, lo, hi, n_undef
    (version, deciding set of its platform, thresholds, states); its run_outcome must agree as at `end`.
    Rows whose gate columns are ALL empty (before gate v3, or written without --gate-json) are not
    checked; any non-empty gate cell makes the row a gate row, with a supported version (GV3-R3-1).
    O1 v11.33: gate_file (gate_c8.GateC8, the summary's --gate-file) gives the so_do set of every gate row
    (row_so_do) and the SHA its gate must carry (gate_file_problem)."""
    for r in tun:
        check_final_platform(r)
        if r.get('gate_report_only'):                   # SD1-7 (Codex 24/11 round 2): on every row, also one that
            raise LogError(f"{r.get('run_uid')}: gate_report_only {r.get('gate_report_only')!r}: report only is no "
                           'rule since 23/11; the column of before is refused on every row (SD1-7)')   # has no gate
        judged = r.get('run_outcome') in VALID
        camp = r.get('campaign') in CAMPAIGNS           # [CACH HIEU] historic rows (no campaign): SC3-2 note above
        # R134 (b); B81-2: a judged car / final_hil campaign row needs both boots, given or not
        if judged and (r.get('boot_before') or r.get('boot_after')
                       or (camp and boot_needed(r.get('platform'), r.get('stage')))):
            bbad = boot_problem(r.get('boot_before'), r.get('boot_after'))
            if bbad:
                raise LogError(f"{r.get('run_uid')}: run_outcome {r.get('run_outcome')!r} but {bbad} (S05 4.7 (iv); "
                               f'a run failed by operation is hong_he_do, {BOOT_FAIL})')
        khong_nut = (r.get('fail_cause') == KHONG_NUT and r.get('run_outcome') == 'chua_ket_luan'
                     and r.get('platform') == 'HW' and _hex64(r.get('gate_json_sha256'))
                     and (r.get('c8b_gio_sha256', '') == '' or _hex64(r.get('c8b_gio_sha256'))))  # F11, GC6-12, GC7-4
        if r.get('fail_cause') == KHONG_NUT and (not khong_nut or has_gate(r)):
            raise LogError(f"{r.get('run_uid')}: fail cause {KHONG_NUT} only on a floor chua_ket_luan row with its "
                           'c8b_luoi file (gate_json_sha256) and no gate columns (F11)')
        if not has_gate(r):
            if r.get('gate_json_sha256') and r.get('run_outcome') != 'hong_he_do' and not khong_nut:  # CC1-2
                raise LogError(f"{r.get('run_uid')}: a scorer file without its gate (gate_json_sha256 only) is for a "
                               'run lost to the measurement only')
            gbad = gate_file_cols_problem(r, gate_file)                                             # GT1-6
            if gbad:
                raise LogError(f"{r.get('run_uid')}: {gbad}")
            # R134 (d) SC3-2 (HIL rows have their own rule for every judged outcome, Codex S315-1)
            if r.get('run_outcome') == 'dat' and r.get('campaign') in CAMPAIGNS and r.get('platform') != 'HIL':
                raise LogError(f"{r.get('run_uid')}: run_outcome 'dat' without gate columns ({NO_GATE_EVIDENCE}, "
                               'SC3-2): a campaign run is never dat without the gate')
            if judged and camp and r.get('platform') == 'HW' and not khong_nut:         # B8-1 B81-1
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
            ev = _recompute(g, r.get('platform'), row_so_do(r, gate_file))
        except LogError as exc:
            raise LogError(f"{r.get('run_uid')}: {exc}")
        why = (outcome_vs_gate(r.get('run_outcome', ''), ev, g['decide'], r.get('fail_cause', ''), g)
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
           'gate_so_do': ','.join(gate.get('so_do') or [])}
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
    for g in gate_c8.MASKABLE:                         # CODE 23/11 (2): the middle node of a grid so_do condition
        qs = (gate.get(g) or {}).get('q_so_do')
        out.update({f'{g}_so_do_{k}': '' for k in ('nut', 'q', 'q_chung')})
        if qs is not None:
            out.update({f'{g}_so_do_nut': '' if qs['nut'] is None else str(int(qs['nut'])), f'{g}_so_do_q': cell(qs['q']),
                        f'{g}_so_do_q_chung': cell(qs['q_chung'])})
    g1 = gate.get('g1') or {}
    if g1.get('t_done') is not None:                   # B81-3: the G1 fields (checked by validate_gate first)
        td = g1['t_done']
        out.update(g1_ways=str(len(td)), g1_t_deadline=cell(g1.get('t_deadline')), g1_t_done_1=cell(td[0]),
                   g1_t_done_2=cell(td[1]) if len(td) > 1 else '', g1_t1=cell(g1.get('t1')),
                   g1_t_som_s=cell(g1.get('t_som_s')))
        for k in ('prog_before_t1', 'cond25_before_deadline', 'cat_t1', 'ca_t_som'):
            v = g1.get(k)
            out[f'g1_{k}'] = '' if v is None else ('true' if v else 'false')
    return out


def viol_before_exception(gate) -> bool:
    """GT1-5 (Codex round 1 of CODE_GATE): whether a gate shows the floor exception before t1 (O1 3.5 R2V-4): with a
    C8b grid, every node has it (design v2 section 2 point 5, Liam 17/11 "Khi mọi nút đều có (Recommended)"); without
    one, its G1 cat_t1. One rule for 'end', the summary and the H1 score of so-h."""
    if not isinstance(gate, dict):
        return False
    lb = gate.get('luoi')
    if lb is not None:
        cat = lb.get('cat_nut') if isinstance(lb, dict) else None
        return isinstance(cat, list) and bool(cat) and all(x is True for x in cat)
    g1 = gate.get('g1')
    return isinstance(g1, dict) and bool(g1.get('cat_t1'))


def viol_before_problem(outcome: str, fail_cause: str, gate: Optional[dict]) -> Optional[str]:
    """O1 3.5 R2V-4 (B81-3): the floor exception before t1 is khong_dat with fail cause VIOL_BEFORE and a gate whose
    G1 has cat_t1 true; one without the other is an error."""
    g1 = (gate or {}).get('g1') or {}
    lb = (gate or {}).get('luoi')
    if lb is not None:                         # C8b grid (design v2 section 2 points 4, 5; Liam 17/11)
        cat = viol_before_exception(gate)
        if cat != (fail_cause == VIOL_BEFORE) or (cat and outcome != 'khong_dat'):
            return (f'grid with the exception before t1 at {sum(lb["cat_nut"])} of {lb["n_nut"]} nodes, outcome '
                    f'{outcome!r}, fail cause {fail_cause!r}: khong_dat with {VIOL_BEFORE} exactly when every node '
                    'has it (Liam 17/11 at CODE_GATE)')
        if all(lb['hong_nut']) and outcome != 'hong_he_do':
            return f'every grid node lost to the measurement: the run is hong_he_do, not {outcome} (C8b v4 section 3)'
        return None
    cat = viol_before_exception(gate)
    if cat != (fail_cause == VIOL_BEFORE) or (cat and outcome != 'khong_dat'):
        return (f'cat_t1 {cat} with outcome {outcome!r}, fail cause {fail_cause!r}: the floor exception before t1 is '
                f'khong_dat with fail cause {VIOL_BEFORE} and cat_t1 true (O1 3.5 R2V-4), and only that')
    floor = isinstance(g1.get('t_done'), list) and len(g1['t_done']) == 2
    if floor and g1.get('t1') is not None and not cat and outcome in VALID:          # R134C4-3
        return (f'floor run with t1 {g1.get("t1")!r} (tracking error or a gap above t_trong) and no violation '
                f'before it (cat_t1 false): a run lost to the measurement, hong_he_do, not {outcome} (O1 3.5)')
    return None


def check_outcome_vs_gate(outcome: str, gate: dict, platform: Optional[str] = None, fail_cause: str = '',
                          so_do=None) -> None:
    """The run outcome typed by the operator against the gate, validated and recomputed first (GV3-5);
    so_do: the so_do set of the run from the gate file (row_so_do; None = no gate file, none)."""
    ev = validate_gate(gate, platform, so_do)
    why = (outcome_vs_gate(outcome, ev, gate['decide'], fail_cause, gate)
           or viol_before_problem(outcome, fail_cause, gate))
    if why:
        raise LogError(f'--outcome {outcome} disagrees with the gate ({why}); check the scoring')


def _num_text(s) -> str:
    """A number as one text whatever way it was typed ('1', '1.0', '1e0' -> '1.0'); other text unchanged."""
    try:
        return '' if s in ('', None) else repr(float(s))
    except (TypeError, ValueError):
        return str(s)


def gain_bits(row):
    import gain_line_check as glc
    try:
        return {x: glc.f32_bits(row['g_' + x]) for x in GAINS if row.get('g_' + x) not in (None, '')}
    except glc.InputError as exc:
        raise LogError(f"{row.get('run_uid')}: {exc}") from exc


def map_key(r: dict) -> tuple:
    """One candidate at one level (S05 v3.15 4.5 step 2): (campaign, gain set id, task, speed as a number).
    The stage is not in it (Codex S315-3: a run of the same gain at the same level in another HIL stage counts
    against the same reruns); the gain values are checked for agreement in map_refusals (S315-2). A figure-8
    task gives its level FIG8[task], not the speed as typed (S315-R2-1: check_task allows 1e-9 around it)."""
    return (r.get('campaign', ''), r.get('gain_set_id', ''), *condition_key(r))


def condition_key(r: dict) -> tuple:
    task = r.get('trajectory', '')
    return task, repr(FINAL_SPEEDS[task]) if task in FINAL_SPEEDS else _num_text(r.get('speed_mps'))


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
        g = gain_bits(r)
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
        if r.get('platform') == 'HIL' and r.get('run_outcome') in VALID and not has_gate(r):
            raise LogError(f"{uid}: HIL run {r.get('run_outcome')} without its gate (--gate-json): S05 v3.15 4.5 "
                           'step 2 needs it to tell a mapped run from a refused one')


def final_gain_key(gs):
    owner = (gs or '').partition('/')[0]
    return owner if owner in ('RQ2-INIT-PID', 'RQ2-INIT-LADRC') else gs


def final_key(r: dict) -> tuple:
    """A condition of the final session (S05 4.6), as in _final_stops."""
    gs = r.get('gain_set_id', '')
    if r.get('stage') in ('final', 'final_hil'):
        gs = final_gain_key(gs)
        if r.get('stage') == 'final' and r.get('campaign') in ('RQ2-INIT-PID', 'RQ2-INIT-LADRC'):
            if gs and gs != r['campaign']:
                raise LogError(f"{r.get('run_uid')}: gain set {gs} does not belong to {r['campaign']}")
            gs = r['campaign']  # Historical floor rows can identify the initial set by campaign alone.
    return (r.get('stage', ''), r.get('campaign', ''), gs, r.get('platform', ''),
            *condition_key(r))


def final_lost_streak(tun: List[dict], key: tuple) -> int:
    """Lost runs in a row at the end of one final condition (S05 4.6), in the order of the log."""
    s = 0
    for r in tun:
        if r.get('stage') in ('final', 'final_hil') and final_key(r) == key:
            s = s + 1 if r.get('run_outcome') == 'hong_he_do' else 0
    return s


def _c8b_counted(r: dict) -> bool:
    return r.get('platform') == 'HW' and r.get('stage') != 'final'


def c8b_refusals(tun: List[dict]) -> Dict[tuple, int]:
    """O1 v11.37 D1: kiem_gio_c8b runs per (campaign, gain set, level) (map_key), stages but final, in the order of
    the log. A counted run of a key after its 1 + C8B_RERUNS_MAX-th such run is an error (LogError), as is a
    kiem_gio_c8b run without gain set id.
    GT1-7 (Codex round 1 of CODE_GATE; Liam 17/11 at CODE_GATE "Sửa theo bảng (Recommended)"): the count cannot be
    reset by the gain identity: every counted run of a campaign that owns gain sets carries its gain set id; the gain
    values of one key agree (as numbers, as S315-2 on HIL); after a key stopped, a run at the same level of the
    campaign with exactly the gain values of the stopped set under another id is an error."""
    n = defaultdict(int)
    gains = {}
    stopped = defaultdict(list)                  # (campaign, task, speed) -> [(gain set id, its gain values)]
    for r in tun:
        if not _c8b_counted(r):
            continue
        uid, k = r.get('run_uid'), map_key(r)
        own = r.get('campaign') in GAIN_OWNERS   # GT2-3: the identity checks only for runs of a campaign with gain sets
        if not r.get('gain_set_id') and own:
            raise LogError(f"{uid}: a floor run of {r.get('campaign')} outside the final session carries its gain set "
                           'id (--gain-set): the reruns after the C8b time check are counted per gain set (GT1-7)')
        lvl = (k[0], k[2], k[3])
        if n[k] > C8B_RERUNS_MAX:
            raise LogError(f"{uid}: {r.get('trajectory')}@{r.get('speed_mps')} of {r.get('gain_set_id')} "
                           f'stopped after {C8B_RERUNS_MAX} reruns for the C8b time check (O1 3.5 v11.37: the level is '
                           'chua_ket_luan), no run after that')
        old = gains.setdefault(k, {})
        if own:
            g = gain_bits(r)
            diff = sorted(x for x in set(g) & set(old) if g[x] != old[x])
            if diff:
                raise LogError(f"{uid}: gain set {r.get('gain_set_id')} with other values of {', '.join(diff)} than an "
                               f"earlier floor run at {r.get('trajectory')}@{r.get('speed_mps')} (one gain set, one "
                               'gain; GT1-7)')
            old.update(g)
            # GT1-7 (rounds 2, 3): another gain set at a level where a set stopped runs only with evidence that it is
            # another candidate: against EVERY stopped set, a gain name known for both (this row and the earlier rows
            # of the key, the stopped set's rows) with another value. No name in common is no evidence: refused
            others = [(gs, gg) for gs, gg in stopped[lvl] if gs != k[1]]
            twin = [gs for gs, gg in others if not any(x in old and old[x] != gg[x] for x in gg)]
            if twin:
                raise LogError(f"{uid}: {r.get('gain_set_id')} at {r.get('trajectory')}@{r.get('speed_mps')} cannot be "
                               f'told from {twin[0]}, stopped after {C8B_RERUNS_MAX} reruns for the C8b time check: no '
                               'gain known for both with another value (one candidate under another name is refused; '
                               'give the gains of a new candidate with --gains; GT1-7)')
        if r.get('fail_cause') == C8B_GIO_FAIL:
            if not r.get('gain_set_id'):
                raise LogError(f"{uid}: a {C8B_GIO_FAIL} run carries its gain set id (counted per gain set)")
            n[k] += 1
            if n[k] == C8B_RERUNS_MAX + 1:
                stopped[lvl].append((k[1], old))
    return dict(n)


def _file_by_sha(folder: str, sha: str) -> Optional[str]:
    """The file under folder (links resolved inside it, clips and big files skipped) whose sha256 is sha."""
    for root, _, names in os.walk(folder):
        for n in sorted(names):
            q = os.path.join(root, n)
            try:
                if n.lower().endswith(C8B_HASH_SKIP) or os.path.getsize(q) > C8B_HASH_MAX_BYTES:
                    continue
                if sha256_file(q) == sha and _inside(q, folder):
                    return q
            except OSError:
                continue
    return None


def _time_file_of(rd: str, sha: str, task: str, what: str) -> dict:
    """The c8b_gio_v1 time file under rd whose sha256 is sha, of task (GC6-12, GC7-3, GC7-4); parsed from the bytes
    hashed (round 3)."""
    q = _file_by_sha(rd, sha) if _hex64(sha) else None
    if q is None:
        raise LogError(f'{what}: no time file with sha256 {str(sha)[:12]}... under {rd}')
    g, got = read_json_sha(q, what)
    if got != sha:
        raise LogError(f'{what}: {q} changed while it was read')
    if not isinstance(g, dict) or g.get('format') != 'c8b_gio_v1' or g.get('task') != task \
            or not isinstance(g.get('sha256'), dict) or not _hex64(g['sha256'].get('esp32')):
        raise LogError(f'{what}: {q} is not a c8b_gio_v1 file of task {task!r} naming its ESP32 log')
    return g


def check_khong_nut_rows(tun: List[dict], ds: str) -> None:
    """GC6-12, GC7-4: a KHONG_NUT row read again from its sources: its H0 file (gate_json_sha256 under results_dir) is
    a c8b_luoi file without a scored node, and the time file it names is under results_dir, of the row's task; the
    row's c8b_gio_sha256 is that file (empty on a row of the writer before GC6-12, then checked the same way). Round 3:
    the rule of 'end' (khong_nut_sources) on the bytes hashed; GC8-2: no two rows on one H0 file or one time file,
    compared on the SHA read back from the sources (an empty c8b_gio_sha256 cell included), and against the
    c8b_gio_sha256 of the other rows."""
    seen = {r.get('c8b_gio_sha256'): r.get('run_uid') for r in tun
            if r.get('fail_cause') != KHONG_NUT and r.get('c8b_gio_sha256')}
    seen_h0 = {}
    for r in tun:
        if r.get('fail_cause') != KHONG_NUT:
            continue
        uid = r.get('run_uid')
        h0 = h0_file(r, ds)
        d, h0_sha = read_json_sha(h0, f'{uid}: H0 file')
        if h0_sha != r.get('gate_json_sha256'):
            raise LogError(f'{uid}: {h0} changed while it was read')
        if not isinstance(d, dict) or check_luoi_dict(d, f'{uid}: H0 file {h0}') != KHONG_NUT:
            raise LogError(f'{uid}: fail cause {KHONG_NUT} but its H0 file is not a c8b_luoi file without a scored node')
        if r.get('c8b_gio_sha256') not in ('', d['gio_sha256']):
            raise LogError(f'{uid}: c8b_gio_sha256 is not the time file its c8b_luoi file was made from (GC6-12)')
        g = _time_file_of(_results_dir(r, ds), d['gio_sha256'], r.get('trajectory'), f'{uid}')
        khong_nut_sources(d, h0_sha, g, d['gio_sha256'], r.get('trajectory'), f'{uid}')
        for sha, book, kind in ((d['gio_sha256'], seen, 'time file'), (h0_sha, seen_h0, 'H0 file')):
            if sha in book:
                raise LogError(f'{uid}: the {kind} {sha[:12]}... is that of another run ({book[sha]}): one per run '
                               '(GT1-8, GC8-2)')
            book[sha] = uid


def check_c8b_rows(tun: List[dict]) -> None:
    """S05 v3.43 D3 on logged rows: a kiem_gio_c8b row is a floor hong_he_do row with the sha256 of its C8b time file;
    c8b_gio_sha256 only on such a row, never the same file on two runs (GT1-8); and the count (c8b_refusals)."""
    seen = {}
    for r in tun:
        sha = r.get('c8b_gio_sha256', '')
        if r.get('fail_cause') == C8B_GIO_FAIL:
            if r.get('platform') != 'HW' or r.get('run_outcome') != 'hong_he_do' or not _hex64(sha):
                raise LogError(f"{r.get('run_uid')}: fail cause {C8B_GIO_FAIL} on a row that is not a floor hong_he_do "
                               'run with the sha256 of its C8b time file (S05 v3.43 6.1)')
        elif sha and r.get('fail_cause') != KHONG_NUT:                       # GC6-12: and its F11 time file
            raise LogError(f"{r.get('run_uid')}: c8b_gio_sha256 without fail cause {C8B_GIO_FAIL} or {KHONG_NUT}")
        if sha:
            if sha in seen:
                raise LogError(f"{r.get('run_uid')}: the C8b time file {sha[:12]}... is that of another run "
                               f'({seen[sha]}): one file per run (GT1-8)')
            seen[sha] = r.get('run_uid')
    c8b_refusals(tun)


def _hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(ch in '0123456789abcdef' for ch in x)


C8B_GIO_INPUTS = ('esp32', 'frames')     # c8b_gio_v1 'sha256': the ESP32 log and the frames of ITS run (CODE_C8B)
C8B_HASH_MAX_BYTES = 200 * 2 ** 20       # [DE XUAT] files hashed when looking for them under the run folder
C8B_HASH_SKIP = ('.mov', '.mp4', '.m4v')  # clips (L88) are never the scored inputs


def _hashes_under(folder: str, skip: str = '') -> set:
    """sha256 of the files under `folder` that may be scored inputs (links resolved inside it; clips, files above
    C8B_HASH_MAX_BYTES, unreadable files and `skip` left out)."""
    out = set()
    for root, _, names in os.walk(folder):
        for n in names:
            p = os.path.join(root, n)
            real = os.path.realpath(p)
            # G8-1: the clip test on the name walked and on the file a link resolves to
            if n.lower().endswith(C8B_HASH_SKIP) or real.lower().endswith(C8B_HASH_SKIP) or (
                    skip and os.path.normcase(real) == os.path.normcase(os.path.realpath(skip))):
                continue
            try:
                if os.path.isfile(p) and os.path.getsize(p) <= C8B_HASH_MAX_BYTES and _inside(p, folder):
                    out.add(sha256_file(p))
            except OSError:
                continue
    return out


def read_c8b_gio(path: str, task: Optional[str] = None, results_dir: Optional[str] = None) -> str:
    """The C8b time file of a run lost to its check (format c8b_gio_v1, ok false, ket_luan hong_he_do); returns its sha256.
    GT1-8 (Codex round 1 of CODE_GATE; the deferred part closed when CODE_C8B fixed c8b_gio_v1, Liam 17/11 at CODE_GATE
    "Làm, nhánh nhỏ + Codex (Recommended)"): task, the run's task, must be the 'task' of the file; with results_dir, the
    file's 'sha256' {esp32, frames} (64 lowercase hex each) must each be the sha256 of a file under the run folder, so
    the time file is the one scored on this run's own log and frames."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError) as exc:
        raise LogError(f'--c8b-gio {path!r}: {exc}')
    if not isinstance(d, dict) or d.get('format') != C8B_GIO_FORMAT or d.get('ok') is not False:
        raise LogError(f'--c8b-gio {path!r}: not a {C8B_GIO_FORMAT} file with ok false (the run must have failed the '
                       'C8b time check, O1 3.5 v11.37)')
    # CODE_GATE 19/11 (bug of e66ecd1, CODE's order of 19/11): c8b_gio.py writes ok false for two verdicts
    # (c8b_gio.py 585, 659): ket_luan hong_he_do (lost to the measurement: run again, counted against
    # C8B_RERUNS_MAX) and ket_luan chua_ket_luan (no finite clock bound: G1-G5 not concluded, the run is
    # chua_ket_luan, not lost, not counted). Only the first is a kiem_gio_c8b run.
    if d.get('ket_luan') != 'hong_he_do':
        raise LogError(f'--c8b-gio {path!r}: ket_luan {d.get("ket_luan")!r}, not hong_he_do: only a run lost to the C8b '
                       'time check is a kiem_gio_c8b run counted against the reruns (ket_luan chua_ket_luan: no finite '
                       'clock bound, log the run chua_ket_luan without --c8b-gio)')
    k1 = d.get('kiem_gio_d1')
    if not isinstance(k1, dict) or k1.get('trang_thai') not in D1_STATES:
        raise LogError(f'--c8b-gio {path!r}: no kiem_gio_d1 block {{trang_thai: {" | ".join(D1_STATES)}}} (F11): a time '
                       'file without it cannot tell a slip of the D1 time check from a pair-count or log-hole fault')
    if k1['trang_thai'] != 'truot':
        raise LogError(f'--c8b-gio {path!r}: kiem_gio_d1 {k1["trang_thai"]!r}: only a slip of the D1 time check '
                       f'(truot) is a {C8B_GIO_FAIL} run counted against the reruns; a run lost by cap_dem / ho_log '
                       'alone is hong_he_do with another fail cause, not counted (F11)')
    # M2 (J v6 5, approved; CODE_C8B design v7 7a): the time file says whether the slip counts (dem_truot_kiem_gio:
    # not with a boot unless the locked check slipped on the whole window before it); counted by the field
    if not isinstance(d.get('dem_truot_kiem_gio'), bool):
        raise LogError(f'--c8b-gio {path!r}: no dem_truot_kiem_gio (true / false; F11, M2)')
    if not d['dem_truot_kiem_gio']:
        raise LogError(f'--c8b-gio {path!r}: dem_truot_kiem_gio false: a slip on a run lost by operation (boot) '
                       f'without independent evidence is not a {C8B_GIO_FAIL} run (M2); log it hong_he_do with '
                       f'{BOOT_FAIL}')
    if task is not None and d.get('task') != task:
        raise LogError(f'--c8b-gio {path!r}: the time file of task {d.get("task")!r}, the run is {task!r} (GT1-8)')
    if results_dir is not None:
        sh = d.get('sha256')
        if not isinstance(sh, dict) or any(not _hex64(sh.get(k)) for k in C8B_GIO_INPUTS):
            raise LogError(f'--c8b-gio {path!r}: no sha256 {{{", ".join(C8B_GIO_INPUTS)}}} of 64 lowercase hex digits '
                           f'({C8B_GIO_FORMAT}, the inputs of the run; GT1-8)')
        have = _hashes_under(results_dir, skip=path)
        miss = [k for k in C8B_GIO_INPUTS if sh[k] not in have]
        if miss:
            raise LogError(f'--c8b-gio {path!r}: the {" and ".join(miss)} input(s) it was made from are not files under '
                           f'the run folder {results_dir!r} (the time file of another run? GT1-8)')
    return sha256_file(path)


# ── C8b v3 §5: session premise and shifted-anchor bound of campaign car runs (CODE_GATE 18/11) ──────────────
# Design CLAUDE_THESIS 03_CODE/docs/2026-11-18_Thiet_ke_rq3_u_moc_v4.md (Liam 18/11 at CODE_GATE, hop chon, nguyen van:
# "Duyệt, code luôn (Recommended)"; Codex Sol 4 rounds RM1-1 ... RM4-1). Two append-only logs written after the
# session: one row per (car run, scoring) with the U file of CODE_CAM (u_moc_luot_v1), one row per session with the
# premise file of CODE_KIEMCHEO (tien_de_buoi_v1). apply_moc (after apply_so_h) turns a dat / khong_dat car run of a
# campaign into chua_ket_luan unless every condition of design section 5 holds; a missing file never keeps it.
# CODE_GATE 19/11 (design v6_C8 section 10; CODE's choice (a) of 19/11, local_e6d36dbe): the C8 gate is gate_c8_v2,
# whose shifted-anchor lock has four numbers; the campaign side below (u-moc, apply_moc) still compares the two
# numbers of the v1 lock. Until the campaign part (design v4 section 3.2, after the C8 lock) it refuses with this one
# text, so every place to reopen is found by grep.
CAMPAIGN_V2_TODO = ('CAMPAIGN_V2_TODO: the campaign side of rq3_log (u-moc, apply_moc) does not run with gate_c8_v2 '
                    'yet (four-number shifted-anchor lock; design 03_CODE/docs/2026-11-19_Thiet_ke_rq3_tuong_doi_v4.md '
                    'section 3.2, after the C8 lock)')
U_MOC = 'm7_u_moc.csv'
U_MOC_COLS = ['run_uid', 'lan', 'u_file', 'u_sha256', 'gate_file', 'gate_file_sha256', 'ok', 'U_moc_xy_m',
              'U_moc_psi_deg', 'khoa_xy_m', 'khoa_psi_deg', 'khoa_sha256', 'ghi_luc']
TIEN_DE = 'm7_tien_de_buoi.csv'
TIEN_DE_COLS = ['session_id', 'ngay', 'ket_qua', 'file', 'sha256', 'ghi_luc']
U_MOC_FORMAT, TIEN_DE_FORMAT = 'u_moc_luot_v1', 'tien_de_buoi_v1'
LANS = ('H0', 'H1')                       # the scoring of the run (H0) and, after so-h above eps_H, the clip H one (H1)
U_ROLES = ('frames', 'lich', 'tap_moc', 'u_so', 'dets', 'session')    # design 3.1, + 'gio' for a luoi.json scoring
TIEN_DE_KQ = ('dat', 'khong_dat', 'thieu_du_lieu')
MOC_DK = ('dat', 'khong_dat')             # the outcomes the rule can turn into chua_ket_luan
MOC_FLAGS = ('cho_so_h', 'khong_gan_duoc_lan_cham', 'cho_tien_de', 'tien_de_khong_dat', 'tien_de_thieu', 'tien_de_khac_ngay',
             'tien_de_pham_vi_thieu', 'tien_de_khong_phu_h1', 'tien_de_luot_thieu', 'cho_u_moc', 'u_moc_khong_tinh',
             'u_moc_nguon_khac_khoa', 'u_moc_khac_buoi', 'u_moc_vuot')


def _moc_run(r: dict) -> bool:
    """A car run of a campaign: the runs the rule covers (design section 2, the runs of so-h)."""
    return r.get('platform') == 'HW' and r.get('campaign') in CAMPAIGNS


def _nonneg(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise LogError(f'{what} {x!r}: a finite number >= 0')
    return float(x)


def _json_file(path: str, what: str) -> dict:
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError) as exc:
        raise LogError(f'{what} {path!r}: {exc}')
    if not isinstance(d, dict):
        raise LogError(f'{what} {path!r}: not a JSON object')
    return d


def _nguon(d: dict, base: str, what: str, unique: bool) -> List[tuple]:
    """The 'nguon' list [{vai_tro, duong_dan, sha256}] of a U or premise file, every file hashed again (RM1-4): a
    missing file or another sha256 refuses. A relative path is taken from the folder of the file. unique: one entry
    per role (U file). Returns [(role, absolute path, sha256)]."""
    ng = d.get('nguon')
    if not isinstance(ng, list):
        raise LogError(f'{what}: nguon must be a list of {{vai_tro, duong_dan, sha256}}')
    out, seen = [], set()
    for e in ng:
        if not isinstance(e, dict) or not isinstance(e.get('vai_tro'), str) or not isinstance(e.get('duong_dan'), str) \
                or not _hex64(e.get('sha256')):
            raise LogError(f'{what}: nguon entry {e!r}: vai_tro, duong_dan (text), sha256 (64 lowercase hex)')
        if unique and e['vai_tro'] in seen:
            raise LogError(f"{what}: nguon role {e['vai_tro']!r} twice")
        seen.add(e['vai_tro'])
        p = e['duong_dan'] if os.path.isabs(e['duong_dan']) else os.path.join(base, e['duong_dan'])
        try:
            same = os.path.isfile(p) and sha256_file(p) == e['sha256']
        except OSError:
            same = False
        if not same:
            raise LogError(f"{what}: source {e['vai_tro']} {p!r} missing or changed (sha256 differs; RM1-4)")
        out.append((e['vai_tro'], os.path.abspath(p), e['sha256']))
    return out


def _so_h_row(ds: str, uid: str) -> Optional[dict]:
    return next((s for s in read_rows(os.path.join(ds, SO_H)) if s.get('run_uid') == uid), None)


def h1_needed(so_row: Optional[dict], outcome: str) -> bool:
    """Design 3.0 (RM2-1): the clip H scoring H1 is needed too when so-h was above eps_H with the clip H at the end
    present and the outcome after it is dat / khong_dat (H0 and H1 judged the same, so both carry the verdict)."""
    if so_row is None or so_row.get('vuot') != 'true' or outcome not in MOC_DK:
        return False
    return 'thieu_H1' not in (so_row.get('flags') or '').split(',')


def scoring_of(r: dict, ds: str, lan: str) -> dict:
    """The scorer output of scoring `lan` of run r: H0 the file under results_dir with the gate_json_sha256 of the row
    (h0_file); H1 the h1_score kept in the score_final of the run's so-h row (that file unchanged since so-h)."""
    if lan == 'H0':
        return _json_file(h0_file(r, ds), f"run {r.get('run_uid')}: H0 scoring")
    s = _so_h_row(ds, r.get('run_uid'))
    if s is None or s.get('vuot') != 'true':
        raise LogError(f"run {r.get('run_uid')}: no H1 scoring (no so-h row above eps_H)")
    p = s.get('score_final_file', '')
    if not os.path.isfile(p) or sha256_file(p) != s.get('score_final_sha256'):
        raise LogError(f"run {r.get('run_uid')}: score_final {p!r} missing or changed since so-h")
    h1 = (_json_file(p, f"run {r.get('run_uid')}: score_final").get('calib_drift') or {}).get('h1_score')
    if not isinstance(h1, dict):
        raise LogError(f"run {r.get('run_uid')}: score_final {p!r} has no calib_drift.h1_score")
    return h1


def frames_of(sc: dict, gio: Optional[tuple], what: str) -> str:
    """frames_L of a scoring (design 3.0): frames_sha256 of a c7_score output; for a luoi.json (gio_sha256 only, RM2-5)
    sha256.frames of the time file given as source 'gio' (path, sha256), whose sha256 must be the gio_sha256 of the
    scoring. Both present: they must agree."""
    fs, gs, via = sc.get('frames_sha256'), sc.get('gio_sha256'), None
    if gio is not None and gs is not None:
        if gio[2] != gs:
            raise LogError(f'{what}: source gio {gio[1]!r} is not the time file of the scoring (gio_sha256 differs)')
        via = (_json_file(gio[1], f'{what}: time file').get('sha256') or {}).get('frames')
        if not _hex64(via):
            raise LogError(f'{what}: time file {gio[1]!r} has no sha256.frames')
    if _hex64(fs):
        if via is not None and via != fs:
            raise LogError(f'{what}: frames_sha256 of the scoring and sha256.frames of its time file differ')
        return fs
    if via is not None:
        return via
    raise LogError(f'{what}: the frames of the scoring cannot be found (no frames_sha256; a luoi.json scoring needs '
                   "the time file as source 'gio' of the U file; design 3.0)")


def read_u_moc(path: str, r: dict, ds: str, lan: str) -> dict:
    """A U file (u_moc_luot_v1, CODE_CAM) of scoring `lan` of run r, checked by design 3.0-3.1. Returns
    {ok, xy, psi, bo_lay_mau, car_pose, tap_moc, u_so, session, frames, gio}."""
    uid = r.get('run_uid')
    w = f'run {uid} {lan}: U file {path!r}'
    rd = _results_dir(r, ds)
    if rd is None or not _inside(path, rd):
        raise LogError(f'{w}: not under the results_dir of the run ({r.get("results_dir")!r})')
    d = _json_file(path, w)
    if d.get('format') != U_MOC_FORMAT:
        raise LogError(f'{w}: format {d.get("format")!r}, not {U_MOC_FORMAT}')
    if d.get('task') != r.get('trajectory'):
        raise LogError(f"{w}: task {d.get('task')!r}, the run is {r.get('trajectory')!r}")
    ok = d.get('ok')
    if not isinstance(ok, bool):
        raise LogError(f'{w}: ok must be true or false')
    sh = d.get('sha256')
    if not isinstance(sh, dict) or not _hex64(sh.get('frames')):
        raise LogError(f'{w}: sha256.frames (64 lowercase hex) is needed in both branches (RM1-6)')
    src = {role: (role, p, s) for role, p, s in _nguon(d, os.path.dirname(os.path.abspath(path)), w, unique=True)}
    gio = src.get('gio')
    sc = scoring_of(r, ds, lan)
    frames = frames_of(sc, gio, w)
    if sh['frames'] != frames or ('frames' in src and src['frames'][2] != frames):
        raise LogError(f'{w}: sha256.frames is not the frames of scoring {lan} of the run (the U file of another run '
                       'or scoring? RM1-2)')
    out = dict(ok=ok, frames=frames, gio=gio, xy=None, psi=None, bo_lay_mau=None, car_pose=None, tap_moc=None,
               u_so=None, session=None)
    ly = d.get('ly_do')
    if not ok:                                         # RM2-3: identity, reason, sources present; nothing compared
        if not isinstance(ly, list) or not ly or any(not isinstance(x, str) for x in ly):
            raise LogError(f'{w}: ok false needs a non-empty ly_do list')
        return out
    miss = [k for k in U_ROLES if k not in src]
    if miss:
        raise LogError(f'{w}: ok true needs every source {list(U_ROLES)}; missing {miss}')
    if sc_is_luoi(sc) and gio is None:
        raise LogError(f"{w}: a luoi.json scoring needs the source 'gio' (RM2-5)")
    blm = d.get('bo_lay_mau')
    if not isinstance(blm, str) or not blm:
        raise LogError(f'{w}: bo_lay_mau must be a version string')
    try:
        cp = gate_c8.car_pose_of(d.get('car_pose'), w)
    except gate_c8.GateFileError as exc:
        raise LogError(str(exc))
    out.update(xy=_nonneg(d.get('U_moc_xy_m'), f'{w}: U_moc_xy_m'),
               psi=_nonneg(d.get('U_moc_psi_deg'), f'{w}: U_moc_psi_deg'), bo_lay_mau=blm, car_pose=cp,
               tap_moc=src['tap_moc'][2], u_so=src['u_so'][2], session=src['session'][2])
    return out


def sc_is_luoi(sc: dict) -> bool:
    return not _hex64(sc.get('frames_sha256')) and sc.get('gio_sha256') is not None


def read_tien_de(path: str, session: str) -> dict:
    """A session premise file (tien_de_buoi_v1, CODE_KIEMCHEO) checked by design 3.2. Returns {ket_qua, ngay, u_so,
    sessions (set of sha256), frames {sha256: id}, thieu (set of ids)}; for ket_qua other than dat only the first two."""
    w = f'session {session}: premise file {path!r}'
    d = _json_file(path, w)
    if d.get('format') != TIEN_DE_FORMAT:
        raise LogError(f'{w}: format {d.get("format")!r}, not {TIEN_DE_FORMAT}')
    if d.get('session_id') != session:
        raise LogError(f"{w}: session_id {d.get('session_id')!r}")
    ngay = d.get('ngay')
    try:
        if not isinstance(ngay, str) or dt.date.fromisoformat(ngay).isoformat() != ngay:
            raise ValueError
    except ValueError:
        raise LogError(f'{w}: ngay {ngay!r}: YYYY-MM-DD (RM1-3)')
    kq, ly = d.get('ket_qua'), d.get('ly_do')
    if kq not in TIEN_DE_KQ or not isinstance(ly, list) or any(not isinstance(x, str) for x in ly):
        raise LogError(f'{w}: ket_qua one of {TIEN_DE_KQ}, ly_do a list of text')
    src = _nguon(d, os.path.dirname(os.path.abspath(path)), w, unique=False)
    out = dict(ket_qua=kq, ngay=ngay)
    if kq != 'dat':                                    # RM2-3: the missing-data schema of S6; the session is lowered
        return out
    sh = d.get('sha256')
    if not isinstance(sh, dict) or not _hex64(sh.get('u_so')):
        raise LogError(f'{w}: ket_qua dat needs sha256.u_so')
    if not any(role == 'u_so' and s == sh['u_so'] for role, _, s in src):
        raise LogError(f'{w}: sha256.u_so is not the sha256 of the source u_so of nguon')
    ses = sh.get('sessions')
    if not isinstance(ses, dict) or not ses or any(not _hex64(v) for v in ses.values()):       # RM2-4: a mapping
        raise LogError(f'{w}: sha256.sessions must be a mapping {{name: sha256}}')
    # S6-10 of CODE_KIEMCHEO (approved 18/11 at KIEMCHEO): 'luot' is the list of run ids (lost runs with a scoring
    # included), their frames at sha256.luot[id].frames; h1_ho_tro false: the file covers no H1 scoring
    ids, luot, thieu, h1 = d.get('luot'), sh.get('luot'), d.get('luot_thieu_kiem'), d.get('h1_ho_tro', False)
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise LogError(f'{w}: ket_qua dat needs luot, the list of run ids')
    if not isinstance(luot, dict) or set(luot) != set(ids) or any(not isinstance(v, dict) or not _hex64(v.get('frames'))
                                                                  for v in luot.values()):
        raise LogError(f'{w}: sha256.luot must give {{frames: sha256}} for every id of luot, no other')
    fr, fr1, seen = {}, {}, set()      # RC1-3 (S6-10b): a record of the H1 scoring says lan_cham H1; none says H0
    for k in ids:
        f, lan = luot[k]['frames'], luot[k].get('lan_cham', 'H0')
        if lan not in LANS:
            raise LogError(f'{w}: sha256.luot[{k!r}].lan_cham {lan!r}: H0 or H1')
        if f in seen:
            raise LogError(f'{w}: frames {f[:12]} on two records ({k})')
        seen.add(f)
        (fr if lan == 'H0' else fr1)[f] = k
    if not isinstance(thieu, list) or any(x not in ids for x in thieu):
        raise LogError(f'{w}: luot_thieu_kiem must list ids of luot')
    if not isinstance(h1, bool):
        raise LogError(f'{w}: h1_ho_tro must be true or false')
    out.update(u_so=sh['u_so'], sessions=set(ses.values()), frames=fr, frames_h1=fr1 if h1 else {}, thieu=set(thieu),
               h1=h1)
    return out


def _live_run(ds: str, uid: str) -> dict:
    r = next((x for x in live_rows(read_rows(os.path.join(ds, TUNING))) if x.get('run_uid') == uid), None)
    if r is None:
        raise LogError(f'run {uid} not in the tuning log')
    if not _moc_run(r):
        raise LogError(f'run {uid}: the C8b v3 §5 files are for car runs of a campaign')
    return r


def _gate_lock(path: str, r: dict) -> tuple:
    """(gate file, its moc_dich lock) for run r: the gate file the run was scored with (RM1-5). A gate_c8_v2 lock
    (four numbers) is not compared yet: CAMPAIGN_V2_TODO."""
    gf = read_gate_file(path, allow_fake=True)          # refused below for any v2 gate (CAMPAIGN_V2_TODO)
    if gf.sha256 != r.get('gate_file_sha256'):
        raise LogError(f"run {r.get('run_uid')}: gate file {path!r} is not the one of the run (gate_file_sha256)")
    raise LogError(f'gate file {path!r} ({gate_c8.FORMAT}): {CAMPAIGN_V2_TODO}')


def u_moc(ds: str, run_uid: str, lan: str, path: str, gate_file: str) -> dict:
    """Command u-moc (design section 4): one row of U_MOC for scoring `lan` of a car run of a campaign."""
    if lan not in LANS:
        raise LogError(f'--lan one of {LANS}')
    r = _live_run(ds, run_uid)
    if any(x.get('run_uid') == run_uid and x.get('lan') == lan for x in read_rows(os.path.join(ds, U_MOC))):
        raise LogError(f'run {run_uid} {lan}: already in {U_MOC} (one row per run and scoring, append only)')
    if lan == 'H1':
        s = _so_h_row(ds, run_uid)
        if s is None or s.get('vuot') != 'true' or 'thieu_H1' in (s.get('flags') or '').split(','):
            raise LogError(f'run {run_uid}: --lan H1 only after so-h above eps_H with the clip H at the end (RM2-1)')
    gf, lock = _gate_lock(gate_file, r)
    info = read_u_moc(path, r, ds, lan)
    row = dict(run_uid=run_uid, lan=lan, u_file=os.path.abspath(path), u_sha256=sha256_file(path),
               gate_file=os.path.abspath(gate_file), gate_file_sha256=gf.sha256, ok='true' if info['ok'] else 'false',
               U_moc_xy_m=_num(info['xy']) if info['ok'] else '', U_moc_psi_deg=_num(info['psi']) if info['ok'] else '',
               khoa_xy_m=_num(lock['U_moc_xy_m']), khoa_psi_deg=_num(lock['U_moc_psi_deg']),
               khoa_sha256=lock['khoa_sha256'], ghi_luc=_iso(now()))
    append_row(os.path.join(ds, U_MOC), U_MOC_COLS, row)
    return row


def tien_de(ds: str, session: str, path: str) -> dict:
    """Command tien-de (design section 4): one row of TIEN_DE for a session with car runs of a campaign."""
    if not any(_moc_run(r) and r.get('session_id') == session for r in live_rows(read_rows(os.path.join(ds, TUNING)))):
        raise LogError(f'session {session}: no car run of a campaign in the tuning log')
    if any(x.get('session_id') == session for x in read_rows(os.path.join(ds, TIEN_DE))):
        raise LogError(f'session {session}: already in {TIEN_DE} (one row per session, append only)')
    info = read_tien_de(path, session)
    row = dict(session_id=session, ngay=info['ngay'], ket_qua=info['ket_qua'], file=os.path.abspath(path),
               sha256=sha256_file(path), ghi_luc=_iso(now()))
    append_row(os.path.join(ds, TIEN_DE), TIEN_DE_COLS, row)
    return row


def _same_file(p: str, sha: str) -> bool:
    try:
        return os.path.isfile(p) and sha256_file(p) == sha
    except OSError:
        return False


def apply_moc(tun: List[dict], ds: str) -> List[dict]:
    """Design section 5 on the rows after apply_so_h: every U_MOC / TIEN_DE row checked again as at its command (file,
    gate file and every source unchanged, run / session in the log, lock numbers as written); then a dat / khong_dat
    car run of a campaign becomes chua_ket_luan with 'moc_flags' when any condition holds. Rows are copies."""
    by_uid = {r.get('run_uid'): r for r in tun}
    urows, trows = read_rows(os.path.join(ds, U_MOC)), read_rows(os.path.join(ds, TIEN_DE))
    gates, uinfo = {}, {}
    for x in urows:
        key = (x.get('run_uid'), x.get('lan'))
        w = f"{U_MOC}: run {key[0]} {key[1]}"
        if key in uinfo:
            raise LogError(f'{w}: two rows')
        r = by_uid.get(key[0])
        if r is None or not _moc_run(r) or key[1] not in LANS:
            raise LogError(f'{w}: not a car run of a campaign in the tuning log')
        if not _same_file(x.get('u_file', ''), x.get('u_sha256')):
            raise LogError(f"{w}: U file {x.get('u_file')!r} missing or changed since u-moc")
        if key[1] == 'H1':
            s = _so_h_row(ds, key[0])
            if s is None or s.get('vuot') != 'true' or 'thieu_H1' in (s.get('flags') or '').split(','):
                raise LogError(f'{w}: no so-h row above eps_H with the clip H at the end for this H1 row')
        gp = x.get('gate_file', '')
        if gp not in gates:
            if not _same_file(gp, x.get('gate_file_sha256')):
                raise LogError(f'{w}: gate file {gp!r} missing or changed since u-moc')
            gates[gp] = _gate_lock(gp, r)
        gf, lock = gates[gp]
        if x.get('gate_file_sha256') != gf.sha256:              # RC1-4: every row, the cache only saves the hash
            raise LogError(f'{w}: gate_file_sha256 of the row is not that of {gp!r}')
        if gf.sha256 != r.get('gate_file_sha256'):
            raise LogError(f"{w}: gate file {gp!r} is not the one of the run (gate_file_sha256)")
        if (x.get('khoa_xy_m'), x.get('khoa_psi_deg'), x.get('khoa_sha256')) != \
                (_num(lock['U_moc_xy_m']), _num(lock['U_moc_psi_deg']), lock['khoa_sha256']):
            raise LogError(f'{w}: lock numbers differ from the gate file')
        info = read_u_moc(x['u_file'], r, ds, key[1])
        want = ('true' if info['ok'] else 'false', _num(info['xy']) if info['ok'] else '',
                _num(info['psi']) if info['ok'] else '')
        if (x.get('ok'), x.get('U_moc_xy_m'), x.get('U_moc_psi_deg')) != want:
            raise LogError(f'{w}: ok / U numbers differ from the U file')
        uinfo[key] = (info, lock)
    tinfo = {}
    for x in trows:
        s = x.get('session_id')
        w = f'{TIEN_DE}: session {s}'
        if s in tinfo:
            raise LogError(f'{w}: two rows')
        if not any(_moc_run(r) and r.get('session_id') == s for r in tun):
            raise LogError(f'{w}: no car run of a campaign in the tuning log')
        if not _same_file(x.get('file', ''), x.get('sha256')):
            raise LogError(f"{w}: premise file {x.get('file')!r} missing or changed since tien-de")
        info = read_tien_de(x['file'], s)
        if (x.get('ngay'), x.get('ket_qua')) != (info['ngay'], info['ket_qua']):
            raise LogError(f'{w}: ngay / ket_qua differ from the premise file')
        tinfo[s] = info

    def frames_l(r: dict, lan: str) -> Optional[str]:
        """frames_L of run r, None when it cannot be resolved (the source gio from the U row of that scoring)."""
        u = uinfo.get((r.get('run_uid'), lan))
        if u is not None:
            return u[0]['frames']
        try:
            return frames_of(scoring_of(r, ds, lan), None, f"run {r.get('run_uid')} {lan}")
        except LogError:
            return None

    scope_ok = {}                                      # RM2-2, RM3-1: the session's runs taken from the log
    for s, info in tinfo.items():
        if info['ket_qua'] == 'dat':
            runs = [r for r in tun if _moc_run(r) and r.get('session_id') == s and r.get('gate_json_sha256')]
            scope_ok[s] = all(frames_l(r, 'H0') in info['frames'] for r in runs)
    out = []
    for r in tun:
        if not _moc_run(r) or r.get('run_outcome') not in MOC_DK:
            out.append(r)
            continue
        uid = r.get('run_uid')
        so_row = _so_h_row(ds, uid)
        lans = ('H0', 'H1') if h1_needed(so_row, r.get('run_outcome')) else ('H0',)
        fl = set()
        if so_row is None:                            # RC1-2: the scorings needed are not known before so-h
            fl.add('cho_so_h')
        fr = {lan: frames_l(r, lan) for lan in lans}
        if any(v is None for v in fr.values()):
            fl.add('khong_gan_duoc_lan_cham')
        td = tinfo.get(r.get('session_id'))
        if td is None:
            fl.add('cho_tien_de')
        elif td['ket_qua'] != 'dat':
            fl.add('tien_de_khong_dat' if td['ket_qua'] == 'khong_dat' else 'tien_de_thieu')
        else:
            if td['ngay'] != r.get('date'):
                fl.add('tien_de_khac_ngay')
            if not scope_ok.get(r.get('session_id')):
                fl.add('tien_de_pham_vi_thieu')
            # RM3-2; S6-10 TD3-1, RC1-3: H1 covered only by a record of the H1 scoring (lan_cham H1, h1_ho_tro true)
            if 'H1' in lans and fr.get('H1') not in td['frames_h1']:
                fl.add('tien_de_khong_phu_h1')
            ids_l = [td['frames'].get(fr['H0'])] + ([td['frames_h1'].get(fr['H1'])] if 'H1' in lans else [])
            if any(i in td['thieu'] for i in ids_l if i is not None):                              # RM4-1
                fl.add('tien_de_luot_thieu')
        xs, ps = [], []
        for lan in lans:
            u = uinfo.get((uid, lan))
            if u is None:
                fl.add('cho_u_moc')
                continue
            info, lock = u
            if not info['ok']:
                fl.add('u_moc_khong_tinh')
                continue
            if (info['bo_lay_mau'], info['car_pose'], info['tap_moc']) != \
                    (lock['bo_lay_mau'], lock['car_pose'], lock['tap_moc']):
                fl.add('u_moc_nguon_khac_khoa')
            if td is not None and td['ket_qua'] == 'dat' and \
                    (info['u_so'] != td['u_so'] or info['session'] not in td['sessions']):
                fl.add('u_moc_khac_buoi')
            xs.append((info['xy'], lock['U_moc_xy_m']))
            ps.append((info['psi'], lock['U_moc_psi_deg']))
        if any(v > k for v, k in xs) or any(v > k for v, k in ps):
            fl.add('u_moc_vuot')
        flags = [f for f in MOC_FLAGS if f in fl]
        um = dict(moc_flags=','.join(flags), U_moc_xy_m=_num(max(v for v, _ in xs)) if xs else '',
                  U_moc_psi_deg=_num(max(v for v, _ in ps)) if ps else '')       # maxima over the scorings (design 6)
        r = dict(r, run_outcome='chua_ket_luan', **um) if flags else dict(r, **um)
        out.append(r)
    return out


def apply_after(tun: List[dict], ds: str) -> List[dict]:
    """The outcomes used by summary, freeze and s05_final_table: clip H comparison (so-h), then C8b v3 §5."""
    return apply_moc(apply_so_h(tun, ds), ds)


def moc_times(ds: str, uids: Iterable[str], sessions: Iterable[str]) -> List[dt.datetime]:
    """The ghi_luc of the U_MOC rows of `uids` and TIEN_DE rows of `sessions` (freeze: frozen_at not earlier)."""
    u, s = set(uids), set(sessions)
    rows = [x for x in read_rows(os.path.join(ds, U_MOC)) if x.get('run_uid') in u] + \
        [x for x in read_rows(os.path.join(ds, TIEN_DE)) if x.get('session_id') in s]
    return [_parse_t(x['ghi_luc']) for x in rows]


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


def final_boot_problem(row, ds, queries):
    from s05_final_table import run_evidence_problem
    rd = _results_dir(row, ds)
    name = 'esp32.csv'
    if rd and row.get('platform') == 'HW' and not row.get('esp32_log') and not os.path.exists(os.path.join(rd, name)):
        # Floor logging is separate. A single named CSV/events pair identifies the capture.
        names = [f for f in os.listdir(rd) if f.lower().endswith('.csv') and
                 os.path.isfile(os.path.join(rd, f + '.events.txt'))]
        if len(names) > 1:
            return 'multiple ESP32 captures: give --esp32-log', None
        if names:
            name = names[0]
    why = run_evidence_problem(row, queries, 'gain_ack.txt', name, None, rd, boot_only=True)
    return why, row.get('esp32_log') or name


def check_trial_gain(row, ds, fill=False):
    """Compare logged gains as firmware floats. At end, fill omitted gains from the SHA-verified file.
    Pre-esp32_log rows could store just the tuned subset. Complete those in memory before counting."""
    if not (row.get('campaign') in TUNED and row.get('stage') in (*TUNE_STAGES, 'hil_check')
            and row.get('platform') in ('HW', 'HIL') and row.get('run_outcome') in VALID):
        return
    import gain_line_check as glc
    from s05_final_table import run_gain_problem

    uid = row.get('run_uid')
    p, sha = row.get('trial_gain_file'), row.get('trial_gain_sha256')
    if not p or not sha:
        raise LogError(f'{uid}: judged tune run needs --gain-file (trial_gain_file / trial_gain_sha256; S05 6.3 L90)')
    try:
        _, want, _ = read_gain(p, row.get('controller'), ds, sha)
        for name in GAINS:
            col = 'g_' + name
            bits = want['gains'].get(glc.ALIASES.get(name, name))
            value = row.get(col)
            if value not in (None, ''):
                if glc.f32_bits(value) != bits:
                    raise LogError(f'{uid}: {col}={value!r} differs from {p!r} as float32')
            elif bits is not None:
                if not fill and row.get('esp32_log'):
                    raise LogError(f'{uid}: missing {col} from trial gain file {p!r}')
                row[col] = repr(struct.unpack('<f', bits)[0])
    except (OSError, ValueError) as exc:
        raise LogError(f'{uid}: {exc}') from exc
    rd = _results_dir(row, ds)
    if not rd:
        raise LogError(f'{uid}: results_dir missing, cannot verify #gain against the trial gain file')
    why = run_gain_problem(esp32_path(row, rd), want, expected_boot=row.get('boot_before'),
                           exact_first_seq=True)
    if why:
        raise LogError(f'{uid}: {why} (S05 6.3 L90): hong_he_do if gain evidence is missing or differs')
    if fill:
        row['esp32_log'] = row.get('esp32_log') or 'esp32.csv'


def end(ds: str, outcome: str, completed: Optional[int] = None, e_perp_max: Optional[float] = None,
        fail_cause: str = '', results_dir: str = '', notes: str = '', gate: Optional[dict] = None,
        gate_sha256: str = '', boot_before: Optional[str] = None, boot_after: Optional[str] = None,
        query_problem: Optional[str] = None, queries: Optional[tuple] = None, gate_file=None,
        c8b_gio: Optional[str] = None, gate_khong_nut: bool = False, gate_path: str = '', gain_file: str = '',
        esp32_log: str = '') -> dict:
    """gate_khong_nut (F11): the scorer file (gate_sha256) is a c8b_luoi file of a run where no node could be scored
    (read_luoi_khong_nut): only chua_ket_luan on the floor, logged with fail cause KHONG_NUT. GC6-12: end reads that
    file itself (gate_path, its sha256 = gate_sha256) and needs the time file c8b_gio (under results_dir, c8b_gio_v1 of
    the run's task) whose sha256 is the gio_sha256 the c8b_luoi file names; that sha256 goes to c8b_gio_sha256.
    query_problem (R134C3-3): why the two query answer files given on the command line are not two
    questions before and after the run (gain_line_check.query_problem); handled as missing / different boot.
    queries (R134C4-1): the two answers as read by gain_line_check.read_query (before, after; None = missing);
    end checks the two replies and takes boot_before / boot_after from them. A judged final_hil
    or final car run also checks the ACK and the starts in the run's ESP32 CSV and events.
    A judged car / final_hil run needs them: two boot strings alone do not show two questions.
    gate_file (O1 v11.33): the C8 gate file (gate_c8.GateC8, CLI --gate-file); the run's so_do set is taken from it
    (row_so_do), and a gate scored with a gate file needs it (gate_file_problem)."""
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
    evidence_row = dict(run['row'], results_dir=results_dir, esp32_log=esp32_log)
    if esp32_log:
        esp32_path(evidence_row, _results_dir(evidence_row, ds, existing=False))
    final_boot = stage in FINAL_PLATFORMS and plat == FINAL_PLATFORMS[stage]
    if not bbad and final_boot and fail_cause == BOOT_FAIL and queries:
        bbad, _ = final_boot_problem(evidence_row, ds, queries)
    if bbad and not (outcome == 'hong_he_do' and fail_cause == BOOT_FAIL):   # R134 (b)
        raise LogError(f'{bbad}: a run failed by operation, outcome hong_he_do with fail cause {BOOT_FAIL} '
                       '(S05 4.7 (iv), R3V-9)')
    if fail_cause == BOOT_FAIL and not bbad:
        raise LogError(f'fail cause {BOOT_FAIL} needs the two boot answers showing it (missing or different)')
    if (fail_cause == C8B_GIO_FAIL) != bool(c8b_gio) and not gate_khong_nut:      # O1 v11.37 D1, S05 v3.43 D3
        raise LogError(f'fail cause {C8B_GIO_FAIL} and --c8b-gio go together (the C8b time file of the run, ok false)')
    c8b_sha = ''
    if c8b_gio and not gate_khong_nut:
        if plat != 'HW' or outcome != 'hong_he_do':
            raise LogError(f'fail cause {C8B_GIO_FAIL}: a floor run lost to the measurement (hong_he_do), not '
                           f'{plat} {outcome}')
        if not results_dir or not os.path.isdir(results_dir) or not _inside(c8b_gio, results_dir):   # GT1-8
            raise LogError(f'--c8b-gio {c8b_gio!r}: the C8b time file of this run, under its --results-dir folder '
                           f'({results_dir!r}; GT1-8)')
        c8b_sha = read_c8b_gio(c8b_gio, task=run['row'].get('trajectory'), results_dir=results_dir)
    if gate_khong_nut:                                                                       # F11 (KG-2)
        if gate is not None or not gate_sha256 or plat != 'HW':
            raise LogError('a c8b_luoi file without a scored node is the scorer file (gate_v3 null) of a floor run')
        if outcome != 'chua_ket_luan':
            raise LogError(f'outcome {outcome!r} with a c8b_luoi file where no node could be scored (trang_thai_luot '
                           f'{KHONG_NUT}): the run is chua_ket_luan, not lost and not judged otherwise (F11)')
        if fail_cause:
            raise LogError(f'--fail-cause {fail_cause!r}: a run without a scored node gets fail cause {KHONG_NUT} itself')
        if not results_dir:
            raise LogError('--gate-json needs --results-dir (so-h looks for the H0 scorer file there, C9D-1)')
        if not c8b_gio or not os.path.isdir(results_dir) or not _inside(c8b_gio, results_dir):          # GC6-12
            raise LogError('a c8b_luoi file without a scored node needs the time file of the run (--c8b-gio, under '
                           '--results-dir): the file it was made from (GC6-12)')
        # GC8-1 (round 3): the H0 file is where the summary looks for it (under results_dir), checked before writing
        if not gate_path or not _inside(gate_path, results_dir):
            raise LogError('a run without a scored node: end reads its c8b_luoi file (--gate-json) under --results-dir, '
                           'where the summary reads it again (C9D-1, GC6-12, GC8-1)')
        lu, lu_sha = read_luoi_sha(gate_path)              # GC6-12 (round 3): hashed and parsed from one read
        if lu is None or lu_sha != gate_sha256:
            raise LogError('a run without a scored node: end reads its c8b_luoi file (gate_path, sha256 = gate_sha256, '
                           f'trang_thai_luot {KHONG_NUT}; GC6-12)')
        tg, c8b_sha = read_json_sha(c8b_gio, '--c8b-gio')
        khong_nut_sources(lu, lu_sha, tg, c8b_sha, run['row'].get('trajectory'), f'--c8b-gio {c8b_gio!r}')
        fail_cause = KHONG_NUT
    elif fail_cause == KHONG_NUT:
        raise LogError(f'fail cause {KHONG_NUT} only with its c8b_luoi file (F11)')
    if fail_cause == MAP_REFUSED and (gate is None or run['row'].get('platform') != 'HIL'):
        raise LogError(f'--fail-cause {MAP_REFUSED}: a HIL run with its scorer output (--gate-json) showing the '
                       'refused mapping (S05 v3.15 4.5 step 2)')
    if gate is None and gate_sha256 and not gate_khong_nut:    # CC1-2: the scorer file of a lost run, no gate
        if outcome != 'hong_he_do':
            raise LogError(f'a scorer file without its gate (gate_v3 null) only for a run lost to the measurement, not '
                           f'{outcome}')
        if not results_dir:
            raise LogError('--gate-json needs --results-dir (so-h looks for the H0 scorer file there, C9D-1)')
    if gate is not None:
        if not results_dir:
            raise LogError('--gate-json needs --results-dir (the run must be scorable again, GV3-6)')
        gbad = gate_file_problem(run['row'], gate.get('gate_file_sha256') or '', gate_file)
        if gbad:
            raise LogError(f'--gate-json: {gbad}')
        check_outcome_vs_gate(outcome, gate, run['row'].get('platform'), fail_cause,
                              row_so_do(run['row'], gate_file))
    if outcome == 'hong_he_do' and not fail_cause:
        raise LogError('hong_he_do needs --fail-cause (S05 §6.1)')
    if run['row'].get('platform') == 'HIL' and outcome in VALID and gate is None:
        raise LogError(f'a HIL run judged {outcome} needs its scorer output (--gate-json): S05 v3.15 4.5 step 2 '
                       '(a run without the run_active edge mapping is a lost run)')        # Codex S315-1
    if plat == 'HW' and outcome in VALID and gate is None and not gate_khong_nut:            # B8-1 B81-1
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
    if esp32_log:
        row['esp32_log'] = esp32_log
    if completed is not None:
        row['completed'] = str(completed)
    if e_perp_max is not None:
        row['e_perp_max_m'] = f'{e_perp_max:.4f}'
    if gate is not None:
        row.update(gate_columns(gate, gate_sha256))
    elif gate_sha256:
        row['gate_json_sha256'] = gate_sha256           # CC1-2
        row.update(lost_gate_file_cols(run['row'], gate_file))                  # GT1-6
    if asked:
        row.update(boot_before=boot_before or '', boot_after=boot_after or '')
    if queries is not None:                            # R134 (1): the files the two answers came from
        for (cf, cs), q in zip((QUERY_FILE_COLS[:2], QUERY_FILE_COLS[2:]), queries):
            p = (q or {}).get('_path')
            if p and os.path.isfile(p):
                row.update({cf: os.path.abspath(p), cs: sha256_file(p)})
    if final_boot and outcome in VALID:
        pair = check_query_rows([row], ds)[row['run_uid']]
        why, name = final_boot_problem(row, ds, pair)
        if why:
            raise LogError(f'{why}: a run failed by operation, outcome hong_he_do with fail cause {BOOT_FAIL}')
        row['esp32_log'] = name
    if notes:
        row['notes'] = (row.get('notes', '') + ' ' + notes).strip()
    if c8b_sha:
        row['c8b_gio_sha256'] = c8b_sha
    if gain_file:
        try:
            row.update(trial_gain_file=os.path.abspath(gain_file), trial_gain_sha256=sha256_file(gain_file))
        except OSError as exc:
            raise LogError(f'--gain-file {gain_file!r}: {exc}') from exc
    check_account_row(row, TUNING)
    if stage in ('final', 'final_hil'):
        check_final_freeze(row, ds=ds)
    # Codex S315-5: the log as it will be, this row included, checked before the row is written
    check_trial_gain(row, ds, fill=True)
    after = live_rows(read_rows(os.path.join(ds, TUNING))) + [row]
    if stage in ('final', 'final_hil'):
        # Earlier clip-H comparisons already known at end can free a slot or reveal a stop.
        check_final_limits(apply_so_h(after[:-1], ds) + [row])
    check_map_rows([row])
    check_c8b_rows(after)                              # GT1-8: the time file of no other run
    if any(r.get('fail_cause') == KHONG_NUT for r in after):
        check_khong_nut_rows(after, ds)                # GC9-10: the rule of the summary on the log as it will be
    if _c8b_counted(row):
        try:
            c8b_refusals(after)
        except LogError as exc:
            raise LogError(f'the run is not logged: {exc}')
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
    if row.get('fail_cause') == C8B_GIO_FAIL:
        if not _c8b_counted(row):
            return 'kiem gio C8b truot o kiem cuoi: luot hong do he do theo S05 4.6 (thay luot, 3 hong lien thi dung)'
        try:
            n = c8b_refusals(live_rows(read_rows(os.path.join(ds, TUNING)))).get(map_key(row), 0)
        except LogError as exc:
            return f'canh bao: dong da ghi, nhung khong dem lai duoc ({exc})'
        return f'kiem gio C8b truot lan {n}: ' + (
            f'qua {C8B_RERUNS_MAX} lan chay lai, muc nay cua bo gain chua ket luan, ghi o san_gio_c8b_dung cua summary'
            if n > C8B_RERUNS_MAX else f'chay lai cung gain (con {C8B_RERUNS_MAX + 1 - n} lan)')
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
        eff = {'ngay': t0.date().isoformat(), 'bat_dau': t0.time().isoformat(), 'ket_thuc': t.time().isoformat(),
               'gio': f'{active / 3600:.4f}', 'chien_dich': row['campaign'],
               'ghi_chu': f"rq3_log cancel truoc buoc run: {row['run_uid']} ({row['trajectory']} {row['speed_mps']}); "
                          f"gio = tong cac buoc da ghi"}
        check_account_row(eff, EFFORT)
        append_row(os.path.join(ds, EFFORT), EFFORT_COLS, eff)
        run['effort_row'] = eff
    os.remove(_open_path(ds))
    return run


# ── rows not tied to one live run ─────────────────────────────────────────
def check_machine_rows(rows):
    seen = set()
    kinds = {}
    for r in rows:
        sid = r.get('script_run_id')
        if not sid:
            continue
        c, ctrl, kind = r.get('campaign'), r.get('controller'), r.get('script_kind') or ''
        accounts = {'': CAMPAIGNS, 'S03': ('B-chung', 'RQ2'), 'S3': ('RQ2',)}
        if (not ctrl or c not in accounts.get(kind, ()) or
                (kind and (r.get('platform'), r.get('stage')) != ('SIM', 'script'))):
            raise LogError(f'{sid}: invalid controller/account for script {kind!r} ({ctrl!r}, {c!r})')
        if _f(r.get('n_sim_diagnostic')) > _f(r.get('n_sim_evals')):
            raise LogError(f'{sid}: diagnostic count exceeds actual count')
        key = sid, ctrl, c
        if key in seen:
            raise LogError(f'{sid}: duplicate script row for {ctrl}, {c}')
        if kinds.setdefault(sid, kind) != kind:
            raise LogError(f'{sid}: inconsistent script_kind')
        seen.add(key)


def _machine_row(it, campaign, controller, script_run_id, results_dir, n_sim_evals, t_machine_s, wall_start,
                 wall_end, platform, stage, session, notes, script_kind, n_sim_diagnostic, active):
    if campaign not in CAMPAIGNS:
        raise LogError(f'--campaign {campaign!r}: one of {CAMPAIGNS}')
    if stage not in STAGES or platform not in PLATFORMS:
        raise LogError(f'--stage one of {STAGES}, --platform one of {PLATFORMS}')
    if not script_run_id:
        raise LogError('--script-run-id is required')
    row = {'session_id': session, 'iter_no': str(it), 'run_uid': f'{session}:{it}', 'date': wall_start[:10],
           'platform': platform, 'controller': controller, 'campaign': campaign, 'stage': stage,
           'script_run_id': script_run_id, 'results_dir': results_dir, 'n_sim_evals': str(n_sim_evals),
           'script_kind': script_kind, 'n_sim_diagnostic': str(n_sim_diagnostic),
           't_machine_s': str(t_machine_s), 'wall_start': wall_start, 'wall_end': wall_end, 'notes': notes}
    for k in STEPS:
        field = f't_{k}_s'
        if active.get(field) is not None:
            row[field] = str(active[field])
    check_account_row(row, TUNING)
    return row


def _phase_script_result(ds, results_dir):
    folder = _results_dir({'results_dir': results_dir}, ds)
    if not folder:
        return False
    path = os.path.join(folder, 'meta.yaml')
    if not os.path.isfile(path):
        return False
    try:
        import yaml
    except ImportError as exc:
        raise LogError(f'{path}: reading script metadata requires PyYAML') from exc
    try:
        with open(path, encoding='utf-8-sig') as f:
            meta = yaml.safe_load(f)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LogError(f'{path}: cannot read script metadata ({exc})') from exc
    if not isinstance(meta, dict):
        raise LogError(f'{path}: expected a metadata mapping')
    spec = str(meta.get('spec', '')).split()
    return meta.get('script_kind') in ('S03', 'S3') or bool(spec and spec[0] in ('S03', 'S3'))


def machine(ds: str, campaign: str, controller: str, script_run_id: str, results_dir: str, n_sim_evals: int,
            t_machine_s: float, wall_start: str, wall_end: str, platform: str = 'SIM', stage: str = 'script',
            session: str = 'MACHINE', notes: str = '', script_kind: str = '', n_sim_diagnostic: int = 0,
            **active) -> dict:
    """Record one already allocated row of a script run."""
    if script_kind in ('S03', 'S3') or _phase_script_result(ds, results_dir):
        raise LogError('S03/S3: use machine-phases to allocate each controller/account from cited phases')
    row = _machine_row(next_iter_no(ds, session), campaign, controller, script_run_id, results_dir, n_sim_evals,
                       t_machine_s, wall_start, wall_end, platform, stage, session, notes, script_kind,
                       n_sim_diagnostic, active)
    path = os.path.join(ds, TUNING)
    check_machine_rows(live_rows(read_rows(path)) + [row])
    append_row(path, TUNING_COLS, row)
    return row


def _phase_source(ds, text, what, script_run_id):
    """Comparison key for a citation, with relative paths based at the dataset, then the working folder."""
    s = text.strip() if isinstance(text, str) else ''
    # Accept a path (including a Windows drive) and a line/range or clock time.
    locator = r'(?:0*[1-9]\d*(?:-0*[1-9]\d*)?|(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?)'
    match = re.fullmatch(r'((?:[A-Za-z]:)?[^:*?"<>|\r\n]+)[:@](' + locator + ')', s)
    if not match or re.split(r'[/\\]', match[1])[-1].strip() in ('', '.', '..'):
        raise LogError(f'{script_run_id}: source of the phase {what}: use FILE:LINE, FILE:FIRST-LAST '
                       'or FILE@HH:MM:SS on one line')
    path, position = match.groups()
    dataset_path = os.path.join(ds, path)
    if os.path.isfile(dataset_path) or not os.path.isfile(path):
        path = dataset_path
    if ':' in position:
        if '.' in position:
            position = position.rstrip('0').rstrip('.')
    else:
        first, _, last = position.partition('-')
        first, last = int(first), int(last or first)
        if last < first:
            raise LogError(f'{script_run_id}: source of the phase {what}: reversed line range {match[2]}')
        position = (first, last)
    return os.path.normcase(os.path.realpath(path)), position


def _logged_phase_sources(ds, rows):
    """Read individual citations, since a machine row may aggregate several phase intervals."""
    sources = set()
    for r in rows:
        if r.get('script_kind') not in ('S03', 'S3'):
            continue
        marker = f"{MANUAL} nguon pha {r.get('campaign')}: "
        _, found, citations = (r.get('notes') or '').rpartition(marker)
        if found:
            for citation in citations.split(' | '):
                m = re.fullmatch(r'(.+?)\.\.(.+?) PID \d+/\d+ LADRC \d+/\d+ \(gio: (.+); so: (.+)\)', citation)
                if not m:
                    raise LogError(f"{r.get('run_uid')}: cannot read logged phase source {citation!r}")
                sources.add((_parse_t(m[1]), _parse_t(m[2]), _phase_source(ds, m[3], 'start/end', r['run_uid']),
                             _phase_source(ds, m[4], 'counts', r['run_uid'])))
    return sources


def machine_phases(ds, script_kind, script_run_id, results_dir, phases, notes=''):
    """Allocate each elapsed phase by actual counts, then write one row per controller and account.

    A phase is (ACCOUNT, START, END, N_PID, N_LADRC, DIAG_PID, DIAG_LADRC, SOURCE_TIMES, SOURCE_COUNTS). The
    numbers are typed by hand until S03 section 7 names a notes.json field for them (X3-1, CODE 26/11): each
    phase carries the source of its START/END (file, line of the run log) and of its four counts (file, line of
    sealed/notes.json); both are required and written, with the typed numbers, in the notes of every row the
    phase goes into, after the [nhap_tay] flag."""
    totals = {}
    cited = {}
    intervals = set()
    path = os.path.join(ds, TUNING)
    existing = read_rows(path)
    if any(r.get('script_run_id') == script_run_id for r in existing):
        raise LogError(f'{script_run_id}: duplicate script import')
    sources = _logged_phase_sources(ds, existing)
    for ph in phases:
        if len(ph) != 9:
            raise LogError(f'{script_run_id}: a phase needs 9 fields (ACCOUNT START END N_PID N_LADRC DIAG_PID '
                           f'DIAG_LADRC SOURCE_TIMES SOURCE_COUNTS), got {len(ph)}')
        account, start, end_, pid, ladrc, pid_diag, ladrc_diag, src_t, src_n = ph
        time_ref = _phase_source(ds, src_t, 'start/end', script_run_id)
        count_ref = _phase_source(ds, src_n, 'counts', script_run_id)
        try:
            t0, t1 = _parse_t(start), _parse_t(end_)
            elapsed = (t1 - t0).total_seconds()
        except (ValueError, TypeError) as exc:
            raise LogError(f'{script_run_id}: invalid phase interval {start!r} to {end_!r}') from exc
        interval = (account, t0, t1)
        source = (t0, t1, time_ref, count_ref)
        if interval in intervals or source in sources:
            raise LogError(f'{script_run_id}: duplicate phase {account} {start}..{end_}')
        intervals.add(interval)
        sources.add(source)
        _nonnegative(elapsed, 'phase elapsed seconds')
        for v in (pid, ladrc, pid_diag, ladrc_diag):
            _nonnegative(v, 'phase count', count=True)
        try:
            pid, ladrc, pid_diag, ladrc_diag = map(int, (pid, ladrc, pid_diag, ladrc_diag))
        except (ValueError, TypeError) as exc:
            raise LogError('phase counts are required') from exc
        for ctrl, n, diag in (('PID', pid, pid_diag), ('LADRC', ladrc, ladrc_diag)):
            if diag > n:
                raise LogError('phase diagnostic count exceeds actual count')
            total = totals.setdefault((account, ctrl), [0, 0, 0.0, t0, t1])
            total[0] += n
            total[1] += diag
            total[2] += elapsed * (n / (pid + ladrc) if pid + ladrc else 0.5)
            total[3], total[4] = min(total[3], t0), max(total[4], t1)
        cited.setdefault(account, []).append(
            f'{start}..{end_} PID {pid}/{pid_diag} LADRC {ladrc}/{ladrc_diag} '
            f'(gio: {src_t.strip()}; so: {src_n.strip()})')
    session = 'MACHINE'
    it = next_iter_no(ds, session)

    def row_notes(account):
        return (f'{notes} ' if notes else '') + f'{MANUAL} nguon pha {account}: ' + ' | '.join(cited[account])

    rows = [_machine_row(it + i, c, ctrl, script_run_id, results_dir, n, seconds, t0.isoformat(), t1.isoformat(),
                         'SIM', 'script', session, row_notes(c), script_kind, diag, {})
            for i, ((c, ctrl), (n, diag, seconds, t0, t1)) in enumerate(totals.items())]
    if not rows:
        raise LogError('at least one phase is required')
    check_machine_rows(rows)
    if os.path.exists(path) and _header(path) != TUNING_COLS:
        raise LogError(f'{TUNING}: header differs from S05 §6; run "rq3_log.py migrate" first')
    _write_all(path, TUNING_COLS, existing + rows)
    return rows


def effort(ds: str, date: str, start: str, end_: str, campaign: str = '', khoan: str = '', nhan: str = '',
           ghi_chu: str = '', **extra) -> dict:
    """One row of m7_dev_effort_log.csv from the times Liam reports (never made up)."""
    t0 = dt.datetime.fromisoformat(f'{date}T{start}')
    t1 = dt.datetime.fromisoformat(f'{date}T{end_}')
    if t1 <= t0:
        t1 += dt.timedelta(days=1)                      # past midnight
    # RS1-2 (Liam 18/11 at CODE_GATE "Sửa luôn"): 4 decimals like cancel and hoc-end, so short rows (clip H) do not
    # vanish nor depend on how the time is split into rows (0.01 h was 36 s per row)
    row = {'ngay': date, 'bat_dau': start, 'ket_thuc': end_, 'gio': f'{(t1 - t0).total_seconds() / 3600:.4f}',
           'chien_dich': campaign, 'khoan': khoan, 'nhan': nhan, 'ghi_chu': ghi_chu}
    row.update({k: v for k, v in extra.items() if v not in (None, '')})
    check_account_row(row, EFFORT)
    if nhan == 'hoc' and not row['ghi_chu'].startswith(MANUAL):
        row['ghi_chu'] = f"{MANUAL} {row['ghi_chu']}"     # EA-7: times typed, not stamped by the script
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
    check_account_row(dict(chien_dich=campaign, nhan='hoc', ghi_chu=ghi_chu), EFFORT)
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
    row = {'ngay': t0.date().isoformat(), 'bat_dau': t0.time().isoformat(), 'ket_thuc': t1.time().isoformat(),
           'gio': f'{(t1 - t0).total_seconds() / 3600:.4f}', 'chien_dich': seg['campaign'], 'nhan': 'hoc',
           'ghi_chu': (seg['ghi_chu'] + (' ' + ghi_chu if ghi_chu else '')).strip()}
    check_account_row(row, EFFORT)
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
    _nonnegative(fix_h, 'gio_sua')
    row = {'ngay': date, 'session_id': session, 'chien_dich': campaign, 'nen_tang': platform, 'thu_hong': what,
           'nguyen_nhan': cause, 'gio_sua': f'{fix_h:.2f}', 'iter_no': iter_no, 'ghi_chu': notes}
    row.update({k: v for k, v in extra.items() if v not in (None, '')})
    check_account_row(row, HWFAIL)
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
    if campaign in freeze_rows(ds, campaign):
        raise LogError(f'{campaign} already frozen')
    try:
        _, _, gain_sha = read_gain(gain_file, campaign.split('-')[-1])
    except ValueError as exc:
        raise LogError(str(exc)) from exc
    try:
        t = _freeze_time(at) if at else now()
    except (ValueError, TypeError) as exc:
        raise LogError(f'--at {at!r}: invalid frozen_at') from exc
    at = t.isoformat(timespec='microseconds')
    if stop_reason == 'khoi_xac_nhan_dat':                                         # C9D-4
        why = confirm_block_problem(apply_after(live_rows(read_rows(os.path.join(ds, TUNING))), ds), ds, campaign,
                                    gain_set, at)
        if why:
            raise LogError(f'--stop-reason khoi_xac_nhan_dat: {why} (C9D-4)')
    row = {'campaign': campaign, 'frozen_at': at, 'stop_reason': stop_reason,
           'stop_cause': stop_cause, 'gain_set_id': gain_set, 'gain_file': gain_file.replace('\\', '/'),       # same text on any OS
           'gain_sha256': gain_sha, 'notes': notes}
    append_row(os.path.join(ds, FREEZE), FREEZE_COLS, row)
    return row


def freeze_rows(ds, campaign=None, *, defer_gain_errors=False):
    rows = {}
    for r in read_rows(os.path.join(ds, FREEZE)):
        c = r.get('campaign')
        if campaign is not None and c != campaign:
            continue
        if c in rows:
            raise LogError(f'{FREEZE}: {c} is frozen twice')
        gs = r.get('gain_set_id')
        if c not in TUNED or gain_owner(gs) != c:
            raise LogError(f'{FREEZE}: gain_set_id {gs!r} does not belong to campaign {c!r}')
        try:
            _freeze_time(r['frozen_at'])
        except (KeyError, TypeError, ValueError) as exc:
            raise LogError(f'{FREEZE}: {gs}: invalid frozen_at {r.get("frozen_at")!r}') from exc
        try:
            path, want, _ = read_gain(r.get('gain_file') or '', c.split('-')[-1], ds, r.get('gain_sha256') or '')
        except ValueError as exc:
            why = f'{FREEZE}: {gs}: {exc}'
            if not defer_gain_errors:
                raise LogError(why) from exc
            rows[c] = dict(r, gain_error=why)
            continue
        rows[c] = dict(r, gain_file=path, want=want)
    return rows


def read_gain(path, controller, ds=None, expected_sha=None):
    """Resolve the logged name and decode exactly the bytes used for its SHA-256."""
    import gain_line_check as glc
    candidates = [os.path.join(ds, path), path] if ds and not os.path.isabs(path) else [path]
    problems = []
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, 'rb') as f:
                raw = f.read()
        except OSError as exc:
            problems.append(f'{p}: khong doc duoc ({exc})')
            continue
        sha = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None and sha != expected_sha:
            problems.append(f'{p}: khac gain_sha256 (SHA-256 {sha})')
            continue
        want = glc.parse_gain_file(raw, p)
        if want['ctrl'] != {'PID': 'PID', 'LADRC': 'ADRC'}.get(controller):
            raise ValueError(f"gain file {p}: ctrl {want['ctrl']}, the set is of {controller}")
        return p, want, sha
    details = '; '.join(problems) if problems else f'not found (tried {candidates})'
    raise ValueError(f'gain_file {path!r}: {details}')


def check_final_freeze(row, freezes=None, *, ds=None):
    c, uid = row.get('campaign'), row.get('run_uid')
    gs = row.get('gain_set_id')
    if row.get('stage') == 'final_hil':
        if c != 'RQ2':
            raise LogError(f'{uid}: a final_hil run is account RQ2 (§4.6)')
        c = gain_owner(gs)
    elif gs:
        check_gain_set(c, 'final', gs)
    elif c in CAMPAIGNS and c not in GAIN_OWNERS:
        raise LogError(f'{uid}: a final car run must belong to the campaign of its gain set (got {c})')
    if c in GAIN_OWNERS and row.get('controller') != c.split('-')[-1]:
        raise LogError(f"{uid}: controller {row.get('controller')!r} does not match the gain set of {c}")
    if c not in TUNED:
        return
    if freezes is None:
        freezes = freeze_rows(ds, c)
    f = freezes.get(c)
    if not f:
        raise LogError(f'{uid}: final run of {c} needs a row in {FREEZE}')
    gs = f.get('gain_set_id')
    if not gs or row.get('gain_set_id') != gs:
        raise LogError(f"{uid}: final run of {c} with gain set {row.get('gain_set_id')}, frozen is {gs}")
    try:
        after = _parse_t(row['wall_start']) > _parse_t(f['frozen_at'])
    except (KeyError, ValueError, TypeError) as exc:
        raise LogError(f'{uid}: invalid wall_start or frozen_at ({exc})') from exc
    if not after:
        raise LogError(f"{uid}: final run must start after frozen_at {f['frozen_at']}")


# ── summary (S05 §6.3) ─────────────────────────────────────────────────────
SUMMARY_ROWS = ('A-PID', 'A-LADRC', 'B-PID', 'B-LADRC', 'B-chung', 'RQ2-INIT-PID', 'RQ2-INIT-LADRC', 'RQ2',
                'C_chung', 'T_H')
SUMMARY_HEAD = ['campaign', 'T_operator_h', 'T_operator_no_hoc_h', 'T_machine_h', 'T_wall_h', 'N_SIM', 'N_HIL',
                'N_HW', 'N_hong', 'T_sua_h', 'stop_reason', 'stop_cause', 'stop_reason_check', 'cap_hit',
                'T_tune_no_hoc_h', 'N_HW_tune', 'gate_result', 'gate_sha256', 'T_wall_end_src', 'B_chung_share',
                'B_chung_after_freeze_h', 'hoc_nhap_tay_n', 'explore_T_operator_h', 'explore_N_SIM',
                'explore_N_HIL', 'explore_N_HW', 'so_h_chua_n', 'moc_cho_n', 'moc_ha_n', 'T_clip_h_h',
                'T_12b_h', 'T_operator_no_12b_h', 'N_SIM_diagnostic']
VALID = ('dat', 'khong_dat', 'chua_ket_luan')
_COUNT = {'dat': 'pass', 'khong_dat': 'fail', 'chua_ket_luan': 'inconclusive', 'hong_he_do': 'invalid'}
_SAME_CONDITION = ('stage', 'campaign', 'gain_set_id', 'platform', 'trajectory', 'speed_mps')


def read_gate_file(path: str, allow_fake: bool = False):
    """EA-9, O1 v11.33: the locked C8 gate file (tools/gate_c8.py: levels, report-only conditions per level), never
    typed again; a gate_c8.GateC8 (its sha256 is that of the file bytes). GC6-14: a FAKE content (tests, demo) only
    with allow_fake; (round 3) any FAKE mark on a content that is not wholly FAKE is refused even with allow_fake."""
    try:
        gf = gate_c8.read_gate_c8(path)
    except gate_c8.GateFileError as exc:
        raise LogError(str(exc))
    if (gf.fake or gf.fake_sign) and not (gf.fake and allow_fake):
        raise LogError(f'gate file {path!r}: a FAKE content or FAKE marks (tests, demo), not a C8 gate (GC6-14)')
    return gf


def read_gate(path: str, allow_fake: bool = False):
    """(levels, sha256) of the locked gate file (read_gate_file). The CSV with one column `level` (before O1 v11.33)
    is no longer read."""
    gf = read_gate_file(path, allow_fake)
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
        diff = [k for k, a, b in zip(_SAME_CONDITION, final_key(old), final_key(r)) if a != b]
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
            groups[final_key(r)].append(r)
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


def check_final_limits(tun):
    stopped = _final_stops(tun)
    valid = defaultdict(int)
    for r in tun:
        if r.get('stage') == 'final_hil' and r.get('campaign') != 'RQ2':
            raise LogError(f"{r.get('run_uid')}: a final_hil run is account RQ2 (§4.6)")
        if r.get('stage') in ('final', 'final_hil') and not r.get(RETRO_FLAG) and r.get('run_outcome') in VALID:
            key = final_key(r)
            valid[key] += 1
            if valid[key] > FINAL_N:
                raise LogError(f'{key[2] or key[1]} {key[0]}_{key[4]}@{key[5]}: more than {FINAL_N} valid runs (§4.7)')
    return stopped


def check_finals(tun):
    _check_replacements(tun)
    return check_final_limits(tun)


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
      * moc_cho_n, moc_ha_n (C8b v3 §5, apply_moc): car runs waiting for a file / lowered by a check; they stay in
        N_HW, N_HW_tune and the time like any chua_ket_luan run (S05 v3.44 §4.3 H4-H5);
      * T_clip_h_h and T_12b_h (§3.2): tagged floor effort in T_operator and in the hour cap.
        T_operator_no_12b_h excludes the recorded 12b hours.
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
    check_gate_rows(raw, gate_file)                 # The provisional outcomes and query boots as originally logged.
    queries = check_query_rows(raw, ds)
    check_map_rows(raw)                             # HIL scoring evidence is checked before counting candidates.
    for r in raw:
        check_trial_gain(r, ds)                     # Bind sparse historical g_* before copies, counts or post-scoring.
    tun = apply_after(raw, ds)                      # S05 v3.37 4.3 clip H comparison, then C8b v3 §5 (apply_moc)
    if gate_file is not None:                       # design 6: the U rows were checked with this gate file
        bad = sorted({x.get('run_uid') for x in read_rows(os.path.join(ds, U_MOC))
                      if x.get('gate_file_sha256') != gate_file.sha256})
        if bad:
            raise LogError(f'summary: {U_MOC} rows of runs {bad} were written with another gate file')
    moc_cho, moc_ha = defaultdict(int), defaultdict(int)     # design 6: runs waiting for a file / lowered by a check
    for r in tun:
        fl = [f for f in (r.get('moc_flags') or '').split(',') if f]
        if fl:
            (moc_cho if any(f.startswith('cho_') for f in fl) else moc_ha)[r.get('campaign', '')] += 1
    so_h_open = defaultdict(int)                    # C9D-2: judged car runs whose clip H comparison is not given
    for r in so_h_missing(raw, ds):
        so_h_open[r.get('campaign', '')] += 1
    eff = read_rows(os.path.join(ds, EFFORT))
    hw = read_rows(os.path.join(ds, HWFAIL))
    frz = freeze_rows(ds)
    by_gain = {r['gain_set_id']: r['campaign'] for r in frz.values()}

    acc = defaultdict(lambda: defaultdict(float))
    times = defaultdict(list)
    finals = defaultdict(lambda: defaultdict(int))
    final_tasks = defaultdict(lambda: defaultdict(list))      # campaign -> task -> outcomes (stage final)
    unassigned = 0

    for r in tun:
        check_account_row(r, TUNING, legacy=True)
        if r.get('platform') in ('HIL', 'HW'):
            try:
                check_task(r.get('trajectory', ''), r.get('speed_mps', ''))
            except LogError as exc:
                raise LogError(f"{r.get('run_uid')}: {exc}")
    check_machine_rows(tun)
    stopped = check_finals(tun)
    for r in raw:
        if r.get('esp32_log'):
            esp32_path(r, _results_dir(r, ds, existing=False))
        if r.get('stage') in ('final', 'final_hil') and r.get('run_uid') in queries:
            why, _ = final_boot_problem(r, ds, queries[r['run_uid']])
            if why:
                raise LogError(f"{r.get('run_uid')}: {why}")
    for r in tun:
        if r.get('stage') in ('final', 'final_hil'):
            check_final_freeze(r, frz)
    check_c8b_rows(tun)                             # O1 v11.37 D1, S05 v3.43 D3
    check_khong_nut_rows(tun, ds)                   # GC6-12, GC7-4
    c8b_rep = defaultdict(lambda: {'n': 0, 'stopped': []})
    for k, n in c8b_refusals(tun).items():
        if n:
            c8b_rep[k[0]]['n'] += n
            if n > C8B_RERUNS_MAX:
                c8b_rep[k[0]]['stopped'].append(f'{k[1]} {k[2]}@{k[3]}')
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
        diag = _f(r.get('n_sim_diagnostic'))
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
            add('B-chung', 'N_SIM_diagnostic', diag)
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
                add(tgt, 'N_SIM_diagnostic', diag, share)
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
            _, _, key, _, task, sp = final_key(r)
            o = r.get('run_outcome', '')
            if stage == 'final':
                home, cond = c, f"final_{task}@{sp}"
                final_tasks[c][task].append(o)
            else:
                home = by_gain.get(key, key)  # check_final_freeze validated the frozen ID or the initial owner.
                cond = f"final_hil_{task}@{sp}"
            f = finals[home]
            if o in VALID:
                f[f'{cond}_n_valid'] += 1
            f[f'{cond}_' + _COUNT.get(o, 'invalid')] += 1
            if final_key(r) in stopped:
                f[f'{cond}_stopped_n3'] = 1

    pending_hours = []
    for r in eff:
        check_account_row(r, EFFORT, legacy=True)
        c, k = r.get('chien_dich', ''), r.get('khoan', '')
        if c == 'chua_biet':
            pending_hours.append(_f(r.get('gio')))
            continue
        if not c and not k:
            continue                       # rows before S05 §6.2 (no account): not in any line
        h = _f(r.get('gio')) * 3600
        hoc = r.get('nhan') == 'hoc'
        t, t_end = _effort_span(r)
        floor_work = floor_effort(r.get('ghi_chu', ''))
        if c == 'B-chung':
            add('B-chung', 'T_op_s', h)
            if not hoc:
                add('B-chung', 'T_noh_s', h)
            if floor_work:
                add('B-chung', f'{floor_work}_s', h)
        targets = b_split(c) if c else [(k, 1.0)]
        for tgt, share in targets:
            add(tgt, 'T_op_s', h, share)
            if floor_work:
                add(tgt, f'{floor_work}_s', h, share)
            if hoc:
                if r.get('ghi_chu', '').startswith(MANUAL):
                    add(tgt, 'hoc_manual', 1.0)
            else:
                add(tgt, 'T_noh_s', h, share)
                start = t
                if tgt in frz and t and len(r['bat_dau']) == 5:
                    # A manual HH:MM start in the freeze minute counts as post-freeze.
                    cut = _parse_t(frz[tgt]['frozen_at'])
                    if cut.replace(second=0, microsecond=0) == t:
                        start = cut
                spans_freeze(tgt, start, t_end, f"effort row {r.get('ngay')} {r.get('bat_dau')}")
                if not frozen_before(tgt, start):
                    add(tgt, 'T_tune_s', h, share)
                elif c == 'B-chung':
                    add(tgt, 'Bc_after_s', h, share)
            if t:
                times[tgt].append(t)

    for r in hw:
        check_account_row(r, HWFAIL, legacy=True)
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
               'gate_sha256': gate_sha256 if c in TUNED or gres else '', 'T_wall_end_src': src,
               'B_chung_share': 'unsplit total' if c == 'B-chung' else '',
               'B_chung_after_freeze_h': _fmt(a.get('Bc_after_s', 0) / 3600, 3) if c in ('B-PID', 'B-LADRC') else '',
               'hoc_nhap_tay_n': str(int(a.get('hoc_manual', 0))),
               'explore_T_operator_h': _fmt(a.get('X_T_s', 0) / 3600, 3),
               'explore_N_SIM': str(int(round(a.get('X_N_SIM', 0)))), 'explore_N_HIL': str(int(a.get('X_N_HIL', 0))),
               'explore_N_HW': str(int(a.get('X_N_HW', 0))), 'so_h_chua_n': str(so_h_open.get(c, 0)),
               'moc_cho_n': str(moc_cho.get(c, 0)), 'moc_ha_n': str(moc_ha.get(c, 0)),
               'T_clip_h_h': _fmt(a.get('clip_h_s', 0) / 3600, 3),
               'T_12b_h': _fmt(a.get('12b_s', 0) / 3600, 3),
               'T_operator_no_12b_h': _fmt((a.get('T_op_s', 0) - a.get('12b_s', 0)) / 3600, 3),
               'N_SIM_diagnostic': str(int(a.get('N_SIM_diagnostic', 0)))}
        row.update({k: str(v) for k, v in sorted(finals.get(c, {}).items())})
        if c in map_rep:                            # extra columns, only for a campaign with refused mappings
            row['hil_map_refused_n'] = str(map_rep[c]['n'])
            row['hil_map_stopped'] = '; '.join(sorted(map_rep[c]['stopped']))
        if c in c8b_rep:                            # O1 v11.37: only for a campaign with C8b time check failures
            row['san_gio_c8b_n'] = str(c8b_rep[c]['n'])
            row['san_gio_c8b_dung'] = '; '.join(sorted(c8b_rep[c]['stopped']))
        out.append(row)
    if pending_hours:
        print(f'note: {len(pending_hours)} effort row(s) chua_biet, {sum(pending_hours):g} h unallocated', file=sys.stderr)
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
    gate_file (gate_c8.GateC8, O1 v11.33): a gate_v3.4 gate scored with it, the so_do conditions of `level` for a
    run of platform / stage / campaign decided by q (CODE 23/11)."""
    n = 20
    frames = {'e_perp': [0.02] * n, 'e_trk': [e_trk] * n, 'e_bao': [0.04] * n, 'e_theta_cmd_deg': [e_theta_deg] * n}
    g = _plain(o1_gate.gate(frames, completed, v_cmd_max=0.729, wz_cmd_max=1.634, u_t=0.0015 if mapped else None,
                            decide=o1_gate.DECIDE_ALL if mapped else o1_gate.DECIDE_HIL, note='FAKE'))
    if gate_file is not None:
        g = _plain(gate_c8.apply_so_do(g, gate_file.run_so_do(level, platform, stage, campaign), level,
                                       gate_file.sha256))
    return g


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
    g.update(o1_gate.evaluate_outcomes(g, g['decide'], GATE_CONDS))
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
    cd = dict(version=drift_evidence.V1, s0_session_sha256='FAKE-H0', s1_session_sha256=None if thieu else 'FAKE-H1',
              eps_H_mm=eps_h_mm, D_H_mm=d_h_mm, vuot=vuot, thieu_H1=thieu, outcome_h0=o0, outcome_h1=o1, flags=flags)
    if o1 is not None:
        cd['h1_score'] = h1
    p = os.path.join(folder, name)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(dict(h0, outcome=final, calib_drift=cd), f)
    return p


def fake_moc_lock(folder: str) -> tuple:
    """FAKE (demo and tests only): the camera.c8b block of a C8 gate file with a moc_dich lock (C8b v3 §4, CODE_C8B
    field names, FAKE numbers) and the common FAKE files of a session it names (tap_moc, u_so, camera session)."""
    os.makedirs(folder, exist_ok=True)
    common = {}
    for k, text in (('tap_moc', 'FAKE tap_moc'), ('u_so', 'FAKE u_so'), ('session', 'FAKE session H_a')):
        common[k] = os.path.join(folder, f'FAKE_{k}.json')
        with open(common[k], 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    md = dict(U_moc_xy_m=0.010, U_moc_psi_deg=0.50, phuong_phap_duyet='FAKE', tien_de_buoi='dat',
              nguon=dict(bo_lay_mau='FAKE_blm_v1', car_pose=dict(tag_offset=None, side_m=0.1445, h_m=0.20),
                         tap_moc=sha256_file(common['tap_moc'])))
    return {'c8b': {'moc_dich': md, 'khoa_sha256': 'f' * 64}}, common


def fake_frames(run_dir: str, label: str, name: str = 'frames.csv') -> str:
    """FAKE frames file of a run folder (its sha256 goes to the scorer file as frames_sha256)."""
    os.makedirs(run_dir, exist_ok=True)
    p = os.path.join(run_dir, name)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'FAKE frames {label}\n')
    return p


def fake_u_moc(run_dir: str, task: str, common: dict, lan: str = 'H0', frames: str = '', xy: float = 0.004,
               psi: float = 0.20, name: str = '') -> str:
    """FAKE (demo and tests only): a U file (u_moc_luot_v1, design 3.1) of scoring `lan` in the run folder, every
    source with its absolute path (FAKE lich and dets written beside it)."""
    frames = frames or os.path.join(run_dir, 'frames.csv')
    src = dict(frames=frames, tap_moc=common['tap_moc'], u_so=common['u_so'], session=common['session'])
    for k in ('lich', 'dets'):
        src[k] = os.path.join(run_dir, f'FAKE_{k}_{lan}.txt')
        with open(src[k], 'w', encoding='utf-8', newline='\n') as f:
            f.write(f'FAKE {k} {lan}')
    d = dict(format=U_MOC_FORMAT, task=task, ok=True, ly_do=[], thieu=[], sha256=dict(frames=sha256_file(frames)),
             nguon=[dict(vai_tro=k, duong_dan=os.path.abspath(v), sha256=sha256_file(v)) for k, v in src.items()],
             U_moc_xy_m=xy, U_moc_psi_deg=psi, bo_lay_mau='FAKE_blm_v1',
             car_pose=dict(tag_offset=None, side_m=0.1445, h_m=0.20), n_khung=0, n_to_hop=200, FAKE=True)
    p = os.path.join(run_dir, name or f'FAKE_u_moc_{lan}.json')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(d, f)
    return p


def fake_tien_de(path: str, session: str, ngay: str, records: list, common: dict, ket_qua: str = 'dat',
                 h1: bool = False, thieu=()) -> str:
    """FAKE (demo and tests only): a session premise file (tien_de_buoi_v1, S6 / S6-10 of CODE_KIEMCHEO field names);
    records: [(id, frames path[, lan_cham])]."""
    luot = {}
    for rec in records:
        luot[rec[0]] = dict(frames=sha256_file(rec[1]), **({'lan_cham': rec[2]} if len(rec) > 2 else {}))
    d = dict(format=TIEN_DE_FORMAT, session_id=session, buoi=session, ngay=ngay, ket_qua=ket_qua, ly_do=[],
             nguon=[dict(vai_tro=k, duong_dan=os.path.abspath(common[k]), sha256=sha256_file(common[k]))
                    for k in ('u_so', 'session')],
             sha256=dict(u_so=sha256_file(common['u_so']), sessions={'H_a': sha256_file(common['session'])},
                         luot=luot),
             luot=list(luot), luot_thieu_kiem=list(thieu), mau_thuan=[], tap_neo={}, h1_ho_tro=h1, FAKE=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(d, f)
    return path


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
        gate_d = gate_c8.fake_gate(tuple(levels))
        _, moc_common = fake_moc_lock(os.path.join(out_dir, 'FAKE_moc'))   # C8b v3 §5: the common FAKE files
        # (the gate keeps the v2 camera block of fake_gate; CAMPAIGN_V2_TODO: no u-moc step until the campaign part)
        gate_c8.write_gate(gate_path, gate_d)
        gf = read_gate_file(gate_path, allow_fake=True)        # the demo gate is FAKE
        gain = dict(ctrl='PID', id=1, gains=dict(kp_xy=1.0, ki_xy=0.5, kp_wz=1.0, ki_wz=0.2, pos_kp=6.0, pos_ki=3.0,
                                                 k_psi=1.0))

        def run_dir(name):
            return os.path.abspath(os.path.join(out_dir, 'luot', name))

        def judged(name, platform, outcome, g, ask=True):
            """C9D-1, C9D-6: the run's folder with its scorer file score.json (outcome, gate_v3); kwargs of end."""
            d = run_dir(name)
            extra = dict(outcome=outcome, gate_v3=g)
            if platform == 'HW':                       # C8b v3 §5: the frames of the car scoring (design 3.0)
                extra['task'] = _require_open(out_dir)['row']['trajectory']
                extra['frames_sha256'] = sha256_file(fake_frames(d, name))
            qp = fr.write_final_run(d, platform, gain, score_extra=extra)
            sp = os.path.join(d, 'score.json')
            kw = dict(results_dir=d, gate=read_gate_json(sp), gate_sha256=sha256_file(sp), gate_file=gf,
                      gain_file=gfile)
            if ask:
                kw['queries'] = tuple(glc.read_query(p) for p in qp)
            return kw

        def tick(s):
            clock[0] += dt.timedelta(seconds=s)

        gfile = fr.write_gain_file(os.path.join(out_dir, 'FAKE_gain_G00.json'), gain)
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
        # C8b v3 §5 (CODE_GATE 18/11): the premise file of the session; the U file of each run (u-moc) is not
        # logged with a gate_c8_v2 (CAMPAIGN_V2_TODO), so the rule lowers the judged car runs (cho_u_moc)
        tien_de(out_dir, 'FAKE-A1', fake_tien_de(
            os.path.join(out_dir, 'FAKE_moc', 'tien_de_FAKE-A1.json'), 'FAKE-A1', a1[0]['date'],
            [(f'a1_{k + 1}', os.path.join(run_dir(f'a1_{k + 1}'), 'frames.csv')) for k in range(len(a1))], moc_common))
        freeze(out_dir, 'A-PID', 'khong_du_luot_xac_nhan', 'A-PID/G00', gfile, at=_iso(clock[0]),
               notes='FAKE')

        def phase(account, start, end_, pid, ladrc, pid_diag=0, ladrc_diag=0):
            return (account, (clock[0] + dt.timedelta(seconds=start)).isoformat(),
                    (clock[0] + dt.timedelta(seconds=end_)).isoformat(), pid, ladrc, pid_diag, ladrc_diag,
                    f'FAKE/console.txt@{(clock[0] + dt.timedelta(seconds=start)):%H:%M:%S}', 'FAKE/sealed/notes.json:1')

        machine_phases(out_dir, 'S03', 'FAKE-S03', 'results/S03/FAKE', [
            phase('B-chung', 0, 900, 100, 200),             # overlapping sweep: a single 15-minute interval
            phase('B-chung', 900, 901, 0, 0),               # aborted before the first run (bo)
            phase('B-chung', 901, 1501, 50, 100, 1, 2),   # replacement S0
            phase('B-chung', 1501, 3901, 90, 120, 3, 1),  # screening
            phase('B-chung', 3901, 9301, 2313, 2313),     # 257 design points per controller
            phase('RQ2', 9301, 9901, 126, 126),           # axis
            phase('RQ2', 9901, 10801, 300, 300),         # test set
        ], 'FAKE: quet, S0 bo va thay, sang loc, thiet ke, truc, tap kiem')
        machine_phases(out_dir, 'S3', 'FAKE-S3', 'results/S04/FAKE/doi_chung_s3', [
            phase('RQ2', 10801, 12601, 1000, 1000, 1, 2),
        ], 'FAKE: doi chung S3')
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
            so_h(out_dir, r['run_uid'], fake_score_final(r['results_dir'], 0.8))   # u-moc: CAMPAIGN_V2_TODO
        tien_de(out_dir, 'FAKE-F1', fake_tien_de(
            os.path.join(out_dir, 'FAKE_moc', 'tien_de_FAKE-F1.json'), 'FAKE-F1', fin[0]['date'],
            [(os.path.basename(r['results_dir']), os.path.join(r['results_dir'], 'frames.csv')) for r in fin],
            moc_common))
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
    e.add_argument('--c8b-gio', default='', help=f'C8b time file of the run ({C8B_GIO_FORMAT}, ok false) with '
                   f'--fail-cause {C8B_GIO_FAIL} (O1 v11.37)')
    e.add_argument('--gate-file-fake', action='store_true', help='accept a FAKE gate file (tests, demo only; GC6-14)')
    e.add_argument('--gate-file', default='', help='the locked C8 gate file (gate_c8_v2): so_do conditions of '
                   'the level (O1 v11.33); needed when the scorer used it')
    e.add_argument('--query-before', default='', help='gain_send.py --query output just before the run (R134 (b))')
    e.add_argument('--query-after', default='', help='gain_send.py --query output just after the run (R134 (b))')
    e.add_argument('--esp32-log', default='', help='ESP32 CSV, absolute or relative to --results-dir (default esp32.csv)')
    e.add_argument('--gain-file', default='', help='trial gain file of this tune run (S05 §6.3 L90)')
    for name in ('fail-cause', 'results-dir', 'notes'):
        e.add_argument('--' + name, default='')
    m = sub.add_parser('machine')
    for name in ('campaign', 'controller', 'script-run-id', 'results-dir', 'wall-start', 'wall-end'):
        m.add_argument('--' + name, required=True)
    m.add_argument('--n-sim-evals', type=int, required=True)
    m.add_argument('--t-machine-s', type=float, required=True)
    m.add_argument('--n-sim-diagnostic', type=int, default=0)
    m.add_argument('--script-kind', choices=('S03', 'S3'), default='', help='S03 and S3 require machine-phases')
    m.add_argument('--platform', default='SIM', choices=PLATFORMS)
    m.add_argument('--stage', default='script', choices=STAGES)
    m.add_argument('--notes', default='')
    for k in STEPS:
        m.add_argument(f'--t-{k}-s', type=float)
    mp = sub.add_parser('machine-phases', help='allocate elapsed S03 or S3 phases and record their totals')
    mp.add_argument('--script-kind', required=True, choices=('S03', 'S3'))
    mp.add_argument('--script-run-id', required=True)
    mp.add_argument('--results-dir', required=True)
    mp.add_argument('--phase', nargs=9, action='append', required=True,
                    metavar=('ACCOUNT', 'START', 'END', 'N_PID', 'N_LADRC', 'DIAG_PID', 'DIAG_LADRC', 'SOURCE_TIMES',
                             'SOURCE_COUNTS'))
    mp.add_argument('--notes', default='')
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
    o.add_argument('--gate-file-fake', action='store_true', help='accept a FAKE gate file (tests, demo only; GC6-14)')
    o.add_argument('--gate-file', default=None, help='the locked C8 gate file (gate_c8_v1 JSON, tools/gate_c8.py)')
    d = sub.add_parser('demo')
    d.add_argument('--out', required=True)
    sh = sub.add_parser('so-h', help='outcome of a car run after the clip H comparison (S05 v3.37 4.3)')
    sh.add_argument('--run-uid', required=True)
    sh.add_argument('--score-final', required=True, help='output of c7_calib_drift.py gop for that run')
    sh.add_argument('--drift', default='', help='original so JSON (v1 or v2), required to identify an old position-only gop')
    um = sub.add_parser('u-moc', help='U file of one scoring of a car run (C8b v3 §5, design section 4)')
    um.add_argument('--run-uid', required=True)
    um.add_argument('--lan', default='H0', choices=LANS)
    um.add_argument('--file', required=True, help='u_moc_luot_v1 file of CODE_CAM, under the results_dir of the run')
    um.add_argument('--gate-file', required=True, help='the C8 gate file the run was scored with')
    td = sub.add_parser('tien-de', help='premise file of a session (C8b v3 §5, design section 4)')
    td.add_argument('--session', required=True)
    td.add_argument('--file', required=True, help='tien_de_buoi_v1 file of CODE_KIEMCHEO')
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
            gj, gj_sha = read_json_sha(a.gate_json, '--gate-json') if a.gate_json else (None, '')   # GC9-11: one read
            lu = None
            if isinstance(gj, dict) and 'trang_thai_luot' in gj:                                  # F11, GC6-12
                check_luoi_dict(gj, f'--gate-json {a.gate_json!r}')
                lu = gj
            kn = lu is not None and lu['trang_thai_luot'] == KHONG_NUT
            if kn and not a.c8b_gio:
                raise LogError('a c8b_luoi file without a scored node needs --c8b-gio (the time file of the run)')
            row = end(ds, a.outcome, a.completed, a.e_perp_max, a.fail_cause, a.results_dir, a.notes,
                      gate=(None if kn else gate_of(gj, a.gate_json, allow_null=a.outcome == 'hong_he_do'))
                      if a.gate_json else None, gate_khong_nut=kn,
                      gate_sha256=gj_sha,
                      gate_path=a.gate_json if kn else '',
                      gate_file=read_gate_file(a.gate_file, a.gate_file_fake) if a.gate_file else None, c8b_gio=a.c8b_gio or None,
                      gain_file=a.gain_file, esp32_log=a.esp32_log, **boots)
            print('row', row['run_uid'], {k: row[f't_{k}_s'] for k in STEPS}, 'wait', row['t_operator_wait_s'])
            note = refusal_note(ds, row)
            if note:
                print(note)
        elif a.cmd == 'machine':
            row = machine(ds, a.campaign, a.controller, a.script_run_id, a.results_dir, a.n_sim_evals, a.t_machine_s,
                          a.wall_start, a.wall_end, a.platform, a.stage, notes=a.notes, script_kind=a.script_kind,
                          n_sim_diagnostic=a.n_sim_diagnostic,
                          **{f't_{k}_s': getattr(a, f't_{k}_s') for k in STEPS})
            print('row', row['run_uid'])
        elif a.cmd == 'machine-phases':
            for row in machine_phases(ds, a.script_kind, a.script_run_id, a.results_dir, a.phase, a.notes):
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
            gf = read_gate_file(a.gate_file, a.gate_file_fake) if a.gate_file else None
            rows = write_summary(ds, a.out or os.path.join(ds, SUMMARY), gate_file=gf)
            for r in rows:
                print(r['campaign'], r['T_operator_h'], 'h', 'N_HW', r['N_HW'], 'gate', r['gate_result'] or '-')
            n_old = count_no_gate(read_rows(os.path.join(ds, TUNING)))         # R134C2-11 (SC3-2)
            if n_old:
                print(f'{NO_GATE_EVIDENCE}: {n_old} dat row(s) outside the campaigns without gate columns '
                      '(historic; not in gate_result nor in a stop rule)')
        elif a.cmd == 'so-h':
            r = so_h(ds, a.run_uid, a.score_final, a.drift)
            print('so-h', r['run_uid'], r['outcome_tam'], '->', r['outcome_sau'], r['flags'] or '(khong co)')
        elif a.cmd == 'u-moc':
            r = u_moc(ds, a.run_uid, a.lan, a.file, a.gate_file)
            print('u-moc', r['run_uid'], r['lan'], 'ok', r['ok'], r['U_moc_xy_m'] or '-', r['U_moc_psi_deg'] or '-',
                  'khoa', r['khoa_xy_m'], r['khoa_psi_deg'])
        elif a.cmd == 'tien-de':
            r = tien_de(ds, a.session, a.file)
            print('tien-de', r['session_id'], r['ngay'], r['ket_qua'])
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
