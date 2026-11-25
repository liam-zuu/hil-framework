#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hil_gioi_han.py — E2: đếm giới hạn HIL trên log frame Pi của các lượt vòng kín, mở rộng
dem_s1.py (bản đọc-duy nhất ở 05_DATA/results/HIL_GIOI_HAN/2026-11-22_run01/, output
ket_qua_s1.csv) bằng ba phần mới: băng mù encoder THEO BÁNH đúng như H7 tính (float32, đóng
tại 0.3f), phân đoạn duty phẳng có dung sai (±0.0055), và thời gian giữ mức A11 đo trên log
ESP32 (pwm -> duty MISO) khi có log phía Pi.

(a) Băng tốc thân (như dem_s1): spd = hypot(state[:,7], state[:,8]) trên khoảng chạy k0..cuối;
    trong băng khi V_LO - EPS <= spd <= V_HI + EPS, V_LO = 0.010, V_HI = 0.0146 m/s
    (đề xuất E mục 2: dải [10; 14,6] mm/s giữa V_D = 0.01 của o1_feasibility.py:67 và
    0.3 rad/s x R 0.0485 m), EPS = 1e-9. t = (k - k0) x period, period = manifest period_us.
    Cửa sổ hoàn thành như dem_s1: task 'O1-*' -> t_cmd = o1_tasks.task(task,
    allow_final_set=True)[0], window [t_cmd; t_cmd + T_LATE), T_LATE = 2.0 s
    (o1_feasibility.py:65), win_kind 'O1_[Tcmd,deadline)'; mọi task khác -> 1.0 s cuối
    (T_D = 1.0 s, o1_feasibility.py:64), win_kind 'last1s_gia_dinh'. Decode giống hệt
    tools/hil_o1_score.py decode_frames (state int16 BE / 32767 x STATE_RANGES; duty MISO
    bytes 0-7 int16 BE / 32767) và command_start (k0 = bước RUN của lifecycle, nếu không có
    thì frame VALID khác 0 đầu tiên; không có cả hai -> ValueError -> mã 2).
    Các cột n_*_win / win_kind / win_n là số TƯƠNG THÍCH dem_s1 (C3-17): cửa sổ danh định
    [Tcmd; Tcmd + T_LATE), hay 1 s cuối khi task không phải O1 / không có --task — KHÔNG dùng
    làm bằng chứng. Số đếm trong CỬA SỔ HOÀN THÀNH của bộ chấm (cột *_ht): mặt nạ đọc từ kết
    quả tools/hil_o1_score.py score() (JSON của --score-json, hay lô --cham chạy chính bộ chấm,
    lưu cham/<run>__<ten>_score.json): frame k >= k0 của bộ chấm, t_k = (k - t0_step) x period
    (mốc của bộ chấm, có thể là cạnh run_active ESP32), T_cmd - 1e-9 <= t_k < t_deadline - 1e-9
    và t_k <= t_end + 1e-9 (t_end = lúc hoàn thành khi completed 'true', không thì frame cuối
    bộ chấm xét). JSON phải khớp log (frame_log cùng tên, frames_scored = n - k0), không thì
    ValueError -> mã 2. Không có kết quả chấm -> win_ht_src 'khong_co_ket_qua_cham', cột *_ht
    rỗng; số 1 s cuối là số mô tả riêng ở cột *_last1s.
    C3-28 (khoá kết quả chấm): JSON chấm phải có khối 'e2_khoa_cham' (LOCK_KEY) do --cham ghi:
    frame_log_sha256 = SHA-256 của chính file --frame-log, task = --task (và trường 'task' của
    bộ chấm = --task); khi đọc khoá còn đối chiếu: mỗi cờ CHAM_FLAGS (allow_final_set,
    allow_pre_b81_15) là bool bằng đúng giá trị --cham dùng; esp_log_sha256 = SHA-256 của file
    --esp-log của lần chạy này (không có log -> cả hai phía None); blob git (CRLF -> LF) của
    mọi module SCORER_MODULES (kể cả rpi5/protocol.py và tools/gate_c8.py mà hil_o1_score
    import) = blob tính lại trên bản đang chạy; k0 là chỉ số frame nguyên trong [0, n); t_end
    (khi có) nằm trong [T_cmd; t_deadline] ±1e-9. JSON thô của hil_o1_score.py --out không có
    khối này -> mã 2. Thêm: frame_log bắt buộc; t0_step, T_cmd, t_deadline phải hữu hạn;
    completed 'true' cần t_end hữu hạn. win_ht_src = 'hil_o1_score:<t0_src>:completed=<..>:
    allow_final_set=<0|1>:allow_pre_b81_15=<0|1>:scorer_blob=<blob hil_o1_score>'.

(b) Băng mù theo bánh (đề xuất v6 C3(b)): blind_band(omega32) -> (n,4) bool, a = |omega32|
    float32, (a > 0) & (a <= float32(0.3)): H7 không phát xung encoder khi
    |omega| <= ENC_OMEGA_MIN = 0.3f (stm32h7/Core/Src/encoder_gen.c:5 và :212-213, new_dir
    giữ 0 trừ khi omega > 0.3f hay < -0.3f). omega32 là giá trị H7 decode từ MOSI bytes 0-7:
    int16 BE x (40.0f / 32767.0f), float32 (tools/hil_edge_map.py h7_omega32).
    KHÁC dem_s1: dem_s1 đếm khoảng MỞ (0; 0.3) trên state float64 và chỉ theo any-wheel;
    công cụ này đếm (0; 0.3f] float32 theo từng bánh (n_wb_*_<banh>) và cột any-wheel
    (n_wband_any_*) để so sánh với n_wband_* của dem_s1 (so_khop_s1.csv).

(c) MISO trên khoảng chạy (C3-14: tách số tương thích khỏi số làm bằng chứng):
    - TƯƠNG THÍCH dem_s1 (chỉ các cột của CMP_SAME, để so_khop_s1; KHÔNG làm bằng chứng):
      n_duty0_any, n_sat_any, n_flat_any, max_flat_ms, n_all4_duty0, max_all4_0_ms từ int16
      raw (duty0 raw == 0, sat |raw| == 32767, flat = duty0 | sat), gồm cả frame MISO rỗng
      (mọi byte 0 -> decode ra duty 0) và frame sai CRC; n_miso_empty_run; max đoạn [ms] =
      int(run * period_ms).
    - Frame đủ bằng chứng (valid): protocol.miso_check == MISO_OK (không rỗng, CRC-16 bytes
      8-9 đúng; cùng luật s06_frames_decode / s06_t6_check). Khoảng mất mẫu (cut[k]: đoạn
      mới bắt đầu ở k): seq MOSI (byte OFS_SEQ = bước & 0xFF) không tăng đúng 1 so với k-1,
      hay t_send_ns[k] - t_send_ns[k-1] > GAP_FACTOR x period [ĐỀ XUẤT 1,5]. Cột:
      n_miso_crc_bad_run, n_mat_mau_run (số chỗ cắt), n_khong_bc_run (frame rỗng + sai CRC).
    - BẰNG CHỨNG: mọi số MISO còn lại chỉ đếm frame valid, đoạn liền bị cắt tại frame không
      valid và tại khoảng mất mẫu: n_d0_<bánh>, n_sat_<bánh>; *_bc (n_duty0_any_bc,
      n_sat_any_bc, n_flat_any_bc, max_flat_bc_ms, n_all4_duty0_bc, max_all4_0_bc_ms; ms =
      số frame x period_ms); tolerance TOL = 0.0055 + EPS (spec S06 v3.5 T6 dòng 120):
      |d| <= TOL + EPS -> loại 0; |d| >= 1 - TOL - EPS -> loại +1 / -1 theo dấu; còn lại
      'khác'; frame không valid là loại 3; loại 2, 3 và chỗ cắt đều cắt đoạn. Mỗi bánh: số
      frame loại 0 / loại sat, số đoạn liên tục tối đa cùng một loại trong {0,+1,-1}, median
      và max độ dài đoạn [ms]; any-wheel: n_d0_tol_any, n_sat_tol_any, n_flat_tol_any,
      max_flat_tol_ms. CSV đoạn phẳng: wheel, kind, k_start (chỉ số frame trong npz),
      len_frames, len_ms.

(c-ESP32) A11 (đề xuất v6 C5(c), E mục 2), chỉ khi có --esp-log. C3-15: số báo ra là THỜI
    GIAN TỚI LẦN ĐẦU QUAN SÁT VÀO DẢI ±TOL (t_vao_dai_ms), KHÔNG phải "theo kịp" theo nghĩa
    T6 (T6: mốc ghi compare, rời mức cũ, 100 frame liền): log cũ không có mốc ghi compare,
    mốc t0 suy từ giờ nhận UDP (moc_t0, bat_dinh_ms ở mỗi sự kiện), và một frame vào dải
    rồi ra lại vẫn là "lần đầu vào dải" (n_lien_vao_dai cho biết số frame valid liền trong
    dải từ lần đầu, trong mức lệnh). Chỉ báo, không là tiêu chí nghiệm thu:
    - read_esp_pwm: CSV agv_logger theo tên cột (cần timestamp_us, log_seq, pwm_fl..pwm_rr;
      thiếu -> ValueError -> mã 2). Trùng lặp xử lý y hệt hil_edge_map.read_esp_log: có
      <log>.segments.csv thì bỏ dòng 'duplicate' sau khi kiểm bằng một dòng đã giữ,
      'conflict'/'ambiguous' hay >1 segment -> từ chối log; không có segments.csv mà gặp dòng
      trùng -> từ chối. Sắp xếp theo log_seq; hai log_seq giống nhau -> từ chối. Từ chối ->
      esp_part 'loi: ...' (phần MISO vẫn ghi, mã 3).
    - Đồng hồ ESP32 -> Pi: cặp (t_rx, t_esp) từ <log>.raw.jsonl (agv_logger.py:879-882:
      t_mono_ns = Pi time.monotonic_ns() lúc nhận gói; trường đầu của dòng dữ liệu b64 là
      timestamp_us), khớp s06_t5_score.clock_fit(rx, te, bin_us = 1e6) như s06_t6_check.py
      (CLOCK_BIN_US = 1e6): t_pi = a_us + (1 + b) x t_esp. None -> 'loi: khong khop dong ho'.
      ĐỘ LỆCH ĐÃ BIẾT: giờ Pi suy ra TRỄ hơn thật một khoảng bằng trễ UDP nhỏ nhất (chưa đo),
      và dòng log được đẩy sau motor_set (AGV main.c 687-699), nên hold_ms đo được là CHẶN
      DƯỚI của thời gian giữ tính từ lúc ghi compare, tới chừng jitter tick-compute (chưa
      đo). Thêm bất định từ chính fit: phần dư của đường least-squares trên bin-minimum
      (s06_t5_score.clock_fit) cỡ sync_resid_max_us — trên lô thật median ~2.2 ms, max
      ~4.3 ms. sync_kind = 'udp_envelope' ghi trong dòng kết quả.
    - Kiểm dấu từng bánh: trên frame khoảng chạy có MISO không rỗng, pwm giữ P = pwm của dòng
      cuối có to_pi(ts) <= t_f (t_f = t_send_ns/1e3 µs); corr_<bánh> = Pearson(P, duty);
      bánh sign_ok khi sign(corr) == SIGN_CODE (-1: từ 07/11 HIL-PWM-WIRE-SIGN duty MISO =
      -pwm/1000, motor_dir -1, tools/tests/test_s06_t6_check.py test_golden_vectors) và
      |corr| >= MIN_CORR = 0.8 [ĐỀ XUẤT]. Bánh không sign_ok: sự kiện ±1000 của nó không
      chấm (đếm n_ev_sign_unknown); sự kiện mục tiêu 0 vẫn chấm.
    - Sự kiện 'lenh': dòng i (thứ tự log_seq) có pwm[i] in {0, ±1000} và khác pwm[i-1];
      yêu cầu log_seq[i] == log_seq[i-1] + 1, nếu không -> bỏ qua, đếm n_ev_seqgap; và khi
      có cột motor_on (tùy chọn, motor_on_col = 1/0) thì thêm motor_on[i] == 1 và
      motor_on[i-1] == 1 — dòng motor_on = 0 không bao giờ là 'lenh'. Mục tiêu
      d* = 0 hay SIGN_CODE x pwm/1000. t0 = to_pi(ts[i]); chỉ giữ t0 trong
      [t_f[k0]; t_f[-1]], ngoài -> n_ev_outside. Cuối mức t_end = to_pi(ts[j]) của dòng j kế
      có pwm[j] != pwm[i] (không có -> +inf). Tìm frame k với t_f[k] >= t0, t_f[k] < t_end,
      bỏ frame không valid (rỗng, sai CRC): frame đầu |duty[k] - d*| <= TOL + EPS -> vao_dai,
      k_vao_dai, t_vao_dai_ms = (t_f[k] - t0)/1000; không có -> censored ("> khoảng giữ",
      KHÔNG bỏ), cens_ms = (min(t_end, t_f[-1]) - t0)/1000, cat = 'doi_lenh' (lệnh đổi mức
      trước frame cuối của log) hay 'het_log' (log frame hết trước). n_khong_bc = số frame
      không valid trong khoảng đã quét [k_first; lần vào dải] (hay tới k_last) + số khoảng
      mất mẫu (t_f[k-1]; t_f[k]) giao cửa sổ quan sát [t0; t_f[k_vao_dai]] (hay [t0; t_end)),
      C3-31: kể cả khoảng vắt qua t0 (mẫu trước t0, mẫu kế sau t0) hay vắt qua t_end
      (n_gaps_in): > 0 thì lần vào dải thật có thể sớm hơn số báo (frame không quan sát
      được). duty_first = duty của frame k_first, để trống khi frame đó không valid (rỗng,
      sai CRC; C3-31). moc_t0 'udp_envelope'; bat_dinh_ms = sync_resid_max_us/1000 (± của
      đường khớp, chỉ phần dư); cột riêng lech_udp = 'chua_do' (C3-31, KHÔNG gộp vào
      bat_dinh_ms): độ lệch MỘT PHÍA chưa đo, trễ UDP nhỏ nhất làm t0 suy ra trễ hơn thật,
      nên t_vao_dai_ms có thể NGẮN hơn thật tới chừng đó (vd. mốc trễ 4 ms: đáp ứng 4 ms ghi
      thành 0 ms).
    - Sự kiện 'tat_motor' (dừng motor cuối lượt; firmware AGV src/main.c: vòng
      vTaskDelay(pdMS_TO_TICKS(10)) :387; tick đầu motor-off gọi motor_stop_all() và
      push_run_end() :641-644; dòng idle 10 Hz pwm 0 chỉ được đẩy ~10 tick sau :646-650, nên
      timestamp của dòng pwm 0 đầu tiên trễ ~100 ms so với tick dừng thật — KHÔNG dùng nó
      làm mốc lệnh): tại dòng i có motor_on[i] == 0 và motor_on[i-1] == 1, mỗi bánh có
      pwm[i-1] != 0 là một ứng viên; yêu cầu seq[i-1] bằng last_data_seq của một dòng
      '#run_end' trong <log>.events.txt (3 dòng retry cùng last_data_seq), nếu không ->
      bỏ qua, đếm n_off_unproven. Mốc tick dừng [ĐỀ XUẤT]: t0 = to_pi(ts[i-1] + dt_med),
      dt_med = median của hiệu ts các cặp dòng motor_on liên tiếp seq (bất định trong
      [dt_min - dt_med; dt_max - dt_med] — suy ra từ vòng firmware và chính log, không đo
      trên pin; cột tick_dt_min/med/max_ms). Mục tiêu d* = 0; t_end = to_pi(ts[j]) của
      dòng j >= i đầu có pwm[j] != 0 (không có -> +inf); cùng phép tìm lần đầu vào dải;
      moc_t0 'udp_envelope+tick_dt_med', bat_dinh_ms = sync_resid_max_us/1000 +
      max(dt_med - dt_min, dt_max - dt_med)/1000. Cột thống kê A11_STAT với hậu tố _off,
      n_off_unproven, n_off_outside — tách khỏi thống kê 'lenh' mục tiêu 0. Ứng viên
      có t0 ngoài [t_f[k0]; t_f[-1]] -> n_off_outside, không chấm: lượt dừng cuối run
      thường đến SAU khi frame log kết thúc (lượt HIL dừng trước, DUT tắt motor sau khi
      mất lệnh), nên thời gian giữ mức lúc dừng không quan sát được trong các log này.
    - Tách sự kiện hiển nhiên ('doi' = đổi thật) cho 'lenh': chỉ giữ sự kiện có
      |SIGN_CODE x pwm_prev/1000 - d*| > TOL + EPS (pwm_prev = pwm[i-1] của bánh đó):
      cột A11_STAT với hậu tố _doi_<T>, T in (0,1). Các cột mọi-sự-kiện (hậu tố _<T>) vẫn giữ.
      A11_STAT: n_ev, n_vao_dai, n_cens, t_vao_dai_med/p95/max_ms, cens_max_ms,
      n_cat_doi_lenh, n_cat_het_log, n_ev_mat_mau (sự kiện có n_khong_bc > 0).
    - CSV sự kiện: loai ('lenh' / 'tat_motor'), pwm_prev, moc_t0, bat_dinh_ms, lech_udp,
      k_first, duty_first, k_vao_dai, t_vao_dai_ms, vao_dai, n_lien_vao_dai, n_khong_bc, cat.
    - 9 npz của 2026-11-08_run08/run09/run11 (w1, w2, w3) không có log ESP32 phía Pi (chỉ
      esp32_COM5.txt serial không giờ Pi) -> chỉ phần MISO, esp_part 'khong_co_log_ESP32'.

Chạy một lượt:
  python tools/hil_gioi_han.py --frame-log X_frames.npz [--esp-log X.csv] [--task O1-F02]
      [--score-json score.json] --out DIR
  Ghi DIR/<ten>_gioi_han.csv (một dòng), DIR/<ten>_su_kien_a11.csv (sự kiện ESP32),
  DIR/<ten>_doan_phang.csv (đoạn phẳng MISO). <ten> = tên npz bỏ '_frames.npz' (hay '.npz').

Chạy theo lô:
  python tools/hil_gioi_han.py --batch-from-s1 ket_qua_s1.csv --root CLOSED_LOOP --out DIR [--cham]
  --cham: dòng có task 'O1-*' được chấm bằng hil_o1_score.score(npz, task,
  allow_final_set=True, esp_log=<log ESP32 nếu có>, allow_pre_b81_15=True) (log phát triển
  trước kiểm B81-15), JSON ghi DIR/cham/<run>__<ten>_score.json kèm khối 'e2_khoa_cham'
  (C3-28: SHA-256 NPZ băm trước khi chấm, task, bộ cờ, blob bộ chấm; mục a), mặt nạ *_ht lấy
  từ đó sau khi kiểm khoá (băm lại NPZ); bộ chấm lỗi -> win_ht_src 'loi_cham: ...', cột *_ht
  rỗng, lô chạy tiếp; kết quả chấm không qua kiểm khoá / mốc -> dòng lỗi (cột 'error').
  Mỗi dòng s1: frame log = root/run/frames/npz, log ESP32 = root/run/pi/<ten>.csv nếu có,
  task = cột 'task' của s1 (nhận diện của dem_s1 từ meta.yaml, đã kiểm). Ghi ket_qua_e2.csv
  (một dòng/npz, cột run, npz, name, task đứng đầu), su_kien/<run>__<ten>_su_kien_a11.csv,
  doan_phang/<run>__<ten>_doan_phang.csv, và so_khop_s1.csv (mỗi lệch một dòng: run, npz,
  cot_s1, cot_e2, gia_tri_s1, gia_tri_e2 — so sánh cùng tên cho các cột dem_s1 và cặp
  n_wband_* <-> n_wband_any_*, KHÔNG chỉnh định nghĩa để giấu lệch). Dòng lỗi ghi cột
  'error' và lô chạy tiếp.

Mã thoát / dòng cuối (#ket_qua,ma=... qua s06_stream.report_result):
  0 phần ESP32 đã tính (lô: mọi dòng tính được và không lỗi);
  3 thieu_bang_chung khi không có --esp-log / phần ESP32 'loi' (lô: có dòng thiếu phần ESP32);
  2 lỗi đầu vào (thiếu file/khoá/cột, task id sai, command_start ValueError; lô: có dòng lỗi).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import protocol                                    # noqa: E402  rpi5/: miso_check, OFS_SEQ
import hil_o1_score                                # noqa: E402
import hil_edge_map                                # noqa: E402
import s06_t5_score                                # noqa: E402
import s06_stream                                  # noqa: E402
import o1_tasks                                    # noqa: E402
from o1_feasibility import T_D, T_LATE             # noqa: E402  o1_feasibility.py:64-65

WHEELS = ('fl', 'fr', 'rl', 'rr')
V_LO, V_HI = 0.010, 0.0146      # m/s, dải tốc độ [10; 14,6] mm/s (đề xuất E mục 2, dem_s1)
EPS = 1e-9
ENC_OMEGA_MIN32 = np.float32(0.3)   # encoder_gen.c:5 ENC_OMEGA_MIN (float32 như H7)
TOL = 0.0055                        # spec S06 v3.5 T6 (dòng 120)
SAT_RAW = 32767                     # |duty| x 32767 bão hoà (dem_s1 SAT_RAW)
SIGN_CODE = -1                      # HIL-PWM-WIRE-SIGN 07/11: duty MISO = -pwm/1000
MIN_CORR = 0.8                      # [ĐỀ XUẤT] ngưỡng tương quan pwm~duty xác nhận dấu
PWM_FLAT = (0, 1000, -1000)         # mức pwm phẳng: pin đứng yên, không cạnh (pwm_capture.h:41)
CLOCK_BIN_US = 1e6                  # s06_t6_check.py CLOCK_BIN_US (bin chặn dưới UDP)
ESP_NEED = ('timestamp_us', 'log_seq') + tuple('pwm_' + w for w in WHEELS)
SEG_MARK_COLS = ['row', 'log_seq', 'timestamp_us', 'segment', 'status']   # <log>.segments.csv
PWM_COLS = tuple('pwm_' + w for w in WHEELS)
# bước t_send > GAP_FACTOR chu kỳ = khoảng mất mẫu (C3-14). QUY ƯỚC, căn cứ (đề F-E2, số đo của
# CODE trên 158 log CLOSED_LOOP, chưa kiểm lại ở đây): bước t_send lớn nhất khi không mất mẫu
# khoảng 1,07 chu kỳ, mất một mẫu khoảng 2 chu kỳ, nên mọi ngưỡng trong (1,07; 2) chu kỳ cho
# cùng kết quả trên các log đó; 1,5 nằm giữa khoảng.
GAP_FACTOR = 1.5
KIND_INVALID = 3                    # loại của frame MISO không valid trong phân đoạn dung sai
# thống kê A11 (C3-15: thời gian tới lần đầu quan sát vào dải, không phải "theo kịp" T6)
A11_STAT = ('n_ev', 'n_vao_dai', 'n_cens', 't_vao_dai_med_ms', 't_vao_dai_p95_ms',
            't_vao_dai_max_ms', 'cens_max_ms', 'n_cat_doi_lenh', 'n_cat_het_log', 'n_ev_mat_mau')
LECH_UDP = 'chua_do'                # C3-31 cột lech_udp: lệch một phía do trễ UDP nhỏ nhất, chưa đo
# C3-28: khoá kết quả chấm (khối LOCK_KEY trong JSON chấm do --cham ghi)
LOCK_KEY = 'e2_khoa_cham'
CHAM_FLAGS = dict(allow_final_set=True, allow_pre_b81_15=True)   # cờ chấm của lô --cham
SCORER_MODULES = ('hil_o1_score', 'o1_gate_eval', 'o1_gate', 'o1_feasibility', 'o1_lanes',
                  'o1_tasks', 'hil_edge_map', 'protocol', 'gate_c8')   # nguồn của bộ chấm (blob git)


def a11_cols(sfx):
    return [f'{c}_{sfx}' for c in A11_STAT]


ROW_COLS = ['n_frames', 'k0', 'k0_src', 'n_run', 'period_us', 't_last_s', 't_cmd_s',
            'win_kind', 'win_n', 'n_vband_win', 'n_vband_last1s', 'n_vband_run',
            'n_wband_any_win', 'n_wband_any_last1s', 'n_wband_any_run']
for _w in WHEELS:
    ROW_COLS += [f'n_wb_win_{_w}', f'n_wb_last1s_{_w}', f'n_wb_run_{_w}']
ROW_COLS += ['win_ht_src', 't_ht0_s', 't_ht1_s', 'win_ht_n', 'n_vband_ht', 'n_wband_any_ht']
ROW_COLS += [f'n_wb_ht_{_w}' for _w in WHEELS]
ROW_COLS += ['n_duty0_any', 'n_sat_any', 'n_flat_any', 'max_flat_ms', 'n_all4_duty0',
             'max_all4_0_ms', 'n_miso_empty_run',
             'n_miso_crc_bad_run', 'n_mat_mau_run', 'n_khong_bc_run', 'n_duty0_any_bc',
             'n_sat_any_bc', 'n_flat_any_bc', 'max_flat_bc_ms', 'n_all4_duty0_bc',
             'max_all4_0_bc_ms']
for _w in WHEELS:
    ROW_COLS += [f'n_d0_{_w}', f'n_sat_{_w}']
for _w in WHEELS:
    ROW_COLS += [f'n_d0_tol_{_w}', f'n_sat_tol_{_w}', f'n_seg_tol_{_w}', f'seg_med_ms_{_w}',
                 f'max_seg_tol_ms_{_w}']
ROW_COLS += ['n_d0_tol_any', 'n_sat_tol_any', 'n_flat_tol_any', 'max_flat_tol_ms',
             'esp_log', 'esp_part', 'sync_kind', 'sync_ppm', 'sync_resid_max_us',
             'sync_span_s']
for _w in WHEELS:
    ROW_COLS += [f'corr_{_w}', f'sign_{_w}', f'sign_ok_{_w}']
for _t in (0, 1):
    ROW_COLS += a11_cols(_t)
ROW_COLS += ['n_ev_seqgap', 'n_ev_outside', 'n_ev_sign_unknown',
             'motor_on_col', 'tick_dt_min_ms', 'tick_dt_med_ms', 'tick_dt_max_ms']
ROW_COLS += a11_cols('off') + ['n_off_unproven', 'n_off_outside']
for _t in (0, 1):
    ROW_COLS += a11_cols(f'doi_{_t}')

EVENT_COLS = ['loai', 'wheel', 'target', 'log_seq', 'pwm_prev', 't_esp_us', 't_pi_us',
              'moc_t0', 'bat_dinh_ms', 'lech_udp', 'k_first', 'duty_first', 'k_vao_dai',
              't_vao_dai_ms', 'vao_dai', 'n_lien_vao_dai', 'n_khong_bc', 'cat', 'cens_ms',
              'cmd_level_ms']
SEG_OUT_COLS = ['wheel', 'kind', 'k_start', 'len_frames', 'len_ms']

CMP_SAME = ['n_frames', 'k0', 'n_run', 'win_kind', 'win_n', 'n_vband_win', 'n_vband_last1s',
            'n_vband_run', 'n_duty0_any', 'n_sat_any', 'n_flat_any', 'max_flat_ms',
            'n_all4_duty0', 'max_all4_0_ms', 'n_miso_empty_run']
CMP_PAIRS = [('n_wband_win', 'n_wband_any_win'), ('n_wband_last1s', 'n_wband_any_last1s'),
             ('n_wband_run', 'n_wband_any_run')]


class EspRefused(Exception):
    """Log ESP32 không dùng được làm bằng chứng (conflict/ambiguous/>1 segment/dòng trùng
    không xác nhận): esp_part 'loi: ...', phần MISO vẫn ghi."""


def _int16(a, ofs, n):
    """Big-endian int16 columns [ofs, ofs+2n) of a (frames, W) uint8 array (như dem_s1)."""
    b = np.ascontiguousarray(a[:, ofs:ofs + 2 * n])
    return b.view('>i2').reshape(len(a), n)


def name_of(npz_name):
    """w1_frames.npz -> w1; X.npz -> X."""
    if npz_name.endswith('_frames.npz'):
        return npz_name[:-len('_frames.npz')]
    return npz_name[:-4] if npz_name.endswith('.npz') else npz_name


def longest_run(mask, cut=None):
    """Đoạn True liên tục dài nhất của mảng bool (như dem_s1, vector hoá). cut (tuỳ chọn,
    C3-14): cut[k] True = đoạn mới bắt đầu ở k (khoảng mất mẫu giữa k-1 và k)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    if cut is None:
        d = np.diff(m.astype(np.int8))
        starts = np.nonzero(d == 1)[0] + 1
        ends = np.nonzero(d == -1)[0] + 1
        if m[0]:
            starts = np.concatenate(([0], starts))
        if m[-1]:
            ends = np.concatenate((ends, [len(m)]))
        return int((ends - starts).max())
    c = np.asarray(cut, dtype=bool)
    st = m.copy()
    st[1:] &= ~m[:-1] | c[1:]
    en = m.copy()
    en[:-1] &= ~m[1:] | c[1:]
    return int((np.nonzero(en)[0] - np.nonzero(st)[0]).max()) + 1


def in_speed_band(spd):
    """spd trong [V_LO - EPS; V_HI + EPS] (băng tốc thân, mục (a) của docstring)."""
    return (spd >= V_LO - EPS) & (spd <= V_HI + EPS)


def blind_band(om32):
    """(n,4) bool: H7 không phát xung khi 0 < |omega32| <= 0.3f (encoder_gen.c:5,212-213)."""
    a = np.abs(np.asarray(om32, dtype=np.float32))
    return (a > np.float32(0.0)) & (a <= ENC_OMEGA_MIN32)


def duty_kind(d):
    """duty -> int8: 0 khi |d| <= TOL+EPS; +1/-1 khi |d| >= 1-TOL-EPS theo dấu; 2 = khác."""
    d = np.asarray(d, dtype=np.float64)
    ad = np.abs(d)
    kind = np.full(d.shape, 2, dtype=np.int8)
    kind[ad <= TOL + EPS] = 0
    kind[d >= 1.0 - TOL - EPS] = 1
    kind[d <= -(1.0 - TOL - EPS)] = -1
    return kind


def tol_segments(kind, cut=None):
    """Các đoạn liên tục tối đa cùng loại in {0,+1,-1} -> list (kind, k_start, len_frames);
    frame 'khác' (kind 2) và không valid (KIND_INVALID) cắt đoạn; cut[k] True (khoảng mất
    mẫu, C3-14) cũng bắt đầu đoạn mới ở k."""
    a = np.asarray(kind)
    chg = np.diff(a.astype(np.int16)) != 0
    if cut is not None:
        chg |= np.asarray(cut, dtype=bool)[1:]
    edges = np.nonzero(chg)[0] + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(a)]))
    return [(int(a[s]), int(s), int(e - s)) for s, e in zip(starts, ends)
            if a[s] not in (2, KIND_INVALID)]


def frame_validity(mosi, miso, t_send_ns, period_us):
    """C3-14 -> (valid, crc_bad, cut), mỗi mảng (n,) bool. valid = protocol.miso_check ==
    MISO_OK (không rỗng, CRC đúng); cut[k] (k >= 1) = khoảng mất mẫu giữa k-1 và k: seq MOSI
    không tăng đúng 1 (mod 256), hay bước t_send_ns > GAP_FACTOR x period (khi có t_send_ns)."""
    miso = np.asarray(miso, dtype=np.uint8)
    n = len(miso)
    chk = np.fromiter((protocol.miso_check(bytes(b)) for b in miso), dtype=np.int8, count=n)
    cut = np.zeros(n, dtype=bool)
    if n > 1:
        sq = np.asarray(mosi, dtype=np.uint8)[:, protocol.OFS_SEQ].astype(np.int16)
        cut[1:] = (np.diff(sq) % 256) != 1
        if t_send_ns is not None:
            cut[1:] |= (np.diff(np.asarray(t_send_ns, dtype=np.int64))
                        > GAP_FACTOR * period_us * 1000.0)
    return chk == protocol.MISO_OK, chk == protocol.MISO_CRC_BAD, cut


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git_blob(path):
    """SHA-1 blob git của file nguồn (nội dung đổi CRLF -> LF như git hash-object khi
    core.autocrlf=true)."""
    with open(path, 'rb') as f:
        data = f.read().replace(b'\r\n', b'\n')
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def scorer_blobs():
    """C3-28: blob git (CRLF -> LF) của mọi module SCORER_MODULES trên bản đang chạy."""
    return {m: git_blob(sys.modules[m].__file__) for m in SCORER_MODULES}


def score_lock(npz_path, task, esp_path):
    """C3-28: khối LOCK_KEY ghi vào JSON chấm của --cham: SHA-256 NPZ, task, bộ cờ chấm
    CHAM_FLAGS, log ESP32 đưa cho bộ chấm (tên, SHA-256) và blob git các nguồn SCORER_MODULES."""
    return dict(frame_log=os.path.basename(npz_path), frame_log_sha256=sha256_file(npz_path),
                task=task, **CHAM_FLAGS,
                esp_log=os.path.basename(esp_path) if esp_path else None,
                esp_log_sha256=sha256_file(esp_path) if esp_path else None,
                scorer=scorer_blobs())


def check_score_lock(score, npz_path, task, esp_path=None):
    """C3-28: kết quả chấm phải mang khối LOCK_KEY của đúng NPZ (SHA-256) và đúng task (trường
    'task' của khoá và của bộ chấm), mỗi cờ CHAM_FLAGS là bool bằng đúng giá trị --cham dùng,
    esp_log_sha256 bằng SHA-256 của file --esp-log của lần chạy này (không có log -> cả hai
    phía None) và blob git của mọi module SCORER_MODULES bằng blob tính lại trên bản đang
    chạy. Sai -> ValueError."""
    lock = score.get(LOCK_KEY)
    if not isinstance(lock, dict):
        raise ValueError(f'ket qua cham khong co khoa {LOCK_KEY} (SHA-256 NPZ + task)')
    sha = sha256_file(npz_path)
    if lock.get('frame_log_sha256') != sha:
        raise ValueError(f"ket qua cham cua NPZ sha256={lock.get('frame_log_sha256')!r}, "
                         f'frame log co sha256={sha}')
    if lock.get('task') != task or score.get('task') != task:
        raise ValueError(f"ket qua cham cua task {lock.get('task')!r}/{score.get('task')!r}, "
                         f'khong phai {task!r}')
    for c, v in CHAM_FLAGS.items():
        x = lock.get(c)
        if not (isinstance(x, bool) and x == v):
            raise ValueError(f'khoa cham co {c}={x!r}, lo --cham dung {v!r}')
    want = sha256_file(esp_path) if esp_path else None
    if 'esp_log_sha256' not in lock or lock['esp_log_sha256'] != want:
        raise ValueError(f"khoa cham esp_log_sha256={lock.get('esp_log_sha256')!r}, "
                         f'--esp-log cua lan nay co sha256={want!r}')
    sc = lock.get('scorer')
    if not isinstance(sc, dict):
        raise ValueError('khoa cham thieu blob bo cham')
    now = scorer_blobs()
    bad = [m for m in SCORER_MODULES if sc.get(m) != now[m]]
    if bad:
        raise ValueError(f'khoa cham: blob bo cham khac ban dang chay: {bad}')
    return lock


def score_window(score, npz_name, n, period, lock=None):
    """C3-17: mặt nạ cửa sổ hoàn thành (n,) bool từ kết quả hil_o1_score.score (docstring mục
    a) -> (mask, t_ht0, t_ht1, src). JSON không khớp log -> ValueError. C3-28: frame_log bắt
    buộc; k0 phải là chỉ số frame nguyên trong [0, n); t0_step, T_cmd, t_deadline phải hữu
    hạn; completed 'true' cần t_end hữu hạn; t_end khi có phải nằm trong
    [T_cmd; t_deadline] +-1e-9. lock (check_score_lock): bộ cờ và blob bộ chấm ghi vào src."""
    try:
        k0s, a = int(score['k0']), float(score['t0_step'])
        t_cmd, t_dl = float(score['T_cmd']), float(score['t_deadline'])
        fs = int(score['frames_scored'])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'ket qua cham thieu/sai truong: {exc}') from None
    if not all(math.isfinite(v) for v in (a, t_cmd, t_dl)):
        raise ValueError(f'ket qua cham co moc khong huu han: t0_step={a}, T_cmd={t_cmd}, '
                         f't_deadline={t_dl}')
    k = score['k0']
    if not (isinstance(k, int) and not isinstance(k, bool) and 0 <= k < n):
        raise ValueError(f'ket qua cham k0={k!r} khong phai chi so frame trong [0, {n})')
    fl = score.get('frame_log')
    if fl != npz_name:
        raise ValueError(f'ket qua cham cua {fl!r}, khong phai {npz_name!r}')
    if fs != n - k0s:
        raise ValueError(f'ket qua cham frames_scored={fs}, log co {n} frame, k0={k0s}')
    t_end = score.get('t_end')
    if t_end is not None:
        try:
            t_end = float(t_end)
        except (TypeError, ValueError):
            raise ValueError(f'ket qua cham t_end={t_end!r} khong phai so') from None
        if not math.isfinite(t_end):
            raise ValueError(f'ket qua cham t_end={t_end} khong huu han')
        if not (t_cmd - 1e-9 <= t_end <= t_dl + 1e-9):
            raise ValueError(f'ket qua cham t_end={t_end} ngoai cua so [T_cmd={t_cmd}; '
                             f't_deadline={t_dl}]')
    if score.get('completed') == 'true' and t_end is None:
        raise ValueError("ket qua cham completed='true' ma khong co t_end huu han")
    t = (np.arange(n) - a) * period
    m = (np.arange(n) >= k0s) & (t >= t_cmd - 1e-9) & (t < t_dl - 1e-9)
    t1 = t_dl
    if t_end is not None:
        m &= t <= t_end + 1e-9
        t1 = min(t_dl, t_end)
    src = f"hil_o1_score:{score.get('t0_src', '')}:completed={score.get('completed', '')}"
    if lock is not None:
        src += ''.join(f':{c}={int(lock[c])}' for c in CHAM_FLAGS)
        src += f":scorer_blob={lock['scorer']['hil_o1_score']}"
    return m, t_cmd, t1, src


def n_gaps_in(t_f, cut, t_lo, t_hi):
    """C3-31: số khoảng mất mẫu (t_f[k-1]; t_f[k]) (cut[k], k >= 1) giao cửa sổ [t_lo; t_hi):
    t_f[k] > t_lo và t_f[k-1] < t_hi (khoảng vắt qua t_lo hay t_hi cũng tính; khoảng kết thúc
    đúng tại t_lo thì mẫu mất đều trước t_lo, không tính)."""
    k = np.nonzero(np.asarray(cut, dtype=bool)[1:])[0] + 1
    return int(((t_f[k] > t_lo) & (t_f[k - 1] < t_hi)).sum())


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2:
        return float('nan')
    da, db = a - a.mean(), b - b.mean()
    sa = float(np.sqrt((da * da).sum()))
    sb = float(np.sqrt((db * db).sum()))
    if sa == 0.0 or sb == 0.0:
        return float('nan')
    return float((da * db).sum() / (sa * sb))


def read_esp_pwm(path):
    """timestamp_us, log_seq, pwm_* của CSV agv_logger (docstring: trùng lặp như
    hil_edge_map.read_esp_log; thiếu cột ESP_NEED -> ValueError; không dùng được -> EspRefused)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f'{path}: empty')
    head, body = rows[0], rows[1:]
    miss = [c for c in ESP_NEED if c not in head]
    if miss:
        raise ValueError(f'{path}: columns {miss} missing')
    if any(len(r) != len(head) for r in body):
        raise ValueError(f'{path}: lines with another column count')
    seg = path + '.segments.csv'
    keep, seen = [], set()
    if os.path.isfile(seg):
        with open(seg, newline='', encoding='utf-8') as f:
            marks = list(csv.reader(f))
        if not marks or marks[0] != SEG_MARK_COLS or len(marks) - 1 != len(body):
            raise EspRefused(f'{os.path.basename(seg)}: header hoặc số dòng không khớp')
        segs = set()
        s, t = head.index('log_seq'), head.index('timestamp_us')
        for i, (r, m) in enumerate(zip(body, marks[1:]), start=1):
            if m[0] != str(i) or m[1] != r[s] or m[2] != r[t]:
                raise EspRefused(f'{os.path.basename(seg)} dòng {i} không khớp dòng CSV')
            if m[4] in ('conflict', 'ambiguous'):
                raise EspRefused(f'{os.path.basename(seg)}: có dòng {m[4]}')
            if m[4] == 'duplicate':
                if tuple(r) not in seen:
                    raise EspRefused(f'{os.path.basename(seg)} dòng {i}: duplicate không có '
                                     'dòng bằng đã giữ')
                continue
            seen.add(tuple(r))
            segs.add(m[3])
            keep.append(r)
        if len(segs) > 1:
            raise EspRefused('hơn một segment (DUT khởi động lại)')
    else:
        for i, r in enumerate(body, start=1):
            if tuple(r) in seen:
                raise EspRefused(f'dòng {i + 1} trùng lặp, không có .segments.csv xác nhận bỏ '
                                 '(S05 6.3)')
            seen.add(tuple(r))
            keep.append(r)
    try:
        ts = np.array([int(r[head.index('timestamp_us')]) for r in keep], dtype=np.int64)
        seq = np.array([int(r[head.index('log_seq')]) for r in keep], dtype=np.int64)
        pw = np.rint(np.array([[float(r[head.index(c)]) for c in PWM_COLS]
                               for r in keep], dtype=float)).astype(np.int64)
        mon = (np.array([int(float(r[head.index('motor_on')])) for r in keep],
                        dtype=np.int8) if 'motor_on' in head else None)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from None
    order = np.argsort(seq, kind='stable')
    ts, seq, pw = ts[order], seq[order], pw[order]
    if mon is not None:
        mon = mon[order]
    if len(np.unique(seq)) != len(seq):
        raise EspRefused('hai dòng khác nhau cùng log_seq')
    return dict(ts=ts, seq=seq, pwm=pw, motor_on=mon)


def run_end_seqs(esp_path):
    """Tập last_data_seq của các dòng '#run_end' trong <log>.events.txt (agv_logger ghi 3
    dòng retry cùng một last_data_seq): bằng chứng seq[i-1] là dòng data motor_on cuối
    được đẩy trước khi DUT dừng. Không có file -> set rỗng -> mọi ứng viên n_off_unproven."""
    out = set()
    p = esp_path + '.events.txt'
    if not os.path.isfile(p):
        return out
    with open(p, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'#run_end.*last_data_seq=(\d+)', line)
            if m:
                out.add(int(m.group(1)))
    return out


def clock_pairs(raw_path):
    """(rx_us, te_us) từ .raw.jsonl: t_mono_ns/1e3 và timestamp_us (trường đầu) của mỗi gói
    chứa dòng dữ liệu (dòng decode bắt đầu bằng chữ số)."""
    rx, te = [], []
    with open(raw_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                rec = json.loads(line)
                txt = base64.b64decode(rec['b64']).decode('utf-8', 'replace').strip()
                t_ns = int(rec['t_mono_ns'])
            except Exception:                                # noqa: BLE001  dòng hỏng -> bỏ
                continue
            if not txt or not txt[0].isdigit():
                continue
            try:
                rx.append(t_ns / 1e3)
                te.append(int(txt.split(',', 1)[0]))
            except ValueError:
                continue
    return np.asarray(rx, dtype=float), np.asarray(te, dtype=float)


def count_run(npz_path, task, score=None, esp_path=None):
    """Phần MOSI/MISO của một frame log (docstring mục a-c) -> (row, aux, segs).
    score: kết quả hil_o1_score.score (dict) cho cửa sổ hoàn thành, hay None.
    aux: duty (n,4) float64, t_f (n,) µs từ t_send_ns hoặc None, miso_empty, valid, cut, k0."""
    with np.load(npz_path, allow_pickle=False) as z:
        if 'manifest' not in z.files:
            raise ValueError(f'{npz_path}: không có khoá manifest')
        manifest = json.loads(str(z['manifest']))
        for k in ('mosi', 'miso'):
            if k not in z.files:
                raise ValueError(f'{npz_path}: không có khoá {k}')
        mosi, miso = z['mosi'], z['miso']
        t_ns = np.asarray(z['t_send_ns'], dtype=np.int64) if 't_send_ns' in z.files else None
    t_f = t_ns.astype(np.float64) / 1e3 if t_ns is not None else None
    n = len(mosi)
    state, ref, flags, duty = hil_o1_score.decode_frames(mosi, miso)
    k0, k0_src = hil_o1_score.command_start(manifest, ref, flags)
    period_us = float(manifest.get('period_us', 1000))
    period = period_us * 1e-6
    period_ms = period_us / 1e3
    t = (np.arange(n) - k0) * period
    sl = slice(k0, n)
    t_r = t[sl]
    vband = in_speed_band(np.hypot(state[sl, 7], state[sl, 8]))
    wband_all = blind_band(hil_edge_map.h7_omega32(mosi))
    wband = wband_all[sl]
    dr = _int16(miso, 0, 4).astype(np.int64)[sl]
    miso_empty = ~np.asarray(miso, dtype=np.uint8).any(axis=1)
    d0, sat = dr == 0, np.abs(dr) == SAT_RAW
    flat_any = np.any(d0 | sat, axis=1)
    all4_0 = np.all(d0, axis=1)
    # C3-14: bằng chứng chỉ trên frame valid, đoạn cắt tại frame không valid và mất mẫu
    valid_all, crc_bad_all, cut_all = frame_validity(mosi, miso, t_ns, period_us)
    valid, crc_bad, cut = valid_all[sl], crc_bad_all[sl], cut_all[sl].copy()
    if len(cut):
        cut[0] = False
    kind = duty_kind(duty[sl])
    kind[~valid] = KIND_INVALID
    d0_bc, sat_bc = d0 & valid[:, None], sat & valid[:, None]
    flat_bc, all4_bc = flat_any & valid, all4_0 & valid
    flat_tol = np.any(np.abs(kind) <= 1, axis=1)

    win_kind, t_cmd = 'last1s_gia_dinh', None
    if task.startswith('O1-'):
        t_cmd = float(o1_tasks.task(task, allow_final_set=True)[0])   # ValueError: task sai
        wmask = (t_r >= t_cmd - 1e-9) & (t_r < t_cmd + T_LATE - 1e-9)
        win_kind = 'O1_[Tcmd,deadline)'
    else:
        wmask = t_r >= (t_r[-1] - T_D - 1e-9 if len(t_r) else 0.0)
    last1 = t_r >= (t_r[-1] - T_D - 1e-9 if len(t_r) else 0.0)

    row = dict(
        n_frames=n, k0=k0, k0_src=k0_src, n_run=len(t_r), period_us=period_us,
        t_last_s=round(float(t_r[-1]), 3) if len(t_r) else 0.0,
        t_cmd_s=round(t_cmd, 3) if t_cmd is not None else '',
        win_kind=win_kind, win_n=int(wmask.sum()),
        n_vband_win=int((vband & wmask).sum()),
        n_vband_last1s=int((vband & last1).sum()),
        n_vband_run=int(vband.sum()),
        n_wband_any_win=int((wband.any(axis=1) & wmask).sum()),
        n_wband_any_last1s=int((wband.any(axis=1) & last1).sum()),
        n_wband_any_run=int(wband.any(axis=1).sum()),
        n_duty0_any=int(np.any(d0, axis=1).sum()),
        n_sat_any=int(np.any(sat, axis=1).sum()),
        n_flat_any=int(flat_any.sum()),
        max_flat_ms=int(longest_run(flat_any) * period * 1e3),
        n_all4_duty0=int(all4_0.sum()),
        max_all4_0_ms=int(longest_run(all4_0) * period * 1e3),
        n_miso_empty_run=int(miso_empty[sl].sum()),
        n_miso_crc_bad_run=int(crc_bad.sum()),
        n_mat_mau_run=int(cut.sum()),
        n_khong_bc_run=int((~valid).sum()),
        n_duty0_any_bc=int(np.any(d0_bc, axis=1).sum()),
        n_sat_any_bc=int(np.any(sat_bc, axis=1).sum()),
        n_flat_any_bc=int(flat_bc.sum()),
        max_flat_bc_ms=float(longest_run(flat_bc, cut) * period_ms),
        n_all4_duty0_bc=int(all4_bc.sum()),
        max_all4_0_bc_ms=float(longest_run(all4_bc, cut) * period_ms),
        n_d0_tol_any=int(np.any(kind == 0, axis=1).sum()),
        n_sat_tol_any=int(np.any(np.abs(kind) == 1, axis=1).sum()),
        n_flat_tol_any=int(flat_tol.sum()),
        max_flat_tol_ms=float(longest_run(flat_tol, cut) * period_ms),
    )
    # C3-17: cửa sổ hoàn thành của bộ chấm (mặt nạ trên mọi frame của log)
    if score is not None:
        lock = check_score_lock(score, npz_path, task, esp_path)         # C3-28
        ht, t_ht0, t_ht1, src = score_window(score, os.path.basename(npz_path), n, period, lock)
        row.update(win_ht_src=src, t_ht0_s=round(t_ht0, 6), t_ht1_s=round(t_ht1, 6),
                   win_ht_n=int(ht.sum()),
                   n_vband_ht=int((in_speed_band(np.hypot(state[:, 7], state[:, 8])) & ht).sum()),
                   n_wband_any_ht=int((wband_all.any(axis=1) & ht).sum()))
        for w, nm in enumerate(WHEELS):
            row[f'n_wb_ht_{nm}'] = int((wband_all[:, w] & ht).sum())
    else:
        row['win_ht_src'] = 'khong_co_ket_qua_cham'
    segs = []
    for w, nm in enumerate(WHEELS):
        row[f'n_wb_win_{nm}'] = int((wband[:, w] & wmask).sum())
        row[f'n_wb_last1s_{nm}'] = int((wband[:, w] & last1).sum())
        row[f'n_wb_run_{nm}'] = int(wband[:, w].sum())
        row[f'n_d0_{nm}'] = int(d0_bc[:, w].sum())
        row[f'n_sat_{nm}'] = int(sat_bc[:, w].sum())
        row[f'n_d0_tol_{nm}'] = int((kind[:, w] == 0).sum())
        row[f'n_sat_tol_{nm}'] = int((np.abs(kind[:, w]) == 1).sum())
        sg = tol_segments(kind[:, w], cut)
        row[f'n_seg_tol_{nm}'] = len(sg)
        lens = [L * period_ms for _, _, L in sg]
        row[f'seg_med_ms_{nm}'] = float(np.median(lens)) if lens else ''
        row[f'max_seg_tol_ms_{nm}'] = max(lens) if lens else ''
        segs += [dict(wheel=nm, kind=kd, k_start=k0 + s, len_frames=L, len_ms=L * period_ms)
                 for kd, s, L in sg]
    aux = dict(duty=duty, t_f=t_f, miso_empty=miso_empty, valid=valid_all, cut=cut_all, k0=k0)
    return row, aux, segs


def esp_measure(esp_path, aux):
    """Phần ESP32 (docstring mục c-ESP32) -> (fields, events). fields['esp_part'] =
    'co' / 'loi: ...'. ValueError của read_esp_pwm (thiếu cột) để caller -> mã 2."""
    log = read_esp_pwm(esp_path)
    out = dict(esp_log=esp_path)
    raw = esp_path + '.raw.jsonl'
    if not os.path.isfile(raw):
        out['esp_part'] = 'loi: khong co raw.jsonl'
        return out, []
    rx, te = clock_pairs(raw)
    fit = s06_t5_score.clock_fit(rx, te, bin_us=CLOCK_BIN_US)
    if fit is None:
        out['esp_part'] = 'loi: khong khop dong ho'
        return out, []
    t_pi = fit['a_us'] + (1.0 + fit['b']) * log['ts'].astype(np.float64)
    out.update(esp_part='co', sync_kind='udp_envelope', sync_ppm=fit['ppm'],
               sync_resid_max_us=fit['resid_max_us'], sync_span_s=fit['span_s'])
    duty, t_f, k0 = aux['duty'], aux['t_f'], aux['k0']
    valid, cut = aux['valid'], aux['cut']
    n = len(t_f)

    # dấu từng bánh: pwm giữ tại mỗi frame valid của khoảng chạy so với duty MISO
    sel = np.nonzero(valid[k0:])[0] + k0
    idx = np.searchsorted(t_pi, t_f[sel], side='right') - 1
    have = idx >= 0
    sign_ok = []
    for w, nm in enumerate(WHEELS):
        c = (_pearson(log['pwm'][idx[have], w], duty[sel[have], w])
             if np.any(have) else float('nan'))
        ok = bool(math.isfinite(c) and int(np.sign(c)) == SIGN_CODE and abs(c) >= MIN_CORR)
        out[f'corr_{nm}'] = round(c, 6) if math.isfinite(c) else ''
        out[f'sign_{nm}'] = int(np.sign(c)) if math.isfinite(c) else ''
        out[f'sign_ok_{nm}'] = int(ok)
        sign_ok.append(ok)

    # frame kế tiếp thoả |duty - d*| <= TOL+EPS cho từng (bánh, mục tiêu) (frame không valid bỏ qua)
    next_hit = {}
    for w in range(4):
        for tv in (-1.0, 0.0, 1.0):
            hit = np.nonzero(valid & (np.abs(duty[:, w] - tv) <= TOL + EPS))[0]
            nxt = np.full(n, n, dtype=np.int64)
            if len(hit):
                pos = np.searchsorted(hit, np.arange(n), 'left')
                nxt[pos < len(hit)] = hit[pos[pos < len(hit)]]
            next_hit[(w, tv)] = nxt

    seq, ts, pwm = log['seq'], log['ts'], log['pwm']
    mon = log['motor_on']
    out['motor_on_col'] = int(mon is not None)
    to_pi = lambda x: float(fit['a_us'] + (1.0 + fit['b']) * x)   # noqa: E731
    events, n_seqgap, n_outside, n_unk = [], 0, 0, 0
    resid_ms = float(fit['resid_max_us']) / 1000.0

    def event(loai, nm, w, d_t, t0, t_end, moc, bat_dinh, **kw):
        """C3-15: lần đầu quan sát vào dải trong [t0; t_end), kèm số frame liền trong dải,
        frame không quan sát được trên đoạn đã quét và lý do bị cắt (docstring c-ESP32)."""
        k_first = int(np.searchsorted(t_f, t0, 'left'))
        k_last = int(np.searchsorted(t_f, t_end, 'left')) if math.isfinite(t_end) else n
        kf = int(next_hit[(w, d_t)][k_first]) if k_first < n else n
        inb = kf < k_last
        k_stop = kf + 1 if inb else min(k_last, n)
        # C3-31: khoảng mất mẫu giao cửa sổ quan sát [t0; t_f[kf]] hay [t0; t_end), kể cả vắt
        # qua t0 hay t_end
        n_kbc = (int((~valid[k_first:k_stop]).sum())
                 + n_gaps_in(t_f, cut, t0, t_f[kf] if inb else t_end))
        n_lien = 0
        if inb:
            k = kf
            while (k < k_last and valid[k] and (k == kf or not cut[k])
                   and abs(duty[k, w] - d_t) <= TOL + EPS):
                n_lien += 1
                k += 1
        cat = '' if inb else ('doi_lenh' if t_end <= t_f[-1] else 'het_log')
        return dict(
            loai=loai, wheel=nm, target=int(d_t), t_pi_us=round(t0, 3), moc_t0=moc,
            bat_dinh_ms=round(bat_dinh, 6), lech_udp=LECH_UDP, k_first=k_first,
            duty_first=(round(float(duty[k_first, w]), 6)        # C3-31: không valid -> trống
                        if k_first < n and valid[k_first] else ''),
            k_vao_dai=kf if inb else '',
            t_vao_dai_ms=round((t_f[kf] - t0) / 1000.0, 6) if inb else '',
            vao_dai=inb, n_lien_vao_dai=n_lien, n_khong_bc=n_kbc, cat=cat,
            cens_ms='' if inb else round((min(t_end, t_f[-1]) - t0) / 1000.0, 6),
            cmd_level_ms='' if not math.isfinite(t_end) else round((t_end - t0) / 1000.0, 6),
            **kw)

    for w, nm in enumerate(WHEELS):
        pw = pwm[:, w]
        for i in range(1, len(pw)):
            if pw[i] not in PWM_FLAT or pw[i] == pw[i - 1]:
                continue
            if mon is not None and not (mon[i] == 1 and mon[i - 1] == 1):
                continue                                # dòng motor_on=0 không là 'lenh'
            if seq[i] != seq[i - 1] + 1:
                n_seqgap += 1
                continue
            t0 = float(t_pi[i])
            if not (t_f[k0] <= t0 <= t_f[-1]):
                n_outside += 1
                continue
            if abs(int(pw[i])) == 1000 and not sign_ok[w]:
                n_unk += 1
                continue
            d_t = 0.0 if pw[i] == 0 else SIGN_CODE * (int(pw[i]) / 1000.0)
            j = i + 1
            while j < len(pw) and pw[j] == pw[i]:
                j += 1
            t_end = float(t_pi[j]) if j < len(pw) else math.inf
            events.append(event('lenh', nm, w, d_t, t0, t_end, 'udp_envelope', resid_ms,
                                log_seq=int(seq[i]), pwm_prev=int(pw[i - 1]),
                                t_esp_us=int(ts[i])))
    out.update(n_ev_seqgap=n_seqgap, n_ev_outside=n_outside, n_ev_sign_unknown=n_unk)

    # sự kiện tắt motor ('tat_motor'): dòng đầu motor_on 1->0, chứng minh bằng last_data_seq
    n_off_unpr, n_off_out = 0, 0
    if mon is not None:
        dt = np.diff(ts)[(np.diff(seq) == 1) & (mon[1:] == 1) & (mon[:-1] == 1)]
        dt_med = float(np.median(dt)) if len(dt) else None
        out.update(tick_dt_min_ms=float(dt.min()) / 1e3 if len(dt) else '',
                   tick_dt_med_ms=dt_med / 1e3 if dt_med is not None else '',
                   tick_dt_max_ms=float(dt.max()) / 1e3 if len(dt) else '')
        proven = run_end_seqs(esp_path)
        bd_off = (resid_ms + max(dt_med - float(dt.min()), float(dt.max()) - dt_med) / 1000.0
                  if dt_med is not None else '')
        for i in range(1, len(mon)):
            if not (mon[i] == 0 and mon[i - 1] == 1):
                continue
            for w, nm in enumerate(WHEELS):
                if pwm[i - 1, w] == 0:
                    continue
                if int(seq[i - 1]) not in proven or dt_med is None:
                    n_off_unpr += 1
                    continue
                t0 = to_pi(float(ts[i - 1]) + dt_med)   # tick stop ~một chu kỳ sau dòng on cuối
                if not (t_f[k0] <= t0 <= t_f[-1]):
                    n_off_out += 1                      # plant đã ngừng log: không phải censored
                    continue
                j = i
                while j < len(pwm) and pwm[j, w] == 0:
                    j += 1
                t_end = float(t_pi[j]) if j < len(pwm) else math.inf
                events.append(event('tat_motor', nm, w, 0.0, t0, t_end,
                                    'udp_envelope+tick_dt_med', bd_off,
                                    log_seq=int(seq[i - 1]), pwm_prev=int(pwm[i - 1, w]),
                                    t_esp_us=int(round(float(ts[i - 1]) + dt_med))))
    out.update(a11_stats([e for e in events if e['loai'] == 'tat_motor'], 'off'),
               n_off_unproven=n_off_unpr, n_off_outside=n_off_out)
    lenh = [e for e in events if e['loai'] == 'lenh']
    for T in (0, 1):
        ev = [e for e in lenh if abs(e['target']) == T]
        out.update(a11_stats(ev, T))
        out.update(a11_stats([e for e in ev if abs(SIGN_CODE * e['pwm_prev'] / 1000.0
                                                   - e['target']) > TOL + EPS], f'doi_{T}'))
    return out, events


def a11_stats(ev, sfx):
    """Các cột A11_STAT (hậu tố sfx) của một nhóm sự kiện (docstring c-ESP32)."""
    fol = [e['t_vao_dai_ms'] for e in ev if e['vao_dai']]
    cen = [e['cens_ms'] for e in ev if not e['vao_dai']]
    v = (len(ev), len(fol), len(cen),
         float(np.median(fol)) if fol else '',
         float(np.percentile(fol, 95, method='higher')) if fol else '',
         max(fol) if fol else '', max(cen) if cen else '',
         sum(1 for e in ev if e['cat'] == 'doi_lenh'),
         sum(1 for e in ev if e['cat'] == 'het_log'),
         sum(1 for e in ev if e['n_khong_bc'] > 0))
    return dict(zip(a11_cols(sfx), v))


def process_one(npz_path, esp_path, task, score=None):
    """Một npz (+ log ESP32 nếu có, + kết quả chấm nếu có) -> (row, segs, events).
    ValueError/OSError -> caller."""
    row, aux, segs = count_run(npz_path, task, score, esp_path=esp_path)
    if esp_path is None:
        row['esp_part'] = 'khong_co_log_ESP32'
        return row, segs, []
    if aux['t_f'] is None:
        row.update(esp_log=esp_path, esp_part='loi: khong co t_send_ns')
        return row, segs, []
    try:
        fields, events = esp_measure(esp_path, aux)
    except EspRefused as exc:
        fields, events = dict(esp_log=esp_path, esp_part=f'loi: {exc}'), []
    row.update(fields)
    return row, segs, events


def write_csv(path, cols, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore', restval='')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _num_eq(a, b):
    """So sánh số của hai ô CSV theo int (so_khop_s1); không parse được thì so theo chuỗi."""
    try:
        return int(float(a)) == int(float(b))
    except (TypeError, ValueError):
        return str(a) == str(b)


def compare_s1(s1, row, run, fn):
    """Một dòng s1 so với dòng e2 (docstring: cùng tên CMP_SAME, cặp CMP_PAIRS)."""
    out = []
    for c in CMP_SAME:
        v1, v2 = s1.get(c, ''), row.get(c, '')
        same = (v1 == v2) if c == 'win_kind' else _num_eq(v1, v2)
        if not same:
            out.append(dict(run=run, npz=fn, cot_s1=c, cot_e2=c,
                            gia_tri_s1=v1, gia_tri_e2=v2))
    for cs, ce in CMP_PAIRS:
        v1, v2 = s1.get(cs, ''), row.get(ce, '')
        if not _num_eq(v1, v2):
            out.append(dict(run=run, npz=fn, cot_s1=cs, cot_e2=ce,
                            gia_tri_s1=v1, gia_tri_e2=v2))
    return out


def batch(a):
    if not a.root:
        print('loi dau vao: --batch-from-s1 can --root', file=sys.stderr)
        return s06_stream.report_result(2)
    try:
        with open(a.batch_from_s1, newline='', encoding='utf-8') as f:
            s1_rows = list(csv.DictReader(f))
    except OSError as exc:
        print(f'loi dau vao: {exc}', file=sys.stderr)
        return s06_stream.report_result(2)
    os.makedirs(os.path.join(a.out, 'su_kien'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'doan_phang'), exist_ok=True)
    if a.cham:
        os.makedirs(os.path.join(a.out, 'cham'), exist_ok=True)
    rows, mis, n_err = [], [], 0
    for r in s1_rows:
        run, fn = r.get('run', ''), r.get('npz', '')
        name = r.get('name') or name_of(fn)
        task = r.get('task', '')
        fl = os.path.join(a.root, run, 'frames', fn)
        esp = os.path.join(a.root, run, 'pi', name + '.csv')
        esp = esp if os.path.isfile(esp) else None
        score, cham_err = None, ''
        if a.cham and task.startswith('O1-'):
            # C3-17: mặt nạ cửa sổ hoàn thành từ chính bộ chấm (docstring "Chạy theo lô");
            # C3-28: khoá SHA-256 NPZ + task, bộ cờ, blob bộ chấm ghi cùng kết quả
            try:
                lock = score_lock(fl, task, esp)
                score = hil_o1_score._json_safe(hil_o1_score.score(fl, task, esp_log=esp,
                                                                   **CHAM_FLAGS))
                score[LOCK_KEY] = lock
                with open(os.path.join(a.out, 'cham', f'{run}__{name}_score.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(score, f, ensure_ascii=False, indent=1)
            except Exception as exc:                         # noqa: BLE001  ghi lại, lô chạy tiếp
                score, cham_err = None, f'loi_cham: {type(exc).__name__}: {exc}'
        try:
            row, segs, events = process_one(fl, esp, task, score)
        except (OSError, KeyError, ValueError) as exc:
            rows.append(dict(run=run, npz=fn, name=name, task=task,
                             error=f'{type(exc).__name__}: {exc}'))
            n_err += 1
            continue
        if cham_err:
            row['win_ht_src'] = cham_err
        rows.append(dict(run=run, npz=fn, name=name, task=task, **row))
        write_csv(os.path.join(a.out, 'su_kien', f'{run}__{name}_su_kien_a11.csv'),
                  EVENT_COLS, events)
        write_csv(os.path.join(a.out, 'doan_phang', f'{run}__{name}_doan_phang.csv'),
                  SEG_OUT_COLS, segs)
        mis += compare_s1(r, row, run, fn)
    write_csv(os.path.join(a.out, 'ket_qua_e2.csv'),
              ['run', 'npz', 'name', 'task'] + ROW_COLS + ['error'], rows)
    write_csv(os.path.join(a.out, 'so_khop_s1.csv'),
              ['run', 'npz', 'cot_s1', 'cot_e2', 'gia_tri_s1', 'gia_tri_e2'], mis)
    n_no_esp = sum(1 for r in rows if not r.get('error') and r.get('esp_part') != 'co')
    print(f'rows={len(rows)} errors={n_err} no_esp={n_no_esp} mismatches={len(mis)}')
    code = 2 if n_err else (3 if n_no_esp else 0)
    return s06_stream.report_result(code, ['thieu_bang_chung'])


def main(argv=None):
    ap = argparse.ArgumentParser(description='Dem gioi han HIL tren frame log (E2)')
    ap.add_argument('--frame-log', help='frames npz của một lượt')
    ap.add_argument('--esp-log', help='CSV agv_logger của cùng lượt (không có -> chỉ MISO)')
    ap.add_argument('--task', default='', help='O1-F02/O1-K1/... (O1-* -> cửa sổ O1; khác -> last1s)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-from-s1', help='ket_qua_s1.csv của dem_s1.py')
    ap.add_argument('--root', help='thư mục CLOSED_LOOP (cho --batch-from-s1)')
    ap.add_argument('--score-json', help='kết quả chấm của cùng lượt do --cham ghi '
                    '(cham/<run>__<ten>_score.json, có khoá SHA-256 NPZ + task): mặt nạ cửa sổ '
                    'hoàn thành, cột *_ht')
    ap.add_argument('--cham', action='store_true',
                    help='lô: chấm dòng O1-* bằng hil_o1_score.score, mặt nạ *_ht từ kết quả')
    a = ap.parse_args(argv)
    if a.batch_from_s1:
        return batch(a)
    if not a.frame_log:
        print('loi dau vao: can --frame-log hoac --batch-from-s1', file=sys.stderr)
        return s06_stream.report_result(2)
    try:
        score = None
        if a.score_json:
            with open(a.score_json, encoding='utf-8') as f:
                score = json.load(f)
            if not isinstance(score, dict):
                raise ValueError(f'{a.score_json}: khong phai JSON object')
        row, segs, events = process_one(a.frame_log, a.esp_log, a.task, score)
    except (OSError, KeyError, ValueError) as exc:
        print(f'loi dau vao: {exc}', file=sys.stderr)
        return s06_stream.report_result(2)
    name = name_of(os.path.basename(a.frame_log))
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, f'{name}_gioi_han.csv'), ['name', 'task'] + ROW_COLS,
              [dict(name=name, task=a.task, **row)])
    write_csv(os.path.join(a.out, f'{name}_su_kien_a11.csv'), EVENT_COLS, events)
    write_csv(os.path.join(a.out, f'{name}_doan_phang.csv'), SEG_OUT_COLS, segs)
    print(f'frames={row["n_frames"]} k0={row["k0"]} n_run={row["n_run"]} '
          f'esp_part {row["esp_part"]}')
    return s06_stream.report_result(0 if row['esp_part'] == 'co' else 3,
                                  ['thieu_bang_chung'])


if __name__ == '__main__':
    raise SystemExit(main())
