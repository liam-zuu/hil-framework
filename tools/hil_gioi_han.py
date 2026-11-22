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

(b) Băng mù theo bánh (đề xuất v6 C3(b)): blind_band(omega32) -> (n,4) bool, a = |omega32|
    float32, (a > 0) & (a <= float32(0.3)): H7 không phát xung encoder khi
    |omega| <= ENC_OMEGA_MIN = 0.3f (stm32h7/Core/Src/encoder_gen.c:5 và :212-213, new_dir
    giữ 0 trừ khi omega > 0.3f hay < -0.3f). omega32 là giá trị H7 decode từ MOSI bytes 0-7:
    int16 BE x (40.0f / 32767.0f), float32 (tools/hil_edge_map.py h7_omega32).
    KHÁC dem_s1: dem_s1 đếm khoảng MỞ (0; 0.3) trên state float64 và chỉ theo any-wheel;
    công cụ này đếm (0; 0.3f] float32 theo từng bánh (n_wb_*_<banh>) và cột any-wheel
    (n_wband_any_*) để so sánh với n_wband_* của dem_s1 (so_khop_s1.csv).

(c) MISO trên khoảng chạy:
    - exact (tương thích dem_s1, từ int16 raw): duty0 (raw == 0), sat (|raw| == 32767),
      flat = duty0 | sat, all4 duty0, dòng MISO rỗng (mọi byte 0 -> decode ra duty 0, đếm
      riêng n_miso_empty_run để trừ khi cần), max đoạn liên tục [ms] = int(run * period_ms).
    - tolerance TOL = 0.0055 + EPS (spec S06 v3.5 T6 dòng 120): |d| <= TOL + EPS -> loại 0;
      |d| >= 1 - TOL - EPS -> loại +1 / -1 theo dấu; còn lại 'khác' (cắt đoạn). Mỗi bánh: số
      frame loại 0 / loại sat, số đoạn liên tục tối đa cùng một loại trong {0,+1,-1}, median
      và max độ dài đoạn [ms]; any-wheel: n_d0_tol_any, n_sat_tol_any, n_flat_tol_any,
      max_flat_tol_ms. CSV đoạn phẳng: wheel, kind, k_start (chỉ số frame trong npz),
      len_frames, len_ms.
    - CRC MISO KHÔNG kiểm (như dem_s1).

(c-ESP32) Thời gian giữ mức A11 (đề xuất v6 C5(c), E mục 2), chỉ khi có --esp-log:
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
      bỏ frame MISO rỗng: frame đầu |duty[k] - d*| <= TOL + EPS -> theo kịp, hold_ms =
      (t_f[k] - t0)/1000; không có -> censored ("> khoảng giữ", KHÔNG bỏ), cens_ms =
      (min(t_end, t_f[-1]) - t0)/1000.
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
      dòng j >= i đầu có pwm[j] != 0 (không có -> +inf); cùng phép tìm follow. Cột:
      n_ev_off, n_follow_off, n_cens_off, hold_med/p95/max_ms_off, cens_max_ms_off,
      n_off_unproven, n_off_outside — tách khỏi thống kê 'lenh' mục tiêu 0. Ứng viên
      có t0 ngoài [t_f[k0]; t_f[-1]] -> n_off_outside, không chấm: lượt dừng cuối run
      thường đến SAU khi frame log kết thúc (lượt HIL dừng trước, DUT tắt motor sau khi
      mất lệnh), nên thời gian giữ mức lúc dừng không quan sát được trong các log này.
    - Tách sự kiện hiển nhiên ('doi' = đổi thật) cho 'lenh': chỉ giữ sự kiện có
      |SIGN_CODE x pwm_prev/1000 - d*| > TOL + EPS (pwm_prev = pwm[i-1] của bánh đó):
      n_ev_doi_<T>, n_follow_doi_<T>, n_cens_doi_<T>, hold_med/p95/max_doi_ms_<T>,
      T in (0,1). Các cột mọi-sự-kiện vẫn giữ.
    - CSV sự kiện: thêm loai ('lenh' / 'tat_motor') và pwm_prev.
    - 9 npz của 2026-11-08_run08/run09/run11 (w1, w2, w3) không có log ESP32 phía Pi (chỉ
      esp32_COM5.txt serial không giờ Pi) -> chỉ phần MISO, esp_part 'khong_co_log_ESP32'.

Chạy một lượt:
  python tools/hil_gioi_han.py --frame-log X_frames.npz [--esp-log X.csv] [--task O1-F02]
      --out DIR
  Ghi DIR/<ten>_gioi_han.csv (một dòng), DIR/<ten>_su_kien_a11.csv (sự kiện ESP32),
  DIR/<ten>_doan_phang.csv (đoạn phẳng MISO). <ten> = tên npz bỏ '_frames.npz' (hay '.npz').

Chạy theo lô:
  python tools/hil_gioi_han.py --batch-from-s1 ket_qua_s1.csv --root CLOSED_LOOP --out DIR
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

ROW_COLS = ['n_frames', 'k0', 'k0_src', 'n_run', 'period_us', 't_last_s', 't_cmd_s',
            'win_kind', 'win_n', 'n_vband_win', 'n_vband_last1s', 'n_vband_run',
            'n_wband_any_win', 'n_wband_any_last1s', 'n_wband_any_run']
for _w in WHEELS:
    ROW_COLS += [f'n_wb_win_{_w}', f'n_wb_last1s_{_w}', f'n_wb_run_{_w}']
ROW_COLS += ['n_duty0_any', 'n_sat_any', 'n_flat_any', 'max_flat_ms', 'n_all4_duty0',
             'max_all4_0_ms', 'n_miso_empty_run']
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
    ROW_COLS += [f'n_ev_{_t}', f'n_follow_{_t}', f'n_cens_{_t}', f'hold_med_ms_{_t}',
                 f'hold_p95_ms_{_t}', f'hold_max_ms_{_t}', f'cens_max_ms_{_t}']
ROW_COLS += ['n_ev_seqgap', 'n_ev_outside', 'n_ev_sign_unknown',
             'motor_on_col', 'tick_dt_min_ms', 'tick_dt_med_ms', 'tick_dt_max_ms',
             'n_ev_off', 'n_follow_off', 'n_cens_off', 'hold_med_ms_off',
             'hold_p95_ms_off', 'hold_max_ms_off', 'cens_max_ms_off', 'n_off_unproven',
             'n_off_outside']
for _t in (0, 1):
    ROW_COLS += [f'n_ev_doi_{_t}', f'n_follow_doi_{_t}', f'n_cens_doi_{_t}',
                 f'hold_med_doi_ms_{_t}', f'hold_p95_doi_ms_{_t}', f'hold_max_doi_ms_{_t}']

EVENT_COLS = ['loai', 'wheel', 'target', 'log_seq', 'pwm_prev', 't_esp_us', 't_pi_us',
              'k_first', 'duty_first', 'k_follow', 'hold_ms', 'followed', 'cens_ms',
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


def longest_run(mask):
    """Đoạn True liên tục dài nhất của mảng bool (như dem_s1, vector hoá)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    d = np.diff(m.astype(np.int8))
    starts = np.nonzero(d == 1)[0] + 1
    ends = np.nonzero(d == -1)[0] + 1
    if m[0]:
        starts = np.concatenate(([0], starts))
    if m[-1]:
        ends = np.concatenate((ends, [len(m)]))
    return int((ends - starts).max())


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


def tol_segments(kind):
    """Các đoạn liên tục tối đa cùng loại in {0,+1,-1} -> list (kind, k_start, len_frames);
    frame 'khác' (kind 2) cắt đoạn."""
    a = np.asarray(kind)
    edges = np.nonzero(np.diff(a.astype(np.int16)) != 0)[0] + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(a)]))
    return [(int(a[s]), int(s), int(e - s)) for s, e in zip(starts, ends) if a[s] != 2]


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


def count_run(npz_path, task):
    """Phần MOSI/MISO của một frame log (docstring mục a-c) -> (row, aux, segs).
    aux: duty (n,4) float64, t_f (n,) µs từ t_send_ns hoặc None, miso_empty, k0."""
    with np.load(npz_path, allow_pickle=False) as z:
        if 'manifest' not in z.files:
            raise ValueError(f'{npz_path}: không có khoá manifest')
        manifest = json.loads(str(z['manifest']))
        for k in ('mosi', 'miso'):
            if k not in z.files:
                raise ValueError(f'{npz_path}: không có khoá {k}')
        mosi, miso = z['mosi'], z['miso']
        t_f = (np.asarray(z['t_send_ns'], dtype=np.float64) / 1e3
               if 't_send_ns' in z.files else None)
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
    wband = blind_band(hil_edge_map.h7_omega32(mosi)[sl])
    dr = _int16(miso, 0, 4).astype(np.int64)[sl]
    miso_empty = ~np.asarray(miso, dtype=np.uint8).any(axis=1)
    d0, sat = dr == 0, np.abs(dr) == SAT_RAW
    flat_any = np.any(d0 | sat, axis=1)
    all4_0 = np.all(d0, axis=1)
    kind = duty_kind(duty[sl])

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
        n_d0_tol_any=int(np.any(kind == 0, axis=1).sum()),
        n_sat_tol_any=int(np.any(np.abs(kind) == 1, axis=1).sum()),
        n_flat_tol_any=int(np.any(kind != 2, axis=1).sum()),
        max_flat_tol_ms=float(longest_run(np.any(kind != 2, axis=1)) * period_ms),
    )
    segs = []
    for w, nm in enumerate(WHEELS):
        row[f'n_wb_win_{nm}'] = int((wband[:, w] & wmask).sum())
        row[f'n_wb_last1s_{nm}'] = int((wband[:, w] & last1).sum())
        row[f'n_wb_run_{nm}'] = int(wband[:, w].sum())
        row[f'n_d0_{nm}'] = int(d0[:, w].sum())
        row[f'n_sat_{nm}'] = int(sat[:, w].sum())
        row[f'n_d0_tol_{nm}'] = int((kind[:, w] == 0).sum())
        row[f'n_sat_tol_{nm}'] = int((np.abs(kind[:, w]) == 1).sum())
        sg = tol_segments(kind[:, w])
        row[f'n_seg_tol_{nm}'] = len(sg)
        lens = [L * period_ms for _, _, L in sg]
        row[f'seg_med_ms_{nm}'] = float(np.median(lens)) if lens else ''
        row[f'max_seg_tol_ms_{nm}'] = max(lens) if lens else ''
        segs += [dict(wheel=nm, kind=kd, k_start=k0 + s, len_frames=L, len_ms=L * period_ms)
                 for kd, s, L in sg]
    aux = dict(duty=duty, t_f=t_f, miso_empty=miso_empty, k0=k0)
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
    duty, t_f, miso_empty, k0 = aux['duty'], aux['t_f'], aux['miso_empty'], aux['k0']
    n = len(t_f)

    # dấu từng bánh: pwm giữ tại mỗi frame không rỗng của khoảng chạy so với duty MISO
    sel = np.nonzero(~miso_empty[k0:])[0] + k0
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

    # frame kế tiếp thoả |duty - d*| <= TOL+EPS cho từng (bánh, mục tiêu) (frame rỗng bỏ qua)
    next_hit = {}
    for w in range(4):
        for tv in (-1.0, 0.0, 1.0):
            hit = np.nonzero(~miso_empty & (np.abs(duty[:, w] - tv) <= TOL + EPS))[0]
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

    def follow(k_first0, k_last0, w0, d_t0):
        kf = int(next_hit[(w0, d_t0)][k_first0]) if k_first0 < n else n
        return kf, kf < k_last0

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
            k_first = int(np.searchsorted(t_f, t0, 'left'))
            k_last = int(np.searchsorted(t_f, t_end, 'left')) if math.isfinite(t_end) else n
            k_follow, followed = follow(k_first, k_last, w, d_t)
            events.append(dict(
                loai='lenh', wheel=nm, target=int(d_t), log_seq=int(seq[i]),
                pwm_prev=int(pw[i - 1]), t_esp_us=int(ts[i]),
                t_pi_us=round(t0, 3), k_first=k_first,
                duty_first=round(float(duty[k_first, w]), 6),
                k_follow=k_follow if followed else '',
                hold_ms=round((t_f[k_follow] - t0) / 1000.0, 6) if followed else '',
                followed=followed,
                cens_ms='' if followed else round((min(t_end, t_f[-1]) - t0) / 1000.0, 6),
                cmd_level_ms='' if not math.isfinite(t_end)
                else round((t_end - t0) / 1000.0, 6)))
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
                k_first = int(np.searchsorted(t_f, t0, 'left'))
                k_last = (int(np.searchsorted(t_f, t_end, 'left'))
                          if math.isfinite(t_end) else n)
                k_follow, followed = follow(k_first, k_last, w, 0.0)
                events.append(dict(
                    loai='tat_motor', wheel=nm, target=0, log_seq=int(seq[i - 1]),
                    pwm_prev=int(pwm[i - 1, w]),
                    t_esp_us=int(round(float(ts[i - 1]) + dt_med)),
                    t_pi_us=round(t0, 3), k_first=k_first,
                    duty_first=round(float(duty[k_first, w]), 6) if k_first < n else '',
                    k_follow=k_follow if followed else '',
                    hold_ms=round((t_f[k_follow] - t0) / 1000.0, 6) if followed else '',
                    followed=followed,
                    cens_ms='' if followed
                    else round((min(t_end, t_f[-1]) - t0) / 1000.0, 6),
                    cmd_level_ms='' if not math.isfinite(t_end)
                    else round((t_end - t0) / 1000.0, 6)))
    off = [e for e in events if e['loai'] == 'tat_motor']
    fol_o = [e['hold_ms'] for e in off if e['followed']]
    cen_o = [e['cens_ms'] for e in off if not e['followed']]
    out.update(n_ev_off=len(off), n_follow_off=len(fol_o), n_cens_off=len(cen_o),
               hold_med_ms_off=float(np.median(fol_o)) if fol_o else '',
               hold_p95_ms_off=(float(np.percentile(fol_o, 95, method='higher'))
                                if fol_o else ''),
               hold_max_ms_off=max(fol_o) if fol_o else '',
               cens_max_ms_off=max(cen_o) if cen_o else '',
               n_off_unproven=n_off_unpr, n_off_outside=n_off_out)
    lenh = [e for e in events if e['loai'] == 'lenh']
    for T in (0, 1):
        ev = [e for e in lenh if abs(e['target']) == T]
        fol = [e['hold_ms'] for e in ev if e['followed']]
        cen = [e['cens_ms'] for e in ev if not e['followed']]
        out.update({
            f'n_ev_{T}': len(ev), f'n_follow_{T}': len(fol), f'n_cens_{T}': len(cen),
            f'hold_med_ms_{T}': float(np.median(fol)) if fol else '',
            f'hold_p95_ms_{T}': float(np.percentile(fol, 95, method='higher')) if fol else '',
            f'hold_max_ms_{T}': max(fol) if fol else '',
            f'cens_max_ms_{T}': max(cen) if cen else ''})
        doi = [e for e in ev
               if abs(SIGN_CODE * e['pwm_prev'] / 1000.0 - e['target']) > TOL + EPS]
        fol_d = [e['hold_ms'] for e in doi if e['followed']]
        out.update({
            f'n_ev_doi_{T}': len(doi),
            f'n_follow_doi_{T}': len(fol_d),
            f'n_cens_doi_{T}': len(doi) - len(fol_d),
            f'hold_med_doi_ms_{T}': float(np.median(fol_d)) if fol_d else '',
            f'hold_p95_doi_ms_{T}': (float(np.percentile(fol_d, 95, method='higher'))
                                     if fol_d else ''),
            f'hold_max_doi_ms_{T}': max(fol_d) if fol_d else ''})
    return out, events


def process_one(npz_path, esp_path, task):
    """Một npz (+ log ESP32 nếu có) -> (row, segs, events). ValueError/OSError -> caller."""
    row, aux, segs = count_run(npz_path, task)
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
    rows, mis, n_err = [], [], 0
    for r in s1_rows:
        run, fn = r.get('run', ''), r.get('npz', '')
        name = r.get('name') or name_of(fn)
        task = r.get('task', '')
        try:
            fl = os.path.join(a.root, run, 'frames', fn)
            esp = os.path.join(a.root, run, 'pi', name + '.csv')
            esp = esp if os.path.isfile(esp) else None
            row, segs, events = process_one(fl, esp, task)
        except (OSError, KeyError, ValueError) as exc:
            rows.append(dict(run=run, npz=fn, name=name, task=task,
                             error=f'{type(exc).__name__}: {exc}'))
            n_err += 1
            continue
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
    a = ap.parse_args(argv)
    if a.batch_from_s1:
        return batch(a)
    if not a.frame_log:
        print('loi dau vao: can --frame-log hoac --batch-from-s1', file=sys.stderr)
        return s06_stream.report_result(2)
    try:
        row, segs, events = process_one(a.frame_log, a.esp_log, a.task)
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
