#!/usr/bin/env python3
"""
s06_t6_check.py — chấm T6 của spec S06 v3.4 (H6: sai số duty PWM đọc được; W2, W15, V3, V10, Y8, Z11,
L7-13, V2-8, V3-7).

Đầu vào: `frames.npz` (run_hil.py --frame-log; MISO byte 0–7 = duty 4 bánh, int16 big-endian ×
1/32767, `protocol.decode_torques`, đúng giá trị plant nhận) và `esp32_events.txt` của s06_logger.py
với env `s06_load` hồ sơ `t6` (dòng `P,qseq,t_before,t_after,wheel,speed,gpioA,cmpA,gpioB,cmpB` mỗi
lần ghi compare; dòng E cho đồng hồ).

  * Chuẩn duty [Z11]: duty_chuẩn = (compare tại GPIO nối vào RPWM của H7 − compare tại GPIO nối vào
    LPWM của H7)/1000, theo bảng nối dây (mặc định `WIRING` = 00_OVERVIEW/Setup_Guides/
    2026-10-22_Huong_dan_Noi_day_HIL.md mục 4; khớp `pwm_capture.c` _ch[w] = RPWM, _ch[w+4] = LPWM).
    Dòng P có GPIO không thuộc bánh đó trong bảng → lỗi ánh xạ, mã 2.
  * Đồng hồ ESP32 → RPi5: `--clock-affine OFFSET RATE` (từ chân đánh dấu, khi có), hoặc mặc định
    chặn dưới t_rx − t_esp của các dòng có giờ ESP32 (tools/s06_t5_score.clock_fit). Cách mặc định
    trễ hơn giờ thật một khoảng bằng trễ UDP nhỏ nhất (cỡ ms, chưa đo): đủ cho phần tĩnh (bỏ 100 frame),
    làm thời gian theo kịp **ngắn hơn thật** tối đa khoảng đó; ghi `sync` trong kết quả.
  * Mỗi đoạn ESP32 giữ duty của một bánh (giữa hai lần ghi compare ĐỔI duty của bánh đó): bỏ 100 frame
    đầu từ t_mốc = t_trước của lệnh ghi (V3-7), chấm đúng 200 frame kế tiếp theo chỉ số (S2); d =
    duty_MISO − duty_chuẩn. Đoạn không đủ 300 frame → `short`. Mục 4 với bất định chuẩn `--u-duty`
    (logic analyzer, Z11): max|d| + u ≤ 0,006 đạt; max|d| − u > 0,006 không đạt; còn lại chưa kết luận.
  * Đủ mức: mỗi bánh có mọi mức của spec ở cả hai chiều và duty 0 (`LEVELS`); thiếu → mã 3.
  * Chỉ báo theo kịp (không ngưỡng; Y8): từ t_mốc tới frame đầu tiên MISO trong ±0,006 quanh mức mới
    và giữ 100 frame liền (`follow_ms`), và tới frame đầu tiên rời ±0,006 quanh mức cũ (`leave_ms`);
    chưa ổn định trước lần đổi kế → `settled` false ("> khoảng giữ"), vẫn ghi. Thành phần mốc
    [0; t_sau − t_trước + 50 µs] ghi ở cột `t_mark_span_us`. Báo lớn nhất và trung vị theo cặp mức.

Sửa theo Codex Sol vòng 1/2 (Liam duyệt 30/10 "Duyệt, sửa hết"), thay các diễn giải cũ:
  * S2: 100 frame bỏ và 200 frame chấm tính theo CHỈ SỐ frame thật từ t_mốc; frame MISO sai CRC hoặc rỗng
    trong 200 frame = mất dữ liệu (Z19) → mức đó `fail_data_loss`, không lấy frame sau bù vào; chuỗi
    "giữ 100 frame liền" của chỉ báo theo kịp cũng theo chỉ số thật, frame hỏng cắt chuỗi.
  * S18: phải có hai dòng W15 (mỗi bánh một duty khác nhau, cùng lúc: chỉ dòng này phân biệt được lẫn bánh).
  * S19, X20: `--u-duty` bỏ trống hoặc 0 = bất định chuẩn chưa có (lượt chính phải có số từ Z11): tính như 0
    nhưng kết luận tối đa mã 3.
  * S21: không có `--clock-affine` (chân đánh dấu) thì đồng hồ theo chặn dưới UDP chỉ để khảo sát: tập 200
    frame chấm dịch theo trễ UDP, kết luận tối đa mã 3.
  * S22: lần ghi compare lặp lại cùng duty (s06_apply ép ghi lúc kết thúc lượt) nối tiếp đoạn cũ; đoạn bắt
    đầu từ #run_end không phải mức của lịch.
  * S23: cột `pin` = chân H7 của GPIO có compare khác 1000 (motor.c `motor_set_info`: chân không lái giữ
    compare = PWM_RESOLUTION); `-` khi duty 0.

Sửa theo Codex Sol vòng 2/2 (Liam 30/10 "Sửa P1 rồi vòng 3"):
  * S40: đủ luồng cả lượt, không chỉ 200 frame chấm: mọi qseq trong [#run_start; #run_end] phải có dòng
    (s06_stream.qseq_stream), `ev_qdrop_run` = 0, và không frame MISO nào sai CRC/rỗng trong lượt (giờ Pi);
    thiếu → mã 1 (Z19). Thiếu #run_start/#run_end (hay qseq của chúng) → mã 3.

Manifest lượt (Codex Sol vòng 3, Liam 30/10 "Làm manifest lượt"):
  * S56: dòng P có qseq mà nội dung hỏng (sai số trường, không phải số) là mất dữ liệu (mã 1), không được
    tính là đã nhận.
  * S57: frames.npz phải có manifest của hil_node (thiếu → mã 3), số frame khớp manifest (thiếu → mã 1), và
    frame Pi phủ lượt ± 1 s (log cắt sớm/bắt đầu muộn → mã 3).
  * S58: P và mốc lượt chọn theo `run` (s06_load ghi run= ở dòng P và các mốc); file nhiều lượt mà không
    có `--run` → mã 3.

Usage:
  python tools/s06_t6_check.py <frames.npz> --esp32-events esp32_events.txt --out <thư mục>
         --clock-affine OFFSET_US RATE --u-duty 0.0017
Mã thoát: 0 đạt; 1 không đạt (giá trị hoặc mất dữ liệu); 3 thiếu mức/W15, đoạn ngắn, chưa kết luận, thiếu
chân đánh dấu hoặc --u-duty; 2 lỗi đầu vào.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'rpi5')]
import s06_frames_decode as dec  # noqa: E402
import s06_stream as stm  # noqa: E402
import s06_t5_score as t5s  # noqa: E402
from protocol import decode_torques  # noqa: E402

LIMIT = 0.006
SKIP, SCORE, HOLD_OK = 100, 200, 100
PWM_PERIOD_US = 50.0
CLOCK_BIN_US = 1e6            # lượt T6 chỉ ~31 s: bin chặn dưới 1 s (T5 dùng 10 s cho lượt 20 phút)
WHEELS = ('FL', 'FR', 'RL', 'RR')
# wheel -> ((GPIO -> H7 RPWM, H7 pin), (GPIO -> H7 LPWM, H7 pin)); Huong_dan_Noi_day_HIL mục 4
WIRING = {0: ((3, 'PA0'), (8, 'PE5')), 1: ((41, 'PF8'), (40, 'PE6')),
          2: ((9, 'PA2'), (46, 'PF6')), 3: ((21, 'PA3'), (47, 'PF7'))}
LEVELS = (0.03, 0.05, 0.10, 0.30, 0.33, 0.50, 0.55, 0.70, 0.77, 0.90, 0.95, 0.97, 1.00)
PWM_RESOLUTION = 1000         # AGV motor.h: compare 0..1000
W15_ROWS = ((0.10, 0.33, 0.55, 0.77), (-0.10, -0.33, -0.55, -0.77))   # s06_load.c T6_STEPS (W15)
W15_T_TOL_US = 50_000.0       # [ĐỀ XUẤT CODE] four wheels of one row written within 50 ms (s06_apply: one loop)

COLUMNS = ['idx', 'wheel', 'pin', 'segment', 'duty_cmd', 'frame', 'seq', 'duty_miso', 'err']
FOLLOW_COLS = ['idx', 'wheel', 'duty_from', 'duty_to', 't_cmd_us', 't_mark_span_us', 'frame_leave',
               'frame_settle', 'follow_ms', 'leave_ms', 'settled']


def read_events(path, marks=None, run=None):
    """Dòng P theo bánh (giờ ESP32) và cặp (t_rx, t_esp) cho đồng hồ. `marks` (dict, tùy chọn) nhận
    timestamp_us (giờ ESP32) của #run_start/#run_end của lượt đã chọn (s06_stream.run_marks, `run`).
    Manifest lượt (S58): dòng P/E có trường run (s06_load) thì chỉ giữ dòng của lượt `run`; dòng P sai dạng
    không bị bỏ im lặng: s06_stream.qseq_stream tính nó là hỏng (S56). S71: không chọn được lượt thì dừng ở
    đây, trước khi đọc dòng nào; mốc có số lượt thì dòng P không có số lượt không thuộc lượt này."""
    P, rx, te = defaultdict(list), [], []
    if marks is not None:
        m = stm.run_marks(path, run)
        stm.require_run(m)
        marks['_run'] = m.get('_run')
        for k in ('run_start', 'run_end'):
            if k in m and 'timestamp_us' in m[k]['kv']:
                marks[k] = int(m[k]['kv']['timestamp_us'])
        run = m.get('_run')
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep:
                continue
            p = line.split(',')
            try:
                if p[0] == 'P' and len(p) in (10, 11):
                    if run is not None and (len(p) != 11 or int(p[10]) != run):
                        continue
                    P[int(p[4])].append(dict(t_before=int(p[2]), t_after=int(p[3]), speed=int(p[5]),
                                             cmp={int(p[6]): int(p[7]), int(p[8]): int(p[9])}))
                    rx.append(int(t_rx))
                    te.append(int(p[3]))
                elif p[0] == 'E':
                    e = stm.parse_e(p)
                    if e is not None:
                        rx.append(int(t_rx))
                        te.append(e[2])
            except ValueError:
                continue
    for w in P:
        P[w].sort(key=lambda e: e['t_before'])
    return P, np.array(rx, dtype=float), np.array(te, dtype=float)


def duty_ref(wheel, cmp, wiring=WIRING):
    (gr, _), (gl, _) = wiring[wheel]
    if set(cmp) != {gr, gl}:
        raise ValueError(f'bánh {WHEELS[wheel]}: dòng P có GPIO {sorted(cmp)}, bảng nối dây là {gr}, {gl}')
    return (cmp[gr] - cmp[gl]) / 1000.0


def active_pin(wheel, cmp, wiring=WIRING, top=PWM_RESOLUTION):
    """S23: chân H7 của kênh đang băm = GPIO có compare khác PWM_RESOLUTION (motor.c `motor_set_info`:
    chân không lái giữ compare = 1000); không có (duty 0) → '-'."""
    pins = {g: p for g, p in wiring[wheel]}
    on = [pins[g] for g, c in sorted(cmp.items()) if c != top]
    return on[0] if len(on) == 1 else '-'


def w15_rows_present(segs, rows=W15_ROWS, t_tol_us=W15_T_TOL_US):
    """S18: the W15 rows (one different duty per wheel at the same time) are the only levels that tell a
    wheel mix-up apart; each must show as a segment of every wheel with its duty, starting within
    t_tol_us of each other. segs: wheel -> [(t0_pi_us, duty)]. Returns the rows missing."""
    missing = []
    for row in rows:
        starts = []
        for w, d in enumerate(row):
            starts.append([t for t, x in segs.get(w, []) if abs(x - d) < 1e-9])
        ok = any(all(any(abs(t - t0) <= t_tol_us for t in s) for s in starts[1:]) for t0 in starts[0])
        if not ok:
            missing.append(list(row))
    return missing


def check(t_send_ns, mosi, miso, P, to_pi, u_duty=None, wiring=WIRING, run_end_esp=None):
    """to_pi: hàm giờ ESP32 (µs) → giờ Pi (µs). u_duty None = bất định chuẩn chưa có (tính như 0, kết
    luận cuối không được đạt: `verdict`). run_end_esp: timestamp_us của #run_end (giờ ESP32)."""
    t_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    rows, _ = dec.decode_frames(t_send_ns, np.asarray(mosi, dtype=np.uint8), np.asarray(miso, dtype=np.uint8))
    good = np.array([not r['empty'] and r['crc_ok'] for r in rows])
    duty = np.array([decode_torques(bytes(m)) for m in np.asarray(miso, dtype=np.uint8)])
    seq = np.asarray(miso, dtype=np.uint8)[:, dec.MISO_SEQ]
    scored, follow, levels = [], [], []
    seen = defaultdict(set)
    seg_starts = defaultdict(list)
    t_stop = float(to_pi(run_end_esp)) if run_end_esp is not None else None
    for w in range(4):
        # S22: one segment per change of duty: a write of the same duty (s06_apply forced at the end of a
        # run) continues the segment; a segment starting at/after #run_end is not a level of the schedule
        ev = []
        for e in P.get(w, []):
            d = duty_ref(w, e['cmp'], wiring)
            if ev and abs(d - ev[-1][1]) < 1e-9:
                continue
            ev.append((e, d))
        for s, (e, d_new) in enumerate(ev):
            t0 = float(to_pi(e['t_before']))
            if t_stop is not None and t0 >= t_stop:
                continue
            seen[w].add(round(d_new, 3))
            seg_starts[w].append((t0, d_new))
            t_end = float(to_pi(ev[s + 1][0]['t_before'])) if s + 1 < len(ev) else t_us[-1] + 1.0
            if t_stop is not None:
                t_end = min(t_end, t_stop)
            k0, k1 = bisect.bisect_left(t_us, t0), bisect.bisect_left(t_us, t_end)
            # S2: the 100 skipped and the 200 scored frames are counted by the REAL frame index from t_mark;
            # a frame with a bad or empty MISO among the 200 is lost data (Z19): the level fails, no frame
            # is taken from later to fill in
            lv = dict(wheel=WHEELS[w], segment=s, duty_cmd=d_new, frames=k1 - k0, bad_frames=0, max_abs_err=None,
                      verdict='short')
            if k1 - k0 >= SKIP + SCORE:
                sel = range(k0 + SKIP, k0 + SKIP + SCORE)
                bad = [k for k in sel if not good[k]]
                errs = []
                for k in sel:
                    if not good[k]:
                        continue
                    err = duty[k, w] - d_new
                    errs.append(abs(err))
                    scored.append(dict(wheel=WHEELS[w], pin=active_pin(w, e['cmp'], wiring), segment=s,
                                       duty_cmd=d_new, frame=k, seq=int(seq[k]), duty_miso=float(duty[k, w]),
                                       err=float(err)))
                m = max(errs) if errs else None
                u = u_duty or 0.0
                lv.update(bad_frames=len(bad), max_abs_err=m)
                if bad:
                    lv['verdict'] = 'fail_data_loss'
                elif m + u <= LIMIT:
                    lv['verdict'] = 'pass'
                elif m - u > LIMIT:
                    lv['verdict'] = 'fail'
                else:
                    lv['verdict'] = 'inconclusive'
            levels.append(lv)
            if s == 0:
                continue
            d_old = ev[s - 1][1]
            leave = settle = None
            run = 0
            for k in range(k0, k1):            # S2: 100 frames IN A ROW by real index; a bad frame breaks it
                if not good[k]:
                    run = 0
                    continue
                v = duty[k, w]
                if leave is None and abs(v - d_old) > LIMIT:
                    leave = k
                if abs(v - d_new) <= LIMIT:
                    run += 1
                    if run == HOLD_OK:
                        settle = k - HOLD_OK + 1
                        break
                else:
                    run = 0
            follow.append(dict(wheel=WHEELS[w], duty_from=d_old, duty_to=d_new, t_cmd_us=t0,
                               t_mark_span_us=e['t_after'] - e['t_before'] + PWM_PERIOD_US,
                               frame_leave='' if leave is None else leave, frame_settle='' if settle is None else settle,
                               follow_ms='' if settle is None else (t_us[settle] - t0) / 1000.0,
                               leave_ms='' if leave is None else (t_us[leave] - t0) / 1000.0,
                               settled=settle is not None))
    want = {round(x, 3) for x in LEVELS} | {-round(x, 3) for x in LEVELS} | {0.0}
    missing = {WHEELS[w]: sorted(want - seen[w]) for w in range(4) if want - seen[w]}
    w15 = w15_rows_present(seg_starts)
    if w15:
        missing['W15'] = w15
    for i, r in enumerate(scored):
        r['idx'] = i
    for i, r in enumerate(follow):
        r['idx'] = i
    return scored, follow, levels, missing, good


def follow_summary(follow):
    by = defaultdict(list)
    for r in follow:
        by[(r['duty_from'], r['duty_to'])].append(r)
    out = []
    for (a, b), rs in sorted(by.items()):
        f = [r['follow_ms'] for r in rs if r['settled']]
        out.append(dict(duty_from=a, duty_to=b, n=len(rs), not_settled=sum(not r['settled'] for r in rs),
                        follow_ms_max=max(f) if f else None, follow_ms_median=statistics.median(f) if f else None))
    return out


def verdict(levels, missing, synced=True, u_known=True, stream_ok=True, stream_known=True):
    """1 any level fails (value or lost data, Z19), or the run lost data anywhere (S40: a P/qseq line of the
    run missing, an ESP32 queue drop, a bad/empty MISO frame inside the run, even outside the 200 scored
    frames); 3 a level missing (incl. W15), short or inconclusive, or the acceptance evidence missing:
    no #run_start/#run_end qseq marks (S40), no marker-pin clock (S21: the UDP envelope moves the 200
    scored frames) or no reference uncertainty `--u-duty` (S19); 0 otherwise."""
    v = [l['verdict'] for l in levels]
    if 'fail' in v or 'fail_data_loss' in v or not stream_ok:
        return 1
    if (missing or 'short' in v or 'inconclusive' in v or not synced or not u_known or not stream_known):
        return 3
    return 0


def run_frames_bad(t_send_ns, good, t0_pi, t1_pi):
    """S40: bad or empty MISO frames among the frames the Pi sent inside the run (Pi clock), the 200
    scored ones or not: lost measurement data of the run (Z19)."""
    t = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    sel = (t >= t0_pi) & (t <= t1_pi)
    return int(np.sum(sel & ~good)), int(np.sum(sel))


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T6 (spec S06 v3.4)')
    ap.add_argument('frames')
    ap.add_argument('--esp32-events', required=True)
    ap.add_argument('--run', type=int, default=None,
                    help='số lượt (run= của #run_start) khi file có nhiều lượt; bỏ trống = file một lượt (S58)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--clock-affine', nargs=2, type=float, default=None, metavar=('OFFSET_US', 'RATE'),
                    help='t_pi_us = OFFSET + RATE * t_esp_us (chân đánh dấu); bỏ trống = chặn dưới UDP')
    ap.add_argument('--u-duty', type=float, default=None,
                    help='bất định duty chuẩn (logic analyzer, Z11); bỏ trống = chưa có, kết luận tối đa mã 3')
    a = ap.parse_args(argv)
    try:
        if a.u_duty is not None and not (np.isfinite(a.u_duty) and a.u_duty >= 0):
            raise ValueError(f'--u-duty phải hữu hạn, ≥ 0: {a.u_duty}')
        z = np.load(a.frames)
        marks = {}
        P, rx, te = read_events(a.esp32_events, marks, a.run)
        if not P:
            raise ValueError('không có dòng P (env s06_load hồ sơ t6?)')
        if a.clock_affine:
            off, rate = a.clock_affine
            to_pi, sync = (lambda t: off + rate * np.asarray(t, dtype=float)), dict(kind='manual', offset_us=off, rate=rate)
        else:
            fit = t5s.clock_fit(rx, te, bin_us=CLOCK_BIN_US)
            if fit is None:
                raise ValueError('không đủ dữ liệu khớp đồng hồ ESP32 → Pi')
            to_pi = lambda t: fit['a_us'] + (1.0 + fit['b']) * np.asarray(t, dtype=float)  # noqa: E731
            sync = dict(kind='udp_envelope', **fit,
                        note='khảo sát: giờ Pi suy ra trễ hơn thật bằng trễ UDP nhỏ nhất (chưa đo), tập 200 frame '
                             'chấm và follow_ms dịch theo; không cấp kết luận nghiệm thu (S21)')
        scored, follow, levels, missing, good = check(z['t_send_ns'], z['mosi'], z['miso'], P, to_pi, a.u_duty,
                                                      run_end_esp=marks.get('run_end'))
        # S40: the whole run must be complete, not only the 200 scored frames of each level
        qs = stm.qseq_stream(a.esp32_events, stm.run_marks(a.esp32_events, a.run))
        # S57: the frame log itself: every step the manifest says was logged, and the frames cover the run
        manifest = dec.load_manifest(z)
        frame_faults = dec.manifest_check(manifest, z['t_send_ns'], z['mosi'], z['miso'])
        cover_note = ''
        if 'run_start' in marks and 'run_end' in marks:
            t0p, t1p = float(to_pi(marks['run_start'])), float(to_pi(marks['run_end']))
            run_bad, run_frames = run_frames_bad(z['t_send_ns'], good, t0p, t1p)
            ts = np.asarray(z['t_send_ns'], dtype=np.float64) / 1000.0
            if ts[0] > t0p - dec.PAD_US or ts[-1] < t1p + dec.PAD_US:
                cover_note = f'frame Pi không phủ lượt ± {dec.PAD_US / 1e6:g} s (log cắt sớm hoặc bắt đầu muộn)'
        else:
            run_bad = run_frames = None
    except stm.RunProblem as exc:
        print(f'không chấm: {exc}', file=sys.stderr)
        return stm.report_result(3, ['khong_du_dieu_kien'])
    except (OSError, KeyError, ValueError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, 'T6.csv'), scored, COLUMNS)
    write_csv(os.path.join(a.out, 'T6_follow.csv'), follow, FOLLOW_COLS)
    for i, l in enumerate(levels):
        l['idx'] = i
    write_csv(os.path.join(a.out, 'T6_levels.csv'), levels,
              ['idx', 'wheel', 'segment', 'duty_cmd', 'frames', 'bad_frames', 'max_abs_err', 'verdict'])
    errs = [l['max_abs_err'] for l in levels if l['max_abs_err'] is not None]
    summ = dict(max_abs_err=max(errs) if errs else None, u_duty=a.u_duty, limit=LIMIT,
                levels={v: sum(l['verdict'] == v for l in levels)
                        for v in ('pass', 'fail', 'fail_data_loss', 'inconclusive', 'short')},
                levels_missing=missing, miso_crc_bad_frames=int(np.sum(~good)), run_frames=run_frames,
                run_frames_bad=run_bad, qseq_stream=qs, manifest=manifest, frame_log_faults=frame_faults,
                frames_cover_note=cover_note, run=marks.get('_run'), sync=sync, run_end_esp_us=marks.get('run_end'),
                follow=follow_summary(follow),
                # X20: an acceptance run needs a number from Z11: `--u-duty` empty or 0 is not known
                exit_code=verdict(levels, missing, synced=sync['kind'] == 'manual', u_known=bool(a.u_duty),
                                  stream_ok=qs['ok'] and not qs['malformed'] and not run_bad and not frame_faults,
                                  stream_known=(qs['known'] and run_bad is not None and manifest is not None
                                                and not cover_note)))
    lv = summ['levels']
    reasons = (['khong_du_dieu_kien'] if missing or lv['short'] or cover_note else []) + \
              (['chua_ket_luan'] if lv['inconclusive'] else []) + \
              (['thieu_bang_chung'] if (sync['kind'] != 'manual' or not a.u_duty or not qs['known'] or run_bad is None
                                        or manifest is None) else [])
    with open(os.path.join(a.out, 'T6_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps(summ, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    return stm.report_result(summ['exit_code'], reasons)


if __name__ == '__main__':
    raise SystemExit(main())
