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
  * Mỗi đoạn ESP32 giữ duty của một bánh (giữa hai lần ghi compare của bánh đó): bỏ 100 frame đầu từ
    t_mốc = t_trước của lệnh ghi (V3-7), chấm đúng 200 frame có CRC MISO đúng kế tiếp; d = duty_MISO −
    duty_chuẩn. Đoạn không đủ 300 frame → `short`. Mục 4 với bất định chuẩn `--u-duty` (logic analyzer,
    Z11; mặc định 0, ghi rõ): max|d| + u ≤ 0,006 đạt; max|d| − u > 0,006 không đạt; còn lại chưa kết luận.
  * Đủ mức: mỗi bánh có mọi mức của spec ở cả hai chiều và duty 0 (`LEVELS`); thiếu → mã 3.
  * Chỉ báo theo kịp (không ngưỡng; Y8): từ t_mốc tới frame đầu tiên MISO trong ±0,006 quanh mức mới
    và giữ 100 frame liền (`follow_ms`), và tới frame đầu tiên rời ±0,006 quanh mức cũ (`leave_ms`);
    chưa ổn định trước lần đổi kế → `settled` false ("> khoảng giữ"), vẫn ghi. Thành phần mốc
    [0; t_sau − t_trước + 50 µs] ghi ở cột `t_mark_span_us`. Báo lớn nhất và trung vị theo cặp mức.

Diễn giải của CODE (spec chưa ghi, cần THESIS xác nhận khi khóa): đồng hồ theo chặn dưới UDP khi chưa có
chân đánh dấu; frame CRC MISO sai bị bỏ qua khi đếm 200 frame (T1 chấm CRC); cột `pin` = chân H7 của
kênh đang băm (RPWM khi duty > 0, LPWM khi < 0, `-` khi 0).

Usage:
  python tools/s06_t6_check.py <frames.npz> --esp32-events esp32_events.txt --out <thư mục>
         [--clock-affine OFFSET_US RATE] [--u-duty 0.0017]
Mã thoát: 0 đạt; 1 không đạt; 3 thiếu mức/đoạn ngắn/chưa kết luận; 2 lỗi đầu vào.
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

COLUMNS = ['idx', 'wheel', 'pin', 'segment', 'duty_cmd', 'frame', 'seq', 'duty_miso', 'err']
FOLLOW_COLS = ['idx', 'wheel', 'duty_from', 'duty_to', 't_cmd_us', 't_mark_span_us', 'frame_leave',
               'frame_settle', 'follow_ms', 'leave_ms', 'settled']


def read_events(path):
    """Dòng P theo bánh (giờ ESP32) và cặp (t_rx, t_esp) cho đồng hồ."""
    P, rx, te = defaultdict(list), [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep:
                continue
            p = line.split(',')
            try:
                if p[0] == 'P' and len(p) == 10:
                    P[int(p[4])].append(dict(t_before=int(p[2]), t_after=int(p[3]), speed=int(p[5]),
                                             cmp={int(p[6]): int(p[7]), int(p[8]): int(p[9])}))
                    rx.append(int(t_rx))
                    te.append(int(p[3]))
                elif p[0] == 'E' and len(p) == 8:
                    rx.append(int(t_rx))
                    te.append(int(p[3]))
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


def active_pin(wheel, d, wiring=WIRING):
    return wiring[wheel][0][1] if d > 0 else (wiring[wheel][1][1] if d < 0 else '-')


def check(t_send_ns, mosi, miso, P, to_pi, u_duty=0.0, wiring=WIRING):
    """to_pi: hàm giờ ESP32 (µs) → giờ Pi (µs)."""
    t_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    rows, _ = dec.decode_frames(t_send_ns, np.asarray(mosi, dtype=np.uint8), np.asarray(miso, dtype=np.uint8))
    good = np.array([not r['empty'] and r['crc_ok'] for r in rows])
    duty = np.array([decode_torques(bytes(m)) for m in np.asarray(miso, dtype=np.uint8)])
    seq = np.asarray(miso, dtype=np.uint8)[:, dec.MISO_SEQ]
    scored, follow, levels = [], [], []
    seen = defaultdict(set)
    for w in range(4):
        ev = P.get(w, [])
        for s, e in enumerate(ev):
            d_new = duty_ref(w, e['cmp'], wiring)
            seen[w].add(round(d_new, 3))
            t0 = float(to_pi(e['t_before']))
            t_end = float(to_pi(ev[s + 1]['t_before'])) if s + 1 < len(ev) else t_us[-1] + 1.0
            k0, k1 = bisect.bisect_left(t_us, t0), bisect.bisect_left(t_us, t_end)
            ks = [k for k in range(k0, k1) if good[k]]
            lv = dict(wheel=WHEELS[w], segment=s, duty_cmd=d_new, frames=len(ks), max_abs_err=None, verdict='short')
            if len(ks) >= SKIP + SCORE:
                errs = []
                for k in ks[SKIP:SKIP + SCORE]:
                    err = duty[k, w] - d_new
                    errs.append(abs(err))
                    scored.append(dict(wheel=WHEELS[w], pin=active_pin(w, d_new, wiring), segment=s, duty_cmd=d_new,
                                       frame=k, seq=int(seq[k]), duty_miso=float(duty[k, w]), err=float(err)))
                m = max(errs)
                lv.update(max_abs_err=m, verdict='pass' if m + u_duty <= LIMIT else
                          ('fail' if m - u_duty > LIMIT else 'inconclusive'))
            levels.append(lv)
            if s == 0:
                continue
            d_old = duty_ref(w, ev[s - 1]['cmp'], wiring)
            leave = settle = None
            run = 0
            for k in ks:
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
    for i, r in enumerate(scored):
        r['idx'] = i
    for i, r in enumerate(follow):
        r['idx'] = i
    return scored, follow, levels, missing, int(np.sum(~good))


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


def verdict(levels, missing):
    v = [l['verdict'] for l in levels]
    if 'fail' in v:
        return 1
    if missing or 'short' in v or 'inconclusive' in v:
        return 3
    return 0


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T6 (spec S06 v3.4)')
    ap.add_argument('frames')
    ap.add_argument('--esp32-events', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clock-affine', nargs=2, type=float, default=None, metavar=('OFFSET_US', 'RATE'),
                    help='t_pi_us = OFFSET + RATE * t_esp_us (chân đánh dấu); bỏ trống = chặn dưới UDP')
    ap.add_argument('--u-duty', type=float, default=0.0, help='bất định duty chuẩn (logic analyzer, Z11)')
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        P, rx, te = read_events(a.esp32_events)
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
                        note='giờ Pi suy ra trễ hơn thật bằng trễ UDP nhỏ nhất (chưa đo): follow_ms ngắn hơn thật')
        scored, follow, levels, missing, crc_bad = check(z['t_send_ns'], z['mosi'], z['miso'], P, to_pi, a.u_duty)
    except (OSError, KeyError, ValueError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, 'T6.csv'), scored, COLUMNS)
    write_csv(os.path.join(a.out, 'T6_follow.csv'), follow, FOLLOW_COLS)
    write_csv(os.path.join(a.out, 'T6_levels.csv'), levels, ['wheel', 'segment', 'duty_cmd', 'frames', 'max_abs_err', 'verdict'])
    errs = [l['max_abs_err'] for l in levels if l['max_abs_err'] is not None]
    summ = dict(max_abs_err=max(errs) if errs else None, u_duty=a.u_duty, limit=LIMIT,
                levels={v: sum(l['verdict'] == v for l in levels) for v in ('pass', 'fail', 'inconclusive', 'short')},
                levels_missing=missing, miso_crc_bad_frames=crc_bad, sync=sync,
                follow=follow_summary(follow), exit_code=verdict(levels, missing))
    with open(os.path.join(a.out, 'T6_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps(summ, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    return summ['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
