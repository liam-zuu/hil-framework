#!/usr/bin/env python3
"""
s06_t4_values.py — chấm T4 (4) của spec S06 v3.4: giá trị IMU tới firmware nghiệm thu
(Z4, L7-1, L7-3, L7-6, V2-3; dung sai A6, Liam duyệt 29/10 "Theo khuyến nghị").

Mỗi report tới callback ứng dụng ESP32 (dòng `I`, applied = 1, `s06_load`) được ghép với frame
nguồn: khóa (epoch, loại, số thứ tự SH-2 mở vòng trong epoch) → bản ghi dựng report của H7
trong luồng log MISO (tools/s06_frames_decode.py) → `src_nf` → frame MOSI của Pi → giá trị IMU
của frame đó (protocol.decode_imu). So từng trục, đúng dấu, đúng trục:
  accel  (x, y, z) với (ax, ay, az)            dung sai 2,55e-3 m/s²   (Q8 + nửa bước int16)
  gyro   (x, y, z) với (0, 0, gz)              dung sai 1,51e-3 rad/s  (Q9 + nửa bước int16)
  game RV (roll, pitch, yaw) [độ → rad] với (0, 0, yaw), hiệu yaw quấn về [−π; π)
                                              dung sai 2,2e-4 rad     (spec V2-3, THESIS 29/10)
`s06_load` in giá trị callback bằng `%.9g`, nên số hạng làm tròn in ≈ 0. Đơn vị game RV ở dòng `I` là
độ: `imu.cpp` gọi `get_euler(meta)` với mặc định `in_degrees = true` (`BNO08xRptRVGeneric.hpp:23`).
Report có ở cổng SPI mà không tới callback (thư viện ghi đè) không có giá trị: không chấm ở đây
(luật "đếm ở cổng SPI", T1). Dòng `I` không ghép được frame nguồn = không đạt.

Sửa theo Codex Sol vòng 1/2 (Liam duyệt 30/10 "Duyệt, sửa hết"):
  * S3: giá trị, giá trị kỳ vọng, sai số phải hữu hạn (NaN/Inf = không đạt).
  * S9, S34: danh tính report của callback lấy từ dòng J cổng SPI (bộ giải mã: cùng seq, gần nhất theo
    thời gian), không mở vòng riêng trên tập callback thưa, không đòi J trước I.
  * S33: chấm cả dòng I applied = 0 (giá trị đã đọc ở callback trước khi lấy mutex), đếm riêng.
  * S32: chỉ chấm callback trong khung chung của lượt (bộ giải mã).
  * S18: đủ kích thích: mỗi loại report phải có ít nhất một callback đạt ở MỖI mức tĩnh của kênh nó mang
    và ở đoạn dốc của lịch khóa `rpi5/t4_imu_profile.py`; thiếu → mã 3.
  * S28: `--esp32-data` (bắt buộc ở chế độ nghiệm thu): cột `yaw_rad` của mỗi tick DATA phải bằng yaw của
    callback game RV cùng `rv_n` (độ → rad, quấn), dung sai 6e-5 rad (%.4f + float32) [ĐỀ XUẤT CODE].

Usage:
  python tools/s06_t4_values.py <frames.npz> <esp32_events.txt> --esp32-data <esp32_data.txt> --out <thư mục>
Mã thoát: 0 đạt đủ; 1 có report sai, NaN, không ghép được, yaw DATA lệch hoặc bộ giải mã báo vấn đề;
3 thiếu bằng chứng (không có callback, chưa phủ đủ lịch, không có/không đọc được esp32_data.txt); 2 lỗi đầu vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'rpi5')]
import s06_frames_decode as dec  # noqa: E402
import t4_imu_profile as prof  # noqa: E402
from protocol import decode_imu  # noqa: E402

TOL = {'accel': 2.55e-3, 'gyro': 1.51e-3, 'game_rv': 2.2e-4}
DATA_YAW_TOL = 6e-5           # S28: %.4f of logger.c (5e-5) + float32 of imu_fusion.c [ĐỀ XUẤT CODE]
UNIT = {'accel': 'm/s^2', 'gyro': 'rad/s', 'game_rv': 'rad'}


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def expected(kind, imu):
    gz, ax, ay, az, yaw = imu[:5]
    if kind == 'accel':
        return (ax, ay, az)
    if kind == 'gyro':
        return (0.0, 0.0, gz)
    return (0.0, 0.0, yaw)


def errors(kind, v, exp):
    if kind == 'game_rv':
        got = tuple(math.radians(x) for x in v)
        return tuple(wrap_pi(g - e) for g, e in zip(got, exp)), got
    return tuple(g - e for g, e in zip(v, exp)), v


CHANNELS = {'gyro': ('gz',), 'accel': ('ax', 'ay'), 'game_rv': ('yaw',)}


def stimulus_point(kind, src_frame, period_us=1000.0):
    """S18: point of the locked T4 schedule (rpi5/t4_imu_profile.py) of a source frame, for the channel
    this report kind carries: a static level 'gz=-5' …, 'ramp', or None (a level of another channel).
    The schedule runs on step × period (hil_node.py `_imu_sample`) and the frame-log index is the step,
    so the point is read from the schedule time, not guessed from the value (the ramp crosses ±5)."""
    seg = prof.segment(src_frame * period_us * 1e-6)
    if seg == 'ramp' or seg.split('=')[0] in CHANNELS[kind]:
        return seg
    return None


def required_points(kind):
    return {f'{nm}={x:g}' for nm, x in prof.LEVELS if nm in CHANNELS[kind]} | {'ramp'}


def data_yaw_rows(path):
    """DATA lines of esp32_data.txt (logger.c): (rv_n, yaw_rad); the header line gives the columns."""
    out, cols = [], None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            _, sep, line = raw.rstrip('\n').partition('\t')
            if not sep:
                continue
            if line.startswith('timestamp_us'):
                names = line.split(',')
                cols = (names.index('rv_n'), names.index('yaw_rad'))
                continue
            if cols is None or line.startswith('#'):
                continue
            f = line.split(',')
            try:
                out.append((int(f[cols[0]]), float(f[cols[1]])))
            except (ValueError, IndexError):
                continue
    return out, cols is not None


def check_data_yaw(data_path, callbacks):
    """S28 (spec dòng 100, V3-5): yaw_rad of each DATA tick must equal the yaw of the game-RV callback
    with the same rv_n (callback I field n = ++rv_n when applied), deg → rad wrapped as imu_fusion.c
    does, within the %.4f print of logger.c + float32 (DATA_YAW_TOL)."""
    rows, header = data_yaw_rows(data_path)
    by_n = {i['n_cb']: i for i in callbacks if i['kind'] == 'game_rv' and i['applied'] == 1}
    n_ok = n_bad = n_nomatch = 0
    worst = 0.0
    for rv_n, y in rows:
        if rv_n <= 0:
            continue
        i = by_n.get(rv_n)
        if i is None:
            n_nomatch += 1
            continue
        e = abs(wrap_pi(y - wrap_pi(math.radians(i['v'][2])))) if math.isfinite(y) and math.isfinite(i['v'][2]) else math.inf
        worst = max(worst, e)
        if e <= DATA_YAW_TOL:
            n_ok += 1
        else:
            n_bad += 1
    return dict(header=header, ticks=len(rows), ok=n_ok, bad=n_bad, no_callback=n_nomatch,
                max_err_rad=worst if (n_ok or n_bad) else None, tol=DATA_YAW_TOL)


def check(t_send_ns, mosi, miso, events_path, data_path=None, cal=None):
    """Score every callback line I that has a report identity (S33: also applied = 0, whose value was
    read before the mutex), inside the common run window (S32)."""
    res = dec.decode(t_send_ns, mosi, miso, events_path, cal)
    m = res['esp32_match']
    off = m['epoch_offset_esp32_minus_h7']
    win = m['window_pi_us']
    h7 = {} if off is None else {(r['epoch'] + off, r['kind'], r['n']): r for r in res['_reports']}
    cb = [i for i in res['_esp32_I'] if win is None or win[0] <= i['t_rx'] <= win[1]]
    rows, stats = [], defaultdict(lambda: dict(n=0, applied0=0, fail=0, no_source=0, non_finite=0, max_ratio=0.0))
    covered = defaultdict(set)
    for r in sorted(cb, key=lambda x: x['t_us']):
        s = stats[r['kind']]
        s['n'] += 1
        s['applied0'] += 0 if r['applied'] else 1
        row = dict(t_us=r['t_us'], kind=r['kind'], epoch=r['epoch'], seq=r['seq'], n=r['n'], applied=r['applied'],
                   v1=r['v'][0], v2=r['v'][1], v3=r['v'][2])
        src = h7.get((r['epoch'], r['kind'], r['n'])) if r['n'] is not None else None
        if src is None or src['src_frame'] is None or r['kind'] not in TOL:
            s['no_source'] += 1
            s['fail'] += 1
            row.update(ok=False, note='không ghép được frame nguồn' if r['kind'] in TOL else 'loại report lạ')
            rows.append(row)
            continue
        imu = decode_imu(bytes(mosi[src['src_frame']]))
        exp = expected(r['kind'], imu)
        err, _ = errors(r['kind'], r['v'], exp)
        tol = TOL[r['kind']]
        # S3: NaN/Inf anywhere is a wrong report (max() would drop a NaN on a later axis)
        finite = all(math.isfinite(x) for x in tuple(r['v']) + tuple(exp) + tuple(err))
        worst = max(abs(e) for e in err) if finite else math.inf
        ok = finite and worst <= tol
        s['non_finite'] += 0 if finite else 1
        s['fail'] += 0 if ok else 1
        if finite:
            s['max_ratio'] = max(s['max_ratio'], worst / tol)
        if ok:
            p = stimulus_point(r['kind'], src['src_frame'])
            if p:
                covered[r['kind']].add(p)
        row.update(src_frame=src['src_frame'], src_nf=src['src_nf'], exp1=exp[0], exp2=exp[1], exp3=exp[2],
                   err1=err[0], err2=err[1], err3=err[2], tol=tol, ok=ok, note='' if finite else 'NaN/Inf')
        rows.append(row)
    missing = {k: sorted(required_points(k) - covered[k]) for k in TOL if required_points(k) - covered[k]}
    data = check_data_yaw(data_path, res['_esp32_I']) if data_path else None
    summary = dict(epoch_offset_esp32_minus_h7=off, epoch_note=m['epoch_note'], window_pi_us=win,
                   callback_reports=len(cb),
                   per_kind={k: dict(v, tol=TOL.get(k), unit=UNIT.get(k)) for k, v in stats.items()},
                   fail=sum(s['fail'] for s in stats.values()),
                   coverage={k: sorted(v) for k, v in covered.items()}, coverage_missing=missing,
                   data_yaw=data, decode_problems=dec.problems(res), run_conditions=dec.conditions(res))
    return rows, summary


def exit_code(summary) -> int:
    """1: a report wrong, not matched, NaN, a DATA yaw that differs, or a decoder problem (Z19 stream,
    epoch not anchored …); 3: run not valid or evidence missing (decoder conditions: V4-1, run marks,
    TICK; no callback; schedule point not covered (S18); no esp32_data.txt or no matched tick (S28));
    0 otherwise."""
    d = summary['data_yaw']
    if summary['fail'] or summary['decode_problems'] or (d and d['bad']):
        return 1
    if (summary['run_conditions'] or summary['callback_reports'] == 0 or summary['coverage_missing']
            or d is None or not d['header'] or not d['ok']):
        return 3
    return 0


COLS = ['idx', 't_us', 'kind', 'epoch', 'seq', 'n', 'applied', 'src_frame', 'src_nf', 'v1', 'v2', 'v3',
        'exp1', 'exp2', 'exp3', 'err1', 'err2', 'err3', 'tol', 'ok', 'note']


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T4 (4) (S06 v3.4): giá trị IMU tới callback ESP32')
    ap.add_argument('frames')
    ap.add_argument('esp32_events')
    ap.add_argument('--esp32-data', default=None, help='esp32_data.txt của s06_logger.py (dòng DATA, S28)')
    ap.add_argument('--out', required=True)
    dec.add_cal_args(ap)
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        rows, summary = check(z['t_send_ns'], z['mosi'], z['miso'], a.esp32_events, a.esp32_data,
                              dec.load_cal(z, a))
        for i, r in enumerate(rows):
            r['idx'] = i
        summary['exit_code'] = exit_code(summary)
    except (OSError, KeyError, ValueError) as ex:
        print(f'lỗi đầu vào: {ex}', file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'T4_values.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, 'T4_values_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
