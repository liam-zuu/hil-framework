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
`s06_load` in giá trị callback bằng `%.9g`, nên số hạng làm tròn in ≈ 0.
Report có ở cổng SPI mà không tới callback (thư viện ghi đè) không có giá trị: không chấm ở đây
(luật "đếm ở cổng SPI", T1). Dòng `I` không ghép được frame nguồn = không đạt.

Usage:
  python tools/s06_t4_values.py <frames.npz> <esp32_events.txt> --out <thư mục>
Mã thoát: 0 mọi report đạt; 1 có report sai hoặc không ghép được; 2 lỗi đầu vào.
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
from protocol import decode_imu  # noqa: E402

TOL = {'accel': 2.55e-3, 'gyro': 1.51e-3, 'game_rv': 2.2e-4}
UNIT = {'accel': 'm/s^2', 'gyro': 'rad/s', 'game_rv': 'rad'}


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def callback_rows(path):
    """Dòng I tới ứng dụng (applied = 1), kèm giá trị."""
    out = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            _, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line.startswith('I,'):
                continue
            p = line.split(',')
            if len(p) != 12:
                continue
            try:
                r = dict(t_us=int(p[2]), kind=dec.ESP32_KIND.get(int(p[3]), p[3]), applied=int(p[4]),
                         seq=int(p[5]), epoch=int(p[6]), v=(float(p[9]), float(p[10]), float(p[11])))
            except ValueError:
                continue
            if r['applied'] == 1:
                out.append(r)
    return out


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


def check(t_send_ns, mosi, miso, events_path):
    res = dec.decode(t_send_ns, mosi, miso, events_path)
    off = res['esp32_match']['epoch_offset_esp32_minus_h7']
    h7 = {(r['epoch'] + off, r['kind'], r['n']): r for r in res['_reports']}
    cb = callback_rows(events_path)
    dec.seq_keys(cb)                                  # adds n (seq unwrapped per epoch, kind)
    rows, stats = [], defaultdict(lambda: dict(n=0, fail=0, no_source=0, max_ratio=0.0))
    for r in sorted(cb, key=lambda x: x['t_us']):
        s = stats[r['kind']]
        s['n'] += 1
        row = dict(t_us=r['t_us'], kind=r['kind'], epoch=r['epoch'], seq=r['seq'], n=r['n'],
                   v1=r['v'][0], v2=r['v'][1], v3=r['v'][2])
        src = h7.get((r['epoch'], r['kind'], r['n']))
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
        worst = max(abs(e) for e in err)
        ok = worst <= tol
        s['fail'] += 0 if ok else 1
        s['max_ratio'] = max(s['max_ratio'], worst / tol)
        row.update(src_frame=src['src_frame'], src_nf=src['src_nf'], exp1=exp[0], exp2=exp[1], exp3=exp[2],
                   err1=err[0], err2=err[1], err3=err[2], tol=tol, ok=ok, note='')
        rows.append(row)
    summary = dict(epoch_offset_esp32_minus_h7=off, callback_reports=len(cb),
                   per_kind={k: dict(v, tol=TOL.get(k), unit=UNIT.get(k)) for k, v in stats.items()},
                   fail=sum(s['fail'] for s in stats.values()),
                   decode_problems=dec.problems(res))
    return rows, summary


COLS = ['t_us', 'kind', 'epoch', 'seq', 'n', 'src_frame', 'src_nf', 'v1', 'v2', 'v3', 'exp1', 'exp2', 'exp3',
        'err1', 'err2', 'err3', 'tol', 'ok', 'note']


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T4 (4) (S06 v3.4): giá trị IMU tới callback ESP32')
    ap.add_argument('frames')
    ap.add_argument('esp32_events')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        rows, summary = check(z['t_send_ns'], z['mosi'], z['miso'], a.esp32_events)
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
    if summary['callback_reports'] == 0:
        print('không có dòng I nào tới callback', file=sys.stderr)
        return 1
    return 1 if summary['fail'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
