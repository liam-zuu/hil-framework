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

Sửa theo Codex Sol vòng 2/2 (Liam 30/10 "Sửa P1 rồi vòng 3"):
  * S38: không bớt 0,5 s mỗi đầu nữa: chấm MỌI callback trong #run_start..#run_end (giờ ESP32); frame Pi
    và dòng J/I phải phủ lượt ± 1 s (PAD_US, bộ giải mã), thiếu → mã 3.
  * S39: DATA phải đủ mọi log_seq trong [first_data_seq; last_data_seq] của lượt; thiếu, trùng, dòng không
    đọc được, hay tick trỏ tới rv_n không có callback → mã 1 (không còn "một tick khớp là đủ").
  * S42: độ phủ chỉ tính khi MOSI thật bằng đúng lịch khóa (so int16 từng frame của lượt và frame nguồn);
    khác → mã 3 (Pi không phát lịch T4, ví dụ quên `--imu-profile t4`).

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
from protocol import N_STATES, decode_imu, encode_state  # noqa: E402

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
    """DATA lines of esp32_data.txt (logger.c): (log_seq, rv_n, yaw_rad); the header line gives the columns.
    A line that does not parse is counted (`bad_lines`), not dropped silently (S39)."""
    out, cols, bad = [], None, 0
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            _, sep, line = raw.rstrip('\n').partition('\t')
            if not sep:
                continue
            if line.startswith('timestamp_us'):
                names = line.split(',')
                cols = (names.index('log_seq'), names.index('rv_n'), names.index('yaw_rad'))
                continue
            if cols is None or line.startswith('#'):
                continue
            f = line.split(',')
            try:
                out.append((int(f[cols[0]]), int(f[cols[1]]), float(f[cols[2]])))
            except (ValueError, IndexError):
                bad += 1
    return out, cols is not None, bad


def check_data_yaw(data_path, callbacks, seq_span=None):
    """S28 (spec dòng 100, V3-5): yaw_rad of each DATA tick must equal the yaw of the game-RV callback
    with the same rv_n (callback I field n = ++rv_n when applied), deg → rad wrapped as imu_fusion.c
    does, within the %.4f print of logger.c + float32 (DATA_YAW_TOL).
    S39: every tick of the run is required, not one: `seq_span` = (first_data_seq, last_data_seq) of the
    producer (#run_start, #run_end of s06_load); a log_seq of that span missing or repeated, a line that
    does not parse, or a tick whose rv_n has no callback line is lost data / a wrong value (Z19), not a
    note. The events file covers the run ± PAD_US (bộ giải mã), so the callback a tick points at is there.
    rv_n ≤ 0 inside the run (no game RV applied yet) cannot be compared: counted apart (mã 3)."""
    rows, header, bad_lines = data_yaw_rows(data_path)
    if seq_span is not None:
        lo, hi = seq_span
        rows = [r for r in rows if lo <= r[0] <= hi]
        seen = [r[0] for r in rows]
        missing = (hi - lo + 1) - len(set(seen)) if hi >= lo else 0
        dups = len(seen) - len(set(seen))
    else:
        missing = dups = None
    by_n = {i['n_cb']: i for i in callbacks if i['kind'] == 'game_rv' and i['applied'] == 1}
    n_ok = n_bad = n_nomatch = n_norv = 0
    worst = 0.0
    for _, rv_n, y in rows:
        if rv_n <= 0:
            n_norv += 1
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
    return dict(header=header, seq_span=seq_span, ticks=len(rows), seq_missing=missing, seq_dups=dups,
                bad_lines=bad_lines, ok=n_ok, bad=n_bad, no_callback=n_nomatch, no_rv=n_norv,
                max_err_rad=worst if (n_ok or n_bad) else None, tol=DATA_YAW_TOL)


def data_seq_span(text):
    """(first_data_seq, last_data_seq) of the run from the s06_load marks; None if not present."""
    try:
        return (int(text['run_start']['kv']['first_data_seq']), int(text['run_end']['kv']['last_data_seq']))
    except (KeyError, ValueError):
        return None


def stimulus_check(t_send_us, mosi, window, src_frames, period_us=1000.0):
    """S42: the coverage names a schedule point from the frame index, so first prove that the MOSI the Pi
    really sent IS the locked schedule: every frame of the run (Pi clock `window`) and every source frame
    of a scored report must carry exactly the int16 values of t4_imu_profile.sample(step × period)
    (same encoder as hil_node.py). Returns (frames checked, frames that differ, first differing frame)."""
    ks = set(src_frames)
    if window is not None:
        t = np.asarray(t_send_us, dtype=float)
        ks |= set(np.nonzero((t >= window[0]) & (t <= window[1]))[0].tolist())
    zero = [0.0] * N_STATES
    bad, first = 0, None
    for k in sorted(ks):
        want = decode_imu(encode_state(zero, imu=prof.sample(k * period_us * 1e-6)))[:5]
        if tuple(decode_imu(bytes(mosi[k]))[:5]) != tuple(want):
            bad += 1
            first = k if first is None else first
    return len(ks), bad, first


def check(t_send_ns, mosi, miso, events_path, data_path=None, cal=None):
    """Score every callback line I that has a report identity (S33: also applied = 0, whose value was
    read before the mutex), S38: every callback inside #run_start..#run_end on the ESP32 clock (the same
    clock the I lines carry; no edge left out)."""
    res = dec.decode(t_send_ns, mosi, miso, events_path, cal)
    m = res['esp32_match']
    off = m['epoch_offset_esp32_minus_h7']
    win = m['window_pi_us']
    esp = res.get('esp32_run_esp_us')
    h7 = {} if off is None else {(r['epoch'] + off, r['kind'], r['n']): r for r in res['_reports']}
    cb = [i for i in res['_esp32_I'] if esp is None or esp[0] <= i['t_us'] <= esp[1]]
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
    srcs = [r['src_frame'] for r in rows if r.get('src_frame') is not None]
    n_chk, n_diff, k_diff = stimulus_check(np.asarray(t_send_ns, dtype=np.int64) / 1e3, mosi, win, srcs)
    text = res.get('_esp32_text', {})
    data = check_data_yaw(data_path, res['_esp32_I'], data_seq_span(text)) if data_path else None
    summary = dict(epoch_offset_esp32_minus_h7=off, epoch_note=m['epoch_note'], window_pi_us=win,
                   window_esp32_us=esp, callback_reports=len(cb),
                   stimulus=dict(frames_checked=n_chk, frames_differ=n_diff, first_differ=k_diff),
                   per_kind={k: dict(v, tol=TOL.get(k), unit=UNIT.get(k)) for k, v in stats.items()},
                   fail=sum(s['fail'] for s in stats.values()),
                   coverage={k: sorted(v) for k, v in covered.items()}, coverage_missing=missing,
                   data_yaw=data, decode_problems=dec.problems(res), run_conditions=dec.conditions(res))
    return rows, summary


def exit_code(summary) -> int:
    """1: a report wrong, not matched, NaN, a decoder problem (Z19 stream, epoch not anchored …), or a DATA
    tick lost / wrong / pointing at no callback / not parsed (S39); 3: run not valid or evidence missing
    (decoder conditions: V4-1, run marks, TICK, ± PAD_US cover (S38); no ESP32 run span; no callback;
    the MOSI is not the locked schedule (S42) or a schedule point not covered (S18); no esp32_data.txt,
    no data seq span in the run marks, or ticks without a game RV yet (S28, S39)); 0 otherwise."""
    d = summary['data_yaw']
    if summary['fail'] or summary['decode_problems']:
        return 1
    if d and (d['bad'] or d['no_callback'] or d['bad_lines'] or d['seq_missing'] or d['seq_dups']):
        return 1
    if (summary['run_conditions'] or summary['window_esp32_us'] is None or summary['callback_reports'] == 0
            or summary['stimulus']['frames_differ'] or summary['coverage_missing']):
        return 3
    if d is None or not d['header'] or d['seq_span'] is None or d['no_rv'] or not d['ok']:
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
