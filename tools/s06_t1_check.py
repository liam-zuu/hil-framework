#!/usr/bin/env python3
"""
s06_t1_check.py — chấm T1 của spec S06 v3.4 (H1 tần số bước plant; V4 frame H7 áp dụng; W3/Y9 frame
trộn và độ phủ trường; Z3 MISO và luồng dựng report IMU). Dùng chung lượt của T2.

Đầu vào: `frames.npz` của `run_hil.py --frame-log` (t_send_ns, MOSI, MISO thô; giải bằng
tools/s06_frames_decode.py), tùy chọn `esp32_events.txt` (ghép report với cổng SPI ESP32) và log serial
H7 (tools/s06_h7_serial.py: bộ đếm Link err, Frame applied/skipped, Link period).

Phần chấm bỏ `--skip-s` đầu (mặc định 10 s, như T2) và chia cửa sổ `--window-s` (mặc định 10 s,
[ĐỀ XUẤT CODE]); đủ dài khi có ≥ `--min-s` giây được chấm (mặc định 600 s, như T2).
  * H1 (đồng hồ RPi5, giờ gửi frame): chu kỳ trung bình = độ dốc hồi quy t_gửi theo chỉ số bước
    (không nhạy với dao động ở hai đầu, vẫn thấy vòng lặp trôi nhịp); đạt khi |dốc − 1 000 µs| ≤ độ
    phân giải đồng hồ (`--clock-res-us`, mặc định 0,001 µs = 1 ns của monotonic_ns, giả định) + 3 sai số
    chuẩn của độ dốc. Báo kèm (t_cuối − t_đầu)/(N − 1), khoảng lớn nhất, số khoảng > 1,5 ms.
  * H7 nhận đủ frame: Δ(số frame H7, MISO byte 18–19 mở vòng) = Δ(frame Pi được trả lời) trên cùng đoạn.
  * H7 áp dụng đủ frame (V4): hai frame MISO liên tiếp (đều CRC đúng) trả lời hai frame Pi liên tiếp;
    hụt (cách > 1) cộng vào `seq_gaps`; lặp (cách 0) và trễ trả lời ≠ 2 frame (`--answer-lag`, bản
    `fix/h7-frame`: N − 2) cộng vào `miso_seq_bad`. Tuổi trạng thái (MISO byte 20–21) lớn nhất ≤ 1 000 µs.
  * Frame trộn (W3): CRC MOSI do H7 tính trên bản sao vòng chính (MISO byte 10–11) khác CRC của frame Pi
    được trả lời. Độ phủ (Y9): % bước mỗi trường MOSI khác bước trước; trộn chỉ thấy trên trường có đổi.
  * MISO (Z3 (1)): CRC MISO sai = MISO trộn hoặc hỏng.
  * Luồng dựng report (Z3 (2)): CRC IMU sai, imu_seq sai, không có frame nguồn, tuổi IMU > 1 frame,
    report bị bỏ vì hàng đợi đầy; có `--esp32-events`: report không ghép được (thiếu một phía) > 0.
  * Luồng đo (§4.3 điều 6, Z19): luồng log H7 hở hoặc H7 bỏ bản ghi vì đầy → thất bại.
  * Log serial H7 (nếu có): Link err tăng, Frame skipped tăng trong lượt → không đạt; Link period báo kèm.

Đạt khi mọi mục trên đạt. Diễn giải của CODE (spec chưa ghi, cần THESIS xác nhận khi khóa): cửa sổ 10 s;
tiêu chí H1 = độ dốc hồi quy trong 1 ns + 3 sai số chuẩn; mất frame MISO (CRC sai) không tính hụt áp dụng;
`seq_gaps` chỉ tính trên cặp frame MISO liền nhau.

Usage:
  python tools/s06_t1_check.py <frames.npz> --out <thư mục> [--esp32-events esp32_events.txt]
         [--h7-serial h7_serial.log] [--skip-s 10] [--window-s 10] [--min-s 600]
Mã thoát: 0 đạt; 1 không đạt; 3 lượt ngắn hơn --min-s; 2 lỗi đầu vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s06_frames_decode as dec  # noqa: E402
import s06_h7_serial as ser  # noqa: E402

PERIOD_US = 1000.0
GAP_US = 1500.0
AGE_MAX_US = 1000
ANSWER_LAG = 2
# MOSI fields (rpi5/protocol.py, frame v3): name -> byte offsets
FIELDS = [(f'w{i + 1}', (2 * i, 2 * i + 1)) for i in range(4)] + [
    ('x', (8, 9)), ('y', (10, 11)), ('theta', (12, 13)), ('vx', (14, 15)), ('vy', (16, 17)), ('wz', (18, 19)),
    ('fault_flags', (20,)), ('seq', (21,)), ('gz', (22, 23)), ('vx_ref', (24, 25)), ('vy_ref', (26, 27)),
    ('wz_ref', (28, 29)), ('traj_flags', (30,)), ('ax', (31, 32)), ('ay', (33, 34)), ('az', (35, 36)),
    ('yaw', (37, 38)), ('imu_seq', (39,))]

WINDOW_COLS = ['idx', 'window', 't_start_us', 'pi_frames', 'frames_rx', 'frames_applied', 'seq_gaps',
               'max_state_age_us', 'miso_seq_bad', 'mixed_frames', 'miso_crc_bad']
STEP_COLS = ['step', 'crc_sent', 'crc_h7', 'field_change_mask', 'miso_crc_ok']
IMU_COLS = ['idx', 'report_type', 'rpt_seq', 'queued', 'qfull', 'imu_seq_used', 'n_frames_src', 'n_frames_latest',
            'imu_crc_h7', 'imu_crc_pi', 'imu_crc_ok', 'imu_age_frames', 'cyccnt', 'esp32_spi']


def change_masks(mosi: np.ndarray) -> np.ndarray:
    """Bit k = trường FIELDS[k] khác bước trước (bước 0: 0)."""
    m = np.zeros(len(mosi), dtype=np.int64)
    for k, (_, ofs) in enumerate(FIELDS):
        cur = mosi[:, list(ofs)]
        diff = np.any(cur[1:] != cur[:-1], axis=1)
        m[1:] |= diff.astype(np.int64) << k
    return m


def check(t_send_ns, mosi, miso, esp32_events=None, serial_events=None, skip_s=10.0, window_s=10.0,
          min_s=600.0, clock_res_us=0.001, answer_lag=ANSWER_LAG) -> dict:
    mosi = np.asarray(mosi, dtype=np.uint8)
    t_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    res = dec.decode(t_send_ns, mosi, miso, esp32_events)
    rows = res.pop('_frames_rows')
    t0 = t_us[0] + skip_s * 1e6
    sel = np.flatnonzero(t_us >= t0)
    scored_s = (t_us[sel[-1]] - t_us[sel[0]]) / 1e6 if len(sel) >= 2 else 0.0
    # H1
    per = np.diff(t_us[sel]) if len(sel) >= 2 else np.array([])
    mean = slope = se = None
    if len(sel) >= 3:
        k = sel - sel[0]
        y = t_us[sel] - t_us[sel[0]]
        slope, icept = np.polyfit(k, y, 1)
        resid = y - (icept + slope * k)
        se = float(np.sqrt(np.sum(resid ** 2) / (len(k) - 2) / np.sum((k - k.mean()) ** 2)))
        slope = float(slope)
        mean = float((t_us[sel[-1]] - t_us[sel[0]]) / (len(sel) - 1))
    h1 = dict(frames=int(len(sel)), mean_period_us=slope, slope_se_us=se, endpoint_mean_us=mean,
              max_period_us=float(per.max()) if len(per) else None,
              gaps_over_1p5ms=int(np.sum(per > GAP_US)), clock_res_us=clock_res_us,
              ok=slope is not None and abs(slope - PERIOD_US) <= clock_res_us + 3.0 * se)
    # per frame (MISO answers) in the scored part
    good = [r for r in rows if r['j'] >= (sel[0] if len(sel) else 0) and not r['empty']]
    win_us = window_s * 1e6
    wins = {}
    steps_crc_h7 = {}
    tot = dict(frames_rx=0, pi_answered=0, seq_gaps=0, miso_seq_bad=0, mixed=0, miso_crc_bad=0,
               age_max_us=0, unanswered=0)
    prev = None
    first_ok = last_ok = None
    for r in good:
        w = int((t_us[r['j']] - t0) // win_us)
        W = wins.setdefault(w, dict(window=w, t_start_us=t0 + w * win_us, pi_frames=0, frames_rx=0,
                                    frames_applied=0, seq_gaps=0, max_state_age_us=0, miso_seq_bad=0,
                                    mixed_frames=0, miso_crc_bad=0, _nf=[], _ans=[]))
        if not r['crc_ok']:
            W['miso_crc_bad'] += 1
            tot['miso_crc_bad'] += 1
            prev = None
            continue
        if r['answered'] is None:
            tot['unanswered'] += 1
            W['miso_seq_bad'] += 1
            tot['miso_seq_bad'] += 1
            prev = None
            continue
        a = r['answered']
        steps_crc_h7[a] = dec.u16(bytes(miso[r['j']]), dec.MISO_MOSI_CRC)
        W['_ans'].append(a)
        W['_nf'].append(r['rx_nf'])
        W['max_state_age_us'] = max(W['max_state_age_us'], r['age_us'])
        tot['age_max_us'] = max(tot['age_max_us'], r['age_us'])
        if not r['mosi_crc_ok']:
            W['mixed_frames'] += 1
            tot['mixed'] += 1
        if r['j'] - a != answer_lag:
            W['miso_seq_bad'] += 1
            tot['miso_seq_bad'] += 1
        if prev is not None and r['j'] == prev['j'] + 1:
            d = a - prev['answered']
            if d == 0:
                W['miso_seq_bad'] += 1
                tot['miso_seq_bad'] += 1
            elif d > 1:
                W['seq_gaps'] += d - 1
                tot['seq_gaps'] += d - 1
        prev = r
        first_ok = first_ok or r
        last_ok = r
    for W in wins.values():
        if W['_ans']:
            W['frames_applied'] = len(set(W['_ans']))
            W['pi_frames'] = W['_ans'][-1] - W['_ans'][0] + 1
            W['frames_rx'] = W['_nf'][-1] - W['_nf'][0] + 1
    if first_ok is not None:
        tot['frames_rx'] = last_ok['rx_nf'] - first_ok['rx_nf']
        tot['pi_answered'] = last_ok['answered'] - first_ok['answered']
    # field coverage over the scored steps
    masks = change_masks(mosi)
    sm = masks[sel[1:]] if len(sel) >= 2 else np.array([], dtype=np.int64)
    coverage = {name: (float(np.mean((sm >> k) & 1)) * 100.0 if len(sm) else None)
                for k, (name, _) in enumerate(FIELDS)}
    # IMU builds
    reps = res['_reports']
    imu = dict(n=len(reps), imu_crc_bad=sum(1 for r in reps if r.get('imu_crc_ok') is False),
               imu_seq_bad=sum(1 for r in reps if r.get('imu_seq_ok') is False),
               no_source_frame=sum(1 for r in reps if r['src_frame'] is None),
               age_over_1=sum(1 for r in reps if r['imu_age_frames'] > 1),
               queue_full=sum(1 for r in reps if r['qfull']))
    m = res.get('esp32_match')
    if m:
        imu.update(not_at_esp32_spi=m['not_at_esp32_spi'], esp32_without_h7=m['esp32_without_h7'])
    lg = res['log']
    stream = dict(log_gaps=lg['gaps'], h7_dropped=lg.get('h7_dropped'))
    serial = None
    if serial_events is not None:
        serial = ser.link_stats(serial_events, ser.run_span(serial_events))
    fails: List[str] = []
    if not h1['ok']:
        fails.append(f"H1: chu kỳ trung bình {slope} µs (sai số chuẩn {se}) lệch 1 ms hơn {clock_res_us} µs")
    if tot['frames_rx'] != tot['pi_answered']:
        fails.append(f"H7 nhận {tot['frames_rx']} frame trên đoạn Pi gửi {tot['pi_answered']}")
    for k, msg in (('seq_gaps', 'frame H7 không áp dụng'), ('miso_seq_bad', 'seq MISO lặp/cũ/sai trễ'),
                   ('mixed', 'frame trộn (CRC MOSI ở H7 khác)'), ('miso_crc_bad', 'MISO sai CRC'),
                   ('unanswered', 'MISO không tìm thấy frame được trả lời')):
        if tot[k]:
            fails.append(f'{tot[k]} {msg}')
    if tot['age_max_us'] > AGE_MAX_US:
        fails.append(f"tuổi trạng thái lớn nhất {tot['age_max_us']} µs > {AGE_MAX_US}")
    for k, msg in (('imu_crc_bad', 'CRC IMU sai'), ('imu_seq_bad', 'imu_seq sai'),
                   ('no_source_frame', 'report không có frame nguồn'), ('age_over_1', 'tuổi IMU > 1 frame'),
                   ('queue_full', 'report bị bỏ vì hàng đợi đầy'), ('not_at_esp32_spi', 'report H7 không tới cổng SPI ESP32'),
                   ('esp32_without_h7', 'report ESP32 không có ở H7')):
        if imu.get(k):
            fails.append(f'{imu[k]} {msg}')
    if stream['log_gaps'] or stream['h7_dropped']:
        fails.append(f"luồng log H7 thiếu (hở {stream['log_gaps']}, H7 bỏ {stream['h7_dropped']}): Z19")
    if serial:
        if serial.get('err'):
            fails.append(f"H7 Link err tăng {serial['err']}")
        if serial.get('skipped'):
            fails.append(f"H7 Frame skipped tăng {serial['skipped']}")
    windows = []
    for i, w in enumerate(sorted(wins)):
        W = {k: v for k, v in wins[w].items() if not k.startswith('_')}
        W['idx'] = i
        windows.append(W)
    steps = [dict(step=int(k), crc_sent=dec.crc16(bytes(mosi[k])), crc_h7=steps_crc_h7.get(int(k), ''),
                  field_change_mask=int(masks[k]), miso_crc_ok=int(k) in steps_crc_h7) for k in sel]
    out = dict(h1=h1, frames=tot, coverage_pct=coverage, imu=imu, stream=stream, serial=serial,
               scored_s=scored_s, enough=scored_s >= min_s, fails=fails,
               verdict='fail' if fails else ('short' if scored_s < min_s else 'pass'))
    out['_windows'], out['_steps'], out['_reports'] = windows, steps, reps
    return out


def _json(o):
    """numpy scalars -> Python numbers for json"""
    return o.item() if hasattr(o, 'item') else str(o)


def exit_code(out) -> int:
    return {'pass': 0, 'fail': 1, 'short': 3}[out['verdict']]


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T1 (spec S06 v3.4)')
    ap.add_argument('frames')
    ap.add_argument('--out', required=True)
    ap.add_argument('--esp32-events', default=None)
    ap.add_argument('--h7-serial', default=None)
    ap.add_argument('--skip-s', type=float, default=10.0)
    ap.add_argument('--window-s', type=float, default=10.0)
    ap.add_argument('--min-s', type=float, default=600.0)
    ap.add_argument('--clock-res-us', type=float, default=0.001)
    ap.add_argument('--answer-lag', type=int, default=ANSWER_LAG)
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        serial = ser.read_log(a.h7_serial) if a.h7_serial else None
        out = check(z['t_send_ns'], z['mosi'], z['miso'], a.esp32_events, serial, a.skip_s, a.window_s,
                    a.min_s, a.clock_res_us, a.answer_lag)
    except (OSError, KeyError, ValueError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, 'T1.csv'), out['_windows'], WINDOW_COLS)
    write_csv(os.path.join(a.out, 'T1_steps.csv'), out['_steps'], STEP_COLS)
    imu_rows = [dict(idx=i, report_type=r['kind'], rpt_seq=r['rpt_seq'], queued=int(r['queued']), qfull=int(r['qfull']),
                     imu_seq_used=r['imu_seq'], n_frames_src=r['src_nf'], n_frames_latest=r['nf_latest'],
                     imu_crc_h7=r['imu_crc_h7'], imu_crc_pi=r.get('imu_crc_pi', ''), imu_crc_ok=r.get('imu_crc_ok', ''),
                     imu_age_frames=r['imu_age_frames'], cyccnt=r['cyccnt'], esp32_spi=r.get('esp32_spi', ''))
                for i, r in enumerate(out['_reports'])]
    write_csv(os.path.join(a.out, 'T1_imu_builds.csv'), imu_rows, IMU_COLS)
    summ = {k: v for k, v in out.items() if not k.startswith('_')}
    summ['exit_code'] = exit_code(out)
    with open(os.path.join(a.out, 'T1_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=_json)
    print(json.dumps(summ, default=_json))
    return summ['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
