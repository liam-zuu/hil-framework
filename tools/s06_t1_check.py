#!/usr/bin/env python3
"""
s06_t1_check.py — chấm T1 của spec S06 v3.4 (H1 tần số bước plant; V4 frame H7 áp dụng; W3/Y9 frame
trộn và độ phủ trường; Z3 MISO và luồng dựng report IMU). Dùng chung lượt của T2.

Đầu vào: `frames.npz` của `run_hil.py --frame-log` (t_send_ns, MOSI, MISO thô; giải bằng
tools/s06_frames_decode.py), tùy chọn `esp32_events.txt` (ghép report với cổng SPI ESP32) và log serial
H7 (tools/s06_h7_serial.py: bộ đếm Link err, Frame applied/skipped, Link period).

Phần chấm là cả lượt (`--skip-s` mặc định 0; S6: spec chỉ bỏ 10 s đầu cho T2) chia cửa sổ `--window-s`
(mặc định 10 s, [ĐỀ XUẤT CODE]); đủ dài khi có ≥ `--min-s` giây (mặc định 600 s, như T2; N frame = N ms).
Sửa theo Codex Sol vòng 1/2 (Liam duyệt 30/10 "Duyệt, sửa hết"): S1 MISO toàn 0 ngoài `--startup-empty`
frame đầu là lỗi, phải có frame trả lời, REPORT và ≥ 2 TICK; S6 như trên; S12 tuổi trạng thái quy đổi µs
danh định (480 MHz) ra µs thật theo tần số H7 khớp được, luật ba trạng thái; S25 log serial bắt buộc,
bộ đếm lấy bao hai đầu lượt, link nối lại giữa lượt là lỗi; thiếu bằng chứng → `incomplete` (mã 3).
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
Mã thoát: 0 đạt; 1 không đạt; 3 lượt ngắn hơn --min-s, thiếu bằng chứng (không có log serial, không có
esp32_events, bộ đếm không bao hai đầu), tuổi trạng thái chưa kết luận hoặc lượt không hợp lệ (V4-1);
2 lỗi đầu vào.
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
F_NOMINAL_MHZ = 480.0             # SystemCoreClock the H7 divides by (main.c:271)
F_FIT_REL = 1e-4                  # [ĐỀ XUẤT CODE] relative bracket kept around the fitted H7 rate
F_UNKNOWN_MHZ = (478.0, 486.0)    # [ĐỀ XUẤT CODE] H7 rate bracket without a clock fit (measured 483.1, 27/10)
# MOSI fields (rpi5/protocol.py, frame v3): name -> byte offsets
FIELDS = [(f'w{i + 1}', (2 * i, 2 * i + 1)) for i in range(4)] + [
    ('x', (8, 9)), ('y', (10, 11)), ('theta', (12, 13)), ('vx', (14, 15)), ('vy', (16, 17)), ('wz', (18, 19)),
    ('fault_flags', (20,)), ('seq', (21,)), ('gz', (22, 23)), ('vx_ref', (24, 25)), ('vy_ref', (26, 27)),
    ('wz_ref', (28, 29)), ('traj_flags', (30,)), ('ax', (31, 32)), ('ay', (33, 34)), ('az', (35, 36)),
    ('yaw', (37, 38)), ('imu_seq', (39,))]

WINDOW_COLS = ['idx', 'window', 't_start_us', 'pi_frames', 'frames_rx', 'frames_applied', 'seq_gaps',
               'max_state_age_us', 'miso_seq_bad', 'mixed_frames', 'miso_crc_bad']
STEP_COLS = ['idx', 'step', 'crc_sent', 'crc_h7', 'field_change_mask', 'miso_crc_ok']
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


def age_verdict(age_max_us, f_mhz):
    """S12: H7 prints the state age as CYCCNT / (SystemCoreClock / 10⁶) truncated (main.c:313), i.e. in
    nominal µs of 480 MHz. True age ∈ [age, age + 1) × 480 / f_true; f_true from the H7 → RPi5 clock
    fit, or, if there is none, the bracket F_UNKNOWN_MHZ. Three-state rule against 1 000 µs."""
    lo_f, hi_f = (f_mhz * (1 - F_FIT_REL), f_mhz * (1 + F_FIT_REL)) if f_mhz else F_UNKNOWN_MHZ
    lo = age_max_us * F_NOMINAL_MHZ / hi_f
    hi = (age_max_us + 1) * F_NOMINAL_MHZ / lo_f
    v = 'pass' if hi <= AGE_MAX_US else ('fail' if lo > AGE_MAX_US else 'inconclusive')
    return dict(age_max_nominal_us=age_max_us, age_true_lo_us=lo, age_true_hi_us=hi, f_mhz=f_mhz, verdict=v)


def check(t_send_ns, mosi, miso, esp32_events=None, serial_events=None, skip_s=0.0, window_s=10.0,
          min_s=600.0, clock_res_us=0.001, answer_lag=ANSWER_LAG, cal=None, startup_empty=None,
          manifest=None) -> dict:
    mosi = np.asarray(mosi, dtype=np.uint8)
    t_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    res = dec.decode(t_send_ns, mosi, miso, esp32_events, cal)
    rows = res.pop('_frames_rows')
    t0 = t_us[0] + skip_s * 1e6
    sel = np.flatnonzero(t_us >= t0)
    # S6: same length rule as T2 (N frames at 1 ms = N ms)
    scored_s = ((t_us[sel[-1]] - t_us[sel[0]]) + PERIOD_US) / 1e6 if len(sel) >= 2 else 0.0
    # S1: an empty MISO is "no answer from the H7"; only the first `startup_empty` frames of the log
    # (the H7 answers frame N in frame N + answer_lag) may be empty
    n_start = answer_lag if startup_empty is None else startup_empty
    first_sc = int(sel[0]) if len(sel) else 0
    miso_empty = sum(1 for r in rows if r['empty'] and r['j'] >= max(n_start, first_sc))
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
    stream = dict(log_gaps=lg['gaps'], h7_dropped=lg.get('h7_dropped'), ticks=lg.get('ticks'),
                  records=lg['records'])
    serial = None
    if serial_events is not None:
        serial = ser.link_stats(serial_events, ser.run_span(serial_events))
    segs = res['clock']['segments']
    f_fit = float(np.median([s['f_h7_mhz'] for s in segs if s['n'] >= 2])) if any(s['n'] >= 2 for s in segs) else None
    age = age_verdict(tot['age_max_us'], f_fit)
    fails: List[str] = []
    incomplete: List[str] = []         # cannot conclude: missing evidence the spec needs (mã 3)
    if not h1['ok']:
        fails.append(f"H1: chu kỳ trung bình {slope} µs (sai số chuẩn {se}) lệch 1 ms hơn {clock_res_us} µs")
    # S1: evidence that the H7 answered at all
    if miso_empty:
        fails.append(f'{miso_empty} frame MISO toàn 0 ngoài {n_start} frame khởi động (H7 không trả lời)')
    if tot['pi_answered'] <= 0:
        fails.append('không có frame MISO hợp lệ trả lời frame Pi nào')
    if not reps:
        fails.append('không có bản ghi REPORT nào từ H7 (giả lập IMU không dựng report)')
    if not lg.get('ticks') or lg['ticks'] < 2:
        fails.append(f"chỉ có {lg.get('ticks', 0)} bản ghi TICK (luồng log H7 không chạy đủ)")
    if tot['frames_rx'] != tot['pi_answered']:
        fails.append(f"H7 nhận {tot['frames_rx']} frame trên đoạn Pi gửi {tot['pi_answered']}")
    for k, msg in (('seq_gaps', 'frame H7 không áp dụng'), ('miso_seq_bad', 'seq MISO lặp/cũ/sai trễ'),
                   ('mixed', 'frame trộn (CRC MOSI ở H7 khác)'), ('miso_crc_bad', 'MISO sai CRC'),
                   ('unanswered', 'MISO không tìm thấy frame được trả lời')):
        if tot[k]:
            fails.append(f'{tot[k]} {msg}')
    lag_hist = {int(k): v for k, v in res['frames']['answer_lag'].items()}
    if len(lag_hist) == 1 and answer_lag not in lag_hist:
        fails.append(f'mọi frame MISO trả lời trễ {next(iter(lag_hist))} frame, không phải {answer_lag}: '
                     f'kiểm bản H7 đã nạp (fix/h7-frame: 2) hoặc --answer-lag')
    if age['verdict'] == 'fail':
        fails.append(f"tuổi trạng thái lớn nhất {tot['age_max_us']} µs danh định = "
                     f"{age['age_true_lo_us']:.1f}–{age['age_true_hi_us']:.1f} µs thật > {AGE_MAX_US}")
    elif age['verdict'] == 'inconclusive':
        incomplete.append(f"tuổi trạng thái {age['age_true_lo_us']:.1f}–{age['age_true_hi_us']:.1f} µs thật "
                          f"chứa {AGE_MAX_US} µs: chưa kết luận")
    for k, msg in (('imu_crc_bad', 'CRC IMU sai'), ('imu_seq_bad', 'imu_seq sai'),
                   ('no_source_frame', 'report không có frame nguồn'), ('age_over_1', 'tuổi IMU > 1 frame'),
                   ('queue_full', 'report bị bỏ vì hàng đợi đầy'), ('not_at_esp32_spi', 'report H7 không tới cổng SPI ESP32'),
                   ('esp32_without_h7', 'report ESP32 không có ở H7')):
        if imu.get(k):
            fails.append(f'{imu[k]} {msg}')
    if stream['log_gaps'] or stream['h7_dropped']:
        fails.append(f"luồng log H7 thiếu (hở {stream['log_gaps']}, H7 bỏ {stream['h7_dropped']}): Z19")
    # S81 (vòng 5): the frame log itself, as in T3b, T4, T6
    fails += dec.manifest_check(manifest, t_send_ns, mosi, miso)
    if manifest is None:
        incomplete.append('frames.npz không có manifest: không chứng minh được log frame đủ')
    rp = res['reports']
    if rp.get('seq_flags'):
        fails.append(f"số thứ tự report H7 bất thường {rp['seq_flags']}")
    if m is not None:
        if m['epoch_offset_esp32_minus_h7'] is None:
            fails.append(f"ghép report với ESP32: {m['epoch_note']}")
        if m.get('esp32_seq_flags'):
            fails.append(f"số thứ tự report ở cổng SPI ESP32 bất thường {m['esp32_seq_flags']}")
        if m['window_pi_us'] is None:
            incomplete.append('esp32_events thiếu #run_start/#run_end: không chấm ghép report theo khung lượt')
        if m.get('pad_note'):
            incomplete.append(f"S38: {m['pad_note']}: report ở đầu/cuối lượt có thể thiếu bản đối chiếu")
        if m.get('udp_note'):
            incomplete.append(m['udp_note'])
        rz = res.get('resets_near_run_start')
        if rz is not None and not rz['ok']:
            incomplete.append(f"reset trong ±1 s quanh #run_start (H7 {rz['h7']}, ESP32 {rz['esp32']}): "
                              f"lượt không hợp lệ (V4-1), đo lại")
    else:
        incomplete.append('không có --esp32-events: không kiểm được report tới cổng SPI ESP32 (Z3 (2))')
    # S25: the serial log of the run is required (spec T1: dòng Link period, bộ đếm của H7)
    if serial is None:
        incomplete.append('không có --h7-serial: không kiểm được bộ đếm Link/Frame của H7')
    else:
        fails += ser.link_fails(serial)
        if not (serial['bounded_start'] and serial['bounded_end']):
            incomplete.append('log serial không có dòng Link/Frame ngay trước link up và ngay sau link lost '
                              'cuối: bộ đếm không bao hai đầu lượt')
    windows = []
    for i, w in enumerate(sorted(wins)):
        W = {k: v for k, v in wins[w].items() if not k.startswith('_')}
        W['idx'] = i
        windows.append(W)
    steps = [dict(idx=i, step=int(k), crc_sent=dec.crc16(bytes(mosi[k])), crc_h7=steps_crc_h7.get(int(k), ''),
                  field_change_mask=int(masks[k]), miso_crc_ok=int(k) in steps_crc_h7) for i, k in enumerate(sel)]
    tot['miso_empty'] = miso_empty
    if fails:
        verdict = 'fail'
    elif scored_s < min_s:
        verdict = 'short'
    elif incomplete:
        verdict = 'incomplete'
    else:
        verdict = 'pass'
    out = dict(h1=h1, frames=tot, answer_lag_expected=answer_lag, answer_lag_hist=lag_hist,
               coverage_pct=coverage, imu=imu, stream=stream, serial=serial, state_age=age,
               clock=dict(calibrated=res['clock']['calibrated'], f_h7_mhz=f_fit),
               scored_s=scored_s, enough=scored_s >= min_s, fails=fails, incomplete=incomplete,
               verdict=verdict)
    out['_windows'], out['_steps'], out['_reports'] = windows, steps, reps
    return out


def _json(o):
    """numpy scalars -> Python numbers for json"""
    return o.item() if hasattr(o, 'item') else str(o)


def exit_code(out) -> int:
    return {'pass': 0, 'fail': 1, 'short': 3, 'incomplete': 3}[out['verdict']]


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
    ap.add_argument('--skip-s', type=float, default=0.0,
                    help='bỏ đầu lượt, giây (mặc định 0: spec chỉ bỏ 10 s đầu cho phần chấm T2)')
    ap.add_argument('--window-s', type=float, default=10.0)
    ap.add_argument('--min-s', type=float, default=600.0)
    ap.add_argument('--clock-res-us', type=float, default=0.001)
    ap.add_argument('--answer-lag', type=int, default=ANSWER_LAG)
    ap.add_argument('--startup-empty', type=int, default=None,
                    help='số frame đầu log được phép MISO toàn 0 (mặc định = --answer-lag)')
    dec.add_cal_args(ap)
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        serial = ser.read_log(a.h7_serial) if a.h7_serial else None
        out = check(z['t_send_ns'], z['mosi'], z['miso'], a.esp32_events, serial, a.skip_s, a.window_s,
                    a.min_s, a.clock_res_us, a.answer_lag, dec.load_cal(z, a), a.startup_empty,
                    dec.load_manifest(z))
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
