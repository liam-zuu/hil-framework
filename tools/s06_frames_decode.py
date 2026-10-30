#!/usr/bin/env python3
"""
s06_frames_decode.py — giải mã `frames.npz` (run_hil.py --frame-log: t_send_ns, MOSI, MISO thô)
cho S06 v3.4, với H7 MISO bản 4 (hil-framework feat/s06-h7 3235ec5; thiết kế
03_CODE/docs/2026-10-29_Thiet_ke_H7_S06.md mục 2–3 của repo luận văn).

Bốn phần:
  1. CRC (T1, Y9, Z3): CRC-16/CCITT-FALSE (= binascii.crc_hqx(data, 0xFFFF)).
     * MISO: byte 8–9 (LE) = CRC trên byte 0–7 và 10–39 của chính frame MISO.
     * MOSI: byte 10–11 của MISO (LE) = CRC trên 40 byte MOSI thô của frame được trả lời; frame
       được trả lời tìm theo `seq` (MISO byte 17 = MOSI byte 21), trong 1–4 frame trước; báo
       phân bố độ trễ trả lời (bản `fix/h7-frame`: luôn 2).
     * Frame MISO toàn 0 (H7 chưa dựng lần nào, đầu lượt) đếm riêng, không tính CRC sai.
  2. Luồng log H7 (§4.3 điều 6): byte 22 loại, 23–24 chỉ số liên tục 16 bit, 25–39 nội dung; lấy
     từ frame MISO có CRC đúng; bỏ trùng theo chỉ số (frame lặp); mở vòng 16 bit; báo chỗ hở và
     số bản ghi H7 bỏ vì đầy (bản ghi TICK cuối).
  3. Khớp đồng hồ H7 → RPi5: cặp (t_send của frame được trả lời, CYCCNT lúc callback = MISO byte
     12–15, mở vòng 32 bit theo thứ tự frame), bình phương tối thiểu t_pi = a + b·cyc theo từng
     đoạn; đoạn cắt ở bản ghi REARM và ở khoảng trống > 0,5 s. Báo f_H7 = 1/b, phần dư lớn nhất.
     Kiểm thêm: hiệu `_cyc_total` giữa hai TICK = hiệu CYCCNT đã mở vòng.
     Mọi bản ghi có `cyc` được mở vòng quanh CYCCNT của frame tham chiếu (frame `nf` trong bản ghi
     nếu có, không thì frame mang bản ghi) rồi quy ra giờ Pi.
  4. Report IMU (T1, T4, T8; khóa report Z3): bản ghi REPORT (0x10 accel, 0x11 gyro, 0x12 game RV)
     → `rpt_seq`, epoch, `imu_seq`, `src_nf`, `imu_crc`. Kiểm `imu_crc` với CRC 10 byte IMU thô
     (MOSI 22–23, 31–38) của frame nguồn và `imu_seq` với MOSI byte 39 (frame nguồn tìm theo
     `rx_nf` → frame Pi). Ghép với ESP32 (dòng J ở cổng SPI, dòng I ở callback của
     `esp32_events.txt`) theo khóa (epoch, loại, số thứ tự mở vòng trong epoch); độ lệch epoch
     ESP32 − H7 chọn theo số cặp khớp nhiều nhất và báo ra.

Thêm (việc (5)): bản ghi 0x35/0x36 (mỗi lần gọi `_start_wheel`/`_stop`: bánh, CNT, số lần gọi, CYCCNT)
→ danh sách lần gọi theo bánh trên giờ Pi cho tools/t3b_score.py; số lần gọi hở = thiếu bản ghi (lần
gọi mất sau lần gọi cuối cùng của một loại thì không thấy được).
Giờ Pi của sự kiện H7 (S10, spec dòng 111): khớp CYCCNT với mốc xung đồng bộ (`t_pulse0_ns`/`t_pulse1_ns`
của frames.npz, ghi ngay trước/sau lệnh nâng chân) rồi cộng d_1w trung vị; bất định theo công thức spec.
Thiếu mốc xung hoặc thiếu d_1w/δ_H/t_s (`--d1w-us` …): khớp theo giờ GỬI frame, trễ gửi → callback nằm
trong phần bù, `clock.calibrated = false`: giờ đó chỉ là tương đối, công cụ không chấm theo nó.
Sửa theo Codex Sol vòng 1/2 (Liam duyệt 30/10 "Duyệt, sửa hết"): S7 neo epoch theo lượt + kiểm reset
±1 s quanh #run_start; S8 mở vòng report có luật k ≥ 1 và cắt ở 128 chu kỳ; S9/S34 callback lấy danh
tính của report cổng SPI cùng seq gần nhất theo thời gian; S10 như trên; S11 mở vòng CYCCNT theo giờ Pi;
S32 chỉ chấm ghép report trong khung chung bớt 0,5 s mỗi đầu [ĐỀ XUẤT CODE]; S36 `dropped` lấy hiệu TICK.
Chưa có (ghi rõ): đồng hồ ESP32 → RPi5 (cần chân đánh dấu). Chưa thử với frame thật của H7 (chỉ dữ liệu tổng hợp, tools/tests/test_s06_frames_decode.py).

Usage:
  python tools/s06_frames_decode.py <frames.npz> [--esp32-events esp32_events.txt] --out <thư mục>
Mã thoát: 0 không có lỗi; 1 có CRC sai, hở log, report không ghép được hoặc CRC IMU sai; 2 lỗi đầu vào.
"""
from __future__ import annotations

import argparse
import binascii
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

FRAME = 40
MOSI_SEQ, MOSI_IMU_SEQ = 21, 39
MOSI_IMU_BYTES = list(range(22, 24)) + list(range(31, 39))      # gz, ax, ay, az, yaw (HIL_Imu_t.raw)
MISO_CRC, MISO_MOSI_CRC, MISO_RX_CYC, MISO_STATUS, MISO_SEQ, MISO_RX_NF, MISO_AGE, MISO_LOG = \
    8, 10, 12, 16, 17, 18, 20, 22
REC_REPORT, REC_PKT_READ, REC_RESET_DONE, REC_HOLD_ON, REC_HOLD_OFF = 0x10, 0x20, 0x30, 0x31, 0x32
REC_RST_LOW, REC_RST_HIGH, REC_TICK, REC_REARM = 0x33, 0x34, 0x40, 0x50
REC_ENC_START, REC_ENC_STOP = 0x35, 0x36            # feat/s06-h7 598a451, with EncoderGen b005aa7
REPORT_KIND = {0: 'accel', 1: 'gyro', 2: 'game_rv'}
ESP32_KIND = {1: 'accel', 2: 'gyro', 8: 'game_rv'}               # SH-2 report ids
SEGMENT_GAP_US = 500_000.0
F_PRED_HZ = 483.1e6           # CYCCNT rate used only to predict the count across gaps (spec value, Phần D)
SEQ_CUT_PERIODS = 128         # spec dòng 96: unwrap only below 128 report periods, cut the chain from 128
EDGE_US = 500_000.0           # S32 [ĐỀ XUẤT CODE]: reports this close to an end of the common window
                              # (Pi frames ∩ ESP32 #run_start..#run_end) are reported, not scored
RESET_NEAR_US = 1_000_000.0   # V4-1: no reset-complete within ±1 s of #run_start
CARRY_MARGIN_CYC = 1_000_000  # a record's CYCCNT may be stamped up to ~2 ms after its carrier's callback


def crc16(b: bytes) -> int:
    return binascii.crc_hqx(bytes(b), 0xFFFF)


def u16(b, o):
    return int(b[o]) | int(b[o + 1]) << 8


def u32(b, o):
    return u16(b, o) | u16(b, o + 2) << 16


def unwrap(vals, bits):
    """Unwrap a counter sampled in order (steps forward < half the range)."""
    m, half = 1 << bits, 1 << (bits - 1)
    out, base, prev = [], 0, None
    for v in vals:
        if prev is not None:
            d = (v - prev) % m
            base += d if d < half else d - m
        else:
            base = v
        out.append(base)
        prev = v
    return out


def unwrap_timed(vals, t_us, f_hz=F_PRED_HZ, bits=32):
    """Unwrap a cycle counter sampled in time order, each step predicted from the Pi time elapsed
    (S11: a link gap longer than half the 32-bit range, ~4.4 s at 483 MHz, is no longer read as a
    step back). f_hz only has to be within ~1 % over gaps up to minutes."""
    out, base, t_prev = [], None, None
    for v, t in zip(vals, t_us):
        base = v if base is None else near(v, base + (t - t_prev) * 1e-6 * f_hz, bits)
        out.append(base)
        t_prev = t
    return out


def near(v, ref, bits=32):
    """Full value of the low `bits` of v closest to the full value ref."""
    m, half = 1 << bits, 1 << (bits - 1)
    return ref + ((v - ref + half) % m) - half


def behind(v, ref, margin=CARRY_MARGIN_CYC, bits=32):
    """Full value of the low `bits` of v that is not later than ref + margin: a log record is made
    before the frame that carries it, possibly long before when the ring backed up during a link
    gap (`near` would put a record made > 4.4 s before its carrier one wrap too late). Holds for a
    backlog up to one wrap (~8.9 s at 483 MHz); longer shows as a TICK mismatch."""
    m = 1 << bits
    top = ref + margin
    return top - ((top - v) % m)


# ── 1. frames, CRC ──

def decode_frames(t_send_ns, mosi, miso, max_lag=4):
    n = len(t_send_ns)
    t = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    rows, lag_hist = [], Counter()
    st = dict(frames=n, empty=0, miso_crc_bad=0, mosi_crc_bad=0, unanswered=0)
    for j in range(n):
        b = bytes(miso[j])
        r = dict(j=j, t_us=t[j])
        if not any(b):
            st['empty'] += 1
            r['empty'] = True
            rows.append(r)
            continue
        r['empty'] = False
        r['crc_ok'] = u16(b, MISO_CRC) == crc16(b[:MISO_CRC] + b[MISO_CRC + 2:])
        if not r['crc_ok']:
            st['miso_crc_bad'] += 1
            rows.append(r)
            continue
        r.update(seq=b[MISO_SEQ], rx_cyc=u32(b, MISO_RX_CYC), rx_nf16=u16(b, MISO_RX_NF),
                 age_us=u16(b, MISO_AGE), status=b[MISO_STATUS], log_type=b[MISO_LOG],
                 log_idx16=u16(b, MISO_LOG + 1), payload=b[MISO_LOG + 3:FRAME])
        ans = None
        for lag in range(1, max_lag + 1):
            i = j - lag
            if i >= 0 and mosi[i][MOSI_SEQ] == r['seq']:
                ans = i
                break
        r['answered'] = ans
        if ans is None:
            st['unanswered'] += 1
        else:
            lag_hist[j - ans] += 1
            r['mosi_crc_ok'] = u16(b, MISO_MOSI_CRC) == crc16(bytes(mosi[ans]))
            st['mosi_crc_bad'] += 0 if r['mosi_crc_ok'] else 1
        rows.append(r)
    st['answer_lag'] = dict(sorted(lag_hist.items()))
    return rows, st


# ── 3. clock (before 2: records need the frame map) ──

def frame_map(rows, t_ref_us):
    """Distinct H7 frames (rx_nf unwrapped) -> (Pi frame answered, CYCCNT unwrapped). CYCCNT is
    unwrapped in Pi send order with the step predicted from the Pi time (S11), not by the 32-bit
    half-range rule, so a long link gap keeps the right count of wraps."""
    good = [r for r in rows if not r['empty'] and r['crc_ok'] and r['answered'] is not None]
    nf_full = unwrap([r['rx_nf16'] for r in good], 16)
    first = {}
    for r, nf in zip(good, nf_full):
        r['rx_nf'] = nf
        first.setdefault(nf, r)
    order = sorted(first, key=lambda k: first[k]['answered'])
    cyc_full = unwrap_timed([first[k]['rx_cyc'] for k in order], [t_ref_us[first[k]['answered']] for k in order])
    return {k: dict(i=first[k]['answered'], cyc=c, carrier=first[k]['j']) for k, c in zip(order, cyc_full)}


def fit_clock(fmap, t_send_us, cut_cycs=()):
    pts = sorted(((v['cyc'], t_send_us[v['i']]) for v in fmap.values()), key=lambda p: p[1])
    segs, cur = [], []
    cuts = sorted(cut_cycs)
    for c, t in pts:
        if cur and (t - cur[-1][1] > SEGMENT_GAP_US or any(cur[-1][0] < x <= c for x in cuts)):
            segs.append(cur)
            cur = []
        cur.append((c, t))
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        c = np.array([p[0] for p in s], dtype=np.float64)
        t = np.array([p[1] for p in s])
        if len(s) >= 2:
            c0 = c[0]
            b, a = np.polyfit(c - c0, t, 1)
            res = t - (a + b * (c - c0))
        else:
            c0, a, b, res = c[0], t[0], 0.0, np.zeros(1)
        out.append(dict(cyc_lo=float(c[0]), cyc_hi=float(c[-1]), c0=float(c0), a=float(a), b=float(b), n=len(s),
                        f_h7_mhz=(1.0 / b if b else float('nan')), res_max_us=float(np.max(np.abs(res))),
                        res_std_us=float(np.std(res))))
    return out


def cyc_to_pi(segs, cyc):
    best = min(segs, key=lambda s: 0 if s['cyc_lo'] <= cyc <= s['cyc_hi'] else min(abs(cyc - s['cyc_lo']), abs(cyc - s['cyc_hi'])))
    return best['a'] + best['b'] * (cyc - best['c0'])


# ── 2. log stream ──

def log_records(rows):
    recs, seen = [], set()
    carried = [r for r in rows if not r['empty'] and r['crc_ok'] and r['log_type'] != 0]
    full = unwrap([r['log_idx16'] for r in carried], 16)
    dups = 0
    for r, idx in zip(carried, full):
        if idx in seen:
            dups += 1
            continue
        seen.add(idx)
        recs.append(dict(idx=idx, type=r['log_type'], payload=r['payload'], carrier_j=r['j']))
    gaps = (max(seen) - min(seen) + 1 - len(seen)) if seen else 0
    return recs, dict(records=len(recs), repeated_frames=dups, gaps=gaps)


def parse_record(rec):
    p, t = rec['payload'], rec['type']
    if REC_REPORT <= t <= REC_REPORT + 2:
        return dict(kind=REPORT_KIND[t - REC_REPORT], rpt_seq=p[0], queued=bool(p[1] & 1), qfull=bool(p[1] & 2),
                    epoch=p[2], imu_seq=p[3], src_nf16=u16(p, 4), nf16=u16(p, 6), imu_crc=u16(p, 8), cyc=u32(p, 10))
    if t == REC_PKT_READ:
        return dict(mask=p[0], seqs=tuple(p[1:4]), epoch=p[4], cyc=u32(p, 5))
    if t in (REC_RESET_DONE, REC_HOLD_ON, REC_HOLD_OFF):
        return dict(epoch=p[0], cyc=u32(p, 1), nf16=u16(p, 5))
    if t in (REC_RST_LOW, REC_RST_HIGH):
        return dict(cyc=u32(p, 0), nf16=u16(p, 4))
    if t == REC_TICK:
        return dict(cyc=u32(p, 0), cyc_total=u32(p, 4) | u32(p, 8) << 32, dropped=u16(p, 12))
    if t == REC_REARM:
        return dict(cyc=u32(p, 0), n_rearms=u16(p, 4))
    if t in (REC_ENC_START, REC_ENC_STOP):
        return dict(wheel=p[0], cnt=u16(p, 1), n=u32(p, 3), cyc=u32(p, 7), nf16=u16(p, 11))
    return dict(unknown=True)


def nf_full(nf16, ref_nf):
    return near(nf16, ref_nf, 16)


# ── 4. reports ──

def seq_chain(items, seq_of, t_of, mod=256, cut=SEQ_CUT_PERIODS):
    """S8: unwrap an 8-bit report sequence of ONE (epoch, kind) in arrival order, spec dòng 96:
    a step k = (seq − seq_prev) mod 256 must be ≥ 1 (0 = the same report number twice: `dup`);
    unwrap only when the two reports are < 128 report periods apart, and then k < 128 (else
    `seq_time_bad`); from 128 periods the chain is cut (`cut`) and the next report starts a new
    chain. The period is the median spacing of the steps with k = 1. Returns per item
    (n, chain, flag) with flag in ('', 'dup', 'seq_time_bad', 'cut'). After a cut n goes on (+k, or
    +256 when k = 0) only to stay unique: the match across the cut is not trusted (flag counted)."""
    seqs, ts = [seq_of(x) for x in items], [t_of(x) for x in items]
    steps = [ts[i] - ts[i - 1] for i in range(1, len(items))
             if ts[i] is not None and ts[i - 1] is not None and (seqs[i] - seqs[i - 1]) % mod == 1]
    period = float(np.median(steps)) if steps else None
    out, n, chain = [], None, 0
    for i in range(len(items)):
        flag = ''
        if i == 0:
            n = seqs[0]
        else:
            k = (seqs[i] - seqs[i - 1]) % mod
            gap = None if (period is None or ts[i] is None or ts[i - 1] is None) else ts[i] - ts[i - 1]
            if gap is not None and gap >= cut * period:
                chain += 1                    # number of steps unknown: keep n unique, flag the run
                n, flag = n + (k or mod), 'cut'
            elif k == 0:
                flag = 'dup'
            elif k >= cut:
                n, flag = n + k, 'seq_time_bad'
            else:
                n += k
        out.append((n, chain, flag))
    return out, period


def h7_reports(recs, fmap, mosi, carrier_nf):
    """REPORT records with source frame checks; seq unwrapped per (epoch, kind) in log order with
    the rules of `seq_chain` on the H7 build time (CYCCNT unwrapped)."""
    out = []
    by_key = defaultdict(list)
    for rec in recs:
        if 'rpt_seq' in rec['f']:
            by_key[(rec['f']['epoch'], rec['f']['kind'])].append(rec)
    nchain = {}
    for key, lst in by_key.items():
        res, _ = seq_chain(lst, lambda r: r['f']['rpt_seq'], lambda r: r.get('cyc_full'))
        for rec, v in zip(lst, res):
            nchain[id(rec)] = v
    for rec in recs:
        f = rec['f']
        if 'rpt_seq' not in f:
            continue
        ref = carrier_nf[rec['carrier_j']]
        src = nf_full(f['src_nf16'], ref)
        fr = fmap.get(src)
        n, chain, flag = nchain[id(rec)]
        latest = nf_full(f['nf16'], ref)
        row = dict(log_idx=rec['idx'], kind=f['kind'], epoch=f['epoch'], rpt_seq=f['rpt_seq'], n=n,
                   chain=chain, seq_flag=flag,
                   queued=f['queued'], qfull=f['qfull'], imu_seq=f['imu_seq'], src_nf=src, nf_latest=latest,
                   imu_age_frames=latest - src, imu_crc_h7=f['imu_crc'], cyccnt=f['cyc'],
                   t_pi_us=rec.get('t_pi_us'), src_frame=fr['i'] if fr else None)
        if fr is not None:
            m = mosi[fr['i']]
            row['imu_crc_pi'] = crc16(bytes(m[k] for k in MOSI_IMU_BYTES))
            row['imu_crc_ok'] = f['imu_crc'] == row['imu_crc_pi']
            row['imu_seq_ok'] = f['imu_seq'] == m[MOSI_IMU_SEQ]
        out.append(row)
    return out


def enc_calls(recs):
    """EncoderGen _start_wheel/_stop records -> {wheel: [('S'|'P', t_pi_us, CNT, n)]} for
    tools/t3b_score.py, and the number of calls missing from the per-wheel running count n."""
    calls, last, gaps = {w: [] for w in range(4)}, {}, 0
    for rec in recs:
        if rec['type'] not in (REC_ENC_START, REC_ENC_STOP):
            continue
        f, k = rec['f'], 'S' if rec['type'] == REC_ENC_START else 'P'
        calls.setdefault(f['wheel'], []).append((k, rec.get('t_pi_us'), f['cnt'], f['n']))
        prev = last.get((f['wheel'], k))
        gaps += (f['n'] - prev - 1) if prev is not None else f['n'] - 1
        last[(f['wheel'], k)] = f['n']
    return calls, gaps


def esp32_lines(path):
    """Dòng J (cổng SPI), I (callback), R (reset hoàn tất) và #run_start/#run_end của esp32_events.txt;
    `t_rx` = giờ Pi lúc logger nhận (monotonic, cùng đồng hồ với t_send_ns), `t_us` = giờ ESP32."""
    J, I, R, text = [], [], [], {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep:
                continue
            p = line.split(',')
            try:
                t_rx = int(t_rx)
                if p[0] == 'J' and len(p) == 7:
                    J.append(dict(t_rx=t_rx, t_us=int(p[2]), kind=ESP32_KIND.get(int(p[3]), p[3]), queued=int(p[4]),
                                  seq=int(p[5]), epoch=int(p[6])))
                elif p[0] == 'I' and len(p) == 12:
                    I.append(dict(t_rx=t_rx, t_us=int(p[2]), kind=ESP32_KIND.get(int(p[3]), p[3]), applied=int(p[4]),
                                  seq=int(p[5]), epoch=int(p[6]),
                                  v=(float(p[9]), float(p[10]), float(p[11]))))
                elif p[0] == 'R' and len(p) == 4:
                    R.append(dict(t_rx=t_rx, t_us=int(p[2]), reset_count=int(p[3])))
                elif p[0] in ('#run_start', '#run_end') and p[0][1:] not in text:
                    kv = dict(x.partition('=')[::2] for x in p[1:] if '=' in x)
                    text[p[0][1:]] = dict(t_rx=t_rx, kv=kv)
            except ValueError:
                continue
    return J, I, R, text


def seq_keys(rows):
    """(epoch, kind, n): n = seq unwrapped in ESP32 time order within (epoch, kind) by `seq_chain`.
    Flags (dup, seq_time_bad, cut) go into r['seq_flag']."""
    by = defaultdict(list)
    for r in sorted(rows, key=lambda x: x['t_us']):
        by[(r['epoch'], r['kind'])].append(r)
    for lst in by.values():
        res, _ = seq_chain(lst, lambda r: r['seq'], lambda r: r['t_us'])
        for r, (n, chain, flag) in zip(lst, res):
            r['n'], r['chain'], r['seq_flag'] = n, chain, flag
    return {(r['epoch'], r['kind'], r['n']): r for r in rows}


def map_callbacks(I, J):
    """S9, S34: each callback (I) takes the identity n of the report at the SPI port (J) with the same
    (epoch, kind, seq) NEAREST in ESP32 time (no `J before I` rule: the producer does not guarantee
    that order). The same seq comes back only every 256 reports, so the nearest one is unique while
    |t_I − t_J| < 128 report periods; farther = not matched. Sets I['n'] (None if not matched) and
    I['j'] (the J row)."""
    by = defaultdict(list)
    for j in J:
        by[(j['epoch'], j['kind'], j['seq'])].append(j)
    periods = {}
    for (ep, kd), lst in _group(J, lambda j: (j['epoch'], j['kind'])).items():
        _, periods[(ep, kd)] = seq_chain(sorted(lst, key=lambda x: x['t_us']), lambda r: r['seq'], lambda r: r['t_us'])
    for i in I:
        cands = by.get((i['epoch'], i['kind'], i['seq']), [])
        per = periods.get((i['epoch'], i['kind']))
        best = min(cands, key=lambda j: abs(j['t_us'] - i['t_us']), default=None)
        if best is not None and per is not None and abs(best['t_us'] - i['t_us']) >= SEQ_CUT_PERIODS * per:
            best = None
        i['j'] = best
        i['n'] = best['n'] if best is not None else None
    return I


def _group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def epoch_offset(h7, J):
    """S7: anchor the two epoch counters on the run instead of maximising matches. The distinct epochs
    seen in the run, in order, must be the same number on both sides and step together; the offset
    is ESP32 − H7 of the first one. Returns (offset or None, note)."""
    eh = sorted({r['epoch'] for r in h7})
    ej = []
    for j in sorted(J, key=lambda x: x['t_us']):
        if not ej or ej[-1] != j['epoch']:
            ej.append(j['epoch'])
    if not eh or not ej:
        return None, 'không có report ở một phía'
    if len(eh) != len(set(ej)) or len(set(ej)) != len(ej):
        return None, f'epoch H7 {eh} và ESP32 {ej} không cùng số/không liên tục: không neo được'
    off = ej[0] - eh[0]
    if any(b - a != off for a, b in zip(eh, ej)):
        return None, f'epoch H7 {eh} và ESP32 {ej} không cùng bước: không neo được'
    return off, ''


def common_window(t_send_us, text):
    """Window scored for report matching on the Pi clock: Pi frames ∩ ESP32 #run_start..#run_end
    (t_rx), shrunk by EDGE_US at both ends (S32). None if the ESP32 run marks are missing."""
    if 'run_start' not in text or 'run_end' not in text:
        return None
    lo = max(float(t_send_us[0]), float(text['run_start']['t_rx'])) + EDGE_US
    hi = min(float(t_send_us[-1]), float(text['run_end']['t_rx'])) - EDGE_US
    return (lo, hi) if hi > lo else None


def match_esp32(h7, J, I, window=None, span=None):
    """Match H7 reports with the ESP32 SPI port (J) and callbacks (I) on (epoch, kind, n). Only reports
    inside `window` (Pi clock: H7 build time t_pi_us; J receive time t_rx) are scored (S32); the rest
    is counted apart. Without a window every report is scored (as before) and that is reported.
    Epochs are anchored on the J received during the Pi frames (`span`, Pi clock)."""
    jk = seq_keys(J)
    map_callbacks(I, J)
    off, note = epoch_offset(h7, [j for j in J if span is None or span[0] <= j['t_rx'] <= span[1]])
    cb_by_j = defaultdict(list)
    for i in I:
        if i['j'] is not None:
            cb_by_j[id(i['j'])].append(i)

    def inside_h7(r):
        return window is None or (r.get('t_pi_us') is not None and window[0] <= r['t_pi_us'] <= window[1])

    def inside_j(j):
        return window is None or window[0] <= j['t_rx'] <= window[1]

    h7k = set()
    for r in h7:
        k = None if off is None else (r['epoch'] + off, r['kind'], r['n'])
        h7k.add(k)
        j = jk.get(k) if k is not None else None
        r['esp32_spi'] = j is not None
        cbs = cb_by_j.get(id(j), []) if j is not None else []
        r['esp32_cb'] = any(i['applied'] == 1 for i in cbs)
        r['esp32_t_us'] = j['t_us'] if j is not None else None
        r['scored'] = inside_h7(r)
    sc = [r for r in h7 if r['scored']]
    matched = sum(1 for r in sc if r['esp32_spi'])
    extra = [j for k, j in jk.items() if k not in h7k and inside_j(j)]
    return dict(epoch_offset_esp32_minus_h7=off, epoch_note=note, window_pi_us=window,
                h7_reports=len(h7), h7_scored=len(sc), matched_spi=matched,
                not_at_esp32_spi=len(sc) - matched if off is not None else len(sc),
                at_spi_not_cb=sum(1 for r in sc if r['esp32_spi'] and not r['esp32_cb']),
                esp32_without_h7=len(extra) if off is not None else sum(1 for j in J if inside_j(j)),
                esp32_seq_flags=dict(Counter(j['seq_flag'] for j in J if j['seq_flag'] and inside_j(j))),
                callbacks_unmatched=sum(1 for i in I if i['n'] is None))


def resets_near_start(recs, R, text):
    """V4-1: reset-complete within ±1 s of #run_start (H7: RESET_DONE build time on the Pi clock;
    ESP32: R lines, receive time). Returns dict or None if #run_start is missing."""
    if 'run_start' not in text:
        return None
    t0 = float(text['run_start']['t_rx'])
    h7 = [rec['t_pi_us'] for rec in recs if rec['type'] == REC_RESET_DONE and rec.get('t_pi_us') is not None
          and abs(rec['t_pi_us'] - t0) <= RESET_NEAR_US]
    esp = [r['t_rx'] for r in R if abs(r['t_rx'] - t0) <= RESET_NEAR_US]
    return dict(h7=len(h7), esp32=len(esp), ok=not h7 and not esp)


# ── all ──

def clock_reference(t_send_ns, cal):
    """Pi reference time of each frame (µs) and the calibration status (S10, spec dòng 111).
    Spec: P_gửi := t_pulse_rise (the sync pulse that also marks d_1w); RPi5 logs the time just before
    and just after raising it (`t_pulse0_ns`, `t_pulse1_ns` in frames.npz); the H7 event time is
    P = fit(H) + d_1w,median; uncertainty = max residual + (d_1w max − min) + δ_H + δ_P max + 2·t_s.
    Without the pulse times or without d_1w/δ_H/t_s the times are only RELATIVE (send time, the
    constant send → callback delay inside the fit) and `calibrated` is False: tools must not score
    on them as if synchronised."""
    cal = dict(cal or {})
    t_send_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    p0, p1 = cal.get('t_pulse0_ns'), cal.get('t_pulse1_ns')
    if p0 is not None and p1 is not None and len(p0) == len(t_send_us):
        p0 = np.asarray(p0, dtype=np.float64) / 1000.0
        p1 = np.asarray(p1, dtype=np.float64) / 1000.0
        t_ref, dp = 0.5 * (p0 + p1), float(np.max(p1 - p0))
        ref = 'pulse'
    else:
        t_ref, dp, ref = t_send_us, None, 'send'
    need = ('d1w_us', 'd1w_span_us', 'delta_h_us', 'ts_us')
    ok = ref == 'pulse' and all(cal.get(k) is not None for k in need)
    return t_ref, dict(reference=ref, calibrated=ok, delta_p_max_us=dp,
                       **{k: cal.get(k) for k in need})


def decode(t_send_ns, mosi, miso, esp32_events: Optional[str] = None, cal: Optional[dict] = None) -> dict:
    mosi = np.asarray(mosi, dtype=np.uint8)
    miso = np.asarray(miso, dtype=np.uint8)
    rows, st = decode_frames(t_send_ns, mosi, miso)
    t_ref_us, calib = clock_reference(t_send_ns, cal)
    fmap = frame_map(rows, t_ref_us)
    recs, lst = log_records(rows)
    # nf of the frame that carried each record (reference for 16-bit nf and 32-bit cyc fields)
    # (CYCCNT of a frame not in fmap, e.g. right after a gap, is predicted from the last matched frame
    # and the Pi time elapsed, not copied from it)
    carrier_nf, carrier_cyc = {}, {}
    by_carrier = {v['carrier']: (k, v['cyc'], v['i']) for k, v in fmap.items()}
    last = None
    for r in rows:
        if r['j'] in by_carrier:
            last = by_carrier[r['j']]
        if last is not None:
            nf, cyc, i = last
            carrier_nf[r['j']] = nf
            carrier_cyc[r['j']] = cyc + int(round((t_ref_us[r['j']] - t_ref_us[i]) * 1e-6 * F_PRED_HZ))
    for rec in recs:
        rec['f'] = parse_record(rec)
    t_send_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    rearm = [behind(rec['f']['cyc'], carrier_cyc[rec['carrier_j']]) for rec in recs
             if rec['type'] == REC_REARM and rec['carrier_j'] in carrier_cyc]
    segs = fit_clock(fmap, t_ref_us, rearm) if fmap else []
    d1w = calib['d1w_us'] if calib['calibrated'] else 0.0
    if calib['calibrated'] and segs:
        calib['uncertainty_us'] = (max(s['res_max_us'] for s in segs) + calib['d1w_span_us'] + calib['delta_h_us']
                                   + calib['delta_p_max_us'] + 2.0 * calib['ts_us'])
    else:
        calib['uncertainty_us'] = None
    for rec in recs:
        f = rec['f']
        if 'cyc' not in f or rec['carrier_j'] not in carrier_cyc:
            continue
        ref = carrier_cyc[rec['carrier_j']]
        nf = f.get('nf16')
        k = nf_full(nf, carrier_nf[rec['carrier_j']]) if nf is not None else None
        if k is not None and k in fmap:
            rec['cyc_full'] = near(f['cyc'], fmap[k]['cyc'])
        else:
            rec['cyc_full'] = behind(f['cyc'], ref)
        rec['t_pi_us'] = cyc_to_pi(segs, rec['cyc_full']) + d1w if segs else None
    ticks = [rec for rec in recs if rec['type'] == REC_TICK and 'cyc_full' in rec]
    tick_bad = sum(1 for a, b in zip(ticks, ticks[1:])
                   if (b['f']['cyc_total'] - a['f']['cyc_total']) != (b['cyc_full'] - a['cyc_full']))
    # S36: TICK carries the H7 drop counter since S06Log_Init, cut to 16 bits: the run figure is the
    # difference between the first and the last TICK of the frames (mod 65 536; drops before the first
    # TICK still show as gaps of the record index). Fewer than 2 TICK: not known.
    if len(ticks) >= 2:
        lst['h7_dropped'] = (ticks[-1]['f']['dropped'] - ticks[0]['f']['dropped']) & 0xFFFF
    else:
        lst['h7_dropped'] = None
    lst['h7_dropped_counter_last'] = ticks[-1]['f']['dropped'] if ticks else None
    lst['ticks'] = len(ticks)
    reps = h7_reports(recs, fmap, mosi, carrier_nf)
    calls, call_gaps = enc_calls(recs)
    res = dict(frames=st, log=lst, clock=dict(segments=segs, tick_total_mismatch=tick_bad, **calib),
               reports=dict(n=len(reps), imu_crc_bad=sum(1 for r in reps if r.get('imu_crc_ok') is False),
                            imu_seq_bad=sum(1 for r in reps if r.get('imu_seq_ok') is False),
                            no_source_frame=sum(1 for r in reps if r['src_frame'] is None),
                            seq_flags=dict(Counter(r['seq_flag'] for r in reps if r['seq_flag'])),
                            per_kind=dict(Counter(r['kind'] for r in reps))),
               enc_calls=dict(per_wheel={w: dict(Counter(c[0] for c in v)) for w, v in calls.items()},
                              count_gaps=call_gaps))
    if esp32_events:
        J, I, R, text = esp32_lines(esp32_events)
        span = (float(t_send_us[0]), float(t_send_us[-1]))
        res['esp32_match'] = match_esp32(reps, J, I, common_window(t_send_us, text), span)
        res['esp32_run_marks'] = sorted(text)
        res['resets_near_run_start'] = resets_near_start(recs, R, text)
        res['_esp32_I'] = I
    res['_records'] = recs
    res['_frames_rows'] = rows
    res['_calls'] = calls
    res['_reports'] = reps
    return res


def problems(res) -> List[str]:
    p = []
    f, lg, rp = res['frames'], res['log'], res['reports']
    if f['miso_crc_bad']:
        p.append(f"{f['miso_crc_bad']} frame MISO sai CRC")
    if f['mosi_crc_bad']:
        p.append(f"{f['mosi_crc_bad']} frame MOSI sai CRC ở H7")
    if f['unanswered']:
        p.append(f"{f['unanswered']} frame MISO không tìm được frame MOSI được trả lời")
    if lg['gaps']:
        p.append(f"luồng log H7 hở {lg['gaps']} bản ghi")
    if lg.get('h7_dropped'):
        p.append(f"H7 bỏ {lg['h7_dropped']} bản ghi vì đầy (hiệu TICK đầu–cuối)")
    if lg.get('h7_dropped') is None:
        p.append(f"chỉ có {lg.get('ticks', 0)} bản ghi TICK: không biết H7 có bỏ bản ghi trong lượt không")
    if rp.get('seq_flags'):
        p.append(f"số thứ tự report H7 bất thường {rp['seq_flags']} (dup = trùng, cut = cách ≥ 128 chu kỳ, "
                 f"seq_time_bad = bước ≥ 128)")
    if res['clock']['tick_total_mismatch']:
        p.append(f"{res['clock']['tick_total_mismatch']} cặp TICK lệch _cyc_total")
    if rp['imu_crc_bad'] or rp['imu_seq_bad'] or rp['no_source_frame']:
        p.append(f"report: CRC IMU sai {rp['imu_crc_bad']}, imu_seq sai {rp['imu_seq_bad']}, "
                 f"không có frame nguồn {rp['no_source_frame']}")
    if res['enc_calls']['count_gaps']:
        p.append(f"số lần gọi _start_wheel/_stop hở {res['enc_calls']['count_gaps']} (thiếu bản ghi)")
    m = res.get('esp32_match')
    if m and m['epoch_offset_esp32_minus_h7'] is None:
        p.append(f"ghép ESP32: {m['epoch_note']}")
    if m and (m['not_at_esp32_spi'] or m['esp32_without_h7']):
        p.append(f"ghép ESP32: {m['not_at_esp32_spi']} report H7 không tới cổng SPI ESP32, "
                 f"{m['esp32_without_h7']} report ESP32 không có ở H7")
    if m and m.get('esp32_seq_flags'):
        p.append(f"số thứ tự report ở cổng SPI ESP32 bất thường {m['esp32_seq_flags']}")
    if m and m['window_pi_us'] is None:
        p.append('esp32_events thiếu #run_start/#run_end: chấm ghép report trên cả luồng, không theo khung lượt')
    rz = res.get('resets_near_run_start')
    if 'esp32_match' in res and rz is None:
        p.append('thiếu #run_start: không kiểm được điều kiện reset ±1 s (V4-1)')
    elif rz is not None and not rz['ok']:
        p.append(f"reset hoàn tất trong ±1 s quanh #run_start (H7 {rz['h7']}, ESP32 {rz['esp32']}): lượt không hợp lệ (V4-1)")
    return p


def add_cal_args(ap):
    """CLI of the H7 → RPi5 time calibration (spec dòng 111), shared by the S06 tools."""
    g = ap.add_argument_group('quy giờ H7 → RPi5 (spec dòng 111; thiếu thì giờ H7 chỉ là tương đối)')
    g.add_argument('--d1w-us', type=float, default=None, help='d_1w trung vị (cạnh xung đồng bộ → chân debug H7), µs')
    g.add_argument('--d1w-span-us', type=float, default=None, help='d_1w lớn nhất − nhỏ nhất, µs')
    g.add_argument('--delta-h-us', type=float, default=None, help='δ_H, µs')
    g.add_argument('--ts-us', type=float, default=None, help='chu kỳ lấy mẫu logic analyzer t_s, µs')


def load_cal(z, a) -> dict:
    """Calibration dict for `decode` from frames.npz (pulse times if logged) and the CLI of add_cal_args."""
    files = getattr(z, 'files', z)
    cal = {k: getattr(a, k) for k in ('d1w_us', 'd1w_span_us', 'delta_h_us', 'ts_us')}
    for k in ('t_pulse0_ns', 't_pulse1_ns'):
        cal[k] = z[k] if k in files else None
    return cal


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Giải mã frames.npz (S06, MISO bản 4)')
    ap.add_argument('frames')
    ap.add_argument('--esp32-events', default=None)
    ap.add_argument('--out', required=True)
    add_cal_args(ap)
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        res = decode(z['t_send_ns'], z['mosi'], z['miso'], a.esp32_events, load_cal(z, a))
    except (OSError, KeyError, ValueError) as ex:
        print(f'lỗi đầu vào: {ex}', file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'h7_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['idx', 'type', 'carrier_frame', 'cyc_full', 't_pi_us', 'fields'])
        for r in res['_records']:
            w.writerow([r['idx'], f"0x{r['type']:02x}", r['carrier_j'], r.get('cyc_full', ''),
                        f"{r['t_pi_us']:.1f}" if r.get('t_pi_us') is not None else '',
                        json.dumps({k: v for k, v in r['f'].items()}, default=str)])
    with open(os.path.join(a.out, 'h7_reports.csv'), 'w', newline='') as fh:
        cols = ['log_idx', 'kind', 'epoch', 'rpt_seq', 'n', 'chain', 'seq_flag', 'queued', 'imu_seq', 'src_nf',
                'src_frame', 't_pi_us', 'imu_crc_ok', 'imu_seq_ok', 'esp32_spi', 'esp32_cb', 'esp32_t_us', 'scored']
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(res['_reports'])
    summ = {k: v for k, v in res.items() if not k.startswith('_')}
    summ['problems'] = problems(res)
    with open(os.path.join(a.out, 'frames_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summ, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    return 1 if summ['problems'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
