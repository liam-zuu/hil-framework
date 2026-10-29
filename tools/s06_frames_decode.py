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

Chưa có (ghi rõ): đồng hồ ESP32 → RPi5 (cần chân đánh dấu); số lần gọi `_start_wheel`/`_stop`
(việc (5)). Chưa thử với frame thật của H7 (chỉ dữ liệu tổng hợp, tools/tests/test_s06_frames_decode.py).

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
REPORT_KIND = {0: 'accel', 1: 'gyro', 2: 'game_rv'}
ESP32_KIND = {1: 'accel', 2: 'gyro', 8: 'game_rv'}               # SH-2 report ids
SEGMENT_GAP_US = 500_000.0


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


def near(v, ref, bits=32):
    """Full value of the low `bits` of v closest to the full value ref."""
    m, half = 1 << bits, 1 << (bits - 1)
    return ref + ((v - ref + half) % m) - half


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

def frame_map(rows):
    """Distinct H7 frames (rx_nf unwrapped) -> (Pi frame answered, CYCCNT unwrapped)."""
    good = [r for r in rows if not r['empty'] and r['crc_ok'] and r['answered'] is not None]
    nf_full = unwrap([r['rx_nf16'] for r in good], 16)
    first = {}
    for r, nf in zip(good, nf_full):
        r['rx_nf'] = nf
        first.setdefault(nf, r)
    nfs = sorted(first)
    cyc_full = unwrap([first[k]['rx_cyc'] for k in nfs], 32)
    return {k: dict(i=first[k]['answered'], cyc=c, carrier=first[k]['j']) for k, c in zip(nfs, cyc_full)}


def fit_clock(fmap, t_send_us, cut_cycs=()):
    pts = sorted((v['cyc'], t_send_us[v['i']]) for v in fmap.values())
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
    return dict(unknown=True)


def nf_full(nf16, ref_nf):
    return near(nf16, ref_nf, 16)


# ── 4. reports ──

def h7_reports(recs, fmap, mosi, carrier_nf):
    """REPORT records with source frame checks; seq unwrapped per (epoch, kind) in log order."""
    out, nseq = [], defaultdict(lambda: None)
    for rec in recs:
        f = rec['f']
        if 'rpt_seq' not in f:
            continue
        ref = carrier_nf[rec['carrier_j']]
        src = nf_full(f['src_nf16'], ref)
        fr = fmap.get(src)
        key = (f['epoch'], f['kind'])
        prev = nseq[key]
        n = f['rpt_seq'] if prev is None else prev + ((f['rpt_seq'] - prev) % 256)
        nseq[key] = n
        row = dict(log_idx=rec['idx'], kind=f['kind'], epoch=f['epoch'], rpt_seq=f['rpt_seq'], n=n,
                   queued=f['queued'], imu_seq=f['imu_seq'], src_nf=src, t_pi_us=rec.get('t_pi_us'),
                   src_frame=fr['i'] if fr else None)
        if fr is not None:
            m = mosi[fr['i']]
            row['imu_crc_ok'] = f['imu_crc'] == crc16(bytes(m[k] for k in MOSI_IMU_BYTES))
            row['imu_seq_ok'] = f['imu_seq'] == m[MOSI_IMU_SEQ]
        out.append(row)
    return out


def esp32_lines(path):
    J, I = [], []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            _, sep, line = raw.rstrip('\n').partition('\t')
            if not sep:
                continue
            p = line.split(',')
            try:
                if p[0] == 'J' and len(p) == 7:
                    J.append(dict(t_us=int(p[2]), kind=ESP32_KIND.get(int(p[3]), p[3]), queued=int(p[4]),
                                  seq=int(p[5]), epoch=int(p[6])))
                elif p[0] == 'I' and len(p) == 12:
                    I.append(dict(t_us=int(p[2]), kind=ESP32_KIND.get(int(p[3]), p[3]), applied=int(p[4]),
                                  seq=int(p[5]), epoch=int(p[6])))
            except ValueError:
                continue
    return J, I


def seq_keys(rows):
    """(epoch, kind, n) with n = seq unwrapped in time order within (epoch, kind)."""
    last = {}
    for r in sorted(rows, key=lambda x: x['t_us']):
        k = (r['epoch'], r['kind'])
        r['n'] = r['seq'] if k not in last else last[k] + ((r['seq'] - last[k]) % 256)
        last[k] = r['n']
    return {(r['epoch'], r['kind'], r['n']): r for r in rows}


def match_esp32(h7, J, I):
    jk, ik = seq_keys(J), seq_keys(I)
    best = None
    for off in range(-3, 4):
        m = sum(1 for r in h7 if (r['epoch'] + off, r['kind'], r['n']) in jk)
        if best is None or m > best[1]:
            best = (off, m)
    off = best[0]
    for r in h7:
        k = (r['epoch'] + off, r['kind'], r['n'])
        r['esp32_spi'] = k in jk
        r['esp32_cb'] = k in ik and ik[k]['applied'] == 1
        r['esp32_t_us'] = jk[k]['t_us'] if k in jk else None
    h7k = {(r['epoch'] + off, r['kind'], r['n']) for r in h7}
    extra = [k for k in jk if k not in h7k]
    return dict(epoch_offset_esp32_minus_h7=off, h7_reports=len(h7), matched_spi=best[1],
                not_at_esp32_spi=len(h7) - best[1], at_spi_not_cb=sum(1 for r in h7 if r['esp32_spi'] and not r['esp32_cb']),
                esp32_without_h7=len(extra))


# ── all ──

def decode(t_send_ns, mosi, miso, esp32_events: Optional[str] = None) -> dict:
    mosi = np.asarray(mosi, dtype=np.uint8)
    miso = np.asarray(miso, dtype=np.uint8)
    rows, st = decode_frames(t_send_ns, mosi, miso)
    fmap = frame_map(rows)
    recs, lst = log_records(rows)
    # nf of the frame that carried each record (reference for 16-bit nf and 32-bit cyc fields)
    carrier_nf, carrier_cyc = {}, {}
    by_carrier = {v['carrier']: (k, v['cyc']) for k, v in fmap.items()}
    last = None
    for r in rows:
        if r['j'] in by_carrier:
            last = by_carrier[r['j']]
        if last is not None:
            carrier_nf[r['j']], carrier_cyc[r['j']] = last
    for rec in recs:
        rec['f'] = parse_record(rec)
    t_send_us = np.asarray(t_send_ns, dtype=np.float64) / 1000.0
    rearm = [near(rec['f']['cyc'], carrier_cyc[rec['carrier_j']]) for rec in recs
             if rec['type'] == REC_REARM and rec['carrier_j'] in carrier_cyc]
    segs = fit_clock(fmap, t_send_us, rearm) if fmap else []
    for rec in recs:
        f = rec['f']
        if 'cyc' not in f or rec['carrier_j'] not in carrier_cyc:
            continue
        ref = carrier_cyc[rec['carrier_j']]
        nf = f.get('nf16')
        if nf is not None:
            k = nf_full(nf, carrier_nf[rec['carrier_j']])
            if k in fmap:
                ref = fmap[k]['cyc']
        rec['cyc_full'] = near(f['cyc'], ref)
        rec['t_pi_us'] = cyc_to_pi(segs, rec['cyc_full']) if segs else None
    ticks = [rec for rec in recs if rec['type'] == REC_TICK and 'cyc_full' in rec]
    tick_bad = sum(1 for a, b in zip(ticks, ticks[1:])
                   if (b['f']['cyc_total'] - a['f']['cyc_total']) != (b['cyc_full'] - a['cyc_full']))
    lst['h7_dropped'] = ticks[-1]['f']['dropped'] if ticks else None
    reps = h7_reports(recs, fmap, mosi, carrier_nf)
    res = dict(frames=st, log=lst, clock=dict(segments=segs, tick_total_mismatch=tick_bad),
               reports=dict(n=len(reps), imu_crc_bad=sum(1 for r in reps if r.get('imu_crc_ok') is False),
                            imu_seq_bad=sum(1 for r in reps if r.get('imu_seq_ok') is False),
                            no_source_frame=sum(1 for r in reps if r['src_frame'] is None),
                            per_kind=dict(Counter(r['kind'] for r in reps))))
    if esp32_events:
        J, I = esp32_lines(esp32_events)
        res['esp32_match'] = match_esp32(reps, J, I)
    res['_records'] = recs
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
        p.append(f"H7 bỏ {lg['h7_dropped']} bản ghi vì đầy")
    if res['clock']['tick_total_mismatch']:
        p.append(f"{res['clock']['tick_total_mismatch']} cặp TICK lệch _cyc_total")
    if rp['imu_crc_bad'] or rp['imu_seq_bad'] or rp['no_source_frame']:
        p.append(f"report: CRC IMU sai {rp['imu_crc_bad']}, imu_seq sai {rp['imu_seq_bad']}, "
                 f"không có frame nguồn {rp['no_source_frame']}")
    m = res.get('esp32_match')
    if m and (m['not_at_esp32_spi'] or m['esp32_without_h7']):
        p.append(f"ghép ESP32: {m['not_at_esp32_spi']} report H7 không tới cổng SPI ESP32, "
                 f"{m['esp32_without_h7']} report ESP32 không có ở H7")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Giải mã frames.npz (S06, MISO bản 4)')
    ap.add_argument('frames')
    ap.add_argument('--esp32-events', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        res = decode(z['t_send_ns'], z['mosi'], z['miso'], a.esp32_events)
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
        cols = ['log_idx', 'kind', 'epoch', 'rpt_seq', 'n', 'queued', 'imu_seq', 'src_nf', 'src_frame', 't_pi_us',
                'imu_crc_ok', 'imu_seq_ok', 'esp32_spi', 'esp32_cb', 'esp32_t_us']
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
