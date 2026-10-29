#!/usr/bin/env python3
"""
s06_t5_score.py — chấm T5 của spec S06 v3.4 (H5: sai số tần số encoder, từng bánh; W15, Y16, Z1, Z5).

Đầu vào: lịch `T5_pass<n>.csv` (rpi5/t5_schedule.py, phát bằng `run_hil.py --omega-steps @CSV`) và
`esp32_events.txt` của s06_logger.py (dòng `E,eseq,t_before,t_after,fl,fr,rl,rr`: count tích lũy
4 bánh, đọc mỗi 1 ms trên đồng hồ ESP32). Mọi thời gian trong phần chấm là **đồng hồ ESP32**
(spec: thạch anh ESP32 làm chuẩn, độc lập với H7); không cần đồng bộ ESP32 → RPi5.

Cách chấm:
  * Căn lịch vào đồng hồ ESP32: t_esp = offset + t_lịch. Offset tìm bằng tương quan chéo giữa tốc độ
    count đo (bin 100 ms) và tốc độ count theo lịch, hoặc nhập tay `--offset-us`. Lệch căn tới cỡ
    vài giây vẫn không đổi kết quả vì cửa sổ đầu và cuối của mỗi lần ghé bị bỏ.
  * Mỗi lần ghé (giữ ω trong `VISIT_WINDOWS` × 10 s) chia thành các cửa sổ 10 s liền nhau từ đầu lần
    ghé; bỏ cửa sổ đầu và cửa sổ cuối, chấm các cửa sổ giữa (5 cửa sổ).
  * Hai đầu cửa sổ là hai lần đọc: lần đọc đầu tiên có t_before ≥ đầu cửa sổ, và lần đọc đầu tiên có
    t_before ≥ đầu cửa sổ + 10 s. Thời điểm một lần đọc = giữa [t_before; t_after]; T = hiệu hai
    thời điểm đó (cửa sổ thực, cỡ 10 s ± 1 ms).
  * f_đo = Δcount / T; f_lệnh = K·ω_eff, K = 1320/2π count/rad, ω_eff = ω_d (ω sau lượng tử frame,
    như T3b) và 0 khi |ω_d| ≤ 0,3 rad/s; e = (f_đo − f_lệnh) / f_lệnh, có dấu (đảo dấu hay lẫn bánh
    cho e ≈ −2 hoặc lớn, nên T5 cũng là nguồn kiểm H4 cho 8 dây encoder).
  * Bất định tương đối u = 1/|K·ω_eff·T| (±1 count) + (thời gian đọc 4 bánh ở hai đầu)/T
    + sai số thạch anh ESP32 (`--xtal-ppm`; mặc định không có nguồn, E6, không cộng và ghi rõ).
  * Mục 4: |e| + u ≤ 0,006 → đạt; |e| − u > 0,006 → không đạt; còn lại chưa kết luận.
    Không có lần đọc trong 100 ms sau một đầu cửa sổ (ngoài dữ liệu, mất dòng E) → `missing`.
  * Điểm chỉ báo 0,29 rad/s: không chấm H5, chỉ ghi số count (kỳ vọng 0).

Thêm theo Liam 30/10 ("tôi đồng ý 3 đề xuất của bạn"):
  * Điều kiện thời gian của spec (lần 1 bắt đầu trong 2 phút sau khi bật, lần 2 sau ≥ 30 phút chạy):
    esp_timer của ESP32 đếm từ lúc ESP32 khởi động, nên offset = thời gian ESP32 đã chạy lúc lịch bắt
    đầu. Lần 1 cần ≤ 120 s, lần 2 cần ≥ 1 800 s; không đạt → `run_condition_ok: false`, mã 3. Chỉ đúng
    khi ESP32 cấp nguồn cùng lúc với H7 và không bị reset giữa chừng (`--no-uptime-check` để bỏ).
  * Lệch nhịp ESP32 so với RPi5 (cho câu thạch anh, E6): t_rx (giờ Pi nhận gói, monotonic) − t_after
    của dòng E; lấy chặn dưới mỗi bin 10 s (gói đến nhanh nhất), khớp đường thẳng → ppm. Chỉ là sai số
    thạch anh ESP32 khi đồng hồ Pi được NTP giữ đúng nhịp (kiểm trên Pi); không tự cộng vào u.
  * Nhiệt độ: `temp_esp32_c` = trung bình trường temp_c của dòng S (cảm biến trong chip ESP32-S3) trong
    cửa sổ; `temp_pi_c` = trung bình `pi_temp.txt` của s06_logger.py (`--pi-temp`, đổi sang đồng hồ
    ESP32 bằng đường khớp trên). Không có số thì để trống. Nhiệt độ H7 chưa có (ADC3, để sau).

Diễn giải của CODE (spec chưa ghi, cần THESIS xác nhận khi khóa):
  (1) f_lệnh theo ω_d (ω đã lượng tử trong frame), không theo ω danh định; cột omega_cmd giữ ω danh định;
  (2) cửa sổ neo vào đầu lần ghé theo offset căn, chứ không neo vào lưới cố định của cả lượt;
  (3) luật đạt/không đạt/chưa kết luận theo mục 4 như T3b; `missing` tính như chưa kết luận;
  (4) mất dòng E giữa hai đầu cửa sổ không ảnh hưởng số đo (count tích lũy); kiểm đủ log (§4.3)
      thuộc tools/s06_esp32_check.py, ở đây chỉ báo số chỗ hở eseq.

Usage:
  python tools/s06_t5_score.py --schedule T5_pass1.csv --esp32-events esp32_events.txt --out <thư mục>
         [--offset-us <µs>] [--count-sign 1,1,1,1] [--xtal-ppm <ppm>] [--pi-temp pi_temp.txt]
         [--no-uptime-check]
Mã thoát: 0 mọi cửa sổ chấm đạt; 1 có cửa sổ không đạt; 3 không có không đạt nhưng có chưa kết luận,
thiếu, hoặc lượt không đúng điều kiện thời gian; 2 lỗi đầu vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
import t3b_schedule as ts  # noqa: E402
import t5_schedule as t5  # noqa: E402

K = ts.K_COUNT
LIMIT = 0.006                 # max |e| ≤ 0,6 %
WINDOW_US = t5.WINDOW_S * 1e6
ALIGN_BIN_US = 100_000.0
MAX_READ_GAP_US = 100_000.0   # đầu cửa sổ phải có lần đọc trong 100 ms sau nó [ĐỀ XUẤT], không thì `missing`
WHEELS = ('FL', 'FR', 'RL', 'RR')
UPTIME_PASS1_MAX_US = 120e6   # spec: lần 1 bắt đầu trong 2 phút sau khi bật
UPTIME_PASS2_MIN_US = 1800e6  # spec: lần 2 sau ≥ 30 phút chạy
CLOCK_BIN_US = 10e6           # bin chặn dưới t_rx − t_esp

COLUMNS = ['idx', 'run_pass', 'visit', 'kind', 'wheel', 'omega_cmd', 'omega_d', 'window',
           't_window_start_us', 't_window_end_us', 'counts', 'f_meas_hz', 'f_cmd_hz', 'err_rel',
           'u_count', 'u_read', 'u_xtal', 'uncertainty', 'verdict', 'temp_esp32_c', 'temp_pi_c']


def omega_eff(w: float) -> float:
    wd = ts.quantize(w)
    return 0.0 if abs(wd) <= ts.DEADBAND else wd


class Reads:
    """Các lần đọc count của ESP32 (đồng hồ ESP32, µs), sắp theo t_before."""

    def __init__(self, t_before, t_after, counts, count_sign=(1, 1, 1, 1), eseq=None):
        self.tb = np.asarray(t_before, dtype=float)
        self.ta = np.asarray(t_after, dtype=float)
        self.c = np.asarray(counts, dtype=np.int64) * np.asarray(count_sign, dtype=np.int64)
        if len(self.tb) < 2 or np.any(np.diff(self.tb) < 0):
            raise ValueError('cần >= 2 lần đọc, t_before tăng dần')
        self.tm = 0.5 * (self.tb + self.ta)
        self.eseq = None if eseq is None else np.asarray(eseq, dtype=np.int64)

    def first_at_or_after(self, t: float) -> Optional[int]:
        """Lần đọc đầu tiên có t_before ≥ t; None khi không có lần đọc nào trong MAX_READ_GAP_US sau t
        (ngoài dữ liệu, hoặc mất dòng E đúng ở đầu cửa sổ)."""
        i = int(np.searchsorted(self.tb, t, side='left'))
        return i if i < len(self.tb) and self.tb[i] - t <= MAX_READ_GAP_US else None

    def eseq_gaps(self) -> int:
        if self.eseq is None or len(self.eseq) < 2:
            return 0
        d = np.diff(np.sort(self.eseq))
        return int(np.sum(d[d > 1] - 1))


def scheduled_rate(sched: Sequence[t5.Visit], n_bins: int, bin_us: float) -> np.ndarray:
    """Tốc độ count theo lịch [bin, bánh] (count/s), bin i phủ [i·bin; (i+1)·bin) từ đầu lịch."""
    p = np.zeros((n_bins, 4))
    for v in sched:
        a = int(np.floor(v.t_start_ms * 1000.0 / bin_us))
        b = int(np.ceil(v.t_end_ms * 1000.0 / bin_us))
        p[a:min(b, n_bins), :] = [K * omega_eff(w) for w in v.omegas]
    return p


def align_offset(reads: Reads, sched: Sequence[t5.Visit], bin_us: float = ALIGN_BIN_US):
    """offset (µs) với t_esp = offset + t_lịch, bằng tương quan chéo tốc độ count; trả (offset, hệ số)."""
    t0 = reads.tm[0]
    n_m = int((reads.tm[-1] - t0) // bin_us)
    if n_m < 2:
        raise ValueError('dữ liệu quá ngắn để căn lịch')
    g = t0 + bin_us * np.arange(n_m + 1)
    m = np.empty((n_m, 4))
    for w in range(4):
        m[:, w] = np.diff(np.interp(g, reads.tm, reads.c[:, w])) / (bin_us * 1e-6)
    n_p = int(np.ceil(sched[-1].t_end_ms * 1000.0 / bin_us))
    p = scheduled_rate(sched, n_p, bin_us)
    corr = sum(np.correlate(m[:, w], p[:, w], mode='full') for w in range(4))
    k = int(np.argmax(corr))
    lag = k - (n_p - 1)                       # bin m[i + lag] ↔ bin p[i]
    norm = float(np.sqrt(np.sum(p * p) * np.sum(m * m)))
    return t0 + lag * bin_us, (float(corr[k]) / norm if norm > 0 else 0.0)


def clock_fit(t_rx_us, t_esp_us, bin_us: float = CLOCK_BIN_US) -> Optional[dict]:
    """t_rx ≈ a + (1 + b)·t_esp trên chặn dưới mỗi bin (gói đến nhanh nhất); None khi < 3 bin."""
    t_rx = np.asarray(t_rx_us, dtype=float)
    t_esp = np.asarray(t_esp_us, dtype=float)
    if len(t_esp) < 3:
        return None
    d = t_rx - t_esp
    k = np.floor((t_esp - t_esp.min()) / bin_us).astype(np.int64)
    xs, ys = [], []
    for b in np.unique(k):
        i = np.flatnonzero(k == b)
        j = i[np.argmin(d[i])]
        xs.append(t_esp[j])
        ys.append(d[j])
    if len(xs) < 3:
        return None
    xs, ys = np.array(xs), np.array(ys)
    b, a = np.polyfit(xs - xs[0], ys, 1)
    a -= b * xs[0]
    resid = ys - (a + b * xs)
    return dict(a_us=float(a), b=float(b), ppm=float(b * 1e6), n_bins=len(xs),
                resid_max_us=float(np.max(np.abs(resid))), span_s=float((xs[-1] - xs[0]) / 1e6))


def pi_to_esp(fit: dict, t_pi_us):
    return (np.asarray(t_pi_us, dtype=float) - fit['a_us']) / (1.0 + fit['b'])


def uptime_condition(run_pass: int, offset_us: float) -> bool:
    return bool(offset_us <= UPTIME_PASS1_MAX_US if run_pass == 1 else offset_us >= UPTIME_PASS2_MIN_US)


def window_mean(series, a: float, b: float):
    """Trung bình giá trị (bỏ nan) có thời điểm trong [a; b); '' khi không có."""
    if series is None:
        return ''
    t, v = series
    i, j = np.searchsorted(t, a, side='left'), np.searchsorted(t, b, side='left')
    x = v[i:j]
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else ''


def score_visit(reads: Reads, v: t5.Visit, offset_us: float, xtal_ppm: Optional[float],
                temps: Optional[dict] = None) -> List[dict]:
    temps = temps or {}
    rows = []
    n_win = int(round((v.t_end_ms - v.t_start_ms) * 1000.0 / WINDOW_US))
    for k in range(1, n_win - 1):              # bỏ cửa sổ đầu và cuối
        s = offset_us + v.t_start_ms * 1000.0 + k * WINDOW_US
        ia, ib = reads.first_at_or_after(s), reads.first_at_or_after(s + WINDOW_US)
        for w in range(4):
            we = omega_eff(v.omegas[w])
            r = dict(run_pass=v.run_pass, visit=v.visit, kind=v.kind, wheel=WHEELS[w],
                     omega_cmd=v.omegas[w], omega_d=ts.quantize(v.omegas[w]), window=k,
                     t_window_start_us='', t_window_end_us='', counts='', f_meas_hz='',
                     f_cmd_hz=K * we, err_rel='', u_count='', u_read='', u_xtal='', uncertainty='',
                     verdict='missing', temp_esp32_c=window_mean(temps.get('esp32'), s, s + WINDOW_US),
                     temp_pi_c=window_mean(temps.get('pi'), s, s + WINDOW_US))
            if ia is None or ib is None:
                rows.append(r)
                continue
            T = reads.tm[ib] - reads.tm[ia]
            n = int(reads.c[ib, w] - reads.c[ia, w])
            f = n / (T * 1e-6)
            r.update(t_window_start_us=reads.tm[ia], t_window_end_us=reads.tm[ib], counts=n, f_meas_hz=f)
            if v.kind == 'indicator' or we == 0.0:
                r['verdict'] = 'indicator_ok' if n == 0 else 'indicator_counts'
                rows.append(r)
                continue
            e = (f - K * we) / (K * we)
            u_c = 1.0 / abs(K * we * T * 1e-6)
            u_r = ((reads.ta[ia] - reads.tb[ia]) + (reads.ta[ib] - reads.tb[ib])) / T
            u_x = 0.0 if xtal_ppm is None else xtal_ppm * 1e-6
            u = u_c + u_r + u_x
            if abs(e) + u <= LIMIT:
                verdict = 'pass'
            elif abs(e) - u > LIMIT:
                verdict = 'fail'
            else:
                verdict = 'inconclusive'
            r.update(err_rel=e, u_count=u_c, u_read=u_r, u_xtal='' if xtal_ppm is None else u_x,
                     uncertainty=u, verdict=verdict)
            rows.append(r)
    return rows


def score_run(reads: Reads, sched: Sequence[t5.Visit], offset_us: float,
              xtal_ppm: Optional[float] = None, temps: Optional[dict] = None) -> List[dict]:
    rows = []
    for v in sched:
        rows += score_visit(reads, v, offset_us, xtal_ppm, temps)
    for i, r in enumerate(rows):
        r['idx'] = i
    return rows


def points(rows: Sequence[dict]) -> List[dict]:
    """max |e| mỗi (lần ghé, bánh) trên các cửa sổ chấm (spec: max|e| ≤ 0,6 % ở mọi điểm)."""
    out: Dict[tuple, dict] = {}
    for r in rows:
        if r['kind'] == 'indicator':
            continue
        key = (r['visit'], r['wheel'])
        p = out.setdefault(key, dict(run_pass=r['run_pass'], visit=r['visit'], kind=r['kind'],
                                     wheel=r['wheel'], omega_cmd=r['omega_cmd'], windows=0,
                                     max_abs_err='', max_uncertainty='', verdict='pass'))
        p['windows'] += 1
        if r['err_rel'] != '':
            p['max_abs_err'] = max(abs(r['err_rel']), p['max_abs_err'] or 0.0)
            p['max_uncertainty'] = max(r['uncertainty'], p['max_uncertainty'] or 0.0)
        rank = {'pass': 0, 'inconclusive': 1, 'missing': 1, 'fail': 2}
        if rank[r['verdict']] > rank[p['verdict']]:
            p['verdict'] = 'fail' if r['verdict'] == 'fail' else 'inconclusive'
    return list(out.values())


def summarize(rows: Sequence[dict]) -> dict:
    scored = [r for r in rows if r['kind'] != 'indicator']
    ind = [r for r in rows if r['kind'] == 'indicator']
    errs = [abs(r['err_rel']) for r in scored if r['err_rel'] != '']
    out = {v: sum(r['verdict'] == v for r in scored) for v in ('pass', 'fail', 'inconclusive', 'missing')}
    out['max_abs_err'] = max(errs) if errs else None
    out['indicator_windows'] = len(ind)
    out['indicator_counts_total'] = int(sum(abs(r['counts']) for r in ind if r['counts'] != ''))
    return out


def exit_code(rows: Sequence[dict]) -> int:
    v = [r['verdict'] for r in rows if r['kind'] != 'indicator']
    if 'fail' in v:
        return 1
    if 'inconclusive' in v or 'missing' in v:
        return 3
    return 0


def write_csv(path: str, rows: Sequence[dict], cols: Sequence[str]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def load_esp32_events(path: str, count_sign=(1, 1, 1, 1)):
    """(Reads, (t_rx, t_after) của dòng E, (t_us, temp_c) của dòng S 16 trường hoặc None)."""
    eseq, tb, ta, c, rx = [], [], [], [], []
    st, sv = [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep:
                continue
            p = line.split(',')
            if p[0] == 'E' and len(p) == 8:
                eseq.append(int(p[1]))
                tb.append(int(p[2]))
                ta.append(int(p[3]))
                c.append([int(x) for x in p[4:8]])
                rx.append(int(t_rx))
            elif p[0] == 'S' and len(p) == 16:
                st.append(int(p[2]))
                sv.append(float(p[15]))
    if not tb:
        raise ValueError(f'{path}: không có dòng E')
    order = np.argsort(tb, kind='stable')
    reads = Reads(np.array(tb)[order], np.array(ta)[order], np.array(c)[order], count_sign,
                  np.array(eseq)[order])
    so = np.argsort(st, kind='stable')
    s_temp = (np.array(st, dtype=float)[so], np.array(sv)[so]) if st else None
    return reads, (np.array(rx, dtype=float), np.array(ta, dtype=float)), s_temp


def load_pi_temp(path: str):
    """pi_temp.txt của s06_logger.py: `<t_us>\\t<m°C>` → (t_pi_us, °C)."""
    t, v = [], []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            a, sep, b = raw.strip().partition('\t')
            if sep:
                t.append(float(a))
                v.append(int(b) / 1000.0)
    return np.array(t), np.array(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T5 (spec S06 v3.4)')
    ap.add_argument('--schedule', required=True, help='T5_pass<n>.csv của rpi5/t5_schedule.py')
    ap.add_argument('--esp32-events', required=True, help='esp32_events.txt của s06_logger.py (dòng E)')
    ap.add_argument('--offset-us', type=float, default=None,
                    help='t_esp_us = OFFSET + t_lịch_us; bỏ trống thì tự căn bằng tương quan chéo')
    ap.add_argument('--count-sign', default='1,1,1,1',
                    help='dấu count ESP32 theo bánh (ENC_PI 2026-10-27_run06: count cùng dấu ω cả 4 bánh)')
    ap.add_argument('--xtal-ppm', type=float, default=None,
                    help='sai số thạch anh ESP32, ppm (chưa có nguồn, E6); bỏ trống = không cộng')
    ap.add_argument('--pi-temp', default=None, help='pi_temp.txt của s06_logger.py (nhiệt độ CPU RPi5)')
    ap.add_argument('--no-uptime-check', action='store_true',
                    help='bỏ kiểm thời gian từ lúc bật (ESP32 không cấp nguồn cùng H7, hoặc bị reset)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        sched = t5.read_csv(a.schedule)
        t5.segments(sched)                      # kiểm lịch liền nhau
        reads, rx, s_temp = load_esp32_events(a.esp32_events, [int(x) for x in a.count_sign.split(',')])
        if a.offset_us is None:
            offset, corr = align_offset(reads, sched)
        else:
            offset, corr = a.offset_us, None
        fit = clock_fit(*rx)
        temps = {'esp32': s_temp}
        if a.pi_temp:
            if fit is None:
                raise ValueError('không đổi được giờ pi_temp sang đồng hồ ESP32 (thiếu dữ liệu khớp đồng hồ)')
            tp, vp = load_pi_temp(a.pi_temp)
            temps['pi'] = (pi_to_esp(fit, tp), vp) if len(tp) else None
    except (OSError, ValueError, KeyError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    rows = score_run(reads, sched, offset, a.xtal_ppm, temps)
    pts = points(rows)
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, 'T5.csv'), rows, COLUMNS)
    write_csv(os.path.join(a.out, 'T5_points.csv'), pts, list(pts[0]) if pts else ['visit'])
    summ = summarize(rows)
    run_pass = sched[0].run_pass
    cond = None if a.no_uptime_check else uptime_condition(run_pass, offset)
    code = exit_code(rows)
    if code == 0 and cond is False:
        code = 3
    offset = float(offset)
    summ.update(run_pass=run_pass, offset_us=offset, offset_source='manual' if corr is None else 'xcorr',
                align_corr=corr, n_reads=len(reads.tb), eseq_gaps=reads.eseq_gaps(),
                esp32_uptime_at_start_s=offset / 1e6, run_condition_ok=cond,
                run_condition_rule='lần 1: ≤ 120 s; lần 2: ≥ 1800 s (uptime ESP32 lúc lịch bắt đầu)',
                clock_vs_pi=fit,
                clock_note='ppm = nhịp đồng hồ Pi (monotonic) so với ESP32; chỉ là sai số thạch anh ESP32 '
                           'khi Pi được NTP giữ nhịp; không cộng vào bất định',
                temp_esp32=s_temp is not None, temp_pi=temps.get('pi') is not None,
                xtal_ppm=a.xtal_ppm, xtal_note=None if a.xtal_ppm is not None else
                'sai số thạch anh ESP32 chưa có nguồn (E6), không cộng vào bất định',
                count_sign=a.count_sign, exit_code=code)
    with open(os.path.join(a.out, 'T5_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps(summ))
    return summ['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
