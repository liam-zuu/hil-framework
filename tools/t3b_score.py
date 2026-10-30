#!/usr/bin/env python3
"""
t3b_score.py — chấm T3b của spec S06 v3.4 (chuyển tiếp encoder: H3 + H5, gồm lỗi A11).

Đầu vào (mọi thời điểm trên MỘT đồng hồ tham chiếu, µs; lượt thật: đồng hồ RPi5):
  * frame thực gửi: t_send[k], ω_d[k][4] (ω đã lượng tử trong frame, giải từ MOSI);
  * các lần đọc count của ESP32: t_before, t_after, c[4] (count tích lũy, 1 ms một lần);
  * (tùy chọn) các lần gọi `_start_wheel`/`_stop` của H7 kèm CNT (việc (5), chưa có trên board);
  * lịch `T3b_schedule.csv` (rpi5/t3b_schedule.py) cho cửa sổ và loại chuyển tiếp.

Luật (spec T3b, Z1, Z2, Y2, Y4, V2-6, V2-7, V3-6; mục 4):
  * ω_eff = 0 khi |ω_d| ≤ 0,3 rad/s; ω_eff của frame k giữ từ t_send[k] tới t_send[k+1].
  * Hai đầu cửa sổ là chính hai lần đọc: lần đọc đầu tiên có t_before ≥ t_send[win_start]
    và lần đọc đầu tiên có t_before ≥ t_send[win_end]. Thời điểm của một lần đọc = giữa
    [t_before; t_after]; bất định của nó nằm trong u_t.
  * (b) e = Δcount − K·∫ω_eff dt giữa hai thời điểm đọc, K = 1320/2π.
    N_tol = K·[0,006·∫_{|ω_eff|≥0,31983}|ω_eff| + ∫_{0<|ω_eff|<0,31983}|ω_eff| + d_max·Σ|Δω_eff|]
            + 1 + n_start + 2·n_dừng, trên các frame giao [t_a; t_b] (Δt thực, không giả 1 ms).
    u_t = K·Σ_{hai đầu}[(t_after − t_before) + δ_sync]·max|ω_eff| trên [t_before − δ_sync; t_after + δ_sync].
    Mục 4: |e| + u_t ≤ N_tol → đạt; |e| − u_t > N_tol → không đạt; còn lại chưa kết luận.
    Yêu cầu công cụ: u_t ≤ ¼·(N_tol − số hạng cố định) (cột u_t_ok).
  * (a) bước đếm ngược = khoảng giữa hai lần đọc liên tiếp trong cửa sổ có count ròng ngược dấu
    lệnh, khi mọi frame giao khoảng [t_before_i; t_after_(i+1)] có cùng dấu ω_eff ≠ 0; loại khoảng
    giao [sự kiện khởi động/dừng theo lịch; + 1 ms]. Đạt khi 0.
  * Sự kiện theo lịch từ chuỗi ω_eff của frame (t3b_schedule._events): khởi động = 0 → ≠ 0 hoặc
    đổi dấu; dừng = ≠ 0 → 0 hoặc đổi dấu; đứng yên từ frame 0 là một khoảng dừng.
  * Số lần gọi thực khác số sự kiện theo lịch trong cửa sổ → lỗi sản phẩm, cửa sổ không đạt.

Diễn giải của CODE (spec chưa định nghĩa chi tiết, cần THESIS xác nhận khi khóa):
  start_phase_counts = n_start (phần dung sai của khởi động), stop_interval_counts = 2·n_dừng
  (phần dung sai của khoảng dừng); count đo được lúc đứng yên báo riêng ở stop_meas_counts;
  start_cnt = CNT timer lúc `_start_wheel` (nhiều lần thì nối bằng ';').

CLI (lượt thật; đồng bộ ESP32 → RPi5 chưa có chân đánh dấu nên phải nhập tay, ghi vào kết quả):
  python tools/t3b_score.py --schedule T3b_schedule.csv --frames frames.npz [--h7-calls] \
      --esp32-events esp32_events.txt --clock-affine <offset_us> <rate> --delta-sync-us <µs> --out <thư mục>
Sửa theo Codex Sol vòng 1/2 (Liam duyệt 30/10 "Duyệt, sửa hết"):
  * S4 (Z19): hở `eseq`/tràn ở #run_end trong khoảng chấm → không đạt (tools/s06_stream.py).
  * S15: `--h7-calls` bắt buộc ở chế độ nghiệm thu; hở số lần gọi hoặc luồng log H7 → không đạt; giờ H7
    chưa hiệu chuẩn (mốc xung + d_1w) → kết luận tối đa mã 3.
  * S16: trễ lịch → frame dò từ chính MOSI (run_hil --omega-schedule: 1 frame), cửa sổ neo theo đó.
  * S17: count int32 mở vòng 32 bit.
  * S18: không có cửa sổ nào → mã 3; lịch phải là lịch khóa `make_schedule(--locked-seed)` (seed 1).
Mã thoát: 0 mọi cửa sổ đạt và đủ bằng chứng; 1 có cửa sổ không đạt hoặc mất dữ liệu; 3 không có không đạt
nhưng có chưa kết luận, u_t vượt yêu cầu công cụ, hoặc thiếu bằng chứng (lần gọi H7, giờ H7 hiệu chuẩn,
#run_end, lịch khóa); 2 lỗi đầu vào (kể cả MOSI không theo lịch).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'rpi5')]
import s06_stream as stm  # noqa: E402
import t3b_schedule as ts  # noqa: E402

K = ts.K_COUNT
EVENT_EXCL_US = 1000.0        # (a): loại khoảng giao [sự kiện; + 1 ms]
LOCKED_SEED = 1               # spec Phần D (đề xuất điền giá trị khóa, Liam "Theo khuyến nghị"): lịch T3b seed 1
LAG_MATCH_MIN = 0.999         # S16: the frames must follow the schedule at the lag found

COLUMNS = ['idx', 'id', 'wheel', 'transition', 'rep', 'omega_from', 'omega_to',
           't_read_before_us', 't_read_after_us', 't_read_before_end_us', 't_read_after_end_us',
           'window_start_us', 'window_end_us', 'start_cnt', 'count_meas', 'count_cmd', 'err_counts',
           'n_tol', 'uncertainty', 'u_t_limit', 'u_t_ok', 'backward_steps', 'start_phase_counts',
           'stop_interval_counts', 'stop_meas_counts', 'n_calls_actual', 'n_calls_expected',
           'deadband_counts', 'h5_part', 'low_part', 'delay_part', 'fixed', 'verdict_b', 'verdict']


def eff(v: float) -> float:
    return 0.0 if abs(v) <= ts.DEADBAND else v


class Run:
    """One run on one reference clock (µs)."""

    def __init__(self, t_send_us, w_d, t_before_us, t_after_us, counts, calls=None,
                 count_sign=(1, 1, 1, 1), delta_sync_us: float = 0.0, frame_lag: int = 0):
        self.t_send = np.asarray(t_send_us, dtype=float)
        self.w_d = np.asarray(w_d, dtype=float)
        if len(self.t_send) < 2 or np.any(np.diff(self.t_send) <= 0):
            raise ValueError('t_send must have >= 2 frames in strictly increasing order')
        self.t_end_frames = self.t_send[-1] + (self.t_send[-1] - self.t_send[-2])
        self.w_eff = np.where(np.abs(self.w_d) <= ts.DEADBAND, 0.0, self.w_d)
        self.tb = np.asarray(t_before_us, dtype=float)
        self.ta = np.asarray(t_after_us, dtype=float)
        # S17: int32 cumulative counts unwrapped before any difference
        self.c = (stm.unwrap_i32(counts) if len(counts) else np.zeros((0, 4), dtype=np.int64)) \
            * np.asarray(count_sign, dtype=np.int64)
        # S16: schedule entry k is in the MOSI of frame k + frame_lag (main() finds it with schedule_lag)
        self.frame_lag = int(frame_lag)
        if len(self.tb) and np.any(np.diff(self.tb) <= 0):
            raise ValueError('reads must be in strictly increasing t_before order')
        self.calls = calls          # None = not logged; else wheel -> [(kind 'S'|'P', t_us, cnt)]
        self.delta_sync = float(delta_sync_us)
        self.events = []
        for w in range(4):
            starts, intervals = ts._events([float(v) for v in self.w_eff[:, w]])
            self.events.append((starts, intervals))

    # ── piecewise-constant frames ──
    def _t_frame_end(self, k: int) -> float:
        return self.t_send[k + 1] if k + 1 < len(self.t_send) else self.t_end_frames

    def overlaps(self, a: float, b: float):
        """[(k, overlap_s)] of the frames whose [t_send[k], t_send[k+1]) meets [a, b]."""
        out = []
        k = max(0, bisect.bisect_right(self.t_send, a) - 1)
        while k < len(self.t_send) and self.t_send[k] <= b:
            lo, hi = max(a, self.t_send[k]), min(b, self._t_frame_end(k))
            if hi > lo or (a == b and self.t_send[k] <= a < self._t_frame_end(k)):
                out.append((k, max(0.0, hi - lo) * 1e-6))
            k += 1
        return out

    def integral(self, w: int, a: float, b: float) -> float:
        return sum(self.w_eff[k, w] * dt for k, dt in self.overlaps(a, b))

    def max_abs(self, w: int, a: float, b: float) -> float:
        ov = self.overlaps(a, b)
        return max((abs(self.w_eff[k, w]) for k, _ in ov), default=0.0)

    def read_after(self, t: float) -> Optional[int]:
        i = int(np.searchsorted(self.tb, t, side='left'))
        return i if i < len(self.tb) else None

    def t_read(self, i: int) -> float:
        return 0.5 * (self.tb[i] + self.ta[i])

    # ── one window ──
    def tolerance(self, w: int, a: float, b: float) -> dict:
        ov = self.overlaps(a, b)
        s1 = sum(abs(self.w_eff[k, w]) * dt for k, dt in ov if abs(self.w_eff[k, w]) >= ts.H5_LOW)
        s2 = sum(abs(self.w_eff[k, w]) * dt for k, dt in ov if 0.0 < abs(self.w_eff[k, w]) < ts.H5_LOW)
        ks = [k for k, _ in ov]
        tv = sum(abs(self.w_eff[k, w] - self.w_eff[k - 1, w]) for k in ks[1:])
        db = sum(abs(self.w_d[k, w]) * dt for k, dt in ov if abs(self.w_d[k, w]) <= ts.DEADBAND)
        starts, intervals = self.events[w]
        n_start = sum(1 for k in starts if a <= self.t_send[k] <= b)
        n_stop_events = sum(1 for s, _ in intervals if s > 0 and a <= self.t_send[s] <= b)
        n_stop_iv = 0
        for s, e in intervals:
            t_s = self.t_send[s]
            t_e = self.t_send[e] if e < len(self.t_send) else math.inf
            if t_s <= b and t_e >= a:
                n_stop_iv += 1
        fixed = 1 + n_start + 2 * n_stop_iv
        return dict(h5_part=K * ts.H5_TOL * s1, low_part=K * s2, delay_part=K * ts.D_MAX_S * tv,
                    n_start=n_start, n_stop_events=n_stop_events, n_stop_intervals=n_stop_iv, fixed=fixed,
                    n_tol=K * (ts.H5_TOL * s1 + s2 + ts.D_MAX_S * tv) + fixed, deadband_counts=K * db)

    def event_times(self, w: int) -> List[float]:
        starts, intervals = self.events[w]
        ks = sorted(set(starts) | {s for s, _ in intervals if s > 0})
        return [self.t_send[k] for k in ks]

    def score(self, r) -> dict:
        w = r.wheel
        if r.win_end_ms + self.frame_lag >= len(self.t_send):
            raise ValueError(f'window {r.id} ends after the last frame')
        Ts, Te = self.t_send[r.win_start_ms + self.frame_lag], self.t_send[r.win_end_ms + self.frame_lag]
        ia, ib = self.read_after(Ts), self.read_after(Te)
        if ia is None or ib is None:
            raise ValueError(f'window {r.id}: no read after its start or end')
        t_a, t_b = self.t_read(ia), self.t_read(ib)
        meas = int(self.c[ib, w] - self.c[ia, w])
        cmd = K * self.integral(w, t_a, t_b)
        e = meas - cmd
        tol = self.tolerance(w, t_a, t_b)
        u = 0.0
        for i in (ia, ib):
            lo, hi = self.tb[i] - self.delta_sync, self.ta[i] + self.delta_sync
            u += K * ((self.ta[i] - self.tb[i] + self.delta_sync) * 1e-6) * self.max_abs(w, lo, hi)
        limit = 0.25 * (tol['n_tol'] - tol['fixed'])
        if abs(e) + u <= tol['n_tol']:
            vb = 'pass'
        elif abs(e) - u > tol['n_tol']:
            vb = 'fail'
        else:
            vb = 'inconclusive'
        # (a) backward steps
        ev = self.event_times(w)
        back, stop_meas = 0, 0
        for i in range(ia, ib):
            lo, hi = self.tb[i], self.ta[i + 1]
            if any(t_ev <= hi and t_ev + EVENT_EXCL_US >= lo for t_ev in ev):
                continue
            signs = {int(np.sign(self.w_eff[k, w])) for k, _ in self.overlaps(lo, hi)}
            d = int(self.c[i + 1, w] - self.c[i, w])
            if signs == {0}:
                stop_meas += abs(d)
            elif len(signs) == 1 and d * signs.pop() < 0:
                back += 1
        # calls (H7 log): expected = scheduled events in [t_a, t_b]; the call lands up to d_max later
        have_calls = self.calls is not None
        calls = [c for c in (self.calls or {}).get(w, []) if t_a <= c[1] <= t_b + ts.D_MAX_S * 1e6]
        n_exp = tol['n_start'] + tol['n_stop_events']
        n_act = len(calls) if have_calls else None
        calls_ok = (not have_calls) or (sum(1 for c in calls if c[0] == 'S') == tol['n_start']
                                        and sum(1 for c in calls if c[0] == 'P') == tol['n_stop_events'])
        if back > 0 or vb == 'fail' or not calls_ok:
            verdict = 'fail'
        elif vb == 'inconclusive':
            verdict = 'inconclusive'
        else:
            verdict = 'pass'
        return dict(id=r.id, wheel=w, transition=r.kind, omega_from=r.w_from, omega_to=r.w_to,
                    t_read_before_us=self.tb[ia], t_read_after_us=self.ta[ia],
                    t_read_before_end_us=self.tb[ib], t_read_after_end_us=self.ta[ib],
                    window_start_us=t_a, window_end_us=t_b,
                    start_cnt=';'.join(str(c[2]) for c in calls if c[0] == 'S') if have_calls else '',
                    count_meas=meas, count_cmd=cmd, err_counts=e, n_tol=tol['n_tol'], uncertainty=u,
                    u_t_limit=limit, u_t_ok=u <= limit + 1e-12, backward_steps=back,
                    start_phase_counts=tol['n_start'], stop_interval_counts=2 * tol['n_stop_intervals'],
                    stop_meas_counts=stop_meas, n_calls_actual=n_act if n_act is not None else '',
                    n_calls_expected=n_exp, deadband_counts=tol['deadband_counts'],
                    h5_part=tol['h5_part'], low_part=tol['low_part'], delay_part=tol['delay_part'],
                    fixed=tol['fixed'], verdict_b=vb, verdict=verdict)


def score_run(run: Run, sched: Sequence) -> List[dict]:
    reps: Dict[tuple, int] = {}
    out = []
    for r in sorted(sched, key=lambda x: (x.wheel, x.t0_ms)):
        key = (r.wheel, r.kind)
        reps[key] = reps.get(key, 0) + 1
        row = run.score(r)
        row['rep'] = reps[key]
        out.append(row)
    out.sort(key=lambda x: x['id'])
    for i, row in enumerate(out):              # S31: common row index of the S06 CSVs (spec mục 5)
        row['idx'] = i
    return out


def summarize(rows: Sequence[dict]) -> dict:
    groups: Dict[tuple, Dict[str, int]] = {}
    for x in rows:
        g = groups.setdefault((x['transition'], x['wheel']), {'pass': 0, 'fail': 0, 'inconclusive': 0, 'u_t_bad': 0})
        g[x['verdict']] += 1
        g['u_t_bad'] += 0 if x['u_t_ok'] else 1
    return groups


def exit_code(rows: Sequence[dict]) -> int:
    if any(x['verdict'] == 'fail' for x in rows):
        return 1
    if not rows or any(x['verdict'] == 'inconclusive' or not x['u_t_ok'] for x in rows):
        return 3                                    # S18: no window scored is no pass
    return 0


def schedule_lag(w_d, sched, max_lag: int = 3):
    """S16: frame lag of the schedule in the frames actually sent: entry k of the ω table (quantized) is
    in the MOSI of frame k + lag (run_hil --omega-schedule: lag 1, the frame carries the plant state
    before the step). Returns (lag, fraction of frames equal) of the best lag."""
    tab = np.array(ts.omega_table(sched), dtype=float).T          # [ms, wheel]
    q = np.vectorize(ts.quantize)(tab)
    w_d = np.asarray(w_d, dtype=float)
    best = (0, -1.0)
    for lag in range(max_lag + 1):
        n = min(len(q), len(w_d) - lag)
        if n <= 0:
            continue
        frac = float(np.mean(np.all(np.abs(w_d[lag:lag + n] - q[:n]) < 1e-9, axis=1)))
        if frac > best[1]:
            best = (lag, frac)
    return best


def schedule_matches_locked(sched, seed: int = LOCKED_SEED) -> List[str]:
    """S18: the schedule must be the locked one (rpi5/t3b_schedule.make_schedule(seed), spec Phần D:
    seed 1): every kind, wheel, repetition and the random start delays."""
    ref = ts.make_schedule(seed)
    key = lambda r: (r.wheel, r.kind, r.t0_ms, r.win_start_ms, r.win_end_ms)  # noqa: E731
    if sorted(key(r) for r in sched) == sorted(key(r) for r in ref):
        return []
    return [f'lịch khác lịch khóa make_schedule({seed}) ({len(sched)} chuyển tiếp so với {len(ref)})']


def write_rows(path: str, rows: Sequence[dict]) -> None:
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        wr.writeheader()
        for x in rows:
            wr.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in x.items()})


# ── real run ──

def load_frames(path: str):
    from protocol import decode_state       # rpi5/protocol.py
    z = np.load(path)
    t_send_us = z['t_send_ns'].astype(np.float64) / 1000.0
    w_d = np.array([decode_state(bytes(m))[0:4] for m in z['mosi']])
    return t_send_us, w_d


def load_esp32_reads(path: str, offset_us: float, rate: float):
    """Reads on the Pi clock (t_pi = offset + rate·t_esp) and, for the Z19 check, eseq + t_before on the
    ESP32 clock."""
    tb, ta, c, eseq, tb_esp = [], [], [], [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            _, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line.startswith('E,'):
                continue
            p = line.split(',')
            if len(p) != 8:
                continue
            tb.append(offset_us + rate * int(p[2]))
            ta.append(offset_us + rate * int(p[3]))
            c.append([int(x) for x in p[4:8]])
            eseq.append(int(p[1]))
            tb_esp.append(int(p[2]))
    order = np.argsort(tb, kind='stable')
    return (np.array(tb)[order], np.array(ta)[order], np.array(c)[order],
            np.array(eseq)[order], np.array(tb_esp)[order])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T3b (spec S06 v3.4)')
    ap.add_argument('--schedule', required=True)
    ap.add_argument('--frames', required=True, help='frames.npz của run_hil.py --frame-log')
    ap.add_argument('--esp32-events', required=True, help='esp32_events.txt của s06_logger.py (dòng E)')
    ap.add_argument('--clock-affine', nargs=2, type=float, required=True, metavar=('OFFSET_US', 'RATE'),
                    help='t_pi_us = OFFSET + RATE * t_esp_us (từ chân đánh dấu; chưa có công cụ tự tính)')
    ap.add_argument('--delta-sync-us', type=float, required=True, help='bất định đồng bộ ESP32 → RPi5, µs')
    ap.add_argument('--count-sign', default='1,1,1,1',
                    help='dấu count ESP32 theo bánh (ENC_PI 2026-10-27_run06: count cùng dấu ω cả 4 bánh)')
    ap.add_argument('--h7-calls', action='store_true',
                    help='lấy các lần gọi _start_wheel/_stop từ luồng log H7 trong frames.npz '
                         '(bản ghi 0x35/0x36, cần H7 có EncoderGen với móc sự kiện); bắt buộc ở chế độ '
                         'nghiệm thu (S15), không có thì kết luận tối đa mã 3')
    ap.add_argument('--locked-seed', type=int, default=LOCKED_SEED, help='seed của lịch khóa (S18)')
    ap.add_argument('--out', required=True)
    import s06_frames_decode as fd
    fd.add_cal_args(ap)
    a = ap.parse_args(argv)
    notes, fails = [], []
    try:
        sched = ts.read_csv(a.schedule)
        t_send, w_d = load_frames(a.frames)
        tb, ta, c, eseq, tb_esp = load_esp32_reads(a.esp32_events, *a.clock_affine)
        lag, frac = schedule_lag(w_d, sched)
        if frac < LAG_MATCH_MIN:
            raise ValueError(f'frame MOSI không theo lịch (khớp tốt nhất {frac:.4f} ở trễ {lag} frame)')
        calls = None
        if a.h7_calls:
            z = np.load(a.frames)
            res = fd.decode(z['t_send_ns'], z['mosi'], z['miso'], cal=fd.load_cal(z, a))
            calls = res['_calls']
            if any(t is None for v in calls.values() for _, t, *_ in v):
                raise ValueError('có lần gọi H7 không quy được ra giờ Pi (thiếu khớp đồng hồ)')
            # S15: the stream of the call records must be whole (Z19), and the H7 times calibrated (S10)
            if res['enc_calls']['count_gaps']:
                fails.append(f"số lần gọi _start_wheel/_stop hở {res['enc_calls']['count_gaps']} bản ghi (Z19)")
            if res['log']['gaps'] or res['log'].get('h7_dropped'):
                fails.append(f"luồng log H7 hở {res['log']['gaps']}, H7 bỏ {res['log'].get('h7_dropped')} (Z19)")
            if not res['clock']['calibrated']:
                notes.append('giờ H7 chưa hiệu chuẩn (thiếu mốc xung/d_1w): giờ lần gọi chỉ tương đối')
        else:
            notes.append('không dùng --h7-calls: không so số lần gọi với lịch (S15)')
        run = Run(t_send, w_d, tb, ta, c, calls=calls, count_sign=[int(x) for x in a.count_sign.split(',')],
                  delta_sync_us=a.delta_sync_us, frame_lag=lag)
        rows = score_run(run, sched)
    except (OSError, ValueError, KeyError) as ex:
        print(f'lỗi đầu vào: {ex}', file=sys.stderr)
        return 2
    # S4 (Z19): lost encoder reads inside the scored span, or an ESP32 overflow, is a failure
    marks = stm.run_marks(a.esp32_events)
    off, rate = a.clock_affine
    w0 = min(r['window_start_us'] for r in rows) if rows else 0.0
    w1 = max(r['window_end_us'] for r in rows) if rows else 0.0
    stream = stm.enc_stream(eseq, tb_esp, (w0 - off) / rate - 1000.0, (w1 - off) / rate + 1000.0, marks)
    notes += stm.stream_notes(stream)
    if not stream['ok']:
        fails.append('luồng đọc encoder ESP32 mất dữ liệu (Z19)')
    sched_bad = schedule_matches_locked(sched, a.locked_seed)
    notes += sched_bad
    os.makedirs(a.out, exist_ok=True)
    write_rows(os.path.join(a.out, 'T3b.csv'), rows)
    with open(os.path.join(a.out, 'T3b_score_args.txt'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    for (kind, w), g in sorted(summarize(rows).items()):
        print(f'{kind:22s} bánh {w}: đạt {g["pass"]}, không đạt {g["fail"]}, chưa kết luận {g["inconclusive"]}, '
              f'u_t vượt {g["u_t_bad"]}')
    print(f'trễ lịch → frame: {lag} frame (khớp {frac:.4f})')
    for x in fails:
        print(f'KHÔNG ĐẠT: {x}')
    for x in notes:
        print(f'! {x}')
    code = exit_code(rows)
    if fails:
        code = 1
    elif code == 0 and (calls is None or sched_bad or not stream['known']
                        or (calls is not None and any('chưa hiệu chuẩn' in x for x in notes))):
        code = 3                                    # acceptance evidence missing: no pass
    return code


if __name__ == '__main__':
    raise SystemExit(main())
