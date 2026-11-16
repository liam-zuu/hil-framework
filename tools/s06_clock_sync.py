#!/usr/bin/env python3
"""
s06_clock_sync.py — đồng hồ ESP32 ↔ RPi5 từ chân đánh dấu (S06 T3; thiết kế
03_CODE/docs/2026-11-15_Thiet_ke_S06_T3_danh_dau_v1.md mục 4 ý 3, Liam 15/11 "Duyệt, code luôn").

Hai phía của cùng một cạnh lên:
  * RPi5: `run_hil.py --marker t3|10hz --frame-log frames.npz` nâng GPIO17 ngay trước `transfer` của frame có
    `marker` = 1; giờ cạnh = giữa `t_pulse0_ns` và `t_pulse1_ns` (hai lần đọc đồng hồ kẹp lệnh nâng chân),
    nửa khoảng kẹp là bất định phía Pi;
  * ESP32 (`s06_load`): dòng `T3,qseq,t_us,M,38,lvl,gseq,run` (ISR GPIO38, `esp_timer`), mọi lượt và ngoài lượt.
Ghép cặp: lấy độ lệch thô từ khoảng cách giữa các cạnh (dãy khoảng cách 100 ms ở 10hz, 60–120 ms ngẫu nhiên ở
t3 nên không nhầm cặp), rồi ghép tham lam theo đường affine cập nhật dần, cuối cùng bình phương nhỏ nhất:
    t_pi_us = OFFSET + RATE · t_esp_us        (cùng quy ước `--clock-affine` của t3b_score.py)
Phần dư = giờ Pi đo − giờ Pi suy từ ESP32, gồm cả trễ ngắt ESP32 thay đổi và nhiễu kẹp phía Pi.
`delta_sync_us` = phần dư tuyệt đối lớn nhất + nửa khoảng kẹp lớn nhất (cận trên, [ĐỀ XUẤT CODE_S06]); trễ ngắt
trung bình nằm trong OFFSET (cạnh M và các cạnh ESP32 khác qua cùng đường ISR).

Dùng cho:
  * t3b_score.py `--clock-affine OFFSET RATE --delta-sync-us …` (trước đây nhập tay);
  * T4 (4), T6, cờ X17 của T1: quy giờ ESP32 về giờ Pi;
  * khối tải nặng (spec Z8, THESIS nhắc): khoảng in H7 (`pwm_es`, ≥ 2 s) có giờ Pi qua khớp CYCCNT với mốc
    xung của s06_frames_decode; đổi vai bánh của `s06_load` (mỗi 15 s) có giờ ESP32 (`#run_start` timestamp_us +
    S06_HEAVY_SWAP_US). Đặt khoảng in lên đồng hồ khối tải nặng = đổi giờ ESP32 của mốc đổi vai sang giờ Pi bằng
    affine này (t_pi = OFFSET + RATE · t_esp), rồi so với hai đầu khoảng in; bất định thêm `delta_sync_us`.
    Công cụ đủ tải chưa dùng file này (việc của bộ chấm khối tải nặng).

Mất dữ liệu: cạnh ESP32 đã chọn không ghép được với cạnh Pi nào, hay cạnh Pi nằm trong khoảng giờ của các cạnh
ESP32 đã chọn mà không có cặp → mã 1. Ít hơn MIN_PAIRS cặp → mã 3 `thieu_bang_chung`. Cạnh Pi ngoài khoảng đó (Pi
chạy trước hay sau lượt ESP32) không tính là mất.

CLI:
  python tools/s06_clock_sync.py --frames frames.npz --esp32-events esp32_events.txt [--run N] --out clock.json
    --run N: chỉ cạnh M có run = N (lượt đang chạy của s06_load); bỏ trống = mọi cạnh M trong file.
Mã thoát: 0 đủ cặp, không mất; 1 mất cạnh; 3 thiếu bằng chứng; 2 lỗi đầu vào. Dòng cuối `#ket_qua,ma=…`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s06_esp32_check as chk   # noqa: E402

MIN_PAIRS = 3                 # ít nhất 3 cặp mới khớp affine có phần dư
MATCH_TOL_US = 3000.0         # cửa sổ ghép quanh giờ dự đoán (≪ 50 ms, khoảng cách nhỏ nhất của cạnh)
COARSE_N = 50                 # số cạnh ESP32 đầu dùng để tinh độ lệch thô
COARSE_WIN_US = 20_000.0      # < nửa khoảng cách nhỏ nhất giữa hai cạnh (50 ms ở T3, 100 ms ở 10hz)
REFIT_EVERY = 20


def pi_marks(z) -> Tuple[np.ndarray, np.ndarray]:
    """(giờ cạnh µs, nửa khoảng kẹp µs) của các frame có marker = 1 trong frames.npz."""
    files = getattr(z, 'files', z)
    for k in ('marker', 't_pulse0_ns', 't_pulse1_ns'):
        if k not in files:
            raise ValueError(f'frames.npz thiếu "{k}" (run_hil.py chưa chạy với --marker)')
    m = np.asarray(z['marker']).astype(bool)
    p0 = np.asarray(z['t_pulse0_ns'], dtype=np.int64)[m]
    p1 = np.asarray(z['t_pulse1_ns'], dtype=np.int64)[m]
    if np.any(p1 < p0):
        raise ValueError('t_pulse1_ns < t_pulse0_ns ở frame có marker')
    return (p0 + p1) / 2000.0, (p1 - p0) / 2000.0


def esp_marks(events, run: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(giờ ESP32 µs, giờ Pi nhận dòng t_rx µs) của các cạnh M, theo giờ ESP32; `events` = kết quả
    s06_esp32_check.parse_events."""
    rows = [r for r in events['T3'] if r['src'] == 'M' and (run is None or r['run'] == run)]
    rows.sort(key=lambda r: r['t_us'])
    return (np.array([r['t_us'] for r in rows], dtype=np.float64),
            np.array([r['t_rx'] for r in rows], dtype=np.float64))


def _fit(e: Sequence[float], p: Sequence[float]) -> Tuple[float, float]:
    e = np.asarray(e, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(e) < 2:
        return float(p[0] - e[0]), 1.0
    e0 = e.mean()
    b, a0 = np.polyfit(e - e0, p, 1)
    return float(a0 - b * e0), float(b)


def coarse_offset(p: np.ndarray, e: np.ndarray, e_rx: np.ndarray) -> Optional[float]:
    """Độ lệch thô o (t_pi ≈ t_esp + o). Cạnh 10 Hz cách đều nên không ghép được theo khoảng cách (mọi bội 100 ms
    đều khớp); neo bằng giờ Pi nhận dòng (`t_rx`, `time.monotonic_ns` của s06_logger chạy trên chính Pi, cùng
    đồng hồ với frames.npz): t_rx = giờ cạnh trên Pi + trễ gửi (≥ 0), nên o0 = phân vị 1 % của t_rx − t_esp. Rồi
    tinh: trung vị của (cạnh Pi gần nhất − (t_esp + o0)) trên các cạnh lệch < COARSE_WIN_US. None = không cạnh
    nào khớp (logger không chạy trên Pi, hay hai bên không cùng lượt)."""
    if len(e) == 0 or len(p) == 0:
        return None
    o0 = float(np.percentile(e_rx - e, 1.0))
    q = e[:COARSE_N] + o0
    idx = np.clip(np.searchsorted(p, q), 1, len(p) - 1) if len(p) > 1 else np.zeros(len(q), int)
    near = np.where(np.abs(p[idx] - q) <= np.abs(p[idx - 1] - q), p[idx], p[idx - 1]) if len(p) > 1 else p[idx]
    d = near - q
    d = d[np.abs(d) < COARSE_WIN_US]
    if len(d) == 0:
        return None
    return o0 + float(np.median(d))


def match(p: np.ndarray, e: np.ndarray, e_rx: np.ndarray, tol_us: float = MATCH_TOL_US) -> List[Tuple[int, int]]:
    """Cặp (chỉ số Pi, chỉ số ESP32), p và e tăng dần: độ lệch thô (coarse_offset), rồi ghép tham lam theo thứ tự
    Pi, mỗi cạnh ESP32 dùng một lần, khớp lại affine sau mỗi REFIT_EVERY cặp (trôi tần số thạch anh)."""
    o = coarse_offset(p, e, e_rx)
    if o is None:
        return []
    a, b = o, 1.0                                # t_pi = a + b·t_esp, ban đầu b = 1
    pairs: List[Tuple[int, int]] = []
    used = set()
    for i in range(len(p)):
        target = (p[i] - a) / b
        j = int(np.searchsorted(e, target))
        cand = [c for c in (j - 1, j) if 0 <= c < len(e) and c not in used]
        if not cand:
            continue
        c = min(cand, key=lambda c: abs(e[c] - target))
        if abs(e[c] - target) <= tol_us:
            pairs.append((i, c))
            used.add(c)
            if len(pairs) % REFIT_EVERY == 0:
                a, b = _fit([e[q] for _, q in pairs], [p[q] for q, _ in pairs])
    return pairs


def sync(p: np.ndarray, half: np.ndarray, e: np.ndarray, e_rx: np.ndarray) -> dict:
    """Khớp affine t_pi = offset + rate·t_esp trên các cặp cạnh; p, e tăng dần."""
    pairs = match(p, e, e_rx)
    out = dict(n_pi=int(len(p)), n_esp=int(len(e)), n_pairs=len(pairs), offset_us=None, rate=None,
               res_max_us=None, res_rms_us=None, half_bracket_max_us=None, delta_sync_us=None,
               esp_unmatched=None, pi_unmatched_in_span=None, span_s=None)
    if len(pairs) < MIN_PAIRS:
        return out
    pi_i = np.array([i for i, _ in pairs])
    es_i = np.array([j for _, j in pairs])
    a, b = _fit(e[es_i], p[pi_i])
    res = p[pi_i] - (a + b * e[es_i])
    lo, hi = a + b * e.min(), a + b * e.max()
    in_span = np.nonzero((p >= lo - MATCH_TOL_US) & (p <= hi + MATCH_TOL_US))[0]
    out.update(offset_us=a, rate=b, res_max_us=float(np.max(np.abs(res))),
               res_rms_us=float(np.sqrt(np.mean(res ** 2))), half_bracket_max_us=float(np.max(half[pi_i])),
               esp_unmatched=int(len(e) - len(pairs)),
               pi_unmatched_in_span=int(len(set(in_span.tolist()) - set(pi_i.tolist()))),
               span_s=float((e.max() - e.min()) / 1e6))
    out['delta_sync_us'] = out['res_max_us'] + out['half_bracket_max_us']
    out['_pairs'] = pairs
    return out


def verdict(s: dict) -> Tuple[int, str]:
    if s['n_pairs'] < MIN_PAIRS:
        return 3, 'thieu_bang_chung'
    if s['esp_unmatched'] or s['pi_unmatched_in_span']:
        return 1, ''
    return 0, ''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Đồng hồ ESP32 ↔ RPi5 từ chân đánh dấu (S06 T3)')
    ap.add_argument('--frames', required=True, help='frames.npz của run_hil.py --marker … --frame-log')
    ap.add_argument('--esp32-events', required=True, help='esp32_events.txt của s06_logger.py')
    ap.add_argument('--run', type=int, default=None, help='chỉ cạnh M có run = N; bỏ trống = mọi cạnh')
    ap.add_argument('--out', required=True, help='file JSON kết quả')
    a = ap.parse_args(argv)
    try:
        z = np.load(a.frames)
        p, half = pi_marks(z)
        raw = chk.read_raw(a.esp32_events)
        if not raw:
            raise ValueError(f'{a.esp32_events}: không có dòng nào')
        ev = chk.parse_events(raw)
        e, e_rx = esp_marks(ev, a.run)
        if len(e) and np.any(np.diff(e) <= 0):
            raise ValueError('giờ ESP32 của cạnh M không tăng (ESP32 khởi động lại giữa file?): chọn --run')
    except (OSError, ValueError, KeyError) as ex:
        print(f'lỗi đầu vào: {ex}', file=sys.stderr)
        return 2
    s = sync(p, half, e, e_rx)
    ma, ly = verdict(s)
    res = {k: v for k, v in s.items() if not k.startswith('_')}
    res.update(run=a.run, frames=a.frames, esp32_events=a.esp32_events, ma=ma, ly_do=ly or None)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"cạnh Pi {s['n_pi']}, cạnh ESP32 {s['n_esp']}, cặp {s['n_pairs']}")
    if s['offset_us'] is not None:
        print(f"t_pi_us = {s['offset_us']:.3f} + {s['rate']:.9f} · t_esp_us; phần dư lớn nhất {s['res_max_us']:.1f} µs, "
              f"delta_sync {s['delta_sync_us']:.1f} µs; ESP32 không cặp {s['esp_unmatched']}, "
              f"Pi không cặp trong khoảng {s['pi_unmatched_in_span']}")
        print(f"--clock-affine {s['offset_us']:.3f} {s['rate']:.9f} --delta-sync-us {s['delta_sync_us']:.1f}")
    print(f'#ket_qua,ma={ma}' + (f',ly_do={ly}' if ly else ''))
    return ma


if __name__ == '__main__':
    raise SystemExit(main())
