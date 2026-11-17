#!/usr/bin/env python3
"""
s06_clock_sync.py — đồng hồ ESP32 ↔ RPi5 từ chân đánh dấu (S06 T3; thiết kế
03_CODE/docs/2026-11-15_Thiet_ke_S06_T3_danh_dau_v1.md mục 4 ý 3, Liam 15/11 "Duyệt, code luôn"; sửa sau Codex
vòng 1, Liam 17/11 "Sửa 12 + vòng 2": T3C-7, T3C-8, T3C-9, T3C-15; sau Codex vòng 2, Liam 17/11 "Sửa 5 + vòng 3":
T3C-8, T3C-R2-1).

Hai phía của cùng một cạnh lên:
  * RPi5: `run_hil.py --marker t3|10hz --frame-log frames.npz` nâng GPIO17 ngay trước `transfer` của frame có
    `marker` = 1; giờ cạnh = giữa `t_pulse0_ns` và `t_pulse1_ns` (kẹp lệnh nâng chân), nửa khoảng kẹp là bất
    định phía Pi;
  * ESP32 (`s06_load`): dòng `T3,qseq,t_us,M,38,lvl,gseq,run` (ISR GPIO38, `esp_timer`), mọi lượt và ngoài lượt.

Neo thô (T3C-8, T3C-15): cạnh 10 Hz cách đều 100 ms nên không ghép được theo khoảng cách, và giờ nhận `t_rx` chỉ
cho cận DƯỚI của trễ gửi. Neo bằng khoảng kẹp của lệnh START có `#ack,ok=1`:
    t_gửi(START, Pi, `capture.yaml` commands, lấy TRƯỚC sendto)  ≤  OFFSET + t_ack_esp  ≤  t_rx(#ack, Pi)
`t_ack_esp` = trường `t_us` của `#ack` (giờ ESP32 lúc cmd_task xử lý lệnh). Sửa sau Codex vòng 2 (Liam 17/11 "Sửa 5 +
vòng 3"): `#ack` chỉ ghép với lệnh START có CÙNG chuỗi (`s06_logger.py` thêm mã lần gửi `n=<k>` vào cuối mỗi lần gửi
START; `s06_load` đọc ba trường đầu bằng `sscanf`, bỏ qua phần thừa, `#ack` chép lại `cmd=` nguyên chuỗi); không có
lệnh cùng chuỗi, hay chuỗi đó gửi hơn một lần (log cũ gửi lại cùng chuỗi) thì không neo (mã 3). Độ rộng khoảng kẹp =
thời gian khứ hồi; rộng hơn ANCHOR_MAX_US thì không ghép (mã 3).
Độ lệch thô (T3C-R2-1): với các cạnh ESP32 gần `#ack` nhất (trong ANCHOR_SPAN_US, tối đa REFIT_EVERY cạnh), ứng viên là
cạnh Pi có (t_pi − t_esp) trong khoảng kẹp ± MATCH_TOL_US; khoảng đó rộng ≤ ANCHOR_MAX_US + 2·MATCH_TOL_US = 46 ms < 50 ms
(khoảng cách nhỏ nhất của cạnh Pi), nên mỗi cạnh ESP32 có tối đa một ứng viên. Độ lệch thô = trung vị các ứng viên;
ít hơn MIN_PAIRS ứng viên, hay ứng viên lệch trung vị quá MATCH_TOL_US, thì không ghép (mã 3). Từ độ lệch thô, ghép
tham lam theo thứ tự cạnh Pi, cửa sổ MATCH_TOL_US quanh giờ dự đoán, khớp lại affine mỗi REFIT_EVERY cặp (bám trôi
tần số thạch anh), rồi bình phương nhỏ nhất:
    t_pi_us = OFFSET + RATE · t_esp_us        (cùng quy ước `--clock-affine` của t3b_score.py)
Affine cuối phải đặt giờ `#ack` vào trong khoảng kẹp (± MATCH_TOL_US), không thì mã 3 (ghép không đáng tin).

Bất định (T3C-9): OFFSET chứa trễ ISR trung bình của cạnh M.
  * `delta_sync_isr_us` = phần dư lớn nhất + nửa khoảng kẹp Pi lớn nhất: cho sự kiện ESP32 cũng lấy giờ trong ISR
    GPIO (D, N, L: cùng đường ISR, trễ trung bình triệt tiêu, phần dao động nằm trong phần dư);
  * `delta_sync_us` = phần trên + trễ O → L lớn nhất của lượt hiệu chuẩn (`--loop-cal`): cho sự kiện ESP32 lấy giờ
    trong task (dòng E, `#run_start`), vì trễ ISR của M (≤ trễ O → L lớn nhất) không triệt tiêu. Không có
    `--loop-cal` dùng được thì `delta_sync_us` = None và mã 3.
Hiệu chuẩn trễ ISR (spec mục 4.1, Y15: tự nối ra → vào, N ≥ 1 000, khi `s06_load` đủ tải; Liam 17/11 "Bật cả trong
heavy"): cặp O → L của một lượt hồ sơ `heavy` (ESP32 đảo GPIO39 mỗi 100 ms trong vùng khóa, ISR GPIO37 qua dây nối
tắt). `loop_latency` đòi mọi O có đúng một L cùng mức trước O kế (trừ O cuối cùng của lượt), không có L thừa
(T3C-7); có cặp lỗi thì không dùng.

Dùng cho: t3b_score.py `--clock-affine OFFSET RATE --delta-sync-us …`; T4 (4), T6, cờ X17 của T1; khối tải nặng
(spec Z8): khoảng in H7 (`pwm_es`, ≥ 2 s) có giờ Pi (khớp CYCCNT của s06_frames_decode), mốc đổi vai bánh của
`s06_load` có giờ ESP32 trong task (`#run_start` timestamp_us + bội S06_HEAVY_SWAP_US): đổi sang giờ Pi bằng affine
này, bất định `delta_sync_us`. Các công cụ đó chưa gọi file này.

Mất dữ liệu: cạnh ESP32 đã chọn không ghép được, hay cạnh Pi trong khoảng giờ của các cạnh ESP32 đã chọn mà không
có cặp → mã 1. Ít hơn MIN_PAIRS cặp, không neo được, neo không khớp, thiếu `--loop-cal` → mã 3 `thieu_bang_chung`.

CLI:
  python tools/s06_clock_sync.py --frames frames.npz --esp32-events <lượt>/esp32_events.txt [--capture capture.yaml]
      [--run N] [--loop-cal <lượt heavy>/esp32_events.txt [--loop-cal-run M]] --out clock.json
    --capture: mặc định capture.yaml cùng thư mục với esp32_events.txt (s06_logger.py ghi).
Mã thoát: 0 đủ cặp, không mất; 1 mất cạnh; 3 thiếu bằng chứng; 2 lỗi đầu vào. Dòng cuối `#ket_qua,…`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s06_esp32_check as chk   # noqa: E402
import s06_stream as stm        # noqa: E402

MIN_PAIRS = 3                 # ít nhất 3 cặp mới khớp affine có phần dư
MATCH_TOL_US = 3000.0         # cửa sổ ghép quanh giờ dự đoán (≪ 50 ms, khoảng cách nhỏ nhất của cạnh)
REFIT_EVERY = 20
ANCHOR_MAX_US = 40_000.0      # độ rộng khoảng kẹp tối đa: + 2·MATCH_TOL_US vẫn < 50 ms (khoảng cách cạnh nhỏ nhất)
# Cạnh ESP32 dùng cho độ lệch thô cách #ack không quá chừng này: trôi 100 ppm × 20 s = 2 ms, dưới biên 7 ms
# (50 − 40 − 3) để cạnh Pi kề bên không lọt vào khoảng kẹp [ĐỀ XUẤT CODE_S06]
ANCHOR_SPAN_US = 20_000_000.0
LOOP_CAL_MIN = 1000           # spec mục 4.1: N = 1 000
CMD_RE = re.compile(r'^\s*-\s*\{t_us:\s*(\d+),\s*cmd:\s*"(.*)"\}\s*$')


# ── dữ liệu ──────────────────────────────────────────────────────────────────

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


def esp_marks(events, run: Optional[int] = None) -> np.ndarray:
    """Giờ ESP32 (µs) của các cạnh M theo gseq; `events` = kết quả s06_esp32_check.parse_events."""
    rows = sorted((r for r in events['T3'] if r['src'] == 'M' and (run is None or r['run'] == run)),
                  key=lambda r: r['gseq'])
    return np.array([r['t_us'] for r in rows], dtype=np.float64)


def read_commands(path: str) -> List[Tuple[int, str]]:
    """(giờ Pi µs lấy trước sendto, lệnh) của mục `commands:` trong capture.yaml của s06_logger.py."""
    out, inside = [], False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('commands:'):
                inside = True
                continue
            if inside:
                m = CMD_RE.match(line)
                if m:
                    out.append((int(m.group(1)), m.group(2).replace('\\"', '"').replace('\\\\', '\\')))
                elif not line.startswith(' '):
                    inside = False
    return out


def anchor(events, commands: Sequence[Tuple[int, str]], t_rx_ref: Optional[int] = None) -> Optional[dict]:
    """Khoảng kẹp của OFFSET từ một `#ack,ok=1` của START: lo ≤ OFFSET + t_ack ≤ hi. Chọn #ack START ok mới nhất
    có t_rx ≤ t_rx_ref + 2 s (t_rx_ref = t_rx của #run_start lượt chọn), không có t_rx_ref thì #ack đầu tiên.
    Lệnh gửi = lệnh START có đúng chuỗi `cmd=` của #ack (T3C-8: mã lần gửi `n=<k>` của s06_logger), gửi đúng một
    lần và không sau t_rx. None = không neo được."""
    acks = []
    for r in events['text']:
        if r['name'] != 'ack' or r['kv'].get('ok') != '1' or not r['kv'].get('cmd', '').startswith('START'):
            continue
        try:
            acks.append((int(r['t_rx']), int(r['kv']['t_us']), r['kv']['cmd']))
        except (KeyError, ValueError):
            continue
    if t_rx_ref is not None:
        acks = [a for a in acks if a[0] <= t_rx_ref + 2_000_000][-1:]
    else:
        acks = acks[:1]
    for t_rx, t_ack, text in acks:
        sent = [t for t, c in commands if c == text]
        if len(sent) != 1 or sent[0] > t_rx:     # không có, hay cùng chuỗi gửi hơn một lần: không gắn chắc
            continue
        lo, hi = sent[0] - t_ack, t_rx - t_ack
        return dict(t_ack_esp=t_ack, t_rx=t_rx, t_sent=sent[0], cmd=text, lo=float(lo), hi=float(hi),
                    width=float(hi - lo))
    return None


def loop_latency(t3_rows: Sequence[dict]) -> dict:
    """Trễ O → L (µs) từ dòng T3 của MỘT lượt (T3C-7): mỗi O phải có đúng một L cùng mức trước O kế; O cuối cùng
    của lượt được phép thiếu L (lượt tắt ngay sau O). Đếm lỗi: O thiếu L, L không có O, L sai mức."""
    ev = sorted((r for r in t3_rows if r['src'] in 'OL'), key=lambda r: r['gseq'])
    lat: List[float] = []
    o_no_l = l_no_o = l_bad = 0
    pend = None
    for r in ev:
        if r['src'] == 'O':
            if pend is not None:
                o_no_l += 1
            pend = r
        elif pend is None:
            l_no_o += 1
        elif r['lvl'] != pend['lvl']:
            l_bad += 1
            pend = None
        else:
            lat.append(r['t_us'] - pend['t_us'])
            pend = None
    out = dict(n=len(lat), o_without_l=o_no_l, l_without_o=l_no_o, l_wrong_level=l_bad,
               last_o_open=pend is not None, min_us=None, median_us=None, max_us=None, spread_us=None)
    if lat:
        a = np.array(lat, dtype=np.float64)
        out.update(min_us=float(a.min()), median_us=float(np.median(a)), max_us=float(a.max()),
                   spread_us=float(a.max() - a.min()))
    out['ok'] = bool(lat) and not (o_no_l or l_no_o or l_bad)
    return out


def gseq_complete(rows: Sequence[dict], marks: dict) -> dict:
    """T3C-11: mọi gseq trong (t3_g0; t3_g1] của #run_stats phải có đúng một dòng T3 của lượt, và không dòng T3
    nào của lượt nằm ngoài khoảng đó."""
    kv = marks.get('run_stats', {}).get('kv', {})
    try:
        g0, g1 = int(kv['t3_g0']), int(kv['t3_g1'])
    except (KeyError, ValueError):
        return dict(ok=False, why='#run_stats không có t3_g0/t3_g1: không chứng minh được luồng T3 đủ',
                    expected=None, missing=None, dups=None, outside=None)
    got = [r['gseq'] for r in rows]
    inr = [g for g in got if g0 < g <= g1]
    missing = (g1 - g0) - len(set(inr))
    dups = len(inr) - len(set(inr))
    outside = len(got) - len(inr)
    ok = missing == 0 and dups == 0 and outside == 0
    return dict(ok=ok, expected=g1 - g0, missing=missing, dups=dups, outside=outside,
                why='' if ok else f'luồng T3 của lượt: thiếu {missing}, trùng {dups}, ngoài khoảng {outside} '
                                  f'(gseq ({g0}; {g1}])')


def loop_cal(path: str, run: Optional[int] = None) -> dict:
    """Hiệu chuẩn trễ ISR từ lượt hồ sơ heavy (spec mục 4.1): loop_latency của lượt đó + điều kiện N ≥ 1 000,
    hồ sơ heavy, luồng gseq của lượt đủ. `ok` = dùng được. RunProblem / ValueError khi không chọn được lượt."""
    marks = stm.run_marks(path, run)
    stm.require_run(marks)
    kv = marks['run_start']['kv']
    rid = int(kv['run'])
    ev = chk.parse_events(chk.read_raw(path))
    rows = [r for r in ev['T3'] if r['run'] == rid]
    res = loop_latency(rows)
    res.update(path=path, run=rid, profile=kv.get('profile'), gseq=gseq_complete(rows, marks))
    why = []
    if kv.get('profile') != 'heavy':
        why.append(f"hồ sơ {kv.get('profile')!r}, cần heavy (đủ tải)")
    if res['n'] < LOOP_CAL_MIN:
        why.append(f"{res['n']} cặp O → L < {LOOP_CAL_MIN}")
    if not res['ok']:
        why.append('cặp O → L lỗi (O thiếu L, L thừa hay sai mức)')
    if not res['gseq']['ok']:
        why.append(res['gseq']['why'])
    res['ok'], res['why'] = not why, why
    return res


# ── ghép ─────────────────────────────────────────────────────────────────────

def _fit(e: Sequence[float], p: Sequence[float]) -> Tuple[float, float]:
    e = np.asarray(e, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(e) < 2:
        return float(p[0] - e[0]), 1.0
    e0 = e.mean()
    b, a0 = np.polyfit(e - e0, p, 1)
    return float(a0 - b * e0), float(b)


def match(p: np.ndarray, e: np.ndarray, off0: float, tol_us: float = MATCH_TOL_US) -> List[Tuple[int, int]]:
    """Cặp (chỉ số Pi, chỉ số ESP32), p và e tăng dần, từ độ lệch thô off0 (t_pi ≈ t_esp + off0): ghép tham lam
    theo thứ tự Pi, mỗi cạnh ESP32 dùng một lần, khớp lại affine sau mỗi REFIT_EVERY cặp."""
    a, b = off0, 1.0
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


def coarse_offset(p: np.ndarray, e: np.ndarray, anc: dict, tol_us: float = MATCH_TOL_US) -> dict:
    """T3C-R2-1: độ lệch thô t_pi − t_esp từ khoảng kẹp neo. Với tối đa REFIT_EVERY cạnh ESP32 gần #ack nhất (cách
    ≤ ANCHOR_SPAN_US), ứng viên = cạnh Pi có (t_pi − t_esp) ∈ [lo − tol; hi + tol] (khoảng < 50 ms: tối đa một ứng
    viên mỗi cạnh). off = trung vị ứng viên; None khi < MIN_PAIRS ứng viên hay ứng viên lệch trung vị > tol."""
    near = np.argsort(np.abs(e - anc['t_ack_esp']), kind='stable')
    near = [j for j in near if abs(e[j] - anc['t_ack_esp']) <= ANCHOR_SPAN_US][:REFIT_EVERY]
    d, multi = [], 0
    for j in near:
        c = p[(p - e[j] >= anc['lo'] - tol_us) & (p - e[j] <= anc['hi'] + tol_us)]
        if len(c) == 1:
            d.append(float(c[0] - e[j]))
        elif len(c) > 1:
            multi += 1
    out = dict(offset=None, n_cand=len(d), n_near=len(near), n_multi=multi, spread_us=None)
    if len(d) < MIN_PAIRS or multi:
        return out
    med = float(np.median(d))
    out['spread_us'] = float(max(abs(x - med) for x in d))
    if out['spread_us'] <= tol_us:
        out['offset'] = med
    return out


def sync(p: np.ndarray, half: np.ndarray, e: np.ndarray, anc: Optional[dict],
         loop_max_us: Optional[float] = None) -> dict:
    """Khớp affine t_pi = offset + rate·t_esp trên các cặp cạnh; p, e tăng dần; anc = anchor(); loop_max_us = trễ
    O → L lớn nhất của lượt hiệu chuẩn (None: delta_sync_us không tính được)."""
    out = dict(n_pi=int(len(p)), n_esp=int(len(e)), n_pairs=0, anchor=anc, anchor_ok=False, offset_us=None,
               rate=None, res_max_us=None, res_rms_us=None, half_bracket_max_us=None, delta_sync_isr_us=None,
               delta_sync_us=None, loop_max_us=loop_max_us, esp_unmatched=None, pi_unmatched_in_span=None,
               span_s=None, coarse=None, problem='')
    if anc is None:
        out['problem'] = ('không neo được: thiếu #ack START ok=1, hay capture.yaml không có lệnh START cùng chuỗi '
                          'cmd= (hoặc chuỗi đó gửi hơn một lần)')
        return out
    if anc['width'] > ANCHOR_MAX_US:
        out['problem'] = f"khoảng kẹp neo rộng {anc['width'] / 1000:.1f} ms > {ANCHOR_MAX_US / 1000:.0f} ms"
        return out
    if len(p) == 0 or len(e) == 0:
        out['problem'] = 'không có cạnh'
        return out
    co = coarse_offset(p, e, anc)
    out['coarse'] = co
    if co['offset'] is None:
        out['problem'] = (f"không có độ lệch thô trong khoảng kẹp: {co['n_cand']} ứng viên / {co['n_near']} cạnh ESP32 "
                          f"gần #ack, {co['n_multi']} cạnh nhiều ứng viên, lệch trung vị {co['spread_us']} µs")
        return out
    pairs = match(p, e, co['offset'])
    out['n_pairs'] = len(pairs)
    if len(pairs) < MIN_PAIRS:
        out['problem'] = f'{len(pairs)} cặp < {MIN_PAIRS}'
        return out
    pi_i = np.array([i for i, _ in pairs])
    es_i = np.array([j for _, j in pairs])
    a, b = _fit(e[es_i], p[pi_i])
    res = p[pi_i] - (a + b * e[es_i])
    t_ack_pi = a + b * anc['t_ack_esp']
    anc_ok = (anc['t_sent'] - MATCH_TOL_US) <= t_ack_pi <= (anc['t_rx'] + MATCH_TOL_US)
    lo, hi = a + b * e.min(), a + b * e.max()
    in_span = np.nonzero((p >= lo - MATCH_TOL_US) & (p <= hi + MATCH_TOL_US))[0]
    out.update(anchor_ok=bool(anc_ok), offset_us=a, rate=b, res_max_us=float(np.max(np.abs(res))),
               res_rms_us=float(np.sqrt(np.mean(res ** 2))), half_bracket_max_us=float(np.max(half[pi_i])),
               esp_unmatched=int(len(e) - len(pairs)),
               pi_unmatched_in_span=int(len(set(in_span.tolist()) - set(pi_i.tolist()))),
               span_s=float((e.max() - e.min()) / 1e6), t_ack_pi_us=float(t_ack_pi))
    out['delta_sync_isr_us'] = out['res_max_us'] + out['half_bracket_max_us']
    if loop_max_us is not None:
        out['delta_sync_us'] = out['delta_sync_isr_us'] + float(loop_max_us)
    if not anc_ok:
        out['problem'] = 'affine đặt giờ #ack START ngoài khoảng kẹp: ghép không đáng tin'
    out['_pairs'] = pairs
    return out


def verdict(s: dict) -> Tuple[int, str]:
    if s['problem'] or s['n_pairs'] < MIN_PAIRS:
        return 3, 'thieu_bang_chung'
    if s['esp_unmatched'] or s['pi_unmatched_in_span']:
        return 1, ''
    if s['delta_sync_us'] is None:
        return 3, 'thieu_bang_chung'
    return 0, ''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Đồng hồ ESP32 ↔ RPi5 từ chân đánh dấu (S06 T3)')
    ap.add_argument('--frames', required=True, help='frames.npz của run_hil.py --marker … --frame-log')
    ap.add_argument('--esp32-events', required=True, help='esp32_events.txt của s06_logger.py')
    ap.add_argument('--capture', default=None, help='capture.yaml của s06_logger.py (mặc định cùng thư mục)')
    ap.add_argument('--run', type=int, default=None, help='chỉ cạnh M có run = N; bỏ trống = mọi cạnh')
    ap.add_argument('--loop-cal', default=None, help='esp32_events.txt của lượt heavy có cặp O → L (spec 4.1)')
    ap.add_argument('--loop-cal-run', type=int, default=None)
    ap.add_argument('--out', required=True, help='file JSON kết quả')
    a = ap.parse_args(argv)
    cal = None
    try:
        z = np.load(a.frames)
        p, half = pi_marks(z)
        raw = chk.read_raw(a.esp32_events)
        if not raw:
            raise ValueError(f'{a.esp32_events}: không có dòng nào')
        ev = chk.parse_events(raw)
        e = esp_marks(ev, a.run)
        if len(e) and np.any(np.diff(e) <= 0):
            raise ValueError('giờ ESP32 của cạnh M không tăng (ESP32 khởi động lại giữa file?): chọn --run')
        cap = a.capture or os.path.join(os.path.dirname(os.path.abspath(a.esp32_events)), 'capture.yaml')
        cmds = read_commands(cap) if os.path.exists(cap) else []
        t_ref = None
        if a.run is not None:
            mk = stm.run_marks(a.esp32_events, a.run)
            stm.require_run(mk)
            t_ref = int(mk['run_start']['t_rx'])
        anc = anchor(ev, cmds, t_ref)
        if a.loop_cal:
            cal = loop_cal(a.loop_cal, a.loop_cal_run)
    except stm.RunProblem as ex:
        return stm.report_run_problem(ex)
    except (OSError, ValueError, KeyError) as ex:
        print(f'lỗi đầu vào: {ex}', file=sys.stderr)
        return 2
    s = sync(p, half, e, anc, cal['max_us'] if cal and cal['ok'] else None)
    ma, ly = verdict(s)
    res = {k: v for k, v in s.items() if not k.startswith('_')}
    res.update(run=a.run, frames=a.frames, esp32_events=a.esp32_events, loop_cal=cal, ma=ma, ly_do=ly or None)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"cạnh Pi {s['n_pi']}, cạnh ESP32 {s['n_esp']}, cặp {s['n_pairs']}" +
          (f"; {s['problem']}" if s['problem'] else ''))
    if cal is not None and not cal['ok']:
        print('hiệu chuẩn trễ ISR không dùng được: ' + '; '.join(cal['why']))
    if s['offset_us'] is not None:
        print(f"t_pi_us = {s['offset_us']:.3f} + {s['rate']:.9f} · t_esp_us; phần dư lớn nhất {s['res_max_us']:.1f} µs, "
              f"delta_sync_isr {s['delta_sync_isr_us']:.1f} µs, delta_sync {s['delta_sync_us']} µs; "
              f"ESP32 không cặp {s['esp_unmatched']}, Pi không cặp trong khoảng {s['pi_unmatched_in_span']}")
        if s['delta_sync_us'] is not None:
            print(f"--clock-affine {s['offset_us']:.3f} {s['rate']:.9f} --delta-sync-us {s['delta_sync_us']:.1f}")
    print(f'#ket_qua,ma={ma}' + (f',ly_do={ly}' if ly else ''))
    return ma


if __name__ == '__main__':
    raise SystemExit(main())
