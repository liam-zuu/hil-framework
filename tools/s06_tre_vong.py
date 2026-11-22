#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s06_tre_vong.py — NHÁP để duyệt: trễ vòng HIL cộng hai nhánh đo được từ kết quả chấm S06
T3 (state -> HIL -> DUT thấy mức mới qua encoder) và T6 (DUT ghi compare -> HIL/plant thấy
duty mới trong MISO). Hai phép đo ở HAI LƯỢT khác nhau: không có ghép cặp theo sự kiện, mọi
tổng tính trên phân bố mọi cặp (giả định độc lập, ghi trong ghi_chu / summary).

Mốc đo (theo các công cụ chấm và spec S06):
  * T3 (spec S06 T3 dòng 116): mốc đầu = cạnh LÊN chân GPIO đánh dấu ngay trước transfer của
    frame có ω mới — plant đã tính trạng thái của bước đó (rpi5/hil_node.py: marker_due
    91-100, nâng chân marker trong [t_pulse0; t_pulse1] tại 465-473, t_send_ns =
    time.monotonic_ns() ngay trước self._spi.transfer tại 474-475; mảng log
    t_send_ns/t_pulse0_ns/t_pulse1_ns khai báo 322-329, 'marker' 336-338). Mốc giữa (i) =
    cạnh D trên PF3 (H7 hẹn lại CCR của xung encoder kế), mốc cuối (ii) = cạnh encoder N kế
    tại chân ESP32. delay_us = mốc->D, quant_us = D->N (s06_t3_score.py COLUMNS).
  * T6 (spec S06 T5/T6 dòng 120): mốc = t_before của lần ghi compare (giờ ESP32 đưa về
    Pi); cuối = frame (t_send_ns) mà MISO rời mức cũ ±LIMIT (leave_ms) / ổn định trong ±LIMIT
    giữ 100 frame liền (follow_ms) — s06_t6_check.py check(), FOLLOW_COLS. CHÚ Ý:
    s06_t6_check.py còn LIMIT = 0.006 của spec v3.4; ngưỡng v3.5 là 0.0055 nên leave/follow
    trong T6_follow.csv đang theo ±0.006 — kiểm 'limit' trong T6_summary.json, khác 0.0055
    thì mã 3 (ghi trong summary.ghi_chu và danh sách 'thieu').

Lượng tử hoá (ghi vào docstring để đọc số):
  * plant 1 ms: T6 đo trên lưới frame (đến lượt frame kế = trễ thật của bàn, không phải sai
    số đo);
  * firmware DUT 10 ms: cạnh encoder -> lần đọc count ở tick kế chờ trong [0; 10) ms — KHÔNG
    có trong hai file này (T3/T6 đều qua s06_load, không qua bộ điều khiển), fwsim đã có tick
    10 ms riêng nên KHÔNG cộng thêm;
  * chu kỳ PWM 50 µs: thay đổi chân nằm trong [t_before; t_after + 50 µs] — t_mark_span_us;
    leave/follow phóng đại tối đa bằng span đó (max ghi trong summary).

Ngữ nghĩa fwsim (để đọc io_delay): ở io_delay_ms = 0 fwsim đọc count = floor góc bánh chính
xác tại tick và áp duty trong CÙNG bước plant (fwsim/fwsim.py 24-25, 336, 369-391), nên
io_delay_ms xấp xỉ trễ do HIL thêm vào = S1 'tre_vong_chet' (dead time; IIR 0.75/0.25 của
capture là lag, không phải trễ thuần) với S2 'tre_vong_tren' là biến thể cận trên
(delay + quant ở đầu T3, settle ở đầu T6). io_floor/io_ceil = floor(x + 1e-9) /
ceil(x - 1e-9) của median/p95/max — biên nguyên ms trên lưới plant 1 ms.

KHÔNG đo được từ hai file đầu vào (ghi vào 'thieu', KHÔNG tự đặt mốc): đường nội bộ DUT
(cạnh encoder -> đọc count ở tick -> tính pwm -> ghi compare — T3 dùng hồ sơ t3 của s06_load
với PWM 0, T6 dùng ghi compare theo lịch, không phải bộ điều khiển); đường IMU (plant ->
giả lập BNO085 -> báo cáo IMU của ESP32); đồng hồ ESP32->Pi mà T6 đã dùng (chỉ trong
T6_summary.json); bất định u của T3 (chỉ trong T3_summary.json); cách ghép cặp hai phần.

Đầu vào:
  --t6-follow  T6_follow.csv (s06_t6_check.py write_csv, header phải đúng FOLLOW_COLS)
  --t3-result  T3_result.csv (s06_t3_score.py write_rows, header phải đúng COLUMNS)
  --t6-summary T6_summary.json (tuỳ chọn; sync.kind, limit, follow, run_frames_bad,
               frame_log_faults, qseq_stream)
  --t3-summary T3_summary.json (tuỳ chọn; u_us, summary)
  --chot TP:CS thống kê A12 đã chốt (TP tre_vong_chet | tre_vong_tren; CS median | p95 | max)
  --out DIR

Ràng buộc CSV với summary của chính capture (C3-16; lệch -> mã 2):
  * T3_summary.json `summary`: steps, measured, expired, expired_ii, extra_pulse, no_data,
    unsynced phải bằng số đếm trên T3_result.csv (cùng định nghĩa s06_t3_score.score), và
    delay_max_us bằng delay_us lớn nhất của dòng do_duoc trong ±T3_MAX_TOL_US (CSV ghi 3 số
    lẻ); thiếu khối `summary` -> mã 2.
  * T6_summary.json `follow` (s06_t6_check.follow_summary): mỗi cặp (duty_from, duty_to) cùng
    n, not_settled, follow_ms_max (±1e-9) với T6_follow.csv; thiếu `follow` -> mã 2.
Đủ bằng chứng (C3-16) — mỗi thành phần và mỗi tổng có cột trang_thai 'du' hay
'thieu: <lý do>'; phần thiếu (bậc hết hạn, chuyển mức không rời / không ổn định) là dữ liệu bị
cắt: thống kê trên phần còn lại lệch thấp, nên KHÔNG đủ:
  * T3 (i): có T3_summary, u_us không None và u_us <= t3m.U_MAX_US (250 µs) + 1e-9; mọi bậc
    verdict_i 'do_duoc' (không het_han / khong_dat / no_data / chua_ghep); có mẫu.
    T3 (i)+(ii): thêm mọi bậc do_duoc có verdict_ii 'do_duoc'.
  * T6 leave / follow: có T6_summary, sync.kind 'manual', limit = 0.0055 (±1e-9),
    run_frames_bad = 0, frame_log_faults rỗng, qseq_stream ok và known; mọi chuyển mức không
    phẳng có leave_ms (leave) / settled (follow); có mẫu.
  * S1 = t3_i ⊕ t6_leave, S2 = t3_i_ii ⊕ t6_follow: đủ khi cả hai thành phần đủ. median và
    p95 của tổng lấy trên TÍCH DESCARTES của hai lượt đo (mọi cặp, giả định độc lập: hai lượt
    khác nhau, không ghép theo sự kiện); max = max_a + max_b đúng với mọi cách ghép.
  * bat_dinh_ms của tổng = (u_us + t_mark_span_us_max) / 1000: u của T3 (±) cộng phần
    phóng đại một phía của mốc ghi compare T6 (span, PWM 50 µs); không tính lại io_* theo nó.
Thống kê dùng cho điều kiện ">= 2 ms" chỉ là thống kê đã chốt (--chot) và chỉ khi tổng đó
'du': summary.chot.dau_vao_2ms = giá trị, không thì None. Chưa chốt -> None.

Đầu ra: DIR/tre_vong.csv (thanh_phan, n, min_ms, median_ms, p95_ms, max_ms, io_floor_median,
io_ceil_median, io_floor_p95, io_ceil_p95, io_floor_max, io_ceil_max, ghi_chu, trang_thai,
bat_dinh_ms) và DIR/tre_vong_summary.json.

Mã thoát (#ket_qua,ma=...): 0 đủ bằng chứng (có --chot: tổng đã chốt 'du'; không có: cả S1
và S2 'du'); 3 thieu_bang_chung khi không đủ (gồm: thiếu --t6-summary / sync kind không phải
'manual', thiếu --t3-summary / u_us None hay > 250 µs, t6 limit != 0.0055, bậc / chuyển mức bị
cắt, không có mẫu); 2 lỗi đầu vào (file/header/số, CSV không khớp summary, --chot sai).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s06_t6_check as t6m                          # noqa: E402
import s06_t3_score as t3m                          # noqa: E402
import s06_stream as stm                            # noqa: E402

LIMIT_V35 = 0.0055            # spec S06 v3.5 T6 (dòng 120); s06_t6_check.py còn LIMIT = 0.006 (v3.4)
FLAT_EPS = 1e-9               # |duty_to| cách 0 hoặc 1 dưới mức này = phẳng (A11: pin đứng yên)
IO_EPS = 1e-9                 # slack quanh số nguyên cho floor/ceil trên lưới 1 ms
T3_EXCLUDE = ('het_han', 'khong_dat', 'no_data', 'chua_ghep')   # verdict_i không 'do_duoc'

CSV_COLS = ['thanh_phan', 'n', 'min_ms', 'median_ms', 'p95_ms', 'max_ms', 'io_floor_median',
            'io_ceil_median', 'io_floor_p95', 'io_ceil_p95', 'io_floor_max', 'io_ceil_max',
            'ghi_chu', 'trang_thai', 'bat_dinh_ms']
T3_SUMM_KEYS = ('steps', 'measured', 'expired', 'expired_ii', 'extra_pulse', 'no_data',
                'unsynced')
T3_MAX_TOL_US = 1e-3          # delay_us trong CSV ghi '%.3f' (s06_t3_score.write_rows)
SUMS = ('tre_vong_chet', 'tre_vong_tren')
CHI_SO = ('median', 'p95', 'max')

# các mảnh không đo được từ hai file đầu vào (docstring); phần tuỳ dữ liệu nối thêm lúc chạy
THIEU_FIXED = [
    'duong noi bo DUT: canh encoder -> doc count o tick -> tinh pwm -> ghi compare '
    '(T3/T6 qua s06_load, khong qua bo dieu khien)',
    'duong IMU: plant -> gia lap BNO085 -> bao cao IMU cua ESP32',
    'ghep cap hai phan T3 va T6 theo su kien (hai luot khac nhau, gia dinh doc lap)',
]


def read_csv_checked(path, cols):
    """CSV có đúng header `cols` -> list dict; sai header/số cột -> ValueError (mã 2)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != list(cols):
        raise ValueError(f'{path}: header khong phai {list(cols)}')
    if any(len(r) != len(cols) for r in rows[1:]):
        raise ValueError(f'{path}: co dong sai so cot')
    return [dict(zip(cols, r)) for r in rows[1:]]


def _num(x, col, path):
    """'' -> None; số -> float; khác -> ValueError (mã 2)."""
    if x is None or str(x).strip() == '':
        return None
    try:
        return float(x)
    except ValueError:
        raise ValueError(f'{path}: cot {col} khong phai so: {x!r}') from None


def io_pair(x):
    """(floor(x + IO_EPS), ceil(x - IO_EPS)) — biên nguyên ms của một thống kê (docstring)."""
    return math.floor(x + IO_EPS), math.ceil(x - IO_EPS)


def is_flat(d):
    """duty_to phẳng: |d| trong FLAT_EPS của 0 hoặc 1 (docstring: trường hợp timeout A11)."""
    a = abs(d)
    return a <= FLAT_EPS or abs(a - 1.0) <= FLAT_EPS


def stats(x):
    """n, min, median, p95 ('higher'), max của một thành phần [ms]."""
    a = np.asarray(sorted(x), dtype=np.float64)
    if not len(a):
        return None
    return dict(n=len(a), min_ms=float(a[0]), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95, method='higher')), max_ms=float(a[-1]))


def row_of(name, st, note=''):
    """Một dòng tre_vong.csv từ kết quả stats(); rỗng -> n=0, thống kê trống."""
    r = dict(thanh_phan=name, ghi_chu=note)
    if st is None:
        r.update(n=0, min_ms='', median_ms='', p95_ms='', max_ms='', io_floor_median='',
                 io_ceil_median='', io_floor_p95='', io_ceil_p95='', io_floor_max='',
                 io_ceil_max='')
        return r
    r.update(st)
    for k in ('median_ms', 'p95_ms', 'max_ms'):
        lo, hi = io_pair(st[k])
        r[f'io_floor_{k[:-3]}'], r[f'io_ceil_{k[:-3]}'] = lo, hi
    return r


def sum_pairs(a, b):
    """Phân bố a+b trên MỌI cặp (giả định độc lập, hai luợt khác nhau): max = max_a + max_b
    (chặn trên đúng cho mọi ghép cặp), median/p95 trên phân bố cặp."""
    if not len(a) or not len(b):
        return None
    pairs = (np.asarray(a, dtype=np.float64)[:, None]
             + np.asarray(b, dtype=np.float64)[None, :]).ravel()
    return stats(pairs)


def _load_json(path, opt):
    if path is None:
        return None
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f'{opt}: {exc}') from None
    if not isinstance(d, dict):
        raise ValueError(f'{opt}: khong phai JSON object')
    return d


def bind_t3(rows3, t3s, path):
    """C3-16: T3_result.csv phải là của capture có T3_summary.json này (docstring); lệch ->
    ValueError (mã 2)."""
    summ = t3s.get('summary')
    if not isinstance(summ, dict) or any(k not in summ for k in T3_SUMM_KEYS):
        raise ValueError(f'--t3-summary: khong co khoi summary {list(T3_SUMM_KEYS)}')
    v1 = [r['verdict_i'] for r in rows3]
    got = dict(steps=len(rows3), measured=v1.count('do_duoc'), expired=v1.count('het_han'),
               expired_ii=sum(1 for r in rows3 if r['verdict_i'] == 'do_duoc'
                              and r['verdict_ii'] == 'het_han'),
               extra_pulse=v1.count('khong_dat'), no_data=v1.count('no_data'),
               unsynced=v1.count('chua_ghep'))
    bad = [f'{k} csv={got[k]} summary={summ[k]}' for k in T3_SUMM_KEYS if got[k] != summ[k]]
    d = [_num(r['delay_us'], 'delay_us', path) for r in rows3 if r['verdict_i'] == 'do_duoc']
    dmax, smax = (max(d) if d else None), summ.get('delay_max_us')
    if (dmax is None) != (smax is None) or (
            dmax is not None and abs(dmax - float(smax)) > T3_MAX_TOL_US + 1e-9):
        bad.append(f'delay_max_us csv={dmax} summary={smax}')
    if bad:
        raise ValueError(f'{path} khong khop T3_summary.json: ' + '; '.join(bad))


def bind_t6(rows6, t6s, path):
    """C3-16: T6_follow.csv phải khớp `follow` của T6_summary.json (docstring); lệch ->
    ValueError (mã 2)."""
    fs = t6s.get('follow')
    if not isinstance(fs, list):
        raise ValueError('--t6-summary: khong co danh sach follow')
    grp = {}
    for r in rows6:
        k = (float(r['duty_from']), float(r['duty_to']))
        g = grp.setdefault(k, [0, 0, None])
        g[0] += 1
        if r['settled'].strip() != 'True':
            g[1] += 1
        elif r['follow_ms'].strip():
            fw = float(r['follow_ms'])
            g[2] = fw if g[2] is None else max(g[2], fw)
    bad = []
    left = dict(grp)
    for e in fs:
        try:
            k = (float(e['duty_from']), float(e['duty_to']))
            want = (int(e['n']), int(e['not_settled']), e.get('follow_ms_max'))
        except (KeyError, TypeError, ValueError):
            raise ValueError('--t6-summary: dong follow sai dang') from None
        hit = next((g for g in left if abs(g[0] - k[0]) <= 1e-9 and abs(g[1] - k[1]) <= 1e-9),
                   None)
        if hit is None:
            bad.append(f'{k} co trong summary, khong co trong csv')
            continue
        n, ns, fmax = left.pop(hit)
        same_max = (fmax is None and want[2] is None) or (
            fmax is not None and want[2] is not None and abs(fmax - float(want[2])) <= 1e-9)
        if (n, ns) != want[:2] or not same_max:
            bad.append(f'{k}: csv n={n} not_settled={ns} max={fmax}, summary {want}')
    bad += [f'{k} co trong csv, khong co trong summary' for k in left]
    if bad:
        raise ValueError(f'{path} khong khop T6_summary.json: ' + '; '.join(bad[:3]))


def status(reasons):
    return 'du' if not reasons else 'thieu: ' + '; '.join(reasons)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Tre vong HIL = T3 + T6 (nhap, de duyet)')
    ap.add_argument('--t6-follow', required=True, help='T6_follow.csv (s06_t6_check)')
    ap.add_argument('--t3-result', required=True, help='T3_result.csv (s06_t3_score)')
    ap.add_argument('--t6-summary', default=None, help='T6_summary.json (sync.kind, limit)')
    ap.add_argument('--t3-summary', default=None, help='T3_summary.json (u_us)')
    ap.add_argument('--chot', default=None,
                    help='thong ke A12 da chot TP:CS, TP tre_vong_chet|tre_vong_tren, '
                         'CS median|p95|max')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        chot = None
        if a.chot is not None:
            tp, _, cs = a.chot.partition(':')
            if tp not in SUMS or cs not in CHI_SO:
                raise ValueError(f'--chot {a.chot!r}: can TP:CS, TP in {SUMS}, CS in {CHI_SO}')
            chot = (tp, cs)
        rows3 = read_csv_checked(a.t3_result, t3m.COLUMNS)
        rows6 = read_csv_checked(a.t6_follow, t6m.FOLLOW_COLS)
        t3s = _load_json(a.t3_summary, '--t3-summary')
        t6s = _load_json(a.t6_summary, '--t6-summary')
        if t3s is not None:
            bind_t3(rows3, t3s, a.t3_result)
        if t6s is not None:
            bind_t6(rows6, t6s, a.t6_follow)
        # T3: delay_us cua bac (i) do duoc; (i)+(ii): delay_us + quant_us khi ca hai do duoc
        t3_i, t3_ii = [], []
        exc3 = {v: 0 for v in T3_EXCLUDE}
        for r in rows3:
            d = _num(r['delay_us'], 'delay_us', a.t3_result)
            q = _num(r['quant_us'], 'quant_us', a.t3_result)
            v1, v2 = r['verdict_i'], r['verdict_ii']
            if v1 in exc3:
                exc3[v1] += 1
            if v1 == 'do_duoc':
                if d is None:
                    raise ValueError(f'{a.t3_result}: dong do_duoc khong co delay_us')
                t3_i.append(d / 1000.0)
                if v2 == 'do_duoc':
                    if q is None:
                        raise ValueError(f'{a.t3_result}: dong do_duoc/do_duoc khong co '
                                         'quant_us')
                    t3_ii.append((d + q) / 1000.0)
        # T6: leave/follow theo duty_to; phang tach rieng (timeout A11), khong cong vao tong
        t6_leave, t6_follow, t6_fleave, t6_ffollow = [], [], [], []
        not_settled, no_leave, span_max = 0, 0, None
        nf_not_settled, nf_no_leave = 0, 0          # chuyển mức không phẳng bị cắt (C3-16)
        for r in rows6:
            dto = _num(r['duty_to'], 'duty_to', a.t6_follow)
            lv = _num(r['leave_ms'], 'leave_ms', a.t6_follow)
            fw = _num(r['follow_ms'], 'follow_ms', a.t6_follow)
            sp = _num(r['t_mark_span_us'], 't_mark_span_us', a.t6_follow)
            if dto is None:
                raise ValueError(f'{a.t6_follow}: dong khong co duty_to')
            st = r['settled'].strip()
            if st not in ('True', 'False'):
                raise ValueError(f'{a.t6_follow}: settled khong phai True/False: {st!r}')
            settled = st == 'True'
            if not settled:
                not_settled += 1
            if lv is None:
                no_leave += 1
            if sp is not None:
                span_max = sp if span_max is None else max(span_max, sp)
            flat = is_flat(dto)
            if not flat:
                nf_not_settled += not settled
                nf_no_leave += lv is None
            if lv is not None:
                (t6_fleave if flat else t6_leave).append(lv)
            if settled and fw is not None:
                (t6_ffollow if flat else t6_follow).append(fw)
            elif settled:
                raise ValueError(f'{a.t6_follow}: dong settled=true khong co follow_ms')
    except (OSError, ValueError) as exc:
        print(f'loi dau vao: {exc}', file=sys.stderr)
        return stm.report_result(2)

    # bằng chứng đồng bộ / bất định từ hai summary (docstring)
    sync_kind = t6s.get('sync', {}).get('kind') if t6s else None
    limit = t6s.get('limit') if t6s else None
    u_us = t3s.get('u_us') if t3s else None
    thieu = list(THIEU_FIXED)
    if t6s is None:
        thieu.append('T6_summary.json (--t6-summary): dong ho ESP32->Pi T6 da dung')
    elif sync_kind != 'manual':
        thieu.append(f'dong ho T6 sync.kind={sync_kind!r} khong phai manual (chi tham khao)')
    if limit is None or not (isinstance(limit, (int, float))
                             and abs(float(limit) - LIMIT_V35) <= 1e-9):
        thieu.append(f't6 limit={limit!r} != {LIMIT_V35} (v3.5): leave/follow trong file '
                     'co the theo nguong khac')
    if t3s is None:
        thieu.append('T3_summary.json (--t3-summary): bat dinh u_us cua T3')
    elif u_us is None:
        thieu.append('u_us = None trong T3_summary.json')

    s1 = sum_pairs(t3_i, t6_leave)
    s2 = sum_pairs(t3_ii, t6_follow)
    if s1 is None:
        thieu.append('S1 tre_vong_chet khong co mau (t3_i hoac t6_leave rong)')

    # C3-16: đủ bằng chứng theo từng thành phần (docstring); phần bị cắt -> 'thieu'
    u_ok = isinstance(u_us, (int, float)) and math.isfinite(u_us)
    why3 = []
    if t3s is None:
        why3.append('khong co T3_summary.json')
    elif not u_ok:
        why3.append(f'u_us={u_us!r}')
    elif u_us > t3m.U_MAX_US + 1e-9:
        why3.append(f'u_us {u_us:g} > {t3m.U_MAX_US:g} us')
    n_not_i = sum(1 for r in rows3 if r['verdict_i'] != 'do_duoc')
    if n_not_i:
        why3.append(f'{n_not_i}/{len(rows3)} bac T3 khong do duoc (i)')
    n_not_ii = sum(1 for r in rows3 if r['verdict_i'] == 'do_duoc'
                   and r['verdict_ii'] != 'do_duoc')
    why3_ii = why3 + ([f'{n_not_ii} bac do_duoc khong do duoc (ii)'] if n_not_ii else [])
    why6 = []
    if t6s is None:
        why6.append('khong co T6_summary.json')
    else:
        if sync_kind != 'manual':
            why6.append(f'sync.kind={sync_kind!r}')
        if not (isinstance(limit, (int, float)) and abs(float(limit) - LIMIT_V35) <= 1e-9):
            why6.append(f'limit={limit!r} != {LIMIT_V35}')
        if t6s.get('run_frames_bad') != 0:
            why6.append(f"run_frames_bad={t6s.get('run_frames_bad')!r}")
        if t6s.get('frame_log_faults'):
            why6.append('frame_log_faults khong rong')
        qs = t6s.get('qseq_stream')
        if not (isinstance(qs, dict) and qs.get('ok') and qs.get('known')):
            why6.append('qseq_stream khong ok/known')
    why6_lv = why6 + ([f'{nf_no_leave} chuyen muc khong phang khong roi muc cu']
                      if nf_no_leave else [])
    why6_fw = why6 + ([f'{nf_not_settled} chuyen muc khong phang khong on dinh']
                      if nf_not_settled else [])

    def comp(vals, why):
        return why + ([] if vals else ['khong co mau'])

    st = dict(t3_i=comp(t3_i, why3), t3_i_ii=comp(t3_ii, why3_ii),
              t6_leave=comp(t6_leave, why6_lv), t6_follow=comp(t6_follow, why6_fw))
    st['tre_vong_chet'] = ([f't3_i: {x}' for x in st['t3_i']]
                           + [f't6_leave: {x}' for x in st['t6_leave']])
    st['tre_vong_tren'] = ([f't3_i_ii: {x}' for x in st['t3_i_ii']]
                           + [f't6_follow: {x}' for x in st['t6_follow']])
    bd = ((u_us + span_max) / 1000.0 if u_ok and span_max is not None else '')

    rows_out = [
        row_of('t3_i', stats(t3_i), 'T3 (i): delay_us/1000, verdict_i do_duoc'),
        row_of('t3_i_ii', stats(t3_ii), 'T3 (i)+(ii): (delay_us+quant_us)/1000, ca hai do_duoc'),
        row_of('t6_leave', stats(t6_leave), 'T6: leave_ms, duty_to khong phang'),
        row_of('t6_follow', stats(t6_follow), 'T6: follow_ms, settled, duty_to khong phang'),
        row_of('t6_flat_leave', stats(t6_fleave),
               'T6: leave_ms, duty_to phang (A11 timeout, khong cong vao tong)'),
        row_of('t6_flat_follow', stats(t6_ffollow),
               'T6: follow_ms settled, duty_to phang (A11 timeout, khong cong vao tong)'),
        row_of('tre_vong_chet', s1, 'S1 = t3_i + t6_leave; median, p95 tren tich Descartes '
               'hai luot (moi cap, gia dinh doc lap); max = max_a + max_b'),
        row_of('tre_vong_tren', s2, 'S2 = t3_i_ii + t6_follow; median, p95 tren tich '
               'Descartes hai luot (moi cap, gia dinh doc lap); max = max_a + max_b'),
    ]
    for r in rows_out:
        if r['thanh_phan'] in st:
            r['trang_thai'] = status(st[r['thanh_phan']])
        if r['thanh_phan'] in SUMS:
            r['bat_dinh_ms'] = bd
    by = {r['thanh_phan']: r for r in rows_out}
    chot_out = dict(thanh_phan=None, chi_so=None, gia_tri_ms=None, trang_thai=None,
                    dau_vao_2ms=None, ghi_chu='chua chot (--chot): khong co dau vao cho '
                    'dieu kien >= 2 ms')
    if chot is not None:
        r = by[chot[0]]
        v = r.get(f'{chot[1]}_ms')
        v = v if v != '' else None
        chot_out = dict(thanh_phan=chot[0], chi_so=chot[1], gia_tri_ms=v,
                        trang_thai=r['trang_thai'],
                        dau_vao_2ms=v if r['trang_thai'] == 'du' else None,
                        ghi_chu='' if r['trang_thai'] == 'du' else
                        'thong ke da chot chua du bang chung: khong dung cho dieu kien >= 2 ms')
    summ = dict(
        t3_excluded=exc3,
        t6_excluded=dict(not_settled=not_settled, no_leave=no_leave),
        t6_cat_khong_phang=dict(not_settled=nf_not_settled, no_leave=nf_no_leave),
        t_mark_span_us_max=span_max,
        t6=dict(sync_kind=sync_kind, limit=limit),
        t3=dict(u_us=u_us, u_max_us=t3m.U_MAX_US),
        trang_thai={k: status(v) for k, v in st.items()},
        bat_dinh_tong_ms=bd if bd != '' else None,
        phuong_phap_tong=('median va p95 cua tong lay tren tich Descartes cua hai luot do (moi '
                          'cap T3 x T6), gia dinh doc lap; max = max_a + max_b dung voi moi cach '
                          'ghep'),
        chot=chot_out,
        thieu=thieu,
        ghi_chu=(f's06_t6_check.py con LIMIT = {t6m.LIMIT} (spec v3.4); nguong v3.5 la '
                 f'{LIMIT_V35}: leave/follow trong T6_follow.csv theo nguong trong file, '
                 'khong tinh lai o day'))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'tre_vong.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore', restval='')
        w.writeheader()
        w.writerows(rows_out)
    with open(os.path.join(a.out, 'tre_vong_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    need = [chot[0]] if chot is not None else list(SUMS)
    # có mục 'thieu' ngoài các mảnh cố định, hay tổng cần dùng chưa đủ bằng chứng -> mã 3
    code = 3 if thieu != THIEU_FIXED or any(st[s] for s in need) else 0
    return stm.report_result(code, ['thieu_bang_chung'])


if __name__ == '__main__':
    raise SystemExit(main())
