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
  --t6-summary T6_summary.json (tuỳ chọn; sync.kind, limit)
  --t3-summary T3_summary.json (tuỳ chọn; u_us)
  --out DIR

Đầu ra: DIR/tre_vong.csv (thanh_phan, n, min_ms, median_ms, p95_ms, max_ms, io_floor_median,
io_ceil_median, io_floor_p95, io_ceil_p95, io_floor_max, io_ceil_max, ghi_chu) và
DIR/tre_vong_summary.json.

Mã thoát (#ket_qua,ma=...): 0 đủ bằng chứng; 3 thieu_bang_chung khi S1 không có mẫu, hoặc
thiếu --t6-summary / sync kind không phải 'manual', thiếu --t3-summary / u_us None, hay t6
limit != 0.0055; 2 lỗi đầu vào (file/header/số).
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
            'ghi_chu']

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


def main(argv=None):
    ap = argparse.ArgumentParser(description='Tre vong HIL = T3 + T6 (nhap, de duyet)')
    ap.add_argument('--t6-follow', required=True, help='T6_follow.csv (s06_t6_check)')
    ap.add_argument('--t3-result', required=True, help='T3_result.csv (s06_t3_score)')
    ap.add_argument('--t6-summary', default=None, help='T6_summary.json (sync.kind, limit)')
    ap.add_argument('--t3-summary', default=None, help='T3_summary.json (u_us)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        rows3 = read_csv_checked(a.t3_result, t3m.COLUMNS)
        rows6 = read_csv_checked(a.t6_follow, t6m.FOLLOW_COLS)
        t3s = _load_json(a.t3_summary, '--t3-summary')
        t6s = _load_json(a.t6_summary, '--t6-summary')
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

    rows_out = [
        row_of('t3_i', stats(t3_i), 'T3 (i): delay_us/1000, verdict_i do_duoc'),
        row_of('t3_i_ii', stats(t3_ii), 'T3 (i)+(ii): (delay_us+quant_us)/1000, ca hai do_duoc'),
        row_of('t6_leave', stats(t6_leave), 'T6: leave_ms, duty_to khong phang'),
        row_of('t6_follow', stats(t6_follow), 'T6: follow_ms, settled, duty_to khong phang'),
        row_of('t6_flat_leave', stats(t6_fleave),
               'T6: leave_ms, duty_to phang (A11 timeout, khong cong vao tong)'),
        row_of('t6_flat_follow', stats(t6_ffollow),
               'T6: follow_ms settled, duty_to phang (A11 timeout, khong cong vao tong)'),
        row_of('tre_vong_chet', s1, 'S1 = t3_i + t6_leave, moi cap (gia dinh doc lap)'),
        row_of('tre_vong_tren', s2, 'S2 = t3_i_ii + t6_follow, moi cap (gia dinh doc lap)'),
    ]
    summ = dict(
        t3_excluded=exc3,
        t6_excluded=dict(not_settled=not_settled, no_leave=no_leave),
        t_mark_span_us_max=span_max,
        t6=dict(sync_kind=sync_kind, limit=limit),
        t3=dict(u_us=u_us),
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
    code = 3 if thieu != THIEU_FIXED else 0     # có mục 'thieu' ngoài các mảnh cố định -> mã 3
    return stm.report_result(code, ['thieu_bang_chung'])


if __name__ == '__main__':
    raise SystemExit(main())
