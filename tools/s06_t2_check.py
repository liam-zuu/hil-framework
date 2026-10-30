#!/usr/bin/env python3
"""
s06_t2_check.py — chấm T2 của spec S06 v3.4 (H2: mỗi bước plant hoàn tất trước hạn) từ CSV của
`run_hil.py --step-log` (cột `step, t_sched_us, t_start_us, t_done_us, t_end_us, miss`) và file
`<CSV>.json` cùng tên (bộ đếm của bộ đệm vòng).

Luật (spec T2: V1, W7, Z9, V10; mục 4 Z19):
  * Mất dữ liệu đo: tràn bộ đệm > 0, bản ghi mất/đọc dở > 0, hay `step` không liên tục → lượt
    THẤT BẠI (Z19: không thay lượt; W7 cũ ghi "không đủ điều kiện đo", Z19 viết sau và nói rõ
    bộ đệm đo tràn là thất bại).
  * Bỏ 10 s đầu (theo `t_sched` so với bước đầu tiên); phần còn lại là phần chấm, phải ≥ 600 s
    (`--min-scored-s`), không đủ thì lượt "không đủ thời lượng".
  * (a) lỡ hạn = `t_done > t_sched + 1 ms`; (b) `t_done − t_sched` lớn nhất, p99, p99,9;
    (c) trễ bắt đầu `t_start − t_sched` (chỉ báo); Z9: `t_end` của bước k > `t_sched` của bước k+1
    cũng là không đạt.
  * Đạt khi 0 lỡ hạn và 0 bước `t_end` muộn ở mọi lượt (và không mất dữ liệu).
  * V10: cận trên 95 % của tỉ lệ lỡ hạn theo quy tắc "ba" = 3 / số bước chấm (chỉ dấu, giả định
    các bước độc lập); báo số cụm lỡ hạn (các lần lỡ cách nhau ≤ `--cluster-gap-steps` bước,
    mặc định 100 = 100 ms [ĐỀ XUẤT CODE]) và khoảng cách giữa các cụm.
  * Kiểm thêm: `t_sched` tăng đúng 1 chu kỳ mỗi bước (lịch không bị đặt lại giữa lượt), cột `miss`
    khớp giá trị tính lại.

Sửa theo Codex Sol vòng 1/2 (Liam duyệt 30/10 "Duyệt, sửa hết"):
  * S4: `rows`, `first_step`, `steps_written` của `.json` phải khớp CSV (CSV bị cắt hay thiếu dòng là
    mất dữ liệu, thất bại); `.json` thiếu các trường đó cũng tính là không kiểm được = thất bại.
  * S13: `t_end` của bước cuối so với hạn kế = hạn của nó + 1 chu kỳ; độ phân giải đồng hồ của `.json`
    (`monotonic_resolution_s`) là bất định: lỡ chắc chắn khi vượt hạn quá độ phân giải, nằm trong độ phân
    giải quanh hạn → `inconclusive` (mã 3). Giới hạn còn lại (diễn giải (4)): `t_end` lấy trước lệnh ghi dấu
    hoàn tất (một lệnh ghi số nguyên) nên phần sau dấu đó không đo; `last_wake_ns` cập nhật trước khi
    `wait_next_tick()` trả về (`timing.py`), nằm trước `t_done`.
  * S14: cận quy tắc ba chỉ in khi tổng số lỡ hạn = 0; có lỡ thì in số và tỉ lệ quan sát.

Khớp spec 3.1 (Liam 30/10, "Duyệt, làm tiếp"):
  * C1-10: giá trị không hữu hạn (NaN, Inf) trong CSV hay ở `monotonic_resolution_s` là mất dữ liệu đo, thất bại
    (so sánh với NaN luôn sai nên trước đây lọt như "không lỡ hạn").
  * X8: `uname` trong `.json` của lượt được chép vào kết quả; thiếu thì ghi chú.

Usage:
  python tools/s06_t2_check.py <steps.csv> [<steps.csv> ...] [--out summary.json]
Mã thoát (spec 3.1, bảng mã kết quả): 0 mọi lượt đạt; 1 có lượt không đạt hoặc thất bại; 3 không có lượt không
đạt nhưng có lượt không đủ thời lượng (`khong_du_dieu_kien`) hoặc chưa kết luận (`chua_ket_luan`), in lý do;
2 lỗi đầu vào.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PERIOD_US = 1000.0
DISCARD_S = 10.0
MIN_SCORED_S = 600.0
CLUSTER_GAP_STEPS = 100


def load(path):
    with open(path, encoding='utf-8') as f:
        names = f.readline().strip().split(',')
    m = np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float64))
    if m.size == 0:
        raise ValueError(f'{path}: no rows')
    a = {n: m[:, i] for i, n in enumerate(names)}
    meta_path = path + '.json'
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    return a, meta


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def clusters(miss_steps, gap):
    out = []
    for s in miss_steps:
        if out and s - out[-1][1] <= gap:
            out[-1][1] = s
            out[-1][2] += 1
        else:
            out.append([s, s, 1])
    return out


def check_run(path, min_scored_s=MIN_SCORED_S, cluster_gap=CLUSTER_GAP_STEPS, period_us=PERIOD_US):
    a, meta = load(path)
    # C1-10: a row with NaN/Inf is lost data (every comparison with NaN is False, it would pass as "no miss")
    fin = np.all([np.isfinite(v) for v in a.values()], axis=0)
    if not fin.any():
        raise ValueError(f'{path}: no finite row')
    non_finite = int(np.sum(~fin))
    a = {k: v[fin] for k, v in a.items()}
    step = a['step'].astype(np.int64)
    ts, t0, td, te = (a[c].astype(float) for c in ('t_sched_us', 't_start_us', 't_done_us', 't_end_us'))
    res = dict(file=path, rows=int(len(step)), notes=[], non_finite_rows=non_finite)
    # data loss (Z19)
    gaps = int((step[-1] - step[0] + 1) - len(np.unique(step)))
    dup = int(len(step) - len(np.unique(step)))
    order_bad = int(np.sum(np.diff(step) <= 0))
    res.update(step_gaps=gaps, step_dups=dup, step_order_bad=order_bad)
    if meta is None:
        res['notes'].append('thiếu file .json của bộ đệm vòng: không kiểm được tràn')
        res.update(overflow=None, lost_or_torn=None)
    else:
        argv = meta.get('argv')
        res.update(overflow=meta.get('overflow'), lost_or_torn=meta.get('lost_or_torn'),
                   monotonic_resolution_s=meta.get('monotonic_resolution_s'), run_argv=argv, uname=meta.get('uname'),
                   frame_log=None if argv is None else any(x == '--frame-log' or x.startswith('--frame-log=')
                                                           for x in argv))
        if argv is None:
            res['notes'].append('file .json không có dòng lệnh (bản run_hil cũ): không biết lượt có --frame-log')
        if meta.get('uname') is None:
            res['notes'].append('file .json không có uname (X8): không biết kernel của lượt')
    # S4: the CSV must be the whole drained log of the ring: rows, first step and steps written of the
    # .json (rpi5/step_log.py StepLogReader.counters) must match the CSV, else rows were cut or lost
    meta_bad = []
    if meta is not None:
        for k, want in (('rows', len(step)), ('first_step', int(step[0])), ('steps_written', int(step[-1]) + 1)):
            if k not in meta:
                meta_bad.append(f'.json thiếu {k}')
            elif meta[k] != want:
                meta_bad.append(f'.json {k} = {meta[k]}, CSV cho {want}')
        r_s = meta.get('monotonic_resolution_s') or 0.0
        if not (np.isfinite(r_s) and r_s >= 0):
            meta_bad.append(f'.json monotonic_resolution_s = {r_s}')      # C1-10
            meta['monotonic_resolution_s'] = 0.0
    if non_finite:
        res['notes'].append(f'{non_finite} dòng có giá trị không hữu hạn (NaN/Inf): mất dữ liệu đo')
        res['notes'] += meta_bad
    res['meta_mismatch'] = meta_bad
    data_loss = bool(gaps or dup or order_bad or non_finite or (meta is None) or meta.get('overflow')
                     or meta.get('lost_or_torn') or meta_bad)
    # schedule continuity
    dts = np.diff(ts)
    res['t_sched_step_bad'] = int(np.sum(np.abs(dts - period_us) > 1e-3))
    # discard 10 s
    keep = ts >= ts[0] + DISCARD_S * 1e6
    res['steps_discarded'] = int(np.sum(~keep))
    sc = np.nonzero(keep)[0]
    n = len(sc)
    res['steps_scored'] = n
    res['scored_s'] = float((ts[sc[-1]] - ts[sc[0]]) / 1e6 + period_us / 1e6) if n else 0.0
    lat = td[sc] - ts[sc]
    st = t0[sc] - ts[sc]
    # S13: clock resolution of the timestamps (the .json) as the measurement uncertainty: definitely late
    # beyond it, not concluded within it (mục 4)
    res_us = float(meta.get('monotonic_resolution_s') or 0.0) * 1e6 if meta else 0.0
    miss = lat > period_us + res_us
    near = (np.abs(lat - period_us) <= res_us) & ~miss
    res['miss_col_mismatch'] = int(np.sum(a['miss'][sc].astype(int) != (lat > period_us).astype(int)))
    # S13: the deadline after the LAST step is its own deadline + one period (no next row to read it from)
    nxt_deadline = np.empty(n)
    nxt = sc + 1
    ok_next = nxt < len(ts)
    nxt_deadline[ok_next] = ts[nxt[ok_next]]
    nxt_deadline[~ok_next] = ts[sc[~ok_next]] + period_us
    late_end = te[sc] > nxt_deadline + res_us
    near_end = (np.abs(te[sc] - nxt_deadline) <= res_us) & ~late_end
    res.update(clock_resolution_us=res_us, near_deadline=int(near.sum() + near_end.sum()))
    res.update(misses=int(miss.sum()), t_end_late=int(late_end.sum()),
               done_max_us=float(lat.max()) if n else None, done_p99_us=pct(lat, 99), done_p999_us=pct(lat, 99.9),
               start_max_us=float(st.max()) if n else None, start_p99_us=pct(st, 99), start_p999_us=pct(st, 99.9),
               end_margin_min_us=float(np.min(nxt_deadline - te[sc])) if n else None)
    cl = clusters(step[sc][miss].tolist(), cluster_gap)
    res['miss_clusters'] = len(cl)
    res['miss_cluster_list'] = [dict(first_step=c[0], last_step=c[1], misses=c[2]) for c in cl[:50]]
    res['cluster_spacing_steps'] = [cl[i + 1][0] - cl[i][1] for i in range(len(cl) - 1)][:50]
    if data_loss:
        res['status'] = 'fail_data_loss'
    elif res['misses'] or res['t_end_late'] or res['t_sched_step_bad']:
        res['status'] = 'fail'
    elif res['scored_s'] < min_scored_s:
        res['status'] = 'too_short'
    elif res['near_deadline']:
        res['status'] = 'inconclusive'
    else:
        res['status'] = 'pass'
    if res['t_sched_step_bad']:
        res['notes'].append('t_sched không tăng đúng 1 chu kỳ: lịch bị đặt lại hoặc nhảy giữa lượt')
    if res['miss_col_mismatch']:
        res['notes'].append('cột miss khác giá trị tính lại từ thời điểm')
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm T2 (S06 v3.4)')
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--min-scored-s', type=float, default=MIN_SCORED_S)
    ap.add_argument('--cluster-gap-steps', type=int, default=CLUSTER_GAP_STEPS)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    try:
        runs = [check_run(p, a.min_scored_s, a.cluster_gap_steps) for p in a.csv]
    except (OSError, ValueError, KeyError) as ex:
        print(f'lỗi đầu vào: {ex}', file=sys.stderr)
        return 2
    n_sc = sum(r['steps_scored'] for r in runs if r['status'] in ('pass', 'fail', 'inconclusive'))
    n_miss = sum(r['misses'] for r in runs if r['status'] in ('pass', 'fail', 'inconclusive'))
    # S14: the rule of three is an upper bound only when NO miss was seen; with misses report the rate
    total = dict(runs=len(runs), steps_scored=n_sc, misses=n_miss,
                 miss_rate_observed=(n_miss / n_sc) if n_sc else None,
                 rule_of_three_upper_95=(3.0 / n_sc) if (n_sc and n_miss == 0) else None,
                 status={s: sum(1 for r in runs if r['status'] == s)
                         for s in ('pass', 'fail', 'fail_data_loss', 'too_short', 'inconclusive')})
    for r in runs:
        print(f"{r['file']}: {r['status']}  chấm {r['steps_scored']} bước ({r['scored_s']:.1f} s), "
              f"lỡ hạn {r['misses']} ({r['miss_clusters']} cụm), t_end muộn {r['t_end_late']}, "
              f"t_done−t_sched max {r['done_max_us']} p99 {r['done_p99_us']} p99.9 {r['done_p999_us']} µs, "
              f"trễ bắt đầu max {r['start_max_us']} µs, tràn {r['overflow']}, mất {r['lost_or_torn']}, "
              f"hở step {r['step_gaps']}" + ''.join(f"\n  ! {x}" for x in r['notes']))
    if total['rule_of_three_upper_95'] is not None:
        print(f"Tổng: {total['status']}; 0 lần lỡ hạn trên {n_sc} bước: cận trên 95 % tỉ lệ lỡ hạn (quy tắc ba, "
              f"chỉ dấu) {total['rule_of_three_upper_95']}")
    else:
        print(f"Tổng: {total['status']}; {n_miss} lần lỡ hạn trên {n_sc} bước (tỉ lệ quan sát "
              f"{total['miss_rate_observed']}); quy tắc ba không áp dụng khi đã có lần lỡ")
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(dict(total=total, runs=runs), f, indent=1, ensure_ascii=False)
    st = total['status']
    if st['fail'] or st['fail_data_loss']:
        print('#ket_qua,ma=1')
        return 1
    if st['too_short'] or st['inconclusive']:
        print('#ket_qua,ma=3,ly_do=' + ('khong_du_dieu_kien' if st['too_short'] else 'chua_ket_luan'))
        return 3
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
