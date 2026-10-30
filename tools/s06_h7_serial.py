#!/usr/bin/env python3
"""
s06_h7_serial.py — đọc log serial của H7 (ST-Link VCP, USART3, 115200; lưu bằng PuTTY hoặc chương trình
terminal bất kỳ) cho S06 v3.4: các dòng T1 (Link, Frame, Link period) và bảng T10 (tải H7, R22).

Định dạng in (hil-framework `feat/s06-h7` 598a451, `main.c`, `s06_load.c`; đầu dòng có thể có dấu thời
gian của chương trình terminal, bỏ qua):
  SPI link up at t=<ms> ms
  SPI link lost (no frame for <ms> ms): ...
  Link period: <frames> frames, H7 <ms> ms from first to last frame (Pi: <ms> ms at 1 kHz)
  Link: <up|down> lost=<n>  SPI frames=<n>/s err=<n> rearm=<n> last_err=0x<hex>          mỗi 0,5 s
  Frame: unread=<n>  new seen=<n> taken=<n> repl=<n> drop=<n>  applied=<n> skipped=<n> age_max=<n>us
  S06log: pushed=<n> dropped=<n>  print_drop=<n>
  T10: win_ms=<ms> isr_us=<µs> idle_pass=<n> idle_us_est=<µs> cal=<n>/<cyc> ovh_cyc=<cyc>       mỗi 10 s
  T10 <tên> n=<n> excl_us=<µs> max_us=<µs>.<2 số>                     một dòng mỗi handler có n > 0
Mọi bộ đếm (trừ T10) cộng dồn từ lúc khởi động; T10 đặt lại sau mỗi lần in.

T10 (spec S06 dòng T10, Y11, W10): chỉ báo số, không có ngưỡng.
  * Khối dùng: khối T10 in sau "SPI link up" đầu tiên và trước "SPI link lost" cuối của lượt (log serial
    là log của một lượt); bỏ khối đầu sau link up (cửa sổ 10 s của nó có phần trước khi Pi chạy).
    `--all-blocks` để lấy mọi khối.
  * Mỗi khối: % ISR = isr_us / (win_ms·1000) (thời gian độc quyền, H7 đã trừ ngắt lồng); % rảnh =
    idle_us_est / (win_ms·1000), idle_us_est = idle_pass × (cal_cyc / cal_n) / f_danh_định (H7 tính);
    phần còn lại = 100 % − ISR − rảnh (vòng chính + vào/ra ngắt + sai số quy đổi; không gọi là "vòng
    chính"). Bất định tương đối của phần rảnh do hiệu chuẩn: 1/cal_n + 1/idle_pass (đếm nguyên, ±1
    vòng). Chi phí đo = Σn × ovh_cyc / f_danh_định (cặp DWT, mục 4.1).
  * Đơn vị thời gian: H7 chia số chu kỳ cho SystemCoreClock danh định (480 MHz), win_ms đếm tick HAL của
    cùng đồng hồ, nên mọi tỉ lệ % đúng dù đồng hồ H7 nhanh 0,64 %. µs của H7 là "µs danh định"; µs thật
    = µs H7 × f_danh_định / f_thật (f_thật ≈ 483,1 MHz, Gia_tri_spec_S06 mục 1, số suy từ đo 27/10):
    cột `max_us_real`.
  * Tổng gộp (cận trên) không có: H7 chỉ in thời gian độc quyền; cột để trống. Vào/ra ngắt: chưa đo.
  * Luồng đủ (§4.3 điều 6): print_drop không tăng trong khoảng dùng, win_ms ∈ [9 900; 10 100], và (S24,
    Codex Sol vòng 1/2, Liam duyệt 30/10) không mất dòng IRQ (Σ excl_us ≤ isr_us ≤ Σ excl_us + số dòng)
    và không mất khối (giờ PC đầu dòng: hai header cách 10 s ± 0,5 s, số khối khớp độ dài đoạn; log
    không có giờ PC thì không kiểm được, ghi là thiếu). Thiếu → `complete: false`, mã 3.

Đơn vị: H7 in µs (đã chia f_cpu), nên cột CSV là µs thay cho "cycles" của danh sách cột spec
(irq, count, total_cycles_exclusive, total_cycles_gross, max_cycles, idle_cycles) [diễn giải CODE].

Usage:
  python tools/s06_h7_serial.py <h7_serial.log> --out <thư mục> [--f-real-mhz 483.1] [--all-blocks]
Mã thoát: 0 đủ dữ liệu; 3 luồng thiếu hoặc không có khối T10 nào; 2 lỗi đầu vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional

F_NOMINAL_MHZ = 480.0             # SystemCoreClock danh định (HSE 8 MHz, PLLN 120, PLLP 2): H7 chia cho số này
F_REAL_MHZ = 483.1                # xung lõi thật suy từ đo 27/10 (H7 nhanh 0,64 %; Gia_tri_spec_S06 mục 1)
WIN_MS_OK = (9_900, 10_100)
T10_GAP_TOL_S = 0.5               # S24 [ĐỀ XUẤT CODE]: hai header T10 liên tiếp cách 10 s ± 0,5 s theo giờ PC

_RE = {
    'link_up': re.compile(r'SPI link up at t=(\d+) ms'),
    'link_lost': re.compile(r'SPI link lost \(no frame for (\d+) ms\)'),
    'link_period': re.compile(r'Link period: (\d+) frames, H7 (\d+) ms from first to last frame '
                              r'\(Pi: (\d+) ms at 1 kHz\)'),
    'link': re.compile(r'Link: (up|down) lost=(\d+)\s+SPI frames=(\d+)/s err=(\d+) rearm=(\d+) '
                       r'last_err=0x([0-9a-fA-F]+)'),
    'frame': re.compile(r'Frame: unread=(\d+)\s+new seen=(\d+) taken=(\d+) repl=(\d+) drop=(\d+)'
                        r'(?:\s+applied=(\d+) skipped=(\d+) age_max=(\d+)us)?'),
    's06log': re.compile(r'S06log: pushed=(\d+) dropped=(\d+)\s+print_drop=(\d+)'),
    't10': re.compile(r'T10: win_ms=(\d+) isr_us=(\d+) idle_pass=(\d+) idle_us_est=(\d+) cal=(\d+)/(\d+) '
                      r'ovh_cyc=(\d+)'),
    't10_irq': re.compile(r'T10 (\S+)\s+n=(\d+) excl_us=(\d+) max_us=(\d+)\.(\d\d)'),
}
_FIELDS = {
    'link_up': ('t_ms',), 'link_lost': ('timeout_ms',), 'link_period': ('frames', 'h7_ms', 'pi_ms'),
    'link': ('state', 'lost', 'fps', 'err', 'rearm', 'last_err'),
    'frame': ('unread', 'new_seen', 'taken', 'repl', 'drop', 'applied', 'skipped', 'age_max_us'),
    's06log': ('pushed', 'dropped', 'print_drop'),
    't10': ('win_ms', 'isr_us', 'idle_pass', 'idle_us_est', 'cal_n', 'cal_cyc', 'ovh_cyc'),
}


_PC_TIME = re.compile(r'^(\d+)\t')


def parse_lines(lines) -> List[dict]:
    """Các dòng nhận ra được, theo thứ tự, mỗi dòng một dict có 'kind', 'line' (số dòng, từ 1) và
    't_pc_us' (giờ PC đầu dòng của tools/h7_serial_capture.py; None nếu log không có, ví dụ PuTTY)."""
    out = []
    for no, raw in enumerate(lines, 1):
        mt = _PC_TIME.match(raw)
        t_pc = int(mt.group(1)) if mt else None
        for kind, rx in _RE.items():
            m = rx.search(raw)
            if not m:
                continue
            if kind == 't10_irq':
                g = m.groups()
                out.append(dict(kind=kind, line=no, t_pc_us=t_pc, irq=g[0], n=int(g[1]), excl_us=int(g[2]),
                                max_us=int(g[3]) + int(g[4]) / 100.0))
            else:
                vals = []
                for name, v in zip(_FIELDS[kind], m.groups()):
                    if v is None:
                        vals.append(None)
                    elif name == 'state':
                        vals.append(v)
                    elif name == 'last_err':
                        vals.append(int(v, 16))
                    else:
                        vals.append(int(v))
                out.append(dict(kind=kind, line=no, t_pc_us=t_pc, **dict(zip(_FIELDS[kind], vals))))
            break
    return out


def read_log(path: str) -> List[dict]:
    with open(path, encoding='utf-8', errors='replace') as f:
        return parse_lines(f.read().splitlines())


def run_span(ev: List[dict]):
    """(chỉ số đầu, chỉ số cuối) của lượt: từ 'SPI link up' ĐẦU TIÊN tới 'SPI link lost' sau lần 'link up'
    cuối (hoặc hết log). Log serial là log của MỘT lượt (hướng dẫn đo); link mất rồi lên lại giữa đoạn
    này là gián đoạn của lượt, đếm ở `link_interruptions` (S25: không lấy riêng đoạn link cuối làm
    đại diện cả lượt)."""
    ups = [i for i, e in enumerate(ev) if e['kind'] == 'link_up']
    if not ups:
        return None
    lost = [i for i, e in enumerate(ev) if e['kind'] == 'link_lost' and i > ups[-1]]
    return ups[0], (lost[0] if lost else len(ev))


def link_interruptions(ev: List[dict]) -> int:
    """Số lần link lên lại sau lần đầu (mỗi lần là một lần mất link giữa lượt)."""
    return max(0, sum(1 for e in ev if e['kind'] == 'link_up') - 1)


def t10_blocks(ev: List[dict], all_blocks: bool = False):
    span = run_span(ev)
    a, b = (0, len(ev)) if (all_blocks or span is None) else span
    blocks, cur = [], None
    for e in ev[a:b]:
        if e['kind'] == 't10':
            cur = dict(head=e, irqs=[])
            blocks.append(cur)
        elif e['kind'] == 't10_irq' and cur is not None:
            cur['irqs'].append(e)
    if not all_blocks and span is not None and blocks:
        blocks = blocks[1:]                      # cửa sổ đầu có phần trước link up
    return blocks, span


def t10_rows(blocks, f_real_mhz: float = F_REAL_MHZ):
    win_rows, irq_rows = [], []
    for k, bl in enumerate(blocks):
        h = bl['head']
        win_us = h['win_ms'] * 1000.0
        n_calls = sum(i['n'] for i in bl['irqs'])
        idle_pct = 100.0 * h['idle_us_est'] / win_us if h['cal_n'] else None
        isr_pct = 100.0 * h['isr_us'] / win_us
        win_rows.append(dict(window=k, line=h['line'], win_ms=h['win_ms'], isr_us=h['isr_us'],
                             isr_pct=isr_pct, idle_pass=h['idle_pass'], idle_us=h['idle_us_est'],
                             idle_pct=idle_pct,
                             rest_pct=None if idle_pct is None else 100.0 - isr_pct - idle_pct,
                             idle_rel_unc=(1.0 / h['cal_n'] + 1.0 / h['idle_pass'])
                             if h['cal_n'] and h['idle_pass'] else None,
                             cal_n=h['cal_n'], cal_cyc=h['cal_cyc'], ovh_cyc=h['ovh_cyc'],
                             meas_cost_us=n_calls * h['ovh_cyc'] / F_NOMINAL_MHZ))
        for i in bl['irqs']:
            irq_rows.append(dict(window=k, irq=i['irq'], count=i['n'], excl_us=i['excl_us'],
                                 gross_us='', max_us=i['max_us'],
                                 max_us_real=i['max_us'] * F_NOMINAL_MHZ / f_real_mhz,
                                 pct=100.0 * i['excl_us'] / win_us))
    return win_rows, irq_rows


def t10_summary(blocks, win_rows, irq_rows, ev, span) -> dict:
    tot_ms = sum(r['win_ms'] for r in win_rows)
    per: Dict[str, dict] = {}
    for r in irq_rows:
        p = per.setdefault(r['irq'], dict(irq=r['irq'], count=0, excl_us=0, max_us=0.0))
        p['count'] += r['count']
        p['excl_us'] += r['excl_us']
        p['max_us'] = max(p['max_us'], r['max_us'])
    for p in per.values():
        p['pct'] = 100.0 * p['excl_us'] / (tot_ms * 1000.0) if tot_ms else None
    problems = []
    if not blocks:
        problems.append('không có khối T10 nào trong lượt')
    bad_win = [r['window'] for r in win_rows if not WIN_MS_OK[0] <= r['win_ms'] <= WIN_MS_OK[1]]
    if bad_win:
        problems.append(f'cửa sổ T10 không cỡ 10 s: {bad_win}')
    # S24: H7 đặt lại cửa sổ trước khi in, nên khối mất ở PC không làm win_ms của khối sau lớn ra.
    # (1) Mất dòng IRQ: isr_us = Σ chu kỳ của mọi handler / f (sàn), excl_us mỗi dòng cũng sàn, nên
    #     Σ excl_us ≤ isr_us ≤ Σ excl_us + số dòng; ra ngoài khoảng đó = thiếu dòng (hoặc dòng lạ).
    irq_bad = []
    for k, bl in enumerate(blocks):
        s = sum(i['excl_us'] for i in bl['irqs'])
        if not s <= bl['head']['isr_us'] <= s + len(bl['irqs']):
            irq_bad.append(k)
    if irq_bad:
        problems.append(f'khối T10 thiếu dòng IRQ (isr_us khác tổng excl_us): {irq_bad}')
    # (2) Mất nguyên khối: theo giờ PC đầu dòng (tools/h7_serial_capture.py), hai header liên tiếp cách
    #     10 s ± T10_GAP_TOL_S, và số khối ≈ thời gian từ header đầu tới hết đoạn / 10 s. Log không có giờ
    #     PC (PuTTY) thì không kiểm được: báo, không coi là đủ.
    heads = [bl['head'] for bl in blocks]
    if heads and all(h.get('t_pc_us') is not None for h in heads):
        gaps = [(b_['t_pc_us'] - a_['t_pc_us']) / 1e6 for a_, b_ in zip(heads, heads[1:])]
        lost = [i for i, g in enumerate(gaps) if abs(g - 10.0) > T10_GAP_TOL_S]
        if lost:
            problems.append(f'khoảng giữa hai khối T10 (giờ PC) khác 10 s sau khối {lost}: mất khối')
        a, b = span if span else (0, len(ev))
        ends = [e['t_pc_us'] for e in ev[a:b] if e.get('t_pc_us') is not None]
        if ends:
            expect = int((ends[-1] - heads[0]['t_pc_us']) / 1e6 // 10.0) + 1
            if len(heads) < expect:
                problems.append(f'{len(heads)} khối T10, kỳ vọng {expect} theo giờ PC: mất khối cuối')
    elif heads:
        problems.append('log serial không có giờ PC đầu dòng (không ghi bằng h7_serial_capture.py): '
                        'không kiểm được mất khối T10')
    a, b = span if span else (0, len(ev))
    pd = [e['print_drop'] for e in ev[a:b] if e['kind'] == 's06log']
    if len(pd) >= 2 and pd[-1] != pd[0]:
        problems.append(f'H7 bỏ {pd[-1] - pd[0]} dòng in trong lượt (print_drop)')
    if len(pd) < 2:
        problems.append('không đủ dòng S06log để kiểm print_drop')
    def mean(k):
        v = [r[k] for r in win_rows if r[k] is not None]
        return sum(v) / len(v) if v else None
    return dict(windows=len(win_rows), total_ms=tot_ms, isr_pct=mean('isr_pct'), idle_pct=mean('idle_pct'),
                rest_pct=mean('rest_pct'), idle_rel_unc_max=max((r['idle_rel_unc'] for r in win_rows
                                                                 if r['idle_rel_unc'] is not None), default=None),
                meas_cost_pct=100.0 * sum(r['meas_cost_us'] for r in win_rows) / (tot_ms * 1000.0) if tot_ms else None,
                per_irq=sorted(per.values(), key=lambda p: -p['excl_us']),
                gross_note='H7 chỉ in thời gian độc quyền; tổng gộp không có. Vào/ra ngắt: chưa đo.',
                excl_note='S35: "độc quyền" là ước lượng: handler cha chỉ trừ _s06_dt của handler con, '
                          'phần kế toán sau dấu thời gian của con và vào/ra ngắt lồng vẫn tính cho cha '
                          '(s06_load.h:30–44); cận trên của phần lẫn = số lần lồng × chi phí cặp đo, '
                          'chưa tách được',
                complete=not problems, problems=problems)


def _bounding(ev, kind, a, b):
    """(dòng `kind` cuối trước chỉ số a hoặc đầu trong đoạn, có bao đầu không;
    dòng `kind` đầu sau chỉ số b hoặc cuối trong đoạn, có bao cuối không)."""
    before = [e for e in ev[:a] if e['kind'] == kind]
    inside = [e for e in ev[a:b] if e['kind'] == kind]
    after = [e for e in ev[b:] if e['kind'] == kind]
    lo = before[-1] if before else (inside[0] if inside else None)
    hi = after[0] if after else (inside[-1] if inside else None)
    return lo, bool(before), hi, bool(after)


def link_stats(ev: List[dict], span=None) -> dict:
    """Hiệu các bộ đếm Link/Frame của lượt (T1), và các dòng Link period.
    S25: hiệu lấy giữa dòng in **ngay trước** 'link up' đầu và dòng in **ngay sau** 'link lost' cuối
    (bao hai đầu; bộ đếm cộng dồn từ lúc khởi động), để lỗi trong 0,5 s đầu/cuối không lọt. Không có
    dòng bao một đầu thì lấy dòng trong lượt và ghi `bounded_* = false` (công cụ chấm báo không kiểm
    đủ). Lần mất link cuối lượt (Pi dừng) làm `lost` và `rearm` tăng 1 (main.c: SPI1_RequestRestart):
    `expected_end_loss` = 1 khi có dòng bao cuối sau 'link lost'."""
    a, b = span if span else (0, len(ev))
    out = dict(link_lines=sum(1 for e in ev[a:b] if e['kind'] == 'link'),
               frame_lines=sum(1 for e in ev[a:b] if e['kind'] == 'frame'),
               interruptions=link_interruptions(ev[a:b]),
               link_periods=[{k: e[k] for k in ('frames', 'h7_ms', 'pi_ms')} for e in ev if e['kind'] == 'link_period'])
    f0, s0, f1, s1 = _bounding(ev, 'link', a, b)
    end_loss = 1 if (s1 and b < len(ev) and ev[b]['kind'] == 'link_lost') else 0
    out.update(bounded_start=s0, bounded_end=s1, expected_end_loss=end_loss)
    if f0 is not None and f1 is not None and f0 is not f1:
        out.update(err=f1['err'] - f0['err'], rearm=f1['rearm'] - f0['rearm'], lost=f1['lost'] - f0['lost'])
    g0, t0, g1, t1 = _bounding(ev, 'frame', a, b)
    if g0 is not None and g1 is not None and g0 is not g1 and g1['applied'] is not None and g0['applied'] is not None:
        out.update(applied=g1['applied'] - g0['applied'], skipped=g1['skipped'] - g0['skipped'],
                   unread=g1['unread'] - g0['unread'], age_max_us_since_boot=g1['age_max_us'])
    out['bounded_start'] = s0 and t0
    out['bounded_end'] = s1 and t1
    return out


def link_fails(st: dict) -> List[str]:
    """Lỗi T1 từ log serial (link_stats): mọi bộ đếm lỗi phải không tăng trong lượt."""
    f = []
    if st['interruptions']:
        f.append(f"link SPI mất rồi lên lại {st['interruptions']} lần trong lượt")
    for k, msg in (('err', 'H7 Link err tăng'), ('skipped', 'H7 Frame skipped tăng'),
                   ('unread', 'H7 Frame unread tăng')):
        if st.get(k):
            f.append(f'{msg} {st[k]}')
    for k in ('lost', 'rearm'):
        if st.get(k) is not None and st[k] - st['expected_end_loss'] != 0:
            f.append(f"H7 {k} tăng {st[k]} (kỳ vọng {st['expected_end_loss']} do mất link cuối lượt)")
    return f


def write_csv(path, rows):
    cols = list(rows[0]) if rows else ['window']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Đọc log serial H7 (S06 T1, T10)')
    ap.add_argument('log')
    ap.add_argument('--out', required=True)
    ap.add_argument('--f-real-mhz', type=float, default=F_REAL_MHZ, help='xung lõi H7 thật, cho cột max_us_real')
    ap.add_argument('--all-blocks', action='store_true', help='lấy mọi khối T10, không cắt theo link up/lost')
    a = ap.parse_args(argv)
    try:
        ev = read_log(a.log)
    except OSError as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    blocks, span = t10_blocks(ev, a.all_blocks)
    win_rows, irq_rows = t10_rows(blocks, a.f_real_mhz)
    summ = t10_summary(blocks, win_rows, irq_rows, ev, None if a.all_blocks else span)
    summ['link'] = link_stats(ev, span)
    summ['f_nominal_mhz'], summ['f_real_mhz'] = F_NOMINAL_MHZ, a.f_real_mhz
    summ['run_span_lines'] = None if span is None else [ev[span[0]]['line'],
                                                        ev[span[1]]['line'] if span[1] < len(ev) else None]
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, 'T10_windows.csv'), win_rows)
    write_csv(os.path.join(a.out, 'T10.csv'), irq_rows)
    with open(os.path.join(a.out, 'T10_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ))
    return 0 if summ['complete'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
