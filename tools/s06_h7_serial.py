#!/usr/bin/env python3
"""
s06_h7_serial.py — đọc log serial của H7 (ST-Link VCP, USART3, 115200; lưu bằng PuTTY hoặc chương trình
terminal bất kỳ) cho S06 v3.4: các dòng T1 (Link, Frame, Link period) và bảng T10 (tải H7, R22).

Định dạng in (hil-framework `feat/s06-h7` 598a451, `main.c`, `s06_load.c`; đầu dòng có thể có dấu thời
gian của chương trình terminal, bỏ qua):
  SPI link up at t=<ms> ms
  SPI link lost (no frame for <ms> ms): ...
  Link period: <frames> frames, H7 <ms> ms from first to last frame (Pi: <ms> ms at 1 kHz)[ up_ms=<tick>]
  Link: <up|down> lost=<n>  SPI frames=<n>/s err=<n> rearm=<n> last_err=0x<hex>          mỗi 0,5 s
  Frame: unread=<n>  new seen=<n> taken=<n> repl=<n> drop=<n>  applied=<n> skipped=<n> age_max=<n>us
  S06log: pushed=<n> dropped=<n>  print_drop=<n>
  T10: win_ms=<ms> isr_us=<µs> idle_pass=<n> idle_us_est=<µs> cal=<n>/<cyc> ovh_cyc=<cyc>       mỗi 10 s
       từ H7 feat/s06-pf3 d93514c thêm ở cuối: ` wn=<n> t0_ms=<tick> lk=<n> up_ms=<tick>` (số cửa sổ, tick đầu
       cửa sổ, số lần link up, tick frame đầu của đoạn link); khi có, mất khối xét bằng `wn` liền nhau và neo đầu/cuối (`wn_analyse`), không bằng giờ PC
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
  * B3 (Liam 12/11, phiên THESIS local_694a2f03; `06_SPECS/2026-11-12_De_xuat_S06_B3_quy_doi_tai_F446_v1.md`
    mục 3 (A)): cận dưới phần rảnh `idle_pct_lo` = % rảnh × (1 − bất định tương đối), không dưới 0;
    cận trên tải `load_upper_pct` = 100 % − cận dưới phần rảnh (cột mỗi cửa sổ; tóm tắt: 100 − trung bình
    các cận dưới, và `load_upper_pct_max_window`; có cho khối tải nặng và nhóm A khi có `--load-schedule`).
    Công cụ không nhân hệ số F446 (2,67; 4,56) và không đọc vùng: việc đó của luận văn.
  * Đơn vị thời gian: H7 chia số chu kỳ cho SystemCoreClock danh định (480 MHz), win_ms đếm tick HAL của
    cùng đồng hồ, nên mọi tỉ lệ % đúng dù đồng hồ H7 lệch (trước 09/11 nhanh 0,64 %; từ 09/11 xung HSE/5 của
    ST-LINK, lệch ≤ 0,002 % theo `H7_CLOCK/2026-11-09_run04`, f thật ≈ 480 MHz, vẫn nằm trong dải mặc định dưới). µs của H7 là "µs danh định"; µs thật
    = µs H7 × f_danh_định / f_thật: cột `max_us_real_lo`, `max_us_real_hi` (X22, C1-15: theo tần số đo của
    chính lượt `--f-real-mhz`, không có thì theo dải [478; 486] MHz; 483,1 MHz là số suy từ đo 27/10, không
    phải của lượt nghiệm thu).
  * Tổng gộp (cận trên) không có: H7 chỉ in thời gian độc quyền; cột để trống. Vào/ra ngắt: chưa đo.
  * Luồng đủ (§4.3 điều 6): print_drop không tăng trong khoảng dùng, win_ms ∈ [9 900; 10 100], và (S24,
    Codex Sol vòng 1/2, Liam duyệt 30/10) không mất dòng IRQ (Σ excl_us ≤ isr_us ≤ Σ excl_us + số dòng)
    và không mất khối (giờ PC đầu dòng: hai header cách 10 s ± 0,5 s, số khối khớp độ dài đoạn; log
    không có giờ PC thì không kiểm được, ghi là thiếu). C1-14 (bảng mã kết quả): mất dữ liệu (print_drop
    tăng, mất khối, thiếu dòng IRQ, cửa sổ sai) → mã 1; thiếu bằng chứng (không có khối, không có giờ PC,
    thiếu S06log) → mã 3 `thieu_bang_chung`.

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

import s06_stream as stm

F_NOMINAL_MHZ = 480.0             # SystemCoreClock danh định (HSE 8 MHz, PLLN 120, PLLP 2): H7 chia cho số này
F_BAND_MHZ = (478.0, 486.0)       # X22: dải tần số H7 khi chưa khớp đồng hồ của lượt
WIN_MS_OK = (9_900, 10_100)
T10_GAP_TOL_S = 0.5               # S24 [ĐỀ XUẤT CODE]: hai header T10 liên tiếp cách 10 s ± 0,5 s theo giờ PC
# T10U-3 (Codex 15/11, Liam "Sửa cả 8 + vòng 2"): H7 chụp cửa sổ rồi mới in (main.c S06Load_Snapshot →
# S06Load_PrintNext ở các vòng sau), cộng thời gian truyền VCP: cửa sổ thật kết thúc sớm hơn giờ PC của header
# một khoảng d trong [0; PRINT_DELAY_MAX_MS] [ĐỀ XUẤT CODE_S06, chưa đo].
PRINT_DELAY_MAX_MS = 1000.0
# T10U-1: idle_us_est H7 tính = (idle_pass · cal_cyc / cal_n) / (SystemCoreClock / 1e6), chia nguyên hai lần
# (s06_load.c S06Load_PrintNext); công cụ tính lại và cho lệch tối đa chừng này.
IDLE_EST_TOL_US = 1
# T10U-R3-1 (Codex round 3; Liam 15/11, CODE_S06 local_da965ab4 'Them kiem SPI1'): the H7 10 s window counter
# runs from boot (main.c _t10_last), not from link up, so PC times alone cannot tell "first header late" from
# "first header lost". The SPI1 IRQ count of the first block of the run can: the Pi sends one frame per ms, so a
# window that started before link up has SPI1 n below win_ms; n >= win_ms - this margin = the window lies (almost)
# wholly after link up: the header before it may be lost, or link came up right at the window start (missing).
# [PROPOSAL CODE_S06, 100 frames = 0.1 s]
T10_SPI_IRQ = 'SPI1'
FIRST_SPI_FULL_MARGIN = 100
# Premise of that check (Codex round 4; Liam 15/11 'Sua 2 + vong 5'): the Pi really sent one frame per ms. The
# 'Link period' line after the run gives frames received and H7 ms from first to last frame: H7 ms - (frames - 1)
# = the frame gaps of the run (+ clock drift). Gaps + LINK_DRIFT_MS must stay under FIRST_SPI_FULL_MARGIN, so no
# window can lose 100 frames; else (or no such line, or link lost inside the run) the check has no ground: missing.
# [PROPOSAL CODE_S06: 30 ms = 50 ppm H7-Pi over a 610 s run]
LINK_DRIFT_MS = 30

_RE = {
    'link_up': re.compile(r'SPI link up at t=(\d+) ms'),
    'link_lost': re.compile(r'SPI link lost \(no frame for (\d+) ms\)'),
    'link_period': re.compile(r'Link period: (\d+) frames, H7 (\d+) ms from first to last frame '
                              r'\(Pi: (\d+) ms at 1 kHz\)(?: up_ms=(\d+))?'),      # up_ms: H7 d93514c
    'link': re.compile(r'Link: (up|down) lost=(\d+)\s+SPI frames=(\d+)/s err=(\d+) rearm=(\d+) '
                       r'last_err=0x([0-9a-fA-F]+)'),
    'frame': re.compile(r'Frame: unread=(\d+)\s+new seen=(\d+) taken=(\d+) repl=(\d+) drop=(\d+)'
                        r'(?:\s+applied=(\d+) skipped=(\d+) age_max=(\d+)us)?'),
    's06log': re.compile(r'S06log: pushed=(\d+) dropped=(\d+)\s+print_drop=(\d+)'),
    # Rounds 6–8 (Liam 15/11, CODE_S06 'H7 in so cua so + vong 6', 'Sua du + vong 7', 'Sua hep + vong 8'; H7
    # feat/s06-pf3 d93514c): optional wn (window number since boot), t0_ms (HAL tick at the window start), lk (link-up
    # count since boot), up_ms (first-frame tick of the current link period) at the end of the header
    't10': re.compile(r'T10: win_ms=(\d+) isr_us=(\d+) idle_pass=(\d+) idle_us_est=(\d+) cal=(\d+)/(\d+) '
                      r'ovh_cyc=(\d+)(?: wn=(\d+) t0_ms=(\d+)(?: lk=(\d+)(?: up_ms=(\d+))?)?)?'),
    't10_irq': re.compile(r'T10 (\S+)\s+n=(\d+) excl_us=(\d+) max_us=(\d+)\.(\d\d)'),
    'relay': re.compile(r'Relay: sent=(\d+) dropped=(\d+)'),       # W8, tools/s06_relay_check.py
}
_FIELDS = {
    'link_up': ('t_ms',), 'link_lost': ('timeout_ms',), 'link_period': ('frames', 'h7_ms', 'pi_ms', 'up_ms'),
    'relay': ('sent', 'dropped'),
    'link': ('state', 'lost', 'fps', 'err', 'rearm', 'last_err'),
    'frame': ('unread', 'new_seen', 'taken', 'repl', 'drop', 'applied', 'skipped', 'age_max_us'),
    's06log': ('pushed', 'dropped', 'print_drop'),
    't10': ('win_ms', 'isr_us', 'idle_pass', 'idle_us_est', 'cal_n', 'cal_cyc', 'ovh_cyc', 'wn', 't0_ms', 'lk', 'up_ms'),
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
    if not all_blocks and span is not None:
        mode = wn_mode(ev)
        if mode == 'lk':
            blocks = wn_analyse(ev, span)['blocks']        # T10U-R6-6: from the headers it checks, even none in span
        elif blocks:
            blocks = blocks[1:]                  # cửa sổ đầu có phần trước link up
    return blocks, span


# ── H7 window numbers (rounds 6–8; Liam 15/11, CODE_S06 'H7 in so cua so + vong 6', 'Sua du + vong 7',
# 'Sua hep + vong 8') ──
# H7 feat/s06-pf3 d93514c: 'T10: ... wn=<n> t0_ms=<tick> lk=<n> up_ms=<tick>' and 'Link period: ... up_ms=<tick>'.
# S06Load_PollSnapshot takes a window only when the tick is S06_WIN_MS past the start and the previous block is
# printed; the start of the next window is the tick the snapshot read, so windows tile (t0 of wn+1 = t0 + win_ms of wn)
# and none is shorter than H7_WIN_MIN_MS. lk = link-up count since boot, up_ms = first-frame tick of the current (or
# last) link period (= 'SPI link up at t=' of that period), both at the snapshot. 60c69df (no lk) and 344f21f (no
# up_ms) were never flashed: not accepted. Ticks are HAL ms (uint32): every comparison is on the difference to the
# link-up tick (T10U-R6-4).
H7_WIN_MIN_MS = 10000
_TICK_MOD = 1 << 32


def tick_rel(t: int, t_ref: int) -> int:
    """Signed ms from t_ref to t on the uint32 HAL tick (T10U-R6-4: correct across one wrap, |difference| < 2^31)."""
    return ((t - t_ref + (1 << 31)) % _TICK_MOD) - (1 << 31)


def wn_mode(ev: List[dict]) -> Optional[str]:
    """'lk' when a header of the log carries up_ms (H7 d93514c and later), 'wn' when headers carry wn but none up_ms
    (60c69df, 344f21f: not accepted), None for older builds (the PC-time / SPI1 path). Decided on the WHOLE log
    (T10U-R6-6: not on the headers of the span only)."""
    heads = [e for e in ev if e['kind'] == 't10']
    if any(h.get('up_ms') is not None for h in heads):
        return 'lk'
    if any(h.get('wn') is not None for h in heads):
        return 'wn'
    return None


def _wn_full(h) -> bool:
    return all(h.get(k) is not None for k in ('wn', 't0_ms', 'lk', 'up_ms'))


def wn_analyse(ev: List[dict], span) -> dict:
    """Windows of the run from the H7 numbers, and why the run is not complete. dict(blocks, data_lost, missing).

    * Run: link-up tick t_up (first frame, 'SPI link up at t='); last frame t_up + h7_ms of the run's 'Link period'
      (link_period_of_span). A window is kept when 0 <= t0 − t_up and t0 + win_ms − t_up <= h7_ms.
    * Headers checked (log order): every header printed after the link-up line, up to and including the END ANCHOR =
      the first one whose window ends after the last frame (headers after it, and after a later link up, are not the
      run's). Round 7 (T10U-R7-1, R7-2): no header printed before the link-up line is used (it may be of another boot,
      or a gap before it lies wholly before the run).
    * Every checked header must carry up_ms = t_up and one lk (T10U-R6-3: a header of another link period, its link
      lines lost): else missing, no windows. A header without wn/t0_ms/lk/up_ms: damaged line, data lost (R6-6).
    * Consecutive checked headers: wn + 1 and t0 = previous t0 + win_ms. A jump of wn by k + 1 loses k windows that
      fill the gap between the two: a gap shorter than k × H7_WIN_MIN_MS cannot hold them, a header is damaged: data
      lost (T10U-R8-1). Else, with room for a whole window inside [first frame; last frame]: data lost when the gap
      lies wholly inside the run, missing when it straddles an edge (the lost windows' edges are unknown; R6-5 rest);
      no such room: not the run's (T10U-R6-5, R7-2). wn not increasing (H7 restarted) or contiguous wn with a wrong
      t0: damaged, data lost.
    * Start (T10U-R6-1): a checked header with t0 < t_up (the window holding link up) proves nothing was lost before
      the first kept window. Without one, the first kept window may start less than H7_WIN_MIN_MS after link up (the
      window before it, >= H7_WIN_MIN_MS long, held link up); otherwise missing (cannot tell).
    * End (T10U-R6-2): the end anchor proves nothing was lost after the last kept window. Without it, fewer than
      H7_WIN_MIN_MS between the last checked window end and the last frame leave no room for a full window;
      otherwise missing (the log may stop before the next header is printed).
    * Link lost and up again inside the run ('SPI link up' lines): missing, no windows."""
    a, b = span
    dl, ms = [], []
    t_up = ev[a]['t_ms']
    i_end = next((i for i in range(b, len(ev)) if ev[i]['kind'] == 'link_up'), len(ev))
    blk_of, cur = {}, None
    for i, e in enumerate(ev[:i_end]):
        if e['kind'] == 't10':
            cur = dict(head=e, irqs=[])
            blk_of[i] = cur
        elif e['kind'] == 't10_irq' and cur is not None:
            cur['irqs'].append(e)
    if link_interruptions(ev[a:b]):
        ms.append('link mất rồi lên lại trong lượt: không lấy cửa sổ T10')
        return dict(blocks=[], data_lost=dl, missing=ms)
    lp, why = link_period_of_span(ev, span)
    t_last = lp['h7_ms'] if lp else None                 # relative to t_up
    if lp is None:
        ms.append(f'không xác định được frame cuối của lượt: {why}')

    def rel0(h):
        return tick_rel(h['t0_ms'], t_up)

    def rel1(h):
        return rel0(h) + h['win_ms']

    post = []
    for i in sorted(k for k in blk_of if k > a):
        h = blk_of[i]['head']
        if t_last is None and i > b:
            break                                          # no last frame: no end anchor, only headers of the span
        if not _wn_full(h):
            dl.append(f"header T10 dòng {h['line']} thiếu wn/t0_ms/lk/up_ms (dòng hỏng): mất khối")
            continue
        post.append(i)
        if h['win_ms'] < H7_WIN_MIN_MS:                    # S06Load_PollSnapshot never takes a shorter one
            dl.append(f"cửa sổ T10 wn={h['wn']} dài {h['win_ms']} ms, ngắn hơn {H7_WIN_MIN_MS} ms "
                      '(bản H7 có up_ms không tạo ra): dòng hỏng')
        if t_last is not None and rel1(h) > t_last:
            break                                          # end anchor
    chk = [blk_of[i]['head'] for i in post]
    other = sorted({h['up_ms'] for h in chk if h['up_ms'] != t_up % _TICK_MOD})
    lks = sorted({h['lk'] for h in chk})
    if other or len(lks) > 1:
        ms.append(f'header T10 sau link up thuộc đoạn link khác (up_ms {other}, link up t={t_up % _TICK_MOD}; lk {lks}): '
                  'link mất rồi lên lại trong lượt, không lấy cửa sổ')
        return dict(blocks=[], data_lost=dl, missing=ms)
    t_end = t_last if t_last is not None else float('inf')
    for p, q in zip(chk, chk[1:]):
        if q['wn'] <= p['wn']:
            dl.append(f"số cửa sổ T10 không tăng {p['wn']} → {q['wn']} (H7 khởi động lại?): dòng hỏng")
        elif q['wn'] != p['wn'] + 1:
            lost_n, gap = q['wn'] - p['wn'] - 1, rel0(q) - rel1(p)
            room = min(rel0(q), t_end) - max(rel1(p), 0)
            if gap < lost_n * H7_WIN_MIN_MS:
                # T10U-R8-1 (Codex round 8; Liam 'Sua 2 muc + vong 9'): k lost windows of >= H7_WIN_MIN_MS each
                # cannot fit in a shorter gap: one of the two headers is damaged
                dl.append(f"số cửa sổ T10 nhảy {p['wn']} → {q['wn']} trong {gap} ms (< {lost_n} × {H7_WIN_MIN_MS}): "
                          'dòng hỏng, mất khối')
            elif room >= H7_WIN_MIN_MS and rel1(p) >= 0 and rel0(q) <= t_end:
                dl.append(f"số cửa sổ T10 nhảy {p['wn']} → {q['wn']}: mất khối")
            elif room >= H7_WIN_MIN_MS:
                # R6-5 rest (round 8): the gap straddles an edge of the run; the lost windows fill it but where their
                # edges fall is unknown, so a whole window of the run may or may not be among them: cannot tell
                ms.append(f"số cửa sổ T10 nhảy {p['wn']} → {q['wn']} qua mép lượt (chỗ trong lượt {room} ms): "
                          'không biết cửa sổ mất có nằm trọn trong lượt không')
        elif q['t0_ms'] != (p['t0_ms'] + p['win_ms']) % _TICK_MOD:
            dl.append(f"cửa sổ T10 wn={q['wn']} bắt đầu {q['t0_ms']} ms, kỳ vọng {(p['t0_ms'] + p['win_ms']) % _TICK_MOD}")
    kept = [blk_of[i] for i in post if rel0(blk_of[i]['head']) >= 0
            and (t_last is None or rel1(blk_of[i]['head']) <= t_last)]
    full = [h for h in chk if rel0(h) >= 0]
    if full and not any(rel0(h) < 0 for h in chk) and rel0(full[0]) >= H7_WIN_MIN_MS:
        ms.append(f"cửa sổ T10 trọn đầu tiên bắt đầu {rel0(full[0])} ms sau link up, không có header của cửa sổ chứa "
                  'link up: không chứng minh được không mất khối đầu')
    if t_last is not None and chk and rel1(chk[-1]) <= t_last and t_last - rel1(chk[-1]) >= H7_WIN_MIN_MS:
        ms.append(f'còn {t_last - rel1(chk[-1])} ms frame sau cửa sổ T10 cuối, không có header của cửa sổ kế tiếp '
                  '(log dừng sớm hay mất khối): không chứng minh được không mất khối cuối')
    if not kept and chk:
        ms.append('không có cửa sổ T10 nào nằm trọn trong đoạn link')
    return dict(blocks=kept, data_lost=dl, missing=ms)


def link_period_of_span(ev: List[dict], span):
    """(Link period line of the run, '' ) or (None, reason). The H7 prints 'SPI link lost' and 'Link period' back to
    back in the same branch (main.c), so the run's line is the LOG LINE right after the span's link-lost line (Codex
    rounds 5–7: the next recognised line may be another period's when lines in between are lost), with pi_ms ==
    frames - 1 (the H7 prints both from the same counter: another value = a damaged line), and, when the line carries
    up_ms (H7 d93514c), up_ms = the run's link-up tick."""
    a, b = span
    if b >= len(ev) or ev[b]['kind'] != 'link_lost':
        return None, 'không có dòng SPI link lost của lượt'
    e = ev[b + 1] if b + 1 < len(ev) else None
    if e is None or e['kind'] != 'link_period' or e['line'] != ev[b]['line'] + 1:
        return None, 'không có dòng Link period ngay dòng sau link lost của lượt'
    if e['pi_ms'] != e['frames'] - 1:
        return None, f"dòng Link period hỏng (Pi {e['pi_ms']} ms khác frames − 1 = {e['frames'] - 1})"
    if e.get('up_ms') is not None and e['up_ms'] != ev[a]['t_ms'] % _TICK_MOD:
        return None, f"dòng Link period của đoạn link khác (up_ms={e['up_ms']}, link up t={ev[a]['t_ms']})"
    return e, ''


def idle_lower_pct(idle_pct: Optional[float], rel_unc: Optional[float], idle_pass: int) -> Optional[float]:
    """B3 (S06 T10, Liam 12/11, THESIS local_694a2f03): lower bound of the idle share from the calibration
    uncertainty, idle_pct × (1 − rel_unc), not below 0. idle_pass = 0 (no idle pass in the window): 0.
    No calibration (idle_pct None): None."""
    if idle_pct is None:
        return None
    if not idle_pass:
        return 0.0
    if rel_unc is None:
        return None
    return max(0.0, idle_pct * (1.0 - rel_unc))


def load_upper(rows) -> dict:
    """B3: L = 100 % − lower bound of the idle share. `load_upper_pct` = 100 − the `win_ms`-weighted mean of
    the per-window lower bounds (T10U-4: windows may differ in length within WIN_MS_OK, so this is the bound of
    the time mean of the run or segment); `load_upper_pct_max_window` = the largest per-window bound. None when
    a window has no bound (no calibration, or a window dropped by the T10U-1 check)."""
    lo = [r['idle_pct_lo'] for r in rows]
    if not lo or any(v is None for v in lo):
        return dict(load_upper_pct=None, load_upper_pct_max_window=None)
    w = [r['win_ms'] for r in rows]
    # T10U-R3-2 round 4: the largest window bound never below that window's ISR share (float rounding)
    return dict(load_upper_pct=100.0 - sum(v * x for v, x in zip(lo, w)) / sum(w),
                load_upper_pct_max_window=max(max(100.0 - v, r.get('isr_pct', 0.0)) for v, r in zip(lo, rows)))


def idle_check(h: dict) -> str:
    """T10U-1: why the idle figures of a T10 header cannot be used ('' = usable). The H7 estimate must equal
    idle_pass · cal_cyc / cal_n / f (integer divisions, ± IDLE_EST_TOL_US), and the LOWER bound of the idle time
    (estimate × (1 − rel. uncertainty), T10U-R2-1: the centre estimate may exceed the window by its own
    uncertainty) plus the ISR time must fit the window, with no tolerance (T10U-R3-2: a tolerance there let the
    load bound fall below the ISR share; isr_us and idle_us_est are both floored on the H7, so H7 rounding alone
    never pushes a genuine line over the window). ASCII text (T10U-R2-3: CSV written on any code page)."""
    win_us = h['win_ms'] * 1000.0
    if win_us <= 0:
        return 'win_ms = 0'
    if h['cal_n']:
        expect = (h['idle_pass'] * h['cal_cyc'] // h['cal_n']) // int(F_NOMINAL_MHZ)
        if abs(h['idle_us_est'] - expect) > IDLE_EST_TOL_US:
            return f'idle_us_est {h["idle_us_est"]} != idle_pass*cal ({expect})'
    unc = (1.0 / h['cal_n'] + 1.0 / h['idle_pass']) if h['cal_n'] and h['idle_pass'] else 0.0
    idle_lo_us = h['idle_us_est'] * max(0.0, 1.0 - unc)
    if h['isr_us'] > win_us or idle_lo_us + h['isr_us'] > win_us:
        return 'idle_lo + isr_us > window'
    return ''


def t10_rows(blocks, f_real_mhz: Optional[float] = None):
    """X22, C1-15: `max_us_real_lo/hi` theo tần số đo của chính lượt, không có thì theo dải [478; 486] MHz."""
    win_rows, irq_rows = [], []
    lo_f, hi_f = (f_real_mhz, f_real_mhz) if f_real_mhz else F_BAND_MHZ
    for k, bl in enumerate(blocks):
        h = bl['head']
        win_us = h['win_ms'] * 1000.0
        n_calls = sum(i['n'] for i in bl['irqs'])
        idle_pct = 100.0 * h['idle_us_est'] / win_us if h['cal_n'] else None
        isr_pct = 100.0 * h['isr_us'] / win_us
        unc = (1.0 / h['cal_n'] + 1.0 / h['idle_pass']) if h['cal_n'] and h['idle_pass'] else None
        bad = idle_check(h)
        lo = None if bad else idle_lower_pct(idle_pct, unc, h['idle_pass'])
        if lo is not None:              # T10U-R3-2 round 4: idle_check passed in µs; keep that in % (float rounding)
            lo = min(lo, 100.0 - isr_pct)
        win_rows.append(dict(window=k, line=h['line'], win_ms=h['win_ms'], isr_us=h['isr_us'],
                             isr_pct=isr_pct, idle_pass=h['idle_pass'], idle_us=h['idle_us_est'],
                             idle_pct=idle_pct,
                             rest_pct=None if idle_pct is None else 100.0 - isr_pct - idle_pct,
                             idle_rel_unc=unc,
                             cal_n=h['cal_n'], cal_cyc=h['cal_cyc'], ovh_cyc=h['ovh_cyc'],
                             meas_cost_us=n_calls * h['ovh_cyc'] / F_NOMINAL_MHZ,
                             idle_pct_lo=lo, load_upper_pct=None if lo is None else max(100.0 - lo, isr_pct),
                             idle_bad=bad))
        for i in bl['irqs']:
            irq_rows.append(dict(window=k, irq=i['irq'], count=i['n'], excl_us=i['excl_us'],
                                 gross_us='', max_us=i['max_us'],
                                 max_us_real_lo=i['max_us'] * F_NOMINAL_MHZ / hi_f,
                                 max_us_real_hi=i['max_us'] * F_NOMINAL_MHZ / lo_f,
                                 pct=100.0 * i['excl_us'] / win_us))
    # S31: common row index first; µs columns are what the H7 prints (cycles already divided and
    # rounded on the H7: the raw cycles of the spec column list are not recoverable, not filled in)
    win_rows = [dict(idx=n, **r) for n, r in enumerate(win_rows)]
    irq_rows = [dict(idx=n, **r) for n, r in enumerate(irq_rows)]
    return win_rows, irq_rows


def read_heavy_ms(path: str) -> Optional[tuple]:
    """(start_ms, end_ms) of the heavy row of a file of rpi5/s06_load_schedule.py, None without one."""
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row['kind'] == 'heavy':
                return int(row['start_ms']), int(row['end_ms'])
    return None


def mark_segments(win_rows, blocks, ev, span, heavy_ms) -> None:
    """Spec 2.2 (Z8): T10 reports the heavy block and the group A part apart. Time 0 = PC time of the first
    'SPI link up'. A line reaches the PC a delay in [0; PRINT_DELAY_MAX_MS] after the H7 event (T10U-3), for the
    header (window end) as for the 'SPI link up' line (time 0), so the true window lies inside
    [b − win − d_max; b + d_max] with b the PC time of the header from time 0: a window goes to a segment only if
    that whole span is on one side of every block edge, else `mixed` (in neither). Without PC times the column
    stays empty."""
    t0 = ev[span[0]].get('t_pc_us') if span else None
    for r, bl in zip(win_rows, blocks):
        end = bl['head'].get('t_pc_us')
        if t0 is None or end is None or heavy_ms is None:
            r['segment'] = ''
            continue
        a_ms = (end - t0) / 1000.0 - r['win_ms'] - PRINT_DELAY_MAX_MS
        b_ms = (end - t0) / 1000.0 + PRINT_DELAY_MAX_MS
        if a_ms >= heavy_ms[0] and b_ms <= heavy_ms[1]:
            r['segment'] = 'heavy'
        elif b_ms <= heavy_ms[0] or a_ms >= heavy_ms[1]:
            r['segment'] = 'nhom_a'
        else:
            r['segment'] = 'mixed'


def segment_means(win_rows) -> dict:
    """% ISR, % idle of the heavy block and of the group A part (empty dict without the column)."""
    out = {}
    for seg in ('heavy', 'nhom_a'):
        rows = [r for r in win_rows if r.get('segment') == seg]
        if rows:
            idle = [r['idle_pct'] for r in rows if r['idle_pct'] is not None]
            out[seg] = dict(windows=len(rows), isr_pct=sum(r['isr_pct'] for r in rows) / len(rows),
                            idle_pct=sum(idle) / len(idle) if idle else None, **load_upper(rows))
    return out


def first_block_spi_n(ev: List[dict], span) -> Optional[int]:
    """SPI1 IRQ count of the first T10 block inside the span (the IRQ lines right after its header, up to the
    next header); None when that block has no SPI1 line."""
    a, b = span
    started = False
    for e in ev[a:b]:
        if e['kind'] == 't10':
            if started:
                return None
            started = True
        elif started and e['kind'] == 't10_irq' and e['irq'] == T10_SPI_IRQ:
            return e['n']
    return None


def spi_premise_problem(ev: List[dict], span) -> str:
    """'' when the run proves the Pi sent ~1 frame per ms (LINK_DRIFT_MS comment), else the reason."""
    a, b = span
    if link_interruptions(ev[a:b]):
        return 'link mất rồi lên lại trong lượt'
    lp, why = link_period_of_span(ev, span)
    if lp is None:
        return why
    gaps = lp['h7_ms'] - lp['pi_ms']
    if gaps + LINK_DRIFT_MS >= FIRST_SPI_FULL_MARGIN:
        return (f"Link period: H7 {lp['h7_ms']} ms, {lp['frames']} frame, hở {gaps} ms "
                f"(+ {LINK_DRIFT_MS} ms trôi >= {FIRST_SPI_FULL_MARGIN})")
    # Codex round 5 (gửi bù): the sum is signed, so it cannot bound one window; only H7 with wn/t0_ms closes that
    return ''


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
    # C1-14: `data_lost` = measurement data lost (Z19, mã 1); `missing` = evidence missing (mã 3)
    data_lost, missing = [], []
    if not blocks:
        missing.append('không có khối T10 nào trong lượt')
    bad_win = [r['window'] for r in win_rows if not WIN_MS_OK[0] <= r['win_ms'] <= WIN_MS_OK[1]]
    if bad_win:
        data_lost.append(f'cửa sổ T10 không cỡ 10 s: {bad_win}')
    # S24: H7 đặt lại cửa sổ trước khi in, nên khối mất ở PC không làm win_ms của khối sau lớn ra.
    # (1) Mất dòng IRQ: isr_us = Σ chu kỳ của mọi handler / f (sàn), excl_us mỗi dòng cũng sàn, nên
    #     Σ excl_us ≤ isr_us ≤ Σ excl_us + số dòng; ra ngoài khoảng đó = thiếu dòng (hoặc dòng lạ).
    irq_bad = []
    for k, bl in enumerate(blocks):
        s = sum(i['excl_us'] for i in bl['irqs'])
        if not s <= bl['head']['isr_us'] <= s + len(bl['irqs']):
            irq_bad.append(k)
    if irq_bad:
        data_lost.append(f'khối T10 thiếu dòng IRQ (isr_us khác tổng excl_us): {irq_bad}')
    # (2) Mất nguyên khối: theo giờ PC đầu dòng (tools/h7_serial_capture.py), hai header liên tiếp cách
    #     10 s ± T10_GAP_TOL_S, và số khối ≈ thời gian từ header đầu tới hết đoạn / 10 s. Log không có giờ
    #     PC (PuTTY) thì không kiểm được: báo, không coi là đủ.
    # T10U-2 (Codex 15/11): the gap check runs on EVERY header of the span, the first one (dropped from the
    # windows because it starts before link up) included, and the first header must come within 10 s + tol
    # of 'SPI link up'; on the kept headers only, a lost header right after the dropped one merged its IRQ
    # lines into the dropped block and nothing saw it.
    a, b = span if span else (0, len(ev))
    heads_all = [e for e in ev[a:b] if e['kind'] == 't10']
    heads = [bl['head'] for bl in blocks]
    # rounds 6, 7: H7 with wn/t0_ms/lk: lost blocks from the H7 window numbers and ticks (wn_analyse), not from PC
    # times or SPI1 counts (those stay for logs of older H7 builds). Mode from the whole log (T10U-R6-6).
    mode = wn_mode(ev) if span is not None else None
    use_wn = mode == 'lk'
    if mode == 'wn':
        missing.append('header T10 có wn, t0_ms mà không có up_ms (bản H7 60c69df, 344f21f, chưa từng nạp): '
                       'công cụ không nhận')
    if use_wn:
        wa = wn_analyse(ev, span)
        data_lost += wa['data_lost']
        missing += wa['missing']
    elif heads_all and all(h.get('t_pc_us') is not None for h in heads_all):
        gaps = [(b_['t_pc_us'] - a_['t_pc_us']) / 1e6 for a_, b_ in zip(heads_all, heads_all[1:])]
        lost = [i for i, g in enumerate(gaps) if abs(g - 10.0) > T10_GAP_TOL_S]
        if lost:
            data_lost.append(f'khoảng giữa hai khối T10 (giờ PC) khác 10 s sau khối {lost}: mất khối')
        t_up = ev[a].get('t_pc_us') if span else None
        # T10U-R2-2: the first header may reach the PC up to PRINT_DELAY_MAX_MS late (T10U-3), the link-up line
        # on time. T10U-R3-1 (Liam 15/11 'Sua 2 + vong 4'): the header carries no window number, so a first header
        # in (10 + tol; 10 + tol + d_max] s may be late or may be the second one (first lost): missing, not
        # complete; later than that it cannot be the first one: data lost
        if t_up is not None:
            dt = (heads_all[0]['t_pc_us'] - t_up) / 1e6
            if dt > 10.0 + T10_GAP_TOL_S + PRINT_DELAY_MAX_MS / 1e3:
                data_lost.append(f'header T10 đầu tới sau SPI link up {dt:.1f} s (> 10 s): mất khối đầu')
            elif dt > 10.0 + T10_GAP_TOL_S:
                missing.append(f'header T10 đầu tới sau SPI link up {dt:.1f} s, trong dải trễ in: '
                               'không phân biệt được trễ in với mất khối đầu')
        ends = [e['t_pc_us'] for e in ev[a:b] if e.get('t_pc_us') is not None]
        if ends:
            expect = int((ends[-1] - heads_all[0]['t_pc_us']) / 1e6 // 10.0) + 1
            if len(heads_all) < expect:
                data_lost.append(f'{len(heads_all)} khối T10, kỳ vọng {expect} theo giờ PC: mất khối cuối')
    elif heads:
        missing.append('log serial không có giờ PC đầu dòng (không ghi bằng h7_serial_capture.py): '
                       'không kiểm được mất khối T10')
    a, b = span if span else (0, len(ev))
    pd = [e['print_drop'] for e in ev[a:b] if e['kind'] == 's06log']
    if len(pd) >= 2 and pd[-1] != pd[0]:
        data_lost.append(f'H7 bỏ {pd[-1] - pd[0]} dòng in trong lượt (print_drop)')
    if len(pd) < 2:
        missing.append('không đủ dòng S06log để kiểm print_drop')
    # T10U-R3-1, SPI1 check (Liam 15/11 'Them kiem SPI1'): the first block of the span (dropped as partial) must
    # really start before link up, i.e. hold fewer SPI1 frames than its window; independent of PC times
    if span is not None and heads_all and not use_wn:
        n_spi = first_block_spi_n(ev, span)
        full = heads_all[0]['win_ms'] - FIRST_SPI_FULL_MARGIN
        if n_spi is None:
            missing.append(f'khối T10 đầu của lượt không có dòng {T10_SPI_IRQ}: không kiểm được mất khối đầu')
        elif n_spi >= full:
            missing.append(f'khối T10 đầu của lượt có {T10_SPI_IRQ} n={n_spi} (>= {full}): cửa sổ gần trọn sau '
                           'link up, không phân biệt được mất khối đầu với link up sát đầu cửa sổ')
        why = spi_premise_problem(ev, span)
        if why:
            missing.append(f'kiểm {T10_SPI_IRQ} khối đầu không có căn cứ: {why}')
    # T10U-1: idle figures that cannot come from the H7 formula, or do not fit the window = a damaged line
    bad_idle = [(r['window'], r['idle_bad']) for r in win_rows if r.get('idle_bad')]
    if bad_idle:
        data_lost.append(f'header T10 có số rảnh không dùng được (dòng hỏng?): {bad_idle}')
    # T10U-5: no idle calibration = no load bound, evidence missing (not a pass)
    no_cal = [r['window'] for r in win_rows if r['cal_n'] == 0]           # T10U-5 round 2: idle_pass 0 too
    if no_cal:
        missing.append(f'không có hiệu chuẩn vòng rỗng (cal=0/…): không có cận tải ở cửa sổ {no_cal}')
    problems = data_lost + missing

    def mean(k):
        v = [r[k] for r in win_rows if r[k] is not None]
        return sum(v) / len(v) if v else None
    return dict(windows=len(win_rows), total_ms=tot_ms, isr_pct=mean('isr_pct'), idle_pct=mean('idle_pct'),
                rest_pct=mean('rest_pct'), idle_rel_unc_max=max((r['idle_rel_unc'] for r in win_rows
                                                                 if r['idle_rel_unc'] is not None), default=None),
                **load_upper(win_rows),
                meas_cost_pct=100.0 * sum(r['meas_cost_us'] for r in win_rows) / (tot_ms * 1000.0) if tot_ms else None,
                per_irq=sorted(per.values(), key=lambda p: -p['excl_us']),
                gross_note='H7 chỉ in thời gian độc quyền; tổng gộp không có. Vào/ra ngắt: chưa đo.',
                excl_note='S35: "độc quyền" là ước lượng: handler cha chỉ trừ _s06_dt của handler con, '
                          'phần kế toán sau dấu thời gian của con và vào/ra ngắt lồng vẫn tính cho cha '
                          '(s06_load.h:30–44); cận trên của phần lẫn = số lần lồng × chi phí cặp đo, '
                          'chưa tách được',
                complete=not problems, problems=problems, data_lost=data_lost, missing=missing)


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
    with open(path, 'w', newline='', encoding='utf-8') as f:          # T10U-R2-3: not the local code page
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Đọc log serial H7 (S06 T1, T10)')
    ap.add_argument('log')
    ap.add_argument('--out', required=True)
    ap.add_argument('--f-real-mhz', type=float, default=None,
                    help='tần số H7 đo được của chính lượt; bỏ trống = dải X22 [478; 486] MHz')
    ap.add_argument('--all-blocks', action='store_true', help='lấy mọi khối T10, không cắt theo link up/lost')
    ap.add_argument('--load-schedule', default=None,
                    help='file lịch của lượt (s06_load_schedule.py): báo riêng khối tải nặng và nhóm A (spec 2.2)')
    a = ap.parse_args(argv)
    try:
        ev = read_log(a.log)
        heavy_ms = read_heavy_ms(a.load_schedule) if a.load_schedule else None
    except (OSError, KeyError, ValueError) as exc:
        print(f'lỗi đầu vào: {exc}', file=sys.stderr)
        return 2
    blocks, span = t10_blocks(ev, a.all_blocks)
    win_rows, irq_rows = t10_rows(blocks, a.f_real_mhz)
    summ = t10_summary(blocks, win_rows, irq_rows, ev, None if a.all_blocks else span)
    if a.load_schedule:
        mark_segments(win_rows, blocks, ev, span, heavy_ms)
        summ['segments'] = segment_means(win_rows)
        # spec 2.2: T10 reports both parts; T10U-6 (Codex 15/11): a schedule with a heavy block needs windows
        # of BOTH parts (a schedule without one, --no-heavy, is not split: mark_segments leaves the column empty)
        need = ('heavy', 'nhom_a') if heavy_ms is not None else ()      # no heavy block: no split (column empty)
        for seg, txt in (('heavy', 'khối tải nặng'), ('nhom_a', 'phần nhóm A')):
            if seg in need and seg not in summ['segments']:
                summ['missing'].append(f'không tách được {txt} (thiếu giờ PC hoặc không có cửa sổ trọn trong phần đó)')
        summ['problems'] = summ['data_lost'] + summ['missing']
        summ['complete'] = not summ['problems']
    summ['link'] = link_stats(ev, span)
    summ['f_nominal_mhz'], summ['f_real_mhz'] = F_NOMINAL_MHZ, a.f_real_mhz
    # T10U-3 (Codex round 2: not verified, Liam 15/11 'Sua 4 + vong 3'): the segment split and the first-header
    # check assume every H7 line reaches the PC within this delay; the value is a proposal, not measured
    summ['print_delay_max_ms'] = PRINT_DELAY_MAX_MS
    summ['print_delay_note'] = ('assumed bound on the H7 print + VCP delay, not measured: a header later than this '
                                'can put a window in the wrong part (heavy / group A)')
    summ['run_span_lines'] = None if span is None else [ev[span[0]]['line'],
                                                        ev[span[1]]['line'] if span[1] < len(ev) else None]
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, 'T10_windows.csv'), win_rows)
    write_csv(os.path.join(a.out, 'T10.csv'), irq_rows)
    with open(os.path.join(a.out, 'T10_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ))
    return stm.report_result(1 if summ['data_lost'] else (3 if summ['missing'] else 0), ['thieu_bang_chung'])


if __name__ == '__main__':
    raise SystemExit(main())
