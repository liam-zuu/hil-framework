#!/usr/bin/env python3
"""
s06_stream.py — kiểm đủ luồng đo phía ESP32 dùng chung cho các công cụ chấm S06 (T3b, T5) và mở vòng
count encoder (Codex Sol vòng 1/2, nhóm A; Liam duyệt 30/10 "Duyệt, sửa hết").

  * Z19 / §4.3 điều 6: mất dữ liệu đo = THẤT BẠI, không phải chỉ báo. Ở đây: dòng E (một lần đọc count
    4 bánh, `eseq` tăng 1 mỗi lần đọc) phải liền trong khoảng chấm; `enc_ovf_run` và `ev_qdrop_run` của
    #run_end (vòng đệm encoder / hàng đợi sự kiện tràn trước khi gửi) phải bằng 0. Thiếu #run_end thì
    không biết có tràn không: báo là thiếu bằng chứng.
  * S17: count tích lũy của firmware là int32 (`encoder.c`, `logger.h:23`); hiệu hai lần đọc lấy sau khi
    mở vòng 32 bit có dấu theo từng bánh (PCNT 16 bit đã được driver cộng dồn, không mở vòng 16 bit).

Bản trùng chức năng có ở tools/s06_esp32_check.py (nhánh feat/s06-pi); khi gộp `int/post-g2` thì
dùng chung một bản.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def unwrap_i32(counts) -> np.ndarray:
    """Cột count int32 (theo thứ tự đọc) → int64 liên tục: mỗi bước lấy hiệu mod 2³² về [−2³¹; 2³¹)."""
    c = np.asarray(counts, dtype=np.int64)
    if c.ndim == 1:
        c = c[:, None]
    out = np.empty_like(c)
    out[0] = c[0]
    d = np.diff(c, axis=0)
    d = (d + (1 << 31)) % (1 << 32) - (1 << 31)
    out[1:] = c[0] + np.cumsum(d, axis=0)
    return out


def run_marks(path: str) -> dict:
    """#run_start/#run_end của esp32_events.txt: {'run_start': {t_rx, kv}, 'run_end': {...}}."""
    out = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line.startswith('#run_'):
                continue
            p = line.split(',')
            name = p[0][1:]
            if name in ('run_start', 'run_end') and name not in out:
                try:
                    out[name] = dict(t_rx=int(t_rx), kv=dict(x.partition('=')[::2] for x in p[1:] if '=' in x))
                except ValueError:
                    continue
    return out


def enc_stream(eseq, t_before, t0_us: float, t1_us: float, marks: Optional[dict] = None) -> dict:
    """Luồng dòng E trong [t0_us; t1_us] (giờ ESP32): số đọc, hở và trùng `eseq`, tràn ở #run_end.
    `ok` False = mất dữ liệu (thất bại theo Z19); `known` False = không đủ bằng chứng (thiếu #run_end)."""
    eseq = np.asarray(eseq, dtype=np.int64)
    tb = np.asarray(t_before, dtype=float)
    sel = (tb >= t0_us) & (tb <= t1_us)
    q = np.sort(eseq[sel])
    uniq = np.unique(q)
    gaps = int(uniq[-1] - uniq[0] + 1 - len(uniq)) if len(uniq) else 0
    dups = int(len(q) - len(uniq))
    out = dict(reads=int(len(q)), eseq_gaps=gaps, eseq_dups=dups, enc_ovf_run=None, ev_qdrop_run=None,
               known=False, ok=gaps == 0 and dups == 0 and len(q) > 0)
    end = (marks or {}).get('run_end')
    if end is not None:
        try:
            out['enc_ovf_run'] = int(end['kv']['enc_ovf_run'])
            out['ev_qdrop_run'] = int(end['kv']['ev_qdrop_run'])
            out['known'] = True
            out['ok'] = out['ok'] and out['enc_ovf_run'] == 0 and out['ev_qdrop_run'] == 0
        except (KeyError, ValueError):
            pass
    return out


def stream_notes(s: dict) -> list:
    n = []
    if s['reads'] == 0:
        n.append('không có dòng E nào trong khoảng chấm')
    if s['eseq_gaps'] or s['eseq_dups']:
        n.append(f"luồng đọc encoder hở {s['eseq_gaps']}, trùng {s['eseq_dups']} (Z19: mất dữ liệu)")
    if s['enc_ovf_run']:
        n.append(f"vòng đệm encoder ESP32 tràn {s['enc_ovf_run']} lần trong lượt (Z19)")
    if s['ev_qdrop_run']:
        n.append(f"hàng đợi sự kiện ESP32 bỏ {s['ev_qdrop_run']} dòng trong lượt (Z19)")
    if not s['known']:
        n.append('thiếu #run_end (enc_ovf_run, ev_qdrop_run): không biết ESP32 có tràn trước khi gửi không')
    return n
