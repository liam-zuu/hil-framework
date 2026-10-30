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
    """#run_start/#run_end/#run_stats của esp32_events.txt: {'run_start': {t_rx, kv}, 'run_end': {...},
    'run_stats': {...}} (lần đầu mỗi loại)."""
    out = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line.startswith('#run_'):
                continue
            p = line.split(',')
            name = p[0][1:]
            if name in ('run_start', 'run_end', 'run_stats') and name not in out:
                try:
                    out[name] = dict(t_rx=int(t_rx), kv=dict(x.partition('=')[::2] for x in p[1:] if '=' in x))
                except ValueError:
                    continue
    return out


def _mark_int(marks, name, key):
    try:
        return int(marks[name]['kv'][key])
    except (KeyError, ValueError, TypeError):
        return None


def enc_stream(eseq, t_before, t0_us: float, t1_us: float, marks: Optional[dict] = None) -> dict:
    """Luồng dòng E (giờ ESP32).

    * Trong khoảng chấm [t0_us; t1_us]: số đọc, hở và trùng `eseq` (chỉ báo vị trí).
    * S41 (Codex Sol vòng 2/2): hở tính giữa eseq nhỏ nhất và lớn nhất nhận được không thấy mất ĐẦU/ĐUÔI.
      Nên đủ luồng được chứng minh bằng số của producer: firmware chỉ đọc count khi lượt đang chạy, nên
      mọi dòng E có t_before ≥ `#run_start` timestamp_us là của lượt; số dòng đó, khác nhau và liền nhau,
      phải bằng `enc_reads_run − enc_ovf_run` của `#run_stats`/`#run_end`. Thiếu → mất dữ liệu (Z19).

    `ok` False = mất dữ liệu (thất bại theo Z19); `known` False = không đủ bằng chứng (thiếu #run_start
    timestamp_us, #run_end hoặc enc_reads_run của #run_stats: firmware cũ hay mất dòng)."""
    eseq = np.asarray(eseq, dtype=np.int64)
    tb = np.asarray(t_before, dtype=float)
    sel = (tb >= t0_us) & (tb <= t1_us)
    q = np.sort(eseq[sel])
    uniq = np.unique(q)
    gaps = int(uniq[-1] - uniq[0] + 1 - len(uniq)) if len(uniq) else 0
    dups = int(len(q) - len(uniq))
    out = dict(reads=int(len(q)), eseq_gaps=gaps, eseq_dups=dups, enc_ovf_run=None, ev_qdrop_run=None,
               run_reads_expected=None, run_reads_received=None, run_reads_missing=None,
               known=False, ok=gaps == 0 and dups == 0 and len(q) > 0)
    m = marks or {}
    ovf, qdrop = _mark_int(m, 'run_end', 'enc_ovf_run'), _mark_int(m, 'run_end', 'ev_qdrop_run')
    reads_run, t_start = _mark_int(m, 'run_stats', 'enc_reads_run'), _mark_int(m, 'run_start', 'timestamp_us')
    out.update(enc_ovf_run=ovf, ev_qdrop_run=qdrop)
    if ovf is not None and qdrop is not None:
        out['ok'] = out['ok'] and ovf == 0 and qdrop == 0
    if None not in (ovf, qdrop, reads_run, t_start):
        rq = np.unique(eseq[tb >= t_start])
        expected = reads_run - ovf
        contiguous = len(rq) == 0 or int(rq[-1] - rq[0] + 1) == len(rq)
        out.update(run_reads_expected=expected, run_reads_received=int(len(rq)),
                   run_reads_missing=expected - int(len(rq)), known=True)
        out['ok'] = out['ok'] and contiguous and len(rq) == expected
    return out


def qseq_stream(path: str) -> dict:
    """S40 (Codex Sol vòng 2/2): đủ luồng các dòng có `qseq` (J, I, R, P, S và dòng văn bản `#…,qseq=`)
    của cả lượt, chứng minh bằng mốc producer ở HAI ĐẦU: `#run_start` và `#run_end` đều mang qseq của
    chính nó, nên mọi qseq trong [q_start; q_end] phải có đúng một dòng. Thiếu → mất dữ liệu (Z19); không
    có hai mốc → không đủ bằng chứng. Kèm `ev_qdrop_run` (mất trước khi gửi) của #run_end."""
    q_start = q_end = qdrop = None
    got = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            _, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line or line[0] == 'E':
                continue
            p = line.split(',')
            try:
                if line[0] == '#':
                    kv = dict(x.partition('=')[::2] for x in p[1:] if '=' in x)
                    q = int(kv['qseq'])
                    if p[0] == '#run_start' and q_start is None:
                        q_start = q
                    elif p[0] == '#run_end' and q_end is None:
                        q_end = q
                        qdrop = int(kv['ev_qdrop_run']) if 'ev_qdrop_run' in kv else None
                else:
                    q = int(p[1])
            except (KeyError, ValueError, IndexError):
                continue
            got.append(q)
    out = dict(q_start=q_start, q_end=q_end, expected=None, received=None, missing=None, dups=None,
               ev_qdrop_run=qdrop, known=False, ok=True)
    if q_start is not None and q_end is not None and q_end >= q_start and qdrop is not None:
        inr = [q for q in got if q_start <= q <= q_end]
        uniq = set(inr)
        out.update(expected=q_end - q_start + 1, received=len(uniq), missing=q_end - q_start + 1 - len(uniq),
                   dups=len(inr) - len(uniq), known=True)
        out['ok'] = out['missing'] == 0 and out['dups'] == 0 and qdrop == 0
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
    if s['run_reads_missing']:
        n.append(f"cả lượt: nhận {s['run_reads_received']} dòng E, producer đọc {s['run_reads_expected']} "
                 f"(enc_reads_run − enc_ovf_run): mất đầu/đuôi hoặc giữa lượt (S41, Z19)")
    if not s['known']:
        n.append('thiếu #run_start timestamp_us, #run_end (enc_ovf_run, ev_qdrop_run) hoặc #run_stats '
                 '(enc_reads_run): không chứng minh được luồng E đủ cả lượt')
    return n
