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


E_FIELDS = (8, 9)          # E,eseq,t_before,t_after,fl,fr,rl,rr[,run] (run: s06_load từ manifest lượt)


def parse_e(p):
    """Dòng E đã tách dấu phẩy → (eseq, t_before, t_after, [4 count], run hoặc None); None nếu sai dạng."""
    if p[0] != 'E' or len(p) not in E_FIELDS:
        return None
    try:
        return (int(p[1]), int(p[2]), int(p[3]), [int(x) for x in p[4:8]],
                int(p[8]) if len(p) == 9 else None)
    except ValueError:
        return None


def run_marks(path: str, run: Optional[int] = None) -> dict:
    """Mốc của MỘT lượt trong esp32_events.txt: {'run_start': {t_rx, kv}, 'run_end': {...}, 'run_stats': {...},
    '_run': số lượt hoặc None, '_runs': mọi lượt thấy trong file}.

    Manifest lượt (Codex Sol vòng 3 S58): s06_load ghi `run=` ở cả ba mốc, nên END và STATS được ghép với
    START bằng số lượt, không theo "lần đầu mỗi loại" (STATS của lượt kế từng bị lấy nhầm). File có hơn một
    `#run_start` mà không chỉ `run` → không chọn (chỉ có '_runs', '_multi' = True): công cụ báo thiếu bằng
    chứng. Firmware cũ (không có run=): một lượt mỗi file, END/STATS là mốc đầu tiên sau START.

    Vòng 4: S69 `run` không phải danh tính duy nhất (ESP32 khởi động lại thì đếm lại từ 1), nên END/STATS chỉ
    tìm tới START kế, và số lượt xuất hiện hơn một lần không được chọn; S71 `_problem` khác rỗng = không chọn
    được lượt (không có, trùng, chưa chỉ): công cụ dừng trước khi chấm, không đọc dòng của lượt nào."""
    marks = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            t_rx, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line.startswith('#run_'):
                continue
            p = line.split(',')
            name = p[0][1:]
            if name not in ('run_start', 'run_end', 'run_stats'):
                continue
            try:
                marks.append((name, dict(t_rx=int(t_rx), kv=dict(x.partition('=')[::2] for x in p[1:] if '=' in x))))
            except ValueError:
                continue
    starts = [m for n, m in marks if n == 'run_start']
    runs = [m['kv'].get('run') for m in starts]
    out = dict(_runs=runs, _run=None, _multi=False, _problem='')
    if run is None and len(starts) > 1:
        out['_multi'] = True
        out['_problem'] = f'file có {len(starts)} dòng #run_start mà không chọn lượt (--run)'
        return out
    if run is not None:
        idx = [i for i, (n, m) in enumerate(marks) if n == 'run_start' and m['kv'].get('run') == str(run)]
        if len(idx) != 1:
            out['_problem'] = (f'không có lượt run={run} trong file' if not idx else
                               f'run={run} xuất hiện {len(idx)} lần (ESP32 khởi động lại giữa file?): không xác định lượt')
            return out
        i0 = idx[0]
    else:
        i0 = next((i for i, (n, _) in enumerate(marks) if n == 'run_start'), None)
        if i0 is None:
            # END/STATS without START: kept so the loss counters they carry still count (S63)
            for n, m in marks:
                out.setdefault(n, m)
            return out
    start = marks[i0][1]
    rid = start['kv'].get('run')
    out['run_start'], out['_run'] = start, (int(rid) if rid is not None else None)
    for n, m in marks[i0 + 1:]:
        if n == 'run_start':
            break                                    # S69: the next run begins, whatever its id
        if n not in out and m['kv'].get('run') == rid:
            out[n] = m
    return out


def e_of_run(e, marks: dict) -> bool:
    """S71: dòng E (của `parse_e`) có thuộc lượt đã chọn không. Mốc có `run=` thì so số lượt (dòng E không có
    số lượt không thuộc lượt nào); mốc không có `run=` là firmware cũ, dòng E có số lượt là định dạng trộn."""
    rid = marks.get('_run')
    if rid is not None:
        return e[4] == rid
    if 'run_start' in marks and e[4] is not None:
        raise ValueError('mốc lượt không có run= nhưng dòng E có: định dạng trộn (S71)')
    return True


def _mark_int(marks, name, key):
    try:
        return int(marks[name]['kv'][key])
    except (KeyError, ValueError, TypeError):
        return None


def enc_stream(eseq, t_before, t0_us: float, t1_us: float, marks: Optional[dict] = None, runs=None) -> dict:
    """Luồng dòng E (giờ ESP32).

    * Trong khoảng chấm [t0_us; t1_us]: số đọc, hở và trùng `eseq` (chỉ báo vị trí).
    * S41 (Codex Sol vòng 2/2): hở tính giữa eseq nhỏ nhất và lớn nhất nhận được không thấy mất ĐẦU/ĐUÔI.
      Nên đủ luồng được chứng minh bằng số của producer: số dòng E của lượt, khác nhau và liền nhau, phải
      bằng `enc_reads_run − enc_ovf_run` của `#run_stats`/`#run_end`. Thiếu → mất dữ liệu (Z19).
    * S58 (vòng 3, manifest lượt): dòng E của lượt = dòng có trường `run` bằng số lượt của mốc (`runs`,
      cùng thứ tự với `eseq`), không theo thời gian (lượt kế trong cùng file, lần đọc dở lúc END). Firmware
      cũ không có `run`: lấy t_before ≥ `#run_start` timestamp_us (một lượt mỗi file).

    `ok` False = mất dữ liệu (thất bại theo Z19); `known` False = không đủ bằng chứng (thiếu #run_start
    timestamp_us, #run_end hoặc enc_reads_run của #run_stats: firmware cũ hay mất dòng; file nhiều lượt
    mà không chọn lượt)."""
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
    out.update(enc_ovf_run=ovf, ev_qdrop_run=qdrop, multi_run=bool(m.get('_multi')), run_problem=m.get('_problem', ''))
    # S63: a loss counter seen > 0 is a failure even when another mark is missing
    if (ovf or 0) > 0 or (qdrop or 0) > 0:
        out['ok'] = False
    rid = m.get('_run')
    if runs is not None and rid is not None:
        in_run = np.asarray([r == rid for r in runs], dtype=bool)
    elif t_start is not None:
        in_run = tb >= t_start
    else:
        in_run = None
    if None not in (ovf, qdrop, reads_run) and in_run is not None:
        rq = np.unique(eseq[in_run])
        expected = reads_run - ovf
        contiguous = len(rq) == 0 or int(rq[-1] - rq[0] + 1) == len(rq)
        out.update(run_reads_expected=expected, run_reads_received=int(len(rq)),
                   run_reads_missing=expected - int(len(rq)), known=True)
        out['ok'] = out['ok'] and contiguous and len(rq) == expected
    return out


QSEQ_FIELDS = {'J': (7,), 'I': (12,), 'R': (4,), 'P': (10, 11), 'S': (15, 16)}   # P 11: + run (manifest lượt)


FLOAT_FROM = {'I': 9, 'S': 15}     # first field printed with %g/%f (s06_load.c); the fields before it are integers


def _p_of_run(p, marks: dict) -> bool:
    """S71, S75: a P line that passed `_line_ok` belongs to the selected run: the run id of the marks (11
    fields); marks without a run id: the old 10 fields. No START in `marks`: not decided here."""
    if 'run_start' not in marks:
        return True
    rid = marks.get('_run')
    return len(p) == 10 if rid is None else len(p) == 11 and int(p[10]) == rid


def _line_ok(p) -> bool:
    """S56 (vòng 3): a line counts for the stream only if its payload is readable: the field count of its
    kind and every field a number (`nan` allowed: the values of an I line, the temperature of an S line).
    S67 (vòng 4): integer fields by the same rule as the scoring parsers (`int`), so a line counted as
    received is never dropped later."""
    if p[0] not in QSEQ_FIELDS or len(p) not in QSEQ_FIELDS[p[0]]:
        return False
    n = FLOAT_FROM.get(p[0], len(p))
    try:
        for x in p[1:n]:
            int(x)
        for x in p[n:]:
            float(x)
    except ValueError:
        return False
    return True


def qseq_stream(path: str, marks: Optional[dict] = None) -> dict:
    """S40 (Codex Sol vòng 2/2): đủ luồng các dòng có `qseq` (J, I, R, P, S và dòng văn bản `#…,qseq=`)
    của cả lượt, chứng minh bằng mốc producer ở HAI ĐẦU: `#run_start` và `#run_end` đều mang qseq của
    chính nó, nên mọi qseq trong [q_start; q_end] phải có đúng một dòng. Thiếu → mất dữ liệu (Z19); không
    có hai mốc → không đủ bằng chứng. Kèm `ev_qdrop_run` (mất trước khi gửi) của #run_end.
    Vòng 3: S56 dòng có qseq mà nội dung không đọc được (sai số trường, trường không phải số) là HỎNG, không
    tính là đã nhận; S58 mốc lấy theo lượt đã chọn (`marks` của `run_marks`); S63 lỗ qseq hay
    `ev_qdrop_run` > 0 là thất bại dù thiếu mốc khác."""
    m = marks if marks is not None else run_marks(path)
    q_start, q_end = _mark_int(m, 'run_start', 'qseq'), _mark_int(m, 'run_end', 'qseq')
    qdrop = _mark_int(m, 'run_end', 'ev_qdrop_run')
    got, bad = [], []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            _, sep, line = raw.rstrip('\n').partition('\t')
            if not sep or not line or line[0] == 'E':
                continue
            p = line.split(',')
            try:
                if line[0] == '#':
                    kv = dict(x.partition('=')[::2] for x in p[1:] if '=' in x)
                    got.append(int(kv['qseq']))
                    continue
                q = int(p[1])
            except (KeyError, ValueError, IndexError):
                continue
            (got if _line_ok(p) and (p[0] != 'P' or _p_of_run(p, m)) else bad).append(q)
    out = dict(q_start=q_start, q_end=q_end, expected=None, received=None, missing=None, dups=None,
               malformed=None, ev_qdrop_run=qdrop, multi_run=bool(m.get('_multi')), known=False,
               ok=(qdrop or 0) == 0)
    if q_start is not None and q_end is not None and q_end >= q_start:
        inr = [q for q in got if q_start <= q <= q_end]
        uniq = set(inr)
        out.update(expected=q_end - q_start + 1, received=len(uniq), missing=q_end - q_start + 1 - len(uniq),
                   dups=len(inr) - len(uniq), malformed=sum(1 for q in bad if q_start <= q <= q_end),
                   known=qdrop is not None)
        out['ok'] = out['ok'] and out['missing'] == 0 and out['dups'] == 0
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
    if s.get('run_problem'):
        n.append(f"{s['run_problem']}: không chấm (S58, S71)")
    if not s['known']:
        n.append('thiếu #run_start timestamp_us, #run_end (enc_ovf_run, ev_qdrop_run) hoặc #run_stats '
                 '(enc_reads_run): không chứng minh được luồng E đủ cả lượt')
    return n
