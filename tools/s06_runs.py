#!/usr/bin/env python3
"""
s06_runs.py — bảng lịch sử lượt S06_runs.csv (spec S06 mục 4 V5, mục 3.1 CB3).

File **chỉ nối thêm**: mỗi lệnh thêm một dòng, không sửa dòng cũ. Ba loại dòng:
  mo          lúc lượt bắt đầu, trước khi có số liệu: mã lượt, phép thử, commit, cáp, nguồn (lượt chết giữa
              chừng hay mã 2 không có `#ket_qua`, dòng này vẫn còn);
  ket_qua     sau khi chấm: công cụ, mã, lý do, lấy nguyên văn dòng cuối `#ket_qua` của script chấm;
  nguyen_nhan người đo tự nhập (W5): nhóm nguyên nhân, bằng chứng. Công cụ không tự phân loại, không tự quyết
              thay lượt, không kết luận lượt đạt hay không.
Lượt **đóng** (CB3 (6), C2-8) khi mọi công cụ bắt buộc của phép thử đã có dòng ket_qua (lần chấm lại thì lấy dòng
mới nhất của công cụ đó) và không còn mã 3 (b) `khong_du_dieu_kien`, (c) `thieu_bang_chung` nào thiếu dòng
nguyen_nhan. Mã 3 (a) `chua_ket_luan` đóng là chưa kết luận; không có bằng chứng ngoại sinh thì ghi
`cause --nhom chua_xac_dinh`, W5 tính là không đạt. Danh sách công cụ bắt buộc ở `s06_required_tools.json` (cùng
thư mục, spec CB3); phép thử chưa có trong đó thì chưa đóng được. Lượt chưa đóng chưa đưa vào mục 5 hay §6.1.
`status` liệt kê mọi lượt mở, còn lượt mở thì mã thoát 1: chưa lập `S06_summary.csv` (CB3 (7); công cụ lập bảng đó
chưa có, phải gọi `status` trước).

Mã 2 (CB1) không in `#ket_qua`, nên `ket_qua` từ chối và không ghi gì: sửa lệnh rồi chấm lại.

Usage:
  python tools/s06_runs.py open --file S06_runs.csv --run 2026-11-10_run01 --test T2 \\
      --commit-pi <sha> --commit-h7 <sha> --commit-esp32 <sha> --cap st-link --nguon usb
  python tools/s06_t1_check.py ... | tee T1_out.txt
  python tools/s06_runs.py result --file S06_runs.csv --run 2026-11-10_run01 --tool s06_t1_check --from T1_out.txt
  python tools/s06_runs.py cause --file S06_runs.csv --run 2026-11-10_run01 --nhom cap_thao_tac --bang-chung "<...>"
  python tools/s06_runs.py status --file S06_runs.csv [--required s06_required_tools.json]
Mã thoát: 0 đã ghi (status: mọi lượt đã đóng); 1 status còn lượt chưa đóng; 2 lệnh hay file không dùng được,
không ghi gì.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import sys
from typing import Dict, List, Tuple

COLUMNS = ['ghi_luc', 'loai_dong', 'lan', 'phep_thu', 'commit_pi', 'commit_h7', 'commit_esp32', 'cap', 'nguon',
           'cong_cu', 'ma', 'ly_do', 'nhom_nguyen_nhan', 'bang_chung', 'ghi_chu']
CAPS = ('st-link', 'khong')                     # spec mục 2 điểm 6, CB5
NGUONS = ('usb', 'buck')
TESTS = ('T1', 'T2', 'T3', 'T3b', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'CB5')
REASONS = ('chua_ket_luan', 'khong_du_dieu_kien', 'thieu_bang_chung')      # spec 3.1, bảng mã kết quả
# W5: danh sách ngoại sinh khóa trước (mục 4) + chưa xác định; `chua_xac_dinh` mặc định là lỗi bộ HIL
NHOMS = ('cap_thao_tac', 'mat_dien', 'sai_quy_trinh', 'chua_xac_dinh')
KET_QUA = re.compile(r'^#ket_qua,ma=(\d+)(?:,ly_do=(\w+))?\s*$')
REQUIRED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's06_required_tools.json')


class Refused(Exception):
    """Không ghi gì; lời nhắn in ra stderr, mã thoát 2."""


def read_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if r.fieldnames != COLUMNS:
            raise Refused(f'{path}: cột khác {COLUMNS}, không ghi tiếp')
        return list(r)


def append(path: str, row: Dict[str, str]) -> None:
    """Nối đúng một dòng, không đụng phần đã có; ghi xuống đĩa trước khi trả về."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    full = {k: row.get(k, '') for k in COLUMNS}
    full['ghi_luc'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(full)
        f.flush()
        os.fsync(f.fileno())


def last_result(text: str) -> Tuple[int, str]:
    """Dòng `#ket_qua` phải là dòng không rỗng cuối cùng của đầu ra script chấm (CB2)."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines or not lines[-1].startswith('#ket_qua'):
        raise Refused('đầu ra không kết thúc bằng #ket_qua: mã 2 không phải kết quả lượt (CB1), không ghi')
    m = KET_QUA.match(lines[-1])
    if not m:
        raise Refused(f'dòng #ket_qua sai dạng: {lines[-1]!r}')
    code, why = int(m.group(1)), m.group(2) or ''
    if code not in (0, 1, 3):
        raise Refused(f'mã {code} không phải kết quả lượt (CB2 chỉ có 0, 1, 3)')
    if code == 3 and why not in REASONS:
        raise Refused(f'mã 3 phải có một lý do trong {REASONS}, nhận {why!r}')
    if code != 3 and why:
        raise Refused(f'mã {code} không có lý do, nhận {why!r}')
    return code, why


def run_rows(rows: List[dict], run: str) -> List[dict]:
    return [r for r in rows if r['lan'] == run]


def need_open(rows: List[dict], run: str) -> dict:
    opens = [r for r in run_rows(rows, run) if r['loai_dong'] == 'mo']
    if not opens:
        raise Refused(f'lượt {run} chưa có dòng mở: chạy `open` trước')
    return opens[0]


def required_tools(path: str = REQUIRED) -> Dict[str, List[str]]:
    """{phép thử: [công cụ bắt buộc]} của file cấu hình; `dung_chung` trỏ phép thử dùng chung lượt của phép khác."""
    with open(path, encoding='utf-8') as f:
        try:
            cfg = json.load(f)
        except ValueError as exc:
            raise Refused(f'{path}: không đọc được: {exc}')
    out = {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'dung_chung'}
    out.update({test: out[base] for test, base in cfg.get('dung_chung', {}).items()})
    return out


def tool_name(tool: str) -> str:
    return tool[:-3] if tool.endswith('.py') else tool


def why_open(rows: List[dict], opened: dict, required: Dict[str, List[str]]) -> str:
    """CB3 (6): rỗng = lượt đã đóng, ngược lại là lý do lượt còn mở."""
    need = required.get(opened['phep_thu'])
    if need is None:
        return 'phép thử chưa có danh sách công cụ bắt buộc'
    rr = run_rows(rows, opened['lan'])
    last = {tool_name(r['cong_cu']): r for r in rr if r['loai_dong'] == 'ket_qua'}
    need = [tool_name(t) for t in need]
    missing = [t for t in need if t not in last]
    if missing:
        return 'thiếu kết quả ' + ', '.join(missing)
    open_b_c = [t for t in need if last[t]['ma'] == '3' and last[t]['ly_do'] != 'chua_ket_luan']
    if open_b_c and not any(r['loai_dong'] == 'nguyen_nhan' for r in rr):
        return ('mã 3 (b), (c) ở ' + ', '.join(open_b_c) + ' chưa ghi nhóm nguyên nhân và bằng chứng '
                '(không có thì ghi `chua_xac_dinh`: không đạt theo W5)')
    return ''


def cmd_open(a, rows) -> int:
    if run_rows(rows, a.run):
        raise Refused(f'lượt {a.run} đã có trong {a.file}: mỗi lượt một mã')
    if not all((a.commit_pi, a.commit_h7, a.commit_esp32)):
        raise Refused('dòng mở phải có commit của cả 3 board')
    append(a.file, dict(loai_dong='mo', lan=a.run, phep_thu=a.test, commit_pi=a.commit_pi, commit_h7=a.commit_h7,
                        commit_esp32=a.commit_esp32, cap=a.cap, nguon=a.nguon, ghi_chu=a.ghi_chu))
    return 0


def cmd_result(a, rows) -> int:
    opened = need_open(rows, a.run)
    with open(a.source, encoding='utf-8', errors='replace') as f:
        code, why = last_result(f.read())
    append(a.file, dict(loai_dong='ket_qua', lan=a.run, phep_thu=opened['phep_thu'], cong_cu=a.tool,
                        ma=str(code), ly_do=why, ghi_chu=a.ghi_chu))
    return 0


def cmd_cause(a, rows) -> int:
    opened = need_open(rows, a.run)
    if not a.bang_chung.strip():
        raise Refused('phải ghi bằng chứng (W5)')
    append(a.file, dict(loai_dong='nguyen_nhan', lan=a.run, phep_thu=opened['phep_thu'], nhom_nguyen_nhan=a.nhom,
                        bang_chung=a.bang_chung, ghi_chu=a.ghi_chu))
    return 0


def cmd_status(a, rows) -> int:
    required = required_tools(a.required)
    runs = [r for r in rows if r['loai_dong'] == 'mo']
    open_runs = []
    for op in runs:
        res = [r for r in run_rows(rows, op['lan']) if r['loai_dong'] == 'ket_qua']
        ca = [r for r in run_rows(rows, op['lan']) if r['loai_dong'] == 'nguyen_nhan']
        marks = ', '.join(f"{r['cong_cu']}={r['ma']}" + (f"({r['ly_do']})" if r['ly_do'] else '') for r in res)
        why = why_open(rows, op, required)
        if why:
            open_runs.append(op['lan'])
        print(f"{op['lan']} {op['phep_thu']} cap={op['cap']} nguon={op['nguon']}: "
              f"{marks or 'chưa có kết quả'}" + (f"; nguyên nhân: {ca[-1]['nhom_nguyen_nhan']}" if ca else '') +
              (f'; CHƯA ĐÓNG: {why}' if why else '; đã đóng'))
    print(f'{len(runs)} lượt, {len(open_runs)} chưa đóng')
    if open_runs:
        print('chưa lập S06_summary.csv, còn lượt mở: ' + ', '.join(open_runs))
    return 1 if open_runs else 0


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add(name, fn, *need_run):
        p = sub.add_parser(name)
        p.add_argument('--file', required=True)
        if need_run:
            p.add_argument('--run', required=True)
            p.add_argument('--ghi-chu', default='')
        p.set_defaults(fn=fn)
        return p

    p = add('open', cmd_open, True)
    p.add_argument('--test', required=True, choices=TESTS)
    p.add_argument('--commit-pi', required=True)
    p.add_argument('--commit-h7', required=True)
    p.add_argument('--commit-esp32', required=True)
    p.add_argument('--cap', required=True, choices=CAPS)
    p.add_argument('--nguon', required=True, choices=NGUONS)
    p = add('result', cmd_result, True)
    p.add_argument('--tool', required=True)
    p.add_argument('--from', dest='source', required=True, help='file đầu ra của script chấm')
    p = add('cause', cmd_cause, True)
    p.add_argument('--nhom', required=True, choices=NHOMS)
    p.add_argument('--bang-chung', required=True)
    p = add('status', cmd_status)
    p.add_argument('--required', default=REQUIRED, help='file danh sách công cụ bắt buộc theo phép thử')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    try:
        return a.fn(a, read_rows(a.file))
    except (Refused, OSError, KeyError) as exc:
        print(f'không ghi: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
