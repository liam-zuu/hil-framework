#!/usr/bin/env python3
"""hoi_codex: STRA_A asks Codex inside D:\\STRA_A (design 03_CODE/docs/2026-11-15_Thiet_ke_CODE_C13_v1.md section 4).

  python D:\\STRA_A\\hoi_codex.py --cau "<câu hỏi>" [--luot <mã lượt> ...] [--doc <file trong D:\\STRA_A> ...]
  python D:\\STRA_A\\hoi_codex.py --thu

Checks before the call (else no call): working folder D:\\STRA_A, SHA of D:\\STRA_A\\.codex\\config.toml = the mark,
every --luot / --doc inside D:\\STRA_A. No argument reaches Codex as such (no -s, -C, --cd, -c sandbox, dangerously).
Writes CODEX\\hoi_<date>_<n>.md (new file), calls codex exec (gpt-6.1-sol, effort high: Liam 15/11 "Ghim high"),
checks the log (workdir, sandbox, model, effort); wrong or exit != 0: the answer is NOT printed. Appends the wait
time to CODEX\\thoi_gian.csv. The copy in D:\\STRA_A is installed by chep_luot_A.py cai-hoi-codex (SHA checked).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import shutil
import subprocess
import sys

ROOT = r'D:\STRA_A'
SHA_CODEX_CONFIG = '748bf0c8c5aeb08fd222337172ee939e6bc83ea1f3caf1ff5bcacbca693c3cf9'   # design section 6
MODEL = 'gpt-6.1-sol'
EFFORT = 'high'
WAIT_COLS = ['hoi', 'bat_dau', 'ket_thuc', 'giay', 'ma_thoat', 'model', 'effort', 'luot', 'codex_home']
CAU_THU = 'Câu hỏi thử, không tune: liệt kê tên các file trong HOP_THU (không đọc nội dung) và trả lời "thử xong".'


class Refused(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def real(p):
    return os.path.normcase(os.path.realpath(p))


def inside_root(p):
    r = real(ROOT)
    q = real(p if os.path.isabs(p) else os.path.join(ROOT, p))
    return q == r or q.startswith(r.rstrip(os.sep) + os.sep)


def run_codex(cmd, stdin_bytes, cwd):
    """The only call of Codex (tests replace this function; no environment variable can change the program)."""
    r = subprocess.run(cmd, input=stdin_bytes, capture_output=True, cwd=cwd, timeout=3600)
    return r.returncode, r.stdout + r.stderr


def codex_cmd(out):
    exe = shutil.which('codex') or 'codex'
    return [exe, 'exec', '--skip-git-repo-check', '-m', MODEL, '-c', f'model_reasoning_effort={EFFORT}', '-o', out, '-']


def build_question(cau, luot, doc):
    files = ['HOP_THU\\README.md (bảng file, đọc phần cuối: dòng mới)', 'HOP_THU\\mien_A.csv (miền gain của A)']
    files += [f'HOP_THU\\luot\\{x}\\ (luot.json, score.json: đọc bằng Python)' for x in luot]
    files += [d for d in doc]
    files.append('các lượt trước trong HOP_THU\\luot\\ khi cần; spec trong HOP_THU khi cần')
    return '\n'.join([
        '# Câu hỏi cho Codex (trợ lý tune tay chiến dịch A)',
        '',
        '**Vai:** trợ lý tune tay chiến dịch A. Liam tự quyết gain; bạn chỉ đề xuất.',
        '**Hàng rào:** chỉ dùng file trong `D:\\STRA_A`. Thấy gain "đã đạt", kết quả DOE, GA, tìm khả thi, kiểm cuối '
        'thì không dùng và nói ra.',
        '**Không chấm thay:** kết quả lượt và quyết định dừng chỉ theo `score.json` chính thức.',
        '',
        '## File cần đọc (tự đọc trong `D:\\STRA_A`; câu hỏi không chép nội dung file)',
        *[f'- {x}' for x in files],
        '',
        '## Câu hỏi (nguyên văn)',
        cau,
        '',
        '## Khuôn trả lời',
        'Tiếng Việt đời thường, tối đa khoảng 15 dòng:',
        '1. Bộ gain đề xuất cho lượt kế, **chỉ trong miền của `HOP_THU\\mien_A.csv`** (không đề xuất nới miền).',
        '2. Lý do từng gain đổi (tối đa 5 dòng).',
        '3. Rủi ro.',
        '4. Câu hỏi cho Liam.',
        ''])


def next_name(codex_dir, day):
    n = 1
    while any(os.path.exists(os.path.join(codex_dir, f'{p}_{day}_{n}{e}'))
              for p, e in (('hoi', '.md'), ('tra_loi', '.md'), ('log', '.txt'))):
        n += 1
    return f'{day}_{n}'


def check_log(text):
    need = [('workdir', r'^workdir: D:\\STRA_A\s*$'), ('sandbox', r'^sandbox: custom permissions\s*$'),
            ('model', rf'^model: {re.escape(MODEL)}\s*$'), ('effort', rf'^reasoning effort: {EFFORT}\s*$')]
    return [k for k, pat in need if not re.search(pat, text, re.M | re.I)]


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False, description='STRA_A hỏi Codex trong D:\\STRA_A')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--cau')
    g.add_argument('--thu', action='store_true')
    ap.add_argument('--luot', nargs='*', default=[])
    ap.add_argument('--doc', nargs='*', default=[])
    a = ap.parse_args(argv)
    try:
        os.chdir(ROOT)
        if real(os.getcwd()) != real(ROOT):
            raise Refused('không đứng ở D:\\STRA_A')
        cfg = os.path.join(ROOT, '.codex', 'config.toml')
        if not os.path.isfile(cfg) or sha256(cfg) != SHA_CODEX_CONFIG:
            raise Refused('khóa Codex (.codex\\config.toml) khác mốc: báo Liam, không gọi')
        for x in a.luot:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', x):
                raise Refused(f'--luot {x!r}: chỉ tên thư mục lượt')
            p = os.path.join(ROOT, 'HOP_THU', 'luot', x)
            if not os.path.isdir(p) or not inside_root(p):
                raise Refused(f'--luot {x}: không có trong HOP_THU\\luot')
        for d in a.doc:
            if not inside_root(d) or not os.path.exists(os.path.join(ROOT, d) if not os.path.isabs(d) else d):
                raise Refused(f'--doc {d}: phải là file có thật trong D:\\STRA_A')
        cau = CAU_THU if a.thu else a.cau
        if not cau.strip():
            raise Refused('câu hỏi rỗng')
        cdir = os.path.join(ROOT, 'CODEX')
        t0 = dt.datetime.now().astimezone()
        tag = next_name(cdir, t0.strftime('%Y%m%d'))
        q, out, log = (os.path.join(cdir, f'{p}_{tag}{e}') for p, e in
                       (('hoi', '.md'), ('tra_loi', '.md'), ('log', '.txt')))
        body = build_question(cau, a.luot, a.doc).encode('utf-8')
        with open(q, 'xb') as f:
            f.write(body)
        code, text = run_codex(codex_cmd(out), body, ROOT)
        t1 = dt.datetime.now().astimezone()
        with open(log, 'xb') as f:
            f.write(text)
        s = text.decode('utf-8', 'replace')
        miss = check_log(s)
        m = re.search(r'^model: (.+)$', s, re.M)
        e = re.search(r'^reasoning effort: (.+)$', s, re.M)
        wp = os.path.join(cdir, 'thoi_gian.csv')
        new = not os.path.exists(wp)
        with open(wp, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f, lineterminator='\n')
            if new:
                w.writerow(WAIT_COLS)
            w.writerow([os.path.basename(q), t0.isoformat(timespec='seconds'), t1.isoformat(timespec='seconds'),
                        f'{(t1 - t0).total_seconds():.1f}', code, m.group(1).strip() if m else '',
                        e.group(1).strip() if e else '', ' '.join(a.luot), os.environ.get('CODEX_HOME', '(mặc định)')])
        if code != 0 or miss or not os.path.isfile(out):
            print(f'Codex không chạy đúng khóa, báo Liam (mã {code}; thiếu {miss}; log {os.path.basename(log)})')
            print('#ket_qua,ma=2')
            return 2
        with open(out, encoding='utf-8', errors='replace') as f:
            sys.stdout.write(f.read())
        print(f'\n(chờ Codex {(t1 - t0).total_seconds():.0f} giây; {os.path.basename(out)})')
        print('#ket_qua,ma=0')
        return 0
    except (Refused, OSError) as exc:
        print(f'TỪ CHỐI: {exc}')
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
