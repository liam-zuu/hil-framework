#!/usr/bin/env python3
"""hoi_codex: STRA_A asks Codex inside D:\\STRA_A (design 03_CODE/docs/2026-11-15_Thiet_ke_CODE_C13_v1.md section 4).

  python D:\\STRA_A\\hoi_codex.py --cau "<câu hỏi>" [--luot <mã lượt> ...] [--doc <file trong D:\\STRA_A> ...]
  python D:\\STRA_A\\hoi_codex.py --thu

Checks before the call (else no call): working folder D:\\STRA_A, SHA of D:\\STRA_A\\.codex\\config.toml = the mark,
CODEX folder is the real D:\\STRA_A\\CODEX (no link), every --luot / --doc inside D:\\STRA_A, the Codex program is
the pinned path (not PATH) with the version of the fence test (SA-1). No argument reaches Codex as such.
Writes CODEX\\hoi_<date>_<n>.md (new file), calls codex exec (gpt-6.1-sol, effort high: Liam 15/11 "Ghim high")
with the answer to a temp file, checks the log HEADER (workdir, sandbox, model, effort, each once; SA-2); wrong or
exit != 0: the answer is NOT printed. The answer is then published under its own new name (SA-21). Appends the wait
time to CODEX\\thoi_gian.csv. The copy in D:\\STRA_A is installed by chep_luot_A.py cai-hoi-codex (SHA checked).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import subprocess
import sys

ROOT = r'D:\STRA_A'
SHA_CODEX_CONFIG = '748bf0c8c5aeb08fd222337172ee939e6bc83ea1f3caf1ff5bcacbca693c3cf9'   # design section 6
# npm install of the CLI version of the fence test (codex-cli 0.163.0, 14/11); the desktop app's codex.exe is another
# version. A new version needs the fence test again (D:\STRA_A\CLAUDE.md), then this path and version.
CODEX_EXE = (r'C:\Users\liam\AppData\Roaming\npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64'
             r'\vendor\x86_64-pc-windows-msvc\bin\codex.exe')
CODEX_VERSION = '0.163.0'
CODEX_SHA256 = '2692c8c7bb2cf8b0aefe638741fc90236f4b824c59578ce84994f9de58200993'   # SA-1, sha256sum 16/11
MODEL = 'gpt-6.1-sol'
EFFORT = 'high'
WAIT_COLS = ['hoi', 'bat_dau', 'ket_thuc', 'giay', 'ma_thoat', 'model', 'effort', 'luot', 'codex_home']
CAU_THU = 'Câu hỏi thử, không tune: liệt kê tên các file trong HOP_THU (không đọc nội dung) và trả lời "thử xong".'
TIMEOUT_S = 3600


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


def codex_version(exe):
    r = subprocess.run([exe, '--version'], capture_output=True, timeout=60)
    m = re.search(rb'(\d+\.\d+\.\d+\S*)', r.stdout + r.stderr)
    return m.group(1).decode() if m else ''


def run_codex(cmd, stdin_bytes, cwd):
    """The only call of Codex (tests replace this function and codex_version). Returns (code, stdout, stderr):
    codex exec writes its progress and header to stderr, the final message to stdout (SA-2: kept apart)."""
    r = subprocess.run(cmd, input=stdin_bytes, capture_output=True, cwd=cwd, timeout=TIMEOUT_S)
    return r.returncode, r.stdout, r.stderr


def codex_cmd(out):
    return [CODEX_EXE, 'exec', '--skip-git-repo-check', '-m', MODEL, '-c', f'model_reasoning_effort={EFFORT}', '-o',
            out, '-']


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
              for p, e in (('hoi', '.md'), ('tra_loi', '.md'), ('log', '.txt'), ('.tam_tra_loi', '.md'))):
        n += 1
    return f'{day}_{n}'


def log_header(text):
    """The CLI header: the lines between the first two '--------' lines, before any prompt or answer (SA-2)."""
    lines = text.replace('\r\n', '\n').split('\n')
    try:
        i = lines.index('--------')
        j = lines.index('--------', i + 1)
    except ValueError:
        return {}
    out = {}
    for x in lines[i + 1:j]:
        k, _, v = x.partition(':')
        out.setdefault(k.strip(), []).append(v.strip())
    return out


def check_log(text):
    h = log_header(text)
    want = dict(workdir=ROOT, sandbox='custom permissions', model=MODEL)
    want['reasoning effort'] = EFFORT
    return [k for k, v in want.items() if h.get(k) != [v] and [x.casefold() for x in h.get(k, [])] != [v.casefold()]]


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
        cdir = os.path.join(ROOT, 'CODEX')
        if not os.path.isdir(cdir) or real(cdir) != os.path.normcase(os.path.abspath(cdir)):
            raise Refused('CODEX không phải thư mục thật trong D:\\STRA_A')
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
        if not os.path.isfile(CODEX_EXE):
            raise Refused(f'không thấy Codex ở đường dẫn ghim {CODEX_EXE}: báo Liam')
        if sha256(CODEX_EXE) != CODEX_SHA256:                                # SA-1: the program itself
            raise Refused('chương trình Codex khác bản đã thử rào (SHA): chạy lại thử rào, báo Liam')
        v = codex_version(CODEX_EXE)
        if v != CODEX_VERSION:
            raise Refused(f'Codex {v or "?"} khác bản đã thử rào {CODEX_VERSION}: chạy lại thử rào, báo Liam')
        t0 = dt.datetime.now().astimezone()
        tag = next_name(cdir, t0.strftime('%Y%m%d'))
        q, out, log, tmp = (os.path.join(cdir, f'{p}_{tag}{e}') for p, e in
                            (('hoi', '.md'), ('tra_loi', '.md'), ('log', '.txt'), ('.tam_tra_loi', '.md')))
        body = build_question(cau, a.luot, a.doc).encode('utf-8')
        with open(q, 'xb') as f:
            f.write(body)
        try:
            code, out_b, err_b = run_codex(codex_cmd(tmp), body, ROOT)
        except subprocess.TimeoutExpired as exc:                             # SA-23
            code, out_b = 124, exc.output or b''
            err_b = (exc.stderr or b'') + f'\n[hoi_codex: quá {TIMEOUT_S} s]'.encode()
        t1 = dt.datetime.now().astimezone()
        with open(log, 'xb') as f:
            f.write(b'=== stderr ===\n' + err_b + b'\n=== stdout ===\n' + out_b)
        s = err_b.decode('utf-8', 'replace')                                 # SA-2: header from stderr only
        miss = check_log(s)
        h = log_header(s)
        wp = os.path.join(cdir, 'thoi_gian.csv')
        new = not os.path.exists(wp)
        with open(wp, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f, lineterminator='\n')
            if new:
                w.writerow(WAIT_COLS)
            w.writerow([os.path.basename(q), t0.isoformat(timespec='seconds'), t1.isoformat(timespec='seconds'),
                        f'{(t1 - t0).total_seconds():.1f}', code, ' / '.join(h.get('model', [])),
                        ' / '.join(h.get('reasoning effort', [])), ' '.join(a.luot),
                        os.environ.get('CODEX_HOME', '(mặc định)')])
        if code != 0 or miss or not os.path.isfile(tmp):
            print(f'Codex không chạy đúng khóa, báo Liam (mã {code}; sai {miss}; log {os.path.basename(log)})')
            print('#ket_qua,ma=2')
            return 2
        os.rename(tmp, out)                                                  # SA-21: fails if the name exists
        with open(out, encoding='utf-8', errors='replace') as f:
            sys.stdout.write(f.read())
        print(f'\n(chờ Codex {(t1 - t0).total_seconds():.0f} giây; {os.path.basename(out)})')
        print('#ket_qua,ma=0')
        return 0
    except (Refused, OSError, subprocess.SubprocessError, UnicodeError) as exc:
        print(f'TỪ CHỐI: {exc}')
        print('#ket_qua,ma=2')
        return 2


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
