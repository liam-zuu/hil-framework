#!/usr/bin/env python3
"""chep_luot_A: the information fence between the thesis data and the STRA_A mailbox (campaign A, hand tuning).

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_C13_v1.md (sections 1-4, 10; approved by Liam 15/11, CODE_C13 local_8feba1cc:
"Ghim high", "Mỗi file một dòng", "Duyệt, code sau C8", "Duyệt, sửa luôn STRA_A"; code now: Liam 16/11, CODE_C13
local_0fc6525a "giờ làm cái này trước đi"). S05 §2.1, §2.5, §2.6 rule 5.

  python tools/chep_luot_A.py luot   --run-uid S:I --tuning-log F --freeze F --esp32 F --score F --rows F
                                     --frames F --track F [--segments F] [--ghi-chu-file F | --ghi-chu TEXT] [--thay]
  python tools/chep_luot_A.py them   --loai {spec,s03_quet,mien_A,tai_lieu} --nguon F [F] --ly-do TEXT [--thay]
  python tools/chep_luot_A.py mien   --s03-run DIR --tuning-log F --out DIR [--thay]
  python tools/chep_luot_A.py ke-khai --out F [--nhiem "LUC|FILE|NOI_DUNG"] ...
  python tools/chep_luot_A.py kiem-c13 --spec-commit SHA [--ke-khai F]
  python tools/chep_luot_A.py cai-hoi-codex
  python tools/chep_luot_A.py dong-goi --dich DIR --freeze F [--transcript F ...]

Every rule refuses the whole command (nothing half copied), prints the reason, exit 2. Last line of every command:
#ket_qua,ma=0 / ma=2. The script blocks by origin (run, path, file type); it does not understand content. Keys that
are locked at C8 (score.json fields, error-series columns, file names of a run) are the constants below.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'CLAUDE_THESIS'))   # worktree D:\HIL_WT\x\tools
if not os.path.isdir(os.path.join(THESIS, '06_SPECS')):
    THESIS = os.path.normpath(os.path.join(HERE, '..', '..', '..'))                   # repo inside 03_CODE

# ── constants (one place; C8 dependent keys marked) ───────────────────────────────────────────────────────────────
STRA_A = r'D:\STRA_A'
HOP_THU_DIR = 'HOP_THU'
LUOT_DIR = 'luot'
CAMPAIGNS_A = ('A-PID', 'A-LADRC')
PLATFORM_A = 'HW'
STAGES_A = ('screen', 'confirm', 'explore')
FORBIDDEN_PARTS = ('sealed', 'sealed_test', 'o1_feas', 's03', 's04', 'logs_sweep', 'final')   # luot, tai_lieu
SEALED_PARTS = ('sealed', 'sealed_test', 'logs_sweep', 'o1_feas')                             # mien, s03_quet
EXTS = ('.csv', '.json', '.txt', '.md')
MAX_BYTES = 20 * 1024 * 1024            # [ĐỀ XUẤT], set at C8 from real log sizes
NOTE_MAX_BYTES = 20 * 1024
SCORE_ESP32_SHA = 'esp32_sha256'        # C8: c7_score.py keys
SCORE_FRAMES_SHA = 'frames_sha256'
SCORE_OUTCOME = 'outcome'
SCORE_VERSION_KEYS = ('version', 'scorer_version', 'gate_version')
ROWS_NEED = ('e_perp', 'e_theta_lane', 'e_theta_cmd')       # C8: c7_score.py --rows columns
ROLES = ('log_esp32', 'log_segments', 'score', 'sai_so', 'quy_dao', 'theo_doi', 'ghi_chu')
SPECS = ('06_SPECS/2026-10-28_Spec_S05_So_quy_trinh_RQ3.md', '06_SPECS/2026-11-04_De_xuat_O1_nhiem_vu_san_v11.md')
S03_HISTORY = ('2026-11-10_run02',)     # S05 §3.2: history, never the A domain
CTRL_MAP = {'PID': 'PID', 'ADRC': 'LADRC'}
N_GAINS = 7                             # S05 §2.4
OMEGA_O = ('omega_o_xy', 'omega_o_wz')
OMEGA_O_MAX = 48.0
DEFAULT_ROWS = ('(default check)', '(default)')
LEVEL_COLS = ('a', 'b', 'b_n', 'L_nguon', 'H_nguon')        # S03 v8.3 §3.1 (CODE_FEAS 15/11)
GAIN_COL = {'kp_xy': 'g_kp_xy', 'ki_xy': 'g_ki_xy', 'kp_wz': 'g_kp_wz', 'ki_wz': 'g_ki_wz', 'pos_kp': 'g_pos_kp',
            'pos_ki': 'g_pos_ki', 'k_psi': 'g_k_psi', 'omega_c_xy': 'g_wc_xy', 'omega_o_xy': 'g_wo_xy',
            'omega_c_wz': 'g_wc_wz', 'omega_o_wz': 'g_wo_wz'}
MIEN_COLS = ['controller_s03', 'controller', 'gain', 'lo', 'hi', 'n_diem', 'co_diem_0', 'nguon_hi']
# design section 6 (sha256sum 15/11): the lock files; hoi_codex.py checks the same Codex one
SHA_CODEX_CONFIG = '748bf0c8c5aeb08fd222337172ee939e6bc83ea1f3caf1ff5bcacbca693c3cf9'
SHA_CLAUDE_SETTINGS = '9778ccbff6db1f22a2c2e62747d1aa4a268d7ddf98af56b545a79af63bc9f055'
RAO_CLAUDE = '2026-11-14_Kiem_rao_cach_ly.md'
RAO_CODEX = os.path.join('_codex_test', 'ket_qua_2026-11-14.txt')
RAO_RUNS = ('1', '2', '6')              # kiem_quyen.ps1: these run, every other test is refused
RAO_VERSIONS = dict(claude='2.1.287', codex='0.163.0')
HOI_CODEX_SRC = os.path.join(HERE, 'stra_a', 'hoi_codex.py')
HOI_CODEX_DST = 'hoi_codex.py'


class Refused(Exception):
    pass


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────

def now():
    return dt.datetime.now().astimezone()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def real(path):
    return os.path.normcase(os.path.realpath(path))


def inside(path, root):
    p, r = real(path), real(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def parts_of(path):
    return [x.lower() for x in re.split(r'[\\/]+', os.path.realpath(path)) if x]


def check_parts(path, banned, what):
    bad = [x for x in parts_of(path) if x in banned]
    if bad:
        raise Refused(f'{what}: path has {bad[0]!r} (S05 §2.1 fence)')


def check_file(path, what, exts=EXTS, max_bytes=None):
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    if not os.path.isfile(path):
        raise Refused(f'{what}: no file {path}')
    if os.path.islink(path):
        raise Refused(f'{what}: symlink refused')
    ext = os.path.splitext(path)[1].lower()
    if ext not in exts:
        raise Refused(f'{what}: extension {ext!r} not allowed (only {", ".join(exts)}; no video, no image)')
    if os.path.getsize(path) > max_bytes:
        raise Refused(f'{what}: {os.path.getsize(path)} bytes > {max_bytes}')


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise Refused(f'{path}: no header row')
        return list(r), list(r.fieldnames)


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (ValueError, UnicodeDecodeError) as exc:
        raise Refused(f'{path}: not JSON ({exc})')


def git(root, *args):
    r = subprocess.run(['git', '-C', root, *args], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise Refused(f'git {" ".join(args)}: {r.stderr.strip()}')
    return r.stdout.strip()


def rel_thesis(path):
    """Path relative to the thesis repo (forward slashes); refused when the file is not in it."""
    try:
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(THESIS)).replace('\\', '/')
    except ValueError:
        rel = '..'
    if rel == '..' or rel.startswith('../'):
        raise Refused(f'{path}: not inside the thesis folder {THESIS}')
    return rel


def thesis_commit(path):
    """Commit of the thesis repo for the README column: last commit of the file, or 'chưa commit (HEAD x)'."""
    try:
        rel = rel_thesis(path)
        dirty = git(THESIS, 'status', '--porcelain', '--', rel)
        last = git(THESIS, 'log', '-1', '--format=%h', '--', rel)
        head = git(THESIS, 'rev-parse', '--short', 'HEAD')
    except (Refused, OSError, subprocess.SubprocessError):
        return 'ngoài repo'
    if dirty or not last:
        return f'chưa commit (HEAD {head})'
    return last


def script_commit():
    try:
        c = git(HERE, 'rev-parse', '--short', 'HEAD')
        d = git(HERE, 'status', '--porcelain', '--', os.path.basename(__file__))
        return c + ('-dirty' if d else '')
    except (Refused, OSError, subprocess.SubprocessError):
        return ''


def stamp():
    return now().strftime('%Y%m%d-%H%M')


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_-]', '-', s)


class Lock:
    """One copy at a time (lock file in %TEMP%, never in the mailbox)."""
    def __init__(self, name='chep_luot_A.lock'):
        self.path = os.path.join(tempfile.gettempdir(), name)

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise Refused(f'another chep_luot_A is running (lock {self.path}); remove it only if none is')
        os.write(self.fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *a):
        os.close(self.fd)
        os.remove(self.path)


# ── README table (one row per file, S05 §2.6 rule 5; Q2) ──────────────────────────────────────────────────────────

README_HEAD = '| File | Nguồn | Commit repo luận văn | SHA-256 (16 ký tự đầu) | Ngày chép |'


def readme_rows(text):
    """(index after the last row of the file table, rows as lists of cells)."""
    lines = text.split('\n')
    try:
        h = next(i for i, x in enumerate(lines) if x.strip() == README_HEAD)
    except StopIteration:
        raise Refused('README.md: file table header not found')
    i, rows = h + 2, []
    while i < len(lines) and lines[i].startswith('|'):
        rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
        i += 1
    return i, rows, lines


def readme_append(hop, new_rows):
    """Insert rows right after the last table row; every other byte kept (checked); temp file then replace."""
    p = os.path.join(hop, 'README.md')
    with open(p, 'rb') as f:
        old = f.read()
    nl = b'\r\n' if b'\r\n' in old else b'\n'
    end, rows, lines = readme_rows(old.decode('utf-8').replace('\r\n', '\n'))
    k = sum(len(x.encode('utf-8')) + len(nl) for x in lines[:end])     # byte offset after the last table row
    if end == len(lines) or k > len(old):
        raise Refused('README.md: table must be followed by a line')
    if any(ch in str(c) for r in new_rows for c in r for ch in ('\n', '\r', '|', '`')):
        raise Refused('README.md: a cell has a line break, | or `')
    add = [f'| `{a}` | {b} | `{c}` | `{d[:16]}` | {e} |' for a, b, c, d, e in new_rows]
    ins = b''.join(x.encode('utf-8') + nl for x in add)
    out = old[:k] + ins + old[k:]
    _, rows2, _ = readme_rows(out.decode('utf-8').replace('\r\n', '\n'))
    if out[:k] != old[:k] or out[k + len(ins):] != old[k:] or len(rows2) != len(rows) + len(add):
        raise Refused('README.md: the new text is not the old text plus the inserted rows')
    tmp = p + '.tmp_chep'
    with open(tmp, 'wb') as f:
        f.write(out)
    os.replace(tmp, p)


# ── luot ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def frozen_campaigns(freeze):
    rows, cols = read_csv(freeze)
    if 'campaign' not in cols:
        raise Refused(f'{freeze}: no campaign column')
    return {r['campaign'] for r in rows}


def tuning_row(tuning_log, run_uid):
    rows, cols = read_csv(tuning_log)
    for c in ('run_uid', 'campaign', 'stage', 'platform', 'results_dir', 'gate_json_sha256', 'run_outcome'):
        if c not in cols:
            raise Refused(f'{tuning_log}: no column {c}')
    hit = [r for r in rows if r['run_uid'] == run_uid]
    if len(hit) != 1:
        raise Refused(f'run_uid {run_uid}: {len(hit)} rows in the tuning log (need exactly one)')
    return hit[0]


def check_run(row, frozen):
    if row['campaign'] not in CAMPAIGNS_A:
        raise Refused(f'campaign {row["campaign"]!r}: not a campaign A run')
    if row['platform'] != PLATFORM_A:
        raise Refused(f'platform {row["platform"]!r}: campaign A runs are {PLATFORM_A}')
    if row['stage'] not in STAGES_A:
        raise Refused(f'stage {row["stage"]!r}: only {STAGES_A} (final check is barred, S05 §2.1 rule 4)')
    if row['campaign'] in frozen:
        raise Refused(f'{row["campaign"]} is frozen (S05_freeze.csv): not a tune run any more')


def results_root(row, tuning_log):
    d = row['results_dir']
    if not d:
        raise Refused('the run has no results_dir')
    if not os.path.isabs(d):
        d = os.path.join(THESIS, d)
    return d


def mien_check(row, hop):
    p = os.path.join(hop, 'mien_A.csv')
    if not os.path.isfile(p):
        return None, 'chưa có HOP_THU\\mien_A.csv'
    rows, _ = read_csv(p)
    ctrl = row['campaign'].split('-', 1)[1]
    out = []
    for r in rows:
        if r['controller'] != ctrl:
            continue
        col = GAIN_COL.get(r['gain'])
        v = row.get(col, '') if col else ''
        if v in ('', None):
            continue
        x = float(v)
        if not (float(r['lo']) <= x <= float(r['hi'])):
            out.append(dict(gain=r['gain'], value=x, lo=float(r['lo']), hi=float(r['hi'])))
    return out, None


def luot(a):
    hop = a.hop_thu
    row = tuning_row(a.tuning_log, a.run_uid)
    check_run(row, frozen_campaigns(a.freeze))
    root = results_root(row, a.tuning_log)
    files = dict(log_esp32=a.esp32, score=a.score, sai_so=a.rows, quy_dao=a.frames, theo_doi=a.track)
    if a.segments:
        files['log_segments'] = a.segments
    for role, p in files.items():
        check_parts(p, FORBIDDEN_PARTS, role)
        if not inside(p, root):
            raise Refused(f'{role}: {p} is not inside results_dir of the run')
        check_file(p, role)
    if a.segments and os.path.basename(a.segments) != os.path.basename(a.esp32) + '.segments.csv':
        raise Refused('log_segments: name must be <log>.segments.csv of log_esp32')
    if os.path.dirname(real(a.track)) != os.path.dirname(real(a.frames)):
        raise Refused('theo_doi: track.json must sit next to frames.csv')
    sc = read_json(a.score)
    if not isinstance(sc, dict):
        raise Refused('score: not a JSON object')
    sh = {r: sha256(p) for r, p in files.items()}
    if sh['score'] != row['gate_json_sha256']:
        raise Refused('score: SHA-256 differs from gate_json_sha256 of the tuning-log row')
    if sc.get(SCORE_ESP32_SHA) != sh['log_esp32']:
        raise Refused(f'log_esp32: SHA-256 differs from {SCORE_ESP32_SHA} in score.json')
    if sc.get(SCORE_FRAMES_SHA) != sh['quy_dao']:
        raise Refused(f'quy_dao: SHA-256 differs from {SCORE_FRAMES_SHA} in score.json')
    if sc.get(SCORE_OUTCOME) != row['run_outcome']:
        raise Refused(f'score: {SCORE_OUTCOME} {sc.get(SCORE_OUTCOME)!r} differs from run_outcome {row["run_outcome"]!r}')
    read_json(a.track)
    for role in ('log_esp32', 'quy_dao', 'log_segments'):
        if role in files:
            read_csv(files[role])
    _, cols = read_csv(a.rows)
    miss = [c for c in ROWS_NEED if c not in cols]
    if miss:
        raise Refused(f'sai_so: no column {miss}')
    note = None
    if a.ghi_chu_file:
        check_parts(a.ghi_chu_file, FORBIDDEN_PARTS, 'ghi_chu')
        check_file(a.ghi_chu_file, 'ghi_chu', exts=('.txt', '.md'), max_bytes=NOTE_MAX_BYTES)
        with open(a.ghi_chu_file, 'rb') as f:
            note = f.read()
        note.decode('utf-8')
        note_name = os.path.basename(a.ghi_chu_file)
    elif a.ghi_chu is not None:
        note = a.ghi_chu.encode('utf-8')
        if len(note) > NOTE_MAX_BYTES:
            raise Refused('ghi_chu: > 20 KB')
        note_name = 'ghi_chu.txt'
    ma = f'{safe_name(row["campaign"])}__{safe_name(row.get("session_id", "") or row["run_uid"].split(":")[0])}__' \
         f'{int(row.get("iter_no") or row["run_uid"].split(":")[-1]):03d}'
    base = os.path.join(hop, LUOT_DIR)
    dst = os.path.join(base, ma)
    out_of, note_m = mien_check(row, hop)
    with Lock():
        if os.path.exists(dst) and not a.thay:
            raise Refused(f'{LUOT_DIR}/{ma} exists: --thay to copy again (old folder kept)')
        os.makedirs(base, exist_ok=True)
        tmp = os.path.join(base, '.dang_chep_' + ma)
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        try:
            os.makedirs(tmp)
            meta, rows_add, t = {}, [], now()
            names = set()
            for role in ROLES:
                if role == 'ghi_chu':
                    if note is None:
                        continue
                    name = note_name
                    with open(os.path.join(tmp, name), 'xb') as f:
                        f.write(note)
                    src_sha, src = sha256_bytes(note), None
                else:
                    if role not in files:
                        continue
                    name = os.path.basename(files[role])
                    shutil.copyfile(files[role], os.path.join(tmp, name))
                    src_sha, src = sh[role], files[role]
                if name in names:
                    raise Refused(f'two roles with the file name {name}')
                names.add(name)
                got = sha256(os.path.join(tmp, name))
                if got != src_sha:
                    raise Refused(f'{role}: copy SHA-256 differs from the source')
                meta[role] = dict(ten=name, sha256=got, co=os.path.getsize(os.path.join(tmp, name)))
                ver = next((str(sc[k]) for k in SCORE_VERSION_KEYS if k in sc), '') if role == 'score' else ''
                rows_add.append((f'{LUOT_DIR}/{ma}/{name}',
                                 f'lượt {row["run_uid"]}, {role}' + (f' (bộ chấm {ver})' if ver else ''),
                                 thesis_commit(src) if src else 'ghi chú Liam', got, t.strftime('%d/%m/%Y %H:%M')))
            info = dict(run_uid=row['run_uid'], campaign=row['campaign'], stage=row['stage'],
                        speed_mps=row.get('speed_mps', ''), gain_set_id=row.get('gain_set_id', ''), files=meta,
                        luc_chep=t.isoformat(timespec='seconds'), script=script_commit(),
                        ngoai_mien=out_of, ghi_chu_mien=note_m)
            with open(os.path.join(tmp, 'luot.json'), 'x', encoding='utf-8') as f:
                f.write(json.dumps(info, ensure_ascii=False, indent=1, allow_nan=False))
            js = sha256(os.path.join(tmp, 'luot.json'))
            rows_add.append((f'{LUOT_DIR}/{ma}/luot.json', f'lượt {row["run_uid"]}, kê khai lượt (chep_luot_A)',
                             f'script {script_commit() or "?"}', js, t.strftime('%d/%m/%Y %H:%M')))
            if os.path.exists(dst):
                os.rename(dst, f'{dst}__cu_{stamp()}')
            os.rename(tmp, dst)
        except BaseException:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            raise
        readme_append(hop, rows_add)
    for r in rows_add:
        print(f'{r[0]}  {r[3][:16]}')
    if out_of:
        print(f'CẢNH BÁO: gain ngoài miền A: {out_of}')
    return 0


# ── them ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _put(hop, src, name, source_text, commit, thay):
    dst = os.path.join(hop, name)
    with Lock():
        if os.path.exists(dst):
            if not thay:
                raise Refused(f'{name} is in the mailbox: --thay (old copy renamed __cu_, its row kept)')
            b, e = os.path.splitext(name)
            os.rename(dst, os.path.join(hop, f'{b}__cu_{stamp()}{e}'))
        tmp = dst + '.tmp_chep'
        shutil.copyfile(src, tmp)
        if sha256(tmp) != sha256(src):
            os.remove(tmp)
            raise Refused(f'{name}: copy SHA-256 differs')
        os.replace(tmp, dst)
        readme_append(hop, [(name, source_text, commit, sha256(dst), now().strftime('%d/%m/%Y %H:%M'))])
    print(f'{name}  {sha256(dst)[:16]}')


def them(a):
    hop = a.hop_thu
    if a.loai == 'spec':
        src = a.nguon[0]
        rel = rel_thesis(src)
        if rel not in SPECS:
            raise Refused(f'spec: only {SPECS}')
        if git(THESIS, 'status', '--porcelain', '--', rel):
            raise Refused(f'{rel}: not clean in git (commit first)')
        check_file(src, 'spec', exts=('.md',))
        _put(hop, src, os.path.basename(src), f'spec ({a.ly_do})', git(THESIS, 'log', '-1', '--format=%h', '--', rel),
             a.thay)
    elif a.loai == 's03_quet':
        if len(a.nguon) != 1 or not os.path.isdir(a.nguon[0]):
            raise Refused('s03_quet: --nguon is ONE S03 run folder')
        run = a.nguon[0]
        s03_run_ok(run)
        for n in ('sweep.csv', 'levels.csv'):
            p = os.path.join(run, n)
            check_file(p, n, exts=('.csv',))
            _put(hop, p, n, f'S03 {os.path.basename(os.path.normpath(run))}, quét công khai ({a.ly_do})',
                 thesis_commit(p), a.thay)
    elif a.loai == 'mien_A':
        src = a.nguon[0]
        if os.path.basename(src) != 'mien_A.csv':
            raise Refused('mien_A: only a mien_A.csv written by the mien command')
        js = read_json(os.path.join(os.path.dirname(src), 'mien_A.json'))
        run = os.path.join(THESIS, js['s03_run'])
        rows = mien_rows(run)
        with open(src, 'rb') as f:
            got = f.read().replace(b'\r\n', b'\n')                          # git autocrlf may give CRLF
        if _mien_csv_bytes(rows) != got:
            raise Refused('mien_A.csv differs from the domain computed again from sweep.csv (hand edit?)')
        _put(hop, src, 'mien_A.csv', f'miền A tính từ S03 {js["s03_run"]} ({a.ly_do})', thesis_commit(src), a.thay)
    else:                                                                  # tai_lieu
        src = a.nguon[0]
        check_parts(src, FORBIDDEN_PARTS, 'tai_lieu')
        check_file(src, 'tai_lieu', exts=('.md', '.csv', '.json'))
        if os.path.basename(src) in ('README.md', 'mien_A.csv', 'sweep.csv', 'levels.csv'):
            raise Refused(f'tai_lieu: name {os.path.basename(src)} is reserved')
        _put(hop, src, os.path.basename(src), f'tài liệu: {a.ly_do}', thesis_commit(src), a.thay)
    return 0


# ── mien (section 10) ─────────────────────────────────────────────────────────────────────────────────────────────

def s03_run_ok(run):
    check_parts(run, SEALED_PARTS, 's03 run')
    if os.path.basename(os.path.normpath(run)) in S03_HISTORY:
        raise Refused(f'{run}: history run (S05 §3.2), never the A domain')
    with open(os.path.join(run, 'meta.yaml'), encoding='utf-8') as f:
        m = yaml.safe_load(f)
    if not isinstance(m, dict):
        raise Refused('meta.yaml: not a mapping')
    if m.get('trial') is not False or m.get('tool_check') is True:
        raise Refused('meta.yaml: tool check run (trial / tool_check), not a main S03 run')
    dirty = [k for k, v in _walk(m) if k.endswith('dirty') and v is not False]
    if dirty:
        raise Refused(f'meta.yaml: dirty build {dirty}')
    return m


def _walk(d, pre=''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f'{pre}.{k}' if pre else str(k))
    else:
        yield pre, d


def _f(x, what):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise Refused(f'{what}: {x!r} not a number')
    if not math.isfinite(v):
        raise Refused(f'{what}: not finite')
    return v


def _f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def mien_rows(run):
    """Opens only sweep.csv, levels.csv, meta.yaml at the root of the run (by name). The A domain per gain."""
    s03_run_ok(run)
    sw, scol = read_csv(os.path.join(run, 'sweep.csv'))
    lv, lcol = read_csv(os.path.join(run, 'levels.csv'))
    miss = [c for c in LEVEL_COLS if c not in lcol] + [c for c in ('luoi',) if c not in scol]
    if miss:
        raise Refused(f'S03 v8.3 columns missing {miss}: run made by a tool before v8.3, a, b not guessed')
    speeds = sorted({r['v'] for r in sw if r['factor'] not in DEFAULT_ROWS})
    lev = {(r['controller'], r['factor']): r for r in lv}
    grid = {}
    for r in sw:
        if r['factor'] in DEFAULT_ROWS:
            continue
        if r['controller'] not in CTRL_MAP:
            raise Refused(f'controller {r["controller"]!r}: unknown')
        x = _f(r['value'], f'{r["controller"]} {r["factor"]} value')
        if x < 0:
            raise Refused(f'{r["controller"]} {r["factor"]}: value {x} < 0')
        grid.setdefault((r['controller'], r['factor']), {}).setdefault(r['value'], []).append(r)
    out = []
    for c in sorted({k[0] for k in grid}):
        fs = sorted(f for (cc, f) in grid if cc == c)
        if len(fs) != N_GAINS:
            raise Refused(f'{c}: {len(fs)} gains in sweep.csv, need {N_GAINS}')
        for fac in fs:
            pts = grid[(c, fac)]
            for v, rs in pts.items():
                if sorted(r['v'] for r in rs) != speeds:
                    raise Refused(f'{c} {fac} = {v}: rows at speeds {sorted(r["v"] for r in rs)}, need {speeds}')
            if (c, fac) not in lev:
                raise Refused(f'{c} {fac}: no row in levels.csv')
            L = lev[(c, fac)]
            vals = sorted(_f(v, 'value') for v in pts)
            lo, hi = vals[0], vals[-1]
            has0 = lo == 0.0
            a_, b_ = _f(L['a'], f'{c} {fac} a'), _f(L['b'], f'{c} {fac} b')
            bn = _f(L['b_n'], f'{c} {fac} b_n') if L['b_n'] not in ('', None) else None
            if not has0 and lo != a_:
                raise Refused(f'{c} {fac}: lowest grid point {lo} is not a = {a_}')
            want_hi = bn if bn is not None else b_
            if hi != want_hi:
                raise Refused(f'{c} {fac}: highest grid point {hi} is not {"b_n" if bn is not None else "b"} = {want_hi}')
            noi = any(r.get('luoi') == 'noi' for rs in pts.values() for r in rs)
            if (bn is not None) != noi:
                raise Refused(f'{c} {fac}: b_n and luoi = noi points disagree')
            if fac in OMEGA_O:
                if bn is not None or noi:
                    raise Refused(f'{c} {fac}: omega_o grid is cut, never extended')
                if hi > OMEGA_O_MAX or _f32(hi) > OMEGA_O_MAX:
                    raise Refused(f'{c} {fac}: hi {hi!r} > {OMEGA_O_MAX} (binary64 or float32)')
            d = _f(L['default'], f'{c} {fac} default')
            Lo, Hi = _f(L['L'], f'{c} {fac} L'), _f(L['H'], f'{c} {fac} H')
            if not (lo <= d <= hi):
                raise Refused(f'{c} {fac}: default {d} outside [{lo}, {hi}]')
            if not (lo <= Lo <= Hi <= hi):
                raise Refused(f'{c} {fac}: [L, H] = [{Lo}, {Hi}] not inside [{lo}, {hi}]')
            src = 'b_n' if bn is not None else ('tran_omega_o' if fac in OMEGA_O and hi == OMEGA_O_MAX else 'b')
            out.append(dict(controller_s03=c, controller=CTRL_MAP[c], gain=fac, lo=repr(lo), hi=repr(hi),
                            n_diem=str(len(vals)), co_diem_0='true' if has0 else 'false', nguon_hi=src))
    if not out:
        raise Refused('sweep.csv: no gain rows')
    return out


def _mien_csv_bytes(rows):
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=MIEN_COLS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode('utf-8')


def mien(a):
    rows, _ = read_csv(a.tuning_log)
    if any(r.get('campaign') in CAMPAIGNS_A for r in rows):
        raise Refused('a campaign A run is in the tuning log: the A domain is locked before the first run')
    out_rows = mien_rows(a.s03_run)
    os.makedirs(a.out, exist_ok=True)
    csv_p, js_p = os.path.join(a.out, 'mien_A.csv'), os.path.join(a.out, 'mien_A.json')
    if os.path.exists(csv_p) or os.path.exists(js_p):
        if not a.thay:
            raise Refused(f'{csv_p} exists: --thay (no A run yet; old copy kept)')
        st = stamp()
        for p in (csv_p, js_p):
            if os.path.exists(p):
                b, e = os.path.splitext(p)
                os.rename(p, f'{b}__cu_{st}{e}')
    run = a.s03_run
    m = s03_run_ok(run)
    with open(csv_p, 'xb') as f:
        f.write(_mien_csv_bytes(out_rows))
    info = dict(s03_run=rel_thesis(run),
                s03_commit=(m.get('script') or {}).get('commit', ''),
                sha256={n: sha256(os.path.join(run, n)) for n in ('sweep.csv', 'levels.csv', 'meta.yaml')},
                mien_A_csv_sha256=sha256(csv_p), script=script_commit(), luc_tinh=now().isoformat(timespec='seconds'),
                spec='S05 v3.29 §2.5, S03 v8.3 §3.1')
    with open(js_p, 'x', encoding='utf-8') as f:
        f.write(json.dumps(info, ensure_ascii=False, indent=1))
    for r in out_rows:
        print(f'{r["controller"]:6s} {r["gain"]:12s} [{r["lo"]}, {r["hi"]}]  {r["nguon_hi"]}')
    print(f'mien_A: {csv_p} {info["mien_A_csv_sha256"][:16]}')
    return 0


# ── ke-khai, kiem-c13 ─────────────────────────────────────────────────────────────────────────────────────────────

def box_check(hop):
    """Every file in the mailbox (but README.md) has its latest row with the same name and SHA; __cu_ files match
    an older row of their name. Returns (rows, problems)."""
    with open(os.path.join(hop, 'README.md'), encoding='utf-8') as f:
        _, rows, _ = readme_rows(f.read().replace('\r\n', '\n'))
    by = {}
    for r in rows:
        by.setdefault(r[0].strip('`'), []).append(r[3].strip('`'))
    bad = []
    for dp, dn, fn in os.walk(hop):
        dn[:] = [d for d in dn if not d.startswith('.dang_chep_')]
        for n in fn:
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, hop).replace('\\', '/')
            if rel == 'README.md':
                continue
            s = sha256(p)[:16]
            m = re.match(r'^(.*)__cu_\d{8}-\d{4}(\.[^/]*)?$', rel)
            parts = rel.split('/')
            cu_dir = len(parts) >= 3 and re.search(r'__cu_\d{8}-\d{4}$', parts[1])
            if cu_dir:
                key = '/'.join([parts[0], re.sub(r'__cu_\d{8}-\d{4}$', '', parts[1])] + parts[2:])
                if s not in by.get(key, []):
                    bad.append(f'{rel}: no row of {key} with this SHA')
            elif m:
                key = m.group(1) + (m.group(2) or '')
                if s not in by.get(key, [])[:-1]:
                    bad.append(f'{rel}: no older row of {key} with this SHA')
            elif rel not in by:
                bad.append(f'{rel}: file with no row (strange file)')
            elif by[rel][-1] != s:
                bad.append(f'{rel}: SHA {s} differs from its latest row {by[rel][-1]}')
    return rows, bad


def rao_check(root):
    bad = []
    pc = os.path.join(root, RAO_CLAUDE)
    px = os.path.join(root, RAO_CODEX)
    if not os.path.isfile(pc):
        bad.append(f'{RAO_CLAUDE}: missing')
    if not os.path.isfile(px):
        bad.append(f'{RAO_CODEX}: missing')
    else:
        with open(px, encoding='utf-8', errors='replace') as f:
            for line in f:
                m = re.match(r'^(\d+[a-z]?) .*?=> (.*)$', line.rstrip('\n'))
                if not m:
                    continue
                k, res = m.group(1), m.group(2)
                ok_run = k in RAO_RUNS
                refused = re.search(r'denied|Access|Cannot find|loi mang|Traceback', res) is not None
                if ok_run == refused:
                    bad.append(f'Codex fence test {k}: {"refused" if refused else "ran"}, expected '
                               f'{"to run" if ok_run else "a refusal"}')
    for n, want in (('.codex/config.toml', SHA_CODEX_CONFIG), ('.claude/settings.json', SHA_CLAUDE_SETTINGS)):
        p = os.path.join(root, n)
        if not os.path.isfile(p) or sha256(p) != want:
            bad.append(f'{n}: SHA differs from the mark (design section 6)')
    return dict(claude=dict(file=RAO_CLAUDE, sha256=sha256(pc) if os.path.isfile(pc) else None),
                codex=dict(file=RAO_CODEX.replace('\\', '/'), sha256=sha256(px) if os.path.isfile(px) else None)), bad


def _tool_version(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        m = re.search(r'(\d+\.\d+\.\d+)', r.stdout + r.stderr)
        return m.group(1) if m else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def nhat_ky(root, prev):
    """Files of NHAT_KY and CODEX: name, size, SHA; append-only against the last declaration prev."""
    out, bad = {}, []
    for d in ('NHAT_KY', 'CODEX'):
        base = os.path.join(root, d)
        for dp, _, fn in os.walk(base):
            for n in fn:
                p = os.path.join(dp, n)
                rel = os.path.relpath(p, root).replace('\\', '/')
                out[rel] = dict(co=os.path.getsize(p), sha256=sha256(p))
    for rel, o in (prev or {}).items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            bad.append(f'{rel}: gone (append-only)')
            continue
        with open(p, 'rb') as f:
            head = f.read(o['co'])
        if len(head) < o['co'] or sha256_bytes(head) != o['sha256']:
            bad.append(f'{rel}: the first {o["co"]} bytes changed (append-only)')
    return out, bad


def nhiem(root, extra):
    ev = []
    for e in extra or []:
        x = e.split('|', 2)
        if len(x) != 3:
            raise Refused('--nhiem "LUC|FILE|NOI_DUNG"')
        ev.append(dict(luc=x[0], file=x[1], noi_dung=x[2], nguon='CODE'))
    d = os.path.join(root, 'NHAT_KY')
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if n.endswith('.md'):
            with open(os.path.join(d, n), encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f, 1):
                    if line.startswith('NHIEM:'):
                        ev.append(dict(luc=f'{n}:{i}', file=n, noi_dung=line[6:].strip(), nguon='NHAT_KY'))
    return dict(co=bool(ev), su_kien=ev)


def ke_khai(a):
    root = a.stra_a
    hop = os.path.join(root, HOP_THU_DIR)
    rows, bad = box_check(hop)
    rao, bad2 = rao_check(root)
    prev = None
    if a.truoc:
        with open(a.truoc, encoding='utf-8') as f:
            prev = (yaml.safe_load(f) or {}).get('nhat_ky')
    nk, bad3 = nhat_ky(root, prev)
    hc = os.path.join(root, HOI_CODEX_DST)
    d = dict(luc=now().isoformat(timespec='seconds'), script=script_commit(),
             hop_thu=dict(readme_sha256=sha256(os.path.join(hop, 'README.md')),
                          dong=[dict(file=r[0].strip('`'), nguon=r[1], commit=r[2].strip('`'), sha16=r[3].strip('`'),
                                     ngay=r[4]) for r in rows]),
             tro_ly_claude=dict(phien=a.phien or [], claude_code=_tool_version(['claude', '--version'])),
             tro_ly_codex=dict(cli=_tool_version(['codex', '--version']), lan_goi=codex_calls(root)),
             rao=rao, nhiem_thong_tin=nhiem(root, a.nhiem), nhat_ky=nk,
             cong_cu=dict(hoi_codex_sha256=sha256(hc) if os.path.isfile(hc) else None),
             loi=bad + bad2 + bad3)
    if os.path.exists(a.out):
        raise Refused(f'{a.out} exists: never overwritten')
    with open(a.out, 'x', encoding='utf-8') as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
    for x in d['loi']:
        print('LỖI:', x)
    print(f'{a.out} {sha256(a.out)[:16]}; {len(rows)} dòng bảng; nhiễm {d["nhiem_thong_tin"]["co"]}')
    return 2 if d['loi'] else 0


def codex_calls(root):
    """model and effort of every Codex call, read from the logs (never from the configuration)."""
    out = []
    d = os.path.join(root, 'CODEX')
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if n.startswith('log_') and n.endswith('.txt'):
            with open(os.path.join(d, n), encoding='utf-8', errors='replace') as f:
                t = f.read(20000)
            m = re.search(r'^model: (.+)$', t, re.M)
            e = re.search(r'^reasoning effort: (.+)$', t, re.M)
            out.append(dict(log=n, model=m.group(1).strip() if m else None, effort=e.group(1).strip() if e else None))
    return out


def kiem_c13(a):
    root = a.stra_a
    hop = os.path.join(root, HOP_THU_DIR)
    res = []
    _, bad = rao_check(root)
    res.append(('1 thử rào có và đúng; 3 SHA khóa', not bad, '; '.join(bad)))
    vc, vx = _tool_version(['claude', '--version']), _tool_version(['codex', '--version'])
    res.append(('2 phiên bản bằng lúc thử rào', (vc, vx) == (RAO_VERSIONS['claude'], RAO_VERSIONS['codex']),
                f'claude {vc}, codex {vx}; khác thì chạy lại thử rào'))
    cfg = os.path.join(os.path.expanduser('~'), '.codex', 'config.toml')
    trust = os.path.isfile(cfg) and "[projects.'D:\\STRA_A']" in open(cfg, encoding='utf-8', errors='replace').read()
    res.append(('3b dòng tin cậy D:\\STRA_A trong ~/.codex/config.toml', trust, 'có' if trust else 'không'))
    _, bb = box_check(hop)
    res.append(('4 hộp thư chỉ có file được phép', not bb, '; '.join(bb)))
    sp = []
    for rel in SPECS:
        n = os.path.basename(rel)
        p = os.path.join(hop, n)
        try:
            want = sha256_bytes(subprocess.run(['git', '-C', THESIS, 'show', f'{a.spec_commit}:{rel}'],
                                               capture_output=True, check=True, timeout=30).stdout)
        except (subprocess.SubprocessError, OSError):
            want = None
        if not os.path.isfile(p) or want is None or sha256(p) != want:
            sp.append(n)
    res.append(('5 spec trong hộp thư là bản khóa', not sp, f'lệch: {sp}' if sp else a.spec_commit))
    hc = os.path.join(root, HOI_CODEX_DST)
    okh = os.path.isfile(hc) and sha256(hc) == sha256(HOI_CODEX_SRC)
    res.append(('6 hoi_codex.py bằng bản gốc', okh, ''))
    res.append(('7 có hop_thu_A.yaml', bool(a.ke_khai and os.path.isfile(a.ke_khai)), a.ke_khai or ''))
    mp = os.path.join(hop, 'mien_A.csv')
    okm = True
    note = ''
    if a.mien_a:
        okm = os.path.isfile(mp) and sha256(mp) == sha256(a.mien_a)
        note = a.mien_a
    res.append(('8 HOP_THU\\mien_A.csv bằng mien_A.csv của meta.yaml A', okm and bool(a.mien_a), note))
    for k, ok, n in res:
        print(f'{"ĐẠT " if ok else "CHƯA"} {k}' + (f' ({n})' if n else ''))
    return 0 if all(ok for _, ok, _ in res) else 2


def cai_hoi_codex(a):
    dst = os.path.join(a.stra_a, HOI_CODEX_DST)
    if os.path.exists(dst):
        os.rename(dst, os.path.join(a.stra_a, f'hoi_codex__cu_{stamp()}.py'))
    shutil.copyfile(HOI_CODEX_SRC, dst)
    if sha256(dst) != sha256(HOI_CODEX_SRC):
        raise Refused('hoi_codex.py: copy SHA differs')
    print(f'{dst} {sha256(dst)}')
    return 0


def dong_goi(a):
    fr = frozen_campaigns(a.freeze)
    miss = [c for c in CAMPAIGNS_A if c not in fr]
    if miss:
        raise Refused(f'{miss} not frozen yet: pack at the end of A only')
    if os.path.exists(a.dich):
        raise Refused(f'{a.dich} exists: never overwritten')
    rows = []
    for d in ('NHAT_KY', 'CODEX', HOP_THU_DIR):
        s = os.path.join(a.stra_a, d)
        shutil.copytree(s, os.path.join(a.dich, d))
    tdir = os.path.join(a.dich, 'transcript')
    for t in a.transcript or []:
        os.makedirs(tdir, exist_ok=True)
        shutil.copyfile(t, os.path.join(tdir, os.path.basename(t)))
    for dp, _, fn in os.walk(a.dich):
        for n in sorted(fn):
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, a.dich).replace('\\', '/')
            if not rel.startswith('transcript/'):
                src = os.path.join(a.stra_a, *rel.split('/'))
                if sha256(src) != sha256(p):
                    raise Refused(f'{rel}: copy SHA differs')
            rows.append((rel, os.path.getsize(p), sha256(p)))
    with open(os.path.join(a.dich, 'SHA256SUMS.csv'), 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['duong_dan', 'co', 'sha256'])
        w.writerows(sorted(rows))
    print(f'{a.dich}: {len(rows)} files')
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stra-a', default=STRA_A, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('luot')
    for k in ('--run-uid', '--tuning-log', '--freeze', '--esp32', '--score', '--rows', '--frames', '--track'):
        p.add_argument(k, required=True)
    p.add_argument('--segments')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--ghi-chu-file')
    g.add_argument('--ghi-chu')
    p.add_argument('--thay', action='store_true')
    p.add_argument('--hop-thu')
    p = sub.add_parser('them')
    p.add_argument('--loai', required=True, choices=('spec', 's03_quet', 'mien_A', 'tai_lieu'))
    p.add_argument('--nguon', required=True, nargs='+')
    p.add_argument('--ly-do', required=True)
    p.add_argument('--thay', action='store_true')
    p.add_argument('--hop-thu')
    p = sub.add_parser('mien')
    p.add_argument('--s03-run', required=True)
    p.add_argument('--tuning-log', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--thay', action='store_true')
    p = sub.add_parser('ke-khai')
    p.add_argument('--out', required=True)
    p.add_argument('--truoc', help='previous hop_thu_A.yaml (append-only check)')
    p.add_argument('--nhiem', action='append')
    p.add_argument('--phien', action='append', help='STRA_A session: id,model,effort')
    p = sub.add_parser('kiem-c13')
    p.add_argument('--spec-commit', required=True)
    p.add_argument('--ke-khai')
    p.add_argument('--mien-a', help='mien_A.csv named in meta.yaml of A')
    sub.add_parser('cai-hoi-codex')
    p = sub.add_parser('dong-goi')
    p.add_argument('--dich', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--transcript', nargs='*')
    a = ap.parse_args(argv)
    if getattr(a, 'hop_thu', None) is None and a.cmd in ('luot', 'them'):
        a.hop_thu = os.path.join(a.stra_a, HOP_THU_DIR)
    try:
        f = dict(luot=luot, them=them, mien=mien, **{'ke-khai': ke_khai, 'kiem-c13': kiem_c13,
                                                     'cai-hoi-codex': cai_hoi_codex, 'dong-goi': dong_goi})[a.cmd]
        ma = f(a)
    except Refused as exc:
        print(f'TỪ CHỐI: {exc}')
        ma = 2
    print(f'#ket_qua,ma={ma}')
    return ma


if __name__ == '__main__':
    sys.exit(main())
