#!/usr/bin/env python3
"""chep_luot_A: the information fence between the thesis data and the STRA_A mailbox (campaign A, hand tuning).

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_C13_v1.md (sections 1-4, 10; approved by Liam 15/11, CODE_C13 local_8feba1cc:
"Ghim high", "Mỗi file một dòng", "Duyệt, code sau C8", "Duyệt, sửa luôn STRA_A"; code now: Liam 16/11, CODE_C13
local_0fc6525a "giờ làm cái này trước đi"). S05 §2.1, §2.5, §2.6 rule 5. Codex Sol round 1 (01a1e3b7) SA-1..SA-23,
Liam 16/11 "sửa hết + vòng 2". S03 v8.4 / S05 v3.41 (CODE_FEAS 17/11, Liam "Code + Codex 2 vòng"; design
03_CODE/docs/2026-11-17_Thiet_ke_S03_v84_kiem_ngoai_mien_v1.md section 3): mien needs kiem_ngoai_mien.csv
'đã kiểm đủ' for both controllers. S03 v8.6 §11 item 5 (c2) / S05 v3.48 §2.5: the A domain of a gain is the
whole sweep grid after the cut at the gain-table ceiling, original and extension points, up to b_n
(omega_o: up to b); a point 'cờ' no longer refuses the domain (S03 v8.5 K6).

  python tools/chep_luot_A.py luot   --run-uid S:I --tuning-log F --freeze F --esp32 F --score F --rows F
                                     --frames F --track F [--segments F] [--ghi-chu-file F | --ghi-chu TEXT] [--thay]
  python tools/chep_luot_A.py them   --loai {spec,s03_quet,mien_A,tai_lieu} --nguon F --ly-do TEXT [--thay]
                                     [--tuning-log F (mien_A)]
  python tools/chep_luot_A.py mien   --s03-run DIR --tuning-log F --out DIR [--thay]
  python tools/chep_luot_A.py ke-khai --out F (--truoc F | --lan-dau) [--nhiem "LUC|FILE|NOI_DUNG"] [--phien ...]
  python tools/chep_luot_A.py kiem-c13 --spec-commit SHA --ke-khai F --mien-a F
  python tools/chep_luot_A.py cai-hoi-codex
  python tools/chep_luot_A.py dong-goi --dich DIR --freeze F --ke-khai F [--transcript F ...]

Every rule refuses the whole command, prints the reason, exit 2; a failure after a change rolls the change back
(SA-11, SA-12). Last line of every command: #ket_qua,ma=0 / ma=2. The script blocks by origin (run, path, file
type, content type); it does not understand the meaning of text (free text --ghi-chu, --ly-do: CODE reads it,
design 3.2). Not a defence against someone racing files during a copy (Liam 16/11: out of scope). Keys locked at
C8 (score.json fields, error-series columns, run file layout; binding files to a run by manifest) are constants.
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
RESULTS = ('05_DATA', 'results')
CAMPAIGNS_A = ('A-PID', 'A-LADRC')
PLATFORM_A = 'HW'
STAGES_A = ('screen', 'confirm', 'explore')
FORBIDDEN_PARTS = ('sealed', 'sealed_test', 'o1_feas', 's03', 's04', 'logs_sweep', 'final')   # luot, tai_lieu
SEALED_PARTS = ('sealed', 'sealed_test', 'logs_sweep', 'o1_feas')                             # mien, s03_quet
EXTS = ('.csv', '.json', '.txt', '.md')
MEDIA_MAGIC = (b'\x89PNG', b'\xff\xd8\xff', b'GIF8', b'RIFF', b'\x1aE\xdf\xa3')   # SA2-5: no 'BM' (text "BMS…")
MAX_BYTES = 20 * 1024 * 1024            # [ĐỀ XUẤT], set at C8 from real log sizes
NOTE_MAX_BYTES = 20 * 1024
SCORE_ESP32_SHA = 'esp32_sha256'        # C8: c7_score.py keys
SCORE_FRAMES_SHA = 'frames_sha256'
SCORE_OUTCOME = 'outcome'
SCORE_VERSION_KEYS = ('version', 'scorer_version', 'gate_version')
ROWS_NEED = ('e_perp', 'e_theta_lane', 'e_theta_cmd')       # C8: c7_score.py --rows columns
ROLES = ('log_esp32', 'log_segments', 'score', 'sai_so', 'quy_dao', 'theo_doi', 'ghi_chu')
ROLE_EXT = dict(log_esp32=('.csv',), log_segments=('.csv',), score=('.json',), sai_so=('.csv',), quy_dao=('.csv',),
                theo_doi=('.json',), ghi_chu=('.txt', '.md'))                 # SA2-1: format by role
S03_SPEEDS = (0.2, 0.5, 0.8)            # S03 v8 §2 (O1 v7 3.1): every gain point at these speeds (SA-15)
TUNING_NEED = ('run_uid', 'campaign', 'stage', 'platform', 'results_dir', 'gate_json_sha256', 'run_outcome')
SPECS = ('06_SPECS/2026-10-28_Spec_S05_So_quy_trinh_RQ3.md', '06_SPECS/2026-11-04_De_xuat_O1_nhiem_vu_san_v11.md')
S03_HISTORY = ('2026-11-10_run02',)     # S05 §3.2: history, never the A domain
CTRL_MAP = {'PID': 'PID', 'ADRC': 'LADRC'}
GAINS_S03 = {'PID': {'kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi'},          # S05 §2.4 (SA-14)
             'ADRC': {'omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi'}}
OMEGA_O = ('omega_o_xy', 'omega_o_wz')
OMEGA_O_MAX = 48.0
DEFAULT_ROWS = ('(default check)', '(default)')
LUOI = ('log', 'mac_dinh', '0', 'noi')
LEVEL_COLS = ('a', 'b', 'b_n', 'L_nguon', 'H_nguon')   # S03 v8.3/v8.4 §3.1 (CODE_FEAS); b_n again: S05 v3.48 §2.5
# S03 v8.4 §3.1, S05 v3.41 §2.5 (CODE_FEAS 17/11, design 03_CODE/docs/2026-11-17_Thiet_ke_S03_v84_kiem_ngoai_mien_v1.md
# section 3): the A domain only from a sweep whose out-of-range check is 'đã kiểm đủ' for both controllers; a point
# 'cờ' no longer refuses (S03 v8.5 K6, S05 v3.47)
KIEM_FILE = 'kiem_ngoai_mien.csv'
KIEM_NEED = ('dong', 'controller', 'factor', 'value', 'can', 'can_value', 'ket_qua', 'n_diem_ngoai', 'n_da_kiem',
             'n_chua_kiem', 'n_co', 'n_chua_ket_luan', 'n_khong_co')
KIEM_POINT = ('không cờ', 'chưa kết luận', 'cờ')
KIEM_TOTAL = ('đã kiểm đủ', 'kiểm dở', 'không kiểm')
S03_PUBLIC = ('sweep.csv', 'levels.csv', KIEM_FILE, 'meta.yaml')     # opened by mien, hashed in mien_A.json
GAIN_COL = {'kp_xy': 'g_kp_xy', 'ki_xy': 'g_ki_xy', 'kp_wz': 'g_kp_wz', 'ki_wz': 'g_ki_wz', 'pos_kp': 'g_pos_kp',
            'pos_ki': 'g_pos_ki', 'k_psi': 'g_k_psi', 'omega_c_xy': 'g_wc_xy', 'omega_o_xy': 'g_wo_xy',
            'omega_c_wz': 'g_wc_wz', 'omega_o_wz': 'g_wo_wz'}
MIEN_COLS = ['controller_s03', 'controller', 'gain', 'lo', 'hi', 'n_diem', 'co_diem_0', 'nguon_hi']
RESERVED = ('readme.md', 'mien_a.csv', 'sweep.csv', 'levels.csv', 'luot', 'luot.json')   # SA-3, compared casefold
# design section 6 (sha256sum 15/11) and the fence results of 14/11 (SA-19, pinned 16/11)
SHA_CODEX_CONFIG = '748bf0c8c5aeb08fd222337172ee939e6bc83ea1f3caf1ff5bcacbca693c3cf9'
SHA_CLAUDE_SETTINGS = '9778ccbff6db1f22a2c2e62747d1aa4a268d7ddf98af56b545a79af63bc9f055'
RAO_CLAUDE = '2026-11-14_Kiem_rao_cach_ly.md'
RAO_CODEX = os.path.join('_codex_test', 'ket_qua_2026-11-14.txt')
SHA_RAO = {RAO_CLAUDE: 'd33a82c1b15d77aa103d034445263b59a18fba7867576731a2d1a6a65d2e58c4',
           RAO_CODEX: '584ec1c4b05f9cead1c54050ec76086511575213a6fa919721b614d7916eed92'}
RAO_IDS = ('1', '2', '2b', '2c', '3', '3b', '3c', '3d', '4', '4b', '4c', '4d', '4e', '4f', '4g', '5', '5b', '6', '6b',
           '7')                         # kiem_quyen.ps1
RAO_RUNS = ('1', '2', '6')              # these run, every other test is refused
RAO_VERSIONS = dict(claude='2.1.287', codex='0.163.0')
HOI_CODEX_SRC = os.path.join(HERE, 'stra_a', 'hoi_codex.py')
HOI_CODEX_DST = 'hoi_codex.py'
CU = r'__cu_\d{8}-\d{4}(?:-\d+)?'


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
    for p in (path, os.path.abspath(path)):
        bad = [x for x in parts_of(p) + [x.lower() for x in re.split(r'[\\/]+', p) if x] if x in banned]
        if bad:
            raise Refused(f'{what}: path has {bad[0]!r} (S05 §2.1 fence)')


def check_file(path, what, exts=EXTS, max_bytes=None):
    """A regular file: not a link, not a hard link (SA-9), allowed extension, size, and text of that type (SA-10)."""
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    if not os.path.isfile(path):
        raise Refused(f'{what}: no file {path}')
    if os.path.islink(path) or real(path) != os.path.normcase(os.path.abspath(path)):
        raise Refused(f'{what}: link or junction in the path refused')
    st = os.stat(path)
    if st.st_nlink > 1:
        raise Refused(f'{what}: hard link refused (one name per file)')
    ext = os.path.splitext(path)[1].lower()
    if ext not in exts:
        raise Refused(f'{what}: extension {ext!r} not allowed (only {", ".join(exts)}; no video, no image)')
    if st.st_size > max_bytes:
        raise Refused(f'{what}: {st.st_size} bytes > {max_bytes}')
    with open(path, 'rb') as f:
        b = f.read()
    if any(b.startswith(m) for m in MEDIA_MAGIC) or b[4:8] == b'ftyp' or b'\x00' in b:
        raise Refused(f'{what}: binary or media content refused')
    try:
        t = b.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise Refused(f'{what}: not UTF-8 text')
    if ext == '.json':
        try:
            json.loads(t)
        except ValueError as exc:
            raise Refused(f'{what}: not JSON ({exc})')
    elif ext == '.csv':
        try:
            r = csv.reader(io.StringIO(t))
            head = next(r, None)
            for _ in r:
                pass
        except csv.Error as exc:
            raise Refused(f'{what}: not CSV ({exc})')
        if not head or not any(h.strip() for h in head):
            raise Refused(f'{what}: CSV with no header row')
    return sha256_bytes(b)


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


def cu_name(path):
    """Free __cu_<stamp>[-n] name next to path (never overwrites)."""
    b, e = (path, '') if os.path.isdir(path) else os.path.splitext(path)
    n, cand = 0, f'{b}__cu_{stamp()}{e}'
    while os.path.exists(cand):
        n += 1
        cand = f'{b}__cu_{stamp()}-{n}{e}'
    return cand


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_-]', '-', s)


def existing_casefold(folder, name):
    """The entry of folder equal to name under Windows rules (casefold), or None (SA-3, SA-4)."""
    if not os.path.isdir(folder):
        return None
    for n in os.listdir(folder):
        if n.casefold() == name.casefold():
            return n
    return None


def results_root():
    return os.path.join(THESIS, *RESULTS)


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
    """(index after the last row of the file table, rows as lists of cells, lines)."""
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


def check_cells(new_rows):
    if any(ch in str(c) for r in new_rows for c in r for ch in ('\n', '\r', '|', '`')):
        raise Refused('README.md: a cell has a line break, | or `')


def readme_new(hop, new_rows):
    """(old bytes, new bytes): rows inserted right after the last table row; every other byte kept (checked)."""
    check_cells(new_rows)
    p = os.path.join(hop, 'README.md')
    with open(p, 'rb') as f:
        old = f.read()
    nl = b'\r\n' if b'\r\n' in old else b'\n'
    end, rows, lines = readme_rows(old.decode('utf-8').replace('\r\n', '\n'))
    k = sum(len(x.encode('utf-8')) + len(nl) for x in lines[:end])     # byte offset after the last table row
    if end == len(lines) or k > len(old):
        raise Refused('README.md: table must be followed by a line')
    add = [f'| `{a}` | {b} | `{c}` | `{d[:16]}` | {e} |' for a, b, c, d, e in new_rows]
    ins = b''.join(x.encode('utf-8') + nl for x in add)
    out = old[:k] + ins + old[k:]
    _, rows2, _ = readme_rows(out.decode('utf-8').replace('\r\n', '\n'))
    if out[:k] != old[:k] or out[k + len(ins):] != old[k:] or len(rows2) != len(rows) + len(add):
        raise Refused('README.md: the new text is not the old text plus the inserted rows')
    return old, out


def readme_write(hop, data):
    p = os.path.join(hop, 'README.md')
    tmp = p + '.tmp_chep'
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):                                            # SA-11: no temp left on failure
            os.remove(tmp)


# ── luot ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def frozen_campaigns(freeze):
    rows, cols = read_csv(freeze)
    if 'campaign' not in cols:
        raise Refused(f'{freeze}: no campaign column')
    return {r['campaign'] for r in rows}


def tuning_rows(tuning_log):
    rows, cols = read_csv(tuning_log)
    miss = [c for c in TUNING_NEED if c not in cols]
    if miss:
        raise Refused(f'{tuning_log}: no column {miss}')
    return rows


def tuning_row(tuning_log, run_uid):
    hit = [r for r in tuning_rows(tuning_log) if r['run_uid'] == run_uid]
    if len(hit) != 1:
        raise Refused(f'run_uid {run_uid}: {len(hit)} rows in the tuning log (need exactly one)')
    return hit[0]


def a_runs(tuning_log):
    return [r for r in tuning_rows(tuning_log) if r['campaign'] in CAMPAIGNS_A]


def check_run(row, frozen):
    if row['campaign'] not in CAMPAIGNS_A:
        raise Refused(f'campaign {row["campaign"]!r}: not a campaign A run')
    if row['platform'] != PLATFORM_A:
        raise Refused(f'platform {row["platform"]!r}: campaign A runs are {PLATFORM_A}')
    if row['stage'] not in STAGES_A:
        raise Refused(f'stage {row["stage"]!r}: only {STAGES_A} (final check is barred, S05 §2.1 rule 4)')
    if row['campaign'] in frozen:
        raise Refused(f'{row["campaign"]} is frozen (S05_freeze.csv): not a tune run any more')


def run_folder(row):
    """results_dir of the row: relative to the thesis folder, inside 05_DATA/results, at least three levels below
    it (results/<code>/<campaign folder>/<run>), no banned part (SA-7)."""
    d = row['results_dir']
    if not d:
        raise Refused('the run has no results_dir')
    if not os.path.isabs(d):
        d = os.path.join(THESIS, d)
    if not os.path.isdir(d):
        raise Refused(f'results_dir {row["results_dir"]}: no folder')
    root = results_root()
    if not inside(d, root):
        raise Refused(f'results_dir {row["results_dir"]}: not inside 05_DATA/results')
    depth = len(os.path.relpath(os.path.realpath(d), os.path.realpath(root)).split(os.sep))
    if depth < 3:
        raise Refused(f'results_dir {row["results_dir"]}: not a run folder (results/<code>/<campaign>/<run>)')
    check_parts(d, FORBIDDEN_PARTS, 'results_dir')
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
        x = _f(v, f'{col}')
        if not (float(r['lo']) <= x <= float(r['hi'])):
            out.append(dict(gain=r['gain'], value=x, lo=float(r['lo']), hi=float(r['hi'])))
    return out, None


def luot(a):
    hop = a.hop_thu
    row = tuning_row(a.tuning_log, a.run_uid)
    check_run(row, frozen_campaigns(a.freeze))
    root = run_folder(row)
    files = dict(log_esp32=a.esp32, score=a.score, sai_so=a.rows, quy_dao=a.frames, theo_doi=a.track)
    if a.segments:
        files['log_segments'] = a.segments
    if a.ghi_chu_file:
        files['ghi_chu'] = a.ghi_chu_file
    sh = {}
    for role, p in files.items():
        check_parts(p, FORBIDDEN_PARTS, role)
        if not inside(p, root):
            raise Refused(f'{role}: {p} is not inside results_dir of the run')
        sh[role] = check_file(p, role, exts=ROLE_EXT[role],
                              **(dict(max_bytes=NOTE_MAX_BYTES) if role == 'ghi_chu' else {}))
    if a.segments and os.path.basename(a.segments) != os.path.basename(a.esp32) + '.segments.csv':
        raise Refused('log_segments: name must be <log>.segments.csv of log_esp32')
    if os.path.dirname(real(a.track)) != os.path.dirname(real(a.frames)):
        raise Refused('theo_doi: track.json must sit next to frames.csv')
    sc = read_json(a.score)
    if not isinstance(sc, dict):
        raise Refused('score: not a JSON object')
    if sh['score'] != row['gate_json_sha256']:
        raise Refused('score: SHA-256 differs from gate_json_sha256 of the tuning-log row')
    if sc.get(SCORE_ESP32_SHA) != sh['log_esp32']:
        raise Refused(f'log_esp32: SHA-256 differs from {SCORE_ESP32_SHA} in score.json')
    if sc.get(SCORE_FRAMES_SHA) != sh['quy_dao']:
        raise Refused(f'quy_dao: SHA-256 differs from {SCORE_FRAMES_SHA} in score.json')
    if sc.get(SCORE_OUTCOME) != row['run_outcome']:
        raise Refused(f'score: {SCORE_OUTCOME} {sc.get(SCORE_OUTCOME)!r} differs from run_outcome {row["run_outcome"]!r}')
    _, cols = read_csv(a.rows)
    miss = [c for c in ROWS_NEED if c not in cols]
    if miss:
        raise Refused(f'sai_so: no column {miss}')
    note = None
    if a.ghi_chu is not None:
        note = a.ghi_chu.encode('utf-8')
        if len(note) > NOTE_MAX_BYTES:
            raise Refused('ghi_chu: > 20 KB')
    names = {}
    for role in ROLES:                                                     # SA-4: names checked before any copy
        if role == 'ghi_chu' and note is not None:
            n = 'ghi_chu.txt'
        elif role in files:
            n = os.path.basename(files[role])
        else:
            continue
        if n.casefold() in {x.casefold() for x in names.values()} or n.casefold() == 'luot.json':
            raise Refused(f'two files of the run with the name {n} (Windows names ignore case)')
        names[role] = n
    try:
        it = int(row.get('iter_no') or row['run_uid'].split(':')[-1])
    except ValueError:
        raise Refused(f'run_uid {row["run_uid"]}: no iteration number')
    ma = f'{safe_name(row["campaign"])}__{safe_name(row.get("session_id", "") or row["run_uid"].split(":")[0])}__{it:03d}'
    base = os.path.join(hop, LUOT_DIR)
    out_of, note_m = mien_check(row, hop)
    t = now()
    ver = next((str(sc[k]) for k in SCORE_VERSION_KEYS if k in sc), '')
    rows_add = []
    for role, n in names.items():
        src = files.get(role)
        rows_add.append((f'{LUOT_DIR}/{ma}/{n}', f'lượt {row["run_uid"]}, {role}' + (
            f' (bộ chấm {ver})' if role == 'score' and ver else ''),
            thesis_commit(src) if src else 'ghi chú Liam',
            sh[role] if role in sh else sha256_bytes(note), t.strftime('%d/%m/%Y %H:%M')))
    rows_add.append((f'{LUOT_DIR}/{ma}/luot.json', f'lượt {row["run_uid"]}, kê khai lượt (chep_luot_A)',
                     f'script {script_commit() or "?"}', '0' * 64, t.strftime('%d/%m/%Y %H:%M')))
    check_cells(rows_add)                                                  # SA-11: before any change
    with Lock():
        os.makedirs(base, exist_ok=True)
        if real(base) != real(os.path.join(hop, LUOT_DIR)) or not inside(base, hop):
            raise Refused('HOP_THU\\luot is a link')
        have = existing_casefold(base, ma)
        if have and not a.thay:
            raise Refused(f'{LUOT_DIR}/{ma} exists: --thay to copy again (old folder kept)')
        tmp = os.path.join(base, '.dang_chep_' + ma)
        if os.path.exists(tmp):
            raise Refused(f'{tmp} left by a failed copy: look, then remove it by hand')
        dst, moved = os.path.join(base, ma), None
        try:
            os.makedirs(tmp)
            meta = {}
            for role, n in names.items():
                p = os.path.join(tmp, n)
                if role == 'ghi_chu' and note is not None:
                    with open(p, 'xb') as f:
                        f.write(note)
                else:
                    shutil.copyfile(files[role], p)
                got = sha256(p)
                if got != (sh.get(role) or sha256_bytes(note)):
                    raise Refused(f'{role}: copy SHA-256 differs from the checked source')
                meta[role] = dict(ten=n, sha256=got, co=os.path.getsize(p))
            info = dict(run_uid=row['run_uid'], campaign=row['campaign'], stage=row['stage'],
                        speed_mps=row.get('speed_mps', ''), gain_set_id=row.get('gain_set_id', ''), files=meta,
                        luc_chep=t.isoformat(timespec='seconds'), script=script_commit(),
                        ngoai_mien=out_of, ghi_chu_mien=note_m)
            with open(os.path.join(tmp, 'luot.json'), 'x', encoding='utf-8') as f:
                f.write(json.dumps(info, ensure_ascii=False, indent=1, allow_nan=False))
            r = list(rows_add[-1])
            r[3] = sha256(os.path.join(tmp, 'luot.json'))
            rows_add[-1] = tuple(r)
            old, new = readme_new(hop, rows_add)
            if have:
                moved = cu_name(os.path.join(base, have))
                os.rename(os.path.join(base, have), moved)
            placed = False
            try:
                os.rename(tmp, dst)
                placed = True
                readme_write(hop, new)
            except BaseException:                                          # SA-11: undo both steps
                if placed:
                    shutil.rmtree(dst)
                if moved:
                    os.rename(moved, os.path.join(base, have))
                raise
        except BaseException:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            raise
    for r in rows_add:
        print(f'{r[0]}  {r[3][:16]}')
    if out_of:
        print(f'CẢNH BÁO: gain ngoài miền A: {out_of}')
    return 0


# ── them ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _put(hop, items, thay):
    """items: (src, name, source text, commit, checked SHA). All or nothing (SA-11): rows checked first, copies
    in temp files checked against the SHA taken at the check (SA-9), old copies renamed __cu_ only then, README
    last; any failure undoes every step."""
    t = now().strftime('%d/%m/%Y %H:%M')
    rows = [(n, s, c, h, t) for _, n, s, c, h in items]
    check_cells(rows)
    if len({n.casefold() for _, n, _, _, _ in items}) != len(items):
        raise Refused('two files with one name')
    with Lock():
        _, new = readme_new(hop, rows)
        have = {}
        for _, n, _, _, _ in items:
            e = existing_casefold(hop, n)
            if e and e != n:
                raise Refused(f'{n}: the mailbox has {e} (same name for Windows)')
            if e and not thay:
                raise Refused(f'{n} is in the mailbox: --thay (old copy renamed __cu_, its row kept)')
            have[n] = e
        tmps, moved, done = [], [], []
        try:
            for src, n, _, _, h in items:
                tmp = os.path.join(hop, '.tmp_chep_' + n)
                tmps.append(tmp)                                           # SA-11: known before the copy
                shutil.copyfile(src, tmp)
                if sha256(tmp) != h:
                    raise Refused(f'{n}: copy SHA-256 differs from the checked source')
            for (src, n, _, _, _), tmp in zip(items, tmps):
                dst = os.path.join(hop, n)
                if have[n]:
                    c = cu_name(dst)
                    os.rename(dst, c)
                    moved.append((c, dst))
                os.rename(tmp, dst)
                done.append(dst)
            readme_write(hop, new)
        except BaseException:
            for d in done:
                if os.path.exists(d):
                    os.remove(d)
            for c, dst in moved:
                os.rename(c, dst)
            for tmp in tmps:
                if os.path.exists(tmp):
                    os.remove(tmp)
            raise
    for r in rows:
        print(f'{r[0]}  {r[3][:16]}')


def them(a):
    hop = a.hop_thu
    if len(a.nguon) != 1:
        raise Refused('--nguon: one source')
    src = a.nguon[0]
    if a.loai == 'spec':
        rel = rel_thesis(src)
        if rel not in SPECS:
            raise Refused(f'spec: only {SPECS}')
        if git(THESIS, 'status', '--porcelain', '--', rel):
            raise Refused(f'{rel}: not clean in git (commit first)')
        h = check_file(src, 'spec', exts=('.md',))
        _put(hop, [(src, os.path.basename(src), f'spec ({a.ly_do})', git(THESIS, 'log', '-1', '--format=%h', '--', rel),
                    h)], a.thay)
    elif a.loai == 's03_quet':
        if not os.path.isdir(src):
            raise Refused('s03_quet: --nguon is ONE S03 run folder')
        s03_run_ok(src)
        items = []
        for n in ('sweep.csv', 'levels.csv'):
            p = s03_file(src, n)
            items.append((p, n, f'S03 {os.path.basename(os.path.realpath(src))}, quét công khai ({a.ly_do})',
                          thesis_commit(p), check_file(p, n, exts=('.csv',))))
        _put(hop, items, a.thay)
    elif a.loai == 'mien_A':
        if not a.tuning_log:
            raise Refused('mien_A: --tuning-log (the domain is locked before the first A run, SA-13)')
        if a_runs(a.tuning_log):
            raise Refused('a campaign A run is in the tuning log: the A domain is locked')
        if os.path.basename(src).casefold() != 'mien_a.csv':
            raise Refused('mien_A: only a mien_A.csv written by the mien command')
        h = check_file(src, 'mien_A', exts=('.csv',))
        js = read_json(os.path.join(os.path.dirname(src), 'mien_A.json'))
        run = os.path.join(THESIS, js['s03_run'])
        with open(src, 'rb') as f:
            got = f.read().replace(b'\r\n', b'\n')                          # git autocrlf may give CRLF
        if _mien_csv_bytes(mien_rows(run)) != got:
            raise Refused('mien_A.csv differs from the domain computed again from sweep.csv (hand edit?)')
        _put(hop, [(src, 'mien_A.csv', f'miền A tính từ S03 {js["s03_run"]} ({a.ly_do})', thesis_commit(src), h)],
             a.thay)
    else:                                                                  # tai_lieu
        check_parts(src, FORBIDDEN_PARTS, 'tai_lieu')
        h = check_file(src, 'tai_lieu', exts=('.md', '.csv', '.json'))
        n = os.path.basename(src)
        if n.casefold() in RESERVED or re.search(CU, n):
            raise Refused(f'tai_lieu: name {n} is reserved')
        _put(hop, [(src, n, f'tài liệu: {a.ly_do}', thesis_commit(src), h)], a.thay)
    return 0


# ── mien (section 10) ─────────────────────────────────────────────────────────────────────────────────────────────

def s03_run_ok(run):
    """A main S03 run: folder 05_DATA/results/S03/<run> of the thesis (real path), not history, meta.yaml with
    trial false and every dirty flag present and false (SA-6)."""
    check_parts(run, SEALED_PARTS, 's03 run')
    s03 = os.path.join(results_root(), 'S03')
    if os.path.dirname(real(run)) != real(s03):
        raise Refused(f'{run}: not a run folder 05_DATA/results/S03/<run>')
    if os.path.basename(os.path.realpath(run)).casefold() in {h.casefold() for h in S03_HISTORY}:
        raise Refused(f'{run}: history run (S05 §3.2), never the A domain')
    p = s03_file(run, 'meta.yaml', exts=('.yaml',))
    with open(p, encoding='utf-8') as f:
        m = yaml.safe_load(f)
    if not isinstance(m, dict):
        raise Refused('meta.yaml: not a mapping')
    if m.get('trial') is not False or m.get('tool_check') is True:
        raise Refused('meta.yaml: tool check run (trial not false / tool_check), not a main S03 run')
    flags = [(k, v) for k, v in _walk(m) if k.split('.')[-1] == 'dirty']
    if not flags or not isinstance(m.get('script'), dict) or m['script'].get('dirty') is not False:
        raise Refused('meta.yaml: script.dirty false is required')
    bad = [k for k, v in flags if v is not False]
    if bad:
        raise Refused(f'meta.yaml: dirty build {bad}')
    return m


def s03_file(run, name, exts=('.csv',)):
    """A file at the root of the run, by name, checked (no link, real path in the run, SA-5)."""
    p = os.path.join(run, name)
    if os.path.splitext(name)[1] == '.yaml':
        if not os.path.isfile(p) or os.path.islink(p) or os.stat(p).st_nlink > 1:
            raise Refused(f'{name}: missing, link or hard link')
    else:
        check_file(p, name, exts=exts)
    if os.path.dirname(real(p)) != real(run):
        raise Refused(f'{name}: real path not in the run folder')
    check_parts(p, SEALED_PARTS, name)
    return p


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


def kiem_check(run):
    """S03 v8.4 §3.1, S05 v3.48 §2.5: kiem_ngoai_mien.csv at the root of the run (by name) has one total row per
    controller, each 'đã kiểm đủ', and every result a valid one; anything else refuses the domain. A flagged point
    no longer refuses (S03 v8.5 K6, S05 v3.47). Returns (point rows, total rows by controller) for kiem_points
    (S84-5: the rows must match the points outside [L, H] of the sweep)."""
    rows, cols = read_csv(s03_file(run, KIEM_FILE))
    miss = [c for c in KIEM_NEED if c not in cols]
    if miss:
        raise Refused(f'{KIEM_FILE}: columns missing {miss}')
    tot, pts = {}, []
    for r in rows:
        if r['dong'] == 'tong':
            if r['controller'] in tot:
                raise Refused(f'{KIEM_FILE}: two total rows for {r["controller"]}')
            if r['ket_qua'] not in KIEM_TOTAL:
                raise Refused(f'{KIEM_FILE}: total {r["ket_qua"]!r} not in {KIEM_TOTAL}')
            tot[r['controller']] = r
        elif r['dong'] == 'diem':
            if r['ket_qua'] not in KIEM_POINT:
                raise Refused(f'{KIEM_FILE}: result {r["ket_qua"]!r} not in {KIEM_POINT}')
            pts.append(r)
        else:
            raise Refused(f'{KIEM_FILE}: dong {r["dong"]!r} is not diem or tong')
    if set(tot) != set(GAINS_S03):
        raise Refused(f'{KIEM_FILE}: total rows for {sorted(tot)}, need {sorted(GAINS_S03)}')
    bad = {c: t['ket_qua'] for c, t in tot.items() if t['ket_qua'] != 'đã kiểm đủ'}
    if bad:
        raise Refused(f'{KIEM_FILE}: {bad}: the A domain needs a sweep checked in full (S05 v3.48 §2.5)')
    return pts, tot


def kiem_points(pts, tot, want):
    """S84-5: the point rows of kiem_ngoai_mien.csv are exactly the points outside [L, H] found from the public
    sweep.csv and levels.csv (want: {(controller, factor, value): ('L', L) | ('H', H)}), one row each, on the right
    side against the right level (S84V2-2: can_value finite and equal to L or H of levels.csv), and the counts of
    each total row agree with them (nothing checked twice, nothing left out)."""
    got = {}
    for r in pts:
        key = (r['controller'], r['factor'], _f(r['value'], f'{KIEM_FILE} value'))
        if key in got:
            raise Refused(f'{KIEM_FILE}: point {key} twice')
        got[key] = r
    if set(got) != set(want):
        raise Refused(f'{KIEM_FILE}: points do not match the sweep outside [L, H]: missing '
                      f'{sorted(set(want) - set(got))[:5]}, extra {sorted(set(got) - set(want))[:5]}')
    side = [k for k, (s, _) in want.items() if got[k]['can'] != s]
    if side:
        raise Refused(f'{KIEM_FILE}: level compared on the wrong side for {side[:5]}')
    lvl = [k for k, (_, x) in want.items() if _f(got[k]['can_value'], f'{KIEM_FILE} can_value {k}') != x]
    if lvl:
        raise Refused(f'{KIEM_FILE}: can_value is not the L or H of levels.csv for {lvl[:5]}')
    for c, t in tot.items():
        res = [got[k]['ket_qua'] for k in got if k[0] == c]
        need = dict(n_diem_ngoai=len(res), n_da_kiem=len(res), n_chua_kiem=0, n_co=res.count('cờ'),
                    n_chua_ket_luan=res.count('chưa kết luận'), n_khong_co=res.count('không cờ'))
        bad = {k: t.get(k) for k, v in need.items() if t.get(k) != str(v)}
        if bad:
            raise Refused(f'{KIEM_FILE}: {c} total {bad} do not agree with its {len(res)} point rows')


def mien_rows(run):
    """Opens only sweep.csv, levels.csv, kiem_ngoai_mien.csv, meta.yaml at the root of the run (by name). The A
    domain per gain: the WHOLE sweep grid after the cut at the gain-table ceiling, original and extension points,
    up to b_n — the largest grid point after the cut (S05 v3.48 §2.5, S03 v8.6 §3.1; omega_o has no extension and
    goes up to b)."""
    s03_run_ok(run)
    k_pts, k_tot = kiem_check(run)
    sw, scol = read_csv(s03_file(run, 'sweep.csv'))
    lv, lcol = read_csv(s03_file(run, 'levels.csv'))
    miss = [c for c in LEVEL_COLS if c not in lcol] + [c for c in ('luoi',) if c not in scol]
    if miss:
        raise Refused(f'S03 v8.3 columns missing {miss}: run made by a tool before v8.3, a, b not guessed')
    speeds = sorted(S03_SPEEDS)                                             # SA-15: fixed by the spec
    got = sorted({_f(r['v'], 'v') for r in sw})
    if got != speeds:
        raise Refused(f'sweep.csv speeds {got}, need {speeds} (S03 v8)')
    ctrls = {r['controller'] for r in sw}
    if ctrls != set(GAINS_S03):
        raise Refused(f'controllers {sorted(ctrls)}: need {sorted(GAINS_S03)}')
    for c in GAINS_S03:
        for alias in DEFAULT_ROWS:                                         # SA2-2: either name, each complete
            by_mu = {}                   # S84-4: S03 writes the default gains at the middle mu AND at the low, high mu
            for r in sw:
                if r['controller'] == c and r['factor'] == alias:
                    by_mu.setdefault(r.get('mu', ''), []).append(_f(r['v'], 'v'))
            for mu, dv in by_mu.items():
                if sorted(dv) != speeds:
                    raise Refused(f'{c}: {alias} rows at mu {mu!r}: speeds {sorted(dv)}, need {speeds} once each')
        if not any(r['controller'] == c and r['factor'] in DEFAULT_ROWS for r in sw):
            raise Refused(f'{c}: no default check rows')
    lev = {}
    for r in lv:
        if (r['controller'], r['factor']) in lev:
            raise Refused(f'levels.csv: two rows {r["controller"]} {r["factor"]}')
        lev[(r['controller'], r['factor'])] = r
    grid = {}
    for r in sw:
        if r['factor'] in DEFAULT_ROWS:
            continue
        if r.get('luoi') not in LUOI:
            raise Refused(f'{r["controller"]} {r["factor"]}: luoi {r.get("luoi")!r} not in {LUOI}')
        x = _f(r['value'], f'{r["controller"]} {r["factor"]} value')
        if x < 0:
            raise Refused(f'{r["controller"]} {r["factor"]}: value {x} < 0')
        grid.setdefault((r['controller'], r['factor']), {}).setdefault(x, []).append(r)    # SA-16: numeric key
    out, want = [], {}
    for c in sorted(GAINS_S03):
        fs = {f for (cc, f) in grid if cc == c}
        if fs != GAINS_S03[c]:                                                              # SA-14: exact set
            raise Refused(f'{c}: gains {sorted(fs)} in sweep.csv, need exactly {sorted(GAINS_S03[c])}')
        for fac in sorted(fs):
            pts = grid[(c, fac)]
            for v, rs in pts.items():
                got = sorted(_f(r['v'], 'v') for r in rs)
                if got != speeds:
                    raise Refused(f'{c} {fac} = {v}: rows at speeds {got}, need {speeds} once each')
            if (c, fac) not in lev:
                raise Refused(f'{c} {fac}: no row in levels.csv')
            L = lev[(c, fac)]
            lab = {}
            for v, rs in pts.items():                                   # S03 v8.4: one part of the grid per point
                ls = {r['luoi'] for r in rs}
                if len(ls) != 1:
                    raise Refused(f'{c} {fac} = {v}: luoi {sorted(ls)} differs between the rows of one point')
                lab[v] = ls.pop()
            goc = sorted(v for v in pts if lab[v] != 'noi')              # the original grid
            noi = sorted(v for v in pts if lab[v] == 'noi')              # the extension points
            vals = sorted(pts)                                           # the whole cut grid: the A domain
            if not goc:
                raise Refused(f'{c} {fac}: no point of the original grid')
            lo, hi, g_hi, n_diem = vals[0], vals[-1], goc[-1], len(vals)
            has0 = lo == 0.0
            a_, b_ = _f(L['a'], f'{c} {fac} a'), _f(L['b'], f'{c} {fac} b')
            if not has0 and lo != a_:
                raise Refused(f'{c} {fac}: lowest grid point {lo} is not a = {a_}')
            if fac in OMEGA_O:
                if noi:
                    raise Refused(f'{c} {fac}: omega_o grid is cut, never extended')
                if hi != b_:                                             # S05 v3.48 §2.5: omega_o ends at b
                    raise Refused(f'{c} {fac}: highest grid point {hi} is not b = {b_}')
                bn = (L.get('b_n') or '').strip()
                if bn and _f(bn, f'{c} {fac} b_n') != b_:
                    raise Refused(f'{c} {fac}: b_n {bn} is not b = {b_} (omega_o has no extension)')
                if hi > OMEGA_O_MAX or _f32(hi) > OMEGA_O_MAX:
                    raise Refused(f'{c} {fac}: hi {hi!r} > {OMEGA_O_MAX} (binary64 or float32)')
                src = 'tran_omega_o' if hi == OMEGA_O_MAX else 'b'
            else:
                bn = (L.get('b_n') or '').strip()
                if not bn:
                    raise Refused(f'{c} {fac}: no b_n in levels.csv (S05 v3.48 §2.5)')
                bn_ = _f(bn, f'{c} {fac} b_n')
                if hi != bn_:                                            # S05 v3.48 §2.5: hi = b_n of levels.csv
                    raise Refused(f'{c} {fac}: highest grid point {hi} is not b_n = {bn_}')
                if g_hi != min(b_, bn_):
                    raise Refused(f'{c} {fac}: highest point of the original grid {g_hi} is not b = {b_} '
                                  f'(b_n = {bn_} when the grid is cut below b)')
                src = 'b_n'
            if any(x <= b_ for x in noi):
                raise Refused(f'{c} {fac}: an extension point (luoi noi) is not above b = {b_}')
            d = _f(L['default'], f'{c} {fac} default')
            Lo, Hi = _f(L['L'], f'{c} {fac} L'), _f(L['H'], f'{c} {fac} H')
            if not (lo <= d <= hi):
                raise Refused(f'{c} {fac}: default {d} outside [{lo}, {hi}]')
            if not (lo <= Lo <= Hi <= g_hi):                             # levels sit on the original grid only
                raise Refused(f'{c} {fac}: [L, H] = [{Lo}, {Hi}] not inside [{lo}, {g_hi}]')
            out.append(dict(controller_s03=c, controller=CTRL_MAP[c], gain=fac, lo=repr(lo), hi=repr(hi),
                            n_diem=str(n_diem), co_diem_0='true' if has0 else 'false', nguon_hi=src))
            for x in pts:                                      # S84-5: every point (original or extension) outside
                if x < Lo or x > Hi:                           # [L, H] must have its row in kiem_ngoai_mien.csv
                    want[(c, fac, x)] = ('L', Lo) if x < Lo else ('H', Hi)    # S84V2-2: side and level value
    kiem_points(k_pts, k_tot, want)
    return out


def _mien_csv_bytes(rows):
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=MIEN_COLS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode('utf-8')


def mien(a):
    if a_runs(a.tuning_log):
        raise Refused('a campaign A run is in the tuning log: the A domain is locked before the first run')
    run = a.s03_run
    out_rows = mien_rows(run)
    m = s03_run_ok(run)
    body = _mien_csv_bytes(out_rows)
    info = dict(s03_run=rel_thesis(run), s03_commit=(m.get('script') or {}).get('commit', ''),
                s03_date=str(m.get('date', '')),
                sha256={n: sha256(os.path.join(run, n)) for n in S03_PUBLIC},
                mien_A_csv_sha256=sha256_bytes(body), script=script_commit(),
                luc_tinh=now().isoformat(timespec='seconds'), spec='S05 v3.48 §2.5, S03 v8.6 §3.1')
    js_body = json.dumps(info, ensure_ascii=False, indent=1).encode('utf-8')
    os.makedirs(a.out, exist_ok=True)
    pair = [(os.path.join(a.out, 'mien_A.csv'), body), (os.path.join(a.out, 'mien_A.json'), js_body)]
    if any(os.path.exists(p) for p, _ in pair) and not a.thay:
        raise Refused(f'{pair[0][0]} exists: --thay (no A run yet; old copy kept)')
    tmps, moved, done = [], [], []
    try:                                                                   # SA-12: the pair or nothing
        for p, b in pair:
            t = p + '.tmp_mien'
            if os.path.exists(t):
                raise Refused(f'{t} left by a failed run: look, then remove it by hand')
            tmps.append(t)                                                 # SA-12: known before the write
            with open(t, 'xb') as f:
                f.write(b)
        for (p, _), t in zip(pair, tmps):
            if os.path.exists(p):
                c = cu_name(p)
                os.rename(p, c)
                moved.append((c, p))
            os.rename(t, p)
            done.append(p)
    except BaseException:
        for p in done:
            os.remove(p)
        for c, p in moved:
            os.rename(c, p)
        for t in tmps:
            if os.path.exists(t):
                os.remove(t)
        raise
    for r in out_rows:
        print(f'{r["controller"]:6s} {r["gain"]:12s} [{r["lo"]}, {r["hi"]}]  {r["nguon_hi"]}')
    print(f'mien_A: {pair[0][0]} {info["mien_A_csv_sha256"][:16]}')
    return 0


# ── ke-khai, kiem-c13 ─────────────────────────────────────────────────────────────────────────────────────────────

def box_check(hop):
    """Both ways (SA-17): every file in the mailbox (but README.md) has a row: a current file its latest row, a
    __cu_ copy an older row of its name (any row when no current file has the name); the latest row of every
    name is on the disk (current or __cu_ copy); temp folders or files left by a failed copy are errors."""
    with open(os.path.join(hop, 'README.md'), encoding='utf-8') as f:
        _, rows, _ = readme_rows(f.read().replace('\r\n', '\n'))
    by = {}
    for r in rows:
        by.setdefault(r[0].strip('`'), []).append(r[3].strip('`'))
    bad, disk = [], {}
    for dp, dn, fn in os.walk(hop):
        for d in dn:
            if d.startswith('.dang_chep_'):
                bad.append(f'{os.path.relpath(os.path.join(dp, d), hop)}: temp folder of a failed copy')
        for n in fn:
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, hop).replace('\\', '/')
            if rel == 'README.md':
                continue
            if n.startswith(('.tmp_chep_', 'README.md.tmp')):
                bad.append(f'{rel}: temp file of a failed copy')
                continue
            parts = rel.split('/')
            key = '/'.join(re.sub(CU + r'$', '', x) if i < len(parts) - 1 else re.sub(CU + r'(?=\.[^.]*$|$)', '', x)
                           for i, x in enumerate(parts))
            cu = key != rel
            disk.setdefault(key, []).append((rel, sha256(p)[:16], cu))
    for key, lst in disk.items():
        cur = [x for x in lst if not x[2]]
        rs = by.get(key, [])
        if not rs:
            bad.extend(f'{rel}: file with no row (strange file)' for rel, _, _ in lst)
            continue
        for rel, s, cu in lst:
            if not cu and s != rs[-1]:
                bad.append(f'{rel}: SHA {s} differs from its latest row {rs[-1]}')
            if cu and s not in (rs[:-1] if cur else rs):
                bad.append(f'{rel}: no older row of {key} with this SHA')
    for key, rs in by.items():                                             # SA-17: one file per row
        on_disk = [s for _, s, _ in disk.get(key, [])]
        for i, s in enumerate(rs):
            if s not in on_disk:
                bad.append(f'{key}: {"latest" if i == len(rs) - 1 else "older"} row {s} has no file on the disk')
            else:
                on_disk.remove(s)
        if on_disk and key in by:
            bad.append(f'{key}: {len(on_disk)} more files than rows')
    return rows, bad


def rao_check(root):
    """Both fence results present, pinned (SA-19), the Codex one with exactly the test ids, each as expected;
    lock files equal to their marks."""
    bad = []
    for rel, want in SHA_RAO.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            bad.append(f'{rel}: missing')
        elif sha256(p) != want:
            bad.append(f'{rel}: SHA differs from the approved result of 14/11 (run the fence test again)')
    px = os.path.join(root, RAO_CODEX)
    if os.path.isfile(px):
        seen = []
        with open(px, encoding='utf-8', errors='replace') as f:
            for line in f:
                m = re.match(r'^(\d+[a-z]?) .*?=> (.*)$', line.rstrip('\n'))
                if not m:
                    continue
                k, res = m.group(1), m.group(2)
                seen.append(k)
                ok_run = k in RAO_RUNS
                refused = re.search(r'denied|Access|Cannot find|loi mang|Traceback', res) is not None
                if ok_run == refused:
                    bad.append(f'Codex fence test {k}: {"refused" if refused else "ran"}, expected '
                               f'{"to run" if ok_run else "a refusal"}')
        if sorted(seen) != sorted(RAO_IDS):
            bad.append(f'Codex fence tests {sorted(seen)} differ from {sorted(RAO_IDS)}')
    for n, want in (('.codex/config.toml', SHA_CODEX_CONFIG), ('.claude/settings.json', SHA_CLAUDE_SETTINGS)):
        p = os.path.join(root, n)
        if not os.path.isfile(p) or sha256(p) != want:
            bad.append(f'{n}: SHA differs from the mark (design section 6)')
    sh = {k: (sha256(os.path.join(root, k)) if os.path.isfile(os.path.join(root, k)) else None) for k in SHA_RAO}
    return dict(claude=dict(file=RAO_CLAUDE, sha256=sh[RAO_CLAUDE]),
                codex=dict(file=RAO_CODEX.replace('\\', '/'), sha256=sh[RAO_CODEX])), bad


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


def read_ke_khai(path):
    """A declaration written by ke-khai, every field used later of the right type (SA2-4)."""
    with open(path, encoding='utf-8') as f:
        d = yaml.safe_load(f)
    ok = (isinstance(d, dict) and isinstance(d.get('loi'), list) and isinstance(d.get('hop_thu'), dict)
          and isinstance(d['hop_thu'].get('readme_sha256'), str) and isinstance(d.get('nhat_ky'), dict)
          and all(isinstance(v, dict) and isinstance(v.get('co'), int) and isinstance(v.get('sha256'), str)
                  for v in d['nhat_ky'].values())
          and isinstance(d.get('tro_ly_codex'), dict) and isinstance(d['tro_ly_codex'].get('lan_goi'), list)
          and all(isinstance(c, dict) and isinstance(c.get('log'), str) for c in d['tro_ly_codex']['lan_goi'])
          and isinstance(d.get('tro_ly_claude'), dict) and isinstance(d['tro_ly_claude'].get('phien'), list))
    # SA3-1, SA4-1: typed call entries; SA-20: the STRA_A sessions list
    if not ok:
        raise Refused(f'{path}: not a declaration of ke-khai (fields nhat_ky, loi, hop_thu, tro_ly_codex)')
    return d


def declaration_problems(kk, root):
    """Why a declaration is not the current state (SA-20, SA-22): errors in it, README changed after it, NHAT_KY or
    CODEX changed after it, a Codex call not on the pinned model and effort."""
    out = []
    if kk['loi']:
        out.append(f'the declaration has {len(kk["loi"])} errors')
    if kk['hop_thu']['readme_sha256'] != sha256(os.path.join(root, HOP_THU_DIR, 'README.md')):
        out.append('README.md changed after the declaration')
    now_nk, _ = nhat_ky(root, None)
    if now_nk != kk['nhat_ky']:
        out.append('NHAT_KY or CODEX changed after the declaration')
    hc = load_hoi_codex()
    bad = [c.get('log') for c in kk['tro_ly_codex']['lan_goi'] if c.get('model') != hc.MODEL or c.get('effort') != hc.EFFORT]
    if bad:
        out.append(f'Codex calls not on {hc.MODEL} / {hc.EFFORT}: {bad}')
    if list(kk['tro_ly_codex']['lan_goi']) != codex_calls(root):           # SA-20: whole entries, read again
        out.append('Codex calls (log, model, effort) differ from the logs')
    ph = kk['tro_ly_claude']['phien']                                      # SA-20: each entry id,model,effort
    if not ph or not all(isinstance(x, str) and len(x.split(',')) == 3 and all(p.strip() for p in x.split(','))
                         for x in ph):
        out.append('STRA_A sessions missing or not "id,model,effort": ke-khai --phien ID,MODEL,EFFORT')
    _, bb = box_check(os.path.join(root, HOP_THU_DIR))                     # SA-22: the mailbox itself, now
    out.extend(f'mailbox: {x}' for x in bb)
    return out


def load_hoi_codex():
    """The wrapper module, for its pinned program, version, model and effort (SA2-3: one source)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('hoi_codex_src', HOI_CODEX_SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def codex_cli_version():
    """Version of the pinned Codex, asked only after its SHA matches the pin (SA3-2: never run another program)."""
    hc = load_hoi_codex()
    if not os.path.isfile(hc.CODEX_EXE):
        return ''
    if sha256(hc.CODEX_EXE) != hc.CODEX_SHA256:
        return 'SHA khác bản đã thử rào'
    return _tool_version([hc.CODEX_EXE, '--version'])


def ke_khai(a):
    root = a.stra_a
    hop = os.path.join(root, HOP_THU_DIR)
    if os.path.exists(a.out):
        raise Refused(f'{a.out} exists: never overwritten')
    if bool(a.truoc) == bool(a.lan_dau):
        raise Refused('give --truoc <last declaration> (append-only check), or --lan-dau for the first one (SA-18)')
    if not re.fullmatch(r'hop_thu_A_[A-Za-z0-9_-]+\.yaml', os.path.basename(a.out)):
        raise Refused('--out: name hop_thu_A_<tag>.yaml, in the folder of A (every declaration in one place, SA-18)')
    folder = os.path.dirname(os.path.abspath(a.out))
    if a.truoc:                                                            # SA-18: the newest declaration only
        if real(os.path.dirname(os.path.abspath(a.truoc))) != real(folder):
            raise Refused('--truoc: a declaration in the same folder as --out')
        mine = str(read_ke_khai(a.truoc).get('luc', ''))
        for n in os.listdir(folder):
            if re.fullmatch(r'hop_thu_A_[A-Za-z0-9_-]+\.yaml', n):
                try:
                    other = str(read_ke_khai(os.path.join(folder, n)).get('luc', ''))
                except (Refused, yaml.YAMLError):
                    continue
                if other > mine:
                    raise Refused(f'--truoc: {n} is newer; give the newest declaration')
                tr = read_ke_khai(os.path.join(folder, n)).get('truoc')     # SA-18: a chain, not only the time
                if isinstance(tr, dict) and str(tr.get('file', '')).casefold() == os.path.basename(a.truoc).casefold():
                    raise Refused(f'--truoc: {n} already follows it; give the newest declaration')
    prev = read_ke_khai(a.truoc)['nhat_ky'] if a.truoc else None
    if a.lan_dau and [n for n in os.listdir(os.path.dirname(os.path.abspath(a.out)))
                      if n.startswith('hop_thu_A') and n.endswith('.yaml')]:
        raise Refused('--lan-dau: a declaration hop_thu_A*.yaml is already in that folder; give --truoc')
    rows, bad = box_check(hop)
    rao, bad2 = rao_check(root)
    nk, bad3 = nhat_ky(root, prev)
    hc = os.path.join(root, HOI_CODEX_DST)
    d = dict(luc=now().isoformat(timespec='seconds'), script=script_commit(),
             hop_thu=dict(readme_sha256=sha256(os.path.join(hop, 'README.md')),
                          dong=[dict(file=r[0].strip('`'), nguon=r[1], commit=r[2].strip('`'), sha16=r[3].strip('`'),
                                     ngay=r[4]) for r in rows]),
             tro_ly_claude=dict(phien=a.phien or [], claude_code=_tool_version(['claude', '--version'])),
             tro_ly_codex=dict(cli=codex_cli_version(), lan_goi=codex_calls(root)),
             rao=rao, nhiem_thong_tin=nhiem(root, a.nhiem), nhat_ky=nk,
             cong_cu=dict(hoi_codex_sha256=sha256(hc) if os.path.isfile(hc) else None),
             truoc=dict(file=os.path.basename(a.truoc), sha256=sha256(a.truoc)) if a.truoc else None,
             loi=bad + bad2 + bad3)
    with open(a.out, 'x', encoding='utf-8') as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
    for x in d['loi']:
        print('LỖI:', x)
    print(f'{a.out} {sha256(a.out)[:16]}; {len(rows)} dòng bảng; nhiễm {d["nhiem_thong_tin"]["co"]}')
    return 2 if d['loi'] else 0


def codex_calls(root):
    """model and effort of every Codex call, read from the log header (never from the configuration)."""
    out = []
    d = os.path.join(root, 'CODEX')
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if n.startswith('log_') and n.endswith('.txt'):
            with open(os.path.join(d, n), encoding='utf-8', errors='replace') as f:
                t = f.read(4000).split('\n--------\n')
            head = t[1] if len(t) > 1 else ''
            m = re.search(r'^model: (.+)$', head, re.M)
            e = re.search(r'^reasoning effort: (.+)$', head, re.M)
            out.append(dict(log=n, model=m.group(1).strip() if m else None, effort=e.group(1).strip() if e else None))
    return out


def kiem_c13(a):
    root = a.stra_a
    hop = os.path.join(root, HOP_THU_DIR)
    res = []
    _, bad = rao_check(root)
    res.append(('1 thử rào có, đúng bản đã duyệt; 3 SHA khóa', not bad, '; '.join(bad)))
    vc, vx = _tool_version(['claude', '--version']), codex_cli_version()
    res.append(('2 phiên bản bằng lúc thử rào', (vc, vx) == (RAO_VERSIONS['claude'], RAO_VERSIONS['codex']),
                f'claude {vc}, codex {vx}; khác thì chạy lại thử rào'))
    cfg = os.path.join(os.path.expanduser('~'), '.codex', 'config.toml')
    trust = os.path.isfile(cfg) and "[projects.'D:\\STRA_A']" in open(cfg, encoding='utf-8', errors='replace').read()
    res.append(('3b dòng tin cậy D:\\STRA_A trong ~/.codex/config.toml', trust, 'có' if trust else 'không'))
    _, bb = box_check(hop)
    res.append(('4 hộp thư chỉ có file được phép, đủ file', not bb, '; '.join(bb)))
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
    try:                                                                   # SA-20: parsed, no error, current
        pr = declaration_problems(read_ke_khai(a.ke_khai), root)
        okk, note = not pr, '; '.join(pr) + ('; chạy ke-khai lại' if pr else '')
    except (Refused, OSError, yaml.YAMLError) as exc:
        okk, note = False, str(exc)
    res.append(('7 hop_thu_A.yaml mới nhất, không lỗi', okk, note))
    mp = os.path.join(hop, 'mien_A.csv')
    okm, note = False, ''
    try:
        js = read_json(os.path.join(os.path.dirname(a.mien_a), 'mien_A.json'))
        run = os.path.join(THESIS, js['s03_run'])                         # SA-20: from the S03 run, not the JSON
        rows_again = mien_rows(run)                                        # SA4-2: files checked before any hash
        m = s03_run_ok(run)
        src = all(sha256(os.path.join(run, n)) == js['sha256'][n] for n in S03_PUBLIC)    # v8.4: the check file too
        with open(a.mien_a, 'rb') as f:
            again = _mien_csv_bytes(rows_again) == f.read().replace(b'\r\n', b'\n')
        same = os.path.isfile(mp) and sha256(mp) == sha256(a.mien_a)
        box = all(os.path.isfile(os.path.join(hop, n)) and sha256(os.path.join(hop, n)) == sha256(os.path.join(run, n))
                  for n in ('sweep.csv', 'levels.csv'))
        lock_day = dt.date.fromisoformat(git(THESIS, 'show', '-s', '--format=%cs', a.spec_commit))
        run_day = m.get('date')
        run_day = run_day if isinstance(run_day, dt.date) else dt.date.fromisoformat(str(run_day))
        after = run_day >= lock_day
        okm = src and again and same and box and after
        note = (f'nguồn S03 khớp {src}, tính lại khớp {again}, hộp thư = meta A {same}, sweep/levels hộp thư cùng '
                f'lượt S03 {box}, S03 {run_day} sau khóa {lock_day} {after}')
    except (Refused, OSError, KeyError, ValueError, TypeError) as exc:
        note = str(exc)
    res.append(('8 miền A: bằng meta.yaml A, cùng lượt S03 với hộp thư, S03 chạy sau khóa S05', okm, note))
    for k, ok, n in res:
        print(f'{"ĐẠT " if ok else "CHƯA"} {k}' + (f' ({n})' if n else ''))
    return 0 if all(ok for _, ok, _ in res) else 2


def cai_hoi_codex(a):
    dst = os.path.join(a.stra_a, HOI_CODEX_DST)
    tmp = dst + '.tmp_cai'
    shutil.copyfile(HOI_CODEX_SRC, tmp)
    if sha256(tmp) != sha256(HOI_CODEX_SRC):
        os.remove(tmp)
        raise Refused('hoi_codex.py: copy SHA differs')
    if os.path.exists(dst):
        os.rename(dst, os.path.join(a.stra_a, os.path.basename(cu_name(dst))))
    os.rename(tmp, dst)
    print(f'{dst} {sha256(dst)}')
    return 0


def dong_goi(a):
    fr = frozen_campaigns(a.freeze)
    miss = [c for c in CAMPAIGNS_A if c not in fr]
    if miss:
        raise Refused(f'{miss} not frozen yet: pack at the end of A only')
    pr = declaration_problems(read_ke_khai(a.ke_khai), a.stra_a)          # SA-22: the last declaration, current
    if pr:
        raise Refused('; '.join(pr) + ': run ke-khai again (with --truoc) first')
    if os.path.exists(a.dich):
        raise Refused(f'{a.dich} exists: never overwritten')
    tr = a.transcript or []
    if len({os.path.basename(t).casefold() for t in tr}) != len(tr):
        raise Refused('two transcripts with one name')
    stage = a.dich + '.tam_dong_goi'
    if os.path.exists(stage):
        raise Refused(f'{stage} left by a failed pack: look, then remove it by hand')
    try:
        rows = []
        for d in ('NHAT_KY', 'CODEX', HOP_THU_DIR):
            shutil.copytree(os.path.join(a.stra_a, d), os.path.join(stage, d))
        for t in tr:
            os.makedirs(os.path.join(stage, 'transcript'), exist_ok=True)
            shutil.copyfile(t, os.path.join(stage, 'transcript', os.path.basename(t)))
        src_of = {f'transcript/{os.path.basename(t)}': t for t in tr}
        for dp, _, fn in os.walk(stage):
            for n in sorted(fn):
                p = os.path.join(dp, n)
                rel = os.path.relpath(p, stage).replace('\\', '/')
                src = src_of.get(rel) or os.path.join(a.stra_a, *rel.split('/'))
                if sha256(src) != sha256(p):
                    raise Refused(f'{rel}: copy SHA differs')
                rows.append((rel, os.path.getsize(p), sha256(p)))
        with open(os.path.join(stage, 'SHA256SUMS.csv'), 'x', newline='', encoding='utf-8') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['duong_dan', 'co', 'sha256'])
            w.writerows(sorted(rows))
        os.rename(stage, a.dich)
    except BaseException:
        if os.path.exists(stage):
            shutil.rmtree(stage)
        raise
    print(f'{a.dich}: {len(rows)} files')
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--stra-a', default=STRA_A, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('luot', allow_abbrev=False)
    for k in ('--run-uid', '--tuning-log', '--freeze', '--esp32', '--score', '--rows', '--frames', '--track'):
        p.add_argument(k, required=True)
    p.add_argument('--segments')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--ghi-chu-file')
    g.add_argument('--ghi-chu')
    p.add_argument('--thay', action='store_true')
    p.add_argument('--hop-thu')
    p = sub.add_parser('them', allow_abbrev=False)
    p.add_argument('--loai', required=True, choices=('spec', 's03_quet', 'mien_A', 'tai_lieu'))
    p.add_argument('--nguon', required=True, nargs='+')
    p.add_argument('--ly-do', required=True)
    p.add_argument('--tuning-log')
    p.add_argument('--thay', action='store_true')
    p.add_argument('--hop-thu')
    p = sub.add_parser('mien', allow_abbrev=False)
    p.add_argument('--s03-run', required=True)
    p.add_argument('--tuning-log', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--thay', action='store_true')
    p = sub.add_parser('ke-khai', allow_abbrev=False)
    p.add_argument('--out', required=True)
    p.add_argument('--truoc', help='previous hop_thu_A.yaml (append-only check)')
    p.add_argument('--lan-dau', action='store_true', help='first declaration (no previous one)')
    p.add_argument('--nhiem', action='append')
    p.add_argument('--phien', action='append', help='STRA_A session: id,model,effort')
    p = sub.add_parser('kiem-c13', allow_abbrev=False)
    p.add_argument('--spec-commit', required=True)
    p.add_argument('--ke-khai', required=True)
    p.add_argument('--mien-a', required=True, help='mien_A.csv named in meta.yaml of A')
    sub.add_parser('cai-hoi-codex', allow_abbrev=False)
    p = sub.add_parser('dong-goi', allow_abbrev=False)
    p.add_argument('--dich', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--ke-khai', required=True)
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
    except (OSError, ValueError, KeyError, csv.Error, yaml.YAMLError, subprocess.SubprocessError) as exc:  # SA-23
        print(f'LỖI: {type(exc).__name__}: {exc}')
        ma = 2
    print(f'#ket_qua,ma={ma}')
    return ma


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
