"""c7_ghep_cua_so.py: join the per-run F2 window files (cua_so_v1.jsonl) of one C8 session into ONE file, in recording
order (CODE 20/11, `03_CODE/docs/2026-11-20_Trinh_tu_chuoi_C8_sau_buoi2_v1.md` 2.4; NHACTAM LLC1-15).

`c7_score cua-so` (pre/post of a car run) and `c8_hc_a cua_so` (12b/E of a HC attempt) each write one
cua_so_v1.jsonl; `c7_nhac_tam hieu_chuan`, `lan_lap`, `c7_kiem_cheo meo_E`, `c7_score tham-chieu` read one --cua-so.
The readers take JSON lines, so the join is a concatenation of the rows (each raw line copied byte for byte, schema
cua_so_v1 kept); the order of the files is the recording order of their clips in thu_tu_clip.csv.

  python c7_ghep_cua_so.py --thu-tu thu_tu_clip.csv --manifest video/manifest_usb.csv
      --cua-so <file> [<file> ...] [--bo KEY=LY_DO ...] [--doan TEN=<session.json>@<buoc H dau> ...]
      --out <dir>/cua_so_v1.jsonl

Several camera segments (CODE 20/11 khuya, chain A "da session"; C8 session 2 had three: the iPhone was taken off and
put back twice): --doan TEN=<session.json>@<buoc> per segment, <buoc> the thu_tu_clip.csv step of the clip H that
opens it (H0a, H2a, H3a). A row of thu_tu_clip.csv belongs to the last segment opened at or above it; every window row
of a file must carry the session_sha256 of ITS segment (sha256 of that session.json); a required row above the first
segment is refused. Without --doan: one session in the whole file, as before.

Binding: each file -> the clip_sha256 of its rows (one clip per file) -> the clip name in manifest_usb.csv -> the
thu_tu_clip.csv row of that clip. Rows of thu_tu_clip.csv that need a window file ("required"): `buoc` a run code
C8-<P|L>-<F02|F05|F08|K1|K2>-<n>[_lai<n>] or a HC place HC_<goc|K1|K2>. Key of a required row = `thu_id` when set,
else `buoc`; it must equal the luot_id of the rows of its file (c7_score cua-so --luot-id <run code>; c8_hc_a sets
luot_id = thu_id = the attempt folder HC_<cho>_<n>).

Refused (exit 2, nothing written): a required row without a file and not given in --bo; a --bo of a key that is not
a required row, or that has a file; two files of one clip, a file whose clip is in no required row or in several; a
file with rows of several clips, rows without luot_id / clip_sha256 / session_sha256, another schema, or a luot_id not the key of its
row; the same luot_id in two files; clips not in recording order (created_utc of the manifest decreasing along
thu_tu_clip.csv); rows of several session_sha256 (one session per session: the H clip at the start, guide 33); the
order check of the readers failing on the joined rows (inside each clip the first PTS of every window increases with
the raw row order, and post after pre of a run: NHACTAM LLC1-15, else every run gets thu_tu_mau_thuan). With --doan:
a --doan whose <buoc> is not exactly one thu_tu_clip.csv row, two segments opened on one row, a repeated name or
session file, a required row above the first segment, and a file whose session_sha256 is not the one of the segment
of its clip.

Writes --out (JSON lines) and <out>.nguon.json (schema cua_so_ghep_v1: each source file with path, sha256, row count,
its thu_tu_clip.csv line, session_sha256 and segment; the --bo list; the segments (doan: name, session path
and sha256, opening step and line; [] without --doan); sha256 of thu_tu_clip.csv, manifest, output; commit_hil). Last line
`#ket_qua,ma=0|2`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys

SCHEMA, SCHEMA_GHEP = 'cua_so_v1', 'cua_so_ghep_v1'
RE_LUOT = re.compile(r'^C8-[PL]-(F02|F05|F08|K1|K2)-\d+(_lai\d+)?$')
RE_HC = re.compile(r'^HC_(goc|K1|K2)$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


class GhepError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def commit_hil():
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', d, 'status', '--porcelain', '--', '.'], capture_output=True, text=True,
                               check=True).stdout.strip()
        return dict(sha=sha, dirty=bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return dict(sha=None, dirty=None)


def is_required(buoc):
    return bool(RE_LUOT.match(buoc) or RE_HC.match(buoc))


def read_thu_tu(path):
    """Rows of thu_tu_clip.csv with their 1-based line number; needs the columns clip, buoc, thu_id."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        miss = {'clip', 'buoc', 'thu_id'} - set(rd.fieldnames or [])
        if miss:
            raise GhepError(f'{path}: thieu cot {sorted(miss)}')
        rows = []
        for n, r in enumerate(rd, start=2):
            r = {k: (v or '').strip() for k, v in r.items() if k is not None}
            r['_dong'] = n
            rows.append(r)
    return rows


def read_manifest(path):
    """{clip name: (sha256, created_utc)} of manifest_usb.csv (src_name, sha256, created_utc)."""
    out = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        miss = {'src_name', 'sha256', 'created_utc'} - set(rd.fieldnames or [])
        if miss:
            raise GhepError(f'{path}: thieu cot {sorted(miss)}')
        for r in rd:
            name, sha = (r['src_name'] or '').strip(), (r['sha256'] or '').strip().lower()
            if not name:
                continue
            if not HEX64.match(sha):
                raise GhepError(f'{path}: {name}: sha256 khong hop le')
            v = (sha, (r['created_utc'] or '').strip())
            if name in out and out[name][0] != sha:
                raise GhepError(f'{path}: {name} co hai sha256 khac nhau')
            out[name] = v
    return out


def read_rows(path):
    """[(raw line without end of line, row dict)] of a JSON lines window file."""
    with open(path, 'rb') as f:
        data = f.read()
    try:
        txt = data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise GhepError(f'{path}: khong phai utf-8 ({e})')
    out = []
    for k, ln in enumerate(txt.splitlines(), start=1):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError as e:
            raise GhepError(f'{path} dong {k}: khong phai JSON lines ({e})')
        if not isinstance(r, dict):
            raise GhepError(f'{path} dong {k}: khong phai mot cua so (dict)')
        out.append((ln, r))
    if not out:
        raise GhepError(f'{path}: rong')
    return out


def first_pts(w):
    """min PTS of the frames of a window (NHACTAM khung_cua; an empty or null list -> None)."""
    pts = []
    for k in w.get('khung') or []:
        p = float(k['pts_s'])
        if not math.isfinite(p):
            raise GhepError(f'luot {w.get("luot_id")} {w.get("loai")}: pts_s khong huu han')
        pts.append(p)
    return min(pts) if pts else None


def check_order(rows):
    """NHACTAM LLC1-15 on the joined rows: per clip, the first PTS of each window strictly increases with the raw row
    order; the post of a run starts after its pre when both are on one clip. Returns a list of problems."""
    sai, by_clip, t0, prepost = [], {}, {}, {}
    for i, w in enumerate(rows):
        p = first_pts(w)
        if p is not None:
            t0[i] = p
            by_clip.setdefault(w['clip_sha256'], []).append((i, p))
        if w.get('loai') in ('pre', 'post'):
            prepost.setdefault(w['luot_id'], {})[w['loai']] = i
    for c, v in by_clip.items():
        for (ia, pa), (ib, pb) in zip(v, v[1:]):
            if pb <= pa:
                sai.append(f'clip {c[:12]}: dong {ib} (pts {pb}) khong sau dong {ia} (pts {pa})')
    for lid, d in prepost.items():
        a, b = d.get('pre'), d.get('post')
        if a in t0 and b in t0 and rows[a]['clip_sha256'] == rows[b]['clip_sha256'] and t0[b] <= t0[a]:
            sai.append(f'luot {lid}: post khong sau pre')
    return sai


def parse_doan(specs, tt):
    """[{ten, session, sha256, buoc_dau, pos, dong}] of --doan TEN=<session.json>@<buoc>, in thu_tu_clip.csv order."""
    out, names, shas = [], set(), set()
    for s in specs:
        ten, sep, rest = s.partition('=')
        path, sep2, buoc = rest.rpartition('@')
        ten, path, buoc = ten.strip(), path.strip(), buoc.strip()
        if not sep or not sep2 or not ten or not path or not buoc:
            raise GhepError(f'--doan {s!r}: can TEN=<session.json>@<buoc clip H dau doan>')
        if ten in names:
            raise GhepError(f'--doan {ten}: dua hai lan')
        if not os.path.isfile(path):
            raise GhepError(f'--doan {ten}: khong co file session {path}')
        sha = sha256_file(path)
        if sha in shas:
            raise GhepError(f'--doan {ten}: session {path} da la session cua doan khac')
        pos = [i for i, r in enumerate(tt) if r['buoc'] == buoc]
        if len(pos) != 1:
            raise GhepError(f'--doan {ten}: buoc {buoc!r} o {len(pos)} dong cua thu_tu_clip.csv (can dung 1)')
        names.add(ten)
        shas.add(sha)
        out.append(dict(ten=ten, session=os.path.abspath(path), sha256=sha, buoc_dau=buoc, pos=pos[0],
                        dong=tt[pos[0]]['_dong']))
    out.sort(key=lambda d: d['pos'])
    for a, b in zip(out, out[1:]):
        if a['pos'] == b['pos']:
            raise GhepError(f'--doan {a["ten"]}, {b["ten"]}: cung mot dong mo doan')
    return out


def doan_of_row(doans, pos):
    """The segment of the thu_tu_clip.csv row at index pos (the last one opened at or above it), or None."""
    d = None
    for x in doans:
        if x['pos'] <= pos:
            d = x
    return d


def ghep(thu_tu_path, manifest_path, files, bo, out_path, doan=()):
    if os.path.exists(out_path):
        raise GhepError(f'{out_path} da co: dung file moi')
    nguon_path = out_path + '.nguon.json'
    if os.path.exists(nguon_path):
        raise GhepError(f'{nguon_path} da co: dung file moi')
    tt = read_thu_tu(thu_tu_path)
    man = read_manifest(manifest_path)
    doans = parse_doan(doan, tt)
    pos = {id(r): i for i, r in enumerate(tt)}
    req = [r for r in tt if is_required(r['buoc'])]
    if doans:
        tren = [r['buoc'] for r in req if doan_of_row(doans, pos[id(r)]) is None]
        if tren:
            raise GhepError(f'luot truoc doan dau tien (khong co session): {tren}')
    key_of = {id(r): (r['thu_id'] or r['buoc']) for r in req}
    keys = [key_of[id(r)] for r in req]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise GhepError(f'thu_tu_clip.csv: khoa luot trung {dup}')
    bo_map = {}
    for s in bo:
        k, sep, why = s.partition('=')
        if not sep or not k.strip() or not why.strip():
            raise GhepError(f'--bo {s!r}: can KEY=LY_DO')
        k = k.strip()
        if k not in keys:
            raise GhepError(f'--bo {k}: khong phai luot can file trong thu_tu_clip.csv')
        if k in bo_map:
            raise GhepError(f'--bo {k}: dua hai lan')
        bo_map[k] = why.strip()
    sha_of_clip = {}
    for r in req:
        if r['clip'] not in man:
            if key_of[id(r)] in bo_map:
                continue
            raise GhepError(f'thu_tu_clip.csv dong {r["_dong"]} ({key_of[id(r)]}): clip {r["clip"]!r} khong co '
                            f'trong manifest')
        sha_of_clip.setdefault(man[r['clip']][0], []).append(r)
    src, seen_luot, by_row, sess = [], {}, {}, set()
    for p in files:
        rr = read_rows(p)
        clips = {w.get('clip_sha256') for _, w in rr}
        if len(clips) != 1 or not isinstance(next(iter(clips)), str) or not HEX64.match(next(iter(clips))):
            raise GhepError(f'{p}: can dung mot clip_sha256 hop le cho moi dong (co {sorted(map(str, clips))})')
        csha = next(iter(clips))
        for k, (_, w) in enumerate(rr, start=1):
            if w.get('schema', SCHEMA) != SCHEMA:
                raise GhepError(f'{p} dong {k}: schema {w.get("schema")!r} khong phai {SCHEMA}')
            if not isinstance(w.get('luot_id'), str) or not w['luot_id']:
                raise GhepError(f'{p} dong {k}: thieu luot_id (c7_score cua-so can --luot-id <ma luot>)')
            if 'loai' not in w:
                raise GhepError(f'{p} dong {k}: thieu loai')
            if not isinstance(w.get('session_sha256'), str) or not HEX64.match(w['session_sha256']):
                raise GhepError(f'{p} dong {k}: thieu session_sha256 (c7_score cua-so can --session; NHACTAM doi)')
            sess.add(w['session_sha256'])
        lids = {w['luot_id'] for _, w in rr}
        if len(lids) != 1:
            raise GhepError(f'{p}: nhieu luot_id trong mot file {sorted(lids)}')
        lid = next(iter(lids))
        rows_tt = sha_of_clip.get(csha, [])
        if len(rows_tt) != 1:
            raise GhepError(f'{p}: clip {csha[:12]} o {len(rows_tt)} dong luot cua thu_tu_clip.csv (can dung 1)')
        r = rows_tt[0]
        k = key_of[id(r)]
        if k in bo_map:
            raise GhepError(f'{p}: luot {k} vua co file vua o --bo')
        if lid != k:
            raise GhepError(f'{p}: luot_id {lid!r} khac khoa {k!r} cua thu_tu_clip.csv dong {r["_dong"]}')
        if id(r) in by_row:
            raise GhepError(f'{p}: luot {k} trung voi {by_row[id(r)]["duong_dan"]}')
        fses = {w['session_sha256'] for _, w in rr}
        d = doan_of_row(doans, pos[id(r)]) if doans else None
        if d is not None and fses != {d['sha256']}:
            raise GhepError(f'{p}: session_sha256 {sorted(x[:12] for x in fses)} khong phai session cua doan '
                            f'{d["ten"]} ({d["sha256"][:12]}, {d["session"]}) cua clip {r["clip"]}')
        if lid in seen_luot:
            raise GhepError(f'{p}: luot_id {lid} trung voi {seen_luot[lid]}')
        seen_luot[lid] = p
        item = dict(r=r, rows=rr, duong_dan=os.path.abspath(p), sha256=sha256_file(p), clip_sha256=csha,
                    session_sha256=next(iter(fses)) if len(fses) == 1 else None, doan=d and d['ten'])
        by_row[id(r)] = item
        src.append(item)
    thieu = [key_of[id(r)] for r in req if id(r) not in by_row and key_of[id(r)] not in bo_map]
    if thieu:
        raise GhepError(f'luot trong thu_tu_clip.csv khong co file cua so (them file hay --bo KEY=LY_DO): {thieu}')
    if len(sess) > 1 and not doans:
        raise GhepError(f'nhieu session_sha256 trong buoi: {sorted(s[:12] for s in sess)} (nhieu doan camera: --doan)')
    src.sort(key=lambda it: pos[id(it['r'])])
    prev = None
    for it in src:
        cu = man[it['r']['clip']][1]
        if prev is not None and cu and prev[1] and cu < prev[1]:
            raise GhepError(f'thu tu ghi hinh nguoc: {it["r"]["clip"]} ({cu}) sau {prev[0]} ({prev[1]}) trong '
                            f'thu_tu_clip.csv')
        prev = (it['r']['clip'], cu)
    rows = [w for it in src for _, w in it['rows']]
    sai = check_order(rows)
    if sai:
        raise GhepError('thu tu cua so sai (LLC1-15): ' + '; '.join(sai))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'xb') as f:
        for it in src:
            for ln, _ in it['rows']:
                f.write(ln.encode('utf-8') + b'\n')
    out_sha = sha256_file(out_path)
    nguon = dict(
        schema=SCHEMA_GHEP,
        thu_tu_clip=dict(duong_dan=os.path.abspath(thu_tu_path), sha256=sha256_file(thu_tu_path)),
        manifest=dict(duong_dan=os.path.abspath(manifest_path), sha256=sha256_file(manifest_path)),
        nguon=[dict(thu_tu_dong=it['r']['_dong'], clip=it['r']['clip'], buoc=it['r']['buoc'],
                    thu_id=it['r']['thu_id'] or None, luot_id=it['rows'][0][1]['luot_id'],
                    clip_sha256=it['clip_sha256'], duong_dan=it['duong_dan'], sha256=it['sha256'],
                    so_dong=len(it['rows']), session_sha256=it['session_sha256'], doan=it['doan']) for it in src],
        doan=[dict(ten=d['ten'], session=dict(duong_dan=d['session'], sha256=d['sha256']), buoc_dau=d['buoc_dau'],
                   thu_tu_dong=d['dong']) for d in doans],
        bo=[dict(khoa=k, ly_do=v) for k, v in bo_map.items()],
        ra=dict(duong_dan=os.path.abspath(out_path), sha256=out_sha, so_dong=len(rows)),
        commit_hil=commit_hil())
    with open(nguon_path, 'x', encoding='utf-8') as f:
        f.write(json.dumps(nguon, ensure_ascii=False, indent=1) + '\n')
    return nguon


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--thu-tu', required=True, help='thu_tu_clip.csv of the session')
    ap.add_argument('--manifest', required=True, help='manifest_usb.csv of the session (clip name -> sha256)')
    ap.add_argument('--cua-so', nargs='+', required=True, help='cua_so_v1.jsonl of each run / HC attempt')
    ap.add_argument('--bo', action='append', default=[], help='KEY=LY_DO: a required run without a window file')
    ap.add_argument('--doan', action='append', default=[],
                    help='TEN=<session.json>@<buoc>: a camera segment and the clip H step that opens it (several)')
    ap.add_argument('--out', required=True, help='joined cua_so_v1.jsonl (new file)')
    a = ap.parse_args(argv)
    try:
        n = ghep(a.thu_tu, a.manifest, a.cua_so, a.bo, a.out, a.doan)
    except (GhepError, OSError, KeyError, ValueError) as exc:
        print(f'c7_ghep_cua_so: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print(f'ghep {len(n["nguon"])} file, {n["ra"]["so_dong"]} dong, bo {len(n["bo"])}, doan {len(n["doan"])}; '
          f'sha256 {n["ra"]["sha256"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
