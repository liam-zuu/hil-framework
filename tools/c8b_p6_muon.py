#!/usr/bin/env python3
"""c8b_p6_muon: the P6 rieng (ii) file of a C8 session that cannot compute its own, BORROWED from other sessions.

Why (L172, Liam 25/11 ~23:45 at CODE local_4932073c, history_CODE.md line L172; rule fixed by CODE the same night):
c8b_p6_ngang needs at least two runs of one camera session (L164). A session with ONE run has no P6 rieng (ii) of
its own, so c8b_khoa_v5 cannot lock it. When such a session must enter the lock, its delta_h_dong of each band
B0 .. B3 is taken as the LARGEST delta_h_m of that band over the P6 files of the other sessions that have the band
(trang_thai dat). This is an assumption, not a measurement of the session: the file says so in its block 'muon', and
c8b_khoa_v5 repeats it in hang_theo_u_ly_do of the session lock.

The output keeps the schema c8b_P6_rieng_ii_v2 (c8b_khoa_v5 --p6-dong reads dai.<B>.delta_h_m, trang_thai) plus
  muon   {version c8b_p6_muon_v1, quy_tac 'max_theo_dai', quyet_dinh (the decision, words of the caller), ly_do,
          session_nhan_sha256 [the session.json of the session that RECEIVES the numbers],
          nguon_p6 [{file, duong_dan, sha256, session_sha256, trang_thai, dai {B: {trang_thai, delta_h_m}}}]}
  dai    B0 .. B3: delta_h_m = max over the sources with the band dat, muon_tu_sha256 (the source of the max),
         n_nguon; no source with the band -> thieu, null (the lock then stops at that band as for any P6 file)
  no 'luot', no source of role 'session': nothing here was counted on frames of the receiving session.
c8b_khoa_v5 accepts such a file ONLY through kiem(): the block is well formed, the file was made for exactly the
--session files of that lock and with its --nguong, every source file is still at its path with its sha256, is not
itself borrowed, is of another session, and dai is exactly what gop() gives from them.

  python tools/c8b_p6_muon.py --p6 run02/..P6.json --p6 run03/..P6.json --session-nhan run06/../session.json \
      --nguong nguong_khoa_c8_v1.json --quyet-dinh "L172 ..." [--ly-do "..."] --out c8b_P6_rieng_ii_v2_muon.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_p6_muon_v1'
P6_VERSION = 'c8b_P6_rieng_ii_v2'        # the same string as c8b_khoa_v5.P6_VERSION (test)
BANDS = ('B0', 'B1', 'B2', 'B3')         # B0 standing + c8b_khoa_v5.P6_BANDS (test)
QUY_TAC = 'max_theo_dai'
REL = 1e-12


class MuonError(ValueError):
    pass


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _hex64(x):
    return isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def _sha_of(p6, vai):
    return {n.get('sha256') for n in (p6.get('nguon') or []) if isinstance(n, dict) and n.get('vai') == vai}


def doc_nguon(d, ten):
    """One source: a c8b_P6_rieng_ii_v2 file computed by c8b_p6_ngang on its own session (not a borrowed one), with
    its session and threshold sources named. Returns (session sha set, nguong sha set, {band: (state, delta, lo,
    hi)}); a band absent from the file is ('thieu', None, None, None)."""
    if not isinstance(d, dict) or d.get('schema') != P6_VERSION:
        raise MuonError(f'{ten}: not a {P6_VERSION} file')
    if 'muon' in d:
        raise MuonError(f'{ten}: a borrowed P6 file (block muon) is not a source')
    ses, ng = _sha_of(d, 'session'), _sha_of(d, 'nguong')
    if not ses or not all(_hex64(s) for s in ses):
        raise MuonError(f'{ten}: no source of role session (sha256): the session of the file is not known')
    if len(ng) != 1 or not all(_hex64(s) for s in ng):
        raise MuonError(f'{ten}: not exactly one source of role nguong')
    dai = d.get('dai')
    if not isinstance(dai, dict):
        raise MuonError(f'{ten}: no dai')
    out = {}
    for b in BANDS:
        x = dai.get(b)
        if not isinstance(x, dict) or x.get('trang_thai') != 'dat':
            if isinstance(x, dict) and x.get('trang_thai') not in ('thieu', None):
                raise MuonError(f'{ten}: band {b} trang_thai {x.get("trang_thai")!r}')
            out[b] = ('thieu', None, None, None)
            continue
        v, lo, hi = _num(x.get('delta_h_m')), _num(x.get('u_lo_m_s')), _num(x.get('u_hi_m_s'))
        if v is None or v < 0.0 or lo is None or hi is None or not lo < hi:
            raise MuonError(f'{ten}: band {b} dat without delta_h_m >= 0 and its edges u_lo_m_s < u_hi_m_s')
        out[b] = ('dat', v, lo, hi)
    return ses, ng, out


def gop(nguon):
    """nguon: [(sha256 of the file, parsed file)], at least one, distinct files of distinct sessions made with one
    threshold file. Returns (dai, rows of muon.nguon_p6 without file / duong_dan, nguong sha): per band the largest
    delta_h_m of the sources that have it; the edges of a band must be the same in every source that has it."""
    if not nguon:
        raise MuonError('no source P6 file')
    shas = [s for s, _ in nguon]
    if len(set(shas)) != len(shas) or not all(_hex64(s) for s in shas):
        raise MuonError('the source P6 files must be distinct (sha256)')
    rows, seen, ngs = [], set(), set()
    for s, d in nguon:
        ses, ng, bands = doc_nguon(d, f'source {s[:12]}')
        if ses & seen:
            raise MuonError(f'source {s[:12]}: a session of this file is already in another source')
        seen |= ses
        ngs |= ng
        rows.append((s, ses, d.get('trang_thai'), bands))
    if len(ngs) != 1:
        raise MuonError(f'the sources were made with different nguong files: {sorted(ngs)}')
    dai = {}
    for b in BANDS:
        have = [(bands[b][1], s, bands[b][2], bands[b][3]) for s, _, _, bands in rows if bands[b][0] == 'dat']
        if not have:
            dai[b] = dict(delta_h_m=None, trang_thai='thieu', n_nguon=0, muon_tu_sha256=None)
            continue
        lo, hi = have[0][2], have[0][3]
        if any(abs(x[2] - lo) > REL * max(1.0, abs(lo)) or abs(x[3] - hi) > REL * max(1.0, abs(hi)) for x in have):
            raise MuonError(f'band {b}: the sources do not have the same edges u_lo_m_s, u_hi_m_s')
        v, s = max((x[0], x[1]) for x in have)
        dai[b] = dict(delta_h_m=v, trang_thai='dat', n_nguon=len(have), muon_tu_sha256=s, u_lo_m_s=lo, u_hi_m_s=hi)
    tom = [dict(sha256=s, session_sha256=sorted(ses), trang_thai=st,
                dai={b: dict(trang_thai=bands[b][0], delta_h_m=bands[b][1]) for b in BANDS})
           for s, ses, st, bands in rows]
    return dai, tom, next(iter(ngs))


def build(p6_paths, session_paths, nguong_path, quyet_dinh, ly_do=None):
    if not isinstance(quyet_dinh, str) or not quyet_dinh.strip():
        raise MuonError('--quyet-dinh: the decision that allows the borrowing must be written')
    if not session_paths:
        raise MuonError('--session-nhan: the session.json of the receiving session')
    nguon, files = [], []
    for p in p6_paths:
        with open(p, encoding='utf-8') as f:
            nguon.append((cu.sha256(p), json.load(f)))
        files.append(dict(file=os.path.basename(p), duong_dan=os.path.abspath(p)))
    dai, tom, ng = gop(nguon)
    nhan = sorted({cu.sha256(p) for p in session_paths})
    if len(nhan) != len(session_paths):
        raise MuonError('--session-nhan: the same file twice')
    for t in tom:
        if set(t['session_sha256']) & set(nhan):
            raise MuonError(f'source {t["sha256"][:12]}: it is a P6 file of the receiving session itself')
    if cu.sha256(nguong_path) != ng:
        raise MuonError('--nguong: not the threshold file the sources were made with')
    thieu = [b for b in BANDS if dai[b]['trang_thai'] != 'dat']
    src = [dict(vai='p6_muon', sha256=t['sha256'], **f) for t, f in zip(tom, files)]
    src.append(dict(vai='nguong', file=os.path.basename(nguong_path), duong_dan=os.path.abspath(nguong_path),
                    sha256=ng))
    src += [dict(vai='session_nhan', file=os.path.basename(p), duong_dan=os.path.abspath(p), sha256=cu.sha256(p))
            for p in session_paths]
    return dict(schema=P6_VERSION,
                muon=dict(version=VERSION, quy_tac=QUY_TAC, quyet_dinh=quyet_dinh.strip(), ly_do=ly_do,
                          session_nhan_sha256=nhan, nguon_p6=[dict(f, **t) for t, f in zip(tom, files)]),
                dai=dai, trang_thai='thieu' if thieu else 'dat',
                ly_do=(f'khong buoi nguon nao co dai {thieu}' if thieu else None), nguon=src)


def kiem(p6, session_sha, nguong_sha):
    """The check of c8b_khoa_v5 --p6-dong on a file with the block muon. session_sha, nguong_sha: the sha256 sets of
    the --session and --nguong files of that lock. Raises MuonError; returns the block when all holds."""
    m = p6.get('muon') if isinstance(p6, dict) else None
    if not isinstance(m, dict) or p6.get('schema') != P6_VERSION or m.get('version') != VERSION \
            or m.get('quy_tac') != QUY_TAC:
        raise MuonError(f'block muon is not a {VERSION} block (quy_tac {QUY_TAC}) of a {P6_VERSION} file')
    if not isinstance(m.get('quyet_dinh'), str) or not m['quyet_dinh'].strip():
        raise MuonError('muon.quyet_dinh: the decision is not written')
    if 'luot' in p6 or _sha_of(p6, 'session'):
        raise MuonError('a borrowed file names no run and no source of role session')
    nhan = m.get('session_nhan_sha256')
    if not session_sha or not isinstance(nhan, list) or set(nhan) != set(session_sha) or len(set(nhan)) != len(nhan):
        raise MuonError('muon.session_nhan_sha256 is not exactly the --session files of this lock: the file was '
                        'made for another session (or the lock names no --session)')
    rows = m.get('nguon_p6')
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise MuonError('muon.nguon_p6: no source')
    nguon = []
    for r in rows:
        p, s = r.get('duong_dan'), r.get('sha256')
        if not isinstance(p, str) or not _hex64(s):
            raise MuonError('muon.nguon_p6: duong_dan / sha256')
        try:
            if cu.sha256(p) != s:
                raise MuonError(f'source {p}: the file changed since the borrowing (sha256)')
            with open(p, encoding='utf-8') as f:
                nguon.append((s, json.load(f)))
        except (OSError, ValueError) as exc:
            if isinstance(exc, MuonError):
                raise
            raise MuonError(f'source {p}: cannot be read again ({exc})')
    dai, tom, ng = gop(nguon)
    if not nguong_sha or set(nguong_sha) != {ng} or _sha_of(p6, 'nguong') != {ng}:
        raise MuonError('the sources were not made with the --nguong file of this lock')
    for t in tom:
        if set(t['session_sha256']) & set(session_sha):
            raise MuonError(f'source {t["sha256"][:12]}: a P6 file of the session of this lock itself')
    if p6.get('dai') != dai:
        raise MuonError('dai is not the largest delta_h_m per band of the sources named')
    if [{k: r.get(k) for k in t} for r, t in zip(rows, tom)] != tom:
        raise MuonError('muon.nguon_p6 does not say what the source files hold')
    return m


def ghi_chu(p6):
    """The sentence c8b_khoa_v5 adds to hang_theo_u_ly_do of a session lock made with a borrowed file."""
    m = p6['muon']
    n = [str(r.get('sha256'))[:12] for r in m.get('nguon_p6') or [] if isinstance(r, dict)]
    return (f'P6 rieng (ii) MUON ({m.get("version")}, quyet dinh: {m.get("quyet_dinh")}): buoi nay khong tinh duoc P6 '
            f'cua chinh no; delta_h_dong cua moi dai B0..B3 (ca B0 trong hang_m) = lon nhat theo dai cua {len(n)} file '
            f'P6 buoi khac (sha256 {", ".join(n)}); la gia dinh, khong phai so do cua buoi nay')


def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_p6_muon.py'], capture_output=True,
                           text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), file='tools/c8b_p6_muon.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_p6_muon.py')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--p6', action='append', required=True, help='a c8b_P6_rieng_ii_v2 of another session (repeat)')
    ap.add_argument('--session-nhan', action='append', required=True,
                    help='session.json of the session that receives the numbers (repeat: every --session of its lock)')
    ap.add_argument('--nguong', required=True)
    ap.add_argument('--quyet-dinh', required=True, help='the decision that allows it (who, when, where, words)')
    ap.add_argument('--ly-do', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        out = build(a.p6, a.session_nhan, a.nguong, a.quyet_dinh, a.ly_do)
        out['script'] = script_state()
        out['luc'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False))
    except (MuonError, cu.InputError, OSError, ValueError) as exc:
        print(f'c8b_p6_muon: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print(f'c8b_p6_muon: {a.out}: ' + ', '.join(
        f'{b} {out["dai"][b]["delta_h_m"]!r} ({out["dai"][b]["n_nguon"]} nguon)' for b in BANDS)
          + f'; trang_thai {out["trang_thai"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
