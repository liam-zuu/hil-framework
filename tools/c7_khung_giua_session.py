#!/usr/bin/env python3
"""c7_khung_giua_session.py: c7_khung_giua_session_v1 files (one per pair of camera sessions of a C8 session), the floor
frame difference between two clip H calibrations, model phys_v1 (CODE_KHUNG design CLAUDE_THESIS
03_CODE/docs/2026-11-21_Thiet_ke_sai_khung_giua_session_v1.md sections 1, 3, 4; Liam 20/11 23:52 at CODE_KHUNG, verbatim:
"Truong tron phys (Recommended)"; CODE gave the file to CAM 20/11 khuya).

The numbers are computed by CODE_KHUNG's estimator uoc_khung_giua_session.py (05_DATA/results/C8/<run>/
khung_giua_session/; affine 6 + projective 2 + radial cubic 1 fitted on the 32 mark centres rebuilt by both sessions;
bound = max over a disc rho around the place of |estimate| + k_se standard errors, max over every combination of clip H
at the start / end of each segment). This tool does not recompute them: it reads the estimator output (--uoc, its sha256
and the estimator's sha256 go into nguon), checks it, and writes, for every pair of scoring sessions (the session of the
clip H that opens each segment), the fields the readers use:

  schema, session_sha256 [A, B], du_m (largest |d| over the marks of the main pair, a look-up number, NOT to be turned
  into an angle), n_moc                                   <- c7_nhac_tam.doc_kgs, c8_hc_a (all files)
  mo_hinh 'phys_v1', k_se, rho_m, truoc_le true, to_hop  <- design 3
  theo_cho.<goc|K1|K2>: xy_m, du_tinh_tien_m, du_xoay_rad, du_thang, du_bien_dang_rad, xoay_bien_thien_rad,
      thang_bien_thien, hanh_lang_12b {du_xoay_rad, du_thang, du_bien_dang_rad}, uoc {omega_rad, se_omega, sigma,
      se_sigma, gamma, se_gamma}                          <- c7_nhac_tam KGS_PHYS, c8_hc_a hanh_lang_12b.du_bien_dang_rad
  phan_du (phys, main pair), nen_nhieu (two clips H back to back of each segment, phys s_du_m),
  phat_hien_ma {delta_canh_m per place, delta_canh_sd_du_m, xoay_moc_sd_rad}  <- design 5 (reported, not a frame term)

Every number before the margin (x 1.25 is the reader's). A place the estimator did not give is left out of theo_cho (the
reader: chua_ket_luan). Refused (exit 2, nothing written): a missing or empty estimator file, sessions of the estimator
not the files given (--session TEN=<session.json>, sha256), a pair without the three places, a number not finite or
negative, an output file that exists.

  python tools/c7_khung_giua_session.py --uoc <run>/khung33_v2/uoc_khung_calib33_v2.json
      --uoc-script <run>/khung_giua_session/uoc_khung_giua_session.py
      --session H0a=<run>/H0a/calib33_v2/session.json --session H2a=... --session H3a=... --out-dir <dir>
  -> <dir>/khung_giua_session_<A>_<B>.json per pair of the --session (all pairs)
Last line #ket_qua,ma=0|2.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys

SCHEMA = 'c7_khung_giua_session_v1'
MO_HINH = 'phys_v1'
CHO = ('goc', 'K1', 'K2')
N_THAM_SO = 9
KHUNG_F = ('du_tinh_tien_m', 'du_xoay_rad', 'du_thang', 'du_bien_dang_rad', 'xoay_bien_thien_rad', 'thang_bien_thien')
HL_F = ('du_xoay_rad', 'du_thang', 'du_bien_dang_rad')


class KgsError(Exception):
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


def num(v, w, lo=0.0):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < lo:
        raise KgsError(f'{w}: {v!r} (so huu han >= {lo})')
    return float(v)


def fnum(v, w):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise KgsError(f'{w}: {v!r} (so huu han)')
    return float(v)


def entries_of(uoc, a, b):
    """tong_hop entries {cho: entry} of the pair of scoring sessions (a, b) (either order in the estimator)."""
    out = {}
    for t in uoc.get('tong_hop') or []:
        if sorted(t.get('session') or []) == sorted([a, b]):
            c = t.get('cho')
            if c in out:
                raise KgsError(f'cap {a}|{b} cho {c}: hai muc tong_hop')
            out[c] = t
    return out


def nen_nhieu(uoc, doan_of):
    """phys s_du_m of the two clips H back to back of a segment (loai trong_doan_chuoi, first pair of the segment)."""
    out = []
    for c in uoc.get('cap') or []:
        if c.get('loai') != 'trong_doan_chuoi':
            continue
        ten = c.get('ten', '')
        pair = ten.split(' ')[0]
        h = pair.split('|')
        if len(h) == 2 and h[0] in doan_of and doan_of[h[0]][1:2] == [h[1]]:
            out.append(dict(cap=pair, s_du_m=num(((c.get('khop') or {}).get('phys') or {}).get('s_du_m'),
                                                 f'nen_nhieu {pair} s_du_m')))
    return out


def build(uoc, uoc_meta, sessions, a, b):
    """The c7_khung_giua_session_v1 dict of the pair (a, b)."""
    ent = entries_of(uoc, a, b)
    thieu = [c for c in CHO if c not in ent]
    if thieu:
        raise KgsError(f'cap {a}|{b}: uoc khong co cho {thieu}')
    first = ent[CHO[0]]
    order = list(first['session'])                       # the estimator's order of the pair
    sha_of = {first['session'][i]: first['session_sha256'][i] for i in range(2)}
    for t in ent.values():
        if [t['session_sha256'][t['session'].index(x)] for x in order] != [sha_of[x] for x in order]:
            raise KgsError(f'cap {a}|{b}: session_sha256 khac nhau giua cac cho')
    for x in (a, b):
        if sha_of[x] != sessions[x]['sha256']:
            raise KgsError(f'session {x}: sha256 cua uoc {sha_of[x][:12]} khac file {sessions[x]["duong_dan"]} '
                           f'({sessions[x]["sha256"][:12]})')
    xy = uoc.get('cho') or {}
    theo_cho, du_m, n_moc, to_hop = {}, [], set(), set()
    for c in CHO:
        t = ent[c]
        k = t.get('khung') or {}
        if k.get('mo_hinh') != 'phys':
            raise KgsError(f'cap {a}|{b} cho {c}: khung.mo_hinh {k.get("mo_hinh")!r} khong phai phys')
        blk = {f: num(k.get(f), f'{a}|{b} {c} {f}') for f in KHUNG_F}
        hl = k.get('hanh_lang') or {}
        blk['hanh_lang_12b'] = {f: num(hl.get(f), f'{a}|{b} {c} hanh_lang {f}') for f in HL_F}
        u = k.get('uoc_tai_cho_chinh') or {}
        blk['uoc'] = dict(omega_rad=fnum(u.get('omega'), f'{c} omega'), se_omega=num(u.get('se_omega'), f'{c} se_omega'),
                          sigma=fnum(u.get('sigma'), f'{c} sigma'), se_sigma=num(u.get('se_sigma'), f'{c} se_sigma'),
                          gamma=num(u.get('gamma'), f'{c} gamma'), se_gamma=num(u.get('se_gamma'), f'{c} se_gamma'))
        p = xy.get(c)
        blk['xy_m'] = [fnum(p[0], f'{c} x'), fnum(p[1], f'{c} y')] if isinstance(p, (list, tuple)) and len(p) == 2 \
            else None
        theo_cho[c] = blk
        du_m.append(num(t.get('du_m'), f'{a}|{b} {c} du_m'))
        n_moc.add(t.get('n_moc'))
        to_hop.update(t.get('to_hop') or [])
    if len(n_moc) != 1 or not isinstance(next(iter(n_moc)), int) or next(iter(n_moc)) < 1:
        raise KgsError(f'cap {a}|{b}: n_moc {sorted(map(str, n_moc))}')
    pd = first.get('phan_du') or {}
    ph = {c: ent[c].get('phat_hien') or {} for c in CHO}
    doan_of = {v[0]: v for v in (uoc.get('doan') or {}).values() if v}
    return dict(
        schema=SCHEMA, session_sha256=[sha_of[a], sha_of[b]], session=[a, b], du_m=max(du_m), n_moc=next(iter(n_moc)),
        mo_hinh=MO_HINH, k_se=num(uoc.get('k_se'), 'k_se'), rho_m=num(uoc.get('rho_m'), 'rho_m'), truoc_le=True,
        to_hop=sorted(to_hop), theo_cho=theo_cho,
        phan_du=dict(n_tham_so=N_THAM_SO, s_du_m=num(pd.get('s_du_m'), 's_du_m'),
                     du_sau_khop_max_m=num(pd.get('du_sau_khop_max_m'), 'du_sau_khop_max_m'),
                     loo_rms_m=num(pd.get('loo_rms_m'), 'loo_rms_m'),
                     tuong_quan_du_lan_can=[fnum(x, 'tuong_quan') for x in pd.get('tuong_quan_du_lan_can') or []]),
        nen_nhieu=[x for x in nen_nhieu(uoc, doan_of) if x['cap'].split('|')[0] in (a, b)],
        phat_hien_ma=dict(delta_canh_m={c: fnum(ph[c].get('delta_canh_tai_cho_m'), f'{c} delta_canh') for c in CHO},
                          delta_canh_sd_du_m=num(ph[CHO[0]].get('delta_canh_sd_du_m'), 'delta_canh_sd_du_m'),
                          xoay_moc_sd_rad=num(ph[CHO[0]].get('xoay_moc_sd_rad'), 'xoay_moc_sd_rad')),
        cach='uoc_khung_giua_session.py (CODE_KHUNG), mo hinh phys; c7_khung_giua_session.py (CAM) chi chep va kiem',
        nguon=[uoc_meta['uoc'], uoc_meta['script']] + [dict(vai_tro=f'session_{x}', **sessions[x]) for x in (a, b)],
        hil_uoc=uoc.get('hil'), commit_hil=commit_hil())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--uoc', required=True, help='uoc_khung_<sub>.json of uoc_khung_giua_session.py')
    ap.add_argument('--uoc-script', required=True, help='uoc_khung_giua_session.py that wrote --uoc (sha256 recorded)')
    ap.add_argument('--session', action='append', required=True,
                    help='TEN=<session.json>: a scoring session (clip H opening a segment), two or more')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    try:
        if not os.path.isfile(a.uoc) or os.path.getsize(a.uoc) == 0:
            raise KgsError(f'{a.uoc}: khong co hay rong')
        with open(a.uoc, encoding='utf-8') as f:
            uoc = json.load(f)
        sessions = {}
        for s in a.session:
            ten, sep, p = s.partition('=')
            if not sep or not ten or not os.path.isfile(p):
                raise KgsError(f'--session {s!r}: can TEN=<session.json> co that')
            if ten in sessions:
                raise KgsError(f'--session {ten}: dua hai lan')
            sessions[ten] = dict(duong_dan=os.path.abspath(p), sha256=sha256_file(p))
        if len(sessions) < 2:
            raise KgsError('can it nhat hai --session')
        meta = dict(uoc=dict(vai_tro='uoc', duong_dan=os.path.abspath(a.uoc), sha256=sha256_file(a.uoc)),
                    script=dict(vai_tro='uoc_script', duong_dan=os.path.abspath(a.uoc_script),
                                sha256=sha256_file(a.uoc_script)))
        names = [n for n in (uoc.get('session') or {}) if n in sessions]
        pairs = list(itertools.combinations([n for n in sessions if n in names], 2))
        if len(names) != len(sessions):
            raise KgsError(f'uoc khong co session {sorted(set(sessions) - set(names))}')
        out = {}
        for x, y in pairs:
            p = os.path.join(a.out_dir, f'khung_giua_session_{x}_{y}.json')
            if os.path.exists(p):
                raise KgsError(f'{p} da co: dung thu muc moi')
            out[p] = build(uoc, meta, sessions, x, y)
        os.makedirs(a.out_dir, exist_ok=True)
        for p, o in out.items():
            with open(p, 'x', encoding='utf-8') as f:
                f.write(json.dumps(o, ensure_ascii=False, indent=1) + '\n')
            t = o['theo_cho']
            print(f'{os.path.basename(p)}: du_m {o["du_m"] * 1000:.3f} mm; ' + '; '.join(
                f'{c} xoay {t[c]["du_xoay_rad"] * 180e3 / math.pi:.2f} bien_dang {t[c]["du_bien_dang_rad"] * 180e3 / math.pi:.2f}'
                f' hl12b {t[c]["hanh_lang_12b"]["du_bien_dang_rad"] * 180e3 / math.pi:.2f} mdeg' for c in CHO))
    except (KgsError, OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        print(f'c7_khung_giua_session: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
