#!/usr/bin/env python3
"""
c8b_d3_hang.py: the rows of D3 (separate position check, design C8B v7 section 8) for c8b_kiem_vi_tri.py, read from the
files of the sources; one CSV row per signed error of a source on its test set (tap kiem: made after the C8b lock).
CODE_C8B local_adeaeda8, 24/11/2026 (CODE local_42161302: "lam D3 truoc").

Columns (c8b_kiem_vi_tri COLS): nguon, tap, dai_luong, mien, sai_m, chi_so_m, co_chuan_dong. Every source file must be
one the C8b lock names (khoa_truoc.nguon, same sha256), so the index compared is the one the lock used.
Sources written here (c8b_kiem_vi_tri lists a missing source):
  Held out by session (P1_A3_H32, replay_neo), rule L155 (Liam 24/11 khuya at CODE, box, verbatim: "So co le, vuot thi
                    noi bat dinh (Recommended)"): chi_so_m = the index of the OTHER sessions x the lock's margin le
                    (LE 1.25, in c8b_khoa_v5 since 8ff78a8) x the held-out factor of the term (c8b_he_so_giu_ngoai:
                    ratio > le -> ratio / le, applied in the lock, so U grows; A3 has no factor). The rule was CHANGED
                    AFTER the numbers of a trial on 2-3 sessions were seen (24/11): the first rule compared the
                    largest error of a session with the largest of the others before the margin, which always fails
                    the session holding the overall largest (a property of the comparison, not of the data).
  replay_neo        (2) held out by session as P1_A3_H32: every valid frame of KIEMCHEO neo_luot --khung-csv
                    (hypot(dx, dy); |dpsi| at rho_max) against P_neo_m / A_neo_rad of the other sessions; frames
                    mat_W / khong_hop_le listed in the sidecar (khung_thieu_replay_neo), no row.
  P1_A3_H32         (1) held out BY SESSION (CODE 24/11 khuya, contract log): pairs of each segment (KIEMCHEO
                    cap_tuong_doi.csv |d|, a3_cap.csv quy_deg) against the index of the OTHER sessions' segments
                    (P1_tuong_doi.theo_vung.<vung>.max_m, A3_tuong_doi.A3loc_max_deg); A3 at rho_max; mien per
                    segment (and place for P1); co_chuan_dong 0.
  E                 (4) kiem_imu of the held-out E clips of the A6 (du_rad vs nguong_rad) at rho_max; heading only,
                    session 2 only; co_chuan_dong 0.
  (12b, source (3) of design 8, is dropped: no signed position error is measured by 12b; it is in the F7 and in the
                    angle a of dau_cuoi_lan_lap; CODE 24/11 khuya, contract log.)
  do_nhay_cao       (5) c8b_u_h_chung_v1 (c8b_u_h.py): every held-out row giu_ngoai[].sai_m (h predicted - h calibrated, m)
                    against u_h_chung_m; dai_luong h, mien chung; co_chuan_dong 0 (no dynamic height reference).
  troi_H            (7) c7_calib_drift files, one per camera segment (BUOI/doanN): sai_m = D_H_mm / 1000 against the
                    lock's eps_H.mm / 1000 (D_H only triggers a scoring again; design 8 (7)); a file without the end
                    clip H (thieu_H1) or without D_H is refused; co_chuan_dong 0.
  dau_cuoi_lan_lap  (6) F8 empty: the angle a between the tag axis and the car motion axis (F7 truc_xe_v1 of HIEUCHUAN)
                    of each later HC_goc against the first (session 2, F7 'khoa'): Liam 24/11 at CODE local_42161302,
                    box, verbatim: "Thay bang goc HC_goc (Recommended)". As a position at the largest lever arm of the A6
                    domain: sai_m = rho_max x rad(a_later - a_first), chi_so_m = rho_max x rad(u_a_noi_bo of the
                    first) (the same rho on both sides: the test is the angle test); dai_luong a_tam_xe, mien = place;
                    co_chuan_dong 0. Limit: heading only, not position; a later F7 'chan_doan' has no u of its own.

  python tools/c8b_d3_hang.py --khoa c8b_khoa_v5.json --a6 c8b_a6_v2.json --u-h c8b_u_h_chung_v1.json \\
      --drift 2026-11-20_run02/doan2=drift.json [--drift ...] --goc-a-dau F7_buoi2.json --goc-a 2026-11-23_run03=F7.json \\
      --out d3_rows.csv
Writes d3_rows.csv and d3_rows.csv.nguon.json (sources with sha256, rules, limits). Never overwrites.
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402
import c8b_kiem_vi_tri as kvt                                            # noqa: E402

VERSION = 'c8b_d3_hang_v1'
LIAM_DAU_CUOI = ('Liam 24/11 o CODE local_42161302, hop chon, nguyen van: "Thay bang goc HC_goc (Recommended)" '
                 '(nhat ky hop dong C8 24/11 khuya)')


class HangError(ValueError):
    pass


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _fnum(x):
    """A finite float from a CSV cell (a string) or a number; None otherwise."""
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return None
    return _num(x)


def _read(p):
    cu.check_path(p)
    with open(p, encoding='utf-8') as f:
        return json.load(f), cu.sha256(p)


def lock_shas(khoa, vai):
    kt = khoa.get('khoa_truoc') if isinstance(khoa.get('khoa_truoc'), dict) else {}
    return {n.get('sha256') for n in kt.get('nguon') or [] if isinstance(n, dict) and n.get('vai') == vai}


def _row(nguon, dl, mien, sai, cs, dong=False):
    return dict(nguon=nguon, tap='kiem', dai_luong=dl, mien=mien, sai_m=repr(float(sai)), chi_so_m=repr(float(cs)),
                co_chuan_dong='1' if dong else '0')


def rows_u_h(d, sha, khoa):
    if sha not in lock_shas(khoa, 'u_h_chung'):
        raise HangError(f'--u-h sha256 {sha[:12]} is not a u_h_chung source of the lock (index must be the one used)')
    # the file c8b_u_h.py writes: schema c8b_u_h_chung_v1 (c8b_a6.SCHEMA_UH), version c8b_u_h_v1 (real data 24/11: the
    # reader asked schema c8b_u_h_v1 and refused the real file), trang_thai dat
    if not isinstance(d, dict) or d.get('schema') != 'c8b_u_h_chung_v1' or d.get('version') != 'c8b_u_h_v1':
        raise HangError('--u-h: not a c8b_u_h_chung_v1 file of c8b_u_h_v1')
    if d.get('trang_thai') != 'dat':
        raise HangError(f'--u-h: trang_thai {d.get("trang_thai")!r}')
    cs = _num(d.get('u_h_chung_m'))
    g = d.get('giu_ngoai')
    if cs is None or cs < 0 or not isinstance(g, list) or not g:
        raise HangError('--u-h: u_h_chung_m / giu_ngoai missing')
    out = []
    for r in g:
        s = _num((r or {}).get('sai_m'))
        if s is None:
            raise HangError(f'--u-h: held-out row {r!r} without a finite sai_m')
        out.append(_row('do_nhay_cao', 'h', 'chung', s, cs))
    return out


def rows_drift(items, khoa):
    eps = _num((khoa.get('eps_H') or {}).get('mm')) if isinstance(khoa.get('eps_H'), dict) else None
    if eps is None or eps <= 0:
        raise HangError('the lock has no eps_H.mm')
    out, seen = [], set()
    for seg, d in items:
        if seg in seen:
            raise HangError(f'--drift {seg}: twice')
        seen.add(seg)
        if not isinstance(d, dict) or d.get('kind') != 'c7_calib_drift':
            raise HangError(f'--drift {seg}: not a c7_calib_drift file')
        dh = _num(d.get('D_H_mm'))
        if d.get('thieu_H1') or dh is None or dh < 0:
            raise HangError(f'--drift {seg}: no D_H (end clip H missing or D_H_mm not finite)')
        out.append(_row('troi_H', 'D_H', seg, dh / 1000.0, eps / 1000.0))
    return out


def rows_goc_a(f_dau, sha_dau, sau, khoa, a6):
    if sha_dau not in lock_shas(khoa, 'truc_xe'):
        raise HangError(f'--goc-a-dau sha256 {sha_dau[:12]} is not the F7 (truc_xe) of the lock')
    if not isinstance(f_dau, dict) or f_dau.get('schema') != 'truc_xe_v1' or f_dau.get('trang_thai') != 'khoa':
        raise HangError('--goc-a-dau: a truc_xe_v1 F7 with trang_thai khoa (the first HC_goc, session 2)')
    a0, u0, cho = _num(f_dau.get('a_k_deg')), _num(f_dau.get('u_a_noi_bo_deg')), f_dau.get('cho')
    if a0 is None or u0 is None or u0 <= 0:
        raise HangError('--goc-a-dau: a_k_deg / u_a_noi_bo_deg missing')
    rho = _num(((a6 or {}).get('mien') or {}).get('rho_max_m'))
    if rho is None or rho <= 0:
        raise HangError('--a6: mien.rho_max_m missing')
    out, gh = [], []
    ses = [f.get('session_sha256') for _, f, _ in sau if isinstance(f, dict)]
    if len(set(ses)) != len(ses):                    # Codex round 2 D3v2-8: one HC_goc under two labels
        raise HangError('--goc-a: two later F7 of the same camera session (one HC_goc under two labels)')
    for buoi, f, sha in sau:
        if not isinstance(f, dict) or f.get('schema') != 'truc_xe_v1' or f.get('cho') != cho:
            raise HangError(f'--goc-a {buoi}: not a truc_xe_v1 F7 of place {cho}')
        if f.get('session_sha256') == f_dau.get('session_sha256'):
            raise HangError(f'--goc-a {buoi}: the same camera session as the first HC_goc (not a later end)')
        a = _num(f.get('a_k_deg'))
        if a is None:
            raise HangError(f'--goc-a {buoi}: a_k_deg missing')
        out.append(_row('dau_cuoi_lan_lap', 'a_tam_xe', cho, rho * math.radians(a - a0), rho * math.radians(u0)))
        if f.get('trang_thai') != 'khoa':
            gh.append(f'{buoi}: F7 {f.get("trang_thai")!r} (no u of its own), sha256 {sha[:12]}')
    return out, dict(a_dau_deg=a0, u_a_noi_bo_dau_deg=u0, rho_max_m=rho, cho=cho, gioi_han_dau_sau=gh)


def _rho(a6):
    rho = _num(((a6 or {}).get('mien') or {}).get('rho_max_m'))
    if rho is None or rho <= 0:
        raise HangError('--a6: mien.rho_max_m missing')
    return rho


def _csv(p):
    cu.check_path(p)
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f)), cu.sha256(p)


def doc_kiem_cheo(p):
    """(kiem_cheo dict, its sha, cap rows, cap sha, a3 rows, a3 sha) of a KIEMCHEO kiem_cheo.json and the two pair CSVs
    beside it, the CSVs tied to the JSON (Codex D3-7): cap_tuong_doi.csv by the SHA the JSON records
    (P1_tuong_doi.cap_csv.sha256); a3_cap.csv, which the JSON records no SHA of, by its number (the largest quy_deg
    of its rows is A3_tuong_doi.A3loc_max_deg; checked on the four real segments 25/11). Rows taken out below the
    maximum of a3_cap.csv are not seen (limit)."""
    kc, s = _read(p)
    d = os.path.dirname(os.path.abspath(p))
    cap, cs = _csv(os.path.join(d, 'cap_tuong_doi.csv'))
    a3, as_ = _csv(os.path.join(d, 'a3_cap.csv'))
    if not isinstance(kc, dict):
        raise HangError(f'{p}: not a kiem_cheo file')
    want = (((kc.get('P1_tuong_doi') or {}).get('cap_csv') or {}).get('sha256'))
    if want != cs:
        raise HangError(f'{p}: cap_tuong_doi.csv beside it is not the file it records (P1_tuong_doi.cap_csv.sha256)')
    q = [_fnum(r.get('quy_deg')) for r in a3]
    am = _num((kc.get('A3_tuong_doi') or {}).get('A3loc_max_deg'))
    if not q or any(x is None for x in q) or am is None or abs(max(q) - am) > 1e-9 * max(1e-6, am):
        raise HangError(f'{p}: a3_cap.csv beside it does not give A3_tuong_doi.A3loc_max_deg (not the pairs of that '
                        'file, or a quy_deg not finite)')
    # Codex round 2 D3v2-6: the number the lock takes at a place is not above the largest pair of the file
    xp = [_num(v) for v in ((kc.get('A3_tuong_doi') or {}).get('A3_xp_deg') or {}).values()]
    if any(v is not None and v > am * (1 + 1e-9) for v in xp):
        raise HangError(f'{p}: an A3_xp_deg above A3loc_max_deg (the file contradicts itself)')
    return kc, s, cap, cs, a3, as_


def khong_lap(items, what):
    """Codex D3-4, D3-8: one file under two labels (sessions, segments) would be compared with itself as 'another
    session'. items [(label, sha256)]."""
    seen = {}
    labs = [lab for lab, _ in items]
    if len(set(labs)) != len(labs):                  # Codex round 2 D3v2-4: the last file of a label would win
        raise HangError(f'{what}: a label is given twice ({sorted(x for x in set(labs) if labs.count(x) > 1)})')
    for lab, sha in items:
        if seen.setdefault(sha, lab) != lab:
            raise HangError(f'{what}: {seen[sha]} and {lab} are the same file (sha256 {sha[:12]})')


def rows_p1a3(segs, rho, le=1.0, he_so=None):
    """P1_A3_H32 held out BY SESSION (CODE 24/11 khuya, applying C8B's recommendation; contract log 24/11): every pair
    of a segment of session s against the index of the segments of the OTHER sessions (KIEMCHEO: no fixed held-out ID
    set; the index of a segment is the max over its own pairs). segs [(seg 'BUOI/doanN', kiem_cheo dict, cap rows,
    a3 rows)]. P1 per place (vung): sai = |d| = hypot(d_x, d_y) of each pair (deduplicated on clip, vung, s, j)
    against max over other sessions of P1_tuong_doi.theo_vung.<vung>.max_m (the same |d| max, KIEMCHEO). A3: sai =
    quy_deg (|dtheta| + gap) of each pair (L <= 1.05 m as written) against max over other sessions of
    A3_tuong_doi.A3_xp_deg.<CHO_MUC> (the A3 number the lock takes, c8b_khoa_v5.hang_dong_tinh; Codex D3-6: until
    25/11 A3loc_max_deg, the max of quy over every pair, which is the same number on the four real segments but need
    not be the one the lock uses), as a position at rho_max. mien = '<seg>:<vung>' / '<seg>'. |sai| only: the sign of
    d_x, d_y, dtheta stays in the CSVs (limit: the table is not signed for these two sources)."""
    import c8b_khoa_v5 as khoa_mod
    hs = he_so or {}
    k_p1 = le * hs.get('P1_m', 1.0)           # L155: the index as the lock uses it (margin, held-out factor)
    buoi = {s.split('/')[0] for s, *_ in segs}
    if len(buoi) < 2:
        raise HangError('--kiem-cheo: held out by session needs segments of at least two sessions')
    p1_of, a3_of = {}, {}
    for s, kc, _, _ in segs:
        tv = ((kc.get('P1_tuong_doi') or {}).get('theo_vung') or {})
        p1_of[s] = {v: _num((tv.get(v) or {}).get('max_m')) for v in tv}
        a3_of[s] = _num(((kc.get('A3_tuong_doi') or {}).get('A3_xp_deg') or {}).get(khoa_mod.CHO_MUC))
    out = []
    for s, kc, cap, a3 in segs:
        b = s.split('/')[0]
        others = [o for o in p1_of if o.split('/')[0] != b]
        seen = set()
        for r in cap:
            v = r.get('vung')
            key = (r.get('clip'), v, r.get('s'), r.get('j'))
            if key in seen:
                continue
            seen.add(key)
            idx = [p1_of[o].get(v) for o in others]
            if not idx or any(x is None for x in idx):
                raise HangError(f'P1 {s} vung {v!r}: an other session has no P1_tuong_doi.theo_vung.{v}.max_m')
            dx, dy = _fnum(r.get('d_x')), _fnum(r.get('d_y'))
            if dx is None or dy is None:
                raise HangError(f'P1 {s}: pair {key} without finite d_x, d_y')
            out.append(_row('P1_A3_H32', 'P1', f'{s}:{v}', math.hypot(dx, dy), k_p1 * max(idx)))
        ia = [a3_of[o] for o in others]
        if any(x is None for x in ia):
            raise HangError(f'A3 {s}: an other session has no A3_tuong_doi.A3_xp_deg.{khoa_mod.CHO_MUC}')
        for r in a3:
            q = _fnum(r.get('quy_deg'))
            if q is None:
                raise HangError(f'A3 {s}: pair {r.get("i")}-{r.get("j")} without finite quy_deg')
            # rho x (...) on both sides: a product by the same rho keeps the order (Codex D3-5)
            out.append(_row('P1_A3_H32', 'A3', s, rho * math.radians(q), rho * (le * math.radians(max(ia)))))
    return out


def rows_e(a6, rho):
    """E (CODE 24/11 khuya): the camera-IMU heading check of the held-out E clips of the A6 (kiem_imu: du_rad against
    nguong_rad, per place), as a position at rho_max; heading only, session 2 only; co_chuan_dong 0."""
    ki = a6.get('kiem_imu')
    if not isinstance(ki, list) or not ki:
        raise HangError('--a6: no kiem_imu (held-out E clips)')
    out = []
    for x in ki:
        du, ng = _num((x or {}).get('du_rad')), _num((x or {}).get('nguong_rad'))
        if du is None or ng is None or ng <= 0 or not x.get('cho'):
            raise HangError(f'--a6 kiem_imu {x.get("luot_id_E")!r}: du_rad / nguong_rad / cho missing')
        out.append(_row('E', 'kiem_imu_E', x['cho'], rho * du, rho * ng))
    return out


def rows_neo(items, rho, le=1.0, he_so=None):
    """replay_neo (2) held out BY SESSION, the rule of P1_A3_H32 (CODE 24/11 khuya): every valid frame of the replay of
    anchors of a run of session s (KIEMCHEO neo_luot --khung-csv: trang_thai hop_le, dx_m, dy_m = e_t - mean_W,
    dpsi_rad) against the index of the OTHER sessions (neo_luot.json theo_cho.<cho>.P_neo_m, A_neo_rad: the max over
    their own frames). items [(buoi, neo_luot dict, khung rows)]. A frame mat_W / khong_hop_le has no number: listed in
    khung_thieu (design 8 (2): that quantity thieu on that frame), not a row. mien = '<buoi>:<cho>'."""
    hs = he_so or {}
    k_p, k_a = le * hs.get('P_neo_m', 1.0), le * hs.get('A_neo_rad', 1.0)      # L155, as rows_p1a3
    if len({b for b, _, _ in items}) < 2:
        raise HangError('--neo: held out by session needs runs of at least two sessions')
    idx = {}
    for b, n, _ in items:
        for cho, v in ((n or {}).get('theo_cho') or {}).items():
            P, A = _num((v or {}).get('P_neo_m')), _num((v or {}).get('A_neo_rad'))
            if P is None or A is None:
                raise HangError(f'--neo {b}: theo_cho.{cho} without P_neo_m / A_neo_rad')
            idx.setdefault(b, {})[cho] = (P, A)
    out, thieu = [], []
    for b, n, kh in items:
        cho_of = {lid: (v or {}).get('cho') for lid, v in ((n or {}).get('luot') or {}).items()}
        # the frames are those of THIS neo_luot.json (the lock's file; real data 24/11: the JSON written beside the
        # CSV differs from the lock's in commit_hil only, so the CSV is tied by its numbers): per run the largest
        # hypot(dx, dy) and |dpsi| of the valid frames are P_neo_m, A_neo_rad of the run
        for lid, v in ((n or {}).get('luot') or {}).items():
            fr = [r for r in kh if r.get('luot_id') == lid and r.get('trang_thai') == 'hop_le']
            P, A = _num((v or {}).get('P_neo_m')), _num((v or {}).get('A_neo_rad'))
            if not fr or P is None or A is None:
                raise HangError(f'--neo {b}: run {lid} without valid frames in the CSV or without P_neo_m / A_neo_rad')
            mp = max(math.hypot(_fnum(r.get('dx_m')) or 0.0, _fnum(r.get('dy_m')) or 0.0) for r in fr)
            ma = max(abs(_fnum(r.get('dpsi_rad')) or 0.0) for r in fr)
            if abs(mp - P) > 1e-9 * max(1e-6, P) or abs(ma - A) > 1e-9 * max(1e-6, A):
                raise HangError(f'--neo {b}: the CSV frames of run {lid} do not give P_neo_m / A_neo_rad of the '
                                f'neo_luot.json ({mp!r}, {ma!r} vs {P!r}, {A!r}): not the frames of that file')
        for r in kh:
            lid = r.get('luot_id')
            cho = cho_of.get(lid)
            if cho is None:
                raise HangError(f'--neo {b}: frame of run {lid!r} not in neo_luot.json luot')
            if r.get('trang_thai') != 'hop_le':
                thieu.append([b, lid, r.get('khung'), r.get('trang_thai'), r.get('ly_do')])
                continue
            o = [idx[x][cho] for x in idx if x != b and cho in idx[x]]
            if not o:
                raise HangError(f'--neo {b}: no other session has an index at place {cho}')
            dx, dy, dp = _fnum(r.get('dx_m')), _fnum(r.get('dy_m')), _fnum(r.get('dpsi_rad'))
            if dx is None or dy is None or dp is None:
                raise HangError(f'--neo {b}: valid frame {lid}/{r.get("khung")} without finite dx, dy, dpsi')
            out.append(_row('replay_neo', 'P_neo', f'{b}:{cho}', math.hypot(dx, dy), k_p * max(x[0] for x in o)))
            # rho x (k_a x index), as c8b_he_so_giu_ngoai checks the factor with rho 1: the same rho on both sides
            # keeps the order, so a factor that passes there passes here (Codex D3-5)
            out.append(_row('replay_neo', 'A_neo', f'{b}:{cho}', rho * abs(dp), rho * (k_a * max(x[1] for x in o))))
    return out, thieu


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--khoa', required=True, help='the C8b lock (c8b_khoa_v5; the combined one of the gate)')
    ap.add_argument('--a6', required=True, help='the A6 of the lock (c8b_a6_v2: mien.rho_max_m)')
    ap.add_argument('--u-h', default=None, help='c8b_u_h_chung_v1 file (do_nhay_cao)')
    ap.add_argument('--drift', action='append', default=[], help='BUOI/doanN=c7_calib_drift json (troi_H), repeat')
    ap.add_argument('--goc-a-dau', default=None, help='F7 truc_xe_v1 of the first HC_goc (session 2, khoa)')
    ap.add_argument('--goc-a', action='append', default=[], help='BUOI=F7 truc_xe_v1 of a later HC_goc, repeat')
    ap.add_argument('--kiem-cheo', action='append', default=[],
                    help='BUOI/doanN=kiem_cheo.json of KIEMCHEO (P1_A3_H32; cap_tuong_doi.csv, a3_cap.csv beside it)')
    ap.add_argument('--e', action='store_true', help='E: kiem_imu of the held-out E clips of --a6')
    ap.add_argument('--neo', action='append', default=[],
                    help='BUOI=neo_luot.json,khung.csv (KIEMCHEO neo_luot --khung-csv; replay_neo), repeat')
    ap.add_argument('--he-so', default=None,
                    help='c8b_he_so_giu_ngoai_v1 file of the lock (needed when the lock names one; L155)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        khoa, ksha = _read(a.khoa)
        if not isinstance(khoa, dict) or khoa.get('version') != kvt.KHOA_VERSION:
            raise HangError(f'--khoa: not a {kvt.KHOA_VERSION} file')
        a6, a6sha = _read(a.a6)
        if a6sha not in lock_shas(khoa, 'a6'):
            raise HangError('--a6 is not the A6 of the lock')
        rows, src, info = [], [dict(vai='khoa', duong_dan=os.path.abspath(a.khoa), sha256=ksha),
                                dict(vai='a6', duong_dan=os.path.abspath(a.a6), sha256=a6sha)], {}
        # L155: the sources held out by session are compared with the index of the OTHER sessions as the lock uses
        # it: x the lock's margin le, x the held-out factor of the lock's c8b_he_so_giu_ngoai file (docstring)
        le = _num(khoa.get('le'))
        if le is None or le < 1.0:
            raise HangError('the lock has no margin le')
        hs_sha, he_so = lock_shas(khoa, 'he_so_giu_ngoai'), None
        if bool(hs_sha) != bool(a.he_so):
            raise HangError('--he-so goes with a lock that names a he_so_giu_ngoai source (and only with it)')
        if a.he_so:
            hd, hsha = _read(a.he_so)
            if hsha not in hs_sha or not isinstance(hd, dict) or not isinstance(hd.get('hang'), dict):
                raise HangError('--he-so is not the he_so_giu_ngoai file of the lock')
            he_so = {k: float(v) for k, v in hd['hang'].items()}
            src.append(dict(vai='he_so_giu_ngoai', duong_dan=os.path.abspath(a.he_so), sha256=hsha))
            hs_nguon = {(n.get('vai'), n.get('sha256')) for n in hd.get('nguon') or [] if isinstance(n, dict)}
        info['giu_ngoai_theo_buoi'] = dict(le=le, he_so=he_so, luat='L155')
        if a.u_h:
            d, s = _read(a.u_h)
            rows += rows_u_h(d, s, khoa)
            src.append(dict(vai='do_nhay_cao', duong_dan=os.path.abspath(a.u_h), sha256=s))
        items = []
        for x in a.drift:
            seg, sep, p = x.partition('=')
            if not sep or not seg:
                raise HangError(f'--drift {x!r}: BUOI/doanN=path')
            d, s = _read(p)
            items.append((seg, d))
            src.append(dict(vai=f'troi_H {seg}', duong_dan=os.path.abspath(p), sha256=s))
        khong_lap([(x['vai'], x['sha256']) for x in src if x['vai'].startswith('troi_H ')], '--drift')
        rows += rows_drift(items, khoa) if items else []
        if bool(a.goc_a_dau) != bool(a.goc_a):
            raise HangError('--goc-a-dau and --goc-a go together')
        if a.goc_a_dau:
            f0, s0 = _read(a.goc_a_dau)
            sau = []
            for x in a.goc_a:
                b, sep, p = x.partition('=')
                if not sep or not b:
                    raise HangError(f'--goc-a {x!r}: BUOI=path')
                f, s = _read(p)
                sau.append((b, f, s))
                src.append(dict(vai=f'goc_a {b}', duong_dan=os.path.abspath(p), sha256=s))
            src.append(dict(vai='goc_a_dau', duong_dan=os.path.abspath(a.goc_a_dau), sha256=s0))
            khong_lap([('goc_a_dau', s0)] + [(b, s) for b, _, s in sau], '--goc-a')
            r, info['dau_cuoi_lan_lap'] = rows_goc_a(f0, s0, sau, khoa, a6)
            rows += r
        segs = []
        for x in a.kiem_cheo:
            seg, sep, p = x.partition('=')
            if not sep or '/' not in seg:
                raise HangError(f'--kiem-cheo {x!r}: BUOI/doanN=path')
            kc, s, cap, cs, a3, as_ = doc_kiem_cheo(p)
            if s not in lock_shas(khoa, 'kiem_cheo'):
                raise HangError(f'--kiem-cheo {seg}: sha256 {s[:12]} is not a kiem_cheo source of the lock')
            d = os.path.dirname(os.path.abspath(p))
            segs.append((seg, kc, cap, a3))
            src += [dict(vai=f'kiem_cheo {seg}', duong_dan=os.path.abspath(p), sha256=s),
                    dict(vai=f'cap_tuong_doi {seg}', duong_dan=os.path.join(d, 'cap_tuong_doi.csv'), sha256=cs),
                    dict(vai=f'a3_cap {seg}', duong_dan=os.path.join(d, 'a3_cap.csv'), sha256=as_)]
        khong_lap([(x['vai'], x['sha256']) for x in src if x['vai'].startswith('kiem_cheo ')], '--kiem-cheo')
        if segs:
            rows += rows_p1a3(segs, _rho(a6), le, he_so)
        if a.e:
            rows += rows_e(a6, _rho(a6))
        neo = []
        for x in a.neo:
            b, sep, ps = x.partition('=')
            pj, sep2, pc = ps.partition(',')
            if not sep or not sep2 or not b:
                raise HangError(f'--neo {x!r}: BUOI=neo_luot.json,khung.csv')
            n, s = _read(pj)
            if s not in lock_shas(khoa, 'neo'):
                raise HangError(f'--neo {b}: neo_luot.json sha256 {s[:12]} is not a neo source of the lock')
            kh, ks = _csv(pc)
            neo.append((b, n, kh))
            src += [dict(vai=f'neo {b}', duong_dan=os.path.abspath(pj), sha256=s),
                    dict(vai=f'neo_khung {b}', duong_dan=os.path.abspath(pc), sha256=ks)]
        khong_lap([(x['vai'], x['sha256']) for x in src if x['vai'].startswith('neo ')], '--neo')
        if neo:
            r, info['khung_thieu_replay_neo'] = rows_neo(neo, _rho(a6), le, he_so)
            rows += r
        if a.he_so:
            # Codex D3-4: the factors are those of exactly the files compared here (session labels and SHA of every
            # kiem_cheo, pair CSV, neo_luot and frame CSV); the lock already asks its own sources among them
            giu = ('kiem_cheo ', 'cap_tuong_doi ', 'a3_cap ', 'neo ', 'neo_khung ')
            day = {(x['vai'], x['sha256']) for x in src if x['vai'].startswith(giu)}
            hsn = {x for x in hs_nguon if isinstance(x[0], str) and x[0].startswith(giu)}
            if day != hsn:
                raise HangError('--he-so was not computed from exactly the --kiem-cheo / --neo files given here '
                                f'(labels and sha256): {sorted(v for v, _ in day ^ hsn)}')
        if not rows:
            raise HangError('no source given')
        with open(a.out, 'x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(kvt.COLS))
            w.writeheader()
            w.writerows(rows)
        meta = dict(version=VERSION, file=os.path.basename(a.out), sha256=cu.sha256(a.out), n_hang=len(rows),
                    nguon=src, thong_tin=info, liam_dau_cuoi_lan_lap=LIAM_DAU_CUOI if a.goc_a_dau else None,
                    nguon_co=sorted({r['nguon'] for r in rows}),
                    nguon_chua=[n for n in kvt.NGUON_CAN if n not in {r['nguon'] for r in rows}],
                    quy_tac='moi file nguon phai la nguon cua khoa (cung sha256); tap kiem (dung sau khoa)')
        with open(a.out + '.nguon.json', 'x', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    except (HangError, cu.InputError, OSError, ValueError) as exc:
        print(f'c8b_d3_hang: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print(f'{len(rows)} hang; nguon co {meta["nguon_co"]}; chua {meta["nguon_chua"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
