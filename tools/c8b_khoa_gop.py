#!/usr/bin/env python3
"""
c8b_khoa_gop.py: ONE C8b lock for several C8 floor sessions (CODE_C8B local_adeaeda8, 24/11/2026).

Why: Liam 24/11 ~21:40 at CODE local_42161302, box A2 of the FABLE_C8 review, verbatim: "Mot khoa, lay lon nhat
(Recommended)": the gate (gate_c8 ghep) takes one c8b_khoa_v5 file, but the C8 runs come from several sessions (session
2, run03, run05, the new LADRC session), each with its own per-session sources (day_doi, kiem_cheo, neo, F12, U_moc,
P6, F8). Each session keeps its own lock made by c8b_khoa_v5 exactly as before (already run on real data); this tool
combines those per-session locks into one file of the same version (c8b_khoa_v5, the name the gate checks), every
bound term the LARGEST over the sessions (J v6: U_t,C8 is the largest over the runs; the same build over sessions).

Rules (a field not named here must be the same in every session lock, else refused):
  largest  muc.<lv>.hang_dong {hang_m, he_so_r, hang_rad, u_phi_rad}; muc.<lv>.hang_dong_truoc_le (same four);
           muc.<lv>.hang_theo_u[i] {P4_m, A4_deg, P6_rieng_ii_m, P4_tay_don_m} (band edges u_lo_m_s, u_hi_m_s, dai the
           same); u_beta_cam_deg[cho]. A null in one session -> null (a term not concluded in one session is not
           concluded).
  state    worst over the sessions: hang_theo_u[i].trang_thai, u_beta_cam_trang_thai[cho] (per place, the field
           c7_score reads), u_beta_cam_trang_thai_cho, u_beta_cam_trang_thai_chung, mien_car_pose.trang_thai.
  hull     mien_car_pose.kappa = [min lo, max hi] (each session's frames lie in its own range; an intersection would
           drop valid frames at the edges) [DE XUAT, CODE chot]; side_m, phien_ban, h_m the same.
  per      fields that only explain a session's numbers (muc.<lv>.hang_dong_thanh_phan, hang_dong_ly_do,
  session  hang_theo_u_ly_do, P4_tay_don, u_beta_cam_dau_chan, u_beta_cam_gia_dinh, u_beta_cam_ly_do): the same in
           every session -> kept; different -> {"theo_buoi": {buoi: value}} (lists of reasons: "buoi: reason").
  lock     khoa_truoc.nguon = the union of the session sources (one entry per (vai, sha256)) + each session lock
           (vai 'khoa_buoi <buoi>'); khoa_truoc.script = this tool; luc = now; commit_hil the same in every lock.
Checks after combining: every session lock version c8b_khoa_v5, trang_thai khoa, made at the same commit_hil with the
per-place field of N1 (u_beta_cam_trang_thai_chung present); exactly one A6 source sha and one plan (ke_hoach_c8) sha
over the sessions (the gate's _check_a6, GC6-1); hang_dong = LE x hang_dong_truoc_le for hang_m, he_so_r, hang_rad and
u_phi_rad = LE x hang_dong_truoc_le.u_phi_rad + radians(u_beta_cam_deg[goc]) (the scorer's formula, c8b_khoa_v5 build);
c8b_khoa_v5.trang_thai_khoa of the result is khoa. The block 'gop' names the sessions, their lock sha256 and, per term,
the session that gave the largest value.

  python tools/c8b_khoa_gop.py --khoa-buoi 2026-11-20_run02=a.json --khoa-buoi 2026-11-23_run03=b.json \\
      --commit-hil <40 hex> --out c8b_khoa_v5.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import copy
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
import c8b_khoa_v5 as kv                                                 # noqa: E402

GOP_VERSION = 'c8b_khoa_gop_v1'
HD_MAX = ('hang_m', 'he_so_r', 'hang_rad', 'u_phi_rad')
BAND_MAX = ('P4_m', 'A4_deg', 'P6_rieng_ii_m', 'P4_tay_don_m')
BAND_SAME = ('u_lo_m_s', 'u_hi_m_s', 'dai')
MUC_PER_BUOI = ('hang_dong_thanh_phan', 'hang_dong_ly_do', 'hang_theo_u_ly_do', 'P4_tay_don')
TOP_PER_BUOI = ('u_beta_cam_dau_chan', 'u_beta_cam_gia_dinh', 'u_beta_cam_ly_do')
ST_FIELDS = ('u_beta_cam_trang_thai', 'u_beta_cam_trang_thai_cho')
# worst first; a state outside this list must be the same in every session
WORST = ('thieu', 'khong_dat', 'chua_ket_luan', 'gioi_han', 'dat')
TOP_HANDLED = {'muc', 'khoa_truoc', 'mien_car_pose', 'u_beta_cam_deg', 'u_beta_cam_trang_thai_chung', 'trang_thai',
               'ly_do_chua_khoa', 'commit_hil', 'version'} | set(TOP_PER_BUOI) | set(ST_FIELDS)
MUC_HANDLED = {'hang_dong', 'hang_dong_truoc_le', 'hang_theo_u'} | set(MUC_PER_BUOI)
REL = 1e-12


class GopError(ValueError):
    pass


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _lon_nhat(vals, w, nguon):
    """(largest, session) of per-session numbers; any null -> (None, the first null session)."""
    for b, v in vals:
        if v is None:
            return None, b
        if _num(v) is None:
            raise GopError(f'{w}: {b} value {v!r} is not a finite number')
    b, v = max(vals, key=lambda t: float(t[1]))
    nguon[w] = b
    return float(v), b


def _te_nhat(vals, w):
    """Worst state over the sessions (WORST order); states outside WORST must agree."""
    s = [v for _, v in vals]
    if all(x in WORST for x in s):
        return min(s, key=WORST.index)
    if len(set(map(json.dumps, s))) != 1:
        raise GopError(f'{w}: states {dict(vals)} not in {WORST} and not the same')
    return s[0]


def _same(vals, w):
    s = [json.dumps(v, sort_keys=True, ensure_ascii=False) for _, v in vals]
    if len(set(s)) != 1:
        raise GopError(f'{w}: differs between the session locks ({", ".join(b for b, _ in vals)}): '
                       f'{"; ".join(x[:160] for x in s)}')
    return copy.deepcopy(vals[0][1])


def _theo_buoi(vals, w):
    s = [json.dumps(v, sort_keys=True, ensure_ascii=False) for _, v in vals]
    if len(set(s)) == 1:
        return copy.deepcopy(vals[0][1])
    if all(isinstance(v, list) and all(isinstance(x, str) for x in v) for _, v in vals):
        return [f'{b}: {x}' for b, v in vals for x in v]
    return dict(theo_buoi={b: copy.deepcopy(v) for b, v in vals})


def _keys_same(vals, w):
    ks = [sorted(v) if isinstance(v, dict) else None for _, v in vals]
    if any(k is None for k in ks) or any(k != ks[0] for k in ks):
        raise GopError(f'{w}: not dicts with the same keys in every session lock')
    return ks[0]


def gop(khoa_buoi, commit_hil, script=None, nguon_khoa=None):
    """khoa_buoi: [(buoi, lock dict)], nguon_khoa: {buoi: dict(file, duong_dan, sha256)}. Returns the combined lock."""
    if not khoa_buoi:
        raise GopError('no session lock')
    names = [b for b, _ in khoa_buoi]
    if len(set(names)) != len(names) or any(not b or '=' in b for b in names):
        raise GopError(f'session names {names}: one lock per session, names non-empty and distinct')
    for b, k in khoa_buoi:
        if not isinstance(k, dict) or k.get('version') != kv.VERSION:
            raise GopError(f'{b}: not a {kv.VERSION} lock')
        if k.get('trang_thai') != 'khoa':
            raise GopError(f'{b}: trang_thai {k.get("trang_thai")!r} ({k.get("ly_do_chua_khoa")!r}); only locked '
                           'session locks are combined')
        if k.get('commit_hil') != commit_hil or (k.get('khoa_truoc') or {}).get('commit_hil') != commit_hil:
            raise GopError(f'{b}: commit_hil {k.get("commit_hil")!r} is not --commit-hil {commit_hil!r} (one scoring '
                           'commit for every session lock and the combined lock; gate bo_tick)')
        if 'u_beta_cam_trang_thai_chung' not in k or not isinstance(k.get('u_beta_cam_trang_thai'), dict):
            raise GopError(f'{b}: made before N1 (u_beta_cam_trang_thai not per place); make it again at {commit_hil}')
    nguon_max = {}
    out = {}
    tops = [set(k) for _, k in khoa_buoi]
    for t in tops[1:]:
        if t != tops[0]:
            raise GopError(f'top-level fields differ between the session locks: {sorted(t ^ tops[0])}')
    for key in sorted(tops[0]):
        vals = [(b, k[key]) for b, k in khoa_buoi]
        if key in TOP_HANDLED:
            continue
        out[key] = _same(vals, key)
    # levels
    lvs = _keys_same([(b, k['muc']) for b, k in khoa_buoi], 'muc')
    muc = {}
    for lv in lvs:
        ms = [(b, k['muc'][lv]) for b, k in khoa_buoi]
        mk = _keys_same(ms, f'muc.{lv}')
        m = {}
        for f in mk:
            vals = [(b, x[f]) for b, x in ms]
            if f in ('hang_dong', 'hang_dong_truoc_le'):
                hk = _keys_same(vals, f'muc.{lv}.{f}')
                m[f] = {}
                for q in hk:
                    qv = [(b, x[q]) for b, x in vals]
                    m[f][q] = (_lon_nhat(qv, f'muc.{lv}.{f}.{q}', nguon_max)[0] if q in HD_MAX
                               else _same(qv, f'muc.{lv}.{f}.{q}'))
            elif f == 'hang_theo_u':
                n = {len(x) for _, x in vals if isinstance(x, list)}
                if len(n) != 1 or any(not isinstance(x, list) for _, x in vals):
                    raise GopError(f'muc.{lv}.hang_theo_u: not lists of the same length in every session lock')
                bands = []
                for i in range(n.pop()):
                    bv = [(b, x[i]) for b, x in vals]
                    bk = _keys_same(bv, f'muc.{lv}.hang_theo_u[{i}]')
                    band = {}
                    for q in bk:
                        qv = [(b, x[q]) for b, x in bv]
                        w = f'muc.{lv}.hang_theo_u[{i}].{q}'
                        if q in BAND_MAX:
                            band[q] = _lon_nhat(qv, w, nguon_max)[0]
                        elif q == 'trang_thai':
                            band[q] = _te_nhat(qv, w)
                        else:
                            band[q] = _same(qv, w)
                    bands.append(band)
                m[f] = bands
            elif f in MUC_PER_BUOI:
                m[f] = _theo_buoi(vals, f'muc.{lv}.{f}')
            else:
                m[f] = _same(vals, f'muc.{lv}.{f}')
        muc[lv] = m
    out['muc'] = muc
    # u_beta_cam per place
    cho = _keys_same([(b, k['u_beta_cam_deg']) for b, k in khoa_buoi], 'u_beta_cam_deg')
    out['u_beta_cam_deg'] = {c: _lon_nhat([(b, k['u_beta_cam_deg'][c]) for b, k in khoa_buoi],
                                          f'u_beta_cam_deg.{c}', nguon_max)[0] for c in cho}
    for f in ST_FIELDS:
        ck = _keys_same([(b, k[f]) for b, k in khoa_buoi], f)
        out[f] = {c: _te_nhat([(b, k[f][c]) for b, k in khoa_buoi], f'{f}.{c}') for c in ck}
        for c in ck:                         # a number from every session or none: the state follows the value
            if out[f][c] == 'dat' and out['u_beta_cam_deg'].get(c) is None:
                raise GopError(f'{f}.{c} dat with u_beta_cam_deg.{c} null')
    out['u_beta_cam_trang_thai_chung'] = _te_nhat([(b, k['u_beta_cam_trang_thai_chung']) for b, k in khoa_buoi],
                                                  'u_beta_cam_trang_thai_chung')
    for f in TOP_PER_BUOI:
        out[f] = _theo_buoi([(b, k[f]) for b, k in khoa_buoi], f)
    # camera domain: hull of kappa
    mcs = [(b, k['mien_car_pose']) for b, k in khoa_buoi]
    mkeys = _keys_same(mcs, 'mien_car_pose')
    mc = {}
    for q in mkeys:
        qv = [(b, x[q]) for b, x in mcs]
        if q == 'kappa':
            if any(not isinstance(v, list) or len(v) != 2 or any(_num(z) is None for z in v) for _, v in qv):
                raise GopError(f'mien_car_pose.kappa: not [lo, hi] in every session lock ({dict(qv)})')
            mc[q] = [min(v[0] for _, v in qv), max(v[1] for _, v in qv)]
        elif q == 'trang_thai':
            mc[q] = _te_nhat(qv, 'mien_car_pose.trang_thai')
        elif q == 'ly_do':
            mc[q] = _theo_buoi(qv, 'mien_car_pose.ly_do')
        else:
            mc[q] = _same(qv, f'mien_car_pose.{q}')
    out['mien_car_pose'] = mc
    # sources: one A6, one plan
    src, seen = [], set()
    for b, k in khoa_buoi:
        kt = k.get('khoa_truoc')
        if not isinstance(kt, dict) or not isinstance(kt.get('nguon'), list):
            raise GopError(f'{b}: khoa_truoc.nguon missing')
        for n in kt['nguon']:
            key = (n.get('vai'), n.get('sha256'))
            if key not in seen:
                seen.add(key)
                src.append(dict(n, buoi=b))
    for vai in ('a6', 'ke_hoach_c8'):
        shas = {n.get('sha256') for n in src if n.get('vai') == vai}
        if len(shas) != 1:
            raise GopError(f'{len(shas)} {vai} source(s) over the session locks; exactly one needed (gate _check_a6, '
                           'GC6-1: one A6, one plan for C8)')
    for b, _ in khoa_buoi:
        nk = (nguon_khoa or {}).get(b)
        if nk:
            src.append(dict(vai=f'khoa_buoi {b}', buoi=b, **nk))
    out.update(version=kv.VERSION, commit_hil=commit_hil,
               khoa_truoc=dict(nguon=src, script=script, commit_hil=commit_hil,
                               luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds')))
    # the scorer's formulas still hold
    for lv, m in muc.items():
        hd, tr = m.get('hang_dong') or {}, m.get('hang_dong_truoc_le') or {}
        for q in ('hang_m', 'he_so_r', 'hang_rad'):
            if (hd.get(q) is None) != (tr.get(q) is None) or (
                    hd.get(q) is not None and abs(hd[q] - kv.LE * tr[q]) > REL * max(1.0, abs(hd[q]))):
                raise GopError(f'muc.{lv}: hang_dong.{q} {hd.get(q)!r} != LE x hang_dong_truoc_le.{q} {tr.get(q)!r}')
        ub = out['u_beta_cam_deg'].get(kv.CHO_MUC)
        if hd.get('u_phi_rad') is not None:
            want = None if ub is None or tr.get('u_phi_rad') is None else \
                kv.LE * tr['u_phi_rad'] + math.radians(ub)
            if want is None or abs(hd['u_phi_rad'] - want) > REL * max(1.0, abs(want)):
                raise GopError(f'muc.{lv}: hang_dong.u_phi_rad {hd["u_phi_rad"]!r} != LE x truoc_le.u_phi_rad + '
                               f'u_beta_cam_deg.{kv.CHO_MUC} ({want!r}): the largest terms come from different sessions '
                               'with different u_phi_hat (F7); not combinable by the largest')
    st, why = kv.trang_thai_khoa(out, out.get('muc_kha_thi') or [])
    if st != 'khoa':
        raise GopError(f'combined lock not locked: {why}')
    out.update(trang_thai=st, ly_do_chua_khoa=why,
               gop=dict(version=GOP_VERSION, buoi=names, khoa_buoi=nguon_khoa or {},
                        lon_nhat_tu=nguon_max,
                        quy_tac='hang so: lon nhat qua cac buoi (Liam 24/11 o CODE local_42161302, hop chon A2, nguyen '
                                'van: "Mot khoa, lay lon nhat (Recommended)"); trang thai: te nhat; '
                                'mien_car_pose.kappa: bao [min lo, max hi]; cac truong khac phai trung'))
    return out


def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_khoa_gop.py', 'c8b_khoa_v5.py'],
                           capture_output=True, text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), file='tools/c8b_khoa_gop.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_khoa_gop.py')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--khoa-buoi', action='append', required=True,
                    help='BUOI=c8b_khoa_v5.json of one session (BUOI: the session folder, e.g. 2026-11-23_run03)')
    ap.add_argument('--commit-hil', required=True, help='the scoring commit (40 hex) of every session lock')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if not kv._hex40(a.commit_hil):
            raise GopError(f'--commit-hil {a.commit_hil!r} is not 40 hex')
        kb, nk = [], {}
        for s in a.khoa_buoi:
            b, sep, pth = s.partition('=')
            if not sep or not b:
                raise GopError(f'--khoa-buoi {s!r}: BUOI=path')
            cu.check_path(pth)
            with open(pth, encoding='utf-8') as f:
                kb.append((b, json.load(f)))
            nk[b] = dict(file=os.path.basename(pth), duong_dan=os.path.abspath(pth), sha256=cu.sha256(pth))
        out = gop(kb, a.commit_hil, script_state(), nk)
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(kv._clean(out), indent=1, ensure_ascii=False, allow_nan=False))
    except (GopError, kv.KhoaError, cu.InputError, OSError, ValueError) as exc:
        print(f'c8b_khoa_gop: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    for lv, m in out['muc'].items():
        hd = m['hang_dong']
        print(f'{lv}: hang_m {hd["hang_m"]} he_so_r {hd["he_so_r"]} u_phi_rad {hd["u_phi_rad"]} hang_rad '
              f'{hd["hang_rad"]}')
    print(f'u_beta_cam_deg {out["u_beta_cam_deg"]}; lon nhat tu {out["gop"]["lon_nhat_tu"]}')
    print(f'trang_thai {out["trang_thai"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
