#!/usr/bin/env python3
"""
c7_chuoi_khung.py: the common per-frame chain of the relative scoring (F1 'c7_chuoi_khung_v1' of the C8 data
contract CLAUDE_THESIS 03_CODE/docs/2026-11-19_Hop_dong_du_lieu_C8_tuong_doi_v1.md; design CODE_CAM v5
03_CODE/docs/2026-11-19_Thiet_ke_CAM_cham_tuong_doi_v5.md section 2; rules J v6 section 2). One path for every user:
c7_score (relative mode), c7_u_moc, c8_hc_a (CODE_HIEUCHUAN), NHACTAM.

Per frame of frames.csv (c7_measure track, ONE tag height, tag_offset (0, 0, 0)):
  1-2  tag centre t = (tag_x, tag_y) [m, floor] and tag heading psi_ma (column tag_psi when present, else psi of a
       track run with tag_offset (0, 0, 0), checked against track.json beside frames.csv);
  3    A6: psi'_ma = psi_ma - dpsi_A6, dpsi_A6 from c8b_a6.delta_psi_a6 of CODE_C8B (the one source of A6, contract
       section 0: no session computes A6 itself); the plate check c8b_a6.ap_dung(a6, tam_id);
  4    centre: p = t - R(psi'_ma) q, q = vector from the rotation centre to the tag centre in the tag frame (xoay_v1 of
       CODE_C8B, fitted on psi' after A6, contract F4); a run on another mounting than the E clip needs the composed
       q of C8B (q_B,dich = q_B,chuan,k + delta_c_B, CODE 19/11) and delta_c_B 'dat' in lan_lap_v1 (NHACTAM);
  gate the frame gates of c7_measure already applied by track (valid) and the car tag seen (car_seen).
A frame is 'khoa' (usable for numbers that decide) when the gate, A6 and q all hold; else 'chan_doan' with the
reasons. The tag-to-axis angle a cancels in the scoring because only CHANGES of the tag heading are used (J v6 3).

trung_binh_go_vong: the circular mean c, then c + mean(wrap(g - c)) (NHACTAM / HIEUCHUAN mean_unwrapped), so the
difference of two such means is the mean of the differences when the spread is small.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PHIEN_BAN = 'c7_chuoi_khung_v1'
A6_SCHEMA = 'c8b_a6_v2'
XOAY_SCHEMA = 'xoay_v1'
LAN_LAP_SCHEMA = 'lan_lap_v1'
PLUMB_16_11 = (-0.4802, -0.0329)     # m, plumb point of 16/11, every session (L125: never measured again)
ZC_16_11 = 2.632                     # m, camera height of 16/11 (L125)
ROW_KEYS = ('frame_id', 't_cam', 'tag_x', 'tag_y', 'tag_psi', 'dpsi_a6', 'psi_ma_p', 'p_x', 'p_y', 'u_q_m', 'cong_ok',
            'a6_ok', 'q_ok', 'ly_do', 'trang_thai')


def wrap(a):
    """Angle in (-pi, pi]."""
    a = np.asarray(a, float)
    w = np.mod(a + math.pi, 2.0 * math.pi) - math.pi
    w = np.where(w == -math.pi, math.pi, w)
    return float(w) if w.ndim == 0 else w


def trung_binh_go_vong(goc_rad):
    """Mean of angles [rad] after unwrapping each around their circular mean (NaN for no finite angle)."""
    g = np.asarray(goc_rad, float).ravel()
    g = g[np.isfinite(g)]
    if not len(g):
        return float('nan')
    c = math.atan2(float(np.mean(np.sin(g))), float(np.mean(np.cos(g))))
    return float(c + np.mean(wrap(g - c)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _load_json(src, what):
    """(dict, path or None, sha256 or None) of a path or a dict."""
    if src is None:
        return None, None, None
    if isinstance(src, dict):
        return src, None, None
    with open(src, encoding='utf-8') as f:
        return json.load(f), os.path.abspath(src), sha256(src)


# ---------------------------------------------------------------- inputs of other sessions
def doc_a6(path):
    """A6 of CODE_C8B (c8b_a6.load_a6, schema c8b_a6_v2) wrapped with its path and sha256; None without a file.
    c8b_a6 is imported here, not at module load, so this module works (every frame 'chan_doan') before it is merged."""
    if path is None:
        return None
    import c8b_a6                                     # CODE_C8B, contract section 2.C
    return dict(obj=c8b_a6.load_a6(path), mod=c8b_a6, path=os.path.abspath(path), sha256=sha256(path))


def doc_xoay(src, cho=None):
    """xoay_v1 of CODE_C8B (contract F4): q_tag_m [x, y] (rotation centre -> tag centre, tag frame, fitted on psi'
    after A6), u_q_m, trang_thai; per place under 'cho' {goc, K1, ...} (C8B ghep_q, hil 6866824: key 'version') or
    flat. Returns dict(q, u_q, ok, ly_do, path, sha256, a6_sha256, raw); ok False when missing, without the place, or
    not 'dat'. raw: the whole file, for c8b_xoay.ghep_q."""
    d, path, sha = _load_json(src, 'xoay')
    out = dict(q=None, u_q=None, ok=False, ly_do='', path=path, sha256=sha, a6_sha256=None, raw=d)
    if d is None:
        out['ly_do'] = 'thieu_xoay'
        return out
    tag = d.get('schema', d.get('version'))
    if tag not in (None, XOAY_SCHEMA):
        raise ValueError(f'xoay: schema/version {tag!r}, expected {XOAY_SCHEMA}')
    if isinstance(d.get('cho'), dict):
        top_a6 = d.get('a6_sha256')
        if cho is None or not isinstance(d['cho'].get(cho), dict):
            out['ly_do'] = f'xoay_thieu_cho_{cho}'
            return out
        d = dict(d['cho'][cho])
        d.setdefault('a6_sha256', top_a6)
    q = d.get('q_tag_m')
    try:
        q = [float(q[0]), float(q[1])]
        u = float(d.get('u_q_m'))
    except (TypeError, ValueError, IndexError):
        out['ly_do'] = 'xoay_thieu_q'
        return out
    if not all(math.isfinite(v) for v in q + [u]) or u < 0.0:
        out['ly_do'] = 'xoay_q_khong_huu_han'
        return out
    out.update(q=q, u_q=u, a6_sha256=d.get('a6_sha256'))
    st = (d.get('trang_thai') if not isinstance(d.get('trang_thai'), dict) else d['trang_thai'].get('q'))
    if st != 'dat':
        out['ly_do'] = f'xoay_trang_thai_{st}'
        return out
    out['ok'] = True
    return out


# F8 rong (CODE 23/11, 03_CODE/docs/2026-11-23_Chot_T12_khoa_C8B_v1.md muc 4, goc 934301af; Chot F8 rong muc 1; Liam 23/11
# at CODE "Dung mot lan lap (Recommended)", "Cach A (Recommended)"): an F8 that passed its identity checks
# (c7_tuong_doi.f8_rong) but gives no run a lan_lap_id -> q of xoay_v1 of the place (K2, no E turn: the place of
# F8_RONG_Q_TU, the one c8_hc_a of HIEUCHUAN borrows from, --q-tu F5 HC_K1_1 in chay_T9b.sh of 97823b4), u_q raised by
# the plate shift bound of session 1 (lift, put back: position 0.57 mm, P3; CAM f126d362), labelled F8_RONG_NHAN.
F8_RONG_NHAN = 'can_lech_tam_buoi1_chua_xac_minh_lan_lap'
U_LECH_TAM_BUOI1_M = 0.00057
F8_RONG_Q_TU = {'K2': 'K1'}


def doc_xoay_cung_lan_lap(src, cho, cho_q=None, ll_id=None, ll_id_q=None, f8_rong=False):
    """doc_xoay of the place cho; a place WITHOUT its own block in a per-place xoay_v1 (K2: no E turn there) takes,
    CODE 22/11 option (A) (contract C8 section 3, "F4 cho K2"), the block of the place cho_q WITH E on the SAME
    mounting of the plate: q is the tag centre seen from the turning centre, fixed while the plate is not taken off.
    The mounting is verified by ll_id (lan_lap_id of the F5 of cho) = ll_id_q (lan_lap_id of the F5 of cho_q), both
    from F8 run 1, non-empty strings. Another mounting, an id missing (not verified) or no cho_q: no q (ok False,
    ly_do says why), so u_q is missing and the place is chua_ket_luan. A place WITH its own block always uses it
    (cho_q ignored). Returns the doc_xoay dict plus nguon_q {cho, lan_lap_id, cach} when a q is taken; cho_q = cho:
    ValueError. f8_rong (F8 rong, the caller checked c7_tuong_doi.f8_rong): the mounting is not compared by ids (the
    F8 gives none) but taken as one mounting (F8_RONG_NHAN); two non-empty ids that differ are still refused; u_q is
    raised by q_luot, not here."""
    xo = doc_xoay(src, cho)
    raw = xo.get('raw')
    per_place = isinstance(raw, dict) and isinstance(raw.get('cho'), dict)
    if xo['q'] is not None or not per_place or isinstance(raw['cho'].get(cho), dict):
        if xo['q'] is not None:
            xo['nguon_q'] = dict(cho=cho, lan_lap_id=ll_id, cach='khoi F4 cua chinh cho')
        return xo
    if cho_q is None:
        return xo                                              # xoay_thieu_cho_<cho>, as before
    if cho_q == cho:
        raise ValueError(f'cho_q {cho_q!r} is the place itself')
    s = lambda v: isinstance(v, str) and bool(v)                  # noqa: E731
    if s(ll_id) and s(ll_id_q) and ll_id != ll_id_q:
        xo['ly_do'] = f'khac_lan_lap_{cho_q} (lan_lap_id {cho} {ll_id!r}, {cho_q} {ll_id_q!r})'
        return xo
    if not f8_rong and not (s(ll_id) and s(ll_id_q)):
        xo['ly_do'] = f'chua_xac_minh_cung_lan_lap_{cho_q} (lan_lap_id {cho} {ll_id!r}, {cho_q} {ll_id_q!r})'
        return xo
    xq = doc_xoay(src, cho_q)
    if xq['q'] is None or not xq['ok']:
        xo['ly_do'] = f'cho_{cho_q}_cung_lan_lap:{xq["ly_do"]}'
        return xo
    xq['nguon_q'] = (dict(cho=cho_q, lan_lap_id=None, nhan=F8_RONG_NHAN,
                          cach='khoi F4 cua cho co E, F8 rong: mot lan lap (CODE 23/11 chot T12 muc 4)') if f8_rong
                     else dict(cho=cho_q, lan_lap_id=ll_id, cach='khoi F4 cua cho co E cung lan lap tam (CODE 22/11 A)'))
    return xq


LAN_LAP_FILE_KEYS = ('cua_so_v1_sha256', 'session_sha256', 'tam_id', 'hieu_chuan', 'a6')


def doc_lan_lap(src, luot_id=None):
    """lan_lap_v1 of NHACTAM (contract F8) as the dict of ONE run with path and sha256 (of the file); None without it.
    The file of a session (c7_nhac_tam lan_lap: file keys + 'luot' list) gives the run whose luot_id is given, with
    the file keys LAN_LAP_FILE_KEYS copied in when the run lacks them (Codex CT1-3: cua_so_v1_sha256, session); a
    file of several runs without luot_id, or without that run: ValueError."""
    d, path, sha = _load_json(src, 'lan_lap')
    if d is None:
        return None
    if d.get('schema') not in (None, LAN_LAP_SCHEMA):
        raise ValueError(f'lan_lap: schema {d.get("schema")!r}, expected {LAN_LAP_SCHEMA}')
    if isinstance(d.get('luot'), list):
        runs = [r for r in d['luot'] if isinstance(r, dict)]
        if luot_id is None:
            if len(runs) != 1:
                raise ValueError(f'lan_lap_v1 holds {len(runs)} runs: give the run id (luot_id of cua_so_v1 pre)')
            pick = runs
        else:
            pick = [r for r in runs if r.get('luot_id') == luot_id]
            if len(pick) != 1:
                raise ValueError(f'lan_lap_v1: {len(pick)} runs with luot_id {luot_id!r} (need exactly one)')
        rec = dict(pick[0])
        for k in LAN_LAP_FILE_KEYS:
            if k not in rec and k in d:
                rec[k] = d[k]
        # F8 rong (CODE 23/11, Chot_T12 muc 4 cach A; Chot_F8_rong muc 1b): no run of the FILE has a lan_lap_id
        rec['_moi_lan_lap_id_null'] = bool(runs) and all(r.get('lan_lap_id') is None for r in runs)
        d = rec
    else:
        d = dict(d)
        d['_moi_lan_lap_id_null'] = d.get('lan_lap_id') is None
    d['_path'], d['_sha256'] = path, sha
    return d


def _trang_thai(d, key):
    tt = d.get('trang_thai')
    return tt.get(key) if isinstance(tt, dict) else None


def q_luot(xoay, lan_lap, cho=None, a6_sha256=None, truc_xe=None, u_psi_w_rad=None, f8_rong_ok=False):
    """(q [x, y] or None, u_q [m] or None, ly_do) of the run. Same mounting as the E clip (lan_lap absent: the
    calibration clips of the same session, 12b / E; or cung_lan_lap true): q of xoay_v1. Another mounting: the q
    composed by CODE_C8B, c8b_xoay.ghep_q(lan_lap, cho, xoay_v1, truc_xe, u_psi_w_rad) -> {q_m, u_q_m, trang_thai,
    ly_do} (contract F16, signature of hil 6866824: q_m = R(-theta_luot) (R(theta_chuan) q_tag + delta_c_B), before
    the margin; u_psi_w_rad: noise of the mean tag heading over the run window, given by the caller); not 'dat', not importable or no place: no q (the
    position through q is not concluded). a6_sha256 (the A6 file used for psi'): q was fitted on psi' after ONE A6
    (xoay_v1 a6_sha256, contract F4); missing there or another A6: no q (Codex CT1-3). f8_rong_ok (F8 rong, checked by
    the caller with c7_tuong_doi.f8_rong): q of xoay_v1 as for one mounting, u_q + U_LECH_TAM_BUOI1_M, reason
    F8_RONG_NHAN (CODE 23/11 chot T12 muc 4)."""
    if not xoay['ok']:
        return None, None, xoay['ly_do']
    if a6_sha256 is not None:
        if not xoay.get('a6_sha256'):
            return None, None, 'xoay_thieu_a6_sha256'
        if xoay['a6_sha256'] != a6_sha256:
            return None, None, 'xoay_a6_khac_a6_dang_dung'
    if lan_lap is None or lan_lap.get('cung_lan_lap') is True:
        return list(xoay['q']), float(xoay['u_q']), ('khong_kiem_lan_lap' if lan_lap is None else '')
    if f8_rong_ok:
        return list(xoay['q']), float(xoay['u_q']) + U_LECH_TAM_BUOI1_M, F8_RONG_NHAN
    if _trang_thai(lan_lap, 'delta_c_B') != 'dat':
        return None, None, 'thieu_delta_c_B'
    if cho is None:
        return None, None, 'thieu_cho_cho_ghep_q'
    try:
        import c8b_xoay                                   # CODE_C8B, contract F16
        g = c8b_xoay.ghep_q
    except (ImportError, AttributeError):
        return None, None, 'cho_ham_ghep_q_C8B'
    try:
        r = g({k: v for k, v in lan_lap.items() if not k.startswith('_')}, cho, xoay.get('raw'), truc_xe, u_psi_w_rad)
    except TypeError:
        return None, None, 'cho_ham_ghep_q_C8B (chu ky cu)'

    if not isinstance(r, dict) or r.get('trang_thai') != 'dat':
        return None, None, 'ghep_q_' + str((r or {}).get('trang_thai')) + ':' + str((r or {}).get('ly_do', ''))
    try:
        q, u = [float(r['q_m'][0]), float(r['q_m'][1])], float(r['u_q_m'])
    except (KeyError, TypeError, ValueError, IndexError):
        return None, None, 'ghep_q_sai_dinh_dang'
    if not all(math.isfinite(v) for v in q + [u]) or u < 0.0:
        return None, None, 'ghep_q_khong_huu_han'
    return q, u, ''


# ---------------------------------------------------------------- frames
def _frames(frames_csv):
    """(rows, path or None): rows of c7_measure.read_frames, or the rows given."""
    if isinstance(frames_csv, (list, tuple)):
        return list(frames_csv), None
    import c7_measure as cm
    return cm.read_frames(frames_csv), os.path.abspath(frames_csv)


def _tag_heading_source(rows, frames_path, track):
    """'tag_psi' when the column exists (whatever its cells: a blank cell is a missing value of that frame, Codex
    CT1-18); 'psi' when track.json (given, or beside frames.csv) has tag_offset (0, 0, 0); else raise (the heading of
    frames.csv is not the tag heading)."""
    if rows and 'tag_psi' in rows[0]:
        return 'tag_psi'
    if track is None and frames_path is not None:
        cand = os.path.join(os.path.dirname(frames_path), 'track.json')
        track = cand if os.path.isfile(cand) else None
    if track is None:
        raise ValueError('frames.csv has no tag_psi column and no track.json to check tag_offset (0, 0, 0): psi '
                         'may not be the tag heading')
    t, _, _ = _load_json(track, 'track')
    off = t.get('tag_offset')
    if off is None or len(off) != 3 or any(abs(float(v)) > 0.0 for v in off):
        raise ValueError(f'track.json tag_offset {off}: the chain needs a track run with tag_offset (0, 0, 0)')
    return 'psi'


def _f(v):
    """A cell as a float: a number, or NaN for a blank / non-number cell (frames.csv keeps extra columns as text)."""
    if v is None or v == '':
        return float('nan')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def tam_cua_luot(lan_lap, tam_id=None):
    """(tam_id for c8b_a6.ap_dung, ly_do). The plate of the run: tam_id of lan_lap_v1 (NHACTAM); without lan_lap, or a
    lan_lap without tam_id, the tam_id given by the caller (CODE 20/11, HIEUCHUAN HCV3-1: HIEUCHUAN reads it from
    c8b_a6_v2.tam_id, one source, and cannot give --lan-lap because F8 is built from F7). Both given and different:
    (None, 'a6_tam_id_khac_lan_lap'), the A6 is not applied."""
    ll_tam = None if lan_lap is None else lan_lap.get('tam_id')
    if ll_tam is None:
        return tam_id, ''
    if tam_id is not None and tam_id != ll_tam:
        return None, 'a6_tam_id_khac_lan_lap'
    return ll_tam, ''


def chuoi_khung(frames_csv, a6, xoay, lan_lap=None, plumb=PLUMB_16_11, zc=ZC_16_11, track=None, cho=None,
                truc_xe=None, u_psi_w_rad=None, tam_id=None, cho_q=None, ll_id=None, ll_id_q=None, f8_rong_ok=False):
    """Rows (dicts with ROW_KEYS) of every frame of frames.csv (module docstring). a6: doc_a6(...) or None; xoay: path,
    dict or doc_xoay(...) result; lan_lap: path, dict or doc_lan_lap(...) result, or None (calibration clips of the
    session of E: same mounting); cho: place k of the run (goc, K1, K2) for the C8B q composition of another
    mounting; truc_xe (truc_xe_v1 dict) and u_psi_w_rad: passed to the C8B q composition of another mounting; tam_id:
    the plate on the car, used for the A6 plate check when lan_lap is None or has no tam_id (tam_cua_luot); cho_q,
    ll_id, ll_id_q: a place without its own F4 block (K2) takes the block of cho_q on the same mounting
    (doc_xoay_cung_lan_lap, CODE 22/11 option A). f8_rong_ok (F8 rong, the caller checked c7_tuong_doi.f8_rong; CODE
    23/11 chot T12 muc 4): q as for one mounting with u_q + U_LECH_TAM_BUOI1_M (q_luot); K2 without cho_q borrows from
    F8_RONG_Q_TU. Angles in rad, positions in m."""
    rows, fpath = _frames(frames_csv)
    if f8_rong_ok and cho_q is None:
        cho_q = F8_RONG_Q_TU.get(cho)
    xo = xoay if (isinstance(xoay, dict) and 'ok' in xoay and 'q' in xoay) else \
        doc_xoay_cung_lan_lap(xoay, cho, cho_q, ll_id, ll_id_q, f8_rong=f8_rong_ok)   # cho_q None: doc_xoay as before
    ll = lan_lap if (lan_lap is None or '_sha256' in lan_lap) else doc_lan_lap(lan_lap)
    src = _tag_heading_source(rows, fpath, track)
    n = len(rows)
    tx = np.array([_f(r.get('tag_x')) for r in rows], float)
    ty = np.array([_f(r.get('tag_y')) for r in rows], float)
    tp = np.array([_f(r.get(src)) for r in rows], float)
    gate = np.array([bool(r.get('valid')) and bool(r.get('car_seen')) for r in rows], bool)
    finite = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(tp)
    cong = gate & finite
    # A6 (CODE_C8B): plate check, then the per-frame correction
    why_a6 = [''] * n
    if a6 is None:
        dps, a6ok = np.full(n, np.nan), np.zeros(n, bool)
        why_a6 = ['thieu_a6'] * n
    else:
        tam, why_tam = tam_cua_luot(ll, tam_id)
        st, ly = (None, why_tam) if why_tam else a6['mod'].ap_dung(a6['obj'], tam)
        if why_tam:                                   # two plates named: the A6 is not applied, every frame diagnosed
            dps, a6ok = np.full(n, np.nan), np.zeros(n, bool)
            why_a6 = [why_tam] * n
        elif st != 'dat' and st != 'khong_kiem_cheo_tay':
            dps, a6ok = np.full(n, np.nan), np.zeros(n, bool)
            why_a6 = [f'a6_{st}:{ly}'] * n
        else:
            dps, a6ok = np.full(n, np.nan), np.zeros(n, bool)
            if finite.any():
                xy = np.column_stack([tx[finite], ty[finite]])
                d, ok, why = a6['mod'].delta_psi_a6(a6['obj'], xy, tp[finite], plumb, zc)
                d, ok = np.asarray(d, float), np.asarray(ok, bool)
                dps[finite], a6ok[finite] = d, ok & np.isfinite(d)
                for j, k in enumerate(np.nonzero(finite)[0]):
                    if not a6ok[k]:
                        why_a6[k] = f'a6_khung:{why[j]}'
    psi_p = np.where(a6ok, tp - np.nan_to_num(dps), np.nan)
    q, uq, why_q = q_luot(xo, ll, cho, None if a6 is None else a6['sha256'], truc_xe, u_psi_w_rad, f8_rong_ok)
    if q is not None:
        c, s = np.cos(psi_p), np.sin(psi_p)
        px = tx - (c * q[0] - s * q[1])
        py = ty - (s * q[0] + c * q[1])
    else:
        px, py = np.full(n, np.nan), np.full(n, np.nan)
    qok = (q is not None) & np.isfinite(px) & np.isfinite(py)
    out = []
    for k, r in enumerate(rows):
        why = []
        if not gate[k]:
            why.append('cong_khung')
        elif not finite[k]:
            why.append('ma_khong_huu_han')
        if why_a6[k]:
            why.append(why_a6[k])
        if q is None:
            why.append(why_q)
        elif why_q:
            why.append(why_q)
        ok = bool(cong[k] and a6ok[k] and qok[k])
        out.append(dict(frame_id=int(r['idx']), t_cam=float(r['t_s']), tag_x=float(tx[k]), tag_y=float(ty[k]),
                        tag_psi=float(tp[k]), dpsi_a6=float(dps[k]), psi_ma_p=float(psi_p[k]), p_x=float(px[k]),
                        p_y=float(py[k]), u_q_m=None if uq is None else float(uq), cong_ok=bool(cong[k]),
                        a6_ok=bool(a6ok[k]), q_ok=bool(qok[k]),
                        ly_do=';'.join(w for w in why if w and w != 'khong_kiem_lan_lap'),
                        trang_thai='khoa' if ok else 'chan_doan'))
    return out


def nguon(frames_csv=None, a6=None, xoay=None, lan_lap=None):
    """Source block of the chain: version and {vai_tro, duong_dan, sha256} of each input (contract section 0)."""
    src = [dict(vai_tro='chuoi_khung', phien_ban=PHIEN_BAN)]
    if isinstance(frames_csv, str):
        src.append(dict(vai_tro='frames', duong_dan=os.path.abspath(frames_csv), sha256=sha256(frames_csv)))
    if a6 is not None:
        src.append(dict(vai_tro=A6_SCHEMA, duong_dan=a6['path'], sha256=a6['sha256']))
    if isinstance(xoay, dict) and xoay.get('path'):
        src.append(dict(vai_tro=XOAY_SCHEMA, duong_dan=xoay['path'], sha256=xoay['sha256']))
    elif isinstance(xoay, str):
        src.append(dict(vai_tro=XOAY_SCHEMA, duong_dan=os.path.abspath(xoay), sha256=sha256(xoay)))
    if isinstance(lan_lap, dict) and lan_lap.get('_path'):
        src.append(dict(vai_tro=LAN_LAP_SCHEMA, duong_dan=lan_lap['_path'], sha256=lan_lap['_sha256']))
    elif isinstance(lan_lap, str):
        src.append(dict(vai_tro=LAN_LAP_SCHEMA, duong_dan=os.path.abspath(lan_lap), sha256=sha256(lan_lap)))
    return src
