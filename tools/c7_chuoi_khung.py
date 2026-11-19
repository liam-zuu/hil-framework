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


def doc_xoay(src):
    """xoay_v1 of CODE_C8B (contract F4): q_tag_m [x, y] (rotation centre -> tag centre, tag frame, fitted on psi'
    after A6), u_q_m, trang_thai. Returns dict(q, u_q, ok, ly_do, path, sha256); ok False when missing or not 'dat'."""
    d, path, sha = _load_json(src, 'xoay')
    out = dict(q=None, u_q=None, ok=False, ly_do='', path=path, sha256=sha, a6_sha256=None)
    if d is None:
        out['ly_do'] = 'thieu_xoay'
        return out
    if d.get('schema') not in (None, XOAY_SCHEMA):
        raise ValueError(f'xoay: schema {d.get("schema")!r}, expected {XOAY_SCHEMA}')
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


def doc_lan_lap(src):
    """lan_lap_v1 of NHACTAM (contract F8) as a dict with path and sha256; None without it."""
    d, path, sha = _load_json(src, 'lan_lap')
    if d is None:
        return None
    if d.get('schema') not in (None, LAN_LAP_SCHEMA):
        raise ValueError(f'lan_lap: schema {d.get("schema")!r}, expected {LAN_LAP_SCHEMA}')
    d = dict(d)
    d['_path'], d['_sha256'] = path, sha
    return d


def _trang_thai(d, key):
    tt = d.get('trang_thai')
    return tt.get(key) if isinstance(tt, dict) else None


def q_luot(xoay, lan_lap, cho=None):
    """(q [x, y] or None, u_q [m] or None, ly_do) of the run. Same mounting as the E clip (lan_lap absent: the
    calibration clips of the same session, 12b / E; or cung_lan_lap true): q of xoay_v1. Another mounting: the q
    composed by CODE_C8B, c8b_xoay.ghep_q(lan_lap, cho) -> {q_m, u_q_m, trang_thai, ly_do} (contract F16: q_m =
    R(-theta_luot) (q_B_chuan_k + delta_c_B), before the margin); not 'dat', not importable or no place: no q (the
    position through q is not concluded)."""
    if not xoay['ok']:
        return None, None, xoay['ly_do']
    if lan_lap is None or lan_lap.get('cung_lan_lap') is True:
        return list(xoay['q']), float(xoay['u_q']), ('khong_kiem_lan_lap' if lan_lap is None else '')
    if _trang_thai(lan_lap, 'delta_c_B') != 'dat':
        return None, None, 'thieu_delta_c_B'
    if cho is None:
        return None, None, 'thieu_cho_cho_ghep_q'
    try:
        import c8b_xoay                                   # CODE_C8B, contract F16
        g = c8b_xoay.ghep_q
    except (ImportError, AttributeError):
        return None, None, 'cho_ham_ghep_q_C8B'
    r = g({k: v for k, v in lan_lap.items() if not k.startswith('_')}, cho)
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
    """'tag_psi' when the column exists; 'psi' when track.json (given, or beside frames.csv) has tag_offset (0, 0, 0);
    else raise (the heading of frames.csv is not the tag heading)."""
    if rows and 'tag_psi' in rows[0] and rows[0]['tag_psi'] not in (None, ''):
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


def chuoi_khung(frames_csv, a6, xoay, lan_lap=None, plumb=PLUMB_16_11, zc=ZC_16_11, track=None, cho=None):
    """Rows (dicts with ROW_KEYS) of every frame of frames.csv (module docstring). a6: doc_a6(...) or None; xoay: path,
    dict or doc_xoay(...) result; lan_lap: path, dict or doc_lan_lap(...) result, or None (calibration clips of the
    session of E: same mounting); cho: place k of the run (goc, K1, K2) for the C8B q composition of another
    mounting. Angles in rad, positions in m."""
    rows, fpath = _frames(frames_csv)
    xo = xoay if (isinstance(xoay, dict) and 'ok' in xoay and 'q' in xoay) else doc_xoay(xoay)
    ll = lan_lap if (lan_lap is None or '_sha256' in lan_lap) else doc_lan_lap(lan_lap)
    src = _tag_heading_source(rows, fpath, track)
    n = len(rows)
    tx = np.array([float(r.get('tag_x', float('nan'))) for r in rows], float)
    ty = np.array([float(r.get('tag_y', float('nan'))) for r in rows], float)
    tp = np.array([float(r.get(src, float('nan'))) for r in rows], float)
    gate = np.array([bool(r.get('valid')) and bool(r.get('car_seen')) for r in rows], bool)
    finite = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(tp)
    cong = gate & finite
    # A6 (CODE_C8B): plate check, then the per-frame correction
    why_a6 = [''] * n
    if a6 is None:
        dps, a6ok = np.full(n, np.nan), np.zeros(n, bool)
        why_a6 = ['thieu_a6'] * n
    else:
        tam = None if ll is None else ll.get('tam_id')
        st, ly = a6['mod'].ap_dung(a6['obj'], tam)
        if st != 'dat' and st != 'khong_kiem_cheo_tay':
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
    q, uq, why_q = q_luot(xo, ll, cho)
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
