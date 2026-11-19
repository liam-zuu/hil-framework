#!/usr/bin/env python3
"""
c8b_khoa_v5.py: the C8b lock per LEVEL for the relative scoring (design C8B v7 sections 0.2, 0.7, 5a, 5c, 6c, 10;
J v6 = O1 v11.41 appendix 06_SPECS/2026-11-19_Spec_O1_v11.41_Phu_luc_J_v6.md section 6). Written by CODE_C8B
local_07e5654d on feat/c8b-3b-ga, 19/11/2026 (CODE work order 19/11; Liam at CODE: "duyet ca 8, giao viec cac phien phu
theo cach nhanh", "Code ngay (Recommended)").

Every index from the other sessions is BEFORE the margin; this file multiplies by LE = 1.25 ONCE. A term without its
source is null with a reason (design 0.2: a missing term makes every bound using it not finite; never 0).

Per level (O1-F02, O1-F05, O1-F08; the K tasks take the F05 3b numbers only for the frames they use, J4-7):
  can_quay      turning bounds of the figure-8 moving frames (P4 position, A4 heading) from c8b_3b (one JSON per
                level): dung_duoc true -> p_m = LE d_p_quay_m, psi_deg = LE d_psi_quay_deg; L131 (Liam 19/11 at CODE,
                "Dung so, ghi gioi han (Recommended)": P4 of A2 finite, limits G3B-2, G3B-3) and L132 ("Dung B buoi 1
                ke ca troi (Recommended)": B of session 1 with its drift) recorded as limits; dung_duoc false -> null,
                trang_thai thieu: J v6 item 12(b): G2, G3, G5 of the figure 8 report only, G4 moving frames undefined,
                the level not lockable, CAN LIAM QUYET. Standing frames (design 0.7) need no turning bound.
  man_trap      the shutter branch of the level (J v6 J4-13): 'alpha' (|alpha_r| Dr + |alpha_c| Dc, LE x alpha) only when
                the 3b of that level says use_alpha, else 'mot_khung' (one frame, NO margin); a run outside the checked
                domain of the alpha branch falls back to one frame (J5-10, the scorer's job).
  chi_bao_cao   conditions always report only at the level: G5 of the figure 8 at F08 (J v6 section 6, Liam 19/11
                "chi bao cao, khong lam bu"); the 2b mask of the other conditions is the gate's (rq3_log / gate_c8).
  u_beta_cam_deg {goc, K1, K2}: LE x the largest change of beta under the map noise at the two tape footprints,
                from NHACTAM lan_lap beta_cam_replay (data contract F15 b, F16): max_abs_dbeta_rad per place; missing ->
                null and u_beta_cam_trang_thai thieu (CAM: u_phi not finite, the lever-arm terms not concluded).
  khac          the other terms of design section 1 (A6, P1/A3, P5, P6, U_thang, W, U_p_tc_san, delta_q): null with the
                reason 'chua co nguon' until their sessions deliver (KIEMCHEO, NHACTAM, HIEUCHUAN, CAM).
khoa_truoc: SHA-256 of every input and of this script (commit, dirty) and the time, written before any held-out part
is looked at (design 0.3).

  python tools/c8b_khoa_v5.py --tb O1-F02=3b_F02.json --tb O1-F05=3b_F05.json --tb O1-F08=3b_F08.json \
      [--lan-lap lan_lap.json] --out c8b_khoa_v5.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_khoa_v5'
LE = 1.25
LEVELS = ('O1-F02', 'O1-F05', 'O1-F08')
CHO = ('goc', 'K1', 'K2')
TB_VERSION = 'c8b_3b_v2'
KHAC = ('A6', 'P1_A3', 'P5', 'P6', 'U_thang', 'W_canh', 'U_p_tc_san', 'delta_q')
L131 = ('L131, Liam 19/11 phien CODE local_e6d36dbe: "Dung so, ghi gioi han (Recommended)"; gioi han nhan dang '
        'G3B-2 (sai camera dang R(psi) a khong tach khoi q), G3B-3 (sai nhu nhau tren ca mien cham bi hap thu)')
L132 = ('L132, Liam 19/11 phien CODE local_e6d36dbe: "Dung B buoi 1 ke ca troi (Recommended)"; gioi han: khong tach '
        'troi ga voi sai camera')


class KhoaError(Exception):
    pass


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def can_quay(tb, level):
    """Turning bounds of one level from its c8b_3b JSON (after the margin), or null with the reason."""
    if not isinstance(tb, dict) or tb.get('version') != TB_VERSION:
        return dict(p_m=None, psi_deg=None, trang_thai='thieu', ly_do=[f'no {TB_VERSION} file for {level}'])
    if tb.get('task_rates') != level:
        raise KhoaError(f'3b file for {tb.get("task_rates")} given as {level}')
    p, s = _num(tb.get('d_p_quay_m')), _num(tb.get('d_psi_quay_deg'))
    if tb.get('dung_duoc') is not True or p is None or s is None:
        why = [f'3b dung_duoc {tb.get("dung_duoc")!r}'] + [f'thieu: {x}' for x in tb.get('thieu') or []]
        if tb.get('ga_ok') is not True and not tb.get('b_dung_ke_ca_troi'):
            why.append('kiem ga khong dat')
        return dict(p_m=None, psi_deg=None, trang_thai='thieu', ly_do=why,
                    he_qua='J v6 muc 12(b): G2, G3, G5 so 8 chi bao cao; G4 khung dong khong xac dinh; muc khong khoa '
                           'duoc; CAN LIAM QUYET bo muc hay cho')
    lim = [L131] + ([L132] if tb.get('b_dung_ke_ca_troi') else []) + list(tb.get('gioi_han') or [])
    return dict(p_m=LE * p, psi_deg=LE * s, p_truoc_le_m=p, psi_truoc_le_deg=s, trang_thai='gioi_han', ly_do=[],
                gioi_han=lim, b_dung_ke_ca_troi=bool(tb.get('b_dung_ke_ca_troi')))


def man_trap(tb):
    sh = ((tb or {}).get('B') or {}).get('shutter') or {}
    if sh.get('use_alpha') and _num(sh.get('alpha_r_le_s_px')) is not None and _num(sh.get('alpha_c_le_s_px')) is not None:
        return dict(nhanh='alpha', alpha_r_s_px=abs(float(sh['alpha_r_le_s_px'])),
                    alpha_c_s_px=abs(float(sh['alpha_c_le_s_px'])), le=LE)
    return dict(nhanh='mot_khung', mot_khung_s=_num(sh.get('mot_khung_s')), le=None,
                ly_do=sh.get('why') or 'no 3b shutter fit')


def u_beta_cam(lan_lap):
    """u_beta_cam_deg {goc, K1, K2} after the margin from NHACTAM beta_cam_replay (F15 b); missing -> null."""
    rep = (lan_lap or {}).get('beta_cam_replay') if isinstance(lan_lap, dict) else None
    out, why = {}, []
    for c in CHO:
        v = _num(((rep or {}).get(c) or {}).get('max_abs_dbeta_rad'))
        st = ((rep or {}).get(c) or {}).get('trang_thai')
        if v is None or st != 'dat':
            out[c] = None
            why.append(f'{c}: beta_cam_replay missing or not dat ({st!r})')
        else:
            out[c] = LE * math.degrees(v)
    return out, ('dat' if not why else 'thieu'), why


def build(tbs, lan_lap=None, nguon=None, script=None):
    muc = {}
    for lv in LEVELS:
        tb = tbs.get(lv)
        muc[lv] = dict(can_quay=can_quay(tb, lv), man_trap=man_trap(tb),
                       chi_bao_cao=['g5'] if lv == 'O1-F08' else [],
                       khac={k: dict(gia_tri=None, trang_thai='thieu', ly_do='chua co nguon') for k in KHAC})
    ub, ust, uwhy = u_beta_cam(lan_lap)
    # F17 bo_cham.bang_man_trap: each level and K one branch; alpha needs the 3b source; K (0.5 m/s, F05 speeds)
    # takes the branch of the F05 3b (J v6 J4-13)
    src = {n['vai'][3:]: n for n in (nguon or []) if n.get('vai', '').startswith('3b ')}
    bang = {}
    for lv, key in [(lv, lv) for lv in LEVELS] + [('O1-F05', 'K')]:
        mt = muc[lv]['man_trap']
        bang[key] = dict(nhanh=mt['nhanh'])
        if mt['nhanh'] == 'alpha':
            n = src.get(lv)
            if n is None:
                raise KhoaError(f'alpha branch for {key} without its 3b source')
            bang[key]['nguon_3b'] = dict(duong_dan=n.get('duong_dan', n['file']), sha256=n['sha256'])
    return dict(version=VERSION, le=LE, muc=muc, bang_man_trap=bang, u_beta_cam_deg=ub, u_beta_cam_trang_thai=ust,
                u_beta_cam_ly_do=uwhy, trang_thai='chua_khoa',
                ly_do_chua_khoa='cac hang khac chua co nguon (thiet ke v7 muc 1, 15)',
                khoa_truoc=dict(nguon=nguon or [], script=script,
                                luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds')))


def script_state():
    import subprocess
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_khoa_v5.py'], capture_output=True,
                           text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), file='tools/c8b_khoa_v5.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_khoa_v5.py')


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tb', action='append', required=True, help='LEVEL=c8b_3b json')
    ap.add_argument('--lan-lap', default=None, help='NHACTAM lan_lap.json with beta_cam_replay (F15 b)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        cu.check_path(a.out)
        if os.path.exists(a.out):
            raise KhoaError(f'{a.out} exists')
        tbs, nguon = {}, []
        for s in a.tb:
            lv, sep, p = s.partition('=')
            if not sep or lv not in LEVELS or lv in tbs:
                raise KhoaError(f'--tb {s!r}: LEVEL=path, LEVEL in {LEVELS}, once each')
            with open(p, encoding='utf-8') as f:
                tbs[lv] = json.load(f)
            nguon.append(dict(vai=f'3b {lv}', file=os.path.basename(p), duong_dan=os.path.abspath(p),
                              sha256=cu.sha256(p)))
        ll = None
        if a.lan_lap:
            with open(a.lan_lap, encoding='utf-8') as f:
                ll = json.load(f)
            nguon.append(dict(vai='lan_lap', file=os.path.basename(a.lan_lap), sha256=cu.sha256(a.lan_lap)))
        out = build(tbs, ll, nguon, script_state())
        with open(a.out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(_clean(out), indent=1, ensure_ascii=False, allow_nan=False))
    except (KhoaError, cu.InputError, OSError, ValueError) as exc:
        print(f'c8b_khoa_v5: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    for lv in LEVELS:
        cq = out['muc'][lv]['can_quay']
        print(f'{lv}: can_quay {cq["trang_thai"]} p {cq["p_m"]} m psi {cq["psi_deg"]} deg; '
              f'man_trap {out["muc"][lv]["man_trap"]["nhanh"]}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
