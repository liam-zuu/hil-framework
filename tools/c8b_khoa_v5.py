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
  hang_theo_u   M5b (names CODE 19/11, GATE 8705918 GC6-5): [{u_lo_m_s, u_hi_m_s, P4_m, A4_deg, trang_thai}] of the
                level: one band [0, u_3b] with u_3b = v_max + wz_max a / sqrt 2 of the level's figure 8 (the speeds
                the 3b covered: c8b_3b dung_duoc needs both ways up to v_max, wz_max), P4_m, A4_deg = can_quay after the
                margin; trang_thai of can_quay (thieu: P4_m, A4_deg null). A frame with u outside every band has no
                P4 / A4 (design v7 section 10). Without a (--canh-ma-m) the band is empty.
  hang_dong     the constant closed terms the gate takes {a_m, hang_m, he_so_r, u_phi_rad, hang_rad} (names GATE
                F19): a_m = the tag side; the others null until their sources come (hang_dong_ly_do says which).
  nhanh_tick    the U_t branch of the C8 plan (bo_tick / giu_tick), copied from it (GATE: = u_t_branch).
  trang_thai    'khoa' only when every level of --muc has hang_dong all numbers, a band, the plan and its branch
                named; else 'chua_khoa' with ly_do_chua_khoa.
khoa_truoc: SHA-256 of every input and of this script (commit, dirty) and the time, written before any held-out part
is looked at (design 0.3); the C8 plan with vai 'ke_hoach_c8' (GATE GC6-1).

  python tools/c8b_khoa_v5.py --tb O1-F02=3b_F02.json --tb O1-F05=3b_F05.json --tb O1-F08=3b_F08.json \
      [--lan-lap lan_lap.json] [--canh-ma-m 0.0964] [--ke-hoach ke_hoach_c8.json] [--nhanh-tick bo_tick] \
      [--muc O1-F02,O1-F05] --out c8b_khoa_v5.json
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


BRANCHES = ('bo_tick', 'giu_tick')
HANG_DONG = ('a_m', 'hang_m', 'he_so_r', 'u_phi_rad', 'hang_rad')


class KhoaError(Exception):
    pass


def u_3b(level, a_m):
    """Upper end of the speed band the 3b of the level covers: v_max + wz_max a / sqrt 2 of its figure 8 (the
    schedule peaks c8b_3b checks the coverage against)."""
    from o1_feasibility import Schedule
    from o1_tasks import FIG8
    import o1_gate
    v, wz = o1_gate.fig8_cmd_max(Schedule(FIG8[level]))
    return float(v) + float(wz) * a_m / math.sqrt(2.0)


def hang_theo_u(cq, level, a_m):
    if a_m is None:
        return [], 'thieu canh ma a (--canh-ma-m): dai u khong xac dinh'
    gone = cq['trang_thai'] == 'thieu'
    return [dict(u_lo_m_s=0.0, u_hi_m_s=u_3b(level, a_m), P4_m=None if gone else cq['p_m'],
                 A4_deg=None if gone else cq['psi_deg'], trang_thai=cq['trang_thai'])], None


def trang_thai_khoa(out, muc_kha_thi):
    why = []
    if out.get('nhanh_tick') not in BRANCHES:
        why.append('nhanh_tick chua co (ke hoach C8)')
    if not any(n.get('vai') == 'ke_hoach_c8' for n in out['khoa_truoc']['nguon']):
        why.append('ke hoach C8 khong co trong khoa_truoc.nguon')
    for lv in muc_kha_thi:
        m = out['muc'][lv]
        miss = [k for k in HANG_DONG if _num(m['hang_dong'].get(k)) is None]
        if miss:
            why.append(f'{lv}: hang_dong {miss} chua co')
        if not m['hang_theo_u'] or all(b['trang_thai'] == 'thieu' for b in m['hang_theo_u']):
            why.append(f'{lv}: hang_theo_u thieu')
    return ('khoa', None) if not why else ('chua_khoa', '; '.join(why))


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


def build(tbs, lan_lap=None, nguon=None, script=None, a_m=None, nhanh_tick=None, muc_kha_thi=LEVELS):
    muc = {}
    for lv in LEVELS:
        tb = tbs.get(lv)
        cq = can_quay(tb, lv)
        bands, bwhy = hang_theo_u(cq, lv, a_m)
        muc[lv] = dict(can_quay=cq, man_trap=man_trap(tb),
                       chi_bao_cao=['g5'] if lv == 'O1-F08' else [],
                       hang_theo_u=bands, hang_theo_u_ly_do=bwhy,
                       hang_dong=dict(a_m=a_m, hang_m=None, he_so_r=None, u_phi_rad=None, hang_rad=None),
                       hang_dong_ly_do=dict(
                           a_m=None if a_m is not None else 'thieu --canh-ma-m',
                           hang_m='chua co nguon: P1/A3, P5, P6, P7, A6 vi tri, W-canh, U_p_tc_san, neo (thiet ke v7 '
                                  'muc 1, 10; KIEMCHEO, NHACTAM, HIEUCHUAN, CAM, M4 A6)',
                           he_so_r='chua co nguon: delta_q, U_thang (C8BX-03), tay don q (thiet ke v7 muc 10)',
                           u_phi_rad='chua co nguon: u_phi (HIEUCHUAN u_phi_hat, NHACTAM u_beta, C8B u_beta_cam)',
                           hang_rad='chua co nguon: A1, A2, A3, A5, A6, A7, neo (thiet ke v7 muc 1)'),
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
    out = dict(version=VERSION, le=LE, muc=muc, bang_man_trap=bang, u_beta_cam_deg=ub, u_beta_cam_trang_thai=ust,
               u_beta_cam_ly_do=uwhy, nhanh_tick=nhanh_tick, muc_kha_thi=list(muc_kha_thi),
               khoa_truoc=dict(nguon=nguon or [], script=script,
                               luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds')))
    st, why = trang_thai_khoa(out, muc_kha_thi)
    out.update(trang_thai=st, ly_do_chua_khoa=why)
    return out


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
    ap.add_argument('--canh-ma-m', type=float, default=None, help='tag side a (m) of the car tag (bo_cham.side_m)')
    ap.add_argument('--ke-hoach', default=None, help='C8 plan written before C8 (anchored in khoa_truoc, GATE GC6-1)')
    ap.add_argument('--nhanh-tick', choices=BRANCHES, default=None,
                    help='U_t branch when the plan file does not carry nhanh_tick')
    ap.add_argument('--muc', default=','.join(LEVELS), help='levels of muc_kha_thi (comma separated)')
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
        nt = a.nhanh_tick
        if a.ke_hoach:
            with open(a.ke_hoach, encoding='utf-8') as f:
                raw = f.read()
            plan_nt = None
            try:
                kh = json.loads(raw)
                plan_nt = kh.get('nhanh_tick') if isinstance(kh, dict) else None
            except ValueError:                                   # the CSV plan: '# nhanh_tick=...' lines
                for line in raw.splitlines():
                    if line.startswith('#') and line.lstrip('# ').startswith('nhanh_tick='):
                        plan_nt = line.split('=', 1)[1].strip()
            if plan_nt is not None and nt is not None and plan_nt != nt:
                raise KhoaError(f'--nhanh-tick {nt} but the C8 plan says {plan_nt}')
            nt = plan_nt or nt
            nguon.append(dict(vai='ke_hoach_c8', file=os.path.basename(a.ke_hoach),
                              duong_dan=os.path.abspath(a.ke_hoach), sha256=cu.sha256(a.ke_hoach)))
        if nt is not None and nt not in BRANCHES:
            raise KhoaError(f'nhanh_tick {nt!r}: one of {BRANCHES}')
        mk = [x.strip() for x in a.muc.split(',') if x.strip()]
        if not mk or any(x not in LEVELS for x in mk) or len(set(mk)) != len(mk):
            raise KhoaError(f'--muc {a.muc!r}: levels of {LEVELS}, once each')
        if a.canh_ma_m is not None and not (0.0 < a.canh_ma_m < 1.0):
            raise KhoaError(f'--canh-ma-m {a.canh_ma_m}: the tag side in metres')
        out = build(tbs, ll, nguon, script_state(), a.canh_ma_m, nt, mk)
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
    print(f'trang_thai {out["trang_thai"]}' + (f' ({out["ly_do_chua_khoa"]})' if out['ly_do_chua_khoa'] else ''))
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
