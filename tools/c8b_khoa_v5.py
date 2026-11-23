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
                F8 all null, cach A (CODE 23/11 chot T12 muc 4, correction 77991725): LE x (2 A3_xp[k] + 2 A6du +
                largest psi_tan_rad of the W at k, --tham-chieu); u_beta_cam_trang_thai_cho per place.
                u_phi_rad (hang_dong) = LE (u_phi_hat + u_beta) + u_beta_cam[goc] (design v7 6c).
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
  commit_hil    (CODE 23/11, Chot_T12_khoa_C8B_v1 muc 3; O1 v11.34 Q4 / OG3-4) the 40-hex hil commit that will
                score the runs (gio + grid scoring through Codex D1), written here and in khoa_truoc before any
                scoring; required with bo_tick (missing -> chua_khoa); GATE checks luoi.json scorer = commit_hil.
  eps_H         {mm, huong_deg} copied from the C8 threshold file nguong_khoa_c8_v1 (CODE writes it when Liam fixes
                eps_H at C8; CODE 20/11: not the CAM constant EPS_MM); SHA in khoa_truoc.nguon (vai 'nguong').
  mien_car_pose {side_m, phien_ban, h_m [lo, hi], kappa [lo, hi]} (GATE camera.c8b; CODE 20/11): h_m = the hull of
                the 4 corner heights of c8b_a6_v2 h_goc_m -/+ u_h_goc_m; kappa = [h_lo / (Z_c + dzc), h_hi / (Z_c - dzc)]
                derived, no new margin (Z_c = tag_h_m / kappa of the session, dzc = day_doi.dzc_m of the camera file,
                L125); side_m = session car_side_m; phien_ban = session car_pose_version (CAM adds it after session 2);
                any part missing -> null values, trang_thai thieu.
  lan_lap_E     {cho: {lan_lap_id, luot_id_E, trang_thai dat | chua_ket_luan, ly_do}} (A6C1-7, CODE 20/11, data contract
                ~05:00): the mounting declared at c8b_a6 do (lan_lap_cho of c8b_a6_v2) checked against F8 lan_lap_v1
                (--lan-lap): an F8 run with the luot_id of the E clip must carry that lan_lap_id (another value ->
                refused, exit 2); no such run -> dat only when every non-null lan_lap_id of the F8 runs of that place is
                one value equal to the declared one; else chua_ket_luan, ly_do lan_lap_E_khong_doi_chieu. F8 whose every run has lan_lap_id null (no
                mounting determined, CODE 23/11): no comparison, dat with nhan can_lech_tam_buoi1_chua_xac_minh_lan_lap
                and gia_dinh (one mounting assumed). F8 of another
                plate -> refused.
  trang_thai    'khoa' only when every level of --muc has hang_dong all numbers, a band, the plan and its branch
                named, eps_H, mien_car_pose and lan_lap_E dat there; else 'chua_khoa' with ly_do_chua_khoa.
khoa_truoc: SHA-256 of every input and of this script (commit, dirty) and the time, written before any held-out part
is looked at (design 0.3); the C8 plan with vai 'ke_hoach_c8' (GATE GC6-1).

  python tools/c8b_khoa_v5.py --tb O1-F02=3b_F02.json --tb O1-F05=3b_F05.json --tb O1-F08=3b_F08.json \
      [--lan-lap lan_lap.json] [--canh-ma-m 0.1445] [--ke-hoach ke_hoach_c8.json] [--nhanh-tick bo_tick] \
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
L136 = ('L136, Liam 20/11 phien CODE local_c29a2de2: "Ap L132 cho F08 (Recommended)": F08 dung khung B doc lai cua '
        'DOCNHOE (hil feat/c8-doc-nhoe c47afaf, 05_DATA/results/C8/2026-11-18_run01/C1_docnhoe/B_ghep/frames.csv) ke ca '
        'troi; gioi han: khong tach troi ga voi sai camera; 5 P2 + 4 P3 DEFERRED cua DOCNHOE (tham dinh '
        '03_CODE/docs/review/2026-11-19_Tham_dinh_CODE_DOCNHOE_code_vong3.md); G5 F08 van chi bao cao (L130)')


BRANCHES = ('bo_tick', 'giu_tick')
HANG_DONG = ('a_m', 'hang_m', 'he_so_r', 'u_phi_rad', 'hang_rad')
CHO_MUC = 'goc'             # CODE 23/11 muc 5: the 12 level cells of the plan are figure 8 at goc (gate 888, 1955)
U_THANG_THUOC = 1.0e-3      # CODE 23/11 muc 5: tape class III max error (0.6 + 0.4 L) mm at L = 1 m (OIML R 35-1)
DQ_LECH_TAM_M = 0.00057     # CODE 23/11 Chot_F8_rong_buoi2_v1 muc 1: plate shift bound of session 1 (P3), F8 null
U_BETA_LECH_TAM_DEG = 0.234  # CODE 23/11 chot T12 muc 4 cach A (Liam 23/11 at CODE, box 'phi buoi 2', verbatim:
                             # 'Cach A (Recommended)'): u_beta = heading bound of session 1 plate re-seating (A2)
CHO_THIEU = dict(           # sources not delivered yet (CODE 23/11 muc 5): the term stays null with this reason
    P6_rieng_i='P6 rieng dau (i) lan lap: theo lua chon cua Liam o muc 4 (cach A: 0; cach B: thieu)',
    P6_rieng_ii='P6 rieng dau (ii) dong: C8B chua tinh (thiet ke v7 muc 3, track cac luot C8)',
    neo='neo P_neo, A_neo: KIEMCHEO chua xuat (hop dong F15)',
    U_moc='U_moc_hieu_xy_m, U_moc_hieu_psi_rad: CAM chua xuat (hop dong muc G (6))',
    W='W (dao dong W, W-canh): thieu tham_chieu_v1 co bon truong w_* dat cua moi luot o cho cua muc (--tham-chieu)',
    u_theta='u_theta cua delta_q: u_a_k_deg cua F7 (HIEUCHUAN tinh) chua co',
    u_phi='u_phi_hat (F7 truc_xe_v1 goc khoa) chua co, hay F8 khong rong (cach A chi khi F8 rong)')


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


def hang_dong_tinh(src, side):
    """CODE 23/11 (03_CODE/docs/2026-11-23_Chot_T12_khoa_C8B_v1.md muc 5; design v7 sections 1, 3, 10): the closed
    terms BEFORE the margin, the largest over the camera segments of the C8 runs (one kiem_cheo.json each) at the
    level place goc. src: dict(kiem_cheo=[...], a6=..., xoay=..., u_h=..., cam=[...]) parsed files (None = absent).
      hang_rad = 2 A3 + 2 A6du + A5 (+ A_neo + U_moc_hieu_psi + W)
      hang_m   = P1 + L (2 A3 + 2 A6du + A5 + A_neo) + 2 thi_sai + P7 + 2 dq (+ P6 rieng (i), (ii) + P_neo + U_moc_hieu_xy + W)
      he_so_r  = U_thang + u_h_chung / (H - h_max) + h_max dzc / ((H - h_max)(H - dzc))
      u_phi_rad (not here: F7 and muc 4)
    A3 = A3_tuong_doi.A3_xp_deg[goc] per end, P1 = P1_tuong_doi.theo_vung.goc.max_m (flags ngoai_suy, mot_phuong,
    cap_hong false), P7 = P7.lech_max_m, A5 = A5.lech_max_deg (KIEMCHEO luot3_g4); A6du = c8b_a6.a6du_rad at the
    largest valid centre rho (rho_max - side / sqrt 2); L = |q_hat| + dq, dq = u_q_m (xoay_v1 goc) + DQ_LECH_TAM_M
    (+ L u_theta, F7); thi_sai = sum_i |r_i| (u_h_goc_i - u_h_chung) / (4 (H - h)) per end (the common height part
    cancels in sum r_i dh_i); U_thang = U_THANG_THUOC + thang_chung.lech_max; h_max = max(h_goc + u_h_goc).
    Returns (values {hang_m, he_so_r, hang_rad, u_phi_rad}, reasons {key: [..]}, parts {name: value}, L or None)."""
    why = {k: [] for k in ('hang_m', 'he_so_r', 'hang_rad', 'u_phi_rad')}
    part = {}
    kcs = [k for k in (src.get('kiem_cheo') or []) if isinstance(k, dict)]
    a6, xo, uh = src.get('a6'), src.get('xoay'), src.get('u_h')
    rong = f8_rong(src.get('lan_lap'))
    if rong:
        part['P6_rieng_i_m'] = 0.0                        # cach A: one mounting assumed (CODE 23/11 muc 4)
    f7 = src.get('truc_xe')
    f7_ok = (isinstance(f7, dict) and f7.get('schema') == 'truc_xe_v1' and f7.get('cho') == CHO_MUC
             and f7.get('trang_thai') == 'khoa')
    if f7_ok and _num(f7.get('u_a_k_deg')) is not None:
        part['u_theta_rad'] = math.radians(float(f7['u_a_k_deg']))
    if f7_ok and rong and _num(f7.get('u_a_noi_bo_deg')) is not None and isinstance(f7.get('dau_doan'), list) \
            and f7['dau_doan']:
        # CODE 23/11: u_a without u_psi_cuc; the camera part is u_beta_cam between the leg starts and W (build checks)
        part['u_phi_rad'] = math.radians(float(f7['u_a_noi_bo_deg']) + U_BETA_LECH_TAM_DEG)
        part['u_phi_cach'] = 'u_a_noi_bo + dau_doan'
    elif f7_ok and rong and _num(f7.get('u_phi_hat_chuan_k_deg')) is not None:
        part['u_phi_rad'] = math.radians(float(f7['u_phi_hat_chuan_k_deg']) + U_BETA_LECH_TAM_DEG)
        part['u_phi_cach'] = 'u_phi_hat_chuan_k + ref_12b'
    neo = src.get('neo')
    ng = ((neo or {}).get('theo_cho') or {}).get(CHO_MUC) if isinstance(neo, dict) else None
    if isinstance(neo, dict) and neo.get('schema') == 'c7_neo_luot_v1' and isinstance(ng, dict) \
            and ng.get('trang_thai', neo.get('trang_thai')) == 'dat' \
            and _num(ng.get('P_neo_m')) is not None and _num(ng.get('A_neo_rad')) is not None:
        part['P_neo_m'], part['A_neo_rad'] = float(ng['P_neo_m']), float(ng['A_neo_rad'])
    um = [u for u in (src.get('u_moc') or []) if isinstance(u, dict)]
    if um and all(_num(u.get('U_moc_hieu_xy_m')) is not None and _num(u.get('U_moc_hieu_psi_rad')) is not None
                  for u in um):
        part['U_moc_hieu_xy_m'] = max(float(u['U_moc_hieu_xy_m']) for u in um)
        part['U_moc_hieu_psi_rad'] = max(float(u['U_moc_hieu_psi_rad']) for u in um)
    if not kcs:
        for k in ('hang_m', 'hang_rad', 'he_so_r'):
            why[k].append('khong co kiem_cheo.json (KIEMCHEO luot3_g4, --kiem-cheo)')
    else:
        a3, p1, p7, a5, th = [], [], [], [], []
        for k in kcs:
            at, pt = k.get('A3_tuong_doi') or {}, (((k.get('P1_tuong_doi') or {}).get('theo_vung') or {})
                                                     .get(CHO_MUC) or {})
            bad = [f for f in ('ngoai_suy', 'mot_phuong', 'cap_hong')
                   if (at.get(f) or {}).get(CHO_MUC) is not False or pt.get(f) is not False]
            if bad:
                why['hang_m'].append(f'A3/P1 {CHO_MUC} co {bad} (thiet ke v7 muc 1: cap do thieu)')
                why['hang_rad'].append(f'A3 {CHO_MUC} co {bad}')
            a3.append(_num((at.get('A3_xp_deg') or {}).get(CHO_MUC)))
            p1.append(_num(pt.get('max_m')))
            p7.append(_num((k.get('P7') or {}).get('lech_max_m')))
            a5.append(_num((k.get('A5') or {}).get('lech_max_deg')))
            th.append(_num((k.get('thang_chung') or {}).get('lech_max')))
        for name, vals, keys in (('A3_rad', a3, ('hang_m', 'hang_rad')), ('P1_m', p1, ('hang_m',)),
                                 ('P7_m', p7, ('hang_m',)), ('A5_rad', a5, ('hang_m', 'hang_rad')),
                                 ('thang_lech', th, ('he_so_r',))):
            if any(v is None for v in vals):
                for kk in keys:
                    why[kk].append(f'kiem_cheo thieu {name}')
            else:
                part[name] = max(vals) if not name.endswith('_rad') else math.radians(max(vals))
    if not isinstance(a6, dict):
        for k in ('hang_m', 'hang_rad', 'he_so_r'):
            why[k].append('khong co A6 (--a6)')
    else:
        du, rho_in = _a6du_of(a6, side)
        if du is None:
            why['hang_m'].append('A6du khong tinh duoc (A6 khong dat hay ngoai mien)')
            why['hang_rad'].append('A6du khong tinh duoc')
        else:
            part['A6du_rad'], part['rho_A6du_m'] = du, rho_in
        hg, uhg, H = a6.get('h_goc_m'), a6.get('u_h_goc_m'), _num(a6.get('H_m'))
        uc = _num((uh or {}).get('u_h_chung_m')) if isinstance(uh, dict) and uh.get('trang_thai') == 'dat' else None
        if uc is None or not isinstance(hg, list) or not isinstance(uhg, list) or len(hg) != 4 or len(uhg) != 4 \
                or H is None:
            why['hang_m'].append('thi sai goc: thieu h_goc_m / u_h_goc_m (A6) hay u_h_chung dat (--u-h-chung)')
            why['he_so_r'].append('P6 chung, P5: thieu u_h_chung dat hay h_goc (A6)')
        else:
            hb = _num(a6.get('h_m'))
            r = float(side) / math.sqrt(2.0)
            part['thi_sai_m'] = sum(r * max(0.0, float(u) - uc) for u in uhg) / (4.0 * (H - hb))
            h_max = max(float(h) + float(u) for h, u in zip(hg, uhg))
            part['P6_chung_he_so'] = uc / (H - h_max)
            dz = [_num(((c or {}).get('day_doi') or {}).get('dzc_m')) for c in (src.get('cam') or [])]
            if not dz or any(x is None for x in dz):
                why['he_so_r'].append('P5: thieu day_doi.dzc_m (--camera)')
            else:
                dzc = max(dz)
                part['P5_he_so'] = h_max * dzc / ((H - h_max) * (H - dzc))
    if isinstance(xo, dict) and isinstance((xo.get('cho') or {}).get(CHO_MUC), dict):
        x = xo['cho'][CHO_MUC]
        q, uq = x.get('q_tag_m'), _num(x.get('u_q_m'))
        if x.get('trang_thai') != 'dat' or uq is None or not isinstance(q, list) or len(q) != 2:
            why['hang_m'].append(f'xoay_v1 {CHO_MUC} khong dat')
        else:
            part['q_hat_m'] = math.hypot(float(q[0]), float(q[1]))
            part['dq_khong_u_theta_m'] = uq + DQ_LECH_TAM_M
    else:
        why['hang_m'].append(f'khong co xoay_v1 {CHO_MUC} (--xoay)')
    w = src.get('W')                                          # direct numbers (tests); else from tham_chieu_v1
    if isinstance(w, dict) and _num(w.get('W_xy_m')) is not None and _num(w.get('W_psi_rad')) is not None:
        part['W_xy_m'], part['W_psi_rad'] = float(w['W_xy_m']), float(w['W_psi_rad'])
    # CAM T11 (hil 34fd9e4; names agreed C8B - CAM 23/11, design v7 6a, 6b), before the margin, the W of the runs at
    # the level place: w_dao_dong_* (two halves of W) + w_canh_* (largest deviation over W and the join + one frame
    # + v_d ho_noi); C8B adds the lever below: L (omega_d ho_noi_max_s + the W heading terms)
    tcw = [t for t in (src.get('tham_chieu') or {}).get(CHO_MUC, []) if isinstance(t, dict)]         if isinstance(src.get('tham_chieu'), dict) else []
    w_rows = []
    for t in tcw:
        c = t.get('w_c8b') or {}
        v = [_num(t.get(k)) for k in ('w_dao_dong_xy_m', 'w_canh_xy_m', 'w_dao_dong_psi_rad', 'w_canh_psi_rad')]             + [_num(c.get('omega_d')), _num(c.get('ho_noi_max_s'))]
        w_rows.append(None if any(x is None or x < 0 for x in v) or t.get('trang_thai') != 'dat' else v)
    if 'W_xy_m' not in part and w_rows and all(r is not None for r in w_rows):
        part['W_psi_rad'] = max(r[2] + r[3] for r in w_rows)
    p6 = src.get('P6_rieng_ii')                               # C8B, design v7 section 3 (waits W; CODE asked)
    if isinstance(p6, dict) and _num(p6.get('gia_tri_m')) is not None:
        part['P6_rieng_ii_m'] = float(p6['gia_tri_m'])
    have = dict(u_theta='u_theta_rad' in part, P6_rieng_i='P6_rieng_i_m' in part,
                P6_rieng_ii='P6_rieng_ii_m' in part, neo='P_neo_m' in part, U_moc='U_moc_hieu_xy_m' in part,
                W='W_psi_rad' in part)
    for k in ('u_theta', 'P6_rieng_i', 'P6_rieng_ii', 'neo', 'U_moc', 'W'):
        if not have[k]:
            why['hang_m'].append(CHO_THIEU[k])
    for k in ('neo', 'U_moc', 'W'):
        if not have[k]:
            why['hang_rad'].append(CHO_THIEU[k])
    if 'u_phi_rad' not in part:
        why['u_phi_rad'].append(CHO_THIEU['u_phi'])
    if 'thang_lech' in part:
        part['U_thang'] = U_THANG_THUOC + part['thang_lech']
    val = dict(hang_m=None, he_so_r=None, hang_rad=None, u_phi_rad=None)
    if not why['he_so_r'] and all(k in part for k in ('U_thang', 'P6_chung_he_so', 'P5_he_so')):
        val['he_so_r'] = part['U_thang'] + part['P6_chung_he_so'] + part['P5_he_so']
    if not why['hang_rad'] and all(k in part for k in ('A3_rad', 'A6du_rad', 'A5_rad')):
        val['hang_rad'] = 2 * part['A3_rad'] + 2 * part['A6du_rad'] + part['A5_rad']
    if not why['u_phi_rad']:
        val['u_phi_rad'] = part['u_phi_rad']   # u_phi_hat + u_beta only; build: 1.25 x this + u_beta_cam (has margin)
    lever = None
    if all(k in part for k in ('q_hat_m', 'dq_khong_u_theta_m', 'u_theta_rad')):
        dq0 = part['dq_khong_u_theta_m']
        part['dq_m'] = dq0 + (part['q_hat_m'] + dq0) * part['u_theta_rad']    # design v7 section 1 step (4)
        lever = part['L_m'] = part['q_hat_m'] + part['dq_m']
        if 'W_xy_m' not in part and w_rows and all(r is not None for r in w_rows):
            part['W_xy_khong_tay_don_m'] = max(r[0] + r[1] for r in w_rows)
            part['W_xy_m'] = max(r[0] + r[1] + lever * (r[4] * r[5] + r[2] + r[3]) for r in w_rows)
            part['W_n_luot'] = len(w_rows)
    need_m = ('P1_m', 'A3_rad', 'A6du_rad', 'A5_rad', 'thi_sai_m', 'P7_m', 'dq_m', 'P6_rieng_i_m', 'P6_rieng_ii_m',
              'P_neo_m', 'U_moc_hieu_xy_m', 'W_xy_m')
    if not why['hang_m'] and all(k in part for k in need_m):
        # CODE 23/11: A_neo is a heading error too, it turns q like A3, A5, A6 (design v7 section 1): (|q|+dq) A_neo
        val['hang_m'] = (part['P1_m'] + lever * (2 * part['A3_rad'] + 2 * part['A6du_rad'] + part['A5_rad']
                                                 + part['A_neo_rad'])
                         + 2 * part['thi_sai_m'] + part['P7_m'] + 2 * part['dq_m'] + part['P6_rieng_i_m']
                         + part['P6_rieng_ii_m'] + part['P_neo_m'] + part['U_moc_hieu_xy_m'] + part['W_xy_m'])
    if val['hang_rad'] is not None:
        if all(k in part for k in ('A_neo_rad', 'U_moc_hieu_psi_rad', 'W_psi_rad')):
            val['hang_rad'] += part['A_neo_rad'] + part['U_moc_hieu_psi_rad'] + part['W_psi_rad']
        else:
            val['hang_rad'] = None
    return val, {k: v for k, v in why.items() if v}, part, lever


def hang_theo_u(cq, level, a_m):
    if a_m is None:
        return [], 'thieu canh ma a (--canh-ma-m): dai u khong xac dinh'
    gone = cq['trang_thai'] == 'thieu'
    return [dict(u_lo_m_s=0.0, u_hi_m_s=u_3b(level, a_m), P4_m=None if gone else cq['p_m'],
                 A4_deg=None if gone else cq['psi_deg'], trang_thai=cq['trang_thai'])], None


def eps_h_of(ng):
    """eps_H {mm, huong_deg} of nguong_khoa_c8_v1, or (None, reason)."""
    if ng is None:
        return None, 'thieu nguong_khoa_c8_v1 (CODE ghi khi Liam chot eps_H o C8)'
    if not isinstance(ng, dict) or ng.get('format') != 'nguong_khoa_c8_v1':
        raise KhoaError('--nguong: not a nguong_khoa_c8_v1 file')
    e = ng.get('eps_H')
    if not isinstance(e, dict) or any(_num(e.get(k)) is None or _num(e.get(k)) < 0 for k in ('mm', 'huong_deg')):
        raise KhoaError('--nguong: eps_H {mm, huong_deg} not finite numbers >= 0')
    return dict(mm=float(e['mm']), huong_deg=float(e['huong_deg'])), None


def _mot_gia_tri(ds, key, what, why):
    """da session (A): the one value of key over the session / camera dicts ds; differing values -> why, None."""
    vals = [(d or {}).get(key) for d in ds]
    if len({json.dumps(v, sort_keys=True) for v in vals}) > 1:
        why.append(f'{what}: {key} differs between the sessions ({vals})')
        return None
    return vals[0] if vals else None


def mien_car_pose(a6, ses, cam):
    """{side_m, phien_ban, h_m, kappa, trang_thai, ly_do} (CODE 20/11). ses, cam: one dict or a list (da session (A),
    contract log 20/11 khuya): the sessions must agree on car_side_m, car_pose_version, tag_h_m, kappa (L125 set-up),
    dzc is the largest of the camera files (each must have it)."""
    why = []
    if isinstance(ses, list):
        ss = ses
        ses = None if not ss else dict(car_side_m=_mot_gia_tri(ss, 'car_side_m', 'session', why),
                                       car_pose_version=_mot_gia_tri(ss, 'car_pose_version', 'session', why),
                                       tag_h_m=_mot_gia_tri(ss, 'tag_h_m', 'session', why),
                                       kappa=_mot_gia_tri(ss, 'kappa', 'session', why))
    if isinstance(cam, list):
        dz = [_num(((c or {}).get('day_doi') or {}).get('dzc_m')) for c in cam]
        cam = None if not dz or any(x is None for x in dz) else dict(day_doi=dict(dzc_m=max(dz)))
    side = _num((ses or {}).get('car_side_m'))
    ver = (ses or {}).get('car_pose_version')
    if side is None:
        why.append('session: car_side_m missing')
    if not isinstance(ver, str) or not ver:
        why.append('session: car_pose_version missing (CAM adds CAR_POSE_VERSION after session 2)')
        ver = None
    h_m = None
    hg, uh = (a6 or {}).get('h_goc_m'), (a6 or {}).get('u_h_goc_m')
    if ((a6 or {}).get('trang_thai') or {}).get('h') != 'dat' or not isinstance(hg, list) or not isinstance(uh, list) \
            or len(hg) != 4 or len(uh) != 4 or any(_num(x) is None for x in hg + uh):
        why.append('c8b_a6_v2: h_goc_m / u_h_goc_m not dat')
    else:
        h_m = [min(h - u for h, u in zip(hg, uh)), max(h + u for h, u in zip(hg, uh))]
    kap = None
    th, k0 = _num((ses or {}).get('tag_h_m')), _num((ses or {}).get('kappa'))
    dzc = _num(((cam or {}).get('day_doi') or {}).get('dzc_m'))
    if th is None or k0 is None or not k0 > 0:
        why.append('session: tag_h_m / kappa missing (Z_c = tag_h_m / kappa)')
    if dzc is None or dzc < 0:
        why.append('camera file: day_doi.dzc_m missing')
    if h_m is not None and th is not None and k0 and dzc is not None and dzc >= 0:
        zc = th / k0
        if not zc - dzc > h_m[1]:
            why.append('Z_c - dzc not above the highest corner')
        else:
            kap = [h_m[0] / (zc + dzc), h_m[1] / (zc - dzc)]
    return dict(side_m=side, phien_ban=ver, h_m=h_m, kappa=kap, trang_thai='dat' if not why else 'thieu', ly_do=why)


LL_E_LY_DO = 'lan_lap_E_khong_doi_chieu'
LL_E_NULL = 'can_lech_tam_buoi1_chua_xac_minh_lan_lap'      # CODE 23/11 (Chot_F8_rong_buoi2_v1 muc 2): F8 all null
LL_E_NULL_CAN_CU = ('F8 khong xac minh duoc lan lap (moi lan_lap_id null; Liam o NHACTAM 23/11 dung duong beta); '
                    'khong doi chieu, gia dinh mot lan lap: Liam 22/11 o CODE_C8B "o ca 3 buoi deu co dinh, dan 1 lan '
                    'va do luon dau co thao ra dau?", Liam 23/11 o CODE "Chua thao, con nguyen", nhat ky khong co '
                    'buoc thao; lech tam lay theo can buoi 1 (0,57 mm; 0,234 do), CODE 23/11 '
                    '03_CODE/docs/2026-11-23_Chot_F8_rong_buoi2_v1.md')


def f8_rong(lan_lap):
    """True when the F8 is a lan_lap_v1-like dict with runs and every lan_lap_id null (identity checks of the
    plate are done by lan_lap_E; CODE 23/11 chot F8 rong and chot T12 muc 4)."""
    rows = lan_lap.get('luot') if isinstance(lan_lap, dict) else None
    return isinstance(rows, list) and bool(rows) and all(isinstance(x, dict) and x.get('lan_lap_id') is None
                                                          for x in rows)


def lan_lap_E(a6, lan_lap):
    """A6C1-7 (CODE 20/11): the E-clip mounting of every place of the A6 file against F8 lan_lap_v1.
    Returns ({cho: dict(lan_lap_id, luot_id_E, trang_thai, ly_do)}, trang_thai, reasons); KhoaError on a contradiction."""
    if not isinstance(a6, dict) or not isinstance(a6.get('lan_lap_cho'), dict) or not a6['lan_lap_cho']:
        return {}, 'chua_ket_luan', [f'{LL_E_LY_DO}: c8b_a6_v2 without lan_lap_cho (--a6)']
    rows = (lan_lap or {}).get('luot') if isinstance(lan_lap, dict) else None
    if not isinstance(rows, list):
        return ({c: dict(v, trang_thai='chua_ket_luan', ly_do=f'{LL_E_LY_DO}: no F8 lan_lap_v1 (--lan-lap)')
                 for c, v in a6['lan_lap_cho'].items()}, 'chua_ket_luan', [f'{LL_E_LY_DO}: no F8 (--lan-lap)'])
    if lan_lap.get('tam_id') != a6.get('tam_id'):
        raise KhoaError(f'F8 lan_lap of plate {lan_lap.get("tam_id")!r}, A6 of {a6.get("tam_id")!r}')
    out, why = {}, []
    # CODE 23/11 (Chot_F8_rong_buoi2_v1 muc 2; Liam at CODE_C8B, verbatim: "may gui cai cau hoi nay cho thang code cho
    # tao"; Liam 23/11 at CODE, box "Khoa T12", verbatim: "Dung mot lan lap (Recommended)"): an F8 whose every run
    # has lan_lap_id null determined no mounting: no comparison,
    # dat with the label LL_E_NULL (an assumption, written in the lock); an F8 with any id keeps the rule below.
    f8_null = f8_rong(lan_lap)
    for cho, v in a6['lan_lap_cho'].items():
        lid, le = (v or {}).get('lan_lap_id'), (v or {}).get('luot_id_E')
        r = dict(lan_lap_id=lid, luot_id_E=le, trang_thai='chua_ket_luan', ly_do=None)
        if not isinstance(lid, str) or not lid or not isinstance(le, str) or not le:
            r['ly_do'] = f'{LL_E_LY_DO}: lan_lap_id / luot_id_E of {cho} not declared in the A6 file'
        elif f8_null:
            r.update(trang_thai='dat', nhan=LL_E_NULL, gia_dinh=LL_E_NULL_CAN_CU)
        else:
            same = [x for x in rows if isinstance(x, dict) and x.get('luot_id') == le]
            if same:
                bad = [x.get('lan_lap_id') for x in same if x.get('lan_lap_id') not in (None, lid)]
                if bad:
                    raise KhoaError(f'{cho}: F8 run {le} has lan_lap_id {bad[0]!r}, the A6 file declares {lid!r}')
                if all(x.get('lan_lap_id') == lid for x in same):
                    r['trang_thai'] = 'dat'
                else:
                    r['ly_do'] = f'{LL_E_LY_DO}: F8 run {le} has no lan_lap_id (not determined)'
            else:
                vals = {x.get('lan_lap_id') for x in rows if isinstance(x, dict) and x.get('cho') == cho
                        and x.get('lan_lap_id') is not None}
                if vals == {lid}:
                    r['trang_thai'] = 'dat'
                else:
                    r['ly_do'] = (f'{LL_E_LY_DO}: no F8 run {le}; the F8 runs at {cho} carry lan_lap_id '
                                  f'{sorted(map(str, vals))}, not only the declared {lid!r}')
        if r['trang_thai'] != 'dat':
            why.append(f'{cho}: {r["ly_do"]}')
        out[cho] = r
    return out, ('dat' if not why else 'chua_ket_luan'), why


def trang_thai_khoa(out, muc_kha_thi):
    why = []
    if out.get('lan_lap_E_trang_thai') != 'dat':
        why.append('lan_lap_E: ' + ('; '.join(out.get('lan_lap_E_ly_do') or []) or 'thieu'))
    if out.get('eps_H') is None:
        why.append(f'eps_H: {out.get("eps_H_ly_do") or "thieu"}')
    mc = out.get('mien_car_pose')
    if not isinstance(mc, dict) or mc.get('trang_thai') != 'dat':
        why.append('mien_car_pose thieu' + (f' ({"; ".join(mc.get("ly_do") or [])})' if isinstance(mc, dict) else ''))
    if out.get('nhanh_tick') not in BRANCHES:
        why.append('nhanh_tick chua co (ke hoach C8)')
    elif out['nhanh_tick'] == 'bo_tick' and not _hex40(out.get('commit_hil')):
        why.append('bo_tick can commit_hil (40 hex) ghi truoc khi cham (O1 v11.34 Q4, CODE 23/11)')
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


def _hex40(x):
    return isinstance(x, str) and len(x) == 40 and all(c in '0123456789abcdef' for c in x)


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
    lim = [L131] + ([L132] if tb.get('b_dung_ke_ca_troi') else []) \
        + ([L136] if tb.get('b_dung_ke_ca_troi') and level == 'O1-F08' else []) + list(tb.get('gioi_han') or [])
    return dict(p_m=LE * p, psi_deg=LE * s, p_truoc_le_m=p, psi_truoc_le_deg=s, trang_thai='gioi_han', ly_do=[],
                gioi_han=lim, b_dung_ke_ca_troi=bool(tb.get('b_dung_ke_ca_troi')))


def man_trap(tb):
    sh = ((tb or {}).get('B') or {}).get('shutter') or {}
    if sh.get('use_alpha') and _num(sh.get('alpha_r_le_s_px')) is not None and _num(sh.get('alpha_c_le_s_px')) is not None:
        return dict(nhanh='alpha', alpha_r_s_px=abs(float(sh['alpha_r_le_s_px'])),
                    alpha_c_s_px=abs(float(sh['alpha_c_le_s_px'])), le=LE)
    return dict(nhanh='mot_khung', mot_khung_s=_num(sh.get('mot_khung_s')), le=None,
                ly_do=sh.get('why') or 'no 3b shutter fit')


def _a6du_of(a6, side):
    """A6du (rad) at the largest valid centre rho (rho_max - side / sqrt 2), with that rho; (None, rho or None)."""
    if not isinstance(a6, dict) or side is None:
        return None, None
    import c8b_a6 as a6m
    rho_in = float((a6.get('mien') or {}).get('rho_max_m', a6m.RHO_MAX_M)) - float(side) / math.sqrt(2.0)
    try:
        return a6m.a6du_rad(a6, rho_in), rho_in
    except (KeyError, TypeError, ValueError):
        return None, rho_in


def _a3_hai_dau_rad(kcs, cho):
    """2 A3_xp (two ends, A3_tuong_doi.max_deg = 2 A3_xp) at cho, the largest over the segments; (None, reason)."""
    if not kcs:
        return None, 'khong co kiem_cheo.json (--kiem-cheo)'
    vals = []
    for k in kcs:
        at = k.get('A3_tuong_doi') or {}
        bad = [f for f in ('ngoai_suy', 'mot_phuong', 'cap_hong') if (at.get(f) or {}).get(cho) is not False]
        v = _num((at.get('A3_xp_deg') or {}).get(cho))
        if bad:
            return None, f'A3 {cho} co {bad} (thiet ke v7 muc 1: cap do thieu)'
        if v is None:
            return None, f'kiem_cheo thieu A3_xp_deg {cho}'
        vals.append(v)
    return 2.0 * math.radians(max(vals)), None


def rho_khung(path, ids, sha, plumb):
    """Largest tag centre distance (tag_x, tag_y; valid 1) to the plumb point over the frames idx in ids of one
    frames.csv whose SHA-256 must be sha; every id must be a valid frame (KhoaError otherwise)."""
    import csv
    if cu.sha256(path) != sha:
        raise KhoaError(f'{path}: SHA-256 is not {sha} (frames of the window)')
    want, got = {int(i) for i in ids}, {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                i = int(row['idx'])
            except (KeyError, TypeError, ValueError):
                continue
            if i in want and row.get('valid') == '1':
                x, y = _num(row.get('tag_x')), _num(row.get('tag_y'))
                if x is not None and y is not None:
                    got[i] = (x, y)
    if not want or set(got) != want:
        raise KhoaError(f'{path}: {len(want - set(got))} window frames without a valid tag centre')
    C = [float(v) for v in plumb]
    return max(math.hypot(x - C[0], y - C[1]) for x, y in got.values())


def cua_so_khung(path, ids, sha, plumb):
    """Footprint of one stationary window: {rho_m (largest tag centre distance to the plumb), T_m (mean tag centre),
    S_rad (circular mean of psi), tan_rad (max - min of wrap(psi - S)), doan (camera segment 'doanN' of the path or
    None), n}; frames.csv SHA must be sha and every id a valid frame with finite tag_x, tag_y, psi."""
    import csv
    import re
    if cu.sha256(path) != sha:
        raise KhoaError(f'{path}: SHA-256 is not {sha} (frames of the window)')
    want, got = {int(i) for i in ids}, {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                i = int(row['idx'])
            except (KeyError, TypeError, ValueError):
                continue
            if i in want and row.get('valid') == '1':
                x, y, p = _num(row.get('tag_x')), _num(row.get('tag_y')), _num(row.get('psi'))
                if x is not None and y is not None and p is not None:
                    got[i] = (x, y, p)
    if not want or set(got) != want:
        raise KhoaError(f'{path}: {len(want - set(got))} window frames without a valid tag centre / psi')
    C = [float(v) for v in plumb]
    v = list(got.values())
    S = math.atan2(sum(math.sin(p) for _, _, p in v), sum(math.cos(p) for _, _, p in v))
    d = [math.atan2(math.sin(p - S), math.cos(p - S)) for _, _, p in v]
    m = re.search(r'[\\/](doan\d+)[\\/]', os.path.abspath(path))
    return dict(rho_m=max(math.hypot(x - C[0], y - C[1]) for x, y, _ in v),
                T_m=[sum(x for x, _, _ in v) / len(v), sum(y for _, y, _ in v) / len(v)], S_rad=S,
                tan_rad=max(d) - min(d), doan=m.group(1) if m else None, n=len(v))


def _a6_du_cua_so(a6, T, S, G, V, DH, cam):
    """A6 residual (rad) of a stationary tag at T, heading S, for each (tilt G (N, 2), warp V (N), common height
    offset DH (N)) seen by the camera centre cam = (x, y, z): the real car_pose path (corners at their heights, ray to
    the floor, lift with kappa = hbar / H about the plumb, Kabsch) minus the A6 correction with the FILE g_b at the
    measured pose, minus the truth S (c8b_a6.ebc_errors, but corrected with g_b instead of the true g)."""
    import numpy as np
    import c8b_a6 as a6m
    C, H, hb, side = (np.asarray(a6['plumb_xy_m'], float), float(a6['H_m']), float(a6['h_m']), float(a6['side_m']))
    r = a6m.body(side)
    a = r - r.mean(0)
    X = np.asarray(T, float)[None, :] + r @ a6m._rot(S).T
    h = hb + np.asarray(DH, float)[:, None] + np.asarray(G, float) @ r.T \
        + np.asarray(V, float)[:, None] * a6m.S_PATTERN[None, :]
    Cc, Hc = np.asarray(cam[:2], float), float(cam[2])
    p = Cc[None, None, :] + (X[None, :, :] - Cc[None, None, :]) * (Hc / (Hc - h))[:, :, None]
    Pm = p + (hb / H) * (C[None, None, :] - p)
    tm = Pm.mean(1)
    hm = np.einsum('ij,nik->njk', a, Pm - tm[:, None, :])
    pm = np.arctan2(hm[:, 0, 1] - hm[:, 1, 0], hm[:, 0, 0] + hm[:, 1, 1])
    gb = np.tile(np.asarray(a6['g_b'], float), (len(pm), 1))
    return a6m._wrap(pm - S - a6m.dpsi_model_vec(gb, tm, pm, C, H, hb))


def a6_hieu_replay(a6, ref, W, eps, cams_ref, cams_W):
    """Largest |A6 residual(W) - A6 residual(ref)| (rad) under ONE common parameter noise of the plate on both windows
    (CODE 23/11 to C8B; design v7 6c: u_beta_cam = the largest change of beta under the noise, a replay of the
    difference): g = g_b + eps (j / nr)(cos, sin) (the A6 disc), warp v in [-v_max, v_max], common height offset dh
    in [h_lo - h, h_hi - h] (corner hull of h_goc_m -/+ u_h_goc_m), every camera pair cams_ref x cams_W, every W of
    the list; grid refined k = EBC_K0 .. EBC_KMAX of c8b_a6 until the maximum changes by < EBC_TOL (an index, no
    claim between the nodes, as e_bc). Returns (value or None when not converged, [[k, maximum], ...])."""
    import numpy as np
    import c8b_a6 as a6m
    hb = float(a6['h_m'])
    hg, uh = np.asarray(a6['h_goc_m'], float), np.asarray(a6['u_h_goc_m'], float)
    dh_lo, dh_hi = float((hg - uh).min()) - hb, float((hg + uh).max()) - hb
    vmax = float((a6.get('mien') or {}).get('v_max_m') or a6m.V_MAX_M)
    gb = np.asarray(a6['g_b'], float)
    lv, k = [], a6m.EBC_K0
    while k <= a6m.EBC_KMAX:
        nr, nd = k // 2, k
        gd = np.linspace(0.0, 2 * math.pi, nd, endpoint=False)
        gs = np.array([gb] + [gb + eps * (j / nr) * np.array([math.cos(t), math.sin(t)])
                              for j in range(1, nr + 1) for t in gd])
        vs, ds = np.linspace(-vmax, vmax, k // 2 + 1), np.linspace(dh_lo, dh_hi, k // 2 + 1)
        iG, iV, iD = (x.ravel() for x in np.meshgrid(np.arange(len(gs)), np.arange(len(vs)), np.arange(len(ds)),
                                                      indexing='ij'))
        G, V, D = gs[iG], vs[iV], ds[iD]
        worst = 0.0
        for cr in cams_ref:
            e_ref = _a6_du_cua_so(a6, ref['T_m'], ref['S_rad'], G, V, D, cr)
            for w in W:
                for cw in cams_W:
                    e_w = _a6_du_cua_so(a6, w['T_m'], w['S_rad'], G, V, D, cw)
                    worst = max(worst, float(np.max(np.abs(a6m._wrap(e_w - e_ref)))))
        lv.append([k, worst])
        if len(lv) >= 2:
            prev, cur = lv[-2][1], lv[-1][1]
            if cur == prev or (prev > 0 and abs(cur - prev) < a6m.EBC_TOL * prev):
                return cur, lv
        k *= 2
    return None, lv


def _cams(a6, session_sha=None):
    """Camera centres of the A6 file (session list); only that session's when it is listed."""
    ss = [s for s in (a6.get('session') or []) if isinstance(s, dict) and isinstance(s.get('cam_tam_m'), list)]
    one = [s['cam_tam_m'] for s in ss if session_sha and s.get('session_sha256') == session_sha]
    return one or [s['cam_tam_m'] for s in ss]


def u_beta_cam_cho(a6, kcs, kc_doan, tam_id, dc, cho):
    """cach A at one place from the two windows (dau_chan[cho] = {ref: window, W: [window + psi_tan_rad]}), CODE
    23/11 (design v7 1: 'A3: a signed pair only from the same output, else the two-end bound'; 6c: replay of the
    difference). Per run: A3 = c8b_a6.u_psi_hai_dau_rad joint per-case term at the two footprints when both windows
    are in the same camera segment with its kiem_cheo (else, or a flagged / failed pair: 2 A3_xp[cho], the bound of
    the two ends); A6 = a6_hieu_replay of the pair with eps_g without the map part when the joint term carried it (else
    eps_g); + psi_tan of W (tham_chieu_v1) + tan of the ref window. Returns (value before the margin or None, reasons,
    per-run parts)."""
    # CODE 23/11 (chot T12 muc 5 'u_phi khong dem hai lan cua so chuan', Liam at CODE: "may lam di"): with the leg-
    # start windows of the 12b long legs (F7 dau_doan, where a_k takes psi') the camera error is taken straight
    # between each leg start and W (phi = psi'_tc - a, the reference window cancels); without them the ref_12b window
    refs, Ws = (dc.get('dau_doan') or [dc['ref']]), dc['W']
    loai = 'dau_doan' if dc.get('dau_doan') else 'ref_12b'
    a2, w2 = _a3_hai_dau_rad(kcs, cho)
    cams_all = _cams(a6)
    if not cams_all:
        return None, ['A6 khong co danh sach camera (session.cam_tam_m)'], []
    vals, why, parts = [], [], []
    for ref, w in [(r_, w_) for r_ in refs for w_ in Ws]:
        import c8b_a6 as a6m
        p = dict(doan_ref=ref.get('doan'), doan_W=w.get('doan'), loai_ref=loai, ref=ref.get('luot_id'),
                 ref_i=ref.get('doan_i'), W=w.get('frames'))
        a3p, eps, cach = None, a6.get('eps_g'), 'bao_hai_dau'
        kc = (kc_doan or {}).get(w.get('doan')) if ref.get('doan') and ref.get('doan') == w.get('doan') else None
        if kc is not None:
            r = a6m.u_psi_hai_dau_rad(a6, kc, ref['T_m'], w['T_m'], tam_id)
            if r.get('trang_thai') == 'dat' and r.get('ghep') == 'replay_chung_theo_ca':
                a3p, eps, cach = float(r['phan']['ca_rad']), a6.get('eps_g_khong_meo'), 'cap_replay_chung_theo_ca'
            elif r.get('trang_thai') == 'dat':
                a3p, cach = float(sum(r['phan']['A3_rad'])), 'cap_cong_rieng'
            else:
                p['cap_ly_do'] = r.get('ly_do')
        if a3p is None and kc_doan:
            # two segments (or no usable pair): the two-end bound by position, each end with the kiem_cheo of its own
            # segment (c8b_a6._a3_rad); an end in a segment without kiem_cheo takes the largest of the others (stand-in)
            ends = []
            for nm, e in (('ref', ref), ('W', w)):
                own = kc_doan.get(e.get('doan'))
                tries = [(own, 'rieng')] if own is not None else [(x, 'thay_the') for x in kc_doan.values()]
                got = [(a6m._a3_rad(x, e['T_m'], a6['plumb_xy_m']), how) for x, how in tries]
                ok = [(r3[0], how) for r3, how in got if r3[0] is not None]
                if not ok:
                    ends.append((None, f'{nm} {e.get("doan")}: ' + '; '.join(str(r3[1]) for r3, _ in got)))
                else:
                    ends.append((max(v for v, _ in ok), ok[0][1]))
            if all(v is not None for v, _ in ends):
                a3p, cach = ends[0][0] + ends[1][0], f'bao_hai_dau_theo_vi_tri ({ends[0][1]}, {ends[1][1]})'
                p['A3_dau_rad'] = [ends[0][0], ends[1][0]]
            else:
                why.append('; '.join(m for v, m in ends if v is None))
                continue
        if a3p is None:
            if a2 is None:
                why.append(f'{w.get("doan")}: {w2}')
                continue
            a3p = a2
        rep, lv = a6_hieu_replay(a6, ref, [w], float(eps), _cams(a6, ref.get('session_sha256')), cams_all)
        if rep is None:
            why.append(f'replay A6 khong hoi tu ({lv})')
            continue
        tan = _num(w.get('psi_tan_rad'))
        if tan is None:
            why.append('tham_chieu_v1 thieu psi_tan_rad')
            continue
        tot = a3p + rep + tan + ref['tan_rad']
        p.update(A3_rad=a3p, A3_cach=cach, A6_replay_rad=rep, A6_lan_chia=lv, eps_g=eps, psi_tan_W_rad=tan,
                 tan_ref_rad=ref['tan_rad'], tong_rad=tot)
        parts.append(p)
        vals.append(tot)
    if why or not vals:
        return None, why or ['khong co W'], parts
    return max(vals), [], parts


def u_beta_cam(lan_lap, kiem_cheo=None, a6=None, side=None, tham_chieu=None, dau_chan=None, kc_doan=None):
    """u_beta_cam_deg {goc, K1, K2} after the margin; returns (values, trang_thai, reasons, gia_dinh, per-place
    trang_thai). F8 with ids: from NHACTAM beta_cam_replay (F15 b). F8 all null, cach A (CODE 23/11 chot T12 muc 4,
    correction 77991725: the body turns with the tag, beta = wrap(psi'_tc - psi'_chuan,k) from the tag yaw, so the
    camera error of that yaw difference between the ref_12b window and W is NOT 0):
      u_beta_cam_deg[k] = LE x (2 A3_xp[k] (kiem_cheo luot1, two ends) + A6 dư of the two footprints + the largest
                        psi_tan_rad of the tham_chieu_v1 W of the runs at k)
    A6 dư of the two footprints = A6du(rho of the ref_12b window of the provider run at k) + A6du(largest rho of the W
    windows of the runs at k), rho = the largest tag centre distance to the plumb over the window frames (dau_chan
    {k: {rho_12b_m, rho_W_m}}); without dau_chan[k]: 2 A6du at the largest valid rho (an upper bound, labelled).
    A missing part (no W of k yet, an A3 flag) -> null at k, thieu."""
    if f8_rong(lan_lap):
        kcs = [k for k in (kiem_cheo or []) if isinstance(k, dict)]
        du, _ = _a6du_of(a6, side)
        out, st, why, cach = {}, {}, [], {}
        for c in CHO:
            a3, w3 = _a3_hai_dau_rad(kcs, c)
            tcs = [t for t in ((tham_chieu or {}).get(c) or []) if isinstance(t, dict)]
            tan = [_num(t.get('psi_tan_rad')) for t in tcs]
            dc = (dau_chan or {}).get(c)
            if isinstance(dc, dict) and (isinstance(dc.get('ref'), dict) or dc.get('dau_doan')) and dc.get('W') \
                    and isinstance(a6, dict):
                v, wv, parts = u_beta_cam_cho(a6, kcs, kc_doan, (lan_lap or {}).get('tam_id'), dc, c)
                cach[c] = dict(cach='replay_hieu_hai_cua_so', loai_ref='dau_doan' if dc.get('dau_doan') else 'ref_12b',
                               theo_luot=parts)
                if v is None:
                    out[c], st[c] = None, 'thieu'
                    why.append(f'{c}: ' + '; '.join(map(str, wv)))
                else:
                    out[c], st[c] = LE * math.degrees(v), 'dat'
                continue
            if isinstance(dc, dict) and isinstance(a6, dict):
                import c8b_a6 as a6m
                d12, dW = a6m.a6du_rad(a6, dc.get('rho_12b_m')), a6m.a6du_rad(a6, dc.get('rho_W_m'))
                a6p = None if d12 is None or dW is None else d12 + dW
                cach[c] = 'dau_chan'
            else:
                a6p = None if du is None else 2.0 * du
                cach[c] = 'rho_max (khong co dau chan; can tren)'
            miss = ([w3] if w3 else []) + ([] if a6p is not None else
                                           ['A6du khong tinh duoc (--a6, --canh-ma-m, dau chan ngoai mien)']) \
                + (['W (T11 tham_chieu_v1) cua cho chua co (--tham-chieu)'] if not tcs else []) \
                + (['tham_chieu_v1 thieu psi_tan_rad'] if tcs and any(x is None for x in tan) else [])
            if miss:
                out[c], st[c] = None, 'thieu'
                why.append(f'{c}: ' + '; '.join(miss))
            else:
                out[c], st[c] = LE * math.degrees(a3 + a6p + max(tan)), 'dat'
        gd = dict(nhan=LL_E_NULL, gia_dinh=LL_E_NULL_CAN_CU + '; cach A: beta = wrap(psi\'_tc - psi\'_chuan,k) tu yaw '
                  'ma, u_beta = 0,234 do, u_beta_cam = 1,25 (A3 cap hai dau + replay A6 hieu hai cua so + tan yaw '
                  'hai cua so) (CODE 23/11, 77991725 va tin sau 08eb411)', a6du_cach=cach)
        return out, ('dat' if not why else 'thieu'), why, gd, st
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
    return out, ('dat' if not why else 'thieu'), why, None, {c: ('dat' if out[c] is not None else 'thieu')
                                                             for c in CHO}


def build(tbs, lan_lap=None, nguon=None, script=None, a_m=None, nhanh_tick=None, muc_kha_thi=LEVELS, nguong=None,
          a6=None, ses=None, cam=None, commit_hil=None, kiem_cheo=None, xoay=None, u_h=None, truc_xe=None,
          neo=None, u_moc=None, tham_chieu=None, dau_chan=None, kc_doan=None):
    muc = {}
    side = a_m
    hd, hd_why, hd_part, lever = (hang_dong_tinh(dict(kiem_cheo=kiem_cheo, a6=a6, xoay=xoay, u_h=u_h,
                                                       lan_lap=lan_lap, truc_xe=truc_xe, neo=neo, u_moc=u_moc,
                                                       tham_chieu=tham_chieu,
                                                       cam=cam if isinstance(cam, list) else [cam] if cam else []),
                                                  side)
                                  if side is not None else (dict(hang_m=None, he_so_r=None, hang_rad=None,
                                                                 u_phi_rad=None), {}, {}, None))
    ub, ust, uwhy, ugd, ust_cho = u_beta_cam(lan_lap, kiem_cheo, a6, side, tham_chieu, dau_chan, kc_doan)
    # design v7 6c: u_phi = 1.25 (u_phi_hat + u_beta) + u_beta_cam (u_beta_cam already has the margin), at goc
    hd = dict(hd)
    u_phi_le = None
    if hd['u_phi_rad'] is not None:
        cg = ((ugd or {}).get('a6du_cach') or {}).get(CHO_MUC)
        if ub.get(CHO_MUC) is None:
            hd_why = dict(hd_why, u_phi_rad=[f'u_beta_cam {CHO_MUC} thieu ({"; ".join(uwhy) or ust})'])
            hd['u_phi_rad'] = None
        elif hd_part.get('u_phi_cach') == 'u_a_noi_bo + dau_doan' \
                and not (isinstance(cg, dict) and cg.get('loai_ref') == 'dau_doan'):
            # u_a_noi_bo leaves the camera error of the leg starts out: it must come from u_beta_cam on dau_doan
            hd_why = dict(hd_why, u_phi_rad=[f'u_a_noi_bo can u_beta_cam {CHO_MUC} tinh tren dau_doan (F7); thieu cua so'])
            hd['u_phi_rad'] = None
        else:
            u_phi_le = LE * hd['u_phi_rad'] + math.radians(ub[CHO_MUC])
            # CODE 23/11 (after 82d5881): the scorer (c7_score) takes 1.25 (u_phi_hat_chuan_k_deg of F7 + u_beta) +
            # u_beta_cam_deg[k] of this lock; the lock must carry exactly that u_phi (under cach A with dau_doan
            # HIEUCHUAN writes u_phi_hat_chuan_k_deg = u_a_noi_bo_deg); a difference -> refused
            uph = _num(truc_xe.get('u_phi_hat_chuan_k_deg')) if isinstance(truc_xe, dict) else None
            cham = None if uph is None else math.radians(LE * (uph + U_BETA_LECH_TAM_DEG) + ub[CHO_MUC])
            if cham is None or abs(cham - u_phi_le) > 1e-12 * max(1.0, abs(u_phi_le)):
                raise KhoaError(f'u_phi_rad cua khoa {u_phi_le!r} khac cong thuc bo cham tu F7 {CHO_MUC} '
                                f'(u_phi_hat_chuan_k_deg {uph!r}) + u_beta_cam_deg cua khoa: {cham!r}; F7 cach A phai co '
                                'u_phi_hat_chuan_k_deg = u_a_noi_bo_deg')
    for lv in LEVELS:
        tb = tbs.get(lv)
        cq = can_quay(tb, lv)
        bands, bwhy = hang_theo_u(cq, lv, a_m)
        if lever is not None:                                 # (|q_hat| + dq) A4(u) into P4 (v7 10, GATE RT4-3)
            for b in bands:
                if b.get('P4_m') is not None and b.get('A4_deg') is not None:
                    b['P4_tay_don_m'] = lever * math.radians(b['A4_deg'])
                    b['P4_m'] = b['P4_m'] + b['P4_tay_don_m']
        muc[lv] = dict(can_quay=cq, man_trap=man_trap(tb),
                       chi_bao_cao=['g5'] if lv == 'O1-F08' else [],
                       hang_theo_u=bands, hang_theo_u_ly_do=bwhy,
                       hang_dong=dict(a_m=a_m, **{k: (None if hd[k] is None else LE * hd[k])
                                                  for k in ('hang_m', 'he_so_r', 'hang_rad')}, u_phi_rad=u_phi_le),
                       hang_dong_truoc_le=dict(hd), hang_dong_thanh_phan=dict(hd_part),
                       P4_tay_don=dict(L_m=lever, ly_do=None if lever is not None else
                                       '(|q_hat| + dq) A4(u) them vao P4_m khi co dq day du (u_theta F7); '
                                       'CODE 23/11 muc 5, GATE RT4-3'),
                       hang_dong_ly_do=dict(
                           a_m=None if a_m is not None else 'thieu --canh-ma-m',
                           **{k: ('; '.join(hd_why[k]) if k in hd_why else None)
                              for k in ('hang_m', 'he_so_r', 'u_phi_rad', 'hang_rad')}),
                       khac={k: dict(gia_tri=None, trang_thai='thieu', ly_do='chua co nguon') for k in KHAC})
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
    eh, ehw = eps_h_of(nguong)
    lle, lle_st, lle_why = lan_lap_E(a6, lan_lap)
    out = dict(lan_lap_E=lle, lan_lap_E_trang_thai=lle_st, lan_lap_E_ly_do=lle_why, version=VERSION, le=LE, muc=muc, bang_man_trap=bang, u_beta_cam_deg=ub, u_beta_cam_trang_thai=ust, u_beta_cam_trang_thai_cho=ust_cho, u_beta_cam_dau_chan=dau_chan or {}, u_beta_cam_gia_dinh=ugd,
               u_beta_cam_ly_do=uwhy, nhanh_tick=nhanh_tick, commit_hil=commit_hil, muc_kha_thi=list(muc_kha_thi),
               eps_H=eh, eps_H_ly_do=ehw, mien_car_pose=mien_car_pose(a6, ses, cam),
               khoa_truoc=dict(nguon=nguon or [], script=script, commit_hil=commit_hil,
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


def dau_chan_of(a, extra, nguon):
    """{cho: {rho_12b_m, rho_W_m, ...}} of the two footprints of beta (cach A): the ref_12b window of --khung-12b in
    the F2 --cua-so (same cho, frames_sha256) and the W windows (ids, frames of nguon) of the --tham-chieu at cho."""
    a6 = extra.get('a6')
    if not isinstance(a6, dict) or not isinstance(a6.get('plumb_xy_m'), list) or (a.khung_12b and not a.cua_so):
        raise KhoaError('--khung-12b / F7 dau_doan need --a6 (plumb_xy_m); --khung-12b needs --cua-so (F2)')
    rows = []
    if a.cua_so:
        with open(a.cua_so, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        nguon.append(dict(vai='cua_so F2', file=os.path.basename(a.cua_so), duong_dan=os.path.abspath(a.cua_so),
                          sha256=cu.sha256(a.cua_so)))

    def w_of(cho):
        Ws = []
        for t in (extra.get('tham_chieu') or {}).get(cho) or []:
            fr = [n for n in t.get('nguon') or [] if n.get('vai_tro') == 'frames']
            if len(fr) != 1 or fr[0].get('sha256') != t.get('frames_sha256') or not isinstance(t.get('ids'), list):
                raise KhoaError(f'tham_chieu_v1 at {cho}: frames source / ids not usable for the W footprint')
            w = cua_so_khung(fr[0]['duong_dan'], t['ids'], t['frames_sha256'], a6['plumb_xy_m'])
            w.update(psi_tan_rad=t.get('psi_tan_rad'), frames=os.path.basename(os.path.dirname(fr[0]['duong_dan'])))
            Ws.append(w)
        return Ws

    out = {}
    # F7 dau_doan (fields proposed to HIEUCHUAN 23/11): one window per long 12b leg start used by a_k:
    # {thu_id, doan_i, frames_duong_dan, frames_sha256, ids}
    for cho, f7 in (extra.get('truc_xe_cho') or {}).items():
        dd = f7.get('dau_doan') if isinstance(f7, dict) else None
        if not isinstance(dd, list) or not dd:
            continue
        wins = []
        for e in dd:
            if not isinstance(e, dict) or not isinstance(e.get('ids'), list) \
                    or not isinstance(e.get('frames_duong_dan'), str) or not isinstance(e.get('frames_sha256'), str):
                raise KhoaError(f'F7 {cho} dau_doan: each entry {{thu_id, doan_i, frames_duong_dan, frames_sha256, ids}}')
            w = cua_so_khung(e['frames_duong_dan'], e['ids'], e['frames_sha256'], a6['plumb_xy_m'])
            w.update(luot_id=e.get('thu_id'), doan_i=e.get('doan_i'), session_sha256=e.get('session_sha256'))
            wins.append(w)
        Ws = w_of(cho)
        if Ws:
            out[cho] = dict(dau_doan=wins, W=Ws, n_W=len(Ws), n_dau_doan=len(wins),
                            quy_tac='cua so dau doan dai 12b (F7 dau_doan) so voi W (tham_chieu_v1)')
    for s in a.khung_12b or []:
        cho, sep, pth = s.partition('=')
        if not sep or cho not in CHO or 'ref' in (out.get(cho) or {}):
            raise KhoaError(f'--khung-12b {s!r}: CHO=frames.csv, CHO in {CHO}, once each')
        sha = cu.sha256(pth)
        win = [r for r in rows if r.get('loai') == 'ref_12b' and r.get('cho') == cho and r.get('frames_sha256') == sha]
        if not win:
            raise KhoaError(f'--khung-12b {s!r}: no F2 ref_12b window of {cho} on these frames')
        if len(win) != 1:
            raise KhoaError(f'--khung-12b {s!r}: {len(win)} F2 ref_12b windows of {cho} on these frames, need one')
        ref = cua_so_khung(pth, [k['id'] for k in win[0]['khung']], sha, a6['plumb_xy_m'])
        ref.update(luot_id=win[0].get('luot_id'), session_sha256=win[0].get('session_sha256'))
        Ws = out[cho]['W'] if cho in out else w_of(cho)
        nguon.append(dict(vai=f'khung_12b {cho}', file=os.path.basename(pth), duong_dan=os.path.abspath(pth),
                          sha256=sha))
        if Ws:
            out[cho] = dict(out.get(cho) or {}, rho_12b_m=ref['rho_m'], rho_W_m=max(w['rho_m'] for w in Ws),
                            luot_12b=[ref['luot_id']], n_W=len(Ws), ref=ref, W=Ws)
            out[cho].setdefault('quy_tac', 'max |tam ma - diem doi| tren khung cua so (ref_12b F2; W tham_chieu)')
    return out


def kc_doan_of(paths, kcs):
    """{doanN: kiem_cheo dict} from the --kiem-cheo paths (segment = the 'doanN' directory of the path)."""
    import re
    out = {}
    for p, k in zip(paths or [], kcs or []):
        m = re.search(r'[\\/](doan\d+)[\\/]', os.path.abspath(p))
        if m:
            if m.group(1) in out:
                raise KhoaError(f'--kiem-cheo: two files of {m.group(1)}')
            out[m.group(1)] = k
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tb', action='append', required=True, help='LEVEL=c8b_3b json')
    ap.add_argument('--lan-lap', default=None, help='NHACTAM lan_lap.json with beta_cam_replay (F15 b)')
    ap.add_argument('--canh-ma-m', type=float, default=None, help='tag side a (m) of the car tag (bo_cham.side_m)')
    ap.add_argument('--ke-hoach', default=None, help='C8 plan written before C8 (anchored in khoa_truoc, GATE GC6-1)')
    ap.add_argument('--nhanh-tick', choices=BRANCHES, default=None,
                    help='U_t branch when the plan file does not carry nhanh_tick')
    ap.add_argument('--muc', default=','.join(LEVELS), help='levels of muc_kha_thi (comma separated)')
    ap.add_argument('--nguong', default=None, help='nguong_khoa_c8_v1.json (eps_H, CODE writes it at C8)')
    ap.add_argument('--a6', default=None, help='c8b_a6_v2 (h_goc_m, u_h_goc_m)')
    ap.add_argument('--session', action='append', default=None,
                    help='session.json of the scoring (car_side_m, tag_h_m, kappa, car_pose_version); repeat once per '
                         'camera session of the C8 session (da session (A)): they must agree, else mien thieu')
    ap.add_argument('--camera', action='append', default=None,
                    help='camera file c8_cam_u_vao (day_doi.dzc_m); repeatable, the largest dzc is used')
    ap.add_argument('--kiem-cheo', action='append', default=None,
                    help='KIEMCHEO kiem_cheo.json (luot3_g4) of a camera segment with C8 runs; repeat (largest used)')
    ap.add_argument('--xoay', default=None, help='F4 xoay_v1 (q_tag_m, u_q_m of goc) for dq and the lever')
    ap.add_argument('--u-h-chung', default=None, help='c8b_u_h_chung_v1 (P6 common, thi sai goc)')
    ap.add_argument('--truc-xe', default=None, help='F7 truc_xe_v1 of goc (u_a_k_deg, u_phi_hat_chuan_k_deg; '
                                                    'u_a_noi_bo_deg + dau_doan: the camera error straight to W)')
    ap.add_argument('--truc-xe-cho', action='append', default=None,
                    help='CHO=F7 truc_xe_v1 of K1 / K2 (dau_doan for u_beta_cam of that place); goc comes from --truc-xe')
    ap.add_argument('--neo', default=None, help='KIEMCHEO neo_luot.json (c7_neo_luot_v1, theo_cho.goc)')
    ap.add_argument('--u-moc', action='append', default=None,
                    help='CAM U_moc file with U_moc_hieu_xy_m, U_moc_hieu_psi_rad; repeat (largest used)')
    ap.add_argument('--tham-chieu', action='append', default=None,
                    help='CHO=CAM T11 tham_chieu_v1 of one run at CHO (goc, K1, K2); repeat for every run (cach A: '
                         'u_beta_cam takes the largest psi_tan_rad at the place); a6_sha256 must be the --a6 file')
    ap.add_argument('--cua-so', default=None, help='HIEUCHUAN F2 cua_so_v1.jsonl (ref_12b windows, for --khung-12b)')
    ap.add_argument('--khung-12b', action='append', default=None,
                    help='CHO=frames.csv of the ref_12b provider run at CHO (J v6 line 59): its F2 ref_12b window '
                         '(same cho, frames_sha256) gives the footprint rho of u_beta_cam (cach A)')
    ap.add_argument('--commit-hil', default=None,
                    help='40-hex hil commit that will score the runs; required with bo_tick (O1 v11.34 Q4)')
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
        if a.commit_hil is not None and not _hex40(a.commit_hil):
            raise KhoaError(f'--commit-hil {a.commit_hil!r}: 40 lowercase hex digits')
        if a.canh_ma_m is not None and not (0.0 < a.canh_ma_m < 1.0):
            raise KhoaError(f'--canh-ma-m {a.canh_ma_m}: the tag side in metres')
        extra = {}
        for key, path, vai in (('nguong', a.nguong, 'nguong'), ('a6', a.a6, 'a6'), ('ses', a.session, 'session'),
                               ('cam', a.camera, 'camera'), ('kiem_cheo', a.kiem_cheo, 'kiem_cheo'),
                               ('xoay', a.xoay, 'xoay'), ('u_h', a.u_h_chung, 'u_h_chung'),
                               ('truc_xe', a.truc_xe, 'truc_xe'), ('neo', a.neo, 'neo'), ('u_moc', a.u_moc, 'u_moc')):
            for pth in (path if isinstance(path, list) else [path] if path else []):   # da session (A): repeatable
                with open(pth, encoding='utf-8') as f:
                    d = json.load(f)
                if key in ('ses', 'cam', 'kiem_cheo', 'u_moc'):
                    extra.setdefault(key, []).append(d)
                else:
                    extra[key] = d
                nguon.append(dict(vai=vai, file=os.path.basename(pth), duong_dan=os.path.abspath(pth),
                                  sha256=cu.sha256(pth)))
        a6_sha = cu.sha256(a.a6) if a.a6 else None
        for s in a.tham_chieu or []:
            cho, sep, pth = s.partition('=')
            if not sep or cho not in CHO:
                raise KhoaError(f'--tham-chieu {s!r}: CHO=path, CHO in {CHO}')
            with open(pth, encoding='utf-8') as f:
                d = json.load(f)
            if not isinstance(d, dict) or d.get('schema') != 'tham_chieu_v1':
                raise KhoaError(f'--tham-chieu {pth}: not a tham_chieu_v1 file')
            if a6_sha is None or d.get('a6_sha256') != a6_sha:
                raise KhoaError(f'--tham-chieu {pth}: a6_sha256 {d.get("a6_sha256")!r} is not the --a6 file '
                                f'{a6_sha!r} (one A6 for the lock)')
            extra.setdefault('tham_chieu', {}).setdefault(cho, []).append(d)
            nguon.append(dict(vai=f'tham_chieu {cho}', file=os.path.basename(pth), duong_dan=os.path.abspath(pth),
                              sha256=cu.sha256(pth)))
        txc = {}
        if isinstance(extra.get('truc_xe'), dict):
            txc['goc'] = extra['truc_xe']
        for s in a.truc_xe_cho or []:
            cho, sep, pth = s.partition('=')
            if not sep or cho not in CHO or cho == 'goc' or cho in txc:
                raise KhoaError(f'--truc-xe-cho {s!r}: CHO=path, CHO K1 or K2, once each (goc: --truc-xe)')
            with open(pth, encoding='utf-8') as f:
                d = json.load(f)
            if not isinstance(d, dict) or d.get('schema') != 'truc_xe_v1' or d.get('cho') != cho:
                raise KhoaError(f'--truc-xe-cho {pth}: not a truc_xe_v1 of {cho}')
            txc[cho] = d
            nguon.append(dict(vai=f'truc_xe {cho}', file=os.path.basename(pth), duong_dan=os.path.abspath(pth),
                              sha256=cu.sha256(pth)))
        co_dd = any(isinstance(v.get('dau_doan'), list) and v['dau_doan'] for v in txc.values())
        if a.khung_12b or co_dd:
            extra['truc_xe_cho'] = txc
            extra['dau_chan'] = dau_chan_of(a, extra, nguon)
            extra.pop('truc_xe_cho')
            extra['kc_doan'] = kc_doan_of(a.kiem_cheo, extra.get('kiem_cheo'))
        out = build(tbs, ll, nguon, script_state(), a.canh_ma_m, nt, mk, commit_hil=a.commit_hil, **extra)
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
