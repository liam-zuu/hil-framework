#!/usr/bin/env python3
"""
c8b_u.py: lock the C8b numbers after C8 and write the C8 gate file (C8b design v4 sections 3, 4b, 5, 9, approved:
Liam 17/11, CODE_C8B local_cccce6cb, "Duyet (Recommended)"; CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_C8b_v4.md;
gate file format gate_c8_v1: 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md section 2; O1 v11.33 3.9 rule 2b).
Written by CODE_C8B local_21b7debc on feat/c8b-u.

khoa (after the C8 clips are tracked and the C8 runs have their clock files):
  troi_ppm       drift rate bound [ppm] = LE x the largest |D_end - D_start| / distance of the third centres over the
                 two bound channels of every C8 run (design 3 table, C8B4R2-4; a convention: fit noise may hide drift);
                 name of the camera scorer's constants file (CODE 17/11)
  dai            per level: [min, max] over the C8 runs of the width U~ - L and of D_gyro - D_RV (design 3 "Kiem moi
                 luot": a campaign run outside the C8 range of its level is not concluded)
  man_trap       {alpha_r_s_px, alpha_c_s_px, nguon}: the rolling-shutter read time per row / column x LE from test 3b
                 when it was usable; null = one frame (design 3 table, 4b; the scorer's names, CODE 17/11)
  The parts follow the sources P1-P7, A1-A5 of the approved "few hand measurements" page v10 section 2 (Liam 18/11 at
  CODE_C8B local_8b35516c, "Duyet v10, Codex ra bo cham CAM (Recommended)"; CLAUDE_THESIS 03_CODE/docs/
  2026-11-18_Trang_C8b_ngan_sach_khong_do_tay_v10.md), field names by v10 (Liam 18/11 at CODE_C8B local_6db0bd09,
  "Sua c8b_u theo ten v10 (Recommended)"; design: 03_CODE/docs/2026-11-18_Thiet_ke_C8b_o_U_v10_v2.md; the sum and the
  margin as O1 v11.39 J1, Liam 18/11 at THESIS "Duyet, ra vong khoa (Recommended)"):
  U_p  (m)       = LE x (P1 + ... + P7),  U_psi (deg) = LE x (A1 + ... + A6), linear, parts may overlap
                 (conservative), none subtracted; U_p, U_psi bound ONE frame (per-frame noise P7, A5 inside).
    P1 image -> floor   max(cross-validation largest at the marks on the clip-mean pose, per frame) + the largest
                        true shift of a mark left in the fit, nguong + 2 u_so (dropped when the observed shift
                        exceeds nguong + u_so; the comparison error u_so may hide as much again)          [camera file]
    P2 tag-body centre  |o_u - R(dpsi_u) q| + du + (|q| + du) A1 + |o_u| (A1 + ... + A6)  [rad; o_u = (dx, dy), dpsi_u
                        the tag_offset of the scorer; last term: the heading error turns the correction vector,
                        C8BU-R1-3]                                                                 [c8b_xoay.py]
    P3 / A2 plate re-seating, position / angle (3 lifts, car still)                                  [camera file]
    P4 / A4 3b          d_p_quay / d_psi_quay                                                         [c8b_3b.py]
    P5 plumb / zc       (h / H) dn + h dzc (rho + dn) / (H (H - dzc - h))  (C8BS-R7-1..3)            [camera file]
    P6 tag height       (rho + s / sqrt 2) dh / (H - h - dh), dh = max_i |c_i - h| + e   (c_i the four corner heights by
                        caliper, h the height the scorer uses, e the caliper error)                [caliper + camera]
    P7 / A5 noise       largest deviation from the mean of a still mark, per frame (not sigma)        [camera file]
    A1 mounting angle   |dpsi_u - theta| + atan(2 e / s) + max edge angle^2 + plate-body angle from the drawing; theta
                        = the Kabsch angle of the black square on the plate, mean of its four edge angles by caliper
                        (a distorted print is seen, C8BU-R1-6; Liam 18/11 at CODE_C8B "Thuoc kep 4 canh")  [caliper]
    A3 camera by area   bound on the Kabsch heading of the car tag (96.4 mm) from the camera model, carried from the
                        held-out mark pairs (page v10 A3; A3.quy_ve_ma_noc must be true, C8BU-R1-4)    [camera file]
    A6 tag tilt         rho (sum_i |c_i - mean c| + 8 e) / (2 sqrt 2 s (H - dzc - max c_i - e))  [rad]: the parallax
                        step uses one h for the four corners, a corner higher by d moves out by (P_i - C) d / (H - c_i),
                        which turns the Kabsch fit of car_pose; domain enforced: sum |c_i - mean| <= 2 mm, rho <= 3 m
                        (C8BU-R1-8). Not in page v10: Liam 18/11 at CODE_C8B "Them A6 (Recommended)"
  Test 5 (ruler at the stop points, dropped in v10, kept as an option) is an end-to-end check: every difference
  <= U_p, else 'xem_lai' (not widened by itself). 3b with ga_ok false: refused (set-up redone).
cong (after every C8 figure-8 run is scored on its clock grid with the locked constants):
  O1 3.9 rule 2b: for each gate level, the runs of the C8 plan written before C8 (slot, task, controller); a rerun
  only replaces a lost run of its slot; runs lost to the measurement system or without a finite U_t are left out;
  each level needs at least one PID and one LADRC run (else C8 is not finished: refused).
  u_t_s of a run = the largest node U_t of its grid + the half width of its clock interval (design 9: so that U-hat
  keeps the clock part the grid carries; node scoring itself uses U_t without the half width).
  U_t,C8(level) = max u_t_s; U-hat: G2 = U_p, G3 = U_p + U_t,C8 v_cmd_max, G5 = U_psi + deg(U_t,C8 wz_cmd_max), with
  the schedule peaks WITHOUT the upper-bound margin; report-only: U-hat > half the threshold (strict). G1, G4 always
  decide. The level blocks come from gate_c8.level_block and the file is written with gate_c8.write_gate (CODE_GATE:
  checked before the file is opened), then read back with gate_c8.read_gate_c8; an existing file is never replaced.

  python tools/c8b_u.py khoa --gio r1=gio_r1.json ... --tb 3b.json --cam-u c8_cam_u_vao.json \
      --thuoc-kep thuoc_kep.json --xoay xoay.json [--u-imu u_imu_dong.json] [--phep5 phep5.csv] --out c8b_khoa.json
  python tools/c8b_u.py cong --khoa c8b_khoa.json --ke-hoach ke_hoach_c8.csv --levels O1-F02,O1-F05,O1-F08 \
      --u-t-branch bo_tick --luot SLOT=RUN_ID,gio.json,luoi.json ... [--cam cam.json] --c8-dir 05_DATA/results/C8/<d> \
      --out gate_c8.json
Last line: #ket_qua,ma=0 written / 2 refused or error.

c8_cam_u_vao.json (CODE_CAM / CODE_KIEMCHEO / CODE_NHACTAM; m and deg; every number used finite and >= 0; other keys
  inside a block are information; a top-level key not listed is refused; format agreed 18/11 with CODE_CAM):
  {"version": "c8_cam_u_vao_v2",
   "P1": {"bo_nhom_max_m", "tung_khung_max_m", ...}, "moc_dich": {"nguong_m", "u_so_m", ...},
   "P3": {"vi_tri_max_m", ...}, "A2": {"goc_max_deg", ...}, "than_dich": {...} (information),
   "P7": {"lech_max_m", ...}, "A3": {"max_deg", "quy_ve_ma_noc": true, ...}, "A5": {"lech_max_deg", ...},
   "day_doi": {"dn_m", "dzc_m", "kappa"}, "do_cao": {"rho_max_m", "h_m", "H_m"},
   "bo_cham": {"tag_offset": [dx_m, dy_m, dpsi_rad] or null, "side_m"}, "nguon": ..., "ghi_chu": ...}
  bo_cham.tag_offset = the offset the scorer gives c7_measure.track for the C8 runs (null = none: P2, A1 then carry
  the whole offset and angle); side_m = the side of the black square of the car tag.
thuoc_kep.json (CODE_C8B from the numbers Liam sends; guide C2.4 and B1 extended to four edges):
  {"version": "c8b_thuoc_kep_v2", "cao_4_goc_mm": [front-left, front-right, back-right, back-left] (on the car, after
  the last re-seating), "e_mm": caliper error, "s_mm": side of the black square (distance between the two points of an
  edge), "canh": {"truoc": [a, b], "phai": [a, b], "sau": [a, b], "trai": [a, b]} (black edge to plate edge at the two
  corners of that edge; a at the FRONT corner for phai / trai, at the LEFT corner for truoc / sau),
  "tam_than_deg": plate-body angle from the drawing (0 only with "tam_than_gioi_han" written), "nguon", "ghi_chu"}
xoay.json: from c8b_xoay.py (c8b_xoay_v1; q_tag_m, du_m).
  do_cao.rho_max_m = the largest horizontal distance of the car tag centre from the plumb point over the image area
  used; do_cao.H_m, h_m = the zc and tag height of the scoring session (day_doi.kappa = its kappa, must equal h / H).
  day_doi.dn_m = bound on |plumb point used - true point under the camera centre| on the day, and day_doi.dzc_m =
  bound on |zc used - true camera height| on the day: each the absolute error of the 16/11 measurement plus what
  the camera may have moved since, shown by a camera-centre estimate with its own error (C8BS-R7-1, R7-2: a median
  shift of the floor marks in the image, or a zc fit with zc as its prior, is not such a bound).
ke_hoach_c8.csv (written before C8): lines '# nhanh_tick=bo_tick|giu_tick', '# commit_hil=<sha>' (O1 v11.34 Q4),
  then the CSV slot, task, controller (PID / LADRC). Each --luot grid file must come from its clock file
  (gio_sha256, task, nodes), be scored with all five conditions deciding, by commit_hil, on the plan's tick branch.
phep5.csv: diem, chenh_m (ruler minus camera at a stop point, absolute or signed).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_khoa_v3'     # v3 = parts by page v10 (P1-P7, A1-A5); a v1, v2 lock is refused by cong (C8BS-R7-4)
CAM_VERSION = 'c8_cam_u_vao_v2'
CAM_KEYS = {'version', 'P1', 'moc_dich', 'P3', 'A2', 'than_dich', 'P7', 'A3', 'A5', 'day_doi', 'do_cao', 'bo_cham',
            'nguon', 'ghi_chu'}
TK_VERSION = 'c8b_thuoc_kep_v2'      # v2: four edges (C8BU-R1-6, Liam 18/11 "Thuoc kep 4 canh")
XOAY_VERSION = 'c8b_xoay_v1'
XOAY_KEYS = {'version', 'q_tag_m', 'du_m', 'spread_m', 'se3_m', 'n_doan', 'n_uoc', 'cho', 'troi_lon', 'quy_uoc',
             'gioi_han', 'written_at', 'script'}
P_PARTS = ('P1_anh_san', 'P2_lech_tam', 'P3_nhac_tam', 'P4_quay', 'P5_day_doi_zc', 'P6_do_cao', 'P7_nhieu_khung')
A_PARTS = ('A1_goc_dan', 'A2_nhac_tam', 'A3_huong_vung', 'A4_quay', 'A5_nhieu_khung', 'A6_nghieng_ma')
LE = 1.25
CONTROLLERS = ('PID', 'LADRC')
MASKABLE = ('g2', 'g3', 'g5')
BRANCHES = ('bo_tick', 'giu_tick')


class UError(Exception):
    pass


def _num(x, what, lo=0.0):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) < lo:
        raise UError(f'{what}: {x!r} is not a finite number >= {lo}')
    return float(x)


def _fin(x, what):
    """A finite number of either sign."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise UError(f'{what}: {x!r} is not a finite number')
    return float(x)


def _get(d, path):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            raise UError(f'camera input: no field {path}')
        cur = cur[k]
    return cur


# ── khoa ─────────────────────────────────────────────────────────────────────────────────────

def _tag_offset(cam):
    bc = _get(cam, 'bo_cham')
    if not isinstance(bc, dict) or 'tag_offset' not in bc:
        raise UError('camera input: no field bo_cham.tag_offset (null when the scorer does not correct)')
    off = bc['tag_offset']
    if off is None:
        return 0.0, 0.0, 0.0
    if not isinstance(off, list) or len(off) != 3:
        raise UError(f'bo_cham.tag_offset {off!r}: [dx_m, dy_m, dpsi_rad] or null')
    return tuple(_fin(x, f'bo_cham.tag_offset[{i}]') for i, x in enumerate(off))


EDGE_SIGN = {'truoc': 1, 'phai': 1, 'sau': -1, 'trai': -1}
EDGE_MAX_RAD = 0.05        # each edge angle within about 2.9 deg: the small-angle mean below holds [DE XUAT]
TK_KEYS = {'version', 'cao_4_goc_mm', 'e_mm', 's_mm', 'canh', 'tam_than_deg', 'tam_than_gioi_han', 'nguon', 'ghi_chu'}


def _caliper(tk):
    """Caliper file c8b_thuoc_kep_v2 (Liam 18/11 at CODE_C8B: four edges, C8BU-R1-6). canh: for each edge of the black
    square (truoc, phai, sau, trai of the car), [a, b] mm = black edge to plate edge at the two corners of that edge,
    a at the FRONT corner for phai / trai and at the LEFT corner for truoc / sau; s_mm = the distance between the two
    corners (the side of the black square). Edge angle (tag minus plate, counter-clockwise +): +atan((a - b) / s) for
    truoc, phai, -atan((a - b) / s) for sau, trai. For small distortions the Kabsch angle of the four corners is the
    mean of the four edge angles (sum of a_i x d_i over 8 h^2, each edge contributing 2 h^2 theta_edge), so a sheared
    or bent print, which one edge alone does not show, is taken in."""
    if not isinstance(tk, dict) or tk.get('version') != TK_VERSION:
        raise UError(f'caliper input: version is not {TK_VERSION}')
    extra = sorted(set(tk) - TK_KEYS)
    if extra:
        raise UError(f'caliper input: unknown top-level fields {extra}')
    c = tk.get('cao_4_goc_mm')
    if not isinstance(c, list) or len(c) != 4:
        raise UError('caliper input: cao_4_goc_mm needs the four corner heights')
    c = np.array([_num(x, f'cao_4_goc_mm[{i}]') for i, x in enumerate(c)]) * 1e-3
    e = _num(tk.get('e_mm'), 'caliper e_mm') * 1e-3
    s = _num(tk.get('s_mm'), 'caliper s_mm') * 1e-3
    if not s > 0:
        raise UError('caliper s_mm must be > 0')
    canh = tk.get('canh')
    if not isinstance(canh, dict) or set(canh) != set(EDGE_SIGN):
        raise UError(f'caliper canh: the four edges {sorted(EDGE_SIGN)} needed (one edge does not show a distorted '
                     f'print, C8BU-R1-6)')
    goc = {}
    for k, sg in EDGE_SIGN.items():
        v = canh[k]
        if not isinstance(v, list) or len(v) != 2:
            raise UError(f'caliper canh.{k}: [a_mm, b_mm]')
        a, b = (_num(x, f'caliper canh.{k}') * 1e-3 for x in v)
        goc[k] = sg * math.atan((a - b) / s)
        if abs(goc[k]) > EDGE_MAX_RAD:
            raise UError(f'caliper canh.{k}: edge angle {math.degrees(goc[k]):.2f} deg beyond '
                         f'{math.degrees(EDGE_MAX_RAD):.1f} deg (small-angle mean not valid; check the numbers)')
    theta = sum(goc.values()) / 4
    xap_xi = max(v * v for v in goc.values())     # second-order rest of the small-angle mean (<= 2.5e-3 rad in domain)
    tt = _num(tk.get('tam_than_deg'), 'caliper tam_than_deg')
    if tt == 0 and not str(tk.get('tam_than_gioi_han') or '').strip():
        raise UError('tam_than_deg 0 needs tam_than_gioi_han (the limit written: plate-body error not bounded)')
    return c, e, s, theta, tt, {k: math.degrees(v) for k, v in goc.items()}, xap_xi


CAM_NUMS = ('P1.bo_nhom_max_m', 'P1.tung_khung_max_m', 'moc_dich.nguong_m', 'moc_dich.u_so_m', 'P3.vi_tri_max_m',
            'A2.goc_max_deg', 'P7.lech_max_m', 'A3.max_deg', 'A5.lech_max_deg', 'day_doi.dn_m', 'day_doi.dzc_m',
            'day_doi.kappa', 'do_cao.rho_max_m', 'do_cao.H_m', 'do_cao.h_m', 'bo_cham.side_m')


def check_cam(cam):
    """The camera file alone (for the writers' tests, CODE_KIEMCHEO 18/11): version, top-level keys, every number
    used finite and >= 0, tag_offset, side > 0, kappa = h / H, H - dzc > h. Raises UError; returns None."""
    if not isinstance(cam, dict) or cam.get('version') != CAM_VERSION:
        raise UError(f'camera input: version is not {CAM_VERSION} (a file with the names of tests 1, 2 is refused)')
    extra = sorted(set(cam) - CAM_KEYS)
    if extra:
        raise UError(f'camera input: unknown top-level fields {extra} (nothing is dropped silently)')
    v = {p: _num(_get(cam, p), p) for p in CAM_NUMS}
    _tag_offset(cam)
    if _get(cam, 'A3').get('quy_ve_ma_noc') is not True:
        raise UError('A3.quy_ve_ma_noc must be true: A3.max_deg bounds the Kabsch heading of the car tag (96.4 mm), '
                     'carried from the mark pairs through the camera model (page v10 A3, C8BU-R1-4), not a pair angle')
    if not v['bo_cham.side_m'] > 0:
        raise UError('bo_cham.side_m must be > 0')
    H, h = v['do_cao.H_m'], v['do_cao.h_m']
    if not H > h:
        raise UError(f'do_cao: camera height {H} m not above the tag height {h} m')
    if not H - v['day_doi.dzc_m'] > h:
        raise UError(f'day_doi.dzc_m {v["day_doi.dzc_m"]} m: camera height {H} m minus it not above the tag height {h} m')
    if abs(v['day_doi.kappa'] - h / H) > 1e-6 * (h / H):  # C8BS-R7-3: H, h here are the ones of the scoring session
        raise UError(f'day_doi.kappa {v["day_doi.kappa"]} is not do_cao.h_m / do_cao.H_m = {h / H} (the kappa of the '
                     f'scoring)')


def u_p_u_psi(cam, tb, tk, xo):
    """U_p (m), U_psi (deg) with their parts P1-P7, A1-A6 (page v10 section 2; design v2)."""
    check_cam(cam)
    if not isinstance(xo, dict) or xo.get('version') != XOAY_VERSION:
        raise UError(f'turning-in-place input: version is not {XOAY_VERSION}')
    extra = sorted(set(xo) - XOAY_KEYS)
    if extra:
        raise UError(f'turning-in-place input: unknown top-level fields {extra}')
    if tb.get('ga_ok') is not True:                     # C8BU-R1-9: a real boolean, not a truthy string
        raise UError('test 3b: ga_ok not true (set-up not within 2 mm at the slowest part): redo 3b, numbers not used')
    dpq = _num(tb.get('d_p_quay_m'), '3b d_p_quay_m')
    dsq = _num(tb.get('d_psi_quay_deg'), '3b d_psi_quay_deg')
    H, h = _num(_get(cam, 'do_cao.H_m'), 'do_cao.H_m'), _num(_get(cam, 'do_cao.h_m'), 'do_cao.h_m')
    rho = _num(_get(cam, 'do_cao.rho_max_m'), 'do_cao.rho_max_m')
    side = _num(_get(cam, 'bo_cham.side_m'), 'bo_cham.side_m')
    dzc = _num(_get(cam, 'day_doi.dzc_m'), 'day_doi.dzc_m')
    c4, e, s_cal, theta, tam_than, goc_canh, xap_xi = _caliper(tk)
    Hl = H - dzc                                            # lowest camera height allowed by the zc bound (R1-8)
    dh = float(np.max(np.abs(c4 - h))) + e                  # corner heights against the h the scorer uses
    if not Hl - h - dh > 0 or not Hl - c4.max() - e > 0:
        raise UError('caliper heights: not below the camera')
    cao = (rho + side / math.sqrt(2)) * dh / (Hl - h - dh)
    # A6: one h for the four corners; corner i at c_i moves by (P_i - C)(c_i - h)/(H - c_i); the small-angle Kabsch
    # turn is (sum_i delta_i a_i) x (T - C) / (2 s^2), |.| <= rho s / sqrt 2 sum |delta_i - mean delta| / (2 s^2).
    # Domain of the small-angle form enforced (C8BU-R1-8): corners within 2 mm in all, rho within 3 m
    spread_c = float(np.sum(np.abs(c4 - c4.mean())))
    if spread_c > 0.002 + 1e-12 or rho > 3.0:
        raise UError(f'tag tilt: sum |c_i - mean| {spread_c * 1e3:.2f} mm (<= 2) or rho {rho} m (<= 3) outside the '
                     f'domain of the A6 bound')
    nghieng = rho * (spread_c + 8 * e) / (2 * math.sqrt(2) * side * (Hl - c4.max() - e))
    # plumb point and zc reused from 16/11 (Liam 17/11): car_pose corrects P = p + kappa (C - p), kappa = h / H with
    # the H, C of the session; truth P* = p + kappa* (C* - p), |C* - p| = rho* H* / (H* - h) (central projection from
    # the true centre), so |P - P*| <= (h / H) dn + h dzc rho* / (H (H* - h)), H* >= H - dzc, rho* <= rho + dn
    # (C8BS-R7-1..3: dn, dzc bound the absolute error of the camera centre position and height, not a check threshold)
    dn = _num(_get(cam, 'day_doi.dn_m'), 'day_doi.dn_m')
    day_doi = h / H * dn + h * dzc * (rho + dn) / (H * (H - dzc - h))
    def f(path):
        return _num(_get(cam, path), path)
    dx, dy, dpsi = _tag_offset(cam)
    # A1: Kabsch angle of the tag to the plate from the four edges (caliper), against the dpsi the scorer uses
    a1 = math.degrees(abs(dpsi - theta) + math.atan(2 * e / s_cal) + xap_xi) + tam_than
    s = dict(A1_goc_dan=a1, A2_nhac_tam=f('A2.goc_max_deg'), A3_huong_vung=f('A3.max_deg'), A4_quay=dsq,
             A5_nhieu_khung=f('A5.lech_max_deg'), A6_nghieng_ma=math.degrees(nghieng))
    # P2: q (tag minus turning centre, tag frame) from c8b_xoay. The scorer takes the body centre as tag -
    # R(psi_body_measured) o_u, o_u = (dx, dy); truth: tag - R(psi_body) o, o = R(dpsi_true) q_true. So the error is
    # <= |o - o_u| + |o_u| |psi_body error| <= |o_u - R(dpsi_u) q| + du + (|q| + du) A1 + |o_u| (A1 + ... + A6)
    # (C8BU-R1-3: the heading error turns the correction vector)
    q = xo.get('q_tag_m')
    if not isinstance(q, list) or len(q) != 2:
        raise UError('turning-in-place input: q_tag_m [x, y] needed')
    qx, qy = (_fin(v, 'xoay q_tag_m') for v in q)
    du = _num(xo.get('du_m'), 'xoay du_m')
    cd, sd = math.cos(dpsi), math.sin(dpsi)
    lech = (math.hypot(dx - (cd * qx - sd * qy), dy - (sd * qx + cd * qy)) + du
            + (math.hypot(qx, qy) + du) * math.radians(a1) + math.hypot(dx, dy) * math.radians(sum(s.values())))
    # a mark is dropped when |observed shift| > nguong + u_so; not dropped only says |true shift| <= nguong + 2 u_so
    # (the comparison error may pull the observed shift down by u_so; CODE_KIEMCHEO 18/11, Codex KC1-3)
    moc = f('moc_dich.nguong_m') + 2 * f('moc_dich.u_so_m')
    p = dict(P1_anh_san=max(f('P1.bo_nhom_max_m'), f('P1.tung_khung_max_m')) + moc,
             P2_lech_tam=lech, P3_nhac_tam=f('P3.vi_tri_max_m'), P4_quay=dpq, P5_day_doi_zc=day_doi,
             P6_do_cao=cao, P7_nhieu_khung=f('P7.lech_max_m'))
    assert tuple(p) == P_PARTS and tuple(s) == A_PARTS
    return LE * sum(p.values()), p, LE * sum(s.values()), s


def khoa(gios, tb, cam, tk, xo, u_imu=None, phep5=None, tb_name='', tb_sha=''):
    """gios: {run_id: c8b_gio_v1 dict} of the C8 runs."""
    if not gios:
        raise UError('no C8 clock file')
    rates, dai = [], {}
    for rid, g in gios.items():
        if g.get('format') != 'c8b_gio_v1':
            raise UError(f'{rid}: not a c8b_gio_v1 file')
        if not g.get('ok'):
            continue                                  # a run without a finite clock bound gives no range, no rate
        if g.get('rate_luot') is not None:
            rates.append(_num(g['rate_luot'], f'{rid} rate_luot'))
        d = dai.setdefault(g['task'], dict(width_s=[math.inf, -math.inf], gyro_minus_rv_s=[math.inf, -math.inf],
                                           runs=[]))
        for k in ('width_s', 'gyro_minus_rv_s'):
            v = float(g[k])
            d[k] = [min(d[k][0], v), max(d[k][1], v)]
        d['runs'].append(rid)
    if not rates:
        raise UError('no C8 run with a drift rate (every run short or not ok)')
    U_p, p_parts, U_psi, s_parts = u_p_u_psi(cam, tb, tk, xo)
    _, _, _, theta, _, goc_canh, _ = _caliper(tk)
    sh = (tb.get('B') or {}).get('shutter') or {}
    man_trap = None
    if sh.get('use_alpha'):
        man_trap = dict(alpha_r_s_px=_num(abs(sh['alpha_r_le_s_px']), 'alpha_r'),
                        alpha_c_s_px=_num(abs(sh['alpha_c_le_s_px']), 'alpha_c'),
                        nguon=dict(file=tb_name, sha256=tb_sha, le=LE,
                                   do=dict(alpha_r_s_px=sh['alpha_r_s_px'], alpha_c_s_px=sh['alpha_c_s_px'])))
    out = dict(version=VERSION, le=LE, troi_ppm=LE * max(rates) * 1e6, troi_rate_luot=sorted(rates),
               dai=dai, man_trap=man_trap, man_trap_kieu='alpha' if man_trap else 'mot_khung',
               mot_khung_s=sh.get('mot_khung_s'), U_p_m=U_p, U_p_phan_m=p_parts, U_psi_deg=U_psi,
               U_psi_phan_deg=s_parts, thuoc_kep=dict(goc_canh_deg=goc_canh, goc_kabsch_deg=math.degrees(theta)))
    if u_imu is not None:
        if u_imu.get('version') != 'u_imu_dong_v2':
            raise UError('u_imu file: version is not u_imu_dong_v2')
        if u_imu.get('thu'):
            raise UError('u_imu file marked thu (trial tau): not lockable')
        out['U_IMU'] = {m: v.get('U_IMU_dong_rad') for m, v in (u_imu.get('muc') or {}).items()}
    if phep5 is not None:
        over = [r for r in phep5 if abs(r[1]) > U_p + 1e-12]
        out['phep5'] = dict(n=len(phep5), max_m=max((abs(r[1]) for r in phep5), default=None),
                            ket_qua='xem_lai' if over else 'dat', vuot=[r[0] for r in over])
    return out


# ── cong (rule 2b, gate file) ─────────────────────────────────────────────────────────────────────

def read_plan(path):
    """The C8 plan written before C8: '# key=value' lines (nhanh_tick = bo_tick / giu_tick; commit_hil = the hil
    commit of the C8b scripts and the scorer, gone through Codex, O1 v11.34 Q4: without it the branch is giu_tick),
    then the CSV slot, task, controller."""
    cu.check_path(path)
    meta, body = {}, []
    with open(path, newline='', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                k, _, v = line[1:].strip().partition('=')
                meta[k.strip()] = v.strip()
            elif line.strip():
                body.append(line)
    plan = {}
    for r in csv.DictReader(body):
        if r.get('controller') not in CONTROLLERS:
            raise UError(f'{path}: controller {r.get("controller")!r} (PID / LADRC)')
        if r['slot'] in plan:
            raise UError(f'{path}: slot {r["slot"]} twice')
        plan[r['slot']] = dict(task=r['task'], controller=r['controller'])
    if not plan:
        raise UError(f'{path}: empty plan')
    return plan, meta


def check_pair(rid, gio, luoi, gio_sha):
    """C8BS-13: the grid result belongs to this clock file (SHA, task, nodes); C8BS-15: it was scored with all five
    conditions deciding (the classification of rule 2b is frozen before the report-only list)."""
    if gio.get('format') != 'c8b_gio_v1':
        raise UError(f'run {rid}: not a c8b_gio_v1 clock file')
    if luoi.get('version') != 'c8b_luoi_v1':
        raise UError(f'run {rid}: not a c8b_luoi_v1 grid file')
    if luoi.get('gio_sha256') != gio_sha:
        raise UError(f'run {rid}: the grid file was made from another clock file (gio_sha256)')
    if luoi.get('task') != gio.get('task'):
        raise UError(f'run {rid}: grid task {luoi.get("task")} but clock task {gio.get("task")}')
    if list(luoi.get('nodes_s') or []) != list(gio.get('nodes_s') or []):
        raise UError(f'run {rid}: grid nodes differ from the clock file nodes')
    if luoi.get('report_only'):
        raise UError(f'run {rid}: scored with report-only conditions {luoi["report_only"]}: rule 2b needs the '
                     f'classification with all five deciding (O1 3.9 2b, C8BS-15)')


def run_u_t_parts(gio, luoi):
    """(u_t_s, parts) of one C8 run for rule 2b (O1 v11.34 3.9 2b: h + u_max), or (None, why): lost, or U_t not
    finite at every node."""
    if not gio.get('ok') or luoi.get('outcome') == 'hong_he_do':
        return None, 'lost to the measurement system or no finite clock bound'
    ut = luoi.get('u_t_nodes') or []
    if not ut or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                     for x in ut):
        return None, 'U_t not finite (or negative) at every node'
    h = _num(gio.get('half_s'), 'half_s')
    if len(ut) != len(gio.get('nodes_s') or []):
        raise UError('U_t of every node needed')
    return max(ut) + h, dict(u_max_s=max(ut), h_s=h, n_nut=len(ut))


def run_u_t(gio, luoi):
    return run_u_t_parts(gio, luoi)[0]


def _lost(gio, luoi):
    return gio.get('ket_luan') == 'hong_he_do' or luoi.get('outcome') == 'hong_he_do'


def slot_runs(plan, luot):
    """luot: [(slot, run_id, gio, luoi, gio_sha256)] in the order run. Returns {slot: (run_id, u_t_s or None, parts
    or why)} with the rerun rule: a later run of a slot only after a lost one. A run (id or clock file) in two slots
    is refused (C8BS-14)."""
    by, seen_id, seen_sha = {}, set(), set()
    for slot, rid, g, l, gsha in luot:
        if slot not in plan:
            raise UError(f'run {rid}: slot {slot} not in the C8 plan (no run added after the plan, O1 3.9 2b)')
        if g.get('task') != plan[slot]['task']:
            raise UError(f'run {rid}: task {g.get("task")} but slot {slot} is {plan[slot]["task"]}')
        if g.get('controller') != plan[slot]['controller']:
            raise UError(f'run {rid}: controller from its log {g.get("controller")!r} but slot {slot} is '
                         f'{plan[slot]["controller"]} (C8BS-14)')
        if rid in seen_id or gsha in seen_sha:
            raise UError(f'run {rid}: the same run or clock file given twice (C8BS-14)')
        seen_id.add(rid)
        seen_sha.add(gsha)
        check_pair(rid, g, l, gsha)
        by.setdefault(slot, []).append((rid, g, l))
    out = {}
    for slot, lst in by.items():
        chosen = None
        for rid, g, l in lst:
            if chosen is not None:
                raise UError(f'slot {slot}: run {rid} after a run that was not lost (a rerun only replaces a lost run)')
            if not _lost(g, l):
                chosen = (rid,) + run_u_t_parts(g, l)
        out[slot] = chosen if chosen is not None else (lst[-1][0], None, 'every run of the slot lost')
    return out


def check_branch(branch, meta, luot):
    """C8BS-16 (O1 v11.34 SC-12 Q4): the branch is the one written in the plan before C8; 'bo_tick' needs the commit
    of the scripts in the plan, and every clock and grid file made by that commit, not dirty; every scored node used
    that branch."""
    if meta.get('nhanh_tick') != branch:
        raise UError(f'u_t_branch {branch} but the C8 plan says {meta.get("nhanh_tick")!r}')
    commit = meta.get('commit_hil', '')
    if branch == 'bo_tick' and len(commit) < 7:
        raise UError('bo_tick needs commit_hil (the scripts gone through Codex) in the C8 plan (O1 v11.34 Q4)')
    want = {'bo_tick': 'bo', 'giu_tick': 'giu'}[branch]
    for slot, rid, g, l, _ in luot:
        # C8BS-R4-1, R5-1: the nodes were scored whenever the clock file is ok, even when every node then says lost;
        # only a run stopped at the clock check (gio not ok) has no scorer to check
        scored = bool(g.get('ok'))
        checks = [('clock', g.get('script')), ('grid', l.get('script'))]
        if scored:
            checks.append(('scorer', l.get('scorer')))
        for nm, sc in checks:
            sc = sc or {}
            if commit and (not str(sc.get('commit', '')).startswith(commit) or sc.get('dirty') is not False):
                raise UError(f'run {rid}: {nm} not from commit_hil {commit} with the C8b chain clean')
        if not scored:
            continue
        # C8BS-16 round 4: every node scored by the canonical scorer of the repo, tracked and clean
        for k, nd in enumerate((l.get('scorer') or {}).get('nut') or [None]):
            if not nd or nd.get('file') != 'tools/c7_score.py' or nd.get('tracked') is not True                     or nd.get('dirty') is not False:
                raise UError(f'run {rid}: node {k} not scored by the tracked, clean tools/c7_score.py (C8BS-16)')
        if l.get('outcome') == 'hong_he_do':
            continue                                   # every node lost: no node carries a tick branch
        used = set(l.get('nhanh_tick') or [])
        if used != {want}:
            raise UError(f'run {rid}: nodes scored with tick branch {sorted(used)}, the plan says {want}')


def cong(k, plan, luot, levels, branch, c8_dir, cam=None, written_by=None, meta=None):
    import gate_c8
    import o1_gate
    from o1_tasks import FIG8
    check_khoa(k)                                       # C8BU-R1-7: also when called from code, not only from the CLI
    if branch not in BRANCHES:
        raise UError(f'u_t_branch {branch!r}: {BRANCHES}')
    if not levels or len(set(levels)) != len(levels) or any(lv not in FIG8 for lv in levels):
        raise UError(f'levels {levels!r}: distinct figure-8 tasks')
    check_branch(branch, meta or {}, luot)
    chosen = slot_runs(plan, luot)
    thr = {c: o1_gate.THRESHOLDS[c] for c in MASKABLE}
    per, parts = {}, {}
    for lv in levels:
        runs = []
        for slot, p in plan.items():
            if p['task'] != lv:
                continue
            if slot not in chosen:
                raise UError(f'slot {slot} ({lv}, {p["controller"]}) has no run')
            rid, ut, part = chosen[slot]
            parts[rid] = dict(slot=slot, task=lv, controller=p['controller'], u_t_s=ut,
                              **(part if isinstance(part, dict) else dict(bo_ra=part)))
            if ut is not None:
                runs.append(dict(run_id=rid, controller=p['controller'], u_t_s=ut))
        ctl = {r['controller'] for r in runs}
        if not set(CONTROLLERS) <= ctl:
            raise UError(f'{lv}: needs a run of each controller not lost and with a finite U_t (has {sorted(ctl)}): '
                         f'C8 not finished (O1 3.9 2b)')
        # CODE_GATE (17/11): the level block comes from gate_c8.level_block: peaks of the schedule WITHOUT the upper-bound
        # margin (schedule_peaks), U_t,C8 = max u_t_s, uhat, report_only by '>' on the recomputed bound
        per[lv] = gate_c8.level_block(lv, k['U_p_m'], k['U_psi_deg'], runs)
    return dict(format='gate_c8_v1', c8_dir=c8_dir, written_by=written_by or {},
                written_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                levels=list(levels), u_t_branch=branch,
                dieu_2b=dict(rule='report_only = {c in g2,g3,g5 : uhat[c] > 0.5 * threshold[c]}', thresholds=thr,
                             u_p_m=k['U_p_m'], u_psi_deg=k['U_psi_deg'], per_level=per),
                camera=dict(c8b=dict(k, u_t_luot_c8=parts), cam=cam or {}))


def write_gate(path, d):
    """Write with gate_c8.write_gate (CODE_GATE: the bytes are checked as the reader reads them before the file is
    opened, so a file the reader refuses is never written), then read it back with read_gate_c8. An existing file is
    never replaced (one gate file per C8, written once). Returns the GateC8 read back."""
    try:
        import gate_c8
    except ImportError:
        raise UError('tools/gate_c8.py (CODE_GATE) is not on this branch: the gate file cannot be checked, not written')
    cu.check_path(path)
    # C8BS-R3-1: claim the path with an exclusive create (atomic: of two writers only one gets it), then let
    # gate_c8.write_gate (which opens 'wb') replace our own empty placeholder; refused -> placeholder removed
    try:
        with open(path, 'xb'):
            pass
    except FileExistsError:
        raise UError(f'{path} exists: the C8 gate file is written once')
    try:
        sha = gate_c8.write_gate(path, d)
        g = gate_c8.read_gate_c8(path)
    except gate_c8.GateFileError as exc:
        if os.path.exists(path) and os.path.getsize(path) == 0:
            os.remove(path)
        raise UError(f'gate_c8 refused the content: {exc}')
    if g.sha256 != sha:
        raise UError(f'{path}: SHA read back {g.sha256} differs from the bytes written {sha}')
    return g


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────

def _json(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def check_khoa(k, name='khoa'):
    """C8BS-R7-4: cong takes only a lock of this version whose U_p, U_psi carry exactly the parts P1-P7, A1-A6 (a v1
    lock has no plumb / zc part, a v2 lock the names of tests 1, 2)."""
    if k.get('version') != VERSION or set(k.get('U_p_phan_m') or {}) != set(P_PARTS) \
            or set(k.get('U_psi_phan_deg') or {}) != set(A_PARTS):
        raise UError(f'{name}: not a {VERSION} file with the parts P1-P7, A1-A6')
    # C8BU-R1-7: every part a finite number >= 0, the margin LE, each total equal to LE x the sum of its parts
    if k.get('le') != LE:
        raise UError(f'{name}: le {k.get("le")!r} is not {LE}')
    for tot, parts in (('U_p_m', 'U_p_phan_m'), ('U_psi_deg', 'U_psi_phan_deg')):
        vals = [_num(v, f'{name} {parts}.{x}') for x, v in k[parts].items()]
        t = _num(k.get(tot), f'{name} {tot}')
        if not math.isclose(t, LE * sum(vals), rel_tol=1e-9, abs_tol=1e-15):
            raise UError(f'{name}: {tot} {t} is not {LE} x the sum of {parts} ({LE * sum(vals)})')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('khoa')
    a1.add_argument('--gio', action='append', required=True, help='RUN_ID=gio.json (every C8 run)')
    a1.add_argument('--tb', required=True)
    a1.add_argument('--cam-u', required=True)
    a1.add_argument('--thuoc-kep', required=True)
    a1.add_argument('--xoay', required=True)
    a1.add_argument('--u-imu')
    a1.add_argument('--phep5')
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('cong')
    a2.add_argument('--khoa', required=True)
    a2.add_argument('--ke-hoach', required=True)
    a2.add_argument('--levels', required=True)
    a2.add_argument('--u-t-branch', required=True, choices=BRANCHES)
    a2.add_argument('--luot', action='append', required=True, help='SLOT=RUN_ID,gio.json,luoi.json (in run order)')
    a2.add_argument('--cam')
    a2.add_argument('--c8-dir', required=True)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'khoa':
            gios, src = {}, []
            for s in a.gio:
                rid, p = s.split('=', 1)
                if rid in gios:
                    raise UError(f'run {rid} twice')
                gios[rid] = _json(p)
                src.append(p)
            phep5 = None
            if a.phep5:
                with open(a.phep5, newline='', encoding='utf-8') as f:
                    phep5 = [(r['diem'], float(r['chenh_m'])) for r in csv.DictReader(f)]
                src.append(a.phep5)
            out = khoa(gios, _json(a.tb), _json(a.cam_u), _json(a.thuoc_kep), _json(a.xoay),
                       _json(a.u_imu) if a.u_imu else None, phep5, os.path.basename(a.tb), cu.sha256(a.tb))
            src += [a.tb, a.cam_u, a.thuoc_kep, a.xoay] + ([a.u_imu] if a.u_imu else [])
            out['nguon'] = [dict(file=os.path.basename(p), sha256=cu.sha256(p)) for p in src]
            out['script'] = cu.script_state()
            cu.check_path(a.out)
            with open(a.out, 'x', encoding='utf-8') as f:
                f.write(json.dumps(_clean(out), indent=1, ensure_ascii=False, allow_nan=False))
            print(f'U_p {out["U_p_m"] * 1e3:.2f} mm, U_psi {out["U_psi_deg"]:.3f} deg, troi '
                  f'{out["troi_ppm"]:.0f} ppm, man trap {out["man_trap_kieu"]}')
        else:
            k = _json(a.khoa)
            check_khoa(k, a.khoa)
            plan, meta = read_plan(a.ke_hoach)
            luot = []
            for s in a.luot:
                slot, rest = s.split('=', 1)
                rid, g, l = rest.split(',')
                luot.append((slot, rid, _json(g), _json(l), cu.sha256(g)))
            st = cu.script_state()
            d = cong(k, plan, luot, [x.strip() for x in a.levels.split(',')], a.u_t_branch, a.c8_dir,
                     _json(a.cam) if a.cam else None,
                     dict(script='tools/c8b_u.py', repo='hil-framework', commit=st.get('commit'), dirty=st.get('dirty')),
                     meta)
            g = write_gate(a.out, _clean(d))
            print(f'gate file {a.out} sha256 {g.sha256}; ' + '; '.join(
                f'{lv}: report_only {d["dieu_2b"]["per_level"][lv]["report_only"]}' for lv in d['levels']))
    except (UError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_u: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
