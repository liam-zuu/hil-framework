#!/usr/bin/env python3
"""
c8b_a6.py: A6, the heading error of the car tag from its tilt, measured on the E clips (turning in place at goc / K1)
for the relative scoring (design C8B v7 section 2, 3, 10: c8b_a6_v2; J v6 = 06_SPECS/2026-11-19_Spec_O1_v11.41_Phu_luc_J_v6.md).
Written by CODE_C8B on feat/c8b-3b-ga, 19-20/11/2026 (M4; CODE order 19/11, Liam at CODE "duyet ca 8, giao viec cac
phien phu theo cach nhanh"; Liam at CODE_C8B 19/11: "Lam M4 o day"). Round 1/2 of Codex (A6C1-1..15, thamdinh
03_CODE/docs/review/2026-11-20_Tham_dinh_c8b_a6_M4.md) fixed 20/11 (CODE_C8B local_f64f609e); the ACCEPTED points of
round 2/2 (A6C2-2..10, rest of A6C1-1, A6C1-2) fixed 20/11 (CODE_C8B local_d24e5a74; Liam at CODE_C8B: "Sua ngay, Codex
sau (Recommended)": no round 3 now, one Codex pass after the data of session 2, L138).

22/11/2026 (CODE_C8B local_bc420001, Fable 5.2; Liam at CODE_C8B, box, verbatim: "Sua mo hinh (Recommended)"; analysis
05_DATA/results/C8/2026-11-20_run02/c8b/T7_a6_ra_mau_thuan/, note 03_CODE/docs/2026-11-22_Ra_mau_thuan_A6_T7_v1.md;
not through Codex yet, L138): on the E clips of session 2 the plane model below does not hold: in the tag frame the
corner deformation is u (x) slopes with FOUR slopes, the stretch slopes gm (-0.005, 0.005) and the slopes along the
edges gk (-0.018, 0.004) at K1, and the camera-IMU heading follows gk. So: do fits fit_gk (MO_HINH) and writes
g_E = gk (c8b_a6_cho_v3; g_gian, g_phang kept as information); gop takes the caliper g_cal as a reference only (not in
eps_g, not in the status), needs the held-out camera-IMU check of E clips (kiem_imu), accepts ONE mounting in the lap
file (mot_lan_lap, lap = 0) and the cell file c8b_a6_o as optional; new subcommands kiem_imu, goc, dong.

22/11/2026 night (CODE_C8B local_30b86c78, Fable 5.2; Liam at CODE_C8B, box, verbatim: "Lam theo de xuat (Recommended)"
after "de xuat giup toi phuong an tot nhat cho van de nay di"; note 03_CODE/docs/2026-11-22_Giam_can_A6_dong_kenh_huong_v1.md,
diagnosis 05_DATA/results/C8/2026-11-20_run02/c8b/T7_a6_dong_chan_doan/; not through Codex yet, L138): fit_gk on the
frames of the scored runs does not measure the slope of the tag (the car does not turn in place there, so the map
residual fixed in the floor, which changes with the position, cannot be told from the columns u (x) slope: same tag,
same paths, camera segment 3 gives gk_x -0.0094 and segment 2 -0.0150). So U_chuyen.dong is measured in the HEADING
channel: new subcommand dong_huong (c8b_a6_dong_v2, kenh huong_imu): per scored run and acceleration band, camera
heading minus IMU yaw on the frames of every OTHER run, regressed on per-run [1, time, omega] and shared [IMU scale,
u_y / 2, -u_x / 2]; SE by the jackknife over runs; the two half bands split at the MEDIAN acceleration of the frames
(the upper half of the range held 1.5 % of them). Same index |g_bar - g_b| + 2 |SE|, same leave-one-run-out; g_b is
still the camera number of the E clips. gop accepts both dong files; a v2 file must carry the lock of the A6 file.

Model (exact for the lifted points of car_pose): a corner at height h_i = hbar + g.r_i + v s_i (r_i the corner in the
tag frame, car_corners_body order ArUco 0..3, s = (+1, -1, +1, -1)) lifted with ONE height hbar lands at
    P_i = X_i + (g.r_i + v s_i) d_i,   d_i = (P_i - C) / (H - hbar),   X_i = T + R(psi) (1 + sigma) r_i
(C the plumb point, H = Z_c; sigma a print scale of the tag). v is fitted as a nuisance only (design 3: v is not
estimated from video; |v| <= v_max stays a domain assumption). g is fitted on the E frames of one place by least
squares with the per-frame pose (T, psi) eliminated (design 2b (a); the camera-IMU agreement is a check only).

  do    one place (goc | K1) of ONE plate and mounting (--tam-id, --lan-lap-id): the E frames (strictly between the one
        E_dau and the one E_cuoi window of F15 mien_E; F15 is required) with |omega| <= OMEGA_MAX_RAD_S, omega from a
        quadratic fit of the Kabsch heading over +-SG_HALF frames on the real times, no window across a gap (A6C1-12);
        alpha_r, alpha_c of the rolling shutter fitted as nuisance (L133, Liam via CODE 20/11, "Khop alpha
        (Recommended)"); rows / cols of the corners from the session camera, kept fixed for the same image in the
        replay (A6C1-12). Every fit (main, jackknife, replay) must converge and be well conditioned (A6C1-11).
        U_meo_E: the map residual held - full of KIEMCHEO F15 (a) at the four corners of every frame, refitted per H32
        case (case ids and signed changes kept, A6C1-3); a usable U_meo_E and s_E only from the FULL schema with
        trang_thai 'dat'; the diagnostic schema gives numbers in the block chan_doan only (A6C1-1). F15 is tied to the
        frames.csv / session (sha), the frame times, the corner order and the full H32 case list (A6C1-2).
        Identification per component (A6C1-11).
  gop   the A6 file c8b_a6_v2 of one plate: g_cal (4 caliper heights of the first mounting of the caliper plate,
        --tam-cal-id; another plate is a new plate: A6, P6 thieu, T16.8, A6C1-7), consistency checked PER PLACE
        (A6C1-6), eps_g = max(s_E, max_place |g_E - g_cal| + U_cal) + U_chuyen (lap + cho + dong, each once, vector
        norms; lap and dong from C8B files with schema, sha, status, numbers >= 0, A6C1-5), e_bc through the real
        car_pose path (camera centre of the session, plumb, kappa) with the correction evaluated at the MEASURED pose,
        grid refined in rho, direction, heading, g and v until the maximum changes < 10 % (A6C1-9), the held-out
        camera-IMU check from a c8b_a6_o_v1 file (schema, sha, the place fits it used, the lock, every cell
        |d| <= U_psi + U_IMU,D, A6C1-4), khoa_truoc_sha256 and the commit of the script (A6C1-4), the domain on the four
        corners (A6C1-8), h_goc_m / u_h_goc_m only when A6 and the c8b_u_h_chung_v1 file are dat (A6C1-7).
        Da session (A) (contract log 20/11 khuya; session 2 has three camera sessions, CODE_C8B fix/c8b-da-session):
        the places may come from different camera sessions when the rest of the set-up (L125 plumb, Z_c, tag_h,
        side, kappa) is the same; g, eps_g are tag-frame numbers (no floor-frame comparison between sessions, so no
        u_khung_giua_session); e_bc is the largest over the camera centres of the places and of --session-cham (the
        sessions of the scored runs); the file lists them in 'session' and 'e_bc_theo_session' (cam_tam_m null when
        more than one). Limit: the readers do not check that the scoring session is in 'session'.

Reader functions (data contract F3 2026-11-19 section 2.C, CAM / HIEUCHUAN / NHACTAM):
  load_a6(path)                                     the file, schema checked, its sha256 in '_sha256'
  ap_dung(a6, tam_id) -> (trang_thai, ly_do)        'dat' only for the plate of the file with A6 dat, else
                                                    'chua_ket_luan'
  delta_psi_a6(a6, tag_xy_m[N,2], tag_psi_rad[N], plumb_xy_m, zc_m) -> (dpsi_rad[N], ok[N], why[N])
                                                    correction [rad] = [R(psi) g] x (t - C) / (2 (H - hbar)), to be
                                                    SUBTRACTED from the car_pose heading; C, Z_c must be the set-up of
                                                    the file (else A6Error, a ValueError); ok false (dpsi NaN) when a
                                                    corner may leave rho_max (centre + side / sqrt 2) or A6 not dat
  a6du_rad(a6, rho_m, luot=None, dai=None)          residual bound eps_g rho / (2 (H - hbar)) + e_bc; with the run and
                                                    the acceleration band(s) of the frame U_chuyen.dong of that band
                                                    (thieu band -> None); without them the largest over runs / bands
                                                    (any thieu band -> None)
  u_psi_hai_dau_rad(a6, kc, p1_xy_m, p2_xy_m, tam_id, luot=None, dai=None)
                                                    bound of the camera heading difference of two ends (HIEUCHUAN
                                                    u_psi_cuc): A3(p1) + A3(p2) + A6du(p1) + A6du(p2), before the margin.
                                                    A3 by position from KIEMCHEO kiem_cheo.json (c7_kiem_cheo_v7, S7):
                                                    an end inside the checked start disc (MIEN_XP_M about xuat_phat[k])
                                                    of place k takes A3_tuong_doi.A3_xp_deg[k] when every flag of k is
                                                    off (A3_tuong_doi mot_phuong / ngoai_suy / cap_hong, P1_tuong_doi
                                                    theo_vung[k]); a flag on, or an end outside every checked disc ->
                                                    thieu (A6C1-10, CODE 20/11). Both ends at the same place with A3
                                                    per H32 case: the joint replay of design 2b per case (CODE 20/11
                                                    changed decision, A6C1-3); else A3 and A6du added as two separate
                                                    bounds (fallback, larger side); named in 'ghep'.
Every number before the margin 1.25 (C8B applies it once, in c8b_khoa_v5). Constants marked [DE XUAT] are proposals.

  python tools/c8b_a6.py do --frames E/frames.csv --session session.json --f15 meo_E_goc.json --cho goc \
      --tam-id T --lan-lap-id L --luot-id-E E_RUN --out a6_goc.json
  python tools/c8b_a6.py gop --cho-file a6_goc.json --cho-file a6_K1.json --tam-id T --tam-cal-id T \
      [--lan-lap-cal-id L] [--lap lap.json] [--dong dong.json] [--giu-ngoai o.json] [--u-h-chung uh.json] \
      [--khoa-truoc khoa.json] [--kiem-imu kiem_imu_goc.json --kiem-imu kiem_imu_K1.json] --out c8b_a6_v2.json
  python tools/c8b_a6.py kiem_imu --frames E2/frames.csv --session session.json --f15 meo_E_2.json --cho goc \
      --log pi/E2/log.csv --tam-id T --luot-id-E E_RUN_2 --khoa-truoc khoa.json --out kiem_imu_goc.json
  python tools/c8b_a6.py goc --track track/RUN --session session.json --cua-so F2/cua_so_v1.jsonl --luot-id RUN \
      --out goc_RUN.json
  python tools/c8b_a6.py dong --goc goc_RUN1.json --goc goc_RUN2.json --tam-id T --out dong.json
  python tools/c8b_a6.py dong_huong --luot goc_RUN1.json pi/RUN1.csv gio_so_bo_RUN1.json --luot ... --tam-id T \
      --khoa-truoc khoa.json --out dong_huong.json
Last line: #ket_qua,ma=0 written / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c8_u_imu as cu                                                    # noqa: E402

SCHEMA = 'c8b_a6_v2'
SCHEMA_CHO = 'c8b_a6_cho_v3'                               # v3 22/11: g_E = the edge slope gk of fit_gk (MO_HINH)
MO_HINH = 'c8b_a6_canh_v1'                                 # four slopes: gk (edges, heading) and gm (stretch)
SCHEMA_O = 'c8b_a6_o_v1'                                   # cells of the camera-IMU check (design 2b, 7c)
SCHEMA_LAP = 'c8b_a6_lap_v1'                               # U_chuyen.lap source (design 2c)
SCHEMA_DONG = 'c8b_a6_dong_v1'                             # U_chuyen.dong source (design 2c, V5-02)
SCHEMA_DONG_H = 'c8b_a6_dong_v2'                           # 22/11 night: the same from the heading channel of the runs
DONG_KENH_H = 'huong_imu'                                  # camera heading - IMU yaw (game RV)
CHIA_TV = 'trung_vi_so_khung'                              # half bands split at the median acceleration of the frames
SCHEMA_GIO = 'c8b_gio_v1'                                  # preliminary clock of a run (c8b_gio): log = camera + D
DONG_H_MIN_RUNS = 3                                        # a jackknife over runs needs three (as se_jackknife: nb >= 3)
SCHEMA_UH = 'c8b_u_h_chung_v1'                             # u_h_chung source (design 3, video multi-height)
SCHEMA_IMU = 'c8b_a6_kiem_imu_v1'                          # held-out camera-IMU check of gk on an E clip (22/11)
SCHEMA_GOC = 'c8b_a6_goc_v1'                               # lifted corners of the frames of a run (for dong)
F15_FULL = 'c7_meo_ma_ao_E_v1'
F15_DIAG = 'c7_meo_ma_ao_E_v1_chan_doan'                   # KIEMCHEO 02dea17 (KCF1-1): never a usable bound
KC_VERSION = 'c7_kiem_cheo_v7'                             # kiem_cheo.json with S7 (A3 by position)
CHO = ('goc', 'K1')
S_PATTERN = np.array([1.0, -1.0, 1.0, -1.0])
# first mounting of the roof plate, caliper heights [mm] (Liam 19/11, 05_DATA/results/C8/2026-11-18_run01/
# nhat_ky_buoi_CAM.md line 27), ArUco order 0..3 = front-left, front-right, back-right, back-left; side 144.5 mm
H4_CAL_MM = (124.3, 123.3, 123.1, 126.0)
SO_TAY = '05_DATA/results/C8/2026-11-18_run01/nhat_ky_buoi_CAM.md:27'
S_CAL_M = 0.1445
E_CAL_M = 0.0001                                           # caliper error per corner (design 2c)
RHO_MAX_M = 1.6                                            # [DE XUAT] design 2a: every corner within it
NGHIENG_MAX_RAD = math.radians(1.5)                        # [DE XUAT] of design 2a; since 22/11 NOT a limit of gop
                                                           # (Liam at CODE_C8B 22/11, box: "Mien theo du lieu
                                                           # (Recommended)"): the tilt domain is the region of
                                                           # the converged e_bc grid; kept as a reported number
V_MAX_M = 0.001                                            # [DE XUAT] design 3 (twice the first plate's 0.475 mm)
OMEGA_MAX_RAD_S = 1.0                                      # [DE XUAT] L133 (1): frames used for the fit with alpha
SG_HALF = 5                                                # [DE XUAT] omega: quadratic fit over +-5 frames (A6C1-12)
GAP_FACTOR = 1.5                                           # [DE XUAT] a step > 1.5 x median frame step is a gap
N_MIN = 200                                                # [DE XUAT] frames used at least
N_BLOCK = 10                                               # [DE XUAT] jackknife blocks in time
COND_MAX = 1e8                                             # [DE XUAT] condition number of the column-scaled normal
                                                           # matrix of (g, v, sigma, alpha); above: not identified
ALPHA_SE_MAX = 1e-5                                        # [DE XUAT] alpha [s/px] identified only with SE <= 10 us/px
                                                           # (twice the largest shutter seen at 3b, 4.53 us/px, design
                                                           # v6 2: above it the data cannot tell alpha from 0; a clip
                                                           # at 0.3 rad/s, 0.1 mm noise, 900 frames gives 1.6 us/px)
RES_FLOOR_M = 1e-5                                         # [DE XUAT] corner noise floor for that SE (0.01 mm)
CONV_M = 1e-10                                             # convergence: largest displacement of a step [m]
FIT_ITERS = 40
EBC_TOL = 0.10                                             # design 2a: refine until the maximum changes < 10 %
EBC_K0, EBC_KMAX = 4, 16                                   # [DE XUAT] grid levels 4, 8, 16 (A6C2-9: every dimension
                                                           # refined from k = 4); not converged -> thieu
LAP_HUONG_MAX_DEG = 5.0                                    # [DE XUAT] design 2c: same image footprint
LAP_VUNG_M = 0.25                                          # [DE XUAT]
O_N_ID_MIN = 10                                            # [DE XUAT] design 7c: frame ids per cell
TOL_SETUP = 1e-9
# L125 (Liam 19/11, 03_CODE/CLAUDE.md "Phep do va viec giao Liam"): the plumb point and Z_c of 16/11 for EVERY session
PLUMB_L125 = (-0.4802, -0.0329)
ZC_L125 = 2.632
TOL_L125 = 1e-6                                            # the session numbers are these constants (float rounding)
SU_THEO_SESSION = ('session_sha256', 'cam_tam_m')                # da session (A): set-up fields of the camera
U_H_HANG_M = 0.0001                                        # design 3: the 0.1 mm term of u_h_chung
HEX64 = re.compile(r'^[0-9a-f]{64}$')
GIOI_HAN = [
    '22/11 dem: U_chuyen.dong tu file c8b_a6_dong_v2 la chi so cua kenh huong camera - IMU tren cac luot khac (bo luot '
    'dang cham): gom ca sai huong theo vi tri cua ban do (A3) va sai IMU ngoai mo hinh, khong tach; hai nua dai chia o '
    'trung vi gia toc, nen hieu ung chi co o gia toc rat cao bi pha loang; phep kiem quyet cho yaw la o D2 (T14)',
    '22/11: g_b la do doc doc theo canh ma gk (mo hinh c8b_a6_canh_v1); mat ma khong phang (gk khac do doc gian gm), co '
    'che vat ly chua xac dinh (mat cong hay bo noi cua tam); e_bc, mien nghieng, h_goc van tinh theo mat phang do doc gk',
    '22/11: g_cal (thuoc kep, lan dat dau buoi 1) chi tham khao; kiem cheo la kiem_imu (camera - IMU) tren clip E giu '
    'ngoai theo clip; cac clip kiem da duoc xem luc ra nguyen nhan (22/11) truoc khi khoa: khong con sach cho viec '
    'chon mo hinh, luot dich thi chua xem',
    'A6C1-14: sai che do chung cua ban do (F15) va do nghieng lan nhau hoan toan; SE_E, U_meo_E khong bat duoc phan '
    'do; vi the moi co kiem cheo g_cal (so tay) va giu ngoai camera-IMU',
    'v khop phu khong xac nhan venh vat ly; |v| <= v_max la gia dinh mien (thiet ke 3)',
    'A6C1-3: replay chung 2b theo ca H32 chi khi hai dau cung mot cho va moi ca co A3 (KIEMCHEO 692462f, A3 khong '
    'dau: khong triet dau A3-A6 trong mot ca); khong thi cong hai bao rieng (du phong, lon hon)',
    'A6C1-12: hang/cot man trap tu camera session (chieu nguoc), chua kiem voi cot pixel goc cua CAM (cho CAM xuat)',
    'U_cal: venh nhua tach bang tham so v trong phep giai 4 goc (mode s_i truc giao do doc), khong cong them',
    'A6C1-7 (CODE 20/11): lan_lap_id cua clip E khai o CLI do (F8 tinh tu g_b cua chinh file A6 nay: vong phu thuoc); '
    'c8b_khoa_v5 doi chieu voi lan_lap_v1 (F8) theo luot_id_E',
    'A6C1-2 (P2 theo L138): clip E, cua so E chi buoc co sha nguon track, cua_so_v1 trong F15; chua doi chieu noi dung',
]


class A6Error(ValueError):
    """Refused input (contract F3: a ValueError)."""


def body(side):
    h = side / 2.0
    return np.array([[h, h], [h, -h], [-h, -h], [-h, h]])


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _wrap(a):
    return np.remainder(np.asarray(a, float) + math.pi, 2 * math.pi) - math.pi


def _num(x, what, lo=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise A6Error(f'{what}: {x!r} is not a finite number')
    if lo is not None and x < lo:
        raise A6Error(f'{what}: {x!r} < {lo}')
    return float(x)


def _vec(x, n, what, lo=None):
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise A6Error(f'{what}: {x!r} is not {n} numbers')
    return np.array([_num(v, f'{what}[{i}]', lo) for i, v in enumerate(x)])


def content_sha(d):
    """sha256 of the content without the key 'sha256' (KIEMCHEO _with_sha: json sort_keys, ensure_ascii False)."""
    return hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != 'sha256'}, ensure_ascii=False,
                                     sort_keys=True).encode()).hexdigest()


def with_sha(d):
    d['sha256'] = content_sha(d)
    return d


def _check_sha(d, what):
    if not isinstance(d, dict) or not isinstance(d.get('sha256'), str) or d['sha256'] != content_sha(d):
        raise A6Error(f'{what}: content sha256 missing or not matching (file edited)')


def g_cal(h4_mm=H4_CAL_MM, side=S_CAL_M):
    """Tilt (gx, gy), warp v and mean height of the plane through four corner heights (exact for 4 corners)."""
    r = body(side)
    A = np.column_stack([np.ones(4), r[:, 0], r[:, 1], S_PATTERN])
    co = np.linalg.solve(A, np.asarray(h4_mm, float) * 1e-3)
    return np.array(co[1:3]), float(co[3]), float(co[0])


def u_cal(side=S_CAL_M, e=E_CAL_M):
    """Per component 2 e / s (two sides of opposite sign, design 2c), and the vector norm (the one convention of
    design 2c, A6C1-13)."""
    c = 2.0 * e / side
    return c, c * math.sqrt(2.0)


# ── the lift and the model ───────────────────────────────────────────────────────────────────────

def lifted_corners(T, psi, g, v, C, H, hbar, side, sigma=0.0):
    """Exact lifted corners P (4, 2) of car_pose for a tag at T, heading psi, tilt g, warp v, camera centre over the
    plumb point at H: P = C + (X - C)(H - hbar)/(H - h)."""
    r = body(side)
    X = np.asarray(T, float) + (1.0 + sigma) * r @ _rot(psi).T
    h = hbar + r @ np.asarray(g, float) + v * S_PATTERN
    return np.asarray(C, float) + (X - np.asarray(C, float)) * ((H - hbar) / (H - h))[:, None]


def kabsch_psi(P, side):
    """Heading of the lifted corners as car_pose takes it (c7_measure.rigid_fit, no scale)."""
    r = body(side)
    a, b = r - r.mean(0), P - P.mean(0)
    hm = a.T @ b
    return math.atan2(hm[0, 1] - hm[1, 0], hm[0, 0] + hm[1, 1])


def dpsi_model(g, t_xy, psi, C, H, hbar):
    """design 2a: delta psi_A6 = [R(psi) g] x (t - C) / (2 (H - hbar)) (2-D cross product); vectorised over psi, t."""
    psi = np.asarray(psi, float)
    t = np.asarray(t_xy, float)
    g = np.asarray(g, float)
    c, s = np.cos(psi), np.sin(psi)
    wx, wy = c * g[0] - s * g[1], s * g[0] + c * g[1]
    d = t - np.asarray(C, float)
    out = (wx * d[..., 1] - wy * d[..., 0]) / (2.0 * (H - hbar))
    return float(out) if np.ndim(out) == 0 else out


# ── fit of g on the E frames ─────────────────────────────────────────────────────────────────────

def _cond_scaled(M):
    """Condition number of M after scaling rows and columns by sqrt(diag) (A6C1-11); inf when a column is empty."""
    D = np.sqrt(np.abs(np.diag(M)))
    if not np.all(np.isfinite(M)) or np.any(D == 0):
        return math.inf
    return float(np.linalg.cond(M / np.outer(D, D)))


def fit_g(P, C, H, hbar, side, iters=FIT_ITERS, fields=()):
    """g (2), sigma, per-frame (T, psi) from lifted corners P (n, 4, 2): Gauss-Newton on
    P_i = T + R(psi)(1 + sigma) r_i + (g.r_i + v s_i) d_i + sum_k alpha_k F_k,i, d_i = (P_i - C)/(H - hbar), the
    per-frame pose eliminated in each step (Schur complement, vectorised). v, sigma and alpha_k (fields F_k (n, 4, 2):
    the rolling-shutter displacement per unit alpha, L133) are nuisance parameters, alpha not bounded in sign. Returns
    also the normal matrix of (g, v, sigma, alpha), hoi_tu (the last step moved every corner by < CONV_M), the
    number of steps and the scaled condition number (A6C1-11)."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        raise A6Error('fewer than 3 frames')
    r = body(side)
    C = np.asarray(C, float)
    d = (P - C) / (H - hbar)                                          # (n, 4, 2), fixed: built from the measured P
    F = [np.asarray(f, float) for f in fields]
    K = len(F)
    g, sig, vv, al = np.zeros(2), 0.0, 0.0, np.zeros(K)
    psi = np.array([kabsch_psi(P[f], side) for f in range(n)])
    T = P.mean(1)
    Bg = np.stack([(r[:, 0][:, None] * d).reshape(n, 8), (r[:, 1][:, None] * d).reshape(n, 8),
                   (S_PATTERN[:, None] * d).reshape(n, 8)] + [x.reshape(n, 8) for x in F], axis=2)   # (n, 8, 3 + K)
    ix = [0, 1, 2] + [4 + k for k in range(K)]
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    M, ok, it = None, False, 0
    for it in range(1, iters + 1):
        Rs = np.stack([np.cos(psi), -np.sin(psi), np.sin(psi), np.cos(psi)], axis=1).reshape(n, 2, 2)
        Rr = np.einsum('ij,fkj->fik', r, Rs)                             # (n, 4, 2): R(psi_f) r_i
        mod = T[:, None, :] + (1.0 + sig) * Rr + (r @ g + vv * S_PATTERN)[None, :, None] * d
        for k in range(K):
            mod = mod + al[k] * F[k]
        e = (P - mod).reshape(n, 8)
        A = np.zeros((n, 8, 3))
        A[:, 0::2, 0], A[:, 1::2, 1] = 1.0, 1.0
        A[:, :, 2] = ((1.0 + sig) * Rr @ J).reshape(n, 8)               # d(R r)/dpsi = (-y, x)
        B = np.zeros((n, 8, 4 + K))
        B[:, :, ix] = Bg
        B[:, :, 3] = Rr.reshape(n, 8)
        try:
            Ai = np.linalg.inv(np.einsum('fij,fik->fjk', A, A))          # (n, 3, 3)
            Q = np.eye(8)[None] - np.einsum('fij,fjk,flk->fil', A, Ai, A)
            M = np.einsum('fij,fik,fkl->jl', B, Q, B)
            b = np.einsum('fij,fik,fk->j', B, Q, e)
            dgam = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            # A6C2-8: a singular step keeps every key (se_alpha inf, M None): the caller reports thieu, not a crash
            return dict(g=g, sigma=float(sig), v_phu=float(vv), alpha=[float(x) for x in al], T=T, psi=psi, M=None,
                        res_rms_m=math.nan, hoi_tu=False, so_vong=it, cond=math.inf, se_alpha=[math.inf] * K,
                        sigma_toa_do_m=math.nan, bac_tu_do=None)
        dn = np.einsum('fij,fkj,fk->fi', Ai, A, e - np.einsum('fij,j->fi', B, dgam))
        T = T + dn[:, :2]
        psi = psi + dn[:, 2]
        g, vv, sig, al = g + dgam[:2], vv + dgam[2], sig + dgam[3], al + dgam[4:]
        # A6C2-10: the step is the largest TOTAL displacement of a corner (pose and common parameters together)
        mv = (np.einsum('fij,fj->fi', A, dn) + np.einsum('fij,j->fi', B, dgam)).reshape(n, 4, 2)
        step = float(np.max(np.linalg.norm(mv, axis=2)))
        if step < CONV_M:
            ok = True
            break
    Rs = np.stack([np.cos(psi), -np.sin(psi), np.sin(psi), np.cos(psi)], axis=1).reshape(n, 2, 2)
    res = P - T[:, None, :] - (1.0 + sig) * np.einsum('ij,fkj->fik', r, Rs) - (r @ g + vv * S_PATTERN)[None, :, None] * d
    for k in range(K):
        res = res - al[k] * F[k]
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=2))))
    # A6C2-10: per-coordinate noise from the residual sum over the degrees of freedom: 8 n coordinates - 3 n pose
    # parameters - (4 + K) common parameters (g 2, v, sigma, alpha K)
    dof = 8 * n - 3 * n - (4 + K)
    s_xy = float(math.sqrt(np.sum(res ** 2) / dof)) if dof > 0 else math.inf
    se_al = [math.inf] * K
    if M is not None and K:
        try:
            dg = np.diag(np.linalg.inv(M))
            se_al = [float(math.sqrt(x) * max(s_xy, RES_FLOOR_M)) if x > 0 and math.isfinite(x) else math.inf
                     for x in dg[4:]]
        except np.linalg.LinAlgError:
            pass
    return dict(g=g, sigma=float(sig), v_phu=float(vv), alpha=[float(x) for x in al], T=T, psi=psi, M=M,
                res_rms_m=rms, hoi_tu=ok, so_vong=it, cond=_cond_scaled(M) if M is not None else math.inf,
                se_alpha=se_al, sigma_toa_do_m=s_xy, bac_tu_do=dof)


def fit_gk(P, C, H, hbar, side, iters=FIT_ITERS, fields=(), bien_dang=('ma', 'san')):
    """The edge-slope model MO_HINH (CODE_C8B 22/11; Liam at CODE_C8B 22/11: "Sua mo hinh (Recommended)"; analysis
    05_DATA/results/C8/2026-11-20_run02/c8b/T7_a6_ra_mau_thuan/). The corners of car_pose are the intersections of the
    four fitted edge lines, and a line only feels the displacement normal to itself, so the parallax of a tag surface
    that is not a plane moves the corners, in the TAG frame (u = R(psi)^T d, d = (P - C) / (H - hbar)), by
        dx = u_x (gxm x + gyk y),   dy = u_y (gxk x + gym y)     (+ v s_i u, the warp nuisance of fit_g)
    with FOUR slopes: gxm, gym from the mean heights of the two opposite edges (stretch) and gxk, gyk from the slope
    ALONG the edges (shear). A plane has gk = gm = g (the model of fit_g). The Kabsch heading of the four corners moves
    by (gxk u_y - gyk u_x) / 2 = [R(psi) gk] x d / 2: the heading error of design 2a with g = gk, so gk is the slope
    A6 needs and gm is a nuisance. On the E clips of session 2 the two differ (K1: gk (-0.018, 0.004), gm (-0.005,
    0.005)) and the camera-IMU heading follows gk. Same Gauss-Newton as fit_g with the per-frame pose eliminated;
    the columns of the slopes are rebuilt from the current heading at every step (their change with the heading is
    left out of the pose Jacobian: second order, the step still converges to the exact residual).
    bien_dang: constant traceless symmetric strains fitted as nuisance (2 numbers each): 'ma' fixed in the TAG frame
    (the print of the tag is not a square, the edge detector's bias by edge direction) and 'san' fixed in the FLOOR
    frame (the map residual at the place of an E clip: only for frames at one place). Both are orthogonal to the
    slopes only when the headings cover whole turns evenly; an E clip does not (at goc of session 2 leaving them out
    moves gk by 0.0025), so they are fitted. A clip without rotation cannot tell them apart: the fit is then ill
    conditioned and refused (fit_ok).
    Returns g = gk, g_gian = gm, v_phu, sigma, bien_dang {ma, san}, alpha, the normal matrix M of (gxk, gyk, gxm, gym,
    v, sigma, strains, alpha) with i_alpha = 6 + number of strain parameters, and the keys of fit_g."""
    bd = tuple(bien_dang or ())
    if any(b not in ('ma', 'san') for b in bd) or len(set(bd)) != len(bd):
        raise A6Error(f'bien_dang {bien_dang!r}: ma | san')
    NS = 2 * len(bd)
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        raise A6Error('fewer than 3 frames')
    r = body(side)
    C = np.asarray(C, float)
    d = (P - C) / (H - hbar)                                          # (n, 4, 2), fixed: built from the measured P
    F = [np.asarray(f, float) for f in fields]
    K = len(F)
    IA = 6 + NS                                                       # first alpha index
    NP = IA + K
    gam = np.zeros(NP)
    psi = np.array([kabsch_psi(P[f], side) for f in range(n)])
    T = P.mean(1)
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    x, y = r[:, 0][None, :], r[:, 1][None, :]
    xb, yb = np.broadcast_to(x, (n, 4)), np.broadcast_to(y, (n, 4))
    Bv = S_PATTERN[None, :, None] * d
    M, ok, it = None, False, 0

    def cols_of(psi_):
        c, s = np.cos(psi_)[:, None], np.sin(psi_)[:, None]
        Rr_ = np.stack([c * x - s * y, s * x + c * y], axis=2)                       # (n, 4, 2): R(psi) r_i
        ux, uy = c * d[:, :, 0] + s * d[:, :, 1], -s * d[:, :, 0] + c * d[:, :, 1]   # u = R^T d per corner

        def rot(a, b):                                                # tag-frame vector (a, b) -> floor
            return np.stack([c * a - s * b, s * a + c * b], axis=2)
        z = np.zeros_like(ux)
        st = []
        for b_ in bd:
            if b_ == 'ma':                                            # tag frame: [[e1, e2], [e2, -e1]] r
                st += [rot(xb, -yb), rot(yb, xb)]
            else:                                                     # floor frame: [[f1, f2], [f2, -f1]] (R r)
                st += [np.stack([Rr_[:, :, 0], -Rr_[:, :, 1]], axis=2), np.stack([Rr_[:, :, 1], Rr_[:, :, 0]], axis=2)]
        return Rr_, [rot(z, uy * x), rot(ux * y, z), rot(ux * x, z), rot(z, uy * y), Bv, Rr_] + st + F

    def nopose(Rr_, cs):
        mod = T[:, None, :] + Rr_
        for j in range(NP):
            mod = mod + gam[j] * cs[j]
        return mod

    for it in range(1, iters + 1):
        Rr, cs = cols_of(psi)
        e = (P - nopose(Rr, cs)).reshape(n, 8)
        A = np.zeros((n, 8, 3))
        A[:, 0::2, 0], A[:, 1::2, 1] = 1.0, 1.0
        A[:, :, 2] = ((1.0 + gam[5]) * Rr @ J).reshape(n, 8)
        B = np.stack([q.reshape(n, 8) for q in cs], axis=2)             # (n, 8, NP)
        try:
            Ai = np.linalg.inv(np.einsum('fij,fik->fjk', A, A))
            Q = np.eye(8)[None] - np.einsum('fij,fjk,flk->fil', A, Ai, A)
            M = np.einsum('fij,fik,fkl->jl', B, Q, B)
            b = np.einsum('fij,fik,fk->j', B, Q, e)
            dgam = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            return dict(g=gam[:2].copy(), g_gian=gam[2:4].copy(), sigma=float(gam[5]), v_phu=float(gam[4]),
                        alpha=[float(v_) for v_ in gam[IA:]], T=T, psi=psi, M=None, i_alpha=IA, res_rms_m=math.nan,
                        bien_dang={b_: [float(v_) for v_ in gam[6 + 2 * j:8 + 2 * j]] for j, b_ in enumerate(bd)},
                        hoi_tu=False, so_vong=it, cond=math.inf, se_alpha=[math.inf] * K, sigma_toa_do_m=math.nan,
                        bac_tu_do=None)
        dn = np.einsum('fij,fkj,fk->fi', Ai, A, e - np.einsum('fij,j->fi', B, dgam))
        T = T + dn[:, :2]
        psi = psi + dn[:, 2]
        gam = gam + dgam
        mv = (np.einsum('fij,fj->fi', A, dn) + np.einsum('fij,j->fi', B, dgam)).reshape(n, 4, 2)
        if float(np.max(np.linalg.norm(mv, axis=2))) < CONV_M:
            ok = True
            break
    Rr, cs = cols_of(psi)
    res = P - nopose(Rr, cs)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=2))))
    dof = 8 * n - 3 * n - NP
    s_xy = float(math.sqrt(np.sum(res ** 2) / dof)) if dof > 0 else math.inf
    se_al = [math.inf] * K
    if M is not None and K:
        try:
            dg = np.diag(np.linalg.inv(M))
            se_al = [float(math.sqrt(v_) * max(s_xy, RES_FLOOR_M)) if v_ > 0 and math.isfinite(v_) else math.inf
                     for v_ in dg[IA:]]
        except np.linalg.LinAlgError:
            pass
    return dict(g=gam[:2].copy(), g_gian=gam[2:4].copy(), sigma=float(gam[5]), v_phu=float(gam[4]),
                alpha=[float(v_) for v_ in gam[IA:]], T=T, psi=psi, M=M, i_alpha=IA, res_rms_m=rms, hoi_tu=ok,
                bien_dang={b_: [float(v_) for v_ in gam[6 + 2 * j:8 + 2 * j]] for j, b_ in enumerate(bd)},
                so_vong=it, cond=_cond_scaled(M) if M is not None else math.inf, se_alpha=se_al, sigma_toa_do_m=s_xy,
                bac_tu_do=dof)


def fit_ok(f):
    """A usable fit: converged, the column-scaled normal matrix well conditioned and every alpha identified (SE <=
    ALPHA_SE_MAX with the residual floored at RES_FLOOR_M: column scaling alone hides an empty shutter column, e.g.
    frames standing still) (A6C1-11)."""
    return bool(f['hoi_tu']) and math.isfinite(f['cond']) and f['cond'] <= COND_MAX         and all(x <= ALPHA_SE_MAX for x in f.get('se_alpha') or [])


def corr_alpha_g(M, i_alpha=4):
    """Largest |correlation| between a component of g and an alpha in the least-squares covariance (M^-1): the price
    of fitting alpha with g (L133 (2)). i_alpha: the first alpha index of M (4 for fit_g, 6 for fit_gk)."""
    if M is None or M.shape[0] <= i_alpha:
        return None
    try:
        cov = np.linalg.inv(M)
    except np.linalg.LinAlgError:                                        # A6C2-8: singular -> no number
        return None
    dg = np.diag(cov)
    if not np.all(np.isfinite(cov)) or np.any(dg <= 0):
        return None
    sd = np.sqrt(dg)
    return float(max(abs(cov[i, j] / (sd[i] * sd[j])) for i in (0, 1) for j in range(i_alpha, M.shape[0])))


def se_jackknife(P, C, H, hbar, side, nb=N_BLOCK, fields=(), fit=None):
    """Block jackknife SE of g over nb blocks of consecutive frames; (SE, every leave-one-block fit usable). fit: the
    fit function (fit_g by default); with fit_gk the SE has four numbers: g (gk) then g_gian (gm)."""
    fit = fit or fit_g
    n = len(P)
    nb = min(nb, n // 3)
    if nb < 3:
        raise A6Error('too few frames for the jackknife')
    edges = np.linspace(0, n, nb + 1).astype(int)
    gs, good = [], True
    for k in range(nb):
        keep = np.r_[0:edges[k], edges[k + 1]:n]
        f = fit(P[keep], C, H, hbar, side, fields=[x[keep] for x in fields])
        good = good and fit_ok(f)
        gs.append(np.concatenate([f['g'], f['g_gian']]) if 'g_gian' in f else f['g'])
    gs = np.array(gs)
    return np.sqrt((nb - 1) / nb * np.sum((gs - gs.mean(0)) ** 2, axis=0)), good


# ── inputs ───────────────────────────────────────────────────────────────────────────────────────

CORNER_COLS = tuple(f'c{i}_{a}' for i in range(4) for a in ('x', 'y'))


def read_e_frames(path):
    """idx, t (s), lifted corners P (n, 4, 2) of the valid frames of a CAM frames.csv with the 8 corner columns
    c0_x .. c3_y (CAM design v2 section 7b, after the parallax, ArUco order), and {idx: t} of every row."""
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in CORNER_COLS + ('idx', 't_s', 'valid')):
        raise A6Error(f'{path}: no corner columns {CORNER_COLS} (CAM frames.csv design v2 7b) or idx/t_s/valid')
    t_all, keep = {}, []
    for r in rows:
        k, tt = int(r['idx']), float(r['t_s'])
        if k in t_all or not math.isfinite(tt):
            raise A6Error(f'{path}: idx {k} repeated or t_s not finite')
        t_all[k] = tt
        if r['valid'] not in ('1', 'True', 'true') or any(r[c] in ('', None) for c in CORNER_COLS):
            continue
        keep.append((k, tt, [float(r[c]) for c in CORNER_COLS]))
    if not keep:
        raise A6Error(f'{path}: no valid frame with its corners')
    idx = np.array([k[0] for k in keep])
    t = np.array([k[1] for k in keep])
    P = np.array([k[2] for k in keep]).reshape(-1, 4, 2)
    if not np.all(np.isfinite(P)) or np.any(np.diff(t) <= 0) or np.any(np.diff(idx) <= 0):
        raise A6Error(f'{path}: corners not finite or idx / times not increasing')
    return idx, t, P, t_all


def read_frame_times(path):
    """{idx: t_s} of every row of a frames.csv (no corner columns needed)."""
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in ('idx', 't_s')):
        raise A6Error(f'{path}: no idx/t_s columns')
    t_all = {}
    for r in rows:
        k, tt = int(r['idx']), float(r['t_s'])
        if k in t_all or not math.isfinite(tt):
            raise A6Error(f'{path}: idx {k} repeated or t_s not finite')
        t_all[k] = tt
    return t_all


def e_frames_from_f15(f15):
    """idx, t, lifted corners P (n, 4, 2) of the E frames from the F15 of KIEMCHEO (khung[].goc_m: the 4 real corners
    of the car tag through the PER-FRAME scoring camera of the track, lifted as car_pose; replayed against frames.csv
    within TOL_REPLAY_M by meo_E). Used when the frames.csv has no corner columns (CAM did not write design v2 7b; the
    frames of session 2 have none; CODE_C8B 22/11). The session camera alone is 3-6 mm off the scoring centre there
    (per-frame mark shift), so the corners are not rebuilt here."""
    g = f15.get('goc') or {}
    if not g:
        raise A6Error('F15 without goc_m: no corners for the E frames')
    ids = sorted(g)
    t = np.array([f15['t'][k] for k in ids])
    P = np.array([g[k] for k in ids])
    if np.any(np.diff(t) <= 0):
        raise A6Error('F15 frame times not increasing')
    return np.array(ids), t, P


def read_session(path):
    """car_pose set-up of the session: C (plumb), H = Z_c = tag_h / kappa, hbar = tag_h, side; every number finite and
    in range (A6C1-8)."""
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    try:
        k, h, side = float(s['kappa']), float(s['tag_h_m']), float(s['car_side_m'])
        C = np.array([float(v) for v in s['plumb_xy_m']])
    except (KeyError, TypeError, ValueError) as exc:
        raise A6Error(f'{path}: kappa / tag_h_m / car_side_m / plumb_xy_m ({exc})')
    if not (all(math.isfinite(x) for x in (k, h, side)) and 0 < k < 1 and h > 0 and side > 0 and C.shape == (2,)
            and np.all(np.isfinite(C))):
        raise A6Error(f'{path}: car_pose numbers not finite or out of range')
    _check_l125(C, h / k, path)
    return dict(C=C, H=h / k, hbar=h, side=side, kappa=k, sha256=cu.sha256(path), file=os.path.abspath(path),
                camera=s.get('camera'))


def _check_l125(C, zc, what):
    """A6C2-6: the set-up is anchored to the plumb point and Z_c of 16/11 (L125), not only consistent between files."""
    C = np.asarray(C, float)
    if C.shape != (2,) or not np.all(np.isfinite(C)) or not math.isfinite(float(zc)) \
            or not np.max(np.abs(C - np.array(PLUMB_L125))) <= TOL_L125 or not abs(float(zc) - ZC_L125) <= TOL_L125:
        raise A6Error(f'{what}: plumb {C.tolist()} / Z_c {zc} are not the 16/11 set-up {list(PLUMB_L125)}, {ZC_L125} '
                      f'(L125)')


def camera_of(ses):
    import c7_clip_probe as cp
    c = ses.get('camera')
    if not isinstance(c, dict):
        raise A6Error('session without camera: the rolling-shutter fields and e_bc need it')
    return cp.Camera(**{k: c[k] for k in ('model', 'f', 'cx', 'cy', 'k', 'rvec', 'tvec')})


def read_f15(path, ses, frames_sha, t_all):
    """KIEMCHEO F15 (a) checked against the session (kappa, plumb, Z_c, h), the frames.csv (sha in nguon, frame times),
    the corner order, the E window and the full H32 case list (A6C1-2). Returns the parsed dict (see _f15_obj)."""
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    _check_sha(d, path)
    sch = d.get('schema')
    if sch not in (F15_FULL, F15_DIAG):
        raise A6Error(f'{path}: schema {sch!r}, not {F15_FULL} / {F15_DIAG}')
    if d.get('chieu') != 'held_minus_full' or d.get('bo_H32') is not True or d.get('truoc_le') is not True:
        raise A6Error(f'{path}: chieu held_minus_full, bo_H32 and truoc_le true needed')
    if not str(d.get('thu_tu_goc', '')).startswith('ArUco 0..3'):
        raise A6Error(f'{path}: thu_tu_goc {d.get("thu_tu_goc")!r} is not ArUco 0..3')
    for k, want in (('kappa', ses['kappa']), ('Zc_m', ses['H']), ('h_m', ses['hbar'])):
        if isinstance(d.get(k), bool) or not isinstance(d.get(k), (int, float)) \
                or abs(float(d[k]) - want) > TOL_SETUP * max(1.0, abs(want)):
            raise A6Error(f'{path}: {k} {d.get(k)!r} is not the car_pose set-up of the session ({want})')
    pl = d.get('plumb_xy_m')
    if not isinstance(pl, list) or len(pl) != 2 or np.max(np.abs(np.array(pl, float) - ses['C'])) > TOL_SETUP:
        raise A6Error(f'{path}: plumb_xy_m is not the plumb point of the session')
    src = {x.get('vai_tro'): x for x in d.get('nguon') or [] if isinstance(x, dict)}
    if (src.get('frames') or {}).get('sha256') != frames_sha:
        raise A6Error(f'{path}: nguon frames sha256 is not the frames.csv given (another clip)')
    if (src.get('session') or {}).get('sha256') != ses['sha256']:
        raise A6Error(f'{path}: nguon session sha256 is not the session given')
    if not isinstance(d.get('clip_sha256'), str) or not HEX64.match(d['clip_sha256']):
        raise A6Error(f'{path}: clip_sha256 missing')
    for v in ('track', 'cua_so_v1'):                 # A6C1-2: the E clip and window tied to their sources (sha only)
        if not isinstance((src.get(v) or {}).get('sha256'), str) or not HEX64.match(src[v]['sha256']):
            raise A6Error(f'{path}: nguon {v} sha256 missing (the E window must come from track / cua_so_v1)')
    ca_l = [x for x in ((d.get('ban_do') or {}).get('ca') or []) if isinstance(x, dict)]
    hid = (d.get('ban_do') or {}).get('H32_ids')
    if not isinstance(hid, list) or not hid:
        raise A6Error(f'{path}: ban_do.H32_ids missing')
    bo = [x.get('bo_ids') for x in ca_l]
    if any(not isinstance(b, list) or not b or not set(b) <= set(hid) for b in bo):
        raise A6Error(f'{path}: a case bo_ids empty or not within H32_ids')
    if set(hid) - set().union(*map(set, bo)):
        raise A6Error(f'{path}: H32 ids {sorted(set(hid) - set().union(*map(set, bo)))} left out by no case (A6C1-2)')
    ca = [x.get('ten') for x in ca_l]
    a3_ca = {x.get('ten'): (x.get('A3') or {}).get('quy_max_deg') for x in ca_l if isinstance(x.get('A3'), dict)}
    khung, goc = [], {}
    for k in d.get('khung') or []:
        fid, ts = k.get('frame_id'), k.get('t_s')
        if isinstance(fid, bool) or not isinstance(fid, int) or fid not in t_all:
            raise A6Error(f'{path}: frame_id {fid!r} not a frame of the frames.csv')
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(float(ts)) \
                or not abs(float(ts) - t_all[fid]) <= 1e-6:
            raise A6Error(f'{path}: frame {fid} t_s {ts!r} is not the time of the frames.csv ({t_all[fid]})')
        gm = np.asarray(k.get('goc_m'), float)
        if gm.shape != (4, 2) or not np.all(np.isfinite(gm)):
            raise A6Error(f'{path}: frame {fid} goc_m not 4 x 2 finite')
        goc[fid] = gm
        cases = {}
        for name, v in (k.get('du_m') or {}).items():
            a = np.asarray(v, float)
            if a.shape != (4, 2) or not np.all(np.isfinite(a)):
                raise A6Error(f'{path}: frame {fid} case {name}: du_m not 4 x 2 finite')
            cases[name] = a
        khung.append((fid, float(ts), cases))
    out = _f15_obj(sch, d.get('trang_thai'), d.get('cho'), d.get('mien_E'), ca, khung, d.get('clip_sha256'),
                   d['sha256'], os.path.abspath(path), cu.sha256(path), d.get('ly_do') or [])
    out['A3_ca_deg'] = a3_ca or None             # KIEMCHEO 692462f: A3 per H32 case on the E session (information)
    out['goc'] = goc                              # lifted corners per frame (KIEMCHEO: per-frame scoring camera)
    return out


def _f15_obj(schema, trang_thai, cho, mien_E, ca, khung, clip_sha256, sha_noi_dung, file, sha_file, ly_do):
    """The parsed F15 (tests build it directly): ca = the H32 case names of ban_do.ca; khung [(frame_id, t_s,
    {case: du (4, 2)})]. Checks that do not need the files: E window, unique frames, the case list at every frame."""
    if not isinstance(mien_E, dict):
        raise A6Error('F15 without mien_E (the E window): refused, not every frame (A6C1-2)')
    lo, hi = mien_E.get('sau_E_dau'), mien_E.get('truoc_E_cuoi')
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (lo, hi)) or not lo < hi:
        raise A6Error(f'F15 mien_E {mien_E!r}: integer sau_E_dau < truoc_E_cuoi needed (A6C1-2)')
    if not ca or len(set(ca)) != len(ca) or not all(isinstance(x, str) and x for x in ca):
        raise A6Error('F15 ban_do.ca: no H32 case list (or names repeated)')
    du, t = {}, {}
    for fid, ts, cases in khung:
        if fid in du:
            raise A6Error(f'F15 frame {fid} twice')
        if set(cases) != set(ca):
            raise A6Error(f'F15 frame {fid}: cases {sorted(cases)} are not the H32 list {sorted(ca)} (A6C1-2)')
        du[fid], t[fid] = cases, ts
    if not du:
        raise A6Error('F15: no frame')
    return dict(schema=schema, trang_thai=trang_thai, cho=cho, mien_E=dict(sau_E_dau=lo, truoc_E_cuoi=hi), ca=list(ca),
                du=du, t=t, clip_sha256=clip_sha256, sha256=sha_noi_dung, file=file, sha256_file=sha_file,
                ly_do=list(ly_do))


def omega_of(t, P, side, half=SG_HALF):
    """Turn rate [rad/s] per frame: quadratic fit of the unwrapped Kabsch heading over +-half frames on the real times
    (A6C1-12: the source locked, a smoothed rate rather than a two-frame difference); NaN where the window is not full
    or crosses a gap (a step > GAP_FACTOR x the median step)."""
    t = np.asarray(t, float)
    psi = np.unwrap(np.array([kabsch_psi(p, side) for p in P]))
    n = len(t)
    w = np.full(n, np.nan)
    if n < 2 * half + 1:
        return w
    dt = np.diff(t)
    gap = dt > GAP_FACTOR * np.median(dt)
    for k in range(half, n - half):
        if gap[k - half:k + half].any():
            continue
        c = np.polyfit(t[k - half:k + half + 1] - t[k], psi[k - half:k + half + 1], 2)
        w[k] = c[1]
    return w


def uv_of(P, cam, C, kappa):
    """Pixel (col, row) of every corner (n, 4, 2), from the session camera at the unlifted floor points (car_pose:
    P = p + kappa (C - p)). Fixed per image: the replay keeps them (A6C1-12)."""
    P = np.asarray(P, float)
    C = np.asarray(C, float)
    p = (P - kappa * C) / (1.0 - kappa)
    uv = np.asarray(cam.project(np.column_stack([p.reshape(-1, 2), np.zeros(p.shape[0] * 4)])), float)
    return uv.reshape(P.shape[0], 4, 2)


def shutter_fields(P, t, uv):
    """Rolling-shutter displacement per unit alpha of every corner: (row - mean row of the tag) x corner velocity and
    (col - mean col) x velocity [m per (s/px)]; the velocity from these corners P, the pixels uv of the same images."""
    vel = np.gradient(np.asarray(P, float), np.asarray(t, float), axis=0)
    fr = (uv[:, :, 1] - uv[:, :, 1].mean(1, keepdims=True))[:, :, None] * vel
    fc = (uv[:, :, 0] - uv[:, :, 0].mean(1, keepdims=True))[:, :, None] * vel
    return fr, fc


# ── do: one place ────────────────────────────────────────────────────────────────────────────────

def do_place(cho, idx, t, P, ses, f15, cam, tam_id, lan_lap_id, luot_id_E=None):
    """g_E, SE_E, U_meo_E, s_E of one place; trang_thai 'dat' / 'thieu' with the reasons. luot_id_E: the run id of
    the E clip windows (F2 E_dau / E_cuoi), kept so c8b_khoa_v5 checks lan_lap_id against F8 (A6C1-7, CODE 20/11)."""
    if cho not in CHO:
        raise A6Error(f'cho {cho!r}: goc | K1')
    if not isinstance(tam_id, str) or not tam_id or not isinstance(lan_lap_id, str) or not lan_lap_id:
        raise A6Error('tam_id and lan_lap_id needed (the plate and mounting of the E clip, A6C1-7)')
    if f15 is None:
        raise A6Error('no F15 (a): the E window (mien_E) and U_meo_E come from it; refused (A6C1-2)')
    if cam is None:
        raise A6Error('no session camera: the rolling-shutter fields cannot be built (L133)')
    if f15.get('cho') != cho:
        raise A6Error(f'F15 is for {f15.get("cho")!r}, not {cho!r}')
    why = []
    C, H, hb, side = ses['C'], ses['H'], ses['hbar'], ses['side']
    lo, hi = f15['mien_E']['sau_E_dau'], f15['mien_E']['truoc_E_cuoi']
    sel = (idx > lo) & (idx < hi)                                      # the moving E frames of F15 (E_doan)
    n_e = int(sel.sum())
    if n_e < N_MIN:
        raise A6Error(f'{cho}: {n_e} E frames < {N_MIN}')
    w = omega_of(t, P, side)
    sel &= np.isfinite(w) & (np.abs(np.nan_to_num(w, nan=np.inf)) <= OMEGA_MAX_RAD_S)
    i1 = np.flatnonzero(sel)
    setup = dict(C=list(map(float, C)), H_m=H, h_m=hb, side_m=side, kappa=ses['kappa'], session_sha256=ses['sha256'],
                 cam_tam_m=[float(x) for x in np.asarray(getattr(cam, 'centre', (math.nan,) * 3), float)])
    base = dict(schema=SCHEMA_CHO, mo_hinh=MO_HINH, cho=cho, tam_id=tam_id, lan_lap_id=lan_lap_id, luot_id_E=luot_id_E, n_khung_E=n_e,
                set_up=setup,
                f15=dict(schema=f15['schema'], trang_thai=f15['trang_thai'], file=f15['file'], sha256=f15['sha256'],
                         sha256_file=f15.get('sha256_file'), clip_sha256=f15['clip_sha256'], ca=f15['ca']),
                omega=dict(nguon='kabsch tung khung, khop bac hai +-%d khung tren gio that, khong qua khoang trong'
                           % SG_HALF, omega_max_rad_s=OMEGA_MAX_RAD_S))
    if len(i1) < N_MIN:
        why.append(f'{len(i1)} E frames with |omega| <= {OMEGA_MAX_RAD_S} rad/s (< {N_MIN}): g not taken (L133)')
        return dict(base, g_E=None, SE_E=None, U_meo_E=None, s_E=None, s_E_norm=None, n_khung=int(len(i1)),
                    trang_thai='thieu', ly_do=why)
    miss = [int(k) for k in idx[i1] if int(k) not in f15['du']]
    if miss:
        raise A6Error(f'F15 lacks {len(miss)} frames used (first {miss[:3]})')
    bad_t = [int(k) for k, tt in zip(idx[i1], t[i1]) if abs(f15['t'][int(k)] - tt) > 1e-6]
    if bad_t:
        raise A6Error(f'F15 frame times differ from the frames used (first {bad_t[:3]}): another clip (A6C1-2)')
    Pu = P[i1]
    uv = uv_of(Pu, cam, C, ses['kappa'])
    fl = list(shutter_fields(Pu, t[i1], uv))
    fit = fit_gk(Pu, C, H, hb, side, fields=fl)                         # MO_HINH: g_E = gk (22/11)
    se4, jk_ok = se_jackknife(Pu, C, H, hb, side, fields=fl, fit=fit_gk)
    se, se_gian = se4[:2], se4[2:]
    fit0 = fit_gk(Pu, C, H, hb, side)                                   # L133 (2): the same frames without alpha
    se0, _ = se_jackknife(Pu, C, H, hb, side, fit=fit_gk)
    se0 = se0[:2]
    fit_p = fit_g(Pu, C, H, hb, side, fields=fl)                        # the plane model (one g), information only
    fit_n = fit_gk(Pu, C, H, hb, side, fields=fl, bien_dang=())         # without the strain nuisances, information
    if not fit_ok(fit):
        why.append(f'fit of g not usable: converged {fit["hoi_tu"]}, scaled condition {fit["cond"]:.3g} '
                   f'(<= {COND_MAX:g}), SE alpha {[f"{x:.3g}" for x in fit["se_alpha"]]} s/px (<= {ALPHA_SE_MAX:g}) '
                   f'(A6C1-11)')
    if not jk_ok:
        why.append('a jackknife fit did not converge or is ill conditioned (A6C1-11)')
    g = fit['g']
    uc_comp = u_cal()[0]
    usable = f15['schema'] == F15_FULL and f15['trang_thai'] == 'dat'
    theo_ca, dmax, rep_ok = {}, np.zeros(2), True
    for nm in f15['ca']:
        D = np.array([f15['du'][int(k)][nm] for k in idx[i1]])
        P2 = Pu + D
        fk = fit_gk(P2, C, H, hb, side, fields=list(shutter_fields(P2, t[i1], uv)))
        rep_ok = rep_ok and fit_ok(fk)
        dg = fk['g'] - g
        theo_ca[nm] = [float(x) for x in dg]
        dmax = np.maximum(dmax, np.abs(dg))
    if not rep_ok:
        why.append('a replay fit (F15 case) did not converge or is ill conditioned (A6C1-11)')
    nhan = [bool(dmax[j] <= max(abs(g[j]), uc_comp)) for j in (0, 1)]
    nhan_dang = dict(quy_tac='moi thanh phan j: max_ca |dg_j| <= max(|g_E,j|, U_cal moi thanh phan) [DE XUAT]',
                     dg_max=[float(x) for x in dmax], g_E=[float(x) for x in g], dat=all(nhan))
    if not all(nhan):
        why.append(f'g not identified per component: map noise moves g by {dmax.round(6).tolist()} > '
                   f'max(|g_E|, U_cal) per component (design 2b, A6C1-11)')
    sE_raw = 2.0 * se + dmax
    chan_doan = None
    if not usable:
        why.append(f'F15 {f15["schema"]} trang_thai {f15["trang_thai"]!r}: diagnostic only (KIEMCHEO KCF1-1), '
                   f'U_meo_E, s_E not usable bounds (A6C1-1)')
        chan_doan = dict(U_meo_E=[float(x) for x in dmax], s_E=[float(x) for x in sE_raw],
                         ghi_chu='so chan doan, khong vao eps_g, u_h')
    rho = float(np.max(np.linalg.norm(Pu - C, axis=2)))                # every corner (A6C1-8)
    gm = fit['g_gian']
    return dict(base, g_E=[float(x) for x in g], sigma=fit['sigma'], SE_E=[float(x) for x in se],
                g_gian=[float(x) for x in gm], SE_gian=[float(x) for x in se_gian],
                lech_canh_gian=float(np.linalg.norm(g - gm)),
                g_phang=[float(x) for x in fit_p['g']], g_phang_dat=bool(fit_ok(fit_p)),
                bien_dang=fit['bien_dang'], g_khong_bien_dang=[float(x) for x in fit_n['g']],
                U_meo_E=[float(x) for x in dmax] if usable else None,
                s_E=[float(x) for x in sE_raw] if usable else None,
                s_E_norm=float(np.linalg.norm(sE_raw)) if usable else None,
                U_meo_E_theo_ca=theo_ca, A3_theo_ca_deg=f15.get('A3_ca_deg'), chan_doan=chan_doan, nhan_dang=nhan_dang,
                fit=dict(hoi_tu=fit['hoi_tu'], so_vong=fit['so_vong'], cond=fit['cond'], cond_max=COND_MAX,
                         se_alpha_s_px=fit['se_alpha'], se_alpha_max=ALPHA_SE_MAX,
                         jackknife_dat=jk_ok, replay_dat=rep_ok),
                res_rms_m=fit['res_rms_m'], n_khung=int(len(i1)),
                man_trap=dict(cach='khop alpha_r, alpha_c lam tham so phu cung g (L133)',
                              alpha_r_s_px=fit['alpha'][0], alpha_c_s_px=fit['alpha'][1],
                              omega_max_rad_s=OMEGA_MAX_RAD_S,
                              tuong_quan_alpha_g_max=corr_alpha_g(fit['M'], fit['i_alpha']),
                              g_khong_alpha=[float(x) for x in fit0['g']], SE_E_khong_alpha=[float(x) for x in se0]),
                rho_max_goc_m=rho, trang_thai='dat' if not why else 'thieu', ly_do=why)


# ── e_bc ─────────────────────────────────────────────────────────────────────────────────────────

def _ebc_level(g_hat, eps_g, C, H, hbar, side, cam_c, rho_max, v_max, k, chunk=400_000):
    """Largest |heading error after the correction - truth| on one grid: tag centre at k + 1 radii in [0, rho_max -
    side / sqrt 2] (every corner within rho_max) x 2k directions, 2k tag headings, g = g_hat + eps_g r (cos, sin) for
    r in {0} and k // 2 radii x k directions, v in k // 2 + 1 levels of [-v_max, v_max] (A6C2-9: from k = 4 every
    dimension is refined at every level).
    Real car_pose path (A6C1-9): ray from the camera centre cam_c = (x, y, z) through the corner at its height to the
    floor, lift with kappa = hbar / H about C, Kabsch; the model evaluated at the MEASURED centre and heading."""
    C = np.asarray(C, float)
    r_in = max(0.0, rho_max - side / math.sqrt(2.0))
    rhos = np.linspace(0.0, r_in, k + 1)
    angs = np.linspace(0.0, 2 * math.pi, 2 * k, endpoint=False)
    if k < 4:
        raise A6Error(f'e_bc grid level k = {k} < 4 (A6C2-9)')
    nr, nd = k // 2, k
    gd = np.linspace(0.0, 2 * math.pi, nd, endpoint=False)
    gs = [np.asarray(g_hat, float)] + [np.asarray(g_hat, float) + eps_g * (j / nr) * np.array([math.cos(a), math.sin(a)])
                                       for j in range(1, nr + 1) for a in gd]
    gs = np.array(gs)
    vs = np.linspace(-v_max, v_max, k // 2 + 1)
    grids = np.meshgrid(np.arange(len(rhos)), np.arange(len(angs)), np.arange(len(angs)), np.arange(len(gs)),
                        np.arange(len(vs)), indexing='ij')
    flat = [x.ravel() for x in grids]
    worst = 0.0
    for s0 in range(0, len(flat[0]), chunk):
        iR, iA, iS, iG, iV = (x[s0:s0 + chunk] for x in flat)
        T = C[None, :] + rhos[iR][:, None] * np.column_stack([np.cos(angs[iA]), np.sin(angs[iA])])
        err = ebc_errors(T, angs[iS], gs[iG], vs[iV], C, H, hbar, side, cam_c)
        worst = max(worst, float(np.max(np.abs(err))))
    return worst


def ebc_errors(T, S, g, v, C, H, hbar, side, cam_c):
    """Heading error after the A6 correction, per configuration (tag centre T (N, 2), heading S (N), tilt g (N, 2),
    warp v (N)): the corners at their heights seen from the camera centre cam_c = (x, y, z) on the floor, lifted with
    kappa = hbar / H about the plumb C (car_pose), Kabsch heading and centre, minus the model at the MEASURED pose,
    minus the truth (A6C1-9)."""
    C = np.asarray(C, float)
    Cc, Hc = np.asarray(cam_c[:2], float), float(cam_c[2])
    kap = hbar / H
    r = body(side)
    a = r - r.mean(0)
    T, S, g, v = (np.asarray(x, float) for x in (T, S, g, v))
    c, s_ = np.cos(S), np.sin(S)
    X = T[:, None, :] + np.stack([c[:, None] * r[None, :, 0] - s_[:, None] * r[None, :, 1],
                                  s_[:, None] * r[None, :, 0] + c[:, None] * r[None, :, 1]], axis=2)
    h = hbar + g @ r.T + v[:, None] * S_PATTERN[None, :]
    p = Cc[None, None, :] + (X - Cc[None, None, :]) * (Hc / (Hc - h))[:, :, None]
    Pm = p + kap * (C[None, None, :] - p)
    tm = Pm.mean(1)
    hm = np.einsum('ij,nik->njk', a, Pm - tm[:, None, :])
    pm = np.arctan2(hm[:, 0, 1] - hm[:, 1, 0], hm[:, 0, 0] + hm[:, 1, 1])
    return _wrap(pm - S - dpsi_model_vec(g, tm, pm, C, H, hbar))


def dpsi_model_vec(g, t, psi, C, H, hbar):
    """dpsi_model with one g per row."""
    c, s = np.cos(psi), np.sin(psi)
    wx, wy = c * g[:, 0] - s * g[:, 1], s * g[:, 0] + c * g[:, 1]
    d = t - np.asarray(C, float)[None, :]
    return (wx * d[:, 1] - wy * d[:, 0]) / (2.0 * (H - hbar))


def e_bc(g_hat, eps_g, C, H, hbar, side, cam_c, rho_max=RHO_MAX_M, v_max=V_MAX_M):
    """design 2a: the grid is halved (k = 4, 8, 16) until the maximum changes by < EBC_TOL; an index, no claim
    between the nodes. Returns (value or None when not converged by EBC_KMAX, [[k, maximum], ...]) (A6C1-9)."""
    lv, k = [], EBC_K0
    while k <= EBC_KMAX:
        lv.append([k, _ebc_level(g_hat, eps_g, C, H, hbar, side, cam_c, rho_max, v_max, k)])
        if len(lv) >= 2:
            prev, cur = lv[-2][1], lv[-1][1]
            if cur == prev or (prev > 0 and abs(cur - prev) < EBC_TOL * prev):
                return cur, lv
        k *= 2
    return None, lv


# ── sources of gop (A6C1-4, A6C1-5, A6C1-7) ──────────────────────────────────────────────────────

def _load(path, schema):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    ok = (schema,) if isinstance(schema, str) else tuple(schema)
    if not isinstance(d, dict) or d.get('schema') not in ok:
        raise A6Error(f'{path}: not a {" / ".join(ok)} file')
    _check_sha(d, path)
    return d


def kiem_lap(d, tam_id):
    """U_chuyen.lap from a c8b_a6_lap_v1: pairs of windows of DIFFERENT mountings of the plate on the same image
    footprint (heading within LAP_HUONG_MAX_DEG, centres within LAP_VUNG_M); u = max over pairs |g_a - g_b| +
    2 sqrt(|SE_a|^2 + |SE_b|^2) (noise added, not removed, as dong [DE XUAT]). Returns (u or None, reasons, {frozenset
    (mounting a, mounting b): u of that pair} for the per-place cross-check, A6C2-3)."""
    if d is None:
        return None, ['U_chuyen.lap missing (c8b_a6_lap_v1: two windows of different mountings on the same image '
                      'footprint, design 2c)'], {}
    why = []
    if d.get('tam_id') != tam_id:
        why.append(f'lap: plate {d.get("tam_id")!r}, not {tam_id!r}')
    if d.get('trang_thai') != 'dat':
        why.append(f'lap: trang_thai {d.get("trang_thai")!r}')
    cap = d.get('cap')
    mot = d.get('mot_lan_lap')
    if isinstance(mot, dict) and not cap:
        # One mounting for the E clips and every scored run (CODE_C8B 22/11): the transfer between mountings is 0 by
        # its definition (design 2c: lap is the change of g between DIFFERENT mountings). Needs the mounting id, the
        # basis (who said so, where) and the repeat pairs of E clips of that mounting on the same footprint, kept as
        # evidence (not added: their noise is already in s_E). c8b_khoa_v5 checks every run against F8 lan_lap_v1.
        if not (isinstance(mot.get('lan_lap_id'), str) and mot['lan_lap_id']
                and isinstance(mot.get('can_cu'), list) and mot['can_cu']
                and all(isinstance(x, str) and x for x in mot['can_cu'])):
            why.append('lap: mot_lan_lap needs lan_lap_id and can_cu (the basis, a list of strings)')
        ll = d.get('lap_lai')
        if not isinstance(ll, list) or not ll:
            why.append('lap: mot_lan_lap needs lap_lai (repeat pairs of E clips of the mounting, evidence)')
        else:
            for i, c in enumerate(ll):
                try:
                    if not isinstance(c, dict) or not (isinstance(c.get('luot_a'), str)
                                                       and isinstance(c.get('luot_b'), str) and c['luot_a']
                                                       and c['luot_b'] and c['luot_a'] != c['luot_b']):
                        raise A6Error('luot_a, luot_b not two different run ids')
                    if abs(_num(c.get('huong_lech_deg'), 'huong_lech_deg')) > LAP_HUONG_MAX_DEG:
                        raise A6Error(f'heading apart {c["huong_lech_deg"]} deg > {LAP_HUONG_MAX_DEG}')
                    if _num(c.get('khoang_m'), 'khoang_m', 0.0) > LAP_VUNG_M:
                        raise A6Error(f'centres {c["khoang_m"]} m apart > {LAP_VUNG_M}')
                    _vec(c.get('g_a'), 2, 'g_a'), _vec(c.get('g_b'), 2, 'g_b')
                    _vec(c.get('SE_a'), 2, 'SE_a', 0.0), _vec(c.get('SE_b'), 2, 'SE_b', 0.0)
                except A6Error as e:
                    why.append(f'lap lap_lai {i}: {e}')
        if why:
            return None, why, {}
        return 0.0, [], {}
    if not isinstance(cap, list) or not cap:
        why.append('lap: no pair (and no mot_lan_lap block)')
        return None, why, {}
    us, per = [], {}
    for i, c in enumerate(cap):
        try:
            if not isinstance(c, dict):
                raise A6Error('not an object')
            a_, b_ = c.get('lan_lap_a'), c.get('lan_lap_b')
            if not (isinstance(a_, str) and isinstance(b_, str) and a_ and b_ and a_ != b_):
                raise A6Error(f'mountings {a_!r}, {b_!r} not two different ids')
            if abs(_num(c.get('huong_lech_deg'), 'huong_lech_deg')) > LAP_HUONG_MAX_DEG:
                raise A6Error(f'heading apart {c["huong_lech_deg"]} deg > {LAP_HUONG_MAX_DEG}')
            if _num(c.get('khoang_m'), 'khoang_m', 0.0) > LAP_VUNG_M:
                raise A6Error(f'centres {c["khoang_m"]} m apart > {LAP_VUNG_M}')
            ga, gb = _vec(c.get('g_a'), 2, 'g_a'), _vec(c.get('g_b'), 2, 'g_b')
            sa, sb = _vec(c.get('SE_a'), 2, 'SE_a', 0.0), _vec(c.get('SE_b'), 2, 'SE_b', 0.0)
            us.append(float(np.linalg.norm(ga - gb) + 2.0 * math.sqrt(sa @ sa + sb @ sb)))
            key = frozenset((a_, b_))
            per[key] = max(per.get(key, 0.0), us[-1])
        except A6Error as e:
            why.append(f'lap pair {i}: {e}')
    if why:
        return None, why, {}
    return max(us), [], per


def kiem_dong(d, tam_id, g_hat, khoa_sha=None):
    """U_chuyen.dong from a c8b_a6_dong_v1 (design 2c, V5-02): per target run, a fit of g by acceleration band and half
    band on the C8 runs WITHOUT the target run; u = |g_bar - g_hat| + 2 |SE| (noise added); a band with fewer than
    N_MIN frames or no other run -> thieu for the frames of that band. Returns (block or None, reasons).
    22/11 night: also a c8b_a6_dong_v2 (dong_huong: g_bar from the heading channel camera - IMU, half bands split at
    the median acceleration); that file is written after the lock, so it must carry khoa_sha, the lock of the A6
    file; the index and the reading of the bands are the same."""
    if d is None:
        return None, ['U_chuyen.dong missing (c8b_a6_dong_v1 / v2: leave-one-run-out by acceleration band, design 2c)']
    why = []
    v2 = d.get('schema') == SCHEMA_DONG_H
    if d.get('schema') not in (SCHEMA_DONG, SCHEMA_DONG_H):
        why.append(f'dong: schema {d.get("schema")!r}')
    if v2:
        if d.get('kenh') != DONG_KENH_H or d.get('chia_nua') != CHIA_TV:
            why.append(f'dong v2: kenh {d.get("kenh")!r} / chia_nua {d.get("chia_nua")!r}')
        if khoa_sha is None or d.get('khoa_truoc_sha256') != khoa_sha:
            why.append('dong v2: khoa_truoc_sha256 is not the lock of this A6 file')
    if d.get('tam_id') != tam_id:
        why.append(f'dong: plate {d.get("tam_id")!r}, not {tam_id!r}')
    if d.get('trang_thai') != 'dat':
        why.append(f'dong: trang_thai {d.get("trang_thai")!r}')
    dn = d.get('dai_dinh_nghia')
    names = [x.get('ten') for x in dn] if isinstance(dn, list) and all(isinstance(x, dict) for x in dn) else []
    if not names or len(set(names)) != len(names) or not all(isinstance(x, str) and x for x in names) \
            or not any(x.get('nua') is True for x in dn) or not any(x.get('nua') is False for x in dn):
        why.append('dong: dai_dinh_nghia needs unique bands, full and half bands')
    else:
        try:
            nua_cua = _dong_halves(dn, CHIA_TV if v2 else None)
        except A6Error as e:
            why.append(f'dong: {e}')
    luot = d.get('luot')
    if not isinstance(luot, dict) or not luot:
        why.append('dong: no target run')
    if why or g_hat is None:
        return None, why + ([] if g_hat is not None else ['dong: no g_hat'])
    theo, thieu = {}, []
    for lid, x in luot.items():
        bo = (x or {}).get('bo_luot')
        # A6C2-4: the fit leaves out the target run only: it uses EVERY other target run of the file
        if not isinstance(bo, list) or len(set(bo)) != len(bo) or set(bo) != set(luot) - {lid}:
            why.append(f'dong {lid}: bo_luot {bo!r} must be every other run of the file ({sorted(set(luot) - {lid})})')
            continue
        theo[lid] = {}
        for nm in names:
            b = ((x or {}).get('dai') or {}).get(nm)
            try:
                if not bo:
                    raise A6Error('no other run')
                if not isinstance(b, dict):
                    raise A6Error('band missing')
                if b.get('trang_thai', 'dat') != 'dat':
                    raise A6Error(f'trang_thai {b.get("trang_thai")!r}')
                n = b.get('n_khung')
                if isinstance(n, bool) or not isinstance(n, int) or n < N_MIN:
                    raise A6Error(f'{n!r} frames < {N_MIN}')
                gb, se = _vec(b.get('g_bar'), 2, 'g_bar'), _vec(b.get('SE'), 2, 'SE', 0.0)
                theo[lid][nm] = float(np.linalg.norm(gb - np.asarray(g_hat, float)) + 2.0 * np.linalg.norm(se))
            except A6Error as e:
                theo[lid][nm] = None
                thieu.append([lid, nm, str(e)])
    if why:
        return None, why
    vals = [v for t in theo.values() for v in t.values() if v is not None]
    if not vals:
        return None, ['dong: no band with data']
    return dict(theo_luot=theo, max=max(vals), thieu=thieu, dai_dinh_nghia=dn,
                nua_cua={k: list(v) for k, v in nua_cua.items()}, schema=d.get('schema'),
                kenh=d.get('kenh') if v2 else 'meo_4_goc', chia_nua=d.get('chia_nua') if v2 else 'giua_khoang'), []


def _dong_halves(dn, chia=None):
    """A6C2-4: every full band [a_lo, a_hi] has exactly its two half bands [a_lo, mid], [mid, a_hi] (mid the middle),
    every half band belongs to one full band; edges finite, a_lo < a_hi. Returns {full band: (low half, high half)}.
    chia CHIA_TV (dong v2): the two halves meet at ONE point strictly inside the band (the median acceleration of
    the frames), not necessarily the middle."""
    eg = {}
    for x in dn:
        lo, hi = _num(x.get('a_lo'), f'band {x.get("ten")} a_lo'), _num(x.get('a_hi'), f'band {x.get("ten")} a_hi')
        if not lo < hi or not isinstance(x.get('nua'), bool):
            raise A6Error(f'band {x.get("ten")}: a_lo < a_hi and nua true / false needed')
        eg[x['ten']] = (lo, hi, x['nua'])
    full = {k: v for k, v in eg.items() if not v[2]}
    half = {k: v for k, v in eg.items() if v[2]}
    out, used = {}, set()
    for k, (lo, hi, _) in full.items():
        mid = 0.5 * (lo + hi)
        tol = 1e-12 * max(1.0, abs(hi))
        if chia == CHIA_TV:
            a = [h for h, (l2, h2, _) in half.items() if abs(l2 - lo) <= tol and lo < h2 < hi]
            b = [h for h, (l2, h2, _) in half.items() if abs(h2 - hi) <= tol and lo < l2 < hi]
            if len(a) != 1 or len(b) != 1 or abs(half[a[0]][1] - half[b[0]][0]) > tol:
                raise A6Error(f'full band {k} [{lo}, {hi}] without exactly two half bands meeting at one point')
            out[k] = (a[0], b[0])
            used |= {a[0], b[0]}
            continue
        a = [h for h, (l2, h2, _) in half.items() if abs(l2 - lo) <= tol and abs(h2 - mid) <= tol]
        b = [h for h, (l2, h2, _) in half.items() if abs(l2 - mid) <= tol and abs(h2 - hi) <= tol]
        if len(a) != 1 or len(b) != 1:
            raise A6Error(f'full band {k} [{lo}, {hi}] without exactly its two half bands')
        out[k] = (a[0], b[0])
        used |= {a[0], b[0]}
    if set(half) - used:
        raise A6Error(f'half bands {sorted(set(half) - used)} of no full band')
    return out


def _json_eq(a, b):
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)


def kiem_o(d, cho_sha, khoa_sha, rows=None, khoa=None):
    """The held-out camera-IMU check (design 2b) from a c8b_a6_o_v1: the place fits it used (cho_sha256) are files
    given to gop, locked before (khoa_truoc_sha256 = the lock of gop), three disjoint NON-EMPTY sets (khop, kiem,
    dich) equal to the sets of the lock (khoa: the parsed lock JSON with 'tap' {khop, kiem, dich}), the cells REBUILT
    here from the rows (c8b_a6_o.read_rows of the rows file named in its nguon; the CLI checks that file's sha256) with
    c8b_a6_o.build and equal to the file (no cell can be dropped, A6C2-2), and in EVERY cell of the test set
    n_id >= O_N_ID_MIN, n_thieu 0 and |d|max <= U_psi + U_IMU,D (absolute, before the margin). Returns (dat, reasons)
    (A6C1-4, A6C2-2)."""
    if d is None:
        return False, ['no camera-IMU held-out check (c8b_a6_o_v1, design 2b): A6 not checked']
    why = []
    if d.get('muc_dich') != 'giu_ngoai_a6':
        why.append(f'giu_ngoai: muc_dich {d.get("muc_dich")!r}, not giu_ngoai_a6')
    tap0 = d.get('tap') or {}
    if not all(isinstance(tap0.get(k), list) and tap0.get(k) for k in ('khop', 'kiem', 'dich')):
        why.append('giu_ngoai: tap khop, kiem, dich must all be non-empty lists (A6C2-2)')
    kt = (khoa or {}).get('tap') if isinstance(khoa, dict) else None
    if not isinstance(kt, dict) or any(not isinstance(kt.get(k), list) or sorted(kt[k]) != sorted(tap0.get(k) or [])
                                       for k in ('khop', 'kiem', 'dich')):
        why.append('giu_ngoai: tap is not the three sets of the lock (khoa_truoc.tap) (A6C2-2)')
    if rows is None:
        why.append('giu_ngoai: rows of the cells not given: cells not rebuilt (A6C2-2)')
    else:
        import c8b_a6_o as om
        try:
            rb = om.build(rows, d.get('muc_dich'), d.get('giu_ngoai_theo'), d.get('khoa_truoc_sha256'),
                          (d.get('a6_khop') or {}).get('cho_sha256') or [],
                          tuple((d.get('khoa_o') or {}).get('DAI_U') or ()))
            rb = json.loads(json.dumps(rb, ensure_ascii=False))
            bad = [k for k in ('o', 'tap', 'trang_thai', 'khoa_o', 'n_ngoai_dai_u') if not _json_eq(rb.get(k), d.get(k))]
            if bad:
                why.append(f'giu_ngoai: {bad} differ from the cells rebuilt from the rows (A6C2-2)')
        except (ValueError, TypeError, KeyError) as e:
            why.append(f'giu_ngoai: cells not rebuilt from the rows ({e}) (A6C2-2)')
    if d.get('truoc_le') is not True:
        why.append('giu_ngoai: truoc_le true needed')
    if khoa_sha is None or d.get('khoa_truoc_sha256') != khoa_sha:
        why.append('giu_ngoai: khoa_truoc_sha256 is not the lock of this A6 file')
    used = (d.get('a6_khop') or {}).get('cho_sha256')
    if not isinstance(used, list) or not used or not set(used) <= set(cho_sha or ()):
        why.append('giu_ngoai: a6_khop.cho_sha256 is not a set of the place files given')
    tap = d.get('tap') or {}
    sets = {k: tap.get(k) for k in ('khop', 'kiem', 'dich')}
    if not all(isinstance(v, list) for v in sets.values()) or not sets['kiem']:
        why.append('giu_ngoai: tap {khop, kiem, dich} lists needed, kiem not empty')
    elif any(set(sets[a]) & set(sets[b]) for a, b in (('khop', 'kiem'), ('khop', 'dich'), ('kiem', 'dich'))):
        why.append('giu_ngoai: tap khop / kiem / dich not disjoint')
    if d.get('giu_ngoai_theo') not in ('cho', 'chieu', 'lan_lap', 'khoi_thoi_gian'):
        why.append(f'giu_ngoai: giu_ngoai_theo {d.get("giu_ngoai_theo")!r}')
    cells = d.get('o')
    if not isinstance(cells, list) or not cells:
        why.append('giu_ngoai: no cell')
        cells = []
    for c in cells:
        try:
            if not isinstance(c, dict) or c.get('tap') != 'kiem':
                raise A6Error('not a cell of the test set')
            n = c.get('n_id')
            if isinstance(n, bool) or not isinstance(n, int) or n < O_N_ID_MIN:
                raise A6Error(f'n_id {n!r} < {O_N_ID_MIN}')
            if c.get('n_thieu') != 0:
                raise A6Error(f'n_thieu {c.get("n_thieu")!r}')
            dm = _num(c.get('d_max_rad'), 'd_max_rad', 0.0)
            lim = _num(c.get('U_psi_rad'), 'U_psi_rad', 0.0) + _num(c.get('U_imu_rad'), 'U_imu_rad', 0.0)
            if dm > lim:
                raise A6Error(f'|d| {dm:.6f} > U_psi + U_IMU {lim:.6f}')
            if c.get('dat') is not True:
                raise A6Error('dat not true')
        except A6Error as e:
            why.append(f'giu_ngoai cell {(c or {}).get("o_id") if isinstance(c, dict) else c!r}: {e}')
    if not why and d.get('trang_thai') != 'dat':
        why.append(f'giu_ngoai: trang_thai {d.get("trang_thai")!r}')
    return not why, why


def kiem_u_h(d, tam_id, h_m=None):
    """u_h_chung from a c8b_u_h_chung_v1 (design 3, A6C2-5): the three parts phan = {do_trai_ba_ga_m (spread of the
    held-out jigs, n_ga_giu_ngoai >= 3), lech_video_so_tay_m (|h_bar video - caliper|), hang_m (= 0.1 mm)}, each finite
    >= 0; u_h_chung_m their sum recomputed here; h_m of the file = the h of the run (the car_pose tag_h_m)."""
    if d is None:
        return None, ['u_h_chung missing (c8b_u_h_chung_v1, video multi-height, design 3)']
    why = []
    if d.get('tam_id') != tam_id:
        why.append(f'u_h_chung: plate {d.get("tam_id")!r}, not {tam_id!r}')
    if d.get('trang_thai') != 'dat':
        why.append(f'u_h_chung: trang_thai {d.get("trang_thai")!r}')
    v = None
    try:
        ph = d.get('phan')
        if not isinstance(ph, dict):
            raise A6Error('u_h_chung: phan {do_trai_ba_ga_m, lech_video_so_tay_m, hang_m} missing')
        parts = [_num(ph.get(k), f'u_h_chung phan.{k}', 0.0) for k in ('do_trai_ba_ga_m', 'lech_video_so_tay_m',
                                                                       'hang_m')]
        if abs(parts[2] - U_H_HANG_M) > 1e-12:
            raise A6Error(f'u_h_chung phan.hang_m {parts[2]} is not {U_H_HANG_M} m (design 3)')
        n = d.get('n_ga_giu_ngoai')
        if isinstance(n, bool) or not isinstance(n, int) or n < 3:
            raise A6Error(f'u_h_chung: n_ga_giu_ngoai {n!r} < 3 (three held-out jigs, design 3)')
        v = _num(d.get('u_h_chung_m'), 'u_h_chung_m', 0.0)
        if abs(v - sum(parts)) > 1e-12:
            raise A6Error(f'u_h_chung_m {v} is not the sum of its three parts {sum(parts)}')
        if h_m is not None:
            hf = _num(d.get('h_m'), 'u_h_chung h_m', 0.0)
            if abs(hf - float(h_m)) > TOL_SETUP:
                raise A6Error(f'u_h_chung h_m {hf} is not the h of the run {h_m}')
    except A6Error as e:
        why.append(str(e))
        v = None
    return (None, why) if why else (v, [])


def kiem_imu_gop(files, places, g_hat, eps, ebc, H, hb, tam_id, khoa_sha, khoa):
    """The held-out camera-IMU check of the edge slope (22/11; design 2b "giu ngoai tuyet doi" on the E clips, in the
    heading channel): every c8b_a6_kiem_imu_v1 file is of an E clip that NO place fit used (clip sha), of this plate,
    written after the lock (khoa_truoc_sha256), its run in the test set of the lock and the runs of the place fits in
    the fit set; every place of the A6 file has at least one. Pass per file: the slope seen by the heading channel
    g_yaw agrees with g_b:  |g_yaw - g_b| rho / (2 (H - h)) <= eps_g rho / (2 (H - h)) + e_bc + 2 |SE_yaw| rho /
    (2 (H - h))  (the left side is the first-harmonic heading residual after the correction; the last term the noise
    of the IMU channel, added). Returns (dat, reasons, blocks)."""
    if not files:
        return False, ['no held-out camera-IMU check on an E clip (c8b_a6_kiem_imu_v1): A6 not checked (22/11)'], []
    why, out = [], []
    used = {(p.get('f15') or {}).get('clip_sha256') for p in places}
    kt = (khoa or {}).get('tap') if isinstance(khoa, dict) else None
    if not isinstance(kt, dict) or not all(isinstance(kt.get(k), list) for k in ('khop', 'kiem', 'dich')) \
            or any(set(kt[a]) & set(kt[b]) for a, b in (('khop', 'kiem'), ('khop', 'dich'), ('kiem', 'dich'))):
        why.append('kiem_imu: the lock has no three disjoint sets tap {khop, kiem, dich}')
        kt = None
    elif any(p.get('luot_id_E') not in kt['khop'] for p in places):
        why.append('kiem_imu: the E run of a place fit is not in the fit set of the lock (tap.khop)')
    seen = set()
    for i, d in enumerate(files):
        w = f'kiem_imu {i}'
        try:
            if not isinstance(d, dict) or d.get('schema') != SCHEMA_IMU:
                raise A6Error(f'not a {SCHEMA_IMU} file')
            _check_sha(d, w)
            w = f'kiem_imu {d.get("luot_id_E")!r}'
            if d.get('tam_id') != tam_id:
                raise A6Error(f'plate {d.get("tam_id")!r}, not {tam_id!r}')
            if d.get('cho') not in CHO or d.get('trang_thai') != 'dat':
                raise A6Error(f'cho {d.get("cho")!r} / trang_thai {d.get("trang_thai")!r}')
            if not isinstance(d.get('clip_sha256'), str) or not HEX64.match(d['clip_sha256']) \
                    or d['clip_sha256'] in used:
                raise A6Error('its clip is a clip of a place fit (not held out) or clip_sha256 missing')
            if khoa_sha is None or d.get('khoa_truoc_sha256') != khoa_sha:
                raise A6Error('khoa_truoc_sha256 is not the lock of this A6 file')
            if kt is not None and d.get('luot_id_E') not in kt['kiem']:
                raise A6Error('its run is not in the test set of the lock (tap.kiem)')
            gy, se = _vec(d.get('g_yaw'), 2, 'g_yaw'), _vec(d.get('SE_yaw'), 2, 'SE_yaw', 0.0)
            rho = _num(d.get('rho_m'), 'rho_m', 0.0)
            if g_hat is None or eps is None or ebc is None:
                raise A6Error('g_b, eps_g or e_bc missing: the check cannot be evaluated')
            k2 = 2.0 * (H - hb)
            du = float(np.linalg.norm(gy - np.asarray(g_hat, float)) * rho / k2)
            lim = float(eps * rho / k2 + ebc + 2.0 * np.linalg.norm(se) * rho / k2)
            out.append(dict(luot_id_E=d.get('luot_id_E'), cho=d['cho'], clip_sha256=d['clip_sha256'],
                            sha256=d['sha256'], g_yaw=[float(v) for v in gy], SE_yaw=[float(v) for v in se],
                            rho_m=rho, du_rad=du, nguong_rad=lim, dat=bool(du <= lim)))
            seen.add(d['cho'])
            if du > lim:
                raise A6Error(f'heading residual {du:.6f} rad > A6du + IMU noise {lim:.6f} rad (kiem_imu_truot)')
        except A6Error as e:
            why.append(f'{w}: {e}')
    miss = sorted({p['cho'] for p in places} - seen)
    if miss:
        why.append(f'kiem_imu: no held-out check at {miss}')
    return not why, why, out


# ── gop: the A6 file ─────────────────────────────────────────────────────────────────────────────

def _deg_rad(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        return None
    return math.radians(float(v))


def _su_chung(su):
    """The set-up without the camera of the session (da session (A), contract log 20/11 khuya): plumb, Z_c, tag_h,
    side, kappa are the L125 constants of every session; only session_sha256 and the camera centre may differ."""
    return {k: v for k, v in su.items() if k not in SU_THEO_SESSION}


def _cam_list(places, cam_cham, s0):
    """Distinct cameras {session_sha256, cam_tam_m, vai} of the E places and of the scoring sessions (cam_cham, from
    gop --session-cham; same L125 set-up as the places, else A6Error): e_bc is taken over all of them (da session)."""
    out, seen = [], set()
    for vai, su in [(f'E {p["cho"]}', p['set_up']) for p in places] + [('cham', c) for c in (cam_cham or ())]:
        if json.dumps(_su_chung(dict(su)), sort_keys=True) != json.dumps(_su_chung(s0), sort_keys=True):
            raise A6Error(f'{vai}: set-up {_su_chung(dict(su))} is not the set-up of the places (L125)')
        key = (su.get('session_sha256'), json.dumps(su.get('cam_tam_m')))
        if key in seen:
            for o in out:
                if (o['session_sha256'], json.dumps(o['cam_tam_m'])) == key and vai not in o['vai']:
                    o['vai'].append(vai)
            continue
        seen.add(key)
        out.append(dict(session_sha256=su.get('session_sha256'), cam_tam_m=su.get('cam_tam_m'), vai=[vai]))
    return out


def set_up_of_session(ses):
    """The set-up dict of a session (read_session) as written by do (for gop --session-cham)."""
    cam = camera_of(ses)
    return dict(C=list(map(float, ses['C'])), H_m=ses['H'], h_m=ses['hbar'], side_m=ses['side'], kappa=ses['kappa'],
                session_sha256=ses['sha256'],
                cam_tam_m=[float(x) for x in np.asarray(getattr(cam, 'centre', (math.nan,) * 3), float)])


def gop(places, tam_id, tam_cal_id, lap=None, dong=None, giu_ngoai=None, u_h=None, khoa_truoc_sha256=None,
        script=None, cho_sha=(), nguon=(), with_ebc=True, lan_lap_cal_id=None, giu_ngoai_rows=None, khoa=None,
        cam_cham=(), kiem_imu=None):
    """c8b_a6_v2 of one plate from the place results (design 2c, 3, 10); the sources are parsed dicts (CLI: files);
    giu_ngoai_rows the rows of the held-out cells (c8b_a6_o.read_rows), khoa the parsed lock JSON (A6C2-2).
    cam_cham: set-ups (set_up_of_session) of the sessions where the file is used for scoring; e_bc is the largest
    over the cameras of the places and of these (da session (A)); the places may come from different sessions.
    22/11 (Liam at CODE_C8B: "Sua mo hinh (Recommended)"): g_b is the edge slope gk of the places (MO_HINH); the
    caliper numbers g_cal are kept as a REFERENCE only (they are 4 corner heights of the plate at the first placement
    of session 1, not the slope along the tag edges of the mounting scored): they no longer enter eps_g or the status;
    the cross-check is the held-out camera-IMU check on E clips (kiem_imu: c8b_a6_kiem_imu_v1 files of clips that no
    place fit used; every place needs one; the heading-channel slope must agree with g_b within eps_g, e_bc and the
    noise of the IMU channel). The cell file giu_ngoai (c8b_a6_o_v1) stays checked when given; without it the cells
    are the D2 check of the scored runs (c8b_a6_o d2, chain step T14)."""
    why = []
    if not places:
        raise A6Error('no place')
    su = {json.dumps(_su_chung(p['set_up']), sort_keys=True) for p in places}
    if len(su) != 1:                                                  # da session (A): only the camera may differ
        raise A6Error('places measured with different car_pose set-ups (other than the camera of the session)')
    if any(p.get('tam_id') != tam_id for p in places):
        raise A6Error(f'a place of another plate than {tam_id!r} (A6C1-7)')
    s0 = places[0]['set_up']
    C, H, hb, side = np.array(s0['C']), s0['H_m'], s0['h_m'], s0['side_m']
    _check_l125(C, H, 'gop set_up')                                       # A6C2-6
    for p in places:
        if p['trang_thai'] != 'dat':
            why += [f'{p["cho"]}: {w}' for w in p['ly_do']] or [f'{p["cho"]}: trang_thai {p["trang_thai"]}']
    tam_moi = tam_id != tam_cal_id
    if tam_moi:
        why.append(f'tam_moi: plate {tam_id!r} is not the caliper plate {tam_cal_id!r}: no hand cross-check, warp '
                   f'domain not confirmed (T16.8, design 2c): A6, P6 thieu')
    ok = [p for p in places if p['g_E'] is not None and p['s_E'] is not None]
    gc, v_cal, h_cal = g_cal()
    uc, uc_norm = u_cal()
    g_hat = s_norm = None
    if ok:
        w = [1.0 / max(p['s_E_norm'], 1e-12) ** 2 for p in ok]
        g_hat = np.sum([wi * np.array(p['g_E']) for wi, p in zip(w, ok)], axis=0) / sum(w)
        s_norm = max(p['s_E_norm'] for p in ok)
    u_lap, w_lap, lap_cap = kiem_lap(lap, tam_id)
    why += w_lap
    # A6C1-1: U_chuyen.cho only from places with a usable bound (full F15 dat), never from a diagnostic g_E
    gp = {p['cho']: np.array(p['g_E']) for p in ok}
    u_cho = None
    if len(gp) == 2:
        u_cho = float(np.linalg.norm(gp['goc'] - gp['K1']))
    elif len(places) == 1:
        why.append('one place only: U_chuyen.cho (goc - K1) not measured')
    else:
        why.append('a place without a usable g (F15 full dat): U_chuyen.cho not measured')
    dong_b, w_dong = kiem_dong(dong, tam_id, g_hat, khoa_truoc_sha256)
    why += w_dong
    u_dong = None if dong_b is None else dong_b['max']
    # per place consistency with the caliper numbers (A6C1-6), every place must pass
    nq = {}
    if not tam_moi:
        for p in ok:
            dist_p = float(np.linalg.norm(np.array(p['g_E']) - gc))
            # A6C2-3: widened only by a pair joining the caliper mounting and the mounting of this place's E clip;
            # no --lan-lap-cal-id or no such pair -> not widened
            lp = p.get('lan_lap_id')
            u_pair = None
            if isinstance(lan_lap_cal_id, str) and lan_lap_cal_id and isinstance(lp, str) and lp != lan_lap_cal_id:
                u_pair = lap_cap.get(frozenset((lan_lap_cal_id, lp)))
            lim = p['s_E_norm'] + uc_norm + (u_pair or 0.0)
            nq[p['cho']] = dict(khoang=dist_p, nguong=lim, dat=bool(dist_p <= lim), u_lap_cap=u_pair,
                                lan_lap_id=lp, lan_lap_cal_id=lan_lap_cal_id)
            # 22/11: reference only (no reason added): the caliper is not the slope along the tag edges of this mounting
    nhat_quan = None if (tam_moi or not nq) else all(v['dat'] for v in nq.values())
    dist = max(v['khoang'] for v in nq.values()) if nq else None
    uch = None if None in (u_lap, u_cho, u_dong) else float(u_lap + u_cho + u_dong)
    eps0 = None if (s_norm is None or uch is None) else float(s_norm + u_lap + u_cho)      # 22/11: no g_cal term
    eps = None if eps0 is None else eps0 + u_dong
    # design 2b joint replay (CODE 20/11, changed decision: A3 per H32 case of KIEMCHEO 692462f): the same eps without
    # the U_meo_E part (max(a + b, c) <= max(a, c) + b), and per case the signed change of g and A3 of that case
    s2 = max((2.0 * float(np.linalg.norm(p['SE_E'])) for p in ok), default=None)
    eps0_km = None if (eps0 is None or s2 is None) else float(s2 + u_lap + u_cho)
    ca_cho = {}
    for p in ok:
        a3c = p.get('A3_theo_ca_deg') or {}
        ca_cho[p['cho']] = {nm: dict(dg=dg, dg_norm=float(np.linalg.norm(dg)), A3_rad=_deg_rad(a3c.get(nm)))
                            for nm, dg in (p.get('U_meo_E_theo_ca') or {}).items()}
    if giu_ngoai is None:                        # 22/11: the cells are optional here (D2 of the runs at T14)
        o_ok = None
    else:
        o_ok, w_o = kiem_o(giu_ngoai, cho_sha, khoa_truoc_sha256, giu_ngoai_rows, khoa)
        why += w_o
    if khoa_truoc_sha256 is None:
        why.append('khoa_truoc_sha256 missing (the lock before the checks, design 0.3, A6C1-4)')
    if not isinstance(script, dict) or not script.get('commit') or script.get('dirty') is not False:
        why.append(f'script commit not recorded or dirty ({script!r}): no code SHA (A6C1-4)')
    cams = _cam_list(places, cam_cham, s0)                            # da session (A): every camera of use
    cam_c = cams[0]['cam_tam_m'] if len(cams) == 1 else None
    ebc, ebc_lv, ebc_ses = None, [], []
    if g_hat is not None and with_ebc:
        bad = [c['session_sha256'] for c in cams
               if c['cam_tam_m'] is None or not np.all(np.isfinite(np.asarray(c['cam_tam_m'], float)))]
        if bad:
            why.append(f'no camera centre of the session(s) {bad}: e_bc not computed (A6C1-9)')
        else:
            vals = []
            for c in cams:
                v, lv = e_bc(g_hat, eps if eps is not None else (s_norm + uc_norm), C, H, hb, side, c['cam_tam_m'])
                ebc_ses.append(dict(c, e_bc_rad=v, e_bc_lan_chia=lv))
                vals.append((v, lv))
            if any(v is None for v, _ in vals):
                why.append(f'e_bc grid not converged by k = {EBC_KMAX} (maxima '
                           f'{[lv for v, lv in vals if v is None]}) (design 2a, A6C1-9)')
            else:
                ebc, ebc_lv = max(vals, key=lambda x: x[0])
    mien_ok = None
    if g_hat is not None:
        tilt = math.atan(float(np.linalg.norm(g_hat)) + (eps or 0.0))
        rho = max(p.get('rho_max_goc_m') or 0.0 for p in ok)
        # 22/11 (Liam at CODE_C8B, box, after the first official gop gave 2.46 deg: "Mien theo du lieu (Recommended)"):
        # no fixed tilt cap. The tilt domain is the set the e_bc grid covers (g_b + eps_g in every direction, v within
        # v_max), usable when that grid converged; a grid that did not converge already has its own reason above.
        mien_ok = bool(rho <= RHO_MAX_M and (ebc is not None or not with_ebc))
        if not rho <= RHO_MAX_M:
            why.append(f'outside the A6 domain: a corner at rho {rho:.3f} m (<= {RHO_MAX_M})')
    imu_ok, w_imu, imu_b = kiem_imu_gop(kiem_imu, places, g_hat, eps, ebc, H, hb, tam_id, khoa_truoc_sha256, khoa)
    why += w_imu
    st_a6 = 'dat' if (not why and eps is not None and ebc is not None and imu_ok and o_ok is not False) else 'thieu'
    u_hc, w_h = kiem_u_h(u_h, tam_id, hb)
    r = body(side)
    why_h = list(w_h) + ([] if st_a6 == 'dat' else ['A6 not dat: h_goc_m from g_b not usable (A6C1-7)'])
    st_h = 'dat' if not why_h else 'thieu'
    h_goc = [float(hb + ri @ g_hat) for ri in r] if st_h == 'dat' else None
    # 22/11: the corner heights of T16.8 from gk; the stretch slope gm is the other candidate, so their largest
    # difference over the places is added per corner
    dkm = max((float(p.get('lech_canh_gian') or 0.0) for p in ok), default=0.0)
    u_h_goc = [float(u_hc + np.linalg.norm(ri) * (eps + dkm) + V_MAX_M) for ri in r] if st_h == 'dat' else None
    lans = sorted({p.get('lan_lap_id') for p in places})
    # A6C1-7 (CODE 20/11, hop dong ~05:00): the declared mounting and the E run of every place; c8b_khoa_v5 checks them
    # against F8 lan_lap_v1 (F8 is computed from this file's g_b, so it cannot be read here)
    lan_cho = {p['cho']: dict(lan_lap_id=p.get('lan_lap_id'), luot_id_E=p.get('luot_id_E')) for p in places}
    return dict(schema=SCHEMA, tam_id=tam_id, tam_cal_id=tam_cal_id, lan_lap_goc_id=lans[0] if len(lans) == 1 else lans,
                lan_lap_cho=lan_cho,
                g_b=None if g_hat is None else [float(x) for x in g_hat],
                quy_tac_g_b='trung binh trong so 1/|s_E|^2 cac cho dat (khoa truoc)', mo_hinh=MO_HINH,
                g_b_y_nghia='do doc doc theo canh ma gk (quyet dinh sai huong); do doc gian gm o g_gian_cho',
                g_E_cho={p['cho']: p['g_E'] for p in places}, g_gian_cho={p['cho']: p.get('g_gian') for p in places},
                g_phang_cho={p['cho']: p.get('g_phang') for p in places},
                eps_g=eps, eps_g_khong_dong=eps0, s_E=s_norm,
                quy_tac_eps_g='s_E + U_chuyen (lap + cho + dong); g_cal khong vao (22/11)',
                SE_E={p['cho']: p['SE_E'] for p in places}, U_meo_E={p['cho']: p['U_meo_E'] for p in places},
                U_meo_E_theo_ca={p['cho']: p.get('U_meo_E_theo_ca') for p in places},
                U_chuyen=dict(lap=u_lap, cho=u_cho, dong=u_dong), dong=dong_b,
                g_cal=None if tam_moi else [float(x) for x in gc], v_cal_m=None if tam_moi else v_cal,
                h_cal_m=None if tam_moi else h_cal, U_cal=uc_norm, U_cal_thanh_phan=uc,
                U_cal_venh='0: venh tach bang tham so v trong phep giai 4 goc (s_i truc giao do doc)',
                khoang_g_cal=dist, nhat_quan=nhat_quan, nhat_quan_cho=nq,
                g_cal_vai_tro='tham khao: 4 do cao goc tam o lan dat dau buoi 1, khong phai do doc doc canh ma cua lan '
                              'lap duoc cham; khong vao eps_g, khong vao trang_thai (Liam 22/11 o CODE_C8B)',
                kiem_imu=imu_b, kiem_imu_dat=imu_ok,
                e_bc_rad=ebc, e_bc_lan_chia=ebc_lv, e_bc_theo_session=ebc_ses,
                session=[dict(session_sha256=c['session_sha256'], cam_tam_m=c['cam_tam_m'], vai=c['vai']) for c in cams],
                a6du_rad_per_m=None if eps is None else float(eps / (2.0 * (H - hb))),
                H_m=float(H), h_m=float(hb), plumb_xy_m=[float(x) for x in C], side_m=float(side), cam_tam_m=cam_c,
                mien=dict(rho_max_m=RHO_MAX_M,
                          nghieng_max_rad=None if g_hat is None else float(math.atan(float(np.linalg.norm(g_hat))
                                                                                     + (eps or 0.0))),
                          nghieng_quy_tac='mien nghieng = vung luoi e_bc da hoi tu: atan(|g_b| + eps_g); khong tran co '
                                          'dinh (Liam 22/11 o CODE_C8B: "Mien theo du lieu (Recommended)"); 1.5 do cua '
                                          'thiet ke 2a chi con la so tham khao',
                          nghieng_de_xuat_cu_rad=NGHIENG_MAX_RAD, v_max_m=V_MAX_M, dat=mien_ok,
                          quy_tac='moi goc trong rho_max; doc chi co tam: tam + canh/sqrt2 <= rho_max'),
                h_goc_m=h_goc, u_h_goc_m=u_h_goc, u_h_chung_m=u_hc,
                u_h_rieng_m=None, u_h_rieng_nguon='theo luot: lan_lap_v1 delta_h (NHACTAM F8), khong o file tam',
                giu_ngoai_dat=o_ok,
                giu_ngoai_ghi_chu=None if giu_ngoai is not None else 'khong co file o (c8b_a6_o_v1) o buoc nay: o D2 '
                                  'kiem tren luot dich o T14 (c8b_a6_o d2)',
                eps_g_khong_meo=None if eps0_km is None else eps0_km + u_dong,
                eps_g_khong_meo_khong_dong=eps0_km,
                ghep_A3_A6=dict(cach='replay chung theo ca H32 (2b) khi hai dau cung mot cho co A3 tung ca; '
                                     'khong thi cong rieng (du phong)', theo_cho=ca_cho),
                trang_thai=dict(a6=st_a6, h=st_h), ly_do=why, ly_do_h=why_h, gioi_han=list(GIOI_HAN),
                khoa_truoc_sha256=khoa_truoc_sha256, khoa_truoc_sha=khoa_truoc_sha256,      # F3 2.C name (alias)
                commit=(script or {}).get('commit'),
                commit_dirty=(script or {}).get('dirty'), nguon=list(nguon), so_tay=SO_TAY, truoc_le=True)


# ── kiem_imu: the heading channel of an E clip (22/11) ───────────────────────────────────────────

IMU_GRID_S = 0.05                                          # as c8b_xoay.imu_check: grid of the turn-rate correlation
IMU_SCAN_S = 0.1                                           # as c8b_xoay.imu_check: +-2 grid steps, in 1 ms steps


def read_psi_series(path):
    """(t, unwrapped car_pose heading psi) of the valid frames of a frames.csv: the whole clip, for the clock offset."""
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r.get('valid') in ('1', 'True', 'true') and r.get('psi') not in ('', None)]
    if len(rows) < 20:
        raise A6Error(f'{path}: fewer than 20 valid frames with a heading')
    t = np.array([float(r['t_s']) for r in rows])
    return t, np.unwrap(np.array([float(r['psi']) for r in rows]))


def _imu_offset(tc, pc, S):
    """Clock offset log - camera from the cross-correlation of the turn rates (c8b_xoay.imu_check)."""
    g_c = np.arange(tc[0], tc[-1], IMU_GRID_S)
    g_i = np.arange(S['t'][0], S['t'][-1], IMU_GRID_S)
    if g_c.size < 20 or g_i.size < 20:
        raise A6Error('kiem_imu: clip or log shorter than 1 s')
    wc = np.gradient(np.interp(g_c, tc, pc), IMU_GRID_S)
    wi = np.gradient(np.interp(g_i, S['t'], S['psi']), IMU_GRID_S)
    nfft = 1 << int(math.ceil(math.log2(len(wc) + len(wi))))
    xc = np.fft.irfft(np.fft.rfft(wi - wi.mean(), nfft) * np.conj(np.fft.rfft(wc - wc.mean(), nfft)), nfft)
    k = int(np.argmax(xc))
    lag = k if k < nfft // 2 else k - nfft
    return float(g_i[0] - g_c[0] + lag * IMU_GRID_S)


def _yaw_design(psi, pim, w, t, u):
    """Columns of psi_cam - psi_imu: constant, IMU scale, turn rate (the rest of the clock offset and the shutter
    skew are proportional to it), drift in time, the two slopes of design 2a (delta psi = (gx u_y - gy u_x) / 2) and
    the second harmonic of the heading (the map residual at the place)."""
    return np.column_stack([np.ones_like(psi), pim - pim.mean(), w, t - t.mean(), 0.5 * u[:, 1], -0.5 * u[:, 0],
                            np.cos(2.0 * psi), np.sin(2.0 * psi)])


def kiem_imu_do(cho, idx, t, P, ses, f15, tc, pc, L, tam_id, luot_id_E, khoa_sha):
    """The slope g_yaw seen by the HEADING channel of one E clip (22/11): the Kabsch heading of the lifted corners
    minus the IMU yaw (c8_u_imu.rv_samples: the game rotation vector of the log, no magnetometer), regressed on the
    columns of _yaw_design over the E frames of F15 with |omega| <= OMEGA_MAX_RAD_S (the frames of do) that have an
    IMU yaw; the clock offset from the turn rates of the whole clip (tc, pc), refined to the smallest residual; SE by
    the block jackknife. Independent of the corner deformation that do fits (it uses the antisymmetric part only)."""
    if cho not in CHO or f15.get('cho') != cho:
        raise A6Error(f'cho {cho!r}: goc | K1, and the F15 of that place')
    if not khoa_sha:
        raise A6Error('kiem_imu: the lock (--khoa-truoc) is needed: the check is written after it')
    C, H, hb, side = ses['C'], ses['H'], ses['hbar'], ses['side']
    lo, hi = f15['mien_E']['sau_E_dau'], f15['mien_E']['truoc_E_cuoi']
    w = omega_of(t, P, side)
    psi_all = np.unwrap(np.array([kabsch_psi(q, side) for q in P]))
    sel = (idx > lo) & (idx < hi) & np.isfinite(w) & (np.abs(np.nan_to_num(w, nan=np.inf)) <= OMEGA_MAX_RAD_S)
    S = cu.rv_samples(L)
    off0 = _imu_offset(tc, pc, S)
    Tc = P.mean(1)
    best = None
    for off in np.arange(off0 - IMU_SCAN_S, off0 + IMU_SCAN_S + 5e-4, 1e-3):
        pim = cu.interp_yaw(S, t + off)
        m = sel & np.isfinite(pim)
        if m.sum() < N_MIN:
            continue
        c_, s_ = np.cos(psi_all[m]), np.sin(psi_all[m])
        dfl = (Tc[m] - C) / (H - hb)
        u = np.column_stack([c_ * dfl[:, 0] + s_ * dfl[:, 1], -s_ * dfl[:, 0] + c_ * dfl[:, 1]])
        X = _yaw_design(psi_all[m], pim[m], w[m], t[m], u)
        e = psi_all[m] - pim[m]
        co = np.linalg.lstsq(X, e, rcond=None)[0]
        rms = float(np.sqrt(np.mean((e - X @ co) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(off), co, m, X, e)
    base = dict(schema=SCHEMA_IMU, tam_id=tam_id, cho=cho, luot_id_E=luot_id_E, clip_sha256=f15['clip_sha256'],
                khoa_truoc_sha256=khoa_sha, truoc_le=True,
                set_up=dict(C=list(map(float, C)), H_m=H, h_m=hb, side_m=side, session_sha256=ses['sha256']),
                cach='psi Kabsch 4 goc da nang - yaw IMU (game RV) hoi quy tren [1, ty le IMU, omega, troi, u_y/2, '
                     '-u_x/2, cos 2psi, sin 2psi]; lech gio tu tuong quan toc do quay, tinh chinh 1 ms',
                omega_max_rad_s=OMEGA_MAX_RAD_S)
    if best is None:
        return dict(base, g_yaw=None, SE_yaw=None, rho_m=None, n_khung=0, trang_thai='thieu',
                    ly_do=[f'fewer than {N_MIN} E frames with an IMU yaw at any clock offset'])
    rms, off, co, m, X, e = best
    ii = np.flatnonzero(m)
    est = []
    for b in np.array_split(np.arange(len(ii)), N_BLOCK):
        keep = np.setdiff1d(np.arange(len(ii)), b)
        est.append(np.linalg.lstsq(X[keep], e[keep], rcond=None)[0][4:6])
    est = np.array(est)
    se = np.sqrt((N_BLOCK - 1) / N_BLOCK * np.sum((est - est.mean(0)) ** 2, axis=0))
    cond = float(np.linalg.cond(X / np.linalg.norm(X, axis=0)))
    why = [] if cond <= math.sqrt(COND_MAX) else [f'design ill conditioned ({cond:.3g})']
    rho = float(np.mean(np.linalg.norm(Tc[m] - C, axis=1)))
    return dict(base, g_yaw=[float(v) for v in co[4:6]], SE_yaw=[float(v) for v in se], rho_m=rho,
                n_khung=int(m.sum()), offset_s=off, rms_rad=rms, he_so=[float(v) for v in co], cond=cond,
                bien_do_bac1_rad=float(np.linalg.norm(co[4:6]) * rho / (2.0 * (H - hb))),
                imu=dict(log=os.path.basename(L['path']), log_sha256=cu.sha256(L['path']), n_late=S.get('n_late')),
                trang_thai='dat' if not why else 'thieu', ly_do=why)


# ── goc, dong: the lifted corners of the runs and U_chuyen.dong (22/11) ──────────────────────────

DONG_BD = ('ma',)                                          # the runs move over the floor: no floor-fixed strain


def read_cua_so(path):
    cu.check_path(path)
    with open(path, 'rb') as f:
        b = f.read()
    try:
        rows = [json.loads(x) for x in b.decode('utf-8').splitlines() if x.strip()]
    except ValueError as e:
        raise A6Error(f'{path}: {e}')
    return [r for r in rows if isinstance(r, dict)], hashlib.sha256(b).hexdigest()


def goc_run(track, session, luot_id, cua_so_rows, model=None):
    """Lifted corners (car_pose: per-frame scoring camera of the 33-mark chain, KIEMCHEO FramePoses on the dets.npz of
    the track, as meo_E builds goc_m) of the valid frames of one run strictly between its pre and post windows of F2
    (the moving part of the run, as E_doan between E_dau and E_cuoi). The replayed tag centre must be the one of
    frames.csv (KIEMCHEO TOL_REPLAY_M); a frame whose pose refit is not trusted is left out and counted."""
    import c7_kiem_cheo as kc
    import c7_measure as cm
    tp, fp = os.path.join(track, 'track.json'), os.path.join(track, 'frames.csv')
    try:
        tr, _ = kc.load_json_sha(tp)
        with open(fp, 'rb') as f:
            fb = f.read()
        fr_sha = hashlib.sha256(fb).hexdigest()
        if tr.get('frames_sha256') != fr_sha:
            raise A6Error(f'{fp}: not the frames.csv of track.json')
        clip = tr.get('video_sha256')
        with open(session, encoding='utf-8') as f:
            s0 = json.load(f)
        s, ses_sha, mk, pc, full, kappa, C = kc._session_ctx(session, tr.get('session_sha256'), model or s0.get('chosen'))
        win = {}
        for r in cua_so_rows:
            if r.get('luot_id') == luot_id and r.get('loai') in ('pre', 'post'):
                if r.get('frames_sha256') != fr_sha or r.get('session_sha256') != ses_sha or r['loai'] in win:
                    raise A6Error(f'cua_so_v1 {luot_id} {r["loai"]}: another frames / session, or twice')
                win[r['loai']] = [int(x['id']) for x in r['khung']]
        if set(win) != {'pre', 'post'}:
            raise A6Error(f'cua_so_v1: run {luot_id!r} needs one pre and one post window')
        lo, hi = max(win['pre']), min(win['post'])
        if not lo < hi:
            raise A6Error(f'cua_so_v1 {luot_id}: pre does not end before post')
        import io as _io
        rows = [r for r in csv.DictReader(_io.StringIO(fb.decode('utf-8')))
                if kc._true(r.get('valid')) and lo < int(r['idx']) < hi and r.get('tag_x') not in ('', None)]
        to = tr.get('tag_offset') or [0.0, 0.0, 0.0]
        side = float(s['car_side_m'])
        dets, dets_sha, _ = kc.load_dets_checked(tr.get('dets_file'), tr.get('dets_sha256'), None, clip, 'dets.npz')
        fp_ = kc.FramePoses(s, pc)
        khung, bo, dev = [], [], 0.0
        for r in rows:
            fid = int(r['idx'])
            if fid not in dets or cm.CAR_ID not in dets[fid][1]:
                raise A6Error(f'frame {fid}: valid in frames.csv but no car tag in dets.npz')
            d = dets[fid][1]
            try:
                cam = fp_.full(kc._ids_of(r['moc_dung']), d, f'frame {fid}')
            except kc.KCError as e:
                bo.append([fid, str(e)])
                continue
            q = cm.car_pose(cam, d[cm.CAR_ID], side, kappa, C, to)
            dev = max(dev, abs(q['tag_x'] - float(r['tag_x'])), abs(q['tag_y'] - float(r['tag_y'])))
            if not dev <= kc.TOL_REPLAY_M:
                raise A6Error(f'frame {fid}: replayed pose differs from frames.csv by {dev:.3g} m (> {kc.TOL_REPLAY_M})')
            Pl = kc.lift(cam.pix2floor(np.asarray(d[cm.CAR_ID], float), 0.0), kappa, C)
            khung.append([fid, float(r['t_s']), [float(v) for v in np.asarray(Pl, float).ravel()]])
    except (kc.KCError, OSError, KeyError, TypeError) as e:
        raise A6Error(f'goc {luot_id}: {e}')
    if len(khung) < 3:
        raise A6Error(f'goc {luot_id}: fewer than 3 frames')
    return dict(schema=SCHEMA_GOC, luot_id=luot_id, clip_sha256=clip, frames_sha256=fr_sha, session_sha256=ses_sha,
                session_file=os.path.abspath(session), dets_sha256=dets_sha, mien=dict(sau_pre=lo, truoc_post=hi),
                thu_tu_goc='ArUco 0..3 (c7_measure.car_corners_body); x0 y0 .. x3 y3 [m], da nang (car_pose)',
                n_khung=len(khung), n_bo=len(bo), bo=bo[:20], replay_max_m=dev, khung=khung)


def acc_of(t, T, half=SG_HALF):
    """Body acceleration magnitude [m/s^2] per frame from the tag centres T (n, 2): twice the quadratic coefficient of
    a fit over +-half frames on the real times; NaN where the window is not full or crosses a gap (as omega_of)."""
    t = np.asarray(t, float)
    n = len(t)
    a = np.full(n, np.nan)
    if n < 2 * half + 1:
        return a
    dt = np.diff(t)
    gap = dt > GAP_FACTOR * np.median(dt)
    for k in range(half, n - half):
        if gap[k - half:k + half].any():
            continue
        tt = t[k - half:k + half + 1] - t[k]
        cx = np.polyfit(tt, T[k - half:k + half + 1, 0], 2)
        cy = np.polyfit(tt, T[k - half:k + half + 1, 1], 2)
        a[k] = 2.0 * math.hypot(cx[0], cy[0])
    return a


def dong_build(runs, tam_id, log=None):
    """c8b_a6_dong_v1 from the corner files of the scored runs (design 2c, V5-02; 22/11 with the model MO_HINH): the
    acceleration bands are ONE full band [0, a_max] and its two halves, a_max the largest body acceleration seen on
    the frames given (no number chosen); per target run and band, gk fitted (fit_gk with the shutter fields of each
    run and the tag-fixed strain; no floor-fixed strain, the runs move) on the frames of EVERY OTHER run in the band,
    SE by the block jackknife. runs: {luot_id: (goc dict, session dict of read_session, camera)}."""
    if len(runs) < 2:
        raise A6Error('dong: at least two runs')
    dat = {}
    for lid, (g, ses, cam) in runs.items():
        if g.get('schema') != SCHEMA_GOC or g.get('luot_id') != lid or g.get('session_sha256') != ses['sha256']:
            raise A6Error(f'dong {lid}: not its {SCHEMA_GOC} file or another session')
        idx = np.array([k[0] for k in g['khung']])
        t = np.array([k[1] for k in g['khung']])
        P = np.array([k[2] for k in g['khung']]).reshape(-1, 4, 2)
        if np.any(np.diff(t) <= 0) or not np.all(np.isfinite(P)):
            raise A6Error(f'dong {lid}: times not increasing or corners not finite')
        uv = uv_of(P, cam, ses['C'], ses['kappa'])
        fr, fc = shutter_fields(P, t, uv)
        dat[lid] = dict(P=P, fr=fr, fc=fc, a=acc_of(t, P.mean(1)), n=len(idx))
    s0 = next(iter(runs.values()))[1]
    C, H, hb, side = s0['C'], s0['H'], s0['hbar'], s0['side']
    for lid, (_, ses, _) in runs.items():
        if abs(ses['H'] - H) > TOL_SETUP or abs(ses['hbar'] - hb) > TOL_SETUP or abs(ses['side'] - side) > TOL_SETUP \
                or np.max(np.abs(ses['C'] - C)) > TOL_SETUP:
            raise A6Error(f'dong {lid}: another car_pose set-up (L125)')
    a_max = float(max(np.nanmax(d['a']) for d in dat.values()))
    mid = 0.5 * a_max
    dn = [dict(ten='a', a_lo=0.0, a_hi=a_max, nua=False), dict(ten='a_thap', a_lo=0.0, a_hi=mid, nua=True),
          dict(ten='a_cao', a_lo=mid, a_hi=a_max, nua=True)]

    def fit(P_, C_, H_, hb_, side_, fields=()):
        return fit_gk(P_, C_, H_, hb_, side_, fields=fields, bien_dang=DONG_BD)

    luot = {}
    for lid in dat:
        others = [x for x in dat if x != lid]
        luot[lid] = dict(bo_luot=others, dai={})
        for b in dn:
            Ps, frs, fcs = [], [], []
            for x in others:
                m = np.isfinite(dat[x]['a']) & (dat[x]['a'] >= b['a_lo']) & (dat[x]['a'] <= b['a_hi'])
                Ps.append(dat[x]['P'][m])
                frs.append(dat[x]['fr'][m])
                fcs.append(dat[x]['fc'][m])
            Pb, fl = np.concatenate(Ps), [np.concatenate(frs), np.concatenate(fcs)]
            ent = dict(n_khung=int(len(Pb)), g_bar=None, SE=None, trang_thai='thieu')
            if len(Pb) < N_MIN:
                ent['ly_do'] = f'{len(Pb)} frames < {N_MIN}'
            else:
                f = fit(Pb, C, H, hb, side, fl)
                se, jk_ok = se_jackknife(Pb, C, H, hb, side, fields=fl, fit=fit)
                ent.update(g_bar=[float(v) for v in f['g']], SE=[float(v) for v in se[:2]],
                           g_gian=[float(v) for v in f['g_gian']], cond=f['cond'], hoi_tu=f['hoi_tu'],
                           se_alpha_s_px=f['se_alpha'], jackknife_dat=bool(jk_ok))
                if fit_ok(f) and jk_ok:
                    ent['trang_thai'] = 'dat'
                else:
                    ent['ly_do'] = 'fit or a jackknife fit not usable (A6C1-11)'
            luot[lid]['dai'][b['ten']] = ent
            if log:
                log(f'dong {lid} {b["ten"]}: n {ent["n_khung"]} g_bar {ent["g_bar"]} SE {ent["SE"]} {ent["trang_thai"]}')
    return dict(schema=SCHEMA_DONG, tam_id=tam_id, trang_thai='dat', mo_hinh=MO_HINH, dai_dinh_nghia=dn, luot=luot,
                a_max_nguon='gia toc than lon nhat tren moi khung dua vao (acc_of: khop bac hai +-%d khung)' % SG_HALF,
                bien_dang=list(DONG_BD), n_khung_luot={k: v['n'] for k, v in dat.items()}, truoc_le=True)


# ── dong_huong: U_chuyen.dong from the heading channel of the runs (22/11 night) ─────────────────

def read_gio_offset(path, frames_sha, log_sha):
    """(D, half) [s] of a run from its preliminary clock file c8b_gio_v1 (chain step T5): log time = camera time + D,
    D = center_s; the file must be ok and of this frames.csv and this log (sha)."""
    cu.check_path(path)
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except ValueError as e:
        raise A6Error(f'{path}: {e}')
    if not isinstance(d, dict) or d.get('format') != SCHEMA_GIO or d.get('ok') is not True:
        raise A6Error(f'{path}: not a {SCHEMA_GIO} file with ok true')
    sh = d.get('sha256') if isinstance(d.get('sha256'), dict) else {}
    if sh.get('frames') != frames_sha or sh.get('esp32') != log_sha:
        raise A6Error(f'{path}: the clock file is of another frames.csv or another log (sha256)')
    return _num(d.get('center_s'), f'{path} center_s'), _num(d.get('half_s'), f'{path} half_s', 0.0)


def yaw_run(g, ses, L, D):
    """Heading-channel data of the frames of one scored run (a c8b_a6_goc_v1 file g, its session, its log L, the clock
    offset D): t, e = Kabsch heading of the lifted corners (no A6 correction) minus the IMU yaw (game RV:
    c8_u_imu.rv_samples, interp_yaw at t + D), the IMU yaw pim, omega, the body acceleration a, u = R^T (T - C) /
    (H - hbar), and ok: the frames with an IMU yaw, omega and a (no window across a gap). The whole turns between the
    two unwrapped headings are a constant of the run: e is taken within pi of its first usable value."""
    if g.get('schema') != SCHEMA_GOC or g.get('session_sha256') != ses['sha256']:
        raise A6Error(f'dong_huong {g.get("luot_id")!r}: not a {SCHEMA_GOC} file of this session')
    t = np.array([k[1] for k in g['khung']], float)
    P = np.array([k[2] for k in g['khung']], float).reshape(-1, 4, 2)
    if len(t) < 2 * SG_HALF + 1 or np.any(np.diff(t) <= 0) or not np.all(np.isfinite(P)):
        raise A6Error(f'dong_huong {g.get("luot_id")!r}: too few frames, times not increasing or corners not finite')
    C, H, hb, side = ses['C'], ses['H'], ses['hbar'], ses['side']
    psi = np.unwrap(np.array([kabsch_psi(q, side) for q in P]))
    w = omega_of(t, P, side)
    Tc = P.mean(1)
    a = acc_of(t, Tc)
    S = cu.rv_samples(L)
    pim = cu.interp_yaw(S, t + float(D))
    c_, s_ = np.cos(psi), np.sin(psi)
    dfl = (Tc - C) / (H - hb)
    u = np.column_stack([c_ * dfl[:, 0] + s_ * dfl[:, 1], -s_ * dfl[:, 0] + c_ * dfl[:, 1]])
    ok = np.isfinite(pim) & np.isfinite(w) & np.isfinite(a)
    e = psi - pim
    if ok.any():
        e0 = e[np.flatnonzero(ok)[0]]
        e = e0 + _wrap(e - e0)
    tb = S.get('t_break', math.inf)
    return dict(t=t, e=e, pim=pim, w=w, a=a, u=u, ok=ok, n=int(len(t)), n_ok=int(ok.sum()),
                n_late=S.get('n_late'), n_imu=int(np.isfinite(pim).sum()),
                t_break_log_s=float(tb) if math.isfinite(tb) else None, C=C, H=H, hb=hb, side=side)


def _yaw_fit_runs(blocks):
    """Least squares of the heading-channel difference over the frames of several runs. blocks: per run (e, t, w, pim,
    u). Columns: per run [1, t - mean, omega] (constant, drift, the rest of the clock offset and the shutter skew);
    shared [pim - mean of the run (IMU scale), u_y / 2, -u_x / 2 (the two slopes of design 2a)]. Returns
    (g (2,), cond of the column-scaled design or inf, rms [rad], n)."""
    m, n = len(blocks), int(sum(len(b[0]) for b in blocks))
    X, y, k = np.zeros((n, 3 * m + 3)), np.zeros(n), 0
    for j, (e, t, w, pim, u) in enumerate(blocks):
        sl = slice(k, k + len(e))
        X[sl, 3 * j], X[sl, 3 * j + 1], X[sl, 3 * j + 2] = 1.0, t - t.mean(), w
        X[sl, 3 * m], X[sl, 3 * m + 1], X[sl, 3 * m + 2] = pim - pim.mean(), 0.5 * u[:, 1], -0.5 * u[:, 0]
        y[sl] = e
        k += len(e)
    nrm = np.linalg.norm(X, axis=0)
    if n <= X.shape[1] or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or np.any(nrm == 0):
        return np.full(2, np.nan), math.inf, math.nan, n
    co = np.linalg.lstsq(X, y, rcond=None)[0]
    return co[3 * m + 1:3 * m + 3], float(np.linalg.cond(X / nrm)), float(np.sqrt(np.mean((y - X @ co) ** 2))), n


def dong_huong_build(runs, tam_id, khoa_sha, log=None):
    """c8b_a6_dong_v2 (22/11 night; Liam at CODE_C8B, box: "Lam theo de xuat (Recommended)"; rules locked before the
    run in khoa_truoc_a6_v2.json): U_chuyen.dong from the HEADING channel. runs: {luot_id: yaw_run dict}. Bands: one
    full band [0, a_max] (a_max the largest body acceleration of the usable frames given) and its two halves split at
    the MEDIAN acceleration of those frames (no number chosen). Per target run and band: _yaw_fit_runs on the usable
    frames of EVERY OTHER run in the band; SE of the two slopes by the jackknife over those runs (leave one run out);
    dat when the band has at least N_MIN frames from at least DONG_H_MIN_RUNS other runs and the main fit and every
    jackknife fit are well conditioned (cond <= sqrt(COND_MAX), as kiem_imu). No omega limit: the bound must cover
    every scored frame. g_bar is compared with g_b in gop (kiem_dong): |g_bar - g_b| + 2 |SE|. A run without a usable
    frame is listed in luot_khong_khung_dung: it gives nothing to the fits and is still a target."""
    if not khoa_sha:
        raise A6Error('dong_huong: the lock (--khoa-truoc) is needed: the file is written after it')
    if len(runs) < 2:
        raise A6Error('dong_huong: at least two runs')
    r0 = next(iter(runs.values()))
    for lid, r in runs.items():
        if abs(r['H'] - r0['H']) > TOL_SETUP or abs(r['hb'] - r0['hb']) > TOL_SETUP \
                or abs(r['side'] - r0['side']) > TOL_SETUP or np.max(np.abs(r['C'] - r0['C'])) > TOL_SETUP:
            raise A6Error(f'dong_huong {lid}: another car_pose set-up (L125)')
    # a run without a usable frame (its IMU yaw not usable: c8_u_imu t_break, a hole) gives no frame to the other
    # runs; it is still a target and takes its bound from the others (the locked rule: the frames of every other run)
    if sum(1 for r in runs.values() if r['n_ok'] > 0) < 2:
        raise A6Error('dong_huong: fewer than two runs with a usable frame (IMU yaw, omega, acceleration)')
    a_all = np.concatenate([r['a'][r['ok']] for r in runs.values()])
    a_max, a_tv = float(np.max(a_all)), float(np.median(a_all))
    if not 0.0 < a_tv < a_max:
        raise A6Error(f'dong_huong: median acceleration {a_tv} not inside (0, {a_max})')
    dn = [dict(ten='a', a_lo=0.0, a_hi=a_max, nua=False), dict(ten='a_thap', a_lo=0.0, a_hi=a_tv, nua=True),
          dict(ten='a_cao', a_lo=a_tv, a_hi=a_max, nua=True)]
    cmax = math.sqrt(COND_MAX)
    luot = {}
    for lid in runs:
        others = [x for x in runs if x != lid]
        luot[lid] = dict(bo_luot=others, dai={})
        for b in dn:
            blocks, ten = [], []
            for x in others:
                r = runs[x]
                m = r['ok'] & (r['a'] >= b['a_lo']) & (r['a'] <= b['a_hi'])
                if m.any():
                    blocks.append((r['e'][m], r['t'][m], r['w'][m], r['pim'][m], r['u'][m]))
                    ten.append(x)
            n = int(sum(len(q[0]) for q in blocks))
            ent = dict(n_khung=n, n_luot=len(blocks), luot_co_khung=ten, g_bar=None, SE=None, trang_thai='thieu')
            if n < N_MIN:
                ent['ly_do'] = f'{n} frames < {N_MIN}'
            elif len(blocks) < DONG_H_MIN_RUNS:
                ent['ly_do'] = f'{len(blocks)} other runs with frames in the band < {DONG_H_MIN_RUNS}'
            else:
                g, cond, rms, _ = _yaw_fit_runs(blocks)
                jk = [_yaw_fit_runs(blocks[:j] + blocks[j + 1:]) for j in range(len(blocks))]
                gs = np.array([q[0] for q in jk])
                cj = max(q[1] for q in jk)
                mb = len(blocks)
                ent.update(cond=cond, cond_jackknife_max=cj, rms_rad=rms)
                if not (math.isfinite(cond) and cond <= cmax and math.isfinite(cj) and cj <= cmax
                        and np.all(np.isfinite(gs)) and np.all(np.isfinite(g))):
                    ent['ly_do'] = f'design ill conditioned (main {cond:.3g}, jackknife {cj:.3g} > {cmax:.3g})'
                else:
                    se = np.sqrt((mb - 1) / mb * np.sum((gs - gs.mean(0)) ** 2, axis=0))
                    ent.update(g_bar=[float(v) for v in g], SE=[float(v) for v in se], trang_thai='dat',
                               g_bo_tung_luot={x: [float(v) for v in q] for x, q in zip(ten, gs)})
            luot[lid]['dai'][b['ten']] = ent
            if log:
                log(f'dong_huong {lid} {b["ten"]}: n {ent["n_khung"]} runs {ent["n_luot"]} g_bar {ent["g_bar"]} '
                    f'SE {ent["SE"]} {ent["trang_thai"]} {ent.get("ly_do", "")}')
    return dict(schema=SCHEMA_DONG_H, kenh=DONG_KENH_H, chia_nua=CHIA_TV, tam_id=tam_id, trang_thai='dat',
                dai_dinh_nghia=dn, luot=luot, khoa_truoc_sha256=khoa_sha,
                a_max_nguon='gia toc than lon nhat tren moi khung dung duoc dua vao (acc_of: khop bac hai +-%d khung)'
                            % SG_HALF,
                a_chia_nguon='trung vi gia toc than cua moi khung dung duoc cua cac luot dua vao',
                cach='e = psi Kabsch 4 goc da nang (chua sua A6) - yaw IMU (game RV, tai t_cam + D cua c8b_gio so bo); '
                     'hoi quy tren moi luot [1, t - tb, omega] va chung [yaw IMU - tb luot (thang), u_y/2, -u_x/2]; '
                     'SE: jackknife bo tung luot; khung dung duoc: co yaw IMU, omega, gia toc huu han; khong loc omega',
                n_khung_luot={k: v['n'] for k, v in runs.items()},
                n_khung_dung_luot={k: v['n_ok'] for k, v in runs.items()},
                luot_khong_khung_dung=[k for k, v in runs.items() if v['n_ok'] < 1], truoc_le=True)


# ── readers ──────────────────────────────────────────────────────────────────────────────────────

def load_a6(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, dict) or d.get('schema') != SCHEMA:
        raise A6Error(f'{path}: not a {SCHEMA} file')
    bad = _a6_loi(d)
    if bad:
        raise A6Error(f'{path}: {bad} (A6C2-6)')
    d['_sha256'] = cu.sha256(path)
    return d


def _a6_loi(a6):
    """A6C2-6: the reason the numbers of an A6 dict are not usable, or None: set-up (plumb, Z_c of 16/11 L125, H > h >
    0, side > 0, rho_max > 0) finite; every bound present finite and >= 0; g_b two finite numbers."""
    def fin(x, lo=None):
        return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and (lo is None or x >= lo)
    try:
        pl = a6.get('plumb_xy_m')
        if not (isinstance(pl, list) and len(pl) == 2 and all(fin(v) for v in pl)):
            return 'plumb_xy_m not two finite numbers'
        H, h, side = a6.get('H_m'), a6.get('h_m'), a6.get('side_m')
        if not (fin(H) and fin(h) and fin(side) and H > h > 0 and side > 0):
            return 'H_m > h_m > 0 and side_m > 0 needed'
        _check_l125(pl, H, 'A6 file')
        if not fin((a6.get('mien') or {}).get('rho_max_m', RHO_MAX_M), 0.0):
            return 'mien.rho_max_m not finite >= 0'
        for k in ('eps_g', 'eps_g_khong_dong', 'eps_g_khong_meo', 'eps_g_khong_meo_khong_dong', 'e_bc_rad'):
            if a6.get(k) is not None and not fin(a6[k], 0.0):
                return f'{k} {a6.get(k)!r} not finite >= 0'
        g = a6.get('g_b')
        if g is not None and not (isinstance(g, list) and len(g) == 2 and all(fin(v) for v in g)):
            return 'g_b not two finite numbers'
        dg = a6.get('dong')
        if isinstance(dg, dict):
            if dg.get('max') is not None and not fin(dg['max'], 0.0):
                return 'dong.max not finite >= 0'
            for tl in (dg.get('theo_luot') or {}).values():
                if any(v is not None and not fin(v, 0.0) for v in (tl or {}).values()):
                    return 'dong.theo_luot value not finite >= 0'
        for cases in ((a6.get('ghep_A3_A6') or {}).get('theo_cho') or {}).values():
            for c in (cases or {}).values():
                if not fin(c.get('dg_norm'), 0.0) or (c.get('A3_rad') is not None and not fin(c['A3_rad'], 0.0)):
                    return 'ghep_A3_A6 case numbers not finite >= 0'
    except (A6Error, AttributeError, TypeError) as e:
        return str(e)
    return None


def _a6_dat(a6):
    return isinstance(a6, dict) and (a6.get('trang_thai') or {}).get('a6') == 'dat' and a6.get('g_b') is not None \
        and a6.get('eps_g') is not None and a6.get('e_bc_rad') is not None and _a6_loi(a6) is None


def ap_dung(a6, tam_id):
    """(trang_thai, ly_do): ('dat', None) only for the plate of the file with A6 dat; another plate or A6 not dat ->
    ('chua_ket_luan', reasons joined by '; ') (contract F3 2.C: trang_thai a string, as CAM c7_chuoi_khung 276-277 and
    NHACTAM c7_nhac_tam 1872 read it; a new plate: T16.8)."""
    why = []
    if not isinstance(a6, dict):
        return 'chua_ket_luan', 'no A6 file'
    if a6.get('tam_id') is None or a6.get('tam_id') != tam_id:
        why.append(f'A6 of plate {a6.get("tam_id")!r}, not {tam_id!r}')
    if not _a6_dat(a6):
        why.append(f'A6 trang_thai {(a6.get("trang_thai") or {}).get("a6")!r}')
    return ('dat', None) if not why else ('chua_ket_luan', '; '.join(why))


def _same_setup(a6, plumb_xy_m, zc_m):
    try:
        pl = np.asarray(plumb_xy_m, float)
        same = bool(pl.shape == (2,) and np.all(np.isfinite(pl))
                    and np.max(np.abs(pl - np.asarray(a6['plumb_xy_m'], float))) <= TOL_SETUP
                    and abs(float(zc_m) - float(a6['H_m'])) <= TOL_SETUP)          # NaN never passes (A6C2-6)
    except (TypeError, ValueError, KeyError):
        same = False
    if not same:
        raise A6Error('plumb point / Z_c are not the car_pose set-up of the A6 file')


def _rho_ok(a6, rho):
    """Every corner of a tag whose centre is at rho within rho_max (sufficient: rho + side / sqrt 2)."""
    return math.isfinite(rho) and rho >= 0 and rho + float(a6['side_m']) / math.sqrt(2.0) <= \
        float((a6.get('mien') or {}).get('rho_max_m', RHO_MAX_M))


def delta_psi_a6(a6, tag_xy_m, tag_psi_rad, plumb_xy_m, zc_m):
    """(dpsi_rad[N], ok[N], why[N]) (contract F3): the heading correction to SUBTRACT from the car_pose tag heading;
    ok false and dpsi NaN when A6 is not dat, the input is not finite or a corner may leave rho_max."""
    _same_setup(a6, plumb_xy_m, zc_m)
    xy = np.asarray(tag_xy_m, float).reshape(-1, 2)
    psi = np.asarray(tag_psi_rad, float).reshape(-1)
    if len(psi) != len(xy):
        raise A6Error('tag_xy_m and tag_psi_rad of different lengths')
    n = len(xy)
    d, ok, why = np.full(n, np.nan), np.zeros(n, bool), [''] * n
    if not _a6_dat(a6):
        return d, ok, [f'a6_{(a6.get("trang_thai") or {}).get("a6")}'] * n
    C = np.asarray(a6['plumb_xy_m'], float)
    for i in range(n):
        if not (np.all(np.isfinite(xy[i])) and math.isfinite(psi[i])):
            why[i] = 'khong_huu_han'
        elif not _rho_ok(a6, float(np.linalg.norm(xy[i] - C))):
            why[i] = 'ngoai_mien'
        else:
            ok[i] = True
    if ok.any():
        d[ok] = dpsi_model(a6['g_b'], xy[ok], psi[ok], C, a6['H_m'], a6['h_m'])
    return d, ok, why


def _u_dong(a6, luot, dai):
    """U_chuyen.dong for the frame: (value or None, reason)."""
    b = a6.get('dong')
    if not isinstance(b, dict):
        return None, 'no dong block'
    if luot is None:
        if b.get('thieu'):
            return None, 'dong: some bands thieu, run / band of the frame not given'
        return b.get('max'), ''
    tl = (b.get('theo_luot') or {}).get(luot)
    if tl is None:
        return None, f'dong: run {luot!r} not in the file'
    names = [dai] if isinstance(dai, str) else list(dai or [])
    if not names:
        if any(v is None for v in tl.values()):
            return None, f'dong: run {luot!r} has thieu bands, band of the frame not given'
        return max(tl.values()), ''
    # A6C2-4: the frame's full band AND its half band (by the edges of dai_dinh_nghia), nothing else
    nua = b.get('nua_cua') or {}
    full = [nm for nm in names if nm in nua]
    if len(names) != 2 or len(full) != 1 or (set(names) - set(full)).pop() not in nua[full[0]]:
        return None, f'dong: bands {names} are not one full band and one of its half bands'
    vals = [tl.get(nm) for nm in names]
    if any(v is None for v in vals):
        return None, f'dong: band {names} of run {luot!r} thieu'
    return max(vals), ''


def a6du_rad(a6, rho_m, luot=None, dai=None, meo=True):
    """Residual bound after the correction at a distance rho from the plumb point, or None when A6 is not dat, rho is
    not a valid distance in the domain, or the dong band is thieu. meo False: eps_g without its U_meo_E part (the joint
    replay of u_psi_hai_dau_rad adds the map noise per case)."""
    if not _a6_dat(a6):
        return None
    try:
        rho = float(rho_m)
    except (TypeError, ValueError):
        return None
    if not _rho_ok(a6, rho):
        return None
    full, nodong = ('eps_g', 'eps_g_khong_dong') if meo else ('eps_g_khong_meo', 'eps_g_khong_meo_khong_dong')
    if luot is None and dai is None:
        eps = a6.get(full) if not ((a6.get('dong') or {}).get('thieu')) else None
    else:
        u, _ = _u_dong(a6, luot, dai)
        eps = None if (u is None or a6.get(nodong) is None) else a6[nodong] + u
    if eps is None:
        return None
    return float(eps * rho / (2.0 * (a6['H_m'] - a6['h_m'])) + a6['e_bc_rad'])


def _flag_on(v):
    return v is not False


def _a3_rad(kc, p_xy, plumb):
    """(A3 [rad] or None, reason, place k) of the camera at the floor point p from KIEMCHEO kiem_cheo.json
    (c7_kiem_cheo_v7, S7): A3 carried to the car tag, over the full pair set (du_lieu_du); then every checked region
    holding the end: the 12b corridor of place k (A3_tuong_doi.duong_12b, KIEMCHEO d21f149, HCV3-2: within ban_kinh_m of
    doan_m[k], its A3 flags off -> A3_deg[k]; no P1 flag for a corridor) and the start disc of place k (MIEN_XP_M about
    hang_so_s7.xuat_phat[k], every A3 and P1 flag of k off -> A3_tuong_doi.A3_xp_deg[k]); the largest usable A3 of
    them; a region with a flag on is not usable; no usable region -> thieu (A6C1-10, CODE 20/11)."""
    if not isinstance(kc, dict) or kc.get('version') != KC_VERSION or kc.get('thu'):
        return None, f'kiem_cheo.json not {KC_VERSION} (or a trial)', None
    a3 = kc.get('A3') or {}
    if a3.get('quy_ve_ma_noc') is not True or not isinstance(a3.get('ma_ao'), dict):
        return None, 'A3 not carried to the car tag (quy_ve_ma_noc, ma_ao)', None
    if a3.get('du_lieu_du') is not True or (kc.get('A3_tuong_doi') or {}).get('du_lieu_du') is not True:
        return None, 'A3 over a partial pair set (A3.du_lieu_du / A3_tuong_doi.du_lieu_du not true)', None
    hs = kc.get('hang_so_s7') or {}
    xp, r_xp = hs.get('xuat_phat'), hs.get('MIEN_XP_M')
    pl = hs.get('plumb')
    if not isinstance(xp, dict) or not xp or isinstance(r_xp, bool) or not isinstance(r_xp, (int, float)) \
            or not math.isfinite(r_xp) or not r_xp > 0:
        return None, 'kiem_cheo.json without S7 start regions (--xuat-phat) or MIEN_XP_M not finite > 0', None
    for k, c in xp.items():                          # A6C2-7: a NaN centre would let a far end pass the disc test
        try:
            cc = np.asarray(c, float)
        except (TypeError, ValueError):
            cc = np.full(1, np.nan)
        if cc.shape != (2,) or not np.all(np.isfinite(cc)):
            return None, f'kiem_cheo.json xuat_phat[{k}] not two finite numbers', None
    try:
        pl_ok = isinstance(pl, list) and len(pl) == 2 \
            and bool(np.max(np.abs(np.asarray(pl, float) - np.asarray(plumb, float))) <= TOL_SETUP)
    except (TypeError, ValueError):
        pl_ok = False
    if not pl_ok:
        return None, 'kiem_cheo.json plumb is not the car_pose plumb of the A6 file', None
    p = np.asarray(p_xy, float)
    if p.shape != (2,) or not np.all(np.isfinite(p)):
        return None, 'end point not finite', None
    at = kc.get('A3_tuong_doi') or {}
    tv = (kc.get('P1_tuong_doi') or {}).get('theo_vung') or {}
    cand, bad = [], []                                   # (A3 deg, place, region) usable; reasons of regions hit
    # HCV3-2 (KIEMCHEO d21f149, CODE 20/11): the 12b corridor of place k = every point within ban_kinh_m of the segment
    # doan_m[k]; only the A3 flags of the corridor (no P1 pair is computed there); tried first
    du = at.get('duong_12b')
    if du is not None:
        if not isinstance(du, dict):
            return None, 'A3_tuong_doi.duong_12b not a block', None
        rk = du.get('ban_kinh_m')
        if isinstance(rk, bool) or not isinstance(rk, (int, float)) or not math.isfinite(rk) or not rk > 0:
            return None, f'A3_tuong_doi.duong_12b.ban_kinh_m {rk!r} not finite > 0', None
        for k, seg in (du.get('doan_m') or {}).items():
            try:
                ab = np.asarray(seg, float)
            except (TypeError, ValueError):
                ab = np.full(1, np.nan)
            if ab.shape != (2, 2) or not np.all(np.isfinite(ab)):
                return None, f'duong_12b.doan_m[{k}] not two finite points', None
            if not _dist_seg(p, ab[0], ab[1]) <= float(rk) + 1e-12:
                continue
            flags = [f'duong_12b.{n}[{k}]' for n in ('mot_phuong', 'ngoai_suy', 'cap_hong')
                     if _flag_on((du.get(n) or {}).get(k))]
            v = (du.get('A3_deg') or {}).get(k)
            if flags:
                bad.append(f'A3 at corridor {k}: flag on {flags}')
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                bad.append(f'duong_12b.A3_deg[{k}] {v!r}')
            else:
                cand.append((float(v), k, 'duong_12b'))
    for k, c in xp.items():
        if not np.linalg.norm(p - np.asarray(c, float)) <= float(r_xp) + 1e-12:
            continue
        flags = [f'A3_tuong_doi.{n}[{k}]' for n in ('mot_phuong', 'ngoai_suy', 'cap_hong')
                 if _flag_on((at.get(n) or {}).get(k))]
        flags += [f'P1_tuong_doi.theo_vung.{k}.{n}' for n in ('mot_phuong', 'ngoai_suy', 'cap_hong')
                  if _flag_on((tv.get(k) or {}).get(n))]
        v = (at.get('A3_xp_deg') or {}).get(k)
        if flags:
            bad.append(f'A3 at {k}: flag on {flags}')
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            bad.append(f'A3_xp_deg[{k}] {v!r}')
        else:
            cand.append((float(v), k, 'dia'))
    if cand:                                             # several checked regions hold the point: the largest A3
        v, k, _ = max(cand)
        return math.radians(v), '', k
    if bad:
        return None, '; '.join(bad), None
    return None, ('end outside every checked start disc and 12b corridor: A3 by position not computed (A3loc needs '
                  'the mark table)'), None


def _dist_seg(p, a, b):
    """Distance from p to the segment a-b."""
    ab = b - a
    L2 = float(ab @ ab)
    s = 0.0 if L2 == 0 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + s * ab)))


def u_psi_hai_dau_rad(a6, kc, p1_xy_m, p2_xy_m, tam_id, luot=None, dai=None):
    """HIEUCHUAN u_psi_cuc (CODE 19/11, 20/11): bound [rad] of the camera heading difference of two ends, before the
    margin: {u_rad, trang_thai dat | thieu, ly_do, phan, ghep}.
      a6        the c8b_a6_v2 dict (load_a6) of the plate; side_m, plumb_xy_m, H_m, h_m are read from it (the car_pose
                set-up of the session of the E clips, written by gop from the session: car_side_m, plumb_xy_m,
                tag_h_m / kappa, tag_h_m).
      kc        KIEMCHEO kiem_cheo.json (c7_kiem_cheo_v7, run with --xuat-phat); A3 is checked HERE (version, trial,
                quy_ve_ma_noc, ma_ao, du_lieu_du, plumb, per place mot_phuong / ngoai_suy / cap_hong of A3_tuong_doi
                and P1_tuong_doi.theo_vung; the 12b corridor A3_tuong_doi.duong_12b with its A3 flags only): a failed
                check -> thieu with the reason; the caller needs no gate of its own.
      p1, p2    tag centres [m] on the floor of the two ends (car_pose tag_x, tag_y of the windows; the caller's p_tag),
                in the car_pose frame of the A6 file.
      tam_id    the plate of the run (lan_lap_v1.tam_id of NHACTAM); another plate -> thieu.
      luot      the run id of the target C8 run (the caller's thu_id / luot_id) for U_chuyen.dong (leave-one-run-out);
      dai       the acceleration band name(s) of the end's window (names of dong.dai_dinh_nghia in the A6 file: the full
                band and its half band, from the body acceleration of that window). luot and dai None -> the largest dong
                over runs and bands, thieu if any band is thieu.
    Joint replay of design 2b (CODE 20/11): when both ends are in the checked start disc of the SAME place k and the A6
    file has, for every H32 case of k, the signed change of g and the A3 of that case (KIEMCHEO meo_E 692462f,
    quy_max_deg, unsigned): u = max_case [2 A3_case + |dg_case| (rho1 + rho2) / (2 (H - h))] + A6du without U_meo_E at
    both ends. Otherwise (ends at two places, a case without A3): the separate sum A3(p1) + A3(p2) (kiem_cheo.json by
    position) + A6du(p1) + A6du(p2), the fallback (larger side), named in 'ghep'."""
    why = []
    st, ly = ap_dung(a6, tam_id)
    if st != 'dat':
        why.append(ly)
    ends = []
    for nm, p in (('p1', p1_xy_m), ('p2', p2_xy_m)):
        if not isinstance(a6, dict):
            why.append(f'{nm}: no A6')
            ends.append(None)
            continue
        a3, wa, k = _a3_rad(kc, p, a6.get('plumb_xy_m', [math.nan] * 2))
        try:
            rho = float(np.linalg.norm(np.asarray(p, float) - np.asarray(a6['plumb_xy_m'], float)))
        except (TypeError, ValueError, KeyError):
            rho = math.nan
        if a3 is None:
            why.append(f'{nm}: {wa}')
        du = du_km = None
        if st == 'dat':
            du, du_km = a6du_rad(a6, rho, luot, dai), a6du_rad(a6, rho, luot, dai, meo=False)
            if du is None:
                why.append(f'{nm}: A6du not available (domain or dong band)')
        ends.append(dict(a3=a3, k=k, rho=rho, du=du, du_km=du_km))
    if why:
        return dict(u_rad=None, trang_thai='thieu', ly_do=why)
    e1, e2 = ends
    sep = float(e1['a3'] + e2['a3'] + e1['du'] + e2['du'])
    cases = ((a6.get('ghep_A3_A6') or {}).get('theo_cho') or {}).get(e1['k']) if e1['k'] == e2['k'] else None
    lyg = None
    if e1['k'] != e2['k']:
        lyg = 'two ends at two places: no common H32 case'
    elif not cases:
        lyg = f'no per-case data of place {e1["k"]} in the A6 file'
    elif any(c.get('A3_rad') is None for c in cases.values()):
        lyg = 'a case without A3 (KIEMCHEO quy_max_deg null)'
    elif e1['du_km'] is None or e2['du_km'] is None:
        lyg = 'A6du without U_meo_E not available'
    if lyg is None:
        den = 2.0 * (float(a6['H_m']) - float(a6['h_m']))
        per = {nm: 2.0 * c['A3_rad'] + c['dg_norm'] * (e1['rho'] + e2['rho']) / den for nm, c in cases.items()}
        worst = max(per, key=per.get)
        u = float(per[worst] + e1['du_km'] + e2['du_km'])
        return dict(u_rad=u, trang_thai='dat', ly_do=[], ghep='replay_chung_theo_ca',
                    phan=dict(cho=e1['k'], ca_lon_nhat=worst, ca_rad=per[worst],
                              A6du_khong_meo_rad=[e1['du_km'], e2['du_km']], cong_rieng_rad=sep),
                    gioi_han='A3 theo ca khong dau (quy_max_deg): khong triet dau giua A3 va A6 trong mot ca')
    return dict(u_rad=sep, trang_thai='dat', ly_do=[], ghep='cong_rieng', ly_do_ghep=lyg,
                phan=dict(A3_rad=[e1['a3'], e2['a3']], A6du_rad=[e1['du'], e2['du']]))


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────

def script_state():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'c8b_a6.py'], capture_output=True,
                           text=True, timeout=10)
        return dict(commit=c.stdout.strip(), dirty=bool(s.stdout.strip()), file='tools/c8b_a6.py')
    except (OSError, subprocess.SubprocessError):
        return dict(commit='', dirty=None, file='tools/c8b_a6.py')


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write(path, d):
    cu.check_path(path)
    with open(path, 'x', encoding='utf-8') as f:
        f.write(json.dumps(_clean(d), indent=1, ensure_ascii=False, allow_nan=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('do')
    a1.add_argument('--frames', required=True)
    a1.add_argument('--session', required=True)
    a1.add_argument('--cho', required=True, choices=CHO)
    a1.add_argument('--f15', required=True)
    a1.add_argument('--tam-id', required=True)
    a1.add_argument('--lan-lap-id', required=True)
    a1.add_argument('--luot-id-E', required=True, help='run id (luot_id) of the E clip windows E_dau / E_cuoi (F2)')
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('gop')
    a2.add_argument('--cho-file', action='append', required=True)
    a2.add_argument('--tam-id', required=True)
    a2.add_argument('--tam-cal-id', required=True, help='plate of the caliper numbers (another plate: new plate)')
    a2.add_argument('--lan-lap-cal-id', default=None, help='mounting of the caliper numbers (others add U_chuyen.lap)')
    a2.add_argument('--lap', default=None, help=f'{SCHEMA_LAP} file')
    a2.add_argument('--dong', default=None, help=f'{SCHEMA_DONG} or {SCHEMA_DONG_H} file')
    a2.add_argument('--giu-ngoai', default=None, help=f'{SCHEMA_O} file (camera-IMU held-out check)')
    a2.add_argument('--u-h-chung', default=None, help=f'{SCHEMA_UH} file')
    a2.add_argument('--khoa-truoc', default=None, help='the lock manifest written before the checks (its sha256)')
    a2.add_argument('--kiem-imu', action='append', default=[],
                    help=f'{SCHEMA_IMU} file of a held-out E clip (repeat; every place needs one)')
    a2.add_argument('--session-cham', action='append', default=[],
                    help='session.json of a camera session where the file is used for scoring (repeat; e_bc is the '
                         'largest over the E places and these, da session (A))')
    a2.add_argument('--out', required=True)
    a3 = sub.add_parser('kiem_imu', help='heading channel (camera - IMU) of a held-out E clip')
    a3.add_argument('--frames', required=True)
    a3.add_argument('--session', required=True)
    a3.add_argument('--cho', required=True, choices=CHO)
    a3.add_argument('--f15', required=True)
    a3.add_argument('--log', required=True, help='agv_logger csv of the E clip (IMU yaw)')
    a3.add_argument('--tam-id', required=True)
    a3.add_argument('--luot-id-E', required=True)
    a3.add_argument('--khoa-truoc', required=True)
    a3.add_argument('--out', required=True)
    a4 = sub.add_parser('goc', help='lifted corners of the frames of a scored run (for dong)')
    a4.add_argument('--track', required=True, help='track directory (track.json, frames.csv; dets.npz named there)')
    a4.add_argument('--session', required=True)
    a4.add_argument('--cua-so', required=True, help='F2 cua_so_v1.jsonl (pre / post windows of the run)')
    a4.add_argument('--luot-id', required=True)
    a4.add_argument('--out', required=True)
    a5 = sub.add_parser('dong', help=f'{SCHEMA_DONG} from the corner files of the scored runs')
    a5.add_argument('--goc', action='append', required=True, help=f'{SCHEMA_GOC} file of a run (repeat)')
    a5.add_argument('--tam-id', required=True)
    a5.add_argument('--out', required=True)
    a6_ = sub.add_parser('dong_huong', help=f'{SCHEMA_DONG_H}: U_chuyen.dong from camera heading - IMU yaw of the runs')
    a6_.add_argument('--luot', action='append', nargs=3, required=True, metavar=('GOC', 'LOG', 'GIO'),
                     help=f'a scored run: its {SCHEMA_GOC} file, its agv_logger csv, its preliminary {SCHEMA_GIO} file')
    a6_.add_argument('--tam-id', required=True)
    a6_.add_argument('--khoa-truoc', required=True)
    a6_.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'do':
            ses = read_session(a.session)
            fsha = cu.sha256(a.frames)
            with open(a.frames, newline='', encoding='utf-8') as fh:
                hdr = next(csv.reader(fh), [])
            goc_f15 = not all(c in hdr for c in CORNER_COLS)
            if goc_f15:                                   # CODE_C8B 22/11: corners from F15 goc_m (KIEMCHEO)
                t_all = read_frame_times(a.frames)
                f15 = read_f15(a.f15, ses, fsha, t_all)
                idx, t, P = e_frames_from_f15(f15)
            else:
                idx, t, P, t_all = read_e_frames(a.frames)
                f15 = read_f15(a.f15, ses, fsha, t_all)
            out = do_place(a.cho, idx, t, P, ses, f15, camera_of(ses), a.tam_id, a.lan_lap_id, a.luot_id_E)
            out.update(nguon=[dict(vai='frames', file=os.path.abspath(a.frames), sha256=fsha),
                              dict(vai='session', file=ses['file'], sha256=ses['sha256']),
                              dict(vai='f15', file=f15['file'], sha256=f15['sha256_file']),
                              dict(vai='clip_E', sha256=f15['clip_sha256'])],
                       goc_nguon='f15 khung.goc_m (camera cham tung khung, KIEMCHEO)' if goc_f15 else 'frames.csv c0..c3',
                       script=script_state(), luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            _write(a.out, out)
            print(f'{a.cho}: g_E {out["g_E"]} SE {out["SE_E"]} U_meo {out["U_meo_E"]} n {out["n_khung"]}; '
                  f'{out["trang_thai"]} {"; ".join(out["ly_do"])}')
        elif a.cmd == 'kiem_imu':
            ses = read_session(a.session)
            fsha = cu.sha256(a.frames)
            t_all = read_frame_times(a.frames)
            f15 = read_f15(a.f15, ses, fsha, t_all)
            idx, t, P = e_frames_from_f15(f15)
            tc, pc = read_psi_series(a.frames)
            cu.check_path(a.khoa_truoc)
            out = kiem_imu_do(a.cho, idx, t, P, ses, f15, tc, pc, cu.load_log(a.log), a.tam_id, a.luot_id_E,
                              cu.sha256(a.khoa_truoc))
            out.update(nguon=[dict(vai='frames', file=os.path.abspath(a.frames), sha256=fsha),
                              dict(vai='session', file=ses['file'], sha256=ses['sha256']),
                              dict(vai='f15', file=f15['file'], sha256=f15['sha256_file']),
                              dict(vai='log', file=os.path.abspath(a.log), sha256=cu.sha256(a.log)),
                              dict(vai='khoa_truoc', file=os.path.abspath(a.khoa_truoc),
                                   sha256=cu.sha256(a.khoa_truoc))],
                       script=script_state(), luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            _write(a.out, with_sha(_clean(out)))
            print(f'{a.cho} {a.luot_id_E}: g_yaw {out["g_yaw"]} SE {out["SE_yaw"]} rho {out["rho_m"]} n '
                  f'{out["n_khung"]}; {out["trang_thai"]} {"; ".join(out["ly_do"])}')
        elif a.cmd == 'goc':
            rows, cs_sha = read_cua_so(a.cua_so)
            out = goc_run(a.track, a.session, a.luot_id, rows)
            out.update(nguon=[dict(vai='track', file=os.path.abspath(a.track)),
                              dict(vai='cua_so_v1', file=os.path.abspath(a.cua_so), sha256=cs_sha)],
                       script=script_state(), luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            cu.check_path(a.out)
            with open(a.out, 'x', encoding='utf-8') as f:
                f.write(json.dumps(with_sha(_clean(out)), ensure_ascii=False, allow_nan=False))
            print(f'{a.luot_id}: {out["n_khung"]} frames, {out["n_bo"]} left out, replay {out["replay_max_m"]:.3g} m')
        elif a.cmd == 'dong':
            runs, nguon = {}, []
            for gp in a.goc:
                g = _load(gp, SCHEMA_GOC)
                ses = read_session(g['session_file'])
                if g['luot_id'] in runs:
                    raise A6Error(f'run {g["luot_id"]} given twice')
                runs[g['luot_id']] = (g, ses, camera_of(ses))
                nguon.append(dict(vai=f'goc {g["luot_id"]}', file=os.path.abspath(gp), sha256=cu.sha256(gp)))
            out = dong_build(runs, a.tam_id, log=lambda m: print(m, flush=True))
            out.update(nguon=nguon, script=script_state(),
                       luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            _write(a.out, with_sha(_clean(out)))
            print(f'dong: a_max {out["dai_dinh_nghia"][0]["a_hi"]:.4f} m/s^2, {len(out["luot"])} runs')
        elif a.cmd == 'dong_huong':
            cu.check_path(a.khoa_truoc)
            khoa = cu.sha256(a.khoa_truoc)
            runs, nguon, gio = {}, [dict(vai='khoa_truoc', file=os.path.abspath(a.khoa_truoc), sha256=khoa)], {}
            for gp, lp, op in a.luot:
                g = _load(gp, SCHEMA_GOC)
                lid = g['luot_id']
                if lid in runs:
                    raise A6Error(f'run {lid} given twice')
                ses = read_session(g['session_file'])
                cu.check_path(lp)
                lsha = cu.sha256(lp)
                D, half = read_gio_offset(op, g['frames_sha256'], lsha)
                runs[lid] = yaw_run(g, ses, cu.load_log(lp), D)
                gio[lid] = dict(D_s=D, half_s=half, n_late=runs[lid]['n_late'], n_khung_co_yaw=runs[lid]['n_imu'],
                                t_break_log_s=runs[lid]['t_break_log_s'])
                nguon += [dict(vai=f'goc {lid}', file=os.path.abspath(gp), sha256=cu.sha256(gp)),
                          dict(vai=f'log {lid}', file=os.path.abspath(lp), sha256=lsha),
                          dict(vai=f'gio {lid}', file=os.path.abspath(op), sha256=cu.sha256(op))]
            out = dong_huong_build(runs, a.tam_id, khoa, log=lambda m: print(m, flush=True))
            out.update(gio_luot=gio, nguon=nguon, script=script_state(),
                       luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            _write(a.out, with_sha(_clean(out)))
            dn = out['dai_dinh_nghia']
            print(f'dong_huong: a_max {dn[0]["a_hi"]:.4f} a_tv {dn[1]["a_hi"]:.4f} m/s^2, {len(out["luot"])} runs')
        else:
            places, nguon, shas = [], [], []
            for p in a.cho_file:
                cu.check_path(p)
                with open(p, encoding='utf-8') as f:
                    d = json.load(f)
                if d.get('schema') != SCHEMA_CHO or d.get('mo_hinh') != MO_HINH:
                    raise A6Error(f'{p}: not a {SCHEMA_CHO} file of the model {MO_HINH}')
                places.append(d)
                shas.append(cu.sha256(p))
                nguon.append(dict(vai=f'cho {d["cho"]}', file=os.path.abspath(p), sha256=shas[-1]))
                nguon += [dict(n, vai=f'{d["cho"]} {n.get("vai")}') for n in d.get('nguon') or []]
            if len({p['cho'] for p in places}) != len(places):
                raise A6Error('a place given twice')
            src = {}
            for key, path, sch in (('lap', a.lap, SCHEMA_LAP), ('dong', a.dong, (SCHEMA_DONG, SCHEMA_DONG_H)),
                                   ('giu_ngoai', a.giu_ngoai, SCHEMA_O), ('u_h', a.u_h_chung, SCHEMA_UH)):
                src[key] = _load(path, sch) if path else None
                if path:
                    nguon.append(dict(vai=key, file=os.path.abspath(path), sha256=cu.sha256(path)))
            khoa, khoa_d = None, None
            if a.khoa_truoc:
                cu.check_path(a.khoa_truoc)
                khoa = cu.sha256(a.khoa_truoc)
                nguon.append(dict(vai='khoa_truoc', file=os.path.abspath(a.khoa_truoc), sha256=khoa))
                try:
                    with open(a.khoa_truoc, encoding='utf-8') as f:
                        khoa_d = json.load(f)
                except ValueError:
                    khoa_d = None                                       # not JSON: kiem_o reports no sets
            cam_cham = []
            for sp in a.session_cham:
                ses_c = read_session(sp)
                cam_cham.append(set_up_of_session(ses_c))
                nguon.append(dict(vai='session_cham', file=ses_c['file'], sha256=ses_c['sha256']))
            rows = None
            if src['giu_ngoai'] is not None:                            # A6C2-2: the rows the cells were built from
                rn = [x for x in src['giu_ngoai'].get('nguon') or [] if isinstance(x, dict) and x.get('vai') == 'rows']
                if len(rn) != 1 or not isinstance(rn[0].get('file'), str):
                    raise A6Error(f'{a.giu_ngoai}: nguon has no single rows file')
                cu.check_path(rn[0]['file'])
                if cu.sha256(rn[0]['file']) != rn[0].get('sha256'):
                    raise A6Error(f'{rn[0]["file"]}: sha256 is not the one in {a.giu_ngoai} (rows edited)')
                import c8b_a6_o as om
                rows = om.read_rows(rn[0]['file'])
                nguon.append(dict(vai='giu_ngoai rows', file=os.path.abspath(rn[0]['file']), sha256=rn[0]['sha256']))
            imu = []
            for ip in a.kiem_imu:
                imu.append(_load(ip, SCHEMA_IMU))
                nguon.append(dict(vai='kiem_imu', file=os.path.abspath(ip), sha256=cu.sha256(ip)))
            out = gop(places, a.tam_id, a.tam_cal_id, src['lap'], src['dong'], src['giu_ngoai'], src['u_h'], khoa,
                      script_state(), shas, nguon, lan_lap_cal_id=a.lan_lap_cal_id, giu_ngoai_rows=rows,
                      khoa=khoa_d, cam_cham=cam_cham, kiem_imu=imu)
            out.update(script=script_state(), luc=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            _write(a.out, out)
            print(f'A6 {out["trang_thai"]}; g_b {out["g_b"]} eps_g {out["eps_g"]} e_bc {out["e_bc_rad"]}; '
                  f'{"; ".join(out["ly_do"])}')
    except (A6Error, cu.InputError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f'c8b_a6: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
