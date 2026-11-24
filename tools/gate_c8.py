"""
gate_c8.py: the C8 gate file (format gate_c8_v2), its reader and its two checks, and the set of conditions that decide
a run. CODE_GATE 19/11/2026, design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_rq3_tuong_doi_v6_C8.md (the C8 part
of the relative-frame scoring, after the scoring rules J v6 03_CODE/docs/2026-11-19_J_Quy_tac_cham_O1_v6.md, root
1836fe65; CODE's order of 19/11, Liam at CODE local_6653c962: "duyệt cả 8, giao việc các phiên phụ theo cách nhanh",
"Code ngay (Recommended)"). gate_c8_v1 (17/11) is refused.

One JSON file written once after the C8 session by `gate_c8.py ghep` (GATE assembles every block from its sources
F10-F19 and reads the C8B files directly; c8b_u cong no longer writes the gate: CODE 19/11, contract with C8B) through
write_gate; its SHA-256 (of the bytes) is "the gate file SHA" carried by every run record. Two checks (design
section 3):

  check(d)              the content alone: types, keys, the chain of every C8 cell recounted (reruns cap, M2), the
                        runs of each level rebuilt from the chains, the three kinds of evidence and the gate levels
                        recomputed from the copied evidence, Uhat recomputed from the schedule (section 4), the
                        so_do conditions recomputed (section 5). Every read (read_gate_c8).
  check_nguon(d, goc)   the sources under the C8 folder goc: every {duong_dan, sha256} hashed again (a path inside
                        goc), and the plan, scoring files, time files, premise file, U-standard dossier and U
                        manifests read back and compared with what the gate copied. When the gate is written
                        (write_gate) and when it is registered for a campaign (after C8).

Any difference refuses the whole file (GateFileError). A case J v6 does not decide that touches the lock of C8 (a
Uhat that cannot be told from half the threshold within the budget, J section 0b) is refused with "CAN LIAM QUYET"
unless the gate carries Liam's choice (quyet_liam).

Rule 2b decided by the measured value (CODE 23/11, 03_CODE/docs/2026-11-23_Chot_luat_mat_na_quyet_bang_so_do_v1.md,
root 7b6b7652; Liam at CODE, verbatim: "duyệt bảng luật mới"): every condition decides a run. A condition of the
level's mask (so_do(level): Uhat above half the threshold at C8, and G5 at F08) decides by its measured value q, U
not added (so_do_state; the branch 'so_do', the three-branch state kept as state_ba_nhanh); the others by the three
branches with U. The mask (which condition takes which branch) is locked once at C8 as before. Before 23/11 the
masked conditions were only reported ("chỉ báo cáo"), the gate_v3.3 field report_only: refused now.

run_so_do(level, platform, stage, campaign): the conditions that decide by q:
  SIM                                   none
  HW at a gate level                    so_do(level)
  HW at another task (K1, K2, ...)      none (Liam 17/11, "Không áp, cả năm quyết")
  HIL, stage hil_check, campaign B-*    so_do(level) (S05 4.5)
  HIL otherwise                         none
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import csv
import io
import math
import os
import posixpath
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'fwsim'), os.path.join(HERE, '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import o1_gate                                                          # noqa: E402
from o1_feasibility import Schedule, A as FIG8_A                        # noqa: E402
from o1_tasks import FIG8                                               # noqa: E402

FORMAT = 'gate_c8_v2'
OLD_FORMATS = ('gate_c8_v1',)
MASKABLE = ('g2', 'g3', 'g5')           # O1 3.9 rule 2b: G1, G4 always take the three branches
NHANH_SO_DO = 'so_do'                   # the branch of a masked condition: decided by q (CODE 23/11, Liam)
GATE_VERSION_SO_DO = 'gate_v3.4'        # a run gate with the so_do conditions of its level (o1_gate.py is in the
# S04 SIM_SOURCES lock, fwsim/s04_ga.py, so the branch is applied here on its gate_v3.2 / v3.3 dict)
CONTROLLERS = ('PID', 'LADRC')
U_T_BRANCHES = ('bo_tick', 'giu_tick')
B_CAMPAIGNS = ('B-PID', 'B-LADRC')
MASK_HIL_STAGE = 'hil_check'            # S05 4.5: the HIL candidate check of procedure B
PLATFORMS = ('SIM', 'HIL', 'HW')
REL_TOL = 1e-9                          # recomputed numbers: relative difference allowed
LICH = 'o1_feasibility.Schedule'        # the schedule of the comparison (G-7, owner CODE_GATE)
J_V6_SHA256 = '98767aa41ac176363795ca87d37f8a7efb4944e6d8d2925d4da293ceafd29d00'   # = c7_score.J_V6_SHA256 (tested)
SO_DO_KEYS = ('nut', 'q', 'q_chung')   # q_so_do of a so_do condition (SD1-1, SD1-3; Liam 24/11 "Sửa theo bảng")
G5_F08 = 'O1-F08'                       # J v6 section 6 (Liam 19/11 "không làm bù"): G5 at F08 always in the mask,
# decided by q since 23/11 (CODE, Liam "duyệt bảng luật mới": no longer "always report only")

# Uhat (design section 4)
DELTA0_S = 1e-3                         # first grid step
DELTA_MIN_S = 1e-6                      # no refinement step below this
BUDGET = 10 ** 7                        # evaluations of (V, W, u, r) per level, the first grid included
UH_STATES = ('quyet', 'so_do', 'thieu', 'bien_nguong')     # so_do: Uhat above half (chi_bao_cao before 23/11)

# Chains of the C8 cells (design section 6; J v6 section 5)
OUTCOMES = ('dat', 'khong_dat', 'chua_ket_luan', 'hong')
LOST_KINDS = ('thao_tac', 'he_do', 'cua_so_tham_chieu', 'duong_lenh')
THAO_TAC = ('boot', 'khoa_sieu_am', 'loi_nap_gain')     # J v6 section 5 operations (score.json thao_tac, c8b_gio)
D1_STATES = ('dat', 'truot', 'khong_kiem')                              # F11: c8b_gio_v1.kiem_gio_d1.trang_thai
RERUNS_MAX = 3                          # J v6 section 5: the first run + 3 reruns per kind
CELL_STATES = ('co_luot', 'dong_tran', 'thieu_u_t',   # thieu_u_t: GC9-6 (Liam 20/11), the last run has no U_t
               'hong_theo_liam')   # Liam 22/11 (C8 contract section 3): last run lost to the measurement, not rerun
EVID_KEYS = ('e_do_lo_vuot_D_moi_nut', 'e_do_xac_dinh_moi_nut', 'g4_can_huu_han_mien_hop_moi_nut')

# U standard (design section 8; J v6 section 6)
CHUAN_U_LOAI = 'uoc_luong_thuc_nghiem'
CHUAN_U_LE = 1.25
CHUAN_U_DK = ('khoa_kiem_truoc_giu_ngoai', 'kiem_yaw_theo_hieu_mien_quyet', 'kiem_vi_tri_rieng',
              'luat_giu_mask_theo_phan_do')
MAN_TRAP = ('alpha', 'mot_khung')
TASK_K = 'K'

KE_HOACH_FORMAT = 'ke_hoach_c8_v1'
GIO_FORMAT = 'c8b_gio_v1'
TIEN_DE_FORMAT = 'tien_de_buoi_v1'
MANIFEST_FORMAT = 'u_moc_manifest_v1'
HO_SO_FORMAT = 'ho_so_c8_v2'   # v2 (CODE 24/11): khoa_kiem_luc per C8 session (the S7 lock of each session)                             # F17: the U-standard dossier, written by ghep (GATE)
KIEM_CHEO_VERSION = 'c7_kiem_cheo_v7'                    # KIEMCHEO 19/11: kiem_cheo.json with its khoa_s7 block
KK_KEYS = ('buoi', 'kiem_cheo', 'khoa_s7', 'git_commit', 'git_time')   # one khoa_kiem_luc entry per C8 session
S7_SAME = ('schema', 'hang_so_s7')    # the part of every session's khoa_s7 file that must be the same (CODE 24/11)
LUOI_STATES = ('cham', 'chua_ket_luan_khong_nut', 'hong_he_do')     # c8b_luoi trang_thai_luot (F11, CODE_C8B)
MOC_KEYS = ('U_moc_dong_xy_m', 'U_moc_tham_xy_m', 'U_moc_dong_psi_rad', 'U_moc_tham_psi_rad')
MOC_NGUON = ('bo_lay_mau', 'car_pose')         # the part of the U manifest source the same for every scoring (CAM, ghep)
MANIFEST_NGUON = MOC_NGUON + ('tap_moc', 'lich')   # CODE 22/11: tap_moc is the result of lich per camera segment
LICH_VERSION = 'c7_lich_v3'                        # the schedule file of c7_measure lich (CAM v3)

TOP_KEYS = ('format', 'levels', 'muc_kha_thi', 'muc_bo_theo_liam', 'quyet_liam', 'u_t_branch', 'lich_doi_chieu',
            'bo_cham', 'ke_hoach_c8', 'thu_tu_clip', 'khoa_c8b', 'o_c8', 'chuan_u', 'tien_de_c8', 'runs_c8',
            'muc_loai_e_do', 'chung_cu_muc', 'dieu_2b', 'camera', 'luot_k')
THU_TU_COLS = ('clip', 'buoc', 'log_pi')                # F18 (1): thu_tu_clip.csv (C8 plan v1 line 87; CODE 19/11)
RUN_CODE = re.compile(r'^C8-[PL]-(F02|F05|F08)-\d+$')    # a figure-8 run code of the session guide (cells of rule 2b);
# rows of clips H (H_*) and calibration runs (HC_*) are not runs of a cell
RUN_CODE_K = re.compile(r'^C8-[PL]-(K1|K2)-\d+(_lai\d+)?$')   # GC9-2 (CODE 20/11): the K runs of C8, held out too
# C8 session 2 (20/11, thu_tu_clip.csv of CAM): the still clip before a K run (car parked at the K start) is written
# <K run code>_khoa; it is no run (no log, no scoring), so neither a cell nor a K run of the gate
CLIP_KHOA_K = re.compile(r'^C8-[PL]-(K1|K2)-\d+_khoa$')
KHOA_FORMAT = 'c8b_khoa_v5'                              # F18 (4): the C8b lock written before the held-out parts
CTRL_LOG = {'PID': 'PID', 'LADRC': 'ADRC'}               # the '#gain ... ctrl=' name of the logger (c8b_gio controller)
SESSION_KIND = 'c7_session'                              # a camera session file (c7_measure calibrate)
# F18 (3): tien_de_buoi_v1 vai_tro_clip_h of ONE camera segment (CODE 20/11 after C8 session 2, three segments: the
# iPhone was taken off twice; one premise file per segment, roles dau1, dau2, cuoi; J3-5 per segment that holds runs)
TD_ROLES = ('dau1', 'dau2', 'cuoi')
BAND_KEYS_C8B = ('u_lo_m_s', 'u_hi_m_s', 'P4_m', 'A4_deg', 'trang_thai')   # c8b_khoa_v5 M5b (CODE 19/11)
BAND_STATES_C8B = ('dat', 'gioi_han', 'thieu')           # C8B term states (data contract 2.G (3))
DRIFT_KIND = 'c7_calib_drift'                            # F18 (2): drift.json of c7_calib_drift so (CODE 19/11)
LEVEL_KEYS = ('a_m', 'u_p', 'u_psi', 'u_t_c8_s', 'uhat', 'so_do', 'ly_do_so_do')
UH_KEYS = ('uhat', 'uhat_duoi', 'delta_cuoi_s', 'so_lan_tinh', 'trang_thai')


class GateFileError(ValueError):
    pass


class CanLiamChon(GateFileError):
    """CAN LIAM QUYET at a bien_nguong Uhat (J section 0b): ghep stops; .phieu is the decision sheet (phieu_chon_v1)
    with both choices of every open case, written by main beside --out (CODE 24/11)."""

    def __init__(self, msg: str, phieu: dict):
        super().__init__(msg)
        self.phieu = phieu


# ── small readers ──────────────────────────────────────────────────────────────────────────────────────────────

def _num(x, what: str, positive: bool = False) -> float:
    """A JSON number (int / float, not bool), finite, >= 0 (> 0 when positive)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise GateFileError(f'{what} {x!r}: a finite number')
    try:
        v = float(x)
    except (OverflowError, ValueError):                 # GC1-5: an integer beyond the float range
        raise GateFileError(f'{what}: a number beyond the float range')
    if not math.isfinite(v):
        raise GateFileError(f'{what} {x!r}: a finite number')
    if v < 0.0 or (positive and v == 0.0):
        raise GateFileError(f'{what} {x!r}: {"> 0" if positive else ">= 0"}')
    return v


def _num_or_null(x, what: str) -> Optional[float]:
    return None if x is None else _num(x, what)


def _isnum(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def _close(a, b) -> bool:
    """Equal within REL_TOL, both finite (GC1-2: inf never matches); two nulls are equal. GC6-15: both JSON numbers
    (not bool, not text) before the tolerance; anything else never matches."""
    if a is None or b is None:
        return a is None and b is None
    if not (_isnum(a) and _isnum(b)):
        return False
    try:
        a, b = float(a), float(b)
    except OverflowError:
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-300)


def _obj(x, keys, what: str, extra_ok: bool = False) -> dict:
    if not isinstance(x, dict) or (set(x) != set(keys) if not extra_ok else not set(keys) <= set(x)):
        raise GateFileError(f'{what}: an object with {"at least " if extra_ok else ""}keys {list(keys)}, got '
                            f'{sorted(x) if isinstance(x, dict) else x!r}')
    return x


def _text(x, what: str) -> str:
    if not isinstance(x, str) or not x:
        raise GateFileError(f'{what} {x!r}: a non-empty text')
    return x


def _bool(x, what: str) -> bool:
    if not isinstance(x, bool):
        raise GateFileError(f'{what} {x!r}: true or false')
    return x


def _hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(ch in '0123456789abcdef' for ch in x)


def _bad_rel(p: str) -> bool:
    """GC6-14: not a plain relative path: absolute, a drive or UNC prefix, an empty / '.' / '..' component."""
    q = p.replace('\\', '/')
    return (os.path.isabs(p) or q.startswith('/') or bool(re.match(r'^[A-Za-z]:', q))
            or bool(os.path.splitdrive(p)[0]) or any(c in ('', '.', '..') for c in q.split('/')))


def _ref(x, what: str) -> dict:
    """A source reference {duong_dan (relative to the C8 folder, no '..', no drive), sha256}."""
    _obj(x, ('duong_dan', 'sha256'), what)
    p = _text(x['duong_dan'], f'{what}.duong_dan')
    if _bad_rel(p):
        raise GateFileError(f'{what}.duong_dan {p!r}: a path relative to the C8 folder, inside it (no "..", no drive)')
    if not _hex64(x['sha256']):
        raise GateFileError(f'{what}.sha256 {x["sha256"]!r}: 64 lowercase hex digits')
    return x


def _liam(x, what: str) -> dict:
    """Liam's words recorded where he said them: {ngay YYYY-MM-DD, phien, nguyen_van} (A1)."""
    _obj(x, ('ngay', 'phien', 'nguyen_van'), what, extra_ok=True)
    try:
        if not isinstance(x['ngay'], str) or dt.date.fromisoformat(x['ngay']).isoformat() != x['ngay']:
            raise ValueError
    except ValueError:
        raise GateFileError(f'{what}.ngay {x["ngay"]!r}: YYYY-MM-DD')
    _text(x['phien'], f'{what}.phien')
    _text(x['nguyen_van'], f'{what}.nguyen_van')
    return x


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    """sha256 of a list of frame ids as HIEUCHUAN / CAM state it (ids_sha: the whole-number ids joined by commas, no
    brackets; CODE 19/11, CT2-6)."""
    ids = list(ids)
    if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise GateFileError(f'frame ids {ids[:5]!r}: whole numbers')
    return hashlib.sha256(','.join(str(i) for i in ids).encode()).hexdigest()


# ── the schedule, its Lipschitz constants, Uhat (design section 4) ─────────────────────────────────────────────

def _tau_vec(sch: Schedule, t: np.ndarray):
    """tau(t), tau_dot(t) of Schedule.tau, vectorised (same branches)."""
    w0, tr, tc = sch.w0, sch.Tr, sch.Tcmd
    t = np.asarray(t, dtype=float)
    tau = np.where(t <= 0.0, 0.0,
          np.where(t < tr, w0 * t * t / (2.0 * tr),
          np.where(t <= tc - tr, w0 * tr / 2.0 + w0 * (t - tr),
          np.where(t < tc, 2.0 * math.pi - w0 * (tc - t) ** 2 / (2.0 * tr), 2.0 * math.pi))))
    td = np.where(t <= 0.0, 0.0,
         np.where(t < tr, w0 * t / tr,
         np.where(t <= tc - tr, w0,
         np.where(t < tc, w0 * (tc - t) / tr, 0.0))))
    return tau, td


def schedule_samples(sch: Schedule, t: np.ndarray, a_m: float, amp: float = FIG8_A):
    """V = |p'| tau_dot, W = |dpsi/dtau| tau_dot, u = V + W a / sqrt 2, r = |p(tau)| at times t (figure 8,
    x = A sin tau, y = A/2 sin 2 tau; 0 outside [0, T_cmd], r = 0 there since the figure ends at its start)."""
    tau, td = _tau_vec(sch, t)
    dx, dy = amp * np.cos(tau), amp * np.cos(2.0 * tau)
    ddx, ddy = -amp * np.sin(tau), -2.0 * amp * np.sin(2.0 * tau)
    n2 = dx * dx + dy * dy
    v = np.sqrt(n2) * td
    w = np.abs(dx * ddy - dy * ddx) / n2 * td
    r = np.hypot(amp * np.sin(tau), amp / 2.0 * np.sin(2.0 * tau))
    t = np.asarray(t, dtype=float)
    r = np.where((t <= 0.0) | (t >= sch.Tcmd), 0.0, r)       # GC6-10: exactly 0 where the schedule stands still
    return v, w, v + w * a_m / math.sqrt(2.0), r


def lipschitz(sch: Schedule, a_m: float, amp: float = FIG8_A) -> Dict[str, float]:
    """Analytic upper bounds of the time Lipschitz constants of V, W, u, r (design section 4): m = A sqrt7 / 4 the
    least |p'| (|p'|^2 = 4c^2 - 3c + 1 with c = cos^2 tau, least 7/16 at c = 3/8), |p'| <= sqrt2 A,
    |p''| <= sqrt5 A, |p'''| <= sqrt17 A, |tau_dot| <= w0, |tau_ddot| <= w0 / T_r."""
    w0, tr = sch.w0, sch.Tr
    m = amp * math.sqrt(7.0) / 4.0
    l_v = math.sqrt(5.0) * amp * w0 ** 2 + math.sqrt(2.0) * amp * w0 / tr
    k_max = math.sqrt(5.0) * amp / m
    dk_max = math.sqrt(17.0) * amp / m + 10.0 * amp * amp / (m * m)
    l_w = dk_max * w0 ** 2 + k_max * w0 / tr
    return dict(L_V=l_v, L_W=l_w, L_u=l_v + a_m / math.sqrt(2.0) * l_w, L_r=math.sqrt(2.0) * amp * w0)


def _slide(x: np.ndarray, w: int, fn) -> np.ndarray:
    """out[i] = fn(x[i : i + w]) for i = 0 .. n - w, fn = np.maximum or np.minimum (van Herk / Gil-Werman, O(n))."""
    n = len(x)
    if w < 1 or w > n:
        raise ValueError('window')
    fill = -np.inf if fn is np.maximum else np.inf
    nb = -(-n // w)
    xp = np.full(nb * w, fill)
    xp[:n] = x
    b = xp.reshape(nb, w)
    g = fn.accumulate(b, axis=1).ravel()                                   # prefix within a block
    h = fn.accumulate(b[:, ::-1], axis=1)[:, ::-1].ravel()                 # suffix within a block
    i = np.arange(n - w + 1)
    return fn(h[i], g[i + w - 1])


def check_bands(bands, what: str) -> List[tuple]:
    """A table 'hang_theo_u' [{u_lo, u_hi, gia_tri}] of closed bands [u_lo, u_hi] (m/s), adjoining, the first at
    0, increasing, the last finite; gia_tri >= 0 or null (no number for that band). Empty list: no u-dependent
    term (value 0 everywhere)."""
    if not isinstance(bands, list):
        raise GateFileError(f'{what}: a list of bands')
    out, prev = [], 0.0
    for i, b in enumerate(bands):
        _obj(b, ('u_lo', 'u_hi', 'gia_tri'), f'{what}[{i}]')
        lo, hi = _num(b['u_lo'], f'{what}[{i}].u_lo'), _num(b['u_hi'], f'{what}[{i}].u_hi')
        if lo != prev or hi <= lo:
            raise GateFileError(f'{what}[{i}]: [u_lo {lo!r}, u_hi {hi!r}] must start where the previous band ends '
                                f'({prev!r}; the first at 0) and be increasing (no gap, no overlap)')
        out.append((lo, hi, _num_or_null(b['gia_tri'], f'{what}[{i}].gia_tri')))
        prev = hi
    return out


def _band_value(bands, lo: np.ndarray, hi: np.ndarray):
    """For u-ranges [lo, hi] per cell: (max value of the bands meeting the range, the range meets a null band,
    the range leaves the table). No table: value 0."""
    val = np.zeros_like(lo)
    null = np.zeros(lo.shape, dtype=bool)
    if not bands:
        return val, null, null.copy()
    out = hi > bands[-1][1]
    for b_lo, b_hi, g in bands:
        meet = (hi >= b_lo) & (lo <= b_hi)
        if g is None:
            null |= meet
        else:
            val = np.where(meet, np.maximum(val, g), val)
    return val, null | out, out


class _Terms:
    """The closed terms of one level and condition: F(t) = c0 + band(u) + cr r(t) + ct V*(t) (or W*)."""

    def __init__(self, cond: str, blk: dict, u_t: float):
        self.cond = cond
        if cond in ('g2', 'g3'):
            up = blk['u_p']
            self.missing = any(up[k] is None for k in ('hang_m', 'he_so_r', 'u_phi_rad'))
            if not self.missing:
                self.c0 = up['hang_m']
                self.cr = up['he_so_r'] + 2.0 * math.sin(min(up['u_phi_rad'], math.pi) / 2.0)
            self.bands = up['_bands']
            self.ct = u_t if cond == 'g3' else 0.0
            self.rate = 'V'
            self.scale = 1.0
        else:
            ups = blk['u_psi']
            self.missing = ups['hang_rad'] is None
            if not self.missing:
                self.c0 = ups['hang_rad']
                self.cr = 0.0
            self.bands = ups['_bands']
            self.ct = u_t
            self.rate = 'W'
            self.scale = 180.0 / math.pi                                  # compared in degrees (THRESHOLDS g5)


GRID_MAX = 10 ** 9                      # GC6-11: no grid size beyond this is even computed (int, allocation)
EPS_T = 1e-12                           # s: above the rounding of grid times (|t| < 1e3 s, double precision)


def grid_size(u_t: float, t_lo: float, t_hi: float, delta: float) -> Tuple[int, int, int]:
    """(m, mp, nc) of a grid of step delta over [t_lo, t_hi] (GC6-10): m rounded OUT (the cell windows E'_j hold
    E_j = [t_j - U_t, t_{j+1} + U_t]), mp rounded IN (every lower-bound sample t_{j+-k}, k <= mp, lies in the true
    window [t_j - U_t, t_j + U_t] even with the rounding of the grid times), nc cells covering [t_lo, t_hi].
    GateFileError when the grid is beyond any budget (GC6-11)."""
    x, y = u_t / delta, (t_hi - t_lo) / delta
    if not (math.isfinite(x) and math.isfinite(y)) or x > GRID_MAX or y > GRID_MAX:
        raise GateFileError(f'U_t {u_t!r} s / span {t_hi - t_lo!r} s on a {delta!r} s grid: beyond any budget')
    m = int(math.ceil(x))
    while m * delta < u_t:
        m += 1
    mp = int(math.floor(x))
    while mp > 0 and mp * delta + EPS_T > u_t:
        mp -= 1
    nc = max(1, int(math.ceil(y)))
    while t_lo + nc * delta < t_hi:
        nc += 1
    return m, mp, nc


def grid_points(u_t: float, t_lo: float, t_hi: float, delta: float) -> int:
    """Evaluations of (V, W, u, r) of one grid (_grid_bounds), known before any allocation (GC6-11)."""
    m, _, nc = grid_size(u_t, t_lo, t_hi, delta)
    return nc + 2 * m + 3


ROUND_REL = 1e-15                       # GC6-10 (round 3): > the relative error of the few float operations of a bound


def _round_out(x, up: bool):
    """GC6-10 (Codex round 3): a bound rounded outward after its float sum and products (every term >= 0, so each of
    the at most 7 roundings is within half an ulp of the result: 7 x 1.1e-16 < ROUND_REL), then one more ulp; a lower
    bound rounded down the same way, never below 0."""
    x = np.asarray(x, dtype=float)
    if up:
        return np.nextafter(x * (1.0 + ROUND_REL), np.inf)
    return np.maximum(np.nextafter(x * (1.0 - ROUND_REL), -np.inf), 0.0)


def _grid_bounds(sch, terms: List[_Terms], lip, a_m, u_t, t_lo, t_hi, delta):
    """Bounds on the cells of a uniform grid of step delta covering [t_lo, t_hi] (design section 4). Returns
    (n_eval, cells: (t_a, t_b) arrays, per condition (F_up per cell, maybe_null per cell, F_low per grid point,
    certain_null))."""
    m, mp, nc = grid_size(u_t, t_lo, t_hi, delta)
    k = np.arange(-(m + 1), nc + m + 2)
    t = t_lo + k * delta
    v, w, u, r = schedule_samples(sch, t, a_m)
    # GC6-10 (Codex round 2): every point of E'_j lies within half the TRUE spacing of the float grid of a sample,
    # rounded out (the spacing of the computed times may exceed the nominal delta by rounding)
    hd = float(np.nextafter(float(np.max(np.diff(t))) / 2.0, math.inf))
    off = m + 1                                                          # index of t_lo
    wu = 2 * m + 2                                                       # E'_j = j - m .. j + 1 + m
    js = off + np.arange(nc)
    vmax = _slide(v, wu, np.maximum)[js - m]
    wmax = _slide(w, wu, np.maximum)[js - m]
    umax = _slide(u, wu, np.maximum)[js - m] + lip['L_u'] * hd
    umin = _slide(u, wu, np.minimum)[js - m] - lip['L_u'] * hd
    r_up = np.maximum(r[js], r[js + 1]) + lip['L_r'] * hd
    # lower: grid points t_j in [t_lo, t_hi] at or after 0 (a frame can be scored there), windows j - mp .. j + mp
    jl = off + np.arange(nc + 1)
    jl = jl[(t[jl] >= 0.0) & (t[jl] <= t_hi)]                           # GC6-10: points of [t_lo, t_hi] only
    wl = 2 * mp + 1
    lv = _slide(v, wl, np.maximum)[jl - mp]
    lw = _slide(w, wl, np.maximum)[jl - mp]
    lu_hi = _slide(u, wl, np.maximum)[jl - mp]
    lu_lo = _slide(u, wl, np.minimum)[jl - mp]
    res = []
    for tm in terms:
        if tm.missing:
            res.append(None)
            continue
        bv, bnull, _ = _band_value(tm.bands, umin, umax)
        rate_up = (vmax + lip['L_V'] * hd) if tm.rate == 'V' else (wmax + lip['L_W'] * hd)
        f_up = _round_out((tm.c0 + bv + tm.cr * r_up + tm.ct * rate_up) * tm.scale, up=True)
        lbv, lbnull, _ = _band_value(tm.bands, lu_lo, lu_hi)
        f_lo = _round_out((tm.c0 + lbv + tm.cr * r[jl] + tm.ct * (lv if tm.rate == 'V' else lw)) * tm.scale, up=False)
        res.append((f_up, bnull, f_lo, bool(lbnull.any())))
    return len(t), (t[js], t[js + 1]), res


def _runs_of(mask: np.ndarray, ta: np.ndarray, tb: np.ndarray) -> List[Tuple[float, float]]:
    """Contiguous runs of the cells where mask holds, as time intervals."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((float(ta[i]), float(tb[j])))
            i = j + 1
        else:
            i += 1
    return out


def uhat_level(level: str, blk: dict, u_t: float, budget: int = BUDGET) -> Dict[str, dict]:
    """Uhat of G2, G3, G5 at a figure-8 level (design section 4) from the level block (a_m, u_p, u_psi checked,
    bands in '_bands') and U_t,C8. Returns {cond: {uhat, uhat_duoi, delta_cuoi_s, so_lan_tinh, trang_thai}}."""
    sch = Schedule(FIG8[level])
    a_m = blk['a_m']
    lip = lipschitz(sch, a_m)
    t_lo, t_hi = -u_t, sch.Tcmd + u_t
    terms = [_Terms(c, blk, u_t) for c in MASKABLE]
    n0 = grid_points(u_t, t_lo, t_hi, DELTA0_S)
    if n0 > budget:                                                      # GC6-11: before any allocation
        raise GateFileError(f'{level}: U_t,C8 {u_t!r} s needs {n0} evaluations on the first grid, above the budget '
                            f'{budget}: Uhat not computed, C8 not locked')
    n, (ta, tb), res = _grid_bounds(sch, terms, lip, a_m, u_t, t_lo, t_hi, DELTA0_S)
    used = n
    out = {}
    for tm, rr in zip(terms, res):
        half = 0.5 * o1_gate.THRESHOLDS[tm.cond]
        if rr is None or rr[3]:                                          # a closed term missing / a null band met
            out[tm.cond] = dict(uhat=None, uhat_duoi=None, delta_cuoi_s=DELTA0_S, so_lan_tinh=n, trang_thai='thieu')
            continue
        f_up, mnull, f_lo, _ = rr
        up, lo, delta, cnt, st = float(f_up.max()), float(f_lo.max()), DELTA0_S, 0, None
        # GC6-9: a cell stays alive while it may exceed half the threshold (only until the lower bound shows it) or
        # may meet a null band (until refined away or found certain); so_do for exceeding needs both settled
        alive = mnull | ((f_up > half) if lo <= half else False)
        maybe_null = bool(mnull.any())
        rest = float(f_up[~alive].max()) if (~alive).any() else -math.inf
        ivs = _runs_of(alive, ta, tb)
        while True:
            if not ivs:
                st = 'so_do' if lo > half else 'quyet'
                break
            if lo > half and not maybe_null:
                st = 'so_do'
                break
            d2 = delta / 2.0
            if d2 < DELTA_MIN_S or used + sum(grid_points(u_t, a, b, d2) for a, b in ivs) > budget:
                # GC7-1: the lower bound above half already decides so_do; only cells that may meet a null
                # band are left: the number is not known (uhat null), the mask is
                st = 'so_do' if lo > half else 'bien_nguong'
                break
            new_ivs, ups = [], [rest]
            certain_null, maybe_null = False, False
            for a, b in ivs:
                k, (sa, sb), rr2 = _grid_bounds(sch, [tm], lip, a_m, u_t, a, b, d2)
                used += k
                cnt += k
                fu, mn, fl, cn = rr2[0]
                certain_null |= cn
                maybe_null |= bool(mn.any())
                lo = max(lo, float(fl.max())) if len(fl) else lo
                al = mn | ((fu > half) if lo <= half else False)
                if (~al).any():
                    ups.append(float(fu[~al].max()))
                if al.any():
                    ups.append(float(fu[al].max()))
                new_ivs.extend(_runs_of(al, sa, sb))
                rest = max(rest, float(fu[~al].max()) if (~al).any() else -math.inf)
            delta = d2
            up = max(ups)
            ivs = new_ivs
            if certain_null:
                st = 'thieu'
                break
        if st == 'thieu':
            out[tm.cond] = dict(uhat=None, uhat_duoi=None, delta_cuoi_s=delta, so_lan_tinh=n + cnt, trang_thai='thieu')
        else:
            unknown = st == 'so_do' and maybe_null and bool(ivs)          # GC7-1: may still be missing
            out[tm.cond] = dict(uhat=None if unknown else up, uhat_duoi=lo, delta_cuoi_s=delta, so_lan_tinh=n + cnt,
                                trang_thai=st)
    return out


def so_do_of(level: str, uh: Dict[str, dict], quyet: Optional[dict] = None) -> Tuple[Tuple[str, ...], dict]:
    """(conditions decided by the measured value in MASKABLE order, their reasons) at a level (design section 5; the
    mask of rule 2b, report only before 23/11). A bien_nguong condition needs Liam's choice in quyet (J v6 section
    0b), else GateFileError CAN LIAM QUYET."""
    ro, why = [], {}
    for c in MASKABLE:
        rs = []
        st = uh[c]['trang_thai']
        if level == G5_F08 and c == 'g5':
            # GC6-7: J v6 section 6 puts G5 at F08 in the mask before any look at Uhat (Liam 19/11), Uhat kept for the
            # report; no choice of Liam is needed nor taken for it (decided by q since 23/11, CODE, Liam)
            if (quyet or {}).get(c) is not None:
                raise GateFileError(f'quyet_liam.{level}.{c}: not needed, J v6 section 6 already puts G5 at F08 in '
                                    'the mask (so_do)')
            ro.append(c)
            why[c] = ['liam_g5_f08']
            continue
        if st == 'so_do':
            rs.append('vuot_nua_nguong')
            if uh[c].get('uhat') is None:
                rs.append('thieu_chua_phan_giai')            # GC7-1: a null band may still be met
        elif st == 'thieu':
            rs.append('thieu')
        elif st == 'bien_nguong':
            q = (quyet or {}).get(c)
            if q is None:
                raise GateFileError(f'{level} {c}: Uhat cannot be told from half the threshold within the budget '
                                    f'(bien_nguong): J v6 does not decide the mask, C8 is not locked; CAN LIAM QUYET '
                                    f'(J section 0b), recorded in quyet_liam.{level}.{c} {{chon, ngay, phien, '
                                    'nguyen_van}')
            if q['chon'] == 'so_do':
                rs.append('quyet_liam')
        if rs:
            ro.append(c)
            why[c] = rs
    return tuple(ro), why


# ── the branch so_do on a run gate (CODE 23/11, Liam "duyệt bảng luật mới") ──────────────────────────────────────

def _fin(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def so_do_state(q_common, q_union, n_undef_union, threshold: float) -> str:
    """A condition of the mask decided by its measured value q (the maximum over the frames, U not added): khong_dat
    (blocks) when q of the common domain is above the threshold; dat when q of the union is a number not above it
    and no frame of the union is undefined; else chua_ket_luan (blocks: a frame that may lie above it is not
    measured; as cond_state2, a value above it only in the union is chua_ket_luan)."""
    if _fin(q_common) and float(q_common) > threshold:
        return 'khong_dat'
    if not isinstance(n_undef_union, int) or isinstance(n_undef_union, bool) or n_undef_union != 0:
        return 'chua_ket_luan'
    if not _fin(q_union) or float(q_union) > threshold:
        return 'chua_ket_luan'
    return 'dat'


def _nut_giua(nodes_s, ok) -> Optional[int]:
    """CODE 23/11 (2): the node of a so_do condition on a C8b grid, the scored node nearest the middle of the grid
    ((first + last offset) / 2, distances to 1 us, the lower index on a tie; c8b_luoi.nut_giua, a test keeps the two
    equal)."""
    scored = [i for i in range(len(ok or [])) if ok[i]]
    if not scored:
        return None
    if not nodes_s:
        c = len(ok) // 2
        return min(scored, key=lambda i: (abs(i - c), i))
    mid = 0.5 * (float(nodes_s[0]) + float(nodes_s[-1]))
    # SD1-2 (Codex 24/11): distances rounded to 1 us, so two nodes the same distance from the middle tie and the lower
    # index is taken whatever the float rounding (c8b_luoi KEY_DEC)
    return min(scored, key=lambda i: (round(abs(float(nodes_s[i]) - mid), 6), i))


def q_common_of(b: dict):
    """q of the common domain of a condition block (its 'chung' block, B81-3), else q (one domain)."""
    ch = b.get('chung')
    return ch.get('q') if isinstance(ch, dict) else b.get('q')


def q_so_do_default(b: dict) -> dict:
    """q_so_do of a condition block without the frames (SD1-1, Codex 24/11): q of the defined frames is known only when
    no frame of the domain is undefined (then it is q); else null, and so_do_state gives chua_ket_luan. The HIL
    scorer (o1_gate block) takes this; the camera scorer gives the numbers from its frames."""
    ch = b.get('chung') if isinstance(b.get('chung'), dict) else b
    def known(x):
        nu = x.get('n_undef')
        return x.get('q') if isinstance(nu, int) and not isinstance(nu, bool) and nu == 0 and _fin(x.get('q')) else None
    return dict(nut=None, q=known(b), q_chung=known(ch))


def check_q_so_do(x, b: dict, w: str, grid: bool) -> dict:
    """A q_so_do block {nut, q, q_chung} of a so_do condition checked against its condition block b (SD1-3): nut a
    whole number on a grid, null without; q, q_chung null or finite; q_chung <= q <= the q of the block, q_chung <=
    the q of its common domain, when the two are numbers (the defined frames are some of the frames)."""
    if not isinstance(x, dict) or set(x) != set(SO_DO_KEYS):
        raise GateFileError(f'{w}.q_so_do {x!r}: an object {{nut, q, q_chung}} (SD1-3)')
    if grid != (type(x['nut']) is int):
        raise GateFileError(f'{w}.q_so_do.nut {x["nut"]!r}: a node index on a C8b grid, null without one')
    for k in ('q', 'q_chung'):
        if x[k] is not None and not _fin(x[k]):
            raise GateFileError(f'{w}.q_so_do.{k} {x[k]!r}: null or a finite number')
    ch = b.get('chung') if isinstance(b.get('chung'), dict) else b
    for lo_k, hi_v, what in (('q_chung', x['q'], 'q'), ('q', b.get('q'), 'q of the block'),
                             ('q_chung', ch.get('q'), 'q of the common domain')):
        if _fin(x[lo_k]) and _fin(hi_v) and float(x[lo_k]) > float(hi_v) * (1 + REL_TOL):
            raise GateFileError(f'{w}.q_so_do.{lo_k} {x[lo_k]!r} above {what} {hi_v!r} (SD1-3)')
    return x


def apply_so_do(gate: dict, so_do: Iterable[str], level: str, gate_file_sha256: str,
                q_so_do: Optional[Dict[str, dict]] = None) -> dict:
    """The run gate (o1_gate.gate: gate_v3.2, or gate_v3.3 without a mask; c7_score.build_gate) with the so_do
    conditions of its level: a new gate_v3.4 dict; each condition of so_do keeps its three-branch state as
    state_ba_nhanh, lo and hi, and takes nhanh 'so_do', q_so_do {nut, q, q_chung} (q of the DEFINED frames of the
    union and of the common domain, CODE 23/11 point (4), SD1-1; nut the middle node of a C8b grid, null without one)
    and the state so_do_state(q_chung, q, n_undef of the block); decide, the outcomes as o1_gate.evaluate_outcomes
    (every condition decides; complete when decide is all five). q_so_do: from the scorer (camera: its frames; grid:
    the middle node); missing: q_so_do_default. level, gate_file_sha256: the task and the C8 gate file (or the J
    source) giving so_do; written even when so_do is empty."""
    sd = list(so_do)
    if len(set(sd)) != len(sd) or set(sd) - set(MASKABLE):
        raise GateFileError(f'so_do {sd!r}: distinct conditions of {MASKABLE}')
    sd = [c for c in MASKABLE if c in set(sd)]
    if gate.get('version') not in (o1_gate.GATE_VERSION, o1_gate.GATE_VERSION_V1133):
        raise GateFileError(f'gate version {gate.get("version")!r}: so_do is applied to a {o1_gate.GATE_VERSION} or '
                            f'{o1_gate.GATE_VERSION_V1133} gate')
    if gate.get('report_only'):
        raise GateFileError(f'gate report_only {gate.get("report_only")!r}: report only is no rule since 23/11 '
                            '(CODE, Liam "duyệt bảng luật mới"); score with every condition, then apply so_do')
    if not isinstance(level, str) or not level:
        raise GateFileError(f'level {level!r}: the task of the run')
    if not isinstance(gate_file_sha256, str) or len(gate_file_sha256) != 64 \
            or any(ch not in '0123456789abcdef' for ch in gate_file_sha256):
        raise GateFileError(f'gate_file_sha256 {gate_file_sha256!r}: 64 lowercase hex digits')
    if q_so_do is not None and (not isinstance(q_so_do, dict) or set(q_so_do) != set(sd)):
        raise GateFileError(f'q_so_do {q_so_do!r}: one entry per so_do condition {sd}')
    g = json.loads(json.dumps(gate))                     # a copy; the caller's dict is not changed
    for k in ('report_only', 'required'):
        g.pop(k, None)
    grid = g.get('luoi') is not None
    for c in sd:
        b = g[c]
        x = json.loads(json.dumps(q_so_do[c])) if q_so_do is not None else q_so_do_default(b)
        check_q_so_do(x, b, c, grid)
        b['state_ba_nhanh'] = b['state']
        b['nhanh'] = NHANH_SO_DO
        b['q_so_do'] = x
        b['state'] = so_do_state(x['q_chung'], x['q'], b.get('n_undef'), o1_gate.THRESHOLDS[c])
    g.update(o1_gate.evaluate_outcomes(g, list(g['decide']), o1_gate.CONDS))
    g.update(version=GATE_VERSION_SO_DO, so_do=sd, level=level, gate_file_sha256=gate_file_sha256)
    return g


# ── the level block (writer side) ──────────────────────────────────────────────────────────────────────────────

def _level_terms(e: dict, w: str) -> dict:
    """a_m, u_p, u_psi of a level block checked; bands parsed into '_bands' of a copy."""
    a_m = _num(e['a_m'], f'{w}.a_m', positive=True)
    up = _obj(e['u_p'], ('hang_m', 'he_so_r', 'u_phi_rad', 'hang_theo_u'), f'{w}.u_p')
    ups = _obj(e['u_psi'], ('hang_rad', 'hang_theo_u'), f'{w}.u_psi')
    u_p = {k: _num_or_null(up[k], f'{w}.u_p.{k}') for k in ('hang_m', 'he_so_r', 'u_phi_rad')}
    u_p['_bands'] = check_bands(up['hang_theo_u'], f'{w}.u_p.hang_theo_u')
    u_psi = dict(hang_rad=_num_or_null(ups['hang_rad'], f'{w}.u_psi.hang_rad'),
                 _bands=check_bands(ups['hang_theo_u'], f'{w}.u_psi.hang_theo_u'))
    return dict(a_m=a_m, u_p=u_p, u_psi=u_psi)


def level_uhat(level: str, a_m: float, u_p: dict, u_psi: dict, runs: list) -> Tuple[float, Dict[str, dict]]:
    """(U_t,C8, Uhat of G2, G3, G5) of a gate level from its terms and its eligible C8 runs (level_block)."""
    t = _level_terms(dict(a_m=a_m, u_p=u_p, u_psi=u_psi), level)
    if not runs:
        raise GateFileError(f'{level}: no eligible C8 run')
    u_t = max(_num(r['u_t_s'], f'{level} u_t_s', positive=True) for r in runs)
    return u_t, uhat_level(level, t, u_t)


def level_block(level: str, a_m: float, u_p: dict, u_psi: dict, runs: list, quyet: Optional[dict] = None,
                uh: Optional[Tuple[float, Dict[str, dict]]] = None) -> dict:
    """The dieu_2b.per_level block of a gate level for the writer (C8b script): a_m (m), u_p {hang_m, he_so_r,
    u_phi_rad, hang_theo_u}, u_psi {hang_rad, hang_theo_u} (null for a term without a checked bound), runs the
    eligible C8 runs of the level (runs_c8 entries, u_t_s each); U_t,C8, Uhat and the mask computed here (uh: the
    result of level_uhat on the same inputs, not computed again)."""
    u_t, uh = uh if uh is not None else level_uhat(level, a_m, u_p, u_psi, runs)
    sd, why = so_do_of(level, uh, quyet)
    return dict(a_m=a_m, u_p=u_p, u_psi=u_psi, u_t_c8_s=u_t, uhat=uh, so_do=list(sd), ly_do_so_do=why)


# ── chains, runs, evidence (design sections 6, 7) ──────────────────────────────────────────────────────────────

CHAIN_KEYS = ('run_id', 'clip', 'ket_qua', 'loai_hong', 'kiem_gio_d1', 'dem_truot_kiem_gio', 'file_cham', 'file_gio',
              'file_chung_cu', 'file_so_h', 'file_cham_h1', 'file_gio_h1', 'file_edo', 'file_edo_h1', 'file_log',
              'file_thao_tac', 'thieu_u_t')
CHAIN_FILES = ('file_cham', 'file_gio', 'file_chung_cu', 'file_cham_h1', 'file_gio_h1', 'file_edo',
               'file_edo_h1', 'file_log', 'file_thao_tac')                    # each used by one run only
# GC6-1 (round 3): file_log, the ESP32 log of the run itself (hashed, not the SHA a time file states); GC6-2 (round 3,
# CODE 19/11): file_thao_tac, the score.json the time file took its operations from (thao_tac_nguon; the one source of
# the operator kinds, it wins over any other)


def _chain_entry(x, w: str) -> dict:
    _obj(x, CHAIN_KEYS, w)
    _text(x['run_id'], f'{w}.run_id')
    _text(x['clip'], f'{w}.clip')                       # GC6-1, F18 (1): the clip name of thu_tu_clip.csv
    # GC9-6 (Liam 20/11, phiên CODE: "Lượt đó chưa kết luận (Recommended)"): a grid without U_t at a node (u_max_s
    # null) leaves the run chua_ket_luan, never used to decide
    if _bool(x['thieu_u_t'], f'{w}.thieu_u_t') and x['ket_qua'] not in ('chua_ket_luan', 'hong'):
        raise GateFileError(f'{w}: a run without U_t at a node (thieu_u_t) is chua_ket_luan (or lost), not '
                            f'{x["ket_qua"]!r} (GC9-6)')
    if x['ket_qua'] not in OUTCOMES:
        raise GateFileError(f'{w}.ket_qua {x["ket_qua"]!r}: one of {OUTCOMES}')
    lk = x['loai_hong']
    if (not isinstance(lk, list) or any(k not in LOST_KINDS for k in lk) or len(set(lk)) != len(lk)
            or bool(lk) != (x['ket_qua'] == 'hong')):
        raise GateFileError(f'{w}.loai_hong {lk!r}: kinds of {LOST_KINDS}, each once, non-empty exactly for a lost '
                            'run (ket_qua hong)')
    if x['kiem_gio_d1'] not in D1_STATES:
        raise GateFileError(f'{w}.kiem_gio_d1 {x["kiem_gio_d1"]!r}: one of {D1_STATES} (F11)')
    if x['kiem_gio_d1'] == 'truot' and x['ket_qua'] != 'hong':
        raise GateFileError(f'{w}: kiem_gio_d1 truot on a run that is not lost (a run failing the per-run time '
                            'check is lost to the measurement, J v6 section 5)')
    _bool(x['dem_truot_kiem_gio'], f'{w}.dem_truot_kiem_gio')
    if x['dem_truot_kiem_gio'] and x['kiem_gio_d1'] != 'truot':
        raise GateFileError(f'{w}: dem_truot_kiem_gio true only on a run whose time check slipped (truot; F11, M2)')
    if x['kiem_gio_d1'] == 'truot' and not x['dem_truot_kiem_gio'] and 'thao_tac' not in lk:
        raise GateFileError(f'{w}: a time check slip not counted (dem_truot_kiem_gio false) on a run not lost by '
                            'operation (M2 exempts only runs lost by operation without independent evidence)')
    _ref(x['file_cham'], f'{w}.file_cham')
    if x['file_chung_cu'] is not None:
        _ref(x['file_chung_cu'], f'{w}.file_chung_cu')
    if x['file_gio'] is None:
        if x['kiem_gio_d1'] != 'khong_kiem' or x['ket_qua'] != 'hong':
            raise GateFileError(f'{w}.file_gio null only for a lost run whose time check did not run (khong_kiem)')
    else:
        _ref(x['file_gio'], f'{w}.file_gio')
    # GC6-1, GC6-2 (round 3): with the time file, its log and the score.json of its operations; (round 4) a lost run
    # without a time file still carries the log thu_tu_clip.csv names
    if (x['file_thao_tac'] is None) != (x['file_gio'] is None) or (x['file_gio'] is not None and x['file_log'] is None):
        raise GateFileError(f'{w}: file_log and file_thao_tac with the time file (the log it read, the score.json its '
                            'operations came from); file_thao_tac only then')
    for k in ('file_log', 'file_thao_tac'):
        if x[k] is not None:
            _ref(x[k], f'{w}.{k}')
    # GC6-3, F18 (2): the clip H comparison of the part of the session (drift.json of c7_calib_drift so) for every run
    # that is not lost; the second scoring (clip H at the end, above eps_H) with its own time file, or neither
    if x['file_so_h'] is None:
        if x['ket_qua'] != 'hong':
            raise GateFileError(f'{w}.file_so_h null only for a lost run (the clip H comparison decides the final '
                                'scoring; F18 (2))')
    else:
        _ref(x['file_so_h'], f'{w}.file_so_h')
    if (x['file_cham_h1'] is None) != (x['file_gio_h1'] is None):
        raise GateFileError(f'{w}: file_cham_h1 and file_gio_h1 go together (the second scoring and its time file)')
    if x['file_cham_h1'] is not None:
        if x['file_so_h'] is None or x['file_gio'] is None:
            raise GateFileError(f'{w}: a second scoring needs the clip H comparison and the first time file')
        _ref(x['file_cham_h1'], f'{w}.file_cham_h1')
        _ref(x['file_gio_h1'], f'{w}.file_gio_h1')
    # GC6-3 (round 2): the e_do file (c8b_edo luot) of each scoring the evidence file was made from; with the evidence
    # file and only then; the second one with the second scoring
    if (x['file_edo'] is None) != (x['file_chung_cu'] is None) \
            or (x['file_edo_h1'] is not None) != (x['file_chung_cu'] is not None and x['file_cham_h1'] is not None):
        raise GateFileError(f'{w}: file_edo goes with file_chung_cu, file_edo_h1 with it and a second scoring')
    for k in ('file_edo', 'file_edo_h1'):
        if x[k] is not None:
            _ref(x[k], f'{w}.{k}')
    return x


def count_chain(chain: list, hong_theo_liam: bool = False) -> Tuple[str, Dict[str, int]]:
    """The reruns cap of one C8 cell (J v6 section 5, O1 3.5 line 342, M2 in force): two counters in chain order,
    never reset: time check (a lost run whose time file says dem_truot_kiem_gio: a slip, not lost by a boot unless
    the locked check slipped on the whole window before the boot; CODE_C8B design v7 7a, counted by the field, not
    inferred) and reference window (cua_so_tham_chieu without thao_tac). The fourth of a kind closes the cell (dong_tran) and must
    be the last run; otherwise the last run must not be lost (co_luot). GateFileError when neither.
    hong_theo_liam (Liam 22/11, C8 contract section 3 (a): PID F08-1, F08-2 of session 2 recorded lost, not saved):
    the cell's last run is lost to the measurement (he_do, no operation) and Liam's recorded choice keeps it so, not
    rerun (hong_theo_liam: no eligible run in the cell); the chain is checked as above, the flag on a cell whose last
    run is not lost is refused."""
    n = {'kiem_gio': 0, 'cua_so': 0}
    for i, x in enumerate(chain):
        if x['ket_qua'] != 'hong':
            if i != len(chain) - 1:
                # GC6-1 (J v6 section 4 rule 2b): a rerun only replaces a LOST run; a run that is not lost ends the chain
                raise GateFileError(f'run {chain[i + 1]["run_id"]!r} after run {x["run_id"]!r} which is not lost '
                                    f'({x["ket_qua"]}): a rerun replaces a lost run only (J v6 section 4, rule 2b)')
            continue
        op = 'thao_tac' in x['loai_hong']
        if x['dem_truot_kiem_gio']:
            n['kiem_gio'] += 1
        if 'cua_so_tham_chieu' in x['loai_hong'] and not op:
            n['cua_so'] += 1
        if max(n.values()) > RERUNS_MAX:
            if i != len(chain) - 1:
                raise GateFileError(f'run {chain[i + 1]["run_id"]!r} after the cell was closed by the '
                                    f'{RERUNS_MAX + 1}th lost run of one kind ({x["run_id"]!r}; J v6 section 5)')
            return 'dong_tran', n
    x = chain[-1]
    if hong_theo_liam:
        if x['ket_qua'] != 'hong' or 'he_do' not in x['loai_hong'] or 'thao_tac' in x['loai_hong']:
            raise GateFileError(f'Liam\'s record keeps the cell lost (hong_theo_liam) but its last run {x["run_id"]!r} '
                                f'is {x["ket_qua"]} {x["loai_hong"]}: only a run lost to the measurement (he_do, no '
                                'operation) is kept lost without a rerun (Liam 22/11)')
        return 'hong_theo_liam', n
    if x['ket_qua'] == 'hong':
        raise GateFileError(f'the last run {x["run_id"]!r} is lost and the cell is not at its cap: run it '
                            'again (C8 is not done; a missing run may hide a larger U_t)')
    return 'co_luot', n


def _evidence(x, w: str) -> dict:
    _obj(x, ('phan_loai_du', 'ban_cham_cuoi', 'bo_so'), w)
    _bool(x['phan_loai_du'], f'{w}.phan_loai_du')
    _bool(x['ban_cham_cuoi'], f'{w}.ban_cham_cuoi')
    bs = x['bo_so']
    if not isinstance(bs, list) or len(bs) not in (1, 2):
        raise GateFileError(f'{w}.bo_so: one or two sets of numbers (two when scored again with both calibrations)')
    for i, b in enumerate(bs):
        _obj(b, EVID_KEYS, f'{w}.bo_so[{i}]')
        for k in EVID_KEYS:
            _bool(b[k], f'{w}.bo_so[{i}].{k}')
    return x


def evidence_of(runs: list) -> dict:
    """The three kinds of C8 evidence of one level (J v6 sections 4, 5; design section 7) from its eligible runs:
    {loai: [run ids excluding the level by e_do], e_do: {ctrl: [run ids]}, g4: {ctrl: [run ids]}}. A run is evidence
    only on its final scoring (ban_cham_cuoi) and when the property holds in every set of numbers."""
    def ok(r, k):
        c = r['chung_cu']
        return c['ban_cham_cuoi'] and all(b[k] for b in c['bo_so'])
    return dict(loai=[r['run_id'] for r in runs if ok(r, 'e_do_lo_vuot_D_moi_nut')],
                e_do={c: [r['run_id'] for r in runs if r['controller'] == c and ok(r, 'e_do_xac_dinh_moi_nut')]
                      for c in CONTROLLERS},
                g4={c: [r['run_id'] for r in runs if r['controller'] == c and ok(r, 'g4_can_huu_han_mien_hop_moi_nut')]
                    for c in CONTROLLERS})


def _runs_from_cells(cells: list, levels: list) -> Dict[str, list]:
    """{level: [(cell, last run)]} for the co_luot cells of each level."""
    out = {lv: [] for lv in levels}
    for c in cells:
        if c['trang_thai'] == 'co_luot':
            out[c['muc']].append((c, c['chuoi'][-1]))
    return out


def _check_cells(d: dict) -> list:
    cells = d['o_c8']
    if not isinstance(cells, list) or not cells:
        raise GateFileError('o_c8: a non-empty list of C8 cells')
    seen, runs_seen, clips_seen, files_seen = set(), set(), set(), {}
    for i, c in enumerate(cells):
        w = f'o_c8[{i}]'
        liam = isinstance(c, dict) and c.get('trang_thai') == 'hong_theo_liam'     # Liam 22/11: his record in the cell
        _obj(c, ('o', 'muc', 'controller', 'trang_thai', 'chuoi') + (('liam',) if liam else ()), w)
        if liam:
            _liam(c['liam'], f'{w}.liam')
        _text(c['o'], f'{w}.o')
        if c['o'] in seen:
            raise GateFileError(f'{w}.o {c["o"]!r}: each cell once')
        seen.add(c['o'])
        if c['muc'] not in d['muc_kha_thi']:
            raise GateFileError(f'{w}.muc {c["muc"]!r}: a level of muc_kha_thi {d["muc_kha_thi"]}')
        if c['controller'] not in CONTROLLERS:
            raise GateFileError(f'{w}.controller {c["controller"]!r}: one of {CONTROLLERS}')
        ch = c['chuoi']
        if not isinstance(ch, list) or not ch:
            raise GateFileError(f'{w}.chuoi: a non-empty list of runs in log order')
        for j, x in enumerate(ch):
            _chain_entry(x, f'{w}.chuoi[{j}]')
            if x['run_id'] in runs_seen:
                raise GateFileError(f'{w}.chuoi[{j}].run_id {x["run_id"]!r}: each run once in the C8 cells')
            runs_seen.add(x['run_id'])
            if x['clip'] in clips_seen:
                raise GateFileError(f'{w}.chuoi[{j}].clip {x["clip"]!r}: each clip is one physical run (GC6-1)')
            clips_seen.add(x['clip'])
            for k in CHAIN_FILES:                    # GC6-1: the files of one run are never another run's
                if x[k] is None:
                    continue
                h = x[k]['sha256']
                if h in files_seen:
                    raise GateFileError(f'{w}.chuoi[{j}].{k}: the same file (sha256) as {files_seen[h]}; the source '
                                        'files of a run belong to that run only (GC6-1)')
                files_seen[h] = f'{c["o"]} run {x["run_id"]} {k}'
        try:
            st, _ = count_chain(ch, hong_theo_liam=liam)
        except GateFileError as exc:
            raise GateFileError(f'{w} ({c["o"]}): {exc}')
        if st == 'co_luot' and ch[-1]['thieu_u_t'] and ch[-1]['ket_qua'] != 'hong':
            st = 'thieu_u_t'                             # GC9-6: no eligible run in the cell
        if c['trang_thai'] != st:
            raise GateFileError(f'{w}.trang_thai {c["trang_thai"]!r}: the chain recounted gives {st!r} (KT6-11, M2)')
    return cells


def _check_runs(d: dict, cells: list) -> Dict[str, list]:
    """runs_c8 {level: [{run_id, controller, o, h_s, u_max_s, u_t_s, chung_cu}]} equal to the eligible runs rebuilt
    from the cells (RT2-3)."""
    rc = d['runs_c8']
    kt = d['muc_kha_thi']
    if not isinstance(rc, dict) or set(rc) != set(kt):
        raise GateFileError(f'runs_c8 keys {sorted(rc) if isinstance(rc, dict) else rc!r}: exactly muc_kha_thi {kt}')
    want = _runs_from_cells(cells, kt)
    for lv in kt:
        rs, w = rc[lv], f'runs_c8.{lv}'
        if not isinstance(rs, list):
            raise GateFileError(f'{w}: a list')
        got = {}
        for i, r in enumerate(rs):
            _obj(r, ('run_id', 'controller', 'o', 'h_s', 'u_max_s', 'u_t_s', 'chung_cu'), f'{w}[{i}]')
            h, um = _num(r['h_s'], f'{w}[{i}].h_s'), _num(r['u_max_s'], f'{w}[{i}].u_max_s')
            ut = _num(r['u_t_s'], f'{w}[{i}].u_t_s', positive=True)
            if not _close(ut, h + um):
                raise GateFileError(f'{w}[{i}].u_t_s {ut!r}: h_s + u_max_s = {h + um!r} (O1 3.9 rule 2b, J v6 section 4)')
            _evidence(r['chung_cu'], f'{w}[{i}].chung_cu')
            if r['run_id'] in got:
                raise GateFileError(f'{w}[{i}].run_id {r["run_id"]!r}: twice')
            got[r['run_id']] = r
        exp = {x['run_id']: c for c, x in want[lv]}
        if set(got) != set(exp):
            raise GateFileError(f'{w}: runs {sorted(got)} but the eligible runs of the cells (last run of each co_luot '
                                f'cell) are {sorted(exp)}: no run left out, none added (RT2-3)')
        last = {x['run_id']: x for _, x in want[lv]}
        for rid, r in got.items():
            c = exp[rid]
            if last[rid]['file_gio'] is None or last[rid]['file_chung_cu'] is None:
                raise GateFileError(f'{w} {rid}: an eligible run carries its time file (h_s) and its evidence file '
                                    '(chung_cu_c8, c8b_edo chung-cu; F17)')
            if r['controller'] != c['controller'] or r['o'] != c['o']:
                raise GateFileError(f'{w} {rid}: controller / cell {r["controller"]!r} / {r["o"]!r}, the cell says '
                                    f'{c["controller"]!r} / {c["o"]!r}')
            if not r['chung_cu']['phan_loai_du']:
                raise GateFileError(f'{w} {rid}: G1-G5 not all classified before the mask (phan_loai_du false): C8 '
                                    'is not done')
    return {lv: rc[lv] for lv in kt}


def _check_levels(d: dict, runs: Dict[str, list]) -> None:
    """The three kinds of evidence and the gate levels recomputed (design section 7)."""
    kt, lv = d['muc_kha_thi'], d['levels']
    bo = d['muc_bo_theo_liam']
    if not isinstance(bo, dict) or any(k not in kt for k in bo):
        raise GateFileError(f'muc_bo_theo_liam: an object keyed by levels of muc_kha_thi')
    for k, x in bo.items():
        _liam(x, f'muc_bo_theo_liam.{k}')
    ml, cm = d['muc_loai_e_do'], d['chung_cu_muc']
    if not isinstance(ml, dict) or set(ml) != set(kt):
        raise GateFileError('muc_loai_e_do: keyed by exactly the levels of muc_kha_thi')
    loai = []
    for k in kt:
        ev = evidence_of(runs[k])
        if ml[k] != ev['loai']:
            raise GateFileError(f'muc_loai_e_do.{k} {ml[k]!r}: from the runs it is {ev["loai"]!r} (J v6 section 4)')
        if ev['loai']:
            loai.append(k)
    want_levels = [k for k in kt if k not in loai and k not in bo]
    if not want_levels:
        raise GateFileError('no gate level left (excluded by e_do or dropped by Liam): O1 stops, Liam narrows the '
                            'task (J v6 section 4)')
    if lv != want_levels:
        raise GateFileError(f'levels {lv!r}: muc_kha_thi minus the levels excluded by e_do {loai} minus those Liam '
                            f'dropped {sorted(bo)} is {want_levels}')
    if not isinstance(cm, dict) or set(cm) != set(kt) - set(loai):
        raise GateFileError(f'chung_cu_muc: keyed by the levels not excluded by e_do {sorted(set(kt) - set(loai))}')
    for k in cm:
        ev = evidence_of(runs[k])
        _obj(cm[k], ('e_do', 'g4'), f'chung_cu_muc.{k}')
        if cm[k] != dict(e_do=ev['e_do'], g4=ev['g4']):
            raise GateFileError(f'chung_cu_muc.{k}: from the runs it is {dict(e_do=ev["e_do"], g4=ev["g4"])!r}')
        if k in bo:
            continue
        ctrls = {r['controller'] for r in runs[k]}
        miss = [c for c in CONTROLLERS if c not in ctrls]
        if miss:
            raise GateFileError(f'{k}: no eligible run of {miss} (O1 3.9 rule 2b: one run of each controller at each '
                                'level, else C8 is not done)')
        for c in CONTROLLERS:
            if not ev['e_do'][c]:
                raise GateFileError(f'{k}: no run of {c} with the e_do bound at every node on its final scoring (both '
                                    'sets of numbers): C8 is not done at this level (J v6 section 5 (ii))')
            if not ev['g4'][c]:
                raise GateFileError(f'{k}: no run of {c} with a finite G4 bound on the union domain at every node on '
                                    'its final scoring (both sets): the level cannot be locked (J v6 section 5 (iii));'
                                    f' CAN LIAM QUYET: drop it (muc_bo_theo_liam.{k}) or wait for another way to '
                                    'measure')


def _check_bo_cham(d: dict) -> None:
    b = _obj(d['bo_cham'], ('phien_ban', 'bang_man_trap'), 'bo_cham', extra_ok=True)
    _text(b['phien_ban'], 'bo_cham.phien_ban')
    t = b['bang_man_trap']
    need = list(d['muc_kha_thi']) + [TASK_K]
    if not isinstance(t, dict) or set(t) != set(need):
        raise GateFileError(f'bo_cham.bang_man_trap: one shutter branch for each of {need} (J v6 section 6, J4-13)')
    for k in need:
        x = t[k]
        if not isinstance(x, dict) or x.get('nhanh') not in MAN_TRAP:
            raise GateFileError(f'bo_cham.bang_man_trap.{k}: {{nhanh: one of {MAN_TRAP}}}')
        if x['nhanh'] == 'alpha':
            _obj(x, ('nhanh', 'nguon_3b'), f'bo_cham.bang_man_trap.{k}')
            _ref(x['nguon_3b'], f'bo_cham.bang_man_trap.{k}.nguon_3b')
        elif set(x) != {'nhanh'}:
            raise GateFileError(f'bo_cham.bang_man_trap.{k}: only {{nhanh}} for one frame (no 1.25 margin)')


def _check_chuan_u(d: dict) -> None:
    c = _obj(d['chuan_u'], ('loai', 'le', 'ho_so_c8', 'dieu_kien'), 'chuan_u')
    if c['loai'] != CHUAN_U_LOAI or c['le'] != CHUAN_U_LE:
        raise GateFileError(f'chuan_u: loai {CHUAN_U_LOAI!r}, le {CHUAN_U_LE} (J v6 section 6)')
    _ref(c['ho_so_c8'], 'chuan_u.ho_so_c8')
    dk = _obj(c['dieu_kien'], CHUAN_U_DK, 'chuan_u.dieu_kien')
    for k in CHUAN_U_DK:
        x = _obj(dk[k], ('trang_thai', 'bang_chung'), f'chuan_u.dieu_kien.{k}')
        if x['trang_thai'] != 'dat':
            raise GateFileError(f'chuan_u.dieu_kien.{k}.trang_thai {x["trang_thai"]!r}: dat (the procedure applied: '
                                'locked first, run on every C8 run, with SHA; J v6 section 6: missing one, the index '
                                'is not used)')
        _ref(x['bang_chung'], f'chuan_u.dieu_kien.{k}.bang_chung')


def _check_luot_k(d: dict) -> list:
    """GC9-2 (CODE 20/11: every C8 run is held out, the K runs too): luot_k [{run_id, clip, file_cham, file_cham_h1}],
    the grid file of every K run of thu_tu_clip.csv (c8b_luoi, one per run; the second scoring with it), only for its
    scoring time and code; K runs are no cells of rule 2b (G1-G5 always decide)."""
    lk = d['luot_k']
    if not isinstance(lk, list):
        raise GateFileError('luot_k: a list of the K runs of C8')
    seen = set()
    chain_clips = {x['clip'] for c in d['o_c8'] for x in c['chuoi']}
    for i, x in enumerate(lk):
        w = f'luot_k[{i}]'
        _obj(x, ('run_id', 'clip', 'file_cham', 'file_cham_h1'), w)
        if not isinstance(x['run_id'], str) or not RUN_CODE_K.match(x['run_id']) or x['run_id'] in seen:
            raise GateFileError(f'{w}.run_id {x["run_id"]!r}: a K run code C8-<P|L>-<K1|K2>-<n>[_lai<n>], once')
        seen.add(x['run_id'])
        if _text(x['clip'], f'{w}.clip') in chain_clips:
            raise GateFileError(f'{w}.clip {x["clip"]!r}: the clip of a figure-8 run (one clip, one run)')
        chain_clips.add(x['clip'])
        _ref(x['file_cham'], f'{w}.file_cham')
        if x['file_cham_h1'] is not None:
            _ref(x['file_cham_h1'], f'{w}.file_cham_h1')
    return lk


def _check_tien_de(d: dict) -> List[dict]:
    """The C8 sessions (CODE 22/11, C8 contract section 3: a lost run rerun in a later floor session; one folder per
    session): thu_tu_clip [one reference per session, in session order; the folder of the session is the folder of its
    file] and tien_de_c8 [{buoi, doan: [{ten, dau [two clips H], cuoi, file, kiem_cheo}]}, same order]. The folders
    differ, none inside another; the files of a segment lie in the folder of its session. doan: one entry per camera
    segment that holds C8 runs, in session order (CODE 20/11 after C8 session 2: three camera segments, each mounting
    with its own sessions; J v6 section 5 J3-5 read per segment: two clips H at its start, one at its end; a segment
    without C8 runs is not listed; KIEMCHEO fix/c8-kiemcheo-doan: one kiem_cheo.json and one premise per segment, its
    name doan.ten, once in its session). Returns [{ten, clips [dau1, dau2, cuoi], file, kiem_cheo, buoi, k}] of every
    session in order (k the session index)."""
    tts, tds = d['thu_tu_clip'], d['tien_de_c8']
    if not isinstance(tts, list) or not tts or not isinstance(tds, list) or len(tds) != len(tts):
        raise GateFileError('thu_tu_clip, tien_de_c8: one entry for each C8 session, in session order, as many of one '
                            'as of the other (CODE 22/11)')
    out, shas, ids, folders = [], set(), set(), []
    for k, (tt, t) in enumerate(zip(tts, tds)):
        f = _folder_of(_ref(tt, f'thu_tu_clip[{k}]'))
        if any(f == g or _under(f, g) or _under(g, f) for g in folders):
            raise GateFileError(f'thu_tu_clip[{k}] {tt["duong_dan"]!r}: one folder for each C8 session, none inside '
                                'another (CODE 22/11)')
        folders.append(f)
        w0 = f'tien_de_c8[{k}]'
        _obj(t, ('buoi', 'doan'), w0)
        b = _text(t['buoi'], f'{w0}.buoi')               # GC6-4: the C8 session the premise belongs to
        if b in ids:
            raise GateFileError(f'{w0}.buoi {b!r}: each C8 session once')
        ids.add(b)
        ds = t['doan']
        if not isinstance(ds, list) or not ds:
            raise GateFileError(f'{w0}.doan: a non-empty list of the camera segments that hold C8 runs')
        names = set()
        for i, s in enumerate(ds):
            w = f'{w0}.doan[{i}]'
            _obj(s, ('ten', 'dau', 'cuoi', 'file', 'kiem_cheo'), w)
            ten = _text(s['ten'], f'{w}.ten')
            if ten in names:
                raise GateFileError(f'{w}.ten {ten!r}: each segment once in its session')
            names.add(ten)
            if not isinstance(s['dau'], list) or len(s['dau']) != 2 or s['cuoi'] is None:
                raise GateFileError(f'{w}: two clips H at the start of the segment (dau), one at its end (cuoi); missing '
                                    'one, C8 is not locked: CAN LIAM QUYET (J v6 section 5 J3-5, section 0b)')
            clips = [_ref(x, f'{w}.dau[{j}]') for j, x in enumerate(s['dau'])] + [_ref(s['cuoi'], f'{w}.cuoi')]
            for x in clips:
                if x['sha256'] in shas:
                    raise GateFileError(f'{w}: a clip H used twice (every clip H of C8 is a different clip)')
                shas.add(x['sha256'])
            _ref(s['file'], f'{w}.file')
            _ref(s['kiem_cheo'], f'{w}.kiem_cheo')
            for x in clips + [s['file'], s['kiem_cheo']]:
                if not _under(x['duong_dan'], f):
                    raise GateFileError(f'{w}: {x["duong_dan"]!r} is not in the folder {f!r} of its session (a session '
                                        'computes its own premise, sessions and checks; CODE 22/11)')
            out.append(dict(ten=ten, clips=clips, file=s['file'], kiem_cheo=s['kiem_cheo'], buoi=b, k=k))
    return out


def car_pose_v2_of(cp, what: str) -> dict:
    """car_pose of the v2 lock {tag_offset, side_m, h_m, kappa, phien_ban} (design v4 section 3.1 table)."""
    _obj(cp, ('tag_offset', 'side_m', 'h_m', 'kappa', 'phien_ban'), f'{what}.car_pose')
    to = cp['tag_offset']
    if to is not None and (not isinstance(to, list) or len(to) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v != 0 for v in to)):
        raise GateFileError(f'{what}.car_pose.tag_offset {to!r}: null or the zero vector')
    return dict(tag_offset=None, side_m=_num(cp['side_m'], f'{what}.car_pose.side_m', positive=True),
                h_m=_num(cp['h_m'], f'{what}.car_pose.h_m'), kappa=_num(cp['kappa'], f'{what}.car_pose.kappa'),
                phien_ban=_text(cp['phien_ban'], f'{what}.car_pose.phien_ban'))


def _range(x, what: str) -> Tuple[float, float]:
    if not isinstance(x, list) or len(x) != 2:
        raise GateFileError(f'{what} {x!r}: [lo, hi]')
    lo, hi = _num(x[0], f'{what}[0]'), _num(x[1], f'{what}[1]')
    if hi < lo:
        raise GateFileError(f'{what} {x!r}: lo <= hi')
    return lo, hi


def _check_camera(d: dict) -> dict:
    cam = d['camera']
    w = 'camera.c8b'
    c8b = cam.get('c8b') if isinstance(cam, dict) else None
    _obj(c8b, ('moc_dich', 'khoa_sha256', 'mien_car_pose', 'eps_H'), w, extra_ok=True)
    if not _hex64(c8b['khoa_sha256']):
        raise GateFileError(f'{w}.khoa_sha256: 64 lowercase hex digits')
    md = _obj(c8b['moc_dich'], MOC_KEYS + ('nguon',), f'{w}.moc_dich', extra_ok=True)
    for k in MOC_KEYS:
        _num(md[k], f'{w}.moc_dich.{k}')
    ng = _obj(md['nguon'], ('bo_lay_mau', 'car_pose', 'luat_lich', 'manifests'), f'{w}.moc_dich.nguon')
    _text(ng['bo_lay_mau'], f'{w}.moc_dich.nguon.bo_lay_mau')
    ll = _obj(ng['luat_lich'], ('version', 'luat'), f'{w}.moc_dich.nguon.luat_lich')    # CODE 22/11: the rule, not a sha
    _text(ll['version'], f'{w}.moc_dich.nguon.luat_lich.version')
    if not isinstance(ll['luat'], dict) or not ll['luat']:
        raise GateFileError(f'{w}.moc_dich.nguon.luat_lich.luat: the rule constants of lich (an object)')
    if not isinstance(ng['manifests'], list) or not ng['manifests']:
        raise GateFileError(f'{w}.moc_dich.nguon.manifests: the U manifests of the C8 scorings used for the lock')
    for i, x in enumerate(ng['manifests']):
        _ref(x, f'{w}.moc_dich.nguon.manifests[{i}]')
    cp = car_pose_v2_of(ng['car_pose'], f'{w}.moc_dich.nguon')
    # KG-1 (Codex N1 + merged lock round 1, CODE 24/11): the lock writes the domain with its state and reasons; the
    # gate takes the six keys and needs the state dat (a domain C8B could not state is no domain)
    mc = _obj(c8b['mien_car_pose'], ('side_m', 'phien_ban', 'h_m', 'kappa', 'trang_thai', 'ly_do'),
              f'{w}.mien_car_pose')
    if mc['trang_thai'] != 'dat':
        raise GateFileError(f'{w}.mien_car_pose.trang_thai {mc["trang_thai"]!r}: dat needed (the C8b lock could not '
                            f'state the camera domain: {mc["ly_do"]!r}); C8 is not locked')
    hl, hh = _range(mc['h_m'], f'{w}.mien_car_pose.h_m')
    kl, kh = _range(mc['kappa'], f'{w}.mien_car_pose.kappa')
    if mc['side_m'] != cp['side_m'] or mc['phien_ban'] != cp['phien_ban'] or not hl <= cp['h_m'] <= hh \
            or not kl <= cp['kappa'] <= kh:
        raise GateFileError(f'{w}.moc_dich.nguon.car_pose: outside mien_car_pose (side_m, phien_ban equal; h_m, kappa '
                            'in their ranges)')
    eh = _obj(c8b['eps_H'], ('mm', 'huong_deg'), f'{w}.eps_H')
    _num(eh['mm'], f'{w}.eps_H.mm', positive=True)
    _num(eh['huong_deg'], f'{w}.eps_H.huong_deg', positive=True)
    return c8b


def check(d) -> dict:
    """The gate file content checked field by field and recomputed (module docstring). Returns it."""
    if not isinstance(d, dict):
        raise GateFileError('not a JSON object')
    if d.get('format') in OLD_FORMATS:
        raise GateFileError(f'format {d["format"]!r}: refused, the gate of C8 is {FORMAT!r} (design v6_C8)')
    for k in TOP_KEYS:
        if k not in d:
            raise GateFileError(f'{k}: missing')
    if d['format'] != FORMAT:
        raise GateFileError(f'format {d["format"]!r}: {FORMAT!r}')
    kt = d['muc_kha_thi']
    if (not isinstance(kt, list) or not kt or any(not isinstance(x, str) or x not in FIG8 for x in kt)
            or len(set(kt)) != len(kt)):
        raise GateFileError(f'muc_kha_thi {kt!r}: a non-empty list of figure-8 tasks {sorted(FIG8)}, each once')
    lv = d['levels']
    if not isinstance(lv, list) or any(not isinstance(x, str) for x in lv):
        raise GateFileError(f'levels {lv!r}: a list of figure-8 tasks')
    if d['u_t_branch'] not in U_T_BRANCHES:
        raise GateFileError(f'u_t_branch {d["u_t_branch"]!r}: one of {U_T_BRANCHES}')
    if d['lich_doi_chieu'] != LICH:
        raise GateFileError(f'lich_doi_chieu {d["lich_doi_chieu"]!r}: {LICH!r} (the schedule Uhat is computed on)')
    _check_bo_cham(d)
    _ref(d['ke_hoach_c8'], 'ke_hoach_c8')
    _ref(d['khoa_c8b'], 'khoa_c8b')
    cells = _check_cells(d)
    runs = _check_runs(d, cells)
    _check_levels(d, runs)
    _check_chuan_u(d)
    _check_tien_de(d)
    _check_luot_k(d)
    q = d['quyet_liam']
    if not isinstance(q, dict) or any(k not in lv for k in q):
        raise GateFileError('quyet_liam: an object keyed by gate levels')
    for k, x in q.items():
        if not isinstance(x, dict) or any(c not in MASKABLE for c in x):
            raise GateFileError(f'quyet_liam.{k}: keyed by conditions of {MASKABLE}')
        for c, y in x.items():
            _liam(y, f'quyet_liam.{k}.{c}')
            if y.get('chon') not in ('so_do', 'quyet'):
                raise GateFileError(f'quyet_liam.{k}.{c}.chon {y.get("chon")!r}: so_do or quyet')
    b = _obj(d['dieu_2b'], ('thresholds', 'per_level'), 'dieu_2b')
    th = b['thresholds']
    if not isinstance(th, dict) or set(th) != set(MASKABLE):
        raise GateFileError(f'dieu_2b.thresholds {th!r}: keys {MASKABLE}')
    for c in MASKABLE:
        if _num(th[c], f'dieu_2b.thresholds.{c}') != o1_gate.THRESHOLDS[c]:
            raise GateFileError(f'dieu_2b.thresholds.{c} {th[c]!r}: expected {o1_gate.THRESHOLDS[c]} (fwsim/o1_gate.py)')
    pl = b['per_level']
    if not isinstance(pl, dict) or set(pl) != set(lv):
        raise GateFileError(f'dieu_2b.per_level keys {sorted(pl) if isinstance(pl, dict) else pl!r}: exactly the '
                            f'levels {lv}')
    for name in lv:
        e, w = pl[name], f'dieu_2b.per_level.{name}'
        _obj(e, LEVEL_KEYS, w)
        t = _level_terms(e, w)
        u_t = _num(e['u_t_c8_s'], f'{w}.u_t_c8_s', positive=True)
        want_ut = max(r['u_t_s'] for r in runs[name])
        if u_t != want_ut:
            raise GateFileError(f'{w}.u_t_c8_s {u_t!r}: the largest h_s + u_max_s of the eligible runs is {want_ut!r}')
        uh = e['uhat']
        if not isinstance(uh, dict) or set(uh) != set(MASKABLE):
            raise GateFileError(f'{w}.uhat: keys {MASKABLE}')
        want = uhat_level(name, t, u_t)
        for c in MASKABLE:
            x = _obj(uh[c], UH_KEYS, f'{w}.uhat.{c}')
            y = want[c]
            # GC6-15: types first (a count is a whole number, a state one of UH_STATES, numbers or null)
            if type(x['so_lan_tinh']) is not int or x['trang_thai'] not in UH_STATES \
                    or not _isnum(x['delta_cuoi_s']) or any(x[k] is not None and not _isnum(x[k])
                                                            for k in ('uhat', 'uhat_duoi')):
                raise GateFileError(f'{w}.uhat.{c} {x!r}: so_lan_tinh a whole number, trang_thai one of {UH_STATES}, '
                                    'delta_cuoi_s a number, uhat / uhat_duoi numbers or null')
            if x['trang_thai'] != y['trang_thai'] or x['so_lan_tinh'] != y['so_lan_tinh'] \
                    or not _close(x['delta_cuoi_s'], y['delta_cuoi_s']) or not _close(x['uhat'], y['uhat']) \
                    or not _close(x['uhat_duoi'], y['uhat_duoi']):
                raise GateFileError(f'{w}.uhat.{c} {x!r}: recomputed {y!r}')
        sd, why = so_do_of(name, want, q.get(name))
        if not isinstance(e['so_do'], list) or tuple(e['so_do']) != sd:
            raise GateFileError(f'{w}.so_do {e["so_do"]!r}: recomputed {list(sd)!r} (design section 5)')
        if e['ly_do_so_do'] != why:
            raise GateFileError(f'{w}.ly_do_so_do {e["ly_do_so_do"]!r}: recomputed {why!r}')
    _check_camera(d)
    return d


# ── sources (check_nguon, design section 3) ────────────────────────────────────────────────────────────────────

def _ref_bytes(goc: str, ref: dict, what: str) -> bytes:
    """The bytes of a reference under goc, read once and hashed (GC6-14: the bytes parsed are the bytes hashed; a
    path on another drive or outside goc refused by name)."""
    if _bad_rel(str(ref['duong_dan'])):
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: a path relative to the C8 folder, inside it')
    base = os.path.realpath(goc)
    p = os.path.realpath(os.path.join(base, ref['duong_dan']))
    try:
        inside = os.path.commonpath([base, p]) == base
    except ValueError:                                   # another drive
        inside = False
    if not inside:
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: resolves outside the C8 folder {goc!r}')
    if not os.path.isfile(p):
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: no such file under {goc!r}')
    with open(p, 'rb') as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: sha256 differs from the gate (file changed)')
    return raw


def _open_ref(goc: str, ref: dict, what: str, as_json: bool = True):
    """The file of a reference under goc, hashed again (realpath inside goc). Its JSON when as_json."""
    raw = _ref_bytes(goc, ref, what)
    if not as_json:
        return None
    try:
        x = json.loads(raw.decode('utf-8'), object_pairs_hook=_no_duplicate_keys)
    except GateFileError as exc:
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: {exc}')
    except (UnicodeDecodeError, ValueError) as exc:
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: not UTF-8 JSON ({exc})')
    if not isinstance(x, dict):
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: not a JSON object')
    return x


def _iso_time(x, what: str) -> dt.datetime:
    try:
        t = dt.datetime.fromisoformat(x)
    except (TypeError, ValueError):
        raise GateFileError(f'{what} {x!r}: an ISO date-time with its offset')
    if t.tzinfo is None:
        raise GateFileError(f'{what} {x!r}: an ISO date-time with its offset')
    return t


def _csv_rows(goc: str, ref: dict, what: str) -> List[dict]:
    """The rows of a CSV reference (hashed and parsed from the same bytes)."""
    raw = _ref_bytes(goc, ref, what)
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: not UTF-8 ({exc})')
    rd = csv.DictReader(io.StringIO(text, newline=''))
    if rd.fieldnames is None or any(c not in rd.fieldnames for c in THU_TU_COLS):
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: a CSV with the columns {THU_TU_COLS} (F18 (1))')
    return list(rd)


def _folder_of(ref: dict) -> str:
    """The folder of a C8 session: the folder of its thu_tu_clip.csv, relative to the C8 folder ('' for the folder
    itself)."""
    return posixpath.dirname(ref['duong_dan'])


def _under(path: str, folder: str) -> bool:
    return folder == '' or path.startswith(folder + '/')


def _buoi_rows(goc: str, refs: list) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """The rows of the thu_tu_clip.csv of every C8 session, session after session (CODE 22/11, C8 contract section 3:
    a rerun of a lost run in a later floor session; the chain of a cell runs through the sessions in order), and
    [(first, end) row index of each session]. Each row carries _buoi (the session index) and _vt (file and row)."""
    rows, slices = [], []
    for k, ref in enumerate(refs):
        lo = len(rows)
        for i, r in enumerate(_csv_rows(goc, ref, f'thu_tu_clip[{k}]')):
            r['_buoi'], r['_vt'] = k, f'{ref["duong_dan"]} row {i + 2}'
            rows.append(r)
        slices.append((lo, len(rows)))
    return rows, slices


def _time_file(goc: str, ref: dict, w: str, cell: dict) -> Tuple[dict, str, list]:
    """A c8b_gio_v1 time file of a run of `cell`: (content, the name of its ESP32 log, the operations it states).
    GC6-1: task and controller are those of the cell; the log is named once (nguon, by the sha256 of the log)."""
    fg = _open_ref(goc, ref, w)
    k1 = fg.get('kiem_gio_d1')
    if fg.get('format') != GIO_FORMAT or not isinstance(k1, dict) or not isinstance(k1.get('truoc_boot'), dict):
        raise GateFileError(f'{w}: not a {GIO_FORMAT} file with its kiem_gio_d1 block and truoc_boot (F11)')
    if fg.get('task') != cell['muc'] or fg.get('controller') != CTRL_LOG[cell['controller']]:
        raise GateFileError(f'{w}: a time file of task {fg.get("task")!r}, controller {fg.get("controller")!r}; the '
                            f'cell is {cell["muc"]} {cell["controller"]} (logger name {CTRL_LOG[cell["controller"]]}; '
                            'GC6-1)')
    sh = fg.get('sha256') if isinstance(fg.get('sha256'), dict) else {}
    esp = sh.get('esp32')
    logs = [n for n in fg.get('nguon') or [] if isinstance(n, dict) and n.get('sha256') == esp]
    if not _hex64(esp) or not _hex64(sh.get('frames')) or len(logs) != 1 or not isinstance(logs[0].get('file'), str) \
            or not logs[0]['file']:
        raise GateFileError(f'{w}: the time file names its ESP32 log once and its frames (sha256.esp32, sha256.frames, '
                            'that entry of nguon)')
    tt = k1['truoc_boot'].get('thao_tac')
    if tt is None:
        tt = []
    if not isinstance(tt, list) or any(v not in THAO_TAC for v in tt) or len(set(tt)) != len(tt):
        raise GateFileError(f'{w}: kiem_gio_d1.truoc_boot.thao_tac {tt!r}: operations of {THAO_TAC}, each once')
    src = fg.get('thao_tac_nguon')
    if not isinstance(src, dict) or not _hex64(src.get('sha256')):
        raise GateFileError(f'{w}: thao_tac_nguon {{file, sha256}}: the score.json the operations came from (c8b_gio '
                            '--score; CODE 19/11: the one source of the operator kinds)')
    return fg, logs[0]['file'], sorted(tt)


def _m2(fg: dict, ops: list, w: str) -> None:
    """GC6-2 (round 3): the slip counter of a time file recomputed from the operations of the run (score.json, CODE
    19/11) by the rule of CODE_C8B design v7 7a (J v6 section 5 M2, J5-9): a slip counts unless the run is lost to an
    operation, except a boot whose locked check ran on its whole window before the boot and slipped there."""
    k1 = fg['kiem_gio_d1']
    tb = k1['truoc_boot']
    co = 'boot' in ops
    if tb.get('co_boot') is not co:
        raise GateFileError(f'{w}: truoc_boot.co_boot {tb.get("co_boot")!r}, the operations of the run {ops} say '
                            f'{co} (GC6-2)')
    cw = fg.get('cua_so_ghep_log_s')
    if tb.get('du_cua_so') is True and not (co and _isnum(tb.get('t_boot_log_s')) and isinstance(cw, list)
                                           and len(cw) == 2 and all(_isnum(v) for v in cw)
                                           and cw[1] < tb['t_boot_log_s']):
        raise GateFileError(f'{w}: truoc_boot.du_cua_so true without a boot time in the log (t_boot_log_s): the window '
                            'before the boot cannot be told (c8b_gio f11; GC6-2)')
    want = k1.get('trang_thai') == 'truot' and (not ops or (co and tb.get('du_cua_so') is True
                                                           and tb.get('trang_thai') == 'truot'))
    if fg.get('dem_truot_kiem_gio') is not want:
        raise GateFileError(f'{w}: dem_truot_kiem_gio {fg.get("dem_truot_kiem_gio")!r}; from kiem_gio_d1 '
                            f'{k1.get("trang_thai")!r}, the operations {ops} and truoc_boot it is {want} (M2, J5-9; '
                            'GC6-2)')


def _scoring_file(goc: str, ref: dict, w: str, cell: dict, gio_ref: Optional[dict], clip: str) -> dict:
    """A c8b_luoi luoi.json of a run of `cell`, tied to its time file (gio_sha256 present and equal; GC6-15), to the
    clip of the run (clip {ten, clip_sha256}, C8B M5b) and stamped with the time it was made (script.luc_chay)."""
    fc = _open_ref(goc, ref, w)
    if fc.get('trang_thai_luot') not in LUOI_STATES:
        raise GateFileError(f'{w}: not a c8b_luoi output (trang_thai_luot one of {LUOI_STATES})')
    if fc.get('outcome') not in ('dat', 'khong_dat', 'chua_ket_luan', 'hong_he_do') \
            or (fc['trang_thai_luot'] == 'hong_he_do') != (fc['outcome'] == 'hong_he_do'):
        raise GateFileError(f'{w}: outcome {fc.get("outcome")!r} with trang_thai_luot {fc["trang_thai_luot"]!r}')
    if fc.get('task') != cell['muc']:
        raise GateFileError(f'{w}: a scoring of task {fc.get("task")!r}, the cell is {cell["muc"]} (GC6-1)')
    if 'gio_sha256' not in fc:
        raise GateFileError(f'{w}: the scoring file carries no gio_sha256 (F11, F17)')
    want = None if gio_ref is None else gio_ref['sha256']
    if fc['gio_sha256'] != want:
        raise GateFileError(f'{w}: the scoring file was made from another time file (gio_sha256)')
    cl = fc.get('clip')
    if not isinstance(cl, dict) or cl.get('ten') != clip or not _hex64(cl.get('clip_sha256')):
        raise GateFileError(f'{w}: clip {cl!r}: {{ten {clip!r} (thu_tu_clip.csv), clip_sha256}} (C8B M5b; GC6-1)')
    sc = fc.get('script')
    _iso_time(sc.get('luc_chay') if isinstance(sc, dict) else None, f'{w}: script.luc_chay (when it was scored)')
    return fc


def _rel_in(goc: str, p) -> object:
    """GC9-1 (Codex, C8B writes absolute paths: c8b_luoi node_sources, c8b_khoa_v5 nguon_3b): an absolute path whose
    real path lies inside the C8 folder, given relative to it; anything else unchanged (then refused by _ref)."""
    if not isinstance(p, str) or not os.path.isabs(p):
        return p
    base, q = os.path.realpath(goc), os.path.realpath(p)
    try:
        if os.path.commonpath([base, q]) == base and q != base:
            return os.path.relpath(q, base).replace(os.sep, '/')
    except ValueError:
        pass
    return p


def _scoring_times(fc: dict, w: str) -> List[dt.datetime]:
    """When a grid was scored (GC6-1; C8B 19/11 khuya): every node scoring time scorer.nut[k].luc_chay (the scorer
    state written just before that node was scored) and the time the grid file was written (script.luc_chay); the
    earliest is its first scoring."""
    out = [_iso_time(fc['script']['luc_chay'], f'{w}: script.luc_chay')]
    sc = fc.get('scorer')
    if isinstance(sc, dict):
        nut = sc.get('nut')
        if not isinstance(nut, list):
            raise GateFileError(f'{w}: luoi.json scorer.nut, the scorer state of every node scored (C8BS-16)')
        out += [_iso_time(e.get('luc_chay') if isinstance(e, dict) else None, f'{w}: scorer.nut[{i}].luc_chay')
                for i, e in enumerate(nut)]
    return out


def _node_scores(goc: str, fc: dict, w: str, cell: dict, fg: Optional[dict]) -> Tuple[list, list]:
    """Every node score named by luoi.json nguon_nut [{nut, duong_dan, sha256}] (C8B node_sources: only the nodes with
    a score file, by node index, absolute paths inside the C8 folder accepted; GC9-1), hashed again and tied to its
    node and its run whether kept or lost (GC6-6): task of the cell; the log and frames of the time file of the
    scoring (esp32_sha256, frames_sha256); lost on the grid (hong_nut) exactly when the score is hong_he_do; offset_s
    (when stated) the node time; nodes_ok true exactly for a kept node. A kept node: khung_tuong_doi_v1 of this time
    file, mien_phai_phu {nut, D_s = nodes_s[k], id_khung, ids_sha} (CAM F13). A node without a score file is not
    scored (nodes_ok false) and relies on no frame. Returns (node scores, mien_phai_phu of the kept nodes)."""
    nn, nd, hn, ok = fc.get('nguon_nut'), fc.get('nodes_s'), fc.get('hong_nut'), fc.get('nodes_ok')
    if fg is not None and (not isinstance(nd, list) or not isinstance(fg.get('nodes_s'), list)
                           or len(nd) != len(fg['nodes_s']) or any(not _close(a, b) for a, b in zip(nd, fg['nodes_s']))):
        raise GateFileError(f'{w}: luoi.json nodes_s {nd!r} is not nodes_s of its time file (GC6-6)')
    if nn == [] and fc['trang_thai_luot'] != 'cham':
        return [], [], []
    if not isinstance(nn, list) or not isinstance(nd, list) or not isinstance(hn, list) or not isinstance(ok, list) \
            or not nn or not len(nd) == len(hn) == len(ok) or len(nn) > len(nd):
        raise GateFileError(f'{w}: luoi.json names the node scores (nguon_nut [{{nut, duong_dan, sha256}}]) with '
                            'hong_nut, nodes_ok, nodes_s of one length per node (C8B, F18)')
    if any(not isinstance(v, bool) for v in hn + ok):
        raise GateFileError(f'{w}: luoi.json hong_nut, nodes_ok: one true / false per node (GC6-6)')
    by_nut = {}
    for i, e in enumerate(nn):
        _obj(e, ('nut', 'duong_dan', 'sha256'), f'{w} nguon_nut[{i}]')
        k = e['nut']
        if type(k) is not int or not 0 <= k < len(nd) or k in by_nut:
            raise GateFileError(f'{w} nguon_nut[{i}].nut {k!r}: a node index, each once')
        by_nut[k] = _ref(dict(duong_dan=_rel_in(goc, e['duong_dan']), sha256=e['sha256']), f'{w} nguon_nut[{i}]')
    scores, out, kept, idx = [], [], set(), [None] * len(nd)
    for k in range(len(nd)):
        if k not in by_nut:                               # not scored: no score file (C8B loi_nut)
            if ok[k] or hn[k]:
                raise GateFileError(f'{w} node {k}: no score file but nodes_ok / hong_nut true (GC6-6)')
            continue
        ref = by_nut[k]
        sc = _open_ref(goc, ref, f'{w} node {k}')
        if sc.get('task') != cell['muc']:
            raise GateFileError(f'{w} node {k}: a score of task {sc.get("task")!r}, the cell is {cell["muc"]} (GC6-6)')
        if fg is not None and (sc.get('esp32_sha256') != fg['sha256']['esp32']
                               or sc.get('frames_sha256') != fg['sha256']['frames']):
            raise GateFileError(f'{w} node {k}: the node score read another log or frames than the time file of its '
                                'scoring (esp32_sha256, frames_sha256; GC6-6)')
        if hn[k] != (sc.get('outcome') == 'hong_he_do'):
            raise GateFileError(f'{w} node {k}: hong_nut {hn[k]} but the node score says {sc.get("outcome")!r} (GC6-6)')
        if 'offset_s' in sc and not (_isnum(sc['offset_s']) and _close(sc['offset_s'], nd[k])):
            raise GateFileError(f'{w} node {k}: offset_s {sc["offset_s"]!r} is not nodes_s[{k}] (GC6-6)')
        kt = sc.get('khung_tuong_doi_v1')
        scores.append(sc)
        idx[k] = sc
        if hn[k]:                                        # a node lost on the grid relies on no frame
            if ok[k]:
                raise GateFileError(f'{w} node {k}: lost (hong_nut) but nodes_ok true (GC6-6)')
            if isinstance(kt, dict) and kt.get('gio_sha256') not in (None, fc['gio_sha256']):
                raise GateFileError(f'{w} node {k}: a lost node score of another time file (GC6-6)')
            continue
        if not ok[k]:                                    # a score the grid could not use: relies on no frame
            continue
        if ref['sha256'] in kept:
            raise GateFileError(f'{w} node {k}: the same node score as another node (GC6-6)')
        kept.add(ref['sha256'])
        if not isinstance(kt, dict) or kt.get('gio_sha256') != fc['gio_sha256']:
            raise GateFileError(f'{w} node {k}: the node score (khung_tuong_doi_v1.gio_sha256) is not of the time file '
                                'of this scoring (GC6-6)')
        mpp = kt.get('mien_phai_phu')
        _obj(mpp, ('nut', 'D_s', 'id_khung', 'id_khung_sha256'), f'{w} node {k} mien_phai_phu')
        if mpp['nut'] != k or not _close(mpp['D_s'], nd[k]) or not isinstance(mpp['id_khung'], list) \
                or not mpp['id_khung'] or sha256_ids(mpp['id_khung']) != mpp['id_khung_sha256']:
            raise GateFileError(f'{w} node {k}: mien_phai_phu {{nut {k}, D_s = nodes_s[{k}], a non-empty id_khung, its '
                                'ids_sha}')
        out.append(mpp)
    return scores, out, idx


LOAI_HONG_C8B = ('thao_tac', 'cua_so_tham_chieu', 'duong_lenh', 'he_do')   # c8b_luoi LOAI_HONG order (J v6 5)


def _kind_of(sc: dict) -> Optional[str]:
    """The lost-run kind one node score gives (the rule of c8b_luoi.loai_hong_of, f613db9: from the scorer's fields
    only): thao_tac, cua_so_tham_chieu (hong_cua_so_tc), duong_lenh, he_do; None for a scored node."""
    fc = sc.get('fail_cause') or ''
    if fc == 'thao_tac' or sc.get('thao_tac'):
        return 'thao_tac'
    if sc.get('hong_cua_so_tc') is True:
        return 'cua_so_tham_chieu'
    if fc == 'duong_lenh':
        return 'duong_lenh'
    if fc == 'he_do' or sc.get('outcome') == 'hong_he_do':
        return 'he_do'
    return None


def _lost_kinds(fc: dict, fg: Optional[dict], ops: list, nodes: list, w: str) -> set:
    """The lost kinds of the RUN one scoring shows (GC6-2, GC9-5, GC9-6; F18 (1)). luoi.json loai_hong is recomputed
    from its node scores (the c8b_luoi rule: the kinds of the nodes, thao_tac when score.json names an operation, he_do
    for a grid lost at every node that shows none), never taken as stated. A grid scored (cham) may carry the kinds of
    its lost nodes: the run is not lost then (GC9-6). The run's kinds: those of a grid lost at every node, he_do when
    the time file is lost to the measurement, thao_tac when score.json names an operation (CODE 19/11: the one
    source)."""
    lost = fc['trang_thai_luot'] == 'hong_he_do'
    want = {k for k in (_kind_of(s) for s in nodes) if k} | ({'thao_tac'} if ops else set())
    if lost and not want:
        want.add('he_do')
    lh = fc.get('loai_hong')
    if lh != [k for k in LOAI_HONG_C8B if k in want]:
        raise GateFileError(f'{w}: luoi.json loai_hong {lh!r}; from its node scores and the operations of score.json '
                            f'{ops} it is {[k for k in LOAI_HONG_C8B if k in want]} (c8b_luoi rule; GC6-2, GC9-5)')
    if ops and not lost:
        raise GateFileError(f'{w}: score.json names the operations {ops} but the grid scored the run '
                            f'({fc["trang_thai_luot"]}): a run lost to an operation is lost (J v6 section 5; GC6-2)')
    s = set(lh) if lost else set()
    if fg is not None and fg.get('ket_luan') == 'hong_he_do':
        s.add('he_do')
    if ops:
        s.add('thao_tac')
    return s


GRID_STATES = ('dat', 'khong_dat', 'chua_ket_luan')


def _grid_outcome(fc: dict, nodes: list, w: str) -> None:
    """GC9-5: the outcome of a scored grid recomputed from its states (the c8b_luoi combine rule, f613db9): over the
    five conditions (every one decides since 23/11; a so_do one by its state from q) khong_dat, else chua_ket_luan,
    else dat; a violation before t1 at every node (fail_cause vi_pham_truoc_loi_he_do) is khong_dat only when every
    node score is khong_dat."""
    if fc['trang_thai_luot'] != 'cham':
        return
    if 'report_only' in fc:
        raise GateFileError(f'{w}: luoi.json report_only {fc.get("report_only")!r}: report only is no rule since 23/11 '
                            '(CODE, Liam "duyệt bảng luật mới"); the grid is scored again with so_do')
    st, sd = fc.get('states'), fc.get('so_do')
    if not isinstance(st, dict) or any(st.get(c) not in GRID_STATES for c in o1_gate.CONDS) \
            or not isinstance(sd, list) or any(c not in MASKABLE for c in sd):
        raise GateFileError(f'{w}: luoi.json states {{g1..g5: dat / khong_dat / chua_ket_luan}}, so_do of '
                            f'{MASKABLE} (GC9-5)')
    ds = [st[c] for c in o1_gate.CONDS]
    if fc.get('fail_cause') == 'vi_pham_truoc_loi_he_do':
        want = 'khong_dat' if nodes and all(s.get('outcome') == 'khong_dat' for s in nodes) else None
    else:
        want = 'khong_dat' if 'khong_dat' in ds else 'chua_ket_luan' if 'chua_ket_luan' in ds else 'dat'
    if fc.get('outcome') != want:
        raise GateFileError(f'{w}: luoi.json outcome {fc.get("outcome")!r}; from its states {st} (so_do {sd}) '
                            f'it is {want!r} (c8b_luoi combine; GC9-5)')


def _grid_states(fc: dict, idx: list, cell: dict, w: str) -> None:
    """GC9-5 (round 5): the states of a scored grid against its node states (the c8b_luoi combine rule, f613db9): a node
    not ok counts chua_ket_luan; G1 by g1_rule; a measured condition with one node is that node's state, dat only when
    every node is dat (a domain moving between nodes may still make it chua_ket_luan), else khong_dat or chua_ket_luan
    (the cell rule needs the cell numbers, not rebuilt here: limit). so_do is that of J for C8 (G5 at F08): its
    three-branch grid state (states_ba_nhanh) follows the node rule, its state is so_do_state of the grid numbers
    (nums: chung.q, q, n_undef; CODE 23/11, Liam "duyệt bảng luật mới")."""
    if fc['trang_thai_luot'] != 'cham':
        return
    sd = ['g5'] if cell['muc'] == G5_F08 else []
    if fc.get('so_do') != sd:
        raise GateFileError(f'{w}: luoi.json so_do {fc.get("so_do")!r}; for {cell["muc"]} J says {sd} (GC9-5)')
    sb, nums = fc.get('states_ba_nhanh'), fc.get('nums')
    if not isinstance(sb, dict) or set(sb) != set(sd) or any(sb[c] not in GRID_STATES for c in sd):
        raise GateFileError(f'{w}: luoi.json states_ba_nhanh {sb!r}: the three-branch state of each so_do condition '
                            f'{sd} (GC9-5; CODE 23/11)')
    ok, st = fc['nodes_ok'], dict(fc['states'])
    qsd = fc.get('q_so_do')
    if not isinstance(qsd, dict) or set(qsd) != set(sd):
        raise GateFileError(f'{w}: luoi.json q_so_do {qsd!r}: the middle node q of each so_do condition {sd} (CODE '
                            '23/11 (2))')
    m = _nut_giua(fc.get('nodes_s'), ok)
    for c in sd:
        b = nums.get(c) if isinstance(nums, dict) else None
        x = qsd[c]
        if not isinstance(x, dict) or x.get('nut') != m:
            raise GateFileError(f'{w}: luoi.json q_so_do.{c} {x!r}: the node is {m}, the scored node nearest the middle '
                                'of the grid (CODE 23/11 (2))')
        nb = (idx[m].get('gate_v3') or {}).get(c) if m is not None and isinstance(idx[m], dict) else None
        nq = nb.get('q_so_do') if isinstance(nb, dict) else None
        if m is not None and (not isinstance(nq, dict) or not _close(x.get('q'), nq.get('q'))
                              or not _close(x.get('q_chung'), nq.get('q_chung'))):
            raise GateFileError(f'{w}: luoi.json q_so_do.{c} {x!r}: not the q_so_do (q, q_chung of the defined '
                                f'frames) of node {m} (its score file; SD1-1)')
        if not isinstance(b, dict) or st[c] != so_do_state(x.get('q_chung'), x.get('q'), b.get('n_undef'),
                                                           o1_gate.THRESHOLDS[c]):
            raise GateFileError(f'{w}: luoi.json {c} state {st[c]!r}: so_do from the middle node q and the grid '
                                'n_undef gives another (GC9-5; CODE 23/11)')
        st[c] = sb[c]                                    # the node rule below checks the three-branch grid state
    for k, sc in enumerate(idx):                     # SD1-6 (Codex 24/11): every scored node with the so_do of J
        if not ok[k] or not isinstance(sc, dict):
            continue
        g = sc.get('gate_v3') if isinstance(sc.get('gate_v3'), dict) else {}
        want = (GATE_VERSION_SO_DO, sd, J_V6_SHA256) if sd else (None, [], None)
        got = (g.get('version') if sd else (GATE_VERSION_SO_DO if 'so_do' in g else None),
               list(g.get('so_do') or []), g.get('gate_file_sha256') or None)
        if got != want or 'report_only' in g or 'report_only' in sc:
            raise GateFileError(f'{w}: node {k} scored with so_do {got[1]} (source {got[2]}, version {g.get("version")!r}'
                                f'{", report_only" if "report_only" in g or "report_only" in sc else ""}); the cell '
                                f'needs {sd} from J v6 (SD1-6)')
    node_st = {}
    for c in o1_gate.CONDS:
        v = []
        for k, sc in enumerate(idx):
            g = sc.get('gate_v3') if ok[k] and isinstance(sc, dict) else None
            b = g.get(c) if isinstance(g, dict) else None
            if ok[k] and not (isinstance(b, dict) and b.get('state') in GRID_STATES):
                raise GateFileError(f'{w}: node {k} scored (nodes_ok) without a {c} state in its gate_v3 (GC9-5)')
            # a node gate of the scorer is three-branch; a so_do node gate keeps that state as state_ba_nhanh
            v.append(b.get('state_ba_nhanh', b['state']) if ok[k] else 'chua_ket_luan')
        node_st[c] = v
    g1 = node_st['g1']
    want = g1[0] if len(g1) == 1 else 'dat' if all(x == 'dat' for x in g1) else \
        'khong_dat' if all(x == 'khong_dat' for x in g1) else 'chua_ket_luan'
    bad = [c for c in o1_gate.CONDS if c != 'g1' and (
        (len(node_st[c]) == 1 and st[c] != node_st[c][0])
        or (st[c] == 'dat' and any(x != 'dat' for x in node_st[c]))
        or (st[c] != 'dat' and all(x == 'dat' for x in node_st[c]) and st[c] != 'chua_ket_luan'))]
    if st['g1'] != want or bad:
        raise GateFileError(f'{w}: luoi.json states {st} disagree with the node states {node_st} '
                            f'({"g1" if st["g1"] != want else ""} {bad}; c8b_luoi combine; GC9-5)')


def _grid_u_t(fc: dict, idx: list, w: str) -> bool:
    """GC10-3: U_t of the nodes (score.json U_t_s; a node without a score file has none) and u_max_s = their largest,
    null when a node has none (the c8b_luoi rule). Returns True when U_t is missing at a node (GC9-6)."""
    if fc['trang_thai_luot'] != 'cham':
        return False
    ut = [None if sc is None else sc.get('U_t_s') for sc in idx]
    ut = [v if _isnum(v) and math.isfinite(v) and v >= 0 else None for v in ut]
    got = fc.get('u_t_nodes')
    if not isinstance(got, list) or len(got) != len(ut) or any(not _close(a, b) for a, b in zip(got, ut)):
        raise GateFileError(f'{w}: luoi.json u_t_nodes {got!r}; the node scores give {ut} (U_t_s; GC10-3)')
    want = None if any(v is None for v in ut) or not ut else max(ut)
    if not _close(fc.get('u_max_s'), want):
        raise GateFileError(f'{w}: luoi.json u_max_s {fc.get("u_max_s")!r}; the largest node U_t is {want!r} (GC10-3)')
    return want is None


def _final_outcome(o0: str, o1: Optional[str], drift: dict) -> Optional[str]:
    """The outcome after the clip H comparison (O1 4, c7_calib_drift gop rule): clip H at the end missing ->
    chua_ket_luan (lost stays lost); below eps_H -> the first scoring; above -> lost in either is lost, equal is kept,
    different is chua_ket_luan. None when the second scoring is due but absent."""
    if drift['thieu_H1']:
        return 'hong_he_do' if o0 == 'hong_he_do' else 'chua_ket_luan'
    if not drift['vuot']:
        return o0
    if o1 is None:
        return None
    if 'hong_he_do' in (o0, o1):
        return 'hong_he_do'
    return o0 if o0 == o1 else 'chua_ket_luan'


def _clip_row(goc: str, ref: dict, w: str, rows: List[dict]) -> int:
    """The row of thu_tu_clip.csv of the clip of a session file: the one row whose clip name is a part of the video
    name of the session (GC6-3 round 2)."""
    s = _open_ref(goc, ref, w)
    base = os.path.basename(str(s.get('video', '')).replace('\\', '/'))
    hit = [i for i, r in enumerate(rows) if (r.get('clip') or '').strip() and re.search(
        r'(^|[_.-])' + re.escape(r['clip'].strip()) + r'($|[_.-])', base)]
    if len(hit) != 1:
        raise GateFileError(f'thu_tu_clip: {w} (video {s.get("video")!r}) is on {len(hit)} rows, one needed (the clips '
                            'H bound the camera segments; GC6-3)')
    return hit[0]


def _doan_spans(goc: str, segs: List[dict], rows: List[dict], slices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """[(row of its last clip H at the start, row of its clip H at the end)] of each camera segment (CODE 20/11 after
    C8 session 2), row indices of rows (_buoi_rows): its runs are the rows strictly between; the clips H of a segment
    are rows of its own session (CODE 22/11); the segments of a session in session order, never overlapping."""
    spans = []
    for n, s in enumerate(segs):
        w = f'tien_de_c8[{s["k"]}] segment {s["ten"]!r}'
        b0, b1 = slices[s['k']]
        sub = rows[b0:b1]
        at = [b0 + _clip_row(goc, x, f'{w} clip {r}', sub) for x, r in zip(s['clips'], TD_ROLES)]
        lo, hi = max(at[:2]), at[2]
        if not lo < hi or (n and segs[n - 1]['k'] == s['k'] and not spans[-1][1] < min(at[:2])):
            raise GateFileError(f'{w}: its clips H at the start come before its clip H at the end, after the end of the '
                                'segment before it in the thu_tu_clip.csv of its session (segments in session order, '
                                'not overlapping)')
        spans.append((lo, hi))
    return spans


DRIFT_VERSION = 'c7_calib_drift_v2'                       # CAM design v4 section 3 TC3-10 (CODE 20/11)
DRIFT_STATES = ('so', 'thieu', 'khong_ap_dung')


def _drift_vuot(dr: dict, eps: dict, w: str) -> bool:
    """GC9-3 (round 4, CODE 20/11; CAM v4 TC3-10, J v6 section 5): vuot of a relative clip H comparison recomputed with
    the thresholds of the GATE (camera.c8b.eps_H, from the C8 lock), never those the drift file states: true when a
    quantity measured (so) is above its threshold (position D_H_vi_tri_mm > eps_H.mm, heading D_H_huong_deg >
    eps_H.huong_deg) or a quantity is missing (thieu, conservative); false only when both are measured (or not
    applicable) and below."""
    tv, th = dr.get('trang_thai_vi_tri'), dr.get('trang_thai_huong')
    if tv not in DRIFT_STATES or th not in DRIFT_STATES or (tv, th) == ('khong_ap_dung', 'khong_ap_dung'):
        raise GateFileError(f'{w}: trang_thai_vi_tri / trang_thai_huong {tv!r} / {th!r}: one of {DRIFT_STATES}, not both '
                            'khong_ap_dung (TC3-10)')
    out = False
    for st, key, lim in ((tv, 'D_H_vi_tri_mm', eps['mm']), (th, 'D_H_huong_deg', eps['huong_deg'])):
        if st == 'so':
            v = dr.get(key)
            if not (_isnum(v) and math.isfinite(v) and v >= 0):
                raise GateFileError(f'{w}: {key} {v!r}: a number >= 0 when measured (TC3-10)')
            out = out or v > lim
        elif st == 'thieu':
            out = True
    if dr.get('vuot_do_thieu') is not ('thieu' in (tv, th)):
        raise GateFileError(f'{w}: vuot_do_thieu {dr.get("vuot_do_thieu")!r}: true exactly when a quantity is missing '
                            '(thieu; TC3-10, GC10-1)')
    return out


def _drift_file(goc: str, ref: dict, w: str, cell: dict, seg: dict, eps: dict) -> dict:
    dr = _open_ref(goc, ref, w)
    if dr.get('kind') != DRIFT_KIND or dr.get('version') != DRIFT_VERSION or not isinstance(dr.get('vuot'), bool) \
            or not isinstance(dr.get('thieu_H1'), bool):
        raise GateFileError(f'{w}: not a drift.json of c7_calib_drift so (kind {DRIFT_KIND}, version {DRIFT_VERSION}, '
                            'vuot, thieu_H1; F18 (2), CAM TC3-10)')
    if dr['thieu_H1'] and not dr['vuot']:
        raise GateFileError(f'{w}: thieu_H1 comes with vuot true')
    if not dr['thieu_H1']:                               # GC6-3, GC9-3: vuot recomputed with the gate thresholds
        want = _drift_vuot(dr, eps, w)
        if dr['vuot'] != want:
            raise GateFileError(f'{w}: vuot {dr["vuot"]!r}; from D_H (position, heading) and eps_H of the gate {eps} '
                                f'it is {want} (GC9-3)')
    # GC6-3 (round 2; CODE 20/11 after C8 session 2): the comparison of the run's camera segment: a clip H at its start
    # against its clip H at the end
    s0, s1 = dr.get('s0_session_sha256'), dr.get('s1_session_sha256')
    sh = [x['sha256'] for x in seg['clips']]
    if not (s0 in sh[:2] and (s1 == sh[2] or (dr['thieu_H1'] and s1 is None))):
        raise GateFileError(f'{w}: the clip H comparison of the camera segment {seg["ten"]!r} of the run compares a clip '
                            f'H at its start with its clip H at the end; this one compares {s0!r} / {s1!r} (GC6-3)')
    if not isinstance(dr.get('tasks'), list) or cell['muc'] not in dr['tasks']:
        raise GateFileError(f'{w}: the clip H comparison covers {dr.get("tasks")!r}, not the task {cell["muc"]}')
    return dr


def _g4_finite(fc: dict) -> bool:
    """G4 bound finite at every frame of the union domain at every node (J v6 section 5, J3-10), from the scoring
    file: every node ok and the combined G4 numbers finite with no undefined frame (the rule of c8b_edo chung-cu)."""
    nums = fc.get('nums').get('g4') if isinstance(fc.get('nums'), dict) else None
    ok = fc.get('nodes_ok')
    if not isinstance(ok, list) or not ok or not all(v is True for v in ok) or not isinstance(nums, dict):
        return False
    hi = nums.get('hi')
    return _isnum(hi) and math.isfinite(hi) and nums.get('n_undef') == 0


def _edo_set(fc: dict, ed: dict, w: str, cell: dict) -> dict:
    """One set of numbers of the evidence (F17 chung_cu_c8 bo_so) recomputed from a scoring file and its e_do file
    (c8b_edo luot: trang_thai, vuot_moi_nut, can_xac_dinh_moi_nut; GC6-3 round 3): e_do counts only when dat."""
    if ed.get('task') != cell['muc'] or ed.get('trang_thai') not in ('dat', 'chua_ket_luan') \
            or not isinstance(ed.get('vuot_moi_nut'), bool) or not isinstance(ed.get('can_xac_dinh_moi_nut'), bool):
        raise GateFileError(f'{w}: an e_do file of task {cell["muc"]} (c8b_edo luot: trang_thai dat / chua_ket_luan, '
                            'vuot_moi_nut, can_xac_dinh_moi_nut; GC6-3)')
    tn = ed.get('theo_nut')
    if not isinstance(tn, list) or not tn or any(not isinstance(n, dict) or not isinstance(n.get('cham'), bool)
                                                for n in tn):
        raise GateFileError(f'{w}: e_do file theo_nut [{{cham, ...}}] per node (c8b_edo luot; GC10-4)')
    if ed['vuot_moi_nut'] and not all(n['cham'] and n.get('khung_vuot') for n in tn):
        raise GateFileError(f'{w}: e_do vuot_moi_nut true with a node not scored or without a frame above D (GC10-4)')
    if ed['can_xac_dinh_moi_nut'] and not all(n['cham'] and n.get('n_undef') == 0 for n in tn):
        raise GateFileError(f'{w}: e_do can_xac_dinh_moi_nut true with a node not scored or undefined frames (GC10-4)')
    ok = ed['trang_thai'] == 'dat'
    return dict(e_do_lo_vuot_D_moi_nut=ok and ed['vuot_moi_nut'], e_do_xac_dinh_moi_nut=ok and ed['can_xac_dinh_moi_nut'],
                g4_can_huu_han_mien_hop_moi_nut=_g4_finite(fc))


def _classified(fc: dict) -> bool:
    st = fc.get('states')
    return fc['trang_thai_luot'] == 'cham' and isinstance(st, dict) and all(c in st for c in o1_gate.CONDS)


def _check_runs_nguon(d: dict, goc: str) -> Tuple[Dict[str, list], Dict[str, dt.datetime], set]:
    """The run sources of every C8 cell (GC6-1, GC6-2, GC6-3, GC6-6; F11, F17, F18). Returns ({scoring sha256 of an
    eligible run: mien_phai_phu of its kept nodes}, {run_id: time of its first scoring (script.luc_chay)}, {scorer
    commits of the scored grids})."""
    plan_cells = {c['o'] for c in d['o_c8']}
    segs = _check_tien_de(d)                             # CODE 20/11 after C8 session 2: the camera segments
    rows, slices = _buoi_rows(goc, d['thu_tu_clip'])     # CODE 22/11: every C8 session, in session order
    by_clip, order = {}, {o: [] for o in plan_cells}
    for i, r in enumerate(rows):
        full = (r.get('buoc') or '').strip()
        b = re.sub(r'_lai\d+$', '', full)                # a rerun row: <run code>_lai<n> (CODE 19/11)
        if b in plan_cells:
            j = len(order[b])                            # GC9-4: the run itself first, then _lai1, _lai2, ... in turn
            if full != (b if j == 0 else f'{b}_lai{j}'):
                raise GateFileError(f'thu_tu_clip {r["_vt"]}: buoc {full!r}; the {j + 1}th row of cell {b} must be '
                                    f'{b if j == 0 else f"{b}_lai{j}"!r} (the run first, then its reruns in turn; GC9-4)')
            order[b].append(r['clip'])
        elif RUN_CODE.match(b):
            raise GateFileError(f'thu_tu_clip {r["_vt"]}: run {b!r} (clip {r.get("clip")!r}) is no cell of the plan '
                                '(a figure-8 run outside the locked list)')
        elif full.startswith('C8-') and not RUN_CODE_K.match(full) and not CLIP_KHOA_K.match(full):
            # CODE 20/11 (the luot_c8 check of c8b_u cong, C8BU4-R1-3, now here): a mistyped run code would drop a run
            raise GateFileError(f'thu_tu_clip {r["_vt"]}: buoc {full!r} (clip {r.get("clip")!r}) reads as a C8 run '
                                'code but is neither a run of the plan (or its _lai<n>) nor a K run: a mistyped code '
                                'would leave a run out of the gate')
        else:
            continue
        if r['clip'] in by_clip:
            raise GateFileError(f'thu_tu_clip: clip {r["clip"]!r} twice ({by_clip[r["clip"]]["_vt"]}, {r["_vt"]})')
        by_clip[r['clip']] = r
    eligible, cham_luc, logs_seen, clip_shas, scorers, elig_seg = {}, {}, {}, {}, set(), {}
    spans = _doan_spans(goc, segs, rows, slices)
    part_drift = {}                                      # GC6-3 (round 3): one comparison per camera segment
    for c in d['o_c8']:
        got = [x['clip'] for x in c['chuoi']]
        if got != order[c['o']]:
            raise GateFileError(f'cell {c["o"]}: chain clips {got} but thu_tu_clip.csv has {order[c["o"]]} for this '
                                'cell, in that order: every physical run of the cell, none left out, none added (GC6-1)')
        for x in c['chuoi']:
            w = f'cell {c["o"]} run {x["run_id"]}'
            row = by_clip[x['clip']]
            ri = rows.index(row)
            part = next((k for k, (lo, hi) in enumerate(spans) if lo < ri < hi), None)
            if part is None:
                raise GateFileError(f'{w}: its row of thu_tu_clip.csv lies in no camera segment of tien_de_c8.doan '
                                    '(between the clips H at the start and the clip H at the end of a segment; J3-5)')
            fg, log, tops = None, None, None
            if x['file_gio'] is not None:
                fg, log, tt = _time_file(goc, x['file_gio'], f'{w}: file_gio', c)
                esp = fg['sha256']['esp32']
                if esp in logs_seen:                      # GC6-1 (round 2): the ESP32 log is that of one run only
                    raise GateFileError(f'{w}: its time file reads the ESP32 log of {logs_seen[esp]} (one physical run, '
                                        'one log; GC6-1)')
                logs_seen[esp] = w
                if x['file_log']['sha256'] != esp or os.path.basename(x['file_log']['duong_dan']) != log:
                    raise GateFileError(f'{w}: file_log {x["file_log"]["duong_dan"]!r} is not the ESP32 log the time '
                                        f'file read ({log!r}, sha256.esp32; GC6-1)')
                # GC6-2 (round 3, CODE 19/11): the operations of the run are those of score.json (thao_tac_nguon)
                sj = _open_ref(goc, x['file_thao_tac'], f'{w}: file_thao_tac')
                tops = sj.get('thao_tac') or []
                if x['file_thao_tac']['sha256'] != fg['thao_tac_nguon']['sha256'] or sj.get('esp32_sha256') != esp \
                        or not isinstance(tops, list) or any(v not in THAO_TAC for v in tops) \
                        or len(set(tops)) != len(tops):
                    raise GateFileError(f'{w}: file_thao_tac is not the score.json of this run the time file took its '
                                        f'operations from (thao_tac_nguon, esp32_sha256, thao_tac of {THAO_TAC}; '
                                        'GC6-2)')
                tops = sorted(tops)
                if tt != tops:
                    raise GateFileError(f'{w}: the time file states the operations {tt}, score.json {tops} (score.json '
                                        'wins: the time file was made from another list; GC6-2)')
                _m2(fg, tops, f'{w}: file_gio')
            # GC6-1 (rounds 3, 4): the log itself, hashed, named as thu_tu_clip.csv names the log of the clip, also on a
            # lost run without a time file
            lp = os.path.basename((row.get('log_pi') or '').strip().replace('\\', '/'))
            if x['file_log'] is None:
                if lp:
                    raise GateFileError(f'{w}: thu_tu_clip.csv names the log {lp!r} for clip {x["clip"]!r}; the chain '
                                        'carries it (file_log; GC6-1)')
            else:
                _ref_bytes(goc, x['file_log'], f'{w}: file_log')
                bf = _folder_of(d['thu_tu_clip'][row['_buoi']])
                if not _under(x['file_log']['duong_dan'], bf):
                    raise GateFileError(f'{w}: file_log {x["file_log"]["duong_dan"]!r} is not in the folder {bf!r} of '
                                        'the session of its row (CODE 22/11)')
                if os.path.basename(x['file_log']['duong_dan']) != lp:
                    raise GateFileError(f'{w}: file_log {x["file_log"]["duong_dan"]!r}: thu_tu_clip.csv names '
                                        f'{row.get("log_pi")!r} for clip {x["clip"]!r} (GC6-1)')
                if fg is None:
                    if x['file_log']['sha256'] in logs_seen:
                        raise GateFileError(f'{w}: the ESP32 log of {logs_seen[x["file_log"]["sha256"]]} (GC6-1)')
                    logs_seen[x['file_log']['sha256']] = w
            fc = _scoring_file(goc, x['file_cham'], f'{w}: file_cham', c, x['file_gio'], x['clip'])
            nodes0, mpp0, idx0 = _node_scores(goc, fc, f'{w}: file_cham', c, fg)
            scorings = [(fc, nodes0, mpp0)]
            idxs = [idx0]
            fc1 = fg1 = None
            if x['file_cham_h1'] is not None:
                fg1, _, tt1 = _time_file(goc, x['file_gio_h1'], f'{w}: file_gio_h1', c)
                if fg1['sha256']['esp32'] != fg['sha256']['esp32']:
                    raise GateFileError(f'{w}: the second time file reads another ESP32 log (same run, GC6-3)')
                fc1 = _scoring_file(goc, x['file_cham_h1'], f'{w}: file_cham_h1', c, x['file_gio_h1'], x['clip'])
                nodes1, mpp1, idx1 = _node_scores(goc, fc1, f'{w}: file_cham_h1', c, fg1)
                scorings.append((fc1, nodes1, mpp1))
                idxs.append(idx1)
            # GC6-1 (round 3): one clip per physical run (luoi.json clip_sha256), the same clip for both scorings
            csh = {s[0]['clip']['clip_sha256'] for s in scorings}
            if len(csh) != 1 or next(iter(csh)) in clip_shas:
                raise GateFileError(f'{w}: luoi.json clip_sha256 {sorted(csh)}: one clip for the run, never another '
                                    "run's (GC6-1)")
            clip_shas[next(iter(csh))] = w
            # GC6-2 (round 3): every node score states the operations of the run (score.json, the one source)
            if tops is None:                             # no time file: the node scores are the source
                seen_ops = {tuple(sorted(s.get('thao_tac') or [])) for _, nodes, _ in scorings for s in nodes}
                if len(seen_ops) > 1:
                    raise GateFileError(f'{w}: the node scores state different operations {sorted(seen_ops)} (GC6-2)')
                tops = list(next(iter(seen_ops))) if seen_ops else []
            for _, nodes, _ in scorings:
                if any(sorted(s.get('thao_tac') or []) != tops for s in nodes):
                    raise GateFileError(f'{w}: a node score states other operations than score.json {tops} (GC6-2)')
            if fg1 is not None:
                allowed = {x['file_thao_tac']['sha256']} | {e['sha256'] for s in scorings for e in s[0]['nguon_nut']}
                if tt1 != tops or fg1['thao_tac_nguon']['sha256'] not in allowed:
                    raise GateFileError(f'{w}: the second time file states the operations {tt1} from another '
                                        f'source; the run has {tops} (GC6-2)')
                _m2(fg1, tops, f'{w}: file_gio_h1')
            src = set()
            thieu = False
            for (fcx, ndx, _), fgx, ix in zip(scorings, (fg, fg1), idxs):
                src |= _lost_kinds(fcx, fgx, tops, ndx, f'{w}: scoring')
                _grid_outcome(fcx, ndx, f'{w}: scoring')
                _grid_states(fcx, ix, c, f'{w}: scoring')
                thieu |= _grid_u_t(fcx, ix, f'{w}: scoring')
                if fcx['trang_thai_luot'] == 'cham':      # D4 (C8B 19/11): the scoring code that actually ran
                    sc = fcx.get('scorer')
                    nut = sc.get('nut') if isinstance(sc, dict) else None
                    if not isinstance(nut, list) or not nut or any(
                            not isinstance(e, dict) or e.get('dirty') is not False or e.get('commit') != sc.get('commit')
                            for e in nut):
                        raise GateFileError(f'{w}: luoi.json scorer.nut: the scorer state of every node scored, one clean '
                                            'commit, that of scorer (c8b_luoi; GC10-2)')
                    if not isinstance(sc, dict) or sc.get('dirty') is not False:
                        raise GateFileError(f'{w}: luoi.json scorer {{commit, dirty false}}: the clean scoring code of '
                                            'the grid (C8BS-16; D4 of the U standard)')
                    scorers.add(sc.get('commit'))
            if x['thieu_u_t'] != (thieu and not src):
                raise GateFileError(f'{w}: thieu_u_t {x["thieu_u_t"]!r}; the grid {"lacks" if thieu else "has"} U_t at '
                                    'every node (u_max_s; GC9-6: a run not lost without U_t is chua_ket_luan)')
            cham_luc[x['run_id']] = min(t for s in scorings for t in _scoring_times(s[0], f'{w}: scoring'))
            # GC6-2: the lost kinds are those the sources show, both ways
            if x['ket_qua'] == 'hong':
                if set(x['loai_hong']) != src:
                    raise GateFileError(f'{w}: loai_hong {sorted(x["loai_hong"])} but the sources show {sorted(src)} '
                                        '(scoring file loai_hong, score.json operations, time file ket_luan; GC6-2, '
                                        'F18 (1))')
            elif src:
                raise GateFileError(f'{w}: ket_qua {x["ket_qua"]!r} but the sources show the run lost {sorted(src)}')
            if fg is not None:
                k1 = fg['kiem_gio_d1']
                if k1.get('trang_thai') != x['kiem_gio_d1']:
                    raise GateFileError(f'{w}: kiem_gio_d1 {x["kiem_gio_d1"]!r}, the time file says '
                                        f'{k1.get("trang_thai")!r} (F11)')
                if fg['dem_truot_kiem_gio'] != x['dem_truot_kiem_gio']:
                    raise GateFileError(f'{w}: dem_truot_kiem_gio {x["dem_truot_kiem_gio"]!r}, the time file says '
                                        f'{fg["dem_truot_kiem_gio"]!r} (F11, M2; CODE_C8B design v7 7a)')
            # GC6-3: the outcome is the one after the clip H comparison of the part of the session
            dr = None
            if x['file_so_h'] is not None:
                if part_drift.setdefault(part, x['file_so_h']['sha256']) != x['file_so_h']['sha256']:
                    raise GateFileError(f'{w}: another clip H comparison than the other runs of the camera segment '
                                        f'{segs[part]["ten"]!r} (one per segment; GC6-3)')
                dr = _drift_file(goc, x['file_so_h'], f'{w}: file_so_h', c, segs[part], d['camera']['c8b']['eps_H'])
                for (_, ndx, _), key, tag in list(zip(scorings, ('s0_session_sha256', 's1_session_sha256'),
                                                      ('H0', 'H1')))[1:]:
                    for sc_ in ndx:
                        if sc_.get('outcome') != 'hong_he_do' and sc_.get('session_sha256') != dr.get(key):
                            raise GateFileError(f'{w}: a node score of the {tag} scoring was made with session '
                                                f'{str(sc_.get("session_sha256"))[:12]}..., the clip H comparison '
                                                f'names {str(dr.get(key))[:12]}... ({key}; GC6-3, CODE 20/11)')
                two = dr['vuot'] and not dr['thieu_H1']
                if x['ket_qua'] != 'hong' and two != (fc1 is not None):
                    raise GateFileError(f'{w}: the clip H comparison says {"two scorings" if two else "one scoring"} '
                                        f'(vuot {dr["vuot"]}, thieu_H1 {dr["thieu_H1"]}), the chain gives '
                                        f'{"two" if fc1 is not None else "one"} (GC6-3)')
                if not two and fc1 is not None:
                    raise GateFileError(f'{w}: a second scoring while the clip H comparison asks one')
                final = _final_outcome(fc['outcome'], None if fc1 is None else fc1['outcome'], dr)
                if final == 'hong_he_do' and x['ket_qua'] != 'hong':
                    raise GateFileError(f'{w}: lost after the clip H comparison, the gate says {x["ket_qua"]!r}')
                if x['thieu_u_t'] and final != 'hong_he_do':
                    final = 'chua_ket_luan'                   # GC9-6 (Liam 20/11)
                if x['ket_qua'] != 'hong' and x['ket_qua'] != final:
                    raise GateFileError(f'{w}: ket_qua {x["ket_qua"]!r}, after the clip H comparison the outcome is '
                                        f'{final!r} (first {fc["outcome"]!r}, second '
                                        f'{None if fc1 is None else fc1["outcome"]!r}; GC6-3)')
            if c['chuoi'][-1] is x and c['trang_thai'] == 'co_luot':
                r = next(r for r in d['runs_c8'][c['muc']] if r['run_id'] == x['run_id'])
                if not _close(fg.get('h_s'), r['h_s']) or not _close(fc.get('u_max_s'), r['u_max_s']):
                    raise GateFileError(f'{w}: h_s / u_max_s {r["h_s"]!r} / {r["u_max_s"]!r}, the time file says h_s '
                                        f'{fg.get("h_s")!r}, the scoring file u_max_s {fc.get("u_max_s")!r} (F17)')
                h1, u1 = (None, None) if fc1 is None else (fg1.get('h_s'), fc1.get('u_max_s'))
                if fc1 is not None and not (_isnum(h1) and _isnum(u1) and _close(h1 + u1, r['u_t_s'])):
                    raise GateFileError(f'{w}: the two scorings give two U_t (h_s + u_max_s); J v6 does not say which '
                                        'one rule 2b takes: C8 is not locked, CAN LIAM QUYET (J section 0b; GC6-3)')
                cc = _open_ref(goc, x['file_chung_cu'], f'{w}: file_chung_cu')
                # GC6-3 (round 2): each scoring with its e_do file, made from the same time file and log; (round 3) the
                # evidence recomputed from those pairs, never taken from the evidence file alone
                pairs = [(x['file_cham'], x['file_edo'], x['file_gio'])]
                if fc1 is not None:
                    pairs.append((x['file_cham_h1'], x['file_edo_h1'], x['file_gio_h1']))
                bo = []
                for (cref, eref, gref), (fcx, _, _), tag in zip(pairs, scorings, ('file_edo', 'file_edo_h1')):
                    ed = _open_ref(goc, eref, f'{w}: {tag}')
                    if ed.get('gio_sha256') != gref['sha256'] or ed.get('esp32_sha256') != fg['sha256']['esp32']:
                        raise GateFileError(f'{w}: {tag} was not made from the time file and log of its scoring '
                                            '(gio_sha256, esp32_sha256; GC6-3)')
                    bo.append(_edo_set(fcx, ed, f'{w}: {tag}', c))
                want_all = [p[0]['sha256'] for p in pairs] + [p[1]['sha256'] for p in pairs]
                srcs = [s.get('sha256') for s in cc.get('nguon') or [] if isinstance(s, dict)]
                if not isinstance(cc.get('chung_cu_c8'), dict) or srcs != want_all:
                    raise GateFileError(f'{w}: file_chung_cu is a c8b_edo chung-cu output (chung_cu_c8) made from '
                                        'exactly the scorings of this run then their e_do files, in that order (nguon; '
                                        'F17, GC6-3)')
                want_cc = dict(phan_loai_du=all(_classified(s[0]) for s in scorings),
                               ban_cham_cuoi=not dr['thieu_H1'], bo_so=bo)
                if cc['chung_cu_c8'] != r['chung_cu'] or r['chung_cu'] != want_cc:
                    raise GateFileError(f'{w}: chung_cu {r["chung_cu"]!r}; its evidence file says '
                                        f'{cc["chung_cu_c8"]!r}, recomputed from the scorings, their e_do files and the '
                                        f'clip H comparison it is {want_cc!r} (F17, GC6-3)')
                for s in scorings:
                    if not s[2]:
                        raise GateFileError(f'{w}: no node relies on any frame (every node lost): not a scoring of an '
                                            'eligible run')
                eligible[x['file_cham']['sha256']] = mpp0
                elig_seg[x['file_cham']['sha256']] = segs[part]
                if fc1 is not None:
                    eligible[x['file_cham_h1']['sha256']] = scorings[1][2]
                    elig_seg[x['file_cham_h1']['sha256']] = segs[part]
    # CODE 20/11 (the luot_c8 check of c8b_u cong, C8BU4-R1-3, now here): each run of the plan is scored once; its
    # scoring is the one first scoring of its clip under the C8 folder, and the clip H comparison, second scoring and
    # second time file of the chain are those that scoring names (so its time files are those of its gio_sha256)
    kl = _open_ref(goc, d['khoa_c8b'], 'khoa_c8b')
    commit = kl.get('commit_hil') if kl.get('nhanh_tick') == 'bo_tick' else None
    first = _first_scorings(goc, commit)                 # KG-6: only the scorings at commit_hil count
    for c in d['o_c8']:
        for x in c['chuoi']:
            w = f'cell {c["o"]} run {x["run_id"]}'
            _one_first_scoring(first, x, w)
            fc = _open_ref(goc, x['file_cham'], f'{w}: file_cham')
            for k in ('file_so_h', 'file_cham_h1', 'file_gio_h1'):
                a, b = x[k], fc.get(k)
                if (a is None) != (b is None) or (a is not None and (not isinstance(b, dict)
                                                                     or b.get('sha256') != a['sha256'])):
                    raise GateFileError(f'{w}: {k} is not the one its scoring (luoi.json) names: the chain takes the '
                                        'files of that scoring only (CODE 20/11)')
    # GC9-2: the K runs, in the order of thu_tu_clip.csv, each with its grid (scoring time and scoring code)
    k_rows = [r for r in rows if RUN_CODE_K.match((r.get('buoc') or '').strip())]
    if [(r['buoc'].strip(), r['clip']) for r in k_rows] != [(x['run_id'], x['clip']) for x in d['luot_k']]:
        raise GateFileError(f'luot_k {[x["run_id"] for x in d["luot_k"]]}: thu_tu_clip.csv has the K runs '
                            f'{[r["buoc"].strip() for r in k_rows]} with their clips, in that order (GC9-2)')
    for x in d['luot_k']:
        w = f'K run {x["run_id"]}'
        task = 'O1-' + x['run_id'].split('-')[2]
        fks = []
        for key in ('file_cham', 'file_cham_h1'):
            if x[key] is None:
                continue
            fk = _open_ref(goc, x[key], f'{w}: {key}')
            cl = fk.get('clip')
            if fk.get('trang_thai_luot') not in LUOI_STATES or fk.get('task') != task or not isinstance(cl, dict) \
                    or cl.get('ten') != x['clip'] or not _hex64(cl.get('clip_sha256')):
                raise GateFileError(f'{w}: {key} is no c8b_luoi file of task {task} for clip {x["clip"]!r} (GC9-2)')
            fks.append(fk)
        h1 = fks[0].get('file_cham_h1')
        if (h1 is None) != (x['file_cham_h1'] is None) or (h1 is not None and (
                not isinstance(h1, dict) or h1.get('sha256') != x['file_cham_h1']['sha256'])):
            raise GateFileError(f'{w}: file_cham_h1 is the second scoring its grid names (GC9-2)')
        if len({f['clip']['clip_sha256'] for f in fks}) != 1 or fks[0]['clip']['clip_sha256'] in clip_shas:
            raise GateFileError(f'{w}: one clip for the run, never another run\'s (GC6-1, GC9-2)')
        clip_shas[fks[0]['clip']['clip_sha256']] = w
        for fk in fks:
            _iso_time((fk.get('script') or {}).get('luc_chay'), f'{w}: script.luc_chay')
            if fk['trang_thai_luot'] == 'cham':
                sc = fk.get('scorer')
                nut = sc.get('nut') if isinstance(sc, dict) else None
                if not isinstance(nut, list) or not nut or sc.get('dirty') is not False or any(
                        not isinstance(e, dict) or e.get('dirty') is not False or e.get('commit') != sc.get('commit')
                        for e in nut):
                    raise GateFileError(f'{w}: luoi.json scorer: one clean commit of every node (GC10-2)')
                scorers.add(sc.get('commit'))
        cham_luc[x['run_id']] = min(t for fk in fks for t in _scoring_times(fk, f'{w}: scoring'))
        _one_first_scoring(first, x, w)
    return eligible, cham_luc, scorers, elig_seg


def luoi_commit(x: dict) -> Optional[str]:
    """The commit a c8b_luoi file was made at (script.commit), None when not stated."""
    sc = x.get('script')
    return sc.get('commit') if isinstance(sc, dict) else None


def _check_mien_theo_buoi(d: dict, khoa: dict, elig_seg: dict) -> None:
    """KG-5 (Codex N1 + merged lock round 1; C8B 24/11: c8b_khoa_gop 887c109 writes mien_car_pose_theo_buoi
    {session id: {side_m, phien_ban, h_m, kappa}}, mien_car_pose being their hull): with several C8 sessions the lock
    names the camera domain of each session of the gate, and the car pose of the U manifests (one for every scoring)
    lies in the domain of every session that has an eligible scoring, not only in the hull."""
    mcb = khoa.get('mien_car_pose_theo_buoi')
    buois = [t['buoi'] for t in d['tien_de_c8']]
    if mcb is None and len(buois) == 1:
        return                                          # one session: its domain is mien_car_pose itself
    if not isinstance(mcb, dict) or set(mcb) != set(buois):
        raise GateFileError(f'khoa_c8b.mien_car_pose_theo_buoi: the camera domain of each C8 session of the gate '
                            f'{buois}, keyed by session id (KG-5)')
    cp = car_pose_v2_of(d['camera']['c8b']['moc_dich']['nguon']['car_pose'], 'camera.c8b.moc_dich.nguon')
    for b in sorted({seg['buoi'] for seg in elig_seg.values()}):
        w = f'khoa_c8b.mien_car_pose_theo_buoi.{b}'
        m = _obj(mcb[b], ('side_m', 'phien_ban', 'h_m', 'kappa'), w)
        hl, hh = _range(m['h_m'], f'{w}.h_m')
        kl, kh = _range(m['kappa'], f'{w}.kappa')
        if m['side_m'] != cp['side_m'] or m['phien_ban'] != cp['phien_ban'] or not hl <= cp['h_m'] <= hh \
                or not kl <= cp['kappa'] <= kh:
            raise GateFileError(f'camera.c8b.moc_dich.nguon.car_pose: outside the camera domain of session {b} '
                                '(inside the hull of the merged lock only; KG-5)')


def scorings_dropped(goc: str, commit: Optional[str]) -> List[dict]:
    """KG-6 (Codex 24/11; CODE 24/11, Liam A2 "Một khóa, lấy lớn nhất"): with a bo_tick lock (commit = its
    commit_hil), the c8b_luoi luoi.json files under the C8 folder (also in folders _...) made at another commit: the
    scorings before the N1 fix and the merged lock, dropped, never a scoring of the gate. [{duong_dan, sha256,
    script_commit}] by path; [] when commit is None (giu_tick)."""
    if commit is None:
        return []
    out = []
    base = os.path.realpath(goc)
    for root, dirs, names in os.walk(base):
        dirs.sort()
        if 'luoi.json' not in names:
            continue
        p = os.path.join(root, 'luoi.json')
        try:
            with open(p, 'rb') as f:
                raw = f.read()
            x = json.loads(raw.decode('utf-8'))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if isinstance(x, dict) and 'trang_thai_luot' in x and luoi_commit(x) != commit:
            out.append(dict(duong_dan=os.path.relpath(p, base).replace(os.sep, '/'),
                            sha256=hashlib.sha256(raw).hexdigest(), script_commit=luoi_commit(x)))
    return sorted(out, key=lambda e: e['duong_dan'])


def _first_scorings(goc: str, commit: Optional[str] = None) -> Dict[str, set]:
    """{clip name: sha256 of every first scoring of it} under the C8 folder: the c8b_luoi luoi.json files (as ghep
    finds them: named luoi.json, with trang_thai_luot) that no other luoi.json names as its second scoring
    (file_cham_h1). Copies with the same bytes are one scoring. A file that is no UTF-8 JSON object is skipped.
    commit (KG-6): with a bo_tick lock its commit_hil; a file made at another commit is no scoring (scorings_dropped)."""
    found, h1 = [], set()
    for root, dirs, names in os.walk(os.path.realpath(goc)):
        dirs.sort()
        if 'luoi.json' not in names:
            continue
        try:
            with open(os.path.join(root, 'luoi.json'), 'rb') as f:
                raw = f.read()
            x = json.loads(raw.decode('utf-8'))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if not isinstance(x, dict) or 'trang_thai_luot' not in x:
            continue
        if commit is not None and luoi_commit(x) != commit:
            continue                                     # KG-6: a dropped scoring (scorings_dropped)
        found.append((hashlib.sha256(raw).hexdigest(), x))
        if isinstance(x.get('file_cham_h1'), dict):
            h1.add(x['file_cham_h1'].get('sha256'))
    out: Dict[str, set] = {}
    for sha, x in found:
        cl = x.get('clip')
        if sha not in h1 and isinstance(cl, dict) and isinstance(cl.get('ten'), str):
            out.setdefault(cl['ten'], set()).add(sha)
    return out


def _one_first_scoring(first: Dict[str, set], x: dict, w: str) -> None:
    """CODE 20/11 (the luot_c8 check of c8b_u cong): the scoring of run x is the only first scoring of its clip."""
    have = first.get(x['clip'], set())
    if have != {x['file_cham']['sha256']}:
        raise GateFileError(f'{w}: clip {x["clip"]!r} has {len(have)} first scoring(s) (luoi.json) under the C8 folder'
                            f'{"" if x["file_cham"]["sha256"] in have else ", none of them file_cham"}: a run is '
                            'scored once and the gate takes that scoring (CODE 20/11; luot_c8 of c8b_u cong)')


def _check_khoa_c8b(d: dict, goc: str) -> Tuple[dict, dt.datetime]:
    """The C8b lock written before the session (GC6-5, F18 (4); c8b_khoa_v5 of CODE_C8B 47726c1; names of C8B's M5b
    passed on by CODE 19/11: muc.<level>.hang_theo_u [{u_lo_m_s, u_hi_m_s, P4_m, A4_deg, trang_thai}], nhanh_tick;
    the constant closed terms muc.<level>.hang_dong {a_m, hang_m, he_so_r, u_phi_rad, hang_rad} named by GATE, waiting
    for C8B): the gate copies them, never states them itself; the plan is anchored in khoa_truoc (GC6-1). The camera
    block of the gate is that of this lock (khoa_sha256, mien_car_pose, eps_H; GATE to C8B 19/11). Returns (the lock,
    khoa_truoc.luc)."""
    k = _open_ref(goc, d['khoa_c8b'], 'khoa_c8b')
    c8b = d['camera']['c8b']
    if c8b['khoa_sha256'] != d['khoa_c8b']['sha256'] or k.get('mien_car_pose') != c8b['mien_car_pose'] \
            or k.get('eps_H') != c8b['eps_H']:
        raise GateFileError('camera.c8b: khoa_sha256 is the sha256 of the C8b lock (khoa_c8b), mien_car_pose and eps_H '
                            'those of the lock (GATE to C8B 19/11)')
    luc = _iso_time((k.get('khoa_truoc') or {}).get('luc') if isinstance(k.get('khoa_truoc'), dict) else None,
                    'khoa_c8b: khoa_truoc.luc')
    if k.get('version') != KHOA_FORMAT:
        raise GateFileError(f'khoa_c8b: version {k.get("version")!r}, a {KHOA_FORMAT} file')
    if k.get('trang_thai') != 'khoa':
        raise GateFileError(f'khoa_c8b: trang_thai {k.get("trang_thai")!r} ({k.get("ly_do_chua_khoa")!r}): the C8b lock '
                            'is not locked, C8 is not locked')
    kt = k.get('khoa_truoc')
    src = kt.get('nguon') if isinstance(kt, dict) else None
    if not isinstance(src, list) or not any(isinstance(n, dict) and n.get('vai') == 'ke_hoach_c8'
                                            and n.get('sha256') == d['ke_hoach_c8']['sha256'] for n in src):
        raise GateFileError('khoa_c8b: khoa_truoc.nguon does not name the plan of the gate (ke_hoach_c8 sha256): the '
                            'plan is not anchored in the lock written before the session (GC6-1)')
    bm, need = k.get('bang_man_trap'), list(d['muc_kha_thi']) + [TASK_K]
    if isinstance(bm, dict):                             # GC9-1: C8B writes nguon_3b with an absolute path
        bm = {q: (dict(v, nguon_3b=dict(v['nguon_3b'], duong_dan=_rel_in(goc, v['nguon_3b'].get('duong_dan'))))
                  if isinstance(v, dict) and isinstance(v.get('nguon_3b'), dict) else v) for q, v in bm.items()}
    if not isinstance(bm, dict) or any(q not in bm for q in need) \
            or {q: bm[q] for q in need} != d['bo_cham']['bang_man_trap']:
        raise GateFileError('bo_cham.bang_man_trap differs from bang_man_trap of the C8b lock on the levels of '
                            f'muc_kha_thi and K {need} (GC6-5, GC7-5)')
    if k.get('nhanh_tick') != d['u_t_branch']:
        raise GateFileError(f'u_t_branch {d["u_t_branch"]!r}, the C8b lock says nhanh_tick {k.get("nhanh_tick")!r} '
                            '(GC6-5)')
    muc = k.get('muc')
    if not isinstance(muc, dict) or any(lv not in muc or not isinstance(muc[lv], dict) for lv in d['muc_kha_thi']):
        raise GateFileError(f'khoa_c8b: muc has no block for every level of muc_kha_thi {d["muc_kha_thi"]}')
    if G5_F08 in muc and 'g5' not in (muc[G5_F08].get('so_do') or []):
        raise GateFileError('khoa_c8b: G5 at F08 not in so_do (J v6 section 6; decided by q since 23/11, CODE, Liam)')
    for lv in d['levels']:
        e, w = d['dieu_2b']['per_level'][lv], f'dieu_2b.per_level.{lv}'
        hd, bands = muc[lv].get('hang_dong'), muc[lv].get('hang_theo_u')
        if not isinstance(hd, dict) or not isinstance(bands, list):
            raise GateFileError(f'{w}: the C8b lock has no muc.{lv}.hang_dong / hang_theo_u (missing; C8B M5b, F18 (4)):'
                                ' the closed terms of the gate cannot be tied to the lock, C8 is not locked')
        mine = dict(a_m=e['a_m'], hang_m=e['u_p']['hang_m'], he_so_r=e['u_p']['he_so_r'],
                    u_phi_rad=e['u_p']['u_phi_rad'], hang_rad=e['u_psi']['hang_rad'])
        if set(hd) != set(mine) or any(not _close(hd[q], mine[q]) for q in mine):
            raise GateFileError(f'{w}: a_m, hang_m, he_so_r, u_phi_rad, hang_rad {mine} differ from muc.{lv}.hang_dong '
                                f'of the C8b lock {hd} (GC6-5)')
        p_b, s_b = [], []
        for i, b in enumerate(bands):
            if not isinstance(b, dict) or any(q not in b for q in BAND_KEYS_C8B):
                raise GateFileError(f'khoa_c8b muc.{lv}.hang_theo_u[{i}]: an object with {BAND_KEYS_C8B}')
            if b['trang_thai'] not in BAND_STATES_C8B or not (_isnum(b['u_lo_m_s']) and _isnum(b['u_hi_m_s'])):
                raise GateFileError(f'khoa_c8b muc.{lv}.hang_theo_u[{i}]: u_lo_m_s, u_hi_m_s numbers, trang_thai one of '
                                    f'{BAND_STATES_C8B} (GC7-2)')
            gone = b['trang_thai'] == 'thieu'
            for q in ('P4_m', 'A4_deg'):
                if not ((gone and b[q] is None) or (_isnum(b[q]) and math.isfinite(b[q]) and b[q] >= 0)):
                    raise GateFileError(f'khoa_c8b muc.{lv}.hang_theo_u[{i}].{q} {b[q]!r}: a number >= 0, null only '
                                        'when trang_thai is thieu (GC7-2)')
            p_b.append((b['u_lo_m_s'], b['u_hi_m_s'], None if gone else b['P4_m']))
            s_b.append((b['u_lo_m_s'], b['u_hi_m_s'], None if gone else math.radians(b['A4_deg'])))
        for name, want, got in (('u_p', p_b, e['u_p']['hang_theo_u']), ('u_psi', s_b, e['u_psi']['hang_theo_u'])):
            if len(want) != len(got) or any(not (_close(lo, g['u_lo']) and _close(hi, g['u_hi'])
                                                 and _close(v, g['gia_tri'])) for (lo, hi, v), g in zip(want, got)):
                raise GateFileError(f'{w}.{name}.hang_theo_u differs from muc.{lv}.hang_theo_u of the C8b lock '
                                    '(P4_m, A4_deg by speed band; GC6-5)')
    return k, luc


def _git(top: str, *args) -> bytes:
    import subprocess
    try:
        return subprocess.run(['git', '-C', top, *args], capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateFileError(f'git {" ".join(args)} in {top!r}: {exc}')


def git_commit_of(goc: str, ref: dict, commit: str, what: str) -> dt.datetime:
    """The commit time of `commit`, after checking that the file of `ref` (under goc) is in that commit with the same
    bytes (sha256 of `git show commit:path`; KCJ1-14: a lock committed before the data). GateFileError otherwise."""
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise GateFileError(f'{what}: git_commit {commit!r}: a full commit hash')
    p = os.path.realpath(os.path.join(goc, ref['duong_dan']))
    top = _git(os.path.dirname(p), 'rev-parse', '--show-toplevel').decode().strip()
    rel = os.path.relpath(p, os.path.realpath(top)).replace(os.sep, '/')
    blob = _git(top, 'show', f'{commit}:{rel}')
    if hashlib.sha256(blob).hexdigest() != ref['sha256']:
        raise GateFileError(f'{what}: {rel} at commit {commit[:12]} is not the file of sha256 {ref["sha256"][:12]}... '
                            '(the lock must be committed unchanged)')
    return _iso_time(_git(top, 'show', '-s', '--format=%cI', commit).decode().strip(), f'{what}: commit time')


def _check_session(s: dict, w: str) -> None:
    """GC6-4 (rounds 3, 4): a camera session file as c7_kiem_cheo reads it (load_session): kind c7_session, video and
    its sha256, width and height, the locked model (chosen = model_khoa), marks {id: [coordinates]}, fit_ids whole ids,
    each once, each a mark with four corners in ref_corners_px."""
    mk, fi, rc = s.get('marks'), s.get('fit_ids'), s.get('ref_corners_px')
    ok = s.get('kind') == SESSION_KIND and _hex64(s.get('video_sha256')) and isinstance(s.get('video'), str) \
        and isinstance(mk, dict) and bool(mk) and isinstance(fi, list) and bool(fi) and isinstance(rc, dict) \
        and all(type(s.get(k)) is int and s[k] > 0 for k in ('width', 'height')) \
        and isinstance(s.get('chosen'), str) and bool(s['chosen']) and s.get('model_khoa') == s['chosen']
    if ok:
        ok = all(re.fullmatch(r'-?\d+', str(i)) and isinstance(v, list) and len(v) in (2, 3)
                 and all(_isnum(c) and math.isfinite(c) for c in v) for i, v in mk.items())
    if ok:
        ok = all(type(i) is int for i in fi) and len(set(fi)) == len(fi) and all(
            str(i) in mk and isinstance(rc.get(str(i)), list) and len(rc[str(i)]) == 4
            and all(isinstance(p, list) and len(p) == 2 and all(_isnum(c) and math.isfinite(c) for c in p)
                    for p in rc[str(i)]) for i in fi)
    if not ok:
        raise GateFileError(f'{w}: not a camera session file as c7_kiem_cheo reads it (kind {SESSION_KIND}, video, '
                            'video_sha256, marks {id: coordinates}, fit_ids whole ids once each, every one a mark with '
                            'four corners in ref_corners_px; GC6-4)')


def _manifest_lich(goc: str, mf: dict, w: str) -> Tuple[dict, dict]:
    """CODE 22/11 (C8 contract section 3, after CAM): the mark set tap_moc is the result of lich for one camera segment
    (session 2 has three), so a U manifest names its tap_moc (sha256 of tap_moc.json) and the lich it came from (lich
    {duong_dan, sha256}): a converged c7_lich_v3 file whose tap_moc_file_sha256 is that tap_moc. Returns (the lich, its
    rule {version, luat}); the rule is what is the same for every segment and session."""
    ng = mf.get('nguon')
    if not isinstance(ng, dict) or set(ng) != set(MANIFEST_NGUON) or not _hex64(ng.get('tap_moc')) \
            or not isinstance(ng.get('lich'), dict):
        raise GateFileError(f'{w}: nguon {{{", ".join(MANIFEST_NGUON)}}}: tap_moc the sha256 of tap_moc.json, lich '
                            '{duong_dan, sha256} of the lich it came from (CODE 22/11)')
    lref = _ref(dict(duong_dan=_rel_in(goc, ng['lich'].get('duong_dan')), sha256=ng['lich'].get('sha256')),
                f'{w}.nguon.lich')
    li = _open_ref(goc, lref, f'{w}.nguon.lich')
    if li.get('version') != LICH_VERSION or li.get('hoi_tu') is not True \
            or li.get('tap_moc_file_sha256') != ng['tap_moc'] or not isinstance(li.get('luat'), dict) \
            or not li['luat'] or not _hex64(li.get('session_sha256')):
        raise GateFileError(f'{w}.nguon.lich: a converged {LICH_VERSION} file (hoi_tu true) whose tap_moc_file_sha256 '
                            'is nguon.tap_moc, with its rule luat and session_sha256 (CODE 22/11)')
    return li, dict(version=li['version'], luat=li['luat'])


DK_KHOA = ('khoa_kiem_truoc_giu_ngoai', 'luat_giu_mask_theo_phan_do')   # D1, D4: the C8b lock itself (C8B 19/11)
DK_FILE = {'kiem_yaw_theo_hieu_mien_quyet': 'c8b_a6_o_v1', 'kiem_vi_tri_rieng': 'c8b_kiem_vi_tri_v1'}   # D2, D3


def _check_a6(d: dict, goc: str, khoa: dict) -> str:
    """CODE 23/11 (A6 changed sha 2aa8c970 -> d6bef847 with the same numbers; nothing tied the A6 of F4, F8 and the
    C8b lock together, only c7_chuoi_khung.q_luot per run): the C8b lock names one A6 (khoa_truoc.nguon, vai a6) and
    every JSON file of a C8 session folder that states an A6 (a6_sha256, as F4 xoay_v1, F5, the scorings; a6.sha256,
    as F8 lan_lap_v1) states that one. A superseded file is kept in a folder whose name starts with '_' (_archive:
    the folder rule of the project), skipped here. Returns the A6 sha256."""
    kt = khoa.get('khoa_truoc') if isinstance(khoa.get('khoa_truoc'), dict) else {}
    a6s = [n.get('sha256') for n in kt.get('nguon') or [] if isinstance(n, dict) and n.get('vai') == 'a6']
    if len(a6s) != 1 or not _hex64(a6s[0]):
        raise GateFileError(f'khoa_c8b: khoa_truoc.nguon names {len(a6s)} A6 file(s) (vai a6); one needed, the A6 every '
                            'F file of C8 must have used (CODE 23/11)')
    want, bad = a6s[0], []
    base = os.path.realpath(goc)
    for f in sorted({_folder_of(t) for t in d['thu_tu_clip']}):
        for root, dirs, names in os.walk(os.path.join(base, f)):
            dirs[:] = sorted(x for x in dirs if not x.startswith('_'))
            for n in sorted(names):
                q = os.path.join(root, n)
                if not n.lower().endswith('.json') or os.path.getsize(q) > INDEX_MAX_BYTES:
                    continue
                try:
                    with open(q, 'rb') as fh:
                        x = json.loads(fh.read().decode('utf-8'))
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
                if not isinstance(x, dict):
                    continue
                a6 = x.get('a6')
                for field, v in (('a6_sha256', x.get('a6_sha256')),
                                 ('a6.sha256', a6.get('sha256') if isinstance(a6, dict) else None)):
                    if v is not None and v != want:
                        bad.append(f'{os.path.relpath(q, base).replace(os.sep, "/")} {field} {str(v)[:12]}...')
    if bad:
        raise GateFileError(f'the C8b lock used the A6 {want[:12]}...; {len(bad)} file(s) of the C8 sessions name another '
                            f'(a step made with another A6; CODE 23/11): {"; ".join(bad[:8])}'
                            + (' ...' if len(bad) > 8 else '') + ' (a superseded file goes to an _archive folder)')
    return want


def _check_commit_hil(d: dict, goc: str, khoa: dict, scorers: set) -> None:
    """CODE 23/11 (chot T12 khoa C8B section 3; O1 v11.34 Q4, Liam 17/11 "Code qua Codex, SHA ghi truoc": the script
    of the time match and the grid scoring, c8b_gio and c8b_luoi): the tick part is dropped (nhanh_tick bo_tick) only
    with the commit written in the C8b lock before the scoring: commit_hil (40 hex) at the top of the lock (and the
    same in khoa_truoc when it states one). Every luoi.json of C8 (cells and K runs) has scorer.commit (c7_score) and
    script {commit, dirty false} (c8b_luoi) = commit_hil, and so has the final time file of every scoring (c8b_gio,
    the one luoi.json names by gio_sha256; the preliminary T5 file is not asked). giu_tick: nothing to check."""
    if khoa.get('nhanh_tick') != 'bo_tick':
        return
    ch = khoa.get('commit_hil')
    if not isinstance(ch, str) or not re.fullmatch(r'[0-9a-f]{40}', ch):
        raise GateFileError(f'khoa_c8b: nhanh_tick bo_tick with commit_hil {ch!r}; the tick part is dropped only with the '
                            'scoring commit (40 hex) written in the lock before the scoring (O1 Q4; CODE 23/11): C8 is '
                            'not locked')
    kt = khoa.get('khoa_truoc') if isinstance(khoa.get('khoa_truoc'), dict) else {}
    if 'commit_hil' in kt and kt['commit_hil'] != ch:
        raise GateFileError(f'khoa_c8b: khoa_truoc.commit_hil {kt["commit_hil"]!r} differs from commit_hil {ch!r}')
    bad = sorted(str(c) for c in scorers if c != ch)
    if bad:
        raise GateFileError(f'luoi.json scorer commit {bad} is not commit_hil {ch[:12]}... of the C8b lock (bo_tick: '
                            'every grid scored by the commit written before the scoring; O1 Q4, CODE 23/11)')
    grids, gios, want_gio = [], [], set()
    for c in d['o_c8']:
        for x in c['chuoi']:
            grids += [(f'cell {c["o"]} run {x["run_id"]}: {k}', x[k]) for k in ('file_cham', 'file_cham_h1') if x[k]]
            gios += [(f'cell {c["o"]} run {x["run_id"]}: {k}', x[k]) for k in ('file_gio', 'file_gio_h1') if x[k]]
    for x in d['luot_k']:
        for k in ('file_cham', 'file_cham_h1'):
            if x[k]:
                grids.append((f'K run {x["run_id"]}: {k}', x[k]))
                g = _open_ref(goc, x[k], f'K run {x["run_id"]}: {k}').get('gio_sha256')
                if g is not None:
                    want_gio.add(g)
    found = {}
    if want_gio:                                         # the time files of the K runs, found by sha256 (c8b_gio_v1)
        base = os.path.realpath(goc)
        for f in sorted({_folder_of(t) for t in d['thu_tu_clip']}):
            for root, dirs, names in os.walk(os.path.join(base, f)):
                dirs[:] = sorted(dirs)
                for n in sorted(names):
                    q = os.path.join(root, n)
                    if not n.lower().endswith('.json') or os.path.getsize(q) > INDEX_MAX_BYTES:
                        continue
                    h = sha256_file(q)
                    if h in want_gio and h not in found:
                        found[h] = dict(duong_dan=os.path.relpath(q, base).replace(os.sep, '/'), sha256=h)
        miss = sorted(want_gio - set(found))
        if miss:
            raise GateFileError(f'the time file(s) {[m[:12] for m in miss]} the K runs\' luoi.json name (gio_sha256) are in '
                                'no C8 session folder: their commit cannot be checked (bo_tick; CODE 23/11)')
        gios += [(f'K run time file {h[:12]}...', r) for h, r in sorted(found.items())]
    for tag, files in (('luoi.json script (c8b_luoi)', grids), ('time file script (c8b_gio)', gios)):
        for w, ref in files:
            sc = _open_ref(goc, ref, w).get('script')
            if not isinstance(sc, dict) or sc.get('commit') != ch or sc.get('dirty') is not False:
                raise GateFileError(f'{w}: {tag} {sc!r} is not commit_hil {ch[:12]}... with dirty false (bo_tick: the '
                                    'time match and the grid scoring of the commit written before the scoring; O1 Q4, '
                                    'CODE 23/11)')


def _check_bang_chung(d: dict, goc: str, scorers: set) -> None:
    """The four evidence files of the U standard (CODE_C8B design v7 section 0, D1-D4; signature agreed with C8B
    19/11): D1 (locked before the held-out part) and D4 (rule 2b masks only G2, G3, G5 at 8) are the C8b lock itself
    (trang_thai khoa), D4 with one clean scorer commit for every scoring (luoi.json scorer); D2 is c8b_a6_o_v1 (yaw by
    difference on the decision domain, section 7c), D3 c8b_kiem_vi_tri_v1 (position, section 8, made after the lock it
    names); each says dat."""
    cu = d['chuan_u']['dieu_kien']
    for k in DK_KHOA:
        if cu[k]['bang_chung']['sha256'] != d['khoa_c8b']['sha256']:
            raise GateFileError(f'chuan_u.dieu_kien.{k}.bang_chung: the C8b lock (khoa_c8b; C8B 19/11)')
    if len(scorers) != 1 or not re.fullmatch(r'[0-9a-f]{40}', next(iter(scorers))):
        raise GateFileError(f'chuan_u.dieu_kien.luat_giu_mask_theo_phan_do: the scorings were made by {sorted(scorers)!r};'
                            ' one clean commit of the scoring code for every scored run needed (luoi.json scorer, D4)')
    for k, ver in DK_FILE.items():
        b = _open_ref(goc, cu[k]['bang_chung'], f'chuan_u.dieu_kien.{k}.bang_chung')
        if b.get('version') != ver or b.get('trang_thai') != 'dat':
            raise GateFileError(f'chuan_u.dieu_kien.{k}.bang_chung: a {ver} file with trang_thai dat, got '
                                f'{b.get("version")!r} / {b.get("trang_thai")!r} (C8B 19/11)')
        if k == 'kiem_vi_tri_rieng' and b.get('khoa_truoc_sha256') != d['khoa_c8b']['sha256']:
            raise GateFileError('chuan_u.dieu_kien.kiem_vi_tri_rieng: khoa_truoc_sha256 is not the C8b lock of the gate')


def _check_premise(d: dict, goc: str, hs: dict, recs: dict) -> None:
    """The premise of the C8 lock, one file per camera segment (design section 9; CODE 20/11 after C8 session 2): hs the
    dossier ho_so_c8, recs {session id: (khoa_s7 record of the session's kiem_cheo.json in the dossier, its sha256)}
    (CODE 24/11: the S7 lock per session). Apart from check_nguon to be run on a session's premise files before the
    other sources exist."""
    # premise of the C8 lock (design section 9; GC6-4: by role, of this session, readable session files), one premise
    # file per camera segment that holds C8 runs (CODE 20/11 after C8 session 2; KIEMCHEO: roles dau1, dau2, cuoi)
    models, tmods, kc_segs = set(), set(), set()
    for seg in _check_tien_de(d):
        w = f'tien_de_c8[{seg["k"]}] segment {seg["ten"]!r}'
        clips = seg['clips']
        for role, x in zip(TD_ROLES, clips):
            _check_session(_open_ref(goc, x, f'{w} clip {role}'), f'{w} clip {role}')
        td = _open_ref(goc, seg['file'], f'{w}.file')
        if td.get('format') != TIEN_DE_FORMAT or td.get('ket_qua') != 'dat':
            raise GateFileError(f'{w}.file: a {TIEN_DE_FORMAT} file with ket_qua dat, got format {td.get("format")!r}, '
                                f'ket_qua {td.get("ket_qua")!r}: C8 is not locked, CAN LIAM QUYET (J v6 section 5 J3-5, '
                                'section 0b)')
        dn = td.get('doan') if isinstance(td.get('doan'), dict) else {}
        if td.get('session_id') != seg['buoi'] or dn.get('ten') != seg['ten']:
            raise GateFileError(f'{w}.file: the premise of session {td.get("session_id")!r}, segment {dn.get("ten")!r}; '
                                f'the gate has it in session {seg["buoi"]!r}, segment {seg["ten"]!r} (GC6-4)')
        # the kiem_cheo.json of the segment: the one its premise names, of that segment, locked by the S7 manifest and
        # commit of the dossier (KCJ1-14; CODE 20/11 after C8 session 2: one kiem_cheo result per segment)
        kcg = _open_ref(goc, seg['kiem_cheo'], f'{w}.kiem_cheo')
        r7 = kcg.get('khoa_s7') if isinstance(kcg.get('khoa_s7'), dict) else {}
        rec = (recs.get(seg['buoi']) or ({}, None))[0]
        if kcg.get('version') != KIEM_CHEO_VERSION or (kcg.get('doan') or {}).get('ten') != seg['ten'] \
                or any(r7.get(q) != rec.get(q) for q in ('sha256', 'git_commit', 'git_time')):
            raise GateFileError(f'{w}.kiem_cheo: a {KIEM_CHEO_VERSION} file of segment {seg["ten"]!r} (doan.ten) '
                                'whose khoa_s7 {sha256, git_commit, git_time} is that of its session in the dossier '
                                '(KCJ1-14; CODE 24/11)')
        kc_segs.add((seg['buoi'], seg['kiem_cheo']['sha256']))
        vt = td.get('vai_tro_clip_h')
        ses = td.get('sha256', {}).get('sessions') if isinstance(td.get('sha256'), dict) else None
        if not isinstance(vt, dict) or sorted(vt.values()) != sorted(TD_ROLES) or not isinstance(ses, dict) \
                or set(ses) != set(vt):
            raise GateFileError(f'{w}.file: vai_tro_clip_h gives the roles {TD_ROLES} once each, and sha256.sessions '
                                'one session per clip of it (GC6-4)')
        by_role = {vt[n]: ses[n] for n in vt}
        models |= {_open_ref(goc, x, f'{w} clip {role}').get('chosen') for role, x in zip(TD_ROLES, clips)}
        if 'model' in td:
            tmods.add(td['model'])
        kcs = (td.get('sha256') or {}).get('kiem_cheo') if isinstance(td.get('sha256'), dict) else None
        if kcs != seg['kiem_cheo']['sha256']:
            raise GateFileError(f'{w}.file: sha256.kiem_cheo is not the kiem_cheo.json of the segment (GC10-5)')
        vid = td.get('sha256', {}).get('clip_h')
        if isinstance(vid, dict):                        # GC6-4 (round 2): the videos too, when the premise names them
            for n in vt:
                sx = next(x for r_, x in zip(TD_ROLES, clips) if r_ == vt[n])
                if vid.get(n) != _open_ref(goc, sx, f'{w} clip {vt[n]}').get('video_sha256'):
                    raise GateFileError(f'{w}.file: the video of {n} ({vt[n]}) is not that of its session (GC6-4)')
        for role, x in zip(TD_ROLES, clips):
            if by_role[role] != x['sha256']:
                raise GateFileError(f'{w}: the clip of role {role} is not the session the premise gives for it (GC6-4)')
    for b_, (_, kc_sha) in recs.items():
        if (b_, kc_sha) not in kc_segs:
            raise GateFileError(f'chuan_u.ho_so_c8: khoa_kiem_luc of session {b_!r}: its kiem_cheo is the kiem_cheo.json '
                                'of no segment of that session in tien_de_c8 (GC10-5)')
    if len(models) != 1 or not tmods <= models:
        raise GateFileError(f'tien_de_c8: the clip H sessions of every segment use the models '
                            f'{sorted(map(str, models))} (premise models {sorted(map(str, tmods))}): one locked model '
                            '(KC1-1; GC6-4)')


def check_nguon(d: dict, goc: str) -> dict:
    """The sources of a checked gate content under the C8 folder goc (design sections 3, 6-10). Returns d."""
    check(d)
    if is_fake(d) or has_fake_sign(d):
        raise GateFileError('a FAKE content has no sources: never registered (GC6-14)')
    if not os.path.isdir(goc):
        raise GateFileError(f'C8 folder {goc!r}: no such folder')
    kh = _open_ref(goc, d['ke_hoach_c8'], 'ke_hoach_c8')
    if kh.get('format') != KE_HOACH_FORMAT or not isinstance(kh.get('o'), list):
        raise GateFileError(f'ke_hoach_c8: format {KE_HOACH_FORMAT} with the list o of cells')
    plan = []
    for i, x in enumerate(kh['o']):
        _obj(x, ('o', 'muc', 'controller'), f'ke_hoach_c8.o[{i}]')
        plan.append((x['o'], x['muc'], x['controller']))
    have = [(c['o'], c['muc'], c['controller']) for c in d['o_c8']]
    if sorted(plan) != sorted(have) or len(set(plan)) != len(plan):
        raise GateFileError(f'o_c8 cells {sorted(have)}: the plan locked before C8 has {sorted(plan)} (each once)')
    khoa, t_c8b = _check_khoa_c8b(d, goc)
    _check_a6(d, goc, khoa)
    eligible, cham_luc, scorers, elig_seg = _check_runs_nguon(d, goc)
    # GC6-1 (round 3): the C8b lock (its khoa_truoc) before the first scoring of any C8 run
    if cham_luc and not t_c8b < min(cham_luc.values()):
        raise GateFileError(f'khoa_c8b: khoa_truoc.luc {t_c8b.isoformat()} is not before the first scoring of a C8 run '
                            f'({min(cham_luc.values()).isoformat()}, luoi.json script.luc_chay; GC6-1)')
    # U standard (design section 8): the dossier ho_so_c8 (ho_so_c8_v1, written by ghep)
    cu = d['chuan_u']
    hs = _open_ref(goc, cu['ho_so_c8'], 'chuan_u.ho_so_c8')
    if hs.get('format') != HO_SO_FORMAT:
        raise GateFileError(f'chuan_u.ho_so_c8: format {HO_SO_FORMAT}')
    bc = hs.get('bang_chung')
    want = {k: cu['dieu_kien'][k]['bang_chung']['sha256'] for k in CHUAN_U_DK}
    if bc != want:
        raise GateFileError(f'chuan_u.ho_so_c8: bang_chung {bc!r} must list exactly the four evidence SHAs {want!r}')
    if hs.get('ke_hoach_c8_sha256') != d['ke_hoach_c8']['sha256'] or hs.get('khoa_c8b_sha256') != d['khoa_c8b']['sha256']:
        raise GateFileError('chuan_u.ho_so_c8: ke_hoach_c8_sha256 / khoa_c8b_sha256 are not the plan / C8b lock of the gate')
    _check_commit_hil(d, goc, khoa, scorers)
    _check_bang_chung(d, goc, scorers)
    cl = hs.get('cham_luc')                       # F17, GC6-1: the time of the first scoring of every C8 run
    if not isinstance(cl, dict) or set(cl) != set(cham_luc) \
            or any(_iso_time(cl[g], f'chuan_u.ho_so_c8.cham_luc.{g}') != cham_luc[g] for g in cl):
        raise GateFileError('chuan_u.ho_so_c8: cham_luc {run id: ISO time of its first scoring} of every C8 run, the '
                            'earliest script.luc_chay of its scoring files (GC6-1)')
    gn = hs.get('giu_ngoai')
    if not isinstance(gn, list) or not gn or len(set(gn)) != len(gn) or set(gn) != set(cham_luc):
        raise GateFileError('chuan_u.ho_so_c8: giu_ngoai, the held-out runs, every C8 run (CODE 20/11: the default of '
                            'GATE, approved; never a list typed by hand; GC9-2)')
    # J v6 section 6 condition (1), KCJ1-14 (CODE 19/11): the S7 check of KIEMCHEO locked by a manifest committed in
    # git before the held-out runs were scored; the commit is read back from git, not taken from the files
    # KG-6 (CODE 24/11): the scorings dropped (made at another commit than commit_hil) recomputed from the folder; the
    # dossier lists them, with Liam's record when there is one
    kl = _open_ref(goc, d['khoa_c8b'], 'khoa_c8b')
    _check_mien_theo_buoi(d, kl, elig_seg)              # KG-5
    bo_want = scorings_dropped(goc, kl.get('commit_hil') if kl.get('nhanh_tick') == 'bo_tick' else None)
    cb = _obj(hs.get('cham_cu_bo'), ('luoi', 'liam'), 'chuan_u.ho_so_c8.cham_cu_bo')
    if cb['luoi'] != bo_want:
        raise GateFileError(f'chuan_u.ho_so_c8.cham_cu_bo.luoi: {len(cb["luoi"]) if isinstance(cb["luoi"], list) else cb["luoi"]!r} '
                            f'files; the C8 folder has {len(bo_want)} c8b_luoi files made at another commit than '
                            'commit_hil, every one listed (KG-6)')
    if bo_want:
        _liam(cb['liam'], 'chuan_u.ho_so_c8.cham_cu_bo.liam')
    elif cb['liam'] is not None:
        raise GateFileError('chuan_u.ho_so_c8.cham_cu_bo.liam: no dropped scoring, no record')
    # CODE 24/11 (FABLE_C8 V2, GATE (d)): one S7 lock per C8 session, in session order; the S7 numbers the same in
    # every session; each lock committed before the first scoring of a held-out run of its session
    kkl = hs.get('khoa_kiem_luc')
    buois = [t['buoi'] for t in d['tien_de_c8']]
    if not isinstance(kkl, list) or len(kkl) != len(buois):
        raise GateFileError(f'chuan_u.ho_so_c8.khoa_kiem_luc: a list, one entry per C8 session ({len(buois)}; CODE 24/11)')
    rows_b, _ = _buoi_rows(goc, d['thu_tu_clip'])
    b_of_clip = {r['clip']: r['_buoi'] for r in rows_b}
    b_of_run = {e['run_id']: b_of_clip.get(e['clip']) for c in d['o_c8'] for e in c['chuoi']}
    b_of_run.update({x['run_id']: b_of_clip.get(x['clip']) for x in d.get('luot_k') or []})
    recs, s7_same = {}, None
    for k, kk in enumerate(kkl):
        w = f'khoa_kiem_luc[{k}]'
        kk = _obj(kk, KK_KEYS, f'chuan_u.ho_so_c8.{w}')
        if kk['buoi'] != buois[k]:
            raise GateFileError(f'{w}.buoi {kk["buoi"]!r}: session {k} is {buois[k]!r} (tien_de_c8 order)')
        kc = _open_ref(goc, _ref(kk['kiem_cheo'], f'{w}.kiem_cheo'), f'{w}.kiem_cheo')
        ks = _ref(kk['khoa_s7'], f'{w}.khoa_s7')
        _ref_bytes(goc, ks, f'{w}.khoa_s7')
        rec = kc.get('khoa_s7')
        if kc.get('version') != KIEM_CHEO_VERSION or not isinstance(rec, dict) or rec.get('sha256') != ks['sha256'] \
                or rec.get('git_commit') != kk['git_commit'] or rec.get('git_time') != kk['git_time']:
            raise GateFileError(f'{w}: kiem_cheo.json ({KIEM_CHEO_VERSION}) khoa_s7 {{sha256, git_commit, '
                                'git_time}} is not the manifest and commit of the dossier (KCJ1-14)')
        same = {q: _open_ref(goc, ks, f'{w}.khoa_s7').get(q) for q in S7_SAME}
        if s7_same is None:
            s7_same = same
        elif same != s7_same:
            raise GateFileError(f'{w}.khoa_s7: {list(S7_SAME)} differ from those of session 0; one S7 check for every '
                                'session (CODE 24/11)')
        t_lock = git_commit_of(goc, ks, kk['git_commit'], w)
        if t_lock != _iso_time(kk['git_time'], f'{w}.git_time'):
            raise GateFileError(f'{w}: git_time is not the time of its commit')
        mine = [cham_luc[g] for g in gn if b_of_run.get(g) == k]
        if mine and not t_lock < min(mine):
            raise GateFileError(f'chuan_u: the check of session {k} ({buois[k]}) was locked at {t_lock.isoformat()}, '
                                f'not before the first scoring of a held-out run of that session '
                                f'({min(mine).isoformat()}; J v6 section 6 condition (1); CODE 24/11)')
        recs[buois[k]] = (rec, kk['kiem_cheo']['sha256'])
    # shutter table sources
    for k, x in d['bo_cham']['bang_man_trap'].items():
        if x['nhanh'] == 'alpha':
            _open_ref(goc, x['nguon_3b'], f'bo_cham.bang_man_trap.{k}.nguon_3b', as_json=False)
    _check_premise(d, goc, hs, recs)
    # shifted anchor (design section 10; GC6-6: one manifest for every scoring of every eligible run, the domain it
    # covered = the frames that scoring relies on, before the maximum is taken)
    md = d['camera']['c8b']['moc_dich']
    tops = {k: 0.0 for k in MOC_KEYS}
    seen, tap_of = {}, {}
    for i, ref in enumerate(md['nguon']['manifests']):
        w = f'camera.c8b.moc_dich.nguon.manifests[{i}]'
        mf = _open_ref(goc, ref, w)
        if mf.get('format') != MANIFEST_FORMAT or mf.get('ok') is not True:
            raise GateFileError(f'{w}: a {MANIFEST_FORMAT} file with ok true')
        fs = mf.get('file_cham_sha256')
        if fs not in eligible:
            raise GateFileError(f'{w}: file_cham_sha256 is not a scoring of an eligible C8 run (the last run of a '
                                'co_luot cell)')
        if fs in seen:
            raise GateFileError(f'{w}: a second manifest for the scoring of {seen[fs]}')
        seen[fs] = w
        if mf.get('mien_da_tinh') != eligible[fs]:
            raise GateFileError(f'{w}: mien_da_tinh is not the domain the scoring relies on (mien_phai_phu of every '
                                'scored node: node, D_s, frame ids incl. the reference window W and the t_done frames)')
        for k in MOC_KEYS:
            tops[k] = max(tops[k], _num(mf.get(k), f'{w}.{k}'))
        # round 3 (ghep): the sampler and car_pose the gate copies are those of every manifest (CAM); CODE 22/11: the
        # mark set is the result of lich per camera segment: its rule is that of the gate, its sha one per segment and
        # made with a session of the run's segment
        li, rule = _manifest_lich(goc, mf, w)
        if {k: mf['nguon'][k] for k in MOC_NGUON} != {k: md['nguon'][k] for k in MOC_NGUON}:
            raise GateFileError(f'{w}: nguon bo_lay_mau, car_pose are not those of the gate (the same for every scoring '
                                'of every session; CODE 22/11)')
        if rule != md['nguon']['luat_lich']:
            raise GateFileError(f'{w}: its lich has the rule {rule!r}, the gate {md["nguon"]["luat_lich"]!r} (one rule '
                                'for every camera segment and session; CODE 22/11)')
        seg = elig_seg[fs]
        if li['session_sha256'] not in [x['sha256'] for x in seg['clips'][:2]]:
            raise GateFileError(f'{w}: its lich was made with session {li["session_sha256"][:12]}..., no clip H at the '
                                f'start of the camera segment {seg["ten"]!r} of its run (tap_moc per segment; CODE 22/11)')
        key = (seg['k'], seg['ten'])
        if tap_of.setdefault(key, mf['nguon']['tap_moc']) != mf['nguon']['tap_moc']:
            raise GateFileError(f'{w}: another tap_moc than the other scorings of the camera segment {seg["ten"]!r} '
                                '(one lich result per segment; CODE 22/11)')
    miss = sorted(set(eligible) - set(seen))
    if miss:
        raise GateFileError(f'camera.c8b.moc_dich.nguon.manifests: no U manifest for {len(miss)} scoring(s) of eligible '
                            'runs; the lock is the maximum over every scoring (GC6-6)')
    for k in MOC_KEYS:
        if md[k] < tops[k]:
            raise GateFileError(f'camera.c8b.moc_dich.{k} {md[k]!r}: below the largest {tops[k]!r} of the U manifests')
    return d


# ── reader ─────────────────────────────────────────────────────────────────────────────────────────────────────

def check_so_do(so_do: Iterable[str]) -> Tuple[str, ...]:
    """so_do as distinct conditions of MASKABLE, in MASKABLE order."""
    sd = list(so_do)
    if len(set(sd)) != len(sd) or set(sd) - set(MASKABLE):
        raise GateFileError(f'so_do {sd!r}: distinct conditions of {MASKABLE}')
    return tuple(c for c in MASKABLE if c in set(sd))


def mask_applies(platform: str, stage: str, campaign: str) -> bool:
    """Whether the so_do list of the level applies to a run (module docstring)."""
    if platform not in PLATFORMS:
        raise GateFileError(f'platform {platform!r}: one of {PLATFORMS}')
    if platform == 'HW':
        return True
    return platform == 'HIL' and stage == MASK_HIL_STAGE and campaign in B_CAMPAIGNS


class GateC8:
    """A checked gate file (read_gate_c8)."""

    def __init__(self, d: dict, sha256: str, path: str = ''):
        self.data = check(d)
        self.sha256 = sha256
        self.path = path
        self.levels = list(d['levels'])
        self.u_t_branch = d['u_t_branch']
        self.fake = is_fake(d)                   # GC6-14: a FAKE content (tests, demo); check_nguon refuses it
        self.fake_sign = has_fake_sign(d)        # GC6-14 (round 3): any FAKE mark, even an incomplete set

    def so_do(self, level: str) -> Tuple[str, ...]:
        """The so_do conditions of a gate level (decided by q); () for a task that is not a gate level."""
        e = self.data['dieu_2b']['per_level'].get(level)
        return tuple(e['so_do']) if e is not None else ()

    def run_so_do(self, level: str, platform: str, stage: str, campaign: str) -> Tuple[str, ...]:
        sd = set(self.so_do(level) if mask_applies(platform, stage, campaign) else ())
        if platform == 'HW' and level == G5_F08:
            # GC6-7: J v6 section 6 (Liam 19/11, "không làm bù", in force at once): G5 of every floor run at F08 is in
            # the mask, also when F08 is no gate level (J section 4: a dropped level still runs its check); decided by
            # q since 23/11 (CODE, Liam "duyệt bảng luật mới")
            sd.add('g5')
        return tuple(c for c in MASKABLE if c in sd)


def _finite(x, what: str, lo: float, strict: bool = False) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or \
            (x <= lo if strict else x < lo):
        raise GateFileError(f'{what} {x!r}: a finite number {">" if strict else ">="} {lo}')
    return float(x)


def car_pose_of(cp, what: str) -> dict:
    """The car_pose configuration {tag_offset, side_m, h_m} of the campaign side (C8B 18/11, U files of a campaign
    run; unchanged until the campaign part): side_m > 0, h_m >= 0 finite; tag_offset null or three finite numbers,
    null read as [0, 0, 0]."""
    if not isinstance(cp, dict) or set(cp) != {'tag_offset', 'side_m', 'h_m'}:
        raise GateFileError(f'{what}: car_pose must be {{tag_offset, side_m, h_m}}, got {cp!r}')
    to = cp['tag_offset']
    if to is None:
        to = [0.0, 0.0, 0.0]
    elif not isinstance(to, list) or len(to) != 3:
        raise GateFileError(f'{what}: car_pose.tag_offset {to!r}: null or [dx_m, dy_m, dpsi_rad]')
    else:
        to = [_finite(v, f'{what}: car_pose.tag_offset', -math.inf) for v in to]
    return dict(tag_offset=to, side_m=_finite(cp['side_m'], f'{what}: car_pose.side_m', 0.0, strict=True),
                h_m=_finite(cp['h_m'], f'{what}: car_pose.h_m', 0.0))


def moc_dich_of(gate: 'GateC8') -> dict:
    """The C8 lock of the shifted-anchor bound (design section 10): the four locked numbers (before the margin),
    the source {bo_lay_mau, car_pose (5 keys), luat_lich, manifests} and khoa_sha256 (CODE 22/11: the lich rule, the
    mark set being per camera segment)."""
    c8b = gate.data['camera']['c8b']
    md = c8b['moc_dich']
    out = {k: float(md[k]) for k in MOC_KEYS}
    out.update(bo_lay_mau=md['nguon']['bo_lay_mau'], car_pose=car_pose_v2_of(md['nguon']['car_pose'], 'moc_dich'),
               luat_lich=md['nguon']['luat_lich'], khoa_sha256=c8b['khoa_sha256'])
    return out


def _no_duplicate_keys(pairs):
    """GC1-4: a JSON object with a key twice is refused (json keeps the last one silently)."""
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise GateFileError(f'duplicate JSON key(s) {dup}')
    return dict(pairs)


def read_gate_c8(path: str) -> GateC8:
    """Read and check a gate file (content only, check); GateFileError (a ValueError) on any problem."""
    if not os.path.isfile(path):
        raise GateFileError(f'gate file {path!r}: no such file')
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        d = json.loads(raw.decode('utf-8'), object_pairs_hook=_no_duplicate_keys)
    except GateFileError as exc:
        raise GateFileError(f'gate file {path!r}: {exc}')
    except (UnicodeDecodeError, ValueError) as exc:
        raise GateFileError(f'gate file {path!r}: not UTF-8 JSON ({exc})')
    try:
        return GateC8(d, hashlib.sha256(raw).hexdigest(), path)
    except GateFileError as exc:
        raise GateFileError(f'gate file {path!r}: {exc}')


def _refs(x):
    """Every {duong_dan, sha256} reference inside a content."""
    if isinstance(x, dict):
        if set(x) == {'duong_dan', 'sha256'}:
            yield x
        for v in x.values():
            yield from _refs(v)
    elif isinstance(x, list):
        for v in x:
            yield from _refs(v)


def is_fake(d: dict) -> bool:
    """A FAKE content (fake_gate: tests, demo): c8_dir FAKE, written_by.script FAKE, every source reference under
    FAKE/. Only such a content may be written without its sources (write_gate with goc None)."""
    wb = d.get('written_by')
    return (d.get('c8_dir') == 'FAKE' and isinstance(wb, dict) and wb.get('script') == 'FAKE'
            and all(str(r['duong_dan']).startswith('FAKE/') for r in _refs(d)))


def has_fake_sign(d: dict) -> bool:
    """GC6-14 (round 3): any one FAKE mark (c8_dir, written_by.script, bo_cham.phien_ban, tien_de_c8[].buoi, a source
    path under FAKE/, a car_pose version): a real gate carries none; the real reading paths refuse it unless the whole
    content is FAKE (is_fake) and FAKE is allowed (tests, demo)."""
    wb = d.get('written_by') if isinstance(d.get('written_by'), dict) else {}
    bc = d.get('bo_cham') if isinstance(d.get('bo_cham'), dict) else {}
    tds = d.get('tien_de_c8') if isinstance(d.get('tien_de_c8'), list) else []
    return ('FAKE' in (d.get('c8_dir'), wb.get('script'), bc.get('phien_ban'))
            or any(isinstance(t, dict) and t.get('buoi') == 'FAKE' for t in tds)
            or any(str(r.get('duong_dan', '')).startswith('FAKE/') for r in _refs(d))
            or '"FAKE"' in json.dumps(d.get('camera'), ensure_ascii=False))


def write_gate(path: str, d: dict, goc: Optional[str] = None) -> str:
    """Write a gate content and return the SHA-256 of the bytes (design section 3): check and check_nguon (sources
    under the C8 folder goc) on the bytes read back in memory as read_gate_c8 reads them, before the file is opened,
    so a file either check refuses is never written. goc None only for a FAKE content (is_fake), checked by check
    alone."""
    check(d)
    raw = (json.dumps(d, indent=1, ensure_ascii=False) + '\n').encode('utf-8')
    try:
        back = json.loads(raw.decode('utf-8'), object_pairs_hook=_no_duplicate_keys)
        if goc is None:
            if not is_fake(back):
                raise GateFileError('a gate that is not FAKE is written with its C8 folder (check_nguon)')
            check(back)
        else:
            check_nguon(back, goc)
    except GateFileError as exc:
        raise GateFileError(f'not written: {exc}')
    try:
        with open(path, 'xb') as f:                      # GC6-14: a gate file is written once, never replaced
            f.write(raw)
    except FileExistsError:
        raise GateFileError(f'not written: {path!r} exists (a gate file is never replaced)')
    return hashlib.sha256(raw).hexdigest()


# ── FAKE content (tests, rq3_log demo; numbers are not measured) ───────────────────────────────────────────────

FAKE_SHA = 'e' * 64


def fake_terms(u_p_m: float = 0.002, u_psi_deg: float = 2.0, a_m: float = 0.1445) -> dict:
    """FAKE closed terms of a level: constant U_p, U_psi, no arm, no u-dependent term."""
    return dict(a_m=a_m, u_p=dict(hang_m=u_p_m, he_so_r=0.0, u_phi_rad=0.0, hang_theo_u=[]),
                u_psi=dict(hang_rad=math.radians(u_psi_deg), hang_theo_u=[]))


def fake_gate(levels=('O1-F05',), u_p_m=0.002, u_psi_deg=2.0, u_t_s=0.010) -> dict:
    """A gate content that passes check (not check_nguon: the source references are FAKE) for tests and the rq3_log
    demo. u_t_s: one number, or a dict {level: U_t,C8 or (U_t PID, U_t LADRC)}."""
    ref = lambda name: dict(duong_dan=f'FAKE/{name}',                                 # noqa: E731
                            sha256=hashlib.sha256(name.encode()).hexdigest())    # GC6-1: one file per run
    ev = dict(phan_loai_du=True, ban_cham_cuoi=True,
              bo_so=[dict(e_do_lo_vuot_D_moi_nut=False, e_do_xac_dinh_moi_nut=True,
                          g4_can_huu_han_mien_hop_moi_nut=True)])
    cells, runs, per = [], {}, {}
    for name in levels:
        ut = u_t_s[name] if isinstance(u_t_s, dict) else u_t_s
        uts = tuple(ut) if isinstance(ut, (tuple, list)) else (ut, ut)
        runs[name] = []
        for ctrl, u in zip(CONTROLLERS, uts):
            rid = f'FAKE_{name}_{ctrl}'
            cells.append(dict(o=f'{name}_{ctrl}', muc=name, controller=ctrl, trang_thai='co_luot',
                              chuoi=[dict(run_id=rid, clip=f'IMG_{rid}', ket_qua='dat', loai_hong=[], kiem_gio_d1='dat',
                                          dem_truot_kiem_gio=False, file_cham=ref(f'{rid}_luoi.json'),
                                          file_gio=ref(f'{rid}_gio.json'), file_chung_cu=ref(f'{rid}_chung_cu.json'),
                                          file_so_h=ref('drift.json'), file_cham_h1=None, file_gio_h1=None,
                                          file_edo=ref(f'{rid}_edo.json'), file_edo_h1=None,
                                          file_log=ref(f'{rid}.csv'), file_thao_tac=ref(f'{rid}_score.json'),
                                          thieu_u_t=False)]))
            runs[name].append(dict(run_id=rid, controller=ctrl, o=f'{name}_{ctrl}', h_s=u / 2.0, u_max_s=u / 2.0,
                                   u_t_s=u / 2.0 + u / 2.0, chung_cu=json.loads(json.dumps(ev))))
        t = fake_terms(u_p_m, u_psi_deg)
        per[name] = level_block(name, t['a_m'], t['u_p'], t['u_psi'], runs[name])
    cu = dict(loai=CHUAN_U_LOAI, le=CHUAN_U_LE, ho_so_c8=ref('ho_so_c8.json'),
              dieu_kien={k: dict(trang_thai='dat', bang_chung=ref(f'{k}.json')) for k in CHUAN_U_DK})
    clip = lambda i: dict(duong_dan=f'FAKE/clip_H{i}.json', sha256=format(i, 'x') * 64)  # noqa: E731
    md = {k: 0.0 for k in MOC_KEYS}
    md.update(U_moc_dong_xy_m=0.005, U_moc_tham_xy_m=0.005, U_moc_dong_psi_rad=0.004, U_moc_tham_psi_rad=0.004,
              nguon=dict(bo_lay_mau='FAKE_blm_v1', luat_lich=dict(version='FAKE', luat=dict(FAKE=0)),
                         manifests=[ref('u_moc_manifest.json')],
                         car_pose=dict(tag_offset=None, side_m=0.1445, h_m=0.20, kappa=0.0, phien_ban='FAKE')))
    cam = {'c8b': dict(moc_dich=md, khoa_sha256='f' * 64, eps_H=dict(mm=1.0, huong_deg=0.2),
                       mien_car_pose=dict(side_m=0.1445, phien_ban='FAKE', h_m=[0.19, 0.21], kappa=[0.0, 0.0],
                                          trang_thai='dat', ly_do=[]))}
    lv = list(levels)
    return dict(format=FORMAT, c8_dir='FAKE', written_by=dict(script='FAKE', commit='', dirty=True),
                levels=lv, muc_kha_thi=lv, muc_bo_theo_liam={}, quyet_liam={}, u_t_branch='bo_tick',
                lich_doi_chieu=LICH,
                bo_cham=dict(phien_ban='FAKE', bang_man_trap={k: dict(nhanh='mot_khung') for k in lv + [TASK_K]}),
                ke_hoach_c8=ref('ke_hoach_c8.json'), thu_tu_clip=[ref('thu_tu_clip.csv')],
                khoa_c8b=ref('c8b_khoa_v5.json'), o_c8=cells, chuan_u=cu,
                tien_de_c8=[dict(buoi='FAKE', doan=[dict(ten='FAKE', dau=[clip(1), clip(2)], cuoi=clip(3),
                                                         file=ref('tien_de_c8.json'), kiem_cheo=ref('kiem_cheo.json'))])],
                runs_c8=runs, muc_loai_e_do={k: [] for k in lv},
                chung_cu_muc={k: dict(e_do={c: [f'FAKE_{k}_{c}'] for c in CONTROLLERS},
                                      g4={c: [f'FAKE_{k}_{c}'] for c in CONTROLLERS}) for k in lv},
                dieu_2b=dict(thresholds={c: o1_gate.THRESHOLDS[c] for c in MASKABLE}, per_level=per),
                camera=cam, luot_k=[])


# ── ke_hoach_c8_v1 (F17: GATE writes it; CODE 19/11) ───────────────────────────────────────────────────────────

KE_HOACH_NGUON = ('CLAUDE_THESIS 03_CODE/docs/2026-11-18_Huong_dan_tung_buoc_C8_Liam.md, phan F (PID: C8-P-F02-1, '
                  'C8-P-F05-1, C8-P-F08-1, C8-P-F02-2, C8-P-F05-2, C8-P-F08-2) and phan G (LADRC: C8-L-...), the run '
                  'list locked before the session (buoi 2, ban 6)')
KE_HOACH_CTRL = (('P', 'PID'), ('L', 'LADRC'))
KE_HOACH_REPS = (1, 2)


def ke_hoach_c8(levels=('O1-F02', 'O1-F05', 'O1-F08')) -> dict:
    """The plan of the C8 cells (format ke_hoach_c8_v1): one cell per locked figure-8 run of the session guide (run
    id C8-<P|L>-<F02|F05|F08>-<1|2>, a rerun keeps its cell); the K1, K2 runs are not cells of rule 2b (G1-G5 of the
    K tasks always decide)."""
    o = []
    for ch, ctrl in KE_HOACH_CTRL:
        for rep in KE_HOACH_REPS:
            for lv in levels:
                if lv not in FIG8:
                    raise GateFileError(f'{lv}: not a figure-8 task')
                o.append(dict(o=f'C8-{ch}-{lv.split("-")[1]}-{rep}', muc=lv, controller=ctrl))
    return dict(format=KE_HOACH_FORMAT, nguon=KE_HOACH_NGUON, o=o)


# ── ghep: the gate content assembled from its sources (F10-F19; CODE 19/11, Liam at CODE "gộp rồi chạy ngay thên TK4")

INDEX_SKIP = ('.mov', '.mp4', '.m4v', '.avi')            # clips are not sources the gate hashes
INDEX_MAX_BYTES = 64 * 2 ** 20


class _Kho:
    """The files under the C8 folder by sha256 (and their JSON, read once): the sources a gate refers to are found by
    the SHA another source states, never by a name typed again."""

    def __init__(self, goc: str):
        self.goc = os.path.realpath(goc)
        self.by_sha: Dict[str, List[str]] = {}
        self._json: Dict[str, object] = {}
        for root, dirs, names in os.walk(self.goc):
            dirs.sort()
            for n in sorted(names):
                p = os.path.join(root, n)
                if n.lower().endswith(INDEX_SKIP) or not os.path.isfile(p) or os.path.getsize(p) > INDEX_MAX_BYTES:
                    continue
                rel = os.path.relpath(p, self.goc).replace(os.sep, '/')
                self.by_sha.setdefault(sha256_file(p), []).append(rel)

    def ref(self, rel: str) -> dict:
        p = os.path.realpath(os.path.join(self.goc, rel))
        if not os.path.isfile(p) or os.path.commonpath([self.goc, p]) != self.goc:
            raise GateFileError(f'ghep: {rel!r} is no file under the C8 folder {self.goc!r}')
        return dict(duong_dan=os.path.relpath(p, self.goc).replace(os.sep, '/'), sha256=sha256_file(p))

    def by(self, sha, what: str, name: Optional[str] = None, under: Optional[str] = None) -> dict:
        """The reference of the file of sha256 sha (the one named `name` when several files have those bytes; one in
        the folder `under`, CODE 22/11: a session's own copy)."""
        rels = [r for r in self.by_sha.get(sha, []) if (name is None or os.path.basename(r) == name)
                and (under is None or _under(r, under))]
        if not _hex64(sha) or not rels:
            raise GateFileError(f'ghep: {what}: no file with sha256 {str(sha)[:12]}... under the C8 folder'
                                + (f' named {name!r}' if name else ''))
        return dict(duong_dan=rels[0], sha256=sha)

    def json(self, ref: dict):
        if ref['duong_dan'] not in self._json:
            self._json[ref['duong_dan']] = _open_ref(self.goc, ref, f'ghep: {ref["duong_dan"]}')
        return self._json[ref['duong_dan']]

    def all_json(self, pred) -> List[Tuple[dict, dict]]:
        out = []
        for sha, rels in sorted(self.by_sha.items()):
            for rel in rels:
                if rel.lower().endswith('.json'):
                    ref = dict(duong_dan=rel, sha256=sha)
                    try:
                        x = self.json(ref)
                    except GateFileError:
                        continue
                    if pred(x, rel):
                        out.append((ref, x))
                        break
        return out


def _ref_of(kho: _Kho, x, what: str) -> Optional[dict]:
    """A {duong_dan, sha256} (or null) a source states, found again under the C8 folder by its sha256."""
    if x is None:
        return None
    if not isinstance(x, dict) or not _hex64(x.get('sha256')):
        raise GateFileError(f'ghep: {what} {x!r}: {{duong_dan, sha256}} or null')
    return kho.by(x['sha256'], what)


def _band_list(bands, key: str, conv) -> list:
    return [dict(u_lo=b['u_lo_m_s'], u_hi=b['u_hi_m_s'], gia_tri=None if b['trang_thai'] == 'thieu' else conv(b[key]))
            for b in bands]


def _man_trap_rel(goc: str, v):
    """A shutter branch of the C8b lock with its 3b source given relative to the C8 folder (GC9-1)."""
    if isinstance(v, dict) and isinstance(v.get('nguon_3b'), dict):
        return dict(v, nguon_3b=dict(v['nguon_3b'], duong_dan=_rel_in(goc, v['nguon_3b'].get('duong_dan'))))
    return v


def _terms_plain(x):
    """u_p / u_psi of a level for the decision sheet (no parsed bands)."""
    return {k: v for k, v in x.items() if not k.startswith('_')}


def _json_plain(o):
    """NaN / inf as null for a JSON file."""
    if isinstance(o, dict):
        return {k: _json_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_plain(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def phieu_chon(open_cases: list) -> dict:
    """The decision sheet (phieu_chon_v1, CODE 24/11: Liam chooses once, never the tool) of the bien_nguong cases:
    per case the level, the condition, its inputs (a_m, u_p, u_psi, U_t,C8 and the u_t_s of the eligible runs), the
    Uhat block (uhat null: not told within the budget; uhat_duoi the lower bound), the threshold and its half with the
    strict comparison of rule 2b, and both choices with the so_do of the level each gives."""
    ca = []
    for x in open_cases:
        lv, c, uh = x['muc'], x['dieu_kien'], x['uhat']
        thr = o1_gate.THRESHOLDS[c]
        opts = {}
        for chon in ('so_do', 'quyet'):
            q = dict(x['quyet_khac'])
            q[c] = dict(chon=chon)
            for o in x['mo_khac']:                     # the other open cases of the level: taken as 'quyet' here
                q.setdefault(o, dict(chon='quyet'))
            opts[chon] = dict(so_do_muc=list(so_do_of(lv, uh, q)[0]),
                              nghia=('quyết bằng số đo q, chưa cộng U (khong_dat khi q > ngưỡng)' if chon == 'so_do'
                                     else 'ba nhánh có U (đạt khi cận trên ≤ ngưỡng, không đạt khi cận dưới > ngưỡng, '
                                          'còn lại chưa kết luận)'))
        ca.append(dict(muc=lv, dieu_kien=c, nguong=thr, nua_nguong=0.5 * thr,
                       so_sanh='Û > ½ ngưỡng (chặt) → so_do; Û ≤ ½ ngưỡng → quyet (thiết kế v6_C8 mục 4, 5)',
                       uhat=uh[c], u_t_c8_s=x['u_t'],
                       dau_vao=dict(a_m=x['a_m'], u_p=_terms_plain(x['u_p']), u_psi=_terms_plain(x['u_psi']),
                                    luot=[dict(run_id=r.get('run_id'), u_t_s=r['u_t_s']) for r in x['runs']]),
                       mo_khac_cung_muc=x['mo_khac'],
                       gia_dinh=('so_do_muc của mỗi lựa chọn tính các ca biên khác cùng mức là quyet'
                                 if x['mo_khac'] else ''),
                       so_do_muc_moi_ca_bien_so_do=list(so_do_of(lv, uh, dict(
                           x['quyet_khac'], **{o: dict(chon='so_do') for o in [c] + x['mo_khac']}))[0]),
                       lua_chon=opts))
    return dict(format='phieu_chon_v1', ly_do=('Û không tách được khỏi ½ ngưỡng trong ngân sách tính (bien_nguong): '
                                               'J v6 không quyết mặt nạ, C8 chưa khóa; CAN LIAM QUYET (J mục 0b)'),
                cach_ghi=('file --liam của ghep: quyet_liam.<muc>.<dieu_kien> = {chon: so_do | quyet, ngay, phien, '
                          'nguyen_van}, rồi chạy lại ghep'),
                ca=ca)


def ghep_va_ghi(out: str, goc: str, *args, **kw) -> str:
    """ghep then write_gate; the dossier ghep wrote in this call is taken back when the gate is refused (GC9-8)."""
    p = os.path.join(goc, args[6] if len(args) > 6 else kw['ho_so'])
    had = os.path.exists(p)
    try:
        return write_gate(out, ghep(goc, *args, **kw), goc)
    except Exception:
        if not had and os.path.exists(p):
            os.remove(p)
        raise


def ghep(goc: str, ke_hoach: str, thu_tu_clip: str, khoa_c8b: str, tien_de, kiem_cheo,
         bang_chung: Dict[str, str], ho_so: str, muc_bo_theo_liam: Optional[dict] = None,
         quyet_liam: Optional[dict] = None, o_hong_theo_liam: Optional[dict] = None,
         bo_cham_cu: Optional[dict] = None) -> dict:
    """The gate content of C8 assembled from its sources under the C8 folder goc (paths relative to it; F10-F19, the
    contract of 19/11 with C8B: the gate reads the C8B files, C8B does not call the gate):
      sessions     thu_tu_clip: the thu_tu_clip.csv of one C8 session (a path) or of each, in session order (a list;
                   CODE 22/11: a lost run rerun in a later floor session; the folder of a session is that of its file)
      cells        the plan (ke_hoach_c8_v1) and thu_tu_clip.csv: every physical run of a cell in clip order, run_id =
                   the buoc of its row; its scoring luoi.json is the one of that clip (clip.ten) that is no second
                   scoring of another (file_cham_h1); the time file, log, score.json of its operations, clip H
                   comparison, second scoring and time file are those the sources name by sha256; the e_do and
                   evidence files those made from them (gio_sha256, nguon)
      outcomes     lost when a source shows a lost kind (the kinds in J v6 order), else the outcome after the clip H
                   comparison; the time check state and slip counter of the time file; the cell state recounted
      levels       the three kinds of evidence, Liam's drops (muc_bo_theo_liam), the closed terms and speed bands of the
                   C8b lock (hang_dong, hang_theo_u), U_t,C8 and Uhat (level_block), Liam's choices (quyet_liam)
      kept lost    o_hong_theo_liam {cell: Liam's record}: the cells whose last run is lost to the measurement and Liam
                   keeps lost without a rerun (Liam 22/11, C8 contract section 3 (a)); hong_theo_liam, the record in
                   the cell, no eligible run there (a level then lacks that controller unless Liam drops it)
      U standard   the four evidence files (bang_chung {condition: path}) and the dossier ho_so_c8_v2 written to ho_so
                   (cham_luc = the first scoring time of every run, _scoring_times; giu_ngoai = every C8 run, CODE
                   20/11, GC9-2; khoa_kiem_luc = per C8 session, in session order, the khoa_s7 record of the
                   kiem_cheo.json of that session (kiem_cheo: a path or a list, one per session, each in the folder of
                   its session), its file by the path it names; CODE 24/11: the S7 lock per session)
      premise      one tien_de_buoi_v1 file per camera segment that holds C8 runs (tien_de: a path or a list) and
                   the three session files each names by sha256, by role (dau1, dau2, cuoi; CODE 20/11); each with the
                   session whose folder holds it, its files and the logs of its rows found in that folder (CODE 22/11)
      camera       khoa_sha256, mien_car_pose, eps_H of the C8b lock; the U manifests of the eligible scorings, the
                   largest of each of their four numbers and their common nguon
    The content is checked by write_gate (check and check_nguon) before anything is written; ho_so is written first
    (a new file, or the same bytes already there). GateFileError when a source is missing or two disagree."""
    if _bad_rel(ho_so):                                  # GC9-8: never a file outside the C8 folder
        raise GateFileError(f'ghep: --ho-so {ho_so!r}: a path relative to the C8 folder, inside it')
    kho = _Kho(goc)
    kh_ref, kc8_ref = (kho.ref(p) for p in (ke_hoach, khoa_c8b))
    tt_refs = [kho.ref(p) for p in ([thu_tu_clip] if isinstance(thu_tu_clip, str) else list(thu_tu_clip))]
    kh, khoa = kho.json(kh_ref), kho.json(kc8_ref)
    kcj_refs = [kho.ref(p) for p in ([kiem_cheo] if isinstance(kiem_cheo, str) else list(kiem_cheo))]
    if len(kcj_refs) != len(tt_refs):
        raise GateFileError(f'ghep: {len(kcj_refs)} kiem_cheo.json for {len(tt_refs)} C8 sessions: one per session, in '
                            'session order (CODE 24/11: the S7 lock of each session)')
    for k, (r_, t_) in enumerate(zip(kcj_refs, tt_refs)):
        if not _under(r_['duong_dan'], _folder_of(t_)):
            raise GateFileError(f'ghep: kiem_cheo {r_["duong_dan"]!r} is not in the folder of session {k} '
                                f'({_folder_of(t_)!r})')
    td_refs = [kho.ref(p) for p in ([tien_de] if isinstance(tien_de, str) else list(tien_de))]   # one per segment
    if set(bang_chung) != set(CHUAN_U_DK):
        raise GateFileError(f'ghep: bang_chung needs exactly the four conditions {CHUAN_U_DK}')
    kt = khoa.get('muc_kha_thi')
    if not isinstance(kt, list) or not kt:
        raise GateFileError('ghep: the C8b lock has no muc_kha_thi')
    rows, slices = _buoi_rows(kho.goc, tt_refs)          # CODE 22/11: one thu_tu_clip.csv per C8 session, in order
    plan = {x['o']: x for x in kh.get('o') or []}
    giu_hong = o_hong_theo_liam or {}
    if not isinstance(giu_hong, dict) or any(o not in plan for o in giu_hong):
        raise GateFileError(f'ghep: o_hong_theo_liam {giu_hong!r}: an object keyed by cells of the plan')
    for o, x in giu_hong.items():
        _liam(x, f'ghep: o_hong_theo_liam.{o}')
    order = {o: [] for o in plan}
    for r in rows:
        b = re.sub(r'_lai\d+$', '', (r.get('buoc') or '').strip())
        if b in plan:
            order[b].append(r)
    commit_bo = khoa.get('commit_hil') if khoa.get('nhanh_tick') == 'bo_tick' else None
    luois = kho.all_json(lambda x, rel: os.path.basename(rel) == 'luoi.json' and isinstance(x, dict)
                         and 'trang_thai_luot' in x and (commit_bo is None or luoi_commit(x) == commit_bo))   # KG-6
    h1_shas = {x['file_cham_h1']['sha256'] for _, x in luois if isinstance(x.get('file_cham_h1'), dict)}
    edos = kho.all_json(lambda x, rel: isinstance(x, dict) and 'vuot_moi_nut' in x and 'gio_sha256' in x)
    ccs = kho.all_json(lambda x, rel: isinstance(x, dict) and isinstance(x.get('chung_cu_c8'), dict))
    cells, runs, cham_luc, eligible = [], {lv: [] for lv in kt}, {}, []
    for o, x in plan.items():
        chain, last = [], None
        for r in order[o]:
            w = f'ghep: run {r["buoc"]} (clip {r["clip"]})'
            h0 = [(ref, l) for ref, l in luois if isinstance(l.get('clip'), dict) and l['clip'].get('ten') == r['clip']
                  and ref['sha256'] not in h1_shas]
            if len(h0) != 1:
                raise GateFileError(f'{w}: {len(h0)} first scorings (luoi.json of that clip); one needed')
            fc_ref, fc = h0[0]
            fg_ref = None if fc.get('gio_sha256') is None else kho.by(fc['gio_sha256'], f'{w}: time file')
            fg = None if fg_ref is None else kho.json(fg_ref)
            log_ref = tt_src = None
            bf = _folder_of(tt_refs[r['_buoi']])                        # CODE 22/11: the log in its session's folder
            if fg is not None:
                esp = (fg.get('sha256') or {}).get('esp32')
                name = next((n.get('file') for n in fg.get('nguon') or [] if isinstance(n, dict)
                             and n.get('sha256') == esp), None)
                log_ref = kho.by(esp, f'{w}: ESP32 log', name, under=bf)
                tt_src = _ref_of(kho, fg.get('thao_tac_nguon'), f'{w}: thao_tac_nguon')
            if fg is None and (r.get('log_pi') or '').strip():          # GC6-1 (round 5): by the name the CSV gives
                ln = os.path.basename(r['log_pi'].strip().replace('\\', '/'))
                hits = [q for qs in kho.by_sha.values() for q in qs if os.path.basename(q) == ln and _under(q, bf)]
                if len(hits) != 1:
                    raise GateFileError(f'{w}: the log {ln!r} thu_tu_clip.csv names is on {len(hits)} files in the '
                                        f'folder {bf!r} of its session; one needed')
                log_ref = kho.ref(hits[0])
            fc1_ref = _ref_of(kho, fc.get('file_cham_h1'), f'{w}: file_cham_h1')
            fg1_ref = _ref_of(kho, fc.get('file_gio_h1'), f'{w}: file_gio_h1')
            dr_ref = _ref_of(kho, fc.get('file_so_h'), f'{w}: file_so_h')
            fc1 = None if fc1_ref is None else kho.json(fc1_ref)
            fg1 = None if fg1_ref is None else kho.json(fg1_ref)
            nodes = {id(fcx): [kho.json(kho.by(nn['sha256'], w)) for nn in fcx.get('nguon_nut') or []]
                     for fcx in (fc, fc1) if fcx is not None}
            ops = sorted((kho.json(tt_src).get('thao_tac') or []) if tt_src is not None else
                         {v for sc in nodes[id(fc)] for v in sc.get('thao_tac') or []})
            src = set()
            for fcx, fgx in ((fc, fg), (fc1, fg1)):
                if fcx is not None:
                    src |= _lost_kinds(fcx, fgx, ops, nodes[id(fcx)], w)
            thieu = any(s_ is not None and s_.get('trang_thai_luot') == 'cham' and s_.get('u_max_s') is None
                        for s_ in (fc, fc1))
            if src:
                kq = 'hong'
            else:
                if dr_ref is None:
                    raise GateFileError(f'{w}: no clip H comparison (luoi.json file_so_h) for a run that is not lost')
                kq = _final_outcome(fc['outcome'], None if fc1 is None else fc1['outcome'], kho.json(dr_ref))
                if kq is None:
                    raise GateFileError(f'{w}: above eps_H the second scoring is due (luoi.json file_cham_h1)')
                if thieu and kq != 'hong_he_do':
                    kq = 'chua_ket_luan'                          # GC9-6 (Liam 20/11): no U_t at a node
            # GC9-7: the evidence file made from exactly these scorings names its e_do files (nguon: the scorings
            # then their e_do files, each made from the time file of its scoring); other e_do copies do not matter
            pairs = [(fc_ref, fg_ref)] + ([(fc1_ref, fg1_ref)] if fc1_ref is not None else [])
            cc_ref, e_refs = None, [None] * len(pairs)
            if fg_ref is not None:
                edo_of = {ref['sha256']: e for ref, e in edos}
                hit = []
                for ref, c in ccs:
                    ns = [q.get('sha256') for q in c.get('nguon') or [] if isinstance(q, dict)]
                    if len(ns) == 2 * len(pairs) and ns[:len(pairs)] == [q[0]['sha256'] for q in pairs] and all(
                            edo_of.get(es, {}).get('gio_sha256') == q[1]['sha256'] for es, q in zip(ns[len(pairs):], pairs)
                            if q[1] is not None):
                        hit.append((ref, ns[len(pairs):]))
                if len(hit) == 1:
                    cc_ref, e_refs = hit[0][0], [kho.by(es, f'{w}: e_do file') for es in hit[0][1]]
            ent = dict(run_id=r['buoc'], clip=r['clip'], ket_qua=kq, loai_hong=[k for k in LOST_KINDS if k in src],
                       kiem_gio_d1=fg['kiem_gio_d1']['trang_thai'] if fg is not None else 'khong_kiem',
                       dem_truot_kiem_gio=fg['dem_truot_kiem_gio'] if fg is not None else False,
                       file_cham=fc_ref, file_gio=fg_ref, file_chung_cu=cc_ref, file_so_h=dr_ref,
                       file_cham_h1=fc1_ref, file_gio_h1=fg1_ref,
                       file_edo=e_refs[0] if cc_ref else None,
                       file_edo_h1=(e_refs[1] if cc_ref and len(e_refs) > 1 else None),
                       file_log=log_ref, file_thao_tac=tt_src, thieu_u_t=bool(thieu and not src))
            chain.append(ent)
            cham_luc[ent['run_id']] = min(t for s in (fc, fc1) if s is not None for t in _scoring_times(s, w)
                                          ).isoformat()
            last = (ent, fc, fg, cc_ref)
        if not chain:
            raise GateFileError(f'ghep: cell {o}: no run in thu_tu_clip.csv')
        try:
            st, _ = count_chain(chain, hong_theo_liam=o in giu_hong)
        except GateFileError as exc:
            raise GateFileError(f'ghep: cell {o}: {exc}')
        if o in giu_hong and st != 'hong_theo_liam':
            raise GateFileError(f'ghep: cell {o}: Liam\'s record keeps it lost but the chain gives {st!r}')
        if st == 'co_luot' and chain[-1]['thieu_u_t']:
            st = 'thieu_u_t'                                  # GC9-6: no eligible run in the cell
        cells.append(dict(o=o, muc=x['muc'], controller=x['controller'], trang_thai=st, chuoi=chain,
                          **({'liam': dict(giu_hong[o])} if o in giu_hong else {})))
        if st == 'co_luot':
            ent, fc, fg, cc_ref = last
            if fg is None or cc_ref is None:
                raise GateFileError(f'ghep: cell {o}: its last run {ent["run_id"]} has no time file or no evidence file '
                                    '(c8b_edo chung-cu made from its scorings and e_do files)')
            h, um = fg.get('h_s'), fc.get('u_max_s')
            runs[x['muc']].append(dict(run_id=ent['run_id'], controller=x['controller'], o=o, h_s=h, u_max_s=um,
                                       u_t_s=_num(h, f'{o} h_s') + _num(um, f'{o} u_max_s'),
                                       chung_cu=kho.json(cc_ref)['chung_cu_c8']))
            eligible += [s['sha256'] for s in (ent['file_cham'], ent['file_cham_h1']) if s is not None]
    luot_k = []                                          # GC9-2: the K runs, their grids (C8B: one luoi.json each)
    for r in rows:
        b = (r.get('buoc') or '').strip()
        if not RUN_CODE_K.match(b):
            continue
        w = f'ghep: K run {b} (clip {r["clip"]})'
        h0 = [(ref, l) for ref, l in luois if isinstance(l.get('clip'), dict) and l['clip'].get('ten') == r['clip']
              and ref['sha256'] not in h1_shas]
        if len(h0) != 1:
            raise GateFileError(f'{w}: {len(h0)} grid files (luoi.json of that clip); one needed: its scoring time is '
                                'that of a held-out run (CODE 20/11)')
        ref, l = h0[0]
        luot_k.append(dict(run_id=b, clip=r['clip'], file_cham=ref,
                           file_cham_h1=_ref_of(kho, l.get('file_cham_h1'), f'{w}: file_cham_h1')))
        cham_luc[b] = min(t for s_ in [l] + ([kho.json(luot_k[-1]['file_cham_h1'])] if luot_k[-1]['file_cham_h1']
                                             else []) for t in _scoring_times(s_, w)).isoformat()
    bo = muc_bo_theo_liam or {}
    ql = quyet_liam or {}
    loai = {k: evidence_of(runs[k])['loai'] for k in kt}
    levels = [k for k in kt if not loai[k] and k not in bo]
    muc = khoa.get('muc') or {}
    per, terms = {}, {}
    for lv in levels:
        hd, bands = (muc.get(lv) or {}).get('hang_dong'), (muc.get(lv) or {}).get('hang_theo_u')
        if not isinstance(hd, dict) or any(hd.get(q) is None for q in ('a_m', 'hang_m', 'he_so_r', 'u_phi_rad',
                                                                       'hang_rad')) or not isinstance(bands, list):
            raise GateFileError(f'ghep: {lv}: the C8b lock has no number for every closed term (muc.{lv}.hang_dong '
                                f'{hd!r}, ly do {(muc.get(lv) or {}).get("hang_dong_ly_do")!r}): C8 is not locked')
        u_p = dict(hang_m=hd['hang_m'], he_so_r=hd['he_so_r'], u_phi_rad=hd['u_phi_rad'],
                   hang_theo_u=_band_list(bands, 'P4_m', float))
        u_psi = dict(hang_rad=hd['hang_rad'], hang_theo_u=_band_list(bands, 'A4_deg', math.radians))
        terms[lv] = (hd['a_m'], u_p, u_psi, level_uhat(lv, hd['a_m'], u_p, u_psi, runs[lv]))
    # CODE 24/11: every bien_nguong without Liam's choice, of every level, on one decision sheet (both choices); no
    # choice is made here
    open_cases = []
    for lv in levels:
        a_m, u_p, u_psi, (u_t, uh) = terms[lv]
        mo = [c for c in MASKABLE if uh[c]['trang_thai'] == 'bien_nguong' and not (lv == G5_F08 and c == 'g5')
              and (ql.get(lv) or {}).get(c) is None]
        for c in mo:
            open_cases.append(dict(muc=lv, dieu_kien=c, uhat=uh, u_t=u_t, a_m=a_m, u_p=u_p, u_psi=u_psi,
                                   runs=runs[lv], quyet_khac=dict(ql.get(lv) or {}),
                                   mo_khac=[o for o in mo if o != c]))
    if open_cases:
        raise CanLiamChon(f'ghep: {len(open_cases)} Uhat cannot be told from half the threshold within the budget '
                          f'({", ".join(x["muc"] + " " + x["dieu_kien"] for x in open_cases)}): C8 is not locked; CAN '
                          'LIAM QUYET (J section 0b); both choices on the decision sheet', phieu_chon(open_cases))
    for lv in levels:
        a_m, u_p, u_psi, uhu = terms[lv]
        per[lv] = level_block(lv, a_m, u_p, u_psi, runs[lv], ql.get(lv), uh=uhu)
    # premise: one tien_de file per camera segment, its three session files by role (CODE 20/11 after C8 session 2);
    # each in the folder of its session (CODE 22/11), the segments of a session in order (row of their clip H at the end
    # in its thu_tu_clip.csv)
    per_buoi = [[] for _ in tt_refs]
    for td_ref in td_refs:
        ks = [k for k, t in enumerate(tt_refs) if _under(td_ref['duong_dan'], _folder_of(t))]
        if len(ks) != 1:
            raise GateFileError(f'ghep: tien_de file {td_ref["duong_dan"]!r} is in the folder of {len(ks)} C8 sessions '
                                '(the folders of the thu_tu_clip files); one needed')
        td = kho.json(td_ref)
        roles = td.get('vai_tro_clip_h') or {}
        ses = (td.get('sha256') or {}).get('sessions') or {}
        by_role = {v: n for n, v in roles.items()}
        if sorted(by_role) != sorted(TD_ROLES) or len(by_role) != len(roles):
            raise GateFileError(f'ghep: tien_de file {td_ref["duong_dan"]!r}: vai_tro_clip_h gives {sorted(roles.values())}'
                                f', the roles {TD_ROLES} once each needed')
        bf = _folder_of(tt_refs[ks[0]])
        cr = [kho.by(ses.get(by_role[role]), f'session of {role} ({td_ref["duong_dan"]})', under=bf) for role in TD_ROLES]
        kcr = kho.by((td.get('sha256') or {}).get('kiem_cheo'), f'kiem_cheo.json of {td_ref["duong_dan"]}', under=bf)
        per_buoi[ks[0]].append((td.get('session_id'), dict(ten=(td.get('doan') or {}).get('ten'), dau=cr[:2],
                                                            cuoi=cr[2], file=td_ref, kiem_cheo=kcr)))
    tien_de_c8 = []
    for k, segs in enumerate(per_buoi):
        ids = {b for b, _ in segs}
        if len(ids) != 1:
            raise GateFileError(f'ghep: the premise files of the session of {tt_refs[k]["duong_dan"]!r} are of the '
                                f'sessions {sorted(map(str, ids))}; one C8 session (one premise per camera segment)')
        b0, b1 = slices[k]
        doan = sorted((s_ for _, s_ in segs), key=lambda s_: _clip_row(
            kho.goc, s_['cuoi'], f'ghep: clip H at the end of {s_["ten"]!r}', rows[b0:b1]))
        tien_de_c8.append(dict(buoi=next(iter(ids)), doan=doan))
    # the U standard and its dossier (after the premise: khoa_kiem_luc names each session)
    bc_refs = {k: kho.ref(p) for k, p in bang_chung.items()}
    kkl, s7_same = [], None
    for k, kcj_ref in enumerate(kcj_refs):
        kcj = kho.json(kcj_ref)
        rec = kcj.get('khoa_s7') if isinstance(kcj.get('khoa_s7'), dict) else {}
        ks_ref = kho.ref(_rel_in(kho.goc, rec.get('file')))   # GC9-7: the very file kiem_cheo.json names
        if ks_ref['sha256'] != rec.get('sha256'):
            raise GateFileError(f'ghep: khoa_s7 {rec.get("file")!r} is not the manifest of sha256 {rec.get("sha256")!r} '
                                f'{kcj_ref["duong_dan"]!r} names')
        ksj = kho.json(ks_ref)
        same = {q: ksj.get(q) for q in S7_SAME}
        if s7_same is None:
            s7_same = same
        elif same != s7_same:
            raise GateFileError(f'ghep: khoa_s7 {ks_ref["duong_dan"]!r}: {list(S7_SAME)} differ from those of session 0; '
                                'one S7 check for every session (CODE 24/11)')
        kkl.append(dict(buoi=tien_de_c8[k]['buoi'], kiem_cheo=kcj_ref, khoa_s7=ks_ref,
                        git_commit=rec.get('git_commit'), git_time=rec.get('git_time')))
    all_runs = [e['run_id'] for c in cells for e in c['chuoi']] + [x['run_id'] for x in luot_k]
    bo_cu = scorings_dropped(kho.goc, commit_bo)            # KG-6: the scorings before the merged lock, dropped
    if bo_cu and bo_cham_cu is None:
        raise GateFileError(f'ghep: {len(bo_cu)} c8b_luoi files under the C8 folder were made at another commit than '
                            'commit_hil of the lock (scorings before the merged lock); the dossier lists them as '
                            'dropped with Liam\'s record: --liam bo_cham_cu {ngay, phien, nguyen_van} (KG-6)')
    if bo_cham_cu is not None:
        _liam(bo_cham_cu, 'ghep: bo_cham_cu')
    hs = dict(format=HO_SO_FORMAT, bang_chung={k: v['sha256'] for k, v in bc_refs.items()},
              ke_hoach_c8_sha256=kh_ref['sha256'], khoa_c8b_sha256=kc8_ref['sha256'],
              giu_ngoai=all_runs, cham_luc=cham_luc, khoa_kiem_luc=kkl,
              cham_cu_bo=dict(luoi=bo_cu, liam=dict(bo_cham_cu) if bo_cu else None))
    raw = (json.dumps(hs, indent=1, ensure_ascii=False) + '\n').encode('utf-8')
    p = os.path.join(kho.goc, ho_so)
    q = os.path.realpath(os.path.dirname(p))
    if not os.path.isdir(q) or os.path.commonpath([kho.goc, q]) != kho.goc:      # GC9-8: a link out of the folder
        raise GateFileError(f'ghep: --ho-so {ho_so!r}: its folder must exist inside the C8 folder')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            if f.read() != raw:
                raise GateFileError(f'ghep: {ho_so!r} exists with other bytes (a dossier is never replaced)')
    else:
        with open(p, 'xb') as f:
            f.write(raw)
    # camera: the U manifests of the eligible scorings
    mans = kho.all_json(lambda m, rel: isinstance(m, dict) and m.get('format') == MANIFEST_FORMAT
                        and m.get('file_cham_sha256') in eligible)
    if not mans:
        raise GateFileError('ghep: no U manifest (u_moc_manifest_v1) of an eligible scoring')
    ngs = [m.get('nguon') if isinstance(m.get('nguon'), dict) else {} for _, m in mans]
    if any({k: n.get(k) for k in MOC_NGUON} != {k: ngs[0].get(k) for k in MOC_NGUON} for n in ngs):
        raise GateFileError(f'ghep: the U manifests do not share one {", ".join(MOC_NGUON)} (CAM; CODE 22/11)')
    rules = [_manifest_lich(kho.goc, m, f'ghep: U manifest {ref["duong_dan"]}')[1] for ref, m in mans]
    if any(r != rules[0] for r in rules):
        raise GateFileError(f'ghep: the U manifests name lich files of {len({json.dumps(r, sort_keys=True) for r in rules})}'
                            ' rules; one rule for every camera segment and session (CODE 22/11)')
    md = {k: max(_num(m.get(k), f'manifest {k}') for _, m in mans) for k in MOC_KEYS}
    md['nguon'] = dict({k: ngs[0][k] for k in MOC_NGUON}, luat_lich=rules[0], manifests=[ref for ref, _ in mans])
    if 'mien_car_pose' not in khoa or 'eps_H' not in khoa:
        raise GateFileError('ghep: the C8b lock has no mien_car_pose / eps_H (GATE to C8B 19/11)')
    cam = {'c8b': dict(moc_dich=md, khoa_sha256=kc8_ref['sha256'], mien_car_pose=khoa['mien_car_pose'],
                       eps_H=khoa['eps_H'])}
    sc = (khoa.get('khoa_truoc') or {}).get('script')
    phien = (f'{sc.get("file", "")}@{sc.get("commit", "")}' if isinstance(sc, dict) else str(sc or '')).strip('@')
    try:
        import subprocess
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', 'gate_c8.py'], capture_output=True,
                           text=True, timeout=10)
        wb = dict(script='tools/gate_c8.py ghep', commit=c.stdout.strip(), dirty=bool(s.stdout.strip()))
    except (OSError, ValueError):
        wb = dict(script='tools/gate_c8.py ghep', commit='', dirty=True)
    return dict(format=FORMAT, c8_dir=os.path.basename(kho.goc), written_by=wb, levels=levels, muc_kha_thi=list(kt),
                muc_bo_theo_liam=bo, quyet_liam=ql, u_t_branch=khoa.get('nhanh_tick'), lich_doi_chieu=LICH,
                bo_cham=dict(phien_ban=phien or kc8_ref['sha256'],
                             bang_man_trap={k: _man_trap_rel(kho.goc, (khoa.get('bang_man_trap') or {}).get(k))
                                            for k in list(kt) + [TASK_K]}),
                ke_hoach_c8=kh_ref, thu_tu_clip=tt_refs, khoa_c8b=kc8_ref, o_c8=cells,
                chuan_u=dict(loai=CHUAN_U_LOAI, le=CHUAN_U_LE, ho_so_c8=kho.ref(ho_so),
                             dieu_kien={k: dict(trang_thai='dat', bang_chung=bc_refs[k]) for k in CHUAN_U_DK}),
                tien_de_c8=tien_de_c8,
                runs_c8=runs, muc_loai_e_do=loai,
                chung_cu_muc={k: {q: v for q, v in evidence_of(runs[k]).items() if q != 'loai'} for k in kt if not loai[k]},
                dieu_2b=dict(thresholds={c: o1_gate.THRESHOLDS[c] for c in MASKABLE}, per_level=per), camera=cam,
                luot_k=luot_k)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='gate_c8: the C8 plan file ke_hoach_c8_v1 (F17) and the C8 gate file '
                                             'gate_c8_v2 assembled from its sources (ghep)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    kh = sub.add_parser('ke-hoach', help='the plan of the C8 cells from the locked run list of the session guide')
    kh.add_argument('--out', required=True, help='new file (never replaced), under the C8 folder')
    g = sub.add_parser('ghep', help='the gate file gate_c8_v2 from the sources under the C8 folder (see ghep())')
    g.add_argument('--goc', required=True, help='the C8 folder (every other path relative to it)')
    g.add_argument('--ke-hoach', required=True)
    g.add_argument('--thu-tu-clip', action='append', required=True,
                   help='the thu_tu_clip.csv of a C8 session, once per session in session order (the folder of the '
                        'session is its folder; CODE 22/11)')
    g.add_argument('--khoa-c8b', required=True, help='c8b_khoa_v5')
    g.add_argument('--tien-de', action='append', required=True,
                   help='tien_de_buoi_v1 of KIEMCHEO, once per camera segment that holds C8 runs')
    g.add_argument('--kiem-cheo', action='append', required=True,
                   help='kiem_cheo.json (c7_kiem_cheo_v7) with its khoa_s7 record, once per C8 session in session '
                        'order, in the folder of its session (CODE 24/11: the S7 lock per session)')
    g.add_argument('--bang-chung', action='append', required=True, help='CONDITION=path, once for each of the four')
    g.add_argument('--ho-so', required=True, help='the dossier ho_so_c8_v1 to write (new, or the same bytes)')
    g.add_argument('--liam', default=None, help='JSON {muc_bo_theo_liam, quyet_liam, o_hong_theo_liam, bo_cham_cu} of Liam\'s '
                   'recorded choices')
    g.add_argument('--out', required=True, help='the gate file (new, never replaced; not under the C8 folder)')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'ke-hoach':
            with open(a.out, 'x', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(ke_hoach_c8(), indent=1, ensure_ascii=False) + '\n')
            print('ke_hoach', a.out, sha256_file(a.out))
            return 0
        bc = {}
        for s in a.bang_chung:
            k, sep, p = s.partition('=')
            if not sep or k not in CHUAN_U_DK or k in bc:
                raise GateFileError(f'--bang-chung {s!r}: CONDITION=path, CONDITION one of {CHUAN_U_DK}, once each')
            bc[k] = p
        liam = {}
        if a.liam:
            with open(a.liam, encoding='utf-8') as f:
                liam = json.load(f)
            keys = ('muc_bo_theo_liam', 'quyet_liam', 'o_hong_theo_liam', 'bo_cham_cu')
            if not isinstance(liam, dict) or not set(liam) <= set(keys):
                raise GateFileError(f'--liam {a.liam}: a JSON object with keys among {list(keys)}')
        try:
            sha = ghep_va_ghi(a.out, a.goc, a.ke_hoach, a.thu_tu_clip, a.khoa_c8b, a.tien_de, a.kiem_cheo, bc,
                              a.ho_so, liam.get('muc_bo_theo_liam'), liam.get('quyet_liam'),
                              liam.get('o_hong_theo_liam'), liam.get('bo_cham_cu'))
        except CanLiamChon as exc:
            # CODE 24/11: the decision sheet beside --out (never replaced), no gate file; exit 3
            sp = os.path.splitext(a.out)[0] + '_phieu_chon.json'
            with open(sp, 'x', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(_json_plain(exc.phieu), indent=1, ensure_ascii=False) + '\n')
            print(f'gate_c8: {exc}', file=sys.stderr)
            print('phieu_chon', sp, sha256_file(sp))
            return 3
        print('gate_c8_v2', a.out, sha)
    except (OSError, ValueError, GateFileError) as exc:
        print(f'gate_c8: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
