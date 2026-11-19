"""
gate_c8.py: the C8 gate file (format gate_c8_v2), its reader and its two checks, and the set of conditions that decide
a run. CODE_GATE 19/11/2026, design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_rq3_tuong_doi_v6_C8.md (the C8 part
of the relative-frame scoring, after the scoring rules J v6 03_CODE/docs/2026-11-19_J_Quy_tac_cham_O1_v6.md, root
1836fe65; CODE's order of 19/11, Liam at CODE local_6653c962: "duyệt cả 8, giao việc các phiên phụ theo cách nhanh",
"Code ngay (Recommended)"). gate_c8_v1 (17/11) is refused.

One JSON file written once after the C8 session by the C8b script (c8b_u cong, CODE_C8B) through write_gate; its
SHA-256 (of the bytes) is "the gate file SHA" carried by every run record. Two checks (design section 3):

  check(d)              the content alone: types, keys, the chain of every C8 cell recounted (reruns cap, M2), the
                        runs of each level rebuilt from the chains, the three kinds of evidence and the gate levels
                        recomputed from the copied evidence, Uhat recomputed from the schedule (section 4), the
                        report-only conditions recomputed (section 5). Every read (read_gate_c8).
  check_nguon(d, goc)   the sources under the C8 folder goc: every {duong_dan, sha256} hashed again (a path inside
                        goc), and the plan, scoring files, time files, premise file, U-standard dossier and U
                        manifests read back and compared with what the gate copied. When the gate is written
                        (write_gate) and when it is registered for a campaign (after C8).

Any difference refuses the whole file (GateFileError). A case J v6 does not decide that touches the lock of C8 (a
Uhat that cannot be told from half the threshold within the budget, J section 0b) is refused with "CAN LIAM QUYET"
unless the gate carries Liam's choice (quyet_liam).

required(level, platform, stage, campaign): the conditions that decide a run:
  SIM                                   all five
  HW at a gate level                    all five minus report_only(level)
  HW at another task (K1, K2, ...)      all five (Liam 17/11, "Không áp, cả năm quyết")
  HIL, stage hil_check, campaign B-*    all five minus report_only(level) (S05 4.5)
  HIL otherwise                         all five
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
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
MASKABLE = ('g2', 'g3', 'g5')           # O1 3.9 rule 2b: G1, G4 always decide
CONTROLLERS = ('PID', 'LADRC')
U_T_BRANCHES = ('bo_tick', 'giu_tick')
B_CAMPAIGNS = ('B-PID', 'B-LADRC')
MASK_HIL_STAGE = 'hil_check'            # S05 4.5: the HIL candidate check of procedure B
PLATFORMS = ('SIM', 'HIL', 'HW')
REL_TOL = 1e-9                          # recomputed numbers: relative difference allowed
LICH = 'o1_feasibility.Schedule'        # the schedule of the comparison (G-7, owner CODE_GATE)
G5_F08 = 'O1-F08'                       # J v6 section 6, Liam 19/11 at CODE: "chỉ báo cáo, không làm bù"

# Uhat (design section 4)
DELTA0_S = 1e-3                         # first grid step
DELTA_MIN_S = 1e-6                      # no refinement step below this
BUDGET = 10 ** 7                        # evaluations of (V, W, u, r) per level, the first grid included
UH_STATES = ('quyet', 'chi_bao_cao', 'thieu', 'bien_nguong')

# Chains of the C8 cells (design section 6; J v6 section 5)
OUTCOMES = ('dat', 'khong_dat', 'chua_ket_luan', 'hong')
LOST_KINDS = ('thao_tac', 'he_do', 'cua_so_tham_chieu', 'duong_lenh')
D1_STATES = ('dat', 'truot', 'khong_kiem')                              # F11: c8b_gio_v1.kiem_gio_d1.trang_thai
RERUNS_MAX = 3                          # J v6 section 5: the first run + 3 reruns per kind
CELL_STATES = ('co_luot', 'dong_tran')
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
LUOI_STATES = ('cham', 'chua_ket_luan_khong_nut', 'hong_he_do')     # c8b_luoi trang_thai_luot (F11, CODE_C8B)
MOC_KEYS = ('U_moc_dong_xy_m', 'U_moc_tham_xy_m', 'U_moc_dong_psi_rad', 'U_moc_tham_psi_rad')

TOP_KEYS = ('format', 'levels', 'muc_kha_thi', 'muc_bo_theo_liam', 'quyet_liam', 'u_t_branch', 'lich_doi_chieu',
            'bo_cham', 'ke_hoach_c8', 'o_c8', 'chuan_u', 'tien_de_c8', 'runs_c8', 'muc_loai_e_do', 'chung_cu_muc',
            'dieu_2b', 'camera')
LEVEL_KEYS = ('a_m', 'u_p', 'u_psi', 'u_t_c8_s', 'uhat', 'report_only', 'ly_do_report_only')
UH_KEYS = ('uhat', 'uhat_duoi', 'delta_cuoi_s', 'so_lan_tinh', 'trang_thai')


class GateFileError(ValueError):
    pass


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


def _close(a, b) -> bool:
    """Equal within REL_TOL, both finite (GC1-2: inf never matches); two nulls are equal."""
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
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


def _ref(x, what: str) -> dict:
    """A source reference {duong_dan (relative to the C8 folder), sha256}."""
    _obj(x, ('duong_dan', 'sha256'), what)
    p = _text(x['duong_dan'], f'{what}.duong_dan')
    if os.path.isabs(p) or p.replace('\\', '/').split('/')[0] == '..':
        raise GateFileError(f'{what}.duong_dan {p!r}: a path relative to the C8 folder, inside it')
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
    """sha256 of a list of frame ids, as the U manifests and scoring files state it (compact JSON, UTF-8)."""
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


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


def _grid_bounds(sch, terms: List[_Terms], lip, a_m, u_t, t_lo, t_hi, delta):
    """Bounds on the cells of a uniform grid of step delta covering [t_lo, t_hi] (design section 4). Returns
    (n_eval, cells: (t_a, t_b) arrays, per condition (F_up per cell, maybe_null per cell, F_low per grid point,
    certain_null))."""
    m = int(math.ceil(u_t / delta - 1e-12))
    mp = int(math.floor(u_t / delta + 1e-12))
    nc = max(1, int(math.ceil((t_hi - t_lo) / delta - 1e-12)))          # cells over [t_lo, t_hi]
    k = np.arange(-(m + 1), nc + m + 2)
    t = t_lo + k * delta
    v, w, u, r = schedule_samples(sch, t, a_m)
    off = m + 1                                                          # index of t_lo
    wu = 2 * m + 2                                                       # E'_j = j - m .. j + 1 + m
    js = off + np.arange(nc)
    vmax = _slide(v, wu, np.maximum)[js - m]
    wmax = _slide(w, wu, np.maximum)[js - m]
    umax = _slide(u, wu, np.maximum)[js - m] + lip['L_u'] * delta / 2.0
    umin = _slide(u, wu, np.minimum)[js - m] - lip['L_u'] * delta / 2.0
    r_up = np.maximum(r[js], r[js + 1]) + lip['L_r'] * delta / 2.0
    # lower: grid points t_j in [t_lo, t_hi] at or after 0 (a frame can be scored there), windows j - mp .. j + mp
    jl = off + np.arange(nc + 1)
    jl = jl[t[jl] >= 0.0]
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
        rate_up = (vmax + lip['L_V'] * delta / 2.0) if tm.rate == 'V' else (wmax + lip['L_W'] * delta / 2.0)
        f_up = (tm.c0 + bv + tm.cr * r_up + tm.ct * rate_up) * tm.scale
        lbv, lbnull, _ = _band_value(tm.bands, lu_lo, lu_hi)
        f_lo = (tm.c0 + lbv + tm.cr * r[jl] + tm.ct * (lv if tm.rate == 'V' else lw)) * tm.scale
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
        alive = (f_up > half) | mnull
        rest = float(f_up[~alive].max()) if (~alive).any() else -math.inf
        ivs = _runs_of(alive, ta, tb)
        while True:
            if lo > half:
                st = 'chi_bao_cao'
                break
            if not ivs:
                st = 'quyet'
                break
            d2 = delta / 2.0
            if d2 < DELTA_MIN_S:
                st = 'bien_nguong'
                break
            m2 = int(math.ceil(u_t / d2 - 1e-12))
            need = sum(int(math.ceil((b - a) / d2 - 1e-12)) + 2 * m2 + 3 for a, b in ivs)
            if used + need > budget:
                st = 'bien_nguong'
                break
            new_ivs, ups = [], [rest]
            certain_null = False
            for a, b in ivs:
                k, (sa, sb), rr2 = _grid_bounds(sch, [tm], lip, a_m, u_t, a, b, d2)
                used += k
                cnt += k
                fu, mn, fl, cn = rr2[0]
                certain_null |= cn
                lo = max(lo, float(fl.max())) if len(fl) else lo
                al = (fu > half) | mn
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
            out[tm.cond] = dict(uhat=up, uhat_duoi=lo, delta_cuoi_s=delta, so_lan_tinh=n + cnt, trang_thai=st)
    return out


def report_only_of(level: str, uh: Dict[str, dict], quyet: Optional[dict] = None) -> Tuple[Tuple[str, ...], dict]:
    """(report-only conditions in MASKABLE order, their reasons) at a level (design section 5). A bien_nguong
    condition needs Liam's choice in quyet (J v6 section 0b), else GateFileError CAN LIAM QUYET."""
    ro, why = [], {}
    for c in MASKABLE:
        rs = []
        st = uh[c]['trang_thai']
        if st == 'chi_bao_cao':
            rs.append('vuot_nua_nguong')
        elif st == 'thieu':
            rs.append('thieu')
        elif st == 'bien_nguong':
            q = (quyet or {}).get(c)
            if q is None:
                raise GateFileError(f'{level} {c}: Uhat cannot be told from half the threshold within the budget '
                                    f'(bien_nguong): J v6 does not decide the mask, C8 is not locked; CAN LIAM QUYET '
                                    f'(J section 0b), recorded in quyet_liam.{level}.{c} {{chon, ngay, phien, '
                                    'nguyen_van}}')
            if q['chon'] == 'chi_bao_cao':
                rs.append('quyet_liam')
        if level == G5_F08 and c == 'g5':
            rs.append('liam_g5_f08')
        if rs:
            ro.append(c)
            why[c] = rs
    return tuple(ro), why


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


def level_block(level: str, a_m: float, u_p: dict, u_psi: dict, runs: list, quyet: Optional[dict] = None) -> dict:
    """The dieu_2b.per_level block of a gate level for the writer (C8b script): a_m (m), u_p {hang_m, he_so_r,
    u_phi_rad, hang_theo_u}, u_psi {hang_rad, hang_theo_u} (null for a term without a checked bound), runs the
    eligible C8 runs of the level (runs_c8 entries, u_t_s each); U_t,C8, Uhat and the mask computed here."""
    e = dict(a_m=a_m, u_p=u_p, u_psi=u_psi)
    t = _level_terms(e, level)
    if not runs:
        raise GateFileError(f'{level}: no eligible C8 run')
    u_t = max(_num(r['u_t_s'], f'{level} u_t_s', positive=True) for r in runs)
    uh = uhat_level(level, t, u_t)
    ro, why = report_only_of(level, uh, quyet)
    return dict(a_m=a_m, u_p=u_p, u_psi=u_psi, u_t_c8_s=u_t, uhat=uh, report_only=list(ro), ly_do_report_only=why)


# ── chains, runs, evidence (design sections 6, 7) ──────────────────────────────────────────────────────────────

def _chain_entry(x, w: str) -> dict:
    _obj(x, ('run_id', 'ket_qua', 'loai_hong', 'kiem_gio_d1', 'dem_truot_kiem_gio', 'file_cham', 'file_gio',
             'file_chung_cu'), w)
    _text(x['run_id'], f'{w}.run_id')
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
    return x


def count_chain(chain: list) -> Tuple[str, Dict[str, int]]:
    """The reruns cap of one C8 cell (J v6 section 5, O1 3.5 line 342, M2 in force): two counters in chain order,
    never reset: time check (a lost run whose time file says dem_truot_kiem_gio: a slip, not lost by a boot unless
    the locked check slipped on the whole window before the boot; CODE_C8B design v7 7a, counted by the field, not
    inferred) and reference window (cua_so_tham_chieu without thao_tac). The fourth of a kind closes the cell (dong_tran) and must
    be the last run; otherwise the last run must not be lost (co_luot). GateFileError when neither."""
    n = {'kiem_gio': 0, 'cua_so': 0}
    for i, x in enumerate(chain):
        if x['ket_qua'] != 'hong':
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
    if chain[-1]['ket_qua'] == 'hong':
        raise GateFileError(f'the last run {chain[-1]["run_id"]!r} is lost and the cell is not at its cap: run it '
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
    seen, runs_seen = set(), set()
    for i, c in enumerate(cells):
        w = f'o_c8[{i}]'
        _obj(c, ('o', 'muc', 'controller', 'trang_thai', 'chuoi'), w)
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
        try:
            st, _ = count_chain(ch)
        except GateFileError as exc:
            raise GateFileError(f'{w} ({c["o"]}): {exc}')
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


def _check_tien_de(d: dict) -> list:
    t = _obj(d['tien_de_c8'], ('clip_h', 'file'), 'tien_de_c8')
    ch = t['clip_h']
    if not isinstance(ch, dict) or set(ch) != {'dau', 'giua', 'cuoi'} or not isinstance(ch['dau'], list) \
            or len(ch['dau']) != 2 or ch['giua'] is None or ch['cuoi'] is None:
        raise GateFileError('tien_de_c8.clip_h: two clips H at the start (dau), one in the middle (giua), one at the '
                            'end (cuoi); missing one (even only H giua), C8 is not locked: CAN LIAM QUYET (J v6 '
                            'section 5 J3-5, section 0b)')
    clips = [_ref(x, f'tien_de_c8.clip_h.dau[{i}]') for i, x in enumerate(ch['dau'])]
    clips += [_ref(ch['giua'], 'tien_de_c8.clip_h.giua'), _ref(ch['cuoi'], 'tien_de_c8.clip_h.cuoi')]
    if len({x['sha256'] for x in clips}) != 4:
        raise GateFileError('tien_de_c8.clip_h: four different clips')
    _ref(t['file'], 'tien_de_c8.file')
    return clips


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
    ng = _obj(md['nguon'], ('bo_lay_mau', 'car_pose', 'tap_moc', 'manifests'), f'{w}.moc_dich.nguon')
    _text(ng['bo_lay_mau'], f'{w}.moc_dich.nguon.bo_lay_mau')
    if not _hex64(ng['tap_moc']):
        raise GateFileError(f'{w}.moc_dich.nguon.tap_moc: sha256 of tap_moc.json (64 hex)')
    if not isinstance(ng['manifests'], list) or not ng['manifests']:
        raise GateFileError(f'{w}.moc_dich.nguon.manifests: the U manifests of the C8 scorings used for the lock')
    for i, x in enumerate(ng['manifests']):
        _ref(x, f'{w}.moc_dich.nguon.manifests[{i}]')
    cp = car_pose_v2_of(ng['car_pose'], f'{w}.moc_dich.nguon')
    mc = _obj(c8b['mien_car_pose'], ('side_m', 'phien_ban', 'h_m', 'kappa'), f'{w}.mien_car_pose')
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
    cells = _check_cells(d)
    runs = _check_runs(d, cells)
    _check_levels(d, runs)
    _check_chuan_u(d)
    _check_tien_de(d)
    q = d['quyet_liam']
    if not isinstance(q, dict) or any(k not in lv for k in q):
        raise GateFileError('quyet_liam: an object keyed by gate levels')
    for k, x in q.items():
        if not isinstance(x, dict) or any(c not in MASKABLE for c in x):
            raise GateFileError(f'quyet_liam.{k}: keyed by conditions of {MASKABLE}')
        for c, y in x.items():
            _liam(y, f'quyet_liam.{k}.{c}')
            if y.get('chon') not in ('chi_bao_cao', 'quyet'):
                raise GateFileError(f'quyet_liam.{k}.{c}.chon {y.get("chon")!r}: chi_bao_cao or quyet')
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
            if x['trang_thai'] != y['trang_thai'] or x['so_lan_tinh'] != y['so_lan_tinh'] \
                    or not _close(x['delta_cuoi_s'], y['delta_cuoi_s']) or not _close(x['uhat'], y['uhat']) \
                    or not _close(x['uhat_duoi'], y['uhat_duoi']):
                raise GateFileError(f'{w}.uhat.{c} {x!r}: recomputed {y!r}')
        ro, why = report_only_of(name, want, q.get(name))
        if not isinstance(e['report_only'], list) or tuple(e['report_only']) != ro:
            raise GateFileError(f'{w}.report_only {e["report_only"]!r}: recomputed {list(ro)!r} (design section 5)')
        if e['ly_do_report_only'] != why:
            raise GateFileError(f'{w}.ly_do_report_only {e["ly_do_report_only"]!r}: recomputed {why!r}')
    _check_camera(d)
    return d


# ── sources (check_nguon, design section 3) ────────────────────────────────────────────────────────────────────

def _open_ref(goc: str, ref: dict, what: str, as_json: bool = True):
    """The file of a reference under goc, hashed again (realpath inside goc). Its JSON when as_json."""
    base = os.path.realpath(goc)
    p = os.path.realpath(os.path.join(base, ref['duong_dan']))
    if os.path.commonpath([base, p]) != base:
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: resolves outside the C8 folder {goc!r}')
    if not os.path.isfile(p):
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: no such file under {goc!r}')
    if sha256_file(p) != ref['sha256']:
        raise GateFileError(f'{what} {ref["duong_dan"]!r}: sha256 differs from the gate (file changed)')
    if not as_json:
        return None
    try:
        with open(p, 'rb') as f:
            x = json.loads(f.read().decode('utf-8'), object_pairs_hook=_no_duplicate_keys)
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


def check_nguon(d: dict, goc: str) -> dict:
    """The sources of a checked gate content under the C8 folder goc (design sections 3, 6-10). Returns d."""
    check(d)
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
    cham_sha, cham_of, chung_of = {}, {}, {}
    for c in d['o_c8']:
        for x in c['chuoi']:
            w = f'cell {c["o"]} run {x["run_id"]}'
            # F17 (CODE 19/11; CODE_C8B 47726c1): the scoring file is the c8b_luoi output luoi.json (outcome,
            # trang_thai_luot, gio_sha256 of its time file, u_max_s); it carries no run id, so it is tied to the run
            # through its time file (one time file per run)
            fc = _open_ref(goc, x['file_cham'], f'{w}: file_cham')
            if fc.get('trang_thai_luot') not in LUOI_STATES:
                raise GateFileError(f'{w}: file_cham is not a c8b_luoi output (trang_thai_luot one of {LUOI_STATES})')
            if x['ket_qua'] != 'hong' and (fc.get('outcome') != x['ket_qua'] or fc['trang_thai_luot'] == 'hong_he_do'):
                raise GateFileError(f'{w}: ket_qua {x["ket_qua"]!r}, the scoring file says outcome '
                                    f'{fc.get("outcome")!r} ({fc["trang_thai_luot"]})')
            if fc['trang_thai_luot'] == 'hong_he_do' and x['ket_qua'] != 'hong':
                raise GateFileError(f'{w}: the scoring file lost every node (hong_he_do), the gate says {x["ket_qua"]!r}')
            if x['file_gio'] is not None and fc.get('gio_sha256') != x['file_gio']['sha256']:
                raise GateFileError(f'{w}: the scoring file was made from another time file (gio_sha256)')
            if x['file_gio'] is None and fc.get('gio_sha256') is not None:
                raise GateFileError(f'{w}: file_gio null but the scoring file names a time file')
            cham_sha[x['file_cham']['sha256']] = fc
            cham_of[x['run_id']] = fc
            if x['file_chung_cu'] is not None:
                cc = _open_ref(goc, x['file_chung_cu'], f'{w}: file_chung_cu')
                src = {s.get('sha256') for s in cc.get('nguon') or [] if isinstance(s, dict)}
                if not isinstance(cc.get('chung_cu_c8'), dict) or x['file_cham']['sha256'] not in src:
                    raise GateFileError(f'{w}: file_chung_cu is a c8b_edo chung-cu output (chung_cu_c8) made from this '
                                        'scoring file (its sha256 in nguon; F17)')
                chung_of[x['run_id']] = cc['chung_cu_c8']
            if x['file_gio'] is None:
                continue
            fg = _open_ref(goc, x['file_gio'], f'{w}: file_gio')
            k1 = fg.get('kiem_gio_d1')
            if fg.get('format') != GIO_FORMAT or not isinstance(k1, dict):
                raise GateFileError(f'{w}: file_gio not a {GIO_FORMAT} file with its kiem_gio_d1 block (F11)')
            if k1.get('trang_thai') != x['kiem_gio_d1']:
                raise GateFileError(f'{w}: kiem_gio_d1 {x["kiem_gio_d1"]!r}, the time file says '
                                    f'{k1.get("trang_thai")!r} (F11)')
            if not isinstance(fg.get('dem_truot_kiem_gio'), bool) or fg['dem_truot_kiem_gio'] != x['dem_truot_kiem_gio']:
                raise GateFileError(f'{w}: dem_truot_kiem_gio {x["dem_truot_kiem_gio"]!r}, the time file says '
                                    f'{fg.get("dem_truot_kiem_gio")!r} (F11, M2; CODE_C8B design v7 7a)')
            if c['chuoi'][-1] is x and c['trang_thai'] == 'co_luot':
                r = next(r for r in d['runs_c8'][c['muc']] if r['run_id'] == x['run_id'])
                if not _close(fg.get('h_s'), r['h_s']) or not _close(fc.get('u_max_s'), r['u_max_s']):
                    raise GateFileError(f'{w}: h_s / u_max_s {r["h_s"]!r} / {r["u_max_s"]!r}, the time file says h_s '
                                        f'{fg.get("h_s")!r}, the scoring file u_max_s {fc.get("u_max_s")!r} (F17)')
                if chung_of.get(x['run_id']) != r['chung_cu']:
                    raise GateFileError(f'{w}: chung_cu differs from chung_cu_c8 of its evidence file (F17)')
    # U standard (design section 8)
    cu = d['chuan_u']
    hs = _open_ref(goc, cu['ho_so_c8'], 'chuan_u.ho_so_c8')
    bc = hs.get('bang_chung')
    want = {k: cu['dieu_kien'][k]['bang_chung']['sha256'] for k in CHUAN_U_DK}
    if bc != want:
        raise GateFileError(f'chuan_u.ho_so_c8: bang_chung {bc!r} must list exactly the four evidence SHAs {want!r}')
    if hs.get('ke_hoach_c8_sha256') != d['ke_hoach_c8']['sha256']:
        raise GateFileError('chuan_u.ho_so_c8: ke_hoach_c8_sha256 is not the plan of the gate')
    for k in CHUAN_U_DK:
        _open_ref(goc, cu['dieu_kien'][k]['bang_chung'], f'chuan_u.dieu_kien.{k}.bang_chung', as_json=False)
    gn = hs.get('giu_ngoai')
    if not isinstance(gn, list) or not gn or any(g not in cham_of for g in gn):
        raise GateFileError('chuan_u.ho_so_c8: giu_ngoai, the held-out runs, a non-empty list of C8 run ids')
    t_lock = _iso_time(hs.get('khoa_kiem_luc'), 'chuan_u.ho_so_c8.khoa_kiem_luc')
    cl = hs.get('cham_luc')                       # F17: GATE assembles the scoring times of the held-out runs
    if not isinstance(cl, dict) or set(cl) != set(gn):
        raise GateFileError('chuan_u.ho_so_c8: cham_luc {run id: ISO time of its first scoring} for exactly giu_ngoai')
    first = min(_iso_time(cl[g], f'held-out run {g}: cham_luc') for g in gn)
    if not t_lock < first:
        raise GateFileError(f'chuan_u: the check was locked at {t_lock.isoformat()}, not before the first scoring of a '
                            f'held-out run ({first.isoformat()}; J v6 section 6 condition (1))')
    # shutter table sources
    for k, x in d['bo_cham']['bang_man_trap'].items():
        if x['nhanh'] == 'alpha':
            _open_ref(goc, x['nguon_3b'], f'bo_cham.bang_man_trap.{k}.nguon_3b', as_json=False)
    # premise of the C8 lock (design section 9)
    clips = _check_tien_de(d)
    for i, x in enumerate(clips):
        _open_ref(goc, x, f'tien_de_c8 clip {i}', as_json=False)
    td = _open_ref(goc, d['tien_de_c8']['file'], 'tien_de_c8.file')
    ses = (td.get('sha256') or {}).get('sessions') if isinstance(td.get('sha256'), dict) else None
    if td.get('format') != TIEN_DE_FORMAT or td.get('ket_qua') != 'dat':
        raise GateFileError(f'tien_de_c8.file: a {TIEN_DE_FORMAT} file with ket_qua dat, got format '
                            f'{td.get("format")!r}, ket_qua {td.get("ket_qua")!r}: C8 is not locked, CAN LIAM QUYET '
                            '(J v6 section 5 J3-5, section 0b)')
    if not isinstance(ses, dict) or set(ses.values()) != {x['sha256'] for x in clips}:
        raise GateFileError('tien_de_c8.file: sha256.sessions must be exactly the four clips H of tien_de_c8.clip_h')
    # shifted anchor (design section 10)
    md = d['camera']['c8b']['moc_dich']
    tops = {k: 0.0 for k in MOC_KEYS}
    for i, ref in enumerate(md['nguon']['manifests']):
        w = f'camera.c8b.moc_dich.nguon.manifests[{i}]'
        mf = _open_ref(goc, ref, w)
        if mf.get('format') != MANIFEST_FORMAT or mf.get('ok') is not True:
            raise GateFileError(f'{w}: a {MANIFEST_FORMAT} file with ok true')
        fs = mf.get('file_cham_sha256')
        if fs not in cham_sha:
            raise GateFileError(f'{w}: file_cham_sha256 is not the scoring file of a C8 run of o_c8')
        fc = cham_sha[fs]
        mien = mf.get('mien_da_tinh')
        if not isinstance(mien, dict) or not isinstance(mien.get('id_khung'), list) \
                or sha256_ids(mien['id_khung']) != mien.get('id_khung_sha256'):
            raise GateFileError(f'{w}: mien_da_tinh {{nut, o_luoi, id_khung, id_khung_sha256}} with the sha256 of its '
                                'frame id list')
        if mien != fc.get('mien_phai_phu'):
            raise GateFileError(f'{w}: mien_da_tinh is not the domain the scoring must cover (mien_phai_phu: nodes, '
                                'grid cells, frame ids incl. the reference window W and the t_done frames)')
        for k in MOC_KEYS:
            tops[k] = max(tops[k], _num(mf.get(k), f'{w}.{k}'))
    for k in MOC_KEYS:
        if md[k] < tops[k]:
            raise GateFileError(f'camera.c8b.moc_dich.{k} {md[k]!r}: below the largest {tops[k]!r} of the U manifests')
    return d


# ── reader ─────────────────────────────────────────────────────────────────────────────────────────────────────

def required_set(report_only: Iterable[str]) -> Tuple[str, ...]:
    ro = set(report_only)
    if ro - set(MASKABLE):
        raise GateFileError(f'report_only {sorted(ro)!r}: a subset of {MASKABLE}')
    return tuple(c for c in o1_gate.CONDS if c not in ro)


def mask_applies(platform: str, stage: str, campaign: str) -> bool:
    """Whether the report-only list of the level applies to a run (module docstring)."""
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

    def report_only(self, level: str) -> Tuple[str, ...]:
        """The report-only conditions of a gate level; () for a task that is not a gate level."""
        e = self.data['dieu_2b']['per_level'].get(level)
        return tuple(e['report_only']) if e is not None else ()

    def run_report_only(self, level: str, platform: str, stage: str, campaign: str) -> Tuple[str, ...]:
        return self.report_only(level) if mask_applies(platform, stage, campaign) else ()

    def required(self, level: str, platform: str, stage: str, campaign: str) -> Tuple[str, ...]:
        return required_set(self.run_report_only(level, platform, stage, campaign))


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
    the source {bo_lay_mau, car_pose (5 keys), tap_moc, manifests} and khoa_sha256."""
    c8b = gate.data['camera']['c8b']
    md = c8b['moc_dich']
    out = {k: float(md[k]) for k in MOC_KEYS}
    out.update(bo_lay_mau=md['nguon']['bo_lay_mau'], car_pose=car_pose_v2_of(md['nguon']['car_pose'], 'moc_dich'),
               tap_moc=md['nguon']['tap_moc'], khoa_sha256=c8b['khoa_sha256'])
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
    with open(path, 'wb') as f:
        f.write(raw)
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
    ref = lambda name: dict(duong_dan=f'FAKE/{name}', sha256=FAKE_SHA)              # noqa: E731
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
                              chuoi=[dict(run_id=rid, ket_qua='dat', loai_hong=[], kiem_gio_d1='dat',
                                          dem_truot_kiem_gio=False, file_cham=ref(f'{rid}_luoi.json'),
                                          file_gio=ref(f'{rid}_gio.json'), file_chung_cu=ref(f'{rid}_chung_cu.json'))]))
            runs[name].append(dict(run_id=rid, controller=ctrl, o=f'{name}_{ctrl}', h_s=u / 2.0, u_max_s=u / 2.0,
                                   u_t_s=u / 2.0 + u / 2.0, chung_cu=json.loads(json.dumps(ev))))
        t = fake_terms(u_p_m, u_psi_deg)
        per[name] = level_block(name, t['a_m'], t['u_p'], t['u_psi'], runs[name])
    cu = dict(loai=CHUAN_U_LOAI, le=CHUAN_U_LE, ho_so_c8=ref('ho_so_c8.json'),
              dieu_kien={k: dict(trang_thai='dat', bang_chung=ref(f'{k}.json')) for k in CHUAN_U_DK})
    clip = lambda i: dict(duong_dan=f'FAKE/clip_H{i}.json', sha256=format(i, 'x') * 64)  # noqa: E731
    md = {k: 0.0 for k in MOC_KEYS}
    md.update(U_moc_dong_xy_m=0.005, U_moc_tham_xy_m=0.005, U_moc_dong_psi_rad=0.004, U_moc_tham_psi_rad=0.004,
              nguon=dict(bo_lay_mau='FAKE_blm_v1', tap_moc=FAKE_SHA, manifests=[ref('u_moc_manifest.json')],
                         car_pose=dict(tag_offset=None, side_m=0.1445, h_m=0.20, kappa=0.0, phien_ban='FAKE')))
    cam = {'c8b': dict(moc_dich=md, khoa_sha256='f' * 64, eps_H=dict(mm=1.0, huong_deg=0.2),
                       mien_car_pose=dict(side_m=0.1445, phien_ban='FAKE', h_m=[0.19, 0.21], kappa=[0.0, 0.0]))}
    lv = list(levels)
    return dict(format=FORMAT, c8_dir='FAKE', written_by=dict(script='FAKE', commit='', dirty=True),
                levels=lv, muc_kha_thi=lv, muc_bo_theo_liam={}, quyet_liam={}, u_t_branch='bo_tick',
                lich_doi_chieu=LICH,
                bo_cham=dict(phien_ban='FAKE', bang_man_trap={k: dict(nhanh='mot_khung') for k in lv + [TASK_K]}),
                ke_hoach_c8=ref('ke_hoach_c8.json'), o_c8=cells, chuan_u=cu,
                tien_de_c8=dict(clip_h=dict(dau=[clip(1), clip(2)], giua=clip(3), cuoi=clip(4)),
                                file=ref('tien_de_c8.json')),
                runs_c8=runs, muc_loai_e_do={k: [] for k in lv},
                chung_cu_muc={k: dict(e_do={c: [f'FAKE_{k}_{c}'] for c in CONTROLLERS},
                                      g4={c: [f'FAKE_{k}_{c}'] for c in CONTROLLERS}) for k in lv},
                dieu_2b=dict(thresholds={c: o1_gate.THRESHOLDS[c] for c in MASKABLE}, per_level=per),
                camera=cam)


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


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='gate_c8: write the C8 plan file ke_hoach_c8_v1 (F17)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    kh = sub.add_parser('ke-hoach', help='the plan of the C8 cells from the locked run list of the session guide')
    kh.add_argument('--out', required=True, help='new file (never replaced), under the C8 folder')
    a = ap.parse_args(argv)
    try:
        with open(a.out, 'x', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(ke_hoach_c8(), indent=1, ensure_ascii=False) + '\n')
    except (OSError, GateFileError) as exc:
        print(f'gate_c8: {exc}', file=sys.stderr)
        return 1
    print('ke_hoach', a.out, sha256_file(a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
