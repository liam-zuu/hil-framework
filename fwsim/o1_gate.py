"""
o1_gate.py: the tune stop gate G1-G5 of one O1 run, from the per-frame values of an evaluator
(fwsim/o1_gate_eval.py: evaluate_fig8 for the figure 8, evaluate_k for K1, K2).

Source: CLAUDE_THESIS 00_OVERVIEW/2026-11-12_De_xuat_cong_dung_tune_bam_huong_van_toc_v3.md (v3.2;
Liam 12/11/2026 evening, THESIS session local_0900b80e, "Duyệt cả (Recommended)") and O1 v11.9 sections
3.4, 3.5 (SC-1, SC3-1). Code: G1, G2, G3, G5 Liam 12/11 CODE local_87282b70 ("làm ngay đi", Codex Sol
4 rounds); G4, the per-frame bounds and the HIL edge mapping Liam 13/11 CODE local_db99d287 ("Làm rs4 +
g4 đi") and local_7f7b0ef4 ("Duyệt gộp luôn lo/hi").

  G1 completion        O1 3.4, unchanged: evaluator 'completed' true / false / inconclusive
  G2 lane              e_perp_max                         <= D = 0.10 m
  G3 schedule          e_trk_max = max |p(t) - p_cmd(t)|  <= D
  G4 body envelope     e_bao_max = max [|e_perp| + a_l |sin th_l| + b_l (|cos th_l| - 1)]  <= D
                       (th_l = wrap(psi - psi_lane(s_perp) - psi_0); evaluator: o1_gate_eval.py)
  G5 heading           e_theta_cmd_max = max |wrap(psi - psi_cmd)|  <= 7.5 deg
                       (K1, K2: the command heading is the start heading)

Three states per condition G2-G5 (O1 3.5, SC-1, SC3-1). Frame k gives the measured value q_k, a lower
bound lo_k and an upper bound hi_k of the quantity (here lo_k = q_k - U, hi_k = q_k + U with one U for
the run; U unknown -> lo_k, hi_k NaN). Over the run, each field is the maximum over the frames where it
is finite, taken separately: q = max q_k, lo = max lo_k, hi = max hi_k; n_undef = number of frames where
q_k, lo_k or hi_k is not finite. Then
  khong_dat      lo > threshold                          (one frame certainly above, whatever the others)
  chua_ket_luan  otherwise, n_undef > 0 or hi > threshold or no frame with a finite hi
  dat            n_undef = 0 and hi <= threshold         (an upper edge equal to the threshold passes, "<=")
Example (SC3-1, THESIS 13/11, Codex TX-4): one frame (0.12, 0.11, +inf), D = 0.10 -> khong_dat.
  U(G2) = U_p;  U(G3) = U_p + v_cmd_max U_t;  U(G5) = U_psi + wz_cmd_max U_t (degrees);
  U(G4) = U_p + r_c (U_psi + kappa_eff U_p): only U_p = U_psi = 0 is handled here (fwsim, HIL score the true
  plant state); the camera scorer (several projection solutions within 2 U_p, O1 3.4) is not this module.
G1: 'true' -> dat, 'false' -> khong_dat, 'inconclusive' (or anything else) -> chua_ket_luan.

Which conditions decide (`decide`): DECIDE_ALL on fwsim, the floor and on HIL when t = 0 is the ESP32
run_active edge mapped to the plant step (tools/hil_edge_map.py, RS-4); DECIDE_HIL when that mapping is
missing: t = 0 is then the plant RUN step, U_t is not known, so G3, G5 report only; G1, G2, G4 do not use
the schedule time and decide. Outcomes (O1 3.5 order: one khong_dat -> khong_dat; else one chua_ket_luan ->
chua_ket_luan; else dat):
  outcome         over G1-G5, report only
  outcome_decide  over the deciding conditions
  complete        all five decide; only then can a run be 'dat' under the gate.
Lost runs (measurement / command path, O1 3.5) are decided outside this module.

U_t has no default: the caller passes the bound, or None when it is not known. On fwsim the 50 Hz
publisher is held for the 100 Hz firmware (about 5 ms, an estimate): a property of the command path,
not added to U (dossier 5.2, CG4-6).
Command maxima (GV3-7): the grid maximum is refined around its peak and raised by a relative margin
CMD_MARGIN, so the value used is an upper bound of the continuous maximum.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

GATE_VERSION = 'gate_v3.2'              # G1-G5, per-frame lo / hi / n_undef (gate_v3_g1235 before 13/11)
D = 0.10                                # m, O1 3.9 (1); same value as o1_feasibility.D (a test checks)
THETA_MAX_DEG = 7.5                     # deg, dossier v3 section 2 (US-015 datasheet, author's choice)
BODY_HALF_LEN = 0.17                    # m, half of 340 mm along the car x axis (O1 3.4, section 6)
BODY_HALF_WID = 0.145                   # m, half of 290 mm
STATES = ('dat', 'khong_dat', 'chua_ket_luan')
CONDS = ('g1', 'g2', 'g3', 'g4', 'g5')
MEASURED = ('g2', 'g3', 'g4', 'g5')
THRESHOLDS = {'g2': D, 'g3': D, 'g4': D, 'g5': THETA_MAX_DEG}
FRAME_KEYS = {'g2': 'e_perp', 'g3': 'e_trk', 'g4': 'e_bao', 'g5': 'e_theta_cmd_deg'}
DECIDE_ALL = CONDS
DECIDE_HIL = ('g1', 'g2', 'g4')         # HIL without the run_active edge mapping (RS-4)
DECIDE_SETS = (DECIDE_ALL, DECIDE_HIL)
CMD_MARGIN = 1e-6                       # relative margin on the refined command maxima (GV3-7)


def _finite(x) -> bool:
    if isinstance(x, bool):
        return False
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def frame_bounds(q_k, u):
    """(lo_k, hi_k) = (q_k - u, q_k + u); u None / NaN / negative -> NaN bounds (not known)."""
    q = np.asarray(q_k, dtype=float)
    if not _finite(u) or float(u) < 0.0:
        nan = np.full(q.shape, np.nan)
        return nan, nan.copy()
    return q - float(u), q + float(u)


def cond_state(lo, hi, n_undef, threshold: float) -> str:
    """O1 3.5 on the stored numbers (SC3-1). lo, hi: maxima over the frames where they are finite,
    None / NaN when no frame has one; n_undef: frames with a value that is not finite."""
    if _finite(lo) and float(lo) > threshold:
        return 'khong_dat'
    if not isinstance(n_undef, int) or isinstance(n_undef, bool) or n_undef != 0:
        return 'chua_ket_luan'
    if not _finite(hi) or float(hi) > threshold:
        return 'chua_ket_luan'
    return 'dat'


def _fmax(a) -> float:
    a = a[np.isfinite(a)]
    return float(a.max()) if len(a) else float('nan')


def aggregate(q_k, lo_k, hi_k, threshold: float) -> dict:
    """One condition over the frames of a run (module docstring)."""
    q, lo, hi = (np.asarray(x, dtype=float).ravel() for x in (q_k, lo_k, hi_k))
    if not len(q) == len(lo) == len(hi):
        raise ValueError(f'q, lo, hi of different lengths ({len(q)}, {len(lo)}, {len(hi)})')
    fin = np.isfinite(q) & np.isfinite(lo) & np.isfinite(hi)
    if np.any(fin & ((lo > q) | (q > hi))):
        raise ValueError('a frame with lo > q or q > hi')
    out = dict(q=_fmax(q), lo=_fmax(lo), hi=_fmax(hi), n_undef=int(np.count_nonzero(~fin)), n_frames=int(len(q)),
               threshold=threshold)
    out['ratio'] = out['q'] / threshold if _finite(out['q']) else float('nan')
    out['state'] = cond_state(out['lo'], out['hi'], out['n_undef'], threshold)
    return out


def g1_state(completed) -> str:
    """Evaluator 'completed' -> state. A bool (O1 v5 evaluate) is taken as true / false."""
    if completed is True or completed == 'true':
        return 'dat'
    if completed is False or completed == 'false':
        return 'khong_dat'
    return 'chua_ket_luan'


def run_outcome(states: Iterable[str]) -> str:
    """O1 3.5 order over the given condition states."""
    states = list(states)
    if 'khong_dat' in states:
        return 'khong_dat'
    if 'chua_ket_luan' in states or not states:
        return 'chua_ket_luan'
    return 'dat'


def _refined_max(f, lo: float, hi: float, n: int = 20001) -> float:
    """Upper bound of max |f| on [lo, hi]: grid, golden-section refinement around each grid peak
    within one step, then the relative margin CMD_MARGIN."""
    x = np.linspace(lo, hi, n)
    y = np.abs(f(x))
    h = x[1] - x[0]
    best = float(y.max())
    for i in np.argsort(y)[-8:]:                         # a few largest grid points (symmetric peaks)
        a, b = max(lo, x[i] - h), min(hi, x[i] + h)
        g = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - g * (b - a), a + g * (b - a)
        for _ in range(80):
            if abs(f(c)) > abs(f(d)):
                b, d = d, c
                c = b - g * (b - a)
            else:
                a, c = c, d
                d = a + g * (b - a)
        best = max(best, float(abs(f(0.5 * (a + b)))))
    return best * (1.0 + CMD_MARGIN)


def fig8_cmd_max(sch):
    """(v_cmd_max [m/s], wz_cmd_max [rad/s]) of the figure-8 command: the schedule has
    0 <= tau_dot <= w0 and the command is 0 after T_cmd, so the maxima at tau_dot = w0 bound it.
    speed = |p'(tau)| w0, wz = d(psi_cmd)/d(tau) w0 (heading on the tangent)."""
    a = getattr(sch, 'A', 1.0)

    def speed(tau):
        return np.hypot(a * np.cos(tau), a * np.cos(2.0 * tau))

    def dpsi(tau):
        dx, dy = a * np.cos(tau), a * np.cos(2.0 * tau)
        ddx, ddy = -a * np.sin(tau), -2.0 * a * np.sin(2.0 * tau)
        return (dx * ddy - dy * ddx) / (dx * dx + dy * dy)

    two_pi = 2.0 * math.pi
    return _refined_max(speed, 0.0, two_pi) * sch.w0, _refined_max(dpsi, 0.0, two_pi) * sch.w0


def k_cmd_max(sch):
    """K1, K2 (O1 3.7): body-frame velocities, no rotation: wz_cmd = 0; speed = the schedule's v_max."""
    return float(sch.vmax), 0.0


def check_decide(decide) -> tuple:
    """`decide` as a tuple in CONDS order; it must be DECIDE_ALL or DECIDE_HIL (as a set)."""
    raw = list(decide)                                   # once: an iterator is consumed (GV3-R2-5)
    if len(set(raw)) != len(raw) or set(raw) not in [set(s) for s in DECIDE_SETS]:
        raise ValueError(f'decide {raw!r}: {DECIDE_ALL} or {DECIDE_HIL}')
    return tuple(c for c in CONDS if c in set(raw))


def evaluate_outcomes(g: dict, decide: Iterable[str]) -> dict:
    """outcome, outcome_decide, complete, any_certain_fail of a gate dict with states g1..g5."""
    dset = set(decide)                                   # once: an iterator is consumed (GV3-R2-5)
    dec = [c for c in CONDS if c in dset]
    return dict(outcome=run_outcome(g[c]['state'] for c in CONDS),
                outcome_decide=run_outcome(g[c]['state'] for c in dec),
                complete=dset == set(CONDS),
                any_certain_fail=any(g[c]['state'] == 'khong_dat' for c in dec))


def gate(frames: dict, completed, *, v_cmd_max: float, wz_cmd_max: float, u_t: Optional[float],
         u_p: float = 0.0, u_psi_deg: float = 0.0, decide: Iterable[str] = DECIDE_ALL, status: str = '',
         note: str = '') -> dict:
    """G1-G5 of one run. frames: per-frame arrays FRAME_KEYS (e_perp, e_trk, e_bao [m], e_theta_cmd_deg
    = |heading error| [deg]) on the scored frames; completed: the evaluator's G1 value.
    u_t: bound of the time offset of the command start [s], or None = not known (G3, G5 inconclusive)."""
    decide = check_decide(decide)
    if u_p != 0.0 or u_psi_deg != 0.0:
        raise ValueError('U_p, U_psi > 0 (camera): G4 needs the projection solutions within 2 U_p (O1 3.4); '
                         'not in this module')
    us = dict(g2=u_p, g4=0.0)
    if u_t is None:
        us.update(g3=float('nan'), g5=float('nan'))
    else:
        if not _finite(u_t) or float(u_t) < 0.0:
            raise ValueError(f'u_t {u_t!r}: None or a finite number >= 0')
        us.update(g3=u_p + v_cmd_max * float(u_t), g5=u_psi_deg + math.degrees(wz_cmd_max * float(u_t)))
    c1 = completed
    q1 = {'dat': 'true', 'khong_dat': 'false'}.get(g1_state(c1), 'inconclusive')   # always a string (GV3-R3-2)
    g = dict(g1=dict(q=q1, state=g1_state(c1)))
    for c in MEASURED:
        if FRAME_KEYS[c] not in frames:
            raise ValueError(f'frames: {FRAME_KEYS[c]!r} missing')
        q = np.asarray(frames[FRAME_KEYS[c]], dtype=float)
        lo, hi = frame_bounds(q, us[c])
        g[c] = dict(aggregate(q, lo, hi, THRESHOLDS[c]), u=us[c])
    ratios = [g[k]['ratio'] for k in MEASURED if _finite(g[k]['ratio'])]
    return dict(version=GATE_VERSION, D=D, theta_max_deg=THETA_MAX_DEG, v_cmd_max=v_cmd_max,
                wz_cmd_max=wz_cmd_max, u_p=u_p, u_psi_deg=u_psi_deg, u_t=u_t, **g,
                **evaluate_outcomes(g, decide), decide=list(decide), conditions='G1,G2,G3,G4,G5',
                max_ratio=max(ratios) if ratios else float('nan'), track_err=status == 'track_err', note=note)
