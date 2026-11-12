"""
o1_gate.py: the tune stop gate G1-G5 of one O1 run, on the summary of an evaluator
(fwsim/o1_feasibility.evaluate_v7 for the figure 8, tools/o1_lanes.evaluate_k for K1, K2).

Source: CLAUDE_THESIS 00_OVERVIEW/2026-11-12_De_xuat_cong_dung_tune_bam_huong_van_toc_v3.md
(Liam 12/11/2026 evening, THESIS session local_0900b80e, "Duyệt cả (Recommended)"); code asked by
THESIS, Liam approved in the CODE session local_87282b70 ("làm ngay đi"), 12/11/2026; changes after
Codex round 1 (GV3-2, 3, 4, 7; Liam "Cho sửa 7 chỗ", same session).

  G1 completion        O1 3.4, unchanged: evaluator 'completed' true / false / inconclusive
  G2 lane              e_perp_max                         <= D = 0.10 m
  G3 schedule          e_trk_max = max |p(t) - p_cmd(t)|  <= D
  G4 body envelope     NOT computed here yet: its formula is under Codex round 3 (state 'chua_tinh')
  G5 heading           e_theta_cmd_max = max |wrap(psi - psi_cmd)|  <= 7.5 deg
                       (figure 8: evaluate_v7 'e_theta_cmd_max_deg'; K1, K2: the command heading
                       is the start heading, evaluate_k 'e_theta_start_max_deg')

Each of G2, G3, G5 has three states (S05 3.4), with U the bound of the measured maximum q (dossier v3
section 5.1; empirical bounds locked at C8). The true value lies in [q - U, q + U]:
  dat            q + U <= threshold   (every value of the interval meets "<= threshold": certain pass;
                                       the upper edge equal to the threshold passes, as "<=" does)
  khong_dat      q - U >  threshold   (every value fails)
  chua_ket_luan  otherwise, or q / U not finite, or U < 0
  U(G2) = U_p;  U(G3) = U_p + v_cmd_max * U_t;  U(G5) = U_psi + wz_cmd_max * U_t (in degrees)
  (GV3-4, rejected: Codex read "the interval holds the threshold" as making an upper edge equal to
  the threshold inconclusive; with a "<=" condition that edge is a certain pass. Rounding of q + U
  near the threshold, of order 1e-17, is below any measurement resolution and is not handled.)
G1: 'true' -> dat, 'false' -> khong_dat, 'inconclusive' (or anything else) -> chua_ket_luan.

Which conditions may decide (GV3-2): `decide` lists them. fwsim: all five. HIL: G1, G2 only, because
t = 0 there is the plant RUN step, not yet mapped to the ESP32 run_active edge (dossier 5.2), which
touches G3, G5 (schedule and command heading), not completion or the lane distance.
Outcomes (O1 3.5 order: one khong_dat -> khong_dat; else one chua_ket_luan -> chua_ket_luan; else dat):
  outcome         over G1, G2, G3, G5 (all computed), report only
  outcome_decide  over the computed conditions that may decide
  complete        G4 computed and all five may decide; only then can a run be 'dat' under the gate.
Lost runs (measurement / command path, O1 3.5) are decided outside this module.

U_t has no default (GV3-3): the caller passes the measured bound, or None when it is not known yet;
None makes G3, G5 chua_ket_luan. fwsim and HIL score the true plant state: U_p = U_psi = 0 there. On
fwsim the 50 Hz publisher is held for the 100 Hz firmware (0 or 10 ms per tick; about 5 ms, Claude's
estimate, not measured: 03_CODE/docs/2026-11-12_Xac_nhan_cong_v3_muc_a.md section 2).
Command maxima (GV3-7): the grid maximum is refined around its peak and raised by a relative margin
CMD_MARGIN, so the value used is an upper bound of the continuous maximum.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

GATE_VERSION = 'gate_v3_g1235'          # G4 not yet in; the version string changes when it is
D = 0.10                                # m, O1 3.9 (1); same value as o1_feasibility.D (a test checks)
THETA_MAX_DEG = 7.5                     # deg, dossier v3 section 2 (US-015 datasheet, author's choice)
STATES = ('dat', 'khong_dat', 'chua_ket_luan')
G4_STATE = 'chua_tinh'
CONDS = ('g1', 'g2', 'g3', 'g4', 'g5')
COMPUTED = ('g1', 'g2', 'g3', 'g5')     # G4 not yet
THRESHOLDS = {'g2': D, 'g3': D, 'g5': THETA_MAX_DEG}
DECIDE_ALL = CONDS
DECIDE_HIL = ('g1', 'g2')
CMD_MARGIN = 1e-6                       # relative margin on the refined command maxima (GV3-7)


def _finite(x) -> bool:
    if isinstance(x, bool):
        return False
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def tri_state(q, threshold: float, u) -> str:
    """S05 3.4 on one maximum q with its bound u >= 0 (module docstring)."""
    if not (_finite(q) and _finite(u)) or float(u) < 0.0:
        return 'chua_ket_luan'
    q, u = float(q), float(u)
    if q - u > threshold:
        return 'khong_dat'
    if q + u <= threshold:
        return 'dat'
    return 'chua_ket_luan'


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


def _cond(q, threshold, u):
    st = tri_state(q, threshold, u)
    qf = float(q) if _finite(q) else float('nan')
    return dict(q=qf, u=float(u) if _finite(u) else float('nan'), threshold=threshold,
                ratio=qf / threshold if _finite(q) else float('nan'), state=st)


def evaluate_outcomes(g: dict, decide: Iterable[str]) -> dict:
    """outcome, outcome_decide, complete, any_certain_fail of a gate dict with states g1..g5."""
    decide = [c for c in CONDS if c in set(decide)]
    computed = [c for c in CONDS if g[c]['state'] != G4_STATE]
    dec = [c for c in decide if c in computed]
    return dict(outcome=run_outcome(g[c]['state'] for c in COMPUTED),
                outcome_decide=run_outcome(g[c]['state'] for c in dec),
                complete=set(decide) == set(CONDS) and set(computed) == set(CONDS),
                any_certain_fail=any(g[c]['state'] == 'khong_dat' for c in dec))


def gate_v3(summary: dict, *, v_cmd_max: float, wz_cmd_max: float, u_t: Optional[float],
            theta_key: str = 'e_theta_cmd_max_deg', u_p: float = 0.0, u_psi_deg: float = 0.0,
            decide: Iterable[str] = DECIDE_ALL, note: str = '') -> dict:
    """G1, G2, G3, G5 of one run from an evaluator summary; G4 'chua_tinh'. Units: m, deg, s.
    u_t: bound of the time offset of the command start [s], or None = not known (G3, G5 inconclusive)."""
    decide = tuple(c for c in CONDS if c in set(decide))
    if not decide or set(decide) - set(CONDS):
        raise ValueError(f'decide: a non-empty subset of {CONDS}')
    if u_t is None:
        u3 = u5 = float('nan')
    else:
        u3 = u_p + v_cmd_max * u_t
        u5 = u_psi_deg + math.degrees(wz_cmd_max * u_t)
    g = dict(g1=dict(q=summary.get('completed'), state=g1_state(summary.get('completed'))),
             g2=_cond(summary.get('e_perp_max'), D, u_p),
             g3=_cond(summary.get('e_trk_max'), D, u3),
             g4=dict(state=G4_STATE),
             g5=_cond(summary.get(theta_key), THETA_MAX_DEG, u5))
    ratios = [g[k]['ratio'] for k in ('g2', 'g3', 'g5') if _finite(g[k]['ratio'])]
    return dict(version=GATE_VERSION, D=D, theta_max_deg=THETA_MAX_DEG, theta_key=theta_key,
                v_cmd_max=v_cmd_max, wz_cmd_max=wz_cmd_max, u_p=u_p, u_psi_deg=u_psi_deg, u_t=u_t,
                **g, **evaluate_outcomes(g, decide), decide=list(decide), conditions='G1,G2,G3,G5',
                max_ratio=max(ratios) if ratios else float('nan'),
                track_err=summary.get('status') == 'track_err', note=note)
