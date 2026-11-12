"""
o1_gate.py: the tune stop gate G1-G5 of one O1 run, on the summary of an evaluator
(fwsim/o1_feasibility.evaluate_v7 for the figure 8, tools/o1_lanes.evaluate_k for K1, K2).

Source: CLAUDE_THESIS 00_OVERVIEW/2026-11-12_De_xuat_cong_dung_tune_bam_huong_van_toc_v3.md
(Liam 12/11/2026 evening, THESIS session local_0900b80e, "Duyệt cả (Recommended)"); code asked by
THESIS, Liam approved in the CODE session local_87282b70 ("làm ngay đi"), 12/11/2026.

  G1 completion        O1 3.4, unchanged: evaluator 'completed' true / false / inconclusive
  G2 lane              e_perp_max                         <= D = 0.10 m
  G3 schedule          e_trk_max = max |p(t) - p_cmd(t)|  <= D
  G4 body envelope     NOT computed here yet: its formula is under Codex round 3 (state 'chua_tinh')
  G5 heading           e_theta_cmd_max = max |wrap(psi - psi_cmd)|  <= 7.5 deg
                       (figure 8: evaluate_v7 'e_theta_cmd_max_deg'; K1, K2: the command heading
                       is the start heading, evaluate_k 'e_theta_start_max_deg')

Each of G2, G3, G5 has three states (S05 3.4: the uncertainty interval against the threshold), with
U the bound of the measured value (dossier v3 section 5.1; empirical bounds locked at C8):
  dat            q + U <= threshold
  khong_dat      q - U >  threshold
  chua_ket_luan  otherwise (the interval holds the threshold), or q / U not finite
  U(G2) = U_p;  U(G3) = U_p + v_cmd_max * U_t;  U(G5) = U_psi + wz_cmd_max * U_t (in degrees)
G1: 'true' -> dat, 'false' -> khong_dat, 'inconclusive' (or anything else) -> chua_ket_luan.
Run outcome over the conditions computed (O1 3.5 order, unchanged): one khong_dat -> khong_dat;
else one chua_ket_luan -> chua_ket_luan; else dat. 'complete' is False while G4 is not computed:
the outcome is then the one over G1, G2, G3, G5 only and must not be read as the gate.
Lost runs (measurement / command path, O1 3.5) are decided outside this module.

fwsim and HIL score the true plant state: U_p = U_psi = 0 there. U_t: on fwsim the 50 Hz publisher
is held for the 100 Hz firmware (0 or 10 ms per tick, about 5 ms; Claude's estimate, not measured,
03_CODE/docs/2026-11-12_Xac_nhan_cong_v3_muc_a.md section 2); on HIL t = 0 is the plant RUN step,
not yet mapped to the ESP32 run_active edge (dossier 5.2): HIL G3, G5 are reported, not used to
decide (usable_for_decision False) until that mapping exists. The caller passes U and that flag.
"""
from __future__ import annotations

import math

import numpy as np

GATE_VERSION = 'gate_v3_g1235'          # G4 not yet in; the version string changes when it is
D = 0.10                                # m, O1 3.9 (1); same value as o1_feasibility.D (a test checks)
THETA_MAX_DEG = 7.5                     # deg, dossier v3 section 2 (US-015 datasheet, author's choice)
STATES = ('dat', 'khong_dat', 'chua_ket_luan')
G4_STATE = 'chua_tinh'


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def tri_state(q, threshold: float, u: float = 0.0) -> str:
    """S05 3.4 on one maximum q with its bound u >= 0."""
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


def run_outcome(states) -> str:
    """O1 3.5 order over the given condition states."""
    states = list(states)
    if 'khong_dat' in states:
        return 'khong_dat'
    if 'chua_ket_luan' in states or not states:
        return 'chua_ket_luan'
    return 'dat'


def fig8_cmd_max(sch, n: int = 20001):
    """(v_cmd_max [m/s], wz_cmd_max [rad/s]) of the figure-8 command at full schedule speed w0:
    speed = |p'(tau)| w0, wz = d(psi_cmd)/d(tau) w0 (heading on the tangent)."""
    a = 1.0 if not hasattr(sch, 'A') else sch.A
    tau = np.linspace(0.0, 2.0 * math.pi, n)
    dx, dy = a * np.cos(tau), a * np.cos(2.0 * tau)
    ddx, ddy = -a * np.sin(tau), -2.0 * a * np.sin(2.0 * tau)
    sp2 = dx * dx + dy * dy
    dpsi = (dx * ddy - dy * ddx) / sp2
    return float(np.sqrt(sp2).max() * sch.w0), float(np.abs(dpsi).max() * sch.w0)


def k_cmd_max(sch):
    """K1, K2 (O1 3.7): body-frame velocities, no rotation: wz_cmd = 0."""
    return float(sch.vmax), 0.0


def _cond(q, threshold, u):
    st = tri_state(q, threshold, u)
    qf = float(q) if _finite(q) else float('nan')
    return dict(q=qf, u=float(u) if _finite(u) else float('nan'), threshold=threshold,
                ratio=qf / threshold if _finite(q) else float('nan'), state=st)


def gate_v3(summary: dict, *, v_cmd_max: float, wz_cmd_max: float, theta_key: str = 'e_theta_cmd_max_deg',
            u_p: float = 0.0, u_psi_deg: float = 0.0, u_t: float = 0.0, usable_for_decision: bool = True,
            note: str = '') -> dict:
    """G1, G2, G3, G5 of one run from an evaluator summary; G4 'chua_tinh'. Units: m, deg, s."""
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
                **g, outcome=run_outcome(g[k]['state'] for k in ('g1', 'g2', 'g3', 'g5')),
                complete=False, conditions='G1,G2,G3,G5',
                any_certain_fail=any(g[k]['state'] == 'khong_dat' for k in ('g1', 'g2', 'g3', 'g5')),
                max_ratio=max(ratios) if ratios else float('nan'),
                track_err=summary.get('status') == 'track_err',
                usable_for_decision=bool(usable_for_decision), note=note)
