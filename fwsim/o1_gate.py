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
  dat            n_undef = 0, lo and hi finite, hi <= threshold   (an upper edge equal to the threshold passes)
Example (SC3-1, THESIS 13/11, Codex TX-4): one frame (0.12, 0.11, +inf), D = 0.10 -> khong_dat.
  U(G2) = U_p;  U(G3) = U_p + v_cmd_max U_t;  U(G5) = U_psi + wz_cmd_max U_t (degrees);
  U(G4) = U_p + r_c (U_psi + kappa_eff U_p): only U_p = U_psi = 0 is handled here (fwsim, HIL score the true
  plant state); the camera scorer (several projection solutions within 2 U_p, O1 3.4) is not this module.
G1: 'true' -> dat, 'false' -> khong_dat, 'inconclusive' (or anything else) -> chua_ket_luan.

Which conditions decide (`decide`): DECIDE_ALL on fwsim, the floor and on HIL when t = 0 is the ESP32
run_active edge mapped to the plant step (tools/hil_edge_map.py, RS-4); DECIDE_HIL = G1, G2 when that mapping
is missing (O1 v11.9 3.5: on HIL before RS-4 only G1, G2 decide): t = 0 is then the plant RUN step, U_t is not
known, G3, G5 are inconclusive and G4 reports only. Outcomes (O1 3.5 order: one khong_dat -> khong_dat; else one chua_ket_luan ->
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

O1 v11.33 (THESIS spec 06_SPECS/2026-11-17_De_xuat_O1_G5_F08_muc_do_duoc_v4.md, Liam 17/11 THESIS "Duyệt, rà vòng 3
bản khóa (Recommended)"; CODE_GATE design 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md, Liam 17/11 at CODE_GATE
"Duyệt, gửi CODE luôn (Recommended)"), version gate_v3.3, only when the caller asks for it (fwsim, S03, S04 call
gate as before and get the gate_v3.2 dict, number for number):
  per-frame bounds   U_k(G3) = U_p + U_t V_k, U_k(G5) = U_psi + U_t W_k (O1 lines 315, 317; OG-12, OG-14): V_k, W_k
                     = sup |dp_cmd/dt|, |dpsi_cmd/dt| over [t_k - U_t, t_k + U_t] (cmd_rate_frames; the reference
                     holds its end values outside the command, OG2-2; heading rate on the continuous branch,
                     OG2-9); lo, hi stay the maxima of lo_k, hi_k taken apart (O1 line 301). U = 0: the same numbers.
  report-only        the conditions of O1 3.9 rule 2b at the run's level, from the C8 gate file (tools/gate_c8.py;
                     never from the run record, OG2-7): still computed and stored; required = CONDS less them;
                     decide = required within DECIDE_ALL / DECIDE_HIL; complete = decide is required (O1 line 321);
                     outcome_decide, any_certain_fail over decide only (so a report-only condition never triggers
                     the violation-before-a-measurement-fault exception, OG2-3).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

GATE_VERSION = 'gate_v3.2'              # G1-G5, per-frame lo / hi / n_undef (gate_v3_g1235 before 13/11)
GATE_VERSION_V1133 = 'gate_v3.3'        # O1 v11.33: per-frame U of G3, G5 and the report-only conditions (CODE_GATE)
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
DECIDE_HIL = ('g1', 'g2')               # HIL without the run_active edge mapping (RS-4; O1 3.5)
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
    """(lo_k, hi_k) = (q_k - u, q_k + u); u None / NaN / negative -> NaN bounds (not known).
    u may be an array, one bound per frame (O1 v11.33: G3, G5 per frame): a frame whose u is not finite gets NaN
    bounds; a negative u is an error."""
    q = np.asarray(q_k, dtype=float)
    if isinstance(u, np.ndarray):
        uk = np.asarray(u, dtype=float)
        if uk.shape != q.shape:
            raise ValueError(f'per-frame u of shape {uk.shape}, frames {q.shape}')
        if np.any(uk[np.isfinite(uk)] < 0.0):
            raise ValueError('per-frame u below 0')
        ok = np.isfinite(uk)
        lo, hi = np.full(q.shape, np.nan), np.full(q.shape, np.nan)
        lo[ok], hi[ok] = q[ok] - uk[ok], q[ok] + uk[ok]
        return lo, hi
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
    if not (_finite(lo) and _finite(hi)) or float(hi) > threshold:     # no 'dat' without both bounds (G32-1)
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


RATE_STEP = 1e-4                        # s, grid of the command rates for cmd_rate_frames


def cmd_rate_frames(f, t_cmd: float, t, u_t: float, turn: bool = True, v_cap: Optional[float] = None,
                    wz_cap: Optional[float] = None):
    """(V_k, W_k) per frame (O1 v11.33 lines 315, 317; OG-12, OG2-2, OG2-9): V_k = sup |dp_cmd/dt| and
    W_k = sup |dpsi_cmd/dt| [rad/s] over [t_k - u_t, t_k + u_t].
    f(t) -> (vx, vy, wz): the body-frame command of the task (rpi5/o1_tasks.task; figure 8: Schedule.setpoint,
    heading on the tangent, wz = d(psi_cmd)/dt from the analytic tangent, i.e. on the continuous branch of the
    heading, never a difference of wrapped angles). |dp_cmd/dt| = hypot(vx, vy) (the body speed: the frame
    rotation does not change it). turn False (K1, K2: G5 against the start heading, O1 3.7): W_k = 0.
    Outside [0, t_cmd] the reference holds its end values, so both rates are 0 there; a window wholly outside
    gives 0. The sup is bounded from above on a grid of RATE_STEP: the largest grid value on the window widened
    by one step, plus the largest grid difference next to it (a Lipschitz allowance of twice the half step), times
    1 + CMD_MARGIN; then capped by v_cap / wz_cap (the peak of the schedule, always an upper bound) when given."""
    t = np.asarray(t, dtype=float)
    if not _finite(u_t) or float(u_t) < 0.0:
        raise ValueError(f'u_t {u_t!r}: a finite number >= 0')
    u = float(u_t)
    n = max(2, int(math.ceil(float(t_cmd) / RATE_STEP)) + 1)
    tt = np.linspace(0.0, float(t_cmd), n)
    h = tt[1] - tt[0]
    sp = np.empty(n)
    wz = np.zeros(n)
    for i, x in enumerate(tt):
        c = f(float(x))
        sp[i] = math.hypot(float(c[0]), float(c[1]))
        if turn:
            wz[i] = abs(float(c[2]))
    dsp = np.r_[np.abs(np.diff(sp)), 0.0]
    dwz = np.r_[np.abs(np.diff(wz)), 0.0]

    def sup(y, dy, a, b):
        if b < 0.0 or a > tt[-1]:
            return 0.0
        i0 = max(0, int(math.floor(max(a, 0.0) / h)) - 1)
        i1 = min(n - 1, int(math.ceil(min(b, tt[-1]) / h)) + 1)
        return (float(y[i0:i1 + 1].max()) + float(dy[max(0, i0 - 1):i1 + 1].max())) * (1.0 + CMD_MARGIN)
    v_k = np.array([sup(sp, dsp, tk - u, tk + u) for tk in t])
    w_k = np.array([sup(wz, dwz, tk - u, tk + u) for tk in t]) if turn else np.zeros(len(t))
    if v_cap is not None:
        v_k = np.minimum(v_k, float(v_cap))
    if wz_cap is not None:
        w_k = np.minimum(w_k, float(wz_cap))
    return v_k, w_k


MASKABLE = ('g2', 'g3', 'g5')           # O1 v11.33 3.9 rule 2b: conditions that may be report only (G1, G4 never)


def required_of(report_only: Iterable[str]) -> tuple:
    """The conditions that must decide (CONDS order) when `report_only` are report only at the run's level."""
    ro = list(report_only)
    if len(set(ro)) != len(ro) or set(ro) - set(MASKABLE):
        raise ValueError(f'report_only {ro!r}: distinct conditions of {MASKABLE}')
    return tuple(c for c in CONDS if c not in set(ro))


def check_decide(decide) -> tuple:
    """`decide` as a tuple in CONDS order; it must be DECIDE_ALL or DECIDE_HIL (as a set)."""
    raw = list(decide)                                   # once: an iterator is consumed (GV3-R2-5)
    if len(set(raw)) != len(raw) or set(raw) not in [set(s) for s in DECIDE_SETS]:
        raise ValueError(f'decide {raw!r}: {DECIDE_ALL} or {DECIDE_HIL}')
    return tuple(c for c in CONDS if c in set(raw))


def check_decide_masked(decide, required) -> tuple:
    """O1 v11.33 (OG2-6, OG2-7): `decide` must be `required` intersected with DECIDE_ALL or DECIDE_HIL. Returns
    it in CONDS order."""
    raw, req = list(decide), set(required)
    allowed = [set(s) & req for s in DECIDE_SETS]
    if len(set(raw)) != len(raw) or set(raw) not in allowed:
        raise ValueError(f'decide {raw!r} with required {sorted(req)!r}: one of {[sorted(a) for a in allowed]}')
    return tuple(c for c in CONDS if c in set(raw))


def evaluate_outcomes(g: dict, decide: Iterable[str], required: Optional[Iterable[str]] = None) -> dict:
    """outcome, outcome_decide, complete, any_certain_fail of a gate dict with states g1..g5. complete: every
    required condition decides (O1 v11.33 line 321; required None = all five, as gate_v3.2)."""
    dset = set(decide)                                   # once: an iterator is consumed (GV3-R2-5)
    dec = [c for c in CONDS if c in dset]
    req = set(CONDS) if required is None else set(required)
    return dict(outcome=run_outcome(g[c]['state'] for c in CONDS),
                outcome_decide=run_outcome(g[c]['state'] for c in dec),
                complete=dset == req,
                any_certain_fail=any(g[c]['state'] == 'khong_dat' for c in dec))


G1_FIELD_KEYS = ('t_done', 't_deadline', 't1', 'prog_before_t1', 'cond25_before_deadline')


def g1_state_from_fields(f: dict) -> str:
    """G1 from its B81-3 fields, one way of using the frames (fwsim, HIL; O1 3.5 last block), the rule of
    tools/rq3_log.gate_g1_state for one way."""
    done = f['t_done'][0] is not None
    if f['t1'] is None or f['prog_before_t1']:
        return 'dat' if done else 'khong_dat'
    return 'chua_ket_luan' if f['cond25_before_deadline'] else 'khong_dat'


def check_g1_fields(f: dict, state: str) -> None:
    """G1F-1 (Codex round 1): the B81-3 G1 fields of one way of using the frames, checked as
    tools/rq3_log.gate_g1_fields does for fwsim / HIL (not imported: tools/ is not in the S04 blob lock), plus
    t_done given exactly when G1 is 'dat' (state from completed). ValueError on any failure."""
    if not isinstance(f, dict) or set(f) != set(G1_FIELD_KEYS):
        raise ValueError(f'g1_fields {sorted(f) if isinstance(f, dict) else f!r}: the keys {G1_FIELD_KEYS}')
    td = f['t_done']
    if not isinstance(td, list) or len(td) != 1:
        raise ValueError(f'g1_fields t_done {td!r}: a list of one time (one way of using the frames)')

    def time(x, what, null_ok=True):
        if x is None and null_ok:
            return None
        # G1F2-1: a number of type int / float (not bool, not a numeric string), as rq3_log._time
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or x < 0:   # G1F3-1
            raise ValueError(f'g1_fields {what} {x!r}: {"null or " if null_ok else ""}a finite time >= 0')
        return float(x)
    t_dl = time(f['t_deadline'], 't_deadline', null_ok=False)
    t_done = time(td[0], 't_done')
    if t_done is not None and not t_done < t_dl:
        raise ValueError(f'g1_fields t_done {t_done!r} not before t_deadline {t_dl!r} (O1 3.4 condition 4)')
    t1 = time(f['t1'], 't1')
    for k in ('prog_before_t1', 'cond25_before_deadline'):
        if f[k] is not None and not isinstance(f[k], bool):
            raise ValueError(f'g1_fields {k} {f[k]!r}: null or a boolean')
    prog, c25 = f['prog_before_t1'], f['cond25_before_deadline']
    if t1 is None and (prog is not None or c25 is not None):
        raise ValueError('g1_fields: prog_before_t1 / cond25_before_deadline without t1')
    if t1 is not None and (prog is None or (not prog and c25 is None)):
        raise ValueError('g1_fields: with t1, prog_before_t1, and cond25_before_deadline when not prog, are needed')
    if (t_done is not None) != (state == 'dat'):
        raise ValueError(f'g1_fields t_done {t_done!r} with G1 {state!r}: a time only for a completed run')


def gate(frames: dict, completed, *, v_cmd_max: float, wz_cmd_max: float, u_t: Optional[float],
         u_p: float = 0.0, u_psi_deg: float = 0.0, decide: Iterable[str] = DECIDE_ALL, status: str = '',
         note: str = '', g1_fields: Optional[dict] = None, v_k=None, w_k=None,
         report_only: Optional[Iterable[str]] = None, level: Optional[str] = None,
         gate_file_sha256: Optional[str] = None) -> dict:
    """G1-G5 of one run. frames: per-frame arrays FRAME_KEYS (e_perp, e_trk, e_bao [m], e_theta_cmd_deg
    = |heading error| [deg]) on the scored frames; completed: the evaluator's G1 value.
    u_t: bound of the time offset of the command start [s], or None = not known (G3, G5 inconclusive).
    g1_fields (B81-3, one way of using the frames: o1_gate_eval.g1_fields_v7): when given, the g1 block also holds
    t_done, t_deadline, t1, prog_before_t1, cond25_before_deadline, G1 recomputed from them must equal the state
    from completed (else ValueError), and g2..g5 hold the common domain 'chung' {q, lo, hi, n_undef}, equal to the
    union one (one frame domain). None (default): the dict is as before.
    O1 v11.33 (design 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md sections 3, 4), version GATE_VERSION_V1133
    when any of the following is given, else the dict is exactly the gate_v3.2 one:
      v_k, w_k     per-frame command rates (cmd_rate_frames, [m/s], [rad/s]): U_k(G3) = U_p + U_t V_k,
                   U_k(G5) = U_psi + deg(U_t W_k); lo, hi stay the maxima of lo_k, hi_k taken apart (O1 line 301).
                   Each condition gets u = max_k U_k and the dict u_mode 'khung' ('hang' without them).
      report_only  the report-only conditions at the run's level from the C8 gate file (tools/gate_c8.py), with
                   level and gate_file_sha256 (OG2-7: never taken from the run record): required = CONDS minus them;
                   decide = required intersected with `decide`; complete = decide is required (O1 line 321)."""
    decide = check_decide(decide)
    v1133 = v_k is not None or w_k is not None or report_only is not None
    if (v_k is None) != (w_k is None):
        raise ValueError('v_k and w_k go together')
    if report_only is None:
        if level is not None or gate_file_sha256 is not None:
            raise ValueError('level / gate_file_sha256 without report_only (the gate file gives all three)')
        required = CONDS
    else:
        required = required_of(report_only)
        if not isinstance(level, str) or not level:
            raise ValueError(f'level {level!r}: the task of the run (a gate level or another task)')
        if (not isinstance(gate_file_sha256, str) or len(gate_file_sha256) != 64
                or any(ch not in '0123456789abcdef' for ch in gate_file_sha256)):
            raise ValueError(f'gate_file_sha256 {gate_file_sha256!r}: 64 lowercase hex digits')
    decide = tuple(c for c in decide if c in required)
    if u_p != 0.0 or u_psi_deg != 0.0:
        raise ValueError('U_p, U_psi > 0 (camera): G4 needs the projection solutions within 2 U_p (O1 3.4); '
                         'not in this module')
    us = dict(g2=u_p, g4=0.0)
    if u_t is None:
        us.update(g3=float('nan'), g5=float('nan'))
    else:
        if not _finite(u_t) or float(u_t) < 0.0:
            raise ValueError(f'u_t {u_t!r}: None or a finite number >= 0')
        if v_k is None:
            us.update(g3=u_p + v_cmd_max * float(u_t), g5=u_psi_deg + math.degrees(wz_cmd_max * float(u_t)))
        else:
            n = len(np.asarray(frames.get(FRAME_KEYS['g3'], [])))
            vk, wk = (np.asarray(x, dtype=float).ravel() for x in (v_k, w_k))
            if len(vk) != n or len(wk) != n or not (np.all(np.isfinite(vk)) and np.all(np.isfinite(wk)))                     or np.any(vk < 0.0) or np.any(wk < 0.0):
                raise ValueError(f'v_k, w_k: {n} finite numbers >= 0 each (one per frame)')
            us.update(g3=u_p + vk * float(u_t), g5=u_psi_deg + np.degrees(wk * float(u_t)))
    c1 = completed
    q1 = {'dat': 'true', 'khong_dat': 'false'}.get(g1_state(c1), 'inconclusive')   # always a string (GV3-R3-2)
    g = dict(g1=dict(q=q1, state=g1_state(c1)))
    for c in MEASURED:
        if FRAME_KEYS[c] not in frames:
            raise ValueError(f'frames: {FRAME_KEYS[c]!r} missing')
        q = np.asarray(frames[FRAME_KEYS[c]], dtype=float)
        lo, hi = frame_bounds(q, us[c])
        u_rep = (float(np.max(us[c])) if len(us[c]) else 0.0) if isinstance(us[c], np.ndarray) else us[c]
        g[c] = dict(aggregate(q, lo, hi, THRESHOLDS[c]), u=u_rep)
    if g1_fields is not None:
        if set(g1_fields) != set(G1_FIELD_KEYS) or not (isinstance(g1_fields['t_done'], list)
                                                       and len(g1_fields['t_done']) == 1):
            raise ValueError(f'g1_fields {sorted(g1_fields)!r}: {G1_FIELD_KEYS}, t_done a list of one time')
        st = g1_state_from_fields(g1_fields)
        if st != g['g1']['state']:
            raise ValueError(f'B81-3: G1 from t_done, t1 and the flags is {st!r}, from completed {completed!r} it is '
                             f'{g["g1"]["state"]!r}')
        check_g1_fields(g1_fields, g['g1']['state'])                       # G1F-1: the rq3_log field checks
        g['g1'].update({k: (list(v) if k == 't_done' else v) for k, v in g1_fields.items()})
        for c in MEASURED:
            g[c]['chung'] = {k: g[c][k] for k in ('q', 'lo', 'hi', 'n_undef')}
    ratios = [g[k]['ratio'] for k in MEASURED if _finite(g[k]['ratio'])]
    out = dict(version=GATE_VERSION_V1133 if v1133 else GATE_VERSION, D=D, theta_max_deg=THETA_MAX_DEG,
               v_cmd_max=v_cmd_max, wz_cmd_max=wz_cmd_max, u_p=u_p, u_psi_deg=u_psi_deg, u_t=u_t, **g,
               **evaluate_outcomes(g, decide, required), decide=list(decide), conditions='G1,G2,G3,G4,G5',
               max_ratio=max(ratios) if ratios else float('nan'), track_err=status == 'track_err', note=note)
    if v1133:
        out.update(u_mode='khung' if v_k is not None else 'hang', level=level, gate_file_sha256=gate_file_sha256,
                   report_only=[c for c in MASKABLE if c not in required], required=list(required))
    return out
