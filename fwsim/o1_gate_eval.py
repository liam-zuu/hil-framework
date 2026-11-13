"""
o1_gate_eval.py: the evaluator of the tune stop gate v3.2 (G1-G5): the O1 run scored as before, plus the
per-frame values of G2-G5 and the body envelope G4 (O1 v11.9 section 3.4 'e_bao,max'; dossier
00_OVERVIEW/2026-11-12_De_xuat_cong_dung_tune_bam_huong_van_toc_v3.md section 2).
Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi": new evaluator, evaluate_k, tie set and the outgoing
side at a K2 corner, THESIS SC3-4, SC-15); after Codex Sol round 1 (G32-7, G32-8; Liam CODE local_7f7b0ef4
"Duyệt cả, vòng 2", L97).

evaluate_v7 (fwsim/o1_feasibility.py) and evaluate_k (tools/o1_lanes.py) are called UNCHANGED: their
summary and rows stay the numbers they gave before (o1_feasibility.py is in the S04 SIM_SOURCES lock). This
module only reads their rows and adds:
  e_bao per frame (G4)  e_perp + a_l |sin th_l| + b_l (|cos th_l| - 1), th_l = wrap(psi - psi_lane(s_perp) - psi_0)
      e_perp: the row's (the value G2 uses). s_perp: the lane point that gives e_perp, the nearest point of the
      TWO-SIDED window [s_prev - D_k, s_prev + D_k] of O1 3.3 (rebuilt from the rows: s_prev = s_car of the
      frame before, D_k = 1.5 v_max dt + delta, as evaluate_v7 / evaluate_k do; the rebuilt grid distance is
      checked against the row's e_perp).
      Tie set (O1 3.4, U_p = 0 on fwsim and HIL): the projection solutions whose distance equals the smallest
      up to rounding; G4 of the frame is the LARGEST value over them, and lo_k = hi_k = that value (O1 3.4).
        K1, K2 (straight sides): solved exactly, side by side: on each side, the point of its part inside the
          window nearest to the centre (interior foot, a corner, or a window end); ties within TIE_NUM.
          A point AT a corner takes the OUTGOING side (O1 3.4 'tại đỉnh và lúc đứng ở góc dùng đoạn ra');
          the closing corner of K2 (end = start) takes the first side as outgoing (G32-8); the open end of K1
          keeps its side.
        figure 8 (smooth, no corner): the local minima of the distance on the 15 um grid of Path8, ties within
          TIE_GRID (the grid's rounding of a true double minimum is below 1e-9 m).
      Nominal body pose: figure 8 psi_0 = 0, a_l = 0.17, b_l = 0.145 (heading on the tangent); K1, K2: the
      body keeps the start heading, so psi_0 = wrap(0 - psi_lane) and (a_l, b_l) = (0.17, 0.145) on a side
      along the car x axis, (0.145, 0.17) on a side along y (O1 3.4: K2 'vx' sides as the figure 8, 'vy'
      sides as K1).
  e_trk, e_perp per frame        from the rows
  e_theta_cmd_deg per frame      |e_theta_cmd| (figure 8) or |e_theta_start| (K1, K2), in degrees
  summary: e_bao_max, e_bao_phys_max = max(0, e_bao_max) (the physical overhang, O1 3.4), g4_tie_frames
  (frames where more than one projection solution was found).
G4 is the signed excess over the nominal body strip: on a straight lane a_l |sin| + b_l |cos| is the half
width of the turned rectangle along the lane normal; in tight curves the local frame is an approximation.

  out, rows, frames = evaluate_fig8(log, path, sch)   /   evaluate_k(log, task)
  frames: dict of arrays per scored frame (o1_gate.FRAME_KEYS) for o1_gate.gate.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..', 'tools'), os.path.join(HERE, '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import o1_feasibility as O                                              # noqa: E402
import o1_lanes as L                                                    # noqa: E402
import o1_gate as G                                                     # noqa: E402

EVAL_VERSION = 'o1_gate_eval_v2'
TIE_GRID = 1e-9                         # m, figure 8 grid (module docstring)
TIE_NUM = 1e-12                         # m, exact K solutions: equal up to rounding
PERP_CHECK = 1e-9                       # m, rebuilt nearest distance vs the row's e_perp


def _wrap(a):
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def window_index(lane, s_lo, s_hi):
    """Index range [i0, i1) of the lane samples in [s_lo, s_hi], as Path8.nearest / Lane.nearest."""
    i0 = int(np.searchsorted(lane.s, max(s_lo, 0.0), side='left'))
    i1 = int(np.searchsorted(lane.s, min(s_hi, lane.S), side='right'))
    return i0, max(i1, i0 + 1)


def grid_projections(lane, px, py, s_lo, s_hi, tie_tol=TIE_GRID):
    """Grid solutions: indices of the local minima of the sampled distance within tie_tol of the smallest
    (a run of equal samples counts once, at its first index, which keeps the argmin of Path8.nearest).
    Returns (indices, distances, smallest distance)."""
    i0, i1 = window_index(lane, s_lo, s_hi)
    d = np.hypot(lane.x[i0:i1] - px, lane.y[i0:i1] - py)
    dmin = float(d.min())
    left = np.r_[np.inf, d[:-1]]
    right = np.r_[d[1:], np.inf]
    idx = np.nonzero((d <= left) & (d <= right) & (d <= dmin + tie_tol))[0]
    idx = idx[np.r_[True, np.diff(idx) > 1]]
    return i0 + idx, d[idx], dmin


class Polyline:
    """Exact sides of a K lane (corners of o1_lanes.k_corners): cumulative arc length and headings."""

    def __init__(self, corners):
        self.p = np.asarray(corners, dtype=float)
        seg = np.diff(self.p, axis=0)
        self.len = np.hypot(seg[:, 0], seg[:, 1])
        self.u = seg / self.len[:, None]
        self.head = np.arctan2(seg[:, 1], seg[:, 0])
        self.cum = np.r_[0.0, np.cumsum(self.len)]
        self.closed = bool(np.hypot(*(self.p[0] - self.p[-1])) < 1e-12)

    def head_at(self, i, s):
        """Heading at arc length s found on side i: a corner takes its outgoing side; the closing corner of
        a closed lane takes side 0; the open end keeps its side."""
        n = len(self.len)
        for j in range(n + 1):
            if abs(s - self.cum[j]) <= 1e-12:
                if j < n:
                    return float(self.head[j])
                return float(self.head[0] if self.closed else self.head[n - 1])
        return float(self.head[i])

    def solutions(self, px, py, s_lo, s_hi, tie_tol=TIE_NUM):
        """Exact projection solutions of (px, py) on the part of the lane in [s_lo, s_hi]:
        [(s, distance, heading)] within tie_tol of the smallest distance, one per distinct point."""
        lo, hi = max(s_lo, 0.0), min(s_hi, float(self.cum[-1]))
        c = []
        for i in range(len(self.len)):
            u_lo, u_hi = max(lo, self.cum[i]) - self.cum[i], min(hi, self.cum[i + 1]) - self.cum[i]
            if u_lo > u_hi + 1e-15:
                continue
            u = float(np.clip(np.dot(np.array([px, py]) - self.p[i], self.u[i]), u_lo, u_hi))
            q = self.p[i] + u * self.u[i]
            s = float(self.cum[i] + u)
            c.append((s, float(np.hypot(px - q[0], py - q[1])), self.head_at(i, s)))
        dmin = min(d for _, d, _ in c)
        out = []
        for s, d, h in sorted(c):
            if d <= dmin + tie_tol and not any(abs(s - s2) <= 1e-12 for s2, _, _ in out):
                out.append((s, d, h))
        return out


def fig8_nominal(head):
    """Figure 8: heading on the tangent, 340 mm along the lane."""
    return 0.0, G.BODY_HALF_LEN, G.BODY_HALF_WID


def k_nominal(head):
    """K1, K2: the body keeps heading 0 (start frame); psi_0 = wrap(0 - head); a_l along the lane."""
    psi0 = float(_wrap(-head))
    if abs(math.cos(psi0)) >= abs(math.sin(psi0)):
        return psi0, G.BODY_HALF_LEN, G.BODY_HALF_WID
    return psi0, G.BODY_HALF_WID, G.BODY_HALF_LEN


def g4_value(e, psi, head, psi0, a_l, b_l):
    th = float(_wrap(psi - head - psi0))
    return e + a_l * abs(math.sin(th)) + b_l * (abs(math.cos(th)) - 1.0)


def g4_frames(rows, grid_lane, vmax, heads_of, nominal):
    """(e_bao per frame, number of projection solutions per frame) on the rows of evaluate_v7 / evaluate_k.
    heads_of(px, py, s_lo, s_hi) -> headings of the projection solutions in the window."""
    e_bao = np.empty(len(rows))
    n_sol = np.empty(len(rows), dtype=int)
    s_prev, t_prev = 0.0, 0.0
    for k, r in enumerate(rows):
        dk = 1.5 * vmax * (r['t'] - t_prev) + O.DELTA
        lo, hi = s_prev - dk, s_prev + dk
        _, _, dmin = grid_projections(grid_lane, r['x'], r['y'], lo, hi)
        if abs(dmin - r['e_perp']) > PERP_CHECK:
            raise ValueError(f'frame {k}: rebuilt nearest distance {dmin!r} is not the row e_perp {r["e_perp"]!r}')
        heads = heads_of(r['x'], r['y'], lo, hi)
        e_bao[k] = max(g4_value(r['e_perp'], r['psi'], h, *nominal(h)) for h in heads)
        n_sol[k] = len(heads)
        s_prev, t_prev = r['s_car'], r['t']
    return e_bao, n_sol


def _frames(rows, e_bao, theta_key):
    return dict(e_perp=np.array([r['e_perp'] for r in rows], dtype=float),
                e_trk=np.array([r['e_trk'] for r in rows], dtype=float),
                e_bao=np.asarray(e_bao, dtype=float),
                e_theta_cmd_deg=np.degrees(np.abs(np.array([r[theta_key] for r in rows], dtype=float))))


def _summary_g4(out, e_bao, n_sol):
    m = float(np.max(e_bao)) if len(e_bao) else float('nan')
    out.update(e_bao_max=m, e_bao_phys_max=max(0.0, m) if math.isfinite(m) else float('nan'),
               g4_tie_frames=int(np.count_nonzero(n_sol > 1)), eval_version=EVAL_VERSION)
    return out


def evaluate_fig8(log, path, sch, lsum=0.24):
    """evaluate_v7 (unchanged) + G4 and the per-frame values. Returns (summary, rows, frames)."""
    out, rows = O.evaluate_v7(log, path, sch, lsum)

    def heads_of(px, py, lo, hi):
        idx, _, _ = grid_projections(path, px, py, lo, hi)
        return [float(path.head[i]) for i in idx]
    e_bao, n_sol = g4_frames(rows, path, sch.vmax, heads_of, fig8_nominal)
    return _summary_g4(out, e_bao, n_sol), rows, _frames(rows, e_bao, 'e_theta_cmd')


def evaluate_k(log, task, lsum=0.24):
    """o1_lanes.evaluate_k (unchanged) + G4 (exact sides, outgoing side at a corner) and the per-frame values."""
    corners = L.k_corners(task)
    lane, sch, poly = L.Lane(corners), L.KSchedule(task), Polyline(corners)
    out, rows = L.evaluate_k(log, task, lane, sch, lsum)

    def heads_of(px, py, lo, hi):
        return [h for _, _, h in poly.solutions(px, py, lo, hi)]
    e_bao, n_sol = g4_frames(rows, lane, sch.vmax, heads_of, k_nominal)
    return _summary_g4(out, e_bao, n_sol), rows, _frames(rows, e_bao, 'e_theta_start')
