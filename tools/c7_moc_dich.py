#!/usr/bin/env python3
"""
c7_moc_dich.py: the mark-shift rule of the C7/C8 camera (pure functions, no video, no camera).

CODE_CAM (local_411f78c5), 18/11/2026. Design CLAUDE_THESIS 03_CODE/docs/2026-11-18_Thiet_ke_33_moc_kiem_cheo_A3_v3.md
section 3, with v4 (2026-11-18_Thiet_ke_33_moc_kiem_cheo_A3_v4.md) sections 1, 2 over it; Liam 18/11 (CODE_CAM
local_a0214fe6, choice box): "Gop vao Codex ra code (Recommended)". c7_measure.py builds the inputs (the per-frame
position of every seen floor mark by its comparator, minus its H_a baseline) and applies the schedule this file
returns; c7_kiem_cheo.py (CODE_KIEMCHEO) computes the thresholds (u_so.json). Everything is per CHAIN (xe, 0..G-1):
the chain k never sees the data of group k (that is done by c7_measure: the deltas of chain k come from cameras and
anchor sets without group k).

Rule (thresholds T = nguong_m, T_raw = nguong_tho_m of the chain):
  delta      d_j(t) = comparator position - baseline [m] (2,), NaN when j is not seen / no comparator fit
  sighting   the samples of j in one clip with no gap above WIN_GAP_S between consecutive samples
  window     at a frame with a sample: the samples of the current sighting with t_e - t_i < WIN_S, at least
             WIN_MIN_N of them; statistic m(t) = MEDIAN OF THE MAGNITUDES |d| over the window (v4 1; not the
             median vector)
  alarm      m(t) > T; onset = first frame of a run of alarm frames
  common     an onset with another onset of a mark not excluded within NEN_WIN_S in the same clip, two onsets in one
             frame, or a frame where at least max(2, ceil(n_seen / 4)) seen marks have |d| > T_raw: the base moved
             (nen_doi), nobody is excluded. Open from the last BASE CHECK (the window start of the last frame where every
             seen mark not excluded had a window with m <= T) to the first frame of a span >= NEN_WIN_S in one clip
             where every alarmed mark of the episode is seen with m <= T (v4 2), or the start of a later clip H where
             every mark of the chain is seen in >= KIEM_H_MIN of the frames and every window has m <= T. Not closed:
             to the end of the session. Every frame inside: not concluded.
  exclusion  an onset that is not common: j leaves the fit of the chain from t_bd to the end of the session.
             t_bd = start of the window of the last frame before the onset, in the same sighting, with m <= T (v4 1:
             if m <= T on W then at least half of the samples of W are <= T, so a shift D > T + n_max started after
             the middle of W). None in the sighting (alarm at the first window, after an occlusion, at a clip start):
             W* = the last such frame before the sighting (earlier sighting or clip); t_bd = start of its window and
             every frame from there to the onset carries bat_dinh_moc {j} (not concluded); no W* at all: from the
             session start.
  moc_nghi   a frame where |d_j| > T_raw while j is not excluded: j leaves the fit of THAT frame only.
Positions are (clip index, frame index) in session order. The schedule is a plain dict (JSON); schedule_sha256 hashes
its canonical form (the fixed-point test of the convergence, v4 3).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

WIN_S = 0.5            # [DE XUAT] window of the filtered statistic (design v2 3; same in c7_kiem_cheo)
WIN_MIN_N = 10         # [DE XUAT] at least 10 samples
WIN_GAP_S = 0.1        # [DE XUAT] a gap above this restarts the sighting
NEN_WIN_S = 1.0        # [DE XUAT] two alarms within 1 s: common (v3 3); a base check span of 1 s closes it (v4 2)
NEN_RAW_FRAC = 0.25    # [DE XUAT] a quarter of the seen marks over T_raw in one frame: common
NEN_RAW_MIN = 2        #   and at least two of them (one mark over T_raw is a moc_nghi, not a base change)
KIEM_H_MIN = 0.95      # [DE XUAT] clip H closing a nen_doi: every mark seen in this share of the frames (c7_kiem_cheo)
SCHEDULE_VERSION = 'c7_moc_dich_v1'


class Clip:
    """One clip of the session in order: name, kind ('luot' or 'H'), frame times t (n,) [s], frame idx (n,) (the
    frame numbers of the video), deltas {mark: (n, 2) array, NaN rows where no sample}."""

    def __init__(self, name, kind, t, d, idx=None):
        self.name, self.kind = str(name), kind
        self.t = np.asarray(t, float)
        self.idx = np.arange(len(self.t)) if idx is None else np.asarray(idx, int)
        self.d = {int(j): np.asarray(v, float).reshape(len(self.t), 2) for j, v in d.items()}
        if len(self.t) > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError(f'clip {name}: frame times must increase')


def windows(t, r, win_s=WIN_S, min_n=WIN_MIN_N, gap_s=WIN_GAP_S):
    """Per frame: (m, ws, sight). r: magnitudes (n,), NaN where no sample. m = median of r over the window of the frame
    (NaN when the frame has no sample or fewer than min_n samples in the window); ws = frame position (0..n-1) of the
    first sample of that window (-1 when m is NaN); sight = sighting number of the frame's sample (-1 without sample)."""
    n = len(t)
    m = np.full(n, np.nan)
    ws = np.full(n, -1, int)
    sight = np.full(n, -1, int)
    have = np.flatnonzero(np.isfinite(r))
    s_id, last, start_pos = -1, None, 0
    pos_in = []                                          # sample positions of the current sighting
    for e in have:
        if last is None or t[e] - t[last] > gap_s:
            s_id += 1
            pos_in = []
            start_pos = 0
        last = e
        pos_in.append(e)
        while start_pos < len(pos_in) and not t[e] - t[pos_in[start_pos]] < win_s:
            start_pos += 1
        sight[e] = s_id
        w = pos_in[start_pos:]
        if len(w) >= min_n:
            m[e] = float(np.median(r[w]))
            ws[e] = w[0]
    return m, ws, sight


def _onsets(alarm, sight):
    """Frames that start a run of alarm frames within one sighting: an alarm frame whose previous frame WITH A SAMPLE
    in the same sighting is not an alarm (a dropped frame inside a run does not start a new one)."""
    out = []
    prev_e = None
    for e in np.flatnonzero(sight >= 0):
        if alarm[e] and (prev_e is None or sight[prev_e] != sight[e] or not alarm[prev_e]):
            out.append(int(e))
        prev_e = e
    return out


def schedule(clips, T, T_raw, ids, win_s=WIN_S, min_n=WIN_MIN_N, gap_s=WIN_GAP_S, nen_win_s=NEN_WIN_S,
             raw_frac=NEN_RAW_FRAC, raw_min=NEN_RAW_MIN, kiem_h_min=KIEM_H_MIN):
    """The schedule of ONE chain over the session (module docstring). clips: [Clip] in session order; T, T_raw [m];
    ids: the marks of the chain's universe U_c (deltas of other marks are ignored). Returns a dict:
      loai          {j: dict(tu=[q, f], t_s, bao_dong=[q, f], bat_dinh)}   f: frame idx of the video
      moc_nghi      [[j, q, f], ...]
      nen_doi       [dict(tu=[q, f], den=[q, f] or None, moc=[...])]      den: first frame NOT in the interval
      bat_dinh_moc  [dict(moc=j, tu=[q, f], den=[q, f])]                  den: the onset frame (inside)
    with q the clip index. Positions are in frame idx of the clip (Clip.idx)."""
    if not (math.isfinite(T) and math.isfinite(T_raw) and T > 0 and T_raw > 0):
        raise ValueError(f'thresholds T {T!r}, T_raw {T_raw!r}: finite and > 0')
    ids = sorted(int(i) for i in ids)
    Q = len(clips)
    # per clip, per mark: magnitudes, windows, alarms
    R, M, WS, SG, ON = [], [], [], [], []
    for c in clips:
        r_q, m_q, ws_q, sg_q, on_q = {}, {}, {}, {}, {}
        for j in ids:
            d = c.d.get(j)
            r = np.full(len(c.t), np.nan) if d is None else np.linalg.norm(d, axis=1)
            m, ws, sg = windows(c.t, r, win_s, min_n, gap_s)
            r_q[j], m_q[j], ws_q[j], sg_q[j] = r, m, ws, sg
            on_q[j] = _onsets(np.isfinite(m) & (m > T), sg)
        R.append(r_q), M.append(m_q), WS.append(ws_q), SG.append(sg_q), ON.append(on_q)

    def pos(q, e):
        return [q, int(clips[q].idx[e])]

    def h_ok(q):
        """A clip H closes an open nen_doi: every mark of the chain seen in >= kiem_h_min of the frames and every
        window of the clip with m <= T."""
        c = clips[q]
        if c.kind != 'H' or not len(c.t):
            return False
        for j in ids:
            if np.mean(np.isfinite(R[q][j])) < kiem_h_min:
                return False
            mm = M[q][j][np.isfinite(M[q][j])]
            if not len(mm) or np.any(mm > T):
                return False
        return True

    E = {}                       # j -> exclusion record
    nen, bat = [], []
    state, episode = 'normal', None
    last_check = None            # (q, e) of the window start of the last base check
    last_ok = {j: None for j in ids}      # (q, e_ws) of the last frame with m <= T, per mark, before the current frame
    for q, c in enumerate(clips):
        n = len(c.t)
        if state == 'nen' and h_ok(q):
            episode['den'] = pos(q, 0)
            nen.append(episode)
            state, episode = 'normal', None
        onset_at = {}
        for j in ids:
            for e in ON[q][j]:
                onset_at.setdefault(e, []).append(j)
        span = None
        for e in range(n):
            t = c.t[e]
            seen = [j for j in ids if np.isfinite(R[q][j][e]) and j not in E]
            on = [j for j in onset_at.get(e, []) if j not in E]
            over = [j for j in seen if R[q][j][e] > T_raw]
            base_raw = len(over) >= max(raw_min, math.ceil(raw_frac * len(seen))) if seen else False
            if state == 'normal':
                partners = set()
                for j in on:
                    for j2 in ids:
                        if j2 == j or j2 in E:
                            continue
                        for e2 in ON[q][j2]:
                            if abs(c.t[e2] - t) <= nen_win_s:
                                partners.add(j2)
                                break
                if base_raw or len(on) >= 2 or partners:
                    state = 'nen'
                    episode = dict(tu=pos(*last_check) if last_check else pos(0, 0), den=None, moc=sorted(set(on) | partners | (set(over) if base_raw else set())))
                    span = None
                else:
                    for j in on:
                        sg, prev = SG[q][j][e], None
                        for e2 in range(e - 1, -1, -1):
                            if SG[q][j][e2] >= 0 and SG[q][j][e2] != sg:
                                break                    # an earlier sighting: W is not in this one
                            if SG[q][j][e2] == sg and np.isfinite(M[q][j][e2]) and M[q][j][e2] <= T:
                                prev = e2
                                break
                        if prev is not None:
                            tu, bd = (q, int(WS[q][j][prev])), False
                        elif last_ok[j] is not None:
                            tu, bd = last_ok[j], True
                        else:
                            tu, bd = (0, 0), True
                        E[j] = dict(tu=pos(*tu), t_s=float(clips[tu[0]].t[tu[1]]), bao_dong=pos(q, e), bat_dinh=bd)
                        if bd:
                            bat.append(dict(moc=j, tu=pos(*tu), den=pos(q, e)))
                    seen = [j for j in seen if j not in E]
                    win = [j for j in seen if np.isfinite(M[q][j][e])]
                    if seen and len(win) == len(seen) and all(M[q][j][e] <= T for j in seen):
                        last_check = (q, int(min(WS[q][j][e] for j in seen)))
            else:
                episode['moc'] = sorted(set(episode['moc']) | set(on) | (set(over) if base_raw else set()))
                al = episode['moc']
                ok = all(j in E or (np.isfinite(M[q][j][e]) and M[q][j][e] <= T) for j in al) and \
                    any(j not in E for j in al)
                if ok and not base_raw and not on:
                    if span is None:
                        span = e
                    if t - c.t[span] >= nen_win_s:
                        episode['den'] = pos(q, span)
                        nen.append(episode)
                        state, episode, span = 'normal', None, None
                else:
                    span = None
            for j in ids:
                if np.isfinite(M[q][j][e]) and M[q][j][e] <= T:
                    last_ok[j] = (q, int(WS[q][j][e]))
    if state == 'nen':
        nen.append(episode)
    # moc_nghi: raw over T_raw while not excluded (v3 3 'dich ngan'); frames of an open base change included
    nghi = []
    for q, c in enumerate(clips):
        for j in ids:
            over = np.flatnonzero(np.isfinite(R[q][j]) & (R[q][j] > T_raw))
            for e in over:
                p = pos(q, int(e))
                if j in E and _le(E[j]['tu'], p):
                    continue
                nghi.append([j, p[0], p[1]])
    nghi.sort(key=lambda x: (x[1], x[2], x[0]))
    return dict(loai={str(j): E[j] for j in sorted(E)}, moc_nghi=nghi, nen_doi=nen, bat_dinh_moc=bat)


def _key(p):
    return int(p[0]), int(p[1])


def _le(a, b):
    """Position a <= b (clip index, then frame idx)."""
    return _key(a) <= _key(b)


def excluded_at(sched, q, f):
    """Marks out of the chain's fit at clip index q, frame idx f: exclusions from their tu on, plus the moc_nghi of
    that very frame."""
    out = {int(j) for j, v in sched.get('loai', {}).items() if _le(v['tu'], (q, f))}
    out |= {int(j) for j, qq, ff in sched.get('moc_nghi', []) if qq == q and ff == f}
    return out


def inconclusive_at(sched, q, f):
    """(in a nen_doi interval, [marks with bat_dinh_moc]) at clip index q, frame idx f."""
    p = (q, f)
    in_nen = any(_le(v['tu'], p) and (v['den'] is None or _key(p) < _key(v['den'])) for v in sched.get('nen_doi', []))
    bd = sorted(int(v['moc']) for v in sched.get('bat_dinh_moc', []) if _le(v['tu'], p) and _le(p, v['den']))
    return in_nen, bd


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def schedule_sha256(obj):
    return hashlib.sha256(canonical(obj).encode('ascii')).hexdigest()
