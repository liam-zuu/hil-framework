"""
s02_plant_ticks.py: spec S02 Viec CODE 3 on HIL: the plant state of the Pi frame log brought onto the ESP32 clock
(every metric tick), and the sender t = 0 of the run on the ESP32 clock (spec S02 v1.13 P2-2b, design Q8 (a)),
CERTIFIED by the set of every ESP32 -> H7 time map that agrees with every logged encoder count.

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van) section 28 (Liam 16/11 CODE_S02
local_95df9cdf, "Tap kha di (Recommended)", "Dung so de xuat (Recommended)"); replaces the least-squares map of
sections 10.2, 25 that Codex Sol round 3 left open (S2P3-1, -2, -3, -7) and the lock CERTIFIED = False.

    python tools/s02_plant_ticks.py --frame-log <run.npz> --esp-log <run.csv> --test A-L2 --out <run>_plant.csv
    -> <run>_plant.csv (PLANT_COLUMNS of fwsim/s02_groupA_sim.py, one row per motor-on ESP32 line; empty where the
       map does not reach) and <run>_plant_map.json; last line '#ket_qua,ma=0 | 1 (not mapped) | 2 (error)'

Time axes. Everything is placed on the H7 clock (as tools/hil_edge_map.py, L100, RS-4 P1; only imported): the H7
apply window of every frame (MISO v5: the speeds of frame k took effect inside [apply_k, apply_k + width_k]) and
its receive stamp (anchors) come from hil_edge_map.h7_apply, which also refuses frame logs it cannot read (48-byte
frames, run_id, reply count, CYCCNT unwrap). The plant step of an H7 time is hil_edge_map.h7_to_steps (between
the receives of answered frames); the Pi clock is not a time axis. The unknown is the map s = a + b u from the
ESP32 tick time u (timestamp_us) to the H7 time s, |b - 1| <= RHO_PPM (ESP32 vs H7 rate, hil_edge_map.RHO_MAX_PPM,
Liam 13/11 L100).

Count model (S2P3-7). The H7 EncoderGen is simulated edge by edge (simulate_edges: encoder_gen.c _start_wheel,
_retime, callbacks adding the current half period, _stop at |omega| <= 0.3 rad/s, the half period in float32 as the
H7, hil_edge_map.h7_half_period) with every frame applied at the start of its window. Each edge time carries an
interval [lo, hi] for what the window widths can change: a start or a re-anchoring retime inside its window; a
callback whose time may fall on either side of a speed change (the old or the new half period); a half period the
PC float may round the other way. A start forces both pins low: the ESP32 may count up to one edge per pin that was
high (|o| <= 2 at the first start of the log, the pin levels after that are known from the toggles); an edge that
may come either side of a stop makes the count after it uncertain by one. Checked against the unchanged
encoder_gen.c on the PC (tools/tests/encgen_oracle.py) in tools/tests/test_s02_plant_ticks.py.

Set of maps (S2P3-1, -2: no grid, no "best minimum"). The ESP32 reads its counts within READ_US of t_now (AGV
16702ff / 5bde4c5, hil_edge_map.FW_COUNTS_AT_TICK; other firmware refused). For a line j on wheel w with count c and
the count c0 the wheel had at rest, the number of edges the H7 made before the read must be c - c0 - O (O the start
offsets): inside one segment of one direction that holds for the read time in one interval [edge_k + lo, edge_k+1 +
hi), a linear constraint on (a, b). Absolute bound (S2P3-3, independent of the command anchors): every wheel must be
at rest (no edge, constant count) for at least STILL_MIN_S before its first start on both sides; the last line
before the count first changes is read before that change, the first changed line after the start window begins.
The first-segment offset O of each wheel is tried at every value it may take; later segments widen it by what each
start or stop may add. A line is used once the current set puts its read time inside one segment (repeated as the
set shrinks). The set is a convex polygon in (s at u_A, s at u_B), handled by linear programming
(scipy.optimize.linprog, HiGHS): empty -> 'inconsistent' (the model or the data is wrong: not mapped). Several
offset choices left -> the union is kept.
Certified (map ok) when the spread of s (largest minus smallest over the set) is <= SPREAD_MAX_US [ĐỀ XUẤT 2 ms,
Liam 16/11] at both ends of [t0, t0 + T_cmd] (t0 = sender t = 0 on the ESP32 clock; the spread is convex in u, so
the ends bound it inside). The map given is the midpoint line of the two end ranges: its error is at most half the
spread inside the window, if the model holds.

Sender t = 0 (Q8 (a)). The first frame with a non-zero command carries the schedule sample k_a; the frame sent k_a x
20 ms of Pi time earlier (t_send_ns) is the sample 0 frame; its H7 receive (hil_edge_map anchors, interpolated for a
frame the H7 did not answer) is t = 0 on the H7 clock, mapped to the ESP32 clock by the midpoint map: sender_t0_us,
with sender_spread_us (the spread of s there).
Plant at a tick: frame state interpolated at the fractional plant step of s (theta unwrapped, wrapped back to
(-pi, pi]); across frames the H7 did not answer the step is only known in [a, b]: the middle is used and the largest
such half gap is reported (step_gap_max). Outside the frame log: empty (no extrapolation, P2-3).
Not mapped either: a frame log with mosi_true (a run with --faults, O-3), a run whose '#us_end' reports cut ticks
(us_cut, S2P2-2), a wheel that never starts, a log without STILL_MIN_S of rest before the start.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hil_edge_map as EM  # noqa: E402
import s02_cmd_path  # noqa: E402

WHEELS = ('fl_count', 'fr_count', 'rl_count', 'rr_count')
SIGN = (1, -1, 1, -1)                 # encoder_gen.c ENC_DIR_SIGN
OMEGA_MIN = EM.OMEGA_MIN              # 0.3 rad/s
RETIME_MARGIN = 3                     # encoder_gen.c ENC_RETIME_MARGIN
READ_US = EM.J_READ_US                # AGV LOG_CNT_WIN_US: counts read within this of t_now
RHO_PPM = EM.RHO_MAX_PPM              # ESP32 vs H7 clock rate bound (L100)
STILL_MIN_S = 1.0                     # [ĐỀ XUẤT] Liam 16/11 CODE_S02 "Dung so de xuat"
SPREAD_MAX_US = 2000.0                # [ĐỀ XUẤT] Liam 16/11 CODE_S02 "Dung so de xuat"
O_FIRST = 2                           # a start forces two pins low: at most 2 counts
MAX_COMBOS = 25                       # offset combinations kept across wheels
TRUST_FRAC = 0.25                     # an edge interval wider than this x the edge spacing ends the trusted edges
TOL_US = 2.0                          # timer tick (1 us: the H7 compare fires within the tick) and rounding
MAX_ITER = 30
PLANT_COLUMNS = ['timestamp_us', 'x', 'y', 'theta', 'vx_w', 'vy_w', 'vx_b', 'vy_b', 'wz', 'w1', 'w2', 'w3', 'w4']


# ---------------------------------------------------------------------------------------------------------------
# EncoderGen, edge by edge (encoder_gen.c)

def simulate_edges(apply_us, width_us, om32):
    """Edges of the four wheels for frames applied at apply_us (H7 us, increasing; the true apply lies in
    [apply, apply + width_us]) with the H7 float32 wheel speeds om32 (n, 4). One dict per wheel:
      t, lo, hi   simulated edge times and the interval of (true time - t)
      seg         segment of each edge; segment 0 is the rest before the first start, a new segment begins at
                  every start and every stop
      segs        per segment: t0, w0 (its boundary's apply window), kind ('rest', 'start', 'stop'), dir,
                  o_lo, o_hi (count the boundary itself may add: forced-low pins at a start, |o| <= pins high;
                  edges that may fall either side of a stop), n_before (simulated signed count before it),
                  first_edge (index of its first edge), n_edges
      t_end       last apply time (edges are simulated up to it)."""
    apply_us = np.asarray(apply_us, dtype=float)
    width_us = np.asarray(width_us, dtype=float)
    hp_all, amb_all = EM.h7_half_period(om32)
    out = []
    for w in range(4):
        om = om32[:, w]
        d_all = np.where(om > np.float32(OMEGA_MIN), 1, np.where(om < -np.float32(OMEGA_MIN), -1, 0))
        t_l, lo_l, hi_l, sg_l = [], [], [], []
        segs = [dict(t0=-math.inf, w0=0.0, kind='rest', dir=0, o_lo=0, o_hi=0, n_before=0, first_edge=0)]
        st = dict(dir=0, hp=0.0, amb=False, ncount=0, last=None)   # last: (T, W, hp_old, hp_new) of the last retime
        nxt = [math.inf, math.inf]           # pending edge of A, B
        err = [[0.0, 0.0], [0.0, 0.0]]
        own = [[0.0, 0.0], [0.0, 0.0]]       # part of err added by each channel's own callbacks since the last
                                             # start or retime (the rest is common to A and B)
        level = [None, None]                 # pin levels; None: unknown (before the first start of the log)

        def emit_until(T):
            """Edges up to apply time T; each callback adds the current half period (encoder_gen.c callback)."""
            while st['dir'] != 0:
                ch = 0 if nxt[0] <= nxt[1] else 1
                e = nxt[ch]
                if e > T:
                    break
                lo, hi = err[ch]
                t_l.append(e); lo_l.append(lo); hi_l.append(hi); sg_l.append(len(segs) - 1)
                st['ncount'] += st['dir']
                if level[ch] is not None:
                    level[ch] ^= 1
                hp = st['hp']
                d_lo, d_hi = (-1.0, 1.0) if st['amb'] else (0.0, 0.0)
                nxt[ch] = e + hp
                err[ch] = [lo + d_lo, hi + d_hi]
                own[ch] = [own[ch][0] + d_lo, own[ch][1] + d_hi]
                lr = st['last']
                if lr is not None and e + lo <= lr['T'] + lr['W']:
                    # this callback may have come before the true apply of the last retime: it then added the old half
                    # period and the retime (after it) took the other channel first, which kept its old pending edge
                    # y_old; this one follows at y_old + hp_new / 2. Both channels move by
                    # delta = (y_old - e) - hp_new / 2 (common; (hp_old - hp_new) / 2 when A and B were half a period
                    # apart). Not the first callback of the retime's leader: the whole hp change.
                    if ch == lr['lead'] and not lr['used']:
                        dl = (lr['y_old'] - e) - lr['hp_new'] / 2.0
                        lo_d, hi_d = min(0.0, dl), max(0.0, dl)
                    else:
                        dh = lr['hp_old'] - lr['hp_new']
                        lo_d, hi_d = min(0.0, dh), max(0.0, dh)
                    lr['used'] = True
                    for c2 in (0, 1):
                        err[c2] = [err[c2][0] + lo_d, err[c2][1] + hi_d]

        def stop_boundary(T, W):
            # emitted edges whose true time may be after the true stop, pending ones that may come before it
            late = sum(1 for k in range(max(0, len(t_l) - 4), len(t_l)) if t_l[k] + hi_l[k] > T)
            early = sum(1 for ch in (0, 1) if math.isfinite(nxt[ch]) and nxt[ch] + err[ch][0] <= T + W)
            d = st['dir']
            o_lo, o_hi = (-late, early) if d > 0 else (-early, late)
            segs.append(dict(t0=T, w0=W, kind='stop', dir=0, o_lo=o_lo, o_hi=o_hi, n_before=st['ncount'],
                             first_edge=len(t_l)))
            if late or early:
                level[0] = level[1] = None
            st['dir'], st['last'] = 0, None
            nxt[0] = nxt[1] = math.inf

        for f in range(len(apply_us)):
            T, W = float(apply_us[f]), float(width_us[f])
            emit_until(T)
            new_dir, hpn, ambn = int(d_all[f]), float(hp_all[f, w]), bool(amb_all[f, w])
            if new_dir == 0:
                if st['dir'] != 0:
                    stop_boundary(T, W)
                continue
            if st['dir'] == 0 or new_dir != st['dir']:
                if st['dir'] != 0:                    # reversal: _stop, then _start_wheel
                    stop_boundary(T, W)
                m = 2 if None in level else int(level[0]) + int(level[1])
                segs.append(dict(t0=T, w0=W, kind='start', dir=new_dir, o_lo=-m, o_hi=m, n_before=st['ncount'],
                                 first_edge=len(t_l)))
                level[0] = level[1] = 0
                st.update(dir=new_dir, hp=hpn, amb=ambn, last=None)
                h2 = float(int(hpn) >> 1)
                a_off, b_off = (hpn, hpn + h2) if new_dir * SIGN[w] > 0 else (hpn + h2, hpn)
                a1 = 1.0 if ambn else 0.0
                nxt[0], nxt[1] = T + a_off, T + b_off
                err[0], err[1] = [-a1, W + a1], [-a1, W + a1]
                own[0], own[1] = [0.0, 0.0], [0.0, 0.0]
                continue
            if hpn != st['hp']:
                # an edge just before T whose true time may be after the true apply: its callback then added the new
                # half period and this retime saw the other channel first (the mirror case of emit_until)
                pre = [k for k in range(max(0, len(t_l) - 2), len(t_l))
                       if sg_l[k] == len(segs) - 1 and t_l[k] + hi_l[k] >= T]
                hp_old = st['hp']
                ch_pre = None
                if pre:
                    k = pre[-1]                        # its channel: the one now pending at t + hp_old
                    ch_pre = 0 if abs(nxt[0] - (t_l[k] + hp_old)) <= abs(nxt[1] - (t_l[k] + hp_old)) else 1
                lead, y_old = _retime(nxt, err, own, T, W, hpn, new_dir * SIGN[w], ambn)
                if pre:
                    e = t_l[pre[-1]]
                    if ch_pre is not None and ch_pre != lead:
                        # the true callback came after the true apply: that channel was first (pending e), kept e,
                        # the other followed at e + hp_new / 2, then the callback added hp_new: both channels move
                        # by delta = e + hp_new / 2 - (new simulated leader edge)
                        dl = e + hpn / 2.0 - nxt[lead]
                        lo_d, hi_d = min(0.0, dl), max(0.0, dl)
                    else:
                        dh = hp_old - hpn
                        lo_d, hi_d = min(0.0, -dh), max(0.0, -dh)
                    for c2 in (0, 1):
                        err[c2] = [err[c2][0] + lo_d, err[c2][1] + hi_d]
                st['last'] = dict(T=T, W=W, hp_old=hp_old, hp_new=hpn, lead=lead, y_old=y_old, used=False)
                st.update(hp=hpn, amb=ambn)
            else:
                st['amb'] = st['amb'] or ambn
        ed = dict(t=np.array(t_l), lo=np.array(lo_l), hi=np.array(hi_l), seg=np.array(sg_l, dtype=np.int64),
                  segs=segs, t_end=float(apply_us[-1]) if len(apply_us) else -math.inf)
        cnt = np.bincount(ed['seg'], minlength=len(segs)) if len(t_l) else np.zeros(len(segs), dtype=np.int64)
        for k, s in enumerate(segs):
            s['n_edges'] = int(cnt[k])
        out.append(ed)
    return out


def _retime(nxt, err, own, T, W, hpn, phase, amb):
    """encoder_gen.c _retime at apply T (true apply A in [T, T + W]): the channel whose edge comes first keeps it, or
    takes T + hp if that is sooner; the other follows it by hp / 2. The true new edge is min(old + err, A + hp). The
    two pending edges share the error common to both channels; only their own callback terms (own) can swap their
    order: when they can, the true first channel may be the other one and both are widened by hp / 2."""
    d1, d2 = nxt[0] - T, nxt[1] - T
    ch = 0 if ((d1 < d2) if d1 != d2 else (phase > 0)) else 1
    old = nxt[ch]
    lo_o, hi_o = err[ch]
    sim = old if ((old - T) <= hpn or hpn < RETIME_MARGIN) else T + hpn
    a1 = 1.0 if amb else 0.0
    new = [min(old + lo_o, T + hpn) - sim - a1, min(old + hi_o, T + W + hpn) - sim + a1]
    o = 1 - ch
    gap = nxt[o] - nxt[ch]                         # >= 0: o comes second in simulation
    if gap <= own[ch][1] - own[o][0]:
        new = [new[0] - hpn / 2.0, new[1] + hpn / 2.0]
    y_old = nxt[o]
    nxt[ch], nxt[o] = sim, sim + float(int(hpn) >> 1)
    err[ch], err[o] = list(new), list(new)
    own[0], own[1] = [0.0, 0.0], [0.0, 0.0]
    return ch, y_old


# ---------------------------------------------------------------------------------------------------------------
# The set of maps

class _Wheel:
    """Count lookups of one wheel's simulated edges."""

    def __init__(self, ed):
        self.t, self.lo, self.hi = ed['t'], ed['lo'], ed['hi']
        self.segs = ed['segs']
        self.t_end = ed['t_end']
        self.s_from = np.array([s['t0'] + s['w0'] for s in self.segs])      # a segment holds after its window
        self.s_to = np.array([s['t0'] for s in self.segs[1:]] + [math.inf])
        self.dir = np.array([s['dir'] for s in self.segs])
        self.n0 = np.array([s['n_before'] for s in self.segs], dtype=float)
        self.e0 = np.array([s['first_edge'] for s in self.segs], dtype=np.int64)
        self.m = np.array([s['n_edges'] for s in self.segs], dtype=np.int64)
        # offset of segment s >= 1 is O (first start) + what boundaries 2..s may add
        add_lo = np.zeros(len(self.segs))
        add_hi = np.zeros(len(self.segs))
        for k in range(2, len(self.segs)):
            add_lo[k] = add_lo[k - 1] + self.segs[k]['o_lo']
            add_hi[k] = add_hi[k - 1] + self.segs[k]['o_hi']
        self.add_lo, self.add_hi = add_lo, add_hi
        self.first_m = int(self.segs[1]['o_hi']) if len(self.segs) > 1 else 0
        # trusted edges: inside each segment, up to the first edge whose interval is wider than TRUST_FRAC of the
        # spacing to its neighbours (past it, which simulated edge is which true edge is not certain; checked against
        # encoder_gen.c in the tests); an untrusted edge bounds nothing
        self.trusted = np.zeros(len(self.t), bool)
        seg = ed['seg']
        for k, sg in enumerate(self.segs):
            i = np.flatnonzero(seg == k)
            if i.size == 0:
                continue
            tt = self.t[i]
            gap = np.minimum(np.r_[np.diff(tt), np.inf], np.r_[np.inf, np.diff(tt)])
            wide = np.flatnonzero(self.hi[i] - self.lo[i] > TRUST_FRAC * gap)
            self.trusted[i[:wide[0] if wide.size else i.size]] = True

    def segment_of(self, r_lo, r_hi):
        """Segment index holding the whole read interval [r_lo, r_hi] (arrays), -1 where none."""
        s = np.searchsorted(self.s_from, r_lo, side='right') - 1
        ok = (s >= 0) & (r_hi < self.s_to[np.clip(s, 0, None)]) & (r_hi <= self.t_end)
        return np.where(ok, s, -1)

    def bounds(self, s, n_obs, O):
        """Arrays (t_min, t_max, ok) of the true read time for lines on segments s with counts n_obs (relative to the
        rest count) and first-segment offset O: the read comes at or after edge k1 and before edge k2 + 1 of the
        segment (k edges since its start); -inf / +inf when no edge bounds it; ok False when no count fits."""
        s = np.asarray(s, dtype=np.int64)
        n_obs = np.asarray(n_obs, dtype=float)
        first = s >= 1
        o_lo = np.where(first, O + self.add_lo[s], 0.0)
        o_hi = np.where(first, O + self.add_hi[s], 0.0)
        v_lo, v_hi = n_obs - o_hi, n_obs - o_lo
        d = self.dir[s]
        n0 = self.n0[s]
        rest = d == 0
        ka, kb = (v_lo - n0) * d, (v_hi - n0) * d
        k1 = np.ceil(np.minimum(ka, kb) - 1e-9)
        k2 = np.floor(np.maximum(ka, kb) + 1e-9)
        m = self.m[s]
        ok = np.where(rest, (v_lo <= n0) & (n0 <= v_hi), (k2 >= 0) & (k1 <= m) & (k1 <= k2))
        k1c = np.clip(k1, 0, None).astype(np.int64)
        k2c = np.clip(k2, None, m).astype(np.int64)
        i_lo = np.clip(self.e0[s] + k1c - 1, 0, max(len(self.t) - 1, 0))
        i_hi = np.clip(self.e0[s] + k2c, 0, max(len(self.t) - 1, 0))
        if len(self.t):
            t_min = np.where(~rest & (k1c >= 1) & self.trusted[i_lo], self.t[i_lo] + self.lo[i_lo] - TOL_US, -np.inf)
            t_max = np.where(~rest & (k2c < m) & self.trusted[i_hi], self.t[i_hi] + self.hi[i_hi] + TOL_US, np.inf)
        else:
            t_min, t_max = np.full(len(s), -np.inf), np.full(len(s), np.inf)
        return t_min, t_max, ok


class MapSet:
    """Maps s(u) = u + off + (1 - l) xA + l xB, l = (u - uA) / (uB - uA): x = (xA, xB) the H7 time at uA, uB minus
    (u + off), in us. Rows A x <= b; the rate bound |(s(uB) - s(uA)) / (uB - uA) - 1| <= RHO_PPM."""

    def __init__(self, uA, uB, off):
        self.uA, self.uB, self.off = float(uA), float(uB), float(off)
        L = self.uB - self.uA
        r = RHO_PPM * 1e-6 * L
        self.A = [np.array([[1.0, -1.0], [-1.0, 1.0]])]
        self.b = [np.array([r, r])]

    def lam(self, u):
        return (np.asarray(u, dtype=float) - self.uA) / (self.uB - self.uA)

    def rows_le(self, u, v):
        """Rows of s(u) <= v (arrays)."""
        l = self.lam(u)
        return np.column_stack([1.0 - l, l]), np.asarray(v, dtype=float) - np.asarray(u, dtype=float) - self.off

    def rows_ge(self, u, v):
        A, b = self.rows_le(u, v)
        return -A, -b

    def add(self, A, b):
        self.A.append(np.atleast_2d(A))
        self.b.append(np.atleast_1d(b))

    def _ab(self, extra):
        A = np.vstack(self.A + [e[0] for e in extra if len(e[1])])
        b = np.concatenate(self.b + [e[1] for e in extra if len(e[1])])
        return A, b

    def extent(self, u, extra=()):
        """(min, max) of s(u) over the polygon with the extra rows (H7 us), None if empty."""
        A, b = self._ab(extra)
        l = float(self.lam(u))
        c = np.array([1.0 - l, l])
        lo, hi = _lp(c, A, b), _lp(-c, A, b)
        if lo is None or hi is None:
            return None
        base = float(u) + self.off
        return base + float(c @ lo), base + float(c @ hi)

    def feasible(self, extra=()):
        A, b = self._ab(extra)
        return _lp(np.zeros(2), A, b) is not None


def _lp(c, A, b):
    from scipy.optimize import linprog
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None)], method='highs')
    return r.x if r.status == 0 else None


def _wheel_rows(ms, wheel, te, n_obs, seg_of_line, O):
    """(A, b) rows of the used lines of one wheel for first-segment offset O, or None when a line fits no count.
    The ESP32 reads its counts in [s(u), s(u) + READ_US]: read >= t_min needs s(u) >= t_min - READ_US, read < t_max
    needs s(u) < t_max."""
    j = np.flatnonzero(seg_of_line >= 0)
    if j.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    t_min, t_max, ok = wheel.bounds(seg_of_line[j], n_obs[j], O)
    if not np.all(ok):
        return None
    out_A, out_b = [], []
    m = np.isfinite(t_min)
    if m.any():
        A, b = ms.rows_ge(te[j][m], t_min[m] - READ_US)
        out_A.append(A); out_b.append(b)
    m = np.isfinite(t_max)
    if m.any():
        A, b = ms.rows_le(te[j][m], t_max[m])
        out_A.append(A); out_b.append(b)
    if not out_A:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(out_A), np.concatenate(out_b)


def certify(te_us, counts, apply_us, width_us, om32, t_window=None):
    """The set of maps s = a + b u (H7 us of an ESP32 time u in us) that agree with every count (module docstring).
    te_us (m,), counts (m, 4) of the motor-on ESP32 lines (NaN: not used); apply_us, width_us, om32 of the applied
    frames. t_window(a, b) -> (u0, u1): ESP32 times whose spread certifies (default the first and last line).
    Returns dict ok, reason, a, b, spread_us, offsets (kept first-start offsets per wheel combination), n_rows,
    n_lines, n_iter, ext (u -> (min, max) of s over the set)."""
    te = np.asarray(te_us, dtype=float)
    counts = np.asarray(counts, dtype=float)
    res = dict(ok=False, reason='', a=None, b=None, spread_us=None, offsets=None, n_rows=0, n_iter=0)
    if len(te) < 2:
        res['reason'] = 'fewer than two ESP32 lines'
        return res
    wh = [_Wheel(e) for e in simulate_edges(apply_us, width_us, om32)]
    starts = []
    for w in range(4):
        if len(wh[w].segs) < 2:
            res['reason'] = f'wheel {w} never starts in the frame log'
            return res
        S, W = wh[w].segs[1]['t0'], wh[w].segs[1]['w0']
        if S - float(apply_us[0]) < STILL_MIN_S * 1e6:
            res['reason'] = f'no_standstill_start (wheel {w}: {(S - apply_us[0]) * 1e-6:.3f} s of frames at rest)'
            return res
        c = counts[:, w]
        idx = np.flatnonzero(np.isfinite(c))
        if idx.size == 0:
            res['reason'] = f'wheel {w}: no count'
            return res
        c0 = c[idx[0]]
        ch = idx[c[idx] != c0]
        if ch.size == 0:
            res['reason'] = f'wheel {w}: the count never changes'
            return res
        j1 = int(ch[0])
        j0 = int(idx[idx < j1][-1])
        if te[j0] - te[idx[0]] < STILL_MIN_S * 1e6:
            res['reason'] = (f'no_standstill_start (wheel {w}: {(te[j0] - te[idx[0]]) * 1e-6:.3f} s of ESP32 lines at '
                             'rest)')
            return res
        e1 = wh[w].t[0] + wh[w].hi[0] if wh[w].t.size else math.inf
        starts.append(dict(c0=c0, j0=j0, j1=j1, S=S, W=W, first_change_max=max(e1, S + W)))
    uA, uB = te[0], te[-1]
    ms = MapSet(uA, uB, starts[0]['S'] - te[starts[0]['j1']])
    for st in starts:
        ms.add(*ms.rows_ge(te[[st['j1']]], [st['S'] - READ_US]))          # first changed line: read >= the start
        ms.add(*ms.rows_le(te[[st['j0']]], [st['first_change_max']]))     # last unchanged: read before the change
    n_obs = [counts[:, w] - starts[w]['c0'] for w in range(4)]
    seg_of = [np.full(len(te), -1, dtype=np.int64) for _ in range(4)]
    O_set = [list(range(-wh[w].first_m, wh[w].first_m + 1)) for w in range(4)]
    rows = {}                                     # (w, O) -> rows
    combos = [None]
    box_prev = None
    n_iter = 0

    def combo_rows(cb):
        return [rows[(w, cb[w])] for w in range(4)]

    def ext(u, cbs):
        lo, hi = math.inf, -math.inf
        for cb in cbs:
            e = ms.extent(u, () if cb is None else combo_rows(cb))
            if e is not None:
                lo, hi = min(lo, e[0]), max(hi, e[1])
        return None if lo == math.inf else (lo, hi)

    while True:
        n_iter += 1
        eA, eB = ext(uA, combos), ext(uB, combos)
        if eA is None or eB is None:
            res.update(reason='inconsistent (no map agrees with every count)', n_iter=n_iter)
            return res
        # localise lines: their read interval inside one segment of the wheel
        l = ms.lam(te)
        s_lo = np.minimum((1 - l) * eA[0] + l * eB[0], (1 - l) * eA[1] + l * eB[1])
        s_hi = np.maximum((1 - l) * eA[0] + l * eB[0], (1 - l) * eA[1] + l * eB[1])
        # (1 - l) eA + l eB is the extent of s at u only between uA and uB: there l is in [0, 1]
        added = 0
        for w in range(4):
            free = (seg_of[w] < 0) & np.isfinite(counts[:, w])
            if not free.any():
                continue
            sg = wh[w].segment_of(s_lo, s_hi + READ_US)
            new = free & (sg >= 0)
            seg_of[w][new] = sg[new]
            added += int(new.sum())
        keep = []
        for w in range(4):
            ok_O = []
            for O in O_set[w]:
                r = _wheel_rows(ms, wh[w], te, n_obs[w], seg_of[w], O)
                if r is not None and ms.feasible([r]):
                    rows[(w, O)] = r
                    ok_O.append(O)
            if not ok_O:
                res.update(reason=f'inconsistent (wheel {w}: no first-start offset fits the counts)', n_iter=n_iter)
                return res
            keep.append(ok_O)
        O_set = keep
        n_cb = int(np.prod([len(x) for x in O_set]))
        if n_cb > MAX_COMBOS:
            res.update(reason=f'offsets ambiguous ({[len(x) for x in O_set]} choices per wheel)', n_iter=n_iter)
            return res
        import itertools
        combos = [cb for cb in itertools.product(*O_set) if ms.feasible(combo_rows(cb))]
        if not combos:
            res.update(reason='inconsistent (no map agrees with every count)', n_iter=n_iter)
            return res
        box = (ext(uA, combos), ext(uB, combos))
        if added == 0 and box_prev is not None and all(abs(x - y) <= 0.5 for p, q in zip(box, box_prev)
                                                        for x, y in zip(p, q)):
            break
        box_prev = box
        if n_iter >= MAX_ITER:
            res.update(reason=f'set did not settle in {MAX_ITER} passes', n_iter=n_iter)
            return res
    res['n_iter'] = n_iter
    res['offsets'] = [list(cb) for cb in combos]
    res['n_lines'] = [int((seg_of[w] >= 0).sum()) for w in range(4)]
    res['n_rows'] = int(sum(len(ms.b[i]) for i in range(len(ms.b)))
                        + max(sum(len(rows[(w, cb[w])][1]) for w in range(4)) for cb in combos))
    E = lambda u: ext(u, combos)                     # noqa: E731
    u0, u1 = uA, uB

    def mid_line(u0, u1):
        e0, e1 = E(u0), E(u1)
        m0, m1 = 0.5 * (e0[0] + e0[1]), 0.5 * (e1[0] + e1[1])
        b = (m1 - m0) / (u1 - u0)
        return m0 - b * u0, b, e0, e1
    a, b, e0, e1 = mid_line(u0, u1)
    if t_window is not None:
        for _ in range(2):                           # the window depends on the map: twice is enough (b ~ 1)
            u0, u1 = t_window(a, b)
            a, b, e0, e1 = mid_line(u0, u1)
        res['window_u'] = [float(u0), float(u1)]
    res['spread_us'] = [float(e0[1] - e0[0]), float(e1[1] - e1[0])]
    res['a'], res['b'] = float(a), float(b)
    res['ext'] = E
    if max(res['spread_us']) > SPREAD_MAX_US:
        res['reason'] = f'spread {max(res["spread_us"]):.0f} us > {SPREAD_MAX_US:.0f} us'
        return res
    res['ok'] = True
    return res


# ---------------------------------------------------------------------------------------------------------------
# Plant rows, sender t = 0, run

def plant_rows(te_us, steps, state):
    """Rows of PLANT_COLUMNS at the ESP32 times te_us from fractional plant steps (NaN: empty row)."""
    th = np.unwrap(state[:, 6])
    n = len(state)
    rows = []
    for t, k in zip(te_us, steps):
        if not (math.isfinite(k) and 0 <= k <= n - 1):
            rows.append([int(t)] + [''] * (len(PLANT_COLUMNS) - 1))
            continue
        v = [float(np.interp(k, np.arange(n), state[:, i])) for i in range(10)]
        theta = float(np.interp(k, np.arange(n), th))
        c, s = math.cos(theta), math.sin(theta)
        vxw, vyw, wz = v[7], v[8], v[9]
        vxb, vyb = c * vxw + s * vyw, -s * vxw + c * vyw
        rows.append([int(t)] + [f'{x:.6f}' for x in (v[4], v[5], math.atan2(s, c), vxw, vyw, vxb, vyb, wz)]
                    + [f'{x:.5f}' for x in v[0:4]])
    return rows


def sender_frame(ref, t_send_ns, k_a):
    """(frame g whose send time is the sample-0 time, Pi offset us from it): the first frame with a non-zero command
    carries sample k_a; sample 0 was emitted k_a x 20 ms earlier on the Pi clock. None when no non-zero command."""
    nz = np.flatnonzero(np.any(np.abs(ref) > 0, axis=1))
    if nz.size == 0:
        return None
    f = int(nz[0])
    ts = np.asarray(t_send_ns, dtype=np.int64)
    target = ts[f] - int(round(k_a * 1e9 / s02_cmd_path.SEND_HZ))
    g = int(np.argmin(np.abs(ts - target)))
    return g, (target - ts[g]) * 1e-3


def map_run(frame_log, esp_log, test_id):
    """(result dict, plant rows) of one HIL run (module docstring)."""
    import hil_o1_score as H
    import s02_metrics
    frames = EM.load_frames(frame_log)
    with np.load(frame_log, allow_pickle=False) as z:
        has_true = 'mosi_true' in z.files
    log = s02_metrics.load_log(esp_log)['_all']
    on = log['motor_on'] == 1
    te_us = log['timestamp_us'][on]
    counts = np.stack([log[w] for w in WHEELS], axis=1)[on].astype(float)
    counts[counts == EM.LOG_CNT_INVALID] = np.nan
    empty = [[int(t)] + [''] * (len(PLANT_COLUMNS) - 1) for t in te_us]
    res = dict(test=test_id, frame_log=os.path.basename(frame_log), esp_log=os.path.basename(esp_log),
               method='set_of_maps_h7_clock', ok=False, reason='')
    us_cut = s02_cmd_path.us_cut_ticks(esp_log + '.events.txt')
    res['us_cut_ticks'] = us_cut
    if us_cut is not None and us_cut > 0:
        res['reason'] = 'us_cut'                     # Liam 15/11 CODE_S02: no t = 0, no family R
        return res, empty
    if has_true:
        res['reason'] = 'faults_not_supported'       # O-3: the H7 got patched frames
        return res, empty
    fwr = EM.fw_check(EM.read_fw(esp_log + '.events.txt') if os.path.isfile(esp_log + '.events.txt') else None)
    if fwr:
        res['reason'] = f'firmware: {fwr}'
        return res, empty
    mosi, miso = frames['mosi'], frames['miso']
    if mosi is None or miso is None:
        res['reason'] = 'frame log without mosi / miso'
        return res, empty
    n = len(mosi)
    h7 = EM.h7_apply(frames, 0, n - EM.LAG_MAX)
    if not h7.get('ok'):
        res['reason'] = f"h7_apply: {h7.get('reason')}"
        return res, empty
    applied = h7['applied']
    om32 = EM.h7_omega32(mosi)[applied]
    state, ref, _, _ = H.decode_frames(mosi, miso)
    K, k_a, _ = s02_cmd_path.schedule(test_id)
    sf = sender_frame(ref, frames['t_send_ns'], k_a)
    if sf is None:
        res['reason'] = 'no_sender_frame'
        return res, empty
    g, dt_us = sf
    s0 = float(h7['t_frame_us'][g]) + dt_us if math.isfinite(h7['t_frame_us'][g]) else math.nan
    if not math.isfinite(s0):
        res['reason'] = 'sender frame outside the H7 answers'
        return res, empty
    T_us = s02_cmd_path.GROUP_A[test_id].duration * 1e6

    def window(a, b):
        u0 = (s0 - a) / b
        return u0, u0 + T_us
    cert = certify(te_us, counts, h7['apply_us'][applied], h7['width_us'][applied], om32, window)
    for k in ('reason', 'a', 'b', 'spread_us', 'offsets', 'n_rows', 'n_iter', 'window_u'):
        if k in cert:
            res[k] = cert[k]
    res['h7_window_us'] = h7['window_us']
    res['n_not_applied'] = h7['n_not_applied']
    if cert['a'] is not None:
        a, b = cert['a'], cert['b']
        u0 = (s0 - a) / b
        e = cert['ext'](u0)
        res['sender_t0_us'] = float(u0)
        res['sender_spread_us'] = float(e[1] - e[0])
        res['sender_frame'] = g
    if not cert['ok']:
        return res, empty
    s = cert['a'] + cert['b'] * te_us
    steps, gap = np.full(len(s), np.nan), 0.0
    for i, x in enumerate(s):
        lo, hi = EM.h7_to_steps(x, h7['anchor_k'], h7['anchor_us'])
        if not (h7['anchor_us'][0] <= x <= h7['anchor_us'][-1]):
            continue
        steps[i] = 0.5 * (lo + hi)
        gap = max(gap, 0.5 * (hi - lo))
    res['step_gap_max'] = gap
    res['ok'] = True
    return res, plant_rows(te_us, steps, state)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame-log', required=True)
    ap.add_argument('--esp-log', required=True)
    ap.add_argument('--test', required=True, choices=sorted(s02_cmd_path.GROUP_A))
    ap.add_argument('--out', required=True, help='<run>_plant.csv (new file); the map goes to <run>_plant_map.json')
    a = ap.parse_args(argv)
    js = os.path.splitext(a.out)[0] + '_map.json'
    for p in (a.out, js):
        if os.path.exists(p):
            print(f'{p} already exists: outputs are not overwritten')
            print('#ket_qua,ma=2')
            return 2
    try:
        res, rows = map_run(a.frame_log, a.esp_log, a.test)
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}')
        print('#ket_qua,ma=2')
        return 2
    with open(a.out, 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(PLANT_COLUMNS)
        w.writerows(rows)
    with open(js, 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('ok', 'reason', 'spread_us', 'sender_spread_us', 'offsets')},
                     default=float))
    code = 0 if res.get('ok') else 1
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
