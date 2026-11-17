"""s02_plant_ticks.py: the set of ESP32 -> H7 maps that agree with every logged encoder count (CODE_S02 design 28,
28.5; Liam 16/11 CODE_S02 local_95df9cdf "Tap kha di (Recommended)", "Dung so de xuat (Recommended)", "Dung gio H7
do (Recommended)").

Counts come from the unchanged H7 encoder_gen.c on the PC (tools/tests/encgen_oracle.py, profile mode), never from
the module's own edge model; frame logs are made by the helpers of test_hil_edge_map.py (imported, not copied).
Covered: the edge simulation edge by edge against the oracle; the edge intervals hold the true edges when the true
apply lies anywhere in its window; certify on made-up runs (the true map inside the set, the true start offsets
kept); the refusals; map_run on made-up frame logs (truth known) and on the real CLOSED_LOOP/2026-11-14_run01
(skipped without the data; S02_REAL_RUN overrides its path). Every number of the made-up runs is invented."""
import csv
import json
import math
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import encgen_oracle as O                 # noqa: E402
import hil_edge_map as EM                 # noqa: E402
import s02_cmd_path                       # noqa: E402
import s02_plant_ticks as pt              # noqa: E402

needs_gcc = pytest.mark.skipif(not O.available(), reason='gcc not on PATH (H7 EncoderGen oracle)')
ROOT = os.path.join(HERE, '..', '..')
FW_LINE = f'2026-11-16T10:00:00 #fw,sha={EM.FW_COUNTS_AT_TICK[0]},dirty=0,env=esp32-s3-devkitc-1\n'
REAL = os.environ.get('S02_REAL_RUN', r'D:\CLAUDE_THESIS\05_DATA\results\CLOSED_LOOP\2026-11-14_run01')
EMPTY = [''] * (len(pt.PLANT_COLUMNS) - 1)


def om32_of(om):
    """Wheel speeds as the H7 gets them: int16 of the MOSI (HIL_OMEGA_MAX 40), float32 (EM.h7_omega32)."""
    raw = np.round(np.clip(om, -39.99, 39.99) / EM.HIL_OMEGA_MAX * 32767).astype(np.int16)
    return raw.astype(np.float32) * (np.float32(EM.HIL_OMEGA_MAX) / np.float32(32767.0))


def profile(n, seed, k_move=1500, kind='ramp'):
    """Wheel speeds (n, 4) per 1 ms frame: rest, a start, a reversal, a stop, a restart, then steps 3 <-> 5 rad/s
    every 50 frames (retimes of both signs). kind 'ramp': a 300 ms ramp from 0 with a 0.3 rad/s ripple and 0.02 rad/s
    noise on every frame (slow start, a retime on every frame: the hard case for the edge intervals); 'step': a
    first-order start (60 ms) with a 0.1 rad/s ripple and 0.005 rad/s noise (as a plant answering a step)."""
    rng = np.random.default_rng(seed)
    k = np.arange(n).astype(float)
    j = np.clip(k - k_move, 0, None)
    if kind == 'ramp':
        base = np.clip(j / 300.0, 0, 1) * 8.0 + 0.3 * np.sin(k / 37.0) * (k > k_move)
        rev = -6.0 * np.clip((k - k_move - 2500) / 100, 0, 1)
        noise = 0.02
    else:
        base = 8.0 * (1 - np.exp(-j / 60.0)) + 0.1 * np.sin(j / 37.0)
        rev = -6.0 * (1 - np.exp(-(k - k_move - 2500) / 60.0))
        noise = 0.005
    base = np.where((k > k_move + 2500) & (k < k_move + 2900), rev, base)                  # reversal
    base = np.where((k >= k_move + 2900) & (k < k_move + 3100), 0.0, base)                  # stop
    base = np.where(k >= k_move + 4500, 3.0 + 2.0 * ((k // 50) % 2), base)                  # steps
    om = np.zeros((n, 4))
    for w, f in enumerate((1.0, -1.07, 0.94, -1.02)):
        om[:, w] = base * f + rng.normal(0, noise, n) * (np.abs(base) > 0)
    om[k <= k_move] = 0.0
    return om


def apply_times(n, seed, kind='harsh'):
    """(window start lo, width, true apply) per frame, H7 us: frames about 1 ms apart, the true apply anywhere in its
    window, integer, increasing. kind 'harsh': 1..80 us for 30 % of the frames (as the 'Apply:' line of
    CLOSED_LOOP/2026-11-13_run14), else 1..5 us; 'real': as the H7 windows of CLOSED_LOOP/2026-11-14_run01 (k1a, k1b:
    median 4 us, 99 % <= 12 us, max 60 us; k2: 9 % over 20 us): 90 % 1..8 us, 9 % 8..24 us, 1 % 24..60 us; 'exact':
    the true apply is the window start."""
    rng = np.random.default_rng(seed + 100)
    lo = np.cumsum(np.full(n, 1000.0)) + rng.integers(0, 300, n)
    for i in range(1, n):
        lo[i] = max(lo[i], lo[i - 1] + 100)
    p = rng.uniform(size=n)
    if kind == 'harsh':
        win = np.where(p < 0.3, rng.uniform(1, 80, n), rng.uniform(1, 5, n))
    elif kind == 'real':
        win = np.where(p < 0.9, rng.uniform(1, 8, n), np.where(p < 0.99, rng.uniform(8, 24, n), rng.uniform(24, 60, n)))
    else:
        win = np.zeros(n)
    true = np.floor(lo + rng.uniform(0, 1, n) * win)
    for i in range(1, n):
        true[i] = max(true[i], true[i - 1] + 1)
    return lo, win + 1.0, true


# ---------------------------------------------------------------------------------------------------------------
# EncoderGen edge by edge

def test_constants_match_encoder_gen():
    src = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Src', 'encoder_gen.c'), encoding='utf-8').read()
    m = re.search(r'ENC_DIR_SIGN\[ENC_NUM_WHEELS\]\s*=\s*\{([^}]*)\}', src)
    assert tuple(int(x) for x in m.group(1).replace('+', '').split(',')) == pt.SIGN
    assert int(re.search(r'#define ENC_RETIME_MARGIN (\d+)u', src).group(1)) == pt.RETIME_MARGIN
    assert float(re.search(r'#define ENC_OMEGA_MIN ([0-9.]+)f', src).group(1)) == pt.OMEGA_MIN


def test_start_by_hand_and_the_stop_threshold():
    """encoder_gen.c _start_wheel: the leading channel's first edge one half period after the apply, the other half a
    half period later (integer hp >> 1); |omega| <= 0.3 rad/s makes no edge."""
    T0 = 5000.0
    om = np.zeros((60, 4))
    om[2:] = [10.0, -10.0, 0.299, 0.301]          # int16 245 -> 0.29908 rad/s, 246 -> 0.30030 rad/s on the H7
    ap = T0 + 1000.0 * np.arange(60)                # 59 ms: one half period at 0.301 rad/s is 31.6 ms
    assert om32_of(om)[2, 2] < np.float32(0.3) < om32_of(om)[2, 3]
    ed = pt.simulate_edges(ap, np.zeros(60), om32_of(om))
    hp, _ = EM.h7_half_period(om32_of(om))
    for w in (0, 1):                     # w0: dir +1, sign +1 -> A leads; w1: dir -1, sign -1 -> A leads
        h = float(hp[2, w])
        assert ed[w]['t'][0] == ap[2] + h and ed[w]['t'][1] == ap[2] + h + float(int(h) >> 1)
        assert ed[w]['segs'][1]['kind'] == 'start' and ed[w]['segs'][1]['o_hi'] == 2      # levels unknown: |o| <= 2
    assert len(ed[2]['t']) == 0 and len(ed[2]['segs']) == 1          # 0.3 rad/s: no start
    assert len(ed[3]['t']) > 0


@needs_gcc
@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_simulated_edges_match_the_h7_code(seed):
    """With the apply times known (zero width) every simulated edge is an edge of encoder_gen.c and every count change
    of encoder_gen.c that is not the forced-low jump of a start is a simulated edge (1 us apart: the compare fires on
    the tick before the simulated time); every forced-low jump is within the start's |o| bound."""
    n = 7000
    om32 = om32_of(profile(n, seed))
    _, _, app = apply_times(n, seed, kind="exact")
    counts = O.run(app.astype(np.int64), om32.astype(np.float64), seed)
    ed = pt.simulate_edges(app, np.zeros(n), om32)
    n_edges = 0
    for w in range(4):
        e = ed[w]
        keep = counts.t[w] <= app[-1]
        ot, oc = counts.t[w][keep], counts.c[w][keep]
        starts = [s for s in e['segs'] if s['kind'] == 'start']
        assert len(starts) >= 3                                     # first start, after the reversal, after the stop
        st = {int(s['t0']) for s in starts}
        sim = {int(t) - 1 for t in e['t']}
        orc = {int(t) for t in ot} - st
        assert sim == orc, (w, sorted(sim ^ orc)[:5])
        sgn = np.array([e['segs'][s]['dir'] for s in e['seg']])
        assert np.all(np.abs(sgn) == 1)
        for s in starts:
            i = int(np.searchsorted(ot, s['t0']))
            if i < len(ot) and ot[i] == s['t0']:
                jump = int(oc[i] - (oc[i - 1] if i > 0 else 0))
                assert s['o_lo'] <= jump <= s['o_hi']
        n_edges += len(e['t'])
    assert n_edges > 20000


@needs_gcc
@pytest.mark.parametrize('seed', [1, 4, 7, 9])
def test_edge_intervals_hold_the_true_edges(seed):
    """Apply windows (true apply anywhere inside): every trusted edge's interval [t + lo, t + hi] holds the true edge.
    The true edges are the simulation with the true apply times, itself checked against encoder_gen.c here. Seeds 4
    and 7 have edges past the trusted ones whose intervals miss (test_untrusted_edges_can_miss)."""
    n = 7000
    om32 = om32_of(profile(n, seed))
    lo_app, width, true_app = apply_times(n, seed)
    ex = pt.simulate_edges(true_app, np.zeros(n), om32)
    counts = O.run(true_app.astype(np.int64), om32.astype(np.float64), seed)
    ed = pt.simulate_edges(lo_app, width, om32)
    used = 0
    for w in range(4):
        st = {int(s['t0']) for s in ex[w]['segs'] if s['kind'] == 'start'}
        keep = counts.t[w] <= true_app[-1]
        assert {int(t) - 1 for t in ex[w]['t']} == {int(t) for t in counts.t[w][keep]} - st
        wh = pt._Wheel(ed[w])
        a, b = ex[w], ed[w]
        assert len(a['segs']) == len(b['segs'])
        for s in range(1, len(b['segs'])):
            ia, ib = np.flatnonzero(a['seg'] == s), np.flatnonzero(b['seg'] == s)
            m = min(len(ia), len(ib))
            d = a['t'][ia[:m]] - b['t'][ib[:m]]
            tr = wh.trusted[ib[:m]]
            bad = tr & ((d < b['lo'][ib[:m]] - pt.TOL_US) | (d > b['hi'][ib[:m]] + pt.TOL_US))
            assert not bad.any(), (w, s, int(np.flatnonzero(bad)[0]))
            used += int(tr.sum())
    assert used > 5000



def test_untrusted_edges_can_miss():
    """Why an edge past the first wide interval of its segment bounds nothing (_Wheel.trusted): with seed 7 some of
    those edges are outside their interval (which simulated edge is which true edge is no longer certain); none of
    the trusted ones is."""
    n = 7000
    om32 = om32_of(profile(n, 7))
    lo_app, width, true_app = apply_times(n, 7)
    ex = pt.simulate_edges(true_app, np.zeros(n), om32)
    ed = pt.simulate_edges(lo_app, width, om32)
    miss_untrusted = miss_trusted = 0
    for w in range(4):
        wh = pt._Wheel(ed[w])
        a, b = ex[w], ed[w]
        for s in range(1, len(b['segs'])):
            ia, ib = np.flatnonzero(a['seg'] == s), np.flatnonzero(b['seg'] == s)
            m = min(len(ia), len(ib))
            d = a['t'][ia[:m]] - b['t'][ib[:m]]
            bad = (d < b['lo'][ib[:m]] - pt.TOL_US) | (d > b['hi'][ib[:m]] + pt.TOL_US)
            tr = wh.trusted[ib[:m]]
            miss_trusted += int((bad & tr).sum())
            miss_untrusted += int((bad & ~tr).sum())
    assert miss_trusted == 0 and miss_untrusted > 0

# ---------------------------------------------------------------------------------------------------------------
# certify on made-up runs

PRIOR = 400                                        # frames of an earlier motion, not in the frame log given


def fake_run(seed, ppm=0.0, n=7000, rest_frames=1500, esp_rest_s=1.4, flip=None, bump=None, windows='real',
             kind='step'):
    """A made-up run: the H7 frames (an earlier motion that leaves the pins at some level, rest, then profile()), the
    ESP32 lines on their own clock (H7 time s = s_first + (1 + ppm) (u - u0)), counts read 0..50 us after the tick
    time from encoder_gen.c; wheel speeds of profile(kind=kind), apply windows of apply_times(kind=windows). Returns the inputs of certify and the truth."""
    rng = np.random.default_rng(seed + 7)
    om = np.zeros((PRIOR + n, 4))
    for w in range(4):                                     # earlier motion of a random length per wheel
        L = int(rng.integers(100, PRIOR - 50))
        om[20:20 + L, w] = (1.0, -1.0, 1.0, -1.0)[w] * rng.uniform(4.0, 9.0)
    om[PRIOR:] = profile(n, seed, k_move=rest_frames, kind=kind)
    om32 = om32_of(om)
    lo_app, width, true_app = apply_times(PRIOR + n, seed, windows)
    counts = O.run(true_app.astype(np.int64), om32.astype(np.float64), seed)
    beta = 1.0 + ppm * 1e-6
    s_first = lo_app[PRIOR + rest_frames] - esp_rest_s * 1e6                  # first ESP32 line, H7 us
    u0 = 3.0e6
    s_last = lo_app[-1] - 30_000.0
    nl = int((s_last - s_first) / beta / 10_000.0)
    u = u0 + np.arange(nl) * 10_000.0 + rng.uniform(-300, 300, nl)
    u[rng.choice(np.arange(1, nl - 1), nl // 30, replace=False)] += 5_000.0     # late ticks
    u = np.sort(u)

    def s_true(x):
        return s_first + beta * (np.asarray(x, dtype=float) - u0)
    read = s_true(u) + rng.uniform(0.0, EM.J_READ_US, nl)
    c = counts.at(np.round(read).astype(np.int64)).astype(float)
    if flip is not None:
        c[:, flip] = -c[:, flip]
    if bump is not None:                                                        # (wheel, from line, to line)
        w, j0, j1 = bump
        c[j0:j1, w] += 1
    # the true first-start offset per wheel: the oracle's forced-low jump at the first start in the log
    ed = pt.simulate_edges(lo_app[PRIOR:], width[PRIOR:], om32[PRIOR:])
    o_true = []
    for w in range(4):
        i = PRIOR + int(np.searchsorted(lo_app[PRIOR:], ed[w]['segs'][1]['t0']))
        T = int(true_app[i])
        o_true.append(int(counts.at([T])[0, w] - counts.at([T - 1])[0, w]))
    return dict(te=u, counts=c, apply_us=lo_app[PRIOR:], width_us=width[PRIOR:], om32=om32[PRIOR:],
                s_true=s_true, o_true=o_true)


def certify(f, **kw):
    return pt.certify(f['te'], f['counts'], f['apply_us'], f['width_us'], f['om32'], **kw)


def check_truth(f, res):
    """The true map lies in the set at every line; the midpoint map is off by at most half the spread there."""
    lo, hi = np.array([res['ext'](u) for u in f['te']]).T
    s = f['s_true'](f['te'])
    assert np.all((lo - 1e-6 <= s) & (s <= hi + 1e-6)), float(np.max(np.maximum(lo - s, s - hi)))
    mid = res['a'] + res['b'] * f['te']
    assert np.max(np.abs(mid - s)) <= 0.5 * max(res['spread_us']) + 1e-6


@needs_gcc
@pytest.mark.parametrize('seed,ppm', [(1, 0.0), (2, -900.0), (3, 900.0), (4, 450.0)])
def test_certify_holds_the_true_map(seed, ppm):
    f = fake_run(seed, ppm)
    res = certify(f)
    assert res['ok'], res['reason']
    assert max(res['spread_us']) <= pt.SPREAD_MAX_US
    check_truth(f, res)
    kept = [{cb[w] for cb in res['offsets']} for w in range(4)]
    assert all(f['o_true'][w] in kept[w] for w in range(4)), (f['o_true'], res['offsets'])
    assert abs(res['b'] - (1.0 + ppm * 1e-6)) * 1e6 < 200.0


@needs_gcc
@pytest.mark.parametrize('seed', [1, 3])
def test_harsh_windows_keep_the_truth_in_the_set(seed):
    """Windows up to 80 us on 30 % of the frames (far wider than run01) and the slow noisy start: the set is wide
    (2-4 ms, not certified) but still holds the true map at every line."""
    f = fake_run(seed, 600.0, windows='harsh', kind='ramp')
    res = certify(f)
    # the whole run (reversal, stop, restart, untrusted edges): the rest bound may not hold (S2M2-1), the start
    # offsets may then stay ambiguous: a refusal; never a set without the truth
    assert not res['reason'].startswith('inconsistent'), res['reason']
    if res.get('ext') is not None:
        lo, hi = np.array([res['ext'](u) for u in f['te']]).T
        s = f['s_true'](f['te'])
        assert np.all((lo - 1e-6 <= s) & (s <= hi + 1e-6))
    # up to just before the reversal (frames and lines): the set exists and holds the truth
    k = 1500 + 2400
    keep = f['s_true'](f['te']) < f['apply_us'][k] - 30_000.0
    cut = dict(f, te=f['te'][keep], counts=f['counts'][keep], apply_us=f['apply_us'][:k], width_us=f['width_us'][:k],
               om32=f['om32'][:k])
    res = certify(cut)
    assert res.get('ext') is not None, res['reason']
    lo, hi = np.array([res['ext'](u) for u in cut['te']]).T
    s = cut['s_true'](cut['te'])
    assert np.all((lo - 1e-6 <= s) & (s <= hi + 1e-6))


@needs_gcc
def test_the_earlier_motion_leaves_nonzero_start_offsets():
    """The made-up runs do exercise |o| > 0 (else the offset search would not be tested)."""
    assert any(any(o != 0 for o in fake_run(seed)['o_true']) for seed in (1, 2, 3, 4))


@needs_gcc
def test_more_motion_narrows_the_set():
    """The set at the first line does not widen when the log goes on (more constraints, same set or smaller)."""
    f = fake_run(2, 300.0)
    keep = f['te'] < f['te'][0] + 3.0e6
    short = dict(f, te=f['te'][keep], counts=f['counts'][keep])
    r_long, r_short = certify(f), certify(short)
    assert r_long['ok'] and r_short['ok']
    u = f['te'][0]
    lo_l, hi_l = r_long['ext'](u)
    lo_s, hi_s = r_short['ext'](u)
    assert hi_l - lo_l <= hi_s - lo_s + 1e-6


@needs_gcc
def test_a_wheel_with_the_wrong_sign_is_refused():
    res = certify(fake_run(1, flip=1))
    assert not res['ok'] and res['reason'].startswith('inconsistent'), res['reason']


@needs_gcc
def test_a_count_change_during_the_rest_is_refused():
    """A count that changes while every frame is at rest: early in the rest -> not 1 s of rest; late -> no map."""
    f = fake_run(1)
    res = certify(dict(f, counts=f['counts'] + np.where(np.arange(len(f['te']))[:, None] >= 30, [0, 0, 1, 0], 0)))
    assert not res['ok'] and res['reason'].startswith('no_standstill_start'), res['reason']
    res = certify(fake_run(1, bump=(2, 120, 130)))                       # 1.2 s into the rest, back again
    assert not res['ok'], res['reason']


@needs_gcc
def test_less_than_one_second_of_rest_is_refused():
    res = certify(fake_run(3, rest_frames=900, esp_rest_s=0.8))         # frames: 0.9 s at rest
    assert not res['ok'] and res['reason'].startswith('no_standstill_start') and 'frames at rest' in res['reason']
    res = certify(fake_run(3, esp_rest_s=0.7))                          # ESP32 lines: 0.7 s at rest
    assert not res['ok'] and res['reason'].startswith('no_standstill_start') and 'ESP32 lines' in res['reason']


@needs_gcc
def test_the_spread_bound_refuses(monkeypatch):
    f = fake_run(4)
    res = certify(f)
    assert res['ok']
    monkeypatch.setattr(pt, 'SPREAD_MAX_US', 0.5 * min(res['spread_us']))
    res = certify(f)
    assert not res['ok'] and res['reason'].startswith('spread')


def test_a_wheel_that_never_starts_is_refused():
    n = 3000
    om = np.zeros((n, 4))
    om[1500:, 1:] = 5.0
    ap = 1000.0 * np.arange(n)
    te = np.arange(0, 2.9e6, 1e4)
    res = pt.certify(te, np.zeros((len(te), 4)), ap, np.ones(n), om32_of(om))
    assert not res['ok'] and res['reason'] == 'wheel 0 never starts in the frame log'
    om[1500:, 0] = 5.0                                                 # every wheel starts, no count ever changes
    res = pt.certify(te, np.zeros((len(te), 4)), ap, np.ones(n), om32_of(om))
    assert not res['ok'] and res['reason'] == 'wheel 0: the count never changes'


# ---------------------------------------------------------------------------------------------------------------
# plant rows, sender frame

def test_plant_rows_interpolate_and_unwrap():
    st = np.zeros((3, 10))
    st[:, 6] = [math.pi - 0.01, -math.pi + 0.01, -math.pi + 0.03]    # theta wraps between frames 0 and 1
    st[:, 4] = [0.0, 1.0, 2.0]
    st[:, 7] = [1.0, 1.0, 1.0]                                        # world vx
    st[:, 0] = [2.0, 4.0, 6.0]
    rows = pt.plant_rows(np.array([10, 20, 30, 40]), np.array([0.5, 1.25, np.nan, 2.5]), st)
    r = rows[0]
    assert float(r[1]) == pytest.approx(0.5) and float(r[9]) == pytest.approx(3.0)
    assert abs(abs(float(r[3])) - math.pi) < 1e-6                     # half way: +-pi, not 0
    assert float(r[6]) == pytest.approx(math.cos(float(r[3])), abs=1e-6)   # body vx
    assert float(rows[1][1]) == pytest.approx(1.25)
    for k in (2, 3):                                                  # no step, or past the last frame: empty
        assert rows[k][0] == [10, 20, 30, 40][k] and rows[k][1:] == EMPTY


def test_sender_frame_steps_back_k_a_packets_of_pi_time():
    ts = (np.arange(10000) * 1_000_000 + 7).astype(np.int64)          # 1 ms frames
    ref = np.zeros((10000, 3))
    ref[6000:, 0] = 0.3
    g, dt = pt.sender_frame(ref, ts, 150)
    assert g == 3000 and dt == pytest.approx(0.0)
    ts2 = ts + np.where(np.arange(10000) >= 3000, 400_000, 0)          # the Pi late by 0.4 ms from frame 3000
    g, dt = pt.sender_frame(ref, ts2, 150)
    assert ts2[g] * 1e-3 + dt == pytest.approx(ts2[6000] * 1e-3 - 3.0e6)
    assert pt.sender_frame(np.zeros((10, 3)), ts[:10], 0) is None


# ---------------------------------------------------------------------------------------------------------------
# map_run on made-up frame logs (helpers of test_hil_edge_map)

def _bench(tmp_path, ppm, edge_ms=1600.0, n=8200, us_cut=None, fw=FW_LINE, faults=False, skip=()):
    """Frame log (npz) and ESP32 log of a made-up HIL run whose command starts with the motion (frame edge_ms);
    returns the paths, the ESP32 log columns, the made-up run, the truth s_true(u) on the H7 axis of
    hil_edge_map.h7_apply, and h7_apply."""
    import protocol as P
    import test_hil_edge_map as T
    s = T.sim(edge_ms=edge_ms, n=n, ppm=ppm, n_before=16, glitch_every=0, seed=3, skip=skip)
    tm = s['timing']
    k_cmd = int(edge_ms)
    om = tm['om']
    mosi = np.array([T.mosi_frame(list(om[k]) + [0.0] * 6, k,
                                  traj_ref=(0.3, 0.0, 0.0) if k >= k_cmd else None,
                                  traj_flags=P.TRAJ_VALID if k >= k_cmd else 0) for k in range(n)])
    assert np.array_equal(mosi[:, 0:8], s['frames']['mosi'][:, 0:8])          # same wheel speeds as the counts
    miso = T.make_miso(mosi, s['duty_cap'], tm, skip=skip)
    # Pi t_recv after the transfer: after the H7 receive (rx, 384 us of bus + 0..50 us after the send T), plus the
    # return of the call (10..60 us)
    rng = np.random.default_rng(11)
    t_recv = s['frames']['t_send_ns'] + np.round((tm['rx'] - tm['T'] + rng.uniform(10.0, 60.0, n)) * 1000.0).astype(np.int64)
    frames = dict(s['frames'], mosi=mosi, miso=miso, t_recv_ns=t_recv)
    fl = str(tmp_path / 'r_frames.npz')
    np.savez(fl, **frames, **(dict(mosi_true=mosi) if faults else {}))
    esp = str(tmp_path / 'r.csv')
    e = s['esp']
    log = {'timestamp_us': e['timestamp_us'].astype(float), 'motor_on': np.ones(len(e['timestamp_us'])),
           **{w: e['counts'][:, i] for i, w in enumerate(pt.WHEELS)}}
    with open(esp + '.events.txt', 'w', encoding='utf-8') as f:
        if fw:
            f.write(fw)
        if us_cut is not None:
            f.write(f'2026-11-16T10:00:09 #us_end,run=1,cut_ticks={us_cut},qseq=9\n')
    h7 = EM.h7_apply(frames, 0, n - EM.LAG_MAX)
    assert h7['ok'], h7
    k = h7['applied']
    off = h7['apply_us'][k] - (tm['rx'][k] + tm['pre'][k])
    assert np.ptp(off) < 1e-3                                      # the H7 axis is the made-up axis shifted
    t_e, ts_e = edge_ms * 1000.0, 5_000_000.0

    def s_true(u):
        return t_e + (np.asarray(u, dtype=float) - ts_e) * (1.0 + ppm * 1e-6) + off[0]
    return fl, esp, log, s, s_true, h7


@needs_gcc
@pytest.mark.parametrize('ppm', [-900.0, 900.0])
def test_map_run_on_a_made_up_bench_run(monkeypatch, tmp_path, ppm):
    import s02_metrics
    fl, esp, log, s, s_true, h7 = _bench(tmp_path, ppm)
    monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {'_all': log})
    res, rows = pt.map_run(fl, esp, 'A-L2')
    assert res['ok'], res['reason']
    u0, u1 = res['window_u']
    te = log['timestamp_us']
    inw = (te >= u0) & (te <= u1)
    assert inw.sum() > 500
    err = res['a'] + res['b'] * te[inw] - s_true(te[inw])
    assert np.max(np.abs(err)) <= 0.5 * max(res['spread_us']) + 1.0
    # sender t = 0: the Pi send of the first frame with a command (k_a = 0; made-up axis T, shifted to the H7 axis
    # as the receives), mapped back by the true map
    g = res['sender_frame']
    assert g == 1600
    send_h7 = s['timing']['T'][g] + (h7['t_frame_us'][g] - s['timing']['rx'][g])
    u_true = 5_000_000.0 + (send_h7 - s_true(5_000_000.0)) / (1.0 + ppm * 1e-6)
    assert res['sender_h7_us'][0] <= send_h7 <= res['sender_h7_us'][1]
    assert abs(res['sender_t0_us'] - u_true) <= 0.5 * res['sender_spread_us'] + 1.0
    assert u0 == pytest.approx(res['sender_t0_us'], abs=1e-3)
    assert u1 - u0 == pytest.approx(s02_cmd_path.GROUP_A['A-L2'].duration * 1e6)
    # S2M1-4: the sender spread (map spread + send interval) is checked on its own: a bound between the two refuses
    if ppm > 0:
        lim = 0.5 * (max(res['spread_us']) + res['sender_spread_us'])
        assert max(res['spread_us']) < lim < res['sender_spread_us']
        monkeypatch.setattr(pt, 'SPREAD_MAX_US', lim)
        r2, rows2 = pt.map_run(fl, esp, 'A-L2')
        assert not r2['ok'] and r2['reason'].startswith('sender spread') and all(r[1] == '' for r in rows2)
        monkeypatch.undo()
        monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {'_all': log})
    # plant rows: filled inside the answered frames; the wheel speed at the mapped step is the true one
    full = np.array([r[1] != '' for r in rows])
    assert full[inw].all()
    om = s['timing']['om']
    for i in np.flatnonzero(inw)[::50]:
        k_true = np.interp(s_true(te[i]), h7['anchor_us'], h7['anchor_k'])
        w1 = np.interp(k_true, np.arange(len(om)), om[:, 0])
        assert float(rows[i][9]) == pytest.approx(w1, abs=0.05)


@needs_gcc
def test_map_run_leaves_rows_empty_across_unanswered_frames(monkeypatch, tmp_path):
    """S2M2-T (Codex round 2 of v3): frames the H7 received but did not answer inside the window: rows whose mapped
    time falls between the answered frames around the gap are empty (n_gap_rows), the others are filled."""
    import s02_metrics
    gap = tuple(range(4000, 4006))
    fl, esp, log, s, s_true, h7 = _bench(tmp_path, 900.0, skip=gap)
    monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {'_all': log})
    res, rows = pt.map_run(fl, esp, 'A-L2')
    assert res['ok'], res['reason']
    ak, au = h7['anchor_k'], h7['anchor_us']
    i = int(np.searchsorted(ak, gap[0]))
    a, b = au[i - 1], au[i]                                           # receives of the answered frames around the gap
    te = log['timestamp_us']
    s_map = res['a'] + res['b'] * te
    inside = (s_map > a) & (s_map < b)
    assert inside.sum() >= 1 and res['n_gap_rows'] == int(inside.sum())
    full = np.array([r[1] != '' for r in rows])
    assert not full[inside].any()
    u0, u1 = res['window_u']
    inw = (te >= u0) & (te <= u1) & ~inside
    assert full[inw].all()


@needs_gcc
def test_map_run_refusals(monkeypatch, tmp_path):
    """us_cut (S2P2-2), a fault run with mosi_true (O-3), firmware not known to read counts at t_now (L98), no #fw
    line, no events file: no map, every plant row empty."""
    import s02_metrics
    cases = [(dict(us_cut=7), 'us_cut'), (dict(faults=True), 'faults_not_supported'),
             (dict(fw=FW_LINE.replace(EM.FW_COUNTS_AT_TICK[0], '0123456789ab')), 'firmware: firmware sha=0123'),
             (dict(fw=''), 'firmware: no #fw line'), (dict(us_cut=0, fw=None), 'firmware: no events file')]
    for i, (kw, why) in enumerate(cases):
        d = tmp_path / f'c{i}'
        d.mkdir()
        fl, esp, log, *_ = _bench(d, 0.0, n=4000, **kw)
        if kw.get('fw', '') is None:
            os.remove(esp + '.events.txt')
        monkeypatch.setattr(s02_metrics, 'load_log', lambda p, log=log: {'_all': log})
        res, rows = pt.map_run(fl, esp, 'A-L2')
        assert not res['ok'] and res['reason'].startswith(why), (kw, res['reason'])
        assert rows and all(r[1:] == EMPTY for r in rows)


@needs_gcc
def test_map_run_refuses_a_run_without_rest(monkeypatch, tmp_path):
    """The command and the motion start 0.4 s into the frame log: no_standstill_start (as k2 of run01)."""
    import s02_metrics
    fl, esp, log, *_ = _bench(tmp_path, 0.0, edge_ms=400.0, n=7000)
    monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {'_all': log})
    res, rows = pt.map_run(fl, esp, 'A-L2')
    assert not res['ok'] and res['reason'].startswith('no_standstill_start'), res['reason']
    assert all(r[1:] == EMPTY for r in rows)


def test_main_writes_both_files_and_never_overwrites(monkeypatch, tmp_path, capsys):
    rows = [[1000, '0.1'] + [''] * (len(pt.PLANT_COLUMNS) - 2)]
    monkeypatch.setattr(pt, 'map_run', lambda f, e, t: (dict(ok=True, reason='', spread_us=[1.0, 2.0]), rows))
    out = str(tmp_path / 'r_plant.csv')
    js = str(tmp_path / 'r_plant_map.json')
    argv = ['--frame-log', 'f.npz', '--esp-log', 'e.csv', '--test', 'A-L2', '--out', out]
    assert pt.main(argv) == 0
    assert capsys.readouterr().out.strip().endswith('#ket_qua,ma=0')
    with open(out, newline='') as f:
        assert next(csv.reader(f)) == pt.PLANT_COLUMNS
    with open(js, encoding='utf-8') as f:
        assert json.load(f)['ok'] is True
    assert pt.main(argv) == 2                                          # outputs exist: not overwritten
    os.remove(out)
    os.remove(js)
    monkeypatch.setattr(pt, 'map_run', lambda f, e, t: (dict(ok=False, reason='us_cut'), rows))
    assert pt.main(argv) == 1


# ---------------------------------------------------------------------------------------------------------------
# the real run CLOSED_LOOP/2026-11-14_run01 (H7 7dd5ab2, ESP32 16702ff; an old run without the 3 s preparation)

@pytest.mark.skipif(not os.path.isdir(REAL), reason='CLOSED_LOOP/2026-11-14_run01 not on this machine')
@pytest.mark.parametrize('run,test,ok', [('k1a', 'A-K1', True), ('k1b', 'A-K1', True), ('k2', 'A-K2', False)])
def test_map_run_on_the_real_run(run, test, ok):
    import s02_metrics
    fl = os.path.join(REAL, 'frames', f'{run}_frames.npz')
    esp = os.path.join(REAL, 'pi', f'{run}.csv')
    res, rows = pt.map_run(fl, esp, test)
    if not ok:
        assert not res['ok'] and res['reason'].startswith('no_standstill_start'), res['reason']
        return
    assert res['ok'], res['reason']
    assert max(res['spread_us']) < 1000.0 and res['sender_spread_us'] < 1000.0     # 0.31-0.45 ms on 16/11
    assert len(res['offsets']) >= 1
    log = s02_metrics.load_log(esp)['_all']
    cmd, _, _ = s02_cmd_path.anchors(log, test)
    lag_ms = (cmd['t0_us'] - res['sender_t0_us']) / 1000.0
    assert 0.0 < lag_ms < 100.0, lag_ms                   # the command anchors come after the sender (path delay)
    assert sum(1 for r in rows if r[1] != '') > 0.9 * len(rows)


# ---------------------------------------------------------------------------------------------------------------
# Codex round 1 of v3 (S2M1-2, -3, -4, -5)

@needs_gcc
def test_counts_past_the_trusted_edges_do_not_exclude_the_true_map():
    """S2M1-2: raw 5199 / 7799 (hp 1500 / 1000 us) on alternate 1 ms frames for 1 s, 4 us windows, hp-1500 frames
    applied at the window end, hp-1000 frames at its start: encoder_gen.c counts 1730 edges where the edge model
    (applies at the window start) has 1484; only the first edges are trusted. The set must still hold the true map
    (or refuse), never exclude it."""
    n = 3200
    lo = 1000.0 * np.arange(n)
    k = np.arange(n)
    on = (lo >= 1_500_000) & (lo < 2_500_000)
    raw = np.zeros((n, 4), dtype=np.int16)
    raw[on] = np.where(k[on] % 2 == 0, 5199, 7799)[:, None]
    om32 = raw.astype(np.float32) * (np.float32(EM.HIL_OMEGA_MAX) / np.float32(32767.0))
    hp, _ = EM.h7_half_period(om32)
    true = lo + np.where(hp[:, 0] == 1500, 4.0, 0.0) * on
    counts = O.run(true.astype(np.int64), om32.astype(np.float64), 1)
    assert counts.at([2_490_000])[0, 0] == 1730
    u = 30_000.0 + 10_000.0 * np.arange(300)
    rng = np.random.default_rng(5)
    c = counts.at(np.round(u + rng.uniform(0, EM.J_READ_US, len(u))).astype(np.int64)).astype(float)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)      # an unbounded end gave 0 x inf = NaN in the line search
        res = pt.certify(u, c, lo, np.full(n, 4.0), om32)
    assert res.get('ext') is not None or not res['ok'], res['reason']
    if res.get('ext') is not None:
        lo_s, hi_s = np.array([res['ext'](x) for x in u]).T
        assert np.all((lo_s - 1e-6 <= u) & (u <= hi_s + 1e-6))
    assert not res['reason'].startswith('inconsistent'), res['reason']


def test_a_negative_forced_low_step_undone_by_the_first_edge():
    """S2M1-3: the start forces a high pin low (count -1), the first edge brings the count back to c0, so a line read
    after the first edge still shows c0; the rest bound must not place that line before the first edge."""
    n = 5000
    lo = 1000.0 * np.arange(n)
    om = np.zeros((n, 4))
    j = np.arange(n - 1500)
    base = 9.52 + np.clip((j - 50) / 300.0, 0, 1) * 12.0                # 50 ms at hp 1000 us, then speeds up
    om[1500:] = base[:, None] * np.array([1.0, -1.07, 0.94, -1.02])
    om32 = om32_of(om)
    ed = pt.simulate_edges(lo, np.zeros(n), om32)
    S = lo[1500]
    u = S - 8800.0 + 10_000.0 * np.arange(-150, 340)            # one line 1200 us after the start
    read = u + np.random.default_rng(2).uniform(0, EM.J_READ_US, len(u))
    c = np.zeros((len(u), 4))
    for w in range(4):
        e = ed[w]
        assert e['t'][0] < S + 1200 < e['t'][1]                   # the first edge before the line, the second after
        sgn = np.array([e['segs'][s]['dir'] for s in e['seg']])
        n_e = np.array([np.sum(sgn[e['t'] - 1 <= r]) for r in read])
        c[:, w] = 1000 + np.where(read >= S, -sgn[0], 0) + n_e    # O = -1 in the direction of motion
    j = int(np.flatnonzero(u == S + 1200)[0])
    assert np.all(c[j] == 1000)
    res = pt.certify(u, c, lo, np.zeros(n), om32)
    assert res['ok'], res['reason']
    lo_s, hi_s = np.array([res['ext'](x) for x in u]).T
    assert np.all((lo_s - 1e-6 <= u) & (u <= hi_s + 1e-6))
    want = [-int(np.sign(om[1500, w])) for w in range(4)]            # -1 in the direction of motion
    assert any(list(cb) == want for cb in res['offsets']), res['offsets']


@needs_gcc
def test_later_segments_after_an_untrusted_edge_bind_nothing():
    """S2M1-2: wheel 0 runs the Codex case (raw 5199 / 7799 on alternate frames, hp-1500 frames applied at the window
    end): encoder_gen.c counts more edges than the model, the edges past the first ones are untrusted; then a stop
    and a clean restart, whose edges are trusted again but whose count carries the lost edges. Wheels 1-3 ramp up
    (their offsets are pinned). The later segment must bind nothing: the set keeps the true map (using it would
    find no offset that fits)."""
    sc = np.float32(EM.HIL_OMEGA_MAX) / np.float32(32767.0)
    n = 3400
    lo = 1000.0 * np.arange(n)
    k = np.arange(n)
    om = np.zeros((n, 4))
    j = np.arange(n - 1500)
    om[1500:] = (9.52 + np.clip((j - 50) / 300.0, 0, 1) * 12.0)[:, None] * np.array([1.0, -1.07, 0.94, -1.02])
    raw = np.round(om / float(sc)).astype(np.int16)
    alt = (k >= 1500) & (k < 2500)
    raw[alt, 0] = np.where(k[alt] % 2 == 0, 5199, 7799)
    raw[(k >= 2500) & (k < 2800), 0] = 0
    raw[k >= 2800, 0] = -raw[1500:1500 + (n - 2800), 1]
    om32 = raw.astype(np.float32) * sc
    hp, _ = EM.h7_half_period(om32)
    counts = O.run((lo + np.where(alt & (hp[:, 0] == 1500), 4.0, 0.0)).astype(np.int64), om32.astype(np.float64), 1)
    wh = pt._Wheel(pt.simulate_edges(lo, np.full(n, 4.0), om32)[0])
    assert wh.broken_seg == 1 and len(wh.segs) == 4 and wh.segs[3]['kind'] == 'start'
    assert counts.at([2_600_000])[0, 0] - counts.at([1_400_000])[0, 0] > wh.segs[2]['n_before'] + 100
    u = 30_000.0 + 10_000.0 * np.arange(335)
    rng = np.random.default_rng(5)
    c = counts.at(np.round(u + rng.uniform(0, EM.J_READ_US, len(u))).astype(np.int64)).astype(float)
    res = pt.certify(u, c, lo, np.full(n, 4.0), om32)
    assert res.get('ext') is not None, res['reason']
    lo_s, hi_s = np.array([res['ext'](x) for x in u]).T
    assert np.all((lo_s - 1e-6 <= u) & (u <= hi_s + 1e-6))


def test_mix_has_no_nan_at_the_ends():
    """The line localisation mixes the set's extents at uA, uB; an unbounded end at l = 0 or 1 is not NaN."""
    import warnings
    l = np.array([0.0, 0.5, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert list(pt._mix(l, -math.inf, 5.0)) == [-math.inf, -math.inf, 5.0]
        assert list(pt._mix(l, 3.0, math.inf)) == [3.0, math.inf, math.inf]
        assert list(pt._mix(l, 2.0, 4.0)) == [2.0, 3.0, 4.0]


def test_an_ending_edge_cancelled_by_a_stop_gives_no_rest_bound():
    """S2M2-1 (Codex round 2 of v3): FR starts at hp 1000 us with the forced-low step -2 against the motion, the
    first two edges bring the count back to c0, the third is cancelled by a stop 2 ms after the start; the 10 ms
    ticks miss the dip, so the count reads c0 until the restart 100 ms later and the last line at c0 is read long
    after the edge the rest bound uses. Every frame applied 4 us late (window 4 us): the true edges are the simulated
    ones + 4 us. The set must keep the true map (identity)."""
    n = 2600
    lo = 1000.0 * np.arange(n)
    S_k = 1600
    j = np.arange(n - S_k)
    ramp = 9.52 + np.clip((j - 50) / 300.0, 0, 1) * 12.0
    om = np.zeros((n, 4))
    om[S_k:] = ramp[:, None] * np.array([1.0, -1.07, 0.94, -1.02])
    om[S_k:S_k + 2, 1] = -9.52                                      # FR: 2 ms at hp 1000 us, stop, restart
    om[S_k + 2:S_k + 100, 1] = 0.0
    om[S_k + 100:, 1] = -1.07 * ramp[:n - S_k - 100]
    om32 = om32_of(om)
    ed = pt.simulate_edges(lo, np.full(n, 4.0), om32)
    S = lo[S_k]
    u = S + 5000.0 + 10_000.0 * np.arange(-120, 95)
    read = u + np.random.default_rng(4).uniform(0, EM.J_READ_US, len(u))
    c = np.zeros((len(u), 4))
    for w in range(4):
        e = ed[w]
        sgn = np.array([e['segs'][x]['dir'] for x in e['seg']])
        keep = np.ones(len(e['t']), bool)
        if w == 1:
            d = e['segs'][1]['dir']
            assert e['segs'][1]['n_edges'] == 3 and e['segs'][2]['kind'] == 'stop'
            assert e['t'][e['segs'][1]['first_edge'] + 2] == S + 2000.0         # true + 4 us: with the stop
            keep[e['segs'][1]['first_edge'] + 2] = False                 # the third edge: cancelled by the stop
            o = np.where(read >= S + 4.0, -2 * d, 0)
        else:
            o = 0
        c[:, w] = 1000 + o + np.array([np.sum(sgn[keep & (e['t'] + 4.0 <= r)]) for r in read])
    j1 = int(np.flatnonzero(c[:, 1] != 1000)[0])
    assert read[j1 - 1] > S + 90_000                                  # the last line at c0 is read late
    res = pt.certify(u, c, lo, np.full(n, 4.0), om32)
    assert res.get('ext') is not None, res['reason']
    lo_s, hi_s = np.array([res['ext'](x) for x in u]).T
    assert np.all((lo_s - 1e-6 <= u) & (u <= hi_s + 1e-6)), float(np.max(np.maximum(lo_s - u, u - hi_s)))


def test_a_wide_stop_window_lets_many_edges_through():
    """S2M3-1 (Codex round 3 of v3): FR starts at hp 1000 us (raw 7799) with the forced-low step -2 against the
    motion; two edges bring the count back to c0, the third falls after a stop; a restart at hp 250 us (raw 31194)
    with another -2, then a stop whose window is 2 ms wide and whose true apply is at its end: some 23 true edges
    where the model emits 7. The count after the second stop is far from what one pending edge per channel allows;
    the rest bound must not be used, and the set must keep the true map. Truth: the edges simulated on the true
    applies (start + 4 us, second stop + 2000 us), counts by hand."""
    sc = np.float32(EM.HIL_OMEGA_MAX) / np.float32(32767.0)
    n = 2600
    S_k = 1600
    lo = 1000.0 * np.arange(n)
    j = np.arange(n - S_k)
    ramp = 9.52 + np.clip((j - 50) / 300.0, 0, 1) * 12.0
    raw = np.zeros((n, 4), dtype=np.int16)
    for w, f in ((0, 1.0), (2, 0.94), (3, -1.02)):
        raw[S_k:, w] = np.round(ramp * f / float(sc)).astype(np.int16)
    raw[S_k:S_k + 2, 1] = 7799                                        # hp 1000 us
    raw[S_k + 100, 1] = 31194                                         # hp 250 us; stop at S_k + 101
    keep = np.ones(n, bool)
    keep[[S_k + 102, S_k + 103]] = False                              # no frame inside the wide stop window
    lo, raw = lo[keep], raw[keep]
    om32 = raw.astype(np.float32) * sc
    i_start, i_stop2 = S_k, S_k + 101
    hp, _ = EM.h7_half_period(om32)
    assert hp[i_start, 1] == 1000.0 and hp[i_stop2 - 1, 1] == 250.0
    width = np.zeros(len(lo))
    width[i_start], width[i_stop2] = 4.0, 2000.0
    true_app = lo.copy()
    true_app[i_start] += 4.0
    true_app[i_stop2] += 2000.0
    ed = pt.simulate_edges(true_app, np.zeros(len(lo)), om32)          # the true edges
    S = lo[i_start]
    u = S + 5000.0 + 10_000.0 * np.arange(-120, int((lo[-1] - S - 40_000.0) / 10_000.0))
    read = u + np.random.default_rng(6).uniform(0, EM.J_READ_US, len(u))
    c = np.zeros((len(u), 4))
    for w in range(4):
        e = ed[w]
        sgn = np.array([e['segs'][x]['dir'] for x in e['seg']])
        o = np.zeros(len(u))
        if w == 1:
            d = e['segs'][1]['dir']
            assert e['segs'][1]['n_edges'] == 2 and e['segs'][3]['n_edges'] > 20
            o = np.where(read >= S + 4.0, -2 * d, 0) + np.where(read >= lo[S_k + 100], -2 * d, 0)
        c[:, w] = 1000 + o + np.array([np.sum(sgn[e['t'] <= r]) for r in read])
    j1 = int(np.flatnonzero(c[:, 1] != 1000)[0])
    assert read[j1 - 1] > S + 90_000 and c[j1, 1] - 1000 > 15             # last line at c0 late; then far from c0
    # S2M4-1 (Liam 17/11): a 2 ms window at hp 250 us is past a quarter of the half period: no map
    res = pt.certify(u, c, lo, width, om32)
    assert not res['ok'] and res['reason'].startswith('apply_window_too_wide'), res['reason']
    assert (i_stop2, 1, 2000.0, 31.25) in pt.wide_windows(width, om32)


def test_a_wide_retime_window_before_a_stop_is_refused():
    """S2M4-1 (Codex round 4 of v3): hp 250 us, then a slow-down to raw 246 (hp 31 701 us) whose window is 2 ms
    wide (the old half period runs on until the true apply: 16 more edges than the model), then a stop: refused
    (apply_window_too_wide), before any count is used."""
    sc = np.float32(EM.HIL_OMEGA_MAX) / np.float32(32767.0)
    lo = np.r_[1000.0 * np.arange(1700), 1_700_000.0 + np.array([540.0, 4000.0, 5000.0]), 1_706_000.0 + 1000.0 * np.arange(400)]
    raw = np.zeros((len(lo), 4), dtype=np.int16)
    raw[1700, :] = 31194
    raw[1701, :] = 246
    om32 = raw.astype(np.float32) * sc
    hp, _ = EM.h7_half_period(om32)
    assert hp[1700, 0] == 250.0 and hp[1701, 0] > 31000.0
    width = np.zeros(len(lo))
    width[1701] = 2000.0
    u = 1_000_000.0 + 10_000.0 * np.arange(100)
    res = pt.certify(u, np.zeros((len(u), 4)), lo, width, om32)
    assert res['reason'].startswith('apply_window_too_wide') and 'frame 1701' in res['reason'], res['reason']
    width[1701] = 0.25 * 250.0 / 2.0                                    # right at the limit: not refused for it
    assert not pt.wide_windows(width, om32)
    width[1701] += 0.01
    assert [(f, w) for f, w, _, _ in pt.wide_windows(width, om32)] == [(1701, 0), (1701, 1), (1701, 2), (1701, 3)]
    width[1701] = 0.0
    width[1702] = 0.25 * 31701.0 / 2.0 - 1.0                            # the stop: limit from the old half period
    assert not pt.wide_windows(width, om32)
    width[1702] = 0.25 * 31701.0 / 2.0 + 1.0
    assert pt.wide_windows(width, om32)[0][0] == 1702


def test_stop_boundary_holds_the_true_edge_count():
    """S2M3-1: random runs of one wheel (start, retimes or a constant speed, a stop; frames 0.3-1.5 ms apart, windows
    up to 2 ms on half the frames, the true apply anywhere in its window): when every edge of the run is trusted, the
    true edges before the true stop (simulated on the true applies with no window; the edge model with no window is
    encoder_gen.c, checked against the oracle above) minus the modelled ones lie in [o_lo, o_hi] of the stop."""
    sc = np.float32(EM.HIL_OMEGA_MAX) / np.float32(32767.0)
    n_checked = 0
    for seed in range(800):
        rng = np.random.default_rng(seed)
        n = 60
        lo = np.cumsum(rng.uniform(300, 1500, n)) + 10_000.0
        raw = np.zeros(n, dtype=np.int16)
        k1 = int(rng.integers(8, 40))
        raw[5:k1] = rng.integers(4000, 32000, k1 - 5) * (1 if rng.uniform() < 0.5 else -1)
        if rng.uniform() < 0.5:
            raw[5:k1] = raw[5]
        width = np.where(rng.uniform(size=n) < 0.5, rng.uniform(0, 2000, n), rng.uniform(0, 5, n))
        width[:-1] = np.minimum(width[:-1], np.diff(lo) - 2)               # a window ends before the next frame
        true = np.floor(lo + rng.uniform(0, 1, n) * width)
        om32 = np.zeros((n, 4), np.float32)
        om32[:, 0] = raw.astype(np.float32) * sc
        m = pt.simulate_edges(lo, width + 1.0, om32)[0]
        if pt._Wheel(m).broken_seg <= 1:
            continue
        t = pt.simulate_edges(true, np.zeros(n), om32)[0]
        sm = next(x for x in m['segs'] if x['kind'] == 'stop')
        st = next(x for x in t['segs'] if x['kind'] == 'stop')
        assert sm['o_lo'] <= st['n_before'] - sm['n_before'] <= sm['o_hi'], (seed, st['n_before'] - sm['n_before'],
                                                                             sm['o_lo'], sm['o_hi'])
        n_checked += 1
    assert n_checked > 150


def _wheel_of(segs, broken_seg=None, n_edges_total=0, trusted_upto=None):
    """A stand-in for _Wheel with given segments (rest first): only what _rest_holds reads."""
    from types import SimpleNamespace
    full = [dict(t0=-math.inf, w0=0.0, kind='rest', dir=0, o_lo=0, o_hi=0, n_edges=0, first_edge=0, hp_min=math.inf)]
    e0 = 0
    for k, sg in enumerate(segs, start=1):
        sg = dict(dict(w0=4.0, o_lo=0, o_hi=0, n_edges=0, hp_min=math.inf, kind='start' if sg['dir'] else 'stop'), **sg,
                  first_edge=e0)
        e0 += sg['n_edges']
        full.append(sg)
    t = np.arange(e0, dtype=float) * 10.0
    tr = np.ones(e0, bool) if trusted_upto is None else (np.arange(e0) < trusted_upto)
    return SimpleNamespace(segs=full, broken_seg=len(full) if broken_seg is None else broken_seg, t=t, hi=np.zeros(e0),
                           trusted=tr)


def test_rest_holds_by_hand():
    """S2M2-1: the counts a wheel can show after the next boundary of segment 1, by hand: a stop that may take back
    3 edges; a moving last segment (no bound on its side); from the first untrusted edge on, the trusted edges before
    the boundary as the lower bound and 4 ceil(dur / hp_min) + 4 as the upper bound."""
    # start 5 edges, stop that may take back 3, a reversal of 1 edge, stop: counts after the stop in [1, 5]
    wh = _wheel_of([dict(dir=1, t0=0.0, n_edges=5), dict(dir=0, t0=1e5, o_lo=-3), dict(dir=-1, t0=2e5, n_edges=1),
                    dict(dir=0, t0=3e5)])
    assert pt._rest_holds(wh, 0, 2) is False and pt._rest_holds(wh, 0, 6) is True
    assert pt._rest_holds(wh, 0, 0) is True                              # 5 - 3 - 1 = 1 is the lowest
    # a moving last segment goes on past the frame log
    wh = _wheel_of([dict(dir=1, t0=0.0, n_edges=5), dict(dir=0, t0=1e5), dict(dir=1, t0=2e5, n_edges=2)])
    assert pt._rest_holds(wh, 0, 100) is False and pt._rest_holds(wh, 0, 4) is True
    # untrusted edges in a later reversal (broken segment 3): up to 4 ceil(100.004 ms / 1 ms) + 4 = 408 edges back
    wh = _wheel_of([dict(dir=1, t0=0.0, n_edges=500), dict(dir=0, t0=6e5),
                    dict(dir=-1, t0=7e5, n_edges=2, hp_min=1000.0), dict(dir=0, t0=8e5)], broken_seg=3,
                   trusted_upto=500)
    assert pt._rest_holds(wh, 0, 200) is False and pt._rest_holds(wh, 0, 92) is False and pt._rest_holds(wh, 0, 91)
    # untrusted from segment 1 on: only the first 100 edges are certain before the stop; up to 4 ceil(...) + 4 edges
    wh = _wheel_of([dict(dir=1, t0=0.0, n_edges=500, hp_min=1000.0), dict(dir=0, t0=500_000.0)], broken_seg=1,
                   trusted_upto=100)
    assert pt._rest_holds(wh, 0, 200) is False and pt._rest_holds(wh, 0, 99) is True
    assert pt._rest_holds(wh, 0, 2009) is True and pt._rest_holds(wh, 0, 2008) is False   # 4 ceil(500.004) + 4
    assert pt._rest_holds(_wheel_of([dict(dir=1, t0=0.0, n_edges=5)]), 0, 3) is True   # no later boundary


def test_rest_bound_per_start_offset():
    """S2M1-3 per offset: the count can read c0 until edge |O| + 1 when O dips against the motion, until the first edge
    when O = 0, only inside the start window when O moves with it; an untrusted ending edge gives no bound."""
    n = 3000
    lo = 1000.0 * np.arange(n)
    om = np.zeros((n, 4))
    om[1500:] = np.array([9.52, -9.52, 9.52, -9.52])                 # hp 1000 us, constant speed
    wh = pt._Wheel(pt.simulate_edges(lo, np.full(n, 5.0), om32_of(om))[0])
    s1 = wh.segs[1]
    S, W, d, i0 = s1['t0'], s1['w0'], s1['dir'], s1['first_edge']
    assert d == 1 and W == 5.0
    edge = lambda k: wh.t[i0 + k] + wh.hi[i0 + k] + pt.TOL_US     # noqa: E731
    assert pt._rest_bound(wh, 0) == edge(0)
    assert pt._rest_bound(wh, -1) == edge(1) and pt._rest_bound(wh, -2) == edge(2)
    assert pt._rest_bound(wh, 1) == pt._rest_bound(wh, 2) == S + W + pt.TOL_US
    wh.trusted[i0 + 1] = False
    assert pt._rest_bound(wh, -1) == math.inf and pt._rest_bound(wh, 0) == edge(0)


def test_plant_steps_at_anchors_and_across_unanswered_frames():
    """S2M1-5: on an answered frame's receive its step (even next to a gap); between consecutive answered frames
    interpolated; across unanswered frames, or outside, empty."""
    ak = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    au = np.array([0.0, 1000.0, 2000.0, 5000.0, 6000.0])
    st, n_gap = pt.plant_steps(np.array([2000.0, 1500.0, 3000.0, 5000.0, 7000.0, -1.0]), ak, au)
    assert st[0] == 2.0 and st[1] == 1.5 and np.isnan(st[2]) and st[3] == 5.0
    assert np.isnan(st[4]) and np.isnan(st[5]) and n_gap == 1


def test_sender_interval_from_the_pi_transfer_and_unanswered_frames():
    """S2M1-4, -5: the Pi send lies in [rx - (t_recv - t_send), rx]; for a frame the H7 did not answer, rx is only
    known between the answered frames around it; dt_us moves both ends."""
    h7 = dict(anchor_k=np.array([0.0, 1.0, 2.0, 5.0]), anchor_us=np.array([400.0, 1400.0, 2400.0, 5400.0]))
    ts = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64) * 1_000_000
    tr = ts + 450_000
    r = pt.RHO_PI_PPM * 1e-6                                         # Pi durations on the H7 clock (S2M2-2)
    assert pt.sender_interval(1, 0.0, h7, ts, tr) == pytest.approx((1400.0 - 450.0 * (1 + r), 1400.0))
    assert pt.sender_interval(3, -20.0, h7, ts, tr) == pytest.approx((2400.0 - 450.0 * (1 + r) - 20.0 - 20.0 * r,
                                                                      5400.0 - 20.0 + 20.0 * r))
    assert pt.sender_interval(1, 0.0, h7, ts, None).startswith('frame log without t_recv_ns')
    assert pt.sender_interval(7, 0.0, h7, ts, tr) == 'sender frame outside the H7 answers'
    assert 'not positive' in pt.sender_interval(1, 0.0, h7, ts, ts)


def test_sender_interval_holds_a_pi_clock_slower_than_the_h7_clock():
    """S2M2-2 (Codex round 2 of v3): t_recv - t_send = 450 us and dt = 49 000 us on the Pi clock, the H7 clock at
    0.999 of the Pi rate: the receive at 440 us (H7) is inside the transfer and the true sample-0 time is 48 951 us on
    the H7 axis (send at 0); the interval must hold it (Pi durations widened by RHO_PI_PPM)."""
    h7 = dict(anchor_k=np.array([0.0, 1.0]), anchor_us=np.array([440.0, 99_440.0]))
    ts = np.array([0, 99_000_000], dtype=np.int64)
    tr = ts + 450_000
    lo, hi = pt.sender_interval(0, 49_000.0, h7, ts, tr)
    assert lo <= 0.999 * 49_000.0 <= hi
    assert hi - lo <= 450.0 * 1.001 + 2 * 49_000.0 * pt.RHO_PI_PPM * 1e-6 + 1e-6
