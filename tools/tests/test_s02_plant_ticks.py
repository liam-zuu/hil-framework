"""s02_plant_ticks.py (CODE_S02 design 10.2 (a)): the Pi -> ESP32 map from the encoder counts, on a model:
Pi frames at 1 kHz with jitter, wheel speeds with first-order starts, ripple, a reversal; ESP32 ticks at
10 ms with jitter on their own clock (offset, +-2000 ppm); counts = floor of the held-speed angle, no count
below the H7 OMEGA_MIN (0.3 rad/s), an integer offset per wheel."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import s02_plant_ticks as pt  # noqa: E402


def profile(tp, kind):
    """Wheel speeds (n, 4) at Pi times tp (s after 100 s): still 3 s, then the test, then still."""
    t = tp - 100.0
    rng = np.random.default_rng(7)
    w = np.zeros((len(t), 4))
    tgt = np.array([10.0, -10.0, 10.0, -10.0]) if kind == 'line' else np.array([6.0, 6.0, 6.0, 6.0])
    for j in range(4):
        if kind == 'line':                                    # A-L2: start at 3 s, stop at 9 s
            on = (t >= 3.0) & (t < 9.0)
            up = tgt[j] * (1 - np.exp(-np.clip(t - 3.0, 0, None) / 0.12))
            down = up * np.exp(-np.clip(t - 9.0, 0, None) / 0.08)
            w[:, j] = np.where(on, up, np.where(t >= 9.0, down, 0.0))
        else:                                                 # A-K2: +v, -v, +v, 3 s each, from 3 s
            seg = np.where(t < 3.0, 0.0, np.where(t < 6.0, 1.0, np.where(t < 9.0, -1.0, np.where(t < 12.0, 1.0, 0.0))))
            w[:, j] = tgt[j] * seg
            # first-order response to the steps
            out = np.zeros(len(t))
            for i in range(1, len(t)):
                out[i] = out[i - 1] + (w[i, j] - out[i - 1]) * min(1.0, (t[i] - t[i - 1]) / 0.1)
            w[:, j] = out
        ripple = 0.3 * np.sin(2 * math.pi * 1.7 * t + j) + 0.05 * rng.standard_normal(len(t))
        w[:, j] += np.where(np.abs(w[:, j]) > 1.0, ripple, 0.0)
    return w


def make(kind='line', ppm=0.0, alpha=None, seed=3, flip=None):
    rng = np.random.default_rng(seed)
    n = 14000
    tp = 100.0 + np.arange(n) * 1e-3 + rng.uniform(0, 5e-5, n)
    om = profile(tp, kind)
    beta = 1.0 + ppm * 1e-6
    alpha = 100.0 - beta * 5.0 + 0.0123 if alpha is None else alpha
    te = 5.0 + np.arange(1300) * 0.01 + rng.uniform(-3e-4, 3e-4, 1300)
    q = alpha + beta * te                                    # true Pi time of each tick
    held = np.where(np.abs(om) >= 0.3, om, 0.0)              # H7: no edges below OMEGA_MIN
    C = pt.expected_counts(tp, held)
    counts = np.stack([np.floor(np.interp(q, tp, C[:, w], left=0, right=C[-1, w])) + 1000 * (w + 1)
                       for w in range(4)], axis=1)
    if flip is not None:
        counts[:, flip] = -counts[:, flip]
    return te, counts, tp, om, alpha, beta


@pytest.mark.parametrize('kind', ['line', 'reverse'])
@pytest.mark.parametrize('ppm', [-2000.0, 0.0, 1500.0])
def test_map_is_found(kind, ppm):
    te, counts, tp, om, alpha, beta = make(kind, ppm)
    start = alpha + beta * te[0]                             # t_Pi at the first tick; the anchors miss by ms
    fit = pt.fit_map(te, counts, tp, om, start + 0.02, beta0=1.0)
    assert fit['ok'], fit
    err = (fit['alpha'] + fit['beta'] * te) - (alpha + beta * te)
    assert np.max(np.abs(err)) < 1.5e-3                       # [ƯỚC LƯỢNG] about 1 ms
    assert abs(fit['drift_ppm'] - ppm) < 300
    assert fit['resid_rms'] < 1.0


def test_wrong_wheel_sign_is_refused():
    te, counts, tp, om, alpha, beta = make('line', flip=1)
    fit = pt.fit_map(te, counts, tp, om, alpha + beta * te[0] + 0.01)
    assert not fit['ok'] and fit['reason'] in ('residual_over_max', 'not_identified', 'sigma_t_over_max')


def test_still_run_is_not_mapped():
    te, counts, tp, om, alpha, beta = make('line')
    fit = pt.fit_map(te, counts, tp, np.zeros_like(om), alpha + beta * te[0])
    assert not fit['ok'] and fit['reason'] == 'too_few_turning_ticks'


def test_plant_rows_interpolate_and_unwrap():
    tp = np.array([10.0, 10.001, 10.002])
    st = np.zeros((3, 10))
    st[:, 6] = [math.pi - 0.01, -math.pi + 0.01, -math.pi + 0.03]    # theta wraps between frames 0 and 1
    st[:, 4] = [0.0, 1.0, 2.0]
    st[:, 7] = [1.0, 1.0, 1.0]                                        # world vx
    rows = pt.plant_rows(np.array([5.0005e6, 5.5e6]), tp, st, alpha=5.0, beta=1.0)
    r = rows[0]
    assert float(r[1]) == pytest.approx(0.5)
    assert abs(abs(float(r[3])) - math.pi) < 1e-6                     # half way: +-pi, not 0
    th = float(r[3])
    assert float(r[6]) == pytest.approx(math.cos(th) * 1.0, abs=1e-6)  # body vx
    assert rows[1][1:] == [''] * (len(pt.PLANT_COLUMNS) - 1)           # outside the frame log: empty


def test_expected_counts_hold_the_frame_speed():
    tp = np.array([0.0, 0.5, 1.0])
    om = np.array([[2 * math.pi] * 4, [0.0] * 4, [0.0] * 4])
    C = pt.expected_counts(tp, om)
    assert C[1, 0] == pytest.approx(660.0) and C[2, 0] == pytest.approx(660.0)


def test_steady_speed_only_is_not_identified():
    """S2P1-1 (Codex round 1): wheels at one constant speed, counts = floor(1320 t): a 50 ms shift gives the
    same residual after the segment offsets, so the map must be refused, not ok at 0.3 counts RMS."""
    tp = 100.0 + np.arange(8000) * 1e-3
    om = np.full((len(tp), 4), 2 * math.pi)
    te = 5.0 + np.arange(700) * 0.01
    q = 95.0 + te
    C = pt.expected_counts(tp, om)
    counts = np.stack([np.floor(np.interp(q, tp, C[:, w])) for w in range(4)], axis=1)
    fit = pt.fit_map(te, counts, tp, om, 95.0 + te[0] + 0.05)
    assert not fit['ok'] and fit['reason'] in ('not_identified', 'sigma_t_over_max')


def test_sigma_is_reported_and_small_when_speed_changes():
    te, counts, tp, om, alpha, beta = make('line', 0.0)
    fit = pt.fit_map(te, counts, tp, om, alpha + beta * te[0] + 0.02)
    assert fit['ok'] and 0 < fit['sigma_t_s'] < 1e-3


@pytest.mark.parametrize('amp', [0.1, 0.3])
def test_weak_excitation_is_refused(amp):
    """S2P1-1: a small speed ripple on a steady speed: the white-noise standard error is 0.1-0.4 ms but the
    real bias reaches 3.9 ms (correlated floor quantization); the shift bound refuses it."""
    tp = 100.0 + np.arange(8000) * 1e-3
    om = np.full((len(tp), 4), 2 * math.pi) + amp * np.sin(2 * math.pi * 0.5 * (tp - 100))[:, None]
    te = 5.0 + np.arange(700) * 0.01
    C = pt.expected_counts(tp, om)
    counts = np.stack([np.floor(np.interp(95.0 + te, tp, C[:, w])) for w in range(4)], axis=1)
    fit = pt.fit_map(te, counts, tp, om, 95.0 + te[0] + 0.005)
    assert not fit['ok'] and fit['reason'] in ('weak_excitation', 'sigma_t_over_max') and fit['t_bound_s'] > 2e-3
