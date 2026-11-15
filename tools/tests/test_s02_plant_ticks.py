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
    assert not fit['ok'] and fit['reason'] in ('weak_excitation', 'sigma_t_over_max') and fit["t_sens_s"] > 2e-3


def test_periodic_ripple_branches_are_ambiguous():
    """S2P2-1 (Codex round 2): omega = 10 + 4 sin(2 pi t / 20 ms) on every wheel, 10 ms ticks 3.1 ms off the
    frames, alpha 95 s, beta 1: branches 20, 40, 60 ms apart give the same cost, so the map must be refused
    (before: ok with 0.29 counts RMS, sigma 0.009 ms, shift scale 1.47 ms on a +60 ms branch)."""
    tp = 100.0 + np.arange(9000) * 1e-3
    om = np.repeat((10 + 4 * np.sin(2 * math.pi * (tp - 100) / 0.020))[:, None], 4, axis=1)
    te = 5.0 + 0.0031 + np.arange(700) * 0.01
    C = pt.expected_counts(tp, pt.encodergen_omega(om))
    counts = np.stack([np.floor(np.interp(95.0 + te, tp, C[:, w])) for w in range(4)], axis=1)
    for miss in (0.0, 0.06):
        fit = pt.fit_map(te, counts, tp, om, 95.0 + te[0] + miss)
        assert not fit['ok'] and fit['reason'] == 'ambiguous', fit
        assert fit['alt_rms'] < fit['resid_rms'] + pt.AMBIG_MARGIN
        assert fit['alt_n'] >= 4                   # the scan covers branches 20, 40 ms away on both sides


@pytest.mark.parametrize('kind', ['line', 'reverse'])
def test_speed_changes_leave_no_other_branch(kind):
    """S2P2-1: starts, steps and reversals fix the branch: the scan finds no other minimum within the margin."""
    te, counts, tp, om, alpha, beta = make(kind, 0.0)
    fit = pt.fit_map(te, counts, tp, om, alpha + beta * te[0] + 0.03)
    assert fit['ok'], fit
    assert fit['alt_rms'] is None or fit['alt_rms'] >= fit['resid_rms'] + pt.AMBIG_MARGIN


def test_encodergen_omega_rounds_the_half_period():
    """O-1 (Codex round 2): 2 pi rad/s -> half period 1515.15 us rounded to 1515 us -> 1320 x 1515.15/1515
    counts/s; nothing at or below 0.3 rad/s; sign kept."""
    w = pt.encodergen_omega(np.array([2 * math.pi, -2 * math.pi, 0.3, -0.3, 0.31, 0.0]))
    rate = w * pt.COUNTS_PER_REV / (2 * math.pi)
    assert rate[0] == pytest.approx(2e6 / 1515, rel=1e-9) and rate[1] == pytest.approx(-2e6 / 1515, rel=1e-9)
    assert rate[0] / 1320.0 - 1 == pytest.approx(1e6 * math.pi / (2 * math.pi * 330) / 1515 - 1, abs=1e-6)
    assert list(w[2:4]) == [0.0, 0.0] and w[4] > 0 and w[5] == 0.0


def test_count_model_uses_the_encodergen_rate():
    """O-1: fast wheels whose half periods round by almost half a 1 us tick (317.5, 400.5, 250.5 us: up to
    1600 ppm off the ideal rate, a different error at each level, so beta cannot take it): counts made with the
    EncoderGen rate map with a small residual only when the fit uses the same rate."""
    def w_of(hp):
        return 1e6 * math.pi / (330 * hp)
    tp = 100.0 + np.arange(9000) * 1e-3
    t = tp - 100.0
    lv1 = np.array([w_of(317.49), -w_of(400.49), w_of(250.49), -w_of(317.49)])
    lv2 = np.array([w_of(400.49), -w_of(250.49), w_of(317.49), -w_of(400.49)])
    tgt = np.where((t < 4.5)[:, None], lv1, lv2)
    om = np.zeros_like(tgt)
    for i in range(1, len(t)):
        om[i] = om[i - 1] + (tgt[i] - om[i - 1]) * 0.01
    te = 5.0 + 0.0031 + np.arange(800) * 0.01
    C = pt.expected_counts(tp, pt.encodergen_omega(om))
    counts = np.stack([np.floor(np.interp(95.0 + te, tp, C[:, w])) for w in range(4)], axis=1)
    fit = pt.fit_map(te, counts, tp, om, 95.0 + te[0] + 0.01)
    assert fit['ok'] and fit['resid_rms'] < 0.5, fit
    err = (fit['alpha'] + fit['beta'] * te) - (95.0 + te)
    assert np.max(np.abs(err)) < 1e-3


def _fake_run(monkeypatch, tmp_path, us_cut=None, faults=False):
    """map_run on a modelled A-L2 run: frames and ESP32 log replaced by the model (load_frames, load_log)."""
    import s02_metrics
    te, counts, tp, om, alpha, beta = make('line', 0.0)
    state = np.zeros((len(tp), 10))
    state[:, 0:4] = om
    ref = np.zeros((len(tp), 3))
    ref[(tp - 100.0 >= 3.0) & (tp - 100.0 < 9.0), 0] = 0.5
    monkeypatch.setattr(pt, 'load_frames', lambda p: (tp, state, ref, {'_has_mosi_true': faults}))
    sp = np.where((alpha + beta * te - 100.0 >= 3.01) & (alpha + beta * te - 100.0 < 9.01), 0.5, 0.0)
    log = {'timestamp_us': te * 1e6, 'motor_on': np.ones(len(te)), 'log_seq': np.arange(len(te)) + 1.0,
           'sp_vx': sp, 'sp_vy': np.zeros(len(te)), 'sp_wz': np.zeros(len(te)),
           **{w: counts[:, i] for i, w in enumerate(pt.WHEELS)}}
    monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {'_all': log})
    esp = str(tmp_path / 'r.csv')
    if us_cut is not None:
        with open(esp + '.events.txt', 'w', encoding='utf-8') as f:
            f.write(f'2026-11-15T10:00:00 #us_end,run=1,cut_ticks={us_cut},qseq=9\n')
    return pt.map_run(str(tmp_path / 'r.npz'), esp, 'A-L2')


def test_map_run_refuses_an_ultrasonic_cut(monkeypatch, tmp_path):
    """S2P2-2 (Codex round 2): '#us_end' with cut ticks -> no map (us_cut), as s02_metrics; 0 cut ticks maps."""
    res, rows = _fake_run(monkeypatch, tmp_path, us_cut=99)
    assert not res['ok'] and res['reason'] == 'us_cut' and res['us_cut_ticks'] == 99
    assert all(r[1:] == [''] * (len(pt.PLANT_COLUMNS) - 1) for r in rows)
    res, _ = _fake_run(monkeypatch, tmp_path, us_cut=0)
    assert res['fit_ok'], res                       # the fit runs; the map is locked (not_certified)


def test_map_run_refuses_a_fault_run(monkeypatch, tmp_path):
    """O-3: a frame log with mosi_true (--faults) is not mapped."""
    res, _ = _fake_run(monkeypatch, tmp_path, faults=True)
    assert not res['ok'] and res['reason'] == 'faults_not_supported'


@pytest.mark.parametrize('ppm', [-2000.0, 2000.0])
def test_map_run_both_drift_signs(monkeypatch, tmp_path, ppm):
    """S2P1-10 (Codex round 2 note): map_run with the anchor drift either way recovers the map."""
    import s02_metrics
    te, counts, tp, om, alpha, beta = make('line', ppm)
    state = np.zeros((len(tp), 10))
    state[:, 0:4] = om
    ref = np.zeros((len(tp), 3))
    ref[(tp - 100.0 >= 3.0) & (tp - 100.0 < 9.0), 0] = 0.5
    monkeypatch.setattr(pt, 'load_frames', lambda p: (tp, state, ref, {}))
    q = alpha + beta * te - 100.0
    sp = np.where((q >= 3.01) & (q < 9.01), 0.5, 0.0)
    log = {'timestamp_us': te * 1e6, 'motor_on': np.ones(len(te)), 'log_seq': np.arange(len(te)) + 1.0,
           'sp_vx': sp, 'sp_vy': np.zeros(len(te)), 'sp_wz': np.zeros(len(te)),
           **{w: counts[:, i] for i, w in enumerate(pt.WHEELS)}}
    monkeypatch.setattr(s02_metrics, 'load_log', lambda p: {'_all': log})
    res, _ = pt.map_run(str(tmp_path / 'r.npz'), str(tmp_path / 'r.csv'), 'A-L2')
    assert res['fit_ok'], res
    err = (res['alpha'] + res['beta'] * te) - (alpha + beta * te)
    assert np.max(np.abs(err)) < 1.5e-3


def test_map_run_is_locked_until_certified(monkeypatch, tmp_path):
    """Liam 15/11 CODE_S02 "Khoa anh xa, gop phan on": a fit that would be ok gives no map (not_certified), no
    plant rows; with the lock lifted (a test only) the same run maps."""
    res, rows = _fake_run(monkeypatch, tmp_path)
    assert res['fit_ok'] and not res['ok'] and res['reason'] == 'not_certified' and res['certified'] is False
    assert all(r[1:] == [''] * (len(pt.PLANT_COLUMNS) - 1) for r in rows)
    monkeypatch.setattr(pt, 'CERTIFIED', True)
    res, rows = _fake_run(monkeypatch, tmp_path)
    assert res['ok'] and any(r[1] != '' for r in rows)
