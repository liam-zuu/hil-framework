"""imu_freq_exp.py (spec controller-IMU section 10, L71, L72): decision rule on synthetic rows,
metric window and pair metrics on synthetic logs, and the fixes of Codex round 1
(IMN-1 common reference, IMN-6 SE over seeds, IMN-9 common window / coverage / no silent
drop, IMN-12 STOP tick, IMN-13 locked test set)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import imu_freq_exp as X  # noqa: E402
import fwsim  # noqa: E402

TESTS9 = list(X.TESTS_LOCKED)


def _rows(d_wz, d_psi, rpwm20=1.0, rpwm100=1.0, seeds=(0, 1, 2, 3, 4), tests=TESTS9,
          jitter=1e-4, base=0.01):
    """d_wz, d_psi: a number (same for all) or a function (test, seed) -> d."""
    f = lambda d: d if callable(d) else (lambda t, s: d)
    fw, fp = f(d_wz), f(d_psi)
    rng = np.random.default_rng(0)
    rows = []
    for ctrl in X.CONTROLLERS:
        for t in tests:
            for s in seeds:
                j = rng.normal(0, jitter, 2)
                rows.append(dict(controller=ctrl, test=t, seed=s, rate_hz=100.0,
                                 E_wz=base, E_psi=base, R_pwm=rpwm100))
                rows.append(dict(controller=ctrl, test=t, seed=s, rate_hz=20.0,
                                 E_wz=base + fw(t, s) + j[0], E_psi=base + fp(t, s) + j[1], R_pwm=rpwm20))
    return rows


class TestDecide:
    def test_no_difference_keeps_20(self):
        d = X.decide(_rows(0.0, 0.0))
        assert all(r['choice_hz'] == 20 for r in d.values())

    def test_clear_wz_gain_chooses_100(self):
        d = X.decide(_rows(d_wz=0.01, d_psi=0.0))
        assert all(r['choice_hz'] == 100 for r in d.values())
        assert d['PID']['E_wz']['improves'] and d['PID']['E_psi']['not_worse']

    def test_clear_psi_gain_alone_chooses_100(self):
        d = X.decide(_rows(d_wz=0.0, d_psi=0.01))
        assert d['PID']['E_psi']['improves'] and not d['PID']['E_wz']['improves']
        assert d['PID']['choice_hz'] == 100

    def test_gain_below_delta_keeps_20(self):
        d = X.decide(_rows(d_wz=0.5 * X.DELTA['E_wz'], d_psi=0.0))
        assert d['PID']['choice_hz'] == 20

    def test_other_metric_much_worse_keeps_20(self):
        d = X.decide(_rows(d_wz=0.01, d_psi=-0.01))
        assert d['PID']['rule']['c2'] is False and d['PID']['choice_hz'] == 20

    def test_pwm_increase_over_10_percent_keeps_20(self):
        d = X.decide(_rows(d_wz=0.01, d_psi=0.0, rpwm20=1.0, rpwm100=1.11))
        assert d['PID']['R_pwm']['ok'] is False and d['PID']['choice_hz'] == 20
        d = X.decide(_rows(d_wz=0.01, d_psi=0.0, rpwm20=1.0, rpwm100=1.09))
        assert d['PID']['choice_hz'] == 100

    def test_se_is_over_seeds_not_over_pairs(self):
        """Codex IMN-6 counterexample: seed differences [-0.01, 0, 0.01, 0.02, 0.03], the same in
        all nine tests. SE over the 45 pairs would give 2 SE = 0.0043 < mean 0.01 (choose 100);
        over the 5 seed means 2 SE = 0.0141 > 0.01 (keep 20)."""
        per = [-0.01, 0.0, 0.01, 0.02, 0.03]
        d = X.decide(_rows(d_wz=lambda t, s: per[s], d_psi=0.0, jitter=0.0))
        r = d['PID']['E_wz']
        assert r['n'] == 5 and r['per_seed_d'] == pytest.approx(per)
        assert r['se'] == pytest.approx(np.std(per, ddof=1) / math.sqrt(5))
        assert 2 * r['se'] == pytest.approx(0.014142, abs=1e-5)
        assert not r['improves'] and d['PID']['choice_hz'] == 20

    def test_tests_have_equal_weight_within_a_seed(self):
        per_test = {t: (0.02 if t == 'A-W1' else 0.0) for t in TESTS9}
        d = X.decide(_rows(d_wz=lambda t, s: per_test[t], d_psi=0.0, jitter=0.0))
        assert d['PID']['E_wz']['mean_d'] == pytest.approx(0.02 / 9)

    def test_missing_run_stops(self):
        rows = [r for r in _rows(0.0, 0.0) if not (r['test'] == 'A-L1' and r['seed'] == 3 and r['rate_hz'] == 20.0)]
        with pytest.raises(X.ExperimentError):
            X.decide(rows)

    def test_nan_stops_not_dropped(self):
        rows = _rows(d_wz=0.01, d_psi=0.0)
        rows[0]['E_psi'] = float('nan')
        with pytest.raises(X.ExperimentError):
            X.decide(rows)

    def test_one_seed_stops(self):
        with pytest.raises(X.ExperimentError):
            X.decide(_rows(0.01, 0.0, seeds=(0,)))


def test_a_w1_profile_and_locked_test_set():
    assert X.a_w1(1.0) == (0.0, 0.0, 0.0)
    assert X.a_w1(3.0)[2] == 1.0 and X.a_w1(5.0)[2] == -1.0 and X.a_w1(7.0)[2] == 0.0
    assert tuple(t[0] for t in X.tests()) == X.TESTS_LOCKED
    assert len(X.TESTS_LOCKED) == 9 and not any(i.startswith('A-K') for i in X.TESTS_LOCKED)


def test_setpoint_rest_then_test_then_zero():
    sp = X.setpoint_of(lambda u: (1.0, 0.0, u), 2.0)
    assert sp(0.0) == (0.0, 0.0, 0.0) and sp(X.PRE_S - 1e-9) == (0.0, 0.0, 0.0)
    assert sp(X.PRE_S + 0.5) == (1.0, 0.0, 0.5)
    assert sp(X.PRE_S + 2.0) == (0.0, 0.0, 0.0)          # t - PRE == duration: STOP (IMN-12)


# ── synthetic logs for run_arrays / pair_metrics ──────────────────────────────

def _log(dur, theta=None, psi_ref=None, head_on=None, wz_err=0.0, ready=True, psi_reset=0):
    t = np.round(np.arange(0.01, X.PRE_S + dur + X.STOP_TAIL_S + 1e-9, 0.01), 6)
    n = len(t)
    imu = np.zeros((n, len(fwsim.IMU_COLS)))
    imu[:, X.C['g_valid']] = imu[:, X.C['r_valid']] = 1.0 if ready else 0.0
    imu[:, X.C['head_on']] = 1.0 if head_on is None else head_on(t)
    imu[:, X.C['psi_ref']] = 0.0 if psi_ref is None else psi_ref(t)
    imu[:, X.C['psi_reset_count']] = psi_reset
    pose = np.zeros((n, 3))
    pose[:, 2] = 0.0 if theta is None else theta(t)
    v = np.zeros((n, 3))
    v[:, 2] = wz_err
    return dict(t=t.tolist(), imu=imu, pose=pose, v_true=v, sp=np.zeros((n, 3)), duty=np.zeros((n, 4)))


class TestWindowAndPairs:
    def test_window_excludes_rest_and_stop_tick(self):
        a = X.run_arrays(_log(2.0), 2.0)
        tw = a['t'][a['w']]
        assert tw[0] >= X.PRE_S - 1e-9 and tw[-1] < X.PRE_S + 2.0 - 1e-9
        assert np.isclose(tw[0], X.PRE_S) and np.isclose(tw[-1], X.PRE_S + 1.99)

    def test_heading_reference_is_relative_to_its_start(self):
        """IMN-1: theta_true has a global zero, psi_ref starts at 0 when the loop turns on. The same
        perfect tracking with the loop on at 50 ms or at 20 ms must give the same E_psi (0)."""
        theta = lambda t: 0.3 + 0.5 * t                       # global offset, turning
        logs = {}
        for rate, t_on in ((20.0, 0.05), (100.0, 0.02)):
            on = lambda t, t_on=t_on: (t >= t_on - 1e-9).astype(float)
            psi = lambda t, t_on=t_on: np.where(t >= t_on - 1e-9, 0.5 * (t - t_on), 0.0)
            logs[rate] = X.run_arrays(_log(2.0, theta=theta, psi_ref=psi, head_on=on), 2.0)
        m = X.pair_metrics(logs[20.0], logs[100.0])
        assert m[20.0]['E_psi'] == pytest.approx(0.0, abs=1e-9)
        assert m[100.0]['E_psi'] == pytest.approx(0.0, abs=1e-9)

    def test_e_psi_on_the_common_window(self):
        """IMN-9: a branch whose loop is off where the error is large is scored on the same ticks."""
        theta = lambda t: np.where(t > X.PRE_S + 1.9, 1.0, 0.0)
        off_end = lambda t: (t <= X.PRE_S + 1.9 + 1e-9).astype(float)
        a = X.run_arrays(_log(2.0, theta=theta), 2.0)
        b = X.run_arrays(_log(2.0, theta=theta, head_on=off_end), 2.0)
        m = X.pair_metrics(a, b)
        assert m[20.0]['E_psi'] == pytest.approx(m[100.0]['E_psi']) == pytest.approx(0.0)
        assert m[20.0]['coverage'] == pytest.approx(0.95, abs=0.006)

    def test_low_common_coverage_stops(self):
        a = X.run_arrays(_log(2.0), 2.0)
        b = X.run_arrays(_log(2.0, head_on=lambda t: ((t < X.PRE_S + 1.0)).astype(float)), 2.0)
        with pytest.raises(X.ExperimentError):
            X.pair_metrics(a, b)

    def test_not_ready_before_motion_stops(self):
        with pytest.raises(X.ExperimentError):
            X.run_arrays(_log(2.0, ready=False), 2.0)

    def test_psi_reset_stops(self):
        with pytest.raises(X.ExperimentError):
            X.run_arrays(_log(2.0, psi_reset=1), 2.0)

    def test_e_wz(self):
        a = X.run_arrays(_log(2.0, wz_err=0.02), 2.0)
        b = X.run_arrays(_log(2.0), 2.0)
        m = X.pair_metrics(a, b)
        assert m[20.0]['E_wz'] == pytest.approx(0.02) and m[100.0]['E_wz'] == 0.0
