"""Decision rule of imu_freq_exp.py (spec controller-IMU section 10, L71) on synthetic pairs."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import imu_freq_exp as X  # noqa: E402


def _rows(d_wz, d_psi, rpwm20=1.0, rpwm100=1.0, n=10, jitter=1e-4, base=0.01):
    rng = np.random.default_rng(0)
    rows = []
    for ctrl in X.CONTROLLERS:
        for i in range(n):
            j = rng.normal(0, jitter, 2)
            rows.append(dict(controller=ctrl, test=f'T{i}', seed=0, rate_hz=100.0,
                             E_wz=base, E_psi=base, R_pwm=rpwm100))
            rows.append(dict(controller=ctrl, test=f'T{i}', seed=0, rate_hz=20.0,
                             E_wz=base + d_wz + j[0], E_psi=base + d_psi + j[1], R_pwm=rpwm20))
    return rows


def test_no_difference_keeps_20():
    d = X.decide(_rows(0.0, 0.0))
    assert all(r['choice_hz'] == 20 for r in d.values())


def test_clear_wz_gain_chooses_100():
    d = X.decide(_rows(d_wz=0.01, d_psi=0.0))
    assert all(r['choice_hz'] == 100 for r in d.values())
    assert d['PID']['E_wz']['improves'] and d['PID']['E_psi']['not_worse']


def test_clear_psi_gain_alone_chooses_100():
    d = X.decide(_rows(d_wz=0.0, d_psi=0.01))
    assert d['PID']['E_psi']['improves'] and not d['PID']['E_wz']['improves']
    assert d['PID']['choice_hz'] == 100


def test_gain_below_delta_keeps_20():
    d = X.decide(_rows(d_wz=0.5 * X.DELTA['E_wz'], d_psi=0.0))
    assert d['PID']['choice_hz'] == 20


def test_other_metric_much_worse_keeps_20():
    d = X.decide(_rows(d_wz=0.01, d_psi=-0.01))
    assert d['PID']['rule']['c2'] is False and d['PID']['choice_hz'] == 20


def test_pwm_increase_over_10_percent_keeps_20():
    d = X.decide(_rows(d_wz=0.01, d_psi=0.0, rpwm20=1.0, rpwm100=1.11))
    assert d['PID']['R_pwm']['ok'] is False and d['PID']['choice_hz'] == 20
    d = X.decide(_rows(d_wz=0.01, d_psi=0.0, rpwm20=1.0, rpwm100=1.09))
    assert d['PID']['choice_hz'] == 100


def test_nan_psi_pairs_are_dropped():
    rows = _rows(d_wz=0.01, d_psi=0.0)
    for r in rows:
        if r['test'] == 'T0':
            r['E_psi'] = float('nan')
    d = X.decide(rows)
    assert d['PID']['E_psi']['n'] == 9 and d['PID']['n_pairs'] == 10


def test_a_w1_profile():
    assert X.a_w1(1.0) == (0.0, 0.0, 0.0)
    assert X.a_w1(3.0)[2] == 1.0 and X.a_w1(5.0)[2] == -1.0 and X.a_w1(7.0)[2] == 0.0
    ids = [t[0] for t in X.tests()]
    assert 'A-W1' in ids and not any(i.startswith('A-K') for i in ids)
