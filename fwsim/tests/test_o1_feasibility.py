"""Tests of fwsim/o1_feasibility.py (O1 v5 figure 8 schedule and measures), PC only,
no DLL needed: the measures are checked on synthetic logs with a known answer.
Run: python -m pytest fwsim/tests/test_o1_feasibility.py -q"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import o1_feasibility as o1  # noqa: E402


@pytest.fixture(scope='module')
def path():
    return o1.Path8()


def test_path_length(path):
    """O1 v5 section 3.1: S = 6.097 m."""
    assert abs(path.S - 6.097) < 0.001


@pytest.mark.parametrize('v,T,Tr,Tcmd', [(0.2, 30.49, 0.58, 31.07), (0.5, 12.19, 1.46, 13.65),
                                         (0.8, 7.62, 2.33, 9.95)])
def test_schedule_table(v, T, Tr, Tcmd):
    """O1 v5 table: T, T_r, T_cmd per level; tau ends at 2 pi; tau_dot integrates to tau."""
    s = o1.Schedule(v)
    assert abs(s.T - T) < 0.01 and abs(s.Tr - Tr) < 0.01 and abs(s.Tcmd - Tcmd) < 0.01
    assert s.tau(s.Tcmd)[0] == pytest.approx(2 * math.pi, abs=1e-12)
    ts = np.linspace(0, s.Tcmd, 200001)
    td = np.array([s.tau(t)[1] for t in ts])
    integ = np.concatenate([[0], np.cumsum((td[1:] + td[:-1]) / 2 * np.diff(ts))])
    tau = np.array([s.tau(t)[0] for t in ts])
    assert np.max(np.abs(integ - tau)) < 1e-6


def _integrate(sch, dt=1e-4, offset=0.0):
    """Ideal car: integrate the body setpoint from the origin, heading 45 deg; plant frame
    (start heading 0) as fwsim logs it, optional lateral offset (left of the heading)."""
    n = int(round((sch.Tcmd + o1.T_LATE) / dt))
    x = y = th = 0.0
    out_t, pose, vel = [], [], []
    for k in range(n + 1):
        t = k * dt
        vx, vy, wz = sch.setpoint(t)
        if k % 100 == 0:          # 100 Hz like the firmware tick log
            out_t.append(t)
            pose.append((x - offset * math.sin(th), y + offset * math.cos(th), th))
            vel.append((vx, vy, wz))
        x += vx * math.cos(th) * dt
        y += vx * math.sin(th) * dt
        th += wz * dt
    return dict(t=np.array(out_t), pose=np.array(pose), v_true=np.array(vel),
                pwm=np.zeros((len(out_t), 4)))


@pytest.mark.parametrize('v', [0.2, 0.8])
def test_ideal_car_follows_path(path, v):
    """Integrating the setpoint traces the path: errors near 0, completes, stops at the origin."""
    sch = o1.Schedule(v)
    res, rows = o1.evaluate(_integrate(sch), path, sch)
    assert res['e_perp_max'] < 0.002
    assert res['e_theta_max_deg'] < 0.5
    assert res['e_s_absmax'] < 0.01
    assert res['completed'] and res['track_err_frames'] == 0
    assert res['stop_dist_origin'] < 0.002
    assert res['t_end'] == pytest.approx(sch.Tcmd + o1.T_D, abs=0.02)


def test_lateral_offset_is_measured(path):
    """A constant 5 cm offset to the left of the heading gives e_perp_max about 0.05 m;
    the progress search still follows (no tracking error)."""
    sch = o1.Schedule(0.5)
    res, _ = o1.evaluate(_integrate(sch, offset=0.05), path, sch)
    assert 0.045 < res['e_perp_max'] < 0.06
    assert res['track_err_frames'] == 0


def test_car_that_does_not_move_is_not_completed(path):
    """No motion: progress stays 0, e_s grows to S, not completed (stop needs s >= S - D)."""
    sch = o1.Schedule(0.8)
    log = _integrate(sch)
    log['pose'][:] = 0.0
    log['v_true'][:] = 0.0
    res, _ = o1.evaluate(log, path, sch)
    assert not res['completed']
    assert res['e_s_absmax'] == pytest.approx(path.S, abs=0.01)
