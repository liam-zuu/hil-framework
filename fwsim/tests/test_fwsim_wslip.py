"""Wheel-consistency slip flag on fwsim: wheel_slip.c of the ESP32 firmware (AGV feat/wheel-slip,
Liam approved in the CODE session 04/11/2026: "giờ sẽ làm luôn cách 1 nhé"). Log-only flag from
the 4-wheel kinematic redundancy fl + fr - rl - rr = 0 (kinematics_inverse), on encoder counts
over a 200 ms window. Plant ground 'traction' (method A, wheels can slip per wheel through mu_hd);
the 'rigid' plant drives each wheel on its own and takes the body velocity as the least-squares
fit of the four, so it does not keep the 4-wheel constraint and is not used here.
All thresholds are [DE XUAT] defaults; noise-free IMU; logic checks, not performance numbers.
Run: python -m pytest fwsim/tests  (needs fwsim/build/fwsim.dll built from firmware with wheel_slip.c)"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import o1_feasibility as o1  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
pytestmark = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')

W = {n: i for i, n in enumerate(fwsim.WS_COLS)}
IMU = dict(rate_hz=20.0, boot_reset=-1.0)
WIN = 20              # ticks, wheel_slip.h default


@pytest.fixture(scope='module')
def fw():
    f = fwsim.FirmwareSim()
    if not f.ws_cols:
        pytest.skip('fwsim.dll built from firmware without wheel_slip.c')
    return f


def plant(mu=0.5):
    p = fwsim.MecanumParams()
    p.ground_model = 'traction'
    p.mu_hd = np.asarray(mu, dtype=float) if np.ndim(mu) else np.full(4, float(mu))
    return p


def ws(log, name):
    return log['ws'][:, W[name]]


def step(vx=0.0, vy=0.0, wz=0.0, t0=0.5):
    return lambda t: (vx, vy, wz) if t >= t0 else (0.0, 0.0, 0.0)


def mass(p, m):
    """The plant with mass m and Iz from it (params.py formula), kinematics recomputed."""
    p.M = m
    p.Iz = (1.0 / 12.0) * m * ((2 * p.lx) ** 2 + (2 * p.ly) ** 2)
    p._compute_kinematics()
    return p


@pytest.mark.parametrize('ctrl', ['PID', 'ADRC'])
def test_no_flag_on_figure8_with_even_grip(fw, ctrl):
    """O1 figure 8 at 0.8 m/s, mu 0.5 on all wheels, M 4.0 kg (the plant this bound was checked
    on): the residual stays within the count quantization (|res| <= 4) and the flag never turns on.
    Pinned to M 4.0, mu 0.5 on 09/11 when params.py took the weighed M 4.12 and the measured
    mu 0.436: there ADRC reaches |res| = 5 on 1-6 samples (real slight slip, not quantization;
    probe 09/11), the flag still never on (next test)."""
    sch = o1.Schedule(0.8)
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + 2.0, mass(plant(), 4.0), seed=1, mode=ctrl, imu=IMU)
    ev = ws(log, 'ws_eval') == 1
    assert ev.mean() > 0.5
    assert np.abs(ws(log, 'ws_res')[ev]).max() <= 4
    assert ws(log, 'ws_on_count')[-1] == 0


@pytest.mark.parametrize('ctrl', ['PID', 'ADRC'])
@pytest.mark.parametrize('mu', [0.413, 0.436, 0.473])
def test_no_flag_on_figure8_at_measured_floor(fw, ctrl, mu):
    """Same figure 8 on the plant defaults (weighed M) at the three measured mu levels of the
    task floor (MU_FLOOR 2026-11-09_run02 min, mean, max): the slip flag never turns on."""
    p = plant(mu)
    assert p.M == pytest.approx(4.12)
    sch = o1.Schedule(0.8)
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + 2.0, p, seed=1, mode=ctrl, imu=IMU)
    assert (ws(log, 'ws_eval') == 1).mean() > 0.5
    assert ws(log, 'ws_on_count')[-1] == 0


def test_one_slippery_wheel_on_acceleration(fw):
    """FL on mu 0.05, step to 0.8 m/s: FL spins up faster than the body, the residual goes
    positive far above the threshold, the flag turns on once during the acceleration and
    turns off again when the speed holds."""
    log = fwsim.simulate(fw, step(vx=0.8), 4.0, plant([0.05, 0.5, 0.5, 0.5]), seed=1, mode='PID', imu=IMU)
    t, f = log['t'], ws(log, 'ws_flag')
    assert ws(log, 'ws_res').max() > 3 * ws(log, 'ws_th')[f == 1].max()
    assert ws(log, 'ws_on_count')[-1] == 1
    t_on = t[np.argmax(f > 0)]
    assert 0.5 < t_on < 1.0
    assert f[-1] == 0


@pytest.mark.parametrize('pos_loop', [True, False], ids=['pos_on', 'pos_off'])
def test_sideways_slip_one_slippery_wheel(fw, pos_loop):
    """FL on mu 0.2, sideways step 0.6 m/s, position loop on (firmware default) and off:
    the wheel-consistency flag turns on, and the yaw-rate residual r of imu_fusion.c follows the
    plant's r_true (wz from wheels - body wz), so that flag measures the real slip.
    Before the int/post-g2 merge (05/11) this test asserted the yaw-rate flag stays off. On the
    merged firmware the body really turns while FL slips at the start (plant |wz| up to
    0.6-0.8 rad/s) and the yaw-rate flag turns on with the position loop on (|r| > r_th for
    25 ticks vs T_slip 20); with the loop off it missed by one sample. So that claim is not
    asserted (Liam CODE session 05/11: 'Viết lại theo điều đúng'). r_th uses sigma_gz = 0 (gyro
    noise not measured yet, K1): the yaw-rate flag behaviour will change once it is."""
    fp = None if pos_loop else fwsim.vel_loop_only(fw)
    log = fwsim.simulate(fw, step(vy=0.6), 3.0, plant([0.2, 0.5, 0.5, 0.5]), seed=1, mode='PID', imu=IMU,
                         fw_params=fp)
    assert ws(log, 'ws_on_count')[-1] >= 1
    imu = np.asarray(log['imu'])
    r = imu[:, fwsim.IMU_COLS.index('r')]
    ev = imu[:, fwsim.IMU_COLS.index('slip_eval')] == 1
    r_true = np.asarray(log['r_true'])
    k = int(np.argmax(np.abs(r_true)))
    assert abs(r_true[k]) > 0.3                                    # a real slip event
    assert np.sign(r[ev][np.argmax(np.abs(r[ev]))]) == np.sign(r_true[k])
    assert np.abs(r[ev]).max() == pytest.approx(np.abs(r_true).max(), rel=0.2)


def test_hold_time_needed(fw):
    """Same slip as above with t_hold 10 s: the residual is over the threshold but never long
    enough, so the flag stays off."""
    log = fwsim.simulate(fw, step(vx=0.8), 4.0, plant([0.05, 0.5, 0.5, 0.5]), seed=1, mode='PID', imu=IMU,
                         fw_params=dict(ws_t_hold_us=10_000_000))
    assert np.any(np.abs(ws(log, 'ws_res')) > ws(log, 'ws_th'))
    assert ws(log, 'ws_on_count')[-1] == 0


def test_radius_mismatch_does_not_flag(fw):
    """Synthetic encoder counts (no plant): all wheels at 10 rad/s, FL counting 2.5 % more as a
    wheel with a 2.5 % smaller radius would, no slip. The residual grows with the distance
    turned (about 11 counts per 200 ms window, over n_min); the k_rel part of the threshold
    covers it, so the flag stays off."""
    fw.reset()
    fw.set_run(True, False)
    per_tick = 10.0 * fwsim.FW_DT_US * 1e-6 * fwsim.COUNTS_PER_REV / (2 * np.pi)
    res, flags = [], []
    for k in range(1, 301):
        n = per_tick * k
        c = [int(n * 1.025), int(n), int(n), int(n)]
        fw.tick(k * fwsim.FW_DT_US, c, (0.5, 0.0, 0.0))
        out = fw.ws_last()
        res.append(out[W['ws_res']]); flags.append(out[W['ws_flag']])
    assert max(res[WIN + 1:]) > 8
    assert max(flags) == 0


def test_not_evaluable_before_window_and_at_standstill(fw):
    """No verdict before a full window, nor while the wheels stand still (max |d_i| < act_min)."""
    log = fwsim.simulate(fw, step(vx=0.3, t0=1.0), 2.5, plant(), seed=1, mode='PID', imu=IMU)
    ev = ws(log, 'ws_eval')
    assert np.all(ev[:WIN] == 0)
    assert np.all(ev[log['t'] < 1.0] == 0)
    assert ev[log['t'] > 2.0].all()
    assert np.all(ws(log, 'ws_flag') == 0)


def test_window_change_mid_run_restarts_history(fw):
    """Codex P11 (05/11): changing ws_win_ticks while running restarts the history, so the
    flag can be evaluated again after the new window fills (before: S.n never equal to len)."""
    fw.reset()
    fw.set_run(True, False)
    k, c = 0, 0
    def tick():
        nonlocal k, c
        k += 1
        c += 20                                     # all wheels forward, 20 counts per tick
        fw.tick(k * fwsim.FW_DT_US, [c, c, c, c], (0.0, 0.0, 0.0))
        return fw.ws_last()
    for _ in range(WIN + 5):
        w = tick()
    assert w[W['ws_eval']] == 1
    fw.set_param('ws_win_ticks', 10)
    evs = [tick()[W['ws_eval']] for _ in range(15)]
    assert evs[0] == 0 and evs[-1] == 1
    fw.set_param('ws_win_ticks', WIN)
