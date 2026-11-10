"""tools/o1_time_sync.py (C7b method A, L77): the offset between a camera speed profile and the
vehicle log is found on synthetic figure-8 profiles with a known offset (encoder quantisation,
camera noise 0.01 m/s, PTS jitter, lost frames, slip scale), the start/end check shows a clock
drift, and on a real HIL run (plant speed as the 'camera', ESP32 log) the fit is consistent.

Error of the method on these profiles, 20 noise seeds each (measured 10/11): unbiased; standard
deviation 3.7 / 1.0 / 0.7 ms, largest 10.3 / 2.5 / 1.7 ms at O1-F02 / F05 / F08; start - end
difference sd 3.7-5.4 ms, largest 13.6 ms. Times the largest path speed (0.29 / 0.73 / 1.17 m/s)
this is about 1 mm along the path. The tolerances below are set from these numbers, per level.
Without camera noise the error is below 0.1 ms."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'fwsim'))
import o1_time_sync as ts  # noqa: E402
import o1_tasks  # noqa: E402

Q_V = 0.0058        # m/s, one encoder count in 10 ms (03_CODE/CLAUDE.md quantisation floor)


def true_speed(task='O1-F05', pre=3.0, post=3.0, tau=0.15, dt=0.001):
    """Vehicle clock t, true path speed: the schedule speed through a first-order lag tau."""
    T, f = o1_tasks.task(task)
    t = np.arange(-pre, T + post, dt)
    cmd = np.array([math.hypot(*f(x)[:2]) for x in t])
    v = np.zeros_like(cmd)
    for i in range(1, len(t)):
        v[i] = v[i - 1] + (cmd[i] - v[i - 1]) * dt / tau
    return t, v


def make(D=3.4567, task='O1-F05', fps=120.0, slip=0.95, cam_noise=0.01, jitter=0.001, drop=0.03,
         drift_ppm=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t, v = true_speed(task)
    # log: 100 Hz on the ESP32 clock, encoder speed = true / slip, quantised
    tl = np.arange(t[0], t[-1], 0.01) + 100.0
    vl = np.round(np.interp(tl - 100.0, t, v) / slip / Q_V) * Q_V
    # camera: PTS on its own clock, t_log = t_cam + D (+ drift), jitter, noise, lost frames
    tc_true = np.arange(t[0] + 0.5, t[-1] - 0.5, 1.0 / fps) + 100.0
    tc = (tc_true - D) / (1.0 + drift_ppm * 1e-6) + rng.uniform(-jitter, jitter, len(tc_true))
    tc = np.sort(tc)
    vc = np.interp(tc_true - 100.0, t, v) + rng.normal(0, cam_noise, len(tc_true))
    vc[rng.random(len(vc)) < drop] = np.nan
    return tc, vc, tl, vl


TOL = {'O1-F02': 0.015, 'O1-F05': 0.004, 'O1-F08': 0.003}      # s, from the 20-seed spread above


@pytest.mark.parametrize('task', ['O1-F02', 'O1-F05', 'O1-F08'])
def test_known_offset_found(task):
    D = 3.4567
    r = ts.sync(*make(D=D, task=task))
    assert r.offset_s == pytest.approx(D, abs=TOL[task])
    assert abs(r.start_end_diff_s) < 0.020 and r.corr > 0.9


@pytest.mark.parametrize('task', ['O1-F02', 'O1-F05', 'O1-F08'])
def test_exact_without_camera_noise(task):
    D = 3.4567
    r = ts.sync(*make(D=D, task=task, cam_noise=0.0))
    assert r.offset_s == pytest.approx(D, abs=0.0005)


def test_unbiased_over_seeds():
    e = [ts.sync(*make(seed=s)).offset_s - 3.4567 for s in range(8)]
    assert abs(np.mean(e)) < 0.0015 and np.max(np.abs(e)) < TOL['O1-F05']


@pytest.mark.parametrize('D', [-1.25, 0.0, 0.8, 17.031])
def test_offset_sign_and_size(D):
    assert ts.sync(*make(D=D, cam_noise=0.0)).offset_s == pytest.approx(D, abs=0.0005)


def test_variable_frame_rate_and_60fps():
    tc, vc, tl, vl = make(fps=60.0, jitter=0.004)       # a phone changing its frame timing
    assert ts.sync(tc, vc, tl, vl).offset_s == pytest.approx(3.4567, abs=TOL['O1-F05'])


def test_long_camera_gap_is_not_bridged():
    tc, vc, tl, vl = make()
    vc[(tc > 108.0) & (tc < 109.5)] = np.nan              # 1.5 s lost (marker hidden)
    assert ts.sync(tc, vc, tl, vl).offset_s == pytest.approx(3.4567, abs=TOL['O1-F05'])


def test_clock_drift_shows_in_the_start_end_check():
    eps = 5000e-6                                         # camera clock slow by 0.5 %
    r = ts.sync(*make(drift_ppm=eps * 1e6, cam_noise=0.0))
    # t_log - t_cam grows by eps per second of camera time; the two thirds are 2/3 of the window apart
    w0, w1 = r.window_cam_s
    assert r.start_end_diff_s == pytest.approx(eps * (w1 - w0) * 2.0 / 3.0, rel=0.2)


def test_errors():
    tc, vc, tl, vl = make()
    with pytest.raises(ValueError):
        ts.sync(tc[::-1], vc, tl, vl)                     # times not increasing
    with pytest.raises(ValueError):
        ts.sync(tc, np.zeros_like(vc), tl, vl)            # camera never moving
    short = tc < tc[0] + 6.0
    with pytest.raises(ValueError):
        ts.sync(tc[short], vc[short], tl, vl)


def _hil_pairs():
    root = os.environ.get('O1_SYNC_HIL_DIR', r'D:\CLAUDE_THESIS\05_DATA\results\CLOSED_LOOP\2026-11-10_run03')
    return [(os.path.join(root, 'frames', f'{n}_frames.npz'), os.path.join(root, 'pi', f'{n}.csv'))
            for n in ('f05', 'f08') if os.path.isfile(os.path.join(root, 'pi', f'{n}.csv'))]


@pytest.mark.skipif(not _hil_pairs(), reason='HIL run CLOSED_LOOP/2026-11-10_run03 not on this machine')
@pytest.mark.parametrize('npz, log', _hil_pairs())
def test_real_hil_run(npz, log):
    """Plant speed (Pi frame clock) as the 'camera', ESP32 log as the log: two independent clocks.
    The true offset is unknown; the fit must be consistent at start and end, and a shift added to the
    camera times must come back exactly."""
    from protocol import decode_state
    z = np.load(npz)
    S = np.array([decode_state(bytes(m)) for m in z['mosi']])
    tc = (z['t_send_ns'] - z['t_send_ns'][0]) / 1e9
    vb = np.hypot(S[:, 7], S[:, 8])                      # body speed = world speed magnitude
    tc, vb = tc[::8], vb[::8]                            # 125 Hz, like a 120 fps camera
    tl, vl = ts.log_speed(log)
    r = ts.sync(tc, vb, tl, vl)
    assert r.corr > 0.95
    assert abs(r.start_end_diff_s) < 0.010
    r2 = ts.sync(tc + 1.234, vb, tl, vl)
    assert r2.offset_s == pytest.approx(r.offset_s - 1.234, abs=0.001)
