"""c8b_gio: the correlation that is judged for the turn channels is taken on the log brought to the band of the camera
wz column (Liam 25/11/2026 at CODE local_c65e48c9, L157 question 1, "Sua phep kiem (Recommended)"; found on run06
C8-L-F02-2_lai1: a real heading oscillation of about 4.7 Hz under LADRC, smoothed away by the camera column). The
threshold, D, L, U~ and the nodes are those of the unsmoothed channels."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c7_measure as CM  # noqa: E402
import c8b_gio as G  # noqa: E402
import o1_time_sync as ts  # noqa: E402
import test_c8b as T  # noqa: E402

F_DD = 4.7          # Hz, the oscillation seen on the LADRC F02 runs
TURN = ('wz_cnt', 'gyro_rx', 'rv_rx')


def _lac(amp):
    """A slow turn of F02 size plus a heading oscillation of F_DD that grows over the run."""
    def m(t):
        s = T._ramp(t, 3.0, 4.5) * (1 - T._ramp(t, 22.0, 23.5))
        v = 0.5 * s * (1 + 0.3 * np.sin(0.7 * t))
        w = 0.25 * s * np.sin(0.45 * t + 0.4) + amp * s * (0.4 + 0.6 * t / 24.0) * np.sin(2 * math.pi * F_DD * t)
        return v, w
    return m


def _cam_wz_nhu_camera(fr):
    """The wz column as c7_measure writes it: the quadratic-fit derivative of the heading over +-SG_HALF frames."""
    t = np.array([float(r['t_s']) for r in fr])
    psi = np.unwrap(np.array([float(r['psi']) for r in fr]))
    wz = CM.sg_derivative(t, psi)
    for r, w in zip(fr, wz):
        r['wz'] = f'{w:.6f}' if np.isfinite(w) else 'nan'
        if not np.isfinite(w):
            r['valid'] = 0


def _run(tmp, amp, frames_them=None):
    mp = pytest.MonkeyPatch()
    mp.setattr(T, 'motion', _lac(amp))

    def fm(fr):
        _cam_wz_nhu_camera(fr)
        if frames_them:
            frames_them(fr)
    fp, lp = T.make_run(str(tmp), frames_mutate=fm)
    mp.undo()
    return fp, lp


def _khong_kiem(fp, lp, task='O1-F05'):
    """The run analysed with the check as it was before L157 (no smoothed channel)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(G, 'kenh_kiem', lambda *a, **k: ({}, dict(cach='khong')))
    try:
        return G.analyse(fp, lp, task)[0]
    finally:
        mp.undo()


def test_the_derivative_is_the_camera_one():
    assert G.KIEM_SG_HALF == CM.SG_HALF
    rng = np.random.default_rng(3)
    for half in (2, 6, 10):
        t = np.cumsum(rng.uniform(0.005, 0.03, 60))
        x = np.cumsum(rng.normal(0, 0.01, 60))
        np.testing.assert_array_equal(G.dao_ham_muot(t, x, half), CM.sg_derivative(t, x, half=half))
    assert np.all(np.isnan(G.dao_ham_muot(np.arange(4.0), np.arange(4.0), 2)))        # fewer than 5 samples


def test_oscillation_the_camera_smooths_away_is_not_a_clock_failure(tmp_path):
    fp, lp = _run(tmp_path, 0.30)
    cu = _khong_kiem(fp, lp)
    assert not cu['ok'] and cu['ket_luan'] == 'hong_he_do'                 # the check before L157: lost
    assert any('corr of the' in w for w in cu['ly_do'])
    out = G.analyse(fp, lp, 'O1-F05')[0]
    assert out['ok'] and out['ket_luan'] == 'ok' and out['ly_do'] == []
    k = out['kenh']
    assert min(k[c]['corr_end'] for c in TURN) < ts.MIN_CORR              # the unsmoothed values are kept, below 0.9
    for c in TURN:
        assert min(k[c]['corr_kiem'], k[c]['corr_kiem_start'], k[c]['corr_kiem_end']) > 0.97
    assert 'corr_kiem' not in k['speed_cnt']                              # the speed channel: judged as before
    assert out['kiem_tuong_quan']['kenh'] == list(TURN)
    assert out['kiem_tuong_quan']['nua_cua_so_s'] == pytest.approx(0.1, abs=1e-4)          # +-6 frames of 60 fps
    assert (out['kiem_tuong_quan']['nua_cua_so_mau_log'], out['kiem_tuong_quan']['nua_cua_so_mau_imu']) == (10, 2)
    # the clock interval does not move: every number of the channels and of the interval as before the change
    for key in ('L_s', 'U_s', 'L_kenh', 'U_kenh', 'width_s', 'center_s', 'half_s', 'nodes_s', 'gyro_minus_rv_s',
                'rate_luot', 'cua_so_ghep_log_s', 'troi_khoang_xa_s', 'khoi_dong'):
        assert out[key] == cu[key], key
    for c in G.CHANNELS:
        for key in ('D_s', 'corr', 'D_start_s', 'D_end_s', 'corr_start', 'corr_end', 'rate', 'window_cam_s'):
            assert k[c][key] == cu['kenh'][c][key], (c, key)


def test_clean_run_unchanged(tmp_path):
    fp, lp = _run(tmp_path, 0.0)
    a, b = _khong_kiem(fp, lp), G.analyse(fp, lp, 'O1-F05')[0]
    assert a['ok'] and b['ok'] and a['nodes_s'] == b['nodes_s'] and (a['L_s'], a['U_s']) == (b['L_s'], b['U_s'])
    assert min(b['kenh'][c]['corr_kiem_end'] for c in TURN) > 0.99


def test_a_third_that_does_not_match_still_fails(tmp_path):
    """The camera turn rate of the last part of the run carries a slow error of the size of the motion (not in the
    log): the smoothed log does not agree with it either; the run is lost as before, on the judged correlation."""
    def khac(fr):
        for r in fr:
            t = float(r['t_s'])
            if t > 15.5 and r['wz'] != 'nan':
                r['wz'] = f'{float(r["wz"]) + 0.25 * math.sin(1.3 * t + 2.0):.6f}'
    fp, lp = _run(tmp_path, 0.05, khac)
    out = G.analyse(fp, lp, 'O1-F05')[0]
    assert not out['ok'] and out['ket_luan'] == 'hong_he_do' and out['dem_truot_kiem_gio']
    assert any('end third (log smoothed as the camera)' in w for w in out['ly_do'])
    assert out['kenh']['gyro_rx']['corr_kiem_end'] < ts.MIN_CORR and not out['kenh']['gyro_rx']['ok']


def test_channels_not_judged_are_matched_as_before(tmp_path):
    fp, lp = _run(tmp_path, 0.30)
    a = _khong_kiem(fp, lp, 'O1-K1')
    b = G.analyse(fp, lp, 'O1-K1')[0]
    assert b['kenh_xet'] == ['speed_cnt']
    for c in G.CHANNELS:
        assert 'corr_kiem' not in b['kenh'][c] and b['kenh'][c] == a['kenh'][c]


def test_nothing_bridged_in_the_smoothed_channels():
    # one angle series with a break after sample 29: no value within half samples of a run end, a NaN between runs
    t = np.arange(60) * 0.01
    t[30:] += 0.5
    goc = 0.7 * t
    br = np.zeros(59, bool)
    br[29] = True
    tt, vv = G._muot_theo_doan(t, goc, br, 10, -1.0, 2.0)
    assert len(tt) == 61 and np.isnan(vv[30]) and tt[29] < tt[30] < tt[31]
    fin = np.isfinite(vv)
    assert fin.sum() == 20 and np.all(fin[10:20]) and np.all(fin[41:51])
    np.testing.assert_allclose(vv[fin], 0.7, atol=1e-9)
    x = np.interp(np.array([0.2, 0.31, 0.55]), tt, vv)                   # across the break: not a number
    assert np.isnan(x[1]) and np.isnan(x[2])
    # a run shorter than the whole window gives nothing
    tt2, vv2 = G._muot_theo_doan(t[:15], goc[:15], np.zeros(14, bool), 10, -1.0, 2.0)
    assert len(tt2) == 0
    # the range: only samples whose window is whole are numbers, the run outside the range is left out
    tt3, vv3 = G._muot_theo_doan(t, goc, br, 2, 0.0, 0.2)
    assert tt3[-1] < 0.3 and np.isfinite(vv3).sum() == len(vv3) - 4


def test_kenh_kiem_rates_of_the_three_channels():
    # constant turn 0.5 rad/s: wheel counts, gyro reports and RV yaw all give 0.5 after the smoothing
    w = 0.5
    tl = 100.0 + np.arange(400) * 0.01
    wheel = np.array([-G.LXY * w, G.LXY * w, -G.LXY * w, G.LXY * w]) / G.R_W        # rad/s of fl, fr, rl, rr
    cps = wheel * G.PPR / (2 * math.pi)
    ip = np.r_[np.arange(0, 150), np.arange(151, 399)]                              # pair 150 missing: two runs
    rate = np.tile(cps, (len(ip), 1))
    tg = 100.0 + np.arange(80) * 0.05
    tr = tg + 0.013
    kk, mo = G.kenh_kiem(tl, ip, rate, tg, np.full(80, w), tr, w * tr, 1 / 60, (101.0, 103.0))
    assert mo['nua_cua_so_s'] == pytest.approx(0.1) and set(kk) == set(TURN)
    for c in TURN:
        t_, v_ = kk[c]
        assert np.isfinite(v_).sum() > 20
        np.testing.assert_allclose(v_[np.isfinite(v_)], w, atol=1e-9)
    t_, v_ = kk['wz_cnt']
    assert np.isnan(v_).sum() >= 1 + 4 * 10                              # the break and the ends of the two runs
    assert np.isnan(np.interp(101.505, t_, v_))                           # the missing pair is not bridged
    # 30 fps camera: the window keeps +-6 frames = 0.2 s
    assert G.kenh_kiem(tl, ip, rate, tg, np.full(80, w), tr, w * tr, 1 / 30, (101.0, 103.0))[1]['nua_cua_so_mau_log'] == 20


def test_no_smoothed_sample_is_not_ok():
    t = np.arange(0, 30, 1 / 60)
    v = np.sin(0.5 * t)
    r = G.match(t, v, np.ones_like(t, bool), t, v, (0.0, 29.0), 0.0, kiem=(np.zeros(0), np.zeros(0)))
    assert not r['ok'] and r['corr_kiem'] is None and r['corr'] > 0.99
    for nm in ('whole window', 'start third', 'end third'):
        assert f'{nm} (log smoothed as the camera): no overlap' in r['why']
    r2 = G.match(t, v, np.ones_like(t, bool), t, v, (0.0, 29.0), 0.0, kiem=(t, v))
    assert r2['ok'] and r2['corr_kiem_end'] > 0.99
    r3 = G.match(t, v, np.ones_like(t, bool), t, v, (0.0, 29.0), 0.0)
    assert r3['ok'] and 'corr_kiem' not in r3 and {k: r2[k] for k in r3 if k != 'ok'} == {k: r3[k] for k in r3 if k != 'ok'}
