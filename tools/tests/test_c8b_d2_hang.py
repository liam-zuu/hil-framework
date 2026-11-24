"""c8b_d2_hang: the fast U_IMU,D equals c8b_u_imu_v3.u_imu_delta, the leave-one-out inputs (design C8B v7 7b, 7c)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c8_u_imu as U  # noqa: E402
import c8b_d2_hang as DH  # noqa: E402
import c8b_u_imu_v3 as V  # noqa: E402
import test_c8b_u_imu_v3 as TV  # noqa: E402


def test_turn_sums_equal_turn_totals(tmp_path):
    S, _ = TV._S(tmp_path)
    ts = DH.TurnSums(S)
    rng = np.random.default_rng(1)
    for _ in range(300):
        a, b = sorted(rng.uniform(S['t_all'][0] - 0.5, S['t_all'][-1] + 0.5, 2))
        try:
            want = U.turn_totals(S, a, b)
        except U.InputError:
            with pytest.raises(U.InputError):
                ts(a, b)
            continue
        got = ts(a, b)
        assert got[0] == pytest.approx(want[0], abs=1e-12) and got[1] == pytest.approx(want[1], abs=1e-12)


def test_turn_sums_suspect_pairs(tmp_path):
    # a log with dropped reports (suspect pairs get the gyro extra on both P and N)
    p = str(tmp_path / 'd.csv')
    TV.T.make_log(p, TV._rate, TV.A + 9.0, run=(TV.A, TV.A + 8.0), miss=[100, 101, 150], gyro_miss=())
    S = V.rv_samples_age(U.load_log(p))
    ts = DH.TurnSums(S)
    for a, b in ((TV.A - 2.0, TV.A + 6.0), (TV.A + 1.0, TV.A + 1.2), (TV.A + 2.0, TV.A + 7.5)):
        want = U.turn_totals(S, a, b)
        got = ts(a, b)
        assert got == pytest.approx(want, abs=1e-12)


def test_unode_equals_u_imu_delta(tmp_path):
    S, _ = TV._S(tmp_path)
    P3 = TV._P3()
    t_W = np.array([TV.A - 1.0, TV.A - 0.95, TV.A - 0.9, TV.A - 0.85])
    om_W, al_W = np.zeros(4), np.zeros(4)
    un = DH.UNode(P3, S, t_W, om_W, al_W, 0.01, 0.01)
    for t_k, om, al in ((TV.A + 2.0, 1.0, 0.0), (TV.A + 4.0, -1.0, 0.5), (TV.A + 5.5, 0.0, 0.0),
                        (TV.A + 50.0, 0.0, 0.0), (TV.A + 2.0, 9.0, 0.0)):
        want = V.u_imu_delta(P3, S, t_k, t_W, om, al, om_W, al_W, 0.01, 0.01)
        got = un(t_k, om, al)
        if want[0] is None:
            assert got[0] is None
        else:
            assert got[0] == pytest.approx(want[0], rel=1e-12)
            for k in ('thang', 'troi', 'q', 'tre', 'tre_W', 'noi', 'noi_W', 'chuyen', 'P_bar', 'N_bar'):
                assert got[1][k] == pytest.approx(want[1][k], rel=1e-12, abs=1e-15)


def test_loo_lags_are_medians_of_the_others():
    tre = dict(tau_hat_s=0.009, runs=[dict(tau_s=t, log=f'r{i}.csv') for i, t in
                                      enumerate((0.007, 0.009, 0.011, 0.010, 0.008))])
    fs = DH.loo_files(tre, dict(delta_plus=0.002, delta_minus=0.003), dict(beta_max=1e-4))
    assert [f['tau_hat_s'] for f in fs] == pytest.approx([0.0095, 0.009, 0.0085, 0.0085, 0.0095])
    ch = V.chuyen_spread(fs)
    assert ch['d_tau_s'] == pytest.approx(0.001) and ch['d_delta_plus'] == 0.0 and ch['n_loo'] == 5
    with pytest.raises(DH.D2Error):
        DH.loo_files(dict(tre, tau_hat_s=0.0091), {'delta_plus': 0, 'delta_minus': 0}, {'beta_max': 0})


def test_noi_loo_leaves_the_target_out(tmp_path):
    caps = []
    for i, w in enumerate((0.5, 1.0, 1.5)):
        p = str(tmp_path / f'l{i}.csv')
        TV.T.make_log(p, lambda t, w=w: w if TV.A <= t < TV.A + 3.0 else 0.0, TV.A + 6.0, run=(TV.A, TV.A + 5.0))
        fp = str(tmp_path / f'f{i}.csv')
        TV.T.write_frames(fp, lambda t, w=w: w if TV.A <= t < TV.A + 3.0 else 0.0, 10.0, TV.A + 6.0, offset=0.5,
                          psi0=0.0)
        caps.append((U.read_frames(fp), U.load_log(p), 0.5, f'r{i}'))
    P = dict(tau_hat_s=0.0, s_plus=0.0, s_minus=0.0)
    out = DH.noi_loo(caps, ['r0', 'r2'], P)
    for r in ('r0', 'r2'):
        want = V.noi_table([c for c in caps if c[3] != r], P)
        assert out[r]['luot'] == want['luot'] and r not in out[r]['luot']
        assert {k: v['max_rad'] for k, v in out[r]['bang'].items()} == \
            pytest.approx({k: v['max_rad'] for k, v in want['bang'].items()})
    with pytest.raises(DH.D2Error):
        DH.noi_loo(caps, ['r7'], P)                                    # a target without its capture


# ── Codex D2 round 1 (24/11 khuya) ───────────────────────────────────────────────────────────────────────────────

def test_turn_sums_nan_gyro_outside_the_window(tmp_path):
    """D2-12: a NaN gyro report inside a suspect pair must not reach the totals of a window that does not hold the
    pair (the prefix sums carried it into every later window); a window holding the pair is an error, not a NaN."""
    p = str(tmp_path / 'd.csv')
    TV.T.make_log(p, TV._rate, TV.A + 9.0, run=(TV.A, TV.A + 8.0), miss=[100, 101, 150], gyro_miss=())
    S = V.rv_samples_age(U.load_log(p))
    ta, tr, na = S['t_all'], S['t_recv'], S['n_all']
    k = int(np.argmax(np.diff(na) > 1))                             # the first suspect pair
    a, b = min(ta[k], tr[k]), max(ta[k + 1], tr[k + 1])
    j = int(np.searchsorted(S['tg'], 0.5 * (a + b)))
    S2 = dict(S, gz=S['gz'].copy())
    S2['gz'][j] = np.nan
    ts = DH.TurnSums(S2)
    assert ts.mono
    for w0, w1 in ((b + 0.2, b + 1.0), (b + 0.3, S['t_all'][-1] - 0.1)):   # after the pair
        got = ts(w0, w1)
        assert all(math.isfinite(x) for x in got) and got == pytest.approx(U.turn_totals(S, w0, w1), abs=1e-12)
    with pytest.raises(U.InputError):
        ts(a - 0.1, b + 0.1)


def test_unode_never_gives_a_bound_that_is_not_a_number(tmp_path):
    """D2-12: a term that is not finite gives no bound (None with a reason), never NaN in the rows."""
    S, _ = TV._S(tmp_path)
    t_W = np.array([TV.A - 1.0, TV.A - 0.95, TV.A - 0.9, TV.A - 0.85])
    P3 = dict(TV._P3())
    assert DH.UNode(P3, S, t_W, np.zeros(4), np.zeros(4), 0.01, 0.01)(TV.A + 2.0, 1.0, 0.0)[0] is not None
    P3['q'] = math.nan
    U_, _, why = DH.UNode(P3, S, t_W, np.zeros(4), np.zeros(4), 0.01, 0.01)(TV.A + 2.0, 1.0, 0.0)
    assert U_ is None and any('not finite' in w for w in why)


def test_noi_loo_one_capture_under_two_names_refused(tmp_path):
    """D2-13: the same frames or log file given as two runs would stay in the table that leaves 'its' run out."""
    import json
    fps, lps = [], []
    for i, w in enumerate((0.5, 1.0)):
        p = str(tmp_path / f'l{i}.csv')
        TV.T.make_log(p, lambda t, w=w: w if TV.A <= t < TV.A + 3.0 else 0.0, TV.A + 6.0, run=(TV.A, TV.A + 5.0))
        fp = str(tmp_path / f'f{i}.csv')
        TV.T.write_frames(fp, lambda t, w=w: w if TV.A <= t < TV.A + 3.0 else 0.0, 10.0, TV.A + 6.0, offset=0.5,
                          psi0=0.0)
        fps.append(fp)
        lps.append(p)
    th, tr = tmp_path / 'thang.json', tmp_path / 'tre.json'
    th.write_text(json.dumps(dict(s_plus=0.0, s_minus=0.0)), encoding='utf-8')
    tr.write_text(json.dumps(dict(tau_hat_s=0.0)), encoding='utf-8')

    def chay(caps, out):
        return DH.main(['noi-loo'] + sum((['--cap', c] for c in caps), []) + ['--dich', 'r0', '--thang', str(th),
                                                                             '--tre', str(tr), '--out-dir', str(out)])
    assert chay([f'{fps[0]},{lps[0]},0.5,r0', f'{fps[1]},{lps[1]},0.5,r1'], tmp_path / 'a') == 0
    assert chay([f'{fps[0]},{lps[0]},0.5,r0', f'{fps[0]},{lps[0]},0.5,r0_alias'], tmp_path / 'b') == 2
    assert chay([f'{fps[0]},{lps[0]},0.5,r0', f'{fps[1]},{lps[0]},0.5,r1'], tmp_path / 'c') == 2     # the same log


def test_hang_needs_a_mounting_name():
    """D2-15: no default mounting id (two mountings under one name would be counted in one cell)."""
    with pytest.raises(SystemExit):
        DH.main(['hang', '--luot', 'R', '--luoi-dir', 'x', '--gio', 'g', '--frames', 'f', '--esp32', 'e',
                 '--hang-so', 'h', '--tham-chieu', 't', '--u-imu', 'u', '--khoa', 'k', '--out', 'o'])
    assert 'mien_hop' in DH.COLS


def test_u_psi_of_a_frame_before_the_margin():
    """L156 (2), Codex D2-9: the heading bound of a frame = the lock's constant term before the margin + A4 of its
    speed band / le; standing frame: A4 = 0; beyond the bands or a band without a number: no bound."""
    le, v_d, a = 1.25, 0.01, 0.1445
    m = dict(hang_dong=dict(a_m=a), hang_dong_truoc_le=dict(hang_rad=0.030),
             hang_theo_u=[dict(u_lo_m_s=0.0, u_hi_m_s=0.358, A4_deg=0.5, trang_thai='gioi_han'),
                          dict(u_lo_m_s=0.358, u_hi_m_s=0.896, A4_deg=1.0, trang_thai='gioi_han'),
                          dict(u_lo_m_s=0.896, u_hi_m_s=1.433, A4_deg=None, trang_thai='thieu')])
    f = DH.u_psi_khung
    assert f(m, le, v_d, 0.0, 0.0) == (0.030, None)                                  # standing
    assert f(m, le, v_d, 0.005, 0.02)[0] == 0.030                                    # u = 0.005 + 0.002 < v_d
    assert f(m, le, v_d, 0.2, 0.0)[0] == pytest.approx(0.030 + math.radians(0.5) / 1.25, rel=1e-12)
    assert f(m, le, v_d, -0.2, 0.0)[0] == f(m, le, v_d, 0.2, 0.0)[0]
    u = 0.3 + 1.0 * a / math.sqrt(2.0)                                               # 0.402: the second band by wz
    assert u > 0.358 and f(m, le, v_d, 0.3, -1.0)[0] == pytest.approx(0.030 + math.radians(1.0) / 1.25, rel=1e-12)
    # an edge is of the LOWER band, as c8b_p6_ngang.in_band counts the frames of the lock (Codex round 2 D2v2-6)
    assert f(m, le, v_d, 0.358, 0.0)[0] == pytest.approx(0.030 + math.radians(0.5) / 1.25, rel=1e-12)
    assert f(m, le, v_d, 0.896, 0.0)[0] == pytest.approx(0.030 + math.radians(1.0) / 1.25, rel=1e-12)
    # u given (the band speed of frames.csv, as P6): it decides, not the speed and wz of the node rows
    assert f(m, le, v_d, 0.011, 0.0, u=0.009)[0] == 0.030 and f(m, le, v_d, 0.009, 0.0, u=0.011)[0] > 0.030
    assert f(m, le, v_d, 0.2, 0.0, u=0.5)[0] == pytest.approx(0.030 + math.radians(1.0) / 1.25, rel=1e-12)
    assert f(m, le, v_d, 0.2, 0.0, u=math.nan)[0] is None and f(m, le, v_d, math.nan, 0.0, u=0.2)[0] is not None
    for sp, wz in ((1.0, 0.0), (1.5, 0.0), (0.2, math.nan), (math.nan, 0.0)):        # band thieu, beyond, no rate
        b, why = f(m, le, v_d, sp, wz)
        assert b is None and why
    assert f(dict(m, hang_dong_truoc_le=dict(hang_rad=None)), le, v_d, 0.2, 0.0)[0] is None
    assert f(m, 0.9, v_d, 0.2, 0.0)[0] is None                                       # a margin below 1 is not a margin
    # smaller than the scorer's constant with the margin on the same numbers: the check is tighter
    assert f(m, le, v_d, 0.5, 0.0)[0] < le * 0.030 + math.radians(1.0)
    khoa = dict(muc={'O1-F05': 1, 'O1-F08': 2})
    assert DH.muc_cua_task(khoa, 'O1-F08') == 'O1-F08' and DH.muc_cua_task(khoa, 'O1-K1') == 'O1-F05'
    with pytest.raises(DH.D2Error):
        DH.muc_cua_task(khoa, 'O1-F02')


def test_turn_sums_absurd_gyro_value_is_an_error_of_its_pair(tmp_path):
    """Codex round 2 D2v2-8: a finite but absurd gyro value (1e20) in a suspect pair swallowed the later steps in the
    prefix sums (0 instead of the turn); it is an error of its pair, the windows after it are those of turn_totals."""
    p = str(tmp_path / 'd.csv')
    TV.T.make_log(p, TV._rate, TV.A + 9.0, run=(TV.A, TV.A + 8.0), miss=[100, 101, 150], gyro_miss=())
    S = V.rv_samples_age(U.load_log(p))
    ta, tr, na = S['t_all'], S['t_recv'], S['n_all']
    k = int(np.argmax(np.diff(na) > 1))
    a, b = min(ta[k], tr[k]), max(ta[k + 1], tr[k + 1])
    j = int(np.searchsorted(S['tg'], 0.5 * (a + b)))
    S2 = dict(S, gz=S['gz'].copy())
    S2['gz'][j] = 1e20
    ts = DH.TurnSums(S2)
    assert ts.mono
    for w0, w1 in ((b + 0.2, b + 1.0), (b + 0.3, S['t_all'][-1] - 0.1)):
        got, want = ts(w0, w1), U.turn_totals(S, w0, w1)
        assert got == pytest.approx(want, abs=1e-12) and max(want) > 0.01
    with pytest.raises(U.InputError):
        ts(a - 0.1, b + 0.1)
