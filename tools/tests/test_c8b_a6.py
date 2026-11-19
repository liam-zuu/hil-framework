"""c8b_a6 (M4: A6 from the E clips, design C8B v7 sections 2, 3, 10; J v6), after Codex round 1/2 (A6C1-1..15).
Synthetic E clips through the exact lift of car_pose (lifted_corners) and a top-down camera; an independent oracle
through c7_clip_probe.Camera.project and c7_measure.car_pose (A6C1-15)."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_a6 as A  # noqa: E402

C = np.array([-0.4802, -0.0329])          # the plumb point of 16/11 (L125)
H, HB, SIDE = 2.632, 0.124175, 0.1445
GC = A.g_cal()[0]
CASES = ['bo:18-19-30', 'bo:18-19', 'bo:18-30']
KHOA = 'k' * 64
SCRIPT = dict(commit='abc', dirty=False)


class TopCam:
    """Pinhole looking straight down from (C, H): rows along +y, f 1700 px."""
    centre = np.array([C[0], C[1], H])

    def project(self, P):
        P = np.asarray(P, float)
        return np.column_stack([1920 + 1700 * (P[:, 0] - C[0]) / (H - P[:, 2]),
                                1080 + 1700 * (P[:, 1] - C[1]) / (H - P[:, 2])])


def _ses():
    return dict(C=C, H=H, hbar=HB, side=SIDE, kappa=HB / H, sha256='0' * 64, file='s.json', camera=None)


def _clip(g=GC, v=-0.000475, rate=0.3, n=900, centre=(0.6, 0.4), noise=0.0001, seed=0, sigma=0.002, fps=30.0):
    """E clip: the car turning in place about centre (offset from C), tag 50 mm from the turning centre."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    psi = rate * t
    cen = C + np.asarray(centre, float)
    P = np.array([A.lifted_corners(cen + 0.05 * np.array([math.cos(p), math.sin(p)]), p, g, v, C, H, HB, SIDE,
                                   sigma=sigma) for p in psi])
    return np.arange(n) + 100, t, P + rng.normal(0, noise, P.shape)


def _f15(idx, t, scale=0.0001, seed=1, schema=A.F15_FULL, trang_thai='dat', cho='goc', fields=None):
    rng = np.random.default_rng(seed)
    fields = fields or {c: rng.normal(0, scale, (4, 2)) for c in CASES}          # smooth: the same at every frame
    khung = [(int(k), float(tt), {c: (f(int(k)) if callable(f) else f) for c, f in fields.items()})
             for k, tt in zip(idx, t)]
    return A._f15_obj(schema, trang_thai, cho, dict(sau_E_dau=int(idx[0]) - 1, truoc_E_cuoi=int(idx[-1]) + 1),
                      list(fields), khung, 'c' * 64, '1' * 64, 'f15.json', '2' * 64, [])


def _do(cho='goc', idx=None, t=None, P=None, f15=None, tam='T1', lan='L1', **kw):
    if idx is None:
        idx, t, P = _clip(**kw)
    return A.do_place(cho, idx, t, P, _ses(), f15 if f15 is not None else _f15(idx, t, cho=cho), TopCam(), tam, lan)


def test_g_cal_from_the_four_caliper_heights():
    g, v, h = A.g_cal()
    assert g == pytest.approx([-0.00519, 0.01349], abs=1e-5) and v == pytest.approx(-0.000475) and h == pytest.approx(HB)
    c, n = A.u_cal()
    assert c == pytest.approx(0.00138, abs=1e-5) and n == pytest.approx(c * math.sqrt(2))


@pytest.mark.parametrize('T,psi', [((0.5, 0.2), 0.3), ((-1.2, 0.7), 2.0), ((0.1, -1.0), -1.1), ((0.9, 0.9), 3.0)])
def test_model_sign_against_the_exact_lift(T, psi):
    Pc = A.lifted_corners(T, psi, GC, 0.0, C, H, HB, SIDE)
    ex = math.remainder(A.kabsch_psi(Pc, SIDE) - psi, 2 * math.pi)
    assert ex == pytest.approx(A.dpsi_model(GC, T, psi, C, H, HB), abs=2e-5)
    assert abs(ex) > 1e-3


def _real_cam(cc, k1=-0.05):
    import c7_clip_probe as cp
    R = np.diag([1.0, -1.0, -1.0])                                    # looking down: rvec (pi, 0, 0)
    return cp.Camera(model='B1', f=1700.0, cx=1920.0, cy=1080.0, k=[k1], rvec=[math.pi, 0.0, 0.0],
                     tvec=list(-R @ np.asarray(cc, float)))


def test_ebc_errors_equal_the_real_car_pose_oracle():
    # A6C1-9, A6C1-15: the e_bc path = Camera.project of the corners at their heights + c7_measure.car_pose (pix2floor,
    # kappa lift about the plumb, Kabsch), the model evaluated at the MEASURED pose; camera centre off the plumb
    import c7_measure as cm
    cc = np.array([C[0] + 0.02, C[1] - 0.015, H + 0.012])
    cam = _real_cam(cc)
    assert cam.centre == pytest.approx(cc)
    rng = np.random.default_rng(3)
    for _ in range(12):
        T = C + rng.uniform(-1.1, 1.1, 2)
        S, g, v = rng.uniform(-math.pi, math.pi), GC + rng.normal(0, 0.003, 2), rng.uniform(-A.V_MAX_M, A.V_MAX_M)
        X = T + A.body(SIDE) @ A._rot(S).T
        h = HB + A.body(SIDE) @ g + v * A.S_PATTERN
        pix = cam.project(np.column_stack([X, h]))
        r = cm.car_pose(cam, pix, SIDE, HB / H, C)
        want = math.remainder(r['tag_psi'] - S - A.dpsi_model(g, (r['tag_x'], r['tag_y']), r['tag_psi'], C, H, HB),
                              2 * math.pi)
        got = A.ebc_errors(T[None], np.array([S]), g[None], np.array([v]), C, H, HB, SIDE, cc)[0]
        assert got == pytest.approx(want, abs=1e-8)


def test_ebc_refined_and_not_converged_is_refused(monkeypatch):
    v, lv = A.e_bc(GC, 0.003, C, H, HB, SIDE, [C[0] + 0.02, C[1] - 0.01, H + 0.01])
    assert v is not None and len(lv) >= 2 and v == lv[-1][1] and lv[0][0] == A.EBC_K0
    calls = []

    def doubling(*a, **k):                                           # the maximum doubles at every halving
        calls.append(1)
        return 1e-4 * 2 ** len(calls)
    monkeypatch.setattr(A, '_ebc_level', doubling)
    v2, lv2 = A.e_bc(GC, 0.003, C, H, HB, SIDE, [C[0], C[1], H])
    assert v2 is None and [x[0] for x in lv2] == [4, 8, 16]


def test_fit_g_recovers_the_tilt_and_v_does_not_bias_it():
    _, _, P = _clip(noise=0.0, v=0.0)
    f = A.fit_g(P, C, H, HB, SIDE)
    assert f['g'] == pytest.approx(GC, abs=1e-6) and f['sigma'] == pytest.approx(0.002, abs=1e-6) and A.fit_ok(f)
    _, _, P2 = _clip(noise=0.0, v=-0.000475)
    f2 = A.fit_g(P2, C, H, HB, SIDE)
    assert f2['g'] == pytest.approx(GC, abs=1e-6) and f2['v_phu'] == pytest.approx(-0.000475, abs=1e-7)


def test_do_place_full_f15_identified():
    r = _do()
    assert r['trang_thai'] == 'dat', r['ly_do']
    assert np.linalg.norm(np.array(r['g_E']) - GC) < 4 * np.linalg.norm(r['SE_E'])
    assert all(u > 0 for u in r['U_meo_E']) and r['s_E'] == pytest.approx(list(2 * np.array(r['SE_E'])
                                                                              + np.array(r['U_meo_E'])))
    assert set(r['U_meo_E_theo_ca']) == set(CASES) and r['tam_id'] == 'T1' and r['lan_lap_id'] == 'L1'
    assert r['fit']['hoi_tu'] and r['nhan_dang']['dat'] and r['n_khung'] >= A.N_MIN
    # U_meo_E is the largest |change| over the cases, the signed change per case is kept (A6C1-3)
    assert r['U_meo_E'] == pytest.approx(np.max(np.abs(list(r['U_meo_E_theo_ca'].values())), axis=0).tolist())


@pytest.mark.parametrize('schema,tt', [(A.F15_DIAG, 'chan_doan'), (A.F15_FULL, 'thieu'), (A.F15_FULL, 'chan_doan')])
def test_do_place_f15_not_full_dat_is_not_a_bound(schema, tt):
    # A6C1-1: only the full schema WITH trang_thai dat gives U_meo_E, s_E; the diagnostic numbers stay apart
    idx, t, P = _clip()
    r = _do(idx=idx, t=t, P=P, f15=_f15(idx, t, schema=schema, trang_thai=tt))
    assert r['trang_thai'] == 'thieu' and any('diagnostic' in w for w in r['ly_do'])
    assert r['U_meo_E'] is None and r['s_E'] is None and r['s_E_norm'] is None
    assert len(r['chan_doan']['U_meo_E']) == 2
    out = A.gop([r], 'T1', 'T1', with_ebc=False)
    assert out['eps_g'] is None and out['h_goc_m'] is None and out['u_h_goc_m'] is None
    assert out['trang_thai'] == dict(a6='thieu', h='thieu')


def test_f15_window_cases_and_times_checked():
    # A6C1-2: no E window, a case missing at EVERY frame, another clip's times -> refused
    idx, t, P = _clip()
    with pytest.raises(A.A6Error, match='mien_E'):
        A._f15_obj(A.F15_FULL, 'dat', 'goc', None, CASES, [], 'c' * 64, '1' * 64, 'f', '2' * 64, [])
    with pytest.raises(A.A6Error, match='H32 list'):
        A._f15_obj(A.F15_FULL, 'dat', 'goc', dict(sau_E_dau=1, truoc_E_cuoi=5), CASES,
                   [(2, 0.1, {c: np.zeros((4, 2)) for c in CASES[:2]})], 'c' * 64, '1' * 64, 'f', '2' * 64, [])
    f = _f15(idx, t + 0.01)
    with pytest.raises(A.A6Error, match='times differ'):
        _do(idx=idx, t=t, P=P, f15=f)
    f = _f15(idx[:-50], t[:-50])
    f['mien_E'] = dict(sau_E_dau=int(idx[0]) - 1, truoc_E_cuoi=int(idx[-1]) + 1)
    with pytest.raises(A.A6Error, match='lacks'):
        _do(idx=idx, t=t, P=P, f15=f)
    with pytest.raises(A.A6Error, match='F15'):
        A.do_place('goc', idx, t, P, _ses(), None, TopCam(), 'T1', 'L1')
    with pytest.raises(A.A6Error, match='camera'):
        A.do_place('goc', idx, t, P, _ses(), _f15(idx, t), None, 'T1', 'L1')
    with pytest.raises(A.A6Error, match='tam_id'):
        A.do_place('goc', idx, t, P, _ses(), _f15(idx, t), TopCam(), '', 'L1')


def test_identification_per_component():
    # A6C1-11: a case moving g_x by more than |g_x| while the norm of the change stays under |g| -> not identified
    idx, t, P = _clip()
    r0 = _do(idx=idx, t=t, P=P)
    gx = abs(r0['g_E'][0])

    def field(k, _P=dict(zip(idx.tolist(), P))):                       # g_x + 0.009 at every frame, through its pose
        p = _P[k]
        psi = A.kabsch_psi(p, SIDE)
        T = p.mean(0)
        return A.lifted_corners(T, psi, np.array(r0['g_E']) + [0.009, 0], 0, C, H, HB, SIDE) \
            - A.lifted_corners(T, psi, r0['g_E'], 0, C, H, HB, SIDE)
    f = _f15(idx, t, fields={'bo:1': field, 'bo:2': np.zeros((4, 2))})
    r = _do(idx=idx, t=t, P=P, f15=f)
    assert 0.009 > gx and np.linalg.norm(r['nhan_dang']['dg_max']) < np.linalg.norm(r0['g_E']) * 1.0 + 0.01
    assert r['trang_thai'] == 'thieu' and not r['nhan_dang']['dat'] and any('per component' in w for w in r['ly_do'])
    # the replay goes FROM full TO held (P + D): the signed change of that case is +0.009 on g_x (A6C1-15: P - D)
    assert r['U_meo_E_theo_ca']['bo:1'] == pytest.approx([0.009, 0.0], abs=3e-4)


def test_singular_fit_on_still_frames():
    # A6C1-11: frames standing still pass the rate filter but the shutter fields are zero: singular -> not usable
    idx, t, P = _clip(rate=0.0, n=300, noise=0.0)
    r = _do(idx=idx, t=t, P=P)
    assert r['trang_thai'] == 'thieu' and any('SE alpha' in w for w in r['ly_do'])


def test_omega_from_the_smoothed_heading_and_gaps():
    # A6C1-12: a turn just above the limit is out (no two-frame noise lets frames in); a gap leaves no rate
    idx, t, P = _clip(rate=1.05, n=600, fps=60.0)
    w = A.omega_of(t, P, SIDE)
    assert np.nanmax(np.abs(w)) > 1.0 and np.sum(np.abs(w) <= A.OMEGA_MAX_RAD_S) == 0
    t2 = t.copy()
    t2[300:] += 0.2
    w2 = A.omega_of(t2, P, SIDE)
    assert np.all(np.isnan(w2[295:305])) and np.isfinite(w2[200])
    r = _do(idx=idx, t=t, P=P)
    assert r['trang_thai'] == 'thieu' and r['g_E'] is None and any('omega' in w for w in r['ly_do'])


def _shuttered(alpha_r, alpha_c, rate=0.3, **kw):
    idx, t, P = _clip(rate=rate, **kw)
    fr, fc = A.shutter_fields(P, t, A.uv_of(P, TopCam(), C, HB / H))
    return idx, t, P + alpha_r * fr + alpha_c * fc


def test_l133_alpha_fitted_with_its_sign():
    idx, t, P = _shuttered(-0.72e-6, 2.0e-6, noise=0.00002)
    r = _do(idx=idx, t=t, P=P, f15=_f15(idx, t, scale=0.00002))
    assert r['trang_thai'] == 'dat', r['ly_do']
    mt = r['man_trap']
    assert mt['alpha_r_s_px'] == pytest.approx(-0.72e-6, abs=0.25e-6) and mt['alpha_r_s_px'] < 0
    assert mt['alpha_c_s_px'] == pytest.approx(2.0e-6, abs=0.25e-6)
    assert np.linalg.norm(np.array(r['g_E']) - GC) < 3 * np.linalg.norm(r['SE_E'])
    assert len(mt['SE_E_khong_alpha']) == 2 and 0 <= mt['tuong_quan_alpha_g_max'] < 1


def test_l133_alpha_removes_the_shutter_bias():
    idx, t, P = _shuttered(4.5e-6, 4.5e-6, rate=0.95, n=900)
    r = _do(idx=idx, t=t, P=P)
    assert r['trang_thai'] == 'dat', r['ly_do']
    assert np.linalg.norm(np.array(r['g_E']) - GC) < 3 * np.linalg.norm(r['SE_E'])


# ── gop ──────────────────────────────────────────────────────────────────────────────────────────

def _lap(u=0.0002, **kw):
    d = dict(schema=A.SCHEMA_LAP, tam_id='T1', trang_thai='dat',
             cap=[dict(lan_lap_a='L1', lan_lap_b='L2', huong_lech_deg=2.0, khoang_m=0.1, g_a=list(GC),
                       g_b=list(GC + [u, 0.0]), SE_a=[0.0001, 0.0001], SE_b=[0.0001, 0.0001])])
    d.update(kw)
    return A.with_sha(d)


def _dong(g_hat, luot=('r1', 'r2'), n=500, **kw):
    dn = [dict(ten='a0', a_lo=0.0, a_hi=1.0, nua=False), dict(ten='a0_lo', a_lo=0.0, a_hi=0.5, nua=True),
          dict(ten='a0_hi', a_lo=0.5, a_hi=1.0, nua=True)]
    d = dict(schema=A.SCHEMA_DONG, tam_id='T1', trang_thai='dat', dai_dinh_nghia=dn,
             luot={r: dict(bo_luot=[x for x in luot if x != r] or ['r9'],
                           dai={b['ten']: dict(n_khung=n, g_bar=list(np.asarray(g_hat) + [0.0003, 0.0]),
                                               SE=[0.0001, 0.0001]) for b in dn}) for r in luot})
    d.update(kw)
    return A.with_sha(d)


def _rows_o(d=0.002, up=0.003, ui=0.002):
    """Parsed rows (c8b_a6_o.read_rows) of the held-out check: three cells of the test set, khop c1, dich r1, r2."""
    out = []
    for luot, tap, n, xs in (('c2', 'kiem', 12, (0.1, 0.7, 1.3)), ('c1', 'khop', 3, (0.1,)), ('r1', 'dich', 3, (0.1,)),
                             ('r2', 'dich', 3, (0.1,))):
        for j, x in enumerate(xs):
            for i in range(n):
                for nd in ('0', '1'):
                    out.append(dict(luot=luot, tap=tap, id=1000 * j + i, nut=nd, x=x, y=0.2, psi=0.3, u=0.4, lan='L1',
                                    d=d if nd == '0' else -d / 2, up=up, ui=ui, ctrl='PID'))
    return out


def _o(cho_sha, rows=None, **kw):
    import c8b_a6_o as O
    d = O.build(_rows_o() if rows is None else rows, 'giu_ngoai_a6', 'cho', KHOA, [cho_sha[0]])
    d.update(kw)
    return A.with_sha(d)


LOCK = dict(tap=dict(khop=['c1'], kiem=['c2'], dich=['r1', 'r2']))


def _uh(**kw):
    d = dict(schema=A.SCHEMA_UH, tam_id='T1', trang_thai='dat', u_h_chung_m=0.0003, h_m=HB, n_ga_giu_ngoai=3,
             phan=dict(do_trai_ba_ga_m=0.0001, lech_video_so_tay_m=0.0001, hang_m=0.0001))
    d.update(kw)
    return A.with_sha(d)


@pytest.fixture(scope='module')
def places():
    return _do('goc'), _do('K1', seed=3, centre=(-0.5, 0.7))


def _gop(pl, **kw):
    sha = ['a' * 64, 'b' * 64]
    g_hat = np.mean([p['g_E'] for p in pl], axis=0)
    base = dict(lap=_lap(), dong=_dong(g_hat), giu_ngoai=_o(sha), u_h=_uh(), khoa_truoc_sha256=KHOA, script=SCRIPT,
                cho_sha=sha, with_ebc=True, lan_lap_cal_id='L1', giu_ngoai_rows=_rows_o(), khoa=LOCK)
    base.update(kw)
    return A.gop(list(pl), 'T1', base.pop('tam_cal_id', 'T1'), **base)


def test_gop_dat_and_eps_g(places):
    pg, pk = places
    out = _gop(places)
    assert out['trang_thai'] == dict(a6='dat', h='dat'), out['ly_do']
    s = max(pg['s_E_norm'], pk['s_E_norm'])
    u = out['U_chuyen']
    assert u['lap'] == pytest.approx(0.0002 + 2 * math.sqrt(4e-8))
    assert u['cho'] == pytest.approx(np.linalg.norm(np.array(pg['g_E']) - np.array(pk['g_E'])))
    g_hat = np.array(out['g_b'])
    assert u['dong'] == pytest.approx(np.linalg.norm(np.mean([pg['g_E'], pk['g_E']], 0) + [0.0003, 0] - g_hat)
                                      + 2 * math.sqrt(2e-8))
    dist = max(np.linalg.norm(np.array(p['g_E']) - GC) for p in places)
    assert out['khoang_g_cal'] == pytest.approx(dist)
    assert out['eps_g'] == pytest.approx(max(s, dist + A.u_cal()[1]) + u['lap'] + u['cho'] + u['dong'])
    assert out['U_cal'] == pytest.approx(A.u_cal()[1]) and out['U_cal_thanh_phan'] == pytest.approx(A.u_cal()[0])
    assert out['e_bc_rad'] is not None and out['e_bc_rad'] < 1e-3 and len(out['e_bc_lan_chia']) >= 2
    assert out['h_goc_m'] == pytest.approx([HB + ri @ g_hat for ri in A.body(SIDE)])
    assert out['u_h_goc_m'][0] == pytest.approx(0.0003 + np.linalg.norm(A.body(SIDE)[0]) * out['eps_g'] + A.V_MAX_M)
    assert out['khoa_truoc_sha256'] == KHOA and out['commit'] == 'abc' and out['nhat_quan'] is True
    assert out['khoa_truoc_sha'] == KHOA                                  # F3 2.C name


@pytest.mark.parametrize('kw,txt', [
    (dict(lap=None), 'lap'), (dict(dong=None), 'dong'), (dict(giu_ngoai=None), 'held-out'),
    (dict(khoa_truoc_sha256=None), 'khoa_truoc'), (dict(script=dict(commit='abc', dirty=True)), 'dirty'),
    (dict(script=None), 'commit'), (dict(u_h=None), None)])
def test_gop_missing_part(places, kw, txt):
    o = _gop(places, **kw)
    if txt is None:                                     # u_h_chung missing: h thieu, A6 still dat
        assert o['trang_thai'] == dict(a6='dat', h='thieu') and o['h_goc_m'] is None
        return
    assert o['trang_thai']['a6'] == 'thieu' and any(txt in w for w in o['ly_do']), o['ly_do']
    assert o['trang_thai']['h'] == 'thieu' and o['h_goc_m'] is None


def test_gop_sources_checked(places):
    # A6C1-5: negative / malformed U_chuyen sources refused; the target run left out
    sha = ['a' * 64, 'b' * 64]
    g_hat = np.mean([p['g_E'] for p in places], axis=0)
    for lap, txt in ((_lap(cap=[dict(_lap()['cap'][0], SE_a=[-0.01, 0.0])]), 'SE_a'),
                     (_lap(cap=[dict(_lap()['cap'][0], lan_lap_b='L1')]), 'mountings'),
                     (_lap(cap=[dict(_lap()['cap'][0], huong_lech_deg=9.0)]), 'heading'),
                     (_lap(trang_thai='thieu'), 'trang_thai'), (_lap(tam_id='T2'), 'plate')):
        o = _gop(places, lap=lap)
        assert o['trang_thai']['a6'] == 'thieu' and o['U_chuyen']['lap'] is None and any(txt in w for w in o['ly_do'])
    d = _dong(g_hat)
    d['luot']['r1']['bo_luot'] = ['r1', 'r2']
    o = _gop(places, dong=A.with_sha(d))
    assert o['trang_thai']['a6'] == 'thieu' and any('every other run' in w for w in o['ly_do'])
    d = _dong(g_hat)
    d['luot']['r1']['bo_luot'] = ['unrelated']                          # A6C2-4: not the other runs of the file
    o = _gop(places, dong=A.with_sha(d))
    assert o['trang_thai']['a6'] == 'thieu' and any('every other run' in w for w in o['ly_do'])
    # held-out check: a cell over U, overlapping sets, another lock, a fit file not given -> thieu
    for gn, txt in ((_o(sha, rows=_rows_o(d=0.006)), 'U_psi'),
                    (_o(sha, o=[dict(_o(sha)['o'][0], n_id=5)]), 'n_id'),
                    (_o(sha, tap=dict(khop=['c1'], kiem=['c1'], dich=['r1'])), 'disjoint'),
                    (_o(sha, khoa_truoc_sha256='z' * 64), 'khoa'),
                    (_o(sha, a6_khop=dict(cho_sha256=['d' * 64])), 'cho_sha256'),
                    (_o(sha, trang_thai='truot'), 'trang_thai')):
        o = _gop(places, giu_ngoai=gn)
        assert o['trang_thai']['a6'] == 'thieu' and any(txt in w for w in o['ly_do']), (txt, o['ly_do'])
    edited = _o(sha)
    edited['o'][0]['d_max_rad'] = 0.001
    with pytest.raises(A.A6Error, match='sha256'):
        A._check_sha(edited, 'o')


def test_gop_dong_band_thieu_reaches_the_reader(places):
    g_hat = np.mean([p['g_E'] for p in places], axis=0)
    d = _dong(g_hat)
    d['luot']['r1']['dai']['a0_hi']['n_khung'] = 50
    o = _gop(places, dong=A.with_sha(d))
    assert o['trang_thai']['a6'] == 'dat' and o['dong']['thieu'][0][:2] == ['r1', 'a0_hi']
    assert A.a6du_rad(o, 0.5) is None                                    # band of the frame unknown: no bound
    assert A.a6du_rad(o, 0.5, 'r1', ['a0', 'a0_hi']) is None            # that band thieu
    ok = A.a6du_rad(o, 0.5, 'r1', ['a0', 'a0_lo'])
    u = o['dong']['theo_luot']['r1']
    assert ok == pytest.approx((o['eps_g_khong_dong'] + max(u['a0'], u['a0_lo'])) * 0.5 / (2 * (H - HB))
                               + (o['e_bc_rad'] or 0))


def test_gop_consistency_per_place_and_new_plate(places):
    # A6C1-6: two places on either side of g_cal, their mean on it: each place checked
    pg, pk = places
    a, b = dict(pg), dict(pk)
    a['g_E'], b['g_E'] = list(GC + [0.008, 0.0]), list(GC - [0.008, 0.0])
    o = _gop((a, b))
    assert o['nhat_quan'] is False and o['trang_thai']['a6'] == 'thieu'
    assert any('g_cal_bat_nhat' in w for w in o['ly_do'])
    assert o['nhat_quan_cho']['goc']['dat'] is False and o['nhat_quan_cho']['K1']['dat'] is False
    # A6C1-7: another plate than the caliper plate is a new plate; a place of another plate is refused
    o2 = _gop(places, tam_cal_id='T0')
    assert o2['trang_thai']['a6'] == 'thieu' and o2['g_cal'] is None and any('tam_moi' in w for w in o2['ly_do'])
    with pytest.raises(A.A6Error, match='plate'):
        A.gop([dict(pg, tam_id='T2'), pk], 'T1', 'T1')


def test_readers_contract_f3(places):
    a6 = _gop(places, with_ebc=True)
    assert a6['trang_thai']['a6'] == 'dat'
    T = np.array([C + [0.7, 0.2], C + [1.55, 0.0], [np.nan, 0.0]])
    psi = np.array([1.0, 0.3, 0.2])
    d, ok, why = A.delta_psi_a6(a6, T, psi, C, H)
    assert d[0] == pytest.approx(A.dpsi_model(a6['g_b'], T[0], 1.0, C, H, HB)) and ok.tolist() == [True, False, False]
    assert why[1] == 'ngoai_mien' and why[2] == 'khong_huu_han' and np.isnan(d[1])      # A6C1-8: corner at 1.652 m
    with pytest.raises(ValueError):
        A.delta_psi_a6(a6, T, psi, C + 0.01, H)                         # another plumb point
    with pytest.raises(ValueError):
        A.delta_psi_a6(a6, T, psi, C, float('nan'))                     # Z_c NaN does not pass (A6C1-8)
    with pytest.raises(ValueError):
        A.delta_psi_a6(a6, T, psi, C, H + 0.01)                         # another Z_c (A6C1-15)
    assert A.ap_dung(a6, 'T1') == ('dat', None) and A.ap_dung(a6, 'T2')[0] == 'chua_ket_luan'
    assert isinstance(A.ap_dung(a6, 'T2')[1], str)                     # (trang_thai str, ly_do str | None)
    rho = 0.73
    assert A.a6du_rad(a6, rho) == pytest.approx(a6['eps_g'] * rho / (2 * (H - HB)) + a6['e_bc_rad'])
    for bad in (-1.0, 1.55, float('nan'), None):
        assert A.a6du_rad(a6, bad) is None
    a6b = dict(a6, trang_thai=dict(a6='thieu', h='thieu'))
    d, ok, _ = A.delta_psi_a6(a6b, T, psi, C, H)
    assert not ok.any() and A.ap_dung(a6b, 'T1')[0] == 'chua_ket_luan' and A.a6du_rad(a6b, rho) is None


def _kc(**kw):
    d = dict(version=A.KC_VERSION, thu=False,
             A3=dict(max_deg=0.4, quy_ve_ma_noc=True, ma_ao=dict(t=[0.0]), du_lieu_du=True),
             hang_so_s7=dict(xuat_phat=dict(goc=list(C + [1.0, 0.0]), K1=list(C + [0.0, 1.2])), MIEN_XP_M=0.10,
                             plumb=list(C)),
             A3_tuong_doi=dict(A3_xp_deg=dict(goc=0.15, K1=0.2), du_lieu_du=True, mot_phuong=dict(goc=False, K1=False),
                               ngoai_suy=dict(goc=False, K1=False), cap_hong=dict(goc=False, K1=False)),
             P1_tuong_doi=dict(theo_vung=dict(goc=dict(ngoai_suy=False, mot_phuong=False, cap_hong=False),
                                              K1=dict(ngoai_suy=False, mot_phuong=False, cap_hong=False))))
    d.update(kw)
    return d


def test_u_psi_two_ends_by_position(places):
    # A6C1-10 (CODE 20/11): A3 by position from kiem_cheo.json; a flag on at an end, or an end outside every checked
    # start disc -> thieu; A3 and A6du added as two separate bounds (A6C1-3)
    a6 = _gop(places, with_ebc=True)
    p1, p2 = C + np.array([1.03, 0.02]), C + np.array([0.0, 1.15])
    u = A.u_psi_hai_dau_rad(a6, _kc(), p1, p2, 'T1')
    want = math.radians(0.15) + math.radians(0.2) + A.a6du_rad(a6, np.linalg.norm(p1 - C)) \
        + A.a6du_rad(a6, np.linalg.norm(p2 - C))
    assert u['trang_thai'] == 'dat' and u['u_rad'] == pytest.approx(want) and u['ghep'] == 'cong_rieng'
    assert 'two places' in u['ly_do_ghep']
    k1 = _kc()
    k1['A3_tuong_doi']['ngoai_suy']['K1'] = True
    k2 = _kc()
    k2['P1_tuong_doi']['theo_vung']['goc']['mot_phuong'] = True
    k3 = _kc()
    del k3['A3_tuong_doi']['cap_hong']['goc']                            # a missing flag counts as on
    for kc, txt in ((k1, 'flag'), (k2, 'flag'), (k3, 'flag'), (_kc(version='c7_kiem_cheo_v6'), 'v7'),
                    (_kc(A3=dict(max_deg=0.4, quy_ve_ma_noc=False, ma_ao={}, du_lieu_du=True)), 'carried'),
                    (_kc(A3=dict(max_deg=0.4, quy_ve_ma_noc=True, ma_ao={}, du_lieu_du=False)), 'partial'),
                    (dict(_kc(), A3_tuong_doi=dict(_kc()['A3_tuong_doi'], du_lieu_du=False)), 'partial'),
                    (dict(_kc(), hang_so_s7=dict(_kc()['hang_so_s7'], plumb=[0.0, 0.0])), 'plumb'), ({}, 'v7')):
        r = A.u_psi_hai_dau_rad(a6, kc, p1, p2, 'T1')
        assert r['trang_thai'] == 'thieu' and any(txt in w for w in r['ly_do']), (txt, r['ly_do'])
    r = A.u_psi_hai_dau_rad(a6, _kc(), C + [0.5, 0.5], p2, 'T1')
    assert r['trang_thai'] == 'thieu' and any('outside every checked start disc' in w for w in r['ly_do'])
    assert A.u_psi_hai_dau_rad(a6, _kc(), p1, p2, 'T2')['trang_thai'] == 'thieu'


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────

def _write_frames(path, idx, t, P):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 't_s', 'valid'] + list(A.CORNER_COLS))
        for k, tt, p in zip(idx, t, P):
            w.writerow([int(k), f'{tt:.6f}', 1] + [f'{v:.9f}' for v in p.reshape(-1)])


def test_cli_do_gop_and_f15_file(tmp_path, monkeypatch):
    import c8_u_imu as cu
    idx, t, P = _clip()
    t = np.round(t, 6)
    fp = tmp_path / 'frames.csv'
    _write_frames(fp, idx, t, P)
    sp = tmp_path / 'session.json'
    sp.write_text(json.dumps(dict(kappa=HB / H, tag_h_m=HB, car_side_m=SIDE, plumb_xy_m=list(C),
                                  camera=dict(model='x', f=1, cx=0, cy=0, k=[], rvec=[0, 0, 0], tvec=[0, 0, 0]))),
                  encoding='utf-8')
    f = _f15(idx, t)
    f15 = A.with_sha(dict(
        schema=A.F15_FULL, trang_thai='dat', cho='goc', chieu='held_minus_full', bo_H32=True, truoc_le=True,
        thu_tu_goc='ArUco 0..3 (c7_measure.car_corners_body)', clip_sha256='c' * 64,
        nguon=[dict(vai_tro='frames', sha256=cu.sha256(fp)), dict(vai_tro='session', sha256=cu.sha256(sp)),
               dict(vai_tro='track', sha256='d' * 64), dict(vai_tro='cua_so_v1', sha256='e' * 64)],
        kappa=HB / H, Zc_m=H, h_m=HB, plumb_xy_m=list(C), mien_E=f['mien_E'],
        ban_do=dict(H32_ids=[18, 19, 30], ca=[dict(ten=c, bo_ids=[int(x) for x in c[3:].split('-')],
                                                 A3=dict(quy_max_deg=0.1)) for c in CASES]),
        khung=[dict(frame_id=int(k), t_s=float(tt), goc_m=f['du'][int(k)][CASES[0]].tolist(),
                    du_m={c: v.tolist() for c, v in f['du'][int(k)].items()}) for k, tt in zip(idx, t)]))
    f15p = tmp_path / 'f15.json'
    f15p.write_text(json.dumps(f15), encoding='utf-8')
    monkeypatch.setattr(A, 'camera_of', lambda ses: TopCam())
    out = tmp_path / 'goc.json'
    args = ['do', '--frames', str(fp), '--session', str(sp), '--cho', 'goc', '--tam-id', 'T1', '--lan-lap-id', 'L1']
    assert A.main(args + ['--f15', str(f15p), '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['schema'] == A.SCHEMA_CHO and d['trang_thai'] == 'dat' and len(d['nguon']) == 4
    assert d['A3_theo_ca_deg'] == {c: 0.1 for c in CASES}
    for bad, n in ((dict(f15, plumb_xy_m=[0.0, 0.0]), 'plumb'),                      # session set-up
                   (dict(f15, nguon=[dict(vai_tro='frames', sha256='e' * 64), f15['nguon'][1]]), 'frames'),
                   (dict(f15, thu_tu_goc='other'), 'order'), (dict(f15, mien_E=None), 'window'),
                   (dict(f15, nguon=f15['nguon'][:3]), 'cua_so'),                       # A6C1-2: E window source
                   (dict(f15, ban_do=dict(f15['ban_do'], H32_ids=[18, 19, 30, 31])), 'h32'),   # id in no case
                   (dict(f15, plumb_xy_m=list(C + 0.002)), 'l125'),
                   (dict(f15, khung=[dict(f15['khung'][0], t_s=float('nan'))] + f15['khung'][1:]), 'tnan')):
        bp = tmp_path / f'bad_{n}.json'
        bp.write_text(json.dumps(A.with_sha(dict(bad))), encoding='utf-8')
        assert A.main(args + ['--f15', str(bp), '--out', str(tmp_path / f'x_{n}.json')]) == 2, n
    ed = dict(f15, h_m=HB)
    ed['kappa'] = HB / H + 0.0                                           # content changed without a new sha: refused
    ed['trang_thai'] = 'dat'
    ed['ly_do'] = ['edited']
    (tmp_path / 'ed.json').write_text(json.dumps(ed), encoding='utf-8')
    assert A.main(args + ['--f15', str(tmp_path / 'ed.json'), '--out', str(tmp_path / 'x_ed.json')]) == 2
    a6p = tmp_path / 'a6.json'
    assert A.main(['gop', '--cho-file', str(out), '--tam-id', 'T1', '--tam-cal-id', 'T1', '--out', str(a6p)]) == 0
    a6 = A.load_a6(str(a6p))
    assert a6['trang_thai']['a6'] == 'thieu' and a6['schema'] == 'c8b_a6_v2'
    assert a6['h_m'] == HB and a6['plumb_xy_m'] == pytest.approx(list(C)) and a6['H_m'] == pytest.approx(H)
    assert a6['commit'] is not None and a6['so_tay'] == A.SO_TAY


def test_u_psi_joint_replay_per_case(places):
    # A6C1-3 (CODE 20/11, changed decision): both ends at the same place, A3 per H32 case of that place (KIEMCHEO
    # 692462f): u = max_case [2 A3_c + |dg_c| (rho1 + rho2) / (2 (H - h))] + A6du without U_meo_E at both ends
    pg, pk = places
    a3 = {'bo:18-19-30': 0.12, 'bo:18-19': 0.05, 'bo:18-30': 0.02}
    pg2, pk2 = dict(pg, A3_theo_ca_deg=a3), dict(pk, A3_theo_ca_deg=a3)
    a6 = _gop((pg2, pk2))
    assert a6['trang_thai']['a6'] == 'dat' and a6['eps_g_khong_meo'] < a6['eps_g']
    s2 = max(2 * np.linalg.norm(p['SE_E']) for p in places)
    u = a6['U_chuyen']
    assert a6['eps_g_khong_meo'] == pytest.approx(max(s2, a6['khoang_g_cal'] + A.u_cal()[1]) + u['lap'] + u['cho']
                                                  + u['dong'])
    p1, p2 = C + np.array([1.03, 0.02]), C + np.array([0.97, -0.03])         # both in the goc disc
    r = A.u_psi_hai_dau_rad(a6, _kc(), p1, p2, 'T1')
    r1, r2 = np.linalg.norm(p1 - C), np.linalg.norm(p2 - C)
    den = 2 * (H - HB)
    per = {c: 2 * math.radians(a3[c]) + np.linalg.norm(pg['U_meo_E_theo_ca'][c]) * (r1 + r2) / den for c in a3}
    want = max(per.values()) + A.a6du_rad(a6, r1, meo=False) + A.a6du_rad(a6, r2, meo=False)
    assert r['trang_thai'] == 'dat' and r['ghep'] == 'replay_chung_theo_ca' and r['u_rad'] == pytest.approx(want)
    assert r['phan']['ca_lon_nhat'] == max(per, key=per.get)
    # a case without A3 at that place: the separate sum (fallback), named
    a6b = _gop((dict(pg, A3_theo_ca_deg=dict(a3, **{'bo:18-30': None})), pk2))
    r = A.u_psi_hai_dau_rad(a6b, _kc(), p1, p2, 'T1')
    assert r['ghep'] == 'cong_rieng' and 'without A3' in r['ly_do_ghep']
    assert r['u_rad'] == pytest.approx(2 * math.radians(0.15) + A.a6du_rad(a6b, r1) + A.a6du_rad(a6b, r2))


# ── Codex round 2/2 (A6C2-2..10), fixed 20/11 (Liam at CODE_C8B: "Sua ngay, Codex sau") ──────────────────────

def test_a6c2_3_lap_pair_must_join_the_caliper_mounting(places):
    # the E clips on L2, the caliper numbers on L1: only an L1-L2 pair widens; an unrelated pair or no cal id: not
    pg, pk = places
    a, b = dict(pg, lan_lap_id='L2'), dict(pk, lan_lap_id='L2')
    far = np.linalg.norm(np.array(a['g_E']) - GC)
    rel = _lap(cap=[dict(_lap()['cap'][0], lan_lap_a='L1', lan_lap_b='L2')])
    unrel = _lap(cap=[dict(_lap()['cap'][0], lan_lap_a='L7', lan_lap_b='L8')])
    o1 = _gop((a, b), lap=rel)
    o2 = _gop((a, b), lap=unrel)
    o3 = _gop((a, b), lap=rel, lan_lap_cal_id=None)
    assert o1['nhat_quan_cho']['goc']['u_lap_cap'] == pytest.approx(o1['U_chuyen']['lap'])
    assert o2['nhat_quan_cho']['goc']['u_lap_cap'] is None and o3['nhat_quan_cho']['goc']['u_lap_cap'] is None
    assert o2['nhat_quan_cho']['goc']['nguong'] == pytest.approx(pg['s_E_norm'] + A.u_cal()[1])
    assert o1['nhat_quan_cho']['goc']['nguong'] > o2['nhat_quan_cho']['goc']['nguong'] and far > 0


def test_a6c2_4_half_bands_and_the_reader(places):
    g_hat = np.mean([p['g_E'] for p in places], axis=0)
    d = _dong(g_hat)
    d['dai_dinh_nghia'] = d['dai_dinh_nghia'][:2]                         # a0_hi not defined
    o = _gop(places, dong=A.with_sha(d))
    assert o['trang_thai']['a6'] == 'thieu' and any('two half bands' in w for w in o['ly_do'])
    d = _dong(g_hat)
    d['luot']['r1']['dai']['a0_lo']['trang_thai'] = 'thieu'               # a cell thieu with its numbers
    o = _gop(places, dong=A.with_sha(d))
    assert ['r1', 'a0_lo'] == o['dong']['thieu'][0][:2]
    o = _gop(places)
    r = 0.5
    assert A.a6du_rad(o, r, 'r1', ['a0', 'a0_hi']) is not None
    for bad in ('a0', ['a0'], ['a0_lo', 'a0_hi'], ['a0', 'a0', ], ['a0', 'a0_lo', 'a0_hi']):
        assert A.a6du_rad(o, r, 'r1', bad) is None, bad


def test_a6c2_5_u_h_three_parts(places):
    for uh, txt in ((_uh(u_h_chung_m=0.0), 'sum'), (_uh(phan=dict(do_trai_ba_ga_m=0.0, lech_video_so_tay_m=0.0,
                                                                   hang_m=0.0), u_h_chung_m=0.0), 'hang_m'),
                    (_uh(phan=None), 'phan'), (_uh(n_ga_giu_ngoai=2), 'n_ga'), (_uh(h_m=HB + 0.001), 'h of the run')):
        o = _gop(places, u_h=uh)
        assert o['trang_thai'] == dict(a6='dat', h='thieu') and any(txt in w for w in o['ly_do_h']), (txt, o['ly_do_h'])


def test_a6c2_6_readers_refuse_bad_numbers_and_setup(places, tmp_path):
    a6 = _gop(places)
    T, psi = np.array([C + [0.7, 0.2]]), np.array([1.0])
    with pytest.raises(ValueError):
        A.delta_psi_a6(a6, T, psi, [math.nan, C[1]], H)
    bad = dict(a6, eps_g=-0.01)
    assert A.a6du_rad(bad, 1.0) is None and A.ap_dung(bad, 'T1')[0] == 'chua_ket_luan'
    p = tmp_path / 'a6.json'
    p.write_text(json.dumps(A._clean(bad), allow_nan=False), encoding='utf-8')
    with pytest.raises(A.A6Error, match='eps_g'):
        A.load_a6(str(p))
    with pytest.raises(A.A6Error, match='L125'):
        A.gop([dict(pl, set_up=dict(pl['set_up'], H_m=H + 0.01)) for pl in places], 'T1', 'T1')
    ses = tmp_path / 's.json'
    ses.write_text(json.dumps(dict(kappa=HB / (H + 0.01), tag_h_m=HB, car_side_m=SIDE, plumb_xy_m=list(C))),
                   encoding='utf-8')
    with pytest.raises(A.A6Error, match='L125'):
        A.read_session(str(ses))


def test_a6c2_7_nan_disc_does_not_widen_a3(places):
    # each guard named: a NaN radius, a NaN centre, a NaN plumb of kiem_cheo.json -> thieu with that reason
    a6 = _gop(places)
    p1, p2 = C + np.array([1.03, 0.02]), C + np.array([0.0, 1.15])           # both inside their discs
    for kc, txt in ((dict(_kc(), hang_so_s7=dict(_kc()['hang_so_s7'], MIEN_XP_M=float('nan'))), 'MIEN_XP_M'),
                    (dict(_kc(), hang_so_s7=dict(_kc()['hang_so_s7'], xuat_phat=dict(goc=[float('nan'), 0.0],
                                                                                    K1=list(C + [0.0, 1.2])))),
                     'xuat_phat[goc]'),
                    (dict(_kc(), hang_so_s7=dict(_kc()['hang_so_s7'], plumb=[float('nan'), C[1]])), 'plumb')):
        r = A.u_psi_hai_dau_rad(a6, kc, p1, p2, 'T1')
        assert r['trang_thai'] == 'thieu' and any(txt in w for w in r['ly_do']), (txt, r)
    assert A.u_psi_hai_dau_rad(a6, _kc(), p1, p2, 'T1')['trang_thai'] == 'dat'


def test_a6c2_8_degenerate_fit_reports_thieu():
    # the same four corners in every frame: no rate, alpha columns zero -> singular, thieu (not a KeyError)
    P = np.repeat(A.lifted_corners(C + [0.6, 0.4], 0.3, GC, 0.0, C, H, HB, SIDE)[None], 300, axis=0)
    t = np.arange(300) / 32.0
    fl = list(A.shutter_fields(P, t, A.uv_of(P, TopCam(), C, HB / H)))
    f = A.fit_g(P, C, H, HB, SIDE, fields=fl)
    assert 'se_alpha' in f and not A.fit_ok(f)
    assert A.corr_alpha_g(f['M']) is None or math.isfinite(A.corr_alpha_g(f['M']))
    assert A.corr_alpha_g(np.zeros((6, 6))) is None


def test_a6c2_10_se_alpha_degrees_of_freedom():
    idx, t, P = _shuttered(-0.72e-6, 2.0e-6, noise=0.0001)
    fl = list(A.shutter_fields(P, t, A.uv_of(P, TopCam(), C, HB / H)))
    f = A.fit_g(P, C, H, HB, SIDE, fields=fl)
    n = len(P)
    assert f['bac_tu_do'] == 5 * n - 6
    res2 = f['res_rms_m'] ** 2 * 4 * n                                   # the residual sum
    assert f['sigma_toa_do_m'] == pytest.approx(math.sqrt(res2 / (5 * n - 6)))
    dg = np.diag(np.linalg.inv(f['M']))[4:]
    assert f['se_alpha'] == pytest.approx([math.sqrt(x) * f['sigma_toa_do_m'] for x in dg])


def test_a6c2_9_grid_refines_every_dimension(monkeypatch):
    seen = []
    real = A.ebc_errors

    def spy(T, S, g, v, *a):
        seen.append((len(np.unique(np.round(g, 12), axis=0)), len(np.unique(np.round(v, 12)))))
        return real(T, S, g, v, *a)
    monkeypatch.setattr(A, 'ebc_errors', spy)
    lv = {}
    for k in (4, 8):
        seen.clear()
        A._ebc_level(GC, 0.003, C, H, HB, SIDE, [C[0], C[1], H], A.RHO_MAX_M, A.V_MAX_M, k, chunk=10 ** 9)
        lv[k] = seen[0]
    assert lv[8][0] > lv[4][0] and lv[8][1] > lv[4][1]
    with pytest.raises(A.A6Error):
        A._ebc_level(GC, 0.003, C, H, HB, SIDE, [C[0], C[1], H], A.RHO_MAX_M, A.V_MAX_M, 2)


def test_a6c1_1_diagnostic_place_gives_no_u_cho(places):
    pg, pk = places
    diag = dict(pk, U_meo_E=None, s_E=None, s_E_norm=None, trang_thai='thieu', ly_do=['diagnostic'])
    o = _gop((pg, diag))
    assert o['U_chuyen']['cho'] is None and o['eps_g'] is None and o['trang_thai']['a6'] == 'thieu'


def _kc_duong(**kw):
    """kiem_cheo.json with the 12b corridors (KIEMCHEO d21f149): goc segment through its start point."""
    d = _kc()
    du = dict(A3_deg=dict(goc=0.3, K1=0.25), mot_phuong=dict(goc=False, K1=False), ngoai_suy=dict(goc=False, K1=False),
              cap_hong=dict(goc=False, K1=False), ban_kinh_m=0.10,
              doan_m=dict(goc=[list(C + [1.0 - 0.35, -0.35]), list(C + [1.0 + 0.35, 0.35])],
                          K1=[list(C + [0.0, 0.95]), list(C + [0.0, 1.45])]))
    du.update(kw)
    d['A3_tuong_doi'] = dict(d['A3_tuong_doi'], duong_12b=du)
    return d


def test_hcv3_2_a3_over_the_12b_corridor(places):
    a6 = _gop(places)
    pc = C + np.array([1.0 + 0.30, 0.31])                 # on the goc corridor, outside the goc start disc (0.10 m)
    p2 = C + np.array([0.0, 1.15])                         # K1 disc and K1 corridor
    r = A.u_psi_hai_dau_rad(a6, _kc_duong(), pc, p2, 'T1')
    assert r['trang_thai'] == 'dat', r
    want = math.radians(0.3) + math.radians(0.25) + A.a6du_rad(a6, np.linalg.norm(pc - C))         + A.a6du_rad(a6, np.linalg.norm(p2 - C))
    assert r['u_rad'] == pytest.approx(want)               # K1: corridor 0.25 > disc 0.2: the larger
    assert A.u_psi_hai_dau_rad(a6, _kc(), pc, p2, 'T1')['trang_thai'] == 'thieu'      # no corridor block
    # no P1 flag asked for the corridor; an A3 flag of the corridor on -> that region not usable
    k = _kc_duong()
    k['P1_tuong_doi']['theo_vung']['goc']['mot_phuong'] = True
    assert A.u_psi_hai_dau_rad(a6, k, pc, p2, 'T1')['trang_thai'] == 'dat'
    for kw, txt in ((dict(ngoai_suy=dict(goc=True, K1=False)), 'corridor goc'),
                    (dict(cap_hong=dict(K1=False)), 'corridor goc'),                    # missing flag = on
                    (dict(A3_deg=dict(goc=float('nan'), K1=0.25)), 'A3_deg[goc]'),
                    (dict(ban_kinh_m=float('nan')), 'ban_kinh_m'),
                    (dict(doan_m=dict(goc=[[float('nan'), 0.0], [1.0, 1.0]], K1=[[0, 0], [0, 1]])), 'doan_m[goc]')):
        r = A.u_psi_hai_dau_rad(a6, _kc_duong(**kw), pc, p2, 'T1')
        assert r['trang_thai'] == 'thieu' and any(txt in w for w in r['ly_do']), (txt, r['ly_do'])
    assert A._dist_seg(np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert A._dist_seg(np.array([3.0, 0.0]), np.array([-1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(2.0)
