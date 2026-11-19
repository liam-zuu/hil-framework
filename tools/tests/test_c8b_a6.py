"""c8b_a6 (M4: A6 from the E clips, design C8B v7 sections 2, 3, 10; J v6). Synthetic E clips through the exact lift
of car_pose (lifted_corners) and a top-down camera."""
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


class TopCam:
    """Pinhole looking straight down from (C, H): rows along +y, f 1700 px."""

    def project(self, P):
        P = np.asarray(P, float)
        return np.column_stack([1920 + 1700 * (P[:, 0] - C[0]) / (H - P[:, 2]),
                                1080 + 1700 * (P[:, 1] - C[1]) / (H - P[:, 2])])


@pytest.fixture
def small_alpha(monkeypatch):
    """The 'dat' path: at the largest alpha seen at 3b (4.53 us/px) the shutter rule of design 2b (sensitivity of g
    below 10 % of SE_E) fails already at 0.05 rad/s (ratio 0.24; 0.3 rad/s: 1.2), so these tests take a small alpha."""
    monkeypatch.setattr(A, 'ALPHA_MAX_S_PX', 1e-7)


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


def _f15(idx, scale=0.0001, seed=1, schema=A.F15_FULL, trang_thai='dat', cho='goc'):
    rng = np.random.default_rng(seed)
    cases = ['bo:18-19-30', 'bo:18-19', 'bo:18-30']
    fields = {c: rng.normal(0, scale, (4, 2)) for c in cases}           # a smooth field: the same at every frame
    return dict(schema=schema, trang_thai=trang_thai, cho=cho, mien_E=dict(sau_E_dau=int(idx[0]) - 1,
                                                                           truoc_E_cuoi=int(idx[-1]) + 1),
                du={int(k): {c: fields[c] for c in cases} for k in idx}, sha256='1' * 64, file='f15.json', ly_do=[])


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
    assert abs(ex) > 1e-3                                         # a real effect at this tilt (about 0.1 to 0.2 deg)


def test_fit_g_recovers_the_tilt_and_v_does_not_bias_it():
    _, _, P = _clip(noise=0.0, v=0.0)
    f = A.fit_g(P, C, H, HB, SIDE)
    assert f['g'] == pytest.approx(GC, abs=1e-6) and f['sigma'] == pytest.approx(0.002, abs=1e-6)
    _, _, P2 = _clip(noise=0.0, v=-0.000475)
    f2 = A.fit_g(P2, C, H, HB, SIDE)
    assert f2['g'] == pytest.approx(GC, abs=1e-6) and f2['v_phu'] == pytest.approx(-0.000475, abs=1e-7)


def test_do_place_full_f15_identified_and_shutter_selection(small_alpha):
    idx, t, P = _clip()
    r = A.do_place('goc', idx, t, P, _ses(), _f15(idx), cam=TopCam())
    assert r['trang_thai'] == 'dat', r['ly_do']
    assert np.linalg.norm(np.array(r['g_E']) - GC) < 4 * np.linalg.norm(r['SE_E'])
    assert all(u > 0 for u in r['U_meo_E']) and r['s_E'] == pytest.approx(list(2 * np.array(r['SE_E'])
                                                                              + np.array(r['U_meo_E'])))
    assert r['man_trap']['do_nhay_g'] <= A.SHUTTER_SHARE * min(r['SE_E']) and r['n_khung'] >= A.N_MIN


def test_do_place_diagnostic_f15_is_not_a_bound(small_alpha):
    # KIEMCHEO 02dea17 (CODE 19/11): the diagnostic F15 never gives a usable U_meo_E: A6 not lockable, the reason kept
    idx, t, P = _clip()
    r = A.do_place('goc', idx, t, P, _ses(), _f15(idx, schema=A.F15_DIAG, trang_thai='chan_doan'), cam=TopCam())
    assert r['trang_thai'] == 'thieu' and any('chan_doan' in w for w in r['ly_do'])


def test_do_place_map_noise_larger_than_g_is_not_identified(small_alpha):
    idx, t, P = _clip()
    rng = np.random.default_rng(5)
    f = _f15(idx)
    for k in f['du']:                                               # a residual with the tilt pattern, 3 mm scale
        f['du'][k] = {c: v + 0.02 * np.outer(A.body(SIDE) @ np.array([1.0, 0.0]), [1.0, 0.0]) for c, v in
                      f['du'][k].items()}
    r = A.do_place('goc', idx, t, P, _ses(), f, cam=TopCam())
    assert r['trang_thai'] == 'thieu' and any('not identified' in w for w in r['ly_do'])
    del rng


def test_do_place_needs_the_frames_of_f15_and_a_camera(small_alpha):
    idx, t, P = _clip()
    f = _f15(idx[:-50])
    f['mien_E'] = dict(sau_E_dau=int(idx[0]) - 1, truoc_E_cuoi=int(idx[-1]) + 1)
    with pytest.raises(A.A6Error, match='lacks'):
        A.do_place('goc', idx, t, P, _ses(), f, cam=TopCam())
    with pytest.raises(A.A6Error, match='camera'):
        A.do_place('goc', idx, t, P, _ses(), _f15(idx), cam=None)


def test_shutter_rule_at_the_alpha_seen():
    # design 2b at alpha 4.53 us/px: an E clip turning 0.3 rad/s fails (g sensitivity 1.2 x SE_E > 10 %)
    idx, t, P = _clip(rate=0.3)
    r = A.do_place('goc', idx, t, P, _ses(), _f15(idx), cam=TopCam())
    assert r['trang_thai'] == 'thieu' and r['g_E'] is None and any('shutter' in w for w in r['ly_do'])


def _place(cho, g=GC, seed=0, centre=(0.6, 0.4)):
    idx, t, P = _clip(g=g, seed=seed, centre=centre)
    return A.do_place(cho, idx, t, P, _ses(), _f15(idx, cho=cho), cam=TopCam())


def test_gop_dat_and_eps_g(small_alpha):
    pg, pk = _place('goc'), _place('K1', seed=3, centre=(-0.5, 0.7))
    out = A.gop([pg, pk], 'T1', u_lap=0.0005, u_dong=0.0007, giu_ngoai=dict(trang_thai='dat'), u_h_chung=0.0003)
    assert out['trang_thai'] == dict(a6='dat', h='dat'), out['ly_do']
    s = max(pg['s_E_norm'], pk['s_E_norm'])
    uch = 0.0005 + out['U_chuyen']['cho'] + 0.0007
    assert out['eps_g'] == pytest.approx(max(s, out['khoang_g_cal'] + A.u_cal()[1]) + uch)
    assert out['U_chuyen']['cho'] == pytest.approx(np.linalg.norm(np.array(pg['g_E']) - np.array(pk['g_E'])))
    assert out['e_bc_rad'] is not None and out['e_bc_rad'] < 1e-3
    assert out['h_goc_m'] == pytest.approx([HB + ri @ np.array(out['g_b']) for ri in A.body(SIDE)])
    assert out['u_h_goc_m'][0] == pytest.approx(0.0003 + np.linalg.norm(A.body(SIDE)[0]) * out['eps_g'] + A.V_MAX_M)
    # every missing part leaves A6 thieu with its reason
    for kw, txt in ((dict(u_dong=None), 'dong'), (dict(u_lap=None), 'lap'), (dict(giu_ngoai=None), 'held-out'),
                    (dict(giu_ngoai=dict(trang_thai='truot')), 'held-out')):
        base = dict(u_lap=0.0005, u_dong=0.0007, giu_ngoai=dict(trang_thai='dat'), u_h_chung=0.0003)
        base.update(kw)
        o = A.gop([pg, pk], 'T1', **base, with_ebc=False)
        assert o['trang_thai']['a6'] == 'thieu' and any(txt in w for w in o['ly_do']), (kw, o['ly_do'])
    o1 = A.gop([pg], 'T1', 0.0005, 0.0007, dict(trang_thai='dat'), 0.0003, with_ebc=False)
    assert o1['trang_thai']['a6'] == 'thieu' and any('one place' in w for w in o1['ly_do'])


def test_gop_g_cal_inconsistent(small_alpha):
    far = GC + np.array([0.01, -0.01])
    pg, pk = _place('goc', g=far), _place('K1', g=far, seed=3, centre=(-0.5, 0.7))
    o = A.gop([pg, pk], 'T1', 0.0005, 0.0007, dict(trang_thai='dat'), 0.0003, with_ebc=False)
    assert o['nhat_quan'] is False and o['trang_thai']['a6'] == 'thieu' and any('g_cal_bat_nhat' in w for w in o['ly_do'])


def test_readers_and_u_psi_two_ends(small_alpha):
    pg, pk = _place('goc'), _place('K1', seed=3, centre=(-0.5, 0.7))
    a6 = A.gop([pg, pk], 'T1', 0.0005, 0.0007, dict(trang_thai='dat'), 0.0003)
    T, psi = C + np.array([0.7, 0.2]), 1.0
    assert A.delta_psi_a6(a6, T, psi, C, H) == pytest.approx(A.dpsi_model(a6['g_b'], T, psi, C, H, HB))
    with pytest.raises(A.A6Error):
        A.delta_psi_a6(a6, T, psi, C + 0.01, H)                     # another plumb point: refused
    assert A.ap_dung(a6, 'T1') and not A.ap_dung(a6, 'T2')
    rho = 0.73
    assert A.a6du_rad(a6, rho) == pytest.approx(a6['eps_g'] * rho / (2 * (H - HB)) + a6['e_bc_rad'])
    kc = dict(A3=dict(max_deg=0.15))
    p1, p2 = C + np.array([1.0, 0.0]), C + np.array([0.0, 1.2])
    u = A.u_psi_hai_dau_rad(a6, kc, p1, p2, 'T1')
    want = 2 * math.radians(0.15) + A.a6du_rad(a6, 1.0) + A.a6du_rad(a6, 1.2)
    assert u['trang_thai'] == 'dat' and u['u_rad'] == pytest.approx(want)
    assert A.u_psi_hai_dau_rad(a6, kc, p1, p2, 'T2')['trang_thai'] == 'thieu'          # another plate
    assert A.u_psi_hai_dau_rad(a6, {}, p1, p2, 'T1')['trang_thai'] == 'thieu'          # no A3
    a6b = dict(a6, trang_thai=dict(a6='thieu', h='thieu'))
    assert A.u_psi_hai_dau_rad(a6b, kc, p1, p2, 'T1')['trang_thai'] == 'thieu'
    assert A.delta_psi_a6(a6b, T, psi, C, H) is None


def _write_frames(path, idx, t, P):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 't_s', 'valid'] + list(A.CORNER_COLS))
        for k, tt, p in zip(idx, t, P):
            w.writerow([int(k), f'{tt:.6f}', 1] + [f'{v:.9f}' for v in p.reshape(-1)])


def test_cli_do_gop_and_f15_set_up(tmp_path, monkeypatch, small_alpha):
    idx, t, P = _clip()
    fp = tmp_path / 'frames.csv'
    _write_frames(fp, idx, t, P)
    sp = tmp_path / 'session.json'
    sp.write_text(json.dumps(dict(kappa=HB / H, tag_h_m=HB, car_side_m=SIDE, plumb_xy_m=list(C),
                                  camera=dict(model='x', f=1, cx=0, cy=0, k=[], rvec=[0, 0, 0], tvec=[0, 0, 0]))),
                  encoding='utf-8')
    f = _f15(idx)
    f15 = dict(schema=A.F15_FULL, trang_thai='dat', cho='goc', chieu='held_minus_full', bo_H32=True, truoc_le=True,
               kappa=HB / H, Zc_m=H, h_m=HB, plumb_xy_m=list(C), mien_E=f['mien_E'],
               khung=[dict(frame_id=int(k), du_m={c: v.tolist() for c, v in f['du'][int(k)].items()}) for k in idx])
    f15p = tmp_path / 'f15.json'
    f15p.write_text(json.dumps(f15), encoding='utf-8')
    import c7_clip_probe
    monkeypatch.setattr(c7_clip_probe, 'Camera', lambda **kw: TopCam())
    out = tmp_path / 'goc.json'
    assert A.main(['do', '--frames', str(fp), '--session', str(sp), '--cho', 'goc', '--f15', str(f15p),
                   '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['schema'] == A.SCHEMA_CHO and d['trang_thai'] == 'dat' and len(d['nguon']) == 3
    bad = dict(f15, plumb_xy_m=[0.0, 0.0])
    (tmp_path / 'bad.json').write_text(json.dumps(bad), encoding='utf-8')
    assert A.main(['do', '--frames', str(fp), '--session', str(sp), '--cho', 'goc', '--f15', str(tmp_path / 'bad.json'),
                   '--out', str(tmp_path / 'x.json')]) == 2
    a6p = tmp_path / 'a6.json'
    assert A.main(['gop', '--cho-file', str(out), '--tam-id', 'T1', '--out', str(a6p)]) == 0
    a6 = A.load_a6(str(a6p))
    assert a6['trang_thai']['a6'] == 'thieu' and a6['schema'] == 'c8b_a6_v2'
    # the fields KIEMCHEO --a6 checks (KCF1-6): h_m, plumb_xy_m, H_m of the session
    assert a6['h_m'] == HB and a6['plumb_xy_m'] == pytest.approx(list(C)) and a6['H_m'] == pytest.approx(H)
