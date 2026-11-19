"""Tests for the v4 commands of tools/c7_nhac_tam.py (hieu_chuan, lan_lap; design 2026-11-19_Thiet_ke_nhac_tam_lan_lap_v4
section 10 (a)-(n), 6b (o)-(v)). SYNTHETIC data (NOT a measurement): per-frame records built with the exact forward
equation of v4 2 (anchors) and with the car_pose contract on 4 projected tag corners (tag); one end-to-end test on
clips rendered from a known camera."""
import itertools
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c7_measure as cm  # noqa: E402
import c7_nhac_tam as nt  # noqa: E402

ZC, TAG_H, H_N, SIDE = 2.632, 0.12, 0.10, 0.145
C = np.array([-0.4802, -0.0329])                 # plumb point of 16/11 (as --plumb)
F_OFF = np.array([0.0413, 0.0])                  # refitted camera centre minus C (LL5-1: 41.3 mm)
H_CAM = 2.60
G_TRUE = np.array([[0.0, 0.0], [0.20, 0.0], [0.12, 0.09]])
H_TRUE = np.array([H_N, 0.103, 0.115])
NEO = 'A:100,100,18,18;B:200,100,18,18;oc1:150,160,18,18'
NEO_BOXES = [[100, 100, 18, 18], [200, 100, 18, 18], [150, 160, 18, 18]]
CB_TRUE = np.array([0.08, -0.06])                # tag centre in the anchor frame
TH_TRUE = math.radians(0.5)                      # tag yaw minus body (anchor) yaw
CFG = dict(min_khung=10)
PLACES = dict(goc=((0.0, 0.0), 45.0, [0, .5, -.5, .5, -.5, .5, 0]),
              K1=((1.0, 0.0), 90.0, [0, .25, -.25, .25, -.25, .25, 0]),
              K2=((-0.5, -0.5), 0.0, [0, .15, -.35, .15, -.35, .15, 0]))


def rot(a):
    return nt.rot2(a)


# ---------------------------------------------------------------- fake A6 (c8b_a6_v2 signature, hop dong 2.C)
class FakeA6Mod:
    """load_a6 / delta_psi_a6 / ap_dung of the contract; dpsi = [R(psi) g_b] x (t - C) / (2 (H - h)) (C8B formula)."""

    @staticmethod
    def load_a6(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def delta_psi_a6(a6, xy, psi, plumb, zc):
        g = np.array(a6['g_b'], float)
        d = []
        for (x, y), p in zip(xy, psi):
            gs = rot(p) @ g
            v = np.array([x, y]) - np.asarray(plumb)
            d.append((gs[0] * v[1] - gs[1] * v[0]) / (2.0 * (zc - TAG_H)))
        if a6.get('hong'):
            return np.array(d), np.zeros(len(d), bool), ['hong'] * len(d)
        lo, hi = a6.get('hong_psi_deg', (1e9, 1e9))
        ok = np.array([not (lo <= math.degrees(p) <= hi) for p in psi])
        return np.array(d), ok, [''] * len(d)

    @staticmethod
    def ap_dung(a6, tam_id):
        return ('chua_ket_luan', 'tam_khac') if tam_id in a6.get('tam_khac', []) else ('dat', '')


# ---------------------------------------------------------------- synthetic scene
def tag_obs(alpha, o, F, H, cB=CB_TRUE, th=TH_TRUE, dh=0.0, g_b=(0.0, 0.0)):
    """car_pose of a tag at body pose (alpha, o): 4 corners at their true heights (tilt g_b in the tag frame) projected
    by the forward equation (a_t), rigid fit with scale -> (x, y, psi, s)."""
    psi = alpha + th
    Pt = np.asarray(o) + rot(alpha) @ cB
    body = cm.car_corners_body(SIDE)
    corners = Pt + body @ rot(psi).T
    hc = TAG_H + dh + body @ np.asarray(g_b, float)
    a_t = 1.0 - TAG_H / ZC
    lam = H / (H - hc)
    Ph = C + a_t * (lam[:, None] * (corners - C) + (1 - lam)[:, None] * (F - C))
    ang, t, _, _ = cm.rigid_fit(body, Ph)
    _, _, sc, rs = cm.rigid_fit(body, Ph, with_scale=True)
    return dict(x=float(t[0]), y=float(t[1]), psi=float(ang), s=float(sc), res_sim=float(rs))


def proj_indep(P, h, F, H, kappa):
    """Independent of nt.thuan: the ray from the camera centre (F, H) through P at height h meets the floor at p, then
    the car_pose lift P^ = p + kappa (C - p)."""
    p = F + (np.asarray(P, float) - F) * (H / (H - np.asarray(h, float)))[:, None]
    return p + kappa * (C - p)


def recs_for(alpha, o, n=12, seed=0, noise=2e-6, tag=True, F=None, H=H_CAM, id0=0, tag_noise=True, **kw):
    rng = np.random.default_rng(seed)
    F = C + F_OFF if F is None else F
    out = []
    P = G_TRUE @ rot(alpha).T + o
    for k in range(n):
        Ph = proj_indep(P, H_TRUE, F, H, H_N / ZC) + rng.normal(0, noise, (3, 2))
        tg = None
        if tag:
            tg = tag_obs(alpha, o, F, H, **kw)
            if tag_noise:
                tg = dict(tg, x=tg['x'] + rng.normal(0, 2e-6), y=tg['y'] + rng.normal(0, 2e-6),
                          psi=tg['psi'] + rng.normal(0, 1e-5), s=tg['s'] + rng.normal(0, 1e-6))
        out.append(dict(id=id0 + k, t=(id0 + k) / 60.0, cam_ok=True, F=np.array(F), H=H, Ph=Ph, neo_ok=True, tag=tg))
    return out


def body_at(place, s, dpsi_deg=0.0):
    (x, y), hd, _ = PLACES[place]
    a = math.radians(hd + dpsi_deg)
    return a, np.array([x, y]) + s * np.array([math.cos(a), math.sin(a)])


def win(luot, cho, loai, recs):
    return {'schema': 'cua_so_v1', 'luot_id': luot, 'cho': cho, 'loai': loai, 'clip': 'clip.mov',
            'clip_sha256': 'x', 'session_sha256': None, 'khung': [{'id': r['id'], 'pts_s': r['t']} for r in recs],
            'khung_ma_hop_le': [r['id'] for r in recs], 't_canh_s': None}


def build(runs=None, fit_places=('goc', 'K1', 'K2'), n=12, noise=2e-6, plate=None, e_places=('goc', 'K1')):
    """rows + records. runs: [(luot, pre=(place, s, dpsi, kw), post=(...)|None)]; plate kw for every fit window."""
    plate = plate or {}
    rows, recs, nid = [], {}, [0]

    def add(luot, cho, loai, a, o, seed, **kw):
        r = recs_for(a, o, n=n, seed=seed, noise=noise, id0=nid[0], **kw)
        nid[0] += n + 5
        recs[len(rows)] = r
        rows.append(win(luot, cho, loai, r))
    sd = 1
    for p in fit_places:
        for j, s in enumerate(PLACES[p][2]):
            a, o = body_at(p, s, 0.02 * j)
            add(f'12b_{p}', p, 'ref_12b' if j == 0 else 'dung_12b', a, o, sd, **plate)
            sd += 1
        if p in e_places:
            for loai, dd in (('E_dau', 0.0), ('E_cuoi', 0.0)):       # E returns to its heading (turned about r)
                a, o = body_at(p, 0.0, dd)
                add(f'E_{p}', p, loai, a, o, sd, **plate)
                sd += 1
    if runs is None:
        runs = default_runs()
    for luot, pre, post in runs:
        for loai, spec in (('pre', pre), ('post', post)):
            if spec is None:
                continue
            p, s, dd, kw = spec
            a, o = body_at(p, s, dd)
            add(luot, p, loai, a, o, sd, **kw)
            sd += 1
    return rows, recs


def default_runs():
    out = []
    for j, (p, q) in enumerate([('goc', 'K1'), ('K1', 'K2'), ('K2', 'K2'), ('goc', 'goc'), ('K1', 'goc'), ('K2', 'K1')]):
        out.append((f'L{j + 1:02d}', (p, 0.01 * j - 0.02, 0.3 * j - 0.6, {}), (q, 0.02 - 0.01 * j, 0.5 - 0.2 * j, {})))
    return out


@pytest.fixture
def files(tmp_path):
    ses = dict(camera=dict(model='B2', f=900.0, cx=960.0, cy=540.0, k=[0.0, 0.0], rvec=[math.pi, 0.0, 0.0],
                           tvec=[0.0, 0.0, 2.6]),
               marks={}, fit_ids=[1, 2], car_side_m=SIDE, tag_h_m=TAG_H, kappa=TAG_H / ZC, plumb_xy_m=C.tolist(),
               rms_px_H=0.3)
    (tmp_path / 'session.json').write_text(json.dumps(ses), encoding='utf-8')
    (tmp_path / 'a6.json').write_text(json.dumps(dict(schema='c8b_a6_v2', g_b=[0.0, 0.0], eps_g=1e-4)), encoding='utf-8')
    (tmp_path / 'kiem.json').write_text(json.dumps(dict(rong=True, duoc=True, n0='mau.mov', vung_than_px=NEO_BOXES)),
                                        encoding='utf-8')
    (tmp_path / 'mau.mov').write_bytes(b'x')
    return tmp_path


def run(files, rows, recs, tag='a', cfg=None, a6='a6.json', truc_xe=None, xoay=None, u_beta_cam=0.01, tam='T1'):
    d = files
    ses_sha = cm.sha256(str(d / 'session.json'))
    for w in rows:
        w['session_sha256'] = ses_sha
    (d / f'cs_{tag}.json').write_text(json.dumps({'schema': 'cua_so_v1', 'cua_so': rows}), encoding='utf-8')
    fit = {i: recs[i] for i, w in enumerate(rows) if w['loai'] in nt.LOAI_KHOP}
    hold = {i: recs[i] for i, w in enumerate(rows) if w['loai'] in nt.LOAI_GIU}
    c1 = nt.run_hieu_chuan(str(d / f'cs_{tag}.json'), str(d / 'session.json'), NEO, (str(d / 'mau.mov'),
                           str(d / 'session.json'), 0), str(d / f'hc_{tag}'), str(d / a6), tam, kiem_r=[str(d / 'kiem.json')],
                           h_neo=H_N, cfg=dict(CFG, **(cfg or {})), a6_mod=FakeA6Mod, _records=fit)
    hc = json.load(open(d / f'hc_{tag}' / 'hieu_chuan_neo.json', encoding='utf-8'))
    tx = None
    if truc_xe is not None:
        tx = d / f'tx_{tag}.json'
        tx.write_text(json.dumps(truc_xe(hc)), encoding='utf-8')
    xo = None
    if xoay is not None:
        xo = d / f'xo_{tag}.json'
        xo.write_text(json.dumps(xoay), encoding='utf-8')
    nt.run_lan_lap(str(d / f'hc_{tag}' / 'hieu_chuan_neo.json'), str(d / f'cs_{tag}.json'), str(d / 'session.json'),
                   str(d / f'll_{tag}'), str(d / a6), tam, truc_xe=None if tx is None else str(tx),
                   xoay=None if xo is None else str(xo), u_beta_cam_deg=u_beta_cam, cfg=dict(CFG, **(cfg or {})),
                   a6_mod=FakeA6Mod, _records=hold)
    ll = json.load(open(d / f'll_{tag}' / 'lan_lap.json', encoding='utf-8'))
    return c1, hc, ll, {x['luot_id']: x for x in ll['luot']}


def truc_xe_ok(hc, a_k=1.3, lech=0.0):
    return {'schema': 'truc_xe_v1', 'cho': {k: {'phi_hat_chuan_k_deg': v['psi_chuan_k_deg'] - a_k + lech, 'a_k_deg': a_k,
                                               'u_phi_hat_chuan_k_deg': 0.02, 'trang_thai': 'dat'}
                                           for k, v in hc['cho'].items() if v['trang_thai'] == 'dat'}}


def true_beta(pre, cho):
    a_d, _ = body_at(pre[0], pre[1], pre[2])
    a_s, _ = body_at(cho, 0.0, 0.0)
    return math.degrees(nt.wrap(a_d - a_s))


# ---------------------------------------------------------------- (a), (m): exact projection, F != C
def test_a_m_forward_equation_and_heights_refitted():
    a = 1.0 - H_N / ZC
    F = C + F_OFF
    Ph = nt.thuan(G_TRUE, H_TRUE, 0.3, [0.4, -0.2], F, H_CAM, C, a)
    assert np.allclose(nt.nghich(Ph, H_TRUE, F, H_CAM, C, a), G_TRUE @ rot(0.3).T + [0.4, -0.2], atol=1e-12)
    # (m) the exact equation against the one with the ray centre at C: the F term is not negligible (mm)
    P_wrong = nt.thuan(G_TRUE, H_TRUE, 0.3, [0.4, -0.2], C, H_CAM, C, a)
    assert np.max(np.linalg.norm(P_wrong - Ph, axis=1)) > 1e-4
    rows, recs = build(runs=[])
    obs = []
    for i, w in enumerate(rows):
        r = recs[i]
        obs.append(dict(Ph=np.mean([x['Ph'] for x in r], axis=0), F=r[0]['F'], H=r[0]['H']))
    fit = nt.khop_hinh(obs, H_N, C, a)
    assert np.allclose(fit['h'], H_TRUE, atol=2e-4)                   # (a) heights (s_i) refitted
    assert np.allclose(fit['G'], G_TRUE, atol=2e-5)


def test_a_body_angle_exact(files):
    _, hc, ll, L = run(files, *build())
    assert hc['trang_thai'] == 'dat', hc['ly_do']
    for lid, (pre, post) in zip(L, [(r[1], r[2]) for r in default_runs()]):
        x = L[lid]
        assert x['beta_deg'] == pytest.approx(true_beta(pre, pre[0]), abs=2e-3)


# ---------------------------------------------------------------- (b), (n): rank and replay
def test_b_rank_full_with_a_triangle_and_missing_with_one_place(files):
    _, hc, _, L = run(files, *build(), tag='full')
    assert not hc['suy_bien'] and hc['huong_yeu'] == []
    assert all(x['trang_thai']['beta'] == 'dat' for x in L.values()), [x['ly_do']['beta'] for x in L.values()]
    # every fit window at the same place and heading: heights not separable from the shape
    rows, recs = build(runs=[('L01', ('goc', 0.3, 0.5, {}), ('goc', 0.3, 0.5, {})),
                             ('L02', ('goc', 0.0, 0.0, {}), ('goc', 0.0, 0.0, {}))])
    one = [(i, w) for i, w in enumerate(rows)]
    for i, w in one:
        if w['loai'] in nt.LOAI_KHOP:
            recs[i] = recs_for(*body_at('goc', 0.0, 0.0), n=12, seed=i, id0=recs[i][0]['id'])
    _, hc2, _, L2 = run(files, rows, recs, tag='one')
    assert max(hc2['sigma_h']) > 100 * max(hc['sigma_h'])            # the heights are not pinned by one place
    assert 'thieu_hang' in L2['L01']['ly_do']['beta'] and L2['L01']['trang_thai']['beta'] == 'chua_ket_luan'
    assert 'thieu_hang' in L2['L01']['ly_do']['delta_c_B']           # LLC1-2: the position path inherits it


def test_n_replay_flags_a_target_sensitive_to_the_heights():
    """Fit windows in a narrow band, target far away: the difference moves with h at +-3 sigma -> thieu_hang."""
    a = 1.0 - H_N / ZC
    obs = []
    for j in range(6):
        al, o = 0.7 + 0.001 * j, np.array([0.002 * j, 0.0])
        obs.append(dict(Ph=nt.thuan(G_TRUE, H_TRUE, al, o, C + F_OFF, H_CAM, C, a)
                        + np.random.default_rng(j).normal(0, 2e-6, (3, 2)), F=C + F_OFF, H=H_CAM))
    fit = nt.khop_hinh(obs, H_N, C, a)
    hg = nt.hang(fit, dict(nt.LL_DEFAULTS))
    far = dict(Ph=nt.thuan(G_TRUE, H_TRUE, 2.0, np.array([1.4, 0.9]), C + F_OFF, H_CAM, C, a), F=C + F_OFF, H=H_CAM)
    a_d, _, _ = nt.tu_the(fit['G'], fit['h'], far['Ph'], far['F'], far['H'], C, a)
    beta = nt.wrap(a_d - fit['poses'][0][0])
    rp = nt._replay(fit['G'], fit['h'], hg, obs, 0, far, _Ctx(), dict(nt.LL_DEFAULTS), beta)
    assert rp > 0.01
    near = dict(Ph=obs[3]['Ph'], F=C + F_OFF, H=H_CAM)
    a_n, _, _ = nt.tu_the(fit['G'], fit['h'], near['Ph'], near['F'], near['H'], C, a)
    assert nt._replay(fit['G'], fit['h'], hg, obs, 0, near, _Ctx(), dict(nt.LL_DEFAULTS),
                      nt.wrap(a_n - fit['poses'][0][0])) < 0.01


class _Ctx:
    h_n, C, a_n = H_N, C, 1.0 - H_N / ZC


# ---------------------------------------------------------------- (c): eps_neo at both ends
def test_c_one_anchor_moved_opposite_at_both_ends_stays_within_two_eps():
    a = 1.0 - H_N / ZC
    cfg = dict(nt.LL_DEFAULTS)
    F = C + F_OFF
    bang = nt.eps_neo_bang(G_TRUE, H_TRUE, F, H_CAM, C, a, cfg)
    r_max, eps1 = nt.r_max_tu(bang, cfg)
    assert 2 * eps1 == pytest.approx(cfg['eps_tong_deg'])
    worst = 0.0
    for i in range(3):
        for k in range(720):
            u = np.array([math.cos(k * math.pi / 360), math.sin(k * math.pi / 360)])
            ends = []
            for sg, al, o in ((1, 0.4, np.array([0.1, 0.2])), (-1, 1.1, np.array([0.9, -0.3]))):
                base = nt.thuan(G_TRUE, H_TRUE, al, o, F, H_CAM, C, a)
                Ph = base.copy()
                Ph[i] += sg * 1e-6 * u
                _, _, res = nt.tu_the(G_TRUE, H_TRUE, Ph, F, H_CAM, C, a)
                Ph = base.copy()
                Ph[i] += sg * 1e-6 * u * (0.999 * r_max / res.max())
                ang, _, res = nt.tu_the(G_TRUE, H_TRUE, Ph, F, H_CAM, C, a)
                assert res.max() <= r_max
                ends.append(nt.wrap(ang - al))
            worst = max(worst, abs(math.degrees(ends[1] - ends[0])))
    assert worst <= 2 * eps1 + 1e-9
    # one end reaches eps_neo (continuous maximum, LLC1-11)
    one = 0.0
    for i, k in itertools.product(range(3), range(3600)):
        u = np.array([math.cos(k * math.pi / 1800), math.sin(k * math.pi / 1800)])
        base = nt.thuan(G_TRUE, H_TRUE, 0.0, [0.0, 0.0], F, H_CAM, C, a)
        Ph = base.copy()
        Ph[i] += 1e-6 * u
        ang, _, res = nt.tu_the(G_TRUE, H_TRUE, Ph, F, H_CAM, C, a)
        one = max(one, abs(math.degrees(ang)) * r_max / res.max())
    assert eps1 * 0.999 <= one <= eps1 * (1 + 1e-6)


# ---------------------------------------------------------------- (d): domain
def test_d_domain_lp():
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert nt.trong_mien([0.2, 0.2, 0.2], pts) and not nt.trong_mien([0.6, 0.6, 0.2], pts)


def test_d_target_outside_the_domain(files):
    runs = default_runs() + [('L99', ('goc', 0.0, 25.0, {}), None)]          # turned 25 deg, its pair not usable
    _, _, ll, L = run(files, *build(runs=runs))
    assert 'ngoai_mien' in L['L99']['ly_do']['beta'] and L['L99']['trang_thai']['beta'] == 'chua_ket_luan'
    assert 'ngoai_mien' in L['L99']['ly_do']['delta_c_B']
    assert L['L01']['trang_thai']['beta'] == 'dat'


# ---------------------------------------------------------------- (e): beta without the tag
def test_e_beta_needs_no_tag(files, monkeypatch):
    """The pair of this run has no common set (its pre has no tag) so the run is not a vertex of the domain; the
    domain is tested in (d): here it is held open to look at the tag-free path only."""
    monkeypatch.setattr(nt, 'trong_mien', lambda x, pts, tol=1e-9: True)
    runs = default_runs()
    runs[0] = ('L01', ('goc', -0.02, -0.6, {}), ('K1', 0.02, 0.5, {}))
    rows, recs = build(runs=runs)
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L01' and w['loai'] == 'pre')
    for r in recs[i]:
        r['tag'] = None
    _, _, ll, L = run(files, rows, recs)
    x = L['L01']
    assert x['trang_thai']['beta'] == 'dat', x['ly_do']['beta']
    assert x['beta_deg'] == pytest.approx(true_beta(('goc', -0.02, -0.6), 'goc'), abs=2e-3)
    assert x['trang_thai']['delta_c_B'] == 'chua_ket_luan' and x['delta_a_deg'] is None
    assert any(c['luot_id'] == 'L01' for c in ll['cap_loai'])                  # its pair has no common set


# ---------------------------------------------------------------- (f), (l): merging, integration check, wrap
def test_f_merging_identity_and_ambiguous_branch():
    rng = np.random.default_rng(3)
    psi = 3.13 + rng.normal(0, 0.01, 50)
    al = 3.12 + rng.normal(0, 0.01, 50)
    lhs = nt.wrap(nt.gop_goc(psi) - nt.gop_goc(al))
    assert lhs == pytest.approx(nt.gop_goc([nt.wrap(p - a) for p, a in zip(psi, al)]), abs=1e-12)
    assert not nt.mo_ho(psi, 5.0) and nt.mo_ho(list(psi) + [3.13 + math.radians(10)], 5.0)


def test_f_integration_check_and_wiring_error(files):
    _, hc, _, L = run(files, *build(), tag='ok', truc_xe=truc_xe_ok)
    x = L['L01']
    assert x['kiem_tich_hop_deg'] <= 1e-6 and x['trang_thai']['phi'] == 'dat', x['ly_do']['phi']
    assert x['phi_deg'] == pytest.approx(hc['cho']['goc']['psi_chuan_k_deg'] - 1.3 + x['beta_deg'], abs=1e-9)
    # wiring error: phi_hat of the axis owner not built from this psi_chuan_k
    _, _, _, L2 = run(files, *build(), tag='bad', truc_xe=lambda h: truc_xe_ok(h, lech=0.01))
    assert L2['L01']['kiem_tich_hop_deg'] > 1e-6 and 'loi_ghep' in L2['L01']['ly_do']['phi']
    assert L2['L01']['trang_thai']['phi'] == 'chua_ket_luan' and L2['L01']['trang_thai']['beta'] == 'dat'


def test_f_internal_wiring_error_is_caught(files, monkeypatch):
    """theta of the standard taken from another frame set (deliberate wiring error) -> the identity breaks."""
    orig = nt._w_from_hc

    def other_set(c):
        w = orig(c)
        return dict(w, theta=w['theta'] + 1e-4)
    monkeypatch.setattr(nt, '_w_from_hc', other_set)
    _, _, _, L = run(files, *build(), truc_xe=truc_xe_ok)
    assert L['L01']['kiem_tich_hop_deg'] > 1e-3 and 'loi_ghep' in L['L01']['ly_do']['phi']


def test_f_ambiguous_window_is_dropped(files):
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L02' and w['loai'] == 'pre')
    recs[i][3]['tag'] = dict(recs[i][3]['tag'], psi=recs[i][3]['tag']['psi'] + math.radians(10))
    _, _, ll, L = run(files, rows, recs)
    assert L['L02']['delta_a_deg'] is None and 'thieu_khung_chung_hoac_a6' in L['L02']['ly_do']['delta_c_B']


def test_l_wrap_across_180(files):
    old = PLACES['goc']
    PLACES['goc'] = ((0.0, 0.0), 179.98 - math.degrees(math.atan2(0.0, 0.2)), old[2])
    try:
        runs = default_runs()
        runs[3] = ('L04', ('goc', 0.0, 0.04, {}), ('goc', 0.0, 0.0, {}))      # 179.98 -> 180.02 = -179.98
        _, _, _, L = run(files, *build(runs=runs))
    finally:
        PLACES['goc'] = old
    assert L['L04']['beta_deg'] == pytest.approx(0.04, abs=2e-3)


# ---------------------------------------------------------------- (g), (h): q diagnostic, always chua_ket_luan
def test_g_h_q_sign_and_diagnostics(files):
    xoay = {'schema': 'xoay_v1', 'cho': {}}
    rows, recs = build()
    # rotation centre 10 mm behind the tag along the tag x axis (tag ahead of the centre by 10 mm): q = (0.01, 0)
    for p in ('goc', 'K1'):
        a, o = body_at(p, 0.0, 0.0)
        Pt = o + rot(a) @ CB_TRUE
        r_true = Pt - rot(a + TH_TRUE) @ [0.01, 0.0]
        a_t = 1.0 - TAG_H / ZC
        F = C + F_OFF
        lam = H_CAM / (H_CAM - TAG_H)
        xoay['cho'][p] = {'r_floor_m': (C + a_t * (lam * (r_true - C) + (1 - lam) * (F - C))).tolist(),
                          'trang_thai': 'dat'}
    _, _, _, L = run(files, rows, recs, xoay=xoay)
    x = L['L04']                                                                # goc -> goc
    assert np.allclose(x['q_m'], [0.01, 0.0], atol=2e-5)
    assert x['theta_luot_rad'] == pytest.approx(TH_TRUE, abs=2e-5) and np.allclose(x['q_B_chuan_k_m'], x['q_B_m'], atol=3e-5)
    for y in L.values():
        assert y['trang_thai']['delta_a'] == y['trang_thai']['q'] == y['trang_thai']['h'] == 'chua_ket_luan'
    # plate turned +0.3 deg about the tag centre for the whole session (one placement, its own 12b, J v6 2):
    # q_B unchanged, q_new = R(-0.3 deg) q_old
    th2 = dict(th=TH_TRUE + math.radians(0.3))
    runs = [(lid, (a[0], a[1], a[2], th2), (b[0], b[1], b[2], th2)) for lid, a, b in default_runs()]
    _, _, _, L2 = run(files, *build(runs=runs, plate=th2), tag='rot', xoay=xoay)
    x2 = L2['L04']
    assert np.allclose(x2['q_B_m'], x['q_B_m'], atol=3e-5)
    assert np.allclose(x2['q_m'], rot(-math.radians(0.3)) @ np.array(x['q_m']), atol=3e-5)


# ---------------------------------------------------------------- (i), (j): held-out pairs, plate inside the run
def test_i_pair_with_angle_deviation_is_kept(files):
    runs = default_runs()
    runs[1] = ('L02', ('K1', 0.0, 0.0, {}), ('K2', 0.0, 0.0, dict(th=TH_TRUE + math.radians(0.12), )))
    rows, recs = build(runs=runs)
    _, _, ll, L = run(files, rows, recs, cfg=dict(doi_theta_deg=0.2))
    assert any(c['luot_id'] == 'L02' and c['lech_deg'] == pytest.approx(0.12, abs=3e-3) for c in ll['cap_giu_ngoai'])
    assert not any(c['luot_id'] == 'L02' for c in ll['cap_loai'])
    assert ll['u_beta_deg'] >= 0.12 + 0.03 and 'u_beta_vuot' in ll['ly_do_u_beta']
    assert all(x['trang_thai']['beta'] == 'chua_ket_luan' for x in L.values())


def test_j_plate_turned_in_the_run_and_missing_comparison(files):
    runs = default_runs()
    runs[3] = ('L04', ('goc', 0.0, 0.0, {}), ('goc', 0.0, 0.0, dict(th=TH_TRUE + math.radians(0.3))))
    rows, recs = build(runs=runs)
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L05' and w['loai'] == 'post')
    for r in recs[i]:
        r['tag'] = None
    _, _, _, L = run(files, rows, recs)
    assert L['L04']['tam_doi_trong_luot'] is True
    assert L['L04']['so_sanh_trong_luot']['theta']['trang_thai'] == 'doi'
    assert L['L05']['tam_doi_trong_luot'] is None
    assert L['L01']['tam_doi_trong_luot'] is False


# ---------------------------------------------------------------- (k): phi needs every term
def test_k_missing_u_beta_cam(files):
    _, _, _, L = run(files, *build(), truc_xe=truc_xe_ok, u_beta_cam=None)
    x = L['L01']
    assert x['trang_thai']['beta'] == 'dat' and x['trang_thai']['phi'] == 'chua_ket_luan'
    assert 'thieu_u_beta_cam' in x['ly_do']['phi'] and x['u_phi_truoc_le_deg'] is None


def test_k_u_phi_sum(files):
    _, _, ll, L = run(files, *build(), truc_xe=truc_xe_ok, u_beta_cam=0.007)
    assert L['L01']['u_phi_truoc_le_deg'] == pytest.approx(0.02 + ll['u_beta_deg'] + 0.007, abs=1e-12)


# ---------------------------------------------------------------- 6b delta_c_B: (o)-(v)
def ws(place, s, dpsi, n=1, F=None, **kw):
    """Window summary as tinh_chung gives it (noise-free) for the 6b functions."""
    a, o = body_at(place, s, dpsi)
    F = C + F_OFF if F is None else F
    t = tag_obs(a, o, F, H_CAM, **kw)
    g = np.asarray(kw.get('g_b', (0.0, 0.0)), float)
    d = FakeA6Mod.delta_psi_a6(dict(g_b=g.tolist()), [[t['x'], t['y']]], [t['psi']], C, ZC)[0][0]
    psi = t['psi'] - d
    return dict(alpha=a, psi=psi, theta=nt.wrap(psi - a), o=o, t=np.array([t['x'], t['y']]), s=t['s'], F=F, H=H_CAM,
                n=n)


def _ctx():
    ses = dict(camera=dict(model='B2', f=900.0, cx=960.0, cy=540.0, k=[0.0, 0.0], rvec=[math.pi, 0, 0],
                           tvec=[0, 0, 2.6]), marks={}, fit_ids=[], car_side_m=SIDE, tag_h_m=TAG_H, kappa=TAG_H / ZC,
               plumb_xy_m=C.tolist(), rms_px_H=0.3)
    return nt.LLCtx(ses, H_N, dict(nt.LL_DEFAULTS))


def test_o_shift_and_higher_seat_at_K1():
    ctx, g = _ctx(), np.zeros(2)
    s0 = ws('K1', 0.0, 0.0)
    d1 = ws('K1', 0.0, 0.0, cB=CB_TRUE + [0.0003, 0.0], dh=0.001)
    dcb, raw, dh = nt.delta_c_B(s0, d1, ctx, g)
    assert dh == pytest.approx(0.001, abs=2e-5)
    assert np.linalg.norm(dcb - [0.0003, 0.0]) <= 5e-5
    assert np.linalg.norm(raw - [0.0003, 0.0]) > 4e-4                     # about 0.6 mm without the correction


def test_r_ray_centre_not_C():
    """LL5-1: F - C = 41.3 mm, the tag 4 mm higher, true centre still -> <= 0.005 mm."""
    ctx, g = _ctx(), np.zeros(2)
    dcb, _, _ = nt.delta_c_B(ws('goc', 0.0, 0.0), ws('goc', 0.0, 0.0, dh=0.004), ctx, g)
    assert np.linalg.norm(dcb) <= 5e-6


def test_t_tilt_and_tag_yaw_change():
    """LL5-3: tilt 1.5 deg fixed in the tag frame, tag yaw 5 deg apart about the same true centre -> <= 0.001 mm."""
    ctx = _ctx()
    gb = np.array([math.tan(math.radians(1.5)), 0.0])
    s0 = ws('goc', 0.0, 0.0, g_b=gb)
    d1 = ws('goc', 0.0, 0.0, g_b=gb, th=TH_TRUE + math.radians(5.0))
    dcb, _, dh = nt.delta_c_B(s0, d1, ctx, gb)
    assert np.linalg.norm(dcb) <= 1e-6 and abs(dh) < 1e-4
    dcb0, _, _ = nt.delta_c_B(s0, d1, ctx, np.zeros(2))                 # tilt ignored: the centre moves
    assert np.linalg.norm(dcb0) > 2e-6


def test_s_ratio_repeatability_of_the_12b_scales():
    """LL5-2: 12b scales 1.0004 / 1.0000 / 0.9996 -> u_Dh >= 2 mm (the pair ratio, not the spread about the mean)."""
    ctx, g = _ctx(), np.zeros(2)
    w = [dict(ws('K1', 0.0, 0.0), s=s) for s in (1.0004, 1.0, 0.9996)]
    tr = nt.trai_nhom(w, ctx, g)
    true_err = abs(nt.delta_h_uoc(w[0], w[2], ctx, g))                 # standard 1.0004, target 0.9996, true Dh = 0
    assert true_err > 0.0019
    s0, d1 = w[0], w[2]
    u, parts, u_dh = nt.u_delta_c_B(s0, d1, ctx, g, 0.0, 0.0, tr['s'], 0.0, 0.0)
    assert u_dh >= true_err and parts['dh'] == pytest.approx(u_dh / (H_CAM - TAG_H) * np.linalg.norm(
        nt.tam_ma_that(d1['t'], TAG_H + nt.delta_h_uoc(s0, d1, ctx, g), d1['F'], H_CAM, ctx) - d1['F']), rel=1e-9)
    assert u >= parts['dh']


def test_u_one_anchor_moved_is_inside_the_anchor_term():
    """LL5-4: one anchor displaced with residual <= r_max -> error of c_B <= the anchor term of the index (one end)."""
    a = 1.0 - H_N / ZC
    cfg = dict(nt.LL_DEFAULTS)
    F = C + F_OFF
    bang = nt.eps_neo_bang(G_TRUE, H_TRUE, F, H_CAM, C, a, cfg)
    r_max, _ = nt.r_max_tu(bang, cfg)
    al, o = 0.7, np.array([0.3, 0.1])
    base = nt.thuan(G_TRUE, H_TRUE, al, o, F, H_CAM, C, a)
    for cb in (CB_TRUE, np.array([0.6, 0.4])):                    # short and long lever arm (the angle part)
        lim = nt.eps_vi_tri(bang, r_max, cb)
        t = o + rot(al) @ cb
        worst = 0.0
        for i, k in itertools.product(range(3), range(720)):
            u = np.array([math.cos(k * math.pi / 360), math.sin(k * math.pi / 360)])
            Ph = base.copy()
            Ph[i] += 1e-6 * u
            _, _, res = nt.tu_the(G_TRUE, H_TRUE, Ph, F, H_CAM, C, a)
            Ph = base.copy()
            Ph[i] += 1e-6 * u * (0.999 * r_max / res.max())
            ang, oo, res = nt.tu_the(G_TRUE, H_TRUE, Ph, F, H_CAM, C, a)
            worst = max(worst, float(np.linalg.norm(rot(-ang) @ (t - oo) - cb)))
        assert lim * 0.99 <= worst <= lim * (1 + 1e-6)


def test_o_p_q_v_states_in_lan_lap(files):
    runs = default_runs()
    runs[3] = ('L04', ('goc', 0.0, 0.0, dict(cB=CB_TRUE + [0.0003, 0.0], dh=0.001)),
               ('goc', 0.0, 0.0, dict(cB=CB_TRUE + [0.0003, 0.0], dh=0.001)))
    runs[4] = ('L05', ('K1', 0.0, 0.0, dict(dh=0.006)), ('goc', 0.0, 0.0, dict(dh=0.006)))
    runs.append(('L07', ('K2', 0.0, 0.0, {}), ('K2', 0.0, 0.0, {})))
    rows, recs = build(runs=runs, fit_places=('goc', 'K1'))
    _, _, ll, L = run(files, rows, recs, cfg=dict(doi_c_m=0.01, doi_s=0.01))
    x = L['L04']
    assert x['trang_thai']['delta_c_B'] == 'dat', x['ly_do']['delta_c_B']           # (o) at the place goc
    assert np.linalg.norm(np.array(x['delta_c_B_m']) - [0.0003, 0.0]) <= 5e-5
    assert x['u_delta_c_B_m'] > 0 and set(x['u_delta_c_B_phan']) == {'nhieu', 'dh', 'nghieng', 'lech_cho', 'neo',
                                                                       'giu_ngoai'}
    assert 'dh_vuot_ngoi_chot' in L['L05']['ly_do']['delta_c_B']                     # (p)
    assert L['L05']['trang_thai']['delta_c_B'] == 'chua_ket_luan'
    assert 'khong_co_cua_so_chuan' in L['L07']['ly_do']['beta']                      # (q) no standard at this place
    assert L['L07']['trang_thai']['delta_c_B'] == 'chua_ket_luan'
    # (v) no same-place held-out pair in the session
    runs2 = [r for r in default_runs() if r[1][0] != r[2][0]]
    _, _, _, L3 = run(files, *build(runs=runs2 + [('L08', ('goc', 0.0, 0.0, {}), ('K1', 0.0, 0.0, {}))]), tag='v')
    assert all('thieu_giu_ngoai_vi_tri' in y['ly_do']['delta_c_B'] for y in L3.values())
    assert all(y['trang_thai']['delta_c_B'] == 'chua_ket_luan' for y in L3.values())


# ---------------------------------------------------------------- lock, refusals, lan_lap_id
def test_lock_sha_and_refusals(files):
    _, hc, _, _ = run(files, *build(), tag='lk')
    p = files / 'hc_lk' / 'hieu_chuan_neo.json'
    (files / 'a6b.json').write_text(json.dumps(dict(g_b=[0.001, 0.0], eps_g=1e-4)), encoding='utf-8')
    args = (str(p), str(files / 'cs_lk.json'), str(files / 'session.json'))
    with pytest.raises(ValueError, match='A6'):                       # another A6 file than the locked one
        nt.run_lan_lap(*args, str(files / 'll_x'), str(files / 'a6b.json'), 'T1', a6_mod=FakeA6Mod, _records={})
    with pytest.raises(ValueError, match='locked'):                   # LLC1-6: a threshold changed after the lock
        nt.run_lan_lap(*args, str(files / 'll_y'), str(files / 'a6.json'), 'T1', cfg=dict(CFG, u_beta_max_deg=0.2),
                       a6_mod=FakeA6Mod, _records={})
    with pytest.raises(ValueError, match='tam_id'):
        nt.run_lan_lap(*args, str(files / 'll_z'), str(files / 'a6.json'), 'T2', cfg=CFG, a6_mod=FakeA6Mod, _records={})
    (files / 'other.mov').write_bytes(b'y')
    with pytest.raises(ValueError, match='mau-clip'):                 # LLC1-6: another template content
        nt.run_lan_lap(*args, str(files / 'll_w'), str(files / 'a6.json'), 'T1', cfg=CFG, mau_clip=str(files / 'other.mov'),
                       a6_mod=FakeA6Mod, _records={})
    with pytest.raises(ValueError, match='u_beta_cam'):               # LLC1-1
        nt.run_lan_lap(*args, str(files / 'll_v'), str(files / 'a6.json'), 'T1', cfg=CFG, u_beta_cam_deg=-0.04,
                       a6_mod=FakeA6Mod, _records={})
    o = json.load(open(p, encoding='utf-8'))
    o['r_max_m'] *= 2
    p.write_text(json.dumps(o), encoding='utf-8')
    with pytest.raises(ValueError, match='lock'):                     # changed after the lock
        nt.doc_hieu_chuan(str(p))


def test_too_few_anchors_after_clip_r(files):
    (files / 'kiem.json').write_text(json.dumps(dict(rong=True, duoc=True, vung_than_px=NEO_BOXES[:2])), encoding='utf-8')
    c, hc, ll, L = run(files, *build(), tag='few')
    assert c == 1 and hc['trang_thai'] == 'chua_ket_luan' and hc['ly_do'] == ['thieu_neo']
    assert ll['luot'] == [] and ll['trang_thai_buoi'] == 'chua_ket_luan'


def test_lan_lap_id_changes_when_the_plate_changes(files):
    runs = default_runs()
    runs[2] = ('L03', ('K2', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])), ('K2', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])))
    runs[3] = ('L04', ('goc', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])), ('goc', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])))
    _, _, _, L = run(files, *build(runs=runs))
    assert L['L01']['lan_lap_id'] == L['L02']['lan_lap_id'] == 'L0'
    assert L['L03']['lan_lap_id'] == L['L04']['lan_lap_id'] != L['L02']['lan_lap_id']
    # J v6 2: a new placement without its own 12b -> no standard, G1-G4 not concluded
    assert 'khong_co_12b_cung_lan_lap' in L['L03']['ly_do']['beta'] and L['L03']['chuan_12b'] is None
    assert L['L01']['chuan_12b'] == '12b_goc'


# ---------------------------------------------------------------- end to end on rendered clips (pixel layer)
def test_end_to_end_rendered_clips(tmp_path, monkeypatch):
    """Clips rendered from a known camera (scene of test_c7_nhac_tam): every window a static clip; the anchor boxes are
    found from the tag-predicted place with the template turned by the image rotation (40 deg between the places).
    beta = true body rotation from the standard window of the place."""
    import test_c7_nhac_tam as t7
    monkeypatch.setattr(nt, 'trong_mien', lambda x, pts, tol=1e-9: True)   # few windows: the domain is tested in (d)
    d = tmp_path
    floor = t7.floor_img(t7.CAM)
    t7.write_video(d / 'H.avi', [floor] * 6)
    ses = cm.calibrate(str(d / 'H.avi'), t7.marks_dict(), t7.ZC, t7.PLUMB, t7.TAG_H, t7.CAR_SIDE)
    (d / 'session.json').write_text(json.dumps(cm._json(ses)), encoding='utf-8')
    ses_sha = cm.sha256(str(d / 'session.json'))
    anchors = [(0.15, 0.125), (-0.15, -0.125), (-0.15, 0.125)]
    places = dict(goc=((0.0, 0.0), 20.0), K1=((0.5, 0.25), 60.0))
    rows, nclip = [], [0]

    def clip(body, name):
        f = floor.copy()
        t7.render_car(f, body, (0.0, 0.0, 0.0))
        path = d / f'{name}.avi'
        t7.write_video(path, [f] * 6)
        fr = [{'id': k, 'pts_s': t} for k, t, _ in cm.iter_frames(str(path))]
        return str(path), fr

    def add(luot, cho, loai, s, dpsi):
        (x, y), hd = places[cho]
        a = math.radians(hd + dpsi)
        body = (x + s * math.cos(a), y + s * math.sin(a), a)
        path, fr = clip(body, f'w{len(rows)}')
        rows.append({'schema': 'cua_so_v1', 'luot_id': luot, 'cho': cho, 'loai': loai, 'clip': os.path.basename(path),
                     'clip_sha256': cm.sha256(path), 'session_sha256': ses_sha, 'khung': fr,
                     'khung_ma_hop_le': [k['id'] for k in fr], 't_canh_s': None})
        return body
    for cho in places:
        for j, s in enumerate([0.0, 0.2, -0.2, 0.2, 0.0]):
            add(f'12b_{cho}', cho, 'ref_12b' if j == 0 else 'dung_12b', s, 0.1 * j)
    truth = {}
    for luot, cho, s, dp in (('L01', 'goc', 0.05, 0.7), ('L02', 'K1', -0.05, -0.4), ('L03', 'goc', 0.0, 0.3)):
        truth[luot] = dp
        add(luot, cho, 'pre', s, dp)
        add(luot, cho, 'post', s + 0.002, dp)          # another recording (a byte-identical clip is refused, LLC1-7)
    (d / 'cs.json').write_text(json.dumps({'schema': 'cua_so_v1', 'cua_so': rows}), encoding='utf-8')
    # anchor boxes on the template frame = first window (goc, ref_12b)
    (x0, y0), hd = places['goc']
    a0 = math.radians(hd)
    spec, boxes = [], []
    for name, (bx, by) in zip(('A', 'B', 'oc1'), anchors):
        P = np.array([x0 + math.cos(a0) * bx - math.sin(a0) * by, y0 + math.sin(a0) * bx + math.cos(a0) * by, t7.TAG_H])
        u, v = t7.CAM.project(P[None])[0]
        b = [int(round(u - 8)), int(round(v - 8)), 16, 16]
        boxes.append(b)
        spec.append(f'{name}:' + ','.join(map(str, b)))
    (d / 'kiem.json').write_text(json.dumps(dict(rong=True, duoc=True, n0='w0.avi', vung_than_px=boxes)), encoding='utf-8')
    (d / 'a6.json').write_text(json.dumps(dict(g_b=[0.0, 0.0], eps_g=1e-4)), encoding='utf-8')
    # 1 px is about 3 mm on this synthetic floor: anchor / residual thresholds opened for the smoke test only
    cfg = dict(min_khung=4, min_cap=2, eps_tong_deg=1.0, still_neo_m=0.003, u_beta_max_deg=1.0, doi_theta_deg=0.5,
               doi_c_m=0.003, doi_s=0.005)
    mau = (str(d / 'w0.avi'), str(d / 'session.json'), 0)
    c = nt.run_hieu_chuan(str(d / 'cs.json'), str(d / 'session.json'), ';'.join(spec), mau, str(d / 'hc'), str(d / 'a6.json'),
                          'T1', kiem_r=[str(d / 'kiem.json')], h_neo=t7.TAG_H, clip_dir=str(d), cfg=cfg, a6_mod=FakeA6Mod)
    hc = json.load(open(d / 'hc' / 'hieu_chuan_neo.json', encoding='utf-8'))
    assert c == 0 and hc['trang_thai'] == 'dat', (hc['ly_do'], hc.get('cua_so_loi'))
    assert np.allclose(hc['h'], t7.TAG_H, atol=0.01)
    nt.run_lan_lap(str(d / 'hc' / 'hieu_chuan_neo.json'), str(d / 'cs.json'), str(d / 'session.json'), str(d / 'll'),
                   str(d / 'a6.json'), 'T1', clip_dir=str(d), cfg=cfg, a6_mod=FakeA6Mod)
    ll = json.load(open(d / 'll' / 'lan_lap.json', encoding='utf-8'))
    L = {x['luot_id']: x for x in ll['luot']}
    for luot, dp in truth.items():
        # window angles scatter about +-0.03 deg on this 3 mm / px rendering (template interpolation): wiring check
        assert L[luot]['beta_deg'] == pytest.approx(dp, abs=0.05), (luot, L[luot]['beta_deg'])
        # the tag angle of this coarse rendering shifts with the pose (about 0.1 deg, not averaged by identical frames):
        # only the centre and the scale comparisons are asserted here (8 is tested on records above)
        so = L[luot]['so_sanh_trong_luot']
        assert so['c_B']['trang_thai'] == so['s']['trang_thai'] == 'khong_doi'
        assert np.linalg.norm(L[luot]['delta_c_B_m']) < 3e-4


# ---------------------------------------------------------------- Codex code round 1 (LLC1-n)
def test_llc1_1_phi_needs_u_phi_hat(files):
    def tx(hc):
        o = truc_xe_ok(hc)
        for v in o['cho'].values():
            del v['u_phi_hat_chuan_k_deg']
        return o
    _, _, _, L = run(files, *build(), truc_xe=tx)
    x = L['L01']
    assert x['trang_thai']['beta'] == 'dat' and x['trang_thai']['phi'] == 'chua_ket_luan'
    assert 'thieu_u_phi_hat' in x['ly_do']['phi'] and x['u_phi_truoc_le_deg'] is None


def test_llc1_2_8_anchor_residual_blocks_position_and_pair(files):
    """One anchor of a held-out end 0.8 mm off: its residual is far above r_max -> no delta_c_B dat for that run, the
    pair leaves the yaw pairs and the domain (LLC1-2, LLC1-8)."""
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L04' and w['loai'] == 'pre')
    for r in recs[i]:
        r['Ph'] = r['Ph'] + np.array([[0.0008, 0.0], [0.0, 0.0], [0.0, 0.0]])
    _, _, ll, L = run(files, rows, recs)
    x = L['L04']
    assert 'du_neo_vuot_r_max' in x['ly_do']['delta_c_B'] and x['trang_thai']['delta_c_B'] == 'chua_ket_luan'
    assert any(c['luot_id'] == 'L04' and c['ly_do'] == 'du_neo_vuot_r_max' for c in ll['cap_loai'])
    assert not any(n.get('luot_id') == 'L04' for n in ll['mien']['nguon'])


def test_llc1_3_position_pairs_not_filtered_by_yaw(files):
    runs = default_runs()
    runs[3] = ('L04', ('goc', 0.0, 0.0, {}), ('goc', 0.0, 0.0, dict(cB=CB_TRUE + [0.0008, 0.0])))
    _, _, ll, L = run(files, *build(runs=runs))
    assert any(c['luot_id'] == 'L04' and c['ly_do'] == 'tam_doi_c_B_hoac_thang' for c in ll['cap_loai'])
    g = {x['luot_id']: x['lech_m'] for x in ll['giu_ngoai_vi_tri']}
    assert g['L04'] == pytest.approx(0.0008, abs=5e-5)
    assert L['L01']['u_delta_c_B_phan']['giu_ngoai'] == pytest.approx(max(g.values()), rel=1e-12)


def test_llc1_4_spread_from_the_standard_12b_run_only(files):
    """A 12b window of another run at the same place (plate moved 0.8 mm) is not in the standard group."""
    rows, recs = build()
    extra_rows, extra = build(runs=[], fit_places=('goc',), e_places=(), plate=dict(cB=CB_TRUE + [0.0008, 0.0]))
    base = max(r['id'] for v in recs.values() for r in v) + 100
    for j, (w, k) in enumerate(zip(extra_rows, sorted(extra))):
        if w['loai'] != 'dung_12b':
            continue
        rr = [dict(r, id=r['id'] + base, t=(r['id'] + base) / 60.0) for r in extra[k]]
        recs[len(rows)] = rr
        rows.append(win('12b_goc_lan2', 'goc', 'dung_12b', rr))
    _, hc, _, L = run(files, rows, recs)
    assert hc['cho']['goc']['n_12b_cung_lan_lap'] == 7
    assert L['L04']['cung_lan_lap_chi_so']['c_B']['nguong'] < 1e-4


def test_llc1_5_a6_not_for_this_plate(files):
    (files / 'a6t.json').write_text(json.dumps(dict(g_b=[0.0, 0.0], eps_g=1e-4, tam_khac=['T9'])), encoding='utf-8')
    _, hc, _, L = run(files, *build(), a6='a6t.json', tam='T9')
    assert all(v['trang_thai'] == 'chua_ket_luan' for v in hc['cho'].values())
    x = L['L01']
    assert x['delta_a_deg'] is None and x['trang_thai']['delta_c_B'] == 'chua_ket_luan'
    assert x['beta_deg'] is not None                                   # the anchor path needs no A6


def test_llc1_6_7_window_file_checks(files):
    rows, recs = build()
    bad = [dict(w) for w in rows]
    bad[-1] = dict(bad[-1], khung=bad[-1]['khung'][:3] + bad[-1]['khung'][2:])
    with pytest.raises(ValueError, match='repeated'):
        nt.kiem_f2(bad)
    bad = [dict(w) for w in rows] + [dict(rows[-1], luot_id='L99')]
    with pytest.raises(ValueError, match='same frames'):
        nt.kiem_f2(bad)
    bad = [dict(w) for w in rows]
    bad[-2] = dict(bad[-2], loai='post')
    with pytest.raises(ValueError, match='two post'):
        nt.kiem_f2(bad)
    bad = [dict(w) for w in rows] + [dict(rows[0], loai='pre', luot_id='L99', khung=rows[0]['khung'][:5])]
    with pytest.raises(ValueError, match='fit and the held-out'):
        nt.kiem_f2(bad)
    rows2, recs2 = build()
    rows2[-1]['ket_luan'] = dict(dung_yen=False)
    _, _, ll, L = run(files, rows2, recs2, tag='kl')
    assert any('cam_khong_dung_yen' in x['ly_do'] for x in ll['cua_so_loi'])


def test_llc1_9_anchor_boxes_and_clip_r_source():
    with pytest.raises(ValueError, match='overlaps'):
        nt.parse_neo('A:100,100,18,18;B:200,100,18,18;oc1:105,100,18,18')
    neo = nt.parse_neo(NEO)
    import tempfile
    d = tempfile.mkdtemp()
    f = os.path.join(d, 'k.json')
    with open(f, 'w', encoding='utf-8') as fh:
        json.dump(dict(rong=True, duoc=True, n0='khac.mov', vung_than_px=NEO_BOXES), fh)
    ok, _ = nt.kiem_r_neo(neo, [f], 'mau.mov')
    assert not any(ok.values())
    ok, _ = nt.kiem_r_neo(neo, [f], 'khac.mov')
    assert all(ok.values())


def test_llc1_10_rank_from_the_projected_jacobian():
    """Two heights nearly collinear after removing G and the poses: the weak direction is not lost."""
    fit = dict(jac=np.array([[1.0, 1.0 + 1e-8, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
               idx_h=[0, 1], m=10, p=3, ssr=1e-12, hoi_tu=True)
    hg = nt.hang(fit, dict(nt.LL_DEFAULTS))
    assert hg['suy_bien'] or max(hg['sigma_h']) > 1.0
    fit['hoi_tu'] = False
    assert nt.hang(fit, dict(nt.LL_DEFAULTS))['suy_bien']


def test_llc1_13_scale_compared_without_a6_and_within_place_thresholds(files):
    """Post window without A6 (theta not known) but the plate seated 2 mm higher: the scale says doi."""
    psi_post = 45.0 + math.degrees(TH_TRUE) + 3.0
    (files / 'a6h.json').write_text(json.dumps(dict(g_b=[0.0, 0.0], eps_g=1e-4, hong_psi_deg=[psi_post - 0.5,
                                                                                                psi_post + 0.5])),
                                    encoding='utf-8')
    runs = default_runs()
    runs[3] = ('L04', ('goc', 0.0, 0.0, {}), ('goc', 0.0, 0.0, dict(dh=0.002, th=TH_TRUE + math.radians(3.0))))
    rows, recs = build(runs=runs)
    _, _, _, L = run(files, rows, recs, a6='a6h.json')
    so = L['L04']['so_sanh_trong_luot']
    assert so['theta']['trang_thai'] == 'khong_xac_dinh'               # no A6 at the post
    assert so['s']['trang_thai'] == 'doi' and L['L04']['tam_doi_trong_luot'] is True


def test_llc1_15_run_order_contradicting_the_clip(files):
    rows, recs = build()
    for w in rows:
        if w['luot_id'] in ('L01', 'L02'):
            w['clip_sha256'] = 'same'
    i1 = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L01' and w['loai'] == 'pre')
    i2 = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L02' and w['loai'] == 'pre')
    for i, t0 in ((i1, 9.0), (i2, 1.0)):                               # L02 shot first in the same clip
        for k, r in enumerate(recs[i]):
            r['t'] = t0 + k / 60.0
        rows[i]['khung'] = [{'id': r['id'], 'pts_s': r['t']} for r in recs[i]]
    _, _, _, L = run(files, rows, recs)
    assert all(x['lan_lap_id'] is None and x['lan_lap_id_ly_do'] == 'thu_tu_mau_thuan' for x in L.values())


def test_llc1_17_q_needs_both_e_ends(files):
    xoay = {'schema': 'xoay_v1', 'cho': {'goc': {'r_floor_m': [0.0, 0.0], 'trang_thai': 'dat'},
                                         'K1': {'r_floor_m': [1.0, 0.0], 'trang_thai': 'chua_ket_luan'}}}
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['cho'] == 'goc' and w['loai'] == 'E_cuoi')
    keep = [j for j in range(len(rows)) if j != i]                       # goc keeps only its E_dau
    rows, recs = [rows[j] for j in keep], {n: recs[j] for n, j in enumerate(keep)}
    _, _, _, L = run(files, rows, recs, xoay=xoay)
    assert 'thieu_hai_dau_E' in L['L04']['ly_do']['q'] and L['L04']['q_m'] is None
    assert 'thieu_xoay_v1_dat' in L['L05']['ly_do']['q']


def test_g_q_with_a_tilted_plate_and_a6(files):
    """LLC1-19: tilt g_b != 0 (A6 removes the yaw bias, the centre shift is removed in 6b): q in the frame of psi' is
    still the 10 mm lever arm, i.e. the rotation centre = t - R(psi') q."""
    gb = [0.01, -0.006]
    (files / 'a6g.json').write_text(json.dumps(dict(g_b=gb, eps_g=1e-4)), encoding='utf-8')
    xoay = {'schema': 'xoay_v1', 'cho': {}}
    for p in ('goc', 'K1'):
        a, o = body_at(p, 0.0, 0.0)
        r_true = o + rot(a) @ CB_TRUE - rot(a + TH_TRUE) @ [0.01, 0.0]
        lam = H_CAM / (H_CAM - TAG_H)
        xoay['cho'][p] = {'r_floor_m': (C + (1.0 - TAG_H / ZC) * (lam * (r_true - C) + (1 - lam) * F_OFF)).tolist(),
                          'trang_thai': 'dat'}
    runs = [(lid, (pre[0], pre[1], pre[2], dict(g_b=gb)), (post[0], post[1], post[2], dict(g_b=gb)))
            for lid, pre, post in default_runs()]
    _, _, _, L = run(files, *build(runs=runs, plate=dict(g_b=gb)), a6='a6g.json', xoay=xoay)
    assert np.allclose(L['L04']['q_m'], [0.01, 0.0], atol=3e-5)



def test_replay_gate_in_lan_lap(files, monkeypatch):
    """The replay result is a gate of beta and of the position path (v4 3)."""
    monkeypatch.setattr(nt, '_replay', lambda *a, **k: 0.5)
    _, _, _, L = run(files, *build())
    assert all('thieu_hang' in x['ly_do']['beta'] and 'thieu_hang' in x['ly_do']['delta_c_B'] for x in L.values())


def test_j6_two_12b_runs_in_a_session(files):
    """J v6 2: 12b at the start and after a new placement; each run uses the 12b of its placement; a new placement
    without a 12b, or a 12b at another place only, has no standard."""
    shift = dict(cB=CB_TRUE + [0.002, 0.0])
    runs = [('L01', ('goc', 0.0, 0.0, {}), ('goc', 0.01, 0.2, {})),
            ('L02', ('goc', 0.0, 0.1, shift), ('goc', 0.01, 0.3, shift)),
            ('L03', ('K1', 0.0, 0.1, shift), ('K1', 0.01, 0.2, shift)),
            ('L04', ('goc', 0.0, 0.1, dict(cB=CB_TRUE + [0.004, 0.0])), ('goc', 0.0, 0.2, dict(cB=CB_TRUE + [0.004, 0.0])))]
    rows, recs = build(runs=runs)
    r2, k2 = build(runs=[], fit_places=('goc',), e_places=(), plate=shift)
    base = max(r['id'] for v in recs.values() for r in v) + 100
    extra = []
    for w, k in zip(r2, sorted(k2)):
        rr = [dict(r, id=r['id'] + base, t=(r['id'] + base) / 60.0) for r in k2[k]]
        extra.append((win('12b_goc_2', 'goc', w['loai'], rr), rr))
    fit = [(w, recs[i]) for i, w in enumerate(rows) if w['loai'] in nt.LOAI_KHOP]
    runw = {lid: [(w, recs[i]) for i, w in enumerate(rows) if w['luot_id'] == lid] for lid in ('L01', 'L02', 'L03', 'L04')}
    order = fit + runw['L01'] + extra + runw['L02'] + runw['L03'] + runw['L04']
    rows2, recs2 = [], {}
    for j, (w, rr) in enumerate(order):
        w = dict(w, clip_sha256=f'c{j}')                                # one clip per window: no PTS order across
        recs2[j] = rr
        rows2.append(w)
    _, hc, _, L = run(files, rows2, recs2)
    assert 'goc|12b_goc' in hc['chuan_12b'] and 'goc|12b_goc_2' in hc['chuan_12b']
    assert hc['cho']['goc']['ly_do'] == 'nhieu_12b_xem_chuan_12b'
    assert L['L01']['chuan_12b'] == '12b_goc' and L['L02']['chuan_12b'] == '12b_goc_2'
    assert L['L01']['lan_lap_id'] != L['L02']['lan_lap_id'] == L['L03']['lan_lap_id']
    assert 'khong_co_cua_so_chuan' in L['L03']['ly_do']['beta']            # 12b of this placement only at goc
    assert 'khong_co_12b_cung_lan_lap' in L['L04']['ly_do']['beta']
    assert L['L02']['delta_c_B_m'] is not None and np.linalg.norm(L['L02']['delta_c_B_m']) < 2e-4   # vs its own 12b
    assert L['L02']['beta_deg'] == pytest.approx(0.1, abs=2e-3)


def test_j6_tag_anchor_intersection_and_mat_neo(files):
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L02' and w['loai'] == 'pre')
    for r in recs[i]:
        r['neo_ok'], r['Ph'] = False, None
    _, _, _, L = run(files, rows, recs)
    assert L['L02']['dau']['pre']['mat_neo'] is True and L['L02']['dau']['pre']['khung_giao_ma_neo']['id'] == []
    x = L['L01']['dau']['pre']
    assert x['mat_neo'] is False and len(x['khung_giao_ma_neo']['id']) == 12
    assert x['khung_giao_sha256'] == nt.sha_ids(x['khung_giao_ma_neo']['id'])
    assert np.array(x['Ph_neo_m']).shape == (3, 2) and np.array(L['L01']['dau']['chuan']['Ph_neo_m']).shape == (3, 2)
