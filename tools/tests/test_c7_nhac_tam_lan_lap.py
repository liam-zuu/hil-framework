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
    w = {'schema': 'cua_so_v1', 'luot_id': luot, 'cho': cho, 'loai': loai, 'clip': 'clip.mov',
         'clip_sha256': 'x', 'session_sha256': None, 'khung': [{'id': r['id'], 'pts_s': r['t']} for r in recs],
         'khung_ma_hop_le': [r['id'] for r in recs], 't_canh_s': None, 'frames_sha256': f'fr_{luot}'}
    if loai in nt.LOAI_GIU:
        w['ket_luan'] = {'dung_yen': True}                            # CAM's verdict (required, LLC1-7 round 2)
    return w


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


def kiem_r(mau, boxes, idx=0):
    """kiem.json of 'kiem --rong' checked against the template clip mau at template frame idx (LLC1-9)."""
    return dict(rong=True, duoc=True, n0=os.path.basename(str(mau)), n0_sha256=cm.sha256(str(mau)), mau_idx=idx,
                vung_than_px=boxes)


@pytest.fixture
def files(tmp_path):
    ses = dict(camera=dict(model='B2', f=900.0, cx=960.0, cy=540.0, k=[0.0, 0.0], rvec=[math.pi, 0.0, 0.0],
                           tvec=[0.0, 0.0, 2.6]),
               marks={}, fit_ids=[1, 2], car_side_m=SIDE, tag_h_m=TAG_H, kappa=TAG_H / ZC, plumb_xy_m=C.tolist(),
               rms_px_H=0.3)
    (tmp_path / 'session.json').write_text(json.dumps(ses), encoding='utf-8')
    (tmp_path / 'a6.json').write_text(json.dumps(dict(schema='c8b_a6_v2', g_b=[0.0, 0.0], eps_g=1e-4)), encoding='utf-8')
    (tmp_path / 'mau.mov').write_bytes(b'x')
    (tmp_path / 'kiem.json').write_text(json.dumps(kiem_r(tmp_path / 'mau.mov', NEO_BOXES)), encoding='utf-8')
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
    if truc_xe is not None:                                           # one truc_xe_v1 file per place (HIEUCHUAN)
        tx = []
        for n, blk in enumerate(truc_xe(hc)):
            nguon = blk.setdefault('nguon', [])                       # HIEUCHUAN always lists the F6 and A6 used
            roles = {x['vai_tro'] for x in nguon}
            if 'hieu_chuan_neo' not in roles:
                nguon.append({'vai_tro': 'hieu_chuan_neo', 'sha256': cm.sha256(str(d / f'hc_{tag}' / 'hieu_chuan_neo.json'))})
            if 'c8b_a6_v2' not in roles:
                nguon.append({'vai_tro': 'c8b_a6_v2', 'sha256': cm.sha256(str(d / a6))})
            tx.append(d / f'tx_{tag}_{n}.json')
            tx[-1].write_text(json.dumps(blk), encoding='utf-8')
    xo = None
    if xoay is not None:
        xo = d / f'xo_{tag}.json'
        xo.write_text(json.dumps(xoay), encoding='utf-8')
    nt.run_lan_lap(str(d / f'hc_{tag}' / 'hieu_chuan_neo.json'), str(d / f'cs_{tag}.json'), str(d / 'session.json'),
                   str(d / f'll_{tag}'), str(d / a6), tam, truc_xe=None if tx is None else [str(x) for x in tx],
                   xoay=None if xo is None else str(xo), u_beta_cam_deg=u_beta_cam, cfg=dict(CFG, **(cfg or {})),
                   a6_mod=FakeA6Mod, _records=hold)
    ll = json.load(open(d / f'll_{tag}' / 'lan_lap.json', encoding='utf-8'))
    return c1, hc, ll, {x['luot_id']: x for x in ll['luot']}


def truc_xe_ok(hc, a_k=1.3, lech=0.0):
    """truc_xe_v1 as HIEUCHUAN writes it (c8_hc_a.py tinh): one block per place and issued 12b attempt, locked state
    'khoa', the attempt's frames.csv in nguon (CT1-2)."""
    return [{'schema': 'truc_xe_v1', 'cho': v['cho'], 'mien_hieu_luc': {'cho': v['cho'], 'thu_id_cap': v['luot_id']},
             'nguon': [{'vai_tro': f"{v['luot_id']}:frames", 'duong_dan': 'x', 'sha256': f"fr_{v['luot_id']}"}],
             'phi_hat_chuan_k_deg': v['psi_chuan_k_deg'] - a_k + lech, 'a_k_deg': a_k, 'u_phi_hat_chuan_k_deg': 0.02,
             'trang_thai': 'khoa'} for v in hc['chuan_12b'].values() if v['trang_thai'] == 'dat']


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
    s2 = [math.inf if v is None else v for v in hc2['sigma_h']]         # null = not finite (strict JSON, LLC2-6)
    assert max(s2) > 100 * max(hc['sigma_h'])                         # the heights are not pinned by one place
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
    # beta itself is computed without any tag frame (anchors only) ...
    assert x['beta_deg'] == pytest.approx(true_beta(('goc', -0.02, -0.6), 'goc'), abs=2e-3)
    # ... but without a tag the pre cannot be linked to the placement of the standard (J v6 2; LLC3-1: the post
    # cannot stand in for the pre), so the state is not concluded
    assert x['trang_thai']['beta'] == 'chua_ket_luan' and 'lan_lap_khong_xac_dinh' in x['ly_do']['beta']
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
    (files / 'kiem.json').write_text(json.dumps(kiem_r(files / 'mau.mov', NEO_BOXES[:2])), encoding='utf-8')
    c, hc, ll, L = run(files, *build(), tag='few')
    assert c == 1 and hc['trang_thai'] == 'chua_ket_luan' and hc['ly_do'] == ['thieu_neo']
    assert ll['trang_thai_buoi'] == 'chua_ket_luan'
    # LLC2-3: still one record per run (the plate check of J v6 5 runs for every run), nothing concluded
    assert sorted(L) == [f'L{j:02d}' for j in range(1, 7)]
    for x in L.values():
        assert all(v == 'chua_ket_luan' for v in x['trang_thai'].values())
        assert x['tam_doi_trong_luot'] is None and x['lan_lap_id'] is None
        assert all(v['trang_thai'] == 'khong_xac_dinh' for v in x['so_sanh_trong_luot'].values())
        assert x['dau']['pre']['mat_neo'] is True and x['dau']['pre']['khung_giao_ma_neo']['id'] == []
        assert 'thieu_hinh_neo_hieu_chuan' in x['ly_do']['beta']


def test_lan_lap_id_changes_when_the_plate_changes(files):
    runs = default_runs()
    runs[2] = ('L03', ('K2', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])), ('K2', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])))
    runs[3] = ('L04', ('goc', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])), ('goc', 0.0, 0.0, dict(cB=CB_TRUE + [0.002, 0.0])))
    _, _, _, L = run(files, *build(runs=runs))
    assert L['L01']['lan_lap_id'] == L['L02']['lan_lap_id'] == 'L0'
    # J v6 2: a new placement without its own 12b -> no standard, G1-G4 not concluded; after a valid doi in the group
    # (L03) a later run that matches the standard again (L05) has no standard either: the plate may have been
    # re-seated back (LLC3-1)
    assert L['L05']['lan_lap_id'] is None and 'lan_lap_sau_doi' in L['L05']['ly_do']['beta']
    for lid, why in (('L03', 'tam_doi_so_chuan'), ('L04', 'lan_lap_sau_doi')):
        assert L[lid]['lan_lap_id'] is None and L[lid]['chuan_12b'] is None
        assert why in L[lid]['ly_do']['beta'] and L[lid]['doi_so_chuan'] is True
    assert L['L01']['chuan_12b'] == '12b_goc' and L['L01']['doi_so_chuan'] is False


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
                     'khung_ma_hop_le': [k['id'] for k in fr], 't_canh_s': None, 'ket_luan': {'dung_yen': True}})
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
    (d / 'kiem.json').write_text(json.dumps(kiem_r(d / 'w0.avi', boxes)), encoding='utf-8')
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
        for v in o:
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
    # LLC2-4 (round 2): without A6 for this plate its tilt is unknown -> c_B / scale comparisons khong_xac_dinh, so the
    # placement of every run is unknown (J v6 2) and no standard is used; never 'doi' from another plate's g_b
    assert all(v['trang_thai'] == 'khong_xac_dinh' for y in L.values() for v in y['so_sanh_trong_luot'].values())
    assert x['beta_deg'] is not None                                   # the anchor path itself needs no A6 ...
    assert x['trang_thai']['beta'] == 'chua_ket_luan' and 'lan_lap_khong_xac_dinh' in x['ly_do']['beta']   # ... but
    assert x['chuan_12b'] is None and x['lan_lap_id'] is None          # the placement is not verified (J v6 2)


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
    os.makedirs(os.path.join(d, 'cu'))
    os.makedirs(os.path.join(d, 'moi'))
    cu, moi = os.path.join(d, 'cu', 'N0.mov'), os.path.join(d, 'moi', 'N0.mov')
    open(cu, 'wb').write(b'buoi 1')
    open(moi, 'wb').write(b'buoi 2')
    f = os.path.join(d, 'k.json')
    with open(f, 'w', encoding='utf-8') as fh:
        json.dump(kiem_r(cu, NEO_BOXES, idx=368), fh)
    ok, _ = nt.kiem_r_neo(neo, [f], moi, 368)            # LLC1-9 round 2: same file name, another clip -> refused
    assert not any(ok.values())
    ok, _ = nt.kiem_r_neo(neo, [f], cu, 500)             # same clip, another template frame -> refused
    assert not any(ok.values())
    ok, _ = nt.kiem_r_neo(neo, [f], cu, 368)
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


ARM_G = 0.05      # LLC1-19: lever arm of the test (round 3: the tolerance below is what separates psi' from raw yaw)


def test_g_q_with_a_tilted_plate_and_a6(files):
    """LLC1-19: tilt g_b != 0 (A6 removes the yaw bias, the centre shift is removed in 6b): q in the frame of psi' is
    still the 50 mm lever arm, i.e. the rotation centre = t - R(psi') q."""
    gb = [0.01, -0.006]
    (files / 'a6g.json').write_text(json.dumps(dict(g_b=gb, eps_g=1e-4)), encoding='utf-8')
    xoay = {'schema': 'xoay_v1', 'cho': {}}
    for p in ('goc', 'K1'):
        a, o = body_at(p, 0.0, 0.0)
        r_true = o + rot(a) @ CB_TRUE - rot(a + TH_TRUE) @ [ARM_G, 0.0]
        lam = H_CAM / (H_CAM - TAG_H)
        xoay['cho'][p] = {'r_floor_m': (C + (1.0 - TAG_H / ZC) * (lam * (r_true - C) + (1 - lam) * F_OFF)).tolist(),
                          'trang_thai': 'dat'}
    runs = [(lid, (pre[0], pre[1], pre[2], dict(g_b=gb)), (post[0], post[1], post[2], dict(g_b=gb)))
            for lid, pre, post in default_runs()]
    _, _, _, L = run(files, *build(runs=runs, plate=dict(g_b=gb)), a6='a6g.json', xoay=xoay)
    # LLC1-19 round 3: correct code 0.95 um off here, the raw yaw instead of psi' 10.3 um -> 3 um separates them
    assert np.allclose(L['L04']['q_m'], [ARM_G, 0.0], atol=3e-6)



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
    assert L['L01']['lan_lap_id'] != L['L02']['lan_lap_id'] and L['L02']['lan_lap_id'] is not None
    assert 'khong_co_cua_so_chuan' in L['L03']['ly_do']['beta']            # 12b of this placement only at goc
    assert 'tam_doi_so_chuan' in L['L04']['ly_do']['beta'] and L['L04']['chuan_12b'] is None
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



# ---------------------------------------------------------------- Codex code round 2 (LLC2-n, LLC1-n round 2, CT1-2)
def _strict(path):
    def bad(c):
        raise ValueError(f'non-standard JSON constant {c}')
    return json.loads(open(path, encoding='utf-8').read(), parse_constant=bad)


def test_llc2_1_ct1_2_phi_from_the_f7_of_the_standard_12b(files):
    """phi uses the truc_xe_v1 of the 12b run that gave the standard (thu_id_cap, frames.csv of its ref_12b row)."""
    _, hc, _, L = run(files, *build(), truc_xe=truc_xe_ok, tag='ok')
    x = L['L01']
    assert x['trang_thai']['phi'] == 'dat', x['ly_do']['phi']
    assert x['chuan_12b'] == '12b_goc' and x['dau']['chuan']['frames_sha256'] == 'fr_12b_goc'
    assert x['dau']['chuan']['phep_gop'] == 'trung_binh_go_vong'

    def khac_12b(hc):                     # the F7 of another 12b run at the same place
        o = truc_xe_ok(hc)
        for b in o:
            b['mien_hieu_luc']['thu_id_cap'] += '_lan2'
            b['nguon'][0]['vai_tro'] = b['mien_hieu_luc']['thu_id_cap'] + ':frames'
        return o

    def khac_frames(hc):                  # right attempt name, another frames.csv
        o = truc_xe_ok(hc)
        for b in o:
            b['nguon'][0]['sha256'] = 'khac'
        return o

    def chan_doan(hc):                    # HIEUCHUAN's diagnostic state is not a locked F7
        o = truc_xe_ok(hc)
        for b in o:
            b['trang_thai'] = 'chan_doan'
        return o
    for tag, f, why in (('a', khac_12b, 'thieu_truc_xe'), ('b', khac_frames, 'truc_xe_khac_cua_so_chuan'),
                        ('c', chan_doan, 'truc_xe_chua_dat')):
        _, _, _, L2 = run(files, *build(), truc_xe=f, tag=tag)
        y = L2['L01']
        assert y['trang_thai']['beta'] == 'dat' and y['trang_thai']['phi'] == 'chua_ket_luan', tag
        assert why in y['ly_do']['phi'], (tag, y['ly_do']['phi'])


def _no_anchor(recs_i):
    for r in recs_i:
        r['neo_ok'], r['Ph'] = False, None


def test_llc2_2_undetermined_link_keeps_no_standard(files):
    """Run L01 has no anchor frames at both ends; the plate is re-seated 2 mm off before L02 (no new 12b): L02, L03
    are checked against the standard itself and get no standard (J v6 2), never the old one."""
    sh = dict(cB=CB_TRUE + [0.002, 0.0])
    runs = [('L01', ('goc', 0.0, 0.0, {}), ('goc', 0.01, 0.2, {})),
            ('L02', ('goc', 0.0, 0.1, sh), ('goc', 0.01, 0.3, sh)),
            ('L03', ('goc', 0.0, 0.2, sh), ('goc', 0.01, 0.1, sh))]
    rows, recs = build(runs=runs)
    for i, w in enumerate(rows):
        if w['luot_id'] == 'L01':
            _no_anchor(recs[i])
    _, _, _, L = run(files, rows, recs)
    for lid, why in (('L02', 'tam_doi_so_chuan'), ('L03', 'lan_lap_sau_doi')):
        x = L[lid]
        assert x['lan_lap_id'] is None and x['chuan_12b'] is None, lid
        assert why in x['ly_do']['beta'], (lid, x['ly_do']['beta'])
        assert x['trang_thai']['beta'] == x['trang_thai']['delta_c_B'] == 'chua_ket_luan'


def test_llc2_2_failed_12b_is_still_a_new_placement(files):
    """A 12b run whose windows all fail (re-seated plate) still opens a placement; the next run has no standard."""
    runs = [('L01', ('goc', 0.0, 0.0, {}), ('goc', 0.01, 0.2, {})), ('L02', ('goc', 0.0, 0.1, {}), ('goc', 0.01, 0.3, {}))]
    rows, recs = build(runs=runs)
    r2, k2 = build(runs=[], fit_places=('goc',), e_places=())
    base = max(r['id'] for v in recs.values() for r in v) + 100
    extra = []
    for w, k in zip(r2, sorted(k2)):
        rr = [dict(r, id=r['id'] + base, t=(r['id'] + base) / 60.0, neo_ok=False, Ph=None) for r in k2[k]]
        extra.append((win('12b_goc_2', 'goc', w['loai'], rr), rr))
    fit = [(w, recs[i]) for i, w in enumerate(rows) if w['loai'] in nt.LOAI_KHOP]
    runw = {lid: [(w, recs[i]) for i, w in enumerate(rows) if w['luot_id'] == lid] for lid in ('L01', 'L02')}
    order = fit + runw['L01'] + extra + runw['L02']
    rows2, recs2 = [], {}
    for j, (w, rr) in enumerate(order):
        recs2[j] = rr
        rows2.append(dict(w, clip_sha256=f'c{j}'))
    _, hc, _, L = run(files, rows2, recs2)
    assert 'goc|12b_goc_2' not in hc['chuan_12b']
    assert L['L01']['chuan_12b'] == '12b_goc' and L['L01']['lan_lap_id'] == 'L0'
    x = L['L02']                                          # the old 12b_goc is NOT used after the failed 12b
    assert x['chuan_12b'] is None and x['trang_thai']['beta'] == 'chua_ket_luan' and x['lan_lap_id'] is None
    assert '12b_cung_lan_lap_khong_hop_le' in x['ly_do']['beta'], x['ly_do']['beta']


def test_llc2_4_comparison_needs_the_anchor_gate(files):
    """One anchor of the pre end 0.8 mm off (residual above r_max): the plate check is khong_xac_dinh, not doi."""
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L04' and w['loai'] == 'pre')
    for r in recs[i]:
        r['Ph'] = r['Ph'] + np.array([[0.0008, 0.0], [0.0, 0.0], [0.0, 0.0]])
    _, _, _, L = run(files, rows, recs)
    so = L['L04']['so_sanh_trong_luot']
    assert all(v['trang_thai'] == 'khong_xac_dinh' for v in so.values()), so
    assert L['L04']['tam_doi_trong_luot'] is None


def test_llc2_5_anchor_term_of_each_standard_target_pair(files):
    """A target end seen from another camera height (not in any held-out pair): its own anchor term enters its u_beta."""
    runs = default_runs() + [('L07', ('goc', 0.0, 0.0, dict(H=3.0)), None)]
    _, hc, ll, L = run(files, *build(runs=runs))
    ctx = _ctx()
    G, h, r_max = np.array(hc['G']), np.array(hc['h']), hc['r_max_m']
    cs = next(c for c in hc['cua_so'] if c['meta']['luot'] == '12b_goc' and c['meta']['loai'] == 'ref_12b')
    eps_s = max(hc['eps_neo_deg'], nt.eps_goc(nt.eps_neo_bang(G, h, np.array(cs['F']), cs['H'], C, ctx.a_n), r_max))
    eps_d = max(hc['eps_neo_deg'], nt.eps_goc(nt.eps_neo_bang(G, h, C + F_OFF, 3.0, C, ctx.a_n), r_max))
    x = L['L07']
    assert x['eps_neo_luot_deg'] == pytest.approx(max(ll['eps_cap_deg'], eps_s + eps_d), rel=1e-9)
    assert x['u_beta_deg'] == pytest.approx(ll['u_beta_deg'] - ll['eps_cap_deg'] + x['eps_neo_luot_deg'], rel=1e-9)
    assert eps_s + eps_d > ll['eps_cap_deg'] and x['u_beta_deg'] > ll['u_beta_deg']


def test_llc2_6_strict_json_when_the_rank_is_missing(files, monkeypatch):
    orig = nt.hang

    def fake(fit, cfg):
        o = orig(fit, cfg)
        k = len(o['sigma_h'])
        return dict(o, sigma_h=[math.inf] * k, cov_h=[[math.inf] * k for _ in range(k)], suy_bien=True)
    monkeypatch.setattr(nt, 'hang', fake)
    _, hc, ll, L = run(files, *build())
    hc_s = _strict(files / 'hc_a' / 'hieu_chuan_neo.json')
    _strict(files / 'll_a' / 'lan_lap.json')
    assert hc_s['sigma_h'] == [None] * len(hc_s['sigma_h']) and hc_s['trang_thai'] == 'chua_ket_luan'
    assert all('thieu_hang' in x['ly_do']['beta'] for x in L.values())


def test_llc2_7_dich_key_kept(files):
    _, _, _, L = run(files, *build())
    assert L['L01']['dau']['dich'] == L['L01']['dau']['pre']


def test_llc1_1_round2_sum_of_finite_terms_overflows(files):
    def tx(hc):
        o = truc_xe_ok(hc)
        for b in o:
            b['u_phi_hat_chuan_k_deg'] = 1e308
        return o
    _, _, _, L = run(files, *build(), truc_xe=tx, u_beta_cam=1e308)
    x = L['L01']
    assert x['trang_thai']['phi'] == 'chua_ket_luan' and 'u_phi_khong_huu_han' in x['ly_do']['phi']
    assert x['u_phi_truoc_le_deg'] is None


def test_llc1_2_round2_anchor_gate_on_the_common_set(files):
    """Anchor A +0.49 mm on the tag frames, -0.49 mm on the others: the anchor-set mean passes, the common set (used
    for the position) does not -> its summary is dropped, no delta_c_B dat."""
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L04' and w['loai'] == 'pre')
    for k, r in enumerate(recs[i]):
        r['Ph'] = r['Ph'] + np.array([[0.00049 if k < 6 else -0.00049, 0.0], [0.0, 0.0], [0.0, 0.0]])
    rows[i]['khung_ma_hop_le'] = [r['id'] for r in recs[i][:6]]
    _, _, _, L = run(files, rows, recs, cfg=dict(min_khung=5))
    x = L['L04']
    assert 'du_neo_tap_chung_vuot_r_max' in x['dau']['pre']['ly_do'], x['dau']['pre']['ly_do']
    assert x['dau']['pre']['khung_chung']['id'] == [] and x['dau']['pre']['khung_chung_sha256'] is None   # LLC3-3
    assert x['trang_thai']['delta_c_B'] == 'chua_ket_luan' and x['delta_c_B_m'] is None


def test_llc1_4_round2_group_at_the_standard_place_only(files):
    rows, recs = build()
    base = max(r['id'] for v in recs.values() for r in v) + 100
    for j in range(2):                                     # two 12b windows of run 12b_goc recorded at K1
        a, o = body_at('K1', 0.1 * j, 0.0)
        rr = recs_for(a, o, n=12, seed=900 + j, id0=base + 20 * j)
        recs[len(rows)] = rr
        rows.append(win('12b_goc', 'K1', 'dung_12b', rr))
    _, hc, _, _ = run(files, rows, recs)
    assert hc['cho']['goc']['n_12b_cung_lan_lap'] == 7


def test_llc1_6_round2_template_and_window_file_locked(files):
    run(files, *build(), tag='lk2')
    p = files / 'hc_lk2' / 'hieu_chuan_neo.json'
    args = (str(p), str(files / 'cs_lk2.json'), str(files / 'session.json'))
    (files / 'mau.mov').write_bytes(b'changed at the locked path')
    with pytest.raises(ValueError, match='template clip'):
        nt.run_lan_lap(*args, str(files / 'll_m'), str(files / 'a6.json'), 'T1', cfg=CFG, a6_mod=FakeA6Mod, _records={})
    (files / 'mau.mov').write_bytes(b'x')
    cs = files / 'cs_lk2.json'
    cs.write_text(cs.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='window file'):
        nt.run_lan_lap(*args, str(files / 'll_f'), str(files / 'a6.json'), 'T1', cfg=CFG, a6_mod=FakeA6Mod, _records={})


def test_llc1_7_round2_overlap_and_missing_stillness(files):
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L01' and w['loai'] == 'pre')
    bad = [dict(w) for w in rows] + [dict(rows[i], luot_id='L99', khung=rows[i]['khung'][1:])]
    with pytest.raises(ValueError, match='same type'):
        nt.kiem_f2(bad)
    rows2, recs2 = build()
    del rows2[i]['ket_luan']
    _, _, ll, _ = run(files, rows2, recs2, tag='kl2')
    assert any('thieu_ket_luan_dung_yen' in x['ly_do'] for x in ll['cua_so_loi'])


def test_llc1_12_round2_place_offset_uses_the_ray_arm():
    """Same apparent tag centre, refitted camera centres 41.3 mm apart, Dh about 4 mm: the place-offset term is
    |Dh| |(t_d - F_d) - (t_s - F_s)| / (H - h) (about 0.067 mm), not 0."""
    ctx, g = _ctx(), np.zeros(2)
    s0 = ws('goc', 0.0, 0.0)
    d1 = dict(s0, F=np.asarray(s0['F']) + [0.0413, 0.0], s=s0['s'] * 1.0016)
    dh = nt.delta_h_uoc(s0, d1, ctx, g)
    assert 0.003 < abs(dh) < 0.005
    _, parts, _ = nt.u_delta_c_B(s0, d1, ctx, g, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert parts['lech_cho'] == pytest.approx(abs(dh) * 0.0413 / (H_CAM - TAG_H), rel=1e-9)
    assert parts['lech_cho'] > 4e-5


def test_llc1_15_round2_post_in_the_order_check(files):
    rows, recs = build()
    idx = {(w['luot_id'], w['loai']): i for i, w in enumerate(rows)}
    for (lid, lo), t0 in ((('L01', 'pre'), 1.0), (('L01', 'post'), 4.0), (('L02', 'pre'), 3.0), (('L02', 'post'), 5.0)):
        i = idx[(lid, lo)]
        rows[i]['clip_sha256'] = 'same'
        for k, r in enumerate(recs[i]):
            r['t'] = t0 + k / 600.0
        rows[i]['khung'] = [{'id': r['id'], 'pts_s': r['t']} for r in recs[i]]
    _, _, _, L = run(files, rows, recs)
    assert all(x['lan_lap_id'] is None and x['lan_lap_id_ly_do'] == 'thu_tu_mau_thuan' for x in L.values())


def test_llc1_17_round2_e_ends_of_one_clip(files):
    xoay = {'schema': 'xoay_v1', 'cho': {'goc': {'r_floor_m': [0.0, 0.0], 'trang_thai': 'dat'}}}
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['cho'] == 'goc' and w['loai'] == 'E_cuoi')
    rows[i]['clip_sha256'] = 'another_E_clip'
    _, _, _, L = run(files, rows, recs, xoay=xoay)
    assert 'hai_dau_E_khac_clip' in L['L04']['ly_do']['q'] and L['L04']['q_m'] is None


def test_llc2_2_consecutive_12b_at_one_place_compared(files):
    """12b at K2, the plate re-seated 2 mm off, 12b at K2 again right after: a new group (the same place compared),
    so a later run at goc (re-seated plate) has no standard of this placement, not the old goc one."""
    sh = dict(cB=CB_TRUE + [0.002, 0.0])
    runs = [('L01', ('K2', 0.0, 0.1, sh), ('K2', 0.01, 0.2, sh)), ('L02', ('goc', 0.0, 0.1, sh), ('goc', 0.01, 0.2, sh))]
    rows, recs = build(runs=runs)
    r2, k2 = build(runs=[], fit_places=('K2',), e_places=(), plate=sh)
    base = max(r['id'] for v in recs.values() for r in v) + 100
    extra = []
    for w, k in zip(r2, sorted(k2)):
        rr = [dict(r, id=r['id'] + base, t=(r['id'] + base) / 60.0) for r in k2[k]]
        extra.append((win('12b_K2_2', 'K2', w['loai'], rr), rr))
    fit = [(w, recs[i]) for i, w in enumerate(rows) if w['loai'] in nt.LOAI_KHOP]
    runw = [(w, recs[i]) for i, w in enumerate(rows) if w['loai'] in nt.LOAI_GIU]
    order = fit + extra + runw
    rows2, recs2 = [], {}
    for j, (w, rr) in enumerate(order):
        recs2[j] = rr
        rows2.append(dict(w, clip_sha256=f'c{j}'))
    _, _, _, L = run(files, rows2, recs2)
    assert L['L01']['chuan_12b'] == '12b_K2_2' and L['L01']['doi_so_chuan'] is False
    assert 'khong_co_cua_so_chuan' in L['L02']['ly_do']['beta'], L['L02']['ly_do']['beta']


def test_llc2_4_comparison_needs_the_anchor_gate_of_the_whole_anchor_set(files):
    """Anchor A +0.98 mm on the frames WITHOUT tag only: the tag (common) frames are clean, but the anchor-set mean is
    0.49 mm off (residual above r_max): the end fails the anchor gate, so the plate check of the run is khong_xac_dinh."""
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L04' and w['loai'] == 'pre')
    for k, r in enumerate(recs[i]):
        if k >= 6:
            r['Ph'] = r['Ph'] + np.array([[0.00098, 0.0], [0.0, 0.0], [0.0, 0.0]])
    rows[i]['khung_ma_hop_le'] = [r['id'] for r in recs[i][:6]]
    _, _, _, L = run(files, rows, recs, cfg=dict(min_khung=5))
    so = L['L04']['so_sanh_trong_luot']
    assert 'du_neo_vuot_r_max' in L['L04']['dau']['pre']['ly_do']
    assert all(v['trang_thai'] == 'khong_xac_dinh' for v in so.values()), so


def test_sha_ids_is_the_common_form():
    """CODE 19/11: one form of the ID-list sha256 for every session: sorted IDs joined by commas, no brackets."""
    import hashlib
    assert nt.sha_ids([12, 3, 7]) == hashlib.sha256(b'3,7,12').hexdigest()
    assert nt.sha_ids([]) == hashlib.sha256(b'').hexdigest()


# ---------------------------------------------------------------- v4 13: lan_lap --f15 -> beta_cam_replay
def _f15(ll, cases, path):
    """c7_meo_dau_chan_bang_v1 as KIEMCHEO meo_bang writes it (941d779), from a first lan_lap.json; cases(cho, lid,
    dau, k_anchor) -> {case: [dx, dy]}."""
    import hashlib
    ten = ll['neo_ten']
    theo = {}
    for r in ll['luot']:
        for dau, x in sorted((r.get('dau') or {}).items()):
            if dau == 'dich' or not x or x.get('Ph_neo_m') is None:
                continue
            cho = x.get('cho') or r.get('cho')
            t = theo.setdefault(cho, dict(ca=[], ca_hong=[], muc=[]))
            t['muc'].append(dict(luot_id=r['luot_id'], dau=dau, clip_sha256=x['clip_sha256'],
                                 khung_neo_sha256=x['khung_neo_sha256'],
                                 neo={n: cases(cho, r['luot_id'], dau, k) for k, n in enumerate(ten)}))
    o = dict(schema='c7_meo_dau_chan_bang_v1', chieu='held_minus_full', bo_H32=True, truoc_le=True, neo_ten=ten,
             h_n_m=H_N, kappa_n=H_N / ZC, lan_lap=dict(file='x', sha256='y', hieu_chuan_sha256=ll['hieu_chuan']['sha256']),
             theo_cho=theo)
    o['sha256'] = hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(o), encoding='utf-8')
    return o


def _again(files, rows, recs, tag, f15):
    hold = {i: recs[i] for i, w in enumerate(rows) if w['loai'] in nt.LOAI_GIU}
    d = files
    nt.run_lan_lap(str(d / f'hc_{tag}' / 'hieu_chuan_neo.json'), str(d / f'cs_{tag}.json'), str(d / 'session.json'),
                   str(d / f'll_{tag}_f15_{f15.stem}'), str(d / 'a6.json'), 'T1', u_beta_cam_deg=0.01, cfg=CFG,
                   a6_mod=FakeA6Mod, _records=hold, f15=str(f15))
    return json.load(open(d / f'll_{tag}_f15_{f15.stem}' / 'lan_lap.json', encoding='utf-8'))


def test_v4_13_f15_beta_cam_replay(files):
    rows, recs = build()
    _, hc, ll, L = run(files, rows, recs, tag='f')
    assert 'beta_cam_replay' not in ll                                          # (5) no --f15: no block

    def cases(cho, lid, dau, k):
        return {'dong': [1e-4, -5e-5], 'A_pre': [2e-4, 0.0] if (k == 0 and dau == 'pre') else [0.0, 0.0],
                'B_chuan': [0.0, 3e-4] if (k == 1 and dau == 'chuan') else [0.0, 0.0]}
    _f15(ll, cases, files / 'f15a.json')
    ll2 = _again(files, rows, recs, 'f', files / 'f15a.json')
    b = ll2['beta_cam_replay']
    # (2) expected from a direct solve of the pre ends at goc (the standard end does not move in case A_pre)
    ctx = _ctx()
    G, h = np.array(hc['G']), np.array(hc['h'])
    exp = 0.0
    for r in ll['luot']:
        if r['cho'] == 'goc' and r['beta_deg'] is not None:
            for dau, dP in (('pre', [[2e-4, 0.0], [0, 0], [0, 0]]), ('chuan', [[0, 0], [0.0, 3e-4], [0, 0]])):
                P = np.array(r['dau'][dau]['Ph_neo_m'])
                a0 = nt.tu_the(G, h, P, C + F_OFF, H_CAM, C, ctx.a_n)[0]
                a1 = nt.tu_the(G, h, P + dP, C + F_OFF, H_CAM, C, ctx.a_n)[0]
                exp = max(exp, abs(nt.wrap(a1 - a0)))
    assert exp > 1e-4
    assert b['goc']['trang_thai'] == 'dat' and b['goc']['max_abs_dbeta_rad'] == pytest.approx(exp, rel=1e-9)
    assert b['goc']['f15_sha256'] == cm.sha256(str(files / 'f15a.json'))
    assert b['goc']['n'] == 3 * sum(1 for r in ll['luot'] if r['cho'] == 'goc' and r['beta_deg'] is not None)

    # (1) one shift of every anchor at both ends: beta does not move (to the lift nonlinearity)
    def dong(cho, lid, dau, k):
        return {'dong': [1e-4, -5e-5]}
    _f15(ll, dong, files / 'f15b.json')
    b2 = _again(files, rows, recs, 'f', files / 'f15b.json')['beta_cam_replay']
    assert all(v['max_abs_dbeta_rad'] < 1e-5 for v in b2.values() if v['n'])

    # (4) one end of another frame set -> that place is not concluded
    o = _f15(ll, dong, files / 'f15c.json')
    o['theo_cho']['goc']['muc'][0]['khung_neo_sha256'] = 'khac'
    o.pop('sha256')
    import hashlib
    o['sha256'] = hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    (files / 'f15c.json').write_text(json.dumps(o), encoding='utf-8')
    b3 = _again(files, rows, recs, 'f', files / 'f15c.json')['beta_cam_replay']
    assert b3['goc']['trang_thai'] == 'chua_ket_luan' and any('sai_nguon' in x for x in b3['goc']['ly_do'])

    # (3) F15 made from a lan_lap of another hieu_chuan -> error (code 2)
    o = _f15(ll, dong, files / 'f15d.json')
    o['lan_lap']['hieu_chuan_sha256'] = 'khac'
    o.pop('sha256')
    o['sha256'] = hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    (files / 'f15d.json').write_text(json.dumps(o), encoding='utf-8')
    with pytest.raises(ValueError, match='hieu_chuan'):
        _again(files, rows, recs, 'f', files / 'f15d.json')


# ---------------------------------------------------------------- Codex round 3 (LLC3-n, LLC2-1 / LLC1 rest, F15C-n)
def test_llc3_1_doi_inside_a_run_breaks_the_group(files):
    """L01: pre as the standard, post 2 mm off (plate moved in the run); the plate is re-seated back before L02 with
    no new 12b: L02 matches the standard again but gets no standard."""
    sh = dict(cB=CB_TRUE + [0.002, 0.0])
    runs = [('L01', ('goc', 0.0, 0.0, {}), ('goc', 0.01, 0.2, sh)), ('L02', ('goc', 0.0, 0.1, {}), ('goc', 0.01, 0.3, {}))]
    _, _, _, L = run(files, *build(runs=runs))
    assert L['L01']['tam_doi_trong_luot'] is True and L['L01']['chuan_12b'] == '12b_goc'
    assert L['L02']['doi_so_chuan'] is False and L['L02']['chuan_12b'] is None
    assert 'lan_lap_sau_doi' in L['L02']['ly_do']['beta'] and L['L02']['lan_lap_id'] is None


def test_llc2_1_round3_f7_of_another_f6_or_unlocked(files):
    def khac_f6(hc):
        o = truc_xe_ok(hc)
        for b in o:
            b['nguon'].append({'vai_tro': 'hieu_chuan_neo', 'sha256': 'khac'})
        return o

    def chua_khoa(hc):                    # LLC3-2: HIEUCHUAN writes null angles when not locked
        o = truc_xe_ok(hc)
        for b in o:
            b.update(trang_thai='chua_ket_luan', a_k_deg=None, phi_hat_chuan_k_deg=None, u_phi_hat_chuan_k_deg=None)
        return o
    for tag, f, why in (('f6', khac_f6, 'truc_xe_khac_f6_hoac_a6'), ('ck', chua_khoa, 'truc_xe_chua_dat')):
        _, _, _, L = run(files, *build(), truc_xe=f, tag=tag)
        assert why in L['L01']['ly_do']['phi'] and L['L01']['trang_thai']['phi'] == 'chua_ket_luan', tag
        assert L['L01']['trang_thai']['beta'] == 'dat', tag


def test_llc1_7_round3_frame_list_hash_checked():
    rows, _ = build()
    ok = [dict(w, khung_sha256=nt.sha_ids([k['id'] for k in w['khung']])) for w in rows]
    nt.kiem_f2(ok)
    bad = [dict(w) for w in ok]
    bad[3] = dict(bad[3], khung_sha256=nt.sha_ids([k['id'] + 1 for k in bad[3]['khung']]))
    with pytest.raises(ValueError, match='khung_sha256'):
        nt.kiem_f2(bad)


def test_llc1_12_round3_ratio_below_one_branch():
    """s_s / s_d < 1: the index uses max(ratio, 1) (never below the design's (H - h) x spread)."""
    ctx, g = _ctx(), np.zeros(2)
    s0 = ws('K1', 0.0, 0.0)
    d1 = dict(s0, s=s0['s'] / 0.9985)                                 # s_s / s_d = 0.9985
    _, _, u_dh = nt.u_delta_c_B(s0, d1, ctx, g, 0.0, 0.0, 0.0008, 0.0, 0.0)
    K = H_CAM * (H_CAM - TAG_H) / H_CAM
    assert u_dh == pytest.approx(K * 0.0008, rel=1e-9) and u_dh > K * 0.9985 * 0.0008


def test_llc1_15_round3_raw_row_order(files):
    """F2 rows of one clip: the post row (4 s) written before the pre row (1 s) -> order contradicted."""
    rows, recs = build()
    i_pre = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L01' and w['loai'] == 'pre')
    i_post = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L01' and w['loai'] == 'post')
    for i, t0 in ((i_pre, 1.0), (i_post, 4.0)):
        rows[i]['clip_sha256'] = 'same'
        for k, r in enumerate(recs[i]):
            r['t'] = t0 + k / 600.0
        rows[i]['khung'] = [{'id': r['id'], 'pts_s': r['t']} for r in recs[i]]
    rows[i_pre], rows[i_post] = rows[i_post], rows[i_pre]
    recs[i_pre], recs[i_post] = recs[i_post], recs[i_pre]
    _, _, _, L = run(files, rows, recs)
    assert all(x['lan_lap_id_ly_do'] == 'thu_tu_mau_thuan' for x in L.values())


def _f15_hash(o):
    import hashlib
    o.pop('sha256', None)
    o['sha256'] = hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return o


def test_f15c_round1_checks(files):
    rows, recs = build()
    _, hc, ll, L = run(files, rows, recs, tag='g')
    ctx = _ctx()
    G, h = np.array(hc['G']), np.array(hc['h'])

    # F15C-6: one case with BOTH ends moved: d beta = wrap(d alpha_pre - d alpha_standard), n = verified runs
    def ab(cho, lid, dau, k):
        if k == 0 and dau == 'pre':
            return {'AB': [2e-4, 0.0]}
        if k == 1 and dau == 'chuan':
            return {'AB': [0.0, 3e-4]}
        return {'AB': [0.0, 0.0]}
    _f15(ll, ab, files / 'g1.json')
    b = _again(files, rows, recs, 'g', files / 'g1.json')['beta_cam_replay']['goc']
    exp, n = 0.0, 0
    for r in ll['luot']:
        if r['cho'] == 'goc' and r['chuan_12b'] is not None:
            d = []
            for dau, dP in (('chuan', [[0, 0], [0.0, 3e-4], [0, 0]]), ('pre', [[2e-4, 0.0], [0, 0], [0, 0]])):
                P = np.array(r['dau'][dau]['Ph_neo_m'])
                d.append(nt.wrap(nt.tu_the(G, h, P + dP, C + F_OFF, H_CAM, C, ctx.a_n)[0]
                                 - nt.tu_the(G, h, P, C + F_OFF, H_CAM, C, ctx.a_n)[0]))
            exp, n = max(exp, abs(nt.wrap(d[1] - d[0]))), n + 1
    assert b['trang_thai'] == 'dat' and b['n'] == n >= 1 and b['max_abs_dbeta_rad'] == pytest.approx(exp, rel=1e-9)

    def dong(cho, lid, dau, k):
        return {'c': [1e-4, 0.0]}
    # F15C-1: a duplicated entry (the first of another frame set) -> not dat
    o = _f15(ll, dong, files / 'g2.json')
    m0 = next(m for m in o['theo_cho']['goc']['muc'] if m['dau'] == 'pre')
    o['theo_cho']['goc']['muc'].insert(0, dict(m0, khung_neo_sha256='khac'))
    (files / 'g2.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'g', files / 'g2.json')['beta_cam_replay']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and any('muc_trung' in x for x in b['ly_do'])
    # F15C-2: not the held-out cases / already with the margin -> error
    for k in ('bo_H32', 'truoc_le'):
        o = _f15(ll, dong, files / f'g3{k}.json')
        o[k] = False
        (files / f'g3{k}.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
        with pytest.raises(ValueError, match='bo_H32'):
            _again(files, rows, recs, 'g', files / f'g3{k}.json')
    # F15C-3: a case vector with one component / a case missing at one anchor -> not dat, no broadcasting
    o = _f15(ll, dong, files / 'g4.json')
    m0 = next(m for m in o['theo_cho']['goc']['muc'] if m['dau'] == 'pre')
    m0['neo']['A']['c'] = [2e-4]
    (files / 'g4.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'g', files / 'g4.json')['beta_cam_replay']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and any('vec_sai' in x for x in b['ly_do'])
    o = _f15(ll, dong, files / 'g5.json')
    m0 = next(m for m in o['theo_cho']['goc']['muc'] if m['dau'] == 'pre')
    del m0['neo']['B']['c']
    (files / 'g5.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'g', files / 'g5.json')['beta_cam_replay']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and any('ca_thieu' in x for x in b['ly_do'])
    # F15C-5: a place with runs to replay but no block -> listed, not dat
    o = _f15(ll, dong, files / 'g6.json')
    del o['theo_cho']['K1']
    (files / 'g6.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'g', files / 'g6.json')['beta_cam_replay']
    assert b['K1']['trang_thai'] == 'chua_ket_luan' and 'thieu_khoi_cho' in b['K1']['ly_do']


def test_f15c_4_only_runs_with_a_verified_standard(files):
    sh = dict(cB=CB_TRUE + [0.002, 0.0])
    runs = default_runs()
    runs[3] = ('L04', ('goc', 0.0, 0.0, sh), ('goc', 0.0, 0.0, sh))          # off the standard: beta only diagnostic
    rows, recs = build(runs=runs)
    _, _, ll, L = run(files, rows, recs, tag='v')
    assert L['L04']['beta_deg'] is not None and L['L04']['chuan_12b'] is None

    def dong(cho, lid, dau, k):
        return {'c': [1e-4, 0.0]}
    _f15(ll, dong, files / 'v1.json')
    b = _again(files, rows, recs, 'v', files / 'v1.json')['beta_cam_replay']['goc']
    assert b['n'] == sum(1 for r in ll['luot'] if r['cho'] == 'goc' and r['chuan_12b'] is not None)


def test_f15c_5_source_checked_without_the_anchor_shape(files):
    (files / 'kiem.json').write_text(json.dumps(kiem_r(files / 'mau.mov', NEO_BOXES[:2])), encoding='utf-8')
    rows, recs = build()
    _, hc, ll, L = run(files, rows, recs, tag='ng')
    o = dict(schema='c7_meo_dau_chan_bang_v1', chieu='held_minus_full', bo_H32=True, truoc_le=True,
             neo_ten=ll['neo_ten'], kappa_n=H_N / ZC, lan_lap=dict(hieu_chuan_sha256='khac'), theo_cho={})
    (files / 'n1.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    with pytest.raises(ValueError, match='hieu_chuan'):
        _again(files, rows, recs, 'ng', files / 'n1.json')
    o['lan_lap']['hieu_chuan_sha256'] = ll['hieu_chuan']['sha256']
    (files / 'n2.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'ng', files / 'n2.json')['beta_cam_replay']
    assert b and all(v['trang_thai'] == 'chua_ket_luan' for v in b.values())


def test_ct4_3_empty_or_null_window_keeps_the_run(files):
    """CODE CT4-3: CAM writes post.khung = [] (clip too short; null in old files): the window fails, the run is still in
    F8; the placement uses the pre; a pre without frames -> lan_lap_khong_xac_dinh."""
    rows, recs = build()
    idx = {(w['luot_id'], w['loai']): i for i, w in enumerate(rows)}
    for key, val in ((('L02', 'post'), []), (('L03', 'post'), None), (('L05', 'pre'), [])):
        i = idx[key]
        rows[i]['khung'], rows[i]['khung_ma_hop_le'], recs[i] = val, [], []
    nt.kiem_f2(rows)
    _, _, ll, L = run(files, rows, recs)
    assert sorted(L) == [f'L{j:02d}' for j in range(1, 7)]
    assert L['L02']['chuan_12b'] is not None and L['L02']['beta_deg'] is not None        # placement from the pre
    assert L['L02']['tam_doi_trong_luot'] is None
    assert L['L05']['lan_lap_id'] is None and 'lan_lap_khong_xac_dinh' in L['L05']['ly_do']['beta']
    bad = {(x['meta']['luot'], x['meta']['loai']) for x in ll['cua_so_loi']}
    assert {('L02', 'post'), ('L03', 'post'), ('L05', 'pre')} <= bad


def test_f15_diagnostic_state_is_not_dat(files):
    """CODE 19/11 (KIEMCHEO 02dea17): an F15 (b) in state chan_doan is not locked -> every place chua_ket_luan."""
    rows, recs = build()
    _, _, ll, _ = run(files, rows, recs, tag='cd')

    def dong(cho, lid, dau, k):
        return {'c': [1e-4, 0.0]}
    o = _f15(ll, dong, files / 'cd1.json')
    o['trang_thai'] = 'chan_doan'
    (files / 'cd1.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'cd', files / 'cd1.json')['beta_cam_replay']
    assert b and all(v['trang_thai'] == 'chua_ket_luan' and 'f15_chan_doan' in v['ly_do'] for v in b.values())


# ---------------------------------------------------------------- Codex round 4 (LLC3-1/3, LLC1-7, LLC2-1, LLC4-1, F15C-3/6, F15C2-1)
def test_llc3_1_round4_doi_at_a_place_without_standard_breaks_the_group(files):
    """No standard at K1 in the group; L01 at K1 has its plate moved 2 mm inside the run (doi): L02 at goc, back on
    the standard, still gets no standard; L03 at goc with an empty pre -> lan_lap_khong_xac_dinh (LLC4-1, CT4-3)."""
    sh = dict(cB=CB_TRUE + [0.002, 0.0])
    runs = [('L01', ('K1', 0.0, 0.0, {}), ('K1', 0.01, 0.2, sh)), ('L02', ('goc', 0.0, 0.1, {}), ('goc', 0.01, 0.3, {})),
            ('L03', ('goc', 0.0, 0.2, {}), ('goc', 0.01, 0.1, {}))]
    rows, recs = build(runs=runs, fit_places=('goc', 'K2'), e_places=('goc',))
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L03' and w['loai'] == 'pre')
    rows[i]['khung'], rows[i]['khung_ma_hop_le'], recs[i] = [], [], []
    _, _, _, L = run(files, rows, recs)
    assert L['L01']['tam_doi_trong_luot'] is True and 'khong_co_cua_so_chuan' in L['L01']['ly_do']['beta']
    assert L['L02']['chuan_12b'] is None and 'lan_lap_sau_doi' in L['L02']['ly_do']['beta']
    assert L['L03']['lan_lap_id_ly_do'] == 'lan_lap_khong_xac_dinh'


def test_llc3_3_round4_common_set_of_a_failed_window_not_exported(files):
    rows, recs = build()
    i = next(i for i, w in enumerate(rows) if w['luot_id'] == 'L04' and w['loai'] == 'pre')
    rows[i]['ket_luan'] = {'dung_yen': False}
    _, _, _, L = run(files, rows, recs)
    x = L['L04']['dau']['pre']
    assert x['khung_chung']['id'] == [] and x['khung_chung_sha256'] is None
    assert len(x['khung_giao_ma_neo']['id']) == 12                    # the tag-anchor intersection is kept (J 2)


def test_llc2_1_round4_f7_of_another_a6(files):
    def khac_a6(hc):
        o = truc_xe_ok(hc)
        for b in o:
            b['nguon'].append({'vai_tro': 'c8b_a6_v2', 'sha256': 'khac'})
        return o
    _, _, _, L = run(files, *build(), truc_xe=khac_a6, tag='a6x')
    assert 'truc_xe_khac_f6_hoac_a6' in L['L01']['ly_do']['phi'] and L['L01']['trang_thai']['phi'] == 'chua_ket_luan'


def test_llc1_7_round4_tag_list_hash_on_the_raw_list():
    rows, _ = build()
    ok = [dict(w, khung_ma_hop_le_sha256=nt.sha_ids(w['khung_ma_hop_le'])) for w in rows]
    nt.kiem_f2(ok)
    bad = [dict(w) for w in ok]
    lst = bad[3]['khung_ma_hop_le'] + [bad[3]['khung_ma_hop_le'][0]]          # one ID twice, hash of the set
    bad[3] = dict(bad[3], khung_ma_hop_le=lst, khung_ma_hop_le_sha256=nt.sha_ids(sorted(set(lst))))
    with pytest.raises(ValueError, match='khung_ma_hop_le_sha256'):
        nt.kiem_f2(bad)


def test_f15c_round2_declared_cases_structure_and_missing_entry(files):
    rows, recs = build()
    _, _, ll, _ = run(files, rows, recs, tag='h')

    def dong(cho, lid, dau, k):
        return {'c': [1e-4, 0.0]}
    # F15C-3: a declared case missing at the run ends (its vectors deleted) -> ca_thieu, not dat
    o = _f15(ll, dong, files / 'h1.json')
    o['theo_cho']['goc']['ca'] = [{'ten': 'c', 'bo_ids': [1]}, {'ten': 'lon', 'bo_ids': [2]}]
    (files / 'h1.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'h', files / 'h1.json')['beta_cam_replay']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and any('lon:ca_thieu' in x for x in b['ly_do'])
    # ca_hong written as dicts by KIEMCHEO held_cases: the case is excluded, no TypeError
    o = _f15(ll, dong, files / 'h2.json')
    o['theo_cho']['goc']['ca'] = [{'ten': 'c', 'bo_ids': [1]}, {'ten': 'lon', 'bo_ids': [2]}]
    o['theo_cho']['goc']['ca_hong'] = [{'ten': 'lon', 'ly_do': 'it_moc'}]
    (files / 'h2.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'h', files / 'h2.json')['beta_cam_replay']['goc']
    assert b['trang_thai'] == 'dat', b['ly_do']
    full = b['n']
    # F15C2-1: neo[t] not a {case: vector} mapping -> listed, not dat, no exception
    o = _f15(ll, dong, files / 'h3.json')
    m0 = next(m for m in o['theo_cho']['goc']['muc'] if m['dau'] == 'pre')
    m0['neo']['B'] = [0.0, 0.0]
    (files / 'h3.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'h', files / 'h3.json')['beta_cam_replay']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and any('neo_sai_cau_truc' in x for x in b['ly_do'])
    # F15C-6: one used entry removed (L01/pre) -> thieu_muc, n one run less, not dat
    o = _f15(ll, dong, files / 'h4.json')
    o['theo_cho']['goc']['muc'] = [m for m in o['theo_cho']['goc']['muc'] if (m['luot_id'], m['dau']) != ('L01', 'pre')]
    (files / 'h4.json').write_text(json.dumps(_f15_hash(o)), encoding='utf-8')
    b = _again(files, rows, recs, 'h', files / 'h4.json')['beta_cam_replay']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and 'L01:pre:thieu_muc' in b['ly_do'] and b['n'] == full - 1
