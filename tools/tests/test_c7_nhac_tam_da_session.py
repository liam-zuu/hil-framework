"""v4 14 (CODE 20/11, chain with several camera sessions): hieu_chuan / lan_lap with several --session files, the
camera of each window by its F2 session_sha256, the cross-session index u_khung_giua_session (file of CAM)."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import c7_measure as cm  # noqa: E402
import c7_nhac_tam as nt  # noqa: E402
import test_c7_nhac_tam_lan_lap as T  # noqa: E402

files = T.files


def _ses2(d, **chg):
    s = json.loads((d / 'session.json').read_text(encoding='utf-8'))
    s['rms_px_H'] = 0.31                                   # another camera session, same geometry (L125)
    s.update(chg)
    p = d / ('session2.json' if not chg else 'session_bad.json')
    p.write_text(json.dumps(s), encoding='utf-8')
    return p


def run_ds(files, rows, recs, tag, s2_of, du=None, ses_ll=None, theo_cho=None):
    """As T.run with two sessions: s2_of(i, w) -> window in session 2; du: None (no CAM file) or du_m of the pair."""
    d = files
    s1, s2 = d / 'session.json', _ses2(d)
    sha1, sha2 = cm.sha256(str(s1)), cm.sha256(str(s2))
    for i, w in enumerate(rows):
        w['session_sha256'] = sha2 if s2_of(i, w) else sha1
    (d / f'cs_{tag}.json').write_text(json.dumps({'schema': 'cua_so_v1', 'cua_so': rows}), encoding='utf-8')
    fit = {i: recs[i] for i, w in enumerate(rows) if w['loai'] in nt.LOAI_KHOP}
    hold = {i: recs[i] for i, w in enumerate(rows) if w['loai'] in nt.LOAI_GIU}
    nt.run_hieu_chuan(str(d / f'cs_{tag}.json'), [str(s1), str(s2)], T.NEO, (str(d / 'mau.mov'), str(s1), 0),
                      str(d / f'hc_{tag}'), str(d / 'a6.json'), 'T1', kiem_r=[str(d / 'kiem.json')], h_neo=T.H_N,
                      cfg=T.CFG, a6_mod=T.FakeA6Mod, _records=fit)
    kg = []
    if du is not None:
        kg = [d / f'kgs_{tag}.json']
        o = {'schema': 'c7_khung_giua_session_v1', 'session_sha256': [sha1, sha2], 'du_m': du, 'n_moc': 12}
        if theo_cho is not None:
            o.update(mo_hinh='phys_v1', theo_cho=theo_cho)
        kg[0].write_text(json.dumps(o), encoding='utf-8')
    nt.run_lan_lap(str(d / f'hc_{tag}' / 'hieu_chuan_neo.json'), str(d / f'cs_{tag}.json'),
                   ses_ll or [str(s1), str(s2)], str(d / f'll_{tag}'), str(d / 'a6.json'), 'T1', u_beta_cam_deg=0.01,
                   cfg=T.CFG, a6_mod=T.FakeA6Mod, _records=hold, khung_giua_session=[str(x) for x in kg])
    hc = json.load(open(d / f'hc_{tag}' / 'hieu_chuan_neo.json', encoding='utf-8'))
    ll = json.load(open(d / f'll_{tag}' / 'lan_lap.json', encoding='utf-8'))
    return hc, ll, {x['luot_id']: x for x in ll['luot']}, (sha1, sha2)


KEYS = ('beta_deg', 'u_beta_deg', 'lan_lap_id', 'chuan_12b', 'delta_c_B_m', 'u_delta_c_B_m', 'trang_thai')


def test_v4_14_same_session_pairs_as_one_session(files):
    """Two sessions, every comparison within one session (K2 fit windows and the runs at K2 in session 2): every run
    number equals the one-session result; F6 / F8 list both sessions, the one-session keys are null."""
    rows, recs = T.build()
    _, _, _, L1 = T.run(files, rows, recs, tag='s1')
    rows, recs = T.build()
    hc, ll, L2, (sha1, sha2) = run_ds(files, rows, recs, 'ds', lambda i, w: (w['loai'] in nt.LOAI_KHOP and w['cho'] == 'K2')
                                      or w['luot_id'] == 'L03')          # L03: pre and post at K2
    assert hc['session'] is None and {s['sha256'] for s in hc['sessions']} == {sha1, sha2}
    assert ll['session_file'] is None and ll['session_sha256'] is None and len(ll['sessions']) == 2
    for lid, a in L1.items():
        b = L2[lid]
        khac = b['dau']['pre']['session_sha256'] != (b['dau'].get('chuan') or b['dau']['pre'])['session_sha256']
        if khac:
            continue
        for k in KEYS:
            assert a[k] == b[k], (lid, k, a[k], b[k])
    assert L2['L03']['dau']['pre']['session_sha256'] == sha2 and 'khung_giua_session' not in L2['L03']


def test_v4_14_cross_session_du_zero_changes_nothing(files, monkeypatch):
    """Every run in session 2, every fit window in session 1, du_m = 0: the band and the beta term are 0. With the
    tag-edge gate off every number equals the one-session result; with the gate on (default, CODE_KHUNG sec 5) the
    scale / c_B comparisons across sessions are khong_xac_dinh: no lan_lap_id, delta_c_B chua_ket_luan."""
    rows, recs = T.build()
    _, _, _, L1 = T.run(files, rows, recs, tag='z1')
    rows, recs = T.build()
    _, _, L2, (sha1, sha2) = run_ds(files, rows, recs, 'z2', lambda i, w: w['loai'] in nt.LOAI_GIU, du=0.0)
    for lid, a in L1.items():
        b = L2[lid]
        assert b['beta_deg'] == a['beta_deg'] and b['u_beta_deg'] == a['u_beta_deg']
        assert b['khung_giua_session']['u_beta_deg'] == 0.0 and b['khung_giua_session']['cach'] == 'tho_du_m'
        assert set(b['khung_giua_session']['session_sha256']) == {sha1, sha2}
        assert b['lan_lap_id'] is None and b['lan_lap_id_ly_do'] == 'lan_lap_khong_xac_dinh'
        assert 'vien_ma_khac_session' in b['ly_do']['delta_c_B'] and b['trang_thai']['delta_c_B'] == 'chua_ket_luan'
    monkeypatch.setattr(nt, 'VIEN_MA_KHAC_SESSION_CHAN', False)
    rows, recs = T.build()
    _, _, L3, _ = run_ds(files, rows, recs, 'z3', lambda i, w: w['loai'] in nt.LOAI_GIU, du=0.0)
    for lid, a in L1.items():
        for k in KEYS:
            assert a[k] == L3[lid][k], (lid, k)


PHYS = dict(du_xoay_rad=1e-4, du_thang=2e-5, du_bien_dang_rad=5e-5, xoay_bien_thien_rad=3e-5)


def test_v4_14_phys_per_place(files, monkeypatch):
    """theo_cho (phys, Liam 20/11 at CODE_KHUNG): beta + (du_xoay + du_bien_dang) of the run's place (not the coarse
    asin(2 du_m / L_AB) of du_m = 0.3 mm); a place missing from theo_cho -> inf, chua_ket_luan (thieu_cho_<cho>).
    Gate off: the small phys band keeps the placement of the one-session run (lan_lap_id given across sessions)."""
    rows, recs = T.build()
    _, _, _, L1 = T.run(files, rows, recs, tag='p1')
    tc = {'goc': dict(PHYS), 'K1': dict(PHYS, du_xoay_rad=2e-4)}
    rows, recs = T.build()
    _, _, L2, _ = run_ds(files, rows, recs, 'p2', lambda i, w: w['loai'] in nt.LOAI_GIU, du=3e-4, theo_cho=tc)
    for lid, a in L1.items():
        b = L2[lid]
        cho = b['cho']
        if cho == 'K2':
            assert b['khung_giua_session']['cach'] == 'thieu_cho_K2' and b['trang_thai']['beta'] == 'chua_ket_luan'
            continue
        ub = math.degrees(tc[cho]['du_xoay_rad'] + PHYS['du_bien_dang_rad'])
        assert b['khung_giua_session']['cach'] == 'phys'
        assert b['u_beta_deg'] == pytest.approx(a['u_beta_deg'] + ub, abs=1e-12)
    monkeypatch.setattr(nt, 'VIEN_MA_KHAC_SESSION_CHAN', False)
    rows, recs = T.build()
    _, _, L3, _ = run_ds(files, rows, recs, 'p3', lambda i, w: w['loai'] in nt.LOAI_GIU, du=3e-4,
                         theo_cho=dict(tc, K2=dict(PHYS)))
    for lid, a in L1.items():
        assert L3[lid]['lan_lap_id'] == a['lan_lap_id'] and L3[lid]['chuan_12b'] == a['chuan_12b'], lid
        th = L3[lid]['so_sanh_trong_luot']                       # pre / post of one run: one session, no band
        assert all('u_khung_giua_session' not in v for v in th.values())


def test_v4_14_cross_session_index_and_missing_file(files):
    """du_m = 0.1 mm: beta gets + asin(2 du / L_AB) (L_AB = 0.20 m), over u_beta_max -> beta chua_ket_luan; the
    placement comparison of the pre with the standard (other session) is inside the band -> no lan_lap_id. No CAM
    file for the pair -> thieu_khung_giua_session, no lan_lap_id, beta chua_ket_luan."""
    rows, recs = T.build()
    _, _, _, L1 = T.run(files, rows, recs, tag='c1')
    rows, recs = T.build()
    hc, _, L2, _ = run_ds(files, rows, recs, 'c2', lambda i, w: w['loai'] in nt.LOAI_GIU, du=1e-4)
    L_AB = float(np.linalg.norm(np.array(hc['G'][1]) - np.array(hc['G'][0])))
    assert L_AB == pytest.approx(0.20, abs=1e-4)
    ub = math.degrees(math.asin(2e-4 / L_AB))
    co_u = []
    for lid, a in L1.items():
        b = L2[lid]
        assert b['beta_deg'] == pytest.approx(a['beta_deg'], abs=1e-12)
        assert b['u_beta_deg'] == pytest.approx(a['u_beta_deg'] + ub, abs=1e-12)
        assert b['trang_thai']['beta'] == 'chua_ket_luan' and 'u_beta_luot_vuot' in b['ly_do']['beta']
        assert b['lan_lap_id'] is None and b['lan_lap_id_ly_do'] == 'lan_lap_khong_xac_dinh'
        assert b['doi_so_chuan'] is None
        if a['u_delta_c_B_m'] is not None:                         # + 2 du + |c_B| 2 du / L_AB (and the ratio part)
            co_u.append(lid)
            ph = b['u_delta_c_B_phan']
            assert ph['khung_giua_session'] >= 2e-4
            assert b['u_delta_c_B_m'] == pytest.approx(sum(ph.values()), rel=1e-12)
            assert b['u_delta_c_B_m'] > a['u_delta_c_B_m'] + ph['khung_giua_session']
    assert co_u                                                     # the check ran on at least one run
    rows, recs = T.build()
    _, _, L3, _ = run_ds(files, rows, recs, 'c3', lambda i, w: w['loai'] in nt.LOAI_GIU, du=None)
    for lid, b in L3.items():
        assert 'thieu_khung_giua_session' in b['ly_do']['beta'] and b['trang_thai']['beta'] == 'chua_ket_luan'
        assert b['lan_lap_id'] is None
        assert b['trang_thai']['delta_c_B'] == 'chua_ket_luan'


def test_v4_14_so_sanh_band():
    """so_sanh with a CtxMap: a pair of different sessions widens the decision by u (doi only above lim + u,
    khong_doi only at or below lim - u); same session: as without cmap."""
    w = dict(theta=0.0, ses='a')
    for th_deg, want in ((0.01, 'khong_doi'), (0.04, 'khong_xac_dinh'), (0.40, 'doi')):
        v = dict(theta=math.radians(th_deg), ses='b')
        cmap = nt.CtxMap({'a': None, 'b': None}, [], {frozenset(('a', 'b')): dict(du_m=2e-5)}, L_AB=0.2)
        ctx = type('X', (), dict(side=0.145, tag_h=0.12))()
        r = nt.so_sanh(w, v, ctx, None, dict(theta=0.05, c_B=2e-4, s=5e-4), cmap)
        u = math.degrees(math.asin(4e-5 / 0.145)) + math.degrees(math.asin(4e-5 / 0.2))
        assert r['theta']['trang_thai'] == want and r['theta']['u_khung_giua_session'] == pytest.approx(u)
        r0 = nt.so_sanh(w, dict(v, ses='a'), ctx, None, dict(theta=0.05, c_B=2e-4, s=5e-4), cmap)
        assert r0['theta']['trang_thai'] == ('khong_doi' if th_deg <= 0.05 else 'doi')
        assert 'u_khung_giua_session' not in r0['theta']
    cmap = nt.CtxMap({'a': None, 'b': None}, [], {}, L_AB=0.2)
    r = nt.so_sanh(w, dict(theta=0.0, ses='b'), ctx, None, dict(theta=0.05, c_B=2e-4, s=5e-4), cmap)
    assert r['theta']['trang_thai'] == 'khong_xac_dinh' and r['theta']['ly_do'] == 'thieu_khung_giua_session'


def test_v4_14_sessions_must_share_geometry(files):
    """L125: one plumb, one z_c; one tag: sessions differing in z_c (kappa), tag_h or the plumb point are refused;
    the same session twice is refused; a hieu_chuan session missing from lan_lap --session is refused."""
    s1 = files / 'session.json'
    for chg in (dict(kappa=T.TAG_H / 2.7), dict(tag_h_m=0.121, kappa=0.121 / T.ZC), dict(plumb_xy_m=[-0.48, -0.03])):
        bad = _ses2(files, **chg)
        with pytest.raises(ValueError, match='sessions differ'):
            nt.tao_ctx([str(s1), str(bad)], lambda s: T.H_N, dict(nt.LL_DEFAULTS))
    with pytest.raises(ValueError, match='twice'):
        nt.tao_ctx([str(s1), str(s1)], lambda s: T.H_N, dict(nt.LL_DEFAULTS))
    rows, recs = T.build()
    with pytest.raises(ValueError, match='hieu_chuan_neo.json is not among'):
        run_ds(files, rows, recs, 'm', lambda i, w: w['cho'] == 'K2', du=0.0, ses_ll=[str(s1)])


def test_v4_14_window_session_not_given(files):
    """A window naming a session that is not one of --session -> error before any measurement."""
    rows, recs = T.build()
    rows[0]['session_sha256'] = 'f' * 64
    (files / 'cs_x.json').write_text(json.dumps({'schema': 'cua_so_v1', 'cua_so': rows}), encoding='utf-8')
    with pytest.raises(ValueError, match='not one of the --session'):
        nt.run_hieu_chuan(str(files / 'cs_x.json'), [str(files / 'session.json')], T.NEO,
                          (str(files / 'mau.mov'), str(files / 'session.json'), 0), str(files / 'hc_x'),
                          str(files / 'a6.json'), 'T1', kiem_r=[str(files / 'kiem.json')], h_neo=T.H_N, cfg=T.CFG,
                          a6_mod=T.FakeA6Mod, _records={})


def test_v4_14_read_khung_giua_session(tmp_path):
    def wr(n, **o):
        p = tmp_path / f'k{n}.json'
        p.write_text(json.dumps(dict(dict(schema='c7_khung_giua_session_v1', session_sha256=['a' * 64, 'b' * 64],
                                          du_m=1e-4, n_moc=10), **o)), encoding='utf-8')
        return str(p)
    ok = nt.doc_kgs([wr(0)])
    assert ok[frozenset(('a' * 64, 'b' * 64))]['du_m'] == 1e-4
    for n, o, m in ((1, dict(schema='x'), 'schema'), (2, dict(du_m=-1.0), 'du_m'), (3, dict(du_m=float('nan')), 'du_m'),
                    (4, dict(n_moc=0), 'n_moc'), (5, dict(session_sha256=['a' * 64, 'a' * 64]), 'two different'),
                    (6, dict(du_m=True), 'du_m')):
        with pytest.raises(ValueError, match=m):
            nt.doc_kgs([wr(n, **o)])
    for n, tc, m in ((9, {'goc': dict(PHYS, du_thang=-1.0)}, 'du_thang'), (10, {'O': dict(PHYS)}, 'theo_cho must'),
                     (11, {'K1': {k: v for k, v in PHYS.items() if k != 'du_bien_dang_rad'}}, 'du_bien_dang_rad')):
        with pytest.raises(ValueError, match=m):
            nt.doc_kgs([wr(n, theo_cho=tc)])
    assert nt.doc_kgs([wr(12, theo_cho={'goc': PHYS})])[frozenset(('a' * 64, 'b' * 64))]['theo_cho']['goc'] == PHYS
    with pytest.raises(ValueError, match='same session pair'):
        nt.doc_kgs([wr(7), wr(8, session_sha256=['b' * 64, 'a' * 64])])


def test_v4_14_measure_one_clip_one_session(tmp_path, monkeypatch):
    """Windows of one clip naming two sessions -> error (one clip, one camera session)."""
    clip = tmp_path / 'c.mov'
    clip.write_bytes(b'clip')
    sha = cm.sha256(str(clip))
    rows = [{'clip': 'c.mov', 'clip_sha256': sha, 'session_sha256': s, 'khung': [{'id': k, 'pts_s': k / 60}],
             'luot_id': 'L01', 'loai': lo} for k, (s, lo) in enumerate((('a', 'pre'), ('b', 'post')))]
    monkeypatch.setattr(nt, '_mau', lambda *a: 'mau')
    cmap = nt.CtxMap({'a': None, 'b': None}, [])
    with pytest.raises(ValueError, match='several sessions'):
        nt._measure(rows, [0, 1], None, None, cmap, None, str(tmp_path))


def test_v4_14_measure_camera_of_each_clip(tmp_path, monkeypatch):
    """Each clip is measured with the context of ITS session (not the first one)."""
    rows = []
    for n, ses in (('c1.mov', 'a'), ('c2.mov', 'b')):
        (tmp_path / n).write_bytes(n.encode())
        rows.append({'clip': n, 'clip_sha256': cm.sha256(str(tmp_path / n)), 'session_sha256': ses,
                     'khung': [{'id': 1, 'pts_s': 0.1}], 'luot_id': 'L01' if ses == 'a' else 'L02', 'loai': 'pre'})
    seen = {}
    monkeypatch.setattr(nt, '_mau', lambda *a: 'mau')
    monkeypatch.setattr(nt, 'do_clip', lambda path, wins, ctx, mau: (seen.setdefault(os.path.basename(path), ctx),
                                                                     ({k: [] for k, _ in wins}, {}))[1])
    cmap = nt.CtxMap({'a': 'ctx_a', 'b': 'ctx_b'}, [])
    nt._measure(rows, [0, 1], None, None, cmap, None, str(tmp_path))
    assert seen == {'c1.mov': 'ctx_a', 'c2.mov': 'ctx_b'}


def test_v4_14_so_sanh_phys_terms(monkeypatch):
    """so_sanh across sessions with a phys file: theta band du_bien_dang + xoay_bien_thien, scale du_thang +
    du_bien_dang of the place (gate off); gate on: scale and c_B khong_xac_dinh (vien_ma_khac_session), theta kept."""
    e = dict(du_m=3e-4, theo_cho={'K1': dict(PHYS)})
    cmap = nt.CtxMap({'a': None, 'b': None}, [], {frozenset(('a', 'b')): e}, L_AB=0.2)
    ctx = type('X', (), dict(side=0.145, tag_h=0.12))()
    w, v = dict(theta=0.0, ses='a', cho='K1'), dict(theta=math.radians(0.001), ses='b', cho='K1')
    ng = dict(theta=0.05, c_B=2e-4, s=5e-4)
    r = nt.so_sanh(w, v, ctx, None, ng, cmap)
    assert r['theta']['u_khung_giua_session'] == pytest.approx(math.degrees(8e-5))
    assert r['theta']['trang_thai'] == 'khong_doi'
    assert r['s']['ly_do'] == 'vien_ma_khac_session' and r['c_B']['ly_do'] == 'vien_ma_khac_session'
    monkeypatch.setattr(nt, 'VIEN_MA_KHAC_SESSION_CHAN', False)
    r = nt.so_sanh(w, v, ctx, None, ng, cmap)
    assert r['s']['u_khung_giua_session'] == pytest.approx(7e-5) and 'ly_do' not in r['s']
    r = nt.so_sanh(w, dict(v, cho='K2'), ctx, None, ng, cmap)                   # K2 not in theo_cho
    assert r['theta']['trang_thai'] == 'khong_xac_dinh' and r['theta']['ly_do'] == 'thieu_cho_K2'
