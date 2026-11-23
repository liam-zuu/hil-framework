"""Tests of the two-tape anchors (neo_cach 'tam_trang') and the clips measured at the same time (tien_trinh) of
tools/c7_nhac_tam.py (design 2026-11-19_Thiet_ke_nhac_tam_lan_lap_v4 section 12f, Liam 23/11). SYNTHETIC data (NOT a
measurement): rendered tapes, per-frame records built with the forward equation of v4 2, clips rendered from a known
camera (scene of test_c7_nhac_tam)."""
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c7_measure as cm  # noqa: E402
import c7_nhac_tam as nt  # noqa: E402
import test_c7_nhac_tam_lan_lap as T  # noqa: E402

CFG = dict(nt.LL_DEFAULTS, neo_cach='tam_trang')


# ---------------------------------------------------------------- tape centre on an image
def tape_img(cx, cy, side, ang_deg, size=200, ss=8, scribble=False):
    """Dark body (20) with a white tape (230) of side px centred at (cx, cy) turned by ang, anti-aliased by drawing at
    ss x and averaging down. scribble: a dark stroke fixed on the tape (its pattern). Returns (image, true centre of
    the grey-weighted blob from the fine drawing)."""
    big = np.full((size * ss, size * ss), 20, np.uint8)
    a = math.radians(ang_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def pts(local):
        return ((np.asarray(local, float) @ R.T + [cx, cy]) * ss + (ss - 1) / 2.0)
    sq = pts([[-side / 2, -side / 2], [side / 2, -side / 2], [side / 2, side / 2], [-side / 2, side / 2]])
    cv2.fillConvexPoly(big, np.round(sq * 16).astype(np.int32), 230, cv2.LINE_AA, 4)
    if scribble:
        st = pts([[-side * 0.3, -side * 0.2], [side * 0.1, side * 0.3], [side * 0.3, -side * 0.1]])
        cv2.polylines(big, [np.round(st * 16).astype(np.int32)], False, 60, max(1, ss), cv2.LINE_AA, 4)
    small = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    return small


@pytest.mark.parametrize('ang', [0.0, 17.0, 45.0, 90.0, 133.6])
@pytest.mark.parametrize('dx,dy', [(0.0, 0.0), (0.3, -0.45)])
def test_tape_centre_any_heading(ang, dx, dy):
    """Symmetric tape at a sub-pixel place, any heading: centre within 0.05 px."""
    g = tape_img(100 + dx, 100 + dy, 24.0, ang)
    u, v, S = nt.tam_bang_dinh(g, 104.0, 97.0, None, CFG)
    assert math.hypot(u - 100 - dx, v - 100 - dy) < 0.05, (u, v)
    assert 450 < S < 700


def test_tape_pattern_and_light_do_not_move_the_centre():
    """NBD1-1: the centre is the geometric centre of the tape: a stroke drawn on it, or two halves lit 225 / 230, do
    not move it (the grey-weighted centre moved 0.07 px with the light)."""
    for ang in (0.0, 30.0, 90.0, 133.6):
        g = tape_img(100.3, 99.6, 24.0, ang, scribble=True)
        u, v, _ = nt.tam_bang_dinh(g, 100.0, 100.0, None, CFG)
        assert math.hypot(u - 100.3, v - 99.6) < 0.05, (ang, u, v)
    g = np.full((200, 200), 20, np.uint8)
    g[88:112, 88:100] = 225
    g[88:112, 100:112] = 230
    u, v, _ = nt.tam_bang_dinh(g, 100.0, 100.0, None, CFG)
    assert math.hypot(u - 99.5, v - 99.5) < 0.002, (u, v)


def test_tape_close_to_the_crop_border_is_refused():
    """NBD1-4: a blob whose grown mask or body ring would be cut by the crop is not used."""
    g = np.full((200, 200), 20, np.uint8)
    g[90:110, 99:129] = 230                                           # crop of uc = 100 is x 70..129
    assert math.isnan(nt.tam_bang_dinh(g, 100.0, 100.0, None, dict(CFG, tam_lech_px=40))[0])
    g = np.full((200, 200), 20, np.uint8)
    g[90:110, 95:118] = 230
    assert math.isfinite(nt.tam_bang_dinh(g, 100.0, 100.0, None, dict(CFG, tam_lech_px=40))[0])


def test_tape_refusals():
    g = tape_img(100, 100, 24.0, 20.0)
    S0 = nt.tam_bang_dinh(g, 100.0, 100.0, None, CFG)[2]
    assert math.isfinite(nt.tam_bang_dinh(g, 100.0, 100.0, S0, CFG)[0])
    assert math.isnan(nt.tam_bang_dinh(g, 100.0, 100.0, S0 * 2.0, CFG)[0])          # area far from the template
    assert math.isnan(nt.tam_bang_dinh(g, 100.0, 100.0, S0 / 2.0, CFG)[0])
    assert math.isnan(nt.tam_bang_dinh(g, 120.0, 100.0, None, CFG)[0])              # more than 15 px from the blob
    assert math.isnan(nt.tam_bang_dinh(g, 100.0 + 28.0, 100.0, None, dict(CFG, tam_lech_px=40))[0])  # cut by the crop
    assert math.isnan(nt.tam_bang_dinh(np.full((200, 200), 90, np.uint8), 100.0, 100.0, None, CFG)[0])  # flat
    assert math.isnan(nt.tam_bang_dinh(g, 10.0, 100.0, None, CFG)[0])               # crop off the image
    for bad in (float('nan'), float('inf'), -float('inf')):                         # NBD1-11
        assert math.isnan(nt.tam_bang_dinh(g, bad, 100.0, None, CFG)[0])
        assert math.isnan(nt.tam_bang_dinh(g, 100.0, bad, None, CFG)[0])


def test_tape_touching_a_bright_region_by_one_diagonal_pixel():
    """CODE 23/11 run03: the tape meets a bright region that runs to the crop border at one corner pixel only. 4-connected
    blobs keep the tape apart (8-connected merged them and the merged blob, cut by the border, was refused); the bright
    region stays out of the weight mask and of the body ring, so the centre is unbiased."""
    g = np.full((200, 200), 20, np.uint8)
    g[90:110, 90:110] = 230                                           # tape, centre (99.5, 99.5)
    g[110:200, 110:200] = 230                                         # touches the tape corner (109, 109) diagonally
    u, v, S = nt.tam_bang_dinh(g, 100.0, 100.0, None, CFG)
    assert S == 400 and math.hypot(u - 99.5, v - 99.5) < 1e-9, (u, v, S)
    n8 = cv2.connectedComponentsWithStats(cv2.threshold(g[70:130, 70:130], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])[0]
    assert n8 == 2                                                    # (the old 8-connected call saw one blob)


def test_tape_scale_follows_the_frame_camera():
    """ti (px per metre of the frame over the template) carries the area, crop and distance bounds: a tape 0.7 x as
    large in pixels (area 0.49 S0) is refused at ti 1 and found at ti 0.7."""
    S0 = nt.tam_bang_dinh(tape_img(100, 100, 24.0, 0.0), 100.0, 100.0, None, CFG)[2]
    g = tape_img(100, 100, 24.0 * 0.7, 30.0)
    assert math.isnan(nt.tam_bang_dinh(g, 100.0, 100.0, S0, CFG, 1.0)[0])
    u, v, _ = nt.tam_bang_dinh(g, 100.0, 100.0, S0, CFG, 0.7)
    assert math.hypot(u - 100, v - 100) < 0.05
    assert math.isnan(nt.tam_bang_dinh(g, 100.0, 100.0, S0, CFG, float('nan'))[0])
    cam = dict(model='B2', f=900.0, cx=960.0, cy=540.0, k=[0.0, 0.0], rvec=[math.pi, 0.0, 0.0], tvec=[0.0, 0.0, 2.6])
    c1, c2 = cm.cam_from_dict(cam), cm.cam_from_dict(dict(cam, f=1451.0 / 1675.0 * 900.0))
    for xy in ((0.0, 0.0), (1.2, -0.4)):
        assert nt.px_moi_m(c2, xy) / nt.px_moi_m(c1, xy) == pytest.approx(1451.0 / 1675.0, rel=1e-6)
        assert nt.px_moi_m(c1, xy) == pytest.approx(900.0 / 2.6, rel=1e-9)  # NBD1-12: straight down, no distortion
    # NBD1-12: through NeoMau (template on camera 1, frame on camera 2): ti_le is the focal ratio
    g = np.full((1080, 1920), 20, np.uint8)
    for (cx, cy) in ((900, 500), (1100, 560)):
        g[cy - 12:cy + 12, cx - 12:cx + 12] = 230
    neo = nt.parse_neo('A:892,492,16,16;B:1092,552,16,16')
    m = nt.NeoMau(g, c1, dict(tag_x=0.0, tag_y=0.0, tag_psi=0.0), neo, T.H_N / T.ZC, T.C, CFG)
    assert np.allclose(m.ti_le(c2, np.zeros(2), 0.0), 1451.0 / 1675.0, rtol=1e-6)
    with pytest.raises(ValueError, match='the same tape'):                        # NBD1-5
        nt.NeoMau(g, c1, dict(tag_x=0.0, tag_y=0.0, tag_psi=0.0), nt.parse_neo('A:892,492,8,8;B:900,500,8,8'),
                  T.H_N / T.ZC, T.C, CFG)


# ---------------------------------------------------------------- two anchors on records (exact forward equation)
G2 = np.array([[0.0, 0.0], [0.29, 0.0]])
H2 = np.array([T.H_N, 0.104])
NEO2 = 'A:100,100,18,18;B:200,100,18,18'
BOX2 = [[100, 100, 18, 18], [200, 100, 18, 18]]


def recs_for2(alpha, o, n=12, seed=0, noise=2e-6, tag=True, F=None, H=T.H_CAM, id0=0, tag_noise=True, dB=None, **kw):
    rng = np.random.default_rng(seed)
    F = T.C + T.F_OFF if F is None else F
    P = G2 @ T.rot(alpha).T + o
    if dB is not None:                                                 # tape B moved (body frame)
        P[1] = P[1] + T.rot(alpha) @ np.asarray(dB, float)
    out = []
    for k in range(n):
        Ph = T.proj_indep(P, H2, F, H, T.H_N / T.ZC) + rng.normal(0, noise, (2, 2))
        tg = None
        if tag:
            tg = T.tag_obs(alpha, o, F, H, **kw)
            if tag_noise:
                tg = dict(tg, x=tg['x'] + rng.normal(0, 2e-6), y=tg['y'] + rng.normal(0, 2e-6),
                          psi=tg['psi'] + rng.normal(0, 1e-5), s=tg['s'] + rng.normal(0, 1e-6))
        out.append(dict(id=id0 + k, t=(id0 + k) / 60.0, cam_ok=True, F=np.array(F), H=H, Ph=Ph, neo_ok=True, tag=tg))
    return out


@pytest.fixture
def files2(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'recs_for', recs_for2)
    monkeypatch.setattr(T, 'NEO', NEO2)
    ses = dict(camera=dict(model='B2', f=900.0, cx=960.0, cy=540.0, k=[0.0, 0.0], rvec=[math.pi, 0.0, 0.0],
                           tvec=[0.0, 0.0, 2.6]),
               marks={}, fit_ids=[1, 2], car_side_m=T.SIDE, tag_h_m=T.TAG_H, kappa=T.TAG_H / T.ZC,
               plumb_xy_m=T.C.tolist(), rms_px_H=0.3)                  # as T.files, boxes of A, B
    d = tmp_path
    (d / 'session.json').write_text(json.dumps(ses), encoding='utf-8')
    (d / 'a6.json').write_text(json.dumps(dict(schema='c8b_a6_v2', g_b=[0.0, 0.0], eps_g=1e-4)), encoding='utf-8')
    (d / 'mau.mov').write_bytes(b'x')
    (d / 'kiem.json').write_text(json.dumps(T.kiem_r(d / 'mau.mov', BOX2)), encoding='utf-8')
    return d


def test_two_tapes_beta_exact_and_no_eps(files2):
    _, hc, ll, L = T.run(files2, *T.build(), cfg=dict(neo_cach='tam_trang'))
    assert hc['trang_thai'] == 'dat', hc['ly_do']
    assert hc['neo_ten'] == ['A', 'B'] and hc['nguong']['neo_cach'] == 'tam_trang'
    assert np.allclose(hc['G'], G2, atol=2e-5) and np.allclose(hc['h'], H2, atol=5e-4)
    assert hc['eps_neo_deg'] == 0.0
    L_AB = 0.29
    assert hc['r_max_m'] == pytest.approx(L_AB * math.tan(math.radians(0.015)) / 2.0, rel=1e-4)
    for lid, (pre, post) in zip(L, [(r[1], r[2]) for r in T.default_runs()]):
        assert L[lid]['beta_deg'] == pytest.approx(T.true_beta(pre, pre[0]), abs=2e-3)
        assert L[lid]['trang_thai']['beta'] == 'dat', L[lid]['ly_do']['beta']


def test_two_tapes_rank_missing_at_one_place(files2):
    """Every fit window at one place and heading: h_B not separable from the A-B length -> thieu_hang."""
    rows, recs = T.build(runs=[('L01', ('goc', 0.3, 0.5, {}), ('goc', 0.3, 0.5, {})),
                               ('L02', ('goc', 0.0, 0.0, {}), ('goc', 0.0, 0.0, {}))])
    for i, w in enumerate(rows):
        if w['loai'] in nt.LOAI_KHOP:
            recs[i] = recs_for2(*T.body_at('goc', 0.0, 0.0), n=12, seed=i, id0=recs[i][0]['id'])
    _, hc, _, L = T.run(files2, rows, recs, tag='one', cfg=dict(neo_cach='tam_trang'))
    assert 'thieu_hang' in L['L01']['ly_do']['beta'] and L['L01']['trang_thai']['beta'] == 'chua_ket_luan'


def test_two_tapes_one_moved_along_ab_fails_the_gate(files2):
    rows, recs = T.build()
    k = next(i for i, w in enumerate(rows) if w['loai'] == 'dung_12b' and w['cho'] == 'K1')
    a, o = T.body_at('K1', T.PLACES['K1'][2][1], 0.02)
    recs[k] = recs_for2(a, o, n=12, seed=99, id0=recs[k][0]['id'], dB=(0.0003, 0.0))   # 0.3 mm along A-B
    c, hc, _, _ = T.run(files2, rows, recs, tag='mv', cfg=dict(neo_cach='tam_trang'))
    assert c == 1 and hc['ly_do'] == ['neo_khong_kha_thi']


def test_two_tapes_rules(files2):
    rows, recs = T.build()
    with pytest.raises(ValueError, match='exactly the anchors A and B'):
        nt.run_hieu_chuan(*_hc_args(files2, rows, 'A:100,100,18,18;B:200,100,18,18;oc1:150,160,18,18'), cfg=dict(neo_cach='tam_trang'),
                          a6_mod=T.FakeA6Mod, _records={})
    with pytest.raises(ValueError, match='mau or tam_trang'):
        nt.run_hieu_chuan(*_hc_args(files2, rows, NEO2), cfg=dict(neo_cach='tam'), a6_mod=T.FakeA6Mod, _records={})
    # template mode keeps >= 3 anchors: two boxes -> thieu_neo
    c = nt.run_hieu_chuan(*_hc_args(files2, rows, NEO2, out='hc_mau2'), cfg=dict(T.CFG), a6_mod=T.FakeA6Mod, _records={})
    hc = json.load(open(files2 / 'hc_mau2' / 'hieu_chuan_neo.json', encoding='utf-8'))
    assert c == 1 and hc['ly_do'] == ['thieu_neo']


def _hc_args(d, rows, neo, out='hc_r'):
    ses_sha = cm.sha256(str(d / 'session.json'))
    for w in rows:
        w['session_sha256'] = ses_sha
    (d / 'cs_r.json').write_text(json.dumps({'schema': 'cua_so_v1', 'cua_so': rows}), encoding='utf-8')
    return (str(d / 'cs_r.json'), str(d / 'session.json'), neo, (str(d / 'mau.mov'), str(d / 'session.json'), 0),
            str(d / out), str(d / 'a6.json'), 'T1', [str(d / 'kiem.json')], T.H_N)


def test_second_height_solution_is_found():
    """NBD1-3 (Codex counter-example): H 2.60, F = C, three windows along y: (h_B, L_AB) = (0.10, 0.30) and
    (0.11, 0.2988) fit with zero residual; the scan finds the second one."""
    H, hn = 2.60, 0.10
    a = 1.0 - hn / 2.60
    C0, F0 = np.zeros(2), np.zeros(2)
    G, h = np.array([[0.0, 0.0], [0.30, 0.0]]), np.array([0.10, 0.10])
    obs = []
    for y in (0.0, 0.05, 1.0):
        B = np.array([75.0 - math.sqrt(74.7 ** 2 - y ** 2), y])
        obs.append(dict(Ph=nt.thuan(G, h, 0.0, B - [0.30, 0.0], F0, H, C0, a), F=F0, H=H))
    fit = nt.khop_hinh(obs, hn, C0, a)
    hg = nt.hang(fit, dict(nt.LL_DEFAULTS))
    ctx = type('X', (), dict(h_n=hn, C=C0, a_n=a))()
    assert nt.quet_h_hai_neo(obs, fit, float(hg['sigma_h'][0]), 3.9e-5, ctx, dict(nt.LL_DEFAULTS)) == [0.11]
    # a generic geometry (three places, headings) has one solution: test_two_tapes_locked_tuple_threshold_...


def test_second_height_solution_blocks_the_lock(files2, monkeypatch):
    """NBD1-3 wiring: a second solution found by the scan makes hieu_chuan chua_ket_luan (thieu_hang), recorded."""
    monkeypatch.setattr(nt, 'quet_h_hai_neo', lambda *a, **k: [0.123])
    c, hc, _, _ = T.run(files2, *T.build(), tag='hn', cfg=dict(neo_cach='tam_trang'))
    assert c == 1 and hc['ly_do'] == ['thieu_hang'] and hc['hai_nghiem_h_B_m'] == [0.123]


def test_two_tapes_locked_tuple_threshold_and_bad_process_count(files2):
    """NBD1-7: the tuple of the defaults and its JSON list are the same locked value; NBD1-9: tien_trinh >= 1."""
    _, hc, ll, _ = T.run(files2, *T.build(), tag='tp', cfg=dict(neo_cach='tam_trang', tam_dt_ti=(0.6, 1.6)))
    assert hc['trang_thai'] == 'dat' and hc['hai_nghiem_h_B_m'] == []
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='positive integer'):
            nt._measure([], [], None, None, None, None, None, tien_trinh=bad)


def test_eps_terms_are_zero_with_two_anchors():
    a = 1.0 - T.H_N / T.ZC
    b = nt.eps_neo_bang(G2, H2, T.C + T.F_OFF, T.H_CAM, T.C, a)
    assert nt.eps_goc(b, 1e-4) == 0.0 and nt.eps_vi_tri(b, 1e-4, [0.1, 0.0]) == 0.0
    b3 = nt.eps_neo_bang(T.G_TRUE, T.H_TRUE, T.C + T.F_OFF, T.H_CAM, T.C, a)
    assert nt.eps_goc(b3, 1e-4) > 0.0                                   # three anchors unchanged


# ---------------------------------------------------------------- rendered clips: tapes, clips at the same time
def _render_setup(tmp_path, monkeypatch, anchors):
    import test_c7_nhac_tam as t7
    tex = np.full_like(t7.BODY_TEX, 25)
    side = int(round(0.03 / t7.RES))
    for bx, by in anchors:                                             # white 30 mm tapes on the dark body
        c0, r0 = int(round((bx + t7.BODY_L / 2) / t7.RES)), int(round((t7.BODY_W / 2 - by) / t7.RES))
        tex[r0 - side // 2:r0 + side // 2, c0 - side // 2:c0 + side // 2] = 235
    monkeypatch.setattr(t7, 'BODY_TEX', tex)
    d = tmp_path
    floor = t7.floor_img(t7.CAM)
    t7.write_video(d / 'H.avi', [floor] * 6)
    ses = cm.calibrate(str(d / 'H.avi'), t7.marks_dict(), t7.ZC, t7.PLUMB, t7.TAG_H, t7.CAR_SIDE)
    (d / 'session.json').write_text(json.dumps(cm._json(ses)), encoding='utf-8')
    return t7, floor, cm.sha256(str(d / 'session.json'))


def test_end_to_end_tapes_rendered_and_parallel_equal(tmp_path, monkeypatch):
    """Clips rendered with two white tapes on a dark body at headings 20 / 80 deg apart: hieu_chuan + lan_lap with
    neo_cach tam_trang give the true body rotation; the same run with 3 clips at a time gives the same file content."""
    monkeypatch.setattr(nt, 'trong_mien', lambda x, pts, tol=1e-9: True)
    anchors = [(0.15, 0.125), (-0.15, -0.125)]
    t7, floor, ses_sha = _render_setup(tmp_path, monkeypatch, anchors)
    d = tmp_path
    places = dict(goc=((0.0, 0.0), 20.0), K1=((0.5, 0.25), 100.0))
    rows = []

    def add(luot, cho, loai, s, dpsi):
        (x, y), hd = places[cho]
        a = math.radians(hd + dpsi)
        f = floor.copy()
        t7.render_car(f, (x + s * math.cos(a), y + s * math.sin(a), a), (0.0, 0.0, 0.0))
        path = d / f'w{len(rows)}.avi'
        t7.write_video(path, [f] * 6)
        fr = [{'id': k, 'pts_s': t} for k, t, _ in cm.iter_frames(str(path))]
        rows.append({'schema': 'cua_so_v1', 'luot_id': luot, 'cho': cho, 'loai': loai, 'clip': path.name,
                     'clip_sha256': cm.sha256(str(path)), 'session_sha256': ses_sha, 'khung': fr,
                     'khung_ma_hop_le': [k['id'] for k in fr], 't_canh_s': None, 'ket_luan': {'dung_yen': True}})
    for cho in places:
        for j, s in enumerate([0.0, 0.2, -0.2, 0.2, 0.0]):
            add(f'12b_{cho}', cho, 'ref_12b' if j == 0 else 'dung_12b', s, 0.1 * j)
    truth = {}
    for luot, cho, s, dp in (('L01', 'goc', 0.05, 0.7), ('L02', 'K1', -0.05, -0.4), ('L03', 'goc', 0.0, 0.3)):
        truth[luot] = dp
        add(luot, cho, 'pre', s, dp)
        add(luot, cho, 'post', s + 0.002, dp)
    (d / 'cs.json').write_text(json.dumps({'schema': 'cua_so_v1', 'cua_so': rows}), encoding='utf-8')
    (x0, y0), hd = places['goc']
    a0 = math.radians(hd)
    spec, boxes = [], []
    for name, (bx, by) in zip(('A', 'B'), anchors):
        P = np.array([x0 + math.cos(a0) * bx - math.sin(a0) * by, y0 + math.sin(a0) * bx + math.cos(a0) * by, t7.TAG_H])
        u, v = t7.CAM.project(P[None])[0]
        b = [int(round(u - 4)), int(round(v - 4)), 8, 8]               # a small box inside the tape (as on C2)
        boxes.append(b)
        spec.append(f'{name}:' + ','.join(map(str, b)))
    (d / 'kiem.json').write_text(json.dumps(T.kiem_r(d / 'w0.avi', boxes)), encoding='utf-8')
    (d / 'a6.json').write_text(json.dumps(dict(g_b=[0.0, 0.0], eps_g=1e-4)), encoding='utf-8')
    cfg = dict(min_khung=4, min_cap=2, eps_tong_deg=1.0, still_neo_m=0.003, u_beta_max_deg=1.0, doi_theta_deg=0.5,
               doi_c_m=0.003, doi_s=0.005, neo_cach='tam_trang')       # 1 px ~ 3 mm here: smoke thresholds
    mau = (str(d / 'w0.avi'), str(d / 'session.json'), 0)
    out = {}
    for tt in (1, 3):
        c = nt.run_hieu_chuan(str(d / 'cs.json'), str(d / 'session.json'), ';'.join(spec), mau, str(d / f'hc{tt}'),
                              str(d / 'a6.json'), 'T1', kiem_r=[str(d / 'kiem.json')], h_neo=t7.TAG_H, clip_dir=str(d),
                              cfg=cfg, a6_mod=T.FakeA6Mod, tien_trinh=tt)
        hc = json.load(open(d / f'hc{tt}' / 'hieu_chuan_neo.json', encoding='utf-8'))
        assert c == 0 and hc['trang_thai'] == 'dat', (hc['ly_do'], hc.get('cua_so_loi'))
        nt.run_lan_lap(str(d / f'hc{tt}' / 'hieu_chuan_neo.json'), str(d / 'cs.json'), str(d / 'session.json'),
                       str(d / f'll{tt}'), str(d / 'a6.json'), 'T1', clip_dir=str(d), cfg=cfg, a6_mod=T.FakeA6Mod,
                       tien_trinh=tt)
        ll = json.load(open(d / f'll{tt}' / 'lan_lap.json', encoding='utf-8'))
        out[tt] = (hc, ll)
        L = {x['luot_id']: x for x in ll['luot']}
        for luot, dp in truth.items():
            assert L[luot]['beta_deg'] == pytest.approx(dp, abs=0.05), (luot, L[luot]['beta_deg'])

    def strip(o):                                                      # lock time / paths differ between the two runs
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()          # the hieu_chuan file differs by its lock time only
                    if k not in ('khoa', 'thoi_diem', 'file', 'luc', 'lan_lap_id') and 'sha256' not in k}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    def diff(x, y, path=''):
        if isinstance(x, dict) and isinstance(y, dict):
            return [p for k in set(x) | set(y) for p in diff(x.get(k), y.get(k), f'{path}/{k}')]
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            return [p for i, (u, v) in enumerate(zip(x, y)) for p in diff(u, v, f'{path}[{i}]')]
        return [] if x == y else [path]
    assert diff(strip(out[1][0]), strip(out[3][0])) == []
    assert diff(strip(out[1][1]), strip(out[3][1])) == []
