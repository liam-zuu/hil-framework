"""Tests of tools/c7_kiem_cheo.py (CODE_KIEMCHEO, design 2026-11-18_Thiet_ke_c7_kiem_cheo_v1.md section 8)."""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import c7_clip_probe as cp  # noqa: E402
import c7_kiem_cheo as kc  # noqa: E402

W, H = 3840, 2160
ZC = 2.632
SIDE = 0.0964


def true_cam(f=1200.0, k=-0.02, tilt=(0.01, -0.008)):
    R = cp._rot([math.pi + tilt[0], tilt[1], 0.0])
    C = np.array([0.03, -0.02, ZC])
    return cp.Camera('B1', f, W / 2, H / 2, [k], cv_rvec(R), -R @ C)


def cv_rvec(R):
    import cv2
    return cv2.Rodrigues(R)[0].ravel()


def grid_marks():
    xs = [-1.9, -1.4, -0.7, 0.0, 0.7, 1.4, 1.9]
    ys = [-1.45, -0.9, 0.0, 0.9, 1.45]
    pts = [(x, y) for y in ys for x in xs][:29] + [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5)]
    pts = list(dict.fromkeys(pts))
    return {i + 1: (x, y, SIDE) for i, (x, y) in enumerate(pts)}


def image_of(cam, true_marks):
    return {i: cam.project(np.column_stack([cp.mark_corners_floor(*m), np.zeros(4)])) for i, m in true_marks.items()}


@pytest.fixture(scope='module')
def scene():
    marks = grid_marks()
    cam = true_cam()
    return marks, cam, image_of(cam, marks)


def test_exact_scene_gives_zero(scene):
    marks, cam, img = scene
    rows = kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=0.8)
    assert rows and max(abs(r['dtheta_deg']) for r in rows) < 1e-3
    groups = kc.make_groups(sorted(img))
    cams = kc.group_cams(img, marks, groups, 'B1', W, H, ZC)
    p1 = kc.p1_clip(img, marks, groups, cams)
    # the centre is the back-projected MEAN of the image corners (CAM v2 1, KCC1-1): under perspective it is not the
    # square's centre, a definition bias of < 0.1 mm here that P1 carries
    assert set(p1) == set(img) and max(np.linalg.norm(v) for v in p1.values()) < 2e-4
    res, _ = _uso(img, img, [img] * 40, marks, groups)
    assert max(max(c['u_so_m'], c['u_so_tho_m']) for c in res.values()) < 1e-6


def test_a3_sees_a_moved_mark_exactly(scene):
    """True position of mark m moved 2 mm across a pair; the tape keeps the old value: the pair (m, j), both left
    out, shows dtheta = atan(2 mm / L) (the fit never saw m or j)."""
    marks, cam, _ = scene
    m, j = 30, 32                           # (0.5, 0) and (0, 0.5)
    vt = np.array(marks[j][:2]) - np.array(marks[m][:2])
    n = np.array([-vt[1], vt[0]]) / np.linalg.norm(vt)
    true = dict(marks)
    true[m] = (marks[m][0] + 0.002 * n[0], marks[m][1] + 0.002 * n[1], SIDE)
    img = image_of(cam, true)
    rows = {(r['i'], r['j']): r for r in kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=0.8)}
    L = float(np.linalg.norm(vt))
    exp = -math.degrees(math.atan2(0.002, L))          # m is the tail of the vector m -> j
    assert rows[(m, j)]['dtheta_deg'] == pytest.approx(exp, abs=2e-3)


def test_a3_shift_along_the_pair_gives_zero_angle(scene):
    """KC2-4: dtheta = atan2(d_perp, L + d_par); a shift along the pair alone turns nothing."""
    marks, cam, _ = scene
    m, j = 30, 32
    vt = np.array(marks[j][:2]) - np.array(marks[m][:2])
    u = vt / np.linalg.norm(vt)
    L = float(np.linalg.norm(vt))
    for d_par, d_perp in ((0.002, 0.0), (0.002, 0.001)):
        true = dict(marks)
        true[j] = (marks[j][0] + d_par * u[0] - d_perp * u[1], marks[j][1] + d_par * u[1] + d_perp * u[0], SIDE)
        rows = {(r['i'], r['j']): r for r in kc.a3_clip(image_of(cam, true), marks, 'B1', W, H, ZC, l_max=0.8)}
        assert rows[(m, j)]['dtheta_deg'] == pytest.approx(math.degrees(math.atan2(d_perp, L + d_par)), abs=2e-3)


def test_held_out_marks_do_not_enter_their_fit(scene):
    """Independence (R6-2): changing the tape coordinates AND image corners of the held-out marks leaves the
    camera used to predict them unchanged; changing a training mark changes it (the test is not empty)."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    k, i = 0, groups[0][0]
    c0 = kc.fit_without(img, marks, sorted(set(img) - set(groups[k])), 'B1', W, H, ZC)
    m2, img2 = dict(marks), dict(img)
    m2[i] = (marks[i][0] + 0.01, marks[i][1] - 0.02, SIDE)
    img2[i] = img[i] + 3.0
    c1 = kc.group_cams(img2, m2, groups, 'B1', W, H, ZC)[k]
    for a in ('f', 'k', 'rvec', 'tvec'):
        assert np.array_equal(np.atleast_1d(getattr(c0, a)), np.atleast_1d(getattr(c1, a)))
    o = groups[1][0]
    img3 = dict(img)
    img3[o] = img[o] + 3.0
    c2 = kc.group_cams(img3, marks, groups, 'B1', W, H, ZC)[k]
    assert not np.array_equal(c0.tvec, c2.tvec)


def test_p1_reports_the_tape_error_of_a_held_out_mark(scene):
    marks, cam, img = scene
    i = 17
    m2 = dict(marks)
    m2[i] = (marks[i][0] + 0.003, marks[i][1], SIDE)       # tape 3 mm off, image true
    groups = kc.make_groups(sorted(img))
    p1 = kc.p1_clip(img, m2, groups, kc.group_cams(img, m2, groups, 'B1', W, H, ZC))
    assert p1[i][0] == pytest.approx(-0.003, abs=2e-5)
    others = [np.linalg.norm(v) for j, v in p1.items() if j != i]
    assert max(others) < 2e-3                               # i is in their training set, small leak only


def _uso(img_a, img_b, frames_b, marks, groups, tap=None, frames_a=None, fps=60.0):
    frames_a = frames_a or [img_a] * 20
    frames_b = frames_b or [img_b] * 20
    gc = kc.GroupCams(img_a, marks, groups, 'B1', W, H, ZC)
    ta = [k / fps for k in range(len(frames_a))]
    tb = [k / fps for k in range(len(frames_b))]
    return kc.u_so_family(ta, frames_a, img_b, tb, frames_b, marks, groups, gc, tap)


def test_comparators_leave_out_the_check_group():
    """KC1-1, KC3-1: chain k compares mark j by the camera without k and without the group of j; chain xe without
    the group of j only; no chain-k comparator uses group k."""
    groups = [[1, 5], [2, 6], [3, 7], [4, 8]]
    c = kc.comparators(groups)
    assert set(c) == {'xe', '0', '1', '2', '3'}
    assert c['xe'] == {t: frozenset([t]) for t in range(4)}
    for k in range(4):
        assert set(c[str(k)]) == set(range(4)) - {k}
        assert all(E == frozenset([k, t]) for t, E in c[str(k)].items())


def test_u_so_sees_a_shift_in_clip_b(scene):
    """A mark moved 1.5 mm (metres, projected) in H_b: every chain's static term sees it."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    true = dict(marks)
    true[5] = (marks[5][0] + 0.0015, marks[5][1], SIDE)
    res, _ = _uso(img, image_of(cam, true), [], marks, groups)
    assert res['xe']['tinh_m'] == pytest.approx(0.0015, abs=2e-5)
    g5 = next(k for k, g in enumerate(groups) if 5 in g)
    assert all(res[str(k)]['tinh_m'] == pytest.approx(0.0015, abs=3e-5) for k in range(len(groups)) if k != g5)


def _taped(marks, seed=3, sd=0.0015):
    """Tape coordinates with errors (the image stays true): fits on different mark sets disagree."""
    rng = np.random.default_rng(seed)
    return {i: (x + rng.normal(0, sd), y + rng.normal(0, sd), s) for i, (x, y, s) in marks.items()}


def test_u_so_counts_the_run_mark_sets(scene):
    """KC1-2, KC1-8: with tape errors, a smaller fit set (marks hidden by the car) moves the prediction although
    nothing moved: u_so with the run sets > 0 although H_b = H_a; without run sets it is 0."""
    marks0, cam, img = scene
    marks = _taped(marks0)
    groups = kc.make_groups(sorted(img))
    res0, _ = _uso(img, img, [], marks, groups)
    assert res0['xe']['u_so_m'] < 1e-6
    hidden = [{1, 2, 3, 8, 9, 10, 15, 16, 17, 30, 31}, {20, 21, 22, 27, 28, 29}]
    tap = [sorted(set(img) - h) for h in hidden]
    res, info = _uso(img, img, [], marks, groups, {'xe': tap})
    assert info['co_tap_luot'] and res['xe']['u_so_m'] > 2e-4
    # KCC1-2: the sets of chain xe never enter another chain's u_so
    assert all(res[c]['u_so_m'] == pytest.approx(res0[c]['u_so_m'], abs=1e-12) for c in res if c != 'xe')
    assert any(x['tap'] != sorted(set(img) - set(groups[x['nhom_dich']])) for x in res['xe']['tap_tinh'])


def test_u_so_window_and_raw_terms(scene):
    """Filtered term: largest |window median - clip mean| (0.5 s window, >= 10 samples); raw term: largest single
    frame. A 1 mm move of mark 12 over 30 frames shows in both; a 1-frame 2 mm jump only in the raw term."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    moved = dict(marks)
    moved[12] = (marks[12][0], marks[12][1] + 0.001, SIDE)
    jump = dict(marks)
    jump[12] = (marks[12][0], marks[12][1] + 0.002, SIDE)
    frames = [img] * 5 + [image_of(cam, moved)] * 30 + [img] * 20 + [image_of(cam, jump)] + [img] * 20
    res, _ = _uso(img, img, frames, marks, groups)
    assert res['xe']['cua_so_m'] == pytest.approx(0.001, abs=3e-5)
    assert res['xe']['tho_m'] == pytest.approx(0.002, abs=3e-5)
    assert res['xe']['u_so_tho_m'] >= res['xe']['u_so_m']


def test_window_restarts_after_a_gap():
    """A window holds only the last continuous sighting: a gap > 0.1 s restarts it; fewer than 10 samples: none."""
    t = [k / 60 for k in range(40)]
    seq = [np.array([0.0, 0.0])] * 15 + [None] * 10 + [np.array([1.0, 0.0])] * 15
    med = kc.windows_by_time(t, seq)
    assert med and all(m[0] in (0.0, 1.0) for m in med)          # never a median across the gap
    assert kc.windows_by_time(t[:9], seq[:9]) == []


def test_u_so_flags_sets_it_cannot_compute(scene):
    """KC2-2: every mark in half the H_b frames but never all at once: no frame holds a whole set, so no window:
    flagged per (chain, target group, set), never replaced by a smaller set; kiem_h shows 0 % at once."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    half_a = {i: c for i, c in img.items() if i % 2}
    half_b = {i: c for i, c in img.items() if not i % 2}
    frames = [half_a, half_b] * 20
    res, _ = _uso(img, img, frames, marks, groups)
    assert all(any(x['vi'] in ('tho', 'cua_so') for x in c['khong_tinh_duoc']) for c in res.values())
    assert all(c['tap_tinh'] == [] or all(x['tap'] for x in c['tap_tinh']) for c in res.values())
    r = kc.kiem_h(frames, sorted(img))
    assert r['ti_le_cung_luc'] == 0.0 and min(r['ti_le_moc'].values()) == 0.5


def test_u_so_without_baseline_refuses(scene):
    """KC3-2: a mark never seen whole-set in H_a has no baseline: the whole session is refused."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    part = {i: c for i, c in img.items() if i != 3}
    with pytest.raises(kc.KCError, match='baseline'):
        _uso(img, img, [], marks, groups, frames_a=[part] * 20)


def test_session_model_lock():
    s = dict(kind='c7_session', chosen='B1', marks={}, fit_ids=[], ref_corners_px={}, width=W, height=H)
    with pytest.raises(kc.KCError, match='model_khoa'):
        kc.clip_from_session(s, 'B1')                       # chosen by cross-validation, not forced
    kc.clip_from_session(dict(s, model_khoa='B1'), 'B1')
    kc.clip_from_session(s, 'B1', thu=True)
    with pytest.raises(kc.KCError, match='locked model'):
        kc.clip_from_session(dict(s, chosen='B2'), 'B1', thu=True)


def _moc_rows(marks):
    return {i: dict(id=str(i), x_cm=repr(x * 100), y_cm=repr(y * 100), canh_den_mm=repr(s * 1000), sigma_x_mm='0.8',
                    sigma_y_mm='0.9', canh_nguon='do') for i, (x, y, s) in marks.items()}


@pytest.mark.parametrize('bad', ['layout', 'no_sigma', 'side', 'moved', 'absent'])
def test_check_moc_refuses(scene, bad):
    marks = scene[0]
    rows = _moc_rows(marks)
    kc.check_moc(rows, marks, sorted(marks))
    r = rows[25]
    if bad == 'layout':
        r.update(canh_den_mm='100.0', canh_nguon='', sigma_x_mm='', sigma_y_mm='')
    elif bad == 'no_sigma':
        r['sigma_x_mm'] = ''
    elif bad == 'side':
        r['canh_nguon'] = 'thiet_ke'
    elif bad == 'moved':
        r['x_cm'] = repr(marks[25][0] * 100 + 0.1)
    elif bad == 'absent':
        del rows[25]
    with pytest.raises(kc.KCError, match='KC1-9'):
        kc.check_moc(rows, marks, sorted(marks))


def _own(a3=None):
    return dict(P1=dict(bo_nhom_max_m=0.0034, tung_khung_max_m=0.003),
                A3=a3 or _a3_conv(max_deg=0.15, quy_ve_ma_noc=True), P7=dict(lech_max_m=0.0006),
                A5=dict(lech_max_deg=0.4), _src=('kc.json', 'a' * 64))


def _moc_dich(**over):
    """The U_moc block of CODE_CAM as c8b_u v3 (198d2e9) reads it."""
    b = dict(U_moc_xy_m=0.0012, U_moc_psi_deg=0.05, phuong_phap_duyet='Liam 19/11 (test)', tien_de_buoi='dat',
             nguon=dict(frames={'r1': 'a' * 64}, lich='1' * 64, tap_moc='2' * 64, u_so='5' * 64, session='0' * 64,
                        car_pose=dict(tag_offset=None, side_m=0.1445, h_m=0.12), bo_lay_mau='c7_u_moc_test'))
    b.update(over)
    return b


def _parts():
    blocks = dict(P3=dict(vi_tri_max_m=0.001), A2=dict(goc_max_deg=0.3), than_dich=dict(ghi='thong tin'),
                  day_doi=dict(dn_m=0.01, dzc_m=0.02, kappa=0.12 / 2.632), do_cao=dict(rho_max_m=1.6, H_m=2.632, h_m=0.12),
                  bo_cham=dict(tag_offset=None, side_m=0.1445), moc_dich=_moc_dich())
    return {n: (b, f'{n}.json', 'b' * 64) for n, b in blocks.items()}


def test_build_cam_u_ok():
    out = kc.build_cam_u(_own(), _parts())
    assert out['version'] == 'c8_cam_u_vao_v3'
    assert out['moc_dich']['nguon']['u_so'] == '5' * 64            # CAM's sources are kept, not replaced
    assert set(out) - {'version', 'nguon'} == set(kc.CAM_BLOCKS)
    assert {n['khoi'] for n in out['nguon']} == set(kc.CAM_BLOCKS)
    assert all('nguon' in out[b] for b in kc.CAM_BLOCKS)
    assert 'moc_dich' not in kc.OWN_BLOCKS
    assert out['bo_cham']['tag_offset'] is None


@pytest.mark.parametrize('bad', ['missing_block', 'missing_key', 'negative', 'nan', 'bool', 'unknown', 'own_as_part',
                                 'offset_missing', 'offset_short', 'offset_nan', 'side_zero'])
def test_build_cam_u_refuses(bad):
    own, parts = _own(), _parts()
    if bad == 'missing_block':
        del parts['do_cao']
    elif bad == 'missing_key':
        del parts['A2'][0]['goc_max_deg']
    elif bad == 'negative':
        parts['day_doi'][0]['dn_m'] = -0.01
    elif bad == 'nan':
        own['P1']['tung_khung_max_m'] = float('nan')
    elif bad == 'bool':
        parts['P3'][0]['vi_tri_max_m'] = True
    elif bad == 'unknown':
        parts['la'] = ({}, 'la.json', 'c' * 64)
    elif bad == 'own_as_part':
        parts['A3'] = (dict(max_deg=0), 'x.json', 'd' * 64)
    elif bad == 'offset_missing':
        del parts['bo_cham'][0]['tag_offset']
    elif bad == 'offset_short':
        parts['bo_cham'][0]['tag_offset'] = [0.0, 0.0]
    elif bad == 'offset_nan':
        parts['bo_cham'][0]['tag_offset'] = [0.0, float('nan'), 0.0]
    elif bad == 'side_zero':
        parts['bo_cham'][0]['side_m'] = 0.0
    with pytest.raises(kc.KCError):
        kc.build_cam_u(own, parts)


def test_c8b_u_reads_the_export():
    """KCC1-13: the file written by build_cam_u, read by CODE_C8B's c8b_u (hil feat/c8b-u d338af4: check_cam,
    nguong + 2 u_so, A3.quy_ve_ma_noc required); runs where c8b_u v2 with check_cam is importable (integration
    branch), skipped before. (1) Our export is refused while A3 is not carried to the car tag (no quy_ve_ma_noc);
    (2) with the flag set here ONLY to reach the formula, the mark-shift part of P1 is nguong + 2 u_so (u 0.3 mm:
    1.6 mm, not 1.3 mm)."""
    with pytest.raises(kc.KCError, match='quy_ve_ma_noc'):              # KCJ1-4: the writer refuses it first
        kc.build_cam_u(_own(dict(max_deg=0.15, rms_deg=0.07)), _parts())
    CU = pytest.importorskip('c8b_u')
    if getattr(CU, 'CAM_VERSION', None) != kc.CAM_U_VERSION or not hasattr(CU, 'check_cam'):
        pytest.skip('c8b_u of another camera-file version (before d338af4)')
    own = dict(_own(), A3=_a3_conv(quy_ve_ma_noc=True))                  # approved (Liam 18/11): the real export path
    cam = json.loads(json.dumps(kc.build_cam_u(own, _parts())))
    CU.check_cam(cam)
    tk = {"version": "c8b_thuoc_kep_v2", "cao_4_goc_mm": [120.2, 120.0, 119.8, 120.1], "e_mm": 0.1, "s_mm": 96.4,
          "canh": {"truoc": [10.3, 10.1], "phai": [10.2, 10.0], "sau": [10.1, 10.3], "trai": [10.0, 10.2]},
          "tam_than_deg": 0.05, "tam_tam_mm": [3.0, -4.0], "tam_tam_dung_sai_mm": 0.5, "ban_ve_nguon": "test"}
    xo = {"version": "c8b_xoay_v1", "q_tag_m": [0.003, -0.004], "du_m": 0.0007}
    tb = dict(ga_ok=True, d_p_quay_m=0.0007, d_psi_quay_deg=0.25)
    up, p, upsi, s = CU.u_p_u_psi(cam, tb, tk, xo)
    assert p['P1_anh_san'] == pytest.approx(max(0.0034, 0.003) + 0.0012)          # v3: + U_moc_xy_m of CODE_CAM
    assert s['A3_huong_vung'] == pytest.approx(0.16) and s['A5_nhieu_khung'] == pytest.approx(0.4)
    assert p['P7_nhieu_khung'] == pytest.approx(0.0006)


def test_noise_clip_finds_the_largest_deviation(scene):
    """P7, A5: frames with corner jitter; one frame moves mark m by 1 mm and turns it 0.5 deg on the floor:
    lech_max of m = its deviation from its own mean over the frames (known exactly from the jitter-free truth)."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    cams = kc.group_cams(img, marks, groups, 'B1', W, H, ZC)
    m, n_fr = 12, 20
    dets = []
    for f in range(n_fr):
        true = dict(marks)
        if f == 7:
            x, y, sd = marks[m]
            c = cp.mark_corners_floor(x + 0.001, y, sd)
            a = math.radians(0.5)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            c = (c - [x + 0.001, y]) @ R.T + [x + 0.001, y]
            d = image_of(cam, true)
            d[m] = cam.project(np.column_stack([c, np.zeros(4)]))
        else:
            d = image_of(cam, true)
        dets.append(d)
    out = kc.noise_clip(dets, marks, groups, cams)
    assert out[m]['n'] == n_fr
    assert out[m]['lech_max_m'] == pytest.approx(0.001 * (1 - 1 / n_fr), abs=2e-5)
    assert out[m]['lech_max_deg'] == pytest.approx(0.5 * (1 - 1 / n_fr), abs=5e-3)
    # m is in the pose fit of the other groups: its move in frame 7 pulls their pose a little (as in a real frame)
    assert max(r['lech_max_m'] for i, r in out.items() if i != m) < 2e-4


def test_write_new_never_replaces(tmp_path):
    p = tmp_path / 'a.json'
    kc.write_new(str(p), {'x': 1})
    with pytest.raises(kc.KCError):
        kc.write_new(str(p), {'x': 2})
    assert json.loads(p.read_text())['x'] == 1


C7 = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'CLAUDE_THESIS', '05_DATA', 'results', 'C7_VIDEO'))


@pytest.mark.skipif(not os.path.isdir(os.path.join(C7, '2026-11-16_buoi_thu', 'calib0')), reason='16/11 data absent')
def test_regression_trial_16_11():
    """A3 on calib0 / calib1 (24 marks, B1) = the CODE_CAM trial (thu_kiem_cheo_A3.py, 7a9f0368):
    RMS 0.074 / 0.072, max 0.148 / 0.139 deg."""
    out = {}
    for n in ('calib0', 'calib1'):
        with open(os.path.join(C7, '2026-11-16_buoi_thu', n, 'session.json'), encoding='utf-8') as f:
            s = json.load(f)
        img, marks, w, h, zc = kc.clip_from_session(s, 'B1', thu=True)
        rows = kc.a3_clip(img, marks, 'B1', w, h, zc, l_max=0.8)
        d = np.array([r['dtheta_deg'] for r in rows])
        out[n] = (len(rows), float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d))))
    assert out['calib0'][0] == out['calib1'][0] == 24
    assert out['calib0'][1] == pytest.approx(0.074, abs=1e-3) and out['calib0'][2] == pytest.approx(0.148, abs=1e-3)
    assert out['calib1'][1] == pytest.approx(0.072, abs=1e-3) and out['calib1'][2] == pytest.approx(0.139, abs=1e-3)


def _kc_file(tmp_path, **over):
    kcres = dict(version=kc.VERSION, buoi='C8_x', model='B1', nhom=[[1, 5], [2, 6]], thu=False,
                 sessions={n: dict(file=n, sha256='0' * 64) for n in kc.CLIPS_REQUIRED},
                 moc_dich=dict(nguong_m=0.001, u_so_m=0.0003, buoi='C8_x'),
                 **{b: dict(blk, buoi='C8_x') for b, blk in _own().items() if b != '_src'})
    kcres.update(u_so_sha256='5' * 64, moc=[dict(file='moc.csv', sha256='6' * 64)], hang_so_sha256='4' * 64)
    kcres.update(over)
    p = tmp_path / 'kiem_cheo.json'
    p.write_text(json.dumps(kcres), encoding='utf-8')
    return p


def _xuat_args(tmp_path, kcp, tk_over=None, part_buoi='C8_x', buoi='C8_x', td_over=None, md_over=None):
    tk = dict(version=kc.TUNG_KHUNG_VERSION, buoi='C8_x', model='B1', nhom=[[1, 5], [2, 6]], cham_lai=True,
              tung_khung_max_m=0.003, kiem_cheo_sha256=kc.sha256_file(str(kcp)), u_so_sha256='5' * 64,
              session_sha256='0' * 64, moc=[dict(file='moc.csv', sha256='6' * 64)], hang_so_sha256='4' * 64,
              hoan_tat=True, luot_chua_co=[], mau_thuan=[], luot_thieu=['r3'], luot_thieu_neo=[],
              luot={'r1': dict(nguon=dict(frames_sha256='a' * 64))})
    tk.update(tk_over or {})
    tkp = tmp_path / 'tk.json'
    tkp.write_text(json.dumps(tk), encoding='utf-8')
    td = dict(format='tien_de_buoi_v1', session_id='C8_x', ket_qua='dat',
              sha256=dict(kiem_cheo=kc.sha256_file(str(kcp)), u_so='5' * 64, tung_khung=kc.sha256_file(str(tkp))))
    td.update(td_over or {})
    tdp = tmp_path / 'td.json'
    tdp.write_text(json.dumps(td), encoding='utf-8')
    mdp = tmp_path / 'md.json'
    mdp.write_text(json.dumps(dict(_moc_dich(**(md_over or {})), buoi='C8_x')), encoding='utf-8')
    args = ['xuat', '--buoi', buoi, '--kiem-cheo', str(kcp), '--tung-khung', str(tkp), '--out', str(tmp_path / 'o.json'),
            '--tien-de', str(tdp), '--moc-dich', str(mdp)]
    for n, (blk, _, _) in _parts().items():
        if n == 'moc_dich':
            continue
        q = tmp_path / f'{n}.json'
        q.write_text(json.dumps(dict(blk, buoi=part_buoi)), encoding='utf-8')
        args += ['--phan', f'{n}={q}']
    return args


def test_xuat_ok(tmp_path):
    assert kc.main(_xuat_args(tmp_path, _kc_file(tmp_path))) == 0
    out = json.loads((tmp_path / 'o.json').read_text(encoding='utf-8'))
    assert out['P1']['tung_khung_max_m'] == 0.003 and out['version'] == 'c8_cam_u_vao_v3'
    assert out['P1']['luot_thieu_kiem'] == ['r3']                     # KCC1-11: runs short of checks stay marked
    md = out['moc_dich']                                             # S8-6: CAM's U_moc + tien_de, u_so as information
    assert md['U_moc_xy_m'] == 0.0012 and md['tien_de_buoi'] == 'dat' and md['u_so_kiem_cheo']['u_so_m'] == 0.0003


@pytest.mark.parametrize('bad', ['td_format', 'td_session', 'td_kc_sha', 'td_u_so', 'td_tk_sha', 'md_u_so', 'md_no_lich',
                                 'md_tien_de_other', 'md_no_method', 'tk_thieu_neo', 'md_frames', 'md_session',
                                 'md_no_bo_lay_mau', 'md_car_pose_h'])
def test_xuat_v3_moc_dich_must_chain(tmp_path, bad):
    """S8-6: the tien_de and the U_moc block must be of this kiem_cheo result; anchors missing in a run refused."""
    kcp = _kc_file(tmp_path)
    td = {'td_format': dict(format='x'), 'td_session': dict(session_id='C8_y'), 'td_kc_sha': dict(sha256=dict(
        kiem_cheo='f' * 64, u_so='5' * 64)), 'td_u_so': dict(sha256=dict(kiem_cheo=kc.sha256_file(str(kcp)), u_so='7' * 64)),
        'td_tk_sha': dict(sha256=dict(kiem_cheo=kc.sha256_file(str(kcp)), u_so='5' * 64, tung_khung='e' * 64))}
    md = {'md_u_so': dict(nguon=dict(_moc_dich()['nguon'], u_so='7' * 64)),
          'md_frames': dict(nguon=dict(_moc_dich()['nguon'], frames={'r1': 'b' * 64})),
          'md_session': dict(nguon=dict(_moc_dich()['nguon'], session='9' * 64)),
          'md_no_bo_lay_mau': dict(nguon=dict(_moc_dich()['nguon'], bo_lay_mau=' ')),
          'md_car_pose_h': dict(nguon=dict(_moc_dich()['nguon'], car_pose=dict(tag_offset=None, side_m=0.1445, h_m=0.10))),
          'md_no_lich': dict(nguon={k: v for k, v in _moc_dich()['nguon'].items() if k != 'lich'}),
          'md_tien_de_other': dict(tien_de_buoi='thieu_du_lieu'), 'md_no_method': dict(phuong_phap_duyet=None)}
    args = _xuat_args(tmp_path, kcp, tk_over=dict(luot_thieu_neo=['r2']) if bad == 'tk_thieu_neo' else None,
                      td_over=td.get(bad), md_over=md.get(bad))
    assert kc.main(args) == 2
    assert not (tmp_path / 'o.json').exists()


@pytest.mark.parametrize('bad', ['thu', 'no_Hm', 'tk_buoi', 'tk_nhom', 'tk_model', 'tk_sha', 'tk_not_rescored',
                                 'part_buoi', 'arg_buoi', 'tk_u_so', 'tk_session', 'tk_moc', 'tk_hang_so',
                                 'tk_not_complete', 'tk_contradiction'])
def test_xuat_refuses_mixed_inputs(tmp_path, bad, capsys):
    """KC1-10: results of another session, grouping, model, kiem_cheo file, or before the rescoring are refused."""
    over = dict(thu=True) if bad == 'thu' else {}
    if bad == 'no_Hm':
        over['sessions'] = {n: dict(file=n, sha256='0' * 64) for n in ('H_a', 'H_b', 'H_z')}
    kcp = _kc_file(tmp_path, **over)
    tk_over = {'tk_buoi': dict(buoi='C8_y'), 'tk_nhom': dict(nhom=[[1], [2]]), 'tk_model': dict(model='B2'),
               'tk_sha': dict(kiem_cheo_sha256='f' * 64), 'tk_not_rescored': dict(cham_lai=False),
               'tk_u_so': dict(u_so_sha256='7' * 64), 'tk_session': dict(session_sha256='9' * 64),
               'tk_moc': dict(moc=[dict(file='moc.csv', sha256='8' * 64)]),
               'tk_hang_so': dict(hang_so_sha256='3' * 64),
               'tk_not_complete': dict(hoan_tat=False, luot_chua_co=['r9']),
               'tk_contradiction': dict(mau_thuan=[dict(clip='H_m', moc=25)])}.get(bad)
    args = _xuat_args(tmp_path, kcp, tk_over, part_buoi='C8_y' if bad == 'part_buoi' else 'C8_x',
                      buoi='C8_y' if bad == 'arg_buoi' else 'C8_x')
    assert kc.main(args) == 2
    assert not (tmp_path / 'o.json').exists()


# ---------------------------------------------------------------- tung_khung (KC1-4, KC1-6, KC3-2, KCC1-3..12)
GR = [[1, 5], [2, 6], [3, 7], [4, 8]]
ALL8 = list(range(1, 9))


def _chains(anchor_sets=(ALL8,)):
    """u_so.json 'chuoi' with every comparator's fit set of the given anchor sets computed (tap_tinh)."""
    out = {}
    gmap = {i: k for k, g in enumerate(GR) for i in g}
    for c, cmp_ in kc.comparators(GR).items():
        tinh = []
        for t, E in cmp_.items():
            for A in anchor_sets:
                fit = sorted(i for i in A if gmap[i] not in E)
                if dict(nhom_dich=t, tap=fit) not in tinh:
                    tinh.append(dict(nhom_dich=t, tap=fit))
        out[c] = dict(tap_tinh=tinh, khong_tinh_duoc=[], nguong_m=0.0013, nguong_tho_m=0.0015)
    return out


def _frames(n=100, fps=60.0, invalid=(), anchors=None, seen=None):
    rows = []
    for k in range(n):
        r = dict(idx=str(k), t_s=repr(k / fps), valid='0' if k in invalid else '1',
                 moc_thay=';'.join(str(i) for i in (seen or {}).get(k, (anchors or {}).get(k, ALL8))))
        for c in kc.comparators(GR):
            r[f'moc_neo_{c}'] = ';'.join(str(i) for i in (anchors or {}).get(k, ALL8))
        rows.append(r)
    return rows


def _khung(n=100, big=None, skip=()):
    rows = [dict(khung=str(k), moc='5', nhom='0', ex='0.001', ey='0.0') for k in range(n) if k not in skip]
    if big:
        rows.append(dict(khung=str(big[0]), moc=str(big[1]), nhom=str(next(g for g, m in enumerate(GR) if big[1] in m)),
                         ex=repr(big[2]), ey='0.0'))
    return rows


def test_centre_is_the_back_projected_image_mean():
    """KCC1-1: with a tilted, distorted camera the two centre definitions differ; ours is CAM's."""
    cam = true_cam(k=-0.08, tilt=(0.25, -0.2))
    c4 = cam.project(np.column_stack([cp.mark_corners_floor(1.5, 1.0, 0.1445), np.zeros(4)]))
    want = cam.pix2floor(c4.mean(axis=0, keepdims=True), 0.0)[0]
    other = cam.pix2floor(c4, 0.0).mean(axis=0)
    assert np.linalg.norm(want - other) > 1e-5
    assert np.allclose(kc.centre_floor(cam, c4), want, atol=1e-12)


def test_frame_check_raw_max_on_valid_frames_only():
    """KC1-4: the raw error of a single frame counts (no median); invalid frames and a mark after its chain-xe t_bd do
    not; the t_bd of other chains does not end the target (KCC1-7), whatever the order of the records."""
    ch = _chains()
    r = kc.run_frame_check(_frames(), _khung(big=(40, 5, 0.005)), {}, GR, ch)
    assert r['max_m'] == pytest.approx(0.005) and not r['thieu']
    r = kc.run_frame_check(_frames(invalid={40}), _khung(big=(40, 5, 0.005)), {}, GR, ch)
    assert r['max_m'] == pytest.approx(0.001)
    recs = [dict(chuoi='xe', id=5, t_bd=30 / 60), dict(chuoi='1', id=5, t_bd=90 / 60)]
    for order in (recs, recs[::-1]):
        r = kc.run_frame_check(_frames(), _khung(big=(40, 5, 0.005)), dict(moc_loai=order), GR, ch)
        assert r['max_m'] == pytest.approx(0.001)
    r = kc.run_frame_check(_frames(), _khung(big=(40, 5, 0.005)), dict(moc_loai=[dict(chuoi='1', id=5, t_bd=0.1)]),
                           GR, ch)
    assert r['max_m'] == pytest.approx(0.005)
    with pytest.raises(kc.KCError, match='chuoi'):
        kc.run_frame_check(_frames(), _khung(), dict(moc_loai=[dict(id=5, t_bd=0.1)]), GR, ch)


@pytest.mark.parametrize('bad', ['nan', 'empty', 'unknown_frame', 'wrong_group'])
def test_frame_check_refuses_bad_rows(bad):
    """KCC1-6: a check row must be finite, of a known frame, mark and its own group."""
    rows = _khung()
    rows[3] = dict(rows[3], **{'nan': dict(ex='nan'), 'empty': dict(ey=''), 'unknown_frame': dict(khung='999'),
                               'wrong_group': dict(nhom='2')}[bad])
    with pytest.raises(kc.KCError, match='KCC1-6'):
        kc.run_frame_check(_frames(), rows, {}, GR, _chains())


def test_frame_check_no_rows_or_all_targets_ended():
    """KCC1-6: no accepted check row is not a P1 of 0: max None and the run is short of checks."""
    r = kc.run_frame_check(_frames(), [], {}, GR, _chains())
    assert r['max_m'] is None and r['thieu']
    r = kc.run_frame_check(_frames(), _khung(), dict(moc_loai=[dict(chuoi='xe', id=5, t_bd=0.0)]), GR, _chains())
    assert r['max_m'] is None and r['thieu']


@pytest.mark.parametrize('skip, thieu', [(set(range(10)), False), (set(range(11)), True),
                                         (set(range(20, 80, 2)), True), (set(range(30, 89, 10)), False), ({1}, False)])
def test_frame_check_coverage_both_sides(skip, thieu):
    """KC1-6: at most 10 % of the valid frames without an accepted check row: 10 of 100 ok, 11 not."""
    r = kc.run_frame_check(_frames(), _khung(skip=skip), {}, GR, _chains())
    assert r['thieu'] is thieu


def test_frame_check_gap_over_one_second():
    r = kc.run_frame_check(_frames(n=200), _khung(200, skip=set(range(50, 112))), {}, GR, _chains())
    assert r['khoang_mat_max_s'] > 1.0 and r['thieu']
    r = kc.run_frame_check(_frames(n=700), _khung(700, skip=set(range(50, 109))), {}, GR, _chains())
    assert r['khoang_mat_max_s'] < 1.0 and not r['thieu']


def test_frame_check_sets_must_be_computed_in_every_frame():
    """KCC1-3, KCC1-4: a comparator fit set that u_so never computed, even in an invalid frame, makes the run short
    of checks; sets u_so computed pass."""
    small = [1, 2, 3, 4, 5, 6, 7]
    r = kc.run_frame_check(_frames(invalid={10}, anchors={10: small}), _khung(), {}, GR, _chains())
    assert r['thieu'] and any('KCC1-3' in x for x in r['ly_do'])
    r = kc.run_frame_check(_frames(invalid={10}, anchors={10: small}), _khung(), {}, GR, _chains((ALL8, small)))
    assert not r['thieu']


def _write_csv(path, rows):
    import csv as _csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _tk_setup(tmp_path, bad=None, theo=None):
    us = tmp_path / 'u_so.json'
    us.write_text(json.dumps(dict(version=kc.U_SO_VERSION, chuoi=_chains())), encoding='utf-8')
    kcp = tmp_path / 'kiem_cheo.json'
    kcp.write_text(json.dumps(dict(version=kc.VERSION, buoi='C8_x', model='B1', nhom=GR, thu=bad == 'thu',
                                   sessions=dict(H_a=dict(file='a', sha256='1' * 64)), moc=[], hang_so_sha256='4' * 64,
                                   u_so_sha256=kc.sha256_file(str(us)), theo_thoi_gian=theo or {})), encoding='utf-8')
    plan = tmp_path / 'plan.txt'
    plan.write_text('# luot\nr1\n' + ('r2\n' if bad == 'plan_more' else ''), encoding='utf-8')
    d = tmp_path / 'r1'
    d.mkdir()
    _write_csv(d / 'frames.csv', _frames())
    _write_csv(d / 'kiem_cheo_khung.csv', _khung(big=(40, 5, 0.004)))
    tr = dict(u_so_sha256=kc.sha256_file(str(us)), session_sha256='1' * 64, model_khoa='B1', nhom=GR,
              moc_loai=[], cham_lai=1, hang_so_sha256='4' * 64,
              nguong_dung={c: dict(nguong_m=0.0013, nguong_tho_m=0.0015) for c in kc.comparators(GR)},
              frames_sha256=kc.sha256_file(str(d / 'frames.csv')),
              kiem_cheo_khung_sha256=kc.sha256_file(str(d / 'kiem_cheo_khung.csv')))
    if bad == 'u_so':
        tr['u_so_sha256'] = '2' * 64
    elif bad == 'session':
        tr['session_sha256'] = '3' * 64
    elif bad == 'model':
        tr['model_khoa'] = 'B2'
    elif bad == 'nhom':
        tr['nhom'] = [[1, 2], [3, 4], [5, 6], [7, 8]]
    elif bad == 'not_rescored':
        tr['moc_loai'] = [dict(chuoi='xe', id=5, t_bd=1.0)]
    elif bad == 'nguong':
        tr['nguong_dung']['2']['nguong_m'] = 0.0012
    elif bad == 'hang_so':
        tr['hang_so_sha256'] = '5' * 64
    elif bad == 'frames_swapped':
        _write_csv(d / 'frames.csv', _frames(invalid={3}))
    elif bad == 'khung_swapped':
        _write_csv(d / 'kiem_cheo_khung.csv', _khung())
    (d / 'track.json').write_text(json.dumps(tr), encoding='utf-8')
    out = tmp_path / 'tk.json'
    args = ['tung_khung', '--buoi', 'C8_x', '--kiem-cheo', str(kcp), '--u-so', str(us), '--ke-hoach', str(plan),
            '--luot', f'r1={d}', '--out', str(out)]
    return args, out


@pytest.mark.parametrize('bad', [None, 'u_so', 'session', 'model', 'nhom', 'not_rescored', 'thu', 'nguong', 'hang_so',
                                 'frames_swapped', 'khung_swapped'])
def test_tung_khung_cli_checks_the_track(tmp_path, bad):
    """KC1-10, KCC1-8, KCC1-9: the track must name the u_so.json, H_a session, model, groups, constants file and the
    thresholds it used, and the two CSV it was read with; rescored when marks were excluded."""
    args, out = _tk_setup(tmp_path, bad)
    rc = kc.main(args)
    if bad is None:
        assert rc == 0
        t = json.loads(out.read_text(encoding='utf-8'))
        assert t['tung_khung_max_m'] == pytest.approx(0.004) and t['luot_thieu'] == [] and t['hoan_tat']
    else:
        assert rc == 2 and not out.exists()


def test_tung_khung_not_complete_without_every_planned_run(tmp_path):
    """KCC1-11: a planned run missing: written but not complete (xuat refuses it)."""
    args, out = _tk_setup(tmp_path, 'plan_more')
    assert kc.main(args) == 0
    t = json.loads(out.read_text(encoding='utf-8'))
    assert not t['hoan_tat'] and t['luot_chua_co'] == ['r2']


def test_tung_khung_contradiction_at_a_later_clip(tmp_path):
    """KCC1-12: a mark whose own shift at H_m is above the chain-xe threshold but that no run excluded."""
    args, out = _tk_setup(tmp_path, theo={'H_m': dict(rieng_m={'5': 0.0020, '6': 0.0005})})
    assert kc.main(args) == 0
    t = json.loads(out.read_text(encoding='utf-8'))
    assert t['mau_thuan'] == [dict(luot='r1', clip='H_m', moc=5)] and not t['hoan_tat']


def test_session_groups_must_partition_the_fit_marks():
    """KCC1-14."""
    with pytest.raises(kc.KCError, match='KCC1-14'):
        kc.session_groups(dict(kiem_cheo=dict(nhom=[[1, 5], [2, 6], [3], [4, 8]])), ALL8)
    assert kc.session_groups(dict(kiem_cheo=dict(nhom=GR)), ALL8) == GR


def test_a3_limits_are_exported_in_full(scene):
    """KCC1-15: limits (a)-(f), no claim of conservatism, not a car-tag heading bound."""
    marks, cam, img = scene
    out = kc.a3_summary({'H_a': kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=0.8)})
    assert len(out['gioi_han']) == 6 and out['gioi_han'][0].startswith('(a)')
    assert 'khong chung minh bao thu' in out['gioi_han'][0] and 'quy_ve_ma_noc' in out['loai']
    assert 'quy_ve_ma_noc' not in out


def test_cut_clip_is_a_trial():
    """KCC1-10."""
    assert kc.thu_flag(False, 20) and kc.thu_flag(True, None) and not kc.thu_flag(False, None)


def test_u_so_needs_h_b_frames(scene):
    """KCC1-5: without H_b frames the noise terms cannot be computed: refused, not 0."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    gc = kc.GroupCams(img, marks, groups, 'B1', W, H, ZC)
    with pytest.raises(kc.KCError, match='KCC1-5'):
        kc.u_so_family([0.0] * 20, [img] * 20, img, [], [], marks, groups, gc)


def test_u_so_flags_a_single_target_without_window(scene):
    """KCC1-5: one target mark missing from the H_b frames is flagged even when its group mates have windows."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    j = groups[0][0]
    frames = [img] * 20 + [{i: c for i, c in img.items() if i != j}] * 20
    gc = kc.GroupCams(img, marks, groups, 'B1', W, H, ZC)
    t = [k / 60 for k in range(40)]
    fit_full = sorted(set(img) - set(groups[0]))
    res, _ = kc.u_so_family(t[:20], [img] * 20, img, t, [{i: c for i, c in f.items() if i != j} for f in frames],
                            marks, groups, gc)
    assert any(x.get('moc') == j and x['vi'] in ('tho', 'cua_so') for x in res['xe']['khong_tinh_duoc'])
    assert dict(nhom_dich=0, tap=fit_full) not in res['xe']['tap_tinh']


def test_u_so_raw_counts_without_a_window(scene):
    """KCC2-2: samples too short for a window (fewer than 10 in a row) still give the raw maximum."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    j = groups[0][0]
    jump = dict(marks)
    jump[j] = (marks[j][0] + 0.005, marks[j][1], SIDE)
    seen = image_of(cam, jump)
    blank = {i: c for i, c in img.items() if i != j}
    frames = [seen if k % 12 == 0 else blank for k in range(48)]     # j seen once every 0.2 s: never 10 in a row
    gc = kc.GroupCams(img, marks, groups, 'B1', W, H, ZC)
    t = [k / 60 for k in range(48)]
    res, _ = kc.u_so_family(t[:20], [img] * 20, img, t, frames, marks, groups, gc)
    assert res['xe']['tho_m'] >= 0.005 - 3e-5
    assert any(x.get('moc') == j and x['vi'] == 'cua_so' for x in res['xe']['khong_tinh_duoc'])


def test_frame_check_idle_comparator_not_checked():
    """KCC1-4: a comparator with no target in the frame (its whole group hidden) is not checked."""
    no_g0 = [2, 3, 4, 6, 7]                                           # group 0 = {1, 5} and mark 8 hidden
    computed = _chains()                                              # only the full-set fits computed
    r = kc.run_frame_check(_frames(anchors={k: no_g0 for k in range(100)}),
                           [dict(khung=str(k), moc='6', nhom='1', ex='0.001', ey='0.0') for k in range(100)],
                           {}, GR, computed)
    lost_target0 = [x for x in r['tap_chua_tinh'] if x['nhom_dich'] == 0]
    assert lost_target0 == []


def test_tung_khung_contradiction_needs_the_chain_xe_schedule(tmp_path):
    """KCC1-12: an exclusion of the mark in another chain does not clear a chain-xe contradiction; KCC2-1: a large
    own part at H_b is not a contradiction (H_b sets u_so)."""
    args, out = _tk_setup(tmp_path, theo={'H_m': dict(rieng_m={'5': 0.0020}), 'H_b': dict(rieng_m={'6': 0.004})})
    d = tmp_path / 'r1'
    tr = json.loads((d / 'track.json').read_text(encoding='utf-8'))
    tr['moc_loai'] = [dict(chuoi='1', id=5, t_bd=1.0)]
    tr['cham_lai'] = 2
    (d / 'track.json').write_text(json.dumps(tr), encoding='utf-8')
    assert kc.main(args) == 0
    t = json.loads(out.read_text(encoding='utf-8'))
    assert t['mau_thuan'] == [dict(luot='r1', clip='H_m', moc=5)]


def _fake_session(tmp_path, name, img, marks, video):
    s = dict(kind='c7_session', model_khoa='B1', chosen='B1', width=W, height=H, zc_tape_m=ZC, car_side_m=0.1445,
             tag_h_m=0.119,
             fit_ids=sorted(img), marks={str(i): list(m) for i, m in marks.items()},
             ref_corners_px={str(i): c.tolist() for i, c in img.items()}, video_sha256=kc.sha256_file(str(video)))
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(s), encoding='utf-8')
    return p


def _phan_tich_args(tmp_path, scene, extra_mark=False, cut=False):
    marks, cam, img = scene
    video = tmp_path / 'clip.mov'
    video.write_bytes(b'not a video')
    moc = tmp_path / 'moc.csv'
    rows = ['id,vai_tro,x_cm,y_cm,canh_den_mm,dict,sigma_x_mm,sigma_y_mm,canh_nguon']
    for i, (x, y, sd) in marks.items():
        rows.append(f'{i},moc_san,{x * 100!r},{y * 100!r},{sd * 1000!r},DICT_4X4_50,0.8,0.9,do')
    if extra_mark:
        rows.append('33,moc_giua,10,10,96.4,DICT_4X4_50,0.8,0.9,do')
    moc.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    hs = tmp_path / 'hang_so.json'
    hs.write_text('{}', encoding='utf-8')
    args = ['phan_tich', '--buoi', 'C8_x', '--model', 'B1', '--moc', str(moc), '--hang-so', str(hs), '--L-max', '0.6',
            '--out', str(tmp_path / 'kc')]
    for n in ('a', 'b', 'm', 'z'):
        args += [f'--H-{n}', str(_fake_session(tmp_path, n, img, marks, video)), f'--video-{n}', str(video)]
    if cut:
        args += ['--max-frames', '20']
    return args


def test_phan_tich_cut_clip_is_a_trial_and_not_exported(tmp_path, scene, monkeypatch):
    """KCC1-10 on the production path: --max-frames makes the kiem_cheo result a trial; tung_khung refuses it."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(20)], [img] * 20))
    assert kc.main(_phan_tich_args(tmp_path, scene, cut=True)) == 0
    r = json.loads((tmp_path / 'kc' / 'kiem_cheo.json').read_text(encoding='utf-8'))
    assert r['thu'] is True
    assert kc.main(['tung_khung', '--buoi', 'C8_x', '--kiem-cheo', str(tmp_path / 'kc' / 'kiem_cheo.json'),
                    '--u-so', str(tmp_path / 'kc' / 'u_so.json'), '--ke-hoach', str(tmp_path / 'moc.csv'),
                    '--luot', f'r1={tmp_path}', '--out', str(tmp_path / 'tk.json')]) == 2


def test_phan_tich_full_run_and_missing_measured_mark(tmp_path, scene, monkeypatch):
    """The production path runs end to end (not a trial); a measured floor mark absent from the sessions is refused
    (KCC1-14)."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(20)], [img] * 20))
    assert kc.main(_phan_tich_args(tmp_path, scene)) == 0
    r = json.loads((tmp_path / 'kc' / 'kiem_cheo.json').read_text(encoding='utf-8'))
    assert r['thu'] is False and r['u_so_sha256'] == kc.sha256_file(str(tmp_path / 'kc' / 'u_so.json'))
    t2 = tmp_path / 'second'
    t2.mkdir()
    assert kc.main(_phan_tich_args(t2, scene, extra_mark=True)) == 2
    assert not (t2 / 'kc' / 'kiem_cheo.json').exists()


def test_frame_check_seen_target_out_of_the_anchors_is_checked():
    """KCC3-1: a mark of group 0 still seen (a target) but left out of the anchors: its comparator is active, and
    an uncomputed fit set makes the run short of checks."""
    no_g0 = [2, 3, 4, 6, 7]
    r = kc.run_frame_check(_frames(anchors={k: no_g0 for k in range(100)}, seen={k: no_g0 + [1] for k in range(100)}),
                           [dict(khung=str(k), moc='6', nhom='1', ex='0.001', ey='0.0') for k in range(100)],
                           {}, GR, _chains())
    assert any(x['nhom_dich'] == 0 for x in r['tap_chua_tinh']) and r['thieu']


def _kiem_h_files(tmp_path, scene, session_ids):
    marks, cam, img = scene
    video = tmp_path / 'h.mov'
    video.write_bytes(b'x')
    sess = tmp_path / 's.json'
    sess.write_text(json.dumps(dict(fit_ids=session_ids)), encoding='utf-8')
    moc = tmp_path / 'moc.csv'
    rows = ['id,vai_tro,x_cm,y_cm,canh_den_mm,dict,sigma_x_mm,sigma_y_mm,canh_nguon']
    rows += [f'{i},moc_san,{x * 100!r},{y * 100!r},{s * 1000!r},DICT_4X4_50,0.8,0.9,do' for i, (x, y, s) in marks.items()]
    moc.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return ['kiem_h', '--video', str(video), '--session', str(sess), '--moc', str(moc)]


def test_kiem_h_cli_at_the_floor(tmp_path, scene, monkeypatch):
    """KCC1-14 through the command used at C8: every mark seen at once and in the session: 0; the video sees every
    mark but the session lacks a measured one: 2 (quay lai clip H); fewer than 95 % at once: 2."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(40)], [img] * 40))
    assert kc.main(_kiem_h_files(tmp_path, scene, sorted(img))) == 0
    t2 = tmp_path / 'b'
    t2.mkdir()
    assert kc.main(_kiem_h_files(t2, scene, sorted(img)[1:])) == 2
    part = {i: c for i, c in img.items() if i != 3}
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(40)],
                                                                          [img] * 37 + [part] * 3))
    t3 = tmp_path / 'c'
    t3.mkdir()
    assert kc.main(_kiem_h_files(t3, scene, sorted(img))) == 2


# ---------------------------------------------------------------- design v5 S5-1: A3 carried to the car tag
class _Lin:
    """A camera whose floor map is the linear map I + D (project ignores z): the local-linear field of S5-1 (a)."""

    def __init__(self, D=((0.0, 0.0), (0.0, 0.0))):
        self.M = np.eye(2) + np.asarray(D, float)

    def project(self, P):
        return np.asarray(P, float)[:, :2].copy()

    def pix2floor(self, uv, z=0.0):
        return np.asarray(uv, float).reshape(-1, 2) @ self.M.T


def _yawed(cam, d):
    """cam with the whole world turned by d about the vertical axis: everything it measures is turned by d."""
    c, s = math.cos(d), math.sin(d)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = cam.R @ Rz.T
    C = Rz @ cam.centre
    return cp.Camera(cam.model, cam.f, cam.cx, cam.cy, cam.k, cv_rvec(R), -R @ C)


def test_tag_heading_through_its_own_camera_is_exact(scene):
    """S5-1: rays to z = 0 then a homothety keep the Kabsch angle, so the tag at 0.119 m seen and measured through
    one camera gives its heading back (difference 0), and so does a pair."""
    marks, cam, _ = scene
    import c7_measure as cm
    body = cm.car_corners_body(0.1445)
    for ph in (0.0, 0.3, 1.2):
        c, s = math.cos(ph), math.sin(ph)
        P = body @ np.array([[c, s], [-s, c]]) + np.array([0.4, -0.3])
        pix = cam.project(np.column_stack([P, np.full(4, 0.119)]))
        assert cm.rigid_fit(body, cam.pix2floor(pix, 0.0))[0] == pytest.approx(ph, abs=1e-12)
        assert kc.tag_heading_diff(cam, cam, 0.4, -0.3, ph, 0.1445, 0.119) == pytest.approx(0.0, abs=1e-12)
    assert kc.pair_angle_diff(cam, cam, marks[1], marks[2]) == pytest.approx(0.0, abs=1e-12)


def test_yaw_of_the_camera_turns_tag_and_pair_alike(scene):
    """A rotation of the whole camera about the vertical is a pure omega field: the tag heading and the pair angle
    change by the same angle, the gap is 0."""
    marks, cam, _ = scene
    d = math.radians(0.3)
    g, dpsi, dp = kc.a3_tag_gap(cam, _yawed(cam, d), marks[30], marks[32], 0.1445, 0.119)
    assert dp == pytest.approx(0.3, abs=1e-6) and dpsi == pytest.approx(0.3, abs=1e-6) and g < 1e-6


def test_linear_field_gives_omega_for_the_tag_and_omega_plus_strain_for_a_pair():
    """S5-1 (a): under the map I + D, D = S + omega J, the tag heading changes by the antisymmetric part only, a pair
    along u by omega + n^T S u; two perpendicular pairs add to 2 omega; the gap is exactly tag - pair."""
    a, b, c, w = 0.004, 0.003, -0.002, 0.001
    D = ((a, b - w), (b + w, c))
    I, L = _Lin(), _Lin(D)
    tag = kc.tag_heading_diff(I, L, 0.2, 0.1, 0.4, 0.1445, 0.0)
    assert tag == pytest.approx(math.atan2(2 * w, 2 + a + c), abs=1e-12)
    px = kc.pair_angle_diff(I, L, (0.0, 0.0, SIDE), (0.5, 0.0, SIDE))
    py = kc.pair_angle_diff(I, L, (0.0, 0.0, SIDE), (0.0, 0.5, SIDE))
    assert px == pytest.approx(math.atan2(b + w, 1 + a), abs=1e-12)
    assert py == pytest.approx(math.atan2(w - b, 1 + c), abs=1e-12)
    assert (px + py) / 2 == pytest.approx(w, abs=2e-5)          # 2 omega to first order
    g, dpsi, dp = kc.a3_tag_gap(I, L, (0.0, 0.0, SIDE), (0.5, 0.0, SIDE), 0.1445, 0.0)
    assert dp == pytest.approx(math.degrees(px)) and dpsi == pytest.approx(math.degrees(abs(tag)))
    assert g == pytest.approx(math.degrees(abs(tag - px)), abs=1e-9)   # the same at every point and heading


def test_a3_converted_on_an_exact_scene_and_flag_follows_approval(scene, monkeypatch):
    """Exact data: every pair and gap 0, max_deg is the converted bound, quy_ve_ma_noc false until approved."""
    marks, cam, img = scene
    rows = kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=0.8, tag=(0.1445, 0.119))
    assert rows and all(r['gap_deg'] < 1e-3 and r['quy_deg'] < 2e-3 for r in rows)
    assert kc.A3_QUY_DUYET is True                                       # Liam 18/11 "Duyet (Recommended)"
    assert kc.a3_summary({'H_a': rows})['quy_ve_ma_noc'] is True
    monkeypatch.setattr(kc, 'A3_QUY_DUYET', False)
    out = kc.a3_summary({'H_a': rows})
    assert out['quy_ve_ma_noc'] is False and 'CHUA DUOC LIAM DUYET' in out['loai']
    assert out['max_deg'] == pytest.approx(max(r['quy_deg'] for r in rows)) and 'max_deg_cap' in out
    assert len(out['gioi_han']) == 13 and out['vuong_goc']['thong_tin'] is True and 'KHONG phai can' in out['loai']


def test_a3_converted_bound_covers_the_pair_angle(scene):
    """Tape of mark 30 wrong by 2 mm across the pair 30-32: quy_deg = |dtheta| + gap >= |dtheta|, and max_deg is the
    largest converted value, never below the pair maximum."""
    marks, cam, _ = scene
    m, j = 30, 32
    vt = np.array(marks[j][:2]) - np.array(marks[m][:2])
    n = np.array([-vt[1], vt[0]]) / np.linalg.norm(vt)
    true = dict(marks)
    true[m] = (marks[m][0] + 0.002 * n[0], marks[m][1] + 0.002 * n[1], SIDE)
    rows = kc.a3_clip(image_of(cam, true), marks, 'B1', W, H, ZC, l_max=0.8, tag=(0.1445, 0.119))
    r = next(r for r in rows if (r['i'], r['j']) == (m, j))
    assert r['quy_deg'] == pytest.approx(abs(r['dtheta_deg']) + r['gap_deg']) and r['gap_deg'] > 0
    out = kc.a3_summary({'H_a': rows, 'H_b': rows[:3]})
    assert out['max_deg'] >= out['max_deg_cap'] and out['cap_quy_lon_nhat']['clip'] == 'H_a'


def test_session_tag_is_required_and_the_same():
    good = dict(car_side_m=0.1445, tag_h_m=0.119)
    assert kc.session_tag({'H_a': good, 'H_b': dict(good)}) == (0.1445, 0.119)
    for bad in (dict(tag_h_m=0.119), dict(car_side_m=0.0, tag_h_m=0.119), dict(car_side_m=0.1445, tag_h_m=-0.1),
                dict(car_side_m=True, tag_h_m=0.119), dict(car_side_m=0.1445, tag_h_m=float('nan'))):
        with pytest.raises(kc.KCError):
            kc.session_tag({'H_a': good, 'H_b': bad})
    with pytest.raises(kc.KCError, match='differs'):
        kc.session_tag({'H_a': good, 'H_b': dict(car_side_m=0.1445, tag_h_m=0.12)})


def test_phan_tich_writes_the_converted_a3(tmp_path, scene, monkeypatch):
    """Production path: A3 block converted (quy_ve_ma_noc false before approval), a3_cap.csv has quy_deg."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(20)], [img] * 20))
    assert kc.main(_phan_tich_args(tmp_path, scene)) == 0
    r = json.loads((tmp_path / 'kc' / 'kiem_cheo.json').read_text(encoding='utf-8'))
    assert r['A3']['quy_ve_ma_noc'] is True and 'gap_max_deg' in r['A3'] and r['version'] == 'c7_kiem_cheo_v7'
    head = (tmp_path / 'kc' / 'a3_cap.csv').read_text(encoding='utf-8').splitlines()[0]
    assert 'quy_deg' in head and 'gap_deg' in head


def _tilted(f, k, C, ry, rx):
    R0 = np.diag([1.0, -1.0, -1.0])
    cy_, sy_, cx_, sx_ = math.cos(ry), math.sin(ry), math.cos(rx), math.sin(rx)
    R = R0 @ np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]]) @ np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
    return cp.Camera('B1', f, W / 2, H / 2, [k], cv_rvec(R), -R @ np.array(C))


def test_a3_gap_uses_the_tag_height_and_both_segment_ends():
    """Codex KCA1-4 (case of Codex, one direction full -> without pair: 0.014101 / midpoint 0.012243 / z = 0 0.013721,
    reproduced by the author); both directions (v5b): 0.014102, maximum at t = 1, phi = 75 deg; midpoint only 0.012247,
    tag at z = 0 0.013723: a conversion that drops the height or the segment ends is seen."""
    F = _tilted(1200, -0.020, (0.030, -0.020, 2.632), 0.150, -0.200)
    c = _tilted(1202, -0.018, (0.031, -0.019, 2.634), 0.151, -0.201)
    mi, mj = (1.1, 0.8, SIDE), (1.7, 1.1, SIDE)
    full = kc.a3_tag_gap(F, c, mi, mj, 0.1445, 0.119)[0]
    assert full == pytest.approx(0.0141020, abs=5e-7)
    assert kc.a3_tag_gap(F, c, mi, mj, 0.1445, 0.119, ts=(0.5,))[0] == pytest.approx(0.0122470, abs=5e-7)
    assert kc.a3_tag_gap(F, c, mi, mj, 0.1445, 0.0)[0] == pytest.approx(0.0137229, abs=5e-7)
    assert kc.a3_tag_gap(F, c, mi, mj, 0.1445, 0.119, ts=(1.0,), phis=(75.0,))[0] == pytest.approx(full, abs=1e-9)
    # Codex KCA2 (round 2): the pair reversed puts the maximum at t = 0; dropping t = 0 gives 0.0131368 only
    assert kc.a3_tag_gap(F, c, mj, mi, 0.1445, 0.119)[0] == pytest.approx(full, abs=1e-9)
    assert kc.a3_tag_gap(F, c, mj, mi, 0.1445, 0.119, ts=(0.25, 0.5, 0.75, 1.0))[0] == pytest.approx(0.0131368, abs=5e-7)


def test_a3_clip_passes_the_session_tag_to_the_conversion(scene, monkeypatch):
    marks, cam, img = scene
    seen = []
    monkeypatch.setattr(kc, 'a3_tag_gap', lambda F, c, mi, mj, side, h: seen.append((side, h)) or (0.0, 0.0, 0.0))
    kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=0.6, tag=(0.1445, 0.119))
    assert seen and set(seen) == {(0.1445, 0.119)}


def _a3_conv(**over):
    a3 = dict(max_deg=0.16, max_deg_cap=0.15, rms_deg=0.07, quy_ve_ma_noc=False,
              ma_ao=dict(t=list(kc.A3_TAG_T), phi_deg=list(kc.A3_TAG_PHI_DEG), hai_chieu=True, canh_m=0.1445, cao_m=0.12))
    a3.update(over)
    return a3


@pytest.mark.parametrize('case', ['flag_true', 'side', 'height', 'no_geom'])
def test_xuat_refuses_unapproved_flag_and_other_tag(tmp_path, case, monkeypatch):
    """Codex KCA1-2: an input A3 with quy_ve_ma_noc true is refused while A3_QUY_DUYET is False; KCA1-3: a converted A3
    must name the tag the scorer uses (bo_cham.side_m 0.1445, do_cao.h_m 0.12 in the parts)."""
    a3 = dict(flag_true=_a3_conv(quy_ve_ma_noc=True), side=_a3_conv(ma_ao=dict(canh_m=0.0964, cao_m=0.12)),
              height=_a3_conv(ma_ao=dict(canh_m=0.1445, cao_m=0.0)), no_geom=_a3_conv(ma_ao={}))[case]
    if case == 'flag_true':
        monkeypatch.setattr(kc, 'A3_QUY_DUYET', False)
    kcp = _kc_file(tmp_path, A3=dict(a3, buoi='C8_x'))
    assert kc.main(_xuat_args(tmp_path, kcp)) == 2
    assert not (tmp_path / 'o.json').exists()


def test_xuat_converted_a3_ok_and_flag_only_after_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(kc, 'A3_QUY_DUYET', False)
    kcp = _kc_file(tmp_path, A3=dict(_a3_conv(), buoi='C8_x'))
    assert kc.main(_xuat_args(tmp_path, kcp)) == 2                   # KCJ1-4: c8b_u v3 refuses it, so does the writer
    assert not (tmp_path / 'o.json').exists()
    t2 = tmp_path / 'b'
    t2.mkdir()
    monkeypatch.setattr(kc, 'A3_QUY_DUYET', True)
    kcp = _kc_file(t2, A3=dict(_a3_conv(quy_ve_ma_noc=True), buoi='C8_x'))
    assert kc.main(_xuat_args(t2, kcp)) == 0


# ---------------------------------------------------------------- design v6 S8 (CODE_KIEMCHEO phien 3, 19/11)
class _MapCam:
    """A camera whose floor map is the true one followed by xy -> A xy + t (a map error with the tape fixed)."""

    def __init__(self, cam, A, t=(0.0, 0.0)):
        self.cam, self.A, self.t = cam, np.asarray(A, float), np.asarray(t, float)

    def pix2floor(self, uv, z=0.0):
        return self.cam.pix2floor(uv, z) @ self.A.T + self.t

    def project(self, P):
        P = np.array(P, float)
        P[:, :2] = (P[:, :2] - self.t) @ np.linalg.inv(self.A).T
        return self.cam.project(P)


class _FakePC:
    """PairCams stand-in: the full camera for (), `held` for any left-out set."""

    def __init__(self, img, full, held):
        self.img, self.ids, self.full, self.held = img, sorted(img), full, held

    def __call__(self, out=()):
        return self.full if not out else self.held


def _rotm(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def test_read_plan_events_and_runs(tmp_path):
    """TD2-2: events keep the clips H, runs do not; order shape checked (TD1-3)."""
    p = tmp_path / 'plan.txt'
    p.write_text('# ke hoach\nluot\nH_a\nH_b\nR1\nH_m\nR2\nH_z\n', encoding='utf-8')
    ev, runs = kc.read_plan(str(p))
    assert ev == ['H_a', 'H_b', 'R1', 'H_m', 'R2', 'H_z'] and runs == ['R1', 'R2']
    assert kc.plan_order_problems(ev) == []
    assert kc.plan_order_problems(['H_a', 'R1', 'H_b', 'H_m', 'R2', 'H_z'])
    assert kc.plan_order_problems(['H_a', 'H_b', 'H_m', 'R1', 'R2', 'H_z'])           # H_m not between runs
    assert kc.plan_order_problems(['H_a', 'H_b', 'R1', 'H_m', 'H_z', 'R2'])
    assert kc.plan_order_problems(['H_a', 'H_b', 'R1', 'R2', 'H_z']) == ['thieu H_m']
    p.write_text('H_a\nH_b\n', encoding='utf-8')
    with pytest.raises(kc.KCError):
        kc.read_plan(str(p))


def _tk_two(tmp_path, r2_frames=None, r1_loai=None, same_dir=False, theo=None):
    """Two runs r1, r2 of one plan, the shared kiem_cheo / u_so of _tk_setup."""
    args, out = _tk_setup(tmp_path, theo=theo)
    plan = tmp_path / 'plan.txt'
    plan.write_text('H_a\nH_b\nr1\nH_m\nr2\nH_z\n', encoding='utf-8')
    d1 = tmp_path / 'r1'
    tr = json.loads((d1 / 'track.json').read_text(encoding='utf-8'))
    if r1_loai:
        tr.update(moc_loai=r1_loai, cham_lai=2)
        (d1 / 'track.json').write_text(json.dumps(tr), encoding='utf-8')
    d2 = tmp_path / 'r2'
    d2.mkdir()
    _write_csv(d2 / 'frames.csv', r2_frames if r2_frames is not None else _frames(invalid={7}))
    _write_csv(d2 / 'kiem_cheo_khung.csv', _khung())
    tr2 = dict(tr, moc_loai=[], cham_lai=1, frames_sha256=kc.sha256_file(str(d2 / 'frames.csv')),
               kiem_cheo_khung_sha256=kc.sha256_file(str(d2 / 'kiem_cheo_khung.csv')))
    (d2 / 'track.json').write_text(json.dumps(tr2), encoding='utf-8')
    args += ['--luot', f'r2={d1 if same_dir else d2}']
    return args, out


def test_tung_khung_contradiction_is_per_run(tmp_path):
    """TD1-2: r1 excluded mark 5 (chain xe) from its first frame, r2 kept it as an anchor: only r2 contradicts; the
    exclusion of one run does not clear another."""
    args, out = _tk_two(tmp_path, r1_loai=[dict(chuoi='xe', id=5, t_bd=0.0)],
                        theo={'H_z': dict(rieng_m={'5': 0.0020, '6': 0.0005})})
    assert kc.main(args) == 0
    t = json.loads(out.read_text(encoding='utf-8'))
    assert t['mau_thuan'] == [dict(luot='r2', clip='H_z', moc=5)] and not t['hoan_tat']
    assert t['ke_hoach']['su_kien'][0] == 'H_a' and t['ke_hoach']['luot'] == ['r1', 'r2']
    assert os.path.isabs(t['luot']['r2']['duong_dan']['frames']) and t['version'] == 'c7_tung_khung_v4'
    # a run that excluded the mark only later still used it before: a contradiction
    assert kc.run_contradictions(_frames(), dict(moc_loai=[dict(chuoi='xe', id=5, t_bd=0.5)]), {5}) == [5]


def test_tung_khung_unique_frames_and_dirs(tmp_path):
    """TD1-6: the same directory or the same frames.csv for two runs is refused."""
    args, out = _tk_two(tmp_path, same_dir=True)
    assert kc.main(args) == 2 and not out.exists()
    t2 = tmp_path / 'b'
    t2.mkdir()
    args, out = _tk_two(t2, r2_frames=_frames())                     # byte-identical to r1's frames.csv
    assert kc.main(args) == 2 and not out.exists()


def test_tung_khung_missing_anchor_columns_is_data_missing(tmp_path):
    """TD2-1: frames.csv without moc_neo columns: the run is written with thieu_du_lieu_neo, not refused."""
    fr = [{k: v for k, v in r.items() if not k.startswith('moc_')} for r in _frames(invalid={9})]
    args, out = _tk_two(tmp_path, r2_frames=fr)
    assert kc.main(args) == 0
    t = json.loads(out.read_text(encoding='utf-8'))
    assert t['luot']['r2']['thieu_du_lieu_neo'] and t['luot_thieu_neo'] == ['r2'] and not t['hoan_tat']


# ---- tien_de
@pytest.fixture
def id8(monkeypatch):
    monkeypatch.setattr(kc, 'ID_KHOA', tuple(ALL8))


def _td_inputs():
    sess = {n: dict(file=f'/x/{n}.json', sha256=str(k) * 64, video=f'/x/{n}.mov', video_sha256=str(k + 4) * 64)
            for k, n in enumerate(kc.CLIPS_REQUIRED)}
    kcr = dict(version=kc.VERSION, buoi='S1', model='B1', nhom=GR, thu=False, sessions=sess, moc=[],
               hang_so_sha256='c' * 64, u_so_sha256='d' * 64,
               kiem_h={n: dict(ti_le_cung_luc=1.0) for n in kc.CLIPS_REQUIRED})
    us = dict(version=kc.U_SO_VERSION, buoi='S1', chuoi=_chains())
    for c in us['chuoi'].values():
        c.update(u_so_m=0.0002, u_so_tho_m=0.0003)
    run = dict(thieu=False, thieu_du_lieu_neo=False, tap_chua_tinh=[], max_m=0.002,
               nguon=dict(track_sha256='e' * 64, frames_sha256='f' * 64, kiem_cheo_khung_sha256='a' * 64))
    r2 = copy.deepcopy(run)
    r2['nguon']['frames_sha256'] = '9' * 64
    tk = dict(version=kc.TUNG_KHUNG_VERSION, buoi='S1', model='B1', nhom=GR, moc=[], hang_so_sha256='c' * 64,
              u_so_sha256='d' * 64, session_sha256=sess['H_a']['sha256'], cham_lai=True, kiem_cheo_sha256='k' * 64,
              ke_hoach=dict(file='/x/plan', sha256='b' * 64, luot=['R1', 'R2'],
                            su_kien=['H_a', 'H_b', 'R1', 'H_m', 'R2', 'H_z']),
              luot={'R1': copy.deepcopy(run), 'R2': r2}, luot_chua_co=[], mau_thuan=[])
    return kcr, us, tk


def _grade(f=None, tk_none=False):
    return _td_full(f, tk_none)                     # KCJ1-1: the fixture carries the H_m / H_z comparison


@pytest.mark.parametrize('case, ket, why', [
    (None, 'dat', None),
    (lambda k, u, t: k.update(thu=True), 'thieu_du_lieu', 'thu'),
    (lambda k, u, t: k['sessions'].pop('H_z'), 'thieu_du_lieu', 'thieu_clip:H_z'),
    (lambda k, u, t: k['kiem_h']['H_m'].update(ti_le_cung_luc=0.9), 'thieu_du_lieu', 'kiem_h:H_m'),
    (lambda k, u, t: u['chuoi']['2'].update(khong_tinh_duoc=[dict(moc=3)]), 'thieu_du_lieu', 'u_so_khong_tinh:2'),
    ('no_tk', 'thieu_du_lieu', 'khong_co_tung_khung'),
    (lambda k, u, t: (t.update(luot_chua_co=['R3']), t['ke_hoach']['luot'].append('R3'),
                      t['ke_hoach']['su_kien'].insert(-1, 'R3')), 'thieu_du_lieu',
     'luot_chua_co:R3'),
    (lambda k, u, t: t['luot']['R1'].update(thieu=True, tap_chua_tinh=[dict(chuoi='xe')]), 'thieu_du_lieu',
     'tap_neo_chua_tinh:R1'),
    (lambda k, u, t: t['luot']['R2'].update(thieu=True, thieu_du_lieu_neo=True), 'thieu_du_lieu', 'thieu_neo:R2'),
    (lambda k, u, t: t.update(mau_thuan=[dict(luot='R1', clip='H_m', moc=5)]), 'khong_dat', 'mau_thuan:R1:H_m:5'),
    (lambda k, u, t: t['ke_hoach'].update(su_kien=['H_b', 'H_a', 'R1', 'H_m', 'R2', 'H_z']), 'thieu_du_lieu', None),
])
def test_tien_de_grades_each_row(case, ket, why, id8):
    """S6-4: one case per row; ket_qua and the reason; TD1-7 schema keys always there."""
    r = _grade(tk_none=True) if case == 'no_tk' else _grade(case)
    assert r['ket_qua'] == ket
    if why:
        assert why in r['ly_do']
    assert set(r['clip_h']) == set(kc.CLIPS_REQUIRED) and r['h1_ho_tro'] is False and r['format'] == 'tien_de_buoi_v1'
    assert r['vai_tro_clip_h'] == dict(H_a='dau1', H_b='dau2', H_m='giua', H_z='cuoi')     # F18 (GATE GC6-4)
    if case == 'no_tk':
        assert r['sha256']['tung_khung'] is None or r['sha256']['luot'] is None
        assert r['luot'] == [] and r['luot_thieu_kiem'] == [] and r['mau_thuan'] == []
    if case is None:
        assert r['luot'] == ['R1', 'R2'] and r['ly_do'] == [] and r['u_so_m'] == pytest.approx(0.0003)


def test_tien_de_short_coverage_lowers_only_the_run_and_worst_wins(id8):
    """TD1-5 (B): a run short of checks (no uncomputed set) only goes to luot_thieu_kiem; several reasons: the worst."""
    r = _grade(lambda k, u, t: t['luot']['R2'].update(thieu=True))
    assert r['ket_qua'] == 'dat' and r['luot_thieu_kiem'] == ['R2'] and set(r['luot_thieu_kiem']) <= set(r['luot'])
    r = _grade(lambda k, u, t: (k.update(thu=True), t.update(mau_thuan=[dict(luot='R1', clip='H_z', moc=2)])))
    assert r['ket_qua'] == 'khong_dat' and 'thu' in r['ly_do']
    r = _grade(lambda k, u, t: u['chuoi']['xe'].update(khong_tinh_duoc=[dict(moc=1)]))
    assert r['u_so_m'] is None and r['u_so_chuoi'] is None and r['sha256']['u_so'] == 'd' * 64


@pytest.mark.parametrize('f', [
    lambda k, u, t: u.update(buoi='S2'), lambda k, u, t: t.update(nhom=[[1], [2]]),
    lambda k, u, t: t.update(kiem_cheo_sha256='z' * 64), lambda k, u, t: t.update(u_so_sha256='y' * 64),
    lambda k, u, t: k['sessions']['H_b'].update(video_sha256=k['sessions']['H_a']['video_sha256']),
    lambda k, u, t: t['luot']['R2']['nguon'].update(frames_sha256='f' * 64), lambda k, u, t: t.update(version='v3')])
def test_tien_de_identity_faults_raise(f, id8):
    """S6-3, TD1-3, TD1-6: another session, grouping, source sha, a repeated clip H or frames: code 2."""
    with pytest.raises(kc.KCError):
        _grade(f)


def _distinct_videos(tmp_path, args):
    for n in 'abmz':                                                # a distinct video per clip H (TD1-3)
        v = tmp_path / f'clip_{n}.mov'
        v.write_bytes(f'video {n}'.encode())
        i = args.index(f'--video-{n}')
        args[i + 1] = str(v)
        sp = tmp_path / f'{n}.json'
        s = json.loads(sp.read_text(encoding='utf-8'))
        s['video_sha256'] = kc.sha256_file(str(v))
        sp.write_text(json.dumps(s), encoding='utf-8')
    return args


def test_tien_de_cli_writes_once_and_rehashes(tmp_path, scene, monkeypatch):
    """End to end on a phan_tich output: written (thieu_du_lieu without tung_khung), never replaced; another session
    id, a bad date, a source changed after the fact (TD3-2): code 2."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(20)], [img] * 20))
    assert kc.main(_distinct_videos(tmp_path, _phan_tich_args(tmp_path, scene))) == 0
    kcp, usp = tmp_path / 'kc' / 'kiem_cheo.json', tmp_path / 'kc' / 'u_so.json'
    out = tmp_path / 'td.json'
    base = ['tien_de', '--session-id', 'C8_x', '--ngay', '2026-11-20', '--kiem-cheo', str(kcp), '--u-so', str(usp)]
    assert kc.main(base + ['--out', str(out)]) == 0
    r = json.loads(out.read_text(encoding='utf-8'))
    assert r['ket_qua'] == 'thieu_du_lieu' and r['ly_do'] == ['khong_co_tung_khung'] and r['ngay'] == '2026-11-20'
    assert {x['vai_tro'] for x in r['nguon']} >= {'u_so', 'session_H_a', 'clip_H_z'}
    assert r['sha256']['u_so'] == kc.sha256_file(str(usp))
    assert kc.main(base + ['--out', str(out)]) == 2                  # never replaced
    assert kc.main(['tien_de', '--session-id', 'C8_y'] + base[3:] + ['--out', str(tmp_path / 'o_x.json')]) == 2
    bad_date = list(base)
    bad_date[4] = '2026-13-01'
    assert kc.main(bad_date + ['--out', str(tmp_path / 'o_y.json')]) == 2
    (tmp_path / 'clip_m.mov').write_bytes(b'changed')                 # TD3-2: evidence no longer matches
    assert kc.main(base + ['--out', str(tmp_path / 'o_z.json')]) == 2
    assert not any((tmp_path / n).exists() for n in ('o_x.json', 'o_y.json', 'o_z.json'))


# ---- S7 geometry
def test_p1_rel_exact_scene_and_rigid_map_error(scene):
    """S7-8, KS7-1: exact cameras give d and chuyen ~ 0; a held camera with a rigid map error Q (tape fixed) gives
    d_sj = (Q - I)(x_j - x_s) and d_he_dau ~ 0 (the common turn is removed)."""
    marks, cam, img = scene
    regions = kc.start_regions(marks, {'goc': (0.0, 0.0)}, 0.3)
    assert regions['goc']['vung_mo_rong'] and len(regions['goc']['moc']) == 3
    pc = _FakePC(img, cam, cam)
    rows, pairs, hong = kc.p1_rel_clip('H_a', pc, marks, regions, (0.1445, 0.119), 0.119 / ZC, (0.0, 0.0),
                                       phis=(0.0, 45.0))
    # KCJ1-9: the centre is the mean of the back-projected corners: exact for an exact camera
    assert not hong and max(p['d_m'] for p in pairs) < 1e-9 and max(p['chuyen_m'] for p in pairs) < 1e-9
    Q = _rotm(0.5)
    pc = _FakePC(img, cam, _MapCam(cam, Q, (0.004, -0.003)))
    rows, pairs, hong = kc.p1_rel_clip('H_a', pc, marks, regions, (0.1445, 0.119), 0.119 / ZC, (0.0, 0.0), phis=(0.0,))
    for r in rows:
        v = np.array(marks[r['j']][:2]) - np.array(marks[r['s']][:2])
        assert np.allclose([r['d_x'], r['d_y']], (Q - np.eye(2)) @ v, atol=3e-4)
        assert math.hypot(r['dh_x'], r['dh_y']) < 3e-4
    assert {r['nhin'] for r in rows} == {'full', 'held'}


def test_p1_rel_moved_far_mark(scene):
    """S7-8, KS7-7: the true far mark j moved 2 mm (tape old): pairs holding j show d = +2 mm (measured - tape)."""
    marks, cam, _ = scene
    j = 7
    true = dict(marks)
    true[j] = (marks[j][0] + 0.002, marks[j][1], SIDE)
    img = image_of(cam, true)
    real = kc.PairCams(img, marks, 'B1', W, H, ZC)
    pc = _FakePC(img, None, None)
    pc.ids = [30, j, 12]                                            # three pairs only (speed); fits use every mark
    pc.__class__ = type('_Sub', (_FakePC,), {'__call__': lambda self, out=(): real(out)})
    regions = {'goc': dict(moc=[30], vung_mo_rong=False)}
    rows, pairs, _ = kc.p1_rel_clip('H_a', pc, marks, regions, (0.1445, 0.119), 0.119 / ZC, (0.0, 0.0), phis=(0.0,))
    pj = [r for r in rows if r['j'] == j]
    assert pj and all(abs(r['d_x'] - 0.002) < 1e-4 and abs(r['d_y']) < 1e-4 for r in pj)
    assert all(math.hypot(r['d_x'], r['d_y']) < 3e-4 for r in rows if r['j'] != j)


def test_chuyen_pair_rotation_on_the_floor(scene):
    """S8-4 (2): tag on the floor (h = 0, kappa 0), held camera with a map turn Q: chuyen = held - full of the
    difference = (Q - I)(b - a), heading difference 0; the same camera: 0."""
    marks, cam, _ = scene
    Q = _rotm(0.3)
    pa, pb = (0.2, -0.3), (1.1, 0.4)
    v = np.array(pb) - pa
    want = {'full': (Q - np.eye(2)) @ v, 'held': (np.eye(2) - Q.T) @ v}   # imaged by the held camera: true = Q^-1 P
    rows = kc.chuyen_pair(cam, _MapCam(cam, Q), pa, pb, 0.4, 0.4, 0.1445, 0.0, 0.0, (0.0, 0.0))
    assert [r[0] for r in rows] == ['full', 'held']
    for nhin, dx, dy, dpsi in rows:
        assert np.allclose([dx, dy], want[nhin], atol=1e-9) and abs(dpsi) < 1e-9
    same = kc.chuyen_pair(cam, cam, pa, pb, 0.1, 0.2, 0.1445, 0.119, 0.05, (0.3, 0.1))
    assert all(abs(x) < 1e-12 for r in same for x in r[1:])


def test_region_cover_flags():
    """S7-12.3, S7-14.3: inside the hull, near marks, two A3 directions: no flag; outside: ngoai_suy; one direction:
    mot_phuong (directions mod 180); far marks: khoang_trong."""
    mk = {1: (0.0, 0.0, SIDE), 2: (0.4, 0.0, SIDE), 3: (0.4, 0.4, SIDE), 4: (0.0, 0.4, SIDE)}
    rows = [dict(i=1, j=2, L_m=0.4, huong_deg=0.0), dict(i=1, j=4, L_m=0.4, huong_deg=90.0),
            dict(i=2, j=3, L_m=0.4, huong_deg=90.0)]
    r = kc.region_cover([1, 2, 3, 4], mk, (0.2, 0.2), rows)
    assert not r['ngoai_suy'] and not r['mot_phuong'] and r['ly_do'] == []
    r = kc.region_cover([1, 2, 3, 4], mk, (0.2, -0.15), rows)
    assert r['ngoai_suy'] and 'mien_ngoai_bao_loi' in r['ly_do']
    r = kc.region_cover([1, 2, 3, 4], mk, (0.2, 0.2), [dict(i=1, j=2, L_m=0.4, huong_deg=10.0),
                                                       dict(i=4, j=3, L_m=0.4, huong_deg=230.0)])
    assert r['mot_phuong']                                           # 10 and 230 = 50 mod 180: 40 deg apart
    far = {1: (0.0, 0.0, SIDE), 2: (2.0, 0.0, SIDE), 3: (2.0, 2.0, SIDE), 4: (0.0, 2.0, SIDE)}
    assert 'khoang_trong' in kc.region_cover([1, 2, 3, 4], far, (1.0, 1.0), rows)['ly_do']


def test_a3_rel_per_clip():
    """S7-4, KS7-3: A3loc per mark, start region and far end in the same clip."""
    rows = [dict(i=1, j=2, quy_deg=0.10), dict(i=2, j=3, quy_deg=0.30), dict(i=4, j=5, quy_deg=0.05)]
    r = kc.a3_rel_clip(rows, {'goc': dict(moc=[1]), 'K1': dict(moc=[9])})
    assert r['A3loc'] == {1: 0.10, 2: 0.30, 3: 0.30, 4: 0.05, 5: 0.05}
    assert r['A3_xp'] == {'goc': 0.10, 'K1': None} and r['max_deg'] == pytest.approx(0.40)


def test_affine_mark_sees_stretch_and_shear(scene):
    """S7-5, KS7-4: exact camera ~ 0; a floor map stretched 1 % along x: one principal stretch 0.01; a shear: cat."""
    marks, cam, img = scene
    i = 17
    r = kc.affine_mark(cam, img[i], SIDE)
    assert max(abs(g) for g in r['gian']) < 1e-4 and r['du_mm'] < 1e-3
    st = _MapCam(cam, np.diag([1.01, 1.0]))
    r = kc.affine_mark(st, img[i], SIDE)
    assert max(r['gian']) == pytest.approx(0.01, abs=2e-4) and abs(r['cat_deg']) < 0.05
    sh = _MapCam(cam, np.array([[1.0, 0.02], [0.0, 1.0]]))
    assert abs(kc.affine_mark(sh, img[i], SIDE)['cat_deg']) == pytest.approx(math.degrees(math.atan(0.02)), abs=0.02)


def test_meo_summary_threshold_is_repeatability_plus_print():
    """S7-12.2: threshold = |q(H_a) - q(H_b)| + tolerance per size; the flag is diagnostic only."""
    def m(g, du):
        return dict(gian=[g, 0.0], cat_deg=0.0, du_mm=du, canh_m=SIDE)
    out = kc.meo_summary({'H_a': {1: m(0.001, 0.1)}, 'H_b': {1: m(0.0015, 0.1)}, 'H_m': {1: m(0.0024, 0.1)}})
    assert out['nguong_lap_HaHb']['96.4']['gian'] == pytest.approx(0.0005 + kc.DUNG_SAI_IN_GIAN)
    assert out['bien_dang_hinh'] is False and out['chi_chan_doan'] is True
    out = kc.meo_summary({'H_a': {1: m(0.001, 0.1)}, 'H_b': {1: m(0.0015, 0.1)}, 'H_m': {1: m(0.003, 0.1)}})
    assert out['bien_dang_hinh'] is True
    assert kc.meo_summary({'H_a': {1: m(0.001, 0.1)}})['bien_dang_hinh'] is None


def test_time_check_reports_common_turn(scene):
    """C8BX-06: the per-clip pose refit absorbs a rigid turn of the whole image (a camera move): xoay_chung_deg ~ 0;
    a turn of the target marks only (the floor plate moved under a still camera) is seen."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    gc = kc.GroupCams(img, marks, groups, 'B1', W, H, ZC)
    Q = _rotm(0.2)
    turned = {i: (*(Q @ np.array(m[:2])), m[2]) for i, m in marks.items()}
    r = kc.time_check([img] * 3, image_of(cam, turned), marks, groups, gc)
    assert abs(r['xoay_chung_deg']) < 0.01 and abs(r['gian_chung']) < 1e-3
    half = {i: (turned[i] if i in groups[0] else marks[i]) for i in marks}
    r = kc.time_check([img] * 3, image_of(cam, half), marks, groups, gc)
    assert abs(r['xoay_chung_deg']) > 0.01 and 'xoay_chung_deg' in r


def test_phan_tich_writes_s7(tmp_path, scene, monkeypatch):
    """Production path with --xuat-phat, --plumb: S7 blocks, cap_tuong_doi.csv with its sha, constants and ids."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(20)], [img] * 20))
    monkeypatch.setattr(kc, 'R_XP_M', 0.3)
    monkeypatch.setattr(kc, 'commit_hil', lambda: dict(sha='c' * 40, dirty=False))
    monkeypatch.setattr(kc, 'manifest_git', lambda p: dict(commit='m' * 40, time='2026-11-19T19:00:00+07:00', sach=True))
    mf = _khoa_manifest(tmp_path, scene)
    assert kc.main(_phan_tich_args(tmp_path, scene) + ['--xuat-phat', 'goc=0,0', '--plumb', '0.03,-0.02',
                                                       '--khoa-s7', str(mf)]) == 0
    r = json.loads((tmp_path / 'kc' / 'kiem_cheo.json').read_text(encoding='utf-8'))
    assert r['khoa_s7']['sha256'] == kc.sha256_file(str(mf)) and r['khoa_s7']['git_commit'] == 'm' * 40
    assert r['P1_tuong_doi']['du_lieu_du'] is True and r['A3']['du_lieu_du'] is True
    p = r['P1_tuong_doi']
    assert p['max_m'] < 5e-4 and p['theo_vung']['goc']['vung_mo_rong'] and p['cap_hong'] == []
    assert p['cap_csv']['sha256'] == kc.sha256_file(p['cap_csv']['file'])
    assert r['A3_tuong_doi']['A3_xp_deg']['goc'] is not None and r['hang_so_s7']['R_XP_M'] == 0.3
    assert r['id_khoa'] == sorted(img) and r['meo_cuc_bo']['bien_dang_hinh'] is False
    assert os.path.isabs(r['sessions']['H_a']['video']) and 'theo_co' in r['P7']
    t2 = tmp_path / 'x'
    t2.mkdir()
    assert kc.main(_phan_tich_args(t2, scene) + ['--xuat-phat', 'goc=0,0']) == 2          # needs --plumb


def test_e_hieu_reads_f5_and_writes_f9(tmp_path, scene):
    """S7-17, S8-5: (i) from kiem_cheo, (ii) on the F5 session (exact scene ~ 0), (iii) missing -> chan_doan; a leg
    without p_m: only (i); an edited F5: code 2; the output is what c8_hc_a.read_e_hieu reads."""
    marks, cam, img = scene
    video = tmp_path / 'v.mov'
    video.write_bytes(b'v')
    sp = _fake_session(tmp_path, 'hc', img, marks, video)
    frp = tmp_path / 'frames.csv'
    frp.write_text('idx,t_s\n0,0.0\n', encoding='utf-8')
    trp = tmp_path / 'track.json'
    trp.write_text(json.dumps(dict(session_sha256=kc.sha256_file(str(sp)), frames_sha256=kc.sha256_file(str(frp)))),
                   encoding='utf-8')
    kcr = dict(version=kc.VERSION, thu=False, model='B1',
               hang_so_s7=dict(xuat_phat=dict(goc=[0.0, 0.0]), R_XP_M=0.6, L_MAX_M=1.05, plumb=[0.03, -0.02]),
               P1_tuong_doi=dict(theo_vung=dict(goc=dict(trong_vung_max_m=0.0007))))
    kcp = tmp_path / 'kc.json'
    kcp.write_text(json.dumps(kcr), encoding='utf-8')

    def win(p, psi):
        return dict(loai='dung_12b', chi_so=0, khung_id=[1], khung_pts_s=[0.0], khung_sha256='a' * 64, p_m=p,
                    psi_rad=psi)
    doan = dict(schema='c8_hc_doan_12b_v1', cho='goc', trang_thai='chan_doan', session_sha256=kc.sha256_file(str(sp)),
                track_sha256=kc.sha256_file(str(trp)), frames_sha256=kc.sha256_file(str(frp)),
                car_pose_cfg=dict(side_m=0.1445, h_m=0.119, tag_offset=None, q_m=None, du_m=None),
                nguon=[dict(vai_tro='session', duong_dan=str(sp), sha256=kc.sha256_file(str(sp))),
                       dict(vai_tro='track', duong_dan=str(trp), sha256=kc.sha256_file(str(trp))),
                       dict(vai_tro='frames', duong_dan=str(frp), sha256=kc.sha256_file(str(frp)))],
                doan=[dict(i=0, trang_thai='hop_le', cua_so_a=win([-0.2, 0.0], 0.0), cua_so_b=win([0.3, 0.1], 0.02)),
                      dict(i=1, trang_thai='hop_le', cua_so_a=win(None, None), cua_so_b=win([0.3, 0.1], 0.0))])
    doan['sha256'] = kc.doan_body_sha(doan)
    dp = tmp_path / 'doan_12b_v1.json'
    dp.write_text(json.dumps(doan), encoding='utf-8')
    out = tmp_path / 'e.json'
    assert kc.main(['e_hieu', '--kiem-cheo', str(kcp), '--doan', str(dp), '--out', str(out)]) == 0
    e = json.loads(out.read_text(encoding='utf-8'))
    assert e['doan_12b_sha256'] == doan['sha256'] and e['bao_q_xoay'] is False and e['trang_thai'] == 'chan_doan'
    legs = {x['i']: x for x in e['doan']}
    assert legs[0]['phan_ii_m'] < 1e-6 and legs[0]['e_hieu_m_chan_doan'] == pytest.approx(0.0007)
    assert legs[1]['phan_ii_m'] is None and any('chi (i)' in w for w in legs[1]['ly_do'])
    # KCJ1-7: no usable e_hieu_m while chan_doan (c8_hc_a.read_e_hieu refuses a row without it: F9 cannot unlock khoa)
    assert all('e_hieu_m' not in x for x in e['doan'])
    tr = json.loads(trp.read_text(encoding='utf-8'))                  # KCJ1-8: a track of another session: code 2
    trp.write_text(json.dumps(dict(tr, session_sha256='1' * 64)), encoding='utf-8')
    d2 = dict(doan, track_sha256=kc.sha256_file(str(trp)))
    d2['nguon'] = [dict(x, sha256=kc.sha256_file(str(trp))) if x['vai_tro'] == 'track' else x for x in doan['nguon']]
    d2['sha256'] = kc.doan_body_sha(d2)
    dp2 = tmp_path / 'doan2.json'
    dp2.write_text(json.dumps(d2), encoding='utf-8')
    assert kc.main(['e_hieu', '--kiem-cheo', str(kcp), '--doan', str(dp2), '--out', str(tmp_path / 'e3.json')]) == 2
    doan['doan'][0]['dai'] = 9                                        # edited after its sha
    dp.write_text(json.dumps(doan), encoding='utf-8')
    assert kc.main(['e_hieu', '--kiem-cheo', str(kcp), '--doan', str(dp), '--out', str(tmp_path / 'e2.json')]) == 2


def test_c8b_u_reads_the_cli_export(tmp_path):
    """S8-6: the file the xuat CLI writes (U_moc of CAM + tien_de + u_so information) passes c8b_u v3 check_cam
    (runs where c8b_u v3 is importable; skipped before)."""
    CU = pytest.importorskip('c8b_u')
    if getattr(CU, 'CAM_VERSION', None) != kc.CAM_U_VERSION or not hasattr(CU, 'check_cam'):
        pytest.skip('c8b_u of another camera-file version')
    kcp = _kc_file(tmp_path, A3=dict(_a3_conv(quy_ve_ma_noc=True), buoi='C8_x'))
    assert kc.main(_xuat_args(tmp_path, kcp)) == 0
    CU.check_cam(json.loads((tmp_path / 'o.json').read_text(encoding='utf-8')))


# ---------------------------------------------------------------- Codex round 1 of the J round (KCJ1-1 ... KCJ1-16)
def _td_full(f=None, tk_none=False):
    """_td_inputs with the comparison at H_m / H_z present for every locked id (KCJ1-1)."""
    kcr, us, tk = _td_inputs()
    kcr['id_khoa'] = ALL8
    kcr['theo_thoi_gian'] = {n: dict(rieng_m={str(i): 0.0002 for i in ALL8}) for n in ('H_b', 'H_m', 'H_z')}
    if f:
        f(kcr, us, tk)
    return kc.tien_de_eval(kcr, us, None if tk_none else tk, 'k' * 64, 'd' * 64, 'g' * 64)


@pytest.mark.parametrize('case, why', [
    (lambda k, u, t: k.pop('theo_thoi_gian'), 'so_H_m_thieu'),
    (lambda k, u, t: k['theo_thoi_gian']['H_z']['rieng_m'].pop('3'), 'so_H_z_thieu'),
    (lambda k, u, t: k['theo_thoi_gian']['H_m']['rieng_m'].update({'4': float('nan')}), 'so_H_m_thieu'),
    (lambda k, u, t: u['chuoi'].pop('2'), 'u_so_khong_tinh:2'),
    (lambda k, u, t: u['chuoi']['1'].update(tap_tinh=[]), 'u_so_khong_tinh:1'),
    (lambda k, u, t: u['chuoi']['xe'].pop('nguong_m'), 'u_so_khong_tinh:xe'),
    (lambda k, u, t: k['kiem_h']['H_m'].update(ti_le_cung_luc=float('nan')), 'kiem_h:H_m'),
])
def test_tien_de_missing_required_data_is_not_dat(case, why, id8):
    """KCJ1-1, KCJ1-2 (J v6 J4-12, M3): a missing comparison at H_m / H_z, a missing chain or set, a NaN: data
    missing, never dat."""
    assert _td_full()['ket_qua'] == 'dat'
    r = _td_full(case)
    assert r['ket_qua'] == 'thieu_du_lieu' and any(x.startswith(why) for x in r['ly_do'])


def test_tien_de_runs_must_make_up_the_plan(id8):
    """KCJ1-2: runs written + runs missing must be the plan (an emptied luot is refused, code 2)."""
    with pytest.raises(kc.KCError, match='KCJ1-2'):
        _td_full(lambda k, u, t: t.update(luot={}))


def test_tung_khung_contradiction_kept_when_another_anchor_column_lacks(tmp_path):
    """KCJ1-5: moc_neo_1 missing but moc_neo_xe present: the run is thieu_du_lieu_neo AND its contradiction stays."""
    fr = [{k: v for k, v in r.items() if k != 'moc_neo_1'} for r in _frames(invalid={9})]
    args, out = _tk_two(tmp_path, r2_frames=fr, theo={'H_z': dict(rieng_m={'5': 0.0020})})
    assert kc.main(args) == 0
    t = json.loads(out.read_text(encoding='utf-8'))
    assert t['luot']['r2']['thieu_du_lieu_neo'] and dict(luot='r2', clip='H_z', moc=5) in t['mau_thuan']


def test_phan_tich_existing_output_leaves_nothing_new(tmp_path, scene, monkeypatch):
    """KCJ1-6: an a3_cap.csv already there: code 2 before anything is written (no u_so.json, kiem_cheo.json)."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(20)], [img] * 20))
    (tmp_path / 'kc').mkdir()
    (tmp_path / 'kc' / 'a3_cap.csv').write_text('x', encoding='utf-8')
    assert kc.main(_phan_tich_args(tmp_path, scene)) == 2
    assert sorted(p.name for p in (tmp_path / 'kc').iterdir()) == ['a3_cap.csv']


def test_publish_new_leaves_no_partial_file(tmp_path):
    """KCJ1-6: a writer that fails half way leaves neither the target nor its temporary file."""
    p = tmp_path / 'x.json'

    def boom(f):
        f.write('{"half": ')
        raise OSError('disk')
    with pytest.raises(OSError):
        kc.publish_new(str(p), boom)
    assert list(tmp_path.iterdir()) == []


def test_repeated_ids_are_refused(tmp_path):
    """KCJ1-10: a repeated id in the session fit or in the marker CSV is refused, not merged; ids 1-32 locked."""
    s = dict(kind='c7_session', model_khoa='B1', chosen='B1', width=W, height=H, zc_tape_m=ZC, marks={'1': [0, 0, SIDE]},
             fit_ids=[1, 1], ref_corners_px={'1': [[0, 0]] * 4})
    with pytest.raises(kc.KCError, match='KCJ1-10'):
        kc.clip_from_session(s, 'B1')
    p = tmp_path / 'm.csv'
    p.write_text('id,x_cm\n3,1\n3,2\n', encoding='utf-8')
    with pytest.raises(kc.KCError, match='repeated'):
        kc.load_moc([str(p)])
    assert kc.ID_KHOA == tuple(range(1, 33))


def test_disc_tests_are_exact_or_conservative():
    """KCJ1-11: a hull edge 0.099 m from the centre of a 0.10 m disc is caught (16 rim samples missed it); the gap
    bound is never below a fine brute force over the disc."""
    c = np.array([0.0, 0.0])
    a = math.radians(11.25)
    nrm = np.array([math.cos(a), math.sin(a)])
    t = np.array([-nrm[1], nrm[0]])
    p0 = 0.099 * nrm
    hull = kc._hull(np.array([p0 + 5 * t, p0 - 5 * t, -3 * nrm + 5 * t, -3 * nrm - 5 * t]))
    assert not kc.disc_in_hull(c, 0.10, hull) and kc.disc_in_hull(c, 0.098, hull)
    pts = np.array([[0.3, 0.05], [-0.2, 0.31], [-0.1, -0.33], [0.25, -0.24]])
    g = np.linspace(-0.1, 0.1, 401)
    X, Y = np.meshgrid(g, g)
    P = np.column_stack([X.ravel(), Y.ravel()])
    P = P[np.hypot(P[:, 0], P[:, 1]) <= 0.1]
    brute = float(np.max(np.min(np.linalg.norm(P[:, None] - pts[None], axis=2), axis=1)))
    assert brute <= kc.disc_max_gap(c, 0.10, pts) <= brute + 0.009


def test_mot_phuong_flags_a_region_mark_without_a3loc_and_uses_l_max():
    """KCJ1-12: a fifth region mark in no A3 pair raises mot_phuong; the pair length limit is the run's L_max."""
    mk = {1: (0.0, 0.0, SIDE), 2: (0.4, 0.0, SIDE), 3: (0.4, 0.4, SIDE), 4: (0.0, 0.4, SIDE), 5: (0.2, 0.6, SIDE)}
    rows = [dict(i=1, j=2, L_m=0.4, huong_deg=0.0), dict(i=1, j=4, L_m=0.4, huong_deg=90.0)]
    r = kc.region_cover([1, 2, 3, 4, 5], mk, (0.2, 0.2), rows)
    assert r['mot_phuong'] and any('moc_khong_co_A3loc' in w for w in r['ly_do'])
    long_rows = [dict(i=1, j=3, L_m=1.10, huong_deg=45.0), dict(i=2, j=4, L_m=1.10, huong_deg=135.0),
                 dict(i=3, j=4, L_m=1.10, huong_deg=0.0)]
    assert kc.region_cover([1, 2, 3, 4], mk, (0.2, 0.2), long_rows, l_max=1.20)['ly_do'] == []
    assert kc.region_cover([1, 2, 3, 4], mk, (0.2, 0.2), long_rows, l_max=1.05)['mot_phuong']


def test_broken_pairs_are_listed_and_mark_the_region(scene):
    """KCJ1-13: a failing A3 pair is listed (hong), not fatal; a broken P1 pair marks its region and du_lieu_du."""
    marks, cam, img = scene
    bad = frozenset((30, 31))

    class PC(kc.PairCams):
        def __call__(self, out=()):
            if frozenset(out) == bad:
                raise kc.KCError('fit failed (test)')
            return super().__call__(out)
    pc = PC(img, marks, 'B1', W, H, ZC)
    hong = []
    rows = kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=1.05, pc=pc, hong=hong)
    assert [(h['i'], h['j']) for h in hong] == [(30, 31)] and all((r['i'], r['j']) != (30, 31) for r in rows)
    regions = {'goc': dict(moc=[30], vung_mo_rong=False)}
    few = _FakePC(img, None, None)
    few.ids = [30, 31, 12]                                         # two pairs only (speed); fits use every mark
    few.__class__ = type('_Sub2', (_FakePC,), {'__call__': lambda self, out=(): pc(out)})
    r_rows, pairs, p_hong = kc.p1_rel_clip('H_a', few, marks, regions, (0.1445, 0.119), 0.119 / ZC, (0.0, 0.0),
                                           phis=(0.0,))
    rel = {'H_a': dict(pairs=pairs, hong=p_hong, n_du_kien=2)}
    cov = {'H_a': {'goc': dict(ngoai_suy=False, mot_phuong=False, ly_do=[])}}
    P, A = kc.s7_summary(rel, {'H_a': kc.a3_rel_clip([dict(i=30, j=12, quy_deg=0.1)], regions)}, cov, regions)
    assert p_hong and P['theo_vung']['goc']['cap_hong'] and P['du_lieu_du'] is False and A['cap_hong']['goc']


def test_meo_sizes_one_rule():
    """KCJ1-15: 96.36 and 96.44 mm share the key 96.4: its maximum is the overall one, one threshold."""
    m = lambda g, sd: dict(gian=[g, 0.0], cat_deg=0.0, du_mm=0.1, canh_m=sd)       # noqa: E731
    out = kc.meo_summary({'H_a': {1: m(0.012, 0.09636), 2: m(0.0, 0.09644)},
                          'H_b': {1: m(0.012, 0.09636), 2: m(0.0, 0.09644)}})
    assert list(out['theo_co']) == ['96.4'] and out['theo_co']['96.4']['gian_chinh_max'] == pytest.approx(0.012)


def test_thang_direction_is_tape_to_camera():
    """KCJ1-16: held-out centres 1.01 x the tape: ti_le - 1 = +0.01, direction written."""
    marks = {1: (0.0, 0.0, SIDE), 2: (1.0, 0.0, SIDE), 3: (0.0, 1.0, SIDE)}
    p1 = {i: 0.01 * np.array(m[:2]) for i, m in marks.items()}
    r = kc.thang_clip(p1, marks)
    assert r['ti_le_tru_1'] == pytest.approx(0.01) and 'thuoc -> tam' in r['chieu']


def test_a3_relative_never_mixes_clips():
    """KS7-3 (Codex J1): H_a has the bad start, H_b the bad far end: the index is the worst SAME-clip sum (0.6), not
    the mixed 0.3 + 0.35."""
    regions = {'goc': dict(moc=[1], vung_mo_rong=False)}
    a3rel = {'H_a': kc.a3_rel_clip([dict(i=1, j=5, quy_deg=0.3), dict(i=3, j=4, quy_deg=0.1)], regions),
             'H_b': kc.a3_rel_clip([dict(i=1, j=5, quy_deg=0.1), dict(i=3, j=4, quy_deg=0.35)], regions)}
    rel = {c: dict(pairs=[dict(clip=c, vung='goc', s=1, j=2, L_m=0.5, trong_vung=False, d_m=0.0, dh_m=0.0,
                               chuyen_m=0.0, chuyen_psi_deg=0.0)], hong=[], n_du_kien=1) for c in a3rel}
    cov = {c: {'goc': dict(ngoai_suy=False, mot_phuong=False, ly_do=[])} for c in a3rel}
    _, A = kc.s7_summary(rel, a3rel, cov, regions)
    assert A['max_deg'] == pytest.approx(0.6) and A['A3_xp_deg']['goc'] == pytest.approx(0.3)


def _khoa_manifest(tmp_path, scene, **over):
    marks, cam, img = scene
    ids = sorted(img)
    hs = dict(R_XP_M=0.3, R_TRONG_M=kc.R_TRONG_M, MIEN_XP_M=kc.MIEN_XP_M, MIEN_XP_BUOC_M=kc.MIEN_XP_BUOC_M,
              HAI_PHUONG_DEG=kc.HAI_PHUONG_DEG, DUNG_SAI_IN_GIAN=kc.DUNG_SAI_IN_GIAN,
              DUNG_SAI_IN_DU_MM=kc.DUNG_SAI_IN_DU_MM, L_MAX_M=0.6, A3_TAG_T=list(kc.A3_TAG_T),
              A3_TAG_PHI_DEG=list(kc.A3_TAG_PHI_DEG), xuat_phat={'goc': [0.0, 0.0]}, plumb=[0.03, -0.02],
              car_side_m=0.1445, tag_h_m=0.119, model='B1', nhom=kc.make_groups(ids), id_khoa=ids)
    hs.update(over)
    p = tmp_path / 'khoa_s7.json'
    p.write_text(json.dumps(dict(schema='c7_khoa_s7_v1', hang_so_s7=hs, commit_hil='c' * 40)), encoding='utf-8')
    return p


def test_phan_tich_s7_needs_the_locked_manifest(tmp_path, scene, monkeypatch):
    """KCJ1-14: --xuat-phat without --khoa-s7 refused; a manifest with another constant, or a dirty code tree, or a
    manifest not committed: refused; the right one is recorded with its sha and commit."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'clip_dets', lambda v, ids, max_frames=None: ([k / 60 for k in range(20)], [img] * 20))
    monkeypatch.setattr(kc, 'R_XP_M', 0.3)
    monkeypatch.setattr(kc, 'A3_TAG_PHI_DEG', (0.0,))
    monkeypatch.setattr(kc, 'commit_hil', lambda: dict(sha='c' * 40, dirty=False))
    monkeypatch.setattr(kc, 'manifest_git', lambda p: dict(commit='m' * 40, time='2026-11-19T19:00:00+07:00', sach=True))
    xp = ['--xuat-phat', 'goc=0,0', '--plumb', '0.03,-0.02']
    for k, over in enumerate([None, dict(L_MAX_M=1.05)]):
        d = tmp_path / f'w{k}'
        d.mkdir()
        args = _phan_tich_args(d, scene) + xp
        if over is not None:
            args += ['--khoa-s7', str(_khoa_manifest(d, scene, **over))]
        assert kc.main(args) == 2 and not (d / 'kc' / 'kiem_cheo.json').exists()
    monkeypatch.setattr(kc, 'commit_hil', lambda: dict(sha='c' * 40, dirty=True))
    d = tmp_path / 'w2'
    d.mkdir()
    assert kc.main(_phan_tich_args(d, scene) + xp + ['--khoa-s7', str(_khoa_manifest(d, scene))]) == 2
    monkeypatch.setattr(kc, 'commit_hil', lambda: dict(sha='c' * 40, dirty=False))
    monkeypatch.setattr(kc, 'manifest_git', lambda p: dict(commit='m' * 40, time='t', sach=False))
    d = tmp_path / 'w3'
    d.mkdir()
    assert kc.main(_phan_tich_args(d, scene) + xp + ['--khoa-s7', str(_khoa_manifest(d, scene))]) == 2


# ---------------------------------------------------------------- Codex round 2 of the J round (KCJ2-1, 2; residuals)
def test_xuat_refuses_a3_over_a_partial_pair_set():
    """KCJ2-1: an A3 block with broken pairs is not exported (its maximum is not over the planned pairs)."""
    with pytest.raises(kc.KCError, match='broken pairs'):
        kc.build_cam_u(_own(_a3_conv(max_deg=0.15, quy_ve_ma_noc=True, cap_hong=[dict(i=1, j=2)], du_lieu_du=False)),
                       _parts())


def test_xuat_frames_by_run_id(tmp_path):
    """KCJ2-2, KCJ1-3: CAM may leave out a clock-failed run (r2), but each run it declares must carry that run's own
    frames sha; swapped shas are refused."""
    kcp = _kc_file(tmp_path)
    two = dict(luot={'r1': dict(nguon=dict(frames_sha256='a' * 64)), 'r2': dict(nguon=dict(frames_sha256='b' * 64))})
    assert kc.main(_xuat_args(tmp_path, kcp, tk_over=two)) == 0
    t2 = tmp_path / 's'
    t2.mkdir()
    kcp = _kc_file(t2)
    sw = dict(nguon=dict(_moc_dich()['nguon'], frames={'r1': 'b' * 64, 'r2': 'a' * 64}))
    assert kc.main(_xuat_args(t2, kcp, tk_over=two, md_over=sw)) == 2


@pytest.mark.parametrize('case, why', [
    (lambda k, u, t: k['theo_thoi_gian']['H_z']['rieng_m'].update({'3': False}), 'so_H_z_thieu'),
    (lambda k, u, t: k['theo_thoi_gian']['H_m']['rieng_m'].update({'3': -0.001}), 'so_H_m_thieu'),
    (lambda k, u, t: k['kiem_h']['H_b'].update(ti_le_cung_luc=float('inf')), 'kiem_h:H_b'),
    (lambda k, u, t: u['chuoi']['2'].update(tap_tinh=u['chuoi']['2']['tap_tinh'][:1]), 'u_so_khong_tinh:2'),
])
def test_tien_de_residuals_round2(case, why, id8):
    """KCJ1-1 (bool, negative), KCJ1-2 (inf, a chain without its full-set fits)."""
    r = _td_full(case)
    assert r['ket_qua'] == 'thieu_du_lieu' and any(x.startswith(why) for x in r['ly_do'])


def test_tien_de_plan_runs_come_from_the_events(id8):
    """KCJ1-2: a run removed from tk.luot AND ke_hoach.luot but still in su_kien: refused (code 2)."""
    def f(k, u, t):
        t['luot'].pop('R2')
        t['ke_hoach']['luot'] = ['R1']
    with pytest.raises(kc.KCError, match='su_kien'):
        _td_full(f)


def test_tien_de_premise_locks_ids_1_32():
    """KCJ1-10 at the premise: a result with ids 1-8 (not 1-32) is not dat."""
    r = _td_full()
    assert r['ket_qua'] == 'thieu_du_lieu' and 'id_khoa_khong_phai_1_32' in r['ly_do']


def test_e_hieu_refuses_f5_without_legs_and_unchained_track(tmp_path, scene):
    """KCJ1-7: F5 with no leg -> code 2; KCJ1-8: nguon track sha not the top-level track_sha256 -> code 2."""
    marks, cam, img = scene
    sp = _fake_session(tmp_path, 'hc', img, marks, tmp_path / 'v.mov') if (tmp_path / 'v.mov').write_bytes(b'v') else None
    frp = tmp_path / 'frames.csv'
    frp.write_text('idx,t_s\n0,0.0\n', encoding='utf-8')
    trp = tmp_path / 'track.json'
    trp.write_text(json.dumps(dict(session_sha256=kc.sha256_file(str(sp)), frames_sha256=kc.sha256_file(str(frp)))),
                   encoding='utf-8')
    kcr = dict(version=kc.VERSION, thu=False, model='B1',
               hang_so_s7=dict(xuat_phat=dict(goc=[0.0, 0.0]), R_XP_M=0.6, L_MAX_M=1.05, plumb=[0.03, -0.02]),
               P1_tuong_doi=dict(theo_vung=dict(goc=dict(trong_vung_max_m=0.0007))))
    kcp = tmp_path / 'kc.json'
    kcp.write_text(json.dumps(kcr), encoding='utf-8')
    win = dict(loai='dung_12b', chi_so=0, khung_id=[1], khung_pts_s=[0.0], khung_sha256='a' * 64, p_m=[0.0, 0.0],
               psi_rad=0.0)
    base = dict(schema='c8_hc_doan_12b_v1', cho='goc', trang_thai='chan_doan', session_sha256=kc.sha256_file(str(sp)),
                track_sha256=kc.sha256_file(str(trp)), frames_sha256=kc.sha256_file(str(frp)),
                car_pose_cfg=dict(side_m=0.1445, h_m=0.119, tag_offset=None, q_m=None, du_m=None),
                nguon=[dict(vai_tro='session', duong_dan=str(sp), sha256=kc.sha256_file(str(sp))),
                       dict(vai_tro='track', duong_dan=str(trp), sha256=kc.sha256_file(str(trp))),
                       dict(vai_tro='frames', duong_dan=str(frp), sha256=kc.sha256_file(str(frp)))])
    for k, d in enumerate([dict(base, doan=[]),
                           dict(base, doan=[dict(i=0, trang_thai='hop_le', cua_so_a=win, cua_so_b=win)],
                                nguon=[x if x['vai_tro'] != 'track' else dict(x, sha256='f' * 64) for x in base['nguon']])]):
        d['sha256'] = kc.doan_body_sha(d)
        dp = tmp_path / f'd{k}.json'
        dp.write_text(json.dumps(d), encoding='utf-8')
        assert kc.main(['e_hieu', '--kiem-cheo', str(kcp), '--doan', str(dp), '--out', str(tmp_path / f'e{k}.json')]) == 2
        assert not (tmp_path / f'e{k}.json').exists()


def test_a3_conversion_failure_is_a_broken_pair(scene, monkeypatch):
    """KCJ1-13: a failure after the fit (tag conversion) is listed for that pair, the run goes on."""
    marks, cam, img = scene
    real = kc.a3_tag_gap

    def gap(F, c, mi, mj, side, h):
        if mi == marks[30] and mj == marks[31]:
            raise ValueError('conversion failed (test)')
        return real(F, c, mi, mj, side, h)
    monkeypatch.setattr(kc, 'a3_tag_gap', gap)
    hong = []
    rows = kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=1.05, tag=(0.1445, 0.119), hong=hong)
    assert [(h['i'], h['j']) for h in hong] == [(30, 31)] and rows


def test_summary_print_failure_after_publication_keeps_code_0(tmp_path, monkeypatch):
    """KCJ1-6: the summary line after the file is published cannot turn the result into code 2."""
    import builtins
    real = builtins.print

    def flaky(*a, **k):
        raise OSError('stdout closed (test)')         # also the last status line (round 3)
    monkeypatch.setattr(builtins, 'print', flaky)
    assert kc.main(_xuat_args(tmp_path, _kc_file(tmp_path))) == 0
    assert (tmp_path / 'o.json').exists()


def test_geometry_guards_outside_the_floor_domain():
    """KCJ1-11 (round 2 note): repeated vertices make no hull; far coordinates keep the bound; no mark refused."""
    assert not kc.disc_in_hull((0.0, 0.0), 0.1, [(0.0, 0.0)] * 3)
    c = (1e16, 1e16)
    assert kc.disc_max_gap(c, 1.0, [c]) >= 1.0
    with pytest.raises(kc.KCError):
        kc.disc_max_gap((0.0, 0.0), 0.1, [])


# ---------------------------------------------------------------- Codex round 3 of the J round (Liam: "Vong 3 hep")
def test_non_finite_samples_are_refused_before_the_maxima(scene, monkeypatch):
    """KCJ1-13: a NaN sample of the tag conversion or of the virtual tag difference raises (the pair becomes broken),
    it is never hidden by max()."""
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'tag_heading_diff', lambda *a, **k: float('nan'))
    with pytest.raises(ValueError):
        kc.a3_tag_gap(cam, cam, marks[30], marks[31], 0.1445, 0.119)
    monkeypatch.undo()
    with pytest.raises(ValueError):
        kc.chuyen_pair(cam, _MapCam(cam, np.array([[float('nan'), 0.0], [0.0, 1.0]])), (0.0, 0.0), (0.5, 0.0), 0.0, 0.0,
                       0.1445, 0.119, 0.05, (0.0, 0.0))


def test_xuat_writes_the_scored_runs_for_khoa(tmp_path):
    """KCJ3-1 (writer side): moc_dich carries every scored run with its frames sha and the runs without U_moc
    sources, so khoa (C8B) can require their clock-failure evidence."""
    kcp = _kc_file(tmp_path)
    two = dict(luot={'r1': dict(nguon=dict(frames_sha256='a' * 64)), 'r2': dict(nguon=dict(frames_sha256='b' * 64))})
    assert kc.main(_xuat_args(tmp_path, kcp, tk_over=two)) == 0
    md = json.loads((tmp_path / 'o.json').read_text(encoding='utf-8'))['moc_dich']
    assert md['luot_cham'] == {'r1': 'a' * 64, 'r2': 'b' * 64} and md['luot_vang_u_moc'] == ['r2']


def test_tien_de_cli_checks_the_plan_content(tmp_path, id8):
    """KCJ1-2 (round 3): end to end on files; the plan file is parsed and hashed from the same bytes, its events must
    be those tung_khung recorded; a plan edited afterwards is refused (code 2), nothing written."""
    kcr, us, tk = _td_inputs()
    for n in kc.CLIPS_REQUIRED:
        sp, vp = tmp_path / f'{n}.json', tmp_path / f'{n}.mov'
        sp.write_text('{"s": "%s"}' % n, encoding='utf-8')
        vp.write_bytes(n.encode())
        kcr['sessions'][n] = dict(file=str(sp), sha256=kc.sha256_file(str(sp)), video=str(vp),
                                  video_sha256=kc.sha256_file(str(vp)))
    kcr.update(id_khoa=ALL8, buoi='S1',
               theo_thoi_gian={n: dict(rieng_m={str(i): 0.0002 for i in ALL8}) for n in ('H_b', 'H_m', 'H_z')})
    usp = tmp_path / 'u_so.json'
    usp.write_text(json.dumps(us), encoding='utf-8')
    kcr['u_so_sha256'] = kc.sha256_file(str(usp))
    kcp = tmp_path / 'kc.json'
    kcp.write_text(json.dumps(kcr), encoding='utf-8')
    plan = tmp_path / 'plan.txt'
    plan.write_text('H_a\nH_b\nR1\nH_m\nR2\nH_z\n', encoding='utf-8')
    tk.update(u_so_sha256=kcr['u_so_sha256'], session_sha256=kcr['sessions']['H_a']['sha256'],
              kiem_cheo_sha256=kc.sha256_file(str(kcp)))
    tk['ke_hoach'].update(file=str(plan), sha256=kc.sha256_file(str(plan)))
    for rid, v in tk['luot'].items():
        dd = {}
        for role, key in (('track', 'track_sha256'), ('frames', 'frames_sha256'), ('kiem_cheo_khung',
                                                                                  'kiem_cheo_khung_sha256')):
            f = tmp_path / f'{rid}_{role}'
            f.write_text(f'{rid} {role}', encoding='utf-8')
            dd[role] = str(f)
            v['nguon'][key] = kc.sha256_file(str(f))
        v['duong_dan'] = dd
    tkp = tmp_path / 'tk.json'
    tkp.write_text(json.dumps(tk), encoding='utf-8')
    base = ['tien_de', '--session-id', 'S1', '--ngay', '2026-11-20', '--kiem-cheo', str(kcp), '--u-so', str(usp),
            '--tung-khung', str(tkp)]
    assert kc.main(base + ['--out', str(tmp_path / 'td.json')]) == 0
    r = json.loads((tmp_path / 'td.json').read_text(encoding='utf-8'))
    assert r['ket_qua'] == 'dat' and r['luot'] == ['R1', 'R2'] and r['sha256']['ke_hoach'] == kc.sha256_file(str(plan))
    plan.write_text('H_a\nH_b\nR1\nH_m\nR2\nR3\nH_z\n', encoding='utf-8')
    assert kc.main(base + ['--out', str(tmp_path / 'td2.json')]) == 2 and not (tmp_path / 'td2.json').exists()


def test_closed_stdout_after_publication(tmp_path, monkeypatch):
    """KCJ1-6 (round 4): stdout whose write and flush fail: code 0, the file published, stdout replaced so the
    interpreter does not fail again at exit."""
    class Dead:
        def write(self, *a):
            raise BrokenPipeError('closed (test)')

        def flush(self):
            raise BrokenPipeError('closed (test)')
    dead = Dead()
    monkeypatch.setattr(sys, 'stdout', dead)
    assert kc.main(_xuat_args(tmp_path, _kc_file(tmp_path))) == 0
    assert (tmp_path / 'o.json').exists() and sys.stdout is not dead


# ---------------------------------------------------------------- F15 (a) meo_ma_ao_E, (b) meo_dau_chan_bang
class _FieldCam:
    """A camera whose floor map is the true one plus a non-linear field f(x, y) = (0.02 x^2, -0.01 x y) (z = 0)."""

    def __init__(self, cam):
        self.cam = cam

    @staticmethod
    def f(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.column_stack([0.02 * p[:, 0] ** 2, -0.01 * p[:, 0] * p[:, 1]])

    def pix2floor(self, uv, z=0.0):
        q = self.cam.pix2floor(uv, z)
        return q + self.f(q)

    def project(self, X):
        """Inverse of the floor map for the xy part (q + f(q) = X, fixed point), height kept: the per-case A3 sees
        through this camera too (CODE 20/11)."""
        X = np.asarray(X, float).reshape(-1, 3)
        q = X[:, :2].copy()
        for _ in range(60):
            q = X[:, :2] - self.f(q)
        return self.cam.project(np.column_stack([q, X[:, 2]]))


F15_C, F15_KAP = np.array([-0.4802, -0.0329]), 0.119 / ZC


def test_map_residual_independent_oracle(scene):
    """KCF1-9: held floor = p + f(p) -> lifted residual = (1 - kappa) f(p), p = (P - kappa C) / (1 - kappa), with C != 0
    (closed form, not the code's unlift); held = full -> 0."""
    marks, cam, img = scene
    P = np.array([[0.4, 0.1], [-0.3, 0.5], [1.2, -0.6]])
    assert np.allclose(kc.map_residual(cam, cam, P, F15_KAP, F15_C), 0.0, atol=1e-12)
    p = (P - F15_KAP * F15_C) / (1 - F15_KAP)
    want = (1 - F15_KAP) * _FieldCam.f(p)
    assert np.allclose(kc.map_residual(cam, _FieldCam(cam), P, F15_KAP, F15_C), want, atol=1e-9)


def test_h32_ids_floor_footprint_and_three_nearest():
    """KCF1-4: marks within R of ANY footprint point; padded to 3 nearest when fewer."""
    mk = {1: (0.1, 0.0, SIDE), 2: (0.2, 0.0, SIDE), 3: (0.61, 0.0, SIDE), 4: (5.0, 5.0, SIDE)}
    assert kc.h32_ids(mk, [[0.0, 0.0]], 0.60) == [1, 2, 3]
    assert kc.h32_ids(mk, [[0.0, 0.0], [1.15, 0.0]], 0.60) == [1, 2, 3]
    assert kc.h32_ids(mk, [[4.6, 4.6]], 0.60) == [2, 3, 4]          # only mark 4 within R: padded by 3, then 2


def test_held_cases_all_or_nothing(scene):
    """KCF1-5: a planned hold-out case that fails stops the command (no partial set)."""
    marks, cam, img = scene

    class PC:
        def __call__(self, out=()):
            if len(out) > 2:
                raise ValueError('fit failed (test)')
            return cam
    with pytest.raises(kc.KCError, match='KCF1-5'):
        kc.held_cases(PC(), [1, 2, 3])
    assert [n for n, _, _ in kc.held_cases(lambda out=(): cam, [1, 2, 3])] == ['bo:1-2-3', 'bo:1-2', 'bo:1-3',
                                                                               'bo:2-3']      # KCF1-3: by id set


def _f15_session(tmp_path, scene):
    import c7_measure as cmeas
    marks, cam, img = scene
    s = dict(kind='c7_session', model_khoa='B1', chosen='B1', width=W, height=H, zc_tape_m=ZC, car_side_m=0.1445,
             tag_h_m=0.119, kappa=F15_KAP, plumb_xy_m=list(F15_C), camera=cmeas.cam_to_dict(cam),
             fit_ids=sorted(img), marks={str(i): list(m) for i, m in marks.items()},
             ref_corners_px={str(i): c.tolist() for i, c in img.items()})
    sp = tmp_path / 'ses.json'
    sp.write_text(json.dumps(s), encoding='utf-8')
    return sp


def _field_cases(monkeypatch, scene):
    marks, cam, img = scene
    monkeypatch.setattr(kc, 'held_cases', lambda pc, ids: [('bo_vung', list(ids), _FieldCam(cam))])


def _e_files(tmp_path, sp, offset=(0.0, 0.0, 0.2), track_over=None, rows_extra=()):
    td = tmp_path / 'E'
    td.mkdir(exist_ok=True)
    rows = ['idx,t_s,valid,psi,tag_x,tag_y']
    for k in range(12):
        rows.append(f"{k},{k / 60!r},{0 if k == 5 else 1},{0.3 * k!r},{0.05 + 0.01 * k!r},{-0.02!r}")
    (td / 'frames.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    fsha = kc.sha256_file(str(td / 'frames.csv'))
    tr = dict(video_sha256='a' * 64, session_sha256=kc.sha256_file(str(sp)), frames_sha256=fsha, tag_offset=list(offset))
    tr.update(track_over or {})
    (td / 'track.json').write_text(json.dumps(tr), encoding='utf-8')

    def w(loai, ids):
        return json.dumps(dict(schema='cua_so_v1', cho='goc', loai=loai, luot_id='E1', clip_sha256='a' * 64,
                               frames_sha256=fsha,
                               session_sha256=kc.sha256_file(str(sp)), khung=[dict(id=i, pts_s=i / 60) for i in ids]))
    cs = tmp_path / 'cua_so_v1.jsonl'
    cs.write_text('\n'.join([w('E_dau', [1, 2]), w('E_cuoi', [8, 9]), w('pre', [10, 11])] + [w(*x) for x in rows_extra])
                  + '\n', encoding='utf-8')
    return td, cs


def test_meo_E_corners_order_and_lift(tmp_path, scene, monkeypatch):
    """F15 (a), KCF1-8, KCF1-9: E_doan = valid frames strictly between E_dau and E_cuoi; each corner (ArUco order of
    the body square turned by psi + dpsi, tag_offset != 0) carries (1 - kappa) f(p) of an independent oracle; output
    is chan_doan with every source path + sha."""
    sp = _f15_session(tmp_path, scene)
    td, cs = _e_files(tmp_path, sp)
    _field_cases(monkeypatch, scene)
    out = tmp_path / 'e.json'
    assert kc.main(['meo_E', '--model', 'B1', '--cho', 'goc', '--session', str(sp), '--track', str(td), '--cua-so',
                    str(cs), '--out', str(out)]) == 0
    r = json.loads(out.read_text(encoding='utf-8'))
    assert [k['frame_id'] for k in r['khung']] == [3, 4, 6, 7] and r['trang_thai'] == 'chan_doan'
    assert {x['vai_tro'] for x in r['nguon']} == {'track', 'frames', 'cua_so_v1', 'session'}
    h = 0.1445 / 2
    body = np.array([[h, h], [h, -h], [-h, -h], [-h, h]])
    for k in r['khung']:
        fid = k['frame_id']
        a = 0.3 * fid + 0.2
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        P = body @ R.T + np.array([0.05 + 0.01 * fid, -0.02])
        assert np.allclose(k['goc_m'], P, atol=1e-12)
        p = (P - F15_KAP * F15_C) / (1 - F15_KAP)
        assert np.allclose(k['du_m']['bo_vung'], (1 - F15_KAP) * _FieldCam.f(p), atol=1e-9)
    assert r['h_bar']['da_kiem'] is False
    a3 = r['ban_do']['ca'][0]['A3']                                      # CODE 20/11: A3 per H32 case
    assert a3['cam_full'] == 'session.camera' and a3['L_max_m'] == kc.L_MAX_M and a3['n_cap'] == len(a3['cap'])


@pytest.mark.parametrize('bad', ['K2', 'no_track_sha', 'dup_dau', 'session', 'a6_other_h', 'a6_nan', 'a6_schema',
                                 'a6_plumb', 'offset_short', 'other_run', 'no_run_id', 'frame_not_in_track',
                                 'frame_twice'])
def test_meo_E_refuses(tmp_path, scene, monkeypatch, bad):
    """KCF1-6, KCF1-7, KCF1-8: wrong place, missing link sha, two E_dau, another session, h-bar of the plate not the
    session tag height: code 2, nothing written."""
    sp = _f15_session(tmp_path, scene)
    over = {'no_track_sha': dict(video_sha256=None), 'session': dict(session_sha256='1' * 64),
            'offset_short': dict(tag_offset=[0])}.get(bad)
    td, cs = _e_files(tmp_path, sp, track_over=over, rows_extra=[('E_dau', [0])] if bad == 'dup_dau' else ())
    if bad in ('other_run', 'frame_not_in_track', 'no_run_id', 'frame_twice'):
        rows = [json.loads(x) for x in cs.read_text(encoding='utf-8').splitlines()]
        for r_ in rows:
            if r_['loai'] == 'E_cuoi':
                if bad == 'other_run':
                    r_['luot_id'] = 'E2'
                elif bad == 'frame_not_in_track':
                    r_['khung'].append(dict(id=-99, pts_s=900.0))
                elif bad == 'frame_twice':
                    r_['khung'].append(dict(r_['khung'][-1]))
            if bad == 'no_run_id' and r_['loai'] in ('E_dau', 'E_cuoi'):
                r_.pop('luot_id')
        cs.write_text('\n'.join(json.dumps(r_) for r_ in rows) + '\n', encoding='utf-8')
    _field_cases(monkeypatch, scene)
    args = ['meo_E', '--model', 'B1', '--cho', 'K2' if bad == 'K2' else 'goc', '--session', str(sp), '--track', str(td),
            '--cua-so', str(cs), '--out', str(tmp_path / 'e.json')]
    if bad.startswith('a6'):
        a6 = tmp_path / 'a6.json'
        h_ = dict(a6_other_h=0.124175, a6_nan=float('nan'), a6_schema=0.119, a6_plumb=0.119)[bad]
        a6.write_text(json.dumps(dict(schema='other' if bad == 'a6_schema' else 'c8b_a6_v2', h_m=h_, H_m=ZC,
                                      plumb_xy_m=[0.0, 0.0] if bad == 'a6_plumb' else list(F15_C))), encoding='utf-8')
        args += ['--a6', str(a6)]
    assert kc.main(args) == 2 and not (tmp_path / 'e.json').exists()


def _bang_files(tmp_path, sp, ten=('A', 'B', 'oc1'), with_session_file=False):
    import hashlib as _h
    ses = kc.sha256_file(str(sp))
    end = lambda d, cho='goc': dict(cho=cho, clip_sha256='c' * 64, session_sha256=ses, khung_neo=dict(id=[3, 1, 2]),  # noqa
                                    khung_neo_sha256=_h.sha256(b'1,2,3').hexdigest(),
                                    Ph_neo_m=[[0.10 + d, 0.05], [0.25 + d, 0.05], [0.18 + d, 0.12]][:len(ten)])
    f6 = dict(session=dict(file=str(sp), sha256=kc.sha256_file(str(sp))), neo_ten=list(ten),
              cua_so=[dict(meta=dict(cho='goc', loai='dung_12b', luot='t12', clip_sha='d' * 64),
                           Ph=[[0.3, 0.2], [0.45, 0.2], [0.38, 0.27]][:len(ten)])])
    f6p = tmp_path / 'hieu_chuan_neo.json'
    f6p.write_text(json.dumps(f6), encoding='utf-8')
    ll = dict(schema='lan_lap_v1', session_sha256=kc.sha256_file(str(sp)), neo_ten=list(ten), h_neo_m=[0.12] * len(ten),
              h_n_m=0.12, hieu_chuan=dict(file=str(f6p), sha256=kc.sha256_file(str(f6p))),
              luot=[dict(luot_id='r1', cho='goc', dau=dict(chuan=end(0.0), pre=end(0.01), post=end(0.02),
                                                           dich=end(0.01))),
                    dict(luot_id='r2', cho='K1', dau=dict(chuan=end(1.0, 'K1'), pre=end(1.01, 'K1')))])
    if with_session_file:
        ll['session_file'] = str(sp)
    lp = tmp_path / 'lan_lap.json'
    lp.write_text(json.dumps(ll), encoding='utf-8')
    return lp, ll


def test_meo_bang_f8_of_f9c4b3d_and_oracle(tmp_path, scene, monkeypatch):
    """F15 (b), KCF1-2, KCF1-3, KCF1-9: F8 without session_file (writer f9c4b3d): the session comes from F6; every
    anchor of every run end and of every calibration window, (1 - kappa_n) f(p) per anchor; A, B, oc1 -> thieu_oc
    false; lan_lap.hieu_chuan_sha256 kept for the NHACTAM reader."""
    sp = _f15_session(tmp_path, scene)
    lp, ll = _bang_files(tmp_path, sp)
    _field_cases(monkeypatch, scene)
    out = tmp_path / 'b.json'
    assert kc.main(['meo_bang', '--model', 'B1', '--lan-lap', str(lp), '--out', str(out)]) == 0
    r = json.loads(out.read_text(encoding='utf-8'))
    kn = 0.12 / ZC
    t = r['theo_cho']['goc']
    assert r['thieu_oc'] is False and r['trang_thai'] == 'chan_doan' and r['kappa_n'] == pytest.approx(kn)
    assert r['schema'] == 'c7_meo_dau_chan_bang_v1_chan_doan'          # KCF1-1: readers of the full schema refuse it
    assert r['theo_cho']['goc']['ca'] == r['theo_cho']['K1']['ca']   # KCF1-3: one case list for every place
    assert r['lan_lap']['hieu_chuan_sha256'] == ll['hieu_chuan']['sha256']
    assert [m['dau'] for m in t['muc']] == ['chuan', 'post', 'pre'] and len(t['muc_hieu_chuan']) == 1
    for m in t['muc']:
        P = np.asarray(ll['luot'][0]['dau'][m['dau']]['Ph_neo_m'])
        p = (P - kn * F15_C) / (1 - kn)
        want = (1 - kn) * _FieldCam.f(p)
        for k, nm in enumerate(['A', 'B', 'oc1']):
            assert np.allclose(m['neo'][nm]['bo_vung'], want[k], atol=1e-9)


@pytest.mark.parametrize('bad', ['name_X', 'count', 'f6_sha', 'schema', 'no_khung_sha', 'khung_sha_other', 'end_session',
                                 'float_id', 'no_clip_sha', 'f6_session'])
def test_meo_bang_refuses(tmp_path, scene, monkeypatch, bad):
    """KCF1-7, KCF1-9: an anchor name outside A, B, oc<n>, an anchor count not neo_ten's, an F6 of another sha: code 2."""
    sp = _f15_session(tmp_path, scene)
    lp, ll = _bang_files(tmp_path, sp, ten=('A', 'B', 'X') if bad == 'name_X' else ('A', 'B', 'oc1'),
                         with_session_file=True)
    if bad == 'count':
        ll['luot'][0]['dau']['pre']['Ph_neo_m'] = [[0.1, 0.05]]
    if bad == 'f6_sha':
        ll['hieu_chuan']['sha256'] = 'e' * 64
    if bad == 'schema':
        ll['schema'] = 'lan_lap_wrong'
    if bad == 'no_khung_sha':
        ll['luot'][0]['dau']['pre'].pop('khung_neo_sha256')
    if bad == 'khung_sha_other':
        ll['luot'][0]['dau']['pre']['khung_neo'] = dict(id=[1, 2])
    if bad == 'end_session':
        ll['luot'][0]['dau']['post'].pop('session_sha256')
    if bad == 'float_id':
        ll['luot'][0]['dau']['pre']['khung_neo'] = dict(id=[1.9, 2, 3])
    if bad == 'no_clip_sha':
        ll['luot'][0]['dau']['pre'].pop('clip_sha256')
    if bad == 'f6_session':
        f6p = tmp_path / 'hieu_chuan_neo.json'
        f6 = json.loads(f6p.read_text(encoding='utf-8'))
        f6['session']['sha256'] = '9' * 64
        f6p.write_text(json.dumps(f6), encoding='utf-8')
        ll['hieu_chuan']['sha256'] = kc.sha256_file(str(f6p))
    lp.write_text(json.dumps(ll), encoding='utf-8')
    _field_cases(monkeypatch, scene)
    assert kc.main(['meo_bang', '--model', 'B1', '--lan-lap', str(lp), '--out', str(tmp_path / 'b.json')]) == 2
    assert not (tmp_path / 'b.json').exists()


def test_meo_bang_accepts_oc_names_of_the_contract(tmp_path, scene, monkeypatch):
    """KCF2-1: F15 allows oc* (oc_trai of c7_nhac_tam parse_neo), not only oc<n>."""
    sp = _f15_session(tmp_path, scene)
    lp, ll = _bang_files(tmp_path, sp, ten=('A', 'B', 'oc_trai'), with_session_file=True)
    _field_cases(monkeypatch, scene)
    assert kc.main(['meo_bang', '--model', 'B1', '--lan-lap', str(lp), '--out', str(tmp_path / 'b.json')]) == 0


def test_meo_E_schema_is_diagnostic(tmp_path, scene, monkeypatch):
    """KCF1-1: the E output carries the diagnostic schema name."""
    sp = _f15_session(tmp_path, scene)
    td, cs = _e_files(tmp_path, sp)
    _field_cases(monkeypatch, scene)
    out = tmp_path / 'e.json'
    assert kc.main(['meo_E', '--model', 'B1', '--cho', 'goc', '--session', str(sp), '--track', str(td), '--cua-so',
                    str(cs), '--out', str(out)]) == 0
    assert json.loads(out.read_text(encoding='utf-8'))['schema'] == 'c7_meo_ma_ao_E_v1_chan_doan'


def test_meo_E_with_the_plate_file(tmp_path, scene, monkeypatch):
    """KCF1-6: a c8b_a6_v2 with the session tag height, plumb point and Z_c: h_bar checked."""
    sp = _f15_session(tmp_path, scene)
    td, cs = _e_files(tmp_path, sp)
    _field_cases(monkeypatch, scene)
    a6 = tmp_path / 'a6.json'
    a6.write_text(json.dumps(dict(schema='c8b_a6_v2', h_m=0.119, H_m=0.119 / F15_KAP, plumb_xy_m=list(F15_C))),
                  encoding='utf-8')
    out = tmp_path / 'e.json'
    assert kc.main(['meo_E', '--model', 'B1', '--cho', 'goc', '--session', str(sp), '--track', str(td), '--cua-so',
                    str(cs), '--a6', str(a6), '--out', str(out)]) == 0
    assert json.loads(out.read_text(encoding='utf-8'))['h_bar']['da_kiem'] is True


def test_meo_bang_accepts_plain_oc(tmp_path, scene, monkeypatch):
    """KCF2-1 (round 3): 'oc' alone matches oc*."""
    sp = _f15_session(tmp_path, scene)
    lp, ll = _bang_files(tmp_path, sp, ten=('A', 'B', 'oc'), with_session_file=True)
    _field_cases(monkeypatch, scene)
    assert kc.main(['meo_bang', '--model', 'B1', '--lan-lap', str(lp), '--out', str(tmp_path / 'b.json')]) == 0


# ---------------------------------------------------------------- F15 (a): A3 per H32 case (CODE 20/11)
def test_a3_held_case_sees_a_moved_mark_through_the_case_camera(scene):
    """Mark 30 truly 2 mm off its tape across the pair 30-32; case bo:30-31-32 (camera fitted without all three):
    only the pairs of the set with L <= L_max, dtheta of 30-32 = -atan(2 mm / L) (independent of the code), quy =
    |dtheta| + gap, quy_max the largest; the pair case bo:30-32 uses the same camera as a3_clip for that pair."""
    marks, cam, _ = scene
    m, j = 30, 32
    vt = np.array(marks[j][:2]) - np.array(marks[m][:2])
    n = np.array([-vt[1], vt[0]]) / np.linalg.norm(vt)
    true = dict(marks)
    true[m] = (marks[m][0] + 0.002 * n[0], marks[m][1] + 0.002 * n[1], SIDE)
    img = image_of(cam, true)
    pc = kc.PairCams(img, marks, 'B1', W, H, ZC)
    tag = (0.1445, 0.119)
    out = kc.a3_held_case('bo:30-31-32', img, marks, cam, pc((30, 31, 32)), (32, 30, 31), tag, l_max=0.8)
    rows = {(r['i'], r['j']): r for r in out['cap']}
    assert set(rows) == {(30, 32), (31, 32)} and out['n_cap'] == 2          # 30-31 is 1.0 m > L_max
    L = float(np.linalg.norm(vt))
    assert rows[(m, j)]['dtheta_deg'] == pytest.approx(-math.degrees(math.atan2(0.002, L)), abs=2e-3)
    for r in rows.values():
        assert r['quy_deg'] == pytest.approx(abs(r['dtheta_deg']) + r['gap_deg']) and r['gap_deg'] >= 0
    assert out['quy_max_deg'] == max(r['quy_deg'] for r in rows.values()) and out['cam_full'] == 'session.camera'
    pair = kc.a3_held_case('bo:30-32', img, marks, cam, pc((30, 32)), (30, 32), tag, l_max=0.8)['cap'][0]
    ref = {(r['i'], r['j']): r for r in kc.a3_clip(img, marks, 'B1', W, H, ZC, l_max=0.8, pc=pc)}[(m, j)]
    assert pair['dtheta_deg'] == ref['dtheta_deg']                         # same name, same camera (KCF1-3)


def test_a3_held_case_without_pair_and_failing(scene):
    """No pair of the set within L_max: quy_max_deg null with the reason; a failing conversion fails the case
    (KCF1-5)."""
    marks, cam, img = scene
    out = kc.a3_held_case('bo:30-31', img, marks, cam, cam, (30, 31), (0.1445, 0.119), l_max=0.8)
    assert out['cap'] == [] and out['quy_max_deg'] is None and 'L_max' in out['ly_do']

    class Bad:
        def pix2floor(self, uv, z=0.0):
            return np.full((len(np.atleast_2d(uv)), 2), np.nan)

        def project(self, X):
            return cam.project(X)
    with pytest.raises(kc.KCError, match='KCF1-5'):
        kc.a3_held_case('bo:30-32', img, marks, cam, Bad(), (30, 32), (0.1445, 0.119), l_max=0.8)


def test_meo_E_a3_per_case_on_real_fits(tmp_path, scene):
    """Production path, no stub: every H32 case carries A3 over exactly the pairs of its left-out set with L <=
    L_max; exact scene (session camera = true camera) -> every quy_deg near 0."""
    sp = _f15_session(tmp_path, scene)
    td, cs = _e_files(tmp_path, sp)
    out = tmp_path / 'e.json'
    assert kc.main(['meo_E', '--model', 'B1', '--cho', 'goc', '--session', str(sp), '--track', str(td), '--cua-so',
                    str(cs), '--r-xp', '0.3', '--out', str(out)]) == 0
    r = json.loads(out.read_text(encoding='utf-8'))
    marks = scene[0]
    assert len(r['ban_do']['ca']) == 1 + math.comb(len(r['ban_do']['H32_ids']), 2)
    for ca in r['ban_do']['ca']:
        ids_ = ca['bo_ids']
        want = {(a, b) for k, a in enumerate(ids_) for b in ids_[k + 1:]
                if math.dist(marks[a][:2], marks[b][:2]) <= kc.L_MAX_M}
        assert {(p['i'], p['j']) for p in ca['A3']['cap']} == want and ca['A3']['cam_ca'] == ca['ten']
        assert all(p['quy_deg'] < 2e-3 for p in ca['A3']['cap'])


def test_meo_E_a3_failure_is_code_2(tmp_path, scene, monkeypatch):
    """A non-finite A3 sample of a case: code 2, nothing written (KCF1-5)."""
    sp = _f15_session(tmp_path, scene)
    td, cs = _e_files(tmp_path, sp)
    _field_cases(monkeypatch, scene)

    def nan_gap(*a, **k):
        raise ValueError('non-finite tag / pair angle sample')
    monkeypatch.setattr(kc, 'a3_tag_gap', nan_gap)
    assert kc.main(['meo_E', '--model', 'B1', '--cho', 'goc', '--session', str(sp), '--track', str(td), '--cua-so',
                    str(cs), '--out', str(tmp_path / 'e.json')]) == 2 and not (tmp_path / 'e.json').exists()
    monkeypatch.undo()                                                  # real conversion again: only L_max is bad
    _field_cases(monkeypatch, scene)
    for bad in ('nan', '0', '-1'):
        assert kc.main(['meo_E', '--model', 'B1', '--cho', 'goc', '--session', str(sp), '--track', str(td),
                        '--cua-so', str(cs), '--L-max', bad, '--out', str(tmp_path / 'e.json')]) == 2
    assert not (tmp_path / 'e.json').exists()
    assert kc.main(['meo_E', '--model', 'B1', '--cho', 'goc', '--session', str(sp), '--track', str(td), '--cua-so',
                    str(cs), '--out', str(tmp_path / 'e.json')]) == 0            # the same inputs pass otherwise
