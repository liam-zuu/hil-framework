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
    assert set(p1) == set(img) and max(np.linalg.norm(v) for v in p1.values()) < 1e-5
    u, info = kc.u_so_v2(img, img, [img] * 40, marks, groups, cams)
    assert u < 1e-6 and info['tinh_m'] < 1e-6 and info['cua_so_m'] < 1e-6


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


def test_u_so_sees_a_shift_in_clip_b(scene):
    """A mark moved 1.5 mm (metres, projected) in H_b: the static term of u_so sees it on that mark."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    cams = kc.group_cams(img, marks, groups, 'B1', W, H, ZC)
    true = dict(marks)
    true[5] = (marks[5][0] + 0.0015, marks[5][1], SIDE)
    img_b = image_of(cam, true)
    u, info = kc.u_so_v2(img, img_b, [], marks, groups, cams)
    assert float(info['d_tinh_m']['5']) == pytest.approx(0.0015, abs=2e-5)
    assert u == pytest.approx(0.0015, abs=2e-5)


def _taped(marks, seed=3, sd=0.0015):
    """Tape coordinates with errors (the image stays true): fits on different mark sets disagree."""
    rng = np.random.default_rng(seed)
    return {i: (x + rng.normal(0, sd), y + rng.normal(0, sd), s) for i, (x, y, s) in marks.items()}


def test_u_so_counts_the_run_mark_sets(scene):
    """KC1-2: with tape errors, a smaller fit set (marks hidden by the car) moves the prediction although nothing
    moved; u_so with the run sets is larger than with the full set only, and equal clips still give 0."""
    marks0, cam, img = scene
    marks = _taped(marks0)
    groups = kc.make_groups(sorted(img))
    cams = kc.group_cams(img, marks, groups, 'B1', W, H, ZC)
    u_full, _ = kc.u_so_v2(img, img, [], marks, groups, cams)
    assert u_full < 1e-6
    hidden = [{1, 2, 3, 8, 9, 10, 15, 16, 17, 30, 31}, {20, 21, 22, 27, 28, 29}]
    tap = [sorted(set(img) - h) for h in hidden]
    u_set, info = kc.u_so_v2(img, img, [], marks, groups, cams, tap_moc=tap)
    assert info['co_tap_luot'] and u_set > 2e-4


def test_u_so_window_term(scene):
    """The window term is the largest |window median - clip mean| of the frames of H_b."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    cams = kc.group_cams(img, marks, groups, 'B1', W, H, ZC)
    frames = [img] * 40
    true = dict(marks)
    true[12] = (marks[12][0], marks[12][1] + 0.001, SIDE)
    frames = frames[:5] + [image_of(cam, true)] * 30 + frames[35:]   # a full window with mark 12 moved 1 mm
    u, info = kc.u_so_v2(img, img, frames, marks, groups, cams, win=30)
    assert info['cua_so_m'] == pytest.approx(0.001, abs=3e-5)


def test_u_so_flags_sets_it_cannot_compute(scene):
    """KC2-2: every mark in half the H_b frames but never all at once: the run set gets no window, flagged, not
    replaced by a smaller set; kiem_h shows 0 % at once."""
    marks, cam, img = scene
    groups = kc.make_groups(sorted(img))
    cams = kc.group_cams(img, marks, groups, 'B1', W, H, ZC)
    half_a = {i: c for i, c in img.items() if i % 2}
    half_b = {i: c for i, c in img.items() if not i % 2}
    frames = [half_a, half_b] * 20
    u, info = kc.u_so_v2(img, img, frames, marks, groups, cams, win=10)
    assert info['thieu'] and any(x['vi'] == 'cua_so' for x in info['khong_tinh_duoc'])
    r = kc.kiem_h(frames, sorted(img))
    assert r['ti_le_cung_luc'] == 0.0 and min(r['ti_le_moc'].values()) == 0.5


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


def _own():
    return dict(P1=dict(bo_nhom_max_m=0.0034, tung_khung_max_m=0.003), moc_dich=dict(nguong_m=0.001, u_so_m=0.0003),
                A3=dict(max_deg=0.15, rms_deg=0.07), P7=dict(lech_max_m=0.0006), A5=dict(lech_max_deg=0.4),
                _src=('kc.json', 'a' * 64))


def _parts():
    blocks = dict(P3=dict(vi_tri_max_m=0.001), A2=dict(goc_max_deg=0.3), than_dich=dict(ghi='thong tin'),
                  day_doi=dict(dn_m=0.01, dzc_m=0.02, kappa=0.12 / 2.632), do_cao=dict(rho_max_m=1.6, H_m=2.632, h_m=0.12),
                  bo_cham=dict(tag_offset=None, side_m=0.1445))
    return {n: (b, f'{n}.json', 'b' * 64) for n, b in blocks.items()}


def test_build_cam_u_ok():
    out = kc.build_cam_u(_own(), _parts())
    assert out['version'] == 'c8_cam_u_vao_v2'
    assert set(out) - {'version', 'nguon'} == set(kc.CAM_BLOCKS)
    assert {n['khoi'] for n in out['nguon']} == set(kc.CAM_BLOCKS)
    assert all('nguon' in out[b] for b in kc.CAM_BLOCKS)
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
    """The file written by build_cam_u is read by c8b_u.u_p_u_psi (CODE_C8B, hil feat/c8b-u 67c229a); runs on the
    integration branch where both files are, skipped before."""
    CU = pytest.importorskip('c8b_u')
    if getattr(CU, 'CAM_VERSION', None) != kc.CAM_U_VERSION:
        pytest.skip('c8b_u of another camera-file version')
    cam = json.loads(json.dumps(kc.build_cam_u(_own(), _parts())))
    tk = {"version": "c8b_thuoc_kep_v1", "cao_4_goc_mm": [120.2, 120.0, 119.8, 120.1], "e_mm": 0.1,
          "goc_dan": {"a_mm": 10.3, "b_mm": 10.1, "L_mm": 96.4, "canh": "trai"}, "tam_than_deg": 0.05}
    xo = {"version": "c8b_xoay_v1", "q_tag_m": [0.003, -0.004], "du_m": 0.0007}
    tb = dict(ga_ok=True, d_p_quay_m=0.0007, d_psi_quay_deg=0.25)
    up, p, upsi, s = CU.u_p_u_psi(cam, tb, tk, xo)
    assert p['P1_anh_san'] == pytest.approx(0.0034 + 0.001 + 0.0003)
    assert s['A3_huong_vung'] == pytest.approx(0.15) and s['A5_nhieu_khung'] == pytest.approx(0.4)
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
                 **{b: dict(blk, buoi='C8_x') for b, blk in _own().items() if b != '_src'})
    kcres.update(u_so_sha256='5' * 64, nguong_track_m=0.0013, moc=[dict(file='moc.csv', sha256='6' * 64)])
    kcres.update(over)
    p = tmp_path / 'kiem_cheo.json'
    p.write_text(json.dumps(kcres), encoding='utf-8')
    return p


def _xuat_args(tmp_path, kcp, tk_over=None, part_buoi='C8_x', buoi='C8_x'):
    tk = dict(version=kc.TUNG_KHUNG_VERSION, buoi='C8_x', model='B1', nhom=[[1, 5], [2, 6]], cham_lai=True,
              tung_khung_max_m=0.003, kiem_cheo_sha256=kc.sha256_file(str(kcp)), u_so_sha256='5' * 64,
              nguong_track_m=0.0013, moc=[dict(file='moc.csv', sha256='6' * 64)])
    tk.update(tk_over or {})
    tkp = tmp_path / 'tk.json'
    tkp.write_text(json.dumps(tk), encoding='utf-8')
    args = ['xuat', '--buoi', buoi, '--kiem-cheo', str(kcp), '--tung-khung', str(tkp), '--out', str(tmp_path / 'o.json')]
    for n, (blk, _, _) in _parts().items():
        q = tmp_path / f'{n}.json'
        q.write_text(json.dumps(dict(blk, buoi=part_buoi)), encoding='utf-8')
        args += ['--phan', f'{n}={q}']
    return args


def test_xuat_ok(tmp_path):
    assert kc.main(_xuat_args(tmp_path, _kc_file(tmp_path))) == 0
    out = json.loads((tmp_path / 'o.json').read_text(encoding='utf-8'))
    assert out['P1']['tung_khung_max_m'] == 0.003 and out['version'] == 'c8_cam_u_vao_v2'


@pytest.mark.parametrize('bad', ['thu', 'no_Hm', 'tk_buoi', 'tk_nhom', 'tk_model', 'tk_sha', 'tk_not_rescored',
                                 'part_buoi', 'arg_buoi', 'tk_u_so', 'tk_nguong', 'tk_moc', 'u_so_thieu'])
def test_xuat_refuses_mixed_inputs(tmp_path, bad, capsys):
    """KC1-10: results of another session, grouping, model, kiem_cheo file, or before the rescoring are refused."""
    over = dict(thu=True) if bad == 'thu' else {}
    if bad == 'u_so_thieu':
        over['moc_dich'] = dict(_own()['moc_dich'], buoi='C8_x', thieu=True)
    if bad == 'no_Hm':
        over['sessions'] = {n: dict(file=n, sha256='0' * 64) for n in ('H_a', 'H_b', 'H_z')}
    kcp = _kc_file(tmp_path, **over)
    tk_over = {'tk_buoi': dict(buoi='C8_y'), 'tk_nhom': dict(nhom=[[1], [2]]), 'tk_model': dict(model='B2'),
               'tk_sha': dict(kiem_cheo_sha256='f' * 64), 'tk_not_rescored': dict(cham_lai=False),
               'tk_u_so': dict(u_so_sha256='7' * 64), 'tk_nguong': dict(nguong_track_m=0.0011),
               'tk_moc': dict(moc=[dict(file='moc.csv', sha256='8' * 64)])}.get(bad)
    args = _xuat_args(tmp_path, kcp, tk_over, part_buoi='C8_y' if bad == 'part_buoi' else 'C8_x',
                      buoi='C8_y' if bad == 'arg_buoi' else 'C8_x')
    assert kc.main(args) == 2
    assert not (tmp_path / 'o.json').exists()
